import Mathlib

namespace no_real_solution_3x2_plus_9x_le_neg12_l143_143573

/-- There are no real values of x such that 3x^2 + 9x ≤ -12. -/
theorem no_real_solution_3x2_plus_9x_le_neg12 (x : ℝ) : ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_real_solution_3x2_plus_9x_le_neg12_l143_143573


namespace probability_Xavier_Yvonne_not_Zelda_l143_143260

-- Define the probabilities of success for Xavier, Yvonne, and Zelda
def pXavier := 1 / 5
def pYvonne := 1 / 2
def pZelda := 5 / 8

-- Define the probability that Zelda does not solve the problem
def pNotZelda := 1 - pZelda

-- The desired probability that we want to prove equals 3/80
def desiredProbability := (pXavier * pYvonne * pNotZelda) = (3 / 80)

-- The statement of the problem in Lean
theorem probability_Xavier_Yvonne_not_Zelda :
  desiredProbability := by
  sorry

end probability_Xavier_Yvonne_not_Zelda_l143_143260


namespace intersection_S_T_l143_143720

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l143_143720


namespace history_books_count_l143_143874

-- Definitions based on conditions
def total_books : Nat := 100
def geography_books : Nat := 25
def math_books : Nat := 43

-- Problem statement: proving the number of history books
theorem history_books_count : total_books - geography_books - math_books = 32 := by
  sorry

end history_books_count_l143_143874


namespace non_basalt_rocks_total_eq_l143_143948

def total_rocks_in_box_A : ℕ := 57
def basalt_rocks_in_box_A : ℕ := 25

def total_rocks_in_box_B : ℕ := 49
def basalt_rocks_in_box_B : ℕ := 19

def non_basalt_rocks_in_box_A : ℕ := total_rocks_in_box_A - basalt_rocks_in_box_A
def non_basalt_rocks_in_box_B : ℕ := total_rocks_in_box_B - basalt_rocks_in_box_B

def total_non_basalt_rocks : ℕ := non_basalt_rocks_in_box_A + non_basalt_rocks_in_box_B

theorem non_basalt_rocks_total_eq : total_non_basalt_rocks = 62 := by
  -- proof goes here
  sorry

end non_basalt_rocks_total_eq_l143_143948


namespace shortest_altitude_triangle_l143_143129

/-- Given a triangle with sides 18, 24, and 30, prove that its shortest altitude is 18. -/
theorem shortest_altitude_triangle (a b c : ℝ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 30) 
  (h_right : a ^ 2 + b ^ 2 = c ^ 2) : 
  exists h : ℝ, h = 18 :=
by
  sorry

end shortest_altitude_triangle_l143_143129


namespace sqrt_9_eq_3_or_neg3_l143_143471

theorem sqrt_9_eq_3_or_neg3 :
  { x : ℝ | x^2 = 9 } = {3, -3} :=
sorry

end sqrt_9_eq_3_or_neg3_l143_143471


namespace minimize_maximum_absolute_value_expression_l143_143941

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l143_143941


namespace area_of_EFCD_l143_143093

theorem area_of_EFCD (AB CD h : ℝ) (H_AB : AB = 10) (H_CD : CD = 30) (H_h : h = 15) :
  let EF := (AB + CD) / 2
  let h_EFCD := h / 2
  let area_EFCD := (1 / 2) * (CD + EF) * h_EFCD
  area_EFCD = 187.5 :=
by
  intros EF h_EFCD area_EFCD
  sorry

end area_of_EFCD_l143_143093


namespace tree_height_at_2_years_l143_143319

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l143_143319


namespace nth_term_pattern_l143_143651

theorem nth_term_pattern (a : ℕ → ℕ) (h : ∀ n, a n = n * (n - 1)) : 
  (a 0 = 0) ∧ (a 1 = 2) ∧ (a 2 = 6) ∧ (a 3 = 12) ∧ (a 4 = 20) ∧ 
  (a 5 = 30) ∧ (a 6 = 42) ∧ (a 7 = 56) ∧ (a 8 = 72) ∧ (a 9 = 90) := sorry

end nth_term_pattern_l143_143651


namespace simplify_polynomial_l143_143969

noncomputable def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 4 * r - 3
noncomputable def g (r : ℝ) : ℝ := r^3 + r^2 + 6 * r - 8

theorem simplify_polynomial (r : ℝ) : f r - g r = r^3 - 2 * r + 5 := by
  sorry

end simplify_polynomial_l143_143969


namespace football_team_total_players_l143_143918

/-- The conditions are:
1. There are some players on a football team.
2. 46 are throwers.
3. All throwers are right-handed.
4. One third of the rest of the team are left-handed.
5. There are 62 right-handed players in total.
And we need to prove that the total number of players on the football team is 70. 
--/

theorem football_team_total_players (P : ℕ) 
  (h_throwers : P >= 46) 
  (h_total_right_handed : 62 = 46 + 2 * (P - 46) / 3)
  (h_remainder_left_handed : 1 * (P - 46) / 3 = (P - 46) / 3) :
  P = 70 :=
by
  sorry

end football_team_total_players_l143_143918


namespace woman_weaves_ten_day_units_l143_143792

theorem woman_weaves_ten_day_units 
  (a₁ d : ℕ)
  (h₁ : 4 * a₁ + 6 * d = 24)
  (h₂ : a₁ + 6 * d = a₁ * (a₁ + d)) :
  a₁ + 9 * d = 21 := 
by
  sorry

end woman_weaves_ten_day_units_l143_143792


namespace recolor_possible_l143_143892

theorem recolor_possible (cell_color : Fin 50 → Fin 50 → Fin 100)
  (H1 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color i (j+1) = l ∧ k ≠ l ∧ j < 49)
  (H2 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color (i+1) j = l ∧ k ≠ l ∧ i < 49) :
  ∃ c1 c2, (c1 ≠ c2) ∧
  ∀ i j, (cell_color i j = c1 → cell_color i j = c2 ∨ ∀ k l, (cell_color k l = c1 → cell_color k l ≠ c2)) :=
  by
  sorry

end recolor_possible_l143_143892


namespace hypotenuse_of_right_triangle_l143_143092

theorem hypotenuse_of_right_triangle (h : height_dropped_to_hypotenuse = 1) (a : acute_angle = 15) :
∃ (hypotenuse : ℝ), hypotenuse = 4 :=
sorry

end hypotenuse_of_right_triangle_l143_143092


namespace probability_of_rolling_3_or_5_is_1_over_4_l143_143491

def fair_8_sided_die := {outcome : Fin 8 // true}

theorem probability_of_rolling_3_or_5_is_1_over_4 :
  (1 / 4 : ℚ) = 2 / 8 :=
by sorry

end probability_of_rolling_3_or_5_is_1_over_4_l143_143491


namespace example_function_indeterminate_unbounded_l143_143623

theorem example_function_indeterminate_unbounded:
  (∀ x, ∃ f : ℝ → ℝ, (f x = (x^2 + x - 2) / (x^3 + 2 * x + 1)) ∧ 
                      (f 1 = (0 / (1^3 + 2 * 1 + 1))) ∧
                      (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε)) :=
by
  sorry

end example_function_indeterminate_unbounded_l143_143623


namespace sally_pokemon_cards_l143_143714

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end sally_pokemon_cards_l143_143714


namespace ruby_height_is_192_l143_143659

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l143_143659


namespace distance_from_x_axis_l143_143721

theorem distance_from_x_axis (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end distance_from_x_axis_l143_143721


namespace job_time_relation_l143_143824

theorem job_time_relation (a b c m n x : ℝ) 
  (h1 : m / a = 1 / b + 1 / c)
  (h2 : n / b = 1 / a + 1 / c)
  (h3 : x / c = 1 / a + 1 / b) :
  x = (m + n + 2) / (m * n - 1) := 
sorry

end job_time_relation_l143_143824


namespace ads_ratio_l143_143111

theorem ads_ratio 
  (first_ads : ℕ := 12)
  (second_ads : ℕ)
  (third_ads := second_ads + 24)
  (fourth_ads := (3 / 4) * second_ads)
  (clicked_ads := 68)
  (total_ads := (3 / 2) * clicked_ads == 102)
  (ads_eq : first_ads + second_ads + third_ads + fourth_ads = total_ads) :
  second_ads / first_ads = 2 :=
by sorry

end ads_ratio_l143_143111


namespace construct_triangle_num_of_solutions_l143_143381

theorem construct_triangle_num_of_solutions
  (r : ℝ) -- Circumradius
  (beta_gamma_diff : ℝ) -- Angle difference \beta - \gamma
  (KA1 : ℝ) -- Segment K A_1
  (KA1_lt_r : KA1 < r) -- Segment K A1 should be less than the circumradius
  (delta : ℝ := beta_gamma_diff) : 1 ≤ num_solutions ∧ num_solutions ≤ 2 :=
sorry

end construct_triangle_num_of_solutions_l143_143381


namespace meet_second_time_4_5_minutes_l143_143367

-- Define the initial conditions
def opposite_ends := true      -- George and Henry start from opposite ends
def pass_in_center := 1.5      -- They pass each other in the center after 1.5 minutes
def no_time_lost := true       -- No time lost in turning
def constant_speeds := true    -- They maintain their respective speeds

-- Prove that they pass each other the second time after 4.5 minutes
theorem meet_second_time_4_5_minutes :
  opposite_ends ∧ pass_in_center = 1.5 ∧ no_time_lost ∧ constant_speeds → 
  ∃ t : ℝ, t = 4.5 := by
  sorry

end meet_second_time_4_5_minutes_l143_143367


namespace coupon_discount_l143_143172

theorem coupon_discount (total_before_coupon : ℝ) (amount_paid_per_friend : ℝ) (number_of_friends : ℕ) :
  total_before_coupon = 100 ∧ amount_paid_per_friend = 18.8 ∧ number_of_friends = 5 →
  ∃ discount_percentage : ℝ, discount_percentage = 6 :=
by
  sorry

end coupon_discount_l143_143172


namespace find_linear_equation_l143_143798

def is_linear_eq (eq : String) : Prop :=
  eq = "2x = 0"

theorem find_linear_equation :
  is_linear_eq "2x = 0" :=
by
  sorry

end find_linear_equation_l143_143798


namespace problem_statement_l143_143228

theorem problem_statement :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
    (∀ x : ℝ, 1 + x^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + 
              a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) ∧
    (a_0 = 2) ∧
    (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 33)) →
  (∃ a_1 a_2 a_3 a_4 a_5 : ℝ, a_1 + a_2 + a_3 + a_4 + a_5 = 31) :=
by
  sorry

end problem_statement_l143_143228


namespace problem_A_inter_complement_B_l143_143890

noncomputable def A : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}
noncomputable def B : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}
noncomputable def complement_B : Set ℝ := {x : ℝ | ¬((x - 2) / (x - 1) ≥ 0)}

theorem problem_A_inter_complement_B : 
  (A ∩ complement_B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end problem_A_inter_complement_B_l143_143890


namespace find_days_A_alone_works_l143_143717

-- Given conditions
def A_is_twice_as_fast_as_B (a b : ℕ) : Prop := a = b / 2
def together_complete_in_12_days (a b : ℕ) : Prop := (1 / b + 1 / a) = 1 / 12

-- We need to prove that A alone can finish the work in 18 days.
def A_alone_in_18_days (a : ℕ) : Prop := a = 18

theorem find_days_A_alone_works :
  ∃ (a b : ℕ), A_is_twice_as_fast_as_B a b ∧ together_complete_in_12_days a b ∧ A_alone_in_18_days a :=
sorry

end find_days_A_alone_works_l143_143717


namespace digit_Phi_l143_143624

theorem digit_Phi (Phi : ℕ) (h1 : 220 / Phi = 40 + 3 * Phi) : Phi = 4 :=
by
  sorry

end digit_Phi_l143_143624


namespace tourist_groups_meet_l143_143857

theorem tourist_groups_meet (x y : ℝ) (h1 : 4.5 * x + 2.5 * y = 30) (h2 : 3 * x + 5 * y = 30) : 
  x = 5 ∧ y = 3 := 
sorry

end tourist_groups_meet_l143_143857


namespace fixed_point_exists_trajectory_M_trajectory_equation_l143_143074

variable (m : ℝ)
def line_l (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0
def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0

theorem fixed_point_exists :
  ∃ x y : ℝ, (line_l m x y ∧ x = 1 ∧ y = -2) :=
by
  sorry

theorem trajectory_M :
  ∃ (M: ℝ × ℝ), (line_l m M.1 M.2 ∧ M = (0, -1)) :=
by
  sorry

theorem trajectory_equation (x y : ℝ) :
  ∃ (x y : ℝ), (x + 1) ^ 2  + y ^ 2 = 2 :=
by
  sorry

end fixed_point_exists_trajectory_M_trajectory_equation_l143_143074


namespace bottles_produced_l143_143905

/-- 
14 machines produce 2520 bottles in 4 minutes, given that 6 machines produce 270 bottles per minute. 
-/
theorem bottles_produced (rate_6_machines : Nat) (bottles_per_minute : Nat) 
  (rate_one_machine : Nat) (rate_14_machines : Nat) (total_production : Nat) : 
  rate_6_machines = 6 ∧ bottles_per_minute = 270 ∧ rate_one_machine = bottles_per_minute / rate_6_machines 
  ∧ rate_14_machines = 14 * rate_one_machine ∧ total_production = rate_14_machines * 4 → 
  total_production = 2520 :=
sorry

end bottles_produced_l143_143905


namespace complex_division_example_l143_143423

theorem complex_division_example : (2 - (1 : ℂ) * Complex.I) / (1 - (1 : ℂ) * Complex.I) = (3 / 2) + (1 / 2) * Complex.I :=
by
  sorry

end complex_division_example_l143_143423


namespace quadratic_solution_exists_l143_143100

-- Define the conditions
variables (a b : ℝ) (h₀ : a ≠ 0)
-- The condition that the first quadratic equation has at most one solution
def has_at_most_one_solution (a b : ℝ) : Prop :=
  b^2 + 4*a*(a - 3) <= 0

-- The second quadratic equation
def second_equation (a b x : ℝ) : ℝ :=
  (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3
  
-- The proof problem invariant in Lean 4
theorem quadratic_solution_exists (h₁ : has_at_most_one_solution a b) :
  ∃ x : ℝ, second_equation a b x = 0 :=
by
  sorry

end quadratic_solution_exists_l143_143100


namespace inequality_proof_l143_143668

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x^2 + y * z) + 1 / (y^2 + z * x) + 1 / (z^2 + x * y)) ≤ 
  (1 / 2) * (1 / (x * y) + 1 / (y * z) + 1 / (z * x)) :=
by sorry

end inequality_proof_l143_143668


namespace x_intercept_of_line_l143_143875

theorem x_intercept_of_line : ∃ x : ℚ, 3 * x + 5 * 0 = 20 ∧ (x, 0) = (20/3, 0) :=
by
  sorry

end x_intercept_of_line_l143_143875


namespace alpha_plus_beta_l143_143666

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  ∀ x : ℝ, ((x - α) / (x + β)) = ((x^2 - 54 * x + 621) / (x^2 + 42 * x - 1764))

theorem alpha_plus_beta : ∃ α β : ℝ, α + β = 86 ∧ alpha_beta α β :=
by
  sorry

end alpha_plus_beta_l143_143666


namespace baoh2_formation_l143_143808

noncomputable def moles_of_baoh2_formed (moles_bao : ℕ) (moles_h2o : ℕ) : ℕ :=
  if moles_bao = moles_h2o then moles_bao else sorry

theorem baoh2_formation :
  moles_of_baoh2_formed 3 3 = 3 :=
by sorry

end baoh2_formation_l143_143808


namespace reciprocal_eq_self_l143_143956

theorem reciprocal_eq_self (x : ℝ) : (1 / x = x) ↔ (x = 1 ∨ x = -1) :=
sorry

end reciprocal_eq_self_l143_143956


namespace hyperbola_focus_coordinates_l143_143736

theorem hyperbola_focus_coordinates :
  let a := 7
  let b := 11
  let h := 5
  let k := -3
  let c := Real.sqrt (a^2 + b^2)
  (∃ x y : ℝ, (x = h + c ∧ y = k) ∧ (∀ x' y', (x' = h + c ∧ y' = k) ↔ (x = x' ∧ y = y'))) :=
by
  sorry

end hyperbola_focus_coordinates_l143_143736


namespace point_P_on_x_axis_l143_143997

noncomputable def point_on_x_axis (m : ℝ) : ℝ × ℝ := (4, m + 1)

theorem point_P_on_x_axis (m : ℝ) (h : point_on_x_axis m = (4, 0)) : m = -1 := 
by
  sorry

end point_P_on_x_axis_l143_143997


namespace greatest_QPN_value_l143_143849

theorem greatest_QPN_value (N : ℕ) (Q P : ℕ) (QPN : ℕ) :
  (NN : ℕ) =
  10 * N + N ∧
  QPN = 100 * Q + 10 * P + N ∧
  N < 10 ∧ N ≥ 1 ∧
  NN * N = QPN ∧
  NN >= 10 ∧ NN < 100  -- Ensuring NN is a two-digit number
  → QPN <= 396 := sorry

end greatest_QPN_value_l143_143849


namespace beth_cans_of_corn_l143_143501

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l143_143501


namespace determine_min_k_l143_143788

open Nat

theorem determine_min_k (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → ℕ) (b : Fin (choose n 2) → ℕ) : 
  ∃ k, k = (n - 1) * (n - 2) / 2 + 1 := 
sorry

end determine_min_k_l143_143788


namespace Keiko_speed_is_pi_div_3_l143_143902

noncomputable def Keiko_avg_speed {r : ℝ} (v : ℝ → ℝ) (pi : ℝ) : ℝ :=
let C1 := 2 * pi * (r + 6) - 2 * pi * r
let t1 := 36
let v1 := C1 / t1

let C2 := 2 * pi * (r + 8) - 2 * pi * r
let t2 := 48
let v2 := C2 / t2

if v r = v1 ∧ v r = v2 then (v1 + v2) / 2 else 0

theorem Keiko_speed_is_pi_div_3 (pi : ℝ) (r : ℝ) (v : ℝ → ℝ) :
  v r = π / 3 ∧ (forall t1 t2 C1 C2, C1 / t1 = π / 3 ∧ C2 / t2 = π / 3 → 
  (C1/t1 + C2/t2)/2 = π / 3) :=
sorry

end Keiko_speed_is_pi_div_3_l143_143902


namespace episodes_count_l143_143123

variable (minutes_per_episode : ℕ) (total_watching_time_minutes : ℕ)
variable (episodes_watched : ℕ)

theorem episodes_count 
  (h1 : minutes_per_episode = 50) 
  (h2 : total_watching_time_minutes = 300) 
  (h3 : total_watching_time_minutes / minutes_per_episode = episodes_watched) :
  episodes_watched = 6 := sorry

end episodes_count_l143_143123


namespace carousel_revolutions_l143_143141

/-- Prove that the number of revolutions a horse 4 feet from the center needs to travel the same distance
as a horse 16 feet from the center making 40 revolutions is 160 revolutions. -/
theorem carousel_revolutions (r₁ : ℕ := 16) (revolutions₁ : ℕ := 40) (r₂ : ℕ := 4) :
  (revolutions₁ * (r₁ / r₂) = 160) :=
sorry

end carousel_revolutions_l143_143141


namespace time_after_1876_minutes_l143_143937

-- Define the structure for Time
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
noncomputable def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.minute + m
  let additional_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let new_hour := (t.hour + additional_hours) % 24
  { hour := new_hour, minute := remaining_minutes }

-- Definition of the starting time
def three_pm : Time := { hour := 15, minute := 0 }

-- The main theorem statement
theorem time_after_1876_minutes : add_minutes three_pm 1876 = { hour := 10, minute := 16 } :=
  sorry

end time_after_1876_minutes_l143_143937


namespace vector_AB_equality_l143_143744

variable {V : Type*} [AddCommGroup V]

variables (a b : V)

theorem vector_AB_equality (BC CA : V) (hBC : BC = a) (hCA : CA = b) :
  CA - BC = b - a :=
by {
  sorry
}

end vector_AB_equality_l143_143744


namespace trader_cloth_sold_l143_143461

variable (x : ℕ)
variable (profit_per_meter total_profit : ℕ)

theorem trader_cloth_sold (h_profit_per_meter : profit_per_meter = 55)
  (h_total_profit : total_profit = 2200) :
  55 * x = 2200 → x = 40 :=
by 
  sorry

end trader_cloth_sold_l143_143461


namespace percentage_increase_l143_143045

theorem percentage_increase (L : ℝ) (h : L + 60 = 240) : ((60 / L) * 100 = 33 + (1 / 3) * 100) :=
by
  sorry

end percentage_increase_l143_143045


namespace intersection_sets_m_n_l143_143912

theorem intersection_sets_m_n :
  let M := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
  let N := { x : ℝ | x > 0 }
  M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_m_n_l143_143912


namespace nathan_dice_roll_probability_l143_143493

noncomputable def probability_nathan_rolls : ℚ :=
  let prob_less4_first_die : ℚ := 3 / 8
  let prob_greater5_second_die : ℚ := 3 / 8
  prob_less4_first_die * prob_greater5_second_die

theorem nathan_dice_roll_probability : probability_nathan_rolls = 9 / 64 := by
  sorry

end nathan_dice_roll_probability_l143_143493


namespace train_length_l143_143003

theorem train_length (speed_kph : ℕ) (tunnel_length_m : ℕ) (time_s : ℕ) : 
  speed_kph = 54 → 
  tunnel_length_m = 1200 → 
  time_s = 100 → 
  ∃ train_length_m : ℕ, train_length_m = 300 := 
by
  intros h1 h2 h3
  have speed_mps : ℕ := (speed_kph * 1000) / 3600 
  have total_distance_m : ℕ := speed_mps * time_s
  have train_length_m : ℕ := total_distance_m - tunnel_length_m
  use train_length_m
  sorry

end train_length_l143_143003


namespace arithmetic_sequence_general_term_and_k_l143_143629

theorem arithmetic_sequence_general_term_and_k (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77) :
  (∀ n : ℕ, a n = (2 * n + 3) / 3) ∧ (∃ k : ℕ, a k = 13 ∧ k = 18) := 
by
  sorry

end arithmetic_sequence_general_term_and_k_l143_143629


namespace max_product_300_l143_143783

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l143_143783


namespace triangle_no_two_obtuse_angles_l143_143219

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l143_143219


namespace yolkino_palkino_l143_143006

open Nat

/-- On every kilometer of the highway between the villages Yolkino and Palkino, there is a post with a sign.
    On one side of the sign, the distance to Yolkino is written, and on the other side, the distance to Palkino is written.
    The sum of all the digits on each post equals 13.
    Prove that the distance from Yolkino to Palkino is 49 kilometers. -/
theorem yolkino_palkino (n : ℕ) (h : ∀ k : ℕ, k ≤ n → (digits 10 k).sum + (digits 10 (n - k)).sum = 13) : n = 49 :=
by
  sorry

end yolkino_palkino_l143_143006


namespace slope_of_line_l143_143855

theorem slope_of_line : 
  (∀ x y : ℝ, (y = (1/2) * x + 1) → ∃ m : ℝ, m = 1/2) :=
sorry

end slope_of_line_l143_143855


namespace chairs_to_remove_is_33_l143_143102

-- Definitions for the conditions
def chairs_per_row : ℕ := 11
def total_chairs : ℕ := 110
def students : ℕ := 70

-- Required statement
theorem chairs_to_remove_is_33 
  (h_divisible_by_chairs_per_row : ∀ n, n = total_chairs - students → ∃ k, n = chairs_per_row * k) :
  ∃ rem_chairs : ℕ, rem_chairs = total_chairs - 77 ∧ rem_chairs = 33 := sorry

end chairs_to_remove_is_33_l143_143102


namespace geometric_series_sum_l143_143563

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l143_143563


namespace factorize_expression_l143_143399

theorem factorize_expression (x y : ℝ) : x^3 * y - 4 * x * y = x * y * (x - 2) * (x + 2) :=
sorry

end factorize_expression_l143_143399


namespace acute_angle_implies_x_range_l143_143411

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l143_143411


namespace mileage_on_city_streets_l143_143580

-- Defining the given conditions
def distance_on_highways : ℝ := 210
def mileage_on_highways : ℝ := 35
def total_gas_used : ℝ := 9
def distance_on_city_streets : ℝ := 54

-- Proving the mileage on city streets
theorem mileage_on_city_streets :
  ∃ x : ℝ, 
    (distance_on_highways / mileage_on_highways + distance_on_city_streets / x = total_gas_used)
    ∧ x = 18 :=
by
  sorry

end mileage_on_city_streets_l143_143580


namespace sum_of_digits_of_n_l143_143597

theorem sum_of_digits_of_n :
  ∃ n : ℕ,
    n > 2000 ∧
    n + 135 % 75 = 15 ∧
    n + 75 % 135 = 45 ∧
    (n = 2025 ∧ (2 + 0 + 2 + 5 = 9)) :=
by
  sorry

end sum_of_digits_of_n_l143_143597


namespace quadratic_two_distinct_real_roots_l143_143007

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l143_143007


namespace eq_a2b2_of_given_condition_l143_143317

theorem eq_a2b2_of_given_condition (a b : ℝ) (h : a^4 + b^4 = a^2 - 2 * a^2 * b^2 + b^2 + 6) : a^2 + b^2 = 3 :=
sorry

end eq_a2b2_of_given_condition_l143_143317


namespace common_ratio_q_l143_143072

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

axiom a5_condition : a_n 5 = 2 * S_n 4 + 3
axiom a6_condition : a_n 6 = 2 * S_n 5 + 3

theorem common_ratio_q : q = 3 :=
by
  sorry

end common_ratio_q_l143_143072


namespace no_nat_solutions_l143_143358
-- Import the Mathlib library

-- Lean statement for the proof problem
theorem no_nat_solutions (x : ℕ) : ¬ (19 * x^2 + 97 * x = 1997) :=
by {
  -- Solution omitted
  sorry
}

end no_nat_solutions_l143_143358


namespace find_t_l143_143826

open Complex Real

theorem find_t (a b : ℂ) (t : ℝ) (h₁ : abs a = 3) (h₂ : abs b = 5) (h₃ : a * b = t - 3 * I) :
  t = 6 * Real.sqrt 6 := by
  sorry

end find_t_l143_143826


namespace complex_expression_evaluation_l143_143895

-- Definition of the imaginary unit i with property i^2 = -1
def i : ℂ := Complex.I

-- Theorem stating that the given expression equals i
theorem complex_expression_evaluation : i * (1 - i) - 1 = i := by
  -- Proof omitted
  sorry

end complex_expression_evaluation_l143_143895


namespace div_neg_cancel_neg_div_example_l143_143509

theorem div_neg_cancel (x y : Int) (h : y ≠ 0) : (-x) / (-y) = x / y := by
  sorry

theorem neg_div_example : (-64 : Int) / (-32) = 2 := by
  apply div_neg_cancel
  norm_num

end div_neg_cancel_neg_div_example_l143_143509


namespace no_solution_2023_l143_143033

theorem no_solution_2023 (a b c : ℕ) (h₁ : a + b + c = 2023) (h₂ : (b + c) ∣ a) (h₃ : (b - c + 1) ∣ (b + c)) : false :=
by
  sorry

end no_solution_2023_l143_143033


namespace increasing_interval_of_y_l143_143184

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end increasing_interval_of_y_l143_143184


namespace find_two_digit_number_l143_143854

theorem find_two_digit_number (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : n % 2 = 0)
  (h3 : (n + 1) % 3 = 0)
  (h4 : (n + 2) % 4 = 0)
  (h5 : (n + 3) % 5 = 0) : n = 62 :=
by
  sorry

end find_two_digit_number_l143_143854


namespace slope_y_intercept_product_eq_neg_five_over_two_l143_143208

theorem slope_y_intercept_product_eq_neg_five_over_two :
  let A := (0, 10)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (10 + 0) / 2) -- midpoint of A and B
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope * y_intercept = -5 / 2 := 
by 
  sorry

end slope_y_intercept_product_eq_neg_five_over_two_l143_143208


namespace tangent_k_value_one_common_point_range_l143_143756

namespace Geometry

-- Definitions:
def line (k : ℝ) : ℝ → ℝ := λ x => k * x - 3 * k + 2
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4
def is_tangent (k : ℝ) : Prop := |-2 * k + 3| / (Real.sqrt (k^2 + 1)) = 2
def has_only_one_common_point (k : ℝ) : Prop :=
  (1 / 2 < k ∧ k <= 5 / 2) ∨ (k = 5 / 12)

-- Theorem statements:
theorem tangent_k_value : ∀ k : ℝ, is_tangent k → k = 5 / 12 := sorry

theorem one_common_point_range : ∀ k : ℝ, has_only_one_common_point k → k ∈
  Set.union (Set.Ioc (1 / 2) (5 / 2)) {5 / 12} := sorry

end Geometry

end tangent_k_value_one_common_point_range_l143_143756


namespace algae_difference_l143_143115

theorem algae_difference :
  let original_algae := 809
  let current_algae := 3263
  current_algae - original_algae = 2454 :=
by
  sorry

end algae_difference_l143_143115


namespace total_wings_l143_143619

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end total_wings_l143_143619


namespace proof_problem_l143_143110

variables {x1 y1 x2 y2 : ℝ}

-- Definitions
def unit_vector (x y : ℝ) : Prop := x^2 + y^2 = 1
def angle_with_p (x y : ℝ) : Prop := (x + y) / Real.sqrt 2 = Real.sqrt 3 / 2
def m := (x1, y1)
def n := (x2, y2)
def p := (1, 1)

-- Conditions
lemma unit_m : unit_vector x1 y1 := sorry
lemma unit_n : unit_vector x2 y2 := sorry
lemma angle_m_p : angle_with_p x1 y1 := sorry
lemma angle_n_p : angle_with_p x2 y2 := sorry

-- Theorem to prove
theorem proof_problem (h1 : unit_vector x1 y1)
                      (h2 : unit_vector x2 y2)
                      (h3 : angle_with_p x1 y1)
                      (h4 : angle_with_p x2 y2) :
                      (x1 * x2 + y1 * y2 = 1/2) ∧ (y1 * y2 / (x1 * x2) = 1) :=
sorry

end proof_problem_l143_143110


namespace circumcircle_eq_of_triangle_vertices_l143_143241

theorem circumcircle_eq_of_triangle_vertices (A B C: ℝ × ℝ) (hA : A = (0, 4)) (hB : B = (0, 0)) (hC : C = (3, 0)) :
  ∃ D E F : ℝ,
    x^2 + y^2 + D*x + E*y + F = 0 ∧
    (x - 3/2)^2 + (y - 2)^2 = 25/4 :=
by 
  sorry

end circumcircle_eq_of_triangle_vertices_l143_143241


namespace jerry_current_average_l143_143634

theorem jerry_current_average (A : ℚ) (h1 : 3 * A + 89 = 4 * (A + 2)) : A = 81 := 
by
  sorry

end jerry_current_average_l143_143634


namespace range_of_a_l143_143521

theorem range_of_a (a : ℝ) :
  let A := {x | x^2 + 4 * x = 0}
  let B := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}
  A ∩ B = B → (a = 1 ∨ a ≤ -1) := 
by
  sorry

end range_of_a_l143_143521


namespace difference_max_min_students_l143_143697

-- Definitions for problem conditions
def total_students : ℕ := 50
def shanghai_university_min : ℕ := 40
def shanghai_university_max : ℕ := 45
def shanghai_normal_university_min : ℕ := 16
def shanghai_normal_university_max : ℕ := 20

-- Lean statement for the math proof problem
theorem difference_max_min_students :
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                15 ≤ a + b - total_students ∧ a + b - total_students ≤ 15) →
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                6 ≤ a + b - total_students ∧ a + b - total_students ≤ 6) →
  (∃ M m : ℕ, 
    (M = 15) ∧ 
    (m = 6) ∧ 
    (M - m = 9)) :=
by
  sorry

end difference_max_min_students_l143_143697


namespace intersection_empty_l143_143163

def A : Set ℝ := {x | x > -1 ∧ x ≤ 3}
def B : Set ℝ := {2, 4}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l143_143163


namespace divisor_problem_l143_143446

theorem divisor_problem :
  ∃ D : ℕ, 12401 = D * 76 + 13 ∧ D = 163 := 
by
  sorry

end divisor_problem_l143_143446


namespace geometric_series_sum_l143_143055

-- Define the first term and common ratio
def a : ℚ := 5 / 3
def r : ℚ := -1 / 6

-- Prove the sum of the infinite geometric series
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 10 / 7 := by
  sorry

end geometric_series_sum_l143_143055


namespace expressions_cannot_all_exceed_one_fourth_l143_143881

theorem expressions_cannot_all_exceed_one_fourth (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := 
by
  sorry

end expressions_cannot_all_exceed_one_fourth_l143_143881


namespace white_paint_amount_is_correct_l143_143091

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end white_paint_amount_is_correct_l143_143091


namespace right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l143_143544

theorem right_triangle_arithmetic_progression_is_345 (a b c : ℕ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ d, b = a + d ∧ c = a + 2 * d)
  : (a, b, c) = (3, 4, 5) :=
by
  sorry

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

noncomputable def sqrt_golden_ratio_div_2 := Real.sqrt ((1 + Real.sqrt 5) / 2)

theorem right_triangle_geometric_progression 
  (a b c : ℝ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ r, b = a * r ∧ c = a * r * r)
  : (a, b, c) = (1, sqrt_golden_ratio_div_2, golden_ratio) :=
by
  sorry

end right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l143_143544


namespace second_valve_rate_difference_l143_143541

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end second_valve_rate_difference_l143_143541


namespace find_three_power_l143_143953

theorem find_three_power (m n : ℕ) (h₁: 3^m = 4) (h₂: 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end find_three_power_l143_143953


namespace problem_proof_l143_143526

open Real

noncomputable def angle_B (A C : ℝ) : ℝ := π / 3

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1/2) * a * c * (sqrt 3 / 2)

theorem problem_proof (A B C a b c : ℝ)
  (h1 : 2 * cos A * cos C * (tan A * tan C - 1) = 1)
  (h2 : a + c = sqrt 15)
  (h3 : b = sqrt 3)
  (h4 : B = π / 3) :
  (B = angle_B A C) ∧ 
  (area_triangle a b c = sqrt 3) :=
by
  sorry

end problem_proof_l143_143526


namespace min_value_pm_pn_l143_143213

theorem min_value_pm_pn (x y : ℝ)
  (h : x ^ 2 - y ^ 2 / 3 = 1) 
  (hx : 1 ≤ x) : (8 * x - 3) = 5 :=
sorry

end min_value_pm_pn_l143_143213


namespace base_8_to_decimal_77_eq_63_l143_143840

-- Define the problem in Lean 4
theorem base_8_to_decimal_77_eq_63 (k a1 a2 : ℕ) (h_k : k = 8) (h_a1 : a1 = 7) (h_a2 : a2 = 7) :
    a2 * k^1 + a1 * k^0 = 63 := 
by
  -- Placeholder for proof
  sorry

end base_8_to_decimal_77_eq_63_l143_143840


namespace f_divides_f_2k_plus_1_f_coprime_f_multiple_l143_143204

noncomputable def f (g n : ℕ) : ℕ := g ^ n + 1

theorem f_divides_f_2k_plus_1 (g : ℕ) (k n : ℕ) :
  f g n ∣ f g ((2 * k + 1) * n) :=
by sorry

theorem f_coprime_f_multiple (g n : ℕ) :
  Nat.Coprime (f g n) (f g (2 * n)) ∧
  Nat.Coprime (f g n) (f g (4 * n)) ∧
  Nat.Coprime (f g n) (f g (6 * n)) :=
by sorry

end f_divides_f_2k_plus_1_f_coprime_f_multiple_l143_143204


namespace max_positive_n_l143_143041

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

noncomputable def sequence_condition (a : ℕ → ℤ) : Prop :=
a 1010 / a 1009 < -1

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * (a 1 + a n) / 2

theorem max_positive_n (a : ℕ → ℤ) (h1 : is_arithmetic_sequence a) 
    (h2 : sequence_condition a) : n = 2018 ∧ sum_of_first_n_terms a 2018 > 0 := sorry

end max_positive_n_l143_143041


namespace chocolates_problem_l143_143400

-- Let denote the quantities as follows:
-- C: number of caramels
-- N: number of nougats
-- T: number of truffles
-- P: number of peanut clusters

def C_nougats_truffles_peanutclusters (C N T P : ℕ) :=
  N = 2 * C ∧
  T = C + 6 ∧
  C + N + T + P = 50 ∧
  P = 32

theorem chocolates_problem (C N T P : ℕ) :
  C_nougats_truffles_peanutclusters C N T P → C = 3 :=
by
  intros h
  have hN := h.1
  have hT := h.2.1
  have hSum := h.2.2.1
  have hP := h.2.2.2
  sorry

end chocolates_problem_l143_143400


namespace find_pq_l143_143472

noncomputable def p_and_q (p q : ℝ) := 
  (Complex.I * 2 - 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0} ∧ 
  - (Complex.I * 2 + 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0}

theorem find_pq : ∃ (p q : ℝ), p_and_q p q ∧ p + q = 38 :=
by
  sorry

end find_pq_l143_143472


namespace fraction_complex_eq_l143_143596

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end fraction_complex_eq_l143_143596


namespace younger_brother_height_l143_143733

theorem younger_brother_height
  (O Y : ℕ)
  (h1 : O - Y = 12)
  (h2 : O + Y = 308) :
  Y = 148 :=
by
  sorry

end younger_brother_height_l143_143733


namespace pencil_cost_l143_143836

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (h1 : total_money = 50) (h2 : num_pencils = 10) :
    (total_money / num_pencils) = 5 :=
by
  sorry

end pencil_cost_l143_143836


namespace problem_ab_value_l143_143966

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

theorem problem_ab_value (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) → a * b = 12 :=
by
  intro h
  let f_eqn := h 1 -- Checking the function equality for x = 1
  sorry

end problem_ab_value_l143_143966


namespace isosceles_triangle_apex_angle_l143_143718

theorem isosceles_triangle_apex_angle (base_angle : ℝ) (h_base_angle : base_angle = 42) : 
  180 - 2 * base_angle = 96 :=
by
  sorry

end isosceles_triangle_apex_angle_l143_143718


namespace C_can_complete_work_in_100_days_l143_143716

-- Definitions for conditions
def A_work_rate : ℚ := 1 / 20
def B_work_rate : ℚ := 1 / 15
def work_done_by_A_and_B : ℚ := 6 * (1 / 20 + 1 / 15)
def remaining_work : ℚ := 1 - work_done_by_A_and_B
def work_done_by_A_in_5_days : ℚ := 5 * (1 / 20)
def work_done_by_C_in_5_days : ℚ := remaining_work - work_done_by_A_in_5_days
def C_work_rate_in_5_days : ℚ := work_done_by_C_in_5_days / 5

-- Statement to prove
theorem C_can_complete_work_in_100_days : 
  work_done_by_C_in_5_days ≠ 0 → 1 / C_work_rate_in_5_days = 100 :=
by
  -- proof of the theorem
  sorry

end C_can_complete_work_in_100_days_l143_143716


namespace range_of_decreasing_function_l143_143935

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x < 3 → (deriv (f a) x) ≤ 0) ↔ 0 ≤ a ∧ a ≤ 3/4 := 
sorry

end range_of_decreasing_function_l143_143935


namespace permutations_of_3_3_3_7_7_l143_143073

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end permutations_of_3_3_3_7_7_l143_143073


namespace maximum_value_x_squared_plus_2y_l143_143049

theorem maximum_value_x_squared_plus_2y (x y b : ℝ) (h_curve : x^2 / 4 + y^2 / b^2 = 1) (h_b_positive : b > 0) : 
  x^2 + 2 * y ≤ max (b^2 / 4 + 4) (2 * b) :=
sorry

end maximum_value_x_squared_plus_2y_l143_143049


namespace emerson_rowed_last_part_l143_143776

-- Define the given conditions
def emerson_initial_distance: ℝ := 6
def emerson_continued_distance: ℝ := 15
def total_trip_distance: ℝ := 39

-- Define the distance Emerson covered before the last part
def distance_before_last_part := emerson_initial_distance + emerson_continued_distance

-- Define the distance Emerson rowed in the last part of his trip
def distance_last_part := total_trip_distance - distance_before_last_part

-- The theorem we need to prove
theorem emerson_rowed_last_part : distance_last_part = 18 := by
  sorry

end emerson_rowed_last_part_l143_143776


namespace sam_initial_investment_is_6000_l143_143079

variables (P : ℝ)
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.10 / 2) ^ (2 * 1)

theorem sam_initial_investment_is_6000 :
  final_amount 6000 = 6615 :=
by
  unfold final_amount
  sorry

end sam_initial_investment_is_6000_l143_143079


namespace library_books_total_l143_143253

-- Definitions for the conditions
def books_purchased_last_year : Nat := 50
def books_purchased_this_year : Nat := 3 * books_purchased_last_year
def books_before_last_year : Nat := 100

-- The library's current number of books
def total_books_now : Nat :=
  books_before_last_year + books_purchased_last_year + books_purchased_this_year

-- The proof statement
theorem library_books_total : total_books_now = 300 :=
by
  -- Placeholder for actual proof
  sorry

end library_books_total_l143_143253


namespace minimum_value_expression_l143_143681

theorem minimum_value_expression (p q r s t u v w : ℝ) (h1 : p > 0) (h2 : q > 0) 
    (h3 : r > 0) (h4 : s > 0) (h5 : t > 0) (h6 : u > 0) (h7 : v > 0) (h8 : w > 0)
    (hpqrs : p * q * r * s = 16) (htuvw : t * u * v * w = 25) 
    (hptqu : p * t = q * u ∧ q * u = r * v ∧ r * v = s * w) : 
    (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 = 80 := sorry

end minimum_value_expression_l143_143681


namespace luna_badges_correct_l143_143598

-- conditions
def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def celestia_badges : ℕ := 52

-- question and answer
theorem luna_badges_correct : total_badges - (hermione_badges + celestia_badges) = 17 :=
by
  sorry

end luna_badges_correct_l143_143598


namespace problem_correct_answer_l143_143364

theorem problem_correct_answer (x y : ℕ) (h1 : y > 3) (h2 : x^2 + y^4 = 2 * ((x - 6)^2 + (y + 1)^2)) : x^2 + y^4 = 1994 :=
  sorry

end problem_correct_answer_l143_143364


namespace semicircle_circumference_correct_l143_143741

noncomputable def perimeter_of_rectangle (l b : ℝ) : ℝ := 2 * (l + b)
noncomputable def side_of_square_by_rectangle (l b : ℝ) : ℝ := perimeter_of_rectangle l b / 4
noncomputable def circumference_of_semicircle (d : ℝ) : ℝ := (Real.pi * (d / 2)) + d

theorem semicircle_circumference_correct :
  let l := 16
  let b := 12
  let d := side_of_square_by_rectangle l b
  circumference_of_semicircle d = 35.98 :=
by
  sorry

end semicircle_circumference_correct_l143_143741


namespace smaller_interior_angle_of_parallelogram_l143_143428

theorem smaller_interior_angle_of_parallelogram (x : ℝ) 
  (h1 : ∃ l, l = x + 90 ∧ x + l = 180) :
  x = 45 :=
by
  obtain ⟨l, hl1, hl2⟩ := h1
  simp only [hl1] at hl2
  linarith

end smaller_interior_angle_of_parallelogram_l143_143428


namespace most_compliant_expression_l143_143879

-- Define the expressions as algebraic terms.
def OptionA : String := "1(1/2)a"
def OptionB : String := "b/a"
def OptionC : String := "3a-1 个"
def OptionD : String := "a * 3"

-- Define a property that represents compliance with standard algebraic notation.
def is_compliant (expr : String) : Prop :=
  expr = OptionB

-- The theorem to prove.
theorem most_compliant_expression :
  is_compliant OptionB :=
by
  sorry

end most_compliant_expression_l143_143879


namespace sculpture_and_base_height_l143_143522

theorem sculpture_and_base_height :
  let sculpture_height_in_feet := 2
  let sculpture_height_in_inches := 10
  let base_height_in_inches := 2
  let total_height_in_inches := (sculpture_height_in_feet * 12) + sculpture_height_in_inches + base_height_in_inches
  let total_height_in_feet := total_height_in_inches / 12
  total_height_in_feet = 3 :=
by
  sorry

end sculpture_and_base_height_l143_143522


namespace simplify_expression_l143_143264

variable (a b : Real)

theorem simplify_expression (a b : Real) : 
    3 * b * (3 * b ^ 2 + 2 * b) - b ^ 2 + 2 * a * (2 * a ^ 2 - 3 * a) - 4 * a * b = 
    9 * b ^ 3 + 5 * b ^ 2 + 4 * a ^ 3 - 6 * a ^ 2 - 4 * a * b := by
  sorry

end simplify_expression_l143_143264


namespace no_correlation_pair_D_l143_143444

-- Define the pairs of variables and their relationships
def pair_A : Prop := ∃ (fertilizer_applied grain_yield : ℝ), (fertilizer_applied ≠ 0 → grain_yield ≠ 0)
def pair_B : Prop := ∃ (review_time scores : ℝ), (review_time ≠ 0 → scores ≠ 0)
def pair_C : Prop := ∃ (advertising_expenses sales : ℝ), (advertising_expenses ≠ 0 → sales ≠ 0)
def pair_D : Prop := ∃ (books_sold revenue : ℕ), (revenue = books_sold * 5)

/-- Prove that pair D does not have a correlation in the context of the problem. --/
theorem no_correlation_pair_D : ¬pair_D :=
by
  sorry

end no_correlation_pair_D_l143_143444


namespace smallest_range_l143_143587

theorem smallest_range {x1 x2 x3 x4 x5 : ℝ} 
  (h1 : (x1 + x2 + x3 + x4 + x5) = 100)
  (h2 : x3 = 18)
  (h3 : 2 * x1 + 2 * x5 + 18 = 100): 
  x5 - x1 = 19 :=
by {
  sorry
}

end smallest_range_l143_143587


namespace range_of_a_l143_143267

theorem range_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, ∀ x : ℝ, x + a * x0 + 1 < 0) → (a ≥ -2 ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l143_143267


namespace total_ticket_revenue_l143_143603

theorem total_ticket_revenue (total_seats : Nat) (price_adult_ticket : Nat) (price_child_ticket : Nat)
  (theatre_full : Bool) (child_tickets : Nat) (adult_tickets := total_seats - child_tickets)
  (rev_adult := adult_tickets * price_adult_ticket) (rev_child := child_tickets * price_child_ticket) :
  total_seats = 250 →
  price_adult_ticket = 6 →
  price_child_ticket = 4 →
  theatre_full = true →
  child_tickets = 188 →
  rev_adult + rev_child = 1124 := 
by
  intros h_total_seats h_price_adult h_price_child h_theatre_full h_child_tickets
  sorry

end total_ticket_revenue_l143_143603


namespace angles_equal_l143_143750

variables {A B C M W L T : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace M] [MetricSpace W] [MetricSpace L] [MetricSpace T]

-- A, B, C are points of the triangle ABC with incircle k.
-- Line_segment AC is longer than line segment BC.
-- M is the intersection of median from C.
-- W is the intersection of angle bisector from C.
-- L is the intersection of altitude from C.
-- T is the point where the tangent from M to the incircle k, different from AB, touches k.
def triangle_ABC (A B C : Type*) : Prop := sorry
def incircle_k (A B C : Type*) (k : Type*) : Prop := sorry
def longer_AC (A B C : Type*) : Prop := sorry
def intersection_median_C (M C : Type*) : Prop := sorry
def intersection_angle_bisector_C (W C : Type*) : Prop := sorry
def intersection_altitude_C (L C : Type*) : Prop := sorry
def tangent_through_M (M T k : Type*) : Prop := sorry
def touches_k (T k : Type*) : Prop := sorry
def angle_eq (M T W L : Type*) : Prop := sorry

theorem angles_equal (A B C M W L T k : Type*)
  (h_triangle : triangle_ABC A B C)
  (h_incircle : incircle_k A B C k)
  (h_longer_AC : longer_AC A B C)
  (h_inter_median : intersection_median_C M C)
  (h_inter_bisector : intersection_angle_bisector_C W C)
  (h_inter_altitude : intersection_altitude_C L C)
  (h_tangent : tangent_through_M M T k)
  (h_touches : touches_k T k) :
  angle_eq M T W L := 
sorry


end angles_equal_l143_143750


namespace remaining_money_l143_143383

-- Definitions
def cost_per_app : ℕ := 4
def num_apps : ℕ := 15
def total_money : ℕ := 66

-- Theorem
theorem remaining_money : total_money - (num_apps * cost_per_app) = 6 := by
  sorry

end remaining_money_l143_143383


namespace find_highest_score_l143_143828

-- Define the conditions for the proof
section
  variable {runs_innings : ℕ → ℕ}

  -- Total runs scored in 46 innings
  def total_runs (average num_innings : ℕ) : ℕ := average * num_innings
  def total_runs_46_innings := total_runs 60 46
  def total_runs_excluding_H_L := total_runs 58 44

  -- Evaluated difference and sum of scores
  def diff_H_and_L : ℕ := 180
  def sum_H_and_L : ℕ := total_runs_46_innings - total_runs_excluding_H_L

  -- Define the proof goal
  theorem find_highest_score (H L : ℕ)
    (h1 : H - L = diff_H_and_L)
    (h2 : H + L = sum_H_and_L) :
    H = 194 :=
  by
    sorry

end

end find_highest_score_l143_143828


namespace people_with_fewer_than_7_cards_l143_143524

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l143_143524


namespace shortest_paths_ratio_l143_143983

theorem shortest_paths_ratio (k n : ℕ) (h : k > 0):
  let paths_along_AB := Nat.choose (k * n + n - 1) (n - 1)
  let paths_along_AD := Nat.choose (k * n + n - 1) k * n - 1
  paths_along_AD = k * paths_along_AB :=
by sorry

end shortest_paths_ratio_l143_143983


namespace solve_for_y_l143_143131

theorem solve_for_y : ∀ y : ℚ, (8 * y^2 + 78 * y + 5) / (2 * y + 19) = 4 * y + 2 → y = -16.5 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l143_143131


namespace car_return_speed_l143_143366

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l143_143366


namespace calc_expression_l143_143961

theorem calc_expression :
  (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end calc_expression_l143_143961


namespace soccer_match_outcome_l143_143069

theorem soccer_match_outcome :
  ∃ n : ℕ, n = 4 ∧
  (∃ (num_wins num_draws num_losses : ℕ),
     num_wins * 3 + num_draws * 1 + num_losses * 0 = 19 ∧
     num_wins + num_draws + num_losses = 14) :=
sorry

end soccer_match_outcome_l143_143069


namespace find_number_l143_143932

theorem find_number (x : ℝ) (h : x = (1 / 3) * x + 120) : x = 180 :=
by
  sorry

end find_number_l143_143932


namespace solution_set_inequality_l143_143510

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := by
  sorry

end solution_set_inequality_l143_143510


namespace trigonometric_expression_identity_l143_143220

theorem trigonometric_expression_identity :
  (2 * Real.sin (100 * Real.pi / 180) - Real.cos (70 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180)
  = 2 * Real.sqrt 3 - 1 :=
sorry

end trigonometric_expression_identity_l143_143220


namespace simplify_fraction_l143_143210

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l143_143210


namespace employees_count_l143_143450

-- Let E be the number of employees excluding the manager
def E (employees : ℕ) : ℕ := employees

-- Let T be the total salary of employees excluding the manager
def T (employees : ℕ) : ℕ := employees * 1500

-- Conditions given in the problem
def average_salary (employees : ℕ) : ℕ := T employees / E employees
def new_average_salary (employees : ℕ) : ℕ := (T employees + 22500) / (E employees + 1)

theorem employees_count : (average_salary employees = 1500) ∧ (new_average_salary employees = 2500) ∧ (manager_salary = 22500) → (E employees = 20) :=
  by sorry

end employees_count_l143_143450


namespace find_numbers_l143_143537

theorem find_numbers (u v : ℝ) (h1 : u^2 + v^2 = 20) (h2 : u * v = 8) :
  (u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2) ∨ (u = -2 ∧ v = -4) ∨ (u = -4 ∧ v = -2) := by
sorry

end find_numbers_l143_143537


namespace large_monkey_doll_cost_l143_143608

theorem large_monkey_doll_cost (S L E : ℝ) 
  (h1 : S = L - 2) 
  (h2 : E = L + 1) 
  (h3 : 300 / S = 300 / L + 25) 
  (h4 : 300 / E = 300 / L - 15) : 
  L = 6 := 
sorry

end large_monkey_doll_cost_l143_143608


namespace parallel_lines_slope_eq_l143_143346

theorem parallel_lines_slope_eq (m : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y - 3 = 0 → 6 * x + m * y + 1 = 0) → m = 4 :=
by
  sorry

end parallel_lines_slope_eq_l143_143346


namespace problem_statement_l143_143487

theorem problem_statement {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) ∧ 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) < 2 :=
by
  sorry

end problem_statement_l143_143487


namespace last_two_nonzero_digits_of_70_factorial_are_04_l143_143075

-- Given conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial_are_04 :
  let n := 70;
  ∀ t : ℕ, 
    t = factorial n → t % 100 ≠ 0 → (t % 100) / 10 != 0 → 
    (t % 100) = 04 :=
sorry

end last_two_nonzero_digits_of_70_factorial_are_04_l143_143075


namespace simplify_and_evaluate_expression_l143_143834

theorem simplify_and_evaluate_expression :
  ∀ x : ℤ, -1 ≤ x ∧ x ≤ 2 →
  (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2) →
  ( ( (x^2 - 1) / (x^2 - 2*x + 1) + ((x^2 - 2*x) / (x - 2)) / x ) = 1 ) :=
by
  intros x hx_constraints x_ne_criteria
  sorry

end simplify_and_evaluate_expression_l143_143834


namespace cardboard_box_height_l143_143046

theorem cardboard_box_height :
  ∃ (x : ℕ), x ≥ 0 ∧ 10 * x^2 + 4 * x ≥ 130 ∧ (2 * x + 1) = 9 :=
sorry

end cardboard_box_height_l143_143046


namespace intersection_complement_eq_singleton_l143_143354

open Set

def U : Set ℤ := {-1, 0, 1, 2, 3, 4}
def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}
def CU_A : Set ℤ := U \ A

theorem intersection_complement_eq_singleton : B ∩ CU_A = {0} := 
by
  sorry

end intersection_complement_eq_singleton_l143_143354


namespace at_least_half_sectors_occupied_l143_143398

theorem at_least_half_sectors_occupied (n : ℕ) (chips : Finset (Fin n.succ)) 
(h_chips_count: chips.card = n + 1) :
  ∃ (steps : ℕ), ∀ (t : ℕ), t ≥ steps → (∃ sector_occupied : Finset (Fin n), sector_occupied.card ≥ n / 2) :=
sorry

end at_least_half_sectors_occupied_l143_143398


namespace minimum_value_expression_l143_143813

open Real

theorem minimum_value_expression (x y z : ℝ) (hxyz : x * y * z = 1 / 2) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * sqrt 6 :=
sorry

end minimum_value_expression_l143_143813


namespace min_sum_length_perpendicular_chords_l143_143283

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end min_sum_length_perpendicular_chords_l143_143283


namespace find_c_l143_143211

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end find_c_l143_143211


namespace shorter_piece_length_l143_143738

theorem shorter_piece_length (total_len : ℝ) (ratio : ℝ) (shorter_len : ℝ) (longer_len : ℝ) 
  (h1 : total_len = 49) (h2 : ratio = 2/5) (h3 : shorter_len = x) 
  (h4 : longer_len = (5/2) * x) (h5 : shorter_len + longer_len = total_len) : 
  shorter_len = 14 := 
by
  sorry

end shorter_piece_length_l143_143738


namespace fraction_difference_l143_143837

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l143_143837


namespace movies_shown_eq_twenty_four_l143_143632

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l143_143632


namespace comparison_of_y1_and_y2_l143_143309

variable {k y1 y2 : ℝ}

theorem comparison_of_y1_and_y2 (hk : 0 < k)
    (hy1 : y1 = k)
    (hy2 : y2 = k / 4) :
    y1 > y2 := by
  sorry

end comparison_of_y1_and_y2_l143_143309


namespace exactly_two_roots_iff_l143_143483

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l143_143483


namespace mixed_doubles_teams_l143_143774

theorem mixed_doubles_teams (m n : ℕ) (h_m : m = 7) (h_n : n = 5) :
  (∃ (k : ℕ), k = 4) ∧ (m ≥ 2) ∧ (n ≥ 2) →
  ∃ (number_of_combinations : ℕ), number_of_combinations = 2 * Nat.choose 7 2 * Nat.choose 5 2 :=
by
  intros
  sorry

end mixed_doubles_teams_l143_143774


namespace ratio_of_octagon_areas_l143_143109

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l143_143109


namespace man_son_work_together_l143_143989

theorem man_son_work_together (man_days : ℝ) (son_days : ℝ) (combined_days : ℝ) :
  man_days = 4 → son_days = 12 → (1 / man_days + 1 / son_days) = 1 / combined_days → combined_days = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end man_son_work_together_l143_143989


namespace fraction_exponentiation_l143_143858

theorem fraction_exponentiation :
  (⟨1/3⟩ : ℝ) ^ 5 = (⟨1/243⟩ : ℝ) :=
by
  sorry

end fraction_exponentiation_l143_143858


namespace point_not_in_third_quadrant_l143_143386

theorem point_not_in_third_quadrant (A : ℝ × ℝ) (h : A.snd = -A.fst + 8) : ¬ (A.fst < 0 ∧ A.snd < 0) :=
sorry

end point_not_in_third_quadrant_l143_143386


namespace difference_in_height_l143_143885

-- Define the heights of the sandcastles
def h_J : ℚ := 3.6666666666666665
def h_S : ℚ := 2.3333333333333335

-- State the theorem
theorem difference_in_height :
  h_J - h_S = 1.333333333333333 := by
  sorry

end difference_in_height_l143_143885


namespace relationship_between_c_squared_and_ab_l143_143987

theorem relationship_between_c_squared_and_ab (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_c : c = (a + b) / 2) : 
  c^2 ≥ a * b := 
sorry

end relationship_between_c_squared_and_ab_l143_143987


namespace john_total_hours_l143_143687

def wall_area (length : ℕ) (width : ℕ) := length * width

def total_area (num_walls : ℕ) (wall_area : ℕ) := num_walls * wall_area

def time_to_paint (area : ℕ) (time_per_square_meter : ℕ) := area * time_per_square_meter

def hours_to_minutes (hours : ℕ) := hours * 60

def total_hours (painting_time : ℕ) (spare_time : ℕ) := painting_time + spare_time

theorem john_total_hours 
  (length width num_walls time_per_square_meter spare_hours : ℕ) 
  (H_length : length = 2) 
  (H_width : width = 3) 
  (H_num_walls : num_walls = 5)
  (H_time_per_square_meter : time_per_square_meter = 10)
  (H_spare_hours : spare_hours = 5) :
  total_hours (time_to_paint (total_area num_walls (wall_area length width)) time_per_square_meter / hours_to_minutes 1) spare_hours = 10 := 
by 
    rw [H_length, H_width, H_num_walls, H_time_per_square_meter, H_spare_hours]
    sorry

end john_total_hours_l143_143687


namespace total_profit_calculation_l143_143105

variable (investment_Tom : ℝ) (investment_Jose : ℝ) (time_Jose : ℝ) (share_Jose : ℝ) (total_time : ℝ) 
variable (total_profit : ℝ)

theorem total_profit_calculation 
  (h1 : investment_Tom = 30000) 
  (h2 : investment_Jose = 45000) 
  (h3 : time_Jose = 10) -- Jose joined 2 months later, so he invested for 10 months out of 12
  (h4 : share_Jose = 30000) 
  (h5 : total_time = 12) 
  : total_profit = 54000 :=
sorry

end total_profit_calculation_l143_143105


namespace min_value_of_function_l143_143807

theorem min_value_of_function :
  ∀ x : ℝ, x > -1 → (y : ℝ) = (x^2 + 7*x + 10) / (x + 1) → y ≥ 9 :=
by
  intros x hx h
  sorry

end min_value_of_function_l143_143807


namespace find_slope_of_l_l143_143964

noncomputable def parabola (x y : ℝ) := y ^ 2 = 4 * x

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Definition of the point M
def M : ℝ × ℝ := (-1, 2)

-- Check if two vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof problem statement
theorem find_slope_of_l (x1 x2 y1 y2 k : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : is_perpendicular (x1 + 1, y1 - 2) (x2 + 1, y2 - 2))
  (eq1 : y1 = k * (x1 - 1))
  (eq2 : y2 = k * (x2 - 1)) :
  k = 1 := by
  sorry

end find_slope_of_l_l143_143964


namespace roots_of_polynomial_l143_143135

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l143_143135


namespace wheel_revolutions_l143_143387

theorem wheel_revolutions (r_course r_wheel : ℝ) (laps : ℕ) (C_course C_wheel : ℝ) (d_total : ℝ) :
  r_course = 7 →
  r_wheel = 5 →
  laps = 15 →
  C_course = 2 * Real.pi * r_course →
  d_total = laps * C_course →
  C_wheel = 2 * Real.pi * r_wheel →
  ((d_total) / (C_wheel)) = 21 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end wheel_revolutions_l143_143387


namespace count_perfect_squares_divisible_by_36_l143_143673

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end count_perfect_squares_divisible_by_36_l143_143673


namespace Elise_savings_l143_143432

theorem Elise_savings :
  let initial_dollars := 8
  let saved_euros := 11
  let euro_to_dollar := 1.18
  let comic_cost := 2
  let puzzle_pounds := 13
  let pound_to_dollar := 1.38
  let euros_to_dollars := saved_euros * euro_to_dollar
  let total_after_saving := initial_dollars + euros_to_dollars
  let after_comic := total_after_saving - comic_cost
  let pounds_to_dollars := puzzle_pounds * pound_to_dollar
  let final_amount := after_comic - pounds_to_dollars
  final_amount = 1.04 :=
by
  sorry

end Elise_savings_l143_143432


namespace negation_of_p_l143_143329

def p : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x
def not_p : Prop := ∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x

theorem negation_of_p : ¬ p ↔ not_p := by
  sorry

end negation_of_p_l143_143329


namespace find_C_l143_143977

theorem find_C (A B C D E : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) (h5 : E < 10) 
  (h : 4 * (10 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) + 4) = 400000 + (10000 * A + 1000 * B + 100 * C + 10 * D + E)) : 
  C = 2 :=
sorry

end find_C_l143_143977


namespace additional_cards_l143_143940

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) 
  (h1 : total_cards = 160) (h2 : num_decks = 3) (h3 : cards_per_deck = 52) :
  total_cards - (num_decks * cards_per_deck) = 4 :=
by
  sorry

end additional_cards_l143_143940


namespace principal_is_400_l143_143005

-- Define the conditions
def rate_of_interest : ℚ := 12.5
def simple_interest : ℚ := 100
def time_in_years : ℚ := 2

-- Define the formula for principal amount based on the given conditions
def principal_amount (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

-- Prove that the principal amount is 400
theorem principal_is_400 :
  principal_amount simple_interest rate_of_interest time_in_years = 400 := 
by
  simp [principal_amount, simple_interest, rate_of_interest, time_in_years]
  sorry

end principal_is_400_l143_143005


namespace money_raised_is_correct_l143_143474

noncomputable def total_money_raised : ℝ :=
  let ticket_sales := 120 * 2.50 + 80 * 4.50 + 40 * 8.00 + 15 * 14.00
  let donations := 3 * 20.00 + 2 * 55.00 + 75.00 + 95.00 + 150.00
  ticket_sales + donations

theorem money_raised_is_correct :
  total_money_raised = 1680 := by
  sorry

end money_raised_is_correct_l143_143474


namespace geometric_progression_first_term_l143_143479

theorem geometric_progression_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 8) 
  (h2 : a + a * r = 5) : 
  a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) := 
by sorry

end geometric_progression_first_term_l143_143479


namespace exp_fixed_point_l143_143010

theorem exp_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : a^0 = 1 :=
by
  exact one_pow 0

end exp_fixed_point_l143_143010


namespace classics_section_books_l143_143568

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end classics_section_books_l143_143568


namespace find_ending_number_of_range_l143_143752

theorem find_ending_number_of_range :
  ∃ n : ℕ, (∀ avg_200_400 avg_100_n : ℕ,
    avg_200_400 = (200 + 400) / 2 ∧
    avg_100_n = (100 + n) / 2 ∧
    avg_100_n + 150 = avg_200_400) ∧
    n = 200 :=
sorry

end find_ending_number_of_range_l143_143752


namespace problem1_problem2_l143_143921

-- Given conditions
variables (x y : ℝ)

-- Problem 1: Prove that ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = -xy
theorem problem1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = - (x * y) :=
sorry

-- Problem 2: Prove that (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2
theorem problem2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
sorry

end problem1_problem2_l143_143921


namespace problem_statement_l143_143682

theorem problem_statement (p x : ℝ) (h : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) := by
sorry

end problem_statement_l143_143682


namespace correct_proposition_l143_143871

theorem correct_proposition : 
  (¬ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 2 * x_0) ↔ (∀ x : ℝ, x^2 + 1 > 2 * x) := 
sorry

end correct_proposition_l143_143871


namespace find_common_divisor_l143_143051

open Int

theorem find_common_divisor (n : ℕ) (h1 : 2287 % n = 2028 % n)
  (h2 : 2028 % n = 1806 % n) : n = Int.gcd (Int.gcd 259 222) 481 := by
  sorry -- Proof goes here

end find_common_divisor_l143_143051


namespace games_within_division_l143_143584

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division_l143_143584


namespace abs_x_ge_abs_4ax_l143_143525

theorem abs_x_ge_abs_4ax (a : ℝ) (h : ∀ x : ℝ, abs x ≥ 4 * a * x) : abs a ≤ 1 / 4 :=
sorry

end abs_x_ge_abs_4ax_l143_143525


namespace maximum_lambda_l143_143373

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end maximum_lambda_l143_143373


namespace initial_population_of_town_l143_143707

theorem initial_population_of_town 
  (final_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (initial_population : ℝ) 
  (h : final_population = initial_population * (1 + growth_rate) ^ years) : 
  initial_population = 297500 / (1 + 0.07) ^ 10 :=
by
  sorry

end initial_population_of_town_l143_143707


namespace peter_bought_large_glasses_l143_143368

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l143_143368


namespace coordinates_of_point_P_l143_143998

theorem coordinates_of_point_P :
  ∀ (P : ℝ × ℝ), (P.1, P.2) = -1 ∧ (P.2 = -Real.sqrt 3) :=
by
  sorry

end coordinates_of_point_P_l143_143998


namespace seokgi_initial_money_l143_143533

theorem seokgi_initial_money (X : ℝ) (h1 : X / 2 - X / 4 = 1250) : X = 5000 := by
  sorry

end seokgi_initial_money_l143_143533


namespace exact_sequence_a2007_l143_143710

theorem exact_sequence_a2007 (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 0) 
  (exact : ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)) :
  a 2007 = -1 := 
sorry

end exact_sequence_a2007_l143_143710


namespace travel_time_third_to_first_l143_143248

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end travel_time_third_to_first_l143_143248


namespace problem_a_problem_b_l143_143913

theorem problem_a (p : ℕ) (hp : Nat.Prime p) : 
  (∃ x : ℕ, (7^(p-1) - 1) = p * x^2) ↔ p = 3 := 
by
  sorry

theorem problem_b (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ x : ℕ, (11^(p-1) - 1) = p * x^2 := 
by
  sorry

end problem_a_problem_b_l143_143913


namespace area_inside_octagon_outside_semicircles_l143_143740

theorem area_inside_octagon_outside_semicircles :
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area := (1/2) * Real.pi * (s / 2)^2
  let total_semicircle_area := 8 * semicircle_area
  octagon_area - total_semicircle_area = 54 + 24 * Real.sqrt 2 - 9 * Real.pi :=
sorry

end area_inside_octagon_outside_semicircles_l143_143740


namespace total_hamburgers_sold_is_63_l143_143391

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l143_143391


namespace school_dinner_theater_tickets_l143_143087

theorem school_dinner_theater_tickets (x y : ℕ)
  (h1 : x + y = 225)
  (h2 : 6 * x + 9 * y = 1875) :
  x = 50 :=
by
  sorry

end school_dinner_theater_tickets_l143_143087


namespace general_term_arithmetic_sequence_l143_143633

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end general_term_arithmetic_sequence_l143_143633


namespace system1_solution_system2_solution_l143_143465

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 4 * x + 8 * y = 12) (h2 : 3 * x - 2 * y = 5) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : (1/2) * x - (y + 1) / 3 = 1) (h2 : 6 * x + 2 * y = 10) :
  x = 2 ∧ y = -1 := by
  sorry

end system1_solution_system2_solution_l143_143465


namespace exists_positive_integer_solution_l143_143017

theorem exists_positive_integer_solution (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ n / m = ⌊(n^2 : ℝ)^(1/3)⌋ + ⌊(n : ℝ)^(1/2)⌋ + 1 := 
by
  sorry

end exists_positive_integer_solution_l143_143017


namespace bucket_holds_120_ounces_l143_143595

theorem bucket_holds_120_ounces :
  ∀ (fill_buckets remove_buckets baths_per_day ounces_per_week : ℕ),
    fill_buckets = 14 →
    remove_buckets = 3 →
    baths_per_day = 7 →
    ounces_per_week = 9240 →
    baths_per_day * (fill_buckets - remove_buckets) * (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = ounces_per_week →
    (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = 120 :=
by
  intros fill_buckets remove_buckets baths_per_day ounces_per_week Hfill Hremove Hbaths Hounces Hcalc
  sorry

end bucket_holds_120_ounces_l143_143595


namespace cost_of_cookbook_l143_143196

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def amount_saved : ℕ := 8
def amount_needed : ℕ := 29

theorem cost_of_cookbook :
  let total_cost := amount_saved + amount_needed
  let accounted_cost := cost_of_dictionary + cost_of_dinosaur_book
  total_cost - accounted_cost = 7 :=
by
  sorry

end cost_of_cookbook_l143_143196


namespace solutions_to_shifted_parabola_l143_143214

noncomputable def solution_equation := ∀ (a b : ℝ) (m : ℝ) (x : ℝ),
  (a ≠ 0) →
  ((a * (x + m) ^ 2 + b = 0) → (x = 2 ∨ x = -1)) →
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0))

-- We'll leave the proof for this theorem as 'sorry'
theorem solutions_to_shifted_parabola (a b m : ℝ) (h : a ≠ 0)
  (h1 : ∀ (x : ℝ), a * (x + m) ^ 2 + b = 0 → (x = 2 ∨ x = -1)) 
  (x : ℝ) : 
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0)) := sorry

end solutions_to_shifted_parabola_l143_143214


namespace integer_solutions_of_inequality_l143_143202

theorem integer_solutions_of_inequality :
  {x : ℤ | 3 ≤ 5 - 2 * x ∧ 5 - 2 * x ≤ 9} = {-2, -1, 0, 1} :=
by
  sorry

end integer_solutions_of_inequality_l143_143202


namespace parametric_to_cartesian_l143_143240

theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : x = 5 + 3 * t) (h2 : y = 10 - 4 * t) : 4 * x + 3 * y = 50 :=
by sorry

end parametric_to_cartesian_l143_143240


namespace M_gt_N_l143_143372

variable (a : ℝ)

def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

theorem M_gt_N : M a > N a := by
  sorry

end M_gt_N_l143_143372


namespace inequality_holds_for_positive_y_l143_143519

theorem inequality_holds_for_positive_y (y : ℝ) (hy : y > 0) : y^2 ≥ 2 * y - 1 :=
by
  sorry

end inequality_holds_for_positive_y_l143_143519


namespace bananas_in_collection_l143_143843

theorem bananas_in_collection
  (groups : ℕ)
  (bananas_per_group : ℕ)
  (h1 : groups = 11)
  (h2 : bananas_per_group = 37) :
  (groups * bananas_per_group) = 407 :=
by sorry

end bananas_in_collection_l143_143843


namespace max_regions_quadratic_trinomials_l143_143489

theorem max_regions_quadratic_trinomials (a b c : Fin 100 → ℝ) :
  ∃ R, (∀ (n : ℕ), n ≤ 100 → R = n^2 + 1) → R = 10001 := 
  sorry

end max_regions_quadratic_trinomials_l143_143489


namespace arithmetic_sequence_sum_l143_143014

variable {a : ℕ → ℝ}

noncomputable def sum_of_first_ten_terms (a : ℕ → ℝ) : ℝ :=
  (10 / 2) * (a 1 + a 10)

theorem arithmetic_sequence_sum (h : a 5 + a 6 = 28) :
  sum_of_first_ten_terms a = 140 :=
by
  sorry

end arithmetic_sequence_sum_l143_143014


namespace dismissed_cases_l143_143866

theorem dismissed_cases (total_cases : Int) (X : Int)
  (total_cases_eq : total_cases = 17)
  (remaining_cases_eq : X = (2 * X / 3) + 1 + 4) :
  total_cases - X = 2 :=
by
  -- Placeholder for the proof
  sorry

end dismissed_cases_l143_143866


namespace base_conversion_403_base_6_eq_223_base_8_l143_143764

theorem base_conversion_403_base_6_eq_223_base_8 :
  (6^2 * 4 + 6^1 * 0 + 6^0 * 3 : ℕ) = (8^2 * 2 + 8^1 * 2 + 8^0 * 3 : ℕ) :=
by
  sorry

end base_conversion_403_base_6_eq_223_base_8_l143_143764


namespace exercise_l143_143330

noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem exercise : f (g 3) = 1457 := by
  sorry

end exercise_l143_143330


namespace lottery_probability_correct_l143_143965

def number_of_winnerballs_ways : ℕ := Nat.choose 50 6

def probability_megaBall : ℚ := 1 / 30

def probability_winnerBalls : ℚ := 1 / number_of_winnerballs_ways

def combined_probability : ℚ := probability_megaBall * probability_winnerBalls

theorem lottery_probability_correct : combined_probability = 1 / 476721000 := by
  sorry

end lottery_probability_correct_l143_143965


namespace age_of_B_l143_143318

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l143_143318


namespace stamps_in_album_l143_143942

theorem stamps_in_album (n : ℕ) : 
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ 
  n % 6 = 5 ∧ n % 7 = 6 ∧ n % 8 = 7 ∧ n % 9 = 8 ∧ 
  n % 10 = 9 ∧ n < 3000 → n = 2519 :=
by
  sorry

end stamps_in_album_l143_143942


namespace total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l143_143171

-- Definition of properties required as per the given conditions
def is_fortunate_number (abcd ab cd : ℕ) : Prop :=
  abcd = 100 * ab + cd ∧
  ab ≠ cd ∧
  ab ∣ cd ∧
  cd ∣ abcd

-- Total number of fortunate numbers is 65
theorem total_fortunate_numbers_is_65 : 
  ∃ n : ℕ, n = 65 ∧ 
  ∀(abcd ab cd : ℕ), is_fortunate_number abcd ab cd → n = 65 :=
sorry

-- Largest odd fortunate number is 1995
theorem largest_odd_fortunate_number_is_1995 : 
  ∃ abcd : ℕ, abcd = 1995 ∧ 
  ∀(abcd' ab cd : ℕ), is_fortunate_number abcd' ab cd ∧ cd % 2 = 1 → abcd = 1995 :=
sorry

end total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l143_143171


namespace expression_equality_l143_143513

theorem expression_equality : 1 + 2 / (3 + 4 / 5) = 29 / 19 := by
  sorry

end expression_equality_l143_143513


namespace eggs_left_after_cupcakes_l143_143162

-- Definitions derived from the given conditions
def dozen := 12
def initial_eggs := 3 * dozen
def crepes_fraction := 1 / 4
def cupcakes_fraction := 2 / 3

theorem eggs_left_after_cupcakes :
  let eggs_after_crepes := initial_eggs - crepes_fraction * initial_eggs;
  let eggs_after_cupcakes := eggs_after_crepes - cupcakes_fraction * eggs_after_crepes;
  eggs_after_cupcakes = 9 := sorry

end eggs_left_after_cupcakes_l143_143162


namespace percent_area_square_in_rectangle_l143_143888

theorem percent_area_square_in_rectangle
  (s : ℝ) 
  (w : ℝ) 
  (l : ℝ)
  (h1 : w = 3 * s) 
  (h2 : l = (9 / 2) * s) 
  : (s^2 / (l * w)) * 100 = 7.41 :=
by
  sorry

end percent_area_square_in_rectangle_l143_143888


namespace abs_sum_neq_3_nor_1_l143_143225

theorem abs_sum_neq_3_nor_1 (a b : ℤ) (h₁ : |a| = 3) (h₂ : |b| = 1) : (|a + b| ≠ 3) ∧ (|a + b| ≠ 1) := sorry

end abs_sum_neq_3_nor_1_l143_143225


namespace cos_half_pi_plus_alpha_l143_143664

open Real

noncomputable def alpha : ℝ := sorry

theorem cos_half_pi_plus_alpha :
  let a := (1 / 3, tan alpha)
  let b := (cos alpha, 1)
  ((1 / 3) / (cos alpha) = (tan alpha) / 1) →
  cos (pi / 2 + alpha) = -1 / 3 :=
by
  intros
  sorry

end cos_half_pi_plus_alpha_l143_143664


namespace egg_problem_l143_143390

theorem egg_problem :
  ∃ (N F E : ℕ), N + F + E = 100 ∧ 5 * N + F + E / 2 = 100 ∧ (N = F ∨ N = E ∨ F = E) ∧ N = 10 ∧ F = 10 ∧ E = 80 :=
by
  sorry

end egg_problem_l143_143390


namespace complex_power_difference_l143_143281

theorem complex_power_difference (i : ℂ) (hi : i^2 = -1) : (1 + 2 * i)^8 - (1 - 2 * i)^8 = 672 * i := 
by
  sorry

end complex_power_difference_l143_143281


namespace distance_to_cheaper_gas_station_l143_143947

-- Define the conditions
def miles_per_gallon : ℕ := 3
def initial_gallons : ℕ := 12
def additional_gallons : ℕ := 18

-- Define the question and proof statement
theorem distance_to_cheaper_gas_station : 
  (initial_gallons + additional_gallons) * miles_per_gallon = 90 := by
  sorry

end distance_to_cheaper_gas_station_l143_143947


namespace triangle_side_length_l143_143769

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end triangle_side_length_l143_143769


namespace find_value_of_function_l143_143637

theorem find_value_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 :=
sorry

end find_value_of_function_l143_143637


namespace find_functions_l143_143076

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem find_functions (f : ℝ × ℝ → ℝ) :
  (is_non_decreasing (λ x => f (0, x))) →
  (∀ x y, f (x, y) = f (y, x)) →
  (∀ x y z, (f (x, y) - f (y, z)) * (f (y, z) - f (z, x)) * (f (z, x) - f (x, y)) = 0) →
  (∀ x y a, f (x + a, y + a) = f (x, y) + a) →
  (∃ a : ℝ, (∀ x y, f (x, y) = a + min x y) ∨ (∀ x y, f (x, y) = a + max x y)) :=
  by sorry

end find_functions_l143_143076


namespace solve_for_x_l143_143430

theorem solve_for_x (x : ℝ) (h : 2 * (1/x + 3/x / 6/x) - 1/x = 1.5) : x = 2 := 
by 
  sorry

end solve_for_x_l143_143430


namespace problem_lean_l143_143844

variable (α : ℝ)

-- Given condition
axiom given_cond : (1 + Real.sin α) * (1 - Real.cos α) = 1

-- Proof to be proven
theorem problem_lean : (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end problem_lean_l143_143844


namespace correct_statements_for_function_l143_143514

-- Definitions and the problem statement
def f (x b c : ℝ) := x * |x| + b * x + c

theorem correct_statements_for_function (b c : ℝ) :
  (c = 0 → ∀ x, f x b c = -f (-x) b c) ∧
  (b = 0 ∧ c > 0 → ∀ x, f x b c = 0 → x = 0) ∧
  (∀ x, f x b c = f (-x) b (-c)) :=
sorry

end correct_statements_for_function_l143_143514


namespace coffee_students_l143_143300

variable (S : ℝ) -- Total number of students
variable (T : ℝ) -- Number of students who chose tea
variable (C : ℝ) -- Number of students who chose coffee

-- Given conditions
axiom h1 : 0.4 * S = 80   -- 40% of the students chose tea
axiom h2 : T = 80         -- Number of students who chose tea is 80
axiom h3 : 0.3 * S = C    -- 30% of the students chose coffee

-- Prove that the number of students who chose coffee is 60
theorem coffee_students : C = 60 := by
  sorry

end coffee_students_l143_143300


namespace inequality_problem_l143_143786

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := Real.logb (1 / 2) (1 / 3)

theorem inequality_problem :
  c > a ∧ a > b := by
  sorry

end inequality_problem_l143_143786


namespace fill_two_thirds_of_bucket_time_l143_143477

theorem fill_two_thirds_of_bucket_time (fill_entire_bucket_time : ℝ) (h : fill_entire_bucket_time = 3) : (2 / 3) * fill_entire_bucket_time = 2 :=
by 
  sorry

end fill_two_thirds_of_bucket_time_l143_143477


namespace voucher_placement_l143_143433

/-- A company wants to popularize the sweets they market by hiding prize vouchers in some of the boxes.
The management believes the promotion is effective and the cost is bearable if a customer who buys 10 boxes has approximately a 50% chance of finding at least one voucher.
We aim to determine how often vouchers should be placed in the boxes to meet this requirement. -/
theorem voucher_placement (n : ℕ) (h_positive : n > 0) :
  (1 - (1 - 1/n)^10) ≥ 1/2 → n ≤ 15 :=
sorry

end voucher_placement_l143_143433


namespace certain_number_value_l143_143063

theorem certain_number_value (x : ℝ) (certain_number : ℝ) 
  (h1 : x = 0.25) 
  (h2 : 625^(-x) + 25^(-2 * x) + certain_number^(-4 * x) = 11) : 
  certain_number = 5 / 53 := 
sorry

end certain_number_value_l143_143063


namespace johns_subtraction_l143_143800

theorem johns_subtraction : 
  ∀ (a : ℕ), 
  a = 40 → 
  (a - 1)^2 = a^2 - 79 := 
by 
  -- The proof is omitted as per instruction
  sorry

end johns_subtraction_l143_143800


namespace sides_of_regular_polygon_with_20_diagonals_l143_143785

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l143_143785


namespace quadratic_polynomial_exists_l143_143678

theorem quadratic_polynomial_exists (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ p : ℝ → ℝ, (∀ x, p x = (a^2 + ab + b^2 + ac + bc + c^2) * x^2 
                   - (a + b) * (b + c) * (a + c) * x 
                   + abc * (a + b + c))
              ∧ p a = a^4 
              ∧ p b = b^4 
              ∧ p c = c^4 := 
sorry

end quadratic_polynomial_exists_l143_143678


namespace ratio_of_x_y_l143_143663

theorem ratio_of_x_y (x y : ℝ) (h : x + y = 3 * (x - y)) : x / y = 2 :=
by
  sorry

end ratio_of_x_y_l143_143663


namespace fraction_addition_l143_143378

theorem fraction_addition (x y : ℚ) (h : x / y = 2 / 3) : (x + y) / y = 5 / 3 := 
by 
  sorry

end fraction_addition_l143_143378


namespace problem_y_equals_x_squared_plus_x_minus_6_l143_143256

theorem problem_y_equals_x_squared_plus_x_minus_6 (x y : ℝ) :
  (y = x^2 + x - 6 ∧ x = 0 → y = -6) ∧ 
  (y = 0 → x = -3 ∨ x = 2) :=
by
  sorry

end problem_y_equals_x_squared_plus_x_minus_6_l143_143256


namespace flowers_brought_at_dawn_l143_143884

theorem flowers_brought_at_dawn (F : ℕ) 
  (h1 : (3 / 5) * F = 180)
  (h2 :  (2 / 5) * F + (F - (3 / 5) * F) = 180) : 
  F = 300 := 
by
  sorry

end flowers_brought_at_dawn_l143_143884


namespace loaves_of_bread_l143_143699

variable (B : ℕ) -- Number of loaves of bread Erik bought
variable (total_money : ℕ := 86) -- Money given to Erik
variable (money_left : ℕ := 59) -- Money left after purchase
variable (cost_bread : ℕ := 3) -- Cost of each loaf of bread
variable (cost_oj : ℕ := 6) -- Cost of each carton of orange juice
variable (num_oj : ℕ := 3) -- Number of cartons of orange juice bought

theorem loaves_of_bread (h1 : total_money - money_left = num_oj * cost_oj + B * cost_bread) : B = 3 := 
by sorry

end loaves_of_bread_l143_143699


namespace vertical_complementary_perpendicular_l143_143357

theorem vertical_complementary_perpendicular (α β : ℝ) (l1 l2 : ℝ) :
  (α = β ∧ α + β = 90) ∧ l1 = l2 -> l1 + l2 = 90 := by
  sorry

end vertical_complementary_perpendicular_l143_143357


namespace ant_paths_l143_143527

theorem ant_paths (n m : ℕ) : 
  ∃ paths : ℕ, paths = Nat.choose (n + m) m := sorry

end ant_paths_l143_143527


namespace sequence_a_general_term_sequence_b_sum_of_first_n_terms_l143_143731

variable {n : ℕ}

def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℕ := 3^(n-1) + 2 * n

def T (n : ℕ) : ℕ := (3^n - 1) / 2 + n^2 + n

theorem sequence_a_general_term :
  (∀ n, a n = 2 * n) :=
by
  intro n
  sorry

theorem sequence_b_sum_of_first_n_terms :
  (∀ n, T n = (3^n - 1) / 2 + n^2 + n) :=
by
  intro n
  sorry

end sequence_a_general_term_sequence_b_sum_of_first_n_terms_l143_143731


namespace zeros_at_end_of_quotient_factorial_l143_143344

def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem zeros_at_end_of_quotient_factorial :
  count_factors_of_five 2018 - count_factors_of_five 30 - count_factors_of_five 11 = 493 :=
by
  sorry

end zeros_at_end_of_quotient_factorial_l143_143344


namespace expression_value_l143_143560

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem expression_value :
  let numerator := factorial 10
  let denominator := (1 + 2) * (3 + 4) * (5 + 6) * (7 + 8) * (9 + 10)
  numerator / denominator = 660 := by
  sorry

end expression_value_l143_143560


namespace sufficient_but_not_necessary_condition_l143_143856

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≤ 0) : 
  a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_l143_143856


namespace find_angle_beta_l143_143286

theorem find_angle_beta (α β : ℝ)
  (h1 : (π / 2) < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19)
  (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := 
sorry

end find_angle_beta_l143_143286


namespace total_cards_1750_l143_143757

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l143_143757


namespace johnny_hours_second_job_l143_143670

theorem johnny_hours_second_job (x : ℕ) (h_eq : 5 * (69 + 10 * x) = 445) : x = 2 :=
by 
  -- The proof will go here, but we skip it as per the instructions
  sorry

end johnny_hours_second_job_l143_143670


namespace most_precise_value_l143_143667

def D := 3.27645
def error := 0.00518
def D_upper := D + error
def D_lower := D - error
def rounded_D_upper := Float.round (D_upper * 10) / 10
def rounded_D_lower := Float.round (D_lower * 10) / 10

theorem most_precise_value :
  rounded_D_upper = 3.3 ∧ rounded_D_lower = 3.3 → rounded_D_upper = 3.3 :=
by sorry

end most_precise_value_l143_143667


namespace correct_operation_l143_143904

theorem correct_operation {a : ℝ} : (a ^ 6 / a ^ 2 = a ^ 4) :=
by sorry

end correct_operation_l143_143904


namespace probabilities_equal_l143_143504

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l143_143504


namespace highest_prob_red_ball_l143_143053

-- Definitions
def total_red_balls : ℕ := 5
def total_white_balls : ℕ := 12
def total_balls : ℕ := total_red_balls + total_white_balls

-- Condition that neither bag is empty
def neither_bag_empty (r1 w1 r2 w2 : ℕ) : Prop :=
  (r1 + w1 > 0) ∧ (r2 + w2 > 0)

-- Define the probability of drawing a red ball from a bag
def prob_red (r w : ℕ) : ℚ :=
  if (r + w) = 0 then 0 else r / (r + w)

-- Define the overall probability if choosing either bag with equal probability
def overall_prob_red (r1 w1 r2 w2 : ℕ) : ℚ :=
  (prob_red r1 w1 + prob_red r2 w2) / 2

-- Problem statement to be proved
theorem highest_prob_red_ball :
  ∃ (r1 w1 r2 w2 : ℕ),
    neither_bag_empty r1 w1 r2 w2 ∧
    r1 + r2 = total_red_balls ∧
    w1 + w2 = total_white_balls ∧
    (overall_prob_red r1 w1 r2 w2 = 0.625) :=
sorry

end highest_prob_red_ball_l143_143053


namespace find_number_eq_150_l143_143719

variable {x : ℝ}

theorem find_number_eq_150 (h : 0.60 * x - 40 = 50) : x = 150 :=
sorry

end find_number_eq_150_l143_143719


namespace value_of_m_l143_143511

-- Definitions of the conditions
def base6_num (m : ℕ) : ℕ := 2 + m * 6^2
def dec_num (d : ℕ) := d = 146

-- Theorem to prove
theorem value_of_m (m : ℕ) (h1 : base6_num m = 146) : m = 4 := 
sorry

end value_of_m_l143_143511


namespace work_completion_rate_l143_143083

theorem work_completion_rate (A B D : ℝ) (W : ℝ) (hB : B = W / 9) (hA : A = W / 10) (hD : D = 90 / 19) : 
  (A + B) * D = W := 
by 
  sorry

end work_completion_rate_l143_143083


namespace total_cakes_served_l143_143106

-- Conditions
def cakes_lunch : Nat := 6
def cakes_dinner : Nat := 9

-- Statement of the problem
theorem total_cakes_served : cakes_lunch + cakes_dinner = 15 := 
by
  sorry

end total_cakes_served_l143_143106


namespace gas_pressure_inversely_proportional_l143_143294

theorem gas_pressure_inversely_proportional :
  ∀ (p v : ℝ), (p * v = 27.2) → (8 * 3.4 = 27.2) → (v = 6.8) → p = 4 :=
by
  intros p v h1 h2 h3
  have h4 : 27.2 = 8 * 3.4 := by sorry
  have h5 : p * 6.8 = 27.2 := by sorry
  exact sorry

end gas_pressure_inversely_proportional_l143_143294


namespace us_more_than_canada_l143_143078

/-- Define the total number of supermarkets -/
def total_supermarkets : ℕ := 84

/-- Define the number of supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- Define the number of supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- The proof problem: Prove that there are 14 more supermarkets in the US than in Canada -/
theorem us_more_than_canada : us_supermarkets - canada_supermarkets = 14 := by
  sorry

end us_more_than_canada_l143_143078


namespace interest_years_l143_143481

theorem interest_years (P : ℝ) (R : ℝ) (N : ℝ) (H1 : P = 2400) (H2 : (P * (R + 1) * N) / 100 - (P * R * N) / 100 = 72) : N = 3 :=
by
  -- Proof can be filled in here
  sorry

end interest_years_l143_143481


namespace range_of_a_l143_143943

theorem range_of_a (a : ℝ) (h : Real.sqrt ((2 * a - 1)^2) = 1 - 2 * a) : a ≤ 1 / 2 :=
sorry

end range_of_a_l143_143943


namespace ticTacToe_CarlWins_l143_143827

def ticTacToeBoard := Fin 3 × Fin 3

noncomputable def countConfigurations : Nat := sorry

theorem ticTacToe_CarlWins :
  countConfigurations = 148 :=
sorry

end ticTacToe_CarlWins_l143_143827


namespace problem_l143_143061

variables (x : ℝ)

-- Define the condition
def condition (x : ℝ) : Prop :=
  0.3 * (0.2 * x) = 24

-- Define the target statement
def target (x : ℝ) : Prop :=
  0.2 * (0.3 * x) = 24

-- The theorem we want to prove
theorem problem (x : ℝ) (h : condition x) : target x :=
sorry

end problem_l143_143061


namespace Q_div_P_l143_143540

theorem Q_div_P (P Q : ℤ) (h : ∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 →
  P / (x + 7) + Q / (x * (x - 6)) = (x^2 - x + 15) / (x^3 + x^2 - 42 * x)) :
  Q / P = 7 :=
sorry

end Q_div_P_l143_143540


namespace pairs_satisfying_equation_l143_143004

theorem pairs_satisfying_equation (a b : ℝ) : 
  (∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ ∃ k : ℤ, a = k ∧ b = k) := 
by
  sorry

end pairs_satisfying_equation_l143_143004


namespace find_x_l143_143495

-- Define the condition from the problem statement
def condition1 (x : ℝ) : Prop := 70 = 0.60 * x + 22

-- Translate the question to the Lean statement form
theorem find_x (x : ℝ) (h : condition1 x) : x = 80 :=
by {
  sorry
}

end find_x_l143_143495


namespace true_inverse_propositions_count_l143_143152

-- Let P1, P2, P3, P4 denote the original propositions
def P1 := "Supplementary angles are congruent, and two lines are parallel."
def P2 := "If |a| = |b|, then a = b."
def P3 := "Right angles are congruent."
def P4 := "Congruent angles are vertical angles."

-- Let IP1, IP2, IP3, IP4 denote the inverse propositions
def IP1 := "Two lines are parallel, and supplementary angles are congruent."
def IP2 := "If a = b, then |a| = |b|."
def IP3 := "Congruent angles are right angles."
def IP4 := "Vertical angles are congruent angles."

-- Counting the number of true inverse propositions
def countTrueInversePropositions : ℕ :=
  let p1_inverse_true := true  -- IP1 is true
  let p2_inverse_true := true  -- IP2 is true
  let p3_inverse_true := false -- IP3 is false
  let p4_inverse_true := true  -- IP4 is true
  [p1_inverse_true, p2_inverse_true, p4_inverse_true].length

-- The statement to be proved
theorem true_inverse_propositions_count : countTrueInversePropositions = 3 := by
  sorry

end true_inverse_propositions_count_l143_143152


namespace correct_survey_option_l143_143307

-- Definitions for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Predicate that checks if an option is suitable for a comprehensive survey method
def suitable_for_comprehensive_survey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => false
  | SurveyOption.B => false
  | SurveyOption.C => false
  | SurveyOption.D => true

-- Theorem statement
theorem correct_survey_option : suitable_for_comprehensive_survey SurveyOption.D := 
  by sorry

end correct_survey_option_l143_143307


namespace cannot_assemble_highlighted_shape_l143_143234

-- Define the rhombus shape with its properties
structure Rhombus :=
  (white_triangle gray_triangle : Prop)

-- Define the assembly condition
def can_rotate (shape : Rhombus) : Prop := sorry

-- Define the specific shape highlighted that Petya cannot form
def highlighted_shape : Prop := sorry

-- The statement we need to prove
theorem cannot_assemble_highlighted_shape (shape : Rhombus) 
  (h_rotate : can_rotate shape)
  (h_highlight : highlighted_shape) : false :=
by sorry

end cannot_assemble_highlighted_shape_l143_143234


namespace calc_expr_l143_143643

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l143_143643


namespace train_speed_l143_143422

-- Defining the lengths and time
def length_train : ℕ := 100
def length_bridge : ℕ := 300
def time_crossing : ℕ := 15

-- Defining the total distance
def total_distance : ℕ := length_train + length_bridge

-- Proving the speed of the train
theorem train_speed : (total_distance / time_crossing : ℚ) = 26.67 := by
  sorry

end train_speed_l143_143422


namespace tower_height_count_l143_143060

theorem tower_height_count (bricks : ℕ) (height1 height2 height3 : ℕ) :
  height1 = 3 → height2 = 11 → height3 = 18 → bricks = 100 →
  (∃ (h : ℕ),  h = 1404) :=
by
  sorry

end tower_height_count_l143_143060


namespace market_value_decrease_l143_143238

noncomputable def percentage_decrease_each_year : ℝ :=
  let original_value := 8000
  let value_after_two_years := 3200
  let p := 1 - (value_after_two_years / original_value)^(1 / 2)
  p * 100

theorem market_value_decrease :
  let p := percentage_decrease_each_year
  abs (p - 36.75) < 0.01 :=
by
  sorry

end market_value_decrease_l143_143238


namespace vector_subtraction_l143_143820

variables (a b : ℝ × ℝ)

-- Definitions based on conditions
def vector_a : ℝ × ℝ := (1, -2)
def m : ℝ := 2
def vector_b : ℝ × ℝ := (4, m)

-- Prove given question equals answer
theorem vector_subtraction :
  vector_a = (1, -2) →
  vector_b = (4, m) →
  (1 * 4 + (-2) * m = 0) →
  5 • vector_a - vector_b = (1, -12) := by
  intros h1 h2 h3
  sorry

end vector_subtraction_l143_143820


namespace anderson_family_seating_l143_143939

def anderson_family_seating_arrangements : Prop :=
  ∃ (family : Fin 5 → String),
    (family 0 = "Mr. Anderson" ∨ family 0 = "Mrs. Anderson") ∧
    (∀ (i : Fin 5), i ≠ 0 → family i ≠ family 0) ∧
    family 1 ≠ family 0 ∧ (family 1 = "Mrs. Anderson" ∨ family 1 = "Child 1" ∨ family 1 = "Child 2") ∧
    family 2 = "Child 3" ∧
    (family 3 ≠ family 0 ∧ family 3 ≠ family 1 ∧ family 3 ≠ family 2) ∧
    (family 4 ≠ family 0 ∧ family 4 ≠ family 1 ∧ family 4 ≠ family 2 ∧ family 4 ≠ family 3) ∧
    (family 3 = "Child 1" ∨ family 3 = "Child 2") ∧
    (family 4 = "Child 1" ∨ family 4 = "Child 2") ∧
    family 3 ≠ family 4 → 
    (2 * 3 * 2 = 12)

theorem anderson_family_seating : anderson_family_seating_arrangements := 
  sorry

end anderson_family_seating_l143_143939


namespace Sara_lunch_bill_l143_143429

theorem Sara_lunch_bill :
  let hotdog := 5.36
  let salad := 5.10
  let drink := 2.50
  let side_item := 3.75
  hotdog + salad + drink + side_item = 16.71 :=
by
  sorry

end Sara_lunch_bill_l143_143429


namespace percentage_of_y_l143_143516

theorem percentage_of_y (x y : ℝ) (h1 : x = 4 * y) (h2 : 0.80 * x = (P / 100) * y) : P = 320 :=
by
  -- Proof goes here
  sorry

end percentage_of_y_l143_143516


namespace Cathy_total_money_l143_143425

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l143_143425


namespace find_a8_l143_143850

variable (a : ℕ → ℤ)

axiom h1 : ∀ n : ℕ, 2 * a n + a (n + 1) = 0
axiom h2 : a 3 = -2

theorem find_a8 : a 8 = 64 := by
  sorry

end find_a8_l143_143850


namespace girls_count_l143_143571

variable (B G : ℕ)

theorem girls_count (h1: B = 387) (h2: G = (B + (54 * B) / 100)) : G = 596 := 
by 
  sorry

end girls_count_l143_143571


namespace smallest_angle_pentagon_l143_143766

theorem smallest_angle_pentagon (x : ℝ) (h : 16 * x = 540) : 2 * x = 67.5 := 
by 
  sorry

end smallest_angle_pentagon_l143_143766


namespace find_N_is_20_l143_143715

theorem find_N_is_20 : ∃ (N : ℤ), ∃ (u v : ℤ), (N + 5 = u ^ 2) ∧ (N - 11 = v ^ 2) ∧ (N = 20) :=
by
  sorry

end find_N_is_20_l143_143715


namespace linear_function_difference_l143_143506

variable (g : ℝ → ℝ)
variable (h_linear : ∀ x y, g (x + y) = g x + g y)
variable (h_value : g 8 - g 4 = 16)

theorem linear_function_difference : g 16 - g 4 = 48 := by
  sorry

end linear_function_difference_l143_143506


namespace remainder_of_product_l143_143134

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) (h1 : a % c = 1) (h2 : b % c = 2) : (a * b) % c = 2 :=
by
  sorry

end remainder_of_product_l143_143134


namespace negation_of_universal_l143_143845

theorem negation_of_universal :
  (¬ (∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1)) ↔ 
  (∃ k : ℝ, ¬ ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1) :=
by
  sorry

end negation_of_universal_l143_143845


namespace jimmy_cards_left_l143_143190

theorem jimmy_cards_left :
  ∀ (initial_cards jimmy_cards bob_cards mary_cards : ℕ),
    initial_cards = 18 →
    bob_cards = 3 →
    mary_cards = 2 * bob_cards →
    jimmy_cards = initial_cards - bob_cards - mary_cards →
    jimmy_cards = 9 := 
by
  intros initial_cards jimmy_cards bob_cards mary_cards h1 h2 h3 h4
  sorry

end jimmy_cards_left_l143_143190


namespace proof_of_truth_values_l143_143576

open Classical

variables (x : ℝ)

-- Original proposition: If x = 1, then x^2 = 1.
def original_proposition : Prop := (x = 1) → (x^2 = 1)

-- Converse of the original proposition: If x^2 = 1, then x = 1.
def converse_proposition : Prop := (x^2 = 1) → (x = 1)

-- Inverse of the original proposition: If x ≠ 1, then x^2 ≠ 1.
def inverse_proposition : Prop := (x ≠ 1) → (x^2 ≠ 1)

-- Contrapositive of the original proposition: If x^2 ≠ 1, then x ≠ 1.
def contrapositive_proposition : Prop := (x^2 ≠ 1) → (x ≠ 1)

-- Negation of the original proposition: If x = 1, then x^2 ≠ 1.
def negation_proposition : Prop := (x = 1) → (x^2 ≠ 1)

theorem proof_of_truth_values :
  (original_proposition x) ∧
  (converse_proposition x = False) ∧
  (inverse_proposition x = False) ∧
  (contrapositive_proposition x) ∧
  (negation_proposition x = False) := by
  sorry

end proof_of_truth_values_l143_143576


namespace sum_sin_double_angles_eq_l143_143440

theorem sum_sin_double_angles_eq (
  α β γ : ℝ
) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

end sum_sin_double_angles_eq_l143_143440


namespace value_of_a_minus_b_l143_143402

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 :=
by
  sorry

end value_of_a_minus_b_l143_143402


namespace isosceles_triangle_circum_incenter_distance_l143_143224

variable {R r d : ℝ}

/-- The distance \(d\) between the centers of the circumscribed circle and the inscribed circle of an isosceles triangle satisfies \(d = \sqrt{R(R - 2r)}\) --/
theorem isosceles_triangle_circum_incenter_distance (hR : 0 < R) (hr : 0 < r) 
  (hIso : ∃ (A B C : ℝ × ℝ), (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (dist A B = dist A C)) 
  : d = Real.sqrt (R * (R - 2 * r)) :=
sorry

end isosceles_triangle_circum_incenter_distance_l143_143224


namespace penny_makes_from_cheesecakes_l143_143562

-- Definitions based on the conditions
def slices_per_pie : ℕ := 6
def cost_per_slice : ℕ := 7
def pies_sold : ℕ := 7

-- The mathematical equivalent proof problem
theorem penny_makes_from_cheesecakes : slices_per_pie * cost_per_slice * pies_sold = 294 := by
  sorry

end penny_makes_from_cheesecakes_l143_143562


namespace laura_owes_amount_l143_143860

def principal : ℝ := 35
def rate : ℝ := 0.04
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount : ℝ := principal + interest

theorem laura_owes_amount :
  total_amount = 36.40 := by
  sorry

end laura_owes_amount_l143_143860


namespace find_b_l143_143799

theorem find_b (a b : ℤ) (h1 : 0 ≤ a) (h2 : a < 2^2008) (h3 : 0 ≤ b) (h4 : b < 8) (h5 : 7 * (a + 2^2008 * b) % 2^2011 = 1) :
  b = 3 :=
sorry

end find_b_l143_143799


namespace problem_solution_l143_143830

noncomputable def negThreePower25 : Real := (-3) ^ 25
noncomputable def twoPowerExpression : Real := 2 ^ (4^2 + 5^2 - 7^2)
noncomputable def threeCubed : Real := 3^3

theorem problem_solution :
  negThreePower25 + twoPowerExpression + threeCubed = -3^25 + 27 + (1 / 256) :=
by
  -- proof omitted
  sorry

end problem_solution_l143_143830


namespace geometric_progression_product_l143_143981

theorem geometric_progression_product (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  (h1 : a 3 = a1 * r^2)
  (h2 : a 10 = a1 * r^9)
  (h3 : a1 * r^2 + a1 * r^9 = 3)
  (h4 : a1^2 * r^11 = -5) :
  a 5 * a 8 = -5 :=
by
  sorry

end geometric_progression_product_l143_143981


namespace sqrt_defined_iff_ge_neg1_l143_143338

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end sqrt_defined_iff_ge_neg1_l143_143338


namespace area_of_shaded_region_l143_143369

theorem area_of_shaded_region
  (r_large : ℝ) (r_small : ℝ) (n_small : ℕ) (π : ℝ)
  (A_large : ℝ) (A_small : ℝ) (A_7_small : ℝ) (A_shaded : ℝ)
  (h1 : r_large = 20)
  (h2 : r_small = 10)
  (h3 : n_small = 7)
  (h4 : π = 3.14)
  (h5 : A_large = π * r_large^2)
  (h6 : A_small = π * r_small^2)
  (h7 : A_7_small = n_small * A_small)
  (h8 : A_shaded = A_large - A_7_small) :
  A_shaded = 942 :=
by
  sorry

end area_of_shaded_region_l143_143369


namespace derivative_sum_l143_143825

theorem derivative_sum (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, deriv f x = f' x)
  (h : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 :=
sorry

end derivative_sum_l143_143825


namespace ratio_equiv_solve_x_l143_143395

theorem ratio_equiv_solve_x (x : ℕ) (h : 3 / 12 = 3 / x) : x = 12 :=
sorry

end ratio_equiv_solve_x_l143_143395


namespace cube_volume_l143_143609

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l143_143609


namespace positional_relationship_l143_143638

-- Definitions of the lines l1 and l2
def l1 (m x y : ℝ) : Prop := (m + 3) * x + 5 * y = 5 - 3 * m
def l2 (m x y : ℝ) : Prop := 2 * x + (m + 6) * y = 8

theorem positional_relationship (m : ℝ) :
  (∃ x y : ℝ, l1 m x y ∧ l2 m x y) ∨ (∀ x y : ℝ, l1 m x y ↔ l2 m x y) ∨
  ¬(∃ x y : ℝ, l1 m x y ∨ l2 m x y) :=
sorry

end positional_relationship_l143_143638


namespace time_spent_driving_l143_143271

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end time_spent_driving_l143_143271


namespace inequality_proof_l143_143445

variable (x y z : ℝ)

theorem inequality_proof (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
  ≥ Real.sqrt (3 / 2 * (x + y + z)) :=
sorry

end inequality_proof_l143_143445


namespace find_x_when_y_equals_2_l143_143197

theorem find_x_when_y_equals_2 :
  ∀ (y x k : ℝ),
  (y * (Real.sqrt x + 1) = k) →
  (y = 5 → x = 1 → k = 10) →
  (y = 2 → x = 16) := by
  intros y x k h_eq h_initial h_final
  sorry

end find_x_when_y_equals_2_l143_143197


namespace expand_product_l143_143954

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 :=
by sorry

end expand_product_l143_143954


namespace Gandalf_reachability_l143_143602

theorem Gandalf_reachability (n : ℕ) (h : n ≥ 1) :
  ∃ (m : ℕ), m = 1 :=
sorry

end Gandalf_reachability_l143_143602


namespace range_of_a_l143_143298

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → abs (2 * a - 1) ≤ abs (x + 1 / x)) →
  -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end range_of_a_l143_143298


namespace option_A_option_D_l143_143170

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- Sum of the first n terms
variable {a1 d : ℤ} -- First term and common difference

-- Conditions for arithmetic sequence
axiom a_n (n : ℕ) : a n = a1 + ↑(n-1) * d
axiom S_n (n : ℕ) : S n = n * a1 + (n * (n - 1) / 2) * d
axiom condition : a 4 + 2 * a 8 = a 6

theorem option_A : a 7 = 0 :=
by
  -- Proof to be done
  sorry

theorem option_D : S 13 = 0 :=
by
  -- Proof to be done
  sorry

end option_A_option_D_l143_143170


namespace div_by_7_of_sum_div_by_7_l143_143760

theorem div_by_7_of_sum_div_by_7 (x y z : ℤ) (h : 7 ∣ x^3 + y^3 + z^3) : 7 ∣ x * y * z := by
  sorry

end div_by_7_of_sum_div_by_7_l143_143760


namespace marissa_tied_boxes_l143_143658

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l143_143658


namespace lara_bouncy_house_time_l143_143577

theorem lara_bouncy_house_time :
  let run1_time := (3 * 60 + 45) + (2 * 60 + 10) + (1 * 60 + 28)
  let door_time := 73
  let run2_time := (2 * 60 + 55) + (1 * 60 + 48) + (1 * 60 + 15)
  run1_time + door_time + run2_time = 874 := by
    let run1_time := 225 + 130 + 88
    let door_time := 73
    let run2_time := 175 + 108 + 75
    sorry

end lara_bouncy_house_time_l143_143577


namespace complex_modulus_product_l143_143263

noncomputable def z1 : ℂ := 4 - 3 * Complex.I
noncomputable def z2 : ℂ := 4 + 3 * Complex.I

theorem complex_modulus_product : Complex.abs z1 * Complex.abs z2 = 25 := by 
  sorry

end complex_modulus_product_l143_143263


namespace find_box_depth_l143_143492

-- Definitions and conditions
noncomputable def length : ℝ := 1.6
noncomputable def width : ℝ := 1.0
noncomputable def edge : ℝ := 0.2
noncomputable def number_of_blocks : ℝ := 120

-- The goal is to find the depth of the box
theorem find_box_depth (d : ℝ) :
  length * width * d = number_of_blocks * (edge ^ 3) →
  d = 0.6 := 
sorry

end find_box_depth_l143_143492


namespace find_temperature_on_December_25_l143_143276

theorem find_temperature_on_December_25 {f : ℕ → ℤ}
  (h_recurrence : ∀ n, f (n - 1) + f (n + 1) = f n)
  (h_initial1 : f 3 = 5)
  (h_initial2 : f 31 = 2) :
  f 25 = -3 :=
  sorry

end find_temperature_on_December_25_l143_143276


namespace Seth_gave_to_his_mother_l143_143842

variable (x : ℕ)

-- Define the conditions as per the problem statement
def initial_boxes := 9
def remaining_boxes_after_giving_to_mother := initial_boxes - x
def remaining_boxes_after_giving_half := remaining_boxes_after_giving_to_mother / 2

-- Specify the final condition
def final_boxes := 4

-- Form the main theorem
theorem Seth_gave_to_his_mother :
  final_boxes = remaining_boxes_after_giving_to_mother / 2 →
  initial_boxes - x = 8 :=
by sorry

end Seth_gave_to_his_mother_l143_143842


namespace actual_cost_of_article_l143_143176

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article_l143_143176


namespace find_angle_A_l143_143363

theorem find_angle_A (A B C a b c : ℝ) 
  (h_triangle: a = Real.sqrt 2)
  (h_sides: b = 2 * Real.sin B + Real.cos B)
  (h_b_eq: b = Real.sqrt 2)
  (h_a_lt_b: a < b)
  : A = Real.pi / 6 := sorry

end find_angle_A_l143_143363


namespace emmalyn_earnings_l143_143847

theorem emmalyn_earnings
  (rate_per_meter : ℚ := 0.20)
  (number_of_fences : ℚ := 50)
  (length_per_fence : ℚ := 500) :
  rate_per_meter * (number_of_fences * length_per_fence) = 5000 := by
  sorry

end emmalyn_earnings_l143_143847


namespace correct_propositions_l143_143455

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_propositions :
  ¬ ∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi ∧
  (∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (- (Real.pi / 6)) = 0) ∧
  ¬ ∀ (x : ℝ), f x = f (-x - Real.pi / 6) :=
sorry

end correct_propositions_l143_143455


namespace video_streaming_budget_l143_143452

theorem video_streaming_budget 
  (weekly_food_budget : ℕ) 
  (weeks : ℕ) 
  (total_food_budget : ℕ) 
  (rent : ℕ) 
  (phone : ℕ) 
  (savings_rate : ℝ)
  (total_savings : ℕ) 
  (total_expenses : ℕ) 
  (known_expenses: ℕ) 
  (total_spending : ℕ):
  weekly_food_budget = 100 →
  weeks = 4 →
  total_food_budget = weekly_food_budget * weeks →
  rent = 1500 →
  phone = 50 →
  savings_rate = 0.10 →
  total_savings = 198 →
  total_expenses = total_food_budget + rent + phone →
  total_spending = (total_savings : ℝ) / savings_rate →
  known_expenses = total_expenses →
  total_spending - known_expenses = 30 :=
by sorry

end video_streaming_budget_l143_143452


namespace total_trees_planted_l143_143592

theorem total_trees_planted (apple_trees orange_trees : ℕ) (h₁ : apple_trees = 47) (h₂ : orange_trees = 27) : apple_trees + orange_trees = 74 := 
by
  -- We skip the proof step
  sorry

end total_trees_planted_l143_143592


namespace students_who_wanted_fruit_l143_143876

theorem students_who_wanted_fruit (red_apples green_apples extra_apples ordered_apples served_apples students_wanted_fruit : ℕ)
    (h1 : red_apples = 43)
    (h2 : green_apples = 32)
    (h3 : extra_apples = 73)
    (h4 : ordered_apples = red_apples + green_apples)
    (h5 : served_apples = ordered_apples + extra_apples)
    (h6 : students_wanted_fruit = served_apples - ordered_apples) :
    students_wanted_fruit = 73 := 
by
    sorry

end students_who_wanted_fruit_l143_143876


namespace sum_of_squares_of_roots_l143_143024

theorem sum_of_squares_of_roots 
  (x1 x2 : ℝ) 
  (h₁ : 5 * x1^2 - 6 * x1 - 4 = 0)
  (h₂ : 5 * x2^2 - 6 * x2 - 4 = 0)
  (h₃ : x1 ≠ x2) :
  x1^2 + x2^2 = 76 / 25 := sorry

end sum_of_squares_of_roots_l143_143024


namespace ratio_almonds_to_walnuts_l143_143285

theorem ratio_almonds_to_walnuts (almonds walnuts mixture : ℝ) 
  (h1 : almonds = 116.67)
  (h2 : mixture = 140)
  (h3 : walnuts = mixture - almonds) : 
  (almonds / walnuts) = 5 :=
by
  sorry

end ratio_almonds_to_walnuts_l143_143285


namespace limit_equivalence_l143_143194

open Nat
open Real

variable {u : ℕ → ℝ} {L : ℝ}

def original_def (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L - u n| ≤ ε

def def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, n < N ∨ |L - u n| ≤ ε)

def def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∀ n : ℕ, ∃ N : ℕ, n ≥ N → |L - u n| ≤ ε

def def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |L - u n| < ε

def def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε > 0, ∀ n ≥ N, |L - u n| ≤ ε

theorem limit_equivalence :
  original_def u L ↔ def1 u L ∧ def3 u L ∧ ¬def2 u L ∧ ¬def4 u L :=
by
  sorry

end limit_equivalence_l143_143194


namespace product_of_roots_eq_25_l143_143899

theorem product_of_roots_eq_25 (t : ℝ) (h : t^2 - 10 * t + 25 = 0) : t * t = 25 :=
sorry

end product_of_roots_eq_25_l143_143899


namespace evaluate_expression_l143_143350

theorem evaluate_expression (x : ℝ) (h : x = 2) : 4 * x ^ 2 + 1 / 2 = 16.5 := by
  sorry

end evaluate_expression_l143_143350


namespace range_of_m_l143_143269

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x > m
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * m * x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ m ∈ Set.Ioo (-2:ℝ) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_m_l143_143269


namespace calc1_calc2_l143_143747

theorem calc1 : (1 * -11 + 8 + (-14) = -17) := by
  sorry

theorem calc2 : (13 - (-12) + (-21) = 4) := by
  sorry

end calc1_calc2_l143_143747


namespace max_large_sculptures_l143_143906

theorem max_large_sculptures (x y : ℕ) (h1 : 1 * x = x) 
  (h2 : 3 * y = y + y + y) 
  (h3 : ∃ n, n = (x + y) / 2) 
  (h4 : x + 3 * y + (x + y) / 2 ≤ 30) 
  (h5 : x > y) : 
  y ≤ 4 := 
sorry

end max_large_sculptures_l143_143906


namespace time_for_second_train_to_cross_l143_143500

def length_first_train : ℕ := 100
def speed_first_train : ℕ := 10
def length_second_train : ℕ := 150
def speed_second_train : ℕ := 15
def distance_between_trains : ℕ := 50

def total_distance : ℕ := length_first_train + length_second_train + distance_between_trains
def relative_speed : ℕ := speed_second_train - speed_first_train

theorem time_for_second_train_to_cross :
  total_distance / relative_speed = 60 :=
by
  -- Definitions and intermediate steps would be handled in the proof here
  sorry

end time_for_second_train_to_cross_l143_143500


namespace rectangle_fitting_condition_l143_143880

variables {a b c d : ℝ}

theorem rectangle_fitting_condition
  (h1: a < c ∧ c ≤ d ∧ d < b)
  (h2: a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 :=
sorry

end rectangle_fitting_condition_l143_143880


namespace wood_needed_l143_143118

variable (total_needed : ℕ) (friend_pieces : ℕ) (brother_pieces : ℕ)

/-- Alvin's total needed wood is 376 pieces, he got 123 from his friend and 136 from his brother.
    Prove that Alvin needs 117 more pieces. -/
theorem wood_needed (h1 : total_needed = 376) (h2 : friend_pieces = 123) (h3 : brother_pieces = 136) :
  total_needed - (friend_pieces + brother_pieces) = 117 := by
  sorry

end wood_needed_l143_143118


namespace find_certain_number_l143_143883

theorem find_certain_number : ∃ x : ℕ, (((x - 50) / 4) * 3 + 28 = 73) → x = 110 :=
by
  sorry

end find_certain_number_l143_143883


namespace math_equivalent_proof_l143_143730

-- Define the probabilities given the conditions
def P_A1 := 3 / 4
def P_A2 := 2 / 3
def P_A3 := 1 / 2
def P_B1 := 3 / 5
def P_B2 := 2 / 5

-- Define events
def P_C : ℝ := (P_A1 * P_B1 * (1 - P_A2)) + (P_A1 * P_B1 * P_A2 * P_B2 * (1 - P_A3))

-- Probability distribution of X
def P_X_0 : ℝ := (1 - P_A1) + P_C
def P_X_600 : ℝ := P_A1 * (1 - P_B1)
def P_X_1500 : ℝ := P_A1 * P_B1 * P_A2 * (1 - P_B2)
def P_X_3000 : ℝ := P_A1 * P_B1 * P_A2 * P_B2 * P_A3

-- Expected value of X
def E_X : ℝ := 600 * P_X_600 + 1500 * P_X_1500 + 3000 * P_X_3000

-- Statement to prove P(C) and expected value E(X)
theorem math_equivalent_proof :
  P_C = 21 / 100 ∧ 
  P_X_0 = 23 / 50 ∧
  P_X_600 = 3 / 10 ∧
  P_X_1500 = 9 / 50 ∧
  P_X_3000 = 3 / 50 ∧ 
  E_X = 630 := 
by 
  sorry

end math_equivalent_proof_l143_143730


namespace fraction_of_4d_nails_l143_143742

variables (fraction2d fraction2d_or_4d fraction4d : ℚ)

theorem fraction_of_4d_nails
  (h1 : fraction2d = 0.25)
  (h2 : fraction2d_or_4d = 0.75) :
  fraction4d = 0.50 :=
by
  sorry

end fraction_of_4d_nails_l143_143742


namespace curve_meets_line_once_l143_143478

theorem curve_meets_line_once (a : ℝ) (h : a > 0) :
  (∃! P : ℝ × ℝ, (∃ θ : ℝ, P.1 = a + 4 * Real.cos θ ∧ P.2 = 1 + 4 * Real.sin θ)
  ∧ (3 * P.1 + 4 * P.2 = 5)) → a = 7 :=
sorry

end curve_meets_line_once_l143_143478


namespace total_price_of_books_l143_143999

theorem total_price_of_books (total_books : ℕ) (math_books : ℕ) (cost_math_book : ℕ) (cost_history_book : ℕ) (remaining_books := total_books - math_books) (total_math_cost := math_books * cost_math_book) (total_history_cost := remaining_books * cost_history_book ) : total_books = 80 → math_books = 27 → cost_math_book = 4 → cost_history_book = 5 → total_math_cost + total_history_cost = 373 :=
by
  intros
  sorry

end total_price_of_books_l143_143999


namespace points_lie_on_line_l143_143818

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t
  let y := (t - 1) / t
  x + y = 2 := by
  sorry

end points_lie_on_line_l143_143818


namespace matrix_characteristic_eq_l143_143889

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 2], ![2, 1, 2], ![2, 2, 1]]

theorem matrix_characteristic_eq :
  ∃ (a b c : ℚ), a = -6 ∧ b = -12 ∧ c = -18 ∧ 
  (B ^ 3 + a • (B ^ 2) + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0) :=
by
  sorry

end matrix_characteristic_eq_l143_143889


namespace max_possible_median_l143_143657

/-- 
Given:
1. The Beverage Barn sold 300 cans of soda to 120 customers.
2. Every customer bought at least 1 can of soda but no more than 5 cans.
Prove that the maximum possible median number of cans of soda bought per customer is 5.
-/
theorem max_possible_median (total_cans : ℕ) (customers : ℕ) (min_can_per_customer : ℕ) (max_can_per_customer : ℕ) :
  total_cans = 300 ∧ customers = 120 ∧ min_can_per_customer = 1 ∧ max_can_per_customer = 5 →
  (∃ median : ℕ, median = 5) :=
by
  sorry

end max_possible_median_l143_143657


namespace cereal_discount_l143_143693

theorem cereal_discount (milk_normal_cost milk_discounted_cost total_savings milk_quantity cereal_quantity: ℝ) 
  (total_milk_savings cereal_savings_per_box: ℝ) 
  (h1: milk_normal_cost = 3)
  (h2: milk_discounted_cost = 2)
  (h3: total_savings = 8)
  (h4: milk_quantity = 3)
  (h5: cereal_quantity = 5)
  (h6: total_milk_savings = milk_quantity * (milk_normal_cost - milk_discounted_cost)) 
  (h7: total_milk_savings + cereal_quantity * cereal_savings_per_box = total_savings):
  cereal_savings_per_box = 1 :=
by 
  sorry

end cereal_discount_l143_143693


namespace problem_xyz_inequality_l143_143485

theorem problem_xyz_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  x * y * z ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ x * y * z + 2 :=
by 
  sorry

end problem_xyz_inequality_l143_143485


namespace product_divisible_by_4_l143_143553

noncomputable def biased_die_prob_divisible_by_4 : ℚ :=
  let q := 1/4  -- probability of rolling a number divisible by 3
  let p4 := 2 * q -- probability of rolling a number divisible by 4
  let p_neither := (1 - p4) * (1 - p4) -- probability of neither roll being divisible by 4
  1 - p_neither -- probability that at least one roll is divisible by 4

theorem product_divisible_by_4 :
  biased_die_prob_divisible_by_4 = 3/4 :=
by
  sorry

end product_divisible_by_4_l143_143553


namespace sum_of_distinct_integers_eq_zero_l143_143339

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero_l143_143339


namespace dans_age_l143_143804

variable {x : ℤ}

theorem dans_age (h : x + 20 = 7 * (x - 4)) : x = 8 := by
  sorry

end dans_age_l143_143804


namespace most_frequent_data_is_mode_l143_143168

def most_frequent_data_name (dataset : Type) : String := "Mode"

theorem most_frequent_data_is_mode (dataset : Type) :
  most_frequent_data_name dataset = "Mode" :=
by
  sorry

end most_frequent_data_is_mode_l143_143168


namespace S_equals_l143_143044
noncomputable def S : Real :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_equals : S = 2 * Real.sqrt 23 - 2 :=
by
  sorry

end S_equals_l143_143044


namespace find_a_l143_143679

def are_parallel (a : ℝ) : Prop :=
  (a + 1) = (2 - a)

theorem find_a (a : ℝ) (h : are_parallel a) : a = 0 :=
sorry

end find_a_l143_143679


namespace reflect_point_P_l143_143151

-- Define the point P
def P : ℝ × ℝ := (-3, 2)

-- Define the reflection across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Theorem to prove the coordinates of the point P with respect to the x-axis
theorem reflect_point_P : reflect_x_axis P = (-3, -2) := by
  sorry

end reflect_point_P_l143_143151


namespace find_digit_D_l143_143496

def is_digit (n : ℕ) : Prop := n < 10

theorem find_digit_D (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C)
  (h5 : B ≠ D) (h6 : C ≠ D) (h7 : is_digit A) (h8 : is_digit B) (h9 : is_digit C) (h10 : is_digit D) :
  (1000 * A + 100 * B + 10 * C + D) * 2 = 5472 → D = 6 := 
by
  sorry

end find_digit_D_l143_143496


namespace alyssa_games_last_year_l143_143137

theorem alyssa_games_last_year (games_this_year games_next_year games_total games_last_year : ℕ) (h1 : games_this_year = 11) (h2 : games_next_year = 15) (h3 : games_total = 39) (h4 : games_last_year + games_this_year + games_next_year = games_total) : games_last_year = 13 :=
by
  rw [h1, h2, h3] at h4
  sorry

end alyssa_games_last_year_l143_143137


namespace Randy_bats_l143_143863

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l143_143863


namespace cone_base_radius_and_slant_height_l143_143212

noncomputable def sector_angle := 300
noncomputable def sector_radius := 10
noncomputable def arc_length := (sector_angle / 360) * 2 * Real.pi * sector_radius

theorem cone_base_radius_and_slant_height :
  ∃ (r l : ℝ), arc_length = 2 * Real.pi * r ∧ l = sector_radius ∧ r = 8 ∧ l = 10 :=
by 
  sorry

end cone_base_radius_and_slant_height_l143_143212


namespace bottles_count_l143_143082

-- Defining the conditions from the problem statement
def condition1 (x y : ℕ) : Prop := 3 * x + 4 * y = 108
def condition2 (x y : ℕ) : Prop := 2 * x + 3 * y = 76

-- The proof statement combining conditions and the solution
theorem bottles_count (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 ∧ y = 12 :=
sorry

end bottles_count_l143_143082


namespace parallel_lines_a_eq_2_l143_143791

theorem parallel_lines_a_eq_2 {a : ℝ} :
  (∀ x y : ℝ, a * x + (a + 2) * y + 2 = 0 ∧ x + a * y - 2 = 0 → False) ↔ a = 2 :=
by
  sorry

end parallel_lines_a_eq_2_l143_143791


namespace remainder_of_55_power_55_plus_55_l143_143970

-- Define the problem statement using Lean

theorem remainder_of_55_power_55_plus_55 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  sorry

end remainder_of_55_power_55_plus_55_l143_143970


namespace friend_spent_11_l143_143802

-- Definitions of the conditions
def total_lunch_cost (you friend : ℝ) : Prop := you + friend = 19
def friend_spent_more (you friend : ℝ) : Prop := friend = you + 3

-- The theorem to prove
theorem friend_spent_11 (you friend : ℝ) 
  (h1 : total_lunch_cost you friend) 
  (h2 : friend_spent_more you friend) : 
  friend = 11 := 
by 
  sorry

end friend_spent_11_l143_143802


namespace max_area_l143_143494

theorem max_area (l w : ℝ) (h : l + 3 * w = 500) : l * w ≤ 62500 :=
by
  sorry

end max_area_l143_143494


namespace simplify_expression_l143_143415

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 5 + 2) + 2 / (Real.sqrt 7 - 2))) = 
  (6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35) :=
  sorry

end simplify_expression_l143_143415


namespace find_speed_l143_143268

variable (d : ℝ) (t : ℝ)
variable (h1 : d = 50 * (t + 1/12))
variable (h2 : d = 70 * (t - 1/12))

theorem find_speed (d t : ℝ)
  (h1 : d = 50 * (t + 1/12))
  (h2 : d = 70 * (t - 1/12)) :
  58 = d / t := by
  sorry

end find_speed_l143_143268


namespace binomial_coefficient_12_10_l143_143418

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l143_143418


namespace subcommittees_with_at_least_one_coach_l143_143002

-- Definitions based on conditions
def total_members : ℕ := 12
def total_coaches : ℕ := 5
def subcommittee_size : ℕ := 5

-- Lean statement of the problem
theorem subcommittees_with_at_least_one_coach :
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - total_coaches) subcommittee_size) = 771 := by
  sorry

end subcommittees_with_at_least_one_coach_l143_143002


namespace hyperbola_asymptote_l143_143343

theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (y = (1/2) * x) ∨ (y = -(1/2) * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptote_l143_143343


namespace bianca_initial_cupcakes_l143_143229

theorem bianca_initial_cupcakes (X : ℕ) (h : X - 6 + 17 = 25) : X = 14 := by
  sorry

end bianca_initial_cupcakes_l143_143229


namespace range_of_a_outside_circle_l143_143988

  variable (a : ℝ)

  def point_outside_circle (a : ℝ) : Prop :=
    let x := a
    let y := 2
    let distance_sqr := (x - a) ^ 2 + (y - 3 / 2) ^ 2
    let r_sqr := 1 / 4
    distance_sqr > r_sqr

  theorem range_of_a_outside_circle {a : ℝ} (h : point_outside_circle a) :
      2 < a ∧ a < 9 / 4 := sorry
  
end range_of_a_outside_circle_l143_143988


namespace time_addition_correct_l143_143708

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct_l143_143708


namespace remaining_food_can_cater_children_l143_143026

theorem remaining_food_can_cater_children (A C : ℝ) 
  (h_food_adults : 70 * A = 90 * C) 
  (h_35_adults_ate : ∀ n: ℝ, (n = 35) → 35 * A = 35 * (9/7) * C) : 
  70 * A - 35 * A = 45 * C :=
by
  sorry

end remaining_food_can_cater_children_l143_143026


namespace value_of_3W5_l143_143507

def W (a b : ℕ) : ℕ := b + 7 * a - a ^ 2

theorem value_of_3W5 : W 3 5 = 17 := by 
  sorry

end value_of_3W5_l143_143507


namespace relationship_between_a_b_c_l143_143982

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l143_143982


namespace keiko_speed_l143_143203

theorem keiko_speed (wA wB tA tB : ℝ) (v : ℝ)
    (h1: wA = 4)
    (h2: wB = 8)
    (h3: tA = 48)
    (h4: tB = 72)
    (h5: v = (24 * π) / 60) :
    v = 2 * π / 5 :=
by
  sorry

end keiko_speed_l143_143203


namespace solution_set_of_inequality_l143_143362

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 - 7*x + 12 < 0 ↔ 3 < x ∧ x < 4 :=
by {
  sorry
}

end solution_set_of_inequality_l143_143362


namespace maximum_possible_value_of_expression_l143_143859

theorem maximum_possible_value_of_expression :
  ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) ∧
  (b = 0 ∨ b = 1 ∨ b = 3 ∨ b = 4) ∧
  (c = 0 ∨ c = 1 ∨ c = 3 ∨ c = 4) ∧
  (d = 0 ∨ d = 1 ∨ d = 3 ∨ d = 4) ∧
  ¬ (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) →
  (c * a^b + d ≤ 196) :=
by sorry

end maximum_possible_value_of_expression_l143_143859


namespace continuous_function_form_l143_143174

noncomputable def f (t : ℝ) : ℝ := sorry

theorem continuous_function_form (f : ℝ → ℝ) (h1 : f 0 = -1 / 2) (h2 : ∀ x y, f (x + y) ≥ f x + f y + f (x * y) + 1) :
  ∃ (a : ℝ), ∀ x, f x = 1 / 2 + a * x + (a/2) * x ^ 2 := sorry

end continuous_function_form_l143_143174


namespace composite_proposition_l143_143259

theorem composite_proposition :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬ (1 < 0) :=
by
  sorry

end composite_proposition_l143_143259


namespace exp_calculation_l143_143086

theorem exp_calculation : 0.125^8 * (-8)^7 = -0.125 :=
by
  -- conditions used directly in proof
  have h1 : 0.125 = 1 / 8 := sorry
  have h2 : (-1)^7 = -1 := sorry
  -- the problem statement
  sorry

end exp_calculation_l143_143086


namespace total_number_of_crickets_l143_143706

def initial_crickets : ℝ := 7.0
def additional_crickets : ℝ := 11.0
def total_crickets : ℝ := 18.0

theorem total_number_of_crickets :
  initial_crickets + additional_crickets = total_crickets :=
by
  sorry

end total_number_of_crickets_l143_143706


namespace shaded_area_is_14_percent_l143_143262

def side_length : ℕ := 20
def rectangle_width : ℕ := 35
def rectangle_height : ℕ := side_length
def rectangle_area : ℕ := rectangle_width * rectangle_height
def overlap_length : ℕ := 2 * side_length - rectangle_width
def shaded_area : ℕ := overlap_length * side_length
def shaded_percentage : ℚ := (shaded_area : ℚ) / rectangle_area * 100

theorem shaded_area_is_14_percent : shaded_percentage = 14 := by
  sorry

end shaded_area_is_14_percent_l143_143262


namespace hyperbola_focal_point_k_l143_143198

theorem hyperbola_focal_point_k (k : ℝ) :
  (∃ (c : ℝ), c = 2 ∧ (5 : ℝ) * 2 ^ 2 - k * 0 ^ 2 = 5) →
  k = (5 : ℝ) / 3 :=
by
  sorry

end hyperbola_focal_point_k_l143_143198


namespace find_a_l143_143426

noncomputable def slope1 (a : ℝ) : ℝ := -3 / (3^a - 3)
noncomputable def slope2 : ℝ := 2

theorem find_a (a : ℝ) (h : slope1 a * slope2 = -1) : a = 2 :=
sorry

end find_a_l143_143426


namespace solution_problem_l143_143665

noncomputable def problem :=
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 →
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c)

theorem solution_problem : problem :=
  sorry

end solution_problem_l143_143665


namespace part_1_part_2_l143_143457

def f (x a : ℝ) : ℝ := |x - a| + 5 * x

theorem part_1 (x : ℝ) : (|x + 1| + 5 * x ≤ 5 * x + 3) ↔ (x ∈ Set.Icc (-4 : ℝ) 2) :=
by
  sorry

theorem part_2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) :=
by
  sorry

end part_1_part_2_l143_143457


namespace last_four_digits_of_5_pow_2013_l143_143870

theorem last_four_digits_of_5_pow_2013 : (5 ^ 2013) % 10000 = 3125 :=
by
  sorry

end last_four_digits_of_5_pow_2013_l143_143870


namespace roots_of_unity_real_root_l143_143384

theorem roots_of_unity_real_root (n : ℕ) (h_even : n % 2 = 0) : ∃ z : ℝ, z ≠ 1 ∧ z^n = 1 :=
by
  sorry

end roots_of_unity_real_root_l143_143384


namespace reflex_angle_at_G_correct_l143_143762

noncomputable def reflex_angle_at_G
    (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80)
    : ℝ :=
  360 - (180 - (180 - angle_BAG) - (180 - angle_GEL))

theorem reflex_angle_at_G_correct :
    (∀ (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80),
    reflex_angle_at_G B A E L G on_line off_line angle_BAG angle_GEL h1 h2 = 340) := sorry

end reflex_angle_at_G_correct_l143_143762


namespace pipeA_fills_tank_in_56_minutes_l143_143672

-- Define the relevant variables and conditions.
variable (t : ℕ) -- Time for Pipe A to fill the tank in minutes

-- Condition: Pipe B fills the tank 7 times faster than Pipe A
def pipeB_time (t : ℕ) := t / 7

-- Combined rate of Pipe A and Pipe B filling the tank in 7 minutes
def combined_rate (t : ℕ) := (1 / t) + (1 / pipeB_time t)

-- Given the combined rate fills the tank in 7 minutes
def combined_rate_equals (t : ℕ) := combined_rate t = 1 / 7

-- The proof statement
theorem pipeA_fills_tank_in_56_minutes (t : ℕ) (h : combined_rate_equals t) : t = 56 :=
sorry

end pipeA_fills_tank_in_56_minutes_l143_143672


namespace sequence_S15_is_211_l143_143630

theorem sequence_S15_is_211 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2)
  (h3 : ∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + S 1)) :
  S 15 = 211 := 
sorry

end sequence_S15_is_211_l143_143630


namespace find_length_DC_l143_143438

noncomputable def length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) : ℕ :=
  let DC := 29
  DC

theorem find_length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) (h6 : 20^2 + BC^2 = DC^2) : length_DC AB BC AD BD h1 h2 h3 h4 h5 = 29 :=
  by
  sorry

end find_length_DC_l143_143438


namespace chip_sheets_per_pack_l143_143414

noncomputable def sheets_per_pack (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) 
                                  (weeks : ℕ) (packs : ℕ) : ℕ :=
(pages_per_day * days_per_week * classes * weeks) / packs

theorem chip_sheets_per_pack :
  sheets_per_pack 2 5 5 6 3 = 100 :=
sorry

end chip_sheets_per_pack_l143_143414


namespace opposite_of_half_l143_143056

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l143_143056


namespace range_of_vector_magnitude_l143_143218

variable {V : Type} [NormedAddCommGroup V]

theorem range_of_vector_magnitude
  (A B C : V)
  (h_AB : ‖A - B‖ = 8)
  (h_AC : ‖A - C‖ = 5) :
  3 ≤ ‖B - C‖ ∧ ‖B - C‖ ≤ 13 :=
sorry

end range_of_vector_magnitude_l143_143218


namespace problem_statement_l143_143635

-- Defining the real numbers and the hypothesis
variables {a b c x y z : ℝ}
variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 31 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

-- State the theorem
theorem problem_statement : 
  (a / (a - 17) + b / (b - 31) + c / (c - 53) = 1) :=
by
  sorry

end problem_statement_l143_143635


namespace remaining_two_by_two_square_exists_l143_143312

theorem remaining_two_by_two_square_exists (grid_size : ℕ) (cut_squares : ℕ) : grid_size = 29 → cut_squares = 99 → 
  ∃ remaining_square : ℕ, remaining_square = 1 :=
by
  intros
  sorry

end remaining_two_by_two_square_exists_l143_143312


namespace find_percentage_l143_143442

def problem_statement (n P : ℕ) := 
  n = (P / 100) * n + 84

theorem find_percentage : ∃ P, problem_statement 100 P ∧ (P = 16) :=
by
  sorry

end find_percentage_l143_143442


namespace least_number_subtracted_l143_143158

/--
  What least number must be subtracted from 9671 so that the remaining number is divisible by 5, 7, and 11?
-/
theorem least_number_subtracted
  (x : ℕ) :
  (9671 - x) % 5 = 0 ∧ (9671 - x) % 7 = 0 ∧ (9671 - x) % 11 = 0 ↔ x = 46 :=
sorry

end least_number_subtracted_l143_143158


namespace find_roots_l143_143625

def polynomial (x: ℝ) := x^3 - 2*x^2 - x + 2

theorem find_roots : { x : ℝ // polynomial x = 0 } = ({1, -1, 2} : Set ℝ) :=
by
  sorry

end find_roots_l143_143625


namespace simplify_expression_l143_143934

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := 
by sorry

end simplify_expression_l143_143934


namespace ratio_of_roots_l143_143768

theorem ratio_of_roots (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) (h₃ : a * x₁^2 + b * x₁ + c = 0) (h₄ : a * x₂^2 + b * x₂ + c = 0) (h₅ : x₁ = 4 * x₂) : (b^2) / (a * c) = 25 / 4 :=
by
  sorry

end ratio_of_roots_l143_143768


namespace max_principals_l143_143926

theorem max_principals (n_years term_length max_principals: ℕ) 
  (h1 : n_years = 12) 
  (h2 : term_length = 4)
  (h3 : max_principals = 4): 
  (∃ p : ℕ, p = max_principals) :=
by
  sorry

end max_principals_l143_143926


namespace prob_exactly_two_trains_on_time_is_0_398_l143_143311

-- Definitions and conditions
def eventA := true
def eventB := true
def eventC := true

def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Question definition (to be proved)
def exact_two_on_time : ℝ :=
  P_A * P_B * P_not_C + P_A * P_not_B * P_C + P_not_A * P_B * P_C

-- Theorem statement
theorem prob_exactly_two_trains_on_time_is_0_398 :
  exact_two_on_time = 0.398 := sorry

end prob_exactly_two_trains_on_time_is_0_398_l143_143311


namespace parallel_lines_slope_l143_143401

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope_l143_143401


namespace sum_midpoints_x_coordinates_is_15_l143_143924

theorem sum_midpoints_x_coordinates_is_15 :
  ∀ (a b : ℝ), a + 2 * b = 15 → 
  (a + 2 * b) = 15 :=
by
  intros a b h
  sorry

end sum_midpoints_x_coordinates_is_15_l143_143924


namespace ratio_of_inscribed_squares_l143_143099

-- Definitions of the conditions
def right_triangle_sides (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2

def inscribed_square_1 (x : ℚ) : Prop := x = 18 / 7

def inscribed_square_2 (y : ℚ) : Prop := y = 32 / 7

-- Statement of the problem
theorem ratio_of_inscribed_squares (x y : ℚ) : right_triangle_sides 6 8 10 ∧ inscribed_square_1 x ∧ inscribed_square_2 y → (x / y) = 9 / 16 :=
by
  sorry

end ratio_of_inscribed_squares_l143_143099


namespace parking_garage_savings_l143_143901

theorem parking_garage_savings :
  let weekly_cost := 10
  let monthly_cost := 35
  let weeks_per_year := 52
  let months_per_year := 12
  let annual_weekly_cost := weekly_cost * weeks_per_year
  let annual_monthly_cost := monthly_cost * months_per_year
  let annual_savings := annual_weekly_cost - annual_monthly_cost
  annual_savings = 100 := 
by
  sorry

end parking_garage_savings_l143_143901


namespace solve_for_p_l143_143793

theorem solve_for_p (a b c p t : ℝ) (h1 : a + b + c + p = 360) (h2 : t = 180 - c) : 
  p = 180 - a - b + t :=
by
  sorry

end solve_for_p_l143_143793


namespace marie_gift_boxes_l143_143620

theorem marie_gift_boxes
  (total_eggs : ℕ)
  (weight_per_egg : ℕ)
  (remaining_weight : ℕ)
  (melted_eggs_weight : ℕ)
  (eggs_per_box : ℕ)
  (total_boxes : ℕ)
  (H1 : total_eggs = 12)
  (H2 : weight_per_egg = 10)
  (H3 : remaining_weight = 90)
  (H4 : melted_eggs_weight = total_eggs * weight_per_egg - remaining_weight)
  (H5 : melted_eggs_weight / weight_per_egg = eggs_per_box)
  (H6 : total_eggs / eggs_per_box = total_boxes) :
  total_boxes = 4 := 
sorry

end marie_gift_boxes_l143_143620


namespace sum_of_coefficients_is_256_l143_143794

theorem sum_of_coefficients_is_256 :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  ((x : ℤ) - a)^8 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 → 
  a5 = 56 →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 256 :=
by
  intros
  sorry

end sum_of_coefficients_is_256_l143_143794


namespace smallest_value_of_expression_l143_143279

theorem smallest_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 - b^2 = 16) : 
  (∃ k : ℚ, k = (a + b) / (a - b) + (a - b) / (a + b) ∧ (∀ x : ℚ, x = (a + b) / (a - b) + (a - b) / (a + b) → x ≥ 9/4)) :=
sorry

end smallest_value_of_expression_l143_143279


namespace proposition_A_l143_143462

variables {m n : Line} {α β : Plane}

def parallel (x y : Line) : Prop := sorry -- definition for parallel lines
def perpendicular (x : Line) (P : Plane) : Prop := sorry -- definition for perpendicular line to plane
def parallel_planes (P Q : Plane) : Prop := sorry -- definition for parallel planes

theorem proposition_A (hmn : parallel m n) (hperp_mα : perpendicular m α) (hperp_nβ : perpendicular n β) : parallel_planes α β :=
sorry

end proposition_A_l143_143462


namespace three_pizzas_needed_l143_143394

noncomputable def masha_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "sausage" ∉ p

noncomputable def vanya_pizza (p : Set String) : Prop :=
  "mushrooms" ∈ p

noncomputable def dasha_pizza (p : Set String) : Prop :=
  "tomatoes" ∉ p

noncomputable def nikita_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "mushrooms" ∉ p

noncomputable def igor_pizza (p : Set String) : Prop :=
  "mushrooms" ∉ p ∧ "sausage" ∈ p

theorem three_pizzas_needed (p1 p2 p3 : Set String) :
  (∃ p1, masha_pizza p1 ∧ vanya_pizza p1 ∧ dasha_pizza p1 ∧ nikita_pizza p1 ∧ igor_pizza p1) →
  (∃ p2, masha_pizza p2 ∧ vanya_pizza p2 ∧ dasha_pizza p2 ∧ nikita_pizza p2 ∧ igor_pizza p2) →
  (∃ p3, masha_pizza p3 ∧ vanya_pizza p3 ∧ dasha_pizza p3 ∧ nikita_pizza p3 ∧ igor_pizza p3) →
  ∀ p, ¬ ((masha_pizza p ∨ dasha_pizza p) ∧ vanya_pizza p ∧ (nikita_pizza p ∨ igor_pizza p)) :=
sorry

end three_pizzas_needed_l143_143394


namespace initial_bones_count_l143_143038

theorem initial_bones_count (B : ℕ) (h1 : B + 8 = 23) : B = 15 :=
sorry

end initial_bones_count_l143_143038


namespace dryer_weight_l143_143861

theorem dryer_weight 
(empty_truck_weight crates_soda_weight num_crates soda_weight_factor 
    fresh_produce_weight_factor num_dryers fully_loaded_truck_weight : ℕ) 

  (h1 : empty_truck_weight = 12000) 
  (h2 : crates_soda_weight = 50) 
  (h3 : num_crates = 20) 
  (h4 : soda_weight_factor = crates_soda_weight * num_crates) 
  (h5 : fresh_produce_weight_factor = 2 * soda_weight_factor) 
  (h6 : num_dryers = 3) 
  (h7 : fully_loaded_truck_weight = 24000) 

  : (fully_loaded_truck_weight - empty_truck_weight 
      - (soda_weight_factor + fresh_produce_weight_factor)) / num_dryers = 3000 := 
by sorry

end dryer_weight_l143_143861


namespace symmetric_point_correct_l143_143388

-- Define the coordinates of point A
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D :=
  { x := -3
    y := -4
    z := 5 }

-- Define the symmetry function with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { p with y := -p.y }

-- The expected coordinates of the point symmetric to A with respect to the plane xOz
def D_expected : Point3D :=
  { x := -3
    y := 4
    z := 5 }

-- Theorem stating that the symmetric point of A with respect to the plane xOz is D_expected
theorem symmetric_point_correct :
  symmetric_xOz A = D_expected := 
by 
  sorry

end symmetric_point_correct_l143_143388


namespace max_ad_minus_bc_l143_143021

theorem max_ad_minus_bc (a b c d : ℤ) (ha : a ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hb : b ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hc : c ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hd : d ∈ Set.image (fun x => x) {(-1), 1, 2}) :
  ad - bc ≤ 6 :=
sorry

end max_ad_minus_bc_l143_143021


namespace simplify_fraction_l143_143542

theorem simplify_fraction :
  (3^100 + 3^98) / (3^100 - 3^98) = 5 / 4 := 
by sorry

end simplify_fraction_l143_143542


namespace set_intersection_eq_l143_143557

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5}
def ComplementU (S : Set ℕ) : Set ℕ := U \ S

theorem set_intersection_eq : 
  A ∩ (ComplementU B) = {1, 3} := 
by
  sorry

end set_intersection_eq_l143_143557


namespace subset_0_in_X_l143_143182

-- Define the set X
def X : Set ℤ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Define the theorem to prove
theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l143_143182


namespace sara_oranges_l143_143165

-- Conditions
def joan_oranges : Nat := 37
def total_oranges : Nat := 47

-- Mathematically equivalent proof problem: Prove that the number of oranges picked by Sara is 10
theorem sara_oranges : total_oranges - joan_oranges = 10 :=
by
  sorry

end sara_oranges_l143_143165


namespace iggy_running_hours_l143_143034

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end iggy_running_hours_l143_143034


namespace initial_amount_l143_143245

theorem initial_amount (x : ℝ) (h : 0.015 * x = 750) : x = 50000 :=
by
  sorry

end initial_amount_l143_143245


namespace range_of_f_l143_143454

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l143_143454


namespace parabola_equation_l143_143949

theorem parabola_equation (a b c : ℝ) (h1 : a^2 = 3) (h2 : b^2 = 1) (h3 : c^2 = a^2 + b^2) : 
  (c = 2) → (vertex = 0) → (focus = 2) → ∀ x y, y^2 = 16 * x := 
by 
  sorry

end parabola_equation_l143_143949


namespace partI_solution_set_partII_range_of_m_l143_143867

def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

theorem partI_solution_set (x : ℝ) :
  ∀ (x : ℝ), f x 5 ≤ 12 ↔ (-13 / 2 ≤ x ∧ x ≤ 11 / 2) :=
by
  sorry

theorem partII_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end partI_solution_set_partII_range_of_m_l143_143867


namespace find_valid_pair_l143_143771

noncomputable def valid_angle (x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ x = 180 * (n - 2) / n

noncomputable def valid_pair (x k : ℕ) : Prop :=
  valid_angle x ∧ valid_angle (k * x) ∧ 1 < k ∧ k < 5

theorem find_valid_pair : valid_pair 60 2 :=
by
  sorry

end find_valid_pair_l143_143771


namespace diff_reading_math_homework_l143_143916

-- Define the conditions as given in the problem
def pages_math_homework : ℕ := 3
def pages_reading_homework : ℕ := 4

-- The statement to prove that Rachel had 1 more page of reading homework than math homework
theorem diff_reading_math_homework : pages_reading_homework - pages_math_homework = 1 := by
  sorry

end diff_reading_math_homework_l143_143916


namespace probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l143_143405

-- Problem 1
theorem probability_meeting_twin (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (p + 1) = (2 * p) / (p + 1) :=
by
  sorry

-- Problem 2
theorem probability_twin_in_family (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (2 * p + (1 - p) ^ 2) = (2 * p) / (2 * p + (1 - p) ^ 2) :=
by
  sorry

-- Problem 3
theorem expected_twin_pairs (N : ℕ) (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  N * p / (p + 1) = N * p / (p + 1) :=
by
  sorry

end probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l143_143405


namespace least_zorgs_to_drop_more_points_than_eating_l143_143626

theorem least_zorgs_to_drop_more_points_than_eating :
  ∃ (n : ℕ), (∀ m < n, m * (m + 1) / 2 ≤ 20 * m) ∧ n * (n + 1) / 2 > 20 * n :=
sorry

end least_zorgs_to_drop_more_points_than_eating_l143_143626


namespace flat_fee_for_solar_panel_equipment_l143_143015

theorem flat_fee_for_solar_panel_equipment
  (land_acreage : ℕ)
  (land_cost_per_acre : ℕ)
  (house_cost : ℕ)
  (num_cows : ℕ)
  (cow_cost_per_cow : ℕ)
  (num_chickens : ℕ)
  (chicken_cost_per_chicken : ℕ)
  (installation_hours : ℕ)
  (installation_cost_per_hour : ℕ)
  (total_cost : ℕ)
  (total_spent : ℕ) :
  land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour = total_spent →
  total_cost = total_spent →
  total_cost - (land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour) = 26000 := by 
  sorry

end flat_fee_for_solar_panel_equipment_l143_143015


namespace combined_apples_sold_l143_143289

theorem combined_apples_sold (red_apples green_apples total_apples : ℕ) 
    (h1 : red_apples = 32) 
    (h2 : green_apples = (3 * (32 / 8))) 
    (h3 : total_apples = red_apples + green_apples) : 
    total_apples = 44 :=
by
  sorry

end combined_apples_sold_l143_143289


namespace total_cost_meal_l143_143835

-- Define the initial conditions
variables (x : ℝ) -- x represents the total cost of the meal

-- Initial number of friends
def initial_friends : ℝ := 4

-- New number of friends after additional friends join
def new_friends : ℝ := 7

-- The decrease in cost per friend
def cost_decrease : ℝ := 15

-- Lean statement to assert our proof
theorem total_cost_meal : x / initial_friends - x / new_friends = cost_decrease → x = 140 :=
by
  sorry

end total_cost_meal_l143_143835


namespace threshold_mu_l143_143622

/-- 
Find threshold values μ₁₀₀ and μ₁₀₀₀₀₀ such that 
F = m * n * sin (π / m) * sqrt (1 / n² + sin⁴ (π / m)) 
is definitely greater than 100 and 1,000,000 respectively for all m greater than μ₁₀₀ and μ₁₀₀₀₀₀, 
assuming n = m³. -/
theorem threshold_mu : 
  (∃ (μ₁₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 100) ∧ 
  (∃ (μ₁₀₀₀₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀₀₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 1000000) :=
sorry

end threshold_mu_l143_143622


namespace relationship_S_T_l143_143931

-- Definitions based on the given conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n

def seq_b (n : ℕ) : ℕ :=
  2 ^ (n - 1) + 1

def S (n : ℕ) : ℕ :=
  (n * (n + 1))

def T (n : ℕ) : ℕ :=
  (2^n) + n - 1

-- The conjecture and proofs
theorem relationship_S_T (n : ℕ) : 
  if n = 1 then T n = S n
  else if (2 ≤ n ∧ n < 5) then T n < S n
  else n ≥ 5 → T n > S n :=
by sorry

end relationship_S_T_l143_143931


namespace bells_toll_together_l143_143120

theorem bells_toll_together (a b c d : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 11) (h4 : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 :=
by
  rw [h1, h2, h3, h4]
  sorry

end bells_toll_together_l143_143120


namespace arithmetic_sequence_sum_l143_143920

variable {a_n : ℕ → ℕ} -- the arithmetic sequence

-- Define condition
def condition (a : ℕ → ℕ) : Prop :=
  a 1 + a 5 + a 9 = 18

-- The sum of the first n terms of arithmetic sequence is S_n
def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- The goal is to prove that S 9 = 54
theorem arithmetic_sequence_sum (h : condition a_n) : S 9 a_n = 54 :=
sorry

end arithmetic_sequence_sum_l143_143920


namespace monic_poly_7_r_8_l143_143389

theorem monic_poly_7_r_8 :
  ∃ (r : ℕ → ℕ), (r 1 = 1) ∧ (r 2 = 2) ∧ (r 3 = 3) ∧ (r 4 = 4) ∧ (r 5 = 5) ∧ (r 6 = 6) ∧ (r 7 = 7) ∧ (∀ (n : ℕ), 8 < n → r n = n) ∧ r 8 = 5048 :=
sorry

end monic_poly_7_r_8_l143_143389


namespace chad_ice_cost_l143_143694

theorem chad_ice_cost
  (n : ℕ) -- Number of people
  (p : ℕ) -- Pounds of ice per person
  (c : ℝ) -- Cost per 10 pound bag of ice
  (h1 : n = 20) 
  (h2 : p = 3)
  (h3 : c = 4.5) :
  (3 * 20 / 10) * 4.5 = 27 :=
by
  sorry

end chad_ice_cost_l143_143694


namespace speed_of_first_half_of_journey_l143_143470

theorem speed_of_first_half_of_journey
  (total_time : ℝ)
  (speed_second_half : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (second_half_distance : ℝ)
  (time_second_half : ℝ)
  (time_first_half : ℝ)
  (speed_first_half : ℝ) :
  total_time = 15 →
  speed_second_half = 24 →
  total_distance = 336 →
  first_half_distance = total_distance / 2 →
  second_half_distance = total_distance / 2 →
  time_second_half = second_half_distance / speed_second_half →
  time_first_half = total_time - time_second_half →
  speed_first_half = first_half_distance / time_first_half →
  speed_first_half = 21 :=
by intros; sorry

end speed_of_first_half_of_journey_l143_143470


namespace monotonic_increasing_interval_l143_143085

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end monotonic_increasing_interval_l143_143085


namespace find_S_l143_143365

variable {R S T c : ℝ}

theorem find_S
  (h1 : R = 2)
  (h2 : T = 1/2)
  (h3 : S = 4)
  (h4 : R = c * S / T)
  (h5 : R = 8)
  (h6 : T = 1/3) :
  S = 32 / 3 :=
by
  sorry

end find_S_l143_143365


namespace compare_expressions_l143_143293

theorem compare_expressions (n : ℕ) (hn : 0 < n):
  (n ≤ 48 ∧ 99^n + 100^n > 101^n) ∨ (n > 48 ∧ 99^n + 100^n < 101^n) :=
sorry  -- Proof is omitted.

end compare_expressions_l143_143293


namespace octahedron_plane_intersection_l143_143959

theorem octahedron_plane_intersection 
  (s : ℝ) 
  (a b c : ℕ) 
  (ha : Nat.Coprime a c) 
  (hb : ∀ p : ℕ, Prime p → p^2 ∣ b → False) 
  (hs : s = 2) 
  (hangle : ∀ θ, θ = 45 ∧ θ = 45) 
  (harea : ∃ A, A = (s^2 * Real.sqrt 3) / 2 ∧ A = a * Real.sqrt b / c): 
  a + b + c = 11 := 
by 
  sorry

end octahedron_plane_intersection_l143_143959


namespace problem_statement_l143_143094

theorem problem_statement (r p q : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hp2r_gt_q2r : p^2 * r > q^2 * r) :
  ¬ (-p > -q) ∧ ¬ (-p < q) ∧ ¬ (1 < -q / p) ∧ ¬ (1 > q / p) :=
by
  sorry

end problem_statement_l143_143094


namespace find_number_of_students_l143_143910

theorem find_number_of_students 
    (N T : ℕ) 
    (h1 : T = 80 * N)
    (h2 : (T - 350) / (N - 5) = 90) : 
    N = 10 :=
sorry

end find_number_of_students_l143_143910


namespace investment_ratio_l143_143393

theorem investment_ratio (total_profit b_profit : ℝ) (a c b : ℝ) :
  total_profit = 150000 ∧ b_profit = 75000 ∧ a / c = 2 ∧ a + b + c = total_profit →
  a / b = 2 / 3 :=
by
  sorry

end investment_ratio_l143_143393


namespace maximal_value_ratio_l143_143754

theorem maximal_value_ratio (a b c h : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_altitude : h = (a * b) / c) :
  ∃ θ : ℝ, a = c * Real.cos θ ∧ b = c * Real.sin θ ∧ (1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2) ∧
  ( Real.cos θ * Real.sin θ = (1 + 2 * Real.cos θ * Real.sin θ - 1) / 2 ) → 
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end maximal_value_ratio_l143_143754


namespace tangent_line_at_point_is_correct_l143_143337

theorem tangent_line_at_point_is_correct :
  ∀ (x y : ℝ), (y = x^2 + 2 * x) → (x = 1) → (y = 3) → (4 * x - y - 1 = 0) :=
by
  intros x y h_curve h_x h_y
  -- Here would be the proof
  sorry

end tangent_line_at_point_is_correct_l143_143337


namespace fraction_identity_l143_143539

theorem fraction_identity (m n r t : ℚ) 
  (h₁ : m / n = 3 / 5) 
  (h₂ : r / t = 8 / 9) :
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := 
by
  sorry

end fraction_identity_l143_143539


namespace rainfall_third_day_is_18_l143_143683

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l143_143683


namespace arithmetic_sequence_sum_l143_143502

variable {a : ℕ → ℕ}

-- Defining the arithmetic sequence condition
axiom arithmetic_sequence_condition : a 3 + a 7 = 37

-- The goal is to prove that the total of a_2 + a_4 + a_6 + a_8 is 74
theorem arithmetic_sequence_sum : a 2 + a 4 + a 6 + a 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l143_143502


namespace cos_double_angle_l143_143993

theorem cos_double_angle (α : ℝ) (h : Real.tan α = -3) : Real.cos (2 * α) = -4 / 5 := sorry

end cos_double_angle_l143_143993


namespace find_k_l143_143466

theorem find_k (x k : ℝ) (h₁ : (x^2 - k) * (x - k) = x^3 - k * (x^2 + x + 3))
               (h₂ : k ≠ 0) : k = -3 :=
by
  sorry

end find_k_l143_143466


namespace sixteen_k_plus_eight_not_perfect_square_l143_143922

theorem sixteen_k_plus_eight_not_perfect_square (k : ℕ) (hk : 0 < k) : ¬ ∃ m : ℕ, (16 * k + 8) = m * m := sorry

end sixteen_k_plus_eight_not_perfect_square_l143_143922


namespace new_cost_percentage_l143_143569

def cost (t b : ℝ) := t * b^5

theorem new_cost_percentage (t b : ℝ) : 
  let C := cost t b
  let W := cost (3 * t) (2 * b)
  W = 96 * C :=
by
  sorry

end new_cost_percentage_l143_143569


namespace division_of_expression_l143_143453

theorem division_of_expression (x y : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : (12 * x^2 * y) / (-6 * x * y) = -2 * x := by
  sorry

end division_of_expression_l143_143453


namespace find_m_l143_143179

open Complex

theorem find_m (m : ℝ) : (re ((1 + I) / (1 - I) + m * (1 - I) / (1 + I)) = ((1 + I) / (1 - I) + m * (1 - I) / (1 + I))) → m = 1 :=
by
  sorry

end find_m_l143_143179


namespace smallest_digit_to_correct_sum_l143_143974

theorem smallest_digit_to_correct_sum :
  ∃ (d : ℕ), d = 3 ∧
  (3 ∈ [3, 5, 7]) ∧
  (371 + 569 + 784 + (d*100) = 1824) := sorry

end smallest_digit_to_correct_sum_l143_143974


namespace cubic_roots_sum_cube_l143_143838

theorem cubic_roots_sum_cube (a b c : ℂ) (h : ∀x : ℂ, (x=a ∨ x=b ∨ x=c) → (x^3 - 2*x^2 + 3*x - 4 = 0)) : a^3 + b^3 + c^3 = 2 :=
sorry

end cubic_roots_sum_cube_l143_143838


namespace molecular_weight_of_6_moles_Al2_CO3_3_l143_143000

noncomputable def molecular_weight_Al2_CO3_3: ℝ :=
  let Al_weight := 26.98
  let C_weight := 12.01
  let O_weight := 16.00
  let CO3_weight := C_weight + 3 * O_weight
  let one_mole_weight := 2 * Al_weight + 3 * CO3_weight
  6 * one_mole_weight

theorem molecular_weight_of_6_moles_Al2_CO3_3 : 
  molecular_weight_Al2_CO3_3 = 1403.94 :=
by
  sorry

end molecular_weight_of_6_moles_Al2_CO3_3_l143_143000


namespace calculate_molecular_weight_CaBr2_l143_143644

def atomic_weight_Ca : ℝ := 40.08                 -- The atomic weight of calcium (Ca)
def atomic_weight_Br : ℝ := 79.904                -- The atomic weight of bromine (Br)
def molecular_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br  -- Definition of molecular weight of CaBr₂

theorem calculate_molecular_weight_CaBr2 : molecular_weight_CaBr2 = 199.888 := by
  sorry

end calculate_molecular_weight_CaBr2_l143_143644


namespace quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l143_143027

theorem quadratic_eq_real_roots_m_ge_neg1 (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0) →
  m ≥ -1 :=
sorry

theorem quadratic_eq_real_roots_cond (m : ℝ) (x1 x2 : ℝ) :
  x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0 ∧
  (x1 - x2)^2 = 16 - x1 * x2 →
  m = 1 :=
sorry

end quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l143_143027


namespace denominator_divisor_zero_l143_143703

theorem denominator_divisor_zero (n : ℕ) : n ≠ 0 → (∀ d, d ≠ 0 → d / n ≠ d / 0) :=
by
  sorry

end denominator_divisor_zero_l143_143703


namespace james_second_hour_distance_l143_143816

theorem james_second_hour_distance :
  ∃ x : ℝ, 
    x + 1.20 * x + 1.50 * x = 37 ∧ 
    1.20 * x = 12 :=
by
  sorry

end james_second_hour_distance_l143_143816


namespace largest_angle_90_degrees_l143_143753

def triangle_altitudes (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (9 * a = 12 * b) ∧ (9 * a = 18 * c)

theorem largest_angle_90_degrees (a b c : ℝ) 
  (h : triangle_altitudes a b c) : 
  exists (A B C : ℝ) (hApos : A > 0) (hBpos : B > 0) (hCpos : C > 0),
    (A^2 = B^2 + C^2) ∧ (B * C / 2 = 9 * a / 2 ∨ 
                         B * A / 2 = 12 * b / 2 ∨ 
                         C * A / 2 = 18 * c / 2) :=
sorry

end largest_angle_90_degrees_l143_143753


namespace intersection_of_M_and_N_l143_143062

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 5}
noncomputable def N : Set ℝ := {x | x * (x - 4) > 0}

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5) } := by
  sorry

end intersection_of_M_and_N_l143_143062


namespace distributor_profit_percentage_l143_143098

theorem distributor_profit_percentage 
    (commission_rate : ℝ) (cost_price : ℝ) (final_price : ℝ) (P : ℝ) (profit : ℝ) 
    (profit_percentage: ℝ) :
  commission_rate = 0.20 →
  cost_price = 15 →
  final_price = 19.8 →
  0.80 * P = final_price →
  P = cost_price + profit →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage = 65 :=
by
  intros h_commission_rate h_cost_price h_final_price h_equation h_profit_eq h_percent_eq
  sorry

end distributor_profit_percentage_l143_143098


namespace automobile_distance_2_minutes_l143_143342

theorem automobile_distance_2_minutes (a : ℝ) :
  let acceleration := a / 12
  let time_minutes := 2
  let time_seconds := time_minutes * 60
  let distance_feet := (1 / 2) * acceleration * time_seconds^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := 
by sorry

end automobile_distance_2_minutes_l143_143342


namespace coprime_divisors_imply_product_divisor_l143_143893

theorem coprime_divisors_imply_product_divisor 
  (a b n : ℕ) (h_coprime : Nat.gcd a b = 1)
  (h_a_div_n : a ∣ n) (h_b_div_n : b ∣ n) : a * b ∣ n :=
by
  sorry

end coprime_divisors_imply_product_divisor_l143_143893


namespace only_n_equal_one_l143_143255

theorem only_n_equal_one (n : ℕ) (hn : 0 < n) : 
  (5 ^ (n - 1) + 3 ^ (n - 1)) ∣ (5 ^ n + 3 ^ n) → n = 1 := by
  intro h_div
  sorry

end only_n_equal_one_l143_143255


namespace total_gallons_needed_l143_143841

def gas_can_capacity : ℝ := 5.0
def number_of_cans : ℝ := 4.0
def total_gallons_of_gas : ℝ := gas_can_capacity * number_of_cans

theorem total_gallons_needed : total_gallons_of_gas = 20.0 := by
  -- proof goes here
  sorry

end total_gallons_needed_l143_143841


namespace two_a_minus_b_values_l143_143019

theorem two_a_minus_b_values (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 5) (h3 : |a + b| = -(a + b)) :
  (2 * a - b = 13) ∨ (2 * a - b = -3) :=
sorry

end two_a_minus_b_values_l143_143019


namespace find_constants_l143_143810

open Nat

variables {n : ℕ} (b c : ℤ)
def S (n : ℕ) := n^2 + b * n + c
def a (n : ℕ) := S n - S (n - 1)

theorem find_constants (a2a3_sum_eq_4 : a 2 + a 3 = 4) : 
  c = 0 ∧ b = -2 := 
by 
  sorry

end find_constants_l143_143810


namespace right_angled_triangle_sets_l143_143124

theorem right_angled_triangle_sets :
  (¬ (1 ^ 2 + 2 ^ 2 = 3 ^ 2)) ∧
  (¬ (2 ^ 2 + 3 ^ 2 = 4 ^ 2)) ∧
  (3 ^ 2 + 4 ^ 2 = 5 ^ 2) ∧
  (¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2)) :=
by
  sorry

end right_angled_triangle_sets_l143_143124


namespace find_g2_l143_143352

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1)
noncomputable def g (f : ℝ → ℝ) (y : ℝ) : ℝ := f⁻¹ y

variable (a : ℝ)
variable (h_inv : ∀ (x : ℝ), g (f a) (f a x) = x)
variable (h_g4 : g (f a) 4 = 2)

theorem find_g2 : g (f a) 2 = 3 / 2 :=
by sorry

end find_g2_l143_143352


namespace area_isosceles_right_triangle_l143_143767

open Real

-- Define the condition that the hypotenuse of an isosceles right triangle is 4√2 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = (4 * sqrt 2)^2

-- State the theorem to prove the area of the triangle is 8 square units
theorem area_isosceles_right_triangle (a b : ℝ) (h : hypotenuse a b) : 
  a = b → 1/2 * a * b = 8 := 
by 
  intros
  -- Proof steps are not required, so we use 'sorry'
  sorry

end area_isosceles_right_triangle_l143_143767


namespace problem1_problem2_l143_143193

section
variables (x a : ℝ)

-- Problem 1: Prove \(2^{3x-1} < 2 \implies x < \frac{2}{3}\)
theorem problem1 : (2:ℝ)^(3*x-1) < 2 → x < (2:ℝ)/3 :=
by sorry

-- Problem 2: Prove \(a^{3x^2+3x-1} < a^{3x^2+3} \implies (a > 1 \implies x < \frac{4}{3}) \land (0 < a < 1 \implies x > \frac{4}{3})\) given \(a > 0\) and \(a \neq 1\)
theorem problem2 (h0 : a > 0) (h1 : a ≠ 1) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) →
  ((1 < a → x < (4:ℝ)/3) ∧ (0 < a ∧ a < 1 → x > (4:ℝ)/3)) :=
by sorry
end

end problem1_problem2_l143_143193


namespace expression_evaluation_l143_143332

theorem expression_evaluation : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end expression_evaluation_l143_143332


namespace minimum_daily_production_to_avoid_losses_l143_143734

theorem minimum_daily_production_to_avoid_losses (x : ℕ) :
  (∀ x, (10 * x) ≥ (5 * x + 4000)) → (x ≥ 800) :=
sorry

end minimum_daily_production_to_avoid_losses_l143_143734


namespace not_divisible_l143_143944

-- Defining the necessary conditions
variable (m : ℕ)

theorem not_divisible (m : ℕ) : ¬ (1000^m - 1 ∣ 1978^m - 1) :=
sorry

end not_divisible_l143_143944


namespace speed_of_stream_l143_143781

variable (b s : ℝ)

-- Conditions:
def downstream_eq : Prop := 90 = (b + s) * 3
def upstream_eq : Prop := 72 = (b - s) * 3

-- Goal:
theorem speed_of_stream (h1 : downstream_eq b s) (h2 : upstream_eq b s) : s = 3 :=
by
  sorry

end speed_of_stream_l143_143781


namespace intersection_A_B_l143_143376

-- Define sets A and B based on given conditions
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- Prove the intersection of A and B equals (2,4)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 4} := 
by
  sorry

end intersection_A_B_l143_143376


namespace power_sum_l143_143484

theorem power_sum (n : ℕ) : (-2 : ℤ)^n + (-2 : ℤ)^(n+1) = 2^n := by
  sorry

end power_sum_l143_143484


namespace value_of_k_l143_143711

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l143_143711


namespace average_students_l143_143656

def ClassGiraffe : ℕ := 225

def ClassElephant (giraffe: ℕ) : ℕ := giraffe + 48

def ClassRabbit (giraffe: ℕ) : ℕ := giraffe - 24

theorem average_students (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) :
  giraffe = 225 → elephant = giraffe + 48 → rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 := by
  sorry

end average_students_l143_143656


namespace tangent_y_intercept_l143_143302

theorem tangent_y_intercept :
  let C1 := (2, 4)
  let r1 := 5
  let C2 := (14, 9)
  let r2 := 10
  let m := 120 / 119
  m > 0 → ∃ b, b = 912 / 119 := by
  sorry

end tangent_y_intercept_l143_143302


namespace game_ends_after_63_rounds_l143_143301

-- Define tokens for players A, B, C, and D at the start
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16
def initial_tokens_D := 14

-- Define the rules of the game
def game_rounds_to_end (A B C D : ℕ) : ℕ :=
  -- This function calculates the number of rounds after which any player runs out of tokens
  if (A, B, C, D) = (20, 18, 16, 14) then 63 else 0

-- Statement to prove
theorem game_ends_after_63_rounds :
  game_rounds_to_end initial_tokens_A initial_tokens_B initial_tokens_C initial_tokens_D = 63 :=
by sorry

end game_ends_after_63_rounds_l143_143301


namespace find_square_l143_143886

theorem find_square (s : ℕ) : 
    (7863 / 13 = 604 + (s / 13)) → s = 11 :=
by
  sorry

end find_square_l143_143886


namespace find_f_pi_l143_143972

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.tan (ω * x + Real.pi / 3)

theorem find_f_pi (ω : ℝ) (h_positive : ω > 0) (h_period : Real.pi / ω = 3 * Real.pi) :
  f (ω := ω) Real.pi = -Real.sqrt 3 :=
by
  -- ω is given to be 1/3 by the condition h_period, substituting that 
  -- directly might be clearer for stating the problem accurately
  have h_omega : ω = 1 / 3 := by
    sorry
  rw [h_omega]
  sorry


end find_f_pi_l143_143972


namespace cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l143_143206

theorem cube_root_of_4913_has_unit_digit_7 :
  (∃ (y : ℕ), y^3 = 4913 ∧ y % 10 = 7) :=
sorry

theorem cube_root_of_50653_is_37 :
  (∃ (y : ℕ), y = 37 ∧ y^3 = 50653) :=
sorry

theorem cube_root_of_110592_is_48 :
  (∃ (y : ℕ), y = 48 ∧ y^3 = 110592) :=
sorry

end cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l143_143206


namespace incorrect_proposition_b_l143_143909

axiom plane (α β : Type) : Prop
axiom line (m n : Type) : Prop
axiom parallel (a b : Type) : Prop
axiom perpendicular (a b : Type) : Prop
axiom intersection (α β : Type) (n : Type) : Prop
axiom contained (a b : Type) : Prop

theorem incorrect_proposition_b (α β m n : Type)
  (hαβ_plane : plane α β)
  (hmn_line : line m n)
  (h_parallel_m_α : parallel m α)
  (h_intersection : intersection α β n) :
  ¬ parallel m n :=
sorry

end incorrect_proposition_b_l143_143909


namespace min_value_l143_143732

open Real

theorem min_value (x y : ℝ) (h : x + y = 4) : x^2 + y^2 ≥ 8 := by
  sorry

end min_value_l143_143732


namespace smallest_number_of_slices_l143_143032

-- Definition of the number of slices in each type of cheese package
def slices_of_cheddar : ℕ := 12
def slices_of_swiss : ℕ := 28

-- Predicate stating that the smallest number of slices of each type Randy could have bought is 84
theorem smallest_number_of_slices : Nat.lcm slices_of_cheddar slices_of_swiss = 84 := by
  sorry

end smallest_number_of_slices_l143_143032


namespace horse_revolutions_l143_143647

theorem horse_revolutions (r1 r2 r3 : ℝ) (rev1 : ℕ) 
  (h1 : r1 = 30) (h2 : r2 = 15) (h3 : r3 = 10) (h4 : rev1 = 40) :
  (r2 / r1 = 1 / 2 ∧ 2 * rev1 = 80) ∧ (r3 / r1 = 1 / 3 ∧ 3 * rev1 = 120) :=
by
  sorry

end horse_revolutions_l143_143647


namespace find_sequence_l143_143660

noncomputable def sequence_satisfies (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (1 / 2) * (a n + 1 / (a n))

theorem find_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h_pos : ∀ n, 0 < a n)
    (h_S : sequence_satisfies a S) :
    ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
sorry

end find_sequence_l143_143660


namespace harry_sandy_meet_point_l143_143925

theorem harry_sandy_meet_point :
  let H : ℝ × ℝ := (10, -3)
  let S : ℝ × ℝ := (2, 7)
  let t : ℝ := 2 / 3
  let meet_point : ℝ × ℝ := (H.1 + t * (S.1 - H.1), H.2 + t * (S.2 - H.2))
  meet_point = (14 / 3, 11 / 3) := 
by
  sorry

end harry_sandy_meet_point_l143_143925


namespace total_games_played_l143_143435

def games_attended : ℕ := 14
def games_missed : ℕ := 25

theorem total_games_played : games_attended + games_missed = 39 :=
by
  sorry

end total_games_played_l143_143435


namespace trapezoid_area_l143_143984

theorem trapezoid_area (base1 base2 height : ℕ) (h_base1 : base1 = 9) (h_base2 : base2 = 11) (h_height : height = 3) :
  (1 / 2 : ℚ) * (base1 + base2 : ℕ) * height = 30 :=
by
  sorry

end trapezoid_area_l143_143984


namespace ratio_of_squares_l143_143763

theorem ratio_of_squares (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + 2 * y + 3 * z = 0) :
    (x^2 + y^2 + z^2) / (x * y + y * z + z * x) = -4 := by
  sorry

end ratio_of_squares_l143_143763


namespace max_value_neg_domain_l143_143839

theorem max_value_neg_domain (x : ℝ) (h : x < 0) : 
  ∃ y, y = 2 * x + 2 / x ∧ y ≤ -4 :=
sorry

end max_value_neg_domain_l143_143839


namespace shortest_distance_correct_l143_143475

noncomputable def shortest_distance_a_to_c1 (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c)

theorem shortest_distance_correct (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  shortest_distance_a_to_c1 a b c h₁ h₂ = Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c) :=
by
  -- This is where the proof would go.
  sorry

end shortest_distance_correct_l143_143475


namespace find_x_l143_143508

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 135) (h2 : x > 0) : x = 11.25 :=
by
  sorry

end find_x_l143_143508


namespace girl_scout_cookie_sales_l143_143349

theorem girl_scout_cookie_sales :
  ∃ C P : ℝ, C + P = 1585 ∧ 1.25 * C + 0.75 * P = 1586.25 ∧ P = 790 :=
by
  sorry

end girl_scout_cookie_sales_l143_143349


namespace hose_removal_rate_l143_143247

def pool_volume (length width depth : ℕ) : ℕ :=
  length * width * depth

def draining_rate (volume time : ℕ) : ℕ :=
  volume / time

theorem hose_removal_rate :
  let length := 150
  let width := 80
  let depth := 10
  let total_volume := pool_volume length width depth
  total_volume = 1200000 ∧
  let time := 2000
  draining_rate total_volume time = 600 :=
by
  sorry

end hose_removal_rate_l143_143247


namespace triangle_length_AX_l143_143990

theorem triangle_length_AX (A B C X : Type*) (AB AC BC AX XB : ℝ)
  (hAB : AB = 70) (hAC : AC = 42) (hBC : BC = 56)
  (h_bisect : ∃ (k : ℝ), AX = 3 * k ∧ XB = 4 * k) :
  AX = 30 := 
by
  sorry

end triangle_length_AX_l143_143990


namespace necessary_sufficient_condition_l143_143169

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
by
  sorry

end necessary_sufficient_condition_l143_143169


namespace abc_ineq_l143_143250

theorem abc_ineq (a b c : ℝ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : c > 0) (h₄ : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27 / 8 :=
sorry

end abc_ineq_l143_143250


namespace largest_gcd_sum_1071_l143_143277

theorem largest_gcd_sum_1071 (x y: ℕ) (h1: x > 0) (h2: y > 0) (h3: x + y = 1071) : 
  ∃ d, d = Nat.gcd x y ∧ ∀ z, (z ∣ 1071 -> z ≤ d) := 
sorry

end largest_gcd_sum_1071_l143_143277


namespace inradius_of_right_triangle_l143_143030

-- Define the side lengths of the triangle
def a : ℕ := 9
def b : ℕ := 40
def c : ℕ := 41

-- Define the semiperimeter of the triangle
def s : ℕ := (a + b + c) / 2

-- Define the area of a right triangle
def A : ℕ := (a * b) / 2

-- Define the inradius of the triangle
def inradius : ℕ := A / s

theorem inradius_of_right_triangle : inradius = 4 :=
by
  -- The proof is omitted since only the statement is requested
  sorry

end inradius_of_right_triangle_l143_143030


namespace lcm_of_1_to_12_l143_143995

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l143_143995


namespace tan_of_13pi_over_6_l143_143773

theorem tan_of_13pi_over_6 : Real.tan (13 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_of_13pi_over_6_l143_143773


namespace smallest_possible_value_of_M_l143_143230

theorem smallest_possible_value_of_M :
  ∃ (N M : ℕ), N > 0 ∧ M > 0 ∧ 
               ∃ (r_6 r_36 r_216 r_M : ℕ), 
               r_6 < 6 ∧ 
               r_6 < r_36 ∧ r_36 < 36 ∧ 
               r_36 < r_216 ∧ r_216 < 216 ∧ 
               r_216 < r_M ∧ 
               r_36 = (r_6 * r) ∧ 
               r_216 = (r_6 * r^2) ∧ 
               r_M = (r_6 * r^3) ∧ 
               Nat.mod N 6 = r_6 ∧ 
               Nat.mod N 36 = r_36 ∧ 
               Nat.mod N 216 = r_216 ∧ 
               Nat.mod N M = r_M ∧ 
               M = 2001 :=
sorry

end smallest_possible_value_of_M_l143_143230


namespace solve_for_m_l143_143556

theorem solve_for_m (m : ℝ) :
  (1 * m + (3 + m) * 2 = 0) → m = -2 :=
by
  sorry

end solve_for_m_l143_143556


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l143_143216

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l143_143216


namespace four_points_no_obtuse_triangle_l143_143036

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l143_143036


namespace no_valid_two_digit_factors_l143_143864

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Main theorem to show: there are no valid two-digit factorizations of 1976
theorem no_valid_two_digit_factors : 
  ∃ (factors : ℕ → ℕ → Prop), (∀ (a b : ℕ), factors a b → (a * b = 1976) → (is_two_digit a) → (is_two_digit b)) → 
  ∃ (count : ℕ), count = 0 := 
sorry

end no_valid_two_digit_factors_l143_143864


namespace rectangle_dimensions_l143_143927

theorem rectangle_dimensions (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w * l) = 2 * (2 * w + w)) :
  w = 6 ∧ l = 12 := 
by sorry

end rectangle_dimensions_l143_143927


namespace children_neither_happy_nor_sad_l143_143306

-- conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10

-- proof problem
theorem children_neither_happy_nor_sad :
  total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l143_143306


namespace syllogism_sequence_l143_143833

theorem syllogism_sequence (P Q R : Prop)
  (h1 : R)
  (h2 : Q)
  (h3 : P) : 
  (Q ∧ R → P) → (R → P) ∧ (Q → (P ∧ R)) := 
by
  sorry

end syllogism_sequence_l143_143833


namespace exam_percentage_l143_143653

theorem exam_percentage (x : ℝ) (h_cond : 100 - x >= 0 ∧ x >= 0 ∧ 60 * x + 90 * (100 - x) = 69 * 100) : x = 70 := by
  sorry

end exam_percentage_l143_143653


namespace probability_of_two_same_color_l143_143691

noncomputable def probability_at_least_two_same_color (reds whites blues greens : ℕ) (total_draws : ℕ) : ℚ :=
  have total_marbles := reds + whites + blues + greens
  let total_combinations := Nat.choose total_marbles total_draws
  let two_reds := Nat.choose reds 2 * (total_marbles - 2)
  let two_whites := Nat.choose whites 2 * (total_marbles - 2)
  let two_blues := Nat.choose blues 2 * (total_marbles - 2)
  let two_greens := Nat.choose greens 2 * (total_marbles - 2)
  
  let all_reds := Nat.choose reds 3
  let all_whites := Nat.choose whites 3
  let all_blues := Nat.choose blues 3
  let all_greens := Nat.choose greens 3
  
  let desired_outcomes := two_reds + two_whites + two_blues + two_greens +
                          all_reds + all_whites + all_blues + all_greens
                          
  (desired_outcomes : ℚ) / (total_combinations : ℚ)

theorem probability_of_two_same_color : probability_at_least_two_same_color 6 7 8 4 3 = 69 / 115 := 
by
  sorry

end probability_of_two_same_color_l143_143691


namespace first_caller_to_win_all_prizes_is_900_l143_143284

-- Define the conditions: frequencies of win types
def every_25th_caller_wins_music_player (n : ℕ) : Prop := n % 25 = 0
def every_36th_caller_wins_concert_tickets (n : ℕ) : Prop := n % 36 = 0
def every_45th_caller_wins_backstage_passes (n : ℕ) : Prop := n % 45 = 0

-- Formalize the problem to prove
theorem first_caller_to_win_all_prizes_is_900 :
  ∃ n : ℕ, every_25th_caller_wins_music_player n ∧
           every_36th_caller_wins_concert_tickets n ∧
           every_45th_caller_wins_backstage_passes n ∧
           n = 900 :=
by {
  sorry
}

end first_caller_to_win_all_prizes_is_900_l143_143284


namespace solve_the_problem_l143_143288

noncomputable def solve_problem : Prop :=
  ∀ (θ t α : ℝ),
    (∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = 4 * Real.sin θ) → 
    (∃ x y : ℝ, x = 1 + t * Real.cos α ∧ y = 2 + t * Real.sin α) →
    (∃ m n : ℝ, m = 1 ∧ n = 2) →
    (-2 = Real.tan α)

theorem solve_the_problem : solve_problem := by
  sorry

end solve_the_problem_l143_143288


namespace area_of_triangle_PQR_is_correct_l143_143325

noncomputable def calculate_area_of_triangle_PQR : ℝ := 
  let side_length := 4
  let altitude := 8
  let WO := (side_length * Real.sqrt 2) / 2
  let center_to_vertex_distance := Real.sqrt (WO^2 + altitude^2)
  let WP := (1 / 4) * WO
  let YQ := (1 / 2) * WO
  let XR := (3 / 4) * WO
  let area := (1 / 2) * (YQ - WP) * (XR - YQ)
  area

theorem area_of_triangle_PQR_is_correct :
  calculate_area_of_triangle_PQR = 2.25 := sorry

end area_of_triangle_PQR_is_correct_l143_143325


namespace largest_integer_value_l143_143233

theorem largest_integer_value (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 2/3 ∧ (x : ℚ) < 10 → x = 3 := 
by
  sorry

end largest_integer_value_l143_143233


namespace solve_r_l143_143246

-- Define E(a, b, c) as given
def E (a b c : ℕ) : ℕ := a * b^c

-- Lean 4 statement for the proof
theorem solve_r (r : ℕ) (r_pos : 0 < r) : E r r 3 = 625 → r = 5 :=
by
  intro h
  sorry

end solve_r_l143_143246


namespace solve_inequality_l143_143138

theorem solve_inequality :
  {x : ℝ | x ∈ { y | (y^2 - 5*y + 6) / (y - 3)^2 > 0 }} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solve_inequality_l143_143138


namespace roots_in_interval_l143_143252

theorem roots_in_interval (a b : ℝ) (hb : b > 0) (h_discriminant : a^2 - 4 * b > 0)
  (h_root_interval : ∃ r1 r2 : ℝ, r1 + r2 = -a ∧ r1 * r2 = b ∧ ((-1 ≤ r1 ∧ r1 ≤ 1 ∧ (r2 < -1 ∨ 1 < r2)) ∨ (-1 ≤ r2 ∧ r2 ≤ 1 ∧ (r1 < -1 ∨ 1 < r1)))) : 
  ∃ r : ℝ, (r + a) * r + b = 0 ∧ -b < r ∧ r < b :=
by
  sorry

end roots_in_interval_l143_143252


namespace sum_of_base4_numbers_is_correct_l143_143503

-- Define the four base numbers
def n1 : ℕ := 2 * 4^2 + 1 * 4^1 + 2 * 4^0
def n2 : ℕ := 1 * 4^2 + 0 * 4^1 + 3 * 4^0
def n3 : ℕ := 3 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Define the expected sum in base 4 interpreted as a natural number
def expected_sum : ℕ := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

-- State the theorem
theorem sum_of_base4_numbers_is_correct : n1 + n2 + n3 = expected_sum := by
  sorry

end sum_of_base4_numbers_is_correct_l143_143503


namespace unique_nat_number_sum_preceding_eq_self_l143_143434

theorem unique_nat_number_sum_preceding_eq_self :
  ∃! (n : ℕ), (n * (n - 1)) / 2 = n :=
sorry

end unique_nat_number_sum_preceding_eq_self_l143_143434


namespace competition_scores_order_l143_143189

theorem competition_scores_order (A B C D : ℕ) (h1 : A + B = C + D) (h2 : C + A > D + B) (h3 : B > A + D) : (B > A) ∧ (A > C) ∧ (C > D) := 
by 
  sorry

end competition_scores_order_l143_143189


namespace ramesh_paid_price_l143_143089

variable (P : ℝ) (P_paid : ℝ)

-- conditions
def discount_price (P : ℝ) : ℝ := 0.80 * P
def additional_cost : ℝ := 125 + 250
def total_cost_with_discount (P : ℝ) : ℝ := discount_price P + additional_cost
def selling_price_without_discount (P : ℝ) : ℝ := 1.10 * P
def given_selling_price : ℝ := 18975

-- the theorem to prove
theorem ramesh_paid_price :
  (∃ P : ℝ, selling_price_without_discount P = given_selling_price ∧ total_cost_with_discount P = 14175) :=
by
  sorry

end ramesh_paid_price_l143_143089


namespace time_to_fill_tank_l143_143409

theorem time_to_fill_tank (T : ℝ) :
  (1 / 2 * T) + ((1 / 2 * T) / 4) = 10 → T = 16 :=
by { sorry }

end time_to_fill_tank_l143_143409


namespace trapezium_area_l143_143341

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l143_143341


namespace determine_c_for_inverse_l143_143313

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_c_for_inverse :
  (∀ x : ℝ, x ≠ 0 → f (f_inv x) c = x) ↔ c = 1 :=
sorry

end determine_c_for_inverse_l143_143313


namespace max_hours_at_regular_rate_l143_143512

-- Define the maximum hours at regular rate H
def max_regular_hours (H : ℕ) : Prop := 
  let regular_rate := 16
  let overtime_rate := 16 + (0.75 * 16)
  let total_hours := 60
  let total_compensation := 1200
  16 * H + 28 * (total_hours - H) = total_compensation

theorem max_hours_at_regular_rate : ∃ H, max_regular_hours H ∧ H = 40 :=
sorry

end max_hours_at_regular_rate_l143_143512


namespace arithmetic_sequence_common_difference_l143_143604

-- Arithmetic sequence with condition and proof of common difference
theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) :
  (a 2015 = a 2013 + 6) → ((a 2015 - a 2013) = 2 * d) → (d = 3) :=
by
  intro h1 h2
  sorry

end arithmetic_sequence_common_difference_l143_143604


namespace fraction_value_l143_143188

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l143_143188


namespace best_fit_slope_eq_l143_143952

theorem best_fit_slope_eq :
  let x1 := 150
  let y1 := 2
  let x2 := 160
  let y2 := 3
  let x3 := 170
  let y3 := 4
  (x2 - x1 = 10 ∧ x3 - x2 = 10) →
  let slope := (x1 - x2) * (y1 - y2) + (x3 - x2) * (y3 - y2) / (x1 - x2)^2 + (x3 - x2)^2
  slope = 1 / 10 :=
sorry

end best_fit_slope_eq_l143_143952


namespace min_value_f_in_interval_l143_143547

def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 1

theorem min_value_f_in_interval : 
  ∃ x ∈ (Set.Icc (-1 : ℝ) 1), f x = -1 :=
by
  sorry


end min_value_f_in_interval_l143_143547


namespace area_inequality_equality_condition_l143_143059

variable (a b c d S : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
variable (s : ℝ) (h5 : s = (a + b + c + d) / 2)
variable (h6 : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)))

theorem area_inequality (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  S ≤ Real.sqrt (a * b * c * d) :=
sorry

theorem equality_condition (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  (S = Real.sqrt (a * b * c * d)) ↔ (a = c ∧ b = d ∨ a = d ∧ b = c) :=
sorry

end area_inequality_equality_condition_l143_143059


namespace quadratic_roots_range_l143_143480

theorem quadratic_roots_range (m : ℝ) :
  (∃ p n : ℝ, p > 0 ∧ n < 0 ∧ 2 * p^2 + (m + 1) * p + m = 0 ∧ 2 * n^2 + (m + 1) * n + m = 0) →
  m < 0 :=
by
  sorry

end quadratic_roots_range_l143_143480


namespace geometric_sequence_fourth_term_l143_143996

theorem geometric_sequence_fourth_term (a r T4 : ℝ)
  (h1 : a = 1024)
  (h2 : a * r^5 = 32)
  (h3 : T4 = a * r^3) :
  T4 = 128 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l143_143996


namespace cube_difference_divisible_by_16_l143_143360

theorem cube_difference_divisible_by_16 (a b : ℤ) : 
  16 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3 + 8) :=
by
  sorry

end cube_difference_divisible_by_16_l143_143360


namespace substitution_correct_l143_143037

theorem substitution_correct (x y : ℝ) (h1 : y = x - 1) (h2 : x - 2 * y = 7) :
  x - 2 * x + 2 = 7 :=
by
  sorry

end substitution_correct_l143_143037


namespace max_price_of_product_l143_143249

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end max_price_of_product_l143_143249


namespace train_speed_correct_l143_143128

noncomputable def train_speed_kmh (length : ℝ) (time : ℝ) (conversion_factor : ℝ) : ℝ :=
  (length / time) * conversion_factor

theorem train_speed_correct 
  (length : ℝ := 350) 
  (time : ℝ := 8.7493) 
  (conversion_factor : ℝ := 3.6) : 
  train_speed_kmh length time conversion_factor = 144.02 := 
sorry

end train_speed_correct_l143_143128


namespace max_andy_l143_143505

def max_cookies_eaten_by_andy (total : ℕ) (k1 k2 a b c : ℤ) : Prop :=
  a + b + c = total ∧ b = 2 * a + 2 ∧ c = a - 3

theorem max_andy (total : ℕ) (a : ℤ) :
  (∀ b c, max_cookies_eaten_by_andy total 2 (-3) a b c) → a ≤ 7 :=
by
  intros H
  sorry

end max_andy_l143_143505


namespace num_five_digit_integers_l143_143001

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers_l143_143001


namespace goshawk_eurasian_reserve_l143_143938

theorem goshawk_eurasian_reserve (B : ℝ)
  (h1 : 0.30 * B + 0.28 * B + K * 0.28 * B = 0.65 * B)
  : K = 0.25 :=
by sorry

end goshawk_eurasian_reserve_l143_143938


namespace negation_proposition_correct_l143_143370

theorem negation_proposition_correct : 
  (∀ x : ℝ, 0 < x → x + 4 / x ≥ 4) :=
by
  intro x hx
  sorry

end negation_proposition_correct_l143_143370


namespace jane_earnings_two_weeks_l143_143448

def num_chickens : ℕ := 10
def num_eggs_per_chicken_per_week : ℕ := 6
def dollars_per_dozen : ℕ := 2
def dozens_in_12_eggs : ℕ := 12

theorem jane_earnings_two_weeks :
  (num_chickens * num_eggs_per_chicken_per_week * 2 / dozens_in_12_eggs * dollars_per_dozen) = 20 := by
  sorry

end jane_earnings_two_weeks_l143_143448


namespace line_intersects_circle_two_points_find_value_of_m_l143_143406

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem line_intersects_circle_two_points (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ),
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  x1 ≠ x2 ∨ y1 ≠ y2 := sorry

theorem find_value_of_m (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ), 
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  dist (x1, y1) (x2, y2) = sqrt 17 → 
  m = sqrt 3 ∨ m = -sqrt 3 := sorry

end line_intersects_circle_two_points_find_value_of_m_l143_143406


namespace volume_range_of_rectangular_solid_l143_143616

theorem volume_range_of_rectangular_solid
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) :
  (16 : ℝ) ≤ a * b * c ∧ a * b * c ≤ 20 :=
by sorry

end volume_range_of_rectangular_solid_l143_143616


namespace ravi_first_has_more_than_500_paperclips_on_wednesday_l143_143601

noncomputable def paperclips (k : Nat) : Nat :=
  5 * 4^k

theorem ravi_first_has_more_than_500_paperclips_on_wednesday :
  ∃ k : Nat, paperclips k > 500 ∧ k = 3 :=
by
  sorry

end ravi_first_has_more_than_500_paperclips_on_wednesday_l143_143601


namespace value_of_c7_l143_143722

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l143_143722


namespace necessary_but_not_sufficient_cond_l143_143685

open Set

variable {α : Type*} (A B C : Set α)

/-- Mathematical equivalent proof problem statement -/
theorem necessary_but_not_sufficient_cond (h1 : A ∪ B = C) (h2 : ¬ B ⊆ A) (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ y ∈ C, y ∉ A) :=
by
  sorry

end necessary_but_not_sufficient_cond_l143_143685


namespace other_root_of_quadratic_l143_143071

theorem other_root_of_quadratic (m : ℝ) (h : (m + 2) * 0^2 - 0 + m^2 - 4 = 0) : 
  ∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x ≠ 0 ∧ x = 1/4 := 
sorry

end other_root_of_quadratic_l143_143071


namespace solve_a_l143_143574

-- Defining sets A and B
def set_A (a : ℤ) : Set ℤ := {a^2, a + 1, -3}
def set_B (a : ℤ) : Set ℤ := {a - 3, 2 * a - 1, a^2 + 1}

-- Defining the condition of intersection
def intersection_condition (a : ℤ) : Prop :=
  (set_A a) ∩ (set_B a) = {-3}

-- Stating the theorem
theorem solve_a (a : ℤ) (h : intersection_condition a) : a = -1 :=
sorry

end solve_a_l143_143574


namespace sin_cos_solution_set_l143_143671
open Real

theorem sin_cos_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * π + (-1)^k * (π / 6) - (π / 3)} =
  {x : ℝ | sin x + sqrt 3 * cos x = 1} :=
by sorry

end sin_cos_solution_set_l143_143671


namespace select_best_athlete_l143_143265

theorem select_best_athlete :
  let avg_A := 185
  let var_A := 3.6
  let avg_B := 180
  let var_B := 3.6
  let avg_C := 185
  let var_C := 7.4
  let avg_D := 180
  let var_D := 8.1
  avg_A = 185 ∧ var_A = 3.6 ∧
  avg_B = 180 ∧ var_B = 3.6 ∧
  avg_C = 185 ∧ var_C = 7.4 ∧
  avg_D = 180 ∧ var_D = 8.1 →
  (∃ x, (x = avg_A ∧ avg_A = 185 ∧ var_A = 3.6) ∧
        (∀ (y : ℕ), (y = avg_A) 
        → avg_A = 185 
        ∧ var_A <= var_C ∧ 
        var_A <= var_D 
        ∧ var_A <= var_B)) :=
by {
  sorry
}

end select_best_athlete_l143_143265


namespace general_solution_of_differential_equation_l143_143424

theorem general_solution_of_differential_equation (a₀ : ℝ) (x : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x = (y x)^2) ∧ y x = a₀ / (1 - a₀ * x) :=
sorry

end general_solution_of_differential_equation_l143_143424


namespace contradiction_method_example_l143_143621

variables {a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a + b + c > 0) (h5 : ab + bc + ca > 0)
variables (h6 : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))

theorem contradiction_method_example : false :=
by {
  sorry
}

end contradiction_method_example_l143_143621


namespace original_solution_percentage_l143_143805

theorem original_solution_percentage (P : ℝ) (h1 : 0.5 * P + 0.5 * 30 = 40) : P = 50 :=
by
  sorry

end original_solution_percentage_l143_143805


namespace system_solution_l143_143121

theorem system_solution (u v w : ℚ) 
  (h1 : 3 * u - 4 * v + w = 26)
  (h2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 :=
sorry

end system_solution_l143_143121


namespace dogs_food_consumption_l143_143548

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l143_143548


namespace new_avg_weight_l143_143427

-- Define the weights of individuals
variables (A B C D E : ℕ)
-- Conditions
axiom avg_ABC : (A + B + C) / 3 = 84
axiom avg_ABCD : (A + B + C + D) / 4 = 80
axiom E_def : E = D + 8
axiom A_80 : A = 80

theorem new_avg_weight (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 8) 
  (h4 : A = 80) 
  : (B + C + D + E) / 4 = 79 := 
by
  sorry

end new_avg_weight_l143_143427


namespace Pyarelal_loss_share_l143_143103

-- Define the conditions
variables (P : ℝ) (A : ℝ) (total_loss : ℝ)

-- Ashok's capital is 1/9 of Pyarelal's capital
axiom Ashok_capital : A = (1 / 9) * P

-- Total loss is Rs 900
axiom total_loss_val : total_loss = 900

-- Prove Pyarelal's share of the loss is Rs 810
theorem Pyarelal_loss_share : (P / (A + P)) * total_loss = 810 :=
by
  sorry

end Pyarelal_loss_share_l143_143103


namespace hh_of_2_eq_91265_l143_143689

def h (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - x + 1

theorem hh_of_2_eq_91265 : h (h 2) = 91265 := by
  sorry

end hh_of_2_eq_91265_l143_143689


namespace triangle_division_point_distances_l143_143011

theorem triangle_division_point_distances 
  {a b c : ℝ} 
  (h1 : a = 13) 
  (h2 : b = 17) 
  (h3 : c = 24)
  (h4 : ∃ p q : ℝ, p = 9 ∧ q = 11) : 
  ∃ p q : ℝ, p = 9 ∧ q = 11 :=
  sorry

end triangle_division_point_distances_l143_143011


namespace combined_weight_l143_143469

variables (G D C : ℝ)

def grandmother_weight (G D C : ℝ) := G + D + C = 150
def daughter_weight (D : ℝ) := D = 42
def child_weight (G C : ℝ) := C = 1/5 * G

theorem combined_weight (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_weight D) (h3 : child_weight G C) : D + C = 60 :=
by
  sorry

end combined_weight_l143_143469


namespace total_biscuits_l143_143545

-- Define the number of dogs and biscuits per dog
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem stating the total number of biscuits needed
theorem total_biscuits : num_dogs * biscuits_per_dog = 6 := by
  -- sorry to skip the proof
  sorry

end total_biscuits_l143_143545


namespace max_value_x3y2z_l143_143050

theorem max_value_x3y2z
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h_total : x + 2 * y + 3 * z = 1)
  : x^3 * y^2 * z ≤ 2048 / 11^6 := 
by
  sorry

end max_value_x3y2z_l143_143050


namespace calculation_result_l143_143029

theorem calculation_result :
  -Real.sqrt 4 + abs (-Real.sqrt 2 - 1) + (Real.pi - 2013) ^ 0 - (1/5) ^ 0 = Real.sqrt 2 - 1 :=
by
  sorry

end calculation_result_l143_143029


namespace masha_number_l143_143221

theorem masha_number (x : ℝ) (n : ℤ) (ε : ℝ) (h1 : 0 ≤ ε) (h2 : ε < 1) (h3 : x = n + ε) (h4 : (n : ℝ) = 0.57 * x) : x = 100 / 57 :=
by
  sorry

end masha_number_l143_143221


namespace mogs_and_mags_to_migs_l143_143150

theorem mogs_and_mags_to_migs:
  (∀ mags migs, 1 * mags = 8 * migs) ∧ 
  (∀ mogs mags, 1 * mogs = 6 * mags) → 
  10 * (6 * 8) + 6 * 8 = 528 := by 
  sorry

end mogs_and_mags_to_migs_l143_143150


namespace computation_is_correct_l143_143612

def large_multiplication : ℤ := 23457689 * 84736521

def denominator_subtraction : ℤ := 7589236 - 3145897

def computed_m : ℚ := large_multiplication / denominator_subtraction

theorem computation_is_correct : computed_m = 447214.999 :=
by 
  -- exact calculation to be provided
  sorry

end computation_is_correct_l143_143612


namespace sqrt_calculation_l143_143680

theorem sqrt_calculation : Real.sqrt ((5: ℝ)^2 - (4: ℝ)^2 - (3: ℝ)^2) = 0 := 
by
  -- The proof is skipped
  sorry

end sqrt_calculation_l143_143680


namespace chord_eq_line_l143_143919

theorem chord_eq_line (x y : ℝ)
  (h_ellipse : (x^2) / 16 + (y^2) / 4 = 1)
  (h_midpoint : ∃ x1 y1 x2 y2 : ℝ, 
    ((x1^2) / 16 + (y1^2) / 4 = 1) ∧ 
    ((x2^2) / 16 + (y2^2) / 4 = 1) ∧ 
    (x1 + x2) / 2 = 2 ∧ 
    (y1 + y2) / 2 = 1) :
  x + 2 * y - 4 = 0 :=
sorry

end chord_eq_line_l143_143919


namespace abs_x_plus_one_ge_one_l143_143518

theorem abs_x_plus_one_ge_one {x : ℝ} : |x + 1| ≥ 1 ↔ x ≤ -2 ∨ x ≥ 0 :=
by
  sorry

end abs_x_plus_one_ge_one_l143_143518


namespace shortest_distance_between_tracks_l143_143677

noncomputable def rational_man_track (x y : ℝ) : Prop :=
x^2 + y^2 = 1

noncomputable def irrational_man_track (x y : ℝ) : Prop :=
(x + 1)^2 + y^2 = 9

noncomputable def shortest_distance : ℝ :=
0

theorem shortest_distance_between_tracks :
  ∀ (A B : ℝ × ℝ), 
  rational_man_track A.1 A.2 → 
  irrational_man_track B.1 B.2 → 
  dist A B = shortest_distance := sorry

end shortest_distance_between_tracks_l143_143677


namespace width_at_bottom_l143_143205

-- Defining the given values and conditions
def top_width : ℝ := 14
def area : ℝ := 770
def depth : ℝ := 70

-- The proof problem
theorem width_at_bottom (b : ℝ) (h : area = (1/2) * (top_width + b) * depth) : b = 8 :=
by
  sorry

end width_at_bottom_l143_143205


namespace range_of_a_minus_abs_b_l143_143831

theorem range_of_a_minus_abs_b {a b : ℝ} (h1 : 1 < a ∧ a < 3) (h2 : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 :=
by
  sorry

end range_of_a_minus_abs_b_l143_143831


namespace flea_returns_to_0_l143_143945

noncomputable def flea_return_probability (p : ℝ) : ℝ :=
if p = 1 then 0 else 1

theorem flea_returns_to_0 (p : ℝ) : 
  flea_return_probability p = (if p = 1 then 0 else 1) :=
by
  sorry

end flea_returns_to_0_l143_143945


namespace population_net_change_l143_143310

theorem population_net_change
  (initial_population : ℝ)
  (year1_increase : initial_population * (6/5) = year1_population)
  (year2_increase : year1_population * (6/5) = year2_population)
  (year3_decrease : year2_population * (4/5) = year3_population)
  (year4_decrease : year3_population * (4/5) = final_population) :
  ((final_population - initial_population) / initial_population) * 100 = -8 :=
  sorry

end population_net_change_l143_143310


namespace twice_product_of_numbers_l143_143133

theorem twice_product_of_numbers (x y : ℝ) (h1 : x + y = 80) (h2 : x - y = 10) : 2 * (x * y) = 3150 := by
  sorry

end twice_product_of_numbers_l143_143133


namespace expression_undefined_count_l143_143054

theorem expression_undefined_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x : ℝ,
  ((x = x1 ∨ x = x2) ↔ (x^2 - 2*x - 3 = 0 ∨ x - 3 = 0)) ∧ 
  ((x^2 - 2*x - 3) * (x - 3) = 0 → (x = x1 ∨ x = x2)) :=
by
  sorry

end expression_undefined_count_l143_143054


namespace product_of_integers_l143_143292

theorem product_of_integers (a b : ℤ) (h1 : Int.gcd a b = 12) (h2 : Int.lcm a b = 60) : a * b = 720 :=
sorry

end product_of_integers_l143_143292


namespace sum_of_roots_of_quadratic_l143_143801

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l143_143801


namespace union_of_A_and_B_l143_143975

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem union_of_A_and_B :
  A ∪ B = {-1, 0, 1, 2, 4} :=
by
  sorry

end union_of_A_and_B_l143_143975


namespace product_of_hypotenuse_segments_eq_area_l143_143183

theorem product_of_hypotenuse_segments_eq_area (x y c t : ℝ) : 
  -- Conditions
  (c = x + y) → 
  (t = x * y) →
  -- Conclusion
  x * y = t :=
by
  intros
  sorry

end product_of_hypotenuse_segments_eq_area_l143_143183


namespace total_sand_donated_l143_143648

theorem total_sand_donated (A B C D: ℚ) (hA: A = 33 / 2) (hB: B = 26) (hC: C = 49 / 2) (hD: D = 28) : 
  A + B + C + D = 95 := by
  sorry

end total_sand_donated_l143_143648


namespace roots_of_quadratic_eq_l143_143114

theorem roots_of_quadratic_eq (x : ℝ) : (x + 1) ^ 2 = 0 → x = -1 := by
  sorry

end roots_of_quadratic_eq_l143_143114


namespace sphere_radius_l143_143287

theorem sphere_radius (tree_height sphere_shadow tree_shadow : ℝ) 
  (h_tree_shadow_pos : tree_shadow > 0) 
  (h_sphere_shadow_pos : sphere_shadow > 0) 
  (h_tree_height_pos : tree_height > 0)
  (h_tangent : (tree_height / tree_shadow) = (sphere_shadow / 15)) : 
  sphere_shadow = 11.25 :=
by
  sorry

end sphere_radius_l143_143287


namespace average_pregnancies_per_kettle_l143_143380

-- Define the given conditions
def num_kettles : ℕ := 6
def babies_per_pregnancy : ℕ := 4
def survival_rate : ℝ := 0.75
def total_expected_babies : ℕ := 270

-- Calculate surviving babies per pregnancy
def surviving_babies_per_pregnancy : ℝ := babies_per_pregnancy * survival_rate

-- Prove that the average number of pregnancies per kettle is 15
theorem average_pregnancies_per_kettle : ∃ P : ℝ, num_kettles * P * surviving_babies_per_pregnancy = total_expected_babies ∧ P = 15 :=
by
  sorry

end average_pregnancies_per_kettle_l143_143380


namespace person_B_age_l143_143126

variables (a b c d e f g : ℕ)

-- Conditions
axiom cond1 : a = b + 2
axiom cond2 : b = 2 * c
axiom cond3 : c = d / 2
axiom cond4 : d = e - 3
axiom cond5 : f = a * d
axiom cond6 : g = b + e
axiom cond7 : a + b + c + d + e + f + g = 292

-- Theorem statement
theorem person_B_age : b = 14 :=
sorry

end person_B_age_l143_143126


namespace cricket_run_target_l143_143031

theorem cricket_run_target
  (run_rate_1st_period : ℝ)
  (overs_1st_period : ℕ)
  (run_rate_2nd_period : ℝ)
  (overs_2nd_period : ℕ)
  (target_runs : ℝ)
  (h1 : run_rate_1st_period = 3.2)
  (h2 : overs_1st_period = 10)
  (h3 : run_rate_2nd_period = 5)
  (h4 : overs_2nd_period = 50) :
  target_runs = (run_rate_1st_period * overs_1st_period) + (run_rate_2nd_period * overs_2nd_period) :=
by
  sorry

end cricket_run_target_l143_143031


namespace particle_hits_origin_l143_143443

def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| x+1, y+1 => 0.25 * P x (y+1) + 0.25 * P (x+1) y + 0.5 * P x y

theorem particle_hits_origin :
    ∃ m n : ℕ, m ≠ 0 ∧ m % 4 ≠ 0 ∧ P 5 5 = m / 4^n :=
sorry

end particle_hits_origin_l143_143443


namespace right_triangle_area_l143_143588

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l143_143588


namespace least_whole_number_subtracted_from_ratio_l143_143463

theorem least_whole_number_subtracted_from_ratio (x : ℕ) : 
  (6 - x) / (7 - x) < 16 / 21 := by
  sorry

end least_whole_number_subtracted_from_ratio_l143_143463


namespace binomial_20_5_l143_143140

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end binomial_20_5_l143_143140


namespace odd_function_a_increasing_function_a_l143_143543

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem odd_function_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - (f x a)) → a = -1 :=
by sorry

theorem increasing_function_a (a : ℝ) :
  (∀ x : ℝ, (Real.exp x - a * Real.exp (-x)) ≥ 0) → a ∈ Set.Iic 0 :=
by sorry

end odd_function_a_increasing_function_a_l143_143543


namespace cafe_purchase_max_items_l143_143223

theorem cafe_purchase_max_items (total_money sandwich_cost soft_drink_cost : ℝ) (total_money_pos sandwich_cost_pos soft_drink_cost_pos : total_money > 0 ∧ sandwich_cost > 0 ∧ soft_drink_cost > 0) :
    total_money = 40 ∧ sandwich_cost = 5 ∧ soft_drink_cost = 1.50 →
    ∃ s d : ℕ, s + d = 10 ∧ total_money = sandwich_cost * s + soft_drink_cost * d :=
by
  sorry

end cafe_purchase_max_items_l143_143223


namespace cube_split_includes_2015_l143_143796

theorem cube_split_includes_2015 (m : ℕ) (h1 : m > 1) (h2 : ∃ (k : ℕ), 2 * k + 1 = 2015) : m = 45 :=
by
  sorry

end cube_split_includes_2015_l143_143796


namespace inverse_of_composite_l143_143646

-- Define the function g
def g (x : ℕ) : ℕ :=
  if x = 1 then 4 else
  if x = 2 then 3 else
  if x = 3 then 1 else
  if x = 4 then 5 else
  if x = 5 then 2 else
  0  -- g is not defined for values other than 1 to 5

-- Define the inverse g_inv
def g_inv (x : ℕ) : ℕ :=
  if x = 4 then 1 else
  if x = 3 then 2 else
  if x = 1 then 3 else
  if x = 5 then 4 else
  if x = 2 then 5 else
  0  -- g_inv is not defined for values other than 1 to 5

theorem inverse_of_composite :
  g_inv (g_inv (g_inv 3)) = 4 :=
by
  sorry

end inverse_of_composite_l143_143646


namespace multiples_of_9_ending_in_5_l143_143226

theorem multiples_of_9_ending_in_5 (n : ℕ) :
  (∃ k : ℕ, n = 9 * k ∧ 0 < n ∧ n < 600 ∧ n % 10 = 5) → 
  ∃ l, l = 7 := 
by
sorry

end multiples_of_9_ending_in_5_l143_143226


namespace mean_equality_l143_143649

theorem mean_equality (y : ℝ) :
  ((3 + 7 + 11 + 15) / 4 = (10 + 14 + y) / 3) → y = 3 :=
by
  sorry

end mean_equality_l143_143649


namespace no_linear_term_l143_143745

theorem no_linear_term (m : ℝ) (x : ℝ) : 
  (x + m) * (x + 3) - (x * x + 3 * m) = 0 → m = -3 :=
by
  sorry

end no_linear_term_l143_143745


namespace tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l143_143739

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def tangent_line_p (x y : ℝ) : Prop :=
  2 * x - sqrt 5 * y - 9 = 0

noncomputable def line_q1 (x y : ℝ) : Prop :=
  x = 3

noncomputable def line_q2 (x y : ℝ) : Prop :=
  8 * x - 15 * y + 51 = 0

theorem tangent_line_through_P :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (2, -sqrt 5) →
    tangent_line_p x y := 
sorry

theorem tangent_line_through_Q1 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q1 x y := 
sorry

theorem tangent_line_through_Q2 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q2 x y := 
sorry

end tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l143_143739


namespace gcd_36_60_l143_143187

theorem gcd_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l143_143187


namespace find_correct_t_l143_143154

theorem find_correct_t (t : ℝ) :
  (∃! x1 x2 x3 : ℝ, x1^2 - 4*|x1| + 3 = t ∧
                     x2^2 - 4*|x2| + 3 = t ∧
                     x3^2 - 4*|x3| + 3 = t) → t = 3 :=
by
  sorry

end find_correct_t_l143_143154


namespace T_shape_perimeter_l143_143561

/-- Two rectangles each measuring 3 inch × 5 inch are placed to form the letter T.
The overlapping area between the two rectangles is 1.5 inch. -/
theorem T_shape_perimeter:
  let l := 5 -- inches
  let w := 3 -- inches
  let overlap := 1.5 -- inches
  -- perimeter of one rectangle
  let P := 2 * (l + w)
  -- total perimeter accounting for overlap
  let total_perimeter := 2 * P - 2 * overlap
  total_perimeter = 29 :=
by
  sorry

end T_shape_perimeter_l143_143561


namespace simplify_expression_l143_143147

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 3) :
  (3 * x ^ 2 - 2 * x - 4) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3)) =
  3 * (x ^ 2 - x - 3) / ((x + 2) * (x - 3)) :=
by
  sorry

end simplify_expression_l143_143147


namespace constant_k_independent_of_b_l143_143018

noncomputable def algebraic_expression (a b k : ℝ) : ℝ :=
  a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2)

theorem constant_k_independent_of_b (a : ℝ) : (algebraic_expression a b 2) = (algebraic_expression a 1 2) :=
by
  sorry

end constant_k_independent_of_b_l143_143018


namespace prism_volume_l143_143274

noncomputable def volume_prism (x y z : ℝ) : ℝ := x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 12) (h2 : y * z = 8) (h3 : z * x = 6) :
  volume_prism x y z = 24 :=
by
  sorry

end prism_volume_l143_143274


namespace ratio_of_speeds_l143_143806

theorem ratio_of_speeds (v_A v_B : ℝ)
  (h₀ : 4 * v_A = abs (600 - 4 * v_B))
  (h₁ : 9 * v_A = abs (600 - 9 * v_B)) :
  v_A / v_B = 2 / 3 :=
sorry

end ratio_of_speeds_l143_143806


namespace alice_burger_spending_l143_143235

theorem alice_burger_spending :
  let daily_burgers := 4
  let burger_cost := 13
  let days_in_june := 30
  let mondays_wednesdays := 8
  let fridays := 4
  let fifth_purchase_coupons := 6
  let discount_10_percent := 0.9
  let discount_50_percent := 0.5
  let full_price := days_in_june * daily_burgers * burger_cost
  let discount_10 := mondays_wednesdays * daily_burgers * burger_cost * discount_10_percent
  let fridays_cost := (daily_burgers - 1) * fridays * burger_cost
  let discount_50 := fifth_purchase_coupons * burger_cost * discount_50_percent
  full_price - discount_10 - fridays_cost - discount_50 + fridays_cost = 1146.6 := by sorry

end alice_burger_spending_l143_143235


namespace polynomial_difference_of_squares_l143_143159

theorem polynomial_difference_of_squares:
  (∀ a b : ℝ, ¬ ∃ x1 x2 : ℝ, a^2 + (-b)^2 = (x1 - x2) * (x1 + x2)) ∧
  (∀ m n : ℝ, ¬ ∃ x1 x2 : ℝ, 5 * m^2 - 20 * m * n = (x1 - x2) * (x1 + x2)) ∧
  (∀ x y : ℝ, ¬ ∃ x1 x2 : ℝ, -x^2 - y^2 = (x1 - x2) * (x1 + x2)) →
  ∃ x1 x2 : ℝ, -x^2 + 9 = (x1 - x2) * (x1 + x2) :=
by 
  sorry

end polynomial_difference_of_squares_l143_143159


namespace area_of_figure_l143_143155

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end area_of_figure_l143_143155


namespace parent_combinations_for_O_l143_143403

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB

open BloodType

-- Define the conditions given in the problem
def parent_not_AB (p : BloodType) : Prop :=
  p ≠ AB

def possible_parent_types : List BloodType :=
  [A, B, O]

-- The math proof problem
theorem parent_combinations_for_O :
  ∀ (mother father : BloodType),
    parent_not_AB mother →
    parent_not_AB father →
    mother ∈ possible_parent_types →
    father ∈ possible_parent_types →
    (possible_parent_types.length * possible_parent_types.length) = 9 := 
by
  intro mother father h1 h2 h3 h4
  sorry

end parent_combinations_for_O_l143_143403


namespace quadratic_binomial_form_l143_143095

theorem quadratic_binomial_form (y : ℝ) : ∃ (k : ℝ), y^2 + 14 * y + 40 = (y + 7)^2 + k :=
by
  use -9
  sorry

end quadratic_binomial_form_l143_143095


namespace ratio_of_areas_l143_143022

theorem ratio_of_areas (s L : ℝ) (h1 : (π * L^2) / (π * s^2) = 9 / 4) : L - s = (1/2) * s :=
by
  sorry

end ratio_of_areas_l143_143022


namespace probability_divisible_by_3_l143_143209

theorem probability_divisible_by_3 (a b c : ℕ) (h : a ∈ Finset.range 2008 ∧ b ∈ Finset.range 2008 ∧ c ∈ Finset.range 2008) :
  (∃ p : ℚ, p = 1265/2007 ∧ (abc + ac + a) % 3 = 0) :=
sorry

end probability_divisible_by_3_l143_143209


namespace range_of_a_for_two_unequal_roots_l143_143896

theorem range_of_a_for_two_unequal_roots (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * Real.log x₁ = x₁ ∧ a * Real.log x₂ = x₂) ↔ a > Real.exp 1 :=
sorry

end range_of_a_for_two_unequal_roots_l143_143896


namespace find_common_ratio_l143_143822

variable (a : ℕ → ℝ) -- represents the geometric sequence
variable (q : ℝ) -- represents the common ratio

-- conditions given in the problem
def a_3_condition : a 3 = 4 := sorry
def a_6_condition : a 6 = 1 / 2 := sorry

-- the general form of the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * q ^ n

-- the theorem we want to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 4) (h2 : a 6 = 1 / 2) 
  (hg : geometric_sequence a q) : q = 1 / 2 :=
sorry

end find_common_ratio_l143_143822


namespace possible_values_sin_plus_cos_l143_143497

variable (x : ℝ)

theorem possible_values_sin_plus_cos (h : 2 * Real.cos x - 3 * Real.sin x = 2) :
    ∃ (values : Set ℝ), values = {3, -31 / 13} ∧ (Real.sin x + 3 * Real.cos x) ∈ values := by
  sorry

end possible_values_sin_plus_cos_l143_143497


namespace find_other_number_l143_143758

-- Define LCM and HCF conditions
def lcm_a_b := 2310
def hcf_a_b := 83
def number_a := 210

-- Define the problem to find the other number
def number_b : ℕ :=
  lcm_a_b * hcf_a_b / number_a

-- Statement: Prove that the other number is 913
theorem find_other_number : number_b = 913 := by
  -- Placeholder for proof
  sorry

end find_other_number_l143_143758


namespace gain_percent_l143_143614

variable (MP CP SP : ℝ)

def costPrice (CP : ℝ) (MP : ℝ) := CP = 0.64 * MP

def sellingPrice (SP : ℝ) (MP : ℝ) := SP = MP * 0.88

theorem gain_percent (h1 : costPrice CP MP) (h2 : sellingPrice SP MP) : 
  ((SP - CP) / CP) * 100 = 37.5 :=
by
  sorry

end gain_percent_l143_143614


namespace solve_for_z_l143_143702

theorem solve_for_z (z : ℂ) (h : 5 - 3 * (I * z) = 3 + 5 * (I * z)) : z = I / 4 :=
sorry

end solve_for_z_l143_143702


namespace lyle_percentage_l143_143589

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end lyle_percentage_l143_143589


namespace s_l143_143340

def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def total_money_spent : ℝ := 15.0

theorem s'mores_per_scout :
  (total_money_spent / cost_per_chocolate_bar * sections_per_chocolate_bar) / scouts = 2 :=
by
  sorry

end s_l143_143340


namespace x2_plus_y2_lt_1_l143_143528

theorem x2_plus_y2_lt_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 :=
sorry

end x2_plus_y2_lt_1_l143_143528


namespace find_a_b_c_l143_143784

theorem find_a_b_c (a b c : ℝ) 
  (h_min : ∀ x, -9 * x^2 + 54 * x - 45 ≥ 36) 
  (h1 : 0 = a * (1 - 1) * (1 - 5)) 
  (h2 : 0 = a * (5 - 1) * (5 - 5)) :
  a + b + c = 36 :=
sorry

end find_a_b_c_l143_143784


namespace largest_bucket_capacity_l143_143081

-- Let us define the initial conditions
def capacity_5_liter_bucket : ℕ := 5
def capacity_3_liter_bucket : ℕ := 3
def remaining_after_pour := capacity_5_liter_bucket - capacity_3_liter_bucket
def additional_capacity_without_overflow : ℕ := 4

-- Problem statement: Prove that the capacity of the largest bucket is 6 liters
theorem largest_bucket_capacity : ∀ (c : ℕ), remaining_after_pour + additional_capacity_without_overflow = c → c = 6 := 
by
  sorry

end largest_bucket_capacity_l143_143081


namespace find_tangent_lines_l143_143748

noncomputable def tangent_lines (x y : ℝ) : Prop :=
  (x = 2 ∨ 3 * x - 4 * y + 10 = 0)

theorem find_tangent_lines :
  ∃ (x y : ℝ), tangent_lines x y ∧ (x^2 + y^2 = 4) ∧ ((x, y) ≠ (2, 4)) :=
by
  sorry

end find_tangent_lines_l143_143748


namespace convert_speed_72_kmph_to_mps_l143_143040

theorem convert_speed_72_kmph_to_mps :
  let kmph := 72
  let factor_km_to_m := 1000
  let factor_hr_to_s := 3600
  (kmph * factor_km_to_m) / factor_hr_to_s = 20 := by
  -- (72 kmph * (1000 meters / 1 kilometer)) / (3600 seconds / 1 hour) = 20 meters per second
  sorry

end convert_speed_72_kmph_to_mps_l143_143040


namespace percentage_of_360_equals_115_2_l143_143520

theorem percentage_of_360_equals_115_2 (p : ℝ) (h : (p / 100) * 360 = 115.2) : p = 32 :=
by
  sorry

end percentage_of_360_equals_115_2_l143_143520


namespace range_of_a_l143_143761

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - 2 * a) * (a * x - 1) < 0 → (x > 1 / a ∨ x < 2 * a)) → (a ≤ -Real.sqrt 2 / 2) :=
by
  intro h
  sorry

end range_of_a_l143_143761


namespace max_min_value_of_fg_l143_143039

noncomputable def f (x : ℝ) : ℝ := 4 - x^2
noncomputable def g (x : ℝ) : ℝ := 3 * x
noncomputable def min' (a b : ℝ) : ℝ := if a < b then a else b

theorem max_min_value_of_fg : ∃ x : ℝ, min' (f x) (g x) = 3 :=
by
  sorry

end max_min_value_of_fg_l143_143039


namespace consecutive_primes_sum_square_is_prime_l143_143538

-- Defining what it means for three numbers to be consecutive primes
def consecutive_primes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ((p < q ∧ q < r) ∨ (p < q ∧ q < r ∧ r < p) ∨ 
   (r < p ∧ p < q) ∨ (q < p ∧ p < r) ∨ 
   (q < r ∧ r < p) ∨ (r < q ∧ q < p))

-- Defining our main problem statement
theorem consecutive_primes_sum_square_is_prime :
  ∀ p q r : ℕ, consecutive_primes p q r → Nat.Prime (p^2 + q^2 + r^2) ↔ (p = 3 ∧ q = 5 ∧ r = 7) :=
by
  -- Sorry is used to skip the proof.
  sorry

end consecutive_primes_sum_square_is_prime_l143_143538


namespace sum_of_four_squares_eq_20_l143_143829

variable (x y : ℕ)

-- Conditions based on the provided problem
def condition1 := 2 * x + 2 * y = 16
def condition2 := 2 * x + 3 * y = 19

-- Theorem to be proven
theorem sum_of_four_squares_eq_20 (h1 : condition1 x y) (h2 : condition2 x y) : 4 * x = 20 :=
by
  sorry

end sum_of_four_squares_eq_20_l143_143829


namespace coordinates_of_S_l143_143397

variable (P Q R S : (ℝ × ℝ))
variable (hp : P = (3, -2))
variable (hq : Q = (3, 1))
variable (hr : R = (7, 1))
variable (h : Rectangle P Q R S)

def Rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  let (xS, yS) := S
  (xP = xQ ∧ yR = yS) ∧ (xS = xR ∧ yP = yQ) 

theorem coordinates_of_S : S = (7, -2) := by
  sorry

end coordinates_of_S_l143_143397


namespace pairs_of_powers_of_two_l143_143531

theorem pairs_of_powers_of_two (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ a : ℕ, m + n = 2^a) (h4 : ∃ b : ℕ, mn + 1 = 2^b) :
  (∃ a : ℕ, m = 2^a - 1 ∧ n = 1) ∨ 
  (∃ a : ℕ, m = 2^(a-1) + 1 ∧ n = 2^(a-1) - 1) :=
sorry

end pairs_of_powers_of_two_l143_143531


namespace dave_total_rides_l143_143814

theorem dave_total_rides (rides_first_day rides_second_day : ℕ) (h1 : rides_first_day = 4) (h2 : rides_second_day = 3) :
  rides_first_day + rides_second_day = 7 :=
by
  sorry

end dave_total_rides_l143_143814


namespace increased_time_between_maintenance_checks_l143_143117

theorem increased_time_between_maintenance_checks (original_time : ℕ) (percentage_increase : ℕ) : 
  original_time = 20 → percentage_increase = 25 →
  original_time + (original_time * percentage_increase / 100) = 25 :=
by
  intros
  sorry

end increased_time_between_maintenance_checks_l143_143117


namespace population_net_increase_per_day_l143_143148

theorem population_net_increase_per_day (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) (net_increase : ℚ) :
  birth_rate = 7 / 2 ∧
  death_rate = 2 / 2 ∧
  seconds_per_day = 24 * 60 * 60 ∧
  net_increase = (birth_rate - death_rate) * seconds_per_day →
  net_increase = 216000 := 
by
  sorry

end population_net_increase_per_day_l143_143148


namespace smallest_circle_radius_l143_143232

-- Define the problem as a proposition
theorem smallest_circle_radius (r : ℝ) (R1 R2 : ℝ) (hR1 : R1 = 6) (hR2 : R2 = 4) (h_right_triangle : (r + R2)^2 + (r + R1)^2 = (R2 + R1)^2) : r = 2 := 
sorry

end smallest_circle_radius_l143_143232


namespace units_digit_a_2017_l143_143407

noncomputable def a_n (n : ℕ) : ℝ :=
  (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

theorem units_digit_a_2017 : (Nat.floor (a_n 2017)) % 10 = 2 :=
  sorry

end units_digit_a_2017_l143_143407


namespace profitable_allocation_2015_l143_143567

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l143_143567


namespace evaporation_days_l143_143607

theorem evaporation_days
    (initial_water : ℝ)
    (evap_rate : ℝ)
    (percent_evaporated : ℝ)
    (evaporated_water : ℝ)
    (days : ℝ)
    (h1 : initial_water = 10)
    (h2 : evap_rate = 0.012)
    (h3 : percent_evaporated = 0.06)
    (h4 : evaporated_water = initial_water * percent_evaporated)
    (h5 : days = evaporated_water / evap_rate) :
  days = 50 :=
by
  sorry

end evaporation_days_l143_143607


namespace andrew_age_l143_143585

theorem andrew_age (a g : ℝ) (h1 : g = 9 * a) (h2 : g - a = 63) : a = 7.875 :=
by
  sorry

end andrew_age_l143_143585


namespace circles_intersect_l143_143166

theorem circles_intersect
  (r : ℝ) (R : ℝ) (d : ℝ)
  (hr : r = 4)
  (hR : R = 5)
  (hd : d = 6) :
  1 < d ∧ d < r + R :=
by
  sorry

end circles_intersect_l143_143166


namespace num_sets_N_l143_143594

open Set

-- Define the set M and the set U
def M : Set ℕ := {1, 2}
def U : Set ℕ := {1, 2, 3, 4}

-- The statement to prove
theorem num_sets_N : 
  ∃ count : ℕ, count = 4 ∧ 
  (∀ N : Set ℕ, M ∪ N = U → N = {3, 4} ∨ N = {1, 3, 4} ∨ N = {2, 3, 4} ∨ N = {1, 2, 3, 4}) :=
by
  sorry

end num_sets_N_l143_143594


namespace num_valid_n_l143_143636

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (Nat.succ n') => Nat.succ n' * factorial n'

def divisible (a b : ℕ) : Prop := b ∣ a

theorem num_valid_n (N : ℕ) :
  N ≤ 30 → 
  ¬ (∃ k, k + 1 ≤ 31 ∧ k + 1 > 1 ∧ (Prime (k + 1)) ∧ ¬ divisible (2 * factorial (k - 1)) (k + 1)) →
  ∃ m : ℕ, m = 20 :=
by
  sorry

end num_valid_n_l143_143636


namespace solutions_of_quadratic_eq_l143_143308

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l143_143308


namespace ratio_surface_area_cube_to_octahedron_l143_143676

noncomputable def cube_side_length := 1

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

noncomputable def edge_length_octahedron := 1

-- Surface area formula for a regular octahedron with side length e is 2 * sqrt(3) * e^2
noncomputable def surface_area_octahedron (e : ℝ) : ℝ := 2 * Real.sqrt 3 * e^2

-- Finally, we want to prove that the ratio of the surface area of the cube to that of the octahedron is sqrt(3)
theorem ratio_surface_area_cube_to_octahedron :
  surface_area_cube cube_side_length / surface_area_octahedron edge_length_octahedron = Real.sqrt 3 :=
by sorry

end ratio_surface_area_cube_to_octahedron_l143_143676


namespace area_of_hexagon_l143_143449

def isRegularHexagon (A B C D E F : Type) : Prop := sorry
def isInsideQuadrilateral (P : Type) (A B C D : Type) : Prop := sorry
def areaTriangle (P X Y : Type) : Real := sorry

theorem area_of_hexagon (A B C D E F P : Type)
    (h1 : isRegularHexagon A B C D E F)
    (h2 : isInsideQuadrilateral P A B C D)
    (h3 : areaTriangle P B C = 20)
    (h4 : areaTriangle P A D = 23) :
    ∃ area : Real, area = 189 :=
sorry

end area_of_hexagon_l143_143449


namespace remainder_is_4_over_3_l143_143962

noncomputable def original_polynomial (z : ℝ) : ℝ := 3 * z ^ 3 - 4 * z ^ 2 - 14 * z + 3
noncomputable def divisor (z : ℝ) : ℝ := 3 * z + 5
noncomputable def quotient (z : ℝ) : ℝ := z ^ 2 - 3 * z + 1 / 3

theorem remainder_is_4_over_3 :
  ∃ r : ℝ, original_polynomial z = divisor z * quotient z + r ∧ r = 4 / 3 :=
sorry

end remainder_is_4_over_3_l143_143962


namespace find_B_find_sin_A_find_sin_2A_minus_B_l143_143305

open Real

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a * cos C + c * cos A = 2 * b * cos B) ∧ (7 * a = 5 * b)

theorem find_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_A (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) :
  sin A = 3 * sqrt 3 / 14 :=
sorry

theorem find_sin_2A_minus_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) (hA : sin A = 3 * sqrt 3 / 14) :
  sin (2 * A - B) = 8 * sqrt 3 / 49 :=
sorry

end find_B_find_sin_A_find_sin_2A_minus_B_l143_143305


namespace acres_left_untouched_l143_143476

def total_acres := 65057
def covered_acres := 64535

theorem acres_left_untouched : total_acres - covered_acres = 522 :=
by
  sorry

end acres_left_untouched_l143_143476


namespace universal_quantifiers_are_true_l143_143684

-- Declare the conditions as hypotheses
theorem universal_quantifiers_are_true :
  (∀ x : ℝ, x^2 - x + 0.25 ≥ 0) ∧ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry -- Proof skipped

end universal_quantifiers_are_true_l143_143684


namespace find_integer_x_l143_143321

theorem find_integer_x (x : ℕ) (pos_x : 0 < x) (ineq : x + 1000 > 1000 * x) : x = 1 :=
sorry

end find_integer_x_l143_143321


namespace solve_eq_l143_143583

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq_l143_143583


namespace inclination_angle_of_line_l143_143570

open Real

theorem inclination_angle_of_line (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  sorry

end inclination_angle_of_line_l143_143570


namespace g_g_g_g_3_l143_143978

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l143_143978


namespace markers_last_group_correct_l143_143488

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end markers_last_group_correct_l143_143488


namespace sequence_a500_l143_143200

theorem sequence_a500 (a : ℕ → ℤ)
  (h1 : a 1 = 2010)
  (h2 : a 2 = 2011)
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) :
  a 500 = 2177 :=
sorry

end sequence_a500_l143_143200


namespace tangent_line_at_2_eq_l143_143207

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2_eq :
  let x := (2 : ℝ)
  let slope := (deriv f) x
  let y := f x
  ∃ (m y₀ : ℝ), m = slope ∧ y₀ = y ∧ 
    (∀ (x y : ℝ), y = m * (x - 2) + y₀ → x - y - 4 = 0)
:= sorry

end tangent_line_at_2_eq_l143_143207


namespace weight_of_green_peppers_l143_143156

-- Definitions for conditions and question
def total_weight : ℝ := 0.6666666667
def is_split_equally (x y : ℝ) : Prop := x = y

-- Theorem statement that needs to be proved
theorem weight_of_green_peppers (g r : ℝ) (h_split : is_split_equally g r) (h_total : g + r = total_weight) :
  g = 0.33333333335 :=
by sorry

end weight_of_green_peppers_l143_143156


namespace common_solutions_form_segment_length_one_l143_143572

theorem common_solutions_form_segment_length_one (a : ℝ) (h₁ : ∀ x : ℝ, x^2 - 4 * x + 2 - a ≤ 0) 
  (h₂ : ∀ x : ℝ, x^2 - 5 * x + 2 * a + 8 ≤ 0) : 
  (a = -1 ∨ a = -7 / 4) :=
by
  sorry

end common_solutions_form_segment_length_one_l143_143572


namespace count_not_squares_or_cubes_l143_143136

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l143_143136


namespace necessary_but_not_sufficient_condition_l143_143057

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 5) → (x > 4) :=
by 
  intro h
  linarith

end necessary_but_not_sufficient_condition_l143_143057


namespace yearly_production_target_l143_143759

-- Definitions for the conditions
def p_current : ℕ := 100
def p_add : ℕ := 50

-- The theorem to be proven
theorem yearly_production_target : (p_current + p_add) * 12 = 1800 := by
  sorry  -- Proof is omitted

end yearly_production_target_l143_143759


namespace both_inequalities_equiv_l143_143958

theorem both_inequalities_equiv (x : ℝ) : (x - 3)/(2 - x) ≥ 0 ↔ (3 - x)/(x - 2) ≥ 0 := by
  sorry

end both_inequalities_equiv_l143_143958


namespace expand_expression_l143_143743

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 :=
by
  sorry

end expand_expression_l143_143743


namespace f_1986_l143_143280

noncomputable def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 3 * f (a * b)
axiom f_1 : f 1 = 2

theorem f_1986 : f 1986 = 2 :=
by
  sorry

end f_1986_l143_143280


namespace triangle_perimeter_is_correct_l143_143178

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

def triangle_perimeter (a b c : ℝ) := a + b + c

theorem triangle_perimeter_is_correct :
  c = sqrt 7 → C = π / 3 → S = 3 * sqrt 3 / 2 →
  S = (1 / 2) * a * b * sin (C) → c^2 = a^2 + b^2 - 2 * a * b * cos (C) →
  ∃ a b : ℝ, triangle_perimeter a b c = 5 + sqrt 7 :=
  by
    intros h1 h2 h3 h4 h5
    sorry

end triangle_perimeter_is_correct_l143_143178


namespace gcd_factorial_8_10_l143_143215

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l143_143215


namespace problem_statement_l143_143144

variable {x y z : ℝ}

theorem problem_statement 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  1 / (x ^ 3 * y) + 1 / (y ^ 3 * z) + 1 / (z ^ 3 * x) ≥ x * y + y * z + z * x :=
by sorry

end problem_statement_l143_143144


namespace height_of_david_l143_143713

theorem height_of_david
  (building_height : ℕ)
  (building_shadow : ℕ)
  (david_shadow : ℕ)
  (ratio : ℕ)
  (h1 : building_height = 50)
  (h2 : building_shadow = 25)
  (h3 : david_shadow = 18)
  (h4 : ratio = building_height / building_shadow) :
  david_shadow * ratio = 36 := sorry

end height_of_david_l143_143713


namespace correct_quotient_l143_143199

theorem correct_quotient (Q : ℤ) (D : ℤ) (h1 : D = 21 * Q) (h2 : D = 12 * 35) : Q = 20 :=
by {
  sorry
}

end correct_quotient_l143_143199


namespace equation_solution_l143_143347

theorem equation_solution (x : ℝ) : (3 : ℝ)^(x-1) = 1/9 ↔ x = -1 :=
by sorry

end equation_solution_l143_143347


namespace height_difference_l143_143335

variable (H_A H_B : ℝ)

-- Conditions
axiom B_is_66_67_percent_more_than_A : H_B = H_A * 1.6667

-- Proof statement
theorem height_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.6667) : 
  (H_B - H_A) / H_B * 100 = 40 := by
sorry

end height_difference_l143_143335


namespace find_p4_q4_l143_143645

-- Definitions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x - 6) * (x + 3)

-- Statement to prove
theorem find_p4_q4 : (p 4) / (q 4) = 6 / 7 :=
by
  sorry

end find_p4_q4_l143_143645


namespace sum_of_decimals_l143_143353

theorem sum_of_decimals : (1 / 10) + (9 / 100) + (9 / 1000) + (7 / 10000) = 0.1997 := 
sorry

end sum_of_decimals_l143_143353


namespace breadth_decrease_percentage_l143_143408

theorem breadth_decrease_percentage
  (L B : ℝ)
  (hLpos : L > 0)
  (hBpos : B > 0)
  (harea_change : (1.15 * L) * (B - p/100 * B) = 1.035 * (L * B)) :
  p = 10 := 
sorry

end breadth_decrease_percentage_l143_143408


namespace limit_of_hours_for_overtime_l143_143696

theorem limit_of_hours_for_overtime
  (R : Real) (O : Real) (total_compensation : Real) (total_hours_worked : Real) (L : Real)
  (hR : R = 14)
  (hO : O = 1.75 * R)
  (hTotalCompensation : total_compensation = 998)
  (hTotalHoursWorked : total_hours_worked = 57.88)
  (hEquation : (R * L) + ((total_hours_worked - L) * O) = total_compensation) :
  L = 40 := 
  sorry

end limit_of_hours_for_overtime_l143_143696


namespace three_digit_even_with_sum_twelve_l143_143013

theorem three_digit_even_with_sum_twelve :
  ∃ n: ℕ, n = 36 ∧ 
    (∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 2 = 0 ∧ 
          ((x / 10) % 10 + x % 10 = 12) → x = n) :=
sorry

end three_digit_even_with_sum_twelve_l143_143013


namespace count_factors_of_product_l143_143227

theorem count_factors_of_product :
  let n := 8^4 * 7^3 * 9^1 * 5^5
  ∃ (count : ℕ), count = 936 ∧ 
    ∀ f : ℕ, f ∣ n → ∃ a b c d : ℕ,
      a ≤ 12 ∧ b ≤ 2 ∧ c ≤ 5 ∧ d ≤ 3 ∧ 
      f = 2^a * 3^b * 5^c * 7^d :=
by sorry

end count_factors_of_product_l143_143227


namespace percent_decrease_computer_price_l143_143852

theorem percent_decrease_computer_price (price_1990 price_2010 : ℝ) (h1 : price_1990 = 1200) (h2 : price_2010 = 600) :
  ((price_1990 - price_2010) / price_1990) * 100 = 50 := 
  sorry

end percent_decrease_computer_price_l143_143852


namespace sin_pi_over_six_eq_half_l143_143558

theorem sin_pi_over_six_eq_half : Real.sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_pi_over_six_eq_half_l143_143558


namespace necessary_but_not_sufficient_condition_l143_143652

theorem necessary_but_not_sufficient_condition {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  ((a + b > 1) ↔ (ab > 1)) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l143_143652


namespace line_perpendicular_through_P_l143_143122

/-
  Given:
  1. The point P(-2, 2).
  2. The line 2x - y + 1 = 0.
  Prove:
  The equation of the line that passes through P and is perpendicular to the given line is x + 2y - 2 = 0.
-/

def P : ℝ × ℝ := (-2, 2)
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem line_perpendicular_through_P :
  ∃ (x y : ℝ) (m : ℝ), (x = -2) ∧ (y = 2) ∧ (m = -1/2) ∧ 
  (∀ (x₁ y₁ : ℝ), (y₁ - y) = m * (x₁ - x)) ∧ 
  (∀ (lx ly : ℝ), line1 lx ly → x + 2 * y - 2 = 0) := sorry

end line_perpendicular_through_P_l143_143122


namespace tan_alpha_beta_l143_143348

theorem tan_alpha_beta (α β : ℝ) (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := 
sorry

end tan_alpha_beta_l143_143348


namespace number_of_ordered_triples_l143_143610

theorem number_of_ordered_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq : a * b * c - b * c - a * c - a * b + a + b + c = 2013) :
    ∃ n, n = 39 :=
by
  sorry

end number_of_ordered_triples_l143_143610


namespace arithmetic_mean_16_24_40_32_l143_143231

theorem arithmetic_mean_16_24_40_32 : (16 + 24 + 40 + 32) / 4 = 28 :=
by
  sorry

end arithmetic_mean_16_24_40_32_l143_143231


namespace number_of_ordered_pairs_l143_143023

theorem number_of_ordered_pairs {x y: ℕ} (h1 : x < y) (h2 : 2 * x * y / (x + y) = 4^30) : 
  ∃ n, n = 61 :=
sorry

end number_of_ordered_pairs_l143_143023


namespace paul_money_left_l143_143735

-- Conditions
def cost_of_bread : ℕ := 2
def cost_of_butter : ℕ := 3
def cost_of_juice : ℕ := 2 * cost_of_bread
def total_money : ℕ := 15

-- Definition of total cost
def total_cost := cost_of_bread + cost_of_butter + cost_of_juice

-- Statement of the theorem
theorem paul_money_left : total_money - total_cost = 6 := by
  -- Sorry, implementation skipped
  sorry

end paul_money_left_l143_143735


namespace total_money_together_is_l143_143145

def Sam_has : ℚ := 750.50
def Billy_has (S : ℚ) : ℚ := 4.5 * S - 345.25
def Lila_has (B S : ℚ) : ℚ := 2.25 * (B - S)
def Total_money (S B L : ℚ) : ℚ := S + B + L

theorem total_money_together_is :
  Total_money Sam_has (Billy_has Sam_has) (Lila_has (Billy_has Sam_has) Sam_has) = 8915.88 :=
by sorry

end total_money_together_is_l143_143145


namespace no_such_a_b_exists_l143_143473

open Set

def A (a b : ℝ) : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, a * x + b) }

def B : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, 3 * (x : ℝ) ^ 2 + 15) }

def C : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 + p.2 ^ 2 ≤ 144 }

theorem no_such_a_b_exists :
  ¬ ∃ (a b : ℝ), 
    ((A a b ∩ B).Nonempty) ∧ ((a, b) ∈ C) :=
sorry

end no_such_a_b_exists_l143_143473


namespace quadratic_value_l143_143334

theorem quadratic_value (a b c : ℤ) (a_pos : a > 0) (h_eq : ∀ x : ℝ, (a * x + b)^2 = 49 * x^2 + 70 * x + c) : a + b + c = -134 :=
by
  -- Proof starts here
  sorry

end quadratic_value_l143_143334


namespace action_figures_per_shelf_l143_143416

theorem action_figures_per_shelf (total_figures shelves : ℕ) (h1 : total_figures = 27) (h2 : shelves = 3) :
  (total_figures / shelves = 9) :=
by
  sorry

end action_figures_per_shelf_l143_143416


namespace fourth_term_of_geometric_sequence_l143_143067

theorem fourth_term_of_geometric_sequence 
  (a r : ℕ) 
  (h₁ : a = 3)
  (h₂ : a * r^2 = 75) :
  a * r^3 = 375 := 
by
  sorry

end fourth_term_of_geometric_sequence_l143_143067


namespace sum_d_e_f_l143_143113

-- Define the variables
variables (d e f : ℤ)

-- Given conditions
def condition1 : Prop := ∀ x : ℤ, x^2 + 18 * x + 77 = (x + d) * (x + e)
def condition2 : Prop := ∀ x : ℤ, x^2 - 19 * x + 88 = (x - e) * (x - f)

-- Prove the statement
theorem sum_d_e_f : condition1 d e → condition2 e f → d + e + f = 26 :=
by
  intros h1 h2
  -- Proof omitted
  sorry

end sum_d_e_f_l143_143113


namespace convert_base_8_to_10_l143_143146

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l143_143146


namespace my_op_example_l143_143971

def my_op (a b : Int) : Int := a^2 - abs b

theorem my_op_example : my_op (-2) (-1) = 3 := by
  sorry

end my_op_example_l143_143971


namespace sequence_inequality_l143_143377

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h_cond : ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1) :
  ∀ p q : ℕ, |a p / p - a q / q| < 1 / p + 1 / q :=
by
  intros p q
  sorry

end sequence_inequality_l143_143377


namespace probability_roots_real_l143_143675

-- Define the polynomial
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + 3*b*x^3 + (3*b - 5)*x^2 + (-6*b + 4)*x - 3

-- Define the intervals for b
def interval_b1 := Set.Icc (-(15:ℝ)) (20:ℝ)
def interval_b2 := Set.Icc (-(15:ℝ)) (-2/3)
def interval_b3 := Set.Icc (4/3) (20:ℝ)

-- Calculate the lengths of the intervals
def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def length_b1 := length_interval (-(15:ℝ)) (20:ℝ)
noncomputable def length_b2 := length_interval (-(15:ℝ)) (-2/3)
noncomputable def length_b3 := length_interval (4/3) (20:ℝ)
noncomputable def effective_length := length_b2 + length_b3

-- The probability is the ratio of effective lengths
noncomputable def probability := effective_length / length_b1

-- The theorem we want to prove
theorem probability_roots_real : probability = 33/35 :=
  sorry

end probability_roots_real_l143_143675


namespace minimum_f_value_minimum_fraction_value_l143_143550

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem minimum_f_value : ∃ x : ℝ, f x = 2 :=
by
  -- proof skipped, please insert proof here
  sorry

theorem minimum_fraction_value (a b : ℝ) (h : a^2 + b^2 = 2) : 
  (1 / (a^2 + 1)) + (4 / (b^2 + 1)) = 9 / 4 :=
by
  -- proof skipped, please insert proof here
  sorry

end minimum_f_value_minimum_fraction_value_l143_143550


namespace inequality_proof_l143_143552

variables (x y z : ℝ)

theorem inequality_proof (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by
  sorry

end inequality_proof_l143_143552


namespace molecular_weight_of_one_mole_l143_143787

theorem molecular_weight_of_one_mole (molecular_weight_8_moles : ℝ) (h : molecular_weight_8_moles = 992) : 
  molecular_weight_8_moles / 8 = 124 :=
by
  -- proof goes here
  sorry

end molecular_weight_of_one_mole_l143_143787


namespace lucy_fish_count_l143_143819

theorem lucy_fish_count (initial_fish : ℕ) (additional_fish : ℕ) (final_fish : ℕ) : 
  initial_fish = 212 ∧ additional_fish = 68 → final_fish = 280 :=
by
  sorry

end lucy_fish_count_l143_143819


namespace points_on_square_diagonal_l143_143725

theorem points_on_square_diagonal (a : ℝ) (ha : a > 1) (Q : ℝ × ℝ) (hQ : Q = (a + 1, 4 * a + 1)) 
    (line : ℝ × ℝ → Prop) (hline : ∀ (x y : ℝ), line (x, y) ↔ y = a * x + 3) :
    ∃ (P R : ℝ × ℝ), line Q ∧ P = (6, 3) ∧ R = (-3, 6) :=
by
  sorry

end points_on_square_diagonal_l143_143725


namespace count_valid_triangles_l143_143261

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem count_valid_triangles : 
  (∃ n : ℕ, n = 14 ∧ 
  ∃ (a b c : ℕ), valid_triangle a b c ∧ 
  ((b = 5 ∧ c > 5) ∨ (c = 5 ∧ b > 5)) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0)) :=
by { sorry }

end count_valid_triangles_l143_143261


namespace find_a_l143_143811

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem find_a (a : ℝ) (h : binom_coeff 9 3 * (-a)^3 = -84) : a = 1 :=
by
  sorry

end find_a_l143_143811


namespace fx_root_and_decreasing_l143_143468

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log x / Real.log 2

theorem fx_root_and_decreasing (a x0 : ℝ) (h0 : 0 < a) (hx0 : 0 < x0) (h_cond : a < x0) (hf_root : f x0 = 0) 
  (hf_decreasing : ∀ x y : ℝ, x < y → f y < f x) : f a > 0 := 
sorry

end fx_root_and_decreasing_l143_143468


namespace girls_count_l143_143327

-- Define the constants according to the conditions
def boys_on_team : ℕ := 28
def groups : ℕ := 8
def members_per_group : ℕ := 4

-- Calculate the total number of members
def total_members : ℕ := groups * members_per_group

-- Calculate the number of girls by subtracting the number of boys from the total members
def girls_on_team : ℕ := total_members - boys_on_team

-- The proof statement: prove that the number of girls on the team is 4
theorem girls_count : girls_on_team = 4 := by
  -- Skip the proof, completing the statement
  sorry

end girls_count_l143_143327


namespace money_left_after_purchase_l143_143498

noncomputable def total_cost : ℝ := 250 + 25 + 35 + 45 + 90

def savings_erika : ℝ := 155

noncomputable def savings_rick : ℝ := total_cost / 2

def savings_sam : ℝ := 175

def combined_cost_cake_flowers_skincare : ℝ := 25 + 35 + 45

noncomputable def savings_amy : ℝ := 2 * combined_cost_cake_flowers_skincare

noncomputable def total_savings : ℝ := savings_erika + savings_rick + savings_sam + savings_amy

noncomputable def money_left : ℝ := total_savings - total_cost

theorem money_left_after_purchase : money_left = 317.5 := by
  sorry

end money_left_after_purchase_l143_143498


namespace value_of_b_l143_143517

variable (a b : ℤ)

theorem value_of_b : a = 105 ∧ a ^ 3 = 21 * 49 * 45 * b → b = 1 := by
  sorry

end value_of_b_l143_143517


namespace overlap_area_rhombus_l143_143605

noncomputable def area_of_overlap (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  1 / (Real.sin (α / 2))

theorem overlap_area_rhombus (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  area_of_overlap α hα = 1 / (Real.sin (α / 2)) :=
sorry

end overlap_area_rhombus_l143_143605


namespace milkman_cows_l143_143181

theorem milkman_cows (x : ℕ) (c : ℕ) :
  (3 * x * c = 720) ∧ (3 * x * c + 50 * c + 140 * c + 63 * c = 3250) → x = 24 :=
by
  sorry

end milkman_cows_l143_143181


namespace dog_total_distance_l143_143911

-- Define the conditions
def distance_between_A_and_B : ℝ := 100
def speed_A : ℝ := 6
def speed_B : ℝ := 4
def speed_dog : ℝ := 10

-- Define the statement we want to prove
theorem dog_total_distance : ∀ t : ℝ, (speed_A + speed_B) * t = distance_between_A_and_B → speed_dog * t = 100 :=
by
  intro t
  intro h
  sorry

end dog_total_distance_l143_143911


namespace remainder_problem_l143_143779

theorem remainder_problem
  (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 :=
by
  sorry

end remainder_problem_l143_143779


namespace city_population_l143_143914

theorem city_population (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 :=
by
  sorry

end city_population_l143_143914


namespace smallest_real_number_l143_143985

theorem smallest_real_number :
  ∃ (x : ℝ), x = -3 ∧ (∀ (y : ℝ), y = 0 ∨ y = (-1/3)^2 ∨ y = -((27:ℝ)^(1/3)) ∨ y = -2 → x ≤ y) := 
by 
  sorry

end smallest_real_number_l143_143985


namespace necessary_but_not_sufficient_for_inequalities_l143_143296

theorem necessary_but_not_sufficient_for_inequalities (a b : ℝ) :
  (a + b > 4) ↔ (a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequalities_l143_143296


namespace price_of_each_cake_is_correct_l143_143908

-- Define the conditions
def total_flour : ℕ := 6
def flour_for_cakes : ℕ := 4
def flour_per_cake : ℚ := 0.5
def remaining_flour := total_flour - flour_for_cakes
def flour_per_cupcake : ℚ := 1 / 5
def total_earnings : ℚ := 30
def cupcake_price : ℚ := 1

-- Number of cakes and cupcakes
def number_of_cakes := flour_for_cakes / flour_per_cake
def number_of_cupcakes := remaining_flour / flour_per_cupcake

-- Earnings from cupcakes
def earnings_from_cupcakes := number_of_cupcakes * cupcake_price

-- Earnings from cakes
def earnings_from_cakes := total_earnings - earnings_from_cupcakes

-- Price per cake
def price_per_cake := earnings_from_cakes / number_of_cakes

-- Final statement to prove
theorem price_of_each_cake_is_correct : price_per_cake = 2.50 := by
  sorry

end price_of_each_cake_is_correct_l143_143908


namespace number_of_elements_less_than_2004_l143_143712

theorem number_of_elements_less_than_2004 (f : ℕ → ℕ) 
    (h0 : f 0 = 0) 
    (h1 : ∀ n : ℕ, (f (2 * n + 1)) ^ 2 - (f (2 * n)) ^ 2 = 6 * f n + 1) 
    (h2 : ∀ n : ℕ, f (2 * n) > f n) 
  : ∃ m : ℕ,  m = 128 ∧ ∀ x : ℕ, f x < 2004 → x < m := sorry

end number_of_elements_less_than_2004_l143_143712


namespace isosceles_triangle_perimeter_l143_143441

theorem isosceles_triangle_perimeter
  (a b c : ℝ )
  (ha : a = 20)
  (hb : b = 20)
  (hc : c = (2/5) * 20)
  (triangle_ineq1 : a ≤ b + c)
  (triangle_ineq2 : b ≤ a + c)
  (triangle_ineq3 : c ≤ a + b) :
  a + b + c = 48 := by
  sorry

end isosceles_triangle_perimeter_l143_143441


namespace soccer_team_games_played_l143_143222

theorem soccer_team_games_played (t : ℝ) (h1 : 0.40 * t = 63.2) : t = 158 :=
sorry

end soccer_team_games_played_l143_143222


namespace arithmetic_sequence_ninth_term_l143_143686

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l143_143686


namespace percentage_reduction_of_faculty_l143_143431

noncomputable def percentage_reduction (original reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem percentage_reduction_of_faculty :
  percentage_reduction 226.74 195 = 13.99 :=
by sorry

end percentage_reduction_of_faculty_l143_143431


namespace number_of_trees_l143_143529

theorem number_of_trees (initial_trees planted_trees : ℕ)
  (h1 : initial_trees = 13)
  (h2 : planted_trees = 12) :
  initial_trees + planted_trees = 25 := by
  sorry

end number_of_trees_l143_143529


namespace trigonometric_identity_l143_143382

open Real

theorem trigonometric_identity (α β : ℝ) :
  sin (2 * α) ^ 2 + sin β ^ 2 + cos (2 * α + β) * cos (2 * α - β) = 1 :=
sorry

end trigonometric_identity_l143_143382


namespace fifth_student_guess_l143_143532

theorem fifth_student_guess (s1 s2 s3 s4 s5 : ℕ) 
(h1 : s1 = 100)
(h2 : s2 = 8 * s1)
(h3 : s3 = s2 - 200)
(h4 : s4 = (s1 + s2 + s3) / 3 + 25)
(h5 : s5 = s4 + s4 / 5) : 
s5 = 630 :=
sorry

end fifth_student_guess_l143_143532


namespace trig_identity_l143_143628

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l143_143628


namespace trigonometric_inequalities_l143_143191

noncomputable def a : ℝ := Real.sin (21 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (72 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (23 * Real.pi / 180)

-- The proof statement
theorem trigonometric_inequalities : c > a ∧ a > b :=
by
  sorry

end trigonometric_inequalities_l143_143191


namespace max_regions_with_five_lines_l143_143582

def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * (n + 1) / 2 + 1

theorem max_regions_with_five_lines (n : ℕ) (h : n = 5) : max_regions n = 16 :=
by {
  rw [h, max_regions];
  norm_num;
  done
}

end max_regions_with_five_lines_l143_143582


namespace terminal_side_in_second_quadrant_l143_143887

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (hcos : Real.cos α = -1/5) 
  (hsin : Real.sin α = 2 * Real.sqrt 6 / 5) : 
  (π / 2 < α ∧ α < π) :=
by
  sorry

end terminal_side_in_second_quadrant_l143_143887


namespace remainder_of_division_l143_143976

theorem remainder_of_division (x r : ℕ) (h1 : 1620 - x = 1365) (h2 : 1620 = x * 6 + r) : r = 90 :=
sorry

end remainder_of_division_l143_143976


namespace unique_solution_l143_143581

theorem unique_solution (k : ℝ) (h : k + 1 ≠ 0) : 
  (∀ x y : ℝ, ((x + 3) / (k * x + x - 3) = x) → ((y + 3) / (k * y + y - 3) = y) → x = y) ↔ k = -7/3 :=
by sorry

end unique_solution_l143_143581


namespace initial_amount_solution_l143_143396

noncomputable def initialAmount (P : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then P else (1 + 1/8) * initialAmount P (n - 1)

theorem initial_amount_solution (P : ℝ) (h₁ : initialAmount P 2 = 2025) : P = 1600 :=
  sorry

end initial_amount_solution_l143_143396


namespace probability_one_black_one_red_l143_143065

theorem probability_one_black_one_red (R B : Finset ℕ) (hR : R.card = 2) (hB : B.card = 3) :
  (2 : ℚ) / 5 = (6 + 6) / (5 * 4) := by
  sorry

end probability_one_black_one_red_l143_143065


namespace problem1_problem2_l143_143891

theorem problem1 :
  Real.sqrt 27 - (Real.sqrt 2 * Real.sqrt 6) + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := 
  by sorry

theorem problem2 :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 := 
  by sorry

end problem1_problem2_l143_143891


namespace custom_op_evaluation_l143_143951

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : custom_op 6 5 - custom_op 5 6 = -4 := by
  sorry

end custom_op_evaluation_l143_143951


namespace knights_and_liars_l143_143490

/--
Suppose we have a set of natives, each of whom is either a liar or a knight.
Each native declares to all others: "You are all liars."
This setup implies that there must be exactly one knight among them.
-/
theorem knights_and_liars (natives : Type) (is_knight : natives → Prop) (is_liar : natives → Prop)
  (h1 : ∀ x, is_knight x ∨ is_liar x) 
  (h2 : ∀ x y, x ≠ y → (is_knight x → is_liar y) ∧ (is_liar x → is_knight y))
  : ∃! x, is_knight x :=
by
  sorry

end knights_and_liars_l143_143490


namespace num_diagonals_29_sides_l143_143375

-- Define the number of sides
def n : Nat := 29

-- Calculate the combination (binomial coefficient) of selecting 2 vertices from n vertices
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the number of diagonals in a polygon with n sides
def num_diagonals (n : Nat) : Nat := binom n 2 - n

-- State the theorem to prove the number of diagonals for a polygon with 29 sides is 377
theorem num_diagonals_29_sides : num_diagonals 29 = 377 :=
by
  sorry

end num_diagonals_29_sides_l143_143375


namespace converse_angles_complements_l143_143566

theorem converse_angles_complements (α β : ℝ) (h : ∀γ : ℝ, α + γ = 90 ∧ β + γ = 90 → α = β) : 
  ∀ δ, α + δ = 90 ∧ β + δ = 90 → α = β :=
by 
  sorry

end converse_angles_complements_l143_143566


namespace arctg_inequality_l143_143862

theorem arctg_inequality (a b : ℝ) :
    |Real.arctan a - Real.arctan b| ≤ |b - a| := 
sorry

end arctg_inequality_l143_143862


namespace positive_solution_x_l143_143898

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 8 - x - 4 * y) (h2 : y * z = 12 - 3 * y - 6 * z) (h3 : x * z = 40 - 5 * x - 2 * z) (hy : y = 3) (hz : z = -1) : x = 6 :=
by
  sorry

end positive_solution_x_l143_143898


namespace largest_of_three_numbers_l143_143458

noncomputable def hcf := 23
noncomputable def factors := [11, 12, 13]

/-- The largest of the three numbers, given the H.C.F is 23 and the other factors of their L.C.M are 11, 12, and 13, is 39468. -/
theorem largest_of_three_numbers : hcf * factors.prod = 39468 := by
  sorry

end largest_of_three_numbers_l143_143458


namespace polynomial_divisibility_l143_143755

def P (a : ℤ) (x : ℤ) : ℤ := x^1000 + a*x^2 + 9

theorem polynomial_divisibility (a : ℤ) : (P a (-1) = 0) ↔ (a = -10) := by
  sorry

end polynomial_divisibility_l143_143755


namespace parabola_distance_l143_143930

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l143_143930


namespace max_bishops_1000x1000_l143_143642

def bishop_max_non_attacking (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem max_bishops_1000x1000 : bishop_max_non_attacking 1000 = 1998 :=
by sorry

end max_bishops_1000x1000_l143_143642


namespace gumball_water_wednesday_l143_143257

variable (water_Mon_Thu_Sat : ℕ)
variable (water_Tue_Fri_Sun : ℕ)
variable (water_total : ℕ)
variable (water_Wed : ℕ)

theorem gumball_water_wednesday 
  (h1 : water_Mon_Thu_Sat = 9) 
  (h2 : water_Tue_Fri_Sun = 8) 
  (h3 : water_total = 60) 
  (h4 : 3 * water_Mon_Thu_Sat + 3 * water_Tue_Fri_Sun + water_Wed = water_total) : 
  water_Wed = 9 := 
by 
  sorry

end gumball_water_wednesday_l143_143257


namespace midpoint_on_hyperbola_l143_143963

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l143_143963


namespace calculate_result_l143_143669

def binary_op (x y : ℝ) : ℝ := x^2 + y^2

theorem calculate_result (h : ℝ) : binary_op (binary_op h h) (binary_op h h) = 8 * h^4 :=
by
  sorry

end calculate_result_l143_143669


namespace percentage_error_computation_l143_143153

theorem percentage_error_computation (x : ℝ) (h : 0 < x) : 
  let correct_result := 8 * x
  let erroneous_result := x / 8
  let error := |correct_result - erroneous_result|
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 :=
by
  sorry

end percentage_error_computation_l143_143153


namespace number_of_students_playing_soccer_l143_143897

-- Definitions of the conditions
def total_students : ℕ := 500
def total_boys : ℕ := 350
def percent_boys_playing_soccer : ℚ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- To be proved
theorem number_of_students_playing_soccer :
  ∃ (S : ℕ), S = 250 ∧ 0.14 * (S : ℚ) = 35 :=
sorry

end number_of_students_playing_soccer_l143_143897


namespace amount_of_bill_is_720_l143_143371

-- Definitions and conditions
def TD : ℝ := 360
def BD : ℝ := 428.21

-- The relationship between TD, BD, and FV
axiom relationship (FV : ℝ) : BD = TD + (TD * BD) / (FV - TD)

-- The main theorem to prove
theorem amount_of_bill_is_720 : ∃ FV : ℝ, BD = TD + (TD * BD) / (FV - TD) ∧ FV = 720 :=
by
  use 720
  sorry

end amount_of_bill_is_720_l143_143371


namespace polynomial_term_equality_l143_143096

theorem polynomial_term_equality (p q : ℝ) (hpq_pos : 0 < p) (hq_pos : 0 < q) 
  (h_sum : p + q = 1) (h_eq : 28 * p^6 * q^2 = 56 * p^5 * q^3) : p = 2 / 3 :=
by
  sorry

end polynomial_term_equality_l143_143096


namespace calculation_proof_l143_143016

theorem calculation_proof :
  5^(Real.log 9 / Real.log 5) + (1 / 2) * (Real.log 32 / Real.log 2) - Real.log (Real.log 8 / Real.log 2) / Real.log 3 = 21 / 2 := 
  sorry

end calculation_proof_l143_143016


namespace Alyssa_total_spent_l143_143530

-- define the amounts spent on grapes and cherries
def costGrapes: ℝ := 12.08
def costCherries: ℝ := 9.85

-- define the total cost based on the given conditions
def totalCost: ℝ := costGrapes + costCherries

-- prove that the total cost equals 21.93
theorem Alyssa_total_spent:
  totalCost = 21.93 := 
  by
  -- proof to be completed
  sorry

end Alyssa_total_spent_l143_143530


namespace money_lent_years_l143_143555

noncomputable def compound_interest_time (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem money_lent_years :
  compound_interest_time 740 671.2018140589569 0.05 1 = 2 := by
  sorry

end money_lent_years_l143_143555


namespace negation_if_positive_then_square_positive_l143_143523

theorem negation_if_positive_then_square_positive :
  (¬ (∀ x : ℝ, x > 0 → x^2 > 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end negation_if_positive_then_square_positive_l143_143523


namespace minimum_rebate_rate_l143_143351

open Real

noncomputable def rebate_rate (s p_M p_N p: ℝ) : ℝ := 100 * (p_M + p_N - p) / s

theorem minimum_rebate_rate 
  (s p_M p_N p : ℝ)
  (h_M : 0.19 * 0.4 * s ≤ p_M ∧ p_M ≤ 0.24 * 0.4 * s)
  (h_N : 0.29 * 0.6 * s ≤ p_N ∧ p_N ≤ 0.34 * 0.6 * s)
  (h_total : 0.10 * s ≤ p ∧ p ≤ 0.15 * s) :
  ∃ r : ℝ, r = rebate_rate s p_M p_N p ∧ 0.1 ≤ r ∧ r ≤ 0.2 :=
sorry

end minimum_rebate_rate_l143_143351


namespace cistern_depth_l143_143164

noncomputable def length : ℝ := 9
noncomputable def width : ℝ := 4
noncomputable def total_wet_surface_area : ℝ := 68.5

theorem cistern_depth (h : ℝ) (h_def : 68.5 = 36 + 18 * h + 8 * h) : h = 1.25 :=
by sorry

end cistern_depth_l143_143164


namespace reciprocal_of_neg2_l143_143593

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l143_143593


namespace degree_g_greater_than_5_l143_143729

-- Definitions according to the given conditions
variables {f g : Polynomial ℤ}
variables (h : Polynomial ℤ)
variables (r : Fin 81 → ℤ)

-- Condition 1: g(x) divides f(x), meaning there exists an h(x) such that f(x) = g(x) * h(x)
def divides (g f : Polynomial ℤ) := ∃ (h : Polynomial ℤ), f = g * h

-- Condition 2: f(x) - 2008 has at least 81 distinct integer roots
def has_81_distinct_roots (f : Polynomial ℤ) (roots : Fin 81 → ℤ) : Prop :=
  ∀ i : Fin 81, f.eval (roots i) = 2008 ∧ Function.Injective roots

-- The theorem to prove
theorem degree_g_greater_than_5 (nonconst_f : f.degree > 0) (nonconst_g : g.degree > 0) 
  (g_div_f : divides g f) (f_has_roots : has_81_distinct_roots (f - Polynomial.C 2008) r) :
  g.degree > 5 :=
sorry

end degree_g_greater_than_5_l143_143729


namespace find_x_l143_143336

noncomputable def x_value : ℝ :=
  let x := 24
  x

theorem find_x (x : ℝ) (h : 7 * x + 3 * x + 4 * x + x = 360) : x = 24 := by
  sorry

end find_x_l143_143336


namespace R_depends_on_a_d_n_l143_143746

-- Definition of sum of an arithmetic progression
def sum_arithmetic_progression (n : ℕ) (a d : ℤ) : ℤ := 
  n * (2 * a + (n - 1) * d) / 2

-- Definitions for s1, s2, and s4
def s1 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression n a d
def s2 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (2 * n) a d
def s4 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (4 * n) a d

-- Definition of R
def R (n : ℕ) (a d : ℤ) : ℤ := s4 n a d - s2 n a d - s1 n a d

-- Theorem stating R depends on a, d, and n
theorem R_depends_on_a_d_n : 
  ∀ (n : ℕ) (a d : ℤ), ∃ (p q r : ℤ), R n a d = p * a + q * d + r := 
by
  sorry

end R_depends_on_a_d_n_l143_143746


namespace off_the_rack_suit_cost_l143_143359

theorem off_the_rack_suit_cost (x : ℝ)
  (h1 : ∀ y, y = 3 * x + 200)
  (h2 : ∀ y, x + y = 1400) :
  x = 300 :=
by
  sorry

end off_the_rack_suit_cost_l143_143359


namespace find_y_coordinate_l143_143749

theorem find_y_coordinate (m n : ℝ) 
  (h₁ : m = 2 * n + 5) 
  (h₂ : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := 
sorry

end find_y_coordinate_l143_143749


namespace largest_of_seven_consecutive_numbers_l143_143417

theorem largest_of_seven_consecutive_numbers (avg : ℕ) (h : avg = 20) :
  ∃ n : ℕ, n + 6 = 23 := 
by
  sorry

end largest_of_seven_consecutive_numbers_l143_143417


namespace units_digit_7_pow_5_l143_143379

theorem units_digit_7_pow_5 : (7 ^ 5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l143_143379


namespace john_read_books_in_15_hours_l143_143973

theorem john_read_books_in_15_hours (hreads_faster_ratio : ℝ) (brother_time : ℝ) (john_read_time : ℝ) : john_read_time = brother_time / hreads_faster_ratio → 3 * john_read_time = 15 :=
by
  intros H
  sorry

end john_read_books_in_15_hours_l143_143973


namespace coffee_table_price_correct_l143_143872

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l143_143872


namespace average_daily_production_correct_l143_143782

noncomputable def average_daily_production : ℝ :=
  let jan_production := 3000
  let monthly_increase := 100
  let total_days := 365
  let total_production := jan_production + (11 * jan_production + (100 * (1 + 11))/2)
  (total_production / total_days : ℝ)

theorem average_daily_production_correct :
  average_daily_production = 121.1 :=
sorry

end average_daily_production_correct_l143_143782


namespace kyle_caught_14_fish_l143_143303

theorem kyle_caught_14_fish (K T C : ℕ) (h1 : K = T) (h2 : C = 8) (h3 : C + K + T = 36) : K = 14 :=
by
  -- Proof goes here
  sorry

end kyle_caught_14_fish_l143_143303


namespace geometric_sum_n_is_4_l143_143907

theorem geometric_sum_n_is_4 
  (a r : ℚ) (n : ℕ) (S_n : ℚ) 
  (h1 : a = 1) 
  (h2 : r = 1 / 4) 
  (h3 : S_n = (a * (1 - r^n)) / (1 - r)) 
  (h4 : S_n = 85 / 64) : 
  n = 4 := 
sorry

end geometric_sum_n_is_4_l143_143907


namespace original_expenditure_mess_l143_143413

theorem original_expenditure_mess : 
  ∀ (x : ℝ), 
  35 * x + 42 = 42 * (x - 1) + 35 * x → 
  35 * 12 = 420 :=
by
  intro x
  intro h
  sorry

end original_expenditure_mess_l143_143413


namespace farmer_owned_land_l143_143957

theorem farmer_owned_land (T : ℝ) (h : 0.10 * T = 720) : 0.80 * T = 5760 :=
by
  sorry

end farmer_owned_land_l143_143957


namespace ratio_of_dividends_l143_143451

-- Definitions based on conditions
def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_per_increment : ℝ := 0.04
def increment_size : ℝ := 0.10

-- Definition for the base dividend D which remains undetermined
variable (D : ℝ)

-- Stating the theorem
theorem ratio_of_dividends 
  (h1 : actual_earnings = 1.10)
  (h2 : expected_earnings = 0.80)
  (h3 : additional_per_increment = 0.04)
  (h4 : increment_size = 0.10) :
  let additional_earnings := actual_earnings - expected_earnings
  let increments := additional_earnings / increment_size
  let additional_dividend := increments * additional_per_increment
  let total_dividend := D + additional_dividend
  let ratio := total_dividend / actual_earnings
  ratio = (D + 0.12) / 1.10 :=
by
  sorry

end ratio_of_dividends_l143_143451


namespace intersecting_lines_sum_l143_143554

theorem intersecting_lines_sum (a b : ℝ) (h1 : 2 = (1/3) * 4 + a) (h2 : 4 = (1/3) * 2 + b) : a + b = 4 :=
sorry

end intersecting_lines_sum_l143_143554


namespace value_of_a_l143_143564

theorem value_of_a (a : ℝ) : (|a| - 1 = 1) ∧ (a - 2 ≠ 0) → a = -2 :=
by
  sorry

end value_of_a_l143_143564


namespace old_manufacturing_cost_l143_143933

theorem old_manufacturing_cost (P : ℝ) :
  (50 : ℝ) = P * 0.50 →
  (0.65 : ℝ) * P = 65 :=
by
  intros hp₁
  -- Proof omitted
  sorry

end old_manufacturing_cost_l143_143933


namespace collinear_vectors_l143_143535

-- Definitions
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Proof statement
theorem collinear_vectors (x : ℝ) (h : ∃ k : ℝ, b x = k • a) : x = 3 :=
by sorry

end collinear_vectors_l143_143535


namespace eq_x4_inv_x4_l143_143946

theorem eq_x4_inv_x4 (x : ℝ) (h : x^2 + (1 / x^2) = 2) : 
  x^4 + (1 / x^4) = 2 := 
by 
  sorry

end eq_x4_inv_x4_l143_143946


namespace find_sum_of_natural_numbers_l143_143775

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end find_sum_of_natural_numbers_l143_143775


namespace train_length_l143_143328

noncomputable def length_of_train (t : ℝ) (v_train_kmh : ℝ) (v_man_kmh : ℝ) : ℝ :=
  let v_relative_kmh := v_train_kmh - v_man_kmh
  let v_relative_ms := v_relative_kmh * 1000 / 3600
  v_relative_ms * t

theorem train_length : length_of_train 30.99752019838413 80 8 = 619.9504039676826 := 
  by simp [length_of_train]; sorry

end train_length_l143_143328


namespace cubic_eq_one_real_root_l143_143929

/-- The equation x^3 - 4x^2 + 9x + c = 0 has exactly one real root for any real number c. -/
theorem cubic_eq_one_real_root (c : ℝ) : 
  ∃! x : ℝ, x^3 - 4 * x^2 + 9 * x + c = 0 :=
sorry

end cubic_eq_one_real_root_l143_143929


namespace euclidean_remainder_2022_l143_143217

theorem euclidean_remainder_2022 : 
  (2022 ^ (2022 ^ 2022)) % 11 = 5 := 
by sorry

end euclidean_remainder_2022_l143_143217


namespace probability_of_same_color_is_34_over_105_l143_143878

-- Define the number of each color of plates
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_yellow_plates : ℕ := 3

-- Define the total number of plates
def total_plates : ℕ := num_red_plates + num_blue_plates + num_yellow_plates

-- Define the total number of ways to choose 2 plates from the total plates
def total_ways_to_choose_2_plates : ℕ := Nat.choose total_plates 2

-- Define the number of ways to choose 2 red plates, 2 blue plates, and 2 yellow plates
def ways_to_choose_2_red_plates : ℕ := Nat.choose num_red_plates 2
def ways_to_choose_2_blue_plates : ℕ := Nat.choose num_blue_plates 2
def ways_to_choose_2_yellow_plates : ℕ := Nat.choose num_yellow_plates 2

-- Define the total number of favorable outcomes (same color plates)
def favorable_outcomes : ℕ :=
  ways_to_choose_2_red_plates + ways_to_choose_2_blue_plates + ways_to_choose_2_yellow_plates

-- Prove that the probability is 34/105
theorem probability_of_same_color_is_34_over_105 :
  (favorable_outcomes : ℚ) / (total_ways_to_choose_2_plates : ℚ) = 34 / 105 := by
  sorry

end probability_of_same_color_is_34_over_105_l143_143878


namespace avg_speed_ratio_l143_143437

theorem avg_speed_ratio 
  (dist_tractor : ℝ) (time_tractor : ℝ) 
  (dist_car : ℝ) (time_car : ℝ) 
  (speed_factor : ℝ) :
  dist_tractor = 575 -> 
  time_tractor = 23 ->
  dist_car = 450 ->
  time_car = 5 ->
  speed_factor = 2 ->

  (dist_car / time_car) / (speed_factor * (dist_tractor / time_tractor)) = 9/5 := 
by
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end avg_speed_ratio_l143_143437


namespace totalCoatsCollected_l143_143704

-- Definitions from the conditions
def highSchoolCoats : Nat := 6922
def elementarySchoolCoats : Nat := 2515

-- Theorem that proves the total number of coats collected
theorem totalCoatsCollected : highSchoolCoats + elementarySchoolCoats = 9437 := by
  sorry

end totalCoatsCollected_l143_143704


namespace AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l143_143270

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition
def AB2A_eq_AB := A * B ^ 2 * A = A * B * A

-- Part (a): Prove that (AB)^2 = AB
theorem AB_squared_eq_AB (h : AB2A_eq_AB A B) : (A * B) ^ 2 = A * B :=
sorry

-- Part (b): Prove that (AB - BA)^3 = 0
theorem AB_minus_BA_cubed_eq_zero (h : AB2A_eq_AB A B) : (A * B - B * A) ^ 3 = 0 :=
sorry

end AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l143_143270


namespace g_function_expression_l143_143968

theorem g_function_expression (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, g (-x) = g x) (h3 : ∀ x : ℝ, f x + g x = x^2 + a * x + 2 * a - 1) (h4 : f 1 = 2) :
  ∀ t : ℝ, g t = t^2 + 4 * t - 1 :=
by
  sorry

end g_function_expression_l143_143968


namespace curve_transformation_l143_143420

variable (x y x0 y0 : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def C (x0 y0 : ℝ) : Prop := (x0 - y0)^2 + y0^2 = 1

def transform (x0 y0 : ℝ) : ℝ × ℝ :=
  let x := 2 * x0 - 2 * y0
  let y := y0
  (x, y)

def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem curve_transformation :
  ∀ x0 y0, C x0 y0 → C' (2 * x0 - 2 * y0) y0 := sorry

end curve_transformation_l143_143420


namespace proof_f_2017_l143_143456

-- Define the conditions provided in the problem
variable (f : ℝ → ℝ)
variable (hf : ∀ x, f (-x) = -f x) -- f is an odd function
variable (h1 : ∀ x, f (-x + 1) = f (x + 1))
variable (h2 : f (-1) = 1)

-- Define the Lean statement that proves the correct answer
theorem proof_f_2017 : f 2017 = -1 :=
sorry

end proof_f_2017_l143_143456


namespace least_milk_l143_143803

theorem least_milk (seokjin jungkook yoongi : ℚ) (h_seokjin : seokjin = 11 / 10)
  (h_jungkook : jungkook = 1.3) (h_yoongi : yoongi = 7 / 6) :
  seokjin < jungkook ∧ seokjin < yoongi :=
by
  sorry

end least_milk_l143_143803


namespace count_three_digit_congruent_to_5_mod_7_l143_143869

theorem count_three_digit_congruent_to_5_mod_7 : 
  (100 ≤ 7 * k + 5 ∧ 7 * k + 5 ≤ 999) → ∃ n : ℕ, n = 129 := sorry

end count_three_digit_congruent_to_5_mod_7_l143_143869


namespace simplify_expression_evaluate_expression_l143_143815

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3 * a / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = a / (a - 2) :=
by sorry

theorem evaluate_expression :
  (-2 - 3 * (-2) / (-2 + 1)) / (((-2)^2 - 4 * (-2) + 4) / (-2 + 1)) = 1 / 2 :=
by sorry

end simplify_expression_evaluate_expression_l143_143815


namespace total_distance_traveled_l143_143236

theorem total_distance_traveled :
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  show total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2)
  sorry

end total_distance_traveled_l143_143236


namespace cylinder_height_relationship_l143_143266

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_equal : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relationship : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l143_143266


namespace composite_has_at_least_three_factors_l143_143108

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n :=
sorry

end composite_has_at_least_three_factors_l143_143108


namespace minimum_nine_points_distance_l143_143923

theorem minimum_nine_points_distance (n : ℕ) : 
  (∀ (p : Fin n → ℝ × ℝ),
    (∀ i, ∃! (four_points : List (Fin n)), 
      List.length four_points = 4 ∧ (∀ j ∈ four_points, dist (p i) (p j) = 1)))
    ↔ n = 9 :=
by 
  sorry

end minimum_nine_points_distance_l143_143923


namespace total_number_of_players_l143_143928

theorem total_number_of_players (n : ℕ) (h1 : n > 7) 
  (h2 : (4 * (n * (n - 1)) / 3 + 56 = (n + 8) * (n + 7) / 2)) : n + 8 = 50 :=
by
  sorry

end total_number_of_players_l143_143928


namespace geometric_sequence_S4_l143_143119

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n)

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 1 * ((1 - (a 2 / a 1)^(n+1)) / (1 - (a 2 / a 1)))

def given_condition (S : ℕ → ℝ) : Prop :=
S 7 - 4 * S 6 + 3 * S 5 = 0

-- Problem statement to prove
theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 1) (h_sum : sum_of_geometric_sequence a S) (h_cond : given_condition S) :
  S 4 = 40 := 
sorry

end geometric_sequence_S4_l143_143119


namespace sin_gamma_plus_delta_l143_143149

theorem sin_gamma_plus_delta (γ δ : ℝ) (hγ : Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
                             (hδ : Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = 33 / 65 :=
by
  sorry

end sin_gamma_plus_delta_l143_143149


namespace smallest_positive_angle_l143_143705

theorem smallest_positive_angle (α : ℝ) (h : (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) = (Real.sin α, Real.cos α)) : 
  α = 11 * Real.pi / 6 := by
sorry

end smallest_positive_angle_l143_143705


namespace intersection_correct_l143_143090

def M : Set Int := {-1, 1, 3, 5}
def N : Set Int := {-3, 1, 5}

theorem intersection_correct : M ∩ N = {1, 5} := 
by 
    sorry

end intersection_correct_l143_143090


namespace infection_probability_l143_143028

theorem infection_probability
  (malaria_percent : ℝ)
  (zika_percent : ℝ)
  (vaccine_reduction : ℝ)
  (prob_random_infection : ℝ)
  (P : ℝ) :
  malaria_percent = 0.40 →
  zika_percent = 0.20 →
  vaccine_reduction = 0.50 →
  prob_random_infection = 0.15 →
  0.15 = (0.40 * 0.50 * P) + (0.20 * P) →
  P = 0.375 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end infection_probability_l143_143028


namespace max_value_on_interval_l143_143326

noncomputable def f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 10

theorem max_value_on_interval :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, f 2 <= f x) → 
  ∃ y ∈ Set.Icc (1 : ℝ) 3, ∀ z ∈ Set.Icc (1 : ℝ) 3, f y >= f z :=
by
  sorry

end max_value_on_interval_l143_143326


namespace wechat_group_member_count_l143_143080

theorem wechat_group_member_count :
  (∃ x : ℕ, x * (x - 1) / 2 = 72) → ∃ x : ℕ, x = 9 :=
by
  sorry

end wechat_group_member_count_l143_143080


namespace total_rent_correct_recoup_investment_period_maximize_average_return_l143_143239

noncomputable def initialInvestment := 720000
noncomputable def firstYearRent := 54000
noncomputable def annualRentIncrease := 4000
noncomputable def maxRentalPeriod := 40

-- Conditions on the rental period
variable (x : ℝ) (hx : 0 < x ∧ x ≤ 40)

-- Function for total rent after x years
noncomputable def total_rent (x : ℝ) := 0.2 * x^2 + 5.2 * x

-- Condition for investment recoup period
noncomputable def recoupInvestmentTime := ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment

-- Function for transfer price
noncomputable def transfer_price (x : ℝ) := -0.3 * x^2 + 10.56 * x + 57.6

-- Function for average return on investment
noncomputable def annual_avg_return (x : ℝ) := (transfer_price x + total_rent x - initialInvestment) / x

-- Statement of theorems
theorem total_rent_correct (x : ℝ) (hx : 0 < x ∧ x ≤ 40) :
  total_rent x = 0.2 * x^2 + 5.2 * x := sorry

theorem recoup_investment_period :
  ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment := sorry

theorem maximize_average_return :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, annual_avg_return x ≥ annual_avg_return y) := sorry

end total_rent_correct_recoup_investment_period_maximize_average_return_l143_143239


namespace distance_probability_l143_143586

theorem distance_probability :
  let speed := 5
  let num_roads := 8
  let total_outcomes := num_roads * (num_roads - 1)
  let favorable_outcomes := num_roads * 3
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 0.375 :=
by
  sorry

end distance_probability_l143_143586


namespace carl_garden_area_l143_143355

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l143_143355


namespace acute_triangle_exists_l143_143655

theorem acute_triangle_exists {a1 a2 a3 a4 a5 : ℝ} 
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a2 + a3 > a4) (h5 : a3 + a4 > a2) (h6 : a2 + a4 > a3)
  (h7 : a3 + a4 > a5) (h8 : a4 + a5 > a3) (h9 : a3 + a5 > a4) : 
  ∃ (t1 t2 t3 : ℝ), (t1 + t2 > t3) ∧ (t1 + t3 > t2) ∧ (t2 + t3 > t1) ∧ (t3 ^ 2 < t1 ^ 2 + t2 ^ 2) :=
sorry

end acute_triangle_exists_l143_143655


namespace amount_of_medication_B_l143_143304

def medicationAmounts (x y : ℝ) : Prop :=
  (x + y = 750) ∧ (0.40 * x + 0.20 * y = 215)

theorem amount_of_medication_B (x y : ℝ) (h : medicationAmounts x y) : y = 425 :=
  sorry

end amount_of_medication_B_l143_143304


namespace max_s_value_l143_143690

variables (X Y Z P X' Y' Z' : Type)
variables (p q r XX' YY' ZZ' s : ℝ)

-- Defining the conditions
def triangle_XYZ (p q r : ℝ) : Prop :=
p ≤ r ∧ r ≤ q ∧ p + q > r ∧ p + r > q ∧ q + r > p

def point_P_inside (X Y Z P : Type) : Prop :=
true -- Simplified assumption since point P is given to be inside

def segments_XX'_YY'_ZZ' (XX' YY' ZZ' : ℝ) : ℝ :=
XX' + YY' + ZZ'

def given_ratio (p q r : ℝ) : Prop :=
(p / (q + r)) = (r / (p + q))

-- The maximum value of s being 3p
def max_value_s_eq_3p (s p : ℝ) : Prop :=
s = 3 * p

-- The final theorem statement
theorem max_s_value 
  (p q r XX' YY' ZZ' s : ℝ)
  (h_triangle : triangle_XYZ p q r)
  (h_ratio : given_ratio p q r)
  (h_segments : s = segments_XX'_YY'_ZZ' XX' YY' ZZ') : 
  max_value_s_eq_3p s p :=
by
  sorry

end max_s_value_l143_143690


namespace tiffany_won_lives_l143_143142
-- Step d: Lean 4 statement incorporating the conditions and the proof goal


-- Define initial lives, lives won in the hard part and the additional lives won
def initial_lives : Float := 43.0
def additional_lives : Float := 27.0
def total_lives_after_wins : Float := 84.0

open Classical

theorem tiffany_won_lives (x : Float) :
    initial_lives + x + additional_lives = total_lives_after_wins →
    x = 14.0 :=
by
  intros h
  -- This "sorry" indicates that the proof is skipped.
  sorry

end tiffany_won_lives_l143_143142


namespace original_price_of_dish_l143_143459

theorem original_price_of_dish : 
  ∀ (P : ℝ), 
  1.05 * P - 1.035 * P = 0.54 → 
  P = 36 :=
by
  intros P h
  sorry

end original_price_of_dish_l143_143459


namespace gcf_75_100_l143_143868

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l143_143868


namespace find_specific_linear_function_l143_143709

-- Define the linear function with given conditions
def linear_function (k b : ℝ) (x : ℝ) := k * x + b

-- Define the condition that the point lies on the line
def passes_through (k b : ℝ) (x y : ℝ) := y = linear_function k b x

-- Define the condition that slope is negative
def slope_negative (k : ℝ) := k < 0

-- The specific function we want to prove
def specific_linear_function (x : ℝ) := -x + 1

-- The theorem to prove
theorem find_specific_linear_function : 
  ∃ (k b : ℝ), slope_negative k ∧ passes_through k b 0 1 ∧ 
  ∀ x, linear_function k b x = specific_linear_function x :=
by
  sorry

end find_specific_linear_function_l143_143709


namespace pentagon_largest_angle_l143_143991

theorem pentagon_largest_angle
    (P Q : ℝ)
    (hP : P = 55)
    (hQ : Q = 120)
    (R S T : ℝ)
    (hR_eq_S : R = S)
    (hT : T = 2 * R + 20):
    R + S + T + P + Q = 540 → T = 192.5 :=
by
    sorry

end pentagon_largest_angle_l143_143991


namespace terry_problems_wrong_l143_143192

theorem terry_problems_wrong (R W : ℕ) 
  (h1 : R + W = 25) 
  (h2 : 4 * R - W = 85) : 
  W = 3 := 
by
  sorry

end terry_problems_wrong_l143_143192


namespace jasmine_added_is_8_l143_143278

noncomputable def jasmine_problem (J : ℝ) : Prop :=
  let initial_volume := 80
  let initial_jasmine_concentration := 0.10
  let initial_jasmine_amount := initial_volume * initial_jasmine_concentration

  let added_water := 12
  let final_volume := initial_volume + J + added_water
  let final_jasmine_concentration := 0.16
  let final_jasmine_amount := final_volume * final_jasmine_concentration

  initial_jasmine_amount + J = final_jasmine_amount 

theorem jasmine_added_is_8 : jasmine_problem 8 :=
by
  sorry

end jasmine_added_is_8_l143_143278


namespace deborah_oranges_zero_l143_143797

-- Definitions for given conditions.
def initial_oranges : Float := 55.0
def oranges_added_by_susan : Float := 35.0
def total_oranges_after : Float := 90.0

-- Defining Deborah's oranges in her bag.
def oranges_in_bag : Float := total_oranges_after - (initial_oranges + oranges_added_by_susan)

-- The theorem to be proved.
theorem deborah_oranges_zero : oranges_in_bag = 0 := by
  -- Placeholder for the proof.
  sorry

end deborah_oranges_zero_l143_143797


namespace ratio_of_arithmetic_seqs_l143_143640

noncomputable def arithmetic_seq_sum (a_1 a_n : ℕ) (n : ℕ) : ℝ := (n * (a_1 + a_n)) / 2

theorem ratio_of_arithmetic_seqs (a_1 a_6 a_11 b_1 b_6 b_11 : ℕ) :
  (∀ n : ℕ, (arithmetic_seq_sum a_1 a_n n) / (arithmetic_seq_sum b_1 b_n n) = n / (2 * n + 1))
  → (a_1 + a_6) / (b_1 + b_6) = 6 / 13
  → (a_1 + a_11) / (b_1 + b_11) = 11 / 23
  → (a_6 : ℝ) / (b_6 : ℝ) = 11 / 23 :=
  by
    intros h₁₁ h₆ h₁₁b
    sorry

end ratio_of_arithmetic_seqs_l143_143640


namespace forester_trees_planted_l143_143275

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end forester_trees_planted_l143_143275


namespace ratio_SP_CP_l143_143320

variables (CP SP P : ℝ)
axiom ratio_profit_CP : P / CP = 2

theorem ratio_SP_CP : SP / CP = 3 :=
by
  -- Proof statement (not required as per the instruction)
  sorry

end ratio_SP_CP_l143_143320


namespace matt_without_calculator_5_minutes_l143_143695

-- Define the conditions
def time_with_calculator (problems : Nat) : Nat := 2 * problems
def time_without_calculator (problems : Nat) (x : Nat) : Nat := x * problems
def time_saved (problems : Nat) (x : Nat) : Nat := time_without_calculator problems x - time_with_calculator problems

-- State the problem
theorem matt_without_calculator_5_minutes (x : Nat) :
  (time_saved 20 x = 60) → x = 5 := by
  sorry

end matt_without_calculator_5_minutes_l143_143695


namespace completing_square_result_l143_143599

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l143_143599


namespace ratio_of_percentage_change_l143_143161

theorem ratio_of_percentage_change
  (P U U' : ℝ)
  (h_price_decrease : U' = 4 * U)
  : (300 / 75) = 4 := 
by
  sorry

end ratio_of_percentage_change_l143_143161


namespace correct_inequality_l143_143546

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b)

theorem correct_inequality : (1 / (a * b^2)) < (1 / (a^2 * b)) :=
by
  sorry

end correct_inequality_l143_143546


namespace sum_odd_is_13_over_27_l143_143251

-- Define the probability for rolling an odd and an even number
def prob_odd := 1 / 3
def prob_even := 2 / 3

-- Define the probability that the sum of three die rolls is odd
def prob_sum_odd : ℚ :=
  3 * prob_odd * prob_even^2 + prob_odd^3

-- Statement asserting the goal to be proved
theorem sum_odd_is_13_over_27 :
  prob_sum_odd = 13 / 27 :=
by
  sorry

end sum_odd_is_13_over_27_l143_143251


namespace largest_prime_factor_l143_143851

theorem largest_prime_factor (a b c d : ℕ) (ha : a = 20) (hb : b = 15) (hc : c = 10) (hd : d = 5) :
  ∃ p, Nat.Prime p ∧ p = 103 ∧ ∀ q, Nat.Prime q ∧ q ∣ (a^3 + b^4 - c^5 + d^6) → q ≤ p :=
by
  sorry

end largest_prime_factor_l143_143851


namespace max_distinct_counts_proof_l143_143482

-- Define the number of boys (B) and girls (G)
def B : ℕ := 29
def G : ℕ := 15

-- Define the maximum distinct dance counts achievable
def max_distinct_counts : ℕ := 29

-- The theorem to prove
theorem max_distinct_counts_proof:
  ∃ (distinct_counts : ℕ), distinct_counts = max_distinct_counts ∧ distinct_counts <= B + G := 
by
  sorry

end max_distinct_counts_proof_l143_143482


namespace min_cells_marked_l143_143790

theorem min_cells_marked (grid_size : ℕ) (triomino_size : ℕ) (total_cells : ℕ) : 
  grid_size = 5 ∧ triomino_size = 3 ∧ total_cells = grid_size * grid_size → ∃ m, m = 9 :=
by
  intros h
  -- Placeholder for detailed proof steps
  sorry

end min_cells_marked_l143_143790


namespace arithmetic_sequence_a14_eq_41_l143_143177

theorem arithmetic_sequence_a14_eq_41 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 5) 
  (h_a6 : a 6 = 17) : 
  a 14 = 41 :=
sorry

end arithmetic_sequence_a14_eq_41_l143_143177


namespace num_integers_satisfying_abs_leq_bound_l143_143447

theorem num_integers_satisfying_abs_leq_bound : ∃ n : ℕ, n = 19 ∧ ∀ x : ℤ, |x| ≤ 3 * Real.sqrt 10 → (x ≥ -9 ∧ x ≤ 9) := by
  sorry

end num_integers_satisfying_abs_leq_bound_l143_143447


namespace weights_sum_l143_143692

theorem weights_sum (e f g h : ℕ) (h₁ : e + f = 280) (h₂ : f + g = 230) (h₃ : e + h = 300) : g + h = 250 := 
by 
  sorry

end weights_sum_l143_143692


namespace complex_number_identity_l143_143551

theorem complex_number_identity (z : ℂ) (h : z = 1 + (1 : ℂ) * I) : z^2 + z = 1 + 3 * I := 
sorry

end complex_number_identity_l143_143551


namespace percentage_problem_l143_143374

theorem percentage_problem (p x : ℝ) (h1 : (p / 100) * x = 400) (h2 : (120 / 100) * x = 2400) : p = 20 := by
  sorry

end percentage_problem_l143_143374


namespace cos_product_identity_l143_143258

theorem cos_product_identity :
  (Real.cos (20 * Real.pi / 180)) * (Real.cos (40 * Real.pi / 180)) *
  (Real.cos (60 * Real.pi / 180)) * (Real.cos (80 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end cos_product_identity_l143_143258


namespace alien_abduction_problem_l143_143412

theorem alien_abduction_problem:
  ∀ (total_abducted people_taken_elsewhere people_taken_home people_returned: ℕ),
  total_abducted = 200 →
  people_taken_elsewhere = 10 →
  people_taken_home = 30 →
  people_returned = total_abducted - (people_taken_elsewhere + people_taken_home) →
  (people_returned : ℕ) / total_abducted * 100 = 80 := 
by
  intros total_abducted people_taken_elsewhere people_taken_home people_returned;
  intros h_total_abducted h_taken_elsewhere h_taken_home h_people_returned;
  sorry

end alien_abduction_problem_l143_143412


namespace optimal_rental_plan_l143_143980

theorem optimal_rental_plan (a b x y : ℕ)
  (h1 : 2 * a + b = 10)
  (h2 : a + 2 * b = 11)
  (h3 : 31 = 3 * x + 4 * y)
  (cost_a : ℕ := 100)
  (cost_b : ℕ := 120) :
  ∃ x y, 3 * x + 4 * y = 31 ∧ cost_a * x + cost_b * y = 940 := by
  sorry

end optimal_rental_plan_l143_143980


namespace distance_traveled_l143_143627

-- Define the conditions
def rate : Real := 60  -- rate of 60 miles per hour
def total_break_time : Real := 1  -- total break time of 1 hour
def total_trip_time : Real := 9  -- total trip time of 9 hours

-- The theorem to prove the distance traveled
theorem distance_traveled : rate * (total_trip_time - total_break_time) = 480 := 
by
  sorry

end distance_traveled_l143_143627


namespace smallest_n_congruent_l143_143726

theorem smallest_n_congruent (n : ℕ) (h : 635 * n ≡ 1251 * n [MOD 30]) : n = 15 :=
sorry

end smallest_n_congruent_l143_143726


namespace gcd_incorrect_l143_143994

theorem gcd_incorrect (a b c : ℕ) (h : a * b * c = 3000) : gcd (gcd a b) c ≠ 15 := 
sorry

end gcd_incorrect_l143_143994


namespace find_z_l143_143180

theorem find_z (z : ℝ) (h : (z^2 - 5 * z + 6) / (z - 2) + (5 * z^2 + 11 * z - 32) / (5 * z - 16) = 1) : z = 1 :=
sorry

end find_z_l143_143180


namespace apples_total_l143_143272

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end apples_total_l143_143272


namespace sampling_method_is_systematic_sampling_l143_143242

-- Definitions based on the problem's conditions
def produces_products (factory : Type) : Prop := sorry
def uses_conveyor_belt (factory : Type) : Prop := sorry
def takes_item_every_5_minutes (inspector : Type) : Prop := sorry

-- Lean 4 statement to prove the question equals the answer given the conditions
theorem sampling_method_is_systematic_sampling
  (factory : Type)
  (inspector : Type)
  (h1 : produces_products factory)
  (h2 : uses_conveyor_belt factory)
  (h3 : takes_item_every_5_minutes inspector) :
  systematic_sampling_method := 
sorry

end sampling_method_is_systematic_sampling_l143_143242


namespace ratio_a7_b7_l143_143618

variables (a b : ℕ → ℤ) (Sa Tb : ℕ → ℤ)
variables (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
variables (h2 : ∀ n : ℕ, b n = b 0 + n * (b 1 - b 0))
variables (h3 : ∀ n : ℕ, Sa n = n * (a 0 + a n) / 2)
variables (h4 : ∀ n : ℕ, Tb n = n * (b 0 + b n) / 2)
variables (h5 : ∀ n : ℕ, n > 0 → Sa n / Tb n = (7 * n + 1) / (4 * n + 27))

theorem ratio_a7_b7 : ∀ n : ℕ, n = 7 → a 7 / b 7 = 92 / 79 :=
by
  intros n hn_eq
  sorry

end ratio_a7_b7_l143_143618


namespace radian_measure_of_sector_l143_143821

-- Lean statement for the proof problem
theorem radian_measure_of_sector (R : ℝ) (hR : 0 < R) (h_area : (1 / 2) * (2 : ℝ) * R^2 = R^2) : 
  (2 : ℝ) = 2 :=
by 
  sorry
 
end radian_measure_of_sector_l143_143821


namespace units_digit_product_is_2_l143_143066

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2_l143_143066


namespace twenty_five_percent_M_eq_thirty_five_percent_1504_l143_143903

theorem twenty_five_percent_M_eq_thirty_five_percent_1504 (M : ℝ) : 
  0.25 * M = 0.35 * 1504 → M = 2105.6 :=
by
  sorry

end twenty_five_percent_M_eq_thirty_five_percent_1504_l143_143903


namespace count_integers_l143_143915

theorem count_integers (n : ℤ) (h : -11 ≤ n ∧ n ≤ 11) : ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x ∈ s, (x - 1) * (x + 3) * (x + 7) < 0 :=
by
  sorry

end count_integers_l143_143915


namespace square_rem_1_mod_9_l143_143173

theorem square_rem_1_mod_9 (N : ℤ) (h : N % 9 = 1 ∨ N % 9 = 8) : (N * N) % 9 = 1 :=
by sorry

end square_rem_1_mod_9_l143_143173


namespace demand_decrease_l143_143254

theorem demand_decrease (original_price_increase effective_price_increase demand_decrease : ℝ)
  (h1 : original_price_increase = 0.2)
  (h2 : effective_price_increase = original_price_increase / 2)
  (h3 : new_price = original_price * (1 + effective_price_increase))
  (h4 : 1 / new_price = original_demand)
  : demand_decrease = 0.0909 := sorry

end demand_decrease_l143_143254


namespace johns_donation_l143_143314

theorem johns_donation
    (A T D : ℝ)
    (n : ℕ)
    (hA1 : A * 1.75 = 100)
    (hA2 : A = 100 / 1.75)
    (hT : T = 10 * A)
    (hD : D = 11 * 100 - T)
    (hn : n = 10) :
    D = 3700 / 7 := 
sorry

end johns_donation_l143_143314


namespace smallest_positive_multiple_45_l143_143486

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l143_143486


namespace number_of_new_students_l143_143591

theorem number_of_new_students (initial_students end_students students_left : ℕ) 
  (h_initial: initial_students = 33) 
  (h_left: students_left = 18) 
  (h_end: end_students = 29) : 
  initial_students - students_left + (end_students - (initial_students - students_left)) = 14 :=
by
  sorry

end number_of_new_students_l143_143591


namespace polynomial_expansion_l143_143728

noncomputable def poly1 (z : ℝ) : ℝ := 3 * z ^ 3 + 2 * z ^ 2 - 4 * z + 1
noncomputable def poly2 (z : ℝ) : ℝ := 2 * z ^ 4 - 3 * z ^ 2 + z - 5
noncomputable def expanded_poly (z : ℝ) : ℝ := 6 * z ^ 7 + 4 * z ^ 6 - 4 * z ^ 5 - 9 * z ^ 3 + 7 * z ^ 2 + z - 5

theorem polynomial_expansion (z : ℝ) : poly1 z * poly2 z = expanded_poly z := by
  sorry

end polynomial_expansion_l143_143728


namespace anya_takes_home_balloons_l143_143780

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end anya_takes_home_balloons_l143_143780


namespace find_geometric_sequence_term_l143_143661

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l143_143661


namespace burmese_pythons_required_l143_143070

theorem burmese_pythons_required (single_python_rate : ℕ) (total_alligators : ℕ) (total_weeks : ℕ) (required_pythons : ℕ) :
  single_python_rate = 1 →
  total_alligators = 15 →
  total_weeks = 3 →
  required_pythons = total_alligators / total_weeks →
  required_pythons = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at *
  simp at h4
  sorry

end burmese_pythons_required_l143_143070


namespace gym_guest_count_l143_143751

theorem gym_guest_count (G : ℕ) (H1 : ∀ G, 0 < G → ∀ G, G * 5.7 = 285 ∧ G = 50) : G = 50 :=
by
  sorry

end gym_guest_count_l143_143751


namespace total_letters_in_names_is_33_l143_143559

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l143_143559


namespace bea_has_max_profit_l143_143986

theorem bea_has_max_profit : 
  let price_bea := 25
  let price_dawn := 28
  let price_carla := 35
  let sold_bea := 10
  let sold_dawn := 8
  let sold_carla := 6
  let cost_bea := 10
  let cost_dawn := 12
  let cost_carla := 15
  let profit_bea := (price_bea * sold_bea) - (cost_bea * sold_bea)
  let profit_dawn := (price_dawn * sold_dawn) - (cost_dawn * sold_dawn)
  let profit_carla := (price_carla * sold_carla) - (cost_carla * sold_carla)
  profit_bea = 150 ∧ profit_dawn = 128 ∧ profit_carla = 120 ∧ ∀ p, p ∈ [profit_bea, profit_dawn, profit_carla] → p ≤ 150 :=
by
  sorry

end bea_has_max_profit_l143_143986


namespace total_votes_cast_l143_143331

variable (total_votes : ℕ)
variable (emily_votes : ℕ)
variable (emily_share : ℚ := 4 / 15)
variable (dexter_share : ℚ := 1 / 3)

theorem total_votes_cast :
  emily_votes = 48 → 
  emily_share * total_votes = emily_votes → 
  total_votes = 180 := by
  intro h_emily_votes
  intro h_emily_share
  sorry

end total_votes_cast_l143_143331


namespace total_detergent_is_19_l143_143201

-- Define the quantities and usage of detergent
def detergent_per_pound_cotton := 2
def detergent_per_pound_woolen := 3
def detergent_per_pound_synthetic := 1

def pounds_of_cotton := 4
def pounds_of_woolen := 3
def pounds_of_synthetic := 2

-- Define the function to calculate the total amount of detergent needed
def total_detergent_needed := 
  detergent_per_pound_cotton * pounds_of_cotton +
  detergent_per_pound_woolen * pounds_of_woolen +
  detergent_per_pound_synthetic * pounds_of_synthetic

-- The theorem to prove the total amount of detergent used
theorem total_detergent_is_19 : total_detergent_needed = 19 :=
  by { sorry }

end total_detergent_is_19_l143_143201


namespace cost_for_15_pounds_of_apples_l143_143724

-- Axiom stating the cost of apples per weight
axiom cost_of_apples (pounds : ℕ) : ℕ

-- Condition given in the problem
def rate_apples : Prop := cost_of_apples 5 = 4

-- Statement of the problem
theorem cost_for_15_pounds_of_apples : rate_apples → cost_of_apples 15 = 12 :=
by
  intro h
  -- Proof to be filled in here
  sorry

end cost_for_15_pounds_of_apples_l143_143724


namespace domain_of_x_l143_143020

-- Conditions
def is_defined_num (x : ℝ) : Prop := x + 1 >= 0
def not_zero_den (x : ℝ) : Prop := x ≠ 2

-- Proof problem statement
theorem domain_of_x (x : ℝ) : (is_defined_num x ∧ not_zero_den x) ↔ (x >= -1 ∧ x ≠ 2) := by
  sorry

end domain_of_x_l143_143020


namespace muffin_to_banana_ratio_l143_143877

variables (m b : ℝ) -- initial cost of a muffin and a banana

-- John's total cost for muffins and bananas
def johns_cost (m b : ℝ) : ℝ :=
  3 * m + 4 * b

-- Martha's total cost for muffins and bananas based on increased prices
def marthas_cost_increased (m b : ℝ) : ℝ :=
  5 * (1.2 * m) + 12 * (1.5 * b)

-- John's total cost times three
def marthas_cost_original_times_three (m b : ℝ) : ℝ :=
  3 * (johns_cost m b)

-- The theorem to prove
theorem muffin_to_banana_ratio
  (h3m4b_eq : johns_cost m b * 3 = marthas_cost_increased m b)
  (hm_eq_2b : m = 2 * b) :
  (1.2 * m) / (1.5 * b) = 4 / 5 := by
  sorry

end muffin_to_banana_ratio_l143_143877


namespace Joey_weekend_study_hours_l143_143139

noncomputable def hours_weekday_per_week := 2 * 5 -- 2 hours/night * 5 nights/week
noncomputable def total_hours_weekdays := hours_weekday_per_week * 6 -- Multiply by 6 weeks
noncomputable def remaining_hours_weekends := 96 - total_hours_weekdays -- 96 total hours - weekday hours
noncomputable def total_weekend_days := 6 * 2 -- 6 weekends * 2 days/weekend
noncomputable def hours_per_day_weekend := remaining_hours_weekends / total_weekend_days

theorem Joey_weekend_study_hours : hours_per_day_weekend = 3 :=
by
  sorry

end Joey_weekend_study_hours_l143_143139


namespace annual_interest_rate_l143_143097

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

-- Define the given parameters
def P : ℝ := 1200
def A : ℝ := 2488.32
def n : ℕ := 1
def t : ℕ := 4

theorem annual_interest_rate : compound_interest_rate P A n t = 0.25 :=
by
  sorry

end annual_interest_rate_l143_143097


namespace number_50_is_sample_size_l143_143025

def number_of_pairs : ℕ := 50
def is_sample_size (n : ℕ) : Prop := n = number_of_pairs

-- We are to show that 50 represents the sample size
theorem number_50_is_sample_size : is_sample_size 50 :=
sorry

end number_50_is_sample_size_l143_143025


namespace find_a_l143_143291

theorem find_a (a : ℤ) : 
  (∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^3)) ↔ (a = 3^9) :=
by
  sorry

end find_a_l143_143291


namespace fraction_comparison_l143_143104

theorem fraction_comparison :
  (1998:ℝ) ^ 2000 / (2000:ℝ) ^ 1998 > (1997:ℝ) ^ 1999 / (1999:ℝ) ^ 1997 :=
by sorry

end fraction_comparison_l143_143104


namespace z_share_profit_correct_l143_143273

-- Define the investments as constants
def x_investment : ℕ := 20000
def y_investment : ℕ := 25000
def z_investment : ℕ := 30000

-- Define the number of months for each investment
def x_months : ℕ := 12
def y_months : ℕ := 12
def z_months : ℕ := 7

-- Define the annual profit
def annual_profit : ℕ := 50000

-- Calculate the active investment
def x_share : ℕ := x_investment * x_months
def y_share : ℕ := y_investment * y_months
def z_share : ℕ := z_investment * z_months

-- Calculate the total investment
def total_investment : ℕ := x_share + y_share + z_share

-- Define Z's ratio in terms of the total investment
def z_ratio : ℚ := z_share / total_investment

-- Calculate Z's share of the annual profit
def z_profit_share : ℚ := z_ratio * annual_profit

-- Theorem to prove Z's share in the annual profit
theorem z_share_profit_correct : z_profit_share = 14000 := by
  sorry

end z_share_profit_correct_l143_143273


namespace relation_between_p_and_q_l143_143042

theorem relation_between_p_and_q (p q : ℝ) (α : ℝ) 
  (h1 : α + 2 * α = -p) 
  (h2 : α * (2 * α) = q) : 
  2 * p^2 = 9 * q := 
by 
  -- simplifying the provided conditions
  sorry

end relation_between_p_and_q_l143_143042


namespace chessboard_game_winner_l143_143979

theorem chessboard_game_winner (m n : ℕ) (initial_position : ℕ × ℕ) :
  (m * n) % 2 = 0 → (∃ A_wins : Prop, A_wins) ∧ 
  (m * n) % 2 = 1 → (∃ B_wins : Prop, B_wins) :=
by
  sorry

end chessboard_game_winner_l143_143979


namespace largest_lcm_18_l143_143460

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l143_143460


namespace average_speed_is_9_mph_l143_143068

-- Define the conditions
def distance_north_ft := 5280
def north_speed_min_per_mile := 3
def rest_time_min := 10
def south_speed_miles_per_min := 3

-- Define a function to convert feet to miles
def feet_to_miles (ft : ℕ) : ℕ := ft / 5280

-- Define the time calculation for north and south trips
def time_north_min (speed : ℕ) (distance_ft : ℕ) : ℕ :=
  speed * feet_to_miles distance_ft

def time_south_min (speed_miles_per_min : ℕ) (distance_ft : ℕ) : ℕ :=
  (feet_to_miles distance_ft) / speed_miles_per_min

def total_time_min (time_north rest_time time_south : ℕ) : Rat :=
  time_north + rest_time + time_south

-- Convert total time into hours
def total_time_hr (total_time_min : Rat) : Rat :=
  total_time_min / 60

-- Define the total distance in miles
def total_distance_miles (distance_ft : ℕ) : ℕ :=
  2 * feet_to_miles distance_ft

-- Calculate the average speed
def average_speed (total_distance : ℕ) (total_time_hr : Rat) : Rat :=
  total_distance / total_time_hr

-- Prove the average speed is 9 miles per hour
theorem average_speed_is_9_mph : 
  average_speed (total_distance_miles distance_north_ft)
                (total_time_hr (total_time_min (time_north_min north_speed_min_per_mile distance_north_ft)
                                              rest_time_min
                                              (time_south_min south_speed_miles_per_min distance_north_ft)))
    = 9 := by
  sorry

end average_speed_is_9_mph_l143_143068


namespace bread_slices_l143_143936

theorem bread_slices (c : ℕ) (cost_each_slice_in_cents : ℕ)
  (total_paid_in_cents : ℕ) (change_in_cents : ℕ) (n : ℕ) (slices_per_loaf : ℕ) :
  c = 3 →
  cost_each_slice_in_cents = 40 →
  total_paid_in_cents = 2 * 2000 →
  change_in_cents = 1600 →
  total_paid_in_cents - change_in_cents = n * cost_each_slice_in_cents →
  n = c * slices_per_loaf →
  slices_per_loaf = 20 :=
by sorry

end bread_slices_l143_143936


namespace calculate_gross_profit_l143_143853

theorem calculate_gross_profit (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 81)
    (h2 : gross_profit = 1.70 * cost)
    (h3 : sales_price = cost + gross_profit) : gross_profit = 51 :=
by
  sorry

end calculate_gross_profit_l143_143853


namespace arun_working_days_l143_143084

theorem arun_working_days (A T : ℝ) 
  (h1 : A + T = 1/10) 
  (h2 : A = 1/18) : 
  (1 / A) = 18 :=
by
  -- Proof will be skipped
  sorry

end arun_working_days_l143_143084


namespace amy_soups_total_l143_143439

def total_soups (chicken_soups tomato_soups : ℕ) : ℕ :=
  chicken_soups + tomato_soups

theorem amy_soups_total : total_soups 6 3 = 9 :=
by
  -- insert the proof here
  sorry

end amy_soups_total_l143_143439


namespace samuel_distance_from_hotel_l143_143615

/-- Samuel's driving problem conditions. -/
structure DrivingConditions where
  total_distance : ℕ -- in miles
  first_speed : ℕ -- in miles per hour
  first_time : ℕ -- in hours
  second_speed : ℕ -- in miles per hour
  second_time : ℕ -- in hours

def distance_remaining (c : DrivingConditions) : ℕ :=
  let distance_covered := (c.first_speed * c.first_time) + (c.second_speed * c.second_time)
  c.total_distance - distance_covered

/-- Prove that Samuel is 130 miles from the hotel. -/
theorem samuel_distance_from_hotel : 
  ∀ (c : DrivingConditions), 
    c.total_distance = 600 ∧
    c.first_speed = 50 ∧
    c.first_time = 3 ∧
    c.second_speed = 80 ∧
    c.second_time = 4 → distance_remaining c = 130 := by
  intros c h
  cases h
  sorry

end samuel_distance_from_hotel_l143_143615


namespace coefficients_sum_eq_zero_l143_143578

theorem coefficients_sum_eq_zero 
  (a b c : ℝ)
  (f g h : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = b * x^2 + c * x + a)
  (h3 : ∀ x, h x = c * x^2 + a * x + b)
  (h4 : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) :
  a + b + c = 0 := 
sorry

end coefficients_sum_eq_zero_l143_143578


namespace canoe_rental_cost_l143_143955

theorem canoe_rental_cost :
  ∃ (C : ℕ) (K : ℕ), 
  (15 * K + C * (K + 4) = 288) ∧ 
  (3 * K + 12 = 12 * C) ∧ 
  (C = 14) :=
sorry

end canoe_rental_cost_l143_143955


namespace find_largest_number_l143_143052

noncomputable def largest_number (a b c : ℚ) : ℚ :=
  if a + b + c = 77 ∧ c - b = 9 ∧ b - a = 5 then c else 0

theorem find_largest_number (a b c : ℚ) 
  (h1 : a + b + c = 77) 
  (h2 : c - b = 9) 
  (h3 : b - a = 5) : 
  c = 100 / 3 := 
sorry

end find_largest_number_l143_143052


namespace correct_average_l143_143244

theorem correct_average
  (incorrect_avg : ℝ)
  (incorrect_num correct_num : ℝ)
  (n : ℕ)
  (h1 : incorrect_avg = 16)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 46)
  (h4 : n = 10) :
  (incorrect_avg * n - incorrect_num + correct_num) / n = 18 :=
sorry

end correct_average_l143_143244


namespace first_student_can_ensure_one_real_root_l143_143600

noncomputable def can_first_student_ensure_one_real_root : Prop :=
  ∀ (b c : ℝ), ∃ a : ℝ, ∃ d : ℝ, ∀ (e : ℝ), 
    (d = 0 ∧ (e = b ∨ e = c)) → 
    (∀ x : ℝ, x^3 + d * x^2 + e * x + (if e = b then c else b) = 0)

theorem first_student_can_ensure_one_real_root :
  can_first_student_ensure_one_real_root := sorry

end first_student_can_ensure_one_real_root_l143_143600


namespace find_f_sqrt2_l143_143322

theorem find_f_sqrt2 (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x * y) = f x + f y)
  (hf8 : f 8 = 3) :
  f (Real.sqrt 2) = 1 / 2 := by
  sorry

end find_f_sqrt2_l143_143322


namespace problem_1_problem_2_l143_143160

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem_1_problem_2_l143_143160


namespace cost_per_revision_l143_143237

theorem cost_per_revision
  (x : ℝ)
  (initial_cost : ℝ)
  (revised_once : ℝ)
  (revised_twice : ℝ)
  (total_pages : ℝ)
  (total_cost : ℝ)
  (cost_per_page_first_time : ℝ) :
  initial_cost = cost_per_page_first_time * total_pages →
  revised_once * x + revised_twice * (2 * x) + initial_cost = total_cost →
  revised_once + revised_twice + (total_pages - (revised_once + revised_twice)) = total_pages →
  total_pages = 200 →
  initial_cost = 1000 →
  cost_per_page_first_time = 5 →
  revised_once = 80 →
  revised_twice = 20 →
  total_cost = 1360 →
  x = 3 :=
by
  intros h_initial h_total_cost h_tot_pages h_tot_pages_200 h_initial_1000 h_cost_5 h_revised_once h_revised_twice h_given_cost
  -- Proof steps to be filled
  sorry

end cost_per_revision_l143_143237


namespace dogs_count_l143_143565

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l143_143565


namespace company_match_percentage_l143_143641

theorem company_match_percentage (total_contribution : ℝ) (holly_contribution_per_paycheck : ℝ) (total_paychecks : ℕ) (total_contribution_one_year : ℝ) : 
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  (company_contribution / holly_contribution) * 100 = 6 :=
by
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  have h : holly_contribution = 2600 := by sorry
  have c : company_contribution = 156 := by sorry
  exact sorry

end company_match_percentage_l143_143641


namespace log_arith_example_l143_143967

noncomputable def log10 (x : ℝ) : ℝ := sorry -- Assume the definition of log base 10

theorem log_arith_example : log10 4 + 2 * log10 5 + 8^(2/3) = 6 := 
by
  -- The proof would go here
  sorry

end log_arith_example_l143_143967


namespace polygon_sides_l143_143436

theorem polygon_sides (R : ℝ) (n : ℕ) (h : R ≠ 0)
  (h_area : (1 / 2) * n * R^2 * Real.sin (360 / n * (Real.pi / 180)) = 4 * R^2) :
  n = 8 := 
by
  sorry

end polygon_sides_l143_143436


namespace second_hand_travel_distance_l143_143361

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l143_143361


namespace value_of_A_l143_143894

theorem value_of_A
  (A B C D E F G H I J : ℕ)
  (h_diff : ∀ x y : ℕ, x ≠ y → x ≠ y)
  (h_decreasing_ABC : A > B ∧ B > C)
  (h_decreasing_DEF : D > E ∧ E > F)
  (h_decreasing_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_odd_DEF : D % 2 = 1 ∧ E % 2 = 1 ∧ F % 2 = 1 ∧ E = D - 2 ∧ F = E - 2)
  (h_consecutive_even_GHIJ : G % 2 = 0 ∧ H % 2 = 0 ∧ I % 2 = 0 ∧ J % 2 = 0 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) : 
  A = 8 :=
sorry

end value_of_A_l143_143894


namespace difference_of_numbers_l143_143421

theorem difference_of_numbers (x y : ℝ) (h₁ : x + y = 25) (h₂ : x * y = 144) : |x - y| = 7 :=
sorry

end difference_of_numbers_l143_143421


namespace triangle_side_length_l143_143392

theorem triangle_side_length (a : ℝ) (B : ℝ) (C : ℝ) (c : ℝ) 
  (h₀ : a = 10) (h₁ : B = 60) (h₂ : C = 45) : 
  c = 10 * (Real.sqrt 3 - 1) :=
sorry

end triangle_side_length_l143_143392


namespace valid_functions_l143_143611

theorem valid_functions (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) * g (x - y) = (g x + g y)^2 - 4 * x^2 * g y + 2 * y^2 * g x) :
  (∀ x, g x = 0) ∨ (∀ x, g x = x^2) :=
by sorry

end valid_functions_l143_143611


namespace train_length_l143_143295

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end train_length_l143_143295


namespace population_is_24000_l143_143410

theorem population_is_24000 (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 := sorry

end population_is_24000_l143_143410


namespace sector_area_correct_l143_143950

-- Definitions based on the conditions
def sector_perimeter := 16 -- cm
def central_angle := 2 -- radians
def radius := 4 -- The radius computed from perimeter condition

-- Lean 4 statement to prove the equivalent math problem
theorem sector_area_correct : ∃ (s : ℝ), 
  (∀ (r : ℝ), (2 * r + r * central_angle = sector_perimeter → r = 4) → 
  (s = (1 / 2) * central_angle * (radius) ^ 2) → 
  s = 16) :=
by 
  sorry

end sector_area_correct_l143_143950


namespace starfish_arms_l143_143727

variable (x : ℕ)

theorem starfish_arms :
  (7 * x + 14 = 49) → (x = 5) := by
  sorry

end starfish_arms_l143_143727


namespace lena_muffins_l143_143299

theorem lena_muffins (x y z : Real) 
  (h1 : x + 2 * y + 3 * z = 3 * x + z)
  (h2 : 3 * x + z = 6 * y)
  (h3 : x + 2 * y + 3 * z = 6 * y)
  (lenas_spending : 2 * x + 2 * z = 6 * y) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end lena_muffins_l143_143299


namespace apple_consumption_l143_143058

-- Definitions for the portions of the apple above and below water
def portion_above_water := 1 / 5
def portion_below_water := 4 / 5

-- Rates of consumption by fish and bird
def fish_rate := 120  -- grams per minute
def bird_rate := 60  -- grams per minute

-- The question statements with the correct answers
theorem apple_consumption :
  (portion_below_water * (fish_rate / (fish_rate + bird_rate)) = 2 / 3) ∧ 
  (portion_above_water * (bird_rate / (fish_rate + bird_rate)) = 1 / 3) := 
sorry

end apple_consumption_l143_143058


namespace product_of_three_numbers_l143_143737

theorem product_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 210) (h2 : 5 * a = b - 11) (h3 : 5 * a = c + 11) : a * b * c = 168504 :=
  sorry

end product_of_three_numbers_l143_143737


namespace num_two_digit_primes_with_ones_digit_3_l143_143778

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l143_143778


namespace min_value_ineq_l143_143186

theorem min_value_ineq (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := 
sorry

end min_value_ineq_l143_143186


namespace largest_4_digit_divisible_by_98_l143_143157

theorem largest_4_digit_divisible_by_98 :
  ∃ n, (n ≤ 9999 ∧ 9999 < n + 98) ∧ 98 ∣ n :=
sorry

end largest_4_digit_divisible_by_98_l143_143157


namespace eggs_in_fridge_l143_143536

theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) (eggs_used : ℕ) (eggs_in_fridge : ℕ)
  (h1 : total_eggs = 60)
  (h2 : eggs_per_cake = 5)
  (h3 : num_cakes = 10)
  (h4 : eggs_used = eggs_per_cake * num_cakes)
  (h5 : eggs_in_fridge = total_eggs - eggs_used) :
  eggs_in_fridge = 10 :=
by
  sorry

end eggs_in_fridge_l143_143536


namespace vertex_angle_isosceles_triangle_l143_143701

theorem vertex_angle_isosceles_triangle (B V : ℝ) (h1 : 2 * B + V = 180) (h2 : B = 40) : V = 100 :=
by
  sorry

end vertex_angle_isosceles_triangle_l143_143701


namespace susie_investment_l143_143130

theorem susie_investment :
  ∃ x : ℝ, x * (1 + 0.04)^3 + (2000 - x) * (1 + 0.06)^3 = 2436.29 → x = 820 :=
by
  sorry

end susie_investment_l143_143130


namespace Penelope_daily_savings_l143_143575

theorem Penelope_daily_savings
  (total_savings : ℝ)
  (days_in_year : ℕ)
  (h1 : total_savings = 8760)
  (h2 : days_in_year = 365) :
  total_savings / days_in_year = 24 :=
by
  sorry

end Penelope_daily_savings_l143_143575


namespace digit_for_divisibility_by_5_l143_143765

theorem digit_for_divisibility_by_5 (B : ℕ) (h : B < 10) :
  (∃ (n : ℕ), n = 527 * 10 + B ∧ n % 5 = 0) ↔ (B = 0 ∨ B = 5) :=
by sorry

end digit_for_divisibility_by_5_l143_143765


namespace number_of_ways_to_select_officers_l143_143515

-- Definitions based on conditions
def boys : ℕ := 6
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def officers_to_select : ℕ := 3

-- Number of ways to choose 3 individuals out of 10
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_choices : ℕ := choose total_people officers_to_select

-- Number of ways to choose 3 boys out of 6 (0 girls)
def all_boys_choices : ℕ := choose boys officers_to_select

-- Number of ways to choose at least 1 girl
def at_least_one_girl_choices : ℕ := total_choices - all_boys_choices

-- Theorem to prove the number of ways to select the officers
theorem number_of_ways_to_select_officers :
  at_least_one_girl_choices = 100 := by
  sorry

end number_of_ways_to_select_officers_l143_143515


namespace minimum_socks_to_guarantee_20_pairs_l143_143795

-- Definitions and conditions
def red_socks := 120
def green_socks := 100
def blue_socks := 80
def black_socks := 50
def number_of_pairs := 20

-- Statement
theorem minimum_socks_to_guarantee_20_pairs 
  (red_socks green_socks blue_socks black_socks number_of_pairs: ℕ) 
  (h1: red_socks = 120) 
  (h2: green_socks = 100) 
  (h3: blue_socks = 80) 
  (h4: black_socks = 50) 
  (h5: number_of_pairs = 20) : 
  ∃ min_socks, min_socks = 43 := 
by 
  sorry

end minimum_socks_to_guarantee_20_pairs_l143_143795


namespace fraction_BC_AD_l143_143175

-- Defining points and segments
variables (A B C D : Point)
variable (len : Point → Point → ℝ) -- length function

-- Conditions
axiom AB_eq_3BD : len A B = 3 * len B D
axiom AC_eq_7CD : len A C = 7 * len C D
axiom B_mid_AD : 2 * len A B = len A D

-- Theorem: Proving the fraction of BC relative to AD is 2/3
theorem fraction_BC_AD : (len B C) / (len A D) = 2 / 3 :=
sorry

end fraction_BC_AD_l143_143175


namespace max_underwear_pairs_l143_143419

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l143_143419


namespace probability_three_digit_divisible_by_5_with_ones_digit_9_l143_143127

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_three_digit_divisible_by_5_with_ones_digit_9 : 
  ∀ (M : ℕ), is_three_digit M → ones_digit M = 9 → ¬ is_divisible_by_5 M := by
  intros M h1 h2
  sorry

end probability_three_digit_divisible_by_5_with_ones_digit_9_l143_143127


namespace canal_cross_section_area_l143_143088

/-- Definitions of the conditions -/
def top_width : Real := 6
def bottom_width : Real := 4
def depth : Real := 257.25

/-- Proof statement -/
theorem canal_cross_section_area : 
  (1 / 2) * (top_width + bottom_width) * depth = 1286.25 :=
by
  sorry

end canal_cross_section_area_l143_143088


namespace f_2015_l143_143195

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 2) = -f x

axiom f_interval : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2 ^ x

theorem f_2015 : f 2015 = 1 / 2 :=
sorry

end f_2015_l143_143195


namespace mail_cars_in_train_l143_143900

theorem mail_cars_in_train (n : ℕ) (hn : n % 2 = 0) (hfront : 1 ≤ n ∧ n ≤ 20)
  (hclose : ∀ i, 1 ≤ i ∧ i < n → (∃ j, i < j ∧ j ≤ 20))
  (hlast : 4 * n ≤ 20)
  (hconn : ∀ k, (k = 4 ∨ k = 5 ∨ k = 15 ∨ k = 16) → 
                  (∃ j, j = k + 1 ∨ j = k - 1)) :
  ∃ (i : ℕ) (j : ℕ), i = 4 ∧ j = 16 :=
by
  sorry

end mail_cars_in_train_l143_143900


namespace right_triangle_hypotenuse_l143_143654

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 6) (k : b = 8) (pt : a^2 + b^2 = c^2) : c = 10 := by
  sorry

end right_triangle_hypotenuse_l143_143654


namespace knights_max_seated_between_knights_l143_143043

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l143_143043


namespace roger_initial_money_l143_143012

theorem roger_initial_money (spent_on_game : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_spent : ℕ) :
  spent_on_game = 48 →
  cost_per_toy = 3 →
  num_toys = 5 →
  total_money_spent = spent_on_game + num_toys * cost_per_toy →
  total_money_spent = 63 :=
by
  intros h_game h_toy_cost h_num_toys h_total_spent
  rw [h_game, h_toy_cost, h_num_toys] at h_total_spent
  exact h_total_spent

end roger_initial_money_l143_143012


namespace solution_set_of_inequality_l143_143639

theorem solution_set_of_inequality :
  {x : ℝ | 4 * x ^ 2 - 4 * x + 1 ≤ 0} = {1 / 2} :=
by
  sorry

end solution_set_of_inequality_l143_143639


namespace age_difference_l143_143101

theorem age_difference (A B : ℕ) (h1 : B = 37) (h2 : A + 10 = 2 * (B - 10)) : A - B = 7 :=
by
  sorry

end age_difference_l143_143101


namespace marked_percentage_above_cost_l143_143534

theorem marked_percentage_above_cost (CP SP : ℝ) (discount_percentage MP : ℝ) 
  (h1 : CP = 540) 
  (h2 : SP = 457) 
  (h3 : discount_percentage = 26.40901771336554) 
  (h4 : SP = MP * (1 - discount_percentage / 100)) : 
  ((MP - CP) / CP) * 100 = 15 :=
by
  sorry

end marked_percentage_above_cost_l143_143534


namespace meeting_time_final_time_statement_l143_143917

-- Define the speeds and distance as given conditions
def brodie_speed : ℝ := 50
def ryan_speed : ℝ := 40
def initial_distance : ℝ := 120

-- Define what we know about their meeting time and validate it mathematically
theorem meeting_time :
  (initial_distance / (brodie_speed + ryan_speed)) = 4 / 3 := sorry

-- Assert the time in minutes for completeness
noncomputable def time_in_minutes : ℝ := ((4 / 3) * 60)

-- Assert final statement matching the answer in hours and minutes
theorem final_time_statement :
  time_in_minutes = 80 := sorry

end meeting_time_final_time_statement_l143_143917


namespace factorization_of_a_cubed_minus_a_l143_143107

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l143_143107


namespace not_necessarily_heavier_l143_143848

/--
In a zoo, there are 10 elephants. It is known that if any four elephants stand on the left pan and any three on the right pan, the left pan will weigh more. If five elephants stand on the left pan and four on the right pan, the left pan does not necessarily weigh more.
-/
theorem not_necessarily_heavier (E : Fin 10 → ℝ) (H : ∀ (L : Finset (Fin 10)) (R : Finset (Fin 10)), L.card = 4 → R.card = 3 → L ≠ R → L.sum E > R.sum E) :
  ∃ (L' R' : Finset (Fin 10)), L'.card = 5 ∧ R'.card = 4 ∧ L'.sum E ≤ R'.sum E :=
by
  sorry

end not_necessarily_heavier_l143_143848


namespace turtles_remaining_l143_143723

-- Define the initial number of turtles
def initial_turtles : ℕ := 9

-- Define the number of turtles that climbed onto the log
def climbed_turtles : ℕ := 3 * initial_turtles - 2

-- Define the total number of turtles on the log before any jump off
def total_turtles_before_jumping : ℕ := initial_turtles + climbed_turtles

-- Define the number of turtles remaining after half jump off
def remaining_turtles : ℕ := total_turtles_before_jumping / 2

theorem turtles_remaining : remaining_turtles = 17 :=
  by
  -- Placeholder for the proof
  sorry

end turtles_remaining_l143_143723


namespace window_total_width_l143_143631

theorem window_total_width 
  (panes : Nat := 6)
  (ratio_height_width : ℤ := 3)
  (border_width : ℤ := 1)
  (rows : Nat := 2)
  (columns : Nat := 3)
  (pane_width : ℤ := 12) :
  3 * pane_width + 2 * border_width + 2 * border_width = 40 := 
by
  sorry

end window_total_width_l143_143631


namespace find_angle_C_find_max_perimeter_l143_143812

-- Define the first part of the problem
theorem find_angle_C 
  (a b c A B C : ℝ) (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  C = (2 * Real.pi) / 3 :=
sorry

-- Define the second part of the problem
theorem find_max_perimeter 
  (a b A B : ℝ)
  (C : ℝ := (2 * Real.pi) / 3)
  (c : ℝ := Real.sqrt 3)
  (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  (2 * Real.sqrt 3 < a + b + c) ∧ (a + b + c <= 2 + Real.sqrt 3) :=
sorry

end find_angle_C_find_max_perimeter_l143_143812


namespace sin_theta_of_triangle_l143_143009

theorem sin_theta_of_triangle (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ)
  (h_area : area = 30)
  (h_side : side = 10)
  (h_median : median = 9) :
  Real.sin θ = 2 / 3 := by
  sorry

end sin_theta_of_triangle_l143_143009


namespace smallest_resolvable_debt_l143_143613

def pig_value : ℤ := 450
def goat_value : ℤ := 330
def gcd_pig_goat : ℤ := Int.gcd pig_value goat_value

theorem smallest_resolvable_debt :
  ∃ p g : ℤ, gcd_pig_goat * 4 = pig_value * p + goat_value * g := 
by
  sorry

end smallest_resolvable_debt_l143_143613


namespace x_coordinate_of_equidistant_point_l143_143035

theorem x_coordinate_of_equidistant_point (x : ℝ) : 
  ((-3 - x)^2 + (-2 - 0)^2) = ((2 - x)^2 + (-6 - 0)^2) → x = 2.7 :=
by
  sorry

end x_coordinate_of_equidistant_point_l143_143035


namespace number_of_whole_numbers_l143_143688

theorem number_of_whole_numbers (x y : ℝ) (hx : 2 < x ∧ x < 3) (hy : 8 < y ∧ y < 9) : 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_whole_numbers_l143_143688


namespace ball_arrangement_l143_143549

theorem ball_arrangement :
  (Nat.factorial 9) / ((Nat.factorial 2) * (Nat.factorial 3) * (Nat.factorial 4)) = 1260 := 
by
  sorry

end ball_arrangement_l143_143549


namespace remainder_when_x_plus_2uy_div_y_l143_143167

theorem remainder_when_x_plus_2uy_div_y (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) :
  (x + 2 * u * y) % y = v := 
sorry

end remainder_when_x_plus_2uy_div_y_l143_143167


namespace Samantha_purse_value_l143_143112

def cents_per_penny := 1
def cents_per_nickel := 5
def cents_per_dime := 10
def cents_per_quarter := 25

def number_of_pennies := 2
def number_of_nickels := 1
def number_of_dimes := 3
def number_of_quarters := 2

def total_cents := 
  number_of_pennies * cents_per_penny + 
  number_of_nickels * cents_per_nickel + 
  number_of_dimes * cents_per_dime + 
  number_of_quarters * cents_per_quarter

def percent_of_dollar := (total_cents * 100) / 100

theorem Samantha_purse_value : percent_of_dollar = 87 := by
  sorry

end Samantha_purse_value_l143_143112


namespace solve_for_phi_l143_143315

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

theorem solve_for_phi (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π / 2)
    (h_min_diff : |x1 - x2| = π / 6)
    (h_condition : |f x1 - g x2 φ| = 4) :
    φ = π / 3 := 
    sorry

end solve_for_phi_l143_143315


namespace part1_minimum_value_part2_zeros_inequality_l143_143823

noncomputable def f (x a : ℝ) := x * Real.exp x - a * (Real.log x + x)

theorem part1_minimum_value (a : ℝ) :
  (∀ x > 0, f x a > 0) ∨ (∃ x > 0, f x a = a - a * Real.log a) :=
sorry

theorem part2_zeros_inequality (a x₁ x₂ : ℝ) (hx₁ : f x₁ a = 0) (hx₂ : f x₂ a = 0) :
  Real.exp (x₁ + x₂ - 2) > 1 / (x₁ * x₂) :=
sorry

end part1_minimum_value_part2_zeros_inequality_l143_143823


namespace cars_meet_after_5_hours_l143_143143

theorem cars_meet_after_5_hours :
  ∀ (t : ℝ), (40 * t + 60 * t = 500) → t = 5 := 
by
  intro t
  intro h
  sorry

end cars_meet_after_5_hours_l143_143143


namespace sin_780_eq_sqrt3_div_2_l143_143125

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l143_143125


namespace reciprocal_and_fraction_l143_143499

theorem reciprocal_and_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : (2/5) * a = 20) : 
  b = (1/a) ∧ (1/3) * a = (50/3) := 
by 
  sorry

end reciprocal_and_fraction_l143_143499


namespace max_value_of_f_prime_div_f_l143_143865

def f (x : ℝ) : ℝ := sorry

theorem max_value_of_f_prime_div_f (f : ℝ → ℝ) (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x) (h2 : f 0 = 1) :
  ∀ x > 0, (deriv f x / f x) ≤ 2 :=
sorry

end max_value_of_f_prime_div_f_l143_143865


namespace AngiesClassGirlsCount_l143_143674

theorem AngiesClassGirlsCount (n_girls n_boys : ℕ) (total_students : ℕ)
  (h1 : n_girls = 2 * (total_students / 5))
  (h2 : n_boys = 3 * (total_students / 5))
  (h3 : n_girls + n_boys = 20)
  : n_girls = 8 :=
by
  sorry

end AngiesClassGirlsCount_l143_143674


namespace reduced_rectangle_area_l143_143698

theorem reduced_rectangle_area
  (w h : ℕ) (hw : w = 5) (hh : h = 7)
  (new_w : ℕ) (h_reduced_area : new_w = w - 2 ∧ new_w * h = 21)
  (reduced_h : ℕ) (hr : reduced_h = h - 1) :
  (new_w * reduced_h = 18) :=
by
  sorry

end reduced_rectangle_area_l143_143698


namespace sufficient_but_not_necessary_condition_l143_143617

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x = 1 → x^2 - 3 * x + 2 = 0) ∧ ¬(∀ (x : ℝ), x^2 - 3 * x + 2 = 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l143_143617


namespace heights_equal_l143_143282

-- Define base areas and volumes
variables {V : ℝ} {S : ℝ}

-- Assume equal volumes and base areas for the prism and cylinder
variables (h_prism h_cylinder : ℝ) (volume_eq : V = S * h_prism) (base_area_eq : S = S)

-- Define a proof goal
theorem heights_equal 
  (equal_volumes : V = S * h_prism) 
  (equal_base_areas : S = S) : 
  h_prism = h_cylinder :=
sorry

end heights_equal_l143_143282


namespace line_tangent_to_ellipse_l143_143047

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1 → m^2 = 35/9) := 
sorry

end line_tangent_to_ellipse_l143_143047


namespace non_science_majors_percentage_l143_143323

-- Definitions of conditions
def women_percentage (class_size : ℝ) : ℝ := 0.6 * class_size
def men_percentage (class_size : ℝ) : ℝ := 0.4 * class_size

def women_science_majors (class_size : ℝ) : ℝ := 0.2 * women_percentage class_size
def men_science_majors (class_size : ℝ) : ℝ := 0.7 * men_percentage class_size

def total_science_majors (class_size : ℝ) : ℝ := women_science_majors class_size + men_science_majors class_size

-- Theorem to prove the percentage of the class that are non-science majors is 60%
theorem non_science_majors_percentage (class_size : ℝ) : total_science_majors class_size / class_size = 0.4 → (class_size - total_science_majors class_size) / class_size = 0.6 := 
by
  sorry

end non_science_majors_percentage_l143_143323


namespace equivalent_proof_problem_l143_143832

variable {x : ℝ}

theorem equivalent_proof_problem (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5 * x^8 + 2 * x^6 = 1944 * Real.sqrt 7 * x - 2494 :=
sorry

end equivalent_proof_problem_l143_143832


namespace percentage_problem_l143_143345

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 400) : 1.20 * x = 2400 :=
by
  sorry

end percentage_problem_l143_143345


namespace seq_an_identity_l143_143882

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l143_143882


namespace intersection_volume_l143_143590

noncomputable def volume_of_intersection (k : ℝ) : ℝ :=
  ∫ x in -k..k, 4 * (k^2 - x^2)

theorem intersection_volume (k : ℝ) : volume_of_intersection k = 16 * k^3 / 3 :=
  by
  sorry

end intersection_volume_l143_143590


namespace sum_of_three_consecutive_natural_numbers_not_prime_l143_143385

theorem sum_of_three_consecutive_natural_numbers_not_prime (n : ℕ) : 
  ¬ Prime (n + (n+1) + (n+2)) := by
  sorry

end sum_of_three_consecutive_natural_numbers_not_prime_l143_143385


namespace sandy_distance_l143_143789

theorem sandy_distance :
  ∃ d : ℝ, d = 18 * (1000 / 3600) * 99.9920006399488 := sorry

end sandy_distance_l143_143789


namespace people_in_room_l143_143770

open Nat

theorem people_in_room (C : ℕ) (P : ℕ) (h1 : 1 / 4 * C = 6) (h2 : 3 / 4 * C = 2 / 3 * P) : P = 27 := by
  sorry

end people_in_room_l143_143770


namespace number_of_students_l143_143579

theorem number_of_students (y c r n : ℕ) (h1 : y = 730) (h2 : c = 17) (h3 : r = 16) :
  y - r = n * c ↔ n = 42 :=
by
  have h4 : 730 - 16 = 714 := by norm_num
  have h5 : 714 / 17 = 42 := by norm_num
  sorry

end number_of_students_l143_143579


namespace motel_total_rent_l143_143356

theorem motel_total_rent (R₅₀ R₆₀ : ℕ) 
  (h₁ : ∀ x y : ℕ, 50 * x + 60 * y = 50 * (x + 10) + 60 * (y - 10) + 100)
  (h₂ : ∀ x y : ℕ, 25 * (50 * x + 60 * y) = 10000) : 
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by
  sorry

end motel_total_rent_l143_143356


namespace hershey_kisses_to_kitkats_ratio_l143_143008

-- Definitions based on the conditions
def kitkats : ℕ := 5
def nerds : ℕ := 8
def lollipops : ℕ := 11
def baby_ruths : ℕ := 10
def reeses : ℕ := baby_ruths / 2
def candy_total_before : ℕ := kitkats + nerds + lollipops + baby_ruths + reeses
def candy_remaining : ℕ := 49
def lollipops_given : ℕ := 5
def total_candy_before : ℕ := candy_remaining + lollipops_given
def hershey_kisses : ℕ := total_candy_before - candy_total_before

-- Theorem to prove the desired ratio
theorem hershey_kisses_to_kitkats_ratio : hershey_kisses / kitkats = 3 := by
  sorry

end hershey_kisses_to_kitkats_ratio_l143_143008


namespace constant_in_price_equation_l143_143316

theorem constant_in_price_equation (x y: ℕ) (h: y = 70 * x) : ∃ c, ∀ (x: ℕ), y = c * x ∧ c = 70 :=
  sorry

end constant_in_price_equation_l143_143316


namespace inequality_holds_l143_143116

theorem inequality_holds (k n : ℕ) (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n :=
by
  sorry

end inequality_holds_l143_143116


namespace time_to_ascend_non_working_escalator_l143_143846

-- Define the variables as given in the conditions
def V := 1 / 60 -- Speed of the moving escalator in units per minute
def U := (1 / 24) - (1 / 60) -- Speed of Gavrila running relative to the escalator

-- Theorem stating that the time to ascend a non-working escalator is 40 seconds
theorem time_to_ascend_non_working_escalator : 
  (1 : ℚ) = U * (40 / 60) := 
by sorry

end time_to_ascend_non_working_escalator_l143_143846


namespace no_five_distinct_natural_numbers_feasible_l143_143290

theorem no_five_distinct_natural_numbers_feasible :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
  sorry

end no_five_distinct_natural_numbers_feasible_l143_143290


namespace least_possible_value_of_s_l143_143297

theorem least_possible_value_of_s (a b : ℤ) 
(h : a^3 + b^3 - 60 * a * b * (a + b) ≥ 2012) : 
∃ a b, a^3 + b^3 - 60 * a * b * (a + b) = 2015 :=
by sorry

end least_possible_value_of_s_l143_143297


namespace grid_divisible_by_rectangles_l143_143960

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end grid_divisible_by_rectangles_l143_143960


namespace brooke_added_balloons_l143_143404

-- Definitions stemming from the conditions
def initial_balloons_brooke : Nat := 12
def added_balloons_brooke (x : Nat) : Nat := x
def initial_balloons_tracy : Nat := 6
def added_balloons_tracy : Nat := 24
def total_balloons_tracy : Nat := initial_balloons_tracy + added_balloons_tracy
def final_balloons_tracy : Nat := total_balloons_tracy / 2
def total_balloons (x : Nat) : Nat := initial_balloons_brooke + added_balloons_brooke x + final_balloons_tracy

-- Mathematical proof problem
theorem brooke_added_balloons (x : Nat) :
  total_balloons x = 35 → x = 8 := by
  sorry

end brooke_added_balloons_l143_143404


namespace length_of_AB_l143_143185

def parabola_eq (y : ℝ) : Prop := y^2 = 8 * y

def directrix_x : ℝ := 2

def dist_to_y_axis (E : ℝ × ℝ) : ℝ := E.1

theorem length_of_AB (A B F E : ℝ × ℝ)
  (p : parabola_eq A.2) (q : parabola_eq B.2) 
  (F_focus : F.1 = 2 ∧ F.2 = 0) 
  (midpoint_E : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (E_distance_from_y_axis : dist_to_y_axis E = 3) : 
  (abs (A.1 - B.1) + abs (A.2 - B.2)) = 10 := 
sorry

end length_of_AB_l143_143185


namespace find_a_l143_143243

theorem find_a (x y : ℝ) (a : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : a * x + 2 * y = 1) : a = -1 := by
  sorry

end find_a_l143_143243


namespace ratio_of_coconut_flavored_red_jelly_beans_l143_143662

theorem ratio_of_coconut_flavored_red_jelly_beans :
  ∀ (total_jelly_beans jelly_beans_coconut_flavored : ℕ)
    (three_fourths_red : total_jelly_beans > 0 ∧ (3/4 : ℝ) * total_jelly_beans = 3 * (total_jelly_beans / 4))
    (h1 : jelly_beans_coconut_flavored = 750)
    (h2 : total_jelly_beans = 4000),
  (250 : ℝ)/(3000 : ℝ) = 1/4 :=
by
  intros total_jelly_beans jelly_beans_coconut_flavored three_fourths_red h1 h2
  sorry

end ratio_of_coconut_flavored_red_jelly_beans_l143_143662


namespace probability_of_draw_l143_143077

-- Let P be the probability of the game ending in a draw.
-- Let PA be the probability of Player A winning.

def PA_not_losing := 0.8
def PB_not_losing := 0.7

theorem probability_of_draw : ¬ (1 - PA_not_losing + PB_not_losing ≠ 1.5) → PA_not_losing + (1 - PB_not_losing) = 1.5 → PB_not_losing + 0.5 = 1 := by
  intros
  sorry

end probability_of_draw_l143_143077


namespace fish_added_l143_143064

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l143_143064


namespace clock_angle_at_3_15_l143_143992

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end clock_angle_at_3_15_l143_143992


namespace f_neg4_plus_f_0_range_of_a_l143_143467

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else if x < 0 then -log (-x) / log 2 else 0

/- Prove that f(-4) + f(0) = -2 given the function properties -/
theorem f_neg4_plus_f_0 : f (-4) + f 0 = -2 :=
sorry

/- Prove the range of a such that f(a) > f(-a) is a > 1 or -1 < a < 0 given the function properties -/
theorem range_of_a (a : ℝ) : f a > f (-a) ↔ a > 1 ∨ (-1 < a ∧ a < 0) :=
sorry

end f_neg4_plus_f_0_range_of_a_l143_143467


namespace g_one_minus_g_four_l143_143606

theorem g_one_minus_g_four (g : ℝ → ℝ)
  (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ x : ℝ, g (x + 1) - g x = 5) :
  g 1 - g 4 = -15 :=
sorry

end g_one_minus_g_four_l143_143606


namespace utensils_in_each_pack_l143_143650

/-- Prove that given John needs to buy 5 packs to get 50 spoons
    and each pack contains an equal number of knives, forks, and spoons,
    the total number of utensils in each pack is 30. -/
theorem utensils_in_each_pack
  (packs : ℕ)
  (total_spoons : ℕ)
  (equal_parts : ∀ p : ℕ, p = total_spoons / packs)
  (knives forks spoons : ℕ)
  (equal_utensils : ∀ u : ℕ, u = spoons)
  (knives_forks : knives = forks)
  (knives_spoons : knives = spoons)
  (packs_needed : packs = 5)
  (total_utensils_needed : total_spoons = 50) :
  knives + forks + spoons = 30 := by
  sorry

end utensils_in_each_pack_l143_143650


namespace Jeanine_more_pencils_than_Clare_l143_143132

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l143_143132


namespace find_abc_l143_143777

theorem find_abc : ∃ (a b c : ℝ), a + b + c = 1 ∧ 4 * a + 2 * b + c = 5 ∧ 9 * a + 3 * b + c = 13 ∧ a - b + c = 5 := by
  sorry

end find_abc_l143_143777


namespace inequality_proof_l143_143772

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  1/a + 1/b + 1/c ≥ 2/(a + b) + 2/(b + c) + 2/(c + a) ∧ 2/(a + b) + 2/(b + c) + 2/(c + a) ≥ 9/(a + b + c) :=
sorry

end inequality_proof_l143_143772


namespace minimum_value_of_fraction_plus_variable_l143_143048

theorem minimum_value_of_fraction_plus_variable (a : ℝ) (h : a > 1) : ∃ m, (∀ b, b > 1 → (4 / (b - 1) + b) ≥ m) ∧ m = 5 :=
by
  use 5
  sorry

end minimum_value_of_fraction_plus_variable_l143_143048


namespace sum_of_common_ratios_l143_143324

theorem sum_of_common_ratios (k p r : ℝ) (h : k ≠ 0) (h1 : k * p ≠ k * r)
  (h2 : k * p ^ 2 - k * r ^ 2 = 3 * (k * p - k * r)) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l143_143324


namespace Pablo_puzzle_completion_l143_143817

theorem Pablo_puzzle_completion :
  let pieces_per_hour := 100
  let puzzles_400 := 15
  let pieces_per_puzzle_400 := 400
  let puzzles_700 := 10
  let pieces_per_puzzle_700 := 700
  let daily_work_hours := 6
  let daily_work_400_hours := 4
  let daily_work_700_hours := 2
  let break_every_hours := 2
  let break_time := 30 / 60   -- 30 minutes break in hours

  let total_pieces_400 := puzzles_400 * pieces_per_puzzle_400
  let total_pieces_700 := puzzles_700 * pieces_per_puzzle_700
  let total_pieces := total_pieces_400 + total_pieces_700

  let effective_daily_hours := daily_work_hours - (daily_work_hours / break_every_hours * break_time)
  let pieces_400_per_day := daily_work_400_hours * pieces_per_hour
  let pieces_700_per_day := (effective_daily_hours - daily_work_400_hours) * pieces_per_hour
  let total_pieces_per_day := pieces_400_per_day + pieces_700_per_day
  
  total_pieces / total_pieces_per_day = 26 := by
sorry

end Pablo_puzzle_completion_l143_143817


namespace add_to_fraction_eq_l143_143464

theorem add_to_fraction_eq (n : ℤ) : (4 + n : ℤ) / (7 + n) = (2 : ℤ) / 3 → n = 2 := 
by {
  sorry
}

end add_to_fraction_eq_l143_143464


namespace seating_chart_example_l143_143333

def seating_chart_representation (a b : ℕ) : String :=
  s!"{a} columns {b} rows"

theorem seating_chart_example :
  seating_chart_representation 4 3 = "4 columns 3 rows" :=
by
  sorry

end seating_chart_example_l143_143333


namespace total_cost_of_stickers_l143_143809

-- Definitions based on given conditions
def initial_funds_per_person := 9
def cost_of_deck_of_cards := 10
def Dora_packs_of_stickers := 2

-- Calculate the total amount of money collectively after buying the deck of cards
def remaining_funds := 2 * initial_funds_per_person - cost_of_deck_of_cards

-- Calculate the total packs of stickers if split evenly
def total_packs_of_stickers := 2 * Dora_packs_of_stickers

-- Prove the total cost of the boxes of stickers
theorem total_cost_of_stickers : remaining_funds = 8 := by
  -- Given initial funds per person, cost of deck of cards, and packs of stickers for Dora, the theorem should hold.
  sorry

end total_cost_of_stickers_l143_143809


namespace calculate_expression_l143_143873

theorem calculate_expression : (3.75 - 1.267 + 0.48 = 2.963) :=
by
  sorry

end calculate_expression_l143_143873


namespace cattle_area_correct_l143_143700

-- Definitions based on the problem conditions
def length_km := 3.6
def width_km := 2.5 * length_km
def total_area_km2 := length_km * width_km
def cattle_area_km2 := total_area_km2 / 2

-- Theorem statement
theorem cattle_area_correct : cattle_area_km2 = 16.2 := by
  sorry

end cattle_area_correct_l143_143700
