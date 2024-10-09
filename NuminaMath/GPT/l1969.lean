import Mathlib

namespace folded_strip_fit_l1969_196975

open Classical

noncomputable def canFitAfterFolding (r : ℝ) (strip : Set (ℝ × ℝ)) (folded_strip : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ folded_strip → (p.1^2 + p.2^2 ≤ r^2)

theorem folded_strip_fit {r : ℝ} {strip folded_strip : Set (ℝ × ℝ)} :
  (∀ p : ℝ × ℝ, p ∈ strip → (p.1^2 + p.2^2 ≤ r^2)) →
  (∀ q : ℝ × ℝ, q ∈ folded_strip → (∃ p : ℝ × ℝ, p ∈ strip ∧ q = p)) →
  canFitAfterFolding r strip folded_strip :=
by
  intros hs hf
  sorry

end folded_strip_fit_l1969_196975


namespace inequality_always_holds_l1969_196988

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
by 
  sorry

end inequality_always_holds_l1969_196988


namespace compute_pqr_l1969_196936

theorem compute_pqr
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_eq : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
  sorry

end compute_pqr_l1969_196936


namespace relationship_between_roses_and_total_flowers_l1969_196946

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers_l1969_196946


namespace initial_bottles_count_l1969_196976

theorem initial_bottles_count
  (players : ℕ)
  (bottles_per_player_first_break : ℕ)
  (bottles_per_player_end_game : ℕ)
  (remaining_bottles : ℕ)
  (total_bottles_taken_first_break : bottles_per_player_first_break * players = 22)
  (total_bottles_taken_end_game : bottles_per_player_end_game * players = 11)
  (total_remaining_bottles : remaining_bottles = 15) :
  players * bottles_per_player_first_break + players * bottles_per_player_end_game + remaining_bottles = 48 :=
by 
  -- skipping the proof
  sorry

end initial_bottles_count_l1969_196976


namespace chloe_sold_strawberries_l1969_196970

noncomputable section

def cost_per_dozen : ℕ := 50
def sale_price_per_half_dozen : ℕ := 30
def total_profit : ℕ := 500
def profit_per_half_dozen := sale_price_per_half_dozen - (cost_per_dozen / 2)
def half_dozens_sold := total_profit / profit_per_half_dozen

theorem chloe_sold_strawberries : half_dozens_sold / 2 = 50 :=
by
  -- proof would go here
  sorry

end chloe_sold_strawberries_l1969_196970


namespace store_paid_price_l1969_196918

-- Definition of the conditions
def selling_price : ℕ := 34
def difference_price : ℕ := 8

-- Statement that needs to be proven.
theorem store_paid_price : (selling_price - difference_price) = 26 :=
by
  sorry

end store_paid_price_l1969_196918


namespace cost_per_person_is_correct_l1969_196905

-- Define the given conditions
def fee_per_30_minutes : ℕ := 4000
def bikes : ℕ := 4
def hours : ℕ := 3
def people : ℕ := 6

-- Calculate the correct answer based on the given conditions
noncomputable def cost_per_person : ℕ :=
  let fee_per_hour := 2 * fee_per_30_minutes
  let fee_per_3_hours := hours * fee_per_hour
  let total_cost := bikes * fee_per_3_hours
  total_cost / people

-- The theorem to be proved
theorem cost_per_person_is_correct : cost_per_person = 16000 := sorry

end cost_per_person_is_correct_l1969_196905


namespace cupboard_cost_price_l1969_196950

theorem cupboard_cost_price (C SP NSP : ℝ) (h1 : SP = 0.84 * C) (h2 : NSP = 1.16 * C) (h3 : NSP = SP + 1200) : C = 3750 :=
by
  sorry

end cupboard_cost_price_l1969_196950


namespace water_in_pool_after_35_days_l1969_196944

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days_l1969_196944


namespace dave_deleted_apps_l1969_196963

def apps_initial : ℕ := 23
def apps_left : ℕ := 5
def apps_deleted : ℕ := apps_initial - apps_left

theorem dave_deleted_apps : apps_deleted = 18 := 
by
  sorry

end dave_deleted_apps_l1969_196963


namespace cd_player_percentage_l1969_196984

-- Define the percentage variables
def powerWindowsAndAntiLock : ℝ := 0.10
def antiLockAndCdPlayer : ℝ := 0.15
def powerWindowsAndCdPlayer : ℝ := 0.22
def cdPlayerAlone : ℝ := 0.38

-- Define the problem statement
theorem cd_player_percentage : 
  powerWindowsAndAntiLock = 0.10 → 
  antiLockAndCdPlayer = 0.15 → 
  powerWindowsAndCdPlayer = 0.22 → 
  cdPlayerAlone = 0.38 → 
  (antiLockAndCdPlayer + powerWindowsAndCdPlayer + cdPlayerAlone) = 0.75 :=
by
  intros
  sorry

end cd_player_percentage_l1969_196984


namespace sum_of_coordinates_A_l1969_196903

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l1969_196903


namespace rectangle_width_eq_six_l1969_196934

theorem rectangle_width_eq_six (w : ℝ) :
  ∃ w, (3 * w = 25 - 7) ↔ w = 6 :=
by
  -- Given the conditions as stated:
  -- Length of the rectangle: 3 inches
  -- Width of the square: 5 inches
  -- Difference in area between the square and the rectangle: 7 square inches
  -- We can show that the width of the rectangle is 6 inches.
  sorry

end rectangle_width_eq_six_l1969_196934


namespace find_number_l1969_196983

-- Define the problem conditions
def problem_condition (x : ℝ) : Prop := 2 * x - x / 2 = 45

-- Main theorem statement
theorem find_number : ∃ (x : ℝ), problem_condition x ∧ x = 30 :=
by
  existsi 30
  -- Include the problem condition and the solution check
  unfold problem_condition
  -- We are skipping the proof using sorry to just provide the statement
  sorry

end find_number_l1969_196983


namespace evaluate_expression_l1969_196996

theorem evaluate_expression :
  (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 :=
by
  sorry

end evaluate_expression_l1969_196996


namespace range_a_l1969_196928

theorem range_a (x a : ℝ) (h1 : x^2 - 8 * x - 33 > 0) (h2 : |x - 1| > a) (h3 : a > 0) :
  0 < a ∧ a ≤ 4 :=
by
  sorry

end range_a_l1969_196928


namespace increasing_function_range_b_l1969_196939

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3 / 2) * x + b - 1 else -x^2 + (2 - b) * x

theorem increasing_function_range_b :
  (∀ x y, x < y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2 ) := 
by
  sorry

end increasing_function_range_b_l1969_196939


namespace value_divided_by_is_three_l1969_196929

theorem value_divided_by_is_three (x : ℝ) (h : 72 / x = 24) : x = 3 := 
by
  sorry

end value_divided_by_is_three_l1969_196929


namespace solve_system_of_equations_l1969_196938

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 2 →
  x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) :=
by
  intros h1 h2
  sorry

end solve_system_of_equations_l1969_196938


namespace marshmallow_challenge_l1969_196910

noncomputable def haley := 8
noncomputable def michael := 3 * haley
noncomputable def brandon := (1 / 2) * michael
noncomputable def sofia := 2 * (haley + brandon)
noncomputable def total := haley + michael + brandon + sofia

theorem marshmallow_challenge : total = 84 :=
by
  sorry

end marshmallow_challenge_l1969_196910


namespace cubic_root_equality_l1969_196914

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end cubic_root_equality_l1969_196914


namespace millet_exceeds_half_l1969_196981

noncomputable def seeds_millet_day (n : ℕ) : ℝ :=
  0.2 * (1 - 0.7 ^ n) / (1 - 0.7) + 0.2 * 0.7 ^ n

noncomputable def seeds_other_day (n : ℕ) : ℝ :=
  0.3 * (1 - 0.1 ^ n) / (1 - 0.1) + 0.3 * 0.1 ^ n

noncomputable def prop_millet (n : ℕ) : ℝ :=
  seeds_millet_day n / (seeds_millet_day n + seeds_other_day n)

theorem millet_exceeds_half : ∃ n : ℕ, prop_millet n > 0.5 ∧ n = 3 :=
by sorry

end millet_exceeds_half_l1969_196981


namespace total_cost_of_hats_l1969_196954

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l1969_196954


namespace fraction_paint_remaining_l1969_196942

theorem fraction_paint_remaining 
  (original_paint : ℝ)
  (h_original : original_paint = 2) 
  (used_first_day : ℝ)
  (h_used_first_day : used_first_day = (1 / 4) * original_paint) 
  (remaining_after_first : ℝ)
  (h_remaining_first : remaining_after_first = original_paint - used_first_day) 
  (used_second_day : ℝ)
  (h_used_second_day : used_second_day = (1 / 3) * remaining_after_first) 
  (remaining_after_second : ℝ)
  (h_remaining_second : remaining_after_second = remaining_after_first - used_second_day) : 
  remaining_after_second / original_paint = 1 / 2 :=
by
  -- Proof goes here.
  sorry

end fraction_paint_remaining_l1969_196942


namespace marbles_in_jar_l1969_196960

theorem marbles_in_jar (M : ℕ) (h1 : M / 24 = 24 * 26 / 26) (h2 : M / 26 + 1 = M / 24) : M = 312 := by
  sorry

end marbles_in_jar_l1969_196960


namespace value_of_y_l1969_196979

theorem value_of_y (y : ℕ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
-- Since we are only required to state the theorem, we leave the proof out for now.
sorry

end value_of_y_l1969_196979


namespace union_A_B_l1969_196958

-- Definitions for the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- The statement to be proven
theorem union_A_B :
  A ∪ B = {x | (-1 < x ∧ x ≤ 3) ∨ x = 4} :=
sorry

end union_A_B_l1969_196958


namespace equal_sets_d_l1969_196931

theorem equal_sets_d : 
  (let M := {x | x^2 - 3*x + 2 = 0}
   let N := {1, 2}
   M = N) :=
by 
  sorry

end equal_sets_d_l1969_196931


namespace ratio_of_original_to_reversed_l1969_196921

def original_number : ℕ := 21
def reversed_number : ℕ := 12

theorem ratio_of_original_to_reversed : 
  (original_number : ℚ) / (reversed_number : ℚ) = 7 / 4 := by
  sorry

end ratio_of_original_to_reversed_l1969_196921


namespace reciprocal_of_neg3_l1969_196967

theorem reciprocal_of_neg3 : 1 / (-3 : ℝ) = - (1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l1969_196967


namespace find_a_n_l1969_196943

theorem find_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = 3^n + 2) :
  ∀ n, a n = if n = 1 then 5 else 2 * 3^(n - 1) := by
  sorry

end find_a_n_l1969_196943


namespace quadrilateral_equality_l1969_196935

-- Variables definitions for points and necessary properties
variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assumptions based on given conditions
variables (AB : ℝ) (AD : ℝ) (BC : ℝ) (DC : ℝ) (beta : ℝ)
variables {angleB : ℝ} {angleD : ℝ}

-- Given conditions
axiom AB_eq_AD : AB = AD
axiom angleB_eq_angleD : angleB = angleD

-- The statement to be proven
theorem quadrilateral_equality (h1 : AB = AD) (h2 : angleB = angleD) : BC = DC :=
by
  sorry

end quadrilateral_equality_l1969_196935


namespace solve_system_l1969_196916

theorem solve_system (x y z : ℝ) (h1 : (x + 1) * y * z = 12) 
                               (h2 : (y + 1) * z * x = 4) 
                               (h3 : (z + 1) * x * y = 4) : 
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
sorry

end solve_system_l1969_196916


namespace lines_perpendicular_to_same_plane_are_parallel_l1969_196989

variables {Point Line Plane : Type*}
variables [MetricSpace Point] [LinearOrder Line]

def line_parallel_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def line_perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def lines_parallel (a b : Line) : Prop := sorry -- Define the formal condition

theorem lines_perpendicular_to_same_plane_are_parallel 
  (a b : Line) (M : Plane) 
  (h₁ : line_perpendicular_to_plane a M) 
  (h₂ : line_perpendicular_to_plane b M) : 
  lines_parallel a b :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l1969_196989


namespace contrapositive_abc_l1969_196906

theorem contrapositive_abc (a b c : ℝ) : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (abc ≠ 0) := 
sorry

end contrapositive_abc_l1969_196906


namespace password_lock_probability_l1969_196985

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l1969_196985


namespace pure_imaginary_solutions_l1969_196977

theorem pure_imaginary_solutions:
  ∀ (x : ℂ), (x.im ≠ 0 ∧ x.re = 0) → (x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0)
         → (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by
  sorry

end pure_imaginary_solutions_l1969_196977


namespace max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l1969_196991

noncomputable def point_in_circle : Prop :=
  let P := (-Real.sqrt 3, 2)
  ∃ (x y : ℝ), x^2 + y^2 = 12 ∧ x = -Real.sqrt 3 ∧ y = 2

theorem max_min_AB_length (α : ℝ) (h1 : -Real.sqrt 3 ≤ α ∧ α ≤ Real.pi / 2) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let R := Real.sqrt 12
  ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12 ∧ (P.1, P.2) = (-Real.sqrt 3, 2)) →
    ((max (dist A B) (dist P P)) = 4 * Real.sqrt 3 ∧ (min (dist A B) (dist P P)) = 2 * Real.sqrt 5) :=
sorry

theorem chord_length_at_angle (α : ℝ) (h2 : α = 120 / 180 * Real.pi) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let A := (Real.sqrt 12, 0)
  let B := (-Real.sqrt 12, 0)
  let AB := (dist A B)
  AB = Real.sqrt 47 :=
sorry

theorem trajectory_midpoint_chord :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  ∀ (M : ℝ × ℝ), (∀ k : ℝ, P.2 - 2 = k * (P.1 + Real.sqrt 3) ∧ M.2 = - 1 / k * M.1) → 
  (M.1^2 + M.2^2 + Real.sqrt 3 * M.1 + 2 * M.2 = 0) :=
sorry

end max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l1969_196991


namespace mass_percentage_Ba_in_BaI2_l1969_196997

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 : 
  (molar_mass_Ba / molar_mass_BaI2) * 100 = 35.11 := 
  by 
    -- implementing the proof here would demonstrate that (137.33 / 391.13) * 100 = 35.11
    sorry

end mass_percentage_Ba_in_BaI2_l1969_196997


namespace acute_triangle_sin_sum_gt_2_l1969_196941

open Real

theorem acute_triangle_sin_sum_gt_2 (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (h_sum : α + β + γ = π) :
  sin α + sin β + sin γ > 2 :=
sorry

end acute_triangle_sin_sum_gt_2_l1969_196941


namespace car_trip_time_difference_l1969_196900

theorem car_trip_time_difference
  (average_speed : ℝ)
  (distance1 distance2 : ℝ)
  (speed_60_mph : average_speed = 60)
  (dist1_540 : distance1 = 540)
  (dist2_510 : distance2 = 510) :
  ((distance1 - distance2) / average_speed) * 60 = 30 := by
  sorry

end car_trip_time_difference_l1969_196900


namespace prasanna_speed_l1969_196901

variable (L_speed P_speed time apart : ℝ)
variable (h1 : L_speed = 40)
variable (h2 : time = 1)
variable (h3 : apart = 78)

theorem prasanna_speed :
  P_speed = apart - (L_speed * time) / time := 
by
  rw [h1, h2, h3]
  simp
  sorry

end prasanna_speed_l1969_196901


namespace find_x_l1969_196955

theorem find_x :
  (12^3 * 6^3) / x = 864 → x = 432 :=
by
  sorry

end find_x_l1969_196955


namespace area_relationship_l1969_196904

theorem area_relationship (x β : ℝ) (hβ : 0.60 * x^2 = β) : α = (4 / 3) * β :=
by
  -- conditions and goal are stated
  let α := 0.80 * x^2
  sorry

end area_relationship_l1969_196904


namespace simplify_expression_correct_l1969_196913

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end simplify_expression_correct_l1969_196913


namespace cattle_train_speed_is_56_l1969_196973

variable (v : ℝ)

def cattle_train_speed :=
  let cattle_distance_until_diesel_starts := 6 * v
  let diesel_speed := v - 33
  let diesel_distance := 12 * diesel_speed
  let cattle_additional_distance := 12 * v
  let total_distance := cattle_distance_until_diesel_starts + diesel_distance + cattle_additional_distance
  total_distance = 1284

theorem cattle_train_speed_is_56 (h : cattle_train_speed v) : v = 56 :=
  sorry

end cattle_train_speed_is_56_l1969_196973


namespace triangle_is_right_angled_l1969_196993

noncomputable def median (a b c : ℝ) : ℝ := (1 / 2) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2))

theorem triangle_is_right_angled (a b c : ℝ) (ha : median a b c = 5) (hb : median b c a = Real.sqrt 52) (hc : median c a b = Real.sqrt 73) :
  a^2 = b^2 + c^2 :=
sorry

end triangle_is_right_angled_l1969_196993


namespace fraction_cal_handled_l1969_196953

theorem fraction_cal_handled (Mabel Anthony Cal Jade : ℕ) 
  (h_Mabel : Mabel = 90)
  (h_Anthony : Anthony = Mabel + Mabel / 10)
  (h_Jade : Jade = 80)
  (h_Cal : Cal = Jade - 14) :
  (Cal : ℚ) / (Anthony : ℚ) = 2 / 3 :=
by
  sorry

end fraction_cal_handled_l1969_196953


namespace car_speed_conversion_l1969_196923

theorem car_speed_conversion (V_kmph : ℕ) (h : V_kmph = 36) : (V_kmph * 1000 / 3600) = 10 := by
  sorry

end car_speed_conversion_l1969_196923


namespace coordinates_C_on_segment_AB_l1969_196951

theorem coordinates_C_on_segment_AB :
  ∃ C : (ℝ × ℝ), 
  (C.1 = 2 ∧ C.2 = 6) ∧
  ∃ A B : (ℝ × ℝ), 
  (A = (-1, 0)) ∧ 
  (B = (3, 8)) ∧ 
  (∃ k : ℝ, (k = 3) ∧ dist (C) (A) = k * dist (C) (B)) :=
by
  sorry

end coordinates_C_on_segment_AB_l1969_196951


namespace bacteria_growth_rate_l1969_196961

theorem bacteria_growth_rate
  (r : ℝ) 
  (h1 : ∃ B D : ℝ, B * r^30 = D) 
  (h2 : ∃ B D : ℝ, B * r^25 = D / 32) :
  r = 2 := 
by 
  sorry

end bacteria_growth_rate_l1969_196961


namespace value_of_expression_l1969_196925

theorem value_of_expression : 
  ∀ (x y : ℤ), x = -5 → y = -10 → (y - x) * (y + x) = 75 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end value_of_expression_l1969_196925


namespace prime_between_40_50_largest_prime_lt_100_l1969_196907

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def between (n m k : ℕ) : Prop := n < k ∧ k < m

theorem prime_between_40_50 :
  {x : ℕ | between 40 50 x ∧ isPrime x} = {41, 43, 47} :=
sorry

theorem largest_prime_lt_100 :
  ∃ p : ℕ, isPrime p ∧ p < 100 ∧ ∀ q : ℕ, isPrime q ∧ q < 100 → q ≤ p :=
sorry

end prime_between_40_50_largest_prime_lt_100_l1969_196907


namespace tan_half_angle_product_l1969_196920

theorem tan_half_angle_product (a b : ℝ) (h : 3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (x : ℝ), x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := 
sorry

end tan_half_angle_product_l1969_196920


namespace sum_six_consecutive_integers_l1969_196987

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l1969_196987


namespace parabola_equation_trajectory_midpoint_l1969_196924

-- Given data and conditions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola_x3 (p : ℝ) : Prop := ∃ y, parabola p 3 y
def distance_point_to_line (x d : ℝ) : Prop := x + d = 5

-- Prove that given these conditions, the parabola equation is y^2 = 8x
theorem parabola_equation (p : ℝ) (h1 : point_on_parabola_x3 p) (h2 : distance_point_to_line (3 + p / 2) 2) : p = 4 :=
sorry

-- Prove the equation of the trajectory for the midpoint of the line segment FP
def point_on_parabola (p x y : ℝ) : Prop := y^2 = 8 * x
theorem trajectory_midpoint (p x y : ℝ) (h1 : parabola 4 x y) : y^2 = 4 * (x - 1) :=
sorry

end parabola_equation_trajectory_midpoint_l1969_196924


namespace tangent_line_equation_range_of_k_l1969_196966

noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x

-- Part (I): Tangent line equation
theorem tangent_line_equation :
  let f (x : ℝ) := x^2 - x * Real.log x
  let p := (1 : ℝ)
  let y := f p
  (∀ x, y = x) :=
sorry

-- Part (II): Range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → (k / x + x / 2 - f x / x < 0)) → k ≤ 1 / 2 :=
sorry

end tangent_line_equation_range_of_k_l1969_196966


namespace points_on_ellipse_l1969_196982

-- Definitions of the conditions
def ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Target set of points
def target_set (x y : ℝ) : Prop :=
  x^2 + y^2 < 5 ∧ |y| > 1

-- Main theorem to prove
theorem points_on_ellipse (a b x y : ℝ) (h₁ : passes_through_point a b) (h₂ : |y| > 1) :
  ellipse a b x y → target_set x y :=
sorry

end points_on_ellipse_l1969_196982


namespace total_investment_with_interest_l1969_196933

def principal : ℝ := 1000
def part3Percent : ℝ := 199.99999999999983
def rate3Percent : ℝ := 0.03
def rate5Percent : ℝ := 0.05
def interest3Percent : ℝ := part3Percent * rate3Percent
def part5Percent : ℝ := principal - part3Percent
def interest5Percent : ℝ := part5Percent * rate5Percent
def totalWithInterest : ℝ := principal + interest3Percent + interest5Percent

theorem total_investment_with_interest :
  totalWithInterest = 1046.00 :=
by
  unfold totalWithInterest interest5Percent part5Percent interest3Percent
  sorry

end total_investment_with_interest_l1969_196933


namespace trade_in_value_of_old_phone_l1969_196952

-- Define the given conditions
def cost_of_iphone : ℕ := 800
def earnings_per_week : ℕ := 80
def weeks_worked : ℕ := 7

-- Define the total earnings from babysitting
def total_earnings : ℕ := earnings_per_week * weeks_worked

-- Define the final proof statement
theorem trade_in_value_of_old_phone : cost_of_iphone - total_earnings = 240 :=
by
  unfold cost_of_iphone
  unfold total_earnings
  -- Substitute in the values
  have h1 : 800 - (80 * 7) = 240 := sorry
  exact h1

end trade_in_value_of_old_phone_l1969_196952


namespace work_completion_days_l1969_196965

-- Definitions based on the conditions
def A_work_days : ℕ := 20
def B_work_days : ℕ := 30
def C_work_days : ℕ := 10  -- Twice as fast as A, and A can do it in 20 days, hence 10 days.
def together_work_days : ℕ := 12
def B_C_half_day_rate : ℚ := (1 / B_work_days) / 2 + (1 / C_work_days) / 2  -- rate per half day for both B and C
def A_full_day_rate : ℚ := 1 / A_work_days  -- rate per full day for A

-- Converting to rate per day when B and C work only half day daily
def combined_rate_per_day_with_BC_half : ℚ := A_full_day_rate + B_C_half_day_rate

-- The main theorem to prove
theorem work_completion_days 
  (A_work_days B_work_days C_work_days together_work_days : ℕ)
  (C_work_days_def : C_work_days = A_work_days / 2) 
  (total_days_def : 1 / combined_rate_per_day_with_BC_half = 60 / 7) :
  (1 / combined_rate_per_day_with_BC_half) = 60 / 7 :=
sorry

end work_completion_days_l1969_196965


namespace solve_for_d_l1969_196911

theorem solve_for_d (n k c d : ℝ) (h₁ : n = 2 * k * c * d / (c + d)) (h₂ : 2 * k * c ≠ n) :
  d = n * c / (2 * k * c - n) :=
by
  sorry

end solve_for_d_l1969_196911


namespace total_seashells_l1969_196917

-- Definitions of the initial number of seashells and the number found
def initial_seashells : Nat := 19
def found_seashells : Nat := 6

-- Theorem stating the total number of seashells in the collection
theorem total_seashells : initial_seashells + found_seashells = 25 := by
  sorry

end total_seashells_l1969_196917


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l1969_196909

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l1969_196909


namespace necessary_and_sufficient_condition_l1969_196919

variable (m n : ℕ)
def positive_integers (m n : ℕ) := m > 0 ∧ n > 0
def at_least_one_is_1 (m n : ℕ) : Prop := m = 1 ∨ n = 1
def sum_gt_product (m n : ℕ) : Prop := m + n > m * n

theorem necessary_and_sufficient_condition (h : positive_integers m n) : 
  sum_gt_product m n ↔ at_least_one_is_1 m n :=
by sorry

end necessary_and_sufficient_condition_l1969_196919


namespace find_d_q_l1969_196968

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def b_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  b1 * q^(n - 1)

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) / 2) * d

-- Sum of the first n terms of a geometric sequence
noncomputable def T_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  if q = 1 then n * b1
  else b1 * (1 - q^n) / (1 - q)

theorem find_d_q (a1 b1 d q : ℕ) (h1 : ∀ n : ℕ, n > 0 →
  n^2 * (T_n b1 q n + 1) = 2^n * S_n a1 d n) : d = 2 ∧ q = 2 :=
by
  sorry

end find_d_q_l1969_196968


namespace y_relationship_l1969_196930

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1: y1 = -4 / x1) (h2: y2 = -4 / x2) (h3: y3 = -4 / x3)
  (h4: x1 < 0) (h5: 0 < x2) (h6: x2 < x3) :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end y_relationship_l1969_196930


namespace floor_of_neg_five_thirds_l1969_196926

theorem floor_of_neg_five_thirds : Int.floor (-5/3 : ℝ) = -2 := 
by 
  sorry

end floor_of_neg_five_thirds_l1969_196926


namespace cot_trig_identity_l1969_196908

noncomputable def cot (x : Real) : Real :=
  Real.cos x / Real.sin x

theorem cot_trig_identity (a b c α β γ : Real) 
  (habc : a^2 + b^2 = 2021 * c^2) 
  (hα : α = Real.arcsin (a / c)) 
  (hβ : β = Real.arcsin (b / c)) 
  (hγ : γ = Real.arccos ((2021 * c^2 - a^2 - b^2) / (2 * 2021 * c^2))) 
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  cot α / (cot β + cot γ) = 1010 :=
by
  sorry

end cot_trig_identity_l1969_196908


namespace quadratic_negativity_cond_l1969_196990

theorem quadratic_negativity_cond {x m k : ℝ} :
  (∀ x, x^2 - m * x - k + m < 0) ↔ k > m - (m^2 / 4) :=
sorry

end quadratic_negativity_cond_l1969_196990


namespace apples_problem_l1969_196999

theorem apples_problem :
  ∃ (jackie rebecca : ℕ), (rebecca = 2 * jackie) ∧ (∃ (adam : ℕ), (adam = jackie + 3) ∧ (adam = 9) ∧ jackie = 6 ∧ rebecca = 12) :=
by
  sorry

end apples_problem_l1969_196999


namespace coprime_exist_m_n_l1969_196986

theorem coprime_exist_m_n (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_a : a ≥ 1) (h_b : b ≥ 1) :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ a^m + b^n ≡ 1 [MOD a * b] :=
by
  use Nat.totient b, Nat.totient a
  sorry

end coprime_exist_m_n_l1969_196986


namespace number_of_cars_l1969_196964

theorem number_of_cars (n s t C : ℕ) (h1 : n = 9) (h2 : s = 4) (h3 : t = 3) (h4 : n * s = t * C) : C = 12 :=
by
  sorry

end number_of_cars_l1969_196964


namespace part1_solution_part2_solution_l1969_196940

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l1969_196940


namespace christina_age_fraction_l1969_196912

theorem christina_age_fraction {C : ℕ} (h1 : ∃ C : ℕ, (6 + 15) = (3/5 : ℚ) * C)
  (h2 : C + 5 = 40) : (C + 5) / 80 = 1 / 2 :=
by
  sorry

end christina_age_fraction_l1969_196912


namespace Dawn_has_10_CDs_l1969_196937

-- Lean definition of the problem conditions
def Kristine_more_CDs (D K : ℕ) : Prop :=
  K = D + 7

def Total_CDs (D K : ℕ) : Prop :=
  D + K = 27

-- Lean statement of the proof
theorem Dawn_has_10_CDs (D K : ℕ) (h1 : Kristine_more_CDs D K) (h2 : Total_CDs D K) : D = 10 :=
by
  sorry

end Dawn_has_10_CDs_l1969_196937


namespace Tino_jellybeans_l1969_196995

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l1969_196995


namespace red_balls_count_l1969_196962

theorem red_balls_count (y : ℕ) (p_yellow : ℚ) (h1 : y = 10)
  (h2 : p_yellow = 5/8) (total_balls_le : ∀ r : ℕ, y + r ≤ 32) :
  ∃ r : ℕ, 10 + r > 0 ∧ p_yellow = 10 / (10 + r) ∧ r = 6 :=
by
  sorry

end red_balls_count_l1969_196962


namespace min_value_of_sequence_l1969_196994

variable (b1 b2 b3 : ℝ)

def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  ∃ s : ℝ, b2 = b1 * s ∧ b3 = b1 * s^2 

theorem min_value_of_sequence (h1 : b1 = 2) (h2 : geometric_sequence b1 b2 b3) :
  ∃ s : ℝ, 3 * b2 + 4 * b3 = -9 / 8 :=
sorry

end min_value_of_sequence_l1969_196994


namespace fill_cistern_time_l1969_196978

theorem fill_cistern_time (F E : ℝ) (hF : F = 1/2) (hE : E = 1/4) : 
  (1 / (F - E)) = 4 :=
by
  -- Definitions of F and E are used as hypotheses hF and hE
  -- Prove the actual theorem stating the time to fill the cistern is 4 hours
  sorry

end fill_cistern_time_l1969_196978


namespace max_cos_product_l1969_196980

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end max_cos_product_l1969_196980


namespace solve_system_l1969_196945

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end solve_system_l1969_196945


namespace range_of_x_l1969_196956

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x - 1

def g (x a : ℝ) : ℝ := 3 * x^2 - a * x + 3 * a - 5

def condition (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem range_of_x (x a : ℝ) (h : condition a) : g x a < 0 → -2/3 < x ∧ x < 1 := 
sorry

end range_of_x_l1969_196956


namespace work_rate_l1969_196998

theorem work_rate (x : ℕ) (hx : 2 * x = 30) : x = 15 := by
  -- We assume the prerequisite 2 * x = 30
  sorry

end work_rate_l1969_196998


namespace field_length_l1969_196948

theorem field_length (w l : ℕ) (Pond_Area : ℕ) (Pond_Field_Ratio : ℚ) (Field_Length_Ratio : ℕ) 
  (h1 : Length = 2 * Width)
  (h2 : Pond_Area = 8 * 8)
  (h3 : Pond_Field_Ratio = 1 / 50)
  (h4 : Pond_Area = Pond_Field_Ratio * Field_Area)
  : l = 80 := 
by
  -- begin solution
  sorry

end field_length_l1969_196948


namespace rita_bought_4_jackets_l1969_196971

/-
Given:
  - Rita bought 5 short dresses costing $20 each.
  - Rita bought 3 pairs of pants costing $12 each.
  - The jackets cost $30 each.
  - She spent an additional $5 on transportation.
  - Rita had $400 initially.
  - Rita now has $139.

Prove that the number of jackets Rita bought is 4.
-/

theorem rita_bought_4_jackets :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let transportation_cost := 5
  let initial_amount := 400
  let remaining_amount := 139
  let jackets_cost_per_unit := 30
  let total_spent := initial_amount - remaining_amount
  let total_clothes_transportation_cost := dresses_cost + pants_cost + transportation_cost
  let jackets_cost := total_spent - total_clothes_transportation_cost
  let number_of_jackets := jackets_cost / jackets_cost_per_unit
  number_of_jackets = 4 :=
by
  sorry

end rita_bought_4_jackets_l1969_196971


namespace necessary_and_sufficient_condition_l1969_196902

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ (a = 1 ∨ b = 1) :=
sorry

end necessary_and_sufficient_condition_l1969_196902


namespace part_i_part_ii_l1969_196959

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

-- Part I: Prove solution to the inequality.
theorem part_i (x : ℝ) : f x 1 > 3 ↔ x ∈ {x | x < 0} ∪ {x | x > 3} :=
sorry

-- Part II: Prove the inequality for general a and b with condition for equality.
theorem part_ii (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  f b a ≥ f a a ∧ ((2 * a - b = 0 ∨ b - a = 0) ∨ (2 * a - b > 0 ∧ b - a > 0) ∨ (2 * a - b < 0 ∧ b - a < 0)) ↔ f b a = f a a :=
sorry

end part_i_part_ii_l1969_196959


namespace trapezoid_area_l1969_196957

noncomputable def area_trapezoid (B1 B2 h : ℝ) : ℝ := (1 / 2 * (B1 + B2) * h)

theorem trapezoid_area
    (h1 : ∀ x : ℝ, 3 * x = 10 → x = 10 / 3)
    (h2 : ∀ x : ℝ, 3 * x = 5 → x = 5 / 3)
    (h3 : B1 = 10 / 3)
    (h4 : B2 = 5 / 3)
    (h5 : h = 5)
    : area_trapezoid B1 B2 h = 12.5 := by
  sorry

end trapezoid_area_l1969_196957


namespace probability_average_is_five_l1969_196947

-- Definitions and conditions
def numbers : List ℕ := [1, 3, 4, 6, 7, 9]

def average_is_five (a b : ℕ) : Prop := (a + b) / 2 = 5

-- Desired statement
theorem probability_average_is_five : 
  ∃ p : ℚ, p = 1 / 5 ∧ (∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ average_is_five a b) := 
sorry

end probability_average_is_five_l1969_196947


namespace books_in_shipment_l1969_196974

theorem books_in_shipment (B : ℕ) (h : 3 / 4 * B = 180) : B = 240 :=
sorry

end books_in_shipment_l1969_196974


namespace fraction_not_on_time_l1969_196915

theorem fraction_not_on_time (n : ℕ) (h1 : ∃ (k : ℕ), 3 * k = 5 * n) 
(h2 : ∃ (k : ℕ), 4 * k = 5 * m) 
(h3 : ∃ (k : ℕ), 5 * k = 6 * f) 
(h4 : m + f = n) 
(h5 : r = rm + rf) 
(h6 : rm = 4/5 * m) 
(h7 : rf = 5/6 * f) :
  (not_on_time : ℚ) = 1/5 := 
by
  sorry

end fraction_not_on_time_l1969_196915


namespace circle_eq_focus_tangent_directrix_l1969_196949

theorem circle_eq_focus_tangent_directrix (x y : ℝ) :
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  ((x - focus.1)^2 + (y - focus.2)^2 = radius^2) :=
by
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  sorry

end circle_eq_focus_tangent_directrix_l1969_196949


namespace substance_volume_proportional_l1969_196927

theorem substance_volume_proportional (k : ℝ) (V₁ V₂ : ℝ) (W₁ W₂ : ℝ) 
  (h1 : V₁ = k * W₁) 
  (h2 : V₂ = k * W₂) 
  (h3 : V₁ = 48) 
  (h4 : W₁ = 112) 
  (h5 : W₂ = 84) 
  : V₂ = 36 := 
  sorry

end substance_volume_proportional_l1969_196927


namespace more_people_attended_l1969_196972

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l1969_196972


namespace boxes_left_l1969_196932

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l1969_196932


namespace div_sub_mult_exp_eq_l1969_196992

-- Lean 4 statement for the mathematical proof problem
theorem div_sub_mult_exp_eq :
  8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := 
sorry

end div_sub_mult_exp_eq_l1969_196992


namespace find_pairs_l1969_196922

theorem find_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) ↔ (a = b) := by
  sorry

end find_pairs_l1969_196922


namespace committee_count_l1969_196969

theorem committee_count (students : Finset ℕ) (Alice : ℕ) (hAlice : Alice ∈ students) (hCard : students.card = 7) :
  ∃ committees : Finset (Finset ℕ), (∀ c ∈ committees, Alice ∈ c ∧ c.card = 4) ∧ committees.card = 20 :=
sorry

end committee_count_l1969_196969
