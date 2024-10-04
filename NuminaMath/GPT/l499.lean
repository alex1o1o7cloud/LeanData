import Mathlib

namespace path_area_approximation_l499_499013

noncomputable def π : ℝ := real.pi

def diameter_of_garden : ℝ := 4 -- meters

def width_of_path : ℝ := 0.25 -- meters

def radius_inner := diameter_of_garden / 2
def radius_outer := radius_inner + width_of_path

def area (r : ℝ) : ℝ := π * (r^2)

def area_inner := area radius_inner
def area_outer := area radius_outer

def area_path := area_outer - area_inner

theorem path_area_approximation : abs (area_path - 3.338) < 0.001 :=
by
  sorry

end path_area_approximation_l499_499013


namespace sum_gcd_values_l499_499259

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499259


namespace angle_bisector_sine_half_angle_l499_499349

theorem angle_bisector_sine_half_angle (h_a h_b l : ℝ) (C : ℝ) (h_pos : h_a > 0) (hb_pos : h_b > 0) (l_pos : l > 0) :
  sin (C / 2) = h_a * h_b / (l * (h_a + h_b)) :=
sorry

end angle_bisector_sine_half_angle_l499_499349


namespace gcd_values_sum_l499_499081

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499081


namespace largest_unpayable_soldo_l499_499897

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499897


namespace infinite_geometric_series_sum_l499_499721

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499721


namespace irrational_last_non_zero_digits_l499_499987

def last_non_zero_digit (n : ℕ) : ℕ := sorry

theorem irrational_last_non_zero_digits :
  (∀ (n : ℕ), let hn := last_non_zero_digit (fact n) in True) →
  ¬ ∃ (N T : ℕ), ∀ n ≥ N, last_non_zero_digit (fact (n + T)) = last_non_zero_digit (fact n) :=
sorry

end irrational_last_non_zero_digits_l499_499987


namespace jerry_initial_tickets_l499_499270

/-- Jerry won some tickets, spent 2 tickets on a beanie, and later won 47 more tickets to have a total of 49 tickets. Prove that Jerry initially won 4 tickets. -/
theorem jerry_initial_tickets
  (spent_on_beanie : ℕ) (won_later : ℕ) (total_now : ℕ)
  (h1 : spent_on_beanie = 2)
  (h2 : won_later = 47)
  (h3 : total_now = 49) :
  total_now - won_later + spent_on_beanie = 4 :=
by
  rw [h1, h2, h3]
  -- lean code ensures correct statement up to this part
  sorry

end jerry_initial_tickets_l499_499270


namespace atomic_weight_of_oxygen_l499_499739

theorem atomic_weight_of_oxygen (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (molecular_weight_Al2O3 : ℝ) (n_Al : ℕ) (n_O : ℕ) :
  atomic_weight_Al = 26.98 →
  molecular_weight_Al2O3 = 102 →
  n_Al = 2 →
  n_O = 3 →
  (molecular_weight_Al2O3 - n_Al * atomic_weight_Al) / n_O = 16.01 :=
by
  sorry

end atomic_weight_of_oxygen_l499_499739


namespace star_area_l499_499388

-- Conditions
def square_ABCD_area (s : ℝ) := s^2 = 72

-- Question and correct answer
theorem star_area (s : ℝ) (h : square_ABCD_area s) : 24 = 24 :=
by sorry

end star_area_l499_499388


namespace x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l499_499366

theorem x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1 (x : ℝ) : (x > 1 → |x| > 1) ∧ (¬(x > 1 ↔ |x| > 1)) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l499_499366


namespace solution_system_l499_499732

noncomputable def solution_pairs : set (ℝ × ℝ) := 
  { (1 / 3, 1 / 3), (real.root 4 31 / 12, real.root 4 31 / 3) }

theorem solution_system :
  ∀ x y : ℝ, 0 < x → 0 < y →
    (3 * y - real.sqrt (y / x) - 6 * real.sqrt (x * y) + 2 = 0 ∧ 
     x^2 + 81 * x^2 * y^4 = 2 * y^2) ↔ 
    (x, y) ∈ solution_pairs := 
by
  sorry

end solution_system_l499_499732


namespace main_problem_l499_499353

-- Define the set A
def A (a : ℝ) : Set ℝ :=
  {0, 1, a^2 - 2 * a}

-- Define the main problem as a theorem
theorem main_problem (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 :=
  sorry

end main_problem_l499_499353


namespace gcd_5n_plus_6_n_sum_l499_499101

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499101


namespace relay_race_proof_l499_499051

def John_time (S : ℝ) : ℝ := 2 * S
def Susan_time (Jen : ℝ) : ℝ := Jen + 15
def Jen_time : ℝ := 200 / 6
def Emily_time (J : ℝ) : ℝ := J / 2
def Ryan_time (S : ℝ) : ℝ := S - 20
def Lydia_time : ℝ := 200 / 4
def Tiffany_time (R E : ℝ) : ℝ := R + E
def David_time (J : ℝ) : ℝ := J

def relay_race_total_time_proof : Prop :=
  ∃ J S Jen E R L T D,
    Jen = Jen_time ∧
    S = Susan_time Jen ∧
    J = John_time S ∧
    E = Emily_time J ∧
    R = Ryan_time S ∧
    L = Lydia_time ∧
    T = Tiffany_time R E ∧
    D = David_time J ∧
    (J + S + Jen + E + R + L + T + D = 478.3)

theorem relay_race_proof : relay_race_total_time_proof := by
  sorry

end relay_race_proof_l499_499051


namespace sequence_repetition_l499_499531

-- defining the initial conditions
def initial_sequence (a : ℕ) : Prop :=
  a > 0

-- defining Foma's operation (adds any digit of the previous number to it)
def Foma_step (a b : ℕ) : Prop :=
  ∃ d ∈ (List.ofDigits a.digits), b = a + d

-- defining Yerema's operation (subtracts any digit of the previous number from it)
def Yerema_step (a b : ℕ) : Prop :=
  ∃ d ∈ (List.ofDigits a.digits), b = a - d

-- main theorem statement
theorem sequence_repetition :
  ∀ (a : ℕ), initial_sequence a →
  ∃ n x, (x > 0) ∧ (∀ k < 100, x = iterate (Foma_step ∘ Yerema_step) k a) 
  -- To prove: some number in this sequence will repeat at least 100 times.
  sorry

end sequence_repetition_l499_499531


namespace compare_logs_l499_499818

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l499_499818


namespace calc1_calc2_calc3_calc4_l499_499635

-- Problem 1
theorem calc1 : (-2: ℝ) ^ 2 - (7 - Real.pi) ^ 0 - (1 / 3) ^ (-1: ℝ) = 0 := by
  sorry

-- Problem 2
variable (m : ℝ)
theorem calc2 : 2 * m ^ 3 * 3 * m - (2 * m ^ 2) ^ 2 + m ^ 6 / m ^ 2 = 3 * m ^ 4 := by
  sorry

-- Problem 3
variable (a : ℝ)
theorem calc3 : (a + 1) ^ 2 + (a + 1) * (a - 2) = 2 * a ^ 2 + a - 1 := by
  sorry

-- Problem 4
variables (x y : ℝ)
theorem calc4 : (x + y - 1) * (x - y - 1) = x ^ 2 - 2 * x + 1 - y ^ 2 := by
  sorry

end calc1_calc2_calc3_calc4_l499_499635


namespace Bryce_received_raisins_l499_499795

theorem Bryce_received_raisins :
  ∃ x : ℕ, (∀ y : ℕ, x = y + 6) ∧ (∀ z : ℕ, z = x / 2) → x = 12 :=
by
  sorry

end Bryce_received_raisins_l499_499795


namespace Jonie_cousins_ages_l499_499927

theorem Jonie_cousins_ages : 
  ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  1 ≤ a ∧ a < 10 ∧
  1 ≤ b ∧ b < 10 ∧
  1 ≤ c ∧ c < 10 ∧
  1 ≤ d ∧ d < 10 ∧
  a * b = 24 ∧
  c * d = 30 ∧
  a + b + c + d = 22 :=
sorry

end Jonie_cousins_ages_l499_499927


namespace polynomial_integer_roots_a_value_l499_499305

open Polynomial

theorem polynomial_integer_roots_a_value (α β γ : ℤ) (a : ℤ) :
  (X - C α) * (X - C β) * (X - C γ) = X^3 - 2 * X^2 - 25 * X + C a →
  α + β + γ = 2 →
  α * β + α * γ + β * γ = -25 →
  a = -50 :=
by
  sorry

end polynomial_integer_roots_a_value_l499_499305


namespace gcd_sum_5n_plus_6_n_l499_499148

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499148


namespace sum_of_gcd_values_l499_499115

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499115


namespace Genna_travelled_distance_l499_499311

theorem Genna_travelled_distance : 
  ∀ (T F c x : ℝ), 
  T = 74.16 → 
  F = 45 → 
  c = 0.12 → 
  T = F + c * x → 
  x = 243 := 
begin
  intros T F c x hT hF hc h_eq,
  sorry
end

end Genna_travelled_distance_l499_499311


namespace sum_of_consecutive_negatives_l499_499512

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l499_499512


namespace geometric_series_sum_l499_499686

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499686


namespace age_of_seventh_person_l499_499574

-- Define conditions as Lean expressions
variables (A : ℕ)
variable (total_age_of_6_after_2_years : ℕ := 6 * 43)
variable (increment_in_age : ℕ := 6 * 2)
variable (current_total_age_of_6 : ℕ := total_age_of_6_after_2_years - increment_in_age)
variable (total_age_with_seventh : ℕ := 7 * 45)

-- Define the main theorem
theorem age_of_seventh_person : 315 - 246 = 69 :=
by
  have h1 : total_age_of_6_after_2_years = 258 := by simp [total_age_of_6_after_2_years]
  have h2 : increment_in_age = 12 := by simp [increment_in_age]
  have h3 : current_total_age_of_6 = 246 := by simp [current_total_age_of_6 h1 h2]
  have h4 : total_age_with_seventh = 315 := by simp [total_age_with_seventh]
  rw [h3] -- plug in the current total age of six
  rw [h4] -- plug in the total age with the seventh
  simp -- simplify to get 69

end age_of_seventh_person_l499_499574


namespace inequality_proof_l499_499334

theorem inequality_proof (n : ℕ) (t : ℝ) (x : Fin n → ℝ)
  (h_t : 0 ≤ t ∧ t ≤ 1)
  (h_x : ∀ i, (1 : ℝ) ≥ x i ∧ 0 < x i ∧ (i.val < n.pred → x i ≥ x ⟨i.val + 1, Nat.succ_lt_succ i.is_lt⟩)) :
  (1 + ∑ i in Finset.range n, x ⟨i, Fin.is_lt⟩) ^ t ≤
  1 + ∑ i in Finset.range n, (i + 1 : ℝ) ^ (t - 1) * (x ⟨i, Fin.is_lt⟩ ^ t) :=
sorry

end inequality_proof_l499_499334


namespace sum_of_digits_of_T_l499_499957

-- Definitions based on the conditions
def horse_lap_time (n : ℕ) : ℕ := n
def horses_starting_together (n : ℕ) : ℕ :=
  (List.range 12).countp (λ k, n % horse_lap_time (k + 1) = 0)

-- Problem statement in Lean
theorem sum_of_digits_of_T :
  ∃ T > 0, horses_starting_together T ≥ 6 ∧ (T = 12) ∧ (Nat.digits 10 T).sum = 3 :=
by
  sorry

end sum_of_digits_of_T_l499_499957


namespace total_listening_days_l499_499339

theorem total_listening_days (x y z t : ℕ) (h1 : x = 8) (h2 : y = 12) (h3 : z = 30) (h4 : t = 2) :
  (x + y + z) * t = 100 :=
by
  sorry

end total_listening_days_l499_499339


namespace sum_gcd_values_l499_499262

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499262


namespace only_point_D_lies_on_graph_l499_499007

def point := ℤ × ℤ

def lies_on_graph (f : ℤ → ℤ) (p : point) : Prop :=
  f p.1 = p.2

def f (x : ℤ) : ℤ := 2 * x - 1

theorem only_point_D_lies_on_graph :
  (lies_on_graph f (-1, 3) = false) ∧ 
  (lies_on_graph f (0, 1) = false) ∧ 
  (lies_on_graph f (1, -1) = false) ∧ 
  (lies_on_graph f (2, 3)) := 
by
  sorry

end only_point_D_lies_on_graph_l499_499007


namespace seats_capacity_l499_499837

theorem seats_capacity (x : ℕ) (h1 : 15 * x + 12 * x + 8 = 89) : x = 3 :=
by
  -- proof to be filled in
  sorry

end seats_capacity_l499_499837


namespace jasmine_laps_per_afternoon_l499_499848

-- Defining the conditions
def swims_each_week (days_per_week : ℕ) := days_per_week = 5
def total_weeks := 5
def total_laps := 300

-- Main proof statement
theorem jasmine_laps_per_afternoon (d : ℕ) (l : ℕ) :
  swims_each_week d →
  total_weeks * d = 25 →
  total_laps = 300 →
  300 / 25 = l →
  l = 12 :=
by
  intros
  -- Skipping the proof
  sorry

end jasmine_laps_per_afternoon_l499_499848


namespace inequality_1_inequality_2_l499_499374

variables {x a b : ℝ}

-- Given condition that the solution set of ax > b is (-∞, 1/5)
def condition (a b : ℝ) : Prop := a < 0 ∧ b / a = 1 / 5

-- 1. Solution set of ax^2 + bx - 4/5a > 0 is (-1, 4/5)
theorem inequality_1 (h : condition a b) : 
  { x : ℝ | a * x ^ 2 + b * x - 4 / 5 * a > 0 } = set.Ioo (-1) (4 / 5) := sorry

-- 2. Solution set of 2^(ax - b) ≤ 1 is [1/5, ∞)
theorem inequality_2 (h : condition a b) : 
  { x : ℝ | 2^(a * x - b) ≤ 1 } = set.Ici (1 / 5) := sorry

end inequality_1_inequality_2_l499_499374


namespace circle_condition_l499_499497

theorem circle_condition (a : ℝ) :
  x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 →
  (coeff_x² = 1 ∧ coeff_y² = a + 2 ∧ coeff_x² = coeff_y²) → a = -1 := 
sorry

end circle_condition_l499_499497


namespace percentage_taxed_l499_499937

theorem percentage_taxed (T : ℝ) (H1 : 3840 = T * (P : ℝ)) (H2 : 480 = 0.25 * T * (P : ℝ)) : P = 0.5 := 
by
  sorry

end percentage_taxed_l499_499937


namespace sum_of_gcd_values_l499_499122

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499122


namespace incenter_length_l499_499537

theorem incenter_length (P Q R I : Point) 
  (h_right : is_right_triangle P Q R) 
  (h_PQ : dist P Q = 20) 
  (h_PR : dist P R = 21) 
  (h_QR : dist Q R = 29)
  (h_incenter : is_incenter I P Q R) : 
  dist P I = 6 := 
sorry

end incenter_length_l499_499537


namespace gcd_sum_5n_plus_6_n_l499_499151

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499151


namespace distance_between_lines_is_correct_l499_499040

-- Definition of the lines and the distance formula
def line1 (x y : ℝ) := 3 * x + 4 * y + 5 = 0
def line2 (x y : ℝ) := 3 * x + 4 * y - 7 = 0

-- Function to calculate the distance between two parallel lines
def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ := abs (C2 - C1) / real.sqrt (A^2 + B^2)

-- The main theorem statement
theorem distance_between_lines_is_correct :
  distance_between_parallel_lines 3 4 5 (-7) = 12 / 5 :=
by
  sorry

end distance_between_lines_is_correct_l499_499040


namespace largest_unpayable_soldo_l499_499898

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499898


namespace palindrome_pair_count_is_35_l499_499035

-- Definition of a four-digit palindrome with given difference constraint.
def is_palindrome (n : ℕ) : Prop := 
  let a := n / 1000 % 10 in
  let b := n / 100 % 10 in
  let c := n / 10 % 10 in
  let d := n % 10 in
  a = d ∧ b = c

noncomputable def palindrome_count_difference_3674 : ℕ :=
  let count := ∑ a in Finset.range 10, 
                ∑ b in Finset.range 10, 
                ∑ c in Finset.range 10, 
                ∑ d in Finset.range 10, 
                if is_palindrome (1001 * a + 110 * b + 11 * c + d) ∧ 
                   is_palindrome (1001 * c + 110 * d + 11 * a + b) ∧ 
                   (abs ((1001 * a + 110 * b + 11 * c + d) - (1001 * c + 110 * d + 11 * a + b)) = 3674)
                then 1 else 0
  count

-- Theorem to match the result with the known solution.
theorem palindrome_pair_count_is_35 : palindrome_count_difference_3674 = 35 :=
sorry

end palindrome_pair_count_is_35_l499_499035


namespace problem1_l499_499989

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1_l499_499989


namespace gcd_sum_5n_plus_6_n_l499_499141

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499141


namespace calculate_estate_l499_499874

noncomputable def estate := ℝ

variables (E : estate) (y : estate)

-- Condition 1: shares of the daughters and son in the ratio 5:3:2
def first_daughter_share := 5 * y
def second_daughter_share := 3 * y
def son_share := 2 * y

-- Condition 2: husband receives twice as much as the son
def husband_share := 4 * y

-- Condition 3: gardener receives $600
def gardener_share := 600

-- Condition 4: charity receives $800
def charity_share := 800

-- Total estate calculation based on conditions
def total_shares :=
  first_daughter_share +
  second_daughter_share +
  son_share +
  husband_share +
  gardener_share +
  charity_share

-- Stating the question with the correct answer in Lean theorem format
theorem calculate_estate : 
  (3/5) * E = first_daughter_share + second_daughter_share + son_share →
  E = 14 * y + 1400 →
  E = 50/3 * y →
  total_shares = E :=
sorry

end calculate_estate_l499_499874


namespace sum_of_gcd_values_l499_499216

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499216


namespace range_of_a_l499_499770

-- Problem statement and conditions definition
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def Q (a : ℝ) : Prop := (5 - 2 * a) > 1

-- Proof problem statement
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
sorry

end range_of_a_l499_499770


namespace S_2014_value_l499_499338

-- Conditions
def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x

def tangent_parallel_to_line (b : ℝ) : Prop :=
  let deriv := 2 * 1 + b
  deriv = 3

def sequence_a (n : ℕ) (b : ℝ) : ℝ :=
  1 / (f n b)

def sum_S (n : ℕ) (b : ℝ) : ℝ :=
  (Finset.range n).sum (λ k, sequence_a (k + 1) b)

-- Proof statement
theorem S_2014_value (b : ℝ) (h1 : tangent_parallel_to_line b) : 
  sum_S 2014 b = 2014 / 2015 :=
by
  sorry

end S_2014_value_l499_499338


namespace triangle_angle_bisector_tangent_l499_499845

theorem triangle_angle_bisector_tangent 
    (A B C D : Type) 
    [IsTriangle ABC]
    (hC : ∠ ACB = 90) 
    (hAD : AngleBisector ∠ BAC D) 
    : (AB.length - AC.length) / CD.length = tan (∠ BAC) :=
sorry

end triangle_angle_bisector_tangent_l499_499845


namespace amount_of_water_at_month_end_l499_499271

noncomputable def water_contained (C : ℝ) : ℝ := 0.70 * C

theorem amount_of_water_at_month_end (C : ℝ) (h1 : water_contained C = 0.70 * C)
    (h2 : 2 * (C - 10) = 0.70 * C) : water_contained C ≈ 10.766 := 
sorry

end amount_of_water_at_month_end_l499_499271


namespace gcd_sum_5n_plus_6_l499_499199

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499199


namespace total_games_over_three_years_l499_499622

-- Definitions based on the problem conditions
def games_this_year : ℕ := 11
def decrease_percentage : ℝ := 0.15
def increase_percentage : ℝ := 0.20

-- Calculations based on the conditions
def games_last_year : ℕ := (games_this_year / (1 - decrease_percentage)).to_nat
def games_next_year : ℕ := (games_this_year * (1 + increase_percentage)).to_nat

-- The main statement we want to prove
theorem total_games_over_three_years :
  games_last_year + games_this_year + games_next_year = 37 :=
  sorry

end total_games_over_three_years_l499_499622


namespace distinct_solutions_count_l499_499505

theorem distinct_solutions_count : 
  {p : ℝ × ℝ // p.1 = p.1^2 + 2 * p.2^2 ∧ p.2 = 3 * p.1 * p.2}.toFinset.card = 4 := 
by
  sorry

end distinct_solutions_count_l499_499505


namespace gcd_sum_5n_plus_6_l499_499201

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499201


namespace no_seven_sum_possible_l499_499871

theorem no_seven_sum_possible :
  let outcomes := [-1, -3, -5, 2, 4, 6]
  ∀ (a b : Int), a ∈ outcomes → b ∈ outcomes → a + b ≠ 7 :=
by
  sorry

end no_seven_sum_possible_l499_499871


namespace gcd_sum_l499_499156

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499156


namespace part1_part2_l499_499784

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 2 * Real.log x

-- Define the first proof problem
theorem part1 (a : ℝ) (x : ℝ) (hx : x ≥ 1) (hf : ∀ x > 0, f a x = -f a (1 / x)) : a ∈ set.Ici 1 → f a x ≥ 0 :=
sorry

-- Define the second proof problem
theorem part2 (n : ℕ) (hn : n ≥ 2) : ∑ i in Finset.range (n - 1) + 2, (1 : ℝ) / (i + 2) ^ 2 > 2 * Real.log (2 * n / (n + 1)) - 3 / 4 :=
sorry

end part1_part2_l499_499784


namespace eval_expression_l499_499279

theorem eval_expression : 
  (-(1/2))⁻¹ - 4 * real.cos (30 * real.pi / 180) - (real.pi + 2013)^0 + real.sqrt 12 = -3 := 
by 
  sorry

end eval_expression_l499_499279


namespace number_of_true_propositions_is_zero_l499_499498

theorem number_of_true_propositions_is_zero :
  (∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0) →
  (¬ ∃ x : ℚ, x^2 = 2) →
  (¬ ∃ x : ℝ, x^2 + 1 = 0) →
  (∀ x : ℝ, 4 * x^2 ≤ 2 * x - 1 + 3 * x^2) →
  true :=  -- representing that the number of true propositions is 0
by
  intros h1 h2 h3 h4
  sorry

end number_of_true_propositions_is_zero_l499_499498


namespace fraction_shaded_square_l499_499467

theorem fraction_shaded_square (x : ℝ) (h : 0 < x) :
  let P := (0, x / 2) in
  let Q := (x / 2, x) in
  (1 - (1 / 8)) = 7 / 8 :=
by 
  -- Definitions
  sorry

end fraction_shaded_square_l499_499467


namespace quotient_when_maxCarsPassing_div_10_l499_499457

noncomputable def maxCarsPassing (carLength : ℝ) (separationFactor : ℕ → ℝ) : ℝ :=
  let speed (n : ℕ) := 10 * n
  let distanceBetweenCars (n : ℕ) := carLength * (n + 1)
  let carsPerHour (n : ℕ) := (10000 * n) / distanceBetweenCars n
  lim (λn, carsPerHour n)

theorem quotient_when_maxCarsPassing_div_10:
  ∀ carLength separationFactor,
  ( carLength = 5 ) →
  ( ∀ n, separationFactor n = 1 ) →
  ( maxCarsPassing carLength separationFactor / 10 = 200) :=
by
  intro carLength separationFactor h1 h2
  have m : ℝ := maxCarsPassing carLength separationFactor
  sorry

end quotient_when_maxCarsPassing_div_10_l499_499457


namespace gumball_machine_total_l499_499588

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l499_499588


namespace seating_arrangement_l499_499662

theorem seating_arrangement (x y : ℕ) (h1 : x * 8 + y * 7 = 55) : x = 6 :=
by
  sorry

end seating_arrangement_l499_499662


namespace remainder_poly_div_l499_499365

theorem remainder_poly_div (D E F : ℝ) (q := λ x : ℝ, D * x^6 + E * x^4 + F * x^2 + 6) :
  q 2 = 16 → q (-2) = 16 :=
by
  -- placeholder for proof
  sorry

end remainder_poly_div_l499_499365


namespace infinite_geometric_series_sum_l499_499718

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499718


namespace sum_of_first_9_terms_l499_499763

-- Define the arithmetic sequence {a_n} and the sum S_n of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions given in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom arith_seq : arithmetic_sequence a
axiom sum_terms : sum_of_first_n_terms a S
axiom S3 : S 3 = 30
axiom S6 : S 6 = 100

-- Goal: Prove that S 9 = 170
theorem sum_of_first_9_terms : S 9 = 170 :=
sorry -- Placeholder for the proof

end sum_of_first_9_terms_l499_499763


namespace speed_of_second_skier_l499_499000

theorem speed_of_second_skier :
  ∃ y : ℝ, 
    (∀ x : ℝ,
      (9 / y + 9 = 9 / x) ∧ 
      (29 / y + 9 = 25 / x) →
      x = 4/5 * y) ∧
    (y = 15) :=
begin
  sorry
end

end speed_of_second_skier_l499_499000


namespace num_divisors_of_3b_plus_18_l499_499425

theorem num_divisors_of_3b_plus_18 (a b : ℤ) (h : 5 * b = 10 - 3 * a) : 
  {n ∈ ([1, 2, 3, 4, 5, 6, 7, 8] : list ℤ) | n ∣ (3 * b + 18)}.card = 3 :=
by
  sorry

end num_divisors_of_3b_plus_18_l499_499425


namespace sum_gcd_possible_values_eq_twelve_l499_499242

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499242


namespace unused_digits_sum_l499_499663

theorem unused_digits_sum :
  (∃ (used_digits : Finset ℕ), used_digits ⊆ (Finset.range 10) ∧ used_digits.card = 8 ∧ ∀ x ∈ used_digits, x ≤ 9) →
  ∑ x in (Finset.range 10 \ used_digits), x = 15 :=
by
  sorry

end unused_digits_sum_l499_499663


namespace one_var_linear_eq_solution_l499_499976

theorem one_var_linear_eq_solution : ∃ (x : ℝ), x = -2 ∧ (x + 2 = 0) :=
by
  use -2
  split
  . rfl
  . linarith

end one_var_linear_eq_solution_l499_499976


namespace find_root_l499_499433

def P (x : ℝ) : ℝ -- Define the polynomial P (x)
def P' (x : ℝ) : ℝ -- Define the derivative P' (x)

-- Given conditions
axiom h1 : P 1 = 0
axiom h2 : P 3 = 0
axiom h3 : P 5 = 0
axiom h4 : P' 7 = 0

-- Prove that the other root r4 of P(x) is 89/11
theorem find_root 
  (hP : P x = a * (x - 1) * (x - 3) * (x - 5) * (x - r4)) 
  (h_deriv : ∀ x, P' x = derivative (λ x, P x)) :
  r4 = 89 / 11 :=
by
  sorry

end find_root_l499_499433


namespace integral_sqrt_4_minus_x2_l499_499955

-- Define the integral and the bounds
def integral_definite (a b : ℝ) (f : ℝ → ℝ) := ∫ x in a..b, f x

-- The function to integrate
def integrand (x : ℝ) : ℝ := sqrt (4 - x^2)

-- The bounds of the integral
def lower_bound : ℝ := -2
def upper_bound : ℝ := 2

-- The statement we want to prove
theorem integral_sqrt_4_minus_x2 : integral_definite lower_bound upper_bound integrand = 2 * real.pi := by
  sorry

end integral_sqrt_4_minus_x2_l499_499955


namespace calc_expression_l499_499276

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end calc_expression_l499_499276


namespace gcd_sum_divisors_eq_12_l499_499179

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499179


namespace percentage_fraction_l499_499028

theorem percentage_fraction (P : ℚ) (hP : P < 35) (h : (P / 100) * 180 = 42) : P = 7 / 30 * 100 :=
by
  sorry

end percentage_fraction_l499_499028


namespace projections_concyclic_l499_499419

open Locale Real

variables {A B C D A' B' C' D' : EuclideanGeometry.Point}
variables {circle : EuclideanGeometry.Circle}

-- Given: Points A, B, C, and D lie on a circle
def is_concyclic (A B C D : EuclideanGeometry.Point) : Prop :=
  ∃ (circle : EuclideanGeometry.Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

-- Projection definitions
def is_orthogonal_projection (P Q R : EuclideanGeometry.Point) : Prop :=
  EuclideanGeometry.isRightAngle (P - Q) (R - Q)

variables (h_cocyclic : is_concyclic A B C D)
variables (h_A_proj : is_orthogonal_projection A' A B D)
variables (h_C_proj : is_orthogonal_projection C' C B D)
variables (h_B_proj : is_orthogonal_projection B' B A C)
variables (h_D_proj : is_orthogonal_projection D' D A C)

theorem projections_concyclic :
  is_concyclic A' B' C' D' :=
sorry

end projections_concyclic_l499_499419


namespace intersection_hyperbola_l499_499284

theorem intersection_hyperbola (t : ℝ) :
  ∃ A B : ℝ, ∀ (x y : ℝ),
  (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 5 = 0) →
  (x^2 / A - y^2 / B = 1) :=
sorry

end intersection_hyperbola_l499_499284


namespace maximize_NPM_l499_499005

theorem maximize_NPM :
  ∃ (M N P : ℕ), 
    (∀ M, M < 10 → (11 * M * M) = N * 100 + P * 10 + M) →
    N * 100 + P * 10 + M = 396 :=
by
  sorry

end maximize_NPM_l499_499005


namespace sum_of_gcd_values_l499_499213

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499213


namespace gcd_divisibility_count_l499_499435

open Nat

theorem gcd_divisibility_count (a b : ℕ) : 
  (finset.card (finset.filter (λ n, b ∣ n * a) (finset.range (b + 1)))) = gcd a b := 
sorry

end gcd_divisibility_count_l499_499435


namespace mode_of_data_set_l499_499322

theorem mode_of_data_set (x : ℝ) 
  (h1 : -1 ≤ 0) (h2 : 0 ≤ 4) (h3 : 4 ≤ x) (h4 : x ≤ 6) (h5 : 6 ≤ 15) 
  (h_median : (4 + x) / 2 = 5) : 
  mode [-1, 0, 4, x, 6, 15] = 6 :=
sorry

end mode_of_data_set_l499_499322


namespace sum_of_given_infinite_geometric_series_l499_499713

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499713


namespace sum_gcd_values_l499_499252

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499252


namespace largest_possible_m_is_correct_l499_499948

noncomputable def largest_possible_m : ℕ :=
  let q := [λ x, x - 1, λ x, x^4 + x^3 + x^2 + x + 1, λ x, x^5 + 1]
  q.length

theorem largest_possible_m_is_correct : largest_possible_m = 3 := 
  sorry

end largest_possible_m_is_correct_l499_499948


namespace Euler_point_l499_499862

noncomputable def Triangle (α : Type*) [Field α] :=
{x : α × α × α // x.1 + x.2 + x.3 = (0 : α)}

variables {α : Type*} [Field α]

structure Configuration (α : Type*) [Field α] :=
(A B C : α × α × α)
(O : α × α × α) -- Circumcenter
(H : α × α × α) -- Orthocenter
(M : α × α × α) -- Centroid

def Euler_line (t : Configuration α) : Prop :=
  ∃ r s : α, r ≠ 0 ∧ s ≠ 0 ∧ 
  t.M = r • t.O + s • t.H ∧ 
  r / s = (1 : α) / (2 : α)

theorem Euler_point (t : Configuration α) : Euler_line t :=
sorry

end Euler_point_l499_499862


namespace geometric_series_sum_l499_499704

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499704


namespace geometric_series_sum_l499_499688

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499688


namespace sum_gcd_possible_values_eq_twelve_l499_499238

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499238


namespace company_profits_ratio_l499_499378

def companyN_2008_profits (RN : ℝ) : ℝ := 0.08 * RN
def companyN_2009_profits (RN : ℝ) : ℝ := 0.15 * (0.8 * RN)
def companyN_2010_profits (RN : ℝ) : ℝ := 0.10 * (1.3 * 0.8 * RN)

def companyM_2008_profits (RM : ℝ) : ℝ := 0.12 * RM
def companyM_2009_profits (RM : ℝ) : ℝ := 0.18 * RM
def companyM_2010_profits (RM : ℝ) : ℝ := 0.14 * RM

def total_profits_N (RN : ℝ) : ℝ :=
  companyN_2008_profits RN + companyN_2009_profits RN + companyN_2010_profits RN

def total_profits_M (RM : ℝ) : ℝ :=
  companyM_2008_profits RM + companyM_2009_profits RM + companyM_2010_profits RM

theorem company_profits_ratio (RN RM : ℝ) :
  total_profits_N RN / total_profits_M RM = (0.304 * RN) / (0.44 * RM) :=
by
  unfold total_profits_N companyN_2008_profits companyN_2009_profits companyN_2010_profits
  unfold total_profits_M companyM_2008_profits companyM_2009_profits companyM_2010_profits
  simp
  sorry

end company_profits_ratio_l499_499378


namespace gumball_machine_total_l499_499590

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l499_499590


namespace range_of_m_l499_499352

theorem range_of_m (p_false : ¬ (∀ x : ℝ, ∃ m : ℝ, 2 * x + 1 + m = 0)) : ∀ m : ℝ, m ≤ 1 :=
sorry

end range_of_m_l499_499352


namespace gcd_sum_l499_499086

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499086


namespace gcd_sum_5n_plus_6_l499_499197

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499197


namespace largest_N_cannot_pay_exactly_without_change_l499_499882

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499882


namespace gumball_machine_total_gumballs_l499_499596

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l499_499596


namespace find_A_plus_B_l499_499428

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l499_499428


namespace gcd_5n_plus_6_n_sum_l499_499104

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499104


namespace gcd_sum_l499_499127

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499127


namespace meeting_percentage_l499_499849

theorem meeting_percentage
    (workday_hours : ℕ)
    (first_meeting_minutes : ℕ)
    (second_meeting_factor : ℕ)
    (hp_workday_hours : workday_hours = 10)
    (hp_first_meeting_minutes : first_meeting_minutes = 60)
    (hp_second_meeting_factor : second_meeting_factor = 2) 
    : (first_meeting_minutes + first_meeting_minutes * second_meeting_factor : ℚ) 
    / (workday_hours * 60) * 100 = 30 := 
by
  have workday_minutes := workday_hours * 60
  have second_meeting_minutes := first_meeting_minutes * second_meeting_factor
  have total_meeting_minutes := first_meeting_minutes + second_meeting_minutes
  have percentage := (total_meeting_minutes : ℚ) / workday_minutes * 100
  sorry

end meeting_percentage_l499_499849


namespace farmer_herd_l499_499032

theorem farmer_herd (n : ℕ) :
  let first_son := (2 / 5 : ℚ)
  let second_son := (1 / 5 : ℚ)
  let third_son := (1 / 10 : ℚ)
  let remaining_cows := 9
  let fourth_son := (1 - (first_son + second_son + third_son))
  (fourth_son * n).natAbs = remaining_cows → n = 30 :=
by
  sorry

end farmer_herd_l499_499032


namespace num_counterexamples_l499_499740

def digit_sum (n : ℕ) : ℕ := n.digits.sum

def has_digit_zero (n : ℕ) : Prop := 0 ∈ n.digits

def is_counterexample (n : ℕ) : Prop := 
  digit_sum n = 5 ∧ ¬(has_digit_zero n) ∧ ¬(Nat.Prime n)

theorem num_counterexamples : 
  (Finset.filter is_counterexample (Finset.Icc 1 99999)).card = 6 := 
sorry

end num_counterexamples_l499_499740


namespace total_money_shared_l499_499292

-- Definitions
def ratio : Nat × Nat × Nat := (3, 4, 5)
def emma_share : Nat := 45

-- Theorem to prove
theorem total_money_shared : 
  let factor := emma_share / ratio.1 in
  let finn_share := ratio.2 * factor in
  let grace_share := ratio.3 * factor in
  emma_share + finn_share + grace_share = 180 := 
by
  sorry

end total_money_shared_l499_499292


namespace angle_between_lines_l499_499469

noncomputable def cube := Type -- Placeholder for actual cube definition
variables (A B C D A1 B1 C1 D1 M N : cube)

axiom centers_of_faces :
  M = center_of_face A B C D ∧ 
  N = center_of_face B C C1 B1 -- Placeholder for center definition

theorem angle_between_lines :
  ∠ (line_thru D1 M) (line_thru A1 N) = 60 :=
by
  sorry

end angle_between_lines_l499_499469


namespace june_earnings_l499_499410

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l499_499410


namespace pentagon_angle_sum_l499_499385

theorem pentagon_angle_sum (A B C D Q : ℝ) (hA : A = 118) (hB : B = 105) (hC : C = 87) (hD : D = 135) :
  (A + B + C + D + Q = 540) -> Q = 95 :=
by
  sorry

end pentagon_angle_sum_l499_499385


namespace total_fencing_cost_l499_499296

-- Define the conditions
def diameter : ℝ := 30
def cost_per_meter : ℝ := 5
def pi_approx : ℝ := 3.14159

-- Calculate circumference using the definition of diameter
def circumference : ℝ := pi_approx * diameter

-- Calculate total cost using the definition of cost per meter
def total_cost : ℝ := circumference * cost_per_meter

-- State the theorem we need to prove
theorem total_fencing_cost : total_cost ≈ 471.25 :=
by
  sorry

end total_fencing_cost_l499_499296


namespace percentage_cleared_land_is_90_l499_499999

-- Let L be the total land owned by the farmer (approx 1000 acres)
def L : ℝ := 1000

-- Let C be the amount of cleared land
def C : ℝ := 0.9 * C + 90

-- Define the percentage P of the land cleared for planting
def P : ℝ := (C / L) * 100

-- Prove that P = 90
theorem percentage_cleared_land_is_90 (h : L ≈ 1000) : P = 90 :=
by
  sorry

end percentage_cleared_land_is_90_l499_499999


namespace inequality_abc_l499_499755

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by
  sorry

end inequality_abc_l499_499755


namespace proof_problem_l499_499569

def a := 876954
def b := 432196
def sqrt_2839 := 53.27
def cbrt_47108 := 36.12

theorem proof_problem :
  (a * a - b * b) / (a * a - b * b) * (sqrt_2839 + cbrt_47108) = 89.39 :=
by
  sorry

end proof_problem_l499_499569


namespace sum_of_consecutive_evens_is_162_l499_499521

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end sum_of_consecutive_evens_is_162_l499_499521


namespace tim_total_spent_l499_499016

variable (lunch_cost : ℝ)
variable (tip_percentage : ℝ)
variable (total_spent : ℝ)

theorem tim_total_spent (h_lunch_cost : lunch_cost = 60.80)
                        (h_tip_percentage : tip_percentage = 0.20)
                        (h_total_spent : total_spent = lunch_cost + (tip_percentage * lunch_cost)) :
                        total_spent = 72.96 :=
sorry

end tim_total_spent_l499_499016


namespace disproving_proposition_l499_499748

theorem disproving_proposition : ∃ (angle1 angle2 : ℝ), angle1 = angle2 ∧ angle1 + angle2 = 90 :=
by
  sorry

end disproving_proposition_l499_499748


namespace probability_odd_divisor_24_factorial_l499_499942

theorem probability_odd_divisor_24_factorial :
  let n := 24!
  let prime_factors := [(2, 22), (3, 10), (5, 4), (7, 3), (11, 2), (13, 1), (17, 1), (19, 1), (23, 1)]
  let num_divisors := (22 + 1) * (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * 2^4
  let num_odd_divisors := (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * 2^4
  in num_odd_divisors / num_divisors = 1 / 23 := sorry

end probability_odd_divisor_24_factorial_l499_499942


namespace angle_XPX_circle_radii_equation_l499_499567

-- Definitions based on given conditions
structure Circle (alpha : Type*) :=
(center : alpha)
(radius : ℝ)
(intersects: Π (P Q : α), P ≠ Q → Prop)

variables {α : Type*} (C C' : Circle α)
variables (P Q X X' R R': α)
variables (d r r' : ℝ)

-- Given Conditions
axiom circle_intersection : C.intersects P Q (by cc)
axiom centers_line_intersection : R ≠ P ∧ R' ≠ P
axiom PR'_line_intersection : P ≠ X ∧ C.intersects P X (by cc)
axiom PR_line_intersection : P ≠ X' ∧ C'.intersects P X' (by cc)
axiom collinear_QXX' : collinear Q X X'

-- Equivalent Proof Problems

theorem angle_XPX'_eq_pi_by_3 
  (h : d = dist C.center C'.center) 
  (r_eq : r = C.radius) 
  (r'_eq : r' = C'.radius): 
  angle X P X' = π / 3 := sorry

theorem circle_radii_equation 
  (h : d = dist C.center C'.center)
  (r_eq : r = C.radius) 
  (r'_eq : r' = C'.radius): 
  (d + r - r') * (d - r + r') = r * r' := sorry

end angle_XPX_circle_radii_equation_l499_499567


namespace gcd_sum_l499_499097

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499097


namespace gcd_sum_l499_499157

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499157


namespace compare_logs_l499_499811

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l499_499811


namespace length_of_chord_l499_499597

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line y = x - 1 with slope 1 passing through the focus (1, 0)
def line (x y : ℝ) : Prop :=
  y = x - 1

-- Prove that the length of the chord |AB| is 8
theorem length_of_chord 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line x1 y1) 
  (h4 : line x2 y2) : 
  abs (x2 - x1) = 8 :=
sorry

end length_of_chord_l499_499597


namespace max_area_rectangle_l499_499944

theorem max_area_rectangle (perimeter : ℕ) (a b : ℕ) (h1 : perimeter = 30) 
  (h2 : b = a + 3) : a * b = 54 :=
by
  sorry

end max_area_rectangle_l499_499944


namespace find_v_l499_499980

variable (v : ℝ)

def operation (v : ℝ) : ℝ := v - v / 3

theorem find_v (h : operation (operation v) = 4) : v = 9 :=
by
  sorry

end find_v_l499_499980


namespace initial_paint_amount_l499_499405

-- Conditions
def paint_used_per_house : ℝ := 4.3
def paint_used_total : ℝ := 2 * paint_used_per_house
def paint_remaining : ℝ := 8.8

-- Question and proof goal
theorem initial_paint_amount :
  let total_paint := paint_used_total + paint_remaining in
  total_paint = 17.4 :=
by
  sorry

end initial_paint_amount_l499_499405


namespace value_of_m_minus_n_over_n_l499_499361

theorem value_of_m_minus_n_over_n (m n : ℚ) (h : (2/3 : ℚ) * m = (5/6 : ℚ) * n) :
  (m - n) / n = 1 / 4 := 
sorry

end value_of_m_minus_n_over_n_l499_499361


namespace geometric_series_sum_l499_499703

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499703


namespace gcd_sum_divisors_eq_12_l499_499181

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499181


namespace min_value_of_n_l499_499624

theorem min_value_of_n 
  (n k : ℕ) 
  (h1 : 8 * n = 225 * k + 3)
  (h2 : k ≡ 5 [MOD 8]) : 
  n = 141 := 
  sorry

end min_value_of_n_l499_499624


namespace ellipse_properties_l499_499766

variables {a b c x y : ℝ}

def ellipse_equation := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (sqrt 2) / c = (sqrt 2) / 2)
  (h4 : 2 * c = 2 * sqrt 2) :
  (a = 2 ∧ b^2 = 2 ∧ ellipse_equation x y 4 2) ∧
  -- The product of vectors is 4.
  (∀ (m : ℝ), let y_P := (4 * m) / (m^2 + 2),
              let x_P := (2 * m^2 - 4) / (m^2 + 2),
              let y_M := 4 / m in
              (2, y_M) • (x_P, y_P) = 4) :=
begin
  -- Proof omitted
  sorry
end

end ellipse_properties_l499_499766


namespace compare_logs_l499_499821

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l499_499821


namespace arithmetic_sequence_common_difference_l499_499840

variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_common_difference
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
  (h_non_zero : d ≠ 0)
  (h_sum : a_n 1 + a_n 2 + a_n 3 = 9)
  (h_geom : a_n 2 ^ 2 = a_n 1 * a_n 5) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l499_499840


namespace sign_choice_sum_leq_a1_l499_499327

theorem sign_choice_sum_leq_a1 (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i)
  (h_cond : ∀ i : Fin (n - 1), a i ≤ a ⟨i + 1, Nat.ltTrans (Fin.isLt i) (Nat.lt_succ_self n)⟩ ∧ 
       a ⟨i + 1, Nat.ltTrans (Fin.isLt i) (Nat.lt_succ_self n)⟩ ≤ 2 * a i) :
  ∃ (s : Fin n → ℤ), 0 ≤ ∑ i, s i * a i ∧ ∑ i, s i * a i ≤ a 0 := sorry

end sign_choice_sum_leq_a1_l499_499327


namespace pair_can_be_combined_l499_499975

def is_combinable (a b : ℝ) : Prop :=
  ∃ k : ℝ, sqrt a = k * sqrt b

theorem pair_can_be_combined : 
  is_combinable 8 2 :=
by {
  use 2,
  calc
    sqrt 8 = sqrt (4 * 2) : by rw [←mul_assoc, show 4 = 2 * 2, by norm_num]
    ... = sqrt 4 * sqrt 2 : by rw sqrt_mul
    ... = 2 * sqrt 2 : by rw sqrt_four,
}

end pair_can_be_combined_l499_499975


namespace compare_logs_l499_499815

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l499_499815


namespace sequence_expression_l499_499393

noncomputable def a : ℕ → ℝ
| 1     => 2
| (n+1) => a n + Real.log (1 + 1 / n)

theorem sequence_expression (n : ℕ) (h : n > 0) : a n = 2 + Real.log n :=
by
  sorry

end sequence_expression_l499_499393


namespace problem1_problem2_l499_499048

noncomputable
def escalator_steps (a b x : ℕ) : Prop := 
  (24 = 2 * b * (x / (2 * b + a))) ∧ 
  (16 = (x / (a + b))) ∧ 
  (x = 48)

def catch_up_steps (a : ℕ) (m n : ℕ) : Prop := 
  ((n = (16 - m) / 6) ∨ (m = 16 - 6 * n)) ∧ 
  (2 * m + n = 16) ∧ 
  ((m = 3) ∧ (n = 13 / 6) ∧ (24 * m + 48 * n = 176))

theorem problem1 : ∃ x, ∀ a b, escalator_steps a b x :=
begin
  use 48,
  intros a b,
  split,
  { sorry },  -- Proof of equation derivation
  split,
  { sorry },
  { refl },   -- By substitution
end

theorem problem2: ∃ m n, ∀ a, catch_up_steps a m n :=
begin
  use 3,
  use 13 / 6,
  intros a,
  split,
  { left, 
    exact eq.symm (mul_div_cancel' _ (by norm_num)) },
  split,
  { exact eq.symm (add_sub_cancel') },
  split,
  { refl },
  { 
    calc
      24 * 3 + 48 * (13 / 6) = 72 + 104 : by norm_num
      ... = 176 : by norm_num
  }
end

end problem1_problem2_l499_499048


namespace chairs_required_l499_499383

def num_graduates : ℕ := 150
def num_parents_per_graduate : ℕ := 2
def percent_additional_family_members : ℝ := 0.40
def num_teachers : ℕ := 35
def num_admin_per_teachers_set : ℕ := 4
def teachers_per_teachers_set : ℕ := 3

/-- 
Prove that the number of chairs required for the graduation ceremony is 589
given the number of graduates, the number of parents, the percentage of
additional family members, the number of teachers, and the ratio of 
administrators to teacher sets.
-/
theorem chairs_required : 
  (num_graduates) + 
  (num_graduates * num_parents_per_graduate) + 
  (percent_additional_family_members * num_graduates).to_nat + 
  (num_teachers) + 
  (((num_teachers / teachers_per_teachers_set).to_nat) * num_admin_per_teachers_set) = 589 
:= 
sorry

end chairs_required_l499_499383


namespace train_cross_tree_time_l499_499576

-- Define the conditions
def length_of_train : ℝ := 1500
def time_to_pass_platform : ℝ := 160
def length_of_platform : ℝ := 500

-- Define the speed of the train
def speed_of_train : ℝ := (length_of_train + length_of_platform) / time_to_pass_platform

-- Define the time to cross a tree
def time_to_cross_tree : ℝ := length_of_train / speed_of_train

-- The theorem to prove
theorem train_cross_tree_time :
  time_to_cross_tree = 120 := by
  sorry

end train_cross_tree_time_l499_499576


namespace trajectory_and_slope_range_l499_499317

open Real

-- Definitions of conditions
def point (x y : ℝ) := (x, y)

def A := point 1 0
def O := point 0 0
def B := point 0 2

-- Moving point P, and points E, F on the line x = -1 satisfying the given conditions
def on_fixed_line_E (y : ℝ) : Prop := point -1 y
def on_fixed_line_F (y : ℝ) : Prop := point -1 y

-- Definitions of vectors and vector operations
def vec (p1 p2 : ℝ × ℝ) := (p2.1 - p1.1, p2.2 - p1.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
def perp (v1 v2 : ℝ × ℝ) := dot_product v1 v2 = 0
def parallel (v1 v2 : ℝ × ℝ) := ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- Conditions as functions
def AE_perp_AF (E F : ℝ × ℝ) : Prop :=
  perp (vec A E) (vec A F)

def EP_parallel_OA (P E : ℝ × ℝ) : Prop :=
  parallel (vec E P) (vec O A)

def FO_parallel_OP (F O P : ℝ × ℝ) : Prop :=
  parallel (vec F O) (vec O P)

-- Statement of the math proof problem in Lean
theorem trajectory_and_slope_range :
  (∀ (P E F : ℝ × ℝ), 
    on_fixed_line_E E.2 →
    on_fixed_line_F F.2 →
    AE_perp_AF E F →
    EP_parallel_OA P E →
    FO_parallel_OP F O P →
    P.fst * P.fst = P.snd * P.snd / 16) ∧
  (∀ (k : ℝ), 
    k ≠ 0 →
    (∀ M N : ℝ × ℝ,
      on_curve_C M →
      on_curve_C N →
      ∃ k, line_through_B_M_N k →
        -12 < k ∧ k < 0))
 :=
sorry

end trajectory_and_slope_range_l499_499317


namespace slope_AA_l499_499536

-- Define the points and conditions
variable (a b c d e f : ℝ)

-- Assumptions
#check (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
#check (a ≠ b ∧ c ≠ d ∧ e ≠ f)
#check (a+2 > 0 ∧ b > 0 ∧ c+2 > 0 ∧ d > 0 ∧ e+2 > 0 ∧ f > 0)

-- Main Statement
theorem slope_AA'_not_negative_one
    (H1: a > 0) (H2: b > 0) (H3: c > 0) (H4: d > 0)
    (H5: e > 0) (H6: f > 0) 
    (H7: a ≠ b) (H8: c ≠ d) (H9: e ≠ f)
    (H10: a + 2 > 0) (H11: c + 2 > 0) (H12: e + 2 > 0) : 
    (a ≠ b) → (c ≠ d) → (e ≠ f) → ¬( (a + 2 - b) / (b - a) = -1 ) :=
by
  sorry

end slope_AA_l499_499536


namespace largest_four_digit_negative_congruent_to_1_pmod_17_l499_499002

theorem largest_four_digit_negative_congruent_to_1_pmod_17 :
  ∃ n : ℤ, 17 * n + 1 < -1000 ∧ 17 * n + 1 ≥ -9999 ∧ 17 * n + 1 ≡ 1 [ZMOD 17] := 
sorry

end largest_four_digit_negative_congruent_to_1_pmod_17_l499_499002


namespace parallelogram_side_lengths_l499_499600

theorem parallelogram_side_lengths (x y : ℝ) (h₁ : 3 * x + 6 = 12) (h₂ : 10 * y - 3 = 15) : x + y = 3.8 :=
by
  sorry

end parallelogram_side_lengths_l499_499600


namespace geometric_series_sum_l499_499699

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499699


namespace pass_number_l499_499647

theorem pass_number (S I M P L E T A K : ℕ) (h_sequence : [S, I, M, P, L, E, T, A, S, K] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 9]) :
  (10^3 * P + 10^2 * A + 10^1 * S + 10^0 * S) = 3700 :=
by
  -- We assume the given conditions
  cases h_sequence with s_is_0 _,
  cases _ with i_is_1 _,
  cases _ with m_is_2 _,
  cases _ with p_is_3 _,
  cases _ with l_is_4 _,
  cases _ with e_is_5 _,
  cases _ with t_is_6 _,
  cases _ with a_is_7 _,
  cases _ with s_is_0_again _,
  cases _ with k_is_9 [],
  -- We have our digits matched to positions.
  -- Now, P = 3, A = 7, S = 0 as stated
  -- Thus, the 4-digit number for PASS is
  -- 3 * 1000 + 7 * 100 + 0 * 10 + 0 * 1 = 3700
  exact rfl

end pass_number_l499_499647


namespace geometric_series_sum_l499_499681

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499681


namespace log216_of_log8_l499_499803

theorem log216_of_log8 (x : ℝ) (h : log 8 (x - 3) = 1 / 3) : log 216 x = (1 / 3) * log 6 5 :=
by
  sorry

end log216_of_log8_l499_499803


namespace geometric_series_sum_l499_499705

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499705


namespace sum_of_consecutive_evens_is_162_l499_499520

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end sum_of_consecutive_evens_is_162_l499_499520


namespace problem_statement_l499_499800

theorem problem_statement (d : ℕ) (h1 : d > 0) (h2 : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2 * x^2 + 2 * x * y + 3 * y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by
  sorry

end problem_statement_l499_499800


namespace geometric_series_sum_l499_499679

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499679


namespace harmon_high_voting_l499_499268

theorem harmon_high_voting
  (U : Finset ℝ) -- Universe of students
  (A B : Finset ℝ) -- Sets of students favoring proposals
  (hU : U.card = 215)
  (hA : A.card = 170)
  (hB : B.card = 142)
  (hAcBc : (U \ (A ∪ B)).card = 38) :
  (A ∩ B).card = 135 :=
by {
  sorry
}

end harmon_high_voting_l499_499268


namespace arithmetic_problem_l499_499575

theorem arithmetic_problem : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end arithmetic_problem_l499_499575


namespace matt_worked_more_on_wednesday_l499_499453

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end matt_worked_more_on_wednesday_l499_499453


namespace gumball_machine_total_gumballs_l499_499594

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l499_499594


namespace gcd_sum_l499_499135

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499135


namespace math_problem_l499_499769

def f (x : ℝ) : ℝ := x ^ (-2)

theorem math_problem 
  (p : Prop := (∃ m : ℝ, f(2) = 2 ^ m ∧ 2 ^ m = 1/4 ∧ f(1/3) = 9))
  (q : Prop := ∀ (A B : ℝ), sin A = sin B → (A ≠ B)) :
  (p ∧ ¬ q) :=
by
  sorry

end math_problem_l499_499769


namespace john_average_increase_l499_499851

theorem john_average_increase :
  let initial_scores := [92, 85, 91]
  let fourth_score := 95
  let initial_avg := (initial_scores.sum / initial_scores.length : ℚ)
  let new_avg := ((initial_scores.sum + fourth_score) / (initial_scores.length + 1) : ℚ)
  new_avg - initial_avg = 1.42 := 
by 
  sorry

end john_average_increase_l499_499851


namespace gcd_sum_l499_499155

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499155


namespace geometric_sequence_common_ratio_l499_499382

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 = 36)
  (h2 : a 4 = 54)
  (h_pos : ∀ n, a n > 0) :
  ∃ q, q > 0 ∧ ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l499_499382


namespace sum_of_gcd_values_l499_499229

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499229


namespace largest_unpayable_soldo_l499_499895

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499895


namespace a4_minus_b4_l499_499313

theorem a4_minus_b4 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : a^4 - b^4 = -1 := by
  sorry

end a4_minus_b4_l499_499313


namespace min_lines_through_grid_points_l499_499836

theorem min_lines_through_grid_points (m n : ℕ) (h_m : m = 10) (h_n : n = 10) : 
  ∃ l : ℕ, l = 18 ∧ ∀ (lines : ℕ → ℤ × ℤ × ℤ), 
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → ∃ k, lines k = ((i : ℤ), (j : ℤ), (i : ℤ) + (j : ℤ)) ∨ lines k = ((i : ℤ), (j : ℤ), (i : ℤ) - (j : ℤ))) :=
begin
  sorry
end

end min_lines_through_grid_points_l499_499836


namespace connectivity_after_color_removal_l499_499525

-- Define the nodes in the complete graph
def nodes : ℕ := 50

-- Define edge colors
inductive color
  | C1 | C2 | C3

open color

-- We should have a complete graph definition here, but as a simplified version,
-- we consider only the conditions necessary for the proof problem
def complete_graph (n : ℕ) : Type := { edges : Type // (fin n × fin n) → color }

-- The main theorem statement.
theorem connectivity_after_color_removal (G : complete_graph nodes) :
  ∃ c : color, ∀ (u v : fin nodes), u ≠ v → 
  ∃ path : list (fin nodes), path.head = u ∧ path.ilast = v ∧
    ∀ i ∈ list.zip path (list.tail path), (G.edges i.1, G.edges i.2) = (G.edges i.1, G.edges i.2) :=
sorry

end connectivity_after_color_removal_l499_499525


namespace smallest_multiple_of_7_not_particular_l499_499280

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (λ d acc => acc + d) 0

def is_particular_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) ^ 2 = 0

theorem smallest_multiple_of_7_not_particular :
  ∃ n, n > 0 ∧ n % 7 = 0 ∧ ¬ is_particular_integer n ∧ ∀ m, m > 0 ∧ m % 7 = 0 ∧ ¬ is_particular_integer m → n ≤ m :=
  by
    use 7
    sorry

end smallest_multiple_of_7_not_particular_l499_499280


namespace sum_of_gcd_values_l499_499223

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499223


namespace diamond_eight_five_l499_499507

def diamond (a b : ℕ) : ℕ := (a + b) * ((a - b) * (a - b))

theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end diamond_eight_five_l499_499507


namespace regular_octagon_angle_EAH_l499_499387

-- Define the regular octagon and the required angle calculation
theorem regular_octagon_angle_EAH
  (ABCDEFGH : Type) (regular_octagon : ∀ (x y : ABCDEFGH), (x ≠ y) → Prop)
  (vertex_ordered : ∀ (A B C D E F G H : ABCDEFGH), regular_octagon A B → regular_octagon B C → 
                    regular_octagon C D → regular_octagon D E → regular_octagon E F → 
                    regular_octagon F G → regular_octagon G H → regular_octagon H A) 
  : ∠ EAH = 45 := 
sorry

end regular_octagon_angle_EAH_l499_499387


namespace piggy_bank_savings_l499_499478

theorem piggy_bank_savings :
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  final_amount = 104 :=
by
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  show final_amount = 104 from sorry

end piggy_bank_savings_l499_499478


namespace number_of_roses_l499_499618

theorem number_of_roses (v c n : ℕ) (h_v : v = 6) (h_c : c = 7) (h_n : n = 9) :
  (n * v) - c = 47 :=
by
  rw [h_v, h_c, h_n]
  norm_num
  done

end number_of_roses_l499_499618


namespace part_one_part_two_part_three_l499_499782

def f (x : ℝ) : ℝ := sin x * (cos x - sqrt 3 * sin x)

theorem part_one : is_periodic f π :=
sorry

theorem part_two (a b : ℝ) (h₀ : 0 < a) (h₁ : a < π / 2) :
  ∃ (a b : ℝ), (sin (2 * (x + a)) - b = f x) ∧ (a = π / 6) ∧ (b = sqrt 3 / 2) ∧ (a * b = sqrt 3 / 12 * π) :=
sorry

theorem part_three : 
  set.image f (set.Icc 0 (π / 2)) = set.Icc (- sqrt 3) (1 - sqrt 3 / 2) :=
sorry

end part_one_part_two_part_three_l499_499782


namespace sum_of_gcd_values_l499_499225

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499225


namespace pinocchio_cannot_pay_exactly_l499_499887

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499887


namespace sum_of_given_infinite_geometric_series_l499_499714

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499714


namespace part_a_part_b_part_c_l499_499437

def f (x : ℝ) := x^2
def g (x : ℝ) := 3 * x - 8
def h (r : ℝ) (x : ℝ) := 3 * x - r

theorem part_a :
  f 2 = 4 ∧ g (f 2) = 4 :=
by {
  sorry
}

theorem part_b :
  ∀ x : ℝ, f (g x) = g (f x) → (x = 2 ∨ x = 6) :=
by {
  sorry
}

theorem part_c :
  ∀ r : ℝ, f (h r 2) = h r (f 2) → (r = 3 ∨ r = 8) :=
by {
  sorry
}

end part_a_part_b_part_c_l499_499437


namespace true_propositions_count_l499_499342

theorem true_propositions_count {X : Type} 
  (hx1 : ∀ σ > 0, ∃ X, X ∼ Normal 1 σ ∧ P(0 < X ∧ X < 1) = 0.4 ∧ P(0 < X ∧ X < 2) = 0.8)
  (hx2 : ∀ a b : ℝ, (0 < a * b ∧ a * b < 1) → ¬ (∀ a b : ℝ, b < 1/a))
  (hx3 : ∀ (p : Prop), (∀ (x1 x2 : ℝ), (f x2 - f x1) * (x2 - x1) ≥ 0) → (¬ ∀ (x1 x2 : ℝ), (f x2 - f x1) * (x2 - x1) < 0))
  (hx4 : ∀ (A B C : ℝ) (ABC_is_triangle : A + B + C = π ),
    (A + C = 2*B) → (∀ (B = π/3 → sin C = (sqrt 3 * cos A + sin A) * cos B))) : 
  (∃ p1 p2 p3 p4 : Prop, 
    p1 ∧ p2 ∧ p3 ∧ p4 ∧ 
    (p1 = hx1) ∧ 
    (p2 = hx2) ∧ 
    (p3 = hx3) ∧ 
    (p4 = hx4) ∧ 
    2) :=
  sorry

end true_propositions_count_l499_499342


namespace gcd_sum_divisors_eq_12_l499_499175

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499175


namespace gcd_5n_plus_6_n_sum_l499_499102

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499102


namespace gcd_sum_l499_499091

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499091


namespace gcd_sum_l499_499096

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499096


namespace gcd_sum_l499_499129

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499129


namespace buffered_decreasing_interval_l499_499369

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := (x / 2) + 1 / x - 2

theorem buffered_decreasing_interval : ∀ x ∈ set.Icc (real.sqrt 2) 2, 
    (∃ y ∈ set.Icc (real.sqrt 2) 2, f x ≤ f y) ∧ 
    (∃ y ∈ set.Icc (real.sqrt 2) 2, g y ≥ g x) :=
sorry

end buffered_decreasing_interval_l499_499369


namespace largest_divisor_of_n_l499_499982

theorem largest_divisor_of_n (n : ℕ) (h_pos: n > 0) (h_div: 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l499_499982


namespace andrew_total_donation_l499_499625

/-
Problem statement:
Andrew started donating 7k to an organization on his 11th birthday. Yesterday, Andrew turned 29.
Verify that the total amount Andrew has donated is 126k.
-/

theorem andrew_total_donation 
  (annual_donation : ℕ := 7000) 
  (start_age : ℕ := 11) 
  (current_age : ℕ := 29) 
  (years_donating : ℕ := current_age - start_age) 
  (total_donated : ℕ := annual_donation * years_donating) :
  total_donated = 126000 := 
by 
  sorry

end andrew_total_donation_l499_499625


namespace sum_of_consecutive_negatives_l499_499511

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l499_499511


namespace fractions_sum_ge_one_l499_499758

variable {a b c : ℝ}

theorem fractions_sum_ge_one (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by {
  sorry,
}

end fractions_sum_ge_one_l499_499758


namespace zero_in_interval_l499_499023

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x0, f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
by
  have h_cont : Continuous f := sorry -- f is continuous (can be proven using the continuity of log and linear functions)
  have h_eval1 : f 2 < 0 := sorry -- f(2) = ln(2) - 6 + 4 < 0
  have h_eval2 : f 3 > 0 := sorry -- f(3) = ln(3) - 6 + 6 > 0
  -- By the Intermediate Value Theorem, since f is continuous and changes signs between (2, 3), there exists a zero x0 in (2, 3).
  exact sorry

end zero_in_interval_l499_499023


namespace gcd_sum_l499_499164

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499164


namespace geometric_series_sum_l499_499706

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499706


namespace gcd_sum_5n_plus_6_n_l499_499143

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499143


namespace gcd_sum_l499_499138

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499138


namespace number_of_people_l499_499036

variable (P M : ℕ)

-- Conditions
def cond1 : Prop := (500 = P * M)
def cond2 : Prop := (500 = (P + 5) * (M - 2))

-- Goal
theorem number_of_people (h1 : cond1 P M) (h2 : cond2 P M) : P = 33 :=
sorry

end number_of_people_l499_499036


namespace sum_of_given_infinite_geometric_series_l499_499708

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499708


namespace problem1_solution_set_problem2_inequality_l499_499753

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|
noncomputable def g (x : ℝ) : ℝ := |x + (3 / 2)| + |x - (3 / 2)|

theorem problem1_solution_set :
  {x : ℝ | f x ≤ x + 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
sorry

theorem problem2_inequality (a : ℝ) (ha : a ≠ 0) :
  (|a + 1| - |2a - 1|) / |a| ≤ g 0 :=
sorry

end problem1_solution_set_problem2_inequality_l499_499753


namespace correct_graph_is_A_l499_499499

def g : ℝ → ℝ
| x := if -3 ≤ x ∧ x ≤ 0 then -2 - x
       else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x - 2) ^ 2) - 2
       else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
       else 0

def transformed_g (x : ℝ) : ℝ :=
  -g (x - 3)

-- Graph A scenario:
def graph_a (x : ℝ) : ℝ := transformed_g x

theorem correct_graph_is_A :
  ∀ x, transformed_g x = graph_a x := by
  sorry

end correct_graph_is_A_l499_499499


namespace distribute_pictures_l499_499529

/-
Tiffany uploaded 34 pictures from her phone, 55 from her camera,
and 12 from her tablet to Facebook. If she sorted the pics into 7 different albums
with the same amount of pics in each album, how many pictures were in each of the albums?
-/

theorem distribute_pictures :
  let phone_pics := 34
  let camera_pics := 55
  let tablet_pics := 12
  let total_pics := phone_pics + camera_pics + tablet_pics
  let albums := 7
  ∃ k r, (total_pics = k * albums + r) ∧ (r < albums) := by
  sorry

end distribute_pictures_l499_499529


namespace shelly_thread_length_l499_499910

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end shelly_thread_length_l499_499910


namespace gcd_sum_5n_6_n_eq_12_l499_499183

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499183


namespace infinite_geometric_series_sum_l499_499694

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499694


namespace gcd_sum_divisors_eq_12_l499_499170

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499170


namespace three_op_six_l499_499652

-- Define the new operation @.
def op (a b : ℕ) : ℕ := (a * a * b) / (a + b)

-- The theorem to prove that the value of 3 @ 6 is 6.
theorem three_op_six : op 3 6 = 6 := by 
  sorry

end three_op_six_l499_499652


namespace oil_remaining_in_tank_l499_499930

/- Definitions for the problem conditions -/
def tankCapacity : Nat := 32
def totalOilPurchased : Nat := 728

/- Theorem statement -/
theorem oil_remaining_in_tank : totalOilPurchased % tankCapacity = 24 := by
  sorry

end oil_remaining_in_tank_l499_499930


namespace factorial_fraction_simplification_l499_499632

-- Define necessary factorial function
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Define the problem
theorem factorial_fraction_simplification :
  (4 * fact 6 + 20 * fact 5) / fact 7 = 22 / 21 := by
  sorry

end factorial_fraction_simplification_l499_499632


namespace sequence_difference_l499_499394

theorem sequence_difference :
  let a : ℕ+ → ℕ := λ n, ∑ i in finset.range n, (i + 1)^(i + 1)
  let b : ℕ+ → ℝ := λ n, Real.cos (a n * Real.pi)
  b 2008 - b 2009 = 2 :=
by
  let a : ℕ+ → ℕ := λ n, ∑ i in finset.range n, (i + 1)^(i + 1)
  let b : ℕ+ → ℝ := λ n, Real.cos (a n * Real.pi)
  -- Skip the proof
  sorry

end sequence_difference_l499_499394


namespace smallest_area_right_triangle_l499_499973

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 10): 
  ∃ (A : ℕ), A = 35 :=
  by
    have hab := 1/2 * a * b
    sorry

-- Note: "sorry" is used as a placeholder for the proof.

end smallest_area_right_triangle_l499_499973


namespace two_people_paint_time_l499_499745

theorem two_people_paint_time (h : 5 * 7 = 35) :
  ∃ t : ℝ, 2 * t = 35 ∧ t = 17.5 := 
sorry

end two_people_paint_time_l499_499745


namespace initial_salary_l499_499413

variable (S : ℝ)

theorem initial_salary :
  12 * S + 24 * 1.3 * S = 259200 → 
  S = 6000 := 
by
  intro h1
  have h2 : 12 * S + 31.2 * S = 259200 := by
    calc
      12 * S + 24 * 1.3 * S = 12 * S + 31.2 * S : by ring
  have h3 : 43.2 * S = 259200 := by
    exact h2
  have h4 : S = 259200 / 43.2 := by
    exact eq_div_of_mul_eq (by norm_num) h3
  exact h4.symm ▸ eq.refl _

end initial_salary_l499_499413


namespace razorback_tshirt_sales_l499_499926

theorem razorback_tshirt_sales 
  (price_per_tshirt : ℕ) (total_money_made : ℕ)
  (h1 : price_per_tshirt = 16) (h2 : total_money_made = 720) :
  total_money_made / price_per_tshirt = 45 :=
by
  sorry

end razorback_tshirt_sales_l499_499926


namespace subset_not_divisible_l499_499314

theorem subset_not_divisible (S : Finset ℤ) (hS : S.card = 10000) (h : ∀ x ∈ S, ¬ (47 ∣ x)):
  ∃ (Y : Finset ℤ), Y ⊆ S ∧ Y.card = 2015 ∧ ∀ a b c d e ∈ Y, ¬ (47 ∣ (a - b + c - d + e)) :=
sorry

end subset_not_divisible_l499_499314


namespace rationalize_denominator_l499_499482

theorem rationalize_denominator : 
  let A := -13 
  let B := -9
  let C := 3
  let D := 2
  let E := 165
  let F := 51
  A + B + C + D + E + F = 199 := by
sorry

end rationalize_denominator_l499_499482


namespace fraction_is_seventh_l499_499274

-- Definition of the condition on x being greater by a certain percentage
def x_greater := 1125.0000000000002 / 100

-- Definition of x in terms of the condition
def x := (4 / 7) * (1 + x_greater)

-- Definition of the fraction f
def f := 1 / x

-- Lean theorem statement to prove the fraction is 1/7
theorem fraction_is_seventh (x_greater: ℝ) : (1 / ((4 / 7) * (1 + x_greater))) = 1 / 7 :=
by
  sorry

end fraction_is_seventh_l499_499274


namespace shelly_thread_length_l499_499909

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end shelly_thread_length_l499_499909


namespace gold_copper_alloy_ratio_l499_499981

theorem gold_copper_alloy_ratio 
  (G C : ℝ) 
  (h_gold : G / weight_of_water = 19) 
  (h_copper : C / weight_of_water = 9)
  (weight_of_alloy : (G + C) / weight_of_water = 17) :
  G / C = 4 :=
sorry

end gold_copper_alloy_ratio_l499_499981


namespace mass_percentage_of_Cl_in_NaClO_l499_499737

noncomputable def molarMassNa : ℝ := 22.99
noncomputable def molarMassCl : ℝ := 35.45
noncomputable def molarMassO : ℝ := 16.00

noncomputable def molarMassNaClO : ℝ := molarMassNa + molarMassCl + molarMassO

theorem mass_percentage_of_Cl_in_NaClO : 
  (molarMassCl / molarMassNaClO) * 100 = 47.61 :=
by 
  sorry

end mass_percentage_of_Cl_in_NaClO_l499_499737


namespace tetrahedron_signs_a_tetrahedron_signs_b_l499_499977

open Real Vector

-- Part a
theorem tetrahedron_signs_a (α β γ δ : ℝ) (A B C D O : Vector ℝ 3) 
    (H : O ∈ tetrahedron_span A B C D) 
    (V_eq_zero : α • (O - A) + β • (O - B) + γ • (O - C) + δ • (O - D) = 0) :
  same_sign α β γ δ :=
sorry

-- Part b
theorem tetrahedron_signs_b (α β γ δ : ℝ) (A1 B1 C1 D1 O : Vector ℝ 3) 
    (H : O ∈ tetrahedron_span A1 B1 C1 D1) 
    (V_eq_zero : α • (O - A1) + β • (O - B1) + γ • (O - C1) + δ • (O - D1) = 0) 
    (P : perpendicular_set O [A1, B1, C1, D1]) :
  same_sign α β γ δ :=
sorry

end tetrahedron_signs_a_tetrahedron_signs_b_l499_499977


namespace village_duration_l499_499056

theorem village_duration (vampire_drain : ℕ) (werewolf_eat : ℕ) (village_population : ℕ)
  (hv : vampire_drain = 3) (hw : werewolf_eat = 5) (hp : village_population = 72) :
  village_population / (vampire_drain + werewolf_eat) = 9 :=
by
  sorry

end village_duration_l499_499056


namespace sum_of_gcd_values_l499_499234

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499234


namespace gcd_5n_plus_6_n_sum_l499_499100

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499100


namespace sum_of_given_infinite_geometric_series_l499_499712

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499712


namespace sum_gcd_possible_values_eq_twelve_l499_499239

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499239


namespace fractional_expression_correct_l499_499974

-- Define what it means to be a fractional expression
def is_fractional_expression (num denom) : Prop :=
  ∃ x : ℤ, denom = x

-- Define the expressions
def expr_A := (a - b) / 2
def expr_B := (5 + y) / π
def expr_C := (x + 3) / x
def expr_D := 1 + x

-- The problem definition
theorem fractional_expression_correct :
  is_fractional_expression (x + 3) x ∧ 
  ¬is_fractional_expression (a - b) 2 ∧ 
  ¬is_fractional_expression (5 + y) π ∧ 
  ¬is_fractional_expression (1 + x) 1 :=
by 
  sorry

end fractional_expression_correct_l499_499974


namespace sin_alpha_plus_4pi_over_3_eq_neg_sqrt3_div_4_l499_499356

variable (α : ℝ)

def vector_a : ℝ × ℝ := (Real.sin (α + π / 6), 1)
def vector_b : ℝ × ℝ := (4, 4 * Real.cos α - Real.sqrt 3)

theorem sin_alpha_plus_4pi_over_3_eq_neg_sqrt3_div_4 
  (h : vector_a α.1 * vector_b α.1 + vector_a α.2 * vector_b α.2 = 0) : 
  Real.sin (α + 4 * π / 3) = -1 / 4 :=
by
  sorry

end sin_alpha_plus_4pi_over_3_eq_neg_sqrt3_div_4_l499_499356


namespace total_cost_l499_499852

-- Definitions:
def amount_beef : ℕ := 1000
def price_per_pound_beef : ℕ := 8
def amount_chicken := amount_beef * 2
def price_per_pound_chicken : ℕ := 3

-- Theorem: The total cost of beef and chicken is $14000.
theorem total_cost : (amount_beef * price_per_pound_beef) + (amount_chicken * price_per_pound_chicken) = 14000 :=
by
  sorry

end total_cost_l499_499852


namespace symmetric_scanning_codes_count_l499_499045

theorem symmetric_scanning_codes_count :
  let grid_size := 5
  let total_squares := grid_size * grid_size
  let symmetry_classes := 5 -- Derived from classification in the solution
  let possible_combinations := 2 ^ symmetry_classes
  let invalid_combinations := 2 -- All black or all white grid
  total_squares = 25 
  ∧ (possible_combinations - invalid_combinations) = 30 :=
by sorry

end symmetric_scanning_codes_count_l499_499045


namespace geometric_series_sum_l499_499676

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499676


namespace roots_lost_extraneous_roots_l499_499541

noncomputable def f1 (x : ℝ) := Real.arcsin x
noncomputable def g1 (x : ℝ) := 2 * Real.arcsin (x / Real.sqrt 2)
noncomputable def f2 (x : ℝ) := x
noncomputable def g2 (x : ℝ) := 2 * x

theorem roots_lost :
  ∃ x : ℝ, f1 x = g1 x ∧ ¬ ∃ y : ℝ, Real.tan (f1 y) = Real.tan (g1 y) :=
sorry

theorem extraneous_roots :
  ∃ x : ℝ, ¬ f2 x = g2 x ∧ ∃ y : ℝ, Real.tan (f2 y) = Real.tan (g2 y) :=
sorry

end roots_lost_extraneous_roots_l499_499541


namespace det_projection_matrix_zero_l499_499424

noncomputable def projection_matrix (v : ℕ → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let normalize : ℝ := real.sqrt (v 0 ^ 2 + v 1 ^ 2)
  (λ i j, (v i / normalize) * (v j / normalize))

theorem det_projection_matrix_zero :
  let v := ![3, 5]
  let Q := projection_matrix v
  Matrix.det Q = 0 :=
by
  -- Proof goes here
  sorry

end det_projection_matrix_zero_l499_499424


namespace geometric_series_sum_l499_499680

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499680


namespace black_area_fraction_after_four_changes_l499_499283

/-- 
Problem: Prove that after four changes, the fractional part of the original black area 
remaining black in an equilateral triangle is 81/256, given that each change splits the 
triangle into 4 smaller congruent equilateral triangles, and one of those turns white.
-/

theorem black_area_fraction_after_four_changes :
  (3 / 4) ^ 4 = 81 / 256 := sorry

end black_area_fraction_after_four_changes_l499_499283


namespace coffee_drunk_on_Thursday_l499_499506

noncomputable def k : ℝ := 30  -- From the calculation on Wednesday
def coffee_drunk (h p : ℕ) : ℝ := k / (h + p)

theorem coffee_drunk_on_Thursday : coffee_drunk 5 3 = 15 / 4 :=
by
  unfold coffee_drunk
  simp [k]
  sorry

end coffee_drunk_on_Thursday_l499_499506


namespace matt_worked_more_on_wednesday_l499_499454

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end matt_worked_more_on_wednesday_l499_499454


namespace infinite_geometric_series_sum_l499_499729

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499729


namespace points_on_parabola_l499_499391

theorem points_on_parabola :
  (∀ (x y : ℝ), (y^2 = x ∧ (x - 11)^2 + (y - 1)^2 = 25) → y = (1/2) * x^2 - (21/2) * x + (97/2)) :=
begin
  sorry
end

end points_on_parabola_l499_499391


namespace relationship4_is_mapping_l499_499793

def M : Set ℤ := {-1, 1, 2, 4}
def N : Set ℤ := {0, 1, 2}
def relation4 (x : ℤ) : ℤ := Int.ofNat (Int.log2 (Int.natAbs x))

theorem relationship4_is_mapping : ∀ x ∈ M, relation4 x ∈ N := by
  sorry

end relationship4_is_mapping_l499_499793


namespace find_A_plus_B_l499_499429

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l499_499429


namespace sum_gcd_values_l499_499263

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499263


namespace gcd_sum_5n_plus_6_l499_499204

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499204


namespace gcd_values_sum_l499_499075

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499075


namespace train_pass_tree_time_l499_499984

-- Define the given conditions in Lean
def train_length_meters : ℕ := 150
def train_speed_kmph : ℕ := 54

-- Conversion factor from km/hr to m/s
def kmph_to_mps (speed_kmph : ℕ) : ℝ := (speed_kmph : ℝ) * (1000 : ℝ) / (3600 : ℝ)

-- Speed of the train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Prove that the time required to pass the oak tree is 10 seconds
theorem train_pass_tree_time : train_length_meters / train_speed_mps = 10 := by
  sorry

end train_pass_tree_time_l499_499984


namespace range_of_a_l499_499785

theorem range_of_a 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x = Real.exp (|x - a|)) 
    (increasing_on_interval : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) :
    a ≤ 1 :=
sorry

end range_of_a_l499_499785


namespace find_a_equidistant_l499_499844

theorem find_a_equidistant :
  ∀ a : ℝ, (abs (a - 2) = abs (6 - 2 * a)) →
    (a = 8 / 3 ∨ a = 4) :=
by
  intro a h
  sorry

end find_a_equidistant_l499_499844


namespace only_unique_point_represents_same_location_l499_499917

theorem only_unique_point_represents_same_location
  (ABCD A'B'C'D': set (ℝ × ℝ))
  (hABCD: is_square ABCD)
  (hA'B'C'D': is_square A'B'C'D')
  (same_region: ∃ f: (ℝ × ℝ) → (ℝ × ℝ), bijective f ∧ ∀ p ∈ ABCD, f p ∈ A'B'C'D' ∧ ∀ q ∈ A'B'C'D', f.symm q ∈ ABCD):
  ∃! (O: ℝ × ℝ), O ∈ ABCD ∧ ∀ (O': ℝ × ℝ), O' ∈ A'B'C'D' → ∃ θ k, k > 0 ∧ similarity_transformation O θ k O' :=
begin
  sorry
end

end only_unique_point_represents_same_location_l499_499917


namespace position_of_TENSOR_in_words_l499_499967

/-- 
Using the letters B, E, N, R, S, and T, we can form six-letter "words".
If these "words" are arranged in alphabetical order, the position of 
the "word" TENSOR among all permutations should be 605.
--/
theorem position_of_TENSOR_in_words :
  let letters := ['B', 'E', 'N', 'R', 'S', 'T'] in
  (∀ w : String, w ∈ List.permutations letters) → 
  List.sorted (<=) (List.permutations letters) →
  List.index_of "TENSOR" (List.permutations letters) + 1 = 605 :=
by
  sorry

end position_of_TENSOR_in_words_l499_499967


namespace find_smallest_x_l499_499801

theorem find_smallest_x :
  ∃ x y : ℕ, 0 < y ∧ 0.8 = y / (196 + x) ∧ x = 49 :=
by {
  sorry
}

end find_smallest_x_l499_499801


namespace angelina_speed_l499_499068

variable (v : ℝ)              -- Speed from home to the grocery
variable (t1 t2 : ℝ)          -- Travel times

-- Define the conditions
def time_home_to_grocery (v : ℝ) : ℝ := 1200 / v
def time_grocery_to_gym (v : ℝ) : ℝ := 480 / (2 * v)
def time_difference (t1 t2 : ℝ) : ℝ := t1 - t2

-- Define the theorem
theorem angelina_speed :
  ∃ (v : ℝ),
  (time_difference (time_home_to_grocery v) (time_grocery_to_gym v) = 40) → 
  2 * v = 48 :=
by
  sorry

end angelina_speed_l499_499068


namespace number_of_roses_l499_499616

def vase_capacity : ℕ := 6
def carnations : ℕ := 7
def vases : ℕ := 9

theorem number_of_roses : ∃ (roses : ℕ), vase_capacity * vases - carnations = roses ∧ roses = 47 :=
by 
  use 47
  simp [vase_capacity, carnations, vases]
  sorry

end number_of_roses_l499_499616


namespace platform_length_l499_499613

theorem platform_length (train_length : ℕ) (train_speed_kmph : ℕ) (crossing_time : ℕ) 
    (h_train_length : train_length = 150) 
    (h_train_speed_kmph : train_speed_kmph = 75) 
    (h_crossing_time : crossing_time = 24) : 
    (platform_length : ℕ) (h_platform_length : platform_length = 350) :=
sorry

end platform_length_l499_499613


namespace rate_percent_simple_interest_l499_499017

theorem rate_percent_simple_interest (SI P T : ℝ) (h1 : SI = 192) (h2 : P = 800) (h3 : T = 4) :
  (R : ℝ) = 6 :=
by
  -- simple interest formula: SI = P * R * T / 100
  let R := (SI * 100) / (P * T)
  have : R = 6, from sorry,
  exact this

end rate_percent_simple_interest_l499_499017


namespace prime_factors_calculation_l499_499870

noncomputable def x := sorry -- Placeholder for actual x value
noncomputable def y := sorry -- Placeholder for actual y value

def m := (multiset.card (multiset.filter nat.prime (multiset.factor (int.natAbs x))))
def n := (multiset.card (multiset.filter nat.prime (multiset.factor (int.natAbs y))))

#eval 4 * m + 3 * n

theorem prime_factors_calculation (x y : ℕ) (hx : real.log10 x + 3 * real.log10 (nat.gcd x y) = 120) 
  (hy : real.log10 y + 3 * real.log10 (nat.lcm x y) = 1050) 
  (hx_pos : 0 < x) (hy_pos : 0 < y) :
  4 * (multiset.card (multiset.filter nat.prime (multiset.factor (int.natAbs x)))) + 
  3 * (multiset.card (multiset.filter nat.prime (multiset.factor (int.natAbs y)))) = 1815 := 
sorry

end prime_factors_calculation_l499_499870


namespace order_of_variables_l499_499805

variable (a b c d : ℝ)

theorem order_of_variables (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : c > a ∧ a > b ∧ b > d :=
by
  sorry

end order_of_variables_l499_499805


namespace triangle_count_l499_499483

theorem triangle_count :
  let x1_values := finset.range (2 - (-8) + 1).image (λ x, -8 + x),
      y1_values := finset.range (9 - 4 + 1).image (λ y, 4 + y),
      countA := x1_values.card * y1_values.card,
      countB_for_A := (y1_values.card - 1),
      countC_for_A := (x1_values.card - 1),
      countTotal := countA * countB_for_A * countC_for_A
  in countTotal = 3300 := by
  let x1_values := finset.range (2 - (-8) + 1).image (λ x, -8 + x)
  let y1_values := finset.range (9 - 4 + 1).image (λ y, 4 + y)
  let countA := x1_values.card * y1_values.card
  let countB_for_A := (y1_values.card - 1)
  let countC_for_A := (x1_values.card - 1)
  let countTotal := countA * countB_for_A * countC_for_A
  have h : countTotal = 3300 := sorry
  exact h

end triangle_count_l499_499483


namespace sufficient_but_not_necessary_l499_499750

theorem sufficient_but_not_necessary (a b c : ℝ) (h1 : a > b) (h2 : c > 0) : ac > bc ∧ (∀ d : ℝ, (ad > bd) → (a > b ∧ d > 0) ∨ (a < b ∧ d < 0)) :=
by
  split
  -- part 1: proving a > b and c > 0 implies ac > bc
  -- part 2: showing ac > bc does not always imply a > b and c > 0


end sufficient_but_not_necessary_l499_499750


namespace trapezoid_max_area_top_base_length_l499_499839

theorem trapezoid_max_area_top_base_length (r : ℝ) : 
  ∃ (x : ℝ), 
  (height := sqrt (r^2 - x^2)) ∧ 
  (S := (r + x) * sqrt (r^2 - x^2)) ∧ 
  (∀ y, y ≠ x → (height' := sqrt (r^2 - y^2)) ∧ (S' := (r + y) * sqrt (r^2 - y^2)) ∧ S' ≤ S) ∧ 
  2 * x = r :=
sorry

end trapezoid_max_area_top_base_length_l499_499839


namespace ratio_of_radii_ratio_of_areas_l499_499540

theorem ratio_of_radii (r1 r2 : ℝ) (h1: 0 < r1 ∧ 0 < r2)
    (h2: 60 * (2 * Mathlib.pi * r1) / 360 = 48 * (2 * Mathlib.pi * r2) / 360) : r1 / r2 = 4 / 5 :=
by
  sorry

theorem ratio_of_areas (r1 r2 : ℝ) (h1: r1 / r2 = 4 / 5) : (Mathlib.pi * r1 ^ 2) / (Mathlib.pi * r2 ^ 2) = 16 / 25 :=
by
  sorry

end ratio_of_radii_ratio_of_areas_l499_499540


namespace max_distance_origin_fourth_vertex_l499_499920

variable {z : ℂ} (hz : abs z = 1)

theorem max_distance_origin_fourth_vertex :
  let w := (1 + 2 * Complex.i) * z + 3 * Complex.conj z in
  ∃ (S : ℂ), abs S = 2 * Real.sqrt 5 :=
sorry

end max_distance_origin_fourth_vertex_l499_499920


namespace line_equation_not_parallel_y_axis_l499_499470

theorem line_equation_not_parallel_y_axis (a b c : ℝ) (hb : b ≠ 0) :
  ∃ k l : ℝ, (∀ x y : ℝ, (a * x + b * y + c = 0) → (y = k * x + l)) ∧ (k = -a / b) ∧ (l = -c / b) :=
begin
  sorry
end

end line_equation_not_parallel_y_axis_l499_499470


namespace prob_three_digit_multiple_of_5_l499_499052

-- Define the range of three-digit numbers
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define the property of being a multiple of 5
def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Define the event of selecting a three-digit multiple of 5
def event_A (n : ℕ) : Prop :=
  is_three_digit n ∧ is_multiple_of_5 n

-- The main theorem to prove that the probability is 1/5
theorem prob_three_digit_multiple_of_5 :
  (Fintype.card { x // is_three_digit x ∧ is_multiple_of_5 x } : ℚ) / 
  (Fintype.card { x // is_three_digit x } : ℚ) = 1 / 5 :=
by
  sorry

end prob_three_digit_multiple_of_5_l499_499052


namespace total_signals_formed_l499_499958

/-- We have flags of four different colors: red, yellow, blue, and white. Each signal is formed 
by hanging one or more flags on a flagpole. Prove that the total number of different signals 
that can be formed is 64. -/
theorem total_signals_formed : 
  let red := 1
  let yellow := 1
  let blue := 1
  let white := 1
  let flags := red + yellow + blue + white in
  (flags.choose 1) + (flags.choose 2) + (flags.choose 3) + (flags.choose 4) = 64 :=
sorry

end total_signals_formed_l499_499958


namespace gcd_sum_l499_499126

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499126


namespace find_A_l499_499490

namespace PolynomialDecomposition

theorem find_A (x A B C : ℝ)
  (h : (x^3 + 2 * x^2 - 17 * x - 30)⁻¹ = A / (x - 5) + B / (x + 2) + C / ((x + 2)^2)) :
  A = 1 / 49 :=
by sorry

end PolynomialDecomposition

end find_A_l499_499490


namespace k_plus_1_prime_l499_499414

open Set

variable {X : Type*}
variable {F : Finset (Finset X)} {k : ℕ}

-- A structured hypothesis capturing the conditions
def valid_configuration (X : Set X) (F : Set (Set X)) (k : ℕ) : Prop :=
  X.card = 2 * k ∧ 
  (∀ s ∈ F, s.card = k) ∧
  (∀ y : Set X, y.card = k - 1 → ∃! z ∈ F, y ⊆ z)

theorem k_plus_1_prime
  (X : Finset X)
  (hX : valid_configuration X F k) :
  Nat.Prime (k + 1) :=
sorry

end k_plus_1_prime_l499_499414


namespace gumball_machine_total_l499_499589

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l499_499589


namespace dryer_sheets_in_box_l499_499916

/-
Sophie does 4 loads of laundry a week and uses 1 dryer sheet per load.
A box of dryer sheets costs $5.50.
On her birthday, she was given wool dryer balls to use instead of dryer sheets.
She saves $11 in a year not buying dryer sheets.
Prove that the number of dryer sheets in a box is 104.
-/

def loads_per_week : ℕ := 4
def dryer_sheets_per_load : ℕ := 1
def box_cost : ℝ := 5.50
def yearly_savings : ℝ := 11.00

theorem dryer_sheets_in_box :
  let weekly_sheets := loads_per_week * dryer_sheets_per_load,
      yearly_sheets := weekly_sheets * 52,
      boxes_per_year := yearly_savings / box_cost
  in yearly_sheets / boxes_per_year = 104 := by
  sorry

end dryer_sheets_in_box_l499_499916


namespace gcd_sum_l499_499163

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499163


namespace ceil_e_add_pi_l499_499665

theorem ceil_e_add_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by
  sorry

end ceil_e_add_pi_l499_499665


namespace marble_189_is_gray_l499_499612

def marble_color (n : ℕ) : String :=
  let cycle_length := 14
  let gray_thres := 5
  let white_thres := 9
  let black_thres := 12
  let position := (n - 1) % cycle_length + 1
  if position ≤ gray_thres then "gray"
  else if position ≤ white_thres then "white"
  else if position ≤ black_thres then "black"
  else "blue"

theorem marble_189_is_gray : marble_color 189 = "gray" :=
by {
  -- We assume the necessary definitions and steps discussed above.
  sorry
}

end marble_189_is_gray_l499_499612


namespace sum_prime_product_series_l499_499986

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, ...]

def P (n : ℕ) : ℕ := primes.get ⟨n, sorry⟩ -- getting nth prime

theorem sum_prime_product_series :
  (∑ k in Finset.range (ℕ), ∏ i in Finset.range (k + 1), (P i - 1) / P (i + 1)) = 1 := sorry

end sum_prime_product_series_l499_499986


namespace june_earnings_l499_499412

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l499_499412


namespace chocolate_pile_division_l499_499407

/-- Jordan has 35/4 pounds of chocolate and wants to divide it into 5 piles of equal weight.
If he gives two of these piles to his friend Shaina, how many pounds of chocolate will Shaina get? -/
theorem chocolate_pile_division : 
  let total_chocolate := (35 : ℚ) / 4,
      number_of_piles := 5,
      weight_per_pile := total_chocolate / number_of_piles,
      piles_given_to_shaina := 2 in
  piles_given_to_shaina * weight_per_pile = 7 / 2 :=
by 
  -- Lean proof steps go here
  sorry

end chocolate_pile_division_l499_499407


namespace unique_triangle_x_value_l499_499774

-- Definitions
variables {a b c x : ℝ}
variables {A B C : ℝ}
variables (T : Type*) [Triangle T]

-- Conditions
def given_conditions (a A b : ℝ) := 
  a = 2 ∧ 
  A = π / 4 ∧ 
  b = x ∧
  (∃ (T : Triangle), is_unique (length T = a) ∧ is_unique (angle T = A))

-- Proof statement
theorem unique_triangle_x_value (h : given_conditions a A b) : x = 2 * sqrt 2 := sorry

end unique_triangle_x_value_l499_499774


namespace geometric_series_sum_l499_499683

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499683


namespace arithmetic_sequence_150th_term_l499_499546

theorem arithmetic_sequence_150th_term :
  ∀ (a₁ d : ℕ), a₁ = 3 → d = 5 → 
  (∑ (n : ℕ) in ((finset.range 150).map (λ n, n.succ)), (a₁ + (n - 1) * d)) 
  = 748 :=
by
  intros a₁ d ha₁ hd
  have h : 150 - 1 = 149 := rfl
  have h_sum : ∑ (n : ℕ) in ((finset.range 150).map (λ n, n.succ)), (a₁ + (n - 1) * d) = a₁ + 149 * d,
  sorry
  rw [ha₁, hd] at h_sum
  simp at h_sum
  exact h_sum

end arithmetic_sequence_150th_term_l499_499546


namespace gcd_sum_5n_plus_6_l499_499208

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499208


namespace window_height_is_4_l499_499933

def length : ℕ := 25
def width : ℕ := 15
def height : ℕ := 12
def door_height : ℕ := 6
def door_width : ℕ := 3
def num_windows : ℕ := 3
def window_width : ℕ := 3
def cost_per_sqft : ℕ := 10
def total_cost : ℕ := 9060

theorem window_height_is_4 :
  ∃ (h : ℕ), 
    (2 * (length + width) * height - (door_height * door_width) - num_windows * (h * window_width)) * cost_per_sqft = total_cost → 
    h = 4 :=
by
  sorry

end window_height_is_4_l499_499933


namespace choir_group_students_l499_499581

theorem choir_group_students : ∃ n : ℕ, (n % 5 = 0) ∧ (n % 9 = 0) ∧ (n % 12 = 0) ∧ (∃ m : ℕ, n = m * m) ∧ n ≥ 360 := 
sorry

end choir_group_students_l499_499581


namespace total_sets_needed_l499_499030

-- Conditions
variable (n : ℕ)

-- Theorem statement
theorem total_sets_needed : 3 * n = 3 * n :=
by sorry

end total_sets_needed_l499_499030


namespace gcd_sum_l499_499159

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499159


namespace points_subtracted_is_correct_l499_499877

-- Define the conditions
def total_questions_answered : ℕ := 82
def total_questions : ℕ := 85
def correct_answers : ℕ := 70
def raw_score : ℤ := 67

-- Define the unknown variable
noncomputable def points_subtracted_per_incorrect_answer : ℚ := sorry

-- Theorem to prove
theorem points_subtracted_is_correct : points_subtracted_per_incorrect_answer = 1 / 4 :=
by
  let incorrect_answers := total_questions_answered - correct_answers
  let total_correct_points := correct_answers
  let total_points_subtracted := (incorrect_answers : ℤ) * points_subtracted_per_incorrect_answer
  have equation : total_correct_points - total_points_subtracted = raw_score,
  sorry
-- Here, I'm assuming that the Lean prover can deduce the final required result from the given conditions, 
-- but we leave the proof as sorry as instructed.

end points_subtracted_is_correct_l499_499877


namespace prove_m_is_2_l499_499754

variable {α β γ : Real}

noncomputable def m := (Real.tan (α + β + γ)) / (Real.tan (α - β + γ))

axiom sin_condition : Real.sin (2 * (α + γ)) = 3 * Real.sin (2 * β)

theorem prove_m_is_2 (h1 : m = (Real.tan (α + β + γ)) / (Real.tan (α - β + γ))) (h2 : Real.sin (2 * (α + γ)) = 3 * Real.sin (2 * β)) : m = 2 := by
  sorry

end prove_m_is_2_l499_499754


namespace hypotenuse_is_18_8_l499_499604

def right_triangle_hypotenuse_perimeter_area (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2 * a * b = 24) ∧ (a^2 + b^2 = c^2)

theorem hypotenuse_is_18_8 : ∃ (a b c : ℝ), right_triangle_hypotenuse_perimeter_area a b c ∧ c = 18.8 :=
by
  sorry

end hypotenuse_is_18_8_l499_499604


namespace total_gumballs_l499_499593

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l499_499593


namespace consecutive_no_carry_pairs_count_l499_499300

-- Predicate to determine if two consecutive integers don't require carry when added.
def no_carry (n : ℕ) : Prop :=
  (n % 10 < 9) ∧ ((n / 10) % 10 < 9) ∧ ((n / 100) % 10 < 9)

-- Predicate to determine if a number is in the range 1000 to 2000.
def in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 2000

-- Theorem statement
theorem consecutive_no_carry_pairs_count :
  ({n | in_range n ∧ no_carry n}.card = 156) :=
sorry

end consecutive_no_carry_pairs_count_l499_499300


namespace differential_is_correct_l499_499734

-- Definitions given in the conditions
def y (x a : ℝ) := x * sqrt (4 - x^2) + a * arcsin (x / 2)

-- The theorem to prove
theorem differential_is_correct (x a : ℝ) : 
  (differential (λ x => y x a)) = (λ x => ∂(x, λ x => (4 - x^2 + a) / sqrt (4 - x^2))) :=
sorry

end differential_is_correct_l499_499734


namespace geometric_series_sum_l499_499689

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499689


namespace find_a_for_minimum_value_l499_499340

theorem find_a_for_minimum_value (x a : ℝ) (h : x > 0)
  (f : ℝ → ℝ := λ x, ((exp(x)-a)^2 + (exp(-x)+a)^2) / (exp(x)-exp(-x))) :
  (∀ x > 0, f x = 6) ↔ (a = -1 ∨ a = 7) :=
sorry

end find_a_for_minimum_value_l499_499340


namespace number_of_trees_l499_499061

theorem number_of_trees (length_of_yard : ℕ) (distance_between_trees : ℕ) 
(h1 : length_of_yard = 273) 
(h2 : distance_between_trees = 21) : 
(length_of_yard / distance_between_trees) + 1 = 14 := by
  sorry

end number_of_trees_l499_499061


namespace number_one_half_more_equals_twenty_five_percent_less_l499_499963

theorem number_one_half_more_equals_twenty_five_percent_less (n : ℤ) : 
    (80 - 0.25 * 80 = 60) → ((3 / 2 : ℚ) * n = 60) → (n = 40) :=
by
  intros h1 h2
  sorry

end number_one_half_more_equals_twenty_five_percent_less_l499_499963


namespace geometric_series_sum_l499_499702

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499702


namespace infinite_geometric_series_sum_l499_499722

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499722


namespace gcd_values_sum_l499_499074

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499074


namespace range_of_a_l499_499752

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 1

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x₁:ℝ), x₁ ∈ Set.Icc 1 Real.exp 1 → ∃ (x₂:ℝ), x₂ ∈ Set.Icc 0 3 ∧ f a x₁ = g x₂) → 
  (-1 / Real.exp 1 ≤ a ∧ a ≤ 3 / Real.exp 1) :=
sorry

end range_of_a_l499_499752


namespace monotonic_g_root_difference_l499_499786

-- Function definitions for f(x) and g(x)
def f (x : ℝ) : ℝ := x * log x
def g (x : ℝ) : ℝ := 2 * (f x) / x - x + 1 / x

-- Part 1: Prove the monotonic interval of g(x)
theorem monotonic_g (x : ℝ) (h1 : 0 < x) : 
  ∀ x, g x ≤ g x
:= sorry

-- Part 2: Prove x_2 - x_1 > 1 + em given f(x) = m has roots x_1 and x_2 with x_2 > x_1
theorem root_difference (m : ℝ) (x1 x2 : ℝ) (h1 : x1 * log x1 = m) (h2 : x2 * log x2 = m) (h3 : x2 > x1) :
  x2 - x1 > 1 + exp 1 * m
:= sorry

end monotonic_g_root_difference_l499_499786


namespace percentage_not_speaking_French_is_60_l499_499586

-- Define the number of students who speak English well and those who do not.
def speakEnglishWell : Nat := 20
def doNotSpeakEnglish : Nat := 60

-- Calculate the total number of students who speak French.
def speakFrench : Nat := speakEnglishWell + doNotSpeakEnglish

-- Define the total number of students surveyed.
def totalStudents : Nat := 200

-- Calculate the number of students who do not speak French.
def doNotSpeakFrench : Nat := totalStudents - speakFrench

-- Calculate the percentage of students who do not speak French.
def percentageDoNotSpeakFrench : Float := (doNotSpeakFrench.toFloat / totalStudents.toFloat) * 100

-- Theorem asserting the percentage of students who do not speak French is 60%.
theorem percentage_not_speaking_French_is_60 : percentageDoNotSpeakFrench = 60 := by
  sorry

end percentage_not_speaking_French_is_60_l499_499586


namespace symmetricPointCorrectCount_l499_499395

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the four symmetry conditions
def isSymmetricXaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricYOZplane (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := -P.z }
def isSymmetricYaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricOrigin (P Q : Point3D) : Prop := Q = { x := -P.x, y := -P.y, z := -P.z }

-- Define a theorem to count the valid symmetric conditions
theorem symmetricPointCorrectCount (P : Point3D) :
  (isSymmetricXaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricYOZplane P { x := P.x, y := -P.y, z := -P.z } = true → false) ∧
  (isSymmetricYaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricOrigin P { x := -P.x, y := -P.y, z := -P.z } = true → true) :=
by
  sorry

end symmetricPointCorrectCount_l499_499395


namespace gcd_sum_l499_499128

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499128


namespace range_of_a_integer_value_a_l499_499354

-- Define the system of equations and the conditions on x and y
def system_has_solution (x y a : ℝ) : Prop :=
  x + y = -7 - a ∧ x - y = 1 + 3 a ∧ x ≤ 0 ∧ y < 0

-- Prove range of values for a
theorem range_of_a : {a : ℝ | -2 < a ∧ a ≤ 3} = {a | a ∈ Set.Ioc (-2) 3} :=
  sorry

-- Simplify |a - 3| + |a + 2| within the range
def simplify_abs_expression (a : ℝ) (h : -2 < a ∧ a ≤ 3) : ℝ :=
  |a - 3| + |a + 2| = 5

-- Determine specific integer value of a
theorem integer_value_a {a : ℝ} (h : -2 < a ∧ a ≤ 3) (ha : a < -1/2) : a = -1 :=
  sorry

end range_of_a_integer_value_a_l499_499354


namespace octal_subtraction_l499_499545

theorem octal_subtraction (a b : ℕ) (h1 : a = 2 * 8 + 4) (h2 : b = 5 * 8 + 3) :
  (a - b) = -((2 * 8 + 7)) :=
by {
  -- Translate the octal numbers to decimal to perform the subtraction
  -- h1: 24 in octal is equivalent to 20 + 4 in decimal
  -- h2: 53 in octal is equivalent to 40 + 13 in decimal
  -- So, h1 - h2 = - (b - a) should be the correct relation
  -- Example proof outline (actual proof needs steps for octal borrowing)
  have h_a: a = 20, from by linarith,
  have h_b: b = 43, from by linarith,
  have h_diff : a - b = -(b - a), from sorry -- detailed steps omitted,
  show a - b = -27, from 
         by { linarith [h_diff, h_a, h_b] }
}

end octal_subtraction_l499_499545


namespace sum_of_gcd_values_l499_499220

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499220


namespace women_with_fair_hair_percentage_l499_499580

theorem women_with_fair_hair_percentage
  (A : ℝ) (B : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.25) :
  A * B = 0.10 := 
by
  rw [hA, hB]
  norm_num

end women_with_fair_hair_percentage_l499_499580


namespace acme_cheaper_than_beta_l499_499620

theorem acme_cheaper_than_beta (x : ℕ) :
  (50 + 9 * x < 25 + 15 * x) ↔ (5 ≤ x) :=
by sorry

end acme_cheaper_than_beta_l499_499620


namespace total_liquid_consumption_l499_499664

-- Define the given conditions
def elijah_drink_pints : ℝ := 8.5
def emilio_drink_pints : ℝ := 9.5
def isabella_drink_liters : ℝ := 3
def xavier_drink_gallons : ℝ := 2
def pint_to_cups : ℝ := 2
def liter_to_cups : ℝ := 4.22675
def gallon_to_cups : ℝ := 16
def xavier_soda_fraction : ℝ := 0.60
def xavier_fruit_punch_fraction : ℝ := 0.40

-- Define the converted amounts
def elijah_cups := elijah_drink_pints * pint_to_cups
def emilio_cups := emilio_drink_pints * pint_to_cups
def isabella_cups := isabella_drink_liters * liter_to_cups
def xavier_total_cups := xavier_drink_gallons * gallon_to_cups
def xavier_soda_cups := xavier_soda_fraction * xavier_total_cups
def xavier_fruit_punch_cups := xavier_fruit_punch_fraction * xavier_total_cups

-- Total amount calculation
def total_cups := elijah_cups + emilio_cups + isabella_cups + xavier_soda_cups + xavier_fruit_punch_cups

-- Proof statement
theorem total_liquid_consumption : total_cups = 80.68025 := by
  sorry

end total_liquid_consumption_l499_499664


namespace range_of_g_l499_499302

noncomputable def g (x : ℝ) := (sin x ^ 3 + 6 * sin x ^ 2 + sin x + 3 * cos x ^ 2 - 9) / (sin x - 1)

theorem range_of_g :
  ∃ (a b : ℝ), a = 2 ∧ b = 12 ∧ ∀ x : ℝ, sin x ≠ 1 → g x ∈ set.Ico a b :=
by
  sorry

end range_of_g_l499_499302


namespace infinite_geometric_series_sum_l499_499697

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499697


namespace equation_satisfied_by_r_l499_499372

theorem equation_satisfied_by_r {x y z r : ℝ} (h1: x ≠ y) (h2: y ≠ z) (h3: z ≠ x) 
    (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0) 
    (h7: ∃ (r: ℝ), x * (y - z) = (y * (z - x)) / r ∧ y * (z - x) = (z * (y - x)) / r ∧ z * (y - x) = (x * (y - z)) * r) 
    : r^2 - r + 1 = 0 := 
sorry

end equation_satisfied_by_r_l499_499372


namespace infinitely_many_divisors_l499_499472

theorem infinitely_many_divisors (a : ℕ) : ∃ᶠ n in at_top, n ∣ a ^ (n - a + 1) - 1 :=
sorry

end infinitely_many_divisors_l499_499472


namespace gcd_sum_5n_6_n_eq_12_l499_499182

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499182


namespace polynomial_irreducible_in_Q_l499_499431

noncomputable def polynomialP (a : List ℕ) : Polynomial ℤ :=
  a.foldr (λ (ai : ℕ) (acc : Polynomial ℤ), Polynomial.C (ai : ℤ) + Polynomial.X * acc) 0

theorem polynomial_irreducible_in_Q
    (p : ℕ) (a : List ℕ)
    (h1: p = a.foldr (λ (x d : ℕ) (s : ℕ), x * 10 ^ d + s) 0)
    (h2 : p.prime) :
    irreducible (polynomialP a) :=
by
  sorry

end polynomial_irreducible_in_Q_l499_499431


namespace max_f_12345_l499_499864

def S : set (ℝ → ℝ) :=
{ f | ∀ x y z : ℝ, f(x^2 + y * f(z)) = x * f(x) + z * f(y) ∧ ∃ x : ℝ, f x ≠ 0 }

noncomputable def f (f : ℝ → ℝ) [h : f ∈ S] : ℝ := f 12345

theorem max_f_12345 (f ∈ S) : f 12345 = 12345 :=
sorry

end max_f_12345_l499_499864


namespace sum_gcd_possible_values_eq_twelve_l499_499246

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499246


namespace gcd_sum_divisors_eq_12_l499_499176

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499176


namespace lines_intersect_l499_499038

theorem lines_intersect :
  ∃ t u, 
    (1 - 2 * t = 2 + 3 * u) ∧ (4 + 3 * t = 5 + u) ∧
    (1 - 2 * t = 3 / 7) ∧ (4 + 3 * t = 34 / 7) := 
begin
  use 2 / 7,
  use 1 / 21,
  split,
  { rw [eq_comm, sub_eq_sub_iff_add_eq_add, add_neg_eq_iff_eq_add],
    field_simp,
    norm_num },
  split,
  { rw [eq_comm, add_neg_eq_iff_eq_add, eq_comm],
    field_simp,
    norm_num },
  split,
  { field_simp,
    norm_num },
  { field_simp,
    norm_num },
end

end lines_intersect_l499_499038


namespace find_k_for_parallelepiped_volume_l499_499285

theorem find_k_for_parallelepiped_volume :
  let v1 := ![3, 4, 5],
      v2 := ![1, k, 3],
      v3 := ![2, 3, k],
      M := λ k, matrix.of_vector 3 3 ![v1, v2, v3]
  in 0 < k →
     ‖det (M k)‖ = 30 →
     k = 7.25 :=
by
  intros k hk hdet
  sorry

end find_k_for_parallelepiped_volume_l499_499285


namespace gcd_values_sum_l499_499071

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499071


namespace total_gumballs_l499_499592

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l499_499592


namespace largest_possible_m_is_correct_l499_499947

noncomputable def largest_possible_m : ℕ :=
  let q := [λ x, x - 1, λ x, x^4 + x^3 + x^2 + x + 1, λ x, x^5 + 1]
  q.length

theorem largest_possible_m_is_correct : largest_possible_m = 3 := 
  sorry

end largest_possible_m_is_correct_l499_499947


namespace largest_N_not_payable_l499_499903

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499903


namespace sum_of_gcd_values_l499_499118

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499118


namespace max_elements_with_min_distance_5_l499_499863

def S : Set (Vector ℕ 8) := { A : Vector ℕ 8 | ∀ i, A.get i ∈ {0, 1} }

def d (A B : Vector ℕ 8) : ℕ := (List.range 8).map (λ i, abs (A.get i - B.get i)).sum

theorem max_elements_with_min_distance_5 (S' : Set (Vector ℕ 8)) (h : ∀ A B ∈ S', d A B ≥ 5) :
  Set.Card S' ≤ 4 := sorry

end max_elements_with_min_distance_5_l499_499863


namespace max_product_of_digits_l499_499310

theorem max_product_of_digits (n : ℕ) (sum_digs_eq_25 : digits_sum n = 25) :
  (∀ m : ℕ, digits_sum m = 25 → digits_product m ≤ digits_product n) ∧
  (∀ k : ℕ, digits_sum k = 25 ∧ digits_product k = digits_product n → k ≥ n) :=
sorry

-- Definitions to use with the theorem
def digits_sum (n : ℕ) : ℕ :=
  n.digits.sum

def digits_product (n : ℕ) : ℕ :=
  n.digits.prod

-- Specific instantiation to show the answer
example : max_product_of_digits 33333334 := 
sorry

end max_product_of_digits_l499_499310


namespace tangent_line_eq_l499_499831

theorem tangent_line_eq :
  (∀ P : ℝ × ℝ, P = (1, 2) → ∃ c ∈ ℝ, P.1 ^ 2 + P.2 ^ 2 = c^2) →
  (∃ A B C : ℝ, A * 1 + B * 2 + C = 0 ∧ A = 1 ∧ B = 2 ∧ C = -5) :=
by
  -- Assuming the point P(1, 2) is on the circle centered at the origin
  intro h
  -- Showing the desired line equation
  existsi [1, 2, -5]
  split
  -- Show that the point satisfies the line equation
  { calc (1 : ℝ) * 1 + (2 : ℝ) * 2 + (-5 : ℝ)
        = 1 + 4 - 5 : by ring
    ... = 0 : by norm_num }
  -- Show the coefficients are what we expect
  split
  { rfl }
  split
  { rfl }
  { rfl }

end tangent_line_eq_l499_499831


namespace fertilizer_on_full_field_l499_499034

def field_area : ℕ := 7200
def partial_field_area : ℕ := 3600
def fertilizer_on_partial_area : ℕ := 600

theorem fertilizer_on_full_field (field_area partial_field_area fertilizer_on_partial_area : ℕ) :
  ({rate : ℚ // rate = (fertilizer_on_partial_area : ℚ) / (partial_field_area : ℚ)}).val * (field_area : ℚ) = 1200 := by
  sorry

end fertilizer_on_full_field_l499_499034


namespace matrix_power_norm_bound_l499_499308

noncomputable def matrix_norm (M : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  supr (λ x : ℝ^n, if h : x = 0 then 0 else ‖M.mulVec x‖ / ‖x‖) -- defining the operator norm induced by the Euclidean norm

theorem matrix_power_norm_bound (A : Matrix (Fin n) (Fin n) ℝ) (hA : ∀ k ∈ ℕ, matrix_norm (A^k - A^(k-1)) ≤ 1 / (2002 * k)) :
  ∀ k ∈ ℕ, matrix_norm (A^k) ≤ 2002 := by
  sorry

end matrix_power_norm_bound_l499_499308


namespace gcd_sum_5n_plus_6_n_l499_499153

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499153


namespace general_term_of_sequence_l499_499791

theorem general_term_of_sequence :
  ∀ (a : ℕ → ℝ),
    (a 1 = 3) ∧ 
    (∀ n, n ≥ 2 → a n + a (n-1) = 2) ∧
    (∀ n, n ≥ 2 ∧ a n > 0 → (n * (3 * n + 1) / (a n - a (n-1)))) →
    (∀ n, n ≥ 1 → a n = 1 + (n+1) * sqrt n) :=
by
  sorry

end general_term_of_sequence_l499_499791


namespace calculate_expression_l499_499639

theorem calculate_expression :
  |-2| + (sqrt 2 - 1) ^ 0 - (-5) - (1 / 3) ^ (-1) = 5 := 
by
  -- Proof goes here
  sorry

end calculate_expression_l499_499639


namespace smallest_c_correct_l499_499657

noncomputable def smallest_c : ℝ := 2002^(2003:ℝ)

theorem smallest_c_correct : ∀ x, x > smallest_c → ∃ y, y = log 2005 (log 2004 (log 2003 (log 2002 x))) :=
by
  intro x hx
  use log 2005 (log 2004 (log 2003 (log 2002 x)))
  sorry

end smallest_c_correct_l499_499657


namespace problem_1_problem_2_l499_499341

-- (1) Conditions and proof statement
theorem problem_1 (x y m : ℝ) (P : ℝ × ℝ) (k : ℝ) :
  (x, y) = (1, 2) → m = 1 →
  ((x - 1)^2 + (y - 2)^2 = 4) →
  P = (3, -1) →
  (l : ℝ → ℝ → Prop) →
  (∀ x y, l x y ↔ x = 3 ∨ (5 * x + 12 * y - 3 = 0)) →
  l 3 (-1) →
  l (x + k * (3 - x)) (y-1) := sorry

-- (2) Conditions and proof statement
theorem problem_2 (x y m : ℝ) (line : ℝ → ℝ) :
  (x - 1)^2 + (y - 2)^2 = 5 - m →
  m < 5 →
  (2 * (5 - m - 20) ^ (1/2) = 2 * (5) ^ (1/2)) →
  m = -20 := sorry

end problem_1_problem_2_l499_499341


namespace arrangement_count_l499_499069

theorem arrangement_count (n : ℕ) (l : List ℕ) (h : l = [1, 2, 3, 4, 5, 6, 7, 8]) : 
  ∃ (count : ℕ), count = 1152 ∧ 
  ∀ (perm : List ℕ), perm.perm l → 
    let (left, right) := perm.splitAt (l.indexOf 8 + 1)
    in (left.sum = 14 ∧ right.sum = 14) :=
by
  sorry

end arrangement_count_l499_499069


namespace four_digit_numbers_neither_multiple_of_4_9_6_l499_499359

open Nat

def is_multiple_of (n k : ℕ) : Prop := k ∣ n

theorem four_digit_numbers_neither_multiple_of_4_9_6 :
  let N := (9999 - 1000 + 1) in
  let count_multiples := λ k : ℕ, (9999 / k - (1000 + k - 1) / k + 1) in
  let count_4 := count_multiples 4 in
  let count_9 := count_multiples 9 in
  let count_6 := count_multiples 6 in
  let count_36 := count_multiples 36 in
  let count_12 := count_multiples 12 in
  let count_18 := count_multiples 18 in
  let count_total := count_4 + count_9 + count_6 - count_36 - count_12 - count_18 + count_36 in
  N - count_total = 4500 := by
{
  let N := (9999 - 1000 + 1)
  let count_multiples := λ k : ℕ, (9999 / k - (1000 + k - 1) / k + 1)
  let count_4 := count_multiples 4
  let count_9 := count_multiples 9
  let count_6 := count_multiples 6
  let count_36 := count_multiples 36
  let count_12 := count_multiples 12
  let count_18 := count_multiples 18
  let count_total := count_4 + count_9 + count_6 - count_36 - count_12 - count_18 + count_36
  have h : N - count_total = 4500 := by sorry
  exact h
}

end four_digit_numbers_neither_multiple_of_4_9_6_l499_499359


namespace gcd_sum_l499_499158

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499158


namespace complete_graph_color_removal_l499_499528

-- Definitions based on conditions
variables (V : Type) [fintype V] [decidable_eq V] (E : V → V → Prop)

noncomputable def K50 : Type := fin 50

namespace GraphTheoryProof

-- Define the complete graph with 50 vertices and edges colored by one of three colors
def is_complete_graph (G : Type) := ∀ (u v : G), u ≠ v → E u v

-- Define the colors
inductive Color
| C1
| C2
| C3

-- Define edges are colored
variables (coloring : K50 → K50 → Color)

-- Desired property after removing edges of one color
def connected_after_removing_color (c : Color) : Prop :=
  ∀ (u v : K50), ∃ (path : list K50),
    ∀ (i : fin (path.length - 1)),
      coloring (path.nth_le i sorry) (path.nth_le (i + 1) sorry) ≠ c

-- The main theorem
theorem complete_graph_color_removal :
  (is_complete_graph K50) →
  (∃ c : Color, connected_after_removing_color coloring c) :=
begin
  intros is_complete,
  sorry
end

end GraphTheoryProof

end complete_graph_color_removal_l499_499528


namespace Mary_income_is_80_percent_of_Juan_l499_499562

variables (J T M : ℝ)

def Tim_income_cond : Prop := T = 0.5 * J
def Mary_income_cond : Prop := M = 1.6 * T

theorem Mary_income_is_80_percent_of_Juan :
  Tim_income_cond J T → Mary_income_cond T M → M = 0.8 * J :=
by
  intro hT hM
  rw [Tim_income_cond, Mary_income_cond] at *
  rw hT at hM
  sorry

end Mary_income_is_80_percent_of_Juan_l499_499562


namespace gcd_5n_plus_6_n_sum_l499_499108

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499108


namespace possible_values_of_product_l499_499290

theorem possible_values_of_product 
  (P_A P_B P_C P_D P_E : ℕ)
  (H1 : P_A = P_B + P_C + P_D + P_E)
  (H2 : ∃ n1 n2 n3 n4, 
          ((P_B = n1 * (n1 + 1)) ∨ (P_B = n2 * (n2 + 1) * (n2 + 2)) ∨ 
           (P_B = n3 * (n3 + 1) * (n3 + 2) * (n3 + 3)) ∨ (P_B = n4 * (n4 + 1) * (n4 + 2) * (n4 + 3) * (n4 + 4))) ∧
          ∃ m1 m2 m3 m4, 
          ((P_C = m1 * (m1 + 1)) ∨ (P_C = m2 * (m2 + 1) * (m2 + 2)) ∨ 
           (P_C = m3 * (m3 + 1) * (m3 + 2) * (m3 + 3)) ∨ (P_C = m4 * (m4 + 1) * (m4 + 2) * (m4 + 3) * (m4 + 4))) ∧
          ∃ o1 o2 o3 o4, 
          ((P_D = o1 * (o1 + 1)) ∨ (P_D = o2 * (o2 + 1) * (o2 + 2)) ∨ 
           (P_D = o3 * (o3 + 1) * (o3 + 2) * (o3 + 3)) ∨ (P_D = o4 * (o4 + 1) * (o4 + 2) * (o4 + 3) * (o4 + 4))) ∧
          ∃ p1 p2 p3 p4, 
          ((P_E = p1 * (p1 + 1)) ∨ (P_E = p2 * (p2 + 1) * (p2 + 2)) ∨ 
           (P_E = p3 * (p3 + 1) * (p3 + 2) * (p3 + 3)) ∨ (P_E = p4 * (p4 + 1) * (p4 + 2) * (p4 + 3) * (p4 + 4))) ∧ 
          ∃ q1 q2 q3 q4, 
          ((P_A = q1 * (q1 + 1)) ∨ (P_A = q2 * (q2 + 1) * (q2 + 2)) ∨ 
           (P_A = q3 * (q3 + 1) * (q3 + 2) * (q3 + 3)) ∨ (P_A = q4 * (q4 + 1) * (q4 + 2) * (q4 + 3) * (q4 + 4)))) :
  P_A = 6 ∨ P_A = 24 :=
by sorry

end possible_values_of_product_l499_499290


namespace equilateral_triangle_properties_l499_499392

noncomputable def equilateral_triangle_perimeter (s : ℕ) : ℕ :=
  3 * s

noncomputable def equilateral_triangle_area (s : ℕ) : ℝ :=
  (sqrt 3 / 4) * (s ^ 2)

theorem equilateral_triangle_properties (s : ℕ) (h : s = 10) :
  equilateral_triangle_perimeter s = 30 ∧ equilateral_triangle_area s = 25 * sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l499_499392


namespace gcd_sum_l499_499084

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499084


namespace area_of_S4_l499_499050

theorem area_of_S4 (S1_area : ℝ) (S2_area : ℝ) (S3_area : ℝ) : 
  S1_area = 25 → 
  S2_area = (S1_area * 2) / 4 → 
  S3_area = (S2_area * 2) / 4 →
  (S3_area * 2) / 4 = 3.125 :=
begin
  sorry,
end

end area_of_S4_l499_499050


namespace pinocchio_cannot_pay_exactly_l499_499889

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499889


namespace part1_l499_499440

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
|x - 1 + a| + |x - a|

theorem part1 (a x : ℝ) (h : a ≥ 2) : f x a ≥ 3 :=
begin
  sorry
end

end part1_l499_499440


namespace infinite_geometric_series_sum_l499_499696

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499696


namespace infinite_geometric_series_sum_l499_499715

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499715


namespace incenters_equal_distances_l499_499416

open EuclideanGeometry

theorem incenters_equal_distances
  {A B C N K L P I J Q : Point}
  (hN : N ∈ line A B)
  (hK : K ∈ line B C)
  (hL : L ∈ line C A)
  (hALBK : dist A L = dist B K)
  (hCN_bisector : is_angle_bisector C N A B)
  (hP : P = intersection_point (line_through A K) (line_through B L))
  (hI : (incenter_triangle A P L) = I)
  (hJ : (incenter_triangle B P K) = J)
  (hQ : Q = intersection_point (line_through I J) (line_through C N)) :
  dist I P = dist J Q := sorry

end incenters_equal_distances_l499_499416


namespace cube_volume_from_surface_area_l499_499563

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 1734) : 
  ∃ V : ℝ, V = 4913 := 
by
  let s := real.sqrt (1734 / 6)
  have : s^2 = 1734 / 6 := by
    rw [real.sqrt_sq (1734 / 6)]
    exact real.eq_div_of_mul_eq (by norm_num)(by norm_num)
  let V := s^3
  have : V = 4913 := by
    rw [pow_succ]
    norm_num
  use V
  exact this
sorry

end cube_volume_from_surface_area_l499_499563


namespace range_of_a_for_monotonic_function_l499_499333

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), is_monotonic_on (f · a) (Set.Iic (-1)) → a ≤ 3 :=
by
  intros a h
  sorry

end range_of_a_for_monotonic_function_l499_499333


namespace compare_logs_l499_499823

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l499_499823


namespace product_of_real_roots_l499_499301

theorem product_of_real_roots (x : ℝ) (hx : x ^ (Real.log x / Real.log 5) = 5) :
  (∃ a b : ℝ, a ^ (Real.log a / Real.log 5) = 5 ∧ b ^ (Real.log b / Real.log 5) = 5 ∧ a * b = 1) :=
sorry

end product_of_real_roots_l499_499301


namespace sum_of_gcd_values_l499_499232

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499232


namespace min_value_sin_cos_expression_l499_499298

theorem min_value_sin_cos_expression (x : ℝ) : 
  abs (sin x + cos x + (cos x - sin x) / (cos (2*x))) ≥ 2 :=
sorry

end min_value_sin_cos_expression_l499_499298


namespace molecular_weight_of_7_moles_of_CaO_l499_499549

/-- The molecular weight of 7 moles of calcium oxide (CaO) -/
def Ca_atomic_weight : Float := 40.08
def O_atomic_weight : Float := 16.00
def CaO_molecular_weight : Float := Ca_atomic_weight + O_atomic_weight

theorem molecular_weight_of_7_moles_of_CaO : 
    7 * CaO_molecular_weight = 392.56 := by 
sorry

end molecular_weight_of_7_moles_of_CaO_l499_499549


namespace seventy_fifth_percentile_correct_l499_499336

theorem seventy_fifth_percentile_correct :
  ∀ (data : List ℝ)
    (h_size : data.length = 100)
    (h_percentile : (List.sort (≤) data).nth_le 74 (by simp [h_size]) = (9.3 + (List.sort (≤) data).nth_le 75 (by simp [h_size])) / 2),
    ∃ i j : ℕ, i = 74 ∧ j = 75 ∧ (List.sort (≤) data).nth_le i (by simp [h_size]) = 9.3 / 2 + (List.sort (≤) data).nth_le j (by simp [h_size]) / 2 :=
begin
    -- proof goes here
    sorry
end

end seventy_fifth_percentile_correct_l499_499336


namespace picture_area_l499_499603

def outer_height : ℕ := 100
def outer_width  : ℕ := 140
def frame_width_longer : ℕ := 15
def frame_width_shorter : ℕ := 20

theorem picture_area:
  let picture_height := outer_height - 2 * frame_width_shorter in
  let picture_width  := outer_width - 2 * frame_width_longer in
  picture_height * picture_width = 6600 := 
by
  sorry

end picture_area_l499_499603


namespace gcd_5n_plus_6_n_sum_l499_499107

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499107


namespace sum_of_gcd_values_l499_499121

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499121


namespace literary_common_senses_incorrect_l499_499064

-- Define the conditions
def condition_A : Prop :=
  ∀ (title : String), title = "Chen Qing Biao" → title.contains "Biao" 

def condition_B : Prop :=
  Wang_Shifu ∈ great_dramatists_Yuan_Dynasty

def condition_C : Prop :=
  Shakespeare's_works = ["Hamlet", "Othello", "King Lear", "Macbeth"]

def condition_D : Prop :=
  Qu_Yuan's_works = ["Li Sao", "Jiu Ge", "Jiu Zhang", "Tian Wen"]

-- Incorrectness definition
def incorrect (option : String) : Prop :=
  option = "Option B is incorrect"

/-- Prove that option B is incorrect given the conditions -/
theorem literary_common_senses_incorrect : incorrect "Option B is incorrect" :=
by
  apply (incorrect "Option B is incorrect"),
  sorry -- Proof not needed as per instructions

end literary_common_senses_incorrect_l499_499064


namespace connections_in_office_network_l499_499960

theorem connections_in_office_network (num_switches : ℕ) (connections_per_switch : ℕ)
  (h1 : num_switches = 30) (h2 : connections_per_switch = 4) : 
  (num_switches * connections_per_switch) / 2 = 60 := by
  rw [h1, h2]
  norm_num
  sorry

end connections_in_office_network_l499_499960


namespace largest_N_cannot_pay_exactly_without_change_l499_499883

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499883


namespace original_sticker_price_l499_499796

-- Define the conditions in Lean
variables {x : ℝ} -- x is the original sticker price of the laptop

-- Definitions based on the problem conditions
def store_A_price (x : ℝ) : ℝ := 0.80 * x - 50
def store_B_price (x : ℝ) : ℝ := 0.70 * x
def heather_saves (x : ℝ) : Prop := store_B_price x - store_A_price x = 30

-- The theorem to prove
theorem original_sticker_price (x : ℝ) (h : heather_saves x) : x = 200 :=
by
  sorry

end original_sticker_price_l499_499796


namespace gcd_values_sum_l499_499070

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499070


namespace gcd_sum_l499_499131

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499131


namespace problem_statement_l499_499422

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 11 - 2 * n

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := n * 11 - n * (n + 1)

theorem problem_statement :
  (∀ n, a n = 11 - 2 * n) ∧ -- a) Sequence definition
  ((∃ d : ℤ, ∀ n, a (n + 1) - a n = d) ∧ d = -2) ∧ -- 1) Arithmetic sequence with common difference -2
  (S 4 = S 6) ∧ -- 2) S₄ = S₆
  (S 5 = max_seq S 5) -- 3) Sₙ reaches maximum at n = 5
:= sorry

end problem_statement_l499_499422


namespace sum_of_gcd_values_l499_499237

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499237


namespace geometric_series_sum_l499_499701

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499701


namespace equidistant_condition_l499_499990

theorem equidistant_condition (x y z : ℝ) (A B P : EuclideanSpace ℝ 3):
  (P = λ _, (x, y, z)) →
  (A = λ _, (-1, 2, 3)) →
  (B = λ _, (0, 0, 5)) →
  dist P A = dist P B →
  2 * x - 4 * y + 4 * z = 11 :=
by
  intros
  sorry

end equidistant_condition_l499_499990


namespace problem1_problem2a_problem2b_problem2c_l499_499025

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 2

theorem problem1 : 27^(2/3) + real.log 5 / real.log 10 - 2 * (real.log 3 / real.log 2) + real.log 2 / real.log 10 + (real.log 9 / real.log 2) = 10 :=
by sorry

theorem problem2a (a : ℝ) : f (-real.sqrt 2) = 8 + 5 * real.sqrt 2 :=
by sorry

theorem problem2b (a : ℝ) : f (-a) = 3 * a^2 + 5 * a + 2 :=
by sorry

theorem problem2c (a : ℝ) : f (a + 3) = 3 * a^2 + 13 * a + 14 :=
by sorry

end problem1_problem2a_problem2b_problem2c_l499_499025


namespace sum_gcd_possible_values_eq_twelve_l499_499250

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499250


namespace intersection_A_B_l499_499328

open Set

def A : Set ℝ := {x | (x - 1) * (x - 3) < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l499_499328


namespace correct_description_of_gametes_l499_499932

/-
The correct description of the formation of male and female gametes through meiosis in higher animals
and the fertilization process is option C (The sperm that enters and fuses with the egg cell carries almost no cytoplasm).

Conditions:
1. Each egg cell inherits 1/2 of the genetic material from the nucleus of the primary oocyte.
2. The chances of alleles entering the egg cell are not equal because only one egg cell is formed from a single meiosis.
3. The sperm that enters and fuses with the egg cell carries almost no cytoplasm.
4. The chances of male and female gametes combining with each other are equal because their numbers are equal.
-/

theorem correct_description_of_gametes (A B C D : Prop)
  (hA : A = "Each egg cell inherits 1/2 of the genetic material from the nucleus of the primary oocyte")
  (hB : B = "The chances of alleles entering the egg cell are not equal because only one egg cell is formed from a single meiosis")
  (hC : C = "The sperm that enters and fuses with the egg cell carries almost no cytoplasm")
  (hD : D = "The chances of male and female gametes combining with each other are equal because their numbers are equal") : 
  C = "The correct description of the formation of male and female gametes through meiosis in higher animals and the fertilization process" := 
sorry

end correct_description_of_gametes_l499_499932


namespace mouse_farther_point_and_sum_l499_499602

theorem mouse_farther_point_and_sum :
  let cheese_x := 15
  let cheese_y := 12
  let line_slope := -4
  let line_y_intercept := 9
  let mouse_trajectory := ∀ x, x * line_slope + line_y_intercept = -4 * x + 9
  let perpendicular_slope := 1 / 4
  let cheese_line := cheese_y + perpendicular_slope * (cheese_x - x) = (perpendicular_slope * x + (33 / 4))
  let intersect :=
    have eq1: 1 / 4 * x + 33 / 4 = -4 * x + 9,
    begin
      sorry
    end,
    begin
      sorry
    end,

  let intersect_x := x = 3 / 17,
  let intersect_y := -4 * (3 / 17) + 9,
  let intersect_point := (intersect_x, intersect_y),
  let sum_a_b := (3 / 17) + ((153 / 17) - (12 / 17)),
  let result := (3 / 17) + (141 / 17),
  sum_a_b = 144 / 17
:= sorry

end mouse_farther_point_and_sum_l499_499602


namespace gcd_sum_5n_6_n_eq_12_l499_499189

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499189


namespace tangent_circle_circumference_l499_499539

theorem tangent_circle_circumference
  (A B C : Point)
  (r1 r2 : ℝ)
  (arc_AC arc_BC: Arc)
  (circle1 circle2 : Circle)
  (h1 : arc_BC.center = B)
  (h2 : arc_AC.center = A)
  (h3 : arc_BC.length = 10 * π)
  (h4 : arc_AC.central_angle = 120 * (π / 180))
  (h5 : circle1.radius = r1)
  (h6 : circle2.radius = r2)
  (h7 : circle1.is_tangent_to arc_AC)
  (h8 : circle1.is_tangent_to arc_BC)
  (h9 : circle1.is_tangent_to (line_AB : Line))
  (h10 : r1 = r2 / 2)
  : circle1.circumference = 15 * π := by
  sorry

end tangent_circle_circumference_l499_499539


namespace probability_negative_product_l499_499965

theorem probability_negative_product :
  let S := {-6, -3, 1, 5, 8, -9}
  ( S.card = 6 ) →
  let neg := { -6, -3, -9 }
  ( neg.card = 3 ) →
  let pos := { 1, 5, 8 }
  ( pos.card = 3 ) →
  ( ∃ nums : Finset ℤ, nums ⊆ S ∧ nums.card = 2 ) →
  ( let neg_prod_count := neg.card * pos.card
    let total_count := S.card.choose 2
    neg_prod_count / total_count = (3 / 5 : ℚ) ) := sorry

end probability_negative_product_l499_499965


namespace circle_tangency_problem_l499_499861

theorem circle_tangency_problem :
  let u1 := ∀ (x y : ℝ), x^2 + y^2 + 8 * x - 30 * y - 63 = 0
  let u2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 30 * y + 99 = 0
  let line := ∀ (b x : ℝ), y = b * x
  ∃ p q : ℕ, gcd p q = 1 ∧ n^2 = (p : ℚ) / (q : ℚ) ∧ p + q = 7 :=
sorry

end circle_tangency_problem_l499_499861


namespace exist_distinct_rational_rectangles_l499_499759

def Rectangle := {width : ℚ // width > 0} × {height : ℚ // height > 0}

def cut_square_to_rectangles (n : ℕ) (square_area : ℚ) : Prop :=
  ∃ rects : list Rectangle, rects.length = n ∧
    (∀r ∈ rects, r.1.1 * r.2.1 > 0 ∧ r.1.1 + r.2.1 ≤ square_area) ∧
    (∀ i j, i < n → j < n → i ≠ j → (rects.nth_le i sorry).1.1 ≠ (rects.nth_le j sorry).1.1 ∧ 
            (rects.nth_le i sorry).2.1 ≠ (rects.nth_le j sorry).2.1)

theorem exist_distinct_rational_rectangles : 
  cut_square_to_rectangles 5 1 :=
sorry

end exist_distinct_rational_rectangles_l499_499759


namespace base_rate_of_second_company_l499_499966

-- Define the conditions
def United_base_rate : ℝ := 8.00
def United_rate_per_minute : ℝ := 0.25
def Other_rate_per_minute : ℝ := 0.20
def minutes : ℕ := 80

-- Define the total bill equations
def United_total_bill (minutes : ℕ) : ℝ := United_base_rate + United_rate_per_minute * minutes
def Other_total_bill (minutes : ℕ) (B : ℝ) : ℝ := B + Other_rate_per_minute * minutes

-- Define the claim to prove
theorem base_rate_of_second_company : ∃ B : ℝ, Other_total_bill minutes B = United_total_bill minutes ∧ B = 12.00 := by
  sorry

end base_rate_of_second_company_l499_499966


namespace calculate_principal_l499_499558

theorem calculate_principal (R T SI : ℝ) (hR : R = 12) (hT : T = 3) (hSI : SI = 6480) :
  let P := 6480 / 0.36 in P = 18000 :=
by 
  sorry

end calculate_principal_l499_499558


namespace gcd_sum_5n_6_n_eq_12_l499_499187

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499187


namespace number_of_roses_l499_499615

def vase_capacity : ℕ := 6
def carnations : ℕ := 7
def vases : ℕ := 9

theorem number_of_roses : ∃ (roses : ℕ), vase_capacity * vases - carnations = roses ∧ roses = 47 :=
by 
  use 47
  simp [vase_capacity, carnations, vases]
  sorry

end number_of_roses_l499_499615


namespace count_valid_three_digit_numbers_l499_499798

-- Definitions for the conditions
def is_valid_digit_set (x y z : ℕ) : Prop :=
  (1 <= x ∧ x <= 9) ∧ (1 <= y ∧ y <= 9) ∧ (1 <= z ∧ z <= 9) ∧ (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (x + z = 2 * y)

-- Main theorem statement
theorem count_valid_three_digit_numbers : 
  (∃ digit_set : Finset (ℕ × ℕ × ℕ), 
       (∀ (a b c : ℕ), (a, b, c) ∈ digit_set → is_valid_digit_set a b c) ∧ 
       (digit_set.card * 6 = 132)) :=
sorry

end count_valid_three_digit_numbers_l499_499798


namespace find_peaches_l499_499272

theorem find_peaches (A P : ℕ) (h1 : A + P = 15) (h2 : 1000 * A + 2000 * P = 22000) : P = 7 := sorry

end find_peaches_l499_499272


namespace david_number_sum_l499_499287

theorem david_number_sum :
  ∃ (x y : ℕ), (10 ≤ x ∧ x < 100) ∧ (100 ≤ y ∧ y < 1000) ∧ (1000 * x + y = 4 * x * y) ∧ (x + y = 266) :=
sorry

end david_number_sum_l499_499287


namespace solution_to_quadratic_eq_l499_499517

theorem solution_to_quadratic_eq (p q : ℝ) (h : ∀ x, 2*x^2 + 5 = 7*x - 2 → x = p + q * complex.I ∨ x = p - q * complex.I) :
  p + q^2 = 35 / 16 :=
by sorry

end solution_to_quadratic_eq_l499_499517


namespace plane_division_by_circles_l499_499838

def divides_plane_into_regions : ℕ → ℕ
| 0       := 1
| (n + 1) := divides_plane_into_regions n + 2 * n

theorem plane_division_by_circles (n : ℕ) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → (∃! p : Type, p ∈ (divides_plane_by_circles i ∩ divides_plane_by_circles j))) ∧ 
  (∀ (i j k : ℕ), i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → ¬∃ p : Type, p ∈ (divides_plane_by_circles i ∩ divides_plane_by_circles j ∩ divides_plane_by_circles k)) 
  → divides_plane_into_regions n = n^2 - n + 2 :=
sorry

end plane_division_by_circles_l499_499838


namespace geometric_series_sum_l499_499682

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499682


namespace rectangle_area_decrease_l499_499560

noncomputable def rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) : ℝ :=
  let L' := 1.10 * L
  let B' := 0.90 * B
  let A  := L * B
  let A' := L' * B'
  A'

theorem rectangle_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  rectangle_area_change L B hL hB = 0.99 * (L * B) := by
  sorry

end rectangle_area_decrease_l499_499560


namespace find_a_value_l499_499771

theorem find_a_value (a : ℝ) (A : set ℝ := {3, 4, 2 * a - 3}) (B : set ℝ := {a}) 
  (h : (A ∩ B) ≠ ∅) : a = 4 :=
sorry

end find_a_value_l499_499771


namespace prime_numbers_in_series_l499_499024

def altern_series (n : ℕ) : ℕ :=
  if n = 1 then 101 else (1 : ℕ + 10^(2*(n-1)))

theorem prime_numbers_in_series :
  (∃ n : ℕ, prime (altern_series n)) ∧ (∀ n : ℕ, n ≠ 1 → ¬prime (altern_series n)) :=
sorry

end prime_numbers_in_series_l499_499024


namespace sum_of_given_infinite_geometric_series_l499_499707

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499707


namespace sum_of_gcd_values_l499_499224

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499224


namespace sum_alternating_series_l499_499551

theorem sum_alternating_series :
  let s := (finset.range 2023).sum (λ n, if n % 2 = 0 then n + 1 else - (n + 1))
  s = 3034 :=
by
  let s := (finset.range 2023).sum (λ n, if n % 2 = 0 then n + 1 else - (n + 1))
  have : s = 3034 := by
    sorry
  exact this

end sum_alternating_series_l499_499551


namespace prove_m_plus_n_l499_499959

noncomputable def find_m_plus_n (a b c d m n : ℝ) : Prop :=
  -20 ∈ (λ x, x^3 + a * x + b) ∧
  -21 ∈ (λ x, x^3 + c * x^2 + d) ∧
  (m + Real.sqrt n * Complex.I) ∈ (λ x, x^3 + a * x + b) ∧
  (m + Real.sqrt n * Complex.I) ∈ (λ x, x^3 + c * x^2 + d) ∧
  m > 0 ∧ n > 0 ∧ Real.I = Complex.I

theorem prove_m_plus_n :
  ∃ m n : ℝ, find_m_plus_n 0 0 1 (-21 * (100 + 320)) 10 320 ∧ m + n = 330 :=
by
  sorry

end prove_m_plus_n_l499_499959


namespace total_gumballs_l499_499591

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end total_gumballs_l499_499591


namespace solution_l499_499363

theorem solution (x : ℝ) (hx : log (3 * x) 343 = x) :
  x * 3 = 7 ∧ x ≠ ⌊x⌋ ∧ ¬∃ n : ℕ, x = n ^ 2 ∧ ¬∃ n : ℕ, x = n ^ 3 :=
by
  sorry

end solution_l499_499363


namespace largest_N_not_payable_l499_499902

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499902


namespace good_permutations_bound_l499_499451

theorem good_permutations_bound (n : ℕ) : 
  ∀ (is_good : list ℕ → Prop), 
    (∀ l, is_good l ↔ (∃ (s : finset ℕ), s.card = 10 ∧ ∀ (l' : list ℕ), (l'.length = 10 ∧ l.subperm l' ∧ l'.sorted (≤).not) → ¬ is_good l)) → 
    #{l : list ℕ // l.perm (list.range n) ∧ is_good l} ≤ 81^n :=
by
  intro is_good hg
  sorry

end good_permutations_bound_l499_499451


namespace cos_alpha_half_l499_499312

theorem cos_alpha_half (α : ℝ) (h : Real.cos (Real.pi + α) = -1/2) : Real.cos α = 1/2 := 
by 
  sorry

end cos_alpha_half_l499_499312


namespace derivative_exp_base_a_l499_499495

variable {x : ℝ} {a : ℝ}

theorem derivative_exp_base_a (h_pos : 0 < a) (h_ne_one : a ≠ 1) : (deriv (λ x : ℝ, a^x)) x = a^x * Real.log a := by
  sorry

end derivative_exp_base_a_l499_499495


namespace distance_from_center_to_line_l499_499934

theorem distance_from_center_to_line :
  let center : ℝ × ℝ := (0, 1)
  let line_x := 2
  let distance : ℝ := abs (line_x - center.1)
  distance = 2 :=
by
  unfold center line_x distance
  sorry

end distance_from_center_to_line_l499_499934


namespace tangent_line_at_x_eq_one_l499_499787

-- Define the function f
def f (a : ℝ) (x : ℝ) := x^4 + (a - 1) * x^3 + a

-- Define what it means for a function to be even
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Condition: f(x) is an even function
axiom h_even : even_function (f a)

-- Calculation of the derivative of the function f
noncomputable def derivative (f : ℝ → ℝ) (x : ℝ) := (deriv f x)

-- Specification of the tangent line equation
def tangent_line (f : ℝ → ℝ) (x : ℝ) (y' : ℝ) := y' = 4 * x - 2

theorem tangent_line_at_x_eq_one (a : ℝ) :
  even_function (f a) → tangent_line (f 1) 1 :=
begin
  sorry
end

end tangent_line_at_x_eq_one_l499_499787


namespace gcd_sum_l499_499095

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499095


namespace train_average_speed_l499_499054

open Real -- Assuming all required real number operations 

noncomputable def average_speed (distances : List ℝ) (times : List ℝ) : ℝ := 
  let total_distance := distances.sum
  let total_time := times.sum
  total_distance / total_time

theorem train_average_speed :
  average_speed [125, 270] [2.5, 3] = 71.82 := 
by 
  -- Details of the actual proof steps are omitted
  sorry

end train_average_speed_l499_499054


namespace shelly_thread_needed_l499_499912

def keychain_thread (classes: ℕ) (clubs: ℕ) (thread_per_keychain: ℕ) : ℕ := 
  let total_friends := classes + clubs
  total_friends * thread_per_keychain

theorem shelly_thread_needed : keychain_thread 6 (6 / 2) 12 = 108 := 
  by
    show 6 + (6 / 2) * 12 = 108
    sorry

end shelly_thread_needed_l499_499912


namespace gcd_sum_5n_6_n_eq_12_l499_499191

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499191


namespace part1_part2_l499_499780

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x
noncomputable def h (x : ℝ) : ℝ := -3 * Real.log x + x^3 + (2 * x^2 - 4 * x) * Real.exp x + 7

theorem part1 (a : ℤ) : 
  (∀ x, (a : ℝ) < x ∧ x < a + 5 → ∀ y, (a : ℝ) < y ∧ y < a + 5 → f x ≤ f y) →
  a = -6 ∨ a = -5 ∨ a = -4 :=
sorry

theorem part2 (x : ℝ) (hx : 0 < x) : 
  f x < h x :=
sorry

end part1_part2_l499_499780


namespace expenditure_may_to_july_l499_499500

theorem expenditure_may_to_july (spent_by_may : ℝ) (spent_by_july : ℝ) (h_may : spent_by_may = 0.8) (h_july : spent_by_july = 3.5) :
  spent_by_july - spent_by_may = 2.7 :=
by
  sorry

end expenditure_may_to_july_l499_499500


namespace triangle_free_graph_max_edges_l499_499021

open Finset

theorem triangle_free_graph_max_edges (n : ℕ) (h1 : ∀ (p1 p2 p3 : ℝ × ℝ), ¬ collinear {p1, p2, p3})
  (V : Finset (ℝ × ℝ)) (hV : V.card = n) (E : Finset (ℝ × ℝ) × (ℝ × ℝ)) (k : ℕ) (hE : E.card = k) :
  (∀ (v1 v2 v3 : ℝ × ℝ), {v1, v2, v3} ⊆ V → (v1, v2) ∈ E → (v2, v3) ∈ E → (v1, v3) ∈ E → false) →
  k ≤ ⌊n^2 / 4⌋ :=
sorry

end triangle_free_graph_max_edges_l499_499021


namespace derivative_at_minus_one_l499_499430

noncomputable def f (x : ℝ) : ℝ := f' 1 * x^2 - 1 / x

theorem derivative_at_minus_one :
  (derivative (f : ℝ → ℝ)) (-1) = 3 :=
sorry

end derivative_at_minus_one_l499_499430


namespace largest_domain_g_l499_499938

theorem largest_domain_g :
  (∃ g : ℝ → ℝ, (∀ x, x ≠ 0 → g(x) + g(1 / x^2) = x^2) ∧ 
               (∀ x, x ≠ 0 → x ∈ real_set → (1 / x^2) ∈ real_set) ∧
               real_set = {-1, 1}) :=
begin
  sorry
end

end largest_domain_g_l499_499938


namespace possible_value_of_a_l499_499376

theorem possible_value_of_a (a : ℕ) : (5 + 8 > a ∧ a > 3) → (a = 9 → True) :=
by
  intros h ha
  sorry

end possible_value_of_a_l499_499376


namespace determinant_roots_of_cubic_l499_499438

theorem determinant_roots_of_cubic (a b c r s t : ℝ)
  (h : ∀ x : ℝ, x^3 - r*x^2 + s*x + t = 0 → x = a ∨ x = b ∨ x = c) :
  (matrix.det ![![1 + a^2, 1, 1], ![1, 1 + b^2, 1], ![1, 1, 1 + c^2]]) = r^2 + s^2 - 2*t := 
sorry

end determinant_roots_of_cubic_l499_499438


namespace expression_value_l499_499656

theorem expression_value (x : ℝ) (h : x = 3 + 5 / (2 + 5 / x)) : x = 5 :=
sorry

end expression_value_l499_499656


namespace gcd_sum_divisors_eq_12_l499_499180

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499180


namespace solve_for_y_l499_499486

-- Define the conditions as Lean functions and statements
def is_positive (y : ℕ) : Prop := y > 0
def multiply_sixteen (y : ℕ) : Prop := 16 * y = 256

-- The theorem that states the value of y
theorem solve_for_y (y : ℕ) (h1 : is_positive y) (h2 : multiply_sixteen y) : y = 16 :=
sorry

end solve_for_y_l499_499486


namespace max_value_of_f_l499_499502

def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (π / 2 - x)

theorem max_value_of_f : ∃ x : ℝ, f x = 5 :=
sorry

end max_value_of_f_l499_499502


namespace calculate_expression_l499_499636

theorem calculate_expression :
  (|-2| + (sqrt 2 - 1)^0 - (-5) - (1/3)^(-1) = 5) := by
  sorry

end calculate_expression_l499_499636


namespace infinite_geometric_series_sum_l499_499670

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499670


namespace infinite_geometric_series_sum_l499_499672

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499672


namespace length_PR_l499_499389

-- Given definitions based on the conditions of the problem
variables {P Q R : Type} -- Points P, Q, and R.
variables (PR QR : ℝ) -- Lengths of the sides PR and QR.
variables (cosQ : ℝ) -- Cosine of angle Q.
variables [hyp : QR = Real.sqrt 145] [cos_eq : cosQ = 8 * Real.sqrt 145 / 145]

-- Statement we want to prove
theorem length_PR (h_cos : cosQ = PR / QR) : PR = 8 :=
by
  rw [cos_eq, hyp, Real.sqrt_mul_self, mul_div_cancel]
  sorry

end length_PR_l499_499389


namespace range_of_f_on_interval_l499_499654

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem range_of_f_on_interval :
  set.Icc (1/2 : ℝ) (2/3 : ℝ) = set.range (λ x, f x) ∩ set.Icc (2 : ℝ) (4 : ℝ) := by
  sorry

end range_of_f_on_interval_l499_499654


namespace find_even_and_mono_increasing_function_l499_499063

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_mono_increasing (f : ℝ → ℝ) (a b : ℝ) (h : a < b) : Prop :=
  f a ≤ f b

def problem_conditions :=
  let f1 := λ x : ℝ, x^3
  let f2 := λ x : ℝ, abs x + 1
  let f3 := λ x : ℝ, -x^2 + 1
  let f4 := λ x : ℝ, x^(-2)
  (is_even f1, is_mono_increasing f1 0 1 sorry, -- just placeholder values for monotonic check
   is_even f2, is_mono_increasing f2 0 1 sorry,
   is_even f3, is_mono_increasing f3 0 1 sorry,
   is_even f4, is_mono_increasing f4 0 1 sorry)

theorem find_even_and_mono_increasing_function :
  ∃ f ∈ { (λ x : ℝ, abs x + 1) }, is_even f ∧ is_mono_increasing f 0 1 sorry :=
by
  sorry

end find_even_and_mono_increasing_function_l499_499063


namespace cubic_function_root_form_l499_499370

theorem cubic_function_root_form (a b c d x1 x2 x3 x : ℝ) 
  (h: g x = a * x^3 + b * x^2 + c * x + d) 
  (hx1: g x1 = 0) 
  (hx2: g x2 = 0) 
  (hx3: g x3 = 0) : 
  g x = a * (x - x1) * (x - x2) * (x - x3) := 
sorry

end cubic_function_root_form_l499_499370


namespace largest_unpayable_soldo_l499_499894

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499894


namespace gcd_values_sum_l499_499080

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499080


namespace proj_matrix_inv_is_zero_l499_499423

open Matrix

variables {R : Type*} [Field R]
variables v : Matrix (Fin 3) (Fin 1) R := ![![1], ![2], ![2]]
def P : Matrix (Fin 3) (Fin 3) R := v ⬝ (v.transpose)

theorem proj_matrix_inv_is_zero : det P = 0 → P⁻¹ = 0 :=
by
  intro h
  sorry

end proj_matrix_inv_is_zero_l499_499423


namespace correct_statement_l499_499554

-- Conditions definitions
def coin_probability : Prop :=
  ∀ (n : ℕ), n > 0 → (probability of heads after n tosses ≈ 0.5)

def data_set : List ℕ := [2, 2, 3, 6]

def sampling_survey_method : Prop :=
  "Sampling survey method is appropriate to understand the low-carbon lifestyle of the entire city."

def student_variances : Prop :=
  ∀ (A B : Type) (mean_A mean_B : ℕ) (variance_A variance_B : ℕ),
  mean_A = 90 → mean_B = 90 →
  variance_A = 5 → variance_B = 12 →
  (variance A > variance B) → (scores of student A are more stable)

-- Theorem stating option C is correct
theorem correct_statement :
  sampling_survey_method :=
sorry

end correct_statement_l499_499554


namespace area_of_rectangle_KLMJ_l499_499266

theorem area_of_rectangle_KLMJ 
  (A B C D E F G H I : Type) 
  (AB AC : ℝ) 
  (h₁: AB = 3)
  (h₂: AC = 4)
  (h₃: square (ABED : set (A × B × E × D)))
  (h₄: square (ACHI : set (A × C × H × I)))
  (h₅: square (BCGF : set (B × C × G × F)))
  (h₆: D ∈ rectangle KLMJ)
  (h₇: E ∈ rectangle KLMJ)
  (h₈: F ∈ rectangle KLMJ)
  (h₉: G ∈ rectangle KLMJ)
  (h₁₀: H ∈ rectangle KLMJ)
  (h₁₁: I ∈ rectangle KLMJ) :
  area_of_rectangle KLMJ = 110 :=
sorry

end area_of_rectangle_KLMJ_l499_499266


namespace amount_due_years_l499_499949

noncomputable def years_due (PV FV : ℝ) (r : ℝ) : ℝ :=
  (Real.log (FV / PV)) / (Real.log (1 + r))

theorem amount_due_years : 
  years_due 200 242 0.10 = 2 :=
by
  sorry

end amount_due_years_l499_499949


namespace car_downhill_speed_l499_499994

/-- A car travels uphill at 30 km/hr and at a certain speed downhill. 
It goes 100 km uphill and 50 km downhill. 
The average speed of the car is 36 km/hr. 
What is the speed of the car when it travels downhill? --/
theorem car_downhill_speed : 
  ∃ v : ℚ, 36 = 150 / ((100 / 30) + (50 / v)) → v = 60 :=
begin
  sorry
end

end car_downhill_speed_l499_499994


namespace a_plus_d_eq_five_l499_499649

theorem a_plus_d_eq_five (a b c d k : ℝ) (hk : 0 < k) 
  (h1 : a + b = 11) 
  (h2 : b^2 + c^2 = k) 
  (h3 : b + c = 9) 
  (h4 : c + d = 3) : 
  a + d = 5 :=
by
  sorry

end a_plus_d_eq_five_l499_499649


namespace find_C_coordinates_l499_499648

variable (A B D C : Point)
variable (AB AC : Real)

def is_midpoint (P Q R : Point) : Prop :=
  ∃ D : Point, D.x = (Q.x + R.x) / 2 ∧ D.y = (Q.y + R.y) / 2

noncomputable def pointC_coordinates : Point :=
{ x := -1,
  y := 6 }

theorem find_C_coordinates : 
  let A := { x := 10, y := 10}
  let B := { x := 1, y := -4}
  let D := { x := 0, y := 1}
  AB = AC ∧ is_midpoint D B C → C = pointC_coordinates :=
by
  sorry

end find_C_coordinates_l499_499648


namespace bobby_books_count_l499_499631

variable (KristiBooks BobbyBooks : ℕ)

theorem bobby_books_count (h1 : KristiBooks = 78) (h2 : BobbyBooks = KristiBooks + 64) : BobbyBooks = 142 :=
by
  sorry

end bobby_books_count_l499_499631


namespace green_balls_in_bag_l499_499026

-- Given conditions
def number_of_blue_balls : ℕ := 8

def probability_blue_ball : ℚ := 1 / 3

-- Define the number of green balls
noncomputable def number_of_green_balls (total_balls green_balls blue_balls : ℕ) : ℕ :=
  total_balls - blue_balls

-- The proof problem
theorem green_balls_in_bag :
  ∀ g : ℕ, (number_of_blue_balls / (number_of_blue_balls + g) = probability_blue_ball) → g = 16 :=
by 
  intros g h,
  have : 8/(8 + g) = 1/3 := h,
  sorry  -- proof steps omitted

end green_balls_in_bag_l499_499026


namespace max_largest_integer_l499_499929

theorem max_largest_integer (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 100) 
  (h_distinct : ∀ (x ∈ S) (y ∈ S), x ≠ y) : ∃ x ∈ S, x = 55 :=
begin
  sorry
end

end max_largest_integer_l499_499929


namespace difference_is_sixty_l499_499001

-- Define percentage functions and constants
def pct (p x : ℝ) : ℝ := (p / 100) * x

-- Given conditions
def seventy_five_pct_of_480 : ℝ := pct 75 480
def three_fifths_of_twenty_pct_of_2500 : ℝ := (3 / 5) * pct 20 2500

-- Define the problem to prove
theorem difference_is_sixty : seventy_five_pct_of_480 - three_fifths_of_twenty_pct_of_2500 = 60 :=
by sorry

end difference_is_sixty_l499_499001


namespace probability_event_A_probability_event_B_expected_value_ξ_l499_499046

noncomputable def P (A : Prop) : ℝ := sorry
noncomputable def E (ξ : ℝ) : ℝ := sorry

variables (students : Finset ℕ) (cities : Finset ℕ) (teacher : ℕ)
variables (num_students : ℕ := 6) (num_cities : ℕ := 3)

-- Defining Event A
def Event_A (student_a teacher city : ℕ) : Prop :=
student_a ∈ cities ∧ teacher ∈ cities ∧ city ∈ cities

-- Defining Event B
def Event_B (student_a student_b city : ℕ) : Prop :=
student_a ∈ cities ∧ student_b ∈ cities ∧ city ∈ cities

-- Defining the random variable ξ
def ξ (student_a student_b teacher : ℕ) : ℕ :=
if Event_A student_a teacher (students.card) then 1 else 0 +
if Event_B student_a student_b (students.card) then 1 else 0

-- Lean statements to show the probabilities and expected value
theorem probability_event_A (student_a teacher : ℕ) (h : student_a ∈ students ∧ teacher ∈ students) :
  P (Event_A student_a teacher (students.card div num_cities)) = 1 / 3 := sorry

theorem probability_event_B (student_a student_b : ℕ) (h : student_a ∈ students ∧ student_b ∈ students) :
  P (Event_B student_a student_b (students.card div num_cities)) = 1 / 5 := sorry

theorem expected_value_ξ (student_a student_b teacher : ℕ) :
  E (ξ student_a student_b teacher) = 8 / 15 := sorry

end probability_event_A_probability_event_B_expected_value_ξ_l499_499046


namespace geometric_series_sum_l499_499677

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499677


namespace ellipse_proof_l499_499764

-- Defining the properties of the ellipse
def ellipse_center_origin : Prop :=
  ∀ x y : ℝ, x^2 / 2 + y^2 = 1

-- Properties of eccentricity and the perpendicularity condition
def ellipse_properties : Prop :=
  let e := Real.sqrt 2 / 2
  ∧ ∃ a b c : ℝ, a = Real.sqrt 2 ∧ b = 1 ∧ c = 1
  ∧ 2 * b^2 / a = Real.sqrt 2

-- Properties of intersection points and the maximization condition
def intersection_and_max_area : Prop :=
  let circle_eq := ∀ x y : ℝ, x^2 + y^2 = 1
  ∧ let line_intersections := ∃ k m : ℝ, ∃ x y : ℝ, y = k * x + m
  ∧ let distance : real := Real.sqrt 2 / 2
  ∧ let max_area_cond : Prop :=
      let angle_MAX := Real.pi / 2 -- Since sin(angle) is maximized
      ∧ let EF := Real.sqrt 2 = Real.sqrt 2
  ∧ ∃ GH, let range := Real.sqrt 3 ≤ GH ∧ GH ≤ 2

-- Lean statement to prove both parts
theorem ellipse_proof : ellipse_center_origin ∧ ellipse_properties → intersection_and_max_area := by
  sorry

end ellipse_proof_l499_499764


namespace gcd_sum_l499_499166

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499166


namespace expression_simplification_l499_499556

variable (x : ℝ)

-- Define the expression as given in the problem
def Expr : ℝ := (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 3)

-- Lean statement to verify that the expression simplifies to the given polynomial
theorem expression_simplification : Expr x = 6 * x^3 - 16 * x^2 + 43 * x - 70 := by
  sorry

end expression_simplification_l499_499556


namespace original_two_digit_number_l499_499402

theorem original_two_digit_number (n : ℕ) (a b : ℕ) (h1 : n = 10 * a + b)
                                     (h2 : 86.9 = n + (a + b / 10)) : n = 79 :=
sorry

end original_two_digit_number_l499_499402


namespace chinese_remainder_theorem_l499_499441

theorem chinese_remainder_theorem
  (k : ℕ)
  (m : Fin k → ℕ)
  (m_coprime : ∀ (i j : Fin k), i ≠ j → Nat.coprime (m i) (m j))
  (a : Fin k → ℤ)
  (product_m : ℕ := (Finset.univ.prod m)) :
  ∃ x : ℤ, 
    (∀ i : Fin k, x % m i = a i % m i) ∧
    (∀ x1 x2 : ℤ, 
      (∀ i : Fin k, x1 % m i = a i % m i ∧ x2 % m i = a i % m i) → 
      x1 % product_m = x2 % product_m) :=
sorry

end chinese_remainder_theorem_l499_499441


namespace sum_of_gcd_values_l499_499113

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499113


namespace correct_mark_l499_499042

def correct_mark_wrong_entered (y : ℕ) (n : ℕ) (increased_avg : ℕ) : ℕ :=
  y - increased_avg

theorem correct_mark :
  let y := 73 in
  let n := 56 in
  let increased_avg := n / 2 in
  correct_mark_wrong_entered y n increased_avg = 45 :=
by 
  let y := 73 
  let n := 56 
  let increased_avg := n / 2 
  show correct_mark_wrong_entered y n increased_avg = 45
  sorry

end correct_mark_l499_499042


namespace team_formation_l499_499878

def nat1 : ℕ := 7  -- Number of natives who know mathematics and physics
def nat2 : ℕ := 6  -- Number of natives who know physics and chemistry
def nat3 : ℕ := 3  -- Number of natives who know chemistry and mathematics
def nat4 : ℕ := 4  -- Number of natives who know physics and biology

def totalWaysToFormTeam (n1 n2 n3 n4 : ℕ) : ℕ := (n1 + n2 + n3 + n4).choose 3
def waysFromSameGroup (n : ℕ) : ℕ := n.choose 3

def waysFromAllGroups (n1 n2 n3 n4 : ℕ) : ℕ := (waysFromSameGroup n1) + (waysFromSameGroup n2) + (waysFromSameGroup n3) + (waysFromSameGroup n4)

theorem team_formation : totalWaysToFormTeam nat1 nat2 nat3 nat4 - waysFromAllGroups nat1 nat2 nat3 nat4 = 1080 := 
by
    sorry

end team_formation_l499_499878


namespace gcd_sum_5n_plus_6_l499_499206

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499206


namespace gcd_sum_divisors_eq_12_l499_499168

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499168


namespace largest_number_l499_499006

theorem largest_number (d1 d2 d3 d4 d5 : ℕ) 
  (H_d1 : d1 = 2) (H_d2 : d2 = 3) (H_d3 : d3 = 4) (H_d4 : d4 = 5) (H_d5 : d5 = 6) :
  0.9 + (d2 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5)) >
  d1 * 10^(-1) + 9 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5) ∧ 
  0.9 + (d2 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5)) >
  d1 * 10^(-1) + d2 * 10^(-2) + 9 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5) ∧ 
  0.9 + (d2 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5)) >
  d1 * 10^(-1) + d2 * 10^(-2) + d3 * 10^(-3) + 9 * 10^(-4) + d5 * 10^(-5) ∧ 
  0.9 + (d2 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + d5 * 10^(-5)) >
  d1 * 10^(-1) + d2 * 10^(-2) + d3 * 10^(-3) + d4 * 10^(-4) + 9 * 10^(-5) :=
sorry

end largest_number_l499_499006


namespace sum_of_gcd_values_l499_499236

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499236


namespace log_sin_eq_l499_499364

theorem log_sin_eq (b x a : ℝ) (h1 : b > 1) (h2 : tan x > 0)
  (h3 : log b (tan x) = a)
  (h4 : tan x = sin x / cos x)
  (h5 : cos x ^ 2 = 1 / (1 + (tan x)^2)) :
  log b (sin x) = a - (1 / 2) * log b (1 + (b ^ a)^2) :=
sorry

end log_sin_eq_l499_499364


namespace count_integers_between_2345_and_2460_with_increasing_digits_l499_499797

theorem count_integers_between_2345_and_2460_with_increasing_digits : 
     (finset.card (finset.filter (λ n : ℕ, 2345 ≤ n ∧ n ≤ 2460 ∧ (let ds := (n.digits 10) in (list.nodup ds ∧ ds = ds.sorted (≤)))) (finset.range (2461)))) = 19 :=
by { sorry }

end count_integers_between_2345_and_2460_with_increasing_digits_l499_499797


namespace keystone_arch_angle_correct_l499_499281

-- Definitions based on conditions
def is_keystone_arch (T : Type) [Inhabited T] : Prop :=
  ∃ (trapezoids : Fin 12 → T), 
    (∀ i, -- Each trapezoid is an isosceles trapezoid
      let t := trapezoids i in
      is_isosceles_trapezoid t ∧
      -- Non-parallel sides meet at the center of the circular arrangement
      non_parallel_sides_meet_at_center t) ∧
    -- Bottom sides of the two end trapezoids are horizontal
    bottom_sides_horizontal trapezoids

-- Prove the extreme angle measure given the isosceles keystone arch structure
def keystone_arch_large_angle (x : ℝ) : Prop :=
  is_keystone_arch ℝ → x = 97.5

-- Lean statement expressing the proof problem
theorem keystone_arch_angle_correct (x : ℝ) : keystone_arch_large_angle x :=
  by sorry

end keystone_arch_angle_correct_l499_499281


namespace unique_real_root_fx_less_x_when_x_greater_alpha_bounded_difference_l499_499337

variable {f : ℝ → ℝ}
variable {α : ℝ}
variable {M : Set ℝ}

-- Conditions
axiom h1 : ∀ x ∈ M, 0 < f'(x) ∧ f'(x) < 1
axiom h2 : f α = α
axiom h3 : ∀ (a b ∈ M), ∃ x ∈ M, f(b) - f(a) = (b - a) * f''(x)

-- (I) The equation f(x) = x has a unique real root α
theorem unique_real_root : ∃! α, f α = α := sorry

-- (II) When x > α, it always holds that f(x) < x
theorem fx_less_x_when_x_greater_alpha (x : ℝ) (hx : x > α) : f(x) < x := sorry

-- (III) For any x₁, x₂, if |x₁ - α| < 2 and |x₂ - α| < 2, prove that |f(x₁) - f(x₂)| < 4
theorem bounded_difference (x₁ x₂ : ℝ) (h₁ : |x₁ - α| < 2) (h₂ : |x₂ - α| < 2) : |f(x₁) - f(x₂)| < 4 := sorry

end unique_real_root_fx_less_x_when_x_greater_alpha_bounded_difference_l499_499337


namespace total_wet_surface_area_l499_499015

def length : ℝ := 8
def width : ℝ := 4
def depth : ℝ := 1.25

theorem total_wet_surface_area : length * width + 2 * (length * depth) + 2 * (width * depth) = 62 :=
by
  sorry

end total_wet_surface_area_l499_499015


namespace infinite_geometric_series_sum_l499_499695

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499695


namespace complete_graph_color_removal_l499_499527

-- Definitions based on conditions
variables (V : Type) [fintype V] [decidable_eq V] (E : V → V → Prop)

noncomputable def K50 : Type := fin 50

namespace GraphTheoryProof

-- Define the complete graph with 50 vertices and edges colored by one of three colors
def is_complete_graph (G : Type) := ∀ (u v : G), u ≠ v → E u v

-- Define the colors
inductive Color
| C1
| C2
| C3

-- Define edges are colored
variables (coloring : K50 → K50 → Color)

-- Desired property after removing edges of one color
def connected_after_removing_color (c : Color) : Prop :=
  ∀ (u v : K50), ∃ (path : list K50),
    ∀ (i : fin (path.length - 1)),
      coloring (path.nth_le i sorry) (path.nth_le (i + 1) sorry) ≠ c

-- The main theorem
theorem complete_graph_color_removal :
  (is_complete_graph K50) →
  (∃ c : Color, connected_after_removing_color coloring c) :=
begin
  intros is_complete,
  sorry
end

end GraphTheoryProof

end complete_graph_color_removal_l499_499527


namespace commission_excess_500_l499_499606

-- Define the conditions as variables and constants
variable (commission_first_500 : ℕ := 20)  -- 20%
variable (total_sale_amount : ℕ := 800)
variable (total_commission_percent : ℚ := 31.25) -- 31.25%

-- Prove that the percentage of the commission for the amount in excess of $500 is 50%
theorem commission_excess_500 : 
  let commission_first_500_dollar := 0.20 * 500
      sale_excess := total_sale_amount - 500
      total_commission_dollar := 0.3125 * total_sale_amount in
  (100 * ((total_commission_dollar - commission_first_500_dollar) / 300)) = 50 := 
by
  let commission_first_500_dollar := 0.20 * 500
  let sale_excess := total_sale_amount - 500
  let total_commission_dollar := 0.3125 * total_sale_amount
  have intermediate := ((total_commission_dollar - commission_first_500_dollar) / 300)
  show 100 * intermediate = 50
  sorry

end commission_excess_500_l499_499606


namespace factory_produces_50_candies_per_hour_l499_499492

-- Declare the necessary constants and definitions based on the conditions
constant total_candies : ℕ := 4000
constant days : ℕ := 8
constant hours_per_day : ℕ := 10

-- Define the total hours worked
def total_hours : ℕ := days * hours_per_day

-- Define the rate of candy production per hour
def candies_per_hour : ℕ := total_candies / total_hours

-- State the theorem to be proved
theorem factory_produces_50_candies_per_hour : candies_per_hour = 50 := by
  sorry

end factory_produces_50_candies_per_hour_l499_499492


namespace minimum_omega_l499_499346

theorem minimum_omega (ω : ℝ) (k : ℤ) (h : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + (π / 6) = k * π + (π / 2)) : ω = 4 :=
by
  existsi (0 : ℤ)
  sorry

end minimum_omega_l499_499346


namespace probability_two_red_balls_given_one_white_l499_499379

theorem probability_two_red_balls_given_one_white :
  let total_ways := Nat.choose 10 3 - Nat.choose 5 3,
      favorable_ways := Nat.choose 5 2 * Nat.choose 5 1,
      probability := favorable_ways / total_ways
  in probability = (5 : ℚ) / 11 :=
by
  let total_ways := Nat.choose 10 3 - Nat.choose 5 3
  let favorable_ways := Nat.choose 5 2 * Nat.choose 5 1
  let probability := (favorable_ways : ℚ) / total_ways
  show probability = (5 : ℚ) / 11
  sorry

end probability_two_red_balls_given_one_white_l499_499379


namespace isosceles_triangle_perimeter_l499_499324

-- Let there be an isosceles triangle ABC with sides a, a, and b such that a = 3 and b = 6.
theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 3 ∧ b = 6) (h₂ : a ≠ b) :
  let p := if (a + a > b ∧ a + b > a ∧ a + b > a) 
           then (2 * a + b)
           else (2 * b + a)
  in p = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l499_499324


namespace membership_fee_problem_l499_499661

theorem membership_fee_problem :
  ∃ (n : ℕ) (f : ℕ), n * f = 300737 ∧ n ≤ 500 ∧ f < 10000 ∧ n = 311 ∧ f = 967 :=
by {
  use [311, 967],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { linarith },
  split,
  { refl },
  { refl }
}

end membership_fee_problem_l499_499661


namespace seq_ratio_l499_499761

theorem seq_ratio (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n, a n * a (n + 1) = 2^n) : 
  a 7 / a 3 = 4 := 
by 
  sorry

end seq_ratio_l499_499761


namespace calculate_expression_l499_499637

theorem calculate_expression :
  (|-2| + (sqrt 2 - 1)^0 - (-5) - (1/3)^(-1) = 5) := by
  sorry

end calculate_expression_l499_499637


namespace infinite_geometric_series_sum_l499_499725

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499725


namespace geometric_series_sum_l499_499690

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499690


namespace points_in_one_square_l499_499323

open Set

/-- Given: Any three points from a set can be translated within a square with vertices at (0, 2), (2, 0), (0, -2), and (-2, 0).
    Show that: All points from the set can be translated to fit within the same square. -/
theorem points_in_one_square (S : Set (ℝ × ℝ))
  (h : ∀ (p1 p2 p3 : ℝ × ℝ), {p1, p2, p3} ⊆ S -> 
    ∃ v : ℝ × ℝ, (λ p, (p.1 + v.1, p.2 + v.2)) '' {p1, p2, p3} ⊆ {(0,2), (2,0), (0,-2), (-2,0)}) : 
  ∃ v : ℝ × ℝ, (λ p, (p.1 + v.1, p.2 + v.2)) '' S ⊆ {(0,2), (2,0), (0,-2), (-2,0)} :=
sorry

end points_in_one_square_l499_499323


namespace midpoint_M_values_y_lambda_l499_499355

-- Define given conditions
def vec_AB : ℝ × ℝ := (4, 3)
def vec_AD : ℝ × ℝ := (-3, -1)
def point_A : ℝ × ℝ := (-1, -2)

-- Define the question 1
theorem midpoint_M (B D M : ℝ × ℝ) 
  (hB : B.1 = -point_A.1 + vec_AB.1 ∧ B.2 = -point_A.2 + vec_AB.2)
  (hD : D.1 = -point_A.1 + vec_AD.1 ∧ D.2 = -point_A.2 + vec_AD.2)
  (hM : M.1 = (B.1 + D.1) / 2 ∧ M.2 = (B.2 + D.2) / 2) :
  M = (-1/2, -1) :=
by
  sorry

-- Define the question 2
theorem values_y_lambda (B D : ℝ × ℝ) (P : ℝ × ℝ) (λ : ℝ) (y : ℝ)
  (hB : B.1 = -point_A.1 + vec_AB.1 ∧ B.2 = -point_A.2 + vec_AB.2)
  (hD : D.1 = -point_A.1 + vec_AD.1 ∧ D.2 = -point_A.2 + vec_AD.2)
  (hP : P.1 = 2 ∧ P.2 = y)
  (hPB_eq_λBD : (P.1 - B.1, P.2 - B.2) = (λ * (D.1 - B.1), λ * (D.2 - B.2))) :
  y = 11/7 ∧ λ = -1/7 :=
by
  sorry

end midpoint_M_values_y_lambda_l499_499355


namespace sum_gcd_values_l499_499255

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499255


namespace number_of_roses_l499_499617

theorem number_of_roses (v c n : ℕ) (h_v : v = 6) (h_c : c = 7) (h_n : n = 9) :
  (n * v) - c = 47 :=
by
  rw [h_v, h_c, h_n]
  norm_num
  done

end number_of_roses_l499_499617


namespace probability_second_or_third_class_l499_499907

noncomputable def P (A B C : ℝ) (hA : A = 0.65) (hB : B = 0.2) (hC : C = 0.1) : ℝ :=
let D := B + C in
D

theorem probability_second_or_third_class (A B C D : ℝ) 
  (hA : A = 0.65) (hB : B = 0.2) (hC : C = 0.1) (hD : D = B + C) :
  D = 0.35 :=
by
  rw [hB, hC, hD]
  sorry

end probability_second_or_third_class_l499_499907


namespace infinite_geometric_series_sum_l499_499674

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499674


namespace largest_perfect_square_factor_of_3780_l499_499970

theorem largest_perfect_square_factor_of_3780 :
  ∃ m : ℕ, (∃ k : ℕ, 3780 = k * m * m) ∧ m * m = 36 :=
by
  sorry

end largest_perfect_square_factor_of_3780_l499_499970


namespace gcd_sum_l499_499167

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499167


namespace bailey_total_spending_l499_499629

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end bailey_total_spending_l499_499629


namespace complement_of_A_with_respect_to_U_l499_499443

open Set

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}
def complement_U_A : Set ℕ := {4, 6}

theorem complement_of_A_with_respect_to_U :
  U \ A = complement_U_A := by
  sorry

end complement_of_A_with_respect_to_U_l499_499443


namespace number_of_true_propositions_is_3_l499_499065

-- Define propositions as per the equivalency criteria.
def proposition_1 (a b c : ℝ) : Prop :=
  a > b → ac² > bc²

def foci_of_ellipse : Prop :=
  let F1 := (0, 0)
  let F2 := (10, 0)
  ∃ (A B : ℝ × ℝ), (A, B).1 = F1 ∧ perimeter_of_triangle (A, B, F2) = 20

def logical_proposition_3 (p q : Prop) : Prop :=
  (¬p ∧ (p ∨ q)) → q

def quadratic_proposition : Prop :=
  (∃ x : ℝ, x² + x + 1 < 0) ↔ (∀ x : ℝ, x² + x + 1 ≥ 0)

-- Main theorem stating the number of true propositions is 3.
theorem number_of_true_propositions_is_3 :
  let proposition_1_correct := ¬proposition_1 a b c
  let proposition_2_correct := foci_of_ellipse
  let proposition_3_correct := logical_proposition_3 p q
  let proposition_4_correct := quadratic_proposition
  (proposition_1_correct, proposition_2_correct, proposition_3_correct, proposition_4_correct)
    → 3 = count_true_propositions [
      proposition_1_correct, proposition_2_correct, proposition_3_correct, proposition_4_correct ] := 
sorry

end number_of_true_propositions_is_3_l499_499065


namespace gcd_values_sum_l499_499076

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499076


namespace gcd_sum_5n_6_n_eq_12_l499_499192

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499192


namespace cost_for_33_people_employees_for_14000_cost_l499_499997

-- Define the conditions for pricing
def price_per_ticket (x : Nat) : Int :=
  if x ≤ 30 then 400
  else max 280 (400 - 5 * (x - 30))

def total_cost (x : Nat) : Int :=
  x * price_per_ticket x

-- Problem Part 1: Proving the total cost for 33 people
theorem cost_for_33_people :
  total_cost 33 = 12705 :=
by
  sorry

-- Problem Part 2: Given a total cost of 14000, finding the number of employees
theorem employees_for_14000_cost :
  ∃ x : Nat, total_cost x = 14000 ∧ price_per_ticket x ≥ 280 :=
by
  sorry

end cost_for_33_people_employees_for_14000_cost_l499_499997


namespace gcd_sum_l499_499136

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499136


namespace sum_of_gcd_values_l499_499217

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499217


namespace largest_N_cannot_pay_exactly_without_change_l499_499886

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499886


namespace distance_BC_l499_499535

-- Definitions of the conditions
def speed_AC : ℝ := 75  -- Speed from A to C in km/h
def speed_CB : ℝ := 145  -- Speed from C to B in km/h
def total_time_AB : ℝ := 4.8  -- Total time in hours for the journey from A to B

def speed_return_AB : ℝ := 100  -- Average speed of the return trip from A to B
def time_BC_return : ℝ := 2  -- Time taken for the return trip from B to C in hours
def speed_CA_return : ℝ := 70  -- Average speed from C to A on the return trip in km/h

-- Variables representing distances
variable (x y : ℝ)

-- Equations based on the conditions
def eq1 : Prop := (y / speed_AC + x / speed_CB = total_time_AB)
def eq2 : Prop := ((x + y) / speed_return_AB = time_BC_return + y / speed_CA_return)

-- Statement to be proven
theorem distance_BC : ∃ x y : ℝ, eq1 ∧ eq2 ∧ x = 290 := 
by 
sory

end distance_BC_l499_499535


namespace probability_not_all_even_l499_499971

/-- The probability that when rolling five fair 6-sided dice, 
    they won't all show an even number is 7533/7776. -/
theorem probability_not_all_even :
  let outcomes := (Finset.range 6).pow 5
  let even_outcomes := ((Finset.filter (λ x : ℕ, x % 2 = 0) (Finset.range 6)).pow 5)
  let p_all_even := even_outcomes.card / outcomes.card.to_real
  p_not_all_even = 1 - p_all_even := 7533 / 7776 := by sorry

end probability_not_all_even_l499_499971


namespace sum_gcd_possible_values_eq_twelve_l499_499243

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499243


namespace gcd_sum_l499_499094

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499094


namespace gcd_sum_5n_plus_6_n_l499_499142

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499142


namespace area_inside_circle_but_outside_triangle_MDN_l499_499843

variables (A B C D M N P Q : Type) [field A] [field B] [field C] [field D] [field M] [field N] [field P] [field Q]
variables (AB BC CD DA : ℝ)
variables (AB_length BC_length : ℝ) (hAB : AB_length = 2) (hBC : BC_length = 4)
variables (M_mid_BC N_mid_CD P_mid_AD : Prop) (Q_mid_MP : Prop)

-- Define the coordinates
def point_A := (0, 0) : ℝ × ℝ
def point_B := (2, 0) : ℝ × ℝ
def point_C := (2, 4) : ℝ × ℝ
def point_D := (0, 4) : ℝ × ℝ
def point_M := (2, 2) : ℝ × ℝ
def point_N := (1, 4) : ℝ × ℝ
def point_P := (0, 2) : ℝ × ℝ
def point_Q := (1, 2) : ℝ × ℝ

-- Radius of the circle
def radius_QM := 1

-- Area calculations
def area_triangle_MDN : ℝ := 1
def area_circle : ℝ := real.pi
def desired_area := area_circle - area_triangle_MDN

theorem area_inside_circle_but_outside_triangle_MDN :
  desired_area = real.pi - 1 :=
by
  sorry

end area_inside_circle_but_outside_triangle_MDN_l499_499843


namespace basketball_free_throws_l499_499508

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = b) 
  (h3 : 2 * a + 3 * b + x = 73) : 
  x = 10 := 
by 
  sorry -- The actual proof is omitted as per the requirements.

end basketball_free_throws_l499_499508


namespace valid_list_count_correct_l499_499573

noncomputable def count_valid_lists : ℕ :=
  let balls := 15 in
  let choices := 4 in
  let first_choice := balls in
  let subsequent_choices := 14 in
  first_choice * subsequent_choices ^ (choices - 1)

theorem valid_list_count_correct :
  count_valid_lists = 41160 :=
by
  sorry

end valid_list_count_correct_l499_499573


namespace relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l499_499027

variable (x y : ℝ)

-- Assume the initial fuel and consumption rate
def initial_fuel : ℝ := 48
def consumption_rate : ℝ := 0.6

-- Define the fuel consumption equation
def fuel_equation (distance : ℝ) : ℝ := -consumption_rate * distance + initial_fuel

-- Theorem proving the fuel equation satisfies the specific conditions
theorem relationship_between_y_and_x :
  ∀ (x : ℝ), y = fuel_equation x :=
by
  sorry

-- Theorem proving the fuel remaining after traveling 35 kilometers
theorem fuel_remaining_after_35_kilometers :
  fuel_equation 35 = 27 :=
by
  sorry

-- Theorem proving the maximum distance the car can travel without refueling
theorem max_distance_without_refueling :
  ∃ (x : ℝ), fuel_equation x = 0 ∧ x = 80 :=
by
  sorry

end relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l499_499027


namespace freshman_class_students_l499_499924

theorem freshman_class_students :
  ∃ n : ℕ, n < 700 ∧
  n % 20 = 19 ∧ 
  n % 25 = 24 ∧
  n % 9 = 3 :=
by 
  use 399
  split
  · norm_num -- proofs that 399 < 700
  split
  · norm_num -- proofs that 399 % 20 = 19
  split
  · norm_num -- proofs that 399 % 25 = 24
  · norm_num -- proofs that 399 % 9 = 3
  sorry

end freshman_class_students_l499_499924


namespace y_squared_range_l499_499828

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) : 15 ≤ y^2 ∧ y^2 ≤ 25 :=
by
  sorry

end y_squared_range_l499_499828


namespace gcd_5n_plus_6_n_sum_l499_499109

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499109


namespace bird_families_flew_away_l499_499555

theorem bird_families_flew_away (original : ℕ) (left : ℕ) (flew_away : ℕ) (h1 : original = 67) (h2 : left = 35) (h3 : flew_away = original - left) : flew_away = 32 :=
by
  rw [h1, h2] at h3
  exact h3

end bird_families_flew_away_l499_499555


namespace gcd_sum_l499_499093

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499093


namespace domain_of_f_l499_499735

noncomputable def f (x : ℝ) := real.sqrt (4^x - 2^(x+1))

theorem domain_of_f :
  {x : ℝ | 4^x - 2^(x + 1) ≥ 0} = {x : ℝ | x ≥ 1} :=
by
  sorry

end domain_of_f_l499_499735


namespace cos_angle_difference_l499_499773

theorem cos_angle_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1): 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_angle_difference_l499_499773


namespace parallel_condition_perpendicular_condition_l499_499331

variable (m : ℝ)

def a := (m, 1)
def b := (1, 2)

noncomputable def is_parallel_to (x y : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, x = k • y

noncomputable def is_perpendicular_to (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

theorem parallel_condition : is_parallel_to (a m) b → m = 1 / 2 :=
by { intro h, sorry }

theorem perpendicular_condition : is_perpendicular_to (a m) b → m = -2 :=
by { intro h, sorry }

end parallel_condition_perpendicular_condition_l499_499331


namespace smallest_n_with_units_digit_and_reorder_l499_499303

theorem smallest_n_with_units_digit_and_reorder :
  ∃ n : ℕ, (∃ a : ℕ, n = 10 * a + 6) ∧ (∃ m : ℕ, 6 * 10^m + a = 4 * n) ∧ n = 153846 :=
by
  sorry

end smallest_n_with_units_digit_and_reorder_l499_499303


namespace matt_worked_more_minutes_l499_499456

-- Define the conditions as constants
def monday_minutes : ℕ := 450
def tuesday_minutes : ℕ := monday_minutes / 2
def wednesday_minutes : ℕ := 300

-- The statement to prove
theorem matt_worked_more_minutes :
  wednesday_minutes - tuesday_minutes = 75 :=
begin
  sorry, -- Proof placeholder
end

end matt_worked_more_minutes_l499_499456


namespace max_product_l499_499775

-- Problem statement: Define the conditions and the conclusion
theorem max_product (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 4) : mn ≤ 4 :=
by
  sorry -- Proof placeholder

end max_product_l499_499775


namespace compare_logs_l499_499813

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l499_499813


namespace percentage_increase_second_movie_l499_499530

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie_l499_499530


namespace sum_of_reciprocals_of_roots_l499_499744

theorem sum_of_reciprocals_of_roots (r1 r2 : ℚ) (h_sum : r1 + r2 = 17) (h_prod : r1 * r2 = 6) :
  1 / r1 + 1 / r2 = 17 / 6 :=
sorry

end sum_of_reciprocals_of_roots_l499_499744


namespace sum_gcd_possible_values_eq_twelve_l499_499245

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499245


namespace product_simplification_l499_499004

theorem product_simplification :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) = 7 :=
by
  sorry

end product_simplification_l499_499004


namespace sum_of_given_infinite_geometric_series_l499_499710

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499710


namespace gcd_5n_plus_6_n_sum_l499_499105

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499105


namespace difference_of_squares_l499_499983

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x + y = 15
def condition2 : Prop := x - y = 10

-- Goal to prove
theorem difference_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 150 := 
by sorry

end difference_of_squares_l499_499983


namespace calculate_expression_l499_499638

theorem calculate_expression :
  |-2| + (sqrt 2 - 1) ^ 0 - (-5) - (1 / 3) ^ (-1) = 5 := 
by
  -- Proof goes here
  sorry

end calculate_expression_l499_499638


namespace gcd_sum_5n_plus_6_n_l499_499150

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499150


namespace figure_perimeter_equals_26_l499_499908

noncomputable def rectangle_perimeter : ℕ := 26

def figure_arrangement (width height : ℕ) : Prop :=
width = 2 ∧ height = 1

theorem figure_perimeter_equals_26 {width height : ℕ} (h : figure_arrangement width height) :
  rectangle_perimeter = 26 :=
by
  sorry

end figure_perimeter_equals_26_l499_499908


namespace adam_initial_savings_l499_499059

theorem adam_initial_savings (additional_money : ℕ) (total_money : ℕ) (initial_money : ℕ) :
  additional_money = 13 → total_money = 92 → total_money - additional_money = initial_money → initial_money = 79 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end adam_initial_savings_l499_499059


namespace num_arrangements_5_people_num_arrangements_A_not_head_B_not_end_num_arrangements_A_B_next_to_each_other_num_arrangements_A_B_not_next_to_each_other_l499_499746

-- Definitions
def num_permutations (n : ℕ) : ℕ := n!

-- Prove that the number of permutations of 5 people is 120
theorem num_arrangements_5_people : num_permutations 5 = 120 := by
  sorry

-- Prove that the number of arrangements of 5 people where A is not at the head and B is not at the end is 78
theorem num_arrangements_A_not_head_B_not_end : 
  (num_permutations 4) + 3 * 3 * (num_permutations 3) = 78 := by
  sorry

-- Prove that the number of arrangements of 5 people where A and B must stand next to each other is 48
theorem num_arrangements_A_B_next_to_each_other : 
  (num_permutations 4) * (num_permutations 2) = 48 := by
  sorry

-- Prove that the number of arrangements of 5 people where A and B cannot stand next to each other is 72
theorem num_arrangements_A_B_not_next_to_each_other :
  (num_permutations 3) * (num_permutations 4-2) = 72 := by
  sorry

end num_arrangements_5_people_num_arrangements_A_not_head_B_not_end_num_arrangements_A_B_next_to_each_other_num_arrangements_A_B_not_next_to_each_other_l499_499746


namespace time_spent_cleaning_bathroom_l499_499873

-- Define the times spent on each task
def laundry_time : ℕ := 30
def room_cleaning_time : ℕ := 35
def homework_time : ℕ := 40
def total_time : ℕ := 120

-- Let b be the time spent cleaning the bathroom
variable (b : ℕ)

-- Total time spent on all tasks is the sum of individual times
def total_task_time := laundry_time + b + room_cleaning_time + homework_time

-- Proof that b = 15 given the total time
theorem time_spent_cleaning_bathroom (h : total_task_time = total_time) : b = 15 :=
by
  sorry

end time_spent_cleaning_bathroom_l499_499873


namespace min_distance_hyperbola_l499_499627

open Real

theorem min_distance_hyperbola 
(hyperbola_eq : ∀ x : ℝ, x > 0 → ∀ y : ℝ, y = 4 / x → True) 
(rect_area : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 4 ∧ ∃ x y : ℝ, x = a ∧ y = b ∧ y = 4 / x) 
: ∃ x : ℝ, x > 0 ∧ (x^2 + (4 / x)^2) = 2:= sorry

end min_distance_hyperbola_l499_499627


namespace angle_ABC_eq_60_l499_499420

theorem angle_ABC_eq_60
  (A B C I A' B' C' : Point)
  (h_incenter : is_incenter I A B C)
  (symm_A' : is_symmetric A' I (line_through B C))
  (symm_B' : is_symmetric B' I (line_through A C))
  (symm_C' : is_symmetric C' I (line_through A B))
  (circumcircle_passes_through_B : on_circumcircle B A' B' C') :
  angle_between B A C = 60 := 
sorry

end angle_ABC_eq_60_l499_499420


namespace favorite_song_not_fully_heard_probability_l499_499641

-- Define the basic properties and constraints
def is_favorite (song : ℕ) : Prop := song = 5
def total_songs : finset ℕ := finset.range 8
def first_n_minutes (n : ℕ) := finset.range n

-- Define the event of hearing the full favorite song in the first 7 minutes
def hear_favorite_in_first_7_min (order : list ℕ) : Prop :=
  ∃ i, i ∈ first_n_minutes 7 ∧ is_favorite (order.nth_le i (by linarith))

-- Definition of permutations
def permutations (l : list ℕ) : list (list ℕ) := list.permutations l

-- Calculate the probability
def probability (event : list ℕ → Prop) (space : list (list ℕ)) :=
  (list.count event space) / (space.length : ℝ)

-- Main theorem to prove
theorem favorite_song_not_fully_heard_probability :
  probability (λ order, ¬ hear_favorite_in_first_7_min order) (permutations (list.range 8)) = 6 / 7 :=
by
  -- Skip the proof as it's not required
  sorry

end favorite_song_not_fully_heard_probability_l499_499641


namespace sum_gcd_possible_values_eq_twelve_l499_499249

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499249


namespace Meryll_problem_solving_questions_l499_499447

variable (P : ℕ)

theorem Meryll_problem_solving_questions : 
  let n_mchoice := 35 in
  let pct_mchoice_written := (2 : ℚ) / 5 in
  let pct_psolving_written := (1 : ℚ) / 3 in
  let mchoice_written := pct_mchoice_written * n_mchoice in
  let psolving_written := pct_psolving_written * P in
  let total_questions_written := 31 in
  let mchoice_remaining := n_mchoice - mchoice_written in
  let psolving_remaining := P - psolving_written in
  mchoice_remaining + psolving_remaining = total_questions_written → P = 15 :=
by
  sorry

end Meryll_problem_solving_questions_l499_499447


namespace no_hundred_equilateral_triangles_in_convex_polygon_l499_499473

theorem no_hundred_equilateral_triangles_in_convex_polygon (P : Type*) [Polygon P] (h_convex : Convex P) :
  ¬(∃ (T : Finset Type*) (ht : ∀ t ∈ T, t = EquilateralTriangle), T.card = 100 ∧ (∀ (t₁ t₂ : Type*) (h₁ : t₁ ∈ T) (h₂ : t₂ ∈ T), Disjoint t₁ t₂)) :=
sorry

end no_hundred_equilateral_triangles_in_convex_polygon_l499_499473


namespace find_c_makes_f_odd_l499_499733

noncomputable def arctan (x : ℝ) : ℝ := sorry -- Assuming arctan is defined elsewhere

def f (x : ℝ) (c : ℝ) : ℝ := arctan ((2 - 2 * x) / (1 + 4 * x)) + c

theorem find_c_makes_f_odd :
  ∃ c : ℝ, (∀ x : ℝ, -1/4 < x ∧ x < 1/4 → f x c = -f (-x) c) :=
begin
  use -arctan 2,
  intros x hx,
  sorry -- proof to be finished
end

end find_c_makes_f_odd_l499_499733


namespace train_speed_is_60_kmph_l499_499614

-- Define the distance and time
def train_length : ℕ := 400
def bridge_length : ℕ := 800
def time_to_pass_bridge : ℕ := 72

-- Define the distances and calculations
def total_distance : ℕ := train_length + bridge_length
def speed_m_per_s : ℚ := total_distance / time_to_pass_bridge
def speed_km_per_h : ℚ := speed_m_per_s * 3.6

-- State and prove the theorem
theorem train_speed_is_60_kmph : speed_km_per_h = 60 := by
  sorry

end train_speed_is_60_kmph_l499_499614


namespace expenses_recorded_as_negative_l499_499931

/-*
  Given:
  1. The income of 5 yuan is recorded as +5 yuan.
  Prove:
  2. The expenses of 5 yuan are recorded as -5 yuan.
*-/

theorem expenses_recorded_as_negative (income_expenses_opposite_sign : ∀ (a : ℤ), -a = -a)
    (income_five_recorded_as_positive : (5 : ℤ) = 5) :
    (-5 : ℤ) = -5 :=
by sorry

end expenses_recorded_as_negative_l499_499931


namespace binom_computation_l499_499646

noncomputable def binom_real (n k : ℝ) : ℝ := real.factorial n / (real.factorial k * real.factorial (n - k))

theorem binom_computation :
  (binom_real (3/2) 2015 * 4 ^ 2015) / binom_real 4030 2015 = -1 / 4032 :=
by
  sorry

end binom_computation_l499_499646


namespace proof_problem_l499_499400

-- Define the conditions in Lean
def a : ℝ := 3
def cos_B : ℝ := -1 / 2

-- Assume b and c are real numbers such that b - c = 2
variables (b c : ℝ)
def condition_bc := b - c = 2

-- Define the proof targets
def target_b_c := b = 7 ∧ c = 5
def target_sin_BC := Real.sin (Real.arccos (-1 / 2) + Real.arccos (-1 / 2)) = (3 * Real.sqrt 3) / 14

-- The statement of the proof problem
theorem proof_problem (h_bc : condition_bc) (h_cos_B : cos_B = -1 / 2) : target_b_c ∧ target_sin_BC :=
by
  sorry

end proof_problem_l499_499400


namespace max_difference_condition_l499_499315

open Real

theorem max_difference_condition (a : ℕ → ℝ) (n : ℕ) (h : ∑ i in Finset.range (2*n - 1), (a (i+1) - a i)^2 = 1) :
  (∑ i in Finset.range n, a (n+i+1)) - (∑ i in Finset.range n, a (i+1)) ≤ sqrt(↑n * (2*(↑n)^2 + 1) / 3) :=
sorry

end max_difference_condition_l499_499315


namespace sum_of_three_consecutive_even_numbers_is_162_l499_499519

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l499_499519


namespace min_product_of_three_l499_499548

theorem min_product_of_three :
  ∀ (list : List Int), 
    list = [-9, -7, -1, 2, 4, 6, 8] →
    ∃ (a b c : Int), a ∈ list ∧ b ∈ list ∧ c ∈ list ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ (x y z : Int), x ∈ list → y ∈ list → z ∈ list → x ≠ y → y ≠ z → x ≠ z → x * y * z ≥ a * b * c) ∧
    a * b * c = -432 :=
by
  sorry

end min_product_of_three_l499_499548


namespace sum_of_gcd_values_l499_499214

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499214


namespace max_composite_difference_subset_l499_499642

/-- Maximum value of n such that choosing n numbers from {1, 2, ..., 2017} with the property that the difference between any two is composite. -/
theorem max_composite_difference_subset :
  ∃ (n : ℕ), n = 505 ∧ ∀ (S : finset ℕ), S.card = n →
    (∀ (a b ∈ S), a ≠ b → ∃ k, (a - b = 4 * k ∨ b - a = 4 * k)) :=
sorry

end max_composite_difference_subset_l499_499642


namespace find_values_and_properties_l499_499318

variable (f : ℝ → ℝ)

axiom f_neg1 : f (-1) = 2
axiom f_pos_x : ∀ x, x < 0 → f x > 1
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y

theorem find_values_and_properties :
  f 0 = 1 ∧
  f (-4) = 16 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (-4 * x^2) * f (10 * x) ≥ 1/16 ↔ x ≤ 1/2 ∨ x ≥ 2) :=
sorry

end find_values_and_properties_l499_499318


namespace ratio_intersections_l499_499319

variables (A B C D M1 N1 M2 N2 : Type) (a b : ℝ)
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space M1] [metric_space N1]
variables [metric_space M2] [metric_space N2]
variables (h_parallelogram : parallelogram A B C D)
variables (h1 : A.distance D = b)
variables (h2 : C.distance D = a)
variables (h3 : same_circle A D M1 N1)
variables (h4 : same_circle C D M2 N2)

theorem ratio_intersections (h5 : metric.ball B (B.distance M1).val = metric.ball B (B.distance M2).val) :
  (M1.distance N1) / (M2.distance N2) = b / a :=
by sorry

end ratio_intersections_l499_499319


namespace all_three_digits_same_two_digits_same_all_digits_different_l499_499853

theorem all_three_digits_same (a : ℕ) (h1 : a < 10) (h2 : 3 * a = 24) : a = 8 :=
by sorry

theorem two_digits_same (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 2 * a + b = 24 ∨ a + 2 * b = 24) : 
  (a = 9 ∧ b = 6) ∨ (a = 6 ∧ b = 9) :=
by sorry

theorem all_digits_different (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) (h7 : a + b + c = 24) :
  (a, b, c) = (7, 8, 9) ∨ (a, b, c) = (7, 9, 8) ∨ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (8, 9, 7) ∨ (a, b, c) = (9, 7, 8) ∨ (a, b, c) = (9, 8, 7) :=
by sorry

end all_three_digits_same_two_digits_same_all_digits_different_l499_499853


namespace greatest_number_same_remainder_l499_499979

theorem greatest_number_same_remainder (d : ℕ) :
  d ∣ (57 - 25) ∧ d ∣ (105 - 57) ∧ d ∣ (105 - 25) → d ≤ 16 :=
by
  sorry

end greatest_number_same_remainder_l499_499979


namespace boat_speed_24_l499_499954

def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  let speed_downstream := x + 3
  let time := 1 / 4 -- 15 minutes in hours
  let distance := 6.75
  let equation := distance = speed_downstream * time
  equation ∧ x = 24

theorem boat_speed_24 (x : ℝ) (rate_of_current : ℝ) (time_minutes : ℝ) (distance_traveled : ℝ) 
  (h1 : rate_of_current = 3) (h2 : time_minutes = 15) (h3 : distance_traveled = 6.75) : speed_of_boat_in_still_water 24 := 
by
  -- Convert time in minutes to hours
  have time_in_hours : ℝ := time_minutes / 60
  -- Effective downstream speed
  have effective_speed := 24 + rate_of_current
  -- The equation to be satisfied
  have equation := distance_traveled = effective_speed * time_in_hours
  -- Simplify and solve
  sorry

end boat_speed_24_l499_499954


namespace quadratic_real_roots_l499_499373

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ m ≤ 1 :=
by
  sorry

end quadratic_real_roots_l499_499373


namespace midpoint_of_polar_line_segment_l499_499386

theorem midpoint_of_polar_line_segment
  (r θ : ℝ)
  (hr : r > 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (hA : ∃ A, A = (8, 5 * Real.pi / 12))
  (hB : ∃ B, B = (8, -3 * Real.pi / 12)) :
  (r, θ) = (4, Real.pi / 12) := 
sorry

end midpoint_of_polar_line_segment_l499_499386


namespace imaginary_part_of_z_l499_499297

def imaginary_unit : ℂ := complex.I

def complex_number_z : ℂ := 1 / (2 + imaginary_unit)

theorem imaginary_part_of_z :
  complex.im (complex_number_z) = -1/5 := by
  sorry

end imaginary_part_of_z_l499_499297


namespace gcd_sum_5n_plus_6_n_l499_499146

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499146


namespace common_difference_gt_30000_l499_499921

open Nat

theorem common_difference_gt_30000 (d : ℕ) (a : Fin 15 → ℕ)
  (h1 : ∀ i, a i > 15) 
  (h2 : ∀ i, Prime (a i)) 
  (h3 : ∀ i, a (i + 1) = a i + d): 
  d > 30000 := 
sorry

end common_difference_gt_30000_l499_499921


namespace compare_logs_l499_499820

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l499_499820


namespace matt_worked_more_minutes_l499_499455

-- Define the conditions as constants
def monday_minutes : ℕ := 450
def tuesday_minutes : ℕ := monday_minutes / 2
def wednesday_minutes : ℕ := 300

-- The statement to prove
theorem matt_worked_more_minutes :
  wednesday_minutes - tuesday_minutes = 75 :=
begin
  sorry, -- Proof placeholder
end

end matt_worked_more_minutes_l499_499455


namespace sum_gcd_values_l499_499256

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499256


namespace june_earnings_l499_499409

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l499_499409


namespace number_of_chain_links_l499_499459

noncomputable def length_of_chain (number_of_links : ℕ) : ℝ :=
  (number_of_links * (7 / 3)) + 1

theorem number_of_chain_links (n m : ℕ) (d : ℝ) (thickness : ℝ) (max_length min_length : ℕ) 
  (h1 : d = 2 + 1 / 3)
  (h2 : thickness = 0.5)
  (h3 : max_length = 36)
  (h4 : min_length = 22)
  (h5 : m = n + 6)
  : length_of_chain n = 22 ∧ length_of_chain m = 36 
  :=
  sorry

end number_of_chain_links_l499_499459


namespace largest_unpayable_soldo_l499_499896

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499896


namespace coord_of_P_satisfies_conditions_l499_499335

noncomputable def coords_of_point_P (x y : ℝ) : Prop :=
(x = sqrt 15 / 2 ∧ y = 1) ∨ (x = -sqrt 15 / 2 ∧ y = 1) ∨
(x = sqrt 15 / 2 ∧ y = -1) ∨ (x = -sqrt 15 / 2 ∧ y = -1)

theorem coord_of_P_satisfies_conditions
  (x y : ℝ)
  (h1 : (x ^ 2) / 5 + (y ^ 2) / 4 = 1)
  (h2 : abs y = 1) :
  coords_of_point_P x y :=
by
  sorry

end coord_of_P_satisfies_conditions_l499_499335


namespace potential_values_of_K_l499_499832

theorem potential_values_of_K (K N : ℕ) (h_sum : (K * (K + 1)) / 2 = N^2) (h_N_lt_150 : N < 150) :
  K = 8 ∨ K = 49 ∨ K = 59 :=
begin
  sorry
end

end potential_values_of_K_l499_499832


namespace verify_costs_best_option_for_680_yuan_suitable_options_for_visits_l499_499579

-- Definitions and conditions
def cost_regular (x : ℕ) : ℕ := 40 * x

def cost_option1 (x : ℕ) : ℕ := 200 + 20 * x

def cost_option2 : ℕ := 1000

-- Proof statement 1: Verify costs for different options
theorem verify_costs (x : ℕ) :
  (cost_regular x = 40 * x) ∧ (cost_option1 x = 200 + 20 * x) ∧ (cost_option2 = 1000) := 
by sorry

-- Proof statement 2: Determine option for 680 yuan
theorem best_option_for_680_yuan : 
  let min_visits_regular : ℕ := 680 / 40 in
  let min_visits_option1 : ℕ := (680 - 200) / 20 in
  min_visits_regular < min_visits_option1 :=
by sorry

-- Proof statement 3: Suitable options for given range
theorem suitable_options_for_visits (x : ℕ) (h1 : 8 < x) (h2 : x < 40) :
  if (8 < x ∧ x < 10) then
    cost_regular x < cost_option1 x
  else if (10 < x ∧ x < 40) then
    cost_option1 x < cost_option2 :=
by sorry

end verify_costs_best_option_for_680_yuan_suitable_options_for_visits_l499_499579


namespace grade_A_probability_l499_499041

-- Define the existence of grades and their properties
def is_defective (grade: ℕ) : Prop :=
  grade = 2 ∨ grade = 3

-- Probabilities of producing grade B and C
def P_B : ℕ → ℝ := 
  λ grade, if grade = 2 then 0.05 else 0

def P_C : ℕ → ℝ :=
  λ grade, if grade = 3 then 0.03 else 0

-- Total probability of defects
def P_defects : ℝ :=
  P_B 2 + P_C 3

-- The probability of non-defective (Grade A)
def P_A : ℝ :=
  1 - P_defects

theorem grade_A_probability :
  P_A = 0.92 :=
by
  -- Proof is to be provided
  sorry

end grade_A_probability_l499_499041


namespace correct_statements_l499_499772

variables {V : Type*} [inner_product_space ℝ V]
variables (v : V) (n1 n2 : V)
variables (l α β : set V)

def is_direction_vector_of (v : V) (l : set V) : Prop := sorry
def is_normal_vector_of (n : V) (α : set V) : Prop := sorry
def are_not_overlapping (α β : set V) : Prop := sorry
def are_parallel (α β : set V) : Prop := sorry
def are_perpendicular (α β : set V) : Prop := sorry

theorem correct_statements (hv : is_direction_vector_of v l)
                          (hn1 : is_normal_vector_of n1 α) 
                          (hn2 : is_normal_vector_of n2 β)
                          (H_not_overlap : are_not_overlapping α β) :
  (are_parallel n1 n2 ↔ are_parallel α β) ∧
  (are_perpendicular n1 n2 ↔ are_perpendicular α β) :=
begin
  sorry
end

end correct_statements_l499_499772


namespace average_consecutive_pairs_l499_499969

-- Define the set {1, 2, 3, ..., 20}
def set_20 : Finset ℕ := Finset.range 21 \ {0}

-- Number of 4-element subsets of the set {1, 2, 3, ..., 20}
noncomputable def num_subsets : ℕ := (Finset.card (Finset.powersetLen 4 set_20)).val

-- Define the average number of pairs of consecutive integers in a randomly selected subset of 4 distinct integers chosen from the set {1, 2, 3, ..., 20}
theorem average_consecutive_pairs : (1 / num_subsets) * (3 * (nat.choose 16 3) + 6 * (nat.choose 16 2) + 4 * 17) = 2468 / 4845 :=
by sorry

end average_consecutive_pairs_l499_499969


namespace sum_of_gcd_values_l499_499218

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499218


namespace largest_unpayable_soldo_l499_499893

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l499_499893


namespace total_exercise_hours_l499_499876

-- Define the conditions
def Natasha_minutes_per_day : ℕ := 30
def Natasha_days : ℕ := 7
def Esteban_minutes_per_day : ℕ := 10
def Esteban_days : ℕ := 9
def Charlotte_monday_minutes : ℕ := 20
def Charlotte_wednesday_minutes : ℕ := 45
def Charlotte_thursday_minutes : ℕ := 30
def Charlotte_sunday_minutes : ℕ := 60

-- Sum up the minutes for each individual
def Natasha_total_minutes : ℕ := Natasha_minutes_per_day * Natasha_days
def Esteban_total_minutes : ℕ := Esteban_minutes_per_day * Esteban_days
def Charlotte_total_minutes : ℕ := Charlotte_monday_minutes + Charlotte_wednesday_minutes + Charlotte_thursday_minutes + Charlotte_sunday_minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Calculation of hours for each individual
noncomputable def Natasha_total_hours : ℚ := minutes_to_hours Natasha_total_minutes
noncomputable def Esteban_total_hours : ℚ := minutes_to_hours Esteban_total_minutes
noncomputable def Charlotte_total_hours : ℚ := minutes_to_hours Charlotte_total_minutes

-- Prove total hours of exercise for all three individuals
theorem total_exercise_hours : Natasha_total_hours + Esteban_total_hours + Charlotte_total_hours = 7.5833 := by
  sorry

end total_exercise_hours_l499_499876


namespace gcd_sum_5n_6_n_eq_12_l499_499185

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499185


namespace allan_balloons_l499_499621

theorem allan_balloons (a j : ℕ) (h1 : a = 2) (h2 : j = 6) (h3 : j = a + 1 + b) : b = 3 :=
by {
  -- Given conditions
  sorry,
}

end allan_balloons_l499_499621


namespace calculate_tan_product_l499_499367

theorem calculate_tan_product :
  let A := 30
  let B := 40
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2.9 :=
by
  sorry

end calculate_tan_product_l499_499367


namespace gcd_sum_5n_6_n_eq_12_l499_499186

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499186


namespace gcd_sum_l499_499133

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499133


namespace largest_N_not_payable_l499_499900

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499900


namespace gcd_values_sum_l499_499072

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499072


namespace player_pass_probability_l499_499380

noncomputable def probability_passing_test : ℝ :=
  let p := 2 / 3 in
  let q := 1 - p in
  /- Probability of making exactly 3 shots out of 3 attempts -/
  (p^3) +
  /- Probability of making 3 shots out of 4 attempts, one miss -/
  (3 * (p^3) * q) +
  /- Probability of making 3 shots out of 5 attempts, two misses -/
  (6 * (p^3) * (q^2))

theorem player_pass_probability : probability_passing_test = 64 / 81 := by
  sorry

end player_pass_probability_l499_499380


namespace trapezoid_KL_l499_499460

noncomputable def find_KL (a b : ℝ) : ℝ :=
  (1 / 11) * (abs (7 * b - 4 * a))

theorem trapezoid_KL (a b : ℝ) :
  let BC := a in
  let AD := b in
  let CK_KA := 7 / 4 in 
  let BL_LD := 7 / 4 in
  CK_KA = BL_LD →
  find_KL a b = (1 / 11) * (abs (7 * b - 4 * a)) :=
by
  intros h
  rw [find_KL]
  sorry

end trapezoid_KL_l499_499460


namespace science_fair_unique_students_l499_499628

/-!
# Problem statement:
At Euclid Middle School, there are three clubs participating in the Science Fair: the Robotics Club, the Astronomy Club, and the Chemistry Club.
There are 15 students in the Robotics Club, 10 students in the Astronomy Club, and 12 students in the Chemistry Club.
Assuming 2 students are members of all three clubs, prove that the total number of unique students participating in the Science Fair is 33.
-/

theorem science_fair_unique_students (R A C : ℕ) (all_three : ℕ) (hR : R = 15) (hA : A = 10) (hC : C = 12) (h_all_three : all_three = 2) :
    R + A + C - 2 * all_three = 33 :=
by
  -- Proof goes here
  sorry

end science_fair_unique_students_l499_499628


namespace sum_of_gcd_values_l499_499226

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499226


namespace initial_tickets_l499_499522

theorem initial_tickets (tickets_sold_week1 : ℕ) (tickets_sold_week2 : ℕ) (tickets_left : ℕ) 
  (h1 : tickets_sold_week1 = 38) (h2 : tickets_sold_week2 = 17) (h3 : tickets_left = 35) : 
  tickets_sold_week1 + tickets_sold_week2 + tickets_left = 90 :=
by 
  sorry

end initial_tickets_l499_499522


namespace barry_sotter_length_increase_l499_499461

theorem barry_sotter_length_increase (n : ℕ) : (n + 3) / 3 = 50 → n = 147 :=
by
  intro h
  sorry

end barry_sotter_length_increase_l499_499461


namespace sum_of_gcd_values_l499_499210

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499210


namespace total_wattage_is_correct_l499_499605

-- Defining the original wattages of the lights
def Light_A_original := 60
def Light_B_original := 40
def Light_C_original := 50

-- Defining the increase percentages for each light
def Light_A_increase_percentage := 12 / 100
def Light_B_increase_percentage := 20 / 100
def Light_C_increase_percentage := 15 / 100

-- Calculating the new wattages for each light
def Light_A_new := Light_A_original + Light_A_original * Light_A_increase_percentage
def Light_B_new := Light_B_original + Light_B_original * Light_B_increase_percentage
def Light_C_new := Light_C_original + Light_C_original * Light_C_increase_percentage

-- Calculating the total new wattage
def total_new_wattage := Light_A_new + Light_B_new + Light_C_new

-- Proof statement: The total wattage is 172.7 watts
theorem total_wattage_is_correct : total_new_wattage = 172.7 := by
  sorry

end total_wattage_is_correct_l499_499605


namespace infinite_geometric_series_sum_l499_499668

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499668


namespace pebbles_collected_at_end_of_twelfth_day_l499_499875

theorem pebbles_collected_at_end_of_twelfth_day : 
  let a := 1
  let d := 2
  let n := 12
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  in S_n = 144 := 
by
  let a := 1
  let d := 2
  let n := 12
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  have h : S_n = 144 := sorry
  exact h

end pebbles_collected_at_end_of_twelfth_day_l499_499875


namespace term_150_is_2280_l499_499514

def sequence (n : ℕ) : ℕ :=
  ∑ i in (nat.binary_digits n), 3 ^ i 

theorem term_150_is_2280 : sequence 150 = 2280 :=
by
  sorry

end term_150_is_2280_l499_499514


namespace pinocchio_cannot_pay_exactly_l499_499890

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499890


namespace infinite_geometric_series_sum_l499_499671

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499671


namespace total_pieces_of_tomatoes_l499_499033

namespace FarmerTomatoes

variables (rows plants_per_row yield_per_plant : ℕ)

def total_plants (rows plants_per_row : ℕ) := rows * plants_per_row

def total_tomatoes (total_plants yield_per_plant : ℕ) := total_plants * yield_per_plant

theorem total_pieces_of_tomatoes 
  (hrows : rows = 30)
  (hplants_per_row : plants_per_row = 10)
  (hyield_per_plant : yield_per_plant = 20) :
  total_tomatoes (total_plants rows plants_per_row) yield_per_plant = 6000 :=
by
  rw [hrows, hplants_per_row, hyield_per_plant]
  unfold total_plants total_tomatoes
  norm_num
  done

end FarmerTomatoes

end total_pieces_of_tomatoes_l499_499033


namespace sum_gcd_possible_values_eq_twelve_l499_499241

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499241


namespace largest_N_cannot_pay_exactly_without_change_l499_499885

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499885


namespace both_hit_given_target_hit_l499_499542

theorem both_hit_given_target_hit (P_A P_B : ℝ) (hA : P_A = 0.6) (hB : P_B = 0.7) :
  let P_C := 1 - (1 - P_A) * (1 - P_B) in
  P_C ≠ 0 → (P_A * P_B) / P_C = 21 / 44 :=
by
  intros
  sorry

end both_hit_given_target_hit_l499_499542


namespace mr_ray_customers_without_fish_l499_499449

def mr_ray_num_customers_without_fish
  (total_customers : ℕ)
  (total_tuna_weight : ℕ)
  (specific_customers_30lb : ℕ)
  (specific_weight_30lb : ℕ)
  (specific_customers_20lb : ℕ)
  (specific_weight_20lb : ℕ)
  (weight_per_customer : ℕ)
  (remaining_tuna_weight : ℕ)
  (num_customers_served_with_remaining_tuna : ℕ)
  (total_satisfied_customers : ℕ) : ℕ :=
  total_customers - total_satisfied_customers

theorem mr_ray_customers_without_fish :
  mr_ray_num_customers_without_fish 100 2000 10 30 15 20 25 1400 56 81 = 19 :=
by 
  sorry

end mr_ray_customers_without_fish_l499_499449


namespace length_of_AB_l499_499856

open Real

def parabola (x: ℝ) := y ^ 2 = 5 * x
def focus : ℝ × ℝ := (5 / 4, 0)
def line_eq (θ: ℝ) (x: ℝ) := y = tan θ * (x - (focus.1))

theorem length_of_AB (x_A x_B : ℝ) (h1 : parabola x_A) (h2 : parabola x_B)
            (h3: x_A + x_B = 25 / 6) :  
  let dist := x_A + 5 / 2 + x_B + 5 / 2 
  in dist = 20 / 3 := 
  sorry

end length_of_AB_l499_499856


namespace circle_standard_eq_circle_polar_eq_line_intersects_circle_chord_length_l499_499316

-- Define the circle parameterization
def circleParam (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line parameterization
def lineParam (t : ℝ) : ℝ × ℝ :=
  (2 + 4/5 * t, 3/5 * t)

-- Define the standard equation of the circle in Cartesian coordinates
def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- Define the polar equation of the circle
def polarEquation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ

-- Define the standard form of the line
def lineStandardForm (x y : ℝ) : Prop :=
  3 * x - 4 * y - 6 = 0

-- Proving standard equation of circle
theorem circle_standard_eq (θ : ℝ) :
  circleEquation (2 + 2 * Real.cos θ) (2 * Real.sin θ) :=
sorry

-- Proving polar equation of circle
theorem circle_polar_eq (ρ θ : ℝ) (h : ρ * Real.cos θ = 2 + 2 * Real.cos θ ∧ ρ * Real.sin θ = 2 * Real.sin θ) :
  polarEquation ρ θ :=
sorry

-- Proving line intersects the circle
theorem line_intersects_circle : ∃ (t θ : ℝ), circleEquation (2 + 4/5 * t) (3/5 * t) :=
sorry

-- Proving length of the chord
theorem chord_length : ∃ (x₁ y₁ x₂ y₂ : ℝ), circleEquation x₁ y₁ ∧ circleEquation x₂ y₂ ∧ 
                                             lineStandardForm x₁ y₁ ∧ lineStandardForm x₂ y₂ ∧ 
                                             Real.dist (x₁, y₁) (x₂, y₂) = 4 :=
sorry

end circle_standard_eq_circle_polar_eq_line_intersects_circle_chord_length_l499_499316


namespace sum_gcd_possible_values_eq_twelve_l499_499248

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499248


namespace gcd_sum_5n_plus_6_l499_499203

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499203


namespace common_chord_and_tangent_of_circles_l499_499778

theorem common_chord_and_tangent_of_circles :
  (∃ P : ℝ² → ℝ, ∀ x y, P (x, y) = x^2 + y^2 - 4 - (x^2 + y^2 + 2 * x - 4 * y + 4)) ∧
  (2-0 / -1)= -2  :=
begin
  -- Condition for common chord
  have P_common_chord: ∃ P : ℝ → ℝ → Prop, ∀ x y, P x y = (x - 2 * y + 4),
  {
    sorry,
  },
  -- Condition for line as common tangent 
    have P_2: ∃ P : ℝ → {0, 1, 2, 3} , P x = ( x = -2 ) ,
  { sorry,
  },
 sorry.
end

end common_chord_and_tangent_of_circles_l499_499778


namespace fraction_of_students_speak_foreign_language_l499_499978

noncomputable def students_speak_foreign_language_fraction (M F : ℕ) (h1 : M = F) (m_frac : ℚ) (f_frac : ℚ) : ℚ :=
  ((3 / 5) * M + (2 / 3) * F) / (M + F)

theorem fraction_of_students_speak_foreign_language (M F : ℕ) (h1 : M = F) :
  students_speak_foreign_language_fraction M F h1 (3 / 5) (2 / 3) = 19 / 30 :=
by 
  sorry

end fraction_of_students_speak_foreign_language_l499_499978


namespace gcd_sum_5n_6_n_eq_12_l499_499194

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499194


namespace log_comparison_l499_499806

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l499_499806


namespace radius_of_sphere_in_truncated_cone_l499_499055

theorem radius_of_sphere_in_truncated_cone (r1 r2 : ℝ) (h : r1 = 25 ∧ r2 = 5) : 
  ∃ r : ℝ, r = 5 * Real.sqrt 2 ∧ 
           (∀ (x : ℝ), x ∈ {x | x = r * 2} :=
           Real.sqrt ((25 + 5:ℝ)^2 - (25 - 5:ℝ)^2)) :=
by
  sorry

end radius_of_sphere_in_truncated_cone_l499_499055


namespace protein_powder_requirement_l499_499872

def protein_intake_per_day (body_weight : ℕ) (intake_per_kg : ℕ) : ℕ := body_weight * intake_per_kg
def protein_powder_needed_per_day (desired_protein : ℕ) (protein_percentage : ℤ) : ℕ := (desired_protein : ℕ) * 100 / protein_percentage.to_nat
def protein_powder_needed_per_week (daily_powder : ℕ) : ℕ := daily_powder * 7

theorem protein_powder_requirement 
  (protein_percentage : ℤ)
  (body_weight : ℕ)
  (intake_per_kg : ℕ)
  (desired_daily_intake : ℕ)
  (daily_powder : ℕ)
  (weekly_powder : ℕ)
  (h1 : protein_percentage = 80)
  (h2 : body_weight = 80)
  (h3 : intake_per_kg = 2)
  (h4 : desired_daily_intake = protein_intake_per_day body_weight intake_per_kg)
  (h5 : daily_powder = protein_powder_needed_per_day desired_daily_intake protein_percentage)
  (h6 : weekly_powder = protein_powder_needed_per_week daily_powder) : 
  weekly_powder = 1400 :=
by
  sorry

end protein_powder_requirement_l499_499872


namespace sum_of_gcd_values_l499_499215

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499215


namespace complex_number_solution_l499_499289

theorem complex_number_solution :
  ∃ z : ℂ, (1 + 2 * complex.I) * z = -3 + 4 * complex.I ∧ z = 1 + 2 * complex.I :=
by
  sorry

end complex_number_solution_l499_499289


namespace recipe_required_ingredients_l499_499446

-- Define the number of cups required for each ingredient in the recipe
def sugar_cups : Nat := 11
def flour_cups : Nat := 8
def cocoa_cups : Nat := 5

-- Define the cups of flour and cocoa already added
def flour_already_added : Nat := 3
def cocoa_already_added : Nat := 2

-- Define the cups of flour and cocoa that still need to be added
def flour_needed_to_add : Nat := 6
def cocoa_needed_to_add : Nat := 3

-- Sum the total amount of flour and cocoa powder based on already added and still needed amounts
def total_flour: Nat := flour_already_added + flour_needed_to_add
def total_cocoa: Nat := cocoa_already_added + cocoa_needed_to_add

-- Total ingredients calculation according to the problem's conditions
def total_ingredients : Nat := sugar_cups + total_flour + total_cocoa

-- The theorem to be proved
theorem recipe_required_ingredients : total_ingredients = 24 := by
  sorry

end recipe_required_ingredients_l499_499446


namespace infinite_geometric_series_sum_l499_499698

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499698


namespace combined_money_l499_499962

-- Definition of amounts for Tom, Nataly, and Raquel
def tom_money (nataly_money : ℕ) : ℕ := nataly_money / 4
def nataly_money (raquel_money : ℕ) : ℕ := 3 * raquel_money
def raquel_money : ℕ := 40

-- Proof statement
theorem combined_money : 
  let T := tom_money (nataly_money raquel_money) in
  let N := nataly_money raquel_money in
  let R := raquel_money in
  T + N + R = 190 :=
by
  sorry

end combined_money_l499_499962


namespace pentagon_is_regular_l499_499515

open set function

noncomputable theory
open_locale classical

-- Define a convex pentagon and the congruence of triangles
variables (A B C D E X Y Z T W : Type)
  [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E]
  [decidable_eq X] [decidable_eq Y] [decidable_eq Z] [decidable_eq T] [decidable_eq W]

structure convex_pentagon (A B C D E : Type) := 
  (convex : true) -- Omitting actual convex verification for simplicity

structure congruent_triangles (tri1 tri2 : Type) :=
  (congruent : true) -- Omitting actual congruence verification for simplicity

def extended_sides_form_congruent_triangles 
  (pentagon : convex_pentagon A B C D E)
  (t1 t2 t3 t4 t5 : Type)
  (h1 : congruent_triangles t1 t2)
  (h2 : congruent_triangles t2 t3)
  (h3 : congruent_triangles t3 t4)
  (h4 : congruent_triangles t4 t5)
  (h5 : congruent_triangles t5 t1): Prop :=
true

theorem pentagon_is_regular
  (pentagon : convex_pentagon A B C D E)
  (t1 t2 t3 t4 t5 : Type)
  (h1 : congruent_triangles t1 t2)
  (h2 : congruent_triangles t2 t3)
  (h3 : congruent_triangles t3 t4)
  (h4 : congruent_triangles t4 t5)
  (h5 : congruent_triangles t5 t1)
  (h_congruent : extended_sides_form_congruent_triangles pentagon t1 t2 t3 t4 t5 h1 h2 h3 h4 h5) :
  true := 
sorry -- Proof omitted

end pentagon_is_regular_l499_499515


namespace triangle_given_abc_cosB_l499_499397

theorem triangle_given_abc_cosB 
  (a : ℝ) (b : ℝ) (c : ℝ) (B : ℝ) 
  (h1 : a = 3) 
  (h2 : b - c = 2) 
  (h3 : cos B = -1/2) : 
  b = 7 ∧ c = 5 ∧ sin (B + (π - B - acos (a / (b * sin B)))) = 3 * sqrt 3 / 14 :=
by
  -- Proof goes here
  sorry

end triangle_given_abc_cosB_l499_499397


namespace gcd_sum_l499_499161

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499161


namespace calc_expression_l499_499277

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end calc_expression_l499_499277


namespace range_of_t_l499_499344

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then
    x^2 + x
  else
    -x^2

-- Specification of the proof problem with given conditions and the correct answer
theorem range_of_t (t : ℝ) (h : f (f t) ≤ 2) : t ≤ sqrt 2 := by
  sorry

end range_of_t_l499_499344


namespace part1_part2_part3_l499_499350

-- Definitions from part (1):
def parabola_Γ1 (x y : ℝ) : Prop := y^2 = 4 * x
def focus_Γ1 := (1, 0)
def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Definition from part (2):
def parabola_Γ2 (x y : ℝ) : Prop := y^2 = 2 * x
def midpoint_on_x_axis (A C : ℝ × ℝ) : Prop := (A.1 + C.1) / 2 = 0 ∧ (A.2 + C.2) / 2 = 0

-- Definitions from part (3):
def vector_equality (A B C D : ℝ × ℝ) : Prop := (B.1 - A.1, B.2 - A.2) = 2 * (D.1 - C.1, D.2 - C.2)
def point_origin := (0.0 : ℝ, 0.0 : ℝ)

-- Prove Statements
theorem part1 (A : ℝ × ℝ) (hA_Γ1 : parabola_Γ1 A.1 A.2)
  (hA_focus_dist : dist A focus_Γ1 = 2) (hA_quadrant : in_first_quadrant A) :
  A = (1, 2) :=
sorry

theorem part2 (A C : ℝ × ℝ) (hA : A = (4, 4)) 
  (hC_Γ2 : parabola_Γ2 C.1 C.2) (h_midpoint : midpoint_on_x_axis A C) :
  ∃ l : ℝ × ℝ → Prop, 
  (point_origin = l ∧ dist point_origin l = 12 * Real.sqrt 5 / 5) :=
sorry

theorem part3 (A B C D : ℝ × ℝ) (hAB_CD : vector_equality A B C D) 
  (hArea : area_ratio (A, (0, 0), D) (B, (0, 0), C) = 10 / 7) :
  ∃ r : ℝ, r = 10 / 7 :=
sorry

end part1_part2_part3_l499_499350


namespace find_m_l499_499751

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the condition of orthogonality
def orthogonality (m : ℝ) : Prop := dot_product a (b m) = 0

-- Our main theorem stating that m = 2 / 3 is the solution
theorem find_m (m : ℝ) : orthogonality m → m = 2 / 3 :=
begin
  intro h,
  -- Placeholder for the actual proof steps
  sorry
end

end find_m_l499_499751


namespace sum_of_valid_z_values_l499_499658

theorem sum_of_valid_z_values : ∃ s : ℕ, ∀ (z : ℕ), (z ≤ 9) ∧ (18 + z) % 3 = 0 → (z = 0 ∨ z = 3 ∨ z = 6 ∨ z = 9) ∧ s = 0 + 3 + 6 + 9 :=
by
  let s := 18
  use s
  intros z hz
  split
  case left =>
    cases hz with _ hdiv3
    have z0 : z = 0 ∨ z = 3 ∨ z = 6 ∨ z = 9 := by
      interval_cases z
      all_goals { nlinarith }
    exact z0
  case right =>
    rfl
  sorry

end sum_of_valid_z_values_l499_499658


namespace thickness_relation_l499_499951

noncomputable def a : ℝ := (1/3) * Real.sin (1/2)
noncomputable def b : ℝ := (1/2) * Real.sin (1/3)
noncomputable def c : ℝ := (1/3) * Real.cos (7/8)

theorem thickness_relation : c > b ∧ b > a := by
  sorry

end thickness_relation_l499_499951


namespace parabola_solution_l499_499961

noncomputable def parabola_question : Prop :=
  let p := 2 in
  let |AB| := 4 / (Real.sin θ)^2 in
  let |CD| := 4 / (Real.cos θ)^2 in
  (1 / |AB| + 1 / |CD| = 1 / 4)

theorem parabola_solution : parabola_question := 
  sorry

end parabola_solution_l499_499961


namespace centroid_tetrahedron_l499_499503

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D M : V)

def is_centroid (M A B C D : V) : Prop :=
  M = (1/4:ℝ) • (A + B + C + D)

theorem centroid_tetrahedron (h : is_centroid M A B C D) :
  (M - A) + (M - B) + (M - C) + (M - D) = (0 : V) :=
by {
  sorry
}

end centroid_tetrahedron_l499_499503


namespace slope_of_line_l_l499_499794

theorem slope_of_line_l (A B : ℝ × ℝ) (tan_slope_AB : ℝ) (alpha : ℝ) (tan_alpha : ℝ) :
  A = (-1, -5) →
  B = (3, -2) →
  tan_alpha = (B.2 - A.2) / (B.1 - A.1) →
  (tan (2 * alpha) = (2 * tan_alpha) / (1 - tan_alpha^2)) →
  tan (2 * alpha) = 24 / 7 :=
by
  sorry

end slope_of_line_l_l499_499794


namespace pet_store_cages_l499_499601

theorem pet_store_cages (n s p : ℕ) (h1 : n = 102) (h2 : s = 21) (h3 : p = 9) : n - s) / p = 9 :=
by
  sorry

end pet_store_cages_l499_499601


namespace pet_center_final_count_l499_499401

/-!
# Problem: Count the total number of pets in a pet center after a series of adoption and collection events.
-/

def initialDogs : Nat := 36
def initialCats : Nat := 29
def initialRabbits : Nat := 15
def initialBirds : Nat := 10

def dogsAdopted1 : Nat := 20
def rabbitsAdopted1 : Nat := 5

def catsCollected : Nat := 12
def rabbitsCollected : Nat := 8
def birdsCollected : Nat := 5

def catsAdopted2 : Nat := 10
def birdsAdopted2 : Nat := 4

def finalDogs : Nat :=
  initialDogs - dogsAdopted1

def finalCats : Nat :=
  initialCats + catsCollected - catsAdopted2

def finalRabbits : Nat :=
  initialRabbits - rabbitsAdopted1 + rabbitsCollected

def finalBirds : Nat :=
  initialBirds + birdsCollected - birdsAdopted2

def totalPets (d c r b : Nat) : Nat :=
  d + c + r + b

theorem pet_center_final_count : 
  totalPets finalDogs finalCats finalRabbits finalBirds = 76 := by
  -- This is where we would provide the proof, but it's skipped as per the instructions.
  sorry

end pet_center_final_count_l499_499401


namespace largest_possible_x_plus_y_l499_499501

def integers : List Int := [1, 2, 4, 5, 6, 9, 10, 11, 13]

def valid_configuration (x y : Int) (shape : List (Option Int)) : Prop :=
  shape.length = 9 ∧  -- We need nine slots (5 squares and 4 circles)
  (∀ i, i < shape.length → (shape.nth i).isSome) ∧ -- All shapes are filled
  x = shape.head.getOrElse 0 ∧ -- x is placed in the leftmost square
  y = shape.length - 1 → (shape.getLast.getOrElse 0) = y ∧ -- y is placed in the rightmost square
  ∀ i, i % 2 = 1 →  -- For circles (odd indices in zero-based list)
    shape.nth i = some (shape.nth (i-1).getOrElse 0 + shape.nth (i+1).getOrElse 0) -- each circle is sum of adjacent squares

theorem largest_possible_x_plus_y : ∃ (x y : Int), valid_configuration x y [none] ∧ x + y = 20 := by
  sorry

end largest_possible_x_plus_y_l499_499501


namespace least_distance_between_ticks_l499_499571

theorem least_distance_between_ticks (x : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, k = n * 11 ∨ k = n * 13) →
  x = 1 / 143 :=
by
  sorry

end least_distance_between_ticks_l499_499571


namespace range_of_a_l499_499792

theorem range_of_a (a : ℝ) : (∀ x, (x ∈ set.Ici (1 : ℝ)) → (x ∈ set.Ici a)) → a ≤ 1 :=
by
  intros h
  sorry

end range_of_a_l499_499792


namespace circumcircle_eq_tangent_lines_l499_499768

theorem circumcircle_eq_tangent_lines (A B C P : Point)
  (hA : A = ⟨-1, 2⟩)
  (hC : C = ⟨-1, 0⟩)
  (h_symm : ∃ m n, B = ⟨m, n⟩ ∧ (m, n) is the symmetric of A w.r.t. the line x - y + 1 = 0)
  (hP : P = ⟨Real.sqrt 2, 2⟩) :
  (∃ center radius, circle (⊨ (A, B, C)) center radius ∧ circle_eq center radius x^2 + (y - 1)^2 = 2)
  ∧ (line_is_tangent P center radius (x = Real.sqrt 2) ∨ line_is_tangent P center radius (Real.sqrt 2 * x + 4 * y - 10 = 0)) :=
by
  sorry

end circumcircle_eq_tangent_lines_l499_499768


namespace largest_N_cannot_pay_exactly_without_change_l499_499881

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499881


namespace sum_of_gcd_values_l499_499123

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499123


namespace monomial_sum_exponents_l499_499375

theorem monomial_sum_exponents (m n : ℕ) (h₁ : m - 1 = 2) (h₂ : n = 2) : m^n = 9 := 
by
  sorry

end monomial_sum_exponents_l499_499375


namespace hyperbola_eccentricity_correct_l499_499788

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) :
  hyperbola_eccentricity a b h_a h_b h_asymptote = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_correct_l499_499788


namespace infinite_geometric_series_sum_l499_499669

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499669


namespace neg_four_is_square_root_of_sixteen_l499_499553

/-
  Definitions:
  - A number y is a square root of x if y^2 = x.
  - A number y is an arithmetic square root of x if y ≥ 0 and y^2 = x.
-/

theorem neg_four_is_square_root_of_sixteen :
  -4 * -4 = 16 := 
by
  -- proof step is omitted
  sorry

end neg_four_is_square_root_of_sixteen_l499_499553


namespace shaded_region_correct_l499_499953

def side_length_ABCD : ℝ := 8
def side_length_BEFG : ℝ := 6

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_ABCD : ℝ := area_square side_length_ABCD
def area_BEFG : ℝ := area_square side_length_BEFG

def shaded_region_area : ℝ :=
  area_ABCD + area_BEFG - 32

theorem shaded_region_correct :
  shaded_region_area = 32 :=
by
  -- Proof omitted, but placeholders match problem conditions and answer
  sorry

end shaded_region_correct_l499_499953


namespace maximum_ab_ac_bd_cd_l499_499956

theorem maximum_ab_ac_bd_cd :
  ∃ (a b c d : ℕ), {a, b, c, d} = {2, 3, 4, 5} ∧ (ab + ac + bd + cd) = 49 :=
by
  sorry -- Proof not required.

end maximum_ab_ac_bd_cd_l499_499956


namespace second_worker_time_l499_499058

theorem second_worker_time (h1 : ∀ h1 > 0, worker1_rate : (1 / 6) trucks / hour) 
  (h2 : ∀ h2 > 0, combined_rate : (h1 + h2) = 1 / 3.428571428571429)
  : second_worker_time = 8 :=
begin
  -- Setup calculation for second worker's time based on conditions
  let worker1_rate := 1 / 6,
  let combined_rate := 1 / 3.428571428571429,

  -- Using the condition that combined_rate = worker1_rate + second_worker_rate
  have combined_rate_eq : combined_rate = (worker1_rate + 1 / second_worker_time),
  sorry,
end

end second_worker_time_l499_499058


namespace equilateral_triangle_symmetry_l499_499008

def symmetric_about_axis (shape : Type) : Prop :=
sorry -- definition of symmetry about an axis

def symmetric_about_center (shape : Type) : Prop :=
sorry -- definition of symmetry about the center

inductive Shapes
| Rectangle
| Rhombus
| EquilateralTriangle
| Circle

open Shapes

theorem equilateral_triangle_symmetry :
  symmetric_about_axis EquilateralTriangle ∧ ¬ symmetric_about_center EquilateralTriangle :=
sorry

end equilateral_triangle_symmetry_l499_499008


namespace largest_n_stones_l499_499833

theorem largest_n_stones
  (initial_piles : ℕ)
  (initial_stones : ℕ)
  (num_piles : ℕ)
  (k_i_max : ℕ)
  (n : ℕ) :
  initial_piles = 100 →
  initial_stones = 100 →
  num_piles = 50 →
  k_i_max = 50 →
  n = 5099 →
  (∀ (k : ℕ) (h : k ≤ n), ∃ (piles : finset ℕ),
     piles.card = num_piles ∧
     ∀ (i : ℕ) (hi : i ∈ piles),
       initial_stones - (∑ j in piles, k) ≥ i) :=
sorry

end largest_n_stones_l499_499833


namespace profit_loss_balance_l499_499513

-- Defining variables
variables (C L : Real)

-- Profit and loss equations according to problem conditions
theorem profit_loss_balance (h1 : 832 - C = C - L) (h2 : 992 = 0.55 * C) : 
  (C + 992 = 2795.64) :=
by
  -- Statement of the theorem
  sorry

end profit_loss_balance_l499_499513


namespace gcd_sum_5n_plus_6_n_l499_499144

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499144


namespace chlorine_weight_is_35_l499_499738

def weight_Na : Nat := 23
def weight_O : Nat := 16
def molecular_weight : Nat := 74

theorem chlorine_weight_is_35 (Cl : Nat) 
  (h : molecular_weight = weight_Na + Cl + weight_O) : 
  Cl = 35 := by
  -- Proof placeholder
  sorry

end chlorine_weight_is_35_l499_499738


namespace range_of_x_satisfying_condition_l499_499922

variable {f : ℝ → ℝ}

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

theorem range_of_x_satisfying_condition
  (h1 : ∀ x, f'' x = f x.derivative)
  (h2 : isOddFunction f)
  (h3 : ∀ x > 0, ln x * (f' x) < - (1 / x) * f x) :
  {x : ℝ | (x^2 - 1) * f x > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1} := 
sorry

end range_of_x_satisfying_condition_l499_499922


namespace interest_calculation_l499_499489

def calculate_interest (P A r : ℝ) : ℝ := A - P

theorem interest_calculation :
  ∀ (A r : ℝ), let n := 1 in let t := 2 in
  let P := A / (1 + r / n) ^ (n * t) in
  r = 0.04 → A = 4326.399999999994 → P ≈ 4000 → calculate_interest P A r ≈ 326.40 :=
by
  intros A r n t P h_r h_A h_P
  simp [calculate_interest]
  sorry

end interest_calculation_l499_499489


namespace gcd_sum_l499_499139

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499139


namespace gcd_5n_plus_6_n_sum_l499_499098

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499098


namespace sum_of_valid_numbers_l499_499550

def is_valid_number (N : ℕ) : Prop :=
  let a := (N / 100) % 10
  let b := (N / 10) % 10
  let c := N % 10
  let S := a + b + c
  let P := a * b * c
  let threeH := 3 * a
  (100 ≤ N ∧ N < 1000) ∧
  (S ∣ N) ∧ 
  ¬(P ∣ N) ∧ 
  (threeH ∣ N)

theorem sum_of_valid_numbers : 
  (Finset.sum (Finset.filter is_valid_number (Finset.range 1000))) = 1269 :=
by
  sorry

end sum_of_valid_numbers_l499_499550


namespace pencil_cost_l499_499462

theorem pencil_cost (P : ℕ) (h1 : ∀ p : ℕ, p = 80) (h2 : ∀ p_est, ((16 * P) + (20 * 80)) = p_est → p_est = 2000) (h3 : 36 = 16 + 20) :
    P = 25 :=
  sorry

end pencil_cost_l499_499462


namespace probability_qualifying_all_probability_exactly_one_unqualified_l499_499524

noncomputable def P_qualify_all (P_A P_B P_C : ℝ) : ℝ :=
P_A * P_B * P_C

noncomputable def P_exactly_one_unqualify (P_A P_B P_C : ℝ) : ℝ :=
(1 - P_A) * P_B * P_C + P_A * (1 - P_B) * P_C + P_A * P_B * (1 - P_C)

theorem probability_qualifying_all :
  let P_A := 0.85;
      P_B := 0.90;
      P_C := 0.95 in
  P_qualify_all P_A P_B P_C ≈ 0.73 :=
by
  sorry

theorem probability_exactly_one_unqualified :
  let P_A := 0.85;
      P_B := 0.90;
      P_C := 0.95 in
  P_exactly_one_unqualify P_A P_B P_C ≈ 0.25 :=
by
  sorry

end probability_qualifying_all_probability_exactly_one_unqualified_l499_499524


namespace gcd_sequence_l499_499866

open BigOperators

noncomputable def a (n : ℕ) : ℕ :=
  2^(3 * n) + 3^(6 * n + 2) + 5^(6 * n + 2)

theorem gcd_sequence : ∀ (n : ℕ), 0 ≤ n → n < 2000 → ∃ k : ℕ, a(k) = 7 := by
  sorry

end gcd_sequence_l499_499866


namespace expected_digits_of_icosahedral_die_l499_499273

theorem expected_digits_of_icosahedral_die : 
  let numbers := finset.range 1 21
  let one_digit_numbers := numbers.filter (λ n, n < 10)
  let two_digit_numbers := numbers.filter (λ n, n >= 10)
  let p_one := (one_digit_numbers.card : ℚ) / (numbers.card : ℚ)
  let p_two := (two_digit_numbers.card : ℚ) / (numbers.card : ℚ)
  let expected_value := p_one * 1 + p_two * 2
  expected_value = 1.55 := 
by 
  sorry

end expected_digits_of_icosahedral_die_l499_499273


namespace gcd_sum_5n_6_n_eq_12_l499_499193

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499193


namespace total_amount_divided_l499_499578

-- Define the conditions
variables (A B C : ℕ)
axiom h1 : 4 * A = 5 * B
axiom h2 : 4 * A = 10 * C
axiom h3 : C = 160

-- Define the theorem to prove the total amount
theorem total_amount_divided (h1 : 4 * A = 5 * B) (h2 : 4 * A = 10 * C) (h3 : C = 160) : 
  A + B + C = 880 :=
sorry

end total_amount_divided_l499_499578


namespace problem_A_plus_B_l499_499427

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l499_499427


namespace gcd_sum_5n_plus_6_l499_499207

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499207


namespace curved_surface_area_proof_l499_499044

noncomputable def curved_surface_area_of_cone (r h l : ℝ) : ℝ := π * r * l

theorem curved_surface_area_proof :
  let r := 3
      h := 12
      l := real.sqrt (r^2 + h^2)
  in
  curved_surface_area_of_cone r h l ≈ 117.81 :=
by
  let r := 3
  let h := 12
  let l := real.sqrt (r^2 + h^2)
  calc
    curved_surface_area_of_cone r h l
      = π * r * l : by rw [curved_surface_area_of_cone]
      ≈ 117.81 : sorry

end curved_surface_area_proof_l499_499044


namespace sum_of_divisors_of_24_l499_499003

theorem sum_of_divisors_of_24 (m : ℕ) (h : (m + 24) % m = 0) (hpos : m > 0) : 
  ∑ i in (Finset.filter (λ d, (24 % d = 0)) (Finset.range 25)), i = 60 := 
by 
  sorry

end sum_of_divisors_of_24_l499_499003


namespace sum_gcd_possible_values_eq_twelve_l499_499247

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499247


namespace infinite_geometric_series_sum_l499_499673

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499673


namespace courtyard_breadth_l499_499031

theorem courtyard_breadth (length_courtyard : ℝ) (brick_length_cm : ℝ) (brick_width_cm : ℝ) (total_bricks : ℕ) :
  length_courtyard = 20 → brick_length_cm = 20 → brick_width_cm = 10 → total_bricks = 16000 →
  let brick_area_m2 := (brick_length_cm / 100) * (brick_width_cm / 100)
    total_area := total_bricks * brick_area_m2
  in total_area / length_courtyard = 16 :=
by
  intros _ _ _ _ brick_area_m2 total_area
  have : brick_area_m2 = 0.02 := sorry
  have : total_area = 320 := sorry
  have : total_area / length_courtyard = 16 := sorry
  exact this

end courtyard_breadth_l499_499031


namespace pinocchio_cannot_pay_exactly_l499_499888

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499888


namespace triangle_circumcenters_l499_499434

-- Define the Lean proof statement
theorem triangle_circumcenters (A B C P M_A M_B D E : Point)
  (h_triangle : triangle A B C)
  (h_interior : inside_triangle P A B C)
  (h_circumcircle_ACP : circumcircle_center M_A A C P)
  (h_circumcircle_BCP : circumcircle_center M_B B C P)
  (h_MA_outside : outside_triangle M_A A B C)
  (h_MB_outside : outside_triangle M_B A B C)
  (h_collinear_APMA : collinear A P M_A)
  (h_collinear_BPMB : collinear B P M_B)
  (h_parallel_line : parallel_line_through P A B intersects_at D E P)
  (h_D_ne_P : D ≠ P)
  (h_E_ne_P : E ≠ P) :
  segment_length D E = segment_length A C + segment_length B C :=
sorry

end triangle_circumcenters_l499_499434


namespace prove_concyclic_l499_499867

-- Definitions for cyclic quadrilateral and circumcircles
variable (A B C D O K L M P Q : Point)
variable [Nontrivial (CyclicQuadrilateral ABCD)]
variable [intersection_diagonals : IntersectingDiagonals ABCD O]
variable [circumcircle_ABO : Circumcircle (Triangle A B O, K)]
variable [circumcircle_CDO : Circumcircle (Triangle C D O, K)]
variable [line_parallel1 : ParallelTo (Line O L) (Line A B)]
variable [line_parallel2 : ParallelTo (Line O M) (Line C D)]
variable [point_on_line1 : PointOnLine L (Circumcircle (Triangle A B O K))]
variable [point_on_line2 : PointOnLine M (Circumcircle (Triangle C D O K))]
variable [ratio_condition : RatiosEqual (Segment O P) (Segment P L) (Segment M Q) (Segment Q O)]

-- Statement: Prove O, K, P, and Q are concyclic
theorem prove_concyclic {ABCD : CyclicQuadrilateral} (O K L M P Q : Point)
  [h1: Intersection (Diagonals ABCD) O]
  [h2: CircumcirclesIntersect (Triangle A B O) (Triangle C D O) K]
  [h3: LineThroughPointParallelTo AB OL]
  [h4: LineThroughPointParallelTo CD OM]
  [h5: Intersection (Line OL) (Circumcircle (Triangle A B O K)) L]
  [h6: Intersection (Line OM) (Circumcircle (Triangle C D O K)) M]
  [h7: RatioEqual (Segment OP) (Segment PL) (Segment MQ) (Segment QO)] : 
  ConcyclicPoints O K P Q :=
sorry

end prove_concyclic_l499_499867


namespace quadrilateral_cyclic_tangential_l499_499466

section Geometry

-- Define a quadrilateral with a point inside matching the conditions.
variables (A B C D M : ℝ)

-- Conditions for equidistance and area.
variables (h1 : dist_from_line M A B = dist_from_line M C D)
variables (h2 : dist_from_line M B C = dist_from_line M A D)
variables (h3 : area_of_quadrilateral A B C D = dist M A * dist M C + dist M B * dist M D)

-- The proof statements that the quadrilateral is cyclic and tangential.
theorem quadrilateral_cyclic_tangential :
  is_cyclic_quadrilateral A B C D ∧ is_tangential_quadrilateral A B C D :=
by
  sorry

end Geometry

end quadrilateral_cyclic_tangential_l499_499466


namespace total_liars_odd_l499_499619

-- Definitions for the problem context
def is_knight (p : ℕ) : Prop := sorry -- Predicate indicating if the person is a knight.
def is_liar (p : ℕ) : Prop := sorry -- Predicate indicating if the person is a liar.
def neighbors (g : ℕ) (p : ℕ) : list ℕ := sorry -- Function that determines the neighbors of a person in the grid.
def odd_number_of_liars (l : list ℕ) : Prop := sorry -- Predicate indicating if the list has an odd number of liars.

-- Assume:
-- 1. Each person in the grid makes a statement about their neighbors.
-- 2. Knights tell the truth and liars lie about their statements.
axiom grid_structure (grid : fin 81) (person : ℕ) :
  (is_knight person → odd_number_of_liars (neighbors 81 person)) ∧ 
  (is_liar person → ¬odd_number_of_liars (neighbors 81 person))

-- Prove:
-- The total number of liars in the village is odd.
theorem total_liars_odd (grid : fin 81) :
  (∃ (l : list ℕ), (∀ p ∈ l, is_liar p) ∧ list.length l % 2 = 1) :=
sorry

end total_liars_odd_l499_499619


namespace gcd_sum_5n_plus_6_n_l499_499140

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499140


namespace gcd_sum_l499_499090

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499090


namespace num_segments_eq_l499_499475

-- Define the initial conditions
variable (n : ℕ) -- Number of lines
variable (λ : ℕ → ℕ) -- Function representing the lines intersecting at each point

-- Define a predicate for the intersection points
def is_intersection_point (P : ℕ) : Prop := ∃ k, λ k = P

-- Define a function for the sum of intersection points
noncomputable def sum_lambda : ℕ := 
  ∑ k in finset.filter is_intersection_point (finset.range n), λ k

-- Statement of the theorem
theorem num_segments_eq (n : ℕ) (λ : ℕ → ℕ) :
  (∃ x, x = n + sum_lambda n λ) ↔ 
  (∀ P, is_intersection_point P → ∑ finset.range n λ = n + ∑ λ) :=
sorry

end num_segments_eq_l499_499475


namespace sqrt_inequality_l499_499914

theorem sqrt_inequality (n : ℕ) (hn : n ≥ 2) : 
  Real.sqrtN 2 n - 1 ≤ Real.sqrt (2 / (n * (n - 1))) :=
by
  sorry

end sqrt_inequality_l499_499914


namespace quadrilateral_cyclic_tangential_l499_499465

section Geometry

-- Define a quadrilateral with a point inside matching the conditions.
variables (A B C D M : ℝ)

-- Conditions for equidistance and area.
variables (h1 : dist_from_line M A B = dist_from_line M C D)
variables (h2 : dist_from_line M B C = dist_from_line M A D)
variables (h3 : area_of_quadrilateral A B C D = dist M A * dist M C + dist M B * dist M D)

-- The proof statements that the quadrilateral is cyclic and tangential.
theorem quadrilateral_cyclic_tangential :
  is_cyclic_quadrilateral A B C D ∧ is_tangential_quadrilateral A B C D :=
by
  sorry

end Geometry

end quadrilateral_cyclic_tangential_l499_499465


namespace gcd_sum_l499_499154

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499154


namespace sum_of_perpendiculars_l499_499267

theorem sum_of_perpendiculars {a b : ℝ} (h1 : a^2 + b^2 = 12^2) : 
  a + b = 12 :=
begin
  sorry
end

end sum_of_perpendiculars_l499_499267


namespace license_plate_count_l499_499491

theorem license_plate_count : 
  (∃ (first : Fin 2) (second : Fin 4) (third : Fin 3), true) → 
  card (Σ (first : Fin 2) (second : Fin 4) (third : Fin 3), true) = 24 := 
by
  sorry

end license_plate_count_l499_499491


namespace value_of_c_over_b_l499_499985

def is_median (a b c : ℤ) (m : ℤ) : Prop :=
a < b ∧ b < c ∧ m = b

def in_geometric_progression (p q r : ℤ) : Prop :=
∃ k : ℤ, k ≠ 0 ∧ q = p * k ∧ r = q * k

theorem value_of_c_over_b (a b c p q r : ℤ) 
  (h1 : (a + b + c) / 3 = (b / 2))
  (h2 : a * b * c = 0)
  (h3 : a < b ∧ b < c ∧ a = 0)
  (h4 : p < q ∧ q < r ∧ r ≠ 0)
  (h5 : in_geometric_progression p q r)
  (h6 : a^2 + b^2 + c^2 = (p + q + r)^2) : 
  c / b = 2 := 
sorry

end value_of_c_over_b_l499_499985


namespace factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l499_499293

noncomputable def factor1 (a : ℝ) : ℝ := a^3 - 9a
noncomputable def factor2 (x y : ℝ) : ℝ := 3x^2 - 6xy + x
noncomputable def factor3 (n m : ℝ) : ℝ := n^2 * (m - 2) + n * (2 - m)
noncomputable def factor4 (x y : ℝ) : ℝ := -4x^2 + 4xy + y^2
noncomputable def factor5 (a : ℝ) : ℝ := a^2 + 2a - 8

theorem factorization_1 (a : ℝ) : factor1 a = a * (a + 3) * (a - 3) := 
by sorry

theorem factorization_2 (x y : ℝ) : factor2 x y = x * (3x - 6y + 1) := 
by sorry

theorem factorization_3 (n m : ℝ) : factor3 n m = n * (m - 2) * (n - 1) := 
by sorry

theorem factorization_4 (x y : ℝ) : factor4 x y = ((2 + 2 * Real.sqrt 2) * x + y) * ((2 - 2 * Real.sqrt 2) * x + y) := 
by sorry

theorem factorization_5 (a : ℝ) : factor5 a = (a - 2) * (a + 4) := 
by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_factorization_5_l499_499293


namespace ellipse_equation_point_P_existence_l499_499765

/-- Given an ellipse with equation (x^2)/(a^2) + (y^2)/(b^2) = 1, a > b > 0 and eccentricity e = 1/2,
    find the equation of the ellipse and verify the given conditions. -/
theorem ellipse_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (eccentricity : a * 2 * (1 / 2))
  (line_AB_perpendicular_length : 2 * b^2 / a = 3) :
  (x y : ℝ) → (x^2 / 4) + (y^2 / 3) = 1 :=
sorry

/-- Given an ellipse with equation (x^2)/(a^2) + (y^2)/(b^2) = 1, a > b > 0 and eccentricity e = 1/2,
    if a line l passing through the right focus intersects the ellipse at A and B,
    and the line is not perpendicular to the x-axis,
    show that there exists a point P(4, 0) on the x-axis such that the distances from any point on x-axis to the lines PA and PB are equal. -/
theorem point_P_existence
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (eccentricity : a * 2 * (1 / 2))
  (line_AB_non_perpendicular : 2 * b^2 / a ≠ 3) :
  ∃ P : ℝ × ℝ, P = (4, 0) :=
sorry

end ellipse_equation_point_P_existence_l499_499765


namespace initial_concentration_is_30_percent_l499_499572

theorem initial_concentration_is_30_percent :
  ∀ (C : ℝ) (weight_initial_solution weight_pure_water total_weight salt_percentage : ℝ),
  weight_initial_solution = 100 ∧
  weight_pure_water = 200 ∧
  total_weight = weight_initial_solution + weight_pure_water ∧
  salt_percentage = 10 ∧
  C / 100 * weight_initial_solution = salt_percentage / 100 * total_weight →
  C = 30 :=
by
  intros C weight_initial_solution weight_pure_water total_weight salt_percentage
  intro h
  cases h with h_weight_initial_solution h_rest
  cases h_rest with h_weight_pure_water h_rest
  cases h_rest with h_total_weight h_rest
  cases h_rest with h_salt_percentage h_equation
  have H : weight_initial_solution + weight_pure_water = 300 :=
    by rw [h_weight_initial_solution, h_weight_pure_water]
  rw [H] at h_total_weight
  norm_num at *
  sorry

end initial_concentration_is_30_percent_l499_499572


namespace proportion_difference_l499_499799

theorem proportion_difference : (0.80 * 40) - ((4 / 5) * 20) = 16 := 
by 
  sorry

end proportion_difference_l499_499799


namespace length_B1_A1_l499_499988

section Perpendicular_Setup

-- Define the points on the circle
variables (A1 A2 A3 A4 A5 A6 : ℝ)

-- Define the radius of the circle
def radius : ℝ := 1

-- Define point B1 on the ray l1
variable (B1 : ℝ)

-- Define the function that checks if the sequence of perpendiculars return to B1
def sequence_returns_to_B1 : Prop :=
  ∀ P, (P = B1) ↔ (perpendicular_drop P = P)

-- Theorem statement to prove the length of segment B1A1
theorem length_B1_A1 (h: sequence_returns_to_B1 B1) : B1 = 2 := 
sorry

end Perpendicular_Setup

end length_B1_A1_l499_499988


namespace gcd_sum_5n_plus_6_l499_499200

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499200


namespace gcd_sum_5n_plus_6_l499_499198

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499198


namespace cyclic_quad_radius_l499_499012

theorem cyclic_quad_radius (AB BC CD DA : ℝ) (h1 : AB = 10) (h2 : BC = 8) (h3 : CD = 25) (h4 : DA = 12) :
  let r := sqrt (1209 / 7)
  in is_tangent_circle_radius AB BC CD DA r :=
by
  -- the content of the proof can be added here
  sorry

end cyclic_quad_radius_l499_499012


namespace sum_of_consecutive_neg_ints_l499_499510

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l499_499510


namespace segment_EF_length_l499_499842

-- Definitions from conditions
variables (A B C D E F : Point)
variables (AB BC AD BEFD : ℝ)
variables (rectABCD : Rectangle A B C D)

-- Lean 4 statement for the problem
theorem segment_EF_length (h1 : AB = 6) (h2 : BC = 8)
  (h3 : folded_pentagon : fold_rectangle_point A D = A ∧ pentagon A B E F D) :
  EF = 15 / 2 :=
by {
  sorry
}

end segment_EF_length_l499_499842


namespace three_parallel_reflections_l499_499474

-- Definitions for reflections and parallel lines.
variables {Point : Type} [MetricSpace Point]
variables (l1 l2 l3 : Line Point)

-- Conditions
axiom lines_parallel : l1 ∥ l2 ∧ l2 ∥ l3
axiom reflection_l1 : ∀ p : Point, reflect l1 p = σ_l1 p
axiom reflection_l2 : ∀ p : Point, reflect l2 p = σ_l2 p
axiom reflection_l3 : ∀ p : Point, reflect l3 p = σ_l3 p

-- Theorem statement
theorem three_parallel_reflections :
  ∃ l : Line Point, (σ_l3 ∘ σ_l2 ∘ σ_l1) = reflect l :=
sorry

end three_parallel_reflections_l499_499474


namespace complex_value_permutations_l499_499829

noncomputable theory

open Complex

theorem complex_value_permutations 
  (z1 z2 z3 : Complex) 
  (h1 : abs z1 = 1) 
  (h2 : abs z2 = 1) 
  (h3 : abs z3 = 1) 
  (h4 : z1 + z2 + z3 = 1) 
  (h5 : z1 * z2 * z3 = 1) :
  (z1 = 1 ∧ z2 = Complex.i ∧ z3 = -Complex.i) ∨
  (z1 = 1 ∧ z2 = -Complex.i ∧ z3 = Complex.i) ∨
  (z1 = Complex.i ∧ z2 = 1 ∧ z3 = -Complex.i) ∨
  (z1 = Complex.i ∧ z2 = -Complex.i ∧ z3 = 1) ∨
  (z1 = -Complex.i ∧ z2 = 1 ∧ z3 = Complex.i) ∨
  (z1 = -Complex.i ∧ z2 = Complex.i ∧ z3 = 1) :=
sorry

end complex_value_permutations_l499_499829


namespace compute_expression_l499_499644

theorem compute_expression (x : ℤ) (h : x = 6) :
  ((x^9 - 24 * x^6 + 144 * x^3 - 512) / (x^3 - 8) = 43264) :=
by
  sorry

end compute_expression_l499_499644


namespace possible_sums_l499_499731

theorem possible_sums :
  ∃ S : ℤ, 
    (∃ a b c d e f g h i : ℤ, 
      multiset.mem a [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem b [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem c [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem d [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem e [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem f [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem g [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem h [2, 4, 6, 8, 10, 12, 14] ∧ 
      multiset.mem i [2, 4, 6, 8, 10, 12, 14] ∧ 
      list.nodup [a, b, c, d, e, f, g, h, i] ∧ 
      a + b + c = S ∧ 
      d + e + f = S ∧ 
      g + h + i = S) ∧ 
    (S = 20 ∨ S = 24 ∨ S = 28) := sorry

end possible_sums_l499_499731


namespace number_of_closed_lockers_l499_499607

theorem number_of_closed_lockers : ((
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 100) → 
  (∀ m : ℕ, (1 ≤ m ∧ m ≤ 100) → 
  let initial_state := λ x : ℕ, false in
  let toggle := λ state k : ℕ, state k ⊕ ((n ∣ k) ∧ k ≤ 100) in
  ∀ final_state : (ℕ → Prop), final_state = toggle initial_state m →
  let count_true := λ count : ℕ, ∑ m in (finset.range 100).filter final_state, (1 : ℕ) in
  let count_false := λ count_true : ℕ, 100 - count_true in
  count_false)) := 90 :=
begin
  sorry
end


end number_of_closed_lockers_l499_499607


namespace number_of_R_divides_Q_l499_499747

noncomputable def Q (n : ℕ) (pi : ℕ → ℕ) : ℕ :=
∏ i in finset.range (n+1), pi i ^ pi i

noncomputable def R (n : ℕ) (ei : ℕ → ℕ) : ℕ :=
∏ i in finset.range (n+1), ei i ^ ei i

theorem number_of_R_divides_Q : ∃ count : ℕ, count = 75 ∧ 
  (∀ n ∈ finset.range 71, 
    R n (λ i, nat.prime_factors n i) ∣ Q n (λ i, nat.primes i)) :=
begin
  sorry
end

end number_of_R_divides_Q_l499_499747


namespace log_comparison_l499_499808

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l499_499808


namespace fraction_inequality_solution_set_l499_499516

theorem fraction_inequality_solution_set : 
  {x : ℝ | (2 - x) / (x + 4) > 0} = {x : ℝ | -4 < x ∧ x < 2} :=
by sorry

end fraction_inequality_solution_set_l499_499516


namespace eval_expression_l499_499278

theorem eval_expression : 
  (-(1/2))⁻¹ - 4 * real.cos (30 * real.pi / 180) - (real.pi + 2013)^0 + real.sqrt 12 = -3 := 
by 
  sorry

end eval_expression_l499_499278


namespace infinite_geometric_series_sum_l499_499667

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * r ^ n = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499667


namespace percentage_of_quarters_l499_499011

/--
Given:
- 60 dimes, each worth 10 cents
- 30 quarters, each worth 25 cents
- 40 nickels, each worth 5 cents

The value of all coins is calculated and the percentage of the total value in quarters is determined.
-/
theorem percentage_of_quarters : 
  let dimes := 60
  let quarters := 30
  let nickels := 40
  let value_dime := 10
  let value_quarter := 25
  let value_nickel := 5
  let total_value := quarters * value_quarter + dimes * value_dime + nickels * value_nickel
  let quarters_value := quarters * value_quarter
  (quarters_value.toFloat / total_value.toFloat * 100).round(1) = 48.4 :=
by
  sorry

end percentage_of_quarters_l499_499011


namespace calculate_expression_l499_499275

noncomputable def expr : ℚ := (5 - 2 * (3 - 6 : ℚ)⁻¹ ^ 2)⁻¹

theorem calculate_expression :
  expr = (9 / 43 : ℚ) := by
  sorry

end calculate_expression_l499_499275


namespace centroid_distances_relationship_l499_499421

variables {A B C O : Type}
variable [metric_space O]

-- Given: O is the centroid of triangle ABC
def is_centroid (O A B C : O) := 
  ∃ (G : O), dist G O = 1 / 3 * (dist O A + dist O B + dist O C)

-- s1 is the sum of distances from O to vertices
def s1 (O A B C: O) := dist O A + dist O B + dist O C

-- s2 is the sum of the side lengths of the triangle
def s2 (A B C: O) := dist A B + dist B C + dist C A

-- Theorem: Relationship between s1 and s2
theorem centroid_distances_relationship (O A B C: O) 
    (h : is_centroid O A B C): 
    s1 O A B C = 3 / 2 * s2 A B C :=
by
  sorry

end centroid_distances_relationship_l499_499421


namespace sum_of_gcd_values_l499_499228

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499228


namespace symmetric_point_with_respect_to_y_eq_x_l499_499494

theorem symmetric_point_with_respect_to_y_eq_x :
  ∃ x₀ y₀ : ℝ, (∃ (M : ℝ × ℝ), M = (3, 1) ∧
  ((x₀ + 3) / 2 = (y₀ + 1) / 2) ∧
  ((y₀ - 1) / (x₀ - 3) = -1)) ∧
  (x₀ = 1 ∧ y₀ = 3) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l499_499494


namespace sum_gcd_values_l499_499264

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499264


namespace proof_problem_l499_499399

-- Define the conditions in Lean
def a : ℝ := 3
def cos_B : ℝ := -1 / 2

-- Assume b and c are real numbers such that b - c = 2
variables (b c : ℝ)
def condition_bc := b - c = 2

-- Define the proof targets
def target_b_c := b = 7 ∧ c = 5
def target_sin_BC := Real.sin (Real.arccos (-1 / 2) + Real.arccos (-1 / 2)) = (3 * Real.sqrt 3) / 14

-- The statement of the proof problem
theorem proof_problem (h_bc : condition_bc) (h_cos_B : cos_B = -1 / 2) : target_b_c ∧ target_sin_BC :=
by
  sorry

end proof_problem_l499_499399


namespace ratio_of_growth_l499_499598

def growth_first_year := 2
def growth_second_year := growth_first_year + 0.5 * growth_first_year
def growth_third_year := growth_second_year + 0.5 * growth_second_year
def total_height (G4 : ℝ) := growth_first_year + growth_second_year + growth_third_year + G4 + (1 / 2) * G4

theorem ratio_of_growth (G4 : ℝ) (h : total_height G4 = 23) : (G4 / growth_third_year) = 2 := by
  sorry

end ratio_of_growth_l499_499598


namespace probability_chinese_on_first_given_math_on_second_l499_499610

-- Context: schoolbag has 2 math books and 2 Chinese books
-- Question: Prove P(A|B) = 2/3 where:
-- Event A: First draw is a Chinese book
-- Event B: Second draw is a math book

theorem probability_chinese_on_first_given_math_on_second : 
  (conditional_probability 
    (event_first_draw_chinese ∧ event_second_draw_math) 
    event_second_draw_math = 2 / 3 :=
sorry

end probability_chinese_on_first_given_math_on_second_l499_499610


namespace P_n_recurrence_l499_499923

def P : ℕ → ℕ
| 0       := 0
| 1       := 0
| 2       := 3
| (n + 3) := P (n + 2) + 2 * P n

theorem P_n_recurrence (n : ℕ) (h : n ≥ 3) : P (n + 1) = P n + 2 * P (n - 1) :=
by {
  cases n with n,
  { simp [P] },
  cases n with n,
  { simp [P] },
  cases n with n,
  repeat { simp [P] }
}

end P_n_recurrence_l499_499923


namespace solve_for_x_l499_499915

theorem solve_for_x : ∃ x : ℚ, (1/3 : ℚ) + (1/4) = (1/x) ∧ x = 12/7 := 
by
  use 12/7
  split
  simp [*, (12 : ℚ).symm, (3 : ℚ).symm, (4 : ℚ).symm]
  norm_num
  sorry

end solve_for_x_l499_499915


namespace range_of_f_l499_499950

theorem range_of_f (x : ℝ) : 
  (∀ x, (1 - x^2 ≥ 0 → √(1 - x^2) + √(x^2 - 1) = 0) ∧ (x ^ 2 - 1 ≥ 0 → √(1 - x^2) + √(x^2 - 1) = 0)) :=
by {
  sorry
}

end range_of_f_l499_499950


namespace dice_to_buy_l499_499445

variable (mark_dice : ℕ) (mark_percent : ℝ)
variable (james_dice : ℕ) (james_percent : ℝ)
variable (sarah_dice : ℕ) (sarah_percent : ℝ)
variable (total_required_dice : ℕ)

def mark_12_sided_dice : ℝ := mark_percent * mark_dice
def james_12_sided_dice : ℝ := james_percent * james_dice
def sarah_12_sided_dice : ℝ := sarah_percent * sarah_dice

def total_12_sided_dice : ℝ := mark_12_sided_dice + (james_12_sided_dice + sarah_12_sided_dice)

def dice_needed : ℝ := total_required_dice - total_12_sided_dice

theorem dice_to_buy :
  mark_dice = 10 → mark_percent = 0.60 → 
  james_dice = 8 → james_percent = 0.75 → 
  sarah_dice = 12 → sarah_percent = 0.50 → 
  total_required_dice = 21 → 
  dice_needed = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h_mark := by rw [h1, h2]; norm_num
  have h_james := by rw [h3, h4]; norm_num
  have h_sarah := by rw [h5, h6]; norm_num
  have h_total := by rw [h_mark, h_james, h_sarah]; norm_num
  rw h_total at h6
  norm_num at h6
  exact h6

end dice_to_buy_l499_499445


namespace largest_N_not_payable_l499_499899

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499899


namespace gcd_sum_l499_499089

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499089


namespace second_order_derivative_parametric_l499_499019

theorem second_order_derivative_parametric :
  (∃ t : ℝ, 
    (λ t, (let x := (1 / t^2), y := (1 / (t^2 + 1)) in
      (d^2 / dx^2 y))) = - (2 * t^6) / (1 + t^2)^3) := by
  sorry

end second_order_derivative_parametric_l499_499019


namespace gcd_values_sum_l499_499073

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499073


namespace rectangle_coordinate_of_circle_polar_coordinate_of_line_area_of_triangle_C1MN_l499_499789

noncomputable def line_l1 (t : ℝ) : ℝ × ℝ :=
  (t, sqrt(3) * t)

noncomputable def circle_C1 (ρ θ : ℝ) : ℝ :=
  ρ^2 - 2 * sqrt(3) * ρ * cos θ - 4 * ρ * sin θ + 6

theorem rectangle_coordinate_of_circle :
  ∀ x y : ℝ, ((x - sqrt(3))^2 + (y - 2)^2 = 1) ↔
  ∃ (ρ θ : ℝ), (ρ^2 - 2 * sqrt(3) * ρ * cos θ - 4 * ρ * sin θ + 6 = 0) ∧ (x = ρ * cos θ) ∧ (y = ρ * sin θ) :=
by 
  sorry

theorem polar_coordinate_of_line :
  (∀ t : ℝ, let (x, y) := line_l1 t in y = sqrt(3) * x) ↔ 
  ∀ ρ : ℝ, θ = π / 3 :=
by
  sorry

theorem area_of_triangle_C1MN :
  ∀ ρ1 ρ2 : ℝ, 
  ρ1^2 - 3 * sqrt(3) * ρ1 + 6 = 0 ∧ 
  ρ2^2 - 3 * sqrt(3) * ρ2 + 6 = 0 ∧ 
  ρ1 ≠ ρ2 → 
  let base := abs (ρ1 - ρ2) in
  base = sqrt(3) → 
  let height := 1 / 2 in
  (1 / 2) * base * height = sqrt(3) / 4 :=
by 
  sorry

end rectangle_coordinate_of_circle_polar_coordinate_of_line_area_of_triangle_C1MN_l499_499789


namespace value_two_stddevs_less_l499_499928

theorem value_two_stddevs_less (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : μ - 2 * σ = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stddevs_less_l499_499928


namespace train_length_150_meters_l499_499053

theorem train_length_150_meters
  (speed_kmh : ℕ)
  (time_seconds : ℕ)
  (bridge_length_meters : ℕ)
  (speed_conversion : speed_kmh = 45)
  (time_conversion : time_seconds = 30)
  (bridge_length_conversion : bridge_length_meters = 225) :
  let speed_mps := speed_kmh * 1000 / 3600,
      distance := speed_mps * time_seconds in
  ∃ train_length_meters : ℕ, train_length_meters = distance - bridge_length_meters ∧ train_length_meters = 150 := 
by
  sorry

end train_length_150_meters_l499_499053


namespace common_point_circumcircles_l499_499390

-- Definitions of points and convex quadrilateral
variables {A B C D E F : Type*}
variable [euclidean_geometry A B C D E F] 

-- Points and their relationships
def is_convex (A B C D : Type*) : Prop := sorry

def on_side (E : Type*) (A B : Type*) : Prop := sorry

-- Definitions of geometric relationships specific to the problem
def intersect (AC DE : Type*) (F : Type*) : Prop := sorry

-- Circumcircles of specified triangles
def on_circumcircle (P : Type*) (Δ : triangle) : Prop := sorry

-- Given conditions
axiom h1 : is_convex A B C D
axiom h2 : on_side E A B
axiom h3 : intersect A C D E F

-- Theorem statement
theorem common_point_circumcircles :
  on_circumcircle C (triangle.mk A B C) ∧ 
  on_circumcircle C (triangle.mk C D F) ∧ 
  on_circumcircle C (triangle.mk B D E) :=
sorry

end common_point_circumcircles_l499_499390


namespace john_burritos_left_l499_499850

-- define the variables from the conditions
def n_b := 3       -- number of boxes
def b_b := 20      -- burritos per box
def frac_give := 1/3  -- fraction given away
def e_d := 3       -- burritos eaten per day
def d := 10        -- number of days

-- defining the proof problem
theorem john_burritos_left :
  let total := n_b * b_b in
  let gave_away := frac_give * total in
  let remaining_after_give := total - gave_away in
  let ate := e_d * d in
  let burritos_left := remaining_after_give - ate in
  burritos_left = 10 :=
by
  -- Here we'll say here the steps leading to the correct answer but skip the actual proof.
  have total_calc : total = 60 := by sorry
  have gave_away_calc : gave_away = 20 := by sorry
  have remaining_after_give_calc : remaining_after_give = 40 := by sorry
  have ate_calc : ate = 30 := by sorry
  have burritos_left_calc : burritos_left = 10 := by sorry
  exact burritos_left_calc

end john_burritos_left_l499_499850


namespace gcd_sum_l499_499162

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499162


namespace gcd_values_sum_l499_499079

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499079


namespace infinite_geometric_series_sum_l499_499691

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499691


namespace sum_gcd_values_l499_499257

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499257


namespace solve_first_equation_solve_second_equation_l499_499487

open Real

/-- Prove solutions to the first equation (x + 8)(x + 1) = -12 are x = -4 and x = -5 -/
theorem solve_first_equation (x : ℝ) : (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 := by
  sorry

/-- Prove solutions to the second equation 2x^2 + 4x - 1 = 0 are x = (-2 + sqrt 6) / 2 and x = (-2 - sqrt 6) / 2 -/
theorem solve_second_equation (x : ℝ) : 2 * x^2 + 4 * x - 1 = 0 ↔ x = (-2 + sqrt 6) / 2 ∨ x = (-2 - sqrt 6) / 2 := by
  sorry

end solve_first_equation_solve_second_equation_l499_499487


namespace gcd_sum_l499_499088

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499088


namespace compare_logs_l499_499816

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l499_499816


namespace sum_of_given_infinite_geometric_series_l499_499711

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499711


namespace twelve_points_probability_l499_499538

/-- Given a set of twelve points equally spaced around a 3x3 square (four at corners and midpoints of each side), 
  the probability of choosing two points that are one unit apart is 2/11 -/
theorem twelve_points_probability 
  (points : Finset (ℝ × ℝ))
  (h1 : points.card = 12)
  (h2 : ∀ p ∈ points, p ∈ { (0, 0), (0, 3), (3, 0), (3, 3), (0, 1.5), (1.5, 0), (1.5, 3), (3, 1.5), (0.75, 0), (0, 0.75), (2.25, 0), (0, 2.25)})
  : (∃ pair : Finset (Finset (ℝ × ℝ)), pair.card = 12 ∧ ∀ q ∈ pair, ∀ p ∈ q, (∃ (a b : ℝ × ℝ), a ∈ points ∧ b ∈ points ∧ a ≠ b ∧ dist a b = 1))
    → (12 / (66 : ℝ)) = (2 / 11 : ℝ) :=
by
  sorry

end twelve_points_probability_l499_499538


namespace compare_logs_l499_499822

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l499_499822


namespace max_age_l499_499408

-- Definitions of the conditions
def born_same_day (max_birth luka_turn4 : ℕ) : Prop := max_birth = luka_turn4
def age_difference (luka_age aubrey_age : ℕ) : Prop := luka_age = aubrey_age + 2
def aubrey_age_on_birthday : ℕ := 8

-- Prove that Max's age is 6 years when Aubrey is 8 years old
theorem max_age (luka_birth aubrey_birth max_birth : ℕ) 
                (h1 : born_same_day max_birth luka_birth) 
                (h2 : age_difference luka_birth aubrey_birth) : 
                (aubrey_birth + 4 - luka_birth) = 6 :=
by
  sorry

end max_age_l499_499408


namespace game_probability_difference_l499_499582

theorem game_probability_difference :
  let
    p_heads : ℚ := 2 / 3,
    p_tails : ℚ := 1 / 3,
    P_game_C : ℚ := p_heads^4 + p_tails^4,
    P_game_D : ℚ := (p_heads^3 + p_tails^3) * (p_heads^2 + p_tails^2),
    difference : ℚ := P_game_C - P_game_D
  in
  difference = 2 / 81 :=
by
  have h1 : p_heads = 2 / 3 := rfl
  have h2 : p_tails = 1 / 3 := rfl
  have P_game_C_calc : P_game_C = (2 / 3)^4 + (1 / 3)^4 := by
    rw [h1, h2]
    sorry
  have P_game_D_calc : P_game_D = ((2 / 3)^3 + (1 / 3)^3) * ((2 / 3)^2 + (1 / 3)^2) := by
    rw [h1, h2]
    sorry
  have P_game_C_val : P_game_C = 17 / 81 := by
    rw P_game_C_calc
    sorry
  have P_game_D_val : P_game_D = 5 / 27 := by
    rw P_game_D_calc
    sorry
  have diff_calc : difference = 17 / 81 - 5 / 27 := by
    rw [P_game_C_val, P_game_D_val]
    sorry
  have simplify_diff : 17 / 81 - 5 / 27 = 2 / 81 := by
    sorry
  rw diff_calc
  exact simplify_diff

end game_probability_difference_l499_499582


namespace randy_piggy_bank_final_amount_l499_499480

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end randy_piggy_bank_final_amount_l499_499480


namespace gcd_sum_divisors_eq_12_l499_499173

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499173


namespace number_of_rows_with_7_eq_5_l499_499291

noncomputable def number_of_rows_with_7_people (x y : ℕ) : Prop :=
  7 * x + 6 * (y - x) = 59

theorem number_of_rows_with_7_eq_5 :
  ∃ x y : ℕ, number_of_rows_with_7_people x y ∧ x = 5 :=
by {
  sorry
}

end number_of_rows_with_7_eq_5_l499_499291


namespace odd_two_digit_combinations_l499_499779

theorem odd_two_digit_combinations (digits : Finset ℕ) (h_digits : digits = {1, 3, 5, 7, 9}) :
  ∃ n : ℕ, n = 20 ∧ (∃ a b : ℕ, a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (10 * a + b) % 2 = 1) :=
by
  sorry

end odd_two_digit_combinations_l499_499779


namespace alpha_is_integer_l499_499841

-- Define the fractional part function
def fractional_part (x : ℝ) : ℝ := x - x.floor

-- Statement of the problem in terms of Lean definitions and conditions
theorem alpha_is_integer (α : ℝ) (finitely_many_distinct_values : set (fractional_part ∧ ℕ) . countable finite) : 
  (∀ n : ℕ, fractional_part (α ^ n) ∈ finitely_many_distinct_values) → (∃ k : ℤ, α = k) :=
by
  sorry

end alpha_is_integer_l499_499841


namespace place_integers_on_cube_l499_499403

theorem place_integers_on_cube:
  ∃ (A B C D A₁ B₁ C₁ D₁ : ℤ),
    A = B + D + A₁ ∧ 
    B = A + C + B₁ ∧ 
    C = B + D + C₁ ∧ 
    D = A + C + D₁ ∧ 
    A₁ = B₁ + D₁ + A ∧ 
    B₁ = A₁ + C₁ + B ∧ 
    C₁ = B₁ + D₁ + C ∧ 
    D₁ = A₁ + C₁ + D :=
sorry

end place_integers_on_cube_l499_499403


namespace boat_problem_l499_499587

theorem boat_problem (x n : ℕ) (h1 : n = 7 * x + 5) (h2 : n = 8 * x - 2) :
  n = 54 ∧ x = 7 := by
sorry

end boat_problem_l499_499587


namespace gcd_sum_5n_plus_6_n_l499_499149

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499149


namespace concurrency_of_lines_l499_499854

-- Definitions of geometrical objects and conditions
variables {A B C D I J K L : Point}
variables {ω_A ω_B : Circle}

-- Assume a convex quadrilateral ABCD
axiom convex_ABCD : convex_quad A B C D

-- Assume incircles ω_A and ω_B of triangles ACD and BCD with centers I and J respectively.
axiom incircle_triang_ACD : incircle ω_A A C D I
axiom incircle_triang_BCD : incircle ω_B B C D J

-- Assume the second common external tangent to ω_A and ω_B touches ω_A at K and ω_B at L.
axiom tangent_touch_ωA_K : tangent ω_A K
axiom tangent_touch_ωB_L : tangent ω_B L

-- Prove that lines AK, BL, and IJ are concurrent
theorem concurrency_of_lines : concurrent (line_through A K) (line_through B L) (line_through I J) :=
sorry

end concurrency_of_lines_l499_499854


namespace area_region_outside_C₁_inside_C₂_l499_499439

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def AB : Real := 1
noncomputable def BC : Real := 1
noncomputable def AC : Real := 2
noncomputable def C₁ : Circle := ⟨A, AB⟩ -- Circle centered at A with radius AB
noncomputable def C₂ : Circle := ⟨A, AC⟩ -- Circle centered at A with radius AC

theorem area_region_outside_C₁_inside_C₂ :
  area (region_outside C₁ ∩ region_inside C₂) = 3 * π := 
sorry

end area_region_outside_C₁_inside_C₂_l499_499439


namespace infinite_geometric_series_sum_l499_499723

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499723


namespace sum_of_consecutive_neg_ints_l499_499509

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l499_499509


namespace inequality_for_positive_x_l499_499477

variables (x : ℝ)

theorem inequality_for_positive_x (hx : x > 0) :
  2^(real.rpow x (1/12)) + 2^(real.rpow x (1/4)) ≥ 2 * 2^(real.rpow x (1/6)) :=
sorry

end inequality_for_positive_x_l499_499477


namespace dorothy_money_left_correct_l499_499659

noncomputable def annualIncome : ℕ := 60000
noncomputable def taxes (income : ℕ) : ℕ :=
  let bracket1 := min 10000 income
  let bracket2 := min 40000 (income - bracket1)
  let bracket3 := max 0 (income - (bracket1 + bracket2))
  let tax1 := bracket1 * 10 / 100
  let tax2 := bracket2 * 15 / 100
  let tax3 := bracket3 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def monthlyBills : ℕ := 800 * 12
noncomputable def healthcarePremiums : ℕ := 300 * 12
noncomputable def retirementContributions (income : ℕ) : ℕ :=
  income * 6 / 100
noncomputable def savingsGoal : ℕ := 5000

noncomputable def totalSpent (income : ℕ) : ℕ :=
  taxes income + monthlyBills + healthcarePremiums
  + retirementContributions income + savingsGoal

noncomputable def moneyLeft (income : ℕ) : ℕ :=
  income - totalSpent income

theorem dorothy_money_left_correct :
  moneyLeft annualIncome = 28700 := by
  rw [annualIncome, taxes, monthlyBills, healthcarePremiums,
      retirementContributions, savingsGoal, totalSpent, moneyLeft]
  -- detailed steps are skipped with sorry for brevity
  sorry

end dorothy_money_left_correct_l499_499659


namespace smallest_number_two_reps_l499_499741

theorem smallest_number_two_reps : 
  ∃ (n : ℕ), (∀ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = n ∧ 3 * x2 + 4 * y2 = n → (x1 = x2 ∧ y1 = y2 ∨ ¬(x1 = x2 ∧ y1 = y2))) ∧ 
  ∀ m < n, (∀ x y : ℕ, ¬(3 * x + 4 * y = m ∧ ¬∃ (x1 y1 : ℕ), 3 * x1 + 4 * y1 = m) ∧ 
            (∃ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = m ∧ 3 * x2 + 4 * y2 = m ∧ ¬(x1 = x2 ∧ y1 = y2))) :=
  sorry

end smallest_number_two_reps_l499_499741


namespace surface_area_relationship_l499_499776

noncomputable def equal_volumes (V : ℝ) := 
  (S1 S2 S3 : ℝ) →
  (R a r : ℝ) →
  (V = (4 / 3) * π * R ^ 3) ∧ 
  (V = a ^ 3) ∧ 
  (V = 2 * π * r ^ 3) ∧ 
  (S1 = 6 * a ^ 2) ∧ 
  (S2 = 4 * π * R ^ 2) ∧
  (S3 = 2 * π * r * (r + r * sqrt 2)) →
  S2 < S3 ∧ S3 < S1

theorem surface_area_relationship (V : ℝ) (S1 S2 S3 : ℝ) (R a r : ℝ) : 
  (V = (4 / 3) * π * R ^ 3) → 
  (V = a ^ 3) → 
  (V = 2 * π * r ^ 3) → 
  (S1 = 6 * a ^ 2) → 
  (S2 = 4 * π * R ^ 2) → 
  (S3 = 2 * π * r * (r + r * sqrt 2)) → 
  S2 < S3 ∧ S3 < S1 :=
  by sorry

end surface_area_relationship_l499_499776


namespace find_a_b_extreme_values_l499_499347

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b / x

def passes_through_P (a b : ℝ) : Prop := f a b 1 = -5

def tangent_at_P (a b : ℝ) : Prop := a - b = 3

theorem find_a_b :
  ∃ a b : ℝ, 
    passes_through_P a b ∧ 
    tangent_at_P a b ∧ 
    a = -1 ∧ 
    b = -4 := 
sorry

theorem extreme_values :
  let f' := λ x : ℝ, (4 - x^2) / x^2 in
  (f' (-2) = 0 → f (-1) (-4) (-2) = 4) ∧
  (f' 2 = 0 → f (-1) (-4) 2 = -4) :=
sorry

end find_a_b_extreme_values_l499_499347


namespace combination_value_gives_specific_n_l499_499666

theorem combination_value_gives_specific_n :
  ∃ n : ℕ, (nat.factorial 98 / (nat.factorial n * nat.factorial (98 - n)) = 4753) ∧ (98 - n = 2) := sorry

end combination_value_gives_specific_n_l499_499666


namespace triangle_inequality_l499_499865

variables {a b c A B C : ℝ}

-- Conditions
axiom triangle_sides (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  : a + b > c ∧ b + c > a ∧ c + a > b

axiom triangle_angles (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C)
  (h7 : A + B + C = Real.pi)
  : A + B + C = π

-- To Prove
theorem triangle_inequality 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  ∑ (a + b) * Real.sin (C / 2) ≤ a + b + c :=
sorry

end triangle_inequality_l499_499865


namespace compare_logs_l499_499812

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l499_499812


namespace hyperbola_distance_to_x_axis_l499_499329

theorem hyperbola_distance_to_x_axis 
  (a : ℝ) (x y : ℝ) ( sqrt_5 : ℝ := real.sqrt 5) 
  (h_focus_distance_a : a = 2)
  (hx2 : real.sqrt (x^2 + y^2) = a * (real.sqrt(5)/2)) 
  (h_eq : x^2 / a^2 - y^2 = 1) :
  abs y = real.sqrt 5 / 5 := 
sorry

end hyperbola_distance_to_x_axis_l499_499329


namespace sum_gcd_possible_values_eq_twelve_l499_499240

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499240


namespace infinite_geometric_series_sum_l499_499730

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499730


namespace gcd_sum_5n_plus_6_n_l499_499152

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499152


namespace log_216_eq_3_log_2_add_3_log_3_l499_499022

theorem log_216_eq_3_log_2_add_3_log_3 (log : ℝ → ℝ) (h1 : ∀ x y, log (x * y) = log x + log y)
  (h2 : ∀ x n, log (x^n) = n * log x) :
  log 216 = 3 * log 2 + 3 * log 3 :=
by
  sorry

end log_216_eq_3_log_2_add_3_log_3_l499_499022


namespace sum_of_gcd_values_l499_499227

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499227


namespace sum_of_gcd_values_l499_499120

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499120


namespace probability_winning_ticket_l499_499905

noncomputable def valid_numbers : List ℕ := [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50]

def sum_log (s: List ℕ) : ℝ := s.map (λ x, Real.log x / Real.log 10).sum

def is_power_of_ten_sum (s: List ℕ) : Prop :=
  (sum_log s).floor = sum_log s

def is_even_sum (s: List ℕ) : Prop :=
  (s.sum % 2 = 0)

def valid_ticket (s: List ℕ) : Prop :=
  s.length = 7 ∧ is_power_of_ten_sum s ∧ is_even_sum s

theorem probability_winning_ticket :
  ∃ (valid_combinations: ℕ), valid_combinations > 0 ∧ 
  let total_tickets := valid_combinations in
  (1:ℚ) / total_tickets = (1/5:ℚ) :=
sorry

end probability_winning_ticket_l499_499905


namespace compare_logs_l499_499817

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l499_499817


namespace geometric_series_sum_l499_499675

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499675


namespace solve_quadratic_l499_499802

theorem solve_quadratic (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end solve_quadratic_l499_499802


namespace problem_A_plus_B_l499_499426

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l499_499426


namespace concyclic_points_ratio_l499_499418

theorem concyclic_points_ratio {A B C D E : Type} 
  (h_concyclic : ∃ (circle : Type), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle)
  (h_intersection : ∃ (line1 line2 : Type), A ∈ line1 ∧ B ∈ line1 ∧ C ∈ line2 ∧ D ∈ line2 ∧ E = line1 ∩ line2) :
  (AC / BC) * (AD / BD) = (AE / BE) := 
by sorry

end concyclic_points_ratio_l499_499418


namespace distinct_sequences_count_l499_499358

theorem distinct_sequences_count :
  let R := 'R'
  let I := 'I'
  let E := 'E'
  let remaining_letters := ['F', 'N', 'D', 'S', 'H', 'P']
  ∃ s : list Char, s.length = 5 ∧ 
                   s.head = R ∧ 
                   s.getLast = I ∧ 
                   E ∈ s ∧ 
                   (E ≠ s.head) ∧ 
                   (E ≠ s.getLast) ∧ 
                   s.nodup ∧
                   (s.filter (fun c => c ∈ remaining_letters)).length = 3 ∧
                   count_solutions = 30 := sorry

end distinct_sequences_count_l499_499358


namespace XD_and_AM_meet_on_Γ_l499_499868

open Triangle Circle Point Line

variables (A B C : Point) (Γ : Circle) (I : Point) (M : Point) (D : Point)
          (F E X : Point) 

-- Given conditions
axiom triangle_with_incenter (hABC : Triangle A B C) (hΓ : Circumcircle Γ hABC)
axiom incenter_of_triangle (hI : Incenter I hABC)
axiom midpoint_of_BC (hM : Midpoint M B C)
axiom foot_of_perpendicular (hD : FootOfPerpendicular D I B C)
axiom perpendicular_to_AI (hFE : PerpendicularTo I A F E I)
axiom circumcircle_AEF_intersects_Γ (hIntersection : Circumcircle AE F ∩ Γ = {A, X} ∧ X ≠ A)

-- Question to be proven
theorem XD_and_AM_meet_on_Γ (hConditions: triangle_with_incenter hΓ hI ∧ 
                                        midpoint_of_BC hM ∧ 
                                        foot_of_perpendicular hD ∧ 
                                        perpendicular_to_AI hFE ∧ 
                                        circumcircle_AEF_intersects_Γ hIntersection) :
  ∃ Y : Point, Y ∈ Γ ∧ Line X D ∩ Line A M = {Y} :=
  sorry

end XD_and_AM_meet_on_Γ_l499_499868


namespace sum_of_gcd_values_l499_499231

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499231


namespace exists_bad_triple_l499_499458

-- Definitions based on the problem statement
def Team : Type := Fin 16
def played_games : Finset (Sym2 Team) := {game | -- define the 55 game pairs here; this is just a placeholder
 sorry}

-- Bad triple definition: a set of 3 teams such that none of them played each other
def bad_triple (t1 t2 t3 : Team) : Prop :=
  ¬(Sym2.mk t1 t2 ∈ played_games) ∧ ¬(Sym2.mk t1 t3 ∈ played_games) ∧ ¬(Sym2.mk t2 t3 ∈ played_games)

-- Theorem statement
theorem exists_bad_triple :
  ∃ (t1 t2 t3 : Team), bad_triple t1 t2 t3 :=
sorry

end exists_bad_triple_l499_499458


namespace gcd_sum_l499_499132

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499132


namespace gcd_5n_plus_6_n_sum_l499_499103

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499103


namespace sin_cos_tan_θ_sin_cos_tan_2θ_l499_499777

-- Define the point
def P : ℝ × ℝ := (-real.sqrt 3, real.sqrt 6)

-- Define θ to be the angle that has P on its terminal side
def θ := real.atan2 P.2 P.1

-- The proof statements

theorem sin_cos_tan_θ :
  real.sin θ = real.sqrt 6 / 3 ∧
  real.cos θ = -real.sqrt 3 / 3 ∧
  real.tan θ = -real.sqrt 2 :=
sorry

theorem sin_cos_tan_2θ :
  real.sin (2 * θ) = -(2 * real.sqrt 2) / 3 ∧
  real.cos (2 * θ) = -1 / 3 ∧
  real.tan (2 * θ) = 2 * real.sqrt 2 :=
sorry

end sin_cos_tan_θ_sin_cos_tan_2θ_l499_499777


namespace sum_of_gcd_values_l499_499124

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499124


namespace min_val_of_expr_min_val_at_pi_over_4_l499_499299

noncomputable def min_expr_val := 2 * sin θ + sec θ + sqrt 2 * cot θ

theorem min_val_of_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  (min_expr_val θ) ≥ 3 * sqrt 2 := 
sorry

theorem min_val_at_pi_over_4 : min_expr_val (π / 4) = 3 * sqrt 2 :=
sorry

end min_val_of_expr_min_val_at_pi_over_4_l499_499299


namespace train_pass_time_l499_499559

-- Define the speeds and distance in the given conditions
def train_length : ℝ := 110 -- in meters
def train_speed : ℝ := 50 -- in km/h
def man_speed : ℝ := 5 -- in km/h

-- Convert the speeds to meters per second
def kmh_to_mps (speed_in_kmh : ℝ) : ℝ := speed_in_kmh * (5 / 18)

-- Calculate the relative speed in m/s
def relative_speed : ℝ := kmh_to_mps train_speed + kmh_to_mps man_speed

-- Calculate the time taken to pass the man
def time_to_pass (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_pass_time :
  time_to_pass train_length relative_speed = 7.20 :=
by
  -- Skip the proof steps
  sorry

end train_pass_time_l499_499559


namespace rodney_total_commission_l499_499484

-- Definitions of given conditions
variable (second_street_sales : ℕ) (commission_per_sale : ℕ)
variable (first_street_sales : ℕ := second_street_sales / 2)
variable (third_street_sales : ℕ := 0)
variable (fourth_street_sales : ℕ := 1)

-- Sum of sales
def total_sales : ℕ := first_street_sales + second_street_sales + third_street_sales + fourth_street_sales

-- Compute total commission
def total_commission : ℕ := total_sales * commission_per_sale

-- Hypotheses from given problem conditions
theorem rodney_total_commission :
  second_street_sales = 4 ∧
  commission_per_sale = 25 →
  total_commission = 175 :=
by
  sorry

end rodney_total_commission_l499_499484


namespace gcd_sum_divisors_eq_12_l499_499177

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499177


namespace gcd_sum_5n_6_n_eq_12_l499_499184

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499184


namespace gcd_sum_l499_499137

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499137


namespace hannahs_weekly_pay_l499_499357

-- Define conditions
def hourly_wage : ℕ := 30
def total_hours : ℕ := 18
def dock_per_late : ℕ := 5
def late_times : ℕ := 3

-- The amount paid after deductions for being late
def pay_after_deductions : ℕ :=
  let wage_before_deductions := hourly_wage * total_hours
  let total_dock := dock_per_late * late_times
  wage_before_deductions - total_dock

-- The proof statement
theorem hannahs_weekly_pay : pay_after_deductions = 525 := 
  by
  -- No proof necessary; statement and conditions must be correctly written to run
  sorry

end hannahs_weekly_pay_l499_499357


namespace sum_gcd_values_l499_499254

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499254


namespace part1_max_value_f_a_eq_1_part2_unique_real_root_l499_499781

noncomputable def f (x a : ℝ) := log x - a * x
noncomputable def g (x a : ℝ) := (1/2) * x^2 - (2 * a + 1) * x + (a + 1) * log x

-- Part (1): When a = 1, prove that the maximum value of f(x) is -1
theorem part1_max_value_f_a_eq_1 : ∃ x, f x 1 = -1 := by
  sorry

-- Part (2): When a ≥ 1, prove that the equation f(x) = g(x) has a unique real root
theorem part2_unique_real_root (a : ℝ) (h : a ≥ 1) : ∃! x, f x a = g x a := by
  sorry

end part1_max_value_f_a_eq_1_part2_unique_real_root_l499_499781


namespace professors_arrangement_l499_499452

theorem professors_arrangement :
  let students := 6
  let professors := 3
  let total_chairs := 9
  (∃! (gaps : Finset (Fin 5)), gaps.card = professors ∧ ∀ (gap : gaps), gap < (students - 1)) → 
  (Finset.card (Finset.perm (Finset.range (students - 1)).choose professors)) = 60 :=
by
  sorry

end professors_arrangement_l499_499452


namespace equilateral_triangle_shaded_area_sum_l499_499523

theorem equilateral_triangle_shaded_area_sum :
  let s := 18
  let r := s / 2
  let sector_area := (1 / 3) * Math.pi * r^2
  let triangle_area := (Math.sqrt 3 / 4) * s^2
  let shaded_area_for_one_sector := sector_area - triangle_area
  let total_shaded_area := 2 * shaded_area_for_one_sector
  let a := 54
  let b := 162
  let c := 3
  total_shaded_area = 54 * Math.pi - 162 * Math.sqrt 3 ∧ a + b + c = 219 := sorry

end equilateral_triangle_shaded_area_sum_l499_499523


namespace sum_of_gcd_values_l499_499219

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499219


namespace sum_gcd_values_l499_499258

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499258


namespace sum_of_gcd_values_l499_499125

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499125


namespace probability_of_exactly_three_integer_points_l499_499282

-- Define the square S with given endpoints
def S_diagonal_endpoint1 := (1 / 8, 3 / 8)
def S_diagonal_endpoint2 := (-1 / 8, -3 / 8)

-- Define the side length of the square S using the calculated value
def S_side_length : ℝ := (Real.sqrt 10) / 4

-- Define the range for x and y
def x_range := Set.Icc 0 1006
def y_range := Set.Icc 0 1006

-- Define the translated square T(v)
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) := 
  let v_centered := translate_square S_side_length v
  v_centered -- translated square centered at v

-- Given point v is uniformly chosen over the specified range
def uniformly_chosen_v := Set.Prod x_range y_range

-- Define the predicate for T(v) containing exactly 3 integer coordinates
def contains_exactly_three_integer_points (T_v : Set (ℝ × ℝ)) : Prop :=
  Set.card { p : ℤ × ℤ | Set.mem (p.1, p.2) T_v } = 3

-- The final probability statement
theorem probability_of_exactly_three_integer_points :
  let V := uniformly_chosen_v
  let T_v := T
  let P := probability V (λ v, contains_exactly_three_integer_points (T_v v))
  P = 1/100 :=
sorry

end probability_of_exactly_three_integer_points_l499_499282


namespace bryden_payment_l499_499583

theorem bryden_payment :
  (let face_value := 0.25
   let quarters := 6
   let collector_multiplier := 16
   let discount := 0.10
   let initial_payment := collector_multiplier * (quarters * face_value)
   let final_payment := initial_payment - (initial_payment * discount)
   final_payment = 21.6) :=
by
  sorry

end bryden_payment_l499_499583


namespace gcd_values_sum_l499_499077

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499077


namespace min_socks_to_guarantee_pair_l499_499998

theorem min_socks_to_guarantee_pair :
  ∀ (drawer : list (Σ c : fin 4, fin 2)), -- Assume we have 4 colors and 2 socks of each color
    (∀ c : fin 4, 2 ≤ list.count (λ s, s.1 = c) drawer) →
    (5 ≤ drawer.length) →
    (∃ (s₁ s₂ : Σ c : fin 4, fin 2), s₁ ≠ s₂ ∧ s₁.1 = s₂.1) :=
  by
    assume drawer hc hl,
    sorry -- Proof goes here but it's not required as per the instructions.

end min_socks_to_guarantee_pair_l499_499998


namespace sum_of_gcd_values_l499_499117

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499117


namespace math_problem_l499_499736

theorem math_problem : 
  ∃ (n m k : ℕ), 
    (∀ d : ℕ, d ∣ n → d > 0) ∧ 
    (n = m * 6^k) ∧
    (∀ d : ℕ, d ∣ m → 6 ∣ d → False) ∧
    (m + k = 60466182) ∧ 
    (n.factors.count 1 = 2023) :=
sorry

end math_problem_l499_499736


namespace jennifer_initial_cards_l499_499404

theorem jennifer_initial_cards (eaten remaining : ℕ) (h_eaten : eaten = 61) (h_remaining : remaining = 11) : 
  (eaten + remaining = 72) :=
by 
  rw [h_eaten, h_remaining]
  exact rfl

end jennifer_initial_cards_l499_499404


namespace lower_right_square_value_l499_499533

def grid_filled_correctly : Prop :=
  ∃ (grid : ℕ → ℕ → ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ 5 → ∀ j, 1 ≤ j ∧ j ≤ 5 → 1 ≤ grid i j ∧ grid i j ≤ 5) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 5 → (∃! j, 1 ≤ j ∧ j ≤ 5 ∧ grid i j = i)) ∧
    (∀ j, 1 ≤ j ∧ j ≤ 5 → (∃! i, 1 ≤ i ∧ i ≤ 5 ∧ grid i j = j)) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → grid n n ≠ grid n (5 - n + 1)) ∧
    grid 1 1 = 1 ∧ grid 1 4 = 4 ∧
    grid 2 2 = 2 ∧ grid 2 5 = 1 ∧
    grid 3 3 = 3 ∧
    grid 4 4 = 5

theorem lower_right_square_value : 
  grid_filled_correctly → 
  ∀ (grid : ℕ → ℕ → ℕ), grid 5 5 = 1 :=
begin
  sorry
end

end lower_right_square_value_l499_499533


namespace find_other_factor_l499_499029

theorem find_other_factor 
    (w : ℕ) 
    (hw_pos : w > 0) 
    (h_factor : ∃ (x y : ℕ), 936 * w = x * y ∧ (2 ^ 5 ∣ x) ∧ (3 ^ 3 ∣ x)) 
    (h_ww : w = 156) : 
    ∃ (other_factor : ℕ), 936 * w = 156 * other_factor ∧ other_factor = 72 := 
by 
    sorry

end find_other_factor_l499_499029


namespace gcd_sum_l499_499134

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499134


namespace expected_quarterly_earnings_is_0_80_l499_499996
noncomputable theory

def expected_earnings
  (actual_earnings : ℝ)
  (dividend_paid_per_person : ℝ)
  (shares_owned : ℕ)
  (additional_dividend_rate : ℝ)
  (exceed_rate : ℝ)
  (total_dividend : ℝ) : ℝ :=
  let E := (total_dividend - shares_owned.to_real * additional_dividend_rate *
                 (actual_earnings - (total_dividend / (shares_owned.to_real * 0.25)))) /
              (shares_owned.to_real * 0.5) in
  E

theorem expected_quarterly_earnings_is_0_80 :
  expected_earnings 1.10 208 400 0.04 0.10 208 = 0.80 :=
by
  sorry

end expected_quarterly_earnings_is_0_80_l499_499996


namespace sum_of_gcd_values_l499_499212

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499212


namespace sum_of_arithmetic_sequence_l499_499442

open BigOperators

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

def arithmetic_sequence (a : ℕ → α) :=
  ∃ (d : α), ∀ (n : ℕ), a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) (h_sum : a 3 + a 4 + a 5 = 12) :
  (∑ i in finset.range 7, a (i + 1)) = 28 :=
sorry

end sum_of_arithmetic_sequence_l499_499442


namespace sum_of_gcd_values_l499_499211

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499211


namespace pastries_sold_correctly_l499_499568

def cupcakes : ℕ := 4
def cookies : ℕ := 29
def total_pastries : ℕ := cupcakes + cookies
def left_over : ℕ := 24
def sold_pastries : ℕ := total_pastries - left_over

theorem pastries_sold_correctly : sold_pastries = 9 :=
by sorry

end pastries_sold_correctly_l499_499568


namespace person_A_leave_time_l499_499067

theorem person_A_leave_time
  (ha : ℚ := 1 / 6) -- Work rate of Person A per hour
  (hb : ℚ := 1 / 8) -- Work rate of Person B per hour
  (hc : ℚ := 1 / 10) -- Work rate of Person C per hour
  (start_time : ℚ := 8) -- Start time in hours (8 AM)
  (end_time : ℚ := 12) -- End time in hours (12 PM)
  (total_work : ℚ := 1) -- Total work to be done
  : ℚ := sorry -- Expected leave time of Person A in hours

end person_A_leave_time_l499_499067


namespace johns_cost_per_use_is_550_l499_499406

-- Define the conditions
def heatingPadCost : ℝ := 30
def operatingCostPerUse : ℝ := 0.50
def usesPerWeek : ℝ := 3
def numberOfWeeks : ℝ := 2

-- Calculate the total number of uses
def totalUses : ℝ := usesPerWeek * numberOfWeeks

-- Calculate the total operating cost
def totalOperatingCost : ℝ := operatingCostPerUse * totalUses

-- Calculate the total cost
def totalCost : ℝ := heatingPadCost + totalOperatingCost

-- Calculate the cost per use
def costPerUse : ℝ := totalCost / totalUses

-- The proof statement
theorem johns_cost_per_use_is_550 : costPerUse = 5.50 := by
  -- proof steps
  sorry

end johns_cost_per_use_is_550_l499_499406


namespace gcd_sum_l499_499092

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499092


namespace total_farm_tax_collected_l499_499294

noncomputable def totalFarmTax (taxPaid: ℝ) (percentage: ℝ) : ℝ := taxPaid / (percentage / 100)

theorem total_farm_tax_collected (taxPaid : ℝ) (percentage : ℝ) (h_taxPaid : taxPaid = 480) (h_percentage : percentage = 16.666666666666668) :
  totalFarmTax taxPaid percentage = 2880 :=
by
  rw [h_taxPaid, h_percentage]
  simp [totalFarmTax]
  norm_num
  sorry

end total_farm_tax_collected_l499_499294


namespace solve_fn_fixed_points_l499_499855

def f (x : ℝ) : ℝ := 1 + 2 / x

def f_n : ℕ → ℝ → ℝ 
| 0 => id
| (n + 1) => f ∘ f_n n

theorem solve_fn_fixed_points (n : ℕ) (hn : n > 0) (x : ℝ) : x = f_n n x ↔ x = 2 ∨ x = -1 := sorry

end solve_fn_fixed_points_l499_499855


namespace no_pqr_exists_l499_499869

def a (n : ℕ) : ℕ 
| 0     := 2
| 1     := 5
| (n+2) := (2 - (n^2)) * a (n+1) + (2 + (n^2)) * a n

theorem no_pqr_exists : ¬ ∃ p q r : ℕ, p > 0 ∧ q > 0 ∧ r > 0 ∧ a p * a q = a r :=
by {
  sorry
}

end no_pqr_exists_l499_499869


namespace infinite_geometric_series_sum_l499_499720

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499720


namespace sum_of_gcd_values_l499_499235

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499235


namespace transformed_graph_result_l499_499913

-- Define the initial function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Define the function after shifting left by π / 3
def g (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 3) - Real.pi / 3)

-- Define the final function after changing the x-coordinates
def h (x : ℝ) : ℝ := Real.sin (4 * x + Real.pi / 3)

-- The theorem we need to prove: that h(x) is the result of the described transformations applied to f(x)
theorem transformed_graph_result : ∀ (x : ℝ), h x = Real.sin (4 * x + Real.pi / 3) := sorry

end transformed_graph_result_l499_499913


namespace find_length_d_l499_499834

theorem find_length_d :
  ∀ (A B C P: Type) (AB AC BC : ℝ) (d : ℝ),
    AB = 425 ∧ BC = 450 ∧ AC = 510 ∧
    (∃ (JG FI HE : ℝ), JG = FI ∧ FI = HE ∧ JG = d ∧ 
      (d / BC + d / AC + d / AB = 2)) 
    → d = 306 :=
by {
  sorry
}

end find_length_d_l499_499834


namespace common_roots_product_l499_499936

theorem common_roots_product (a b c C : ℤ) (p q r t : ℂ) :
  (a = 5) ∧ (b = 3) ∧ (c = 2) →
  (Polynomial.root_set (Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.C 15) ℂ = {p, q, r}) →
  (Polynomial.root_set (Polynomial.monomial 3 1 + Polynomial.monomial 1 C + Polynomial.C 30) ℂ = {p, q, t}) →
  let pq_product := p * q in
  pq_product = -5 * (2 ^ (1 / 3) : ℂ) →
  a + b + c = 10 :=
sorry

end common_roots_product_l499_499936


namespace triangle_angle_side_determination_l499_499377

theorem triangle_angle_side_determination
  (A C B : Point)
  (angle_A : ℝ)
  (BC : ℝ)
  (AB : ℝ)
  (h₁ : angle_A = π / 3)
  (h₂ : BC = 3)
  (h₃ : AB = √6) :
  ∃ (angle_C : ℝ) (AC : ℝ), angle_C = π / 4 ∧ AC = (√6 + 3 * √2) / 2 := by
  sorry

end triangle_angle_side_determination_l499_499377


namespace log_identity_l499_499804

theorem log_identity (x : ℝ) (h : log 16 (x - 5) = 1 / 4) : 
  1 / log x 2 = log 2 7 :=
sorry

end log_identity_l499_499804


namespace sum_of_gcd_values_l499_499233

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499233


namespace quad_cyclic_quad_tangential_l499_499464

variables {Point : Type} [AffineSpace ℝ Point]

-- Variables representing quadrilateral and point M
variables (A B C D M : Point)
variable [ConvexQuadrilateral ABCD : AffineSubspace ℝ]

-- Assumptions
variable (hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
variable (h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
variable (h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
variable (h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D))

-- To Prove: The quadrilateral is cyclic
theorem quad_cyclic (A B C D M: Point) 
[ConvexQuadrilateral ABCD] 
(hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
(h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
(h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
(h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D)) 
: CyclicQuadrilateral ABCD := 
sorry

-- To Prove: The quadrilateral is tangential
theorem quad_tangential (A B C D M: Point) 
[ConvexQuadrilateral ABCD] 
(hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
(h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
(h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
(h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D)) 
: TangentialQuadrilateral ABCD := 
sorry

end quad_cyclic_quad_tangential_l499_499464


namespace ms_mosel_fills_243_boxes_each_week_l499_499269

theorem ms_mosel_fills_243_boxes_each_week: 
  (let 
    total_hens := 270,
    laying_hens := 0.9 * total_hens,
    eggs_per_day := laying_hens,
    days_in_week := 7,
    eggs_per_week := eggs_per_day * days_in_week,
    eggs_per_box := 7
  in eggs_per_week / eggs_per_box) = 243 :=
sorry

end ms_mosel_fills_243_boxes_each_week_l499_499269


namespace highest_score_not_necessarily_20_l499_499037

-- Define the conditions of the round-robin tournament
def total_teams : ℕ := 16
def total_games : ℕ := (total_teams * (total_teams - 1)) / 2 -- \( \binom{16}{2} \)

-- Define the scoring rule
def points_win : ℕ := 3
def points_draw : ℕ := 1

-- Prove that the highest score must be at least 20 is false
theorem highest_score_not_necessarily_20 
  (points : fin total_teams → ℕ)
  (h_conditions : ∀ i, points i = points_win * n_wins i + points_draw * n_draws i) :
  ¬ (∃ i, points i ≥ 20) :=
sorry

end highest_score_not_necessarily_20_l499_499037


namespace trig_expression_tan_4alpha_l499_499304

theorem trig_expression_tan_4alpha (α : Real) : 
  (sin (2 * α + 2 * π) + 2 * sin (4 * α - π) + sin (6 * α + 4 * π)) 
  / (cos (6 * π - 2 * α) + 2 * cos (4 * α - π) + cos (6 * α - 4 * π)) = 
  tan (4 * α) := 
sorry

end trig_expression_tan_4alpha_l499_499304


namespace pinocchio_cannot_pay_exactly_l499_499892

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499892


namespace density_of_weight_material_l499_499049

noncomputable def density_of_material (ρ_kerosene : ℝ) : ℝ :=
  ρ_kerosene / 2

theorem density_of_weight_material : density_of_material 800 = 400 := 
by
  rw [density_of_material]
  norm_num
  sorry

end density_of_weight_material_l499_499049


namespace number_of_ways_to_fill_board_l499_499444

theorem number_of_ways_to_fill_board :
  let a : ℕ → ℕ :=
    λ n, if n = 1 then 1 else
         if n = 2 then 2 else
         a (n - 1) + a (n - 2)
  in a 10 = 89 :=
by
  let a : ℕ → ℕ := 
    λ n, if n = 1 then 1 else
         if n = 2 then 2 else
         a (n - 1) + a (n - 2)
  sorry

end number_of_ways_to_fill_board_l499_499444


namespace distance_point_to_line_example_l499_499496

noncomputable def distance_from_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / real.sqrt (a^2 + b^2)

theorem distance_point_to_line_example :
  distance_from_point_to_line (-1) 2 2 1 (-10) = 2 * real.sqrt 5 :=
by
  sorry

end distance_point_to_line_example_l499_499496


namespace june_earnings_l499_499411

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l499_499411


namespace time_for_Q_l499_499880

-- Definitions of conditions
def time_for_P := 252
def meet_time := 2772

-- Main statement to prove
theorem time_for_Q : (∃ T : ℕ, lcm time_for_P T = meet_time) ∧ (lcm time_for_P meet_time = meet_time) :=
    by 
    sorry

end time_for_Q_l499_499880


namespace triangle_given_abc_cosB_l499_499398

theorem triangle_given_abc_cosB 
  (a : ℝ) (b : ℝ) (c : ℝ) (B : ℝ) 
  (h1 : a = 3) 
  (h2 : b - c = 2) 
  (h3 : cos B = -1/2) : 
  b = 7 ∧ c = 5 ∧ sin (B + (π - B - acos (a / (b * sin B)))) = 3 * sqrt 3 / 14 :=
by
  -- Proof goes here
  sorry

end triangle_given_abc_cosB_l499_499398


namespace smallest_number_diminished_by_35_l499_499972

def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

def conditions : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def lcm_conditions := lcm_list conditions

theorem smallest_number_diminished_by_35 :
  ∃ n, n - 35 = lcm_conditions :=
sorry

end smallest_number_diminished_by_35_l499_499972


namespace compare_logs_l499_499825

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l499_499825


namespace convert_89_to_binary_l499_499286

def divide_by_2_remainders (n : Nat) : List Nat :=
  if n = 0 then [] else (n % 2) :: divide_by_2_remainders (n / 2)

def binary_rep (n : Nat) : List Nat :=
  (divide_by_2_remainders n).reverse

theorem convert_89_to_binary :
  binary_rep 89 = [1, 0, 1, 1, 0, 0, 1] := sorry

end convert_89_to_binary_l499_499286


namespace geometric_series_sum_l499_499685

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499685


namespace gcd_sum_5n_6_n_eq_12_l499_499188

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499188


namespace fuel_consumption_per_bag_l499_499640

noncomputable def fuel_consumption_per_mile_due_to_bags (base_fuel_person : ℕ) (num_passengers : ℕ) 
(num_crew : ℕ) (fuel_consum_total : ℕ) (distance : ℕ) (empty_plane_fuel : ℕ) (additional_fuel_person : ℕ) 
(bags_per_person : ℕ) : ℕ :=
  let num_people := num_passengers + num_crew
  let total_people_fuel := num_people * additional_fuel_person
  let total_base_fuel := empty_plane_fuel + total_people_fuel
  let total_bags := (num_passengers + num_crew) * bags_per_person
  let actual_fuel_per_mile := fuel_consum_total / distance
  let additional_bags_fuel := actual_fuel_per_mile - total_base_fuel
  additional_bags_fuel / total_bags

theorem fuel_consumption_per_bag (base_fuel_person : ℕ) (num_passengers : ℕ) (num_crew : ℕ) 
(fuel_consum_total : ℕ) (distance : ℕ) (empty_plane_fuel : ℕ) 
(additional_fuel_person : ℕ) (bags_per_person : ℕ) : 
  fuel_consumption_per_mile_due_to_bags base_fuel_person num_passengers num_crew 
  fuel_consum_total distance empty_plane_fuel additional_fuel_person bags_per_person = 2 :=
by
  sorry

-- Example usage:
#eval fuel_consumption_per_mile_due_to_bags 20 30 5 106000 400 20 3 2

end fuel_consumption_per_bag_l499_499640


namespace solution_inequality_l499_499348

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom increasing_function (x y : ℝ) : x < y → f x < f y

theorem solution_inequality (x : ℝ) : f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1 / 3 := sorry

end solution_inequality_l499_499348


namespace embankment_completion_time_l499_499846

noncomputable def completion_time (w1 w2 : ℕ) (d1 : ℕ) (portion : ℝ) : ℝ :=
  (portion * d1 * w1) / w2

theorem embankment_completion_time :
  completion_time 60 80 5 0.5 = 7.5 :=
by
  unfold completion_time
  norm_num

end embankment_completion_time_l499_499846


namespace min_distance_PM_PN_l499_499326

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem min_distance_PM_PN :
  let M : ℝ × ℝ := (-1, 3)
  let N : ℝ × ℝ := (2, 1)
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ distance M P + distance N P = 5 :=
by
  let M := (-1 : ℝ, 3 : ℝ)
  let N := (2 : ℝ, 1 : ℝ)
  use (5 / 4, 0)
  split
  sorry
  sorry

end min_distance_PM_PN_l499_499326


namespace sum_of_gcd_values_l499_499116

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499116


namespace x_intercept_of_line_l499_499611

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -4)) (h2 : (x2, y2) = (6, 8)) : 
  ∃ x0 : ℝ, (x0 = (10 / 3) ∧ ∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ ∀ y : ℝ, y = m * x0 + b) := 
sorry

end x_intercept_of_line_l499_499611


namespace number_of_sections_proof_l499_499608

-- Define the given conditions
def students_section : List ℕ := [55, 35, 45, 42]
def mean_marks_section : List ℕ := [50, 60, 55, 45]
def overall_average : ℚ := 52.06

-- Define the proof statement
theorem number_of_sections_proof :
  length students_section = 4 :=
by
  -- Sorry is used to skip the proof for now
  sorry

end number_of_sections_proof_l499_499608


namespace swimming_speed_l499_499557

theorem swimming_speed (v_m v_s : ℝ) 
  (h1 : v_m + v_s = 6)
  (h2 : v_m - v_s = 8) : 
  v_m = 7 :=
by
  sorry

end swimming_speed_l499_499557


namespace pyramid_base_height_is_two_l499_499325

-- Given conditions
variables {AB AD AP : ℝ × ℝ × ℝ}

-- Set the given vector values
def vector_AB : ℝ × ℝ × ℝ := (4, -2, 3)
def vector_AD : ℝ × ℝ × ℝ := (-4, 1, 0)
def vector_AP : ℝ × ℝ × ℝ := (-6, 2, -8)

-- Define distance function (height)
def distance_from_point_to_plane (P A B D : ℝ × ℝ × ℝ) : ℝ :=
  let normal_vector := (3, 12, 4) in
  let num := abs (((P.1 * normal_vector.1) + (P.2 * normal_vector.2) + (P.3 * normal_vector.3))) in
  let denom := real.sqrt ((normal_vector.1 ^ 2) + (normal_vector.2 ^ 2) + (normal_vector.3 ^ 2)) in
  num / denom

-- Theorem statement
theorem pyramid_base_height_is_two : distance_from_point_to_plane (vector_AP) (0,0,0) (vector_AB) (vector_AD) = 2 := by sorry

end pyramid_base_height_is_two_l499_499325


namespace highest_powers_sum_l499_499743

def legendre (n p : ℕ) : ℕ := 
  if p = 0 then 0 
  else nat.div (nat.factorial n / nat.factorial (n / p)) (nat.factorial (n % p))

theorem highest_powers_sum (n : ℕ) (p1 : ℕ) (p2 : ℕ) (k1 k2 : ℕ) :
  (legendre n 2) = 18 →
  p1 = 4 →
  p2 = 16 →
  k1 = 2 →
  k2 = 4 →
  (legendre n 2 / k1) + (legendre n 2 / k2) = 13 :=
by sorry

end highest_powers_sum_l499_499743


namespace sum_gcd_values_l499_499265

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499265


namespace infinitely_many_integers_not_sum_of_three_cubes_l499_499476

theorem infinitely_many_integers_not_sum_of_three_cubes :
  ∃ (N : ℤ) (k : ℤ), ∀ (a1 a2 a3 : ℤ),
  (N = 9 * k + 4 ∨ N = 9 * k + 5) ∧
  (N % 9 ≠ (a1 ^ 3 % 9 + a2 ^ 3 % 9 + a3 ^ 3 % 9)) :=
begin
  sorry
end

end infinitely_many_integers_not_sum_of_three_cubes_l499_499476


namespace min_omega_l499_499783

noncomputable def function_has_three_zeros (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x1 x2 x3, a ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ b ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧
  ∀ x, (a ≤ x ∧ x ≤ b ∧ f x = 0) → x = x1 ∨ x = x2 ∨ x = x3

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  sqrt 2 * sin (2 * ω * x - (Real.pi / 12)) + 1

theorem min_omega (ω : ℝ) :
  (0 < ω) 
  → function_has_three_zeros (f ω) 0 Real.pi
  → ω = 5 / 3 :=
sorry

end min_omega_l499_499783


namespace inequality_abc_l499_499756

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by
  sorry

end inequality_abc_l499_499756


namespace infinite_geometric_series_sum_l499_499724

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499724


namespace sum_of_three_consecutive_even_numbers_is_162_l499_499518

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l499_499518


namespace finite_swaps_possible_l499_499968

variable (n : ℕ)
variable (circle_points : Fin n → ℝ)

def consecutive_four_points (i : Fin n) : (ℝ × ℝ × ℝ × ℝ) :=
  let a := circle_points i
  let b := circle_points ((i + 1) % n)
  let c := circle_points ((i + 2) % n)
  let d := circle_points ((i + 3) % n)
  (a, b, c, d)

theorem finite_swaps_possible (h : ∀ {i : Fin n}, (let (a, b, c, d) := consecutive_four_points n circle_points i in (a - d) * (b - c) < 0)) :
  ∃ m : ℕ, ∀ (k : ℕ), k > m → ¬ (can_swap k) := sorry

def can_swap (j : Fin n) : Prop := sorry

end finite_swaps_possible_l499_499968


namespace factorial_base_823_l499_499504
 
theorem factorial_base_823 :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ),
  823 = a_1 * 1! + a_2 * 2! + a_3 * 3! + a_4 * 4! + a_5 * 5! ∧ 
  0 ≤ a_1 ∧ a_1 ≤ 1 ∧
  0 ≤ a_2 ∧ a_2 ≤ 2 ∧
  0 ≤ a_3 ∧ a_3 ≤ 3 ∧
  0 ≤ a_4 ∧ a_4 ≤ 4 ∧
  0 ≤ a_5 ∧ a_5 ≤ 5 ∧
  a_4 = 4 := sorry

end factorial_base_823_l499_499504


namespace gcd_5n_plus_6_n_sum_l499_499106

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499106


namespace gcd_5n_plus_6_n_sum_l499_499111

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499111


namespace ellipse_center_and_axis_addition_l499_499857

theorem ellipse_center_and_axis_addition :
  ∃ h k a b : ℝ,
  let F1 := (0, 2) in
  let F2 := (6, 2) in
  (∀ P : ℝ × ℝ, (euclidean_distance P F1 + euclidean_distance P F2 = 10) →
    (P ∈ ellipse_eq (h, k) (a, b))) ∧
  (h + k + a + b = 14) :=
begin
  sorry
end

def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def ellipse_eq (center : ℝ × ℝ) (a b : ℝ) :=
  { P : ℝ × ℝ | (P.1 - center.1) ^ 2 / a ^ 2 + (P.2 - center.2) ^ 2 / b ^ 2 = 1 }

end ellipse_center_and_axis_addition_l499_499857


namespace sum_gcd_values_l499_499261

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499261


namespace equality_of_integers_l499_499651

theorem equality_of_integers (a b : ℕ) (h1 : ∀ n : ℕ, ∃ m : ℕ, m > 0 ∧ (a^m + b^m) % (a^n + b^n) = 0) : a = b :=
sorry

end equality_of_integers_l499_499651


namespace sum_gcd_values_l499_499260

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499260


namespace gcd_5n_plus_6_n_sum_l499_499099

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499099


namespace gcd_sum_l499_499130

theorem gcd_sum (n : ℕ) (h : n > 0) : 
  ((gcd (5 * n + 6) n).toNat ∈ {1, 2, 3, 6}) → (1 + 2 + 3 + 6 = 12) :=
by
  sorry

end gcd_sum_l499_499130


namespace waiter_customers_l499_499057

theorem waiter_customers
    (initial_tables : ℝ)
    (left_tables : ℝ)
    (customers_per_table : ℝ)
    (remaining_tables : ℝ) 
    (total_customers : ℝ) 
    (h1 : initial_tables = 44.0)
    (h2 : left_tables = 12.0)
    (h3 : customers_per_table = 8.0)
    (remaining_tables_def : remaining_tables = initial_tables - left_tables)
    (total_customers_def : total_customers = remaining_tables * customers_per_table) :
    total_customers = 256.0 :=
by
  sorry

end waiter_customers_l499_499057


namespace pinocchio_cannot_pay_exactly_l499_499891

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l499_499891


namespace larger_cube_weight_l499_499585

variable {ρ : ℝ} -- density in pounds per cubic unit
variable {s : ℝ} -- side length of the original cube

theorem larger_cube_weight (h : ρ * s^3 = 5) : ρ * (3 * s)^3 = 135 := by
  have hs : 3 * s = 3 * s := by rfl
  -- Calculate the volume of the larger cube
  have hvol : (3 * s)^3 = 27 * s^3 := by
    calc
      (3 * s)^3 = 27 * s^3 : by ring
  -- Use density and volume to calculate the weight of the larger cube
  calc
    ρ * (3 * s)^3 = ρ * 27 * s^3 : by rw hvol
            ...   = 27 * (ρ * s^3) : by ring
            ...   = 27 * 5 : by rw h
            ...   = 135 : by norm_num

end larger_cube_weight_l499_499585


namespace class_schedule_arrangements_l499_499626

-- Definitions from conditions
def morning_periods := {1, 2, 3, 4}
def afternoon_periods := {5, 6}
def classes := { "Chinese", "Mathematics", "Physics", "English", "Biology", "Chemistry" }

noncomputable def num_arrangements : ℕ := 
  let chinese_options := 4
  let biology_options := 2
  let remaining_permutations := 4.factorial
  chinese_options * biology_options * remaining_permutations

-- The theorem stating the conclusion
theorem class_schedule_arrangements : num_arrangements = 192 := by
  sorry

end class_schedule_arrangements_l499_499626


namespace find_k_common_term_l499_499320

def sequence_a (k : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 
  else if n = 2 then k 
  else if n = 3 then 3*k - 3 
  else if n = 4 then 6*k - 8 
  else (n * (n-1) * (k-2)) / 2 + n

def is_fermat (x : ℕ) : Prop :=
  ∃ m : ℕ, x = 2^(2^m) + 1

theorem find_k_common_term (k : ℕ) :
  k > 2 → ∃ n m : ℕ, sequence_a k n = 2^(2^m) + 1 :=
by
  sorry

end find_k_common_term_l499_499320


namespace solve_for_x_l499_499288

theorem solve_for_x (x : ℝ) (hx : 0 < x) (h : (real.sqrt (12 * x)) * (real.sqrt (20 * x)) * (real.sqrt (6 * x)) * (real.sqrt (30 * x)) = 60) :
    x = real.sqrt (real.sqrt 30) / 60 := 
sorry

end solve_for_x_l499_499288


namespace sequence_positive_integers_no_divisor_2015_l499_499790

noncomputable def sequence (n : ℕ) : ℤ :=
if n = 0 then 1 else 3 * sequence (n - 1) + 2 * Int.sqrt (2 * (sequence (n - 1))^2 - 1)

theorem sequence_positive_integers (n : ℕ) : ∀ n, sequence n > 0 :=
sorry

theorem no_divisor_2015 (m : ℕ) : ¬ ∃ m, 2015 ∣ sequence m :=
sorry

end sequence_positive_integers_no_divisor_2015_l499_499790


namespace course_selection_ways_l499_499609

def humanities_courses : Finset ℕ := {1, 2, 3, 4} -- representing A1, A2, A3, A4
def natural_science_courses : Finset ℕ := {5, 6, 7} -- representing B1, B2, B3
def conflict : (ℕ × ℕ) := (1, 5) -- representing (A1, B1)

theorem course_selection_ways :
  (hum_courses: Finset ℕ) (sci_courses: Finset ℕ)
  (conflict : (ℕ × ℕ))
  (hhum : hum_courses = humanities_courses)
  (hsci : sci_courses = natural_science_courses)
  (hconf : conflict = (1, 5)) :
  ∃ n, n = 25 :=
  sorry

end course_selection_ways_l499_499609


namespace symmetry_origin_l499_499939

def f (x : ℝ) : ℝ := x^3 + x

theorem symmetry_origin : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end symmetry_origin_l499_499939


namespace fundamental_theorem_of_algebra_l499_499925

noncomputable def is_root (f : ℂ → ℂ) (x : ℂ) : Prop :=
  f(x) = 0

noncomputable def polynomial_f : (ℂ → ℂ) :=
  λ x, (x - 1) * (x^2 + x + 1)

theorem fundamental_theorem_of_algebra :
  (∃ ω : ℂ, is_root polynomial_f ω) ∧
  (∀ ω : ℂ, is_root (λ x, x^2 + x + 1) ω →
    (ω^2 + ω + 1 = 0) ∧
    (ω = -1/2 + complex.sqrt 3 / 2 * complex.I ∨ ω = -1/2 - complex.sqrt 3 / 2 * complex.I) ∧
    (ω * complex.conj(ω) = 1) ∧
    (ω^2 = complex.conj(ω))) :=
sorry

end fundamental_theorem_of_algebra_l499_499925


namespace nat_as_sum_of_distinct_fib_l499_499471

-- Define the Fibonacci sequence.
def fib : ℕ → ℕ 
| 1 := 1
| 2 := 2
| (n + 2) := fib (n + 1) + fib n

-- State the theorem for natural numbers being representable as sums of distinct Fibonacci numbers.
theorem nat_as_sum_of_distinct_fib (n : ℕ) : 
  ∃ (S : finset ℕ), (∀ x ∈ S, ∃ k, x = fib k) ∧ S.sum id = n := 
sorry

end nat_as_sum_of_distinct_fib_l499_499471


namespace complex_inequality_contradiction_l499_499306

open Complex

theorem complex_inequality_contradiction {z1 z2 z3 z4 : ℂ}
  (h1 : |z1| = 1)
  (h2 : |z2| = 1)
  (h3 : |z3| = 1)
  (h4 : |z4| = 1)
  (nz1 : z1 ≠ 1)
  (nz2 : z2 ≠ 1)
  (nz3 : z3 ≠ 1)
  (nz4 : z4 ≠ 1) :
  3 - z1 - z2 - z3 - z4 + z1 * z2 * z3 * z4 ≠ 0 := by
  sorry

end complex_inequality_contradiction_l499_499306


namespace quad_cyclic_quad_tangential_l499_499463

variables {Point : Type} [AffineSpace ℝ Point]

-- Variables representing quadrilateral and point M
variables (A B C D M : Point)
variable [ConvexQuadrilateral ABCD : AffineSubspace ℝ]

-- Assumptions
variable (hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
variable (h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
variable (h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
variable (h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D))

-- To Prove: The quadrilateral is cyclic
theorem quad_cyclic (A B C D M: Point) 
[ConvexQuadrilateral ABCD] 
(hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
(h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
(h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
(h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D)) 
: CyclicQuadrilateral ABCD := 
sorry

-- To Prove: The quadrilateral is tangential
theorem quad_tangential (A B C D M: Point) 
[ConvexQuadrilateral ABCD] 
(hM_in_ABCD : M ∈ ConvexHull ℝ {A, B, C, D})
(h_eq_dist_AB_CD : ∀ l, IsPerpendicular l AB → IsPerpendicular l CD → distance M l = distance M l)
(h_eq_dist_BC_AD : ∀ l, IsPerpendicular l BC → IsPerpendicular l AD → distance M l = distance M l)
(h_area_eq : area ABCD = (distance M A * distance M C) + (distance M B * distance M D)) 
: TangentialQuadrilateral ABCD := 
sorry

end quad_cyclic_quad_tangential_l499_499463


namespace compare_logs_l499_499824

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l499_499824


namespace gcd_sum_l499_499165

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499165


namespace geometric_series_sum_l499_499687

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499687


namespace piggy_bank_savings_l499_499479

theorem piggy_bank_savings :
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  final_amount = 104 :=
by
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  show final_amount = 104 from sorry

end piggy_bank_savings_l499_499479


namespace infinite_geometric_series_sum_l499_499727

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499727


namespace average_sequence_150_terms_l499_499650

theorem average_sequence_150_terms :
  let a : ℕ → ℤ := λ n, (-1) ^ n * (n + 1) in
  (∑ i in Finset.range 150, a i : ℤ) / 150 = 1 / 2 :=
by
  sorry

end average_sequence_150_terms_l499_499650


namespace tangent_circles_l499_499767

noncomputable def problem : Real :=
  let C1 := (0, 0)
  let r1 : Real := 2
  let C2 := (3, -4)
  let m := 3
  let r2 := m
  let distance := Math.sqrt ((C2.1 - C1.1)^2 + (C2.2 - C1.2)^2)
  distance = r1 + r2

theorem tangent_circles :
  let C1 := (0, 0)
  let r1 : Real := 2
  let C2 := (3, -4)
  let r2 := 3
  ∃ m, m > 0 ∧ r2 = m ∧ ∃ (d : Real), d = Math.sqrt ((C2.1 - C1.1)^2 + (C2.2 - C1.2)^2) ∧ 
  d = r1 + r2 := 
by
  sorry

end tangent_circles_l499_499767


namespace history_paper_pages_l499_499918

/-
Stacy has a history paper due in 3 days.
She has to write 21 pages per day to finish on time.
Prove that the total number of pages for the history paper is 63.
-/

theorem history_paper_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 21) (h2 : days = 3) : total_pages = 63 :=
by
  -- We would include the proof here, but for now, we use sorry to skip the proof.
  sorry

end history_paper_pages_l499_499918


namespace log_comparison_l499_499810

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l499_499810


namespace largest_possible_m_for_x10_minus_1_factorization_l499_499946

theorem largest_possible_m_for_x10_minus_1_factorization :
  ∃ (q : Fin 5 → Polynomial ℝ), (∀ i, degree q i > 0) ∧ (x^10 - 1 = (List.prod (List.ofFn q))) ∧ 5 ≥ (λ p, ∃ (q : Fin p → Polynomial ℝ), (∀ i, degree q i > 0) ∧ (x^10 - 1 = List.prod (List.ofFn q))) :=
by
  sorry

end largest_possible_m_for_x10_minus_1_factorization_l499_499946


namespace tetrahedron_max_projection_area_l499_499964

theorem tetrahedron_max_projection_area :
  ∃ (max_proj_area : ℝ), 
    (∀ (T : tetrahedron) (a : ℝ) (θ : ℝ),
      (T.adj_faces_are_equilateral_triangles ∧ T.side_length = 1 ∧ T.dihedral_angle = π / 4) →
      (max_proj_area = a ∧ a = T.projected_area_onto_common_edge_plane)) 
    →
    max_proj_area = (√3 / 4) :=
sorry

end tetrahedron_max_projection_area_l499_499964


namespace carpet_interior_length_l499_499995

/--
A carpet is designed using three different colors, forming three nested rectangles with different areas in an arithmetic progression. 
The innermost rectangle has a width of two feet. Each of the two colored borders is 2 feet wide on all sides.
Determine the length in feet of the innermost rectangle. 
-/
theorem carpet_interior_length 
  (x : ℕ) -- length of the innermost rectangle
  (hp : ∀ (a b c : ℕ), a = 2 * x ∧ b = (4 * x + 24) ∧ c = (4 * x + 56) → (b - a) = (c - b)) 
  : x = 4 :=
by
  sorry

end carpet_interior_length_l499_499995


namespace statement_a_statement_b_statement_c_statement_d_l499_499009

/-- Given conditions for statement A, prove statement A is correct --/
theorem statement_a (α : ℝ) (h1 : sin α = -1/3) (h2 : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) : 
    tan α = -sqrt 2 / 4 :=
sorry

/-- Given conditions for statement B, prove statement B is correct --/
theorem statement_b (α : ℝ) (h : π / 2 < α ∧ α < π) : 
    (π / 4 < α / 2 ∧ α / 2 < π / 2) ∨ (5 * π / 4 < α / 2 ∧ α / 2 < 3 * π / 2) :=
sorry

/-- Given conditions for statement C, prove statement C is incorrect --/
theorem statement_c (C : ℝ) (θ : ℝ) (hC : C = 30) (hθ : θ = 3) : 
    ¬(1/2 * θ * (C / (θ + 2))^2 = 48) :=
sorry

/-- Given conditions for statement D, prove statement D is correct --/
theorem statement_d (α : ℝ) (h : 3 * π / 2 < α ∧ α < 2 * π) : 
    ∃ P : ℝ × ℝ, P = (cos α, tan α) ∧ cos α > 0 ∧ tan α < 0 :=
sorry

end statement_a_statement_b_statement_c_statement_d_l499_499009


namespace probability_all_even_l499_499847

theorem probability_all_even :
  let die1_even_count := 3
  let die1_total := 6
  let die2_even_count := 3
  let die2_total := 7
  let die3_even_count := 4
  let die3_total := 9
  let prob_die1_even := die1_even_count / die1_total
  let prob_die2_even := die2_even_count / die2_total
  let prob_die3_even := die3_even_count / die3_total
  let probability_all_even := prob_die1_even * prob_die2_even * prob_die3_even
  probability_all_even = 1 / 10.5 :=
by
  sorry

end probability_all_even_l499_499847


namespace infinite_geometric_series_sum_l499_499716

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499716


namespace monotonically_increasing_interval_l499_499345

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos ((2 / 3) * x - (5 * Real.pi / 12))

theorem monotonically_increasing_interval 
  (φ : ℝ) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0) 
  (h3 : 2 * (Real.pi / 8) + φ = Real.pi / 4) : 
  ∀ x : ℝ, (-(Real.pi / 2) ≤ x) ∧ (x ≤ Real.pi / 2) ↔ ∃ k : ℤ, x ∈ [(-7 * Real.pi / 8 + 3 * k * Real.pi), (5 * Real.pi / 8 + 3 * k * Real.pi)] :=
sorry

end monotonically_increasing_interval_l499_499345


namespace part1_inequality_l499_499532

theorem part1_inequality (a b x y : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
    (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_a_ge_x : a ≥ x) : 
    (a - x) ^ 2 + (b - y) ^ 2 ≤ (a + b - x) ^ 2 + y ^ 2 := 
by 
  sorry

end part1_inequality_l499_499532


namespace bailey_total_spending_l499_499630

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end bailey_total_spending_l499_499630


namespace gcd_sum_divisors_eq_12_l499_499169

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499169


namespace gcd_sum_divisors_eq_12_l499_499178

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499178


namespace gcd_sum_l499_499160

theorem gcd_sum : ∀ n : ℕ, n > 0 → 
  let gcd1 := Nat.gcd (5 * n + 6) n in
  ∀ d ∈ {1, 2, 3, 6}, gcd1 = d ∨ gcd1 = 1 ∧ gcd1 = 2 ∧ gcd1 = 3 ∧ gcd1 = 6 → 
  d ∈ {1, 2, 3, 6} → d.sum = 12 :=
begin
  sorry
end

end gcd_sum_l499_499160


namespace min_coins_for_less_than_1_dollar_l499_499544

theorem min_coins_for_less_than_1_dollar :
  ∃ (p n q h : ℕ), 1*p + 5*n + 25*q + 50*h ≥ 1 ∧ 1*p + 5*n + 25*q + 50*h < 100 ∧ p + n + q + h = 8 :=
by 
  sorry

end min_coins_for_less_than_1_dollar_l499_499544


namespace geometric_series_sum_l499_499700

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end geometric_series_sum_l499_499700


namespace compare_logs_l499_499819

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
def c : ℝ := 1 / 2

theorem compare_logs (a b c : ℝ) (ha : a = Real.log 2 / Real.log 5) (hb : b = Real.log 3 / Real.log 8) (hc : c = 1 / 2) :
  a < c ∧ c < b :=
by {
  rw [ha, hb, hc],
  have ha_lt : Real.log 2 / Real.log 5 < 1 / 2,
  { sorry },
  have hb_gt : 1 / 2 < Real.log 3 / Real.log 8,
  { sorry },
  exact ⟨ha_lt, hb_gt⟩
}

end compare_logs_l499_499819


namespace largest_N_not_payable_l499_499901

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499901


namespace lines_through_point_with_equal_intercepts_l499_499039

/-- 
  Theorem: There are exactly 2 lines that pass through the point (5, 2) 
  and have equal intercepts on the x-axis and y-axis.
-/
theorem lines_through_point_with_equal_intercepts : 
  ∃ (l1 l2 : ℝ → ℝ), 
  (l1 = λ x, (2 / 5) * x) ∧ 
  (l2 = λ x, 7 - x) ∧ 
  (∀ (x y : ℝ), (5, 2) = (x,y) → l1 x = y ∧ l2 x = y) ∧ 
  (λ ∃ slope1 slope2, (slope1 = (2 / 5)) ∧ (slope2 = -1)) ∧
  -- Ensures that our lines pass through the point (5, 2) 
  (l1 5 = 2) ∧ (l2 5 = 2) ∧    -- Ensures they intersect only on x-y axis and are not colinear
  (l1 = l2). 
sorry

end lines_through_point_with_equal_intercepts_l499_499039


namespace solve_equation_l499_499295

theorem solve_equation (x : ℝ) : 
  (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ↔ 
    (x = 1 ∨ x = -16 ∨ x = 4 ∨ x = -4) :=
by
  sorry

end solve_equation_l499_499295


namespace sum_of_given_infinite_geometric_series_l499_499709

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
a / (1 - r)

theorem sum_of_given_infinite_geometric_series :
  infinite_geometric_series_sum (1/4) (1/2) (by norm_num) = 1/2 :=
sorry

end sum_of_given_infinite_geometric_series_l499_499709


namespace triangle_sides_l499_499564

/-- Side lengths of triangle ABC given mentioned conditions. --/
theorem triangle_sides {A B C D M N : Point}
  (S1 S2 : Circle)
  (hD_on_AC : D ∈ Segment A C)
  (hS1_inscribed : inscribed S1 (Triangle A B D))
  (hS1_touches_BD_at_M : touches_at S1 (Segment B D) M)
  (hS2_inscribed : inscribed S2 (Triangle B C D))
  (hS2_touches_DC_at_N : touches_at S2 (Segment D C) N)
  (h_ratio_radii : radius S1 / radius S2 = 7 / 4)
  (hBM : distance B M = 3)
  (hMN : distance M N = 1)
  (hND : distance N D = 1) :
  distance A B = 10 ∧ distance B C = 6 ∧ distance A C = 12 := 
sorry

end triangle_sides_l499_499564


namespace polynomial_transformation_l499_499827

variable {x y : ℝ}

theorem polynomial_transformation
  (h : y = x + 1/x) 
  (poly_eq_0 : x^4 + x^3 - 5*x^2 + x + 1 = 0) :
  x^2 * (y^2 + y - 7) = 0 :=
sorry

end polynomial_transformation_l499_499827


namespace gcd_sum_divisors_eq_12_l499_499174

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499174


namespace Zoe_has_the_least_amount_of_money_l499_499384

/-- Bo, Coe, Flo, Jo, Moe, and Zoe have different amounts of money. -/
variables (money : Type)
variables (Bo Coe Flo Jo Moe Zoe : money)

/-
Conditions:
1. Neither Jo nor Bo has as much money as Flo.
2. Both Bo and Coe have more money than Zoe.
3. Jo has more money than Zoe but less than Bo.
4. Flo has more money than Moe but less than Coe.
-/
variables (h1 : Jo < Flo) (h2 : Bo < Flo)
variables (h3 : Zoe < Bo) (h4 : Zoe < Coe)
variables (h5 : Zoe < Jo) (h6 : Jo < Bo)
variables (h7 : Moe < Flo) (h8 : Flo < Coe)

theorem Zoe_has_the_least_amount_of_money :
  ∀ x, x = Zoe :=
begin
  sorry -- Proof not provided, as per the instructions.
end

end Zoe_has_the_least_amount_of_money_l499_499384


namespace infinite_geometric_series_sum_l499_499726

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499726


namespace sum_of_gcd_values_l499_499112

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499112


namespace largest_cos_x_l499_499432

-- Define the initial conditions and the target property in the theorem
theorem largest_cos_x (x y z : ℝ) (hx : sin x = cot y) (hy : sin y = cot z) (hz : sin z = cot x) : 
  cos x = (Real.sqrt 5 - 1) / 2 :=
sorry

end largest_cos_x_l499_499432


namespace a_2017_eq_10_l499_499047

-- Definition for the digit sum function S
def S (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Definition of the sequence a_n
noncomputable def a : ℕ → ℕ
| 1 := 2017
| 2 := 22
| n := S (a (n - 1)) + S (a (n - 2))

-- The theorem we want to prove
theorem a_2017_eq_10 : a 2017 = 10 := by
  sorry

end a_2017_eq_10_l499_499047


namespace infinite_geometric_series_sum_l499_499692

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499692


namespace num_divisors_of_n_squared_less_than_n_not_dividing_n_l499_499859

def n : ℕ := 2^29 * 5^17

theorem num_divisors_of_n_squared_less_than_n_not_dividing_n : 
  ∀ (n : ℕ), n = 2^29 * 5^17 → 
  let total_divisors_n_squared := (58 + 1) * (34 + 1),
      total_divisors_n := (29 + 1) * (17 + 1),
      divisors_n_squared_less_than_n := (total_divisors_n_squared - 1) / 2
  in
  divisors_n_squared_less_than_n - total_divisors_n = 492 :=
by
  intros n hn,
  sorry

end num_divisors_of_n_squared_less_than_n_not_dividing_n_l499_499859


namespace gcd_sum_5n_plus_6_l499_499209

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499209


namespace probability_of_multiple_of_4_from_5_digits_l499_499577

noncomputable def probability_multiple_of_4 : ℚ := sorry

theorem probability_of_multiple_of_4_from_5_digits :
  let digits := {1, 2, 3, 4, 5}
  ∃ (three_digit_numbers : set (nat × nat × nat)),
    (∀ (a b c : nat), (a, b, c) ∈ three_digit_numbers → a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
    (probability_multiple_of_4 = 7 / 60) := sorry

end probability_of_multiple_of_4_from_5_digits_l499_499577


namespace sequence_pattern_l499_499488

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern_l499_499488


namespace domain_of_f_x_l499_499935

theorem domain_of_f_x (f : ℝ → ℝ) :
  (∀ x, x ∈ [-1, 2] → (x + 1) ∈ domain f) →
  (∀ y, y ∈ [0, 3] → y ∈ domain f) :=
by
  sorry

end domain_of_f_x_l499_499935


namespace find_a_l499_499343

def f (a x : ℝ) : ℝ :=
if x > 2 then f a (x + 5) 
else if x >= -2 then a * Real.exp(x) 
else f a (-x)

theorem find_a : (∃ a : ℝ, f a (-2016) = Real.exp(1) → a = 1) := sorry

end find_a_l499_499343


namespace red_tickets_for_one_yellow_l499_499534

-- Define the conditions given in the problem
def yellow_needed := 10
def red_for_yellow (R : ℕ) := R -- This function defines the number of red tickets for one yellow
def blue_for_red := 10

def toms_yellow := 8
def toms_red := 3
def toms_blue := 7
def blue_needed := 163

-- Define the target function that converts the given conditions into a statement.
def red_tickets_for_yellow_proof : Prop :=
  ∀ R : ℕ, (2 * R = 14) → (R = 7)

-- Statement for proof where the condition leads to conclusion
theorem red_tickets_for_one_yellow : red_tickets_for_yellow_proof :=
by
  intros R h
  rw [← h, mul_comm] at h
  sorry

end red_tickets_for_one_yellow_l499_499534


namespace fractions_sum_ge_one_l499_499757

variable {a b c : ℝ}

theorem fractions_sum_ge_one (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 :=
by {
  sorry,
}

end fractions_sum_ge_one_l499_499757


namespace triangle_least_perimeter_l499_499415

noncomputable def least_perimeter_of_triangle : ℕ :=
  let a := 7
  let b := 17
  let c := 13
  a + b + c

theorem triangle_least_perimeter :
  let a := 7
  let b := 17
  let c := 13
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  4 ∣ (a^2 + b^2 + c^2) - 2 * c^2 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  least_perimeter_of_triangle = 37 :=
by
  intros _ _ _ h
  sorry

end triangle_least_perimeter_l499_499415


namespace sum_gcd_possible_values_eq_twelve_l499_499244

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499244


namespace find_C_coordinates_l499_499468

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨7, 2⟩
def B : Point := ⟨-1, 9⟩
def D : Point := ⟨2, 7⟩

-- defining the condition that triangle ABC is isosceles with AB = AC
def isIsosceles (A B C : Point) : Prop :=
  (Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)) = (Real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2))

-- defining the condition that D is the foot of the altitude from A
def isAltitudeFoot (A D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates :
  ∃ C : Point, isIsosceles A B C ∧ isAltitudeFoot A D B C ∧ C.x = 5 ∧ C.y = 5 :=
sorry

end find_C_coordinates_l499_499468


namespace row_col_value_2002_2003_l499_499450

theorem row_col_value_2002_2003 :
  let base_num := (2003 - 1)^2 + 1 
  let result := base_num + 2001 
  result = 2002 * 2003 :=
by
  sorry

end row_col_value_2002_2003_l499_499450


namespace gcd_sum_5n_plus_6_l499_499196

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499196


namespace sum_of_smallest_angles_l499_499634

theorem sum_of_smallest_angles :
  let Q : ℂ[X] := (X^24 - 1)^2 / (X - 1)^2 - X^23 + X^15
  ∑ k in {1, 2}, (Q.coeff k).arg / π = 48 / 575 :=
begin
  sorry
end

end sum_of_smallest_angles_l499_499634


namespace gcd_sum_l499_499087

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499087


namespace ant_at_B_after_5_minutes_l499_499066

noncomputable def lattice_position_probability (start : ℤ × ℤ) (end : ℤ × ℤ) (minutes : ℕ) : ℚ := sorry

theorem ant_at_B_after_5_minutes :
  lattice_position_probability (0, 0) (0, 1) 5 = 1 / 4 := by
  sorry

end ant_at_B_after_5_minutes_l499_499066


namespace polynomial_square_roots_correct_l499_499351

noncomputable def polynomial_square_roots (n : ℕ) (a : Fin n → ℝ) : Polynomial ℝ :=
  let y := Polynomial.C
  Polynomial.ofFinsupp (y ^ n + (2 * a 2 - a 1 ^ 2) * y ^ (n - 1) +
    (a 2 ^ 2 - 2 * a 1 * a 3 + 2 * a 4) * y ^ (n - 2) + sorry)

theorem polynomial_square_roots_correct (P : ℝ[X]) (x_roots : Fin n → ℝ) (a : Fin n → ℝ) :
  P = Polynomial.ofRoots x_roots →
  polynomial_square_roots n a =
    Polynomial.ofRoots (λ i, x_roots i ^ 2) :=
sorry

end polynomial_square_roots_correct_l499_499351


namespace gcd_values_sum_l499_499082

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499082


namespace range_of_y_l499_499655

theorem range_of_y :
  (∃ (b : Fin 30 → ℝ), (∀ i, b i = 1 ∨ b i = 3) ∧ (∃ y, y = ∑ i in finset.range 30, b (i:ℕ) / 4^(i + 1) 
  ∧ (1/3 ≤ y ∧ y < 1))) := sorry

end range_of_y_l499_499655


namespace swimmers_meeting_times_l499_499543

noncomputable def pool_length : ℝ := 100
noncomputable def swimmer_A_time : ℝ := 72
noncomputable def swimmer_B_time : ℝ := 60
noncomputable def total_time : ℝ := 720  -- 12 minutes in seconds

theorem swimmers_meeting_times :
  let x := (total_time * (pool_length / swimmer_A_time + pool_length / swimmer_B_time)) / (2 * pool_length)
  in x = 11 :=
by 
  -- the proof would go here
  sorry

end swimmers_meeting_times_l499_499543


namespace center_of_circle_l499_499493

theorem center_of_circle :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 1 → (x = -2 ∧ y = 1) :=
by
  intros x y hyp
  -- Here, we would perform the steps of comparing to the standard form and proving the center.
  sorry

end center_of_circle_l499_499493


namespace sum_of_gcd_values_l499_499221

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499221


namespace am_gm_inequality_two_vars_l499_499991
-- Import the entire necessary library

-- Define the theorem
theorem am_gm_inequality_two_vars (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  (x + y) / 2 ≥ real.sqrt (x * y) :=
by 
  sorry

end am_gm_inequality_two_vars_l499_499991


namespace largest_N_cannot_pay_exactly_without_change_l499_499884

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end largest_N_cannot_pay_exactly_without_change_l499_499884


namespace shelly_thread_needed_l499_499911

def keychain_thread (classes: ℕ) (clubs: ℕ) (thread_per_keychain: ℕ) : ℕ := 
  let total_friends := classes + clubs
  total_friends * thread_per_keychain

theorem shelly_thread_needed : keychain_thread 6 (6 / 2) 12 = 108 := 
  by
    show 6 + (6 / 2) * 12 = 108
    sorry

end shelly_thread_needed_l499_499911


namespace integral_sin6_cos2_l499_499633

open Real

theorem integral_sin6_cos2 :
  ∫ x in 0..2 * π, (sin x ^ 6) * (cos x ^ 2) = (5 * π / 64) := 
by
  sorry

end integral_sin6_cos2_l499_499633


namespace sum_of_gcd_values_l499_499222

theorem sum_of_gcd_values : 
  (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ {1, 2, 3, 6} ∧ ∑ d in {1, 2, 3, 6}, d = 12) :=
by
  sorry

end sum_of_gcd_values_l499_499222


namespace hyperbola_equation_l499_499760

-- Define the hyperbola with vertices and other conditions
def Hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions and the proof goal
theorem hyperbola_equation
  (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (k_PA k_PB : ℝ)
  (PA_PB_condition : k_PA * k_PB = 3)
  (MN_min_value : |(M.1 - N.1) + (M.2 - N.2)| = 4) :
  Hyperbola a b h →
  (a = 2 ∧ b = 2 * Real.sqrt 3 ∧ (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1)) ∨ 
   a = 2 / 3 ∧ b = 2 * Real.sqrt 3 / 3 ∧ (∀ (x y : ℝ), (9 * x^2 / 4 - 3 * y^2 / 4 = 1)))
:=
sorry

end hyperbola_equation_l499_499760


namespace infinite_geometric_series_sum_l499_499717

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499717


namespace gcd_sum_5n_6_n_eq_12_l499_499190

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499190


namespace P_100_has_100_distinct_real_roots_l499_499566

def P_seq (P : ℕ → ℝ → ℝ) : Prop :=
  P 0 = λ x, 1 ∧
  P 1 = λ x, x ∧
  (∀ n, P (n + 1) = λ x, x * P n x - P (n - 1) x)

def P_100_roots (P : ℕ → ℝ → ℝ) : Prop :=
  ∀ k : ℕ, k < 101 → 
  P 100 (2 * Real.cos (↑ (2 * k + 1) * Real.pi / 202)) = 0

def distinct_real_roots_of_P_100 (P : ℕ → ℝ → ℝ) : Prop := 
  ∀ k m : ℕ, k < 101 → m < 101 → k ≠ m → 
  2 * Real.cos ((↑ (2 * k + 1) * Real.pi) / 202) ≠ 
  2 * Real.cos ((↑ (2 * m + 1) * Real.pi) / 202)

theorem P_100_has_100_distinct_real_roots (P : ℕ → ℝ → ℝ) :
  P_seq P →
  P_100_roots P ∧ distinct_real_roots_of_P_100 P :=
sorry

end P_100_has_100_distinct_real_roots_l499_499566


namespace no_solution_in_nat_l499_499906

theorem no_solution_in_nat (x y : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
  (1 / (x:ℝ)^2 + 1 / (x:ℝ * y) + 1 / (y:ℝ)^2 = 1) → false :=
by
  intro h_eqn -- placeholder for the equality condition
  sorry -- proof to be provided

end no_solution_in_nat_l499_499906


namespace randy_piggy_bank_final_amount_l499_499481

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end randy_piggy_bank_final_amount_l499_499481


namespace evaluate_expression_l499_499309

noncomputable def M (x y : ℝ) : ℝ := if x < y then y else x
noncomputable def m (x y : ℝ) : ℝ := if x < y then x else y

theorem evaluate_expression
  (p q r s t : ℝ)
  (h1 : p < q)
  (h2 : q < r)
  (h3 : r < s)
  (h4 : s < t)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ p ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t):
  M (M p (m q r)) (m s (m p t)) = q := 
sorry

end evaluate_expression_l499_499309


namespace combined_cost_price_correct_l499_499547

noncomputable def combined_cost_price : ℝ := 
  let stock1_price := 100 * (1 - 0.04) * (1 + 0.002) * (1 + 0.12) + 2
  let stock2_price := 200 * (1 - 0.06) * (1 + 0.0025) * (1 + 0.10) + 3
  let stock3_price := 150 * (1 - 0.03) * (1 + 0.005) * (1 + 0.15) + 1
  in stock1_price + stock2_price + stock3_price

theorem combined_cost_price_correct :
  combined_cost_price = 489.213665 := by
  sorry

end combined_cost_price_correct_l499_499547


namespace total_population_is_correct_l499_499879

-- Define the known values
def proportion_adults : ℝ := 0.40
def proportion_male_adults : ℝ := 0.35
def male_adults : ℕ := 23040

-- The ultimate goal is to find P
theorem total_population_is_correct : 
  ∃ (P : ℝ), P ≈ 164571 ∧ proportion_male_adults * (proportion_adults * P) = male_adults :=
  sorry

end total_population_is_correct_l499_499879


namespace geometric_series_sum_l499_499684

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l499_499684


namespace find_Robert_pens_l499_499660

variable (x y : ℕ)   -- x is the number of pens Julia's other friend bought.
                     -- y is the number of pens Robert bought.
variable (cost_per_pen total_cost : ℚ)
variable (pens_Dorothy pens_Julia pens_Robert : ℕ)

-- Conditions
def pens_Dorothy := 1.5 * x
def pens_Julia := 3 * x
def pens_Robert := y
def cost_per_pen := 1.5
def total_cost := 33
def total_pens := total_cost / cost_per_pen

-- Equation that must hold: 
theorem find_Robert_pens (h1 : pens_Dorothy = 1.5 * x)
                         (h2 : pens_Julia = 3 * x)
                         (h3 : pens_Robert = y)
                         (h4 : cost_per_pen = 1.5)
                         (h5 : total_cost = 33)
                         (h6 : total_pens = 22) :
  4.5 * x + y = 22 → y = 13 :=
by
  sorry

end find_Robert_pens_l499_499660


namespace at_least_two_foxes_met_same_number_of_koloboks_l499_499749

-- Define the conditions
def number_of_foxes : ℕ := 14
def number_of_koloboks : ℕ := 92

-- The theorem statement to be proven
theorem at_least_two_foxes_met_same_number_of_koloboks :
  ∃ (f : Fin number_of_foxes.succ → ℕ), 
    (∀ i, f i ≤ number_of_koloboks) ∧ 
    ∃ i j, i ≠ j ∧ f i = f j :=
by
  sorry

end at_least_two_foxes_met_same_number_of_koloboks_l499_499749


namespace company_pays_per_month_l499_499014

theorem company_pays_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1.08 * 10^6)
  (h5 : cost_per_box = 0.6) :
  (total_volume / (length * width * height) * cost_per_box) = 360 :=
by
  -- sorry to skip proof
  sorry

end company_pays_per_month_l499_499014


namespace average_employees_per_week_l499_499584

variable (x : ℕ)

theorem average_employees_per_week (h1 : x + 200 > x)
                                   (h2 : x < 200)
                                   (h3 : 2 * 200 = 400) :
  (x + 200 + x + 200 + 200 + 400) / 4 = 250 := by
  sorry

end average_employees_per_week_l499_499584


namespace function_properties_l499_499062

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = - f x

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≥ f y

theorem function_properties :
  let f := λ x : ℝ, x⁻¹ in
  is_odd_function f ∧ is_monotonically_decreasing_on f (set.Ioi 0) :=
by
  sorry

end function_properties_l499_499062


namespace find_AC_l499_499835

-- Define the given conditions
variables (A B C D E : Type)
variables [linear_ordered_field x y] [D_on_AC : D ∈ line_segment A C]
variables [E_on_BC : E ∈ line_segment B C]
variables [AB_perp_BC : ∠ (line_through A B) (line_through B C) = 90]
variables [AE_perp_BC : ∠ (line_through A E) (line_through E C) = 90]
variables [BD_eq_x : dist B D = x]
variables [DC_eq_x : dist D C = x]
variables [CE_eq_x : dist C E = x]

-- Define the conclusion
theorem find_AC : dist A C = 2 * x :=
sorry

end find_AC_l499_499835


namespace geometric_series_sum_l499_499678

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l499_499678


namespace log_proof_l499_499362

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem log_proof (x : ℝ) (h : log_base 7 (x + 6) = 2) : log_base 13 x = log_base 13 43 :=
by
  sorry

end log_proof_l499_499362


namespace gcd_sum_divisors_eq_12_l499_499171

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499171


namespace sheila_weekly_earnings_l499_499485

-- Variables
variables {hours_mon_wed_fri hours_tue_thu rate_per_hour : ℕ}

-- Conditions
def sheila_works_mwf : hours_mon_wed_fri = 8 := by sorry
def sheila_works_tue_thu : hours_tue_thu = 6 := by sorry
def sheila_rate : rate_per_hour = 11 := by sorry

-- Main statement to prove
theorem sheila_weekly_earnings : 
  3 * hours_mon_wed_fri + 2 * hours_tue_thu = 36 →
  rate_per_hour = 11 →
  (3 * hours_mon_wed_fri + 2 * hours_tue_thu) * rate_per_hour = 396 :=
by
  intros h_hours h_rate
  sorry

end sheila_weekly_earnings_l499_499485


namespace possible_k_values_l499_499952

theorem possible_k_values
  (n : ℕ)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h1 : ∀ i : ℕ, 1 ≤ i → i ≤ n → a (i + 1) = a i * r)
  (h2 : ∀ j : ℕ, 1 ≤ j → j ≤ n → b (j + 1) = b j + d)
  (h3 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ∀ i : ℕ, 1 ≤ i → i ≤ n → (x^2 + a i * x + b i).discriminant < 0) :
  (k = 1 ∨ k = n) :=
sorry

end possible_k_values_l499_499952


namespace better_fitting_model_l499_499552

variables (R2_A R2_B : ℝ)
noncomputable def model_A_better_fit : Prop :=
  R2_A > R2_B

theorem better_fitting_model (h1 : R2_A = 0.96) (h2 : R2_B = 0.85) : model_A_better_fit R2_A R2_B :=
by {
  rw [h1, h2],
  norm_num,
  exact 0.96 > 0.85,
}

end better_fitting_model_l499_499552


namespace michael_choose_classes_l499_499448

-- Michael's scenario setup
def total_classes : ℕ := 10
def compulsory_class : ℕ := 1
def remaining_classes : ℕ := total_classes - compulsory_class
def total_to_choose : ℕ := 4
def additional_to_choose : ℕ := total_to_choose - compulsory_class

-- Correct answer based on the conditions
def correct_answer : ℕ := 84

-- Function to compute the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove the number of ways Michael can choose his classes
theorem michael_choose_classes : binomial 9 3 = correct_answer := by
  rw [binomial, Nat.factorial]
  sorry

end michael_choose_classes_l499_499448


namespace circles_intersect_l499_499643

theorem circles_intersect
  (O1 O2 A B R S T: ℝ) -- Representing points on the Euclidean plane
  (h1: intersect(A, B, O1, O2)) -- Circles O1 and O2 intersect at A and B
  (h2: circumcircle(O1, B, O2, R, S, T)) -- Circumcircle of triangle O1BO2 intersects at R, S, T
  (h3: intersects_line(AB, T)) -- Circumcircle intersects line AB at T
  (h4: intersects_circle(O1, R)) -- Circumcircle intersects circle O1 at R
  (h5: intersects_circle(O2, S)) -- Circumcircle intersects circle O2 at S
  : TR = TS := sorry

end circles_intersect_l499_499643


namespace area_triangle_AOB_l499_499940

theorem area_triangle_AOB : 
  let m := -1 in
  let f (x : ℝ) := -2 * x + m in
  let A := (λ (y : ℝ), (y + m) / -2) 0,
  let B := (0, f 0) in
  1/2 * real.abs A.1 * real.abs B.2 = 1/4 :=
by
  let m := -1
  let f : ℝ → ℝ := λ x, -2 * x + m
  let A : prod ℝ ℝ := (λ (y : ℝ), (y + m) / -2) 0
  let B : prod ℝ ℝ := (0, f 0)
  calc
    1/2 * real.abs A.1 * real.abs B.2
        = 1/2 * real.abs (-1/2) * real.abs (-1) : by { sorry }
    ... = 1/2 * (1/2) * 1                          : by { sorry }
    ... = 1/4                                      : by { sorry }

end area_triangle_AOB_l499_499940


namespace greatest_value_in_T_l499_499858

open Set

def T : Set ℕ := {x | x ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}

noncomputable def max_element (S : Set ℕ) : ℕ := Sup (S ∩ T)

theorem greatest_value_in_T : ∃ T : Set ℕ, T ⊆ {2, 3, ..., 15} ∧
  (∀ c d ∈ T, c < d → ¬ (d ∣ c)) ∧
  max_element T = 15 := sorry

end greatest_value_in_T_l499_499858


namespace q_sum_0_5_l499_499860

noncomputable def q : ℝ → ℝ := sorry

axiom q_monic : polynomial.monic (q : polynomial ℝ)
axiom q_degree : polynomial.degree (q : polynomial ℝ) = 4
axiom q_at_1 : q 1 = 10
axiom q_at_2 : q 2 = 20
axiom q_at_3 : q 3 = 30

theorem q_sum_0_5 : q 0 + q 5 = 50 := sorry

end q_sum_0_5_l499_499860


namespace find_sum_a_b_l499_499332

-- Digit definition
def is_digit (n : ℕ) : Prop := n < 10

-- Given conditions
variables (a b : ℕ)

-- Hypotheses based on the multiplication setup
axiom a_digit : is_digit a
axiom b_digit : is_digit b
axiom mult_setup : 32 * a + 32 * 4 * b = 1486

-- Theorem statement
theorem find_sum_a_b (h1 : a_digit) (h2 : b_digit) (h3 : mult_setup) : a + b = 5 := 
sorry

end find_sum_a_b_l499_499332


namespace gumball_machine_total_gumballs_l499_499595

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l499_499595


namespace gcd_values_sum_l499_499083

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499083


namespace green_eyed_fish_leave_l499_499020

-- Conditions encapsulated in Lean code

-- Definition of the predicate for the number of green-eyed fish
def green_eyed_fish_leave_on_night (N : ℕ) : Prop :=
  ∀ (n : ℕ), 
  (∃ (green_eyed : ℕ), green_eyed = N) →
  (∀ f : fin green_eyed, fish (f.nat_val) = green_eyed → ∀ t : ℕ, t = N)

-- Theorem that proves the required condition
theorem green_eyed_fish_leave 
  (N : ℕ) 
  (fish : ℕ → ℕ)
  (announcement : ℕ) :
  announcement = N → 
  green_eyed_fish_leave_on_night N :=
by sorry

end green_eyed_fish_leave_l499_499020


namespace largest_possible_m_for_x10_minus_1_factorization_l499_499945

theorem largest_possible_m_for_x10_minus_1_factorization :
  ∃ (q : Fin 5 → Polynomial ℝ), (∀ i, degree q i > 0) ∧ (x^10 - 1 = (List.prod (List.ofFn q))) ∧ 5 ≥ (λ p, ∃ (q : Fin p → Polynomial ℝ), (∀ i, degree q i > 0) ∧ (x^10 - 1 = List.prod (List.ofFn q))) :=
by
  sorry

end largest_possible_m_for_x10_minus_1_factorization_l499_499945


namespace sum_gcd_values_l499_499253

theorem sum_gcd_values : (∑ d in {1, 2, 3, 6}, d) = 12 := by
  sorry

end sum_gcd_values_l499_499253


namespace largest_N_not_payable_l499_499904

theorem largest_N_not_payable (coins_5 coins_6 : ℕ) (h1 : coins_5 > 10) (h2 : coins_6 > 10) :
  ∃ N : ℕ, N ≤ 50 ∧ (¬ ∃ (x y : ℕ), N = 5 * x + 6 * y) ∧ N = 19 :=
by 
  use 19
  have h₃ : 19 ≤ 50 := by norm_num
  have h₄ : ¬ ∃ (x y : ℕ), 19 = 5 * x + 6 * y := sorry
  exact ⟨h₃, h₄⟩

end largest_N_not_payable_l499_499904


namespace correct_expr_using_multiplication_formula_l499_499623

-- Defining the expressions as hypotheses and proving the correct one
def expr_A := (y + x) * (y - x) = y^2 - x^2
def expr_B := (2 * x - y) * (2 * y - x) = y^2 - 4 * x^2
def expr_C := (2 * a - 1)^2 = 4 * a^2 - 2 * a + 1
def expr_D := (3 - x)^2 = 9 - x^2

theorem correct_expr_using_multiplication_formula : expr_A ∧ ¬ expr_B ∧ ¬ expr_C ∧ ¬ expr_D :=
by {
  sorry
}

end correct_expr_using_multiplication_formula_l499_499623


namespace infinite_geometric_series_sum_l499_499719

theorem infinite_geometric_series_sum :
  (let a := (1 : ℝ) / 4;
       r := (1 : ℝ) / 2 in
    a / (1 - r) = 1 / 2) :=
by
  sorry

end infinite_geometric_series_sum_l499_499719


namespace compare_logs_l499_499814

theorem compare_logs (a b c : ℝ) (h_a : a = Real.log 2 / Real.log 5) (h_b : b = Real.log 3 / Real.log 8) (h_c : c = 1 / 2) : a < c ∧ c < b :=
by {
  sorry,
}

end compare_logs_l499_499814


namespace log_problem_l499_499830

noncomputable def x (k : ℝ) := 2401

theorem log_problem (k : ℝ) (hk : k > 0 ∧ k ≠ 1) (hx : log k 2401 * log 7 k = 4) : x k = 2401 :=
by
  sorry

end log_problem_l499_499830


namespace intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l499_499992

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem intervals_monotonicity_f :
  ∀ k : ℤ,
    (∀ x : ℝ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 → f x = Real.cos (2 * x)) ∧
    (∀ x : ℝ, k * Real.pi + Real.pi / 2 ≤ x ∧ x ≤ k * Real.pi + Real.pi → f x = Real.cos (2 * x)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem intervals_monotonicity_g_and_extremum :
  ∀ x : ℝ,
    (-Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3 → (g x ≤ 1 ∧ g x ≥ -1)) :=
sorry

end intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l499_499992


namespace find_k_l499_499321

variable (a : ℕ → ℕ)

-- conditions
def condition1 := a 1 = 1
def condition2 (n : ℕ) : n ≥ 2 → a n = 1 + ∑ i in Finset.range (n - 1), (1 / (i + 1)) * a (i + 2) -- using Finset to sum over range
def condition3 := ∃ k, a k = 2017

-- theorem
theorem find_k : (∃ k, a k = 2017) → k = 4034 :=
by
  sorry

end find_k_l499_499321


namespace solve_for_a_plus_b_l499_499368

theorem solve_for_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, a * (x + b) = 3 * x + 12) → a + b = 7 :=
by
  intros h
  sorry

end solve_for_a_plus_b_l499_499368


namespace required_reduction_white_sugar_required_reduction_brown_sugar_required_reduction_powdered_sugar_l499_499561

-- Define the initial and new prices
def initial_price_white_sugar : ℝ := 6
def new_price_white_sugar : ℝ := 7.50

def initial_price_brown_sugar : ℝ := 8
def new_price_brown_sugar : ℝ := 9.75

def initial_price_powdered_sugar : ℝ := 10
def new_price_powdered_sugar : ℝ := 11.50

-- Calculate the percentage increase for each type of sugar
def percentage_increase (initial_price new_price : ℝ) : ℝ :=
  ((new_price - initial_price) / initial_price) * 100

def percentage_increase_white_sugar :=
  percentage_increase initial_price_white_sugar new_price_white_sugar

def percentage_increase_brown_sugar :=
  percentage_increase initial_price_brown_sugar new_price_brown_sugar

def percentage_increase_powdered_sugar :=
  percentage_increase initial_price_powdered_sugar new_price_powdered_sugar

-- Prove that the reduction in consumption required is equal to the percentage increase
theorem required_reduction_white_sugar :
  percentage_increase_white_sugar = 25 := 
sorry

theorem required_reduction_brown_sugar :
  percentage_increase_brown_sugar = 21.875 := 
sorry

theorem required_reduction_powdered_sugar :
  percentage_increase_powdered_sugar = 15 := 
sorry

end required_reduction_white_sugar_required_reduction_brown_sugar_required_reduction_powdered_sugar_l499_499561


namespace smallest_integer_n_exists_l499_499307

theorem smallest_integer_n_exists :
  ∃ (n : ℕ) (x : ℕ → ℝ), (n > 0) ∧ (∑ i in finset.range n, x i = 500) ∧ (∑ i in finset.range n, (x i)^4 = 160000) ∧ n = 50 :=
begin
  sorry,
end

end smallest_integer_n_exists_l499_499307


namespace sum_of_gcd_values_l499_499119

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499119


namespace f_log3_54_l499_499653

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then 3^x else sorry

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f (x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f (x + 2) = -1 / f (x)

-- Hypotheses based on conditions
variable (f : ℝ → ℝ)
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 4
axiom f_functional : functional_equation f

-- Main goal
theorem f_log3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 := by
  sorry

end f_log3_54_l499_499653


namespace sum_y_coordinates_of_rectangle_vertices_l499_499360

theorem sum_y_coordinates_of_rectangle_vertices (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 5) (h2 : y1 = 20) (h3 : x2 = -3) (h4 : y2 = -7) : 
  (∃ y3 y4 : ℝ, (y3 + y4) = 13) := 
by {
  use [10.5, 2.5], -- Example values that satisfy y3 + y4 = 13
  simp, 
  sorry }

end sum_y_coordinates_of_rectangle_vertices_l499_499360


namespace infinite_geometric_series_sum_l499_499693

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l499_499693


namespace unique_number_in_lists_B_or_C_l499_499010

theorem unique_number_in_lists_B_or_C (n : ℕ) (h : n > 1) :
  ∃! x, (x ∈ (λ k : ℕ, (10 ^ k).digits 2) ∨ x ∈ (λ k : ℕ, (10 ^ k).digits 5)) ∧ ((x.digits 2).length = n ∨ (x.digits 5).length = n) :=
sorry

end unique_number_in_lists_B_or_C_l499_499010


namespace connectivity_after_color_removal_l499_499526

-- Define the nodes in the complete graph
def nodes : ℕ := 50

-- Define edge colors
inductive color
  | C1 | C2 | C3

open color

-- We should have a complete graph definition here, but as a simplified version,
-- we consider only the conditions necessary for the proof problem
def complete_graph (n : ℕ) : Type := { edges : Type // (fin n × fin n) → color }

-- The main theorem statement.
theorem connectivity_after_color_removal (G : complete_graph nodes) :
  ∃ c : color, ∀ (u v : fin nodes), u ≠ v → 
  ∃ path : list (fin nodes), path.head = u ∧ path.ilast = v ∧
    ∀ i ∈ list.zip path (list.tail path), (G.edges i.1, G.edges i.2) = (G.edges i.1, G.edges i.2) :=
sorry

end connectivity_after_color_removal_l499_499526


namespace min_x_value_l499_499417

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 18 * x + 50 * y + 56

theorem min_x_value : 
  ∃ (x : ℝ), ∃ (y : ℝ), circle_eq x y ∧ x = 9 - Real.sqrt 762 :=
by
  sorry

end min_x_value_l499_499417


namespace range_of_a_l499_499436

def f (x a : ℝ) : ℝ := x^2 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ∧ (∃ x : ℝ, f (f x a) a = 0) → (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l499_499436


namespace gcd_sum_5n_6_n_eq_12_l499_499195

-- Define the problem conditions and what we need to prove.
theorem gcd_sum_5n_6_n_eq_12 : 
  (let possible_values := { d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n } in
  ∑ d in possible_values, d) = 12 :=
sorry

end gcd_sum_5n_6_n_eq_12_l499_499195


namespace sum_of_gcd_values_l499_499230

theorem sum_of_gcd_values : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Int.gcd (5 * n + 6) n}, d) = 12 :=
sorry

end sum_of_gcd_values_l499_499230


namespace gcd_sum_l499_499085

theorem gcd_sum (n : ℕ) (h : n > 0) : ∑ k in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5*n + 6) n}, d = 12 := by
  sorry

end gcd_sum_l499_499085


namespace log_comparison_l499_499809

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l499_499809


namespace solve_for_y_l499_499919

theorem solve_for_y (y : ℝ) (h : sqrt (3 + (5 * y - 4)^(1/3)) = sqrt 8) : y = 25.8 := 
by
  sorry

end solve_for_y_l499_499919


namespace infinite_geometric_series_sum_l499_499728

theorem infinite_geometric_series_sum :
  ∀ (a r : ℝ), a = 1 / 4 → r = 1 / 2 → abs r < 1 → (∑' n : ℕ, a * r ^ n) = 1 / 2 :=
begin
  intros a r ha hr hr_lt_1,
  rw [ha, hr],   -- substitute a and r with given values
  sorry
end

end infinite_geometric_series_sum_l499_499728


namespace gcd_values_sum_l499_499078

theorem gcd_values_sum : 
  (λ S, (∀ n : ℕ, n > 0 → ∃ d : ℕ, d ∈ S ∧ d = gcd (5 * n + 6) n) ∧ (∀ d1 d2 : ℕ, d1 ≠ d2 → ¬ (d1 ∈ S ∧ d2 ∈ S)) ∧ S.sum = 12) :=
begin
  sorry
end

end gcd_values_sum_l499_499078


namespace save_Ponchik_l499_499565

-- Definitions for conditions
def distance : ℕ := 18         -- Distance from Dunno to Ponchik
def speed : ℕ := 6             -- Speed of movement in km/h
def Ponchik_air_hours : ℕ := 4 -- Total air supply for Ponchik in hours
def tank_duration : ℕ := 2     -- Each air tank lasts for 2 hours
def max_tanks : ℕ := 2         -- Dunno can carry maximum 2 tanks

-- The main theorem stating Dunno can save Ponchik and both can return to the base safely
theorem save_Ponchik : 
  (exists 
    (travel_time : ℕ) 
    (D : { movement_time : ℕ // movement_time ≤ Ponchik_air_hours }) 
    (intermediate_tanks : list { t : ℕ // t ≤ max_tanks }),
    travel_time * speed = distance 
    ∧ sorry -- More detailed conditions regarding intermediate steps here and air management
  ) :=
sorry

end save_Ponchik_l499_499565


namespace gcd_5n_plus_6_n_sum_l499_499110

theorem gcd_5n_plus_6_n_sum : ∑ (d ∈ {d | ∃ n : ℕ, n > 0 ∧ gcd (5 * n + 6) n = d}) = 12 :=
by
  sorry

end gcd_5n_plus_6_n_sum_l499_499110


namespace gcd_sum_5n_plus_6_n_l499_499147

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499147


namespace count_distinct_quadruples_l499_499645

def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

theorem count_distinct_quadruples :
  (∃ u : finset (ℕ × ℕ × ℕ × ℕ),
    (∀ t ∈ u, let (a, b, c, d) := t in
      (binom (binom a b) (binom c d) = 21)) ∧
    (∀ t ∈ u, let (a, b, c, d) := t in
      a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d)
  ) → u.card = 13 :=
by
  sorry

end count_distinct_quadruples_l499_499645


namespace parallelogram_area_ratio_l499_499599

-- Definition of parallelogram conditions
variables {ABCD P Q : Type} 
variables (A B C D P Q : ℝ)
variables (h1 : A + B = 14) 
variables (h2 : C + D = 18)
variables (h3 : P = 7) 
variables (h4 : Q = 9)

-- The theorem stating the ratio of areas is 1:1 and the sum is 2
theorem parallelogram_area_ratio (hABCD : A + B = C + D ∧ P = 7 ∧ Q = 9) : 
  (APDQ_area P Q / PBCQ_area P Q = 1) ∧ (1 + 1 = 2) :=
by {
  sorry
}

end parallelogram_area_ratio_l499_499599


namespace log_comparison_l499_499807

variables (a b c : ℝ)
def log_base (b x : ℝ) := log x / log b

theorem log_comparison 
  (a_def : a = log_base 5 2)
  (b_def : b = log_base 8 3)
  (c_def : c = 1 / 2) :
  a < c ∧ c < b :=
by
  sorry

end log_comparison_l499_499807


namespace milk_percentage_difference_l499_499060

noncomputable def perc_difference (a b: ℕ) : ℚ :=
  (a - b : ℚ) / a * 100

theorem milk_percentage_difference (A B C : ℕ) (h1 : A = 1216) (h2 : B = 532) (h3 : C = 532) (h4 : A = B + C) (h5 : B = C) :
  perc_difference A B = 56.25 :=
by
  rw [perc_difference, h1, h2]
  norm_num
  sorry

end milk_percentage_difference_l499_499060


namespace gcd_sum_5n_plus_6_l499_499205

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499205


namespace problem_1_main_theorem_1_problem_2_l499_499762

-- Definitions based on conditions
def a (n : ℕ) : ℚ := if n = 1 then 1 / 2 else sorry
def S (n : ℕ) : ℚ := n^2 * a n - n * (n - 1)

-- Problem 1 statement
theorem problem_1 (n : ℕ) : (n + 1) / n * S n = if n = 1 then 1 else sorry := sorry

-- Main theorem from problem 1
theorem main_theorem_1 (n : ℕ) (h1 : ∀ n, S n = n^2 / (n + 1)) : (n + 1) / n * S n = n := by
  sorry

-- Problem 2 statement
def b (n : ℕ) : ℚ := S n / (n^3 + 3 * n^2)

theorem problem_2 (n : ℕ) (h2 : ∀ k, S k = k^2 / (k + 1)) : (∑ k in Finset.range (n + 1), b k) < 5 / 12 := by
  sorry

end problem_1_main_theorem_1_problem_2_l499_499762


namespace sum_gcd_possible_values_eq_twelve_l499_499251

theorem sum_gcd_possible_values_eq_twelve {n : ℕ} (hn : 0 < n) : 
  (∑ d in {d : ℕ | ∃ (m : ℕ), d = gcd (5 * m + 6) m}.to_finset, d) = 12 := 
by
  sorry

end sum_gcd_possible_values_eq_twelve_l499_499251


namespace isosceles_trapezoid_AC_length_l499_499941

noncomputable def length_of_AC (AB AD BC CD AC : ℝ) :=
  AB = 30 ∧ AD = 15 ∧ BC = 15 ∧ CD = 12 → AC = 23.32

theorem isosceles_trapezoid_AC_length :
  length_of_AC 30 15 15 12 23.32 := by
  sorry

end isosceles_trapezoid_AC_length_l499_499941


namespace find_n_l499_499943

theorem find_n (a b : ℝ) 
  (h1 : ∃ d, (log (a^7 * b^14) - log (a^4 * b^9)) = d 
             ∧ (log (a^10 * b^17) - log (a^7 * b^14)) = d)
  (h2 : log (a^(-2) * b^(319)) = log (a ^ (-2) * b^n)) : 
  n = 319 := 
by sorry

end find_n_l499_499943


namespace count_and_sum_arithmetic_sequence_l499_499570

theorem count_and_sum_arithmetic_sequence :
  let list := List.range 21 |>.map (λ n, -50 + 6 * n)
  list.length = 21 ∧ list.sum = 231 :=
by
  let list := List.range 21 |>.map (λ n, -50 + 6 * n)
  have h_length : list.length = 21 := sorry
  have h_sum : list.sum = 231 := sorry
  exact ⟨h_length, h_sum⟩

end count_and_sum_arithmetic_sequence_l499_499570


namespace chess_tournament_participants_and_days_l499_499381

theorem chess_tournament_participants_and_days:
  ∃ n d : ℕ, 
    (n % 2 = 1) ∧
    (n * (n - 1) / 2 = 630) ∧
    (d = 34 / 2) ∧
    (n = 35) ∧
    (d = 17) :=
sorry

end chess_tournament_participants_and_days_l499_499381


namespace asterisk_replacement_l499_499018

theorem asterisk_replacement (x : ℝ) : 
  (x / 20) * (x / 80) = 1 ↔ x = 40 :=
by sorry

end asterisk_replacement_l499_499018


namespace gcd_sum_5n_plus_6_n_l499_499145

theorem gcd_sum_5n_plus_6_n : 
  (∑ d in {d | ∃ n : ℕ+, gcd (5 * n + 6) n = d}, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_n_l499_499145


namespace floor_of_root_eq_zero_l499_499330

def floor (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := Real.exp x - 2 / x

def is_root (x : ℝ) : Prop := f x = 0

theorem floor_of_root_eq_zero (x₀ : ℝ) (hx₀ : is_root x₀) : floor x₀ = 0 := 
sorry

end floor_of_root_eq_zero_l499_499330


namespace triangle_area_is_three_l499_499371

open Real

-- Define the points A, B, and C
def A := (0 : ℝ, 0 : ℝ)
def B := (2 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 3 : ℝ)

-- Calculate the area of triangle ABC
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let base := dist A B in
  let height := abs (snd C - snd A) in
  (1 / 2) * base * height

-- Prove that the area of triangle ABC is 3
theorem triangle_area_is_three : area_triangle A B C = 3 := by
  sorry

end triangle_area_is_three_l499_499371


namespace gcd_sum_5n_plus_6_l499_499202

theorem gcd_sum_5n_plus_6 (n : ℕ) (hn : n > 0) : 
  let possible_gcds := {d : ℕ | d ∈ {1, 2, 3, 6}}
  (∑ d in possible_gcds, d) = 12 :=
by
  sorry

end gcd_sum_5n_plus_6_l499_499202


namespace smallest_n_exists_satisfying_condition_l499_499742

-- Definitions of rational polynomials and the sum of squares problem.
def rational_polynomial (f : ℚ[X]) : Prop :=
  true  -- Placeholder for the type definition of polynomials over rationals.

def sum_of_squares (fs : List (ℚ[X])) : ℚ[X] :=
  fs.foldr (λ f acc, f^2 + acc) 0

-- Main theorem stating the proof problem.
theorem smallest_n_exists_satisfying_condition :
  (∃ fs : List (ℚ[X]), fs.length = 5 ∧ sum_of_squares fs = X^2 + 7) ∧
  (¬ ∃ fs : List (ℚ[X]), fs.length = 4 ∧ sum_of_squares fs = X^2 + 7) :=
by
  sorry

end smallest_n_exists_satisfying_condition_l499_499742


namespace next_perfect_square_l499_499826

theorem next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) : ∃ y : ℕ, y = x + 4 * (nat.sqrt x) + 4 := by
  sorry

end next_perfect_square_l499_499826


namespace gcd_sum_divisors_eq_12_l499_499172

open Nat

theorem gcd_sum_divisors_eq_12 : 
  (∑ d in (finset.filter (λ d, d ∣ 6) (finset.range (7))), d) = 12 :=
by {
  -- We will prove that the sum of all possible gcd(5n+6, n) values, where d is a divisor of 6, is 12
  -- This statement will be proven assuming necessary properties of Euclidean algorithm, handling finite sets, and divisors.
  sorry
}

end gcd_sum_divisors_eq_12_l499_499172


namespace sum_of_gcd_values_l499_499114

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) : (∑ d in {d | ∃ n : ℕ, n > 0 ∧ d = Nat.gcd (5 * n + 6) n}, d) = 12 :=
by
  sorry

end sum_of_gcd_values_l499_499114


namespace measure_of_angle_E_l499_499396

variable (D E F : ℝ)
variable (h1 : E = F)
variable (h2 : F = 3 * D)
variable (h3 : D + E + F = 180)

theorem measure_of_angle_E : E = 540 / 7 :=
by
  -- Proof omitted
  sorry

end measure_of_angle_E_l499_499396


namespace common_area_of_rotated_rectangle_l499_499043

theorem common_area_of_rotated_rectangle :
  ∀ (β : ℝ),
  0 < β ∧ β < (π / 2) ∧ sin β = 3 / 5 →
  (common_area_of_rectangles (rotate_rectangle_around_vertex (create_rectangle 2 1) β) (create_rectangle 2 1)) = 39 / 50 :=
by
  sorry

end common_area_of_rotated_rectangle_l499_499043


namespace min_ones_in_20_row_matrix_l499_499993

theorem min_ones_in_20_row_matrix :
  ∃ (n m : ℕ), n = 20 ∧ 
  (∀ i j : ℕ, i < j → column_unique i j) ∧ 
  (∀ i j k l : ℕ, column_pair_constraint i j k l) ∧ 
  m = 3820 :=
begin
  let rows := 20,
  let max_columns := 1 + (nat.choose 20 1) + (nat.choose 20 2) + (nat.choose 20 3),
  have columns_unique : ∀ (i j : ℕ), i < j → true := sorry,
  have pairwise_constraint : ∀ (i j k l : ℕ), true := sorry,
  let num_1s := (nat.choose 20 1 * 1) + (nat.choose 20 2 * 2) + (nat.choose 20 3 * 3),
  use [rows, num_1s],
  split,
  { refl, },
  split,
  { apply columns_unique, },
  split,
  { apply pairwise_constraint, },
  { refl, },
end

end min_ones_in_20_row_matrix_l499_499993
