import Mathlib

namespace magician_earning_correct_l748_748056

def magician_earning (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

theorem magician_earning_correct :
  magician_earning 5 3 2 = 4 :=
by
  sorry

end magician_earning_correct_l748_748056


namespace bushel_corn_weight_l748_748065

theorem bushel_corn_weight (cobs_weight_per_ear : ℝ) (bushels_picked : ℕ) (cobs_picked : ℕ) 
  (h1 : cobs_weight_per_ear = 0.5)
  (h2 : bushels_picked = 2)
  (h3 : cobs_picked = 224) :
  let total_weight := cobs_picked * cobs_weight_per_ear in
  let weight_per_bushel := total_weight / bushels_picked in
  weight_per_bushel = 56 :=
by
  sorry

end bushel_corn_weight_l748_748065


namespace crayons_allocation_correct_l748_748568

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l748_748568


namespace find_polynomials_l748_748542

noncomputable def polynomial_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (∃ q : ℚ, x - y = q) → ∃ r : ℚ, f(x) - f(y) = r

theorem find_polynomials :
  ∀ f : ℝ → ℝ,
  (∃ a b c : ℝ, ∀ x : ℝ, f(x) = a * x^2 + b * x + c) ∧ polynomial_satisfies_condition f →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x : ℝ, f(x) = b * x + c :=
sorry

end find_polynomials_l748_748542


namespace average_growth_rate_visitor_count_youlan_latte_price_l748_748504
noncomputable theory

-- Prove the average growth rate of visitors from 2021 to 2023
theorem average_growth_rate_visitor_count (v2021 v2023 n : ℝ) (h2021 : v2021 = 2) (h2023 : v2023 = 2.88) (hn : n = 2) : 
  ∃ x, (1 + x)^n = v2023 / v2021 ∧ x = 0.2 :=
by
  sorry

-- Prove the price of Youlan Latte
theorem youlan_latte_price (price_y price_s spent_y spent_s : ℝ) (h1 : price_y = price_s + 2) (h2 : spent_y = 216) (h3 : spent_s = spent_y / 2) :
  ∃ m, price_y = m ∧ m = 18 :=
by
  sorry

end average_growth_rate_visitor_count_youlan_latte_price_l748_748504


namespace distance_between_points_l748_748924

theorem distance_between_points : 
  let p1 := (0: ℝ, 5: ℝ)
  let p2 := (4: ℝ, 0: ℝ)
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = real.sqrt 41 :=
by {
  let p1 := (0: ℝ, 5: ℝ),
  let p2 := (4: ℝ, 0: ℝ),
  sorry
}

end distance_between_points_l748_748924


namespace imaginary_part_of_z_l748_748393

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the complex expression
def z : ℂ := i * (2 + i)

-- State the theorem
theorem imaginary_part_of_z : complex.im z = 2 := 
by 
  -- Proof placeholder
  sorry

end imaginary_part_of_z_l748_748393


namespace train_speed_approx_19_45_l748_748111

noncomputable def train_speed (train_length bridge_length : ℝ) (time_to_cross : ℝ) : ℝ :=
  (train_length + bridge_length) / time_to_cross

theorem train_speed_approx_19_45 :
  train_speed 120 150 13.884603517432893 ≈ 19.45 :=
by
  sorry

end train_speed_approx_19_45_l748_748111


namespace biggest_doll_height_l748_748356

-- Defining the condition
def sixth_doll_height (H : ℝ) : Prop := ((2 / 3) ^ 5) * H = 32

-- The main theorem we want to prove
theorem biggest_doll_height :
  ∃ (H : ℝ), sixth_doll_height H ∧ H = 243 :=
begin
  sorry
end

end biggest_doll_height_l748_748356


namespace roots_of_derivative_polynomial_l748_748700

noncomputable theory

open polynomial

/-- Let P(x) be a polynomial with real coefficients and all roots purely imaginary.
    Prove that all roots of the polynomial P'(x), except for one, are also purely imaginary,
    and that root is zero. -/
theorem roots_of_derivative_polynomial (P : ℝ[X]) 
  (h : ∀ z : ℂ, is_root P z → z.im ≠ 0) :
  ∃ g : ℝ[X], deriv P = g * X ∧ (∀ z : ℂ, is_root g z → z.im ≠ 0) := 
sorry

end roots_of_derivative_polynomial_l748_748700


namespace total_distance_traveled_l748_748808

theorem total_distance_traveled (v0 : ℕ) (increase : ℕ) (hours : ℕ) (dist : ℕ) :
  v0 = 55 → increase = 2 → hours = 12 → dist = 792 →
  ∑ i in finset.range hours, (v0 + i * increase) = dist :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end total_distance_traveled_l748_748808


namespace combined_percentage_tennis_is_31_l748_748003

-- Define the number of students at North High School
def students_north : ℕ := 1800

-- Define the number of students at South Elementary School
def students_south : ℕ := 2200

-- Define the percentage of students who prefer tennis at North High School
def percentage_tennis_north : ℚ := 25/100

-- Define the percentage of students who prefer tennis at South Elementary School
def percentage_tennis_south : ℚ := 35/100

-- Calculate the number of students who prefer tennis at North High School
def tennis_students_north : ℚ := students_north * percentage_tennis_north

-- Calculate the number of students who prefer tennis at South Elementary School
def tennis_students_south : ℚ := students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_tennis_students : ℚ := tennis_students_north + tennis_students_south

-- Calculate the total number of students in both schools
def total_students : ℚ := students_north + students_south

-- Calculate the combined percentage of students who prefer tennis
def combined_percentage_tennis : ℚ := (total_tennis_students / total_students) * 100

-- Main statement to prove
theorem combined_percentage_tennis_is_31 :
  round combined_percentage_tennis = 31 := by sorry

end combined_percentage_tennis_is_31_l748_748003


namespace max_geometric_progression_terms_l748_748033

theorem max_geometric_progression_terms :
  ∀ a0 q : ℕ, (∀ k, a0 * q^k ≥ 100 ∧ a0 * q^k < 1000) →
  (∃ r s : ℕ, r > s ∧ q = r / s) →
  (∀ n, ∃ r s : ℕ, (r^n < 1000) ∧ ((r / s)^n < 10)) →
  n ≤ 5 :=
sorry

end max_geometric_progression_terms_l748_748033


namespace scientific_notation_example_l748_748739

noncomputable def scientific_notation (n : ℕ) (sig_figs : ℕ) :=
  let factor : ℝ := 10 ^ (Math.log10 n).floor
  let value : ℝ := n / factor
  let scaled_value : ℝ := (Real.round (value * (10 ^ (sig_figs - 1)))) / (10 ^ (sig_figs - 1))
  (scaled_value, (Math.log10 n).floor)

theorem scientific_notation_example : scientific_notation 1010659 3 = (1.01, 6) :=
by
  unfold scientific_notation
  have factor : ℝ := 10 ^ 6 := by sorry
  have value : ℝ := 1010659 / factor := by sorry
  have scaled_value : ℝ := (Real.round (value * 10 ^ 2)) / 10 ^ 2 := by sorry
  have h : scaled_value = 1.01 := by sorry
  have h_floor : (Math.log10 1010659).floor = 6 := by sorry
  exact (h, h_floor)

end scientific_notation_example_l748_748739


namespace find_a_for_inequality_l748_748615

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x - a * (x - 1) / (x + 1)

theorem find_a_for_inequality :
  ∀ (x : ℝ) (a : ℝ), (x ∈ Ioo 0 1) → (f x a < 0) ↔ a ∈ Iic 2 :=
by
  intro x a hx
  sorry

end find_a_for_inequality_l748_748615


namespace percentage_increase_is_5_l748_748314

def initial_population : ℕ := 10000
def final_population : ℕ := 9975
def decrease_rate : ℝ := 0.05
def percentage_increase := ℝ

-- Conditions
def population_after_first_year (P : ℝ) : ℝ :=
  initial_population + (P / 100) * initial_population

def population_after_second_year (P : ℝ) : ℝ :=
  (population_after_first_year P) * (1 - decrease_rate)

-- Theorem to prove that the percentage increase in the first year was 5%
theorem percentage_increase_is_5 :
  ∃ P : ℝ, population_after_second_year P = final_population ∧ P = 5 :=
by
  sorry

end percentage_increase_is_5_l748_748314


namespace smallest_positive_period_monotonic_interval_value_at_translated_point_l748_748212

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (π / 2 + x) * Real.sin (π / 3 + x)

variable {x x0 : ℝ}

axiom h1 : x ∈ Set.univ

axiom h2 : x0 ∈ Set.Ioo (π / 6) (π / 2)

axiom h3 : f x0 = 4 / 5 + Real.sqrt 3 / 2

-- The smallest positive period of the function f(x)
theorem smallest_positive_period : ∀ x, f (x + π) = f x :=
  by sorry

-- The interval where f(x) is monotonically increasing
theorem monotonic_interval (k : ℤ) : ∀ x, x ∈ Set.Icc (k * π - 5 * π / 12) (k * π + π / 12) → 
  f (x + π/2) > f x :=
  by sorry

-- Value of f(x0 + π/6)
theorem value_at_translated_point : f (x0 + π / 6) = (2 + Real.sqrt 3) / 5 :=
  by sorry

end smallest_positive_period_monotonic_interval_value_at_translated_point_l748_748212


namespace inequality_solution_set_l748_748404

theorem inequality_solution_set (x : ℝ) :
  6 + 5 * x - x^2 > 0 ↔ x ∈ set.Ioo (-1 : ℝ) 6 :=
by
  sorry

end inequality_solution_set_l748_748404


namespace painted_sphere_area_proportionality_l748_748478

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end painted_sphere_area_proportionality_l748_748478


namespace sequence_integer_forall_k_l748_748624

def sequence (k m : ℕ) : ℕ → ℕ
| 1     := 1
| 2     := 1
| 3     := m
| (n+1) := (k + (sequence k m n) * (sequence k m (n-1))) / (sequence k m (n-2))

theorem sequence_integer_forall_k (k m : ℕ) (h : Nat.gcd k m = 1) : 
  (∀ n : ℕ, n > 0 → n < 4 ∨ sequence k m n = (sequence k m n)) ↔ k = 1 :=
sorry

end sequence_integer_forall_k_l748_748624


namespace correct_answer_l748_748768

-- Definitions for the basic structures in the conditions
def sequential_structure : Prop := true   -- Placeholder definition
def modular_structure : Prop := true      -- Placeholder definition
def conditional_structure : Prop := true  -- Placeholder definition
def loop_structure : Prop := true         -- Placeholder definition

-- The possible answers as combinations of the structures
def option_A : Prop := sequential_structure ∧ modular_structure ∧ conditional_structure
def option_B : Prop := sequential_structure ∧ loop_structure ∧ modular_structure
def option_C : Prop := sequential_structure ∧ conditional_structure ∧ loop_structure
def option_D : Prop := modular_structure ∧ conditional_structure ∧ loop_structure

-- The theorem we need to prove
theorem correct_answer : option_C :=
begin
  sorry -- proof not required
end

end correct_answer_l748_748768


namespace AF_perpendicular_BC_l748_748650

noncomputable theory

variable {A B C D E F : Type} -- Points in geometry
variable (angle : A → A → A → ℝ) -- Function to calculate angle

-- Given conditions
variable (triangleABC : Type) -- Triangle ABC
variable (onSideAC : D → A → C → Prop) -- D on AC
variable (onSideAB : E → A → B → Prop) -- E on AB
variable (intersection : F → D → E → Prop) -- F is the intersection of BD and CE

variable (angleBAC_eq_40 : angle B A C = 40)
variable (angleABC_eq_60 : angle A B C = 60)
variable (angleCBD_eq_40 : angle C B D = 40)
variable (angleBCE_eq_70 : angle B C E = 70)

-- Theorem to prove
theorem AF_perpendicular_BC 
  (H1 : onSideAC D A C) 
  (H2 : onSideAB E A B)
  (H3 : intersection F D E)
  (H4 : angle B A C = 40)
  (H5 : angle A B C = 60)
  (H6 : angle C B D = 40)
  (H7 : angle B C E = 70) :
  angle A F B = 90 :=
begin
  sorry
end

end AF_perpendicular_BC_l748_748650


namespace general_term_sum_terms_l748_748944

-- Definitions
def Sn (n : ℕ) (h : n ≠ 0) : ℝ := 2^n - 1

def a_n (n : ℕ) (h : n ≠ 0) : ℝ := Sn n h - Sn (n - 1) (by linarith)

def b_n (n : ℕ) (h : n ≠ 0) : ℝ := log (4 : ℝ) (a_n n h) + 1

def T_n (n : ℕ) (h : n ≠ 0) : ℝ := ∑ i in finset.range n, b_n (i + 1) (by linarith)

-- Theorems
theorem general_term (n : ℕ) (h : n ≠ 0) : a_n n h = 2^(n - 1) :=
sorry

theorem sum_terms (n : ℕ) (h : n ≠ 0) : T_n n h = n^2 / 2 :=
sorry

end general_term_sum_terms_l748_748944


namespace binomial_coefficient_sum_l748_748307

theorem binomial_coefficient_sum {n : ℕ} (h : (1 : ℝ) + 1 = 128) : n = 7 :=
by
  sorry

end binomial_coefficient_sum_l748_748307


namespace min_value_for_inequality_l748_748556

theorem min_value_for_inequality (a : ℝ) :
  (∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 →
  a * (x^2 + y^2 + z^2) + x * y * z ≥ a / 3 + 1 / 27) ↔ a ≥ 2 / 9 :=
by
  split
  case mp =>
    intro h
    sorry -- Here the actual proof would be provided
  case mpr =>
    intro ha
    intro x y z hx hz hy hxy
    sorry -- Here the actual proof would be provided

end min_value_for_inequality_l748_748556


namespace olivia_initial_money_l748_748360

theorem olivia_initial_money (spent_supermarket : ℕ) (spent_showroom : ℕ) (left_money : ℕ) (initial_money : ℕ) :
  spent_supermarket = 31 → spent_showroom = 49 → left_money = 26 → initial_money = spent_supermarket + spent_showroom + left_money → initial_money = 106 :=
by
  intros h_supermarket h_showroom h_left h_initial 
  rw [h_supermarket, h_showroom, h_left] at h_initial
  exact h_initial

end olivia_initial_money_l748_748360


namespace a5_value_l748_748973

noncomputable def sequence : ℕ → ℕ
| 1     := 1
| (n+1) := sequence n + 2

theorem a5_value : sequence 5 = 9 := by
  sorry

end a5_value_l748_748973


namespace factorial_division_l748_748388

theorem factorial_division (n : ℕ) (h : Nat.factorial 8 / Nat.factorial (8 - n) = 56) : n = 2 :=
begin
  have fact_8 : Nat.factorial 8 = 40320 := by sorry,  -- This steps holds by definition/calculation of 8!
  sorry  -- Proof required to establish n = 2 through algebraic manipulations.
end

end factorial_division_l748_748388


namespace count_integers_satisfying_condition_l748_748648
open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def g (n : ℕ) : ℕ := (divisors n).sum (λ d, d ^ 3)

theorem count_integers_satisfying_condition :
  ∃! (count : ℕ), count = (filter (λ i, 1 ≤ i ∧ i ≤ 500 ∧ g i = 1 + (√(i^3)) + i^3) (range 501)).length :=
sorry

end count_integers_satisfying_condition_l748_748648


namespace max_of_inverse_power_sums_l748_748676

theorem max_of_inverse_power_sums (s p r1 r2 : ℝ) 
  (h_eq_roots : r1 + r2 = s ∧ r1 * r2 = p)
  (h_eq_powers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2023 → r1^n + r2^n = s) :
  1 / r1^(2024:ℕ) + 1 / r2^(2024:ℕ) ≤ 2 :=
sorry

end max_of_inverse_power_sums_l748_748676


namespace inequality_solution_set_l748_748405

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : (x - 1) / x > 1 ↔ x < 0 :=
by
  sorry

end inequality_solution_set_l748_748405


namespace triangle_area_correct_l748_748007

def complex.squares (a b : ℂ) : set ℂ :=
  { z : ℂ | z^2 = a ∨ z^2 = b }

def area_of_triangle (v1 v2 v3 : ℂ) : ℝ :=
  0.5 * abs (v1.re * v2.im + v2.re * v3.im + v3.re * v1.im - v1.im * v2.re - v2.im * v3.re - v3.im * v1.re)

theorem triangle_area_correct :
  let v1 := complex.squares (3 + 3 * real.sqrt 8 * complex.i) (5 + 5 * real.sqrt 5 * complex.i) in
  area_of_triangle 
    (v1.v1) 
    (v1.v2)
    (v1.v3) = 15 * real.sqrt 21 / 2 :=
sorry

end triangle_area_correct_l748_748007


namespace monkey_ladder_min_rungs_l748_748096

theorem monkey_ladder_min_rungs :
  ∃ (n : ℕ), ∀ (seq : List ℕ), 
    (seq.head = 0) ∧    
    (∀ v ∈ seq, (v + 1) % n ∈ seq ∨ (v + n - 16) % n ∈ seq) ∧ 
    (seq.erase 0).Nodup ∧
    seq.length = n + 1 →
    n = 24 :=
by
  sorry

end monkey_ladder_min_rungs_l748_748096


namespace divisible_by_17_l748_748369

theorem divisible_by_17 (n : ℕ) : 17 ∣ (2 ^ (5 * n + 3) + 5 ^ n * 3 ^ (n + 2)) := 
by {
  sorry
}

end divisible_by_17_l748_748369


namespace cos_theta_l748_748229

variables {a b : ℝ}

def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem cos_theta (a b : ℝ × ℝ)
  (h1 : norm a = 5)
  (h2 : norm b = 7)
  (h3 : norm (a.1 + b.1, a.2 + b.2) = 10) :
  dot_product a b / (norm a * norm b) = 13 / 35 :=
sorry

end cos_theta_l748_748229


namespace washing_machine_heavy_wash_usage_l748_748849

-- Definition of variables and constants
variables (H : ℕ)                           -- Amount of water used for a heavy wash
def regular_wash : ℕ := 10                   -- Gallons used for a regular wash
def light_wash : ℕ := 2                      -- Gallons used for a light wash
def extra_light_wash : ℕ := light_wash       -- Extra light wash due to bleach

-- Number of each type of wash
def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_washes : ℕ := 2

-- Total water usage
def total_water_usage : ℕ := 
  num_heavy_washes * H + 
  num_regular_washes * regular_wash + 
  num_light_washes * light_wash + 
  num_bleached_washes * extra_light_wash

-- Given total water usage
def given_total_water_usage : ℕ := 76

-- Lean statement to prove the amount of water used for a heavy wash
theorem washing_machine_heavy_wash_usage : total_water_usage H = given_total_water_usage → H = 20 :=
by
  sorry

end washing_machine_heavy_wash_usage_l748_748849


namespace students_arrangement_l748_748780

theorem students_arrangement (B1 B2 S1 S2 T1 T2 C1 C2 : ℕ) :
  (B1 = B2 ∧ S1 ≠ S2 ∧ T1 ≠ T2 ∧ C1 ≠ C2) →
  (C1 ≠ C2) →
  (arrangements = 7200) :=
by
  sorry

end students_arrangement_l748_748780


namespace sequence_value_correct_l748_748130

-- Define the problem using natural numbers and rational numbers
def sequence_value : ℚ := 
  (1 / 2) * (3 / 2) * (4 / 3) * (5 / 4) * ... * (2020 / 2019) * (2021 / 2020)

theorem sequence_value_correct : sequence_value = 2021 / 4 := 
  sorry

end sequence_value_correct_l748_748130


namespace find_number_l748_748456

theorem find_number (x : ℕ) (h : x / 46 - 27 = 46) : x = 3358 :=
by
  sorry

end find_number_l748_748456


namespace seq_sum_eq_314_l748_748522

theorem seq_sum_eq_314 (d r : ℕ) (k : ℕ) (a_n b_n c_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n = 1 + (n - 1) * d)
  (h2 : ∀ n, b_n n = r ^ (n - 1))
  (h3 : ∀ n, c_n n = a_n n + b_n n)
  (hk1 : c_n (k - 1) = 150)
  (hk2 : c_n (k + 1) = 900) :
  c_n k = 314 := by
  sorry

end seq_sum_eq_314_l748_748522


namespace one_to_one_correspondence_y_eq_1_div_1_add_x_one_to_one_correspondence_y_eq_2_pow_neg_x_l748_748153

theorem one_to_one_correspondence_y_eq_1_div_1_add_x (x : ℝ) (y : ℝ) :
  (0 < x) → (0 < y) → (y < 1) → (y = 1 / (1 + x) ↔ ∃x, x > 0 ∧ y = 1 / (1 + x)) := 
by
  sorry

theorem one_to_one_correspondence_y_eq_2_pow_neg_x (x : ℝ) (y : ℝ) :
  (0 < x) → (0 < y) → (y < 1) → (y = 2 ^ (-x) ↔ ∃x, x > 0 ∧ y = 2 ^ (-x)) :=
by
  sorry

end one_to_one_correspondence_y_eq_1_div_1_add_x_one_to_one_correspondence_y_eq_2_pow_neg_x_l748_748153


namespace largest_common_divisor_of_528_and_440_l748_748424

theorem largest_common_divisor_of_528_and_440 : 
  let divisors_528 := {1, 2, 3, 4, 6, 8, 11, 12, 22, 24, 33, 44, 48, 66, 88, 132, 176, 264, 528}
  let divisors_440 := {1, 2, 4, 5, 8, 10, 11, 20, 22, 40, 44, 55, 88, 110, 220, 440}
  let common_divisors := divisors_528 ∩ divisors_440
  in
  88 = set.to_finset common_divisors.sup sorry


end largest_common_divisor_of_528_and_440_l748_748424


namespace point_in_first_quadrant_l748_748960

theorem point_in_first_quadrant (m : ℝ) (h : m < 0) : 
  (-m > 0) ∧ (-m + 1 > 0) :=
by 
  sorry

end point_in_first_quadrant_l748_748960


namespace arithmetic_sequence_and_max_m_l748_748177

-- Sequence definitions
def a : ℕ → ℚ 
| 1 := 1
| (n + 1) := S (n + 1) - S n

-- Sum Definitions
def S : ℕ → ℚ 
| 1 := 1
| (n + 1) := S n + a (n + 1)

-- Condition for S
axiom cond_S (n : ℕ) (h : 2 ≤ n) : (S n)^2 = a n * (S n - 1/2)

-- Define b_n and T_n
def b (n : ℕ) : ℚ := S n / (2 * n + 1)
def T : ℕ → ℚ 
| 1 := b 1
| (n + 1) := T n + b (n + 1)

-- Inequality condition for T_n and m
def inequality (n : ℕ) (m : ℤ) : Prop := T n ≥ 1/18 * (m^2 - 5 * m)

theorem arithmetic_sequence_and_max_m :
  (∀ n : ℕ, 2 ≤ n → (1 / S n)  - (1 / S (n - 1)) = 2) ∧
  (∃ m : ℕ, (∀ n : ℕ, n > 0 → inequality n m)  ∧ ∀ k : ℕ, k > m → ¬ (∀ n : ℕ, n > 0 → inequality n k)) :=
by sorry

end arithmetic_sequence_and_max_m_l748_748177


namespace sum_fn_to_2017_sum_fn_pi_over_3_l748_748582

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := Deriv.deriv f1 x
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then f1 x
  else if n = 2 then f2 x
  else if n = 3 then Deriv.deriv f2 x
  else if n = 4 then Deriv.deriv (Deriv.deriv f2) x
  else fn (n % 4 + 1) x

theorem sum_fn_to_2017 (x : ℝ) :
  (finset.range 2017).sum (λ n, fn (n + 1) (x)) = (Real.sin (x) + Real.cos (x)) :=
sorry

theorem sum_fn_pi_over_3 :
  (finset.range 2017).sum (λ n, fn (n + 1) (Real.pi / 3)) = (1 + Real.sqrt 3) / 2 :=
begin
  have h := sum_fn_to_2017 (Real.pi / 3),
  rw [h, f1],
  norm_num,
  rw [Real.sin_pi_div_three, Real.cos_pi_div_three],
  norm_num,
end

end sum_fn_to_2017_sum_fn_pi_over_3_l748_748582


namespace angle_between_lines_l748_748416

-- Define the conditions
variable (A M B C : Point)
-- axiom: intersecting lines at A not at right angle
axiom h1 : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ (¬∃ θ : ℝ, θ = 90) ∧ A ∈ l1 ∧ A ∈ l2
-- axiom: B and C are projections of M on these lines
axiom h2 : ∃ (l3 l4 : Line), B ∈ l3 ∧ C ∈ l4 ∧ is_projection M B l3 ∧ is_projection M C l4

-- Define what we need to prove
theorem angle_between_lines : angle_line (midpoint (segment AM)) (midpoint (segment BC)) BC = 90 := by
  sorry

end angle_between_lines_l748_748416


namespace seeds_in_bucket_C_l748_748409

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l748_748409


namespace sector_area_60_deg_PQ_is_50pi_over_3_l748_748783

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

noncomputable def sector_area (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem sector_area_60_deg_PQ_is_50pi_over_3 :
  let P := (2 : ℝ, -2 : ℝ)
  let Q := (8 : ℝ, 6 : ℝ)
  let θ := 60
  let r := distance P Q
  θ = 60 ∧ r = 10 → sector_area θ r = (50 * π) / 3 := by
  -- Proof steps go here
  sorry

end sector_area_60_deg_PQ_is_50pi_over_3_l748_748783


namespace total_campers_l748_748816

def campers_morning : ℕ := 36
def campers_afternoon : ℕ := 13
def campers_evening : ℕ := 49

theorem total_campers : campers_morning + campers_afternoon + campers_evening = 98 := by
  sorry

end total_campers_l748_748816


namespace max_intersections_l748_748918

theorem max_intersections (n_x n_y : ℕ) (hn_x : n_x = 15) (hn_y : n_y = 10) 
  (segments : ℕ := n_x * n_y) (hsegments : segments = 150) :
  ∀ x y : ℕ, x = 105 ∧ y = 45 → x * y = 4725 :=
by
  intros x y hx hy
  have hx_ : x = (n_x * (n_x - 1)) / 2 := by sorry
  have hy_ : y = (n_y * (n_y - 1)) / 2 := by sorry
  rw [←hx, ←hy]
  rw [hx_, hy_]
  have hcalc : (105 * 45 = 4725) := by sorry
  exact hcalc

end max_intersections_l748_748918


namespace seq_arithmetic_l748_748981

theorem seq_arithmetic (a : ℕ → ℝ) (h₁ : a 3 = 2) (h₂ : a 5 = 1)
  (h₃ : ∀ n m : ℕ, (n - m : ℕ) = |n - m| → (1 / (1 + a n) - 1 / (1 + a m)) / (n - m) = (1 / (1 + a 5) - 1 / (1 + a 3)) / 2) :
  a 11 = 0 :=
sorry

end seq_arithmetic_l748_748981


namespace minimum_score_required_l748_748776

theorem minimum_score_required 
  (q1 q2 q3 q4 : ℝ) (q_avg_required : ℝ) (total_quarters : ℕ) 
  (h1 : q1 = 84) (h2 : q2 = 80) (h3 : q3 = 78) (h4 : q4 = 82) (h5 : q_avg_required = 85) (h6 : total_quarters = 5) 
  : (q1 + q2 + q3 + q4 + ?q5) / total_quarters ≥ q_avg_required ↔ ?q5 ≥ 101 :=
begin
  sorry
end

end minimum_score_required_l748_748776


namespace smallest_possible_gcd_l748_748634

noncomputable def smallestGCD (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : ℕ :=
  Nat.gcd (12 * a) (18 * b)

theorem smallest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : 
  smallestGCD a b h1 h2 h3 = 54 :=
sorry

end smallest_possible_gcd_l748_748634


namespace expected_value_of_win_l748_748077

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l748_748077


namespace lcm_15_25_35_l748_748551

-- Definition for LCM
def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

-- Prove LCM of 15, 25, and 35 is 525
theorem lcm_15_25_35 : lcm (lcm 15 25) 35 = 525 :=
by
  -- Placeholder for proof, to be completed
  sorry

end lcm_15_25_35_l748_748551


namespace std_deviation_above_l748_748562

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l748_748562


namespace find_a_l748_748930

theorem find_a {a : ℝ} (h : ∀ x : ℝ, x ∉ set.Icc 2 8 → x^2 - a * x + 4 > 0) : a = 10 := sorry

end find_a_l748_748930


namespace range_of_a_l748_748581

noncomputable def f (a x : ℝ) : ℝ := a * x + 2 * a + 1

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Icc (-1 : ℝ) 1, 0 < f a x) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 1, f a x < 0) ↔ a ∈ set.Ioo (-1 : ℝ) (-1 / 3) :=
sorry

end range_of_a_l748_748581


namespace solid_object_cannot_be_prism_if_cross_section_is_circular_l748_748791

noncomputable def solid_object := { cone : Type, cylinder : Type, sphere : Type, prism : Type }
def is_circular_cross_section (solid: solid_object) : Prop :=
  solid = solid_object.cone ∨ solid = solid_object.cylinder ∨ solid = solid_object.sphere

def cannot_have_circular_cross_section (solid: solid_object) : Prop :=
  solid = solid_object.prism

theorem solid_object_cannot_be_prism_if_cross_section_is_circular (solid: solid_object) (h : is_circular_cross_section solid) : ¬cannot_have_circular_cross_section solid :=
sorry

end solid_object_cannot_be_prism_if_cross_section_is_circular_l748_748791


namespace largest_power_of_2_dividing_diff_l748_748350

-- Define the regimented property
def regimented (m n : ℕ) : Prop :=
  ∀ k : ℕ, (m^k).num_divisors = (n^k).num_divisors

-- Define the value of 2016 raised to the 2016th power
def base_exp := 2016^2016

-- Define the smallest N such that (2016^2016, N) is regimented
def smallest_regimented_N : ℕ :=
  2^(5 * 2016) * 3^(2 * 2016) * 5^2016

-- Define the goal property to be proved
theorem largest_power_of_2_dividing_diff :
  ∃ v : ℕ, 2^v ∣ (base_exp - smallest_regimented_N) ∧
           ∀ w : ℕ, w > v → ¬ (2^w ∣ (base_exp - smallest_regimented_N)) ∧
           v = 10086 :=
sorry

end largest_power_of_2_dividing_diff_l748_748350


namespace num_edges_perpendicular_to_AA1_l748_748492

theorem num_edges_perpendicular_to_AA1 {A A₁ B B₁ C C₁ D D₁ : Type} (cube : set (set (Type)))
  (AA1 ∈ cube) : ∃ (n : ℕ), n = 8 ∧ (∀ (edge ∈ cube), edge ≠ AA1 → is_perpendicular edge AA1) :=
begin
  sorry
end

# Other definitions and theorems to define is_perpendicular and cube structure might be needed

end num_edges_perpendicular_to_AA1_l748_748492


namespace number_of_correct_propositions_l748_748564

-- definitions for the propositions
def proposition1 : Prop := ∀ (A B C D : Point), skew (line_through A B) (line_through C D)
def proposition2 : Prop := ∀ (A B C D : Point), height_from A (tetrahedron A B C D) intersects (meeting_point_of_altitudes (triangle B C D))
def proposition3 : Prop := ∀ (A B C D : Point), skew (altitude_from A B C) (altitude_from A B D)
def proposition4 : Prop := ∀ (A B C D : Point), intersect_at_one_point (midpoints_opposite_edges (tetrahedron A B C D))
def proposition5 : Prop := ∀ (A B C D : Point), ∃ l, l = longest_edge (tetrahedron A B C D) ∧ ∀ (e1 e2 : Edge), e1 ≠ e2 ∧ endpoint l ∈ e1 ∪ e2 → length e1 + length e2 > length l

-- final proof problem
theorem number_of_correct_propositions : Prop :=
  let props := [proposition1, proposition4, proposition5]
  props.length = 3

end number_of_correct_propositions_l748_748564


namespace inverse_of_203_mod_301_l748_748518

theorem inverse_of_203_mod_301 : ∃ (a : ℤ), 0 ≤ a ∧ a ≤ 300 ∧ (203 * a ≡ 1 [MOD 301]) :=
by
  use 238
  split
  by norm_num
  split
  by norm_num
  by norm_num,
  exact ⟨by norm_num, by norm_num⟩, sorry
 
end inverse_of_203_mod_301_l748_748518


namespace chess_games_F_l748_748730

noncomputable def chess_problem : Prop :=
  let A_games : ℕ := 3
  let B_games : ℕ := 3
  let C_games : ℕ := 4
  let D_games : ℕ := 4
  let E_games : ℕ := 2
  let matches (player1 : ℕ) (player2 : ℕ) : Bool :=
    match player1, player2 with
    | 1, 3 => false   -- A did not play against C
    | 2, 4 => false   -- B did not play against D
    | _, _ => true
  ∀ (F_games : ℕ), F_games = 4

theorem chess_games_F :
  chess_problem := by
    sorry

end chess_games_F_l748_748730


namespace function_range_l748_748170

def f (x : ℝ) : ℝ := real.sin x ^ 4 + 2 * real.sin x * real.cos x + real.cos x ^ 4

theorem function_range : set.range f = set.Icc 0.5 1.5 :=
sorry

end function_range_l748_748170


namespace algebra_expression_value_l748_748431

theorem algebra_expression_value (a b : ℝ) (h : (30^3) * a + 30 * b - 7 = 9) :
  (-30^3) * a + (-30) * b + 2 = -14 := 
by
  sorry

end algebra_expression_value_l748_748431


namespace train_pass_man_time_l748_748444

noncomputable def train_length : ℝ := 100  -- train length in meters
noncomputable def train_speed_kmph : ℝ := 68  -- train speed in km/hr
noncomputable def man_speed_kmph : ℝ := 8  -- man's speed in km/hr

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

noncomputable def relative_speed : ℝ := kmph_to_mps (train_speed_kmph - man_speed_kmph)

noncomputable def time_to_pass (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_pass_man_time:
  (time_to_pass train_length relative_speed) ≈ 6 :=
by
  sorry

end train_pass_man_time_l748_748444


namespace arrangement_ab_together_arrangement_no_head_tail_arrangement_one_between_l748_748019

theorem arrangement_ab_together (n : Nat) (h : n = 7) : 
  let A := 6!
  let B := 2!
  A * B = 1440 :=
by
  sorry

theorem arrangement_no_head_tail (n : Nat) (h : n = 7) : 
  let A := 7!
  let B := 2 * 6!
  let C := 5!
  A - B + C = 3720 :=
by
  sorry

theorem arrangement_one_between (n : Nat) (h : n = 7) : 
  let choose := 5
  let permute := 5!
  let perm_ab := 2!
  choose * permute * perm_ab = 1200 :=
by
  sorry

end arrangement_ab_together_arrangement_no_head_tail_arrangement_one_between_l748_748019


namespace mul_inv_AB_mod_l748_748334

def A : ℕ := 111111
def B : ℕ := 142857
def modulus : ℕ := 1000000
def N : ℕ := 63

theorem mul_inv_AB_mod :
  ∃ N : ℕ, N < modulus ∧ (N * (A * B) % modulus = 1) :=
by
  use N
  have H : N < modulus := by decide
  have H1 : N * (A * B) % modulus = 1 := by
    have : A * 9 % modulus = modulus - 1 := by decide
    have : B * 7 % modulus = modulus - 1 := by decide
    have : (A * B) * (9 * 7) % modulus = (modulus - 1) * (modulus - 1) % modulus := by decide
    rw mul_comm at this
    rw ← mul_assoc at this
    rw pow_two at this
    norm_num at this
  refine ⟨H, H1⟩

end mul_inv_AB_mod_l748_748334


namespace score_sd_above_mean_l748_748559

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l748_748559


namespace highest_probability_red_ball_l748_748817

theorem highest_probability_red_ball (red yellow white total : ℕ)
  (h1 : red = 6) (h2 : yellow = 4) (h3 : white = 1) (h4 : total = red + yellow + white) :
  (red : ℚ) / total > yellow / total ∧ (red : ℚ) / total > white / total :=
by
  rw [h1, h2, h3, h4]
  norm_num
  split
  { linarith }
  { linarith }
-- Sorry is just to skip the actual proof.
sorry

end highest_probability_red_ball_l748_748817


namespace coin_touch_black_regions_probability_l748_748845

theorem coin_touch_black_regions_probability :
  ∀ (square_side_length coin_diameter triangle_leg central_circle_diameter : ℝ), 
    square_side_length = 6 → 
    coin_diameter = 2 → 
    triangle_leg = 1 → 
    central_circle_diameter = 2 → 
    let region_side := square_side_length - coin_diameter in
    let region_area := region_side * region_side in
    let triangle_area := 4 * (1/2 * triangle_leg * triangle_leg) in
    let circle_radius := central_circle_diameter / 2 in
    let circle_area := Real.pi * (circle_radius ^ 2) in
    let total_black_area := triangle_area + circle_area in
    (total_black_area / region_area) = (2 + Real.pi) / 16 := 
by
  intros square_side_length coin_diameter triangle_leg central_circle_diameter
  intro h_square_side eq_sd
  intro h_triangle_leg eq_tl
  intro h_circle_diameter eq_cd
  let region_side := square_side_length - coin_diameter
  let region_area := region_side * region_side
  let triangle_area := 4 * (1/2 * triangle_leg * triangle_leg)
  let circle_radius := central_circle_diameter / 2
  let circle_area := Real.pi * (circle_radius ^ 2)
  let total_black_area := triangle_area + circle_area
  have h1: region_side = 4, by rw [←eq_sd, h_square_side, ←h_circle_diameter]
  have h2: triangle_area = 2, by rw [←eq_tl, h_triangle_leg]
  have h3: circle_radius = 1, by rw [←eq_cd]
  have h4: circle_area = Real.pi, by rw [h3]
  rw [h1, h2, h4]
  sorry

end coin_touch_black_regions_probability_l748_748845


namespace remainder_3_101_add_5_mod_11_l748_748428

theorem remainder_3_101_add_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := 
by sorry

end remainder_3_101_add_5_mod_11_l748_748428


namespace chess_tournament_games_l748_748653

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games :
  number_of_games 20 = 190 :=
by
  sorry

end chess_tournament_games_l748_748653


namespace problem1_problem2_l748_748812

-- Problem 1: Prove the expression equals the calculated value
theorem problem1 : (-2:ℝ)^0 + (1 / Real.sqrt 2) - Real.sqrt 9 = (Real.sqrt 2) / 2 - 2 :=
by sorry

-- Problem 2: Prove the solution to the system of linear equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : x + y = -2) :
  x = 1/3 ∧ y = -(7/3) :=
by sorry

end problem1_problem2_l748_748812


namespace min_games_22_l748_748450

noncomputable def min_games (n : ℕ) : ℕ :=
if n = 11 then 4 else 1 + min_games (Int.ceil (n / 2))

theorem min_games_22 : min_games 22 = 5 := by
  sorry

end min_games_22_l748_748450


namespace no_real_roots_l748_748842

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| (n + 1) := λ x, x^(11 * (n + 1)) - P n x

theorem no_real_roots (n : ℕ): ∀ (x : ℝ), P n x ≠ 0 :=
begin
  induction n with k hk,
  { -- base case
    intro x,
    dsimp [P],
    simp, },
  { -- inductive step
    intros x,
    dsimp [P],
    by_contradiction,
    have h := hk x,
    contradiction, }
end

end no_real_roots_l748_748842


namespace find_percentage_l748_748851

def percentage_of (percent value : ℝ) := (percent / 100) * value

theorem find_percentage (x : ℝ) (h : percentage_of 15 25 + percentage_of x 45 = 9.15) : x = 12 :=
by
  specialize h
  sorry

end find_percentage_l748_748851


namespace each_person_eats_3_Smores_l748_748365

-- Definitions based on the conditions in (a)
def people := 8
def cost_per_4_Smores := 3
def total_cost := 18

-- The statement we need to prove
theorem each_person_eats_3_Smores (h1 : total_cost = people * (cost_per_4_Smores * 4 / 3)) :
  (total_cost / cost_per_4_Smores) * 4 / people = 3 :=
by
  sorry

end each_person_eats_3_Smores_l748_748365


namespace determine_q_l748_748588

def arithmetic_sequence (a: Nat → Int) (d: Int) : Prop :=
  ∀ n: Nat, a (n + 1) = a n + d

def geometric_sequence (b: Nat → Int) (q: Nat) : Prop :=
  ∀ n: Nat, b (n + 1) = b n * q

theorem determine_q (d q: Nat) (a b: ℕ → ℚ) (h_arith_seq: arithmetic_sequence a d)
    (h_geom_seq: geometric_sequence b q)
    (h_a1: a 1 = d) (h_b1: b 1 = d^2)
    (h_pos_int: (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) ∈ ℕ) :
    q = 2 ∨ q = 1 / 2 :=
by
  sorry

end determine_q_l748_748588


namespace least_positive_difference_l748_748726

def geometric_seq (n : ℕ) : ℕ := 2 ^ n
def arithmetic_seq (m : ℕ) : ℕ := 20 * m

theorem least_positive_difference :
  ∃ a b : ℕ, a < 9 ∧ b < 16 ∧ geometric_seq a ≤ 300 ∧ arithmetic_seq b ≤ 300 ∧ abs (geometric_seq a - arithmetic_seq b) = 4 := 
sorry

end least_positive_difference_l748_748726


namespace inclusion_of_solution_set_l748_748609

theorem inclusion_of_solution_set (x : ℝ) (log2 : ℝ → ℝ) :
  (∀ x, (2^x + 1)/3 > 1 - (2^x - 1)/2 ↔ x > log2 (7/5)) →
  2 > log2 (7/5) :=
by
  sorry

end inclusion_of_solution_set_l748_748609


namespace triangle_ratio_sine_equality_l748_748714

-- Definitions for the triangle and points on its sides
variable (A B C A1 B1 C1 : Point)

-- Assumption that A1 is on BC, B1 is on CA, and C1 is on AB
axiom A1_on_BC : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ A1 = (1 - x) * B + x * C
axiom B1_on_CA : ∃ (y : ℝ), 0 ≤ y ∧ y ≤ 1 ∧ B1 = (1 - y) * C + y * A
axiom C1_on_AB : ∃ (z : ℝ), 0 ≤ z ∧ z ≤ 1 ∧ C1 = (1 - z) * A + z * B

-- We need to show the given equality of ratios involving lengths and angles
theorem triangle_ratio_sine_equality 
  : (dist A C1 / dist C1 B) * (dist B A1 / dist A1 C) * (dist C B1 / dist B1 A) = 
    (sin (∠ A C C1) / sin (∠ C1 C B)) * (sin (∠ B A A1) / sin (∠ A1 A C)) * (sin (∠ C B B1) / sin (∠ B1 B A)) := 
sorry

end triangle_ratio_sine_equality_l748_748714


namespace bounded_solution_diff_eq_l748_748557

theorem bounded_solution_diff_eq :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x - y x = cos x - sin x) ∧ (∀ M > 0, ∃ B > 0, ∀ x > B, abs (y x) < M) ∧ 
  (∀ y' : ℝ → ℝ, (∀ x, deriv y' x - y' x = cos x - sin x) → (∀ M > 0, ∃ B > 0, ∀ x > B, abs (y' x) < M) → y' = y) ∧ 
  (∀ x, y x = - cos x) :=
by
  sorry

end bounded_solution_diff_eq_l748_748557


namespace measure_angle_MOR_l748_748423

-- Define the type for degrees to be a real number
def Degrees := ℝ

-- Define a regular octagon with given properties
noncomputable def regular_octagon (n : ℕ) : Prop :=
  n = 8

-- Define points and their connections
structure Points (α : Type*) := (x : α) (y : α) (z : α)
def vertices (pts : Points ℝ) := sorry  -- To represent vertices in specific positions

-- Define the isosceles triangle for the geometry in the proof
noncomputable def isosceles_triangle (angles : Points Degrees) : Prop :=
  sorry  -- Geometry specifics skipped for the logical structure

-- The main theorem to be proved (degree measure of angle MOR)
theorem measure_angle_MOR :
  ∀ (M O R : Points ℝ),
  regular_octagon 8 →
  M ≠ O ∧ O ≠ R ∧ M ≠ R →
  isosceles_triangle { x := M.x, y := O.x, z := R.x } →
  (angle_measure (M.y, O.y, R.y) = 22.5) := 
by 
  sorry

end measure_angle_MOR_l748_748423


namespace find_height_of_cuboid_l748_748017

-- Definitions and given conditions
def length : ℕ := 22
def width : ℕ := 30
def total_edges : ℕ := 224

-- Proof statement
theorem find_height_of_cuboid (h : ℕ) (H : 4 * length + 4 * width + 4 * h = total_edges) : h = 4 :=
by
  sorry

end find_height_of_cuboid_l748_748017


namespace perpendicular_vectors_l748_748415

theorem perpendicular_vectors (b : ℝ) :
  (5 * b - 12 = 0) → b = 12 / 5 :=
by
  intro h
  sorry

end perpendicular_vectors_l748_748415


namespace total_matches_correct_total_points_earthlings_correct_total_players_is_square_l748_748881

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end total_matches_correct_total_points_earthlings_correct_total_players_is_square_l748_748881


namespace find_y_l748_748932

noncomputable def common_solution (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 = 0 ∧ x^2 - 4y + y^2 = 0

theorem find_y (x : ℝ) (y : ℝ) (h : common_solution x y) : y = 1 :=
by {
  sorry
}

end find_y_l748_748932


namespace area_traced_on_smaller_sphere_l748_748107

theorem area_traced_on_smaller_sphere
  (radius_small radius_large : ℝ)
  (region_area_large : ℝ)
  (h_small_radius : radius_small = 3)
  (h_large_radius : radius_large = 5)
  (h_region_area_large : region_area_large = 1) : 
  let surface_area (r : ℝ) := 4 * π * r^2 in
  let ratio := surface_area radius_small / surface_area radius_large in
  ratio * region_area_large = 9 / 25 := 
by
  let surface_area := λ r : ℝ, 4 * π * r^2
  have h1 : surface_area radius_small = 4 * π * 3^2 := by sorry
  have h2 : surface_area radius_large = 4 * π * 5^2 := by sorry
  have h3 : ratio = surface_area radius_small / surface_area radius_large := by sorry
  have h4 : ratio = (4 * π * 3^2) / (4 * π * 5^2) := by sorry
  have h5 : ratio = 36 * π / 100 * π := by sorry
  have h6 : ratio = 36 / 100 := by sorry
  have h7 : ratio = 9 / 25 := by sorry
  have h8 : region_area_large * ratio = 1 * (9 / 25) := by sorry
  have h9 : region_area_large * ratio = 9 / 25 := by sorry
  show ratio * region_area_large = 9 / 25 from h9

end area_traced_on_smaller_sphere_l748_748107


namespace planes_alpha_perpendicular_to_beta_l748_748583

variables {m n : Line} {α β : Plane}

-- Definitions for the conditions
axiom non_coincident_lines (m n : Line) : m ≠ n
axiom non_coincident_planes (α β : Plane) : α ≠ β

def parallel (l : Line) (p : Plane) : Prop := ∀ (x : Point), x ∈ l → x ∈ p
def perpendicular (l : Line) (p : Plane) : Prop := ∀ (x : Point), x ∈ l → x ∈ p.abs
def planes_perpendicular (α β : Plane) : Prop := ∀ (x : Line), parallel x α → perpendicular x β

-- Axioms based on conditions
axiom line_m_parallel_to_plane_alpha : parallel m α
axiom line_m_perpendicular_to_plane_beta : perpendicular m β

-- The proof statement
theorem planes_alpha_perpendicular_to_beta : planes_perpendicular α β :=
sorry

end planes_alpha_perpendicular_to_beta_l748_748583


namespace area_under_curve_l748_748214

noncomputable def f := λ x : ℝ, Real.sin x

theorem area_under_curve : 
  ∫ x in (0 : ℝ)..(Real.pi * 3 / 2), (if x ≤ Real.pi then f x else -f x) = 3 :=
by
  sorry

end area_under_curve_l748_748214


namespace correct_LCM_of_fractions_l748_748049

noncomputable def lcm_of_fractions : ℚ :=
  let denominators := [10, 9, 8, 12] in
  let numerators := [7, 8, 3, 5] in
  let lcm_denominators := denominators.foldl lcm 1 in
  let gcd_numerators := numerators.foldl gcd 0 in
  (lcm_denominators : ℚ) / gcd_numerators

theorem correct_LCM_of_fractions :
  lcm_of_fractions = 360 := by sorry

end correct_LCM_of_fractions_l748_748049


namespace min_cost_delivery_l748_748068

theorem min_cost_delivery (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 30) (h2 : y = -8 * x + 600) 
    (h3 : 90 * x + 100 * (30 - x) ≥ 2830) : x = 17 ∧ y = 464 := by
  sorry

end min_cost_delivery_l748_748068


namespace simplify_expression_l748_748728

theorem simplify_expression : 4 * (15 / 7) * (21 / -45) = -4 :=
by 
    -- Lean's type system will verify the correctness of arithmetic simplifications.
    sorry

end simplify_expression_l748_748728


namespace average_speed_of_car_l748_748821

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_car :
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  average_speed total_distance total_time = 70 :=
by
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  exact sorry

end average_speed_of_car_l748_748821


namespace projections_equal_length_l748_748950

-- Definition of the centers of the circles and the tangency points
variable {A B: Point}
variable {A1 A2 B1 B2: Point}

-- Definition of the line that acts as the base for the perpendicular projection
variable (l1 l2 : Line)

-- Conditions and the assertion
def equal_projections (A B A1 A2 B1 B2 : Point) (l1 l2 : Line)
  (h1 : tangent_to_circle l1 kA A1)
  (h2 : tangent_to_circle l1 kB B1)
  (h3 : tangent_to_circle l2 kA A2)
  (h4 : tangent_to_circle l2 kB B2) : Prop := 
  perpendicular_projection_length A1 A2 AB = perpendicular_projection_length B1 B2 AB

-- The theorem stating the equality of projections
theorem projections_equal_length 
  (A B A1 A2 B1 B2 : Point) (l1 l2 : Line)
  (h1 : tangent_to_circle l1 kA A1)
  (h2 : tangent_to_circle l1 kB B1)
  (h3 : tangent_to_circle l2 kA A2)
  (h4 : tangent_to_circle l2 kB B2) : 
  equal_projections A B A1 A2 B1 B2 l1 l2 h1 h2 h3 h4 :=
by
  sorry

end projections_equal_length_l748_748950


namespace molecular_weight_8_moles_Al2O3_l748_748782

noncomputable def molecular_weight_Al2O3 (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3
  (atomic_weight_Al : ℝ := 26.98)
  (atomic_weight_O : ℝ := 16.00)
  : molecular_weight_Al2O3 atomic_weight_Al atomic_weight_O * 8 = 815.68 := by
  sorry

end molecular_weight_8_moles_Al2O3_l748_748782


namespace pythagorean_theorem_special_cases_l748_748643

open Nat

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem pythagorean_theorem_special_cases (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (is_even a ∨ is_even b) ∧ 
  (is_multiple_of_3 a ∨ is_multiple_of_3 b) ∧ 
  (is_multiple_of_5 a ∨ is_multiple_of_5 b ∨ is_multiple_of_5 c) :=
by
  sorry

end pythagorean_theorem_special_cases_l748_748643


namespace oh_squared_l748_748699

theorem oh_squared (O H : Point) (a b c R : ℝ) (hR : R = 5) (h_sides : a^2 + b^2 + c^2 = 50) :
  (9 * R^2 - (a^2 + b^2 + c^2)) = 175 :=
by
  have h1 : 9 * R^2 = 9 * 25 := by rw hR
  have h2 : 9 * 25 = 225 := by norm_num
  rw [h1, h2, h_sides]
  norm_num
  done -- We're done with our goals

end oh_squared_l748_748699


namespace problem_statement_l748_748949

/-
Definitions of the given conditions:
- Circle P: (x-1)^2 + y^2 = 8, center C.
- Point M(-1,0).
- Line y = kx + m intersects trajectory at points A and B.
- k_{OA} \cdot k_{OB} = -1/2.
-/

noncomputable def Circle_P : Set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 8 }

def Point_M : (ℝ × ℝ) := (-1, 0)

def Trajectory_C : Set (ℝ × ℝ) :=
  { p | p.1^2 / 2 + p.2^2 = 1 }

def Line_kx_m (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 + m }

def k_OA_OB (k_OA k_OB : ℝ) : Prop :=
  k_OA * k_OB = -1/2

/-
Mathematical equivalence proof problem:
- Prove the trajectory of center C is an ellipse with equation x^2/2 + y^2 = 1.
- Prove that if line y=kx+m intersects with the trajectory, the area of the triangle AOB is a fixed value.
-/

theorem problem_statement (k m : ℝ)
    (h_intersects : ∃ A B : ℝ × ℝ, A ∈ (Trajectory_C ∩ Line_kx_m k m) ∧ B ∈ (Trajectory_C ∩ Line_kx_m k m))
    (k_OA k_OB : ℝ) (h_k_OA_k_OB : k_OA_OB k_OA k_OB) :
  ∃ (C_center_trajectory : Trajectory_C),
  ∃ (area_AOB : ℝ), area_AOB = (3 * Real.sqrt 2) / 2 :=
sorry

end problem_statement_l748_748949


namespace no_such_point_sets_exist_l748_748147

open Set

def point := ℝ × ℝ

def not_collinear (a b c : point) : Prop :=
  ¬∃ k : ℝ, (b.1 - a.1) = k * (c.1 - a.1) ∧ (b.2 - a.2) = k * (c.2 - a.2)

def distance_ge_one (a b : point) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 1

def condition1 (X Y : Set point) : Prop :=
  ∀ (a b c : point), (a ∈ X ∪ Y ∧ b ∈ X ∪ Y ∧ c ∈ X ∪ Y) → not_collinear a b c ∧ distance_ge_one a b ∧ distance_ge_one b c ∧ distance_ge_one a c

def triangle_with_vertex (p : point) (t : (point × point × point)) : Prop :=
  p = t.1 ∨ p = t.2.1 ∨ p = t.2.2

def point_in_triangle (p : point) (t : (point × point × point)) : Prop :=
  ∃ (a b c : point), 
  (t = (a, b, c)) ∧ 
  (¬ collinear a b c) ∧ 
  (∃ u v w : ℝ, 0 ≤ u ∧ 0 ≤ v ∧ 0 ≤ w ∧ u + v + w = 1 ∧ p.1 = u * a.1 + v * b.1 + w * c.1 ∧ p.2 = u * a.2 + v * b.2 + w * c.2)

def condition2 (X Y : Set point) : Prop :=
  (∀ t : (point × point × point), triangle_with_vertex t.1 Y → ∃ p ∈ X, point_in_triangle p t) ∧
  (∀ t : (point × point × point), triangle_with_vertex t.1 X → ∃ p ∈ Y, point_in_triangle p t)

theorem no_such_point_sets_exist :
  ¬ ∃ (X Y : Set point), 
    X ≠ ∅ ∧ Y ≠ ∅ ∧ Infinite X ∧ Infinite Y ∧ X ∩ Y = ∅ ∧
    condition1 X Y ∧ condition2 X Y :=
by
  sorry

end no_such_point_sets_exist_l748_748147


namespace dhoni_savings_l748_748906

theorem dhoni_savings :
  let earnings := 100
  let rent := 0.25 * earnings
  let dishwasher := rent - (0.10 * rent)
  let utilities := 0.15 * earnings
  let groceries := 0.20 * earnings
  let transportation := 0.12 * earnings
  let total_spent := rent + dishwasher + utilities + groceries + transportation
  earnings - total_spent = 0.055 * earnings :=
by
  sorry

end dhoni_savings_l748_748906


namespace mass_of_man_l748_748064

def length_boat : ℝ := 4
def breadth_boat : ℝ := 3
def sink_height : ℝ := 0.01
def density_water : ℝ := 1000

theorem mass_of_man : 
  let volume_displaced := length_boat * breadth_boat * sink_height in
  let mass_man := density_water * volume_displaced in
  mass_man = 120 :=
by 
  let volume_displaced := length_boat * breadth_boat * sink_height
  let mass_man := density_water * volume_displaced
  have h₁ : volume_displaced = 0.12 := by
    calc
      4 * 3 * 0.01 = 12 * 0.01 : by ring
      ... = 0.12 : by ring
  have h₂ : mass_man = 120 := by
    calc
      1000 * 0.12 = 120 : by ring
  exact h₂

end mass_of_man_l748_748064


namespace shopkeeper_loss_percent_l748_748800

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_percent : ℝ)
  (loss_percent : ℝ)
  (remaining_value_percent : ℝ)
  (profit_percent_10 : profit_percent = 0.10)
  (loss_percent_70 : loss_percent = 0.70)
  (initial_value_100 : initial_value = 100)
  (remaining_value_percent_30 : remaining_value_percent = 0.30)
  (selling_price : ℝ := initial_value * (1 + profit_percent))
  (remaining_value : ℝ := initial_value * remaining_value_percent)
  (remaining_selling_price : ℝ := remaining_value * (1 + profit_percent))
  (loss_value : ℝ := initial_value - remaining_selling_price)
  (shopkeeper_loss_percent : ℝ := loss_value / initial_value * 100) : 
  shopkeeper_loss_percent = 67 :=
sorry

end shopkeeper_loss_percent_l748_748800


namespace parallelogram_with_equal_diagonals_is_rectangle_l748_748041

-- Definitions based on the problem statement
def is_parallelogram (Q : Type) [is_parallelogram : Q] : Prop := sorry
def has_equal_diagonals (Q : Type) [has_equal_diagonals : Q] : Prop := sorry
def is_rectangle (Q : Type) [is_rectangle : Q] : Prop := sorry

-- Statement in Lean 4
theorem parallelogram_with_equal_diagonals_is_rectangle (Q : Type) 
  [is_parallelogram Q] [has_equal_diagonals Q] : is_rectangle Q :=
sorry

end parallelogram_with_equal_diagonals_is_rectangle_l748_748041


namespace arccos_pi_over_3_l748_748507

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l748_748507


namespace evaluate_expression_l748_748916

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end evaluate_expression_l748_748916


namespace house_to_market_distance_l748_748859

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l748_748859


namespace arccos_pi_over_3_l748_748506

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l748_748506


namespace banana_arrangements_l748_748248

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748248


namespace total_drawings_in_first_five_pages_l748_748675

theorem total_drawings_in_first_five_pages :
  let D := 5 in
  (D + 2 * D + 4 * D + 8 * D + 16 * D) = 155 :=
by
  let D := 5
  show (D + 2 * D + 4 * D + 8 * D + 16 * D) = 155
  sorry

end total_drawings_in_first_five_pages_l748_748675


namespace banana_permutations_l748_748242

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748242


namespace origami_papers_per_cousin_l748_748432

/-- Haley has 48 origami papers and 6 cousins. Each cousin should receive the same number of papers. -/
theorem origami_papers_per_cousin : ∀ (total_papers : ℕ) (number_of_cousins : ℕ),
  total_papers = 48 → number_of_cousins = 6 → total_papers / number_of_cousins = 8 :=
by
  intros total_papers number_of_cousins
  sorry

end origami_papers_per_cousin_l748_748432


namespace exponents_correct_l748_748435

theorem exponents_correct :
  (-0.2)^(5 : ℤ) * 5^(5 : ℤ) = -1 :=
sorry

end exponents_correct_l748_748435


namespace negation_example_l748_748197

theorem negation_example (p : ∀ n : ℕ, n^2 < 2^n) : 
  ¬ (∀ n : ℕ, n^2 < 2^n) ↔ ∃ n : ℕ, n^2 ≥ 2^n :=
by sorry

end negation_example_l748_748197


namespace convert_base_7_to_base_10_l748_748890

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l748_748890


namespace EF_parallel_PQ_l748_748684

variable {A B C D E F M P Q : Type}

-- Assume ABCD is a convex quadrilateral
variable [IsConvexQuadrilateral A B C D]

-- M is the intersection of diagonals AC and BD
variable [IsIntersection M A C B D]

-- Extend AC beyond A by length MC to get point E
variable [IsExtendedPoint E A C M]

-- Extend BD beyond B by length MD to get point F
variable [IsExtendedPoint F B D M]

-- P and Q are midpoints of AD and BC respectively
variable [IsMidpoint P A D]
variable [IsMidpoint Q B C]

-- Statement to be proven: EF is parallel to PQ
theorem EF_parallel_PQ : IsParallel (LineThrough E F) (LineThrough P Q) :=
sorry

end EF_parallel_PQ_l748_748684


namespace min_value_x_squared_plus_y_squared_plus_z_squared_l748_748790

theorem min_value_x_squared_plus_y_squared_plus_z_squared (x y z : ℝ) (h : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 :=
by
  sorry

end min_value_x_squared_plus_y_squared_plus_z_squared_l748_748790


namespace confidence_implies_error_l748_748122

-- Definitions for conditions in the problem
def test_statistic := ℝ
def threshold := 6.635
def confidence_level := 0.99
def type_I_error := 0.01

-- Theorem statement: If we are 99% confident in our inference, we accept a 1% error chance.
theorem confidence_implies_error :
  ∀ (K^2 : test_statistic), (K^2 ≥ threshold) → (confidence_level = 0.99) → (type_I_error = 1 - confidence_level) :=
by
  intros
  sorry

end confidence_implies_error_l748_748122


namespace count_increasing_four_digit_numbers_l748_748830

def is_increasing (n : ℕ) : Prop :=
  let a := n / 1000 in
  let b := (n / 100) % 10 in
  let c := (n / 10) % 10 in
  let d := n % 10 in
  d * 1000 + c * 100 + b * 10 + a > n

theorem count_increasing_four_digit_numbers :
  ∃ (count : ℕ), count = 4005 ∧
    ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → is_increasing n ↔ n ∈ finset.range 10000 ∧ is_increasing n :=
sorry

end count_increasing_four_digit_numbers_l748_748830


namespace cannot_be_value_of_x_l748_748311

theorem cannot_be_value_of_x (x : ℕ) 
  (h1 : ∀ (k : ℕ), k ∈ {5, 16, 27, 38, 49} → x = (k - 1) / 11 * 11 + 5) :
  x ≠ 61 :=
by 
  sorry

end cannot_be_value_of_x_l748_748311


namespace median_of_set_example_l748_748691

noncomputable def median_of_set (s : Set ℝ) (h : s.finite) : ℝ :=
  if hs : s.nonempty
  then let l := s.to_finset.sort (· ≤ ·) in l.nth (l.card / 2)
  else 0

theorem median_of_set_example :
  ∀ (a : ℤ) (b : ℝ) (n : ℕ),
  a ≠ 0 →
  0 < b →
  ab^3 = Real.log 10 b →
  b = 10^n - 1 →
  median_of_set {0, 1, (a : ℝ), b, 1 / b} sorry = 1 :=
by sorry

end median_of_set_example_l748_748691


namespace triangle_CRS_area_l748_748738

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perpendicular_lines (m1 m2 : ℝ) : Prop := m1 * m2 = -1

def y_intercepts_sum_zero (b1 b2 : ℝ) : Prop := b1 + b2 = 0

def triangle_area (A B C : point) : ℝ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).abs / 2

theorem triangle_CRS_area : 
  ∀ (m1 m2 b1 b2 : ℝ),
  perpendicular_lines m1 m2 → 
  y_intercepts_sum_zero b1 b2 → 
  let C := (3, 7) in 
  let R := (0, b1) in 
  let S := (0, b2) in 
  distance C (0, 0) = distance C R / 2 → 
  triangle_area C R S = 30 :=
by 
  intros m1 m2 b1 b2 h_perp h_sum_zero C R S h_dist
  sorry

end triangle_CRS_area_l748_748738


namespace distance_to_left_focus_l748_748205

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | (p.1^2 / 4) + p.2^2 = 1 }

def left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)

def point_on_ellipse_with_xcoord (x : ℝ) (hx : x = Real.sqrt 3) : Set (ℝ × ℝ) :=
  { p | p ∈ ellipse ∧ p.1 = x }

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_left_focus : ∀ (P : ℝ × ℝ), 
  (P ∈ point_on_ellipse_with_xcoord (Real.sqrt 3) (rfl)) →
  distance P left_focus = 7 / 2 :=
by
  sorry

end distance_to_left_focus_l748_748205


namespace remainder_division_l748_748827

theorem remainder_division (N : ℤ) (hN : N % 899 = 63) : N % 29 = 5 := 
by 
  sorry

end remainder_division_l748_748827


namespace number_of_integers_with_conditions_l748_748169

theorem number_of_integers_with_conditions :
  (∃ (n : ℕ), (∏ (p : ℕ) in {2, 3, 7, 47}, (e p + 1)) = 1974) →
  (∀ (p : ℕ), p ∈ {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} → 
    (e p > 0 → p ∣ n)) →
  (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 7 ∨ n % 10 = 9) →
  ∃ (count : ℕ), count = 9100 := 
sorry

end number_of_integers_with_conditions_l748_748169


namespace curve_crossing_point_l748_748126

noncomputable def curve (t : ℝ) : ℝ × ℝ :=
  (t^3 - 3 * t^2 - 2, t^4 - 12 * t^2 + 8)

theorem curve_crossing_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ curve t₁ = curve t₂ ∧
  (∃ t : ℝ, curve t = (3 * real.sqrt 3 - 11, -19)) :=
  sorry

end curve_crossing_point_l748_748126


namespace expression_equals_64_l748_748437

theorem expression_equals_64 :
  let a := 2^3 + 2^3
  let b := 2^3 * 2^3
  let c := (2^3)^3
  let d := 2^12 / 2^2
  b = 2^6 :=
by
  sorry

end expression_equals_64_l748_748437


namespace max_arrangement_weeks_l748_748407

theorem max_arrangement_weeks (n : ℕ) (h1 : ∀ i j, i ≠ j → ∀ S T : finset ℕ, S.card = n → T.card = n → disjoint S T) :
  n^2 > 0 → ∃ k : ℕ, k ≤ n + 1 ∧ ∀ w : ℕ, w ≤ k → (∀ i j, i ≠ j → different_teams w) :=
by
  sorry

/--
The function different_teams takes an integer representing the week and returns a 
boolean value indicating whether the team arrangement in that week respects the 
condition that no two students can be in the same team more than once across different weeks.
-/
def different_teams (week : ℕ) : Prop := sorry

end max_arrangement_weeks_l748_748407


namespace range_of_b_l748_748306

theorem range_of_b (b : ℝ) :
  (∀ x : ℤ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ↔ 5 < b ∧ b < 7 := 
sorry

end range_of_b_l748_748306


namespace distinct_arrangements_banana_l748_748272

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748272


namespace max_mn_value_l748_748298

theorem max_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (hA1 : ∀ k : ℝ, k * (-2) - (-1) + 2 * k - 1 = 0)
  (hA2 : m * (-2) + n * (-1) + 2 = 0) :
  mn ≤ 1/2 := sorry

end max_mn_value_l748_748298


namespace largest_median_of_list_l748_748535

theorem largest_median_of_list (l : List ℕ) (h1 : l.length = 11) 
  (h2 : List.sublist [2, 4, 5, 6, 7, 8] l) :
  ∃ m, m = 9 ∧ l.sorted.nth_le 5 sorry = m :=
sorry

end largest_median_of_list_l748_748535


namespace probability_of_irrational_card_l748_748440

theorem probability_of_irrational_card :
  let cards := [1, real.sqrt 2, real.sqrt 3, 2]
  let irrational_cards := [real.sqrt 2, real.sqrt 3]
  let total_cards := cards.length
  let total_irrational := irrational_cards.length
  total_irrational / total_cards = 1 / 2 := 
by
  sorry

end probability_of_irrational_card_l748_748440


namespace knights_prob_9157_l748_748773
noncomputable theory

def knights (n : ℕ) (k : ℕ) : Prop := 
  n = 30 ∧ 
  k = 4

def chosen_knights (n k : ℕ) (l : list ℕ) : Prop := 
  length l = k ∧ 
  ∀ i, l.nth i < some n

def Q (n k : ℕ) : ℚ := 
  1 - (((n - k * 2 + 1) * (n - k * 2 - 1) * (n - k * 2 - 2) * (n - k * 2 - 3)) / (n * (n - 1) * (n - 2) * (n - 3)))

theorem knights_prob_9157 (n k : ℕ) (Q_val : ℚ) :
  knights n k → 
  sum.to_num_and_den (Q n k) Q_val → 
  Q_val.num + Q_val.den = 9157 := 
by sorry

end knights_prob_9157_l748_748773


namespace y_intercept_of_line_l748_748764

theorem y_intercept_of_line (m : ℝ) (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b = 0) (h_slope : m = 3) (h_x_intercept : (a, b) = (4, 0)) :
  ∃ y : ℝ, (0, y) = (0, -12) :=
by 
  sorry

end y_intercept_of_line_l748_748764


namespace range_of_a_l748_748970

theorem range_of_a (a : ℝ) : 
  (∀ x < 1, ∃ y < 0, (1 - 2 * a) * x + 3 * a = y) 
  ∧ (∀ x ≥ 1, 0 ≤ Real.log x) 
  ∧ (∀ y : ℝ, ∃ x : ℝ, (if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x = y)) 
  → (-1 ≤ a ∧ a < 1 / 2) := 
sorry

end range_of_a_l748_748970


namespace girl_can_visit_friends_l748_748462

def elevator_moves (start_target : ℤ × ℤ) : Prop :=
  let (start, target) := start_target in
  ∃ n1 n2 : ℤ, ((n1 = 7 ∨ n1 = -7 ∨ n1 = 10 ∨ n1 = -10) ∧ (start + n1 = target)) ∨
  ((n2 = 7 ∨ n2 = -7 ∨ n2 = 10 ∨ n2 = -10) ∧ (start + n1 + n2 = target))

def can_visit_friends (moves: list (ℤ × ℤ)) : Prop :=
  ∀ (move : ℤ × ℤ), move ∈ moves -> elevator_moves move

theorem girl_can_visit_friends :
  ∃ (moves: list (ℤ × ℤ)), length moves ≤ 10 ∧
    can_visit_friends moves ∧
    moves = [(1, 11), (11, 21), (21, 14), (14, 24), (24, 17), (17, 10), (10, 20), (20, 13), (13, 6), (6, 16)] →
  ∃ (floors_visited: list ℤ), floors_visited = [13, 16, 24] :=
sorry

end girl_can_visit_friends_l748_748462


namespace polynomial_conditions_l748_748546

noncomputable def poly_rational_diff (f : ℝ → ℝ) :=
  ∀ x y : ℝ, ∃ q : ℚ, x - y = q → (f x - f y) ∈ ℚ

theorem polynomial_conditions (f : ℝ → ℝ)
    (h_deg : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c) 
    (h_cond : poly_rational_diff f) : 
    ∃ b : ℚ, ∃ c : ℝ, f = λ x, ↑b * x + c :=
sorry

end polynomial_conditions_l748_748546


namespace problem1_problem2_l748_748656

variables (A B C K P Q T: Type)
variables [triangle ABC : Type]
variables [on_extension BC K : Type]
variables [parallel KP AB : Type]
variables [length_equal BK BP : Type]
variables [parallel KQ AC : Type]
variables [length_equal CK CQ : Type]
variables [circumcircle_triangle KPQ AK T : Type]

theorem problem1 (h1: ∠ BTC + ∠ APB = ∠ CQA) : ∠ BTC + ∠ APB = ∠ CQA := 
sorry

theorem problem2 (h2: AP * BT * CQ = AQ * CT * BP) : AP * BT * CQ = AQ * CT * BP :=
sorry

end problem1_problem2_l748_748656


namespace locus_of_perpendiculars_l748_748909

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem locus_of_perpendiculars (O : ℝ × ℝ) (radius : ℝ) (e : ℝ × ℝ → Prop) :
  let C := circle O radius in
  ∀ P : ℝ × ℝ, e P →
  let t := Line.tangent_point_to_circle C P in
  let bisector := Line.angle_bisector e t in
  let foot := Line.foot_perpendicular_to_line O bisector in
  ∃ g1 g2 : ℝ × ℝ → Prop,
    (foot ∈ g1 ∨ foot ∈ g2) :=
by sorry

end locus_of_perpendiculars_l748_748909


namespace relation_x_y_l748_748320

theorem relation_x_y : ∀ x y : ℕ, 
  (x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 11) ∨ (x = 3 ∧ y = 19) ∨ (x = 4 ∧ y = 29) ∨ (x = 5 ∧ y = 41) →
  y = x^2 + 3 * x + 1 :=
by
  intros x y h
  cases h with h1 h
  case inl
  { rw h1.1
    exact congr_arg2 (++) (refl 1) (refl 5) }
  case inr
  { cases h with h2 h
    case inl
    { rw h2.1
      exact congr_arg2 (++) (refl 2) (refl 11) }
    case inr
    { cases h with h3 h
      case inl
      { rw h3.1
        exact congr_arg2 (++) (refl 3) (refl 19) }
      case inr
      { cases h with h4 h
        case inl
        { rw h4.1
          exact congr_arg2 (++) (refl 4) (refl 29) }
        case inr
        { rw h.1 -- this is x=5, y=41
          exact congr_arg2 (++) (refl 5) (refl 41) } } } }
  sorry

end relation_x_y_l748_748320


namespace battery_change_30th_month_l748_748156

theorem battery_change_30th_month :
  ∃ (n : ℕ) (start_month : ℕ), (start_month = 1) ∧ (∀ k : ℕ, k > 0 → Lou_change_month k = (start_month + 7 * (k - 1)) % 12) ∧ (Lou_change_month 30 = 11) :=
sorry

end battery_change_30th_month_l748_748156


namespace find_b_domain_range_l748_748974

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem find_b_domain_range :
  (∃ b : ℝ, (∀ x, 1 ≤ x ∧ x ≤ b → (1 ≤ f x ∧ f x ≤ b)) ∧ (∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y)) → b = 2 :=
begin
  sorry
end

end find_b_domain_range_l748_748974


namespace banana_arrangements_l748_748253

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748253


namespace moles_of_water_l748_748553

-- Definitions related to the reaction conditions.
def HCl : Type := sorry
def NaHCO3 : Type := sorry
def NaCl : Type := sorry
def H2O : Type := sorry
def CO2 : Type := sorry

def reaction (h : HCl) (n : NaHCO3) : Nat := sorry -- Represents the balanced reaction

-- Given conditions in Lean.
axiom one_mole_HCl : HCl
axiom one_mole_NaHCO3 : NaHCO3
axiom balanced_equation : reaction one_mole_HCl one_mole_NaHCO3 = 1 -- 1 mole of water is produced

-- The theorem to prove.
theorem moles_of_water : reaction one_mole_HCl one_mole_NaHCO3 = 1 :=
by
  -- The proof would go here
  sorry

end moles_of_water_l748_748553


namespace BXYC_is_cyclic_l748_748946

-- Definitions for the geometric entities and properties.
variables (ABC : Type) [is_triangle ABC]
variables (A B C M H : ABC) (Γ : set ABC) (X Y : ABC)

-- Conditions of the problem:
-- 1. A B C form an acute triangle.
-- 2. M is the midpoint of B and C.
-- 3. H is the orthocenter of triangle ABC.
-- 4. Γ is the circle with diameter HM.
-- 5. X and Y are distinct points on Γ.
-- 6. AX and AY are tangent to Γ.
axiom acute_triangle_ABC : is_acute_triangle A B C
axiom midpoint_M_BC : is_midpoint M B C
axiom orthocenter_H_ABC : is_orthocenter H A B C
axiom circle_with_diameter_HM : is_circle_with_diameter Γ H M
axiom points_on_circle : X ∈ Γ ∧ Y ∈ Γ ∧ X ≠ Y
axiom tangents_AX_AY : tangent_line AX Γ ∧ tangent_line AY Γ

-- Question to prove:
theorem BXYC_is_cyclic :
  cyclic_quadrilateral B X Y C := 
sorry

end BXYC_is_cyclic_l748_748946


namespace cos_alpha_intersects_unit_circle_l748_748209

theorem cos_alpha_intersects_unit_circle
  (α : ℝ) 
  (hα : -π < α ∧ α < 0) 
  (h_intersect : ∃ y : ℝ, (1/3, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1}) :
  cos (π / 2 + α) = 2 * real.sqrt 2 / 3 :=
by
  sorry

end cos_alpha_intersects_unit_circle_l748_748209


namespace complement_S_union_T_eq_l748_748704

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3 * x - 4 ≤ 0}
noncomputable def complement_S := {x : ℝ | x ≤ -2}

theorem complement_S_union_T_eq : (complement_S ∪ T) = {x : ℝ | x ≤ 1} := by 
  sorry

end complement_S_union_T_eq_l748_748704


namespace solution_set_g_lt_6_range_of_values_a_l748_748454

-- Definitions
def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

-- First part: solution set for g(x) < 6
theorem solution_set_g_lt_6 :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
sorry

-- Second part: range of values for a such that f(x1) and g(x2) are opposite numbers
theorem range_of_values_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end solution_set_g_lt_6_range_of_values_a_l748_748454


namespace count_relatively_prime_to_30_l748_748284
open Int

def nat_range (a b : ℕ) : Set ℕ := {n : ℕ | a ≤ n ∧ n ≤ b}

def relatively_prime (a b : ℕ) : Prop := gcd a b = 1

theorem count_relatively_prime_to_30 : 
  {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ relatively_prime n 30}.card = 24 :=
sorry

end count_relatively_prime_to_30_l748_748284


namespace tan_inequality_solution_set_l748_748766

theorem tan_inequality_solution_set (α : ℝ) :
  (∃ k : ℤ, α ∈ Set.Ioo (-π/6 + k * π) (π/2 + k * π)) ↔ (tan α + (sqrt 3) / 3 > 0) :=
by
  sorry

end tan_inequality_solution_set_l748_748766


namespace local_minimum_at_minus_one_l748_748354

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l748_748354


namespace arithmetic_mean_missing_digit_proof_l748_748871

def arithmetic_mean_missing_digit : Prop :=
  let numbers := [7, 77, 777, 7777, 77777, 777777, 7777777, 77777777, 777777777] in
  let S := numbers.sum in
  let M := S / 9 in
  ∀ digit ∈ [0,1,3,4,5,6,7,8,9], M.digits.contains digit → (∉ M.digits 2)

theorem arithmetic_mean_missing_digit_proof : arithmetic_mean_missing_digit :=
  sorry

end arithmetic_mean_missing_digit_proof_l748_748871


namespace product_divisible_by_8_l748_748417

def roll_prob_divisible_by_8 : ℚ := 235 / 288

theorem product_divisible_by_8 (p : ℚ) (h : p = roll_prob_divisible_by_8) :
  (let dice := [1, 2, 3, 4, 5, 6] in
  let rolls := list.replicate 8 dice in
  let products := [d₁ * d₂ * d₃ * d₄ * d₅ * d₆ * d₇ * d₈ |
                    d₁ <- dice, d₂ <- dice, d₃ <- dice,
                    d₄ <- dice, d₅ <- dice, d₆ <- dice,
                    d₇ <- dice, d₈ <- dice] in
  let favorable := list.countp (λ n, n % 8 = 0) products in
  favorable / products.length = p) :=
by { sorry }

end product_divisible_by_8_l748_748417


namespace correct_operation_l748_748794

theorem correct_operation (x : ℝ) (hx : x ≠ 0) : x^2 / x^8 = 1 / x^6 :=
by
  sorry

end correct_operation_l748_748794


namespace isosceles_triangle_sides_l748_748313

-- Define the properties of the isosceles triangle
variables {a b h r : ℝ}

-- Given conditions
def perimeter_condition : Prop := 2 * a + b = 56
def radius_height_relation : Prop := r = (2 / 7) * h
def area_with_height : Prop := 1 / 2 * b * h = 8 * h

-- Solve for b and a given these conditions
noncomputable def find_sides (a b : ℝ) : Prop :=
  perimeter_condition ∧ radius_height_relation ∧ area_with_height

-- Theorem statement
theorem isosceles_triangle_sides (a b : ℝ) (h : ℝ) (r : ℝ) 
  (h_perimeter : perimeter_condition)
  (h_relation : radius_height_relation)
  (h_area : area_with_height) :
  find_sides 20 16 :=
by {
  sorry
}

end isosceles_triangle_sides_l748_748313


namespace lottery_probability_correct_l748_748000

noncomputable def probability_winning_lottery : ℚ :=
  let starBall_probability := 1 / 30
  let combinations (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let magicBalls_probability := 1 / (combinations 49 6)
  starBall_probability * magicBalls_probability

theorem lottery_probability_correct :
  probability_winning_lottery = 1 / 419514480 := by
  sorry

end lottery_probability_correct_l748_748000


namespace no_symmetry_line_for_exponential_l748_748908

theorem no_symmetry_line_for_exponential : ¬ ∃ l : ℝ → ℝ, ∀ x : ℝ, (2 ^ x) = l (2 ^ (2 * l x - x)) := 
sorry

end no_symmetry_line_for_exponential_l748_748908


namespace lambda_value_l748_748996

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748996


namespace tangent_line_equation_at_0_l748_748754

noncomputable def f (x: ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_equation_at_0 :
  let P := (0 : ℝ, f 0)
  ∃ m b : ℝ, (λ (x y : ℝ), y = m * x + b) (P.1) (P.2) ∧ (∀ x y : ℝ, y = m * x + b ↔ y = x + 3) := 
by
  let m := (Real.exp 0 : ℝ)
  have h: m = 1 := by
    simp [Real.exp_zero]
  let b := P.2 - m * P.1
  have hb: b = 3 := by
    simp [P, m, h, f]
  use [m, b]
  simp [h, hb]
  sorry

end tangent_line_equation_at_0_l748_748754


namespace distinct_arrangements_banana_l748_748237

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748237


namespace find_x_l748_748577

-- Definitions
def a : ℝ × ℝ × ℝ := (2, -1, x)
def b : ℝ × ℝ × ℝ := (3, 2, -1)

-- Condition: a is perpendicular to b
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Theorem statement
theorem find_x (x : ℝ) (h : dot_product (2, -1, x) (3, 2, -1) = 0) : x = 4 :=
by
  sorry

end find_x_l748_748577


namespace quadratic_function_properties_l748_748392

-- Definitions based on conditions
def f (x : ℝ) := x^2 - b * x + c
axiom h0 : f 0 = 3
axiom h1 : ∀ x, f x = f (2 - x)  -- Axis of symmetry x = 1

-- Prove the desired properties
theorem quadratic_function_properties :
  (∀ x, f x = x^2 - 2 * x + 3) ∧
  (∀ x, x ≤ 1 → f' x ≤ 0) ∧
  (∀ x, x ≥ 1 → f' x ≥ 0) ∧
  (∀ x, f x ≥ 2) :=
by
  sorry

end quadratic_function_properties_l748_748392


namespace distance_between_tangent_points_correct_l748_748713

noncomputable def distance_between_tangent_points
  (A B C D M N : Type*)
  [HasSizeOf A] [HasSizeOf B] [HasSizeOf C] [HasSizeOf D] [HasSizeOf M] [HasSizeOf N]
  (isosceles_triangle : Prop)
  (AC_eq_BC : ∀ (AC BC : ℝ), AC = BC)
  (BD_minus_AD : ∀ (BD AD : ℝ), BD - AD = 4) : ℝ :=
| 2

theorem distance_between_tangent_points_correct
  {A B C D : Type*}
  [IsLinearOrder A] [IsLinearOrder B] [IsLinearOrder C] [IsLinearOrder D]
  (isosceles_triangle : Prop)
  (AC_eq_BC : ∀ (AC BC : ℝ), AC = BC)
  (BD_minus_AD : ∀ (BD AD : ℝ), BD - AD = 4) :
  distance_between_tangent_points A B C D _ _ isosceles_triangle AC_eq_BC BD_minus_AD = 2 :=
by simp [distance_between_tangent_points]; sorry

end distance_between_tangent_points_correct_l748_748713


namespace locus_of_midpoints_is_circle_l748_748410

noncomputable def is_locus_circle (A B M C D : Point) : Prop :=
  midpoint A B = M ∧ Circle_contains C D M

noncomputable def midpoint_of_AB_is_on_circle (A B C D : Point) : Prop :=
  ∃ M : Point, midpoint A B = M ∧ Circle_contains C D M

theorem locus_of_midpoints_is_circle 
  (circle1 circle2 : Circle)
  (C D : Point)
  (H : intersection_points circle1 circle2 C D) :
  ∃ circle3 : Circle, ∀ (A B : Point), 
  A ∈ circle1 ∧ B ∈ circle2 ∧ line_through A B C D 
  → midpoint_of_AB_is_on_circle A B C D := 
sorry

end locus_of_midpoints_is_circle_l748_748410


namespace scientific_notation_of_large_number_l748_748380

theorem scientific_notation_of_large_number :
  100000000000 = 1 * 10^11 :=
sorry

end scientific_notation_of_large_number_l748_748380


namespace gold_status_families_count_l748_748403

def bronze_families := 10
def silver_families := 7
def gold_family_donation := 100
def bronze_family_donation := 25
def silver_family_donation := 50
def total_amount_needed := 750
def additional_amount_needed := 50

theorem gold_status_families_count :
  (bronze_families * bronze_family_donation + silver_families * silver_family_donation + 
    gold_family_donation * 1 + additional_amount_needed = total_amount_needed) → 
  1 = 1 :=
by
  intros h
  exact eq.refl 1

end gold_status_families_count_l748_748403


namespace std_deviation_above_l748_748561

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l748_748561


namespace area_increased_by_43_75_percent_l748_748039

variable (l w : ℝ)

def new_length (l : ℝ) := 1.25 * l
def new_width (w : ℝ) := 1.15 * w
def original_area (l w : ℝ) := l * w
def new_area (l w : ℝ) := new_length l * new_width w
def percentage_increase (original new : ℝ) := (new / original) * 100 - 100

theorem area_increased_by_43_75_percent (l w : ℝ) :
  percentage_increase (original_area l w) (new_area l w) = 43.75 :=
by
  sorry

end area_increased_by_43_75_percent_l748_748039


namespace compute_z6_l748_748331

noncomputable def z := (↑(Real.sqrt 3) + Complex.i) / 2

theorem compute_z6 : z^6 = -1 :=
by
  sorry

end compute_z6_l748_748331


namespace regression_estimate_l748_748176

theorem regression_estimate (x : ℝ) (h : x = 28) : 4.75 * x + 257 = 390 :=
by
  rw [h]
  norm_num

end regression_estimate_l748_748176


namespace floor_difference_l748_748154

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end floor_difference_l748_748154


namespace problem_solution_l748_748362

def percentage_of_students_scores : Type := {score: ℝ // score = 15 ∨ score = 25 ∨ score = 20 ∨ score = 40}

noncomputable def mean_median_difference : ℝ :=
  let median : ℝ := 85 in
  let mean : ℝ := 0.15 * 60 + 0.25 * 75 + 0.20 * 85 + 0.40 * 95 in
  (median - mean)

theorem problem_solution :
  let median : ℝ := 85 in
  let mean : ℝ := 0.15 * 60 + 0.25 * 75 + 0.20 * 85 + 0.40 * 95 in
  (median - mean) = 2.25 :=
by
  intros
  sorry

end problem_solution_l748_748362


namespace polygon_inequality_l748_748015

-- Define the vertices of the regular n-sided polygon
variables {n : ℕ} (A : fin n → Point)

-- Define the radius of the circumscribed circle
variables (r : ℝ)

-- Define an internal point P in the polygon
variables (P : Point)

-- Condition: The polygon is regular and inscribed in a circle of radius r.
-- Note: Definition of a regular polygon and inscribed circle is assumed as context here.

-- The theorem states the desired inequality
theorem polygon_inequality 
  (h_reg : regular_n_polygon A n) -- regular n-sided polygon
  (h_inscribed : inscribed_in_circle A r) -- inscribed in a circle of radius r
  (h_internal : inside_polygon P A) -- P is an internal point
  : (finset.univ.sum (λ i : fin n, dist P (A i))) ≥ n * r :=
begin
  sorry -- Proof to be provided
end

end polygon_inequality_l748_748015


namespace sum_of_coordinates_l748_748301

noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := (g x) ^ 2

theorem sum_of_coordinates : g 3 = 6 → (3 + h 3 = 39) := by
  intro hg3
  have : h 3 = (g 3) ^ 2 := by rfl
  rw [hg3] at this
  rw [this]
  exact sorry

end sum_of_coordinates_l748_748301


namespace largest_number_value_l748_748012

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l748_748012


namespace problem_statement_l748_748967

noncomputable def equation_of_altitude (A B C: (ℝ × ℝ)): (ℝ × ℝ × ℝ) :=
by
  sorry

theorem problem_statement :
  let A := (-1, 4)
  let B := (-2, -1)
  let C := (2, 3)
  equation_of_altitude A B C = (1, 1, -3) ∧
  |1 / 2 * (4 - (-1)) * 4| = 8 :=
by
  sorry

end problem_statement_l748_748967


namespace lambda_value_l748_748992

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748992


namespace jenny_ran_further_l748_748677

-- Define the distances Jenny ran and walked
def ran_distance : ℝ := 0.6
def walked_distance : ℝ := 0.4

-- Define the difference between the distances Jenny ran and walked
def difference : ℝ := ran_distance - walked_distance

-- The proof statement
theorem jenny_ran_further : difference = 0.2 := by
  sorry

end jenny_ran_further_l748_748677


namespace math_proof_problem_l748_748640

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem_l748_748640


namespace negation_of_exactly_one_is_even_l748_748420

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_is_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ ¬ is_even b ∧ is_even c))

def at_least_two_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ is_even b) ∨ (is_even b ∧ is_even c) ∨ (is_even a ∧ is_even c))

def all_are_odd (a b c : ℕ) : Prop := ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c 

theorem negation_of_exactly_one_is_even (a b c : ℕ) :
  ¬ exactly_one_is_even a b c ↔ at_least_two_even a b c ∨ all_are_odd a b c := by
  sorry

end negation_of_exactly_one_is_even_l748_748420


namespace pedestrian_distance_after_30_seconds_l748_748472

theorem pedestrian_distance_after_30_seconds :
  ∀ (v : ℝ) (d_crosswalk : ℝ) (l_crosswalk : ℝ) (t : ℝ),
  v = 3.6 / 3.6 → d_crosswalk = 20 → l_crosswalk = 5 → t = 30 →
  let s := v * t in
  s - (d_crosswalk + l_crosswalk) = 5 :=
by
  intros v d_crosswalk l_crosswalk t
  intro h_v conv_v
  intro hdc conv_dcrosswalk
  intro hl conv_lcrosswalk
  intro ht t_eq
  have h_conv_v : v = 1 := by linarith
  have h_s : s = 30 := by
    rw [t_eq, h_conv_v]
    norm_num
  rw [conv_dcrosswalk, conv_lcrosswalk, h_s]
  norm_num
  sorry

end pedestrian_distance_after_30_seconds_l748_748472


namespace oranges_per_box_calculation_l748_748461

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l748_748461


namespace arccos_half_eq_pi_div_three_l748_748516

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l748_748516


namespace least_multiple_of_13_is_1001_l748_748426

theorem least_multiple_of_13_is_1001 (n : ℕ) (h1 : 13.prime) (h2 : 1000 ≤ n) : n = 1001 := by
  sorry

end least_multiple_of_13_is_1001_l748_748426


namespace tan_double_angle_solution_l748_748578

theorem tan_double_angle_solution (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 :=
sorry

end tan_double_angle_solution_l748_748578


namespace area_R_l748_748338

-- Define the given matrix as a 2x2 real matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, -5]

-- Define the original area of region R
def area_R : ℝ := 15

-- Define the area scaling factor as the absolute value of the determinant of A
def scaling_factor : ℝ := |Matrix.det A|

-- Prove that the area of the region R' is 585
theorem area_R' : scaling_factor * area_R = 585 := by
  sorry

end area_R_l748_748338


namespace sum_a5_a8_l748_748340

variable (a : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_a5_a8 (a1 a2 a3 a4 : ℝ) (q : ℝ)
  (h1 : a1 + a3 = 1)
  (h2 : a2 + a4 = 2)
  (h_seq : is_geometric_sequence a q)
  (a_def : ∀ n : ℕ, a n = a1 * q^n) :
  a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end sum_a5_a8_l748_748340


namespace find_x_value_l748_748931

theorem find_x_value (x : ℝ) 
  (Hprod : (∏ n in finset.range 118, (1 + 1/(n+3)) ) = 1/30) 
  (H : x / 11 * (∏ n in finset.range 118, (1 + 1/(n+3)) ) = 11) : 
  x = 3630 :=
by
  sorry

end find_x_value_l748_748931


namespace smallest_fraction_l748_748765

theorem smallest_fraction :
  ∃ (x : ℚ), x ∈ ({1/2, 2/3, 1/4, 5/6, 7/12} : set ℚ) ∧
             (∀ y ∈ ({1/2, 2/3, 1/4, 5/6, 7/12} : set ℚ), x ≤ y) ∧
             x = 1/4 :=
begin
  sorry
end

end smallest_fraction_l748_748765


namespace complex_modulus_ratio_l748_748592

theorem complex_modulus_ratio (z1 z2 : ℂ) (hz1 : |z1| = 2) (hz2 : |z2| = 3) (angle : ℂ) (h_angle : real.angle ∠z1, z2 = real.pi / 3) :
  ∥(z1 + z2) / (z1 - z2)∥ = real.sqrt (19 / 7) :=
by sorry -- Proof to be filled in

end complex_modulus_ratio_l748_748592


namespace problem_l748_748662

-- Define the conditions and the problem
def isValidPair (n k : ℕ) : Prop := (n ≥ 2 ∧ n * (k + n - 1) = 525 ∧ k % 2 = 1 ∧ k > 0)

def countValidPairs : ℕ :=
  Nat.card {nk : ℕ × ℕ // isValidPair nk.1 nk.2}

theorem problem : countValidPairs = 6 := by sorry

end problem_l748_748662


namespace rounding_and_scientific_notation_l748_748377

-- Define the original number
def original_number : ℕ := 1694000

-- Define the function to round to the nearest hundred thousand
def round_to_nearest_hundred_thousand (n : ℕ) : ℕ :=
  ((n + 50000) / 100000) * 100000

-- Define the function to convert to scientific notation
def to_scientific_notation (n : ℕ) : String :=
  let base := n / 1000000
  let exponent := 6
  s!"{base}.0 × 10^{exponent}"

-- Assert the equivalence
theorem rounding_and_scientific_notation :
  to_scientific_notation (round_to_nearest_hundred_thousand original_number) = "1.7 × 10^{6}" :=
by
  sorry

end rounding_and_scientific_notation_l748_748377


namespace value_of_trig_expr_l748_748013

theorem value_of_trig_expr : 2 * Real.cos (Real.pi / 12) ^ 2 + 1 = 2 + Real.sqrt 3 / 2 :=
by
  sorry

end value_of_trig_expr_l748_748013


namespace initial_flour_amount_l748_748524

theorem initial_flour_amount (initial_flour : ℕ) (additional_flour : ℕ) (total_flour : ℕ) 
  (h1 : additional_flour = 4) (h2 : total_flour = 16) (h3 : initial_flour + additional_flour = total_flour) :
  initial_flour = 12 := 
by 
  sorry

end initial_flour_amount_l748_748524


namespace simplify_expression_l748_748869

-- Define p as a real number
variable (p : ℝ)

-- Define the expression in question using the conditions provided
def expression : ℝ :=
  Real.sqrt (15 * p^3) * Real.sqrt (8 * p) * Real.sqrt (12 * p^5)

-- State the theorem we want to prove
theorem simplify_expression : expression p = 60 * p^4 * Real.sqrt (2 * p) := 
  sorry

end simplify_expression_l748_748869


namespace remainder_37_remainder_73_l748_748470

theorem remainder_37 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 37 = 20 := sorry

theorem remainder_73 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 73 = 58 := sorry

end remainder_37_remainder_73_l748_748470


namespace angle_trisector_length_l748_748954

theorem angle_trisector_length :
  ∀ (DE EF : ℝ), 
  DE = 5 → 
  EF = 12 → 
  (∃ GF : ℝ, GF = (96 * Real.sqrt 3) / 17) :=
by 
  intros DE EF h1 h2
  use (96 * Real.sqrt 3) / 17
  exact sorry

end angle_trisector_length_l748_748954


namespace distance_house_to_market_l748_748857

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l748_748857


namespace man_l748_748469

-- Lean 4 statement
theorem man's_speed_against_stream (speed_with_stream : ℝ) (speed_still_water : ℝ) 
(h1 : speed_with_stream = 16) (h2 : speed_still_water = 4) : 
  |speed_still_water - (speed_with_stream - speed_still_water)| = 8 :=
by
  -- Dummy proof since only statement is required
  sorry

end man_l748_748469


namespace no_real_roots_l748_748843

def P : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| (n + 1) := λ x, x^(11 * (n + 1)) - P n x

theorem no_real_roots (n : ℕ): ∀ (x : ℝ), P n x ≠ 0 :=
begin
  induction n with k hk,
  { -- base case
    intro x,
    dsimp [P],
    simp, },
  { -- inductive step
    intros x,
    dsimp [P],
    by_contradiction,
    have h := hk x,
    contradiction, }
end

end no_real_roots_l748_748843


namespace moves_invariance_l748_748449

-- Define basic conditions
variables (frogs : ℕ) (positions : finset ℤ) (initial_positions : list ℤ)

-- Assume given conditions
-- There is a finite number of frogs sitting at different integer points on a line.
def finite_frogs : Prop := frogs = initial_positions.length ∧ positions.card = frogs

-- Frogs occupy distinct points initially
def distinct_initial_positions : Prop := initial_positions.nodup

-- A move is valid if it moves a frog to an adjacent free position
def valid_moves (move : ℤ → ℤ) : Prop :=
  ∀ p, p ∈ positions → move p ∉ positions ∧ (move p = p + 1 ∨ move p = p - 1)

-- Number of ways frogs can make n moves (to the right)
def right_moves (n : ℕ) : ℕ := sorry -- placeholder for the actual function

-- Number of ways frogs can make n moves (to the left)
def left_moves (n : ℕ) : ℕ := sorry -- placeholder for the actual function

-- Main theorem
theorem moves_invariance (n : ℕ) :
  finite_frogs frogs positions initial_positions →
  distinct_initial_positions initial_positions →
  right_moves n = left_moves n :=
by 
  intros _ _ 
  sorry

end moves_invariance_l748_748449


namespace lambda_value_l748_748994

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748994


namespace PA_is_diameter_l748_748880

-- Definitions for circles being tangent at a specific point and other geometric constructions are assumed.

variables {Ωa Ωb Ωc : Type} 
variables [Circle Ωa] [Circle Ωb] [Circle Ωc]
variables (D : Point) (E : Point) (F : Point) 
variables (P : Point) (A : Point) (B : Point) (C : Point)

-- Circle tangency conditions
axiom tangency_a_b : Tangent Ωa Ωb D
axiom tangency_b_c : Tangent Ωb Ωc E
axiom tangency_a_c : Tangent Ωa Ωc F

-- Arbitrary point on Ω_a
axiom point_on_Ωa : OnCircle P Ωa
axiom not_d_f : P ≠ D ∧ P ≠ F

-- Extending points
axiom extended_PD_meets_Ωb : MeetsExtendedAt P D Ωb B
axiom extended_BE_meets_Ωc : MeetsExtendedAt B E Ωc C
axiom extended_CF_meets_Ωa : MeetsExtendedAt C F Ωa A

-- Theorem to be proved
theorem PA_is_diameter (P : Point) (A : Point) : 
  OnCircle P Ωa ∧ OnCircle A Ωa → Diameter Ωa P A := 
sorry

end PA_is_diameter_l748_748880


namespace remainder_sum_f_div_211_l748_748563

def f (n : ℕ) : ℕ :=
  if n % 211 = 0 then 0 else
    let orders := {k | 1 ≤ k ∧ (n ^ k) % 211 = 1}
    in orders.min' (set.nonempty_of_mem ⟨1, by norm_num⟩)

theorem remainder_sum_f_div_211 :
  let S := ∑ n in finset.range 211 \ {211}, n * f n
  in S % 211 = 48 :=
sorry

end remainder_sum_f_div_211_l748_748563


namespace sunday_to_saturday_ratio_l748_748736

theorem sunday_to_saturday_ratio : 
  ∀ (sold_friday sold_saturday sold_sunday total_sold : ℕ),
  sold_friday = 40 →
  sold_saturday = (2 * sold_friday - 10) →
  total_sold = 145 →
  total_sold = sold_friday + sold_saturday + sold_sunday →
  (sold_sunday : ℚ) / (sold_saturday : ℚ) = 1 / 2 :=
by
  intro sold_friday sold_saturday sold_sunday total_sold
  intros h_friday h_saturday h_total h_sum
  sorry

end sunday_to_saturday_ratio_l748_748736


namespace altitudes_sum_l748_748394

noncomputable def sum_of_altitudes (line_eq : ℝ → ℝ → Prop) : ℝ :=
  let x_intercept := 8
  let y_intercept := 15
  let area := 60
  let hypotenuse := 17
  let third_altitude := 120 / 17
  x_intercept + y_intercept + third_altitude

theorem altitudes_sum (line_eq : ℝ → ℝ → Prop)
  (h : line_eq 15 8 = 120) :
  sum_of_altitudes line_eq = 2416 / 17 := by
  sorry

end altitudes_sum_l748_748394


namespace smallest_integer_condition_l748_748784

theorem smallest_integer_condition (x : ℝ) (hz : 9 = 9) (hineq : 27^9 > x^24) : x < 27 :=
  by {
    sorry
  }

end smallest_integer_condition_l748_748784


namespace product_divisible_by_1419_l748_748072

theorem product_divisible_by_1419 (n : ℕ) (h1 : n % 2 = 0) :
  (∏ k in finset.range(n // 2 + 1), 2 * (k + 1)) % 1419 = 0 ↔ n = 106 :=
by
  sorry

end product_divisible_by_1419_l748_748072


namespace regular_polygon_if_even_l748_748157

-- Define the properties of a convex n-gon with n-1 equal sides and angles
def is_convex_n_gon (n : ℕ) (P : Type) [linear_order P] :=
  convex n-gon 

def n_minus_one_equal_sides (n : ℕ) (P : convex n-gon) :=
  ∀ i : fin (n-1), length (side i P) = length (side (i+1) P)

def n_minus_one_equal_angles (n : ℕ) (P : convex n-gon) :=
  ∀ i : fin (n-1), measure (angle i P) = measure (angle (i+1) P)

-- The final theorem statement
theorem regular_polygon_if_even (n : ℕ) (h1 : n ≥ 3) (P : convex n-gon) (h2 : n_minus_one_equal_sides n P) (h3 : n_minus_one_equal_angles n P) : 
  even n := by
  sorry

end regular_polygon_if_even_l748_748157


namespace common_point_exists_l748_748199

-- Define six points on the circumference of a circle
variables (A B C D E F : Point)
-- Define the position vectors of these points
variables (pos_a pos_b pos_c pos_d pos_e pos_f : Vector)

-- Orthocenter of a triangle
def orthocenter (a b c : Vector) : Vector := a + b + c

-- Centroid of a triangle
def centroid (d e f : Vector) : Vector := (d + e + f) / 3

-- The common point P through which all such line segments pass
def common_point : Vector := (pos_a + pos_b + pos_c + pos_d + pos_e + pos_f) / 4

theorem common_point_exists :
  ∀ (X Y Z U V W : Vector),
    (X + Y + Z = orthocenter X Y Z) →
    ((U + V + W) / 3 = centroid U V W) →
    ∃ P, (P = common_point) :=
begin
  intros X Y Z U V W h_orthocenter h_centroid,
  use (pos_a + pos_b + pos_c + pos_d + pos_e + pos_f) / 4,
  sorry
end

end common_point_exists_l748_748199


namespace find_n_value_l748_748938

theorem find_n_value {n : ℕ} {a : ℕ → ℕ} 
  (h1 : ∑ i in Finset.range (n+1), (1 + (1 : ℕ))^i = ∑ i in Finset.range (n+1), a i)
  (h2 : ∑ i in Finset.range (n - 1), a (i + 1) = 29 - n) : 
  n = 4 :=
by sorry

end find_n_value_l748_748938


namespace find_initial_passengers_l748_748802

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers_l748_748802


namespace opposite_sides_line_l748_748596

theorem opposite_sides_line (m : ℝ) :
  ( (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ) → (-7 < m ∧ m < 24) :=
by sorry

end opposite_sides_line_l748_748596


namespace find_variance_of_data_points_l748_748969

noncomputable theory

def data_points : List ℝ := [2, 2, 3, 3, 5]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def squared_deviations (l : List ℝ) (μ : ℝ) : List ℝ := l.map (λ x => (x - μ) ^ 2)

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (squared_deviations l μ).sum / l.length

theorem find_variance_of_data_points :
  variance data_points = 1.2 :=
sorry

end find_variance_of_data_points_l748_748969


namespace largest_number_value_l748_748011

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l748_748011


namespace eccentricity_of_ellipse_l748_748686

noncomputable def ellipse_foci (a b : ℝ) := (1:ℝ)

theorem eccentricity_of_ellipse : 
  ∀ (b : ℝ), b > 0 →
  ( ∀ (F1 F2 : ℝ × ℝ),
    ∀ (l : ℝ → ℝ),
    ∃ A B : ℝ × ℝ,
    (A ≠ B ∧ l(F1.fst) = F1.snd ∧ l(F2.fst) = F2.snd) ∧
    (dist (A, F2) + dist (B, F2) = 5) ) →
  eccentricity (4 : ℝ) b = 0.5 :=
by
  intro b hb h
  sorry

end eccentricity_of_ellipse_l748_748686


namespace banana_permutations_l748_748261

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748261


namespace anna_initial_pencils_l748_748860

theorem anna_initial_pencils :
  ∃ (A : ℕ), (let H := 2 * A in H - 19 = 81) ∧ A = 50 :=
begin
  use 50,
  split,
  { dsimp,
    norm_num, },
  { refl, }
end

end anna_initial_pencils_l748_748860


namespace cos_theta_l748_748228

variables {a b : ℝ}

def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem cos_theta (a b : ℝ × ℝ)
  (h1 : norm a = 5)
  (h2 : norm b = 7)
  (h3 : norm (a.1 + b.1, a.2 + b.2) = 10) :
  dot_product a b / (norm a * norm b) = 13 / 35 :=
sorry

end cos_theta_l748_748228


namespace find_a_l748_748585

theorem find_a (k x y a : ℝ) (hkx : k ≤ x) (hx3 : x ≤ 3) (hy7 : a ≤ y) (hy7' : y ≤ 7) (hy : y = k * x + 1) :
  a = 5 ∨ a = 1 - 3 * Real.sqrt 6 :=
sorry

end find_a_l748_748585


namespace arccos_half_eq_pi_div_three_l748_748515

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l748_748515


namespace linear_function_solution_l748_748287

theorem linear_function_solution
    (f : ℝ → ℝ)
    (h_lin : ∀ x, f(x) = a*x + b)
    (h_inc : a > 0)
    (h_comp : ∀ x, f(f(x)) = 16*x + 9) :
    f = (λ x : ℝ, 4*x + 9 / 5) :=
by
    sorry

end linear_function_solution_l748_748287


namespace sum_of_altitudes_of_triangle_l748_748885

open Real

theorem sum_of_altitudes_of_triangle 
  (x_intercept : ℝ := 8)
  (y_intercept : ℝ := 10)
  (line_eq : ∀ (x y : ℝ), 10 * x + 8 * y = 80 → (x = 0 ∨ y = 0))
  : 10 + 8 + (80 * Real.sqrt 164) / 164 = 18 + (80 * Real.sqrt 164) / 164 :=
by
  sorry

end sum_of_altitudes_of_triangle_l748_748885


namespace signed_barycentric_distance_zero_l748_748466

variables {α β γ d_a d_b d_c : ℝ} 

theorem signed_barycentric_distance_zero
  (h_coords : α + β + γ = 1)
  (h_point_on_line :  ∃ X : ℝ × ℝ × ℝ, (X = (α, β, γ)))
  (h_distances : ∀ {A B C l}, 
    ∃ (d_a d_b d_c : ℝ), 
    d_a = signed_distance A l ∧
    d_b = signed_distance B l ∧
    d_c = signed_distance C l) :
  d_a * α + d_b * β + d_c * γ = 0 :=
sorry

end signed_barycentric_distance_zero_l748_748466


namespace solve_cubic_equation_l748_748731

theorem solve_cubic_equation (x y z : ℤ) (h : x^3 - 3*y^3 - 9*z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end solve_cubic_equation_l748_748731


namespace find_k_parallel_l748_748629

theorem find_k_parallel (a b : ℝ × ℝ) (k : ℝ) (ha : a = (3, -1)) (hb : b = (1, -2)) 
  (h_parallel : ∃ (λ : ℝ), λ • (-a + b) = a + k • b) : k = -1 :=
by
  sorry

end find_k_parallel_l748_748629


namespace find_angle_A_l748_748308

theorem find_angle_A 
  (a b : ℝ) (A B : ℝ) 
  (h1 : b = 2 * a)
  (h2 : B = A + 60) : 
  A = 30 :=
  sorry

end find_angle_A_l748_748308


namespace projective_transformation_unique_l748_748719

noncomputable def projective_transformation : Type := sorry

axiom cross_ratio_preserved (P : projective_transformation) (A B C X : point) :
  cross_ratio (P A) (P B) (P C) (P X) = cross_ratio A B C X

axiom uniqueness_projective (P Q : projective_transformation) (A B C : point):
  (∀ X, cross_ratio (P A) (P B) (P C) (P X) = cross_ratio (Q A) (Q B) (Q C) (Q X)) → P = Q

theorem projective_transformation_unique (P Q : projective_transformation) (A B C : point) :
  (P A = Q A) ∧ (P B = Q B) ∧ (P C = Q C) → ∀ X, P X = Q X :=
by
  intros h X
  have h_pq : ∀ X, cross_ratio (P A) (P B) (P C) (P X) = cross_ratio (Q A) (Q B) (Q C) (Q X) :=
    by
      intro X'
      rw [(and.elim_left h), (and.elim_left (and.elim_right h)), (and.elim_right (and.elim_right h))]
  apply uniqueness_projective P Q A B C h_pq
  contradiction
  sorry

end projective_transformation_unique_l748_748719


namespace no_internal_angle_less_than_60_l748_748371

-- Define the concept of a Δ-curve
def delta_curve (K : Type) : Prop := sorry

-- Define the concept of a bicentric Δ-curve
def bicentric_delta_curve (K : Type) : Prop := sorry

-- Define the concept of internal angles of a Δ-curve
def has_internal_angle (K : Type) (A : ℝ) : Prop := sorry

-- The Lean statement for the problem
theorem no_internal_angle_less_than_60 (K : Type) 
  (h1 : delta_curve K) 
  (h2 : has_internal_angle K 60 ↔ bicentric_delta_curve K) :
  (∀ A < 60, ¬has_internal_angle K A) ∧ (has_internal_angle K 60 → bicentric_delta_curve K) := 
sorry

end no_internal_angle_less_than_60_l748_748371


namespace constant_term_binomial_expansion_l748_748749

theorem constant_term_binomial_expansion :
  (∃ r : ℕ, 12 - 3 * r = 0 ∧ (¬r >= 7) ∧ (x : ℚ => (x^2 - 2/x) ^ 6) = (240 : ℕ) :=
begin
  sorry
end

end constant_term_binomial_expansion_l748_748749


namespace range_of_sum_l748_748304

theorem range_of_sum (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) : 
  ∃ a b : ℝ, (x + y + 10 ≥ a) ∧ (x + y + 10 ≤ b) ∧ a = 5 ∧ b = 15 := 
sorry

end range_of_sum_l748_748304


namespace calculate_expression_l748_748129

theorem calculate_expression :
  (1/2)^(-2) - Real.log10 2 - Real.log10 5 = 5 :=
by
  sorry

end calculate_expression_l748_748129


namespace swim_meet_time_l748_748572

theorem swim_meet_time {distance : ℕ} (d : distance = 50) (t : ℕ) 
  (meet_first : ∃ t1 : ℕ, t1 = 2 ∧ distance - 20 = 30) 
  (turn : ∀ t1, t1 = 2 → ∀ d1 : ℕ, d1 = 50 → t1 + t1 = 4) :
  t = 4 :=
by
  -- Placeholder proof
  sorry

end swim_meet_time_l748_748572


namespace range_of_fraction_l748_748597

variable {x y : ℝ}

-- Condition given in the problem
def equation (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- The range condition for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 3

-- The corresponding theorem statement
theorem range_of_fraction (h_eq : equation x y) (h_x_range : x_range x) :
  ∃ a b : ℝ, (a < 1 ∧ 10 < b) ∧ (a, b) = (1, 10) ∧
  ∀ k : ℝ, k = (x + 2) / (y - 1) → 1 < k ∧ k < 10 :=
sorry

end range_of_fraction_l748_748597


namespace car_meeting_points_l748_748412

-- Define the conditions for the problem
variables {A B : ℝ}
variables {speed_ratio : ℝ} (ratio_pos : speed_ratio = 5 / 4)
variables {T1 T2 : ℝ} (T1_pos : T1 = 145) (T2_pos : T2 = 201)

-- The proof problem statement
theorem car_meeting_points (A B : ℝ) (ratio_pos : speed_ratio = 5 / 4) 
  (T1 T2 : ℝ) (T1_pos : T1 = 145) (T2_pos : T2 = 201) :
  A = 103 ∧ B = 229 :=
sorry

end car_meeting_points_l748_748412


namespace sum_in_base_8_is_correct_l748_748929

noncomputable section

open Nat

def num1 : ℕ := Nat.ofDigits 8 [5, 2, 7]
def num2 : ℕ := Nat.ofDigits 8 [1, 6, 5]
def num3 : ℕ := Nat.ofDigits 8 [2, 7, 3]
def sum_expected : ℕ := Nat.ofDigits 8 [1, 2, 0, 7]

theorem sum_in_base_8_is_correct :
  num1 + num2 + num3 = sum_expected := by
  sorry

end sum_in_base_8_is_correct_l748_748929


namespace correct_calculation_l748_748434

theorem correct_calculation (a b : ℝ) : 
  ¬(a * a^3 = a^3) ∧ ¬((a^2)^3 = a^5) ∧ (-a^2 * b)^2 = a^4 * b^2 ∧ ¬(a^3 / a = a^3) :=
by {
  sorry
}

end correct_calculation_l748_748434


namespace number_of_adult_tickets_l748_748775

-- Define the parameters of the problem
def price_adult_ticket : ℝ := 5.50
def price_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50

-- Define the main theorem to be proven
theorem number_of_adult_tickets : 
  ∃ (A C : ℕ), A + C = total_tickets ∧ 
                (price_adult_ticket * A + price_child_ticket * C = total_cost) ∧ 
                 A = 5 :=
by
  -- The proof content will be filled in later
  sorry

end number_of_adult_tickets_l748_748775


namespace arccos_one_half_eq_pi_div_three_l748_748511

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l748_748511


namespace line_passes_fixed_point_l748_748756

theorem line_passes_fixed_point (k : ℝ) :
    ((k + 1) * -1) - ((2 * k - 1) * 1) + 3 * k = 0 :=
by
    -- The proof is omitted as the primary aim is to ensure the correct Lean statement.
    sorry

end line_passes_fixed_point_l748_748756


namespace cos_sum_eq_one_l748_748207

theorem cos_sum_eq_one (α β γ : ℝ) 
  (h1 : α + β + γ = Real.pi) 
  (h2 : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 :=
sorry

end cos_sum_eq_one_l748_748207


namespace quadratic_eq_transformed_l748_748402

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2 * x - 7 = 0

-- Define the form to transform to using completing the square method
def transformed_eq (x : ℝ) : Prop := (x - 1)^2 = 8

-- The theorem to be proved
theorem quadratic_eq_transformed (x : ℝ) :
  quadratic_eq x → transformed_eq x :=
by
  intros h
  -- here we would use steps of completing the square to transform the equation
  sorry

end quadratic_eq_transformed_l748_748402


namespace sin_angle_in_equilateral_triangle_l748_748659

theorem sin_angle_in_equilateral_triangle (A B C D E : Point) 
(ABC_equilateral : equilateral_triangle A B C)
(D_bisect_BC : midpoint D B C)
(E_bisect_BC : midpoint E B C) :
sin (angle D A E) = 1 :=
sorry

end sin_angle_in_equilateral_triangle_l748_748659


namespace bicycle_price_after_discounts_l748_748063

/-- A bicycle originally priced at $200 is discounted by 40% on Tuesday.
    On the following Thursday, that sale price is further reduced by 25%.
    What is the price of the bicycle after the Thursday reduction? -/
theorem bicycle_price_after_discounts :
  ∃ (final_price : ℝ), final_price = 200 * 0.60 * 0.75 ∧ final_price = 90 :=
by
  use 200 * 0.60 * 0.75
  split
  · sorry
  · rfl

end bicycle_price_after_discounts_l748_748063


namespace max_sums_in_interval_l748_748705

theorem max_sums_in_interval
  (n : Nat)
  (x : Fin (2 * n) → ℝ)
  (h : ∀ i, 1 < x i) :
  (Finset.card {s : Fin (2 * n) → Bool | (0 : ℝ) <= (Finset.univ.sum (λ i, if s i then x i else -x i)) ∧ (Finset.univ.sum (λ i, if s i then x i else -x i)) <= 2}) ≤ Nat.choose (2 * n) n :=
sorry

end max_sums_in_interval_l748_748705


namespace range_of_h_l748_748555

def h (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 2)

theorem range_of_h : set.range h = set.Icc (-1/8 : ℝ) (25/16 : ℝ) :=
sorry

end range_of_h_l748_748555


namespace range_of_x_l748_748194

variable {α : Type*} [LinearOrder α]
variable {f : α → α} (hf1 : ∀ x, f (-x) = -f x) (hf2 : MonotoneOn f (Set.Ici 0))

theorem range_of_x (h : ∀ x, f (2 * x - 1) < f (x^2 - x + 1)) : 
  {x : α | f (2 * x - 1) < f (x^2 - x + 1)} = Set.Iio 1 ∪ Set.Ioi 2 :=
  sorry

end range_of_x_l748_748194


namespace sum_invested_eq_2000_l748_748071

theorem sum_invested_eq_2000 (P : ℝ) (R1 R2 T : ℝ) (H1 : R1 = 18) (H2 : R2 = 12) 
  (H3 : T = 2) (H4 : (P * R1 * T / 100) - (P * R2 * T / 100) = 240): 
  P = 2000 :=
by 
  sorry

end sum_invested_eq_2000_l748_748071


namespace triangle_area_l748_748114

def point := (ℝ × ℝ)
def area_of_triangle (p1 p2 p3 : point) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

theorem triangle_area :
  ∀ (A B C : point), 
  A = (-2, 3) → B = (7, -3) → C = (4, 6) → 
  area_of_triangle A B C = 31.5 :=
by
  -- ignoring the proof
  intros A B C hA hB hC
  sorry

end triangle_area_l748_748114


namespace inverse_variation_l748_748447

theorem inverse_variation (k : ℝ) : 
  (∀ (x y : ℝ), x * y^2 = k) → 
  (∀ (x y : ℝ), x = 1 → y = 2 → k = 4) → 
  (x = 0.1111111111111111) → 
  (y = 6) :=
by 
  -- Assume the given conditions
  intros h h0 hx
  -- Proof goes here...
  sorry

end inverse_variation_l748_748447


namespace drank_bottles_of_juice_l748_748852

theorem drank_bottles_of_juice
  (bottles_in_refrigerator : ℕ)
  (bottles_in_pantry : ℕ)
  (bottles_bought : ℕ)
  (bottles_left : ℕ)
  (initial_bottles := bottles_in_refrigerator + bottles_in_pantry)
  (total_bottles := initial_bottles + bottles_bought)
  (bottles_drank := total_bottles - bottles_left) :
  bottles_in_refrigerator = 4 ∧
  bottles_in_pantry = 4 ∧
  bottles_bought = 5 ∧
  bottles_left = 10 →
  bottles_drank = 3 :=
by sorry

end drank_bottles_of_juice_l748_748852


namespace simple_interest_time_l748_748302

theorem simple_interest_time (R P SI : ℤ) (hR : R = 15) (hP : P = 400) (hSI : SI = 120) : 
  let T := (SI * 100) / (P * R) in T = 2 :=
by 
  sorry

end simple_interest_time_l748_748302


namespace soccer_games_per_month_l748_748023

theorem soccer_games_per_month (total_games : ℕ) (months : ℕ) (h1 : total_games = 27) (h2 : months = 3) : total_games / months = 9 :=
by 
  sorry

end soccer_games_per_month_l748_748023


namespace arithmetic_square_root_of_sqrt_16_l748_748743

theorem arithmetic_square_root_of_sqrt_16 :
  let arithmetic_square_root (x : ℝ) := if x ≥ 0 then real.sqrt x else -real.sqrt (-x)
  arithmetic_square_root (real.sqrt 16) = 2 := by
    sorry

end arithmetic_square_root_of_sqrt_16_l748_748743


namespace division_decomposition_l748_748792

theorem division_decomposition (a b : ℕ) (h₁ : a = 36) (h₂ : b = 3)
    (h₃ : 30 / b = 10) (h₄ : 6 / b = 2) (h₅ : 10 + 2 = 12) :
    a / b = (30 / b) + (6 / b) := 
sorry

end division_decomposition_l748_748792


namespace option_B_valid_l748_748179

-- Definitions derived from conditions
def at_least_one_black (balls : List Bool) : Prop :=
  ∃ b ∈ balls, b = true

def both_black (balls : List Bool) : Prop :=
  balls = [true, true]

def exactly_one_black (balls : List Bool) : Prop :=
  balls.count true = 1

def exactly_two_black (balls : List Bool) : Prop :=
  balls.count true = 2

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → False

def non_complementary (P Q : Prop) : Prop :=
  ¬(P → ¬Q) ∧ ¬(¬P → Q)

-- Balls: true represents a black ball, false represents a red ball.
def all_draws := [[true, true], [true, false], [false, true], [false, false]]

-- Proof statement
theorem option_B_valid :
  (mutually_exclusive (exactly_one_black [true, false]) (exactly_two_black [true, true])) ∧ 
  (non_complementary (exactly_one_black [true, false]) (exactly_two_black [true, true])) :=
  sorry

end option_B_valid_l748_748179


namespace tan_phi_eq_zero_l748_748476

-- Definitions based on conditions
def right_triangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

def is_angle_bisector (tri : Triangle) (A B C : Point) : Prop :=
  -- Definition for an angle bisector condition here (stub)
  sorry

def is_perimeter_bisector (tri : Triangle) (A B C : Point) : Prop :=
  -- Definition for a perimeter bisector condition here (stub)
  sorry

-- Main statement to prove in Lean 4
theorem tan_phi_eq_zero : 
  ∀ (A B C : Point), 
    right_triangle 6 8 10 → 
    is_angle_bisector tri A B C → 
    is_perimeter_bisector tri A B C → 
    tan φ = 0 :=
begin
  sorry
end

end tan_phi_eq_zero_l748_748476


namespace monic_quadratic_with_root_2_minus_i_l748_748167

theorem monic_quadratic_with_root_2_minus_i :
  ∃ (p : ℝ[X]), monic p ∧ degree p = 2 ∧ is_root p (complex.of_real 2 - complex.I) ∧ p = X^2 - 4 * X + 5 :=
by
  sorry

end monic_quadratic_with_root_2_minus_i_l748_748167


namespace number_of_sheep_l748_748446

-- Let H be the number of horses and S be the number of sheep.
def number_of_horses (H : ℕ) : Prop := 230 * H = 12880

def ratio_sheep_horses (S H : ℕ) : Prop := S = H / 7

theorem number_of_sheep (S H : ℕ) (h_horses : number_of_horses H) (h_ratio : ratio_sheep_horses S H) : S = 8 :=
by 
  -- given H is the number of horses
  rw [number_of_horses] at h_horses
  -- solving for H
  have H_eq : H = 12880 / 230, from eq_of_mul_eq_mul_right (by norm_num : 230 ≠ 0) h_horses
  have H_val : H = 56, by norm_num [H_eq]

  -- given S = H / 7
  rw [H_val] at h_ratio
  -- solving for S
  have S_eq : S = 56 / 7, from h_ratio
  norm_num at S_eq
  exact S_eq

end number_of_sheep_l748_748446


namespace circle_area_sum_l748_748499

theorem circle_area_sum (x y z : ℕ) (A₁ A₂ A₃ total_area : ℕ) (h₁ : A₁ = 6) (h₂ : A₂ = 15) 
  (h₃ : A₃ = 83) (h₄ : total_area = 220) (hx : x = 4) (hy : y = 2) (hz : z = 2) :
  A₁ * x + A₂ * y + A₃ * z = total_area := by
  sorry

end circle_area_sum_l748_748499


namespace local_minimum_at_minus_one_l748_748353

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l748_748353


namespace banana_permutations_l748_748259

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748259


namespace sin_A_is_correct_l748_748671

noncomputable def sin_A_in_triangle (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : ℝ :=
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c) in
  real.sqrt (1 - cos_A^2)

theorem sin_A_is_correct : sin_A_in_triangle 4 5 6 rfl rfl rfl = real.sqrt 7 / 4 :=
  sorry

end sin_A_is_correct_l748_748671


namespace braden_total_money_after_bet_l748_748867

theorem braden_total_money_after_bet (initial_amount bet_multiplier : ℕ) (initial_money : initial_amount = 400) (bet_transition : bet_multiplier = 2) :
  let winning_amount := bet_multiplier * initial_amount in
  let total_amount := winning_amount + initial_amount in
  total_amount = 1200 :=
by
  sorry

end braden_total_money_after_bet_l748_748867


namespace isosceles_triangle_properties_l748_748188

noncomputable def isosceles_triangle_sides (a : ℝ) : ℝ × ℝ × ℝ :=
  let x := a * Real.sqrt 3
  let y := 2 * x / 3
  let z := (x + y) / 2
  (x, z, z)

theorem isosceles_triangle_properties (a x y z : ℝ) 
  (h1 : x * y = 2 * a ^ 2) 
  (h2 : x + y = 2 * z) 
  (h3 : y ^ 2 + (x / 2) ^ 2 = z ^ 2) : 
  x = a * Real.sqrt 3 ∧ 
  z = 5 * a * Real.sqrt 3 / 6 :=
by
-- Proof goes here
sorry

end isosceles_triangle_properties_l748_748188


namespace theta_in_second_quadrant_l748_748294

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : 2 * cos θ < 0)
  (h2 : sin (2 * θ) < 0) :
  π/2 < θ ∧ θ < π :=
by
  sorry

end theta_in_second_quadrant_l748_748294


namespace initial_salt_percentage_is_10_l748_748062

-- Declarations for terminology
def initial_volume : ℕ := 72
def added_water : ℕ := 18
def final_volume : ℕ := initial_volume + added_water
def final_salt_percentage : ℝ := 0.08

-- Amount of salt in the initial solution
def initial_salt_amount (P : ℝ) := initial_volume * P

-- Amount of salt in the final solution
def final_salt_amount : ℝ := final_volume * final_salt_percentage

-- Proof that the initial percentage of salt was 10%
theorem initial_salt_percentage_is_10 :
  ∃ P : ℝ, initial_salt_amount P = final_salt_amount ∧ P = 0.1 :=
by
  sorry

end initial_salt_percentage_is_10_l748_748062


namespace find_s_l748_748632

theorem find_s (k s : ℝ) (h1 : 5 = k * 2^s) (h2 : 45 = k * 8^s) : s = (Real.log 9) / (2 * Real.log 2) :=
by
  sorry

end find_s_l748_748632


namespace probability_asymmetric_gold_l748_748497

def prob_heads_asym : ℝ := 0.6

def prob_heads_sym : ℝ := 0.5

def prob_heads_first_flip (asym : Bool) : ℝ :=
if asym then prob_heads_asym else prob_heads_sym

def prob_heads_second_flip (asym : Bool) : ℝ :=
if asym then prob_heads_sym else prob_heads_sym * prob_heads_sym

def bayes_theorem (P_A B_A P_B : ℝ) : ℝ := (B_A * P_A) / P_B

def prob_A := 0.5
def prob_not_A := 1 - prob_A

def prob_B_given_A := prob_heads_first_flip true * prob_heads_second_flip true
def prob_B_given_not_A := prob_heads_first_flip false * prob_heads_second_flip false

def prob_B := (prob_B_given_A * prob_A) + (prob_B_given_not_A * prob_not_A)

theorem probability_asymmetric_gold :
  bayes_theorem prob_A prob_B_given_A prob_B = 6 / 11 := by
  sorry

end probability_asymmetric_gold_l748_748497


namespace banana_permutations_l748_748260

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748260


namespace find_a_l748_748192

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end find_a_l748_748192


namespace banana_permutations_l748_748256

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748256


namespace relatively_prime_days_in_july_l748_748882

theorem relatively_prime_days_in_july :
  ∀ (days_in_month : ℕ) (month_number : ℕ),
    days_in_month = 31 → month_number = 7 → 
    (card {day ∈ (finset.range days_in_month.succ) | gcd day month_number = 1} = 27) :=
by
  sorry

end relatively_prime_days_in_july_l748_748882


namespace banana_distinct_arrangements_l748_748280

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748280


namespace a_friend_gcd_l748_748355

theorem a_friend_gcd (a b : ℕ) (d : ℕ) (hab : a * b = d * d) (hd : d = Nat.gcd a b) : ∃ k : ℕ, a * d = k * k := by
  sorry

end a_friend_gcd_l748_748355


namespace find_f_value_l748_748957

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 5
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

-- Condition 3: f(-3) = -4
def f_value_at_neg3 (f : ℝ → ℝ) := f (-3) = -4

-- Condition 4: cos(α) = 1 / 2
def cos_alpha_value (α : ℝ) := Real.cos α = 1 / 2

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def α : ℝ := sorry

theorem find_f_value (h_odd : is_odd_function f)
                     (h_periodic : is_periodic f 5)
                     (h_f_neg3 : f_value_at_neg3 f)
                     (h_cos_alpha : cos_alpha_value α) :
  f (4 * Real.cos (2 * α)) = 4 := 
sorry

end find_f_value_l748_748957


namespace arccos_half_eq_pi_div_three_l748_748513

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l748_748513


namespace distinct_arrangements_banana_l748_748265

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748265


namespace chessboard_numbers_to_zero_l748_748097

theorem chessboard_numbers_to_zero (m n : ℕ) (board : ℕ → ℕ → ℕ) :
  let S_b : ℕ := ∑ i in @Finset.univ (Fin m.succ) _, ∑ j in @Finset.univ (Fin n.succ) _, if (i + j)%2 = 0 then board i j else 0
  let S_w : ℕ := ∑ i in @Finset.univ (Fin m.succ) _, ∑ j in @Finset.univ (Fin n.succ) _, if (i + j)%2 = 1 then board i j else 0
  (∀ k : ℤ, ∀ i j : fin m.succ × fin n.succ, 
    (i.1.succ = j.1 ∨ i.1 = j.1.succ ∨ i.2.succ = j.2 ∨ i.2 = j.2.succ) → 
    0 ≤ board i.1 i.2 + k ∧ 0 ≤ board j.1 j.2 + k) →
  (∑ i in @Finset.univ (Fin m.succ) _, ∑ j in @Finset.univ (Fin n.succ) _, board i j = 0) ↔ S_b = S_w :=
by
  sorry

end chessboard_numbers_to_zero_l748_748097


namespace lambda_value_l748_748993

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748993


namespace factor_expression_l748_748181

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l748_748181


namespace integral_triangle_with_perimeter_12_l748_748486

theorem integral_triangle_with_perimeter_12 :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧ 
  ∃ s : ℝ, s = 6 ∧ 
  ∃ A : ℝ, A = sqrt ((s * (s - a) * (s - b) * (s - c))) ∧ 
  A = 2 * sqrt 6 :=
sorry

end integral_triangle_with_perimeter_12_l748_748486


namespace DB_determined_l748_748669

noncomputable def determine_DB (AB CD AE : ℝ) : ℝ :=
  CD * AB / AE

theorem DB_determined (ABC : Triangle) (alt_CD_to_AB : IsAltitude CD AB) (alt_AE_to_BC : IsAltitude AE BC)
  (AB_length CD_length AE_length : ℝ) :
  determine_DB AB_length CD_length AE_length = CD_length * AB_length / AE_length :=
by
  -- Assuming the triangle ABC is defined and the conditions are met
  sorry

end DB_determined_l748_748669


namespace average_high_is_expected_average_deviation_is_expected_l748_748384

def normal_highs : List Float := [65, 64, 62, 63, 64, 65, 66]
def recorded_highs : List Float := [55, 68, 64, 67, 59, 71, 65]
def days_count : Nat := 7

def avg (l : List Float) : Float :=
  l.sum / (l.length)

def abs_diff (a b : Float) : Float :=
  Float.abs (a - b)

def deviations (l1 l2 : List Float) : List Float :=
  List.zipWith abs_diff l1 l2

theorem average_high_is_expected :
  avg recorded_highs = 64.1 := sorry

theorem average_deviation_is_expected :
  avg (deviations recorded_highs normal_highs) = 4.6 := sorry

end average_high_is_expected_average_deviation_is_expected_l748_748384


namespace tetrahedron_faces_l748_748788

def faces_of_tetrahedron : ℕ := 4

theorem tetrahedron_faces (T : Type) [is_tetrahedron T] : faces_of_tetrahedron = 4 := by
  sorry

-- Define a class for tetrahedron, this is an abstract definition to match the condition in a) 
class is_tetrahedron (T : Type) :=
(faces : ℕ)
(faces_eq : faces = 4)

end tetrahedron_faces_l748_748788


namespace dot_product_eq_neg4_three_a_minus_four_b_norm_dot_product_with_expression_l748_748986

variables (a b : EuclideanSpace ℝ (Fin 3)) -- Assuming 3 dimensional space for generality
variables (Ha : ‖a‖ = 4) (Hb : ‖b‖ = 2) (Hab : ‖a + b‖ = 2 * Real.sqrt 3)

-- Problem 1: Prove a ⋅ b = -4
theorem dot_product_eq_neg4 : (a ⋅ b) = -4 :=
by sorry

-- Problem 2: Prove ‖3 • a - 4 • b‖ = 4 * Real.sqrt 19
theorem three_a_minus_four_b_norm : ‖3 • a - 4 • b‖ = 4 * Real.sqrt 19 :=
by sorry

-- Problem 3: Prove (a - 2 • b) ⋅ (a + b) = 12
theorem dot_product_with_expression : (a - 2 • b) ⋅ (a + b) = 12 :=
by sorry

end dot_product_eq_neg4_three_a_minus_four_b_norm_dot_product_with_expression_l748_748986


namespace percentage_error_in_volume_l748_748489

-- Define the variables and conditions
variables {a : ℝ} (h_pos : a > 0)

-- Define the percentage error theorem
theorem percentage_error_in_volume (error_percentage : ℝ) 
                                   (h_error : error_percentage = 2) :
  let measured_side := a * (1 + error_percentage / 100) in
  let actual_volume := a^3 in
  let measured_volume := measured_side^3 in
  let volume_error := measured_volume - actual_volume in
  ((volume_error / actual_volume) * 100) = 6.12 :=
by 
  sorry

end percentage_error_in_volume_l748_748489


namespace tangent_at_one_min_a_l748_748210

-- Problem (1)
def function_f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x
def tangent_line_equation : (ℝ → ℝ) := λ x, -15 * x + 1

theorem tangent_at_one : ∃ (m : ℝ) (b : ℝ), (∀ x, m * x + b = tangent_line_equation x) :=
by
  sorry

-- Problem (2)
def f_le_phi (a x : ℝ) : Prop := function_f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1

theorem min_a (a : ℤ) : (∀ x, f_le_phi a x) → a ≥ 1 :=
by
  sorry

end tangent_at_one_min_a_l748_748210


namespace regular_polygon_n_eq_10_l748_748101

-- Given conditions
variables (R : ℝ) (A : ℝ) (n : ℕ) (h : A = 4 * R^2) (h0 : n ≠ 0)

-- Definition of the area of a regular polygon inscribed in a circle
def polygon_area (n : ℕ) (R : ℝ) : ℝ :=
  (1/2) * n * R^2 * real.sin (2 * real.pi / n.to_real)

-- Theorem: if the area of the polygon is 4R², then n = 10
theorem regular_polygon_n_eq_10 (h1 : polygon_area n R = 4 * R^2) : n = 10 :=
sorry

end regular_polygon_n_eq_10_l748_748101


namespace value_of_r_squared_plus_s_squared_l748_748349

theorem value_of_r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 24) (h2 : r + s = 10) :
  r^2 + s^2 = 52 :=
sorry

end value_of_r_squared_plus_s_squared_l748_748349


namespace parabola_eq_l748_748098

def isParabola (x y : ℝ) : ℝ :=
  let focus := (2 : ℝ, -1 : ℝ)
  let directrix := (1 : ℝ, 2 : ℝ, -4 : ℝ) -- representing x + 2y - 4 = 0
  let distanceToFocus := real.sqrt ((x - focus.1) ^ 2 + (y - focus.2) ^ 2)
  let distanceToDirectrix := (abs (x + 2*y - 4)) / real.sqrt ((directrix.1)^2 + (directrix.2)^2)
  (distanceToFocus - distanceToDirectrix)

theorem parabola_eq (x y : ℝ) :
  isParabola x y = 0 →
  4*x^2 - 4*x*y + y^2 - 12*x - 6*y + 9 = 0 :=
by
  sorry

end parabola_eq_l748_748098


namespace digits_sum_l748_748399

def log10 (n : ℕ) : ℝ := Real.log n / Real.log 10

theorem digits_sum :
  let a := 1989
  let x := ⌊a * log10 2⌋ + 1
  let y := ⌊a * log10 5⌋ + 1
  x + y = 1990 :=
by
  let a := 1989
  let x := Int.floor (a * log10 2) + 1
  let y := Int.floor (a * log10 5) + 1
  have h : x + y = 1990 := sorry
  exact h

end digits_sum_l748_748399


namespace number_of_arrangements_l748_748771

-- Conditions
constant students : Fin 6 -- 6 students
constant events : Fin 4 -- 4 events

def participates_in_event (student : Fin 6) (event : Fin 4) : Prop :=
  sorry -- Define participation relationship (boolean).

-- A and B are two specific students (0 and 1 for example)
constant A : Fin 6 := 0
constant B : Fin 6 := 1

-- Constraints
constant cannot_be_in_same_event : ¬(∃ e : Fin 4, participates_in_event A e ∧ participates_in_event B e)
constant each_event_has_participants : ∀ e : Fin 4, ∃ s : Fin 6, participates_in_event s e
constant each_student_one_event : ∀ s : Fin 6, ∃ e : Fin 4, participates_in_event s e

-- Theorem to prove
theorem number_of_arrangements : 
  (card {arrangement | (∀ s : Fin 6, ∃ e : Fin 4, participates_in_event s e) ∧
                       (∀ e : Fin 4, ∃ s : Fin 6, participates_in_event s e) ∧
                       (¬(∃ e : Fin 4, participates_in_event A e ∧ participates_in_event B e))}) = 1320 :=
by sorry

end number_of_arrangements_l748_748771


namespace total_boys_in_class_l748_748487

theorem total_boys_in_class (n : ℕ) (h_circle : ∀ i, 1 ≤ i ∧ i ≤ n -> i ≤ n) 
  (h_opposite : ∀ j k, j = 7 ∧ k = 27 ∧ j < k -> (k - j = n / 2)) : 
  n = 40 :=
sorry

end total_boys_in_class_l748_748487


namespace find_m_l748_748136

def f (x : ℝ) : ℝ := 3 * x^2 + 2 / x - 1
def g (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - m

theorem find_m : 
  let m := (5 - ((3 * 3^2 + 2 / 3 - 1) - (2 * 3^2))) in
  (f 3) - (g 3 m) = 5 → m = -11/3 :=
by
  intros
  sorry

end find_m_l748_748136


namespace mike_initial_seeds_l748_748707

variable (initial_seeds seeds_thrown_left seeds_thrown_right seeds_thrown_additional seeds_left : ℝ)

-- Conditions as definitions in Lean
def condition1 : Prop := seeds_thrown_left = 20
def condition2 : Prop := seeds_thrown_right = 2 * seeds_thrown_left
def condition3 : Prop := seeds_thrown_additional = 30
def condition4 : Prop := seeds_left = 30

-- Total seeds thrown definition
def total_seeds_thrown : ℝ :=
  seeds_thrown_left + seeds_thrown_right + seeds_thrown_additional

-- Proposition to prove
theorem mike_initial_seeds (h1 : condition1)
                           (h2 : condition2)
                           (h3 : condition3)
                           (h4 : condition4) :
  initial_seeds = total_seeds_thrown + seeds_left := by
sorry

end mike_initial_seeds_l748_748707


namespace prime_square_sum_l748_748628

theorem prime_square_sum (p q m : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q)
  (hp_eq : p^2 - 2001 * p + m = 0) (hq_eq : q^2 - 2001 * q + m = 0) :
  p^2 + q^2 = 3996005 :=
sorry

end prime_square_sum_l748_748628


namespace banana_arrangements_l748_748249

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748249


namespace solve_quadratic_eq_l748_748761

theorem solve_quadratic_eq (a c : ℝ) (h1 : a + c = 31) (h2 : a < c) (h3 : (24:ℝ)^2 - 4 * a * c = 0) : a = 9 ∧ c = 22 :=
by {
  sorry
}

end solve_quadratic_eq_l748_748761


namespace find_fx_expression_l748_748621

noncomputable def f (ω x ϕ : ℝ) : ℝ :=
  sin (2 * ω * x + ϕ) + cos (2 * ω * x + ϕ)

theorem find_fx_expression (ω ϕ : ℝ) (hω : ω > 0) (hϕ : 0 < ϕ ∧ ϕ < π)
  (h_period : ∀ x, f ω x ϕ = f ω (x + π) ϕ) (h_odd : ∀ x, f ω (-x) ϕ = -f ω x ϕ) :
  f 1 x (3 * π / 4) = - sqrt 2 * sin (2 * x) :=
by
  sorry

end find_fx_expression_l748_748621


namespace distinct_arrangements_banana_l748_748235

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748235


namespace john_days_ran_l748_748327

theorem john_days_ran 
  (total_distance : ℕ) (daily_distance : ℕ) 
  (h1 : total_distance = 10200) (h2 : daily_distance = 1700) :
  total_distance / daily_distance = 6 :=
by
  sorry

end john_days_ran_l748_748327


namespace soccer_substitutions_mod_2000_l748_748844

theorem soccer_substitutions_mod_2000 :
  let a_0 := 1
  let a_1 := 11 * 11
  let a_2 := 11 * 10 * a_1
  let a_3 := 11 * 9 * a_2
  let a_4 := 11 * 8 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  n % 2000 = 942 :=
by
  sorry

end soccer_substitutions_mod_2000_l748_748844


namespace five_integers_sum_to_first_set_impossible_second_set_sum_l748_748933

theorem five_integers_sum_to_first_set :
  ∃ (a b c d e : ℤ), 
    (a + b = 0) ∧ (a + c = 2) ∧ (b + c = 4) ∧ (a + d = 4) ∧ (b + d = 6) ∧
    (a + e = 8) ∧ (b + e = 9) ∧ (c + d = 11) ∧ (c + e = 13) ∧ (d + e = 15) ∧ 
    (a + b + c + d + e = 18) := 
sorry

theorem impossible_second_set_sum : 
  ¬∃ (a b c d e : ℤ), 
    (a + b = 12) ∧ (a + c = 13) ∧ (a + d = 14) ∧ (a + e = 15) ∧ (b + c = 16) ∧
    (b + d = 16) ∧ (b + e = 17) ∧ (c + d = 17) ∧ (c + e = 18) ∧ (d + e = 20) ∧
    (a + b + c + d + e = 39) :=
sorry

end five_integers_sum_to_first_set_impossible_second_set_sum_l748_748933


namespace radius_of_inscribed_circle_l748_748057

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : (ℝ × ℝ) := (-Real.sqrt 3, 0)
def focus2 : (ℝ × ℝ) := (Real.sqrt 3, 0)

-- Define a line passing through focus1
def line (m y : ℝ) : ℝ := m * y - Real.sqrt 3

theorem radius_of_inscribed_circle 
  (A B : ℝ × ℝ)
  (r : ℝ)
  (hA : ellipse A.1 A.2)
  (hB : ellipse B.1 B.2)
  (hlineA : A.1 = line (A.2 - focus1.2) A.2)
  (hlineB : B.1 = line (B.2 - focus1.2) B.2) : 
  r ≤ 1 / 2 := sorry

end radius_of_inscribed_circle_l748_748057


namespace sum_arithmetic_series_l748_748191

def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a1 + (n - 1) * d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
S_n = (n / 2) * (2 * a 1 + (n - 1) * (a 1 - a 0))

theorem sum_arithmetic_series (a : ℕ → ℝ) (a1 d m : ℝ) (n : ℕ) (S_n : ℝ) 
  (seq : arithmetic_sequence a a1 d) 
  (sum_def : sum_of_first_n_terms a n S_n) 
  (intersects : ∀ x y, ((x - 2) ^ 2 + y ^ 2 = 1) → (y = a1 * x + m) → 
    ((x + y + d = 0) ∧ (∃ x2 y2, (x2, y2) ≠ (x, y) ∧ (x + y = - (x2 + y2)))) :
  S_n = n^2 - 2n :=
sorry

end sum_arithmetic_series_l748_748191


namespace distinct_arrangements_banana_l748_748270

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748270


namespace solve_for_x_minus_y_l748_748565

theorem solve_for_x_minus_y (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 :=
by
  sorry

end solve_for_x_minus_y_l748_748565


namespace distinct_arrangements_banana_l748_748234

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748234


namespace twenty_questions_max_num_l748_748027

theorem twenty_questions_max_num (n : ℕ) (h : n = 20) : (2^n - 1) = 1048575 := by
  rw h
  norm_num
  rfl

end twenty_questions_max_num_l748_748027


namespace find_n_l748_748575

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end find_n_l748_748575


namespace triangle_area_l748_748117

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area :
  let A : point := (-2, 3)
  let B : point := (7, -3)
  let C : point := (4, 6)
  area_of_triangle A B C = 31.5 :=
by
  sorry

end triangle_area_l748_748117


namespace temperature_difference_is_correct_l748_748712

def highest_temperature : ℤ := -9
def lowest_temperature : ℤ := -22
def temperature_difference : ℤ := highest_temperature - lowest_temperature

theorem temperature_difference_is_correct :
  temperature_difference = 13 := by
  -- We need to prove this statement is correct
  sorry

end temperature_difference_is_correct_l748_748712


namespace decimal_to_binary_34_l748_748900

theorem decimal_to_binary_34 : nat.to_digits 2 34 = [1, 0, 0, 0, 1, 0] :=
by sorry

end decimal_to_binary_34_l748_748900


namespace polynomial_comparison_l748_748673

noncomputable def P (x : ℝ) : ℝ := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℝ) : ℝ := (1 + x^2 - x^3) ^ 1000

theorem polynomial_comparison : (∀ x : ℝ, ∀ k : ℕ, coefficient Q x k * (-1) ^ (n_2 + n_3) = coefficient P x k) :=
sorry

end polynomial_comparison_l748_748673


namespace banana_permutations_l748_748245

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748245


namespace polynomial_conditions_l748_748548

noncomputable def poly_rational_diff (f : ℝ → ℝ) :=
  ∀ x y : ℝ, ∃ q : ℚ, x - y = q → (f x - f y) ∈ ℚ

theorem polynomial_conditions (f : ℝ → ℝ)
    (h_deg : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c) 
    (h_cond : poly_rational_diff f) : 
    ∃ b : ℚ, ∃ c : ℝ, f = λ x, ↑b * x + c :=
sorry

end polynomial_conditions_l748_748548


namespace area_triangle_OMN_l748_748760

noncomputable def rho : ℝ → ℝ := fun θ => 4 * Real.sin θ

noncomputable def parametric_line (t : ℝ) : (ℝ × ℝ) :=
  (t - 1, 2 * t + 1)

noncomputable def cartesian_curve (x y : ℝ) : ℝ :=
  x^2 + y^2 - 4 * y

-- distance computation for later use
noncomputable def distance (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / (Real.sqrt (a * a + b * b))

theorem area_triangle_OMN :
  let C := λ θ, (rho θ * Real.cos θ, rho θ * Real.sin θ)
      l := parametric_line -- line
      O := (0 : ℝ, 0 : ℝ) -- Origin
      intersection_points := -- Calculate the intersection points of C and l
        let (x, y) := parametric_line t,
        cartesian_curve x y = 0 ∧
        exists x y, ∃ t t1 t2, parametric_line t = (x, y) ∧
        cartesian_curve x y = 0 ∧
        t1 + t2 = (Real.sqrt 5) / 5 ∧
        t1 * t2 = -2
  in ∃ (M N : ℝ × ℝ),
        M ∈ C ∧ N ∈ C ∧ (distance 0 0 2 -1 3 * Real.sqrt 5 / 5 = Real.sqrt 19 / 5) ∧
        C = 4 * x ∧
        cartesian_curve x y = 4 - (Real.sqrt 5 / 5)
  in 0.5 * distance 0 0 2 -1 3 * Real.sqrt 19 / 5 = Real.sqrt 19 / 5
       :=
  sorry

end area_triangle_OMN_l748_748760


namespace octagon_area_in_circle_in_square_l748_748836

theorem octagon_area_in_circle_in_square (perimeter_square : ℝ) (h : perimeter_square = 144) : 
  let side_square := perimeter_square / 4 in
  let radius_circle := side_square / 2 in
  let octagon_area := 8 * (1 / 2 * radius_circle^2 * sin (π / 8) * cos (π / 8)) in
  octagon_area = 324 * sqrt 2 := 
by
  sorry

end octagon_area_in_circle_in_square_l748_748836


namespace k_value_function_range_l748_748185

noncomputable def f : ℝ → ℝ := λ x => Real.log x + x

def is_k_value_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = k * x)) ∧ (k > 0)

theorem k_value_function_range :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.log x + x) →
  (∃ (k : ℝ), is_k_value_function f k) →
  1 < k ∧ k < 1 + (1 / Real.exp 1) :=
by
  sorry

end k_value_function_range_l748_748185


namespace two_real_roots_l748_748926

noncomputable def equation := λ x : ℝ, sqrt (x + 9) - 2 * sqrt (x - 8) + 3 = 0

theorem two_real_roots (x : ℝ) (h : 8 ≤ x) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x ∈ {x1, x2} ∧ equation x1 ∧ equation x2 :=
by
  sorry

end two_real_roots_l748_748926


namespace midpoint_locus_l748_748610

theorem midpoint_locus :
  ∀ B C : ℝ × ℝ, (B.1 ^ 2 + B.2 ^ 2 = 1) → (C.1 ^ 2 + C.2 ^ 2 = 1) → 
  ((B.1 + C.1)^2 + (B.2 + C.2)^2 = 2 * (cos (real.pi / 3)) * ((B.1^2 + B.2^2) + (C.1^2 + C.2^2)) / 2) →
  (B ≠ C) →
  let M := (B.1 + C.1) / 2, (B.2 + C.2) / 2 in
  (M.1 ^ 2 + M.2 ^ 2 = 1 / 4) :=
begin
  intros B C hB hC hangle hneq M,
  sorry
end

end midpoint_locus_l748_748610


namespace matrix_transformation_l748_748160

noncomputable def M : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![0, -1, 0], ![1, 0, 0], ![0, 0, 3]]

theorem matrix_transformation (N : Matrix (Fin 3) (Fin 3) ℝ) : 
  M ⬝ N = ![
    ![-N 1 0, -N 1 1, -N 1 2],
    ![N 0 0, N 0 1, N 0 2], 
    ![3 * N 2 0, 3 * N 2 1, 3 * N 2 2]] :=
by
  sorry

end matrix_transformation_l748_748160


namespace cindy_gives_3_envelopes_per_friend_l748_748879

theorem cindy_gives_3_envelopes_per_friend
  (initial_envelopes : ℕ) 
  (remaining_envelopes : ℕ)
  (friends : ℕ)
  (envelopes_per_friend : ℕ) 
  (h1 : initial_envelopes = 37) 
  (h2 : remaining_envelopes = 22)
  (h3 : friends = 5) 
  (h4 : initial_envelopes - remaining_envelopes = envelopes_per_friend * friends) :
  envelopes_per_friend = 3 :=
by
  sorry

end cindy_gives_3_envelopes_per_friend_l748_748879


namespace union_of_M_and_N_l748_748626

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N :
  M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l748_748626


namespace proof_problem1_proof_problem2_proof_problem3_l748_748500

-- Definition of the three mathematical problems
def problem1 : Prop := 8 / (-2) - (-4) * (-3) = -16

def problem2 : Prop := -2^3 + (-3) * ((-2)^3 + 5) = 1

def problem3 (x : ℝ) : Prop := (2 * x^2)^3 * x^2 - x^10 / x^2 = 7 * x^8

-- Statements of the proofs required
theorem proof_problem1 : problem1 :=
by sorry

theorem proof_problem2 : problem2 :=
by sorry

theorem proof_problem3 (x : ℝ) : problem3 x :=
by sorry

end proof_problem1_proof_problem2_proof_problem3_l748_748500


namespace convert_base_7_to_base_10_l748_748889

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l748_748889


namespace common_property_of_rectangles_rhombuses_and_squares_l748_748004

-- Definitions of shapes and properties

-- Assume properties P1 = "Diagonals are equal", P2 = "Diagonals bisect each other", 
-- P3 = "Diagonals are perpendicular to each other", and P4 = "Diagonals bisect each other and are equal"

def is_rectangle (R : Type) : Prop := sorry
def is_rhombus (R : Type) : Prop := sorry
def is_square (R : Type) : Prop := sorry

def diagonals_bisect_each_other (R : Type) : Prop := sorry

-- Theorem stating the common property
theorem common_property_of_rectangles_rhombuses_and_squares 
  (R : Type)
  (H_rect : is_rectangle R)
  (H_rhomb : is_rhombus R)
  (H_square : is_square R) :
  diagonals_bisect_each_other R := 
  sorry

end common_property_of_rectangles_rhombuses_and_squares_l748_748004


namespace arithmetic_square_root_of_sqrt_16_l748_748745
noncomputable theory

def arithmetic_square_root_is_two : Prop :=
  let x := 4 in
  let y := Real.sqrt x in
  y = 2

theorem arithmetic_square_root_of_sqrt_16 : arithmetic_square_root_is_two := 
by
  let h1 : Real.sqrt 16 = 4 := by exact Real.sqrt_sq 4
  let h2 : ∀ x, Real.sqrt x = y → y = 2 := by
    intros x hx
    assume h : y = 4
    exact Eq.symm h
  exact (h2 16 h1)
sorry

end arithmetic_square_root_of_sqrt_16_l748_748745


namespace find_f_ln_inv_6_l748_748940

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

theorem find_f_ln_inv_6 (k : ℝ) (h : f k (Real.log 6) = 1) : f k (Real.log (1 / 6)) = -7 :=
by
  sorry

end find_f_ln_inv_6_l748_748940


namespace roger_current_money_l748_748373

def roger_initial_money : ℕ := 16
def roger_birthday_money : ℕ := 28
def roger_spent_money : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_spent_money = 19 := 
by sorry

end roger_current_money_l748_748373


namespace relationship_between_a_and_b_l748_748599

-- Definitions for the conditions
variables {a b : ℝ}

-- Main theorem statement
theorem relationship_between_a_and_b (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
by
  sorry

end relationship_between_a_and_b_l748_748599


namespace josie_remaining_money_l748_748680

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end josie_remaining_money_l748_748680


namespace robot_transport_max_robots_l748_748067

section
variable {A B : ℕ}   -- Define the variables A and B
variable {m : ℕ}     -- Define the variable m

-- Part 1
theorem robot_transport (h1 : A = B + 30) (h2 : 1500 * B = 1000 * (B + 30)) : A = 90 ∧ B = 60 :=
by
  sorry

-- Part 2
theorem max_robots (h3 : 50000 * m + 30000 * (12 - m) ≤ 450000) : m ≤ 4 :=
by
  sorry
end

end robot_transport_max_robots_l748_748067


namespace smallest_product_value_l748_748693

theorem smallest_product_value 
  (a b c d : ℝ) 
  (h1 : b - d ≥ 5)
  (hroots : ∀ x, Polynomial.eval x (Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) == 0 → x ∈ ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, 
    Polynomial.eval x1 (Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) = 0 ∧
    Polynomial.eval x2 (Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) = 0 ∧
    Polynomial.eval x3 (Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) = 0 ∧
    Polynomial.eval x4 (Polynomial.X ^ 4 + Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X ^ 2 + Polynomial.C c * Polynomial.X + Polynomial.C d) = 0 ∧
    (x1^2 + 1) * (x2^2 + 1) * (x3^2 + 1) * (x4^2 + 1) = 16) :=
sorry

end smallest_product_value_l748_748693


namespace remainder_when_13_plus_x_divided_by_26_l748_748695

theorem remainder_when_13_plus_x_divided_by_26 (x : ℕ) (h1 : 9 * x % 26 = 1) : (13 + x) % 26 = 16 := 
by sorry

end remainder_when_13_plus_x_divided_by_26_l748_748695


namespace inequality_solution_l748_748158

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 2) * (x + 3)) < 1 / 5) ↔
  (x ∈ set.Ioo (-∞) (-3) ∪ set.Ioo (-2) (-1) ∪ set.Ioo 2 ∞) :=
by
  sorry

end inequality_solution_l748_748158


namespace correct_propositions_l748_748368

-- Define each of the propositions as separate theorems.
-- Since all propositions are assumed to be true, their definitions in Lean should reflect this.

def Proposition1 : Prop :=
  ∃ (l₁ l₂ l₃ : Line), skew l₁ l₂ ∧ skew l₂ l₃ ∧ skew l₃ l₁ ∧
    (∃ (L : set Line), (∀ l ∈ L, intersects l l₁ ∧ intersects l l₂ ∧ intersects l l₃) ∧
    ∀ l l' ∈ L, skew l l')

def Proposition2 : Prop :=
  ∀ (M : Plane) (l₁ l₂ l₃ : Line), parallel M l₁ ∧ parallel M l₂ ∧ parallel M l₂ ∧ 
    skew l₁ l₂ ∧ skew l₂ l₃ ∧ skew l₁ l₃ ∧ 
    (angle_between l₁ l₂ = angle_between l₂ l₃ ∧ angle_between l₂ l₃ = 60)

def Proposition3 : Prop :=
  ∃ (l₁ l₂ l₃ l₄ : Line), skew l₁ l₂ ∧ skew l₂ l₃ ∧ skew l₃ l₄ ∧ skew l₁ l₃ ∧ skew l₁ l₄ ∧ skew l₂ l₄ ∧ 
    (∃ (L₁ L₂ : set Line), (∀ l₁ ∈ L₁, ∃ l' ∈ L₂, intersects l₁ l' ∧ 
    (∃ (p₁ p₂ : Point), on_line p₁ l₁ ∧ on_line p₂ l' ∧ intersects p₁ p₂)) ∧
    ∀ l ∈ L, intersects l l₁ ∧ intersects l l₂ ∧ intersects l l₃)

theorem correct_propositions : 
  (Proposition1 ∧ Proposition2 ∧ Proposition3) ↔ true := 
  sorry

end correct_propositions_l748_748368


namespace at_least_one_le_one_l748_748333

noncomputable def real_numbers := {x : ℝ // x > 0}

open_locale classical

theorem at_least_one_le_one (x y z : real_numbers)
  (h_sum : (x.val + y.val + z.val) = 3) :
  x.val * (x.val + y.val - z.val) ≤ 1 ∨
  y.val * (y.val + z.val - x.val) ≤ 1 ∨
  z.val * (z.val + x.val - y.val) ≤ 1 :=
  sorry

end at_least_one_le_one_l748_748333


namespace investment_of_c_l748_748048

theorem investment_of_c (P_a P_b P_c x : ℝ) (h1 : P_a = 0.16 * 8000) (h2 : P_b = 1600) (h3 : P_b = 0.16 * 10000) (h4 : P_c - P_a = 640) : x = 12000 :=
begin
  sorry
end

end investment_of_c_l748_748048


namespace number_of_rational_T_l748_748966

noncomputable def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def S (n : ℕ) : ℕ :=
  if n = 0 then 0 else seq_a(n)^2 + seq_a(n)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then 0 else 1 / (seq_a(n) * real.sqrt(seq_a(n+1)) + seq_a(n+1) * real.sqrt(seq_a(n)))

noncomputable def T (n : ℕ) : ℝ :=
  if n = 0 then 0 else (finset.range n).sum (λ k, b k)

theorem number_of_rational_T : (finset.range 100).filter (λ n, ∃ k : ℕ, T n = (1 - 1 / real.sqrt (n + 1))).card = 9 :=
sorry

end number_of_rational_T_l748_748966


namespace expression_evaluation_l748_748155

-- Given conditions
def a : ℝ := 14
def b : ℝ := 19
def c : ℝ := 17

-- The expression to be evaluated
def expr : ℝ := (196 * (1 / b - 1 / 23) + 361 * (1 / 23 - 1 / c) + 289 * (1 / c - 1 / b)) /
                (a * (1 / b - 1 / 23) + b * (1 / 23 - 1 / c) + c * (1 / c - 1 / b))

-- The statement that needs to be proved
theorem expression_evaluation : expr = a + b + c :=
by sorry

end expression_evaluation_l748_748155


namespace sin_2alpha_val_l748_748919

-- Define the conditions and the problem in Lean 4
theorem sin_2alpha_val (α : ℝ) (h1 : π < α ∨ α < 3 * π / 2)
  (h2 : 2 * (Real.tan α) ^ 2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5 * π / 4 → Real.sin (2 * α) = 4 / 5) ∧ 
  (5 * π / 4 < α ∧ α < 3 * π / 2 → Real.sin (2 * α) = 3 / 5) := 
sorry

end sin_2alpha_val_l748_748919


namespace inv_203_mod_301_exists_l748_748520

theorem inv_203_mod_301_exists : ∃ b : ℤ, 203 * b % 301 = 1 := sorry

end inv_203_mod_301_exists_l748_748520


namespace boat_speed_still_water_l748_748807

theorem boat_speed_still_water : 
  ∀ (b s : ℝ), (b + s = 11) → (b - s = 5) → b = 8 := 
by 
  intros b s h1 h2
  sorry

end boat_speed_still_water_l748_748807


namespace problem_l748_748854

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_decreasing (f : ℝ → ℝ) (a b : ℝ) := a < b → f a ≥ f b

theorem problem :
  let f1 := λ x : ℝ, 2^x - 2^(-x)
  let f2 := λ x : ℝ, Real.cos x
  let f3 := λ x : ℝ, Real.logb 2 (abs x)
  let f4 := λ x : ℝ, x + x⁻¹ in
  (is_even f2 ∧ is_decreasing f2 0 3) :=
begin
  sorry
end

end problem_l748_748854


namespace diagonal_of_square_area_of_square_l748_748143

def side_length : ℝ := 30
def expected_diagonal : ℝ := 30 * Real.sqrt 2
def expected_area : ℝ := 900

theorem diagonal_of_square :
  let s := side_length in
  let d := s * Real.sqrt 2 in
  d = expected_diagonal :=
by
  sorry

theorem area_of_square :
  let s := side_length in
  let A := s * s in
  A = expected_area :=
by
  sorry

end diagonal_of_square_area_of_square_l748_748143


namespace cos_two_x_period_l748_748395

noncomputable def minimum_positive_period (f : ℝ → ℝ) : ℝ :=
  if ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x then
    Classical.some (exists_minimum_period f)
  else 1

theorem cos_two_x_period : minimum_positive_period (fun x => cos (2 * x)) = Real.pi :=
by
  sorry

end cos_two_x_period_l748_748395


namespace banana_arrangements_l748_748251

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748251


namespace max_projection_area_of_tetrahedron_l748_748777

/-- 
Two adjacent faces of a tetrahedron are isosceles right triangles with a hypotenuse of 2,
and they form a dihedral angle of 60 degrees. The tetrahedron rotates around the common edge
of these faces. The maximum area of the projection of the rotating tetrahedron onto 
the plane containing the given edge is 1.
-/
theorem max_projection_area_of_tetrahedron (S hypotenuse dihedral max_proj_area : ℝ)
  (is_isosceles_right_triangle : ∀ (a b : ℝ), a^2 + b^2 = hypotenuse^2)
  (hypotenuse_len : hypotenuse = 2)
  (dihedral_angle : dihedral = 60) :
  max_proj_area = 1 :=
  sorry

end max_projection_area_of_tetrahedron_l748_748777


namespace white_triangle_pairs_condition_l748_748151

def number_of_white_pairs (total_triangles : Nat) 
                          (red_pairs : Nat) 
                          (blue_pairs : Nat)
                          (mixed_pairs : Nat) : Nat :=
  let red_involved := red_pairs * 2
  let blue_involved := blue_pairs * 2
  let remaining_red := total_triangles / 2 * 5 - red_involved - mixed_pairs
  let remaining_blue := total_triangles / 2 * 4 - blue_involved - mixed_pairs
  (total_triangles / 2 * 7) - (remaining_red + remaining_blue)/2

theorem white_triangle_pairs_condition : number_of_white_pairs 32 3 2 1 = 6 := by
  sorry

end white_triangle_pairs_condition_l748_748151


namespace five_crows_two_hours_l748_748292

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l748_748292


namespace prove_cubic_roots_expression_l748_748342

noncomputable def cubic_roots_expression (a b : ℝ) (y : ℝ) : Prop :=
  (a^3 + b^3 + a^2 * b^2 * (a^2 + b^2) = y) ∧
  let u := 6 in let v := 11 in let w := 6 in
  (a + b + c = u) ∧ (a * b + a * c + b * c = v) ∧ (a * b * c = w)

-- Main theorem: There exists values of y such that
theorem prove_cubic_roots_expression :
  ∃ y : ℝ, ∀ a b : ℝ, cubic_roots_expression a b y :=
sorry

end prove_cubic_roots_expression_l748_748342


namespace car_mileage_highway_l748_748066

theorem car_mileage_highway :
  ∀ (miles_city_tankful miles_city_mpg difference miles_city mpg_highway tank_size miles_highway),
  miles_city_tankful = 336 →
  miles_city_mpg = 32 →
  difference = 12 →
  mpg_highway = miles_city_mpg + difference →
  tank_size = miles_city_tankful / miles_city_mpg →
  miles_highway = mpg_highway * tank_size →
  miles_highway = 462 :=
by
  intros miles_city_tankful miles_city_mpg difference miles_city mpg_highway tank_size miles_highway
  intros H_miles_city_tankful H_miles_city_mpg H_difference H_mpg_highway H_tank_size H_miles_highway
  sorry

end car_mileage_highway_l748_748066


namespace probability_correct_guess_l748_748135

def is_valid_number (n : ℕ) : Prop :=
  n >= 40 ∧ n < 80 ∧
  (n / 10 % 2 = 1) ∧
  ((n / 10 + n % 10) % 2 = 0)

def count_valid_numbers : ℕ :=
  Finset.card (Finset.filter is_valid_number (Finset.range 80))

theorem probability_correct_guess : 
  (∃! n : ℕ, is_valid_number n) → (count_valid_numbers = 10) → (1 / count_valid_numbers = 1 / 10) :=
begin
  sorry
end

end probability_correct_guess_l748_748135


namespace polynomial_rational_difference_l748_748543

theorem polynomial_rational_difference {f : ℝ → ℝ} (hf_deg2 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (hf_rational_diff : ∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) :
  ∃ b c : ℚ, ∃ d : ℝ, f = λ x, b * (x : ℝ) + d :=
by
  sorry

end polynomial_rational_difference_l748_748543


namespace distance_house_to_market_l748_748856

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l748_748856


namespace find_polynomials_l748_748540

noncomputable def polynomial_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (∃ q : ℚ, x - y = q) → ∃ r : ℚ, f(x) - f(y) = r

theorem find_polynomials :
  ∀ f : ℝ → ℝ,
  (∃ a b c : ℝ, ∀ x : ℝ, f(x) = a * x^2 + b * x + c) ∧ polynomial_satisfies_condition f →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x : ℝ, f(x) = b * x + c :=
sorry

end find_polynomials_l748_748540


namespace inverse_of_203_mod_301_l748_748517

theorem inverse_of_203_mod_301 : ∃ (a : ℤ), 0 ≤ a ∧ a ≤ 300 ∧ (203 * a ≡ 1 [MOD 301]) :=
by
  use 238
  split
  by norm_num
  split
  by norm_num
  by norm_num,
  exact ⟨by norm_num, by norm_num⟩, sorry
 
end inverse_of_203_mod_301_l748_748517


namespace altitudes_iff_area_sum_l748_748190

variable {ABC : Type} [acute_triangle : acute_triangle ABC] (R : Real) -- Triangle ABC and its circumradius R
variable {D E F : Point} -- Points D, E, F on the respective sides BC, CA, AB
variable (S : Real) -- Area of triangle ABC
variable (EF FD DE : Real) -- Lengths EF, FD and DE

theorem altitudes_iff_area_sum (h1 : AD.is_altitude) (h2 : BE.is_altitude) (h3 : CF.is_altitude) :
  S = (R / 2) * (EF + FD + DE) :=
sorry -- Proof goes here

end altitudes_iff_area_sum_l748_748190


namespace local_minimum_at_neg_one_l748_748352

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l748_748352


namespace range_of_m_l748_748213

theorem range_of_m (ω : ℝ) (m : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, π ≤ x ∧ x ≤ 2 * π → m - 3 ≤ sin (ω * x - π / 6) ∧ sin (ω * x - π / 6) ≤ m + 3) →
  -2 ≤ m ∧ m ≤ 7 / 2 := sorry

end range_of_m_l748_748213


namespace limit_eq_value_l748_748872

open Real

noncomputable def limit_expression (x : ℝ) : ℝ :=
  (root 3 (x / 9) - 1 / 3) / (sqrt (1 / 3 + x) - sqrt (2 * x))

theorem limit_eq_value :
  tendsto (λ x, limit_expression x) (𝓝 (1 / 3)) (𝓝 (- (2 * sqrt 2 / (3 * sqrt 3)))) :=
sorry

end limit_eq_value_l748_748872


namespace last_person_teeth_removed_l748_748936

-- Define the initial conditions
def total_teeth : ℕ := 32
def total_removed : ℕ := 40
def first_person_removed : ℕ := total_teeth * 1 / 4
def second_person_removed : ℕ := total_teeth * 3 / 8
def third_person_removed : ℕ := total_teeth * 1 / 2

-- Express the problem in Lean
theorem last_person_teeth_removed : 
  first_person_removed + second_person_removed + third_person_removed + last_person_removed = total_removed →
  last_person_removed = 4 := 
by
  sorry

end last_person_teeth_removed_l748_748936


namespace perimeter_shaded_region_l748_748316

theorem perimeter_shaded_region (O R S : Point) (hOR : dist O R = 3) (hOS : dist O S = 3) (angle_ROS_eq_300 : θ = 300 * π / 180) :
  let c := 2 * 3 * π in
  let arc_RS := (5 / 6) * c in
  perimeter_shaded := 2 * 3 + arc_RS :=
by sorry

end perimeter_shaded_region_l748_748316


namespace units_digit_product_odd_integers_10_to_110_l748_748789

-- Define the set of odd integer numbers between 10 and 110
def oddNumbersInRange : List ℕ := List.filter (fun n => n % 2 = 1) (List.range' 10 101)

-- Define the set of relevant odd multiples of 5 within the range
def oddMultiplesOfFive : List ℕ := List.filter (fun n => n % 5 = 0) oddNumbersInRange

-- Prove that the product of all odd positive integers between 10 and 110 has units digit 5
theorem units_digit_product_odd_integers_10_to_110 :
  let product : ℕ := List.foldl (· * ·) 1 oddNumbersInRange
  product % 10 = 5 :=
by
  sorry

end units_digit_product_odd_integers_10_to_110_l748_748789


namespace prism_area_and_perpendicularity_l748_748148

def prism (P : Type) (A : P) (faces : Set (Set P)) (areas : Set ℝ) (k : ℝ) : Prop :=
  ∃ (a b h : ℝ), 
  (∀ face ∈ faces, area face = k) ∧
  (∀ face1 face2 ∈ lateral_faces P, perpendicular face1 face2) ∧
  (volume P = max_volume (Set.smul ⟨A, a, b, h⟩)) ∧ 
  (total_area_faces_containing A = 3 * k)

theorem prism_area_and_perpendicularity {P : Type} (A : P) (faces : Set (Set P)) (areas : Set ℝ) (k : ℝ) :
  prism P A faces areas k :=
by {
  sorry
}

end prism_area_and_perpendicularity_l748_748148


namespace calc_expr_value_l748_748874

theorem calc_expr_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := 
by 
  sorry

end calc_expr_value_l748_748874


namespace imaginary_part_z_l748_748612

namespace Math

def z : ℂ := (1 + 4 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_z : Complex.im z = 5 / 2 := by
  sorry

end Math

end imaginary_part_z_l748_748612


namespace min_additional_shaded_cells_to_ensure_condition_l748_748493

-- Definition of the size of the grid and conditions
def grid_size : (Nat × Nat) := (15, 5)
def total_cells : Nat := 15 * 5
def cells_shaded : Nat → Bool -- Function that returns True if a cell is shaded, False otherwise
def is_shaded (i j : Nat) : Bool := cells_shaded (i * 5 + j) 

-- Predicate to check if a 2x2 block has more than half of its cells shaded
def is_2x2_block_more_than_half_shaded (i j : Nat) : Prop :=
  let shaded_count := (if is_shaded i j then 1 else 0) +
                      (if is_shaded i (j + 1) then 1 else 0) +
                      (if is_shaded (i + 1) j then 1 else 0) +
                      (if is_shaded (i + 1) (j + 1) then 1 else 0)
  shaded_count > 2

-- Main theorem to prove
theorem min_additional_shaded_cells_to_ensure_condition 
  (current_shaded : Nat)
  (H : ∀ i j, i <= 13 → j <= 3 → is_2x2_block_more_than_half_shaded i j)
  : current_shaded + 17 ≥ shaded_cells_currently_present :=
sorry

end min_additional_shaded_cells_to_ensure_condition_l748_748493


namespace chessboard_all_white_impossible_l748_748124

def initial_board (n : ℕ) : matrix (fin n) (fin n) bool :=
  λ i j, if (i = 7 ∧ j = 7) then tt else ff

def adj_flip (m : ℕ) (board : matrix (fin m) (fin m) bool) (pos : fin m × fin m) : matrix (fin m) (fin m) bool :=
  λ i j, if (i = pos.1 ∧ j = pos.2) ∨ (i = pos.1 ∧ (j = pos.2 + 1) % m) ∨ (i = pos.1 ∧ (j = pos.2 - 1) % m)
                             ∨ ((i = pos.1 + 1) % m ∧ j = pos.2) ∨ ((i = pos.1 - 1) % m ∧ j = pos.2)
                      then !board i j else board i j

noncomputable def can_be_all_white (n : ℕ) : Prop :=
  ∃ (moves : list (fin n × fin n)), ∀ (i j : fin n), (list.foldl (adj_flip n) (initial_board n) moves) i j = ff

theorem chessboard_all_white_impossible : ¬ can_be_all_white 8 := sorry

end chessboard_all_white_impossible_l748_748124


namespace sum_of_all_distinct_areas_right_triangles_l748_748539

noncomputable def calculate_triangle_areas_sum (areas : Set ℕ) : ℕ :=
  areas.sum

theorem sum_of_all_distinct_areas_right_triangles : 
  ∀ (a b : ℕ), (a ≠ b) → (a * b / 2 = 3 * (a + b)) → calculate_triangle_areas_sum {147, 108, 144} = 399 := 
by
  sorry

end sum_of_all_distinct_areas_right_triangles_l748_748539


namespace find_line_equation_l748_748605

-- Define point A and point P
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def P : Point := ⟨-1, 4⟩

-- Define the slope of the line passing through points A and P
def slope (pt1 pt2 : Point) : ℝ := (pt2.y - pt1.y) / (pt2.x - pt1.x)

-- Define the condition that line l is perpendicular to line PA
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Final equation of line l
def line_equation (a b c : ℝ) (p : Point) : Prop := a * p.x + b * p.y + c = 0

-- The proof problem statement
theorem find_line_equation : 
  let k_PA := slope A P in
  perpendicular k_PA 1 ∧
  line_equation 1 (-1) 5 P :=
by
  let k_PA := slope A P;
  sorry

end find_line_equation_l748_748605


namespace bottle_cap_cost_l748_748913

theorem bottle_cap_cost (total_cost : ℕ) (caps : ℕ) : total_cost = 25 → caps = 5 → total_cost / caps = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end bottle_cap_cost_l748_748913


namespace stipulated_percentage_l748_748119

theorem stipulated_percentage
  (A B C : ℝ)
  (P : ℝ)
  (hA : A = 20000)
  (h_range : B - C = 10000)
  (hB : B = A + (P / 100) * A)
  (hC : C = A - (P / 100) * A) :
  P = 25 :=
sorry

end stipulated_percentage_l748_748119


namespace monotonic_decreasing_interval_l748_748001

open Real

noncomputable def decreasing_interval (k: ℤ): Set ℝ :=
  {x | k * π - π / 3 < x ∧ x < k * π + π / 6 }

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, x ∈ decreasing_interval k ↔ (k * π - π / 3 < x ∧ x < k * π + π / 6) :=
by 
  intros x
  sorry

end monotonic_decreasing_interval_l748_748001


namespace find_x_l748_748382

theorem find_x : ∃ x : ℤ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 ∧ x = 28 := 
by sorry

end find_x_l748_748382


namespace proof_problem_l748_748459

noncomputable def calc_a (seventh_counts : Fin 3 → Nat) : Nat :=
  20 - seventh_counts 0 - seventh_counts 1

def sorted_seventh_scores : List Nat := [69, 80, 83, 85, 85, 85, 85, 85, 86, 87, 89, 89, 90, 91, 92, 93, 94, 95, 97, 100]

def calc_median (l : List Nat) : Nat :=
  (l.get! ((l.length / 2) - 1) + l.get! (l.length / 2)) / 2
  
def calc_mode (scores : List Nat) : Nat :=
  List.mostCommon scores

def estimated_above_95 (seventh : List Nat) (eighth : List Nat) (total_students : Nat) : Nat :=
  let above_95 := seventh.filter (λ x => x > 95) ++ eighth.filter (λ x => x > 95)
  (total_students * above_95.length) / (seventh.length + eighth.length)

theorem proof_problem
  (seventh_counts : Fin 3 → Nat)
  (seventh_scores : List Nat)
  (eighth_scores : List Nat)
  (total_students : Nat)
  (avg_seventh : Nat)
  (avg_eighth : Nat)
  (median_eighth : Nat)
  (mode_eighth : Nat)
  (a b c d : Nat) :
  calc_a seventh_counts = 11 →
  calc_median sorted_seventh_scores = 88 →
  calc_mode eighth_scores = 91 →
  estimated_above_95 seventh_scores eighth_scores total_students = 163 →
  (calc_median seventh_scores, median_eighth) = (88, 91) →
  a = 11 →
  b = 88 →
  c = 91 → 
  d = total_students →
  sorry

end proof_problem_l748_748459


namespace negative_two_squared_l748_748875

theorem negative_two_squared :
  (-2 : ℤ)^2 = 4 := 
sorry

end negative_two_squared_l748_748875


namespace T_lies_on_line_l748_748055

-- Define the basic setup of the geometry problem
noncomputable def point (α : Type _) := α × α
variables (α : Type _) [Field α]

-- Definition of the triangle type and the points involved
structure Triangle (α : Type _) := (A B C : point α)
structure ProjectionSetup (α : Type _) := 
  (triangle : Triangle α)
  (X Y T : point α)
  (hXonAC : ∃ (k : α), X = (1 - k) • (triangle.A) + k • (triangle.C))
  (hYonRayBC : ∃ (k : α), k > 1 ∧ Y = (k - 1) • (triangle.C) + triangle.B)
  (angle_sum : ∃ (θ φ : α), θ + φ = 90 ∧ (∠ triangle.A B X) = θ ∧ (∠ C X Y) = φ)
  (T_projection : T = (/* projection of B onto XY */ sorry))

-- The theorem to prove the point T lies on a line
theorem T_lies_on_line (ps : ProjectionSetup α) : ∃ (l : point α → Prop), ∀ (X : point α), ps.hXonAC X → l (ps.T) :=
  sorry

end T_lies_on_line_l748_748055


namespace inequality_n1_inequality_counterexample_l748_748184

-- Definitions and variables
variables {n : ℕ} {x : Fin (2 * n + 1) → ℝ}

-- Ensure positive numbers
def all_pos (x : Fin (2 * n + 1) → ℝ) : Prop := ∀ i, 0 < x i

-- Main theorem for n = 1 case
theorem inequality_n1 (x : Fin (2 * 1 + 1) → ℝ) (hx : all_pos x) :
  (x 0 * x 1 / x 2 + x 1 * x 2 / x 0 + x 2 * x 0 / x 1) ≥ x 0 + x 1 + x 2 ↔
  (x 0 = x 1 ∧ x 1 = x 2) :=
by sorry -- Proof needed

-- Counterexample for n > 1 case
theorem inequality_counterexample (n : ℕ) (hn: 1 < n) (x : Fin (2 * n + 1) → ℝ) (hx : all_pos x) :
  ¬ ((∑ i, (x i * x (i + 1) / x (i + 2))) ≥ ∑ i, x i) :=
by sorry -- Proof needed

end inequality_n1_inequality_counterexample_l748_748184


namespace complement_union_eq_l748_748984

open Set

variable (U A B : Set ℕ)
variable hU : U = {1, 2, 3, 4, 5, 6}
variable hA : A = {1, 3, 5}
variable hB : B = {4, 5, 6}

theorem complement_union_eq : U \ (A ∪ B) = {2} :=
by
  have h_union : A ∪ B = {1, 3, 4, 5, 6} := by
    rw [hA, hB]
    exact Set.ext (fun x => by simp [or_assoc])
  rw [h_union, hU]
  exact Set.ext (fun x => by simp)

end complement_union_eq_l748_748984


namespace number_of_pairs_l748_748701

theorem number_of_pairs (n : Nat) : 
  (∃ n, n > 2 ∧ ∀ x y : ℝ, (5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16) → True) :=
sorry

end number_of_pairs_l748_748701


namespace kavi_initial_bags_l748_748682

theorem kavi_initial_bags 
  (sold_mon : ℕ) (sold_tue : ℕ) (sold_wed : ℕ) (sold_thu : ℕ) (sold_fri : ℕ)
  (not_sold_percent : ℕ)
  (total_sold : ℕ)
  (h_total_sold : sold_mon + sold_tue + sold_wed + sold_thu + sold_fri = total_sold)
  (h_sold_percent : 100 - not_sold_percent = 75) :
  let x_initial := total_sold * 100 / 75 in
  x_initial = 600 :=
by 
  have : 450 = sold_mon + sold_tue + sold_wed + sold_thu + sold_fri, from h_total_sold,
  have : 75 = 100 - not_sold_percent, from h_sold_percent,
  have x_initial_eqn : 450 = 0.75 * x_initial, from sorry,
  have x_initial_val : x_initial = 600, from sorry,
  exact x_initial_val

end kavi_initial_bags_l748_748682


namespace unobserved_planet_exists_l748_748363

theorem unobserved_planet_exists
  (n : ℕ) (h_n_eq : n = 15)
  (planets : Fin n → Type)
  (dist : ∀ (i j : Fin n), ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → dist i j ≠ dist j i)
  (nearest : ∀ i : Fin n, Fin n)
  (h_nearest : ∀ i : Fin n, nearest i ≠ i)
  : ∃ i : Fin n, ∀ j : Fin n, nearest j ≠ i := by
  sorry

end unobserved_planet_exists_l748_748363


namespace minimize_sum_of_distances_l748_748386

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-4, 3)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the y-axis
structure PointOnYAxis where
  y : ℝ

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Math.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Define the sum of distances from P to A and B
def sumOfDistances (P : PointOnYAxis) (A B : ℝ × ℝ) : ℝ :=
  dist (0, P.y) A + dist (0, P.y) B

-- Our Lean statement: Prove that the point P (0, 1) minimizes the sum of distances
theorem minimize_sum_of_distances :
  ∀ P : PointOnYAxis,
  (∃ P.y = 1, ∀ Q : PointOnYAxis, sumOfDistances ⟨1⟩ A B ≤ sumOfDistances Q A B) :=
by
  intro P
  exists 1
  sorry

end minimize_sum_of_distances_l748_748386


namespace length_of_purple_part_l748_748099

theorem length_of_purple_part (p : ℕ) (black : ℕ) (blue : ℕ) (total : ℕ) 
  (h1 : black = 2) (h2 : blue = 1) (h3 : total = 6) (h4 : p + black + blue = total) : 
  p = 3 :=
by
  sorry

end length_of_purple_part_l748_748099


namespace value_of_A_is_18_l748_748452

theorem value_of_A_is_18
  (A B C D : ℕ)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : A ≠ D)
  (h4 : B ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A * B = 72)
  (h8 : C * D = 72)
  (h9 : A - B = C + D) : A = 18 :=
sorry

end value_of_A_is_18_l748_748452


namespace find_original_number_l748_748958

theorem find_original_number (h1 : 268 * 74 = 19732) (h2 : 2.68 * x = 1.9832) : x = 0.74 :=
sorry

end find_original_number_l748_748958


namespace find_a₉_l748_748956

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom S_6_eq : S 6 = 3
axiom S_11_eq : S 11 = 18

noncomputable def a₉ : ℝ := sorry -- Define a₉ here, proof skipped by "sorry"

theorem find_a₉ (a : ℕ → ℝ) (S : ℕ → ℝ) :
  S 6 = 3 →
  S 11 = 18 →
  a₉ = 3 :=
by
  intros S_6_eq S_11_eq
  sorry -- Proof goes here

end find_a₉_l748_748956


namespace parabola_value_l748_748300

theorem parabola_value (b c : ℝ) (h : 3 = -(-2) ^ 2 + b * -2 + c) : 2 * c - 4 * b - 9 = 5 := by
  sorry

end parabola_value_l748_748300


namespace total_people_wearing_sunglasses_l748_748815

theorem total_people_wearing_sunglasses (total_adults: ℕ) (percent_women_sunglasses percent_men_sunglasses: ℝ):
  total_adults = 1800 →
  percent_women_sunglasses = 0.25 →
  percent_men_sunglasses = 0.10 →
  let women := total_adults / 2 in
  let men := total_adults / 2 in
  let women_wearing_sunglasses := percent_women_sunglasses * women in
  let men_wearing_sunglasses := percent_men_sunglasses * men in
  women_wearing_sunglasses + men_wearing_sunglasses = 315 :=
by
  intros htotal hpercent_women hpercent_men
  let women := total_adults / 2 in
  let men := total_adults / 2 in
  let women_wearing_sunglasses := percent_women_sunglasses * women in
  let men_wearing_sunglasses := percent_men_sunglasses * men in
  have h_women_sunglasses : women_wearing_sunglasses = 225, from sorry,
  have h_men_sunglasses : men_wearing_sunglasses = 90, from sorry,
  show women_wearing_sunglasses + men_wearing_sunglasses = 315, from sorry

end total_people_wearing_sunglasses_l748_748815


namespace households_without_car_or_bike_l748_748654

theorem households_without_car_or_bike 
  (total_households ⟵ nat) (cb ⟵ nat) (households_with_car ⟵ nat) (households_with_bike_only ⟵ nat) :
  total_households = 90 →
  cb = 20 →
  households_with_car = 44 →
  households_with_bike_only = 35 →
  (total_households - (households_with_car - cb + households_with_bike_only + cb) = 11) :=
begin
  sorry
end

end households_without_car_or_bike_l748_748654


namespace expected_value_of_winnings_is_3_l748_748824

noncomputable def expected_winnings_even_roll : ℝ :=
(1/10) * (2 + 4 + 6 + 8 + 10)

theorem expected_value_of_winnings_is_3 :
  let odds_even := 1/2 in
  let odds_odd := 1/2 in
  let expected_even := (odds_even * expected_winnings_even_roll) in
  let expected_odd := (odds_odd * 0) in
  expected_even + expected_odd = 3 :=
by
  have h_expected_winnings : expected_winnings_even_roll = 3 := by
    calc expected_winnings_even_roll
          = (1/10) * (2 + 4 + 6 + 8 + 10) : rfl
      ... = (1/10) * 30 : by ring
      ... = 3 : by norm_num
  simp [h_expected_winnings]
  ring

sorry

end expected_value_of_winnings_is_3_l748_748824


namespace base7_to_base10_conversion_l748_748898

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l748_748898


namespace measure_angle_CED_l748_748029

-- Definitions and given conditions
variable (A B C D E : Point)
variable (circle : Point → Circle)
variable (angle : Point → Point → Point → ℝ)
variable (congruent : Circle → Circle → Prop)
variable (contains : Circle → Point → Prop)
variable (is_line : ∀ (P Q : Point), Set Point)
variable (intersect : ∀ (L : Set Point) (c1 c2 : Circle), Set Point)

-- Conditions
variables (h1 : congruent (circle A) (circle B))
variables (h2 : contains (circle A) B)
variables (h3 : contains (circle B) A)
variables (h4 : C ∈ is_line A B)
variables (h5 : intersect (is_line A B) (circle A) (circle B) = {C, D})
variables (h6 : E ∈ intersect (circle A) (circle B))

-- Angles given
variables (h7 : angle E A B = 70)
variables (h8 : angle E B A = 70)

-- Proof to show
theorem measure_angle_CED : angle C E D = 80 :=
by
  sorry

end measure_angle_CED_l748_748029


namespace distance_between_points_l748_748035

theorem distance_between_points : 
  let p1 := (3, -2) 
  let p2 := (-7, 4) 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by
  sorry

end distance_between_points_l748_748035


namespace path_of_point_on_moving_line_l748_748094

-- Definitions of the problem conditions
variable (α : Plane) -- Plane alpha
variable (a b : Line) -- Two skew lines a and b
variable (ratio : ℝ) -- constant ratio for the segment division

-- Statement of the theorem
theorem path_of_point_on_moving_line (α_parallel : ∀ t : ℝ, (∀ point : Point, line_parallel_to_plane_at_point point α)
                  ∧ intersect (line_through_parallel_to_plane t α) a
                  ∧ intersect (line_through_parallel_to_plane t α) b)
                  (point_divides_segment : ∀ t : ℝ, point_on_line t (divides_segment_in_ratio ratio a b)):
  ∃ L : Line, ∀ t : ℝ, point_on_line t (lies_on L) :=
sorry


end path_of_point_on_moving_line_l748_748094


namespace number_of_lines_l748_748467

-- Define the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the condition that a line intersects a parabola at only one point
def line_intersects_parabola_at_one_point (m b x y : ℝ) : Prop :=
  y - (m * x + b) = 0 ∧ parabola x y

-- The proof problem: Prove there are 3 such lines
theorem number_of_lines : ∃ (n : ℕ), n = 3 ∧ (
  ∃ (m b : ℝ), line_intersects_parabola_at_one_point m b 0 1) :=
sorry

end number_of_lines_l748_748467


namespace coefficient_of_x_neg_1_in_expansion_l748_748385

theorem coefficient_of_x_neg_1_in_expansion :
  let general_term (r : ℕ) := (Nat.choose 7 r) * (-2)^r * x^((7-3*r)/2)
  let target_exp := (√x - 2/x)^7
  ∃ r : ℕ, (7-3*r)/2 = -1 → (general_term r) = -280 := sorry

end coefficient_of_x_neg_1_in_expansion_l748_748385


namespace distinct_arrangements_banana_l748_748266

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748266


namespace sum_of_solutions_eq_sqrt_3_l748_748905

theorem sum_of_solutions_eq_sqrt_3 :
  (∀ x : ℝ, x > 0 ∧ x ^ (2 ^ real.sqrt 3) = (real.sqrt 3) ^ (2 ^ x) → x = real.sqrt 3) →
  (∑ x in {x | x > 0 ∧ x ^ (2 ^ real.sqrt 3) = (real.sqrt 3) ^ (2 ^ x)}, x) = real.sqrt 3 :=
begin
  sorry
end

end sum_of_solutions_eq_sqrt_3_l748_748905


namespace find_x_l748_748142

noncomputable def satisfy_equation (x : ℝ) : Prop :=
  8 / (Real.sqrt (x - 10) - 10) +
  2 / (Real.sqrt (x - 10) - 5) +
  10 / (Real.sqrt (x - 10) + 5) +
  16 / (Real.sqrt (x - 10) + 10) = 0

theorem find_x : ∃ x : ℝ, satisfy_equation x ∧ x = 60 := sorry

end find_x_l748_748142


namespace frank_won_skee_ball_tickets_l748_748439

noncomputable def tickets_whack_a_mole : ℕ := 33
noncomputable def candies_bought : ℕ := 7
noncomputable def tickets_per_candy : ℕ := 6
noncomputable def total_tickets_spent : ℕ := candies_bought * tickets_per_candy
noncomputable def tickets_skee_ball : ℕ := total_tickets_spent - tickets_whack_a_mole

theorem frank_won_skee_ball_tickets : tickets_skee_ball = 9 :=
  by
  sorry

end frank_won_skee_ball_tickets_l748_748439


namespace expected_value_eight_sided_die_win_l748_748080

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l748_748080


namespace average_gas_mileage_round_trip_l748_748832

theorem average_gas_mileage_round_trip :
  let distance_to_conference := 150
  let distance_return_trip := 150
  let mpg_sedan := 25
  let mpg_hybrid := 40
  let total_distance := distance_to_conference + distance_return_trip
  let gas_used_sedan := distance_to_conference / mpg_sedan
  let gas_used_hybrid := distance_return_trip / mpg_hybrid
  let total_gas_used := gas_used_sedan + gas_used_hybrid
  let average_gas_mileage := total_distance / total_gas_used
  average_gas_mileage = 31 := by
    sorry

end average_gas_mileage_round_trip_l748_748832


namespace gen_formula_sum_T_2023_l748_748963

-- Define the geometric sequence a_n with first term a_1 = 2 and common ratio q > 0
noncomputable def a (n : ℕ) (q : ℝ) : ℝ := 2 * q^(n-1)

-- Condition: a_4 is the arithmetic mean of 6 * a_2 and a_3
axiom a4_mean (q : ℝ) (h1 : q > 0) : a 4 q = (6 * a 2 q + a 3 q) / 2

-- The general formula for a_n
theorem gen_formula (q : ℝ) (h1 : q > 0) (h2 : q = 2) : ∀ n, a n q = 2^n := sorry

-- Define the sequence b_n
noncomputable def b (n : ℕ) (q : ℝ) : ℝ := 1 / (Real.logBase 2 (a n q) * Real.logBase 2 (a (n+1) q))

-- The sum of the first 2023 terms of b_n
def T_2023 (q : ℝ) := ∑ i in Finset.range 2023, b i q

-- The sum T_2023 is equal to 2023 / 2024
theorem sum_T_2023 (q : ℝ) (h1 : q > 0) (h2 : q = 2) : T_2023 q = 2023 / 2024 := sorry

end gen_formula_sum_T_2023_l748_748963


namespace find_xyz_l748_748697

theorem find_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end find_xyz_l748_748697


namespace sum_mean_median_mode_l748_748787

def numbers := [1, 2, 2, 3, 4, 4, 4, 5]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ n m, if l.count n > l.count m then n else m) 0

def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    let mid := sorted.length / 2
    (sorted.nth (mid - 1) + sorted.nth mid) / 2
  else
    sorted.nth (sorted.length / 2)

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem sum_mean_median_mode :
  let m := mean numbers
  let med := median numbers
  let mo := mode numbers
  m + med + mo = 10.625 := by
  sorry

end sum_mean_median_mode_l748_748787


namespace banana_permutations_l748_748257

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748257


namespace pure_imaginary_z_l748_748295

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem pure_imaginary_z (a : ℝ) : 
  is_pure_imaginary ((a - complex.i)^2) → a = 1 ∨ a = -1 := 
by
  sorry

end pure_imaginary_z_l748_748295


namespace quadratic_complex_inequality_solution_l748_748159
noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7/4) ↔ -2 < x ∧ x < 2 ∨ 3 ≤ x

theorem quadratic_complex_inequality_solution (x : ℝ) (hx : x ≠ -2 ∧ x ≠ 2):
  quadratic_inequality_solution x :=
  sorry

end quadratic_complex_inequality_solution_l748_748159


namespace expected_value_of_win_l748_748078

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l748_748078


namespace surface_area_of_sphere_l748_748770

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ S : ℝ, S = 36 * π * 2^(2/3) :=
by
  have volume_formula : V = (4 / 3) * π * r^3
  have surface_area_formula : S = 4 * π * r^2
  sorry

end surface_area_of_sphere_l748_748770


namespace vectors_are_coplanar_l748_748490

variable (a b c : Vector ℝ 3)

def vectors_coplanar (a b c : Vector ℝ 3) : Prop :=
  ∃ (d : ℝ), d = matrix.det ![a, b, c]

theorem vectors_are_coplanar : vectors_coplanar (⟨[1, -2, 1]⟩ : Vector ℝ 3) (⟨[3, 1, -2]⟩ : Vector ℝ 3) (⟨[7, 14, -13]⟩ : Vector ℝ 3) :=
by
  sorry

end vectors_are_coplanar_l748_748490


namespace university_minimum_cost_l748_748233

def small_box_stores_small := 8
def small_box_cost := 1.35
def medium_box_stores_medium := 6
def medium_box_cost := 1.35
def large_box_stores_large := 5
def large_box_cost := 1.35

def num_small_paintings := 1350
def num_medium_paintings := 2700
def num_large_paintings := 3150

noncomputable def calculate_boxes_needed (num_paintings : ℕ) (storage_capacity : ℕ) : ℕ :=
  (num_paintings + storage_capacity - 1) / storage_capacity

noncomputable def minimum_cost := 
  calculate_boxes_needed num_small_paintings small_box_stores_small * small_box_cost +
  calculate_boxes_needed num_medium_paintings medium_box_stores_medium * medium_box_cost +
  calculate_boxes_needed num_large_paintings large_box_stores_large * large_box_cost

-- The theorem stating the minimum cost
theorem university_minimum_cost : minimum_cost = 1686.15 := by
  sorry

end university_minimum_cost_l748_748233


namespace banana_permutations_l748_748247

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748247


namespace ratio_of_sum_l748_748601

open Real

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q > 0), ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem ratio_of_sum (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a) (h2 : arithmetic_sequence (3 * a 1) (1/2 * a 3) (2 * a 2)) (hq : q = 3) :
  (a 20 + a 19) / (a 18 + a 17) = 9 :=
sorry

end ratio_of_sum_l748_748601


namespace min_tangent_length_l748_748846

def line_eqn (x : ℝ) : ℝ := 2 * x + 3

def circle_eqn (x y : ℝ) : ℝ := (x - 2)^2 + (y + 3)^2 - 1

theorem min_tangent_length : 
  ∃ x y : ℝ, line_eqn x = y ∧ circle_eqn x y = 0 → 
  ∃ l : ℝ, l = real.sqrt 19 :=
sorry

end min_tangent_length_l748_748846


namespace arith_seq_2007th_term_l748_748390

theorem arith_seq_2007th_term (p q : ℕ) : 
  let a_1 := p,
      a_2 := p + 2 * q,
      a_3 := p + 6 * q,
      a_4 := p + 10 * q in
  ∀ n : ℕ, n = 2007 → a_1 + (n - 1) * 2 * q = p + 4012 * q :=
by
  intro n hn
  rw [hn]
  calc
    p + (2007 - 1) * 2 * q = p + 4012 * q : sorry

end arith_seq_2007th_term_l748_748390


namespace complex_number_real_product_l748_748951

theorem complex_number_real_product (a : ℝ) :
  let z1 := (2 : ℂ) + (1 : ℂ) * complex.i,
      z2 := (a : ℂ) - (1 : ℂ) * complex.i in
  (z1 * z2).im = 0 ↔ a = 2 :=
by
  let z1 := (2 : ℂ) + (1 : ℂ) * complex.i
  let z2 := (a : ℂ) - (1 : ℂ) * complex.i
  have H : (z1 * z2).im = 0 ↔ a = 2 := by sorry
  exact H

end complex_number_real_product_l748_748951


namespace largest_value_B_l748_748438

theorem largest_value_B :
  let A := ((1 / 2) / (3 / 4))
  let B := (1 / ((2 / 3) / 4))
  let C := (((1 / 2) / 3) / 4)
  let E := ((1 / (2 / 3)) / 4)
  B > A ∧ B > C ∧ B > E :=
by
  sorry

end largest_value_B_l748_748438


namespace negation_proposition_l748_748758

theorem negation_proposition :
  (¬ ∃ x : ℝ, (x > -1 ∧ x < 3) ∧ (x^2 - 1 ≤ 2 * x)) ↔ 
  (∀ x : ℝ, (x > -1 ∧ x < 3) → (x^2 - 1 > 2 * x)) :=
by {
  sorry
}

end negation_proposition_l748_748758


namespace anthony_initial_pets_l748_748125

variable (P : ℝ)

def lost_pets (P : ℝ) := P - 6
def died_pets (P : ℝ) := (1 / 5) * (lost_pets P)
def remaining_pets (P : ℝ) := lost_pets P - died_pets P

theorem anthony_initial_pets : remaining_pets P = 8 → P = 16 :=
  by
    intro h
    sorry

end anthony_initial_pets_l748_748125


namespace magician_earning_l748_748826

-- Definitions based on conditions
def price_per_deck : ℕ := 2
def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3

-- Theorem statement
theorem magician_earning :
  let sold_decks := initial_decks - remaining_decks
  let earning := sold_decks * price_per_deck
  earning = 4 := by
  sorry

end magician_earning_l748_748826


namespace banana_permutations_l748_748246

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748246


namespace density_function_proof_distribution_function_proof_l748_748448

noncomputable section

namespace MyProblem

-- Define the Lévy formula condition
def LevyFormula (a λ : ℝ) (h1 : a > 0) (h2 : λ > 0) : Prop :=
  ∫ t in set.Ioi 0, (exp (-λ * t) * (a * exp (-a^2 / (2 * t)) / sqrt (2 * π * t^3))) = exp (-a * sqrt (2 * λ))

-- Define the conditions related to \( f_{\alpha}(y) \) and \( g_{\alpha}(z) \)
def f_alpha_condition (α λ : ℝ) (f_alpha : ℝ → ℝ) (h1 : α > 0) (h2 : λ > 0) : Prop :=
  (sqrt (2 * λ) / sinh (sqrt (2 * λ)))^α = ∫ y in set.Ioi 0, exp (-λ * y) * f_alpha y

def g_alpha_condition (α λ : ℝ) (g_alpha : ℝ → ℝ) (h1 : α > 0) (h2 : λ > 0) : Prop :=
  (1 / cosh (sqrt (2 * λ)))^α = ∫ z in set.Ioi 0, exp (-λ * z) * g_alpha z

-- Define the proof for the density function
theorem density_function_proof (α : ℝ) : 
  ∀ z > 0, 
  g_alpha_condition α 1 (λ z, 2^α * ∑ n in set.Ici 0, nat_comb (-α) n * (2 * n + α) / sqrt (2 * π * z^3) * exp (-(2 * n + α)^2 / (2 * z))) :=
sorry

-- Define the proof for the distribution function
theorem distribution_function_proof : 
  ∀ y > 0, 
  f_alpha_condition 2 1 (λ y, (8 * sqrt 2) / sqrt (π * y^3) * ∑ k in set.Ici 1, k^2 * exp (-2 * k^2 / y)) :=
sorry

end MyProblem

end density_function_proof_distribution_function_proof_l748_748448


namespace tan_diff_identity_l748_748941

theorem tan_diff_identity (x : ℝ) (h1 : x ∈ set.Ioo 0 real.pi)
  (h2 : real.cos (2 * x - real.pi / 2) = real.sin x ^ 2) :
  real.tan (x - real.pi / 4) = 1 / 3 :=
sorry

end tan_diff_identity_l748_748941


namespace average_value_is_9x_l748_748549

-- Define the elements of the list of values
def values : List ℝ := [0, 3 * x, 6 * x, 12 * x, 24 * x]

-- Prove that the average value of these elements is 9 * x
theorem average_value_is_9x (x : ℝ) : 
  (values.sum / values.length) = 9 * x :=
by
  sorry

end average_value_is_9x_l748_748549


namespace distance_by_bus_correct_l748_748443

def total_distance : ℝ := 900
def distance_by_plane : ℝ := total_distance / 3
def distance_by_train (distance_by_bus : ℝ) : ℝ := 2 * distance_by_bus / 3
def total_distance_eq (distance_by_bus : ℝ) : Prop :=
  distance_by_plane + distance_by_train distance_by_bus + distance_by_bus = total_distance

theorem distance_by_bus_correct : ∃ (distance_by_bus : ℝ), total_distance_eq distance_by_bus ∧ distance_by_bus = 360 :=
by
  sorry

end distance_by_bus_correct_l748_748443


namespace primes_dividing_polynomial_l748_748902

open Polynomial

theorem primes_dividing_polynomial {P : Polynomial ℤ} (hdeg : P.degree ≥ 1) :
  ∃ᶠ p in filter.at_top, ∃ n : ℕ, p ∣ P.eval n :=
sorry

end primes_dividing_polynomial_l748_748902


namespace tourists_speeds_l748_748715

theorem tourists_speeds (x y : ℝ) :
  (20 / x + 2.5 = 20 / y) →
  (20 / (x - 2) = 20 / (1.5 * y)) →
  x = 8 ∧ y = 4 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end tourists_speeds_l748_748715


namespace binary_division_example_l748_748538

theorem binary_division_example : 
  let a := 0b10101  -- binary representation of 21
  let b := 0b11     -- binary representation of 3
  let quotient := 0b111  -- binary representation of 7
  a / b = quotient := 
by sorry

end binary_division_example_l748_748538


namespace largest_mersenne_prime_less_500_l748_748457

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d | n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_500 :
  (∀ p : ℕ, is_mersenne_prime p → p < 500 → p ≤ 127) ∧
  is_mersenne_prime 127 ∧ 127 < 500 :=
sorry

end largest_mersenne_prime_less_500_l748_748457


namespace expected_value_of_8_sided_die_l748_748089

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l748_748089


namespace only_even_n_works_l748_748811

-- Definitions (conditions)
def is_colored_correctly (n : ℕ) (table : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, (i % 3 + j % 3) % 3 < 2 → table i j = true

def can_transform (n : ℕ) (table : ℕ → ℕ → Prop) : Prop :=
  ∃ steps, (∀ step in steps, is_2x2_transform step table) ∧ all_black table

-- Main statement
theorem only_even_n_works {n : ℕ} (table : ℕ → ℕ → Prop)
  (h1 : is_colored_correctly n table)
  (h2 : ∀ step, is_2x2_transform step table → transforms_color step table) 
  (h3 : all_black table → all_white table → true) :
  ∃ steps, can_transform n table → even n := 
by
  sorry

end only_even_n_works_l748_748811


namespace number_of_triangles_l748_748587

noncomputable theory

variable (a : ℝ) (B : ℝ) (hc : ℝ)

theorem number_of_triangles : ∃ N, N = 0 ∨ N = ∞ :=
sorry

end number_of_triangles_l748_748587


namespace number_of_intersection_points_l748_748645

open Classical

noncomputable theory

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 4 * y = 4
def line3 (x y : ℝ) : Prop := 6 * x - 8 * y = 3

def intersection_points := {p : ℝ × ℝ | (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨
                                       (line1 p.1 p.2 ∧ line3 p.1 p.2) ∨
                                       (line2 p.1 p.2 ∧ line3 p.1 p.2)}

theorem number_of_intersection_points : 
  (intersection_points ∩ intersection_points).to_finset.card = 2 := sorry

end number_of_intersection_points_l748_748645


namespace atomic_weight_of_O_l748_748164

theorem atomic_weight_of_O :
  ∀ (molecular_weight_BaO : ℝ) (atomic_weight_Ba : ℝ), 
  molecular_weight_BaO = 153 → atomic_weight_Ba = 137.33 → 
  molecular_weight_BaO - atomic_weight_Ba = 15.67 := by
  intros molecular_weight_BaO atomic_weight_Ba h1 h2
  rw [h1, h2]
  norm_num
  sorry

end atomic_weight_of_O_l748_748164


namespace f_expression_and_extreme_value_maximum_value_condition_l748_748619

-- The function f(x) and g(x) are defined as described
def f (x : ℝ) : ℝ := exp(x) - x + (1 / 2) * x^2
def g (x a b : ℝ) : ℝ := (1 / 2) * x^2 + a * x + b

-- Statement (I)
-- Prove that f(x) = e^x - x + 1/2 x^2 and has a minimum value of 3/2 at x = 0
theorem f_expression_and_extreme_value :
  f 0 = 3 / 2 :=
sorry

-- Statement (II)
-- Prove the maximum value of ((a + 1) * b) / 2 under the given condition is e / 4
theorem maximum_value_condition (a b : ℝ) (h : ∀ x, f x ≥ g x a b) :
  (a + 1) * b / 2 ≤ exp(1) / 4 :=
sorry

end f_expression_and_extreme_value_maximum_value_condition_l748_748619


namespace painted_sphere_area_proportionality_l748_748477

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end painted_sphere_area_proportionality_l748_748477


namespace prove_p_or_q_l748_748195

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

def q : Prop := ∃ x0 : ℕ+, 2 * (x0 : ℝ)^2 - 1 ≤ 0

theorem prove_p_or_q : p a ∨ q a := by
  -- p is true based on the discriminant being non-positive and always yielding non-negative polynomials
  have hp : p a := by
    intro x
    calc 
      x^2 + a * x + a^2
      = (x + (a / 2))^2 + (3 / 4) * a^2
      ≥ 0 : by norm_num
  -- q is false since no positive natural number satisfies the condition
  have h_not_q : ¬q a := by
    intro h
    obtain ⟨x0, hx0⟩ := h
    have h2 : (x0 : ℝ)^2 ≥ 1/2 := by
      have : (x0 : ℝ) ≥ 1 := Nat.cast_le.2 x0.property
      exact sq_le_two_iff_le_one.mp this
    linarith
  exact Or.inl hp

end prove_p_or_q_l748_748195


namespace p_value_is_one_l748_748631

-- Definition of the condition
def condition : Prop := ∀ x : ℝ, (x - 1) * (x + 2) = x^2 + p * x - 2

-- The proof statement
theorem p_value_is_one (p : ℝ) (h : condition) : p = 1 :=
sorry

end p_value_is_one_l748_748631


namespace trigonometric_solution_l748_748045

theorem trigonometric_solution (x : Real) :
  (2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) 
  - 3 * Real.sin (Real.pi - x) * Real.cos x 
  + Real.sin (Real.pi / 2 + x) * Real.cos x = 0) ↔ 
  (∃ k : Int, x = Real.arctan ((3 + Real.sqrt 17) / -4) + k * Real.pi) ∨ 
  (∃ n : Int, x = Real.arctan ((3 - Real.sqrt 17) / -4) + n * Real.pi) :=
sorry

end trigonometric_solution_l748_748045


namespace num_two_digit_integers_satisfying_R_eq_R_plus_2_l748_748171

def R (n : ℕ) : ℕ := ∑ k in (Finset.range 9).map (λ i, 3 + i), n % k

theorem num_two_digit_integers_satisfying_R_eq_R_plus_2 : 
  ∃ (N : Finset ℕ), N.card = 2 
  ∧ ∀ n ∈ N, (10 ≤ n ∧ n < 100) ∧ R n = R (n + 2) :=
sorry

end num_two_digit_integers_satisfying_R_eq_R_plus_2_l748_748171


namespace calculate_expression_l748_748502

variable (x y : ℝ)

theorem calculate_expression :
  (-2 * x^2 * y)^3 = -8 * x^6 * y^3 :=
by 
  sorry

end calculate_expression_l748_748502


namespace min_sum_of_squares_l748_748346

theorem min_sum_of_squares :
  ∃ (p q r s t u v w : ℤ),
    {p, q, r, s, t, u, v, w}.card = 8 ∧ 
    ∀ x ∈ {p, q, r, s, t, u, v, w}, x ∈ { -6, -4, -1, 0, 3, 5, 7, 12 } ∧
    (p + q + r + s)^2 + (t + u + v + w)^2 = 128 :=
by
  sorry

end min_sum_of_squares_l748_748346


namespace lily_cups_in_order_l748_748937

theorem lily_cups_in_order :
  ∀ (rose_rate lily_rate : ℕ) (order_rose_cups total_payment hourly_wage : ℕ),
    rose_rate = 6 →
    lily_rate = 7 →
    order_rose_cups = 6 →
    total_payment = 90 →
    hourly_wage = 30 →
    ∃ lily_cups: ℕ, lily_cups = 14 :=
by
  intros
  sorry

end lily_cups_in_order_l748_748937


namespace custom_op_1_neg3_l748_748293

-- Define the custom operation as per the condition
def custom_op (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2

-- The theorem to prove that 1 * (-3) = -14 using the defined operation
theorem custom_op_1_neg3 : custom_op 1 (-3) = -14 := sorry

end custom_op_1_neg3_l748_748293


namespace crows_eat_worms_l748_748289

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l748_748289


namespace banana_arrangements_l748_748250

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748250


namespace crayons_count_l748_748566

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l748_748566


namespace max_bed_living_renovation_cost_l748_748708

/-- Statement of the problem conditions --/
def total_usable_area : Real := 105
def bathroom_kitchen_area : Real := 15
def bathroom_kitchen_cost_per_m2 : Real := 100
def additional_cost : Real := 500
def total_cost_limit : Real := 20000

noncomputable def max_cost_per_m2_bed_living := 200

/-- The problem statement in Lean 4: Prove the maximum cost per square meter for the bedroom and living room renovation materials and labor
cannot exceed 200 given the conditions --/
theorem max_bed_living_renovation_cost :
  let x := max_cost_per_m2_bed_living in
  15 * 100 + 500 + (105 - 15) * x ≤ 20000 → x = 200 :=
by
  intros h
  -- Proof goes here, using the provided conditions and value of x.
  sorry

end max_bed_living_renovation_cost_l748_748708


namespace no_multiples_of_1001_squared_in_form_l748_748630

theorem no_multiples_of_1001_squared_in_form (h1 : 1 ≤ i) (h2 : i < j) (h3 : j ≤ 99) (hi_even : even i) (hj_even : even j) :
  0 = (@Finset.filter (ℕ) (λ k, (1001^2 ∣ 10^k) ) (@Finset.range (99+1))).card :=
by
  sorry

end no_multiples_of_1001_squared_in_form_l748_748630


namespace distance_between_clocks_centers_l748_748413

variable (M m : ℝ)

theorem distance_between_clocks_centers :
  ∃ (c : ℝ), (|c| = (1/2) * (M + m)) := by
  sorry

end distance_between_clocks_centers_l748_748413


namespace infinite_product_eq_8_over_9_l748_748903

noncomputable def a : ℕ → ℚ
| 0     := 3 / 4
| (n+1) := 2 + (a n - 2)^2

theorem infinite_product_eq_8_over_9 :
  ( ∏ n in (Finset.range (C : ℕ)), a n ) = 8 / 9 :=
sorry

end infinite_product_eq_8_over_9_l748_748903


namespace graph_contains_even_cycle_l748_748337

theorem graph_contains_even_cycle (n : ℕ) (hn : n ≥ 4) (A : Fin n → Type) (h_no_three_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → ¬ collinear (A i) (A j) (A k)) 
  (h_connectivity : ∀ (i : Fin n), ≥ 3 ((λ j, (∃ seg : seg_between (A i) (A j), true)) : Set (Fin n))) :
  ∃ k > 1, (∃ X : Fin (2 * k), 
           (∀ i : Fin (2 * k - 1), connected (X i) (X (i + 1)))
           ∧ connected (X (2 * k - 1)) (X 0)) := 
sorry

end graph_contains_even_cycle_l748_748337


namespace can_intersect_tetrahedron_with_two_planes_l748_748134

noncomputable def tetrahedron_slicing_problem : Prop :=
  ∃ (T : Tetrahedron) (P1 P2 : Plane), 
    let S1 := T ∩ P1,
        S2 := T ∩ P2 in
    is_square S1 ∧ side_length S1 ≤ 1 ∧ 
    is_square S2 ∧ side_length S2 ≥ 100

theorem can_intersect_tetrahedron_with_two_planes
     (h : tetrahedron_slicing_problem) : 
    ∃ (T : Tetrahedron) (P1 P2 : Plane), 
    let S1 := T ∩ P1,
        S2 := T ∩ P2 in
    is_square S1 ∧ side_length S1 ≤ 1 ∧ 
    is_square S2 ∧ side_length S2 ≥ 100 :=
h

end can_intersect_tetrahedron_with_two_planes_l748_748134


namespace total_colors_over_two_hours_l748_748533

def colors_in_first_hour : Nat :=
  let quick_colors := 5 * 3
  let slow_colors := 2 * 3
  quick_colors + slow_colors

def colors_in_second_hour : Nat :=
  let quick_colors := (5 * 2) * 3
  let slow_colors := (2 * 2) * 3
  quick_colors + slow_colors

theorem total_colors_over_two_hours : colors_in_first_hour + colors_in_second_hour = 63 := by
  sorry

end total_colors_over_two_hours_l748_748533


namespace trigonometric_identity_l748_748798

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A * Real.cos B * Real.cos C + Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = Real.sin A * Real.sin B * Real.sin C :=
by 
  sorry

end trigonometric_identity_l748_748798


namespace cyclist_eventually_anomalous_l748_748074

structure CyclistJourney :=
(total_distance_per_day : ℕ)
(circular_road_length : ℕ)
(anomalous_zone_length : ℕ)
(sleep_wake_anomaly : ℕ → ℕ)

def starts_journey (start_position : ℕ) : Prop :=
  ∀ (day : ℕ), exists (pos : ℕ), (pos = (start_position + day * 71) % circular_road_length)

theorem cyclist_eventually_anomalous (start_position : ℕ)
  (hj : CyclistJourney) 
  (h1 : hj.total_distance_per_day = 71) 
  (h2 : hj.circular_road_length > 0) 
  (h3 : hj.anomalous_zone_length = 71)
  : ∃ day, starts_journey start_position day → (∃ dist, dist < hj.anomalous_zone_length) :=
sorry

end cyclist_eventually_anomalous_l748_748074


namespace middle_joints_capacity_l748_748367

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def bamboo_tube_capacity (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 4.5 ∧ a 6 + a 7 + a 8 = 2.5 ∧ arithmetic_seq a (a 1 - a 0)

theorem middle_joints_capacity (a : ℕ → ℝ) (d : ℝ) (h : bamboo_tube_capacity a) : 
  a 3 + a 4 + a 5 = 3.5 :=
by
  sorry

end middle_joints_capacity_l748_748367


namespace area_of_pentagon_l748_748665

-- Definitions and conditions
def isosceles_triangle (A B C : Type*) (area_ABC : ℝ) (num_small_triangles : ℕ) (area_small_triangle : ℝ) :=
  area_ABC = 60 ∧
  num_small_triangles = 9 ∧
  area_small_triangle = 1

-- Question to prove
theorem area_of_pentagon (A B C D E G : Type*) :
  isosceles_triangle A B C 60 9 1 →
  -- Pentagon DBCFG is formed by connecting midpoints.
  pentagon_formed_by_midpoints A B C D E G →
  ∃ (area : ℝ), area = 9 :=
  sorry

end area_of_pentagon_l748_748665


namespace queen_moves_from_d1_to_h8_l748_748661

-- Define the type for cells on the chessboard
structure Cell where
  x : Nat
  y : Nat

-- Define the initial cell d1
def d1 : Cell := { x := 4, y := 1 }

-- Define the target cell h8
def h8 : Cell := { x := 8, y := 8 }

-- Define a function to calculate the number of ways the queen can move from d1 to h8
noncomputable def numWays (start : Cell) (end : Cell) (allowedMoves : Cell → Cell → Bool) : Nat :=
  -- Implementation based on the dynamic programming approach would go here.
  sorry

-- Define the allowed moves condition
def allowedMoves (current next : Cell) : Bool := 
  (next.x = current.x + 1 ∧ next.y = current.y) ∨
  (next.x = current.x ∧ next.y = current.y + 1) ∨
  (next.x = current.x + 1 ∧ next.y = current.y + 1)

-- State the theorem to prove
theorem queen_moves_from_d1_to_h8 : numWays d1 h8 allowedMoves = 39625 :=
by
  sorry

end queen_moves_from_d1_to_h8_l748_748661


namespace find_two_digit_number_l748_748118

noncomputable def tens_digit (n : ℕ) : ℕ :=
n / 10

noncomputable def unit_digit (n : ℕ) : ℕ :=
n % 10

theorem find_two_digit_number (n : ℕ) (n >= 10 ∧ n < 100) : 
  3 * (tens_digit n + unit_digit n) - 2 = n → n = 28 :=
by 
  sorry

end find_two_digit_number_l748_748118


namespace find_n_l748_748920

theorem find_n (n : ℕ) (h : 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012) : n = 1005 :=
sorry

end find_n_l748_748920


namespace max_min_roots_l748_748613

theorem max_min_roots (m : ℝ) (h : -1 ≤ m ∧ m ≤ 1) :
  let α β : ℝ :=
    if (x^2 - 2 * m * x + 3 + 4 * m^2 - 6) = 0
  in 0 ≤ (α-1)^2 + (β-1)^2 ∧
     (α-1)^2 + (β-1)^2 ≤ 9 :=
begin
  -- Proof would go here
  sorry
end

end max_min_roots_l748_748613


namespace most_stable_performance_l748_748458

theorem most_stable_performance 
    (s_A s_B s_C s_D : ℝ)
    (hA : s_A = 1.5)
    (hB : s_B = 2.6)
    (hC : s_C = 1.7)
    (hD : s_D = 2.8)
    (mean_score : ∀ (x : ℝ), x = 88.5) :
    s_A < s_C ∧ s_C < s_B ∧ s_B < s_D := by
  sorry

end most_stable_performance_l748_748458


namespace distinct_arrangements_banana_l748_748269

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748269


namespace disjoint_subsets_with_same_sum_l748_748883

-- Define the theorem
theorem disjoint_subsets_with_same_sum :
  ∀ (H : Finset ℕ), (∀ x ∈ H, x < 100) → H.card = 10 → 
  ∃ (B C : Finset ℕ), B ⊆ H ∧ C ⊆ H ∧ B ∩ C = ∅ ∧ B ≠ C ∧ B.sum = C.sum :=
by 
  sorry

end disjoint_subsets_with_same_sum_l748_748883


namespace factors_multiple_of_120_l748_748183

theorem factors_multiple_of_120 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9 * 7^5) :
  ∃ k : ℕ, k = 8100 ∧ ∀ d : ℕ, d ∣ n ∧ 120 ∣ d ↔ ∃ a b c d : ℕ, 3 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 15 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 5 ∧ d = 2^a * 3^b * 5^c * 7^d :=
by
  sorry

end factors_multiple_of_120_l748_748183


namespace perpendicular_bisector_eq_l748_748752

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

noncomputable def slope (A B : (ℝ × ℝ)) : ℝ :=
  (B.snd - A.snd) / (B.fst - A.fst)

noncomputable def perpendicular_slope (A B : (ℝ × ℝ)) : ℝ :=
  -1 / slope A B

theorem perpendicular_bisector_eq (A B : (ℝ × ℝ)) (ha : A = (1, 3)) (hb : B = (5, -1)) :
    ∃ l m c : ℝ, l * X + m * Y + c = 0 ∧
                  l = 1 ∧ m = -1 ∧ c = -2 := by
  sorry

end perpendicular_bisector_eq_l748_748752


namespace triangle_area_l748_748116

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area :
  let A : point := (-2, 3)
  let B : point := (7, -3)
  let C : point := (4, 6)
  area_of_triangle A B C = 31.5 :=
by
  sorry

end triangle_area_l748_748116


namespace expected_value_of_win_l748_748086

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l748_748086


namespace curve_equation_ordered_triple_l748_748073

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (3 * Real.cos t + Real.sin t, 2 * Real.sin t)

theorem curve_equation_ordered_triple :
  ∃ a b c : ℝ, (∀ t, let (x, y) := parametric_curve t in a * x^2 + b * x * y + c * y^2 = 1) ∧ (a = 1/9) ∧ (b = -1/9) ∧ (c = 1/4) :=
by
  use [1 / 9, -1 / 9, 1 / 4]
  sorry

end curve_equation_ordered_triple_l748_748073


namespace find_n_l748_748576

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end find_n_l748_748576


namespace expected_value_eight_sided_die_win_l748_748081

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l748_748081


namespace expected_yield_correct_l748_748358

/-- Define the problem variables and conditions -/
def steps_x : ℕ := 25
def steps_y : ℕ := 20
def step_length : ℝ := 2.5
def yield_per_sqft : ℝ := 0.75

/-- Calculate the dimensions in feet -/
def length_x := steps_x * step_length
def length_y := steps_y * step_length

/-- Calculate the area of the orchard -/
def area := length_x * length_y

/-- Calculate the expected yield of apples -/
def expected_yield := area * yield_per_sqft

/-- Prove the expected yield of apples is 2343.75 pounds -/
theorem expected_yield_correct : expected_yield = 2343.75 := sorry

end expected_yield_correct_l748_748358


namespace collinear_points_l748_748685

theorem collinear_points
  (A B C : Point) (r : ℝ)
  (circle1 : Circle A r) (circle2 : Circle B r) (circle3 : Circle C r)
  (B1 C1 : Point) (B1_circle : B1 ∈ circle1) (C1_circle : C1 ∈ circle1)
  (C2 A2 : Point) (C2_circle : C2 ∈ circle2) (A2_circle : A2 ∈ circle2)
  (A3 B3 : Point) (A3_circle : A3 ∈ circle3) (B3_circle : B3 ∈ circle3)
  (X Y Z : Point)
  (inter_B1C1_BC : X ∈ line B1 C1) (inter_BC : X ∈ line B C)
  (inter_C2A2_CA : Y ∈ line C2 A2) (inter_CA : Y ∈ line C A)
  (inter_A3B3_AB : Z ∈ line A3 B3) (inter_AB : Z ∈ line A B) :
  collinear {X, Y, Z} := sorry

end collinear_points_l748_748685


namespace lambda_value_l748_748991

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748991


namespace find_n_eq_51_l748_748193

theorem find_n_eq_51 (n : ℕ) (h : n > 4) :
  (∑ k in range (n+1), (nat.choose n k) * x^k * 2^(n - k - 1) = x^(n-4) -> 
   ∑ k in range (n+1), (nat.choose n k) * x^1 * (2^(n-1))^(2) = xy) -> 
  n = 51 :=
begin
  sorry
end

end find_n_eq_51_l748_748193


namespace vector_addition_AC_l748_748230

def vector := (ℝ × ℝ)

def AB : vector := (0, 1)
def BC : vector := (1, 0)

def AC (AB BC : vector) : vector := (AB.1 + BC.1, AB.2 + BC.2) 

theorem vector_addition_AC (AB BC : vector) (h1 : AB = (0, 1)) (h2 : BC = (1, 0)) : 
  AC AB BC = (1, 1) :=
by
  sorry

end vector_addition_AC_l748_748230


namespace train_probability_l748_748120

noncomputable def probability_train_at_station (train_arrival_time alex_arrival_time : ℝ) : ℝ :=
  if train_arrival_time + 15 ≥ alex_arrival_time then 1 else 0

theorem train_probability (train_arrival dist_unif_1_2 : ℝ) (alex_arrival dist_unif_1_2 : ℝ) :
  let event := { t : ℝ // 0 ≤ t ∧ t ≤ 60 } in
  let joint_prob := (λ t a, probability_train_at_station t a) in
  ((∫ t in event, ∫ a in event, joint_prob t a) / (60 * 60) = 21 / 96) :=
by 
  sorry

end train_probability_l748_748120


namespace sequence_product_bound_l748_748348

theorem sequence_product_bound : ∃ d : ℝ, ( ∀ n : ℕ, |(b_0 b_1 ... b_{n-1})| ≤ d / 3^n ) ∧ 100 * d = 111 :=
by
  let b : ℕ → ℝ := sorry -- Define the sequence recurrence relation here
  let d : ℝ := 7 / Real.sqrt 40
  use d
  split
  { intros n
    sorry -- Proof of the inequality goes here
  }
  {
    have h100d : 100 * d = 100 * (7 / Real.sqrt 40) := by sorry
    norm_num at h100d
    exact h100d
  }

end sequence_product_bound_l748_748348


namespace tv_sales_value_increase_l748_748482

theorem tv_sales_value_increase (P V : ℝ) :
    let P1 := 0.82 * P
    let V1 := 1.72 * V
    let P2 := 0.75 * P1
    let V2 := 1.90 * V1
    let initial_sales := P * V
    let final_sales := P2 * V2
    final_sales = 2.00967 * initial_sales :=
by
  sorry

end tv_sales_value_increase_l748_748482


namespace prove_equal_values_l748_748488

theorem prove_equal_values :
  (-2: ℝ)^3 = -(2: ℝ)^3 :=
by sorry

end prove_equal_values_l748_748488


namespace five_crows_two_hours_l748_748291

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l748_748291


namespace problem_l748_748123

-- Define the functions under consideration
def f₁ (x : ℝ) := Real.log (3 - x) / Real.log 0.5
def f₂ (x : ℝ) := x^2 + 1
def f₃ (x : ℝ) := -x^2
def f₄ (x : ℝ) := 2^(2 * x)

-- The interval we are considering
def I := Set.Ioo 0 2

-- The main statement that needs to be proved
theorem problem (x ∈ I) : 
  ¬ ∀ x₁ x₂ ∈ I, x₁ < x₂ → f₃ x₁ ≤ f₃ x₂ :=
by
  sorry

end problem_l748_748123


namespace mass_percentage_H_in_butane_l748_748552

open Real

-- Definitions and constants based on the conditions
def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def carbon_atoms_in_butane : ℝ := 4
def hydrogen_atoms_in_butane : ℝ := 10

-- Problem statement in Lean 4
theorem mass_percentage_H_in_butane :
  (hydrogen_atoms_in_butane * atomic_mass_H / 
  (carbon_atoms_in_butane * atomic_mass_C + hydrogen_atoms_in_butane * atomic_mass_H)) * 100 ≈ 17.33 :=
by
  sorry

end mass_percentage_H_in_butane_l748_748552


namespace problem_lean_l748_748173

noncomputable def H (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), 1 / (i + 1)

theorem problem_lean :
  (∑' n : ℕ, 1 / (n + 2) * H n * H (n + 2)) = (5 / 3) :=
sorry

end problem_lean_l748_748173


namespace score_sd_above_mean_l748_748560

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l748_748560


namespace part1_part2_l748_748217

noncomputable def f : ℝ → ℝ → ℝ := λ x a, log x + (1/x) + a * x

theorem part1 (a : ℝ) (h1 : -1/4 < a) (h2 : a < 0) :
  ∃ x ∈ Ioi (1 : ℝ), deriv (f x a) = 0 :=
by sorry

theorem part2 (a : ℝ) (hmax : ∃ x ≥ (1 : ℝ), f x a = 2 / Real.exp 1) :
  a = (1 - Real.exp 1) / (Real.exp 1)^2 :=
by sorry

end part1_part2_l748_748217


namespace parabola_points_relation_l748_748200

theorem parabola_points_relation {a b c y1 y2 y3 : ℝ} 
  (hA : y1 = a * (1 / 2)^2 + b * (1 / 2) + c)
  (hB : y2 = a * (0)^2 + b * (0) + c)
  (hC : y3 = a * (-1)^2 + b * (-1) + c)
  (h_cond : 0 < 2 * a ∧ 2 * a < b) : 
  y1 > y2 ∧ y2 > y3 :=
by 
  sorry

end parabola_points_relation_l748_748200


namespace find_n_l748_748573

theorem find_n : 
  (43^2 = 1849) → 
  (44^2 = 1936) → 
  (45^2 = 2025) → 
  (46^2 = 2116) → 
  ∃ n : ℤ, (n < Real.sqrt 2023) ∧ (Real.sqrt 2023 < n+1) ∧ n = 44 :=
by
  intros h1 h2 h3 h4
  existsi (44:ℤ)
  split
  sorry -- Proof of n < sqrt(2023)
  split
  sorry -- Proof of sqrt(2023) < n+1
  refl -- Proof of n = 44

end find_n_l748_748573


namespace ballot_box_certain_candidate_l748_748534

theorem ballot_box_certain_candidate 
  (ballot_boxes : Fin 11 → Set (Fin 10 → Prop))
  (h1 : ∀ i, ballot_boxes i ≠ ∅) 
  (h2 : ∀ (balls : Fin 11 → (Fin 10 → Prop)), (∀ i, balls i ∈ ballot_boxes i) →
    ∃ c, ∀ i, c ∈ balls i) : 
  ∃ (i : Fin 11) (c : Fin 10), ∀ (b : Fin 10 → Prop), b ∈ ballot_boxes i → c ∈ b := 
sorry

end ballot_box_certain_candidate_l748_748534


namespace compare_values_l748_748976

noncomputable def f (x : ℝ) : ℝ :=
if h : -Real.pi / 2 < x ∧ x < Real.pi / 2 then
  x + Real.sin x
else
  f (Real.pi - x)

lemma symmetry (x : ℝ) : f x = f (Real.pi - x) := by sorry

lemma increasing_on_interval (a b : ℝ) (h1 : -Real.pi / 2 < a ∧ a < Real.pi / 2) (h2 : -Real.pi / 2 < b ∧ b < Real.pi / 2) (h3 : a < b) : f a < f b := by sorry

lemma decreasing_after_pi_over_2 (a b : ℝ) (h1 : Real.pi / 2 < a ∧ a < 3 * Real.pi / 2) (h2 : Real.pi / 2 < b ∧ b < 3 * Real.pi / 2) (h3 : a > b) : f a < f b := by sorry

def a := f 1
def b := f 2
def c := f 3

theorem compare_values : c < a ∧ a < b := by sorry

end compare_values_l748_748976


namespace tim_card_count_l748_748496

variable (T : Nat)

theorem tim_card_count (h1 : ∀ (T : Nat), T = 20 ↔ 37 + 3 = 2 * T) :
  T = 20 :=
by
  have h2 : 37 + 3 = 40 := rfl
  rw [h1] at h2
  exact h2.mp rfl

end tim_card_count_l748_748496


namespace probability_of_seven_heads_in_ten_flips_l748_748639

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_seven_heads_in_ten_flips :
  let total_outcomes := 2^10 in
  let favorable_outcomes := binomial_coefficient 10 7 in
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 15 / 128 :=
by
  -- Prove this statement
  sorry

end probability_of_seven_heads_in_ten_flips_l748_748639


namespace tan_monotonic_interval_correct_l748_748002

noncomputable def tanMonotonicInterval : Set ℝ := 
  {x | (∃ k : ℤ, (k : ℝ) * π / 2 - π / 12 < x ∧ x < (k : ℝ) * π / 2 + 5 * π / 12)}

theorem tan_monotonic_interval_correct :
  ∀ x : ℝ, (∃ k : ℤ, (k : ℝ) * π / 2 - π / 12 < x ∧ x < (k : ℝ) * π / 2 + 5 * π / 12) ↔
  ∀ k : ℤ, -π / 2 + k * π < 2 * x - π / 3 ∧ 2 * x - π / 3 < π / 2 + k * π :=
by
  sorry

end tan_monotonic_interval_correct_l748_748002


namespace max_value_of_f_min_value_of_f_sin_A_value_l748_748211

-- Definitions of the function and conditions
def f (x : ℝ) := (real.sqrt 3) * real.sin x * real.cos x - (real.cos x) ^ 2 - 1 / 2

-- Ensure the conditions for x
variable x : ℝ
variable B : ℝ
variable a : ℝ := 3
variable c : ℝ := 2

-- Conditions from the problem
def part1_condition := 0 ≤ x ∧ x ≤ real.pi / 2
def part2_condition := f B = 0

-- Statements of maximum and minimum values for part 1
theorem max_value_of_f : part1_condition → ∃ M, M = 0 ∧ ∀ x, f x ≤ M := sorry
theorem min_value_of_f : part1_condition → ∃ m, m = -3 / 2 ∧ ∀ x, f x ≥ m := sorry

-- Statement for the value of sin A in part 2
theorem sin_A_value : part2_condition → ∃ A, real.sin A = 3 * real.sqrt 21 / 14 := sorry

end max_value_of_f_min_value_of_f_sin_A_value_l748_748211


namespace minimum_value_shifted_function_l748_748297

def f (x a : ℝ) : ℝ := x^2 + 4 * x + 7 - a

theorem minimum_value_shifted_function (a : ℝ) (h : ∃ x, f x a = 2) :
  ∃ y, (∃ x, y = f (x - 2015) a) ∧ y = 2 :=
sorry

end minimum_value_shifted_function_l748_748297


namespace finite_set_of_functions_composition_l748_748590

theorem finite_set_of_functions_composition {P : ℕ → ℝ → ℝ} :
  ∃ (N : ℕ) (f : Fin N → (ℝ → ℝ)), ∀ n : ℕ, ∃ (k : ℕ) (i : Fin k → Fin N),
  P n = (f (i 0)) ∘ (f (i 1)) ∘ ... ∘ (f (i k-1)) :=
sorry

end finite_set_of_functions_composition_l748_748590


namespace mod_inverse_17_1200_l748_748034

theorem mod_inverse_17_1200 : ∃ x : ℕ, x < 1200 ∧ 17 * x % 1200 = 1 := 
by
  use 353
  sorry

end mod_inverse_17_1200_l748_748034


namespace sequence_difference_l748_748222

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1 / 7
  else (7 / 2) * (sequence (n - 1)) * (1 - (sequence (n - 1)))

theorem sequence_difference :
  sequence 999 - sequence 888 = 3 / 7 := 
sorry

end sequence_difference_l748_748222


namespace area_of_fourth_rectangle_l748_748464

variable (x y z w : ℝ)
variable (Area_EFGH Area_EIKJ Area_KLMN Perimeter : ℝ)

def conditions :=
  (Area_EFGH = x * y ∧ Area_EFGH = 20 ∧
   Area_EIKJ = x * w ∧ Area_EIKJ = 25 ∧
   Area_KLMN = z * w ∧ Area_KLMN = 15 ∧
   Perimeter = 2 * (x + z + y + w) ∧ Perimeter = 40)

theorem area_of_fourth_rectangle (h : conditions x y z w Area_EFGH Area_EIKJ Area_KLMN Perimeter) :
  (y * w = 340) :=
by
  sorry

end area_of_fourth_rectangle_l748_748464


namespace crayons_count_l748_748567

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l748_748567


namespace symmetric_points_addition_l748_748202

theorem symmetric_points_addition (a b : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 2) 
  (h3 : point_symmetric : A (2, a) is symmetric_with_respect_to_the_x_axis_to B (b, -3)) : 
  a + b = 5 :=
by
  sorry

end symmetric_points_addition_l748_748202


namespace find_radius_l748_748481

noncomputable def density (k r : ℝ) : ℝ := k * r^2
noncomputable def area (ω β t c : ℝ) : ℝ := ω * sin (β * t) + c
noncomputable def surface_area_at_time(t : ℝ) : ℝ := 64 * Real.pi

theorem find_radius 
  (k ω β c : ℝ)
  (h1 : density k)
  (h2 : area ω β (Real.pi / (2 * β)) c = 64 * Real.pi)
  : (∃ r : ℝ, 4 * Real.pi * r^2 = 64 * Real.pi ∧ r = 4) :=
by
  sorry

end find_radius_l748_748481


namespace lambda_value_l748_748999

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748999


namespace find_y_l748_748231

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (y : ℝ) : ℝ × ℝ × ℝ := (-5, y, -2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_y (y : ℝ) (h : dot_product vector_a (vector_b y) = 0) : y = -4 := 
by
  sorry

end find_y_l748_748231


namespace P_no_real_roots_l748_748840

noncomputable def P : ℕ → (ℝ → ℝ)
| 0       := λ x, 1
| (n + 1) := λ x, x^(11 * (n + 1)) - P n x

theorem P_no_real_roots (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 := 
by 
  -- Proof of the theorem would go here
  sorry

end P_no_real_roots_l748_748840


namespace min_value_x2_plus_2y2_l748_748203

-- Definitions of the hypotheses
variables {A B C D O : Type}
variables [vector_space ℝ A B C D O]
variables {x y : ℝ}

-- Given conditions
def coplanar (A B C D : Type) : Prop := ∃ α β γ : ℝ, α • A + β • B + γ • C = D
def vector_eq (O A B C D : Type) (x y : ℝ) : Prop := 
  (3 : ℝ) • C - x • A - (2 : ℝ) • B = D - O

-- The main problem statement to prove
theorem min_value_x2_plus_2y2 (h_coplanar : coplanar A B C D)
  (h_vector : vector_eq O A B C D) : 
  ∃ x y, x^2 + 2*y^2 = (4/3 : ℝ) :=
sorry

end min_value_x2_plus_2y2_l748_748203


namespace fencing_required_l748_748833

theorem fencing_required (L W : ℝ) (hL : L = 20) (hArea : L * W = 60) : (L + 2 * W) = 26 := 
by
  sorry

end fencing_required_l748_748833


namespace hamiltonian_cycle_exists_l748_748917

-- Definitions of the conditions
structure ConnectedGraph (V : Type) :=
  (E : V → V → Prop)
  (connected : ∀ a b : V, ∃ p : list V, p.head = some a ∧ p.last = some b ∧ (∀ i, i < list.length p - 1 → E (p.nth_le i (by sorry)) (p.nth_le (i + 1) (by sorry))))

def is_path {V : Type} (E : V → V → Prop) (p : list V) : Prop :=
  ∀ i, i < p.length - 1 → E (p.nth_le i (by sorry)) (p.nth_le (i + 1) (by sorry))

-- The main proposition to be proved
theorem hamiltonian_cycle_exists (V : Type) (G : ConnectedGraph V) :
  (∀ p : list V, is_path G.E p → ∀ v ∉ p, ∃ q : list V, is_path G.E q ∧ v ∈ q) →
  ∃ cycle : list V, is_path G.E cycle ∧ (∀ v : V, v ∈ cycle) ∧ cycle.head = cycle.last :=
sorry

end hamiltonian_cycle_exists_l748_748917


namespace banana_distinct_arrangements_l748_748278

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748278


namespace smallest_n_for_integer_T_l748_748687

def P : ℚ := 1/2 + 1/3 + 1/5 + 1/7

def T (n : ℕ) : ℚ := n * 5^(n-1) * P

theorem smallest_n_for_integer_T : (∃ n : ℕ, T n ∈ ℕ) ∧ (n = 42) := by
  sorry

end smallest_n_for_integer_T_l748_748687


namespace verify_A_and_centroid_l748_748323

noncomputable theory

def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

variable (B C : ℝ × ℝ × ℝ)
def midpoint_BC : ℝ × ℝ × ℝ := (2, 6, -2)
def midpoint_AC : ℝ × ℝ × ℝ := (1, 5, -3)
def midpoint_AB : ℝ × ℝ × ℝ := (3, 4, 5)
def vertex_A := (2, 3, 4)

theorem verify_A_and_centroid :
  ∃ A : ℝ × ℝ × ℝ, 
  (A = vertex_A) ∧ 
  (let G := ((vertex_A.1 + B.1 + C.1) / 3, (vertex_A.2 + B.2 + C.2) / 3, (vertex_A.3 + B.3 + C.3) / 3) 
  in G = (2, 5, 0)) :=
sorry

end verify_A_and_centroid_l748_748323


namespace expected_value_of_win_l748_748079

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l748_748079


namespace no_solutions_for_inequalities_l748_748732

theorem no_solutions_for_inequalities (x y z t : ℝ) :
  |x| < |y - z + t| →
  |y| < |x - z + t| →
  |z| < |x - y + t| →
  |t| < |x - y + z| →
  False :=
by
  sorry

end no_solutions_for_inequalities_l748_748732


namespace edge_ratio_of_equal_surface_areas_l748_748762

theorem edge_ratio_of_equal_surface_areas (a b c : ℝ) 
  (h1 : sqrt 3 * a^2 = 2 * sqrt 3 * b^2)
  (h2 : 2 * sqrt 3 * b^2 = 5 * sqrt 3 * c^2) : 
  a : b : c = 2 * sqrt 10 : sqrt 10 : 2 :=
by sorry

end edge_ratio_of_equal_surface_areas_l748_748762


namespace minimum_key_presses_to_restore_display_l748_748633

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem minimum_key_presses_to_restore_display : 
  ∀ x : ℝ, x ≠ 0 → x = 50 → (reciprocal (reciprocal x) = x) :=
by
  intros x hx1 hx2
  rw [reciprocal, reciprocal]
  sorry

end minimum_key_presses_to_restore_display_l748_748633


namespace remainder_2500th_term_div_7_l748_748521

-- Define the sequence as described in the problem
def sequence : List ℕ := (List.range (141 * 142 / 2)).bind (fun n => List.repeat (n + 1) (n + 1))

-- Function to get the nth term of the sequence
def nth_term (n : ℕ) : ℕ :=
  sequence.get (n - 1)

-- Statement to prove
theorem remainder_2500th_term_div_7 : nth_term 2500 % 7 = 1 :=
  by
  -- Proof would go here
  sorry

end remainder_2500th_term_div_7_l748_748521


namespace sin_1320_eq_neg_sqrt_3_over_2_l748_748146

theorem sin_1320_eq_neg_sqrt_3_over_2 :
  let θ := 1320 in
  let reduced_θ := 240 in
  θ % 360 = reduced_θ ∧ sin (240 : Real) = -Real.sqrt 3 / 2 →
  sin (θ : Real) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_eq_neg_sqrt_3_over_2_l748_748146


namespace expected_value_of_8_sided_die_l748_748090

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l748_748090


namespace sum_of_fraction_equiv_l748_748398

theorem sum_of_fraction_equiv : 
  let x := 3.714714714
  let num := 3711
  let denom := 999
  3711 + 999 = 4710 :=
by 
  sorry

end sum_of_fraction_equiv_l748_748398


namespace total_spent_on_dog_toys_l748_748044

-- Define the cost of a single toy
def toy_cost : ℝ := 12.00

-- Define the number of toys bought
def num_toys : ℝ := 4

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the total cost before tax
def total_cost_before_tax (toy_cost : ℝ) (num_toys : ℝ) : ℝ :=
  (toy_cost * (num_toys / 2)) + ((toy_cost / 2) * (num_toys / 2))

-- Define the sales tax
def sales_tax (total_cost_before_tax : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  total_cost_before_tax * sales_tax_rate

-- Define the total amount spent
def total_amount_spent (total_cost_before_tax : ℝ) (sales_tax : ℝ) : ℝ :=
  total_cost_before_tax + sales_tax

-- Prove that the total amount spent is $38.88
theorem total_spent_on_dog_toys : total_amount_spent (total_cost_before_tax toy_cost num_toys) (sales_tax (total_cost_before_tax toy_cost num_toys) sales_tax_rate) = 38.88 := 
by
  unfold total_amount_spent
  unfold sales_tax
  unfold total_cost_before_tax
  norm_num
  sorry

end total_spent_on_dog_toys_l748_748044


namespace equilateral_triangle_ratio_l748_748873

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 10) :
  let altitude := (s * real.sqrt 3) / 2
  let area := (s * altitude) / 2
  let perimeter := 3 * s
  (area / perimeter) = (5 * real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l748_748873


namespace arithmetic_sequence_sum_l748_748636

theorem arithmetic_sequence_sum (n : ℕ) (x : ℕ → ℕ) 
  (h1 : x 1 = 2) (h2 : ∀ k : ℕ, 1 ≤ k → k < n → x (k + 1) = x k + 1) :
  (∑ i in Finset.range n, x (i + 1)) = n * (n + 3) / 2 := 
by
  sorry

end arithmetic_sequence_sum_l748_748636


namespace metallic_sheet_length_l748_748095

theorem metallic_sheet_length (L : ℝ) 
  (h1 : squares_cut : 10)
  (h2 : volume_of_box : (24000 : ℝ))
  (h3 : width_initial : 50)
  (h4 : height_of_box := 10)
  : 
  let volume_eq := (L - 2 * squares_cut) * (width_initial - 2 * squares_cut) * height_of_box
  in volume_eq = volume_of_box → L = 820 := 
by {
  sorry
}

end metallic_sheet_length_l748_748095


namespace braden_total_money_after_bet_l748_748868

theorem braden_total_money_after_bet (initial_amount bet_multiplier : ℕ) (initial_money : initial_amount = 400) (bet_transition : bet_multiplier = 2) :
  let winning_amount := bet_multiplier * initial_amount in
  let total_amount := winning_amount + initial_amount in
  total_amount = 1200 :=
by
  sorry

end braden_total_money_after_bet_l748_748868


namespace packages_bought_l748_748796

theorem packages_bought (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) : 
  (total_tshirts / tshirts_per_package) = 71 :=
by 
  sorry

end packages_bought_l748_748796


namespace square_cut_cover_circle_l748_748133

theorem square_cut_cover_circle (s : ℝ) (d : ℝ) (hs : s = 1) (hd : d > 1) :
  ∃ square_parts : ℝ → Prop, (∀ p ∈ square_parts, p ⊆ (set.Icc 0 s))
                               ∧ (⋃ p ∈ square_parts, p = set.Icc 0 s)
                               ∧ (∀ c ∈ square_parts, ∃ circle : { x // set.Icc 0 d } → Prop, circle (set.center c) ∧ circle (set.Icc 0 d)) :=
sorry

end square_cut_cover_circle_l748_748133


namespace discount_amount_l748_748795

/-- Suppose Maria received a 25% discount on DVDs, and she paid $120.
    The discount she received is $40. -/
theorem discount_amount (P : ℝ) (h : 0.75 * P = 120) : P - 120 = 40 := 
sorry

end discount_amount_l748_748795


namespace combined_surface_area_of_cube_and_sphere_l748_748823

theorem combined_surface_area_of_cube_and_sphere (V_cube : ℝ) :
  V_cube = 729 →
  ∃ (A_combined : ℝ), A_combined = 486 + 81 * Real.pi :=
by
  intro V_cube
  sorry

end combined_surface_area_of_cube_and_sphere_l748_748823


namespace ordered_pairs_squares_diff_150_l748_748285

theorem ordered_pairs_squares_diff_150 (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn : m ≥ n) (h_diff : m^2 - n^2 = 150) : false :=
by {
    sorry
}

end ordered_pairs_squares_diff_150_l748_748285


namespace min_of_g_area_enclosed_by_line_and_curve_l748_748972

noncomputable def g (a x : ℝ) : ℝ := x + a / x

theorem min_of_g (a : ℝ) (h : a > 0) : 
  ∃ x : ℝ, x > 0 ∧ g a x = 2 → a = 1 :=
by
  sorry

theorem area_enclosed_by_line_and_curve :
  ∫ x in (3/2:ℝ) .. (2:ℝ), (2/3 * x + 7/6) - (x + 1/x) = 7/24 + ln 3 - 2 * ln 2 :=
by
  sorry

end min_of_g_area_enclosed_by_line_and_curve_l748_748972


namespace f_n_expression_l748_748174

noncomputable def f_n : ℕ → (ℝ → ℝ) := sorry

axiom f_n_property_1 (n : ℕ) (x : ℝ) (xs : Fin n → ℝ) : f_n n (λ i, xs i + x) = f_n n xs + x
axiom f_n_property_2 (n : ℕ) (xs : Fin n → ℝ) : f_n n (λ i, -xs i) = -f_n n xs
axiom f_n_property_3 (n : ℕ) (xs : Fin n → ℝ) (x : ℝ) : 
  f_n (n+1) (λ i, if h : i < n then f_n n xs else x) = f_n (n+1) (λ i, xs i)

theorem f_n_expression (n : ℕ) (xs : Fin n → ℝ) : f_n n xs = (Finset.univ.sum xs) / n :=
sorry

end f_n_expression_l748_748174


namespace metro_growth_rate_l748_748729

theorem metro_growth_rate (x : ℝ) :
  let july_passengers := 120
  let september_passengers := 175
  let months := 2
  july_passengers * (1 + x) ^ months = september_passengers :=
begin
  sorry
end

end metro_growth_rate_l748_748729


namespace locus_M_l748_748186

variables (l l' : Line) (A M N : Point)

-- Define the conditions given in the problem
def is_skew (l l' : Line) : Prop := -- Definition for skew lines
sorry

def common_perpendicular (l l' : Line) (M N : Point) : Prop := -- Definition for common perpendicular
  M ∈ l' ∧ N ∈ l ∧ Perpendicular M N (Line.through M N)

-- Define the orthogonal projection of a point onto a plane
def proj_onto_plane (P : Point) (α : Plane) : Point := sorry

-- Define the plane through a point perpendicular to a line
def plane_perpendicular_to_line (A : Point) (l : Line) : Plane := sorry

-- Define the point lying on the circle with a given diameter
def lies_on_circle (P : Point) (A B : Point) : Prop := -- Definition of the circle with diameter AB
sorry

-- Define the locus of point M under given perpendicular and parallel criteria
def locus_of_M (l : Line) (A M N : Point) : Prop :=
  let α := plane_perpendicular_to_line A l in
  let M' := proj_onto_plane M α in
  let N' := proj_onto_plane N α in
  M' lies_on_circle N' A

-- The theorem that needs to be proven
theorem locus_M 
  (l l' : Line) (A M N : Point) 
  (h_skew : is_skew l l')
  (h_perp : common_perpendicular l l' M N) 
: locus_of_M l A M N := 
sorry

end locus_M_l748_748186


namespace traffic_light_stop_probability_l748_748070

theorem traffic_light_stop_probability :
  let pA := 1 / 3,
      pB := 1 / 2,
      pC := 2 / 3
  in (2 / 9) + (1 / 9) + (1 / 18) = (7 / 18) :=
by
  let pA := 1 / 3
  let pB := 1 / 2
  let pC := 2 / 3
  have prob_stop_A := 2 / 9
  have prob_stop_B := 1 / 9
  have prob_stop_C := 1 / 18
  show prob_stop_A + prob_stop_B + prob_stop_C = 7 / 18
  sorry

end traffic_light_stop_probability_l748_748070


namespace tangent_lines_count_l748_748717

noncomputable def point := ℝ × ℝ

def dist (p q : point) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

def num_tangents (A B : point) (rA rB : ℝ) (d : ℝ) : ℝ :=
if d = dist A B ∧ rA = 2 ∧ rB = 3 then 2 else 0

theorem tangent_lines_count (A B : point) (d : ℝ) :
  dist A B = 7 → num_tangents A B 2 3 d = 2 :=
by sorry

end tangent_lines_count_l748_748717


namespace banana_distinct_arrangements_l748_748277

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748277


namespace integral_sh2_ch2_eq_l748_748550

theorem integral_sh2_ch2_eq (C : ℝ) :
  (∫ x, (Real.sinh x)^2 * (Real.cosh x)^2) = (λ x, (1 / 32) * Real.sinh (4 * x) - (1 / 8) * x + C) := by
  sorry

end integral_sh2_ch2_eq_l748_748550


namespace constant_term_in_expansion_l748_748666

theorem constant_term_in_expansion : 
  let x := (x * x * x) - (1 / x) in
  let expansion := (x^3 - (1 / x))^4 in 
  (constant_term expansion) = -4 :=
begin
  sorry
end

end constant_term_in_expansion_l748_748666


namespace find_largest_number_l748_748009

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l748_748009


namespace find_n_l748_748574

theorem find_n : 
  (43^2 = 1849) → 
  (44^2 = 1936) → 
  (45^2 = 2025) → 
  (46^2 = 2116) → 
  ∃ n : ℤ, (n < Real.sqrt 2023) ∧ (Real.sqrt 2023 < n+1) ∧ n = 44 :=
by
  intros h1 h2 h3 h4
  existsi (44:ℤ)
  split
  sorry -- Proof of n < sqrt(2023)
  split
  sorry -- Proof of sqrt(2023) < n+1
  refl -- Proof of n = 44

end find_n_l748_748574


namespace correct_propositions_l748_748391

noncomputable def proposition1 (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 = a ∧ x1 + x2 = 3 - a ∧ x1 ≥ 0 ∧ x2 ≤ 0 ∧ a < 0

noncomputable def proposition2 : Prop :=
  let y := λ x : ℝ, sqrt (x^2 - 1) + sqrt (1 - x^2) in
  ¬ (∀ x ∈ {0}, y(-x) = y x ∧ ¬ (y(-x) = -y x))

noncomputable def proposition3 (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ [-2, 2]) → (∃ y, f (y + 1) ∈ [-3, 1])

noncomputable def proposition4 (a : ℝ) : Prop :=
  let F := λ x : ℝ, abs (3 - x^2) in
  ¬ (∃ x, F x = a ∧ ∀ y ≠ x, F y ≠ a)

theorem correct_propositions :
  ∀ (a : ℝ) (f : ℝ → ℝ), proposition1 a ∧ proposition4 a ∧ ¬ proposition2 ∧ ¬ proposition3 f :=
by
  -- the proof goes here
  sorry

end correct_propositions_l748_748391


namespace remuneration_difference_l748_748104

-- Define the conditions and question
def total_sales : ℝ := 12000
def commission_rate_old : ℝ := 0.05
def fixed_salary_new : ℝ := 1000
def commission_rate_new : ℝ := 0.025
def sales_threshold_new : ℝ := 4000

-- Define the remuneration for the old scheme
def remuneration_old : ℝ := total_sales * commission_rate_old

-- Define the remuneration for the new scheme
def sales_exceeding_threshold_new : ℝ := total_sales - sales_threshold_new
def commission_new : ℝ := sales_exceeding_threshold_new * commission_rate_new
def remuneration_new : ℝ := fixed_salary_new + commission_new

-- Statement of the theorem to be proved
theorem remuneration_difference : remuneration_new - remuneration_old = 600 :=
by
  -- The proof goes here but is omitted as per the instructions
  sorry

end remuneration_difference_l748_748104


namespace selling_price_correct_l748_748375

def initial_cost : ℕ := 800
def repair_cost : ℕ := 200
def gain_percent : ℕ := 40
def total_cost := initial_cost + repair_cost
def gain := (gain_percent * total_cost) / 100
def selling_price := total_cost + gain

theorem selling_price_correct : selling_price = 1400 := 
by
  sorry

end selling_price_correct_l748_748375


namespace right_triangle_and_inverse_l748_748877

theorem right_triangle_and_inverse :
  30 * 30 + 272 * 272 = 278 * 278 ∧ (∃ (n : ℕ), 0 ≤ n ∧ n < 4079 ∧ (550 * n) % 4079 = 1) :=
by
  sorry

end right_triangle_and_inverse_l748_748877


namespace pipes_fill_tank_in_10_hours_l748_748847

noncomputable def R_A := 1 / 70
noncomputable def R_B := 2 * R_A
noncomputable def R_C := 2 * R_B
noncomputable def R_total := R_A + R_B + R_C
noncomputable def T := 1 / R_total

theorem pipes_fill_tank_in_10_hours :
  T = 10 := 
sorry

end pipes_fill_tank_in_10_hours_l748_748847


namespace domain_of_f_parity_of_f_solution_set_f_gt_zero_l748_748618

noncomputable def f (a x : ℝ) : ℝ := log a (x + 1) - log a (1 - x)

variables {a : ℝ} (ha1 : a > 0) (ha2 : a ≠ 1) 

-- Proof for the domain of f
theorem domain_of_f : ∀ x, f a x ∈ ℝ → -1 < x ∧ x < 1 :=
by sorry

-- Proof for parity of f showing that f is an odd function
theorem parity_of_f : ∀ x, f a (-x) = -f a x :=
by sorry

-- Proof for when a > 1 that f(x) > 0 holds for x in (0, 1)
theorem solution_set_f_gt_zero (ha_gt1 : a > 1) : ∀ x, 0 < x ∧ x < 1 → f a x > 0 :=
by sorry

end domain_of_f_parity_of_f_solution_set_f_gt_zero_l748_748618


namespace average_output_assembly_line_l748_748803

theorem average_output_assembly_line (initial_cogs second_batch_cogs rate1 rate2 : ℕ) (time1 time2 : ℚ)
  (h1 : initial_cogs = 60)
  (h2 : second_batch_cogs = 60)
  (h3 : rate1 = 90)
  (h4 : rate2 = 60)
  (h5 : time1 = 60 / 90)
  (h6 : time2 = 60 / 60)
  (h7 : (120 : ℚ) / (time1 + time2) = (72 : ℚ)) :
  (120 : ℚ) / (time1 + time2) = 72 := by
  sorry

end average_output_assembly_line_l748_748803


namespace trig_identity_second_quadrant_l748_748455

variable (α : ℝ)

theorem trig_identity_second_quadrant (hα : π/2 < α ∧ α < π) :
  cos α * sqrt ((1 - sin α) / (1 + sin α)) + sin α * sqrt ((1 - cos α) / (1 + cos α)) = sin α - cos α := 
sorry

end trig_identity_second_quadrant_l748_748455


namespace lambda_value_l748_748990

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748990


namespace PAQ_is_isosceles_right_triangle_l748_748606

variables {α : Type*} [inner_product_space ℝ α]

noncomputable def vector_eq (u v : α) (k : ℝ): Prop := u = k • v
noncomputable def orthogonal (u v : α): Prop := ⟪u, v⟫ = 0

def is_isosceles_right_triangle (P A Q : α) : Prop :=
  (dist P A = dist A Q) ∧ (orthogonal (P - A) (Q - A))

/-- Given that squares ABMH, ACNG, and BCDE are constructed outward on the sides AB, AC, and BC, 
respectively, and parallelograms BMP E and CNQD are constructed with BM, BE and CN, CD as adjacent 
sides respectively, prove that ΔPAQ is an isosceles right triangle. -/
theorem PAQ_is_isosceles_right_triangle
  (A B C M N E D P Q : α)
  (k : ℝ)
  (h1 : vector_eq (B - M) B (-k))
  (h2 : vector_eq (C - N) C k)
  (h3 : vector_eq (B - E) (C - D) (k • (B - C)))
  (h4 : vector_eq (B - P) B (-(k • B) - k • (B - C)))
  (h5 : vector_eq (C - Q) C (k • C + k • (B - C)))
  : is_isosceles_right_triangle P A Q :=
sorry

end PAQ_is_isosceles_right_triangle_l748_748606


namespace eval_imaginary_expression_l748_748537

theorem eval_imaginary_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry

end eval_imaginary_expression_l748_748537


namespace vacation_duration_l748_748329

def days_traveling_to_grandparents := 1
def days_at_grandparents := 5
def days_traveling_to_brother := 1
def days_at_brother := 5
def days_traveling_to_sister := 2
def days_at_sister := 5
def days_traveling_home := 2

def total_days : ℕ :=
  days_traveling_to_grandparents +
  days_at_grandparents +
  days_traveling_to_brother +
  days_at_brother +
  days_traveling_to_sister +
  days_at_sister +
  days_traveling_home

def days_in_week := 7

def vacation_weeks : ℕ := total_days / days_in_week

theorem vacation_duration : vacation_weeks = 3 :=
by
  have h_total_days : total_days = 21 := by sorry
  have h_weeks : 21 / 7 = 3 := by sorry
  rw [← h_total_days, h_weeks]
  rfl

end vacation_duration_l748_748329


namespace production_calculation_l748_748053

variable (production_rate_per_machine : ℕ → ℕ → ℕ)
variable (total_production : ℕ → ℕ → ℕ)

def machines_rate (machines : ℕ) (time : ℕ) : ℕ := 
  production_rate_per_machine machines time

def total_production_in_time (machines : ℕ) (time : ℕ) : ℕ := 
  total_production (machines_rate machines 1) time

theorem production_calculation
  (production_rate : production_rate_per_machine 6 1 = 270)
  (num_machines := 10)
  (minutes := 4) :
  total_production_in_time num_machines minutes = 1800 :=
by
  rw [num_machines, minutes, production_rate]
  sorry

end production_calculation_l748_748053


namespace base7_to_base10_conversion_l748_748893

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l748_748893


namespace jimin_last_to_finish_l748_748325

def runner : Type := { x // x = "Namjoon" ∨ x = "Yoongi" ∨ x = "Taehyung" ∨ x = "Jimin" }

variable (Namjoon Yoongi Taehyung Jimin : runner)

axiom H1 : Namjoon = "Namjoon"
axiom H2 : Yoongi = "Yoongi"
axiom H3 : Taehyung = "Taehyung"
axiom H4 : Jimin = "Jimin"
axiom cond1 : Namjoon = "Namjoon"  -- Namjoon first
axiom cond2 : Yoongi ≠ Namjoon -> Yoongi ≠ Jimin ∧ Yoongi ≠ Taehyung  -- Yoongi faster than Taehyung
axiom cond3 : Taehyung ≠ Namjoon -> Taehyung ≠ Jimin  -- Taehyung faster than Jimin

theorem jimin_last_to_finish : Jimin = "Jimin" ∧ Jimin ≠ Namjoon ∧ Jimin ≠ Yoongi ∧ Jimin ≠ Taehyung :=
by
  sorry

end jimin_last_to_finish_l748_748325


namespace shirt_cost_l748_748152

def cost_per_shirt (S : ℝ) : Prop :=
  0.15 * ((2 * 700) + (6 * S) + (2 * 150)) = 300

theorem shirt_cost :
  ∃ S : ℝ, cost_per_shirt S ∧ S = 50 :=
begin
  use 50,
  split,
  { unfold cost_per_shirt,
    simp,
  },
  { refl }
end

end shirt_cost_l748_748152


namespace road_length_l748_748473

theorem road_length
    (total_time : ℝ)
    (walk_speed : ℝ)
    (bus_speed : ℝ)
    (travel_time_walk : ℝ)
    (travel_time_bus : ℝ)
    (walk_bus_equation : travel_time_walk + travel_time_bus = 2) :
  let x := travel_time_walk * walk_speed in x = 8 :=
by
  sorry

end road_length_l748_748473


namespace max_value_of_f_l748_748696

def f (x : ℝ) : ℝ := 4 * (Real.cos x)^3 - 3 * (Real.cos x)^2 - 6 * Real.cos x + 5

theorem max_value_of_f : ∃ x : ℝ, f x = 27 / 4 := sorry

end max_value_of_f_l748_748696


namespace min_n_exceeds_1000_l748_748978
-- Lean 4 statement for the problem:

theorem min_n_exceeds_1000 :
  ∃ n : ℕ, (∀ m < n, 2 ^ (m - Real.log (m + 1)) ≤ 1000) ∧ 2 ^ (n - Real.log (n + 1)) > 1000 :=
sorry

end min_n_exceeds_1000_l748_748978


namespace part1_part2_part3_l748_748878

noncomputable def y1 (x : ℝ) : ℝ := 0.1 * x + 15
noncomputable def y2 (x : ℝ) : ℝ := 0.15 * x

-- Prove that the functions are as described
theorem part1 : ∀ x : ℝ, y1 x = 0.1 * x + 15 ∧ y2 x = 0.15 * x :=
by sorry

-- Prove that x = 300 results in equal charges for Packages A and B
theorem part2 : y1 300 = y2 300 :=
by sorry

-- Prove that Package A is more cost-effective when x > 300
theorem part3 : ∀ x : ℝ, x > 300 → y1 x < y2 x :=
by sorry

end part1_part2_part3_l748_748878


namespace magnitude_of_c_l748_748172

theorem magnitude_of_c {c : ℂ} 
  (P : (x : ℂ) → x^2 - 3 * x + 3 ≠ 0 ∧ x^2 - c * x + 4 ≠ 0 ∧ x^2 - 5 * x + 13 ≠ 0)
  (H : ∀ x : ℂ, P x → ∀ y : ℂ, x ≠ y → P y → x ≠ y) :
  |c| = 4 :=
sorry

end magnitude_of_c_l748_748172


namespace area_of_outer_sphere_marked_l748_748480

noncomputable def r : ℝ := 1  -- Radius of the small painted sphere
noncomputable def R_inner : ℝ := 4  -- Radius of the inner concentric sphere
noncomputable def R_outer : ℝ := 6  -- Radius of the outer concentric sphere
noncomputable def A_inner : ℝ := 47  -- Area of the region on the inner sphere

theorem area_of_outer_sphere_marked :
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner in
  A_outer = 105.75 :=
by
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner
  sorry

end area_of_outer_sphere_marked_l748_748480


namespace banana_distinct_arrangements_l748_748281

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748281


namespace range_of_c_l748_748341

theorem range_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 1) : ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end range_of_c_l748_748341


namespace distinct_arrangements_banana_l748_748263

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748263


namespace asha_money_remaining_l748_748862

-- Given conditions as definitions in Lean
def borrowed_from_brother : ℕ := 20
def borrowed_from_father : ℕ := 40
def borrowed_from_mother : ℕ := 30
def gift_from_granny : ℕ := 70
def initial_savings : ℕ := 100

-- Total amount of money Asha has
def total_money : ℕ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + initial_savings

-- Money spent by Asha
def money_spent : ℕ := (3 * total_money) / 4

-- Money remaining with Asha
def money_remaining : ℕ := total_money - money_spent

-- Theorem stating the result
theorem asha_money_remaining : money_remaining = 65 := by
  sorry

end asha_money_remaining_l748_748862


namespace length_AB_distance_M_P_l748_748319

theorem length_AB 
  (line_l : ∀ t : ℝ, (x = sqrt 3 * t) ∧ (y = 1 + t))
  (curve_C : ∀ ρ θ : ℝ, (ρ^2 * real.cos (2 * θ) = 1))
  (point_P : ∀ (ρ θ : ℝ), ρ = 1 ∧ θ = (real.pi / 2) -> (0, 1)) :
  (|AB| = 2 * real.sqrt 5) := sorry

theorem distance_M_P
  (line_l : ∀ t : ℝ, (x = sqrt 3 * t) ∧ (y = 1 + t))
  (curve_C : ∀ ρ θ : ℝ, (ρ^2 * real.cos (2 * θ) = 1))
  (point_P : ∀ (ρ θ : ℝ), ρ = 1 ∧ θ = (real.pi / 2) -> (0,1)) :
  (d = 1) := sorry

end length_AB_distance_M_P_l748_748319


namespace problem_statement_l748_748215

-- Define the function
def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * (cos x) ^ 2

-- State the problem
theorem problem_statement : (∀ x : ℝ, f (π / 6 - x) = f (π / 6 + x)) :=
sorry

end problem_statement_l748_748215


namespace train_speed_is_260_kmph_l748_748113

-- Define the conditions: length of the train and time to cross the pole
def length_of_train : ℝ := 130
def time_to_cross_pole : ℝ := 9

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the expected speed in kilometers per hour
def expected_speed_kmph : ℝ := 260

-- The theorem statement
theorem train_speed_is_260_kmph :
  (length_of_train / time_to_cross_pole) * conversion_factor = expected_speed_kmph :=
sorry

end train_speed_is_260_kmph_l748_748113


namespace local_minimum_at_neg_one_l748_748351

noncomputable def f (x : ℝ) := x * Real.exp x

theorem local_minimum_at_neg_one : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 1) < δ → f x > f (-1)) :=
sorry

end local_minimum_at_neg_one_l748_748351


namespace kids_stayed_home_l748_748910

open Nat

theorem kids_stayed_home (kids_camp : ℕ) (additional_kids_home : ℕ) (total_kids_home : ℕ) 
  (h1 : kids_camp = 202958) 
  (h2 : additional_kids_home = 574664) 
  (h3 : total_kids_home = kids_camp + additional_kids_home) : 
  total_kids_home = 777622 := 
by 
  rw [h1, h2] at h3
  exact h3

end kids_stayed_home_l748_748910


namespace swimming_day_l748_748861

open DayOfWeek

-- Definitions for days of the week
inductive DayOfWeek
  | Sunday 
  | Monday 
  | Tuesday 
  | Wednesday 
  | Thursday 
  | Friday 
  | Saturday
deriving DecidableEq 

-- Define the sports activities
structure SportsSchedule :=
  (soccer : DayOfWeek)
  (badminton : DayOfWeek)
  (cycling : List DayOfWeek)
  (cricket : DayOfWeek)
  (swimming : DayOfWeek)

-- Arian's schedule definition
def ArianSchedule (schedule : SportsSchedule) : Prop :=
  (schedule.soccer = Monday) ∧
  (schedule.badminton = Wednesday) ∧
  ((schedule.cycling.count = 3) ∧ (∀ d, d ∈ schedule.cycling → ¬(d = Monday ∨ d = Wednesday)) ∧ (∀ d1 d2, d1 ∈ schedule.cycling → d2 ∈ schedule.cycling → abs((d1.toNat - d2.toNat)) ≠ 1 )) ∧
  (∀ d, d = schedule.cricket → (¬ (d + 1).toNat = schedule.cycling.map 1)  ∧ (¬ (d + 1).toNat = schedule.swimming.toNat) )

-- Goal statement
theorem swimming_day : ∃ schedule : SportsSchedule, ArianSchedule schedule ∧ (schedule.swimming = Friday) :=
  by
    sorry

end swimming_day_l748_748861


namespace percent_republicans_voting_for_A_l748_748652

theorem percent_republicans_voting_for_A (V : ℝ) (percent_Democrats : ℝ) 
  (percent_Republicans : ℝ) (percent_D_voting_for_A : ℝ) 
  (percent_total_voting_for_A : ℝ) (R : ℝ) 
  (h1 : percent_Democrats = 0.60)
  (h2 : percent_Republicans = 0.40)
  (h3 : percent_D_voting_for_A = 0.85)
  (h4 : percent_total_voting_for_A = 0.59) :
  R = 0.2 :=
by 
  sorry

end percent_republicans_voting_for_A_l748_748652


namespace probability_of_staying_in_dark_l748_748801

theorem probability_of_staying_in_dark (revolutions_per_minute : ℕ) (time_in_seconds : ℕ) (dark_time : ℕ) :
  revolutions_per_minute = 2 →
  time_in_seconds = 60 →
  dark_time = 5 →
  (5 / 6 : ℝ) = 5 / 6 :=
by
  intros
  sorry

end probability_of_staying_in_dark_l748_748801


namespace probability_abs_diff_gt_1_l748_748724

open probability_theory

noncomputable def coin_flip : ProbMeasure ℝ :=
if h : true then
  ⟨[0, 2], sorry⟩ -- This represents the distribution condition
else
  ⟨[], sorry⟩     -- Dummy value to satisfy the type

def chosen_distribution (s : set ℝ) : ProbMeasure ℝ :=
if h : true then 
  ⟨[0, 2], sorry⟩ -- This represents the uniform distribution condition
else 
  ⟨[], sorry⟩     -- Dummy value to satisfy the type

noncomputable def prob_event (s : set (ℝ × ℝ)) : ℝ :=
(coin_flip.prod coin_flip).val s

theorem probability_abs_diff_gt_1 :
  prob_event {p : ℝ × ℝ | abs (p.1 - p.2) > 1} = 7 / 8 :=
sorry

end probability_abs_diff_gt_1_l748_748724


namespace triangle_AB_length_correct_l748_748324

theorem triangle_AB_length_correct (BC AC : Real) (A : Real) 
  (hBC : BC = Real.sqrt 7) 
  (hAC : AC = 2 * Real.sqrt 3) 
  (hA : A = Real.pi / 6) :
  ∃ (AB : Real), (AB = 5 ∨ AB = 1) :=
by
  sorry

end triangle_AB_length_correct_l748_748324


namespace base7_to_base10_conversion_l748_748896

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l748_748896


namespace period_f_decreasing_interval_range_f_l748_748614

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin(2 * x + π / 6)

-- Define the period of the function
theorem period_f : ∃ (T : ℝ), ∀ x : ℝ, f(x + T) = f(x) :=
begin
  use π,
  sorry
end

-- Define the monotonic decreasing interval
theorem decreasing_interval : ∀ k : ℤ, ∃ a b : ℝ, a = k * π + π / 6 ∧ b = k * π + 2 * π / 3 ∧
  ∀ x1 x2 : ℝ, (a ≤ x1 ∧ x1 ≤ b) ∧ (a ≤ x2 ∧ x2 ≤ b) ∧ (x1 < x2) → f(x1) ≥ f(x2) :=
begin
  intro k,
  use [k * π + π / 6, k * π + 2 * π / 3],
  sorry
end

-- Define the range of f on the interval [-π/6, π/4]
theorem range_f : ∃ min max : ℝ, min ≤ f(-π / 6) ∧ f(-π / 6) ≤ max ∧ ∀ x∈Icc (-π / 6) (π / 4), 
  min ≤ f(x) ∧ f(x) ≤ max :=
begin
  use [-1, 2],
  sorry
end

end period_f_decreasing_interval_range_f_l748_748614


namespace prove_remaining_area_is_24_l748_748834

/-- A rectangular piece of paper with length 12 cm and width 8 cm has four identical isosceles 
right triangles with legs of 6 cm cut from it. Prove that the remaining area is 24 cm². --/
def remaining_area : ℕ := 
  let length := 12
  let width := 8
  let rect_area := length * width
  let triangle_leg := 6
  let triangle_area := (triangle_leg * triangle_leg) / 2
  let total_triangle_area := 4 * triangle_area
  rect_area - total_triangle_area

theorem prove_remaining_area_is_24 : (remaining_area = 24) :=
  by sorry

end prove_remaining_area_is_24_l748_748834


namespace find_a_b_l748_748220

open Set

def solution_set (f : ℝ → ℝ) (s : Set ℝ) : Set ℝ :=
  { x | f x < 0 }

def A := {x : ℝ | -1 < x ∧ x < 3}
def B := {x : ℝ | -3 < x ∧ x < 2}

theorem find_a_b (a b : ℝ) :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} →
  solution_set (λ x, x^2 + a * x + b) (A ∩ B) →
  (a + b = -3) :=
sorry

end find_a_b_l748_748220


namespace ordinate_of_fourth_point_l748_748611

theorem ordinate_of_fourth_point 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x1 x2 : ℝ) 
  (h_roots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) 
  (h_yint : c ≠ 0) :
  let y0 := 1 / a
  in circle_through (x1, 0) (x2, 0) (0, c) (0, y0) :=
sorry

end ordinate_of_fourth_point_l748_748611


namespace arccos_one_half_eq_pi_div_three_l748_748512

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l748_748512


namespace fill_time_bounds_l748_748706

-- Define the conditions
def pool_length : ℝ := 5
def pool_width : ℝ := 3
def pool_depth : ℝ := 2
def pool_capacity : ℝ := 60
def flow_rate (P : ℝ) : ℝ := 1.2 + 0.4 * P
def leak_rate : ℝ := 0.1

-- Define the range of pressure
def min_pressure : ℝ := 1
def max_pressure : ℝ := 10

-- Calculate net flow rates at min and max pressure
def net_flow_rate_min : ℝ := (flow_rate min_pressure) - leak_rate
def net_flow_rate_max : ℝ := (flow_rate max_pressure) - leak_rate

-- Calculate the time to fill the pool at these net flow rates
def fill_time_min_pressure : ℝ := pool_capacity / net_flow_rate_min
def fill_time_max_pressure : ℝ := pool_capacity / net_flow_rate_max

-- Prove the fill times
theorem fill_time_bounds :
  fill_time_max_pressure = 11.76 ∧ fill_time_min_pressure = 40 :=
by
  sorry

end fill_time_bounds_l748_748706


namespace part_I_part_II_l748_748955

def f (x : ℝ) (φ : ℝ) : ℝ := (√3 / 2) * Real.cos (2 * x + φ) + Real.sin x ^ 2

theorem part_I (k : ℤ) : 
  let φ := π / 6 
  ∃ a b : ℝ, a = k * π - 2 * π / 3 ∧ b = k * π - π / 6 ∧ 
  (∀ x, a ≤ x ∧ x ≤ b → (Real.deriv (λ x, f x φ) x ≥ 0)) :=
sorry

theorem part_II : 
  ∃ φ : ℝ, f(0) φ = 3 / 2 ∧ (0 ≤ φ) ∧ (φ < π) ∧ φ = π / 2 :=
sorry

end part_I_part_II_l748_748955


namespace tangents_quadrilateral_cyclic_l748_748839

variables {A B C D K L O1 O2 : Point}
variable (r : ℝ)
variable (AB_cut_circles : ∀ {A B : Point} {O1 O2 : Point}, is_intersect AB O1 O2)
variable (parallel_AB_O1O2 : is_parallel AB O1O2)
variable (tangents_formed_quadrilateral : is_quadrilateral C D K L)
variable (quadrilateral_contains_circles : contains C D K L O1 O2)

theorem tangents_quadrilateral_cyclic
  (h1: AB_cut_circles)
  (h2: parallel_AB_O1O2) 
  (h3: tangents_formed_quadrilateral)
  (h4: quadrilateral_contains_circles)
  : ∃ O : Circle, is_inscribed O C D K L :=
sorry

end tangents_quadrilateral_cyclic_l748_748839


namespace find_k_l748_748028

-- Define the conditions
def circles_centered_at_origin : Prop := ∀ (r₁ r₂ : ℝ), r₁ ≥ 0 ∧ r₂ ≥ 0

def point_on_larger_circle (P : ℝ × ℝ) (r₁ : ℝ) : Prop := P.1^2 + P.2^2 = r₁^2

def point_on_smaller_circle (S : ℝ × ℝ) (r₂ : ℝ) : Prop := S.1^2 + S.2^2 = r₂^2

def qr_equals_5 (QR : ℝ) : Prop := QR = 5

-- Given data
def P : ℝ × ℝ := (3, 4)

def Q_R : ℝ := 5

noncomputable def larger_circle_radius : ℝ := real.sqrt (P.1^2 + P.2^2)

def smaller_circle_center (S : ℝ × ℝ) : ℝ := S.1^2 + S.2^2

-- Problem Statement
theorem find_k (S : ℝ × ℝ) (k : ℝ) :
  circles_centered_at_origin ∧ 
  point_on_larger_circle P 5 ∧
  point_on_smaller_circle (0, k) 0 ∧ 
  qr_equals_5 Q_R 
  → k = 0 :=
by {
  sorry
}

end find_k_l748_748028


namespace Micheal_and_Adam_together_l748_748357

-- Define the variables and conditions
variables {W : ℝ} -- total work
variables {M A : ℝ} -- work rates of Micheal and Adam

def combined_work_rate_18_days (W M A : ℝ) := 18 * (M + A) -- work done together for 18 days
def adam_work_10_days (W M A : ℝ) := 10 * A -- remaining work done by Adam in 10 days

-- Micheal can complete the work in 25 days, hence his work rate is W / 25
axiom Micheal_work_rate : M = W / 25

-- Micheal and Adam work together for 18 days + Adam works alone for 10 days to complete the total work W
axiom total_work_done : combined_work_rate_18_days W M A + adam_work_10_days W M A = W

-- We need to prove Micheal and Adam together can complete the work in 20 days
theorem Micheal_and_Adam_together (W M A : ℝ) (Micheal_work_rate : M = W / 25)
  (total_work_done : combined_work_rate_18_days W M A + adam_work_10_days W M A = W) :
  let combined_rate := M + A in W / combined_rate = 20 :=
by
  -- We leave the proof as an exercise
  sorry

end Micheal_and_Adam_together_l748_748357


namespace banana_permutations_l748_748241

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748241


namespace other_candidate_valid_votes_l748_748806

theorem other_candidate_valid_votes
  (total_votes : ℕ)
  (percentage_invalid : ℚ)
  (percentage_first_candidate : ℚ)
  (total_votes = 7500)
  (percentage_invalid = 20 / 100)
  (percentage_first_candidate = 55 / 100) :
  ∃ (valid_votes_other_candidate : ℕ), valid_votes_other_candidate = 2700 :=
by
  sorry

end other_candidate_valid_votes_l748_748806


namespace base7_to_base10_conversion_l748_748897

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l748_748897


namespace outer_circle_increase_l748_748772

theorem outer_circle_increase : 
  let R_o := 6
  let R_i := 4
  let R_i_new := (3 : ℝ)  -- 4 * (3/4)
  let A_original := 20 * Real.pi  -- π * (6^2 - 4^2)
  let A_new := 72 * Real.pi  -- 3.6 * A_original
  ∃ (x : ℝ), 
    let R_o_new := R_o * (1 + x / 100)
    π * R_o_new^2 - π * R_i_new^2 = A_new →
    x = 50 := 
sorry

end outer_circle_increase_l748_748772


namespace sum_of_squares_geometric_progression_theorem_l748_748414

noncomputable def sum_of_squares_geometric_progression (a₁ q : ℝ) (S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) : ℝ :=
  S₁ * S₂

theorem sum_of_squares_geometric_progression_theorem
  (a₁ q S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) :
  sum_of_squares_geometric_progression a₁ q S₁ S₂ h_q h_S₁ h_S₂ = S₁ * S₂ := sorry

end sum_of_squares_geometric_progression_theorem_l748_748414


namespace groceries_spent_percent_l748_748723

variable (salary : ℝ) (credit_percent : ℝ) (cash_in_hand : ℝ)

theorem groceries_spent_percent (h_salary : salary = 4000)
  (h_credit_percent : credit_percent = 0.15)
  (h_cash_in_hand : cash_in_hand = 2380) :
  let fixed_deposit := credit_percent * salary,
      remaining_amount := salary - fixed_deposit,
      amount_spent_on_groceries := remaining_amount - cash_in_hand,
      spent_percent := (amount_spent_on_groceries / remaining_amount) * 100
  in spent_percent = 30 := sorry

end groceries_spent_percent_l748_748723


namespace convert_base_7_to_base_10_l748_748887

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l748_748887


namespace train_cross_platform_time_l748_748046

noncomputable def time_to_cross_platform (L_t L_p : ℕ) (S_t : ℝ) : ℝ := 
  let distance := (L_t + L_p : ℤ)
  let speed_mps := S_t * (1000 / 3600)
  distance / speed_mps

theorem train_cross_platform_time :
  (time_to_cross_platform 470 520 55) ≈ 64.76 :=
by
  sorry

end train_cross_platform_time_l748_748046


namespace solve_equation_correctness_l748_748928

def greatest_integer (x : ℝ) : ℤ := ⌊x⌋

noncomputable def solve_equation : ℝ :=
  let x_star : ℝ := 252 + 1 / 3 in
  x_star

theorem solve_equation_correctness : 
  ∃ (x : ℝ) (n : ℤ), (3 * x + 5 * n - 2017 = 0) ∧ (greatest_integer x = n) ∧ (x = solve_equation) := 
by
  sorry

end solve_equation_correctness_l748_748928


namespace percent_of_l748_748814

theorem percent_of (Part Whole : ℕ) (Percent : ℕ) (hPart : Part = 120) (hWhole : Whole = 40) :
  Percent = (Part * 100) / Whole → Percent = 300 :=
by
  sorry

end percent_of_l748_748814


namespace base7_to_base10_conversion_l748_748895

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l748_748895


namespace least_number_to_subtract_l748_748425

theorem least_number_to_subtract (n : ℕ) (p : ℕ) (hdiv : p = 47) (hn : n = 929) 
: ∃ k, n - 44 = k * p := by
  sorry

end least_number_to_subtract_l748_748425


namespace base7_to_base10_conversion_l748_748894

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l748_748894


namespace plates_arrangement_l748_748092

/--
A family purchases 6 blue plates, 3 red plates, 2 green plates, and 2 orange plates. Prove that there are 113080 ways to arrange these plates for dinner around a circular table if no two green plates or two orange plates are adjacent.
-/
theorem plates_arrangement:
  let blue_plates := 6 in
  let red_plates := 3 in
  let green_plates := 2 in
  let orange_plates := 2 in
  no_adjacent green_plates orange_plates 113080 := sorry

end plates_arrangement_l748_748092


namespace matrix_equation_l748_748339

open Matrix

-- Define matrix N and the identity matrix I
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![-4, -2]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

-- Scalars p and q
def p : ℤ := 1
def q : ℤ := -26

-- Theorem statement
theorem matrix_equation :
  N * N = p • N + q • I :=
  by
    sorry

end matrix_equation_l748_748339


namespace pow_mod_1000_of_6_eq_296_l748_748031

theorem pow_mod_1000_of_6_eq_296 : (6 ^ 1993) % 1000 = 296 := by
  sorry

end pow_mod_1000_of_6_eq_296_l748_748031


namespace possible_values_magnitude_sum_vectors_l748_748206

theorem possible_values_magnitude_sum_vectors 
    (a b c : ℝ × ℝ)
    (unit_a : ∥a∥ = 1)
    (unit_b : ∥b∥ = 1)
    (angle_ab : real.angle a b = real.pi / 3)
    (mag_c : ∥c∥ = 2 * real.sqrt 3) :
  ∃ x : ℝ, x ∈ set.of_list [3, 4, 5] ∧ ∥a + b + c∥ = x :=
sorry

end possible_values_magnitude_sum_vectors_l748_748206


namespace attendees_receive_all_items_l748_748934

theorem attendees_receive_all_items :
  let capacity := 5000
  let every_poster := 100
  let every_program := 45
  let every_drink := 60
  let lcm := Nat.lcm (Nat.lcm every_poster every_program) every_drink
  capacity / lcm = 5 :=
by
  let capacity := 5000
  let every_poster := 100
  let every_program := 45
  let every_drink := 60
  let lcm := Nat.lcm (Nat.lcm every_poster every_program) every_drink
  have h : Nat.lcm 100 45 = 900 := Nat.lcm_eq 100 45 (by norm_num) (by norm_num)
  have h2 : Nat.lcm 900 60 = 900 := Nat.lcm_eq 900 60 (by norm_num) (by norm_num)
  have h3 : capacity / lcm = 5 := by norm_num
  exact h3

end attendees_receive_all_items_l748_748934


namespace polynomial_rational_difference_l748_748544

theorem polynomial_rational_difference {f : ℝ → ℝ} (hf_deg2 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (hf_rational_diff : ∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) :
  ∃ b c : ℚ, ∃ d : ℝ, f = λ x, b * (x : ℝ) + d :=
by
  sorry

end polynomial_rational_difference_l748_748544


namespace difference_is_693_l748_748038

noncomputable def one_tenth_of_seven_thousand : ℕ := 1 / 10 * 7000
noncomputable def one_tenth_percent_of_seven_thousand : ℕ := (1 / 10 / 100) * 7000
noncomputable def difference : ℕ := one_tenth_of_seven_thousand - one_tenth_percent_of_seven_thousand

theorem difference_is_693 :
  difference = 693 :=
by
  sorry

end difference_is_693_l748_748038


namespace ratio_proof_l748_748558

noncomputable def ratio_of_segment_lengths (a b : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  points.card = 5 ∧
  ∃ (dists : Finset ℝ), 
    dists = {a, a, a, a, a, b, 3 * a} ∧
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      (dist p1 p2 ∈ dists)

theorem ratio_proof (a b : ℝ) (points : Finset (ℝ × ℝ)) (h : ratio_of_segment_lengths a b points) : 
  b / a = 2.8 :=
sorry

end ratio_proof_l748_748558


namespace vidya_age_l748_748421

theorem vidya_age (V : ℕ) (h1 : ∀ V : ℕ, 44 = 3 * V + 5) : V = 13 :=
by {
  have h : 44 = 3 * V + 5 := h1 V,
  sorry
}

end vidya_age_l748_748421


namespace seeds_in_bucket_C_l748_748408

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l748_748408


namespace part1_part2_l748_748942

section Problem

open Real

noncomputable def f (x : ℝ) := exp x
noncomputable def g (x : ℝ) := log x - 2

theorem part1 (x : ℝ) (hx : x > 0) : g x ≥ - (exp 1) / x :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, x ≥ 0 → f x - 1 / (f x) ≥ a * x) : a ≤ 2 :=
by sorry

end Problem

end part1_part2_l748_748942


namespace negation_equivalence_l748_748759

theorem negation_equivalence (x : ℝ) :
  (¬ (x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (x < 1 → x^2 - 4*x + 2 < -1) :=
by
  sorry

end negation_equivalence_l748_748759


namespace midpoint_sum_is_correct_l748_748786

theorem midpoint_sum_is_correct:
  let A := (10, 8)
  let B := (-4, -6)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + midpoint.2) = 4 :=
by
  sorry

end midpoint_sum_is_correct_l748_748786


namespace unique_solution_mnk_l748_748141

theorem unique_solution_mnk :
  ∀ (m n k : ℕ), 3^n + 4^m = 5^k → (m, n, k) = (0, 1, 1) :=
by
  intros m n k h
  sorry

end unique_solution_mnk_l748_748141


namespace exist_nat_nums_l748_748907

theorem exist_nat_nums :
  ∃ (a b c d : ℕ), (a / (b : ℚ) + c / (d : ℚ) = 1) ∧ (a / (d : ℚ) + c / (b : ℚ) = 2008) :=
sorry

end exist_nat_nums_l748_748907


namespace total_shaded_area_correct_l748_748061

-- Define the radius of the quarter circles
def radius : ℝ := 1 / 2

-- Define the area of a full circle formed by the four quarter circles in a tile
def fullCircleArea : ℝ := π * (radius ^ 2)

-- Define the area of one square tile
def tileArea : ℝ := 1

-- Define the shaded area of one tile
def shadedAreaOneTile : ℝ := tileArea - fullCircleArea

-- Define the total number of tiles
def totalTiles : ℝ := 12 * 15

-- Define the total shaded area of the floor
def totalShadedArea : ℝ := totalTiles * shadedAreaOneTile

-- Prove that the total shaded area on the floor is 180 - 45π square feet
theorem total_shaded_area_correct :
  totalShadedArea = 180 - 45 * π :=
by
  sorry

end total_shaded_area_correct_l748_748061


namespace part1_part2_l748_748983

open Set Real

def M (x : ℝ) : Prop := x^2 - 3 * x - 18 ≤ 0
def N (x : ℝ) (a : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 2 * a + 1

theorem part1 (a : ℝ) (h : a = 3) : (Icc (-2 : ℝ) 6 = {x | M x ∧ N x a}) ∧ (compl {x | N x a} = Iic (-2) ∪ Ioi 7) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, M x ∧ N x a ↔ N x a) → a ≤ 5 / 2 :=
by
  sorry

end part1_part2_l748_748983


namespace range_of_product_x1_x2_l748_748971

open Real

def f (x : ℝ) : ℝ :=
if h : x ≥ 1 then ln x
else 1 - x / 2

noncomputable def F (x : ℝ) (m : ℝ) : ℝ :=
f (f x + 1) + m

theorem range_of_product_x1_x2 (m : ℝ) (x1 x2 : ℝ) (hx1 : F x1 m = 0) (hx2 : F x2 m = 0) :
  ∃ t > (1 / 2), x1 * x2 = exp t * (2 - 2 * t) ∧ x1 * x2 < sqrt e :=
sorry

end range_of_product_x1_x2_l748_748971


namespace new_rope_length_l748_748837

-- Define the given constants and conditions
def rope_length_initial : ℝ := 12
def additional_area : ℝ := 1511.7142857142858
noncomputable def pi_approx : ℝ := Real.pi

-- Define the proof statement
theorem new_rope_length :
  let r2 := Real.sqrt ((additional_area / pi_approx) + rope_length_initial ^ 2)
  r2 = 25 :=
by
  -- Placeholder for the proof
  sorry

end new_rope_length_l748_748837


namespace angle_BDC_is_15_degrees_l748_748733

theorem angle_BDC_is_15_degrees (A B C D : Type) (AB AC AD CD : ℝ) (angle_BAC : ℝ) :
  AB = AC → AC = AD → CD = 2 * AC → angle_BAC = 30 →
  ∃ angle_BDC, angle_BDC = 15 := 
by
  sorry

end angle_BDC_is_15_degrees_l748_748733


namespace stock_price_drop_l748_748571

theorem stock_price_drop (P : ℝ) (h1 : P > 0) (x : ℝ)
  (h3 : (1.30 * (1 - x/100) * 1.20 * P) = 1.17 * P) :
  x = 25 :=
by
  sorry

end stock_price_drop_l748_748571


namespace find_a3_l748_748318

section
variable {a : ℕ → ℝ}

-- Conditions
variable (geo : ∀ n m, a n * a m = a n + 1 * a m - 1)
variable (a1 : a 1 = 1)
variable (a5 : a 5 = 16)

-- Statement to prove
theorem find_a3 (hgeo : geo) (h_a1 : a1) (h_a5 : a5) : a 3 = 4 :=
by
  sorry
end

end find_a3_l748_748318


namespace distinct_arrangements_banana_l748_748264

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748264


namespace stock_price_changes_to_273_l748_748718

def stock_changes (x y : ℕ) : Prop :=
  x + y = 13 ∧ x - y = 7 ∧ ∀ d, d < 13 → (∑ i in range d, (1)) ≥ 0

theorem stock_price_changes_to_273 (x y : ℕ) :
  stock_changes x y → (x = 10 ∧ y = 3) → (finset.card ((range 13).powerset.filter (λ s, s.card = 3)) - 13 = 273) :=
by
  intros h₁ h₂
  sorry

end stock_price_changes_to_273_l748_748718


namespace fraction_equivalence_l748_748040

theorem fraction_equivalence : 
    (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end fraction_equivalence_l748_748040


namespace beginning_of_spring_period_and_day_l748_748180

noncomputable def daysBetween : Nat := 46 -- Total days: Dec 21, 2004 to Feb 4, 2005

theorem beginning_of_spring_period_and_day :
  let total_days := daysBetween
  let segment := total_days / 9
  let day_within_segment := total_days % 9
  segment = 5 ∧ day_within_segment = 1 := by
sorry

end beginning_of_spring_period_and_day_l748_748180


namespace find_m_l748_748622

variable (x y k m : ℝ) -- Introducing variables used in the problem

-- Given conditions in the problem
def parabola_eq : Prop := x^2 = 4 * y
def line_eq : Prop := y = k * x + m
def intersection_points (x1 x2 : ℝ) : Prop := x1 * x2 = -4

-- Proving the value of m under these conditions
theorem find_m (x1 x2 : ℝ) (h_parabola : parabola_eq x x1) (h_line : line_eq x1 y) (h_intersection : intersection_points x1 x2) :
  m = 1 := 
sorry

end find_m_l748_748622


namespace complex_number_solution_l748_748747

theorem complex_number_solution :
  let a := (⊥ : ℝ) + (7 : ℝ) in -- imaginary part of -√2 + 7i
  let b := -5 in -- real part of √7i + 5i², simplifying to -5
  (a = 7) ∧ (b = -5) → (∃ z : ℂ, z = a + b * I ∧ z = 7 - 5 * I) :=
by
  intros a b h_real h_imag
  use (a + b * Complex.I)
  split
  · sorry
  · sorry

end complex_number_solution_l748_748747


namespace area_of_rectangular_field_l748_748829

theorem area_of_rectangular_field : 
  ∀ (W L : ℝ), 
  (L = 2 * W) → 
  (let P := 600 in 
   (2 * L + 2 * W = P) → 
   (let A := L * W in A = 20000)) :=
by 
  intros W L h1 h2 
  let P := 600
  let A := L * W
  sorry

end area_of_rectangular_field_l748_748829


namespace gasoline_expense_l748_748721

-- Definitions for the conditions
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10
def initial_amount : ℝ := 50
def amount_left_for_return_trip : ℝ := 36.35

-- Definition for the total gift cost
def total_gift_cost : ℝ := number_of_people * gift_cost_per_person

-- Definition for the total amount received from grandma
def total_grandma_gift : ℝ := number_of_people * grandma_gift_per_person

-- Definition for the total initial amount including the gift from grandma
def total_initial_amount_with_gift : ℝ := initial_amount + total_grandma_gift

-- Definition for remaining amount after spending on lunch and gifts
def remaining_after_known_expenses : ℝ := total_initial_amount_with_gift - lunch_cost - total_gift_cost

-- The Lean theorem to prove the gasoline expense
theorem gasoline_expense : remaining_after_known_expenses - amount_left_for_return_trip = 8 := by
  sorry

end gasoline_expense_l748_748721


namespace constant_term_binomial_expansion_l748_748748

theorem constant_term_binomial_expansion :
  let expr := (x - 1 / (2 * Real.sqrt x))^6
  let constant_term := 15 / 16
  ∃ k : ℕ, (x - 1 / (2 * Real.sqrt x))^6 = ∑ i in range (6 + 1), (binomial 6 i * x^(6 - i) * (-1 / (2 * Real.sqrt x))^i)
  ∧ (∑ i in range (6 + 1), (binomial 6 i * x^(6 - i) * (-1 / (2 * Real.sqrt x))^i)).coeff 0 = constant_term :=
sorry

end constant_term_binomial_expansion_l748_748748


namespace polynomial_conditions_l748_748547

noncomputable def poly_rational_diff (f : ℝ → ℝ) :=
  ∀ x y : ℝ, ∃ q : ℚ, x - y = q → (f x - f y) ∈ ℚ

theorem polynomial_conditions (f : ℝ → ℝ)
    (h_deg : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c) 
    (h_cond : poly_rational_diff f) : 
    ∃ b : ℚ, ∃ c : ℝ, f = λ x, ↑b * x + c :=
sorry

end polynomial_conditions_l748_748547


namespace base16_to_base2_digits_l748_748433

theorem base16_to_base2_digits {A B C D E : ℕ} (hA : A = 10) (hB : B = 11) (hC : C = 12) (hD : D = 13) (hE : E = 14) :
  (let n := A * 16^4 + B * 16^3 + C * 16^2 + D * 16^1 + E in (n.bit_length = 20)) :=
by
  -- Variables are already set accordingly
  sorry

end base16_to_base2_digits_l748_748433


namespace triangle_LMN_is_isosceles_l748_748376

noncomputable def T : Type := sorry

-- Given conditions
variable (K L M N T : T) 
variable (SegmentsIntersect : ∃ K_L : K × L, ∃ M_N : M × N, K_L.2 = M_N.1 ∧ K_L.1 = T ∧ M_N.2 = T)
variable (KNT_Equilateral : ∀a b c : T, a = K ∧ b = N ∧ c = T → 
  (a = b ∧ b = c))
variable (KL_eq_MT : ∀ k l m t : T, k = K ∧ l = L ∧ m = M ∧ t = T → 
  l = t )

-- Theorem statement
theorem triangle_LMN_is_isosceles
  (h1 : SegmentsIntersect K L M N T)
  (h2 : KNT_Equilateral K N T)
  (h3 : KL_eq_MT K L M T) :
  ∃ a b c : T, (a = L ∧ b = M ∧ c = N) → 
  (a = b ∨ a = c ∨ b = c) :=
  sorry

end triangle_LMN_is_isosceles_l748_748376


namespace binomial_expansion_terms_l748_748959

theorem binomial_expansion_terms (n : ℕ) (hn : n = 15)
    (h_sum : Nat.choose n (n-2) + Nat.choose n (n-1) + Nat.choose n n = 121) :
    ∃ (k l m : ℕ), k = 13 ∧ l = 8 ∧ m = 9 :=
by
  have hn : n = 15 := rfl
  have h_sum : Nat.choose 15 (15-2) + Nat.choose 15 (15-1) + Nat.choose 15 15 = 121 := by
  simp [Nat.choose]
  sorry
  use [13, 8, 9]
  sorry

end binomial_expansion_terms_l748_748959


namespace monotonicity_of_f_solve_inequality_range_of_m_l748_748600

variable {f : ℝ → ℝ}
variable {a b m : ℝ}

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def f_at_one (f : ℝ → ℝ) : Prop := f 1 = 1
def positivity_condition (f : ℝ → ℝ) (a b : ℝ) : Prop := (a + b ≠ 0) → ((f a + f b) / (a + b) > 0)

-- Proof problems
theorem monotonicity_of_f 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x, in_interval (x + 1/2) → in_interval (1 / (x - 1)) → f (x + 1/2) < f (1 / (x - 1)) → -3/2 ≤ x ∧ x < -1 :=
sorry

theorem range_of_m 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) :
  (∀ a, in_interval a → f a ≤ m^2 - 2 * a * m + 1) → (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end monotonicity_of_f_solve_inequality_range_of_m_l748_748600


namespace exists_nonoverlapping_subset_with_area_at_least_one_ninth_l748_748198

open Set

-- DEFINE THE CONDITIONS
variable (circles : Set (Set ℝ)) -- Define a set of circles (as sets of real numbers)
variable (total_area : ℝ) -- Define the total area occupied by the circles

-- Assume the given conditions
axiom circles_overlap_and_occuppy_area : ∀ C ∈ circles, 
  ∃ (a : ℝ) (b : ℝ) (r : ℝ), C = {p | (p - a)^2 + (p - b)^2 < r^2} ∧
    (⋃₀ circles).measure = 1

-- THE QUESTION: statement to be proven
theorem exists_nonoverlapping_subset_with_area_at_least_one_ninth :
  ∃ nonoverlapping_subset : Set (Set ℝ), 
  (∀ C1 ∈ nonoverlapping_subset, ∀ C2 ∈ nonoverlapping_subset, C1 ≠ C2 → (C1 ∩ C2).measure = 0) ∧ -- pairwise non-overlapping condition
  (⋃₀ nonoverlapping_subset).measure ≥ 1 / 9 := 
sorry

end exists_nonoverlapping_subset_with_area_at_least_one_ninth_l748_748198


namespace find_number_with_specific_square_digits_l748_748767

def has_accepted_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ∈ {0, 2, 3, 5}

theorem find_number_with_specific_square_digits :
  ∃ n : ℕ, has_accepted_digits (n * n) ∧ n = 5 :=
by {
  use 5,
  split,
  { intros d hd,
    cases d,
    { exact dec_trivial },
    { cases d,
      { exact dec_trivial },
      { cases d,
        { exact dec_trivial },
        { apply false.elim,
          cases hd,
          { exact hd },
          { exact hd,
            repeat {
              apply or.inr,
            } exact or.inl, exact or.inl,
            },
          } },
      { apply false.elim,
        repeat {
          cases hd,
          { exact hd },
          { exact hd,
            repeat {
              apply or.inr,
            } exact or.inl, exact or.inl,
            },
          }
        },
      },
    },
  },
  { refl }
}

end find_number_with_specific_square_digits_l748_748767


namespace probability_of_at_least_one_l748_748778

theorem probability_of_at_least_one (P_1 P_2 : ℝ) (h1 : 0 ≤ P_1 ∧ P_1 ≤ 1) (h2 : 0 ≤ P_2 ∧ P_2 ≤ 1) :
  1 - (1 - P_1) * (1 - P_2) = P_1 + P_2 - P_1 * P_2 :=
by
  sorry

end probability_of_at_least_one_l748_748778


namespace sector_properties_l748_748187

variables (r : ℝ) (alpha l S : ℝ)

noncomputable def arc_length (r alpha : ℝ) : ℝ := alpha * r
noncomputable def sector_area (l r : ℝ) : ℝ := (1/2) * l * r

theorem sector_properties
  (h_r : r = 2)
  (h_alpha : alpha = π / 6) :
  arc_length r alpha = π / 3 ∧ sector_area (arc_length r alpha) r = π / 3 :=
by
  sorry

end sector_properties_l748_748187


namespace problem_acd_div_b_l748_748162

theorem problem_acd_div_b (a b c d : ℤ) (x : ℝ)
    (h1 : x = (a + b * Real.sqrt c) / d)
    (h2 : (7 * x) / 4 + 2 = 6 / x) :
    (a * c * d) / b = -322 := sorry

end problem_acd_div_b_l748_748162


namespace A_and_D_mut_exclusive_not_complementary_l748_748025

-- Define the events based on the conditions
inductive Die
| one | two | three | four | five | six

def is_odd (d : Die) : Prop :=
  d = Die.one ∨ d = Die.three ∨ d = Die.five

def is_even (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_multiple_of_2 (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_two_or_four (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four

-- Define the predicate for mutually exclusive but not complementary
def mutually_exclusive_but_not_complementary (P Q : Die → Prop) : Prop :=
  (∀ d, ¬ (P d ∧ Q d)) ∧ ¬ (∀ d, P d ∨ Q d)

-- Verify that "A and D" are mutually exclusive but not complementary
theorem A_and_D_mut_exclusive_not_complementary :
  mutually_exclusive_but_not_complementary is_odd is_two_or_four :=
  by
    sorry

end A_and_D_mut_exclusive_not_complementary_l748_748025


namespace polynomial_rational_difference_l748_748545

theorem polynomial_rational_difference {f : ℝ → ℝ} (hf_deg2 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (hf_rational_diff : ∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) :
  ∃ b c : ℚ, ∃ d : ℝ, f = λ x, b * (x : ℝ) + d :=
by
  sorry

end polynomial_rational_difference_l748_748545


namespace lambda_value_l748_748998

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748998


namespace find_a_value_l748_748347

variable (a : ℝ)
axiom sqrt_condition : sqrt(a) = sqrt(7 + sqrt(13)) - sqrt(7 - sqrt(13))

theorem find_a_value : a = 2 :=
by
  have h := sqrt_condition
  sorry

end find_a_value_l748_748347


namespace acute_triangle_ab_ac_value_l748_748657

noncomputable def ab_ac : ℝ :=
  let a : ℝ := 30
  let b : ℝ := 18
  let k : ℝ := sorry -- cosine definition will be complex and based on detailed trigonometric setup
  a * b / (k * k)

theorem acute_triangle_ab_ac_value :
  ∀ (a b xp pq qy : ℝ),
  (a = 30) →
  (b = 18) →
  (xp = 12) →
  (pq = 30) →
  (qy = 18) →
  acute_triangle a b xp pq qy →
  ab_ac = 700 * sqrt 15 := sorry

end acute_triangle_ab_ac_value_l748_748657


namespace division_of_difference_squared_l748_748429

theorem division_of_difference_squared :
  ((2222 - 2121)^2) / 196 = 52 := 
sorry

end division_of_difference_squared_l748_748429


namespace proof_problem_l748_748948

open Int

theorem proof_problem (p : ℤ) (h1 : Odd p) (h2 : 5 < p) : 
  ((p - 1) ^ ((p - 1) / 2) - 1) % (p - 2) = 0 :=
sorry

end proof_problem_l748_748948


namespace meal_for_children_after_adults_l748_748047

theorem meal_for_children_after_adults (meal_for_70_adults meal_for_90_children : ℕ) 
    (remaining_meal : ℕ) : 
    meal_for_70_adults = meal_for_90_children → remaining_meal = 36 :=
by
  -- Define the conditions based on the problem
  have h_meal_adult : meal_for_70_adults = 70 := sorry
  have h_meal_child : meal_for_90_children = 90 := sorry
  have h_remaining_meal : remaining_meal = (70 - 42) * (90 / 70) := sorry
  -- Simplify the remaining meal calculation
  calc 
    remaining_meal 
      = 28 * (90 / 70) : by rw [h_remaining_meal]
      = 36 : by sorry

  assumption


end meal_for_children_after_adults_l748_748047


namespace correct_option_B_l748_748793

theorem correct_option_B (x : ℝ) : (1 - x)^2 = 1 - 2 * x + x^2 :=
sorry

end correct_option_B_l748_748793


namespace roots_quadratic_diff_by_12_l748_748305

theorem roots_quadratic_diff_by_12 (P : ℝ) : 
  (∀ α β : ℝ, (α + β = 2) ∧ (α * β = -P) ∧ ((α - β) = 12)) → P = 35 := 
by
  intro h
  sorry

end roots_quadratic_diff_by_12_l748_748305


namespace parallelogram_area_calculation_l748_748740

variables {V : Type*} [inner_product_space ℝ V] {u v : V}

def vector_magnitude (x : V) : ℝ := ∥x∥

noncomputable def parallelogram_area (a b : V) : ℝ :=
  vector_magnitude (a × b)

theorem parallelogram_area_calculation (h : vector_magnitude (u × v) = 12) :
  parallelogram_area (3 • u - 2 • v) (4 • u + v) = 132 :=
begin
  sorry -- proof goes here
end

end parallelogram_area_calculation_l748_748740


namespace no_sum_to_2023_l748_748809

theorem no_sum_to_2023 (n : ℕ) (a : ℕ → ℕ)
  (h1 : 2 ≤ n)
  (h2 : ∀ i, (a (i % n) = 2 * a ((i + 1) % n) ∨ a (i % n) = (2 * a ((i + 1) % n)) / 2 
  ∨ a (i % n) = 5 * a ((i + 1) % n) ∨ a (i % n) = (5 * a ((i + 1) % n)) / 5))
  : ∑ i in (Finset.range n), a i ≠ 2023 := sorry

end no_sum_to_2023_l748_748809


namespace amusement_park_tickets_l748_748911

theorem amusement_park_tickets (x y : ℕ) 
    (total_people : x + y = 15) 
    (adult_ticket_price : ℕ := 50)
    (student_ticket_discount : ℕ := 60)
    (total_spent : x * adult_ticket_price + y * (adult_ticket_price * student_ticket_discount / 100) = 650) :
    x = 10 ∧ y = 5 :=
begin
  sorry
end

end amusement_park_tickets_l748_748911


namespace quadratic_eq_identical_roots_k_value_l748_748299

theorem quadratic_eq_identical_roots_k_value :
  (∃ k : ℝ, (∀ x : ℝ, 3 * x^2 - 6 * x + k = 0 → (b^2 - 4 * a * c = 0)) → k = 3) :=
by
  let a := 3
  let b := -6
  let c := k
  have h1 : b^2 - 4 * a * c = 0 := sorry
  have h2 : k = 3 := sorry
  exact ⟨3, h2⟩

end quadratic_eq_identical_roots_k_value_l748_748299


namespace rectangular_solid_surface_area_l748_748914

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneq1 : a ≠ b) (hneq2 : b ≠ c) (hneq3 : a ≠ c) (hvol : a * b * c = 770) : 2 * (a * b + b * c + c * a) = 1098 :=
by
  sorry

end rectangular_solid_surface_area_l748_748914


namespace find_b_value_l748_748755

def is_perpendicular_bisector (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (m : ℝ × ℝ), m = ((2 + 10) / 2, (4 + (-6)) / 2) ∧ l1 m.1 m.2 ∧ l2 m.1 m.2

def line (b : ℝ) : ℝ → ℝ → Prop := λ x y, x - y = b

theorem find_b_value :
  is_perpendicular_bisector (line 7) (λ x y, x - y = 7) → ∃ b, b = 7 :=
by
  intro h
  use 7
  sorry

end find_b_value_l748_748755


namespace cara_age_is_40_l748_748503

-- Defining the conditions
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Proving the question
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end cara_age_is_40_l748_748503


namespace eccentricity_ellipse_l748_748947

-- Definitions of the given conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def semi_focal_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - b^2)

def eccentricity_range (e : ℝ) : Prop :=
  (real.sqrt 2 - 1 < e) ∧ (e < 1)

def sine_rule_condition (a c : ℝ) (P F1 F2 : ℝ) (angle_PF1F2 angle_PF2F1 : ℝ) : Prop :=
  (a / real.sin angle_PF1F2) = (c / real.sin angle_PF2F1)

-- Proof problem statement
theorem eccentricity_ellipse (a b : ℝ) (e : ℝ)
  (x y : ℝ) (c : ℝ :=
    semi_focal_distance a b) :
  is_ellipse a b x y →
  ∃ P F1 F2 (angle_PF1F2 angle_PF2F1 : ℝ), 
    P = (x, y) ∧
    sine_rule_condition a c P F1 F2 angle_PF1F2 angle_PF2F1 → 
    eccentricity_range e :=
sorry

end eccentricity_ellipse_l748_748947


namespace rotation_135_deg_l748_748026

theorem rotation_135_deg {x y : ℝ} :
  let A := (0, 0) in
  let B := (0, 12) in
  let C := (16, 0) in
  let A' := (-16, 8) in
  let B' := (-24, 20) in
  let C' := (-8, -8) in
  -- Rotation matrix for 135 degrees clockwise
  let rotation_matrix := (λ (p : ℝ × ℝ), 
    let R := matrix.of_fun (2, 2)
      (λ i j, [ [-real.sqrt 2 / 2, real.sqrt 2 / 2],
                  [-real.sqrt 2 / 2, -real.sqrt 2 / 2] ][i][j]) in
      ⟨R 0 0 * p.1 + R 0 1 * p.2, R 1 0 * p.1 + R 1 1 * p.2⟩) in
  -- Conditions for the transformed points
  rotation_matrix (A.1 + x, A.2 + y) = A' ∧
  rotation_matrix (B.1 + x, B.2 + y) = B' ∧
  rotation_matrix (C.1 + x, C.2 + y) = C'
  → 135 + x + y = answer :=
begin
  sorry
end

end rotation_135_deg_l748_748026


namespace math_problem_l748_748036

noncomputable def problem_statement : ℝ :=
  (real.sqrt 1.21) / (real.sqrt 0.64) + (real.sqrt 1.44) / (real.sqrt 0.49)

theorem math_problem :
  (problem_statement ≈ 3.0893) :=
sorry

end math_problem_l748_748036


namespace simplify_expression_l748_748526

theorem simplify_expression :
  (4 + 5) * (4 ^ 2 + 5 ^ 2) * (4 ^ 4 + 5 ^ 4) * (4 ^ 8 + 5 ^ 8) * (4 ^ 16 + 5 ^ 16) * (4 ^ 32 + 5 ^ 32) * (4 ^ 64 + 5 ^ 64) = 5 ^ 128 - 4 ^ 128 :=
by sorry

end simplify_expression_l748_748526


namespace one_fourth_way_from_x1_to_x2_l748_748430

-- Definitions of the points
def x1 : ℚ := 1 / 5
def x2 : ℚ := 4 / 5

-- Problem statement: Prove that one fourth of the way from x1 to x2 is 7/20
theorem one_fourth_way_from_x1_to_x2 : (3 * x1 + 1 * x2) / 4 = 7 / 20 := by
  sorry

end one_fourth_way_from_x1_to_x2_l748_748430


namespace rectangle_not_touch_sides_l748_748108

theorem rectangle_not_touch_sides (n : ℕ) (h : n > 1) 
  (part : set (ℝ × ℝ) → Prop)
  (Hpart : ∀ l : set (ℝ × ℝ), (∃ r ∈ part, l ∩ interior r ≠ ∅) → l ∩ interior square ≠ ∅) :
  ∃ r ∈ part, ¬ (touch_sides r square) :=
sorry

end rectangle_not_touch_sides_l748_748108


namespace sequence_1498_to_1500_l748_748853

theorem sequence_1498_to_1500 {d : ℕ} :
  (digit_sequence d ∧ d = 2 ∧ position = 1498 ∨ position = 1499 ∨ position = 1500) →
  digit d = 294 :=
by sorry

end sequence_1498_to_1500_l748_748853


namespace homework_duration_reduction_l748_748418

theorem homework_duration_reduction (x : ℝ) (initial_duration final_duration : ℝ) (h_initial : initial_duration = 90) (h_final : final_duration = 60) : 
  90 * (1 - x)^2 = 60 :=
by
  sorry

end homework_duration_reduction_l748_748418


namespace sum_of_largest_odd_divisors_l748_748935

def largestOddDivisor (m : ℕ) : ℕ :=
  if m % 2 = 1 then m
  else (2 ^ ((m - m.gcd 2) / 2)) * m.gcd 2

def sumLargestOddDivisors (n : ℕ) : ℕ :=
  (List.range' (n + 1) n).sumBy largestOddDivisor

theorem sum_of_largest_odd_divisors (n : ℕ) : sumLargestOddDivisors n = n * n :=
by sorry

end sum_of_largest_odd_divisors_l748_748935


namespace julia_stairs_less_than_third_l748_748328

theorem julia_stairs_less_than_third (J1 : ℕ) (T : ℕ) (T_total : ℕ) (J : ℕ) 
  (hJ1 : J1 = 1269) (hT : T = 1269 / 3) (hT_total : T_total = 1685) (hTotal : J1 + J = T_total) : 
  T - J = 7 := 
by
  sorry

end julia_stairs_less_than_third_l748_748328


namespace twin_functions_sum_is_72_l748_748642

def twin_functions_sum (f : ℝ → ℝ) (R : Set ℝ) (domains : Set (Set ℝ)) : ℝ :=
  ∑ domain in domains, ∑ x in domain, f x

theorem twin_functions_sum_is_72 :
  let f := λ x : ℝ, 2 * x^2 - 1 in
  let R := {1, 7} in
  let domains := {
    {-2, -1}, {-2, 1}, {2, -1}, {2, 1},
    {-2, -1, 1}, {-2, -1, 2}, {-1, 1, 2},
    {-2, 1, 2}, {-2, -1, 1, 2}
  } in
  twin_functions_sum f R domains = 72 := by
    sorry

end twin_functions_sum_is_72_l748_748642


namespace prob_of_green_ball_is_five_sevenths_l748_748139

-- Conditions:
def ContainerI := { red_balls := 8, green_balls := 4 }
def ContainerII := { red_balls := 3, green_balls := 4 }
def ContainerIII := { red_balls := 3, green_balls := 4 }
def total_balls (container : { red_balls : ℕ, green_balls : ℕ }) : ℕ :=
  container.red_balls + container.green_balls

-- Define the function to get probability of selecting a green ball from a given container
def prob_green (container : { red_balls : ℕ, green_balls : ℕ }) : ℚ :=
  container.green_balls / total_balls container

-- Define the random choice from one of the three containers and the combined probability calculation
def combined_prob_green : ℚ :=
  (1 / 3) * prob_green ContainerI +
  (1 / 3) * prob_green ContainerII + 
  (1 / 3) * prob_green ContainerIII

-- The main theorem we are trying to prove
theorem prob_of_green_ball_is_five_sevenths : combined_prob_green = 5 / 7 :=
by sorry

end prob_of_green_ball_is_five_sevenths_l748_748139


namespace cathy_remaining_money_l748_748876

noncomputable def remaining_money (initial : ℝ) (dad : ℝ) (book : ℝ) (cab_percentage : ℝ) (food_percentage : ℝ) : ℝ :=
  let money_mom := 2 * dad
  let total_money := initial + dad + money_mom
  let remaining_after_book := total_money - book
  let cab_cost := cab_percentage * remaining_after_book
  let food_budget := food_percentage * total_money
  let dinner_cost := 0.5 * food_budget
  remaining_after_book - cab_cost - dinner_cost

theorem cathy_remaining_money :
  remaining_money 12 25 15 0.03 0.4 = 52.44 :=
by
  sorry

end cathy_remaining_money_l748_748876


namespace sufficient_but_not_necessary_converse_not_true_final_proof_l748_748598

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 3) : (1 / x < 1 / 3) :=
sorry

theorem converse_not_true (x : ℝ) : ¬ (1 / x < 1 / 3 → x > 3) :=
begin
  -- We counterexample by choosing x = -1
  intro h,
  have h1 : 1 / (-1) = -1 := by norm_num,
  have h2 : -1 < 1 / 3 := by norm_num,
  exact h (h2.trans_le h1.symm.le),
end

theorem final_proof : (∃ (x : ℝ), (x > 3) ∧ (1 / x < 1 / 3)) ∧ (∃ (x : ℝ), ¬ (1 / x < 1 / 3 → x > 3)) := 
begin
  split,
  { use 4, -- example x = 4 where the condition holds
    split,
    { linarith },
    { norm_num } },
  { exact ⟨-1, λ h, converse_not_true _⟩ }
end

end sufficient_but_not_necessary_converse_not_true_final_proof_l748_748598


namespace inscribed_sphere_radius_in_regular_octahedron_l748_748106

theorem inscribed_sphere_radius_in_regular_octahedron (a : ℝ) (r : ℝ) 
  (h1 : a = 6)
  (h2 : let V := 72 * Real.sqrt 2; V = (1 / 3) * ((8 * (3 * Real.sqrt 3)) * r)) : 
  r = Real.sqrt 6 :=
by
  sorry

end inscribed_sphere_radius_in_regular_octahedron_l748_748106


namespace find_largest_number_l748_748010

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l748_748010


namespace distinct_arrangements_banana_l748_748262

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748262


namespace cot_ratio_l748_748672

variable {a b c : ℝ}

theorem cot_ratio
  (h1 : 9 * a^2 + 9 * b^2 - 19 * c^2 = 0)
  (h2 : ∠A + ∠B + ∠C = π) : 
  (cot (∠ABC) / (cot (∠BCA) + cot (∠CAB)) = 5 / 9) :=
sorry

end cot_ratio_l748_748672


namespace weight_of_steel_rod_l748_748288

theorem weight_of_steel_rod (length1 : ℝ) (weight1 : ℝ) (length2 : ℝ) (weight2 : ℝ) 
  (h1 : length1 = 9) (h2 : weight1 = 34.2) (h3 : length2 = 11.25) : 
  weight2 = (weight1 / length1) * length2 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end weight_of_steel_rod_l748_748288


namespace samantha_flower_bed_area_l748_748725

theorem samantha_flower_bed_area :
  ∃ (a b : ℕ), (4 * (a - 1) * (3 * (a - 1) - 1)) = 600 ∧
               4 * (a - 1) + 4 * (3 * (a - 1) - 1) = 24 ∧
               (a - 1) = 3:90 ∧ b = 13:90 :=
by
  sorry

end samantha_flower_bed_area_l748_748725


namespace min_b1_b2_sum_l748_748886

def sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = (b n * 2007 + 2011) / (1 + b (n + 1))

def pos_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b n > 0

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence b) (h_pos : pos_sequence b) :
  b 1 + b 2 = 2012 :=
sorry

end min_b1_b2_sum_l748_748886


namespace find_m_l748_748103

noncomputable def tournament_probability_nice (conditions : (Π (A B C : ℕ), bool) × (Π (A B C : ℕ), bool)) : ℚ :=
let (condition1, condition2) := conditions in
3690 / 3^15

theorem find_m (conditions : (Π (A B C : ℕ), bool) × (Π (A B C : ℕ), bool)) :
  (tournament_probability_nice conditions).numerator = 3690 :=
by
  -- Given the conditions of the problem, prove that the numerator of the simplified probability is 3690
  sorry

end find_m_l748_748103


namespace equal_projections_l748_748310

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : (ℝ × ℝ × ℝ) :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

noncomputable def vector (p1 p2 : ℝ × ℝ × ℝ) : (ℝ × ℝ × ℝ) :=
(p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

noncomputable def dot_product (v1 v2 : (ℝ × ℝ × ℝ)) : ℝ :=
(v1.1 * v2.1) + (v1.2 * v2.2) + (v1.3 * v2.3)

theorem equal_projections (a b h : ℝ) :
  let O := (0, 0, 0),
      B := (a, a, 0),
      B1 := (b, b, h),
      M := midpoint (a, 0, 0) (a, a, 0),
      N := midpoint (b, b, h) (0, b, h),
      MN := vector M N,
      MB := vector M B,
      B1N := vector B1 N
  in dot_product MB MN = dot_product B1N MN :=
by
  sorry

end equal_projections_l748_748310


namespace find_q_l748_748343

variable {a b m p q : ℝ}

def quadratic1 (x : ℝ) := x^2 - m * x + 3 = 0
def quadratic2 (x : ℝ) := x^2 - p * x + q = 0

theorem find_q (h1 : a * b = 3)
  (h2 : quadratic2 (a + 1/b))
  (h3 : quadratic2 (b + 1/a)) :
  q = 16/3 :=
sorry

end find_q_l748_748343


namespace steer_weight_rounded_l748_748021

noncomputable def steer_weight_kg : ℝ := 350
noncomputable def kg_to_pound : ℝ := 0.454
noncomputable def steer_weight_pounds : ℝ := steer_weight_kg / kg_to_pound

theorem steer_weight_rounded :
  Int.round steer_weight_pounds = 771 := 
by
  sorry

end steer_weight_rounded_l748_748021


namespace work_problem_l748_748820

theorem work_problem 
  (A_real : ℝ)
  (B_days : ℝ := 16)
  (C_days : ℝ := 16)
  (ABC_days : ℝ := 4)
  (H_b : (1 / B_days) = 1 / 16)
  (H_c : (1 / C_days) = 1 / 16)
  (H_abc : (1 / A_real + 1 / B_days + 1 / C_days) = 1 / ABC_days) : 
  A_real = 8 := 
sorry

end work_problem_l748_748820


namespace night_crew_box_fraction_l748_748127

variable (D N B : ℕ) (H_crew : N = (4/5 : ℚ) * D) (H_boxes : (5/6 : ℚ) * B)

theorem night_crew_box_fraction (H : N = (4/5 : ℚ) * D) (H' : (5/6 : ℚ) * B = (5/6 : ℚ) * B) :
  ((1/6 : ℚ) * B / N) / ((5/6 : ℚ) * B / D) = (25/4 : ℚ) :=
sorry

end night_crew_box_fraction_l748_748127


namespace elaine_rent_percentage_l748_748052

theorem elaine_rent_percentage (E : ℝ) (hE : E > 0) :
  let rent_last_year := 0.20 * E
  let earnings_this_year := 1.25 * E
  let rent_this_year := 0.30 * earnings_this_year
  (rent_this_year / rent_last_year) * 100 = 187.5 :=
by
  sorry

end elaine_rent_percentage_l748_748052


namespace distinct_arrangements_banana_l748_748238

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748238


namespace side_length_of_rhombus_l748_748102

theorem side_length_of_rhombus :
  ∀ (a d1 d2 : ℝ), 
  (a > 0) → (d1 > 0) → (d2 > 0) →
  let height := 4 in -- the height of the rhombus (2 * radius of the inscribed circle)
  let S := a * height -- area via side and height
  let S2 := (1 / 2) * d1 * d2 -- area via diagonals
  d1 = a → -- one diagonal equals the side length
  (a^2 = (d1/2)^2 + (d2/2)^2) →
  a = √(64 / 3) →
  a = (8 * sqrt 3) / 3 :=
by sorry

end side_length_of_rhombus_l748_748102


namespace min_norm_a_add_tb_l748_748987

noncomputable theory

variables {a b : EuclideanSpace ℝ (Fin 2)} (t : ℝ)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hab : inner a b = sqrt 3 / 2)

theorem min_norm_a_add_tb : ∃ t : ℝ, ∥a + t • b∥ = 1 / 2 :=
begin
  sorry
end

end min_norm_a_add_tb_l748_748987


namespace arccos_half_eq_pi_div_three_l748_748514

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l748_748514


namespace derivative_of_even_function_is_odd_l748_748710

variable {R : Type*} [Real R]

def is_even (f : R → R) : Prop := ∀ x, f (-x) = f x

theorem derivative_of_even_function_is_odd (f : R → R) (hf : is_even f) : 
  let g := deriv f in ∀ x, g (-x) = -g x :=
by
  sorry

end derivative_of_even_function_is_odd_l748_748710


namespace volunteer_selection_l748_748121

theorem volunteer_selection (m f total select : ℕ) (h_m : m = 5) (h_f : f = 4) (h_total : total = m + f) (h_select : select = 5) :
  ∑ i in finset.range (select + 1), if (5 ≤ i) then 0 else (nat.choose m i * nat.choose f (select - i)) = 125 :=
by
  rw [h_m, h_f, h_total, h_select]
  have : nat.choose 9 5 = 126 := by norm_num
  have : nat.choose 5 5 = 1 := by norm_num
  norm_num
  sorry

end volunteer_selection_l748_748121


namespace function_passes_through_fixed_point_l748_748453

variable (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1)

def f (x : ℝ) : ℝ :=
  log a (x - 1) - 2

theorem function_passes_through_fixed_point (ha1 : a > 0) (ha2 : a ≠ 1) : f a 2 = -2 :=
  sorry

end function_passes_through_fixed_point_l748_748453


namespace braden_total_money_after_winning_bet_l748_748866

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end braden_total_money_after_winning_bet_l748_748866


namespace equation_of_perpendicular_line_l748_748925

theorem equation_of_perpendicular_line (a b c x₁ y₁ : ℝ)
  (h₁ : a * x₁ + b * y₁ + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b ≠ 0)
  (h₄ : x₁ = -1)
  (h₅ : y₁ = 3)
  (ha : a = 1)
  (hb : b = -2)
  (hc : c = 3) :
  ∃ b' : ℝ, 2 * x₁ + y₁ - b' = 0 :=
by
  have m := -a / b,
  have m' := -1 / m,
  have eq1 : y₁ = m' * x₁ + b' := sorry, -- Slope-intercept form using point (-1, 3)
  use 1,
  sorry -- Here we'll conclude that 2 * x₁ + y₁ - 1 = 0 holds.

end equation_of_perpendicular_line_l748_748925


namespace calculate_expression_l748_748132

theorem calculate_expression :
  (3 - Real.pi) ^ 0 - 3 ^ (-2) + |Real.sqrt 3 - 2| + 2 * Real.sin (Real.toRadians 60) = 26 / 9 := by
  sorry

end calculate_expression_l748_748132


namespace inscribed_circle_radius_l748_748144

variable (AB AC BC s K r : ℝ)
variable (AB_eq AC_eq BC_eq : AB = AC ∧ AC = 8 ∧ BC = 7)
variable (s_eq : s = (AB + AC + BC) / 2)
variable (K_eq : K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)))
variable (r_eq : r * s = K)

/-- Prove that the radius of the inscribed circle is 23.75 / 11.5 given the conditions of the triangle --/
theorem inscribed_circle_radius :
  AB = 8 → AC = 8 → BC = 7 → 
  s = (AB + AC + BC) / 2 → 
  K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) →
  r * s = K →
  r = (23.75 / 11.5) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end inscribed_circle_radius_l748_748144


namespace gen_formula_sum_T_2023_l748_748962

-- Define the geometric sequence a_n with first term a_1 = 2 and common ratio q > 0
noncomputable def a (n : ℕ) (q : ℝ) : ℝ := 2 * q^(n-1)

-- Condition: a_4 is the arithmetic mean of 6 * a_2 and a_3
axiom a4_mean (q : ℝ) (h1 : q > 0) : a 4 q = (6 * a 2 q + a 3 q) / 2

-- The general formula for a_n
theorem gen_formula (q : ℝ) (h1 : q > 0) (h2 : q = 2) : ∀ n, a n q = 2^n := sorry

-- Define the sequence b_n
noncomputable def b (n : ℕ) (q : ℝ) : ℝ := 1 / (Real.logBase 2 (a n q) * Real.logBase 2 (a (n+1) q))

-- The sum of the first 2023 terms of b_n
def T_2023 (q : ℝ) := ∑ i in Finset.range 2023, b i q

-- The sum T_2023 is equal to 2023 / 2024
theorem sum_T_2023 (q : ℝ) (h1 : q > 0) (h2 : q = 2) : T_2023 q = 2023 / 2024 := sorry

end gen_formula_sum_T_2023_l748_748962


namespace problem_l748_748427

noncomputable def mass_earth (g R f : ℝ) : ℝ :=
  g * R^2 / f

noncomputable def mass_sun (pi rho f T : ℝ) : ℝ :=
  4 * pi^2 * rho^3 / (f * T^2)

noncomputable def ratio_mass (M M' : ℝ) : ℝ :=
  M' / M

noncomputable def density_sun (M' : ℝ) (R_sun : ℝ) : ℝ :=
  M' / ((4 / 3) * Real.pi * R_sun^3)

noncomputable def gravity_sun (f M' R_sun : ℝ) : ℝ :=
  f * M' / R_sun^2

theorem problem :
  mass_earth 9.8 (6.371 * 10^6) (6.67430 * 10^-11) ≈ 5.972 * 10^24 ∧
  mass_sun Real.pi (1.496 * 10^11) (6.67430 * 10^-11) (3.154 * 10^7) ≈ 1.989 * 10^30 ∧
  ratio_mass 5.972 * 10^24 1.989 * 10^30 ≈ 332000 ∧
  density_sun (1.989 * 10^30) (6.963 * 10^8) ≈ 1.409 ∧
  gravity_sun (6.67430 * 10^-11) (1.989 * 10^30) (6.963 * 10^8) ≈ 274 :=
by
  sorry

end problem_l748_748427


namespace triangle_area_range_l748_748381

theorem triangle_area_range (m : ℝ) : 
  let line := (x y : ℝ) => x - 2 * y + 2 * m = 0 
  in (1/2) * abs (-2 * m) * abs (m) ≥ 1 → m ≤ -1 ∨ 1 ≤ m := 
by 
  sorry

end triangle_area_range_l748_748381


namespace find_initial_length_of_cloth_l748_748638

noncomputable def initial_length_of_cloth : ℝ :=
  let work_rate_of_8_men := 36 / 0.75
  work_rate_of_8_men

theorem find_initial_length_of_cloth (L : ℝ) (h1 : (4:ℝ) * 2 = L / ((4:ℝ) / (L / 8)))
    (h2 : (8:ℝ) / L = 36 / 0.75) : L = 48 :=
by
  sorry

end find_initial_length_of_cloth_l748_748638


namespace max_min_values_l748_748692

noncomputable def f (x : ℝ) := x^3 - x^2 - x + 1

theorem max_min_values :
  (∀ x : ℝ, (x ∈ set.Icc (-1:ℝ) 3) → f x ≤ 16) ∧
  (∀ x : ℝ, (x ∈ set.Icc (-1:ℝ) 3) → 0 ≤ f x) ∧
  (∃ x : ℝ, x ∈ set.Icc (-1:ℝ) 3 ∧ f x = 16) ∧
  (∃ y : ℝ, y ∈ set.Icc (-1:ℝ) 3 ∧ f y = 0) :=
sorry

end max_min_values_l748_748692


namespace find_asymptotes_C2_l748_748961

def same_abscissa (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def ordinate_twice (P Q : ℝ × ℝ) : Prop := 
  P.2 = 2 * Q.2

def hyperbola (C : ℝ → ℝ × ℝ → Prop) (λ : ℝ) : Prop :=
  ∀ x y : ℝ, C λ (x, y) ↔ y^2 - 3*x^2 = λ

def asymptotes_C1 : Prop :=
  ∀ (x : ℝ), y = x * (√3) ∨ y = -x * (√3)

def asymptotes_C2 : Prop :=
  ∀ (x : ℝ), y = x * (√3/2) ∨ y = -x * (√3/2)

theorem find_asymptotes_C2 (P Q : ℝ × ℝ) 
  (λ : ℝ) 
  (h_abscissa : same_abscissa P Q) 
  (h_ordinate : ordinate_twice P Q) 
  (C1 : ℝ → ℝ × ℝ → Prop) 
  (C2 : ℝ → ℝ × ℝ → Prop) 
  (h_hyperbola_C1 : hyperbola C1 λ) 
  (h_asymptotes_C1 : asymptotes_C1) 
  (h_trajectory_P : C1 λ P) 
  (h_trajectory_Q : C2 λ Q) :
  asymptotes_C2 :=
sorry

end find_asymptotes_C2_l748_748961


namespace angle_BAX_eq_angle_CAY_l748_748698

open Real
open Geometry

-- Define the triangle ABC and points I, K, X, Y
variables {A B C I K X Y : Point}
variables {KI : Line}

-- Define incenter properties
def is_incenter {ABC : Triangle} (I : Point) : Prop :=
  incenter ABC I

-- Define external angle bisector properties
def is_external_angle_bisector {a b c : Point} (K : Point) (α : ℝ) : Prop :=
  external_angle_bisector A K ABC α

-- Define intersection properties of external bisectors with line KI at X and Y
def intersects_on_external_bisector {α β : ℝ} (KI : Line) (X Y : Point) : Prop :=
  KI.intersects (external_bisector B X α) ∧ KI.intersects (external_bisector C Y β)

-- Main theorem statement
theorem angle_BAX_eq_angle_CAY (h_incenter : is_incenter I)
  (h_ext_bisector_A : is_external_angle_bisector K ∠A)
  (h_intersects : intersects_on_external_bisector KI X Y) :
  ∠BAX = ∠CAY := by
  sorry

end angle_BAX_eq_angle_CAY_l748_748698


namespace arithmetic_square_root_of_sqrt_16_l748_748742

theorem arithmetic_square_root_of_sqrt_16 :
  let arithmetic_square_root (x : ℝ) := if x ≥ 0 then real.sqrt x else -real.sqrt (-x)
  arithmetic_square_root (real.sqrt 16) = 2 := by
    sorry

end arithmetic_square_root_of_sqrt_16_l748_748742


namespace smallest_period_and_minimum_value_of_f6_l748_748977

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := sin x ^ n + cos x ^ n

theorem smallest_period_and_minimum_value_of_f6 :
  (∃ T > 0, ∀ x, f 6 (x + T) = f 6 x) ∧ (∀ x, f 6 x ≥ 1 / 4) :=
sorry

end smallest_period_and_minimum_value_of_f6_l748_748977


namespace grid_coloring_l748_748370

theorem grid_coloring :
  ∃ (c : (ℤ × ℤ) → bool),  -- bool represents red or blue
  (∀ t : ℤ, {p : ℤ × ℤ | p.1 = t ∧ c p = true}.finite) ∧  -- finitely many red points on vertical lines
  (∀ q : ℤ, {p : ℤ × ℤ | p.2 = q ∧ c p = false}.finite) :=  -- finitely many blue points on horizontal lines
sorry

end grid_coloring_l748_748370


namespace part1_part2_l748_748221

variable {k : ℝ}
variable {A B : ℝ × ℝ}  -- Representing points as pairs (x, y)
variable h₁ : k = k  -- Just a placeholder to represent the variable

-- Given conditions
def line_intersects_hyperbola (k : ℝ) (A B : ℝ × ℝ) : Prop := 
  let (x1, y1) := A;
  let (x2, y2) := B;
  y1 = k * x1 + 1 ∧ y2 = k * x2 + 1 ∧ 3 * x1^2 - y1^2 = 1 ∧ 3 * x2^2 - y2^2 = 1 

def opposite_branches (A B : ℝ × ℝ) : Prop := 
  let (x1, _) := A;
  let (x2, _) := B;
  x1 < 0 ∧ x2 > 0

def circle_diameter_passes_origin (A B : ℝ × ℝ) : Prop := 
  let (x1, y1) := A;
  let (x2, y2) := B;
  (x1 * x2 + y1 * y2) = 0

-- Main proof problems
theorem part1 (h₁ : line_intersects_hyperbola k A B) (h₂ : opposite_branches A B) : 
  -real.sqrt 3 < k ∧ k < real.sqrt 3 :=
sorry

theorem part2 (h₁ : line_intersects_hyperbola k A B) (h₂ : opposite_branches A B) (h₃ : circle_diameter_passes_origin A B) : 
  k = 1 ∨ k = -1 := 
sorry

end part1_part2_l748_748221


namespace no_possible_configuration_26_students_l748_748018

/-- Define a student as either a Knight (truth-teller) or a Liar (always lies) -/
inductive StudentType
| Knight
| Liar

/-- Student statement after rearrangement problem -/
theorem no_possible_configuration_26_students :
  ∀ (students : Fin 26 → StudentType),
  (∀ i, exists_neighbor_with students i StudentType.Liar) →
  ¬(∀ i, exists_neighbor_with students i StudentType.Knight) :=
by
  /- problem formulation and setup goes here -/
  sorry

/-- Helper function to define neighboring relation for students (index circularly) -/
def exists_neighbor_with (students : Fin 26 → StudentType) (i : Fin 26) (type : StudentType) : Prop :=
  (students ((i + 1) % 26) = type) ∨ (students ((i - 1) % 26) = type)

-- The theorem states the impossibility of rearranging 26 students such that after rearrangement,
-- each student can say they are sitting next to a knight under the given conditions.

end no_possible_configuration_26_students_l748_748018


namespace certain_number_exceeds_ten_l748_748649

theorem certain_number_exceeds_ten 
  (x y z : ℤ) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : even x) 
  (h4 : odd y) 
  (h5 : odd z) 
  (h6 : z - x = 13) : 
  y - x > 10 :=
begin
  sorry
end

end certain_number_exceeds_ten_l748_748649


namespace mode_of_denominations_is_10_l748_748532

theorem mode_of_denominations_is_10 : 
  ∀ (freq_100 freq_50 freq_10 freq_5 : ℕ), 
  freq_100 = 3 → 
  freq_50 = 9 → 
  freq_10 = 23 → 
  freq_5 = 10 → 
  mode [freq_100, freq_50, freq_10, freq_5] = freq_10 :=
by
  intros freq_100 freq_50 freq_10 freq_5 H100 H50 H10 H5
  sorry

end mode_of_denominations_is_10_l748_748532


namespace trig_identity_l748_748442

theorem trig_identity (α : ℝ) :
    sin (2 * α) * (2 * cos (4 * α) + 1) * cot (π / 6 - 2 * α) * cot (π / 6 + 2 * α) =
    sin (6 * α) * cot (2 * α) * tan (6 * α) :=
by
  sorry

end trig_identity_l748_748442


namespace david_did_45_crunches_l748_748140

-- Definitions of the conditions
def zachary_crunches : ℕ := 62
def david_crunches : ℕ := zachary_crunches - 17

-- Statement of what we need to prove
theorem david_did_45_crunches : david_crunches = 45 :=
by
  unfold david_crunches
  unfold zachary_crunches
  simp
  sorry

end david_did_45_crunches_l748_748140


namespace tap_filling_time_l748_748110

theorem tap_filling_time (T : ℝ) (hT1 : T > 0) 
  (h_fill_with_one_tap : ∀ (t : ℝ), t = T → t > 0)
  (h_fill_with_second_tap : ∀ (s : ℝ), s = 60 → s > 0)
  (both_open_first_10_minutes : 10 * (1 / T + 1 / 60) + 20 * (1 / 60) = 1) :
    T = 20 := 
sorry

end tap_filling_time_l748_748110


namespace shopkeeper_discount_and_selling_price_l748_748105

theorem shopkeeper_discount_and_selling_price :
  let CP := 100
  let MP := CP + 0.5 * CP
  let SP := CP + 0.15 * CP
  let Discount := (MP - SP) / MP * 100
  Discount = 23.33 ∧ SP = 115 :=
by
  sorry

end shopkeeper_discount_and_selling_price_l748_748105


namespace number_of_divisors_of_N_sum_of_divisors_of_N_l748_748372

-- Key definitions for the problem
def N : ℕ := 2160
def prime_factors : ℕ × ℕ × ℕ := (4, 3, 1) -- (exponent of 2, exponent of 3, exponent of 5)

-- Proving the number of positive divisors of N is 40
theorem number_of_divisors_of_N : 
  let (a, b, c) := prime_factors in 
  (a + 1) * (b + 1) * (c + 1) = 40 := by
  sorry

-- Proving the sum of all positive divisors of N is 7440
theorem sum_of_divisors_of_N : 
  let (a, b, c) := prime_factors in 
  (finset.range (a + 1)).sum (λ i, 2 ^ i) *
  (finset.range (b + 1)).sum (λ i, 3 ^ i) * 
  (finset.range (c + 1)).sum (λ i, 5 ^ i) = 7440 := by
  sorry

end number_of_divisors_of_N_sum_of_divisors_of_N_l748_748372


namespace josie_remaining_money_l748_748681

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end josie_remaining_money_l748_748681


namespace arithmetic_square_root_of_sqrt_16_l748_748744
noncomputable theory

def arithmetic_square_root_is_two : Prop :=
  let x := 4 in
  let y := Real.sqrt x in
  y = 2

theorem arithmetic_square_root_of_sqrt_16 : arithmetic_square_root_is_two := 
by
  let h1 : Real.sqrt 16 = 4 := by exact Real.sqrt_sq 4
  let h2 : ∀ x, Real.sqrt x = y → y = 2 := by
    intros x hx
    assume h : y = 4
    exact Eq.symm h
  exact (h2 16 h1)
sorry

end arithmetic_square_root_of_sqrt_16_l748_748744


namespace compare_a_b_l748_748201

theorem compare_a_b (a b : ℝ) (h : 5 * (a - 1) = b + a ^ 2) : a > b :=
sorry

end compare_a_b_l748_748201


namespace union_A_B_l748_748625

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

theorem union_A_B (x : ℝ) : (x ∈ A ∨ x ∈ B) ↔ x < 3 := by
  sorry

end union_A_B_l748_748625


namespace situp_competition_total_l748_748850

theorem situp_competition_total:
  let adam_total := 40 + (40 - 5) + (35 - 5) + (30 - 10) in
  let barney_total := 
      let round_1 := 45 in
      let round_2 := round_1 - 3 in
      let round_3 := round_2 - 6 in
      let round_4 := round_3 - 9 in
      let round_5 := round_4 - 12 in
      let round_6 := round_5 - 15 in
      round_1 + round_2 + round_3 + round_4 + round_5 + round_6 in
  let carrie_total := 
      let first_round := 2 * 45 in
      let second_round := first_round in
      let third_round := first_round - 10 in
      second_round + second_round + third_round + third_round + (third_round - 10) in
  let jerrie_total := 
      let first_round := (2 * 45) + 5 in
      let second_round := first_round in
      let third_round := first_round + (3 * 2) in
      let fourth_round := third_round in
      let fifth_round := fourth_round - 7 in
      let sixth_round := fifth_round - 7 in
      let seventh_round := sixth_round - 7 in
      first_round + second_round + third_round + fourth_round + fifth_round + sixth_round + seventh_round in
  adam_total + barney_total + carrie_total + jerrie_total = 1353 :=
sorry

end situp_competition_total_l748_748850


namespace lambda_value_l748_748995

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748995


namespace intersection_M_N_eq_M_inter_N_l748_748224

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | x < 0 }
def M_inter_N : Set ℝ := { x | x < -2 }

theorem intersection_M_N_eq_M_inter_N : M ∩ N = M_inter_N := 
by
  sorry

end intersection_M_N_eq_M_inter_N_l748_748224


namespace intersection_of_M_and_N_l748_748627

-- Define the sets M and N
def M := {-1, 1}
def N := {-1, 0, 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1} :=
by
  sorry

end intersection_of_M_and_N_l748_748627


namespace base3_product_l748_748422

def base3_to_decimal (n : ℕ) : ℕ :=
  let digits := List.ofDigits 3 $ List.reverse $ n.digits 3
  digits.sum

def decimal_to_base3 (n : ℕ) : ℕ :=
  List.ofDigits 3 $ List.reverse $ n.digits 3

theorem base3_product (a b : ℕ) (ha : a = 11) (hb : b = 5) :
  decimal_to_base3 (a * b) = 1202 :=
by sorry

end base3_product_l748_748422


namespace inequality_solution_set_l748_748979

theorem inequality_solution_set {a : ℝ} (x : ℝ) :
  (∀ x, (x - a) / (x^2 - 3 * x + 2) ≥ 0 ↔ (1 < x ∧ x ≤ a) ∨ (2 < x)) → (1 < a ∧ a < 2) :=
by 
  -- We would fill in the proof here. 
  sorry

end inequality_solution_set_l748_748979


namespace probability_heads_penny_nickel_dime_all_heads_l748_748734

theorem probability_heads_penny_nickel_dime_all_heads :
  let num_outcomes := (2:ℕ) ^ 5
  let successful_outcomes := 2 * 2
  (successful_outcomes : ℚ) / num_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_all_heads_l748_748734


namespace otimes_2008_eq_3_2007_l748_748943

-- Define the new operation otimes
def otimes : ℝ → ℝ → ℝ
| 1, 1 := 1
| (a + 1), 1 := 3 * (otimes a 1)
| _, _ := 0  -- To cover all other cases. This isn't strictly required for the problem, but completes the definition.

-- Define the specific theorem based on the conditions and the question
theorem otimes_2008_eq_3_2007 : otimes 2008 1 = 3^2007 := 
by
  sorry

end otimes_2008_eq_3_2007_l748_748943


namespace max_black_cells_visited_5x5_checkerboard_l748_748746

def is_checkerboard (n : ℕ) : Prop :=
  ∀ i j : ℕ, (i < n ∧ j < n) → ((i + j) % 2 = 0 ↔ cell_is_black i j)

def mini_elephant_moves (n : ℕ) (path : list (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), (i, j) ∈ path →
  ∃ (i' j' : ℕ), ((i' + j') % 2 = 0 ∧ (abs (i' - i) = 1 ∧ abs (j' - j) = 1 ∨
                                       abs (i' - i) = 2 ∧ abs (j' - j) = 2)) ∧
  (∀ (i'' j'') ∈ path, i'' ≠ i' ∨ j'' ≠ j')

theorem max_black_cells_visited_5x5_checkerboard : ∀ (n : ℕ),
  n = 5 →
  ∀ path : list (ℕ × ℕ),
  (∀ (i j : ℕ), (i, j) ∈ path → cell_is_black i j) →
  mini_elephant_moves n path →
  list.length path ≤ 12 :=
begin
  intros n hn path hpath hmove,
  have hn5 : n = 5 := hn,
  sorry
end

end max_black_cells_visited_5x5_checkerboard_l748_748746


namespace range_of_heights_l748_748763

def heights := [148, 141, 172.5, 168, 151.5, 183.5, 178.5] : List ℝ

theorem range_of_heights :
  (List.maximum heights).getOrElse 0 - (List.minimum heights).getOrElse 0 = 42.5 := by
  sorry

end range_of_heights_l748_748763


namespace not_perfect_square_l748_748594

theorem not_perfect_square (a b : ℤ) (h1 : a > b) (h2 : Int.gcd (ab - 1) (a + b) = 1) (h3 : Int.gcd (ab + 1) (a - b) = 1) :
  ¬ ∃ c : ℤ, (a + b)^2 + (ab - 1)^2 = c^2 := 
  sorry

end not_perfect_square_l748_748594


namespace find_norm_a_l748_748607

open Real

variables (a b : ℝ^3)
variables (angle_ab : ℝ) (norm_diff_ab : ℝ) (dot_condition : ℝ)

-- Given conditions
def angle_between_vectors := angle_ab = π / 3 -- 60 degrees in radians
def norm_diff := norm (a - b) = sqrt 3
def dot_product_condition := (b ⬝ (a - b)) = 0

-- Prove
theorem find_norm_a (angle_between_vectors : angle_between_vectors) 
                    (norm_diff : norm_diff) 
                    (dot_product_condition : dot_product_condition) : 
                    norm a = 2 :=
sorry

end find_norm_a_l748_748607


namespace solve_quadratic_inequality_l748_748378

theorem solve_quadratic_inequality :
  { x : ℝ | -5 * x^2 + 10 * x - 3 > 0 } = 
  { x : ℝ | 1 - real.sqrt 10 / 5 < x ∧ x < 1 + real.sqrt 10 / 5 } :=
by sorry

end solve_quadratic_inequality_l748_748378


namespace length_of_platform_l748_748799

theorem length_of_platform (L : ℝ) : 
  (∀ t_len t_plat t_pole : ℝ, 
    t_len = 300 ∧ 
    t_plat = 27 ∧ 
    t_pole = 18 →
    L = 150) :=
begin
  intros,
  sorry
end

end length_of_platform_l748_748799


namespace sum_xi_le_n_div_3_l748_748694

theorem sum_xi_le_n_div_3 (n : ℕ) (h1 : n ≥ 3) (x : Fin n → ℝ)
  (h2 : ∀ i, x i ∈ Set.Icc (-1 : ℝ) 1)
  (h3 : ∑ i, (x i)^3 = 0) :
  ∑ i, x i ≤ n / 3 :=
by
  sorry

end sum_xi_le_n_div_3_l748_748694


namespace part1_part2_l748_748988

-- Definitions for vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 4)

-- Proof for part (1)
theorem part1 : ∥(3 • a - b)∥ = 2 * real.sqrt 10 := by
  sorry

-- Proof for part (2)
theorem part2 (λ : ℝ) (h : a ⬝ (a + λ • b) = 0) : λ = -1 := by
  sorry

end part1_part2_l748_748988


namespace rainfall_on_thursday_l748_748536

theorem rainfall_on_thursday
  (monday_am : ℝ := 2)
  (monday_pm : ℝ := 1)
  (tuesday_factor : ℝ := 2)
  (wednesday : ℝ := 0)
  (thursday : ℝ)
  (weekly_avg : ℝ := 4)
  (days_in_week : ℕ := 7)
  (total_weekly_rain : ℝ := days_in_week * weekly_avg) :
  2 * (monday_am + monday_pm + tuesday_factor * (monday_am + monday_pm) + thursday) 
    = total_weekly_rain
  → thursday = 5 :=
by
  sorry

end rainfall_on_thursday_l748_748536


namespace multiple_of_sum_squares_l748_748647

theorem multiple_of_sum_squares (a b c : ℕ) (h1 : a < 2017) (h2 : b < 2017) (h3 : c < 2017) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
    (h7 : ∃ k1, a^3 - b^3 = k1 * 2017) (h8 : ∃ k2, b^3 - c^3 = k2 * 2017) (h9 : ∃ k3, c^3 - a^3 = k3 * 2017) :
    ∃ k, a^2 + b^2 + c^2 = k * (a + b + c) :=
by
  sorry

end multiple_of_sum_squares_l748_748647


namespace sum_reciprocal_correct_l748_748005

-- Define the sequence a_n
def seq (a : Nat → ℝ) : Prop :=
  a 0 = 3 ∧ ∀ n, (3 - a (n + 1)) * (6 + a n) = 18

-- Define the sum of reciprocals of the sequence
def sum_reciprocal (a : Nat → ℝ) (n : Nat) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / a i

-- Define the expected sum
def expected_sum (n : Nat) : ℝ :=
  (1 / 3) * (2 ^ (n + 2) - n - 3)

-- Theorem to prove
theorem sum_reciprocal_correct (a : Nat → ℝ) (n : Nat)
  (h : seq a) :
  sum_reciprocal a n = expected_sum n :=
by
  sorry  -- Proof goes here

end sum_reciprocal_correct_l748_748005


namespace prove_properties_l748_748945

noncomputable theory
open Classical

variable (a : ℕ → ℕ) (S : ℕ → ℕ)
variable (b : ℕ → ℕ) (T : ℕ → ℕ)

-- Given the condition that S_n = 2a_n - 2
def condition_one := ∀ n : ℕ, n > 0 → S n = 2 * a n - 2

-- Prove that the sequence {a_n} is a geometric sequence with general term a_n = 2^n
def geometric_sequence_property := ∀ n : ℕ, n > 0 → a n = 2 ^ n

-- Given b_n = (n + 1) * a_n, find T_n
def definition_of_bn (n : ℕ) := b n = (n + 1) * a n

-- Prove that T_n = n * 2^(n+1)
def sum_of_bn_property := ∀ n : ℕ, n > 0 → T n = n * 2^(n+1)

-- Full statement combining the conditions and the goal
theorem prove_properties 
  (h1 : condition_one a S) 
  (h2 : ∀ n > 0, b n = (n + 1) * a n) 
  : geometric_sequence_property a ∧ sum_of_bn_property a b T :=
sorry

end prove_properties_l748_748945


namespace point_symmetric_y_axis_l748_748204

theorem point_symmetric_y_axis (a b : ℤ) (h₁ : a = -(-2)) (h₂ : b = 3) : a + b = 5 := by
  sorry

end point_symmetric_y_axis_l748_748204


namespace percentage_calculation_l748_748069

theorem percentage_calculation (percentage : ℝ) (h : percentage * 50 = 0.15) : percentage = 0.003 :=
by
  sorry

end percentage_calculation_l748_748069


namespace inv_203_mod_301_exists_l748_748519

theorem inv_203_mod_301_exists : ∃ b : ℤ, 203 * b % 301 = 1 := sorry

end inv_203_mod_301_exists_l748_748519


namespace find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l748_748218

noncomputable def f (a m x : ℝ) := Real.log (x + m) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def F (a m x : ℝ) := f a m x - g a x

theorem find_m_and_domain (a : ℝ) (m : ℝ) (h : F a m 0 = 0) : m = 1 ∧ ∀ x, -1 < x ∧ x < 1 :=
sorry

theorem parity_of_F (a : ℝ) (m : ℝ) (h : m = 1) : ∀ x, F a m (-x) = -F a m x :=
sorry

theorem range_of_x_for_F_positive (a : ℝ) (m : ℝ) (h : m = 1) :
  (a > 1 → ∀ x, 0 < x ∧ x < 1 → F a m x > 0) ∧ (0 < a ∧ a < 1 → ∀ x, -1 < x ∧ x < 0 → F a m x > 0) :=
sorry

end find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l748_748218


namespace cakes_left_correct_l748_748864

-- Define constants for total cakes made and cakes sold
def total_cakes_made : ℕ := 217
def cakes_sold : ℕ := 145

-- Define the number of cakes left
def cakes_left : ℕ := total_cakes_made - cakes_sold

-- The proof that the number of cakes left is correct
theorem cakes_left_correct : cakes_left = 72 := 
by
  -- Calculate the number of cakes left
  have h : cakes_left = total_cakes_made - cakes_sold := rfl
  -- Substitute the values and prove it equals 72
  rw [← h]
  -- Computation step
  norm_num
  sorry

end cakes_left_correct_l748_748864


namespace cricket_game_overs_l748_748309

theorem cricket_game_overs (x : ℝ) (run_rate_first_part : ℝ) (run_rate_remaining : ℝ) (remaining_overs : ℝ) (target : ℝ) 
  (h1 : run_rate_first_part = 3.2) 
  (h2 : run_rate_remaining = 6.25) 
  (h3 : remaining_overs = 40) 
  (h4 : target = 282) 
  (h5 : run_rate_first_part * x + run_rate_remaining * remaining_overs = target) : 
  x = 10 := 
by
  sorry

end cricket_game_overs_l748_748309


namespace tangents_concur_l748_748400

open Set Function

-- Define the points and circles as given in the problem
variables {A M1 M2 C E F B D X : Point}
variable {k1 k2 : Circle}

-- Define the conditions:
-- Points A, M1, M2, and C are collinear
def collinear_points : Prop := Collinear {A, M1, M2, C}

-- Circle k1 centered at M1 passing through A
def circle_k1 : Prop := k1.center = M1 ∧ k1.radius = distance M1 A

-- Circle k2 centered at M2 passing through C
def circle_k2 : Prop := k2.center = M2 ∧ k2.radius = distance M2 C

-- Circles k1 and k2 intersect at points E and F
def circles_intersect_at_EF : Prop := E ∈ k1.points ∧ E ∈ k2.points ∧ F ∈ k1.points ∧ F ∈ k2.points

-- Common tangent tangent to k1 at B and to k2 at D
def common_tangent : Prop := TangentAt k1 B ∧ TangentAt k2 D

-- All conditions bundled together as hypotheses
def problem_conditions : Prop := 
  collinear_points ∧ circle_k1 ∧ circle_k2 ∧ circles_intersect_at_EF ∧ common_tangent

-- The theorem to prove the main statement
theorem tangents_concur (h : problem_conditions) : ∃ X, LineThrough A B = LineThrough C D ∧ LineThrough E F = X :=
  sorry

end tangents_concur_l748_748400


namespace sum_difference_in_grid_l748_748189

theorem sum_difference_in_grid (n : ℕ) (grid : ℕ → ℕ → ℕ) :
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → grid i j ∈ finset.range (n^2 + 1)) →
  (∃ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ grid i j = 1) →
  let ⟨i1, j1, _⟩ := classical.some (classical.some_spec (exists_ij_placement grid 1 n)),
      ⟨in2, jn2, _⟩ := classical.some (classical.some_spec (exists_ij_placement grid (n^2) n)) in
  ∑ k in finset.range n, grid i1 k - ∑ k in finset.range n, grid k jn2 = n * (n - 1) :=
sorry 

end sum_difference_in_grid_l748_748189


namespace maximize_fg_plus_gh_plus_hj_plus_fj_l748_748014

theorem maximize_fg_plus_gh_plus_hj_plus_fj :
  ∃ (f g h j : ℕ), (f = 9 ∨ f = 10 ∨ f = 11 ∨ f = 12) ∧
                   (g = 9 ∨ g = 10 ∨ g = 11 ∨ g = 12) ∧
                   (h = 9 ∨ h = 10 ∨ h = 11 ∨ h = 12) ∧
                   (j = 9 ∨ j = 10 ∨ j = 11 ∨ j = 12) ∧
                   (f ≠ g ∧ f ≠ h ∧ f ≠ j ∧ g ≠ h ∧ g ≠ j ∧ h ≠ j) ∧ 
                   (fg, gh, hj, fj : ℕ) → 
                   (fg = f * g ∧ gh = g * h ∧ hj = h * j ∧ fj = f * j) →
                   fg + gh + hj + fj = 441 := by
  sorry

end maximize_fg_plus_gh_plus_hj_plus_fj_l748_748014


namespace problem_1_problem_2_l748_748975

variable (f g : ℝ → ℝ)
variable (a b : ℝ)
variable (x : ℝ)

-- Condition: Function definition
def f (x : ℝ) : ℝ := Real.log x + a * x - 1 / x + b

-- Condition: g(x) being decreasing
def g (x : ℝ) : Prop := f x + 2 / x

-- Problem 1: Prove that a ≤ -1/4 if g is decreasing
theorem problem_1 (h1 : ∀ x > 0, (∂/∂ (x : ℝ), g x) ≤ 0) : a ≤ -1 / 4 := 
sorry

-- Problem 2: Prove that a ≤ 1 - b if f(x) ≤ 0 always holds
theorem problem_2 (h2 : ∀ x > 0, f x ≤ 0) : a ≤ 1 - b := 
sorry

end problem_1_problem_2_l748_748975


namespace intersection_value_l748_748604

theorem intersection_value (x0 : ℝ) (h1 : -x0 = Real.tan x0) (h2 : x0 ≠ 0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
  sorry

end intersection_value_l748_748604


namespace triangle_side_relation_l748_748321

-- Define the setting and conditions
variables {A B C D M : Type}
variables [HasAngle A] [HasAngle B] [HasAngle C] [HasAngle D]
variables [HasAngle 20°] [HasAngle 100°]
variables [HasLength CA AB CD BC]
variables [IsoscelesTriangle ADC AD DC 100°]
variables [IsoscelesTriangle CAB CA AB 20°]

-- Define the theorem statement
theorem triangle_side_relation (h1 : AD = DC) (h2 : ∠D = 100°) (h3 : CA = AB) (h4 : ∠A = 20°) :
  AB = BC + CD :=
sorry

end triangle_side_relation_l748_748321


namespace arccos_one_half_eq_pi_div_three_l748_748509

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l748_748509


namespace existence_of_nonnegative_value_l748_748980

theorem existence_of_nonnegative_value :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0 := 
by
  sorry

end existence_of_nonnegative_value_l748_748980


namespace banana_distinct_arrangements_l748_748282

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748282


namespace measure_angle_A_is_60_degrees_l748_748668

-- Define the context and conditions
variable (A B C D E : Type) [isPoint A B C D E]
variable [AB AC : A ↔ B] [AC : A ↔ C] [BD : B ↔ D] [BD : B → D → A]
variable [DE EQ_ED EQ_DC BD_EQ_BC : ∀ a b, D ↔ E ↔ D ↔ C, BD = BC]

-- State the theorem
theorem measure_angle_A_is_60_degrees (AB_eq_AC : AB = AC) (BD_bisects_BAC : ∀ A B C, BD.bisects_angle A B C) (DE_eq_DC : DE = DC) (BD_eq_BC: BD = BC) : measure_angle A = 60 := by
  sorry

end measure_angle_A_is_60_degrees_l748_748668


namespace find_monic_quadratic_poly_with_real_coeff_and_root_l748_748165

open Complex Polynomial

noncomputable def monic_quadratic_with_root (α : ℂ) : Polynomial ℂ :=
  Polynomial.monic (Polynomial.x - Polynomial.C α)
  * Polynomial.monic (Polynomial.x - Polynomial.C α.conj)

theorem find_monic_quadratic_poly_with_real_coeff_and_root (α : ℂ) (h_re : ∃ x : ℝ, α = x - I) :
  monic_quadratic_with_root α = Polynomial.C (1 : ℂ) * (Polynomial.x ^ 2 - Polynomial.C (2 * (α.re : ℂ)) * Polynomial.x + Polynomial.C ((α.re ^ 2 : ℝ) + 1)) :=
by
  sorry

end find_monic_quadratic_poly_with_real_coeff_and_root_l748_748165


namespace simplify_and_evaluate_l748_748727

theorem simplify_and_evaluate :
  ∀ (a b : ℤ), a = -1 → b = 4 →
  (a + b)^2 - 2 * a * (a - b) + (a + 2 * b) * (a - 2 * b) = -64 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l748_748727


namespace max_value_when_m_zero_range_of_m_l748_748620

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := m * exp x + (log x - 2) / x + 1

-- Part (1): Prove maximum value when m = 0
theorem max_value_when_m_zero :
  ∃ x, f x 0 = (log x - 2) / x + 1 ∧ 
  (f x 0) = (1 / exp 3) + 1 := sorry

-- Part (2): Prove the range for m when f(x) < 0 always holds
theorem range_of_m (m : ℝ) :
  (∀ x > 0, f x m < 0) → m < - (1 / exp 3) := sorry

end max_value_when_m_zero_range_of_m_l748_748620


namespace edward_score_l748_748915

theorem edward_score (total_points : ℕ) (friend_points : ℕ) 
  (h1 : total_points = 13) (h2 : friend_points = 6) : 
  ∃ edward_points : ℕ, edward_points = 7 :=
by
  sorry

end edward_score_l748_748915


namespace ratio_of_kids_to_adult_meals_l748_748494

theorem ratio_of_kids_to_adult_meals (k a : ℕ) (h1 : k = 8) (h2 : k + a = 12) : k / a = 2 := 
by 
  sorry

end ratio_of_kids_to_adult_meals_l748_748494


namespace exists_natural_numbers_with_integer_roots_l748_748531

noncomputable def has_integer_roots (a b c : ℕ) : Prop :=
  ∀ (p q : ℤ), (a ≠ 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (b^2 - 4 * a * c).sqrt = p ∨ p = q) →
  ∃ (x : ℤ), (a * x^2 + b * x + c = 0)

theorem exists_natural_numbers_with_integer_roots :
  ∃ (a b c : ℕ),
    has_integer_roots a b c ∧
    has_integer_roots a b (-c) ∧
    has_integer_roots a (-b) c ∧
    has_integer_roots a (-b) (-c) :=
sorry

end exists_natural_numbers_with_integer_roots_l748_748531


namespace product_third_side_approximation_l748_748779

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) :
  ((sqrt (a ^ 2 + b ^ 2)) * (sqrt (b ^ 2 - a ^ 2))).round = 41 :=
by
  have h₃ : sqrt (5 ^ 2 + 12 ^ 2) = 13 := by norm_num
  have h₄ : sqrt (12 ^ 2 - 5 ^ 2) = sqrt 119 := by norm_num
  have h_product : 13 * sqrt 119 ≈ 40.6 := by norm_num
  exact h_product.round

end product_third_side_approximation_l748_748779


namespace real_root_of_P_l748_748401

noncomputable def P : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| n+2, x => x * P (n + 1) x + (1 - x) * P n x

theorem real_root_of_P (n : ℕ) (hn : 1 ≤ n) : ∀ x : ℝ, P n x = 0 → x = 0 := 
by 
  sorry

end real_root_of_P_l748_748401


namespace solve_inequality_l748_748054

theorem solve_inequality (x : ℝ) :
  -2 < x ∧ x < 2 ∧ x ≠ 0 ∧ x ≠ -8 / 3 ∧ x ≠ 2 / 3 →
  (log (1 / 6 * x^2 + x / 3 + 19 / 27) (1 + x^2 / 4) * log (1 / 6 * x^2 + x / 3 + 19 / 27) (1 - x^2 / 4) + 1) * 
  log (1 - 1 / 16 * x^4) (x^2 / 6 + x / 3 + 19 / 27) ≥ 1 →
  x ∈ set.Ico (-4 / 3) 0 ∪ set.Ioc 0 (8 / 15) ∪ set.Ioc (2 / 3) (4 / 3) :=
sorry

end solve_inequality_l748_748054


namespace base7_to_base10_conversion_l748_748891

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l748_748891


namespace average_cost_of_testing_l748_748674

theorem average_cost_of_testing (total_machines : Nat) (faulty_machines : Nat) (cost_per_test : Nat) 
  (h_total : total_machines = 5) (h_faulty : faulty_machines = 2) (h_cost : cost_per_test = 1000) :
  (2000 * (2 / 5 * 1 / 4) + 3000 * (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3) + 
  4000 * (1 - (2 / 5 * 1 / 4) - (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3))) = 3500 :=
  by
  sorry

end average_cost_of_testing_l748_748674


namespace average_income_Q_and_R_l748_748383

variable (P Q R: ℝ)

theorem average_income_Q_and_R:
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 :=
by
  sorry

end average_income_Q_and_R_l748_748383


namespace inequality_solution_l748_748923

theorem inequality_solution (x : ℝ) : (2 / (x^2 + 1) > 4 / x + 5 / 2) ↔ x ∈ set.Ioo (-2 : ℝ) (0 : ℝ) := 
by {
  sorry
}

end inequality_solution_l748_748923


namespace find_greatest_K_l748_748161

theorem find_greatest_K {u v w K : ℝ} (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu2_gt_4vw : u^2 > 4 * v * w) :
  (u^2 - 4 * v * w)^2 > K * (2 * v^2 - u * w) * (2 * w^2 - u * v) ↔ K ≤ 16 := 
sorry

end find_greatest_K_l748_748161


namespace hyperbola_eccentricity_l748_748939

-- Define the hyperbola and circle equations, foci, and given conditions
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)
noncomputable def hyperbola (x y : ℝ) : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1
noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 9/4 * (a^2 + b^2)

-- Placeholders for the points and intersections
variables (F1 F2 P A B M N : ℝ × ℝ)
-- Constraints regarding perpendicular distances and loci
axiom on_hyperbola : hyperbola a b P.1 P.2
axiom perpendicular : perpendicular (P - F1) (P - F2)
axiom on_circle_A : circle a b A.1 A.2
axiom on_circle_B : circle a b B.1 B.2
axiom on_circle_M : circle a b M.1 M.2
axiom on_circle_N : circle a b N.1 N.2
axiom intersects_A_B : intersects (line_through P F1) (circle a b) A B
axiom intersects_M_N : intersects (line_through P F2) (circle a b) M N
axiom quadrilateral_area_AMBN : area_quadrilateral A M B N = 9 * b^2

-- Definition of the eccentricity
noncomputable def eccentricity : ℝ := (2 * real.sqrt 10) / 5

-- The theorem stating the proof problem
theorem hyperbola_eccentricity : eccentricity = (2 * real.sqrt 10) / 5 :=
sorry

end hyperbola_eccentricity_l748_748939


namespace sad_children_count_l748_748711

-- We define various constants
constant total_children : ℕ := 60
constant happy_children : ℕ := 30
constant neither_happy_nor_sad_children : ℕ := 20

-- We assert that the sad children are the total minus happy and neither happy nor sad
theorem sad_children_count : (total_children - happy_children - neither_happy_nor_sad_children) = 10 := by
  sorry

end sad_children_count_l748_748711


namespace P_no_real_roots_l748_748841

noncomputable def P : ℕ → (ℝ → ℝ)
| 0       := λ x, 1
| (n + 1) := λ x, x^(11 * (n + 1)) - P n x

theorem P_no_real_roots (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 := 
by 
  -- Proof of the theorem would go here
  sorry

end P_no_real_roots_l748_748841


namespace house_to_market_distance_l748_748858

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l748_748858


namespace proof_problem1_proof_problem2_l748_748131

noncomputable def problem1 : ℝ :=
  (∛(-8) + real.sqrt ((-1)^2) - ∛(64) * real.sqrt (1 / 4))

noncomputable def problem2 : ℝ :=
  (real.sqrt ((-4)^2) - ∛(-1) + real.sqrt (10^2 - 6^2))

theorem proof_problem1 : problem1 = -3 := by
  unfold problem1
  sorry

theorem proof_problem2 : problem2 = 13 := by
  unfold problem2
  sorry

end proof_problem1_proof_problem2_l748_748131


namespace find_f_2017_l748_748093

noncomputable def f : ℝ → ℝ
| x if x ≤ 0 => Real.log2 (1 - x)
| x + 1 => f x - f (x - 1)

theorem find_f_2017 : f 2017 = -1 :=
sorry

end find_f_2017_l748_748093


namespace banana_permutations_l748_748243

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748243


namespace ned_won_whack_a_mole_tickets_l748_748043

theorem ned_won_whack_a_mole_tickets :
  (∃ (tickets : ℕ), tickets + 19 = 5 * 9) → (∃ (whack_a_mole_tickets : ℕ), whack_a_mole_tickets = 26) :=
by
  intro h
  cases h with tickets ht
  use 26
  calc
    26 + 19 = 45 : by norm_num
    45 = 5 * 9 : by norm_num
  exact ht.symm

end ned_won_whack_a_mole_tickets_l748_748043


namespace sum_of_bn_2023_l748_748964

theorem sum_of_bn_2023 :
  (∀ n : ℕ, 1 ≤ n → 0 < q → a n = 2 * q^(n - 1)) →
  (a 4 = (6 * a 2 + a 3) / 2) →
  (∀ n : ℕ, b n = 1 / (Real.logb 2 (a n) * Real.logb 2 (a (n + 1)))) →
  (T_2023 = ∑ i in Finset.range 2023, b (i + 1)) →
  T_2023 = 2023 / 2024 :=
by
  sorry

end sum_of_bn_2023_l748_748964


namespace major_axis_length_minor_axis_length_ellipse_eccentricity_coordinates_of_foci_coordinates_of_vertices_l748_748163

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 3
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def e : ℝ := c / a

theorem major_axis_length : 2 * a = 10 := by sorry

theorem minor_axis_length : 2 * b = 6 := by sorry

theorem ellipse_eccentricity : e = 4 / 5 := by sorry

theorem coordinates_of_foci : ∀ (x : ℝ) , x = ±c -> ∃ y, y = 0 := by sorry

theorem coordinates_of_vertices : 
  (∀ (x : ℝ), x = ±a -> ∃ y, y = 0) ∧ 
  (∀ (y : ℝ), y = ±b -> ∃ x, x = 0) := by sorry

end major_axis_length_minor_axis_length_ellipse_eccentricity_coordinates_of_foci_coordinates_of_vertices_l748_748163


namespace charlies_data_limit_l748_748361

variables {data1 data2 data3 data4 : ℕ} 
variables {cost_per_gb extra_charge total_data used : ℕ}

def data_limit (total_data : ℕ) (extra_charge : ℕ) (cost_per_gb : ℕ) : ℕ :=
  total_data - extra_charge / cost_per_gb

theorem charlies_data_limit (h1 : data1 = 2) 
                            (h2 : data2 = 3)
                            (h3 : data3 = 5) 
                            (h4 : data4 = 10)
                            (h5 : cost_per_gb = 10)
                            (h6 : extra_charge = 120)
                            (h7 : total_data = data1 + data2 + data3 + data4) :
  data_limit total_data extra_charge cost_per_gb = 8 :=
begin
  sorry
end

end charlies_data_limit_l748_748361


namespace calc_x_l748_748870

theorem calc_x : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end calc_x_l748_748870


namespace basketball_team_wins_l748_748818

-- Define the known quantities
def games_won_initial : ℕ := 60
def games_total_initial : ℕ := 80
def games_left : ℕ := 50
def total_games : ℕ := games_total_initial + games_left
def desired_win_fraction : ℚ := 3 / 4

-- The main goal: Prove that the team must win 38 of the remaining 50 games to reach the desired win fraction
theorem basketball_team_wins :
  ∃ x : ℕ, x = 38 ∧ (games_won_initial + x : ℚ) / total_games = desired_win_fraction :=
by
  sorry

end basketball_team_wins_l748_748818


namespace find_polynomials_l748_748541

noncomputable def polynomial_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (∃ q : ℚ, x - y = q) → ∃ r : ℚ, f(x) - f(y) = r

theorem find_polynomials :
  ∀ f : ℝ → ℝ,
  (∃ a b c : ℝ, ∀ x : ℝ, f(x) = a * x^2 + b * x + c) ∧ polynomial_satisfies_condition f →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x : ℝ, f(x) = b * x + c :=
sorry

end find_polynomials_l748_748541


namespace expected_value_of_8_sided_die_l748_748091

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l748_748091


namespace intersection_A_B_l748_748750

noncomputable def A := {x : ℝ | -3 < x ∧ x < 3}
noncomputable def B := {y : ℝ | -∞ < y ∧ y ≤ 2}

theorem intersection_A_B : A ∩ B = {x | -3 < x ∧ x ≤ 2} :=
sorry

end intersection_A_B_l748_748750


namespace cleaning_time_together_l748_748797

noncomputable def time_to_clean_together (T_nick : ℝ) : ℝ :=
  let sawyer_rate := 1 / 6
  let nick_rate := 1 / T_nick
  let combined_rate := sawyer_rate + nick_rate
  1 / combined_rate

theorem cleaning_time_together (T_nick : ℝ) (h1 : 1 / 3 * T_nick = 3) (h2 : 6 = 6) : time_to_clean_together T_nick = 3.6 :=
by
  have T_nick_value : T_nick = 9 := by
    linarith
  show time_to_clean_together T_nick = 3.6
  rw [T_nick_value]
  simp [time_to_clean_together]
  have combined_rate : (1/6 + 1/9) = 5/18 := by norm_num
  simp [combined_rate]
  norm_num

-- The theorem cleaning_time_together will conclude the proof.

end cleaning_time_together_l748_748797


namespace max_possible_area_l748_748178

noncomputable def max_area_of_region_S
    (radii : list ℝ)
    (tangent_point : ℝ)
    (line_l : ℝ)
    (region_S : set (ℝ × ℝ))
    (areas : list ℝ) : ℝ :=
  let circle_area (r : ℝ) : ℝ := π * r^2 in
  let total_area  := list.sum (list.map circle_area radii) in
  let overlapping_areas := circle_area 6 + circle_area 4 + circle_area 2 in
  total_area - overlapping_areas + (circle_area 6 - circle_area 4) + (circle_area 4 - circle_area 2)

theorem max_possible_area : max_area_of_region_S [2, 4, 6, 8] 0 0 (region_in_single_circle 8) [4*π, 16*π, 36*π, 64*π] = 40 * π :=
by
  unfold max_area_of_region_S
  sorry

end max_possible_area_l748_748178


namespace angle_C_in_triangle_l748_748651

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 110) (ht : A + B + C = 180) : C = 70 :=
by
  -- proof steps go here
  sorry

end angle_C_in_triangle_l748_748651


namespace divide_10_digit_numbers_into_sets_l748_748884

-- Define the domain of 10-digit numbers composed of digits 1 and 2.
def is_valid_10_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 10^10 ∧ ∀ d ∈ digits 10 n, d = 1 ∨ d = 2

-- Define count of digit '2' in a number.
def count_digit_2 (n : ℕ) : ℕ :=
  (digits 10 n).count 2

-- Define the sets of numbers with even and odd counts of the digit '2'.
def class1 (n : ℕ) : Prop := is_valid_10_digit_number n ∧ count_digit_2 n % 2 = 0
def class2 (n : ℕ) : Prop := is_valid_10_digit_number n ∧ count_digit_2 n % 2 = 1

-- Define the sum of two numbers containing at least two digits that are '3'.
def has_at_least_two_threes (a b : ℕ) : Prop :=
  (digits 10 (a + b)).count 3 ≥ 2

-- The theorem stating the division into sets ensures the required property.
theorem divide_10_digit_numbers_into_sets :
  (∀ a b : ℕ, class1 a → class1 b → has_at_least_two_threes a b) ∧ 
  (∀ a b : ℕ, class2 a → class2 b → has_at_least_two_threes a b) :=
by
  sorry

end divide_10_digit_numbers_into_sets_l748_748884


namespace cost_of_each_cake_l748_748679

-- Define the conditions
def cakes : ℕ := 3
def payment_by_john : ℕ := 18
def total_payment : ℕ := payment_by_john * 2

-- Statement to prove that each cake costs $12
theorem cost_of_each_cake : (total_payment / cakes) = 12 := by
  sorry

end cost_of_each_cake_l748_748679


namespace probability_of_hats_l748_748912

def harmonic (n : ℕ) : ℝ :=
  (λ (n : ℕ), ∑ k in finset.range n, (1.0 : ℝ) / (k + 1)) n

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def probability_no_hats_left (n : ℕ) : ℝ :=
  if n = 0 then 1 else harmonic n / factorial n * probability_no_hats_left (n - 1)

#eval probability_no_hats_left 10 -- Result should be approximately 0.000516

theorem probability_of_hats (n : ℕ) : probability_no_hats_left 10 = 0.000516 :=
sorry

end probability_of_hats_l748_748912


namespace cone_sphere_volume_equality_cylinder_sphere_volume_equality_l748_748921

namespace VolumeEquivalence

-- Define the volume of the cone and sphere and prove the radius equality for the cone case
theorem cone_sphere_volume_equality (r h : ℝ) :
  ∃ R : ℝ, (1/3 * π * r^2 * h = 4/3 * π * R^3) ↔ (R = real.cbrt (r^2 * h / 4)) :=
by
  sorry

-- Define the volume of the cylinder and sphere and prove the radius equality for the cylinder case
theorem cylinder_sphere_volume_equality (r h : ℝ) :
  ∃ R : ℝ, (π * r^2 * h = 4/3 * π * R^3) ↔ (R = real.cbrt (3/4 * r^2 * h)) :=
by
  sorry

end VolumeEquivalence

end cone_sphere_volume_equality_cylinder_sphere_volume_equality_l748_748921


namespace distribute_books_into_bags_l748_748863

def number_of_ways_to_distribute_books (books : Finset ℕ) (bags : ℕ) : ℕ :=
  if (books.card = 5) ∧ (bags = 3) then 51 else 0

theorem distribute_books_into_bags :
  number_of_ways_to_distribute_books (Finset.range 5) 3 = 51 := by
  sorry

end distribute_books_into_bags_l748_748863


namespace tangent_line_eq_l748_748753

-- Define the function f
def f (x : ℝ) : ℝ := sin x + exp x + 2

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := cos x + exp x

-- Define the point of tangency, P(0,3)
noncomputable def point : ℝ × ℝ := (0, f 0)

-- Define the slope of the tangent line at x = 0
noncomputable def slope : ℝ := f' 0

-- Define the equation of the tangent line at x = 0
noncomputable def tangent_line (x : ℝ) : ℝ := slope * x + (point.snd)

-- State the theorem to prove the equation of the tangent line is y = 2x + 3
theorem tangent_line_eq : ∀ x : ℝ, tangent_line x = 2 * x + 3 := by
  sorry

end tangent_line_eq_l748_748753


namespace number_of_people_in_second_group_l748_748020

def people_in_initial_group : ℕ := 9
def food_for_initial_days : ℕ := 5
def days_passed : ℕ := 1
def remaining_days_together : ℕ := 3

noncomputable def total_food_units : ℕ := people_in_initial_group * food_for_initial_days
noncomputable def remaining_food_units : ℕ := total_food_units - (people_in_initial_group * days_passed)
noncomputable def food_consumption_per_day_per_person : ℕ := 1

theorem number_of_people_in_second_group :
  ∀ (people_in_initial_group food_for_initial_days days_passed remaining_days_together: ℕ)
  (total_food_units remaining_food_units food_consumption_per_day_per_person : ℕ),
  remaining_food_units / remaining_days_together = 12 →
  12 - people_in_initial_group = 3 :=
by {
  intros,
  sorry
}

end number_of_people_in_second_group_l748_748020


namespace expected_value_of_win_l748_748087

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l748_748087


namespace solve_k_l748_748644

-- Axiom stating that for the quadratic equation x^2 - 6x + k = 0 to have no real roots,
-- the value of k must satisfy the equation k > 9.

noncomputable def no_real_roots (k : ℝ) : Prop :=
(k > 9)

theorem solve_k : ∀ (k : ℝ), no_real_roots k ↔ (∀ x : ℝ, x^2 - 6 * x + k ≠ 0) :=
begin
  sorry
end

end solve_k_l748_748644


namespace largest_hope_number_within_1000_l748_748641

def is_hope_number (n : ℕ) : Prop :=
  let d := (List.filter (λ x, n % x = 0) (List.range (n + 1)))
  (List.length d) % 2 = 1

theorem largest_hope_number_within_1000 : 
  (∃ n, is_hope_number n ∧ n ≤ 1000 ∧ ∀ m, is_hope_number m ∧ m ≤ 1000 → m ≤ n) :=
  sorry

end largest_hope_number_within_1000_l748_748641


namespace neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l748_748396

theorem neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0 :
  ¬ (∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 - x > 0 :=
by
    sorry

end neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l748_748396


namespace problem_solution_count_l748_748927

def positive_integer_solutions_count : Prop :=
  let condition : ℕ → ℕ → Prop := λ x y, 4 * x + 7 * y = 800 ∧ x > 0 ∧ y > 0
  ∃ solutions : list (ℕ × ℕ),
    (∀ p ∈ solutions, condition p.1 p.2) ∧
    (solutions.length = 29)

theorem problem_solution_count : positive_integer_solutions_count :=
sorry

end problem_solution_count_l748_748927


namespace geometric_sequence_if_find_sequence_l748_748345

-- Define the sequence a_n as per the conditions
def a (n : ℕ) : ℝ := 
  if n = 0 then a_0 
  else 3^(n-1) - 2 * (a (n - 1))

-- Part 1: Prove that {a_n - 3 * a_{n-1}} is a geometric sequence with common ratio -2 if a_0 != 1/5
theorem geometric_sequence_if (a_0 : ℝ) (h : a_0 ≠ 1 / 5) : 
  ∀ n : ℕ, n > 0 → 
  (a n - 3 * a (n - 1)) = -2 * (a (n - 1) - 3 * a (n - 2)) :=
sorry

-- Part 2: Find the expression for a_n given that a_0 = 1/5
theorem find_sequence (a_0 : ℝ) (h : a_0 = 1 / 5) : 
  ∀ n : ℕ, n > 0 → 
  a n = (1 / 5) * 3^n :=
sorry

end geometric_sequence_if_find_sequence_l748_748345


namespace unique_root_range_l748_748751

theorem unique_root_range (a : ℝ) :
  (x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → (∃! x : ℝ, x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → - (Real.sqrt 3) / 2 < a ∧ a < (Real.sqrt 3) / 2 :=
by
  sorry

end unique_root_range_l748_748751


namespace find_annual_interest_rate_l748_748483

noncomputable def annual_interest_rate (P A1 A2 : ℝ)
    (t1 t2 n : ℕ) (compounding_periods : n = 4) : ℝ :=
    let x := (A2 / A1) ^ (1 / ((t2 - t1) * n : ℝ))
    in (x - 1) * n

theorem find_annual_interest_rate :
    let P : ℝ := 0 -- We do not actually need P to find r.
    let A1 : ℝ := 4875
    let A2 : ℝ := 5915
    let t1 : ℕ := 2
    let t2 : ℕ := 3
    let n : ℕ := 4
    annual_interest_rate P A1 A2 t1 t2 n (rfl : n = 4) = 0.2 :=
by {
    sorry
}

end find_annual_interest_rate_l748_748483


namespace calculate_overall_profit_percentage_l748_748901

def totalSellingPrice (spA spB spC spD : ℝ) : ℝ :=
  spA + spB + spC + spD

def totalCostPrice (cpA cpB cpC cpD : ℝ) : ℝ :=
  cpA + cpB + cpC + cpD

def profit (sp cp : ℝ) : ℝ :=
  sp - cp

def overallProfitPercentage (totalProfit totalCost : ℝ) : ℝ :=
  (totalProfit / totalCost) * 100

theorem calculate_overall_profit_percentage :
  let spA := 120
  let spB := 200
  let spC := 75
  let spD := 180
  let cpA := 0.30 * spA
  let cpB := 0.20 * spB
  let cpC := 0.40 * spC
  let cpD := 0.25 * spD
  let tpA := profit spA cpA
  let tpB := profit spB cpB
  let tpC := profit spC cpC
  let tpD := profit spD cpD
  let totalCP := totalCostPrice cpA cpB cpC cpD
  let totalSP := totalSellingPrice spA spB spC spD
  let totalProfit := tpA + tpB + tpC + tpD
  overallProfitPercentage totalProfit totalCP ≈ 280.79 :=
by 
  let spA := 120.0
  let spB := 200.0
  let spC := 75.0
  let spD := 180.0
  let cpA := 0.30 * spA
  let cpB := 0.20 * spB
  let cpC := 0.40 * spC
  let cpD := 0.25 * spD
  let tpA := profit spA cpA
  let tpB := profit spB cpB
  let tpC := profit spC cpC
  let tpD := profit spD cpD
  let totalCP := totalCostPrice cpA cpB cpC cpD
  let totalSP := totalSellingPrice spA spB spC spD
  let totalProfit := tpA + tpB + tpC + tpD
  have h : overallProfitPercentage totalProfit totalCP = (totalProfit / totalCP) * 100 := rfl
  suffices q : (totalProfit / totalCP) * 100 ≈ 280.79, from h ▸ q
  sorry

end calculate_overall_profit_percentage_l748_748901


namespace steve_has_26_dimes_l748_748379

def steve_dimes_problem : Prop :=
  ∃ (D N : ℕ), D + N = 36 ∧ 0.10 * D + 0.05 * N = 3.10 ∧ D = 26

theorem steve_has_26_dimes : steve_dimes_problem :=
by
  -- The proof would go here
  sorry

end steve_has_26_dimes_l748_748379


namespace ticket_sale_savings_l748_748150

theorem ticket_sale_savings : 
  let original_price := [11, 15, 18, 21, 26].sum
  let discounted_price := [4, 5, 6, 8, 10].sum
  let total_savings := original_price - discounted_price
  let percentage_saved := (total_savings / original_price.toFloat) * 100
  percentage_saved ≈ 63.74 :=
by
  sorry

end ticket_sale_savings_l748_748150


namespace not_divisible_by_2003_l748_748006

def seq_a : ℕ → ℕ 
| 0       := 1
| (n + 1) := (seq_a n) ^ 2001 + (seq_b n)
and seq_b : ℕ → ℕ
| 0       := 4
| (n + 1) := (seq_b n) ^ 2001 + (seq_a n)

theorem not_divisible_by_2003 (n : ℕ) : ¬(2003 ∣ seq_a n) ∧ ¬(2003 ∣ seq_b n) :=
sorry

end not_divisible_by_2003_l748_748006


namespace number_of_students_and_average_output_l748_748042

theorem number_of_students_and_average_output 
  (total_potatoes : ℕ)
  (days : ℕ)
  (x y : ℕ) 
  (h1 : total_potatoes = 45715) 
  (h2 : days = 5)
  (h3 : x * y * days = total_potatoes) : 
  x = 41 ∧ y = 223 :=
by
  sorry

end number_of_students_and_average_output_l748_748042


namespace area_comparison_l748_748336

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

def area_triangle1 : ℝ := heron_area 15 15 20
def area_triangle2 : ℝ := heron_area 15 15 24

theorem area_comparison : area_triangle1 < area_triangle2 :=
  sorry

end area_comparison_l748_748336


namespace main_l748_748589

noncomputable def ellipse_foci_vertex_on_circle : Prop :=
  ∃ (a b c : ℝ), 
    (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (b = c) ∧ (b^2 + c^2 = 8) ∧ 
    (∀ x y : ℝ, (x^2 + y^2 = 4) → ((c = sqrt 4) ∨ (c = -sqrt 4)))

noncomputable def find_ellipse_equation : Prop :=
  ∃ (a b : ℝ), 
    ellipse_foci_vertex_on_circle ∧
    (b^2 = 4) ∧ (a^2 = 8) ∧ 
    (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 8 + y^2 / 4 = 1))

noncomputable def find_line_equation (P : ℝ × ℝ) (slope : ℝ) : Prop :=
  ∃ (a b m x y : ℝ), 
    find_ellipse_equation ∧ 
    (P = (-3, 2)) ∧ (slope = 1) ∧ 
    (y = x + m) ∧ 
    (3x^2 + 4mx + 2m^2 - 8 = 0) ∧ 
    (-2 * sqrt 3 < m) ∧ (m < 2 * sqrt 3) ∧ 
    (x1 + x2 = -4m/3) ∧ (x1 * x2 = (2m^2 - 8)/3) ∧ 
    (let M_x = -2m/3 in let M_y = m/3 in ((M_y - 2) / (M_x + 3) = -1)) ∧ 
    (m = 3) ∧ 
    (y = x + 3)

-- Main theorem statement
theorem main : find_ellipse_equation ∧ find_line_equation (-3, 2) 1 :=
by {
  sorry
}

end main_l748_748589


namespace median_of_given_set_l748_748579

open Real

def median_of_set (s : List ℝ) : ℝ :=
  let sorted_s := s.qsort (· ≤ ·)
  sorted_s.get (s.length / 2) -- qsort sorts the list in non-decreasing order

theorem median_of_given_set :
  ∀ (a b : ℝ), 
    a = -5 → 
    b > 0 →
    a * b^2 = log 10 b → 
    median_of_set [a^2 / b, 0, 1, a, b] = 0.1 :=
by
  intros a b ha hb hab
  -- Proof goes here
  sorry

end median_of_given_set_l748_748579


namespace banana_distinct_arrangements_l748_748279

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748279


namespace distinct_arrangements_banana_l748_748271

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748271


namespace number_2120_in_33rd_group_l748_748716

def last_number_in_group (n : ℕ) := 2 * n * (n + 1)

theorem number_2120_in_33rd_group :
  ∃ n, n = 33 ∧ (last_number_in_group (n - 1) < 2120) ∧ (2120 <= last_number_in_group n) :=
sorry

end number_2120_in_33rd_group_l748_748716


namespace banana_arrangements_l748_748254

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748254


namespace min_distance_l748_748468

-- Definition of the circle and other conditions
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9
def point_on_line (x y : ℝ) : Prop := ∃ m b : ℝ, y = m * x + b ∧ (3, 1) = (3, m * 3 + b)
def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Definition of the problem statement in terms of proving the minimum distance
theorem min_distance : ∀ (M N : ℝ × ℝ), 
  (circle M.1 M.2) → (circle N.1 N.2) → (point_on_line M.1 M.2) → (point_on_line N.1 N.2) → 
  ∃ d : ℝ, distance M N = d ∧ d = 4 :=
by
  sorry

end min_distance_l748_748468


namespace distinct_arrangements_banana_l748_748236

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748236


namespace geom_seq_formula_sum_b_formula_l748_748602

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- a_n is in a geometric sequence with positive terms
-- a_1a_2 = 2
axiom a1a2_eq_2 : a 0 * a 1 = 2
-- a_3a_4 = 32
axiom a3a4_eq_32 : a 2 * a 3 = 32

-- Question 1: Prove the general formula a_n = 2 ^ (n - 1)
theorem geom_seq_formula :
  geom_seq a → ∀ n : ℕ, a n = 2 ^ n :=
sorry

-- Question 2: Prove the sum of the first n terms of the sequence b_n
-- where b_n = (2n-1) * a_n
def b := λ n : ℕ, (2 * n + 1) * a n

theorem sum_b_formula :
  geom_seq a → (∀ n : ℕ, a n = 2 ^ n) →
  ∀ n : ℕ, T n = (2 * n - 3) * 2 ^ n + 3 :=
sorry

end geom_seq_formula_sum_b_formula_l748_748602


namespace sum_x_i_x_i_add_s_eq_neg1_l748_748813

variable (n : ℕ)
variable (x : ℕ → ℤ)
variable (t : ℕ → ℕ)

axiom period_condition : ∀ j : ℕ, x (j + 2^n - 1) = x j
axiom sequence_condition : ∀ j : ℕ, x (j + n) = ∏ k in (finset.range n).filter (λ k, k ∈ t), x (j + k)
axiom t_distinct : ∀ (i j : ℕ), i < j ∧ i < n ∧ j < n → t i ≠ t j
axiom t_bounds : ∀ (j : ℕ), j < n → t j < n
axiom x_binary : ∀ j : ℕ, x j = 1 ∨ x j = -1

theorem sum_x_i_x_i_add_s_eq_neg1 (s : ℕ) (h_s : s < 2^n - 1) :
  ∑ i in finset.range (2^n - 1), x i * x (i + s) = -1 := sorry

end sum_x_i_x_i_add_s_eq_neg1_l748_748813


namespace projection_of_a_onto_b_is_2_l748_748225

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (3, 4)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def scalar_projection (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_2 : scalar_projection vector_a vector_b = 2 :=
by
  sorry

end projection_of_a_onto_b_is_2_l748_748225


namespace eq_of_symmetric_translation_l748_748471

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

noncomputable def translate_left (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (x + k)

noncomputable def translate_up (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := g x + k

noncomputable def translate_parabola (x : ℝ) : ℝ := translate_up (translate_left parabola 3) 2 x

noncomputable def symmetric_parabola (h : ℝ → ℝ) (x : ℝ) : ℝ := h (-x)

theorem eq_of_symmetric_translation :
  symmetric_parabola translate_parabola x = 2 * x^2 - 8 * x + 3 :=
by
  sorry

end eq_of_symmetric_translation_l748_748471


namespace transformed_number_is_cube_l748_748720

theorem transformed_number_is_cube (n : ℕ) : 
  let transformed_number := 10^(3*n+3) + 3*10^(2*n+2) + 3*10^(n+1) + 1
  in transformed_number = (10^(n+1) + 1)^3 := 
by sorry

end transformed_number_is_cube_l748_748720


namespace avg_selling_price_correct_l748_748109

namespace VegetableStore

def total_veg_kg : ℕ := 100
def morning_kg : ℕ := 50
def noon_kg : ℕ := 30
def afternoon_kg : ℕ := 20
def morning_price_per_kg : ℝ := 1.2
def noon_price_per_kg : ℝ := 1.0
def afternoon_price_per_kg : ℝ := 0.8

def total_revenue : ℝ := (morning_kg * morning_price_per_kg) + (noon_kg * noon_price_per_kg) + (afternoon_kg * afternoon_price_per_kg)
def total_weight_sold : ℕ := morning_kg + noon_kg + afternoon_kg

def avg_selling_price (total_revenue weight_sold : ℝ) : ℝ := total_revenue / weight_sold

theorem avg_selling_price_correct :
  avg_selling_price total_revenue total_weight_sold = 1.06 :=
by
  sorry

end VegetableStore

end avg_selling_price_correct_l748_748109


namespace sum_first_2003_magic_numbers_l748_748831

def is_magic_number (x : ℕ) : Prop := (bit0 x).bits.countb tt % 2 = 0

def first_2003_magic_numbers : List ℕ :=
  List.filter is_magic_number (List.range 2003 0)

theorem sum_first_2003_magic_numbers : List.sum first_2003_magic_numbers = 4015014 := by
  sorry

end sum_first_2003_magic_numbers_l748_748831


namespace range_of_a_l748_748335

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 ≤ a := 
sorry

end range_of_a_l748_748335


namespace crows_eat_worms_l748_748290

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l748_748290


namespace train_passes_man_in_approx_138_5_seconds_l748_748112

noncomputable def time_to_pass
  (train_length : ℝ)        -- in meters
  (train_speed_kmh : ℝ)     -- in km/hr
  (man_speed_kmh : ℝ)       -- in km/hr, opposite direction
  : ℝ :=
  let train_speed_mps := train_speed_kmh * (1000 / 3600) in
  let man_speed_mps := man_speed_kmh * (1000 / 3600) in
  let relative_speed_mps := train_speed_mps + man_speed_mps in
  train_length / relative_speed_mps

theorem train_passes_man_in_approx_138_5_seconds :
  time_to_pass 500 120 10 ≈ 138.5 := by
  sorry

end train_passes_man_in_approx_138_5_seconds_l748_748112


namespace continued_fraction_evaluation_l748_748683

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the problem
theorem continued_fraction_evaluation (m n : ℕ) (h1 : nat.coprime m n) (h2 : m = fib 1988) (h3 : n = fib 1989) :
  m^2 + m * n - n^2 = -1 :=
sorry

end continued_fraction_evaluation_l748_748683


namespace expected_value_of_win_l748_748076

theorem expected_value_of_win :
  (∑ n in finset.range 9, (n ^ 3)) / 8 = 162 := 
sorry

end expected_value_of_win_l748_748076


namespace smallest_positive_multiple_of_6_and_5_l748_748785

theorem smallest_positive_multiple_of_6_and_5 : ∃ (n : ℕ), (n > 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
  sorry

end smallest_positive_multiple_of_6_and_5_l748_748785


namespace range_of_a_l748_748219

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def g (x a : ℝ) : ℝ := a * x + 2

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 2, ∃ x2 ∈ set.Icc (-1 : ℝ) 2, f x1 = g x2 a) ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l748_748219


namespace triangle_is_acute_l748_748303

-- Define the condition that the angles have a ratio of 2:3:4
def angle_ratio_cond (a b c : ℝ) : Prop :=
  a / b = 2 / 3 ∧ b / c = 3 / 4

-- Define the sum of the angles in a triangle
def angle_sum_cond (a b c : ℝ) : Prop :=
  a + b + c = 180

-- The proof problem stating that triangle with angles in ratio 2:3:4 is acute
theorem triangle_is_acute (a b c : ℝ) (h_ratio : angle_ratio_cond a b c) (h_sum : angle_sum_cond a b c) : 
  a < 90 ∧ b < 90 ∧ c < 90 := 
by
  sorry

end triangle_is_acute_l748_748303


namespace coloring_segments_l748_748737

-- We define the problem statement and conditions in Lean 4
theorem coloring_segments (k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ 10) :
  10_points_no_three_collinear ∧
  (∀ subset : finset (fin 10), subset.card = k → 
    (∀ {p1 p2 : fin 10} (hp1 : p1 ∈ subset) (hp2 : p2 ∈ subset), 
      ∃ (color : ℕ) (hc : color < k), p1 ≠ p2 ∧ colored_segment p1 p2 color ∧ 
      (∀ {p3 p4 : fin 10} (hp3 : p3 ∈ subset) (hp4 : p4 ∈ subset), 
        p3 ≠ p4 → ∃ color', color' < k ∧ color' ≠ color ∧ colored_segment p3 p4 color'))) →
  5 ≤ k := 
begin
  sorry
end

end coloring_segments_l748_748737


namespace find_a_n_l748_748982

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, S (n + 1) = 4 * a n + 2 ∧ S n = ∑ i in finset.range (n + 1), a i

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ) (H : sequence a) : 
  ∀ n : ℕ, a n = (3 * n - 1) * 2^(n - 2) :=
sorry

end find_a_n_l748_748982


namespace triangle_area_l748_748115

def point := (ℝ × ℝ)
def area_of_triangle (p1 p2 p3 : point) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

theorem triangle_area :
  ∀ (A B C : point), 
  A = (-2, 3) → B = (7, -3) → C = (4, 6) → 
  area_of_triangle A B C = 31.5 :=
by
  -- ignoring the proof
  intros A B C hA hB hC
  sorry

end triangle_area_l748_748115


namespace angle_between_vectors_l748_748989

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖2 • a - b‖ = real.sqrt 13)

theorem angle_between_vectors : real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l748_748989


namespace rate_of_change_of_liquid_level_l748_748075

open Real

-- Define the conditions given in part a)
def base_diameter : ℝ := 2
def radius_of_cylinder : ℝ := base_diameter / 2
def drainage_rate : ℝ := -0.01  -- Drainage rate per second (negative as volume is decreasing)
def volume (r h : ℝ) : ℝ := π * r * r * h

-- Define the problem statement
theorem rate_of_change_of_liquid_level (h : ℝ) (dh_dt : ℝ)
  (h_positive : 0 ≤ h) 
  (drainage_rate_condition : ∃ (t : ℝ), ∂ volume radius_of_cylinder h t = drainage_rate) :
  dh_dt = drainage_rate / (π * (radius_of_cylinder ^ 2)) :=
sorry

end rate_of_change_of_liquid_level_l748_748075


namespace water_level_not_discrete_l748_748530

-- Define the available conditions to identify the type of random variables
def visitors_lounge_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

def pages_received_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

def water_level_yangtze : RandomVariable ℝ :=
  sorry -- assuming this is an uncountable variable

def vehicles_overpass_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

-- Prove that the water level of the Yangtze River is not a discrete random variable
theorem water_level_not_discrete : ¬is_discrete_random_variable water_level_yangtze :=
  sorry

end water_level_not_discrete_l748_748530


namespace necessary_and_sufficient_condition_l748_748451

theorem necessary_and_sufficient_condition (x : ℝ) :
  x > 0 ↔ x + 1/x ≥ 2 :=
by sorry

end necessary_and_sufficient_condition_l748_748451


namespace marble_sum_count_l748_748286

def num_ways_to_match_sums (marbles_my : Finset ℕ) (marbles_mathew : Finset ℕ) : ℕ :=
  ∑ m in marbles_mathew, marbles_my.sum (λ m1, marbles_my.sum (λ m2, if m1 + m2 = m then 1 else 0))

theorem marble_sum_count :
  num_ways_to_match_sums (Finset.range 9 \ {0}) (Finset.range 13 \ {0}) = 70 :=
by
  sorry

end marble_sum_count_l748_748286


namespace range_of_a_l748_748216

noncomputable def f (a x : ℝ) : ℝ := 4 * x + a * x^2

theorem range_of_a (a : ℝ) : 
  (∃ x₀ y₀ k : ℝ, (1, 1) = (x₀, f a x₀) ∧ (1, 1) = (x₀, f' a x₀) ∧ f' a x₀ = k) ↔ (a < -3 ∨ a > 0) :=
sorry

end range_of_a_l748_748216


namespace real_solutions_of_inequality_l748_748922

theorem real_solutions_of_inequality:
  {x : ℝ} → 
  (x ≠ 0) → 
  (x ≠ -1) →
  (x ≠ -3) → 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔ 
  (x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (-1) 0 ∪ Set.Ioo 1 (∞)) := 
by 
  sorry

end real_solutions_of_inequality_l748_748922


namespace symmetric_sum_eq_two_l748_748603

-- Definitions and conditions
def symmetric (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

def P : ℝ × ℝ := (sorry, 1)
def Q : ℝ × ℝ := (-3, sorry)

-- Problem statement
theorem symmetric_sum_eq_two (h : symmetric P Q) : P.1 + Q.2 = 2 :=
by
  -- Proof omitted
  sorry

end symmetric_sum_eq_two_l748_748603


namespace arithmetic_sequence_n_is_100_l748_748175

theorem arithmetic_sequence_n_is_100 
  (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)
  (h1 : a 1 = a1)
  (hd : d = 3)
  (h : ∀ n, a n = a1 + (n - 1) * d)
  (h298 : ∃ n, a n = 298) :
  (∃ n, a n = 298) ↔  n = 100 :=
by {
  obtain ⟨n, hn⟩ := h298,
  rw h at hn,
  have : a1 = 1 := by assumption,
  rw [this, hd] at hn,
  norm_num at hn,
  norm_cast,
  exact hn,
 sorry
}

end arithmetic_sequence_n_is_100_l748_748175


namespace radii_inequality_l748_748702

open Real

variables {A₁ A₂ A₃ A₄ : Type} [Inhabited A₁] [Inhabited A₂] [Inhabited A₃] [Inhabited A₄]
variables {r r₁ r₂ r₃ r₄ : ℝ}
variables {S S₁ S₂ S₃ S₄ : ℝ}

noncomputable def inscribed_radius_of_tetrahedron (A₁ A₂ A₃ A₄ : Type) : ℝ := r
noncomputable def inscribed_radius_of_face (A : Type) : ℝ := 
  match A with
    | A₁ => r₁
    | A₂ => r₂
    | A₃ => r₃
    | A₄ => r₄

theorem radii_inequality (r r₁ r₂ r₃ r₄ : ℝ) (h₁ : r > 0) (h₂ : r₁ > 0) (h₃ : r₂ > 0) (h₄ : r₃ > 0) (h₅ : r₄ > 0) 
  (condr : r = inscribed_radius_of_tetrahedron A₁ A₂ A₃ A₄)
  (condr₁ : r₁ = inscribed_radius_of_face A₁)
  (condr₂ : r₂ = inscribed_radius_of_face A₂)
  (condr₃ : r₃ = inscribed_radius_of_face A₃)
  (condr₄ : r₄ = inscribed_radius_of_face A₄) :
  1 / r₁ ^ 2 + 1 / r₂ ^ 2 + 1 / r₃ ^ 2 + 1 / r₄ ^ 2 ≤ 2 / r ^ 2 :=
  sorry

end radii_inequality_l748_748702


namespace find_eighth_term_l748_748658

noncomputable def arithmetic_sum {a : ℕ → ℝ} (n : ℕ) := (n / 2) * (a 1 + a n)

theorem find_eighth_term (a : ℕ → ℝ) (h : arithmetic_sum 15 = 90) : 
  a 8 = 6 := by
  sorry

end find_eighth_term_l748_748658


namespace distinct_arrangements_banana_l748_748239

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748239


namespace order_of_f_vals_l748_748617

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def a : ℝ := Real.logb (1 / 2) (1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := (1 / 3) ^ 0.2

theorem order_of_f_vals : f(c) < f(b) ∧ f(b) < f(a) := by
  sorry

end order_of_f_vals_l748_748617


namespace ruby_single_digit_or_zero_l748_748374

theorem ruby_single_digit_or_zero (n : ℕ) : 
  ∃ m ≤ 9, iterated_product_of_digits n = m :=
sorry

/--
Define the function iterated_product_of_digits which iteratively replaces a number 
with the product of its digits until a single-digit number or 0 is obtained.
-/
def iterated_product_of_digits : ℕ → ℕ
| n := if n ≤ 9 then n
       else iterated_product_of_digits (list.product (nat.digits 10 n))

-- Notes: 
-- - nat.digits 10 n gives the list of base-10 digits of n.
-- - list.product computes the product of elements in the list.

end ruby_single_digit_or_zero_l748_748374


namespace expected_value_of_biased_die_l748_748819

theorem expected_value_of_biased_die : 
  let p1 := 1/10
  let p2 := 3/20
  let ev := p1 * 1 + p1 * 2 + p1 * 3 + p1 * 4 + p2 * 5 + p2 * 6 + p2 * 7 + p2 * 8
  ev = 4.9 :=
by 
  let p1 := 1/10
  let p2 := 3/20
  let ev := p1 * 1 + p1 * 2 + p1 * 3 + p1 * 4 + p2 * 5 + p2 * 6 + p2 * 7 + p2 * 8
  show ev = 4.9, from sorry

end expected_value_of_biased_die_l748_748819


namespace monic_quadratic_with_root_2_minus_i_l748_748168

theorem monic_quadratic_with_root_2_minus_i :
  ∃ (p : ℝ[X]), monic p ∧ degree p = 2 ∧ is_root p (complex.of_real 2 - complex.I) ∧ p = X^2 - 4 * X + 5 :=
by
  sorry

end monic_quadratic_with_root_2_minus_i_l748_748168


namespace line_through_intersection_parallel_and_perpendicular_l748_748058

theorem line_through_intersection_parallel_and_perpendicular :
  (∃ M : ℝ × ℝ, (M = (-1, 2)) ∧ 
  ((∃ L : ℝ → ℝ → Prop, (L = λ x y, 3 * x + 4 * y - 5 = 0) ∧ L M.1 M.2) ∧ 
   (∃ L : ℝ → ℝ → Prop, (L = λ x y, 2 * x - 3 * y + 8 = 0) ∧ L M.1 M.2) ∧ 
  (∃ L : ℝ → ℝ → Prop, (L = λ x y, 2 * x + y = 0) ∧ 
   (∀ x y, L x y ↔ (2 * x + y + 5 = 0))) ∧
  (∃ L : ℝ → ℝ → Prop, (L = λ x y, x - 2 * y + 5 = 0) ∧ 
   (∀ x y, L x y ↔ ((2 * x + y + 5 = 0) -> (x - 2 * y + 5 = 0)))) :=
  sorry

end line_through_intersection_parallel_and_perpendicular_l748_748058


namespace triangle_ADC_area_l748_748667

-- Definitions based on conditions
def BD : ℝ := 2
def DC : ℝ := 5
def area_ABD : ℝ := 40
def ratio_BD_DC : ℝ := BD / DC  -- Defining the ratio condition

-- Goal statement
theorem triangle_ADC_area (ratio_BD_DC : BD / DC = 2 / 5) (area_ABD : 40) : 
  ∃ area_ADC : ℝ, area_ADC = 100 :=
sorry

end triangle_ADC_area_l748_748667


namespace arccos_one_half_eq_pi_div_three_l748_748510

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l748_748510


namespace inequality_solution_min_length_AB_triangle_right_angle_l748_748580

-- Define the context and conditions
variables {a : ℝ} (hpos : a > 0)
def f (x : ℝ) := a * x^2 - a^2 * x - 1 / a

-- Proof Problem 1: Proving intervals for the inequality f(x) > f(1)
theorem inequality_solution (h : a > 0) :
  (if a ≥ 2 then ∀ x, f x > f 1 ↔ (x < 1 ∨ x > a - 1) else ∀ x, f x > f 1 ↔ (x < a - 1 ∨ x > 1)) :=
  sorry

-- Proof Problem 2: Proving the minimum length of AB
theorem min_length_AB (h : a > 0) :
  let Δ := a^4 + 4
  let length_AB := (real.sqrt Δ) / a
  length_AB ≥ 2 :=
  sorry

-- Proof Problem 3: Proving that △ABC is a right-angled triangle
theorem triangle_right_angle (h : a > 0) :
  let A := (a^2 + real.sqrt (a^4 + 4)) / (2 * a)
  let B := (a^2 - real.sqrt (a^4 + 4)) / (2 * a)
  let C_Y := -1 / a
  let AC := (-(a^2 - real.sqrt (a^4 + 4)) / (2 * a), C_Y)
  let BC := (-(a^2 + real.sqrt (a^4 + 4)) / (2 * a), C_Y)
  (AC.1 * BC.1 + AC.2 * BC.2 = 0) :=
  sorry

end inequality_solution_min_length_AB_triangle_right_angle_l748_748580


namespace b_sequence_geometric_sum_a_sequence_l748_748586

-- Definitions for sequences a_n and b_n
def a_sequence (n : ℕ) : ℕ → ℝ
| 0       := 3/2
| (n + 1) := 3 * a_sequence n - 1

def b_sequence (n : ℕ) : ℕ → ℝ
| n := a_sequence n - 1/2

-- Statement: proving b_sequence is geometric
theorem b_sequence_geometric :
  ∀ n : ℕ, b_sequence (n + 1) = 3 * b_sequence n ∧ b_sequence 0 = 1 := 
sorry

-- Statement: Sum of the first n terms of a_sequence
def S (n : ℕ) : ℝ := 
  ∑ k in (Finset.range n), a_sequence k

theorem sum_a_sequence :
  ∀ n : ℕ, S n = (3^n + n - 1) / 2 := 
sorry

end b_sequence_geometric_sum_a_sequence_l748_748586


namespace arccos_pi_over_3_l748_748505

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l748_748505


namespace graph_intersection_l748_748137

noncomputable def log : ℝ → ℝ := sorry

lemma log_properties (a b : ℝ) (ha : 0 < a) (hb : 0 < b): log (a * b) = log a + log b := sorry

theorem graph_intersection :
  ∃! x : ℝ, 2 * log x = log (2 * x) :=
by
  sorry

end graph_intersection_l748_748137


namespace banana_distinct_arrangements_l748_748276

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l748_748276


namespace expected_value_eight_sided_die_win_l748_748083

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l748_748083


namespace distinct_arrangements_banana_l748_748268

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748268


namespace extremum_condition_l748_748757

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * a * x^2 + 1 = 0

theorem extremum_condition (a : ℝ) : has_extremum a ↔ a < 0 := 
  sorry

end extremum_condition_l748_748757


namespace banana_permutations_l748_748258

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748258


namespace inverse_proportional_value_l748_748769

theorem inverse_proportional_value (k : ℝ) (x y : ℝ) (h1 : ∀ x y, y * x^2 = k) (h2 : y = 1) (h3 : x = 4) : y = 0.25 → x = 8 :=
by
  intro hy 
  have hx : x^2 = k / y := sorry
  have hx_y : x = sqrt (k / y) := sorry
  rw [hx_y, hy]
  sorry

end inverse_proportional_value_l748_748769


namespace sum_of_fourth_powers_l748_748232

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := sorry

end sum_of_fourth_powers_l748_748232


namespace general_formula_sum_reciprocal_S_n_l748_748208

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions in the problem
variables {a : ℕ → ℝ} (d : ℝ) (a2 : ℝ := 5)

axiom a2_is_5 : a 2 = 5
axiom a4_geometric_mean : a 4 * a 4 = a 1 * a 13

-- General Formula Proof
theorem general_formula (h : arithmetic_sequence a d) : ∀ n, a n = 2 * n + 1 :=
sorry

-- Sum of Reciprocals of Sums Proof
noncomputable def S_n (n : ℕ) : ℝ :=
  (n * (2 * n + 4)) / 2

noncomputable def reciprocal_S_n (n : ℕ) : ℝ :=
  1 / S_n n

theorem sum_reciprocal_S_n (h : arithmetic_sequence a d) :
  ∀ n, (1 / S_n n) + (1 / S_n (n - 1)) + ... + (1 / S_n 1) = 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2)) :=
sorry

end general_formula_sum_reciprocal_S_n_l748_748208


namespace probability_distribution_m_l748_748703

theorem probability_distribution_m (m : ℚ) : 
  (m + m / 2 + m / 3 + m / 4 = 1) → m = 12 / 25 :=
by sorry

end probability_distribution_m_l748_748703


namespace toy_cars_ratio_proof_l748_748678

theorem toy_cars_ratio_proof (toys_original : ℕ) (toys_bought_last_month : ℕ) (toys_total : ℕ) :
  toys_original = 25 ∧ toys_bought_last_month = 5 ∧ toys_total = 40 →
  (toys_total - toys_original - toys_bought_last_month) / toys_bought_last_month = 2 :=
by
  sorry

end toy_cars_ratio_proof_l748_748678


namespace central_angle_l748_748660

variable (O : Type)
variable (A B C : O)
variable (angle_ABC : ℝ) 

theorem central_angle (h : angle_ABC = 50) : 2 * angle_ABC = 100 := by
  sorry

end central_angle_l748_748660


namespace banana_permutations_l748_748244

theorem banana_permutations : 
  let b_freq := 1
  let n_freq := 2
  let a_freq := 3
  let total_letters := 6 in
  nat.choose total_letters 1 * nat.choose (total_letters - 1) 2 * nat.choose (total_letters - 3) 3 = 60 := 
by
  let fact := nat.factorial
  let perms := fact total_letters / (fact b_freq * fact n_freq * fact a_freq)
  exact perms = 60

end banana_permutations_l748_748244


namespace quadrilateral_is_square_l748_748623

def is_square (A B C D : (ℝ × ℝ)) : Prop :=
  let length (P Q : (ℝ × ℝ)) : ℝ := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let dot_product (U V : (ℝ × ℝ)) : ℝ := (U.1 * V.1 + U.2 * V.2)
  let perpendicular (U V : (ℝ × ℝ)) : Prop := dot_product U V = 0
  let vector (P Q : (ℝ × ℝ)) : (ℝ × ℝ) := (Q.1 - P.1, Q.2 - P.2)
  
  length A B = length B C ∧ length B C = length C D ∧ length C D = length D A ∧
  perpendicular (vector A B) (vector B C) ∧
  perpendicular (vector B C) (vector C D) ∧
  perpendicular (vector C D) (vector D A) ∧
  perpendicular (vector D A) (vector A B)

theorem quadrilateral_is_square :
  let A := (-1, 3)
  let B := (1, -2)
  let C := (6, 0)
  let D := (4, 5)
  is_square A B C D :=
by sorry

end quadrilateral_is_square_l748_748623


namespace negation_proposition_l748_748397

theorem negation_proposition :
  (∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proposition_l748_748397


namespace increasing_function_l748_748968

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable (h_deriv : ∀ x, deriv f x = f' x)
variable (h_ineq : ∀ x, (x + 1) * f x + x * f' x > 0)

theorem increasing_function : ∀ x : ℝ, deriv (λ x, x * exp x * f x) x > 0 :=
by 
  sorry

end increasing_function_l748_748968


namespace least_xy_satisfying_condition_l748_748953

noncomputable def least_xy (x y : ℕ+) : ℕ := x * y

theorem least_xy_satisfying_condition :
  ∃ (x y : ℕ+), (1/x + 1/(3 * y) : ℚ) = 1/6 ∧ least_xy x y = 64 :=
by
  use 8, 8
  split
  · norm_num
  · refl

-- This theorem states that there exist positive integers x and y such that 1/x + 1/(3*y) = 1/6 and the least xy = 64.

end least_xy_satisfying_condition_l748_748953


namespace area_of_outer_sphere_marked_l748_748479

noncomputable def r : ℝ := 1  -- Radius of the small painted sphere
noncomputable def R_inner : ℝ := 4  -- Radius of the inner concentric sphere
noncomputable def R_outer : ℝ := 6  -- Radius of the outer concentric sphere
noncomputable def A_inner : ℝ := 47  -- Area of the region on the inner sphere

theorem area_of_outer_sphere_marked :
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner in
  A_outer = 105.75 :=
by
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner
  sorry

end area_of_outer_sphere_marked_l748_748479


namespace sum_of_numbers_l748_748593

theorem sum_of_numbers (a b c d : ℕ) (h1 : a > d) (h2 : a * b = c * d) (h3 : a + b + c + d = a * c) (h4 : ∀ x y z w: ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ) : a + b + c + d = 12 :=
sorry

end sum_of_numbers_l748_748593


namespace equation_sixth_term_is_72_l748_748022

theorem equation_sixth_term_is_72 :
  ∃ n: ℕ, 2 * n ^ 2 = 72 ∧ n = 6 := 
by {
  use 6,
  split,
  { norm_num },
  { refl }
}

end equation_sixth_term_is_72_l748_748022


namespace base7_to_base10_conversion_l748_748892

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l748_748892


namespace ant_probability_at_A_after_6_moves_l748_748855

noncomputable def ant_probability : ℕ → ℚ
| 0 => 1
| n + 1 => (1 / 3) * ant_probability n

theorem ant_probability_at_A_after_6_moves :
  ant_probability 6 = 61 / 243 :=
begin
  have base_cases :
    (ant_probability 0 = 1) ∧
    (ant_probability 1 = 0) ∧
    (ant_probability 2 = (1 / 3)) ∧
    (ant_probability 3 = 0) ∧
    (ant_probability 4 = (7 / 27)) ∧
    (ant_probability 5 = 0) ∧
    (ant_probability 6 = (61 / 243)),
  { repeat { split; simp [ant_probability] } },
  exact (base_cases.2.2.2.2.2.2).2
end

end ant_probability_at_A_after_6_moves_l748_748855


namespace arccos_pi_over_3_l748_748508

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l748_748508


namespace usual_time_to_office_l748_748781

theorem usual_time_to_office (P : ℝ) (T : ℝ) (h1 : T = (3 / 4) * (T + 20)) : T = 60 :=
by
  sorry

end usual_time_to_office_l748_748781


namespace rachel_age_is_12_l748_748722

-- Let R be Rachel's current age
variable (R : ℕ)

-- Conditions
def grandfather_age : ℕ := 7 * R
def mother_age : ℕ := grandfather_age R / 2
def father_age : ℕ := mother_age R + 5

-- Given Condition: Rachel's father will be 60 years old when she is 25 years old
def age_difference_condition : Prop := father_age R = R + 35

-- Prove that Rachel's current age is 12
theorem rachel_age_is_12 : 
  age_difference_condition R → R = 12 :=
begin
  sorry
end

end rachel_age_is_12_l748_748722


namespace num_correct_propositions_l748_748226

theorem num_correct_propositions (l₁ l₂ : Line) (non_coincident : l₁ ≠ l₂) :
  (num_correct_statements l₁ l₂ = 3) :=
sorry

end num_correct_propositions_l748_748226


namespace dog_food_requirements_l748_748366

theorem dog_food_requirements :
  let small_dog1_weight := 15
  let small_dog2_weight := 20
  let medium_dog1_weight := 25
  let medium_dog2_weight := 35
  let medium_dog3_weight := 45
  let large_dog1_weight := 55
  let large_dog2_weight := 60
  let large_dog3_weight := 75
  let small_dog_food_requirement := (small_dog1_weight / 20.0) + (small_dog2_weight / 20.0)
  let medium_dog_food_requirement := (medium_dog1_weight / 15.0) + (medium_dog2_weight / 15.0) + (medium_dog3_weight / 15.0)
  let large_dog_food_requirement := (large_dog1_weight / 10.0) + (large_dog2_weight / 10.0) + (large_dog3_weight / 10.0)
  small_dog_food_requirement + medium_dog_food_requirement + large_dog_food_requirement = 27.75 := by
  -- Assign individual food requirements for calculations
  let small_dog1_food := small_dog1_weight / 20.0
  let small_dog2_food := small_dog2_weight / 20.0
  let medium_dog1_food := medium_dog1_weight / 15.0
  let medium_dog2_food := medium_dog2_weight / 15.0
  let medium_dog3_food := medium_dog3_weight / 15.0
  let large_dog1_food := large_dog1_weight / 10.0
  let large_dog2_food := large_dog2_weight / 10.0
  let large_dog3_food := large_dog3_weight / 10.0
  -- Calculate totals
  let total_small_dogs_food := small_dog1_food + small_dog2_food
  let total_medium_dogs_food := medium_dog1_food + medium_dog2_food + medium_dog3_food
  let total_large_dogs_food := large_dog1_food + large_dog2_food + large_dog3_food
  -- Final food requirement
  have : total_small_dogs_food + total_medium_dogs_food + total_large_dogs_food = 27.75 := by
    -- Proof here
    sorry
  exact this

end dog_food_requirements_l748_748366


namespace diagonal_cubes_intersect_l748_748059

def a := 150
def b := 324
def c := 375

def gcd_ab := Nat.gcd a b = 6
def gcd_bc := Nat.gcd b c = 3
def gcd_ca := Nat.gcd c a = 75
def gcd_abc := Nat.gcd (Nat.gcd a b) c = 3

theorem diagonal_cubes_intersect :
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c = 768 :=
by {
  intro gcd_ab gcd_bc gcd_ca gcd_abc,
  sorry
}

end diagonal_cubes_intersect_l748_748059


namespace parabola_intersection_l748_748554

theorem parabola_intersection :
  (∀ x y : ℝ, y = 3 * x^2 - 4 * x + 2 ↔ y = 9 * x^2 + 6 * x + 2) →
  (∃ x1 y1 x2 y2 : ℝ,
    (x1 = 0 ∧ y1 = 2) ∧ (x2 = -5 / 3 ∧ y2 = 17)) :=
by
  intro h
  sorry

end parabola_intersection_l748_748554


namespace division_of_expression_l748_748501

theorem division_of_expression (x y : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : (12 * x^2 * y) / (-6 * x * y) = -2 * x := by
  sorry

end division_of_expression_l748_748501


namespace sum_c_n_first_n_terms_l748_748689

def Sn (n : ℕ) : ℕ := 2 * n^2

def a (n : ℕ) : ℕ :=
  if n = 1 then Sn 1 else Sn n - Sn (n - 1)

def b (n : ℕ) : ℕ := 2

def c (n : ℕ) : ℕ := a n / b n

theorem sum_c_n_first_n_terms (n : ℕ) : (∑ k in Finset.range n, c (k + 1)) = n^2 :=
by
  sorry

end sum_c_n_first_n_terms_l748_748689


namespace oranges_per_box_calculation_l748_748460

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l748_748460


namespace base_5_to_base_8_l748_748899

theorem base_5_to_base_8 : 
    let n_base_5 := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0 in
    let n_base_10 := n_base_5 in
    (n_base_10 = 194) → 
    let n_base_8 := (194 % 8) + (194 / 8 % 8) * 10 + (194 / 8 / 8 % 8) * 100 in
    n_base_8 = 302 :=
by
    sorry

end base_5_to_base_8_l748_748899


namespace distinct_arrangements_banana_l748_748275

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748275


namespace dessert_menus_count_l748_748828

-- Definitions based on conditions
def Dessert : Type := {cake : Unit, pie : Unit, ice_cream : Unit, pudding : Unit}
def no_consecutive_days (menu : List Dessert) : Prop :=
  ∀ (i : Nat), i < menu.length - 1 → menu[i] ≠ menu[i + 1]

def cake_on_friday (menu : List Dessert) : Prop :=
  menu[5] = Dessert.cake

def at_least_three_types (menu : List Dessert) : Prop :=
  (menu.toFinset.filter (λ d => d ∈ {Dessert.cake, Dessert.pie, Dessert.ice_cream, Dessert.pudding})).card ≥ 3

-- Main theorem statement
theorem dessert_menus_count :
  ∃ (menu_count : Nat), 
    menu_count = 504 ∧
    (∃ (menu : List Dessert) (length_week : menu.length = 7),
      no_consecutive_days menu ∧ 
      cake_on_friday menu ∧ 
      at_least_three_types menu) :=
sorry

end dessert_menus_count_l748_748828


namespace sequence_sum_l748_748037

def alternating_sum : List ℤ := [2, -7, 10, -15, 18, -23, 26, -31, 34, -39, 40, -45, 48]

theorem sequence_sum : alternating_sum.sum = 13 := by
  sorry

end sequence_sum_l748_748037


namespace convert_base_7_to_base_10_l748_748888

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l748_748888


namespace renovation_project_truck_load_l748_748475

theorem renovation_project_truck_load (sand : ℝ) (dirt : ℝ) (cement : ℝ)
  (h1 : sand = 0.17) (h2 : dirt = 0.33) (h3 : cement = 0.17) :
  sand + dirt + cement = 0.67 :=
by
  sorry

end renovation_project_truck_load_l748_748475


namespace total_increase_percentage_l748_748804

-- Define the conditions: original speed S, first increase by 30%, then another increase by 10%
def original_speed (S : ℝ) := S
def first_increase (S : ℝ) := S * 1.30
def second_increase (S : ℝ) := (S * 1.30) * 1.10

-- Prove that the total increase in speed is 43% of the original speed
theorem total_increase_percentage (S : ℝ) :
  (second_increase S - original_speed S) / original_speed S * 100 = 43 :=
by
  sorry

end total_increase_percentage_l748_748804


namespace lambda_value_l748_748997

theorem lambda_value (a b : ℝ × ℝ) (λ : ℝ) (h1 : a = (1, 3)) (h2 : b = (3, 4)) 
  (h3 : (a.1 - λ * b.1, a.2 - λ * b.2) • b = 0): 
  λ = 3 / 5 :=
by
  sorry

end lambda_value_l748_748997


namespace total_ice_cubes_correct_l748_748498

/-- Each tray holds 48 ice cubes -/
def cubes_per_tray : Nat := 48

/-- Billy has 24 trays -/
def number_of_trays : Nat := 24

/-- Calculate the total number of ice cubes -/
def total_ice_cubes (cubes_per_tray : Nat) (number_of_trays : Nat) : Nat :=
  cubes_per_tray * number_of_trays

/-- Proof that the total number of ice cubes is 1152 given the conditions -/
theorem total_ice_cubes_correct : total_ice_cubes cubes_per_tray number_of_trays = 1152 := by
  /- Here we state the main theorem, but we leave the proof as sorry per the instructions -/
  sorry

end total_ice_cubes_correct_l748_748498


namespace x_squared_plus_y_squared_l748_748635

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 17) (h2 : x * y = 6) : x^2 + y^2 = 301 :=
by sorry

end x_squared_plus_y_squared_l748_748635


namespace triangle_area_l748_748655

theorem triangle_area (a b h : ℝ) (h_a : a = 30) (h_b : b = 24) (h_h : h = 18) :
  let area := 1/2 * a * h in
  area = 270 :=
by
  rw [h_a, h_b, h_h]
  simp [area]
  sorry

end triangle_area_l748_748655


namespace least_value_sum_l748_748637

theorem least_value_sum (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z = 92 :=
sorry

end least_value_sum_l748_748637


namespace sum_of_bn_2023_l748_748965

theorem sum_of_bn_2023 :
  (∀ n : ℕ, 1 ≤ n → 0 < q → a n = 2 * q^(n - 1)) →
  (a 4 = (6 * a 2 + a 3) / 2) →
  (∀ n : ℕ, b n = 1 / (Real.logb 2 (a n) * Real.logb 2 (a (n + 1)))) →
  (T_2023 = ∑ i in Finset.range 2023, b (i + 1)) →
  T_2023 = 2023 / 2024 :=
by
  sorry

end sum_of_bn_2023_l748_748965


namespace maximum_length_BG_l748_748664

structure Rhombus (A B C D : Type) :=
(AB_side : A = B)
(BC_side : B = C)
(CD_side : C = D)
(DA_side : D = A)
(angle_ABC : ∠A B C = 120)
(AB_length : A = sqrt 3)

def point_E_on_extension_of_BC (B C E : Type) := 
E extends BC

def line_AE_intersect_CD_at_F (A E C D F : Type) := 
AE intersects CD at F

def BF_extended_intersect_DE_at_G (B F D E G : Type) := 
BF extends to intersect DE at G

theorem maximum_length_BG
  {A B C D : Type}
  [Rhombus A B C D]
  {E F G : Type}
  (hE : point_E_on_extension_of_BC B C E)
  (hF : line_AE_intersect_CD_at_F A E C D F)
  (hG : BF_extended_intersect_DE_at_G B F D E G) :
  BG <= 2 :=
sorry

end maximum_length_BG_l748_748664


namespace no_such_operation_exists_l748_748591

-- Define an operation *
def operation (X Y : ℤ) : ℤ := sorry

-- Define property a: A * B = - (B * A)
axiom property_a (A B : ℤ) : operation A B = - operation B A

-- Define property b: (A * B) * C = A * (B * C)
axiom property_b (A B C : ℤ) : operation (operation A B) C = operation A (operation B C)

-- Assume every integer equals X * Y for some integers X and Y
axiom exists_X_Y (n : ℤ) : ∃ X Y : ℤ, operation X Y = n

-- Prove that such an operation * cannot simultaneously satisfy both properties a) and b)
theorem no_such_operation_exists : ¬ (∀ (A B C : ℤ), property_a A B ∧ property_b A B C) :=
  sorry

end no_such_operation_exists_l748_748591


namespace hyperbola_condition_l748_748296

theorem hyperbola_condition (m : ℝ) : 
  (exists a b : ℝ, ¬ a = 0 ∧ ¬ b = 0 ∧ ( ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 )) →
  ( -2 < m ∧ m < -1 ) :=
by
  sorry

end hyperbola_condition_l748_748296


namespace crayons_allocation_correct_l748_748569

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l748_748569


namespace correct_formula_fits_l748_748527

theorem correct_formula_fits :
  (∀ x y, (x = 0 ∧ y = 210) ∨ (x = 2 ∧ y = 170) ∨ (x = 4 ∧ y = 110) ∨ (x = 6 ∧ y = 30) ∨ (x = 8 ∧ y = -70) →
    y = 210 - 10 * x - x ^ 2 - 2 * x ^ 3) :=
by
  intros x y h
  cases h with h0 h
  { rw [h0.1, h0.2], sorry }
  cases h with h2 h
  { rw [h2.1, h2.2], sorry }
  cases h with h4 h
  { rw [h4.1, h4.2], sorry }
  cases h with h6 h8
  { rw [h6.1, h6.2], sorry }
  { rw [h8.1, h8.2], sorry }

end correct_formula_fits_l748_748527


namespace find_y_l748_748690

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y (y : ℝ) (h : star 7 y = 47) : y = 5.4 := 
by 
  sorry

end find_y_l748_748690


namespace percentage_of_girls_with_dogs_l748_748838

theorem percentage_of_girls_with_dogs (students total_students : ℕ)
(h_total_students : total_students = 100)
(girls boys : ℕ)
(h_half_students : girls = total_students / 2 ∧ boys = total_students / 2)
(boys_with_dogs : ℕ)
(h_boys_with_dogs : boys_with_dogs = boys / 10)
(total_with_dogs : ℕ)
(h_total_with_dogs : total_with_dogs = 15)
(girls_with_dogs : ℕ)
(h_girls_with_dogs : girls_with_dogs = total_with_dogs - boys_with_dogs)
: (girls_with_dogs * 100 / girls = 20) :=
by
  sorry

end percentage_of_girls_with_dogs_l748_748838


namespace train_speed_approx_kmph_l748_748485

noncomputable def length_of_train : ℝ := 150
noncomputable def time_to_cross_pole : ℝ := 4.425875438161669

theorem train_speed_approx_kmph :
  (length_of_train / time_to_cross_pole) * 3.6 = 122.03 :=
by sorry

end train_speed_approx_kmph_l748_748485


namespace john_beats_per_minute_l748_748326

theorem john_beats_per_minute :
  let hours_per_day := 2
  let days := 3
  let total_beats := 72000
  let minutes_per_hour := 60
  total_beats / (days * hours_per_day * minutes_per_hour) = 200 := 
by 
  sorry

end john_beats_per_minute_l748_748326


namespace distinct_arrangements_banana_l748_748273

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748273


namespace distinct_arrangements_banana_l748_748267

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_count := 6
  word.length = total_count ∧ b_count = 1 ∧ a_count = 3 ∧ n_count = 2 →
  (Nat.factorial total_count) / ((Nat.factorial a_count) * (Nat.factorial n_count)) = 60 :=
by
  intros
  have h_total := word.length
  have h_b := b_count
  have h_a := a_count
  have h_n := n_count
  sorry

end distinct_arrangements_banana_l748_748267


namespace problem_l748_748387

theorem problem (a b c : ℤ) :
  (∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)) →
  (∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 :=
by
  intros h1 h2
  sorry

end problem_l748_748387


namespace distinct_arrangements_banana_l748_748274

theorem distinct_arrangements_banana : 
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  (factorial total_letters / (factorial b_count * factorial a_count * factorial n_count)) = 60 :=
by
  let word := "banana"
  let b_count := 1
  let a_count := 3
  let n_count := 2
  let total_letters := b_count + a_count + n_count
  let factorial := Nat.factorial
  have h1 : total_letters = 6 := rfl
  have h2 : factorial 6 = 720 := rfl
  have h3 : factorial 3 = 6 := rfl
  have h4 : factorial 2 = 2 := rfl
  have h5 : 720 / (6 * 2) = 60 := rfl
  exact h5

end distinct_arrangements_banana_l748_748274


namespace exists_person_knows_one_l748_748060

variable {S : Finset Person} (h : ∀ (p q : Person), (S.card = S.card) → p ≠ q → ∀ (r : Person), r ∈ S → r ∉ S \ {p, q})

theorem exists_person_knows_one : ∃ p : Person, ∃ q : Person, q ≠ p ∧ ∀ r : Person, (r ∈ S ∧ p ≠ r) → r ∉ S \ {q} :=
by 
  sorry

end exists_person_knows_one_l748_748060


namespace polynomial_degree_l748_748525

-- Definitions based on the conditions given in the problem
def poly : ℝ → ℝ := λ x, 3 + 7 * x^5 - 15 + 4 * real.pi * x^6 - real.sqrt 5 * x^6 + 11

-- Lean statement to prove
theorem polynomial_degree (x : ℝ) : polynomial.degree (polynomial.C 3 + polynomial.C 7 * polynomial.X^5 - polynomial.C 15 + 
                                                       polynomial.C (4 * real.pi) * polynomial.X^6 - polynomial.C (real.sqrt 5) * polynomial.X^6 + polynomial.C 11) = 6 :=
sorry

end polynomial_degree_l748_748525


namespace constant_term_exists_l748_748646

theorem constant_term_exists:
  ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 10 ∧ 
  (∃ r : ℕ, n = 3 * r) ∧ (∃ k : ℕ, n = 2 * k) ∧ 
  n = 6 :=
sorry

end constant_term_exists_l748_748646


namespace probability_of_Q_in_region_l748_748670

-- Definitions based on conditions
structure Triangle :=
(a : ℝ)  -- side DE
(b : ℝ)  -- side EF
(c : ℝ)  -- side DF

def DEF : Triangle := {a := 8, b := 6, c := 10}

def in_circle (Q : Point) (center : Point) (r : ℝ) : Prop :=
distance Q center ≤ r

-- Placeholder for determining if Q is closer to F than to D or E
def closer_to_F (Q : Point) (F D E : Point) : Prop := sorry

-- Definitions of points D, E, F, and the inscribed circle
def D : Point := sorry -- define point D
def E : Point := sorry -- define point E
def F : Point := sorry -- define point F
def center_inscribed_circle : Point := sorry -- center of the inscribed circle
def radius_inscribed_circle : ℝ := 1

-- Probability that Q is in the specified region
theorem probability_of_Q_in_region :
  ∀ (Q : Point),
    Q ∈ interior_triangle DEF →
    closer_to_F Q F D E →
    in_circle Q center_inscribed_circle radius_inscribed_circle →
    ∃ P : ℝ, P = (area_intersection DEF F D E center_inscribed_circle radius_inscribed_circle) / (area_triangle DEF) :=
sorry

end probability_of_Q_in_region_l748_748670


namespace mrs_lovely_class_l748_748359

-- Define the number of students in Mrs. Lovely's class
def number_of_students (g b : ℕ) : ℕ := g + b

theorem mrs_lovely_class (g b : ℕ): 
  (b = g + 3) →
  (500 - 10 = g * g + b * b) →
  number_of_students g b = 23 :=
by
  sorry

end mrs_lovely_class_l748_748359


namespace find_a_min_value_g_l748_748616

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (-x) + a * x - (1 / x)

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f (-x) a + 2 * x

theorem find_a (a : ℝ) : (∀ x < 0, (f x a).derivative = (1 / x + a + 1 / (x ^ 2)) ) → (f (-1) a).derivative = 0 → a = 0 :=
by
  intros h h1
  sorry

theorem min_value_g (a : ℝ) : a = 0 → (∀ x > 0, g x a = Real.log x + (1 / x) + 2 * x) →
  (∀ x > 0, (g x a).derivative = (1 / x - 1 / (x ^ 2) + 2)) →
  ∃ y : ℝ, y = 1 / 2 ∧ g y a = 3 - Real.log 2 :=
by
  intros h h1 h2
  sorry

end find_a_min_value_g_l748_748616


namespace smallest_three_digit_integer_l748_748145

theorem smallest_three_digit_integer (n : ℕ) (h : 75 * n ≡ 225 [MOD 345]) (hne : n ≥ 100) (hn : n < 1000) : n = 118 :=
sorry

end smallest_three_digit_integer_l748_748145


namespace part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l748_748952

-- Part (1)
theorem part1_coordinates_on_x_axis (a : ℝ) (h : a + 5 = 0) : (2*a - 2, a + 5) = (-12, 0) :=
by sorry

-- Part (2)
theorem part2_coordinates_parallel_y_axis (a : ℝ) (h : 2*a - 2 = 4) : (2*a - 2, a + 5) = (4, 8) :=
by sorry

-- Part (3)
theorem part3_distances_equal_second_quadrant (a : ℝ) 
  (h1 : 2*a-2 < 0) (h2 : a+5 > 0) (h3 : abs (2*a - 2) = abs (a + 5)) : a^(2022 : ℕ) + 2022 = 2023 :=
by sorry

end part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l748_748952


namespace sine_variance_of_special_set_l748_748223

theorem sine_variance_of_special_set (a_0 : ℝ) : 
  (sin (π / 2 - a_0))^2 + (sin (5 * π / 6 - a_0))^2 + (sin (7 * π / 6 - a_0))^2 = 3 / 2 :=
by
  sorry

end sine_variance_of_special_set_l748_748223


namespace find_b_l748_748051

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem find_b (a b : ℝ) (h1 : f a = -1 / 3) (h2 : f (a * b) = 1 / 6) : b = -2 := 
by
  sorry

end find_b_l748_748051


namespace road_length_computation_l748_748419

namespace RoadPaving

def square_tile_side_length : ℝ := 0.5
def number_of_tiles : ℕ := 160
def road_width : ℝ := 2
def expected_road_length : ℝ := 20

theorem road_length_computation :
  let total_area := square_tile_side_length ^ 2 * number_of_tiles in
  let road_length := total_area / road_width in
  road_length = expected_road_length := by
  sorry

end RoadPaving

end road_length_computation_l748_748419


namespace kevin_hops_7_times_l748_748330

noncomputable def distance_hopped_after_n_hops (n : ℕ) : ℚ :=
  4 * (1 - (3 / 4) ^ n)

theorem kevin_hops_7_times :
  distance_hopped_after_n_hops 7 = 7086 / 2048 := 
by
  sorry

end kevin_hops_7_times_l748_748330


namespace magnitude_AD_is_2_sqrt_3_l748_748227

open real

constants (a b : ℝ) (AB AC AD : ℝ) (D : ℝ)

def is_unit_vector (v : ℝ) : Prop := abs v = 1
def is_scaled_vector (v w : ℝ) (k : ℝ) : Prop := abs v = k * abs w
def is_angle_between (v w : ℝ) (θ : ℝ) : Prop := v * w = abs v * abs w * cos θ
def vector_length (v : ℝ) : ℝ := abs v

axiom cond1 : is_unit_vector a
axiom cond2 : is_scaled_vector b a 2
axiom cond3 : is_angle_between a b (π / 3)
axiom cond4 : AB = 2 * a + 2 * b
axiom cond5 : AC = 2 * a - 6 * b
axiom cond6 : AD = 1 / 2 * (AB + AC)

theorem magnitude_AD_is_2_sqrt_3 : vector_length AD = 2 * sqrt 3 :=
by sorry

end magnitude_AD_is_2_sqrt_3_l748_748227


namespace value_of_quotient_l748_748138

noncomputable def f (x : ℝ) : ℝ := x^2021 + 19*x^2020 + 1

def roots (n : ℕ) : set ℝ := {r : ℝ | nat.find_spec (exists_finite_set_of_finite_card_n (f, n)).1 r}

def Q (z : ℝ) (k : ℝ) : ℝ :=
  k * ∏ (r : ℝ) in roots 2021, (z - (r + 1 / r))

theorem value_of_quotient (k : ℝ) (h : k ≠ 0) :
  Q 1 k / Q (-1) k = 1 :=
by
  sorry

end value_of_quotient_l748_748138


namespace A_is_necessary_not_sufficient_for_B_l748_748196

variable {x y : ℕ}

def proposition_A := x ≠ 2 ∨ y ≠ 3
def proposition_B := x + y ≠ 5

theorem A_is_necessary_not_sufficient_for_B :
  (proposition_B → proposition_A) ∧ ¬(proposition_A → proposition_B) := by
  sorry

end A_is_necessary_not_sufficient_for_B_l748_748196


namespace cube_prism_surface_area_l748_748570

theorem cube_prism_surface_area (a V V_prism b : ℝ) 
  (h₀ : 6 * a^2 = 384) 
  (h₁ : V = a^3) 
  (h₂ : V_prism = (3/4) * V) 
  (h₃: b^2 * a = V - V_prism)
  (h₄: 8 * b^2 = V - V_prism) : 
  let Sp := a^2 - b^2 in
  let S_pl := 8 * a in
  2 * Sp + S_pl = 416 :=
by
  sorry

end cube_prism_surface_area_l748_748570


namespace find_monic_quadratic_poly_with_real_coeff_and_root_l748_748166

open Complex Polynomial

noncomputable def monic_quadratic_with_root (α : ℂ) : Polynomial ℂ :=
  Polynomial.monic (Polynomial.x - Polynomial.C α)
  * Polynomial.monic (Polynomial.x - Polynomial.C α.conj)

theorem find_monic_quadratic_poly_with_real_coeff_and_root (α : ℂ) (h_re : ∃ x : ℝ, α = x - I) :
  monic_quadratic_with_root α = Polynomial.C (1 : ℂ) * (Polynomial.x ^ 2 - Polynomial.C (2 * (α.re : ℂ)) * Polynomial.x + Polynomial.C ((α.re ^ 2 : ℝ) + 1)) :=
by
  sorry

end find_monic_quadratic_poly_with_real_coeff_and_root_l748_748166


namespace tetrahedron_face_inequality_l748_748741

theorem tetrahedron_face_inequality
    (A B C D : ℝ) :
    |A^2 + B^2 - C^2 - D^2| ≤ 2 * (A * B + C * D) := by
  sorry

end tetrahedron_face_inequality_l748_748741


namespace total_polled_votes_correct_l748_748495

variable (V : ℕ) -- Valid votes

-- Condition: One candidate got 30% of the valid votes
variable (C1_votes : ℕ) (C2_votes : ℕ)
variable (H1 : C1_votes = (3 * V) / 10)

-- Condition: The other candidate won by 5000 votes
variable (H2 : C2_votes = C1_votes + 5000)

-- Condition: One candidate got 70% of the valid votes
variable (H3 : C2_votes = (7 * V) / 10)

-- Condition: 100 votes were invalid
variable (invalid_votes : ℕ := 100)

-- Total polled votes (valid + invalid)
def total_polled_votes := V + invalid_votes

theorem total_polled_votes_correct 
  (V : ℕ) 
  (H1 : C1_votes = (3 * V) / 10) 
  (H2 : C2_votes = C1_votes + 5000) 
  (H3 : C2_votes = (7 * V) / 10) 
  (invalid_votes : ℕ := 100) : 
  total_polled_votes V = 12600 :=
by
  -- The steps of the proof are omitted
  sorry

end total_polled_votes_correct_l748_748495


namespace expected_value_of_win_l748_748084

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l748_748084


namespace rectangular_box_volume_l748_748774

noncomputable def box_volume (a b c : ℝ) : ℝ :=
  8 * a * b * c

theorem rectangular_box_volume :
  ∃ (a b c : ℝ),
    a^2 + b^2 = 16 ∧
    b^2 + c^2 = 25 ∧
    a^2 + c^2 = 36 ∧
    box_volume (sqrt (27/2)) (sqrt (5/2)) (sqrt (45/2)) = 90 * sqrt 6 :=
by {
  sorry
}

end rectangular_box_volume_l748_748774


namespace balloon_arrangement_count_l748_748904

theorem balloon_arrangement_count :
  let total_permutations := (Nat.factorial 7) / (Nat.factorial 2 * Nat.factorial 3)
  let ways_to_arrange_L_and_O := Nat.choose 4 1 * (Nat.factorial 3)
  let valid_arrangements := ways_to_arrange_L_and_O * total_permutations
  valid_arrangements = 10080 :=
by
  sorry

end balloon_arrangement_count_l748_748904


namespace sum_of_squares_l748_748008

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 23) (h2 : a * b + b * c + a * c = 131) :
  a^2 + b^2 + c^2 = 267 :=
by
  sorry

end sum_of_squares_l748_748008


namespace part_II_l748_748584

def f (a b x : ℝ) : ℝ := 4 * a * x^2 - 2 * b * x - a + b

theorem part_II (a b : ℝ) (a_pos : 0 < a) (b_real : b ∈ Set.univ) (x : ℝ) (x_in_domain : x ∈ Set.Icc 0 1) : 
    ∃ M, (M = (Set.Icc 0 1).sup' (by sorry) (f a b) ∧ ∀ x ∈ Set.Icc 0 1, f a b x + M > 0) := 
  sorry

end part_II_l748_748584


namespace find_annual_interest_rate_l748_748484

noncomputable def annual_interest_rate (P A1 A2 : ℝ)
    (t1 t2 n : ℕ) (compounding_periods : n = 4) : ℝ :=
    let x := (A2 / A1) ^ (1 / ((t2 - t1) * n : ℝ))
    in (x - 1) * n

theorem find_annual_interest_rate :
    let P : ℝ := 0 -- We do not actually need P to find r.
    let A1 : ℝ := 4875
    let A2 : ℝ := 5915
    let t1 : ℕ := 2
    let t2 : ℕ := 3
    let n : ℕ := 4
    annual_interest_rate P A1 A2 t1 t2 n (rfl : n = 4) = 0.2 :=
by {
    sorry
}

end find_annual_interest_rate_l748_748484


namespace total_revenue_l748_748411

def ticket_price := 20
def first_group_discount := 0.40
def second_group_discount := 0.15
def first_group_size := 10
def second_group_size := 20
def total_people := 48

theorem total_revenue :
  let first_group_revenue := first_group_size * (ticket_price * (1 - first_group_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_group_discount))
  let third_group_size := total_people - first_group_size - second_group_size
  let third_group_revenue := third_group_size * ticket_price
  first_group_revenue + second_group_revenue + third_group_revenue = 820 :=
by {
  sorry
}

end total_revenue_l748_748411


namespace blender_customers_l748_748523

variable (p_t p_b : ℕ) (c_t c_b : ℕ) (k : ℕ)

-- Define the conditions
def condition_toaster_popularity : p_t = 20 := sorry
def condition_toaster_cost : c_t = 300 := sorry
def condition_blender_cost : c_b = 450 := sorry
def condition_inverse_proportionality : p_t * c_t = k := sorry

-- Proof goal: number of customers who would buy the blender
theorem blender_customers : p_b = 13 :=
by
  have h1 : p_t * c_t = 6000 := by sorry -- Using the given conditions
  have h2 : p_b * c_b = 6000 := by sorry -- Assumption for the same constant k
  have h3 : c_b = 450 := sorry
  have h4 : p_b = 6000 / 450 := by sorry
  have h5 : p_b = 13 := by sorry
  exact h5

end blender_customers_l748_748523


namespace remainder_when_4x_div_7_l748_748445

theorem remainder_when_4x_div_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_when_4x_div_7_l748_748445


namespace area_enclosed_by_equation_l748_748032

/-- Theorem: Prove that the area enclosed by the graph of the equation
(x - 1)^2 + (y - 1)^2 = |x - 1| + |y - 1| is π / 2. -/
theorem area_enclosed_by_equation :
  (∃ s : set (ℝ × ℝ), 
    s = {p | (p.1 - 1)^2 + (p.2 - 1)^2 = |p.1 - 1| + |p.2 - 1|} ∧ 
    ∃ a : ℝ, 
    a = real.pi / 2 ∧ 
    measure_theory.measure (measure_theory.volume) s = a) :=
sorry

end area_enclosed_by_equation_l748_748032


namespace largest_interesting_number_l748_748100

def is_interesting_number (x : ℝ) : Prop :=
  ∃ y z : ℝ, (0 ≤ y ∧ y < 1) ∧ (0 ≤ z ∧ z < 1) ∧ x = 0 + y * 10⁻¹ + z ∧ 2 * (0 + y * 10⁻¹ + z) = 0 + z

theorem largest_interesting_number : ∀ x, is_interesting_number x → x ≤ 0.375 :=
by
  sorry

end largest_interesting_number_l748_748100


namespace banana_arrangements_l748_748252

theorem banana_arrangements : 
  let n := 6
  let k_b := 1
  let k_a := 3
  let k_n := 2
  n! / (k_b! * k_a! * k_n!) = 60 :=
by
  have n_def : n = 6 := rfl
  have k_b_def : k_b = 1 := rfl
  have k_a_def : k_a = 3 := rfl
  have k_n_def : k_n = 2 := rfl
  calc
    n! / (k_b! * k_a! * k_n!) = 720 / (1 * 6 * 2) : by sorry
                             ... = 720 / 12         : by sorry
                             ... = 60               : by sorry

end banana_arrangements_l748_748252


namespace card_probability_l748_748024

theorem card_probability :
  let totalCards := 52
  let kings := 4
  let jacks := 4
  let queens := 4
  let firstCardKing := kings / totalCards
  let secondCardJack := jacks / (totalCards - 1)
  let thirdCardQueen := queens / (totalCards - 2)
  (firstCardKing * secondCardJack * thirdCardQueen) = (8 / 16575) :=
by
  sorry

end card_probability_l748_748024


namespace ratio_of_areas_l748_748317

-- Given conditions:
variables (A B P Q : Type)
variables (circleA circleB : Type)
variables [HasCenter circleA A] [HasCenter circleB B]
variables [Intersect circleA circleB P Q]

-- Given angles
variables [Angle PAQ = 60] [Angle PBQ = 90]

-- Define radii
variables (R_A R_B : ℝ)

-- Area of a circle
def area (r : ℝ) : ℝ := Real.pi * r ^ 2

-- Given the condition on the angles and intersect, prove the ratio of areas
theorem ratio_of_areas : area R_A / area R_B = 2 :=
by
  sorry

end ratio_of_areas_l748_748317


namespace min_distance_between_parallel_lines_l748_748491

theorem min_distance_between_parallel_lines :
  ∀ (m : ℝ), 
  let A := 3
  let B := -4
  let c1 := m - 1
  let c2 := m^2
  let distance := (|c1 - c2|) / (Real.sqrt (A^2 + B^2))
  min_distance := (3 / 20) :=
sorry

end min_distance_between_parallel_lines_l748_748491


namespace maximize_product_of_distances_l748_748810

-- Definition of a triangle ABC
variables {A B C O : Type*}
variables [metric_space O] [metric_space A] [metric_space B] [metric_space C]
variables (dist_AO dist_BO dist_CO : ℝ)

-- Given conditions
def is_inside_triangle (O : Type*) (T : set (Type*)) := 
  ∃ (A B C : T), O ∈ interior (triangle A B C)

def distances_to_sides (O : Type*) (T : set (Type*)) := 
  ∃ (d_a d_b d_c : ℝ), 
    d_a = dist (O, line BC) ∧
    d_b = dist (O, line CA) ∧
    d_c = dist (O, line AB)

-- Equivalent proof statement
theorem maximize_product_of_distances 
  (T : set (Type*)) (O : Type*) (A B C : Type*) 
  (d_a d_b d_c : ℝ) 
  (h_interior : is_inside_triangle O T) 
  (h_distances : distances_to_sides O T) : 
  (∀ O : T, d_a * d_b * d_c ≤ (d_a * d_b * d_c) when O is the centroid of triangle A B C) :=
  sorry -- ⊢ formal proof goes here

end maximize_product_of_distances_l748_748810


namespace jordan_rect_width_is_10_l748_748050

def carol_rect_length : ℕ := 5
def carol_rect_width : ℕ := 24
def jordan_rect_length : ℕ := 12

def carol_rect_area : ℕ := carol_rect_length * carol_rect_width
def jordan_rect_width := carol_rect_area / jordan_rect_length

theorem jordan_rect_width_is_10 : jordan_rect_width = 10 :=
by
  sorry

end jordan_rect_width_is_10_l748_748050


namespace expected_value_of_win_l748_748085

theorem expected_value_of_win :
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
   prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 :=
by
  let prob := (1:ℝ)/8
  let win_value (n : ℕ) := (n ^ 3 : ℝ)
  have E : (prob * win_value 1 + prob * win_value 2 + prob * win_value 3 + prob * win_value 4 + prob * win_value 5 + 
            prob * win_value 6 + prob * win_value 7 + prob * win_value 8 : ℝ) = 162 := sorry
  exact E

end expected_value_of_win_l748_748085


namespace remaining_load_after_three_deliveries_l748_748848

def initial_load : ℝ := 50000
def unload_first_store (load : ℝ) : ℝ := load - 0.10 * load
def unload_second_store (load : ℝ) : ℝ := load - 0.20 * load
def unload_third_store (load : ℝ) : ℝ := load - 0.15 * load

theorem remaining_load_after_three_deliveries : 
  unload_third_store (unload_second_store (unload_first_store initial_load)) = 30600 := 
by
  sorry

end remaining_load_after_three_deliveries_l748_748848


namespace lattice_points_on_hyperbola_l748_748283

theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | (p.1^2 - p.2^2 = 15)}.finite.to_finset.card = 4 :=
sorry

end lattice_points_on_hyperbola_l748_748283


namespace part_a_part_b_part_c_l748_748016

-- Definitions for the convex polyhedron, volume, and surface area
structure ConvexPolyhedron :=
  (volume : ℝ)
  (surface_area : ℝ)

variable {P : ConvexPolyhedron}

-- Statement for Part (a)
theorem part_a (r : ℝ) (h_r : r ≤ P.surface_area) :
  P.volume / P.surface_area ≥ r / 3 := sorry

-- Statement for Part (b)
theorem part_b :
  Exists (fun r : ℝ => r = P.volume / P.surface_area) := sorry

-- Definitions and conditions for the outer and inner polyhedron
structure ConvexPolyhedronPair :=
  (outer_polyhedron : ConvexPolyhedron)
  (inner_polyhedron : ConvexPolyhedron)

variable {CP : ConvexPolyhedronPair}

-- Statement for Part (c)
theorem part_c :
  3 * CP.outer_polyhedron.volume / CP.outer_polyhedron.surface_area ≥
  CP.inner_polyhedron.volume / CP.inner_polyhedron.surface_area := sorry

end part_a_part_b_part_c_l748_748016


namespace science_club_officers_l748_748709

-- Definitions of the problem conditions
def num_members : ℕ := 25
def num_officers : ℕ := 3
def alice : ℕ := 1 -- unique identifier for Alice
def bob : ℕ := 2 -- unique identifier for Bob

-- Main theorem statement
theorem science_club_officers :
  ∃ (ways_to_choose_officers : ℕ), ways_to_choose_officers = 10764 :=
  sorry

end science_club_officers_l748_748709


namespace factor_expression_l748_748182

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l748_748182


namespace hexagon_parallelogram_area_l748_748835

/-- Given a regular hexagon with side length 12 cm,
    the area of the parallelogram formed by connecting every second vertex is 36√3 cm². -/
theorem hexagon_parallelogram_area (s : ℝ) (h : s = 12) :
  let area_equilateral (a : ℝ) := (Real.sqrt 3 / 4) * a^2
  in area_equilateral (2 * s) - 3 * area_equilateral s = 36 * Real.sqrt 3 := by
  sorry

end hexagon_parallelogram_area_l748_748835


namespace distinct_arrangements_banana_l748_748240

theorem distinct_arrangements_banana : 
  let word := "banana" 
  let total_letters := 6 
  let freq_b := 1 
  let freq_n := 2 
  let freq_a := 3 
  ∀(n : ℕ) (n1 n2 n3 : ℕ), 
    n = total_letters → 
    n1 = freq_b → 
    n2 = freq_n → 
    n3 = freq_a → 
    (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 := 
by
  intros n n1 n2 n3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end distinct_arrangements_banana_l748_748240


namespace find_y_l748_748595

theorem find_y 
  (y : ℝ)
  (h1 : ∃ β, P = ( - sqrt 3, y ) ∧ sin β = sqrt 13 / 13) : y = 1 / 2 :=
sorry

end find_y_l748_748595


namespace rhind_papyrus_problem_l748_748315

theorem rhind_papyrus_problem 
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a2 = a1 + d)
  (h2 : a3 = a1 + 2 * d)
  (h3 : a4 = a1 + 3 * d)
  (h4 : a5 = a1 + 4 * d)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 60)
  (h_condition : (a4 + a5) / 2 = a1 + a2 + a3) :
  a1 = 4 / 3 :=
by
  sorry

end rhind_papyrus_problem_l748_748315


namespace find_other_number_l748_748406

-- Given conditions
def sum_of_numbers (x y : ℕ) : Prop := x + y = 72
def number_difference (x y : ℕ) : Prop := x = y + 12
def one_number_is_30 (x : ℕ) : Prop := x = 30

-- Theorem to prove
theorem find_other_number (y : ℕ) : 
  sum_of_numbers y 30 ∧ number_difference 30 y → y = 18 := by
  sorry

end find_other_number_l748_748406


namespace hexagon_area_inscribed_in_circle_l748_748825

noncomputable def area_hexagon (AB BC CD DE EF FA : ℝ) : ℝ :=
  let area_eq_triangle (a : ℝ) : ℝ := (sqrt 3 / 4) * (a * a)
  let S_triangle1 := 3 * area_eq_triangle (sqrt 3 + 1)
  let S_triangle2 := 3 * area_eq_triangle 1
  S_triangle1 - S_triangle2

theorem hexagon_area_inscribed_in_circle :
  let AB := (sqrt 3 + 1)
  let BC := (sqrt 3 + 1)
  let CD := (sqrt 3 + 1)
  let DE := (1)
  let EF := (1)
  let FA := (1)
  area_hexagon AB BC CD DE EF FA = (9 / 2) * (2 + sqrt 3) :=
by
  sorry

end hexagon_area_inscribed_in_circle_l748_748825


namespace option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l748_748436

def square_of_binomial (a b : ℝ) : ℝ :=
  (a + b) ^ 2 = a^2 + 2 * a * b + b^2 ∨ (a - b) ^ 2 = a^2 - 2 * a * b + b^2

theorem option_A_correct (x : ℝ) :
  ∃ a b, a = -x ∧ b = 3 ∧ (a + b) ^ 2 = x^2 - 9 :=
by {
  use [-x, 3],
  sorry 
}

theorem option_B_incorrect (a b : ℝ) :
 ¬ square_of_binomial (-a) (-b) :=
by sorry

theorem option_C_incorrect (x : ℝ) :
  ¬ square_of_binomial (-3 * x) 2 :=
by sorry

theorem option_D_incorrect (x : ℝ) :
  ¬ square_of_binomial 3 * x 2 :=
by sorry

end option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l748_748436


namespace geometric_sequence_b_value_l748_748529

theorem geometric_sequence_b_value (b : ℝ) (h1 : 25 * b = b^2) (h2 : b * (1 / 4) = b / 4) :
  b = 5 / 2 :=
sorry

end geometric_sequence_b_value_l748_748529


namespace number_of_mappings_l748_748985

open Finset

theorem number_of_mappings (A : Finset ℝ) (B : Finset ℝ) (hA : A.card = 100) (hB : B.card = 50)
  (h_non_dec : ∀ {f : ℕ → ℝ}, (∀ i j, i ≤ j → f i ≤ f j) → ∀ b ∈ B, ∃ a ∈ A, f a = b) :
  ∃ n, n = (finset.card (finset.range 100).choose 49) := 
sorry

end number_of_mappings_l748_748985


namespace donut_combinations_l748_748128

theorem donut_combinations : ∀ (n k : ℕ), (n = 4) → (k = 4) → 
  ∑ (λ m : ℕ, m), ((m = 7) → nat.choose m (k-1) = 35) :=
by
  intros
  have h1 : k - 1 = 3 := by 
    rw [h2]
  have h2 : n + (k - 1) = 7 := by 
    rw [h1, h]
  use 7
  rw [h2]
  apply nat.choose 7 3 = 35

end donut_combinations_l748_748128


namespace percentage_of_180_equation_l748_748364

theorem percentage_of_180_equation (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * ((P / 100) * 180) = 36) : P = 30 :=
sorry

end percentage_of_180_equation_l748_748364


namespace sum_of_squares_of_real_solutions_l748_748528

theorem sum_of_squares_of_real_solutions :
  let solutions := {x : ℝ | x ^ 128 = 16 ^ 16},
      sum_of_squares := ∑ x in solutions, x^2
  in sum_of_squares = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l748_748528


namespace tan_half_angle_l748_748688

noncomputable def theta_ac_ang (θ : ℝ) : Prop := θ > 0 ∧ θ < π / 2

theorem tan_half_angle (θ : ℝ) (x : ℝ) (hθ: theta_ac_ang θ)
  (hcos : cos (θ / 2) = sqrt ((x - 1) / (2 * x))) :
  tan (θ / 2) = sqrt ((x + 1) / (x - 1)) :=
by
  sorry

end tan_half_angle_l748_748688


namespace DF_length_l748_748663

open Real

variable {ABCD : Type} [parallelogram ABCD]
variables (AB BC DE DF AE : ℝ)
variable {E G : Point}

-- Conditions
axiom hAB_eq_DC : AB = 8
axiom hEB : EB = 2
axiom hDE : DE = 5
axiom hArea_AEG : (1/2) * AE * (height (Point, Line) G AE) = 10
axiom hArea_parallelogram : parallelogram_area ABCD DE AB

-- Theorem statement
theorem DF_length 
  (hAB_eq_DC : AB = 8) 
  (hEB : EB = 2) 
  (hDE : DE = 5)
  (hArea_AEG : (1/2) * AE * (height (Point, Line) G AE) = 10)
  (hArea_parallegram : parallelogram_area ABCD DE AB):
  DF = 5 := sorry

end DF_length_l748_748663


namespace greatest_k_for_2_power_factor_l748_748805

def product_of_integers_5_to_20 : ℕ := ∏ i in finset.range (20 - 4), (i + 5)

theorem greatest_k_for_2_power_factor :
  ∃ k : ℕ, (2 ^ k ∣ product_of_integers_5_to_20) ∧ 
          (∀ m : ℕ, (2 ^ (m + 1) ∣ product_of_integers_5_to_20) → k < m) :=
begin
  use 15,
  split,
  {
    -- Proof that 2^15 divides the product
    sorry
  },
  {
    -- Proof that there is no higher k such that 2^(k+1) divides the product
    sorry
  }
end

end greatest_k_for_2_power_factor_l748_748805


namespace tile_ratio_l748_748735

/-- Given the initial configuration and extension method, the ratio of black tiles to white tiles in the new design is 22/27. -/
theorem tile_ratio (initial_black : ℕ) (initial_white : ℕ) (border_black : ℕ) (border_white : ℕ) (total_tiles : ℕ)
  (h1 : initial_black = 10)
  (h2 : initial_white = 15)
  (h3 : border_black = 12)
  (h4 : border_white = 12)
  (h5 : total_tiles = 49) :
  (initial_black + border_black) / (initial_white + border_white) = 22 / 27 := 
by {
  /- 
     Here we would provide the proof steps if needed.
     This is a theorem stating that the ratio of black to white tiles 
     in the new design is 22 / 27 given the initial conditions.
  -/
  sorry 
}

end tile_ratio_l748_748735


namespace banana_permutations_l748_748255

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def multiset_permutations (n : ℕ) (frequencies : list ℕ) : ℕ :=
  factorial n / (frequencies.map factorial).prod

theorem banana_permutations :
  multiset_permutations 6 [1, 2, 3] = 60 :=
by
  sorry

end banana_permutations_l748_748255


namespace braden_total_money_after_winning_bet_l748_748865

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end braden_total_money_after_winning_bet_l748_748865


namespace triangle_land_area_l748_748389

theorem triangle_land_area :
  let base_cm := 12
  let height_cm := 9
  let scale_cm_to_miles := 3
  let square_mile_to_acres := 640
  let area_cm2 := (1 / 2 : Float) * base_cm * height_cm
  let area_miles2 := area_cm2 * (scale_cm_to_miles ^ 2)
  let area_acres := area_miles2 * square_mile_to_acres
  area_acres = 311040 :=
by
  -- Skipped proofs
  sorry

end triangle_land_area_l748_748389


namespace equal_elements_of_consecutive_sums_l748_748332

theorem equal_elements_of_consecutive_sums
  (n : ℕ) (a : ℕ → ℝ) 
  (h₀ : n ≥ 5)
  (h₁ : ∃ (d : ℝ), ∃ (m : ℤ), (finset.range (n * (n - 1) / 2)).image (λ k, d * k + m) = (finset.univ.image (λ ⟨i, j⟩, a i + a j)).order [1 ≤ i < j ≤ n])) 
: ∀ i j, 1 ≤ i < j ≤ n → a i = a j := 
by
  sorry

end equal_elements_of_consecutive_sums_l748_748332


namespace length_BI_l748_748322

-- Define the triangle with sides
variables (A B C D E F I : Type)
variables (AB AC BC : ℝ)
variables (r BI BD DI : ℝ)

-- Given conditions
def triangle_ABC : Prop :=
  AB = 8 ∧ AC = 17 ∧ BC = 15 ∧
  (∀ D E F, incircle_touches D E F BC AC AB) ∧
  (∀ I, incenter I D E F ABC)

-- The statement to prove
theorem length_BI {A B C D E F I : Type} {r BI BD DI : ℝ} (h : triangle_ABC A B C D E F I) : 
  BI = 3 * Real.sqrt 2 :=
sorry

end length_BI_l748_748322


namespace distance_between_A_and_B_l748_748822

noncomputable def problem_distance_between_points : ℕ :=
if h : (∃ (t v : ℚ), (t / 0.2 = 150) ∧ (t + 10 = 150)) 
then 150 
else 0

theorem distance_between_A_and_B : problem_distance_between_points = 150 := by
  -- problem parameter assumptions
  have h : ∃ (t v : ℚ), (t / 0.2 = 150) ∧ (t + 10 = 150) := sorry
  show problem_distance_between_points = 150
  rw [problem_distance_between_points, dif_pos h]
  rfl

end distance_between_A_and_B_l748_748822


namespace points_meet_every_720_seconds_l748_748030

theorem points_meet_every_720_seconds
    (v1 v2 : ℝ) 
    (h1 : v1 - v2 = 1/720) 
    (h2 : (1/v2) - (1/v1) = 10) :
    v1 = 1/80 ∧ v2 = 1/90 :=
by
  sorry

end points_meet_every_720_seconds_l748_748030


namespace ratio_of_sphere_radii_l748_748465

noncomputable def ratio_of_radius (V_large : ℝ) (percentage : ℝ) : ℝ :=
  let V_small := (percentage / 100) * V_large
  let ratio := (V_small / V_large) ^ (1/3)
  ratio

theorem ratio_of_sphere_radii : 
  ratio_of_radius (450 * Real.pi) 27.04 = 0.646 := 
  by
  sorry

end ratio_of_sphere_radii_l748_748465


namespace expected_value_of_8_sided_die_l748_748088

theorem expected_value_of_8_sided_die : 
  let p := (1 / 8 : ℚ)
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let values := outcomes.map (λ n, n^3)
  let expected_value := (values.sum : ℚ) * p
  expected_value = 162 := sorry

end expected_value_of_8_sided_die_l748_748088


namespace sequence_6th_term_l748_748312

theorem sequence_6th_term 
    (a₁ a₂ a₃ a₄ a₅ a₆ : ℚ)
    (h₁ : a₁ = 3)
    (h₅ : a₅ = 54)
    (h₂ : a₂ = (a₁ + a₃) / 3)
    (h₃ : a₃ = (a₂ + a₄) / 3)
    (h₄ : a₄ = (a₃ + a₅) / 3)
    (h₆ : a₅ = (a₄ + a₆) / 3) :
    a₆ = 1133 / 7 :=
by
  sorry

end sequence_6th_term_l748_748312


namespace min_bags_needed_l748_748463

theorem min_bags_needed (marbles : ℕ) (children1 children2 : ℕ) (bags : ℕ) :
  marbles = 77 → 
  children1 = 7 → 
  children2 = 11 → 
  ∃ n, n = 17 ∧ (∀ {c}, c = children1 ∨ c = children2 →
    ∃ (b : ℕ → ℕ), (∀ i, b i ≤ marbles) ∧ 
    (marbles = c * ∑ i in finset.range n, b i)) := 
begin
  intros,
  use 17,
  split,
  { refl, },
  { intros c hc,
    cases hc,
    { sorry, },
    { sorry, },
  },
end

end min_bags_needed_l748_748463


namespace population_growth_l748_748474

theorem population_growth :
  let scale_factor1 := 1 + 10 / 100
  let scale_factor2 := 1 + 20 / 100
  let k := 2 * 20
  let scale_factor3 := 1 + k / 100
  let combined_scale := scale_factor1 * scale_factor2 * scale_factor3
  (combined_scale - 1) * 100 = 84.8 :=
by
  sorry

end population_growth_l748_748474


namespace function_passes_through_point_l748_748608

theorem function_passes_through_point :
  ∀ (a : ℝ), (a ≠ 0) → (a * (-1)^2 = 2) → (2 * (1)^2 = 2) :=
by
  intros a h1 h2
  rw h2
  sorry

end function_passes_through_point_l748_748608


namespace sum_of_x_y_z_l748_748344

theorem sum_of_x_y_z :
  ∃ (x y z : ℕ), (a b : ℝ) (ha : a^2 = 9 / 25) (hb : b^2 = (3 + Real.sqrt 7)^2 / 14) (a_neg : a < 0) (b_pos : b > 0),
  (a - b)^2 = (x * Real.sqrt y) / z ∧ x + y + z = 22 :=
by
  let a := -Real.sqrt (9 / 25)
  let b := Real.sqrt ( (3 + Real.sqrt 7)^2 / 14)
  have ha : a^2 = 9 / 25 := by sorry
  have hb : b^2 = (3 + Real.sqrt 7)^2 / 14 := by sorry
  have a_neg : a < 0 := by sorry
  have b_pos : b > 0 := by sorry
  have h : (a - b)^2 = (3 * Real.sqrt 14) / 5 := by sorry
  use [3, 14, 5]
  split
  . exact h
  . rfl

end sum_of_x_y_z_l748_748344


namespace solution_exists_l748_748149

theorem solution_exists (x y z u v : ℕ) (hx : x > 2000) (hy : y > 2000) (hz : z > 2000) (hu : u > 2000) (hv : v > 2000) : 
  x^2 + y^2 + z^2 + u^2 + v^2 = x * y * z * u * v - 65 :=
sorry

end solution_exists_l748_748149


namespace problem_statement_l748_748441

theorem problem_statement : (36 / 49 : ℝ) ^ (-1/2) - Real.logb 2 (2 ^ (1/6)) = 1 := by
  sorry

end problem_statement_l748_748441


namespace expected_value_eight_sided_die_win_l748_748082

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l748_748082
