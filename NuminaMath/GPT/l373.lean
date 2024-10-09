import Mathlib

namespace investment_worth_l373_37370

theorem investment_worth {x : ℝ} (x_pos : 0 < x) :
  ∀ (initial_investment final_value : ℝ) (years : ℕ),
  (initial_investment * 3^years = final_value) → 
  initial_investment = 1500 → final_value = 13500 → 
  8 = x → years = 2 →
  years * (112 / x) = 28 := 
by
  sorry

end investment_worth_l373_37370


namespace rectangle_perimeter_gt_16_l373_37382

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l373_37382


namespace inequality_and_equality_condition_l373_37330

theorem inequality_and_equality_condition (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) :=
  sorry

end inequality_and_equality_condition_l373_37330


namespace range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l373_37365

-- Define the propositions p and q
def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + 2 + m = 0

def q (m : ℝ) : Prop :=
  1 - 2 * m < 0 ∧ m + 2 > 0 ∨ 1 - 2 * m > 0 ∧ m + 2 < 0 -- Hyperbola condition

-- Prove the ranges of m
theorem range_m_for_p {m : ℝ} (hp : p m) : m ≤ -2 ∨ m ≥ 1 :=
sorry

theorem range_m_for_q {m : ℝ} (hq : q m) : m < -2 ∨ m > (1 / 2) :=
sorry

theorem range_m_for_not_p_or_q {m : ℝ} (h_not_p : ¬ (p m)) (h_not_q : ¬ (q m)) : -2 < m ∧ m ≤ (1 / 2) :=
sorry

end range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l373_37365


namespace magnitude_of_z_l373_37353

open Complex

theorem magnitude_of_z :
  ∃ z : ℂ, (1 + 2 * Complex.I) * z = -1 + 3 * Complex.I ∧ Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l373_37353


namespace find_x_l373_37362

theorem find_x (x : ℝ) (h : (2 * x + 8 + 5 * x + 3 + 3 * x + 9) / 3 = 3 * x + 2) : x = -14 :=
by
  sorry

end find_x_l373_37362


namespace sum_first_n_terms_arithmetic_sequence_l373_37371

theorem sum_first_n_terms_arithmetic_sequence 
  (S : ℕ → ℕ) (m : ℕ) (h1 : S m = 2) (h2 : S (2 * m) = 10) :
  S (3 * m) = 24 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l373_37371


namespace determine_x_l373_37319

theorem determine_x
  (total_area : ℝ)
  (side_length_square1 : ℝ)
  (side_length_square2 : ℝ)
  (h1 : total_area = 1300)
  (h2 : side_length_square1 = 3 * x)
  (h3 : side_length_square2 = 7 * x) :
    x = Real.sqrt (2600 / 137) :=
by
  sorry

end determine_x_l373_37319


namespace arithmetic_seq_8th_term_l373_37368

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l373_37368


namespace Jina_has_51_mascots_l373_37306

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l373_37306


namespace negation_proposition_equivalence_l373_37359

theorem negation_proposition_equivalence :
  (¬ ∃ x₀ : ℝ, (2 / x₀ + Real.log x₀ ≤ 0)) ↔ (∀ x : ℝ, 2 / x + Real.log x > 0) := 
sorry

end negation_proposition_equivalence_l373_37359


namespace min_value_geometric_sequence_l373_37355

theorem min_value_geometric_sequence (a_2 a_3 : ℝ) (r : ℝ) 
(h_a2 : a_2 = 2 * r) (h_a3 : a_3 = 2 * r^2) : 
  (6 * a_2 + 7 * a_3) = -18 / 7 :=
by
  sorry

end min_value_geometric_sequence_l373_37355


namespace sector_area_l373_37342

theorem sector_area (r θ : ℝ) (hr : r = 2) (hθ : θ = (45 : ℝ) * (Real.pi / 180)) : 
  (1 / 2) * r^2 * θ = Real.pi / 2 := 
by
  sorry

end sector_area_l373_37342


namespace equality_of_fractions_l373_37367

theorem equality_of_fractions
  (a b c x y z : ℝ)
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  (a^2) / (1 - x^2) = (b^2) / (1 - y^2) ∧ (b^2) / (1 - y^2) = (c^2) / (1 - z^2) :=
by
  sorry

end equality_of_fractions_l373_37367


namespace problem1_problem2_l373_37323

-- Definitions
variables {a b z : ℝ}

-- Problem 1 translated to Lean
theorem problem1 (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) : -2 < a ∧ a < 1 := 
sorry

-- Problem 2 translated to Lean
theorem problem2 (h1 : a + 2 * b = 9) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  ∃ z : ℝ, z = a * b^2 ∧ ∀ w : ℝ, (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 2 * b = 9 ∧ w = a * b^2) → w ≤ 27 :=
sorry

end problem1_problem2_l373_37323


namespace inv_geom_seq_prod_next_geom_seq_l373_37318

variable {a : Nat → ℝ} (q : ℝ) (h_q : q ≠ 0)
variable (h_geom : ∀ n, a (n + 1) = q * a n)

theorem inv_geom_seq :
  ∀ n, ∃ c q_inv, (q_inv ≠ 0) ∧ (1 / a n = c * q_inv ^ n) :=
sorry

theorem prod_next_geom_seq :
  ∀ n, ∃ c q_sq, (q_sq ≠ 0) ∧ (a n * a (n + 1) = c * q_sq ^ n) :=
sorry

end inv_geom_seq_prod_next_geom_seq_l373_37318


namespace probability_of_odd_product_is_zero_l373_37394

-- Define the spinners
def spinnerC : List ℕ := [1, 3, 5, 7]
def spinnerD : List ℕ := [2, 4, 6]

-- Define the condition that the odds and evens have a specific product property
axiom odd_times_even_is_even {a b : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 0) : (a * b) % 2 = 0

-- Define the probability of getting an odd product
noncomputable def probability_odd_product : ℕ :=
  if ∃ a ∈ spinnerC, ∃ b ∈ spinnerD, (a * b) % 2 = 1 then 1 else 0

-- Main theorem
theorem probability_of_odd_product_is_zero : probability_odd_product = 0 := by
  sorry

end probability_of_odd_product_is_zero_l373_37394


namespace william_library_visits_l373_37324

variable (W : ℕ) (J : ℕ)
variable (h1 : J = 4 * W)
variable (h2 : 4 * J = 32)

theorem william_library_visits : W = 2 :=
by
  sorry

end william_library_visits_l373_37324


namespace ratio_of_Frederick_to_Tyson_l373_37396

-- Definitions of the ages based on given conditions
def Kyle : Nat := 25
def Tyson : Nat := 20
def Julian : Nat := Kyle - 5
def Frederick : Nat := Julian + 20

-- The ratio of Frederick's age to Tyson's age
def ratio : Nat × Nat := (Frederick / Nat.gcd Frederick Tyson, Tyson / Nat.gcd Frederick Tyson)

-- Proving the ratio is 2:1
theorem ratio_of_Frederick_to_Tyson : ratio = (2, 1) := by
  sorry

end ratio_of_Frederick_to_Tyson_l373_37396


namespace gum_sharing_l373_37312

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l373_37312


namespace cristina_running_pace_l373_37322

theorem cristina_running_pace
  (nicky_pace : ℝ) (nicky_headstart : ℝ) (time_nicky_run : ℝ) 
  (distance_nicky_run : ℝ) (time_cristina_catch : ℝ) :
  (nicky_pace = 3) →
  (nicky_headstart = 12) →
  (time_nicky_run = 30) →
  (distance_nicky_run = nicky_pace * time_nicky_run) →
  (time_cristina_catch = time_nicky_run - nicky_headstart) →
  (cristina_pace : ℝ) →
  (cristina_pace = distance_nicky_run / time_cristina_catch) →
  cristina_pace = 5 :=
by
  sorry

end cristina_running_pace_l373_37322


namespace Sean_Julie_ratio_l373_37350

-- Define the sum of the first n natural numbers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of even numbers up to 2n
def sum_even (n : ℕ) : ℕ := 2 * sum_n n

theorem Sean_Julie_ratio : 
  (sum_even 250) / (sum_n 250) = 2 := 
by
  sorry

end Sean_Julie_ratio_l373_37350


namespace plane_intersects_unit_cubes_l373_37309

-- Definitions:
def isLargeCube (cube : ℕ × ℕ × ℕ) : Prop := cube = (4, 4, 4)
def isUnitCube (size : ℕ) : Prop := size = 1

-- The main theorem we want to prove:
theorem plane_intersects_unit_cubes :
  ∀ (cube : ℕ × ℕ × ℕ) (plane : (ℝ × ℝ × ℝ) → ℝ),
  isLargeCube cube →
  (∀ point : ℝ × ℝ × ℝ, plane point = 0 → 
       ∃ (x y z : ℕ), x < 4 ∧ y < 4 ∧ z < 4 ∧ 
                     (x, y, z) ∈ { coords : ℕ × ℕ × ℕ | true }) →
  (∃ intersects : ℕ, intersects = 16) :=
by
  intros cube plane Hcube Hplane
  sorry

end plane_intersects_unit_cubes_l373_37309


namespace distance_midpoint_to_origin_l373_37395

variables {a b c d m k l n : ℝ}

theorem distance_midpoint_to_origin (h1 : b = m * a + k) (h2 : d = m * c + k) (h3 : n = -1 / m) :
  dist (0, 0) ( ((a + c) / 2), ((m * (a + c) + 2 * k) / 2) ) = (1 / 2) * Real.sqrt ((1 + m^2) * (a + c)^2 + 4 * k^2 + 4 * m * (a + c) * k) :=
by
  sorry

end distance_midpoint_to_origin_l373_37395


namespace find_x_plus_inv_x_l373_37380

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l373_37380


namespace modulus_of_2_plus_i_over_1_plus_2i_l373_37378

open Complex

noncomputable def modulus_of_complex_fraction : ℂ := 
  let z : ℂ := (2 + I) / (1 + 2 * I)
  abs z

theorem modulus_of_2_plus_i_over_1_plus_2i :
  modulus_of_complex_fraction = 1 := by
  sorry

end modulus_of_2_plus_i_over_1_plus_2i_l373_37378


namespace Lakota_spent_l373_37377

-- Define the conditions
def U : ℝ := 9.99
def Mackenzies_cost (N : ℝ) : ℝ := 3 * N + 8 * U
def cost_of_Lakotas_disks (N : ℝ) : ℝ := 6 * N + 2 * U

-- State the theorem
theorem Lakota_spent (N : ℝ) (h : Mackenzies_cost N = 133.89) : cost_of_Lakotas_disks N = 127.92 :=
by
  sorry

end Lakota_spent_l373_37377


namespace triploid_fruit_fly_chromosome_periodicity_l373_37334

-- Define the conditions
def normal_chromosome_count (organism: Type) : ℕ := 8
def triploid_fruit_fly (organism: Type) : Prop := true
def XXY_sex_chromosome_composition (organism: Type) : Prop := true
def periodic_change (counts: List ℕ) : Prop := counts = [9, 18, 9]

-- State the theorem
theorem triploid_fruit_fly_chromosome_periodicity (organism: Type)
  (h1: triploid_fruit_fly organism) 
  (h2: XXY_sex_chromosome_composition organism)
  (h3: normal_chromosome_count organism = 8) : 
  periodic_change [9, 18, 9] :=
sorry

end triploid_fruit_fly_chromosome_periodicity_l373_37334


namespace pow_mod_sub_l373_37337

theorem pow_mod_sub (a b : ℕ) (n : ℕ) (h1 : a ≡ 5 [MOD 6]) (h2 : b ≡ 4 [MOD 6]) : (a^n - b^n) % 6 = 1 :=
by
  let a := 47
  let b := 22
  let n := 1987
  sorry

end pow_mod_sub_l373_37337


namespace line_does_not_pass_through_second_quadrant_l373_37340
-- Import the Mathlib library

-- Define the properties of the line
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the condition for a point to be in the second quadrant:
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the proof statement
theorem line_does_not_pass_through_second_quadrant:
  ∀ x y : ℝ, line_eq x y → ¬ in_second_quadrant x y :=
by
  sorry

end line_does_not_pass_through_second_quadrant_l373_37340


namespace weight_of_8_moles_of_AlI3_l373_37304

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_I : ℝ := 126.90
noncomputable def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I

theorem weight_of_8_moles_of_AlI3 : 
  (8 * molecular_weight_AlI3) = 3261.44 := by
sorry

end weight_of_8_moles_of_AlI3_l373_37304


namespace number_of_divisions_l373_37358

-- Definitions
def hour_in_seconds : ℕ := 3600

def is_division (n m : ℕ) : Prop :=
  n * m = hour_in_seconds ∧ n > 0 ∧ m > 0

-- Proof problem statement
theorem number_of_divisions : ∃ (count : ℕ), count = 44 ∧ 
  (∀ (n m : ℕ), is_division n m → ∃ (d : ℕ), d = count) :=
sorry

end number_of_divisions_l373_37358


namespace exchange_rate_l373_37379

theorem exchange_rate (a b : ℕ) (h : 5000 = 60 * a) : b = 75 * a → b = 6250 := by
  sorry

end exchange_rate_l373_37379


namespace square_plot_area_l373_37315

theorem square_plot_area
  (cost_per_foot : ℕ)
  (total_cost : ℕ)
  (s : ℕ)
  (area : ℕ)
  (h1 : cost_per_foot = 55)
  (h3 : total_cost = 3740)
  (h4 : total_cost = 4 * s * cost_per_foot)
  (h5 : area = s * s) :
  area = 289 := sorry

end square_plot_area_l373_37315


namespace angle_sum_triangle_l373_37335

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end angle_sum_triangle_l373_37335


namespace existence_of_xyz_l373_37325

theorem existence_of_xyz (n : ℕ) (hn_pos : 0 < n)
    (a b c : ℕ) (ha : 0 < a ∧ a ≤ 3 * n^2 + 4 * n) 
                (hb : 0 < b ∧ b ≤ 3 * n^2 + 4 * n) 
                (hc : 0 < c ∧ c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), (|x| ≤ 2 * n) ∧ (|y| ≤ 2 * n) ∧ (|z| ≤ 2 * n) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
  sorry

end existence_of_xyz_l373_37325


namespace hours_week3_and_4_l373_37393

variable (H3 H4 : Nat)

def hours_worked_week1_and_2 : Nat := 35 + 35
def extra_hours_worked_week3_and_4 : Nat := 26
def total_hours_week3_and_4 : Nat := hours_worked_week1_and_2 + extra_hours_worked_week3_and_4

theorem hours_week3_and_4 :
  H3 + H4 = total_hours_week3_and_4 := by
sorry

end hours_week3_and_4_l373_37393


namespace alice_is_10_years_older_l373_37332

-- Problem definitions
variables (A B : ℕ)

-- Conditions of the problem
def condition1 := A + 5 = 19
def condition2 := A + 6 = 2 * (B + 6)

-- Question to prove
theorem alice_is_10_years_older (h1 : condition1 A) (h2 : condition2 A B) : A - B = 10 := 
by
  sorry

end alice_is_10_years_older_l373_37332


namespace product_of_geometric_progressions_is_geometric_general_function_form_geometric_l373_37388

variables {α β γ : Type*} [CommSemiring α] [CommSemiring β] [CommSemiring γ]

-- Define the terms of geometric progressions
def term (a r : α) (k : ℕ) : α := a * r ^ (k - 1)

-- Define a general function with respective powers
def general_term (a r : α) (k p : ℕ) : α := a ^ p * (r ^ p) ^ (k - 1)

theorem product_of_geometric_progressions_is_geometric
  {a b c : α} {r1 r2 r3 : α} (k : ℕ) :
  term a r1 k * term b r2 k * term c r3 k = 
  (a * b * c) * (r1 * r2 * r3) ^ (k - 1) := 
sorry

theorem general_function_form_geometric
  {a b c : α} {r1 r2 r3 : α} {p q r : ℕ} (k : ℕ) :
  general_term a r1 k p * general_term b r2 k q * general_term c r3 k r = 
  (a^p * b^q * c^r) * (r1^p * r2^q * r3^r) ^ (k - 1) := 
sorry

end product_of_geometric_progressions_is_geometric_general_function_form_geometric_l373_37388


namespace seven_circle_divisors_exists_non_adjacent_divisors_l373_37397

theorem seven_circle_divisors_exists_non_adjacent_divisors (a : Fin 7 → ℕ)
  (h_adj : ∀ i : Fin 7, a i ∣ a (i + 1) % 7 ∨ a (i + 1) % 7 ∣ a i) :
  ∃ (i j : Fin 7), i ≠ j ∧ j ≠ i + 1 % 7 ∧ j ≠ i + 6 % 7 ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end seven_circle_divisors_exists_non_adjacent_divisors_l373_37397


namespace sufficient_but_not_necessary_l373_37363

theorem sufficient_but_not_necessary (x y : ℝ) (h : x ≥ 1 ∧ y ≥ 1) : x ^ 2 + y ^ 2 ≥ 2 ∧ ∃ (x y : ℝ), x ^ 2 + y ^ 2 ≥ 2 ∧ (¬ (x ≥ 1 ∧ y ≥ 1)) :=
by
  sorry

end sufficient_but_not_necessary_l373_37363


namespace max_value_of_f_l373_37338

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∀ x : ℝ, x > 0 → f x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) :=
by
  sorry

end max_value_of_f_l373_37338


namespace man_speed_in_still_water_l373_37339

theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) (h1 : upstream_speed = 25) (h2 : downstream_speed = 45) :
  (upstream_speed + downstream_speed) / 2 = 35 :=
by
  sorry

end man_speed_in_still_water_l373_37339


namespace single_elimination_games_l373_37392

theorem single_elimination_games (n : Nat) (h : n = 21) : games_needed = n - 1 :=
by
  sorry

end single_elimination_games_l373_37392


namespace odd_tiling_numbers_l373_37301

def f (n k : ℕ) : ℕ := sorry -- Assume f(n, 2k) is defined appropriately.

theorem odd_tiling_numbers (n : ℕ) : (∀ k : ℕ, f n (2*k) % 2 = 1) ↔ ∃ i : ℕ, n = 2^i - 1 := sorry

end odd_tiling_numbers_l373_37301


namespace train_carriages_l373_37384

theorem train_carriages (num_trains : ℕ) (total_wheels : ℕ) (rows_per_carriage : ℕ) 
  (wheels_per_row : ℕ) (carriages_per_train : ℕ) :
  num_trains = 4 →
  total_wheels = 240 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  carriages_per_train = 
    (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains →
  carriages_per_train = 4 :=
by
  sorry

end train_carriages_l373_37384


namespace no_five_consecutive_terms_divisible_by_2005_l373_37383

noncomputable def a (n : ℕ) : ℤ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_terms_divisible_by_2005 : ¬ ∃ n : ℕ, (a n % 2005 = 0) ∧ (a (n+1) % 2005 = 0) ∧ (a (n+2) % 2005 = 0) ∧ (a (n+3) % 2005 = 0) ∧ (a (n+4) % 2005 = 0) := sorry

end no_five_consecutive_terms_divisible_by_2005_l373_37383


namespace solve_fraction_equation_l373_37311

theorem solve_fraction_equation (x : ℚ) :
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 := 
by
  sorry

end solve_fraction_equation_l373_37311


namespace total_fish_count_l373_37341

theorem total_fish_count (kyle_caught_same_as_tasha : ∀ kyle tasha : ℕ, kyle = tasha) 
  (carla_caught : ℕ) (kyle_caught : ℕ) (tasha_caught : ℕ)
  (h0 : carla_caught = 8) (h1 : kyle_caught = 14) (h2 : tasha_caught = kyle_caught) : 
  8 + 14 + 14 = 36 :=
by sorry

end total_fish_count_l373_37341


namespace range_of_x_l373_37398

theorem range_of_x (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ {a b : ℝ}, a ≤ b → b ≤ 0 → f a ≤ f b) :
  (∀ x : ℝ, f (2^(2*x^2 - x - 1)) ≥ f (-4)) → ∀ x, x ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) :=
by 
  sorry

end range_of_x_l373_37398


namespace remainder_of_8_pow_2023_l373_37352

theorem remainder_of_8_pow_2023 :
  8^2023 % 100 = 12 :=
sorry

end remainder_of_8_pow_2023_l373_37352


namespace percent_increase_quarter_l373_37316

-- Define the profit changes over each month
def profit_march (P : ℝ) := P
def profit_april (P : ℝ) := 1.40 * P
def profit_may (P : ℝ) := 1.12 * P
def profit_june (P : ℝ) := 1.68 * P

-- Starting Lean theorem statement
theorem percent_increase_quarter (P : ℝ) (hP : P > 0) :
  ((profit_june P - profit_march P) / profit_march P) * 100 = 68 :=
  sorry

end percent_increase_quarter_l373_37316


namespace no_integer_roots_l373_37310

theorem no_integer_roots (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 :=
by {
  sorry
}

end no_integer_roots_l373_37310


namespace nutmeg_amount_l373_37305

def amount_of_cinnamon : ℝ := 0.6666666666666666
def difference_cinnamon_nutmeg : ℝ := 0.16666666666666666

theorem nutmeg_amount (x : ℝ) 
  (h1 : amount_of_cinnamon = x + difference_cinnamon_nutmeg) : 
  x = 0.5 :=
by 
  sorry

end nutmeg_amount_l373_37305


namespace number_of_self_inverse_subsets_is_15_l373_37366

-- Define the set M
def M : Set ℚ := ({-1, 0, 1/2, 1/3, 1, 2, 3, 4} : Set ℚ)

-- Definition of self-inverse set
def is_self_inverse (A : Set ℚ) : Prop := ∀ x ∈ A, 1/x ∈ A

-- Theorem stating the number of non-empty self-inverse subsets of M
theorem number_of_self_inverse_subsets_is_15 :
  (∃ S : Finset (Set ℚ), S.card = 15 ∧ ∀ A ∈ S, A ⊆ M ∧ is_self_inverse A) :=
sorry

end number_of_self_inverse_subsets_is_15_l373_37366


namespace sqrt_operations_correctness_l373_37313

open Real

theorem sqrt_operations_correctness :
  (sqrt 2 + sqrt 3 ≠ sqrt 5) ∧
  (sqrt (2/3) * sqrt 6 = 2) ∧
  (sqrt 9 = 3) ∧
  (sqrt ((-6) ^ 2) = 6) :=
by
  sorry

end sqrt_operations_correctness_l373_37313


namespace no_solution_for_triples_l373_37317

theorem no_solution_for_triples :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 66) ∧ (a * c + b * c = 35) :=
by {
  sorry
}

end no_solution_for_triples_l373_37317


namespace find_k_l373_37348

theorem find_k (k : ℝ) :
  (∃ x : ℝ, 8 * x - k = 2 * (x + 1) ∧ 2 * (2 * x - 3) = 1 - 3 * x) → k = 4 :=
by
  sorry

end find_k_l373_37348


namespace evaluate_expression_l373_37303

theorem evaluate_expression (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l373_37303


namespace inequality_N_value_l373_37344

theorem inequality_N_value (a c : ℝ) (ha : 0 < a) (hc : 0 < c) (b : ℝ) (hb : b = 2 * a) : 
  (a^2 + b^2) / c^2 > 5 / 9 := 
by sorry

end inequality_N_value_l373_37344


namespace temperature_on_friday_l373_37343

def temperatures (M T W Th F : ℝ) : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 40 ∧
  M = 42

theorem temperature_on_friday (M T W Th F : ℝ) (h : temperatures M T W Th F) : 
  F = 10 :=
  by
    -- problem statement
    sorry

end temperature_on_friday_l373_37343


namespace geese_in_marsh_l373_37364

theorem geese_in_marsh (D : ℝ) (hD : D = 37.0) (G : ℝ) (hG : G = D + 21) : G = 58.0 := 
by 
  sorry

end geese_in_marsh_l373_37364


namespace inequality_solution_set_l373_37399

noncomputable def solution_set := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem inequality_solution_set :
  (solution_set = {x : ℝ | x ≤ -3 ∨ x ≥ 1}) :=
sorry

end inequality_solution_set_l373_37399


namespace together_finish_work_in_10_days_l373_37391

theorem together_finish_work_in_10_days (x_days y_days : ℕ) (hx : x_days = 15) (hy : y_days = 30) :
  let x_rate := 1 / (x_days : ℚ)
  let y_rate := 1 / (y_days : ℚ)
  let combined_rate := x_rate + y_rate
  let total_days := 1 / combined_rate
  total_days = 10 :=
by
  sorry

end together_finish_work_in_10_days_l373_37391


namespace range_of_a_l373_37331

open Real

-- Definitions based on given conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → -3^x ≤ a

-- The main proposition combining the conditions
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l373_37331


namespace height_of_box_l373_37376

-- Definitions of given conditions
def length_box : ℕ := 9
def width_box : ℕ := 12
def num_cubes : ℕ := 108
def volume_cube : ℕ := 3
def volume_box : ℕ := num_cubes * volume_cube  -- Volume calculated from number of cubes and volume of each cube

-- The statement to prove
theorem height_of_box : 
  ∃ h : ℕ, volume_box = length_box * width_box * h ∧ h = 3 := by
  sorry

end height_of_box_l373_37376


namespace polynomial_coefficient_sum_l373_37357

theorem polynomial_coefficient_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2 * x - 3) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  sorry

end polynomial_coefficient_sum_l373_37357


namespace sum_base5_eq_l373_37373

theorem sum_base5_eq :
  (432 + 43 + 4 : ℕ) = 1034 :=
by sorry

end sum_base5_eq_l373_37373


namespace teacher_zhang_friends_l373_37346

-- Define the conditions
def num_students : ℕ := 50
def both_friends : ℕ := 30
def neither_friend : ℕ := 1
def diff_in_friends : ℕ := 7

-- Prove that Teacher Zhang has 43 friends on social media
theorem teacher_zhang_friends : ∃ x : ℕ, 
  x + (x - diff_in_friends) - both_friends + neither_friend = num_students ∧ x = 43 := 
by
  sorry

end teacher_zhang_friends_l373_37346


namespace bird_cages_count_l373_37308

/-- 
If each bird cage contains 2 parrots and 2 parakeets,
and the total number of birds is 36,
then the number of bird cages is 9.
-/
theorem bird_cages_count (parrots_per_cage parakeets_per_cage total_birds cages : ℕ)
  (h1 : parrots_per_cage = 2)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 36)
  (h4 : total_birds = (parrots_per_cage + parakeets_per_cage) * cages) :
  cages = 9 := 
by 
  sorry

end bird_cages_count_l373_37308


namespace total_marbles_proof_l373_37385

def dan_violet_marbles : Nat := 64
def mary_red_marbles : Nat := 14
def john_blue_marbles (x : Nat) : Nat := x

def total_marble (x : Nat) : Nat := dan_violet_marbles + mary_red_marbles + john_blue_marbles x

theorem total_marbles_proof (x : Nat) : total_marble x = 78 + x := by
  sorry

end total_marbles_proof_l373_37385


namespace question_1_question_2_l373_37314

noncomputable def f (x m : ℝ) : ℝ := abs (x + m) - abs (2 * x - 2 * m)

theorem question_1 (x : ℝ) (m : ℝ) (h : m = 1/2) (h_pos : m > 0) : 
  (f x m ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

theorem question_2 (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, ∃ t : ℝ, f x m + abs (t - 3) < abs (t + 4)) ↔ (0 < m ∧ m < 7/2) :=
sorry

end question_1_question_2_l373_37314


namespace increasing_interval_f_l373_37375

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_f :
  ∀ x, (2 < x) → (∃ ε > 0, ∀ δ > 0, δ < ε → f (x + δ) ≥ f x) :=
by
  sorry

end increasing_interval_f_l373_37375


namespace right_triangle_angles_l373_37328

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : 
  (α = 58.5 ∧ β = 31.5) :=
by sorry

end right_triangle_angles_l373_37328


namespace find_second_number_in_denominator_l373_37390

theorem find_second_number_in_denominator :
  (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 → x = 0.24847 :=
by
  intro h
  sorry

end find_second_number_in_denominator_l373_37390


namespace birdhouse_flown_distance_l373_37369

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l373_37369


namespace multiple_of_savings_l373_37354

theorem multiple_of_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1 / 4) * P
  let monthly_non_savings := (3 / 4) * P
  let total_yearly_savings := 12 * monthly_savings
  ∃ M : ℝ, total_yearly_savings = M * monthly_non_savings ∧ M = 4 := 
by
  sorry

end multiple_of_savings_l373_37354


namespace initial_number_l373_37356

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l373_37356


namespace range_of_m_l373_37302

theorem range_of_m (x m : ℝ) (h₀ : -2 ≤ x ∧ x ≤ 11)
  (h₁ : 1 - 3 * m ≤ x ∧ x ≤ 3 + m)
  (h₂ : ¬ (-2 ≤ x ∧ x ≤ 11) → ¬ (1 - 3 * m ≤ x ∧ x ≤ 3 + m)) :
  m ≥ 8 :=
by
  sorry

end range_of_m_l373_37302


namespace calories_difference_l373_37320

theorem calories_difference
  (calories_squirrel : ℕ := 300)
  (squirrels_per_hour : ℕ := 6)
  (calories_rabbit : ℕ := 800)
  (rabbits_per_hour : ℕ := 2) :
  ((squirrels_per_hour * calories_squirrel) - (rabbits_per_hour * calories_rabbit)) = 200 :=
by
  sorry

end calories_difference_l373_37320


namespace sum_first_five_terms_geometric_sequence_l373_37336

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end sum_first_five_terms_geometric_sequence_l373_37336


namespace man_speed_is_approximately_54_009_l373_37347

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ := 
  -- Convert distance to kilometers and time to hours
  let distance_km := d / 1000
  let time_hours := t / 3600
  distance_km / time_hours

theorem man_speed_is_approximately_54_009 :
  abs (speed_in_kmh 375.03 25 - 54.009) < 0.001 := 
by
  sorry

end man_speed_is_approximately_54_009_l373_37347


namespace slope_of_tangent_at_minus_1_l373_37300

theorem slope_of_tangent_at_minus_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2) * (x^2 + c))
  (h_extremum : deriv f 1 = 0) :
  deriv f (-1) = 8 :=
by
  sorry

end slope_of_tangent_at_minus_1_l373_37300


namespace annes_score_l373_37307

theorem annes_score (a b : ℕ) (h1 : a = b + 50) (h2 : (a + b) / 2 = 150) : a = 175 := 
by
  sorry

end annes_score_l373_37307


namespace number_of_girls_l373_37351

theorem number_of_girls (total_students : ℕ) (sample_size : ℕ) (girls_sampled_minus : ℕ) (girls_sampled_ratio : ℚ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_sampled_minus = 20 →
  girls_sampled_ratio = 90 / 200 →
  (∃ x, x / (total_students : ℚ) = girls_sampled_ratio ∧ x = 720) :=
by intros _ _ _ _; sorry

end number_of_girls_l373_37351


namespace mostWaterIntake_l373_37386

noncomputable def dailyWaterIntakeDongguk : ℝ := 5 * 0.2 -- Total water intake in liters per day for Dongguk
noncomputable def dailyWaterIntakeYoonji : ℝ := 6 * 0.3 -- Total water intake in liters per day for Yoonji
noncomputable def dailyWaterIntakeHeejin : ℝ := 4 * 500 / 1000 -- Total water intake in liters per day for Heejin (converted from milliliters)

theorem mostWaterIntake :
  dailyWaterIntakeHeejin = max dailyWaterIntakeDongguk (max dailyWaterIntakeYoonji dailyWaterIntakeHeejin) :=
by
  sorry

end mostWaterIntake_l373_37386


namespace orange_harvest_exists_l373_37329

theorem orange_harvest_exists :
  ∃ (A B C D : ℕ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ A + B + C + D = 56 :=
by
  use 10
  use 15
  use 16
  use 15
  repeat {split};
  sorry

end orange_harvest_exists_l373_37329


namespace charles_initial_bananas_l373_37387

theorem charles_initial_bananas (W C : ℕ) (h1 : W = 48) (h2 : C = C - 35 + W - 13) : C = 35 := by
  -- W = 48
  -- Charles loses 35 bananas
  -- Willie will have 13 bananas
  sorry

end charles_initial_bananas_l373_37387


namespace principal_sum_l373_37374

theorem principal_sum (R P : ℝ) (h : (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81) : P = 900 :=
by
  sorry

end principal_sum_l373_37374


namespace part1_part2_part3_l373_37360

variable {x y z : ℝ}

-- Given condition
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)

theorem part1 : 
  (x / y + y / z + z / x) / 3 ≥ 1 := sorry

theorem part2 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3 := sorry

theorem part3 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x := sorry

end part1_part2_part3_l373_37360


namespace power_product_to_seventh_power_l373_37321

theorem power_product_to_seventh_power :
  (2 ^ 14) * (2 ^ 21) = (32 ^ 7) :=
by
  sorry

end power_product_to_seventh_power_l373_37321


namespace common_ratio_of_geometric_sequence_l373_37381

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ)
  (h_geom : ∃ q, ∀ n, a (n+1) = a n * q)
  (h1 : a 1 = 1 / 8)
  (h4 : a 4 = -1) :
  ∃ q, q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l373_37381


namespace probability_meeting_part_a_l373_37326

theorem probability_meeting_part_a :
  ∃ p : ℝ, p = (11 : ℝ) / 36 :=
sorry

end probability_meeting_part_a_l373_37326


namespace fraction_to_decimal_l373_37372

theorem fraction_to_decimal :
  ∀ x : ℚ, x = 52 / 180 → x = 0.1444 := 
sorry

end fraction_to_decimal_l373_37372


namespace unique_perpendicular_line_through_point_l373_37349

variables (a b : ℝ → ℝ) (P : ℝ)

def are_skew_lines (a b : ℝ → ℝ) : Prop :=
  ¬∃ (t₁ t₂ : ℝ), a t₁ = b t₂

def is_point_not_on_lines (P : ℝ) (a b : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P ≠ a t ∧ P ≠ b t

theorem unique_perpendicular_line_through_point (ha : are_skew_lines a b) (hp : is_point_not_on_lines P a b) :
  ∃! (L : ℝ → ℝ), (∀ (t : ℝ), L t ≠ P) ∧ (∀ (L' : ℝ → ℝ), (∀ (t : ℝ), L' t ≠ P) → L' = L) := sorry

end unique_perpendicular_line_through_point_l373_37349


namespace prove_square_ratio_l373_37333
noncomputable section

-- Definitions from given conditions
variables (a b : ℝ) (d : ℝ := Real.sqrt (a^2 + b^2))

-- Condition from the problem
def ratio_condition : Prop := a / b = (a + 2 * b) / d

-- The theorem we need to prove
theorem prove_square_ratio (h : ratio_condition a b d) : 
  ∃ k : ℝ, k = a / b ∧ k^4 - 3*k^2 - 4*k - 4 = 0 := 
by
  sorry

end prove_square_ratio_l373_37333


namespace xy_product_l373_37361

theorem xy_product (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x * y = -24 := 
by {
  sorry
}

end xy_product_l373_37361


namespace biology_to_general_ratio_l373_37327

variable (g b m : ℚ)

theorem biology_to_general_ratio (h1 : g = 30) 
                                (h2 : m = (3/5) * (g + b)) 
                                (h3 : g + b + m = 144) : 
                                b / g = 2 / 1 := 
by 
  sorry

end biology_to_general_ratio_l373_37327


namespace continuous_linear_function_l373_37345

theorem continuous_linear_function {f : ℝ → ℝ} (h_cont : Continuous f) 
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_a_half : a < 1/2) (h_b_half : b < 1/2) 
  (h_eq : ∀ x : ℝ, f (f x) = a * f x + b * x) : 
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ (k * k - a * k - b = 0) := 
sorry

end continuous_linear_function_l373_37345


namespace increased_sales_type_B_l373_37389

-- Definitions for sales equations
def store_A_sales (x y : ℝ) : Prop :=
  60 * x + 15 * y = 3600

def store_B_sales (x y : ℝ) : Prop :=
  40 * x + 60 * y = 4400

-- Definition for the price of clothing items
def price_A (x : ℝ) : Prop :=
  x = 50

def price_B (y : ℝ) : Prop :=
  y = 40

-- Definition for the increased sales in May for type A
def may_sales_A (x : ℝ) : Prop :=
  100 * x * 1.2 = 6000

-- Definition to prove percentage increase for type B sales in May
noncomputable def percentage_increase_B (x y : ℝ) : ℝ :=
  ((4500 - (100 * y * 0.4)) / (100 * y * 0.4)) * 100

theorem increased_sales_type_B (x y : ℝ)
  (h1 : store_A_sales x y)
  (h2 : store_B_sales x y)
  (hA : price_A x)
  (hB : price_B y)
  (hMayA : may_sales_A x) :
  percentage_increase_B x y = 50 :=
sorry

end increased_sales_type_B_l373_37389
