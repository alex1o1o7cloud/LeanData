import Mathlib

namespace problem1_problem2_problem3_problem4_l1690_169077

theorem problem1 (α : ℝ) (h₁ : Real.sin α > 0) (h₂ : Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π/2 } := sorry

theorem problem2 (α : ℝ) (h₁ : Real.tan α * Real.sin α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > π ∧ x < 3 * π / 2) } := sorry

theorem problem3 (α : ℝ) (h₁ : Real.sin α * Real.cos α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > 3 * π / 2 ∧ x < 2 * π) } := sorry

theorem problem4 (α : ℝ) (h₁ : Real.cos α * Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π ∨ x > π ∧ x < 3 * π / 2 } := sorry

end problem1_problem2_problem3_problem4_l1690_169077


namespace num_positive_integer_N_l1690_169047

def num_valid_N : Nat := 7

theorem num_positive_integer_N (N : Nat) (h_pos : N > 0) :
  (∃ k : Nat, k > 3 ∧ N = k - 3 ∧ 48 % k = 0) ↔ (N < 45) ∧ (num_valid_N = 7) := 
by
sorry

end num_positive_integer_N_l1690_169047


namespace sum_of_first_40_terms_l1690_169021

def a : ℕ → ℤ := sorry

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem sum_of_first_40_terms :
  (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = n) →
  S 40 = 420 := 
sorry

end sum_of_first_40_terms_l1690_169021


namespace range_of_a_l1690_169058

theorem range_of_a 
    (x y a : ℝ) 
    (hx_pos : 0 < x) 
    (hy_pos : 0 < y) 
    (hxy : x + y = 1) 
    (hineq : ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → (1 / x + a / y) ≥ 4) :
    a ≥ 1 := 
by sorry

end range_of_a_l1690_169058


namespace remainder_1425_1427_1429_mod_12_l1690_169071

theorem remainder_1425_1427_1429_mod_12 :
  (1425 * 1427 * 1429) % 12 = 11 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l1690_169071


namespace prime_problem_l1690_169038

open Nat

-- Definition of primes and conditions based on the problem
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The formalized problem and conditions
theorem prime_problem (p q s : ℕ) 
  (p_prime : is_prime p) 
  (q_prime : is_prime q) 
  (s_prime : is_prime s) 
  (h1 : p + q = s + 4) 
  (h2 : 1 < p) 
  (h3 : p < q) : 
  p = 2 :=
sorry

end prime_problem_l1690_169038


namespace sin_870_eq_half_l1690_169053

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l1690_169053


namespace min_value_a4b3c2_l1690_169092

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l1690_169092


namespace ab_range_l1690_169031

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a * b = a + b + 8)

theorem ab_range (h : a * b = a + b + 8) : 16 ≤ a * b :=
by sorry

end ab_range_l1690_169031


namespace area_of_square_is_correct_l1690_169004

-- Define the nature of the problem setup and parameters
def radius_of_circle : ℝ := 7
def diameter_of_circle : ℝ := 2 * radius_of_circle
def side_length_of_square : ℝ := 2 * diameter_of_circle
def area_of_square : ℝ := side_length_of_square ^ 2

-- Statement of the problem to prove
theorem area_of_square_is_correct : area_of_square = 784 := by
  sorry

end area_of_square_is_correct_l1690_169004


namespace find_k_value_for_unique_real_solution_l1690_169019

noncomputable def cubic_has_exactly_one_real_solution (k : ℝ) : Prop :=
    ∃! x : ℝ, 4*x^3 + 9*x^2 + k*x + 4 = 0

theorem find_k_value_for_unique_real_solution :
  ∃ (k : ℝ), k > 0 ∧ cubic_has_exactly_one_real_solution k ∧ k = 6.75 :=
sorry

end find_k_value_for_unique_real_solution_l1690_169019


namespace initial_loss_percentage_l1690_169022

theorem initial_loss_percentage 
  (CP : ℝ := 250) 
  (SP : ℝ) 
  (h1 : SP + 50 = 1.10 * CP) : 
  (CP - SP) / CP * 100 = 10 := 
sorry

end initial_loss_percentage_l1690_169022


namespace exactly_three_assertions_l1690_169082

theorem exactly_three_assertions (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧
  ((x % 3 = 0) ∧ (x % 5 = 0) ∧ (x % 9 ≠ 0) ∧ (x % 15 = 0) ∧ (x % 25 ≠ 0) ∧ (x % 45 ≠ 0)) ↔
  (x = 15 ∨ x = 30 ∨ x = 60) :=
by
  sorry

end exactly_three_assertions_l1690_169082


namespace conference_handshakes_l1690_169020

theorem conference_handshakes (total_people : ℕ) (group1_people : ℕ) (group2_people : ℕ)
  (group1_knows_each_other : group1_people = 25)
  (group2_knows_no_one_in_group1 : group2_people = 15)
  (total_group : total_people = group1_people + group2_people)
  (total_handshakes : ℕ := group2_people * (group1_people + group2_people - 1) - group2_people * (group2_people - 1) / 2) :
  total_handshakes = 480 := by
  -- Placeholder for proof
  sorry

end conference_handshakes_l1690_169020


namespace vacation_cost_l1690_169055

theorem vacation_cost (C : ℝ) (h1 : C / 3 - C / 4 = 30) : C = 360 :=
by
  sorry

end vacation_cost_l1690_169055


namespace Carol_mother_carrots_l1690_169049

theorem Carol_mother_carrots (carol_picked : ℕ) (total_good : ℕ) (total_bad : ℕ) (total_carrots : ℕ) (mother_picked : ℕ) :
  carol_picked = 29 → total_good = 38 → total_bad = 7 → total_carrots = total_good + total_bad → mother_picked = total_carrots - carol_picked → mother_picked = 16 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end Carol_mother_carrots_l1690_169049


namespace percent_increase_surface_area_l1690_169078

theorem percent_increase_surface_area (a b c : ℝ) :
  let S := 2 * (a * b + b * c + a * c)
  let S' := 2 * (1.8 * a * 1.8 * b + 1.8 * b * 1.8 * c + 1.8 * c * 1.8 * a)
  (S' - S) / S * 100 = 224 := by
  sorry

end percent_increase_surface_area_l1690_169078


namespace solve_linear_eq_l1690_169033

theorem solve_linear_eq (x : ℝ) : 3 * x - 6 = 0 ↔ x = 2 :=
sorry

end solve_linear_eq_l1690_169033


namespace closest_multiple_of_15_to_2023_is_2025_l1690_169054

theorem closest_multiple_of_15_to_2023_is_2025 (n : ℤ) (h : 15 * n = 2025) : 
  ∀ m : ℤ, abs (2023 - 2025) ≤ abs (2023 - 15 * m) :=
by
  exact sorry

end closest_multiple_of_15_to_2023_is_2025_l1690_169054


namespace problem_statement_l1690_169010

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l1690_169010


namespace larger_box_can_carry_more_clay_l1690_169023

variable {V₁ : ℝ} -- Volume of the first box
variable {V₂ : ℝ} -- Volume of the second box
variable {m₁ : ℝ} -- Mass the first box can carry
variable {m₂ : ℝ} -- Mass the second box can carry

-- Defining the dimensions of the first box.
def height₁ : ℝ := 1
def width₁ : ℝ := 2
def length₁ : ℝ := 4

-- Defining the dimensions of the second box.
def height₂ : ℝ := 3 * height₁
def width₂ : ℝ := 2 * width₁
def length₂ : ℝ := 2 * length₁

-- Volume calculation for the first box.
def volume₁ : ℝ := height₁ * width₁ * length₁

-- Volume calculation for the second box.
def volume₂ : ℝ := height₂ * width₂ * length₂

-- Condition: The first box can carry 30 grams of clay
def mass₁ : ℝ := 30

-- Given the above conditions, prove the second box can carry 360 grams of clay.
theorem larger_box_can_carry_more_clay (h₁ : volume₁ = height₁ * width₁ * length₁)
                                      (h₂ : volume₂ = height₂ * width₂ * length₂)
                                      (h₃ : mass₁ = 30)
                                      (h₄ : V₁ = volume₁)
                                      (h₅ : V₂ = volume₂) :
  m₂ = 12 * mass₁ := by
  -- Skipping the detailed proof.
  sorry

end larger_box_can_carry_more_clay_l1690_169023


namespace intersection_A_complement_B_l1690_169083

def set_A : Set ℝ := {x | 1 < x ∧ x < 4}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_Complement_B : Set ℝ := {x | x < -1 ∨ x > 3}
def set_Intersection : Set ℝ := {x | set_A x ∧ set_Complement_B x}

theorem intersection_A_complement_B : set_Intersection = {x | 3 < x ∧ x < 4} := by
  sorry

end intersection_A_complement_B_l1690_169083


namespace find_x_l1690_169067

theorem find_x (x : ℕ) (h1 : x % 6 = 0) (h2 : x^2 > 144) (h3 : x < 30) : x = 18 ∨ x = 24 :=
sorry

end find_x_l1690_169067


namespace corset_total_cost_l1690_169066

def purple_bead_cost : ℝ := 50 * 20 * 0.12
def blue_bead_cost : ℝ := 40 * 18 * 0.10
def gold_bead_cost : ℝ := 80 * 0.08
def red_bead_cost : ℝ := 30 * 15 * 0.09
def silver_bead_cost : ℝ := 100 * 0.07

def total_cost : ℝ := purple_bead_cost + blue_bead_cost + gold_bead_cost + red_bead_cost + silver_bead_cost

theorem corset_total_cost : total_cost = 245.90 := by
  sorry

end corset_total_cost_l1690_169066


namespace complete_work_together_in_days_l1690_169074

noncomputable def a_days := 16
noncomputable def b_days := 6
noncomputable def c_days := 12

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem complete_work_together_in_days :
  let combined_rate := (work_rate a_days) + (work_rate b_days) + (work_rate c_days)
  let days_to_complete := 1 / combined_rate
  days_to_complete = 3.2 :=
  sorry

end complete_work_together_in_days_l1690_169074


namespace pow_div_l1690_169011

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l1690_169011


namespace ratio_of_d_to_s_l1690_169028

theorem ratio_of_d_to_s (s d : ℝ) (n : ℕ) (h1 : n = 15) (h2 : (n^2 * s^2) / ((n * s + 2 * n * d)^2) = 0.75) :
  d / s = 1 / 13 :=
by
  sorry

end ratio_of_d_to_s_l1690_169028


namespace b_10_eq_64_l1690_169036

noncomputable def a (n : ℕ) : ℕ := -- Definition of the sequence a_n
  sorry

noncomputable def b (n : ℕ) : ℕ := -- Definition of the sequence b_n
  a n + a (n + 1)

theorem b_10_eq_64 (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 1) = 2^n) :
  b 10 = 64 :=
sorry

end b_10_eq_64_l1690_169036


namespace smallest_perimeter_iso_triangle_l1690_169061

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l1690_169061


namespace even_sum_sufficient_not_necessary_l1690_169081

theorem even_sum_sufficient_not_necessary (m n : ℤ) : 
  (∀ m n : ℤ, (Even m ∧ Even n) → Even (m + n)) 
  ∧ (∀ a b : ℤ, Even (a + b) → ¬ (Odd a ∧ Odd b)) :=
by
  sorry

end even_sum_sufficient_not_necessary_l1690_169081


namespace count_teams_of_6_l1690_169040

theorem count_teams_of_6 
  (students : Fin 12 → Type)
  (played_together_once : ∀ (s : Finset (Fin 12)) (h : s.card = 5), ∃! t : Finset (Fin 12), t.card = 6 ∧ s ⊆ t) :
  (∃ team_count : ℕ, team_count = 132) :=
by
  -- Proof omitted
  sorry

end count_teams_of_6_l1690_169040


namespace product_of_three_numbers_l1690_169001

theorem product_of_three_numbers
  (x y z n : ℤ)
  (h1 : x + y + z = 165)
  (h2 : n = 7 * x)
  (h3 : n = y - 9)
  (h4 : n = z + 9) :
  x * y * z = 64328 := 
by
  sorry

end product_of_three_numbers_l1690_169001


namespace most_probable_hits_l1690_169070

theorem most_probable_hits (p : ℝ) (q : ℝ) (k0 : ℕ) (n : ℤ) 
  (h1 : p = 0.7) (h2 : q = 1 - p) (h3 : k0 = 16) 
  (h4 : 21 < (n : ℝ) * 0.7) (h5 : (n : ℝ) * 0.7 < 23.3) : 
  n = 22 ∨ n = 23 :=
sorry

end most_probable_hits_l1690_169070


namespace car_speed_l1690_169013

theorem car_speed (v : ℝ) (h : (1/v) * 3600 = ((1/48) * 3600) + 15) : v = 40 := 
by 
  sorry

end car_speed_l1690_169013


namespace parallel_lines_implies_m_no_perpendicular_lines_solution_l1690_169051

noncomputable def parallel_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ = y₂

noncomputable def perpendicular_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ * y₂ = -1

theorem parallel_lines_implies_m (m : ℝ) : parallel_slopes m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by
  sorry

theorem no_perpendicular_lines_solution (m : ℝ) : perpendicular_slopes m → false :=
by
  sorry

end parallel_lines_implies_m_no_perpendicular_lines_solution_l1690_169051


namespace fraction_value_l1690_169043

theorem fraction_value (p q x : ℚ) (h₁ : p / q = 4 / 5) (h₂ : 2 * q + p ≠ 0) (h₃ : 2 * q - p ≠ 0) :
  x + (2 * q - p) / (2 * q + p) = 2 → x = 11 / 7 :=
by
  sorry

end fraction_value_l1690_169043


namespace no_solution_for_equation_l1690_169095

theorem no_solution_for_equation : 
  ∀ x : ℝ, (x ≠ 3) → (x-1)/(x-3) = 2 - 2/(3-x) → False :=
by
  intro x hx heq
  sorry

end no_solution_for_equation_l1690_169095


namespace chandler_saves_weeks_l1690_169025

theorem chandler_saves_weeks 
  (cost_of_bike : ℕ) 
  (grandparents_money : ℕ) 
  (aunt_money : ℕ) 
  (cousin_money : ℕ) 
  (weekly_earnings : ℕ)
  (total_birthday_money : ℕ := grandparents_money + aunt_money + cousin_money) 
  (total_money : ℕ := total_birthday_money + weekly_earnings * 24):
  (cost_of_bike = 600) → 
  (grandparents_money = 60) → 
  (aunt_money = 40) → 
  (cousin_money = 20) → 
  (weekly_earnings = 20) → 
  (total_money = cost_of_bike) → 
  24 = ((cost_of_bike - total_birthday_money) / weekly_earnings) := 
by 
  intros; 
  sorry

end chandler_saves_weeks_l1690_169025


namespace sum_two_triangular_numbers_iff_l1690_169037

theorem sum_two_triangular_numbers_iff (m : ℕ) : 
  (∃ a b : ℕ, m = (a * (a + 1)) / 2 + (b * (b + 1)) / 2) ↔ 
  (∃ x y : ℕ, 4 * m + 1 = x * x + y * y) :=
by sorry

end sum_two_triangular_numbers_iff_l1690_169037


namespace james_monthly_earnings_l1690_169015

theorem james_monthly_earnings (initial_subscribers gifted_subscribers earnings_per_subscriber : ℕ)
  (initial_subscribers_eq : initial_subscribers = 150)
  (gifted_subscribers_eq : gifted_subscribers = 50)
  (earnings_per_subscriber_eq : earnings_per_subscriber = 9) :
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber = 1800 := by
  sorry

end james_monthly_earnings_l1690_169015


namespace find_radius_of_circle_l1690_169090

theorem find_radius_of_circle :
  ∀ (r : ℝ) (α : ℝ) (ρ : ℝ) (θ : ℝ), r > 0 →
  (∀ (x y : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α → x^2 + y^2 = r^2) →
  (∃ (x y: ℝ), x - y + 2 = 0 ∧ 2 * Real.sqrt (r^2 - 2) = 2 * Real.sqrt 2) →
  r = 2 :=
by
  intro r α ρ θ r_pos curve_eq polar_eq
  sorry

end find_radius_of_circle_l1690_169090


namespace solve_for_y_l1690_169041

theorem solve_for_y (x y : ℝ) (h₁ : x^(2 * y) = 64) (h₂ : x = 8) : y = 1 :=
by
  sorry

end solve_for_y_l1690_169041


namespace James_pays_35_l1690_169034

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end James_pays_35_l1690_169034


namespace relationship_P_Q_l1690_169014

theorem relationship_P_Q (x : ℝ) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.exp x + Real.exp (-x)) 
  (hQ : Q = (Real.sin x + Real.cos x) ^ 2) : 
  P ≥ Q := 
sorry

end relationship_P_Q_l1690_169014


namespace sqrt_expression_value_l1690_169012

variable (a b : ℝ) 

theorem sqrt_expression_value (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * |b| * Real.sqrt (-a) := by
  sorry

end sqrt_expression_value_l1690_169012


namespace find_x_l1690_169029

theorem find_x (x : ℚ) (h : |x - 1| = |x - 2|) : x = 3 / 2 :=
sorry

end find_x_l1690_169029


namespace total_beads_correct_l1690_169097

-- Definitions of the problem conditions
def blue_beads : ℕ := 5
def red_beads : ℕ := 2 * blue_beads
def white_beads : ℕ := blue_beads + red_beads
def silver_beads : ℕ := 10

-- Definition of the total number of beads
def total_beads : ℕ := blue_beads + red_beads + white_beads + silver_beads

-- The main theorem statement
theorem total_beads_correct : total_beads = 40 :=
by 
  sorry

end total_beads_correct_l1690_169097


namespace angle_of_inclination_l1690_169039

theorem angle_of_inclination (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, 3)) : 
  ∃ θ : ℝ, θ = (3 * Real.pi) / 4 ∧ (∃ k : ℝ, k = (A.2 - B.2) / (A.1 - B.1) ∧ Real.tan θ = k) :=
by
  sorry

end angle_of_inclination_l1690_169039


namespace max_divisor_f_l1690_169064

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_f (m : ℕ) : (∀ n : ℕ, m ∣ f n) → m = 36 :=
sorry

end max_divisor_f_l1690_169064


namespace find_c_and_d_l1690_169042

theorem find_c_and_d (c d : ℝ) (h : ℝ → ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 * x - 5)
  (finv_eq : ∀ x, finv x = 6 * x - 3)
  (f_def : ∀ x, f x = c * x + d)
  (inv_prop : ∀ x, f (finv x) = x ∧ finv (f x) = x) :
  4 * c + 6 * d = 11 / 3 :=
by
  sorry

end find_c_and_d_l1690_169042


namespace min_value_expr_min_max_value_expr_max_l1690_169027

noncomputable def min_value_expr (a b : ℝ) : ℝ := 
  1 / (a - b) + 4 / (b - 1)

noncomputable def max_value_expr (a b : ℝ) : ℝ :=
  a * b - b^2 - a + b

theorem min_value_expr_min (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) : 
  min_value_expr a b = 25 :=
sorry

theorem max_value_expr_max (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) :
  max_value_expr a b = 1 / 16 :=
sorry

end min_value_expr_min_max_value_expr_max_l1690_169027


namespace parity_of_f_l1690_169068

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ :=
  x * (x - 2) * (x - 1) * x * (x + 1) * (x + 2)

theorem parity_of_f :
  is_even_function f ∧ ¬ (∃ g : ℝ → ℝ, g = f ∧ (∀ x : ℝ, g (-x) = -g x)) :=
by
  sorry

end parity_of_f_l1690_169068


namespace collinear_iff_linear_combination_l1690_169007

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (k : ℝ)

theorem collinear_iff_linear_combination (O A B C : V) (k : ℝ) :
  (C = k • A + (1 - k) • B) ↔ ∃ (k' : ℝ), C - B = k' • (A - B) :=
sorry

end collinear_iff_linear_combination_l1690_169007


namespace jerry_total_bill_l1690_169093

-- Definitions for the initial bill and late fees
def initial_bill : ℝ := 250
def first_fee_rate : ℝ := 0.02
def second_fee_rate : ℝ := 0.03

-- Function to calculate the total bill after applying the fees
def total_bill (init : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let first_total := init * (1 + rate1)
  first_total * (1 + rate2)

-- Theorem statement
theorem jerry_total_bill : total_bill initial_bill first_fee_rate second_fee_rate = 262.65 := by
  sorry

end jerry_total_bill_l1690_169093


namespace insulation_cost_of_rectangular_tank_l1690_169080

theorem insulation_cost_of_rectangular_tank
  (l w h cost_per_sq_ft : ℕ)
  (hl : l = 4) (hw : w = 5) (hh : h = 3) (hc : cost_per_sq_ft = 20) :
  2 * l * w + 2 * l * h + 2 * w * h * 20 = 1880 :=
by
  sorry

end insulation_cost_of_rectangular_tank_l1690_169080


namespace middle_number_of_five_consecutive_numbers_l1690_169063

theorem middle_number_of_five_consecutive_numbers (n : ℕ) 
  (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 60) : n = 12 :=
by
  sorry

end middle_number_of_five_consecutive_numbers_l1690_169063


namespace jeans_and_shirts_l1690_169026

-- Let's define the necessary variables and conditions.
variables (J S X : ℝ)

-- Given conditions
def condition1 := 3 * J + 2 * S = X
def condition2 := 2 * J + 3 * S = 61

-- Given the price of one shirt
def price_of_shirt := S = 9

-- The problem we need to prove
theorem jeans_and_shirts : condition1 J S X ∧ condition2 J S ∧ price_of_shirt S →
  X = 69 :=
by
  sorry

end jeans_and_shirts_l1690_169026


namespace binary_addition_l1690_169052

theorem binary_addition :
  (0b1101 : Nat) + 0b101 + 0b1110 + 0b111 + 0b1010 = 0b10101 := by
  sorry

end binary_addition_l1690_169052


namespace flag_count_l1690_169003

-- Definitions of colors as a datatype
inductive Color
| red : Color
| white : Color
| blue : Color
| green : Color
| yellow : Color

open Color

-- Total number of distinct flags possible
theorem flag_count : 
  (∃ m : Color, 
   (∃ t : Color, 
    (t ≠ m ∧ 
     ∃ b : Color, 
     (b ≠ m ∧ b ≠ red ∧ b ≠ blue)))) ∧ 
  (5 * 4 * 2 = 40) := 
  sorry

end flag_count_l1690_169003


namespace distinct_numbers_mean_inequality_l1690_169059

open Nat

theorem distinct_numbers_mean_inequality (n m : ℕ) (h_n_m : m ≤ n)
  (a : Fin m → ℕ) (ha_distinct : Function.Injective a)
  (h_cond : ∀ (i j : Fin m), i ≠ j → i.val + j.val ≤ n → ∃ (k : Fin m), a i + a j = a k) :
  (1 : ℝ) / m * (Finset.univ.sum (fun i => a i)) ≥  (n + 1) / 2 :=
by
  sorry

end distinct_numbers_mean_inequality_l1690_169059


namespace jennifer_sweets_l1690_169024

theorem jennifer_sweets :
  let green_sweets := 212
  let blue_sweets := 310
  let yellow_sweets := 502
  let total_sweets := green_sweets + blue_sweets + yellow_sweets
  let number_of_people := 4
  total_sweets / number_of_people = 256 := 
by
  sorry

end jennifer_sweets_l1690_169024


namespace Louisa_traveled_240_miles_first_day_l1690_169006

noncomputable def distance_first_day (h : ℕ) := 60 * (h - 3)

theorem Louisa_traveled_240_miles_first_day :
  ∃ h : ℕ, 420 = 60 * h ∧ distance_first_day h = 240 :=
by
  sorry

end Louisa_traveled_240_miles_first_day_l1690_169006


namespace no_partition_exists_l1690_169094

theorem no_partition_exists : ¬ ∃ (x y : ℕ), 
    (1 ≤ x ∧ x ≤ 15) ∧ 
    (1 ≤ y ∧ y ≤ 15) ∧ 
    (x * y = 120 - x - y) :=
by
  sorry

end no_partition_exists_l1690_169094


namespace smallest_cube_dividing_pq2r4_l1690_169060

-- Definitions of conditions
variables {p q r : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] [Fact (Nat.Prime r)]
variables (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- Definitions used in the proof
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

def smallest_perfect_cube_dividing (n k : ℕ) : Prop :=
  is_perfect_cube k ∧ n ∣ k ∧ ∀ k', is_perfect_cube k' ∧ n ∣ k' → k ≤ k'

-- The proof problem
theorem smallest_cube_dividing_pq2r4 (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  smallest_perfect_cube_dividing (p * q^2 * r^4) ((p * q * r^2)^3) :=
sorry

end smallest_cube_dividing_pq2r4_l1690_169060


namespace angles_arith_prog_tangent_tangent_parallel_euler_line_l1690_169044

-- Define a non-equilateral triangle with angles in arithmetic progression
structure Triangle :=
  (A B C : ℝ) -- Angles in a non-equilateral triangle
  (non_equilateral : A ≠ B ∨ B ≠ C ∨ A ≠ C)
  (angles_arith_progression : (2 * B = A + C))

-- Additional geometry concepts will be assumptions as their definition 
-- would involve extensive axiomatic setups

-- The main theorem to state the equivalence
theorem angles_arith_prog_tangent_tangent_parallel_euler_line (Δ : Triangle)
  (common_tangent_parallel_euler : sorry) : 
  ((Δ.A = 60) ∨ (Δ.B = 60) ∨ (Δ.C = 60)) :=
sorry

end angles_arith_prog_tangent_tangent_parallel_euler_line_l1690_169044


namespace eulers_polyhedron_theorem_l1690_169084

theorem eulers_polyhedron_theorem 
  (V E F t h : ℕ) (T H : ℕ) :
  (F = 30) →
  (t = 20) →
  (h = 10) →
  (T = 3) →
  (H = 2) →
  (E = (3 * t + 6 * h) / 2) →
  (V - E + F = 2) →
  100 * H + 10 * T + V = 262 :=
by
  intros F_eq t_eq h_eq T_eq H_eq E_eq euler_eq
  rw [F_eq, t_eq, h_eq, T_eq, H_eq, E_eq] at *
  sorry

end eulers_polyhedron_theorem_l1690_169084


namespace max_a_is_2_l1690_169032

noncomputable def max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : ℝ :=
  2

theorem max_a_is_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  max_value_of_a a b c h1 h2 = 2 :=
sorry

end max_a_is_2_l1690_169032


namespace c_amount_correct_b_share_correct_l1690_169048

-- Conditions
def total_sum : ℝ := 246    -- Total sum of money
def c_share : ℝ := 48      -- C's share in Rs
def c_per_rs : ℝ := 0.40   -- C's amount per Rs

-- Expressing the given condition c_share = total sum * c_per_rs
theorem c_amount_correct : c_share = total_sum * c_per_rs := 
  by
  -- Substitute that can be more elaboration of the calculations done
  sorry

-- Additional condition for the total per Rs distribution
axiom a_b_c_total : ∀ (a b : ℝ), a + b + c_per_rs = 1

-- Proving B's share per Rs is approximately 0.4049
theorem b_share_correct : ∃ a b : ℝ, c_share = 246 * 0.40 ∧ a + b + 0.40 = 1 ∧ b = 1 - (48 / 246) - 0.40 := 
  by
  -- Substitute that can be elaboration of the proof arguments done in the translated form
  sorry

end c_amount_correct_b_share_correct_l1690_169048


namespace max_value_of_f_l1690_169056

open Real

noncomputable def f (θ : ℝ) : ℝ :=
  sin (θ / 2) * (1 + cos θ)

theorem max_value_of_f : 
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π → f θ' ≤ f θ) ∧ f θ = 4 * sqrt 3 / 9) := 
by
  sorry

end max_value_of_f_l1690_169056


namespace line_fixed_point_l1690_169075

theorem line_fixed_point (m : ℝ) : ∃ x y, (∀ m, y = m * x + (2 * m + 1)) ↔ (x = -2 ∧ y = 1) :=
by
  sorry

end line_fixed_point_l1690_169075


namespace marts_income_percentage_of_juans_l1690_169073

variable (T J M : Real)
variable (h1 : M = 1.60 * T)
variable (h2 : T = 0.40 * J)

theorem marts_income_percentage_of_juans : M = 0.64 * J :=
by
  sorry

end marts_income_percentage_of_juans_l1690_169073


namespace tan_neg_five_pi_over_three_l1690_169000

theorem tan_neg_five_pi_over_three : Real.tan (-5 * Real.pi / 3) = Real.sqrt 3 := 
by 
  sorry

end tan_neg_five_pi_over_three_l1690_169000


namespace total_number_of_flowers_l1690_169002

theorem total_number_of_flowers (pots : ℕ) (flowers_per_pot : ℕ) (h_pots : pots = 544) (h_flowers_per_pot : flowers_per_pot = 32) : 
  pots * flowers_per_pot = 17408 := by
  sorry

end total_number_of_flowers_l1690_169002


namespace reading_ratio_l1690_169099

theorem reading_ratio (x : ℕ) (h1 : 10 * x + 5 * (75 - x) = 500) : 
  (10 * x) / 500 = 1 / 2 :=
by sorry

end reading_ratio_l1690_169099


namespace sequence_nth_term_16_l1690_169096

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end sequence_nth_term_16_l1690_169096


namespace P_never_77_l1690_169072

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_never_77 (x y : ℤ) : P x y ≠ 77 := sorry

end P_never_77_l1690_169072


namespace man_upstream_rate_l1690_169086

theorem man_upstream_rate (rate_downstream : ℝ) (rate_still_water : ℝ) (rate_current : ℝ) 
    (h1 : rate_downstream = 32) (h2 : rate_still_water = 24.5) (h3 : rate_current = 7.5) : 
    rate_still_water - rate_current = 17 := 
by 
  sorry

end man_upstream_rate_l1690_169086


namespace funfair_initial_visitors_l1690_169018

theorem funfair_initial_visitors {a : ℕ} (ha1 : 50 * a - 40 > 0) (ha2 : 90 - 20 * a > 0) (ha3 : 50 * a - 40 > 90 - 20 * a) :
  (50 * a - 40 = 60) ∨ (50 * a - 40 = 110) ∨ (50 * a - 40 = 160) :=
sorry

end funfair_initial_visitors_l1690_169018


namespace removing_zeros_changes_value_l1690_169035

noncomputable def a : ℝ := 7.0800
noncomputable def b : ℝ := 7.8

theorem removing_zeros_changes_value : a ≠ b :=
by
  -- proof goes here
  sorry

end removing_zeros_changes_value_l1690_169035


namespace Cornelia_three_times_Kilee_l1690_169085

variable (x : ℕ)

def Kilee_current_age : ℕ := 20
def Cornelia_current_age : ℕ := 80

theorem Cornelia_three_times_Kilee (x : ℕ) :
  Cornelia_current_age + x = 3 * (Kilee_current_age + x) ↔ x = 10 :=
by
  sorry

end Cornelia_three_times_Kilee_l1690_169085


namespace sin_geq_tan_minus_half_tan_cubed_l1690_169098

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (hx : 0 ≤ x ∧ x < π / 2) :
  Real.sin x ≥ Real.tan x - 1/2 * (Real.tan x) ^ 3 := 
sorry

end sin_geq_tan_minus_half_tan_cubed_l1690_169098


namespace chord_line_parabola_l1690_169017

theorem chord_line_parabola (x1 x2 y1 y2 : ℝ) (hx1 : y1^2 = 8*x1) (hx2 : y2^2 = 8*x2)
  (hmid : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1) : 4*(1/2*(x1 + x2)) + (1/2*(y1 + y2)) - 3 = 0 :=
by
  sorry

end chord_line_parabola_l1690_169017


namespace parallel_lines_intersect_parabola_l1690_169005

theorem parallel_lines_intersect_parabola {a k b c x1 x2 x3 x4 : ℝ} 
    (h₁ : x1 < x2) 
    (h₂ : x3 < x4) 
    (intersect1 : ∀ y : ℝ, y = k * x1 + b ∧ y = a * x1^2 ∧ y = k * x2 + b ∧ y = a * x2^2) 
    (intersect2 : ∀ y : ℝ, y = k * x3 + c ∧ y = a * x3^2 ∧ y = k * x4 + c ∧ y = a * x4^2) :
    (x3 - x1) = (x2 - x4) := 
by 
    sorry

end parallel_lines_intersect_parabola_l1690_169005


namespace user_count_exceed_50000_l1690_169076

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end user_count_exceed_50000_l1690_169076


namespace percentage_of_employees_driving_l1690_169008

theorem percentage_of_employees_driving
  (total_employees : ℕ)
  (drivers : ℕ)
  (public_transport : ℕ)
  (H1 : total_employees = 200)
  (H2 : drivers = public_transport + 40)
  (H3 : public_transport = (total_employees - drivers) / 2) :
  (drivers:ℝ) / (total_employees:ℝ) * 100 = 46.5 :=
by {
  sorry
}

end percentage_of_employees_driving_l1690_169008


namespace largest_angle_in_triangle_l1690_169087

theorem largest_angle_in_triangle
    (a b c : ℝ)
    (h_sum_two_angles : a + b = (7 / 5) * 90)
    (h_angle_difference : b = a + 40) :
    max a (max b c) = 83 :=
by
  sorry

end largest_angle_in_triangle_l1690_169087


namespace interest_rate_difference_l1690_169079

theorem interest_rate_difference (P T : ℝ) (R1 R2 : ℝ) (I_diff : ℝ) (hP : P = 2100) 
  (hT : T = 3) (hI : I_diff = 63) :
  R2 - R1 = 0.01 :=
by
  sorry

end interest_rate_difference_l1690_169079


namespace tan_alpha_plus_beta_tan_beta_l1690_169009

variable (α β : ℝ)

-- Given conditions
def tan_condition_1 : Prop := Real.tan (Real.pi + α) = -1 / 3
def tan_condition_2 : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Proving the results
theorem tan_alpha_plus_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) : 
  Real.tan (α + β) = 5 / 16 :=
sorry

theorem tan_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) :
  Real.tan β = 31 / 43 :=
sorry

end tan_alpha_plus_beta_tan_beta_l1690_169009


namespace new_cube_weight_l1690_169046

-- Define the weight function for a cube given side length and density.
def weight (ρ : ℝ) (s : ℝ) : ℝ := ρ * s^3

-- Given conditions: the weight of the original cube.
axiom original_weight : ∃ ρ s : ℝ, weight ρ s = 7

-- The goal is to prove that a new cube with sides twice as long weighs 56 pounds.
theorem new_cube_weight : 
  (∃ ρ s : ℝ, weight ρ (2 * s) = 56) := by
  sorry

end new_cube_weight_l1690_169046


namespace solve_inequality_l1690_169030

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := (x - 2) * (a * x + 2 * a)

-- Theorem Statement
theorem solve_inequality (f_even : ∀ x a, f x a = f (-x) a) (f_inc : ∀ x y a, 0 < x → x < y → f x a ≤ f y a) :
    ∀ a > 0, { x : ℝ | f (2 - x) a > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  -- Sorry to skip the proof
  sorry

end solve_inequality_l1690_169030


namespace no_integer_solution_l1690_169016

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 - 7 * y = 10 :=
by
  sorry

end no_integer_solution_l1690_169016


namespace dot_product_bounds_l1690_169050

theorem dot_product_bounds
  (A : ℝ × ℝ)
  (hA : A.1 ^ 2 + (A.2 - 1) ^ 2 = 1) :
  -2 ≤ A.1 * 2 ∧ A.1 * 2 ≤ 2 := 
sorry

end dot_product_bounds_l1690_169050


namespace sunlovers_happy_days_l1690_169045

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l1690_169045


namespace rectangle_area_l1690_169062

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l1690_169062


namespace find_number_chosen_l1690_169091

theorem find_number_chosen (x : ℤ) (h : 4 * x - 138 = 102) : x = 60 := sorry

end find_number_chosen_l1690_169091


namespace geometric_sequence_b_l1690_169065

theorem geometric_sequence_b (b : ℝ) (h1 : b > 0) (h2 : 30 * (b / 30) = b) (h3 : b * (b / 30) = 9 / 4) :
  b = 3 * Real.sqrt 30 / 2 :=
by
  sorry

end geometric_sequence_b_l1690_169065


namespace car_speed_correct_l1690_169069

noncomputable def car_speed (d v_bike t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0): ℝ := 2 * v_bike

theorem car_speed_correct:
  ∀ (d v_bike : ℝ) (t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0),
    (d / v_bike - t_delay = d / (car_speed d v_bike t_delay h1 h2)) → 
    car_speed d v_bike t_delay h1 h2 = 0.6 :=
by
  intros
  -- The proof would go here
  sorry

end car_speed_correct_l1690_169069


namespace simplify_expression_l1690_169057

variables (a b : ℝ)

theorem simplify_expression : 
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := by
  -- proof here
  sorry

end simplify_expression_l1690_169057


namespace spherical_to_rectangular_conversion_l1690_169089

theorem spherical_to_rectangular_conversion :
  ∃ x y z : ℝ, 
    x = -Real.sqrt 2 ∧ 
    y = 0 ∧ 
    z = Real.sqrt 2 ∧ 
    (∃ rho theta phi : ℝ, 
      rho = 2 ∧
      theta = π ∧
      phi = π/4 ∧
      x = rho * Real.sin phi * Real.cos theta ∧
      y = rho * Real.sin phi * Real.sin theta ∧
      z = rho * Real.cos phi) :=
by
  sorry

end spherical_to_rectangular_conversion_l1690_169089


namespace top_card_is_club_probability_l1690_169088

-- Definitions based on the conditions
def deck_size := 52
def suit_count := 4
def cards_per_suit := deck_size / suit_count

-- The question we want to prove
theorem top_card_is_club_probability :
  (13 : ℝ) / (52 : ℝ) = 1 / 4 :=
by 
  sorry

end top_card_is_club_probability_l1690_169088
