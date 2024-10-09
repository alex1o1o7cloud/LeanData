import Mathlib

namespace problem1_problem2_l340_34084

-- Define variables
variables {x y m : ℝ}
variables (h1 : x + y > 0) (h2 : xy ≠ 0)

-- Problem (1): Prove that x^3 + y^3 ≥ x^2 y + y^2 x
theorem problem1 (h1 : x + y > 0) (h2 : xy ≠ 0) : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
sorry

-- Problem (2): Given the conditions, the range of m is [-6, 2]
theorem problem2 (h1 : x + y > 0) (h2 : xy ≠ 0) (h3 : (x / y^2) + (y / x^2) ≥ (m / 2) * ((1 / x) + (1 / y))) : m ∈ Set.Icc (-6 : ℝ) 2 :=
sorry

end problem1_problem2_l340_34084


namespace spike_hunts_20_crickets_per_day_l340_34098

/-- Spike the bearded dragon hunts 5 crickets every morning -/
def spike_morning_crickets : ℕ := 5

/-- Spike hunts three times the morning amount in the afternoon and evening -/
def spike_afternoon_evening_multiplier : ℕ := 3

/-- Total number of crickets Spike hunts per day -/
def spike_total_crickets_per_day : ℕ := spike_morning_crickets + spike_morning_crickets * spike_afternoon_evening_multiplier

/-- Prove that the total number of crickets Spike hunts per day is 20 -/
theorem spike_hunts_20_crickets_per_day : spike_total_crickets_per_day = 20 := 
by
  sorry

end spike_hunts_20_crickets_per_day_l340_34098


namespace quadratic_equation_root_zero_l340_34003

/-- Given that x = -3 is a root of the quadratic equation x^2 + 3x + k = 0,
    prove that the other root of the equation is 0 and k = 0. -/
theorem quadratic_equation_root_zero (k : ℝ) (h : -3^2 + 3 * -3 + k = 0) :
  (∀ t : ℝ, t^2 + 3 * t + k = 0 → t = 0) ∧ k = 0 :=
sorry

end quadratic_equation_root_zero_l340_34003


namespace edward_initial_money_l340_34080

variable (spent_books : ℕ) (spent_pens : ℕ) (money_left : ℕ)

theorem edward_initial_money (h_books : spent_books = 6) 
                             (h_pens : spent_pens = 16)
                             (h_left : money_left = 19) : 
                             spent_books + spent_pens + money_left = 41 := by
  sorry

end edward_initial_money_l340_34080


namespace james_drinks_per_day_l340_34097

-- condition: James buys 5 packs of sodas, each contains 12 sodas
def num_packs : Nat := 5
def sodas_per_pack : Nat := 12
def sodas_bought : Nat := num_packs * sodas_per_pack

-- condition: James already had 10 sodas
def sodas_already_had : Nat := 10

-- condition: James finishes all the sodas in 1 week (7 days)
def days_in_week : Nat := 7

-- total sodas
def total_sodas : Nat := sodas_bought + sodas_already_had

-- number of sodas james drinks per day
def sodas_per_day : Nat := 10

-- proof problem
theorem james_drinks_per_day : (total_sodas / days_in_week) = sodas_per_day :=
  sorry

end james_drinks_per_day_l340_34097


namespace total_number_of_legs_is_40_l340_34065

-- Define the number of octopuses Carson saw.
def number_of_octopuses := 5

-- Define the number of legs per octopus.
def legs_per_octopus := 8

-- Define the total number of octopus legs Carson saw.
def total_octopus_legs : Nat := number_of_octopuses * legs_per_octopus

-- Prove that the total number of octopus legs Carson saw is 40.
theorem total_number_of_legs_is_40 : total_octopus_legs = 40 := by
  sorry

end total_number_of_legs_is_40_l340_34065


namespace length_of_platform_l340_34090

noncomputable def train_length : ℝ := 450
noncomputable def signal_pole_time : ℝ := 18
noncomputable def platform_time : ℝ := 39

theorem length_of_platform : 
  ∃ (L : ℝ), 
    (train_length / signal_pole_time = (train_length + L) / platform_time) → 
    L = 525 := 
by
  sorry

end length_of_platform_l340_34090


namespace find_number_l340_34070

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 56) : x = 140 := 
by {
  -- The proof would be written here,
  -- but it is indicated to skip it using "sorry"
  sorry
}

end find_number_l340_34070


namespace number_of_footballs_l340_34043

theorem number_of_footballs (x y : ℕ) (h1 : x + y = 20) (h2 : 6 * x + 3 * y = 96) : x = 12 :=
by {
  sorry
}

end number_of_footballs_l340_34043


namespace triangle_angle_contradiction_l340_34021

theorem triangle_angle_contradiction (a b c : ℝ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : a + b + c = 180) (h₃ : 60 < a ∧ 60 < b ∧ 60 < c) : false :=
by
  sorry

end triangle_angle_contradiction_l340_34021


namespace eggs_division_l340_34047

theorem eggs_division (n_students n_eggs : ℕ) (h_students : n_students = 9) (h_eggs : n_eggs = 73):
  n_eggs / n_students = 8 ∧ n_eggs % n_students = 1 :=
by
  rw [h_students, h_eggs]
  exact ⟨rfl, rfl⟩

end eggs_division_l340_34047


namespace population_weight_of_500_students_l340_34001

-- Definitions
def number_of_students : ℕ := 500
def number_of_selected_students : ℕ := 60

-- Conditions
def condition1 := number_of_students = 500
def condition2 := number_of_selected_students = 60

-- Statement
theorem population_weight_of_500_students : 
  condition1 → condition2 → 
  (∃ p, p = "the weight of the 500 students") := by
  intros _ _
  existsi "the weight of the 500 students"
  rfl

end population_weight_of_500_students_l340_34001


namespace time_between_ticks_at_6_oclock_l340_34045

theorem time_between_ticks_at_6_oclock (ticks6 ticks12 intervals6 intervals12 total_time12: ℕ) (time_per_tick : ℕ) :
  ticks6 = 6 →
  ticks12 = 12 →
  total_time12 = 66 →
  intervals12 = ticks12 - 1 →
  time_per_tick = total_time12 / intervals12 →
  intervals6 = ticks6 - 1 →
  (time_per_tick * intervals6) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end time_between_ticks_at_6_oclock_l340_34045


namespace bees_multiple_l340_34015

theorem bees_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 :=
by
  sorry

end bees_multiple_l340_34015


namespace train_speed_faster_l340_34064

-- The Lean statement of the problem
theorem train_speed_faster (Vs : ℝ) (L : ℝ) (T : ℝ) (Vf : ℝ) :
  Vs = 36 ∧ L = 340 ∧ T = 17 ∧ (Vf - Vs) * (5 / 18) = L / T → Vf = 108 :=
by 
  intros 
  sorry

end train_speed_faster_l340_34064


namespace count_integer_b_for_log_b_256_l340_34053

theorem count_integer_b_for_log_b_256 :
  (∃ b : ℕ, b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) ∧ 
  (∀ b : ℕ, (b > 1 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 256) → (b = 2 ∨ b = 4 ∨ b = 16 ∨ b = 256)) :=
by sorry

end count_integer_b_for_log_b_256_l340_34053


namespace john_age_l340_34091

/-
Problem statement:
John is 24 years younger than his dad. The sum of their ages is 68 years.
We need to prove that John is 22 years old.
-/

theorem john_age:
  ∃ (j d : ℕ), (j = d - 24 ∧ j + d = 68) → j = 22 :=
by
  sorry

end john_age_l340_34091


namespace sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l340_34006

theorem sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0 :
  (9^25 + 11^25) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_9pow25_plus_11pow25_eq_0_l340_34006


namespace number_of_boys_l340_34057

-- Definitions from the problem conditions
def trees : ℕ := 29
def trees_per_boy : ℕ := 3

-- Prove the number of boys is 10
theorem number_of_boys : (trees / trees_per_boy) + 1 = 10 :=
by sorry

end number_of_boys_l340_34057


namespace cubes_side_length_l340_34032

theorem cubes_side_length (s : ℝ) (h : 2 * (s * s + s * 2 * s + s * 2 * s) = 10) : s = 1 :=
by
  sorry

end cubes_side_length_l340_34032


namespace floor_negative_fraction_l340_34023

theorem floor_negative_fraction : (Int.floor (-7 / 4 : ℚ)) = -2 := by
  sorry

end floor_negative_fraction_l340_34023


namespace no_solutions_l340_34081

theorem no_solutions (x : ℝ) (hx : x ≠ 0): ¬ (12 * Real.sin x + 5 * Real.cos x = 13 + 1 / |x|) := 
by 
  sorry

end no_solutions_l340_34081


namespace find_c_d_l340_34076

theorem find_c_d (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∧ x = d)) :
  c = 1 ∧ d = -2 :=
by
  sorry

end find_c_d_l340_34076


namespace find_x_l340_34088

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end find_x_l340_34088


namespace correct_average_is_26_l340_34093

noncomputable def initial_average : ℕ := 20
noncomputable def number_of_numbers : ℕ := 10
noncomputable def incorrect_number : ℕ := 26
noncomputable def correct_number : ℕ := 86
noncomputable def incorrect_total_sum : ℕ := initial_average * number_of_numbers
noncomputable def correct_total_sum : ℕ := incorrect_total_sum + (correct_number - incorrect_number)
noncomputable def correct_average : ℕ := correct_total_sum / number_of_numbers

theorem correct_average_is_26 :
  correct_average = 26 := by
  sorry

end correct_average_is_26_l340_34093


namespace larger_segment_length_l340_34055

open Real

theorem larger_segment_length (a b c : ℝ) (h : a = 50 ∧ b = 110 ∧ c = 120) :
  ∃ x : ℝ, x = 100 ∧ (∃ h : ℝ, a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2) :=
by
  sorry

end larger_segment_length_l340_34055


namespace a2_a3_equals_20_l340_34096

-- Sequence definition
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

-- Proof that a_2 * a_3 = 20
theorem a2_a3_equals_20 :
  a_n 2 * a_n 3 = 20 :=
by
  sorry

end a2_a3_equals_20_l340_34096


namespace distinct_triangles_from_tetrahedron_l340_34002

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l340_34002


namespace maria_strawberries_l340_34009

theorem maria_strawberries (S : ℕ) :
  (21 = 8 + 9 + S) → (S = 4) :=
by
  intro h
  sorry

end maria_strawberries_l340_34009


namespace flooring_sq_ft_per_box_l340_34061

/-- The problem statement converted into a Lean theorem -/
theorem flooring_sq_ft_per_box
  (living_room_length : ℕ)
  (living_room_width : ℕ)
  (flooring_installed : ℕ)
  (additional_boxes : ℕ)
  (correct_answer : ℕ) 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_installed = 250)
  (h4 : additional_boxes = 7)
  (h5 : correct_answer = 10) :
  
  (living_room_length * living_room_width - flooring_installed) / additional_boxes = correct_answer :=
by 
  sorry

end flooring_sq_ft_per_box_l340_34061


namespace FC_value_l340_34025

variables (DC CB AB AD ED FC CA BD : ℝ)

-- Set the conditions as variables
variable (h_DC : DC = 10)
variable (h_CB : CB = 12)
variable (h_AB : AB = (1/3) * AD)
variable (h_ED : ED = (2/3) * AD)
variable (h_BD : BD = 22)
variable (BD_eq : BD = DC + CB)
variable (CA_eq : CA = CB + AB)

-- Define the relationship for the final result
def find_FC (DC CB AB AD ED FC CA BD : ℝ) := FC = (ED * CA) / AD

-- The main statement to be proven
theorem FC_value : 
  find_FC DC CB AB (33 : ℝ) (22 : ℝ) FC (23 : ℝ) (22 : ℝ) → 
  FC = (506/33) :=
by 
  intros h
  sorry

end FC_value_l340_34025


namespace cupcake_difference_l340_34013

def betty_rate : ℕ := 10
def dora_rate : ℕ := 8
def total_hours : ℕ := 5
def betty_break_hours : ℕ := 2

theorem cupcake_difference :
  (dora_rate * total_hours) - (betty_rate * (total_hours - betty_break_hours)) = 10 :=
by
  sorry

end cupcake_difference_l340_34013


namespace Tino_has_correct_jellybeans_total_jellybeans_l340_34056

-- Define the individuals and their amounts of jellybeans
def Arnold_jellybeans := 5
def Lee_jellybeans := 2 * Arnold_jellybeans
def Tino_jellybeans := Lee_jellybeans + 24
def Joshua_jellybeans := 3 * Arnold_jellybeans

-- Verify Tino's jellybean count
theorem Tino_has_correct_jellybeans : Tino_jellybeans = 34 :=
by
  -- Unfold definitions and perform calculations
  sorry

-- Verify the total jellybean count
theorem total_jellybeans : (Arnold_jellybeans + Lee_jellybeans + Tino_jellybeans + Joshua_jellybeans) = 64 :=
by
  -- Unfold definitions and perform calculations
  sorry

end Tino_has_correct_jellybeans_total_jellybeans_l340_34056


namespace walnuts_amount_l340_34026

theorem walnuts_amount (w : ℝ) (total_nuts : ℝ) (almonds : ℝ) (h1 : total_nuts = 0.5) (h2 : almonds = 0.25) (h3 : w + almonds = total_nuts) : w = 0.25 :=
by
  sorry

end walnuts_amount_l340_34026


namespace fifth_segment_student_l340_34012

variable (N : ℕ) (n : ℕ) (second_segment_student : ℕ)

def sampling_interval (N n : ℕ) : ℕ := N / n

def initial_student (second_segment_student interval : ℕ) : ℕ := second_segment_student - interval

def student_number (initial_student interval : ℕ) (segment : ℕ) : ℕ :=
  initial_student + (segment - 1) * interval

theorem fifth_segment_student (N n : ℕ) (second_segment_student : ℕ) (hN : N = 700) (hn : n = 50) (hsecond : second_segment_student = 20) :
  student_number (initial_student second_segment_student (sampling_interval N n)) (sampling_interval N n) 5 = 62 := by
  sorry

end fifth_segment_student_l340_34012


namespace sum_of_squares_of_roots_l340_34072

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 10) (h2 : s₁ * s₂ = 9) : 
  s₁^2 + s₂^2 = 82 := by
  sorry

end sum_of_squares_of_roots_l340_34072


namespace gilbert_herb_plants_count_l340_34078

variable (initial_basil : Nat) (initial_parsley : Nat) (initial_mint : Nat)
variable (dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool)

def total_initial_plants (initial_basil initial_parsley initial_mint : Nat) : Nat :=
  initial_basil + initial_parsley + initial_mint

def total_plants_after_dropping_seeds (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) : Nat :=
  total_initial_plants initial_basil initial_parsley initial_mint + dropped_basil_seeds

def total_plants_after_rabbit (initial_basil initial_parsley initial_mint dropped_basil_seeds : Nat) (rabbit_ate_all_mint : Bool) : Nat :=
  if rabbit_ate_all_mint then 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds - initial_mint 
  else 
    total_plants_after_dropping_seeds initial_basil initial_parsley initial_mint dropped_basil_seeds

theorem gilbert_herb_plants_count
  (h1 : initial_basil = 3)
  (h2 : initial_parsley = 1)
  (h3 : initial_mint = 2)
  (h4 : dropped_basil_seeds = 1)
  (h5 : rabbit_ate_all_mint = true) :
  total_plants_after_rabbit initial_basil initial_parsley initial_mint dropped_basil_seeds rabbit_ate_all_mint = 5 := by
  sorry

end gilbert_herb_plants_count_l340_34078


namespace find_g1_gneg1_l340_34087

variables {f g : ℝ → ℝ}

theorem find_g1_gneg1 (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
                      (h2 : f (-2) = f 1 ∧ f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end find_g1_gneg1_l340_34087


namespace sum_numbers_l340_34029

theorem sum_numbers :
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end sum_numbers_l340_34029


namespace platform_length_l340_34041

theorem platform_length (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) (V : ℕ) (L : ℕ)
  (h_train_length : train_length = 500)
  (h_pole_time : pole_time = 50)
  (h_platform_time : platform_time = 100)
  (h_speed : V = train_length / pole_time)
  (h_platform_distance : V * platform_time = train_length + L) : 
  L = 500 := 
sorry

end platform_length_l340_34041


namespace smallest_initial_number_l340_34008

theorem smallest_initial_number (N : ℕ) (h₁ : N ≤ 999) (h₂ : 27 * N - 240 ≥ 1000) : N = 46 :=
by {
    sorry
}

end smallest_initial_number_l340_34008


namespace fill_time_first_and_fourth_taps_l340_34086

noncomputable def pool_filling_time (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) : ℝ :=
  m / (x + u)

theorem fill_time_first_and_fourth_taps (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) :
  pool_filling_time m x y z u h₁ h₂ h₃ = 12 / 5 :=
sorry

end fill_time_first_and_fourth_taps_l340_34086


namespace factorize_expression_l340_34058

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l340_34058


namespace z_neq_5_for_every_k_l340_34018

theorem z_neq_5_for_every_k (z : ℕ) (h₁ : z = 5) :
  ¬ (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (∃ m, n ^ 9 % 10 ^ k = z * (10 ^ m))) :=
by
  intro h
  sorry

end z_neq_5_for_every_k_l340_34018


namespace jack_received_more_emails_l340_34020

-- Definitions representing the conditions
def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

-- The theorem statement
theorem jack_received_more_emails : afternoon_emails - morning_emails = 2 := 
by 
  sorry

end jack_received_more_emails_l340_34020


namespace question_1_question_2_question_3_l340_34059

theorem question_1 (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) < 1) ↔ 
    m < (1 - 2 * Real.sqrt 7) / 3 := sorry

theorem question_2 (m : ℝ) : 
  ∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ (m + 1) * x := sorry

theorem question_3 (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
    m ≥ 1 := sorry

end question_1_question_2_question_3_l340_34059


namespace gcd_proof_l340_34031

noncomputable def gcd_problem : Prop :=
  let a := 765432
  let b := 654321
  Nat.gcd a b = 111111

theorem gcd_proof : gcd_problem := by
  sorry

end gcd_proof_l340_34031


namespace max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l340_34033

open Real

theorem max_sum_sqrt_expr (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h_sum : a + b + c = 8) :
  sqrt (3 * a^2 + 1) + sqrt (3 * b^2 + 1) + sqrt (3 * c^2 + 1) ≤ sqrt 201 :=
  sorry

theorem max_sum_sqrt_expr_attained : sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) + sqrt (3 * (8/3)^2 + 1) = sqrt 201 :=
  sorry

end max_sum_sqrt_expr_max_sum_sqrt_expr_attained_l340_34033


namespace length_of_AX_l340_34030

theorem length_of_AX 
  (A B C X : Type) 
  (AB AC BC AX BX : ℕ) 
  (hx : AX + BX = AB)
  (h_angle_bisector : AC * BX = BC * AX)
  (h_AB : AB = 40)
  (h_BC : BC = 35)
  (h_AC : AC = 21) : 
  AX = 15 :=
by
  sorry

end length_of_AX_l340_34030


namespace y_increase_by_20_l340_34066

-- Define the conditions
def relationship (Δx Δy : ℕ) : Prop :=
  Δy = (11 * Δx) / 5

-- The proof problem statement
theorem y_increase_by_20 : relationship 5 11 → relationship 20 44 :=
by
  intros h
  sorry

end y_increase_by_20_l340_34066


namespace fraction_subtraction_l340_34034

theorem fraction_subtraction (a : ℝ) (h : a ≠ 0) : 1 / a - 3 / a = -2 / a := 
by
  sorry

end fraction_subtraction_l340_34034


namespace total_cost_is_correct_l340_34044

-- Define the costs as constants
def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52

-- Assert that the total cost is correct
theorem total_cost_is_correct : marbles_cost + football_cost + baseball_cost = 20.52 :=
by sorry

end total_cost_is_correct_l340_34044


namespace percentage_increase_first_to_second_l340_34028

theorem percentage_increase_first_to_second (D1 D2 D3 : ℕ) (h1 : D2 = 12)
  (h2 : D3 = D2 + Nat.div (D2 * 25) 100) (h3 : D1 + D2 + D3 = 37) :
  Nat.div ((D2 - D1) * 100) D1 = 20 := by
  sorry

end percentage_increase_first_to_second_l340_34028


namespace sum_m_n_l340_34082

theorem sum_m_n (m n : ℤ) (h1 : m^2 - n^2 = 18) (h2 : m - n = 9) : m + n = 2 := 
by
  sorry

end sum_m_n_l340_34082


namespace find_k_value_l340_34004

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l340_34004


namespace relationship_y1_y2_l340_34074

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l340_34074


namespace determinant_tan_matrix_l340_34052

theorem determinant_tan_matrix (B C : ℝ) (h : B + C = 3 * π / 4) :
  Matrix.det ![
    ![Real.tan (π / 4), 1, 1],
    ![1, Real.tan B, 1],
    ![1, 1, Real.tan C]
  ] = 1 :=
by
  sorry

end determinant_tan_matrix_l340_34052


namespace students_with_two_skills_l340_34050

theorem students_with_two_skills :
  ∀ (n_students n_chess n_puzzles n_code : ℕ),
  n_students = 120 →
  n_chess = n_students - 50 →
  n_puzzles = n_students - 75 →
  n_code = n_students - 40 →
  (n_chess + n_puzzles + n_code - n_students) = 75 :=
by 
  sorry

end students_with_two_skills_l340_34050


namespace solve_for_a_l340_34069

theorem solve_for_a (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5 * a * x + a) = x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a) →
  (1 - 5 * a = 0) →
  a = 1 / 5 := 
by
  intro h₁ h₂
  sorry

end solve_for_a_l340_34069


namespace four_digit_number_conditions_l340_34016

-- Define the needed values based on the problem conditions
def first_digit := 1
def second_digit := 3
def third_digit := 4
def last_digit := 9

def number := 1349

-- State the theorem
theorem four_digit_number_conditions :
  (second_digit = 3 * first_digit) ∧ 
  (last_digit = 3 * second_digit) ∧ 
  (number = 1349) :=
by
  -- This is where the proof would go
  sorry

end four_digit_number_conditions_l340_34016


namespace study_group_members_l340_34089

theorem study_group_members (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end study_group_members_l340_34089


namespace correct_weight_misread_l340_34075

theorem correct_weight_misread : 
  ∀ (x : ℝ) (n : ℝ) (avg1 : ℝ) (avg2 : ℝ) (misread : ℝ),
  n = 20 → avg1 = 58.4 → avg2 = 59 → misread = 56 → 
  (n * avg2 - n * avg1 + misread) = x → 
  x = 68 :=
by
  intros x n avg1 avg2 misread
  intros h1 h2 h3 h4 h5
  sorry

end correct_weight_misread_l340_34075


namespace compare_logs_l340_34051

noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8
noncomputable def c : ℝ := 1 / 2

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l340_34051


namespace no_real_roots_iff_k_lt_neg_one_l340_34048

theorem no_real_roots_iff_k_lt_neg_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_iff_k_lt_neg_one_l340_34048


namespace num_triangles_with_perimeter_9_l340_34010

theorem num_triangles_with_perimeter_9 : 
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 6 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 9 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ a ≤ b ∧ b ≤ c) := 
sorry

end num_triangles_with_perimeter_9_l340_34010


namespace minimize_y_at_x_l340_34049

-- Define the function y
def y (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 

-- State the theorem
theorem minimize_y_at_x (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x' a b ≥ y ((3 * a + b) / 4) a b) :=
by
  sorry

end minimize_y_at_x_l340_34049


namespace solve_inequality_l340_34092

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 2 * x - 3) * (x ^ 2 - 4 * x + 4) < 0 ↔ (-1 < x ∧ x < 3 ∧ x ≠ 2) := by
  sorry

end solve_inequality_l340_34092


namespace vincent_total_laundry_loads_l340_34060

theorem vincent_total_laundry_loads :
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  total_loads = 26 :=
by {
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  show total_loads = 26
  sorry
}

end vincent_total_laundry_loads_l340_34060


namespace neg_exists_n_sq_gt_two_pow_n_l340_34039

open Classical

theorem neg_exists_n_sq_gt_two_pow_n :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end neg_exists_n_sq_gt_two_pow_n_l340_34039


namespace scenario_a_scenario_b_l340_34063

-- Define the chessboard and the removal function
def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def is_square (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

-- Define a Hamiltonian path on the chessboard
inductive HamiltonianPath : (ℕ × ℕ) → (ℕ → (ℕ × ℕ)) → ℕ → Prop
| empty : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)), HamiltonianPath start path 0
| step : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)) (n : ℕ),
    is_adjacent (path n).1 (path n).2 (path (n+1)).1 (path (n+1)).2 →
    HamiltonianPath start path n →
    (is_square (path (n + 1)).1 (path (n + 1)).2 ∧ ¬ (∃ m < n + 1, path m = path (n + 1))) →
    HamiltonianPath start path (n + 1)

-- State the main theorems
theorem scenario_a : 
  ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 62 ∧
    (∀ n, path n ≠ (2, 2)) := sorry

theorem scenario_b :
  ¬ ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 61 ∧
    (∀ n, path n ≠ (2, 2) ∧ path n ≠ (7, 7)) := sorry

end scenario_a_scenario_b_l340_34063


namespace initial_earning_members_l340_34019

theorem initial_earning_members (n T : ℕ)
  (h₁ : T = n * 782)
  (h₂ : T - 1178 = (n - 1) * 650) :
  n = 14 :=
by sorry

end initial_earning_members_l340_34019


namespace events_per_coach_l340_34077

theorem events_per_coach {students events_per_student coaches events total_participations total_events : ℕ} 
  (h1 : students = 480) 
  (h2 : events_per_student = 4) 
  (h3 : (students * events_per_student) = total_participations) 
  (h4 : ¬ students * events_per_student ≠ total_participations)
  (h5 : total_participations = 1920) 
  (h6 : (total_participations / 20) = total_events) 
  (h7 : ¬ total_participations / 20 ≠ total_events)
  (h8 : total_events = 96)
  (h9 : coaches = 16) :
  (total_events / coaches) = 6 := sorry

end events_per_coach_l340_34077


namespace fibonacci_periodicity_l340_34035

-- Definitions for p-arithmetic and Fibonacci sequence
def is_prime (p : ℕ) := Nat.Prime p
def sqrt_5_extractable (p : ℕ) : Prop := ∃ k : ℕ, p = 5 * k + 1 ∨ p = 5 * k - 1

-- Definitions of Fibonacci sequences and properties
def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci n + fibonacci (n + 1)

-- Main theorem
theorem fibonacci_periodicity (p : ℕ) (r : ℕ) (h_prime : is_prime p) (h_not_2_or_5 : p ≠ 2 ∧ p ≠ 5)
    (h_period : r = (p+1) ∨ r = (p-1)) (h_div : (sqrt_5_extractable p → r ∣ (p - 1)) ∧ (¬ sqrt_5_extractable p → r ∣ (p + 1)))
    : (fibonacci (p+1) % p = 0 ∨ fibonacci (p-1) % p = 0) := by
          sorry

end fibonacci_periodicity_l340_34035


namespace doves_eggs_l340_34038

theorem doves_eggs (initial_doves total_doves : ℕ) (fraction_hatched : ℚ) (E : ℕ)
  (h_initial_doves : initial_doves = 20)
  (h_total_doves : total_doves = 65)
  (h_fraction_hatched : fraction_hatched = 3/4)
  (h_after_hatching : total_doves = initial_doves + fraction_hatched * E * initial_doves) :
  E = 3 :=
by
  -- The proof would go here.
  sorry

end doves_eggs_l340_34038


namespace min_cos_C_l340_34095

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = (5 / 2) * c^2) 
  (h2 : ∃ (A B C : ℝ), a ≠ b ∧ 
    c = (a ^ 2 + b ^ 2 - 2 * a * b * (Real.cos C))) : 
  ∃ (C : ℝ), Real.cos C = 3 / 5 :=
by
  sorry

end min_cos_C_l340_34095


namespace family_total_weight_gain_l340_34079

def orlando_gain : ℕ := 5
def jose_gain : ℕ := 2 * orlando_gain + 2
def fernando_gain : ℕ := (jose_gain / 2) - 3
def total_weight_gain : ℕ := orlando_gain + jose_gain + fernando_gain

theorem family_total_weight_gain : total_weight_gain = 20 := by
  -- proof omitted
  sorry

end family_total_weight_gain_l340_34079


namespace find_k_l340_34007

theorem find_k
  (k x1 x2 : ℝ)
  (h1 : x1^2 - 3*x1 + k = 0)
  (h2 : x2^2 - 3*x2 + k = 0)
  (h3 : x1 = 2 * x2) :
  k = 2 :=
sorry

end find_k_l340_34007


namespace find_m_l340_34085

def triangle (x y : ℤ) := x * y + x + y

theorem find_m (m : ℤ) (h : triangle 2 m = -16) : m = -6 :=
by
  sorry

end find_m_l340_34085


namespace four_digit_perfect_square_is_1156_l340_34022

theorem four_digit_perfect_square_is_1156 :
  ∃ (N : ℕ), (N ≥ 1000) ∧ (N < 10000) ∧ (∀ a, a ∈ [N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10] → a < 7) 
              ∧ (∃ n : ℕ, N = n * n) ∧ (∃ m : ℕ, (N + 3333 = m * m)) ∧ (N = 1156) :=
by
  sorry

end four_digit_perfect_square_is_1156_l340_34022


namespace arithmetic_sequence_a1_value_l340_34071

theorem arithmetic_sequence_a1_value (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 3 = -6) 
  (h2 : a 7 = a 5 + 4) 
  (h_seq : ∀ n, a (n+1) = a n + d) : 
  a 1 = -10 := 
by
  sorry

end arithmetic_sequence_a1_value_l340_34071


namespace find_x_l340_34011

theorem find_x (a x : ℤ) (h1 : -6 * a^2 = x * (4 * a + 2)) (h2 : a = 1) : x = -1 :=
sorry

end find_x_l340_34011


namespace jason_cuts_lawns_l340_34042

theorem jason_cuts_lawns 
  (time_per_lawn: ℕ)
  (total_cutting_time_hours: ℕ)
  (total_cutting_time_minutes: ℕ)
  (total_yards_cut: ℕ) : 
  time_per_lawn = 30 → 
  total_cutting_time_hours = 8 → 
  total_cutting_time_minutes = total_cutting_time_hours * 60 → 
  total_yards_cut = total_cutting_time_minutes / time_per_lawn → 
  total_yards_cut = 16 :=
by
  intros
  sorry

end jason_cuts_lawns_l340_34042


namespace arithmetic_sequence_squares_l340_34083

theorem arithmetic_sequence_squares (a b c : ℝ) :
  (1 / (a + b) - 1 / (b + c) = 1 / (c + a) - 1 / (b + c)) →
  (2 * b^2 = a^2 + c^2) :=
by
  intro h
  sorry

end arithmetic_sequence_squares_l340_34083


namespace proof_problem_l340_34036

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem proof_problem (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : f x * f (-x) = 1 := 
by 
  sorry

end proof_problem_l340_34036


namespace michael_exceeds_suresh_by_36_5_l340_34067

noncomputable def shares_total : ℝ := 730
noncomputable def punith_ratio_to_michael : ℝ := 3 / 4
noncomputable def michael_ratio_to_suresh : ℝ := 3.5 / 3

theorem michael_exceeds_suresh_by_36_5 :
  ∃ P M S : ℝ, P + M + S = shares_total
  ∧ (P / M = punith_ratio_to_michael)
  ∧ (M / S = michael_ratio_to_suresh)
  ∧ (M - S = 36.5) :=
by
  sorry

end michael_exceeds_suresh_by_36_5_l340_34067


namespace avg_of_last_11_eq_41_l340_34068

def sum_of_first_11 : ℕ := 11 * 48
def sum_of_all_21 : ℕ := 21 * 44
def eleventh_number : ℕ := 55

theorem avg_of_last_11_eq_41 (S1 S : ℕ) :
  S1 = sum_of_first_11 →
  S = sum_of_all_21 →
  (S - S1 + eleventh_number) / 11 = 41 :=
by
  sorry

end avg_of_last_11_eq_41_l340_34068


namespace base_4_last_digit_of_389_l340_34099

theorem base_4_last_digit_of_389 : (389 % 4) = 1 :=
by {
  sorry
}

end base_4_last_digit_of_389_l340_34099


namespace final_result_l340_34017

noncomputable def f : ℝ → ℝ := sorry
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (3 + x) = f x
axiom f_half_periodic : ∀ x : ℝ, f (3 / 2 - x) = f x
axiom f_value_neg2 : f (-2) = -3

axiom a1_value : a 1 = -1
axiom S_n : ∀ n : ℕ, S n = 2 * a n + n

theorem final_result : f (a 5) + f (a 6) = 3 :=
sorry

end final_result_l340_34017


namespace fifth_element_is_17_l340_34046

-- Define the sequence pattern based on given conditions
def seq : ℕ → ℤ 
| 0 => 5    -- first element
| 1 => -8   -- second element
| n + 2 => seq n + 3    -- each following element is calculated by adding 3 to the two positions before

-- Additional condition: the sign of sequence based on position
def seq_sign : ℕ → ℤ
| n => if n % 2 = 0 then 1 else -1

-- The final adjusted sequence based on the above observations
def final_seq (n : ℕ) : ℤ := seq n * seq_sign n

-- Assert the expected outcome for the 5th element
theorem fifth_element_is_17 : final_seq 4 = 17 :=
by
  sorry

end fifth_element_is_17_l340_34046


namespace jane_exercises_per_day_l340_34054

-- Conditions
variables (total_hours : ℕ) (total_weeks : ℕ) (days_per_week : ℕ)
variable (goal_achieved : total_hours = 40 ∧ total_weeks = 8 ∧ days_per_week = 5)

-- Statement
theorem jane_exercises_per_day : ∃ hours_per_day : ℕ, hours_per_day = (total_hours / total_weeks) / days_per_week :=
by
  sorry

end jane_exercises_per_day_l340_34054


namespace negation_of_existential_statement_l340_34073

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement_l340_34073


namespace length_of_LM_l340_34000

-- Definitions of the conditions
variable (P Q R L M : Type)
variable (b : Real) (PR_area : Real) (PR_base : Real)
variable (PR_base_eq : PR_base = 15)
variable (crease_parallel : Parallel L M)
variable (projected_area_fraction : Real)
variable (projected_area_fraction_eq : projected_area_fraction = 0.25 * PR_area)

-- Theorem statement to prove the length of LM
theorem length_of_LM : ∀ (LM_length : Real), (LM_length = 7.5) :=
sorry

end length_of_LM_l340_34000


namespace highest_probability_face_l340_34037

theorem highest_probability_face :
  let faces := 6
  let face_3 := 3
  let face_2 := 2
  let face_1 := 1
  (face_3 / faces > face_2 / faces) ∧ (face_2 / faces > face_1 / faces) →
  (face_3 / faces > face_1 / faces) →
  (face_3 = 3) :=
by {
  sorry
}

end highest_probability_face_l340_34037


namespace find_x_in_sequence_l340_34005

theorem find_x_in_sequence :
  ∃ x y z : ℤ, 
    (z - 1 = 0) ∧ (y - z = -1) ∧ (x - y = 1) ∧ x = 1 :=
by
  sorry

end find_x_in_sequence_l340_34005


namespace distance_between_x_intercepts_l340_34014

theorem distance_between_x_intercepts 
  (s1 s2 : ℝ) (P : ℝ × ℝ)
  (h1 : s1 = 2) 
  (h2 : s2 = -4) 
  (hP : P = (8, 20)) :
  let l1_x_intercept := (0 - (20 - P.2)) / s1 + P.1
  let l2_x_intercept := (0 - (20 - P.2)) / s2 + P.1
  abs (l1_x_intercept - l2_x_intercept) = 15 := 
sorry

end distance_between_x_intercepts_l340_34014


namespace common_number_is_eight_l340_34040

theorem common_number_is_eight (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 7)
  (h2 : (d + e + f + g) / 4 = 9)
  (h3 : (a + b + c + d + e + f + g) / 7 = 8) :
  d = 8 :=
by
sorry

end common_number_is_eight_l340_34040


namespace small_circle_to_large_circle_ratio_l340_34062

theorem small_circle_to_large_circle_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 3 * π * a^2) :
  a / b = 1 / 2 :=
sorry

end small_circle_to_large_circle_ratio_l340_34062


namespace train_passes_man_in_15_seconds_l340_34027

theorem train_passes_man_in_15_seconds
  (length_of_train : ℝ)
  (speed_of_train : ℝ)
  (speed_of_man : ℝ)
  (direction_opposite : Bool)
  (h1 : length_of_train = 275)
  (h2 : speed_of_train = 60)
  (h3 : speed_of_man = 6)
  (h4 : direction_opposite = true) : 
  ∃ t : ℝ, t = 15 :=
by
  sorry

end train_passes_man_in_15_seconds_l340_34027


namespace find_a_2018_l340_34094

noncomputable def a : ℕ → ℕ
| n => if n > 0 then 2 * n else sorry

theorem find_a_2018 (a : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 ∧ n > 0 → a m + a n = a (m + n)) 
  (h1 : a 1 = 2) : a 2018 = 4036 := by
  sorry

end find_a_2018_l340_34094


namespace city_council_vote_l340_34024

theorem city_council_vote :
  ∀ (x y x' y' m : ℕ),
    x + y = 350 →
    y > x →
    y - x = m →
    x' - y' = 2 * m →
    x' + y' = 350 →
    x' = (10 * y) / 9 →
    x' - x = 66 :=
by
  intros x y x' y' m h1 h2 h3 h4 h5 h6
  -- proof goes here
  sorry

end city_council_vote_l340_34024
