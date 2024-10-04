import Mathlib

namespace jason_books_is_21_l21_21162

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l21_21162


namespace question_l21_21434

variable (a : ℝ)

def condition_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3

def condition_q (a : ℝ) : Prop := ∀ (x y : ℝ) , x > y → (5 - 2 * a)^x < (5 - 2 * a)^y

theorem question (h1 : condition_p a ∨ condition_q a)
                (h2 : ¬ (condition_p a ∧ condition_q a)) : a = 2 ∨ a ≥ 5 / 2 :=
sorry

end question_l21_21434


namespace primes_less_than_200_with_3_as_ones_digit_are_12_l21_21081

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_with_digit_three_and_less_than_200 : List ℕ :=
  [3, 13, 23, 43, 53, 73, 83, 103, 113, 163, 173, 193]

theorem primes_less_than_200_with_3_as_ones_digit_are_12 :
  (primes_with_digit_three_and_less_than_200.filter is_prime).length = 12 := by
  sorry

end primes_less_than_200_with_3_as_ones_digit_are_12_l21_21081


namespace condition1_not_sufficient_nor_necessary_condition2_necessary_l21_21397

variable (x y : ℝ)

-- ① Neither sufficient nor necessary
theorem condition1_not_sufficient_nor_necessary (h1 : x ≠ 1 ∧ y ≠ 2) : ¬ ((x ≠ 1 ∧ y ≠ 2) → x + y ≠ 3) ∧ ¬ (x + y ≠ 3 → x ≠ 1 ∧ y ≠ 2) := sorry

-- ② Necessary condition
theorem condition2_necessary (h2 : x ≠ 1 ∨ y ≠ 2) : x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2) := sorry

end condition1_not_sufficient_nor_necessary_condition2_necessary_l21_21397


namespace probability_BC_same_activity_l21_21679

theorem probability_BC_same_activity
  (s : Finset String) (A B C D : String)
  (contains_A : A ∈ s) (contains_B : B ∈ s) (contains_C : C ∈ s) (contains_D : D ∈ s)
  (card_s : s.card = 4) :
  let groups := s.powerset.filter (λ t, t.card = 2) in
  let desired_groups := groups.filter (λ t, B ∈ t ∧ C ∈ t) in
  (desired_groups.card:ℚ) / groups.card = 1 / 3 :=
by sorry

end probability_BC_same_activity_l21_21679


namespace find_length_of_BC_l21_21943

-- Define the geometrical setup and the problem statement
noncomputable def length_of_BC (AB CD : ℝ) (angle_AED : ℝ) (equilateral_BCE : Prop) : ℝ :=
  let x := BC in
  if h1 : AB = 4 ∧ CD = 16 ∧ angle_AED = 120 ∧ equilateral_BCE then
    8
  else
    0

-- Condition for angle measure in degrees
def is_degrees (angle : ℝ) : Prop := 
  ∃ (n : ℕ), angle = n * 60

-- Equilateral triangle property
def is_equilateral (BC BE EC : ℝ) : Prop :=
  BC = BE ∧ BE = EC

-- Problem statement as a theorem
theorem find_length_of_BC (AB CD : ℝ) (angle_AED : ℝ) :
  AB = 4 ∧ CD = 16 ∧ angle_AED = 120 ∧ (is_equilateral BC BE EC) → BC = 8 :=
by
  intro h
  have h_AB : AB = 4 := h.left.left.left
  have h_CD : CD = 16 := h.left.left.right
  have h_angle : angle_AED = 120 := h.left.right
  have h_equilateral : is_equilateral BC BE EC := h.right
  exact sorry

end find_length_of_BC_l21_21943


namespace line_equation_with_equal_intercepts_l21_21874

theorem line_equation_with_equal_intercepts 
  (a : ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h : ∀ x y, l x y ↔ (a+1)*x + y + 2 - a = 0) 
  (intercept_condition : ∀ x y, l x 0 = l 0 y) : 
  (∀ x y, l x y ↔ x + y + 2 = 0) ∨ (∀ x y, l x y ↔ 3*x + y = 0) :=
sorry

end line_equation_with_equal_intercepts_l21_21874


namespace problem_part1_l21_21882

def f (x : ℝ) (a : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * (x ^ 2 - 2 * x)

theorem problem_part1 : f 2 Real.exp = 0 := by sorry

end problem_part1_l21_21882


namespace total_recovery_time_correct_l21_21956

def initial_healing_time : ℕ := 4
def healing_factor : ℚ := 0.5

def graft_healing_time (t₁ : ℕ) (f : ℚ) : ℕ :=
  t₁ + (f * t₁).toNat

def total_recovery_time (t₁ : ℕ) (t₂ : ℕ) : ℕ :=
  t₁ + t₂

theorem total_recovery_time_correct : total_recovery_time initial_healing_time (graft_healing_time initial_healing_time healing_factor) = 10 :=
by
  sorry

end total_recovery_time_correct_l21_21956


namespace count_integers_between_cubes_l21_21476

theorem count_integers_between_cubes : 
  let a := 10
  let b1 := 0.4
  let b2 := 0.5
  let cube1 := a^3 + 3 * a^2 * b1 + 3 * a * (b1^2) + b1^3
  let cube2 := a^3 + 3 * a^2 * b2 + 3 * a * (b2^2) + b2^3
  ∃ n : ℤ, n = 33 ∧ ((⌈cube1⌉.val ≤ n.val) ∧ (n.val ≤ ⌊cube2⌋.val)) :=
sorry

end count_integers_between_cubes_l21_21476


namespace distance_A1_to_plane_l21_21412

-- Define the rectangular prism and its vertices
structure RectangularPrism :=
(AB AD AA1 : ℝ)
(volume_of_pyramid : ℝ)
(area_of_triangle : ℝ)

-- Given conditions
def prism : RectangularPrism := {
  AB := 4,
  AD := 4,
  AA1 := 2,
  volume_of_pyramid := 16 / 3,
  area_of_triangle := 4 * Real.sqrt 6
}

-- Mathematical statement to prove
theorem distance_A1_to_plane (p : RectangularPrism) : 
  (1 / 3) * p.area_of_triangle * (2 * Real.sqrt 6 / 3) = p.volume_of_pyramid := 
by
  sorry

end distance_A1_to_plane_l21_21412


namespace Julio_mocktails_days_l21_21963

theorem Julio_mocktails_days 
    (lime_juice_per_mocktail: ℕ)
    (lime_juice_per_lime: ℕ)
    (limes_per_dollar: ℕ)
    (dollars_spent: ℕ)
    (lime_juice: ℕ)
    (total_mocktails: ℕ) : 
    (lime_juice_per_mocktail = 1) →
    (lime_juice_per_lime = 2) →
    (limes_per_dollar = 3) →
    (dollars_spent = 5) →
    (lime_juice = (dollars_spent * limes_per_dollar * lime_juice_per_lime)) →
    (total_mocktails = (lime_juice / lime_juice_per_mocktail)) →
    total_mocktails = 30 := 
begin
    -- Proof goes here
    sorry
end

end Julio_mocktails_days_l21_21963


namespace find_value_of_expression_l21_21843

theorem find_value_of_expression (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) : 2 * a + 2 * b - 3 * (a * b) = 9 :=
by
  sorry

end find_value_of_expression_l21_21843


namespace pass_rate_l21_21741

variable (α β : Type)
variable (a b : α)
variable [Noncomputable (1 - a) : α] [Noncomputable (1 - b) : α]

theorem pass_rate (a b : α) [Ha : has_zero α] [Hb : has_zero β] (h_def_rate_a : has_rate a) (h_def_rate_b : has_rate b) (h_indep: independent a b):
  pass_rate = (1 - a) * (1 - b) :=
by
  sorry

end pass_rate_l21_21741


namespace cot_difference_abs_eq_sqrt3_l21_21135

theorem cot_difference_abs_eq_sqrt3 
  (A B C D P : Point) (x y : ℝ) (h1 : is_triangle A B C) 
  (h2 : is_median A D B C) (h3 : ∠(D, A, P) = 60)
  (BD_eq_CD : BD = x) (CD_eq_x : CD = x)
  (BP_eq_y : BP = y) (AP_eq_sqrt3 : AP = sqrt(3) * (x + y))
  (cot_B : cot B = -y / ((sqrt 3) * (x + y)))
  (cot_C : cot C = (2 * x + y) / (sqrt 3 * (x + y))) 
  (x_y_neq_zero : x + y ≠ 0) :
  abs (cot B - cot C) = sqrt 3
  := sorry

end cot_difference_abs_eq_sqrt3_l21_21135


namespace number_of_distinguishable_arrangements_l21_21079

-- Define the conditions
def num_blue_tiles : Nat := 1
def num_red_tiles : Nat := 2
def num_green_tiles : Nat := 3
def num_yellow_tiles : Nat := 2
def total_tiles : Nat := num_blue_tiles + num_red_tiles + num_green_tiles + num_yellow_tiles

-- The goal is to prove the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  (Nat.factorial total_tiles) / ((Nat.factorial num_green_tiles) * 
                                (Nat.factorial num_red_tiles) * 
                                (Nat.factorial num_yellow_tiles) * 
                                (Nat.factorial num_blue_tiles)) = 1680 := by
  sorry

end number_of_distinguishable_arrangements_l21_21079


namespace limit_of_na_n_l21_21798

noncomputable def L (x : ℝ) : ℝ := x - (x^2 / 2)

def a_n (n : ℕ) : ℝ :=
  let iter (m : ℕ) (x : ℝ) : ℝ := nat.iterate L m x
  iter n (17 / n)

theorem limit_of_na_n : (tendsto (λ n : ℕ, n * a_n n) at_top (𝓝 (34 / 19))) :=
sorry

end limit_of_na_n_l21_21798


namespace range_of_m_l21_21072

noncomputable def A : Set ℝ := {y | ∃ x > 0, y = 1 / x}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (2 * x - 4)}

theorem range_of_m (m : ℝ) (h1 : m ∈ A) (h2 : m ∉ B) : m ∈ Ioc 0 2 := 
by 
  sorry

end range_of_m_l21_21072


namespace sock_probability_l21_21896

open Nat

-- Definitions of the conditions
def total_socks : ℕ := 10
def sock_colors : ℕ := 5
def draw_socks : ℕ := 5
def pairs_per_color : ℕ := 2

-- Number of ways to draw five socks from ten
noncomputable def total_ways := choose total_socks draw_socks

-- Number of ways to choose four out of five colors
noncomputable def choose_colors := choose sock_colors (sock_colors - 1)

-- Number of ways to choose which one of the four colors will form the pair
def choose_pair := sock_colors - 1

-- The number of ways to form socks pair from the same color
def pair_way := 1

-- The number of ways to choose the rest of the socks from different colors
noncomputable def choose_rest := pow pairs_per_color (draw_socks - pairs_per_color)

-- Calculate the total favorable outcomes
noncomputable def favorable_ways := choose_colors * choose_pair * pair_way * choose_rest

-- Calculate the probability
noncomputable def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

-- The main theorem to prove
theorem sock_probability : probability = 20 / 21 := by
  sorry

end sock_probability_l21_21896


namespace david_age_l21_21360

theorem david_age (A B C D : ℕ)
  (h1 : A = B - 5)
  (h2 : B = C + 2)
  (h3 : D = C + 4)
  (h4 : A = 12) : D = 19 :=
sorry

end david_age_l21_21360


namespace mass_percentage_O_is_correct_l21_21402

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_Al_atoms : ℕ := 2
noncomputable def num_O_atoms : ℕ := 3

noncomputable def molar_mass_Al2O3 : ℝ :=
  (num_Al_atoms * molar_mass_Al) + (num_O_atoms * molar_mass_O)

noncomputable def mass_percentage_O_in_Al2O3 : ℝ :=
  ((num_O_atoms * molar_mass_O) / molar_mass_Al2O3) * 100

theorem mass_percentage_O_is_correct :
  mass_percentage_O_in_Al2O3 = 47.07 :=
by
  sorry

end mass_percentage_O_is_correct_l21_21402


namespace isabella_initial_hair_length_l21_21157

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l21_21157


namespace proof_f_second_derivatives_l21_21868

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x * f''(2)

-- Define the second derivative of f as f''
noncomputable def f'' (x : ℝ) : ℝ := 6 * x + 2 * f''(2)
 
theorem proof_f_second_derivatives : f''(5) + f''(2) = -6 := by
  -- proof goes here
  sorry

end proof_f_second_derivatives_l21_21868


namespace blood_expiry_date_l21_21750

theorem blood_expiry_date (expiration_seconds : ℕ) (donation_day : String := "January 5") :
  expiration_seconds = 5040 → donation_day = "January 5" → 
  ∃ t, t = "January 5" :=
begin
  intros h1 h2,
  use donation_day,
  split,
  { exact h2 },
  { sorry }
end

end blood_expiry_date_l21_21750


namespace find_cot_difference_l21_21139

-- Define necessary elements for the problem
variable {A B C D : Type}
variable [EuclideanGeometry A]
variables (ABC : Triangle A B C)

-- Define the condition where median AD makes an angle of 60 degrees with BC
variable (ADmedian : median A B C D ∧ angle D A B = 60)

theorem find_cot_difference:
  |cot (angle B) - cot (angle C)| = 2 :=
sorry

end find_cot_difference_l21_21139


namespace range_of_a_l21_21466

-- Definitions for the parametric equations of the line and the circle
def line_parametric (a t : ℝ) : ℝ × ℝ := (a - 2 * t, -4 * t)
def circle_parametric (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 4 * Real.sin θ)

-- Definition of the standard equations
def line_standard_eq (a x y : ℝ) : Prop := 2 * x - y - 2 * a = 0
def circle_standard_eq (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 16

-- Proof problem statement
theorem range_of_a (a : ℝ) : 
  (∃ t θ, line_parametric a t = circle_parametric θ) →
  -2 * Real.sqrt 5 ≤ a ∧ a ≤ 2 * Real.sqrt 5 :=
sorry

end range_of_a_l21_21466


namespace union_M_N_eq_M_l21_21589

noncomputable def M : set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
noncomputable def N : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≠ 0}

theorem union_M_N_eq_M : M ∪ N = M := by sorry

end union_M_N_eq_M_l21_21589


namespace ben_time_to_school_l21_21706

/-- Amy's steps per minute -/
def amy_steps_per_minute : ℕ := 80

/-- Length of each of Amy's steps in cm -/
def amy_step_length : ℕ := 70

/-- Time taken by Amy to reach school in minutes -/
def amy_time_to_school : ℕ := 20

/-- Ben's steps per minute -/
def ben_steps_per_minute : ℕ := 120

/-- Length of each of Ben's steps in cm -/
def ben_step_length : ℕ := 50

/-- Given the above conditions, we aim to prove that Ben takes 18 2/3 minutes to reach school. -/
theorem ben_time_to_school : (112000 / 6000 : ℚ) = 18 + 2 / 3 := 
by sorry

end ben_time_to_school_l21_21706


namespace trigonometric_identity_1_trigonometric_identity_2_l21_21232

theorem trigonometric_identity_1 :
  tan (20 * Real.pi / 180) + tan (40 * Real.pi / 180) + sqrt 3 * tan (20 * Real.pi / 180) * tan (40 * Real.pi / 180) = sqrt 3 :=
by
  sorry

theorem trigonometric_identity_2 :
  sin (50 * Real.pi / 180) * (1 + sqrt 3 * tan (10 * Real.pi / 180)) = 1 :=
by
  sorry

end trigonometric_identity_1_trigonometric_identity_2_l21_21232


namespace maximum_eccentricity_is_correct_l21_21892

noncomputable def maximum_eccentricity_of_ellipse 
  (A B : ℝ × ℝ) (P : ℝ × ℝ → Prop) (l : ℝ → ℝ → Prop) 
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hP : ∃ x y : ℝ, P (x, y) ∧ l x y) 
  (hl : ∀ x y : ℝ, l x y ↔ y = x + 3) : ℝ :=
    (2 * real.sqrt 26) / 13

theorem maximum_eccentricity_is_correct 
  (eccentricity : ℝ)
  (h : eccentricity = maximum_eccentricity_of_ellipse (-2, 0) (2, 0) 
        (λ P, P ∈ {(x, y) | x : ℝ ∧ y : ℝ}) 
        (λ x y, y = x + 3) 
        rfl 
        rfl 
        (exists.intro 0 (exists.intro 3 ⟨rfl, rfl⟩))
        (λ x y, iff.rfl)) :
  eccentricity = (2 * real.sqrt 26) / 13 :=
by sorry

end maximum_eccentricity_is_correct_l21_21892


namespace geometric_sequence_k_eq_6_l21_21125

theorem geometric_sequence_k_eq_6 
  (a : ℕ → ℝ) (q : ℝ) (k : ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n = a 1 * q ^ (n - 1))
  (h3 : q ≠ 1)
  (h4 : q ≠ -1)
  (h5 : a k = a 2 * a 5) :
  k = 6 :=
sorry

end geometric_sequence_k_eq_6_l21_21125


namespace count_integers_between_cubes_l21_21477

theorem count_integers_between_cubes : 
  let a := 10
  let b1 := 0.4
  let b2 := 0.5
  let cube1 := a^3 + 3 * a^2 * b1 + 3 * a * (b1^2) + b1^3
  let cube2 := a^3 + 3 * a^2 * b2 + 3 * a * (b2^2) + b2^3
  ∃ n : ℤ, n = 33 ∧ ((⌈cube1⌉.val ≤ n.val) ∧ (n.val ≤ ⌊cube2⌋.val)) :=
sorry

end count_integers_between_cubes_l21_21477


namespace flowers_per_bouquet_l21_21953

-- Defining the problem parameters
def total_flowers : ℕ := 66
def wilted_flowers : ℕ := 10
def num_bouquets : ℕ := 7

-- The goal is to prove that the number of flowers per bouquet is 8
theorem flowers_per_bouquet :
  (total_flowers - wilted_flowers) / num_bouquets = 8 :=
by
  sorry

end flowers_per_bouquet_l21_21953


namespace negation_of_universal_l21_21662

theorem negation_of_universal (P : ∀ x : ℝ, x^2 > 0) : ¬ ( ∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by 
  sorry

end negation_of_universal_l21_21662


namespace total_blood_cells_correct_l21_21753

def first_sample : ℕ := 4221
def second_sample : ℕ := 3120
def total_blood_cells : ℕ := first_sample + second_sample

theorem total_blood_cells_correct : total_blood_cells = 7341 := by
  -- proof goes here
  sorry

end total_blood_cells_correct_l21_21753


namespace min_lights_needed_l21_21472

theorem min_lights_needed (s : ℕ) (h : s = 20) : 
  (s * 4 - 4) = 76 := 
by 
  rw h
  show (20 * 4 - 4) = 76 
  sorry

end min_lights_needed_l21_21472


namespace required_run_rate_equivalence_l21_21944

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.5
def overs_first_phase : ℝ := 10
def total_target_runs : ℝ := 350
def remaining_overs : ℝ := 35
def total_overs : ℝ := 45

-- Define the already scored runs
def runs_scored_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_phase

-- Define the required runs for the remaining overs
def runs_needed : ℝ := total_target_runs - runs_scored_first_10_overs

-- Theorem stating the required run rate in the remaining 35 overs
theorem required_run_rate_equivalence :
  runs_needed / remaining_overs = 9 :=
by
  sorry

end required_run_rate_equivalence_l21_21944


namespace coefficient_of_x3_l21_21447

theorem coefficient_of_x3 
  (n : ℕ)
  (h₁ : (3 + -1)^n = 128)
  : (∃ c, c = 945 ∧ ∃ k, 14 - ((11 * k) / 4) = 3 ∧ (-1)^4 * 3^(7 - k) * choose 7 k = c) :=
by
  have h_value_of_n : n = 7 := by sorry
  have h_value_of_r : k = 4 := by sorry
  use 945
  split
  · sorry
  · use 4
    split
    · sorry
    · sorry


end coefficient_of_x3_l21_21447


namespace cylindrical_to_rectangular_l21_21797

theorem cylindrical_to_rectangular (r θ z : ℝ) 
  (h₁ : r = 7) (h₂ : θ = 5 * Real.pi / 4) (h₃ : z = 6) : 
  (r * Real.cos θ, r * Real.sin θ, z) = 
  (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, 6) := 
by 
  sorry

end cylindrical_to_rectangular_l21_21797


namespace alice_age_l21_21417

theorem alice_age (x : ℕ) (h1 : ∃ n : ℕ, x - 4 = n^2) (h2 : ∃ m : ℕ, x + 2 = m^3) : x = 58 :=
sorry

end alice_age_l21_21417


namespace fountains_fill_pool_together_l21_21210

-- Define the times in hours for each fountain to fill the pool
def time_fountain1 : ℚ := 5 / 2  -- 2.5 hours
def time_fountain2 : ℚ := 15 / 4 -- 3.75 hours

-- Define the rates at which each fountain can fill the pool
def rate_fountain1 : ℚ := 1 / time_fountain1
def rate_fountain2 : ℚ := 1 / time_fountain2

-- Calculate the combined rate
def combined_rate : ℚ := rate_fountain1 + rate_fountain2

-- Define the time for both fountains working together to fill the pool
def combined_time : ℚ := 1 / combined_rate

-- Prove that the combined time is indeed 1.5 hours
theorem fountains_fill_pool_together : combined_time = 3 / 2 := by
  sorry

end fountains_fill_pool_together_l21_21210


namespace shanna_harvests_total_56_l21_21229

variables (tomato_plants_initial : ℕ) (eggplant_plants_initial : ℕ) (pepper_plants_initial : ℕ)
variables (tomato_plants_remaining : ℕ) (pepper_plants_remaining : ℕ)
variables (vegetables_per_plant : ℕ) (total_vegetables_harvested : ℕ)

-- Initial conditions
def conditions :=
  tomato_plants_initial = 6 ∧
  eggplant_plants_initial = 2 ∧
  pepper_plants_initial = 4 ∧
  tomato_plants_remaining = tomato_plants_initial / 2 ∧
  pepper_plants_remaining = pepper_plants_initial - 1 ∧
  vegetables_per_plant = 7

-- Proof problem: Prove that the total number of vegetables harvested is 56.
theorem shanna_harvests_total_56 : 
  (∃ tomato_plants_remaining eggplant_plants_initial pepper_plants_remaining vegetables_per_plant total_vegetables_harvested,
    conditions ∧ total_vegetables_harvested = (tomato_plants_remaining + eggplant_plants_initial + pepper_plants_remaining) * vegetables_per_plant) :=
  by
    sorry

end shanna_harvests_total_56_l21_21229


namespace y1_lt_y2_l21_21866

theorem y1_lt_y2 (y1 y2 : ℝ) (h1 : y1 = -2 * 2 + 1) (h2 : y2 = -2 * (-1) + 1) : y1 < y2 := 
by {
  simp [h1, h2],
  exact sorry
}

end y1_lt_y2_l21_21866


namespace circumscribedCircleProof_l21_21273

noncomputable def circumscribedCircleRadius 
  (ABC : Triangle)
  (B, C : Point) 
  (l : Line)
  (M, N : Point)
  (P, Q, D : Point) 
  (sideLength : ℝ) 
  (PQ : ℝ)
  (h_eqTriangle : Equilateral ABC) 
  (h_B_onLine : B ∈ l) 
  (h_L_extendsAC : ∃ C', C' ∈ l ∧ Line.extendAC l = C') 
  (h_BMeqSide : BM = sideLength) 
  (h_BNeqSide : BN = sideLength) 
  (h_intersectMCNA : MC ∩ AB = P ∧ NA ∩ BC = Q ∧ MC ∩ NA = D)
  (h_PQ_length : PQ = sqrt(3)) : ℝ :=
1

theorem circumscribedCircleProof 
  (ABC : Triangle)
  (B, C : Point) 
  (l : Line)
  (M, N : Point)
  (P, Q, D : Point) 
  (sideLength : ℝ) 
  (PQ : ℝ)
  (h_eqTriangle : Equilateral ABC) 
  (h_B_onLine : B ∈ l) 
  (h_L_extendsAC : ∃ C', C' ∈ l ∧ Line.extendAC l = C') 
  (h_BMeqSide : BM = sideLength) 
  (h_BNeqSide : BN = sideLength) 
  (h_intersectMCNA : MC ∩ AB = P ∧ NA ∩ BC = Q ∧ MC ∩ NA = D)
  (h_PQ_length : PQ = sqrt(3)) :
  ∃ (circumcircle : Circle),
  Circumcircle PBQD ∧
  circumcircle.radius = circumscribedCircleRadius ABC B C l M N P Q D sideLength PQ h_eqTriangle h_B_onLine h_L_extendsAC h_BMeqSide h_BNeqSide h_intersectMCNA h_PQ_length :=
sorry

end circumscribedCircleProof_l21_21273


namespace calculate_sum_l21_21795

def f (x : ℝ) : ℝ :=
  if x < 5 then x - 3 else real.sqrt (x + 3)

def g_inv (y : ℝ) : ℝ := y + 3
def h_inv (y : ℝ) : ℝ := y^2 - 3

theorem calculate_sum :
  (g_inv (-6) + g_inv (-5) + g_inv (-4) + g_inv (-3) + g_inv (-2) + 
   g_inv (-1) + g_inv (0) + g_inv (1)) + 
  (h_inv (real.sqrt 8) + h_inv (real.sqrt 9) + h_inv (real.sqrt 10) +
   h_inv (real.sqrt 11) + h_inv (real.sqrt 12) + h_inv (real.sqrt 13) +
   h_inv (real.sqrt 14)) = 31 :=
  sorry

end calculate_sum_l21_21795


namespace pairings_16_points_l21_21677

def f : ℕ → ℕ 
| 0 := 1
| (n+1) := if (n+1) % 2 = 1 then 0 else f' (n+1)

noncomputable def f' : ℕ → ℕ 
| 2       := 1
| 4       := 2
| 6       := 5
| 8       := 14
| 10      := 42
| 12      := 132
| 14      := 429
| (2 * n) := ∑ k in finset.range (n - 1), f' (2 * k) * f' (2 * n - 2 - 2 * k)
| _       := 0

theorem pairings_16_points : f 16 = 1430 :=
by {
    unfold f,
    unfold f',
    sorry
}

end pairings_16_points_l21_21677


namespace sum_areas_triangles_l21_21793

def R1 : Point := ⟨0, 1 / 4⟩
def S1 : Point := ⟨1 / 4, 3 / 4⟩
def T1 : Point := ⟨1 / 2, 1⟩ -- Assuming based on further reflection context
def area_triangle_WSR1 := 1 / 32
def geometric_sum_triangles := area_triangle_WSR1 * (1 / (1 - 1 / 4))

theorem sum_areas_triangles :
  ∑' (i : ℕ), (area_triangle_WSR1 * (1/4)^i) = 1 / 24 :=
by
  sorry

end sum_areas_triangles_l21_21793


namespace difference_of_squares_l21_21012

theorem difference_of_squares : 73^2 - 47^2 = 3120 :=
by sorry

end difference_of_squares_l21_21012


namespace proof_problem_part1_proof_problem_part2_l21_21888

noncomputable def parabola_y_squared_4x (x y : ℝ) : Prop :=
  y^2 = 4 * x

structure point (x y : ℝ)

def line_through_point (m : ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - m)

def point_D (m : ℝ) : point := ⟨m, 0⟩

def point_E (m : ℝ) : point := ⟨-m, 0⟩

def circle_with_diameter_AB_passes_fixed_point (A B : point) (fixed_point : point): Prop :=
  let (⟨x1, y1⟩, ⟨x2, y2⟩, ⟨xf, yf⟩) := (A, B, fixed_point) in
  (x1 - xf) * (x2 - xf) + (y1 - yf) * (y2 - yf) = 0

theorem proof_problem_part1 (m : ℝ) (k : ℝ) (A B D E : point)
  (hm : m > 0) (hD : D = point_D m) (hE : E = point_E m) (hA : parabola_y_squared_4x A.1 A.2)
  (hB : parabola_y_squared_4x B.1 B.2) (hl : line_through_point m k A.1 A.2) (hlB : line_through_point m k B.1 B.2) :
  ∠AED = ∠BED := sorry

theorem proof_problem_part2 (A B : point) (hm : 4) :
  circle_with_diameter_AB_passes_fixed_point A B (⟨0, 0⟩) := sorry

end proof_problem_part1_proof_problem_part2_l21_21888


namespace problem1_problem2_l21_21368

theorem problem1 (e : ℝ) : 8^(-1/3) + ((-5/9) ^ 0) - real.sqrt((e-3)^2) = e - 3/2 :=
by sorry

theorem problem2 : (1/2) * real.log 25 / real.log 10 + real.log 2 / real.log 10 - (real.log 9 / real.log 2) * (real.log 2 / real.log 3) = -1 :=
by sorry

end problem1_problem2_l21_21368


namespace num_perfect_squares_in_range_l21_21493

theorem num_perfect_squares_in_range : 
  ∃ (k : ℕ), k = 12 ∧ ∀ n : ℕ, (100 < n^2 ∧ n^2 < 500 → 11 ≤ n ∧ n ≤ 22): sorry

end num_perfect_squares_in_range_l21_21493


namespace isosceles_triangle_perimeter_l21_21110

noncomputable theory

def is_isosceles_triangle (a b c : ℝ) : Prop := 
  a = b ∨ b = c ∨ a = c

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : is_isosceles_triangle a b 4) (h_valid : is_valid_triangle a b 4) :
  perimeter a b 4 = 10 :=
  sorry

end isosceles_triangle_perimeter_l21_21110


namespace unspent_portion_correct_l21_21718

-- Definitions of the conditions
variables (G : ℝ) (hG_pos : 0 < G) -- Spending limit on gold card is positive

-- Definition of balance on gold and platinum cards
def balance_gold_card := G / 3
def balance_platinum_card := 2 * G / 4

-- Define new balance on platinum after transferring from gold card
def new_balance_platinum_card := balance_platinum_card G + balance_gold_card G

-- Calculate the unspent portion as a fraction of the spending limit
def unspent_portion : ℝ := 
  let spending_limit_platinum := 2 * G in
  (spending_limit_platinum - new_balance_platinum_card G) / spending_limit_platinum

theorem unspent_portion_correct :
  unspent_portion G = 7 / 12 :=
by { sorry }

end unspent_portion_correct_l21_21718


namespace greatest_whole_number_with_odd_factors_less_than_150_l21_21203

theorem greatest_whole_number_with_odd_factors_less_than_150 :
  ∃ (n : ℕ), (∀ (m : ℕ), m < 150 ∧ odd_factors m → m ≤ n) ∧ n = 144 :=
by
  sorry

def odd_factors (k : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = k

end greatest_whole_number_with_odd_factors_less_than_150_l21_21203


namespace solving_inequality_problem_fix_l21_21673

def solution_fixes_inequality_errors : Prop :=
  ∀ (solution_fails_if_equalities : Prop)
    (initial_fix_replacing_inequalities : Prop)
    (additional_factors_considered : Prop)
    (no_other_errors_exist : Prop),
  solution_fails_if_equalities →
  no_other_errors_exist →
  (initial_fix_replacing_inequalities ∨ additional_factors_considered) →
  (initial_fix_replacing_inequalities ∧ additional_factors_considered)

theorem solving_inequality_problem_fix :
  solution_fixes_inequality_errors :=
by 
  intros solution_fails_if_equalities initial_fix_replacing_inequalities additional_factors_considered no_other_errors_exist h_fails h_no_other_errors h_fix_or_factors,
  split,
  { sorry },
  { sorry }

end solving_inequality_problem_fix_l21_21673


namespace num_integers_between_10_4_cubed_and_10_5_cubed_l21_21475

noncomputable def cube (x : ℝ) : ℝ := x^3

theorem num_integers_between_10_4_cubed_and_10_5_cubed :
  let lower_bound := 10.4
  let upper_bound := 10.5
  let lower_cubed := cube lower_bound
  let upper_cubed := cube upper_bound
  let num_integers := (⌊upper_cubed⌋₊ - ⌈lower_cubed⌉₊ + 1 : ℕ)
  lower_cubed = 1124.864 ∧ upper_cubed = 1157.625 → 
  num_integers = 33 := 
by
  intro lower_bound upper_bound lower_cubed upper_cubed num_integers h
  have : lower_cubed = 1124.864 := h.1
  have : upper_cubed = 1157.625 := h.2
  rw [this, this]
  exact 33

end num_integers_between_10_4_cubed_and_10_5_cubed_l21_21475


namespace old_supervisor_salary_correct_l21_21641

def old_supervisor_salary (W S_old : ℝ) : Prop :=
  let avg_old := (W + S_old) / 9
  let avg_new := (W + 510) / 9
  avg_old = 430 ∧ avg_new = 390 → S_old = 870

theorem old_supervisor_salary_correct (W : ℝ) :
  old_supervisor_salary W 870 :=
by
  unfold old_supervisor_salary
  intro h
  sorry

end old_supervisor_salary_correct_l21_21641


namespace necessary_and_sufficient_condition_for_absolute_inequality_l21_21419

theorem necessary_and_sufficient_condition_for_absolute_inequality (a : ℝ) :
  (a < 3) ↔ (∀ x : ℝ, |x + 2| + |x - 1| > a) :=
sorry

end necessary_and_sufficient_condition_for_absolute_inequality_l21_21419


namespace greatest_odd_factors_under_150_l21_21200

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l21_21200


namespace function_monotonically_increasing_intervals_l21_21801

noncomputable def monotonic_intervals (k : ℤ) : set ℝ :=
  { x : ℝ | k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 }

theorem function_monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ monotonic_intervals k, ∀ y ∈ monotonic_intervals k,
    x ≤ y → 
    3 * sin(π / 6 - 2 * x) ≤ 3 * sin(π / 6 - 2 * y) :=
begin
  sorry,
end

end function_monotonically_increasing_intervals_l21_21801


namespace water_tank_capacity_l21_21752

theorem water_tank_capacity :
  ∃ (x : ℝ), 0.9 * x - 0.4 * x = 30 → x = 60 :=
by
  sorry

end water_tank_capacity_l21_21752


namespace reflection_line_slope_intercept_l21_21649

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l21_21649


namespace monotonically_increasing_interval_range_of_m_l21_21861

noncomputable def a (x : ℝ) := (sqrt 3 * sin x, cos x + sin x)
noncomputable def b (x : ℝ) := (2 * cos x, sin x - cos x)
noncomputable def f (x : ℝ) := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π → monotone f :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x t, (5 * π / 24 ≤ x ∧ x ≤ 5 * π / 12) → (m * t^2 + m * t + 3 ≥ f x)) ↔ (0 ≤ m ∧ m ≤ 4) :=
sorry

end monotonically_increasing_interval_range_of_m_l21_21861


namespace calculate_distribution_l21_21371

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l21_21371


namespace spendPercentage_l21_21192

def numberOfStudents : Nat := 30

def percentageWantValentines : Float := 0.60

def costPerValentine : Nat := 2

def totalMoney : Nat := 40

theorem spendPercentage :
  let studentsGettingValentines := percentageWantValentines * numberOfStudents
  let totalCost := costPerValentine * studentsGettingValentines
  (totalCost / totalMoney) * 100 = 90 :=
by
  sorry

end spendPercentage_l21_21192


namespace solution_set_f_ge_0_l21_21878

noncomputable def f (x a : ℝ) : ℝ := 1 / Real.exp x - a / x

theorem solution_set_f_ge_0 (a m n : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ n ↔ 1 / Real.exp x - a / x ≥ 0) : 
  0 < a ∧ a < 1 / Real.exp 1 :=
  sorry

end solution_set_f_ge_0_l21_21878


namespace minimize_tank_construction_cost_l21_21357

noncomputable def minimum_cost (l w h : ℝ) (P_base P_wall : ℝ) : ℝ :=
  P_base * (l * w) + P_wall * (2 * h * (l + w))

theorem minimize_tank_construction_cost :
  ∃ l w : ℝ, l * w = 9 ∧ l = w ∧
  minimum_cost l w 2 200 150 = 5400 :=
by
  sorry

end minimize_tank_construction_cost_l21_21357


namespace length_of_XY_l21_21217

structure Point : Type := 
(x : ℝ)
(y : ℝ)

abbreviation length (p1 p2 : Point) : ℝ := 
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def midpoint (p1 p2 : Point) : Point :=
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

variables (X Y G H I J : Point)
variables (XJ_length : ℝ)
variables (midpoint_G : G = midpoint X Y)
variables (midpoint_H : H = midpoint X G)
variables (midpoint_I : I = midpoint X H)
variables (midpoint_J : J = midpoint X I)
variables (XJ_is_5 : length X J = 5)

theorem length_of_XY : length X Y = 80 :=
by
  sorry

end length_of_XY_l21_21217


namespace find_cot_difference_l21_21137

-- Define necessary elements for the problem
variable {A B C D : Type}
variable [EuclideanGeometry A]
variables (ABC : Triangle A B C)

-- Define the condition where median AD makes an angle of 60 degrees with BC
variable (ADmedian : median A B C D ∧ angle D A B = 60)

theorem find_cot_difference:
  |cot (angle B) - cot (angle C)| = 2 :=
sorry

end find_cot_difference_l21_21137


namespace log_expression_value_l21_21768

theorem log_expression_value (lg : ℕ → ℤ) :
  (lg 4 = 2 * lg 2) →
  (lg 20 = lg 4 + lg 5) →
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 :=
by
  intros h1 h2
  sorry

end log_expression_value_l21_21768


namespace _l21_21949

noncomputable def area_of_MPQY (XY XZ area_XYZ YM MZ XN NZ YQ QZ XQZ MPN : ℝ)
                                (M_midpoint_XY : XY / 2 = YM)
                                (N_midpoint_XZ : XZ / 2 = XN)
                                (similarity_ratio : (YM = MZ) ∧ (XN = NZ))
                                (area_similarity : area_XYZ / 4 = 22.5)
                                (angle_bisector_theorem : (2 * YQ = QZ * 2) ∧ (2 * QZ + QZ = YQ) ∧ (area_XQZ = 30))
                                (area_MPN : area_XMN / 3 = MPN)
                                (total_area : area_XYZ - (22.5 + 30 - 7.5) = 45) : Prop :=
  area_XYZ = 90 ∧ 45 = area_XYZ - ((area_XMN = (1/4) * area_XYZ) ∧ (area_XQZ = (1/3) * area_XYZ) ∧ (area_MPN = (1/3) * area_XMN))

end _l21_21949


namespace drum_oil_ratio_l21_21808

theorem drum_oil_ratio (C_X C_Y : ℝ) (h1 : (1 / 2) * C_X + (1 / 5) * C_Y = 0.45 * C_Y) : 
  C_Y / C_X = 2 :=
by
  -- Cannot provide the proof
  sorry

end drum_oil_ratio_l21_21808


namespace evaluate_101_mul_101_l21_21393

noncomputable def a : ℕ := 100
noncomputable def b : ℕ := 1

theorem evaluate_101_mul_101 :
  (a + b) ^ 2 = 10201 := by
  let x := (a + b)^2
  have h : x = a^2 + 2*a*b + b^2 := by sorry
  have a_sq : a^2 = 10000 := by sorry
  have two_ab : 2*a*b = 200 := by sorry
  have b_sq : b^2 = 1 := by sorry
  show 10201 = a^2 + 2*a*b + b^2 from by sorry
  show 10201 = x from by sorry

end evaluate_101_mul_101_l21_21393


namespace perimeter_difference_l21_21664

-- Define the height of the screen
def height_of_screen : ℕ := 100

-- Define the side length of the square paper
def side_of_square_paper : ℕ := 20

-- Define the perimeter of the square paper
def perimeter_of_paper : ℕ := 4 * side_of_square_paper

-- Prove the difference between the height of the screen and the perimeter of the paper
theorem perimeter_difference : height_of_screen - perimeter_of_paper = 20 := by
  -- Sorry is used here to skip the actual proof
  sorry

end perimeter_difference_l21_21664


namespace rectangle_other_side_length_l21_21245

variables (a b : ℝ)

theorem rectangle_other_side_length:
  let area := 9 * a^2 - 6 * a * b + 3 * a in
  let side1 := 3 * a in
  (area / side1) = 3 * a - 2 * b + 1 :=
by
  sorry

end rectangle_other_side_length_l21_21245


namespace intersection_of_A_and_B_l21_21859

def setA : Set ℝ := { x | x - 2 ≥ 0 }
def setB : Set ℝ := { x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | 2 ≤ x ∧ x < 4 } :=
sorry

end intersection_of_A_and_B_l21_21859


namespace tetrahedron_pass_through_0_45_tetrahedron_not_pass_through_0_44_l21_21771

def can_pass_through_hole (edge_length : ℝ) (hole_radius : ℝ) : Prop :=
  ∃ (x : ℝ), 3 * x^3 - 6 * x^2 + 7 * x - 2 = 0 ∧
              let R := (x^2 - x + 1) / (real.sqrt (3 * x^2 - 4 * x + 4)) in
              edge_length = 1 ∧ R <= hole_radius

theorem tetrahedron_pass_through_0_45 :
  can_pass_through_hole 1 0.45 :=
begin
  sorry
end

theorem tetrahedron_not_pass_through_0_44 :
  ¬ can_pass_through_hole 1 0.44 :=
begin
  sorry
end

end tetrahedron_pass_through_0_45_tetrahedron_not_pass_through_0_44_l21_21771


namespace part1_calc_value_part2_log_expressions_l21_21309

-- Part 1: Proof that the complicated expression evaluates to -7.325
theorem part1_calc_value : 
  0.0081^(1/4) + (4^(-3/4))^2 + (sqrt 8)^(-4/3) - 16^(0.75) = -7.325 := sorry

-- Part 2: Proof that log_5 in terms of p and q evaluates to the given expression
open Real

theorem part2_log_expressions 
  (p q : ℝ) 
  (hp : log 32 9 = p)
  (hq : log 27 25 = q) 
: log 5 = (15 * p * q) / (15 * p * q + 4) := sorry

end part1_calc_value_part2_log_expressions_l21_21309


namespace problem1_problem2_l21_21723

variable (α : ℝ)

theorem problem1 (ha : tan α = 1 / 4) :
  (tan (3 * π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) / 
  (cos (-α - π) * sin (-π + α) * cos (α + 5 * π / 2)) = -1 / sin α :=
sorry

theorem problem2 (ha : tan α = 1 / 4) :
  1 / (2 * cos α ^ 2 - 3 * sin α * cos α) = 17 / 20 :=
sorry

end problem1_problem2_l21_21723


namespace community_members_l21_21728

theorem community_members (n k : ℕ) (hk : k = 2) (hn : n = 6) 
    (H : ∀ (C1 C2 : Fin n) (hC1C2 : C1 ≠ C2), 
          ∃! (x : Fin ((n * (n - 1)) / 2)), 
          (x:Fin ((n * (n - 1)) / 2)) = (C1.val * n + C2.val - C1.val * (C1.val + 1) / 2)) :
    (∃ (x : ℕ), x = (n * (n - 1)) / (2 * k)) :=
by sorry

end community_members_l21_21728


namespace sixth_term_is_486_terms_exceed_2007_from_8th_l21_21338

def a1 := 3
def a2 := 6
def a3 := 18

def seq : ℕ → ℕ 
| 1 => a1
| 2 => a2
| 3 => a3
| n => 2 * ((List.range (n - 1)).sum (fun i => seq (i + 1)))

theorem sixth_term_is_486 : seq 6 = 486 := sorry

theorem terms_exceed_2007_from_8th : ∀ n ≥ 8, seq n > 2007 := sorry

end sixth_term_is_486_terms_exceed_2007_from_8th_l21_21338


namespace total_recovery_time_correct_l21_21957

def initial_healing_time : ℕ := 4
def healing_factor : ℚ := 0.5

def graft_healing_time (t₁ : ℕ) (f : ℚ) : ℕ :=
  t₁ + (f * t₁).toNat

def total_recovery_time (t₁ : ℕ) (t₂ : ℕ) : ℕ :=
  t₁ + t₂

theorem total_recovery_time_correct : total_recovery_time initial_healing_time (graft_healing_time initial_healing_time healing_factor) = 10 :=
by
  sorry

end total_recovery_time_correct_l21_21957


namespace minimize_x_l21_21711

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end minimize_x_l21_21711


namespace Aaron_cards_count_l21_21760

theorem Aaron_cards_count (initial_cards : ℕ) (found_cards : ℕ) (h1 : initial_cards = 5) (h2 : found_cards = 62) : initial_cards + found_cards = 67 :=
by
  rw [h1, h2]
  exact Nat.add_comm 5 62

end Aaron_cards_count_l21_21760


namespace max_suitable_pairs_l21_21423

theorem max_suitable_pairs (m n : ℕ) (hm : 1 < m) (hn : 1 < n)
  (As : Fin m → Finset (ℕ)) (h_size : ∀ i, (As i).card = n)
  (h_disjoint : ∀ i j, i ≠ j → As i ∩ As j = ∅)
  (h_div_cond : ∀ i (a ∈ As i) (b ∈ As ((i+1) % m)), ¬ (a ∣ b)) :
  (∑ i in Finset.range m, (As i).sum) ∑ a, ∑ b, (a ∣ b) = n^2 * (nat.choose (m-1) 2) :=
sorry

end max_suitable_pairs_l21_21423


namespace An_integer_condition_An_not_prime_l21_21399

theorem An_integer_condition (n k : Nat) (h: n = 3 * k + 1) : 
  (2^(4 * n + 2) + 1) % 65 = 0 := sorry

theorem An_not_prime (n : Nat) (h : n > 0) (h1 : (2^(4 * n + 2) + 1) % 65 = 0) : 
  ¬ isPrime ((2^(4 * n + 2) + 1) / 65) := sorry

end An_integer_condition_An_not_prime_l21_21399


namespace volume_NKLB_l21_21218

variables (A B C D K L N : Type) 
variables [Field A] [Field B] [Field C] [Field D] 
variables [AddGroup K] [AddGroup L] [AddGroup N]

def volume (tetrahedron : A × B × C × D) : ℝ := sorry

theorem volume_NKLB {V : ℝ} (A B C D K L N : Type)
  (h1 : K ∈ segment (A, B)) 
  (h2 : L ∈ segment (B, C))
  (h3 : N ∈ segment (A, D))
  (h4 : 2 • (0 : ℝ) + 1 = 1)
  (h5 : 3 • (0 : ℝ) + 1 = 1)
  (h6 : 5 • (0 : ℝ) + 1 = 1)
  (hh1 : volume (A, B, C, D) = V)
  : volume (N, K, L, B) = (2 / 45) * V :=
sorry

end volume_NKLB_l21_21218


namespace sequence_general_formula_l21_21039

theorem sequence_general_formula :
  (∀ n : ℕ, n ≥ 1 → (∃ a : ℕ → ℕ, a n + 5 = n + 5) ∧ (∃ b : ℕ → ℕ, b n + 2 = 3 * n + 2)) ∧
  (∃ T : ℕ → ℕ, T n = 32 * (n - 1) * 2^(n + 1) + 64) ∧
  (∃ m : ℕ, m > 0 ∧ 
    (∀ f : ℕ → ℕ,
      (∀ n, (if n % 2 = 1 then f n = n + 5 else f n = 3 * n + 2)) →
      f (m + 15) = 5 * f m)) :=
begin
  sorry
end

end sequence_general_formula_l21_21039


namespace reflection_line_slope_intercept_l21_21651

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l21_21651


namespace maximum_value_existence_check_l21_21029

open Real
open Classical

noncomputable def conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ A = sqrt a + sqrt b ∧ B = a + b

noncomputable def max_value_of_expr (a b : ℝ) : ℝ := 
sqrt (2: ℝ) * (sqrt a + sqrt b) - a - b

theorem maximum_value 
(a b : ℝ) (h : conditions a b) : 
max_value_of_expr a b ≤ 1 := 
sorry

theorem existence_check 
(a b : ℝ) (h1 : a > 0 ∧ b > 0) (h2 : a * b = 4) 
(h3 : sqrt a + sqrt b + a + b = 6) : 
∃ a b : ℝ, conditions a b ∧ a * b = 4 ∧ sqrt a + sqrt b + a + b = 6 := 
sorry

end maximum_value_existence_check_l21_21029


namespace problem1_problem2_l21_21269

-- Definitions for context
constant A B C D E : Type

constant isAdjacent (x y : Type) : Prop
constant isNotAdjacent (x y : Type) : Prop

-- Our specific problem definitions
def students : List Type := [A, B, C, D, E]

def validArrangement : List Type → Prop := λ lst,
  isAdjacent A B ∧ isNotAdjacent C D

-- Statements of the problems in Lean

-- Problem 1: The number of valid arrangements with A and B adjacent, and C and D not adjacent, is 12.
theorem problem1 : ∃ (arrangements : List (List Type)), (validArrangement arrangements) → arrangements.length = 12 :=
by
  sorry

-- Problem 2: The number of ways to distribute students into three classes with at least one student in each class is 150.
theorem problem2 : ∃ (distributions : List (List (List Type))), 
                   (∀ class, class.length ≥ 1 ∧ class.length ≤ 3) → distributions.length = 150 :=
by
  sorry

end problem1_problem2_l21_21269


namespace product_complex_l21_21788

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem product_complex : 
  ∏ k in finset.range 15, (∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
sorry

end product_complex_l21_21788


namespace determine_x3_value_l21_21692

noncomputable def x3_value (x1 x2 : ℝ) (f : ℝ → ℝ) : ℝ :=
  let y1 := f x1
  let y2 := f x2
  let C_y := (2/3) * y1 + (1/3) * y2
  Real.log ((2 + Real.exp 3) / 3)

theorem determine_x3_value :
  let f := λ x, Real.exp x
  ∀ x1 x2, x1 = 0 → x2 = 3 → x3_value x1 x2 f = Real.log ((2 + Real.exp 3) / 3) :=
by
  intros f x1 x2 h1 h2
  rw [h1, h2]
  sorry

end determine_x3_value_l21_21692


namespace dividend_calculation_l21_21823

theorem dividend_calculation (d : ℕ) (h : 11997708 / 12 = 999809) : d = 11997708 :=
by
  have h1 : 11997708 % 12 = 0 := by sorry
  rw [← Nat.mul_div_cancel_left 999809 12 h1] at h
  exact h.symm

end dividend_calculation_l21_21823


namespace probability_non_intersecting_chords_l21_21826

theorem probability_non_intersecting_chords (n : ℕ) (h : n = 2000) (A B C D : Fin n) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  : (let chords_intersect (X Y Z W : Fin n) := ∀ i : Fin 4, ∃ j : Fin 4, X ≠ Y ∧ ¬((i.val = X.val ∨ i.val = Y.val) ∧ (j.val = Z.val ∨ j.val = W.val)) in  
    let non_intersecting_count := (4 : ℤ) in
    let total_orders := (6 : ℤ) in
    non_intersecting_count/total_orders) = (2 / 3 : ℚ) := sorry

end probability_non_intersecting_chords_l21_21826


namespace find_BD_length_l21_21925

theorem find_BD_length (AC BC AB CD : ℝ) (hAC : AC = 10) (hBC : BC = 10) (hAB : AB = 5) (hCD : CD = 13)
  (D_on_AB : ∃ D : ℝ, D ∈ [AB, ∞) ∧ B ∈ (A, D) ∧ AB + BD = AD): 
  BD ≈ 6.17 := 
by sorry

end find_BD_length_l21_21925


namespace find_m_and_b_sum_l21_21656

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l21_21656


namespace factor_x_squared_minus_four_x_minus_five_max_value_neg_two_x_squared_minus_four_x_plus_three_l21_21722

-- Proof Problem for Question 1
theorem factor_x_squared_minus_four_x_minus_five :
  ∀ x : ℝ, x^2 - 4 * x - 5 = (x + 1) * (x - 5) :=
by
  intro x
  sorry

-- Proof Problem for Question 2
theorem max_value_neg_two_x_squared_minus_four_x_plus_three :
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, -2 * y^2 - 4 * y + 3 ≤ -2 * x^2 - 4 * x + 3 :=
by
  use -1
  split
  · sorry
  · intro y
    sorry

end factor_x_squared_minus_four_x_minus_five_max_value_neg_two_x_squared_minus_four_x_plus_three_l21_21722


namespace second_player_always_wins_l21_21691

theorem second_player_always_wins :
  ∀ (board : list (option char)), -- representing the game board
    2000 = board.length → -- condition 1: game is played on a line of 2000 squares
    (∀ n, (board.nth n = some 'S' ∨ board.nth n = some 'O' ∨ board.nth n = none) → -- condition 2: either S, O, or empty
      (∀ three_adjacent_squares, -- condition 3: game stops and last player wins on "SOS"
          ((three_adjacent_squares = ['S', 'O', 'S']) → 
           (∃ i, three_adjacent_squares = [board.nth i, board.nth (i + 1), board.nth (i + 2)] ∧
                ∀ i ∈ range 2000, 
                ((board.nth i = some 'S' ∧ board.nth (i + 1) = some 'O' ∧ board.nth (i + 2) = some 'S') ∨
                 (board.nth i = some 'O' ∧ board.nth (i + 1) = some 'S' ∧ board.nth (i + 2) = some 'O') → false)) →
          (∀ all_squares_filled, -- condition 4: if all squares are filled without "SOS", the game is a draw
              (all_squares_filled = list.all (λ x, x ≠ none) → 
               ∀ three_adjacent_squares ≠ ['S', 'O', 'S'] → 
                  (∃ i, all_squares_filled = list.all (λ i, board.nth i = some 'S' ∨ board.nth i = some 'O') → 
                       ∀ i ∈ range 2000, 
                       ((board.nth i = some 'S' ∧ board.nth (i + 1) = some 'O' ∧ board.nth (i + 2) = some 'S')) → false)))))) → 
  second_player_can_win board)
 : sorry

end second_player_always_wins_l21_21691


namespace keith_spent_correctly_l21_21584

def packs_of_digimon_cards := 4
def cost_per_digimon_pack := 4.45
def cost_of_baseball_deck := 6.06
def total_spent := 23.86

theorem keith_spent_correctly :
  (packs_of_digimon_cards * cost_per_digimon_pack) + cost_of_baseball_deck = total_spent :=
sorry

end keith_spent_correctly_l21_21584


namespace find_b_l21_21398

theorem find_b (b : ℤ) : (∃ x : ℤ, x^4 + 4 * x^3 + 2 * x^2 + b * x + 12 = 0) → (b ∈ {-34, -19, -10, -9, -3, 2, 4, 6, 8, 11}) :=
by
  -- Proof omitted
  sorry

end find_b_l21_21398


namespace max_subset_size_no_sum_divisible_by_5_l21_21975

theorem max_subset_size_no_sum_divisible_by_5 :
  ∃ S ⊆ (finset.range 101).erase 0, (∀ x ∈ S, ∀ y ∈ S, x ≠ y → (x + y) % 5 ≠ 0) ∧ S.card = 40 :=
sorry

end max_subset_size_no_sum_divisible_by_5_l21_21975


namespace scientific_notation_equivalence_l21_21188

theorem scientific_notation_equivalence : 3 * 10^(-7) = 0.0000003 :=
by
  sorry

end scientific_notation_equivalence_l21_21188


namespace most_people_attend_tuesday_l21_21392

structure PersonAvailability :=
  (name : String)
  (mon : Bool)
  (tues : Bool)
  (wed : Bool)
  (thurs : Bool)
  (fri : Bool)
  (sat : Bool)

def Anna := PersonAvailability.mk "Anna" true false true false false false
def Bill := PersonAvailability.mk "Bill" false true false true true false
def Carl := PersonAvailability.mk "Carl" true true false true true false
def Dana := PersonAvailability.mk "Dana" false false true false false true
def Eve := PersonAvailability.mk "Eve" false false false false true true

def attendees_count (day : PersonAvailability → Bool) : Nat :=
  [Anna, Bill, Carl, Dana, Eve].count day

theorem most_people_attend_tuesday :
  attendees_count (λ x => ¬ x.mon) = 2 ∧
  attendees_count (λ x => ¬ x.tues) = 3 ∧
  attendees_count (λ x => ¬ x.wed) = 2 ∧
  attendees_count (λ x => ¬ x.thurs) = 3 ∧
  attendees_count (λ x => ¬ x.fri) = 2 ∧
  attendees_count (λ x => ¬ x.sat) = 2 ∧
  3 = List.maximum [attendees_count (λ x => ¬ x.mon),
                    attendees_count (λ x => ¬ x.tues),
                    attendees_count (λ x => ¬ x.wed),
                    attendees_count (λ x => ¬ x.thurs),
                    attendees_count (λ x => ¬ x.fri),
                    attendees_count (λ x => ¬ x.sat)] :=
by
  -- proof omitted
  sorry

end most_people_attend_tuesday_l21_21392


namespace centrally_symmetric_convex_octagon_l21_21240

theorem centrally_symmetric_convex_octagon
  (O : Type)
  [Octagon O]
  (convex : is_convex O)
  (equal_angles : ∀ (i j : ℕ), i ≠ j → angle_eq O i j)
  (rational_ratios : ∀ (i j : ℕ), i ≠ j → ∃ (q : ℚ), length_ratio O i j = q) :
  is_centrally_symmetric O :=
by
  -- Proof is skipped
  sorry

end centrally_symmetric_convex_octagon_l21_21240


namespace dot_product_is_constant_l21_21425

-- Define the trajectory C as the parabola given by the equation y^2 = 4x
def trajectory (x y : ℝ) : Prop := y^2 = 4 * x

-- Prove the range of k for the line passing through point (-1, 0) and intersecting trajectory C
def valid_slope (k : ℝ) : Prop := (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)

-- Prove that ∀ D ≠ A, B on the parabola y^2 = 4x, and lines DA and DB intersect vertical line through (1, 0) on points P, Q, OP ⋅ OQ = 5
theorem dot_product_is_constant (D A B P Q : ℝ × ℝ) 
  (hD : trajectory D.1 D.2)
  (hA : trajectory A.1 A.2)
  (hB : trajectory B.1 B.2)
  (hDiff : D ≠ A ∧ D ≠ B)
  (hP : P = (1, (D.2 * A.2 + 4) / (D.2 + A.2))) 
  (hQ : Q = (1, (D.2 * B.2 + 4) / (D.2 + B.2))) :
  (1 + (D.2 * A.2 + 4) / (D.2 + A.2)) * (1 + (D.2 * B.2 + 4) / (D.2 + B.2)) = 5 :=
sorry

end dot_product_is_constant_l21_21425


namespace range_of_a_l21_21043

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x₀ : ℝ, x₀^2 + 4 * x₀ + a = 0) : e ≤ a ∧ a ≤ 4 :=
begin
  sorry
end

end range_of_a_l21_21043


namespace lcm_of_fractions_is_correct_l21_21288

-- Define denominators
def denom1 (x : ℚ) : ℚ := 5 * x
def denom2 (x : ℚ) : ℚ := 10 * x
def denom3 (x : ℚ) : ℚ := 15 * x

-- Define the fractions
def frac1 (x : ℚ) : ℚ := 1 / denom1 x
def frac2 (x : ℚ) : ℚ := 1 / denom2 x
def frac3 (x : ℚ) : ℚ := 1 / denom3 x

-- Define the least common multiple (LCM) of the denominators
def lcm_den (x : ℚ) : ℚ := 30 * x

-- The statement to be proven
theorem lcm_of_fractions_is_correct (x : ℚ) :
  ∃ y : ℚ, y = lcm_den x ∧ (frac1 x ≤ 1 / y) ∧ (frac2 x ≤ 1 / y) ∧ (frac3 x ≤ 1 / y) :=
sorry

end lcm_of_fractions_is_correct_l21_21288


namespace part_i_part_ii_l21_21182

-- Define the variables and conditions
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a + b = 1 / a + 1 / b)

-- Prove the first part: a + b ≥ 2
theorem part_i : a + b ≥ 2 := by
  sorry

-- Prove the second part: It is impossible for both a² + a < 2 and b² + b < 2 simultaneously
theorem part_ii : ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end part_i_part_ii_l21_21182


namespace fare_for_100_miles_is_240_l21_21755

variable (base_fare : ℕ) 
variable (fare_80_miles : ℕ) 
variable (distance_80_miles : ℕ)
variable (distance_100_miles : ℕ)

def total_fare_for_100_miles (base_fare : ℕ) (fare_80_miles : ℕ) (distance_80_miles : ℕ) (distance_100_miles : ℕ) : ℕ :=
  let variable_fare := fare_80_miles - base_fare in
  let rate := variable_fare / distance_80_miles in
  let variable_fare_100 := rate * distance_100_miles in
  variable_fare_100 + base_fare

theorem fare_for_100_miles_is_240 :
  base_fare = 40 →
  fare_80_miles = 200 →
  distance_80_miles = 80 →
  distance_100_miles = 100 →
  total_fare_for_100_miles base_fare fare_80_miles distance_80_miles distance_100_miles = 240 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end fare_for_100_miles_is_240_l21_21755


namespace greatest_whole_number_with_odd_factors_less_than_150_l21_21201

theorem greatest_whole_number_with_odd_factors_less_than_150 :
  ∃ (n : ℕ), (∀ (m : ℕ), m < 150 ∧ odd_factors m → m ≤ n) ∧ n = 144 :=
by
  sorry

def odd_factors (k : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = k

end greatest_whole_number_with_odd_factors_less_than_150_l21_21201


namespace pelican_fish_count_l21_21737

theorem pelican_fish_count 
(P K F : ℕ) 
(h1: K = P + 7) 
(h2: F = 3 * (P + K)) 
(h3: F = P + 86) : P = 13 := 
by 
  sorry

end pelican_fish_count_l21_21737


namespace reflection_line_slope_intercept_l21_21652

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l21_21652


namespace dot_product_ad_zero_l21_21979

variables (a b c d : ℝ³)
variables [fact (|a| = 1)] [fact (|b| = 2)] [fact (|c| = 3)] [fact (|d| = 2)]
variables [fact (a.dot b = 1)] [fact (a.dot c = -1)] [fact (b.dot c = 0)]
variables [fact (b.dot d = -1)] [fact (c.dot d = 3)]

theorem dot_product_ad_zero : a.dot d = 0 :=
sorry

end dot_product_ad_zero_l21_21979


namespace number_of_valid_functions_l21_21175

noncomputable def A : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_fun (f : (Set (Finset ℕ) \ {∅}) → ℕ) : Prop :=
  (∀ B ∈ (Set (Finset ℕ)).nonempty, f B ∈ B) ∧
  (∀ B C ∈ (Set (Finset ℕ)).nonempty, f (B ∪ C) ∈ {f B, f C})

theorem number_of_valid_functions : 
  (Finset.univ.filter (λ f : (Set (Finset ℕ) \ {∅}) → ℕ, is_valid_fun f)).card = 120 :=
sorry

end number_of_valid_functions_l21_21175


namespace solve_x_l21_21916

theorem solve_x (x : ℝ) (h : (x / 3) / 5 = 5 / (x / 3)) : x = 15 ∨ x = -15 :=
by sorry

end solve_x_l21_21916


namespace concurrency_or_parallel_l21_21051

-- Define the conditions of the problem:
variables {O A B C D Q E F G H P : Type*}
variables [circle O ABCD] -- Quadrilateral \(ABCD\) is inscribed in circle \(\odot O\)

-- [Diagonals \(AC\) and \(BD\) intersect at point \(Q\)]
variables (inter_diags : AC = BD)

-- [Lines through \(Q\) are perpendicular to \(AB\), \(BC\), \(CD\), and \(DA\)]
variables (perpendicular_Q_AB : perpendicular (Q,AB)) 
variables (perpendicular_Q_BC : perpendicular (Q,BC)) 
variables (perpendicular_Q_CD : perpendicular (Q,CD)) 
variables (perpendicular_Q_DA : perpendicular (Q,DA))

-- [The feet of the perpendiculars from \(Q\) to these lines are \(E\), \(F\), \(G\), and \(H\), respectively.]
variables (foot_E : foot_of_perpendicular Q AB = E)
variables (foot_F : foot_of_perpendicular Q BC = F)
variables (foot_G : foot_of_perpendicular Q CD = G)
variables (foot_H : foot_of_perpendicular Q DA = H)

theorem concurrency_or_parallel :
  concurrent  (EH, BD, FG) ∨ 
  parallel     (EH, BD, FG) :=
sorry

end concurrency_or_parallel_l21_21051


namespace haley_initial_music_files_l21_21897

theorem haley_initial_music_files (M : ℕ) 
  (h1 : M + 42 - 11 = 58) : M = 27 := 
by
  sorry

end haley_initial_music_files_l21_21897


namespace problem_statement_l21_21566

-- Definitions for the problem elements
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 2) / 2 * t, (real.sqrt 2) / 2 * t)

def curve_polar (θ : ℝ) : ℝ :=
  2 * real.sqrt 2 * real.sin (θ + real.pi / 4)

-- Proof problem
theorem problem_statement :
  -- Condition 1: General equation of line l
  (∀ t : ℝ, let p := line_parametric t in (p.1 - p.2 - 1 = 0)) ∧ 
  -- Condition 2: Rectangular coordinate equation of curve C
  (∀ θ : ℝ, let ρ := curve_polar θ,
      ρ ^ 2 = (ρ * real.cos θ)^2 + (ρ * real.sin θ)^2 →
      (ρ * real.cos θ)^2 + (ρ * real.sin θ)^2 - 2 * (ρ * real.cos θ) - 2 * (ρ * real.sin θ) = 0) ∧
  -- Condition 3: Value of 1/|PA| + 1/|PB|
  let P : ℝ × ℝ := (4, 3) in
  let t_vals := [root1, root2] in
    (root1 * root2 = 11 ∧ root1 + root2 = -5 * real.sqrt 2 ∧ ∀ t ∈ t_vals, t < 0) →
  (1 / |root1| + 1 / |root2| = 5 * real.sqrt 2 / 11) := sorry

end problem_statement_l21_21566


namespace find_point_B_l21_21989

noncomputable def parabola (x : ℝ) : ℝ := x^2

def A : ℝ × ℝ := (2, 4)

def normal_line (x : ℝ) : ℝ := -1/4 * x + 9/2

theorem find_point_B : 
    ∃ B : ℝ × ℝ, 
    A = (2, parabola 2) ∧
    B ≠ A ∧
    B.snd = parabola B.fst ∧
    B.snd = normal_line B.fst ∧
    B = (-(9 / 4), (81 / 16)) :=
by
  sorry

end find_point_B_l21_21989


namespace find_ellipse_equation_max_radius_inscribed_circle_l21_21041

-- Define the conditions of the problem
variables (a b : ℝ) (x y : ℝ) (F1 F2 : ℝ × ℝ) (eccentricity : ℝ)

-- Condition 1: The equation of the ellipse
def ellipse_equation := (x^2 / a^2) + (y^2 / b^2) = 1

-- Condition 2: The ellipse passes through (sqrt(3), 1/2)
def passes_through_point := ellipse_equation a b (sqrt 3) (1 / 2)

-- Condition 3: The eccentricity of the ellipse
def is_eccentricity := (c : ℝ) → c / a = sqrt 3 / 2

-- Condition 4: A line passing through F1 intersects at P and Q.
-- This one implies a line equation and can be considered in geometric definitions

-- Proof goal 1: Equation of the ellipse
theorem find_ellipse_equation : ellipse_equation 2 1 :=
sorry

-- Proof goal 2: Maximum radius of inscribed circle in triangle formed
theorem max_radius_inscribed_circle : ∃ r : ℝ, r = 1 / 2 :=
sorry

end find_ellipse_equation_max_radius_inscribed_circle_l21_21041


namespace product_simplification_l21_21013

theorem product_simplification :
  (∏ n in Finset.range 149, (1 - (1 / (n + 2)))) = (1 / 150) :=
by sorry

end product_simplification_l21_21013


namespace total_area_of_squares_l21_21270

theorem total_area_of_squares (x : ℝ) (hx : 4 * x^2 = 240) : 
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  2 * small_square_area + large_square_area = 360 :=
by
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  sorry

end total_area_of_squares_l21_21270


namespace probability_jqka_is_correct_l21_21414

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end probability_jqka_is_correct_l21_21414


namespace david_marks_in_physics_l21_21004

theorem david_marks_in_physics
  (marks_english : ℤ)
  (marks_math : ℤ)
  (marks_chemistry : ℤ)
  (marks_biology : ℤ)
  (average_marks : ℚ)
  (number_of_subjects : ℤ)
  (h_english : marks_english = 96)
  (h_math : marks_math = 98)
  (h_chemistry : marks_chemistry = 100)
  (h_biology : marks_biology = 98)
  (h_average : average_marks = 98.2)
  (h_subjects : number_of_subjects = 5) : 
  ∃ (marks_physics : ℤ), marks_physics = 99 := 
by {
  sorry
}

end david_marks_in_physics_l21_21004


namespace range_k_real_solutions_l21_21410

theorem range_k_real_solutions (k : ℝ) (h : ∃ x : ℝ, 2 * k * sin x = 1 + k^2) : k = -1 ∨ k = 1 :=
by {
  sorry
}

end range_k_real_solutions_l21_21410


namespace big_bottles_sold_percentage_l21_21340

theorem big_bottles_sold_percentage (total_small : ℕ) (total_big : ℕ) 
  (perc_small_sold : ℝ) (remaining_bottles : ℕ) 
  (h1 : total_small = 6000) (h2 : total_big = 10000) 
  (h3 : perc_small_sold = 0.12) (h4 : remaining_bottles = 13780) : 
  (x : ℝ) (h5 : x = 0.15) := 
sorry

end big_bottles_sold_percentage_l21_21340


namespace minimum_crooks_l21_21557

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l21_21557


namespace students_prefer_windows_l21_21317

theorem students_prefer_windows (total_students students_prefer_mac equally_prefer_both no_preference : ℕ) 
  (h₁ : total_students = 210)
  (h₂ : students_prefer_mac = 60)
  (h₃ : equally_prefer_both = 20)
  (h₄ : no_preference = 90) :
  total_students - students_prefer_mac - equally_prefer_both - no_preference = 40 := 
  by
    -- Proof goes here
    sorry

end students_prefer_windows_l21_21317


namespace count_multiples_of_3003_in_form_l21_21903

noncomputable def is_multiple_form (i j : ℕ) : Prop :=
  0 ≤ i ∧ i < j ∧ j ≤ 99 ∧ ∃ k : ℕ, 3003 * k = 10^j - 10^i

theorem count_multiples_of_3003_in_form : 
  (finset.filter
    (λ p : ℕ × ℕ, is_multiple_form p.1 p.2)
    ((finset.range 100).product (finset.range 100))).card = 784 :=
begin
  sorry
end

end count_multiples_of_3003_in_form_l21_21903


namespace eccentricity_range_l21_21860

noncomputable def semi_focal_distance (a b : ℝ) : ℝ := 
  real.sqrt (a^2 - b^2)

def eccentricity (a b : ℝ) : ℝ := 
  semi_focal_distance a b / a

theorem eccentricity_range 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : c = semi_focal_distance a b)
  (h4 : ∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1) 
  (h5 : ∀ (x y : ℝ), ((-c - x, -y) • (c - x, -y)) = c^2) :
  (real.sqrt 3) / 3 ≤ eccentricity a b ∧ eccentricity a b ≤ (real.sqrt 2) / 2 :=
sorry

end eccentricity_range_l21_21860


namespace minimum_crooks_l21_21559

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l21_21559


namespace cot_difference_triangle_l21_21142

theorem cot_difference_triangle (ABC : Triangle)
  (angle_condition : ∠AD BC = 60) :
  |cot ∠B - cot ∠C| = 5 / 2 :=
sorry

end cot_difference_triangle_l21_21142


namespace age_of_15th_student_l21_21303

noncomputable def average_age_15_students := 15
noncomputable def average_age_7_students_1 := 14
noncomputable def average_age_7_students_2 := 16
noncomputable def total_students := 15
noncomputable def group_students := 7

theorem age_of_15th_student :
  let total_age_15_students := total_students * average_age_15_students
  let total_age_7_students_1 := group_students * average_age_7_students_1
  let total_age_7_students_2 := group_students * average_age_7_students_2
  let total_age_14_students := total_age_7_students_1 + total_age_7_students_2
  let age_15th_student := total_age_15_students - total_age_14_students
  age_15th_student = 15 :=
by
  sorry

end age_of_15th_student_l21_21303


namespace count_perfect_squares_between_100_and_500_l21_21499

def smallest_a (x : ℕ) : ℕ :=
  Nat.find (λ a, a^2 > x)

def largest_b (x : ℕ) : ℕ :=
  Nat.find (λ b, b^2 > x) - 1

theorem count_perfect_squares_between_100_and_500 :
  let a := smallest_a 100
  let b := largest_b 500
  b - a + 1 = 12 :=
by
  -- Definitions based on conditions
  let a := smallest_a 100
  have ha : a = 11 := 
    -- the proof follows here
    sorry
  let b := largest_b 500
  have hb : b = 22 := 
    -- the proof follows here
    sorry
  calc
    b - a + 1 = 22 - 11 + 1 : by rw [ha, hb]
           ... = 12          : by norm_num

end count_perfect_squares_between_100_and_500_l21_21499


namespace num_perfect_squares_in_range_l21_21495

theorem num_perfect_squares_in_range : 
  ∃ (k : ℕ), k = 12 ∧ ∀ n : ℕ, (100 < n^2 ∧ n^2 < 500 → 11 ≤ n ∧ n ≤ 22): sorry

end num_perfect_squares_in_range_l21_21495


namespace third_candle_remaining_fraction_l21_21271

theorem third_candle_remaining_fraction (t : ℝ) 
  (h1 : 0 < t)
  (second_candle_fraction_remaining : ℝ := 2/5)
  (third_candle_fraction_remaining : ℝ := 3/7)
  (second_candle_burned_fraction : ℝ := 3/5)
  (third_candle_burned_fraction : ℝ := 4/7)
  (second_candle_burn_rate : ℝ := 3 / (5 * t))
  (third_candle_burn_rate : ℝ := 4 / (7 * t))
  (remaining_burn_time_second : ℝ := (2 * t) / 3)
  (third_candle_burned_in_remaining_time : ℝ := (2 * t * 4) / (3 * 7 * t))
  (common_denominator_third : ℝ := 21)
  (converted_third_candle_fraction_remaining : ℝ := 9 / 21)
  (third_candle_fraction_subtracted : ℝ := 8 / 21) :
  (converted_third_candle_fraction_remaining - third_candle_fraction_subtracted) = 1 / 21 := by
  sorry

end third_candle_remaining_fraction_l21_21271


namespace min_bombings_to_destroy_tank_l21_21727

/-- 
In a 41×41 battlefield grid, a tank is initially hidden in one of the 1681 unit squares. 
A bomber makes one bombing attempt at a time. If a bomb hits an empty square, the tank remains where it is. If the bomb hits the tank, 
the tank moves to an adjacent square (squares sharing a side are considered adjacent). 
The tank must be hit twice to be destroyed. The bomber does not know the tank's position or whether it has been hit before. 
Prove that the minimum number of bombings required to ensure the tank is destroyed is 2521.
-/
theorem min_bombings_to_destroy_tank (n : ℕ) (battlefield : grid 41 41) (tank_pos : (ℕ × ℕ)) 
  (hit_once_move_to_adjacent : ∀ (pos : (ℕ × ℕ)), pos ∈ battlefield → ∃ adj_pos : (ℕ × ℕ), adj_pos ∈ battlefield ∧ are_adjacent pos adj_pos)
  (destroyed_after_two_hits : ∀ (state : tank_state), state = hit_twice → destroyed state)
  : n >= 2521 := 
sorry

end min_bombings_to_destroy_tank_l21_21727


namespace value_of_x_plus_2y_l21_21704

-- Define the conditions
variables x y : ℕ

-- Define the main problem
theorem value_of_x_plus_2y :
  x = 10 ∧ y = 5 → x + 2 * y = 20 := 
by 
  intro h,
  cases h with hx hy,
  rw [hx, hy],
  norm_num


end value_of_x_plus_2y_l21_21704


namespace cars_on_river_road_l21_21304

-- Define the number of buses and cars
variables (B C : ℕ)

-- Given conditions
def ratio_condition : Prop := (B : ℚ) / C = 1 / 17
def fewer_buses_condition : Prop := B = C - 80

-- Problem statement
theorem cars_on_river_road (h_ratio : ratio_condition B C) (h_fewer : fewer_buses_condition B C) : C = 85 :=
by
  sorry

end cars_on_river_road_l21_21304


namespace sally_wrapped_candies_count_l21_21364

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

def lcm_surface_area_sally_dean_june (sally_radius dean_radius june_radius: ℝ) : ℝ :=
  let s_sally := surface_area sally_radius
  let s_dean := surface_area dean_radius
  let s_june := surface_area june_radius
  let lcm_val := Nat.lcm (Nat.lcm (s_sally.natAbs) (s_dean.natAbs)) (s_june.natAbs)
  lcm_val * Real.pi

theorem sally_wrapped_candies_count :
  (lcm_surface_area_sally_dean_june 5 7 9) = (44100 * Real.pi) → (44100 * Real.pi) / (surface_area 5) = 441 :=
by
  sorry

end sally_wrapped_candies_count_l21_21364


namespace no_solutions_l21_21628

theorem no_solutions (x : ℝ) (hx : x ≠ 0): ¬ (12 * Real.sin x + 5 * Real.cos x = 13 + 1 / |x|) := 
by 
  sorry

end no_solutions_l21_21628


namespace find_S6_l21_21033

noncomputable def geometric_sequence_sum (S : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, S n = S 1 * ((1 - r ^ n) / (1 - r))

theorem find_S6 (S : ℕ → ℝ) (r : ℝ) (h_geo : geometric_sequence_sum S r) 
  (h_S2 : S 2 = 3) (h_S4 : S 4 = 15) :
  S 6 = 63 :=
begin
  sorry
end

end find_S6_l21_21033


namespace trigonometric_identity_l21_21028

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 :=
by
  sorry

end trigonometric_identity_l21_21028


namespace minimum_box_volume_l21_21810

def pyramid_height := 15
def pyramid_base_side := 12
def clearance := 3
def box_side := pyramid_height + clearance

theorem minimum_box_volume : box_side ^ 3 = 5832 := 
by
  unfold box_side
  unfold clearance
  unfold pyramid_height
  unfold pyramid_base_side
  sorry

end minimum_box_volume_l21_21810


namespace cosine_and_k_l21_21334

-- Definitions for the problem
variables (a k : ℝ)
variables (α : ℝ) (h base_diagonal SF : ℝ)
variables (SABCD : Type) [regular_quadrilateral_pyramid SABCD] (SO : line) [perpendicular SO (base SABCD)]

-- Given conditions
def cross_section_area_ratio (k : ℝ) :=
  let SF := a / (2 * sin (α / 2)) in
  let SO := a / 2 * cot (α / 2) in
  (((a ^ 2 * sqrt 2 * cot (α / 2)) / 4) / ((a ^ 2) / sin (α / 2))) = k

def permissible_k_values (k : ℝ) :=
  0 < k ∧ k ≤ sqrt 2 / 4

theorem cosine_and_k (h1 : cross_section_area_ratio k) (h2 : permissible_k_values k) :
  cos α = 64 * k ^ 2 - 1 ∧ 0 < k ∧ k ≤ sqrt 2 / 4 := 
sorry  -- proof to be constructed.

end cosine_and_k_l21_21334


namespace sum_of_digits_of_N_l21_21380

theorem sum_of_digits_of_N : 
  let N := (∑ i in (range 1 (501)), 10^i - 1)
  in sum_of_digits (N) = 15 :=
by
  sorry

end sum_of_digits_of_N_l21_21380


namespace math_proof_l21_21445

noncomputable def problem_statement (ξ1 ξ2 : Type) [normal_distribution ξ1 90 86] [normal_distribution ξ2 93 79] : Prop :=
  mean ξ2 > mean ξ1 ∧ variance ξ2 < variance ξ1

theorem math_proof :
  problem_statement ℝ ℝ :=
by
  simp [mean, variance]
  sorry

end math_proof_l21_21445


namespace equivalent_problem_l21_21118

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 - (Real.sqrt 2 / 2) * t, 4 - (Real.sqrt 2 / 2) * t)

def circle_cartesian (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

def point_M : ℝ × ℝ := (1, 4)

def circle_c_eqn_in_polar (theta : ℝ) : ℝ :=
  4 * Real.sin theta

theorem equivalent_problem : 
  (∀ t : ℝ, ∃ x y : ℝ, line_parametric t = (x, y) ∧ circle_cartesian x y) →
  (|Real.sqrt 2 * t_1 - 1| + |Real.sqrt 2 * t_2 - 1| = 3 * Real.sqrt 2)
  := sorry

end equivalent_problem_l21_21118


namespace sum_integers_neg50_to_65_l21_21703

theorem sum_integers_neg50_to_65 : (∑ i in Finset.range (65 + 51), (i - 50)) = 870 := by
  sorry

end sum_integers_neg50_to_65_l21_21703


namespace area_of_ABGF_l21_21721

structure Hexagon :=
  (side_length : ℝ)
  (regular : Prop)

def intersection (line1 line2 : Prop) : Prop := 
-- placeholder definition for line intersection
sorry

def area_hexagon (hex : Hexagon) : ℝ := 
  (3 * real.sqrt 3 / 2) * (hex.side_length ^ 2)

def area_triangle (base height : ℝ) : ℝ := 
  (1 / 2) * base * height

noncomputable def area_ABGF :=
  let hex := Hexagon.mk 2 (by sorry : true) in
  let area_hex := area_hexagon hex in
  let area_triangle_EDG := area_triangle 2 3 in
  area_hex - area_triangle_EDG

theorem area_of_ABGF :
  area_ABGF = 3 * real.sqrt 3 :=
  sorry

end area_of_ABGF_l21_21721


namespace rain_monday_tuesday_l21_21104

variables {α : Type*} (Ω : α → Prop) [probability_space α]
variables (A B : α → Prop)

axiom P_A : P {a | A a} = 0.70
axiom P_B : P {a | B a} = 0.55
axiom P_Ac_Bc : P {a | ¬ A a ∧ ¬ B a} = 0.35

theorem rain_monday_tuesday :
  P {a | A a ∧ B a} = 0.60 :=
by
  sorry

end rain_monday_tuesday_l21_21104


namespace binary_010_pattern_l21_21405

noncomputable def a : ℕ → ℕ
| 3 := 1
| 4 := 2
| 5 := 3
| n := if n >= 6 then 2^(n-3) - (a(n-2) + a(n-3) + 2*a(n-4)) else 0

theorem binary_010_pattern (n : ℕ) (hn : n ≥ 3) : 
  a(n) = match n with 
  | 3 := 1
  | 4 := 2
  | 5 := 3
  | k := if k ≥ 6 then 2^(k-3) - (a(k-2) + a(k-3) + 2*a(k-4)) else 0 
  end := sorry

end binary_010_pattern_l21_21405


namespace compute_product_l21_21786

noncomputable def P (x : ℂ) : ℂ := ∏ k in (finset.range 15).map (nat.cast), (x - exp (2 * pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ := ∏ j in (finset.range 12).map (nat.cast), (x - exp (2 * pi * complex.I * j / 13))

theorem compute_product : 
  (∏ k in (finset.range 15).map (nat.cast), ∏ j in (finset.range 12).map (nat.cast), (exp (2 * pi * complex.I * j / 13) - exp (2 * pi * complex.I * k / 17))) = 1 :=
by
  sorry

end compute_product_l21_21786


namespace compute_xy_l21_21280

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := 
by 
  sorry

end compute_xy_l21_21280


namespace factored_quadratic_even_b_l21_21637

theorem factored_quadratic_even_b
  (c d e f y : ℤ)
  (h1 : c * e = 45)
  (h2 : d * f = 45) 
  (h3 : ∃ b, 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) :
  ∃ b, (45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) ∧ (b % 2 = 0) :=
by
  sorry

end factored_quadratic_even_b_l21_21637


namespace parabola_vector_magnitude_sum_l21_21166

noncomputable def focus_of_parabola : ℕ -> ℕ -> Prop :=
  λ x y, y^2 = 4 * x

theorem parabola_vector_magnitude_sum
  (A B C : ℝ × ℝ) (F : ℝ × ℝ)
  (h1 : F = (1, 0))
  (h2 : focus_of_parabola (Prod.fst A) (Prod.snd A))
  (h3 : focus_of_parabola (Prod.fst B) (Prod.snd B))
  (h4 : focus_of_parabola (Prod.fst C) (Prod.snd C))
  (h5 : Prod.fst A - 1 + 2 * (Prod.fst B - 1) + 3 * (Prod.fst C - 1) = 0) :
  (metric.dist F A) + 2 * (metric.dist F B) + 3 * (metric.dist F C) = 12 :=
sorry

end parabola_vector_magnitude_sum_l21_21166


namespace odd_prime_does_not_divide_odd_nat_number_increment_l21_21987

theorem odd_prime_does_not_divide_odd_nat_number_increment (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_odd : n % 2 = 1) :
  ¬ (p * n + 1 ∣ p ^ p - 1) :=
by
  sorry

end odd_prime_does_not_divide_odd_nat_number_increment_l21_21987


namespace num_perfect_squares_in_range_l21_21491

theorem num_perfect_squares_in_range : 
  ∃ (k : ℕ), k = 12 ∧ ∀ n : ℕ, (100 < n^2 ∧ n^2 < 500 → 11 ≤ n ∧ n ≤ 22): sorry

end num_perfect_squares_in_range_l21_21491


namespace vasya_can_place_99_chips_l21_21620

theorem vasya_can_place_99_chips (occupied_cells : fin 50 → fin 50 → bool) :
  (∃ S : fin 50 → fin 50 → bool, 
   (∀ i j, occupied_cells i j = tt → S i j = ff) ∧ 
   (∀ i, even (finset.univ.filter (λ j, occupied_cells i j ∨ S i j = tt)).card) ∧ 
   (∀ j, even (finset.univ.filter (λ i, occupied_cells i j ∨ S i j = tt)).card) ∧ 
   (finset.univ.sum (λ i, finset.univ.filter (λ j, S i j = tt)).card ≤ 99)) :=
sorry

end vasya_can_place_99_chips_l21_21620


namespace sum_of_quadratics_root_condition_l21_21211

theorem sum_of_quadratics_root_condition {a1 b1 c1 a2 b2 c2 : ℝ} :
  (∀ x : ℝ, x < 1000 → (a1 * x^2 + b1 * x + c1) = 0) ∧
  (∀ x : ℝ, 1000 < x → (a2 * x^2 + b2 * x + c2) = 0) →
  ¬ (∃ x1 x2 : ℝ, x1 < 1000 ∧ 1000 < x2 ∧ (a1 + a2) * x1^2 + (b1 + b2) * x1 + (c1 + c2) = 0 ∧
    (a1 + a2) * x2^2 + (b1 + b2) * x2 + (c1 + c2) = 0) :=
begin
  sorry,
end

end sum_of_quadratics_root_condition_l21_21211


namespace complement_U_P_l21_21992

def U : Set ℝ := {x | x ≥ 0}
def P : Set ℝ := {1}

theorem complement_U_P :
  (U \ P) = (Ico 0 1 ∪ Ioi 1) :=
by
  sorry

end complement_U_P_l21_21992


namespace perimeter_of_triangle_PXY_is_40_l21_21567

-- Definitions of the given side lengths and conditions
variable (P Q R X Y I : Type)
variable (PQ PR QR : ℝ) 
variable (hPQ : PQ = 15) (hPR : PR = 25) (hQR : QR = 30)
variable (incenter_PQR : bool) -- I is the incenter for now represented as a boolean condition
variable (XY_parallel_QR : bool) -- XY is parallel to QR

-- Theorem to prove the perimeter of ΔPXY is 40
theorem perimeter_of_triangle_PXY_is_40 
  (PX : P -> X -> ℝ) (XY : X -> Y -> ℝ) (YP : Y -> P -> ℝ) 
  (h_parallel : XY_parallel_QR = true) (h_incenter : incenter_PQR = true) 
  (h1 : PX = PQ) (h2 : YP = PR) : 
  PX + XY + YP = 40 := 
sorry

end perimeter_of_triangle_PXY_is_40_l21_21567


namespace power_function_value_at_2_l21_21885

theorem power_function_value_at_2 (a : ℝ) (α : ℝ) (f : ℝ → ℝ) : 
  (a > 0) ∧ (a ≠ 1) ∧ (f = fun x => x ^ α) ∧ (∃ y, y = log a (4 - 3) + 2) ∧ (y = 2) ∧ (4 ^ α = 2) → f 2 = Real.sqrt 2 :=
by
  intros h
  sorry

end power_function_value_at_2_l21_21885


namespace part_one_part_two_l21_21037

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Given a sequence {a_n} whose sum of the first \( n \) terms \( S_n \) satisfies \( S_n = \frac{3}{2} a_n - \frac{1}{2} \)
def Sn (n : ℕ) := (3 / 2) * (a n) - (1 / 2)

-- And a sequence {b_n} \( b_n = 2 \log_3 a_n + 1 \)
def bn (n : ℕ) := 2 * real.log (a n) / real.log 3 + 1

-- Problem (I)
/-- Prove that given \( S_n = \frac{3}{2} a_n - \frac{1}{2} \) and \( a_1 = 1 \),
    the sequence \( \{a_n\} \) satisfies \( a_n = 3^{n-1} \) and \( \{b_n\} \) satisfies \( b_n = 2n - 1 \), for \( n ∈ ℕ^* \) -/
theorem part_one (n : ℕ) (h : ∀ n, Sn n = (3 / 2) * (a n) - (1 / 2)) (a1 : a 1 = 1) : a n = 3 ^ (n - 1) ∧ b n = 2 * n - 1 :=
sorry

-- Let \( c_n = \frac{b_n}{a_n} \)
def cn (n : ℕ) := bn n / a n

-- And let \( T_n \) be the sum of the first \( n \) terms of sequence {c_n}
def Tn (n : ℕ) := ∑ i in finset.range n, cn (i + 1)

-- Problem (II)
/-- Prove that if \( c_n = \frac{b_n}{a_n} \) and \( T_n \) is the sum of the first \( n \) terms of \( \{c_n\} \)
    such that \( T_n < c^2 - 2c \) for all \( n ∈ ℕ^* \), then the range of \( c \) is \( (-∞, -1] ∪ [3, +∞) \) -/
theorem part_two (n : ℕ) (Tn_lt : ∀ n, Tn n < c^2 - 2 * c) : c ≥ 3 ∨ c ≤ -1 :=
sorry

end part_one_part_two_l21_21037


namespace range_of_a_l21_21837

/--
Given the sets A = {x | |x - 1| ≤ 2} and B = {x | x - a > 0}, 
if A ∪ B = B, then the range of the real number a is (-∞, -1).
-/
theorem range_of_a (a : ℝ) :
  (∃ A B: set ℝ, (A = {x : ℝ | |x - 1| ≤ 2}) ∧ (B = {x : ℝ | x - a > 0}) ∧ A ∪ B = B) → a < -1 :=
  by
    sorry

end range_of_a_l21_21837


namespace inequality_proof_l21_21621

theorem inequality_proof {k l m n : ℕ} (h_pos_k : 0 < k) (h_pos_l : 0 < l) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_klmn : k < l ∧ l < m ∧ m < n)
  (h_equation : k * n = l * m) : 
  (n - k) / 2 ^ 2 ≥ k + 2 := 
by sorry

end inequality_proof_l21_21621


namespace prove_annual_interest_rate_l21_21287

noncomputable def annual_compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  real.root (n * t) (A / P) - 1

theorem prove_annual_interest_rate :
  annual_compound_interest_rate 780 1300 1 4 = 0.129830674 :=
by
  -- Here 0.129830674 corresponds to 12.98% interest rate in decimal form
  sorry

end prove_annual_interest_rate_l21_21287


namespace expression_evaluation_l21_21626

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 
  ((2 * x - 2) / x - 1) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - x))

-- State the theorem to be proven
theorem expression_evaluation (x : ℝ) (hx : x = 4) : given_expression x = 3 / 2 := 
by 
  sorry

end expression_evaluation_l21_21626


namespace solve_quadratic_l21_21632

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l21_21632


namespace jim_out_of_pocket_cost_l21_21577

theorem jim_out_of_pocket_cost {price1 price2 sale : ℕ} 
    (h1 : price1 = 10000)
    (h2 : price2 = 2 * price1)
    (h3 : sale = price1 / 2) :
    (price1 + price2 - sale = 25000) :=
by
  sorry

end jim_out_of_pocket_cost_l21_21577


namespace remove_cubes_to_form_cave_l21_21952

def isCube (x : ℕ × ℕ × ℕ) : Prop :=
  x.1 < 5 ∧ x.2 < 5 ∧ x.3 < 3

def is1x1x2Parallelepiped (x : ℕ × ℕ × ℕ) : Prop :=
  ∃ i j k, (x = (i, j, k) ∨ x = (i+1, j, k))

def isCaveExit (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (2, 0, 1) ∨ x = (2, 1, 1))

theorem remove_cubes_to_form_cave :
  ∃ (removed : set (ℕ × ℕ × ℕ)),
    (∀ x, isCube x → x ∉ removed ∨ is1x1x2Parallelepiped x)
    ∧ (∀ x, isCaveExit x → x ∉ removed)
    ∧ (∀ x y z, (x, y, z) ∈ removed → (x ≠ 2 ∨ y ≠ 1 ∨ z ≠ 1) 
                 ∧ (x ≠ 2 ∨ y ≠ 0 ∨ z ≠ 1)
                 ∧ (x ≠ 3 ∨ y ≠ 1 ∨ z ≠ 1) 
                 ∧ (x ≠ 3 ∨ y ≠ 0 ∨ z ≠ 1))
    sorry

end remove_cubes_to_form_cave_l21_21952


namespace sqrt_three_irrational_sqrt_three_infinite_non_repeating_decimal_l21_21951

theorem sqrt_three_irrational 
  (h : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p * p = 3 * q * q ∧ Int.gcd p q = 1)) : 
  ∃ d : ℚ, d * d = 3 → false :=
by sorry

theorem sqrt_three_infinite_non_repeating_decimal :
  ∃ r : ℝ, r * r = 3 ∧ irrational r :=
by sorry

end sqrt_three_irrational_sqrt_three_infinite_non_repeating_decimal_l21_21951


namespace minimum_crooks_l21_21547

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l21_21547


namespace sequence_sum_l21_21001

theorem sequence_sum (a : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : a 2 = 6)
  (h₃ : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 2) :
  ⌊∑ i in range 2017 + 1, 2017 / a i⌋ = 2016 :=
by sorry

end sequence_sum_l21_21001


namespace even_of_even_square_sqrt_two_irrational_l21_21311

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l21_21311


namespace find_t_l21_21227

-- Define the utility on both days
def utility_monday (t : ℝ) := t * (10 - t)
def utility_tuesday (t : ℝ) := (4 - t) * (t + 5)

-- Define the total hours spent on activities condition for both days
def total_hours_monday (t : ℝ) := t + (10 - t)
def total_hours_tuesday (t : ℝ) := (4 - t) + (t + 5)

theorem find_t : ∃ t : ℝ, t * (10 - t) = (4 - t) * (t + 5) ∧ 
                            total_hours_monday t ≥ 8 ∧ 
                            total_hours_tuesday t ≥ 8 :=
by
  sorry

end find_t_l21_21227


namespace measure_GAC_triangle_pentagon_l21_21378

theorem measure_GAC_triangle_pentagon :
  ∀ {A B C D E F G : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]
  (triangle_ABC: RightAngledIsoscelesTriangle A B C (pi / 2))
  (pentagon_BDEFG: RegularPentagon B D E F G)
  (shared_vertex: CommonVertex B triangle_ABC pentagon_BDEFG),
  angle G A C = 18 :=
by
  sorry

end measure_GAC_triangle_pentagon_l21_21378


namespace min_value_fraction_sum_l21_21042

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (4 / (x + 2) + 1 / (y + 1)) ≥ 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l21_21042


namespace sum_smallest_10_S_n_div_by_5_eq_285_l21_21023

def S_n (n : ℕ) : ℕ :=
  (n - 1) * n * (n + 1) * (3 * n + 2) / 24

theorem sum_smallest_10_S_n_div_by_5_eq_285 : ∑ k in (Finset.range 10).map (λ i, 5 * i + 6), id = 285 := by
  sorry

end sum_smallest_10_S_n_div_by_5_eq_285_l21_21023


namespace polynomial_roots_property_l21_21981

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l21_21981


namespace count_perfect_squares_between_100_500_l21_21486

theorem count_perfect_squares_between_100_500 :
  ∃ (count : ℕ), count = finset.card ((finset.Icc 11 22).filter (λ n, 100 < n^2 ∧ n^2 < 500)) :=
begin
  use 12,
  rw ← finset.card_Icc,
  sorry
end

end count_perfect_squares_between_100_500_l21_21486


namespace lara_flowers_l21_21165

theorem lara_flowers (M : ℕ) : 52 - M - (M + 6) - 16 = 0 → M = 15 :=
by
  sorry

end lara_flowers_l21_21165


namespace part_a_l21_21301

-- Define the sequences and their properties
variables {n : ℕ} (h1 : n ≥ 3)
variables (a b : ℕ → ℝ)
variables (h_arith : ∀ k, a (k+1) = a k + d)
variables (h_geom : ∀ k, b (k+1) = b k * q)
variables (h_a1_b1 : a 1 = b 1)
variables (h_an_bn : a n = b n)

-- State the theorem to be proven
theorem part_a (k : ℕ) (h_k : 2 ≤ k ∧ k ≤ n - 1) : a k > b k :=
  sorry

end part_a_l21_21301


namespace smallest_n_integer_expression_l21_21167

def a : ℝ := Real.pi / 2008

def is_integer_expression (n : ℕ) : ℝ :=
  2 * ∑ k in Finset.range (n + 1), (Real.cos (↑k ^ 2 * a) * Real.sin (↑k * a))

theorem smallest_n_integer_expression :
  (∃ (n : ℕ), 0 < n ∧ ∃ m : ℤ, is_integer_expression n = m) ∧ 
  (∀ k : ℕ, 0 < k ∧ k < 251 → ¬ (∃ m : ℤ, is_integer_expression k = m)) :=
begin
  sorry
end

end smallest_n_integer_expression_l21_21167


namespace fraction_identity_l21_21173

variables {a b c x : ℝ}

theorem fraction_identity (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : c ≠ a) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + 2 * b + 3 * c) / (a - b - 3 * c) = (b * (x + 2) + 3 * c) / (b * (x - 1) - 3 * c) :=
by {
  sorry
}

end fraction_identity_l21_21173


namespace problem_solution_l21_21783

noncomputable def product(problem1: ℕ, problem2: ℕ): ℂ :=
∏ k in (finset.range problem1).image (λ (n : ℕ), e ^ (2 * π * complex.I * n / 17)),
  ∏ j in (finset.range problem2).image (λ (m : ℕ), e ^ (2 * π * complex.I * m / 13)),
    (j - k)

theorem problem_solution : product 15 12 = 13 := 
sorry

end problem_solution_l21_21783


namespace ratio_benedict_helen_l21_21910

theorem ratio_benedict_helen
  (lottery_winning : ℕ = 100)
  (paid_colin : ℕ = 20)
  (paid_helen : ℕ = 2 * paid_colin)
  (remaining_money : ℕ = 20)
  (total_debt_paid : ℕ := lottery_winning - remaining_money)
  (paid_benedict : ℕ := total_debt_paid - paid_colin - paid_helen) :
  paid_benedict / paid_helen = 1 / 2 :=
sorry

end ratio_benedict_helen_l21_21910


namespace trains_crossing_time_l21_21282

theorem trains_crossing_time
  (train_length : ℕ)
  (time_first_train : ℕ)
  (time_second_train : ℕ)
  (relative_speed : ℝ)
  (crossing_time : ℝ)
  (train_length = 120)
  (time_first_train = 10)
  (time_second_train = 20)
  (relative_speed = (train_length / time_first_train) + (train_length / time_second_train))
  (crossing_time = (2 * train_length) / relative_speed) :
  crossing_time = 13.33 := sorry

end trains_crossing_time_l21_21282


namespace remainder_when_divided_by_100_l21_21914

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end remainder_when_divided_by_100_l21_21914


namespace diagonal_length_is_correct_l21_21758

noncomputable def length_of_diagonal 
  (A B C D : ℝ) 
  (base1 base2 side : ℝ)
  (h₁ : A ≤ B) 
  (h₂ : C ≤ D) 
  (h₃ : B - A = 24) 
  (h₄ : D - C = 12) 
  (h₅ : side = 13) :
  (diag : ℝ) := 13

theorem diagonal_length_is_correct 
  (A B C D : ℝ) 
  (base1 base2 side : ℝ)
  (h₁ : A ≤ B) 
  (h₂ : C ≤ D) 
  (h₃ : B - A = 24) 
  (h₄ : D - C = 12) 
  (h₅ : side = 13) :
  length_of_diagonal A B C D base1 base2 side h₁ h₂ h₃ h₄ h₅ = 13 :=
by 
  -- proof goes here
  sorry

end diagonal_length_is_correct_l21_21758


namespace sum_of_squares_as_fraction_l21_21047

-- Definitions and conditions
variables (a b c d : ℤ)
variable (p : ℤ)
variable (m : ℤ)

-- Setting up the conditions
def condition1 : Prop := Nat.Prime p
def condition2 : Prop := p = a^2 + b^2
def condition3 : Prop := p ∣ (c^2 + d^2)

-- The theorem statement
theorem sum_of_squares_as_fraction (h1 : condition1 p) (h2 : condition2 a b p) (h3 : condition3 a b c d p) :
  ∃ t s : ℤ, (c^2 + d^2) / p = t^2 + s^2 := 
sorry

end sum_of_squares_as_fraction_l21_21047


namespace count_perfect_squares_between_100_and_500_l21_21497

def smallest_a (x : ℕ) : ℕ :=
  Nat.find (λ a, a^2 > x)

def largest_b (x : ℕ) : ℕ :=
  Nat.find (λ b, b^2 > x) - 1

theorem count_perfect_squares_between_100_and_500 :
  let a := smallest_a 100
  let b := largest_b 500
  b - a + 1 = 12 :=
by
  -- Definitions based on conditions
  let a := smallest_a 100
  have ha : a = 11 := 
    -- the proof follows here
    sorry
  let b := largest_b 500
  have hb : b = 22 := 
    -- the proof follows here
    sorry
  calc
    b - a + 1 = 22 - 11 + 1 : by rw [ha, hb]
           ... = 12          : by norm_num

end count_perfect_squares_between_100_and_500_l21_21497


namespace collinear_points_l21_21075

noncomputable def vectors := Type

variables (a b : vectors) (A B C D : vectors)

def AB := a + 2 * b
def BC := -5 * a + 6 * b
def CD := 7 * a - 2 * b

theorem collinear_points (h1 : B - A = AB a b) (h2 : C - B = BC a b) (h3 : D - C = CD a b) :
  ∃ k : ℝ, (B - D) = k • (B - A) := by
  sorry

end collinear_points_l21_21075


namespace sum_fraction_series_l21_21284

theorem sum_fraction_series (n : ℕ) (h : n > 0) :
  ∑ i in Finset.range n, 1 / (2 * (i + 1) * (2 * (i + 1) + 2)) = n / (4 * (n + 1)) := by
sorry

end sum_fraction_series_l21_21284


namespace tetrahedron_formable_l21_21853

theorem tetrahedron_formable (x : ℝ) (hx_pos : 0 < x) (hx_bound : x < (Real.sqrt 6 + Real.sqrt 2) / 2) :
  true := 
sorry

end tetrahedron_formable_l21_21853


namespace point_distance_to_y_axis_l21_21535

def point := (x : ℝ , y : ℝ)

def distance_to_y_axis (p : point) : ℝ :=
  |p.1|

theorem point_distance_to_y_axis (P : point):
  P = (-3, 4) →
  distance_to_y_axis P = 3 :=
by
  intro h
  rw [h, distance_to_y_axis]
  sorry

end point_distance_to_y_axis_l21_21535


namespace even_of_even_square_sqrt_two_irrational_l21_21312

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l21_21312


namespace sin_angle_BAO_of_rectangle_l21_21117

theorem sin_angle_BAO_of_rectangle
  (ABCD : Type*)
  [Rectangle ABCD]
  (A B C D : ABCD)
  (O : Point)
  (AB BC : ℝ)
  (h1 : AB = 12)
  (h2 : BC = 16)
  (h3 : is_diagonal_intersection A C B D O)
  (h4 : are_perpendicular AB BC) :
  sin_angle BAO = 0.6 :=
  sorry

end sin_angle_BAO_of_rectangle_l21_21117


namespace find_phone_number_l21_21209

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10 in s = s.reverse

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def has_three_consecutive_ones (n : ℕ) : Prop :=
  let s := n.digits 10 in s.contains [1, 1, 1]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

theorem find_phone_number : 
  ∃ n,
    n.digits 10 = [5,6,7,1,1,1,7] ∧  -- The number is 7111765
    n.digits 10.length = 7 ∧          -- The phone number has 7 digits
    is_consecutive 5 6 7 ∧            -- The last three digits are consecutive (765)
    is_palindrome (n / 10000) ∧       -- The first five digits form a palindrome (71117)
    (n / 1000) % 10 ^ 3 % 9 = 0 ∧     -- The three-digit number formed by the first three digits is divisible by 9 (711 is divisible by 9)
    has_three_consecutive_ones n ∧    -- The phone number contains exactly three consecutive ones
    (is_prime (n.digits 10.drop 3.take 2).to_nat xor 
     is_prime (n.digits 10.drop 5.take 2).to_nat) := -- Only one of the two-digit numbers obtained (from the groups) is prime (76 or 65)
sorry

end find_phone_number_l21_21209


namespace find_phone_number_l21_21206

-- Define the conditions
def is_palindrome (l: List ℕ) : Prop := l = l.reverse
def is_divisible_by_9 (n: ℕ) : Prop := n % 9 = 0
def is_prime (n: ℕ) : Prop := nat.prime n
def contains_three_consecutive_ones (n: ℕ) : Prop := ∃ i, (n.to_string.nth i = '1') ∧ (n.to_string.nth (i+1) = '1') ∧ (n.to_string.nth (i+2) = '1')
def last_three_consecutive (n: ℕ) : Prop := 
  let digits := n.to_string.data.drop 4 in
  (∃ x1 x2 x3 : ℕ, [x1, x2, x3] = digits.map (λ c, c.to_digit 10).reverse ∧ (x1 + 1 = x2) ∧ (x2 + 1 = x3))

-- Define the goal statement
theorem find_phone_number (n: ℕ) (h1: n / 10^6 < 10) (h2: contains_three_consecutive_ones n)
  (h3: last_three_consecutive n) (h4: is_palindrome ([n / 10^4 % 10, n / 10^3 % 10, n / 10^2 % 10, n / 10^1 % 10, n % 10]))
  (h5: is_divisible_by_9 (n / 10^4)) 
  (h6: ∃ m, is_prime m ∧ (n / 10 % 100 = m ∨ n % 100 = m)) :
  n = 7111765 := 
  sorry

end find_phone_number_l21_21206


namespace proof_problem_l21_21777

-- Define complex numbers for the roots
def P (x : ℂ) := ∏ k in Finset.range 15, (x - complex.exp (2 * real.pi * k * complex.I / 17))

def Q (x : ℂ) := ∏ j in Finset.range 12, (x - complex.exp (2 * real.pi * j * complex.I / 13))

-- Conditions as Lean definitions
noncomputable def e_k (k : ℕ) (h : k < 16) : ℂ := complex.exp (2 * real.pi * k * complex.I / 17)
noncomputable def e_j (j : ℕ) (h : j < 13) : ℂ := complex.exp (2 * real.pi * j * complex.I / 13)

theorem proof_problem : 
  (∏ k in Finset.range 15, ∏ j in Finset.range 12, (e_j j (by linarith) - e_k k (by linarith))) = 1 :=
sorry

end proof_problem_l21_21777


namespace abs_frac_lt_one_l21_21219

theorem abs_frac_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |(x - y) / (1 - x * y)| < 1 :=
sorry

end abs_frac_lt_one_l21_21219


namespace sample_group_frequencies_l21_21848

theorem sample_group_frequencies :
  (sample_volume = 80)
  (number_of_groups = 6)
  (freq3 = 10)
  (freq4 = 12)
  (freq5 = 14)
  (freq6 = 20)
  (freq_ratio1 = 0.2)
  → (frequency1 = 16)
  ∧ (frequency_rate2 = 0.1) :=
by
  sorry

end sample_group_frequencies_l21_21848


namespace division_result_l21_21407

theorem division_result : (0.284973 / 29 = 0.009827) := 
by sorry

end division_result_l21_21407


namespace altitude_product_l21_21046

theorem altitude_product 
  (ABC : Type) [right_triangle ABC]
  (A B C D: ABC)
  (h1: hypotenuse AB)
  (h2: altitude CD AB)
  (h3: CD = 4) :
  AD * BD = 16 :=
by
  sorry

end altitude_product_l21_21046


namespace angle_CFD_is_right_angle_l21_21216

theorem angle_CFD_is_right_angle
  (A B C D E F : Type)
  [metric_space A] [add_comm_group A] [module ℝ A]
  [metric_space B] [add_comm_group B] [module ℝ B]
  [metric_space C] [add_comm_group C] [module ℝ C]
  [metric_space D] [add_comm_group D] [module ℝ D]
  [metric_space E] [add_comm_group E] [module ℝ E]
  [metric_space F] [add_comm_group F] [module ℝ F]
  (AB AD BC DE EC : ℝ)
  (parallelogram : ∀ (E : A), (AB C D + AB D E = 0))
  (midpoint : ∀ (E : A), E = (AB / 2))
  (AD_eq_BF : ∀ (F A D B : A), AD = BF) :
  ∠ (C F D) = 90 :=
begin
  sorry,
end

end angle_CFD_is_right_angle_l21_21216


namespace isosceles_triangle_perimeter_l21_21111

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 4) (isosceles : (a = b) ∨ (a = 2) ∨ (b = 2)) :
  (a = 2 ∧ b = 4 → 10) :=
begin
  -- assuming isosceles triangle means either two sides are equal or a = 2 or b = 2 which fits the isosceles definition in the context of provided lengths.
  sorry
end

end isosceles_triangle_perimeter_l21_21111


namespace find_k_l21_21598

theorem find_k (k : ℝ) (h1 : k > 1) (h2 : ∑ n : ℕ in (set.univ \ {0}).to_finset, ((7 * n - 3) / k ^ n) = 5) : k = 1.2 + 0.2 * real.sqrt 46 :=
sorry

end find_k_l21_21598


namespace trajectory_equation_l21_21605

noncomputable def ellipse_equation (a b : ℝ) (h_ab : a > b > 0) (h_passes : b = 4) (h_eccentricity : 3 * a = 5 * (Real.sqrt (a^2 - b^2))) : Prop :=
    (a = 5) ∧ (b = 4) ∧ a^2 = b^2 + 9

theorem trajectory_equation (x y : ℝ) (h : { (p : ℝ × ℝ) | (4/25) * x^2 + (1/16) * y^2 = 1 } (3, 0)) :
  16 * x^2 + 25 * y^2 - 48 * x = 0 :=
sorry

end trajectory_equation_l21_21605


namespace minimum_sin_condition_l21_21098

variable {a b c : ℝ}
variable {B : ℝ}

-- conditions
def f (x : ℝ) : ℝ :=
  (1/3) * x^3 + b * x^2 + (a^2 + c^2 - a * c) * x + 1

-- Given that f has extreme points
def hasExtremePoints : Prop :=
  ∃ x : ℝ, f (x) = 0 ∧ (∃ y : ℝ, y ≠ x ∧ f' y = 0)

noncomputable def discriminant : ℝ :=
  (2 * b) ^ 2 - 4 * (a^2 + c^2 - a * c)

-- $\Delta > 0$ implies ac > a^2 + c^2 - b^2
theorem minimum_sin_condition (h : hasExtremePoints) (hΔ : discriminant > 0) :
  sin (2 * B - π / 3) = -1 :=
sorry

end minimum_sin_condition_l21_21098


namespace data_set_range_l21_21339

/-- Prove that the range of the set {-1, -2, 3, 4, 5} is equal to 7. -/
theorem data_set_range : 
  let data := {-1, -2, 3, 4, 5}
  in (set.max data - set.min data) = 7 :=
by
  sorry

end data_set_range_l21_21339


namespace unique_corresponding_point_l21_21281

-- Define the points for the squares
structure Point := (x : ℝ) (y : ℝ)

structure Square :=
  (a b c d : Point)

def contains (sq1 sq2: Square) : Prop :=
  sq2.a.x >= sq1.a.x ∧ sq2.a.y >= sq1.a.y ∧
  sq2.b.x <= sq1.b.x ∧ sq2.b.y >= sq1.b.y ∧
  sq2.c.x <= sq1.c.x ∧ sq2.c.y <= sq1.c.y ∧
  sq2.d.x >= sq1.d.x ∧ sq2.d.y <= sq1.d.y

theorem unique_corresponding_point
  (sq1 sq2 : Square)
  (h1 : contains sq1 sq2)
  (h2 : sq1.a.x - sq1.c.x = sq2.a.x - sq2.c.x ∧ sq1.a.y - sq1.c.y = sq2.a.y - sq2.c.y):
  ∃! (O : Point), ∃ O' : Point, contains sq1 sq2 ∧ 
  (O.x - sq1.a.x) / (sq1.b.x - sq1.a.x) = (O'.x - sq2.a.x) / (sq2.b.x - sq2.a.x) ∧ 
  (O.y - sq1.a.y) / (sq1.d.y - sq1.a.y) = (O'.y - sq2.a.y) / (sq2.d.y - sq2.a.y) := 
sorry

end unique_corresponding_point_l21_21281


namespace perfect_cube_prime_addition_l21_21319

theorem perfect_cube_prime_addition (x : ℕ) :
  (27 = 3 ^ 3) → (prime 3) → (prime (3 + x)) → x = 2 :=
by
  intros h1 h2 h3
  sorry

end perfect_cube_prime_addition_l21_21319


namespace seating_arrangements_l21_21936

-- Definitions and conditions
def total_people : ℕ := 10
def Alice := 1
def Bob := 1
def Cindy := 1
def Dave := 1
def Emma := 1

-- The condition that Alice, Bob, and Cindy refuse to sit in three consecutive seats
def abc_not_consecutive (seating: list ℕ) : Prop := 
  -- implementation of this condition would be complex, but we state it as a Prop
  True -- placeholder for actual condition checking

-- The condition that Dave and Emma must sit next to each other
def de_together (seating: list ℕ) : Prop :=
  -- implementation of this condition would be complex, but we state it as a Prop
  True -- placeholder for actual condition checking

-- Now the main theorem statement
theorem seating_arrangements : ∃ seating : list ℕ, abc_not_consecutive seating ∧ de_together seating ∧ (seating.length = total_people ∧ (factorial 10 - (factorial 8 * factorial 3 + factorial 9 * 2 - factorial 8 * factorial 3 * 2)) = 3144960) :=
  sorry

end seating_arrangements_l21_21936


namespace sum_of_coefficients_l21_21681

theorem sum_of_coefficients : ∃ (a b c d : ℕ), 
  (∀ (x y : ℝ), x + y = 5 ∧ 3 * x * y = 5 → 
  (x = (a + b * real.sqrt c) / d ∨ x = (a - b * real.sqrt c) / d) 
  ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  ∧ a + b + c + d = 63 :=
sorry

end sum_of_coefficients_l21_21681


namespace find_irrational_satisfying_conditions_l21_21015

-- Define a real number x which is irrational
def is_irrational (x : ℝ) : Prop := ¬∃ (q : ℚ), (x : ℝ) = q

-- Define that x satisfies the given conditions
def rational_conditions (x : ℝ) : Prop :=
  (∃ (r1 : ℚ), x^3 - 17 * x = r1) ∧ (∃ (r2 : ℚ), x^2 + 4 * x = r2)

-- The main theorem statement
theorem find_irrational_satisfying_conditions (x : ℝ) 
  (hx_irr : is_irrational x) 
  (hx_cond : rational_conditions x) : x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
by
  sorry

end find_irrational_satisfying_conditions_l21_21015


namespace goldfish_equal_months_l21_21365

theorem goldfish_equal_months :
  ∃ (n : ℕ), 
    let B_n := 3 * 3^n 
    let G_n := 125 * 5^n 
    B_n = G_n ∧ n = 5 :=
by
  sorry

end goldfish_equal_months_l21_21365


namespace unique_x_value_l21_21794

def sequence_term (a b : ℝ) : ℝ := (b + 2) / (2 * a)

def appears_in_sequence (x : ℝ) (target : ℝ) (n : ℕ) : Prop :=
  ∃ (seq : ℕ → ℝ), seq 1 = x ∧ seq 2 = 3000 ∧ (∀ k, k ≥ 2 → seq (k+1) = sequence_term (seq k) (seq (k-1)))
                    ∧ ∃ m < n, seq m = target

theorem unique_x_value :
  ∃! x : ℝ, 0 < x ∧ ∃ n : ℕ, appears_in_sequence x 3001 n :=
sorry

end unique_x_value_l21_21794


namespace projection_is_circumcenter_l21_21130

-- Define a structure for the tetrahedron P-ABC with conditions PA = PB = PC
structure Tetrahedron (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
(equality : dist P A = dist P B ∧ dist P B = dist P C)

-- Define a function that provides the projection of P onto the plane ABC
noncomputable def projection (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] : Type :=
sorry -- This can be defined as the unique point O on the plane ABC with equal distances to A, B, and C under the conditions provided.

-- The theorem we want to prove
theorem projection_is_circumcenter
  {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (tet : Tetrahedron P A B C) :
  let O := projection P A B C in ∀ Q : Type, [MetricSpace Q] → (dist Q A = dist Q B ∧ dist Q B = dist Q C) → Q = O :=
sorry

-- This statement follows directly from the problem and given solution.

end projection_is_circumcenter_l21_21130


namespace not_four_cyclic_l21_21931

theorem not_four_cyclic (A B C D E F G : Point) (h_convex : ConvexHeptagon A B C D E F G) :
  ¬ (cyclicQuadrilateral A B C D ∧ cyclicQuadrilateral B C D E ∧ cyclicQuadrilateral C D E F ∧ cyclicQuadrilateral D E F G) :=
sorry

end not_four_cyclic_l21_21931


namespace cot_diff_equal_l21_21147

variable (A B C D : Type)

-- Define the triangle and median.
variable [triangle ABC : Type] (median : Type)

-- Define the angle condition.
def angle_condition (ABC : triangle) (AD : median) : Prop :=
  ∠(AD, BC) = 60

-- Prove the cotangent difference
theorem cot_diff_equal
  (ABC : triangle)
  (AD : median)
  (h : angle_condition ABC AD) :
  abs (cot B - cot C) = (9 - 3 * sqrt 3) / 2 :=
by
  sorry -- Proof to be constructed

end cot_diff_equal_l21_21147


namespace divides_all_four_digit_palindromes_l21_21333

theorem divides_all_four_digit_palindromes :
  ∃ d > 1, ∀ (a b : ℕ), a < 10 → b < 10 → d ∣ (1001 * a + 110 * b) :=
by
  use 11
  split
  · norm_num
  · intros a b ha hb
    sorry

end divides_all_four_digit_palindromes_l21_21333


namespace product_complex_l21_21791

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem product_complex : 
  ∏ k in finset.range 15, (∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
sorry

end product_complex_l21_21791


namespace minimum_crooks_l21_21549

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l21_21549


namespace lifespan_mistake_l21_21720

/-- Conditions: Expected lifespans of sensor and transmitter --/
variables (ξ η : ℝ) 
variables (E_ξ : E[ξ] = 3) 
variables (E_η : E[η] = 5)

theorem lifespan_mistake : 
  E[min ξ η] = 11 / 3 → 
  E[min ξ η] ≤ 3 :=
by {
  intro h,
  rw h,
  linarith,
}

end lifespan_mistake_l21_21720


namespace part_1_part_2_l21_21180

variables {n : ℕ} (x : ℕ → ℝ) (i : ℕ) 

def p (x : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, x i
def q (x : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, ∑ j in finset.range i, x i * x j

theorem part_1 (n_ge_3 : n ≥ 3) : 
  let p := p x n, q := q x n in 
  (n - 1) * p^2 / n - 2 * q ≥ 0 := 
sorry

theorem part_2 (n_ge_3 : n ≥ 3) (i_range : i < n) : 
  let p := p x n, q := q x n in 
  |x i - p / n| ≤ (n - 1) / n * sqrt (p^2 - 2 * n / (n - 1) * q) := 
sorry

end part_1_part_2_l21_21180


namespace min_na_l21_21448

-- Given conditions: S_n = n^2 - 10n
def S : ℕ → ℤ
| n := n^2 - 10 * n

-- a_n defined based on given conditions
def a : ℕ → ℤ
| 1 := S 1
| n := if n = 1 then S 1 else S n - S (n - 1)

-- Define na_n as n * a_n
def na : ℕ → ℤ
| n := n * a n

-- The goal is to find n with the minimum value in the sequence {na_n}
def n_min := 3

theorem min_na : ∃ n_min : ℕ, n_min = 3 ∧ ∀ n ≥ 1, na n ≥ na n_min :=
by
  sorry

end min_na_l21_21448


namespace solution_interval_l21_21817

theorem solution_interval (x : ℝ) : 
  (3/8 + |x - 1/4| < 7/8) ↔ (-1/4 < x ∧ x < 3/4) := 
sorry

end solution_interval_l21_21817


namespace part1_monotonicity_part2_two_zeros_l21_21462
noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x + 2)

theorem part1_monotonicity (x : ℝ) : 
  ∃ (a : ℝ), a = 1 ∧ (∀ x, x < 0 → deriv (λ x, f x 1) x < 0) ∧ (∀ x, x > 0 → deriv (λ x, f x 1) x > 0) := sorry

theorem part2_two_zeros (a : ℝ) :
  (∀ x, f x a = 0 → f x a < 0) → (a > 1 / Real.exp 1) := sorry

end part1_monotonicity_part2_two_zeros_l21_21462


namespace limsup_ge_e_l21_21426

theorem limsup_ge_e (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) :
  filter.limsup at_top (λ n, (a 1 + a (n + 1)) / (a n)) ^ n ≥ real.exp 1 :=
begin
  sorry
end

end limsup_ge_e_l21_21426


namespace bounds_on_a_k_l21_21667

theorem bounds_on_a_k 
  (a : Fin 100 → ℝ) 
  (h : ∑ i, (a i)^2 + (∑ i, a i)^2 = 101) : 
  ∀ k, |a k| ≤ 10 :=
by 
  sorry

end bounds_on_a_k_l21_21667


namespace num_quadrilaterals_containing_P_even_l21_21376

def is_convex (vertices : list ℝ × ℝ) : Prop := sorry

def no_three_diagonals_intersect (vertices : list ℝ × ℝ) : Prop := sorry

def lies_inside (P : ℝ × ℝ) (vertices : list ℝ × ℝ) : Prop := sorry

def does_not_lie_on_diagonals (P : ℝ × ℝ) (vertices : list ℝ × ℝ) : Prop := sorry

theorem num_quadrilaterals_containing_P_even 
  (k : ℕ) (P : ℝ × ℝ) (vertices : list ℝ × ℝ)
  (h_convex : is_convex vertices)
  (h_vertices_count : vertices.length = 4 * k + 3)
  (h_no_intersect : no_three_diagonals_intersect vertices)
  (h_inside : lies_inside P vertices)
  (h_not_on_diagonals : does_not_lie_on_diagonals P vertices) :
  ∃ n : ℕ, n % 2 = 0 ∧ n = count_quadrilaterals_containing_P vertices P := sorry

end num_quadrilaterals_containing_P_even_l21_21376


namespace stuart_segments_to_return_l21_21238

open Real

noncomputable def mABC := 60 -- in degrees
noncomputable def arcAC := 2 * mABC
noncomputable def arcAB := arcAC / 2

/-- The number of segments Stuart draws to return to starting point at A -/
theorem stuart_segments_to_return : ∃ m : ℕ, let n := 3 * m in n = 3 :=
by
  use 1
  sorry

end stuart_segments_to_return_l21_21238


namespace midpoint_between_points_l21_21401

theorem midpoint_between_points : 
  let (x1, y1, z1) := (2, -3, 5)
  let (x2, y2, z2) := (8, 1, 3)
  (1 / 2 * (x1 + x2), 1 / 2 * (y1 + y2), 1 / 2 * (z1 + z2)) = (5, -1, 4) :=
by
  sorry

end midpoint_between_points_l21_21401


namespace probability_X_geq_0_l21_21870

noncomputable def normalDist : Type := sorry  -- Placeholder for the normal distribution definition

variables {σ : ℝ}  -- Variance of the normal distribution

-- Assuming a random variable X follows a normal distribution with mean 1 and variance σ²
axiom X_normal : normalDist
axiom mean_one : ∀ X : normalDist, X = 1
axiom variance : ∀ X : normalDist, sorry  -- Placeholder for the variance definition
axiom P_X_gt_2 : ∀ X : normalDist, P (X > 2) = 0.3

theorem probability_X_geq_0 : P (X ≥ 0) = 0.7 :=
by
  sorry  -- Proof is omitted

end probability_X_geq_0_l21_21870


namespace intersection_point_polar_coords_l21_21127

open Real

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def curve_C2 (t x y : ℝ) : Prop :=
  (x = 2 - t) ∧ (y = t)

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), (ρ = sqrt 2) ∧ (θ = π / 4) ∧
  ∃ (x y t : ℝ), curve_C2 t x y ∧ curve_C1 x y ∧
  (ρ = sqrt (x^2 + y^2)) ∧ (tan θ = y / x) :=
by
  sorry

end intersection_point_polar_coords_l21_21127


namespace compute_value_l21_21375

theorem compute_value : 302^2 - 298^2 = 2400 :=
by
  sorry

end compute_value_l21_21375


namespace orthocenter_triangle_AXY_lies_on_line_BD_l21_21933

noncomputable theory

variable {ABCD : Type} [Plane ABCD]
variable {A B C D X Y : Point}

-- Conditions given in the problem
axiom diagonal_AC_bisector_∠BAD : bisects (diagonal A C) (∠ B A D)
axiom ∠ADC_eq_∠ACB : ∠ D A C = ∠ A C B
axiom X_perpendicular_A_BC : perpendicular_from_to (A) (X) (line B C)
axiom Y_perpendicular_A_CD : perpendicular_from_to (A) (Y) (line C D)

-- Definition to represent the orthocenter of a triangle
def orthocenter (P Q R : Point) : Point := sorry

-- Statement of the problem
theorem orthocenter_triangle_AXY_lies_on_line_BD :
  lies_on (orthocenter A X Y) (line B D) :=
sorry

end orthocenter_triangle_AXY_lies_on_line_BD_l21_21933


namespace multiplication_mistake_l21_21327

theorem multiplication_mistake (x : ℕ) (H : 43 * x - 34 * x = 1215) : x = 135 :=
sorry

end multiplication_mistake_l21_21327


namespace triangles_congruent_l21_21228

open Geometry

-- We define the geometric entities and hypotheses.
variables (A B C D O : Point)
variables (AC BD : Line)
variables (h_intersect_AC_BD : Intersects AC BD O)
variables (h_angle : ∠ BAO = ∠ DCO)
variables (h_side : Segment AO = Segment OC)

-- We state the theorem to prove the congruence of the triangles.
theorem triangles_congruent :
  ∠ BAO = ∠ DCO → 
  Segment AO = Segment OC → 
  Intersects AC BD O → 
  CongruentTriangle ΔBAO ΔDCO := by
  sorry

end triangles_congruent_l21_21228


namespace largest_expression_is_B_l21_21295

noncomputable def largest_expression : ℝ :=
  let A := Real.sqrt (Real.cbrt (8 * 7))
  let B := Real.sqrt (7 * Real.cbrt 8)
  let C := Real.sqrt (8 * Real.cbrt 7)
  let D := Real.cbrt (8 * Real.sqrt 7)
  let E := Real.cbrt (7 * Real.sqrt 8)
  if B > A && B > C && B > D && B > E then B else 0

theorem largest_expression_is_B :
  let B := Real.sqrt (7 * Real.cbrt 8)
  largest_expression = B :=
  by
  let A := Real.sqrt (Real.cbrt (8 * 7))
  let C := Real.sqrt (8 * Real.cbrt 7)
  let D := Real.cbrt (8 * Real.sqrt 7)
  let E := Real.cbrt (7 * Real.sqrt 8)
  sorry

end largest_expression_is_B_l21_21295


namespace range_of_F_l21_21869

-- Defining the piecewise function f_M
def f_M (M : Set ℝ) (x : ℝ) : ℝ :=
if x ∈ M then 1 else 0

-- Assume A and B are non-empty proper subsets of ℝ, and A ∩ B = ∅
variables (A B : Set ℝ) (hA : A.nonempty) (hB : B.nonempty) (hA_proper : A ≠ Set.univ) (hB_proper : B ≠ Set.univ) (h_disjoint : disjoint A B)

-- Defining the function F(x)
def F (x : ℝ) : ℝ :=
(f_M (A ∪ B) x + 1) / (f_M A x + f_M B x + 1)

theorem range_of_F : Set.range (F A B) = {1} :=
sorry

end range_of_F_l21_21869


namespace set_of_elements_of_a_l21_21384

open Set

def is_sequence_a (a : ℕ → ℕ) := 
  (a 1 = 1) ∧ ∀ n : ℕ, a (n + 1) = Nat.find (λ k, (∀ m, m ≤ n → a m ≠ k) ∧ (Nat.lcm (Finset.range (n+1) ).image a k > Nat.lcm (Finset.range n).image a))

theorem set_of_elements_of_a (a : ℕ → ℕ) (h : is_sequence_a a) : 
  { x : ℕ | ∃ n : ℕ, a n = x } = {1} ∪ { p ^ k | p Prime ∧ k ≥ 1 } :=
sorry

end set_of_elements_of_a_l21_21384


namespace find_negative_number_less_than_its_reciprocal_l21_21294

noncomputable def is_less_than_reciprocal_and_negative (x : ℝ) : Prop :=
  x < 1 / x ∧ x < 0

theorem find_negative_number_less_than_its_reciprocal :
  ∃ x ∈ ({-3, -1/2, -1, 1/3, 3} : set ℝ), is_less_than_reciprocal_and_negative x ∧
  ∀ y ∈ ({-3, -1/2, -1, 1/3, 3} : set ℝ), is_less_than_reciprocal_and_negative y → y = x :=
begin
  sorry
end

end find_negative_number_less_than_its_reciprocal_l21_21294


namespace trigonometric_identity_l21_21233

theorem trigonometric_identity :
  sqrt (1 + real.cos (real.pi * 100 / 180)) - sqrt (1 - real.cos (real.pi * 100 / 180)) = -2 * real.sin (real.pi * 5 / 180) :=
by
  sorry

end trigonometric_identity_l21_21233


namespace p_sq_plus_q_sq_l21_21599

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := 
by 
  sorry

end p_sq_plus_q_sq_l21_21599


namespace find_n_l21_21523

open Classical

theorem find_n (n : ℕ) (h : (8 * Nat.choose n 3) = 8 * (2 * Nat.choose n 1)) : n = 5 := by
  sorry

end find_n_l21_21523


namespace james_recovery_time_l21_21954

theorem james_recovery_time :
  let initial_healing_time := 4
  let skin_graft_healing_time := initial_healing_time + (initial_healing_time * 0.5)
  let total_recovery_time := initial_healing_time + skin_graft_healing_time
  total_recovery_time = 10 :=
by
  let initial_healing_time := 4
  let skin_graft_healing_time := initial_healing_time + (initial_healing_time * 0.5)
  let total_recovery_time := initial_healing_time + skin_graft_healing_time
  have h : total_recovery_time = 10 := sorry
  exact h

end james_recovery_time_l21_21954


namespace a_n_formula_b_n_formula_sum_c_n_l21_21428

-- Definitions
def a (n : ℕ) : ℕ := 2n - 1
def b (n : ℕ) : ℕ := 2^n
def c (n : ℕ) : ℚ := (2n - 1) / 2^n
def S (n : ℕ) : ℕ := 2 * b n - 2
def T (n : ℕ) : ℚ := ∑ i in Finset.range n, c (i + 1)

-- Conditions
axiom a1_plus_a2_plus_a3 : a 1 + a 2 + a 3 = 9
axiom a2_plus_a8 : a 2 + a 8 = 18

-- Proof goals
theorem a_n_formula : a n = 2n - 1 := by
  sorry

theorem b_n_formula : b n = 2^n := by
  sorry

theorem sum_c_n : T n = 3 - (2n + 3) / 2^n := by
  sorry

end a_n_formula_b_n_formula_sum_c_n_l21_21428


namespace number_of_integers_between_cubes_l21_21480

theorem number_of_integers_between_cubes :
  let a := 10.4
  let b := 10.5
  let lower_bound := a ^ 3
  let upper_bound := b ^ 3
  let start := Int.ceil lower_bound
  let end_ := Int.floor upper_bound
  end_ - start + 1 = 33 :=
by
  have h1 : lower_bound = 1124.864 := by sorry
  have h2 : upper_bound = 1157.625 := by sorry
  have h3 : start = 1125 := by sorry
  have h4 : end_ = 1157 := by sorry
  sorry

end number_of_integers_between_cubes_l21_21480


namespace cot_difference_triangle_l21_21144

theorem cot_difference_triangle (ABC : Triangle)
  (angle_condition : ∠AD BC = 60) :
  |cot ∠B - cot ∠C| = 5 / 2 :=
sorry

end cot_difference_triangle_l21_21144


namespace total_cost_one_pizza_and_three_burgers_l21_21105

def burger_cost : ℕ := 9
def pizza_cost : ℕ := burger_cost * 2
def total_cost : ℕ := pizza_cost + (burger_cost * 3)

theorem total_cost_one_pizza_and_three_burgers :
  total_cost = 45 :=
by
  rw [total_cost, pizza_cost, burger_cost]
  norm_num

end total_cost_one_pizza_and_three_burgers_l21_21105


namespace num_perfect_squares_in_range_l21_21494

theorem num_perfect_squares_in_range : 
  ∃ (k : ℕ), k = 12 ∧ ∀ n : ℕ, (100 < n^2 ∧ n^2 < 500 → 11 ≤ n ∧ n ≤ 22): sorry

end num_perfect_squares_in_range_l21_21494


namespace greatest_integer_sum_l21_21824

theorem greatest_integer_sum (perm : List ℕ) (h : perm ~ List.finRange 100)
  (hc : ∀ i, 1 ≤ i → i ≤ 91 → (List.sum (perm.slice i (i + 10))) ≥ A) :
  A = 505 := by
  sorry

end greatest_integer_sum_l21_21824


namespace exists_zero_in_2_3_l21_21449

def f (x : ℝ) : ℝ := x + Real.log x - 4

-- Given conditions
variable {x : ℝ}
variable (h1 : 0 < x) (h2 : x < 3)
variable (pos2 : f 2 < 0) (pos3 : f 3 > 0)

-- The theorem we want to prove
theorem exists_zero_in_2_3 : ∃ c : ℝ, (2 < c ∧ c < 3) ∧ f c = 0 :=
sorry

end exists_zero_in_2_3_l21_21449


namespace parallelogram_inscribed_area_perimeter_l21_21212

theorem parallelogram_inscribed_area_perimeter :
  ∀ (A B C D E F E' F' AF_ratio AD_ratio : ℝ)
    (h_square : true) 
    (h_reflection : true)
    (common_area : ℝ) (common_perimeter : ℝ),
    AF_ratio = 1 / 4 →
    AD_ratio = 1 →
    common_area / (common_perimeter^2) = 1 / 40 :=
by
  intros A B C D E F E' F' AF_ratio AD_ratio h_square h_reflection common_area common_perimeter,
  assume h_AF : AF_ratio = 1 / 4,
  assume h_AD : AD_ratio = 1,
  sorry

end parallelogram_inscribed_area_perimeter_l21_21212


namespace cost_price_correct_l21_21348

variable (SP : ℝ) (n : ℕ) (P : ℝ)

def total_cost_price (SP : ℝ) (P : ℝ) (n : ℕ) : ℝ := SP - P * n

def cost_price_per_meter (SP : ℝ) (P : ℝ) (n : ℕ) : ℝ :=
  (SP - P * n) / n

theorem cost_price_correct :
  SP = 660 → n = 66 → P = 5 → cost_price_per_meter SP P n = 5 := by
  sorry 

end cost_price_correct_l21_21348


namespace Anchuria_min_crooks_l21_21543

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l21_21543


namespace cot_diff_equal_l21_21151

variable (A B C D : Type)

-- Define the triangle and median.
variable [triangle ABC : Type] (median : Type)

-- Define the angle condition.
def angle_condition (ABC : triangle) (AD : median) : Prop :=
  ∠(AD, BC) = 60

-- Prove the cotangent difference
theorem cot_diff_equal
  (ABC : triangle)
  (AD : median)
  (h : angle_condition ABC AD) :
  abs (cot B - cot C) = (9 - 3 * sqrt 3) / 2 :=
by
  sorry -- Proof to be constructed

end cot_diff_equal_l21_21151


namespace sequence_formula_sum_inequality_l21_21066

noncomputable def a_n : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * a_n n + 2^(n+1)

def S_n (n : ℕ) : ℕ :=
(nat.rec_on n 0 (λ m acc, a_n (m + 1) + acc))

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) :
  a_n n = (n - 1 / 2) * 2^n :=
sorry

theorem sum_inequality (n : ℕ) : 
  (S_n n) / 2^n > 2 * n - 3 :=
sorry

end sequence_formula_sum_inequality_l21_21066


namespace range_of_m_l21_21441

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∀ m, m = 9 / 4 → (1 / x + 4 / y) ≥ m := 
by
  sorry

end range_of_m_l21_21441


namespace number_of_subsets_in_intersection_l21_21082

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | x > Real.exp 1 ∧ Int.floor x = x ∧ x > 0}

def B : Set ℝ := {x | x < -1 ∨ x > 7}

def complement_B : Set ℝ := U \ B

def intersection : Set ℝ := complement_B ∩ A

def num_subsets (s : Set ℝ) : ℕ := 2 ^ ((finite s.to_finset).to_finset.card)

theorem number_of_subsets_in_intersection :
  num_subsets intersection = 32 := sorry

end number_of_subsets_in_intersection_l21_21082


namespace no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l21_21807

theorem no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k :=
by
  sorry  

end no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l21_21807


namespace find_cot_difference_l21_21138

-- Define necessary elements for the problem
variable {A B C D : Type}
variable [EuclideanGeometry A]
variables (ABC : Triangle A B C)

-- Define the condition where median AD makes an angle of 60 degrees with BC
variable (ADmedian : median A B C D ∧ angle D A B = 60)

theorem find_cot_difference:
  |cot (angle B) - cot (angle C)| = 2 :=
sorry

end find_cot_difference_l21_21138


namespace not_guarantee_similarity_l21_21850

variables {A B C D E P : Type} [square A B C D] [midpoint E D C] [P_on_BC : points_on_side P B C]

-- Conditions
def angle_APB_eq_angle_EPC (h1 : ∠APB = ∠EPC) : Prop := sorry
def angle_APE_eq_90 (h2 : ∠APE = 90) : Prop := sorry
def P_midpoint_BC (h3 : midpoint P B C) : Prop := sorry
def ratio_BP_BC_23 (h4 : ratio BP BC = 2/3) : Prop := sorry

-- The proof statement
theorem not_guarantee_similarity (h3 : midpoint P B C) :
  ¬ (similar (triangle A B P) (triangle E C P)) :=
sorry

end not_guarantee_similarity_l21_21850


namespace pages_left_l21_21213

-- Define the conditions
def initial_books := 10
def pages_per_book := 100
def books_lost := 2

-- The total pages Phil had initially
def initial_pages := initial_books * pages_per_book

-- The number of books left after losing some during the move
def books_left := initial_books - books_lost

-- Prove the number of pages worth of books Phil has left
theorem pages_left : books_left * pages_per_book = 800 := by
  sorry

end pages_left_l21_21213


namespace parametric_curve_ellipse_foci_l21_21063

theorem parametric_curve_ellipse_foci :
  ∀ (θ : ℝ), (x = 4 * Real.cos θ) ∧ (y = 3 * Real.sin θ) →
  ∃ (f1 f2 : ℝ × ℝ), f1 = (√7, 0) ∧ f2 = (-√7, 0) :=
by
  intros θ h
  have hx : x = 4 * Real.cos θ := h.1
  have hy : y = 3 * Real.sin θ := h.2
  sorry

end parametric_curve_ellipse_foci_l21_21063


namespace find_general_term_and_Tn_l21_21107

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Define conditions
def arithmetic_sequence (d : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def geometric_mean (a1 a4 : ℝ) (a2: ℝ) : Prop :=
  a2^2 = a1 * a4

def bn_def (a_n : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), b_n n = a_n (n * (n + 1) / 2)

def Tn_def (T_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), T_n n = ∑ i in range n, (-1) ^ i * b_n (i + 1)

theorem find_general_term_and_Tn :
  (∃ a1 d a4 a2 : ℝ, arithmetic_sequence d ∧ geometric_mean a1 a4 a2 ∧ d = 2 ∧ a2 = a1 + d ∧ 
  (∀ n : ℕ, a_n n = a1 + (n - 1) * d) ∧
  (bn_def a_n) ∧ (Tn_def T_n b_n)) →
  (∀ n : ℕ, 
    (a_n n = 2 * n) ∧ 
    (Tn_def T_n b_n →
    (T_n n = if even n then (n * (n + 2)) / 2 else -((n + 1) ^ 2) / 2))) :=
sorry

end find_general_term_and_Tn_l21_21107


namespace distance_point_to_line_l21_21220

theorem distance_point_to_line 
  (a b c x0 y0 : ℝ) 
  (h_line : a*x0 + b*y0 + c = 0) :
  ∀ (x y : ℝ), (a*x + b*y + c = 0) →
  dist (x0, y0) (x, y) = abs (a*x0 + b*y0 + c) / real.sqrt (a^2 + b^2) :=
by
  sorry

end distance_point_to_line_l21_21220


namespace problem1_problem2_problem3_l21_21224

-- (1) Prove 1 - 2(x - y) + (x - y)^2 = (1 - x + y)^2
theorem problem1 (x y : ℝ) : 1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
sorry

-- (2) Prove 25(a - 1)^2 - 10(a - 1) + 1 = (5a - 6)^2
theorem problem2 (a : ℝ) : 25 * (a - 1)^2 - 10 * (a - 1) + 1 = (5 * a - 6)^2 :=
sorry

-- (3) Prove (y^2 - 4y)(y^2 - 4y + 8) + 16 = (y - 2)^4
theorem problem3 (y : ℝ) : (y^2 - 4 * y) * (y^2 - 4 * y + 8) + 16 = (y - 2)^4 :=
sorry

end problem1_problem2_problem3_l21_21224


namespace plane_region_position_l21_21665

theorem plane_region_position (x y : ℝ) (h₁ : x - 2*y + 6 > 0) : 
  ∃P1 P2 : ℝ × ℝ, 
  (P1 = (-6, 0) ∧ P2 = (0, 3)) ∧ (0, 0) ∈ { p : ℝ × ℝ | let (x, y) := p in x - 2*y + 6 > 0 } := 
sorry

end plane_region_position_l21_21665


namespace reflection_line_slope_l21_21659

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l21_21659


namespace cos_alpha_sub_beta_l21_21518

theorem cos_alpha_sub_beta (α β γ : ℝ) 
  (h₁ : sin α + sin β + sin γ = 0)
  (h₂ : cos α + cos β + cos γ = 0) :
  cos (α - β) = -1/2 :=
by sorry

end cos_alpha_sub_beta_l21_21518


namespace sqrt_x_squared_non_neg_l21_21709

theorem sqrt_x_squared_non_neg (x : ℝ) : 0 ≤ Real.sqrt (x * x) :=
begin
  sorry
end

end sqrt_x_squared_non_neg_l21_21709


namespace quadrilateral_iff_segments_lt_half_l21_21337

theorem quadrilateral_iff_segments_lt_half (a b c d : ℝ) (h₁ : a + b + c + d = 1) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ d) : 
    (a + b > d) ∧ (a + c > d) ∧ (a + b + c > d) ∧ (b + c > d) ↔ a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 :=
by
  sorry

end quadrilateral_iff_segments_lt_half_l21_21337


namespace polar_to_rectangular_l21_21739

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end polar_to_rectangular_l21_21739


namespace true_proposition_l21_21857

-- Definitions of propositions
def p := ∃ (x : ℝ), x - x + 1 ≥ 0
def q := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- Theorem statement
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l21_21857


namespace PQRS_cyclic_iff_BG_bisects_CBD_l21_21275

open EuclideanGeometry

axiom trapezoid_cyclic_condition (A B C D G P Q R S: Point) (ω: Circle):
  (AB_parallel_CD: are_parallel ℝ (line_segment A B) (line_segment C D)) →
  (ABCD_inscribed: inscribed A B C D ω) →
  (G_in_triangle_BCD: in_triangle G B C D) →
  (AG_meets_ω_at_P: meets_again ω (ray_through A G) P) →
  (BG_meets_ω_at_Q: meets_again ω (ray_through B G) Q) →
  (line_through_G_parallel_AB_intersects_BD_at_R: intersects (parallel_line G (line_segment A B)) (line_segment B D) R) →
  (line_through_G_parallel_AB_intersects_BC_at_S: intersects (parallel_line G (line_segment A B)) (line_segment B C) S) →
  (BG_bisects_angle_CBD: angle_bisector B G C) →
  cyclic_quadrilateral P Q R S

/-- Statement: Quadrilateral PQRS is cyclic if and only if BG bisects angle CBD. -/
theorem PQRS_cyclic_iff_BG_bisects_CBD (A B C D G P Q R S: Point) (ω: Circle):
  (AB_parallel_CD: are_parallel ℝ (line_segment A B) (line_segment C D)) →
  (ABCD_inscribed: inscribed A B C D ω) →
  (G_in_triangle_BCD: in_triangle G B C D) →
  (AG_meets_ω_at_P: meets_again ω (ray_through A G) P) →
  (BG_meets_ω_at_Q: meets_again ω (ray_through B G) Q) →
  (line_through_G_parallel_AB_intersects_BD_at_R: intersects (parallel_line G (line_segment A B)) (line_segment B D) R) →
  (line_through_G_parallel_AB_intersects_BC_at_S: intersects (parallel_line G (line_segment A B)) (line_segment B C) S) →
  (cyclic_quadrilateral P Q R S ↔ angle_bisector B G C) :=
by
  sorry

end PQRS_cyclic_iff_BG_bisects_CBD_l21_21275


namespace number_of_valid_subsets_l21_21084

def is_valid_subset (A : Set ℕ) : Prop :=
  {1, 2} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5}

theorem number_of_valid_subsets : 
  (finset.univ.filter is_valid_subset).card = 8 := 
by 
  sorry

end number_of_valid_subsets_l21_21084


namespace distance_to_y_axis_l21_21538

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end distance_to_y_axis_l21_21538


namespace seashells_total_l21_21226

def seashells_sam : ℕ := 18
def seashells_mary : ℕ := 47
def seashells_john : ℕ := 32
def seashells_emily : ℕ := 26

theorem seashells_total : seashells_sam + seashells_mary + seashells_john + seashells_emily = 123 := by
    sorry

end seashells_total_l21_21226


namespace unique_tags_proof_l21_21729

def unique_tags_count (letters: Finset Char) (digits: Finset Char) (len: ℕ) : ℕ :=
  let chars := letters ∪ digits
  let no_repeated := Multiset.prod (Finset.powersetLen len chars).to_multiset
  let repeated_2 := 10 * (Finset.choose 3 chars.card) * Factorial (len - 2)
  let repeated_3 := 10 * (Finset.choose 3 chars.card) * Factorial (len - 2)
  no_repeated.card + repeated_2 + repeated_3

theorem unique_tags_proof (letters: Finset Char) (digits: Finset Char) (len: ℕ) (N: ℕ) :
  letters = {'S', 'T', 'E', 'M'} →
  digits = {'2', '0', '2', '3'} →
  len = 5 →
  N = unique_tags_count letters digits len →
  N / 10 = 492 :=
by
  intros hletters hdigits hlen hN
  sorry

end unique_tags_proof_l21_21729


namespace arithmetic_sequence_find_Tn_l21_21038

variable {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ)

-- Condition 1
def positive_sequence (a : ℕ → ℕ) := ∀ n, a n > 0

-- Condition 2
def Sn_formula (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = (a n * (a n + 1)) / 2

-- Arithmetic sequence with common difference 1
theorem arithmetic_sequence (h1 : positive_sequence a) (h2 : Sn_formula S a) :
  ∀ n, n ≥ 1 → a n = n := sorry

-- Given b_n and sum T_n, find T_n
def b_n := λ n, (1 : ℚ) / ((a n * (a n + 1)) / 2)

def T_n := λ n, ∑ i in Finset.range n, b_n i

theorem find_Tn (h1 : positive_sequence a) (h2 : Sn_formula S a) :
  ∀ n, T_n n = 2 * n / (n + 1) := sorry

end arithmetic_sequence_find_Tn_l21_21038


namespace fraction_equality_l21_21917

theorem fraction_equality (a b : ℚ) (h₁ : a = 1/2) (h₂ : b = 2/3) : 
    (6 * a + 18 * b) / (12 * a + 6 * b) = 3 / 2 := by
  sorry

end fraction_equality_l21_21917


namespace minimum_crooks_l21_21551

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l21_21551


namespace reciprocal_of_sum_of_repeating_decimals_l21_21698

theorem reciprocal_of_sum_of_repeating_decimals :
  let x := 5 / 33
  let y := 1 / 3
  1 / (x + y) = 33 / 16 :=
by
  -- The following is the proof, but it will be skipped for this exercise.
  sorry

end reciprocal_of_sum_of_repeating_decimals_l21_21698


namespace area_of_right_triangle_ABC_l21_21608

theorem area_of_right_triangle_ABC 
  (right_triangle : ∀ {α β γ: Type}, α → β → γ → Prop)
  (xy_plane : Type)
  (A B C : xy_plane)
  (m1 m2 : line)
  (hypotenuse : ℝ)
  (hyp_eq : hypotenuse = 50)
  (med_A : m1 = (fun p : ℝ × ℝ => p.2 = p.1 + 5))
  (med_C : m2 = (fun p : ℝ × ℝ => p.2 = 3 * p.1 + 6))
  (right_angle_at_B : right_triangle A B C)
  (perp: ∀ {a b : Type}, α → β → Prop)
  (perp_angle : perp A B)
  (dist : ∀ {α β: Type}, α → β → ℝ)
  (h_dist : dist A C = 50) :
  area right_triangle B A C = 937.5 :=
sorry

end area_of_right_triangle_ABC_l21_21608


namespace ball_box_arrangement_l21_21907

-- Given n distinguishable balls and m distinguishable boxes,
-- prove that the number of ways to place the n balls into the m boxes is m^n.
-- Specifically for n = 6 and m = 3.

theorem ball_box_arrangement : (3^6 = 729) :=
by
  sorry

end ball_box_arrangement_l21_21907


namespace parabola_vertex_calc_l21_21336

noncomputable def vertex_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_vertex_calc 
  (a b c : ℝ) 
  (h_vertex : vertex_parabola a b c 2 = 5)
  (h_point : vertex_parabola a b c 1 = 8) : 
  a - b + c = 32 :=
sorry

end parabola_vertex_calc_l21_21336


namespace collinear_ABD_find_k_for_collinearity_l21_21187

noncomputable theory

-- Definitions used in the conditions
variables {E : Type*} [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E) (AB BC CD : E)

-- Conditions: 
def ab := e1 + e2
def bc := 2 • e1 + 8 • e2
def cd := 3 • (e1 - e2)

-- Proof Problem 1: Prove A, B, D are collinear
theorem collinear_ABD (e1_nonzero : e1 ≠ 0) (e2_nonzero : e2 ≠ 0) (not_collinear : ¬ collinear ℝ ({e1, e2} : set E)) :
  let AD := ab + bc + cd in
  AD = 6 • ab :=
begin
  sorry
end

-- Proof Problem 2: Determine k such that k e1 + e2 and e1 + k e2 are collinear
theorem find_k_for_collinearity (k : ℝ) (e1_nonzero : e1 ≠ 0) (e2_nonzero : e2 ≠ 0) : 
  collinear ℝ ({k • e1 + e2, e1 + k • e2} : set E) ↔ k = 1 ∨ k = -1 :=
begin
  sorry
end

end collinear_ABD_find_k_for_collinearity_l21_21187


namespace part1_part2_part3_l21_21593

-- Definitions from conditions
def S (n : ℕ) : ℕ := (3^(n + 1) - 3) / 2
def a (n : ℕ) : ℕ := 3^n
def b (n : ℕ) : ℕ := 2 * a n / (a n - 2)^2
def T (n : ℕ) : ℕ := ∑ i in finset.range n, b i

-- Proof Statements
theorem part1 (n : ℕ) : S (n - 1) - S (n - 2) = a n := sorry

theorem part2 (n : ℕ) : 
  (k : ℕ) → (1 = k) → n = (finset.min' (finset.image (λ n, (S (2 * n) + 15)/a n) (finset.range k)) _) := sorry

theorem part3 (n : ℕ) : T n < 13 / 2 := sorry

end part1_part2_part3_l21_21593


namespace find_b_value_l21_21250

theorem find_b_value (b : ℚ) (x : ℚ) (h1 : 3 * x + 9 = 0) (h2 : b * x + 15 = 5) : b = 10 / 3 :=
by
  sorry

end find_b_value_l21_21250


namespace magnitude_of_c_l21_21077

open Real

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b : ℝ × ℝ := (-2, 6)
noncomputable def angle_ca : ℝ := π / 3 -- 60 degrees in radians
noncomputable def dot_product_condition (c : ℝ × ℝ) : ℝ :=
  (c.1 * (vector_a.1 + vector_b.1) + c.2 * (vector_a.2 + vector_b.2))

theorem magnitude_of_c (c : ℝ × ℝ)
  (h_angle: real.angle (atan2 c.2 c.1 - atan2 vector_a.2 vector_a.1) = angle_ca)
  (h_dot: dot_product_condition c = -10) :
  sqrt (c.1 ^ 2 + c.2 ^ 2) = 2 * sqrt 10 := sorry

end magnitude_of_c_l21_21077


namespace sum_of_solutions_l21_21236

theorem sum_of_solutions (x : ℝ) : 
  (2 : ℝ)^(x^2 + 6 * x + 9) = (16 : ℝ)^(x + 3) → 
  ∑ y in {-3, 1}, y = -2 :=
by
  sorry

end sum_of_solutions_l21_21236


namespace tangent_circumcircle_BDE_l21_21588

variable {α : Type*} [EuclideanGeometry α]

noncomputable def isosceles_triangle (A B C : α) (h : AC = BC)
  : Prop := ... -- Define the properties of isosceles triangle

noncomputable def circumcircle (A B C : α)
  : Prop := ... -- Define the properties of a circumcircle

noncomputable def cyclic_quadrilateral (A B C D : α)
  : Prop := ... -- Define the properties of a cyclic quadrilateral

noncomputable def tangent (C D E: α)
  : Prop := ... -- Define the properties of tangent

theorem tangent_circumcircle_BDE {A B C D E : α}
  (h1 : isosceles_triangle A B C)
  (h2 : circumcircle A B C k)
  (h3 : D ∈ k ∧ (¬ (D = B ∨ D = C)))
  (h4 : E = CD ∩ AB)
  : tangent B C (circumcircle B D E) :=
sorry

end tangent_circumcircle_BDE_l21_21588


namespace length_of_third_side_l21_21920

-- Define the properties and setup for the problem
variables {a b : ℝ} (h1 : a = 4) (h2 : b = 8)

-- Define the condition for an isosceles triangle
def isosceles_triangle (x y z : ℝ) : Prop :=
  (x = y ∧ x ≠ z) ∨ (x = z ∧ x ≠ y) ∨ (y = z ∧ y ≠ x)

-- Define the condition for a valid triangle
def valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- State the theorem to be proved
theorem length_of_third_side (c : ℝ) (h : isosceles_triangle a b c ∧ valid_triangle a b c) : c = 8 :=
sorry

end length_of_third_side_l21_21920


namespace sum_of_geometric_series_is_correct_sum_of_geometric_series_as_decimal_is_correct_l21_21702

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

def sum_5_terms_of_series : ℚ :=
  sum_geometric_series (1/4) (1/4) 5

theorem sum_of_geometric_series_is_correct :
  sum_5_terms_of_series = 1023 / 3072 := 
sorry

theorem sum_of_geometric_series_as_decimal_is_correct :
  (sum_5_terms_of_series : ℚ).toReal ≈ 0.333 :=
  begin
    sorry,

  end

end sum_of_geometric_series_is_correct_sum_of_geometric_series_as_decimal_is_correct_l21_21702


namespace problem_statement_l21_21266

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Least common multiple of a set of numbers
noncomputable def LCM_set (s : List ℕ) : ℕ :=
s.foldl LCM 1

-- Calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem problem_statement :
  let T := LCM_set [2, 3, 5, 7, 11, 13]
  sum_of_digits T = 6 := by
  sorry

end problem_statement_l21_21266


namespace min_value_expression_l21_21435

theorem min_value_expression {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (c : ℝ), c = (2 * Real.sqrt 5) / 5 ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → 
  x^2 + 2*y^2 + z^2) / (x*y + 3*y*z) ≥ c :=
sorry

end min_value_expression_l21_21435


namespace distance_to_y_axis_l21_21537

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end distance_to_y_axis_l21_21537


namespace chef_earns_less_than_manager_l21_21715

noncomputable def manager_wage : ℝ := 6.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage + 0.2 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 2.60 :=
by
  sorry

end chef_earns_less_than_manager_l21_21715


namespace cot_difference_triangle_l21_21143

theorem cot_difference_triangle (ABC : Triangle)
  (angle_condition : ∠AD BC = 60) :
  |cot ∠B - cot ∠C| = 5 / 2 :=
sorry

end cot_difference_triangle_l21_21143


namespace sum_f_from_1_to_2020_l21_21061

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (2 * x - 1) + Real.cos (x - (Real.pi + 1) / 2)

theorem sum_f_from_1_to_2020 {
  let sum_f := Finset.sum (Finset.range 2020) (λ k, f (k + 1) / 2021)
  (sum_f = 2020) :=
by 
  sorry

end sum_f_from_1_to_2020_l21_21061


namespace number_of_blue_crayons_given_to_Becky_l21_21190

-- Definitions based on the conditions
def initial_green_crayons : ℕ := 5
def initial_blue_crayons : ℕ := 8
def given_out_green_crayons : ℕ := 3
def total_crayons_left : ℕ := 9

-- Statement of the problem and expected proof
theorem number_of_blue_crayons_given_to_Becky (initial_green_crayons initial_blue_crayons given_out_green_crayons total_crayons_left : ℕ) : 
  initial_green_crayons = 5 →
  initial_blue_crayons = 8 →
  given_out_green_crayons = 3 →
  total_crayons_left = 9 →
  ∃ num_blue_crayons_given_to_Becky, num_blue_crayons_given_to_Becky = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_blue_crayons_given_to_Becky_l21_21190


namespace determine_m_l21_21836

variable (A B : Set ℝ)
variable (m : ℝ)

theorem determine_m (hA : A = {-1, 3, m}) (hB : B = {3, 4}) (h_inter : B ∩ A = B) : m = 4 :=
sorry

end determine_m_l21_21836


namespace real_part_of_z_l21_21452

variable (i : ℂ) (z : ℂ)
variable (h_i : i = complex.I)
variable (h_z : z = (2 + i) / i)

theorem real_part_of_z : complex.re z = 1 :=
by
  sorry

end real_part_of_z_l21_21452


namespace find_picture_area_l21_21908

variable (x y : ℕ)
    (h1 : x > 1)
    (h2 : y > 1)
    (h3 : (3 * x + 2) * (y + 4) - x * y = 62)

theorem find_picture_area : x * y = 10 :=
by
  sorry

end find_picture_area_l21_21908


namespace Anchuria_min_crooks_l21_21541

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l21_21541


namespace quadratic_y_minimum_l21_21094

theorem quadratic_y_minimum (m : ℝ) (h : m^2 - m - 6 ≥ 0) :
  let x1 := 2 * m - 5 in
  let x2 := 2 * m + 2 in
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 := sorry

end quadratic_y_minimum_l21_21094


namespace coord_of_z_in_complex_plane_l21_21644

def z : ℂ := complex.I * (2 - complex.I)
def real_part_z (z : ℂ) : ℝ := z.re
def imag_part_z (z : ℂ) : ℝ := z.im

theorem coord_of_z_in_complex_plane : 
  (real_part_z z, imag_part_z z) = (1, 2) := 
by {
  -- Proof omitted
  sorry
}

end coord_of_z_in_complex_plane_l21_21644


namespace average_rate_l21_21685

variable (d_run : ℝ) (d_swim : ℝ) (r_run : ℝ) (r_swim : ℝ)
variable (t_run : ℝ := d_run / r_run) (t_swim : ℝ := d_swim / r_swim)

theorem average_rate (h_dist_run : d_run = 4) (h_dist_swim : d_swim = 4)
                      (h_run_rate : r_run = 10) (h_swim_rate : r_swim = 6) : 
                      ((d_run + d_swim) / (t_run + t_swim)) / 60 = 0.125 :=
by
  -- Properly using all the conditions given
  have := (4 + 4) / (4 / 10 + 4 / 6) / 60 = 0.125
  sorry

end average_rate_l21_21685


namespace count_valid_numbers_is_16_l21_21938

-- Define the digits
def digits : List ℕ := [4, 5, 5, 5, 0]

-- Define a predicate to check if a number does not start with 0
def valid_number (l : List ℕ) : Prop :=
  l.head ≠ 0

-- List all 5-digit permutations of the digits
def permuted_lists : List (List ℕ) := List.permutations digits

-- Filter valid permutations
def valid_numbers : List (List ℕ) :=
  permuted_lists.filter valid_number

-- Define the number of valid numbers
def count_valid_numbers : ℕ :=
  valid_numbers.length

theorem count_valid_numbers_is_16 : count_valid_numbers = 16 := by
  sorry

end count_valid_numbers_is_16_l21_21938


namespace area_of_hexadecagon_l21_21743

theorem area_of_hexadecagon (r : ℝ) : 
  regular_polygon_area 16 r = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) :=
sorry

end area_of_hexadecagon_l21_21743


namespace num_integers_between_10_4_cubed_and_10_5_cubed_l21_21473

noncomputable def cube (x : ℝ) : ℝ := x^3

theorem num_integers_between_10_4_cubed_and_10_5_cubed :
  let lower_bound := 10.4
  let upper_bound := 10.5
  let lower_cubed := cube lower_bound
  let upper_cubed := cube upper_bound
  let num_integers := (⌊upper_cubed⌋₊ - ⌈lower_cubed⌉₊ + 1 : ℕ)
  lower_cubed = 1124.864 ∧ upper_cubed = 1157.625 → 
  num_integers = 33 := 
by
  intro lower_bound upper_bound lower_cubed upper_cubed num_integers h
  have : lower_cubed = 1124.864 := h.1
  have : upper_cubed = 1157.625 := h.2
  rw [this, this]
  exact 33

end num_integers_between_10_4_cubed_and_10_5_cubed_l21_21473


namespace count_subsets_sum_eight_l21_21468

open Finset

theorem count_subsets_sum_eight :
  let M := (range 10).map (λ x => x + 1)
  in (M.powerset.filter (λ A => A.sum id = 8)).card = 6 := 
by
  sorry

end count_subsets_sum_eight_l21_21468


namespace area_quadrilateral_EFGH_l21_21623

def quadrilateralEFGH (E F G H : Type) [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H] :=
  right_angle E F G ∧ right_angle E H G ∧ dist E G = 5 ∧ dist F G = 1 ∧
  (∃ (EF EH HG : ℕ), EF ≠ EH ∧ EH ≠ HG ∧ HG ≠ EF) ∧
  (areaEFGH E F G H = √6 + 6)

theorem area_quadrilateral_EFGH :
  ∃ (EF GH : ℕ), (quadrilateralEFGH EF GH) ∧ (areaEFGH EF GH = √6 + 6) :=
sorry

end area_quadrilateral_EFGH_l21_21623


namespace binary_to_octal_of_101101110_l21_21002

def binaryToDecimal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 2 + b) 0 (Nat.digits 2 n)

def decimalToOctal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 10 + b) 0 (Nat.digits 8 n)

theorem binary_to_octal_of_101101110 :
  decimalToOctal (binaryToDecimal 0b101101110) = 556 :=
by sorry

end binary_to_octal_of_101101110_l21_21002


namespace cot_difference_abs_eq_sqrt3_l21_21134

theorem cot_difference_abs_eq_sqrt3 
  (A B C D P : Point) (x y : ℝ) (h1 : is_triangle A B C) 
  (h2 : is_median A D B C) (h3 : ∠(D, A, P) = 60)
  (BD_eq_CD : BD = x) (CD_eq_x : CD = x)
  (BP_eq_y : BP = y) (AP_eq_sqrt3 : AP = sqrt(3) * (x + y))
  (cot_B : cot B = -y / ((sqrt 3) * (x + y)))
  (cot_C : cot C = (2 * x + y) / (sqrt 3 * (x + y))) 
  (x_y_neq_zero : x + y ≠ 0) :
  abs (cot B - cot C) = sqrt 3
  := sorry

end cot_difference_abs_eq_sqrt3_l21_21134


namespace abs_sum_first_six_l21_21467

def sequence (n : ℕ) : Int :=
  -5 + 2 * (n - 1)

def abs_sum_up_to (n : ℕ) : Int :=
  (Finset.range n).sum (λ i, Int.natAbs (sequence (i + 1)))

theorem abs_sum_first_six :
  abs_sum_up_to 6 = 18 := 
by 
  sorry

end abs_sum_first_six_l21_21467


namespace inclination_angle_of_parametric_line_eqs_l21_21093

theorem inclination_angle_of_parametric_line_eqs :
  (∃ (α : ℝ), α ∈ set.Ico 0 180 ∧
    (∀ t : ℝ, ∃ k : ℝ, k = (2 - real.sqrt 3 * t - 2) / (1 + 3 * t - 1) ∧
      tan (real.to_radians α) = k) ∧
    α = 150) := by
  sorry

end inclination_angle_of_parametric_line_eqs_l21_21093


namespace find_m_and_quadratics_l21_21919

variable (x m a k : ℝ)
variable (m_positive : m > 0)

def y1 := a * (x - m) ^ 2 + 4
def generating_function := x ^ 2 + 4 * x + 14

def y2 := generating_function - y1

theorem find_m_and_quadratics (h1 : y2 = x ^ 2 - a * (x - m) ^ 2 + 4 * x + 10) 
                              (h2 : y2.eval m = 15) 
                              (h3 : ∃ k, (2, k) is_vertex_of_quadratic y2) : 
  m = 1 ∧ 
  y1 = (4 : ℝ) * (x - 1) ^ 2 + 4 ∧ 
  y2 = -3 * x ^ 2 + 12 * x + 6 := 
  sorry

end find_m_and_quadratics_l21_21919


namespace intersection_of_sets_l21_21839

theorem intersection_of_sets :
  let M := {-2, -1, 0, 1, 2}
  let N := {x | x < 0 ∨ x > 3}
  M ∩ N = {-2, -1} :=
by
  intro M N
  rw Set.inter_def
  rfl
  sorry

end intersection_of_sets_l21_21839


namespace abc_relationship_l21_21062

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * Real.exp (Real.abs x)

-- Define specific values for a, b, and c
def a : ℝ := f (Real.log 3 / Real.log 2)
def b : ℝ := f (- (Real.log 8 / Real.log 5))
def c : ℝ := f (- (Real.exp 1.001 * Real.log 2))

-- State the desired relationship between a, b, and c
theorem abc_relationship : c > a ∧ a > b :=
by 
  -- The proof is omitted here
  sorry

end abc_relationship_l21_21062


namespace sub_neg_four_l21_21369

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l21_21369


namespace fixed_point_of_log_l21_21256

theorem fixed_point_of_log (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (-1, 0) ∧ (P.2 = Real.log a (P.1 + 2)) :=
by
  sorry

end fixed_point_of_log_l21_21256


namespace line_PQ_is_parallel_to_x_axis_l21_21855

structure Point where
  x : ℤ
  y : ℤ

def is_parallel_to_x_axis (p q : Point) : Prop :=
  p.y = q.y

theorem line_PQ_is_parallel_to_x_axis :
  ∀ (P Q : Point),
    P.x = 6 → P.y = -6 →
    Q.x = -6 → Q.y = -6 →
    is_parallel_to_x_axis P Q :=
by
  intros P Q hPx hPy hQx hQy
  simp [is_parallel_to_x_axis, *]
  sorry

end line_PQ_is_parallel_to_x_axis_l21_21855


namespace kevin_card_total_l21_21585

theorem kevin_card_total (initial_cards found_cards : ℕ) (h1 : initial_cards = 7) (h2 : found_cards = 47) :
  initial_cards + found_cards = 54 :=
by
  -- initial setup based on conditions
  have h : 7 + 47 = 54 := rfl,
  rw [h1, h2, h]

end kevin_card_total_l21_21585


namespace interval_length_l21_21379

noncomputable def g (x : ℝ) : ℝ :=
  log 4 (log (1/4) (log 8 (log 2 (x ^ 2))))

def is_domain (x : ℝ) : Prop :=
  (-real.sqrt 8 < x ∧ x < -real.sqrt 2) ∨ 
  (real.sqrt 2 < x ∧ x < real.sqrt 8)

def length_L : ℝ := 
  2 * real.sqrt 2

def p : ℕ := 2
def q : ℕ := 1

theorem interval_length :
  length_L = 2 * real.sqrt 2 ∧ nat.gcd p q = 1 ∧ p + q = 3 :=
by {
  -- proof omitted
  sorry
}

end interval_length_l21_21379


namespace expected_value_coins_heads_l21_21332

theorem expected_value_coins_heads :
  let P := (1/2) : ℚ
  let nickel_value := 5 : ℚ
  let dime_value := 10 : ℚ
  let quarter_value := 25 : ℚ
  let nickel_EV := P * nickel_value
  let dime_EV := P * dime_value
  let first_quarter_EV := P * quarter_value
  let second_quarter_EV := P * quarter_value
  nickel_EV + dime_EV + first_quarter_EV + second_quarter_EV = 32.5 :=
by
  sorry

end expected_value_coins_heads_l21_21332


namespace problem1_range_decreasing_interval_problem2_range_of_a_l21_21887

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log a (-x^2 + a*x - 9)

theorem problem1_range_decreasing_interval {a : ℝ} (h0 : a > 0) (h1 : a ≠ 1) :
  (range (λ x : ℝ, log 10 (-x^2 + 10*x - 9)) = Iic (log 10 16)) ∧
  (decreasing_on (λ x : ℝ, log 10 (-x^2 + 10*x - 9)) (Icc 5 9)) :=
sorry

theorem problem2_range_of_a {a : ℝ} (h0 : a > 0) (h1 : a ≠ 1) :
  (∃ x : ℝ, increasing_on (λ x : ℝ, log a (-x^2 + a*x - 9)) Icc 1 9) → a > 6 :=
sorry

end problem1_range_decreasing_interval_problem2_range_of_a_l21_21887


namespace problem1_problem2_solution_x_1993_problem3_problem4_l21_21316

def MorseThueSeq : ℕ → ℕ
| 0     := 0
| (n+1) := if h : even (n + 1) then MorseThueSeq (n / 2) else 1 - MorseThueSeq (n / 2)

theorem problem1 (n : ℕ) :
  MorseThueSeq (2 * n) = MorseThueSeq n ∧ MorseThueSeq (2 * n + 1) = 1 - MorseThueSeq (2 * n) := by
  sorry

theorem problem2 (n : ℕ) (k : ℕ) (h : 2^k ≤ n ∧ n < 2^(k+1)) :
  MorseThueSeq n = 1 - MorseThueSeq (n - 2^k) := by
  sorry

theorem solution_x_1993 : MorseThueSeq 1993 = 0 := by
  sorry

theorem problem3 :
  ¬ (∃ p : ℕ, ∀ n : ℕ, MorseThueSeq n = MorseThueSeq (n + p)) := by
  sorry

def binary_sum_mod2_seq (n : ℕ) : ℕ :=
  n.binaryDigits.sum % 2

theorem problem4 (n : ℕ) :
  MorseThueSeq n = binary_sum_mod2_seq n := by
  sorry

end problem1_problem2_solution_x_1993_problem3_problem4_l21_21316


namespace total_expenditure_now_l21_21103

variable (A : ℝ) -- Original average budget per student
def original_students : ℝ := 100
def joined_students : ℝ := 32
def reduction_budget_per_student : ℝ := 10
def increase_total_expenditure : ℝ := 400

theorem total_expenditure_now : 
  ∃ (new_total_expenditure : ℝ), 
  new_total_expenditure = 5775 := 
by
  let E := original_students * A
  let new_total_expenditure := 132 * (A - reduction_budget_per_student)
  have h1 : 132 * (A - reduction_budget_per_student) = E + increase_total_expenditure, 
    from sorry
  have h2 : E = 100 * A,
    from sorry
  calc
    new_total_expenditure = 132 * (A - reduction_budget_per_student) : by sorry
                        ... = 100 * A + increase_total_expenditure : by sorry
                        ... = 5375 + 400 : by sorry
                        ... = 5775 : by sorry

end total_expenditure_now_l21_21103


namespace isosceles_triangle_perimeter_l21_21115

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l21_21115


namespace main_theorem_l21_21841

theorem main_theorem (α : ℝ) (h1 : sin α * cos α = 3/8) (h2 : π/4 < α ∧ α < π) :
  cos α - sin α = -real.sqrt (2)/2 :=
by
  sorry

end main_theorem_l21_21841


namespace ryan_learning_hours_l21_21813

theorem ryan_learning_hours :
  (∀ (hours_chinese hours_english : ℕ), hours_chinese = 5 ∧ hours_english = hours_chinese + 2 → hours_english = 7) :=
by
  intros hours_chinese hours_english
  intros h
  have h_chinese : hours_chinese = 5 := h.1
  have h_english : hours_english = hours_chinese + 2 := h.2
  rw h_chinese at h_english
  rw h_english
  norm_num
  sorry

end ryan_learning_hours_l21_21813


namespace find_a_l21_21971

-- Definitions of the conditions
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

-- The proof goal
theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 := 
by 
  sorry

end find_a_l21_21971


namespace numValidDistributions_l21_21341

-- Define the 3x3 grid and the condition
def isValidDistribution (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ i j, (i < 2 → |grid i j - grid (i+1) j| ≤ 3) ∧ (j < 2 → |grid i j - grid i (j+1)| ≤ 3))

-- The formal statement of the problem
theorem numValidDistributions : 
  ∃ (configurations : Finset (Matrix (Fin 3) (Fin 3) ℕ)),
  (∀ grid ∈ configurations, isValidDistribution grid) ∧ 
  configurations.card = 32 :=
sorry

end numValidDistributions_l21_21341


namespace minimum_crooks_l21_21554

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l21_21554


namespace num_real_solutions_abs_eq_l21_21906

theorem num_real_solutions_abs_eq :
  {x : ℝ | abs (x - 2) = abs (x - 5) + abs (x - 8)}.card = 2 :=
sorry

end num_real_solutions_abs_eq_l21_21906


namespace triangle_side_a_l21_21152

noncomputable def calculate_a (A B : ℝ) (b : ℝ) : ℝ := 
  let sin_B := sqrt (1 - (cos B)^2)
  let sin_A := sqrt 3 / 2
  (b * sin_A / sin_B)

theorem triangle_side_a {A B : ℝ} (b : ℝ) (cos_B_eq : cos B = (2 * sqrt 7) / 7) :
  A = π / 3 ∧ b = 3 ∧ cos B = (2 * sqrt 7) / 7 → calculate_a A B b = (3 * sqrt 7) / 2 :=
begin
  intros h, 
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  sorry
end

end triangle_side_a_l21_21152


namespace agate_precious_stones_l21_21204

theorem agate_precious_stones (A : ℕ) :
  let O := A + 5 in
  let D := A + 16 in
  A + O + D = 111 -> A = 30 :=
by
  sorry

end agate_precious_stones_l21_21204


namespace isosceles_right_triangle_quotient_l21_21358

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
sorry

end isosceles_right_triangle_quotient_l21_21358


namespace intersection_of_sets_l21_21838

theorem intersection_of_sets :
  let M := {-2, -1, 0, 1, 2}
  let N := {x | x < 0 ∨ x > 3}
  M ∩ N = {-2, -1} :=
by
  intro M N
  rw Set.inter_def
  rfl
  sorry

end intersection_of_sets_l21_21838


namespace annual_interest_rate_l21_21757

theorem annual_interest_rate (r : ℝ) :
  (6000 * r + 4000 * 0.09 = 840) → r = 0.08 :=
by sorry

end annual_interest_rate_l21_21757


namespace triangle_AEF_area_l21_21934

-- Definitions of conditions
variables {A B C D E F : Type}
variables [Field F]

structure Triangle (P Q R : Type) := (area : F)

axiom midpoint_of_segment {A B C : Type} : Prop
axiom area_of_triangle_ABC_is_120 {A B C : Type} : Triangle A B C
axiom D_is_midpoint_of_AB {A B D : Type} : Prop
axiom E_is_midpoint_of_DB {D B E : Type} : Prop
axiom F_is_midpoint_of_BC {B C F : Type} : Prop

-- The proof statement
theorem triangle_AEF_area :
  area_of_triangle_ABC_is_120 =
  45 :=
sorry

end triangle_AEF_area_l21_21934


namespace probability_failed_both_tests_eq_l21_21315

variable (total_students pass_test1 pass_test2 pass_both : ℕ)

def students_failed_both_tests (total pass1 pass2 both : ℕ) : ℕ :=
  total - (pass1 + pass2 - both)

theorem probability_failed_both_tests_eq 
  (h_total : total_students = 100)
  (h_pass1 : pass_test1 = 60)
  (h_pass2 : pass_test2 = 40)
  (h_pass_both : pass_both = 20) :
  students_failed_both_tests total_students pass_test1 pass_test2 pass_both / (total_students : ℚ) = 0.2 :=
by
  sorry

end probability_failed_both_tests_eq_l21_21315


namespace area_OAB_l21_21432

noncomputable def a := sqrt 5
def c := 1
noncomputable def b := sqrt (a^2 - c^2)
def ellipse_eq (x y : ℝ) := (x^2 / 5) + (y^2 / 4) = 1
def line_eq (x y : ℝ) := y = 2 * x - 2

theorem area_OAB :
  ∃ (A B : ℝ × ℝ),
  ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧
  ∃ (O : ℝ × ℝ), O = (0, 0) ∧
  let AB_length := real.sqrt ((1 + 4) * (5/3)^2) in
  let distance_O_line := 2 * sqrt 5 / 5 in
  (1 / 2) * AB_length * distance_O_line = 5 / 3 := sorry

end area_OAB_l21_21432


namespace abs_T_n_lt_4n_l21_21040

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 * n

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := (3 * n * (n + 1)) / 2

-- Define T_n based on the sequence
def T (n : ℕ) : ℕ := ∑ k in finset.range n, if (S k) % 2 = 0 then a k else -a k

-- State the theorem to be proven
theorem abs_T_n_lt_4n (n : ℕ) (hn : n ≥ 3) : |T n| < 4 * n := sorry

end abs_T_n_lt_4n_l21_21040


namespace part_a_l21_21586

open Complex

theorem part_a (z : ℂ) (hz : abs z = 1) :
  (abs (z + 1) - Real.sqrt 2) * (abs (z - 1) - Real.sqrt 2) ≤ 0 :=
by
  -- Proof will go here
  sorry

end part_a_l21_21586


namespace find_angle_C_find_side_b_l21_21924

noncomputable def triangle_side_opposite_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a ∧ A + B + C = π ∧ a = 2 * c * Real.sin B + c * Real.sin A + (2 * a + b) * Real.sin A = 2 * c * Real.sin C

axiom cos_A (A : ℝ) : Prop :=
A ∈ (0, π) ∧ Real.cos A = 4 / 5

theorem find_angle_C (A B C a b c : ℝ) (h₁ : triangle_side_opposite_angles A B C a b c)
    (h2 : c = 5)
    (h3: b * (2 * Real.sin B + Real.sin A) + (2 * a + b) * Real.sin A = 2 * c * Real.sin C) :
    C = 2 * π / 3 :=
sorry

theorem find_side_b (A B C a b c : ℝ) (h₁ : triangle_side_opposite_angles A B C a b c)
    (h2 : c = 5)
    (h3: b * (2 * Real.sin B + Real.sin A) + (2 * a + b) * Real.sin A = 2 * c * Real.sin C)
    (h4: Real.cos A = 4 / 5) :
    b = 4 - Real.sqrt 3 :=
sorry

end find_angle_C_find_side_b_l21_21924


namespace Nair_wins_game_l21_21615

theorem Nair_wins_game :
  ∃ (strategy : ℕ → ℕ), (∀ (n m : ℕ), 1 ≤ n ∧ n ≤ 3 ∧ 1 ≤ m ∧ m ≤ 3 → strategy n ∈ {1, 2, 3}) ∧ 
  (∀ (turn : ℕ), (∃ (k : ℕ), turn = 4 * k + 3) → 
    (∃ (moves : ℕ × ℕ), moves.1 ∈ {1, 2, 3} ∧ moves.2 = 4 - moves.1 ∧ 
    (∃ (next_turn : ℕ), next_turn = turn + moves.1 + moves.2 ∧ next_turn = 4 * (k + 1) + 3))) :=
sorry

end Nair_wins_game_l21_21615


namespace monograms_in_alphabetical_order_l21_21193

def count_monograms : ℕ :=
  let alphabet_size : ℕ := 26  -- Number of letters in the alphabet 'A' to 'Z'
  let & := 1                   -- Including the special '&' character
  let total_size : ℕ := alphabet_size + &  -- Total set size
  (total_size - &) * (total_size - & - 1) / 2  -- Binomial coefficient C(26, 2)

theorem monograms_in_alphabetical_order : count_monograms = 325 :=
by
  sorry

end monograms_in_alphabetical_order_l21_21193


namespace even_and_increasing_l21_21708

open Set

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- The function in question
def f (x : ℝ) := x^2

-- The interval [0, ∞)
def I := Icc 0 (⊤ : ℝ) -- or Set.Ici 0

-- Statement to be proved
theorem even_and_increasing :
  is_even_function f ∧ is_increasing_on f I :=
by
  sorry

end even_and_increasing_l21_21708


namespace min_value_ge_54_l21_21601

open Real

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) : ℝ :=
2 * x + 3 * y + 6 * z

theorem min_value_ge_54 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  min_value x y z h1 h2 h3 h4 ≥ 54 :=
sorry

end min_value_ge_54_l21_21601


namespace find_honda_cars_l21_21264

variable (H : ℕ)

-- Given conditions
axiom total_car_population : 900 = total_cars
axiom honda_car_red_ratio  : 0.9 * H = red_honda_cars
axiom red_car_ratio        : 0.6 * total_cars = 540
axiom non_honda_red_ratio  : 0.225 * (total_cars - H) = red_non_honda_cars
axiom total_red_car_sum    : red_honda_cars + red_non_honda_cars = 540

-- The statement to prove
theorem find_honda_cars : H = 500 :=
by
  sorry

end find_honda_cars_l21_21264


namespace find_phone_number_l21_21208

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10 in s = s.reverse

def is_consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

def has_three_consecutive_ones (n : ℕ) : Prop :=
  let s := n.digits 10 in s.contains [1, 1, 1]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

theorem find_phone_number : 
  ∃ n,
    n.digits 10 = [5,6,7,1,1,1,7] ∧  -- The number is 7111765
    n.digits 10.length = 7 ∧          -- The phone number has 7 digits
    is_consecutive 5 6 7 ∧            -- The last three digits are consecutive (765)
    is_palindrome (n / 10000) ∧       -- The first five digits form a palindrome (71117)
    (n / 1000) % 10 ^ 3 % 9 = 0 ∧     -- The three-digit number formed by the first three digits is divisible by 9 (711 is divisible by 9)
    has_three_consecutive_ones n ∧    -- The phone number contains exactly three consecutive ones
    (is_prime (n.digits 10.drop 3.take 2).to_nat xor 
     is_prime (n.digits 10.drop 5.take 2).to_nat) := -- Only one of the two-digit numbers obtained (from the groups) is prime (76 or 65)
sorry

end find_phone_number_l21_21208


namespace part1_union_part2_condition_l21_21184

noncomputable def domain_A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
noncomputable def range_B (x_lower x_upper : ℝ) : Set ℝ := { y | ∃ x, x ∈ Ioo (x_lower) (x_upper) ∧ y = 3 / (x + 1) }

theorem part1_union (m : ℝ) (hm : m = 1) :
  domain_A ∪ range_B 0 m = Set.Ioo 1 4 :=
sorry

theorem part2_condition (h : ∀ x, x ∈ domain_A → x ∈ range_B 0 1) :
  ∃ m, 0 < m ∧ m ≤ 2 :=
sorry

end part1_union_part2_condition_l21_21184


namespace minimum_crooks_l21_21555

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l21_21555


namespace value_of_f_prime_and_f_l21_21524

def f (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem value_of_f_prime_and_f :
  (derivative f 1) + derivative (fun (_ : ℝ) => f 1) 1 = 5 :=
by
  sorry

end value_of_f_prime_and_f_l21_21524


namespace investment_plans_l21_21325

theorem investment_plans : ∃ plans : ℕ, 
  (∀ (projects cities : ℕ) (c : cities ≥ projects ∧ projects = 4 ∧ cities = 4), 
    plans = 204) :=
begin
  sorry
end

end investment_plans_l21_21325


namespace green_beans_to_onions_ratio_l21_21611

def cut_conditions
  (potatoes : ℕ)
  (carrots : ℕ)
  (onions : ℕ)
  (green_beans : ℕ) : Prop :=
  carrots = 6 * potatoes ∧ onions = 2 * carrots ∧ potatoes = 2 ∧ green_beans = 8

theorem green_beans_to_onions_ratio (potatoes carrots onions green_beans : ℕ) :
  cut_conditions potatoes carrots onions green_beans →
  green_beans / gcd green_beans onions = 1 ∧ onions / gcd green_beans onions = 3 :=
by
  sorry

end green_beans_to_onions_ratio_l21_21611


namespace count_perfect_squares_between_100_and_500_l21_21496

def smallest_a (x : ℕ) : ℕ :=
  Nat.find (λ a, a^2 > x)

def largest_b (x : ℕ) : ℕ :=
  Nat.find (λ b, b^2 > x) - 1

theorem count_perfect_squares_between_100_and_500 :
  let a := smallest_a 100
  let b := largest_b 500
  b - a + 1 = 12 :=
by
  -- Definitions based on conditions
  let a := smallest_a 100
  have ha : a = 11 := 
    -- the proof follows here
    sorry
  let b := largest_b 500
  have hb : b = 22 := 
    -- the proof follows here
    sorry
  calc
    b - a + 1 = 22 - 11 + 1 : by rw [ha, hb]
           ... = 12          : by norm_num

end count_perfect_squares_between_100_and_500_l21_21496


namespace max_S_at_n_four_l21_21068

-- Define the sequence sum S_n
def S (n : ℕ) : ℤ := -(n^2 : ℤ) + (8 * n : ℤ)

-- Prove that S_n attains its maximum value at n = 4
theorem max_S_at_n_four : ∀ n : ℕ, S n ≤ S 4 :=
by
  sorry

end max_S_at_n_four_l21_21068


namespace num_perfect_squares_in_range_l21_21492

theorem num_perfect_squares_in_range : 
  ∃ (k : ℕ), k = 12 ∧ ∀ n : ℕ, (100 < n^2 ∧ n^2 < 500 → 11 ≤ n ∧ n ≤ 22): sorry

end num_perfect_squares_in_range_l21_21492


namespace dispatch_plans_count_l21_21326

-- Let’s first define the given conditions
noncomputable def total_vehicles : ℕ := 7
noncomputable def dispatch_count : ℕ := 4
noncomputable def required_vehicles : List ℕ := [0, 1] -- assuming A and B are indexed as 0 and 1

-- Define the main theorem as a Lean statement
theorem dispatch_plans_count :
  (total_vehicles = 7) ∧
  (dispatch_count = 4) ∧
  (A_and_B_participates : ∀ (a b : ℕ), a = 0 ∧ b = 1) ∧
  (A_before_B : ∀ (a b : ℕ), a = 0 ∧ b = 1 → a < b) →
  ∃ (n : ℕ), n = 120 :=
by sorry

end dispatch_plans_count_l21_21326


namespace exists_root_in_interval_l21_21456

-- Assume the function g is continuous on ℝ
variable (g : ℝ → ℝ)
variable (hg : Continuous g)

-- Define the function f as per the problem statement
def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) * g(x) + 3*x - 4

-- State the theorem: there exists a root of f in the interval (1,2)
theorem exists_root_in_interval :
  ∃ (x : ℝ), 1 < x ∧ x < 2 ∧ f g x = 0 :=
by
  sorry

end exists_root_in_interval_l21_21456


namespace simultaneous_arrival_l21_21619

-- Define the point structure on the circle
structure Point :=
  (x y : ℝ)

-- Define distances between points (this will be simplified for the circular lake)
def distance (A B : Point) : ℝ := real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Define the proof statement
theorem simultaneous_arrival
  (K L P Q X : Point)
  (u v : ℝ)
  (h_collide : distance K X / u = distance L X / v) :
  distance K Q / u = distance L P / v :=
begin
  -- Placeholder for the proof
  sorry
end

end simultaneous_arrival_l21_21619


namespace J_eval_l21_21411

def J (a b c : ℝ) : ℝ :=
  a / b + b / c + c / a

theorem J_eval : J (-3) 15 (-5) = - 23 / 15 :=
by 
  sorry

end J_eval_l21_21411


namespace swim_distance_downstream_l21_21735

theorem swim_distance_downstream 
  (V_m V_s : ℕ) 
  (t d : ℕ) 
  (h1 : V_m = 9) 
  (h2 : t = 3) 
  (h3 : 3 * (V_m - V_s) = 18) : 
  t * (V_m + V_s) = 36 := 
by 
  sorry

end swim_distance_downstream_l21_21735


namespace proposition_p_converse_q_correct_conclusion_l21_21858

theorem proposition_p (m : ℝ) (hm : m > 0) : ∃ x : ℝ, x^2 + x - m = 0 := sorry

theorem converse_q (m : ℝ) : (∃ x : ℝ, x^2 + x - m = 0) → (m > 0) ≠ True := sorry

theorem correct_conclusion : 
  (proposition_p m hm) ∧ (converse_q m → False) := sorry

end proposition_p_converse_q_correct_conclusion_l21_21858


namespace perfect_squares_count_l21_21506

theorem perfect_squares_count : (finset.filter (λ n, n * n ≥ 100 ∧ n * n ≤ 500) (finset.range 23)).card = 13 :=
by
  sorry

end perfect_squares_count_l21_21506


namespace segments_on_line_l21_21032

theorem segments_on_line (segments : Fin 50 → Set ℝ) :
  (∃ (pts : Fin 50 → ℝ), ∃ (P : Fin 8 → Fin 50), ∃ (pt : ℝ), 
    (∀ i : Fin 8, pt ∈ segments (P i)) ∧ 
    (∀ i j : Fin 8, i ≠ j → P i ≠ P j)) ∨
  (∃ (P : Fin 8 → Fin 50), ∀ i j : Fin 8, i ≠ j → disjoint (segments (P i)) (segments (P j))) :=
sorry

end segments_on_line_l21_21032


namespace mart_income_percentage_of_juan_l21_21612

variable (M T J : Real)

-- Conditions
def condition1 : Prop := M = 1.30 * T
def condition2 : Prop := T = 0.60 * J

-- Theorem to prove
theorem mart_income_percentage_of_juan (h1 : condition1 M T J) (h2 : condition2 T J) : M = 0.78 * J := by
  -- For the purpose of the example, we mark the proof as sorry
  sorry

end mart_income_percentage_of_juan_l21_21612


namespace sum_of_first_10_terms_of_b_2n_minus_1_l21_21055

variable (b : ℕ → ℕ) (n q : ℕ)

def geom_seq (b : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b 1 * q^n

def b1_1 (b : ℕ → ℕ) : Prop := b 1 = 1
def common_ratio_2 (q : ℕ) : Prop := q = 2

theorem sum_of_first_10_terms_of_b_2n_minus_1 
  (b : ℕ → ℕ) (q : ℕ) 
  (h_seq : geom_seq b q) 
  (h_b1 : b1_1 b) 
  (h_q : common_ratio_2 q) : 
  (∑ i in Finset.range 10, b (2 * i + 1)) = (1/3) * (4^10 - 1) :=
sorry

end sum_of_first_10_terms_of_b_2n_minus_1_l21_21055


namespace triangle_third_side_l21_21450

theorem triangle_third_side (a b c : ℕ) (ha : a = 4) (hb : b = 10) (hc : c = 12) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  -- Substitute the values of a, b, and c
  rw [ha, hb, hc],
  -- Assert and prove the triangle inequalities
  exact ⟨by decide, by decide, by decide⟩

end triangle_third_side_l21_21450


namespace vertical_asymptote_unique_c_l21_21009

-- Variables
variable {c : ℝ}

-- Function definition f
def f (x : ℝ) := (x^2 - x + c) / (x^2 + x - 18)

-- Theorem statement
theorem vertical_asymptote_unique_c : (∀ x, x^2 + x - 18 = (x - 3) * (x + 6)) →
  (∃ c,  (∀ x, f(x) = (x^2 - x + c) / ((x - 3) * (x + 6))) ∧
          (x = 3 ∨ x = -6) ∧ (x ≠ 3 ∨ x ≠ -6) ↔ (c = -6 ∨ c = -42)) := 
  by
    sorry

end vertical_asymptote_unique_c_l21_21009


namespace average_gas_mileage_round_trip_l21_21344

/-
A student drives 150 miles to university in a sedan that averages 25 miles per gallon.
The same student drives 150 miles back home in a minivan that averages 15 miles per gallon.
Calculate the average gas mileage for the entire round trip.
-/
theorem average_gas_mileage_round_trip (d1 d2 m1 m2 : ℝ) (h1 : d1 = 150) (h2 : m1 = 25) 
  (h3 : d2 = 150) (h4 : m2 = 15) : 
  (2 * d1) / ((d1/m1) + (d2/m2)) = 18.75 := by
  sorry

end average_gas_mileage_round_trip_l21_21344


namespace proof_problem_l21_21186

-- Define the universal set U
def U : set ℤ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the set A based on real solutions
def A : set ℝ := {x | (x - 1) * (x - 2) = 0}

-- Define the set B based on natural number solutions
def B : set ℕ := {x | (4 - x) / 2 > 1}

-- The complement of A in U in Lean
def complement_of_A_in_U : set ℤ := {x | x ∈ U ∧ x ∉ (A ∩ U)}

-- The union of A and B
def union_of_A_and_B : set ℝ := {x | x ∈ A ∨ x ∈ B}

-- The intersection of A and B
def intersection_of_A_and_B : set ℝ := {x | x ∈ A ∧ x ∈ B}

-- The proof problem
theorem proof_problem :
  (complement_of_A_in_U = {-1, 0, 3, 4, 5}) ∧
  (union_of_A_and_B = {1, 2}) ∧
  (intersection_of_A_and_B = {1}) :=
by
  sorry -- Proof omitted

end proof_problem_l21_21186


namespace log_0_319_approx_l21_21522

noncomputable def log_approx (x : ℝ) : ℝ :=
  if x = 0.317 then 0.33320
  else if x = 0.318 then 0.3364
  else 0 -- This is only a mock for the structure

theorem log_0_319_approx:
  log_approx 0.318 = 0.3364 →
  log_approx 0.317 = 0.33320 →
  log_approx 0.319 ≈ 0.3396 :=
by
  intros h1 h2
  sorry

end log_0_319_approx_l21_21522


namespace find_value_of_a_lambda_range_decreasing_max_value_lambda_l21_21457

theorem find_value_of_a (f : ℝ → ℝ) (h1 : ∀ x, f x = 3^x) (h2 : f (a + 2) = 27) :
  a = 1 :=
sorry

theorem lambda_range_decreasing (λ : ℝ) (g : ℝ → ℝ)
(h1 : ∀ x, g x = λ * 2^x - 4^x) (h2 : ∀ x, 0 ≤ x ∧ x ≤ 2) :
  ∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x2 - g x1 < 0 ↔ λ < 15 :=
sorry

theorem max_value_lambda (λ : ℝ) (g : ℝ → ℝ)
(h1 : ∀ x, g x = λ * 2^x - 4^x) (h2 : ∀ x, 0 ≤ x ∧ x ≤ 2)
(h3 : ∃ x ∈ Icc 0 2, g x = 1) :
  λ = 2 :=
sorry

end find_value_of_a_lambda_range_decreasing_max_value_lambda_l21_21457


namespace greatest_whole_number_with_odd_factors_less_than_150_l21_21202

theorem greatest_whole_number_with_odd_factors_less_than_150 :
  ∃ (n : ℕ), (∀ (m : ℕ), m < 150 ∧ odd_factors m → m ≤ n) ∧ n = 144 :=
by
  sorry

def odd_factors (k : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = k

end greatest_whole_number_with_odd_factors_less_than_150_l21_21202


namespace not_prime_by_changing_one_digit_l21_21572

theorem not_prime_by_changing_one_digit (n : ℕ) : 
  ¬ (∃ m : ℕ, (∃ d : ℕ, d < 10 ∧ ∃ i : ℕ, i < (nat.digits 10 n).length ∧ m = list.to_nat (list.update_nth (nat.digits 10 n) i d)) ∧ nat.prime m) := 
sorry

end not_prime_by_changing_one_digit_l21_21572


namespace ellipse_tangency_construction_l21_21796

theorem ellipse_tangency_construction
  (a : ℝ)
  (e1 e2 : ℝ → Prop)  -- Representing the parallel lines as propositions
  (F1 F2 : ℝ × ℝ)  -- Foci represented as points in the plane
  (d : ℝ)  -- Distance between the parallel lines
  (angle_condition : ℝ)
  (conditions : 2 * a > d ∧ angle_condition = 1 / 3) : 
  ∃ O : ℝ × ℝ,  -- Midpoint O
    ∃ (T1 T1' T2 T2' : ℝ × ℝ),  -- Points of tangency
      (∃ E1 E2 : ℝ, e1 E1 ∧ e2 E2) ∧  -- Intersection points on the lines
      (F1.1 * (T1.1 - F1.1) + F1.2 * (T1.2 - F1.2)) / 
      (F2.1 * (T2.1 - F2.1) + F2.2 * (T2.2 - F2.2)) = 1 / 3 :=
sorry

end ellipse_tangency_construction_l21_21796


namespace lemonade_water_l21_21610

theorem lemonade_water (L S W : ℝ) (h1 : S = 1.5 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 18 :=
by
  sorry

end lemonade_water_l21_21610


namespace find_minimized_angle_l21_21185

-- Definitions based on the conditions given.
def base_side_length : ℝ := 2017
def lateral_edge_length : ℝ := 2000

-- Definition of the half-diagonal of the base.
def half_diagonal : ℝ := base_side_length / Real.sqrt 2

-- Definition of the angle α using the tangent function.
def tan_alpha : ℝ := half_diagonal / lateral_edge_length

-- Definition to calculate the angle α using arctan function.
def alpha_angle : ℝ := Real.arctan tan_alpha

-- Defining the correct answer based on the given options.
def correct_option := 40

--The statement to prove that the correct option is the angle that minimizes the absolute difference.
theorem find_minimized_angle : abs (alpha_angle.to_degrees - 40) < abs (alpha_angle.to_degrees - 30) ∧
                               abs (alpha_angle.to_degrees - 40) < abs (alpha_angle.to_degrees - 50) ∧
                               abs (alpha_angle.to_degrees - 40) < abs (alpha_angle.to_degrees - 60) :=
by sorry

end find_minimized_angle_l21_21185


namespace cartesian_eqn_of_circle_arc_length_ratio_of_circle_divided_by_line_l21_21539

-- Define the polar equation of circle C and parametric equation of line l
def circle_polar_eqn (rho θ : ℝ) := rho = 6 * Real.cos θ

def line_param_eqn (t : ℝ) : ℝ × ℝ := (3 + 1/2 * t, -3 + (Real.sqrt 3) / 2 * t)

-- Prove that the Cartesian coordinate equation of circle C is (x - 3)² + y² = 9
theorem cartesian_eqn_of_circle :
  ∀ (x y : ℝ), (∃ ρ θ, (ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x) ∧ circle_polar_eqn ρ θ) →
  (x - 3) ^ 2 + y ^ 2 = 9 :=
by
  sorry

-- Prove that the ratio of the lengths of the two arcs into which line l divides circle C is 1:2
theorem arc_length_ratio_of_circle_divided_by_line
  (t : ℝ) :
  let (x, y) := line_param_eqn t in 
  ∃ angle : ℝ, 
  (∃ ρ, circle_polar_eqn ρ angle ∧ ∃ d, d = abs ((Real.sqrt 3 * x - y - 3 * Real.sqrt 3 - 3) / (Real.sqrt 1 + 3))) →
  angle = 120 → 1:2 :=
by
  sorry

end cartesian_eqn_of_circle_arc_length_ratio_of_circle_divided_by_line_l21_21539


namespace solve_problem_l21_21894

open Real

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def problem_statement
  (a b : ℝ × ℝ)
  (ha : magnitude a = 2)
  (hb : magnitude b = 1)
  (angle_ab : ∀ (θ : ℝ), cos θ = 1/2 ∧ θ = π / 3) : Prop :=
  magnitude (a.1 - 2 * b.1, a.2 - 2 * b.2) = 2

theorem solve_problem : ∃ (a b : ℝ × ℝ), 
  problem_statement a b sorry sorry sorry := sorry

end solve_problem_l21_21894


namespace total_flour_used_l21_21297

theorem total_flour_used :
  let wheat_flour := 0.2
  let white_flour := 0.1
  let rye_flour := 0.15
  let almond_flour := 0.05
  let oat_flour := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + oat_flour = 0.6 :=
by
  sorry

end total_flour_used_l21_21297


namespace log_function_fixed_point_l21_21091

noncomputable theory

open Real

theorem log_function_fixed_point (a m n : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1)
    (h₂ : ∀ x, f x = log a (x + m) + 1)
    (h₃ : f 2 = n) : 
    m + n = 0 :=
by
  sorry

end log_function_fixed_point_l21_21091


namespace total_unique_markings_l21_21736

theorem total_unique_markings :
  let stick_length := 1
  let markings_1_3 := {0, 1/3, 2/3, 1}
  let markings_1_5 := {0, 1/5, 2/5, 3/5, 4/5, 1}
  let all_markings := markings_1_3 ∪ markings_1_5
  all_markings.card = 8 :=
by
  sorry

end total_unique_markings_l21_21736


namespace max_f_on_interval_local_max_range_l21_21880

-- Define the function f(x)
def f (x a : ℝ) := (x - 2) * exp x - (a / 2) * (x ^ 2 - 2 * x)

-- Maximum value of f(x) on [1, 2] when a = e
theorem max_f_on_interval : f 2 Real.exp = 0 := sorry

-- Range of a for f(x) to have a local maximum at x_0 with f(x_0) < 0
theorem local_max_range (a : ℝ) (x0 : ℝ) (h₀ : f' x0 a = 0) (hx0 : f x0 a < 0) : (0 < a ∧ a < Real.exp) ∨ (Real.exp < a ∧ a < 2 * Real.exp) := sorry

end max_f_on_interval_local_max_range_l21_21880


namespace cot_difference_triangle_l21_21146

theorem cot_difference_triangle (ABC : Triangle)
  (angle_condition : ∠AD BC = 60) :
  |cot ∠B - cot ∠C| = 5 / 2 :=
sorry

end cot_difference_triangle_l21_21146


namespace ordered_pairs_count_l21_21006

theorem ordered_pairs_count : 
  (∃ (S : Set (ℤ × ℤ)), (∀ p : ℤ × ℤ, p ∈ S ↔ (p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 75 * p.1 * p.2 = 27^3)) ∧ S.card = 29) :=
sorry

end ordered_pairs_count_l21_21006


namespace imaginary_part_conjugate_of_complex_l21_21453

theorem imaginary_part_conjugate_of_complex :
  ∀ z : ℂ, z = -1 + 2 * Complex.i → Complex.im (Complex.conj z) = -2 :=
by
  intros z hz
  rw hz
  rw Complex.conj
  sorry

end imaginary_part_conjugate_of_complex_l21_21453


namespace sunny_bakes_initial_cakes_l21_21239

theorem sunny_bakes_initial_cakes (cakes_after_giving_away : ℕ) (total_candles : ℕ) (candles_per_cake : ℕ) (given_away_cakes : ℕ) (initial_cakes : ℕ) :
  cakes_after_giving_away = total_candles / candles_per_cake →
  given_away_cakes = 2 →
  total_candles = 36 →
  candles_per_cake = 6 →
  initial_cakes = cakes_after_giving_away + given_away_cakes →
  initial_cakes = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sunny_bakes_initial_cakes_l21_21239


namespace box_cookies_count_l21_21634

theorem box_cookies_count (cookies_per_bag : ℕ) (cookies_per_box : ℕ) :
  cookies_per_bag = 7 →
  8 * cookies_per_box = 9 * cookies_per_bag + 33 →
  cookies_per_box = 12 :=
by
  intros h1 h2
  sorry

end box_cookies_count_l21_21634


namespace find_xy_l21_21950

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C M N P : V}
variables {x y : ℝ}

-- Define the conditions given in the problem
def condition1 (A B C M : V) : Prop := M = (3/4) • (B - A) + (1/4) • (C - A)
def condition2 (A B C N : V) : Prop := N = (1/2) • (C - B) + (1/2) • (A - B)
def intersection (A M : V) (P : V) (x : ℝ) : Prop := P = x • (M - A)
def intersection2 (C N : V) (P : V) (y : ℝ) : Prop := P = y • (N - C)

theorem find_xy {V : Type*} [inner_product_space ℝ V] 
  (A B C M N P : V) (x y : ℝ)
  (h1 : condition1 A B C M)
  (h2 : condition2 A B C N)
  (h3 : intersection A M P x)
  (h4 : intersection2 C N P y) :
  x + y = 4 / 7 :=
sorry

end find_xy_l21_21950


namespace luke_games_l21_21997

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l21_21997


namespace sufficient_condition_a_ge_0_l21_21247

noncomputable def sufficient_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, a ≥ 0 → ax^2 + x + 1 ≥ 0

theorem sufficient_condition_a_ge_0 (a : ℝ) : (a ≥ 0) → sufficient_condition a :=
  by
    -- proof goes here (left as sorry)
    sorry

end sufficient_condition_a_ge_0_l21_21247


namespace sally_quarters_l21_21624

theorem sally_quarters :
  (initial_quarters spent_first spent_second remaining : ℕ)
  (h1 : initial_quarters = 760)
  (h2 : spent_first = 418)
  (h3 : spent_second = 215)
  (h_remaining : remaining = initial_quarters - spent_first - spent_second) :
  remaining = 127 :=
by
  rw [h1, h2, h3] at h_remaining
  exact h_remaining

end sally_quarters_l21_21624


namespace unique_combination_value_l21_21070

-- Define the possible set of values {0, 1, 2}
def possible_values := {0, 1, 2}

-- Define the conditions based on the problem
def condition1 (a : ℕ) := a ≠ 2
def condition2 (b : ℕ) := b = 2
def condition3 (c : ℕ) := c ≠ 0

-- Define the function to calculate the desired value
def calc_value (a b c : ℕ) := 100 * a + 10 * b + c

-- Prove the main statement
theorem unique_combination_value : 
  ∃ (a b c : ℕ), a ∈ possible_values ∧ b ∈ possible_values ∧ c ∈ possible_values ∧
  ((condition1 a ∧ ¬ condition2 b ∧ ¬ condition3 c) ∨
   (¬ condition1 a ∧ condition2 b ∧ ¬ condition3 c) ∨
   (¬ condition1 a ∧ ¬ condition2 b ∧ condition3 c)) ∧
  (calc_value a b c = 201) :=
by
  sorry

end unique_combination_value_l21_21070


namespace problem_l21_21915

noncomputable def sqrt (x : ℝ) : ℝ := real.sqrt x

theorem problem 
  (x y : ℝ)
  (h : |x - sqrt 3 + 1| + sqrt (y - 2) = 0) :
  x = sqrt 3 - 1 ∧ y = 2 ∧ (x^2 + 2*x - 3*y = -4) :=
by
  -- The proof itself is omitted as required  
  sorry

end problem_l21_21915


namespace find_m_and_b_sum_l21_21654

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l21_21654


namespace find_a_l21_21983

variable {a : ℝ}  -- Declaring 'a' as a real number

theorem find_a (h : (2 * a) / (1 + (1 : ℂ) * I) + (1 : ℂ) + I ∈ ℝ) : a = 1 :=
sorry

end find_a_l21_21983


namespace percent_decrease_is_30_l21_21961

def original_price : ℝ := 100
def sale_price : ℝ := 70
def decrease_in_price : ℝ := original_price - sale_price

theorem percent_decrease_is_30 : (decrease_in_price / original_price) * 100 = 30 :=
by
  sorry

end percent_decrease_is_30_l21_21961


namespace original_price_of_book_l21_21666

theorem original_price_of_book (final_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) 
  (h1 : final_price = 360) (h2 : increase_percentage = 0.20) 
  (h3 : final_price = (1 + increase_percentage) * original_price) : original_price = 300 := 
by
  sorry

end original_price_of_book_l21_21666


namespace luke_games_l21_21996

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l21_21996


namespace trapezoid_construction_l21_21131

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (A B C D O : Point)
  (AB_parallel_CD : parallel (line_segment A B) (line_segment C D))
  (diagonals_intersect : intersects (line_segment A C) (line_segment B D) = O)
  (OCB_equilateral : equilateral_triangle O C B)
  (OAD_right : right_triangle O A D)

-- Define equilateral triangle condition
def equilateral_triangle (A B C : Point) : Prop :=
  distance A B = distance B C ∧ distance B C = distance C A ∧ distance C A = 1

-- Define right triangle condition
def right_triangle (A B C : Point) : Prop :=
  ∃ (angle_ABC : ℝ), angle_ABC = 90 ∧
  distance_between_the_points B C = distance_between_the_points O D

-- Declare the specified lengths
def length_OD : ℝ := (sqrt 2) / 2
def length_OA : ℝ := sqrt 2

-- Final theorem to state the conditions
theorem trapezoid_construction :
  ∃ (ABCD : Trapezoid),
  ABCD.AB_parallel_CD ∧
  ABCD.diagonals_intersect ∧
  ABCD.OCB_equilateral ∧
  ABCD.OAD_right ∧
  distance O D = length_OD ∧
  distance O A = length_OA :=
sorry

end trapezoid_construction_l21_21131


namespace invite_students_l21_21669

theorem invite_students (total_students : ℕ) (invitees : ℕ) (A B : ℕ) :
  total_students = 10 →
  invitees = 6 →
  (∃ (f : Finset ℕ), f.card = invitees ∧ A ∉ f ∧ B ∉ f ∨ B ∈ f ∧ A ∉ f ∨ A ∈ f ∧ B ∉ f) →
  ∑ i in {0, 1, 2}, combinatorial.choose 8 (6 - i) * combinatorial.choose 2 i = 140 :=
by
  sorry

end invite_students_l21_21669


namespace ellipse_lines_intersect_l21_21443

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1^2) / 2 + p.2^2 = 1 }

def focus_left : ℝ × ℝ := (-1, 0)

def line_MN : set (ℝ × ℝ) := { p | p.1 = -1 }

def line_PQ : set (ℝ × ℝ) := { p | p.1 = 0 }

theorem ellipse_lines_intersect (M N P Q : ℝ × ℝ) (hM : M ∈ ellipse) (hN : N ∈ ellipse) (hP : P ∈ ellipse) (hQ : Q ∈ ellipse) 
  (hM_MN : M ∈ line_MN) (hN_MN : N ∈ line_MN) (hP_PQ : P ∈ line_PQ) (hQ_PQ : Q ∈ line_PQ) :
  ( dist P Q )^2 / dist M N = 2 * sqrt 2 := sorry

end ellipse_lines_intersect_l21_21443


namespace reflection_over_y_eq_x_matrix_l21_21021

-- Define the basis vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

-- Define the transformation
def A : ℝ × ℝ → ℝ × ℝ := λ p, (p.2, p.1)

-- Prove that A performs the reflection

theorem reflection_over_y_eq_x_matrix :
  A e1 = (0, 1) ∧ A e2 = (1, 0) :=
by {
  -- Proof goes here
  sorry
}

end reflection_over_y_eq_x_matrix_l21_21021


namespace range_of_m_l21_21095

noncomputable def f (m : ℝ) : ℝ → ℝ := 
λ x, if x ≤ 0 then 2^x else -x^2 + m

theorem range_of_m (m : ℝ) :
  (∀ y ∈ (Set.range (f m)), y ≤ 1) → ∃ m, 0 < m ∧ m ≤ 1 := by
  sorry

end range_of_m_l21_21095


namespace reflection_line_slope_l21_21658

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l21_21658


namespace ratio_third_to_second_is_one_l21_21011

variable (x y : ℕ)

-- The second throw skips 2 more times than the first throw
def second_throw := x + 2
-- The third throw skips y times
def third_throw := y
-- The fourth throw skips 3 fewer times than the third throw
def fourth_throw := y - 3
-- The fifth throw skips 1 more time than the fourth throw
def fifth_throw := (y - 3) + 1

-- The fifth throw skipped 8 times
axiom fifth_throw_condition : fifth_throw y = 8
-- The total number of skips between all throws is 33
axiom total_skips_condition : x + second_throw x + y + fourth_throw y + fifth_throw y = 33

-- Prove the ratio of skips in third throw to the second throw is 1:1
theorem ratio_third_to_second_is_one : (third_throw y) / (second_throw x) = 1 := sorry

end ratio_third_to_second_is_one_l21_21011


namespace ratio_of_distances_l21_21972

variables (P A B C D E : Type)
variable [Inhabited P] 
variable (h a : ℝ)
variables (E_inside : E ∈ (geometry.square A B C D) ∧ E ≠ geometry.centroid A B C D)

def distance_from_E_to_faces (s : ℝ) : Prop :=
s = 4 * real.sqrt (h ^ 2 + (a ^ 2 / 2))

def distance_from_E_to_edges (S : ℝ) : Prop :=
S = a / 2

theorem ratio_of_distances (s S : ℝ) 
  (h_pos : h > 0) (a_pos : a > 0) :
  distance_from_E_to_faces s →
  distance_from_E_to_edges S →
  s / S = 8 * real.sqrt (h ^ 2 + (a ^ 2 / 2)) / a :=
by
  intros h_pos a_pos E_inside
  sorry 

end ratio_of_distances_l21_21972


namespace exact_sunny_days_probability_l21_21099

noncomputable def choose (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

def rain_prob : ℚ := 3 / 4
def sun_prob : ℚ := 1 / 4
def days : ℕ := 5

theorem exact_sunny_days_probability : (choose days 2 * (sun_prob^2 * rain_prob^3) = 135 / 512) :=
by
  sorry

end exact_sunny_days_probability_l21_21099


namespace tan_period_odd_even_l21_21355

theorem tan_period_odd_even :
  (∀ x, tan (x + π) = tan x) ∧ (∀ x, tan (-x) = -tan x) :=
by
  -- Minimum positive period π
  have h_period : ∀ x, tan (x + π) = tan x := sorry
  -- Odd function property
  have h_odd : ∀ x, tan (-x) = -tan x := sorry
  exact ⟨h_period, h_odd⟩

end tan_period_odd_even_l21_21355


namespace iterated_kernels_l21_21019

noncomputable def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < t then 
    x + t 
  else if t < x ∧ x ≤ 1 then 
    x - t 
  else 
    0

noncomputable def K1 (x t : ℝ) : ℝ := K x t

noncomputable def K2 (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0

theorem iterated_kernels (x t : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  K1 x t = K x t ∧
  K2 x t = 
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0 := by
  sorry

end iterated_kernels_l21_21019


namespace students_absent_afternoon_l21_21618

theorem students_absent_afternoon
  (morning_registered afternoon_registered total_students morning_absent : ℕ)
  (h_morning_registered : morning_registered = 25)
  (h_morning_absent : morning_absent = 3)
  (h_afternoon_registered : afternoon_registered = 24)
  (h_total_students : total_students = 42) :
  (afternoon_registered - (total_students - (morning_registered - morning_absent))) = 4 :=
by
  sorry

end students_absent_afternoon_l21_21618


namespace train_length_proof_l21_21731

-- Define the conditions
def train_speed_kmph := 72
def platform_length_m := 290
def crossing_time_s := 26

-- Conversion factor
def kmph_to_mps := 5 / 18

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Distance covered by train while crossing the platform (in meters)
def distance_covered := train_speed_mps * crossing_time_s

-- Length of the train (in meters)
def train_length := distance_covered - platform_length_m

-- The theorem to be proved
theorem train_length_proof : train_length = 230 :=
by 
  -- proof would be placed here 
  sorry

end train_length_proof_l21_21731


namespace men_in_second_group_l21_21726

theorem men_in_second_group (M : ℕ) (h1 : 36 * 18 = M * 24) : M = 27 :=
by {
  sorry
}

end men_in_second_group_l21_21726


namespace james_recovery_time_l21_21955

theorem james_recovery_time :
  let initial_healing_time := 4
  let skin_graft_healing_time := initial_healing_time + (initial_healing_time * 0.5)
  let total_recovery_time := initial_healing_time + skin_graft_healing_time
  total_recovery_time = 10 :=
by
  let initial_healing_time := 4
  let skin_graft_healing_time := initial_healing_time + (initial_healing_time * 0.5)
  let total_recovery_time := initial_healing_time + skin_graft_healing_time
  have h : total_recovery_time = 10 := sorry
  exact h

end james_recovery_time_l21_21955


namespace power_sum_l21_21767

theorem power_sum : 1 ^ 2009 + (-1) ^ 2009 = 0 := 
by 
  sorry

end power_sum_l21_21767


namespace score_order_l21_21580

theorem score_order 
  (M Q S K : ℕ)
  (h1 : K ≥ min M (min Q S))
  (h2 : ∃ x y, x ∈ {M, Q, S, K} ∧ y ∈ {M, Q, S, K} ∧ x = y)
  (h3 : M = S)
  (h4 : S ≠ max M (max Q K)) :
  S = M ∧ S < K ∧ K < Q :=
by
  sorry -- Proof to be filled in

end score_order_l21_21580


namespace num_matrices_satisfying_congruences_l21_21178

-- Definition of the matrix
structure Matrix2x2 (α : Type) := 
(a11 a12 a21 a22 : α)

def satisfies_congruences (p : ℕ) [Nat.Prime p] (J : Matrix2x2 ℕ) : Prop :=
  (J.a11 + J.a22) % p = 1 ∧ (J.a11 * J.a22 - J.a12 * J.a21) % p = 0

-- The theorem to state the result
theorem num_matrices_satisfying_congruences (p : ℕ) [Nat.Prime p] :
  ∃! n : ℕ, n = p^2 + p ∧ (∃ L : List (Matrix2x2 ℕ), (L.length = n) ∧ (∀ J ∈ L, ∃ (a b c d : ℕ), J = Matrix2x2.mk a b c d ∧ satisfies_congruences p J)) :=
by
  sorry

end num_matrices_satisfying_congruences_l21_21178


namespace max_real_roots_polynomial_l21_21022

theorem max_real_roots_polynomial (n : ℕ) (h : 0 < n) :
  let P (x : ℝ) := ∑ i in finset.range (n + 1), (-1 : ℝ) ^ i * x ^ (n - i)
  in if n % 2 = 1 then ∀ (x : ℝ), P(x) = 0 → (x = 1) ∨ (x = -1)
     else ∀ (x : ℝ), P(x) = 0 → (x = 1 ∧ (∃ k : ℕ, k < n ∧ x = (-1) ^ (2 * k + 1))) :=
by sorry

end max_real_roots_polynomial_l21_21022


namespace num_of_containers_is_four_l21_21225

def rice_weight_pounds : ℝ := 35 / 2
def ounces_per_pound : ℝ := 16
def rice_weight_ounces : ℝ := rice_weight_pounds * ounces_per_pound
def rice_per_container : ℝ := 70
def num_containers : ℝ := rice_weight_ounces / rice_per_container

theorem num_of_containers_is_four : num_containers = 4 := by
  sorry

end num_of_containers_is_four_l21_21225


namespace smallest_n_satisfying_conditions_l21_21223

-- We need variables and statements
variables (n : ℕ)

-- Define the conditions
def condition1 : Prop := n % 6 = 4
def condition2 : Prop := n % 7 = 3
def condition3 : Prop := n > 20

-- The main theorem statement to be proved
theorem smallest_n_satisfying_conditions (h1 : condition1 n) (h2 : condition2 n) (h3 : condition3 n) : n = 52 :=
by 
  sorry

end smallest_n_satisfying_conditions_l21_21223


namespace perfect_squares_between_100_and_500_l21_21501

theorem perfect_squares_between_100_and_500 : 
  ∃ (count : ℕ), count = 12 ∧ 
    ∀ n : ℕ, (100 ≤ n^2 ∧ n^2 ≤ 500) ↔ (11 ≤ n ∧ n ≤ 22) :=
begin
  -- Proof goes here
  sorry
end

end perfect_squares_between_100_and_500_l21_21501


namespace count_of_odd_three_digit_numbers_l21_21027

-- Define the set of digits
def digits := {0, 1, 2, 3, 4, 5}

-- Define the condition that a digit must be odd
def is_odd (d : ℕ) : Prop := d % 2 = 1

-- Define the condition that a number is three-digit and odd
def is_three_digit_odd_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_odd n % 10

-- Define the function to count valid three-digit odd numbers
def count_valid_three_digit_odd_numbers : ℕ :=
  let odd_digits := {1, 3, 5} in -- unit digit choices
  let remaining_digits := digits \ {0} in -- hundreds digit choices excluding 0
  3 * 4 * 4

theorem count_of_odd_three_digit_numbers : count_valid_three_digit_odd_numbers = 48 := by
  rw [count_valid_three_digit_odd_numbers]
  sorry

end count_of_odd_three_digit_numbers_l21_21027


namespace total_people_sitting_l21_21560

theorem total_people_sitting (people_between : ℕ) (h : people_between = 30) : people_between + 2 = 32 :=
by { rw h, exact rfl }

end total_people_sitting_l21_21560


namespace exists_polyhedron_with_nonvisible_vertices_l21_21389

theorem exists_polyhedron_with_nonvisible_vertices :
  ∃ (P : Polyhedron) (p : Point), 
    p ∉ P ∧ 
    (∀ v : Point, v ∈ vertices P → ¬ visible_from p v) :=
sorry

end exists_polyhedron_with_nonvisible_vertices_l21_21389


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l21_21707

variable {a b : ℝ}

theorem correct_calculation : a ^ 3 * a = a ^ 4 := 
by
  sorry

theorem incorrect_calculation_A : a ^ 3 + a ^ 3 ≠ 2 * a ^ 6 := 
by
  sorry

theorem incorrect_calculation_B : (a ^ 3) ^ 3 ≠ a ^ 6 :=
by
  sorry

theorem incorrect_calculation_D : (a - b) ^ 2 ≠ a ^ 2 - b ^ 2 :=
by
  sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l21_21707


namespace x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l21_21307

theorem x_gt_1_implies_inv_x_lt_1 (x : ℝ) (h : x > 1) : 1 / x < 1 :=
by
  sorry

theorem inv_x_lt_1_not_necessitates_x_gt_1 (x : ℝ) (h : 1 / x < 1) : ¬(x > 1) ∨ (x ≤ 1) :=
by
  sorry

end x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l21_21307


namespace lily_drive_distance_l21_21993

-- Define the conditions
def car_mileage (miles_per_gallon gallon : ℕ) : ℕ := miles_per_gallon * gallon 

def length_drive (initial_gas gas_used miles_per_gallon : ℕ) : ℕ := 
  initial_gas - gas_used * miles_per_gallon

theorem lily_drive_distance : 
  ∀ (tank_capacity miles_per_gallon : ℕ)
  (initial_gas drive1 additional_gas1 drive2 additional_gas2 final_gas : ℕ),
  tank_capacity = 12 →
  miles_per_gallon = 40 →
  initial_gas = 12 →
  drive1 = 480 →
  additional_gas1 = 6 →
  drive2 = (additional_gas1 * miles_per_gallon) →
  additional_gas2 = 4 →
  final_gas = (9 * 12 / 4) →
  length_drive initial_gas [(drive1 / miles_per_gallon), (drive2 / miles_per_gallon), (additional_gas2 * miles_per_gallon)] = 880 :=
by
  sorry

end lily_drive_distance_l21_21993


namespace product_complex_l21_21789

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem product_complex : 
  ∏ k in finset.range 15, (∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
sorry

end product_complex_l21_21789


namespace problem_1_problem_2_l21_21460

noncomputable def f (x : ℝ) : ℝ := 
  √3 * (sin (x + π / 4))^2 - (cos x)^2 - (1 + √3) / 2

theorem problem_1 :
  (∀ x : ℝ, f x ≥ -2) ∧ ((∃ x : ℝ, f x = -2) ∧ f (x + π) = f x) :=
sorry

theorem problem_2 (A : ℝ) (hA : 0 < A ∧ A < π / 2) :
  let n := (1, f (π / 4 - A)) 
  in (5 * (f (π / 4 - A)) = -1) → 
     cos (2 * A) = (4 * √3 + 3) / 10 :=
sorry

end problem_1_problem_2_l21_21460


namespace circles_common_tangents_not_2_l21_21278

-- Definition of circles with radii r and 2r
noncomputable def Circle (r : ℝ) := { x // x = r }

-- The theorem to prove
theorem circles_common_tangents_not_2 (r : ℝ) : 
  ∀ (c1 c2 : Circle r), ¬ (number_of_common_tangents c1 c2 = 2) :=
  sorry

end circles_common_tangents_not_2_l21_21278


namespace cot_diff_equal_l21_21149

variable (A B C D : Type)

-- Define the triangle and median.
variable [triangle ABC : Type] (median : Type)

-- Define the angle condition.
def angle_condition (ABC : triangle) (AD : median) : Prop :=
  ∠(AD, BC) = 60

-- Prove the cotangent difference
theorem cot_diff_equal
  (ABC : triangle)
  (AD : median)
  (h : angle_condition ABC AD) :
  abs (cot B - cot C) = (9 - 3 * sqrt 3) / 2 :=
by
  sorry -- Proof to be constructed

end cot_diff_equal_l21_21149


namespace largest_square_area_l21_21359

theorem largest_square_area (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R]
  (angle_PQR_right : angle P Q R = π / 2)
  (sum_areas_338 : (dist P R)^2 + (dist P Q)^2 + (dist Q R)^2 = 338) :
  (dist P R)^2 = 169 :=
by
  sorry

end largest_square_area_l21_21359


namespace a_n_is_perfect_square_l21_21065

def sequence_c (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 0
  else if n = 2 then 2005
  else -3 * sequence_c (n - 2) - 4 * sequence_c (n - 3) + 2008

def sequence_a (n : ℕ) :=
  if n < 2 then 0
  else 5 * (sequence_c (n + 2) - sequence_c n) * (502 - sequence_c (n - 1) - sequence_c (n - 2)) + (4 ^ n) * 2004 * 501

theorem a_n_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, sequence_a n = k^2 :=
sorry

end a_n_is_perfect_square_l21_21065


namespace reflection_line_slope_intercept_l21_21650

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l21_21650


namespace exponent_division_l21_21700

theorem exponent_division :
  (1000 ^ 7) / (10 ^ 17) = 10 ^ 4 := 
  sorry

end exponent_division_l21_21700


namespace num_valid_pairs_l21_21904

-- Define variables for 3003 and its factorization
def N := 3003
def factors := [3, 7, 11, 13]

-- Define the gcd condition
def gcd_condition (i : ℕ) : Prop :=
  Nat.gcd (10 ^ i) N = 1

-- Define the divisibility condition
def divisibility_condition (j i : ℕ) : Prop :=
  (N ∣ (10 ^ (j - i) - 1))

-- Define the range condition for i and j
def range_condition (i j : ℕ) : Prop :=
  0 ≤ i ∧ i < j ∧ j ≤ 99

-- Define the main condition that combines everything
def main_condition (i j : ℕ) : Prop :=
  range_condition i j ∧ gcd_condition i ∧ divisibility_condition j i

-- Define the set of valid (i, j) pairs and count them
def valid_pairs : ℕ :=
  (Finset.range 100).sum (λ i, (Finset.range (100 - i - 1)).count (λ j, main_condition i (j + i + 1)))

-- State the theorem
theorem num_valid_pairs : valid_pairs = 368 :=
  sorry

end num_valid_pairs_l21_21904


namespace find_ab_l21_21918
-- Import the necessary Lean libraries 

-- Define the statement for the proof problem
theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : ab = 9 :=
by {
    sorry
}

end find_ab_l21_21918


namespace solve_quadratic_eq_l21_21629

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l21_21629


namespace reflection_line_slope_l21_21660

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l21_21660


namespace inequality_abc_l21_21048

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥
  (3 / (Real.cbrt (a * b * c) * (1 + Real.cbrt (a * b * c)))) :=
by
  sorry

end inequality_abc_l21_21048


namespace luke_games_l21_21995

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l21_21995


namespace count_perfect_squares_between_100_500_l21_21489

theorem count_perfect_squares_between_100_500 :
  ∃ (count : ℕ), count = finset.card ((finset.Icc 11 22).filter (λ n, 100 < n^2 ∧ n^2 < 500)) :=
begin
  use 12,
  rw ← finset.card_Icc,
  sorry
end

end count_perfect_squares_between_100_500_l21_21489


namespace amount_earned_by_16_men_l21_21521

theorem amount_earned_by_16_men (
    W : ℝ 
    (h1 : 40 * 30 * W = 21600)
    (h2 : ∀ M : ℝ, M = 2 * W)
    (h3 : ∀ amount : ℝ, 16 * 25 * M = amount)
    )
    : 16 * 25 * (2 * W) = 14400 := by
  sorry

end amount_earned_by_16_men_l21_21521


namespace sub_neg_four_l21_21370

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l21_21370


namespace find_phone_number_l21_21207

-- Define the conditions
def is_palindrome (l: List ℕ) : Prop := l = l.reverse
def is_divisible_by_9 (n: ℕ) : Prop := n % 9 = 0
def is_prime (n: ℕ) : Prop := nat.prime n
def contains_three_consecutive_ones (n: ℕ) : Prop := ∃ i, (n.to_string.nth i = '1') ∧ (n.to_string.nth (i+1) = '1') ∧ (n.to_string.nth (i+2) = '1')
def last_three_consecutive (n: ℕ) : Prop := 
  let digits := n.to_string.data.drop 4 in
  (∃ x1 x2 x3 : ℕ, [x1, x2, x3] = digits.map (λ c, c.to_digit 10).reverse ∧ (x1 + 1 = x2) ∧ (x2 + 1 = x3))

-- Define the goal statement
theorem find_phone_number (n: ℕ) (h1: n / 10^6 < 10) (h2: contains_three_consecutive_ones n)
  (h3: last_three_consecutive n) (h4: is_palindrome ([n / 10^4 % 10, n / 10^3 % 10, n / 10^2 % 10, n / 10^1 % 10, n % 10]))
  (h5: is_divisible_by_9 (n / 10^4)) 
  (h6: ∃ m, is_prime m ∧ (n / 10 % 100 = m ∨ n % 100 = m)) :
  n = 7111765 := 
  sorry

end find_phone_number_l21_21207


namespace find_m_and_b_sum_l21_21655

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l21_21655


namespace cupcakes_left_l21_21189

def num_packages : ℝ := 3.5
def cupcakes_per_package : ℝ := 7
def cupcakes_eaten : ℝ := 5.75

theorem cupcakes_left :
  num_packages * cupcakes_per_package - cupcakes_eaten = 18.75 :=
by
  sorry

end cupcakes_left_l21_21189


namespace range_of_m_l21_21851

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotonically_decreasing_in_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_m (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing_in_domain f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → -2 < m ∧ m < 1 :=
sorry

end range_of_m_l21_21851


namespace problem_solution_l21_21780

noncomputable def product(problem1: ℕ, problem2: ℕ): ℂ :=
∏ k in (finset.range problem1).image (λ (n : ℕ), e ^ (2 * π * complex.I * n / 17)),
  ∏ j in (finset.range problem2).image (λ (m : ℕ), e ^ (2 * π * complex.I * m / 13)),
    (j - k)

theorem problem_solution : product 15 12 = 13 := 
sorry

end problem_solution_l21_21780


namespace arithmetic_mean_of_first_n_integers_l21_21640

theorem arithmetic_mean_of_first_n_integers (m n : ℤ) (hm : m ≠ 0) :
  (1 / (n : ℚ)) * (finset.sum (finset.range n) (λ k, m + k)) = (2 * m + n - 1) / 2 :=
by
  sorry

end arithmetic_mean_of_first_n_integers_l21_21640


namespace measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l21_21153

noncomputable def measure_of_angle_a (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : Prop :=
  A = Real.pi / 3

theorem measure_of_angle_A_proof (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : measure_of_angle_a a b c S h_c h_S :=
sorry

noncomputable def range_of_values_of_b_plus_c_over_a (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : Set ℝ :=
  {x : ℝ | 1 < x ∧ x ≤ 2}

theorem range_of_values_of_b_plus_c_over_a_proof (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : 
  ∃ x, x ∈ range_of_values_of_b_plus_c_over_a a b c A h_A h_c :=
sorry

end measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l21_21153


namespace train_cross_platform_time_l21_21322

-- Definitions based on given conditions
def train_length : ℝ := 300
def time_to_cross_signal_pole : ℝ := 24
def platform_length : ℝ := 187.5

-- The speed of the train
def train_speed : ℝ := train_length / time_to_cross_signal_pole

-- The total distance to cross both the train and platform
def total_distance : ℝ := train_length + platform_length

-- The time to cross the platform
def time_to_cross_platform : ℝ := total_distance / train_speed

-- The theorem stating the time
theorem train_cross_platform_time : time_to_cross_platform = 39 := 
by
  sorry

end train_cross_platform_time_l21_21322


namespace M_moves_along_circle_l21_21222

variables {A B P Q X S T M ω : Type}
variables (circle : ω.is_circle)
variables (APBQ : is_inscribed_quadrilateral A P B Q ω)
variables (angleP : angle P = 90)
variables (angleQ : angle Q = 90)
variables (AP_AQ : A.distance_to P = A.distance_to Q)
variables (BP : B.distance_to P > A.distance_to P)
variables (X_on_PQ : X ∈ segment P Q)
variables (AX_S : AX_meets_circle_S A X ω S)
variables (T_on_arc : T ∈ arc A Q B (ω))
variables (_XT_perp_AX : XT_perpendicular_AX X T A X)

def M_midpoint (S T : Type) : Type := midpoint M S T

theorem M_moves_along_circle {X : Type} (h : X ∈ segment P Q) :
  moves_along_circle (λ X, midpoint (intersection AX ω S) T X) :=
sorry

end M_moves_along_circle_l21_21222


namespace jason_books_is_21_l21_21163

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l21_21163


namespace minimum_value_y_l21_21031

noncomputable def tan_22_5 := Real.tan (Real.pi / 8)

theorem minimum_value_y (x : ℝ) (hx : 1 < x) :
  let m := tan_22_5 / (1 - tan_22_5^2),
      y := 2 * m * x + 3 / (x - 1) + 1
  in y ≥ 2 + 2 * Real.sqrt 3 := by
  sorry

end minimum_value_y_l21_21031


namespace compute_product_l21_21784

noncomputable def P (x : ℂ) : ℂ := ∏ k in (finset.range 15).map (nat.cast), (x - exp (2 * pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ := ∏ j in (finset.range 12).map (nat.cast), (x - exp (2 * pi * complex.I * j / 13))

theorem compute_product : 
  (∏ k in (finset.range 15).map (nat.cast), ∏ j in (finset.range 12).map (nat.cast), (exp (2 * pi * complex.I * j / 13) - exp (2 * pi * complex.I * k / 17))) = 1 :=
by
  sorry

end compute_product_l21_21784


namespace find_matrix_M_l21_21819

open Matrix
open FinVec

def e₁ : Fin 4 → ℝ := λ i, if i = 0 then 1 else 0
def e₂ : Fin 4 → ℝ := λ i, if i = 1 then 1 else 0
def e₃ : Fin 4 → ℝ := λ i, if i = 2 then 1 else 0
def e₄ : Fin 4 → ℝ := λ i, if i = 3 then 1 else 0

def M : Matrix (Fin 4) (Fin 4) ℝ := fun i j => if i = j then -2 else 0

theorem find_matrix_M (u : Fin 4 → ℝ) (h1 : M.vecMul u = -2 • u) (h2 : M.vecMul e₄ = ![0, 0, 0, -2]) :
  M = λ i j, if i = j then -2 else 0 :=
by
  sorry

end find_matrix_M_l21_21819


namespace proof_of_A_inter_complement_B_l21_21052

variable (U : Set Nat) 
variable (A B : Set Nat)

theorem proof_of_A_inter_complement_B :
    (U = {1, 2, 3, 4}) →
    (B = {1, 2}) →
    (compl (A ∪ B) = {4}) →
    (A ∩ compl B = {3}) :=
by
  intros hU hB hCompl
  sorry

end proof_of_A_inter_complement_B_l21_21052


namespace find_a_l21_21442

noncomputable def x := Real.sqrt (7 - 4 * Real.sqrt 3)
def a := (x^2 - 4 * x + 5) / (x^2 - 4 * x + 3)

theorem find_a : a = 2 := by
  sorry

end find_a_l21_21442


namespace intersection_of_lines_l21_21697

theorem intersection_of_lines : ∃ x y : ℚ, y = 3 * x ∧ y - 5 = -7 * x ∧ x = 1 / 2 ∧ y = 3 / 2 :=
by
  sorry

end intersection_of_lines_l21_21697


namespace expr_equals_2_l21_21769

noncomputable def calculate_expr : ℝ :=
  (real.sqrt 3) ^ 2 + (4 - real.pi) ^ 0 - abs (-3) + real.sqrt 2 * real.cos (real.pi / 4)

theorem expr_equals_2 : calculate_expr = 2 := by
  sorry

end expr_equals_2_l21_21769


namespace minimum_real_roots_l21_21986

noncomputable def f (x : ℂ) : ℂ := sorry -- the polynomial function f(x)

def degree : ℕ := 2006  -- the degree of the polynomial
def roots : fin 2006 → ℂ := sorry  -- the array of complex roots c_i
def magnitudes : set ℝ := { |roots i| | i : fin 2006 }  -- set of magnitudes of the roots
def distinct_magnitudes : ℕ := 1006  -- number of distinct magnitudes

axiom polynomial_degree : polynomial.degree f = degree  -- polynomial f has degree 2006
axiom distinct_magnitude_count : set.card magnitudes = distinct_magnitudes  -- exactly 1006 distinct magnitudes

theorem minimum_real_roots : ∃ r ≥ 6, ∀ roots : finset ℂ, (∀ i, roots i ∈ ℝ) → roots.card = r :=
  sorry -- statement to assert the minimum number of real roots is 6

end minimum_real_roots_l21_21986


namespace avg_diff_noah_liam_l21_21257

-- Define the daily differences over 14 days
def daily_differences : List ℤ := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

-- Define the function to calculate the average difference
def average_daily_difference (daily_diffs : List ℤ) : ℚ :=
  (daily_diffs.sum : ℚ) / daily_diffs.length

-- The proposition we want to prove
theorem avg_diff_noah_liam : average_daily_difference daily_differences = 60 / 14 := by
  sorry

end avg_diff_noah_liam_l21_21257


namespace train_stops_at_t_30_l21_21749

noncomputable def distance_traveled (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

theorem train_stops_at_t_30 :
  (∃ t : ℝ, t = 30 ∧ ∀ t' : ℝ, distance_traveled t' - distance_traveled t ≥ 0) ∧
  (distance_traveled 30 = 405) :=
by
  have t := 30
  have h1 : t = 30 := rfl
  have h2 : distance_traveled t = 405 := by
    calc
      distance_traveled t = 27 * t - 0.45 * t^2 := rfl
                     ... = 27 * 30 - 0.45 * 30^2 := by rw h1
                     ... = 405 := by norm_num
  split
  · use t
    -- show t = 30
    exact ⟨h1, sorry⟩
  · exact h2

end train_stops_at_t_30_l21_21749


namespace total_area_covered_l21_21363

theorem total_area_covered :
  let A_rect := 40 →
  let A_square := 25 →
  let A_tri := 12 →
  let A_rect_square := 6 →
  let A_rect_tri := 4 →
  let A_square_tri := 3 →
  let A_all := 2 →
  A_rect + A_square + A_tri - A_rect_square - A_rect_tri - A_square_tri + A_all = 66 :=
by
  intros A_rect A_square A_tri A_rect_square A_rect_tri A_square_tri A_all
  sorry

end total_area_covered_l21_21363


namespace choose_8_3_l21_21937

/- 
  Prove that the number of ways to choose 3 elements out of 8 is 56 
-/
theorem choose_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end choose_8_3_l21_21937


namespace operation_result_l21_21699

def a : ℝ := 0.8
def b : ℝ := 0.5
def c : ℝ := 0.40

theorem operation_result :
  (a ^ 3 - b ^ 3 / a ^ 2 + c + b ^ 2) = 0.9666875 := by
  sorry

end operation_result_l21_21699


namespace max_n_sum_pos_largest_term_seq_l21_21607

-- Define the arithmetic sequence {a_n} and sum of first n terms S_n along with given conditions
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

variable (a_1 d : ℤ)
-- Conditions from problem
axiom a8_pos : arithmetic_seq a_1 d 8 > 0
axiom a8_a9_neg : arithmetic_seq a_1 d 8 + arithmetic_seq a_1 d 9 < 0

-- Prove the maximum n for which Sum S_n > 0 is 15
theorem max_n_sum_pos : ∃ n_max : ℤ, sum_arith_seq a_1 d n_max > 0 ∧ 
  ∀ n : ℤ, n > n_max → sum_arith_seq a_1 d n ≤ 0 := by
    exact ⟨15, sorry⟩  -- Substitute 'sorry' for the proof part

-- Determine the largest term in the sequence {S_n / a_n} for 1 ≤ n ≤ 15
theorem largest_term_seq : ∃ n_largest : ℤ, ∀ n : ℤ, 1 ≤ n → n ≤ 15 → 
  (sum_arith_seq a_1 d n / arithmetic_seq a_1 d n) ≤ (sum_arith_seq a_1 d n_largest / arithmetic_seq a_1 d n_largest) := by
    exact ⟨8, sorry⟩  -- Substitute 'sorry' for the proof part

end max_n_sum_pos_largest_term_seq_l21_21607


namespace product_of_digits_in_base7_7891_is_zero_l21_21289

/-- The function to compute the base 7 representation. -/
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else 
    let rest := to_base7 (n / 7)
    rest ++ [n % 7]

/-- The function to compute the product of the digits of a list. -/
def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * d) 1

theorem product_of_digits_in_base7_7891_is_zero :
  product_of_digits (to_base7 7891) = 0 := by
  sorry

end product_of_digits_in_base7_7891_is_zero_l21_21289


namespace infinite_sets_unique_representation_l21_21391

theorem infinite_sets_unique_representation :
  ∃ (A B : set ℕ), (∀ n : ℕ, ∃! (a b : ℕ), (a ∈ A ∧ b ∈ B ∧ n = a + b)) ∧
                   A.nonempty ∧ B.nonempty :=
by {
  -- A and B exist, every non-negative integer n can be uniquely represented as n = a + b with a ∈ A and b ∈ B
  sorry
}

end infinite_sets_unique_representation_l21_21391


namespace standard_parts_bounds_l21_21742

noncomputable def n : ℕ := 900
noncomputable def p : ℝ := 0.9
noncomputable def confidence_level : ℝ := 0.95
noncomputable def lower_bound : ℝ := 792
noncomputable def upper_bound : ℝ := 828

theorem standard_parts_bounds : 
  792 ≤ n * p - 1.96 * (n * p * (1 - p)).sqrt ∧ 
  n * p + 1.96 * (n * p * (1 - p)).sqrt ≤ 828 :=
sorry

end standard_parts_bounds_l21_21742


namespace fraction_of_jugX_after_pour_l21_21306

variables (Cx Cy : ℝ)
variables (x_full y_full : ℝ)

-- Definitions based on the conditions given
def jugX_initial_fraction_full := 1 / 3
def jugY_initial_fraction_full := 2 / 3
def water_poured_into_y := Cy * (1 - jugY_initial_fraction_full)

-- Condition: initial water content
def waterX := jugX_initial_fraction_full * Cx
def waterY := jugY_initial_fraction_full * Cy

axiom equal_initial_water (Cx Cy : ℝ) : waterX = waterY

theorem fraction_of_jugX_after_pour :
  (Cx : Type) -> (Cy : Type) ->
  waterX - water_poured_into_y  / Cx = 1 / 6 :=
begin
  sorry
end

end fraction_of_jugX_after_pour_l21_21306


namespace ratio_problem_l21_21671

/-- Define the sequence based on the initial conditions and recurrence relation -/
def a : ℕ → ℝ
| 0     := 2  -- Here we use 0 instead of 1 to align with Lean's ℕ starting from 0
| (n+1) := (2 * (n + 3) / (n + 2)) * a n

/-- Define the sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, a k)

/-- The main theorem: the desired ratio -/
theorem ratio_problem : 
  (a 2013 + 2 * (2016 / 2015)) / (a 0 + S 2013) = 2015 / 2013 :=
  sorry

end ratio_problem_l21_21671


namespace find_m_l21_21465

-- Conditions given in the problem
variables (p m : ℝ) (hp : 0 < p) (hm : 1 + p / 2 = 5)

-- Parabola equation and point M on it
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
def M_on_parabola : Prop := parabola 1 m

-- Lean statement asserting the proof problem
theorem find_m (hM : M_on_parabola) (dist_focus : (1 - p / 2)^2 + m^2 = 25) :
  m = 4 ∨ m = -4 := 
sorry

end find_m_l21_21465


namespace pyramid_edges_sum_l21_21335

theorem pyramid_edges_sum
  (base_side_length : ℝ)
  (height : ℝ)
  (h1 : base_side_length = 8)
  (h2 : height = 10) :
  let diagonal := base_side_length * real.sqrt 2,
      half_diagonal := diagonal / 2,
      slant_height := real.sqrt (height * height + half_diagonal * half_diagonal)
  in
  4 * base_side_length + 4 * slant_height ≈ 78 :=
by
  -- Mathlib includes real.sqrt and real.sqrt properties.
  let diagonal := base_side_length * real.sqrt 2,
      half_diagonal := diagonal / 2,
      slant_height := real.sqrt (height * height + half_diagonal * half_diagonal)
  in
  have h3 : diagonal = 8 * real.sqrt 2 := by sorry,
  have h4 : half_diagonal = 4 * real.sqrt 2 := by sorry,
  have h5 : slant_height ≈ 11.489 := by sorry,
  have h6 : 4 * base_side_length + 4 * slant_height ≈ 77.956 := by sorry,
  have h7 : (4 * 8 + 4 * 11.489) ≈ 77.956 := by sorry,
  have h8 : 77.956 ≈ 78 := by sorry,
  h8 -- The final goal


end pyramid_edges_sum_l21_21335


namespace no_real_solutions_l21_21007

theorem no_real_solutions : ∀ x : ℝ, ¬(3 * x - 2 * x + 8) ^ 2 = -|x| - 4 :=
by
  intro x
  sorry

end no_real_solutions_l21_21007


namespace count_perfect_squares_between_100_500_l21_21488

theorem count_perfect_squares_between_100_500 :
  ∃ (count : ℕ), count = finset.card ((finset.Icc 11 22).filter (λ n, 100 < n^2 ∧ n^2 < 500)) :=
begin
  use 12,
  rw ← finset.card_Icc,
  sorry
end

end count_perfect_squares_between_100_500_l21_21488


namespace intersection_A_B_is_correct_l21_21071

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
noncomputable def B : Set ℝ := {x | 2 - x > 0}
def A_inter_B : Set ℝ := {-1 ≤ x ∧ x < 2}

theorem intersection_A_B_is_correct : (A ∩ B) = [-1, 2) := by
  sorry

end intersection_A_B_is_correct_l21_21071


namespace math_problem_l21_21884

def f (x : ℝ) : ℝ := (x - 1)^2 - 4
def a (x : ℝ) (n : ℕ) : ℝ :=
if x = 0 then -3/2 * (n - 1)
else 3/2 * (n - 3)

theorem math_problem (x : ℝ) (a1 a3 : ℝ) :
  (f (x + 1) = x^2 - 4) →
  (a1 = f (x - 1)) →
  (a2 : ℝ) → (a2 = -3/2) →
  (a3 = f x) →
  (a1 + a3 = 2 * a2) →
  (x = 0 ∨ x = 3) ∧ 
  (∀ n : ℕ, (x = 0 → a x n = -3/2 * (n - 1)) ∧ 
            (x = 3 → a x n = 3/2 * (n - 3))) :=
by
  sorry

end math_problem_l21_21884


namespace average_more_than_liam_l21_21616

theorem average_more_than_liam 
  (students_apple_counts : List ℕ) 
  (h_len : students_apple_counts.length = 9)
  (h_counts : students_apple_counts = [3, 5, 9, 6, 8, 7, 4, 5, 2]) :
  (students_apple_counts.sum.toFloat / 9 - 2) = 3.444 := by
  norm_num [students_apple_counts.sum, students_apple_counts.length]
  sorry

end average_more_than_liam_l21_21616


namespace pyramid_lateral_surface_area_l21_21648

noncomputable def lateral_surface_area_of_pyramid (H R : ℝ) : ℝ :=
  (R / 4) * (4 * H + Real.sqrt (3 * R^2 + 12 * H^2))

theorem pyramid_lateral_surface_area (H R : ℝ) :
  let surface_area := (R / 4) * (4 * H + Real.sqrt (3 * R^2 + 12 * H^2))
  in
  lateral_surface_area_of_pyramid H R = surface_area :=
by
  sorry

end pyramid_lateral_surface_area_l21_21648


namespace measure_of_side_XY_l21_21276

-- Define the conditions of the problem
def is_isosceles_right_triangle (XYZ : Triangle) : Prop :=
XYZ.isosceles ∧ XYZ.right_angle

def is_hypotenuse (XY : ℝ) (a : ℝ) : Prop :=
XY = a * Math.sqrt 2

def triangle_area (XYZ : Triangle) (A : ℝ) : Prop :=
A = (XYZ.side1 * XYZ.side2) / 2

-- Define the main theorem to be proven
theorem measure_of_side_XY
  (XYZ : Triangle)
  (H1 : is_isosceles_right_triangle XYZ)
  (H2 : XYZ.side1 > XYZ.side2)
  (H3 : triangle_area XYZ 64) :
  XYZ.side1 = 16 :=
sorry

end measure_of_side_XY_l21_21276


namespace line_exists_symmetric_diagonals_l21_21852

-- Define the initial conditions
def Circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def Line_l1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the symmetric circle C about the line l1
def Symmetric_Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the origion and intersection points
def Point_O : (ℝ × ℝ) := (0, 0)
def Point_Intersection (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := ∃ x_A y_A x_B y_B : ℝ,
  l x_A = y_A ∧ l x_B = y_B ∧ Symmetric_Circle x_A y_A ∧ Symmetric_Circle x_B y_B

-- Define diagonal equality condition
def Diagonals_Equal (O A S B : ℝ × ℝ) : Prop := 
  let (xO, yO) := O
  let (xA, yA) := A
  let (xS, yS) := S
  let (xB, yB) := B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xS)^2 + (yB - yS)^2

-- Prove existence of line where diagonals are equal and find the equation
theorem line_exists_symmetric_diagonals :
  ∃ l : ℝ → ℝ, (l (-1) = 0) ∧
    (∃ (A B S : ℝ × ℝ), Point_Intersection l A B ∧ Diagonals_Equal Point_O A S B) ∧
    (∀ x : ℝ, l x = x + 1) :=
by
  sorry

end line_exists_symmetric_diagonals_l21_21852


namespace Jans_original_speed_l21_21576

theorem Jans_original_speed
  (doubled_speed : ℕ → ℕ) (skips_after_training : ℕ) (time_in_minutes : ℕ) (original_speed : ℕ) :
  (∀ (s : ℕ), doubled_speed s = 2 * s) → 
  skips_after_training = 700 → 
  time_in_minutes = 5 → 
  (original_speed = (700 / 5) / 2) → 
  original_speed = 70 := 
by
  intros h1 h2 h3 h4
  exact h4

end Jans_original_speed_l21_21576


namespace perpendicular_planes_l21_21893

-- Definitions of the planes (α), (β), and lines (m), (n)
variables (α β: Type) [Plane α] [Plane β] (m n: Type) [Line m] [Line n]

-- Conditions:
-- m is perpendicular to α.
-- m is parallel to n.
-- n is contained in β.

-- Statement of the theorem to prove
theorem perpendicular_planes
  (h1: is_perpendicular m α)
  (h2: is_parallel m n)
  (h3: is_contained n β) :
  is_perpendicular α β :=
sorry

end perpendicular_planes_l21_21893


namespace compound_interest_calculation_l21_21672

theorem compound_interest_calculation :
  let SI := (1833.33 * 16 * 6) / 100
  let CI := 2 * SI
  let principal_ci := 8000
  let rate_ci := 20
  let n := Real.log (1.4399995) / Real.log (1 + rate_ci / 100)
  n = 2 := by
  sorry

end compound_interest_calculation_l21_21672


namespace joe_probability_diff_fruit_l21_21578

-- Define the problem conditions in Lean
def fruits : Type := {f : fin 4 // f.val < 4}
instance : DecidableEq fruits := by apply_instance

def meal := {m : fin 3 // m.val < 3}
instance : DecidableEq meal := by apply_instance

noncomputable def random_fruit (m : meal) : Prob fruits := uniform

-- Define the problem and state the theorem
theorem joe_probability_diff_fruit :
  let probability_same_fruit := ((1 : ℚ) / 4) ^ 3 * 4 in
  1 - probability_same_fruit = 15/16 :=
  by
    let probability_same_fruit := ((1 : ℚ) / 4) ^ 3 * 4
    exact calc
      1 - probability_same_fruit = 1 - (1/4)^3 * 4 : by sorry
      ... = 1 - 4 * (1/4)^3 : by sorry
      ... = 1 - 4 * 1/64 : by sorry
      ... = 1 - 1/16               : by sorry
      ... = 15/16                  : by sorry

end joe_probability_diff_fruit_l21_21578


namespace minimum_crooks_l21_21552

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l21_21552


namespace perfect_squares_count_l21_21507

theorem perfect_squares_count : (finset.filter (λ n, n * n ≥ 100 ∧ n * n ≤ 500) (finset.range 23)).card = 13 :=
by
  sorry

end perfect_squares_count_l21_21507


namespace minimum_crooks_l21_21548

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l21_21548


namespace count_perfect_squares_between_100_and_500_l21_21498

def smallest_a (x : ℕ) : ℕ :=
  Nat.find (λ a, a^2 > x)

def largest_b (x : ℕ) : ℕ :=
  Nat.find (λ b, b^2 > x) - 1

theorem count_perfect_squares_between_100_and_500 :
  let a := smallest_a 100
  let b := largest_b 500
  b - a + 1 = 12 :=
by
  -- Definitions based on conditions
  let a := smallest_a 100
  have ha : a = 11 := 
    -- the proof follows here
    sorry
  let b := largest_b 500
  have hb : b = 22 := 
    -- the proof follows here
    sorry
  calc
    b - a + 1 = 22 - 11 + 1 : by rw [ha, hb]
           ... = 12          : by norm_num

end count_perfect_squares_between_100_and_500_l21_21498


namespace max_f_on_interval_local_max_range_l21_21881

-- Define the function f(x)
def f (x a : ℝ) := (x - 2) * exp x - (a / 2) * (x ^ 2 - 2 * x)

-- Maximum value of f(x) on [1, 2] when a = e
theorem max_f_on_interval : f 2 Real.exp = 0 := sorry

-- Range of a for f(x) to have a local maximum at x_0 with f(x_0) < 0
theorem local_max_range (a : ℝ) (x0 : ℝ) (h₀ : f' x0 a = 0) (hx0 : f x0 a < 0) : (0 < a ∧ a < Real.exp) ∨ (Real.exp < a ∧ a < 2 * Real.exp) := sorry

end max_f_on_interval_local_max_range_l21_21881


namespace not_prime_by_changing_one_digit_l21_21571

theorem not_prime_by_changing_one_digit (n : ℕ) : 
  ¬ (∃ m : ℕ, (∃ d : ℕ, d < 10 ∧ ∃ i : ℕ, i < (nat.digits 10 n).length ∧ m = list.to_nat (list.update_nth (nat.digits 10 n) i d)) ∧ nat.prime m) := 
sorry

end not_prime_by_changing_one_digit_l21_21571


namespace range_of_a_l21_21458

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 / 2 then x / 3 else 2 * x ^ 3 / (x + 1)

def g (a x : ℝ) : ℝ :=
  a * x - a / 2 + 3

theorem range_of_a (a : ℝ) (hpos : 0 < a) :
  (∀ x1 ∈ set.Icc 0 1, ∃ x2 ∈ set.Icc 0 (1 / 2), f x1 = g a x2) ↔ a ≥ 6 :=
sorry

end range_of_a_l21_21458


namespace limit_integral_eq_l21_21968

-- Define the necessary elements 
def tangent_point (t : ℝ) : ℝ × ℝ := (t, t^2 / 2)

def circle_center_x (t : ℝ) : ℝ := t * (1 + real.sqrt (t^2 + 1)) / 2

def circle_center_y (t : ℝ) : ℝ := (t^2 + 1 - real.sqrt (t^2 + 1)) / 2

noncomputable def integrand (t : ℝ) : ℝ := circle_center_x t * circle_center_y t

theorem limit_integral_eq :
  (∃ x y : ℝ → ℝ, (x = circle_center_x ∧ y = circle_center_y) ∧ 
  (∀ t : ℝ, 0 < t → x t = t * (1 + real.sqrt (t^2 + 1)) / 2)) → 
  (∀ t : ℝ, 0 < t → y t = (t^2 + 1 - real.sqrt (t^2 + 1)) / 2) → 
  ∀ t : ℝ, ∃ r : ℝ, (0 < r → ∫ u in r..1, integrand u = 
  (∫ u in r..1, circle_center_x u * circle_center_y u)) → 
  (∀ r > 0, (t) = ∫ u in r..1, circle_center_x u * circle_center_y u) = (∫ u in r..1, (u * (1 + real.sqrt (u^2 + 1)) / 2) * 
  ((u^2 + 1 - real.sqrt (u^2 + 1)) / 2)) = sorry := 
  ∀ (0 < r → ∫ t in r..1, (t^3) * real.sqrt (t^2 + 1) / 4 →  (∫ t in r..1, t^3 * real.sqrt (t^2 + 1) / 4) = (sqrt(2)+1)/30 := 
 sorry

end limit_integral_eq_l21_21968


namespace arrival_time_l21_21298

def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem arrival_time (departure_time : ℕ) (stop1 stop2 stop3 travel_hours : ℕ) (stops_total_time := stop1 + stop2 + stop3) (stops_total_hours := minutes_to_hours stops_total_time) : 
  departure_time = 7 → 
  stop1 = 25 → 
  stop2 = 10 → 
  stop3 = 25 → 
  travel_hours = 12 → 
  (departure_time + (travel_hours - stops_total_hours)) % 24 = 18 :=
by
  sorry

end arrival_time_l21_21298


namespace john_paid_correct_amount_l21_21959

def cost_bw : ℝ := 160
def markup_percentage : ℝ := 0.5

def cost_color : ℝ := cost_bw * (1 + markup_percentage)

theorem john_paid_correct_amount : 
  cost_color = 240 := 
by
  -- proof required here
  sorry

end john_paid_correct_amount_l21_21959


namespace isabella_initial_hair_length_l21_21156

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l21_21156


namespace triangle_inequality_integer_solutions_l21_21680

theorem triangle_inequality_integer_solutions :
  let s1 (m : ℤ) := 2 * m - 1
  let s2 (m : ℤ) := 4 * m + 5
  let s3 (m : ℤ) := 20 - m
  (∀ m : ℤ, 
    s1 m + s2 m > s3 m ∧
    s1 m + s3 m > s2 m ∧
    s2 m + s3 m > s1 m) →
  {m : ℤ | (s1 m + s2 m > s3 m) ∧ (s1 m + s3 m > s2 m) ∧ (s2 m + s3 m > s1 m)}.to_finset.card = 2 :=
begin
  sorry
end

end triangle_inequality_integer_solutions_l21_21680


namespace kimberly_gumballs_last_days_l21_21965

theorem kimberly_gumballs_last_days :
  (let earrings_day1 := 3 in
   let earrings_day2 := 2 * earrings_day1 in
   let earrings_day3 := earrings_day2 - 1 in
   let total_earrings := earrings_day1 + earrings_day2 + earrings_day3 in
   let total_gumballs := 9 * total_earrings in
   let days_last := total_gumballs / 3 in
   days_last = 42) :=
by {
  let earrings_day1 := 3,
  let earrings_day2 := 2 * earrings_day1,
  let earrings_day3 := earrings_day2 - 1,
  let total_earrings := earrings_day1 + earrings_day2 + earrings_day3,
  let total_gumballs := 9 * total_earrings,
  let days_last := total_gumballs / 3,
  exact sorry
}

end kimberly_gumballs_last_days_l21_21965


namespace combined_area_four_removed_triangles_l21_21342

noncomputable def area_equilateral_triangle (s : ℕ) : ℝ :=
  (Math.sqrt 3 / 4) * (s ^ 2)

theorem combined_area_four_removed_triangles :
  let side_length_square : ℕ := 16
  let side_length_triangle : ℕ := 4
  4 * area_equilateral_triangle side_length_triangle = 16 * Math.sqrt 3 :=
by
  sorry

end combined_area_four_removed_triangles_l21_21342


namespace greatest_distance_between_A_and_B_l21_21562

-- Define set A as the solutions to z^5 - 32 = 0 in the complex plane
noncomputable def setA : set ℂ := { z | z^5 = 32 }

-- Define set B as the solutions to z^3 - 16z^2 - 32z + 256 = 0 in the complex plane
noncomputable def setB : set ℂ := { z | z^3 - 16 * z^2 - 32 * z + 256 = 0 }

-- Prove that the greatest distance between a point in A and a point in B is 6
theorem greatest_distance_between_A_and_B :
  ∃ v ∈ setA, ∃ w ∈ setB, 
  ∀ x ∈ setA, ∀ y ∈ setB, complex.abs (v - w) ≥ complex.abs (x - y) ∧ complex.abs (v - w) = 6 :=
sorry

end greatest_distance_between_A_and_B_l21_21562


namespace interval_bounds_l21_21463

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + 13 / 2

def minimum_value (a b : ℝ) : Prop := 
  ∀ x ∈ Icc a b, f x ≥ f a

def maximum_value (a b : ℝ) : Prop := 
  ∀ x ∈ Icc a b, f x ≤ f b

theorem interval_bounds (a b : ℝ) (h_min : f a = 2 * a) (h_max : f b = 2 * b) :
  (minimum_value a b) → (maximum_value a b) → ([a, b] = [1, 3] ∨ [a, b] = [-2 - sqrt 17, 13 / 4]) :=
sorry

end interval_bounds_l21_21463


namespace student_average_greater_l21_21746

theorem student_average_greater (w x y z : ℝ) (h1 : w < x) (h2 : x < y) (h3 : y < z) :
  (let A := (w + x + y + z) / 4 in
   let B := (((w + x) / 2 + y) / 2 + z) / 2 in
   B > A) :=
sorry

end student_average_greater_l21_21746


namespace rms_voltage_is_77_06_l21_21561

noncomputable def rms_voltage (U R R1 : ℝ) : ℝ :=
  let ω := 1 -- This is a placeholder. Actual ω is irrelevant for RMS.
  let f (t : ℝ) := 25 * (3 + Real.sin (ω * t)) -- U_p
  let T := 2 * Real.pi / ω -- Period of the oscillation
  sqrt ((1 / T) * ∫ t in 0..T, (f t) ^ 2)

theorem rms_voltage_is_77_06 :
  rms_voltage 100 1000 1000 = 77.06 := by
  -- Definitions and actual proof can follow.
  -- Integrals and square roots are handled using appropriate mathlib tools.
  sorry

end rms_voltage_is_77_06_l21_21561


namespace evaluate_expression_l21_21395

theorem evaluate_expression : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := 
by
  sorry

end evaluate_expression_l21_21395


namespace find_p_l21_21716

theorem find_p (m n p : ℝ)
  (h1 : m = 4 * n + 5)
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 :=
sorry

end find_p_l21_21716


namespace minimal_sum_proof_l21_21563

def distinct_digits (a b c d e f g h : ℕ) := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h

def digit_constraints (T I X O G R S P : ℕ) := 
  T ≠ 0 ∧ I ≠ 0 ∧ X ≠ 0 ∧ O ≠ 0 ∧ G ≠ 0 ∧ R ≠ 0 ∧ S ≠ 0 ∧ P ≠ 0

noncomputable def TIXO (T I X O : ℕ) := 1000 * T + 100 * I + 10 * X + O
noncomputable def TIGR (T I G R : ℕ) := 1000 * T + 100 * I + 10 * G + R
noncomputable def SPIT (S P I T : ℕ) := 1000 * S + 100 * P + 10 * I + T

theorem minimal_sum_proof (T I X O G R S P : ℕ) (H_distinct : distinct_digits T I X O G R S P)
  (H_constraints: digit_constraints T I X O G R S P) :
  TIXO T I X O + TIGR T I G R = SPIT S P I T :=
by
  have H1 : T = 1 := rfl
  have H2 : I = 3 := rfl
  have H3 : X = 8 := rfl
  have H4 : O = 6 := rfl
  have H5 : G = 4 := rfl
  have H6 : R = 5 := rfl
  have H7 : S = 2 := rfl
  have H8 : P = 7 := rfl
  -- expected sum
  show TIXO 1 3 8 6 + TIGR 1 3 4 5 = SPIT 2 7 3 1 := by
    -- explicit calculation check
    exact eq.refl 2731

end minimal_sum_proof_l21_21563


namespace smallest_addition_to_divisible_by_13_l21_21701

theorem smallest_addition_to_divisible_by_13:
  ∃ (x : ℕ), (913475821 + x) % 13 = 0 ∧ (∀ y, y < x → (913475821 + y) % 13 ≠ 0) :=
begin
  use 2,
  split,
  {
    -- Proof that 913475821 + 2 is exactly divisible by 13
    sorry
  },
  {
    -- Proof that no smaller number than 2 will work
    intros y h,
    -- Show (913475821 + y) % 13 ≠ 0 for y < 2
    sorry
  }

end smallest_addition_to_divisible_by_13_l21_21701


namespace juniors_score_90_l21_21928

open Classical

-- Define the conditions
variable (n : ℕ)  -- total number of students
variable (h1 : 0.2 * n = (20 / 100) * n)
variable (h2 : 0.8 * n = (80 / 100) * n)
variable (h3 : ∀ m : ℕ, 86 * n = m)  -- total score of all students
variable (h4 : ∀ p : ℕ, 85 * (0.8 * n) = p)  -- total score of seniors

-- This theorem states that the juniors' average score is 90
theorem juniors_score_90 (h_avg_class : 86) 
                         (h_avg_seniors : 85) 
                         (h_total_students : n > 0) : 
                         let juniors_count := 0.2 * n
                         let seniors_count := 0.8 * n
                         let total_score := h_avg_class * n
                         let seniors_score := h_avg_seniors * seniors_count
                         let juniors_score := total_score - seniors_score
                         let juniors_avg := juniors_score / juniors_count
                         in juniors_avg = 90 := by sorry

end juniors_score_90_l21_21928


namespace prob_first_odd_second_even_is_half_l21_21418

noncomputable def is_odd (n : ℕ) : Prop := n % 2 = 1
noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

def cards := [1, 2, 3, 4, 5]

def valid_pairs := (finset.product cards.to_finset cards.to_finset).filter (λ p, p.1 ≠ p.2)

def prob_first_odd_second_even : ℝ :=
  let pairs_odd_even := valid_pairs.filter (λ p, is_odd p.1 ∧ is_even p.2) in
  (pairs_odd_even.card : ℝ) / (valid_pairs.card : ℝ)

theorem prob_first_odd_second_even_is_half : prob_first_odd_second_even = 1 / 2 :=
sorry

end prob_first_odd_second_even_is_half_l21_21418


namespace proposition_B_proposition_C_l21_21085

variable (a b c d : ℝ)

-- Proposition B: If |a| > |b|, then a² > b²
theorem proposition_B (h : |a| > |b|) : a^2 > b^2 :=
sorry

-- Proposition C: If (a - b)c² > 0, then a > b
theorem proposition_C (h : (a - b) * c^2 > 0) : a > b :=
sorry

end proposition_B_proposition_C_l21_21085


namespace meal_problem_solution_l21_21686

open Nat

-- Definitions based on the conditions in part a)

-- Twelve people situation
def num_people : ℕ := 12

-- Numbers of each meal type
def num_meals : ℕ := 4
def num_each_meal : ℕ := 3

-- Given the two people received their correct meal, it is related to derangements of remaining
def num_correct : ℕ := 2

-- The derangements of 10 people (!10)
def derangements_10 : ℕ := 1334961

-- The total number of ways the waiter can serve the meal types
def total_ways : ℕ :=
  nat.choose num_people num_correct * derangements_10

-- Proving the value total_ways
theorem meal_problem_solution : total_ways = 88047666 :=
by
  -- Use nat.choose as combinatorial selection and derangements calculation.
  have h : nat.choose num_people num_correct = 66 := by norm_num
  have derangements_eq : derangements_10 = 1334961 := by norm_num
  unfold total_ways
  rw [h, derangements_eq]
  norm_num
  exact rfl

end meal_problem_solution_l21_21686


namespace digit_change_not_prime_l21_21573

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the numbers we are considering
def num1 := factorial 10 -- 10!
def num2 := (factorial 10) ^ 3 -- (10!)^3
def num3 := (factorial 19) + 10 -- 19! + 10

-- A digit-changing function to apply digit modifications
def change_digit (n : ℕ) (pos : ℕ) (new_digit : ℕ) : ℕ := sorry

-- A helper function to decide if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Formal statement to show the change in individual digits does not form a prime number
theorem digit_change_not_prime :
  (∀ (n pos new_digit : ℕ), 
    (n = num1 ∨ n = num2 ∨ n = num3) → 
    pos < Nat.log10 n + 1 → 
    0 < new_digit < 10 → 
    ¬ is_prime (change_digit n pos new_digit)) :=
  sorry

end digit_change_not_prime_l21_21573


namespace fraction_of_compositions_l21_21168

theorem fraction_of_compositions :
  let f := λ x : ℤ, 3 * x + 4
  let g := λ x : ℤ, 2 * x - 1
  f (g (f 3)) / g (f (g 3)) = 79 / 37 :=
by
  sorry

end fraction_of_compositions_l21_21168


namespace boris_needs_to_climb_four_times_l21_21512

/-- Hugo's mountain elevation is 10,000 feet above sea level. --/
def hugo_mountain_elevation : ℕ := 10_000

/-- Boris' mountain is 2,500 feet shorter than Hugo's mountain. --/
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2_500

/-- Hugo climbed his mountain 3 times. --/
def hugo_climbs : ℕ := 3

/-- The total number of feet Hugo climbed. --/
def total_hugo_climb : ℕ := hugo_mountain_elevation * hugo_climbs

/-- The number of times Boris needs to climb his mountain to equal Hugo's climb. --/
def boris_climbs_needed : ℕ := total_hugo_climb / boris_mountain_elevation

theorem boris_needs_to_climb_four_times :
  boris_climbs_needed = 4 :=
by
  sorry

end boris_needs_to_climb_four_times_l21_21512


namespace messenger_speed_l21_21347

noncomputable def team_length : ℝ := 6

noncomputable def team_speed : ℝ := 5

noncomputable def total_time : ℝ := 0.5

theorem messenger_speed (x : ℝ) :
  (6 / (x + team_speed) + 6 / (x - team_speed) = total_time) →
  x = 25 := by
  sorry

end messenger_speed_l21_21347


namespace expansion_terms_count_l21_21511

theorem expansion_terms_count (a b c d e f g h : ℤ) :
  let expr1 := a + b + c,
      expr2 := d + e + f + g + h in
  ∀ (s t : Set ℤ), s = {a, b, c} → t = {d, e, f, g, h} →
  s.card = 3 ∧ t.card = 5 →
  ((λ x y, x * y) '' s ×ˢ t).card = 15 :=
by
  sorry

end expansion_terms_count_l21_21511


namespace find_numDotNoLine_l21_21927

-- Definitions according to the given conditions
def numDotAndLine : ℕ := 9
def numLineNoDot : ℕ := 24
def totalLetters : ℕ := 40

-- The number of letters containing a dot but not a straight line
def numDotNoLine : ℕ := totalLetters - numDotAndLine - numLineNoDot

-- Proof statement
theorem find_numDotNoLine : numDotNoLine = 7 := by
  -- Definition combined in the 3rd step of problem statement
  have h : numDotNoLine = 7 := by
    simp [numDotNoLine, totalLetters, numDotAndLine, numLineNoDot]
    exact rfl
  exact h

end find_numDotNoLine_l21_21927


namespace sum_base6_numbers_l21_21828

def base6_to_nat (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6^1 + d

theorem sum_base6_numbers :
  let n1 := base6_to_nat 1 2 3 4,
      n2 := base6_to_nat 0 2 3 4,
      n3 := base6_to_nat 0 0 3 4,
      sum_base10 := n1 + n2 + n3 in
  ∃ a b c d : ℕ, sum_base10 = base6_to_nat a b c d ∧ (a = 2 ∧ b = 5 ∧ c = 5 ∧ d = 0) :=
by sorry

end sum_base6_numbers_l21_21828


namespace minimum_crooks_l21_21550

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l21_21550


namespace dave_initial_boxes_l21_21003

def pieces_per_box : ℕ := 3
def boxes_given_away : ℕ := 5
def pieces_left : ℕ := 21
def total_pieces_given_away := boxes_given_away * pieces_per_box
def total_pieces_initially := total_pieces_given_away + pieces_left

theorem dave_initial_boxes : total_pieces_initially / pieces_per_box = 12 := by
  sorry

end dave_initial_boxes_l21_21003


namespace boris_climbs_needed_l21_21514

-- Definitions
def elevation_hugo : ℕ := 10000
def shorter_difference : ℕ := 2500
def climbs_hugo : ℕ := 3

-- Derived Definitions
def elevation_boris : ℕ := elevation_hugo - shorter_difference
def total_climbed_hugo : ℕ := climbs_hugo * elevation_hugo

-- Theorem
theorem boris_climbs_needed : (total_climbed_hugo / elevation_boris) = 4 :=
by
  -- conditions and definitions are used here
  sorry

end boris_climbs_needed_l21_21514


namespace max_friends_l21_21530

-- Defining friendship in the context of our problem
variable {Person : Type}

-- Symmetric friendship relation
def Friends (a b : Person) : Prop :=
  a ≠ b ∧ a = b

-- Assumption: Any m persons have a unique common friend
axiom unique_common_friend {m : ℕ} (h : m ≥ 3) (s : Finset Person) (hs : s.card = m) :
  ∃! f : Person, ∀ x ∈ s, Friends f x

-- Proposition to prove: The maximum number of friends any person can have in the carriage is exactly m
theorem max_friends (m : ℕ) (h : m ≥ 3) : 
  ∃ p : Person, (card {q : Person | Friends p q}) = m :=
sorry -- Proof to be provided

end max_friends_l21_21530


namespace min_dist_l21_21941

open Real

theorem min_dist (a b : ℝ) :
  let A := (0, -1)
  let B := (1, 3)
  let C := (2, 6)
  let D := (0, b)
  let E := (1, a + b)
  let F := (2, 2 * a + b)
  let AD_sq := (b + 1) ^ 2
  let BE_sq := (a + b - 3) ^ 2
  let CF_sq := (2 * a + b - 6) ^ 2
  AD_sq + BE_sq + CF_sq = (b + 1) ^ 2 + (a + b - 3) ^ 2 + (2 * a + b - 6) ^ 2 → 
  a = 7 / 2 ∧ b = -5 / 6 :=
sorry

end min_dist_l21_21941


namespace ratio_of_selling_prices_l21_21748

theorem ratio_of_selling_prices (C SP1 SP2 : ℝ)
  (h1 : SP1 = C + 0.20 * C)
  (h2 : SP2 = C + 1.40 * C) :
  SP2 / SP1 = 2 := by
  sorry

end ratio_of_selling_prices_l21_21748


namespace final_amount_is_212_l21_21010

-- Definitions based on conditions
def earl_initial := 90
def fred_initial := 48
def greg_initial := 36
def hannah_initial := 72

def earl_owes_fred := 28
def earl_owes_hannah := 30

def fred_owes_greg := 32
def fred_owes_hannah := 10

def greg_owes_earl := 40
def greg_owes_hannah := 20

def hannah_owes_greg := 15
def hannah_owes_earl := 25

-- Final amounts after settling debts
def earl_final := earl_initial - earl_owes_fred - earl_owes_hannah + greg_owes_earl + hannah_owes_earl
def fred_final := fred_initial + earl_owes_fred - fred_owes_greg - fred_owes_hannah
def greg_final := greg_initial + fred_owes_greg - greg_owes_earl + hannah_owes_greg - greg_owes_hannah
def hannah_final := hannah_initial + earl_owes_hannah + fred_owes_hannah - hannah_owes_greg - hannah_owes_earl + greg_owes_hannah

-- Total amount after debts are settled.
def total_final := greg_final + earl_final + hannah_final

theorem final_amount_is_212 : total_final = 212 :=
by
  -- capture the final amounts in separate variables to clarify the proof process.
  have h1: earl_final = 97, sorry,
  have h2: greg_final = 23, sorry,
  have h3: hannah_final = 92, sorry,
  have h_total : total_final = earl_final + greg_final + hannah_final := rfl, -- by definition
  rw [h1, h2, h3] at h_total,
  norm_num at h_total,
  exact h_total

end final_amount_is_212_l21_21010


namespace perfect_squares_count_l21_21509

theorem perfect_squares_count : (finset.filter (λ n, n * n ≥ 100 ∧ n * n ≤ 500) (finset.range 23)).card = 13 :=
by
  sorry

end perfect_squares_count_l21_21509


namespace sum_convex_polygon_valid_n_l21_21268

theorem sum_convex_polygon_valid_n :
  let valid_n_values := {n : ℕ | ∃ (a d : ℕ), 
                            a > 0 ∧ d > 0 ∧ 
                            (∀ i j : ℕ, i < j ∧ j < n → 
                              (a + i * d) < (a + j * d) ∧ 
                              (n - 2) * 180 ∣ (a + (n - 1) * d - a)) ∧
                            (∀ i : ℕ, i < n → a + i * d < 180)} 
  in ∑ n in valid_n_values, n = 106 :=
by
  -- The proof would go here
  sorry

end sum_convex_polygon_valid_n_l21_21268


namespace a_2_value_a_3_value_a_4_value_geometric_progression_general_formula_l21_21067

open Nat

def a_sequence : ℕ → ℤ
| 0       := 2 -- Lean typically starts sequences at 0, so we define a_0.
| (n + 1) := 2 * a_sequence n + 3

theorem a_2_value : a_sequence 1 = 7 := sorry
theorem a_3_value : a_sequence 2 = 17 := sorry
theorem a_4_value : a_sequence 3 = 37 := sorry

theorem geometric_progression (n : ℕ) : (a_sequence n) + 3 = 5 * 2^n := sorry

theorem general_formula (n : ℕ) : a_sequence n = 5 * 2^n - 3 := sorry

end a_2_value_a_3_value_a_4_value_geometric_progression_general_formula_l21_21067


namespace value_of_expression_l21_21086

theorem value_of_expression (m n : ℝ) (h : m + n = 3) :
  2 * m^2 + 4 * m * n + 2 * n^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l21_21086


namespace function_max_value_l21_21725

theorem function_max_value (a b : ℝ) :
  (∀ x : ℝ, f x = ax^2 + bx + 3a + b ∧ x ∈ set.Icc (a-1) (2a) ∧ 
           ∀ x : ℝ, f x = f (-x)) →
  (b = 0 ∧ a = 1/3) →
  (∃ x : ℝ, x ∈ set.Icc (-2/3) (2/3) ∧ f x = 31/27) :=
begin
  intros h1 h2,
  sorry
end

end function_max_value_l21_21725


namespace identify_coefficients_l21_21517

theorem identify_coefficients (m n : ℤ) :
  (∀ x : ℤ, (x + 1) * (2 * x - 3) = 2 * x^2 + m * x + n) →
  (m = -1 ∧ n = -3) :=
by
  intro h
  have : (x + 1) * (2 * x - 3) = 2 * x^2 - x - 3 := by sorry
  have eq_m : m = -1 := by sorry
  have eq_n : n = -3 := by sorry
  exact ⟨eq_m, eq_n⟩

end identify_coefficients_l21_21517


namespace least_possible_b_l21_21638

noncomputable def a : ℕ := 8

theorem least_possible_b (b : ℕ) (h1 : ∀ n : ℕ, n > 0 → a.factors.count n = 1 → a = n^3)
  (h2 : b.factors.count a = 1)
  (h3 : b % a = 0) :
  b = 24 :=
sorry

end least_possible_b_l21_21638


namespace repeating_decimal_equation_l21_21973

variable {D P Q R : ℚ}
variable {r s : ℕ}
def repeating_decimal (D : ℚ) (P Q : ℕ) (r s : ℕ) : Prop :=
  D = P / 10^r + Q / (10^r * (10^s - 1))

theorem repeating_decimal_equation (h : repeating_decimal D P Q r s) :
  (10^s - 1) * 10^(r+s) * D = R * 10^(r+s) :=
sorry

end repeating_decimal_equation_l21_21973


namespace g_ge_f_range_max_g_minus_f_l21_21464

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) : ℝ := -x^2 + 6*x - 5

theorem g_ge_f_range (x : ℝ) : g(x) ≥ f(x) ↔ 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

theorem max_g_minus_f : ∀ x ∈ set.Icc (1 : ℝ) (4 : ℝ), (-x^2 + 6*x - 5) - |x - 1| ≤ 9/4 :=
by
  sorry

end g_ge_f_range_max_g_minus_f_l21_21464


namespace range_of_a_l21_21469

variable {α : Type*} [LinearOrderedField α]

def setA (a : α) : Set α := {x | abs (x - a) < 1}
def setB : Set α := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : α) (h : setA a ∩ setB = ∅) : a ≤ 0 ∨ a ≥ 6 :=
sorry

end range_of_a_l21_21469


namespace range_yield_improved_l21_21717

theorem range_yield_improved (H L : ℝ) (range_last_year : H - L = 10000) (improvement : ∀ x, x' = x * 1.15) : 
  let range_this_year := (1.15 * H) - (1.15 * L) in
  range_this_year = 11500 :=
by
  sorry

end range_yield_improved_l21_21717


namespace range_F_of_range_f_l21_21054

def range_f : Set ℝ := Set.Icc (1 / 2) 3

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 / f x

theorem range_F_of_range_f (f : ℝ → ℝ) (hf : ∀ x, f x ∈ range_f) : 
  ∃ a b, (∀ x, F f x ∈ Set.Icc a b) ∧ a = 2 ∧ b = 10 / 3 := by
  sorry

end range_F_of_range_f_l21_21054


namespace length_CD_l21_21102

variable {O A C D : Point}
variable {r t s x : ℝ}
variable (circle : Circle O r)
variable (tangent : Tangent O A t)
variable (secant : Secant A C D)

axiom tangent_length : ∀ {O A : Point} {t : ℝ}, Tangent O A t → // specific details about tangent here
axiom secant_length : ∀ {A C D : Point} {s x : ℝ}, Secant A C D ∧ AC = s → CD = x → // specific details about secant here

theorem length_CD (h : tangent_length tangent ∧ secant_length secant AC s) :
  ∃ x : ℝ, x = (t^2 - s^2) / s := sorry

end length_CD_l21_21102


namespace integral_inequality_l21_21470

-- statement of the problem transformed into Lean
theorem integral_inequality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Icc 0 1) 
  (hg : ∀ x, g x ∈ Icc 0 1) 
  (Hf : ∀ x y, x ≤ y → f x ≤ f y) 
  (Hfc : ContinuousOn f (Icc 0 1)) 
  (Hgc : ContinuousOn g (Icc 0 1)) : 
  ∫ x in 0..1, f (g x) ≤ ∫ x in 0..1, f x + ∫ x in 0..1, g x :=
by
  sorry

end integral_inequality_l21_21470


namespace isosceles_triangle_perimeter_l21_21114

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l21_21114


namespace top_card_is_ace_hearts_probability_l21_21331

def probability_top_card_is_ace_hearts (total_cards : ℕ) (ace_of_hearts_count : ℕ) (randomly_shuffled : Prop) : ℚ :=
  if randomly_shuffled then (ace_of_hearts_count : ℚ) / (total_cards : ℚ) else 0

theorem top_card_is_ace_hearts_probability :
  ∀ (total_cards ace_of_hearts_count : ℕ) (randomly_shuffled : Prop),
  total_cards = 54 →
  ace_of_hearts_count = 1 →
  randomly_shuffled →
  probability_top_card_is_ace_hearts total_cards ace_of_hearts_count randomly_shuffled = 1 / 54 :=
by sorry

end top_card_is_ace_hearts_probability_l21_21331


namespace greatest_odd_factors_under_150_l21_21199

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l21_21199


namespace nature_of_singularity_at_1_l21_21385

open Complex

def f (z : ℂ) : ℂ := (z - 1) * exp (1 / (z - 1))

theorem nature_of_singularity_at_1 : 
  ∃ g : ℂ → ℂ, is_essential_singularity (g) 1 :=
begin
  use f,
  sorry
end

end nature_of_singularity_at_1_l21_21385


namespace parallel_line_l21_21034

variables (m : Set Point) (α β : Set Plane)

-- Define the conditions
def is_parallel (x y : Set Type) : Prop := sorry
def is_subset (x y : Set Type) : Prop := sorry

-- Condition ②: α is parallel to β
axiom α_parallel_β : is_parallel α β 

-- Condition ③: m is a subset of β
axiom m_subset_β : is_subset m β  

-- Prove that if α is parallel to β and m is a subset of β, then m is parallel to α
theorem parallel_line : is_parallel α β → is_subset m β → is_parallel m α :=
by 
  intros h1 h2 
  sorry

end parallel_line_l21_21034


namespace number_of_valid_paths_l21_21377

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_valid_paths (n : ℕ) :
  let valid_paths := binomial (2 * n) n / (n + 1)
  valid_paths = binomial (2 * n) n - binomial (2 * n) (n + 1) := 
sorry

end number_of_valid_paths_l21_21377


namespace not_divisible_by_1998_l21_21323

theorem not_divisible_by_1998 (n : ℕ) :
  ∀ k : ℕ, ¬ (2^(k+1) * n + 2^k - 1) % 2 = 0 → ¬ (2^(k+1) * n + 2^k - 1) % 1998 = 0 :=
by
  intros _ _
  sorry

end not_divisible_by_1998_l21_21323


namespace find_polynomial_l21_21406

def polynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

theorem find_polynomial
  (a b c : ℚ)
  (h1 : polynomial a b c (-3) = 0)
  (h2 : polynomial a b c 6 = 0)
  (h3 : polynomial a b c 2 = -24) :
  a = 6/5 ∧ b = -18/5 ∧ c = -108/5 :=
by 
  sorry

end find_polynomial_l21_21406


namespace segments_intersect_with_ratio_3_2_midpoints_form_parallelogram_area_ratio_parallelogram_base_l21_21661

open_locale classical

namespace QuadrilateralPyramid

-- Definitions of basic elements and conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ) 
structure Pyramid := 
(base_midpoints : list Point)
(medians_intersection_points : list Point)

-- The main theorem proving the given conditions and required properties
theorem segments_intersect_with_ratio_3_2 (p : Pyramid) :
  ∃ (intersection_point : Point), 
  (∀ seg ∈ p.base_midpoints.zip p.medians_intersection_points, 
    seg ∩ intersection_point ≠ ∅ ∧ 
    segment_divided_ratio seg intersection_point = (3:ℝ) / (2:ℝ)) :=
sorry

theorem midpoints_form_parallelogram (p : Pyramid):
  let midpoints_segments := (p.base_midpoints.zip p.medians_intersection_points).map midpoint_segment in
  is_parallelogram(midpoints_segments) :=
sorry

theorem area_ratio_parallelogram_base (p : Pyramid):
  let midpoints_segments := (p.base_midpoints.zip p.medians_intersection_points).map midpoint_segment in
  let parallelogram_area := area_of_parallelogram(midpoints_segments) in
  let base_area := area_of_polygon(p.base_midpoints) in
  parallelogram_area / base_area = (1:ℝ) / (72:ℝ) :=
sorry

end QuadrilateralPyramid

end segments_intersect_with_ratio_3_2_midpoints_form_parallelogram_area_ratio_parallelogram_base_l21_21661


namespace total_cost_one_pizza_and_three_burgers_l21_21106

def burger_cost : ℕ := 9
def pizza_cost : ℕ := burger_cost * 2
def total_cost : ℕ := pizza_cost + (burger_cost * 3)

theorem total_cost_one_pizza_and_three_burgers :
  total_cost = 45 :=
by
  rw [total_cost, pizza_cost, burger_cost]
  norm_num

end total_cost_one_pizza_and_three_burgers_l21_21106


namespace number_of_integers_between_cubes_l21_21479

theorem number_of_integers_between_cubes :
  let a := 10.4
  let b := 10.5
  let lower_bound := a ^ 3
  let upper_bound := b ^ 3
  let start := Int.ceil lower_bound
  let end_ := Int.floor upper_bound
  end_ - start + 1 = 33 :=
by
  have h1 : lower_bound = 1124.864 := by sorry
  have h2 : upper_bound = 1157.625 := by sorry
  have h3 : start = 1125 := by sorry
  have h4 : end_ = 1157 := by sorry
  sorry

end number_of_integers_between_cubes_l21_21479


namespace lucie_can_achieve_final_state_l21_21609

def final_token_count_in_B1 (n : ℕ) : ℕ :=
if n = 0 then 1 else 2019^(final_token_count_in_B1 (n - 1))

def initial_tokens := (1, 1, 1, 1, 1, 1, 1 : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)

-- Definition of the two operations
def operation1 (config : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) (k : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
let ⟨a, b, c, d, e, f, g⟩ := config in
if k = 1 then (a + 2, b - 1, c, d, e, f, g)
else if k = 2 then (a, b + 2, c - 1, d, e, f, g)
else if k = 3 then (a, b, c + 2, d - 1, e, f, g)
else if k = 4 then (a, b, c, d + 2, e - 1, f, g)
else if k = 5 then (a, b, c, d, e + 2, f - 1, g)
else (a, b, c, d, e, f + 2, g - 1) -- k = 6 

def operation2 (config : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) (k : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
let ⟨a, b, c, d, e, f, g⟩ := config in
if k = 1 then (c, a, b - 1, d, e, f, g)
else if k = 2 then (a, d, b, c - 1, e, f, g)
else if k = 3 then (a, b, e, c, d - 1, f, g)
else if k = 4 then (a, b, c, f, d, e - 1, g)
else (a, b, c, d, g, e, f - 1) -- k = 5

theorem lucie_can_achieve_final_state :
  ∃ (N : ℕ) (final_config : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ),
    (∀ i : ℕ, i ∈ [1..N] →
       ∃ (k : ℕ),
         (k ≤ 6 ∧ (final_config = operation1 final_config k) ∨
          k ≤ 5 ∧ (final_config = operation2 final_config k))) ∧
    final_config.1 = final_token_count_in_B1 13 ∧
    final_config.2 = 0 ∧ final_config.3 = 0 ∧ final_config.4 = 0 ∧
    final_config.5 = 0 ∧ final_config.6 = 0 ∧ final_config.7 = 0 :=
sorry

end lucie_can_achieve_final_state_l21_21609


namespace mutuallyExclusiveButNotContradictoryEvents_l21_21834

noncomputable def mutuallyExclusiveButNotContradictoryEvents : Prop :=
  let events (draws : List (String × String)) : List (Set (String × String)) :=
    [
      {x | x ≠ ("red", "red") ∧ x ≠ ("white", "white")},
      {x | x = ("white", "white")},
      {x | x ≠ ("white", "white")},
      {x | x = ("red", "red")},
      {x | x = ("red", "white") ∨ x = ("white", "red")},
      {x | x ≠ ("red", "white") ∧ x ≠ ("white", "white") ∧ x ≠ ("red", "red")}
    ]
  let cond1 := (∀ (e₁ e₂ : Set (String × String)), e₁ ∈ events ["white", "red"] ∧ e₂ ∈ events ["white", "red"] → Set.disjoint e₁ e₂)
  let cond2 := ¬ (∀ draws, cond1)
  let cond3 := ¬ (events ["at least one white ball", "both are white balls"] ≠ ∅ ∧ events ["exactly one white ball", "exactly two white balls"] ≠ ∅)
  local noncomputable def problem1 : ∀ (events : List (Set (String × String))), 
      ((("exactly one white ball", "exactly two white balls") ∈ events) →
      cond1 → cond2 → cond3)

#print problem1

theorem mutuallyExclusiveButNotContradictoryEvents 
  (draw : List (String × String)) :
  draw = [("red", "red"), ("red", "white"), ("red", "white"), ("white", "white")] →
  ∃ (e₁ e₂ : Set (String × String)), e₁ = {("red", "white"), ("white", "red")} ∧ 
  e₂ = {("white", "white")} ∧ 
  Set.disjoint e₁ e₂
:=
sorry

end mutuallyExclusiveButNotContradictoryEvents_l21_21834


namespace count_sum_coprime_15_l21_21803

theorem count_sum_coprime_15 :
  let coprime_with_15 := {a | 1 ≤ a ∧ a < 15 ∧ Int.gcd a 15 = 1},
      count := (coprime_with_15.toFinset.card : ℕ),
      sum := (coprime_with_15.toFinset.sum : ℕ)
  in count = 8 ∧ sum = 60 :=
by
  sorry

end count_sum_coprime_15_l21_21803


namespace double_binom_6_2_l21_21792

theorem double_binom_6_2 : 2 * Nat.choose 6 2 = 30 := by
  sorry

end double_binom_6_2_l21_21792


namespace plane_B_overtakes_plane_A_in_120_minutes_l21_21305

-- Definitions of the problem's conditions.
def plane_A_speed : ℝ := 200  -- in mph
def plane_B_speed : ℝ := 300  -- in mph
def head_start : ℝ := 40 / 60  -- in hours

-- Calculate the total time for Plane B to overtake plane A from when Plane A took off.
theorem plane_B_overtakes_plane_A_in_120_minutes :
  let t := (head_start * plane_A_speed) / (plane_B_speed - plane_A_speed) in
  t * 60 + head_start * 60 = 120 := by
  sorry

end plane_B_overtakes_plane_A_in_120_minutes_l21_21305


namespace find_cot_difference_l21_21140

-- Define necessary elements for the problem
variable {A B C D : Type}
variable [EuclideanGeometry A]
variables (ABC : Triangle A B C)

-- Define the condition where median AD makes an angle of 60 degrees with BC
variable (ADmedian : median A B C D ∧ angle D A B = 60)

theorem find_cot_difference:
  |cot (angle B) - cot (angle C)| = 2 :=
sorry

end find_cot_difference_l21_21140


namespace number_of_passed_candidates_l21_21246

theorem number_of_passed_candidates :
  ∀ (P F : ℕ),
  (P + F = 500) →
  (P * 80 + F * 15 = 500 * 60) →
  P = 346 :=
by
  intros P F h1 h2
  sorry

end number_of_passed_candidates_l21_21246


namespace kimberly_gumballs_last_days_l21_21964

theorem kimberly_gumballs_last_days :
  (let earrings_day1 := 3 in
   let earrings_day2 := 2 * earrings_day1 in
   let earrings_day3 := earrings_day2 - 1 in
   let total_earrings := earrings_day1 + earrings_day2 + earrings_day3 in
   let total_gumballs := 9 * total_earrings in
   let days_last := total_gumballs / 3 in
   days_last = 42) :=
by {
  let earrings_day1 := 3,
  let earrings_day2 := 2 * earrings_day1,
  let earrings_day3 := earrings_day2 - 1,
  let total_earrings := earrings_day1 + earrings_day2 + earrings_day3,
  let total_gumballs := 9 * total_earrings,
  let days_last := total_gumballs / 3,
  exact sorry
}

end kimberly_gumballs_last_days_l21_21964


namespace problem_statement_l21_21519

variable (a : ℝ)

theorem problem_statement (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := 
by sorry

end problem_statement_l21_21519


namespace cheaper_to_buy_more_cheaper_2_values_l21_21078

def cost_function (n : ℕ) : ℕ :=
  if (1 ≤ n ∧ n ≤ 30) then 15 * n - 20
  else if (31 ≤ n ∧ n ≤ 55) then 14 * n
  else if (56 ≤ n) then 13 * n + 10
  else 0  -- Assuming 0 for n < 1 as it shouldn't happen in this context

theorem cheaper_to_buy_more_cheaper_2_values : 
  ∃ n1 n2 : ℕ, n1 < n2 ∧ cost_function (n1 + 1) < cost_function n1 ∧ cost_function (n2 + 1) < cost_function n2 ∧
  ∀ n : ℕ, (cost_function (n + 1) < cost_function n → n = n1 ∨ n = n2) := 
sorry

end cheaper_to_buy_more_cheaper_2_values_l21_21078


namespace increase_in_volume_l21_21719

-- Define the volume of the right circular cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Given that the original volume is 6 liters
def original_volume : ℝ := 6

-- The increase in volume when the radius is doubled
theorem increase_in_volume (r h : ℝ) (hV : volume r h = original_volume) :
  volume (2 * r) h - volume r h = 18 :=
by
  sorry

end increase_in_volume_l21_21719


namespace winning_percentage_votes_l21_21935

theorem winning_percentage_votes (P : ℝ) (votes_total : ℝ) (majority_votes : ℝ) (winning_votes : ℝ) : 
  votes_total = 4500 → majority_votes = 900 → 
  winning_votes = (P / 100) * votes_total → 
  majority_votes = winning_votes - ((100 - P) / 100) * votes_total → P = 60 := 
by
  intros h_total h_majority h_winning_votes h_majority_eq
  sorry

end winning_percentage_votes_l21_21935


namespace range_of_a1_l21_21124

noncomputable def geometric_sequence_cond (a_1 : ℝ) (q : ℝ) : Prop :=
  ∃ (S_n : ℕ → ℝ), (S_n = λ n, a_1 * (1 - q^n) / (1 - q)) ∧ (tendsto S_n at_top (𝓝 (1 / a_1)))

theorem range_of_a1 {a_1 q : ℝ} (h1 : a_1 > 1) (h2 : abs q < 1)
  (h3 : geometric_sequence_cond a_1 q) : 1 < a_1 ∧ a_1 < sqrt 2 :=
by sorry

end range_of_a1_l21_21124


namespace max_sum_of_bn_l21_21049

noncomputable theory

open_locale big_operators

-- Definitions based on conditions:
def geometric_sequence (a : ℕ → ℝ) := ∀ n m, a n / a m = a (n - m)  -- positive sequence not equal to 1
def b (a : ℕ → ℝ) (n : ℕ) := real.log 10 (a n)  -- b_n = log_10(a_n)

-- Given conditions
variables {a : ℕ → ℝ} (h_geom : geometric_sequence a) (h_pos : ∀ n, 0 < a n) (h_not_one : ∀ n, a n ≠ 1)
          (h_b3 : b a 3 = 18) (h_b6 : b a 6 = 12)

-- Question to prove
theorem max_sum_of_bn : ∃ n : ℕ, ∑ i in finset.range n, b a i = 132 :=
sorry

end max_sum_of_bn_l21_21049


namespace isosceles_triangle_perimeter_l21_21116

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l21_21116


namespace base_prime_rep_441_l21_21820

-- introducing the necessary structure and definitions.
def prime_exp (n : ℕ) (p : ℕ) : ℕ := 
if p ∣ n then multiplicity (p : ℕ) n else 0

theorem base_prime_rep_441 : prime_exp 441 2 = 0 ∧ prime_exp 441 3 = 2 ∧ prime_exp 441 5 = 0 ∧ prime_exp 441 7 = 2 :=
by
  sorry

end base_prime_rep_441_l21_21820


namespace even_of_even_square_sqrt_two_irrational_l21_21313

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l21_21313


namespace find_B_find_C_l21_21594

-- Define the problem data
variable {a b c A B C : ℝ}

-- Conditions from the problem
variable (h1 : (a + b + c) * (a - b + c) = a * c)
variable (h2 : sin A * sin C = (sqrt 3 - 1) / 4)

-- Results we want to prove
theorem find_B (h3 : (cos B) = -1 / 2) : B = 120 :=
  by sorry

theorem find_C (h4 : A + C = 60) : C = 15 ∨ C = 45 :=
  by sorry

end find_B_find_C_l21_21594


namespace smallest_positive_option_l21_21387

def option_A : ℝ := 12 - 2 * Real.sqrt 15
def option_B : ℝ := 4 * Real.sqrt 15 - 12
def option_C : ℝ := 23 - 3 * Real.sqrt 23
def option_D : ℝ := 60 - 8 * Real.sqrt 30
def option_E : ℝ := 8 * Real.sqrt 30 - 60

theorem smallest_positive_option:
  (0 < option_B) ∧ 
  (option_B < option_A) ∧ 
  (option_B < option_C) ∧ 
  (option_B < option_D) := 
by
  -- Proof for the theorem goes here
  sorry

end smallest_positive_option_l21_21387


namespace average_people_moving_to_florida_each_hour_l21_21945

theorem average_people_moving_to_florida_each_hour :
  let total_people := 1800
  let days := 4
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  let average_per_hour := total_people / total_hours
  let rounded_average := Int.round (average_per_hour : Real)
  rounded_average = 19 :=
by
  sorry

end average_people_moving_to_florida_each_hour_l21_21945


namespace Victor_worked_hours_l21_21695

theorem Victor_worked_hours (h : ℕ) (pay_rate : ℕ) (total_earnings : ℕ) 
  (H1 : pay_rate = 6) 
  (H2 : total_earnings = 60) 
  (H3 : 2 * (pay_rate * h) = total_earnings): 
  h = 5 := 
by 
  sorry

end Victor_worked_hours_l21_21695


namespace original_ticket_price_l21_21684

-- Lean definitions for conditions described in the problem 
def total_revenue (P : ℕ) : ℕ :=
  (10 * (0.6 * P)) + 
  (20 * (0.85 * P)) + 
  (20 * P)

theorem original_ticket_price (P : ℕ) 
  (h : total_revenue P = 860) : 
  P = 20 :=
by 
  sorry

end original_ticket_price_l21_21684


namespace find_vector_b_l21_21977

def vector_dot (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def vector_cross (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

theorem find_vector_b : 
  let a := (3, 2, 4)
  let b := ( (3 * 28 + 328) / 164,
             ( (28 / 41) + (205 / 82) ) / 2,
             28 / 41) in
  vector_dot a b = 18 ∧ 
  vector_cross a b = (-5, 8, -1) :=
by sorry

end find_vector_b_l21_21977


namespace find_ab_l21_21863

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
sorry

end find_ab_l21_21863


namespace trains_meet_in_2067_seconds_l21_21283

def length_of_train1 : ℝ := 100  -- Length of Train 1 in meters
def length_of_train2 : ℝ := 200  -- Length of Train 2 in meters
def initial_distance : ℝ := 630  -- Initial distance between trains in meters
def speed_of_train1_kmh : ℝ := 90  -- Speed of Train 1 in km/h
def speed_of_train2_kmh : ℝ := 72  -- Speed of Train 2 in km/h

noncomputable def speed_of_train1_ms : ℝ := speed_of_train1_kmh * (1000 / 3600)
noncomputable def speed_of_train2_ms : ℝ := speed_of_train2_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := speed_of_train1_ms + speed_of_train2_ms
noncomputable def total_distance : ℝ := initial_distance + length_of_train1 + length_of_train2
noncomputable def time_to_meet : ℝ := total_distance / relative_speed

theorem trains_meet_in_2067_seconds : time_to_meet = 20.67 := 
by
  sorry

end trains_meet_in_2067_seconds_l21_21283


namespace digit_change_not_prime_l21_21574

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the numbers we are considering
def num1 := factorial 10 -- 10!
def num2 := (factorial 10) ^ 3 -- (10!)^3
def num3 := (factorial 19) + 10 -- 19! + 10

-- A digit-changing function to apply digit modifications
def change_digit (n : ℕ) (pos : ℕ) (new_digit : ℕ) : ℕ := sorry

-- A helper function to decide if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Formal statement to show the change in individual digits does not form a prime number
theorem digit_change_not_prime :
  (∀ (n pos new_digit : ℕ), 
    (n = num1 ∨ n = num2 ∨ n = num3) → 
    pos < Nat.log10 n + 1 → 
    0 < new_digit < 10 → 
    ¬ is_prime (change_digit n pos new_digit)) :=
  sorry

end digit_change_not_prime_l21_21574


namespace tangent_sequence_sum_l21_21263

theorem tangent_sequence_sum :
  let seq : ℕ → ℝ
    | 0     := 16
    | (n+1) := seq n / 2 in
  seq 0 + seq 2 + seq 4 = 21 :=
by
  sorry

end tangent_sequence_sum_l21_21263


namespace problem_incorrect_option_A_l21_21756

-- Define our hypotheses
def two_planes_parallel_to_line {P₁ P₂ : Type} (h₁ : P₁ → Prop) (h₂ : P₂ → Prop) : Prop :=
  ∃ L : Type, ∀ p₁ p₂ : Type, h₁ p₁ → h₂ p₂ → (parallel P₁ L ∧ parallel P₂ L) → (parallel P₁ P₂ ∨ intersect P₁ P₂)

def two_planes_parallel_to_plane {P₁ P₂ : Type} (parallel_to : P₁ → P₂ → Prop) : Prop :=
  ∀ P₁ P₂, parallel_to P₁ P₂ → parallel P₁ P₂

def line_intersects_parallel_planes {L P₁ P₂ : Type} (intersects : L → P₁ → Prop) (parallel_planes : P₁ → P₂ → Prop) : Prop :=
  ∀ L P₁ P₂, intersects L P₁ → parallel_planes P₁ P₂ → intersects L P₂

def angles_with_parallel_planes_equal {L P₁ P₂ : Type} (angle : L → P₁ → Prop) (parallel_planes : P₁ → P₂ → Prop) : Prop :=
  ∀ L P₁ P₂, angle L P₁ → parallel_planes P₁ P₂ → angle L P₂

-- Definition of problem
theorem problem_incorrect_option_A
  (P₁ P₂: Type) (L : Type) (h₁ : P₁ → Prop) (h₂ : P₂ → Prop) (parallel_planes : P₁ → P₂ → Prop)
  (parallel : P₁ → L → Prop) (intersects : L → P₁ → Prop) (angle : L → P₁ → Prop)
  (h1 : two_planes_parallel_to_line h₁ h₂)
  (h2 : two_planes_parallel_to_plane parallel_planes)
  (h3 : line_intersects_parallel_planes intersects parallel_planes)
  (h4 : angles_with_parallel_planes_equal angle parallel_planes) :
  ¬ (∀ p₁ p₂ : P₁, parallel p₁ p₂ → parallel p₁ L → parallel p₂ L) := sorry

end problem_incorrect_option_A_l21_21756


namespace find_lambda_and_coordinates_l21_21862

noncomputable theory
open_locale classical

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

variables (e1 e2 : V) (A B C D : V) (lambda : ℝ)

-- Given conditions
variables (h1: linear_independent ℝ ![e1, e2]) -- e1 and e2 are non-collinear, non-zero
variables (h2: B - A = 2 • e1 + e2) -- AB = 2e1 + e2
variables (h3: C - E = -2 • e1 + e2) -- EC = -2e1 + e2
variables (h4: B - E = - e1 + lambda • e2) -- BE = -e1 + λe2
variables (h5: A - E = k * (C - E)) -- AE = k * EC (A, E, C collinear)

-- Proof of the lambda value, coordinates of BC and the point A.
theorem find_lambda_and_coordinates (lambda : ℝ) : 
  lambda = -3/2 ∧ B - C =    -7 • e1 - 2 • e2 :=
begin
  -- Proof skipped
  sorry
end

end find_lambda_and_coordinates_l21_21862


namespace bird_cost_l21_21531

variable (scost bcost : ℕ)

theorem bird_cost (h1 : bcost = 2 * scost)
                  (h2 : (5 * bcost + 3 * scost) = (3 * bcost + 5 * scost) + 20) :
                  scost = 10 ∧ bcost = 20 :=
by {
  sorry
}

end bird_cost_l21_21531


namespace minimum_crooks_l21_21558

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l21_21558


namespace max_possible_value_xv_l21_21602

noncomputable def max_xv_distance (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) : ℝ :=
|x - v|

theorem max_possible_value_xv 
  (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  max_xv_distance x y z w v h1 h2 h3 h4 = 11 :=
sorry

end max_possible_value_xv_l21_21602


namespace compute_product_l21_21785

noncomputable def P (x : ℂ) : ℂ := ∏ k in (finset.range 15).map (nat.cast), (x - exp (2 * pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ := ∏ j in (finset.range 12).map (nat.cast), (x - exp (2 * pi * complex.I * j / 13))

theorem compute_product : 
  (∏ k in (finset.range 15).map (nat.cast), ∏ j in (finset.range 12).map (nat.cast), (exp (2 * pi * complex.I * j / 13) - exp (2 * pi * complex.I * k / 17))) = 1 :=
by
  sorry

end compute_product_l21_21785


namespace count_perfect_squares_between_100_and_500_l21_21500

def smallest_a (x : ℕ) : ℕ :=
  Nat.find (λ a, a^2 > x)

def largest_b (x : ℕ) : ℕ :=
  Nat.find (λ b, b^2 > x) - 1

theorem count_perfect_squares_between_100_and_500 :
  let a := smallest_a 100
  let b := largest_b 500
  b - a + 1 = 12 :=
by
  -- Definitions based on conditions
  let a := smallest_a 100
  have ha : a = 11 := 
    -- the proof follows here
    sorry
  let b := largest_b 500
  have hb : b = 22 := 
    -- the proof follows here
    sorry
  calc
    b - a + 1 = 22 - 11 + 1 : by rw [ha, hb]
           ... = 12          : by norm_num

end count_perfect_squares_between_100_and_500_l21_21500


namespace exists_polyhedron_with_nonvisible_vertices_l21_21390

theorem exists_polyhedron_with_nonvisible_vertices :
  ∃ (P : Polyhedron) (p : Point), 
    p ∉ P ∧ 
    (∀ v : Point, v ∈ vertices P → ¬ visible_from p v) :=
sorry

end exists_polyhedron_with_nonvisible_vertices_l21_21390


namespace certain_number_is_120_l21_21320

theorem certain_number_is_120 : ∃ certain_number : ℤ, 346 * certain_number = 173 * 240 ∧ certain_number = 120 :=
by
  sorry

end certain_number_is_120_l21_21320


namespace tan_alpha_eq_neg_five_twelfths_l21_21842

-- Define the angle α and the given conditions
variables (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π)

-- The goal is to prove that tan α = -5 / 12
theorem tan_alpha_eq_neg_five_twelfths (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π) :
  Real.tan α = -5 / 12 :=
sorry

end tan_alpha_eq_neg_five_twelfths_l21_21842


namespace multiply_decimals_l21_21367

theorem multiply_decimals :
  0.25 * 0.08 = 0.02 :=
sorry

end multiply_decimals_l21_21367


namespace range_f_l21_21060

def f (x : ℝ) : ℝ := 2 * x - 1

theorem range_f : (set.image f { x : ℝ | 0 ≤ x ∧ x ≤ 2 }) = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end range_f_l21_21060


namespace strength_order_l21_21025

variables (a b c d : ℝ)
-- Conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d > b + c
def condition3 : Prop := b > a + c

theorem strength_order (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) : d > b ∧ b > a ∧ a > c :=
by
  sorry

end strength_order_l21_21025


namespace keith_total_spent_l21_21582

theorem keith_total_spent :
  let cost_digimon_pack := 4.45
  let num_digimon_packs := 4
  let cost_baseball_deck := 6.06
  let total_cost := num_digimon_packs * cost_digimon_pack + cost_baseball_deck
  in total_cost = 23.86 :=
by
  let cost_digimon_pack := 4.45
  let num_digimon_packs := 4
  let cost_baseball_deck := 6.06
  let total_cost := num_digimon_packs * cost_digimon_pack + cost_baseball_deck
  show total_cost = 23.86
  sorry

end keith_total_spent_l21_21582


namespace correct_statements_l21_21437

namespace ProofProblem

variable (f : ℕ+ × ℕ+ → ℕ+)
variable (h1 : f (1, 1) = 1)
variable (h2 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2)
variable (h3 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1))

theorem correct_statements :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by
  sorry

end ProofProblem

end correct_statements_l21_21437


namespace diameter_of_large_circle_l21_21235

noncomputable def radius_of_large_circle (r_small : ℝ) : ℝ :=
  let side_length := 2 * r_small in
  let half_side_length := side_length / 2 in
  let half_hypotenuse := half_side_length * 2 in
  r_small + half_hypotenuse

theorem diameter_of_large_circle (r_small : ℝ) (h_r_small : r_small = 4) :
  2 * radius_of_large_circle r_small = 20 :=
by
  rw [radius_of_large_circle, h_r_small]
  simp only [mul_two, nat.cast_bit0, nat.cast_one]
  sorry

end diameter_of_large_circle_l21_21235


namespace length_of_locus_l21_21939

-- The coordinates of points A, B, and C are given by the stated conditions
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (0, 0)

-- The condition given is the sum of distances from a point P to the sides.
def distance_to_sides_sum (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  x + y + (|4 * x + 3 * y - 12| / 5) = 13 / 5

-- The proof goal: calculate the length of the locus of point P
def calculate_locus_length : ℝ := by
  let D := (1, 0)
  let E := (0, 1 / 2)
  exact Real.sqrt ((D.1 - E.1) ^ 2 + (D.2 - E.2) ^ 2)

theorem length_of_locus :
  (∀ P : ℝ × ℝ, distance_to_sides_sum P → True) →
  calculate_locus_length = Real.sqrt 5 / 2 :=
by
  intro h
  -- Proceed with the proof (which is not required for this task)
  sorry

end length_of_locus_l21_21939


namespace increase_by_50_percent_l21_21724

theorem increase_by_50_percent (orig : ℕ) (perc : ℚ) (h_orig : orig = 100) (h_perc : perc = 0.5) : orig + (orig * perc) = 150 := 
by 
  rw [h_orig, h_perc]
  norm_num
  sorry

end increase_by_50_percent_l21_21724


namespace find_a_l21_21089

def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem find_a (a b : ℤ) (h1 : b = a + 1) (h2 : f (a : ℝ) < 0) (h3 : f (b : ℝ) > 0) : a = 2 :=
by
  sorry

end find_a_l21_21089


namespace flawed_reasoning_error_due_to_major_premise_l21_21252

theorem flawed_reasoning_error_due_to_major_premise:
  (∀ z : ℂ, z^2 ≥ 0) →
  (i : ℂ) →
  ¬(i^2 > 0 ∧ -1 > 0) →
  ¬(-1 > 0) :=
by 
  intros h_major h_i h_conclusion
  sorry

end flawed_reasoning_error_due_to_major_premise_l21_21252


namespace card_probability_sequence_l21_21416

/-- 
Four cards are dealt at random from a standard deck of 52 cards without replacement.
The probability that the first card is a Jack, the second card is a Queen, the third card is a King, and the fourth card is an Ace is given by:
-/
theorem card_probability_sequence :
  let p1 := 4 / 52,
      p2 := 4 / 51,
      p3 := 4 / 50,
      p4 := 4 / 49
  in p1 * p2 * p3 * p4 = 64 / 1624350 :=
by
  let p1 := 4 / 52
  let p2 := 4 / 51
  let p3 := 4 / 50
  let p4 := 4 / 49
  show p1 * p2 * p3 * p4 = 64 / 1624350
  sorry

end card_probability_sequence_l21_21416


namespace bus_stops_for_18_minutes_l21_21396

-- Definitions based on conditions
def speed_without_stoppages : ℝ := 50 -- kmph
def speed_with_stoppages : ℝ := 35 -- kmph
def distance_reduced_due_to_stoppage_per_hour : ℝ := speed_without_stoppages - speed_with_stoppages

noncomputable def time_bus_stops_per_hour (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem bus_stops_for_18_minutes :
  time_bus_stops_per_hour distance_reduced_due_to_stoppage_per_hour (speed_without_stoppages / 60) = 18 := by
  sorry

end bus_stops_for_18_minutes_l21_21396


namespace slope_midpoints_l21_21290

-- Define the midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the slope calculation
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Problem definition
theorem slope_midpoints :
  let m1 := midpoint (1, 2) (3, 6) in
  let m2 := midpoint (4, 1) (7, 4) in
  slope m1 m2 = -3 / 7 :=
by
  let m1 := midpoint (1, 2) (3, 6)
  let m2 := midpoint (4, 1) (7, 4)
  have h_m1 : m1 = (2, 4) := rfl
  have h_m2 : m2 = (5.5, 2.5) := rfl
  rw [h_m1, h_m2]
  -- Here we need a proof step, so we use sorry
  sorry

end slope_midpoints_l21_21290


namespace proof_problem_l21_21059

def f (x : ℝ) := 2 * Real.sin ((π / 6) * x + π / 3)

def is_on_graph (p : ℝ × ℝ) : Prop := p.2 = f p.1

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

def is_symmetric_about (a b c : ℝ × ℝ) : Prop :=
  b.1 + c.1 = 2 * a.1 ∧ b.2 + c.2 = 0

def is_a_point (a : ℝ × ℝ) (x : ℝ) : Prop :=
  a.1 = x ∧ a.2 = 0

theorem proof_problem : 
  ∃ (a b c : ℝ × ℝ), 
    is_a_point a 4 ∧ 
    is_on_graph b ∧ is_on_graph c ∧ 
    is_on_x_axis a ∧ 
    is_symmetric_about a b c ∧ 
    ((b.1, b.2) + (c.1, c.2)).1  • (a.1, a.2).1 = 32 :=
by
  sorry

end proof_problem_l21_21059


namespace platform_length_l21_21328

/-- 
Problem Statement: 
A goods train runs at the speed of 72 km/hr and crosses a platform of a certain length in 26 sec. 
The length of the goods train is 240 m. Prove that the length of the platform is 280 meters.
--/
theorem platform_length 
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (train_length_m : ℕ)
  (conversion_factor : ℚ := 5 / 18)
  (speed_mps : ℚ := speed_kmph * conversion_factor)
  (distance_covered : ℚ := speed_mps * time_sec) :
  speed_kmph = 72 → 
  time_sec = 26 → 
  train_length_m = 240 → 
  distance_covered - train_length_m = 280 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have hc : conversion_factor = 5 / 18 := rfl
  rw hc
  norm_num
  have hs : speed_mps = 20 := by norm_num
  rw hs
  have hd : distance_covered = 520 := by norm_num
  rw hd
  norm_num

end platform_length_l21_21328


namespace number_of_possible_flags_l21_21383

theorem number_of_possible_flags : 
  let colors := { "purple", "gold", "silver" } in
  ∃ (f : Fin 3 → colors), (∀ i : Fin 2, f i ≠ f (i + 1)) → (Fin 3 → colors) :=
by
  let colors := { "purple", "gold", "silver" }
  have h1 : 3 = card colors := by sorry
  have h2 : 3 = card { c // ¬(f 1 = c) } := by sorry
  have h3 : 3 = card { c // ¬(f 2 = c) } := by sorry
  have total_possibilities : 3 * 2 * 2 = 12 := by sorry
  existsi total_possibilities
  exact sorry

end number_of_possible_flags_l21_21383


namespace part1_part2_part3_l21_21592

-- Definitions from conditions
def S (n : ℕ) : ℕ := (3^(n + 1) - 3) / 2
def a (n : ℕ) : ℕ := 3^n
def b (n : ℕ) : ℕ := 2 * a n / (a n - 2)^2
def T (n : ℕ) : ℕ := ∑ i in finset.range n, b i

-- Proof Statements
theorem part1 (n : ℕ) : S (n - 1) - S (n - 2) = a n := sorry

theorem part2 (n : ℕ) : 
  (k : ℕ) → (1 = k) → n = (finset.min' (finset.image (λ n, (S (2 * n) + 15)/a n) (finset.range k)) _) := sorry

theorem part3 (n : ℕ) : T n < 13 / 2 := sorry

end part1_part2_part3_l21_21592


namespace find_a10_l21_21565

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def a2_eq_4 (a : ℕ → ℝ) := a 2 = 4

def a6_eq_6 (a : ℕ → ℝ) := a 6 = 6

theorem find_a10 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h2 : a2_eq_4 a) (h6 : a6_eq_6 a) : 
  a 10 = 9 :=
sorry

end find_a10_l21_21565


namespace Anchuria_min_crooks_l21_21544

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l21_21544


namespace Pythagorean_triple_l21_21775

theorem Pythagorean_triple : ∃ (a b c : ℕ), (a = 6) ∧ (b = 8) ∧ (c = 10) ∧ (a^2 + b^2 = c^2) :=
by
  use 6, 8, 10
  split; try {refl}
  calc
    6^2 + 8^2 = 36 + 64 := by rw [pow_two, pow_two]
            ... = 100 := by norm_num
            ... = 10^2 := by norm_num
  done

end Pythagorean_triple_l21_21775


namespace ratio_of_areas_of_circles_l21_21087

theorem ratio_of_areas_of_circles 
  (R_A R_B : ℝ) 
  (h : (π / 2 * R_A) = (π / 3 * R_B)) : 
  (π * R_A ^ 2) / (π * R_B ^ 2) = (4 : ℚ) / 9 := 
sorry

end ratio_of_areas_of_circles_l21_21087


namespace must_be_true_l21_21422

noncomputable def f (x : ℝ) := |Real.log x|

theorem must_be_true (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) 
                     (h3 : f b < f a) (h4 : f a < f c) :
                     (c > 1) ∧ (1 / c < a) ∧ (a < 1) ∧ (a < b) ∧ (b < 1 / a) :=
by
  sorry

end must_be_true_l21_21422


namespace cube_root_of_16_div_54_as_fraction_l21_21814

theorem cube_root_of_16_div_54_as_fraction : 
  (∛(16 / 54) = (2 / 3)) := 
by
  sorry

end cube_root_of_16_div_54_as_fraction_l21_21814


namespace sum_of_real_parts_of_conjugate_complex_numbers_l21_21845

noncomputable def conjugate_of (z : ℂ) : ℂ := complex.conj z

theorem sum_of_real_parts_of_conjugate_complex_numbers
  (x y : ℝ)
  (h_conj : x + y * complex.I = conjugate_of ((3 + complex.I) / (1 + complex.I)))
  : x + y = 3 := by
  sorry

end sum_of_real_parts_of_conjugate_complex_numbers_l21_21845


namespace hyperbola_slope_product_constant_l21_21872

variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (h_hyperbola : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)

noncomputable def is_on_hyperbola (m n : ℝ) : Prop :=
  (m^2 / a^2) - (n^2 / b^2) = 1

theorem hyperbola_slope_product_constant
  {m n x y : ℝ}
  (hM : is_on_hyperbola a b m n)
  (hP : is_on_hyperbola a b x y)
  (h_slope_exist : x ≠ m ∧ x ≠ -m) :
  let k_PM := (y - n) / (x - m)
      k_PN := (y + n) / (x + m)
  in k_PM * k_PN = (b^2 / a^2) :=
sorry

end hyperbola_slope_product_constant_l21_21872


namespace smallest_m_l21_21386

theorem smallest_m (m : ℕ) (h1 : 7 ≡ 2 [MOD 5]) : 
  (7^m ≡ m^7 [MOD 5]) ↔ (m = 7) :=
by sorry

end smallest_m_l21_21386


namespace initial_customers_l21_21751

theorem initial_customers (x : ℝ) : (x - 8 + 4 = 9) → x = 13 :=
by
  sorry

end initial_customers_l21_21751


namespace max_value_fraction_squares_l21_21985

-- Let x and y be positive real numbers
variable (x y : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)

theorem max_value_fraction_squares (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k, (x + 2 * y)^2 / (x^2 + y^2) ≤ k) ∧ (∀ z, (x + 2 * y)^2 / (x^2 + y^2) ≤ z) → k = 9 / 2 :=
by
  sorry

end max_value_fraction_squares_l21_21985


namespace num_valid_pairs_l21_21905

-- Define variables for 3003 and its factorization
def N := 3003
def factors := [3, 7, 11, 13]

-- Define the gcd condition
def gcd_condition (i : ℕ) : Prop :=
  Nat.gcd (10 ^ i) N = 1

-- Define the divisibility condition
def divisibility_condition (j i : ℕ) : Prop :=
  (N ∣ (10 ^ (j - i) - 1))

-- Define the range condition for i and j
def range_condition (i j : ℕ) : Prop :=
  0 ≤ i ∧ i < j ∧ j ≤ 99

-- Define the main condition that combines everything
def main_condition (i j : ℕ) : Prop :=
  range_condition i j ∧ gcd_condition i ∧ divisibility_condition j i

-- Define the set of valid (i, j) pairs and count them
def valid_pairs : ℕ :=
  (Finset.range 100).sum (λ i, (Finset.range (100 - i - 1)).count (λ j, main_condition i (j + i + 1)))

-- State the theorem
theorem num_valid_pairs : valid_pairs = 368 :=
  sorry

end num_valid_pairs_l21_21905


namespace range_of_distance_l21_21088

noncomputable def A (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def B (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

theorem range_of_distance (α β : ℝ) :
  1 ≤ Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ∧
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ≤ 5 :=
by
  sorry

end range_of_distance_l21_21088


namespace triangle_area_squared_l21_21351

theorem triangle_area_squared
  (R : ℝ)
  (A : ℝ)
  (AC_minus_AB : ℝ)
  (area : ℝ)
  (hx : R = 4)
  (hy : A = 60)
  (hz : AC_minus_AB = 4)
  (area_eq : area = 8 * Real.sqrt 3) :
  area^2 = 192 :=
by
  -- We include the conditions 
  have hR := hx
  have hA := hy
  have hAC_AB := hz
  have harea := area_eq
  -- We will use these to construct the required proof 
  sorry

end triangle_area_squared_l21_21351


namespace max_area_of_triangle_ABC_l21_21154

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) :=
  (1 / 2) * a * c * Real.sin B

theorem max_area_of_triangle_ABC (a b c A B C : ℝ)
  (h1 : a * c = 6)
  (h2 : Real.sin B + 2 * Real.sin C * Real.cos A = 0) :
  ∃ A, B, C, area_triangle a b c A B C ≤ 3 / 2 :=
by
  sorry

end max_area_of_triangle_ABC_l21_21154


namespace count_multiples_of_12_between_15_and_250_l21_21485

theorem count_multiples_of_12_between_15_and_250 : 
  (∃ n : ℕ, ∃ m : ℕ, (2 ≤ n ∧ n ≤ 20) ∧ (m = 12 * n) ∧ (15 < m ∧ m < 250)) → 
  Nat.card { x ∈ Nat | ∃ k : ℕ, x = 12 * k ∧ 15 < x ∧ x < 250 } = 19 := 
by
  sorry

end count_multiples_of_12_between_15_and_250_l21_21485


namespace vector_norm_sq_sum_l21_21978

theorem vector_norm_sq_sum (a b : ℝ × ℝ) (m : ℝ × ℝ) (h_m : m = (4, 6))
  (h_midpoint : m = ((2 * a.1 + 2 * b.1) / 2, (2 * a.2 + 2 * b.2) / 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 10) :
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32 :=
by 
  sorry

end vector_norm_sq_sum_l21_21978


namespace carol_first_toss_six_probability_l21_21353

theorem carol_first_toss_six_probability :
  let p := 1 / 6
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  (prob_carol_first_six / (1 - prob_cycle)) = 125 / 671 :=
by
  let p := (1 / 6:ℚ)
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  have sum_geo_series : prob_carol_first_six / (1 - prob_cycle) = 125 / 671 := sorry
  exact sum_geo_series

end carol_first_toss_six_probability_l21_21353


namespace number_of_solutions_f_proof_of_extremes_l21_21461

-- Part (1)
theorem number_of_solutions_f (m : ℝ) (h_m : m ≥ 1) :
  (∀ x, f x = me^(x-1) - log x) →
  (m = 1 → ∃! x, f x - 1 = 0) ∧ (m > 1 → ¬∃ x, f x - 1 = 0) :=
by
  sorry

-- Part (2)
theorem proof_of_extremes (t : ℝ) (h_t : e < t ∧ t < e^2/2) (x_1 x_2 : ℝ) (h_x1_lt_x2 : x_1 < x_2) :
  let f (x : ℝ) := e^(x-1) - log x,
      g (x : ℝ) := f x + log x - (t * x^2 + e) / 2 in
  (2 < x_1 + x_2 ∧ x_1 + x_2 < 3) ∧ (g x_1 + 2 * g x_2 < 0) :=
by
  sorry

end number_of_solutions_f_proof_of_extremes_l21_21461


namespace correct_rounded_result_l21_21994

def round_to_nearest_ten (n : ℤ) : ℤ :=
  (n + 5) / 10 * 10

theorem correct_rounded_result :
  round_to_nearest_ten ((57 + 68) * 2) = 250 :=
by
  sorry

end correct_rounded_result_l21_21994


namespace max_distance_l21_21259

-- Defining a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  h : (x^2 / 16) + (y^2 / 4) = 1

-- Define the line equation
def line (p : PointOnEllipse) : Prop :=
  p.x + 2 * p.y - Real.sqrt 2 = 0

-- Distance from a point to a line
def distance (p : PointOnEllipse) : ℝ :=
  abs (p.x + 2 * p.y - Real.sqrt 2) / Real.sqrt (1^2 + 2^2)

theorem max_distance : ∃ p : PointOnEllipse, distance p = Real.sqrt 10 :=
by
  sorry

end max_distance_l21_21259


namespace lines_meet_perpendicular_l21_21352

theorem lines_meet_perpendicular
  (A B C P Q R S : Point)
  (collinear_ABC : Collinear A B C)
  (B_between_AC : Between B A C)
  (circle_K1 : Circle A B)
  (circle_K2 : Circle B C)
  (circle_conditions : ∃ circle : Circle, TangentAt circle B AC ∧ Meets circle circle_K1 P ∧ Meets circle circle_K2 Q)
  (PQ_meets_K1 : Meets (Line P Q) circle_K1 R)
  (PQ_meets_K2 : Meets (Line P Q) circle_K2 S) :
  ∃ T : Point, Intersection T (Perpendicular B AC) (Line A R) ∧ Intersection T (Perpendicular B AC) (Line C S) :=
sorry

end lines_meet_perpendicular_l21_21352


namespace probability_jqka_is_correct_l21_21413

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end probability_jqka_is_correct_l21_21413


namespace box_filling_possibilities_l21_21815

def possible_numbers : List ℕ := [2015, 2016, 2017, 2018, 2019]

def fill_the_boxes (D O G C W : ℕ) : Prop :=
  D + O + G = C + O + W

theorem box_filling_possibilities :
  (∃ D O G C W : ℕ, 
    D ∈ possible_numbers ∧
    O ∈ possible_numbers ∧
    G ∈ possible_numbers ∧
    C ∈ possible_numbers ∧
    W ∈ possible_numbers ∧
    D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ W ∧
    O ≠ G ∧ O ≠ C ∧ O ≠ W ∧
    G ≠ C ∧ G ≠ W ∧
    C ≠ W ∧
    fill_the_boxes D O G C W) → 
    ∃ ways : ℕ, ways = 24 :=
  sorry

end box_filling_possibilities_l21_21815


namespace range_of_a_l21_21520

theorem range_of_a (a: ℝ) : (∀ x: ℝ, 0 < x ∧ x < 4 → |x - a| < 3) → 1 ≤ a ∧ a ≤ 3 :=
begin
  sorry
end

end range_of_a_l21_21520


namespace compute_product_l21_21787

noncomputable def P (x : ℂ) : ℂ := ∏ k in (finset.range 15).map (nat.cast), (x - exp (2 * pi * complex.I * k / 17))

noncomputable def Q (x : ℂ) : ℂ := ∏ j in (finset.range 12).map (nat.cast), (x - exp (2 * pi * complex.I * j / 13))

theorem compute_product : 
  (∏ k in (finset.range 15).map (nat.cast), ∏ j in (finset.range 12).map (nat.cast), (exp (2 * pi * complex.I * j / 13) - exp (2 * pi * complex.I * k / 17))) = 1 :=
by
  sorry

end compute_product_l21_21787


namespace max_electronic_thermometers_l21_21532

-- Definitions
def budget : ℕ := 300
def price_mercury : ℕ := 3
def price_electronic : ℕ := 10
def total_students : ℕ := 53

-- The theorem statement
theorem max_electronic_thermometers : 
  (∃ x : ℕ, x ≤ total_students ∧ 10 * x + 3 * (total_students - x) ≤ budget ∧ 
            ∀ y : ℕ, y ≤ total_students ∧ 10 * y + 3 * (total_students - y) ≤ budget → y ≤ x) :=
sorry

end max_electronic_thermometers_l21_21532


namespace complex_number_quadrant_l21_21438

open Complex

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) / z = Complex.I) : 
  (0 < z.re) ∧ (0 < z.im) :=
by
  -- sorry to skip the actual proof
  sorry

end complex_number_quadrant_l21_21438


namespace diane_harvested_honey_this_year_l21_21806

-- Define the conditions
def last_years_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

-- Lean statement to prove the question equals the answer
theorem diane_harvested_honey_this_year :
  last_years_harvest + increase_this_year = 8564 :=
by
  rw [last_years_harvest, increase_this_year]
  simp
  exact rfl

end diane_harvested_honey_this_year_l21_21806


namespace g_one_eq_l21_21597

-- Given definitions
variables {a b c : ℝ}
variable (r : ℝ → ℝ)

-- Conditions
def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c
def g (x : ℝ) : ℝ := (x - 1/(r 0)) * (x - 1/(r 1)) * (x - 1/(r 2))

-- Assumptions
axiom roots_f {p q r : ℝ} : f(p) = 0 ∧ f(q) = 0 ∧ f(r) = 0

-- Prove g(1) = (1 + a + b + c) / c
theorem g_one_eq : ∀ a b c : ℝ, 1 < a → a < b → b < c → g(1) = (1 + a + b + c) / c :=
by 
  sorry

end g_one_eq_l21_21597


namespace jane_paid_cashier_l21_21159

-- Define the conditions in Lean
def skirts_bought : ℕ := 2
def price_per_skirt : ℕ := 13
def blouses_bought : ℕ := 3
def price_per_blouse : ℕ := 6
def change_received : ℤ := 56

-- Calculate the total cost in Lean
def cost_of_skirts : ℕ := skirts_bought * price_per_skirt
def cost_of_blouses : ℕ := blouses_bought * price_per_blouse
def total_cost : ℕ := cost_of_skirts + cost_of_blouses
def amount_paid : ℤ := total_cost + change_received

-- Lean statement to prove the question
theorem jane_paid_cashier :
  amount_paid = 100 :=
by
  sorry

end jane_paid_cashier_l21_21159


namespace inequality_proof_l21_21179

noncomputable def inequality (x y z : ℝ) : Prop :=
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y + z) (hx_pos: 0 < x) (hy_pos: 0 < y) (hz_pos: 0 < z) :
  inequality x y z :=
by
  sorry

end inequality_proof_l21_21179


namespace trigonometric_identity_solution_l21_21712

theorem trigonometric_identity_solution (k : ℤ) : 
  8.433 * (cos (π * k)) ^ (-4) + (cos (π * k)) ^ 4 = 1 + cos (2 * (π * k)) - 2 * (sin (2 * (π * k))) ^ 2 :=
by {
  sorry
}

end trigonometric_identity_solution_l21_21712


namespace minimum_crooks_l21_21553

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l21_21553


namespace compose_homothety_is_homothety_l21_21291

variables {A B C A' B' C' A1 B1 A2 B2 A3 B3 : Type} [real_space A] [real_space B] [real_space C]
variables {S1 S2 S3 : Type} [real_space S1] [real_space S2] [real_space S3]

def homothety (k : ℝ) (S : Type) (P Q : Type) [real_space S] [real_space P] [real_space Q] : Type :=
{ center : S 
, coefficient : k 
, transform : P × Q → (P × Q) }

def compose_homothety 
  (h1 : homothety k1 S1 (A1 × B1))
  (h2 : homothety k2 S2 (A2 × B2))
  : homothety (k2 * k1) S3 (A1 × B3) :=
{ center := let S1 := h1.center in let S2 := h2.center in (S1 ∩ S2).some,
  coefficient := k2 * k1,
  transform := λ (A1 B3 : A1 × B3), (A1, (k2 * k1) • (B3 - A1) + A1) -- simplified transformation rule
}

theorem compose_homothety_is_homothety
  (h1 : homothety k1 S1 (A1 × B1))
  (h2 : homothety k2 S2 (A2 × B2))
  : ∃ h3 : homothety (k2 * k1) S3 (A1 × B3),
    h3 = compose_homothety h1 h2 :=
begin
  let h3 := compose_homothety h1 h2,
  use h3,
  sorry
end

end compose_homothety_is_homothety_l21_21291


namespace arithmetic_sequence_a1_l21_21429

theorem arithmetic_sequence_a1 (a : ℕ → ℤ) (h1 : (∑ i in Finset.range 10, a i.succ) = 65)
  (h2 : (∑ i in Finset.range 10, a (i + 11)) = 165)
  (h3 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = 2 :=
by
  sorry

end arithmetic_sequence_a1_l21_21429


namespace find_n_l21_21890

theorem find_n (x y n : ℝ) (h1 : 2 * x - 5 * y = 3 * n + 7) (h2 : x - 3 * y = 4) 
  (h3 : x = y):
  n = -1 / 3 := 
by 
  sorry

end find_n_l21_21890


namespace range_of_m_l21_21674

theorem range_of_m (x m : ℝ) :
  (∀ x : ℝ, x + 3 < 3x + 1 → x > m + 1 → x > 1) → m ≤ 0 :=
sorry

end range_of_m_l21_21674


namespace f_val_neg1_f_val_2_f_expr_monotonicity_f_min_max_l21_21058

variable {x : ℝ}
variable {k : ℝ} (hk : k < 0)

def f : ℝ → ℝ 
| y =>
  if y ∈ set.Ico (-3 : ℝ) (-2) then k^2 * (y + 2) * (y + 4)
  else if y ∈ set.Ico (-2 : ℝ) 0 then k * y * (y + 2)
  else if y ∈ set.Ico 0 2 then y * (y - 2)
  else if y ∈ set.Icc 2 3 then (1 / k) * (y - 2) * (y - 4)
  else 0

theorem f_val_neg1 : f hk (-1) = -k := sorry
theorem f_val_2.5 : f hk 2.5 = - (3 / (4 * k)) := sorry

theorem f_expr_monotonicity : 
  (∀ y ∈ set.Ico (-3 : ℝ) (-2), f hk y = k^2 * (y + 2) * (y + 4)) ∧
  (∀ y ∈ set.Ico (-2 : ℝ) 0, f hk y = k * y * (y + 2)) ∧
  (∀ y ∈ set.Ico 0 2, f hk y = y * (y - 2)) ∧
  (∀ y ∈ set.Icc 2 3, f hk y = (1 / k) * (y - 2) * (y - 4)) ∧
  (∀ y ∈ set.Ico (-3 : ℝ) (-1), increasing (f hk) y) ∧
  (∀ y ∈ set.Ico (-1 : ℝ) 1, decreasing (f hk) y) ∧
  (∀ y ∈ set.Ico 1 3, increasing (f hk) y) := sorry

theorem f_min_max :
  ((k < -1) → (f hk (-3) = -k^2 ∧ f hk (-1) = -k)) ∧
  ((k = -1) → (f hk (-3) = -1 ∧ f hk 1 = -1 ∧ f hk (-1) = 1 ∧ f hk 3 = 1)) ∧
  ((-1 < k) → (f hk 1 = -1 ∧ f hk 3 = -1/k)) := sorry

end f_val_neg1_f_val_2_f_expr_monotonicity_f_min_max_l21_21058


namespace train_speed_l21_21350

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (distance_eq : distance = 20) (time_eq : time_minutes = 20) : 
  distance / (time_minutes / 60) = 60 :=
by
  rw [distance_eq, time_eq]
  -- convert minutes to hours
  have time_hours : ℝ := time_minutes / 60
  rw time_eq at time_hours
  -- calculate speed
  have speed_eq : distance / time_hours = 20 / (20 / 60) :=
    by congr
  exact speed_eq
  sorry

end train_speed_l21_21350


namespace projection_of_a_on_b_l21_21285

-- Define vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of the dot product between two 2D vectors
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- Definition of the magnitude of a 2D vector
def magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 * x.1 + x.2 * x.2)

-- Definition of the projection of vector a onto vector b
def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_ab := dot_product a b
  let mag_b_sq := (magnitude b) ^ 2
  (dot_ab / mag_b_sq) • b

-- The theorem to prove
theorem projection_of_a_on_b : projection a b = (2, 0) :=
  sorry

end projection_of_a_on_b_l21_21285


namespace sum_q_p_l21_21172

def p_domain : Set ℤ := {-2, -1, 0, 1}
def p_range : Set ℤ := {-1, 1, 3, 5}
def q_domain : Set ℤ := {0, 1, 2, 3, 4}

noncomputable def q (x : ℤ) : ℤ := x + 2

theorem sum_q_p :
  (∀ x ∈ p_domain, p(x) ∈ p_range) →
  (∀ x ∈ q_domain, q(x) ∈ Set.range q) →
  let q_p_values : Set ℤ := { q(y) | y ∈ p_range ∩ q_domain } in
  (∑ y in q_p_values, y) = 8 :=
sorry

end sum_q_p_l21_21172


namespace concyclic_F_L_M_N_l21_21155

variables 
  (A B C O D E F L M N : Type)
  [triangle_ABC : triangle A B C]
  [circumcenter_ABC : circumcenter A B C O]
  [point_D_on_AB : point_on D A B]
  [point_E_on_AC : point_on E A C]
  [perpendicular_OF_DE : perpendicular O F D E]
  [midpoint_L_DE : midpoint L D E]
  [midpoint_M_BE : midpoint M B E]
  [midpoint_N_CD : midpoint N C D]

theorem concyclic_F_L_M_N :
  concyclic F L M N := sorry

end concyclic_F_L_M_N_l21_21155


namespace age_ratio_l21_21529
open Nat

theorem age_ratio (B_c : ℕ) (h1 : B_c = 42) (h2 : ∀ A_c, A_c = B_c + 12) : (A_c + 10) / (B_c - 10) = 2 :=
by
  sorry

end age_ratio_l21_21529


namespace total_chairs_correct_l21_21024

-- Define the problem conditions
def chairs_in_section_A := 25 * 17

def sum_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def chairs_in_section_B := sum_arith_seq 20 2 30

def chairs_in_section_C := sum_arith_seq 16 (-1) 29

-- The total number of chairs for the play
def total_chairs_for_play : ℕ :=
  chairs_in_section_A + chairs_in_section_B + chairs_in_section_C

-- Theorem stating the total number of chairs
theorem total_chairs_correct : total_chairs_for_play = 1953 := by
  -- Definition of chairs in Section A
  let A := 425
  -- Calculation of chairs in Section B
  let B := (30 * (2 * 20 + 29 * 2)) / 2
  -- Calculation of chairs in Section C
  let C := (29 * (2 * 16 + 28 * (-1))) / 2
  -- Summing all chairs
  have t : total_chairs_for_play = A + B + C := rfl
  -- Verifying the sum
  calc
    total_chairs_for_play = 425 + 1470 + 58 : by rw [t]
    ... = 1953 : by sorry

end total_chairs_correct_l21_21024


namespace smallest_positive_value_l21_21812

noncomputable def exprA := 30 - 4 * Real.sqrt 14
noncomputable def exprB := 4 * Real.sqrt 14 - 30
noncomputable def exprC := 25 - 6 * Real.sqrt 15
noncomputable def exprD := 75 - 15 * Real.sqrt 30
noncomputable def exprE := 15 * Real.sqrt 30 - 75

theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  exprC < exprA ∧
  exprC < exprB ∧
  exprC < exprD ∧
  exprC < exprE ∧
  exprC > 0 :=
by sorry

end smallest_positive_value_l21_21812


namespace percentage_books_not_sold_correct_l21_21579

-- Definitions of initial stock and books sold each day
def initial_stock : ℕ := 620
def books_sold_monday : ℕ := 50
def books_sold_tuesday : ℕ := 82
def books_sold_wednesday : ℕ := 60
def books_sold_thursday : ℕ := 48
def books_sold_friday : ℕ := 40

-- Definition of total books sold
def total_books_sold : ℕ := 
  books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday + books_sold_friday

-- Definition of books not sold
def books_not_sold : ℕ := 
  initial_stock - total_books_sold

-- Definition of the percentage of books not sold
def percentage_books_not_sold : ℚ := 
  (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

-- The theorem to prove
theorem percentage_books_not_sold_correct :
  percentage_books_not_sold ≈ 54.84 := 
sorry

end percentage_books_not_sold_correct_l21_21579


namespace find_other_solution_l21_21439

theorem find_other_solution (x : ℚ) :
  (72 * x ^ 2 + 43 = 113 * x - 12) → (x = 3 / 8) → (x = 43 / 36 ∨ x = 3 / 8) :=
by
  sorry

end find_other_solution_l21_21439


namespace at_least_two_children_same_forename_and_surname_l21_21676

theorem at_least_two_children_same_forename_and_surname
    (children : Fin 33)
    (forenames surnames : Fin 33 → ℕ)
    (forename_sum : Σ i, Fin (forenames i) = 33)
    (surname_sum : Σ j, Fin (surnames j) = 33)
    (distinct_numbers : ∀ n : ℕ, (n <= 10 → ∃ c ∈ children, c = n)) : 
    ∃ c1 c2 ∈ children, c1 ≠ c2 ∧ forenames c1 = forenames c2 ∧ surnames c1 = surnames c2 := sorry

end at_least_two_children_same_forename_and_surname_l21_21676


namespace reflection_line_slope_l21_21657

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l21_21657


namespace keith_total_spent_l21_21581

theorem keith_total_spent :
  let cost_digimon_pack := 4.45
  let num_digimon_packs := 4
  let cost_baseball_deck := 6.06
  let total_cost := num_digimon_packs * cost_digimon_pack + cost_baseball_deck
  in total_cost = 23.86 :=
by
  let cost_digimon_pack := 4.45
  let num_digimon_packs := 4
  let cost_baseball_deck := 6.06
  let total_cost := num_digimon_packs * cost_digimon_pack + cost_baseball_deck
  show total_cost = 23.86
  sorry

end keith_total_spent_l21_21581


namespace parallelogram_with_equal_diagonals_is_rectangle_l21_21293

def parallelogram (α : Type) [euclidean_space α] (A B C D : α) : Prop :=
  -- A quadrilateral ABCD is a parallelogram if opposite sides are equal and parallel
  (A - B) = (C - D) ∧ (B - C) = (D - A)

def equal_diagonals (A B C D : α) : Prop :=
  -- Parallelogram ABCD has equal diagonals if AC = BD
  dist A C = dist B D

def rectangle (α : Type) [euclidean_space α] (A B C D : α) : Prop :=
  -- Parallelogram ABCD is a rectangle if it has equal diagonals and has right angles.
  parallelogram α A B C D ∧ (angle A B C = π/2)

theorem parallelogram_with_equal_diagonals_is_rectangle (α : Type) [euclidean_space α] 
  (A B C D : α) : 
  parallelogram α A B C D ∧ equal_diagonals A B C D → rectangle α A B C D :=
sorry

end parallelogram_with_equal_diagonals_is_rectangle_l21_21293


namespace rise_ratio_l21_21693

-- Define necessary variables
variables (h3 h4 : ℝ)
variables (r3 r4 : ℝ := 4) (r8 r8 : ℝ := 8)
variables (v_sphere := 4 / 3 * Real.pi * 2 ^ 3)

-- Initial volumes
def V3 (h3 : ℝ) := (1 / 3) * Real.pi * r3 ^ 2 * h3
def V4 (h4 : ℝ) := (1 / 3) * Real.pi * r4 ^ 2 * h4

-- New volumes after adding sphere
def V3' (h3 : ℝ) := V3 h3 + v_sphere
def V4' (h4 : ℝ) := V4 h4 + v_sphere

-- Prove the rise ratio of the liquid level
theorem rise_ratio : 
  ∀ (h3 h4 : ℝ), V3 h3 = V4 h4 → (h3 / h4 = 4) → 
  let Δh3 := 2 
  let Δh4 := 1 / 2 
  Δh3 / Δh4 = 4 := 
by
  intros h3 h4 hv_eq h_ratio
  have Δh3 := 2
  have Δh4 := 1 / 2
  show Δh3 / Δh4 = 4 by
  sorry

end rise_ratio_l21_21693


namespace luke_games_l21_21998

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l21_21998


namespace cot_difference_abs_eq_sqrt3_l21_21136

theorem cot_difference_abs_eq_sqrt3 
  (A B C D P : Point) (x y : ℝ) (h1 : is_triangle A B C) 
  (h2 : is_median A D B C) (h3 : ∠(D, A, P) = 60)
  (BD_eq_CD : BD = x) (CD_eq_x : CD = x)
  (BP_eq_y : BP = y) (AP_eq_sqrt3 : AP = sqrt(3) * (x + y))
  (cot_B : cot B = -y / ((sqrt 3) * (x + y)))
  (cot_C : cot C = (2 * x + y) / (sqrt 3 * (x + y))) 
  (x_y_neq_zero : x + y ≠ 0) :
  abs (cot B - cot C) = sqrt 3
  := sorry

end cot_difference_abs_eq_sqrt3_l21_21136


namespace cut_decagon_into_two_regular_polygons_l21_21570

theorem cut_decagon_into_two_regular_polygons :
  ∃ (P Q : Finset (Fin 10)), 
  (∀ x ∈ P, x ∈ (Finset.range 10)) ∧ 
  (∀ x ∈ Q, x ∈ (Finset.range 10)) ∧ 
  ((P ∪ Q) = Finset.range 10) ∧ 
  (∀ x y ∈ P, Finset.card (Finset.filter (λ z => z ∈ P) (Finset.range 10)) = 5) ∧ 
  (∀ x y ∈ Q, Finset.card (Finset.filter (λ z => z ∈ Q) (Finset.range 10)) = 5) := 
sorry

end cut_decagon_into_two_regular_polygons_l21_21570


namespace unique_solution_count_l21_21830

theorem unique_solution_count :
  (Finset.card ({ a : ℕ | ∀ x : ℕ, (2x > 3x - 3) ∧ (4x - a > -8) → (x = 2 → x = 1 ∨ x = 2)}))
  = 4 :=
sorry

end unique_solution_count_l21_21830


namespace application_methods_l21_21321

variables (students : Fin 6) (colleges : Fin 3)

def total_applications_without_restriction : ℕ := 3^6
def applications_missing_one_college : ℕ := 2^6
def overcounted_applications_missing_two_college : ℕ := 1

theorem application_methods (h1 : total_applications_without_restriction = 729)
    (h2 : applications_missing_one_college = 64)
    (h3 : overcounted_applications_missing_two_college = 1) :
    ∀ (students : Fin 6), ∀ (colleges : Fin 3),
      (total_applications_without_restriction - 3 * applications_missing_one_college + 3 * overcounted_applications_missing_two_college = 540) :=
by {
  sorry
}

end application_methods_l21_21321


namespace largest_number_of_gold_coins_l21_21710

theorem largest_number_of_gold_coins (n : ℕ) (h1 : n % 15 = 4) (h2 : n < 150) : n ≤ 139 :=
by {
  -- This is where the proof would go.
  sorry
}

end largest_number_of_gold_coins_l21_21710


namespace count_perfect_squares_between_100_500_l21_21487

theorem count_perfect_squares_between_100_500 :
  ∃ (count : ℕ), count = finset.card ((finset.Icc 11 22).filter (λ n, 100 < n^2 ∧ n^2 < 500)) :=
begin
  use 12,
  rw ← finset.card_Icc,
  sorry
end

end count_perfect_squares_between_100_500_l21_21487


namespace find_alpha_find_MN_l21_21533

variables {t : ℝ} {α : ℝ} (x y : ℝ)

def line_l_parametric (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (x = -3 + t * Real.cos α, y = Real.sqrt 3 + t * Real.sin α)

def curve_C_polar : ℝ := 2 * Real.sqrt 3

def intersect_conditions : Prop :=
  abs ((Real.tan α) * x - y + 3 * (Real.tan α) + Real.sqrt 3) / Real.sqrt((Real.tan α) ^ 2 + 1) = 3

def distance_AB : ℝ := 2 * Real.sqrt 3

theorem find_alpha (h_line : line_l_parametric t α) (h_curve : curve_C_polar) (h_intersect : intersect_conditions) :
  α = (π / 6) :=
sorry

theorem find_MN (h_AB : distance_AB) :
  let AB := 2 * Real.sqrt 3,
      angle := Real.pi / 6 in
  AB / Real.cos angle = 4 :=
sorry

end find_alpha_find_MN_l21_21533


namespace minimum_seated_people_to_mandate_adjacent_seating_l21_21747

theorem minimum_seated_people_to_mandate_adjacent_seating :
  ∃ n : ℕ, n = 75 ∧ ∀ seats : Fin 150 → bool,
    (∀ i : Fin (150 - 1), ¬(seats i = ff ∧ seats (i + 1) = ff)) →
    (∃ p : Fin 150 → bool, (∀ q : Fin 150, p q = tt → seats q = ff) ∧
     (∃ empty : Fin 150 → bool, (∀ q : Fin 150, empty q = tt ↔ seats q = ff) ∧
      (∃ m : ℕ, m = n ∧ (∀ r : Fin 150, empty r = ff → p r = tt ∨ (∃ s : Fin 149, (p r = ff ∧ p (s + 1) = ff)))))) :=
sorry

end minimum_seated_people_to_mandate_adjacent_seating_l21_21747


namespace perfect_squares_count_l21_21508

theorem perfect_squares_count : (finset.filter (λ n, n * n ≥ 100 ∧ n * n ≤ 500) (finset.range 23)).card = 13 :=
by
  sorry

end perfect_squares_count_l21_21508


namespace always_quadratic_radical_l21_21354

theorem always_quadratic_radical (x : ℝ) : (sqrt (x^2 + 1)).is_real :=
by
  sorry

end always_quadratic_radical_l21_21354


namespace polynomial_no_extreme_points_l21_21169

variable {R : Type*} [CommRing R]

theorem polynomial_no_extreme_points
  (f : Polynomial R) (a b : R) :
  (∃ (x : R), f.derivative.eval x = 0 ∧ x ∈ Set.Icc a b) → False := sorry

end polynomial_no_extreme_points_l21_21169


namespace problem_part_1_problem_part_2_l21_21035
open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem problem_part_1 : A 3 ∪ B = {x | 1 < x ∧ x ≤ 7} := 
  by
  sorry

theorem problem_part_2 : (∀ a : ℝ, A a ∪ B = B → 2 < a ∧ a < sqrt 7) :=
  by 
  sorry

end problem_part_1_problem_part_2_l21_21035


namespace find_n_l21_21251

-- Define the given conditions
def condition1 (n : ℕ) : Prop :=
  factorial 5 / factorial (5 - n) = 60

-- Define the theorem to be proved
theorem find_n : ∃ n : ℕ, condition1 n ∧ n = 3 :=
by
  sorry

end find_n_l21_21251


namespace problem_AC_length_l21_21528

-- Definitions used in the conditions
variables (A B C D E : Type)
variables [has_dist A B] [has_dist A C] [has_dist B C]
variables [has_dist D E] [has_dist D B] [has_dist C E]
variables [has_add (dist A B)] [has_add (dist D E)] [has_add (dist C E)]

-- Given conditions
def bisects (AD AE : Type) (angle BAC : Type) := sorry
def length (BD DE EC : ℝ) := (BD = 4) ∧ (DE = 6) ∧ (EC = 9)

-- Target definition
def length_AC (AC : ℝ) : Prop := AC = 19

-- The Lean statement
theorem problem_AC_length (AC : ℝ) 
  (h1 : bisects AD AE BAC)
  (h2 : length 4 6 9) : 
  length_AC AC :=
sorry

-- To make sure the Lean code can be checked
#print axioms problem_AC_length

end problem_AC_length_l21_21528


namespace christen_potatoes_peeled_l21_21898

theorem christen_potatoes_peeled :
  ∀ (rate_homer rate_christen potatoes_initial peel_time_homer),
  rate_homer = 4 →
  rate_christen = 6 →
  potatoes_initial = 60 →
  peel_time_homer = 6 →
  let peeled_homer := rate_homer * peel_time_homer in
  let remaining_potatoes := potatoes_initial - peeled_homer in
  let combined_rate := rate_homer + rate_christen in
  let combined_time := remaining_potatoes / combined_rate in
  let peeled_christen := rate_christen * combined_time in
  peeled_christen = 22 :=
by
  intros rate_homer rate_christen potatoes_initial peel_time_homer
         rate_homer_eq rate_christen_eq potatoes_initial_eq peel_time_homer_eq
  let peeled_homer := rate_homer * peel_time_homer
  let remaining_potatoes := potatoes_initial - peeled_homer
  let combined_rate := rate_homer + rate_christen
  let combined_time := remaining_potatoes / combined_rate
  let peeled_christen := rate_christen * combined_time
  show peeled_christen = 22 from sorry

end christen_potatoes_peeled_l21_21898


namespace fishing_moratorium_main_purpose_l21_21026

def fishing_moratorium_purpose (A B C D : Prop) : Prop :=
  D

axiom fishing_moratorium_conditions (R : Prop) (A B C D : Prop) : Prop :=
  R → (A ∨ B ∨ C ∨ D)

theorem fishing_moratorium_main_purpose (R A B C D : Prop) :
  fishing_moratorium_conditions R A B C D →
  fishing_moratorium_purpose A B C D = D :=
by
  intros
  sorry

end fishing_moratorium_main_purpose_l21_21026


namespace length_CF_l21_21123

-- Definitions based on the given conditions
def isosceles_trapezoid (A B C D : Type) :=
  ∀ (AD BC AB DC : ℕ), AD = 3 ∧ BC = 3 ∧ AB = 2 ∧ DC = 8

def midpoint_of_leg_in_right_triangle (B D E : Type) :=
  ∀ x y, (x + y) / 2 = DE

-- The theorem we need to prove
theorem length_CF {A B C D E F : Type}
  (isosceles: isosceles_trapezoid A B C D)
  (midpoint_B: midpoint_of_leg_in_right_triangle B D E)
  : CF = 2 :=
sorry

end length_CF_l21_21123


namespace boris_needs_to_climb_four_times_l21_21513

/-- Hugo's mountain elevation is 10,000 feet above sea level. --/
def hugo_mountain_elevation : ℕ := 10_000

/-- Boris' mountain is 2,500 feet shorter than Hugo's mountain. --/
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2_500

/-- Hugo climbed his mountain 3 times. --/
def hugo_climbs : ℕ := 3

/-- The total number of feet Hugo climbed. --/
def total_hugo_climb : ℕ := hugo_mountain_elevation * hugo_climbs

/-- The number of times Boris needs to climb his mountain to equal Hugo's climb. --/
def boris_climbs_needed : ℕ := total_hugo_climb / boris_mountain_elevation

theorem boris_needs_to_climb_four_times :
  boris_climbs_needed = 4 :=
by
  sorry

end boris_needs_to_climb_four_times_l21_21513


namespace part1_part2_part3_l21_21591

def S (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2
def a (n : ℕ) : ℕ := 3 ^ n
def b (n : ℕ) : ℕ := 2 * a n / (a n - 2)^2
def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

theorem part1 (n : ℕ) : ∀ n, a n = 3^n := by
  sorry

theorem part2 : ∀ n, (S (2 * n) + 15) / a n ≥ 9 := by
  sorry

theorem part3 (n : ℕ) : T n < 13 / 2 := by
  sorry

end part1_part2_part3_l21_21591


namespace even_of_even_square_sqrt_two_irrational_l21_21310

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l21_21310


namespace sum_of_x_and_y_l21_21922

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 :=
by
  sorry

end sum_of_x_and_y_l21_21922


namespace intersection_A_B_l21_21069

-- Conditions
def A : Set ℝ := {1, 2, 0.5}
def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x^2}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1} :=
sorry

end intersection_A_B_l21_21069


namespace log_problem_solution_l21_21083

theorem log_problem_solution :
  ∀ (x : ℝ), (log (3 * x) 125 = x) → (∃ (a b : ℚ), x = a / b ∧ ¬ (∃ n : ℤ, x = n ^ 2) ∧ ¬ (∃ n : ℤ, x = n ^ 3)) := 
by
  sorry

end log_problem_solution_l21_21083


namespace Angstadt_seniors_l21_21613

theorem Angstadt_seniors (total_students : ℕ)
  (stat_percentage : ℚ) (geom_percentage : ℚ) (calc_percentage : ℚ)
  (stat_senior_percentage : ℚ) (geom_senior_percentage : ℚ) (calc_senior_percentage : ℚ)
  (h_total_students : total_students = 240) 
  (h_stat_percentage : stat_percentage = 0.5) 
  (h_geom_percentage : geom_percentage = 0.3) 
  (h_calc_percentage : calc_percentage = 0.2) 
  (h_stat_senior_percentage : stat_senior_percentage = 0.9) 
  (h_geom_senior_percentage : geom_senior_percentage = 0.6) 
  (h_calc_senior_percentage : calc_senior_percentage = 0.8) : 
  let stats_students := total_students * stat_percentage,
      geom_students := total_students * geom_percentage,
      calc_students := total_students * calc_percentage,
      seniors_stat := stats_students * stat_senior_percentage,
      seniors_geom := geom_students * geom_senior_percentage,
      seniors_calc := calc_students * calc_senior_percentage in 
  (seniors_stat = 108) ∧ (seniors_geom = 43) ∧ (seniors_calc = 38) :=
by
  sorry

end Angstadt_seniors_l21_21613


namespace shaded_area_is_46_point_4_l21_21564

-- Define the dimensions and properties of the circles and rectangle
def radius_small_circle : ℝ := 3
def radius_large_circle : ℝ := 6
def width_rectangle : ℝ := 12
def length_rectangle : ℝ := 18

-- Define the areas based on the given dimensions
def area_rectangle : ℝ := width_rectangle * length_rectangle
def area_large_circle : ℝ := real.pi * (radius_large_circle ^ 2)
def area_small_circle : ℝ := real.pi * (radius_small_circle ^ 2)
def area_two_small_circles : ℝ := 2 * area_small_circle

-- Define the total shaded area
def area_shaded : ℝ := area_rectangle - (area_large_circle + area_two_small_circles)

-- Prove that the shaded area is approximately 46.4 square feet
theorem shaded_area_is_46_point_4 : abs (area_shaded - 46.4) < 0.1 := 
by 
  -- Proof omitted 
  sorry

end shaded_area_is_46_point_4_l21_21564


namespace point_distance_to_y_axis_l21_21536

def point := (x : ℝ , y : ℝ)

def distance_to_y_axis (p : point) : ℝ :=
  |p.1|

theorem point_distance_to_y_axis (P : point):
  P = (-3, 4) →
  distance_to_y_axis P = 3 :=
by
  intro h
  rw [h, distance_to_y_axis]
  sorry

end point_distance_to_y_axis_l21_21536


namespace distance_focus_parabola_to_hyperbola_asymptotes_l21_21119

theorem distance_focus_parabola_to_hyperbola_asymptotes :
  let F := (2, 0)
  let hyperbola_asymptote1 := λ x y : ℝ, 3 * x + 4 * y = 0
  let hyperbola_asymptote2 := λ x y : ℝ, 3 * x - 4 * y = 0
  let distance := λ (F : ℝ × ℝ) (line : ℝ → ℝ → Prop), 
    |3 * F.1 + 4 * F.2| / Real.sqrt (3 ^ 2 + (-4) ^ 2)
  distance F (λ x y, 3 * x + 4 * y = 0) = 6 / 5 := by sorry

end distance_focus_parabola_to_hyperbola_asymptotes_l21_21119


namespace factorial_prime_exponent_l21_21764

noncomputable def exponent_of_prime (n p : ℕ) : ℕ :=
  let rec aux (n power : ℕ) : ℕ :=
    if n = 0 then 0 else n / power + aux (n / power) power
  aux n p

theorem factorial_prime_exponent :
  (exponent_of_prime 100 2 = 97) ∧
  (exponent_of_prime 100 3 = 48) ∧
  (exponent_of_prime 100 5 = 24) ∧
  (exponent_of_prime 100 7 = 16) :=
by
  sorry

end factorial_prime_exponent_l21_21764


namespace circle_eq_tangent_lines_l21_21471

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 2)

-- Define the midpoint (center) of the circle
def mid (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the circle C using the midpoint and radius
def circle_center : ℝ × ℝ := mid A B
def circle_radius : ℝ := Math.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2
def circle_eqn (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Define point M
def M : ℝ × ℝ := (3, 1)

-- Define the tangent lines
def tangent_line1 (x y : ℝ) : Prop := x - 3 = 0
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 5 = 0

-- The main statements to prove
theorem circle_eq (x y : ℝ) :
  circle_eqn x y ↔ (x - 1)^2 + (y - 2)^2 = 4 := by sorry

theorem tangent_lines (x y : ℝ) :
  ∃ tl, (tl = tangent_line1 x y ∨ tl = tangent_line2 x y) := by sorry

end circle_eq_tangent_lines_l21_21471


namespace sum_of_possible_values_of_y_l21_21277

-- Definitions of the conditions
variables (y : ℝ)
-- Angle measures in degrees
variables (a b c : ℝ)
variables (isosceles : Bool)

-- Given conditions
def is_isosceles_triangle (a b c : ℝ) (isosceles : Bool) : Prop :=
  isosceles = true ∧ (a = b ∨ b = c ∨ c = a)

-- Sum of angles in any triangle
def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180

-- Main statement to be proven
theorem sum_of_possible_values_of_y (y : ℝ) (a b c : ℝ) (isosceles : Bool) :
  is_isosceles_triangle a b c isosceles →
  sum_of_angles_in_triangle a b c →
  ((y = 60) → (a = y ∨ b = y ∨ c = y)) →
  isosceles = true → a = 60 ∨ b = 60 ∨ c = 60 →
  y + y + y = 180 :=
by
  intros h1 h2 h3 h4 h5
  sorry  -- Proof will be provided here

end sum_of_possible_values_of_y_l21_21277


namespace kim_boxes_on_thursday_l21_21164

theorem kim_boxes_on_thursday (Tues Wed Thurs : ℕ) 
(h1 : Tues = 4800)
(h2 : Tues = 2 * Wed)
(h3 : Wed = 2 * Thurs) : Thurs = 1200 :=
by
  sorry

end kim_boxes_on_thursday_l21_21164


namespace parallelogram_diagonal_length_l21_21645

-- Define a structure to represent a parallelogram
structure Parallelogram :=
  (side_length : ℝ) 
  (diagonal_length : ℝ)
  (perpendicular : Bool)

-- State the theorem about the relationship between the diagonals in a parallelogram
theorem parallelogram_diagonal_length (a b : ℝ) (P : Parallelogram) (h₀ : P.side_length = a) (h₁ : P.diagonal_length = b) (h₂ : P.perpendicular = true) : 
  ∃ (AC : ℝ), AC = Real.sqrt (4 * a^2 + b^2) :=
by
  sorry

end parallelogram_diagonal_length_l21_21645


namespace chess_team_selection_l21_21930

theorem chess_team_selection (total_members siblings : ℕ) 
    (members_choose selected_choose remaining_siblings_choose : ℕ) 
    (H_total : total_members = 18)
    (H_siblings : siblings = 4)
    (H_members_choose : members_choose = Nat.choose 18 8)
    (H_selected_choose : selected_choose = Nat.choose 14 4 * Nat.choose 4 4)
    (H_remaining_siblings_choose : remaining_siblings_choose = selected_choose)
    (H_calculation : members_choose - remaining_siblings_choose = 42757) : 
    ∃ num_ways, num_ways = 42757 :=
by 
  use 42757
  sorry

end chess_team_selection_l21_21930


namespace sufficient_condition_for_perpendicularity_l21_21073

noncomputable def is_perpendicular 
(lines : Type) [has_perp lines]
(l m : lines) : Prop :=
perp l m

noncomputable def are_intersecting_planes
(planes : Type) [has_inter_line planes]
(α β : planes) : lines := 
inter_line α β

noncomputable def is_line_in_plane
(planes : Type) [has_line_subset planes]
(m : lines) (β : planes) : Prop :=
line_subset m β

noncomputable def is_line_perpendicular_to_plane
(planes : Type) [has_perp_to_plane planes]
(m : lines) (α : planes) : Prop :=
perp_to_plane m α

variable (lines : Type)
variable (planes : Type)
variable [has_perp lines]
variable [has_inter_line planes]
variable [has_line_subset planes]
variable [has_perp_to_plane planes]

variable (l m : lines)
variable (α β : planes)

theorem sufficient_condition_for_perpendicularity :
  (are_intersecting_planes planes α β = l) →
  (is_line_in_plane planes m β) →
  (is_line_perpendicular_to_plane planes m α) →
  is_perpendicular lines l m :=
sorry

end sufficient_condition_for_perpendicularity_l21_21073


namespace numberOfWaysToPlaceCoinsSix_l21_21241

def numberOfWaysToPlaceCoins (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * numberOfWaysToPlaceCoins (n - 1)

theorem numberOfWaysToPlaceCoinsSix : numberOfWaysToPlaceCoins 6 = 32 :=
by
  sorry

end numberOfWaysToPlaceCoinsSix_l21_21241


namespace number_of_lattice_points_on_segment_l21_21329

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  ∃ x y : ℤ, p = (x, y)

def on_line_segment (start end p : ℤ × ℤ) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ 
           ((1 - t) • start.1 + t • end.1 = p.1) ∧ 
           ((1 - t) • start.2 + t • end.2 = p.2)

theorem number_of_lattice_points_on_segment : 
  let start := (3, 17)
  let end := (48, 281)
  ∃! (lattice_points : Finset (ℤ × ℤ)), 
    (∀ p ∈ lattice_points, is_lattice_point p ∧ on_line_segment start end p) ∧
    lattice_points.card = 4 := 
by
  sorry

end number_of_lattice_points_on_segment_l21_21329


namespace total_students_correct_l21_21636

def num_first_graders : ℕ := 358
def num_second_graders : ℕ := num_first_graders - 64
def total_students : ℕ := num_first_graders + num_second_graders

theorem total_students_correct : total_students = 652 :=
by
  sorry

end total_students_correct_l21_21636


namespace domain_z_l21_21800

noncomputable def z (x : ℝ) : ℝ := real.sqrt (x - 5)^4 + real.sqrt (x + 1)

theorem domain_z : ∀ x, (5 ≤ x → (∃ y, z x = y)) :=
by
  intro x
  split
  intro hx
  use z x
  sorry

end domain_z_l21_21800


namespace proof_problem_l21_21776

-- Define complex numbers for the roots
def P (x : ℂ) := ∏ k in Finset.range 15, (x - complex.exp (2 * real.pi * k * complex.I / 17))

def Q (x : ℂ) := ∏ j in Finset.range 12, (x - complex.exp (2 * real.pi * j * complex.I / 13))

-- Conditions as Lean definitions
noncomputable def e_k (k : ℕ) (h : k < 16) : ℂ := complex.exp (2 * real.pi * k * complex.I / 17)
noncomputable def e_j (j : ℕ) (h : j < 13) : ℂ := complex.exp (2 * real.pi * j * complex.I / 13)

theorem proof_problem : 
  (∏ k in Finset.range 15, ∏ j in Finset.range 12, (e_j j (by linarith) - e_k k (by linarith))) = 1 :=
sorry

end proof_problem_l21_21776


namespace proof_problem_l21_21778

-- Define complex numbers for the roots
def P (x : ℂ) := ∏ k in Finset.range 15, (x - complex.exp (2 * real.pi * k * complex.I / 17))

def Q (x : ℂ) := ∏ j in Finset.range 12, (x - complex.exp (2 * real.pi * j * complex.I / 13))

-- Conditions as Lean definitions
noncomputable def e_k (k : ℕ) (h : k < 16) : ℂ := complex.exp (2 * real.pi * k * complex.I / 17)
noncomputable def e_j (j : ℕ) (h : j < 13) : ℂ := complex.exp (2 * real.pi * j * complex.I / 13)

theorem proof_problem : 
  (∏ k in Finset.range 15, ∏ j in Finset.range 12, (e_j j (by linarith) - e_k k (by linarith))) = 1 :=
sorry

end proof_problem_l21_21778


namespace parallelogram_area_l21_21017

def vector_a : ℝ^3 := ⟨4, 2, -1⟩
def vector_b : ℝ^3 := ⟨2, -1, 5⟩

noncomputable def cross_product (u v : ℝ^3) : ℝ^3 :=
  let i := u.2.1 * v.2.2 - u.2.2 * v.2.1 in
  let j := u.2.2 * v.1 - u.1 * v.2.2 in
  let k := u.1 * v.2.1 - u.2.1 * v.1 in
  ⟨i, -j, k⟩

noncomputable def magnitude (v : ℝ^3) : ℝ :=
  real.sqrt(v.1^2 + v.2.1^2 + v.2.2^2)

theorem parallelogram_area : magnitude (cross_product vector_a vector_b) = real.sqrt 469 :=
by sorry

end parallelogram_area_l21_21017


namespace cot_difference_abs_eq_sqrt3_l21_21133

theorem cot_difference_abs_eq_sqrt3 
  (A B C D P : Point) (x y : ℝ) (h1 : is_triangle A B C) 
  (h2 : is_median A D B C) (h3 : ∠(D, A, P) = 60)
  (BD_eq_CD : BD = x) (CD_eq_x : CD = x)
  (BP_eq_y : BP = y) (AP_eq_sqrt3 : AP = sqrt(3) * (x + y))
  (cot_B : cot B = -y / ((sqrt 3) * (x + y)))
  (cot_C : cot C = (2 * x + y) / (sqrt 3 * (x + y))) 
  (x_y_neq_zero : x + y ≠ 0) :
  abs (cot B - cot C) = sqrt 3
  := sorry

end cot_difference_abs_eq_sqrt3_l21_21133


namespace chord_length_is_two_l21_21454

noncomputable def ellipse_chord_length (a b : ℝ) (angle : ℝ) : ℝ := 
  let c := real.sqrt (a ^ 2 - b ^ 2)
  let line_eq (x : ℝ) := (real.sqrt 3 / 3) * (x + 2 * c)
  let x1 := (-3 * real.sqrt 2 + real.sqrt ((3 * real.sqrt 2)^2 - 4 * 4 * 15)) / (2 * 4)
  let x2 := (-3 * real.sqrt 2 - real.sqrt ((3 * real.sqrt 2)^2 - 4 * 4 * 15)) / (2 * 4)
  real.sqrt((1 + 1 / 3) * ((x1 + x2)^2 - 4 * (x1 * x2)))

theorem chord_length_is_two : 
  ellipse_chord_length 3 1 (π / 6) = 2 :=
by
  sorry

end chord_length_is_two_l21_21454


namespace problem_statement_l21_21420

theorem problem_statement (x y : ℝ) (p : x > 0 ∧ y > 0) : (∃ p, p → xy > 0) ∧ ¬(xy > 0 → x > 0 ∧ y > 0) :=
by
  sorry

end problem_statement_l21_21420


namespace count_valid_four_digit_1_3_numbers_l21_21694

-- Definitions based on conditions
def is_valid_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def valid_four_digit_1_3 (n : ℕ) : Prop :=
  is_four_digit_number n ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 4 → is_valid_digit ((n / 10^k) % 10)) ∧
  (∃ i : ℕ, 0 ≤ i ∧ i < 4 ∧ (n / 10^i) % 10 = 1) ∧ 
  (∃ j : ℕ, 0 ≤ j ∧ j < 4 ∧ (n / 10^j) % 10 = 3)

theorem count_valid_four_digit_1_3_numbers :
  { n | valid_four_digit_1_3 n }.to_finset.card = 14 :=
by
  sorry

end count_valid_four_digit_1_3_numbers_l21_21694


namespace tangent_line_to_circle_l21_21867

-- Definition: Point on the circle
def point_on_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 = 2

-- Definition: Line equation
def line_eq (x0 y0 x y : ℝ) : ℝ :=
  x0 * x - y0 * y - 2

-- Theorem: Positional relationship
theorem tangent_line_to_circle (x0 y0 : ℝ) (h : point_on_circle x0 y0) :
  ∀ (x y : ℝ), x0 * x - y0 * y = 2 → ∃ (r : ℝ), r = sqrt 2 ∧ r = abs ((2 : ℝ) / sqrt (x0^2 + y0^2)) :=
sorry

end tangent_line_to_circle_l21_21867


namespace strawberries_jam_profit_l21_21763

noncomputable def betty_strawberries : ℕ := 25
noncomputable def matthew_strawberries : ℕ := betty_strawberries + 30
noncomputable def natalie_strawberries : ℕ := matthew_strawberries / 3  -- Integer division rounds down
noncomputable def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
noncomputable def strawberries_per_jar : ℕ := 12
noncomputable def jars_of_jam : ℕ := total_strawberries / strawberries_per_jar  -- Integer division rounds down
noncomputable def money_per_jar : ℕ := 6
noncomputable def total_money_made : ℕ := jars_of_jam * money_per_jar

theorem strawberries_jam_profit :
  total_money_made = 48 := by
  sorry

end strawberries_jam_profit_l21_21763


namespace average_of_groups_is_1010_l21_21633

-- Define the sequence of natural numbers from 1 to 2019
def seq : List ℕ := List.range' 1 2019

-- Calculate the sum of the sequence
def sum_seq : ℕ := Finset.sum (Finset.range 2020)

-- Define the number of groups
def num_groups : ℕ := 20

-- Define the average of each group
def avg := sum_seq / num_groups

-- Theorem statement
theorem average_of_groups_is_1010 (h1 : seq = List.range' 1 2019)
  (h2 : sum_seq = 2039190)
  (h3 : num_groups = 20) :
  avg = 1010 := by
  sorry

end average_of_groups_is_1010_l21_21633


namespace exposed_surface_area_hemisphere_l21_21733

-- Given conditions
def radius : ℝ := 10
def height_above_liquid : ℝ := 5

-- The attempt to state the problem as a proposition
theorem exposed_surface_area_hemisphere : 
  (π * radius ^ 2) + (π * radius * height_above_liquid) = 200 * π :=
by
  sorry

end exposed_surface_area_hemisphere_l21_21733


namespace steve_speed_back_home_l21_21635

-- Define a structure to hold the given conditions:
structure Conditions where
  home_to_work_distance : Float := 35 -- km
  v  : Float -- speed on the way to work in km/h
  additional_stop_time : Float := 0.25 -- hours
  total_weekly_time : Float := 30 -- hours

-- Define the main proposition:
theorem steve_speed_back_home (c: Conditions)
  (h1 : 5 * ((c.home_to_work_distance / c.v) + (c.home_to_work_distance / (2 * c.v))) + 3 * c.additional_stop_time = c.total_weekly_time) :
  2 * c.v = 18 := by
  sorry

end steve_speed_back_home_l21_21635


namespace distances_max_min_line_l1_intersection_product_l21_21128

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

noncomputable def line_l (ρ θ a : ℝ) : ℝ :=
  ρ * cos (θ - a)

-- Define the parametric equations of curve C1
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sqrt 3 * sin θ)

-- Define 'A' and 'l'
def A := polar_to_cartesian (4 * sqrt 2) (π / 4)
def l (ρ θ : ℝ) := ρ * cos (θ - π / 4) = 4 * sqrt 2

-- Define the general equation of curve C1 in rectangular coordinates
def general_eq_curve_C1 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define line l1 passing through point B(-2, 2) and its intersection with curve C1
def line_l1 (t : ℝ) : ℝ × ℝ :=
  (-2 + t * cos (3 * π / 4), 2 + t * sin (3 * π / 4))

theorem distances_max_min (θ : ℝ) :
  let x := 2 * cos θ,
      y := sqrt 3 * sin θ,
      d := abs ((2 * cos θ + sqrt 3 * sin θ - 8) / sqrt 2)
  in let d_max := (sqrt 14 + 8 * sqrt 2) / 2,
         d_min := (8 * sqrt 2 - sqrt 14) / 2
      in d = d_max ∨ d = d_min :=
sorry

theorem line_l1_intersection_product :
  let t1t2 := (7 / 2 : ℝ),
      result := 32 / 7
  in t1t2 = result :=
sorry

end distances_max_min_line_l1_intersection_product_l21_21128


namespace number_of_integers_between_cubes_l21_21481

theorem number_of_integers_between_cubes :
  let a := 10.4
  let b := 10.5
  let lower_bound := a ^ 3
  let upper_bound := b ^ 3
  let start := Int.ceil lower_bound
  let end_ := Int.floor upper_bound
  end_ - start + 1 = 33 :=
by
  have h1 : lower_bound = 1124.864 := by sorry
  have h2 : upper_bound = 1157.625 := by sorry
  have h3 : start = 1125 := by sorry
  have h4 : end_ = 1157 := by sorry
  sorry

end number_of_integers_between_cubes_l21_21481


namespace problem_solution_l21_21781

noncomputable def product(problem1: ℕ, problem2: ℕ): ℂ :=
∏ k in (finset.range problem1).image (λ (n : ℕ), e ^ (2 * π * complex.I * n / 17)),
  ∏ j in (finset.range problem2).image (λ (m : ℕ), e ^ (2 * π * complex.I * m / 13)),
    (j - k)

theorem problem_solution : product 15 12 = 13 := 
sorry

end problem_solution_l21_21781


namespace max_terms_fixed_in_valid_rearrangement_l21_21129

-- Definition of the sequence and rearrangement condition
def sequence := list ℕ
def original_sequence : sequence := list.range 100

def valid_rearrangement (s : sequence) : Prop :=
  s.length = 100 ∧ ∀ i, i < 99 → 
    abs (s[i] - s[i + 1]) = 1 ∨ 
    abs (s[i] % 10 - s[i + 1] % 10) = 1

-- Proof statement
theorem max_terms_fixed_in_valid_rearrangement:
  ∀ s : sequence,
    valid_rearrangement s →
    ∑ i in list.range 100, (if s[i] = original_sequence[i] then 1 else 0) ≤ 50 :=
by
  sorry

end max_terms_fixed_in_valid_rearrangement_l21_21129


namespace can_be_divided_into_6_triangles_l21_21833

-- Define the initial rectangle dimensions
def initial_rectangle_length := 6
def initial_rectangle_width := 5

-- Define the cut-out rectangle dimensions
def cutout_rectangle_length := 2
def cutout_rectangle_width := 1

-- Total area before the cut-out
def total_area : Nat := initial_rectangle_length * initial_rectangle_width

-- Cut-out area
def cutout_area : Nat := cutout_rectangle_length * cutout_rectangle_width

-- Remaining area after the cut-out
def remaining_area : Nat := total_area - cutout_area

-- The statement to be proved
theorem can_be_divided_into_6_triangles :
  remaining_area = 28 → (∃ (triangles : List (Nat × Nat × Nat)), triangles.length = 6) :=
by 
  intros h
  sorry

end can_be_divided_into_6_triangles_l21_21833


namespace distance_from_P_to_y_axis_l21_21249

theorem distance_from_P_to_y_axis (P : ℝ × ℝ) :
  (P.2 ^ 2 = -12 * P.1) → (dist P (-3, 0) = 9) → abs P.1 = 6 :=
by
  sorry

end distance_from_P_to_y_axis_l21_21249


namespace find_a2_find_general_formula_harmonic_sum_bound_l21_21990

-- Definitions and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i

axiom cond_a1 : ∀ (a : ℕ → ℝ), a 1 = 1
axiom cond_Sn : ∀ (a : ℕ → ℝ) (n : ℕ), n > 0 → 2 * S a n / n = a (n + 1) - (1 / 3) * n^2 - n - (2 / 3)

-- Problem Ⅰ
theorem find_a2 (a : ℕ → ℝ) : a 2 = 4 := sorry

-- Problem Ⅱ
theorem find_general_formula (a : ℕ → ℝ) : ∀ n : ℕ, n > 0 -> a n = n^2 := sorry

-- Problem Ⅲ
theorem harmonic_sum_bound (a : ℕ → ℝ) : ∀ n : ℕ, n > 0 -> (∑ i in finset.range (n + 1), (1 / a i)) < 7 / 4 := sorry

end find_a2_find_general_formula_harmonic_sum_bound_l21_21990


namespace smallest_c_for_inequality_l21_21008

theorem smallest_c_for_inequality :
  ∃ c : ℝ, c > 0 ∧ (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → (sqrt (x^2 * y^2) + c * |x^2 - y^2| ≥ (x^2 + y^2) / 2)) ∧ c = 1 / 2 :=
sorry

end smallest_c_for_inequality_l21_21008


namespace simplify_expression_l21_21231

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * y + 15 * y + 18 + 21 = 18 * x + 27 * y + 39 :=
by
  sorry

end simplify_expression_l21_21231


namespace unique_albums_count_l21_21759

open Set

structure Collections :=
  (andrew : Set ℕ)
  (john : Set ℕ)
  (samantha : Set ℕ)

def albums_unique_count (c : Collections) : ℕ :=
  (c.andrew \ c.john).card + (c.john \ c.andrew).card + (c.samantha \ c.andrew).card + (c.samantha \ c.john).card

theorem unique_albums_count (c : Collections) (h₁ : c.andrew.card = 23) 
  (h₂ : c.john.card = 20) 
  (h₃ : (c.andrew ∩ c.john).card = 12) 
  (h₄ : c.samantha.card = 15) 
  (h₅ : (c.samantha ∩ c.andrew).card = 3)
  (h₆ : (c.samantha ∩ c.john).card = 5)
  (h₇ : Disjoint (c.andrew ∩ c.john) (c.samantha ∩ c.andrew ∪ c.samantha ∩ c.john)) :
  albums_unique_count c = 26 := sorry

end unique_albums_count_l21_21759


namespace range_f_l21_21827
noncomputable theory

def g (x : ℝ) : ℝ := 3 / ((x - 2)^2 + 1)

def g2 (x : ℝ) : ℝ := g (g x)

def f (x : ℝ) : ℝ := g (g2 x)

theorem range_f : ∀ (x : ℝ), ∃ y ∈ set.Icc (3/50 : ℝ) 3, f x = y :=
by
  sorry

end range_f_l21_21827


namespace roots_of_quadratic_identity_l21_21921

namespace RootProperties

theorem roots_of_quadratic_identity (a b : ℝ) 
(h1 : a^2 - 2*a - 1 = 0) 
(h2 : b^2 - 2*b - 1 = 0) 
(h3 : a ≠ b) 
: a^2 + b^2 = 6 := 
by sorry

end RootProperties

end roots_of_quadratic_identity_l21_21921


namespace complex_square_areas_l21_21120

noncomputable def possible_areas_of_square (z : ℂ) : set ℝ :=
  { area : ℝ | ∃ (w1 w2 w3 : ℂ) (h1: w1 = z ∨ w1 = z^2 ∨ w1 = z^3)
    (h2: w2 = z ∨ w2 = z^2 ∨ w2 = z^3) (h3: w3 = z ∨ w3 = z^2 ∨ w3 = z^3) (h4: w1 ≠ w2 ∧ w2 ≠ w3 ∧ w1 ≠ w3)
    (h5: (w3 - w1) = (w2 - w1) * complex.I ∨ (w3 - w1) = (w2 - w1) * -complex.I),
    (complex.abs (w2 - w1))^2 }

theorem complex_square_areas (z : ℂ) (hz1 : z ≠ 0) (hz2 : z ≠ 1) :
  possible_areas_of_square z = {10, 2, 5/8} :=
sorry

end complex_square_areas_l21_21120


namespace boris_climbs_needed_l21_21515

-- Definitions
def elevation_hugo : ℕ := 10000
def shorter_difference : ℕ := 2500
def climbs_hugo : ℕ := 3

-- Derived Definitions
def elevation_boris : ℕ := elevation_hugo - shorter_difference
def total_climbed_hugo : ℕ := climbs_hugo * elevation_hugo

-- Theorem
theorem boris_climbs_needed : (total_climbed_hugo / elevation_boris) = 4 :=
by
  -- conditions and definitions are used here
  sorry

end boris_climbs_needed_l21_21515


namespace find_b_l21_21606

theorem find_b (b : ℝ) : (∃ x : ℝ, (x^3 - 3*x^2 = -3*x + b ∧ (3*x^2 - 6*x = -3))) → b = 1 :=
by
  intros h
  sorry

end find_b_l21_21606


namespace trig_identity_l21_21455

theorem trig_identity :
  let s60 := Real.sin (60 * Real.pi / 180)
  let c1 := Real.cos (1 * Real.pi / 180)
  let c20 := Real.cos (20 * Real.pi / 180)
  let s10 := Real.sin (10 * Real.pi / 180)
  s60 * c1 * c20 - s10 = Real.sqrt 3 / 2 - s10 :=
by
  sorry

end trig_identity_l21_21455


namespace sum_A_B_l21_21176

noncomputable def num_four_digit_odd_numbers_divisible_by_3 : ℕ := 1500
noncomputable def num_four_digit_multiples_of_7 : ℕ := 1286

theorem sum_A_B (A B : ℕ) :
  A = num_four_digit_odd_numbers_divisible_by_3 →
  B = num_four_digit_multiples_of_7 →
  A + B = 2786 :=
by
  intros hA hB
  rw [hA, hB]
  exact rfl

end sum_A_B_l21_21176


namespace proof_expression_l21_21627

def simplify_expression (x : ℝ) : ℝ :=
  (1 - 3 / (x + 2)) / ((x^2 - 2 * x + 1) / (3 * (x + 6)))

theorem proof_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  simplify_expression x = Real.sqrt 3 := by
  sorry

end proof_expression_l21_21627


namespace tan_A_tan_B_value_max_value_ab_sin_C_l21_21436

noncomputable def given_triangle_conditions (a b c : ℝ) (A B C : ℝ)
  (m n : ℝ × ℝ) : Prop :=
  ∃ (A B C : ℝ), 
  m = (1 - Real.cos (A + B), Real.cos ((A - B) / 2)) ∧
  n = (5 / 8, Real.cos ((A - B) / 2)) ∧
  (m.1 * n.1 + m.2 * n.2 = 9 / 8)

theorem tan_A_tan_B_value (a b c A B C : ℝ) (m n : ℝ × ℝ)
  (h : given_triangle_conditions a b c A B C m n) :
  Real.tan A * Real.tan B = 1 / 9 :=
sorry

theorem max_value_ab_sin_C (a b c A B C : ℝ) (m n : ℝ × ℝ)
  (h : given_triangle_conditions a b c A B C m n) :
  ∃ (M : ℝ), M = -3 / 8 ∧
  ∀ (A B : ℝ), max (a * b * Real.sin C / (a^2 + b^2 - c^2)) = M :=
sorry

end tan_A_tan_B_value_max_value_ab_sin_C_l21_21436


namespace solve_equation_l21_21016

theorem solve_equation (x : ℝ) (h : x > 6):
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ x ≥ 18 := 
by
  sorry

end solve_equation_l21_21016


namespace arithmetic_progr_property_l21_21427

theorem arithmetic_progr_property (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 + a 3 = 5 / 2)
  (h2 : a 2 + a 4 = 5 / 4)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h4 : a 3 = a 1 + 2 * (a 2 - a 1))
  (h5 : a 2 = a 1 + (a 2 - a 1)) :
  S 3 / a 3 = 6 := sorry

end arithmetic_progr_property_l21_21427


namespace chewing_gum_revenue_percentage_l21_21272

theorem chewing_gum_revenue_percentage :
  let initial_revenue_standard := 100000
  let initial_revenue_sugarfree := 150000
  let initial_revenue_bubble := 200000

  let projected_increase_standard := 0.30
  let projected_increase_sugarfree := 0.50
  let projected_increase_bubble := 0.40

  let actual_decrease_standard := 0.20
  let actual_decrease_sugarfree := 0.30
  let actual_decrease_bubble := 0.25

  let projected_revenue_standard := initial_revenue_standard * (1 + projected_increase_standard)
  let projected_revenue_sugarfree := initial_revenue_sugarfree * (1 + projected_increase_sugarfree)
  let projected_revenue_bubble := initial_revenue_bubble * (1 + projected_increase_bubble)
  let total_projected_revenue := projected_revenue_standard + projected_revenue_sugarfree + projected_revenue_bubble

  let actual_revenue_standard := initial_revenue_standard * (1 - actual_decrease_standard)
  let actual_revenue_sugarfree := initial_revenue_sugarfree * (1 - actual_decrease_sugarfree)
  let actual_revenue_bubble := initial_revenue_bubble * (1 - actual_decrease_bubble)
  let total_actual_revenue := actual_revenue_standard + actual_revenue_sugarfree + actual_revenue_bubble

  (total_actual_revenue / total_projected_revenue) * 100 ≈ 52.76 :=
begin
  sorry
end

end chewing_gum_revenue_percentage_l21_21272


namespace fraction_of_7000_l21_21705

theorem fraction_of_7000 (x : ℝ) 
  (h1 : (1 / 10 / 100) * 7000 = 7) 
  (h2 : x * 7000 - 7 = 700) : 
  x = 0.101 :=
by
  sorry

end fraction_of_7000_l21_21705


namespace perfect_squares_between_100_and_500_l21_21503

theorem perfect_squares_between_100_and_500 : 
  ∃ (count : ℕ), count = 12 ∧ 
    ∀ n : ℕ, (100 ≤ n^2 ∧ n^2 ≤ 500) ↔ (11 ≤ n ∧ n ≤ 22) :=
begin
  -- Proof goes here
  sorry
end

end perfect_squares_between_100_and_500_l21_21503


namespace intercept_sum_modulo_17_l21_21308

theorem intercept_sum_modulo_17 :
  ∀ (x y : ℤ), 0 ≤ x ∧ x < 17 → 0 ≤ y ∧ y < 17 → (7 * x ≡ 3 * y + 2 [MOD 17]) → 
  let x0 := natMod (10) 17 in
  let y0 := natMod (5) 17 in
  x0 + y0 = 15 :=
by
  intros x y x_bounds y_bounds H_congr
  sorry

end intercept_sum_modulo_17_l21_21308


namespace martingale_expected_value_product_martingale_expected_value_square_l21_21595

/-- Martingales xi and eta with initial conditions xi_1 = eta_1 = 0 -/
variables {Ω : Type*} {ℱ : Type*}
variables [probability_space Ω] 
variables (ℱₖ : ℕ → measurable_space Ω) (ξ η : ℕ → Ω → ℝ)

-- Assuming the martingales have these properties
def is_martingale (ξ : ℕ → Ω → ℝ) (ℱₖ : ℕ → measurable_space Ω) : Prop :=
∀ n, measurable (ξ n) (ℱₖ n) ∧ ∀ k ≤ n, measurable (conditional_expectation (ℱₖ k) (ξ n)) (ℱₖ k) = ξ k

def initial_conditions : Prop := ξ 1 = 0 ∧ η 1 = 0

-- Theorem: Expected value of the product of the terminal values of two martingales
theorem martingale_expected_value_product (hξ : is_martingale ξ ℱₖ) (hη : is_martingale η ℱₖ) (hinit : initial_conditions ξ η) :
  (∃ n, expectation (ξ n * η n)) = 
  ∑ k in finset.range n, expectation ((ξ k - ξ (k-1)) * (η k - η (k-1))) :=
sorry

-- Theorem: Expected value of the square of the terminal value of a martingale
theorem martingale_expected_value_square (hξ : is_martingale ξ ℱₖ) (hinit : initial_conditions ξ xi) :
  (∃ n, expectation (ξ n ^ 2)) = 
  ∑ k in finset.range n, expectation ((ξ k - ξ (k-1)) ^ 2) :=
sorry

end martingale_expected_value_product_martingale_expected_value_square_l21_21595


namespace john_volunteer_hours_l21_21960

noncomputable def total_volunteer_hours :=
  let first_six_months_hours := 2 * 3 * 6
  let next_five_months_hours := 1 * 2 * 4 * 5
  let december_hours := 3 * 2
  first_six_months_hours + next_five_months_hours + december_hours

theorem john_volunteer_hours : total_volunteer_hours = 82 := by
  sorry

end john_volunteer_hours_l21_21960


namespace frank_fence_length_l21_21832

theorem frank_fence_length (L W total_fence : ℝ) 
  (hW : W = 40) 
  (hArea : L * W = 200) 
  (htotal_fence : total_fence = 2 * L + W) : 
  total_fence = 50 := 
by 
  sorry

end frank_fence_length_l21_21832


namespace a_sqrt_plus_b_sqrt_eq_1_l21_21844

theorem a_sqrt_plus_b_sqrt_eq_1 (a b : ℝ) (h : a * sqrt (1 - b^2) + b * sqrt (1 - a^2) = 1) : a^2 + b^2 = 1 :=
sorry

end a_sqrt_plus_b_sqrt_eq_1_l21_21844


namespace intersecting_lines_l21_21255

theorem intersecting_lines (x y : ℝ) : x ^ 2 - y ^ 2 = 0 ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l21_21255


namespace find_a_l21_21534

-- Define the given problem
def curve := λ x : ℝ, x^2 - 2 * x - 3  -- the curve y = x^2 - 2x - 3

-- Define the intersection points
def points : set (ℝ × ℝ) := {p | (p = (0, -3)) ∨ (p = (-1, 0)) ∨ (p = (3, 0))}

-- Define the circle passing through the above points
def circle_eq (x y : ℝ): ℝ := x^2 + y^2 - 2 * x + 2 * y - 3

-- Prove that for the line to intersect the circle at points A and B such that AB = 2, a must be ±2√2
theorem find_a (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A ∈ {p | circle_eq p.1 p.2 = 0} ∧ B ∈ {p | circle_eq p.1 p.2 = 0} ∧ 
    (A.1 + A.2 + a = 0) ∧ (B.1 + B.2 + a = 0) ∧ dist A B = 2) →
  a = 2 * real.sqrt 2 ∨ a = -2 * real.sqrt 2 :=
by sorry

end find_a_l21_21534


namespace points_same_color_1m_apart_l21_21909

theorem points_same_color_1m_apart :
  ∀ (color : ℝ × ℝ → Prop), 
    (∀ x, ∀ y, (x ≠ y) → ∥x - y∥ = 1 → (color x = color y)) := 
by
  sorry

end points_same_color_1m_apart_l21_21909


namespace keith_spent_correctly_l21_21583

def packs_of_digimon_cards := 4
def cost_per_digimon_pack := 4.45
def cost_of_baseball_deck := 6.06
def total_spent := 23.86

theorem keith_spent_correctly :
  (packs_of_digimon_cards * cost_per_digimon_pack) + cost_of_baseball_deck = total_spent :=
sorry

end keith_spent_correctly_l21_21583


namespace min_inequality_l21_21174

theorem min_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 2) :
  ∃ L, L = 9 / 4 ∧ (1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ L) :=
sorry

end min_inequality_l21_21174


namespace find_period_l21_21265

variable (x : ℕ)
variable (theo_daily : ℕ := 8)
variable (mason_daily : ℕ := 7)
variable (roxy_daily : ℕ := 9)
variable (total_water : ℕ := 168)

theorem find_period (h : (theo_daily + mason_daily + roxy_daily) * x = total_water) : x = 7 :=
by
  sorry

end find_period_l21_21265


namespace polynomial_roots_property_l21_21982

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l21_21982


namespace magician_cannot_determine_polynomial_l21_21734

theorem magician_cannot_determine_polynomial (n : ℕ) (hn : 0 < n) 
  (x : Fin 2n → ℝ) (hx : ∀ i j, i < j → x i < x j)
  (P : ℝ[X]) (hdeg : P.degree ≤ n) : 
  ∃ Q : ℝ[X], Q.degree ≤ n ∧
    (∀ i : Fin 2n, eval (x i) P = eval (x i) Q) ∧ P ≠ Q := 
begin
  -- Placeholder for the actual proof
  sorry
end

end magician_cannot_determine_polynomial_l21_21734


namespace max_value_of_f_interval_of_monotonic_increasing_l21_21057

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 6) ^ 2 - sin x ^ 2

theorem max_value_of_f :
  ∃ (M : ℝ) (S : Set ℝ), 
    M = 1 / 2 ∧ 
    S = {x | ∃ (k : ℤ), x = π / 6 + k * π} ∧
    ∀ x, f x ≤ M := 
sorry

theorem interval_of_monotonic_increasing :
  ∃ (I : Set (Set ℝ)), 
    I = {I | ∃ (k : ℤ), I = Ioo (-π/3 + k * π) (π/6 + k * π)} :=
sorry

end max_value_of_f_interval_of_monotonic_increasing_l21_21057


namespace solve_quadratic_eq_l21_21630

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l21_21630


namespace num_integers_between_10_4_cubed_and_10_5_cubed_l21_21474

noncomputable def cube (x : ℝ) : ℝ := x^3

theorem num_integers_between_10_4_cubed_and_10_5_cubed :
  let lower_bound := 10.4
  let upper_bound := 10.5
  let lower_cubed := cube lower_bound
  let upper_cubed := cube upper_bound
  let num_integers := (⌊upper_cubed⌋₊ - ⌈lower_cubed⌉₊ + 1 : ℕ)
  lower_cubed = 1124.864 ∧ upper_cubed = 1157.625 → 
  num_integers = 33 := 
by
  intro lower_bound upper_bound lower_cubed upper_cubed num_integers h
  have : lower_cubed = 1124.864 := h.1
  have : upper_cubed = 1157.625 := h.2
  rw [this, this]
  exact 33

end num_integers_between_10_4_cubed_and_10_5_cubed_l21_21474


namespace min_value_of_function_l21_21403

noncomputable def func (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem min_value_of_function : ∃ x : ℝ, func x = 1 - Real.sqrt 2 :=
by sorry

end min_value_of_function_l21_21403


namespace calculate_distribution_l21_21372

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l21_21372


namespace visible_shaded_region_area_l21_21100

theorem visible_shaded_region_area :
  let grid_side := 10
  let total_area := grid_side * grid_side
  let small_circle_radius := 1
  let large_circle_radius := 3
  let area_small_circles := 4 * real.pi * (small_circle_radius ^ 2)
  let area_large_circle := real.pi * (large_circle_radius ^ 2)
  let area_circles := area_small_circles + area_large_circle
  let area_shaded := total_area - area_circles
  let A := total_area
  let B := 13 -- derived from the subtraction 100 - 13pi = area_shaded -> B = 13
in
  A + B = 113 := 
by
  repeat (assumption <|> apply sorry)

end visible_shaded_region_area_l21_21100


namespace campaign_funds_total_l21_21244

variable (X : ℝ)

def campaign_funds (friends family remaining : ℝ) : Prop :=
  friends = 0.40 * X ∧
  family = 0.30 * (X - friends) ∧
  remaining = X - (friends + family) ∧
  remaining = 4200

theorem campaign_funds_total (X_val : ℝ) (friends family remaining : ℝ)
    (h : campaign_funds X friends family remaining) : X = 10000 :=
by
  have h_friends : friends = 0.40 * X := h.1
  have h_family : family = 0.30 * (X - friends) := h.2.1
  have h_remaining : remaining = X - (friends + family) := h.2.2.1
  have h_remaining_amount : remaining = 4200 := h.2.2.2
  sorry

end campaign_funds_total_l21_21244


namespace brokerage_percentage_l21_21642

theorem brokerage_percentage (cash_realized : ℝ) (net_amount : ℝ) (brokerage_percentage : ℝ)
  (h₁ : cash_realized = 108.25)
  (h₂ : net_amount = 108)
  (h₃ : brokerage_percentage = (cash_realized - net_amount) / cash_realized * 100) :
  brokerage_percentage ≈ 0.2310 :=
by
  sorry

end brokerage_percentage_l21_21642


namespace MN_bisects_AC_l21_21603

variables {A B C H M N : Point}
variables {AC : Line}
variables (TriangleABC : Triangle A B C)
variables (orthocenterH : orthocenter H A B C)
variables (projM : is_projection M H (internal_angle_bisector B A C))
variables (projN : is_projection N H (external_angle_bisector B A C))

theorem MN_bisects_AC (midpoint_B1 : is_midpoint B1 A C) :
  bisects (line MN) A C :=
sorry

end MN_bisects_AC_l21_21603


namespace problem1_problem2_l21_21424

-- Part 1: Prove the range of m
theorem problem1 (m : ℝ) (h_m : m > 0)
  (h_p : ∀ x : ℝ, (x + 2) * (x - 6) ≤ 0 → 2 - m ≤ x ∧ x ≤ 2 + m) :
  4 ≤ m :=
sorry

-- Part 2: Prove the range of x
theorem problem2 (x : ℝ) (m : ℝ) (h_m : m = 5)
  (h_p_q : (∀ x : ℝ, (x + 2) * (x - 6) ≤ 0) ∨ (2 - m ≤ x ∧ x ≤ 2 + m))
  (h_not_both : ¬((∀ x : ℝ, (x + 2) * (x - 6) ≤ 0) ∧ (2 - m ≤ x ∧ x ≤ 2 + m))) :
  (x ∈ Iio (-2) ∪ Ioi (6)) :=
sorry

end problem1_problem2_l21_21424


namespace isosceles_triangle_perimeter_l21_21112

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 4) (isosceles : (a = b) ∨ (a = 2) ∨ (b = 2)) :
  (a = 2 ∧ b = 4 → 10) :=
begin
  -- assuming isosceles triangle means either two sides are equal or a = 2 or b = 2 which fits the isosceles definition in the context of provided lengths.
  sorry
end

end isosceles_triangle_perimeter_l21_21112


namespace even_function_and_inverse_property_l21_21459

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem even_function_and_inverse_property (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  f (-x) = f x ∧ f (1 / x) = -f x := by
  sorry

end even_function_and_inverse_property_l21_21459


namespace find_t_l21_21875

noncomputable def minimum_value_of_MT (t : ℝ) : ℝ :=
  if 0 < t ∧ t < 3 then
    1 -- given in the problem statement
  else
    0 -- just for completeness

theorem find_t (t : ℝ) (x y : ℝ) (h1 : 0 < t) (h2 : t < 3) 
  (h3 : sqrt ((x - sqrt 5) ^ 2 + y ^ 2) + sqrt ((x + sqrt 5) ^ 2 + y ^ 2) = 6)
  (h4 : minimum_value_of_MT t = 1) :
  t = 2 :=
sorry

end find_t_l21_21875


namespace Anchuria_min_crooks_l21_21540

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l21_21540


namespace ribbon_cutting_time_and_remaining_length_l21_21000

theorem ribbon_cutting_time_and_remaining_length (length_of_ribbon : ℕ) 
                                                 (increments : ℕ) 
                                                 (decrements : ℕ) 
                                                 (initial_time : ℕ) 
                                                 : 
-- Assuming the pattern alternates every two steps in the form (a, b)
-- and repeats multiple times as mentioned in the problem
-- We assume the time alternates by adding increments and then decrements 
-- to half the pairs resulting in an overall step of increments -> decrements pairs
  length_of_ribbon = 200 → increments = 15 → decrements = 5 → initial_time = 10 →
let pair1_time := 10 + 25,
    pair2_time := 30 + 10,
    pairs := 100 in
  (pairs * (pair1_time + pair2_time)) = 3750 ∧ 
  let total_time := (pairs * (pair1_time + pair2_time)) in
    total_time / 2 = 1875 →
    let cut_pairs_half_time := (total_time / 2) ÷ (pair1_time + pair2_time) in
    let cut_length_half_time := cut_pairs_half_time * 2 in
    length_of_ribbon - cut_length_half_time = 150 :=
by
  intros;
  sorry

end ribbon_cutting_time_and_remaining_length_l21_21000


namespace perfect_squares_count_l21_21510

theorem perfect_squares_count : (finset.filter (λ n, n * n ≥ 100 ∧ n * n ≤ 500) (finset.range 23)).card = 13 :=
by
  sorry

end perfect_squares_count_l21_21510


namespace probability_boxes_l21_21682

def box_A_tiles := (Finset.range 30).map (λ n, n+1)
def box_B_tiles := (Finset.range 20).map (λ n, n+21)

def prob_box_A_less_20 : nnreal :=
let favorable_A := (Finset.range 20).map (λ n, n+1) in
(favorable_A.card : nnreal) / (box_A_tiles.card : nnreal)

def prob_box_B_odd_or_greater_35 : nnreal :=
let odd_B := (Finset.range 10).filter (λ n, (n+21) % 2 = 1) in
let greater_35 := (Finset.range (40-35+1)).map (λ n, n+35) in
((odd_B.card + greater_35.card - odd_B.bUnion (λ n, if n + 21 >= 35 then {n+21} else ∅ ).card) : nnreal) / (box_B_tiles.card : nnreal)

theorem probability_boxes :
  prob_box_A_less_20 * prob_box_B_odd_or_greater_35 = 19 / 50 := by
sorry

end probability_boxes_l21_21682


namespace paintable_area_l21_21958

def length : ℝ := 14
def width : ℝ := 11
def height : ℝ := 9
def bedrooms : ℝ := 4
def unpainted_area_per_room : ℝ := 80

theorem paintable_area :
  let wall_area_one_room := 2 * (length * height) + 2 * (width * height)
  let paintable_area_one_room := wall_area_one_room - unpainted_area_per_room
  let total_paintable_area := bedrooms * paintable_area_one_room
  total_paintable_area = 1480 := by
  sorry

end paintable_area_l21_21958


namespace exists_triangle_l21_21745

variables {α : Type*} [linear_ordered_field α]

-- Definition: A convex polygon
structure convex_polygon (vertices : list (α × α)) : Prop :=
(convex : is_convex vertices)

-- Assumption: A square divided into convex polygons
structure square_division :=
(polygons : set (list (α × α))) -- set of polygons represented with lists of vertices
(square_decomposed : 
  ∃ (s : convex_polygon [A, B, C, D]), 
    (all polygons are convex polygons)
    ∧ (union of all polygons equals the square s))
(distinct_sides : ∀ {p1 p2 : list (α × α)} (h1 : p1 ∈ polygons) (h2 : p2 ∈ polygons), 
  p1 ≠ p2 → num_sides p1 ≠ num_sides p2)

-- Prove that there exists a triangular polygon in the division.
theorem exists_triangle (d : square_division) : 
  ∃ (p : list (α × α)), p ∈ d.polygons ∧ num_sides p = 3 :=
sorry

end exists_triangle_l21_21745


namespace isosceles_triangle_perimeter_l21_21109

noncomputable theory

def is_isosceles_triangle (a b c : ℝ) : Prop := 
  a = b ∨ b = c ∨ a = c

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : is_isosceles_triangle a b 4) (h_valid : is_valid_triangle a b 4) :
  perimeter a b 4 = 10 :=
  sorry

end isosceles_triangle_perimeter_l21_21109


namespace math_crackers_initial_l21_21191

def crackers_initial (gave_each : ℕ) (left : ℕ) (num_friends : ℕ) : ℕ :=
  (gave_each * num_friends) + left

theorem math_crackers_initial :
  crackers_initial 7 17 3 = 38 :=
by
  -- The definition of crackers_initial and the theorem statement should be enough.
  -- The exact proof is left as a sorry placeholder.
  sorry

end math_crackers_initial_l21_21191


namespace xy_condition_l21_21911

theorem xy_condition (x y z : ℝ) (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (posx : 0 < x) (posy : 0 < y) (posz : 0 < z) 
  (h : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : x / y = 2 :=
by
  sorry

end xy_condition_l21_21911


namespace solve_quadratic_l21_21631

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l21_21631


namespace minimum_crooks_l21_21545

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l21_21545


namespace two_digit_perfect_squares_divisible_by_3_l21_21901

theorem two_digit_perfect_squares_divisible_by_3 :
  ∃! n1 n2 : ℕ, (10 ≤ n1^2 ∧ n1^2 < 100 ∧ n1^2 % 3 = 0) ∧
               (10 ≤ n2^2 ∧ n2^2 < 100 ∧ n2^2 % 3 = 0) ∧
                (n1 ≠ n2) :=
by sorry

end two_digit_perfect_squares_divisible_by_3_l21_21901


namespace Anchuria_min_crooks_l21_21542

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l21_21542


namespace seq_properties_l21_21045

noncomputable def seq (r : ℕ) : ℕ → ℕ
| 1     := 1
| (n+1) := (n * seq n + 2 * (n + 1)^(2 * r)) / (n + 2)

theorem seq_properties (r : ℕ) (hr : 8 * 45 * r > 0):
  (∀ n, 0 < seq r n) ∧ ∀ n, seq r n % 2 = 0 ↔ n % 4 = 0 ∨ n % 4 = 3 :=
by
  sorry

end seq_properties_l21_21045


namespace number_of_cars_l21_21962

theorem number_of_cars (total_wheels cars_bikes trash_can tricycle roller_skates : ℕ) 
  (h1 : cars_bikes = 2) 
  (h2 : trash_can = 2) 
  (h3 : tricycle = 3) 
  (h4 : roller_skates = 4) 
  (h5 : total_wheels = 25) 
  : (total_wheels - (cars_bikes * 2 + trash_can * 2 + tricycle * 3 + roller_skates * 4)) / 4 = 3 :=
by
  sorry

end number_of_cars_l21_21962


namespace Lana_wins_3_games_l21_21242

theorem Lana_wins_3_games :
  ∀ (sus_wins sus_losses mike_wins mike_losses lana_losses : ℕ),
    sus_wins = 5 →
    sus_losses = 1 →
    mike_wins = 2 →
    mike_losses = 4 →
    lana_losses = 5 →
    (∃ (lana_wins : ℕ), 
      sus_wins + sus_losses = 6 ∧ 
      mike_wins + mike_losses = 6 ∧ 
      sus_wins + mike_wins + lana_wins = (17 + lana_wins) / 2 ∧
      lana_wins = 3) :=
begin
  intros,
  existsi 3,
  split, 
  { exact 6 },
  split,
  { exact 6 },
  split,
  { exact (5 + 2 + 3 = (17 + 3) / 2) },
  { refl }
end

end Lana_wins_3_games_l21_21242


namespace polygon_perimeter_bound_l21_21740

variables {n : ℕ} {S R : ℝ}

open_locale real

-- Define the conditions
def inscribed_polygon_perimeter (n : ℕ) (S R : ℝ) : Prop :=
  ∀ (vertices : fin n → ℝ) (marked_points : fin n → ℝ),
    ∃ (b : fin n → ℝ),
    (∀ i, 0 ≤ b i) ∧ (∑ i in finset.fin_range n, b i ≥ (2 * S / R))

-- State the theorem
theorem polygon_perimeter_bound (n : ℕ) (S R : ℝ) (hS : 0 ≤ S) (hR : 0 < R) :
  inscribed_polygon_perimeter n S R :=
begin
  sorry
end

end polygon_perimeter_bound_l21_21740


namespace ratio_of_areas_l21_21688

theorem ratio_of_areas (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : r₁ = (Real.sqrt 2) / 4)
  (h₂ : A₁ = π * r₁^2) (h₃ : r₂ = (Real.sqrt 2) * r₁) (h₄ : A₂ = π * r₂^2) :
  A₂ / A₁ = 2 :=
by
  sorry

end ratio_of_areas_l21_21688


namespace sum_solutions_of_system_l21_21181

theorem sum_solutions_of_system : 
  let solutions := [(x, y) | x y : ℝ, (|x - 5| = |y - 12| ∧ |x - 12| = 3 * |y - 5|)]
  in ∑ (xi,yi) in solutions, (xi + yi) = 3 :=
sorry

end sum_solutions_of_system_l21_21181


namespace hyperbola_conjugate_axis_length_l21_21258

theorem hyperbola_conjugate_axis_length :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 → (2 * real.sqrt 2) * 2 = 4 * real.sqrt 2 :=
by
  sorry

end hyperbola_conjugate_axis_length_l21_21258


namespace min_value_of_2gx_sq_minus_fx_l21_21074

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_of_2gx_sq_minus_fx (a b c : ℝ) (h_a_nonzero : a ≠ 0)
  (h_min_fx : ∃ x : ℝ, 2 * (f a b x)^2 - g a c x = 7 / 2) :
  ∃ x : ℝ, 2 * (g a c x)^2 - f a b x = -15 / 4 :=
sorry

end min_value_of_2gx_sq_minus_fx_l21_21074


namespace polynomial_factorization_l21_21912

variable (x y : ℝ)

theorem polynomial_factorization (m : ℝ) :
  (∃ (a b : ℝ), 6 * x^2 - 5 * x * y - 4 * y^2 - 11 * x + 22 * y + m = (3 * x - 4 * y + a) * (2 * x + y + b)) →
  m = -10 :=
sorry

end polynomial_factorization_l21_21912


namespace min_distance_from_ellipse_to_line_l21_21444

-- Define the given conditions
def polar_equation (ρ θ : Real) : Prop := 
  ρ * Real.cos (θ - Real.pi / 4) = 3 * Real.sqrt 2

def cartesian_equation (x y : Real) : Prop := 
  x + y - 6 = 0

def ellipse (x y : Real) : Prop := 
  x^2 / 16 + y^2 / 9 = 1

-- The proof problem in Lean
theorem min_distance_from_ellipse_to_line :
  (∀ (ρ θ : ℝ), polar_equation ρ θ → ∀ (x y : ℝ), cartesian_equation x y → 
    (∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ P.2 ≤ dist P.1 (x, y))) := sorry

end min_distance_from_ellipse_to_line_l21_21444


namespace athlete_B_more_stable_l21_21762

noncomputable def distribution_A := [(8, 0.3), (9, 0.2), (10, 0.5)]
noncomputable def distribution_B := [(8, 0.2), (9, 0.4), (10, 0.4)]

def expected_value (dist : List (ℕ × ℝ)) : ℝ :=
  dist.foldl (λ sum (k : ℕ × ℝ), sum + k.1 * k.2) 0

def variance (dist : List (ℕ × ℝ)) : ℝ :=
  let mean := expected_value dist
  dist.foldl (λ var (k : ℕ × ℝ), var + (k.1 - mean) ^ 2 * k.2) 0

theorem athlete_B_more_stable :
  variance distribution_B < variance distribution_A := 
  sorry

end athlete_B_more_stable_l21_21762


namespace perfect_squares_between_100_and_500_l21_21505

theorem perfect_squares_between_100_and_500 : 
  ∃ (count : ℕ), count = 12 ∧ 
    ∀ n : ℕ, (100 ≤ n^2 ∧ n^2 ≤ 500) ↔ (11 ≤ n ∧ n ≤ 22) :=
begin
  -- Proof goes here
  sorry
end

end perfect_squares_between_100_and_500_l21_21505


namespace greatest_odd_factors_under_150_l21_21198

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l21_21198


namespace rectangle_count_l21_21234

universe u v

def set_of_lines (n : ℕ) : Set (Fin n) := Set.univ

noncomputable def num_rectangles (H V : ℕ) : ℕ :=
  Nat.choose H 2 * Nat.choose V 2

theorem rectangle_count :
  num_rectangles 6 7 = 315 :=
by
  unfold num_rectangles
  simp
  norm_num
  exact Nat.factorial_ne_zero 4
  sorry

end rectangle_count_l21_21234


namespace sum_of_highest_powers_10_and_3_dividing_20_fact_l21_21388

theorem sum_of_highest_powers_10_and_3_dividing_20_fact (p10 p3 : ℕ) :
  (∃ p5 p2 : ℕ, p5 = (20 / 5).natFloor ∧ p2 = (20 / 2).natFloor + (20 / 4).natFloor + (20 / 8).natFloor + (20 / 16).natFloor ∧ p10 = min p2 p5) ∧
  (∃ q3 : ℕ, q3 = (20 / 3).natFloor + (20 / 9).natFloor ∧ p3 = q3) →
  p10 + p3 = 12 := 
by sorry

end sum_of_highest_powers_10_and_3_dividing_20_fact_l21_21388


namespace min_sum_f_l21_21889

def A := Finset.range 101
def f (i : ℕ) : ℕ := sorry  -- Define f i as required

axiom f_def : ∀ i, i ∈ A → f(f(i)) = 100
axiom f_bound : ∀ i, i ∈ A ∧ i < 100 → abs (f i - f (i + 1)) ≤ 1

theorem min_sum_f : ∑ i in A, f i = 8350 :=
by
  sorry

end min_sum_f_l21_21889


namespace angle_l1_l2_no_triangle_values_a_l21_21683

noncomputable def slope (a b c : ℝ) : ℝ := - (a / b)

def angle_between_lines (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  let slope1 := slope l1.1 l1.2 l1.3
  let slope2 := slope l2.1 l2.2 l2.3
  Real.arctan ((slope1 - slope2) / (1 + slope1 * slope2))

def values_of_a (l3 : ℝ → ℝ × ℝ × ℝ) : set ℝ :=
  {a | let l3' := l3 a
       let parallel_l1_l3 := l3'.1 = -4
       let parallel_l2_l3 := l3'.1 = 8/3
       let intersection := l3'.1 = 3
       l3'.1 = -4 ∨ l3'.1 = 8/3 ∨ l3'.1 = 3}

theorem angle_l1_l2 (l1 l2 : ℝ × ℝ × ℝ) (h1 : l1 = (2, -1, -10)) (h2 : l2 = (4, 3, -10)) :
  angle_between_lines l1 l2 = Real.arctan 2 := sorry

theorem no_triangle_values_a (l1 l2 : ℝ × ℝ × ℝ) (l3 : ℝ → ℝ × ℝ × ℝ)
  (h1 : l1 = (2, -1, -10)) (h2 : l2 = (4, 3, -10)) :
  values_of_a l3 = {-4, 8/3, 3} := sorry

end angle_l1_l2_no_triangle_values_a_l21_21683


namespace cot_diff_equal_l21_21148

variable (A B C D : Type)

-- Define the triangle and median.
variable [triangle ABC : Type] (median : Type)

-- Define the angle condition.
def angle_condition (ABC : triangle) (AD : median) : Prop :=
  ∠(AD, BC) = 60

-- Prove the cotangent difference
theorem cot_diff_equal
  (ABC : triangle)
  (AD : median)
  (h : angle_condition ABC AD) :
  abs (cot B - cot C) = (9 - 3 * sqrt 3) / 2 :=
by
  sorry -- Proof to be constructed

end cot_diff_equal_l21_21148


namespace complement_intersection_l21_21526

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  -- Proof is omitted.
  sorry

end complement_intersection_l21_21526


namespace exists_non_prime_sequence_l21_21177

theorem exists_non_prime_sequence (n : ℕ) : ∃ (a b : ℕ), a = b + n ∧ (∀ k, k ∈ set.Icc a b -> ¬ nat.prime k) :=
by
  sorry

end exists_non_prime_sequence_l21_21177


namespace trader_profit_l21_21300

theorem trader_profit (P : ℝ) :
  let buy_price := 0.80 * P
  let sell_price := 1.20 * P
  sell_price - P = 0.20 * P := 
by
  sorry

end trader_profit_l21_21300


namespace find_tank_capacity_l21_21714

noncomputable def tank_capacity (C : ℝ) : Prop :=
  let outlet_rate := C / 5 -- litres per hour
  let inlet_rate := 4 * 60 -- litres per hour
  let effective_rate := C / 8 -- litres per hour
  C / 5 - inlet_rate = C / 8

theorem find_tank_capacity : ∃ C : ℝ, tank_capacity C ∧ C = 3200 :=
by  { use 3200, 
      unfold tank_capacity, 
      simp, 
      norm_num,
      sorry }

end find_tank_capacity_l21_21714


namespace height_ratio_l21_21361

noncomputable def Anne_height := 80
noncomputable def Bella_height := 3 * Anne_height
noncomputable def Sister_height := Bella_height - 200

theorem height_ratio : Anne_height / Sister_height = 2 :=
by
  /-
  The proof here is omitted as requested.
  -/
  sorry

end height_ratio_l21_21361


namespace purchased_both_books_l21_21663

theorem purchased_both_books: 
  ∀ (A B AB C : ℕ), A = 2 * B → AB = 2 * (B - AB) → C = 1000 → C = A - AB → AB = 500 := 
by
  intros A B AB C h1 h2 h3 h4
  sorry

end purchased_both_books_l21_21663


namespace projections_properties_l21_21451

noncomputable def P (p q r : ℤ) (m n : ℕ) : Prop :=
  let u := p^m
  let v := q^n
  x^2 + y^2 = r^2 ∧ 
  r % 2 = 1 ∧
  u = p^m ∧
  v = q^n ∧
  u > v ∧
  dist(A, M) = 1 ∧ 
  dist(B, M) = 9 ∧ 
  dist(C, N) = 8 ∧ 
  dist(D, N) = 2

theorem projections_properties (p q r : ℤ) (m n : ℕ) (x y : ℝ) (A B C D M N : ℝ × ℝ) :
  P p q r m n → P x y r A B C D M N :=
sorry

end projections_properties_l21_21451


namespace tangent_line_eq_l21_21525

open Real

/-
Given a point P on the curve y = x * ln x, and the tangent line at P is perpendicular 
to the line x + y + 1 = 0, prove that the equation of this tangent line is x - y - 1 = 0.
-/
theorem tangent_line_eq :
  ∀ (P : ℝ × ℝ), P = (1, 0) ∧ (∃ x y, y = x * ln x ∧ y = x - 1) ∧ (1:ℝ) = 1 → P.1 - P.2 - 1 = 0 :=
begin
  sorry
end

end tangent_line_eq_l21_21525


namespace school_year_hours_per_week_l21_21356

-- Definitions based on the conditions of the problem
def summer_weeks : ℕ := 8
def summer_hours_per_week : ℕ := 40
def summer_earnings : ℕ := 3200

def school_year_weeks : ℕ := 24
def needed_school_year_earnings : ℕ := 6400

-- Question translated to a Lean statement
theorem school_year_hours_per_week :
  let hourly_rate := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := needed_school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks = (80 / 3) :=
by {
  -- The implementation of the proof goes here
  sorry
}

end school_year_hours_per_week_l21_21356


namespace quadrilateral_diagonal_parallel_side_l21_21617

variables (A B C D M N K L : Type) [affine_space A] [affine_space B] [affine_space C] 
          [affine_space D] [affine_space M] [affine_space N] [affine_space K] [affine_space L]

structure parallelogram (A B C D : Type) [affine_space A] [affine_space B] 
                        [affine_space C] [affine_space D] :=
(is_parallelogram : ∀ (x y : affine_space), (x, y) = (A, B) ∨ (x, y) = (B, C) ∨ (x, y) = (C, D) ∨ (x, y) = (D, A))

variables (AM AL BN DK AB AD BC CD x y z t a b : ℝ)

def area_half_s_abcd (MN KL ABCD : ℝ) := MN = KL ∧ MN = (1/2 : ℝ) * ABCD

theorem quadrilateral_diagonal_parallel_side 
         (p : parallelogram A B C D)
         (h : area_half_s_abcd (area (M N K L)) (area (A B C D))) :
  (x = t ∨ y = z) ∧ ((x = t → ∥ M - K ∥ ∥ D - A ∥) ∨ (y = z → ∥ L - N ∥ ∥ B - A)) := 
by { sorry }


end quadrilateral_diagonal_parallel_side_l21_21617


namespace point_A_is_minus_five_l21_21215

theorem point_A_is_minus_five 
  (A B C : ℝ)
  (h1 : A + 4 = B)
  (h2 : B - 2 = C)
  (h3 : C = -3) : 
  A = -5 := 
by 
  sorry

end point_A_is_minus_five_l21_21215


namespace days_gumballs_last_l21_21966

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end days_gumballs_last_l21_21966


namespace greatest_odd_factors_le_150_l21_21196

theorem greatest_odd_factors_le_150 : 
  let n := 144 in
  n < 150 ∧ (∃ k, k * k = n) :=
by
  sorry

end greatest_odd_factors_le_150_l21_21196


namespace solve_pos_int_a_l21_21816

theorem solve_pos_int_a :
  ∀ a : ℕ, (0 < a) →
  (∀ n : ℕ, (n ≥ 5) → ((2^n - n^2) ∣ (a^n - n^a))) →
  (a = 2 ∨ a = 4) :=
by
  sorry

end solve_pos_int_a_l21_21816


namespace comparison_problem_l21_21374

theorem comparison_problem :
  (-(+ 0.3 : ℝ)) > -(abs (- (1 / 3 : ℝ))) :=
by
  sorry

end comparison_problem_l21_21374


namespace christina_has_three_snakes_l21_21373

def snake_lengths : List ℕ := [24, 16, 10]

def total_length : ℕ := 50

theorem christina_has_three_snakes
  (lengths : List ℕ)
  (total : ℕ)
  (h_lengths : lengths = snake_lengths)
  (h_total : total = total_length)
  : lengths.length = 3 :=
by
  sorry

end christina_has_three_snakes_l21_21373


namespace max_gcd_value_l21_21761

theorem max_gcd_value (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k = 2 ∧ ∀ n, gcd (13 * m + 4) (7 * m + 2) ≤ 2 := by
  sorry

end max_gcd_value_l21_21761


namespace proof_problem_l21_21779

-- Define complex numbers for the roots
def P (x : ℂ) := ∏ k in Finset.range 15, (x - complex.exp (2 * real.pi * k * complex.I / 17))

def Q (x : ℂ) := ∏ j in Finset.range 12, (x - complex.exp (2 * real.pi * j * complex.I / 13))

-- Conditions as Lean definitions
noncomputable def e_k (k : ℕ) (h : k < 16) : ℂ := complex.exp (2 * real.pi * k * complex.I / 17)
noncomputable def e_j (j : ℕ) (h : j < 13) : ℂ := complex.exp (2 * real.pi * j * complex.I / 13)

theorem proof_problem : 
  (∏ k in Finset.range 15, ∏ j in Finset.range 12, (e_j j (by linarith) - e_k k (by linarith))) = 1 :=
sorry

end proof_problem_l21_21779


namespace combination_sum_property_l21_21754

theorem combination_sum_property :
  ∑ (k : ℕ) in finset.range 9, nat.choose (k + 2) 2 = nat.choose 11 3 :=
sorry

end combination_sum_property_l21_21754


namespace area_of_ade_l21_21948

noncomputable def area_of_triangle_abc (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def sin_of_angle_a (K a b : ℝ) : ℝ :=
  (2 * K) / (a * b)

noncomputable def area_of_triangle_ade (AD AE sinA : ℝ) : ℝ :=
  (1 / 2) * AD * AE * sinA

theorem area_of_ade :
  let AB : ℝ := 10
  let BC : ℝ := 12
  let AC : ℝ := 13
  let AD : ℝ := 3
  let AE : ℝ := 9
  let K : ℝ := area_of_triangle_abc AB BC AC
  let sinA : ℝ := sin_of_angle_a K AB AC
  area_of_triangle_ade AD AE sinA = 150206 / 6500 :=
by
  sorry

end area_of_ade_l21_21948


namespace problem_solution_l21_21782

noncomputable def product(problem1: ℕ, problem2: ℕ): ℂ :=
∏ k in (finset.range problem1).image (λ (n : ℕ), e ^ (2 * π * complex.I * n / 17)),
  ∏ j in (finset.range problem2).image (λ (m : ℕ), e ^ (2 * π * complex.I * m / 13)),
    (j - k)

theorem problem_solution : product 15 12 = 13 := 
sorry

end problem_solution_l21_21782


namespace diameter_of_cylinder_is_approx_3_39_l21_21248

noncomputable def radius_of_cylinder (V : ℝ) (h : ℝ) : ℝ :=
  real.sqrt (V / (real.pi * h))

noncomputable def diameter_of_cylinder (V : ℝ) (h : ℝ) : ℝ :=
  2 * radius_of_cylinder V h

theorem diameter_of_cylinder_is_approx_3_39 :
  diameter_of_cylinder 45 5 ≈ 3.39 :=
by
  sorry

end diameter_of_cylinder_is_approx_3_39_l21_21248


namespace five_card_derangement_l21_21214

open Finset

def derangement_count (n : ℕ) : ℕ :=
  (n.factorial * (((Finset.range (n + 1)).sum (λ k, (-1) ^ k / (k.factorial : ℚ)))).toRational.denominator : ℕ)

theorem five_card_derangement : derangement_count 5 = 44 := by
  sorry

end five_card_derangement_l21_21214


namespace all_vints_are_xaffs_l21_21929

variables
  (Zibbs : Type) (Xaffs : Type) (Yurns : Type) (Worbs : Type) (Vints : Type)
  (subset_Zibbs_Xaffs : Zibbs → Xaffs)
  (subset_Yurns_Xaffs : Yurns → Xaffs)
  (subset_Worbs_Zibbs : Worbs → Zibbs)
  (subset_Yurns_Worbs : Yurns → Worbs)
  (subset_Worbs_Vints : Worbs → Vints)
  (subset_Vints_Yurns : Vints → Yurns)

theorem all_vints_are_xaffs : ∀ v : Vints, ∃ x : Xaffs, x = subset_Vints_Yurns (subset_Yurns_Worbs (subset_Worbs_Zibbs (subset_Zibbs_Xaffs v))) :=
by
  sorry

end all_vints_are_xaffs_l21_21929


namespace third_term_is_18_l21_21430

-- Define the first term and the common ratio
def a_1 : ℕ := 2
def q : ℕ := 3

-- Define the function for the nth term of an arithmetic-geometric sequence
def a_n (n : ℕ) : ℕ :=
  a_1 * q^(n-1)

-- Prove that the third term is 18
theorem third_term_is_18 : a_n 3 = 18 := by
  sorry

end third_term_is_18_l21_21430


namespace minimum_crooks_l21_21556

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l21_21556


namespace distinct_integer_solutions_l21_21802

theorem distinct_integer_solutions :
  (∃! x : ℤ, |3 * x - |x + 2|| = 5) := sorry

end distinct_integer_solutions_l21_21802


namespace dishes_combinations_is_correct_l21_21772

-- Define the number of dishes
def num_dishes : ℕ := 15

-- Define the number of appetizers
def num_appetizers : ℕ := 5

-- Compute the total number of combinations
def combinations_of_dishes : ℕ :=
  num_dishes * num_dishes * num_appetizers

-- The theorem that states the total number of combinations is 1125
theorem dishes_combinations_is_correct :
  combinations_of_dishes = 1125 := by
  sorry

end dishes_combinations_is_correct_l21_21772


namespace meeting_time_l21_21299

-- Definitions for the problem conditions.
def track_length : ℕ := 1800
def speed_A_kmph : ℕ := 36
def speed_B_kmph : ℕ := 54

-- Conversion factor from kmph to mps.
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Calculate the speeds in mps.
def speed_A_mps : ℕ := kmph_to_mps speed_A_kmph
def speed_B_mps : ℕ := kmph_to_mps speed_B_kmph

-- Calculate the time to complete one lap for A and B.
def time_lap_A : ℕ := track_length / speed_A_mps
def time_lap_B : ℕ := track_length / speed_B_mps

-- Prove the time to meet at the starting point.
theorem meeting_time : (Nat.lcm time_lap_A time_lap_B) = 360 := by
  -- Skipping the proof with sorry placeholder
  sorry

end meeting_time_l21_21299


namespace find_diameter_of_field_l21_21018

noncomputable def diameter_of_field (total_cost : ℝ) (cost_per_meter : ℝ) : ℝ :=
  total_cost / Real.pi

theorem find_diameter_of_field :
  diameter_of_field 219.9114857512855 1 ≈ 70 := 
by
  unfold diameter_of_field
  norm_num
  sorry

end find_diameter_of_field_l21_21018


namespace days_gumballs_last_l21_21967

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end days_gumballs_last_l21_21967


namespace median_name_length_is_4_l21_21346

-- Define the survey conditions
def name_lengths : list ℕ :=
  [2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7]

-- Define a function to find the median
def median (l : list ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! ((sorted.length + 1) / 2 - 1)

-- Prove that the median length of the names is 4
theorem median_name_length_is_4 : median name_lengths = 4 :=
by
  sorry

end median_name_length_is_4_l21_21346


namespace evaluate_expression_l21_21394

theorem evaluate_expression :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = (3^1002) / (2^1000) :=
by sorry

end evaluate_expression_l21_21394


namespace solve_Bernoulli_l21_21569

theorem solve_Bernoulli (C₁ : ℝ) (x : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, 
    has_deriv_at y (-(y / x) + (1 / x ^ 2) * (1 / y ^ 2)) x ∧
    y x = y x) ∧
    y x = (√[3]((3 / (2 * x)) + (C₁ / x^3))) := sorry

end solve_Bernoulli_l21_21569


namespace intersection_of_sets_condition_l21_21096

theorem intersection_of_sets_condition {a b c : ℝ} 
  (h1 : a^2 + a + b * complex.I < 2 + c * complex.I)
  : (a^2 + a < 2) → 
    ((∀ x, -2 < x ∧ x < 1 ↔ -2 < x  ∧ x < 0 ∨ 0 < x  ∧ x < 1)) :=
by
  intro h2
  have h3 : b = 0 := sorry
  have h4 : c = 0 := sorry
  have h5 : ∀ x, -2 < x ∧ x < 1 := by sorry
  have h6 : ¬(0 = 0) := by sorry
  have h7 : ∀ x, ¬(x = 0) := by sorry
  have h8 : (∀ x, -2 < x ∧ x < 1 ↔ -2 < x  ∧ x < 0 ∨ 0 < x  ∧ x < 1) := by sorry
  exact h8

end intersection_of_sets_condition_l21_21096


namespace ellipse_equation_no_k_symmetric_B_exists_l21_21431

-- Define fundamental properties of the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Given Values
def focal_length : ℝ := 2 * Real.sqrt 2
def c : ℝ := Real.sqrt 2

noncomputable def pointA : ℝ × ℝ := (3 / 2, -1 / 2)

-- Derived values based on a^2 = b^2 + c^2
def a_squared : ℝ := 3
def b_squared : ℝ := 1
def equation_of_ellipse : Prop := ∀ x y, x^2 / 3 + y^2 = 1

-- Questions to prove
axiom focal_len : 2 * c = 2 * Real.sqrt 2

theorem ellipse_equation : equation_of_ellipse :=
by sorry

theorem no_k_symmetric_B_exists (k : ℝ) (x y : ℝ) :
  ¬ (∃ (B : ℝ × ℝ), B ≠ pointA ∧ ellipse 3 1 B.1 B.2 ∧ (x = k * B.1 + 1 ∧ y = (1 / k) * (x - pointA.1) + pointA.2))) :=
by sorry

end ellipse_equation_no_k_symmetric_B_exists_l21_21431


namespace incorrect_major_premise_l21_21527

theorem incorrect_major_premise (a : ℝ) (h : a = -2) :
  (∀ (n : ℕ), (n % 2 = 0) -> (√[n]a)^n = a) → false :=
by
-- The actual proof would be inserted here.
sorry

end incorrect_major_premise_l21_21527


namespace area_of_ABCD_l21_21122

theorem area_of_ABCD (A B C D E : Type) 
  (h1 : ∃ (triangle_AEB triangle_BCE triangle_CDE : Type), 
    (right_angle triangle_AEB ∧ 
     right_angle triangle_BCE ∧ 
     right_angle triangle_CDE)) 
  (h2 : ∠ A E B = 45 ∧ ∠ B E C = 45 ∧ ∠ C E D = 45)
  (h3 : AE = 20) : 
  area_ABCD = 175 := 
sorry

end area_of_ABCD_l21_21122


namespace sum_of_coefficients_l21_21805

theorem sum_of_coefficients (x y : ℝ) : 
  (2 * x - 3 * y) ^ 9 = -1 :=
by
  sorry

end sum_of_coefficients_l21_21805


namespace constant_ratio_lemma_l21_21050

noncomputable def C := λ P : ℝ × ℝ, (P.1 - 4)^2 + P.2^2 = 12

def A := (7, Real.sqrt 3) 
def B := (1, Real.sqrt 3)
def M := (-2, 0)
def N := (2, 0)

def tangent_line := λ P : ℝ × ℝ, Real.sqrt 3 * P.1 - P.2 = 0

theorem constant_ratio_lemma :
  (C A) ∧ (C B) ∧ (tangent_line B) →
  ∀ P : ℝ × ℝ, C P → Real.abs ((Real.sqrt ((P.1 + 2)^2 + P.2^2)) / (Real.sqrt ((P.1 - 2)^2 + P.2^2))) = Real.sqrt 3 :=
sorry

end constant_ratio_lemma_l21_21050


namespace sequence_solution_l21_21262

noncomputable def sequence (x : ℕ → ℕ) : Prop :=
  (x 1 = 2) ∧ (x 2 = 3) ∧
  (∀ m ≥ 1, x (2 * m + 1) = x (2 * m) + x (2 * m - 1)) ∧
  (∀ m ≥ 2, x (2 * m) = x (2 * m - 1) + 2 * x (2 * m - 2))

theorem sequence_solution (x : ℕ → ℕ) (h : sequence x) :
  ∀ n : ℕ,
    (∀ m ≥ 1, x (2 * m) = 4 * x (2 * m - 2) - 2 * x (2 * m - 4)) ∧
    (∀ m ≥ 1, x (2 * m + 1) = 4 * x (2 * m - 1) - 2 * x (2 * m - 3)) :=
by
  sorry

end sequence_solution_l21_21262


namespace james_old_hourly_wage_eq_16_l21_21158

-- Definitions for conditions
constant hourly_wage_new : ℕ := 20
constant weekly_hours_new : ℕ := 40
constant weeks_per_year : ℕ := 52
constant annual_difference : ℕ := 20800
constant weekly_hours_old : ℕ := 25

-- Theorem to prove Jame's hourly wage in his old job.
theorem james_old_hourly_wage_eq_16 
  (hw_new : hourly_wage_new = 20)
  (wh_new : weekly_hours_new = 40)
  (wpy : weeks_per_year = 52)
  (annual_dif : annual_difference = 20800)
  (wh_old : weekly_hours_old = 25) :
  let old_hourly_wage := 16 in
  let weekly_earnings_new := hourly_wage_new * weekly_hours_new in
  let annual_earnings_new := weekly_earnings_new * weeks_per_year in
  let annual_earnings_old := annual_earnings_new - annual_difference in
  let weekly_earnings_old := annual_earnings_old / weeks_per_year in
  old_hourly_wage = weekly_earnings_old / weekly_hours_old := 
sorry

end james_old_hourly_wage_eq_16_l21_21158


namespace largest_number_l21_21639

theorem largest_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 42) (h_dvd_a : 42 ∣ a) (h_dvd_b : 42 ∣ b)
  (a_eq : a = 42 * 11) (b_eq : b = 42 * 12) : max a b = 504 := by
  sorry

end largest_number_l21_21639


namespace monotonically_increasing_iff_m_geq_4_3_l21_21600

def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2 * x^2 + m * x + 1

-- Proposition: The function f is monotonically increasing on ℝ if and only if m ≥ 4/3
theorem monotonically_increasing_iff_m_geq_4_3 (m : ℝ) :
  (∀ x y : ℝ, x ≤ y → f(x, m) ≤ f(y, m)) ↔ m ≥ 4 / 3 := by
  sorry

end monotonically_increasing_iff_m_geq_4_3_l21_21600


namespace find_cot_difference_l21_21141

-- Define necessary elements for the problem
variable {A B C D : Type}
variable [EuclideanGeometry A]
variables (ABC : Triangle A B C)

-- Define the condition where median AD makes an angle of 60 degrees with BC
variable (ADmedian : median A B C D ∧ angle D A B = 60)

theorem find_cot_difference:
  |cot (angle B) - cot (angle C)| = 2 :=
sorry

end find_cot_difference_l21_21141


namespace exterior_angle_hexagon_l21_21865

theorem exterior_angle_hexagon (θ : ℝ) (hθ : θ = 60) (h_sum : θ * 6 = 360) : n = 6 :=
sorry

end exterior_angle_hexagon_l21_21865


namespace triangle_ABC_is_right_angled_l21_21891

-- Define the vertices of the triangle
def A := (2: ℝ, 2: ℝ, 0: ℝ)
def B := (0: ℝ, 2: ℝ, 0: ℝ)
def C := (0: ℝ, 1: ℝ, 4: ℝ)

-- Distance formula to compute the length of sides
def distance (p q : ℝ × ℝ × ℝ) : ℝ := 
(real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2))

-- Side lengths
def AB := distance A B
def BC := distance B C
def CA := distance C A

-- Triangle is right-angled
theorem triangle_ABC_is_right_angled : 
  (AB^2 + BC^2 = CA^2) ∨ (BC^2 + CA^2 = AB^2) ∨ (CA^2 + AB^2 = BC^2) :=
sorry

end triangle_ABC_is_right_angled_l21_21891


namespace path_X_is_faster_l21_21614

-- Definitions for distances and speeds
def distance_Path_X : ℝ := 8
def speed_Path_X : ℝ := 40

def distance_Y1 : ℝ := 5
def speed_Y1 : ℝ := 50
def distance_Y2 : ℝ := 1
def speed_Y2 : ℝ := 10
def distance_Y3 : ℝ := 1
def speed_Y3 : ℝ := 25

-- Calculations for times
def time_Path_X : ℝ := distance_Path_X / speed_Path_X * 60

def time_Y1 : ℝ := distance_Y1 / speed_Y1 * 60
def time_Y2 : ℝ := distance_Y2 / speed_Y2 * 60
def time_Y3 : ℝ := distance_Y3 / speed_Y3 * 60

def time_Path_Y : ℝ := time_Y1 + time_Y2 + time_Y3

-- Statement of the problem
theorem path_X_is_faster : time_Path_X = time_Path_Y - 2.4 :=
by
  sorry

end path_X_is_faster_l21_21614


namespace isosceles_triangle_perimeter_l21_21108

noncomputable theory

def is_isosceles_triangle (a b c : ℝ) : Prop := 
  a = b ∨ b = c ∨ a = c

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : is_isosceles_triangle a b 4) (h_valid : is_valid_triangle a b 4) :
  perimeter a b 4 = 10 :=
  sorry

end isosceles_triangle_perimeter_l21_21108


namespace max_distance_sum_l21_21170

open Real

-- Definitions extracted from conditions
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)
def L₁ (m : ℝ) (P : ℝ × ℝ) : Prop := P.1 + m * P.2 = 0
def L₂ (m : ℝ) (P : ℝ × ℝ) : Prop := m * P.1 - P.2 - m + 3 = 0
def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Statement to prove maximum value
theorem max_distance_sum (m : ℝ) (P : ℝ × ℝ) 
    (h1 : L₁ m P) (h2 : L₂ m P) : distance P A + distance P B ≤ 2 * sqrt 5 := 
sorry

end max_distance_sum_l21_21170


namespace logical_judgment_structure_l21_21126

-- Definitions of the structures
def sequential_structure : Type := unit
def conditional_structure : Type := unit
def loop_structure : Type := conditional_structure

-- Statement of the problem
theorem logical_judgment_structure :
  (∃ s : Type, s = conditional_structure ∨ s = loop_structure) :=
sorry

end logical_judgment_structure_l21_21126


namespace omega_range_l21_21253

theorem omega_range
  (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, x ∈ set.Icc 0 real.pi → f x = real.cos (ω * x + real.pi / 3))
  (range_f : set.range f = set.Icc (-1) (1 / 2)) :
  ∀ ω, (2 / 3 : ℝ) ≤ ω ∧ ω ≤ (4 / 3 : ℝ) := 
sorry

end omega_range_l21_21253


namespace books_count_is_8_l21_21926

theorem books_count_is_8
  (k a p_k p_a : ℕ)
  (h1 : k = a + 6)
  (h2 : k * p_k = 1056)
  (h3 : a * p_a = 56)
  (h4 : p_k > p_a + 100) :
  k = 8 := 
sorry

end books_count_is_8_l21_21926


namespace problem_statement_l21_21381

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem problem_statement : ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 := by
  sorry

end problem_statement_l21_21381


namespace maximum_area_triangle_ABC_l21_21876

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x - 2 * Real.sin x ^ 2 + 1

theorem maximum_area_triangle_ABC 
  (a b c A : ℝ)
  (h0 : a = Real.sqrt 3)
  (h1 : 0 < A ∧ A < Real.pi / 2)
  (h2 : f (A + Real.pi / 8) = Real.sqrt 2 / 3) :
  ∃ S, S = (3 * (Real.sqrt 3 + Real.sqrt 2)) / 4 :=
begin
  use (3 * (Real.sqrt 3 + Real.sqrt 2)) / 4,
  sorry
end

end maximum_area_triangle_ABC_l21_21876


namespace transformation_correct_l21_21274

noncomputable def g : ℝ → ℝ := λ x, Real.cos x
noncomputable def f : ℝ → ℝ := λ x, Real.sin (3 * x + Real.pi / 4)
noncomputable def transform (h : ℝ → ℝ) : ℝ → ℝ :=
  λ x, h ((3 * x) - (Real.pi / 4))

theorem transformation_correct :
  transform g = f := by
  sorry

end transformation_correct_l21_21274


namespace coeff_x2_in_binomial_expansion_l21_21821

theorem coeff_x2_in_binomial_expansion :
  ∃ c : ℤ, (∑ (r : ℤ) in finset.range 7, (binomial 6 r) * ((2 : ℝ)^6-r * (-1 : ℝ)^r * x^(3-r)) = (c * x^2)) ∧ c = -192 :=
by
  sorry

end coeff_x2_in_binomial_expansion_l21_21821


namespace min_own_all_luxuries_l21_21101

variables (Population Size : ℕ)
variables (Refrigerator_percentage : ℝ) (TV_percentage : ℝ) (Computer_percentage : ℝ) (AC_percentage : ℝ) (Washing_machine_percentage : ℝ) (Smartphone_percentage : ℝ)

def min_people_owning_luxuries (population : ℕ) (refrigerator : ℝ) (tv : ℝ) (computer : ℝ) (air_conditioner : ℝ) (washing_machine : ℝ) (smartphone : ℝ) : ℕ :=
  (population * smartphone).to_nat

theorem min_own_all_luxuries :
  min_people_owning_luxuries 10000 0.96 0.93 0.89 0.87 0.83 0.79 = 7900 := by
  sorry

end min_own_all_luxuries_l21_21101


namespace least_k_satisfying_l21_21625

noncomputable theory
open_locale classical

-- Define the sequence
def u : ℕ → ℝ
| 0       := 1 / 4
| (n + 1) := 2 * u n - 2 * (u n)^(2 : ℕ)

-- Define the limit L of the sequence
def L : ℝ := 1 / 2

-- Define the absolute difference
def abs_diff (k : ℕ) : ℝ := |u k - L|

-- Problem statement in Lean
theorem least_k_satisfying :
  ∃ k : ℕ, abs_diff k ≤ 1 / (2^1000) ∧ ∀ m < k, abs_diff m > 1 / (2^1000) :=
sorry

end least_k_satisfying_l21_21625


namespace perpendicular_planes_l21_21980

-- Definitions for lines and planes
variables (a b : Line) (α β : Plane)

-- Conditions
variables (h1 : a ≠ b) (h2 : α ≠ β) (h3 : a ⟂ b) (h4 : a ⟂ α) (h5 : b ⟂ β)

-- Prove that α ⟂ β
theorem perpendicular_planes (a b : Line) (α β : Plane)
  (h1 : a ≠ b) (h2 : α ≠ β) (h3 : a ⟂ b) (h4 : a ⟂ α) (h5 : b ⟂ β) : α ⟂ β :=
sorry

end perpendicular_planes_l21_21980


namespace correct_statements_l21_21831

def f (x : ℝ) : ℝ := sin (x + 3 * π / 2) * cos (π / 2 + x)

theorem correct_statements :
  let statements := [
    "The smallest positive period of the function f(x) is π",
    "If f(x₁) = -f(x₂), then x₁ = -x₂",
    "The graph of f(x) is symmetric about the line x = -π/4",
    "f(x) is a decreasing function on [π/4, 3π/4]"
  ]
  ∃ (result : ℕ), result = 3 :=
by
  sorry

end correct_statements_l21_21831


namespace cosine_identity_l21_21044

theorem cosine_identity (α : ℝ) (h : Real.sin (α - π / 4) = sqrt 5 / 5) : 
  Real.cos (α + π / 4) = - (sqrt 5 / 5) :=
by
  sorry

end cosine_identity_l21_21044


namespace calculation_correct_l21_21765

theorem calculation_correct :
  (Int.ceil ((15 : ℚ) / 8 * ((-35 : ℚ) / 4)) - 
  Int.floor (((15 : ℚ) / 8) * Int.floor ((-35 : ℚ) / 4 + (1 : ℚ) / 4))) = 1 := by
  sorry

end calculation_correct_l21_21765


namespace joe_new_average_score_after_dropping_lowest_l21_21161

theorem joe_new_average_score_after_dropping_lowest 
  (initial_average : ℕ)
  (lowest_score : ℕ)
  (num_tests : ℕ)
  (new_num_tests : ℕ)
  (total_points : ℕ)
  (new_total_points : ℕ)
  (new_average : ℕ) :
  initial_average = 70 →
  lowest_score = 55 →
  num_tests = 4 →
  new_num_tests = 3 →
  total_points = num_tests * initial_average →
  new_total_points = total_points - lowest_score →
  new_average = new_total_points / new_num_tests →
  new_average = 75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end joe_new_average_score_after_dropping_lowest_l21_21161


namespace return_path_length_l21_21696

noncomputable theory

def circumference_lower_base := 8
def circumference_upper_base := 6
def slope_angle := 60 -- degree

theorem return_path_length
  (clb : circumference_lower_base = 8)
  (cub : circumference_upper_base = 6)
  (sa : slope_angle = 60) :
  ∃ (length : ℝ), length = 4 * (real.sqrt 3) / real.pi :=
sorry

end return_path_length_l21_21696


namespace inequality_solution_l21_21675

theorem inequality_solution (x : ℝ) : |2 * x - 7| < 3 → 2 < x ∧ x < 5 :=
by
  sorry

end inequality_solution_l21_21675


namespace range_for_a_l21_21064

theorem range_for_a (a : ℝ) : (∃ x ∈ set.Icc 1 2, x^2 + 2*x - a ≥ 0) → a ≤ 8 :=
by
  sorry

end range_for_a_l21_21064


namespace least_four_digit_solution_l21_21825

theorem least_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
    5 * x ≡ 15 [MOD 20] ∧
    3 * x + 10 ≡ 19 [MOD 14] ∧
    -3 * x + 4 ≡ 2 * x [MOD 35] ∧
    x + 1 ≡ 0 [MOD 11] ∧
    x = 1163 :=
by {
  use 1163,
  split,
  {
    exact dec_trivial, -- 1000 ≤ 1163 < 10000
  },
  split,
  {
    norm_num, -- Verifying 1000 ≤ 1163
  },
  split,
  {
    norm_num, -- Verifying 1163 < 10000
  },
  split,
  {
    norm_num, -- Verifying 5 * 1163 ≡ 15 [MOD 20]
  },
  split,
  {
    norm_num, -- Verifying 3 * 1163 + 10 ≡ 19 [MOD 14]
  },
  split,
  {
    norm_num, -- Verifying -3 * 1163 + 4 ≡ 2 * 1163 [MOD 35]
  },
  split,
  {
    norm_num, -- Verifying 1163 + 1 ≡ 0 [MOD 11]
  },
  norm_num -- Verifying x = 1163
}

end least_four_digit_solution_l21_21825


namespace angle_between_cone_slant_heights_l21_21690

theorem angle_between_cone_slant_heights:
  ∃ (A O : Point) (AM AO : ℝ),
    AM = AO / (cos (acos (1/3))) ∧
    AO = 1 ∧
    (angle (Line.mk A O) (Line.mk A (AM * cos ((1 / 2) * (acos (1/3))))) = 60) :=
by
  sorry

end angle_between_cone_slant_heights_l21_21690


namespace problem1_proof_problem2_proof_l21_21864

noncomputable def problem1 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a| + |b| ≤ Real.sqrt 2

noncomputable def problem2 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a^3 / b| + |b^3 / a| ≥ 1

theorem problem1_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem1 a b h₁ h₂ h₃ :=
  sorry

theorem problem2_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem2 a b h₁ h₂ h₃ :=
  sorry

end problem1_proof_problem2_proof_l21_21864


namespace cot_difference_triangle_l21_21145

theorem cot_difference_triangle (ABC : Triangle)
  (angle_condition : ∠AD BC = 60) :
  |cot ∠B - cot ∠C| = 5 / 2 :=
sorry

end cot_difference_triangle_l21_21145


namespace limit_recurrence_l21_21587

noncomputable def recurrence_x (x y : ℝ) : ℝ := x * Real.cos y - y * Real.sin y
noncomputable def recurrence_y (x y : ℝ) : ℝ := x * Real.sin y + y * Real.cos y

theorem limit_recurrence :
  ∃ (x_lim y_lim : ℝ), 
    x_lim = -1 ∧ y_lim = 0 ∧ 
      ∀ (x_n y_n : ℕ → ℝ),
        (x_n 1 = 0.8 ∧ y_n 1 = 0.6) ∧ 
        (∀ n, x_n (n+1) = recurrence_x (x_n n) (y_n n)) ∧ 
        (∀ n, y_n (n+1) = recurrence_y (x_n n) (y_n n)) →
        x_lim = Real.lim x_n ∧ y_lim = Real.lim y_n :=
begin
  sorry
end

end limit_recurrence_l21_21587


namespace count_correct_statements_l21_21847

-- Definitions for the planes and the line
variable (α β : Type) [Plane α] [Plane β]
variable (l : Type) [Line l]

-- Definitions for the relationships (perpendicular, parallel)
variable (Perp : Line → Plane → Prop)
variable (Par : Line → Plane → Prop)
variable (ParPlanes : Plane → Plane → Prop)

-- Statements rewritten
def statement1 : Prop := ∀ (l : Type) [Line l], Perp l α → Perp α β → Par l β
def statement2 : Prop := ∀ (l : Type) [Line l], Perp l α → ParPlanes α β → Par l β
def statement3 : Prop := ∀ (l : Type) [Line l], Perp l α → ParPlanes α β → Perp l β
def statement4 : Prop := ∀ (l : Type) [Line l], Par l α → Perp α β → Perp l β

-- Count number of true statements
def correct_statements_count : ℕ :=
  (if statement1 then 1 else 0) +
  (if statement2 then 1 else 0) +
  (if statement3 then 1 else 0) +
  (if statement4 then 1 else 0)

-- The main theorem asserting the count
theorem count_correct_statements : correct_statements_count = 1 :=
  sorry

end count_correct_statements_l21_21847


namespace smallest_k_l21_21886

theorem smallest_k (k : ℕ) :
  (∀ x : ℝ, x ∈ set.Icc (0 : ℝ) 1 → y = 10 * real.tan ((2 * k - 1) * x / 5)) →
  (∃ k : ℕ, ∀ x : ℝ, x ∈ set.Icc (x : ℝ) (x + 1) → y = 10 * real.tan ((2 * k - 1) * x / 5) → false) →
  k = 13 :=
by sorry

end smallest_k_l21_21886


namespace general_term_sequence_l21_21670

theorem general_term_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : ∀ (m : ℕ), m ≥ 2 → a m - a (m - 1) + 1 = 0) : 
  a n = 3 - n :=
sorry

end general_term_sequence_l21_21670


namespace system_solution_l21_21237

theorem system_solution (x y z : ℝ) :
    x + y + z = 2 ∧ 
    x^2 + y^2 + z^2 = 26 ∧
    x^3 + y^3 + z^3 = 38 →
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    (x = 1 ∧ y = -3 ∧ z = 4) ∨
    (x = 4 ∧ y = 1 ∧ z = -3) ∨
    (x = 4 ∧ y = -3 ∧ z = 1) ∨
    (x = -3 ∧ y = 1 ∧ z = 4) ∨
    (x = -3 ∧ y = 4 ∧ z = 1) := by
  sorry

end system_solution_l21_21237


namespace socks_distribution_l21_21296

theorem socks_distribution (n : ℕ) (socks : ℕ) (children : ℕ → ℕ) 
  (h_total : socks = 9) 
  (h_four : ∀slist, list.length slist = 4 → ∃ i j, (i ≠ j) ∧ children i ≥ 2 ∧ children j ≥ 2)
  (h_five : ∀slist, list.length slist = 5 → ∀ i, children i ≤ 3) :
  ∃ c, c = 3 ∧ (∀ i, i < c → children i = 3 ∧ socks = c * 3) :=
by
  sorry

end socks_distribution_l21_21296


namespace perfect_squares_between_100_and_500_l21_21504

theorem perfect_squares_between_100_and_500 : 
  ∃ (count : ℕ), count = 12 ∧ 
    ∀ n : ℕ, (100 ≤ n^2 ∧ n^2 ≤ 500) ↔ (11 ≤ n ∧ n ≤ 22) :=
begin
  -- Proof goes here
  sorry
end

end perfect_squares_between_100_and_500_l21_21504


namespace calculation_proof_l21_21366

-- Definitions as conditions
def twenty_seven : ℕ := 3 ^ 3
def nine : ℕ := 3 ^ 2

-- The actual statement to prove
theorem calculation_proof : (twenty_seven^2 * nine^2) / 3^20 = 3^(-10) :=
by
  sorry

end calculation_proof_l21_21366


namespace isosceles_triangle_perimeter_l21_21113

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 4) (isosceles : (a = b) ∨ (a = 2) ∨ (b = 2)) :
  (a = 2 ∧ b = 4 → 10) :=
begin
  -- assuming isosceles triangle means either two sides are equal or a = 2 or b = 2 which fits the isosceles definition in the context of provided lengths.
  sorry
end

end isosceles_triangle_perimeter_l21_21113


namespace steven_peaches_l21_21575

theorem steven_peaches (jake_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_peaches = 3) (h2 : jake_peaches + 10 = steven_peaches) : steven_peaches = 13 :=
by
  sorry

end steven_peaches_l21_21575


namespace tax_collected_from_village_l21_21014

theorem tax_collected_from_village
  (T : ℝ) -- The total amount collected by the tax department from the village
  (william_paid : ℝ) -- The amount Mr. William paid as farm tax
  (william_percentage : ℝ) -- The percentage of Mr. William's land over the total taxable land
  (H : william_percentage * T = william_paid) : 
  T = 3456 :=
by 
  have hp : william_percentage = 0.1388888888888889 := sorry
  have hp' : william_paid = 480 := sorry
  simp [hp, hp'] at H
  exact H sorry

end tax_collected_from_village_l21_21014


namespace num_squares_in_6_by_6_grid_l21_21732

def squares_in_grid (m n : ℕ) : ℕ :=
  (m - 1) * (m - 1) + (m - 2) * (m - 2) + 
  (m - 3) * (m - 3) + (m - 4) * (m - 4) + 
  (m - 5) * (m - 5)

theorem num_squares_in_6_by_6_grid : squares_in_grid 6 6 = 55 := 
by 
  sorry

end num_squares_in_6_by_6_grid_l21_21732


namespace solve_for_x_l21_21809

-- Definitions based on conditions:
-- Let x be the number of meat pies baked in a day.
variable (x : ℕ)

-- Each meat pie is sold for $20.
def total_sales : ℕ := 20 * x

-- Du Chin uses 3/5 of the sales for ingredients.
def cost_of_ingredients : ℕ := (3 / 5 : ℚ) * total_sales x

-- Du Chin remains with $1600 after setting aside money for ingredients, thus (2/5) of total sales = 1600.
def remaining_money : ℕ := 1600

-- Prove that x = 200 given the conditions above.
theorem solve_for_x : x := by
  have eqn : (2 / 5 : ℚ) * total_sales x = remaining_money
  -- rewrite total_sales x as 20 * x
  rw [total_sales]
  -- simplified equation is (2 / 5) * (20 * x) = 1600
  -- multiply both sides by 5 to get rid of the fraction
  have : 2 * 20 * x = 1600 * 5 := by
    exact (calc
      (2 / 5 : ℚ) * (20 * x) = remaining_money : eqn
      ... ⇔ 2 * 20 * x / 5 = 1600 : by sorry -- skipping simplification
      ... ⇔ 2 * 20 * x = 1600 * 5 : by sorry -- cross-multiplying
    )
  -- solve for x
  exact (1600 * 5 / (2 * 20)) sorry

-- Lean infers x = 200 based on the equation above.
example : x = 200 := by
  rw [solve_for_x]

#reduce x  -- Reduction should show x = 200

end solve_for_x_l21_21809


namespace sum_derivative_equals_2024_l21_21053

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x - 1) = -f(-x - 1)

theorem sum_derivative_equals_2024
  (domain_f : ∀ x : ℝ, x ∈ ℝ)
  (domain_f' : ∀ x : ℝ, x ∈ ℝ)
  (odd_f_x_minus_1 : is_odd_function f)
  (derivative_relation : ∀ x : ℝ, f'(2 - x) + f'(x) = 2)
  (derivative_value : f'(-1) = 2) :
  ∑ i in Finset.range 2024, f'(2 * (i + 1) - 1) = 2024 :=
sorry

end sum_derivative_equals_2024_l21_21053


namespace sequence_general_formula_l21_21991

noncomputable def sequence (n : ℕ) : ℕ := sorry

theorem sequence_general_formula (S : ℕ → ℕ) :
  (∀ n, S n = ∑ i in Finset.range (n+1), sequence i) → 
  sequence 1 = 1 →
  (∀ n, S n = (n + 2) / 3 * sequence n) → 
  ∀ n, sequence n = (n * (n + 1)) / 2 :=
by
  intros h_sum h_base h_S
  sorry

end sequence_general_formula_l21_21991


namespace units_digit_base_9_l21_21829

theorem units_digit_base_9 (a b : ℕ) (h1 : a = 3 * 9 + 5) (h2 : b = 4 * 9 + 7) : 
  ((a + b) % 9) = 3 := by
  sorry

end units_digit_base_9_l21_21829


namespace train_speed_l21_21349

/--
Suppose a train 110 meters long takes 29.997600191984642 seconds to cross a bridge 390 meters in length.
Prove that the speed of the train is approximately 60 kilometers per hour.
-/
theorem train_speed :
  let train_length := 110
  let bridge_length := 390
  let total_distance := train_length + bridge_length
  let time := 29.997600191984642
  let speed_m_s := total_distance / time
  let speed_kmh := speed_m_s * 3.6
  speed_kmh ≈ 60 :=
by
  sorry -- Proof steps skipped

end train_speed_l21_21349


namespace lg2_eq_in_terms_of_a_b_l21_21840

-- Define the conditions: log base 2 of 9 equals a, log base 3 of 5 equals b
axiom log2_eq_a (a : ℝ) : log 2 9 = a
axiom log3_eq_b (b : ℝ) : log 3 5 = b

-- The theorem states that under these conditions, lg 2 equals 2 / (2 + ab)
theorem lg2_eq_in_terms_of_a_b (a b : ℝ) (h1 : log 2 9 = a) (h2 : log 3 5 = b) :
  log 10 2 = 2 / (2 + a * b) :=
sorry

end lg2_eq_in_terms_of_a_b_l21_21840


namespace freshmen_sophomores_percentage_l21_21267

theorem freshmen_sophomores_percentage
  (total_students : ℕ)
  (F : ℕ)
  (students_with_no_pet : ℕ)
  (total_students_eq : total_students = 400)
  (students_with_no_pet_eq : students_with_no_pet = 160)
  (own_pet_fraction : ℚ)
  (own_pet_fraction_eq : own_pet_fraction = 1/5)
  (no_pet_eq : F - fraction_to_nat own_pet_fraction * F = students_with_no_pet)
  : (F * 100 / total_students) = 50 := 
by 
  sorry

noncomputable def fraction_to_nat (fraction : ℚ) : ℕ :=
  by sorry  -- Assume a function to convert fraction to an integer


end freshmen_sophomores_percentage_l21_21267


namespace correct_operation_l21_21292

variable (a b : ℝ)

theorem correct_operation : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := 
sorry

end correct_operation_l21_21292


namespace minimum_x_y_l21_21440

theorem minimum_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
by sorry

end minimum_x_y_l21_21440


namespace marching_band_formations_l21_21713

theorem marching_band_formations :
  (∃ (s t : ℕ), s * t = 240 ∧ 8 ≤ t ∧ t ≤ 30) →
  ∃ (z : ℕ), z = 4 := sorry

end marching_band_formations_l21_21713


namespace area_of_ellipse_l21_21766

theorem area_of_ellipse (a b : ℝ) (h₀: a > 0) (h₁: b > 0) :
  (∫ x in -a..a, (b * sqrt (1 - (x ^ 2) / (a ^ 2))) * 2) = π * a * b :=
sorry

end area_of_ellipse_l21_21766


namespace percentage_more_than_cost_price_l21_21261

noncomputable def SP : ℝ := 7350
noncomputable def CP : ℝ := 6681.818181818181

theorem percentage_more_than_cost_price : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end percentage_more_than_cost_price_l21_21261


namespace count_integers_between_cubes_l21_21478

theorem count_integers_between_cubes : 
  let a := 10
  let b1 := 0.4
  let b2 := 0.5
  let cube1 := a^3 + 3 * a^2 * b1 + 3 * a * (b1^2) + b1^3
  let cube2 := a^3 + 3 * a^2 * b2 + 3 * a * (b2^2) + b2^3
  ∃ n : ℤ, n = 33 ∧ ((⌈cube1⌉.val ≤ n.val) ∧ (n.val ≤ ⌊cube2⌋.val)) :=
sorry

end count_integers_between_cubes_l21_21478


namespace aluminum_atomic_weight_l21_21404

theorem aluminum_atomic_weight (Al_w : ℤ) 
  (compound_molecular_weight : ℤ) 
  (num_fluorine_atoms : ℕ) 
  (fluorine_atomic_weight : ℤ) 
  (h1 : compound_molecular_weight = 84) 
  (h2 : num_fluorine_atoms = 3) 
  (h3 : fluorine_atomic_weight = 19) :
  Al_w = 27 := 
by
  -- Proof goes here, but it is skipped.
  sorry

end aluminum_atomic_weight_l21_21404


namespace evaluate_expression_l21_21811

theorem evaluate_expression :
  1002^3 - 1001 * 1002^2 - 1001^2 * 1002 + 1001^3 - 1000^3 = 2009007 :=
by
  sorry

end evaluate_expression_l21_21811


namespace sum_of_non_common_roots_zero_l21_21873

theorem sum_of_non_common_roots_zero (m α β γ : ℝ) 
  (h1 : α + β = -(m + 1))
  (h2 : α * β = -3)
  (h3 : α + γ = 4)
  (h4 : α * γ = -m)
  (h_common : α^2 + (m + 1)*α - 3 = 0)
  (h_common2 : α^2 - 4*α - m = 0)
  : β + γ = 0 := sorry

end sum_of_non_common_roots_zero_l21_21873


namespace students_per_group_l21_21318

-- Defining the conditions
def total_students : ℕ := 256
def number_of_teachers : ℕ := 8

-- The statement to prove
theorem students_per_group :
  total_students / number_of_teachers = 32 :=
by
  sorry

end students_per_group_l21_21318


namespace stretches_per_meter_l21_21097

variables (p q r s t u : ℕ)

theorem stretches_per_meter (h1 : p * q = r * s) (h2 : t * r = u * s) :
  1 meter = ts / ur := by
  sorry

end stretches_per_meter_l21_21097


namespace arthur_heads_probability_l21_21362

theorem arthur_heads_probability :
  let a := 231 in
  let b := 1024 in
  ∃ a b : ℕ, a.gcd b = 1 ∧ a + b = 1255 :=
begin
  sorry
end

end arthur_heads_probability_l21_21362


namespace solution_set_inequality_l21_21877

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.exp x else -x / Real.exp x

theorem solution_set_inequality : {x : ℝ | f (x - 2) < Real.exp 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end solution_set_inequality_l21_21877


namespace cot_diff_equal_l21_21150

variable (A B C D : Type)

-- Define the triangle and median.
variable [triangle ABC : Type] (median : Type)

-- Define the angle condition.
def angle_condition (ABC : triangle) (AD : median) : Prop :=
  ∠(AD, BC) = 60

-- Prove the cotangent difference
theorem cot_diff_equal
  (ABC : triangle)
  (AD : median)
  (h : angle_condition ABC AD) :
  abs (cot B - cot C) = (9 - 3 * sqrt 3) / 2 :=
by
  sorry -- Proof to be constructed

end cot_diff_equal_l21_21150


namespace count_multiples_of_12_l21_21483

theorem count_multiples_of_12 (a b : ℕ) (h₁ : 15 < a) (h₂ : a < 250) 
                               (h₃ : 15 < b) (h₄ : b < 250) :
  count (fun x ↦ x % 12 = 0) (finset.range (b.succ) \ finset.range a) = 19 :=
by
  sorry

end count_multiples_of_12_l21_21483


namespace sampling_method_correct_l21_21946

theorem sampling_method_correct:
  ∀ (classes students_per_class student_for_exchange: ℕ),
    classes = 14 →
    students_per_class = 50 →
    student_for_exchange = 14 →
    (sample_method classes students_per_class student_for_exchange) = "Systematic Sampling" :=
by
  intro classes students_per_class student_for_exchange 
  intro h1 h2 h3
  -- Placeholder for the actual proof
  sorry

-- Define the sample_method function
def sample_method (classes students_per_class student_for_exchange: ℕ) : String :=
  if classes = 14 ∧ students_per_class = 50 ∧ student_for_exchange = 14 then "Systematic Sampling"
  else "Unknown"


end sampling_method_correct_l21_21946


namespace point_on_angle_bisector_pq_parallel_to_x_axis_l21_21854

/-- If point P(a+1, 2a-3) lies on the angle bisector in the first and third quadrants and point Q(2, 3), then coordinates of P are (5, 5). --/
theorem point_on_angle_bisector (a : ℝ) (P Q : ℝ × ℝ) (hP : P = (a + 1, 2 * a - 3)) (hQ : Q = (2, 3)) :
  (a + 1 = 2 * a - 3) → P = (5, 5) :=
by
  sorry

/-- If the line segment PQ is parallel to the x-axis, then PQ = 2. --/
theorem pq_parallel_to_x_axis (a : ℝ) (P Q : ℝ × ℝ) (hP : P = (a + 1, 2 * a - 3)) (hQ : Q = (2, 3)) :
  (2 * a - 3 = 3) → (P.fst - Q.fst).abs = 2 :=
by
  sorry

end point_on_angle_bisector_pq_parallel_to_x_axis_l21_21854


namespace card_probability_sequence_l21_21415

/-- 
Four cards are dealt at random from a standard deck of 52 cards without replacement.
The probability that the first card is a Jack, the second card is a Queen, the third card is a King, and the fourth card is an Ace is given by:
-/
theorem card_probability_sequence :
  let p1 := 4 / 52,
      p2 := 4 / 51,
      p3 := 4 / 50,
      p4 := 4 / 49
  in p1 * p2 * p3 * p4 = 64 / 1624350 :=
by
  let p1 := 4 / 52
  let p2 := 4 / 51
  let p3 := 4 / 50
  let p4 := 4 / 49
  show p1 * p2 * p3 * p4 = 64 / 1624350
  sorry

end card_probability_sequence_l21_21415


namespace count_multiples_of_12_between_15_and_250_l21_21484

theorem count_multiples_of_12_between_15_and_250 : 
  (∃ n : ℕ, ∃ m : ℕ, (2 ≤ n ∧ n ≤ 20) ∧ (m = 12 * n) ∧ (15 < m ∧ m < 250)) → 
  Nat.card { x ∈ Nat | ∃ k : ℕ, x = 12 * k ∧ 15 < x ∧ x < 250 } = 19 := 
by
  sorry

end count_multiples_of_12_between_15_and_250_l21_21484


namespace solve_quadratics_l21_21773

theorem solve_quadratics :
  ∃ x y : ℝ, (9 * x^2 - 36 * x - 81 = 0) ∧ (y^2 + 6 * y + 9 = 0) ∧ (x + y = -1 + Real.sqrt 13 ∨ x + y = -1 - Real.sqrt 13) := 
by 
  sorry

end solve_quadratics_l21_21773


namespace Pythagorean_triple_l21_21774

theorem Pythagorean_triple : ∃ (a b c : ℕ), (a = 6) ∧ (b = 8) ∧ (c = 10) ∧ (a^2 + b^2 = c^2) :=
by
  use 6, 8, 10
  split; try {refl}
  calc
    6^2 + 8^2 = 36 + 64 := by rw [pow_two, pow_two]
            ... = 100 := by norm_num
            ... = 10^2 := by norm_num
  done

end Pythagorean_triple_l21_21774


namespace eight_div_repeating_eight_l21_21286

theorem eight_div_repeating_eight : 8 / (0.8 + 0.08 + 0.008 + ...) = 9 := by
  have q : ℚ := 8 / 9
  sorry

end eight_div_repeating_eight_l21_21286


namespace min_value_expr_l21_21988

theorem min_value_expr (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_pos_z : 0 < z) (h_xyz : x * y * z = 1) :
  (x + 3*y) * (y + 3*z) * (x + 3*z + 1) ≥ 24*sqrt(3) :=
sorry

end min_value_expr_l21_21988


namespace s1_eq_third_s2_l21_21974

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (GA GB GC AB BC CA : ℝ) (G : A)

-- Definitions based on the problem conditions
def is_centroid (G : A) (A B C : A) : Prop :=
  ∃ (GA GB GC : ℝ), GA + GB + GC = GA + GB + GC

def s1 (GA GB GC : ℝ) : ℝ := GA + GB + GC

def s2 (AB BC CA : ℝ) : ℝ := AB + BC + CA

-- The theorem to be proved
theorem s1_eq_third_s2 (G : A) (GA GB GC AB BC CA : ℝ) (h1 : is_centroid G A B C)
  (h2 : s1 GA GB GC = GA + GB + GC) (h3 : s2 AB BC CA = AB + BC + CA) :
  s1 GA GB GC = (1/3) * s2 AB BC CA :=
sorry

end s1_eq_third_s2_l21_21974


namespace movie_watching_l21_21382

theorem movie_watching :
  let total_duration := 120 
  let watched1 := 35
  let watched2 := 20
  let watched3 := 15
  let total_watched := watched1 + watched2 + watched3
  total_duration - total_watched = 50 :=
by
  sorry

end movie_watching_l21_21382


namespace lcm_25_35_50_l21_21020

theorem lcm_25_35_50 : Nat.lcm (Nat.lcm 25 35) 50 = 350 := by
  sorry

end lcm_25_35_50_l21_21020


namespace trig_identity_example_l21_21770

theorem trig_identity_example :
  sin (Real.pi * 21 / 180) * cos (Real.pi * 39 / 180) + cos (Real.pi * 21 / 180) * sin (Real.pi * 39 / 180) = sqrt 3 / 2 :=
by
  sorry

end trig_identity_example_l21_21770


namespace problem_statement_l21_21976

noncomputable def geom_seq_common_ratio (q : ℝ) (a1 a3 a4 : ℝ) : Prop :=
S3 + S4 = 2 * S2

noncomputable def initial_condition (a1 a3 a4 : ℝ) : Prop :=
a1 + 2 * a3 + a4 = 4

noncomputable def general_term (a_n : ℕ → ℝ) (n : ℕ) :=
a_n n = (-2)^(n + 1)

noncomputable def seq_b (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
∀ n, b_n n = 1 / (n + 1)

noncomputable def seq_T (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) : Prop :=
∀ n, T_n n = Σ i in (0..n-1), b_n i * b_n (i + 1)

noncomputable def inequality_condition (T_n : ℕ → ℝ) (a : ℝ) : Prop :=
∀ n, a * T_n n < n + 4

theorem problem_statement (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) (a q a1 a3 a4 : ℝ) :
  geom_seq_common_ratio q a1 a3 a4 →
  initial_condition a1 a3 a4 →
  (∀ n, general_term a_n n) →
  seq_b a_n b_n →
  seq_T b_n T_n →
  inequality_condition T_n a →
  a < 70 / 3 := 
sorry

end problem_statement_l21_21976


namespace minimum_crooks_l21_21546

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l21_21546


namespace angle_ABC_110_degrees_l21_21568

noncomputable def triangle (A B C: Type) :=
  -- Defining points and line segments
  ∃ (A B C M D: Type) 
    (BM AB BD : ℝ)
    (α : ℝ), 
  BM = 2 * AB ∧
  AB = BD ∧
  -- Given angles
  ∠(BM, AB) = 40 ∧
  -- The requirement is to prove
  ∠ABC = 110

theorem angle_ABC_110_degrees (A B C M D : Type) 
  (BM AB BD : ℝ) (α : ℝ):
  BM = 2 * AB ∧ BD = AB → 
  ∠(BM, AB) = 40 →
  ∠ABC = 110 :=
  by
    -- Proof construction placeholder
    sorry

end angle_ABC_110_degrees_l21_21568


namespace jim_loan_inequality_l21_21160

noncomputable def A (t : ℕ) : ℝ := 1500 * (1.06 ^ t)

theorem jim_loan_inequality : ∃ t : ℕ, A t > 3000 ∧ ∀ t' : ℕ, t' < t → A t' ≤ 3000 :=
by
  sorry

end jim_loan_inequality_l21_21160


namespace segment_P1P2_len_l21_21005

noncomputable def length_P1P2 : ℝ :=
  let x := by 
    have h1 : 6 * cos x = 9 * tan x := sorry
    have h2 : 2 * cos x * cos x = 3 * sin x := sorry
    have h3 : sin x = 1 / 2 := sorry
    exact x in
  sin x

theorem segment_P1P2_len : length_P1P2 = 1 / 2 :=
by
  sorry

end segment_P1P2_len_l21_21005


namespace diamond_equation_l21_21409

noncomputable def diamond (a b : ℝ) (h : a ≠ b) : ℝ := (a^2 + b^2) / (a - b)

theorem diamond_equation : 
  ((diamond 2 3 (by norm_num)) \diamond 4 (by norm_num)) = -185 / 17 :=
sorry

end diamond_equation_l21_21409


namespace repeated_application_l21_21090

def piDigits : ℕ → ℕ :=
  λ n, match n with
       | 1 := 1
       | 2 := 4
       | 3 := 1
       | 4 := 5
       | 5 := 9
       | 6 := 2
       | 7 := 6
       | 8 := 5
       | 9 := 3
       | _ := 0 -- handle other cases; this is oversimplified
       end

def f : ℕ → ℕ := piDigits

theorem repeated_application :
  (Nat.iterate f 2007 (f 7)) = 1 :=
by
  sorry

end repeated_application_l21_21090


namespace at_least_332_visiting_the_same_place_l21_21345

theorem at_least_332_visiting_the_same_place(
  campers : ℕ,
  places : ℕ,
  categories : ℕ
) : campers = 1987 ∧ places = 3 ∧ categories = 6 → 
∃ p, p ≥ 332 ∧ p = ⌈(1987 : ℚ) / 6⌉ + 1 :=
by
  sorry

end at_least_332_visiting_the_same_place_l21_21345


namespace find_y_coordinate_l21_21856

theorem find_y_coordinate (y : ℝ) : y = 12 → 
  let A := (-3 : ℝ, 9 : ℝ)
      B := (6 : ℝ, y)
      slope := 1/3 : ℝ in 
  slope = (B.2 - A.2) / (B.1 - A.1) := 
by
  sorry

end find_y_coordinate_l21_21856


namespace shortest_distance_point_parabola_l21_21408

theorem shortest_distance_point_parabola :
  let P : ℝ × ℝ := (4, 10)
  let parabola := λ y : ℝ, y^2 / 4 
  shortest_distance P parabola = 6 :=
by
  sorry

end shortest_distance_point_parabola_l21_21408


namespace coefficient_of_x6_in_expansion_l21_21643

theorem coefficient_of_x6_in_expansion :
  let n := 8
  let p := 2
  (binom n p * (-2)^p : ℤ) = 112 :=
by
  sorry

end coefficient_of_x6_in_expansion_l21_21643


namespace number_of_sets_A_l21_21260

def num_sets_subset_condition : ℕ :=
  let A := { {1}, {1, 2}, {1, 3}, {1, 2, 3} }
  A.card

theorem number_of_sets_A : num_sets_subset_condition = 4 :=
by
  -- placeholder for proof
  sorry

end number_of_sets_A_l21_21260


namespace find_least_n_l21_21604

noncomputable def v : ℕ → ℝ
| 0       := -3 / 8
| (n + 1) := - (v n) ^ 2

def M : ℝ := 0

theorem find_least_n 
  (h : ∀ n, v n = (λ n, if n = 0 then -3 / 8 else - (v (n - 1)) ^ 2) n)
  : ∃ n : ℕ, abs (v n - M) ≤ 1 / 2^500 → n = 8 :=
sorry

end find_least_n_l21_21604


namespace minimum_time_to_assess_25_students_l21_21279

theorem minimum_time_to_assess_25_students :
  ∃ T : ℝ, T >= 110 ∧ 
    (∀ X Y : ℝ, 0 ≤ X ∧ 0 ≤ Y ∧ X + Y ≤ 25 ∧ 
     5 * X + 7 * Y ≤ T ∧ 
     3 * (25 - X) + 4 * (25 - Y) ≤ T) :=
begin
  sorry
end

end minimum_time_to_assess_25_students_l21_21279


namespace product_complex_l21_21790

noncomputable def P (x : ℂ) : ℂ := ∏ k in finset.range 15, (x - complex.exp (2 * real.pi * complex.I * k / 17))
noncomputable def Q (x : ℂ) : ℂ := ∏ j in finset.range 12, (x - complex.exp (2 * real.pi * complex.I * j / 13))

theorem product_complex : 
  ∏ k in finset.range 15, (∏ j in finset.range 12, (complex.exp (2 * real.pi * complex.I * j / 13) - complex.exp (2 * real.pi * complex.I * k / 17))) = 1 := 
sorry

end product_complex_l21_21790


namespace malcolm_green_lights_l21_21999

def colored_lights (red_lights blue_lights green_lights total_colored_lights remaining_colored_lights total_lights_bought : ℕ) :=
red_lights = 12 ∧
blue_lights = 3 * red_lights ∧
remaining_colored_lights = 5 ∧
total_colored_lights = 59 - remaining_colored_lights ∧
total_lights_bought = red_lights + blue_lights ∧
total_colored_lights - total_lights_bought = green_lights

theorem malcolm_green_lights :
  ∃ green_lights : ℕ,
    colored_lights 12 (3 * 12) green_lights 54 5 (12 + 3 * 12) ∧ green_lights = 6 :=
begin
  sorry
end

end malcolm_green_lights_l21_21999


namespace perfect_squares_between_100_and_500_l21_21502

theorem perfect_squares_between_100_and_500 : 
  ∃ (count : ℕ), count = 12 ∧ 
    ∀ n : ℕ, (100 ≤ n^2 ∧ n^2 ≤ 500) ↔ (11 ≤ n ∧ n ≤ 22) :=
begin
  -- Proof goes here
  sorry
end

end perfect_squares_between_100_and_500_l21_21502


namespace JerkTunaFishCount_l21_21243

def JerkTunaHasJFish (J : ℕ) : Prop :=
  ∃ T : ℕ, T = 2 * J ∧ J + T = 432

theorem JerkTunaFishCount : ∃ J : ℕ, JerkTunaHasJFish J ∧ J = 144 :=
by
  use 144
  split
  { unfold JerkTunaHasJFish
    use 288
    split
    { norm_num }
    { norm_num }},
  { norm_num }

-- sorry included to indicate the proof is not required

end JerkTunaFishCount_l21_21243


namespace age_difference_between_two_children_l21_21314

theorem age_difference_between_two_children 
  (avg_age_10_years_ago : ℕ)
  (present_avg_age : ℕ)
  (youngest_child_present_age : ℕ)
  (initial_family_members : ℕ)
  (current_family_members : ℕ)
  (H1 : avg_age_10_years_ago = 24)
  (H2 : present_avg_age = 24)
  (H3 : youngest_child_present_age = 3)
  (H4 : initial_family_members = 4)
  (H5 : current_family_members = 6) :
  ∃ (D: ℕ), D = 2 :=
by
  sorry

end age_difference_between_two_children_l21_21314


namespace greatest_odd_factors_le_150_l21_21197

theorem greatest_odd_factors_le_150 : 
  let n := 144 in
  n < 150 ∧ (∃ k, k * k = n) :=
by
  sorry

end greatest_odd_factors_le_150_l21_21197


namespace sum_slope_y_intercept_line_ZG_l21_21121

-- Define points X, Y, Z
def X := (0 : ℝ, 8 : ℝ)
def Y := (0 : ℝ, 0 : ℝ)
def Z := (10 : ℝ, 0 : ℝ)

-- Define the midpoint G of line segment XY
def G : ℝ × ℝ := (0, 4)

-- Define the conditions and question as a theorem
theorem sum_slope_y_intercept_line_ZG : 
  let slope := (G.2 - Z.2) / (G.1 - Z.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Proof will go here
  sorry

end sum_slope_y_intercept_line_ZG_l21_21121


namespace part_a_part_b_part_c_l21_21799

-- Step (a) definitions
def f (x : ℝ) : ℤ := if x - x.floor < 0.5 then x.floor else x.ceil

def R_n (n : ℤ) : Set (ℝ × ℝ) := {p | f (abs p.1) + f (abs p.2) < n}

-- Placeholder for S_{nO} definition based on perimeter definition.
def S_nO (n : ℤ) (O : (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Step (c) proofs
theorem part_a (n : ℤ) (hn : n > 0) : (perimeter (R_n n)) % 8 = 4 :=
  sorry

theorem part_b (n : ℤ) (hn : n > 0) (O1 O2 : (ℝ × ℝ)) : 
  area (S_nO n O1) = area (S_nO n O2) :=
  sorry

theorem part_c (n : ℤ) (hn : n > 0) (O : (ℝ × ℝ)) :
  area (R_n n) + area (S_nO n O) = (2 * n - 1) ^ 2 :=
  sorry

end part_a_part_b_part_c_l21_21799


namespace find_m_and_b_sum_l21_21653

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l21_21653


namespace equalize_milk_impossible_l21_21421

theorem equalize_milk_impossible (m : Fin 30 → ℕ) :
  let T := ∑ i, m i
  ∃ i, ∃ j, (m i ≠ m j) →
    let target := T / 30
    ¬ ∃ k, target * 2^k ∈ ℕ  :=
begin
  sorry,
end

end equalize_milk_impossible_l21_21421


namespace relationship_l21_21030

noncomputable def a : ℝ := (1/3)^(2/3)
noncomputable def b : ℝ := (1/4)^(1/3)
noncomputable def c : ℝ := Real.logBase 3 π

theorem relationship : a < b ∧ b < c :=
by
  sorry

end relationship_l21_21030


namespace rectangle_tiling_possible_l21_21171

-- Definitions and conditions
def is_even (m : ℤ) : Prop := ∃ k : ℤ, m = 2 * k
def can_tile (m : ℤ) : Prop := ∃ tiling : list (ℤ × ℤ), length tiling = 5 * m / 5 

-- Theorem statement
theorem rectangle_tiling_possible (m : ℤ) : can_tile m ↔ is_even m :=
sorry

end rectangle_tiling_possible_l21_21171


namespace no_prime_factors_of_form_6k_plus_5_l21_21221

theorem no_prime_factors_of_form_6k_plus_5 (n : ℕ) (hn : n > 0) : 
  ∀ p, prime p → p ∣ (n^2 - n + 1) → ∀ k, k > 0 → p ≠ 6 * k + 5 :=
by
  sorry

end no_prime_factors_of_form_6k_plus_5_l21_21221


namespace count_multiples_of_12_l21_21482

theorem count_multiples_of_12 (a b : ℕ) (h₁ : 15 < a) (h₂ : a < 250) 
                               (h₃ : 15 < b) (h₄ : b < 250) :
  count (fun x ↦ x % 12 = 0) (finset.range (b.succ) \ finset.range a) = 19 :=
by
  sorry

end count_multiples_of_12_l21_21482


namespace solve_system_l21_21818

theorem solve_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 35 ∧ x = -155 / 47 ∧ y = 205 / 47 :=
by {
  use [-155 / 47, 205 / 47],
  split,
  -- Verification of the first equation
  calc 7 * (-155 / 47)
      = -1085 / 47 : by ring
  ... = -10 - 3 * (205 / 47) : by ring,

  split,
  -- Verification of the second equation
  calc 4 * (-155 / 47)
       = -620 / 47 : by ring
  ... = 5 * (205 / 47) - 35 : by ring,

  -- Verification of the solutions
  refl,
  refl,
}

end solve_system_l21_21818


namespace A_is_equidistant_l21_21822

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def distance (P Q : Point3D) : ℝ := 
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2 + (Q.z - P.z)^2)

noncomputable def A := Point3D.mk 0 9 0
def B := Point3D.mk 0 (-4) 1
def C := Point3D.mk 1 (-3) 5

theorem A_is_equidistant (A B C : Point3D) (hA : A = ⟨0, 9, 0⟩) (hB : B = ⟨0, -4, 1⟩) (hC : C = ⟨1, -3, 5⟩) : 
  distance A B = distance A C :=
by
  rw [hA, hB, hC]
  sorry  -- Proof steps will go here

end A_is_equidistant_l21_21822


namespace proj_u_onto_v_l21_21804

-- Define vector u
def u : ℝ × ℝ := (1, 4)

-- Define vector v
def v : ℝ × ℝ := (1, 2)

-- Define the dot product of two vectors
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- Define the projection of u onto v
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let v_dot_v := dot_product v v
  let u_dot_v := dot_product u v
  (u_dot_v / v_dot_v * v.1, u_dot_v / v_dot_v * v.2)

-- Theorem statement: projection of u onto v is the expected vector
theorem proj_u_onto_v :
  proj u v = (9/5, 18/5) :=
by
  sorry

end proj_u_onto_v_l21_21804


namespace integral_f_eq_neg_one_third_l21_21913

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * ∫ t in 0..1, f t

theorem integral_f_eq_neg_one_third :
  (∫ x in 0..1, f x) = -1 / 3 :=
by
  have h : ∫ t in 0..1, f t = x
  sorry

end integral_f_eq_neg_one_third_l21_21913


namespace problem_statement_l21_21984

theorem problem_statement
  (a b c : ℝ)
  (h_ineq_iff : ∀ x, (x < -4 ∨ |x - 25| ≤ 1) ↔ ((x-a)*(x-b)/(x-c)) ≤ 0)
  (h_a_lt_b : a < b) :
  a + 2b + 3c = 64 := by
  sorry

end problem_statement_l21_21984


namespace simplify_and_evaluate_expr_l21_21230

/-- Simplification of the given expression and evaluation for x = 3. -/
theorem simplify_and_evaluate_expr :
  ∀ (x : ℝ), x ≠ 1 ∧ x ≠ 2 →
  (1 + 1 / (x - 2)) / (x - 1) / ((x^2 - 4*x + 4) = x - 2) →
  (x = 3 → (1 + 1 / (x - 2)) / (x - 1) / ((x^2 - 4*x + 4) = 1 :=
by
  intros x hx hx3
  sorry -- proof not required

end simplify_and_evaluate_expr_l21_21230


namespace line_equation_l21_21330

theorem line_equation (x y : ℝ) :
  (∃ x : ℝ, y : ℝ, (λ x y, 2 * (x - 4) - (y + 3)) x y = 0) → (y = 2 * x - 11) :=
by
  sorry

end line_equation_l21_21330


namespace find_x_l21_21516

-- Definitions of binomial coefficients as conditions
def binomial (n k : ℕ) : ℕ := n.choose k

-- The specific conditions given
def C65_eq_6 : Prop := binomial 6 5 = 6
def C64_eq_15 : Prop := binomial 6 4 = 15

-- The theorem we need to prove: ∃ x, binomial 7 x = 21
theorem find_x (h1 : C65_eq_6) (h2 : C64_eq_15) : ∃ x, binomial 7 x = 21 :=
by
  -- Proof will go here
  sorry

end find_x_l21_21516


namespace true_prices_for_pie_and_mead_l21_21940

-- Definitions for true prices
variable (k m : ℕ)

-- Definitions for conditions
def honest_pravdoslav (k m : ℕ) : Prop :=
  4*k = 3*(m + 2) ∧ 4*(m+2) = 3*k + 14

theorem true_prices_for_pie_and_mead (k m : ℕ) (h : honest_pravdoslav k m) : k = 6 ∧ m = 6 := sorry

end true_prices_for_pie_and_mead_l21_21940


namespace convex_quadrilaterals_lower_bound_l21_21846

open_locale classical

theorem convex_quadrilaterals_lower_bound
  (n : ℕ) (h_n_gt_4 : n > 4) (points : fin n → ℝ × ℝ)
  (h_no_collinear : ∀ ⦃p1 p2 p3 : ℝ × ℝ⦄,
    (∃ i1 i2 i3 : fin n, points i1 = p1 ∧ points i2 = p2 ∧ points i3 = p3) → 
    ¬ collinear p1 p2 p3):
  ∃ (quadrilaterals : finset (fin n) set), 
  quadrilaterals.card ≥ (n - 3).choose 2 ∧ 
  ∀ q ⦃i1 i2 i3 i4 : fin n⦄, 
  {i1, i2, i3, i4} ∈ quadrilaterals → 
  is_convex_quadrilateral (points i1) (points i2) (points i3) (points i4) := 
begin
  sorry,
end

end convex_quadrilaterals_lower_bound_l21_21846


namespace quadrilateral_area_l21_21205

-- Define the basic properties and conditions of the problem
variables {A B C D: Point} -- Points representing vertices of the quadrilateral
variable {r : Real} -- The radius of the circles
variable (O1 O2 : Point) -- Centers of the circles

-- Conditions
def circle1_tangent (circle1 : Circle) : Prop :=
  circle1.radius = r ∧ circle1.center = O1 ∧
  circle1.tangent_to_side A B ∧ circle1.tangent_to_side A D ∧ circle1.tangent_to_side B C

def circle2_tangent (circle2 : Circle) : Prop :=
  circle2.radius = r ∧ circle2.center = O2 ∧
  circle2.tangent_to_side B C ∧ circle2.tangent_to_side C D ∧ circle2.tangent_to_side A D

def circles_touch_externally (circle1 circle2 : Circle) : Prop :=
  circle1.radius = circle2.radius ∧
  (distance_between_centers circle1.center circle2.center = 2 * r)

-- Defining the quadrilateral and the problem
noncomputable def area_of_quadrilateral (A B C D : Point) (r : Real) : Real :=
  4 * (Real.sqrt 2 + 1) * r^2

-- The main theorem to prove
theorem quadrilateral_area:
  ∀ (ABCD : ConvexQuadrilateral) (circle1 circle2: Circle),
    circle1_tangent circle1 →
    circle2_tangent circle2 →
    circles_touch_externally circle1 circle2 →
    area_of_quadrilateral A B C D r = 4 * (Real.sqrt 2 + 1) * r^2 :=
  sorry

end quadrilateral_area_l21_21205


namespace general_formula_for_sequence_sum_of_new_sequence_l21_21849

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > 0

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 1 / 2 * (a n - 1) * (a n + 2)

theorem general_formula_for_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (h_seq : sequence a) (h_sum : sum_of_terms S a) :
  ∀ n : ℕ, n > 0 → a n = n + 1 :=
by
  sorry

def new_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (-1)^n * a n * a (n + 1)

def sum_of_first_2n_terms (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → T (2 * n) = 2 * n^2 + 4 * n

theorem sum_of_new_sequence
  (a : ℕ → ℝ) (T : ℕ → ℝ) (h_seq : sequence a) (h_sum : sum_of_terms S a)
  (h_general : ∀ n : ℕ, n > 0 → a n = n + 1) :
  sum_of_first_2n_terms T a :=
by
  sorry

end general_formula_for_sequence_sum_of_new_sequence_l21_21849


namespace rectangle_area_increase_l21_21092

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  let A := l * w
  let increase := A_new - A
  let percent_increase := (increase / A) * 100
  percent_increase = 56 := sorry

end rectangle_area_increase_l21_21092


namespace greatest_odd_factors_le_150_l21_21195

theorem greatest_odd_factors_le_150 : 
  let n := 144 in
  n < 150 ∧ (∃ k, k * k = n) :=
by
  sorry

end greatest_odd_factors_le_150_l21_21195


namespace count_multiples_of_3003_in_form_l21_21902

noncomputable def is_multiple_form (i j : ℕ) : Prop :=
  0 ≤ i ∧ i < j ∧ j ≤ 99 ∧ ∃ k : ℕ, 3003 * k = 10^j - 10^i

theorem count_multiples_of_3003_in_form : 
  (finset.filter
    (λ p : ℕ × ℕ, is_multiple_form p.1 p.2)
    ((finset.range 100).product (finset.range 100))).card = 784 :=
begin
  sorry
end

end count_multiples_of_3003_in_form_l21_21902


namespace midpoint_locus_is_circle_l21_21687

noncomputable def locus_of_midpoint {α : Type*} [normed_group α] [normed_space ℝ α]
  (O₁ O₂ : α) (r₁ r₂ : ℝ) (ω : ℝ) : set α :=
  let midpoint := (O₁ + O₂) / 2 in
  let radius := (r₁ + r₂) / 2 in
  {M | ∃ (t : ℝ), M = midpoint + radius • (real.cos (ω * t), real.sin (ω * t))}

theorem midpoint_locus_is_circle {α : Type*} [normed_group α] [normed_space ℝ α]
  (O₁ O₂ : α) (r₁ r₂ : ℝ) (ω : ℝ) :
  midpoint_locus O₁ O₂ r₁ r₂ ω = locus_of_midpoint O₁ O₂ r₁ r₂ ω :=
sorry

end midpoint_locus_is_circle_l21_21687


namespace problem_proof_l21_21969

noncomputable def exists_distinct_indices (n : ℕ) (t : ℝ) (a : fin (2 * n - 1) → ℝ) : Prop :=
  ∃ (i : fin n → fin (2 * n - 1)), function.injective i ∧
    ∀ k l : fin n, k ≠ l → a (i k) - a (i l) ≠ t

theorem problem_proof (n : ℕ) (t : ℝ) (a : fin (2 * n - 1) → ℝ) (hn : 0 < n) (ht : t ≠ 0) :
  exists_distinct_indices n t a :=
sorry

end problem_proof_l21_21969


namespace coefficient_of_x3_l21_21446

theorem coefficient_of_x3 
  (n : ℕ)
  (h₁ : (3 + -1)^n = 128)
  : (∃ c, c = 945 ∧ ∃ k, 14 - ((11 * k) / 4) = 3 ∧ (-1)^4 * 3^(7 - k) * choose 7 k = c) :=
by
  have h_value_of_n : n = 7 := by sorry
  have h_value_of_r : k = 4 := by sorry
  use 945
  split
  · sorry
  · use 4
    split
    · sorry
    · sorry


end coefficient_of_x3_l21_21446


namespace intersection_eq_l21_21183

def M (x : ℝ) : Prop := (x + 3) * (x - 2) < 0

def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

def intersection (x : ℝ) : Prop := M x ∧ N x

theorem intersection_eq : ∀ x, intersection x ↔ (1 ≤ x ∧ x < 2) :=
by sorry

end intersection_eq_l21_21183


namespace problem_part1_l21_21883

def f (x : ℝ) (a : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * (x ^ 2 - 2 * x)

theorem problem_part1 : f 2 Real.exp = 0 := by sorry

end problem_part1_l21_21883


namespace cot_difference_abs_eq_sqrt3_l21_21132

theorem cot_difference_abs_eq_sqrt3 
  (A B C D P : Point) (x y : ℝ) (h1 : is_triangle A B C) 
  (h2 : is_median A D B C) (h3 : ∠(D, A, P) = 60)
  (BD_eq_CD : BD = x) (CD_eq_x : CD = x)
  (BP_eq_y : BP = y) (AP_eq_sqrt3 : AP = sqrt(3) * (x + y))
  (cot_B : cot B = -y / ((sqrt 3) * (x + y)))
  (cot_C : cot C = (2 * x + y) / (sqrt 3 * (x + y))) 
  (x_y_neq_zero : x + y ≠ 0) :
  abs (cot B - cot C) = sqrt 3
  := sorry

end cot_difference_abs_eq_sqrt3_l21_21132


namespace find_m_l21_21076

variable {m : ℝ}

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vector_diff (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_m (hm: dot_product vector_a (vector_diff vector_a (vector_b m)) = 0) : m = 3 :=
  by
  sorry

end find_m_l21_21076


namespace table_price_l21_21343

theorem table_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : C + T = 60) :
  T = 52.5 :=
by
  sorry

end table_price_l21_21343


namespace honey_earned_per_day_l21_21899

theorem honey_earned_per_day :
  ∀ (E : ℝ), (20 * E = 1360 + 240) → E = 80 :=
by
  intro E
  intro h
  simp at h
  linarith

end honey_earned_per_day_l21_21899


namespace find_a_l21_21947

theorem find_a 
  (x y z a : ℤ)
  (h1 : z + a = -2)
  (h2 : y + z = 1)
  (h3 : x + y = 0) : 
  a = -2 := 
  by 
    sorry

end find_a_l21_21947


namespace diff_f_values_l21_21596

def sigma (n : ℕ) : ℕ := 
  (Finset.range (n+1)).filter (fun k => n % k = 0).sum id

noncomputable def f (n : ℕ) : ℚ := (sigma n : ℚ) / n

theorem diff_f_values : f 512 - f 256 = 1 / 512 := 
by
  sorry

end diff_f_values_l21_21596


namespace sqrt_expression_exists_l21_21400

theorem sqrt_expression_exists :
  ∃ (a b c : ℤ), (c ≠ 0 ∧  ∃ d : ℕ, nat.prime d ∧ ¬ (∃ e : ℕ, e * e = d)) ∧ 
  (132 + 46 * Real.sqrt 11 = (a : ℝ) + (b : ℝ) * Real.sqrt c) :=
by
  sorry

end sqrt_expression_exists_l21_21400


namespace find_a_l21_21433

theorem find_a (a : ℝ) 
  (z1 : ℂ := a + 2 * Complex.i) 
  (z2 : ℂ := a + (a + 3) * Complex.i) 
  (h : (z1 * z2).re > 0 ∧ (z1 * z2).im = 0) :
  a = -5 := 
begin
  sorry
end

end find_a_l21_21433


namespace circle_radius_l21_21835

/-- Consider a circle with two chords of lengths 10 cm and 12 cm. 
The distance from the midpoint of the shorter chord (10 cm) to the longer chord (12 cm) is 4 cm. 
Prove that the radius of the circle is 6.25 cm. -/
theorem circle_radius (r : ℝ) :
  ∃ (AB BC MN : ℝ),
  AB = 10 ∧ BC = 12 ∧ MN = 4 ∧
  (∃ M B : ℝ, related_by_pythagoras AB BC MN M B ∧ r = 6.25) :=
sorry

/--
Auxiliary theorem that helps with defining the relationship based on Pythagorean theorem
for the purposes of the main proof.
--/
def related_by_pythagoras (AB BC MN M B : ℝ) : Prop :=
  let BM := AB / 2 in
  let BK := BC / 2 in
  let BN := Math.sqrt (BM ^ 2 - MN ^ 2) in   -- Based on Pythagorean theorem in triangle MNB
  BN = 3 ∧ BK = 6

end circle_radius_l21_21835


namespace concentric_circles_chord_probability_l21_21689

noncomputable def probability_chord_intersects_inner_circle : ℝ := 73.74 / 360

theorem concentric_circles_chord_probability :
  ∀ (R1 R2 : ℝ), (R1 = 3 ∧ R2 = 5) → 
  ∃ (prob : ℝ), prob = probability_chord_intersects_inner_circle ∧
  prob = 73.74 / 360 :=
by
  intros R1 R2 h
  cases h with h1 h2
  use 73.74 / 360
  split
  swap
  refl
  sorry

end concentric_circles_chord_probability_l21_21689


namespace count_perfect_squares_between_100_500_l21_21490

theorem count_perfect_squares_between_100_500 :
  ∃ (count : ℕ), count = finset.card ((finset.Icc 11 22).filter (λ n, 100 < n^2 ∧ n^2 < 500)) :=
begin
  use 12,
  rw ← finset.card_Icc,
  sorry
end

end count_perfect_squares_between_100_500_l21_21490


namespace domain_of_f_l21_21646

def f (x : ℝ) : ℝ := (Real.sqrt (2 - x)) / (Real.log x / Real.log 2)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1} = {x : ℝ | 0 < x ∧ Real.sqrt (2 - x) / (Real.log x / Real.log 2) = f x} :=
by
  sorry

end domain_of_f_l21_21646


namespace first_player_wins_under_optimal_play_l21_21932

-- Define the conditions of the problem
def grid_width := 49
def grid_height := 69
def vertices := (grid_width + 1) * (grid_height + 1)

-- Winning condition for the first player
def first_player_wins (optimal_play : Prop) : Prop :=
  ∃ (pair_strategy : vertex → vertex → Prop), 
    (∀ (v : vertex), ∃ (v1 v2 : vertex), pair_strategy v1 v2) →
    (∀ (segment_directions : vertex × vertex → ℤ × ℤ), 
      (∑ segment, segment_directions segment = (0, 0)))
      → optimal_play

-- Statement of the proof without the implementation of the solution
theorem first_player_wins_under_optimal_play : first_player_wins optimal_play :=
sorry

end first_player_wins_under_optimal_play_l21_21932


namespace cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l21_21923

theorem cos_B_arithmetic_sequence (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180) :
  Real.cos B = 1 / 2 :=
by
  sorry

theorem sin_A_sin_C_geometric_sequence (A B C a b c : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180)
  (h3 : b^2 = a * c) (h4 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :
  Real.sin A * Real.sin C = 3 / 4 :=
by
  sorry

end cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l21_21923


namespace area_inside_circle_outside_square_l21_21744

noncomputable def square_side_length := 2
noncomputable def circle_radius := 1

theorem area_inside_circle_outside_square (s : ℝ) (r : ℝ) (area_circle : ℝ) (area_square : ℝ) :
  s = square_side_length → r = circle_radius → 
  area_circle = (real.pi * r * r) → area_square = (s * s) →
  (area_circle - (area_square / 2)) = real.pi / 4 :=
by
  intros
  sorry

end area_inside_circle_outside_square_l21_21744


namespace gyms_treadmills_count_l21_21730

variable (num_gyms : ℕ) (num_bikes : ℕ) (num_ellipticals : ℕ)
variable (bike_cost : ℕ) (total_cost : ℕ)

def treadmill_cost : ℕ := bike_cost + (bike_cost / 2)
def elliptical_cost : ℕ := 2 * treadmill_cost

def total_bike_cost_in_one_gym : ℕ := num_bikes * bike_cost
def total_elliptical_cost_in_one_gym : ℕ := num_ellipticals * elliptical_cost

def total_bike_cost : ℕ := num_gyms * total_bike_cost_in_one_gym
def total_elliptical_cost : ℕ := num_gyms * total_elliptical_cost_in_one_gym

def total_treadmill_cost : ℕ := total_cost - (total_bike_cost + total_elliptical_cost)

def treadmill_cost_per_gym : ℕ := total_treadmill_cost / num_gyms
def treadmills_per_gym : ℕ := treadmill_cost_per_gym / treadmill_cost

-- The statement that encapsulates the proof problem
theorem gyms_treadmills_count
(h1 : num_gyms = 20)
(h2 : num_bikes = 10)
(h3 : num_ellipticals = 5)
(h4 : bike_cost = 700)
(h5 : total_cost = 455000) :
  treadmills_per_gym = 5 := 
sorry

end gyms_treadmills_count_l21_21730


namespace problem_1_problem_2_l21_21036

variable (a : ℕ → ℝ)

variables (h1 : ∀ n, 0 < a n) (h2 : ∀ n, a (n + 1) + 1 / a n < 2)

-- Prove that: (1) a_{n+2} < a_{n+1} < 2 for n ∈ ℕ*
theorem problem_1 (n : ℕ) : a (n + 2) < a (n + 1) ∧ a (n + 1) < 2 := 
sorry

-- Prove that: (2) a_n > 1 for n ∈ ℕ*
theorem problem_2 (n : ℕ) : 1 < a n := 
sorry

end problem_1_problem_2_l21_21036


namespace not_taking_ship_probability_l21_21738

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end not_taking_ship_probability_l21_21738


namespace major_arc_circumference_l21_21970

noncomputable def circumference_major_arc 
  (A B C : Point) (r : ℝ) (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) : ℝ :=
  let total_circumference := 2 * Real.pi * r
  let major_arc_angle := 360 - angle_ACB
  major_arc_angle / 360 * total_circumference

theorem major_arc_circumference (A B C : Point) (r : ℝ)
  (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) :
  circumference_major_arc A B C r angle_ACB h1 h2 = (500 / 3) * Real.pi :=
  sorry

end major_arc_circumference_l21_21970


namespace determine_f2_l21_21879

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (5^x) = x

theorem determine_f2 : f 2 = log 5 2 := by
  sorry

end determine_f2_l21_21879


namespace number_of_unique_even_integers_l21_21900

def is_even (n : ℕ) : Prop := n % 2 = 0
def unique_digits (n : ℕ) : Prop := 
  let digits := (nat.digits 10 n).nodup
  digits

theorem number_of_unique_even_integers : ∃ cnt, cnt = 1008 ∧ ∀ n, 
  5000 ≤ n ∧ n ≤ 9000 ∧ is_even n ∧ unique_digits n → true :=
begin
  sorry
end

end number_of_unique_even_integers_l21_21900


namespace tangent_sum_half_angles_l21_21622

-- Lean statement for the proof problem
theorem tangent_sum_half_angles (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.tan (A / 2) * Real.tan (B / 2) + 
  Real.tan (B / 2) * Real.tan (C / 2) + 
  Real.tan (C / 2) * Real.tan (A / 2) = 1 := 
by
  sorry

end tangent_sum_half_angles_l21_21622


namespace num_of_cows_is_7_l21_21302

variables (C H : ℕ)

-- Define the conditions
def cow_legs : ℕ := 4 * C
def chicken_legs : ℕ := 2 * H
def cow_heads : ℕ := C
def chicken_heads : ℕ := H

def total_legs : ℕ := cow_legs C + chicken_legs H
def total_heads : ℕ := cow_heads C + chicken_heads H
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 14

-- The theorem to be proved
theorem num_of_cows_is_7 (h : legs_condition C H) : C = 7 :=
by sorry

end num_of_cows_is_7_l21_21302


namespace current_average_age_l21_21678

theorem current_average_age 
  (Y : ℕ) (Y = 5)
  (num_members : ℕ) (num_members = 7)
  (average_age_6_years_ago : ℕ) (average_age_6_years_ago = 28)
  (total_age_6_years_ago : ℕ) (total_age_6_years_ago = 6 * average_age_6_years_ago)
  :
  let age_of_6_members_now := total_age_6_years_ago + 6 * 6,
      total_age_now := age_of_6_members_now + Y,
      average_age_now := total_age_now / num_members
  in average_age_now = 29.857 := sorry

end current_average_age_l21_21678


namespace part1_part2_part3_l21_21590

def S (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2
def a (n : ℕ) : ℕ := 3 ^ n
def b (n : ℕ) : ℕ := 2 * a n / (a n - 2)^2
def T (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

theorem part1 (n : ℕ) : ∀ n, a n = 3^n := by
  sorry

theorem part2 : ∀ n, (S (2 * n) + 15) / a n ≥ 9 := by
  sorry

theorem part3 (n : ℕ) : T n < 13 / 2 := by
  sorry

end part1_part2_part3_l21_21590


namespace typist_current_salary_l21_21668

theorem typist_current_salary (original_salary : ℝ) (raise_percent : ℝ) (reduce_percent : ℝ) (current_salary : ℝ) :
  original_salary = 5000 →
  raise_percent = 0.10 →
  reduce_percent = 0.05 →
  current_salary = original_salary * (1 + raise_percent) * (1 - reduce_percent) →
  current_salary = 5225 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end typist_current_salary_l21_21668


namespace projection_correct_l21_21895

open Real
open Finset

-- Define the vectors a and b 
def a : EuclideanSpace ℝ (Fin 2) := ![1, 2]
def b : EuclideanSpace ℝ (Fin 2) := ![-1, 3]

-- Define the projection operation
def proj (u v : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (dot_product u v / (dot_product v v)) • v

-- The specific projection statement to prove
theorem projection_correct :
  proj a b = ![-(1/2 : ℝ), 3/2] :=
  sorry

end projection_correct_l21_21895


namespace inequality_bound_l21_21056

theorem inequality_bound (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ e^x * (x^2 - x + 1) * (a * x + 3 * a - 1) < 1) : a < 2 / 3 :=
by
  sorry

end inequality_bound_l21_21056


namespace eccentricity_of_ellipse_l21_21647

-- Define the equation of the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 5 = 1)

-- Define the eccentricity of the ellipse
def eccentricity (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^'2 - b^2)
  c / a

-- Definitions of constants based on the problem
def a : ℝ := 3
def b : ℝ := Real.sqrt 5

-- Problem statement
theorem eccentricity_of_ellipse : 
  (∃ (e : ℝ), e = eccentricity a b ∧ e = 2/3) :=
sorry

end eccentricity_of_ellipse_l21_21647


namespace circle_radius_three_points_on_line_l21_21871

theorem circle_radius_three_points_on_line :
  ∀ R : ℝ,
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = R^2 → (4 * x + 3 * y = 11) → (dist (x, y) (1, -1) = 1)) →
  R = 3
:= sorry

end circle_radius_three_points_on_line_l21_21871


namespace find_a_l21_21254

noncomputable theory

def f (a x : ℝ) : ℝ := log (1 / 2) (a * x^2 - 2 * x + 4) 

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, f a x ≤ 1): 
  a = 2 / 7 :=
by
  have h_nonneg : ∀ x : ℝ, a * x^2 - 2 * x + 4 > 0, from sorry,
  have h_min : a * (1 / (2a))^2 - 2 * (1 / a) + 4 = 1 / 2, from sorry,
  have h_pos_a : a > 0, from sorry,
  -- given these conditions, we need to conclude a = 2 / 7
  sorry

end find_a_l21_21254


namespace angle_WTS_l21_21942

theorem angle_WTS {P Q R S T U V W : Type} [Square PQRS] [EquilateralTriangle PST] [RegularPentagon SRUVW] :
  angle W T S = 39 :=
sorry

end angle_WTS_l21_21942


namespace count_distinct_integers_l21_21080

theorem count_distinct_integers (c : ℕ) :
  (∃ n : finset ℕ, 
    ∀ x ∈ n, 2000 ≤ x ∧ x ≤ 9999 ∧ 
    (id.digits x).nodup ∧ 
    (id.digits x).nth 0 % 2 = 0 ∧ 
    (id.digits x).nth 2 % 2 = 1 ∧ 
    n.card = c) ↔ c = 756 :=
begin
  sorry
end

end count_distinct_integers_l21_21080


namespace food_additives_budget_allocation_l21_21324

-- Definitions of the given conditions
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 19
def gmo_percentage : ℝ := 24
def industrial_lubricants_percentage : ℝ := 8
def basic_astrophysics_degrees : ℝ := 90
def full_circle_degrees : ℝ := 360
def full_budget_percentage : ℝ := 100
def food_additives_percentage : ℝ := 10

-- Condition expressions
def total_other_percentages : ℝ := microphotonics_percentage + home_electronics_percentage + gmo_percentage + industrial_lubricants_percentage
def combined_food_and_astrophysics_percentage : ℝ := full_budget_percentage - total_other_percentages
def basic_astrophysics_percentage : ℝ := (basic_astrophysics_degrees / full_circle_degrees) * full_budget_percentage

-- Theorem to prove
theorem food_additives_budget_allocation :
  combined_food_and_astrophysics_percentage - basic_astrophysics_percentage = food_additives_percentage :=
by
  sorry

end food_additives_budget_allocation_l21_21324


namespace sale_price_of_television_l21_21194

theorem sale_price_of_television :
  ∀ (regular_price sale_price : ℝ), 
    regular_price = 600 → 
    sale_price = regular_price - 0.20 * regular_price → 
    sale_price = 480 :=
begin
  sorry
end

end sale_price_of_television_l21_21194
