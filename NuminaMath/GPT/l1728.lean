import Mathlib

namespace base_triangle_not_equilateral_l1728_172839

-- Define the lengths of the lateral edges
def SA := 1
def SB := 2
def SC := 4

-- Main theorem: the base triangle is not equilateral
theorem base_triangle_not_equilateral 
  (a : ℝ)
  (equilateral : a = a)
  (triangle_inequality1 : SA + SB > a)
  (triangle_inequality2 : SA + a > SC) : 
  a ≠ a :=
by 
  sorry

end base_triangle_not_equilateral_l1728_172839


namespace min_distance_sq_l1728_172876

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end min_distance_sq_l1728_172876


namespace age_of_20th_student_l1728_172863

theorem age_of_20th_student (avg_age_20 : ℕ) (avg_age_9 : ℕ) (avg_age_10 : ℕ) :
  (avg_age_20 = 20) →
  (avg_age_9 = 11) →
  (avg_age_10 = 24) →
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  (age_20th = 61) :=
by
  intros h1 h2 h3
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  sorry

end age_of_20th_student_l1728_172863


namespace while_loop_output_correct_do_while_loop_output_correct_l1728_172837

def while_loop (a : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (7 - i)).map (λ n => (i + n, a + n + 1))

def do_while_loop (x : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (10 - i + 1)).map (λ n => (i + n, x + (n + 1) * 10))

theorem while_loop_output_correct : while_loop 2 1 = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)] := 
sorry

theorem do_while_loop_output_correct : do_while_loop 100 1 = [(1, 110), (2, 120), (3, 130), (4, 140), (5, 150), (6, 160), (7, 170), (8, 180), (9, 190), (10, 200)] :=
sorry

end while_loop_output_correct_do_while_loop_output_correct_l1728_172837


namespace find_certain_age_l1728_172879

theorem find_certain_age 
(Kody_age : ℕ) 
(Mohamed_age : ℕ) 
(certain_age : ℕ) 
(h1 : Kody_age = 32) 
(h2 : Mohamed_age = 2 * certain_age) 
(h3 : ∀ four_years_ago, four_years_ago = Kody_age - 4 → four_years_ago * 2 = Mohamed_age - 4) :
  certain_age = 30 := sorry

end find_certain_age_l1728_172879


namespace chocolate_and_gum_l1728_172868

/--
Kolya says that two chocolate bars are more expensive than five gum sticks, 
while Sasha claims that three chocolate bars are more expensive than eight gum sticks. 
When this was checked, only one of them was right. Is it true that seven chocolate bars 
are more expensive than nineteen gum sticks?
-/
theorem chocolate_and_gum (c g : ℝ) (hk : 2 * c > 5 * g) (hs : 3 * c > 8 * g) (only_one_correct : ¬((2 * c > 5 * g) ∧ (3 * c > 8 * g)) ∧ (2 * c > 5 * g ∨ 3 * c > 8 * g)) : 7 * c < 19 * g :=
by
  sorry

end chocolate_and_gum_l1728_172868


namespace sum_series_eq_three_l1728_172871

theorem sum_series_eq_three : 
  ∑' (k : ℕ), (k^2 : ℝ) / (2^k : ℝ) = 3 := sorry

end sum_series_eq_three_l1728_172871


namespace xyz_value_l1728_172844

variable (x y z : ℝ)

theorem xyz_value :
  (x + y + z) * (x*y + x*z + y*z) = 36 →
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24 →
  x * y * z = 4 :=
by
  intros h1 h2
  sorry

end xyz_value_l1728_172844


namespace Diane_bakes_160_gingerbreads_l1728_172811

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end Diane_bakes_160_gingerbreads_l1728_172811


namespace total_afternoon_evening_emails_l1728_172807

-- Definitions based on conditions
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

-- Statement to be proven
theorem total_afternoon_evening_emails : afternoon_emails + evening_emails = 13 :=
by 
  sorry

end total_afternoon_evening_emails_l1728_172807


namespace find_slower_speed_l1728_172885

-- Variables and conditions definitions
variable (v : ℝ)

def slower_speed (v : ℝ) : Prop :=
  (20 / v = 2) ∧ (v = 10)

-- The statement to be proven
theorem find_slower_speed : slower_speed 10 :=
by
  sorry

end find_slower_speed_l1728_172885


namespace correct_statement_B_l1728_172813

/-- Define the diameter of a sphere -/
def diameter (d : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop :=
∃ x y : ℝ × ℝ × ℝ, x ∈ s ∧ y ∈ s ∧ dist x y = d ∧ ∀ z ∈ s, dist x y ≥ dist x z ∧ dist x y ≥ dist z y

/-- Define that a line segment connects two points on the sphere's surface and passes through the center -/
def connects_diameter (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ) : Prop :=
dist center x = radius ∧ dist center y = radius ∧ (x + y) / 2 = center

/-- A sphere is the set of all points at a fixed distance from the center -/
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
{x | dist center x = radius}

theorem correct_statement_B (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ):
  (∀ (s : Set (ℝ × ℝ × ℝ)), sphere center radius = s → diameter (2 * radius) s)
  → connects_diameter center radius x y
  → (∃ d : ℝ, diameter d (sphere center radius)) := 
by
  intros
  sorry

end correct_statement_B_l1728_172813


namespace remainder_g_x12_div_g_x_l1728_172806

-- Define the polynomial g
noncomputable def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Proving the remainder when g(x^12) is divided by g(x) is 6
theorem remainder_g_x12_div_g_x : 
  (g (x^12) % g x) = 6 :=
sorry

end remainder_g_x12_div_g_x_l1728_172806


namespace horner_method_multiplications_additions_count_l1728_172862

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - 2 * x^2 + 4 * x - 6

-- Define the property we want to prove
theorem horner_method_multiplications_additions_count : 
  ∃ (multiplications additions : ℕ), multiplications = 4 ∧ additions = 4 := 
by
  sorry

end horner_method_multiplications_additions_count_l1728_172862


namespace intersection_with_x_axis_l1728_172872

theorem intersection_with_x_axis (t : ℝ) (x y : ℝ) 
  (h1 : x = -2 + 5 * t) 
  (h2 : y = 1 - 2 * t) 
  (h3 : y = 0) : x = 1 / 2 := 
by 
  sorry

end intersection_with_x_axis_l1728_172872


namespace count_divisors_of_100000_l1728_172851

theorem count_divisors_of_100000 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (k ∣ 100000) → ∃ (i j : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 ∧ k = 2^i * 5^j := by
  sorry

end count_divisors_of_100000_l1728_172851


namespace symmetric_axis_of_quadratic_fn_l1728_172860

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 9 

-- State the theorem that the axis of symmetry for the quadratic function y = x^2 + 8x + 9 is x = -4
theorem symmetric_axis_of_quadratic_fn : ∃ h : ℝ, h = -4 ∧ ∀ x, quadratic_function x = quadratic_function (2 * h - x) :=
by sorry

end symmetric_axis_of_quadratic_fn_l1728_172860


namespace cart_total_distance_l1728_172878

-- Definitions for the conditions
def first_section_distance := (15/2) * (8 + (8 + 14 * 10))
def second_section_distance := (15/2) * (148 + (148 + 14 * 6))

-- Combining both distances
def total_distance := first_section_distance + second_section_distance

-- Statement to be proved
theorem cart_total_distance:
  total_distance = 4020 :=
by
  sorry

end cart_total_distance_l1728_172878


namespace emma_bank_account_balance_l1728_172843

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l1728_172843


namespace number_of_non_congruent_triangles_l1728_172842

theorem number_of_non_congruent_triangles :
  ∃ q : ℕ, q = 3 ∧ 
    (∀ (a b : ℕ), (a ≤ 2 ∧ 2 ≤ b) → (a + 2 > b) ∧ (a + b > 2) ∧ (2 + b > a) →
    (q = 3)) :=
by
  sorry

end number_of_non_congruent_triangles_l1728_172842


namespace determine_b_for_constant_remainder_l1728_172877

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end determine_b_for_constant_remainder_l1728_172877


namespace good_apples_count_l1728_172832

theorem good_apples_count (total_apples : ℕ) (rotten_percentage : ℝ) (good_apples : ℕ) (h1 : total_apples = 75) (h2 : rotten_percentage = 0.12) :
  good_apples = (1 - rotten_percentage) * total_apples := by
  sorry

end good_apples_count_l1728_172832


namespace totalStudents_correct_l1728_172887

-- Defining the initial number of classes, students per class, and new classes
def initialClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def newClasses : ℕ := 5

-- Prove that the total number of students is 400
theorem totalStudents_correct : 
  initialClasses * studentsPerClass + newClasses * studentsPerClass = 400 := by
  sorry

end totalStudents_correct_l1728_172887


namespace inequality_proof_l1728_172873

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (habc : a * b * c = 1)

theorem inequality_proof :
  (a + 1 / b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 ≥ 3 * (a + b + c + 1) :=
by
  sorry

end inequality_proof_l1728_172873


namespace min_value_of_sequence_l1728_172852

theorem min_value_of_sequence :
  ∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → |a n| = |a (n - 1) + 1|) ∧ (a 1 + a 2 + a 3 + a 4 = -2) :=
by
  sorry

end min_value_of_sequence_l1728_172852


namespace sum_first_12_terms_l1728_172864

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 ^ (n - 1) else 2 * n - 1

def S (n : ℕ) : ℕ := 
  (Finset.range n).sum a

theorem sum_first_12_terms : S 12 = 1443 :=
by
  sorry

end sum_first_12_terms_l1728_172864


namespace triangle_leg_ratio_l1728_172831

theorem triangle_leg_ratio :
  ∀ (a b : ℝ) (h₁ : a = 4) (h₂ : b = 2 * Real.sqrt 5),
    ((a / b) = (2 * Real.sqrt 5) / 5) :=
by
  intros a b h₁ h₂
  sorry

end triangle_leg_ratio_l1728_172831


namespace remaining_fuel_relation_l1728_172898

-- Define the car's travel time and remaining fuel relation
def initial_fuel : ℝ := 100

def fuel_consumption_rate : ℝ := 6

def remaining_fuel (t : ℝ) : ℝ := initial_fuel - fuel_consumption_rate * t

-- Prove that the remaining fuel after t hours is given by the linear relationship Q = 100 - 6t
theorem remaining_fuel_relation (t : ℝ) : remaining_fuel t = 100 - 6 * t := by
  -- Proof is omitted, as per instructions
  sorry

end remaining_fuel_relation_l1728_172898


namespace intersection_points_polar_coords_l1728_172847

theorem intersection_points_polar_coords :
  (∀ (x y : ℝ), ((x - 4)^2 + (y - 5)^2 = 25 ∧ (x^2 + y^2 - 2*y = 0)) →
  (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((x, y) = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ((ρ = 2 ∧ θ = Real.pi / 2) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4)))) :=
sorry

end intersection_points_polar_coords_l1728_172847


namespace black_white_ratio_l1728_172853

theorem black_white_ratio 
  (x y : ℕ) 
  (h1 : (y - 1) * 7 = x * 9) 
  (h2 : y * 5 = (x - 1) * 7) : 
  y - x = 7 := 
by 
  sorry

end black_white_ratio_l1728_172853


namespace sphere_surface_area_of_solid_l1728_172880

theorem sphere_surface_area_of_solid (l w h : ℝ) (hl : l = 2) (hw : w = 1) (hh : h = 2) 
: 4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 9 * Real.pi := 
by 
  sorry

end sphere_surface_area_of_solid_l1728_172880


namespace harry_travel_ratio_l1728_172809

theorem harry_travel_ratio
  (bus_initial_time : ℕ)
  (bus_rest_time : ℕ)
  (total_travel_time : ℕ)
  (walking_time : ℕ := total_travel_time - (bus_initial_time + bus_rest_time))
  (bus_total_time : ℕ := bus_initial_time + bus_rest_time)
  (ratio : ℚ := walking_time / bus_total_time)
  (h1 : bus_initial_time = 15)
  (h2 : bus_rest_time = 25)
  (h3 : total_travel_time = 60)
  : ratio = (1 / 2) := 
sorry

end harry_travel_ratio_l1728_172809


namespace find_x_l1728_172856

namespace MathProof

variables {a b x : ℝ}
variables (h1 : a > 0) (h2 : b > 0)

theorem find_x (h3 : (a^2)^(2 * b) = a^b * x^b) : x = a^3 :=
by sorry

end MathProof

end find_x_l1728_172856


namespace total_boys_in_class_l1728_172816

theorem total_boys_in_class (n : ℕ) (h_circle : ∀ i, 1 ≤ i ∧ i ≤ n -> i ≤ n) 
  (h_opposite : ∀ j k, j = 7 ∧ k = 27 ∧ j < k -> (k - j = n / 2)) : 
  n = 40 :=
sorry

end total_boys_in_class_l1728_172816


namespace sofia_running_time_l1728_172814

theorem sofia_running_time :
  let distance_first_section := 100 -- meters
  let speed_first_section := 5 -- meters per second
  let distance_second_section := 300 -- meters
  let speed_second_section := 4 -- meters per second
  let num_laps := 6
  let time_first_section := distance_first_section / speed_first_section -- in seconds
  let time_second_section := distance_second_section / speed_second_section -- in seconds
  let time_per_lap := time_first_section + time_second_section -- in seconds
  let total_time_seconds := num_laps * time_per_lap -- in seconds
  let total_time_minutes := total_time_seconds / 60 -- integer division for minutes
  let remaining_seconds := total_time_seconds % 60 -- modulo for remaining seconds
  total_time_minutes = 9 ∧ remaining_seconds = 30 := 
  by
  sorry

end sofia_running_time_l1728_172814


namespace simplify_expression_l1728_172858

theorem simplify_expression (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y :=
by
  sorry

end simplify_expression_l1728_172858


namespace number_of_ordered_pairs_l1728_172815

theorem number_of_ordered_pairs (h : ∀ (m n : ℕ), 0 < m → 0 < n → 6/m + 3/n = 1 → true) : 
∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x : ℕ × ℕ), x ∈ s → 0 < x.1 ∧ 0 < x.2 ∧ 6 / ↑x.1 + 3 / ↑x.2 = 1 :=
by
-- Sorry, skipping the proof
  sorry

end number_of_ordered_pairs_l1728_172815


namespace general_term_of_sequence_l1728_172870

theorem general_term_of_sequence (n : ℕ) :
  ∃ (a : ℕ → ℚ),
    a 1 = 1 / 2 ∧ 
    a 2 = -2 ∧ 
    a 3 = 9 / 2 ∧ 
    a 4 = -8 ∧ 
    a 5 = 25 / 2 ∧ 
    ∀ n, a n = (-1) ^ (n + 1) * (n ^ 2 / 2) := 
by
  sorry

end general_term_of_sequence_l1728_172870


namespace rob_total_cards_l1728_172803

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l1728_172803


namespace sufficient_not_necessary_l1728_172893

theorem sufficient_not_necessary (x : ℝ) (h1 : -1 < x) (h2 : x < 3) :
    x^2 - 2*x < 8 :=
by
    -- Proof to be filled in.
    sorry

end sufficient_not_necessary_l1728_172893


namespace prove_value_of_expressions_l1728_172823

theorem prove_value_of_expressions (a b : ℕ) 
  (h₁ : 2^a = 8^b) 
  (h₂ : a + 2 * b = 5) : 
  2^a + 8^b = 16 := 
by 
  -- proof steps go here
  sorry

end prove_value_of_expressions_l1728_172823


namespace Rajesh_work_completion_time_l1728_172812

-- Definitions based on conditions in a)
def Mahesh_rate := 1 / 60 -- Mahesh's rate of work (work per day)
def Mahesh_work := 20 * Mahesh_rate -- Work completed by Mahesh in 20 days
def Rajesh_time_to_complete_remaining_work := 30 -- Rajesh time to complete remaining work (days)
def Remaining_work := 1 - Mahesh_work -- Remaining work after Mahesh's contribution

-- Statement that needs to be proved
theorem Rajesh_work_completion_time :
  (Rajesh_time_to_complete_remaining_work : ℝ) * (1 / Remaining_work) = 45 :=
sorry

end Rajesh_work_completion_time_l1728_172812


namespace inv_f_of_neg3_l1728_172841

def f (x : Real) : Real := 5 - 2 * x

theorem inv_f_of_neg3 : f⁻¹ (-3) = 4 :=
by
  sorry

end inv_f_of_neg3_l1728_172841


namespace sandy_balloons_l1728_172874

def balloons_problem (A S T : ℕ) : ℕ :=
  T - (A + S)

theorem sandy_balloons : balloons_problem 37 39 104 = 28 := by
  sorry

end sandy_balloons_l1728_172874


namespace pencil_case_costs_l1728_172801

variable {x y : ℝ}

theorem pencil_case_costs :
  (2 * x + 3 * y = 108) ∧ (5 * x = 6 * y) → 
  (x = 24) ∧ (y = 20) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  sorry

end pencil_case_costs_l1728_172801


namespace minimize_expression_l1728_172819

theorem minimize_expression (x y : ℝ) (k : ℝ) (h : k = -1) : (xy + k)^2 + (x - y)^2 ≥ 0 ∧ (∀ x y : ℝ, (xy + k)^2 + (x - y)^2 = 0 ↔ k = -1) := 
by {
  sorry
}

end minimize_expression_l1728_172819


namespace sin_value_l1728_172845

theorem sin_value (α : ℝ) (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (α + Real.pi / 6) = -3 / 5) : 
  Real.sin (2 * α + Real.pi / 12) = -17 * Real.sqrt 2 / 50 := 
sorry

end sin_value_l1728_172845


namespace joan_games_attended_l1728_172846
-- Mathematical definitions based on the provided conditions

def total_games_played : ℕ := 864
def games_missed_by_Joan : ℕ := 469

-- Theorem statement
theorem joan_games_attended : total_games_played - games_missed_by_Joan = 395 :=
by
  -- Proof omitted
  sorry

end joan_games_attended_l1728_172846


namespace maximum_M_for_right_triangle_l1728_172894

theorem maximum_M_for_right_triangle (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  (1 / a + 1 / b + 1 / c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) :=
sorry

end maximum_M_for_right_triangle_l1728_172894


namespace frog_probability_l1728_172824

noncomputable def frog_escape_prob (P : ℕ → ℚ) : Prop :=
  P 0 = 0 ∧
  P 11 = 1 ∧
  (∀ N, 0 < N ∧ N < 11 → 
    P N = (N + 1) / 12 * P (N - 1) + (1 - (N + 1) / 12) * P (N + 1)) ∧
  P 2 = 72 / 167

theorem frog_probability : ∃ P : ℕ → ℚ, frog_escape_prob P :=
sorry

end frog_probability_l1728_172824


namespace range_of_a_l1728_172867

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, x - a ≥ 0 → 2 * x - 10 < 0) :
  3 < a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l1728_172867


namespace total_distance_traveled_l1728_172892

theorem total_distance_traveled :
  let time1 := 3  -- hours
  let speed1 := 70  -- km/h
  let time2 := 4  -- hours
  let speed2 := 80  -- km/h
  let time3 := 3  -- hours
  let speed3 := 65  -- km/h
  let time4 := 2  -- hours
  let speed4 := 90  -- km/h
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  distance1 + distance2 + distance3 + distance4 = 905 :=
by
  sorry

end total_distance_traveled_l1728_172892


namespace sin_60_eq_sqrt3_div_2_l1728_172802

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l1728_172802


namespace magician_trick_success_l1728_172836

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end magician_trick_success_l1728_172836


namespace Priyanka_chocolates_l1728_172882

variable (N S So P Sa T : ℕ)

theorem Priyanka_chocolates :
  (N + S = 10) →
  (So + P = 15) →
  (Sa + T = 10) →
  (N = 4) →
  ((S = 2 * y) ∨ (P = 2 * So)) →
  P = 10 :=
by
  sorry

end Priyanka_chocolates_l1728_172882


namespace barbara_total_cost_l1728_172859

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l1728_172859


namespace thirteen_pow_2023_mod_1000_l1728_172827

theorem thirteen_pow_2023_mod_1000 :
  (13^2023) % 1000 = 99 :=
sorry

end thirteen_pow_2023_mod_1000_l1728_172827


namespace cone_lateral_area_l1728_172817

/--
Given that the radius of the base of a cone is 3 cm and the slant height is 6 cm,
prove that the lateral area of this cone is 18π cm².
-/
theorem cone_lateral_area {r l : ℝ} (h_radius : r = 3) (h_slant_height : l = 6) :
  (π * r * l) = 18 * π :=
by
  have h1 : r = 3 := h_radius
  have h2 : l = 6 := h_slant_height
  rw [h1, h2]
  norm_num
  sorry

end cone_lateral_area_l1728_172817


namespace proof_problem_l1728_172869

theorem proof_problem (x : ℕ) (h : 320 / (x + 26) = 4) : x = 54 := 
by 
  sorry

end proof_problem_l1728_172869


namespace min_value_geom_seq_l1728_172875

theorem min_value_geom_seq (a : ℕ → ℝ) (r m n : ℕ) (h_geom : ∃ r, ∀ i, a (i + 1) = a i * r)
  (h_ratio : r = 2) (h_a_m : 4 * a 1 = a m) :
  ∃ (m n : ℕ), (m + n = 6) → (1 / m + 4 / n) = 3 / 2 :=
by 
  sorry

end min_value_geom_seq_l1728_172875


namespace files_deleted_l1728_172808

-- Definitions based on the conditions
def initial_files : ℕ := 93
def files_per_folder : ℕ := 8
def num_folders : ℕ := 9

-- The proof problem
theorem files_deleted : initial_files - (files_per_folder * num_folders) = 21 :=
by
  sorry

end files_deleted_l1728_172808


namespace orthogonality_implies_x_value_l1728_172883

theorem orthogonality_implies_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, -1)
  a.1 * b.1 + a.2 * b.2 = 0 → x = 1 :=
sorry

end orthogonality_implies_x_value_l1728_172883


namespace greatest_integer_x_l1728_172822

theorem greatest_integer_x (x : ℤ) : 
  (∃ n : ℤ, (x^2 + 4*x + 10) = n * (x - 4)) → x ≤ 46 := 
by
  sorry

end greatest_integer_x_l1728_172822


namespace olivia_initial_money_l1728_172838

theorem olivia_initial_money (spent_supermarket : ℕ) (spent_showroom : ℕ) (left_money : ℕ) (initial_money : ℕ) :
  spent_supermarket = 31 → spent_showroom = 49 → left_money = 26 → initial_money = spent_supermarket + spent_showroom + left_money → initial_money = 106 :=
by
  intros h_supermarket h_showroom h_left h_initial 
  rw [h_supermarket, h_showroom, h_left] at h_initial
  exact h_initial

end olivia_initial_money_l1728_172838


namespace least_number_subtracted_divisible_l1728_172850

theorem least_number_subtracted_divisible (n : ℕ) (d : ℕ) (h : n = 1234567) (k : d = 37) :
  n % d = 13 :=
by 
  rw [h, k]
  sorry

end least_number_subtracted_divisible_l1728_172850


namespace sqrt_of_9_is_3_l1728_172840

theorem sqrt_of_9_is_3 {x : ℝ} (h₁ : x * x = 9) (h₂ : x ≥ 0) : x = 3 := sorry

end sqrt_of_9_is_3_l1728_172840


namespace smallest_c_value_l1728_172861

theorem smallest_c_value :
  ∃ a b c : ℕ, a * b * c = 3990 ∧ a + b + c = 56 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
by {
  -- Skipping proof as instructed
  sorry
}

end smallest_c_value_l1728_172861


namespace sarah_marry_age_l1728_172826

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end sarah_marry_age_l1728_172826


namespace brick_width_is_10_cm_l1728_172896

-- Define the conditions
def courtyard_length_meters := 25
def courtyard_width_meters := 16
def brick_length_cm := 20
def number_of_bricks := 20000

-- Convert courtyard dimensions to area in square centimeters
def area_of_courtyard_cm2 := courtyard_length_meters * 100 * courtyard_width_meters * 100

-- Total area covered by bricks
def total_brick_area_cm2 := area_of_courtyard_cm2

-- Area covered by one brick
def area_per_brick := total_brick_area_cm2 / number_of_bricks

-- Find the brick width
def brick_width_cm := area_per_brick / brick_length_cm

-- Prove the width of each brick is 10 cm
theorem brick_width_is_10_cm : brick_width_cm = 10 := 
by 
  -- Placeholder for the proof
  sorry

end brick_width_is_10_cm_l1728_172896


namespace original_number_l1728_172820

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def permutations_sum (a b c : ℕ) : ℕ :=
  let abc := 100 * a + 10 * b + c
  let acb := 100 * a + 10 * c + b
  let bac := 100 * b + 10 * a + c
  let bca := 100 * b + 10 * c + a
  let cab := 100 * c + 10 * a + b
  let cba := 100 * c + 10 * b + a
  abc + acb + bac + bca + cab + cba

theorem original_number (abc : ℕ) (a b c : ℕ) :
  is_three_digit abc →
  abc = 100 * a + 10 * b + c →
  permutations_sum a b c = 3194 →
  abc = 358 :=
by
  sorry

end original_number_l1728_172820


namespace cuboid_volume_l1728_172866

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 5) (h3 : a * c = 15) : a * b * c = 15 :=
sorry

end cuboid_volume_l1728_172866


namespace zoo_problem_l1728_172895

variables
  (parrots : ℕ)
  (snakes : ℕ)
  (monkeys : ℕ)
  (elephants : ℕ)
  (zebras : ℕ)
  (f : ℚ)

-- Conditions from the problem
theorem zoo_problem
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : elephants = f * (parrots + snakes))
  (h5 : zebras = elephants - 3)
  (h6 : monkeys - zebras = 35) :
  f = 1 / 2 :=
sorry

end zoo_problem_l1728_172895


namespace multiply_then_divide_eq_multiply_l1728_172855

theorem multiply_then_divide_eq_multiply (x : ℚ) :
  (x * (2 / 5)) / (3 / 7) = x * (14 / 15) :=
by
  sorry

end multiply_then_divide_eq_multiply_l1728_172855


namespace max_blocks_fit_in_box_l1728_172828

def box_dimensions : ℕ × ℕ × ℕ := (4, 6, 2)
def block_dimensions : ℕ × ℕ × ℕ := (3, 2, 1)
def block_volume := 6
def box_volume := 48

theorem max_blocks_fit_in_box (box_dimensions : ℕ × ℕ × ℕ)
    (block_dimensions : ℕ × ℕ × ℕ) : 
  (box_volume / block_volume = 8) := 
by
  sorry

end max_blocks_fit_in_box_l1728_172828


namespace count_distinct_m_in_right_triangle_l1728_172897

theorem count_distinct_m_in_right_triangle (k : ℝ) (hk : k > 0) :
  ∃! m : ℝ, (m = -3/8 ∨ m = -3/4) :=
by
  sorry

end count_distinct_m_in_right_triangle_l1728_172897


namespace part_a_part_b_l1728_172881

-- Definitions for maximum factor increases
def f (n : ℕ) (a : ℕ) : ℚ := sorry
def t (n : ℕ) (a : ℕ) : ℚ := sorry

-- Part (a): Prove the factor increase for exactly 1 blue cube in 100 boxes
theorem part_a : f 100 1 = 2^100 / 100 := sorry

-- Part (b): Prove the factor increase for some integer \( k \) blue cubes in 100 boxes, \( 1 < k \leq 100 \)
theorem part_b (k : ℕ) (hk : 1 < k ∧ k ≤ 100) : t 100 k = 2^100 / (2^100 - k - 1) := sorry

end part_a_part_b_l1728_172881


namespace inequality_a6_b6_l1728_172834

theorem inequality_a6_b6 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end inequality_a6_b6_l1728_172834


namespace number_of_ways_to_assign_roles_l1728_172886

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end number_of_ways_to_assign_roles_l1728_172886


namespace find_f_ln_inv_6_l1728_172891

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

theorem find_f_ln_inv_6 (k : ℝ) (h : f k (Real.log 6) = 1) : f k (Real.log (1 / 6)) = -7 :=
by
  sorry

end find_f_ln_inv_6_l1728_172891


namespace black_squares_covered_by_trominoes_l1728_172884

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

noncomputable def min_trominoes (n : ℕ) : ℕ :=
  ((n + 1) ^ 2) / 4

theorem black_squares_covered_by_trominoes (n : ℕ) (h1 : n ≥ 7) (h2 : is_odd n):
  ∀ n : ℕ, ∃ k : ℕ, k = min_trominoes n :=
by
  sorry

end black_squares_covered_by_trominoes_l1728_172884


namespace left_vertex_of_ellipse_l1728_172857

theorem left_vertex_of_ellipse : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 8 = 0 ∧ x = a - 5) ∧
  2 * b = 8 → left_vertex = (-5, 0) :=
sorry

end left_vertex_of_ellipse_l1728_172857


namespace subtraction_example_l1728_172810

theorem subtraction_example : 34.256 - 12.932 - 1.324 = 20.000 := 
by
  sorry

end subtraction_example_l1728_172810


namespace no_prime_for_equation_l1728_172829

theorem no_prime_for_equation (x k : ℕ) (p : ℕ) (h_prime : p.Prime) (h_eq : x^5 + 2 * x + 3 = p^k) : False := 
sorry

end no_prime_for_equation_l1728_172829


namespace find_original_number_l1728_172849

noncomputable def three_digit_number (d e f : ℕ) := 100 * d + 10 * e + f

/-- Given conditions and the sum S, determine the original three-digit number -/
theorem find_original_number (S : ℕ) (d e f : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9)
  (h2 : 0 ≤ e ∧ e ≤ 9) (h3 : 0 ≤ f ∧ f ≤ 9) (h4 : S = 4321) :
  three_digit_number d e f = 577 :=
sorry


end find_original_number_l1728_172849


namespace find_length_DE_l1728_172818

theorem find_length_DE (AB AC BC : ℝ) (angleA : ℝ) 
                         (DE DF EF : ℝ) (angleD : ℝ) :
  AB = 9 → AC = 11 → BC = 7 →
  angleA = 60 → DE = 3 → DF = 5.5 → EF = 2.5 →
  angleD = 60 →
  DE = 9 * 2.5 / 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end find_length_DE_l1728_172818


namespace starting_number_l1728_172889

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l1728_172889


namespace total_students_playing_one_sport_l1728_172835

noncomputable def students_playing_at_least_one_sport (total_students B S Ba C B_S B_Ba B_C S_Ba C_S C_Ba B_C_S: ℕ) : ℕ :=
  B + S + Ba + C - B_S - B_Ba - B_C - S_Ba - C_S - C_Ba + B_C_S

theorem total_students_playing_one_sport : 
  students_playing_at_least_one_sport 200 50 60 35 80 10 15 20 25 30 5 10 = 130 := by
  sorry

end total_students_playing_one_sport_l1728_172835


namespace functional_eq_zero_function_l1728_172865

theorem functional_eq_zero_function (f : ℝ → ℝ) (k : ℝ) (h : ∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_function_l1728_172865


namespace xy_squared_value_l1728_172899

variable {x y : ℝ}

theorem xy_squared_value :
  (y + 6 = (x - 3)^2) ∧ (x + 6 = (y - 3)^2) ∧ (x ≠ y) → (x^2 + y^2 = 25) := 
by
  sorry

end xy_squared_value_l1728_172899


namespace circle_radius_l1728_172890

theorem circle_radius :
  ∃ c : ℝ × ℝ, 
    c.2 = 0 ∧
    (dist c (2, 3)) = (dist c (3, 7)) ∧
    (dist c (2, 3)) = (Real.sqrt 1717) / 2 :=
by
  sorry

end circle_radius_l1728_172890


namespace find_arrays_l1728_172825

theorem find_arrays (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a ∣ b * c * d - 1 ∧ b ∣ a * c * d - 1 ∧ c ∣ a * b * d - 1 ∧ d ∣ a * b * c - 1 →
  (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
  (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) := by
  sorry

end find_arrays_l1728_172825


namespace xyz_inequality_l1728_172888

theorem xyz_inequality (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x := 
  sorry

end xyz_inequality_l1728_172888


namespace range_of_omega_l1728_172833

noncomputable def function_with_highest_points (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + Real.pi / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
  (h : ∀ x ∈ Set.Icc 0 1, 2 * Real.sin (ω * x + Real.pi / 4) = 2) :
  Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by
  sorry

end range_of_omega_l1728_172833


namespace fraction_value_l1728_172830

theorem fraction_value
  (x y z : ℝ)
  (h1 : x / 2 = y / 3)
  (h2 : y / 3 = z / 5)
  (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  -- Add sorry to skip the proof.
  sorry

end fraction_value_l1728_172830


namespace Greg_more_than_Sharon_l1728_172848

-- Define the harvest amounts
def Greg_harvest : ℝ := 0.4
def Sharon_harvest : ℝ := 0.1

-- Show that Greg harvested 0.3 more acres than Sharon
theorem Greg_more_than_Sharon : Greg_harvest - Sharon_harvest = 0.3 := by
  sorry

end Greg_more_than_Sharon_l1728_172848


namespace simplify_expression_l1728_172804

theorem simplify_expression :
  64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 := 
by
  sorry

end simplify_expression_l1728_172804


namespace not_proportional_l1728_172800

theorem not_proportional (x y : ℕ) :
  (∀ k : ℝ, y ≠ 3 * x - 7 ∧ y ≠ (13 - 4 * x) / 3) → 
  ((y = 3 * x - 7 ∨ y = (13 - 4 * x) / 3) → ¬(∃ k : ℝ, (y = k * x) ∨ (y = k / x))) := sorry

end not_proportional_l1728_172800


namespace isosceles_triangle_perimeter_l1728_172854

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : 2 * a - 3 * b + 5 = 0) (h₂ : 2 * a + 3 * b - 13 = 0) :
  ∃ p : ℝ, p = 7 ∨ p = 8 :=
sorry

end isosceles_triangle_perimeter_l1728_172854


namespace find_AG_l1728_172805

-- Defining constants and variables
variables (DE EC AD BC FB AG : ℚ)
variables (BC_def : BC = (1 / 3) * AD)
variables (FB_def : FB = (2 / 3) * AD)
variables (DE_val : DE = 8)
variables (EC_val : EC = 6)
variables (sum_AD : BC + FB = AD)

-- The theorem statement
theorem find_AG : AG = 56 / 9 :=
by
  -- Placeholder for the proof
  sorry

end find_AG_l1728_172805


namespace find_balcony_seat_cost_l1728_172821

-- Definitions based on conditions
variable (O B : ℕ) -- Number of orchestra tickets and cost of balcony ticket
def orchestra_ticket_cost : ℕ := 12
def total_tickets : ℕ := 370
def total_cost : ℕ := 3320
def tickets_difference : ℕ := 190

-- Lean statement to prove the cost of a balcony seat
theorem find_balcony_seat_cost :
  (2 * O + tickets_difference = total_tickets) ∧
  (orchestra_ticket_cost * O + B * (O + tickets_difference) = total_cost) →
  B = 8 :=
by
  sorry

end find_balcony_seat_cost_l1728_172821
