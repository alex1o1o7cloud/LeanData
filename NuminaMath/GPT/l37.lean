import Mathlib

namespace NUMINAMATH_GPT_h_eq_20_at_y_eq_4_l37_3759

noncomputable def k (y : ℝ) : ℝ := 40 / (y + 5)

noncomputable def h (y : ℝ) : ℝ := 4 * (k⁻¹ y)

theorem h_eq_20_at_y_eq_4 : h 4 = 20 := 
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_h_eq_20_at_y_eq_4_l37_3759


namespace NUMINAMATH_GPT_garden_perimeter_l37_3796

theorem garden_perimeter (w l : ℕ) (garden_width : ℕ) (garden_perimeter : ℕ)
  (garden_area playground_length playground_width : ℕ)
  (h1 : garden_width = 16)
  (h2 : playground_length = 16)
  (h3 : garden_area = 16 * l)
  (h4 : playground_area = w * playground_length)
  (h5 : garden_area = playground_area)
  (h6 : garden_perimeter = 2 * l + 2 * garden_width)
  (h7 : garden_perimeter = 56):
  l = 12 :=
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l37_3796


namespace NUMINAMATH_GPT_find_F1C_CG1_l37_3719

variable {A B C D E F G H E1 F1 G1 H1 : Type*}
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ) (a : ℝ)

axiom convex_quadrilateral (AE EB BF FC CG GD DH HA : ℝ) : 
  AE / EB * BF / FC * CG / GD * DH / HA = 1 

axiom quadrilaterals_similar 
  (E1F1 EF F1G1 FG G1H1 GH H1E1 HE : Prop) :
  E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True)

axiom given_ratio (E1A AH1 : ℝ) (a : ℝ) :
  E1A / AH1 = a

theorem find_F1C_CG1
  (conv : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (parallel_lines : E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True))
  (ratio : E1A / AH1 = a) :
  F1C / CG1 = a := 
sorry

end NUMINAMATH_GPT_find_F1C_CG1_l37_3719


namespace NUMINAMATH_GPT_chord_intersection_probability_l37_3778

theorem chord_intersection_probability
  (points : Finset Point)
  (hp : points.card = 2000)
  (A B C D E : Point)
  (hA : A ∈ points)
  (hB : B ∈ points)
  (hC : C ∈ points)
  (hD : D ∈ points)
  (hE : E ∈ points)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  : probability_chord_intersection := by
    sorry

end NUMINAMATH_GPT_chord_intersection_probability_l37_3778


namespace NUMINAMATH_GPT_blue_whale_tongue_weight_l37_3713

theorem blue_whale_tongue_weight (ton_in_pounds : ℕ) (tons : ℕ) (blue_whale_tongue_weight : ℕ) :
  ton_in_pounds = 2000 → tons = 3 → blue_whale_tongue_weight = tons * ton_in_pounds → blue_whale_tongue_weight = 6000 :=
  by
  intros h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  exact h3

end NUMINAMATH_GPT_blue_whale_tongue_weight_l37_3713


namespace NUMINAMATH_GPT_correct_statement_l37_3782

def synthetic_method_is_direct : Prop := -- define the synthetic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

def analytic_method_is_direct : Prop := -- define the analytic method
  True  -- We'll say True to assume it's a direct proof method. This is a simplification.

theorem correct_statement : synthetic_method_is_direct ∧ analytic_method_is_direct → 
                             "Synthetic method and analytic method are direct proof methods" = "A" :=
by
  intros h
  cases h
  -- This is where you would provide the proof steps. We skip this with sorry.
  sorry

end NUMINAMATH_GPT_correct_statement_l37_3782


namespace NUMINAMATH_GPT_max_value_of_expression_l37_3733

theorem max_value_of_expression 
  (x y : ℝ) 
  (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  x^2 + y^2 + 2 * x ≤ 15 := sorry

end NUMINAMATH_GPT_max_value_of_expression_l37_3733


namespace NUMINAMATH_GPT_largest_solution_of_equation_l37_3798

theorem largest_solution_of_equation :
  let eq := λ x : ℝ => x^4 - 50 * x^2 + 625
  ∃ x : ℝ, eq x = 0 ∧ ∀ y : ℝ, eq y = 0 → y ≤ x :=
sorry

end NUMINAMATH_GPT_largest_solution_of_equation_l37_3798


namespace NUMINAMATH_GPT_mascot_toy_profit_l37_3712

theorem mascot_toy_profit (x : ℝ) :
  (∀ (c : ℝ) (sales : ℝ), c = 40 → sales = 1000 - 10 * x → (x - c) * sales = 8000) →
  (x = 60 ∨ x = 80) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mascot_toy_profit_l37_3712


namespace NUMINAMATH_GPT_sugar_total_l37_3709

variable (sugar_for_frosting sugar_for_cake : ℝ)

theorem sugar_total (h1 : sugar_for_frosting = 0.6) (h2 : sugar_for_cake = 0.2) :
  sugar_for_frosting + sugar_for_cake = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_sugar_total_l37_3709


namespace NUMINAMATH_GPT_exists_alpha_l37_3720

variable {a : ℕ → ℝ}

axiom nonzero_sequence (n : ℕ) : a n ≠ 0
axiom recurrence_relation (n : ℕ) : a n ^ 2 - a (n - 1) * a (n + 1) = 1

theorem exists_alpha (n : ℕ) : ∃ α : ℝ, ∀ n ≥ 1, a (n + 1) = α * a n - a (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_alpha_l37_3720


namespace NUMINAMATH_GPT_total_growth_of_trees_l37_3731

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end NUMINAMATH_GPT_total_growth_of_trees_l37_3731


namespace NUMINAMATH_GPT_number_of_pairs_l37_3777

theorem number_of_pairs (n : ℕ) (h : n ≥ 3) : 
  ∃ a : ℕ, a = (n-2) * 2^(n-1) + 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pairs_l37_3777


namespace NUMINAMATH_GPT_binom_150_1_eq_150_l37_3768

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end NUMINAMATH_GPT_binom_150_1_eq_150_l37_3768


namespace NUMINAMATH_GPT_hcf_of_three_numbers_l37_3794

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : Nat.lcm a (Nat.lcm b c) = 45600) (h2 : a * b * c = 109183500000) :
  Nat.gcd a (Nat.gcd b c) = 2393750 := by
  sorry

end NUMINAMATH_GPT_hcf_of_three_numbers_l37_3794


namespace NUMINAMATH_GPT_cody_tickets_l37_3797

theorem cody_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (won_tickets : ℕ) : 
  initial_tickets = 49 ∧ spent_tickets = 25 ∧ won_tickets = 6 → 
  initial_tickets - spent_tickets + won_tickets = 30 :=
by sorry

end NUMINAMATH_GPT_cody_tickets_l37_3797


namespace NUMINAMATH_GPT_landscape_avoid_repetition_l37_3741

theorem landscape_avoid_repetition :
  let frames : ℕ := 5
  let days_per_month : ℕ := 30
  (Nat.factorial frames) / days_per_month = 4 := by
  sorry

end NUMINAMATH_GPT_landscape_avoid_repetition_l37_3741


namespace NUMINAMATH_GPT_space_station_cost_share_l37_3737

def total_cost : ℤ := 50 * 10^9
def people_count : ℤ := 500 * 10^6
def per_person_share (C N : ℤ) : ℤ := C / N

theorem space_station_cost_share :
  per_person_share total_cost people_count = 100 :=
by
  sorry

end NUMINAMATH_GPT_space_station_cost_share_l37_3737


namespace NUMINAMATH_GPT_max_pairs_300_grid_l37_3752

noncomputable def max_pairs (n : ℕ) (k : ℕ) (remaining_squares : ℕ) [Fintype (Fin n × Fin n)] : ℕ :=
  sorry

theorem max_pairs_300_grid :
  max_pairs 300 100 50000 = 49998 :=
by
  -- problem conditions
  let grid_size := 300
  let corner_size := 100
  let remaining_squares := 50000
  let no_checkerboard (squares : Fin grid_size × Fin grid_size → Prop) : Prop :=
    ∀ i j, ¬(squares (i, j) ∧ squares (i + 1, j) ∧ squares (i, j + 1) ∧ squares (i + 1, j + 1))
  -- statement of the bound
  have max_pairs := max_pairs grid_size corner_size remaining_squares
  exact sorry

end NUMINAMATH_GPT_max_pairs_300_grid_l37_3752


namespace NUMINAMATH_GPT_radical_axis_of_non_concentric_circles_l37_3791

theorem radical_axis_of_non_concentric_circles 
  {a R1 R2 : ℝ} (a_pos : a ≠ 0) (R1_pos : R1 > 0) (R2_pos : R2 > 0) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ x = (R2^2 - R1^2) / (4 * a) :=
by sorry

end NUMINAMATH_GPT_radical_axis_of_non_concentric_circles_l37_3791


namespace NUMINAMATH_GPT_total_employees_in_buses_l37_3772

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_employees_in_buses_l37_3772


namespace NUMINAMATH_GPT_mary_sugar_cups_l37_3763

theorem mary_sugar_cups (sugar_required : ℕ) (sugar_remaining : ℕ) (sugar_added : ℕ) (h1 : sugar_required = 11) (h2 : sugar_added = 1) : sugar_remaining = 10 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_mary_sugar_cups_l37_3763


namespace NUMINAMATH_GPT_possible_values_of_quadratic_l37_3704

theorem possible_values_of_quadratic (x : ℝ) (hx : x^2 - 7 * x + 12 < 0) :
  1.75 ≤ x^2 - 7 * x + 14 ∧ x^2 - 7 * x + 14 ≤ 2 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_quadratic_l37_3704


namespace NUMINAMATH_GPT_alice_paid_percentage_l37_3716

theorem alice_paid_percentage (SRP P : ℝ) (h1 : P = 0.60 * SRP) (h2 : P_alice = 0.60 * P) :
  (P_alice / SRP) * 100 = 36 := by
sorry

end NUMINAMATH_GPT_alice_paid_percentage_l37_3716


namespace NUMINAMATH_GPT_distance_traveled_by_center_of_ball_l37_3725

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80
noncomputable def R4 : ℝ := 40

noncomputable def effective_radius_inner (R : ℝ) (r : ℝ) : ℝ := R - r
noncomputable def effective_radius_outer (R : ℝ) (r : ℝ) : ℝ := R + r

noncomputable def dist_travel_on_arc (R : ℝ) : ℝ := R * Real.pi

theorem distance_traveled_by_center_of_ball :
  dist_travel_on_arc (effective_radius_inner R1 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R2 ball_radius) +
  dist_travel_on_arc (effective_radius_inner R3 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R4 ball_radius) = 280 * Real.pi :=
by 
  -- Calculation steps can be filled in here but let's skip
  sorry

end NUMINAMATH_GPT_distance_traveled_by_center_of_ball_l37_3725


namespace NUMINAMATH_GPT_runners_meet_l37_3754

theorem runners_meet (T : ℕ) 
  (h1 : T > 4) 
  (h2 : Nat.lcm 2 (Nat.lcm 4 T) = 44) : 
  T = 11 := 
sorry

end NUMINAMATH_GPT_runners_meet_l37_3754


namespace NUMINAMATH_GPT_solution_to_quadratic_solution_to_cubic_l37_3788

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end NUMINAMATH_GPT_solution_to_quadratic_solution_to_cubic_l37_3788


namespace NUMINAMATH_GPT_linear_inequality_solution_l37_3761

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end NUMINAMATH_GPT_linear_inequality_solution_l37_3761


namespace NUMINAMATH_GPT_find_angle_A_find_tan_C_l37_3743

-- Import necessary trigonometric identities and basic Lean setup
open Real

-- First statement: Given the dot product condition, find angle A
theorem find_angle_A (A : ℝ) (h1 : cos A + sqrt 3 * sin A = 1) :
  A = 2 * π / 3 := 
sorry

-- Second statement: Given the trigonometric condition, find tan C
theorem find_tan_C (B C : ℝ)
  (h1 : 1 + sin (2 * B) = 2 * (cos B ^ 2 - sin B ^ 2))
  (h2 : B + C = π) :
  tan C = (5 * sqrt 3 - 6) / 3 := 
sorry

end NUMINAMATH_GPT_find_angle_A_find_tan_C_l37_3743


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l37_3717

open Set

variable {α : Type*}

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient : 
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ b, b ∈ M ∧ b ∉ N) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l37_3717


namespace NUMINAMATH_GPT_part_a_l37_3758

theorem part_a (students : Fin 64 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (A B : Fin 64), (students A).1 ≥ (students B).1 ∧ (students A).2.1 ≥ (students B).2.1 ∧ (students A).2.2 ≥ (students B).2.2 :=
sorry

end NUMINAMATH_GPT_part_a_l37_3758


namespace NUMINAMATH_GPT_Amy_gets_fewest_cookies_l37_3755

theorem Amy_gets_fewest_cookies:
  let area_Amy := 4 * Real.pi
  let area_Ben := 9
  let area_Carl := 8
  let area_Dana := (9 / 2) * Real.pi
  let num_cookies_Amy := 1 / area_Amy
  let num_cookies_Ben := 1 / area_Ben
  let num_cookies_Carl := 1 / area_Carl
  let num_cookies_Dana := 1 / area_Dana
  num_cookies_Amy < num_cookies_Ben ∧ num_cookies_Amy < num_cookies_Carl ∧ num_cookies_Amy < num_cookies_Dana :=
by
  sorry

end NUMINAMATH_GPT_Amy_gets_fewest_cookies_l37_3755


namespace NUMINAMATH_GPT_quadratic_unique_solution_l37_3779

theorem quadratic_unique_solution (k : ℝ) (x : ℝ) :
  (16 ^ 2 - 4 * 2 * k * 4 = 0) → (k = 8 ∧ x = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_unique_solution_l37_3779


namespace NUMINAMATH_GPT_prob_pass_kth_intersection_l37_3705

variable {n k : ℕ}

-- Definitions based on problem conditions
def prob_approach_highway (n : ℕ) : ℚ := 1 / n
def prob_exit_highway (n : ℕ) : ℚ := 1 / n

-- Theorem stating the required probability
theorem prob_pass_kth_intersection (h_n : n > 0) (h_k : k > 0) (h_k_le_n : k ≤ n) :
  (prob_approach_highway n) * (prob_exit_highway n * n) * (2 * k - 1) / n ^ 2 = 
  (2 * k * n - 2 * k ^ 2 + 2 * k - 1) / n ^ 2 := sorry

end NUMINAMATH_GPT_prob_pass_kth_intersection_l37_3705


namespace NUMINAMATH_GPT_product_of_four_integers_l37_3732

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end NUMINAMATH_GPT_product_of_four_integers_l37_3732


namespace NUMINAMATH_GPT_beavers_still_working_l37_3747

theorem beavers_still_working (total_beavers : ℕ) (wood_beavers dam_beavers lodge_beavers : ℕ)
  (wood_swimming dam_swimming lodge_swimming : ℕ) :
  total_beavers = 12 →
  wood_beavers = 5 →
  dam_beavers = 4 →
  lodge_beavers = 3 →
  wood_swimming = 3 →
  dam_swimming = 2 →
  lodge_swimming = 1 →
  (wood_beavers - wood_swimming) + (dam_beavers - dam_swimming) + (lodge_beavers - lodge_swimming) = 6 :=
by
  intros h_total h_wood h_dam h_lodge h_wood_swim h_dam_swim h_lodge_swim
  sorry

end NUMINAMATH_GPT_beavers_still_working_l37_3747


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l37_3795

theorem arithmetic_sequence_value :
  ∀ (a_n : ℕ → ℤ) (d : ℤ),
    (∀ n : ℕ, a_n n = a_n 0 + ↑n * d) →
    a_n 2 = 4 →
    a_n 4 = 8 →
    a_n 10 = 20 :=
by
  intros a_n d h_arith h_a3 h_a5
  --
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l37_3795


namespace NUMINAMATH_GPT_prime_root_range_l37_3711

-- Let's define our conditions first
def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_integer_roots (p : ℕ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧ x + y = p ∧ x * y = -156 * p

-- Now state the theorem
theorem prime_root_range (p : ℕ) (hp : is_prime p) (hr : has_integer_roots p) : 11 < p ∧ p ≤ 21 :=
by
  sorry

end NUMINAMATH_GPT_prime_root_range_l37_3711


namespace NUMINAMATH_GPT_coffee_cost_per_week_l37_3718

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end NUMINAMATH_GPT_coffee_cost_per_week_l37_3718


namespace NUMINAMATH_GPT_inequality_a_b_cubed_l37_3722

theorem inequality_a_b_cubed (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 :=
sorry

end NUMINAMATH_GPT_inequality_a_b_cubed_l37_3722


namespace NUMINAMATH_GPT_find_M_l37_3727

variable (p q r M : ℝ)
variable (h1 : p + q + r = 100)
variable (h2 : p + 10 = M)
variable (h3 : q - 5 = M)
variable (h4 : r / 5 = M)

theorem find_M : M = 15 := by
  sorry

end NUMINAMATH_GPT_find_M_l37_3727


namespace NUMINAMATH_GPT_proof_problem_l37_3708

def sequence : Nat → Rat
| 0 => 2000000
| (n + 1) => sequence n / 2

theorem proof_problem :
  (∀ n, ((sequence n).den = 1) → n < 7) ∧ 
  (sequence 7 = 15625) ∧ 
  (sequence 7 - 3 = 15622) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l37_3708


namespace NUMINAMATH_GPT_Amy_homework_time_l37_3762

def mathProblems : Nat := 18
def spellingProblems : Nat := 6
def problemsPerHour : Nat := 4
def totalProblems : Nat := mathProblems + spellingProblems
def totalHours : Nat := totalProblems / problemsPerHour

theorem Amy_homework_time :
  totalHours = 6 := by
  sorry

end NUMINAMATH_GPT_Amy_homework_time_l37_3762


namespace NUMINAMATH_GPT_problem_statement_l37_3715

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : xy = -2) : (1 - x) * (1 - y) = -3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l37_3715


namespace NUMINAMATH_GPT_polygon_sides_l37_3783

theorem polygon_sides (n : ℕ) : (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_l37_3783


namespace NUMINAMATH_GPT_point_on_circle_l37_3748

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle_radius := 5

def A : Point := {x := 2, y := -3}
def M : Point := {x := 5, y := -7}

theorem point_on_circle :
  distance A.x A.y M.x M.y = circle_radius :=
by
  sorry

end NUMINAMATH_GPT_point_on_circle_l37_3748


namespace NUMINAMATH_GPT_second_pipe_fill_time_l37_3785

theorem second_pipe_fill_time
  (rate1: ℝ) (rate_outlet: ℝ) (combined_time: ℝ)
  (h1: rate1 = 1 / 18)
  (h2: rate_outlet = 1 / 45)
  (h_combined: combined_time = 0.05):
  ∃ (x: ℝ), (1 / x) = 60 :=
by
  sorry

end NUMINAMATH_GPT_second_pipe_fill_time_l37_3785


namespace NUMINAMATH_GPT_tangent_line_to_curve_at_P_l37_3766

noncomputable def tangent_line_at_point (x y : ℝ) := 4 * x - y - 2 = 0

theorem tangent_line_to_curve_at_P :
  (∃ (b: ℝ), ∀ (x: ℝ), b = 2 * 1^2 → tangent_line_at_point 1 2)
:= 
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_curve_at_P_l37_3766


namespace NUMINAMATH_GPT_calculate_10_odot_5_l37_3729

def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem calculate_10_odot_5 : odot 10 5 = 38 / 3 := by
  sorry

end NUMINAMATH_GPT_calculate_10_odot_5_l37_3729


namespace NUMINAMATH_GPT_scientific_notation_of_distance_l37_3745

theorem scientific_notation_of_distance :
  ∃ (n : ℝ), n = 384000 ∧ 384000 = n * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_distance_l37_3745


namespace NUMINAMATH_GPT_students_in_classroom_l37_3730

theorem students_in_classroom (n : ℕ) :
  n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 → n = 21 ∨ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_students_in_classroom_l37_3730


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_thirteen_l37_3749

theorem greatest_three_digit_multiple_of_thirteen : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (13 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (13 ∣ m) → m ≤ n) ∧ n = 988 :=
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_thirteen_l37_3749


namespace NUMINAMATH_GPT_perfect_square_trinomial_l37_3769

theorem perfect_square_trinomial (m : ℝ) (h : ∃ a : ℝ, x^2 + 2 * x + m = (x + a)^2) : m = 1 := 
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l37_3769


namespace NUMINAMATH_GPT_sequence_solution_l37_3707

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 1/2)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 1 / (n^2 + n)) : ∀ n : ℕ, n ≥ 1 → a n = 3/2 - 1/n :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_sequence_solution_l37_3707


namespace NUMINAMATH_GPT_fruits_in_box_l37_3776

theorem fruits_in_box (initial_persimmons : ℕ) (added_apples : ℕ) (total_fruits : ℕ) :
  initial_persimmons = 2 → added_apples = 7 → total_fruits = initial_persimmons + added_apples → total_fruits = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_fruits_in_box_l37_3776


namespace NUMINAMATH_GPT_general_formula_sequence_sum_first_n_terms_l37_3756

-- Define the axioms or conditions of the arithmetic sequence
axiom a3_eq_7 : ∃ a1 d : ℝ, a1 + 2 * d = 7
axiom a5_plus_a7_eq_26 : ∃ a1 d : ℝ, (a1 + 4 * d) + (a1 + 6 * d) = 26

-- State the theorem for the general formula of the arithmetic sequence
theorem general_formula_sequence (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, a1 + (n - 1) * d = 2 * n + 1 :=
sorry

-- State the theorem for the sum of the first n terms of the arithmetic sequence
theorem sum_first_n_terms (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, n * (a1 + (n - 1) * d + a1) / 2 = (n^2 + 2 * n) :=
sorry

end NUMINAMATH_GPT_general_formula_sequence_sum_first_n_terms_l37_3756


namespace NUMINAMATH_GPT_part_I_part_II_l37_3789

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l37_3789


namespace NUMINAMATH_GPT_total_volume_l37_3721

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem total_volume {d_cylinder d_cone_top d_cone_bottom h_cylinder h_cone : ℝ}
  (h1 : d_cylinder = 2) (h2 : d_cone_top = 2) (h3 : d_cone_bottom = 1)
  (h4 : h_cylinder = 14) (h5 : h_cone = 4) :
  volume_cylinder (d_cylinder / 2) h_cylinder +
  volume_cone (d_cone_top / 2) h_cone =
  (46 / 3) * π :=
by
  sorry

end NUMINAMATH_GPT_total_volume_l37_3721


namespace NUMINAMATH_GPT_opposite_signs_abs_larger_l37_3724

theorem opposite_signs_abs_larger (a b : ℝ) (h1 : a + b < 0) (h2 : a * b < 0) :
  (a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |b| > |a|) :=
sorry

end NUMINAMATH_GPT_opposite_signs_abs_larger_l37_3724


namespace NUMINAMATH_GPT_value_of_expression_l37_3742

theorem value_of_expression (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 3) :
  (2 * a - (3 * b - 4 * c)) - ((2 * a - 3 * b) - 4 * c) = 24 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l37_3742


namespace NUMINAMATH_GPT_number_of_yellow_parrots_l37_3753

-- Given conditions
def fraction_red : ℚ := 5 / 8
def total_parrots : ℕ := 120

-- Proof statement
theorem number_of_yellow_parrots : 
    (total_parrots : ℚ) * (1 - fraction_red) = 45 :=
by 
    sorry

end NUMINAMATH_GPT_number_of_yellow_parrots_l37_3753


namespace NUMINAMATH_GPT_percentage_books_not_sold_l37_3706

theorem percentage_books_not_sold :
    let initial_stock := 700
    let books_sold_mon := 50
    let books_sold_tue := 82
    let books_sold_wed := 60
    let books_sold_thu := 48
    let books_sold_fri := 40
    let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri 
    let books_not_sold := initial_stock - total_books_sold
    let percentage_not_sold := (books_not_sold * 100) / initial_stock
    percentage_not_sold = 60 :=
by
  -- definitions
  let initial_stock := 700
  let books_sold_mon := 50
  let books_sold_tue := 82
  let books_sold_wed := 60
  let books_sold_thu := 48
  let books_sold_fri := 40
  let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold * 100) / initial_stock
  have : percentage_not_sold = 60 := sorry
  exact this

end NUMINAMATH_GPT_percentage_books_not_sold_l37_3706


namespace NUMINAMATH_GPT_platform_length_l37_3793

/-- Mathematical proof problem:
The problem is to prove that given the train's length, time taken to cross a signal pole and 
time taken to cross a platform, the length of the platform is 525 meters.
-/
theorem platform_length 
    (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) (P : ℕ) 
    (h_train_length : train_length = 450) (h_time_pole : time_pole = 18) 
    (h_time_platform : time_platform = 39) (h_P : P = 525) : 
    P = 525 := 
  sorry

end NUMINAMATH_GPT_platform_length_l37_3793


namespace NUMINAMATH_GPT_probability_of_X_eq_4_l37_3760

noncomputable def probability_X_eq_4 : ℝ :=
  let total_balls := 12
  let new_balls := 9
  let old_balls := 3
  let draw := 3
  -- Number of ways to choose 2 old balls from 3
  let choose_old := Nat.choose old_balls 2
  -- Number of ways to choose 1 new ball from 9
  let choose_new := Nat.choose new_balls 1
  -- Total number of ways to choose 3 balls from 12
  let total_ways := Nat.choose total_balls draw
  -- Probability calculation
  (choose_old * choose_new) / total_ways

theorem probability_of_X_eq_4 : probability_X_eq_4 = 27 / 220 := by
  sorry

end NUMINAMATH_GPT_probability_of_X_eq_4_l37_3760


namespace NUMINAMATH_GPT_oranges_ratio_l37_3775

theorem oranges_ratio (initial_oranges_kgs : ℕ) (additional_oranges_kgs : ℕ) (total_oranges_three_weeks : ℕ) :
  initial_oranges_kgs = 10 →
  additional_oranges_kgs = 5 →
  total_oranges_three_weeks = 75 →
  (2 * (total_oranges_three_weeks - (initial_oranges_kgs + additional_oranges_kgs)) / 2) / (initial_oranges_kgs + additional_oranges_kgs) = 2 :=
by
  intros h_initial h_additional h_total
  sorry

end NUMINAMATH_GPT_oranges_ratio_l37_3775


namespace NUMINAMATH_GPT_find_x_l37_3728

theorem find_x (x : ℤ) (h : x + -27 = 30) : x = 57 :=
sorry

end NUMINAMATH_GPT_find_x_l37_3728


namespace NUMINAMATH_GPT_find_remainder_l37_3734

-- Main statement with necessary definitions and conditions
theorem find_remainder (x : ℤ) (h : (x + 11) % 31 = 18) :
  x % 62 = 7 :=
sorry

end NUMINAMATH_GPT_find_remainder_l37_3734


namespace NUMINAMATH_GPT_inequality_l37_3700

def domain (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality (a b : ℝ) (ha : domain a) (hb : domain b) :
  |a + b| < |3 + ab / 3| :=
by
  sorry

end NUMINAMATH_GPT_inequality_l37_3700


namespace NUMINAMATH_GPT_third_smallest_four_digit_in_pascals_triangle_l37_3740

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end NUMINAMATH_GPT_third_smallest_four_digit_in_pascals_triangle_l37_3740


namespace NUMINAMATH_GPT_tickets_total_l37_3773

theorem tickets_total (T : ℝ) (h1 : T / 2 + (T / 2) / 4 = 3600) : T = 5760 :=
by
  sorry

end NUMINAMATH_GPT_tickets_total_l37_3773


namespace NUMINAMATH_GPT_polynomial_value_l37_3765

noncomputable def polynomial_spec (p : ℝ) : Prop :=
  p^3 - 5 * p + 1 = 0

theorem polynomial_value (p : ℝ) (h : polynomial_spec p) : 
  p^4 - 3 * p^3 - 5 * p^2 + 16 * p + 2015 = 2018 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l37_3765


namespace NUMINAMATH_GPT_eval_expression_l37_3723

theorem eval_expression : (2 ^ (-1 : ℤ)) + (Real.sin (Real.pi / 6)) - (Real.pi - 3.14) ^ (0 : ℤ) + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l37_3723


namespace NUMINAMATH_GPT_rectangle_area_l37_3784

-- Define the given dimensions
def length : ℝ := 1.5
def width : ℝ := 0.75
def expected_area : ℝ := 1.125

-- State the problem
theorem rectangle_area (l w : ℝ) (h_l : l = length) (h_w : w = width) : l * w = expected_area :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l37_3784


namespace NUMINAMATH_GPT_total_cars_l37_3770

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end NUMINAMATH_GPT_total_cars_l37_3770


namespace NUMINAMATH_GPT_problem_statement_l37_3767

-- Given the conditions and the goal
theorem problem_statement (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz_sum : x + y + z = 1) :
  (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l37_3767


namespace NUMINAMATH_GPT_range_of_x_minus_cos_y_l37_3751

theorem range_of_x_minus_cos_y
  (x y : ℝ)
  (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (A : Set ℝ), A = {z | -1 ≤ z ∧ z ≤ 1 + Real.sqrt 3} ∧ x - Real.cos y ∈ A :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_minus_cos_y_l37_3751


namespace NUMINAMATH_GPT_geometric_series_sum_l37_3790

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l37_3790


namespace NUMINAMATH_GPT_max_students_can_distribute_equally_l37_3757

-- Define the given numbers of pens and pencils
def pens : ℕ := 1001
def pencils : ℕ := 910

-- State the problem in Lean 4 as a theorem
theorem max_students_can_distribute_equally :
  Nat.gcd pens pencils = 91 :=
sorry

end NUMINAMATH_GPT_max_students_can_distribute_equally_l37_3757


namespace NUMINAMATH_GPT_balls_into_boxes_l37_3739

-- Define the conditions
def balls : ℕ := 7
def boxes : ℕ := 4

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the equivalent proof problem
theorem balls_into_boxes :
    (binom (balls - 1) (boxes - 1) = 20) ∧ (binom (balls + (boxes - 1)) (boxes - 1) = 120) := by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l37_3739


namespace NUMINAMATH_GPT_fred_limes_l37_3787

theorem fred_limes (limes_total : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) (fred_limes : ℕ)
  (h_total : limes_total = 103)
  (h_alyssa : alyssa_limes = 32)
  (h_nancy : nancy_limes = 35)
  (h_fred : fred_limes = limes_total - (alyssa_limes + nancy_limes)) :
  fred_limes = 36 :=
by
  sorry

end NUMINAMATH_GPT_fred_limes_l37_3787


namespace NUMINAMATH_GPT_max_value_of_a_l37_3764

theorem max_value_of_a
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

example 
  (a b c : ℝ)
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 12) : 
  (7 - Real.sqrt 46) / 3 ≤ a :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l37_3764


namespace NUMINAMATH_GPT_distinct_cubed_mod_7_units_digits_l37_3701

theorem distinct_cubed_mod_7_units_digits : 
  (∃ S : Finset ℕ, S.card = 3 ∧ ∀ n ∈ (Finset.range 7), (n^3 % 7) ∈ S) :=
  sorry

end NUMINAMATH_GPT_distinct_cubed_mod_7_units_digits_l37_3701


namespace NUMINAMATH_GPT_uniformity_comparison_l37_3771

theorem uniformity_comparison (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : S1^2 < S2^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_uniformity_comparison_l37_3771


namespace NUMINAMATH_GPT_positive_difference_l37_3744

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end NUMINAMATH_GPT_positive_difference_l37_3744


namespace NUMINAMATH_GPT_election_votes_l37_3726

theorem election_votes (V : ℝ) 
  (h1 : 0.15 * V = 0.15 * V)
  (h2 : 0.85 * V = 309400 / 0.65)
  (h3 : 0.65 * (0.85 * V) = 309400) : 
  V = 560000 :=
by {
  sorry
}

end NUMINAMATH_GPT_election_votes_l37_3726


namespace NUMINAMATH_GPT_sum_of_even_factors_900_l37_3702

theorem sum_of_even_factors_900 : 
  ∃ (S : ℕ), 
  (∀ a b c : ℕ, 900 = 2^a * 3^b * 5^c → 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 2) → 
  (∀ a : ℕ, 1 ≤ a ∧ a ≤ 2 → ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ (2^a * 3^b * 5^c = 900 ∧ a ≠ 0)) → 
  S = 2418 := 
sorry

end NUMINAMATH_GPT_sum_of_even_factors_900_l37_3702


namespace NUMINAMATH_GPT_mean_of_set_l37_3781

theorem mean_of_set (x y : ℝ) 
  (h : (28 + x + 50 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 :=
by
  -- we would now proceed to prove this according to lean's proof tactics.
  sorry

end NUMINAMATH_GPT_mean_of_set_l37_3781


namespace NUMINAMATH_GPT_sum_mod_9237_9241_l37_3750

theorem sum_mod_9237_9241 :
  (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_9237_9241_l37_3750


namespace NUMINAMATH_GPT_greatest_product_sum_300_l37_3792

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end NUMINAMATH_GPT_greatest_product_sum_300_l37_3792


namespace NUMINAMATH_GPT_warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l37_3780

noncomputable def netChange (tonnages : List Int) : Int :=
  List.sum tonnages

noncomputable def initialGoods (finalGoods : Int) (change : Int) : Int :=
  finalGoods + change

noncomputable def totalFees (tonnages : List Int) (feePerTon : Int) : Int :=
  feePerTon * List.sum (tonnages.map (Int.natAbs))

theorem warehouseGoodsDecreased 
  (tonnages : List Int) (finalGoods : Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20]) 
  (h2 : finalGoods = 580)
  (h3 : feePerTon = 4) : 
  netChange tonnages < 0 := by
  sorry

theorem initialTonnage 
  (tonnages : List Int) (finalGoods : Int) (change : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : finalGoods = 580)
  (h3 : change = netChange tonnages) : 
  initialGoods finalGoods change = 630 := by
  sorry

theorem totalLoadingFees 
  (tonnages : List Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : feePerTon = 4) : 
  totalFees tonnages feePerTon = 648 := by
  sorry

end NUMINAMATH_GPT_warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l37_3780


namespace NUMINAMATH_GPT_four_people_complete_task_in_18_days_l37_3714

theorem four_people_complete_task_in_18_days :
  (forall r : ℝ, (3 * 24 * r = 1) → (4 * 18 * r = 1)) :=
by
  intro r
  intro h
  sorry

end NUMINAMATH_GPT_four_people_complete_task_in_18_days_l37_3714


namespace NUMINAMATH_GPT_students_band_and_chorus_l37_3710

theorem students_band_and_chorus (Total Band Chorus Union Intersection : ℕ) 
  (h₁ : Total = 300) 
  (h₂ : Band = 110) 
  (h₃ : Chorus = 140) 
  (h₄ : Union = 220) :
  Intersection = Band + Chorus - Union :=
by
  -- Given the conditions, the proof would follow here.
  sorry

end NUMINAMATH_GPT_students_band_and_chorus_l37_3710


namespace NUMINAMATH_GPT_parabola_shift_l37_3735

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end NUMINAMATH_GPT_parabola_shift_l37_3735


namespace NUMINAMATH_GPT_race_time_l37_3799

theorem race_time (t_A t_B : ℝ) (v_A v_B : ℝ)
  (h1 : t_B = t_A + 7)
  (h2 : v_A * t_A = 80)
  (h3 : v_B * t_B = 80)
  (h4 : v_A * (t_A + 7) = 136) :
  t_A = 10 :=
by
  sorry

end NUMINAMATH_GPT_race_time_l37_3799


namespace NUMINAMATH_GPT_calc_fraction_cube_l37_3738

theorem calc_fraction_cube : (88888 ^ 3 / 22222 ^ 3) = 64 := by 
    sorry

end NUMINAMATH_GPT_calc_fraction_cube_l37_3738


namespace NUMINAMATH_GPT_max_members_choir_l37_3746

variable (m k n : ℕ)

theorem max_members_choir :
  (∃ k, m = k^2 + 6) ∧ (∃ n, m = n * (n + 6)) → m = 294 :=
by
  sorry

end NUMINAMATH_GPT_max_members_choir_l37_3746


namespace NUMINAMATH_GPT_bridge_length_l37_3786

-- Definitions based on conditions
def Lt : ℕ := 148
def Skm : ℕ := 45
def T : ℕ := 30

-- Conversion from km/h to m/s
def conversion_factor : ℕ := 1000 / 3600
def Sm : ℝ := Skm * conversion_factor

-- Calculation of distance traveled in 30 seconds
def distance : ℝ := Sm * T

-- The length of the bridge
def L_bridge : ℝ := distance - Lt

theorem bridge_length : L_bridge = 227 := sorry

end NUMINAMATH_GPT_bridge_length_l37_3786


namespace NUMINAMATH_GPT_unit_digit_of_15_pow_100_l37_3703

-- Define a function to extract the unit digit of a number
def unit_digit (n : ℕ) : ℕ := n % 10

-- Given conditions:
def base : ℕ := 15
def exponent : ℕ := 100

-- Define what 'unit_digit' of a number raised to an exponent means
def unit_digit_pow (base exponent : ℕ) : ℕ :=
  unit_digit (base ^ exponent)

-- Goal: Prove that the unit digit of 15^100 is 5.
theorem unit_digit_of_15_pow_100 : unit_digit_pow base exponent = 5 :=
by
  sorry

end NUMINAMATH_GPT_unit_digit_of_15_pow_100_l37_3703


namespace NUMINAMATH_GPT_exradii_product_eq_area_squared_l37_3736

variable (a b c : ℝ) (t : ℝ)
variable (s := (a + b + c) / 2)
variable (exradius_a exradius_b exradius_c : ℝ)

-- Define the conditions
axiom Heron : t^2 = s * (s - a) * (s - b) * (s - c)
axiom exradius_definitions : exradius_a = t / (s - a) ∧ exradius_b = t / (s - b) ∧ exradius_c = t / (s - c)

-- The theorem we want to prove
theorem exradii_product_eq_area_squared : exradius_a * exradius_b * exradius_c = t^2 := sorry

end NUMINAMATH_GPT_exradii_product_eq_area_squared_l37_3736


namespace NUMINAMATH_GPT_simplify_expression_l37_3774

theorem simplify_expression :
  ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l37_3774
