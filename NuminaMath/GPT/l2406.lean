import Mathlib

namespace geometric_sequence_first_term_l2406_240622

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 18) 
  (h2 : a * r^4 = 1458) : 
  a = 6 := 
by 
  sorry

end geometric_sequence_first_term_l2406_240622


namespace problem_statement_l2406_240617

theorem problem_statement
  (a b : ℝ)
  (ha : a = Real.sqrt 2 + 1)
  (hb : b = Real.sqrt 2 - 1) :
  a^2 - a * b + b^2 = 5 :=
sorry

end problem_statement_l2406_240617


namespace divide_money_equally_l2406_240648

-- Length of the road built by companies A, B, and total length of the road
def length_A : ℕ := 6
def length_B : ℕ := 10
def total_length : ℕ := 16

-- Money contributed by company C
def money_C : ℕ := 16 * 10^6

-- The equal contribution each company should finance
def equal_contribution := total_length / 3

-- Deviations from the expected length for firms A and B
def deviation_A := length_A - (total_length / 3)
def deviation_B := length_B - (total_length / 3)

-- The ratio based on the deviations to divide the money
def ratio_A := deviation_A * (total_length / (deviation_A + deviation_B))
def ratio_B := deviation_B * (total_length / (deviation_A + deviation_B))

-- The amount of money firms A and B should receive, respectively
def money_A := money_C * ratio_A / total_length
def money_B := money_C * ratio_B / total_length

-- Theorem statement
theorem divide_money_equally : money_A = 2 * 10^6 ∧ money_B = 14 * 10^6 :=
by 
  sorry

end divide_money_equally_l2406_240648


namespace apple_tree_distribution_l2406_240647

-- Definition of the problem
noncomputable def paths := 4

-- Definition of the apple tree positions
structure Position where
  x : ℕ -- Coordinate x
  y : ℕ -- Coordinate y

-- Definition of the initial condition: one existing apple tree
def existing_apple_tree : Position := {x := 0, y := 0}

-- Problem: proving the existence of a configuration with three new apple trees
theorem apple_tree_distribution :
  ∃ (p1 p2 p3 : Position),
    (p1 ≠ existing_apple_tree) ∧ (p2 ≠ existing_apple_tree) ∧ (p3 ≠ existing_apple_tree) ∧
    -- Ensure each path has equal number of trees on both sides
    (∃ (path1 path2 : ℕ), 
      -- Horizontal path balance
      path1 = (if p1.x > 0 then 1 else 0) + (if p2.x > 0 then 1 else 0) + (if p3.x > 0 then 1 else 0) + 1 ∧
      path2 = (if p1.x < 0 then 1 else 0) + (if p2.x < 0 then 1 else 0) + (if p3.x < 0 then 1 else 0) ∧
      path1 = path2) ∧
    (∃ (path3 path4 : ℕ), 
      -- Vertical path balance
      path3 = (if p1.y > 0 then 1 else 0) + (if p2.y > 0 then 1 else 0) + (if p3.y > 0 then 1 else 0) + 1 ∧
      path4 = (if p1.y < 0 then 1 else 0) + (if p2.y < 0 then 1 else 0) + (if p3.y < 0 then 1 else 0) ∧
      path3 = path4)
  := by sorry

end apple_tree_distribution_l2406_240647


namespace solution_set_l2406_240601

def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

theorem solution_set (x : ℝ) :
  (f x 2) ≥ 1 ↔ x ≥ 2 :=
sorry

end solution_set_l2406_240601


namespace initial_rate_of_commission_is_4_l2406_240665

noncomputable def initial_commission_rate (B : ℝ) (x : ℝ) : Prop :=
  B * (x / 100) = 0.8 * B * (5 / 100)

theorem initial_rate_of_commission_is_4 (B : ℝ) (hB : B > 0) :
  initial_commission_rate B 4 :=
by
  unfold initial_commission_rate
  sorry

end initial_rate_of_commission_is_4_l2406_240665


namespace even_function_a_value_l2406_240677

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (a * (-x)^2 + (2 * a + 1) * (-x) - 1) = (a * x^2 + (2 * a + 1) * x - 1)) →
  a = - 1 / 2 :=
by sorry

end even_function_a_value_l2406_240677


namespace range_of_m_l2406_240678

theorem range_of_m (m : ℝ) :
  let p := (2 < m ∧ m < 4)
  let q := (m > 1 ∧ 4 - 4 * m < 0)
  (¬ (p ∧ q) ∧ (p ∨ q)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 4) :=
by intros p q h
   let p := 2 < m ∧ m < 4
   let q := m > 1 ∧ 4 - 4 * m < 0
   sorry

end range_of_m_l2406_240678


namespace girls_dropped_out_l2406_240687

theorem girls_dropped_out (B_initial G_initial B_dropped G_remaining S_remaining : ℕ)
  (hB_initial : B_initial = 14)
  (hG_initial : G_initial = 10)
  (hB_dropped : B_dropped = 4)
  (hS_remaining : S_remaining = 17)
  (hB_remaining : B_initial - B_dropped = B_remaining)
  (hG_remaining : G_remaining = S_remaining - B_remaining) :
  (G_initial - G_remaining) = 3 := 
by 
  sorry

end girls_dropped_out_l2406_240687


namespace solve_equation_l2406_240671

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 2) = (x - 2) → (x = 2 ∨ x = 1 / 3) :=
by
  intro x
  intro h
  sorry

end solve_equation_l2406_240671


namespace mass_percentages_correct_l2406_240610

noncomputable def mass_percentage_of_Ba (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * 137.327 + (y / 153.326) * 137.327) / (x + y) ) * 100

noncomputable def mass_percentage_of_F (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * (2 * 18.998)) / (x + y) ) * 100

noncomputable def mass_percentage_of_O (x y : ℝ) : ℝ :=
  ( ((y / 153.326) * 15.999) / (x + y) ) * 100

theorem mass_percentages_correct (x y : ℝ) :
  ∃ (Ba F O : ℝ), 
    Ba = mass_percentage_of_Ba x y ∧
    F = mass_percentage_of_F x y ∧
    O = mass_percentage_of_O x y :=
sorry

end mass_percentages_correct_l2406_240610


namespace value_of_nabla_expression_l2406_240652

namespace MathProblem

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem value_of_nabla_expression : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end MathProblem

end value_of_nabla_expression_l2406_240652


namespace distinct_sequences_six_sided_die_rolled_six_times_l2406_240658

theorem distinct_sequences_six_sided_die_rolled_six_times :
  let count := 6
  (count ^ 6 = 46656) :=
by
  let count := 6
  sorry

end distinct_sequences_six_sided_die_rolled_six_times_l2406_240658


namespace factor_27x6_minus_512y6_sum_coeffs_is_152_l2406_240626

variable {x y : ℤ}

theorem factor_27x6_minus_512y6_sum_coeffs_is_152 :
  ∃ a b c d e f g h j k : ℤ, 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) ∧ 
    (a + b + c + d + e + f + g + h + j + k = 152) := 
sorry

end factor_27x6_minus_512y6_sum_coeffs_is_152_l2406_240626


namespace simple_interest_years_l2406_240655

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end simple_interest_years_l2406_240655


namespace sofia_total_time_l2406_240698

-- Definitions for the conditions
def laps : ℕ := 5
def track_length : ℕ := 400  -- in meters
def speed_first_100 : ℕ := 4  -- meters per second
def speed_remaining_300 : ℕ := 5  -- meters per second

-- Times taken for respective distances
def time_first_100 (distance speed : ℕ) : ℕ := distance / speed
def time_remaining_300 (distance speed : ℕ) : ℕ := distance / speed

def time_one_lap : ℕ := time_first_100 100 speed_first_100 + time_remaining_300 300 speed_remaining_300
def total_time_seconds : ℕ := laps * time_one_lap
def total_time_minutes : ℕ := 7
def total_time_extra_seconds : ℕ := 5

-- Problem statement
theorem sofia_total_time :
  total_time_seconds = total_time_minutes * 60 + total_time_extra_seconds :=
by
  sorry

end sofia_total_time_l2406_240698


namespace reduce_to_original_l2406_240624

theorem reduce_to_original (x : ℝ) (factor : ℝ) (original : ℝ) :
  original = x → factor = 1/1000 → x * factor = 0.0169 :=
by
  intros h1 h2
  sorry

end reduce_to_original_l2406_240624


namespace vacant_seats_l2406_240668

open Nat

-- Define the conditions as Lean definitions
def num_tables : Nat := 5
def seats_per_table : Nat := 8
def occupied_tables : Nat := 2
def people_per_occupied_table : Nat := 3
def unusable_tables : Nat := 1

-- Calculate usable tables
def usable_tables : Nat := num_tables - unusable_tables

-- Calculate total occupied people
def total_occupied_people : Nat := occupied_tables * people_per_occupied_table

-- Calculate total seats for occupied tables
def total_seats_occupied_tables : Nat := occupied_tables * seats_per_table

-- Calculate vacant seats in occupied tables
def vacant_seats_occupied_tables : Nat := total_seats_occupied_tables - total_occupied_people

-- Calculate completely unoccupied tables
def unoccupied_tables : Nat := usable_tables - occupied_tables

-- Calculate total seats for unoccupied tables
def total_seats_unoccupied_tables : Nat := unoccupied_tables * seats_per_table

-- Calculate total vacant seats
def total_vacant_seats : Nat := vacant_seats_occupied_tables + total_seats_unoccupied_tables

-- Theorem statement to prove
theorem vacant_seats : total_vacant_seats = 26 := by
  sorry

end vacant_seats_l2406_240668


namespace cream_cheese_cost_l2406_240694

theorem cream_cheese_cost:
  ∃ (B C : ℝ), (2 * B + 3 * C = 12) ∧ (4 * B + 2 * C = 14) ∧ (C = 2.5) :=
by
  sorry

end cream_cheese_cost_l2406_240694


namespace eq_of_fraction_eq_l2406_240653

variable {R : Type*} [Field R]

theorem eq_of_fraction_eq (a b : R) (h : (1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b))) : a = b :=
sorry

end eq_of_fraction_eq_l2406_240653


namespace solve_quadratic_l2406_240642

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 + 5 * x = 0 ↔ x = 0 ∨ x = -5/2 :=
by
  intro x
  sorry

end solve_quadratic_l2406_240642


namespace striped_octopus_has_eight_legs_l2406_240685

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l2406_240685


namespace greatest_integer_difference_l2406_240674

theorem greatest_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  ∃ d : ℤ, d = y - x ∧ ∀ z, 4 < z ∧ z < 8 ∧ 8 < y ∧ y < 12 → (y - z ≤ d) :=
sorry

end greatest_integer_difference_l2406_240674


namespace anticipated_margin_l2406_240619

noncomputable def anticipated_profit_margin (original_purchase_price : ℝ) (decrease_percentage : ℝ) (profit_margin_increase : ℝ) (selling_price : ℝ) : ℝ :=
original_purchase_price * (1 + profit_margin_increase / 100)

theorem anticipated_margin (x : ℝ) (original_purchase_price_decrease : ℝ := 0.064) (profit_margin_increase : ℝ := 8) (selling_price : ℝ) :
  selling_price = original_purchase_price * (1 + x / 100) ∧ selling_price = (1 - original_purchase_price_decrease) * (1 + (x + profit_margin_increase) / 100) →
  true :=
by
  sorry

end anticipated_margin_l2406_240619


namespace total_crayons_l2406_240640
-- Import the whole Mathlib to ensure all necessary components are available

-- Definitions of the number of crayons each person has
def Billy_crayons : ℕ := 62
def Jane_crayons : ℕ := 52
def Mike_crayons : ℕ := 78
def Sue_crayons : ℕ := 97

-- Theorem stating the total number of crayons is 289
theorem total_crayons : (Billy_crayons + Jane_crayons + Mike_crayons + Sue_crayons) = 289 := by
  sorry

end total_crayons_l2406_240640


namespace number_of_uncool_parents_l2406_240621

variable (total_students cool_dads cool_moms cool_both : ℕ)

theorem number_of_uncool_parents (h1 : total_students = 40)
                                  (h2 : cool_dads = 18)
                                  (h3 : cool_moms = 22)
                                  (h4 : cool_both = 10) :
    total_students - (cool_dads + cool_moms - cool_both) = 10 := by
  sorry

end number_of_uncool_parents_l2406_240621


namespace inequality_proof_equality_condition_l2406_240682

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b)) → a = b :=
sorry

end inequality_proof_equality_condition_l2406_240682


namespace remainder_7_pow_2010_l2406_240611

theorem remainder_7_pow_2010 :
  (7 ^ 2010) % 100 = 49 := 
by 
  sorry

end remainder_7_pow_2010_l2406_240611


namespace music_stand_cost_proof_l2406_240637

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l2406_240637


namespace proof_problem_l2406_240695

structure Plane := (name : String)
structure Line := (name : String)

def parallel_planes (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def parallel_lines (m n : Line) : Prop := sorry

theorem proof_problem (m : Line) (α β : Plane) :
  parallel_planes α β → in_plane m α → parallel_lines m (Line.mk β.name) :=
sorry

end proof_problem_l2406_240695


namespace fault_line_total_movement_l2406_240670

theorem fault_line_total_movement (a b : ℝ) (h1 : a = 1.25) (h2 : b = 5.25) : a + b = 6.50 := by
  -- Definitions:
  rw [h1, h2]
  -- Proof:
  sorry

end fault_line_total_movement_l2406_240670


namespace total_right_handed_players_is_60_l2406_240667

def total_players : ℕ := 70
def throwers : ℕ := 40
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed_players : ℕ := right_handed_throwers + right_handed_non_throwers

theorem total_right_handed_players_is_60 : total_right_handed_players = 60 := by
  sorry

end total_right_handed_players_is_60_l2406_240667


namespace solve_x_in_equation_l2406_240636

theorem solve_x_in_equation : ∃ (x : ℤ), 24 - 4 * 2 = 3 + x ∧ x = 13 :=
by
  use 13
  sorry

end solve_x_in_equation_l2406_240636


namespace union_of_P_and_Q_l2406_240693

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem union_of_P_and_Q : (P ∪ Q) = {x | -1 < x ∧ x < 3} := by
  -- skipping the proof
  sorry

end union_of_P_and_Q_l2406_240693


namespace defense_attorney_mistake_l2406_240659

variable (P Q : Prop)

theorem defense_attorney_mistake (h1 : P → Q) (h2 : ¬ (P → Q)) : P ∧ ¬ Q :=
by {
  sorry
}

end defense_attorney_mistake_l2406_240659


namespace area_of_roof_l2406_240600

-- Definitions and conditions
def length (w : ℝ) := 4 * w
def difference_eq (l w : ℝ) := l - w = 39
def area (l w : ℝ) := l * w

-- Theorem statement
theorem area_of_roof (w l : ℝ) (h_length : l = length w) (h_diff : difference_eq l w) : area l w = 676 :=
by
  sorry

end area_of_roof_l2406_240600


namespace log_a_less_than_neg_b_minus_one_l2406_240603

variable {x : ℝ} (a b : ℝ) (f : ℝ → ℝ)

theorem log_a_less_than_neg_b_minus_one
  (h1 : 0 < a)
  (h2 : ∀ x > 0, f x ≥ f 3)
  (h3 : ∀ x > 0, f x = -3 * Real.log x + a * x^2 + b * x) :
  Real.log a < -b - 1 :=
  sorry

end log_a_less_than_neg_b_minus_one_l2406_240603


namespace interest_rate_is_six_percent_l2406_240630

noncomputable def amount : ℝ := 1120
noncomputable def principal : ℝ := 979.0209790209791
noncomputable def time_years : ℝ := 2 + 2 / 5

noncomputable def total_interest (A P: ℝ) : ℝ := A - P

noncomputable def interest_rate_per_annum (I P T: ℝ) : ℝ := I / (P * T) * 100

theorem interest_rate_is_six_percent :
  interest_rate_per_annum (total_interest amount principal) principal time_years = 6 := 
by
  sorry

end interest_rate_is_six_percent_l2406_240630


namespace range_m_plus_2n_l2406_240613

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def m_value (t : ℝ) : ℝ := 1 / t + 1 / (t ^ 2)

noncomputable def n_value (t : ℝ) : ℝ := Real.log t - 2 / t - 1

noncomputable def g (x : ℝ) : ℝ := (1 / (x ^ 2)) + 2 * Real.log x - (3 / x) - 2

theorem range_m_plus_2n :
  ∀ m n : ℝ, (∃ t > 0, m = m_value t ∧ n = n_value t) →
  (m + 2 * n) ∈ Set.Ici (-2 * Real.log 2 - 4) := by
  sorry

end range_m_plus_2n_l2406_240613


namespace multiplication_factor_correct_l2406_240628

theorem multiplication_factor_correct (N X : ℝ) (h1 : 98 = abs ((N * X - N / 10) / (N * X)) * 100) : X = 5 := by
  sorry

end multiplication_factor_correct_l2406_240628


namespace median_of_consecutive_integers_l2406_240635

theorem median_of_consecutive_integers (a b : ℤ) (h : a + b = 50) : 
  (a + b) / 2 = 25 := 
by 
  sorry

end median_of_consecutive_integers_l2406_240635


namespace height_percentage_difference_l2406_240607

theorem height_percentage_difference
  (h_B h_A : ℝ)
  (hA_def : h_A = h_B * 0.55) :
  ((h_B - h_A) / h_A) * 100 = 81.82 := by 
  sorry

end height_percentage_difference_l2406_240607


namespace dante_initially_has_8_jelly_beans_l2406_240633

-- Conditions
def aaron_jelly_beans : ℕ := 5
def bianca_jelly_beans : ℕ := 7
def callie_jelly_beans : ℕ := 8
def dante_jelly_beans_initially (D : ℕ) : Prop := 
  ∀ (D : ℕ), (6 ≤ D - 1 ∧ D - 1 ≤ callie_jelly_beans - 1)

-- Theorem
theorem dante_initially_has_8_jelly_beans :
  ∃ (D : ℕ), (aaron_jelly_beans + 1 = 6) →
             (callie_jelly_beans = 8) →
             dante_jelly_beans_initially D →
             D = 8 := 
by
  sorry

end dante_initially_has_8_jelly_beans_l2406_240633


namespace total_yards_run_l2406_240666

-- Define the yardages and games for each athlete
def Malik_yards_per_game : ℕ := 18
def Malik_games : ℕ := 5

def Josiah_yards_per_game : ℕ := 22
def Josiah_games : ℕ := 7

def Darnell_yards_per_game : ℕ := 11
def Darnell_games : ℕ := 4

def Kade_yards_per_game : ℕ := 15
def Kade_games : ℕ := 6

-- Prove that the total yards run by the four athletes is 378
theorem total_yards_run :
  (Malik_yards_per_game * Malik_games) +
  (Josiah_yards_per_game * Josiah_games) +
  (Darnell_yards_per_game * Darnell_games) +
  (Kade_yards_per_game * Kade_games) = 378 :=
by
  sorry

end total_yards_run_l2406_240666


namespace total_capacity_is_1600_l2406_240669

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l2406_240669


namespace factorization_of_x_squared_minus_64_l2406_240606

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l2406_240606


namespace temperature_decrease_l2406_240618

theorem temperature_decrease (initial : ℤ) (decrease : ℤ) : initial = -3 → decrease = 6 → initial - decrease = -9 :=
by
  intros
  sorry

end temperature_decrease_l2406_240618


namespace souvenirs_expenses_l2406_240686

/--
  Given:
  1. K = T + 146.00
  2. T + K = 548.00
  Prove: 
  - K = 347.00
-/
theorem souvenirs_expenses (T K : ℝ) (h1 : K = T + 146) (h2 : T + K = 548) : K = 347 :=
  sorry

end souvenirs_expenses_l2406_240686


namespace cupcakes_per_package_l2406_240625

theorem cupcakes_per_package
  (packages : ℕ) (total_left : ℕ) (cupcakes_eaten : ℕ) (initial_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_packages = 3)
  (h2 : cupcakes_eaten = 5)
  (h3 : total_left = 7)
  (h4 : packages = initial_packages * cupcakes_per_package - cupcakes_eaten)
  (h5 : packages = total_left) : 
  cupcakes_per_package = 4 := 
by
  sorry

end cupcakes_per_package_l2406_240625


namespace solution_set_inequality_l2406_240675

theorem solution_set_inequality (a b x : ℝ) (h₀ : {x : ℝ | ax - b < 0} = {x : ℝ | 1 < x}) :
  {x : ℝ | (ax + b) * (x - 3) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_inequality_l2406_240675


namespace exponential_function_solution_l2406_240683

theorem exponential_function_solution (a : ℝ) (h₁ : ∀ x : ℝ, a ^ x > 0) :
  (∃ y : ℝ, y = a ^ 2 ∧ y = 4) → a = 2 :=
by
  sorry

end exponential_function_solution_l2406_240683


namespace system_of_equations_solution_l2406_240672

theorem system_of_equations_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 3) : 
  x = 4 ∧ y = 1 :=
by
  sorry

end system_of_equations_solution_l2406_240672


namespace annette_weights_more_l2406_240673

variable (A C S B : ℝ)

theorem annette_weights_more :
  A + C = 95 ∧
  C + S = 87 ∧
  A + S = 97 ∧
  C + B = 100 ∧
  A + C + B = 155 →
  A - S = 8 := by
  sorry

end annette_weights_more_l2406_240673


namespace arithmetic_mean_l2406_240639

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 5/9) :
  (a + b) / 2 = 31/63 := 
by 
  sorry

end arithmetic_mean_l2406_240639


namespace books_in_library_l2406_240649

theorem books_in_library (n_shelves : ℕ) (n_books_per_shelf : ℕ) (h_shelves : n_shelves = 1780) (h_books_per_shelf : n_books_per_shelf = 8) :
  n_shelves * n_books_per_shelf = 14240 :=
by
  -- Skipping the proof as instructed
  sorry

end books_in_library_l2406_240649


namespace divide_two_equal_parts_divide_four_equal_parts_l2406_240660

-- the figure is bounded by three semicircles
def figure_bounded_by_semicircles 
-- two have the same radius r1 
(r1 r2 r3 : ℝ) 
-- the third has twice the radius r3 = 2 * r1
(h_eq : r3 = 2 * r1) 
-- Let's denote the figure as F
(F : Type) :=
-- conditions for r1 and r2
r1 > 0 ∧ r2 = r1 ∧ r3 = 2 * r1

-- Prove the figure can be divided into two equal parts.
theorem divide_two_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 : F), H1 ≠ H2 ∧ H1 = H2 :=
sorry

-- Prove the figure can be divided into four equal parts.
theorem divide_four_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 H3 H4 : F), H1 ≠ H2 ∧ H2 ≠ H3 ∧ H3 ≠ H4 ∧ H1 = H2 ∧ H2 = H3 ∧ H3 = H4 :=
sorry

end divide_two_equal_parts_divide_four_equal_parts_l2406_240660


namespace proof_problem_l2406_240620

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l2406_240620


namespace no_rectangle_from_five_distinct_squares_l2406_240616

theorem no_rectangle_from_five_distinct_squares (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : q1 < q2) 
  (h2 : q2 < q3) 
  (h3 : q3 < q4) 
  (h4 : q4 < q5) : 
  ¬∃(a b: ℝ), a * b = 5 ∧ a = q1 + q2 + q3 + q4 + q5 := sorry

end no_rectangle_from_five_distinct_squares_l2406_240616


namespace highland_high_students_highland_high_num_both_clubs_l2406_240623

theorem highland_high_students (total_students drama_club science_club either_both both_clubs : ℕ)
  (h1 : total_students = 320)
  (h2 : drama_club = 90)
  (h3 : science_club = 140)
  (h4 : either_both = 200) : 
  both_clubs = drama_club + science_club - either_both :=
by
  sorry

noncomputable def num_both_clubs : ℕ :=
if h : 320 = 320 ∧ 90 = 90 ∧ 140 = 140 ∧ 200 = 200
then 90 + 140 - 200
else 0

theorem highland_high_num_both_clubs : num_both_clubs = 30 :=
by
  sorry

end highland_high_students_highland_high_num_both_clubs_l2406_240623


namespace smallest_w_l2406_240651

theorem smallest_w (w : ℕ) (w_pos : w > 0) (h1 : ∀ n : ℕ, 2^4 ∣ 1452 * w)
                              (h2 : ∀ n : ℕ, 3^3 ∣ 1452 * w)
                              (h3 : ∀ n : ℕ, 13^3 ∣ 1452 * w) :
  w = 676 := sorry

end smallest_w_l2406_240651


namespace inequality_for_positive_real_numbers_l2406_240614

theorem inequality_for_positive_real_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
  sorry

end inequality_for_positive_real_numbers_l2406_240614


namespace g_of_3_l2406_240680

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 4 * g x + 3 * g (1 / x) = 2 * x) :
  g 3 = 22 / 7 :=
sorry

end g_of_3_l2406_240680


namespace problem_l2406_240634

def vec_a : ℝ × ℝ := (5, 3)
def vec_b : ℝ × ℝ := (1, -2)
def two_vec_b : ℝ × ℝ := (2 * 1, 2 * -2)
def expected_result : ℝ × ℝ := (3, 7)

theorem problem : (vec_a.1 - two_vec_b.1, vec_a.2 - two_vec_b.2) = expected_result :=
by
  sorry

end problem_l2406_240634


namespace algebraic_expression_value_l2406_240632

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
by
  sorry

end algebraic_expression_value_l2406_240632


namespace person_speed_l2406_240646

noncomputable def distance_meters : ℝ := 1080
noncomputable def time_minutes : ℝ := 14
noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := time_minutes / 60
noncomputable def speed_km_per_hour : ℝ := distance_kilometers / time_hours

theorem person_speed :
  abs (speed_km_per_hour - 4.63) < 0.01 :=
by
  -- conditions extracted
  let distance_in_km := distance_meters / 1000
  let time_in_hours := time_minutes / 60
  let speed := distance_in_km / time_in_hours
  -- We expect speed to be approximately 4.63
  sorry 

end person_speed_l2406_240646


namespace find_k_l2406_240644

theorem find_k (k : ℕ) : (1 / 2) ^ 18 * (1 / 81) ^ k = 1 / 18 ^ 18 → k = 0 := by
  intro h
  sorry

end find_k_l2406_240644


namespace sequence_count_l2406_240608

theorem sequence_count :
  ∃ f : ℕ → ℕ,
    (f 3 = 1) ∧ (f 4 = 1) ∧ (f 5 = 1) ∧ (f 6 = 2) ∧ (f 7 = 2) ∧
    (∀ n, n ≥ 8 → f n = f (n-4) + 2 * f (n-5) + f (n-6)) ∧
    f 15 = 21 :=
by {
  sorry
}

end sequence_count_l2406_240608


namespace fixed_monthly_fee_l2406_240604

theorem fixed_monthly_fee (x y : ℝ)
  (h₁ : x + y = 18.70)
  (h₂ : x + 3 * y = 34.10) : x = 11.00 :=
by sorry

end fixed_monthly_fee_l2406_240604


namespace hulk_jump_distance_exceeds_1000_l2406_240691

theorem hulk_jump_distance_exceeds_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 3^m ≤ 1000) ∧ 3^n > 1000 :=
sorry

end hulk_jump_distance_exceeds_1000_l2406_240691


namespace worker_and_robot_capacity_additional_workers_needed_l2406_240662

-- Definitions and conditions
def worker_capacity (x : ℕ) : Prop :=
  (1 : ℕ) * x + 420 = 420 + x

def time_equivalence (x : ℕ) : Prop :=
  900 * 10 * x = 600 * (x + 420)

-- First part of the proof problem
theorem worker_and_robot_capacity (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  x = 30 ∧ x + 420 = 450 :=
by
  sorry

-- Second part of the proof problem
theorem additional_workers_needed (x : ℕ) (hx_w : worker_capacity x) (hx_t : time_equivalence x) :
  3 * (x + 420) * 2 < 3600 →
  2 * 30 * 15 ≥ 3600 - 2 * 3 * (x + 420) :=
by
  sorry

end worker_and_robot_capacity_additional_workers_needed_l2406_240662


namespace total_treats_value_l2406_240643

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l2406_240643


namespace meeting_time_l2406_240641

/--
The Racing Magic takes 150 seconds to circle the racing track once.
The Charging Bull makes 40 rounds of the track in an hour.
Prove that Racing Magic and Charging Bull meet at the starting point for the second time 
after 300 minutes.
-/
theorem meeting_time (rac_magic_time : ℕ) (chrg_bull_rounds_hour : ℕ)
  (h1 : rac_magic_time = 150) (h2 : chrg_bull_rounds_hour = 40) : 
  ∃ t: ℕ, t = 300 := 
by
  sorry

end meeting_time_l2406_240641


namespace sam_gave_2_puppies_l2406_240688

theorem sam_gave_2_puppies (original_puppies given_puppies remaining_puppies : ℕ) 
  (h1 : original_puppies = 6) (h2 : remaining_puppies = 4) :
  given_puppies = original_puppies - remaining_puppies := by 
  sorry

end sam_gave_2_puppies_l2406_240688


namespace some_number_value_l2406_240645

theorem some_number_value (some_number : ℝ): 
  (∀ n : ℝ, (n / some_number) * (n / 80) = 1 → n = 40) → some_number = 80 :=
by
  sorry

end some_number_value_l2406_240645


namespace calc_sqrt_expr_l2406_240679

theorem calc_sqrt_expr :
  (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end calc_sqrt_expr_l2406_240679


namespace min_girls_in_class_l2406_240627

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l2406_240627


namespace jogger_ahead_engine_l2406_240638

-- Define the given constants for speed and length
def jogger_speed : ℝ := 2.5 -- in m/s
def train_speed : ℝ := 12.5 -- in m/s
def train_length : ℝ := 120 -- in meters
def passing_time : ℝ := 40 -- in seconds

-- Define the target distance
def jogger_ahead : ℝ := 280 -- in meters

-- Lean 4 statement to prove the jogger is 280 meters ahead of the train's engine
theorem jogger_ahead_engine :
  passing_time * (train_speed - jogger_speed) - train_length = jogger_ahead :=
by
  sorry

end jogger_ahead_engine_l2406_240638


namespace scientific_notation_example_l2406_240684

theorem scientific_notation_example :
  ∃ (a : ℝ) (b : ℤ), 1300000 = a * 10 ^ b ∧ a = 1.3 ∧ b = 6 :=
sorry

end scientific_notation_example_l2406_240684


namespace shirts_made_today_l2406_240689

def shirts_per_minute : ℕ := 8
def working_minutes : ℕ := 2

theorem shirts_made_today (h1 : shirts_per_minute = 8) (h2 : working_minutes = 2) : shirts_per_minute * working_minutes = 16 := by
  sorry

end shirts_made_today_l2406_240689


namespace frog_vertical_boundary_prob_l2406_240664

-- Define the type of points on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define the type of the rectangle
structure Rectangle where
  left_bottom : Point
  right_top : Point

-- Conditions
def start_point : Point := ⟨2, 3⟩
def boundary : Rectangle := ⟨⟨0, 0⟩, ⟨5, 5⟩⟩

-- Define the probability function
noncomputable def P (p : Point) : ℚ := sorry

-- Symmetry relations and recursive relations
axiom symmetry_P23 : P ⟨2, 3⟩ = P ⟨3, 3⟩
axiom symmetry_P22 : P ⟨2, 2⟩ = P ⟨3, 2⟩
axiom recursive_P23 : P ⟨2, 3⟩ = 1 / 4 + 1 / 4 * P ⟨2, 2⟩ + 1 / 4 * P ⟨1, 3⟩ + 1 / 4 * P ⟨3, 3⟩

-- Main Theorem
theorem frog_vertical_boundary_prob :
  P start_point = 2 / 3 := sorry

end frog_vertical_boundary_prob_l2406_240664


namespace original_number_eq_0_000032_l2406_240631

theorem original_number_eq_0_000032 (x : ℝ) (hx : 0 < x) 
  (h : 10^8 * x = 8 * (1 / x)) : x = 0.000032 :=
sorry

end original_number_eq_0_000032_l2406_240631


namespace variance_of_data_set_l2406_240681

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set_l2406_240681


namespace simplify_one_simplify_two_simplify_three_simplify_four_l2406_240656

-- (1) Prove that (1 / 2) * sqrt(4 / 7) = sqrt(7) / 7
theorem simplify_one : (1 / 2) * Real.sqrt (4 / 7) = Real.sqrt 7 / 7 := sorry

-- (2) Prove that sqrt(20 ^ 2 - 15 ^ 2) = 5 * sqrt(7)
theorem simplify_two : Real.sqrt (20 ^ 2 - 15 ^ 2) = 5 * Real.sqrt 7 := sorry

-- (3) Prove that sqrt((32 * 9) / 25) = (12 * sqrt(2)) / 5
theorem simplify_three : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := sorry

-- (4) Prove that sqrt(22.5) = (3 * sqrt(10)) / 2
theorem simplify_four : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := sorry

end simplify_one_simplify_two_simplify_three_simplify_four_l2406_240656


namespace reachable_target_l2406_240657

-- Define the initial state of the urn
def initial_urn_state : (ℕ × ℕ) := (150, 50)

-- Define the operations as changes in counts of black and white marbles
def operation1 (state : ℕ × ℕ) := (state.1 - 2, state.2)
def operation2 (state : ℕ × ℕ) := (state.1 - 1, state.2)
def operation3 (state : ℕ × ℕ) := (state.1, state.2 - 2)
def operation4 (state : ℕ × ℕ) := (state.1 + 2, state.2 - 3)

-- Define a predicate that a state can be reached from the initial state
def reachable (target : ℕ × ℕ) : Prop :=
  ∃ n1 n2 n3 n4 : ℕ, 
    operation1^[n1] (operation2^[n2] (operation3^[n3] (operation4^[n4] initial_urn_state))) = target

-- The theorem to be proved
theorem reachable_target : reachable (1, 2) :=
sorry

end reachable_target_l2406_240657


namespace B_grazed_months_l2406_240609

-- Define the conditions
variables (A_cows B_cows C_cows D_cows : ℕ)
variables (A_months B_months C_months D_months : ℕ)
variables (A_rent total_rent : ℕ)

-- Given conditions
def A_condition := (A_cows = 24 ∧ A_months = 3)
def B_condition := (B_cows = 10)
def C_condition := (C_cows = 35 ∧ C_months = 4)
def D_condition := (D_cows = 21 ∧ D_months = 3)
def A_rent_condition := (A_rent = 720)
def total_rent_condition := (total_rent = 3250)

-- Define cow-months calculation
def cow_months (cows months : ℕ) : ℕ := cows * months

-- Define cost per cow-month
def cost_per_cow_month (rent cow_months : ℕ) : ℕ := rent / cow_months

-- Define B's months of grazing proof problem
theorem B_grazed_months
  (A_cows_months : cow_months 24 3 = 72)
  (B_cows := 10)
  (C_cows_months : cow_months 35 4 = 140)
  (D_cows_months : cow_months 21 3 = 63)
  (A_rent_condition : A_rent = 720)
  (total_rent_condition : total_rent = 3250) :
  ∃ (B_months : ℕ), 10 * B_months = 50 ∧ B_months = 5 := sorry

end B_grazed_months_l2406_240609


namespace nadine_white_pebbles_l2406_240696

variable (W R : ℝ)

theorem nadine_white_pebbles :
  (R = 1/2 * W) →
  (W + R = 30) →
  W = 20 :=
by
  sorry

end nadine_white_pebbles_l2406_240696


namespace notebooks_difference_l2406_240690

noncomputable def price_more_than_dime (p : ℝ) : Prop := p > 0.10
noncomputable def payment_equation (nL nN : ℕ) (p : ℝ) : Prop :=
  (nL * p = 2.10 ∧ nN * p = 2.80)

theorem notebooks_difference (nL nN : ℕ) (p : ℝ) (h1 : price_more_than_dime p) (h2 : payment_equation nL nN p) :
  nN - nL = 2 :=
by sorry

end notebooks_difference_l2406_240690


namespace number_of_uncracked_seashells_l2406_240699

theorem number_of_uncracked_seashells (toms_seashells freds_seashells cracked_seashells : ℕ) 
  (h_tom : toms_seashells = 15) 
  (h_fred : freds_seashells = 43) 
  (h_cracked : cracked_seashells = 29) : 
  toms_seashells + freds_seashells - cracked_seashells = 29 :=
by
  sorry

end number_of_uncracked_seashells_l2406_240699


namespace find_units_min_selling_price_l2406_240629

-- Definitions for the given conditions
def total_units : ℕ := 160
def cost_A : ℕ := 150
def cost_B : ℕ := 350
def total_cost : ℕ := 36000
def min_profit : ℕ := 11000

-- Part 1: Proving number of units purchased
theorem find_units :
  ∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x :=
by
  sorry

-- Part 2: Finding the minimum selling price per unit of model A for the profit condition
theorem min_selling_price (t : ℕ) :
  (∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x) →
  100 * (t - cost_A) + 60 * 2 * (t - cost_A) ≥ min_profit →
  t ≥ 200 :=
by
  sorry

end find_units_min_selling_price_l2406_240629


namespace john_total_distance_l2406_240676

theorem john_total_distance (speed1 time1 speed2 time2 : ℕ) (distance1 distance2 : ℕ) :
  speed1 = 35 →
  time1 = 2 →
  speed2 = 55 →
  time2 = 3 →
  distance1 = speed1 * time1 →
  distance2 = speed2 * time2 →
  distance1 + distance2 = 235 := by
  intros
  sorry

end john_total_distance_l2406_240676


namespace nickels_eq_100_l2406_240605

variables (P D N Q H DollarCoins : ℕ)

def conditions :=
  D = P + 10 ∧
  N = 2 * D ∧
  Q = 4 ∧
  P = 10 * Q ∧
  H = Q + 5 ∧
  DollarCoins = 3 * H ∧
  (P + 10 * D + 5 * N + 25 * Q + 50 * H + 100 * DollarCoins = 2000)

theorem nickels_eq_100 (h : conditions P D N Q H DollarCoins) : N = 100 :=
by {
  sorry
}

end nickels_eq_100_l2406_240605


namespace expression_values_l2406_240650

theorem expression_values (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ b)
  (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 := 
sorry

end expression_values_l2406_240650


namespace cost_price_of_book_l2406_240697

theorem cost_price_of_book
(marked_price : ℝ)
(list_price : ℝ)
(cost_price : ℝ)
(h1 : marked_price = 69.85)
(h2 : list_price = marked_price * 0.85)
(h3 : list_price = cost_price * 1.25) :
cost_price = 65.75 :=
by
  sorry

end cost_price_of_book_l2406_240697


namespace increase_by_thirteen_possible_l2406_240612

-- Define the main condition which states the reduction of the original product
def product_increase_by_thirteen (a : Fin 7 → ℕ) : Prop :=
  let P := (List.range 7).map (fun i => a ⟨i, sorry⟩) |>.prod
  let Q := (List.range 7).map (fun i => a ⟨i, sorry⟩ - 3) |>.prod
  Q = 13 * P

-- State the theorem to be proved
theorem increase_by_thirteen_possible : ∃ (a : Fin 7 → ℕ), product_increase_by_thirteen a :=
sorry

end increase_by_thirteen_possible_l2406_240612


namespace tiffany_reading_homework_pages_l2406_240602

theorem tiffany_reading_homework_pages 
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (reading_pages : ℕ)
  (H1 : math_pages = 6)
  (H2 : problems_per_page = 3)
  (H3 : total_problems = 30)
  (H4 : reading_pages = (total_problems - math_pages * problems_per_page) / problems_per_page) 
  : reading_pages = 4 := 
sorry

end tiffany_reading_homework_pages_l2406_240602


namespace intersection_eq_N_l2406_240663

def U := Set ℝ                                        -- Universal set U = ℝ
def M : Set ℝ := {x | x ≥ 0}                         -- Set M = {x | x ≥ 0}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}                 -- Set N = {x | 0 ≤ x ≤ 1}

theorem intersection_eq_N : M ∩ N = N := by
  sorry

end intersection_eq_N_l2406_240663


namespace water_current_speed_l2406_240615

theorem water_current_speed (v : ℝ) (swimmer_speed : ℝ := 4) (time : ℝ := 3.5) (distance : ℝ := 7) :
  (4 - v) = distance / time → v = 2 := 
by
  sorry

end water_current_speed_l2406_240615


namespace distinct_sequences_count_l2406_240692

-- Defining the set of letters in "PROBLEMS"
def letters : List Char := ['P', 'R', 'O', 'B', 'L', 'E', 'M']

-- Defining a sequence constraint: must start with 'S' and not end with 'M'
def valid_sequence (seq : List Char) : Prop :=
  seq.head? = some 'S' ∧ seq.getLast? ≠ some 'M'

-- Counting valid sequences according to the constraints
noncomputable def count_valid_sequences : Nat :=
  6 * 120

theorem distinct_sequences_count :
  count_valid_sequences = 720 := by
  sorry

end distinct_sequences_count_l2406_240692


namespace problem_statement_l2406_240661

theorem problem_statement (p q m n : ℕ) (x : ℚ)
  (h1 : p / q = 4 / 5) (h2 : m / n = 4 / 5) (h3 : x = 1 / 7) :
  x + (2 * q - p + 3 * m - 2 * n) / (2 * q + p - m + n) = 71 / 105 :=
by
  sorry

end problem_statement_l2406_240661


namespace each_interior_angle_of_regular_octagon_l2406_240654

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l2406_240654
