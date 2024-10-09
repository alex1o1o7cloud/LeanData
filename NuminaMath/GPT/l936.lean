import Mathlib

namespace terminal_velocity_steady_speed_l936_93666

variable (g : ℝ) (t₁ t₂ : ℝ) (a₀ a₁ : ℝ) (v_terminal : ℝ)

-- Conditions
def acceleration_due_to_gravity := g = 10 -- m/s²
def initial_time := t₁ = 0 -- s
def intermediate_time := t₂ = 2 -- s
def initial_acceleration := a₀ = 50 -- m/s²
def final_acceleration := a₁ = 10 -- m/s²

-- Question: Prove the terminal velocity
theorem terminal_velocity_steady_speed 
  (h_g : acceleration_due_to_gravity g)
  (h_t1 : initial_time t₁)
  (h_t2 : intermediate_time t₂)
  (h_a0 : initial_acceleration a₀)
  (h_a1 : final_acceleration a₁) :
  v_terminal = 25 :=
  sorry

end terminal_velocity_steady_speed_l936_93666


namespace simplify_expression_l936_93665

variable (d : ℤ)

theorem simplify_expression :
  (5 + 4 * d) / 9 - 3 + 1 / 3 = (4 * d - 19) / 9 := by
  sorry

end simplify_expression_l936_93665


namespace mary_earnings_max_hours_l936_93607

noncomputable def earnings (hours : ℕ) : ℝ :=
  if hours <= 40 then 
    hours * 10
  else if hours <= 60 then 
    (40 * 10) + ((hours - 40) * 13)
  else 
    (40 * 10) + (20 * 13) + ((hours - 60) * 16)

theorem mary_earnings_max_hours : 
  earnings 70 = 820 :=
by
  sorry

end mary_earnings_max_hours_l936_93607


namespace flight_cost_A_to_B_l936_93690

-- Definitions based on conditions in the problem
def distance_AB : ℝ := 2000
def flight_cost_per_km : ℝ := 0.10
def booking_fee : ℝ := 100

-- Statement: Given the distances and cost conditions, the flight cost from A to B is $300
theorem flight_cost_A_to_B : distance_AB * flight_cost_per_km + booking_fee = 300 := by
  sorry

end flight_cost_A_to_B_l936_93690


namespace N_subset_proper_M_l936_93699

open Set Int

def set_M : Set ℝ := {x | ∃ k : ℤ, x = (k + 2) / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) / 4}

theorem N_subset_proper_M : set_N ⊂ set_M := by
  sorry

end N_subset_proper_M_l936_93699


namespace lowest_fraction_of_job_done_l936_93630

theorem lowest_fraction_of_job_done :
  ∀ (rateA rateB rateC rateB_plus_C : ℝ),
  (rateA = 1/4) → (rateB = 1/6) → (rateC = 1/8) →
  (rateB_plus_C = rateB + rateC) →
  rateB_plus_C = 7/24 := by
  intros rateA rateB rateC rateB_plus_C hA hB hC hBC
  sorry

end lowest_fraction_of_job_done_l936_93630


namespace basis_transformation_l936_93670

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem basis_transformation (h_basis : ∀ (v : V), ∃ (x y z : ℝ), v = x • a + y • b + z • c) :
  ∀ (v : V), ∃ (x y z : ℝ), v = x • (a + b) + y • (a - c) + z • b :=
by {
  sorry  -- to skip the proof steps for now
}

end basis_transformation_l936_93670


namespace additional_airplanes_needed_l936_93696

theorem additional_airplanes_needed (total_current_airplanes : ℕ) (airplanes_per_row : ℕ) 
  (h_current_airplanes : total_current_airplanes = 37) 
  (h_airplanes_per_row : airplanes_per_row = 8) : 
  ∃ additional_airplanes : ℕ, additional_airplanes = 3 ∧ 
  ((total_current_airplanes + additional_airplanes) % airplanes_per_row = 0) :=
by
  sorry

end additional_airplanes_needed_l936_93696


namespace remaining_fruits_correct_l936_93618

-- The definitions for the number of fruits in terms of the number of plums
def apples := 180
def plums := apples / 3
def pears := 2 * plums
def cherries := 4 * apples

-- Damien's portion of each type of fruit picked
def apples_picked := (3/5) * apples
def plums_picked := (2/3) * plums
def pears_picked := (3/4) * pears
def cherries_picked := (7/10) * cherries

-- The remaining number of fruits
def apples_remaining := apples - apples_picked
def plums_remaining := plums - plums_picked
def pears_remaining := pears - pears_picked
def cherries_remaining := cherries - cherries_picked

-- The total remaining number of fruits
def total_remaining_fruits := apples_remaining + plums_remaining + pears_remaining + cherries_remaining

theorem remaining_fruits_correct :
  total_remaining_fruits = 338 :=
by {
  -- The conditions ensure that the imported libraries are broad
  sorry
}

end remaining_fruits_correct_l936_93618


namespace usb_drive_total_capacity_l936_93610

-- Define the conditions as α = total capacity, β = busy space (50%), γ = available space (50%)
variable (α : ℕ) -- Total capacity of the USB drive in gigabytes
variable (β γ : ℕ) -- Busy space and available space in gigabytes
variable (h1 : β = α / 2) -- 50% of total capacity is busy
variable (h2 : γ = 8)  -- 8 gigabytes are still available

-- Define the problem as a theorem that these conditions imply the total capacity
theorem usb_drive_total_capacity (h : γ = α / 2) : α = 16 :=
by
  -- defer the proof
  sorry

end usb_drive_total_capacity_l936_93610


namespace min_n_such_that_no_more_possible_l936_93661

-- Define a seven-cell corner as a specific structure within the grid
inductive Corner
| cell7 : Corner

-- Function to count the number of cells clipped out by n corners
def clipped_cells (n : ℕ) : ℕ := 7 * n

-- Statement to be proven
theorem min_n_such_that_no_more_possible (n : ℕ) (h_n : n ≥ 3) (h_max : n < 4) :
  ¬ ∃ k : ℕ, k > n ∧ clipped_cells k ≤ 64 :=
by {
  sorry -- Proof goes here
}

end min_n_such_that_no_more_possible_l936_93661


namespace remainder_of_division_l936_93620

theorem remainder_of_division (x r : ℕ) (h : 23 = 7 * x + r) : r = 2 :=
sorry

end remainder_of_division_l936_93620


namespace original_number_of_men_l936_93653

variable (M W : ℕ)

def original_work_condition := M * W / 60 = W
def larger_group_condition := (M + 8) * W / 50 = W

theorem original_number_of_men : original_work_condition M W ∧ larger_group_condition M W → M = 48 :=
by
  sorry

end original_number_of_men_l936_93653


namespace band_first_set_songs_count_l936_93697

theorem band_first_set_songs_count 
  (total_repertoire : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ)
  (h_total_repertoire : total_repertoire = 30)
  (h_second_set : second_set = 7)
  (h_encore : encore = 2)
  (h_avg_third_fourth : avg_third_fourth = 8)
  : ∃ (x : ℕ), x + second_set + encore + avg_third_fourth * 2 = total_repertoire := 
  sorry

end band_first_set_songs_count_l936_93697


namespace find_original_denominator_l936_93656

theorem find_original_denominator (d : ℕ) 
  (h : (10 : ℚ) / (d + 7) = 1 / 3) : 
  d = 23 :=
by 
  sorry

end find_original_denominator_l936_93656


namespace equation_of_latus_rectum_l936_93647

theorem equation_of_latus_rectum (y x : ℝ) : (x = -1/4) ∧ (y^2 = x) ↔ (2 * (1 / 2) = 1) ∧ (l = - (1 / 2) / 2) := sorry

end equation_of_latus_rectum_l936_93647


namespace smallest_number_divisible_by_15_and_36_l936_93608

theorem smallest_number_divisible_by_15_and_36 : 
  ∃ x, (∀ y, (y % 15 = 0 ∧ y % 36 = 0) → y ≥ x) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_15_and_36_l936_93608


namespace rotten_eggs_prob_l936_93645

theorem rotten_eggs_prob (T : ℕ) (P : ℝ) (R : ℕ) :
  T = 36 ∧ P = 0.0047619047619047615 ∧ P = (R / T) * ((R - 1) / (T - 1)) → R = 3 :=
by
  sorry

end rotten_eggs_prob_l936_93645


namespace smallest_enclosing_sphere_radius_l936_93685

-- Define the radius of each small sphere and the center set
def radius (r : ℝ) : Prop := r = 2

def center_set (C : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ c ∈ C, ∃ x y z : ℝ, 
    (x = 2 ∨ x = -2) ∧ 
    (y = 2 ∨ y = -2) ∧ 
    (z = 2 ∨ z = -2) ∧
    (c = (x, y, z))

-- Prove the radius of the smallest enclosing sphere is 2√3 + 2
theorem smallest_enclosing_sphere_radius (r : ℝ) (C : Set (ℝ × ℝ × ℝ)) 
  (h_radius : radius r) (h_center_set : center_set C) :
  ∃ R : ℝ, R = 2 * Real.sqrt 3 + 2 :=
sorry

end smallest_enclosing_sphere_radius_l936_93685


namespace rectangles_containment_existence_l936_93671

theorem rectangles_containment_existence :
  (∃ (rects : ℕ → ℕ × ℕ), (∀ n : ℕ, (rects n).fst > 0 ∧ (rects n).snd > 0) ∧
   (∀ n m : ℕ, n ≠ m → ¬((rects n).fst ≤ (rects m).fst ∧ (rects n).snd ≤ (rects m).snd))) →
  false :=
by
  sorry

end rectangles_containment_existence_l936_93671


namespace eq_x2_inv_x2_and_x8_inv_x8_l936_93605

theorem eq_x2_inv_x2_and_x8_inv_x8 (x : ℝ) 
  (h : 47 = x^4 + 1 / x^4) : 
  (x^2 + 1 / x^2 = 7) ∧ (x^8 + 1 / x^8 = -433) :=
by
  sorry

end eq_x2_inv_x2_and_x8_inv_x8_l936_93605


namespace base_b_of_256_has_4_digits_l936_93624

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l936_93624


namespace find_number_of_white_balls_l936_93648

-- Define the conditions
variables (n k : ℕ)
axiom k_ge_2 : k ≥ 2
axiom prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100

-- State the theorem
theorem find_number_of_white_balls (n k : ℕ) (k_ge_2 : k ≥ 2) (prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100) : n = 19 :=
sorry

end find_number_of_white_balls_l936_93648


namespace total_green_marbles_l936_93675

-- Conditions
def Sara_green_marbles : ℕ := 3
def Tom_green_marbles : ℕ := 4

-- Problem statement: proving the total number of green marbles
theorem total_green_marbles : Sara_green_marbles + Tom_green_marbles = 7 := by
  sorry

end total_green_marbles_l936_93675


namespace circle_tangent_proof_l936_93664

noncomputable def circle_tangent_range : Set ℝ :=
  { k : ℝ | k > 0 ∧ ((3 - 2 * k)^2 + (1 - k)^2 > k) }

theorem circle_tangent_proof :
  ∀ k > 0, ((3 - 2 * k)^2 + (1 - k)^2 > k) ↔ (k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 2)) :=
by
  sorry

end circle_tangent_proof_l936_93664


namespace proof_f_value_l936_93698

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 1 - x^2 else 2^x

theorem proof_f_value : f (1 / f (Real.log 6 / Real.log 2)) = 35 / 36 := by
  sorry

end proof_f_value_l936_93698


namespace arith_seq_formula_geom_seq_sum_l936_93660

-- Definitions for condition 1: Arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  (a 4 = 7) ∧ (a 10 = 19)

-- Definitions for condition 2: Sum of the first n terms of {a_n}
def sum_arith_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Definitions for condition 3: Geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop :=
  (b 1 = 2) ∧ (∀ n, b (n + 1) = b n * 2)

-- Definitions for condition 4: Sum of the first n terms of {b_n}
def sum_geom_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n, T n = (b 1 * (1 - (2 ^ n))) / (1 - 2)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula (a : ℕ → ℤ) (S : ℕ → ℤ) :
  arithmetic_seq a ∧ sum_arith_seq S a → 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, S n = n ^ 2) :=
sorry

-- Proving the sum of the first n terms for geometric sequence
theorem geom_seq_sum (b : ℕ → ℤ) (T : ℕ → ℤ) (S : ℕ → ℤ) :
  geometric_seq b ∧ sum_geom_seq T b ∧ b 4 = S 4 → 
  (∀ n, T n = 2 ^ (n + 1) - 2) :=
sorry

end arith_seq_formula_geom_seq_sum_l936_93660


namespace part1_part2_l936_93627

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part1 (hx : f (-x) = 2 * f x) : f x ^ 2 = 2 / 5 := 
  sorry

theorem part2 : 
  ∀ k : ℤ, ∃ a b : ℝ, [a, b] = [2 * π * k + (5 * π / 6), 2 * π * k + (11 * π / 6)] ∧ 
  ∀ x : ℝ, x ∈ Set.Icc a b → ∀ y : ℝ, y = f (π / 12 - x) → 
  ∃ δ > 0, ∀ ε > 0, 0 < |x - y| ∧ |x - y| < δ → y < x := 
  sorry

end part1_part2_l936_93627


namespace JamesFlowers_l936_93662

noncomputable def numberOfFlowersJamesPlantedInADay (F : ℝ) := 0.5 * (F + 0.15 * F)

theorem JamesFlowers (F : ℝ) (H₁ : 6 * F + (F + 0.15 * F) = 315) : numberOfFlowersJamesPlantedInADay F = 25.3:=
by
  sorry

end JamesFlowers_l936_93662


namespace symmetry_axis_of_f_l936_93658

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∃ k : ℤ, ∃ k_π_div_2 : ℝ, (f (k * π / 2 + π / 12) = f ((k * π / 2 + π / 12) + π)) :=
by {
  sorry
}

end symmetry_axis_of_f_l936_93658


namespace trapezoid_area_l936_93609

variable (a b : ℝ) (h1 : a > b)

theorem trapezoid_area (h2 : ∃ (angle1 angle2 : ℝ), angle1 = 30 ∧ angle2 = 45) : 
  (1/4) * ((a^2 - b^2) * (Real.sqrt 3 - 1)) = 
    ((1/2) * (a + b) * ((b - a) * (Real.sqrt 3 - 1) / 2)) := 
sorry

end trapezoid_area_l936_93609


namespace simplify_cos_diff_l936_93689

theorem simplify_cos_diff :
  let a := Real.cos (36 * Real.pi / 180)
  let b := Real.cos (72 * Real.pi / 180)
  (b = 2 * a^2 - 1) → 
  (a = 1 - 2 * b^2) →
  a - b = 1 / 2 :=
by
  sorry

end simplify_cos_diff_l936_93689


namespace inequality_holds_l936_93691

variable {a b c : ℝ}

theorem inequality_holds (h : a > 0) (h' : b > 0) (h'' : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end inequality_holds_l936_93691


namespace seashells_total_l936_93619

theorem seashells_total {sally tom jessica : ℕ} (h₁ : sally = 9) (h₂ : tom = 7) (h₃ : jessica = 5) : sally + tom + jessica = 21 := by
  sorry

end seashells_total_l936_93619


namespace cost_of_adult_ticket_l936_93626

theorem cost_of_adult_ticket
  (A : ℝ) -- Cost of an adult ticket in dollars
  (x y : ℝ) -- Number of children tickets and number of adult tickets respectively
  (hx : x = 90) -- Condition: number of children tickets sold
  (hSum : x + y = 130) -- Condition: total number of tickets sold
  (hTotal : 4 * x + A * y = 840) -- Condition: total receipts from all tickets
  : A = 12 := 
by
  -- Proof is skipped as per instruction
  sorry

end cost_of_adult_ticket_l936_93626


namespace unknown_road_length_l936_93636

/-
  Given the lengths of four roads and the Triangle Inequality condition, 
  prove the length of the fifth road.
  Given lengths: a = 10 km, b = 5 km, c = 8 km, d = 21 km.
-/

theorem unknown_road_length
  (a b c d : ℕ) (h0 : a = 10) (h1 : b = 5) (h2 : c = 8) (h3 : d = 21)
  (x : ℕ) :
  2 < x ∧ x < 18 ∧ 16 < x ∧ x < 26 → x = 17 :=
by
  intros
  sorry

end unknown_road_length_l936_93636


namespace Sheelas_monthly_income_l936_93643

theorem Sheelas_monthly_income (I : ℝ) (h : 0.32 * I = 3800) : I = 11875 :=
by
  sorry

end Sheelas_monthly_income_l936_93643


namespace find_value_of_f2_sub_f3_l936_93679

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem find_value_of_f2_sub_f3 (h_odd : is_odd_function f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 :=
by
  sorry

end find_value_of_f2_sub_f3_l936_93679


namespace time_for_one_essay_l936_93623

-- We need to define the times for questions and paragraphs first.

def time_per_short_answer_question := 3 -- in minutes
def time_per_paragraph := 15 -- in minutes
def total_homework_time := 4 -- in hours
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15

-- Now we need to state the total homework time and define the goal
def computed_homework_time :=
  (time_per_short_answer_question * num_short_answer_questions +
   time_per_paragraph * num_paragraphs) / 60 + num_essays * sorry -- time for one essay in hours

theorem time_for_one_essay :
  (total_homework_time = computed_homework_time) → sorry = 1 :=
by
  sorry

end time_for_one_essay_l936_93623


namespace prince_spending_l936_93622

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l936_93622


namespace find_other_endpoint_l936_93674

theorem find_other_endpoint (x1 y1 x_m y_m x y : ℝ) 
  (h1 : (x_m, y_m) = (3, 7))
  (h2 : (x1, y1) = (0, 11)) :
  (x, y) = (6, 3) ↔ (x_m = (x1 + x) / 2 ∧ y_m = (y1 + y) / 2) :=
by
  simp at h1 h2
  simp
  sorry

end find_other_endpoint_l936_93674


namespace largest_digit_divisible_by_6_l936_93646

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l936_93646


namespace exp_gt_f_n_y_between_0_and_x_l936_93673

open Real

noncomputable def f_n (x : ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => x^k / k.factorial)

theorem exp_gt_f_n (x : ℝ) (n : ℕ) (h1 : 0 < x) :
  exp x > f_n x n :=
sorry

theorem y_between_0_and_x (x : ℝ) (n : ℕ) (y : ℝ)
  (h1 : 0 < x)
  (h2 : exp x = f_n x n + x^(n+1) / (n + 1).factorial * exp y) :
  0 < y ∧ y < x :=
sorry

end exp_gt_f_n_y_between_0_and_x_l936_93673


namespace cost_of_green_pill_l936_93616

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill_l936_93616


namespace sara_spent_on_hotdog_l936_93649

def total_cost_of_lunch: ℝ := 10.46
def cost_of_salad: ℝ := 5.10
def cost_of_hotdog: ℝ := total_cost_of_lunch - cost_of_salad

theorem sara_spent_on_hotdog :
  cost_of_hotdog = 5.36 := by
  sorry

end sara_spent_on_hotdog_l936_93649


namespace min_period_and_sym_center_l936_93657

open Real

noncomputable def func (x α β : ℝ) : ℝ :=
  sin (x - α) * cos (x - β)

theorem min_period_and_sym_center (α β : ℝ) :
  (∀ x, func (x + π) α β = func x α β) ∧ (func α 0 β = 0) :=
by
  sorry

end min_period_and_sym_center_l936_93657


namespace two_point_three_five_as_fraction_l936_93680

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l936_93680


namespace alpha_cubed_plus_5beta_plus_10_l936_93612

noncomputable def α: ℝ := sorry
noncomputable def β: ℝ := sorry

-- Given conditions
axiom roots_eq : ∀ x : ℝ, x^2 + 2 * x - 1 = 0 → (x = α ∨ x = β)
axiom sum_eq : α + β = -2
axiom prod_eq : α * β = -1

-- The theorem stating the desired result
theorem alpha_cubed_plus_5beta_plus_10 :
  α^3 + 5 * β + 10 = -2 :=
sorry

end alpha_cubed_plus_5beta_plus_10_l936_93612


namespace product_of_all_possible_values_of_x_l936_93672

def conditions (x : ℚ) : Prop := abs (18 / x - 4) = 3

theorem product_of_all_possible_values_of_x:
  ∃ x1 x2 : ℚ, conditions x1 ∧ conditions x2 ∧ ((18 * 18) / (x1 * x2) = 324 / 7) :=
sorry

end product_of_all_possible_values_of_x_l936_93672


namespace radius_squared_l936_93606

theorem radius_squared (r : ℝ) (AB_len CD_len BP_len : ℝ) (angle_APD : ℝ) (r_squared : ℝ) :
  AB_len = 10 →
  CD_len = 7 →
  BP_len = 8 →
  angle_APD = 60 →
  r_squared = r^2 →
  r_squared = 73 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end radius_squared_l936_93606


namespace total_tires_l936_93639

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l936_93639


namespace common_difference_is_two_l936_93652

-- Define the properties and conditions.
variables {a : ℕ → ℝ} {d : ℝ}

-- An arithmetic sequence definition.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement to be proved.
theorem common_difference_is_two (h1 : a 1 + a 5 = 10) (h2 : a 4 = 7) (h3 : arithmetic_sequence a d) : 
  d = 2 :=
sorry

end common_difference_is_two_l936_93652


namespace product_relationship_l936_93681

variable {a_1 a_2 b_1 b_2 : ℝ}

theorem product_relationship (h1 : a_1 < a_2) (h2 : b_1 < b_2) : 
  a_1 * b_1 + a_2 * b_2 > a_1 * b_2 + a_2 * b_1 := 
sorry

end product_relationship_l936_93681


namespace cookie_radius_proof_l936_93641

-- Define the given equation of the cookie
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6 * x + 9 * y

-- Define the radius computation for the circle derived from the given equation
def cookie_radius (r : ℝ) : Prop :=
  r = 3 * Real.sqrt 5 / 2

-- The theorem to prove that the radius of the described cookie is as obtained
theorem cookie_radius_proof :
  ∀ x y : ℝ, cookie_equation x y → cookie_radius (Real.sqrt (45 / 4)) :=
by
  sorry

end cookie_radius_proof_l936_93641


namespace combined_total_years_l936_93617

theorem combined_total_years (A : ℕ) (V : ℕ) (D : ℕ)
(h1 : V = A + 9)
(h2 : V = D - 9)
(h3 : D = 34) : A + V + D = 75 :=
by sorry

end combined_total_years_l936_93617


namespace find_solutions_l936_93632

theorem find_solutions :
  ∀ x y : Real, 
  (3 / 20) + abs (x - (15 / 40)) < (7 / 20) →
  y = 2 * x + 1 →
  (7 / 20) < x ∧ x < (2 / 5) ∧ (17 / 10) ≤ y ∧ y ≤ (11 / 5) :=
by
  intros x y h₁ h₂
  sorry

end find_solutions_l936_93632


namespace angle_ratio_half_l936_93633

theorem angle_ratio_half (a b c : ℝ) (A B C : ℝ) (h1 : a^2 = b * (b + c))
  (h2 : A = 2 * B ∨ A + 2 * B = Real.pi) 
  (h3 : A + B + C = Real.pi) : 
  (B / A = 1 / 2) :=
sorry

end angle_ratio_half_l936_93633


namespace total_area_of_pyramid_faces_l936_93688

theorem total_area_of_pyramid_faces (base_edge lateral_edge : ℝ) (h : base_edge = 8) (k : lateral_edge = 5) : 
  4 * (1 / 2 * base_edge * 3) = 48 :=
by
  -- Base edge of the pyramid
  let b := base_edge
  -- Lateral edge of the pyramid
  let l := lateral_edge
  -- Half of the base
  let half_b := 4
  -- Height of the triangular face using Pythagorean theorem
  let h := 3
  -- Total area of four triangular faces
  have triangular_face_area : 1 / 2 * base_edge * h = 12 := sorry
  have total_area_of_faces : 4 * (1 / 2 * base_edge * h) = 48 := sorry
  exact total_area_of_faces

end total_area_of_pyramid_faces_l936_93688


namespace georgia_total_cost_l936_93629

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l936_93629


namespace erick_total_money_collected_l936_93692

noncomputable def new_lemon_price (old_price increase : ℝ) : ℝ := old_price + increase
noncomputable def new_grape_price (old_price increase : ℝ) : ℝ := old_price + increase / 2

noncomputable def total_money_collected (lemons grapes : ℕ)
                                       (lemon_price grape_price lemon_increase : ℝ) : ℝ :=
  let new_lemon_price := new_lemon_price lemon_price lemon_increase
  let new_grape_price := new_grape_price grape_price lemon_increase
  lemons * new_lemon_price + grapes * new_grape_price

theorem erick_total_money_collected :
  total_money_collected 80 140 8 7 4 = 2220 := 
by
  sorry

end erick_total_money_collected_l936_93692


namespace smallest_prime_with_conditions_l936_93695

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end smallest_prime_with_conditions_l936_93695


namespace gardener_total_expenses_l936_93682

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l936_93682


namespace reflected_line_eq_l936_93621

noncomputable def point_symmetric_reflection :=
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ),
  A = (-1 / 2, 0) ∧ B = (0, 1) ∧ A' = (1 / 2, 0) ∧ 
  ∀ (x y : ℝ), 2 * x + y = 1 ↔
  (y - 1) / (0 - 1) = x / (1 / 2 - 0)

theorem reflected_line_eq :
  point_symmetric_reflection :=
sorry

end reflected_line_eq_l936_93621


namespace ball_in_boxes_l936_93613

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l936_93613


namespace max_value_expression_l936_93615

theorem max_value_expression (a b c d : ℤ) (hb_pos : b > 0)
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a - 2 * b + 3 * c - 4 * d = -7 := 
sorry

end max_value_expression_l936_93615


namespace chess_tournament_time_spent_l936_93600

theorem chess_tournament_time_spent (games : ℕ) (moves_per_game : ℕ)
  (opening_moves : ℕ) (middle_moves : ℕ) (endgame_moves : ℕ)
  (polly_opening_time : ℝ) (peter_opening_time : ℝ)
  (polly_middle_time : ℝ) (peter_middle_time : ℝ)
  (polly_endgame_time : ℝ) (peter_endgame_time : ℝ)
  (total_time_hours : ℝ) :
  games = 4 →
  moves_per_game = 38 →
  opening_moves = 12 →
  middle_moves = 18 →
  endgame_moves = 8 →
  polly_opening_time = 35 →
  peter_opening_time = 45 →
  polly_middle_time = 30 →
  peter_middle_time = 45 →
  polly_endgame_time = 40 →
  peter_endgame_time = 60 →
  total_time_hours = (4 * ((12 * 35 + 18 * 30 + 8 * 40) + (12 * 45 + 18 * 45 + 8 * 60))) / 3600 :=
sorry

end chess_tournament_time_spent_l936_93600


namespace part_I_part_II_l936_93640

-- Define the triangle and sides
structure Triangle :=
  (A B C : ℝ)   -- angles in the triangle
  (a b c : ℝ)   -- sides opposite to respective angles

-- Express given conditions in the problem
def conditions (T: Triangle) : Prop :=
  2 * (1 / (Real.tan T.A) + 1 / (Real.tan T.C)) = 1 / (Real.sin T.A) + 1 / (Real.sin T.C)

-- First theorem statement
theorem part_I (T : Triangle) : conditions T → (T.a + T.c = 2 * T.b) :=
sorry

-- Second theorem statement
theorem part_II (T : Triangle) : conditions T → (T.B ≤ Real.pi / 3) :=
sorry

end part_I_part_II_l936_93640


namespace num_of_valid_numbers_l936_93687

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ (9 * a) % 10 = 4

theorem num_of_valid_numbers : ∃ n, n = 10 :=
by {
  sorry
}

end num_of_valid_numbers_l936_93687


namespace pears_seed_avg_l936_93637

def apple_seed_avg : ℕ := 6
def grape_seed_avg : ℕ := 3
def total_seeds_required : ℕ := 60
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def seeds_short : ℕ := 3
def total_seeds_obtained : ℕ := total_seeds_required - seeds_short

theorem pears_seed_avg :
  (apples_count * apple_seed_avg) + (grapes_count * grape_seed_avg) + (pears_count * P) = total_seeds_obtained → 
  P = 2 :=
by
  sorry

end pears_seed_avg_l936_93637


namespace product_of_two_equal_numbers_l936_93614

-- Definitions and conditions
def arithmetic_mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

-- Theorem stating the product of the two equal numbers
theorem product_of_two_equal_numbers (a b c : ℚ) (x : ℚ) :
  arithmetic_mean [a, b, c, x, x] = 20 → a = 22 → b = 18 → c = 32 → x * x = 196 :=
by
  intros h_mean h_a h_b h_c
  sorry

end product_of_two_equal_numbers_l936_93614


namespace find_starting_number_l936_93693

theorem find_starting_number (x : ℕ) (h1 : (50 + 250) / 2 = 150)
  (h2 : (x + 400) / 2 = 150 + 100) : x = 100 := by
  sorry

end find_starting_number_l936_93693


namespace roots_of_quadratic_eq_l936_93683

noncomputable def r : ℂ := sorry
noncomputable def s : ℂ := sorry

def roots_eq (h : 3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : Prop :=
  (1 / r^3) + (1 / s^3) = 1

theorem roots_of_quadratic_eq (h:3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : roots_eq h :=
sorry

end roots_of_quadratic_eq_l936_93683


namespace math_problem_proof_l936_93602

-- Define the system of equations
structure equations :=
  (x y m : ℝ)
  (eq1 : x + 2*y - 6 = 0)
  (eq2 : x - 2*y + m*x + 5 = 0)

-- Define the problem conditions and prove the required solutions in Lean 4
theorem math_problem_proof :
  -- Part 1: Positive integer solutions for x + 2y - 6 = 0
  (∀ x y : ℕ, x + 2*y = 6 → (x, y) = (2, 2) ∨ (x, y) = (4, 1)) ∧
  -- Part 2: Given x + y = 0, find m
  (∀ x y : ℝ, x + y = 0 → x + 2*y - 6 = 0 → x - 2*y - (13/6)*x + 5 = 0) ∧
  -- Part 3: Fixed solution for x - 2y + mx + 5 = 0
  (∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0) :=
sorry

end math_problem_proof_l936_93602


namespace range_m_l936_93611

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = f (-x)

noncomputable def decreasing_on_non_neg (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_m (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : decreasing_on_non_neg f) :
  ∀ m, f (1 - m) < f m → m < 1 / 2 :=
by
  sorry

end range_m_l936_93611


namespace find_train_length_l936_93669

noncomputable def speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 245.03
noncomputable def time_seconds : ℝ := 30
noncomputable def speed_ms : ℝ := (speed_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := speed_ms * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length

theorem find_train_length : train_length = 129.97 := 
by
  sorry

end find_train_length_l936_93669


namespace max_value_of_a_exists_max_value_of_a_l936_93634

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  a ≤ (Real.sqrt 6 / 3) :=
sorry

theorem exists_max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ∃ a_max: ℝ, a_max = (Real.sqrt 6 / 3) ∧ (∀ a', (a' ≤ a_max)) :=
sorry

end max_value_of_a_exists_max_value_of_a_l936_93634


namespace distance_in_scientific_notation_l936_93677

theorem distance_in_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n = 4 ∧ 38000 = a * 10^n ∧ a = 3.8 :=
by
  sorry

end distance_in_scientific_notation_l936_93677


namespace harry_spends_1920_annually_l936_93667

def geckoCount : Nat := 3
def iguanaCount : Nat := 2
def snakeCount : Nat := 4

def geckoFeedTimesPerMonth : Nat := 2
def iguanaFeedTimesPerMonth : Nat := 3
def snakeFeedTimesPerMonth : Nat := 1 / 2

def geckoFeedCostPerMeal : Nat := 8
def iguanaFeedCostPerMeal : Nat := 12
def snakeFeedCostPerMeal : Nat := 20

def annualCostHarrySpends (geckoCount guCount scCount : Nat) (geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth : Nat) (geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal : Nat) : Nat :=
  let geckoAnnualCost := geckoCount * (geckoFeedTimesPerMonth * 12 * geckoFeedCostPerMeal)
  let iguanaAnnualCost := iguanaCount * (iguanaFeedTimesPerMonth * 12 * iguanaFeedCostPerMeal)
  let snakeAnnualCost := snakeCount * ((12 / (2 : Nat)) * snakeFeedCostPerMeal)
  geckoAnnualCost + iguanaAnnualCost + snakeAnnualCost

theorem harry_spends_1920_annually : annualCostHarrySpends geckoCount iguanaCount snakeCount geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal = 1920 := 
  sorry

end harry_spends_1920_annually_l936_93667


namespace total_trees_planted_l936_93659

/-- A yard is 255 meters long, with a tree at each end and trees planted at intervals of 15 meters. -/
def yard_length : ℤ := 255

def tree_interval : ℤ := 15

def total_trees : ℤ := 18

theorem total_trees_planted (L : ℤ) (d : ℤ) (n : ℤ) : 
  L = yard_length →
  d = tree_interval →
  n = total_trees →
  n = (L / d) + 1 :=
by
  intros hL hd hn
  rw [hL, hd, hn]
  sorry

end total_trees_planted_l936_93659


namespace angle_of_inclination_of_line_l936_93678

-- Definition of the line l
def line_eq (x : ℝ) : ℝ := x + 1

-- Statement of the theorem about the angle of inclination
theorem angle_of_inclination_of_line (x : ℝ) : 
  ∃ (θ : ℝ), θ = 45 ∧ line_eq x = x + 1 := 
sorry

end angle_of_inclination_of_line_l936_93678


namespace math_problem_l936_93684

variables {x y : ℝ}

theorem math_problem (h1 : x + y = 6) (h2 : x * y = 5) :
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y) ^ 2 = 16) ∧ (x ^ 2 + y ^ 2 = 26) :=
by
  sorry

end math_problem_l936_93684


namespace numWaysElectOfficers_l936_93628

-- Definitions and conditions from part (a)
def numMembers : Nat := 30
def numPositions : Nat := 5
def members := ["Alice", "Bob", "Carol", "Dave"]
def allOrNoneCondition (S : List String) : Bool := 
  S.all (members.contains)

-- Function to count the number of ways to choose the officers
def countWays (n : Nat) (k : Nat) (allOrNone : Bool) : Nat :=
if allOrNone then
  -- All four members are positioned
  Nat.factorial k * (n - k)
else
  -- None of the four members are positioned
  let remaining := n - members.length
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem numWaysElectOfficers :
  let casesWithNone := countWays numMembers numPositions false
  let casesWithAll := countWays numMembers numPositions true
  (casesWithNone + casesWithAll) = 6378720 :=
by
  sorry

end numWaysElectOfficers_l936_93628


namespace Jerry_travel_time_l936_93635

theorem Jerry_travel_time
  (speed_j speed_b distance_j distance_b time_j time_b : ℝ)
  (h_speed_j : speed_j = 40)
  (h_speed_b : speed_b = 30)
  (h_distance_b : distance_b = distance_j + 5)
  (h_time_b : time_b = time_j + 1/3)
  (h_distance_j : distance_j = speed_j * time_j)
  (h_distance_b_eq : distance_b = speed_b * time_b) :
  time_j = 1/2 :=
by
  sorry

end Jerry_travel_time_l936_93635


namespace find_point_P_l936_93654

-- Define the function
def f (x : ℝ) := x^4 - 2 * x

-- Define the derivative of the function
def f' (x : ℝ) := 4 * x^3 - 2

theorem find_point_P :
  ∃ (P : ℝ × ℝ), (f' P.1 = 2) ∧ (f P.1 = P.2) ∧ (P = (1, -1)) :=
by
  -- here would go the actual proof
  sorry

end find_point_P_l936_93654


namespace eval_expression_l936_93694

theorem eval_expression (x y z : ℝ) (hx : x = 1/3) (hy : y = 2/3) (hz : z = -9) :
  x^2 * y^3 * z = -8/27 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end eval_expression_l936_93694


namespace geo_seq_b_formula_b_n_sum_T_n_l936_93603

-- Define the sequence a_n 
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else sorry -- Definition based on provided conditions

-- Define the partial sum S_n
def S (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * a (n-1) + 2 -- Given condition S_{n+1} = 4a_n + 2

-- Condition for b_n
def b (n : ℕ) : ℕ :=
  a (n+1) - 2 * a n

-- Definition for c_n
def c (n : ℕ) := (b n) / 3

-- Define the sequence terms for c_n based sequence
def T (n : ℕ) : ℝ :=
  sorry -- Needs explicit definition from given sequence part

-- Proof statements
theorem geo_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n :=
  sorry

theorem formula_b_n : ∀ n : ℕ, b n = 3 * 2^(n-1) :=
  sorry

theorem sum_T_n : ∀ n : ℕ, T n = n / (n + 1) :=
  sorry

end geo_seq_b_formula_b_n_sum_T_n_l936_93603


namespace total_pebbles_count_l936_93644

def white_pebbles : ℕ := 20
def red_pebbles : ℕ := white_pebbles / 2
def blue_pebbles : ℕ := red_pebbles / 3
def green_pebbles : ℕ := blue_pebbles + 5

theorem total_pebbles_count : white_pebbles + red_pebbles + blue_pebbles + green_pebbles = 41 := by
  sorry

end total_pebbles_count_l936_93644


namespace find_functions_l936_93676

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem find_functions (f : ℤ → ℤ) (h : satisfies_equation f) : (∀ x, f x = 2 * x) ∨ (∀ x, f x = 0) :=
sorry

end find_functions_l936_93676


namespace min_value_inequality_l936_93642

theorem min_value_inequality (a b : ℝ) (h : a * b = 1) : 4 * a^2 + 9 * b^2 ≥ 12 :=
by sorry

end min_value_inequality_l936_93642


namespace count_three_digit_numbers_between_l936_93655

theorem count_three_digit_numbers_between 
  (a b : ℕ) 
  (ha : a = 137) 
  (hb : b = 285) : 
  ∃ n, n = (b - a - 1) + 1 := 
sorry

end count_three_digit_numbers_between_l936_93655


namespace integral_negative_of_negative_function_l936_93663

theorem integral_negative_of_negative_function {f : ℝ → ℝ} 
  (hf_cont : Continuous f) 
  (hf_neg : ∀ x, f x < 0) 
  {a b : ℝ} 
  (hab : a < b) 
  : ∫ x in a..b, f x < 0 := 
sorry

end integral_negative_of_negative_function_l936_93663


namespace monotonicity_and_range_of_a_l936_93668

noncomputable def f (x a : ℝ) := Real.log x - a * x - 2

theorem monotonicity_and_range_of_a (a : ℝ) (h : a ≠ 0) :
  ((∀ x > 0, (Real.log x - a * x - 2) < (Real.log (x + 1) - a * (x + 1) - 2)) ↔ (a < 0)) ∧
  ((∃ M, M = Real.log (1/a) - a * (1/a) - 2 ∧ M > a - 4) → 0 < a ∧ a < 1) := sorry

end monotonicity_and_range_of_a_l936_93668


namespace cos_fourth_power_sum_l936_93625

open Real

theorem cos_fourth_power_sum :
  (cos (0 : ℝ))^4 + (cos (π / 6))^4 + (cos (π / 3))^4 + (cos (π / 2))^4 +
  (cos (2 * π / 3))^4 + (cos (5 * π / 6))^4 + (cos π)^4 = 13 / 4 := 
by
  sorry

end cos_fourth_power_sum_l936_93625


namespace range_of_a_for_root_l936_93631

noncomputable def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a * x^2 + 2 * x - 1) = 0

theorem range_of_a_for_root :
  { a : ℝ | has_root_in_interval a } = { a : ℝ | -1 ≤ a } :=
by 
  sorry

end range_of_a_for_root_l936_93631


namespace radii_difference_of_concentric_circles_l936_93604

theorem radii_difference_of_concentric_circles 
  (r : ℝ) 
  (h_area_ratio : (π * (2 * r)^2) / (π * r^2) = 4) : 
  (2 * r) - r = r :=
by
  sorry

end radii_difference_of_concentric_circles_l936_93604


namespace ticket_difference_l936_93686

-- Definitions representing the number of VIP and general admission tickets
def numTickets (V G : Nat) : Prop :=
  V + G = 320

def totalCost (V G : Nat) : Prop :=
  40 * V + 15 * G = 7500

-- Theorem stating that the difference between general admission and VIP tickets is 104
theorem ticket_difference (V G : Nat) (h1 : numTickets V G) (h2 : totalCost V G) : G - V = 104 := by
  sorry

end ticket_difference_l936_93686


namespace transformed_roots_l936_93638

theorem transformed_roots 
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * (-1)^2 + b * (-1) + c = 0)
  (h₃ : a * 2^2 + b * 2 + c = 0) :
  (a * 0^2 + b * 0 + c = 0) ∧ (a * 3^2 + b * 3 + c = 0) :=
by 
  sorry

end transformed_roots_l936_93638


namespace max_cookies_eaten_l936_93651

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem max_cookies_eaten 
  (total_cookies : ℕ)
  (andy_cookies : ℕ)
  (alexa_cookies : ℕ)
  (hx : andy_cookies + alexa_cookies = total_cookies)
  (hp : ∃ p : ℕ, prime p ∧ alexa_cookies = p * andy_cookies)
  (htotal : total_cookies = 30) :
  andy_cookies = 10 :=
  sorry

end max_cookies_eaten_l936_93651


namespace complementary_angles_l936_93650

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l936_93650


namespace maximum_triangle_area_l936_93601

-- Define the maximum area of a triangle given two sides.
theorem maximum_triangle_area (a b : ℝ) (h_a : a = 1984) (h_b : b = 2016) :
  ∃ (max_area : ℝ), max_area = 1998912 :=
by
  sorry

end maximum_triangle_area_l936_93601
