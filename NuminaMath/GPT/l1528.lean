import Mathlib

namespace find_f_of_one_half_l1528_152835

def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := (1 - x ^ 2) / x ^ 2

theorem find_f_of_one_half :
  f (g (1 / 2)) = 15 :=
by
  sorry

end find_f_of_one_half_l1528_152835


namespace graph_of_equation_l1528_152854

theorem graph_of_equation :
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (x + y + 2 = 0 ∨ x+y = 0 ∨ x-y = 0) ∧ 
  ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    (x₁ + y₁ + 2 = 0 ∧ x₁ + y₁ = 0) ∧ 
    (x₂ + y₂ + 2 = 0 ∧ x₂ = -x₂) ∧ 
    (x₃ + y₃ + 2 = 0 ∧ x₃ - y₃ = 0)) := 
sorry

end graph_of_equation_l1528_152854


namespace graduation_messages_total_l1528_152863

/-- Define the number of students in the class -/
def num_students : ℕ := 40

/-- Define the combination formula C(n, 2) for choosing 2 out of n -/
def combination (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Prove that the total number of graduation messages written is 1560 -/
theorem graduation_messages_total : combination num_students = 1560 :=
by
  sorry

end graduation_messages_total_l1528_152863


namespace matrix_power_identity_l1528_152889

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l1528_152889


namespace hexagon_side_equality_l1528_152824

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

def parallel (x y : A) : Prop := ∀ r : ℝ, x = r • y
noncomputable def length_eq (x y : A) : Prop := ∃ r : ℝ, r • x = y

variables (AB DE BC EF CD FA : A)
variables (h1 : parallel AB DE)
variables (h2 : parallel BC EF)
variables (h3 : parallel CD FA)
variables (h4 : length_eq AB DE)

theorem hexagon_side_equality :
  length_eq BC EF ∧ length_eq CD FA :=
by
  sorry

end hexagon_side_equality_l1528_152824


namespace triangle_DEF_all_acute_l1528_152882

theorem triangle_DEF_all_acute
  (α : ℝ)
  (hα : 0 < α ∧ α < 90)
  (DEF : Type)
  (D : DEF) (E : DEF) (F : DEF)
  (angle_DFE : DEF → DEF → DEF → ℝ) 
  (angle_FED : DEF → DEF → DEF → ℝ) 
  (angle_EFD : DEF → DEF → DEF → ℝ)
  (h1 : angle_DFE D F E = 45)
  (h2 : angle_FED F E D = 90 - α / 2)
  (h3 : angle_EFD E D F = 45 + α / 2) :
  (0 < angle_DFE D F E ∧ angle_DFE D F E < 90) ∧ 
  (0 < angle_FED F E D ∧ angle_FED F E D < 90) ∧ 
  (0 < angle_EFD E D F ∧ angle_EFD E D F < 90) := by
  sorry

end triangle_DEF_all_acute_l1528_152882


namespace initial_amount_of_liquid_A_l1528_152896

-- Definitions for liquids A and B and their ratios in the initial and modified mixtures
def initial_ratio_A_over_B : ℚ := 4 / 1
def final_ratio_A_over_B_after_replacement : ℚ := 2 / 3
def mixture_replacement_volume : ℚ := 30

-- Proof of the initial amount of liquid A
theorem initial_amount_of_liquid_A (x : ℚ) (A B : ℚ) (initial_mixture : ℚ) :
  (initial_ratio_A_over_B = 4 / 1) →
  (final_ratio_A_over_B_after_replacement = 2 / 3) →
  (mixture_replacement_volume = 30) →
  (A + B = 5 * x) →
  (A / B = 4 / 1) →
  ((A - 24) / (B - 6 + 30) = 2 / 3) →
  A = 48 :=
by {
  sorry
}

end initial_amount_of_liquid_A_l1528_152896


namespace smallest_y_value_l1528_152870

noncomputable def f (y : ℝ) : ℝ := 3 * y ^ 2 + 27 * y - 90
noncomputable def g (y : ℝ) : ℝ := y * (y + 15)

theorem smallest_y_value (y : ℝ) : (∀ y, f y = g y → y ≠ -9) → false := by
  sorry

end smallest_y_value_l1528_152870


namespace remainder_of_N_eq_4101_l1528_152809

noncomputable def N : ℕ :=
  20 + 3^(3^(3+1) - 13)

theorem remainder_of_N_eq_4101 : N % 10000 = 4101 := by
  sorry

end remainder_of_N_eq_4101_l1528_152809


namespace inequality_system_solution_l1528_152876

theorem inequality_system_solution (a b x : ℝ) 
  (h1 : x - a > 2)
  (h2 : x + 1 < b)
  (h3 : -1 < x)
  (h4 : x < 1) :
  (a + b) ^ 2023 = -1 :=
by 
  sorry

end inequality_system_solution_l1528_152876


namespace geometric_sequence_a5_l1528_152823

theorem geometric_sequence_a5 (α : Type) [LinearOrderedField α] (a : ℕ → α)
  (h1 : ∀ n, a (n + 1) = a n * 2)
  (h2 : ∀ n, a n > 0)
  (h3 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end geometric_sequence_a5_l1528_152823


namespace construct_points_PQ_l1528_152860

-- Given Conditions
variable (a b c : ℝ)
def triangle_ABC_conditions : Prop := 
  let s := (a + b + c) / 2
  s^2 ≥ 2 * a * b

-- Main Statement
theorem construct_points_PQ (a b c : ℝ) (P Q : ℝ) 
(h1 : triangle_ABC_conditions a b c) :
  let s := (a + b + c) / 2
  let x := (s + Real.sqrt (s^2 - 2 * a * b)) / 2
  let y := (s - Real.sqrt (s^2 - 2 * a * b)) / 2
  x + y = s ∧ x * y = (a * b) / 2 :=
by
  sorry

end construct_points_PQ_l1528_152860


namespace fraction_r_over_b_l1528_152874

-- Definition of the conditions
def initial_expression (k : ℝ) : ℝ := 8 * k^2 - 12 * k + 20

-- Proposition statement
theorem fraction_r_over_b : ∃ a b r : ℝ, 
  (∀ k : ℝ, initial_expression k = a * (k + b)^2 + r) ∧ 
  r / b = -47.33 :=
sorry

end fraction_r_over_b_l1528_152874


namespace game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l1528_152897

-- Definitions and conditions for the problem
def num_girls : ℕ := 1994
def tokens (n : ℕ) := n

-- Main theorem statements
theorem game_terminates_if_n_lt_1994 (n : ℕ) (h : n < num_girls) :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (∀ j : ℕ, 1 ≤ j ∧ j ≤ num_girls → (tokens n % num_girls) ≤ 1) :=
by
  sorry

theorem game_does_not_terminate_if_n_eq_1994 :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (tokens 1994 % num_girls = 0) :=
by
  sorry

end game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l1528_152897


namespace david_marks_in_english_l1528_152807

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l1528_152807


namespace total_inheritance_money_l1528_152830

-- Defining the conditions
def number_of_inheritors : ℕ := 5
def amount_per_person : ℕ := 105500

-- The proof problem
theorem total_inheritance_money :
  number_of_inheritors * amount_per_person = 527500 :=
by sorry

end total_inheritance_money_l1528_152830


namespace sqrt_eq_pm_four_l1528_152806

theorem sqrt_eq_pm_four (a : ℤ) : (a * a = 16) ↔ (a = 4 ∨ a = -4) :=
by sorry

end sqrt_eq_pm_four_l1528_152806


namespace quadratic_root_is_zero_then_m_neg_one_l1528_152888

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l1528_152888


namespace evaluate_fractions_l1528_152839

-- Define the fractions
def frac1 := 7 / 12
def frac2 := 8 / 15
def frac3 := 2 / 5

-- Prove that the sum and difference is as specified
theorem evaluate_fractions :
  frac1 + frac2 - frac3 = 43 / 60 :=
by
  sorry

end evaluate_fractions_l1528_152839


namespace slope_of_PQ_l1528_152850

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

theorem slope_of_PQ :
  ∃ P Q : ℝ × ℝ,
    P = (0, 0) ∧ Q = (1, 8 / 3) ∧
    (∃ m : ℝ,
      m = 2 * Real.cos 0 ∧
      m = Real.sqrt 1 + 1 / Real.sqrt 1) ∧
    (Q.snd - P.snd) / (Q.fst - P.fst) = 8 / 3 :=
by
  sorry

end slope_of_PQ_l1528_152850


namespace frank_reads_pages_per_day_l1528_152884

theorem frank_reads_pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) : pages_per_book / days_per_book = 83 :=
by {
  sorry
}

end frank_reads_pages_per_day_l1528_152884


namespace chess_tournament_games_l1528_152836

def players : ℕ := 12

def games_per_pair : ℕ := 2

theorem chess_tournament_games (n : ℕ) (h : n = players) : 
  (n * (n - 1) * games_per_pair) = 264 := by
  sorry

end chess_tournament_games_l1528_152836


namespace bruce_total_cost_l1528_152813

def cost_of_grapes : ℕ := 8 * 70
def cost_of_mangoes : ℕ := 11 * 55
def cost_of_oranges : ℕ := 5 * 45
def cost_of_apples : ℕ := 3 * 90
def cost_of_cherries : ℕ := (45 / 10) * 120  -- use rational division and then multiplication

def total_cost : ℕ :=
  cost_of_grapes + cost_of_mangoes + cost_of_oranges + cost_of_apples + cost_of_cherries

theorem bruce_total_cost : total_cost = 2200 := by
  sorry

end bruce_total_cost_l1528_152813


namespace smallest_k_for_min_period_15_l1528_152846

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l1528_152846


namespace A_work_days_l1528_152838

theorem A_work_days (x : ℝ) (H : 3 * (1 / x + 1 / 20) = 0.35) : x = 15 := 
by
  sorry

end A_work_days_l1528_152838


namespace Jennifer_more_boxes_l1528_152845

-- Definitions based on conditions
def Kim_boxes : ℕ := 54
def Jennifer_boxes : ℕ := 71

-- Proof statement (no actual proof needed, just the statement)
theorem Jennifer_more_boxes : Jennifer_boxes - Kim_boxes = 17 := by
  sorry

end Jennifer_more_boxes_l1528_152845


namespace greatest_product_l1528_152898

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l1528_152898


namespace arithmetic_geometric_sequence_min_value_l1528_152831

theorem arithmetic_geometric_sequence_min_value (x y a b c d : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (arithmetic_seq : a = (x + y) / 2) (geometric_seq : c * d = x * y) :
  ( (a + b) ^ 2 ) / (c * d) ≥ 4 := 
by
  sorry

end arithmetic_geometric_sequence_min_value_l1528_152831


namespace number_of_paths_l1528_152808

-- Define the conditions of the problem
def grid_width : ℕ := 7
def grid_height : ℕ := 6
def diagonal_steps : ℕ := 2

-- Define the main proof statement
theorem number_of_paths (width height diag : ℕ) 
  (Nhyp : width = grid_width ∧ height = grid_height ∧ diag = diagonal_steps) : 
  ∃ (paths : ℕ), paths = 6930 := 
sorry

end number_of_paths_l1528_152808


namespace train_passes_jogger_in_39_seconds_l1528_152864

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_head_start : ℝ := 270
noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45

noncomputable def to_meters_per_second (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def jogger_speed_mps : ℝ :=
  to_meters_per_second jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  to_meters_per_second train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance : ℝ :=
  jogger_head_start + train_length

noncomputable def time_to_pass_jogger : ℝ :=
  total_distance / relative_speed_mps

theorem train_passes_jogger_in_39_seconds :
  time_to_pass_jogger = 39 := by
  sorry

end train_passes_jogger_in_39_seconds_l1528_152864


namespace find_selling_price_l1528_152802

-- Define the cost price of the article
def cost_price : ℝ := 47

-- Define the profit when the selling price is Rs. 54
def profit : ℝ := 54 - cost_price

-- Assume that the profit is the same as the loss
axiom profit_equals_loss : profit = 7

-- Define the selling price that yields the same loss as the profit
def selling_price_loss : ℝ := cost_price - profit

-- Now state the theorem to prove that the selling price for loss is Rs. 40
theorem find_selling_price : selling_price_loss = 40 :=
sorry

end find_selling_price_l1528_152802


namespace macey_saving_weeks_l1528_152849

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l1528_152849


namespace smallest_non_factor_product_of_100_l1528_152804

/-- Let a and b be distinct positive integers that are factors of 100. 
    The smallest value of their product which is not a factor of 100 is 8. -/
theorem smallest_non_factor_product_of_100 (a b : ℕ) (hab : a ≠ b) (ha : a ∣ 100) (hb : b ∣ 100) (hprod : ¬ (a * b ∣ 100)) : a * b = 8 :=
sorry

end smallest_non_factor_product_of_100_l1528_152804


namespace quadrilateral_perpendicular_diagonals_l1528_152851

theorem quadrilateral_perpendicular_diagonals
  (AB BC CD DA : ℝ)
  (m n : ℝ)
  (hAB : AB = 6)
  (hBC : BC = m)
  (hCD : CD = 8)
  (hDA : DA = n)
  (h_diagonals_perpendicular : true)
  : m^2 + n^2 = 100 := 
by
  sorry

end quadrilateral_perpendicular_diagonals_l1528_152851


namespace garden_remaining_area_is_250_l1528_152812

open Nat

-- Define the dimensions of the rectangular garden
def garden_length : ℕ := 18
def garden_width : ℕ := 15
-- Define the dimensions of the square cutouts
def cutout1_side : ℕ := 4
def cutout2_side : ℕ := 2

-- Calculate areas based on the definitions
def garden_area : ℕ := garden_length * garden_width
def cutout1_area : ℕ := cutout1_side * cutout1_side
def cutout2_area : ℕ := cutout2_side * cutout2_side

-- Calculate total area excluding the cutouts
def remaining_area : ℕ := garden_area - cutout1_area - cutout2_area

-- Prove that the remaining area is 250 square feet
theorem garden_remaining_area_is_250 : remaining_area = 250 :=
by
  sorry

end garden_remaining_area_is_250_l1528_152812


namespace banana_distinct_arrangements_l1528_152857

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l1528_152857


namespace polynomial_form_l1528_152887

theorem polynomial_form (P : ℝ → ℝ) (h₁ : P 0 = 0) (h₂ : ∀ x, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_form_l1528_152887


namespace problem_I_l1528_152843

theorem problem_I (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : 
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 := 
by
  sorry

end problem_I_l1528_152843


namespace find_x_l1528_152866

theorem find_x (x : ℝ) 
  (h: 3 * x + 6 * x + 2 * x + x = 360) : 
  x = 30 := 
sorry

end find_x_l1528_152866


namespace monotone_decreasing_sequence_monotone_increasing_sequence_l1528_152861

theorem monotone_decreasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) < a n) ↔ c < 0 :=
by sorry

theorem monotone_increasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) > a n) ↔ c > 1/4 :=
by sorry

end monotone_decreasing_sequence_monotone_increasing_sequence_l1528_152861


namespace find_f_l1528_152853

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (h : ∀ x, x ≠ -1 → f ((1-x) / (1+x)) = (1 - x^2) / (1 + x^2)) 
               (hx : x ≠ -1) :
  f x = 2 * x / (1 + x^2) :=
sorry

end find_f_l1528_152853


namespace part1_solution_part2_solution_l1528_152873

-- Conditions
variables (x y : ℕ) -- Let x be the number of parcels each person sorts manually per hour,
                     -- y be the number of machines needed

def machine_efficiency : ℕ := 20 * x
def time_machines (parcels : ℕ) (machines : ℕ) : ℕ := parcels / (machines * machine_efficiency x)
def time_people (parcels : ℕ) (people : ℕ) : ℕ := parcels / (people * x)
def parcels_per_day : ℕ := 100000

-- Problem 1: Find x
axiom problem1 : (time_people 6000 20) - (time_machines 6000 5) = 4

-- Problem 2: Find y to sort 100000 parcels in a day with machines working 16 hours/day
axiom problem2 : 16 * machine_efficiency x * y ≥ parcels_per_day

-- Correct answers:
theorem part1_solution : x = 60 := by sorry
theorem part2_solution : y = 6 := by sorry

end part1_solution_part2_solution_l1528_152873


namespace periodic_function_implies_rational_ratio_l1528_152801

noncomputable def g (i : ℕ) (a ω θ x : ℝ) : ℝ := 
  a * Real.sin (ω * x + θ)

theorem periodic_function_implies_rational_ratio 
  (a1 a2 ω1 ω2 θ1 θ2 : ℝ) (h1 : a1 * ω1 ≠ 0) (h2 : a2 * ω2 ≠ 0)
  (h3 : |ω1| ≠ |ω2|) 
  (hf_periodic : ∃ T : ℝ, ∀ x : ℝ, g 1 a1 ω1 θ1 (x + T) + g 2 a2 ω2 θ2 (x + T) = g 1 a1 ω1 θ1 x + g 2 a2 ω2 θ2 x) :
  ∃ m n : ℤ, n ≠ 0 ∧ ω1 / ω2 = m / n :=
sorry

end periodic_function_implies_rational_ratio_l1528_152801


namespace value_is_85_over_3_l1528_152834

theorem value_is_85_over_3 (a b : ℚ)  (h1 : 3 * a + 6 * b = 48) (h2 : 8 * a + 4 * b = 84) : 2 * a + 3 * b = 85 / 3 := 
by {
  -- Proof will go here
  sorry
}

end value_is_85_over_3_l1528_152834


namespace ratio_of_riding_to_total_l1528_152880

-- Define the primary conditions from the problem
variables (H R W : ℕ)
variables (legs_on_ground : ℕ := 50)
variables (total_owners : ℕ := 10)
variables (legs_per_horse : ℕ := 4)
variables (legs_per_owner : ℕ := 2)

-- Express the conditions
def conditions : Prop :=
  (legs_on_ground = 6 * W) ∧
  (total_owners = H) ∧
  (H = R + W) ∧
  (H = 10)

-- Define the theorem with the given conditions and prove the required ratio
theorem ratio_of_riding_to_total (H R W : ℕ) (h : conditions H R W) : R / 10 = 1 / 5 := by
  sorry

end ratio_of_riding_to_total_l1528_152880


namespace sin_alpha_value_l1528_152868

theorem sin_alpha_value (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_value_l1528_152868


namespace sum_of_coordinates_of_D_l1528_152821

-- Definition of points M, C and D
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨4, 7⟩
def C : Point := ⟨6, 2⟩

-- Conditions that M is the midpoint of segment CD
def isMidpoint (M C D : Point) : Prop :=
  ((C.x + D.x) / 2 = M.x) ∧
  ((C.y + D.y) / 2 = M.y)

-- Definition for the sum of the coordinates of a point
def sumOfCoordinates (P : Point) : ℝ :=
  P.x + P.y

-- The main theorem stating the sum of the coordinates of D is 14 given the conditions
theorem sum_of_coordinates_of_D :
  ∃ D : Point, isMidpoint M C D ∧ sumOfCoordinates D = 14 := 
sorry

end sum_of_coordinates_of_D_l1528_152821


namespace sum_ratio_l1528_152841

variable {α : Type _} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0       => a₁
| (n + 1) => (geometric_sequence a₁ q n) * q

noncomputable def sum_geometric (a₁ q : α) (n : ℕ) : α :=
  if q = 1 then a₁ * (n + 1)
  else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_ratio {a₁ q : α} (h : 8 * (geometric_sequence a₁ q 1) + (geometric_sequence a₁ q 4) = 0) :
  (sum_geometric a₁ q 4) / (sum_geometric a₁ q 1) = -11 :=
sorry

end sum_ratio_l1528_152841


namespace bob_more_than_alice_l1528_152817

-- Definitions for conditions
def initial_investment_alice : ℕ := 10000
def initial_investment_bob : ℕ := 10000
def multiple_alice : ℕ := 3
def multiple_bob : ℕ := 7

-- Derived conditions based on the investment multiples
def final_amount_alice : ℕ := initial_investment_alice * multiple_alice
def final_amount_bob : ℕ := initial_investment_bob * multiple_bob

-- Statement of the problem
theorem bob_more_than_alice : final_amount_bob - final_amount_alice = 40000 :=
by
  -- Proof to be filled in
  sorry

end bob_more_than_alice_l1528_152817


namespace jane_doe_gift_l1528_152871

theorem jane_doe_gift (G : ℝ) (h1 : 0.25 * G + 0.1125 * (0.75 * G) = 15000) : G = 41379 := 
sorry

end jane_doe_gift_l1528_152871


namespace part_one_part_two_l1528_152814

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem part_one (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n > 1) : f m >= 0 ∨ f n >= 0 :=
sorry

theorem part_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hf : f a = f b) : a + b < 4 / 3 :=
sorry

end part_one_part_two_l1528_152814


namespace possible_values_of_AC_l1528_152869

theorem possible_values_of_AC (AB CD AC : ℝ) (m n : ℝ) (h1 : AB = 16) (h2 : CD = 4)
  (h3 : Set.Ioo m n = {x : ℝ | 4 < x ∧ x < 16}) : m + n = 20 :=
by
  sorry

end possible_values_of_AC_l1528_152869


namespace domain_of_f_2x_minus_1_l1528_152893

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → (f x ≠ 0)) →
  (∀ y, 0 ≤ y ∧ y ≤ 1 ↔ exists x, (2 * x - 1 = y) ∧ (0 ≤ x ∧ x ≤ 1)) :=
by
  sorry

end domain_of_f_2x_minus_1_l1528_152893


namespace dual_colored_numbers_l1528_152828

theorem dual_colored_numbers (table : Matrix (Fin 10) (Fin 20) ℕ)
  (distinct_numbers : ∀ (i j k l : Fin 10) (m n : Fin 20), 
    (i ≠ k ∨ m ≠ n) → table i m ≠ table k n)
  (row_red : ∀ (i : Fin 10), ∃ r₁ r₂ : Fin 20, r₁ ≠ r₂ ∧ 
    (∀ (j : Fin 20), table i j ≤ table i r₁ ∨ table i j ≤ table i r₂))
  (col_blue : ∀ (j : Fin 20), ∃ b₁ b₂ : Fin 10, b₁ ≠ b₂ ∧ 
    (∀ (i : Fin 10), table i j ≤ table b₁ j ∨ table i j ≤ table b₂ j)) : 
  ∃ i₁ i₂ i₃ : Fin 10, ∃ j₁ j₂ j₃ : Fin 20, 
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧ 
    ((table i₁ j₁ ≤ table i₁ j₂ ∨ table i₁ j₁ ≤ table i₃ j₂) ∧ 
     (table i₂ j₂ ≤ table i₂ j₁ ∨ table i₂ j₂ ≤ table i₃ j₁) ∧ 
     (table i₃ j₃ ≤ table i₃ j₁ ∨ table i₃ j₃ ≤ table i₂ j₁)) := 
  sorry

end dual_colored_numbers_l1528_152828


namespace students_after_joining_l1528_152818

theorem students_after_joining (N : ℕ) (T : ℕ)
  (h1 : T = 48 * N)
  (h2 : 120 * 32 / (N + 120) + (T / (N + 120)) = 44)
  : N + 120 = 480 :=
by
  sorry

end students_after_joining_l1528_152818


namespace square_area_l1528_152855

theorem square_area (x : ℝ) (G H : ℝ) (hyp_1 : 0 ≤ G) (hyp_2 : G ≤ x) (hyp_3 : 0 ≤ H) (hyp_4 : H ≤ x) (AG : ℝ) (GH : ℝ) (HD : ℝ)
  (hyp_5 : AG = 20) (hyp_6 : GH = 20) (hyp_7 : HD = 20) (hyp_8 : x = 20 * Real.sqrt 2) :
  x^2 = 800 :=
by
  sorry

end square_area_l1528_152855


namespace brianna_fraction_left_l1528_152856

theorem brianna_fraction_left (m n c : ℕ) (h : (1 : ℚ) / 4 * m = 1 / 2 * n * c) : 
  (m - (n * c) - (1 / 10 * m)) / m = 2 / 5 :=
by
  sorry

end brianna_fraction_left_l1528_152856


namespace correct_operation_l1528_152805

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l1528_152805


namespace find_w_l1528_152811

variable (x y z w : ℝ)

theorem find_w (h : (x + y + z) / 3 = (y + z + w) / 3 + 10) : w = x - 30 := by 
  sorry

end find_w_l1528_152811


namespace sum_of_first_9_terms_l1528_152848

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- a_n is the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of first n terms of the arithmetic sequence
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Hypotheses
axiom h1 : 2 * a 8 = 6 + a 11
axiom h2 : arithmetic_seq a
axiom h3 : sum_seq S a

-- The theorem we want to prove
theorem sum_of_first_9_terms : S 9 = 54 :=
sorry

end sum_of_first_9_terms_l1528_152848


namespace largest_class_students_l1528_152819

theorem largest_class_students (x : ℕ) (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 105) : x = 25 :=
by {
  sorry
}

end largest_class_students_l1528_152819


namespace katherine_has_5_bananas_l1528_152894

theorem katherine_has_5_bananas
  (apples : ℕ) (pears : ℕ) (bananas : ℕ) (total_fruits : ℕ)
  (h1 : apples = 4)
  (h2 : pears = 3 * apples)
  (h3 : total_fruits = apples + pears + bananas)
  (h4 : total_fruits = 21) :
  bananas = 5 :=
by
  sorry

end katherine_has_5_bananas_l1528_152894


namespace quadratic_root_value_l1528_152879

theorem quadratic_root_value (a b : ℤ) (h : 2 * a - b = -3) : 6 * a - 3 * b + 6 = -3 :=
by 
  sorry

end quadratic_root_value_l1528_152879


namespace min_value_l1528_152816

theorem min_value (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 2) : 
  (∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ (1/3 * x^3 + y^2 + z = 13/12)) :=
sorry

end min_value_l1528_152816


namespace length_of_second_race_l1528_152867

theorem length_of_second_race :
  ∀ (V_A V_B V_C T T' L : ℝ),
  (V_A * T = 200) →
  (V_B * T = 180) →
  (V_C * T = 162) →
  (V_B * T' = L) →
  (V_C * T' = L - 60) →
  (L = 600) :=
by
  intros V_A V_B V_C T T' L h1 h2 h3 h4 h5
  sorry

end length_of_second_race_l1528_152867


namespace probability_three_different_suits_l1528_152815

noncomputable def pinochle_deck := 48
noncomputable def total_cards := 48
noncomputable def different_suits_probability := (36 / 47) * (23 / 46)

theorem probability_three_different_suits :
  different_suits_probability = 414 / 1081 :=
sorry

end probability_three_different_suits_l1528_152815


namespace loss_per_metre_l1528_152891

def total_metres : ℕ := 500
def selling_price : ℕ := 18000
def cost_price_per_metre : ℕ := 41

theorem loss_per_metre :
  (cost_price_per_metre * total_metres - selling_price) / total_metres = 5 :=
by sorry

end loss_per_metre_l1528_152891


namespace maximize_sum_of_arithmetic_seq_l1528_152844

theorem maximize_sum_of_arithmetic_seq (a d : ℤ) (n : ℤ) : d < 0 → a^2 = (a + 10 * d)^2 → n = 5 ∨ n = 6 :=
by
  intro h_d_neg h_a1_eq_a11
  have h_a1_5d_neg : a + 5 * d = 0 := sorry
  have h_sum_max : n = 5 ∨ n = 6 := sorry
  exact h_sum_max

end maximize_sum_of_arithmetic_seq_l1528_152844


namespace ella_age_l1528_152892

theorem ella_age (s t e : ℕ) (h1 : s + t + e = 36) (h2 : e - 5 = s) (h3 : t + 4 = (3 * (s + 4)) / 4) : e = 15 := by
  sorry

end ella_age_l1528_152892


namespace acute_angle_at_3_16_l1528_152810

def angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60) * 360
  let hour_angle := (hour % 12) * 30 + (minute / 60) * 30
  |hour_angle - minute_angle|

theorem acute_angle_at_3_16 : angle_between_clock_hands 3 16 = 2 := 
sorry

end acute_angle_at_3_16_l1528_152810


namespace distribution_schemes_36_l1528_152837

def num_distribution_schemes (total_students english_excellent computer_skills : ℕ) : ℕ :=
  if total_students = 8 ∧ english_excellent = 2 ∧ computer_skills = 3 then 36 else 0

theorem distribution_schemes_36 :
  num_distribution_schemes 8 2 3 = 36 :=
by
 sorry

end distribution_schemes_36_l1528_152837


namespace proposition_false_at_9_l1528_152822

theorem proposition_false_at_9 (P : ℕ → Prop) 
  (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1))
  (hne10 : ¬ P 10) : ¬ P 9 :=
by
  intro hp9
  have hp10 : P 10 := h _ (by norm_num) hp9
  contradiction

end proposition_false_at_9_l1528_152822


namespace fruits_given_away_l1528_152833

-- Definitions based on the conditions
def initial_pears := 10
def initial_oranges := 20
def initial_apples := 2 * initial_pears
def initial_fruits := initial_pears + initial_oranges + initial_apples
def fruits_left := 44

-- Theorem to prove the total number of fruits given to her sister
theorem fruits_given_away : initial_fruits - fruits_left = 6 := by
  sorry

end fruits_given_away_l1528_152833


namespace power_neg_two_inverse_l1528_152803

theorem power_neg_two_inverse : (-2 : ℤ) ^ (-2 : ℤ) = (1 : ℚ) / (4 : ℚ) := by
  -- Condition: a^{-n} = 1 / a^n for any non-zero number a and any integer n
  have h: ∀ (a : ℚ) (n : ℤ), a ≠ 0 → a ^ (-n) = 1 / a ^ n := sorry
  -- Proof goes here
  sorry

end power_neg_two_inverse_l1528_152803


namespace unit_digit_product_zero_l1528_152842

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_product_zero :
  let a := 785846
  let b := 1086432
  let c := 4582735
  let d := 9783284
  let e := 5167953
  let f := 3821759
  let g := 7594683
  unit_digit (a * b * c * d * e * f * g) = 0 := 
by {
  sorry
}

end unit_digit_product_zero_l1528_152842


namespace problem_solution_l1528_152826

noncomputable def time_min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * (Real.cos α) / (2 * c * (1 - Real.sin α))

noncomputable def min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * Real.sqrt ((1 - (Real.sin α)) / 2)

theorem problem_solution (α : ℝ) (c : ℝ) (a : ℝ) 
  (α_30 : α = Real.pi / 6) (c_50 : c = 50) (a_50sqrt3 : a = 50 * Real.sqrt 3) :
  (time_min_distance c α a = 1.5) ∧ (min_distance c α a = 25 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l1528_152826


namespace grade_distribution_sum_l1528_152820

theorem grade_distribution_sum (a b c d : ℝ) (ha : a = 0.6) (hb : b = 0.25) (hc : c = 0.1) (hd : d = 0.05) :
  a + b + c + d = 1.0 :=
by
  -- Introduce the hypothesis
  rw [ha, hb, hc, hd]
  -- Now the goal simplifies to: 0.6 + 0.25 + 0.1 + 0.05 = 1.0
  sorry

end grade_distribution_sum_l1528_152820


namespace find_workers_l1528_152890

def total_workers := 20
def male_work_days := 2
def female_work_days := 3

theorem find_workers (X Y : ℕ) 
  (h1 : X + Y = total_workers)
  (h2 : X / male_work_days + Y / female_work_days = 1) : 
  X = 12 ∧ Y = 8 :=
sorry

end find_workers_l1528_152890


namespace quadratic_inequality_solution_l1528_152832

variable (a x : ℝ)

theorem quadratic_inequality_solution (h : 0 < a ∧ a < 1) : (x - a) * (x - (1 / a)) > 0 ↔ (x < a ∨ x > 1 / a) :=
sorry

end quadratic_inequality_solution_l1528_152832


namespace largest_vertex_sum_l1528_152899

def parabola_vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := T
  let y_vertex := a * T^2 - 2 * a * T^2
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (hA : 0 = a * 0^2 + 0 * 0 + 0)
  (hB : 0 = a * (2 * T)^2 + (2 * T) * (2 * -T))
  (hC : 36 = a * (2 * T + 1)^2 + (2 * T - 2 * T * (2 * T + 1)))
  : parabola_vertex_sum a T hT ≤ -14 :=
sorry

end largest_vertex_sum_l1528_152899


namespace sum_mod_13_l1528_152872

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l1528_152872


namespace product_of_largest_and_second_largest_l1528_152883

theorem product_of_largest_and_second_largest (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  (max (max a b) c * (max (min a (max b c)) (min b (max a c)))) = 132 :=
by
  sorry

end product_of_largest_and_second_largest_l1528_152883


namespace find_n_l1528_152878

variable (a b c n : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)

theorem find_n (h1 : (a + b) / a = 3)
  (h2 : (b + c) / b = 4)
  (h3 : (c + a) / c = n) :
  n = 7 / 6 := 
sorry

end find_n_l1528_152878


namespace cos_of_angle_B_l1528_152865

theorem cos_of_angle_B (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : 6 * Real.sin A = 4 * Real.sin B) (h3 : 4 * Real.sin B = 3 * Real.sin C) : 
  Real.cos B = Real.sqrt 7 / 4 :=
by
  sorry

end cos_of_angle_B_l1528_152865


namespace keiko_speed_l1528_152881

theorem keiko_speed (a b s : ℝ) 
  (width : ℝ := 8) 
  (radius_inner := b) 
  (radius_outer := b + width)
  (time_difference := 48) 
  (L_inner := 2 * a + 2 * Real.pi * radius_inner)
  (L_outer := 2 * a + 2 * Real.pi * radius_outer) :
  (L_outer / s = L_inner / s + time_difference) → 
  s = Real.pi / 3 :=
by 
  sorry

end keiko_speed_l1528_152881


namespace arithmetic_square_root_of_4_l1528_152862

theorem arithmetic_square_root_of_4 : ∃ y : ℝ, y^2 = 4 ∧ y = 2 := 
  sorry

end arithmetic_square_root_of_4_l1528_152862


namespace interest_rate_correct_l1528_152827

-- Definitions based on the problem conditions
def P : ℝ := 7000 -- Principal investment amount
def A : ℝ := 8470 -- Future value of the investment
def n : ℕ := 1 -- Number of times interest is compounded per year
def t : ℕ := 2 -- Number of years

-- The interest rate r to be proven
def r : ℝ := 0.1 -- Annual interest rate

-- Statement of the problem that needs to be proven in Lean
theorem interest_rate_correct :
  A = P * (1 + r / n)^(n * t) :=
by
  sorry

end interest_rate_correct_l1528_152827


namespace problem_proof_l1528_152885

theorem problem_proof (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) : 3 * a^2 * b + 3 * a * b^2 = 18 := 
by
  sorry

end problem_proof_l1528_152885


namespace cart_total_books_l1528_152800

theorem cart_total_books (fiction non_fiction autobiographies picture: ℕ) 
  (h1: fiction = 5)
  (h2: non_fiction = fiction + 4)
  (h3: autobiographies = 2 * fiction)
  (h4: picture = 11)
  : fiction + non_fiction + autobiographies + picture = 35 := by
  -- Proof is omitted
  sorry

end cart_total_books_l1528_152800


namespace power_function_decreasing_m_l1528_152875

theorem power_function_decreasing_m :
  ∀ (m : ℝ), (m^2 - 5*m - 5) * (2*m + 1) < 0 → m = -1 :=
by
  sorry

end power_function_decreasing_m_l1528_152875


namespace remainder_when_squared_mod_seven_l1528_152877

theorem remainder_when_squared_mod_seven
  (x y : ℤ) (k m : ℤ)
  (hx : x = 52 * k + 19)
  (hy : 3 * y = 7 * m + 5) :
  ((x + 2 * y)^2 % 7) = 1 := by
  sorry

end remainder_when_squared_mod_seven_l1528_152877


namespace olivia_race_time_l1528_152852

theorem olivia_race_time (total_time : ℕ) (time_difference : ℕ) (olivia_time : ℕ)
  (h1 : total_time = 112) (h2 : time_difference = 4) (h3 : olivia_time + (olivia_time - time_difference) = total_time) :
  olivia_time = 58 :=
by
  sorry

end olivia_race_time_l1528_152852


namespace problem_mod_l1528_152886

theorem problem_mod (a b c d : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) (h4 : d = 2014) :
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end problem_mod_l1528_152886


namespace graph_of_equation_l1528_152840

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_equation_l1528_152840


namespace lily_account_balance_l1528_152825

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l1528_152825


namespace simplify_expression_find_value_a_m_2n_l1528_152829

-- Proof Problem 1
theorem simplify_expression : ( (-2 : ℤ) * x )^3 * x^2 + ( (3 : ℤ) * x^4 )^2 / x^3 = x^5 := by
  sorry

-- Proof Problem 2
theorem find_value_a_m_2n (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2*n) = 18 := by
  sorry

end simplify_expression_find_value_a_m_2n_l1528_152829


namespace circle_radius_l1528_152858

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 180 * π) : r = 10 := 
by
  sorry

end circle_radius_l1528_152858


namespace area_of_circle_l1528_152895

-- Given condition as a Lean definition
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 + 9 * x - 12 * y - 27 = 0

-- Theorem stating the goal
theorem area_of_circle : ∀ (x y : ℝ), circle_eq x y → ∃ r : ℝ, r = 15.25 ∧ ∃ a : ℝ, a = π * r := 
sorry

end area_of_circle_l1528_152895


namespace perimeter_of_ABCD_is_35_2_l1528_152859

-- Definitions of geometrical properties and distances
variable (AB BC DC : ℝ)
variable (AB_perp_BC : ∃P, is_perpendicular AB BC)
variable (DC_parallel_AB : ∃Q, is_parallel DC AB)
variable (AB_length : AB = 7)
variable (BC_length : BC = 10)
variable (DC_length : DC = 6)

-- Target statement to be proved
theorem perimeter_of_ABCD_is_35_2
  (h1 : AB_perp_BC)
  (h2 : DC_parallel_AB)
  (h3 : AB_length)
  (h4 : BC_length)
  (h5 : DC_length) :
  ∃ P : ℝ, P = 35.2 :=
sorry

end perimeter_of_ABCD_is_35_2_l1528_152859


namespace third_number_lcm_l1528_152847

theorem third_number_lcm (n : ℕ) :
  n ∣ 360 ∧ lcm (lcm 24 36) n = 360 →
  n = 5 :=
by sorry

end third_number_lcm_l1528_152847
