import Mathlib

namespace daily_sacks_per_section_l363_36314

theorem daily_sacks_per_section (harvests sections : ℕ) (h_harvests : harvests = 360) (h_sections : sections = 8) : harvests / sections = 45 := by
  sorry

end daily_sacks_per_section_l363_36314


namespace intersection_correct_l363_36392

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end intersection_correct_l363_36392


namespace find_k_l363_36339

noncomputable def f (k : ℤ) (x : ℝ) := (k^2 + k - 1) * x^(k^2 - 3 * k)

-- The conditions in the problem
variables (k : ℤ) (x : ℝ)
axiom sym_y_axis : ∀ (x : ℝ), f k (-x) = f k x
axiom decreasing_on_positive : ∀ x1 x2, 0 < x1 → x1 < x2 → f k x1 > f k x2

-- The proof problem statement
theorem find_k : k = 1 :=
sorry

end find_k_l363_36339


namespace compute_cd_l363_36353

variable (c d : ℝ)

theorem compute_cd (h1 : c + d = 10) (h2 : c^3 + d^3 = 370) : c * d = 21 := by
  -- Proof would go here
  sorry

end compute_cd_l363_36353


namespace change_received_is_zero_l363_36363

noncomputable def combined_money : ℝ := 10 + 8
noncomputable def cost_chicken_wings : ℝ := 6
noncomputable def cost_chicken_salad : ℝ := 4
noncomputable def cost_cheeseburgers : ℝ := 2 * 3.50
noncomputable def cost_fries : ℝ := 2
noncomputable def cost_sodas : ℝ := 2 * 1.00
noncomputable def total_cost_before_discount : ℝ := cost_chicken_wings + cost_chicken_salad + cost_cheeseburgers + cost_fries + cost_sodas
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discounted_total : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * tax_rate
noncomputable def total_cost_after_tax : ℝ := discounted_total + tax_amount

theorem change_received_is_zero : combined_money < total_cost_after_tax → 0 = combined_money - total_cost_after_tax + combined_money := by
  intros h
  sorry

end change_received_is_zero_l363_36363


namespace negation_exists_cube_positive_l363_36368

theorem negation_exists_cube_positive :
  ¬ (∃ x : ℝ, x^3 > 0) ↔ ∀ x : ℝ, x^3 ≤ 0 := by
  sorry

end negation_exists_cube_positive_l363_36368


namespace sons_ages_l363_36366

theorem sons_ages (x y : ℕ) (h1 : 2 * x = x + y + 18) (h2 : y = (x - y) - 6) : 
  x = 30 ∧ y = 12 := by
  sorry

end sons_ages_l363_36366


namespace expression_equals_1390_l363_36394

theorem expression_equals_1390 :
  (25 + 15 + 8) ^ 2 - (25 ^ 2 + 15 ^ 2 + 8 ^ 2) = 1390 := 
by
  sorry

end expression_equals_1390_l363_36394


namespace number_is_7612_l363_36302

-- Definitions of the conditions
def digits_correct_wrong_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10, 
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 ≠ (guess / 1000) % 10 ∧ 
      digits_placed 1 ≠ (guess / 100) % 10 ∧ 
      digits_placed 2 ≠ (guess / 10) % 10 ∧ 
      digits_placed 3 ≠ guess % 10)))

def digits_correct_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 = (guess / 1000) % 10 ∨ 
      digits_placed 1 = (guess / 100) % 10 ∨ 
      digits_placed 2 = (guess / 10) % 10 ∨ 
      digits_placed 3 = guess % 10)))

def digits_not_correct (guess : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → False)

-- The main theorem to prove
theorem number_is_7612 :
  digits_correct_wrong_positions 8765 2 ∧
  digits_correct_wrong_positions 1023 2 ∧
  digits_correct_positions 8642 2 ∧
  digits_not_correct 5430 →
  ∃ (num : Nat), 
    (num / 1000) % 10 = 7 ∧
    (num / 100) % 10 = 6 ∧
    (num / 10) % 10 = 1 ∧
    num % 10 = 2 ∧
    num = 7612 :=
sorry

end number_is_7612_l363_36302


namespace jacob_peter_age_ratio_l363_36398

theorem jacob_peter_age_ratio
  (Drew Maya Peter John Jacob : ℕ)
  (h1: Drew = Maya + 5)
  (h2: Peter = Drew + 4)
  (h3: John = 2 * Maya)
  (h4: John = 30)
  (h5: Jacob = 11) :
  Jacob + 2 = 1 / 2 * (Peter + 2) := by
  sorry

end jacob_peter_age_ratio_l363_36398


namespace pounds_of_apples_needed_l363_36372

-- Define the conditions
def n : ℕ := 8
def c_p : ℕ := 1
def a_p : ℝ := 2.00
def c_crust : ℝ := 2.00
def c_lemon : ℝ := 0.50
def c_butter : ℝ := 1.50

-- Define the theorem to be proven
theorem pounds_of_apples_needed : 
  (n * c_p - (c_crust + c_lemon + c_butter)) / a_p = 2 := 
by
  sorry

end pounds_of_apples_needed_l363_36372


namespace ines_bought_3_pounds_l363_36354

-- Define initial and remaining money of Ines
def initial_money : ℕ := 20
def remaining_money : ℕ := 14

-- Define the cost per pound of peaches
def cost_per_pound : ℕ := 2

-- The total money spent on peaches
def money_spent := initial_money - remaining_money

-- The number of pounds of peaches bought
def pounds_of_peaches := money_spent / cost_per_pound

-- The proof problem
theorem ines_bought_3_pounds :
  pounds_of_peaches = 3 :=
by
  sorry

end ines_bought_3_pounds_l363_36354


namespace total_sides_is_48_l363_36331

-- Definitions based on the conditions
def num_dice_tom : Nat := 4
def num_dice_tim : Nat := 4
def sides_per_die : Nat := 6

-- The proof problem statement
theorem total_sides_is_48 : (num_dice_tom + num_dice_tim) * sides_per_die = 48 := by
  sorry

end total_sides_is_48_l363_36331


namespace spherical_segment_equals_circle_area_l363_36337

noncomputable def spherical_segment_surface_area (R H : ℝ) : ℝ := 2 * Real.pi * R * H
noncomputable def circle_area (b : ℝ) : ℝ := Real.pi * (b * b)

theorem spherical_segment_equals_circle_area
  (R H b : ℝ) 
  (hb : b^2 = 2 * R * H) 
  : spherical_segment_surface_area R H = circle_area b :=
by
  sorry

end spherical_segment_equals_circle_area_l363_36337


namespace min_large_buses_proof_l363_36342

def large_bus_capacity : ℕ := 45
def small_bus_capacity : ℕ := 30
def total_students : ℕ := 523
def min_small_buses : ℕ := 5

def min_large_buses_required (large_capacity small_capacity total small_buses : ℕ) : ℕ :=
  let remaining_students := total - (small_buses * small_capacity)
  let buses_needed := remaining_students / large_capacity
  if remaining_students % large_capacity = 0 then buses_needed else buses_needed + 1

theorem min_large_buses_proof :
  min_large_buses_required large_bus_capacity small_bus_capacity total_students min_small_buses = 9 :=
by
  sorry

end min_large_buses_proof_l363_36342


namespace race_victory_l363_36341

variable (distance : ℕ := 200)
variable (timeA : ℕ := 18)
variable (timeA_beats_B_by : ℕ := 7)

theorem race_victory : ∃ meters_beats_B : ℕ, meters_beats_B = 56 :=
by
  let speedA := distance / timeA
  let timeB := timeA + timeA_beats_B_by
  let speedB := distance / timeB
  let distanceB := speedB * timeA
  let meters_beats_B := distance - distanceB
  use meters_beats_B
  sorry

end race_victory_l363_36341


namespace point_B_represents_2_or_neg6_l363_36351

def A : ℤ := -2

def B (move : ℤ) : ℤ := A + move

theorem point_B_represents_2_or_neg6 (move : ℤ) (h : move = 4 ∨ move = -4) : 
  B move = 2 ∨ B move = -6 :=
by
  cases h with
  | inl h1 => 
    rw [h1]
    unfold B
    unfold A
    simp
  | inr h1 => 
    rw [h1]
    unfold B
    unfold A
    simp

end point_B_represents_2_or_neg6_l363_36351


namespace households_accommodated_l363_36381

theorem households_accommodated (floors_per_building : ℕ)
                                (households_per_floor : ℕ)
                                (number_of_buildings : ℕ)
                                (total_households : ℕ)
                                (h1 : floors_per_building = 16)
                                (h2 : households_per_floor = 12)
                                (h3 : number_of_buildings = 10)
                                : total_households = 1920 :=
by
  sorry

end households_accommodated_l363_36381


namespace geometric_arithmetic_sequence_ratio_l363_36384

-- Given a positive geometric sequence {a_n} with a_3, a_5, a_6 forming an arithmetic sequence,
-- we need to prove that (a_3 + a_5) / (a_4 + a_6) is among specific values {1, (sqrt 5 - 1) / 2}

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos: ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 :=
by
  -- The proof is omitted
  sorry

end geometric_arithmetic_sequence_ratio_l363_36384


namespace f_at_zero_f_positive_f_increasing_l363_36387

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : true
axiom f_nonzero : f 0 ≠ 0
axiom f_pos_gt1 (x : ℝ) : x > 0 → f x > 1
axiom f_add (a b : ℝ) : f (a + b) = f a * f b

theorem f_at_zero : f 0 = 1 :=
sorry

theorem f_positive (x : ℝ) : f x > 0 :=
sorry

theorem f_increasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

end f_at_zero_f_positive_f_increasing_l363_36387


namespace zoo_problem_l363_36325

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end zoo_problem_l363_36325


namespace crowdfunding_total_amount_l363_36319

theorem crowdfunding_total_amount
  (backers_highest_level : ℕ := 2)
  (backers_second_level : ℕ := 3)
  (backers_lowest_level : ℕ := 10)
  (amount_highest_level : ℝ := 5000) :
  ((backers_highest_level * amount_highest_level) + 
   (backers_second_level * (amount_highest_level / 10)) + 
   (backers_lowest_level * (amount_highest_level / 100))) = 12000 :=
by
  sorry

end crowdfunding_total_amount_l363_36319


namespace factor_expression_l363_36361

theorem factor_expression (a b c : ℝ) :
  3*a^3*(b^2 - c^2) - 2*b^3*(c^2 - a^2) + c^3*(a^2 - b^2) =
  (a - b)*(b - c)*(c - a)*(3*a^2 - 2*b^2 - 3*a^3/c + c) :=
sorry

end factor_expression_l363_36361


namespace hulk_jump_geometric_seq_l363_36362

theorem hulk_jump_geometric_seq :
  ∃ n : ℕ, (2 * 3^(n-1) > 2000) ∧ n = 8 :=
by
  sorry

end hulk_jump_geometric_seq_l363_36362


namespace p_sufficient_not_necessary_for_q_l363_36320

-- Given conditions p and q
def p_geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def q_product_equality (a b c d : ℝ) : Prop :=
  a * d = b * c

-- Theorem statement: p implies q, but q does not imply p
theorem p_sufficient_not_necessary_for_q (a b c d : ℝ) :
  (p_geometric_sequence a b c d → q_product_equality a b c d) ∧
  (¬ (q_product_equality a b c d → p_geometric_sequence a b c d)) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l363_36320


namespace ashley_age_l363_36396

theorem ashley_age (A M : ℕ) (h1 : 4 * M = 7 * A) (h2 : A + M = 22) : A = 8 :=
sorry

end ashley_age_l363_36396


namespace pencil_length_after_sharpening_l363_36376

def initial_length : ℕ := 50
def monday_sharpen : ℕ := 2
def tuesday_sharpen : ℕ := 3
def wednesday_sharpen : ℕ := 4
def thursday_sharpen : ℕ := 5

def total_sharpened : ℕ := monday_sharpen + tuesday_sharpen + wednesday_sharpen + thursday_sharpen

def final_length : ℕ := initial_length - total_sharpened

theorem pencil_length_after_sharpening : final_length = 36 := by
  -- Here would be the proof body
  sorry

end pencil_length_after_sharpening_l363_36376


namespace number_of_ways_two_girls_together_l363_36395

theorem number_of_ways_two_girls_together
  (boys girls : ℕ)
  (total_people : ℕ)
  (ways : ℕ) :
  boys = 3 →
  girls = 3 →
  total_people = boys + girls →
  ways = 432 :=
by
  intros
  sorry

end number_of_ways_two_girls_together_l363_36395


namespace probability_of_selecting_one_marble_each_color_l363_36345

theorem probability_of_selecting_one_marble_each_color
  (total_red_marbles : ℕ) (total_blue_marbles : ℕ) (total_green_marbles : ℕ) (total_selected_marbles : ℕ) 
  (total_marble_count : ℕ) : 
  total_red_marbles = 3 → total_blue_marbles = 3 → total_green_marbles = 3 → total_selected_marbles = 3 → total_marble_count = 9 →
  (27 / 84) = 9 / 28 :=
by
  intros h_red h_blue h_green h_selected h_total
  sorry

end probability_of_selecting_one_marble_each_color_l363_36345


namespace partition_of_sum_l363_36323

-- Define the conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_bounded_integer (n : ℕ) : Prop := n ≤ 10
def can_be_partitioned (S : ℕ) (integers : List ℕ) : Prop :=
  ∃ (A B : List ℕ), 
    A.sum ≤ 70 ∧ 
    B.sum ≤ 70 ∧ 
    A ++ B = integers

-- Define the theorem statement
theorem partition_of_sum (S : ℕ) (integers : List ℕ)
  (h1 : ∀ x ∈ integers, is_positive_integer x ∧ is_bounded_integer x)
  (h2 : List.sum integers = S) :
  S ≤ 133 ↔ can_be_partitioned S integers :=
sorry

end partition_of_sum_l363_36323


namespace blue_tickets_per_red_ticket_l363_36370

-- Definitions based on conditions
def yellow_tickets_to_win_bible : Nat := 10
def red_tickets_per_yellow_ticket : Nat := 10
def blue_tickets_needed : Nat := 163
def additional_yellow_tickets_needed (current_yellow : Nat) : Nat := yellow_tickets_to_win_bible - current_yellow
def additional_red_tickets_needed (current_red : Nat) (needed_yellow : Nat) : Nat := needed_yellow * red_tickets_per_yellow_ticket - current_red

-- Given conditions
def current_yellow_tickets : Nat := 8
def current_red_tickets : Nat := 3
def current_blue_tickets : Nat := 7
def needed_yellow_tickets : Nat := additional_yellow_tickets_needed current_yellow_tickets
def needed_red_tickets : Nat := additional_red_tickets_needed current_red_tickets needed_yellow_tickets

-- Theorem to prove
theorem blue_tickets_per_red_ticket : blue_tickets_needed / needed_red_tickets = 10 :=
by
  sorry

end blue_tickets_per_red_ticket_l363_36370


namespace inequality_system_solution_l363_36382

theorem inequality_system_solution:
  ∀ (x : ℝ),
  (1 - (2*x - 1) / 2 > (3*x - 1) / 4) ∧ (2 - 3*x ≤ 4 - x) →
  -1 ≤ x ∧ x < 1 :=
by
  intro x
  intro h
  sorry

end inequality_system_solution_l363_36382


namespace geometric_sequence_correct_l363_36365

theorem geometric_sequence_correct (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 8)
  (h2 : a 2 * a 3 = -8)
  (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r) :
  a 4 = -1 :=
by {
  sorry
}

end geometric_sequence_correct_l363_36365


namespace proof_problem_l363_36316

variables (p q : Prop)

-- Assuming p is true and q is false
axiom p_is_true : p
axiom q_is_false : ¬ q

-- Proving that (¬p) ∨ (¬q) is true
theorem proof_problem : (¬p) ∨ (¬q) :=
by {
  sorry
}

end proof_problem_l363_36316


namespace students_in_zack_classroom_l363_36378

theorem students_in_zack_classroom 
(T M Z : ℕ)
(h1 : T = M)
(h2 : Z = (T + M) / 2)
(h3 : T + M + Z = 69) :
Z = 23 :=
by
  sorry

end students_in_zack_classroom_l363_36378


namespace trigonometric_identity_l363_36397

theorem trigonometric_identity (x : ℝ) (h₁ : Real.sin x = 4 / 5) (h₂ : π / 2 ≤ x ∧ x ≤ π) :
  Real.cos x = -3 / 5 ∧ (Real.cos (-x) / (Real.sin (π / 2 - x) - Real.sin (2 * π - x)) = -3) := 
by
  sorry

end trigonometric_identity_l363_36397


namespace christmas_tree_problem_l363_36307

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end christmas_tree_problem_l363_36307


namespace conjunction_used_in_proposition_l363_36306

theorem conjunction_used_in_proposition (x : ℝ) (h : x^2 = 4) :
  (x = 2 ∨ x = -2) :=
sorry

end conjunction_used_in_proposition_l363_36306


namespace two_digit_numbers_l363_36386

theorem two_digit_numbers :
  ∃ (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ x < y ∧ 2000 + x + y = x * y := 
sorry

end two_digit_numbers_l363_36386


namespace contractor_daily_wage_l363_36347

theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_absent_day total_amount : ℝ) (daily_wage : ℝ)
  (h_total_days : total_days = 30)
  (h_absent_days : absent_days = 8)
  (h_fine : fine_per_absent_day = 7.50)
  (h_total_amount : total_amount = 490) 
  (h_work_days : total_days - absent_days = 22)
  (h_total_fined : fine_per_absent_day * absent_days = 60)
  (h_total_earned : 22 * daily_wage - 60 = 490) :
  daily_wage = 25 := 
by 
  sorry

end contractor_daily_wage_l363_36347


namespace find_a8_l363_36389

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

theorem find_a8
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_terms a_n S)
  (h_S15 : S 15 = 45) :
  a_n 8 = 3 :=
sorry

end find_a8_l363_36389


namespace polynomial_factorization_l363_36318

-- Define the given polynomial expression
def given_poly (x : ℤ) : ℤ :=
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2

-- Define the supposed factored form
def factored_poly (x : ℤ) : ℤ :=
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895)

-- The theorem stating the equality of the two expressions
theorem polynomial_factorization (x : ℤ) : given_poly x = factored_poly x :=
  sorry

end polynomial_factorization_l363_36318


namespace find_number_l363_36357

theorem find_number {x : ℝ} (h : 0.5 * x - 10 = 25) : x = 70 :=
sorry

end find_number_l363_36357


namespace range_of_reciprocal_sum_l363_36315

theorem range_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : a + b = 1) :
  ∃ c > 4, ∀ x, x = (1 / a + 1 / b) → c < x :=
sorry

end range_of_reciprocal_sum_l363_36315


namespace largest_root_is_sqrt6_l363_36322

theorem largest_root_is_sqrt6 (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -6) 
  (h3 : p * q * r = -18) : 
  max p (max q r) = Real.sqrt 6 := 
sorry

end largest_root_is_sqrt6_l363_36322


namespace liars_are_C_and_D_l363_36373
open Classical 

-- We define inhabitants and their statements
inductive Inhabitant
| A | B | C | D

open Inhabitant

axiom is_liar : Inhabitant → Prop

-- Statements by the inhabitants:
-- A: "At least one of us is a liar."
-- B: "At least two of us are liars."
-- C: "At least three of us are liars."
-- D: "None of us are liars."

def statement_A : Prop := is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D
def statement_B : Prop := (is_liar A ∧ is_liar B) ∨ (is_liar A ∧ is_liar C) ∨ (is_liar A ∧ is_liar D) ∨
                          (is_liar B ∧ is_liar C) ∨ (is_liar B ∧ is_liar D) ∨ (is_liar C ∧ is_liar D)
def statement_C : Prop := (is_liar A ∧ is_liar B ∧ is_liar C) ∨ (is_liar A ∧ is_liar B ∧ is_liar D) ∨
                          (is_liar A ∧ is_liar C ∧ is_liar D) ∨ (is_liar B ∧ is_liar C ∧ is_liar D)
def statement_D : Prop := ¬(is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D)

-- Given that there are some liars
axiom some_liars_exist : ∃ x, is_liar x

-- Lean proof statement
theorem liars_are_C_and_D : is_liar C ∧ is_liar D ∧ ¬(is_liar A) ∧ ¬(is_liar B) :=
by
  sorry

end liars_are_C_and_D_l363_36373


namespace shooting_enthusiast_l363_36377

variables {P : ℝ} -- Declare P as a real number

-- Define the conditions where X follows a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) :=
  n * p * (1 - p)

-- State the theorem in Lean 4
theorem shooting_enthusiast (h : binomial_variance 3 P = 3 / 4) : 
  P = 1 / 2 :=
by
  sorry -- Proof goes here

end shooting_enthusiast_l363_36377


namespace industrial_lubricants_percentage_l363_36344

noncomputable def percentage_microphotonics : ℕ := 14
noncomputable def percentage_home_electronics : ℕ := 19
noncomputable def percentage_food_additives : ℕ := 10
noncomputable def percentage_gmo : ℕ := 24
noncomputable def total_percentage : ℕ := 100
noncomputable def percentage_basic_astrophysics : ℕ := 25

theorem industrial_lubricants_percentage :
  total_percentage - (percentage_microphotonics + percentage_home_electronics + 
  percentage_food_additives + percentage_gmo + percentage_basic_astrophysics) = 8 := 
sorry

end industrial_lubricants_percentage_l363_36344


namespace find_a_l363_36390

-- Define the circle equation and the line equation as conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1
def line_eq (x y a : ℝ) : Prop := y = x + a
def chord_length (l : ℝ) : Prop := l = 2

-- State the main problem
theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, circle_eq x y → ∃ y', line_eq x y' a ∧ chord_length 2) :
  a = -2 :=
sorry

end find_a_l363_36390


namespace rajas_salary_percentage_less_than_rams_l363_36305

-- Definitions from the problem conditions
def raja_salary : ℚ := sorry -- Placeholder, since Raja's salary doesn't need a fixed value
def ram_salary : ℚ := 1.25 * raja_salary

-- Theorem to be proved
theorem rajas_salary_percentage_less_than_rams :
  ∃ r : ℚ, (ram_salary - raja_salary) / ram_salary * 100 = 20 :=
by
  sorry

end rajas_salary_percentage_less_than_rams_l363_36305


namespace range_of_a_plus_b_l363_36379

variable {a b : ℝ}

def has_two_real_roots (a b : ℝ) : Prop :=
  let discriminant := b^2 - 4 * a * (-4)
  discriminant ≥ 0

def has_root_in_interval (a b : ℝ) : Prop :=
  (a + b - 4) * (4 * a + 2 * b - 4) < 0

theorem range_of_a_plus_b 
  (h1 : has_two_real_roots a b) 
  (h2 : has_root_in_interval a b) 
  (h3 : a > 0) : 
  a + b < 4 :=
sorry

end range_of_a_plus_b_l363_36379


namespace trig_identity_example_l363_36399

open Real -- Using the Real namespace for trigonometric functions

theorem trig_identity_example :
  sin (135 * π / 180) * cos (-15 * π / 180) + cos (225 * π / 180) * sin (15 * π / 180) = 1 / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end trig_identity_example_l363_36399


namespace smallest_number_two_reps_l363_36348

theorem smallest_number_two_reps : 
  ∃ (n : ℕ), (∀ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = n ∧ 3 * x2 + 4 * y2 = n → (x1 = x2 ∧ y1 = y2 ∨ ¬(x1 = x2 ∧ y1 = y2))) ∧ 
  ∀ m < n, (∀ x y : ℕ, ¬(3 * x + 4 * y = m ∧ ¬∃ (x1 y1 : ℕ), 3 * x1 + 4 * y1 = m) ∧ 
            (∃ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = m ∧ 3 * x2 + 4 * y2 = m ∧ ¬(x1 = x2 ∧ y1 = y2))) :=
  sorry

end smallest_number_two_reps_l363_36348


namespace desired_depth_is_50_l363_36317

noncomputable def desired_depth_dig (d days : ℝ) : ℝ :=
  let initial_man_hours := 45 * 8 * d
  let additional_man_hours := 100 * 6 * d
  (initial_man_hours / additional_man_hours) * 30

theorem desired_depth_is_50 (d : ℝ) : desired_depth_dig d = 50 :=
  sorry

end desired_depth_is_50_l363_36317


namespace parabola_transformation_l363_36369

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end parabola_transformation_l363_36369


namespace quotient_is_eight_l363_36330

theorem quotient_is_eight (d v r q : ℕ) (h₁ : d = 141) (h₂ : v = 17) (h₃ : r = 5) (h₄ : d = v * q + r) : q = 8 :=
by
  sorry

end quotient_is_eight_l363_36330


namespace range_of_m_l363_36321

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), -2 ≤ x ∧ x ≤ 3 ∧ m * x + 6 = 0) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l363_36321


namespace find_principal_amount_l363_36356

variable {P R T : ℝ} -- variables for principal, rate, and time
variable (H1: R = 25)
variable (H2: T = 2)
variable (H3: (P * (0.5625) - P * (0.5)) = 225)

theorem find_principal_amount
    (H1 : R = 25)
    (H2 : T = 2)
    (H3 : (P * 0.0625) = 225) : 
    P = 3600 := 
  sorry

end find_principal_amount_l363_36356


namespace percentage_error_l363_36338

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error_l363_36338


namespace gcd_765432_654321_l363_36327

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end gcd_765432_654321_l363_36327


namespace polygon_sides_l363_36312

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end polygon_sides_l363_36312


namespace find_A_find_b_and_c_l363_36329

open Real

variable {a b c A B C : ℝ}

-- Conditions for the problem
axiom triangle_sides : ∀ {A B C : ℝ}, a > 0
axiom sine_law_condition : b * sin B + c * sin C - sqrt 2 * b * sin C = a * sin A
axiom degrees_60 : B = π / 3
axiom side_a : a = 2

theorem find_A : A = π / 4 :=
by sorry

theorem find_b_and_c (h : A = π / 4) (hB : B = π / 3) (ha : a = 2) : b = sqrt 6 ∧ c = 1 + sqrt 3 :=
by sorry

end find_A_find_b_and_c_l363_36329


namespace circle_radius_equivalence_l363_36375

theorem circle_radius_equivalence (OP_radius : ℝ) (QR : ℝ) (a : ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ)
  (h1 : P = (12, 5))
  (h2 : S = (a, 0))
  (h3 : QR = 5)
  (h4 : OP_radius = 13) :
  a = 8 := 
sorry

end circle_radius_equivalence_l363_36375


namespace shaded_fraction_is_correct_l363_36309

-- Definitions based on the identified conditions
def initial_fraction_shaded : ℚ := 4 / 9
def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)
def infinite_series_fraction_shaded : ℚ := 4 / 9 * (4 / 3)

-- The theorem stating the problem
theorem shaded_fraction_is_correct :
  infinite_series_fraction_shaded = 16 / 27 :=
by
  sorry -- proof to be provided

end shaded_fraction_is_correct_l363_36309


namespace lottery_probability_correct_l363_36359

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_correct :
  let MegaBall_probability := 1 / 30
  let WinnerBalls_probability := 1 / (combination 50 6)
  MegaBall_probability * WinnerBalls_probability = 1 / 476721000 :=
by
  sorry

end lottery_probability_correct_l363_36359


namespace find_interest_rate_l363_36383

-- Translating the identified conditions into Lean definitions
def initial_deposit (P : ℝ) : Prop := P > 0
def compounded_semiannually (n : ℕ) : Prop := n = 2
def growth_in_sum (A : ℝ) (P : ℝ) : Prop := A = 1.1592740743 * P
def time_period (t : ℝ) : Prop := t = 2.5

theorem find_interest_rate (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (A : ℝ)
  (h_init : initial_deposit P)
  (h_n : compounded_semiannually n)
  (h_A : growth_in_sum A P)
  (h_t : time_period t) :
  r = 0.06 :=
by
  sorry

end find_interest_rate_l363_36383


namespace smallest_N_for_equal_adults_and_children_l363_36303

theorem smallest_N_for_equal_adults_and_children :
  ∃ (N : ℕ), N > 0 ∧ (∀ a b : ℕ, 8 * N = a ∧ 12 * N = b ∧ a = b) ∧ N = 3 :=
sorry

end smallest_N_for_equal_adults_and_children_l363_36303


namespace series_converges_to_one_l363_36391

noncomputable def infinite_series := ∑' n, (3^n) / (3^(2^n) + 2)

theorem series_converges_to_one :
  infinite_series = 1 := by
  sorry

end series_converges_to_one_l363_36391


namespace trapezium_distance_l363_36346

theorem trapezium_distance (a b h: ℝ) (area: ℝ) (h_area: area = 300) (h_sides: a = 22) (h_sides_2: b = 18)
  (h_formula: area = (1 / 2) * (a + b) * h): h = 15 :=
by
  sorry

end trapezium_distance_l363_36346


namespace enclosed_area_is_correct_l363_36336

noncomputable def area_between_curves : ℝ := 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let cubic_parabola (x : ℝ) := - 1 / 2 * x^3 + 2 * x
  let x1 : ℝ := -2
  let x2 : ℝ := Real.sqrt 2
  -- Properly calculate the area between the two curves
  sorry

theorem enclosed_area_is_correct :
  area_between_curves = 3 * ( Real.pi + 1 ) / 2 :=
sorry

end enclosed_area_is_correct_l363_36336


namespace ratio_of_areas_l363_36308

theorem ratio_of_areas
  (R_X R_Y : ℝ)
  (h : (60 / 360) * 2 * Real.pi * R_X = (40 / 360) * 2 * Real.pi * R_Y) :
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l363_36308


namespace find_y_l363_36326

theorem find_y (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 4 * y + 2 = 0)
  (h2 : 3 * x + y + 4 = 0) :
  y^2 + 17 * y - 11 = 0 :=
by 
  sorry

end find_y_l363_36326


namespace sequence_general_formula_l363_36360

theorem sequence_general_formula :
  (∃ a : ℕ → ℕ, a 1 = 4 ∧ a 2 = 6 ∧ a 3 = 8 ∧ a 4 = 10 ∧ (∀ n : ℕ, a n = 2 * (n + 1))) :=
by
  sorry

end sequence_general_formula_l363_36360


namespace simplify_expression_l363_36300

variables {x p q r : ℝ}

theorem simplify_expression (h1 : p ≠ q) (h2 : p ≠ r) (h3 : q ≠ r) :
   ( (x + p)^4 / ((p - q) * (p - r)) + (x + q)^4 / ((q - p) * (q - r)) + (x + r)^4 / ((r - p) * (r - q)) 
   ) = p + q + r + 4 * x :=
sorry

end simplify_expression_l363_36300


namespace brenda_total_erasers_l363_36355

theorem brenda_total_erasers (number_of_groups : ℕ) (erasers_per_group : ℕ) (h1 : number_of_groups = 3) (h2 : erasers_per_group = 90) : number_of_groups * erasers_per_group = 270 := 
by
  sorry

end brenda_total_erasers_l363_36355


namespace cone_volume_l363_36364

theorem cone_volume (diameter height : ℝ) (h_diam : diameter = 14) (h_height : height = 12) :
  (1 / 3 : ℝ) * Real.pi * ((diameter / 2) ^ 2) * height = 196 * Real.pi := by
  sorry

end cone_volume_l363_36364


namespace complement_of_A_relative_to_U_l363_36313

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 4, 5}

-- Define the proof statement for the complement of A with respect to U
theorem complement_of_A_relative_to_U : (U \ A) = {2} := by
  sorry

end complement_of_A_relative_to_U_l363_36313


namespace right_triangle_legs_sum_squares_area_l363_36350

theorem right_triangle_legs_sum_squares_area:
  ∀ (a b c : ℝ), 
  (0 < a) → (0 < b) → (0 < c) → 
  (a^2 + b^2 = c^2) → 
  (1 / 2 * a * b = 24) → 
  (a^2 + b^2 = 48) → 
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) := 
by
  sorry

end right_triangle_legs_sum_squares_area_l363_36350


namespace opposite_face_of_orange_is_blue_l363_36334

structure CubeOrientation :=
  (top : String)
  (front : String)
  (right : String)

def first_view : CubeOrientation := { top := "B", front := "Y", right := "S" }
def second_view : CubeOrientation := { top := "B", front := "V", right := "S" }
def third_view : CubeOrientation := { top := "B", front := "K", right := "S" }

theorem opposite_face_of_orange_is_blue
  (colors : List String)
  (c1 : CubeOrientation)
  (c2 : CubeOrientation)
  (c3 : CubeOrientation)
  (no_orange_in_views : "O" ∉ colors.erase c1.top ∧ "O" ∉ colors.erase c1.front ∧ "O" ∉ colors.erase c1.right ∧
                         "O" ∉ colors.erase c2.top ∧ "O" ∉ colors.erase c2.front ∧ "O" ∉ colors.erase c2.right ∧
                         "O" ∉ colors.erase c3.top ∧ "O" ∉ colors.erase c3.front ∧ "O" ∉ colors.erase c3.right) :
  (c1.top = "B" → c2.top = "B" → c3.top = "B" → c1.right = "S" → c2.right = "S" → c3.right = "S" → 
  ∃ opposite_color, opposite_color = "B") :=
sorry

end opposite_face_of_orange_is_blue_l363_36334


namespace decorations_total_l363_36343

def number_of_skulls : Nat := 12
def number_of_broomsticks : Nat := 4
def number_of_spiderwebs : Nat := 12
def number_of_pumpkins (spiderwebs : Nat) : Nat := 2 * spiderwebs
def number_of_cauldron : Nat := 1
def number_of_lanterns (trees : Nat) : Nat := 3 * trees
def number_of_scarecrows (trees : Nat) : Nat := 1 * (trees / 2)
def total_stickers : Nat := 30
def stickers_per_window (stickers : Nat) (windows : Nat) : Nat := (stickers / 2) / windows
def additional_decorations (bought : Nat) (used_percent : Nat) (leftover : Nat) : Nat := ((bought * used_percent) / 100) + leftover

def total_decorations : Nat :=
  number_of_skulls +
  number_of_broomsticks +
  number_of_spiderwebs +
  (number_of_pumpkins number_of_spiderwebs) +
  number_of_cauldron +
  (number_of_lanterns 5) +
  (number_of_scarecrows 4) +
  (additional_decorations 25 70 15)

theorem decorations_total : total_decorations = 102 := by
  sorry

end decorations_total_l363_36343


namespace intersection_x_coordinate_l363_36340

-- Definitions based on conditions
def line1 (x : ℝ) : ℝ := 3 * x + 5
def line2 (x : ℝ) : ℝ := 35 - 5 * x

-- Proof statement
theorem intersection_x_coordinate : ∃ x : ℝ, line1 x = line2 x ∧ x = 15 / 4 :=
by
  use 15 / 4
  sorry

end intersection_x_coordinate_l363_36340


namespace percentage_rent_this_year_l363_36352

variables (E : ℝ)

-- Define the conditions from the problem
def rent_last_year (E : ℝ) : ℝ := 0.20 * E
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 1.4375 * rent_last_year E

-- The main statement to prove
theorem percentage_rent_this_year : 
  0.2875 * E = (25 / 100) * (earnings_this_year E) :=
by sorry

end percentage_rent_this_year_l363_36352


namespace sufficient_not_necessary_condition_abs_eq_one_l363_36367

theorem sufficient_not_necessary_condition_abs_eq_one (a : ℝ) :
  (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 ∨ a = -1) :=
by
  sorry

end sufficient_not_necessary_condition_abs_eq_one_l363_36367


namespace probability_twice_correct_l363_36388

noncomputable def probability_at_least_twice (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 3000) then
  if y ≥ 2*x then (1/6 : ℝ) else 0
else 0

theorem probability_twice_correct : probability_at_least_twice 500 1000 = (1/6 : ℝ) :=
sorry

end probability_twice_correct_l363_36388


namespace mathematical_proof_l363_36333

noncomputable def proof_problem (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : Prop :=
  (1 + x) / y < 2 ∨ (1 + y) / x < 2

theorem mathematical_proof (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : proof_problem x y hx_pos hxy_gt2 :=
by {
  sorry
}

end mathematical_proof_l363_36333


namespace average_after_11th_inning_is_30_l363_36310

-- Define the conditions as Lean 4 definitions
def score_in_11th_inning : ℕ := 80
def increase_in_avg : ℕ := 5
def innings_before_11th : ℕ := 10

-- Define the average before 11th inning
def average_before (x : ℕ) : ℕ := x

-- Define the total runs before 11th inning
def total_runs_before (x : ℕ) : ℕ := innings_before_11th * (average_before x)

-- Define the total runs after 11th inning
def total_runs_after (x : ℕ) : ℕ := total_runs_before x + score_in_11th_inning

-- Define the new average after 11th inning
def new_average_after (x : ℕ) : ℕ := total_runs_after x / (innings_before_11th + 1)

-- Theorem statement
theorem average_after_11th_inning_is_30 : 
  ∃ (x : ℕ), new_average_after x = average_before x + increase_in_avg → new_average_after 25 = 30 :=
by
  sorry

end average_after_11th_inning_is_30_l363_36310


namespace problem_statement_l363_36374

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end problem_statement_l363_36374


namespace geometric_series_sum_first_four_terms_l363_36301

theorem geometric_series_sum_first_four_terms :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  (a * (1 - r^n) / (1 - r)) = 40 / 27 := by
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  sorry

end geometric_series_sum_first_four_terms_l363_36301


namespace largest_divisor_poly_l363_36304

-- Define the polynomial and the required properties
def poly (n : ℕ) : ℕ := (n+1) * (n+3) * (n+5) * (n+7) * (n+11)

-- Define the conditions and the proof statement
theorem largest_divisor_poly (n : ℕ) (h_even : n % 2 = 0) : ∃ d, d = 15 ∧ ∀ m, m ∣ poly n → m ≤ d :=
by
  sorry

end largest_divisor_poly_l363_36304


namespace number_of_valid_triples_l363_36335

theorem number_of_valid_triples : 
  ∃ n, n = 7 ∧ ∀ (a b c : ℕ), b = 2023 → a ≤ b → b ≤ c → a * c = 2023^2 → (n = 7) :=
by 
  sorry

end number_of_valid_triples_l363_36335


namespace one_third_percent_of_200_l363_36380

theorem one_third_percent_of_200 : ((1206 / 3) / 200) * 100 = 201 := by
  sorry

end one_third_percent_of_200_l363_36380


namespace shaded_area_l363_36358

-- Definition of square side lengths
def side_lengths : List ℕ := [2, 4, 6, 8, 10]

-- Definition for the area of the largest square
def largest_square_area : ℕ := 10 * 10

-- Definition for the area of the smallest non-shaded square
def smallest_square_area : ℕ := 2 * 2

-- Total area of triangular regions
def triangular_area : ℕ := 2 * (2 * 4 + 2 * 6 + 2 * 8 + 2 * 10)

-- Question to prove
theorem shaded_area : largest_square_area - smallest_square_area - triangular_area = 40 := by
  sorry

end shaded_area_l363_36358


namespace driver_spending_increase_l363_36349

theorem driver_spending_increase (P Q : ℝ) (X : ℝ) (h1 : 1.20 * P = (1 + 20 / 100) * P) (h2 : 0.90 * Q = (1 - 10 / 100) * Q) :
  (1 + X / 100) * (P * Q) = 1.20 * P * 0.90 * Q → X = 8 := 
by
  sorry

end driver_spending_increase_l363_36349


namespace unique_toy_value_l363_36332

/-- Allie has 9 toys in total. The total worth of these toys is $52. 
One toy has a certain value "x" dollars and the remaining 8 toys each have a value of $5. 
Prove that the value of the unique toy is $12. -/
theorem unique_toy_value (x : ℕ) (h1 : 1 + 8 = 9) (h2 : x + 8 * 5 = 52) : x = 12 :=
by
  sorry

end unique_toy_value_l363_36332


namespace total_handshakes_calculation_l363_36328

-- Define the conditions
def teams := 3
def players_per_team := 5
def total_players := teams * players_per_team
def referees := 2

def handshakes_among_players := (total_players * (players_per_team * (teams - 1))) / 2
def handshakes_with_referees := total_players * referees

def total_handshakes := handshakes_among_players + handshakes_with_referees

-- Define the theorem statement
theorem total_handshakes_calculation :
  total_handshakes = 105 :=
by
  sorry

end total_handshakes_calculation_l363_36328


namespace range_of_m_l363_36385

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 1 > m

def proposition_q (m : ℝ) : Prop :=
  3 - m > 1

theorem range_of_m (m : ℝ) (p_false : ¬proposition_p m) (q_true : proposition_q m) (pq_false : ¬(proposition_p m ∧ proposition_q m)) (porq_true : proposition_p m ∨ proposition_q m) : 
  1 ≤ m ∧ m < 2 := 
sorry

end range_of_m_l363_36385


namespace smallest_constant_for_triangle_l363_36324

theorem smallest_constant_for_triangle 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)  
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 := 
  sorry

end smallest_constant_for_triangle_l363_36324


namespace matches_between_withdrawn_players_l363_36393

theorem matches_between_withdrawn_players (n r : ℕ) (h : 50 = (n - 3).choose 2 + (6 - r) + r) : r = 1 :=
sorry

end matches_between_withdrawn_players_l363_36393


namespace largest_integer_l363_36311

theorem largest_integer (a b c d : ℤ) 
  (h1 : a + b + c = 210) 
  (h2 : a + b + d = 230) 
  (h3 : a + c + d = 245) 
  (h4 : b + c + d = 260) : 
  d = 105 :=
by 
  sorry

end largest_integer_l363_36311


namespace estimate_white_balls_l363_36371

theorem estimate_white_balls :
  (∃ x : ℕ, (6 / (x + 6) : ℝ) = 0.2 ∧ x = 24) :=
by
  use 24
  sorry

end estimate_white_balls_l363_36371
