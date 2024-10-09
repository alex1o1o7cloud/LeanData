import Mathlib

namespace timothy_read_pages_l1666_166646

theorem timothy_read_pages 
    (mon_tue_pages : Nat) (wed_pages : Nat) (thu_sat_pages : Nat) 
    (sun_read_pages : Nat) (sun_review_pages : Nat) : 
    mon_tue_pages = 45 → wed_pages = 50 → thu_sat_pages = 40 → sun_read_pages = 25 → sun_review_pages = 15 →
    (2 * mon_tue_pages + wed_pages + 3 * thu_sat_pages + sun_read_pages + sun_review_pages = 300) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end timothy_read_pages_l1666_166646


namespace solve_eq1_solve_eq2_l1666_166666

noncomputable def eq1_solution1 := -2 + Real.sqrt 5
noncomputable def eq1_solution2 := -2 - Real.sqrt 5

noncomputable def eq2_solution1 := 3
noncomputable def eq2_solution2 := 1

theorem solve_eq1 (x : ℝ) :
  x^2 + 4 * x - 1 = 0 → (x = eq1_solution1 ∨ x = eq1_solution2) :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  (x - 3)^2 + 2 * x * (x - 3) = 0 → (x = eq2_solution1 ∨ x = eq2_solution2) :=
by 
  sorry

end solve_eq1_solve_eq2_l1666_166666


namespace message_forwarding_time_l1666_166661

theorem message_forwarding_time :
  ∃ n : ℕ, (∀ m : ℕ, (∀ p : ℕ, (∀ q : ℕ, 1 + (2 * (2 ^ n)) - 1 = 2047)) ∧ n = 10) :=
sorry

end message_forwarding_time_l1666_166661


namespace field_area_l1666_166649

-- Define the given conditions and prove the area of the field
theorem field_area (x y : ℕ) 
  (h1 : 2*(x + 20) + 2*y = 2*(2*x + 2*y))
  (h2 : 2*x + 2*(2*y) = 2*x + 2*y + 18) : x * y = 99 := by 
{
  sorry
}

end field_area_l1666_166649


namespace general_term_less_than_zero_from_13_l1666_166622

-- Define the arithmetic sequence and conditions
def an (n : ℕ) : ℝ := 12 - n

-- Condition: a_3 = 9
def a3_condition : Prop := an 3 = 9

-- Condition: a_9 = 3
def a9_condition : Prop := an 9 = 3

-- Prove the general term of the sequence is 12 - n
theorem general_term (n : ℕ) (h3 : a3_condition) (h9 : a9_condition) :
  an n = 12 - n := 
sorry

-- Prove that the sequence becomes less than 0 starting from the 13th term
theorem less_than_zero_from_13 (h3 : a3_condition) (h9 : a9_condition) :
  ∀ n, n ≥ 13 → an n < 0 :=
sorry

end general_term_less_than_zero_from_13_l1666_166622


namespace intersects_line_l1666_166615

theorem intersects_line (x y : ℝ) : 
  (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) → ∃ x y : ℝ, (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) :=
by
  intro h
  sorry

end intersects_line_l1666_166615


namespace antonieta_tickets_needed_l1666_166694

-- Definitions based on conditions:
def ferris_wheel_tickets : ℕ := 6
def roller_coaster_tickets : ℕ := 5
def log_ride_tickets : ℕ := 7
def antonieta_initial_tickets : ℕ := 2

-- Theorem to prove the required number of tickets Antonieta should buy
theorem antonieta_tickets_needed : ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - antonieta_initial_tickets = 16 :=
by
  sorry

end antonieta_tickets_needed_l1666_166694


namespace percentage_of_women_in_study_group_l1666_166632

variable (W : ℝ) -- W is the percentage of women in the study group in decimal form

-- Given conditions as hypotheses
axiom h1 : 0 < W ∧ W <= 1         -- W represents a percentage, so it must be between 0 and 1.
axiom h2 : 0.40 * W = 0.28         -- 40 percent of women are lawyers, and the probability of selecting a woman lawyer is 0.28.

-- The statement to prove
theorem percentage_of_women_in_study_group : W = 0.7 :=
by
  sorry

end percentage_of_women_in_study_group_l1666_166632


namespace no_integer_solution_l1666_166655

theorem no_integer_solution :
  ∀ (x : ℤ), ¬ (x^2 + 3 < 2 * x) :=
by
  intro x
  sorry

end no_integer_solution_l1666_166655


namespace time_to_fill_bottle_l1666_166684

-- Definitions
def flow_rate := 500 / 6 -- mL per second
def volume := 250 -- mL

-- Target theorem
theorem time_to_fill_bottle (r : ℝ) (v : ℝ) (t : ℝ) (h : r = flow_rate) (h2 : v = volume) : t = 3 :=
by
  sorry

end time_to_fill_bottle_l1666_166684


namespace find_starting_number_l1666_166609

theorem find_starting_number : 
  ∃ x : ℕ, (∃ n : ℕ, n = 21 ∧ (forall k, 1 ≤ k ∧ k ≤ n → x + k*19 ≤ 500) ∧ 
  (forall k, 1 ≤ k ∧ k < n → x + k*19 > 0)) ∧ x = 113 := by {
  sorry
}

end find_starting_number_l1666_166609


namespace fraction_spent_on_raw_material_l1666_166626

variable (C : ℝ)
variable (x : ℝ)

theorem fraction_spent_on_raw_material :
  C - x * C - (1/10) * (C * (1 - x)) = 0.675 * C → x = 1/4 :=
by
  sorry

end fraction_spent_on_raw_material_l1666_166626


namespace calculate_otimes_l1666_166630

def otimes (x y : ℝ) : ℝ := x^3 - y^2 + x

theorem calculate_otimes (k : ℝ) : 
  otimes k (otimes k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end calculate_otimes_l1666_166630


namespace value_of_f_g_l1666_166692

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g (h₁ : f (g 3) = 35) (h₂ : g (f 3) = 11) : f (g 3) - g (f 3) = 24 :=
by
  calc
    f (g 3) - g (f 3) = 35 - 11 := by rw [h₁, h₂]
                      _         = 24 := by norm_num

end value_of_f_g_l1666_166692


namespace value_of_f_at_3_l1666_166687

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end value_of_f_at_3_l1666_166687


namespace find_number_l1666_166676

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l1666_166676


namespace triangle_DEF_rotate_180_D_l1666_166637

def rotate_180_degrees_clockwise (E D : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let ED := (D.1 - E.1, D.2 - E.2)
  (E.1 - ED.1, E.2 - ED.2)

theorem triangle_DEF_rotate_180_D (D E F : (ℝ × ℝ))
  (hD : D = (3, 2)) (hE : E = (6, 5)) (hF : F = (6, 2)) :
  rotate_180_degrees_clockwise E D = (9, 8) :=
by
  rw [hD, hE, rotate_180_degrees_clockwise]
  sorry

end triangle_DEF_rotate_180_D_l1666_166637


namespace ways_to_sum_2022_l1666_166688

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end ways_to_sum_2022_l1666_166688


namespace proof_problem_l1666_166621

noncomputable def A : Set ℝ := { x | x^2 - 4 = 0 }
noncomputable def B : Set ℝ := { y | ∃ x, y = x^2 - 4 }

theorem proof_problem :
  (A ∩ B = A) ∧ (A ∪ B = B) :=
by {
  sorry
}

end proof_problem_l1666_166621


namespace value_of_a_c_l1666_166691

theorem value_of_a_c {a b c d : ℝ} :
  (∀ x y : ℝ, y = -|x - a| + b → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) ∧
  (∀ x y : ℝ, y = |x - c| - d → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) →
  a + c = 8 :=
by
  sorry

end value_of_a_c_l1666_166691


namespace range_of_m_l1666_166685

def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B (m : ℝ) : Set ℝ := {x | abs (x - 3) ≤ m}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (m : ℝ) : Prop := x ∈ B m

theorem range_of_m (m : ℝ) (hm : m > 0):
  (∀ x, p x → q x m) ↔ (6 ≤ m) := by
  sorry

end range_of_m_l1666_166685


namespace larger_number_is_eight_l1666_166627

variable {x y : ℝ}

theorem larger_number_is_eight (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l1666_166627


namespace part1_part2_l1666_166693

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Prove part 1: For all x in ℝ, log(f(x, -8)) ≥ 1
theorem part1 : ∀ x : ℝ, Real.log (f x (-8)) ≥ 1 :=
by 
  sorry

-- Prove part 2: For all x in ℝ, if f(x,a) ≥ a, then a ≤ 1
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ a) → a ≤ 1 :=
by
  sorry

end part1_part2_l1666_166693


namespace contradictory_statement_of_p_l1666_166644

-- Given proposition p
def p : Prop := ∀ (x : ℝ), x + 3 ≥ 0 → x ≥ -3

-- Contradictory statement of p
noncomputable def contradictory_p : Prop := ∀ (x : ℝ), x + 3 < 0 → x < -3

-- Proof statement
theorem contradictory_statement_of_p : contradictory_p :=
sorry

end contradictory_statement_of_p_l1666_166644


namespace cannot_be_six_l1666_166668

theorem cannot_be_six (n r : ℕ) (h_n : n = 6) : 3 * n ≠ 4 * r :=
by
  sorry

end cannot_be_six_l1666_166668


namespace positive_integer_solutions_equation_l1666_166624

theorem positive_integer_solutions_equation (x y : ℕ) (positive_x : x > 0) (positive_y : y > 0) :
  x^2 + 6 * x * y - 7 * y^2 = 2009 ↔ (x = 252 ∧ y = 251) ∨ (x = 42 ∧ y = 35) ∨ (x = 42 ∧ y = 1) :=
sorry

end positive_integer_solutions_equation_l1666_166624


namespace radius_of_inscribed_circle_in_rhombus_l1666_166612

noncomputable def radius_of_inscribed_circle (d₁ d₂ : ℕ) : ℝ :=
  (d₁ * d₂) / (2 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

theorem radius_of_inscribed_circle_in_rhombus :
  radius_of_inscribed_circle 8 18 = 36 / Real.sqrt 97 :=
by
  -- Skip the detailed proof steps
  sorry

end radius_of_inscribed_circle_in_rhombus_l1666_166612


namespace min_value_fraction_8_l1666_166607

noncomputable def min_value_of_fraction (x y: ℝ) : Prop :=
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  x > 0 ∧ y > 0 ∧ parallel → (∀ z, z = (3 / x) + (2 / y) → z ≥ 8)

theorem min_value_fraction_8 (x y : ℝ) (h_posx : x > 0) (h_posy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  parallel → (3 / x) + (2 / y) ≥ 8 :=
by
  sorry

end min_value_fraction_8_l1666_166607


namespace sum_of_all_possible_N_l1666_166631

theorem sum_of_all_possible_N
  (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c = a + b)
  (h3 : N = a * b * c)
  (h4 : N = 6 * (a + b + c)) :
  N = 156 ∨ N = 96 ∨ N = 84 ∧
  (156 + 96 + 84 = 336) :=
by {
  -- proof will go here
  sorry
}

end sum_of_all_possible_N_l1666_166631


namespace length_of_DE_l1666_166667

theorem length_of_DE 
  (area_ABC : ℝ) 
  (area_trapezoid : ℝ) 
  (altitude_ABC : ℝ) 
  (h1 : area_ABC = 144) 
  (h2 : area_trapezoid = 96)
  (h3 : altitude_ABC = 24) :
  ∃ (DE_length : ℝ), DE_length = 2 * Real.sqrt 3 := 
sorry

end length_of_DE_l1666_166667


namespace determine_value_of_expression_l1666_166606

theorem determine_value_of_expression (x y : ℤ) (h : y^2 + 4 * x^2 * y^2 = 40 * x^2 + 817) : 4 * x^2 * y^2 = 3484 :=
sorry

end determine_value_of_expression_l1666_166606


namespace dosage_range_l1666_166699

theorem dosage_range (d : ℝ) (h : 60 ≤ d ∧ d ≤ 120) : 15 ≤ (d / 4) ∧ (d / 4) ≤ 30 :=
by
  sorry

end dosage_range_l1666_166699


namespace hawkeye_remaining_money_l1666_166686

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end hawkeye_remaining_money_l1666_166686


namespace problem_correct_answer_l1666_166657

theorem problem_correct_answer :
  (∀ (P L : Type) (passes_through_point : P → L → Prop) (parallel_to : L → L → Prop),
    (∀ (l₁ l₂ : L) (p : P), passes_through_point p l₁ ∧ ¬ passes_through_point p l₂ → (∃! l : L, passes_through_point p l ∧ parallel_to l l₂)) ->
  (∃ (l₁ l₂ : L) (A : P), passes_through_point A l₁ ∧ ¬ passes_through_point A l₂ ∧ ∃ l : L, passes_through_point A l ∧ parallel_to l l₂) ) :=
sorry

end problem_correct_answer_l1666_166657


namespace percentage_of_water_in_fresh_grapes_l1666_166696

theorem percentage_of_water_in_fresh_grapes
  (P : ℝ)  -- Let P be the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 5)  -- weight of fresh grapes in kg
  (dried_grapes_weight : ℝ := 0.625)  -- weight of dried grapes in kg
  (dried_water_percentage : ℝ := 20)  -- percentage of water in dried grapes
  (h1 : (100 - P) / 100 * fresh_grapes_weight = (100 - dried_water_percentage) / 100 * dried_grapes_weight) :
  P = 90 := 
sorry

end percentage_of_water_in_fresh_grapes_l1666_166696


namespace sufficient_but_not_necessary_condition_for_negative_root_l1666_166643

def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem sufficient_but_not_necessary_condition_for_negative_root 
  (a : ℝ) (h : a < 0) : 
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) ∧ 
  (∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) → a ≤ 0) :=
sorry

end sufficient_but_not_necessary_condition_for_negative_root_l1666_166643


namespace inequality_solution_set_l1666_166664

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (2 * x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} 
  = {x : ℝ | (0 < x ∧ x ≤ 1/5) ∨ (2 < x ∧ x ≤ 6)} := 
by {
  sorry
}

end inequality_solution_set_l1666_166664


namespace minimum_value_exists_l1666_166614

noncomputable def minimized_function (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

theorem minimum_value_exists :
  ∃ (x y : ℝ), minimized_function x y = minimized_function (4/3 - 2 * y/3) y :=
sorry

end minimum_value_exists_l1666_166614


namespace Lin_trip_time_l1666_166656

theorem Lin_trip_time
  (v : ℕ) -- speed on the mountain road in miles per minute
  (h1 : 80 = d_highway) -- Lin travels 80 miles on the highway
  (h2 : 20 = d_mountain) -- Lin travels 20 miles on the mountain road
  (h3 : v_highway = 2 * v) -- Lin drives twice as fast on the highway
  (h4 : 40 = 20 / v) -- Lin spent 40 minutes driving on the mountain road
  : 40 + 80 = 120 :=
by
  -- proof steps would go here
  sorry

end Lin_trip_time_l1666_166656


namespace expected_americans_with_allergies_l1666_166613

theorem expected_americans_with_allergies (prob : ℚ) (sample_size : ℕ) (h_prob : prob = 1/5) (h_sample_size : sample_size = 250) :
  sample_size * prob = 50 := by
  rw [h_prob, h_sample_size]
  norm_num

#print expected_americans_with_allergies

end expected_americans_with_allergies_l1666_166613


namespace reimbursement_correct_l1666_166654

-- Define the days and miles driven each day
def miles_monday : ℕ := 18
def miles_tuesday : ℕ := 26
def miles_wednesday : ℕ := 20
def miles_thursday : ℕ := 20
def miles_friday : ℕ := 16

-- Define the mileage rate
def mileage_rate : ℝ := 0.36

-- Define the total miles driven
def total_miles_driven : ℕ := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday

-- Define the total reimbursement
def reimbursement : ℝ := total_miles_driven * mileage_rate

-- Prove that the reimbursement is $36
theorem reimbursement_correct : reimbursement = 36 := by
  sorry

end reimbursement_correct_l1666_166654


namespace fraction_meaningful_l1666_166683

theorem fraction_meaningful (x : ℝ) : (x ≠ -1) ↔ (∃ k : ℝ, k = 1 / (x + 1)) :=
by
  sorry

end fraction_meaningful_l1666_166683


namespace kayla_score_fourth_level_l1666_166642

theorem kayla_score_fourth_level 
  (score1 score2 score3 score5 score6 : ℕ) 
  (h1 : score1 = 2) 
  (h2 : score2 = 3) 
  (h3 : score3 = 5) 
  (h5 : score5 = 12) 
  (h6 : score6 = 17)
  (h_diff : ∀ n : ℕ, score2 - score1 + n = score3 - score2 + n + 1 ∧ score3 - score2 + n + 2 = score5 - score3 + n + 3 ∧ score5 - score3 + n + 4 = score6 - score5 + n + 5) :
  ∃ score4 : ℕ, score4 = 8 :=
by
  sorry

end kayla_score_fourth_level_l1666_166642


namespace lcm_of_two_numbers_hcf_and_product_l1666_166602

theorem lcm_of_two_numbers_hcf_and_product (a b : ℕ) (h_hcf : Nat.gcd a b = 20) (h_prod : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_of_two_numbers_hcf_and_product_l1666_166602


namespace binary_multiplication_l1666_166608

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l1666_166608


namespace inequality_and_equality_condition_l1666_166641

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + 4 * b^2 + 4 * b - 4 * a + 5 ≥ 0 ∧ (a^2 + 4 * b^2 + 4 * b - 4 * a + 5 = 0 ↔ (a = 2 ∧ b = -1 / 2)) :=
by
  sorry

end inequality_and_equality_condition_l1666_166641


namespace S_30_zero_l1666_166671

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {n : ℕ} 

-- Definitions corresponding to the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a_n n = a1 + d * n

def sum_arithmetic_sequence (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
  
-- The given conditions
axiom S_eq (S_10 S_20 : ℝ) : S 10 = S 20

-- The theorem we need to prove
theorem S_30_zero (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_arithmetic_sequence S a_n)
  (h_eq : S 10 = S 20) :
  S 30 = 0 :=
sorry

end S_30_zero_l1666_166671


namespace total_people_ball_l1666_166682

theorem total_people_ball (n m : ℕ) (h1 : n + m < 50) (h2 : 3 * n = 20 * m) : n + m = 41 := 
sorry

end total_people_ball_l1666_166682


namespace arithmetic_sequence_term_l1666_166653

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Conditions
def common_difference := d = 2
def value_a_2007 := a 2007 = 2007

-- Question to be proved
theorem arithmetic_sequence_term :
  common_difference d →
  value_a_2007 a →
  a 2009 = 2011 :=
by
  sorry

end arithmetic_sequence_term_l1666_166653


namespace marcus_has_210_cards_l1666_166698

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l1666_166698


namespace smallest_x_l1666_166618

theorem smallest_x : ∃ x : ℕ, x + 6721 ≡ 3458 [MOD 12] ∧ x % 5 = 0 ∧ x = 45 :=
by
  sorry

end smallest_x_l1666_166618


namespace temperature_relationship_l1666_166658

def temperature (t : ℕ) (T : ℕ) :=
  ∀ t < 10, T = 7 * t + 30

-- Proof not required, hence added sorry.
theorem temperature_relationship (t : ℕ) (T : ℕ) (h : t < 10) :
  temperature t T :=
by {
  sorry
}

end temperature_relationship_l1666_166658


namespace shop_owner_percentage_profit_l1666_166660

theorem shop_owner_percentage_profit
  (cp : ℝ)  -- cost price of 1 kg
  (cheat_buy : ℝ) -- cheat percentage when buying
  (cheat_sell : ℝ) -- cheat percentage when selling
  (h_cp : cp = 100) -- cost price is $100
  (h_cheat_buy : cheat_buy = 15) -- cheat by 15% when buying
  (h_cheat_sell : cheat_sell = 20) -- cheat by 20% when selling
  :
  let weight_bought := 1 + (cheat_buy / 100)
  let weight_sold := 1 - (cheat_sell / 100)
  let real_selling_price_per_kg := cp / weight_sold
  let total_selling_price := weight_bought * real_selling_price_per_kg
  let profit := total_selling_price - cp
  let percentage_profit := (profit / cp) * 100
  percentage_profit = 43.75 := 
by
  sorry

end shop_owner_percentage_profit_l1666_166660


namespace tangent_line_m_value_l1666_166635

theorem tangent_line_m_value : 
  (∀ m : ℝ, ∃ (x y : ℝ), (x = my + 2) ∧ (x + one)^2 + (y + one)^2 = 2) → 
  (m = 1 ∨ m = -7) :=
  sorry

end tangent_line_m_value_l1666_166635


namespace exists_base_and_digit_l1666_166648

def valid_digit_in_base (B : ℕ) (V : ℕ) : Prop :=
  V^2 % B = V ∧ V ≠ 0 ∧ V ≠ 1

theorem exists_base_and_digit :
  ∃ B V, valid_digit_in_base B V :=
by {
  sorry
}

end exists_base_and_digit_l1666_166648


namespace anton_has_more_cards_than_ann_l1666_166603

-- Define Heike's number of cards
def heike_cards : ℕ := 60

-- Define Anton's number of cards in terms of Heike's cards
def anton_cards (H : ℕ) : ℕ := 3 * H

-- Define Ann's number of cards as equal to Heike's cards
def ann_cards (H : ℕ) : ℕ := H

-- Theorem statement
theorem anton_has_more_cards_than_ann 
  (H : ℕ) (H_equals : H = heike_cards) : 
  anton_cards H - ann_cards H = 120 :=
by
  -- At this point, the actual proof would be inserted.
  sorry

end anton_has_more_cards_than_ann_l1666_166603


namespace polygon_sides_eq_2023_l1666_166620

theorem polygon_sides_eq_2023 (n : ℕ) (h : n - 2 = 2021) : n = 2023 :=
sorry

end polygon_sides_eq_2023_l1666_166620


namespace simplify_evaluate_expr_l1666_166675

theorem simplify_evaluate_expr (x : ℕ) (h : x = 2023) : (x + 1) ^ 2 - x * (x + 1) = 2024 := 
by 
  sorry

end simplify_evaluate_expr_l1666_166675


namespace starting_positions_P0_P1024_l1666_166610

noncomputable def sequence_fn (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

def find_starting_positions (n : ℕ) : ℕ := 2^n - 2

theorem starting_positions_P0_P1024 :
  ∃ P0 : ℝ, ∀ n : ℕ, P0 = sequence_fn^[n] P0 → P0 = sequence_fn^[1024] P0 ↔ find_starting_positions 1024 = 2^1024 - 2 :=
sorry

end starting_positions_P0_P1024_l1666_166610


namespace fractional_exponent_representation_of_sqrt_l1666_166652

theorem fractional_exponent_representation_of_sqrt (a : ℝ) : 
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3 / 4) := 
sorry

end fractional_exponent_representation_of_sqrt_l1666_166652


namespace log_expression_value_l1666_166695

theorem log_expression_value : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  -- Assuming necessary properties and steps are already known and prove the theorem accordingly:
  sorry

end log_expression_value_l1666_166695


namespace solve_problem_l1666_166659

theorem solve_problem
    (product_trailing_zeroes : ∃ (x y z w v u p q r : ℕ), (10 ∣ (x * y * z * w * v * u * p * q * r)) ∧ B = 0)
    (digit_sequences : (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) % 10 = 8 ∧
                       (11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19) % 10 = 4 ∧
                       (21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29) % 10 = 4 ∧
                       (31 * 32 * 33 * 34 * 35) % 10 = 4 ∧
                       A = 2 ∧ B = 0)
    (divisibility_rule_11 : ∀ C D, (71 + C) - (68 + D) = 11 → C - D = -3 ∨ C - D = 8)
    (divisibility_rule_9 : ∀ C D, (139 + C + D) % 9 = 0 → C + D = 5 ∨ C + D = 14)
    (system_of_equations : ∀ C D, (C - D = -3 ∧ C + D = 5) → (C = 1 ∧ D = 4)) :
  A = 2 ∧ B = 0 ∧ C = 1 ∧ D = 4 :=
by
  sorry

end solve_problem_l1666_166659


namespace union_sets_l1666_166611

noncomputable def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
noncomputable def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem union_sets (A : Set ℝ) (B : Set ℝ) : (A ∪ B = C) := by
  sorry

end union_sets_l1666_166611


namespace min_bn_of_arithmetic_sequence_l1666_166677

theorem min_bn_of_arithmetic_sequence :
  (∃ n : ℕ, 1 ≤ n ∧ b_n = n + 1 + 7 / n ∧ (∀ m : ℕ, 1 ≤ m → b_m ≥ b_n)) :=
sorry

def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

def S_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n + 1) / 2

def b_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else (2 * S_n n + 7) / n

end min_bn_of_arithmetic_sequence_l1666_166677


namespace abs_sum_inequality_for_all_x_l1666_166678

theorem abs_sum_inequality_for_all_x (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ (m ≤ 3) :=
by
  sorry

end abs_sum_inequality_for_all_x_l1666_166678


namespace exist_a_b_not_triangle_l1666_166697

theorem exist_a_b_not_triangle (h₁ : ∀ a b : ℕ, (a > 1000) → (b > 1000) →
  ∃ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  ∃ (a b : ℕ), (a > 1000 ∧ b > 1000) ∧ 
  ∀ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
sorry

end exist_a_b_not_triangle_l1666_166697


namespace equilateral_triangle_circumradius_ratio_l1666_166651

variables (B b S s : ℝ)

-- Given two equilateral triangles with side lengths B and b, and respectively circumradii S and s
-- B and b are not equal
-- Prove that S / s = B / b
theorem equilateral_triangle_circumradius_ratio (hBneqb : B ≠ b)
  (hS : S = B * Real.sqrt 3 / 3)
  (hs : s = b * Real.sqrt 3 / 3) : S / s = B / b :=
by
  sorry

end equilateral_triangle_circumradius_ratio_l1666_166651


namespace mariela_cards_l1666_166617

theorem mariela_cards (cards_after_home : ℕ) (total_cards : ℕ) (cards_in_hospital : ℕ) : 
  cards_after_home = 287 → 
  total_cards = 690 → 
  cards_in_hospital = total_cards - cards_after_home → 
  cards_in_hospital = 403 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3


end mariela_cards_l1666_166617


namespace original_inhabitants_l1666_166633

theorem original_inhabitants (X : ℝ) (h : 0.75 * 0.9 * X = 5265) : X = 7800 :=
by
  sorry

end original_inhabitants_l1666_166633


namespace farmer_apples_count_l1666_166663

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l1666_166663


namespace sequence_property_l1666_166689

theorem sequence_property (a : ℕ → ℝ)
    (h_rec : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
    (h_a1 : a 1 = 1 + Real.sqrt 7)
    (h_1776 : a 1776 = 13 + Real.sqrt 7) :
    a 2009 = -1 + 2 * Real.sqrt 7 := 
    sorry

end sequence_property_l1666_166689


namespace monthly_salary_l1666_166690

variables (S : ℕ) (h1 : S * 20 / 100 * 96 / 100 = 4 * 250)

theorem monthly_salary : S = 6250 :=
by sorry

end monthly_salary_l1666_166690


namespace matt_profit_trade_l1666_166645

theorem matt_profit_trade
  (total_cards : ℕ := 8)
  (value_per_card : ℕ := 6)
  (traded_cards_count : ℕ := 2)
  (trade_value_per_card : ℕ := 6)
  (received_cards_count_1 : ℕ := 3)
  (received_value_per_card_1 : ℕ := 2)
  (received_cards_count_2 : ℕ := 1)
  (received_value_per_card_2 : ℕ := 9)
  (profit : ℕ := 3) :
  profit = (received_cards_count_1 * received_value_per_card_1 
           + received_cards_count_2 * received_value_per_card_2) 
           - (traded_cards_count * trade_value_per_card) :=
  by
  sorry

end matt_profit_trade_l1666_166645


namespace circle_center_transformation_l1666_166623

def original_center : ℤ × ℤ := (3, -4)

def reflect_x_axis (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

def translate_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1 + d, p.2)

def final_center : ℤ × ℤ := (8, 4)

theorem circle_center_transformation :
  translate_right (reflect_x_axis original_center) 5 = final_center :=
by
  sorry

end circle_center_transformation_l1666_166623


namespace dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l1666_166674

-- Question 1
theorem dual_expr_result (m n : ℝ) (h1 : m = 2 - Real.sqrt 3) (h2 : n = 2 + Real.sqrt 3) :
  m * n = 1 :=
sorry

-- Question 2
theorem solve_sqrt_eq_16 (x : ℝ) (h : Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) :
  x = 39 :=
sorry

-- Question 3
theorem solve_sqrt_rational_eq_4x (x : ℝ) (h : Real.sqrt (4 * x^2 + 6 * x - 5) + Real.sqrt (4 * x^2 - 2 * x - 5) = 4 * x) :
  x = 3 :=
sorry

end dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l1666_166674


namespace trig_second_quadrant_l1666_166679

theorem trig_second_quadrant (α : ℝ) (h1 : α > π / 2) (h2 : α < π) :
  (|Real.sin α| / Real.sin α) - (|Real.cos α| / Real.cos α) = 2 :=
by
  sorry

end trig_second_quadrant_l1666_166679


namespace triangle_angle_bisector_segment_length_l1666_166680

theorem triangle_angle_bisector_segment_length
  (DE DF EF DG EG : ℝ)
  (h_ratio : DE / 12 = 1 ∧ DF / DE = 4 / 3 ∧ EF / DE = 5 / 3)
  (h_angle_bisector : DG / EG = DE / DF ∧ DG + EG = EF) :
  EG = 80 / 7 :=
by
  sorry

end triangle_angle_bisector_segment_length_l1666_166680


namespace highest_score_is_151_l1666_166681

-- Definitions for the problem conditions
def total_runs : ℕ := 2704
def total_runs_excluding_HL : ℕ := 2552

variables (H L : ℕ) 

-- Problem conditions as hypotheses
axiom h1 : H - L = 150
axiom h2 : H + L = 152
axiom h3 : 2704 = 2552 + H + L

-- Proof statement
theorem highest_score_is_151 (H L : ℕ) (h1 : H - L = 150) (h2 : H + L = 152) (h3 : 2704 = 2552 + H + L) : H = 151 :=
by sorry

end highest_score_is_151_l1666_166681


namespace sufficient_but_not_necessary_condition_l1666_166629

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a = 1) (h2 : |a| = 1) : 
  (a = 1 → |a| = 1) ∧ ¬(|a| = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1666_166629


namespace solution_to_problem_l1666_166669

theorem solution_to_problem (x y : ℕ) (h : (2*x - 5) * (2*y - 5) = 25) : x + y = 10 ∨ x + y = 18 := by
  sorry

end solution_to_problem_l1666_166669


namespace cone_volume_filled_88_8900_percent_l1666_166634

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l1666_166634


namespace isosceles_triangle_leg_length_l1666_166601

-- Define the necessary condition for the isosceles triangle
def isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a + b + c = 16 ∧ a = 4

-- State the theorem we want to prove
theorem isosceles_triangle_leg_length :
  ∃ (b c : ℕ), isosceles_triangle 4 b c ∧ b = 6 :=
by
  -- Formal proof will be provided here
  sorry

end isosceles_triangle_leg_length_l1666_166601


namespace equivalent_expr_l1666_166672

theorem equivalent_expr (a y : ℝ) (ha : a ≠ 0) (hy : y ≠ a ∧ y ≠ -a) :
  ( (a / (a + y) + y / (a - y)) / ( y / (a + y) - a / (a - y)) ) = -1 :=
by
  sorry

end equivalent_expr_l1666_166672


namespace complex_quadrant_l1666_166670

open Complex

theorem complex_quadrant
  (z1 z2 z : ℂ) (h1 : z1 = 2 + I) (h2 : z2 = 1 - I) (h3 : z = z1 / z2) :
  0 < z.re ∧ 0 < z.im :=
by
  -- sorry to skip the proof steps
  sorry

end complex_quadrant_l1666_166670


namespace range_M_l1666_166619

theorem range_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  1 < (1 / (1 + a)) + (1 / (1 + b)) ∧ (1 / (1 + a)) + (1 / (1 + b)) < 2 := by
  sorry

end range_M_l1666_166619


namespace part1_purchase_price_part2_minimum_A_l1666_166628

section
variables (x y m : ℝ)

-- Part 1: Purchase price per piece
theorem part1_purchase_price (h1 : 10 * x + 15 * y = 3600) (h2 : 25 * x + 30 * y = 8100) :
  x = 180 ∧ y = 120 :=
sorry

-- Part 2: Minimum number of model A bamboo mats
theorem part2_minimum_A (h3 : x = 180) (h4 : y = 120) 
    (h5 : (260 - x) * m + (180 - y) * (60 - m) ≥ 4400) : 
  m ≥ 40 :=
sorry
end

end part1_purchase_price_part2_minimum_A_l1666_166628


namespace nate_total_distance_l1666_166605

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l1666_166605


namespace find_a_l1666_166650

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

theorem find_a (a : ℝ) (h : (deriv (f a)) e = 3) : a = 3 / 2 :=
by
-- placeholder for the proof
sorry

end find_a_l1666_166650


namespace updated_mean_of_decremented_observations_l1666_166636

theorem updated_mean_of_decremented_observations (n : ℕ) (initial_mean decrement : ℝ)
  (h₀ : n = 50) (h₁ : initial_mean = 200) (h₂ : decrement = 6) :
  ((n * initial_mean) - (n * decrement)) / n = 194 := by
  sorry

end updated_mean_of_decremented_observations_l1666_166636


namespace find_5_digit_number_l1666_166625

theorem find_5_digit_number {A B C D E : ℕ} 
  (hA_even : A % 2 = 0) 
  (hB_even : B % 2 = 0) 
  (hA_half_B : A = B / 2) 
  (hC_sum : C = A + B) 
  (hDE_prime : Prime (10 * D + E)) 
  (hD_3B : D = 3 * B) : 
  10000 * A + 1000 * B + 100 * C + 10 * D + E = 48247 := 
sorry

end find_5_digit_number_l1666_166625


namespace truncated_cone_sphere_radius_l1666_166600

structure TruncatedCone :=
(base_radius_top : ℝ)
(base_radius_bottom : ℝ)

noncomputable def sphere_radius (c : TruncatedCone) : ℝ :=
  if c.base_radius_top = 24 ∧ c.base_radius_bottom = 6 then 12 else 0

theorem truncated_cone_sphere_radius (c : TruncatedCone) (h_radii : c.base_radius_top = 24 ∧ c.base_radius_bottom = 6) :
  sphere_radius c = 12 :=
by
  sorry

end truncated_cone_sphere_radius_l1666_166600


namespace trig_triple_angle_l1666_166639

theorem trig_triple_angle (θ : ℝ) (h : Real.tan θ = 5) :
  Real.tan (3 * θ) = 55 / 37 ∧
  Real.sin (3 * θ) = 55 * Real.sqrt 1369 / (37 * Real.sqrt 4394) ∨ Real.sin (3 * θ) = -(55 * Real.sqrt 1369 / (37 * Real.sqrt 4394)) ∧
  Real.cos (3 * θ) = Real.sqrt (1369 / 4394) ∨ Real.cos (3 * θ) = -Real.sqrt (1369 / 4394) :=
by
  sorry

end trig_triple_angle_l1666_166639


namespace triangle_ABC_no_common_factor_l1666_166638

theorem triangle_ABC_no_common_factor (a b c : ℕ) (h_coprime: Nat.gcd (Nat.gcd a b) c = 1)
  (h_angleB_eq_2angleC : True) (h_b_lt_600 : b < 600) : False :=
by
  sorry

end triangle_ABC_no_common_factor_l1666_166638


namespace factorize_expression_l1666_166662

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end factorize_expression_l1666_166662


namespace minimum_distance_after_9_minutes_l1666_166647

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l1666_166647


namespace mary_picked_nine_lemons_l1666_166640

def num_lemons_sally := 7
def total_num_lemons := 16
def num_lemons_mary := total_num_lemons - num_lemons_sally

theorem mary_picked_nine_lemons :
  num_lemons_mary = 9 := by
  sorry

end mary_picked_nine_lemons_l1666_166640


namespace sin_value_l1666_166673

theorem sin_value (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 :=
by
  sorry

end sin_value_l1666_166673


namespace xy_square_diff_l1666_166616

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l1666_166616


namespace find_d_l1666_166665

-- Define the conditions
variables (x₀ y₀ c : ℝ)

-- Define the system of equations
def system_of_equations : Prop :=
  x₀ * y₀ = 6 ∧ x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2

-- Define the target proof problem
theorem find_d (h : system_of_equations x₀ y₀ c) : x₀^2 + y₀^2 = 69 :=
sorry

end find_d_l1666_166665


namespace darryl_books_l1666_166604

variable (l m d : ℕ)

theorem darryl_books (h1 : l + m + d = 97) (h2 : l = m - 3) (h3 : m = 2 * d) : d = 20 := 
by
  sorry

end darryl_books_l1666_166604
