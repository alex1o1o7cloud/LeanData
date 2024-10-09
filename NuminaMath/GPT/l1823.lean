import Mathlib

namespace max_value_of_M_l1823_182339

noncomputable def M (x y z : ℝ) := min (min x y) z

theorem max_value_of_M
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_zero : b^2 - 4 * a * c ≥ 0) :
  M ((b + c) / a) ((c + a) / b) ((a + b) / c) ≤ 5 / 4 :=
sorry

end max_value_of_M_l1823_182339


namespace increase_80_by_150_percent_l1823_182333

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l1823_182333


namespace find_DY_length_l1823_182343

noncomputable def angle_bisector_theorem (DE DY EF FY : ℝ) : ℝ :=
  (DE * FY) / EF

theorem find_DY_length :
  ∀ (DE EF FY : ℝ), DE = 26 → EF = 34 → FY = 30 →
  angle_bisector_theorem DE DY EF FY = 22.94 := 
by
  intros
  sorry

end find_DY_length_l1823_182343


namespace simplify_and_evaluate_l1823_182369

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_l1823_182369


namespace sample_size_l1823_182300

theorem sample_size (T : ℕ) (f_C : ℚ) (samples_C : ℕ) (n : ℕ) 
    (hT : T = 260)
    (hfC : f_C = 3 / 13)
    (hsamples_C : samples_C = 3) : n = 13 :=
by
  -- Proof goes here
  sorry

end sample_size_l1823_182300


namespace football_cost_is_correct_l1823_182368

def total_spent_on_toys : ℝ := 12.30
def spent_on_marbles : ℝ := 6.59
def spent_on_football := total_spent_on_toys - spent_on_marbles

theorem football_cost_is_correct : spent_on_football = 5.71 :=
by
  sorry

end football_cost_is_correct_l1823_182368


namespace no_7_edges_edges_greater_than_5_l1823_182397

-- Define the concept of a convex polyhedron in terms of its edges and faces.
structure ConvexPolyhedron where
  V : ℕ    -- Number of vertices
  E : ℕ    -- Number of edges
  F : ℕ    -- Number of faces
  Euler : V - E + F = 2   -- Euler's characteristic

-- Define properties of convex polyhedron

-- Part (a) statement: A convex polyhedron cannot have exactly 7 edges.
theorem no_7_edges (P : ConvexPolyhedron) : P.E ≠ 7 :=
sorry

-- Part (b) statement: A convex polyhedron can have any number of edges greater than 5 and different from 7.
theorem edges_greater_than_5 (n : ℕ) (h : n > 5) (h2 : n ≠ 7) : ∃ P : ConvexPolyhedron, P.E = n :=
sorry

end no_7_edges_edges_greater_than_5_l1823_182397


namespace maximize_profit_l1823_182302

variables (a x : ℝ) (t : ℝ := 5 - 12 / (x + 3)) (cost : ℝ := 10 + 2 * t) 
  (price : ℝ := 5 + 20 / t) (profit : ℝ := 2 * (price * t - cost - x))

-- Assume non-negativity and upper bound on promotional cost
variable (h_a_nonneg : 0 ≤ a)
variable (h_a_pos : 0 < a)

noncomputable def profit_function (x : ℝ) : ℝ := 20 - 4 / x - x

-- Prove the maximum promotional cost that maximizes the profit
theorem maximize_profit : 
  (if a ≥ 2 then ∃ y, y = 2 ∧ profit_function y = profit_function 2 
   else ∃ y, y = a ∧ profit_function y = profit_function a) := 
sorry

end maximize_profit_l1823_182302


namespace difference_of_squares_example_l1823_182326

theorem difference_of_squares_example (a b : ℕ) (h₁ : a = 650) (h₂ : b = 350) :
  a^2 - b^2 = 300000 :=
by
  sorry

end difference_of_squares_example_l1823_182326


namespace problem_3_div_27_l1823_182351

theorem problem_3_div_27 (a b : ℕ) (h : 2^a = 8^(b + 1)) : 3^a / 27^b = 27 := by
  -- proof goes here
  sorry

end problem_3_div_27_l1823_182351


namespace smallest_pos_integer_l1823_182305

-- Definitions based on the given conditions
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Given conditions
def condition1 (a1 d : ℤ) : Prop := arithmetic_seq a1 d 11 - arithmetic_seq a1 d 8 = 3
def condition2 (a1 d : ℤ) : Prop := sum_seq a1 d 11 - sum_seq a1 d 8 = 3

-- The claim we want to prove
theorem smallest_pos_integer 
  (n : ℕ) (a1 d : ℤ) 
  (h1 : condition1 a1 d) 
  (h2 : condition2 a1 d) : n = 10 :=
by
  sorry

end smallest_pos_integer_l1823_182305


namespace find_interest_rate_l1823_182308

theorem find_interest_rate (P : ℕ) (diff : ℕ) (T : ℕ) (I2_rate : ℕ) (r : ℚ) 
  (hP : P = 15000) (hdiff : diff = 900) (hT : T = 2) (hI2_rate : I2_rate = 12)
  (h : P * (r / 100) * T = P * (I2_rate / 100) * T + diff) :
  r = 15 :=
sorry

end find_interest_rate_l1823_182308


namespace proof_problem_l1823_182313

variable {a1 a2 b1 b2 b3 : ℝ}

-- Condition: -2, a1, a2, -8 form an arithmetic sequence
def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = -2 / 3 * (-2 - 8)

-- Condition: -2, b1, b2, b3, -8 form a geometric sequence
def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  b2^2 = (-2) * (-8) ∧ b1^2 = (-2) * b2 ∧ b3^2 = b2 * (-8)

theorem proof_problem (h1 : arithmetic_sequence a1 a2) (h2 : geometric_sequence b1 b2 b3) : b2 * (a2 - a1) = 8 :=
by
  admit -- Convert to sorry to skip the proof

end proof_problem_l1823_182313


namespace wrapping_paper_fraction_used_l1823_182365

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l1823_182365


namespace largest_non_formable_amount_l1823_182344

-- Definitions and conditions from the problem
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def cannot_be_formed (n a b : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ a * x + b * y

-- The statement to prove
theorem largest_non_formable_amount :
  is_coprime 8 15 ∧ cannot_be_formed 97 8 15 :=
by
  sorry

end largest_non_formable_amount_l1823_182344


namespace sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l1823_182331

def seq1 (n : ℕ) : ℕ := 2 * (n + 1)
def seq2 (n : ℕ) : ℕ := 3 * 2 ^ n
def seq3 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 36 + n
  else 10 + n
  
theorem sequence1_sixth_seventh_terms :
  seq1 5 = 12 ∧ seq1 6 = 14 :=
by
  sorry

theorem sequence2_sixth_term :
  seq2 5 = 96 :=
by
  sorry

theorem sequence3_ninth_tenth_terms :
  seq3 8 = 44 ∧ seq3 9 = 19 :=
by
  sorry

end sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l1823_182331


namespace add_num_denom_fraction_l1823_182352

theorem add_num_denom_fraction (n : ℚ) : (2 + n) / (7 + n) = 3 / 5 ↔ n = 11 / 2 := 
by
  sorry

end add_num_denom_fraction_l1823_182352


namespace total_amount_l1823_182360

noncomputable def A : ℝ := 360.00000000000006
noncomputable def B : ℝ := (3/2) * A
noncomputable def C : ℝ := 4 * B

theorem total_amount (A B C : ℝ)
  (hA : A = 360.00000000000006)
  (hA_B : A = (2/3) * B)
  (hB_C : B = (1/4) * C) :
  A + B + C = 3060.0000000000007 := by
  sorry

end total_amount_l1823_182360


namespace hexagon_side_lengths_l1823_182336

theorem hexagon_side_lengths (n : ℕ) (h1 : n ≥ 0) (h2 : n ≤ 6) (h3 : 10 * n + 8 * (6 - n) = 56) : n = 4 :=
sorry

end hexagon_side_lengths_l1823_182336


namespace squirrel_cones_l1823_182304

theorem squirrel_cones :
  ∃ (x y : ℕ), 
    x + y < 25 ∧ 
    2 * x > y + 26 ∧ 
    2 * y > x - 4 ∧
    x = 17 ∧ 
    y = 7 :=
by
  sorry

end squirrel_cones_l1823_182304


namespace other_root_of_quadratic_l1823_182396

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, x^2 + a * x - 2 = 0 → x = -1) → ∃ m, x = m ∧ m = 2 :=
by
  sorry

end other_root_of_quadratic_l1823_182396


namespace product_of_third_side_l1823_182388

/-- Two sides of a right triangle have lengths 5 and 7. The product of the possible lengths of 
the third side is exactly √1776. -/
theorem product_of_third_side :
  let a := 5
  let b := 7
  (Real.sqrt (a^2 + b^2) * Real.sqrt (b^2 - a^2)) = Real.sqrt 1776 := 
by 
  let a := 5
  let b := 7
  sorry

end product_of_third_side_l1823_182388


namespace remainder_7623_div_11_l1823_182384

theorem remainder_7623_div_11 : 7623 % 11 = 0 := 
by sorry

end remainder_7623_div_11_l1823_182384


namespace complete_the_square_l1823_182323

theorem complete_the_square (x : ℝ) (h : x^2 - 8 * x - 1 = 0) : (x - 4)^2 = 17 :=
by
  -- proof steps would go here, but we use sorry for now
  sorry

end complete_the_square_l1823_182323


namespace average_age_of_guardians_and_fourth_graders_l1823_182387

theorem average_age_of_guardians_and_fourth_graders (num_fourth_graders num_guardians : ℕ)
  (avg_age_fourth_graders avg_age_guardians : ℕ)
  (h1 : num_fourth_graders = 40)
  (h2 : avg_age_fourth_graders = 10)
  (h3 : num_guardians = 60)
  (h4 : avg_age_guardians = 35)
  : (num_fourth_graders * avg_age_fourth_graders + num_guardians * avg_age_guardians) / (num_fourth_graders + num_guardians) = 25 :=
by
  sorry

end average_age_of_guardians_and_fourth_graders_l1823_182387


namespace Q_share_of_profit_l1823_182389

def P_investment : ℕ := 54000
def Q_investment : ℕ := 36000
def total_profit : ℕ := 18000

theorem Q_share_of_profit : Q_investment * total_profit / (P_investment + Q_investment) = 7200 := by
  sorry

end Q_share_of_profit_l1823_182389


namespace exists_negative_root_of_P_l1823_182355

def P(x : ℝ) : ℝ := x^7 - 2 * x^6 - 7 * x^4 - x^2 + 10

theorem exists_negative_root_of_P : ∃ x : ℝ, x < 0 ∧ P x = 0 :=
sorry

end exists_negative_root_of_P_l1823_182355


namespace woman_finishes_work_in_225_days_l1823_182372

theorem woman_finishes_work_in_225_days
  (M W : ℝ)
  (h1 : (10 * M + 15 * W) * 6 = 1)
  (h2 : M * 100 = 1) :
  1 / W = 225 :=
by
  sorry

end woman_finishes_work_in_225_days_l1823_182372


namespace ribbon_cuts_l1823_182319

theorem ribbon_cuts (rolls : ℕ) (length_per_roll : ℕ) (piece_length : ℕ) (total_rolls : rolls = 5) (roll_length : length_per_roll = 50) (piece_size : piece_length = 2) : 
  (rolls * ((length_per_roll / piece_length) - 1) = 120) :=
by
  sorry

end ribbon_cuts_l1823_182319


namespace find_z_l1823_182335

open Complex

theorem find_z (z : ℂ) (h : ((1 - I) ^ 2) / z = 1 + I) : z = -1 - I :=
sorry

end find_z_l1823_182335


namespace minimum_possible_value_of_Box_l1823_182328

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l1823_182328


namespace min_marks_required_l1823_182395

-- Definitions and conditions
def grid_size := 7
def strip_size := 4

-- Question and answer as a proof statement
theorem min_marks_required (n : ℕ) (h : grid_size = 2 * n - 1) : 
  (∃ marks : ℕ, 
    (∀ row col : ℕ, 
      row < grid_size → col < grid_size → 
      (∃ i j : ℕ, 
        i < strip_size → j < strip_size → 
        (marks ≥ 12)))) :=
sorry

end min_marks_required_l1823_182395


namespace max_students_per_class_l1823_182340

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l1823_182340


namespace amount_paid_is_correct_l1823_182375

-- Conditions given in the problem
def jimmy_shorts_count : ℕ := 3
def jimmy_short_price : ℝ := 15.0
def irene_shirts_count : ℕ := 5
def irene_shirt_price : ℝ := 17.0
def discount_rate : ℝ := 0.10

-- Define the total cost for jimmy
def jimmy_total_cost : ℝ := jimmy_shorts_count * jimmy_short_price

-- Define the total cost for irene
def irene_total_cost : ℝ := irene_shirts_count * irene_shirt_price

-- Define the total cost before discount
def total_cost_before_discount : ℝ := jimmy_total_cost + irene_total_cost

-- Define the discount amount
def discount_amount : ℝ := total_cost_before_discount * discount_rate

-- Define the total amount to pay
def total_amount_to_pay : ℝ := total_cost_before_discount - discount_amount

-- The proposition we need to prove
theorem amount_paid_is_correct : total_amount_to_pay = 117 := by
  sorry

end amount_paid_is_correct_l1823_182375


namespace value_of_x2017_l1823_182316

-- Definitions and conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

variables (f : ℝ → ℝ) (x : ℕ → ℝ)
variables (d : ℝ)
variable (h_odd : is_odd_function f)
variable (h_increasing : is_increasing_function f)
variable (h_arithmetic : arithmetic_sequence x 2)
variable (h_condition : f (x 7) + f (x 8) = 0)

-- Define the proof goal
theorem value_of_x2017 : x 2017 = 4019 :=
by
  sorry

end value_of_x2017_l1823_182316


namespace pudding_cups_initial_l1823_182386

theorem pudding_cups_initial (P : ℕ) (students : ℕ) (extra_cups : ℕ) 
  (h1 : students = 218) (h2 : extra_cups = 121) (h3 : P + extra_cups = students) : P = 97 := 
by
  sorry

end pudding_cups_initial_l1823_182386


namespace min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l1823_182330

-- Definitions for the problem conditions
def initial_points : ℕ := 52
def record_points : ℕ := 89
def max_shots : ℕ := 10
def points_range : Finset ℕ := Finset.range 11 \ {0}

-- Lean statement for the first question
theorem min_score_seventh_shot_to_break_record (x₇ : ℕ) (h₁: x₇ ∈ points_range) :
  initial_points + x₇ + 30 > record_points ↔ x₇ ≥ 8 :=
by sorry

-- Lean statement for the second question
theorem shots_hitting_10_to_break_record_when_7th_shot_is_8 (x₈ x₉ x₁₀ : ℕ)
  (h₂ : 8 ∈ points_range) 
  (h₃ : x₈ ∈ points_range) (h₄ : x₉ ∈ points_range) (h₅ : x₁₀ ∈ points_range) :
  initial_points + 8 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∧ x₉ = 10 ∧ x₁₀ = 10) :=
by sorry

-- Lean statement for the third question
theorem necessary_shot_of_10_when_7th_shot_is_10 (x₈ x₉ x₁₀ : ℕ)
  (h₆ : 10 ∈ points_range)
  (h₇ : x₈ ∈ points_range) (h₈ : x₉ ∈ points_range) (h₉ : x₁₀ ∈ points_range) :
  initial_points + 10 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∨ x₉ = 10 ∨ x₁₀ = 10) :=
by sorry

end min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l1823_182330


namespace find_constant_d_l1823_182373

noncomputable def polynomial_g (d : ℝ) (x : ℝ) := d * x^4 + 17 * x^3 - 5 * d * x^2 + 45

theorem find_constant_d (d : ℝ) : polynomial_g d 5 = 0 → d = -4.34 :=
by
  sorry

end find_constant_d_l1823_182373


namespace solve_for_buttons_l1823_182376

def number_of_buttons_on_second_shirt (x : ℕ) : Prop :=
  200 * 3 + 200 * x = 1600

theorem solve_for_buttons : ∃ x : ℕ, number_of_buttons_on_second_shirt x ∧ x = 5 := by
  sorry

end solve_for_buttons_l1823_182376


namespace arcsin_neg_half_eq_neg_pi_six_l1823_182361

theorem arcsin_neg_half_eq_neg_pi_six : 
  Real.arcsin (-1 / 2) = -Real.pi / 6 := 
sorry

end arcsin_neg_half_eq_neg_pi_six_l1823_182361


namespace total_earnings_proof_l1823_182379

-- Definitions of the given conditions
def monthly_earning : ℕ := 4000
def monthly_saving : ℕ := 500
def total_savings_needed : ℕ := 45000

-- Lean statement for the proof problem
theorem total_earnings_proof : 
  (total_savings_needed / monthly_saving) * monthly_earning = 360000 :=
by
  sorry

end total_earnings_proof_l1823_182379


namespace range_of_a_l1823_182329

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1/2)^x = 3 * a + 2 ∧ x < 0) ↔ (a > -1 / 3) :=
by
  sorry

end range_of_a_l1823_182329


namespace number_of_people_l1823_182399

theorem number_of_people (total_eggs : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : 
  total_eggs = 36 → eggs_per_omelet = 4 → omelets_per_person = 3 → 
  (total_eggs / eggs_per_omelet) / omelets_per_person = 3 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_l1823_182399


namespace wholesale_price_l1823_182378

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l1823_182378


namespace new_person_weight_l1823_182357

theorem new_person_weight (W x : ℝ) (h1 : (W - 55 + x) / 8 = (W / 8) + 2.5) : x = 75 := by
  -- Proof omitted
  sorry

end new_person_weight_l1823_182357


namespace probability_at_least_one_deciphers_l1823_182377

theorem probability_at_least_one_deciphers (P_A P_B : ℚ) (hA : P_A = 1/2) (hB : P_B = 1/3) :
    P_A + P_B - P_A * P_B = 2/3 := by
  sorry

end probability_at_least_one_deciphers_l1823_182377


namespace min_value_f_l1823_182371

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l1823_182371


namespace greatest_fourth_term_l1823_182312

theorem greatest_fourth_term (a d : ℕ) (h1 : a > 0) (h2 : d > 0) 
  (h3 : 5 * a + 10 * d = 50) (h4 : a + 2 * d = 10) : 
  a + 3 * d = 14 :=
by {
  -- We introduced the given constraints and now need a proof
  sorry
}

end greatest_fourth_term_l1823_182312


namespace amount_paid_l1823_182301

theorem amount_paid (cost_price : ℝ) (percent_more : ℝ) (h1 : cost_price = 6525) (h2 : percent_more = 0.24) : 
  cost_price + percent_more * cost_price = 8091 :=
by 
  -- Proof here
  sorry

end amount_paid_l1823_182301


namespace sufficient_not_necessary_ellipse_l1823_182358

theorem sufficient_not_necessary_ellipse (m n : ℝ) (h : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m ≠ n) ∧
  ¬(∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m > n ∧ n > 0) :=
sorry

end sufficient_not_necessary_ellipse_l1823_182358


namespace perpendicular_planes_normal_vector_l1823_182309

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem perpendicular_planes_normal_vector {m : ℝ} 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (h₁ : a = (1, 2, -2)) 
  (h₂ : b = (-2, 1, m)) 
  (h₃ : dot_product a b = 0) : 
  m = 0 := 
sorry

end perpendicular_planes_normal_vector_l1823_182309


namespace cost_of_450_chocolates_l1823_182347

theorem cost_of_450_chocolates :
  ∀ (cost_per_box : ℝ) (candies_per_box total_candies : ℕ),
  cost_per_box = 7.50 →
  candies_per_box = 30 →
  total_candies = 450 →
  (total_candies / candies_per_box : ℝ) * cost_per_box = 112.50 :=
by
  intros cost_per_box candies_per_box total_candies h1 h2 h3
  sorry

end cost_of_450_chocolates_l1823_182347


namespace simplify_polynomial_expression_l1823_182398

noncomputable def polynomial_expression (x : ℝ) := 
  (3 * x^3 + x^2 - 5 * x + 9) * (x + 2) - (x + 2) * (2 * x^3 - 4 * x + 8) + (x^2 - 6 * x + 13) * (x + 2) * (x - 3)

theorem simplify_polynomial_expression (x : ℝ) :
  polynomial_expression x = 2 * x^4 + x^3 + 9 * x^2 + 23 * x + 2 :=
sorry

end simplify_polynomial_expression_l1823_182398


namespace num_triangles_square_even_num_triangles_rect_even_l1823_182334

-- Problem (a): Proving that the number of triangles is even 
theorem num_triangles_square_even (a : ℕ) (n : ℕ) (h : a * a = n * (3 * 4 / 2)) : 
  n % 2 = 0 :=
sorry

-- Problem (b): Proving that the number of triangles is even
theorem num_triangles_rect_even (L W k : ℕ) (hL : L = k * 2) (hW : W = k * 1) (h : L * W = k * 1 * 2 / 2) :
  k % 2 = 0 :=
sorry

end num_triangles_square_even_num_triangles_rect_even_l1823_182334


namespace triangle_max_distance_product_l1823_182314

open Real

noncomputable def max_product_of_distances
  (a b c : ℝ) (P : {p : ℝ × ℝ // True}) : ℝ :=
  let h_a := 1 -- placeholder for actual distance calculation
  let h_b := 1 -- placeholder for actual distance calculation
  let h_c := 1 -- placeholder for actual distance calculation
  h_a * h_b * h_c

theorem triangle_max_distance_product
  (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (P : {p : ℝ × ℝ // True}) :
  max_product_of_distances a b c P = (16/15 : ℝ) :=
sorry

end triangle_max_distance_product_l1823_182314


namespace major_arc_circumference_l1823_182353

noncomputable def circumference_major_arc 
  (A B C : Point) (r : ℝ) (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) : ℝ :=
  let total_circumference := 2 * Real.pi * r
  let major_arc_angle := 360 - angle_ACB
  major_arc_angle / 360 * total_circumference

theorem major_arc_circumference (A B C : Point) (r : ℝ)
  (angle_ACB : ℝ) (h1 : r = 24) (h2 : angle_ACB = 110) :
  circumference_major_arc A B C r angle_ACB h1 h2 = (500 / 3) * Real.pi :=
  sorry

end major_arc_circumference_l1823_182353


namespace num_clerks_l1823_182320

def manager_daily_salary := 5
def clerk_daily_salary := 2
def num_managers := 2
def total_daily_salary := 16

theorem num_clerks (c : ℕ) (h1 : num_managers * manager_daily_salary + c * clerk_daily_salary = total_daily_salary) : c = 3 :=
by 
  sorry

end num_clerks_l1823_182320


namespace prove_min_max_A_l1823_182306

theorem prove_min_max_A : 
  ∃ (A_max A_min : ℕ), 
  (∃ B : ℕ, 
    A_max = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧
    B % 10 = 9) ∧ 
  (∃ B : ℕ, 
    A_min = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧ 
    B % 10 = 1) ∧ 
  A_max = 999999998 ∧ 
  A_min = 166666667 := sorry

end prove_min_max_A_l1823_182306


namespace greater_number_l1823_182356

theorem greater_number (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 2) (h3 : a > b) : a = 21 := by
  sorry

end greater_number_l1823_182356


namespace total_tin_in_new_alloy_l1823_182370

-- Define the weights of alloy A and alloy B
def weightAlloyA : Float := 135
def weightAlloyB : Float := 145

-- Define the ratio of lead to tin in alloy A
def ratioLeadToTinA : Float := 3 / 5

-- Define the ratio of tin to copper in alloy B
def ratioTinToCopperB : Float := 2 / 3

-- Define the total parts for alloy A and alloy B
def totalPartsA : Float := 3 + 5
def totalPartsB : Float := 2 + 3

-- Define the fraction of tin in alloy A and alloy B
def fractionTinA : Float := 5 / totalPartsA
def fractionTinB : Float := 2 / totalPartsB

-- Calculate the amount of tin in alloy A and alloy B
def tinInAlloyA : Float := fractionTinA * weightAlloyA
def tinInAlloyB : Float := fractionTinB * weightAlloyB

-- Calculate the total amount of tin in the new alloy
def totalTinInNewAlloy : Float := tinInAlloyA + tinInAlloyB

-- The theorem to be proven
theorem total_tin_in_new_alloy : totalTinInNewAlloy = 142.375 := by
  sorry

end total_tin_in_new_alloy_l1823_182370


namespace salmon_at_rest_oxygen_units_l1823_182315

noncomputable def salmonSwimSpeed (x : ℝ) : ℝ := (1/2) * Real.log (x / 100 * Real.pi) / Real.log 3

theorem salmon_at_rest_oxygen_units :
  ∃ x : ℝ, salmonSwimSpeed x = 0 ∧ x = 100 / Real.pi :=
by
  sorry

end salmon_at_rest_oxygen_units_l1823_182315


namespace percentage_favoring_all_three_l1823_182362

variable (A B C A_union_B_union_C Y X : ℝ)

-- Conditions
axiom hA : A = 0.50
axiom hB : B = 0.30
axiom hC : C = 0.20
axiom hA_union_B_union_C : A_union_B_union_C = 0.78
axiom hY : Y = 0.17

-- Question: Prove that the percentage of those asked favoring all three proposals is 5%
theorem percentage_favoring_all_three :
  A = 0.50 → B = 0.30 → C = 0.20 →
  A_union_B_union_C = 0.78 →
  Y = 0.17 →
  X = 0.05 :=
by
  intros
  sorry

end percentage_favoring_all_three_l1823_182362


namespace least_subtr_from_12702_to_div_by_99_l1823_182363

theorem least_subtr_from_12702_to_div_by_99 : ∃ k : ℕ, 12702 - k = 99 * (12702 / 99) ∧ 0 ≤ k ∧ k < 99 :=
by
  sorry

end least_subtr_from_12702_to_div_by_99_l1823_182363


namespace theater_revenue_l1823_182345

theorem theater_revenue
  (total_seats : ℕ)
  (adult_price : ℕ)
  (child_price : ℕ)
  (child_tickets_sold : ℕ)
  (total_sold_out : total_seats = 80)
  (child_tickets_sold_cond : child_tickets_sold = 63)
  (adult_ticket_price_cond : adult_price = 12)
  (child_ticket_price_cond : child_price = 5)
  : total_seats * adult_price + child_tickets_sold * child_price = 519 :=
by
  -- proof omitted
  sorry

end theater_revenue_l1823_182345


namespace product_largest_smallest_using_digits_l1823_182367

theorem product_largest_smallest_using_digits (a b : ℕ) (h1 : 100 * 6 + 10 * 2 + 0 = a) (h2 : 100 * 2 + 10 * 0 + 6 = b) : a * b = 127720 := by
  -- The proof will go here
  sorry

end product_largest_smallest_using_digits_l1823_182367


namespace f_inv_f_inv_14_l1823_182392

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem f_inv_f_inv_14 : f_inv (f_inv 14) = -14 / 9 :=
by {
  sorry
}

end f_inv_f_inv_14_l1823_182392


namespace find_a_l1823_182321

theorem find_a (a : ℝ) : (∀ x : ℝ, (x + 1) * (x - 3) = x^2 + a * x - 3) → a = -2 :=
  by
    sorry

end find_a_l1823_182321


namespace positive_difference_two_numbers_l1823_182366

theorem positive_difference_two_numbers (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 2 * y - 3 * x = 5) : abs (y - x) = 8 := 
sorry

end positive_difference_two_numbers_l1823_182366


namespace find_number_l1823_182364

theorem find_number (n : ℕ) (h₁ : ∀ x : ℕ, 21 + 7 * x = n ↔ 3 + x = 47):
  n = 329 :=
by
  -- Proof will go here
  sorry

end find_number_l1823_182364


namespace multiples_6_8_not_both_l1823_182380

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l1823_182380


namespace max_value_of_k_l1823_182349

theorem max_value_of_k:
  ∃ (k : ℕ), 
  (∀ (a b : ℕ → ℕ) (h : ∀ i, a i < b i) (no_share : ∀ i j, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)) (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) (sum_limit : ∀ i, a i + b i ≤ 3011), 
    k ≤ 3011 ∧ k = 1204) := sorry

end max_value_of_k_l1823_182349


namespace remainder_is_neg_x_plus_60_l1823_182303

theorem remainder_is_neg_x_plus_60 (R : Polynomial ℝ) :
  (R.eval 10 = 50) ∧ (R.eval 50 = 10) → 
  ∃ Q : Polynomial ℝ, R = (Polynomial.X - 10) * (Polynomial.X - 50) * Q + (- Polynomial.X + 60) :=
by
  sorry

end remainder_is_neg_x_plus_60_l1823_182303


namespace largest_triangle_angle_l1823_182311

-- Define the angles
def angle_sum := (105 : ℝ) -- Degrees
def delta_angle := (36 : ℝ) -- Degrees
def total_sum := (180 : ℝ) -- Degrees

-- Theorem statement
theorem largest_triangle_angle (a b c : ℝ) (h1 : a + b = angle_sum)
  (h2 : b = a + delta_angle) (h3 : a + b + c = total_sum) : c = 75 :=
sorry

end largest_triangle_angle_l1823_182311


namespace keith_turnips_l1823_182348

theorem keith_turnips (a t k : ℕ) (h1 : a = 9) (h2 : t = 15) : k = t - a := by
  sorry

end keith_turnips_l1823_182348


namespace part_I_part_II_l1823_182310

theorem part_I (a b : ℝ) (h1 : 0 < a) (h2 : b * a = 2)
  (h3 : (1 + b) * a = 3) :
  (a = 1) ∧ (b = 2) :=
by {
  sorry
}

theorem part_II (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1 : ℝ) / x + 2 / y = 1)
  (k : ℝ) : 2 * x + y ≥ k^2 + k + 2 → (-3 ≤ k) ∧ (k ≤ 2) :=
by {
  sorry
}

end part_I_part_II_l1823_182310


namespace find_c_l1823_182346

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end find_c_l1823_182346


namespace tree_height_at_2_years_l1823_182325

-- Define the conditions
def triples_height (height : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, height (n + 1) = 3 * height n

def height_at_5_years (height : ℕ → ℕ) : Prop :=
  height 5 = 243

-- Set up the problem statement
theorem tree_height_at_2_years (height : ℕ → ℕ) 
  (H1 : triples_height height) 
  (H2 : height_at_5_years height) : 
  height 2 = 9 :=
sorry

end tree_height_at_2_years_l1823_182325


namespace regression_estimate_l1823_182359

theorem regression_estimate:
  ∀ (x : ℝ), (1.43 * x + 257 = 400) → x = 100 :=
by
  intro x
  intro h
  sorry

end regression_estimate_l1823_182359


namespace area_of_rectangle_l1823_182381

theorem area_of_rectangle (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 90) : w * l = 379.6875 :=
by
  sorry

end area_of_rectangle_l1823_182381


namespace P_shape_points_length_10_l1823_182385

def P_shape_points (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem P_shape_points_length_10 :
  P_shape_points 10 = 31 := 
by 
  sorry

end P_shape_points_length_10_l1823_182385


namespace triangle_area_inscribed_in_circle_l1823_182383

theorem triangle_area_inscribed_in_circle :
  ∀ (x : ℝ), (2 * x)^2 + (3 * x)^2 = (4 * x)^2 → (5 = (4 * x) / 2) → (1/2 * (2 * x) * (3 * x) = 18.75) :=
by
  -- Assume all necessary conditions
  intros x h_ratio h_radius
  -- Skip the proof part using sorry
  sorry

end triangle_area_inscribed_in_circle_l1823_182383


namespace certain_percentage_of_1600_l1823_182393

theorem certain_percentage_of_1600 (P : ℝ) 
  (h : 0.05 * (P / 100 * 1600) = 20) : 
  P = 25 :=
by 
  sorry

end certain_percentage_of_1600_l1823_182393


namespace ice_cream_flavors_l1823_182354

-- We have four basic flavors and want to combine four scoops from these flavors.
def ice_cream_combinations : ℕ :=
  Nat.choose 7 3

theorem ice_cream_flavors : ice_cream_combinations = 35 :=
by
  sorry

end ice_cream_flavors_l1823_182354


namespace number_of_methods_l1823_182390

def doctors : ℕ := 6
def days : ℕ := 3

theorem number_of_methods : (days^doctors) = 729 := 
by sorry

end number_of_methods_l1823_182390


namespace factory_workers_total_payroll_l1823_182341

theorem factory_workers_total_payroll (total_office_payroll : ℝ) (number_factory_workers : ℝ) 
(number_office_workers : ℝ) (salary_difference : ℝ) 
(average_office_salary : ℝ) (average_factory_salary : ℝ) 
(h1 : total_office_payroll = 75000) (h2 : number_factory_workers = 15)
(h3 : number_office_workers = 30) (h4 : salary_difference = 500)
(h5 : average_office_salary = total_office_payroll / number_office_workers)
(h6 : average_office_salary = average_factory_salary + salary_difference) :
  number_factory_workers * average_factory_salary = 30000 :=
by
  sorry

end factory_workers_total_payroll_l1823_182341


namespace matchstick_game_winner_a_matchstick_game_winner_b_l1823_182337

def is_winning_position (pile1 pile2 : Nat) : Bool :=
  (pile1 % 2 = 1) && (pile2 % 2 = 1)

theorem matchstick_game_winner_a : is_winning_position 101 201 = true := 
by
  -- Theorem statement for (101 matches, 201 matches)
  -- The second player wins
  sorry

theorem matchstick_game_winner_b : is_winning_position 100 201 = false := 
by
  -- Theorem statement for (100 matches, 201 matches)
  -- The first player wins
  sorry

end matchstick_game_winner_a_matchstick_game_winner_b_l1823_182337


namespace sum_three_digit_integers_from_200_to_900_l1823_182322

theorem sum_three_digit_integers_from_200_to_900 : 
  let a := 200
  let l := 900
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 385550 := by
    let a := 200
    let l := 900
    let d := 1
    let n := (l - a) / d + 1
    let S := n / 2 * (a + l)
    sorry

end sum_three_digit_integers_from_200_to_900_l1823_182322


namespace initial_roses_count_l1823_182342

theorem initial_roses_count 
  (roses_to_mother : ℕ)
  (roses_to_grandmother : ℕ)
  (roses_to_sister : ℕ)
  (roses_kept : ℕ)
  (initial_roses : ℕ)
  (h_mother : roses_to_mother = 6)
  (h_grandmother : roses_to_grandmother = 9)
  (h_sister : roses_to_sister = 4)
  (h_kept : roses_kept = 1)
  (h_initial : initial_roses = roses_to_mother + roses_to_grandmother + roses_to_sister + roses_kept) :
  initial_roses = 20 :=
by
  rw [h_mother, h_grandmother, h_sister, h_kept] at h_initial
  exact h_initial

end initial_roses_count_l1823_182342


namespace minimum_number_of_colors_l1823_182307

theorem minimum_number_of_colors (n : ℕ) (h_n : 2 ≤ n) :
  ∀ (f : (Fin n) → ℕ),
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) →
  (∃ c : ℕ, c = n) :=
by sorry

end minimum_number_of_colors_l1823_182307


namespace linear_function_not_passing_through_third_quadrant_l1823_182350

theorem linear_function_not_passing_through_third_quadrant
  (m : ℝ)
  (h : 4 + 4 * m < 0) : 
  ∀ x y : ℝ, (y = m * x - m) → ¬ (x < 0 ∧ y < 0) :=
by
  sorry

end linear_function_not_passing_through_third_quadrant_l1823_182350


namespace centipede_shoes_and_socks_l1823_182382

-- Define number of legs
def num_legs : ℕ := 10

-- Define the total number of items
def total_items : ℕ := 2 * num_legs

-- Define the total permutations without constraints
def total_permutations : ℕ := Nat.factorial total_items

-- Define the probability constraint for each leg
def single_leg_probability : ℚ := 1 / 2

-- Define the combined probability constraint for all legs
def all_legs_probability : ℚ := single_leg_probability ^ num_legs

-- Define the number of valid permutations (the answer to prove)
def valid_permutations : ℚ := total_permutations / all_legs_probability

theorem centipede_shoes_and_socks : valid_permutations = (Nat.factorial 20 : ℚ) / 2^10 :=
by
  -- The proof is omitted
  sorry

end centipede_shoes_and_socks_l1823_182382


namespace binomial_coefficient_fourth_term_l1823_182391

theorem binomial_coefficient_fourth_term (n k : ℕ) (hn : n = 5) (hk : k = 3) : Nat.choose n k = 10 := by
  sorry

end binomial_coefficient_fourth_term_l1823_182391


namespace wendy_washing_loads_l1823_182394

theorem wendy_washing_loads (shirts sweaters machine_capacity : ℕ) (total_clothes := shirts + sweaters) 
  (loads := total_clothes / machine_capacity) 
  (remainder := total_clothes % machine_capacity) 
  (h_shirts : shirts = 39) 
  (h_sweaters : sweaters = 33) 
  (h_machine_capacity : machine_capacity = 8) : loads = 9 ∧ remainder = 0 := 
by 
  sorry

end wendy_washing_loads_l1823_182394


namespace find_x_square_l1823_182338

theorem find_x_square (x : ℝ) (h_pos : x > 0) (h_condition : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end find_x_square_l1823_182338


namespace triangle_sides_and_angles_l1823_182332

theorem triangle_sides_and_angles (a : Real) (α β : Real) :
  (a ≥ 0) →
  let sides := [a, a + 1, a + 2]
  let angles := [α, β, 2 * α]
  (∀ s, s ∈ sides) → (∀ θ, θ ∈ angles) →
  a = 4 ∧ a + 1 = 5 ∧ a + 2 = 6 := 
by {
  sorry
}

end triangle_sides_and_angles_l1823_182332


namespace city_mpg_l1823_182374

-- Define the conditions
variables {T H C : ℝ}
axiom cond1 : H * T = 560
axiom cond2 : (H - 6) * T = 336

-- The formal proof goal
theorem city_mpg : C = 9 :=
by
  have h1 : H = 560 / T := by sorry
  have h2 : (560 / T - 6) * T = 336 := by sorry
  have h3 : C = H - 6 := by sorry
  have h4 :  C = 9 := by sorry
  exact h4

end city_mpg_l1823_182374


namespace periodic_sequence_condition_l1823_182317

theorem periodic_sequence_condition (m : ℕ) (a : ℕ) 
  (h_pos : 0 < m)
  (a_seq : ℕ → ℕ) (h_initial : a_seq 0 = a)
  (h_relation : ∀ n, a_seq (n + 1) = if a_seq n % 2 = 0 then a_seq n / 2 else a_seq n + m) :
  (∃ p, ∀ k, a_seq (k + p) = a_seq k) ↔ 
  (a ∈ ({n | 1 ≤ n ∧ n ≤ m} ∪ {n | ∃ k, n = m + 2 * k + 1 ∧ n < 2 * m + 1})) :=
sorry

end periodic_sequence_condition_l1823_182317


namespace p_sufficient_not_necessary_q_l1823_182327

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l1823_182327


namespace minimize_wage_l1823_182318

def totalWorkers : ℕ := 150
def wageA : ℕ := 2000
def wageB : ℕ := 3000

theorem minimize_wage : ∃ (a : ℕ), a = 50 ∧ (totalWorkers - a) ≥ 2 * a ∧ 
  (wageA * a + wageB * (totalWorkers - a) = 400000) := sorry

end minimize_wage_l1823_182318


namespace solve_fractional_equation_l1823_182324

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l1823_182324
