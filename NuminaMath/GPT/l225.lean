import Mathlib

namespace complex_expression_value_l225_225062

theorem complex_expression_value {i : ℂ} (h : i^2 = -1) : i^3 * (1 + i)^2 = 2 := 
by
  sorry

end complex_expression_value_l225_225062


namespace average_of_side_lengths_of_squares_l225_225208

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l225_225208


namespace total_blue_marbles_l225_225079

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end total_blue_marbles_l225_225079


namespace inequality_positives_l225_225467

theorem inequality_positives (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
sorry

end inequality_positives_l225_225467


namespace increasing_interval_l225_225731

-- Given function definition
def quad_func (x : ℝ) : ℝ := -x^2 + 1

-- Property to be proven: The function is increasing on the interval (-∞, 0]
theorem increasing_interval : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → quad_func x < quad_func y := by
  sorry

end increasing_interval_l225_225731


namespace not_divisible_by_5_count_l225_225309

-- Define the total number of four-digit numbers using the digits 0, 1, 2, 3, 4, 5 without repetition
def total_four_digit_numbers : ℕ := 300

-- Define the number of four-digit numbers ending with 0
def numbers_ending_with_0 : ℕ := 60

-- Define the number of four-digit numbers ending with 5
def numbers_ending_with_5 : ℕ := 48

-- Theorem stating the number of four-digit numbers that cannot be divided by 5
theorem not_divisible_by_5_count : total_four_digit_numbers - numbers_ending_with_0 - numbers_ending_with_5 = 192 :=
by
  -- Proof skipped
  sorry

end not_divisible_by_5_count_l225_225309


namespace number_of_sets_B_l225_225648

def A : Set ℕ := {1, 2, 3}

theorem number_of_sets_B :
  ∃ B : Set ℕ, (A ∪ B = A ∧ 1 ∈ B ∧ (∃ n : ℕ, n = 4)) :=
by
  sorry

end number_of_sets_B_l225_225648


namespace area_square_hypotenuse_l225_225727

theorem area_square_hypotenuse 
(a : ℝ) 
(h1 : ∀ a: ℝ,  ∃ YZ: ℝ, YZ = a + 3) 
(h2: ∀ XY: ℝ, ∃ total_area: ℝ, XY^2 + XY * (XY + 3) + (2 * XY^2 + 6 * XY + 9) = 450) :
  ∃ XZ: ℝ, (2 * a^2 + 6 * a + 9 = XZ) → XZ = 201 := by
  sorry

end area_square_hypotenuse_l225_225727


namespace probability_A_not_winning_l225_225770

theorem probability_A_not_winning 
  (prob_draw : ℚ := 1/2)
  (prob_B_wins : ℚ := 1/3) : 
  (prob_draw + prob_B_wins) = 5 / 6 := 
by
  sorry

end probability_A_not_winning_l225_225770


namespace sum_of_1_to_17_is_odd_l225_225735

-- Define the set of natural numbers from 1 to 17
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

-- Proof that the sum of these numbers is odd
theorem sum_of_1_to_17_is_odd : (List.sum nums) % 2 = 1 := 
by
  sorry  -- Proof goes here

end sum_of_1_to_17_is_odd_l225_225735


namespace reciprocal_square_inequality_l225_225657

variable (x y : ℝ)
variable (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≤ y)

theorem reciprocal_square_inequality :
  (1 / y^2) ≤ (1 / x^2) :=
sorry

end reciprocal_square_inequality_l225_225657


namespace sin_cos_value_sin_minus_cos_value_tan_value_l225_225060

variable (x : ℝ)

theorem sin_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x * Real.cos x = - 12 / 25 := 
sorry

theorem sin_minus_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x - Real.cos x = - 7 / 5 := 
sorry

theorem tan_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.tan x = - 3 / 4 := 
sorry

end sin_cos_value_sin_minus_cos_value_tan_value_l225_225060


namespace filtration_concentration_l225_225597

-- Variables and conditions used in the problem
variable (P P0 : ℝ) (k t : ℝ)
variable (h1 : P = P0 * Real.exp (-k * t))
variable (h2 : Real.exp (-2 * k) = 0.8)

-- Main statement: Prove the concentration after 5 hours is approximately 57% of the original
theorem filtration_concentration :
  (P0 * Real.exp (-5 * k)) / P0 = 0.57 :=
by sorry

end filtration_concentration_l225_225597


namespace cos_C_sin_B_area_l225_225505

noncomputable def triangle_conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧
  (b / c = 2 * Real.sqrt 3 / 3) ∧
  (A + 3 * C = Real.pi)

theorem cos_C (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.cos C = Real.sqrt 3 / 3 :=
sorry

theorem sin_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) :
  Real.sin B = 2 * Real.sqrt 2 / 3 :=
sorry

theorem area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) (hb : b = 3 * Real.sqrt 3) :
  (1 / 2) * b * c * Real.sin A = 9 * Real.sqrt 2 / 4 :=
sorry

end cos_C_sin_B_area_l225_225505


namespace exactly_one_root_in_interval_l225_225213

theorem exactly_one_root_in_interval (p q : ℝ) (h : q * (q + p + 1) < 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ (x^2 + p * x + q = 0) := sorry

end exactly_one_root_in_interval_l225_225213


namespace gcd_two_powers_l225_225548

def m : ℕ := 2 ^ 1998 - 1
def n : ℕ := 2 ^ 1989 - 1

theorem gcd_two_powers :
  Nat.gcd (2 ^ 1998 - 1) (2 ^ 1989 - 1) = 511 := 
sorry

end gcd_two_powers_l225_225548


namespace problem1_problem2_l225_225141

-- Problem 1: Prove the simplification of an expression
theorem problem1 (x : ℝ) : (2*x + 1)^2 + x*(x-4) = 5*x^2 + 1 := 
by sorry

-- Problem 2: Prove the solution set for the system of inequalities
theorem problem2 (x : ℝ) (h1 : 3*x - 6 > 0) (h2 : (5 - x) / 2 < 1) : x > 3 := 
by sorry

end problem1_problem2_l225_225141


namespace ethanol_total_amount_l225_225246

-- Definitions based on Conditions
def total_tank_capacity : ℕ := 214
def fuel_A_volume : ℕ := 106
def fuel_B_volume : ℕ := total_tank_capacity - fuel_A_volume
def ethanol_in_fuel_A : ℚ := 0.12
def ethanol_in_fuel_B : ℚ := 0.16

-- Theorem Statement
theorem ethanol_total_amount :
  (fuel_A_volume * ethanol_in_fuel_A + fuel_B_volume * ethanol_in_fuel_B) = 30 := 
sorry

end ethanol_total_amount_l225_225246


namespace largest_x_63_over_8_l225_225494

theorem largest_x_63_over_8 (x : ℝ) (h1 : ⌊x⌋ / x = 8 / 9) : x = 63 / 8 :=
by
  sorry

end largest_x_63_over_8_l225_225494


namespace find_the_number_l225_225388

theorem find_the_number (n : ℤ) 
    (h : 45 - (28 - (n - (15 - 18))) = 57) :
    n = 37 := 
sorry

end find_the_number_l225_225388


namespace minimum_value_1_minimum_value_2_l225_225833

noncomputable section

open Real -- Use the real numbers

theorem minimum_value_1 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + y^2 + z^2 >= 8 / 3 :=
by
  sorry  -- Proof omitted
 
theorem minimum_value_2 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + (y - 1)^2 + z^2 >= 6 :=
by
  sorry  -- Proof omitted

end minimum_value_1_minimum_value_2_l225_225833


namespace closest_number_to_fraction_l225_225291

theorem closest_number_to_fraction (x : ℝ) : 
  (abs (x - 2000) < abs (x - 1500)) ∧ 
  (abs (x - 2000) < abs (x - 2500)) ∧ 
  (abs (x - 2000) < abs (x - 3000)) ∧ 
  (abs (x - 2000) < abs (x - 3500)) :=
by
  let x := 504 / 0.252
  sorry

end closest_number_to_fraction_l225_225291


namespace arithmetic_proof_l225_225343

theorem arithmetic_proof : (28 + 48 / 69) * 69 = 1980 :=
by
  sorry

end arithmetic_proof_l225_225343


namespace increase_to_restore_l225_225581

noncomputable def percentage_increase_to_restore (P : ℝ) : ℝ :=
  let reduced_price := 0.9 * P
  let restore_factor := P / reduced_price
  (restore_factor - 1) * 100

theorem increase_to_restore :
  percentage_increase_to_restore 100 = 100 / 9 :=
by
  sorry

end increase_to_restore_l225_225581


namespace translation_min_point_correct_l225_225560

-- Define the original equation
def original_eq (x : ℝ) := |x| - 5

-- Define the translation function
def translate_point (p : ℝ × ℝ) (tx ty : ℝ) : ℝ × ℝ := (p.1 + tx, p.2 + ty)

-- Define the minimum point of the original equation
def original_min_point : ℝ × ℝ := (0, original_eq 0)

-- Translate the original minimum point three units right and four units up
def new_min_point := translate_point original_min_point 3 4

-- Prove that the new minimum point is (3, -1)
theorem translation_min_point_correct : new_min_point = (3, -1) :=
by
  sorry

end translation_min_point_correct_l225_225560


namespace find_k_l225_225743

theorem find_k (x y z k : ℝ) 
  (h1 : 9 / (x + y) = k / (x + 2 * z)) 
  (h2 : 9 / (x + y) = 14 / (z - y)) 
  (h3 : y = 2 * x) 
  (h4 : x + z = 10) :
  k = 46 :=
by
  sorry

end find_k_l225_225743


namespace magician_guarantee_success_l225_225663

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end magician_guarantee_success_l225_225663


namespace vasya_did_not_buy_anything_days_l225_225677

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l225_225677


namespace intersection_of_lines_l225_225591

-- Define the first and second lines
def line1 (x : ℚ) : ℚ := 3 * x + 1
def line2 (x : ℚ) : ℚ := -7 * x - 5

-- Statement: Prove that the intersection of the lines given by
-- y = 3x + 1 and y + 5 = -7x is (-3/5, -4/5).

theorem intersection_of_lines :
  ∃ x y : ℚ, y = line1 x ∧ y = line2 x ∧ x = -3 / 5 ∧ y = -4 / 5 :=
by
  sorry

end intersection_of_lines_l225_225591


namespace sum_of_b_values_l225_225834

theorem sum_of_b_values :
  let discriminant (b : ℝ) := (b + 6) ^ 2 - 4 * 3 * 12
  ∃ b1 b2 : ℝ, discriminant b1 = 0 ∧ discriminant b2 = 0 ∧ b1 + b2 = -12 :=
by sorry

end sum_of_b_values_l225_225834


namespace ben_time_to_school_l225_225465

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

end ben_time_to_school_l225_225465


namespace tory_needs_to_sell_more_packs_l225_225549

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end tory_needs_to_sell_more_packs_l225_225549


namespace cost_per_person_is_125_l225_225620

-- Defining the conditions
def totalCost : ℤ := 25000000000
def peopleSharing : ℤ := 200000000

-- Define the expected cost per person based on the conditions
def costPerPerson : ℤ := totalCost / peopleSharing

-- Proving that the cost per person is 125 dollars.
theorem cost_per_person_is_125 : costPerPerson = 125 := by
  sorry

end cost_per_person_is_125_l225_225620


namespace min_value_z_l225_225712

variable {x y : ℝ}

def constraint1 (x y : ℝ) : Prop := x + y ≤ 3
def constraint2 (x y : ℝ) : Prop := x - y ≥ -1
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) 
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : 
  ∃ x y, x > 0 ∧ y ≥ 1 ∧ x + y ≤ 3 ∧ x - y ≥ -1 ∧ (∀ x' y', x' > 0 ∧ y' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - y' ≥ -1 → (y' / x' ≥ y / x)) ∧ y / x = 1 / 2 := 
sorry

end min_value_z_l225_225712


namespace sin_300_eq_neg_sqrt3_div_2_l225_225168

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l225_225168


namespace more_student_tickets_l225_225282

-- Definitions of given conditions
def student_ticket_price : ℕ := 6
def nonstudent_ticket_price : ℕ := 9
def total_sales : ℕ := 10500
def total_tickets : ℕ := 1700

-- Definitions of the variables for student and nonstudent tickets
variables (S N : ℕ)

-- Lean statement of the problem
theorem more_student_tickets (h1 : student_ticket_price * S + nonstudent_ticket_price * N = total_sales)
                            (h2 : S + N = total_tickets) : S - N = 1500 :=
by
  sorry

end more_student_tickets_l225_225282


namespace total_chickens_on_farm_l225_225238

noncomputable def total_chickens (H R : ℕ) : ℕ := H + R

theorem total_chickens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H = 67) : total_chickens H R = 75 := 
by
  sorry

end total_chickens_on_farm_l225_225238


namespace ratio_of_work_speeds_l225_225524

theorem ratio_of_work_speeds (B_speed : ℚ) (combined_speed : ℚ) (A_speed : ℚ) 
  (h1 : B_speed = 1/12) 
  (h2 : combined_speed = 1/4) 
  (h3 : A_speed + B_speed = combined_speed) : 
  A_speed / B_speed = 2 := 
sorry

end ratio_of_work_speeds_l225_225524


namespace notepad_last_duration_l225_225600

def note_duration (folds_per_paper : ℕ) (pieces_of_paper : ℕ) (notes_per_day : ℕ) : ℕ :=
  let note_size_papers_per_letter_paper := 2 ^ folds_per_paper
  let total_note_size_papers := pieces_of_paper * note_size_papers_per_letter_paper
  total_note_size_papers / notes_per_day

theorem notepad_last_duration :
  note_duration 3 5 10 = 4 := by
  sorry

end notepad_last_duration_l225_225600


namespace quadratic_equation_roots_l225_225762

theorem quadratic_equation_roots (a b k k1 k2 : ℚ)
  (h_roots : ∀ x : ℚ, k * (x^2 - x) + x + 2 = 0)
  (h_ab_condition : (a / b) + (b / a) = 3 / 7)
  (h_k_values : ∀ x : ℚ, 7 * x^2 - 20 * x - 21 = 0)
  (h_k1k2 : k1 + k2 = 20 / 7)
  (h_k1k2_prod : k1 * k2 = -21 / 7) :
  (k1 / k2) + (k2 / k1) = -104 / 21 :=
sorry

end quadratic_equation_roots_l225_225762


namespace new_rate_of_commission_l225_225171

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end new_rate_of_commission_l225_225171


namespace p_necessary_for_q_l225_225154

def p (x : ℝ) := x ≠ 1
def q (x : ℝ) := x ≥ 2

theorem p_necessary_for_q : ∀ x, q x → p x :=
by
  intro x
  intro hqx
  rw [q] at hqx
  rw [p]
  sorry

end p_necessary_for_q_l225_225154


namespace smallest_integer_mod_conditions_l225_225256

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end smallest_integer_mod_conditions_l225_225256


namespace octal_to_decimal_l225_225143

theorem octal_to_decimal : (1 * 8^3 + 7 * 8^2 + 4 * 8^1 + 3 * 8^0) = 995 :=
by
  sorry

end octal_to_decimal_l225_225143


namespace train_speed_proof_l225_225167

variables (distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse : ℝ)

def question_statement : Prop :=
  distance_to_syracuse = 120 ∧
  total_time_hours = 5.5 ∧
  return_trip_speed = 38.71 →
  average_speed_to_syracuse = 50

theorem train_speed_proof :
  question_statement distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end train_speed_proof_l225_225167


namespace loaves_on_friday_l225_225305

theorem loaves_on_friday
  (bread_wed : ℕ)
  (bread_thu : ℕ)
  (bread_sat : ℕ)
  (bread_sun : ℕ)
  (bread_mon : ℕ)
  (inc_wed_thu : bread_thu - bread_wed = 2)
  (inc_sat_sun : bread_sun - bread_sat = 5)
  (inc_sun_mon : bread_mon - bread_sun = 6)
  (pattern : ∀ n : ℕ, bread_wed + (2 + n) + n = bread_thu + n)
  : bread_thu + 3 = 10 := 
sorry

end loaves_on_friday_l225_225305


namespace pedro_plums_l225_225393

theorem pedro_plums :
  ∃ P Q : ℕ, P + Q = 32 ∧ 2 * P + Q = 52 ∧ P = 20 :=
by
  sorry

end pedro_plums_l225_225393


namespace probability_win_l225_225127

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l225_225127


namespace compute_expression_l225_225566

theorem compute_expression : 12 * (1 / 7) * 14 * 2 = 48 := 
sorry

end compute_expression_l225_225566


namespace vampire_daily_blood_suction_l225_225804

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end vampire_daily_blood_suction_l225_225804


namespace quadratic_equation_m_condition_l225_225601

theorem quadratic_equation_m_condition (m : ℝ) :
  (m + 1 ≠ 0) ↔ (m ≠ -1) :=
by sorry

end quadratic_equation_m_condition_l225_225601


namespace second_group_students_l225_225753

-- Define the number of groups and their respective sizes
def num_groups : ℕ := 4
def first_group_students : ℕ := 5
def third_group_students : ℕ := 7
def fourth_group_students : ℕ := 4
def total_students : ℕ := 24

-- Define the main theorem to prove
theorem second_group_students :
  (∃ second_group_students : ℕ,
    total_students = first_group_students + second_group_students + third_group_students + fourth_group_students ∧
    second_group_students = 8) :=
sorry

end second_group_students_l225_225753


namespace letters_received_per_day_l225_225766

-- Define the conditions
def packages_per_day := 20
def total_pieces_in_six_months := 14400
def days_in_month := 30
def months := 6

-- Calculate total days in six months
def total_days := months * days_in_month

-- Calculate pieces of mail per day
def pieces_per_day := total_pieces_in_six_months / total_days

-- Define the number of letters per day
def letters_per_day := pieces_per_day - packages_per_day

-- Prove that the number of letters per day is 60
theorem letters_received_per_day : letters_per_day = 60 := sorry

end letters_received_per_day_l225_225766


namespace intervals_of_monotonicity_of_f_l225_225422

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem intervals_of_monotonicity_of_f (a b c d : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ d = P.2 ∧ (12 * P.1 - P.2 - 4 = 0))
  (h2 : ∃ x : ℝ, x = 2 ∧ (f a b c d x = 0) ∧ (∃ x : ℝ, x = 0 ∧ (3 * a * x^2 + 2 * b * x + c = 12))) 
  : ( ∃ a b c d : ℝ , (f a b c d) = (2 * x^3 - 9 * x^2 + 12 * x -4)) := 
  sorry

end intervals_of_monotonicity_of_f_l225_225422


namespace john_total_time_l225_225895

noncomputable def total_time_spent : ℝ :=
  let landscape_pictures := 10
  let landscape_drawing_time := 2
  let landscape_coloring_time := landscape_drawing_time * 0.7
  let landscape_enhancing_time := 0.75
  let total_landscape_time := (landscape_drawing_time + landscape_coloring_time + landscape_enhancing_time) * landscape_pictures
  
  let portrait_pictures := 15
  let portrait_drawing_time := 3
  let portrait_coloring_time := portrait_drawing_time * 0.75
  let portrait_enhancing_time := 1.0
  let total_portrait_time := (portrait_drawing_time + portrait_coloring_time + portrait_enhancing_time) * portrait_pictures
  
  let abstract_pictures := 20
  let abstract_drawing_time := 1.5
  let abstract_coloring_time := abstract_drawing_time * 0.6
  let abstract_enhancing_time := 0.5
  let total_abstract_time := (abstract_drawing_time + abstract_coloring_time + abstract_enhancing_time) * abstract_pictures
  
  total_landscape_time + total_portrait_time + total_abstract_time

theorem john_total_time : total_time_spent = 193.25 :=
by sorry

end john_total_time_l225_225895


namespace sum_of_all_digits_divisible_by_nine_l225_225436

theorem sum_of_all_digits_divisible_by_nine :
  ∀ (A B C D : ℕ),
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
  sorry

end sum_of_all_digits_divisible_by_nine_l225_225436


namespace dice_probability_l225_225961

theorem dice_probability :
  let outcomes : List ℕ := [2, 3, 4, 5]
  let total_possible_outcomes := 6 * 6 * 6
  let successful_outcomes := 4 * 4 * 4
  (successful_outcomes / total_possible_outcomes : ℚ) = 8 / 27 :=
by
  sorry

end dice_probability_l225_225961


namespace find_a_and_b_find_monotonic_intervals_and_extreme_values_l225_225685

-- Definitions and conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def takes_extreme_values (f : ℝ → ℝ) (a b c : ℝ) : Prop := 
  ∃ x₁ x₂, x₁ = 1 ∧ x₂ = -2/3 ∧ 3*x₁^2 + 2*a*x₁ + b = 0 ∧ 3*x₂^2 + 2*a*x₂ + b = 0

def f_at_specific_point (f : ℝ → ℝ) (x v : ℝ) : Prop :=
  f x = v

theorem find_a_and_b (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  a = -1/2 ∧ b = -2 :=
sorry

theorem find_monotonic_intervals_and_extreme_values (a b c : ℝ) :
  takes_extreme_values (f a b c) a b c →
  f_at_specific_point (f a b c) (-1) (3/2) →
  c = 1 ∧ 
  (∀ x, x < -2/3 ∨ x > 1 → deriv (f a b c) x > 0) ∧
  (∀ x, -2/3 < x ∧ x < 1 → deriv (f a b c) x < 0) ∧
  f a b c (-2/3) = 49/27 ∧ 
  f a b c 1 = -1/2 :=
sorry

end find_a_and_b_find_monotonic_intervals_and_extreme_values_l225_225685


namespace ratio_of_x_and_y_l225_225857

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 1 / 4 :=
by
  sorry

end ratio_of_x_and_y_l225_225857


namespace bottle_caps_cost_l225_225721

-- Conditions
def cost_per_bottle_cap : ℕ := 2
def number_of_bottle_caps : ℕ := 6

-- Statement of the problem
theorem bottle_caps_cost : (cost_per_bottle_cap * number_of_bottle_caps) = 12 :=
by
  sorry

end bottle_caps_cost_l225_225721


namespace jason_initial_pears_l225_225091

-- Define the initial number of pears Jason picked.
variable (P : ℕ)

-- Conditions translated to Lean:
-- Jason gave Keith 47 pears and received 12 from Mike, leaving him with 11 pears.
variable (h1 : P - 47 + 12 = 11)

-- The theorem stating the problem:
theorem jason_initial_pears : P = 46 :=
by
  sorry

end jason_initial_pears_l225_225091


namespace smallest_integer_remainder_l225_225809

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l225_225809


namespace opposite_of_one_l225_225275

theorem opposite_of_one (a : ℤ) (h : a = -1) : a = -1 := 
by 
  exact h

end opposite_of_one_l225_225275


namespace value_of_m_l225_225065

theorem value_of_m (m : ℤ) : 
  (∃ f : ℤ → ℤ, ∀ x : ℤ, x^2 - (m+1)*x + 1 = (f x)^2) → (m = 1 ∨ m = -3) := 
by
  sorry

end value_of_m_l225_225065


namespace measure_of_obtuse_angle_APB_l225_225849

-- Define the triangle type and conditions
structure Triangle :=
  (A B C : Point)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

-- Define the point type
structure Point :=
  (x y : ℝ)

-- Property of the triangle is isotropic and it contains right angles 90 degrees 
def IsoscelesRightTriangle (T : Triangle) : Prop :=
  T.angle_A = 45 ∧ T.angle_B = 45 ∧ T.angle_C = 90

-- Define the angle bisector intersection point P
def AngleBisectorIntersection (T : Triangle) (P : Point) : Prop :=
  -- (dummy properties assuming necessary geometric constructions can be proven)
  true

-- Statement we want to prove
theorem measure_of_obtuse_angle_APB (T : Triangle) (P : Point) 
    (h1 : IsoscelesRightTriangle T) (h2 : AngleBisectorIntersection T P) :
  ∃ APB : ℝ, APB = 135 :=
  sorry

end measure_of_obtuse_angle_APB_l225_225849


namespace fraction_always_irreducible_l225_225321

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l225_225321


namespace second_player_wins_l225_225242

noncomputable def is_winning_position (n : ℕ) : Prop :=
  n % 4 = 0

theorem second_player_wins (n : ℕ) (h : n = 100) :
  ∃ f : ℕ → ℕ, (∀ k, 0 < k → k ≤ n → (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 5) → is_winning_position (n - k)) ∧ is_winning_position n := 
sorry

end second_player_wins_l225_225242


namespace number_of_elements_in_set_l225_225562

theorem number_of_elements_in_set 
  (S : ℝ) (n : ℝ) 
  (h_avg : S / n = 6.8) 
  (a : ℝ) (h_a : a = 6) 
  (h_new_avg : (S + 2 * a) / n = 9.2) : 
  n = 5 := 
  sorry

end number_of_elements_in_set_l225_225562


namespace final_prices_l225_225233

noncomputable def hat_initial_price : ℝ := 15
noncomputable def hat_first_discount : ℝ := 0.20
noncomputable def hat_second_discount : ℝ := 0.40

noncomputable def gloves_initial_price : ℝ := 8
noncomputable def gloves_first_discount : ℝ := 0.25
noncomputable def gloves_second_discount : ℝ := 0.30

theorem final_prices :
  let hat_price_after_first_discount := hat_initial_price * (1 - hat_first_discount)
  let hat_final_price := hat_price_after_first_discount * (1 - hat_second_discount)
  let gloves_price_after_first_discount := gloves_initial_price * (1 - gloves_first_discount)
  let gloves_final_price := gloves_price_after_first_discount * (1 - gloves_second_discount)
  hat_final_price = 7.20 ∧ gloves_final_price = 4.20 :=
by
  sorry

end final_prices_l225_225233


namespace boat_distance_along_stream_l225_225478

-- Define the conditions
def speed_of_boat_still_water : ℝ := 9
def distance_against_stream_per_hour : ℝ := 7

-- Define the speed of the stream
def speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour

-- Define the speed of the boat along the stream
def speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream

-- Theorem statement
theorem boat_distance_along_stream (speed_of_boat_still_water : ℝ)
                                    (distance_against_stream_per_hour : ℝ)
                                    (effective_speed_against_stream : ℝ := speed_of_boat_still_water - speed_of_stream)
                                    (speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour)
                                    (speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream)
                                    (one_hour : ℝ := 1) :
  speed_of_boat_along_stream = 11 := 
  by
    sorry

end boat_distance_along_stream_l225_225478


namespace ribbon_total_length_l225_225903

theorem ribbon_total_length (R : ℝ)
  (h_first : R - (1/2)*R = (1/2)*R)
  (h_second : (1/2)*R - (1/3)*((1/2)*R) = (1/3)*R)
  (h_third : (1/3)*R - (1/2)*((1/3)*R) = (1/6)*R)
  (h_remaining : (1/6)*R = 250) :
  R = 1500 :=
sorry

end ribbon_total_length_l225_225903


namespace rachel_picked_2_apples_l225_225621

def apples_picked (initial_apples picked_apples final_apples : ℕ) : Prop :=
  initial_apples - picked_apples = final_apples

theorem rachel_picked_2_apples (initial_apples final_apples : ℕ)
  (h_initial : initial_apples = 9)
  (h_final : final_apples = 7) :
  apples_picked initial_apples 2 final_apples :=
by
  rw [h_initial, h_final]
  sorry

end rachel_picked_2_apples_l225_225621


namespace inequality_for_abcd_one_l225_225986

theorem inequality_for_abcd_one (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) + (1 / (1 + d)) > 1 := 
by
  sorry

end inequality_for_abcd_one_l225_225986


namespace cube_splitting_height_l225_225813

/-- If we split a cube with an edge of 1 meter into small cubes with an edge of 1 millimeter,
what will be the height of a column formed by stacking all the small cubes one on top of another? -/
theorem cube_splitting_height :
  let edge_meter := 1
  let edge_mm := 1000
  let num_cubes := (edge_meter * edge_mm) ^ 3
  let height_mm := num_cubes * edge_mm
  let height_km := height_mm / (1000 * 1000 * 1000)
  height_km = 1000 :=
by
  sorry

end cube_splitting_height_l225_225813


namespace retailer_received_extra_boxes_l225_225714
-- Necessary import for mathematical proofs

-- Define the conditions
def dozen_boxes := 12
def dozens_ordered := 3
def discount_percent := 25

-- Calculate the total boxes ordered and the discount factor
def total_boxes := dozen_boxes * dozens_ordered
def discount_factor := (100 - discount_percent) / 100

-- Define the number of boxes paid for and the extra boxes received
def paid_boxes := total_boxes * discount_factor
def extra_boxes := total_boxes - paid_boxes

-- Statement of the proof problem
theorem retailer_received_extra_boxes : extra_boxes = 9 :=
by
    -- This is the place where the proof would be written
    sorry

end retailer_received_extra_boxes_l225_225714


namespace measure_A_l225_225093

noncomputable def angle_A (C B A : ℝ) : Prop :=
  C = 3 / 2 * B ∧ B = 30 ∧ A = 180 - B - C

theorem measure_A (A B C : ℝ) (h : angle_A C B A) : A = 105 :=
by
  -- Extract conditions from h
  obtain ⟨h1, h2, h3⟩ := h
  
  -- Use the conditions to prove the thesis
  simp [h1, h2, h3]
  sorry

end measure_A_l225_225093


namespace find_first_discount_l225_225131

theorem find_first_discount (price_initial : ℝ) (price_final : ℝ) (discount_additional : ℝ) (x : ℝ) :
  price_initial = 350 → price_final = 266 → discount_additional = 5 →
  price_initial * (1 - x / 100) * (1 - discount_additional / 100) = price_final →
  x = 20 :=
by
  intros h1 h2 h3 h4
  -- skippable in proofs, just holds the place
  sorry

end find_first_discount_l225_225131


namespace max_dist_AC_l225_225181

open Real EuclideanGeometry

variables (P A B C : ℝ × ℝ)
  (hPA : dist P A = 1)
  (hPB : dist P B = 1)
  (hPA_PB : dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = - 1 / 2)
  (hBC : dist B C = 1)

theorem max_dist_AC : ∃ C : ℝ × ℝ, dist A C ≤ dist A B + dist B C ∧ dist A C = sqrt 3 + 1 :=
by
  sorry

end max_dist_AC_l225_225181


namespace books_purchased_with_grant_l225_225307

-- Define the conditions
def total_books_now : ℕ := 8582
def books_before_grant : ℕ := 5935

-- State the theorem that we need to prove
theorem books_purchased_with_grant : (total_books_now - books_before_grant) = 2647 := by
  sorry

end books_purchased_with_grant_l225_225307


namespace different_values_of_t_l225_225924

-- Define the conditions on the numbers
variables (p q r s t : ℕ)

-- Define the constraints: p, q, r, s, and t are distinct single-digit numbers
def valid_single_digit (x : ℕ) := x > 0 ∧ x < 10
def distinct_single_digits (p q r s t : ℕ) := 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

-- Define the relationships given in the problem
def conditions (p q r s t : ℕ) :=
  valid_single_digit p ∧
  valid_single_digit q ∧
  valid_single_digit r ∧
  valid_single_digit s ∧
  valid_single_digit t ∧
  distinct_single_digits p q r s t ∧
  p - q = r ∧
  r - s = t

-- Theorem to be proven
theorem different_values_of_t : 
  ∃! (count : ℕ), count = 6 ∧ (∃ p q r s t, conditions p q r s t) := 
sorry

end different_values_of_t_l225_225924


namespace possible_values_of_n_l225_225574

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := a (b n)

noncomputable def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem possible_values_of_n (n : ℕ) :
  T n < 2021 → n = 8 ∨ n = 9 := by
  sorry

end possible_values_of_n_l225_225574


namespace math_problem_l225_225308

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h_has_derivative_f : ∀ x, Differentiable ℝ f)
variable (h_has_derivative_g : ∀ x, Differentiable ℝ g)
variable (h_deriv_ineq : ∀ x, deriv f x > deriv g x)
variable (h_interval : x ∈ Ioo a b)

theorem math_problem :
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) :=
sorry

end math_problem_l225_225308


namespace eval_exp_l225_225836

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l225_225836


namespace fraction_division_l225_225715

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := 
by
  -- We need to convert this division into multiplication by the reciprocal
  -- (3 / 4) / (2 / 5) = (3 / 4) * (5 / 2)
  -- Now perform the multiplication of the numerators and denominators
  -- (3 * 5) / (4 * 2) = 15 / 8
  sorry

end fraction_division_l225_225715


namespace manny_remaining_money_l225_225794

def cost_chair (cost_total_chairs : ℕ) (number_of_chairs : ℕ) : ℕ :=
  cost_total_chairs / number_of_chairs

def cost_table (cost_chair : ℕ) (chairs_for_table : ℕ) : ℕ :=
  cost_chair * chairs_for_table

def total_cost (cost_table : ℕ) (cost_chairs : ℕ) : ℕ :=
  cost_table + cost_chairs

def remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem manny_remaining_money : remaining_money 100 (total_cost (cost_table (cost_chair 55 5) 3) ((cost_chair 55 5) * 2)) = 45 :=
by
  sorry

end manny_remaining_money_l225_225794


namespace num_pos_four_digit_integers_l225_225690

theorem num_pos_four_digit_integers : 
  ∃ (n : ℕ), n = (Nat.factorial 4) / ((Nat.factorial 3) * (Nat.factorial 1)) ∧ n = 4 := 
by
  sorry

end num_pos_four_digit_integers_l225_225690


namespace larger_to_smaller_ratio_l225_225558

theorem larger_to_smaller_ratio (x y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end larger_to_smaller_ratio_l225_225558


namespace Matthew_initial_cakes_l225_225366

theorem Matthew_initial_cakes (n_cakes : ℕ) (n_crackers : ℕ) (n_friends : ℕ) (crackers_per_person : ℕ) :
  n_friends = 4 →
  n_crackers = 32 →
  crackers_per_person = 8 →
  n_crackers = n_friends * crackers_per_person →
  n_cakes = n_friends * crackers_per_person →
  n_cakes = 32 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  exact h5

end Matthew_initial_cakes_l225_225366


namespace constant_term_binomial_l225_225115

theorem constant_term_binomial (n : ℕ) (h : n = 5) : ∃ (r : ℕ), r = 6 ∧ (Nat.choose (2 * n) r) = 210 := by
  sorry

end constant_term_binomial_l225_225115


namespace cory_prime_sum_l225_225410

def primes_between_30_and_60 : List ℕ := [31, 37, 41, 43, 47, 53, 59]

theorem cory_prime_sum :
  let smallest := 31
  let largest := 59
  let median := 43
  smallest ∈ primes_between_30_and_60 ∧
  largest ∈ primes_between_30_and_60 ∧
  median ∈ primes_between_30_and_60 ∧
  primes_between_30_and_60 = [31, 37, 41, 43, 47, 53, 59] → 
  smallest + largest + median = 133 := 
by
  intros; sorry

end cory_prime_sum_l225_225410


namespace sample_size_is_correct_l225_225904

-- Define the conditions
def total_students : ℕ := 40 * 50
def students_selected : ℕ := 150

-- Theorem: The sample size is 150 given that 150 students are selected
theorem sample_size_is_correct : students_selected = 150 := by
  sorry  -- Proof to be completed

end sample_size_is_correct_l225_225904


namespace trees_died_l225_225349

theorem trees_died (initial_trees dead surviving : ℕ) 
  (h_initial : initial_trees = 11) 
  (h_surviving : surviving = dead + 7) 
  (h_total : dead + surviving = initial_trees) : 
  dead = 2 :=
by
  sorry

end trees_died_l225_225349


namespace area_of_annulus_l225_225462

variables (R r x : ℝ) (hRr : R > r) (h : R^2 - r^2 = x^2)

theorem area_of_annulus : π * R^2 - π * r^2 = π * x^2 :=
by
  sorry

end area_of_annulus_l225_225462


namespace calculate_sum_l225_225871

theorem calculate_sum :
  (1 : ℚ) + 3 / 6 + 5 / 12 + 7 / 20 + 9 / 30 + 11 / 42 + 13 / 56 + 15 / 72 + 17 / 90 = 81 + 2 / 5 :=
sorry

end calculate_sum_l225_225871


namespace merchant_loss_is_15_yuan_l225_225748

noncomputable def profit_cost_price : ℝ := (180 : ℝ) / 1.2
noncomputable def loss_cost_price : ℝ := (180 : ℝ) / 0.8

theorem merchant_loss_is_15_yuan :
  (180 + 180) - (profit_cost_price + loss_cost_price) = -15 := by
  sorry

end merchant_loss_is_15_yuan_l225_225748


namespace quarters_per_jar_l225_225848

/-- Jenn has 5 jars full of quarters. Each jar can hold a certain number of quarters.
    The bike costs 180 dollars, and she will have 20 dollars left over after buying it.
    Prove that each jar can hold 160 quarters. -/
theorem quarters_per_jar (num_jars : ℕ) (cost_bike : ℕ) (left_over : ℕ)
  (quarters_per_dollar : ℕ) (total_quarters : ℕ) (quarters_per_jar : ℕ) :
  num_jars = 5 → cost_bike = 180 → left_over = 20 → quarters_per_dollar = 4 →
  total_quarters = ((cost_bike + left_over) * quarters_per_dollar) →
  quarters_per_jar = (total_quarters / num_jars) →
  quarters_per_jar = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end quarters_per_jar_l225_225848


namespace rectangle_right_triangle_max_area_and_hypotenuse_l225_225914

theorem rectangle_right_triangle_max_area_and_hypotenuse (x y h : ℝ) (h_triangle : h^2 = x^2 + y^2) (h_perimeter : 2 * (x + y) = 60) :
  (x * y ≤ 225) ∧ (x = 15) ∧ (y = 15) ∧ (h = 15 * Real.sqrt 2) :=
by
  sorry

end rectangle_right_triangle_max_area_and_hypotenuse_l225_225914


namespace opposite_of_neg_2023_l225_225459

theorem opposite_of_neg_2023 :
  ∃ y : ℝ, (-2023 + y = 0) ∧ y = 2023 :=
by
  sorry

end opposite_of_neg_2023_l225_225459


namespace suzanne_donation_total_l225_225886

theorem suzanne_donation_total : 
  (10 + 10 * 2 + 10 * 2^2 + 10 * 2^3 + 10 * 2^4 = 310) :=
by
  sorry

end suzanne_donation_total_l225_225886


namespace domain_ln_x_minus_1_l225_225277

def domain_of_log_function (x : ℝ) : Prop := x > 1

theorem domain_ln_x_minus_1 (x : ℝ) : domain_of_log_function x ↔ x > 1 :=
by {
  sorry
}

end domain_ln_x_minus_1_l225_225277


namespace train_speed_l225_225511

def distance : ℕ := 500
def time : ℕ := 10
def conversion_factor : ℝ := 3.6

theorem train_speed :
  (distance / time : ℝ) * conversion_factor = 180 :=
by
  sorry

end train_speed_l225_225511


namespace solution_set_l225_225819

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) :
  f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by {
  sorry
}

end solution_set_l225_225819


namespace trigonometric_identity_l225_225829

open Real

theorem trigonometric_identity :
  (sin (20 * π / 180) * sin (80 * π / 180) - cos (160 * π / 180) * sin (10 * π / 180) = 1 / 2) :=
by
  -- Trigonometric calculations
  sorry

end trigonometric_identity_l225_225829


namespace number_of_real_solutions_of_equation_l225_225342

theorem number_of_real_solutions_of_equation :
  (∀ x : ℝ, ((2 : ℝ)^(4 * x + 2)) * ((4 : ℝ)^(2 * x + 8)) = ((8 : ℝ)^(3 * x + 7))) ↔ x = -3 :=
by sorry

end number_of_real_solutions_of_equation_l225_225342


namespace age_difference_l225_225358

variable (A : ℕ) -- Albert's age
variable (B : ℕ) -- Albert's brother's age
variable (F : ℕ) -- Father's age
variable (M : ℕ) -- Mother's age

def age_conditions : Prop :=
  (B = A - 2) ∧ (F = A + 48) ∧ (M = B + 46)

theorem age_difference (h : age_conditions A B F M) : F - M = 4 :=
by
  sorry

end age_difference_l225_225358


namespace minimum_hexagon_perimeter_l225_225408

-- Define the conditions given in the problem
def small_equilateral_triangle (side_length : ℝ) (triangle_count : ℕ) :=
  triangle_count = 57 ∧ side_length = 1

def hexagon_with_conditions (angle_condition : ℝ → Prop) :=
  ∀ θ, angle_condition θ → θ ≤ 180 ∧ θ > 0

-- State the main problem as a theorem
theorem minimum_hexagon_perimeter : ∀ n : ℕ, ∃ p : ℕ,
  (small_equilateral_triangle 1 57) → 
  (∃ angle_condition, hexagon_with_conditions angle_condition) →
  (n = 57) →
  p = 19 :=
by
  sorry

end minimum_hexagon_perimeter_l225_225408


namespace bob_second_week_hours_l225_225128

theorem bob_second_week_hours (total_earnings : ℕ) (total_hours_first_week : ℕ) (regular_hours_pay : ℕ) 
  (overtime_hours_pay : ℕ) (regular_hours_max : ℕ) (total_hours_overtime_first_week : ℕ) 
  (earnings_first_week : ℕ) (earnings_second_week : ℕ) : 
  total_earnings = 472 →
  total_hours_first_week = 44 →
  regular_hours_pay = 5 →
  overtime_hours_pay = 6 →
  regular_hours_max = 40 →
  total_hours_overtime_first_week = total_hours_first_week - regular_hours_max →
  earnings_first_week = regular_hours_max * regular_hours_pay + 
                          total_hours_overtime_first_week * overtime_hours_pay →
  earnings_second_week = total_earnings - earnings_first_week → 
  ∃ h, earnings_second_week = h * regular_hours_pay ∨ 
  earnings_second_week = (regular_hours_max * regular_hours_pay + (h - regular_hours_max) * overtime_hours_pay) ∧ 
  h = 48 :=
by 
  intros 
  sorry 

end bob_second_week_hours_l225_225128


namespace salary_for_may_l225_225188

theorem salary_for_may
  (J F M A May : ℝ)
  (h1 : J + F + M + A = 32000)
  (h2 : F + M + A + May = 34400)
  (h3 : J = 4100) :
  May = 6500 := 
by 
  sorry

end salary_for_may_l225_225188


namespace johnny_earnings_l225_225878

theorem johnny_earnings :
  let job1 := 3 * 7
  let job2 := 2 * 10
  let job3 := 4 * 12
  let daily_earnings := job1 + job2 + job3
  let total_earnings := 5 * daily_earnings
  total_earnings = 445 :=
by
  sorry

end johnny_earnings_l225_225878


namespace solve_real_eq_l225_225339

theorem solve_real_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6) ↔
  ((x ^ 3 - 3 * x ^ 2) / (x ^ 2 - 4) + 2 * x = -16) :=
by sorry

end solve_real_eq_l225_225339


namespace average_weight_increase_l225_225718

theorem average_weight_increase (W_new : ℝ) (W_old : ℝ) (num_persons : ℝ): 
  W_new = 94 ∧ W_old = 70 ∧ num_persons = 8 → 
  (W_new - W_old) / num_persons = 3 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end average_weight_increase_l225_225718


namespace andrea_reaches_lauren_in_25_minutes_l225_225956

noncomputable def initial_distance : ℝ := 30
noncomputable def decrease_rate : ℝ := 90
noncomputable def Lauren_stop_time : ℝ := 10 / 60

theorem andrea_reaches_lauren_in_25_minutes :
  ∃ v_L v_A : ℝ, v_A = 2 * v_L ∧ v_A + v_L = decrease_rate ∧ ∃ remaining_distance remaining_time final_time : ℝ, 
  remaining_distance = initial_distance - decrease_rate * Lauren_stop_time ∧ 
  remaining_time = remaining_distance / v_A ∧ 
  final_time = Lauren_stop_time + remaining_time ∧ 
  final_time * 60 = 25 :=
sorry

end andrea_reaches_lauren_in_25_minutes_l225_225956


namespace giraffes_difference_l225_225072

theorem giraffes_difference :
  ∃ n : ℕ, (300 = 3 * n) ∧ (300 - n = 200) :=
by
  sorry

end giraffes_difference_l225_225072


namespace vanya_speed_increased_by_4_l225_225378

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l225_225378


namespace problem_solution_l225_225266

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem problem_solution (x1 x2 : ℝ) 
  (hx1 : x1 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (hx2 : x2 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (h : f x1 < f x2) : x1^2 > x2^2 := 
sorry

end problem_solution_l225_225266


namespace surface_area_of_cross_shape_with_five_unit_cubes_l225_225575

noncomputable def unit_cube_surface_area : ℕ := 6
noncomputable def num_cubes : ℕ := 5
noncomputable def total_surface_area_iso_cubes : ℕ := num_cubes * unit_cube_surface_area
noncomputable def central_cube_exposed_faces : ℕ := 2
noncomputable def surrounding_cubes_exposed_faces : ℕ := 5
noncomputable def surrounding_cubes_count : ℕ := 4
noncomputable def cross_shape_surface_area : ℕ := 
  central_cube_exposed_faces + (surrounding_cubes_count * surrounding_cubes_exposed_faces)

theorem surface_area_of_cross_shape_with_five_unit_cubes : cross_shape_surface_area = 22 := 
by sorry

end surface_area_of_cross_shape_with_five_unit_cubes_l225_225575


namespace scientific_notation_142000_l225_225835

theorem scientific_notation_142000 : (142000 : ℝ) = 1.42 * 10^5 := sorry

end scientific_notation_142000_l225_225835


namespace max_of_three_numbers_l225_225064

theorem max_of_three_numbers : ∀ (a b c : ℕ), a = 10 → b = 11 → c = 12 → max (max a b) c = 12 :=
by
  intros a b c h1 h2 h3
  rw [h1, h2, h3]
  sorry

end max_of_three_numbers_l225_225064


namespace Jungkook_has_the_largest_number_l225_225553

theorem Jungkook_has_the_largest_number :
  let Yoongi := 4
  let Yuna := 5
  let Jungkook := 6 + 3
  Jungkook > Yoongi ∧ Jungkook > Yuna := by
    sorry

end Jungkook_has_the_largest_number_l225_225553


namespace find_ab_plus_a_plus_b_l225_225024

-- Define the polynomial
def quartic_poly (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x - 1

-- Define the roots conditions
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

-- State the proof problem
theorem find_ab_plus_a_plus_b :
  ∃ a b : ℝ,
    is_root quartic_poly a ∧
    is_root quartic_poly b ∧
    ab = a * b ∧
    a_plus_b = a + b ∧
    ab + a_plus_b = 4 :=
by sorry

end find_ab_plus_a_plus_b_l225_225024


namespace min_value_of_a2_b2_l225_225081

noncomputable def f (x a b : ℝ) := Real.exp x + a * x + b

theorem min_value_of_a2_b2 {a b : ℝ} (h : ∃ t ∈ Set.Icc (1 : ℝ) (3 : ℝ), f t a b = 0) :
  a^2 + b^2 ≥ (Real.exp 1)^2 / 2 :=
by
  sorry

end min_value_of_a2_b2_l225_225081


namespace courtyard_length_proof_l225_225696

noncomputable def paving_stone_area (length width : ℝ) : ℝ := length * width

noncomputable def total_area_stones (stone_area : ℝ) (num_stones : ℝ) : ℝ := stone_area * num_stones

noncomputable def courtyard_length (total_area width : ℝ) : ℝ := total_area / width

theorem courtyard_length_proof :
  let stone_length := 2.5
  let stone_width := 2
  let courtyard_width := 16.5
  let num_stones := 99
  let stone_area := paving_stone_area stone_length stone_width
  let total_area := total_area_stones stone_area num_stones
  courtyard_length total_area courtyard_width = 30 :=
by
  sorry

end courtyard_length_proof_l225_225696


namespace range_of_x_l225_225257

theorem range_of_x (θ : ℝ) (h0 : 0 < θ) (h1 : θ < Real.pi / 2) (h2 : ∀ θ, (0 < θ) → (θ < Real.pi / 2) → (1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2 ≥ abs (2 * x - 1))) :
  -4 ≤ x ∧ x ≤ 5 := sorry

end range_of_x_l225_225257


namespace assorted_candies_count_l225_225810

theorem assorted_candies_count
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (chocolate_bars : ℕ)
  (assorted_candies : ℕ) :
  total_candies = 50 →
  chewing_gums = 15 →
  chocolate_bars = 20 →
  assorted_candies = total_candies - (chewing_gums + chocolate_bars) →
  assorted_candies = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end assorted_candies_count_l225_225810


namespace insulation_cost_per_sq_ft_l225_225679

theorem insulation_cost_per_sq_ft 
  (l w h : ℤ) 
  (surface_area : ℤ := (2 * l * w) + (2 * l * h) + (2 * w * h))
  (total_cost : ℤ)
  (cost_per_sq_ft : ℤ := total_cost / surface_area)
  (h_l : l = 3)
  (h_w : w = 5)
  (h_h : h = 2)
  (h_total_cost : total_cost = 1240) :
  cost_per_sq_ft = 20 := 
by
  sorry

end insulation_cost_per_sq_ft_l225_225679


namespace triplet_sum_not_zero_l225_225532

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

theorem triplet_sum_not_zero :
  ¬ (sum_triplet 3 (-5) 2 = 0) ∧
  (sum_triplet (1/4) (1/4) (-1/2) = 0) ∧
  (sum_triplet 0.3 (-0.1) (-0.2) = 0) ∧
  (sum_triplet 0.5 (-0.3) (-0.2) = 0) ∧
  (sum_triplet (1/3) (-1/6) (-1/6) = 0) :=
by 
  sorry

end triplet_sum_not_zero_l225_225532


namespace exponentiation_property_l225_225414

variable (a : ℝ)

theorem exponentiation_property : a^2 * a^3 = a^5 := by
  sorry

end exponentiation_property_l225_225414


namespace chocolate_chips_per_cookie_l225_225379

theorem chocolate_chips_per_cookie
  (num_batches : ℕ)
  (cookies_per_batch : ℕ)
  (num_people : ℕ)
  (chocolate_chips_per_person : ℕ) :
  (num_batches = 3) →
  (cookies_per_batch = 12) →
  (num_people = 4) →
  (chocolate_chips_per_person = 18) →
  (chocolate_chips_per_person / (num_batches * cookies_per_batch / num_people) = 2) :=
by
  sorry

end chocolate_chips_per_cookie_l225_225379


namespace total_animals_received_l225_225622

-- Define the conditions
def cats : ℕ := 40
def additionalCats : ℕ := 20
def dogs : ℕ := cats - additionalCats

-- Prove the total number of animals received
theorem total_animals_received : (cats + dogs) = 60 := by
  -- The proof itself is not required in this task
  sorry

end total_animals_received_l225_225622


namespace simplest_fraction_is_D_l225_225493

def fractionA (x : ℕ) : ℚ := 10 / (15 * x)
def fractionB (a b : ℕ) : ℚ := (2 * a * b) / (3 * a * a)
def fractionC (x : ℕ) : ℚ := (x + 1) / (3 * x + 3)
def fractionD (x : ℕ) : ℚ := (x + 1) / (x * x + 1)

theorem simplest_fraction_is_D (x a b : ℕ) :
  ¬ ∃ c, c ≠ 1 ∧
    (fractionA x = (fractionA x / c) ∨
     fractionB a b = (fractionB a b / c) ∨
     fractionC x = (fractionC x / c)) ∧
    ∀ d, d ≠ 1 → fractionD x ≠ (fractionD x / d) := 
  sorry

end simplest_fraction_is_D_l225_225493


namespace unique_solution_condition_l225_225120

theorem unique_solution_condition {a b : ℝ} : (∃ x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 :=
by
  sorry

end unique_solution_condition_l225_225120


namespace carrie_pays_199_27_l225_225814

noncomputable def carrie_payment : ℝ :=
  let shirts := 8 * 12
  let pants := 4 * 25
  let jackets := 4 * 75
  let skirts := 3 * 30
  let shoes := 2 * 50
  let shirts_discount := 0.20 * shirts
  let jackets_discount := 0.20 * jackets
  let skirts_discount := 0.10 * skirts
  let total_cost := shirts + pants + jackets + skirts + shoes
  let discounted_cost := (shirts - shirts_discount) + (pants) + (jackets - jackets_discount) + (skirts - skirts_discount) + shoes
  let mom_payment := 2 / 3 * discounted_cost
  let carrie_payment := discounted_cost - mom_payment
  carrie_payment

theorem carrie_pays_199_27 : carrie_payment = 199.27 :=
by
  sorry

end carrie_pays_199_27_l225_225814


namespace simplify_expression_correct_l225_225701

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end simplify_expression_correct_l225_225701


namespace total_CDs_in_stores_l225_225692

def shelvesA := 5
def racksPerShelfA := 7
def cdsPerRackA := 8

def shelvesB := 4
def racksPerShelfB := 6
def cdsPerRackB := 7

def totalCDsA := shelvesA * racksPerShelfA * cdsPerRackA
def totalCDsB := shelvesB * racksPerShelfB * cdsPerRackB

def totalCDs := totalCDsA + totalCDsB

theorem total_CDs_in_stores :
  totalCDs = 448 := 
by 
  sorry

end total_CDs_in_stores_l225_225692


namespace simplify_expression_l225_225950

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := 
by 
  sorry

end simplify_expression_l225_225950


namespace swim_club_percentage_l225_225993

theorem swim_club_percentage (P : ℕ) (total_members : ℕ) (not_passed_taken_course : ℕ) (not_passed_not_taken_course : ℕ) :
  total_members = 50 →
  not_passed_taken_course = 5 →
  not_passed_not_taken_course = 30 →
  (total_members - (total_members * P / 100) = not_passed_taken_course + not_passed_not_taken_course) →
  P = 30 :=
by
  sorry

end swim_club_percentage_l225_225993


namespace ratio_of_boys_l225_225177

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 5) * (1 - p)) 
  : p = 3 / 8 := 
by
  sorry

end ratio_of_boys_l225_225177


namespace toms_final_stamp_count_l225_225067

-- Definitions of the given conditions

def initial_stamps : ℕ := 3000
def mike_gift : ℕ := 17
def harry_gift : ℕ := 2 * mike_gift + 10
def sarah_gift : ℕ := 3 * mike_gift - 5
def damaged_stamps : ℕ := 37

-- Statement of the goal
theorem toms_final_stamp_count :
  initial_stamps + mike_gift + harry_gift + sarah_gift - damaged_stamps = 3070 :=
by
  sorry

end toms_final_stamp_count_l225_225067


namespace determine_q_l225_225861

theorem determine_q (q : ℝ) (x1 x2 x3 x4 : ℝ) 
  (h_first_eq : x1^2 - 5 * x1 + q = 0 ∧ x2^2 - 5 * x2 + q = 0)
  (h_second_eq : x3^2 - 7 * x3 + 2 * q = 0 ∧ x4^2 - 7 * x4 + 2 * q = 0)
  (h_relation : x3 = 2 * x1) : 
  q = 6 :=
by
  sorry

end determine_q_l225_225861


namespace continuous_stripe_probability_l225_225337

-- Define a structure representing the configuration of each face.
structure FaceConfiguration where
  is_diagonal : Bool
  edge_pair_or_vertex_pair : Bool

-- Define the cube configuration.
structure CubeConfiguration where
  face1 : FaceConfiguration
  face2 : FaceConfiguration
  face3 : FaceConfiguration
  face4 : FaceConfiguration
  face5 : FaceConfiguration
  face6 : FaceConfiguration

noncomputable def total_configurations : ℕ := 4^6

-- Define the function that checks if a configuration results in a continuous stripe.
def results_in_continuous_stripe (c : CubeConfiguration) : Bool := sorry

-- Define the number of configurations resulting in a continuous stripe.
noncomputable def configurations_with_continuous_stripe : ℕ :=
  Nat.card {c : CubeConfiguration // results_in_continuous_stripe c}

-- Define the probability calculation.
noncomputable def probability_continuous_stripe : ℚ :=
  configurations_with_continuous_stripe / total_configurations

-- The statement of the problem: Prove the probability of continuous stripe is 3/256.
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 256 :=
sorry

end continuous_stripe_probability_l225_225337


namespace stock_worth_l225_225744

theorem stock_worth (W : Real) 
  (profit_part : Real := 0.25 * W * 0.20)
  (loss_part1 : Real := 0.35 * W * 0.10)
  (loss_part2 : Real := 0.40 * W * 0.15)
  (overall_loss_eq : loss_part1 + loss_part2 - profit_part = 1200) : 
  W = 26666.67 :=
by
  sorry

end stock_worth_l225_225744


namespace measure_of_one_interior_angle_of_regular_octagon_l225_225332

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l225_225332


namespace sum_of_decimals_l225_225165

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end sum_of_decimals_l225_225165


namespace smaller_circle_radius_l225_225314

theorem smaller_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : R = (2 * r) / Real.sqrt 3) : r = 5 * Real.sqrt 3 :=
by
  sorry

end smaller_circle_radius_l225_225314


namespace Seojun_apples_decimal_l225_225988

theorem Seojun_apples_decimal :
  let total_apples := 100
  let seojun_apples := 11
  seojun_apples / total_apples = 0.11 :=
by
  let total_apples := 100
  let seojun_apples := 11
  sorry

end Seojun_apples_decimal_l225_225988


namespace total_time_proof_l225_225641

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end total_time_proof_l225_225641


namespace exists_infinitely_many_m_l225_225828

theorem exists_infinitely_many_m (k : ℕ) (hk : 0 < k) : 
  ∃ᶠ m in at_top, 3 ^ k ∣ m ^ 3 + 10 :=
sorry

end exists_infinitely_many_m_l225_225828


namespace value_of_m_l225_225280

theorem value_of_m
  (m : ℝ)
  (a : ℝ × ℝ := (-1, 3))
  (b : ℝ × ℝ := (m, m - 2))
  (collinear : a.1 * b.2 = a.2 * b.1) :
  m = 1 / 2 :=
sorry

end value_of_m_l225_225280


namespace quadratic_polynomial_AT_BT_l225_225262

theorem quadratic_polynomial_AT_BT (p s : ℝ) :
  ∃ (AT BT : ℝ), (AT + BT = p + 3) ∧ (AT * BT = s^2) ∧ (∀ (x : ℝ), (x^2 - (p+3) * x + s^2) = (x - AT) * (x - BT)) := 
sorry

end quadratic_polynomial_AT_BT_l225_225262


namespace range_of_m_l225_225627

open Real

theorem range_of_m 
    (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (m : ℝ)
    (h : m * (a + 1/a) / sqrt 2 > 1) : 
    m ≥ sqrt 2 / 2 :=
sorry

end range_of_m_l225_225627


namespace min_a_plus_b_l225_225160

theorem min_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : a + b >= 4 :=
sorry

end min_a_plus_b_l225_225160


namespace passes_through_point_l225_225564

theorem passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end passes_through_point_l225_225564


namespace second_rooster_weight_l225_225634

theorem second_rooster_weight (cost_per_kg : ℝ) (weight_1 : ℝ) (total_earnings : ℝ) (weight_2 : ℝ) :
  cost_per_kg = 0.5 →
  weight_1 = 30 →
  total_earnings = 35 →
  total_earnings = weight_1 * cost_per_kg + weight_2 * cost_per_kg →
  weight_2 = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end second_rooster_weight_l225_225634


namespace maximize_side_area_of_cylinder_l225_225094

noncomputable def radius_of_cylinder (x : ℝ) : ℝ :=
  (6 - x) / 3

noncomputable def side_area_of_cylinder (x : ℝ) : ℝ :=
  2 * Real.pi * (radius_of_cylinder x) * x

theorem maximize_side_area_of_cylinder :
  ∃ x : ℝ, (0 < x ∧ x < 6) ∧ (∀ y : ℝ, (0 < y ∧ y < 6) → (side_area_of_cylinder y ≤ side_area_of_cylinder x)) ∧ x = 3 :=
by
  sorry

end maximize_side_area_of_cylinder_l225_225094


namespace ellipse_foci_on_y_axis_l225_225710

theorem ellipse_foci_on_y_axis (k : ℝ) (h1 : 5 + k > 3 - k) (h2 : 3 - k > 0) (h3 : 5 + k > 0) : -1 < k ∧ k < 3 :=
by 
  sorry

end ellipse_foci_on_y_axis_l225_225710


namespace not_less_than_x3_y5_for_x2y_l225_225400

theorem not_less_than_x3_y5_for_x2y (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : x^2 * y ≥ x^3 + y^5 :=
sorry

end not_less_than_x3_y5_for_x2y_l225_225400


namespace sum_of_number_and_square_is_306_l225_225951

theorem sum_of_number_and_square_is_306 (n : ℕ) (h : n = 17) : n + n^2 = 306 :=
by
  sorry

end sum_of_number_and_square_is_306_l225_225951


namespace equilateral_triangle_roots_l225_225418

theorem equilateral_triangle_roots (p q : ℂ) (z1 z2 : ℂ) (h1 : z2 = Complex.exp (2 * Real.pi * Complex.I / 3) * z1)
  (h2 : 0 + p * z1 + q = 0) (h3 : p = -z1 - z2) (h4 : q = z1 * z2) : (p^2 / q) = 1 :=
by
  sorry

end equilateral_triangle_roots_l225_225418


namespace simplify_and_evaluate_l225_225973

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : ((2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1))) = 1 / 2 := by
  sorry

end simplify_and_evaluate_l225_225973


namespace winning_votes_cast_l225_225148

variable (V : ℝ) -- Total number of votes (real number)
variable (winner_votes_ratio : ℝ) -- Ratio for winner's votes
variable (votes_difference : ℝ) -- Vote difference due to winning

-- Conditions given
def election_conditions (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) : Prop :=
  winner_votes_ratio = 0.54 ∧
  votes_difference = 288

-- Proof problem: Proving the number of votes cast to the winning candidate is 1944
theorem winning_votes_cast (V : ℝ) (winner_votes_ratio : ℝ) (votes_difference : ℝ) 
  (h : election_conditions V winner_votes_ratio votes_difference) :
  winner_votes_ratio * V = 1944 :=
by
  sorry

end winning_votes_cast_l225_225148


namespace william_time_on_road_l225_225687

-- Define departure and arrival times
def departure_time := 7 -- 7:00 AM
def arrival_time := 20 -- 8:00 PM in 24-hour format

-- Define stop times in minutes
def stop1 := 25
def stop2 := 10
def stop3 := 25

-- Define total journey time in hours
def total_travel_time := arrival_time - departure_time

-- Define total stop time in hours
def total_stop_time := (stop1 + stop2 + stop3) / 60

-- Define time spent on the road
def time_on_road := total_travel_time - total_stop_time

-- The theorem to prove
theorem william_time_on_road : time_on_road = 12 := by
  sorry

end william_time_on_road_l225_225687


namespace dividend_rate_is_16_l225_225656

noncomputable def dividend_rate_of_shares : ℝ :=
  let share_value := 48
  let interest_rate := 0.12
  let market_value := 36.00000000000001
  (interest_rate * share_value) / market_value * 100

theorem dividend_rate_is_16 :
  dividend_rate_of_shares = 16 := by
  sorry

end dividend_rate_is_16_l225_225656


namespace find_four_real_numbers_l225_225536

theorem find_four_real_numbers (x1 x2 x3 x4 : ℝ) :
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end find_four_real_numbers_l225_225536


namespace divisible_by_11_of_sum_divisible_l225_225737

open Int

theorem divisible_by_11_of_sum_divisible (a b : ℤ) (h : 11 ∣ (a^2 + b^2)) : 11 ∣ a ∧ 11 ∣ b :=
sorry

end divisible_by_11_of_sum_divisible_l225_225737


namespace totalPeoplePresent_l225_225045

-- Defining the constants based on the problem conditions
def associateProfessors := 2
def assistantProfessors := 7

def totalPencils := 11
def totalCharts := 16

-- The main proof statement
theorem totalPeoplePresent :
  (∃ (A B : ℕ), (2 * A + B = totalPencils) ∧ (A + 2 * B = totalCharts)) →
  (associateProfessors + assistantProfessors = 9) :=
  by
  sorry

end totalPeoplePresent_l225_225045


namespace cube_triangulation_impossible_l225_225942

theorem cube_triangulation_impossible (vertex_sum : ℝ) (triangle_inter_sum : ℝ) (triangle_sum : ℝ) :
  vertex_sum = 270 ∧ triangle_inter_sum = 360 ∧ triangle_sum = 180 → ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), m ≠ 3 → false :=
by
  sorry

end cube_triangulation_impossible_l225_225942


namespace unique_sum_of_cubes_lt_1000_l225_225719

theorem unique_sum_of_cubes_lt_1000 : 
  let max_cube := 11 
  let max_val := 1000 
  ∃ n : ℕ, n = 35 ∧ ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ max_cube → 1 ≤ b ∧ b ≤ max_cube → a^3 + b^3 < max_val :=
sorry

end unique_sum_of_cubes_lt_1000_l225_225719


namespace enemy_defeat_points_l225_225452

theorem enemy_defeat_points 
    (points_per_enemy : ℕ) (total_enemies : ℕ) (undefeated_enemies : ℕ) (defeated : ℕ) (points_earned : ℕ) :
    points_per_enemy = 8 →
    total_enemies = 7 →
    undefeated_enemies = 2 →
    defeated = total_enemies - undefeated_enemies →
    points_earned = defeated * points_per_enemy →
    points_earned = 40 :=
by
  intros
  sorry

end enemy_defeat_points_l225_225452


namespace number_of_equilateral_triangles_in_lattice_l225_225946

-- Definitions representing the conditions of the problem
def is_unit_distance (a b : ℕ) : Prop :=
  true -- Assume true as we are not focusing on the definition

def expanded_hexagonal_lattice (p : ℕ) : Prop :=
  true -- Assume true as the specific construction details are abstracted

-- The target theorem statement
theorem number_of_equilateral_triangles_in_lattice 
  (lattice : ℕ → Prop) (dist : ℕ → ℕ → Prop) 
  (h₁ : ∀ p, lattice p → dist p p) 
  (h₂ : ∀ p, (expanded_hexagonal_lattice p) ↔ lattice p ∧ dist p p) : 
  ∃ n, n = 32 :=
by 
  existsi 32
  sorry

end number_of_equilateral_triangles_in_lattice_l225_225946


namespace probability_red_blue_l225_225164

-- Declare the conditions (probabilities for white, green and yellow marbles).
variables (total_marbles : ℕ) (P_white P_green P_yellow P_red_blue : ℚ)
-- implicitly P_white, P_green, P_yellow, P_red_blue are probabilities, therefore between 0 and 1

-- Assume the conditions given in the problem
axiom total_marbles_condition : total_marbles = 250
axiom P_white_condition : P_white = 2 / 5
axiom P_green_condition : P_green = 1 / 4
axiom P_yellow_condition : P_yellow = 1 / 10

-- Proving the required probability of red or blue marbles
theorem probability_red_blue :
  P_red_blue = 1 - (P_white + P_green + P_yellow) :=
sorry

end probability_red_blue_l225_225164


namespace compound_interest_amount_l225_225451

theorem compound_interest_amount:
  let SI := (5250 * 4 * 2) / 100
  let CI := 2 * SI
  let P := 420 / 0.21 
  CI = P * ((1 + 0.1) ^ 2 - 1) →
  SI = 210 →
  CI = 420 →
  P = 2000 :=
by
  sorry

end compound_interest_amount_l225_225451


namespace measure_smaller_angle_east_northwest_l225_225885

/-- A mathematical structure for a circle with 12 rays forming congruent central angles. -/
structure CircleWithRays where
  rays : Finset (Fin 12)  -- There are 12 rays
  congruent_angles : ∀ i, i ∈ rays

/-- The measure of the central angle formed by each ray is 30 degrees (since 360/12 = 30). -/
def central_angle_measure : ℝ := 30

/-- The measure of the smaller angle formed between the ray pointing East and the ray pointing Northwest is 150 degrees. -/
theorem measure_smaller_angle_east_northwest (c : CircleWithRays) : 
  ∃ angle : ℝ, angle = 150 := by
  sorry

end measure_smaller_angle_east_northwest_l225_225885


namespace sum_of_possible_values_l225_225228

variable (N K : ℝ)

theorem sum_of_possible_values (h1 : N ≠ 0) (h2 : N - (3 / N) = K) : N + (K / N) = K := 
sorry

end sum_of_possible_values_l225_225228


namespace cost_price_of_one_toy_l225_225990

-- Definitions translating the conditions into Lean
def total_revenue (toys_sold : ℕ) (price_per_toy : ℕ) : ℕ := toys_sold * price_per_toy
def gain (cost_per_toy : ℕ) (toys_gained : ℕ) : ℕ := cost_per_toy * toys_gained

-- Given the conditions in the problem
def total_cost_price_of_sold_toys := 18 * (1300 : ℕ)
def gain_from_sale := 3 * (1300 : ℕ)
def selling_price := total_cost_price_of_sold_toys + gain_from_sale

-- The target theorem we want to prove
theorem cost_price_of_one_toy : (selling_price = 27300) → (1300 = 27300 / 21) :=
by
  intro h
  sorry

end cost_price_of_one_toy_l225_225990


namespace cement_bought_l225_225945

-- Define the three conditions given in the problem
def original_cement : ℕ := 98
def son_contribution : ℕ := 137
def total_cement : ℕ := 450

-- Using those conditions, state that the amount of cement he bought is 215 lbs
theorem cement_bought :
  original_cement + son_contribution = 235 ∧ total_cement - (original_cement + son_contribution) = 215 := 
by {
  sorry
}

end cement_bought_l225_225945


namespace parallelogram_area_l225_225840

noncomputable def area_parallelogram (b s θ : ℝ) : ℝ := b * (s * Real.sin θ)

theorem parallelogram_area : area_parallelogram 20 10 (Real.pi / 6) = 100 := by
  sorry

end parallelogram_area_l225_225840


namespace find_d_l225_225626

noncomputable def d_value (a b c : ℝ) := (2 * a + 2 * b + 2 * c - (3 / 4)^2) / 3

theorem find_d (a b c d : ℝ) (h : 2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + (2 * a + 2 * b + 2 * c - 3 * d)^(1/2)) : 
  d = 23 / 48 :=
sorry

end find_d_l225_225626


namespace alex_singles_percentage_l225_225765

theorem alex_singles_percentage (total_hits home_runs triples doubles: ℕ) 
  (h1 : total_hits = 50) 
  (h2 : home_runs = 2) 
  (h3 : triples = 3) 
  (h4 : doubles = 10) :
  ((total_hits - (home_runs + triples + doubles)) / total_hits : ℚ) * 100 = 70 := 
by
  sorry

end alex_singles_percentage_l225_225765


namespace white_clothing_probability_l225_225573

theorem white_clothing_probability (total_athletes sample_size k_min k_max : ℕ) 
  (red_upper_bound white_upper_bound yellow_upper_bound sampled_start_interval : ℕ)
  (h_total : total_athletes = 600)
  (h_sample : sample_size = 50)
  (h_intervals : total_athletes / sample_size = 12)
  (h_group_start : sampled_start_interval = 4)
  (h_red_upper : red_upper_bound = 311)
  (h_white_upper : white_upper_bound = 496)
  (h_yellow_upper : yellow_upper_bound = 600)
  (h_k_min : k_min = 26)   -- Calculated from 312 <= 12k + 4
  (h_k_max : k_max = 41)  -- Calculated from 12k + 4 <= 496
  : (k_max - k_min + 1) / sample_size = 8 / 25 := 
by
  sorry

end white_clothing_probability_l225_225573


namespace max_lateral_surface_area_l225_225205

theorem max_lateral_surface_area (x y : ℝ) (h₁ : x + y = 10) : 
  2 * π * x * y ≤ 50 * π :=
by
  sorry

end max_lateral_surface_area_l225_225205


namespace colton_stickers_left_l225_225525

theorem colton_stickers_left :
  let C := 72
  let F := 4 * 3 -- stickers given to three friends
  let M := F + 2 -- stickers given to Mandy
  let J := M - 10 -- stickers given to Justin
  let T := F + M + J -- total stickers given away
  C - T = 42 := by
  sorry

end colton_stickers_left_l225_225525


namespace min_value_of_expression_l225_225716

theorem min_value_of_expression : ∀ x : ℝ, ∃ (M : ℝ), (∀ x, 16^x - 4^x - 4^(x+1) + 3 ≥ M) ∧ M = -4 :=
by
  sorry

end min_value_of_expression_l225_225716


namespace find_f_10_l225_225113

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : f x = f (1 / x) * Real.log x + 10

theorem find_f_10 : f 10 = 10 :=
by
  sorry

end find_f_10_l225_225113


namespace arccos_proof_l225_225123

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l225_225123


namespace triangle_angle_sum_property_l225_225470

theorem triangle_angle_sum_property (A B C : ℝ) (h1: C = 3 * B) (h2: B = 15) : A = 120 :=
by
  -- Proof goes here
  sorry

end triangle_angle_sum_property_l225_225470


namespace set_difference_is_single_element_l225_225647

-- Define the sets M and N based on the given conditions
def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}
def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

-- State the theorem that we need to prove
theorem set_difference_is_single_element : (N \ M) = {2003} :=
sorry

end set_difference_is_single_element_l225_225647


namespace oli_scoops_l225_225012

theorem oli_scoops : ∃ x : ℤ, ∀ y : ℤ, y = 2 * x ∧ y = x + 4 → x = 4 :=
by
  sorry

end oli_scoops_l225_225012


namespace number_of_children_riding_tricycles_l225_225083

-- Definitions
def bicycles_wheels := 2
def tricycles_wheels := 3

def adults := 6
def total_wheels := 57

-- Problem statement
theorem number_of_children_riding_tricycles (c : ℕ) (H : 12 + 3 * c = total_wheels) : c = 15 :=
by
  sorry

end number_of_children_riding_tricycles_l225_225083


namespace magnitude_of_angle_B_value_of_k_l225_225920

-- Define the conditions and corresponding proofs

variable {a b c : ℝ}
variable {A B C : ℝ} -- Angles in the triangle
variable (k : ℝ) -- Define k
variable (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) -- Given condition for part 1
variable (h2 : (A + B + C) = Real.pi) -- Angle sum in triangle
variable (h3 : k > 1) -- Condition for part 2
variable (m_dot_n_max : ∀ (t : ℝ), 4 * k * t + Real.cos (2 * Real.arcsin t) = 5) -- Given condition for part 2

-- Proofs Required

theorem magnitude_of_angle_B (hA : 0 < A ∧ A < Real.pi) : B = Real.pi / 3 :=
by 
  sorry -- proof to be completed

theorem value_of_k : k = 3 / 2 :=
by 
  sorry -- proof to be completed

end magnitude_of_angle_B_value_of_k_l225_225920


namespace max_marks_l225_225259

theorem max_marks (M : ℝ) (pass_percent : ℝ) (obtained_marks : ℝ) (failed_by : ℝ) (pass_marks : ℝ) 
  (h1 : pass_percent = 0.40) 
  (h2 : obtained_marks = 150) 
  (h3 : failed_by = 50) 
  (h4 : pass_marks = 200) 
  (h5 : pass_marks = obtained_marks + failed_by) 
  : M = 500 :=
by 
  -- Placeholder for the proof
  sorry

end max_marks_l225_225259


namespace total_cost_of_tickets_l225_225542

def number_of_adults := 2
def number_of_children := 3
def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6

theorem total_cost_of_tickets :
  let total_cost := number_of_adults * cost_of_adult_ticket + number_of_children * cost_of_child_ticket
  total_cost = 77 :=
by
  sorry

end total_cost_of_tickets_l225_225542


namespace total_practice_hours_l225_225874

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l225_225874


namespace john_ate_2_bags_for_dinner_l225_225881

variable (x y : ℕ)
variable (h1 : x + y = 3)
variable (h2 : y ≥ 1)

theorem john_ate_2_bags_for_dinner : x = 2 := 
by sorry

end john_ate_2_bags_for_dinner_l225_225881


namespace license_plate_count_l225_225441

theorem license_plate_count : 
  let vowels := 5
  let consonants := 21
  let digits := 10
  21 * 21 * 5 * 5 * 10 = 110250 := 
by 
  sorry

end license_plate_count_l225_225441


namespace taxi_fare_l225_225292

theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let starting_price := 6
  let additional_fare_per_km := 1.4
  let fare := starting_price + additional_fare_per_km * (x - 3)
  fare = 1.4 * x + 1.8 :=
by
  sorry

end taxi_fare_l225_225292


namespace evaluate_f_at_1_l225_225471

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem evaluate_f_at_1 : f 1 = 6 := 
  sorry

end evaluate_f_at_1_l225_225471


namespace sum_even_integers_l225_225745

theorem sum_even_integers (sum_first_50_even : Nat) (sum_from_100_to_200 : Nat) : 
  sum_first_50_even = 2550 → sum_from_100_to_200 = 7550 :=
by
  sorry

end sum_even_integers_l225_225745


namespace value_of_a_l225_225159

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B : Set ℝ := {1, 7}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B ⊆ A a) : a = 7 :=
sorry

end value_of_a_l225_225159


namespace volume_of_mixture_l225_225966

section
variable (Va Vb Vtotal : ℝ)

theorem volume_of_mixture :
  (Va / Vb = 3 / 2) →
  (800 * Va + 850 * Vb = 2460) →
  (Vtotal = Va + Vb) →
  Vtotal = 2.998 :=
by
  intros h1 h2 h3
  sorry
end

end volume_of_mixture_l225_225966


namespace find_original_price_l225_225384

variable (P : ℝ)

def final_price (discounted_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : Prop :=
  discounted_price = (1 - discount_rate) * original_price

theorem find_original_price (h1 : final_price 120 0.4 P) : P = 200 := 
by
  sorry

end find_original_price_l225_225384


namespace probability_of_purple_marble_l225_225506

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) 
  (h_blue : p_blue = 0.3) 
  (h_green : p_green = 0.4) 
  (h_sum : p_blue + p_green + p_purple = 1) : 
  p_purple = 0.3 := 
by 
  -- proof goes here
  sorry

end probability_of_purple_marble_l225_225506


namespace songs_listened_l225_225260

theorem songs_listened (x y : ℕ) 
  (h1 : y = 9) 
  (h2 : y = 2 * (Nat.sqrt x) - 5) 
  : y + x = 58 := 
  sorry

end songs_listened_l225_225260


namespace selection_ways_l225_225265

-- Step a): Define the conditions
def number_of_boys := 26
def number_of_girls := 24

-- Step c): State the problem
theorem selection_ways :
  number_of_boys + number_of_girls = 50 := by
  sorry

end selection_ways_l225_225265


namespace parallel_lines_condition_l225_225497

theorem parallel_lines_condition (a : ℝ) (l : ℝ) :
  (∀ (x y : ℝ), ax + 3*y + 3 = 0 → x + (a - 2)*y + l = 0 → a = -1) ∧ (a = -1 → ∀ (x y : ℝ), (ax + 3*y + 3 = 0 ↔ x + (a - 2)*y + l = 0)) :=
sorry

end parallel_lines_condition_l225_225497


namespace no_real_solution_l225_225888

-- Given conditions as definitions in Lean 4
def eq1 (x : ℝ) : Prop := x^5 + 3 * x^4 + 5 * x^3 + 5 * x^2 + 6 * x + 2 = 0
def eq2 (x : ℝ) : Prop := x^3 + 3 * x^2 + 4 * x + 1 = 0

-- The theorem to prove
theorem no_real_solution : ¬ ∃ x : ℝ, eq1 x ∧ eq2 x :=
by sorry

end no_real_solution_l225_225888


namespace max_value_of_f_range_of_m_l225_225866

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem max_value_of_f (a b : ℝ) (x : ℝ) (h1 : 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_tangent : ∀ (x : ℝ), f a b x - ((-1/2) * x + (Real.log 1 - 1/2)) = 0) : 
  ∃ x_max, f a b x_max = -1/2 := sorry

theorem range_of_m (m : ℝ) 
  (h_ineq : ∀ (a : ℝ) (x : ℝ), 1 ≤ a ∧ a ≤ 3 / 2 ∧ 1 ≤ x ∧ x ≤ Real.exp 2 → a * Real.log x ≥ m + x) : 
  m ≤ 2 - Real.exp 2 := sorry

end max_value_of_f_range_of_m_l225_225866


namespace negation_equivalence_l225_225922

-- Define the proposition P stating 'there exists an x in ℝ such that x^2 - 2x + 4 > 0'
def P : Prop := ∃ x : ℝ, x^2 - 2*x + 4 > 0

-- Define the proposition Q which is the negation of proposition P
def Q : Prop := ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0

-- State the proof problem: Prove that the negation of proposition P is equivalent to proposition Q
theorem negation_equivalence : ¬ P ↔ Q := by
  -- Proof to be provided.
  sorry

end negation_equivalence_l225_225922


namespace integer_roots_7_values_of_a_l225_225879

theorem integer_roots_7_values_of_a :
  (∃ a : ℝ, (∀ r s : ℤ, (r + s = -a ∧ (r * s = 8 * a))) ∧ (∃ n : ℕ, n = 7)) :=
sorry

end integer_roots_7_values_of_a_l225_225879


namespace rectangle_ratio_l225_225702

-- Given conditions
variable (w : ℕ) -- width is a natural number

-- Definitions based on conditions 
def length := 10
def perimeter := 30

-- Theorem to prove
theorem rectangle_ratio (h : 2 * length + 2 * w = perimeter) : w = 5 ∧ 1 = 1 ∧ 2 = 2 :=
by
  sorry

end rectangle_ratio_l225_225702


namespace sum_geometric_series_l225_225077

noncomputable def S_n (n : ℕ) : ℝ :=
  3 - 3 * ((2 / 3)^n)

theorem sum_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) (h_a : a = 1) (h_r : r = 2 / 3) :
  S_n n = a * (1 - r^n) / (1 - r) :=
by
  sorry

end sum_geometric_series_l225_225077


namespace divisible_l225_225559

def P (x : ℝ) : ℝ := 6 * x^3 + x^2 - 1
def Q (x : ℝ) : ℝ := 2 * x - 1

theorem divisible : ∃ R : ℝ → ℝ, ∀ x : ℝ, P x = Q x * R x :=
sorry

end divisible_l225_225559


namespace supplement_of_complement_of_35_degree_angle_l225_225596

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l225_225596


namespace fg_of_2_eq_15_l225_225475

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2_eq_15 : f (g 2) = 15 :=
by
  -- The detailed proof would go here
  sorry

end fg_of_2_eq_15_l225_225475


namespace find_c_l225_225877

theorem find_c (x c : ℝ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 15 = -3) : c = -12 := 
by
  -- Equations and conditions
  have h1 : 3 * x + 8 = 5 := h1
  have h2 : c * x - 15 = -3 := h2
  -- The proof script would go here
  sorry

end find_c_l225_225877


namespace b_divisible_by_a_l225_225251

theorem b_divisible_by_a (a b c : ℕ) (ha : a > 1) (hbc : b > c ∧ c > 1) (hdiv : (abc + 1) % (ab - b + 1) = 0) : a ∣ b :=
  sorry

end b_divisible_by_a_l225_225251


namespace range_of_k_l225_225994

-- Definitions to use in statement
variable (k : ℝ)

-- Statement: Proving the range of k
theorem range_of_k (h : ∀ x : ℝ, k * x^2 - k * x - 1 < 0) : -4 < k ∧ k ≤ 0 :=
  sorry

end range_of_k_l225_225994


namespace x_y_difference_is_perfect_square_l225_225407

theorem x_y_difference_is_perfect_square (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℕ, k^2 = x - y :=
by {sorry}

end x_y_difference_is_perfect_square_l225_225407


namespace knight_liar_grouping_l225_225331

noncomputable def can_be_partitioned_into_knight_liar_groups (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : Prop :=
  ∃ t : ℕ, n = (m + 1) * t

-- Show that if the company has n people, where n ≥ 2, and there exists at least one knight,
-- then n can be partitioned into groups where each group contains 1 knight and m liars.
theorem knight_liar_grouping (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : can_be_partitioned_into_knight_liar_groups n m h1 h2 :=
sorry

end knight_liar_grouping_l225_225331


namespace sequence_bound_l225_225215

/-- This definition states that given the initial conditions and recurrence relation
for a sequence of positive integers, the 2021st term is greater than 2^2019. -/
theorem sequence_bound (a : ℕ → ℕ) (h_initial : a 2 > a 1)
  (h_recurrence : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 2021 > 2 ^ 2019 :=
sorry

end sequence_bound_l225_225215


namespace highest_temperature_l225_225842

theorem highest_temperature (lowest_temp : ℝ) (max_temp_diff : ℝ) :
  lowest_temp = 18 → max_temp_diff = 4 → lowest_temp + max_temp_diff = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end highest_temperature_l225_225842


namespace chi_square_association_l225_225978

theorem chi_square_association (k : ℝ) :
  (k > 3.841 → (∃ A B, A ∧ B)) ∧ (k ≤ 2.076 → (∃ A B, ¬(A ∧ B))) :=
by
  sorry

end chi_square_association_l225_225978


namespace minimum_radius_third_sphere_l225_225583

-- Definitions for the problem
def height_cone := 4
def base_radius_cone := 3
def cos_alpha := 4 / 5
def radius_identical_sphere := 4 / 3
def cos_beta := 1 -- since beta is maximized

-- Define the required minimum radius for the third sphere based on the given conditions
theorem minimum_radius_third_sphere :
  ∃ x : ℝ, x = 27 / 35 ∧
    (height_cone = 4) ∧ 
    (base_radius_cone = 3) ∧ 
    (cos_alpha = 4 / 5) ∧ 
    (radius_identical_sphere = 4 / 3) ∧ 
    (cos_beta = 1) :=
sorry

end minimum_radius_third_sphere_l225_225583


namespace complement_U_B_eq_D_l225_225481

def B (x : ℝ) : Prop := x^2 - 3 * x + 2 < 0
def U : Set ℝ := Set.univ
def complement_U_B : Set ℝ := U \ {x | B x}

theorem complement_U_B_eq_D : complement_U_B = {x | x ≤ 1 ∨ x ≥ 2} := by
  sorry

end complement_U_B_eq_D_l225_225481


namespace main_theorem_l225_225764

-- Define the distribution
def P0 : ℝ := 0.4
def P2 : ℝ := 0.4
def P1 (p : ℝ) : ℝ := p

-- Define a hypothesis that the sum of probabilities is 1
def prob_sum_eq_one (p : ℝ) : Prop := P0 + P1 p + P2 = 1

-- Define the expected value of X
def E_X (p : ℝ) : ℝ := 0 * P0 + 1 * P1 p + 2 * P2

-- Define variance computation
def variance (p : ℝ) : ℝ := P0 * (0 - E_X p) ^ 2 + P1 p * (1 - E_X p) ^ 2 + P2 * (2 - E_X p) ^ 2

-- State the main theorem
theorem main_theorem : (∃ p : ℝ, prob_sum_eq_one p) ∧ variance 0.2 = 0.8 :=
by
  sorry

end main_theorem_l225_225764


namespace complex_quadratic_solution_l225_225531

theorem complex_quadratic_solution (a b : ℝ) (h₁ : ∀ (x : ℂ), 5 * x ^ 2 - 4 * x + 20 = 0 → x = a + b * Complex.I ∨ x = a - b * Complex.I) :
 a + b ^ 2 = 394 / 25 := 
sorry

end complex_quadratic_solution_l225_225531


namespace journey_speed_l225_225561

theorem journey_speed (v : ℝ) 
  (h1 : 3 * v + 60 * 2 = 240)
  (h2 : 3 + 2 = 5) :
  v = 40 :=
by
  sorry

end journey_speed_l225_225561


namespace cornelia_travel_countries_l225_225438

theorem cornelia_travel_countries (europe south_america asia half_remaining : ℕ) 
  (h1 : europe = 20)
  (h2 : south_america = 10)
  (h3 : asia = 6)
  (h4 : asia = half_remaining / 2) : 
  europe + south_america + half_remaining = 42 :=
by
  sorry

end cornelia_travel_countries_l225_225438


namespace arithmetic_operations_result_eq_one_over_2016_l225_225376

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end arithmetic_operations_result_eq_one_over_2016_l225_225376


namespace number_of_hockey_players_l225_225777

theorem number_of_hockey_players 
  (cricket_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (hockey_players : ℕ) 
  (h1 : cricket_players = 10) 
  (h2 : football_players = 16) 
  (h3 : softball_players = 13) 
  (h4 : total_players = 51) 
  (calculation : hockey_players = total_players - (cricket_players + football_players + softball_players)) : 
  hockey_players = 12 :=
by 
  rw [h1, h2, h3, h4] at calculation
  exact calculation

end number_of_hockey_players_l225_225777


namespace percent_of_a_is_4b_l225_225287

variables (a b : ℝ)
theorem percent_of_a_is_4b (h : a = 2 * b) : 4 * b / a = 2 :=
by 
  sorry

end percent_of_a_is_4b_l225_225287


namespace prove_values_of_a_and_b_prove_range_of_k_l225_225550

variable {f : ℝ → ℝ}

-- (1) Prove values of a and b
theorem prove_values_of_a_and_b (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  (∀ x, f x = 2 * x - 1) := by
sorry

-- (2) Prove range of k
theorem prove_range_of_k (h_fx_2x_minus_1 : ∀ x : ℝ, f x = 2 * x - 1) :
  (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1 / 3 := by
sorry

end prove_values_of_a_and_b_prove_range_of_k_l225_225550


namespace reduced_rates_apply_two_days_l225_225203

-- Definition of total hours in a week
def total_hours_in_week : ℕ := 7 * 24

-- Given fraction of the week with reduced rates
def reduced_rate_fraction : ℝ := 0.6428571428571429

-- Total hours covered by reduced rates
def reduced_rate_hours : ℝ := reduced_rate_fraction * total_hours_in_week

-- Hours per day with reduced rates on weekdays (8 p.m. to 8 a.m.)
def hours_weekday_night : ℕ := 12

-- Total weekdays with reduced rates
def total_weekdays : ℕ := 5

-- Total reduced rate hours on weekdays
def reduced_rate_hours_weekdays : ℕ := total_weekdays * hours_weekday_night

-- Remaining hours for 24 hour reduced rates
def remaining_reduced_rate_hours : ℝ := reduced_rate_hours - reduced_rate_hours_weekdays

-- Prove that the remaining reduced rate hours correspond to exactly 2 full days
theorem reduced_rates_apply_two_days : remaining_reduced_rate_hours = 2 * 24 := 
by
  sorry

end reduced_rates_apply_two_days_l225_225203


namespace frac_sum_is_one_l225_225227

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l225_225227


namespace students_received_B_l225_225984

/-!
# Problem Statement

Given:
1. In Mr. Johnson's class, 18 out of 30 students received a B.
2. Ms. Smith has 45 students in total, and the ratio of students receiving a B is the same as in Mr. Johnson's class.
Prove:
27 students in Ms. Smith's class received a B.
-/

theorem students_received_B (s1 s2 b1 : ℕ) (r1 : ℚ) (r2 : ℕ) (h₁ : s1 = 30) (h₂ : b1 = 18) (h₃ : s2 = 45) (h₄ : r1 = 3/5) 
(H : (b1 : ℚ) / s1 = r1) : r2 = 27 :=
by
  -- Conditions provided
  -- h₁ : s1 = 30
  -- h₂ : b1 = 18
  -- h₃ : s2 = 45
  -- h₄ : r1 = 3/5
  -- H : (b1 : ℚ) / s1 = r1
  sorry

end students_received_B_l225_225984


namespace symmetric_circle_eqn_l225_225351

theorem symmetric_circle_eqn (x y : ℝ) :
  (∃ (x0 y0 : ℝ), (x - 2)^2 + (y - 2)^2 = 7 ∧ x + y = 2) → x^2 + y^2 = 7 :=
by
  sorry

end symmetric_circle_eqn_l225_225351


namespace complex_pure_imaginary_solution_l225_225868

theorem complex_pure_imaginary_solution (m : ℝ) 
  (h_real_part : m^2 + 2*m - 3 = 0) 
  (h_imaginary_part : m - 1 ≠ 0) : 
  m = -3 :=
sorry

end complex_pure_imaginary_solution_l225_225868


namespace intersection_with_y_axis_l225_225302

theorem intersection_with_y_axis :
  ∃ (x y : ℝ), x = 0 ∧ y = 5 * x - 6 ∧ (x, y) = (0, -6) := 
sorry

end intersection_with_y_axis_l225_225302


namespace yellow_tiled_area_is_correct_l225_225466

noncomputable def length : ℝ := 3.6
noncomputable def width : ℝ := 2.5 * length
noncomputable def total_area : ℝ := length * width
noncomputable def yellow_tiled_area : ℝ := total_area / 2

theorem yellow_tiled_area_is_correct (length_eq : length = 3.6)
    (width_eq : width = 2.5 * length)
    (total_area_eq : total_area = length * width)
    (yellow_area_eq : yellow_tiled_area = total_area / 2) :
    yellow_tiled_area = 16.2 := 
by sorry

end yellow_tiled_area_is_correct_l225_225466


namespace each_car_has_4_wheels_l225_225500
-- Import necessary libraries

-- Define the conditions
def number_of_guests := 40
def number_of_parent_cars := 2
def wheels_per_parent_car := 4
def number_of_guest_cars := 10
def total_wheels := 48
def parent_car_wheels := number_of_parent_cars * wheels_per_parent_car
def guest_car_wheels := total_wheels - parent_car_wheels

-- Define the proposition to prove
theorem each_car_has_4_wheels : (guest_car_wheels / number_of_guest_cars) = 4 :=
by
  sorry

end each_car_has_4_wheels_l225_225500


namespace sum_of_x_and_y_l225_225365

theorem sum_of_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
    (hx : ∃ (a : ℕ), 720 * x = a^2)
    (hy : ∃ (b : ℕ), 720 * y = b^4) :
    x + y = 1130 :=
sorry

end sum_of_x_and_y_l225_225365


namespace towel_bleach_decrease_l225_225406

theorem towel_bleach_decrease (L B L' B' A A' : ℝ)
    (hB : B' = 0.6 * B)
    (hA : A' = 0.42 * A)
    (hA_def : A = L * B)
    (hA'_def : A' = L' * B') :
    L' = 0.7 * L :=
by
  sorry

end towel_bleach_decrease_l225_225406


namespace probability_heart_and_face_card_club_l225_225061

-- Conditions
def num_cards : ℕ := 52
def num_hearts : ℕ := 13
def num_face_card_clubs : ℕ := 3

-- Define the probabilities
def prob_heart_first : ℚ := num_hearts / num_cards
def prob_face_card_club_given_heart : ℚ := num_face_card_clubs / (num_cards - 1)

-- Proof statement
theorem probability_heart_and_face_card_club :
  prob_heart_first * prob_face_card_club_given_heart = 3 / 204 :=
by
  sorry

end probability_heart_and_face_card_club_l225_225061


namespace profit_percentage_l225_225499

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 500) (h_selling : selling_price = 750) :
  ((selling_price - cost_price) / cost_price) * 100 = 50 :=
by
  sorry

end profit_percentage_l225_225499


namespace min_value_of_f_l225_225109

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 16) + Real.sqrt ((x + 1)^2 + 9))

theorem min_value_of_f :
  ∃ (x : ℝ), f x = 5 * Real.sqrt 2 := sorry

end min_value_of_f_l225_225109


namespace simplify_to_linear_binomial_l225_225754

theorem simplify_to_linear_binomial (k : ℝ) (x : ℝ) : 
  (-3 * k * x^2 + x - 1) + (9 * x^2 - 4 * k * x + 3 * k) = 
  (1 - 4 * k) * x + (3 * k - 1) → 
  k = 3 := by
  sorry

end simplify_to_linear_binomial_l225_225754


namespace lines_intersect_at_l225_225788

def Line1 (t : ℝ) : ℝ × ℝ :=
  let x := 1 + 3 * t
  let y := 2 - t
  (x, y)

def Line2 (u : ℝ) : ℝ × ℝ :=
  let x := -1 + 4 * u
  let y := 4 + 3 * u
  (x, y)

theorem lines_intersect_at :
  ∃ t u : ℝ, Line1 t = Line2 u ∧
             Line1 t = (-53 / 17, 56 / 17) :=
by
  sorry

end lines_intersect_at_l225_225788


namespace seven_b_equals_ten_l225_225216

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : a = b - 2) : 7 * b = 10 := 
sorry

end seven_b_equals_ten_l225_225216


namespace river_and_building_geometry_l225_225545

open Real

theorem river_and_building_geometry (x y : ℝ) :
  (tan 60 * x = y) ∧ (tan 30 * (x + 30) = y) → x = 15 ∧ y = 15 * sqrt 3 :=
by
  sorry

end river_and_building_geometry_l225_225545


namespace range_of_a_l225_225360

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x + a < 0) ∧ (∀ x : ℝ, 2 * x + 7 > 4 * x - 1) ∧ (∀ x : ℝ, x < 0) → a = 0 := 
by sorry

end range_of_a_l225_225360


namespace frequency_in_interval_l225_225223

-- Definitions for the sample size and frequencies in given intervals
def sample_size : ℕ := 20
def freq_10_20 : ℕ := 2
def freq_20_30 : ℕ := 3
def freq_30_40 : ℕ := 4
def freq_40_50 : ℕ := 5

-- The goal: Prove that the frequency of the sample in the interval (10, 50] is 0.7
theorem frequency_in_interval (h₁ : sample_size = 20)
                              (h₂ : freq_10_20 = 2)
                              (h₃ : freq_20_30 = 3)
                              (h₄ : freq_30_40 = 4)
                              (h₅ : freq_40_50 = 5) :
  ((freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) : ℝ) / sample_size = 0.7 := 
by
  sorry

end frequency_in_interval_l225_225223


namespace number_of_new_players_l225_225258

variable (returning_players : ℕ)
variable (groups : ℕ)
variable (players_per_group : ℕ)

theorem number_of_new_players
  (h1 : returning_players = 6)
  (h2 : groups = 9)
  (h3 : players_per_group = 6) :
  (groups * players_per_group - returning_players = 48) := 
sorry

end number_of_new_players_l225_225258


namespace problem1_problem2_l225_225276

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then (1/2)^x - 2 else (x - 2) * (|x| - 1)

theorem problem1 : f (f (-2)) = 0 := by 
  sorry

theorem problem2 (x : ℝ) (h : f x ≥ 2) : x ≥ 3 ∨ x = 0 := by
  sorry

end problem1_problem2_l225_225276


namespace reduced_price_l225_225568

variable (original_price : ℝ) (final_amount : ℝ)

noncomputable def sales_tax (price : ℝ) : ℝ :=
  if price <= 2500 then price * 0.04
  else if price <= 4500 then 2500 * 0.04 + (price - 2500) * 0.07
  else 2500 * 0.04 + 2000 * 0.07 + (price - 4500) * 0.09

noncomputable def discount (price : ℝ) : ℝ :=
  if price <= 2000 then price * 0.02
  else if price <= 4000 then 2000 * 0.02 + (price - 2000) * 0.05
  else 2000 * 0.02 + 2000 * 0.05 + (price - 4000) * 0.10

theorem reduced_price (P : ℝ) (original_price := 5000) (final_amount := 2468) :
  P = original_price - discount original_price + sales_tax original_price → P = 2423 :=
by
  sorry

end reduced_price_l225_225568


namespace hyperbola_equation_Q_on_fixed_circle_l225_225982

-- Define the hyperbola and necessary conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Given conditions
variables (a : ℝ) (h_pos : a > 0)
variables (F1 F2 : ℝ × ℝ)
variables (dist_F2_asymptote : ℝ) (h_dist : dist_F2_asymptote = sqrt 3)
variables (left_vertex : ℝ × ℝ) (right_branch_intersect : ℝ × ℝ)
variables (line_x_half : ℝ × ℝ)
variables (line_PF2 : ℝ × ℝ)
variables (point_Q : ℝ × ℝ)

-- Prove that the equation of the hyperbola is correct
theorem hyperbola_equation :
  hyperbola a x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

-- Prove that point Q lies on a fixed circle
theorem Q_on_fixed_circle :
  dist point_Q F2 = 4 :=
sorry

end hyperbola_equation_Q_on_fixed_circle_l225_225982


namespace first_two_cards_black_prob_l225_225419

noncomputable def probability_first_two_black : ℚ :=
  let total_cards := 52
  let black_cards := 26
  let first_draw_prob := black_cards / total_cards
  let second_draw_prob := (black_cards - 1) / (total_cards - 1)
  first_draw_prob * second_draw_prob

theorem first_two_cards_black_prob :
  probability_first_two_black = 25 / 102 :=
by
  sorry

end first_two_cards_black_prob_l225_225419


namespace intersection_with_x_axis_intersection_with_y_axis_l225_225832

theorem intersection_with_x_axis (x y : ℝ) : y = -2 * x + 4 ∧ y = 0 ↔ x = 2 ∧ y = 0 := by
  sorry

theorem intersection_with_y_axis (x y : ℝ) : y = -2 * x + 4 ∧ x = 0 ↔ x = 0 ∧ y = 4 := by
  sorry

end intersection_with_x_axis_intersection_with_y_axis_l225_225832


namespace simplify_expression_l225_225867

theorem simplify_expression :
  (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := 
by {
  sorry
}

end simplify_expression_l225_225867


namespace total_shaded_area_l225_225330

-- Problem condition definitions
def side_length_carpet := 12
def ratio_large_square : ℕ := 4
def ratio_small_square : ℕ := 4

-- Problem statement
theorem total_shaded_area : 
  ∃ S T : ℚ, 
    12 / S = ratio_large_square ∧ S / T = ratio_small_square ∧ 
    (12 * (T * T)) + (S * S) = 15.75 := 
sorry

end total_shaded_area_l225_225330


namespace arthur_walks_total_distance_l225_225763

theorem arthur_walks_total_distance :
  let east_blocks := 8
  let north_blocks := 10
  let west_blocks := 3
  let block_distance := 1 / 3
  let total_blocks := east_blocks + north_blocks + west_blocks
  let total_miles := total_blocks * block_distance
  total_miles = 7 :=
by
  sorry

end arthur_walks_total_distance_l225_225763


namespace minimum_xy_l225_225080

noncomputable def f (x y : ℝ) := 2 * x + y + 6

theorem minimum_xy (x y : ℝ) (h : 0 < x ∧ 0 < y) (h1 : f x y = x * y) : x * y = 18 :=
by
  sorry

end minimum_xy_l225_225080


namespace degree_measure_supplement_complement_l225_225637

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end degree_measure_supplement_complement_l225_225637


namespace minimum_questions_needed_to_determine_birthday_l225_225108

def min_questions_to_determine_birthday : Nat := 9

theorem minimum_questions_needed_to_determine_birthday : min_questions_to_determine_birthday = 9 :=
sorry

end minimum_questions_needed_to_determine_birthday_l225_225108


namespace exponent_property_l225_225383

theorem exponent_property (a b : ℕ) : (a * b^2)^3 = a^3 * b^6 :=
by sorry

end exponent_property_l225_225383


namespace age_sum_l225_225402

-- Defining the ages of Henry and Jill
def Henry_age : ℕ := 20
def Jill_age : ℕ := 13

-- The statement we need to prove
theorem age_sum : Henry_age + Jill_age = 33 := by
  -- Proof goes here
  sorry

end age_sum_l225_225402


namespace sum_of_all_possible_values_of_x_l225_225050

noncomputable def sum_of_roots_of_equation : ℚ :=
  let eq : Polynomial ℚ := 4 * Polynomial.X ^ 2 + 3 * Polynomial.X - 5
  let roots := eq.roots
  roots.sum

theorem sum_of_all_possible_values_of_x :
  sum_of_roots_of_equation = -3/4 := 
  sorry

end sum_of_all_possible_values_of_x_l225_225050


namespace problem_solved_probability_l225_225114

theorem problem_solved_probability :
  let PA := 1 / 2
  let PB := 1 / 3
  let PC := 1 / 4
  1 - ((1 - PA) * (1 - PB) * (1 - PC)) = 3 / 4 := 
sorry

end problem_solved_probability_l225_225114


namespace max_min_of_f_on_interval_l225_225102

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end max_min_of_f_on_interval_l225_225102


namespace solve_inequality_l225_225565

theorem solve_inequality (x : ℝ) :
  (4 ≤ x^2 - 3 * x - 6 ∧ x^2 - 3 * x - 6 ≤ 2 * x + 8) ↔ (5 ≤ x ∧ x ≤ 7 ∨ x = -2) :=
by
  sorry

end solve_inequality_l225_225565


namespace four_digit_numbers_count_eq_l225_225483

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l225_225483


namespace pressure_relation_l225_225174

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end pressure_relation_l225_225174


namespace johns_average_speed_is_correct_l225_225875

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end johns_average_speed_is_correct_l225_225875


namespace max_value_2ab_2bc_2cd_2da_l225_225429

theorem max_value_2ab_2bc_2cd_2da {a b c d : ℕ} :
  (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
  (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
  (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
  (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
  (b ≠ c) ∧ (b ≠ d) ∧
  (c ≠ d)
  → 2 * (a * b + b * c + c * d + d * a) ≤ 144 :=
by
  sorry

end max_value_2ab_2bc_2cd_2da_l225_225429


namespace supermarket_selection_expected_value_l225_225225

noncomputable def small_supermarkets := 72
noncomputable def medium_supermarkets := 24
noncomputable def large_supermarkets := 12
noncomputable def total_supermarkets := small_supermarkets + medium_supermarkets + large_supermarkets
noncomputable def selected_supermarkets := 9

-- Problem (I)
noncomputable def small_selected := (small_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def medium_selected := (medium_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def large_selected := (large_supermarkets * selected_supermarkets) / total_supermarkets

theorem supermarket_selection :
  small_selected = 6 ∧ medium_selected = 2 ∧ large_selected = 1 :=
sorry

-- Problem (II)
noncomputable def further_analysis := 3
noncomputable def prob_small := small_selected / selected_supermarkets
noncomputable def E_X := prob_small * further_analysis

theorem expected_value :
  E_X = 2 :=
sorry

end supermarket_selection_expected_value_l225_225225


namespace sin_double_angle_l225_225519

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
  sorry

end sin_double_angle_l225_225519


namespace range_of_k_l225_225350

theorem range_of_k {k : ℝ} :
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end range_of_k_l225_225350


namespace determine_k_value_l225_225854

theorem determine_k_value (x y z k : ℝ) 
  (h1 : 5 / (x + y) = k / (x - z))
  (h2 : k / (x - z) = 9 / (z + y)) :
  k = 14 :=
sorry

end determine_k_value_l225_225854


namespace technician_completion_percentage_l225_225908

noncomputable def percentage_completed (D : ℝ) : ℝ :=
  let total_distance := 2.20 * D
  let completed_distance := 1.12 * D
  (completed_distance / total_distance) * 100

theorem technician_completion_percentage (D : ℝ) (hD : D > 0) :
  percentage_completed D = 50.91 :=
by
  sorry

end technician_completion_percentage_l225_225908


namespace solve_problem_l225_225261

-- Declare the variables n and m
variables (n m : ℤ)

-- State the theorem with given conditions and prove that 2n + m = 36
theorem solve_problem
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 :=
sorry

end solve_problem_l225_225261


namespace both_students_given_correct_l225_225010

open ProbabilityTheory

variables (P_A P_B : ℝ)

-- Define the conditions from part a)
def student_a_correct := P_A = 3 / 5
def student_b_correct := P_B = 1 / 3

-- Define the event that both students correctly answer
def both_students_correct := P_A * P_B

-- Define the event that the question is answered correctly
def question_answered_correctly := (P_A * (1 - P_B)) + ((1 - P_A) * P_B) + (P_A * P_B)

-- Define the conditional probability we need to prove
theorem both_students_given_correct (hA : student_a_correct P_A) (hB : student_b_correct P_B) :
  both_students_correct P_A P_B / question_answered_correctly P_A P_B = 3 / 11 := 
sorry

end both_students_given_correct_l225_225010


namespace parabola_chords_reciprocal_sum_l225_225075

theorem parabola_chords_reciprocal_sum (x y : ℝ) (AB CD : ℝ) (p : ℝ) :
  (y = (4 : ℝ) * x) ∧ (AB ≠ 0) ∧ (CD ≠ 0) ∧
  (p = (2 : ℝ)) ∧
  (|AB| = (2 * p / (Real.sin (Real.pi / 4))^2)) ∧ 
  (|CD| = (2 * p / (Real.cos (Real.pi / 4))^2)) →
  (1 / |AB| + 1 / |CD| = 1 / 4) :=
by
  sorry

end parabola_chords_reciprocal_sum_l225_225075


namespace set_intersection_l225_225787

def setM : Set ℝ := {x | x^2 - 1 < 0}
def setN : Set ℝ := {y | ∃ x ∈ setM, y = Real.log (x + 2)}

theorem set_intersection : setM ∩ setN = {y | 0 < y ∧ y < Real.log 3} :=
by
  sorry

end set_intersection_l225_225787


namespace driver_travel_distance_per_week_l225_225415

noncomputable def daily_distance := 30 * 3 + 25 * 4 + 40 * 2

noncomputable def total_weekly_distance := daily_distance * 6 + 35 * 5

theorem driver_travel_distance_per_week : total_weekly_distance = 1795 := by
  simp [daily_distance, total_weekly_distance]
  done

end driver_travel_distance_per_week_l225_225415


namespace merchant_profit_percentage_l225_225651

theorem merchant_profit_percentage 
    (cost_price : ℝ) 
    (markup_percentage : ℝ) 
    (discount_percentage : ℝ) 
    (h1 : cost_price = 100) 
    (h2 : markup_percentage = 0.20) 
    (h3 : discount_percentage = 0.05) 
    : ((cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price) / cost_price * 100) = 14 := 
by 
    sorry

end merchant_profit_percentage_l225_225651


namespace oven_clock_actual_time_l225_225392

theorem oven_clock_actual_time :
  ∀ (h : ℕ), (oven_time : h = 10) →
  (oven_gains : ℕ) = 8 →
  (initial_time : ℕ) = 18 →          
  (initial_wall_time : ℕ) = 18 →
  (wall_time_after_one_hour : ℕ) = 19 →
  (oven_time_after_one_hour : ℕ) = 19 + 8/60 →
  ℕ := sorry

end oven_clock_actual_time_l225_225392


namespace picture_books_count_l225_225319

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l225_225319


namespace find_theta_l225_225440

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (x θ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + θ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_theta (θ : ℝ) : 
  (∀ x, g x θ = g (-x) θ) → θ = Real.pi / 4 :=
by
  intros h
  sorry

end find_theta_l225_225440


namespace complementary_three_card_sets_l225_225297

-- Definitions for the problem conditions
inductive Shape | circle | square | triangle | star
inductive Color | red | blue | green | yellow
inductive Shade | light | medium | dark | very_dark

-- Definition of a Card as a combination of shape, color, shade
structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

-- Definition of a set being complementary
def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape)) ∧
  ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color)) ∧
  ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))

-- Definition of the problem statement
def complementary_three_card_sets_count : Nat :=
  360

-- The theorem to be proved
theorem complementary_three_card_sets : ∃ (n : Nat), n = complementary_three_card_sets_count :=
  by
    use 360
    sorry

end complementary_three_card_sets_l225_225297


namespace isosceles_triangle_and_sin_cos_range_l225_225157

theorem isosceles_triangle_and_sin_cos_range 
  (A B C : ℝ) (a b c : ℝ) 
  (hA_pos : 0 < A) (hA_lt_pi_div_2 : A < π / 2) (h_triangle : a * Real.cos B = b * Real.cos A) :
  (A = B ∧
  ∃ x, x = Real.sin B + Real.cos (A + π / 6) ∧ (1 / 2 < x ∧ x ≤ 1)) :=
by
  sorry

end isosceles_triangle_and_sin_cos_range_l225_225157


namespace ratio_of_donations_l225_225353

theorem ratio_of_donations (x : ℝ) (h1 : ∀ (y : ℝ), y = 40) (h2 : ∀ (y : ℝ), y = 40 * x)
  (h3 : ∀ (y : ℝ), y = 0.30 * (40 + 40 * x)) (h4 : ∀ (y : ℝ), y = 36) : x = 2 := 
by 
  sorry

end ratio_of_donations_l225_225353


namespace find_m_l225_225644

-- Define the lines l1 and l2
def line1 (x y : ℝ) (m : ℝ) : Prop := x + m^2 * y + 6 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- The statement that two lines are parallel
def lines_parallel (m : ℝ) : Prop :=
  ∀ (x y : ℝ), line1 x y m → line2 x y m

-- The mathematically equivalent proof problem
theorem find_m (m : ℝ) (H_parallel : lines_parallel m) : m = 0 ∨ m = -1 :=
sorry

end find_m_l225_225644


namespace find_k_l225_225084

theorem find_k (k : ℝ) (α β : ℝ) 
  (h1 : α + β = -k) 
  (h2 : α * β = 12) 
  (h3 : α + 7 + β + 7 = k) : 
  k = -7 :=
sorry

end find_k_l225_225084


namespace route_y_saves_time_l225_225786

theorem route_y_saves_time (distance_X speed_X : ℕ)
                           (distance_Y_WOCZ distance_Y_CZ speed_Y speed_Y_CZ : ℕ)
                           (time_saved_in_minutes : ℚ) :
  distance_X = 8 → 
  speed_X = 40 → 
  distance_Y_WOCZ = 6 → 
  distance_Y_CZ = 1 → 
  speed_Y = 50 → 
  speed_Y_CZ = 25 → 
  time_saved_in_minutes = 2.4 →
  (distance_X / speed_X : ℚ) * 60 - 
  ((distance_Y_WOCZ / speed_Y + distance_Y_CZ / speed_Y_CZ) * 60) = time_saved_in_minutes :=
by
  intros
  sorry

end route_y_saves_time_l225_225786


namespace magic_square_y_l225_225751

theorem magic_square_y (a b c d e y : ℚ) (h1 : y - 61 = a) (h2 : 2 * y - 125 = b) 
    (h3 : y + 25 + 64 = 3 + (y - 61) + (2 * y - 125)) : y = 272 / 3 :=
by
  sorry

end magic_square_y_l225_225751


namespace inequalities_hold_l225_225592

variable {a b c x y z : ℝ}

theorem inequalities_hold 
  (h1 : x ≤ a)
  (h2 : y ≤ b)
  (h3 : z ≤ c) :
  x * y + y * z + z * x ≤ a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  x * y * z ≤ a * b * c :=
sorry

end inequalities_hold_l225_225592


namespace avg_children_in_families_with_children_l225_225403

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end avg_children_in_families_with_children_l225_225403


namespace problem1_problem2_l225_225980

-- Problem 1
theorem problem1 (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 0) :
  (x^2 + x) / (x^2 - 2 * x + 1) / (2 / (x - 1) - 1 / x) = x^2 / (x - 1) := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (hx1 : x > 0) :
  (2 * x + 1) / 3 - (5 * x - 1) / 2 < 1 ∧ 
  (5 * x - 1 < 3 * (x + 2)) →
  x = 1 ∨ x = 2 ∨ x = 3 := by
  sorry

end problem1_problem2_l225_225980


namespace goose_eggs_l225_225437

theorem goose_eggs (E : ℕ) 
  (H1 : (2/3 : ℚ) * E = h) 
  (H2 : (3/4 : ℚ) * h = m)
  (H3 : (2/5 : ℚ) * m = 180) : 
  E = 2700 := 
sorry

end goose_eggs_l225_225437


namespace ratio_arithmetic_sequences_l225_225082

variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h : ∀ n : ℕ, S n / T n = (3 * n - 1) / (2 * n + 3))

theorem ratio_arithmetic_sequences :
  a 7 / b 7 = 38 / 29 :=
sorry

end ratio_arithmetic_sequences_l225_225082


namespace solve_for_a_l225_225860

theorem solve_for_a (a : ℝ) (h : (a + 3)^(a + 1) = 1) : a = -2 ∨ a = -1 :=
by {
  -- proof here
  sorry
}

end solve_for_a_l225_225860


namespace score_entered_twice_l225_225139

theorem score_entered_twice (scores : List ℕ) (h : scores = [68, 74, 77, 82, 85, 90]) :
  ∃ (s : ℕ), s = 82 ∧ ∀ (entered : List ℕ), entered.length = 7 ∧ (∀ i, (List.take (i + 1) entered).sum % (i + 1) = 0) →
  (List.count (List.insertNth i 82 scores)) = 2 ∧ (∀ x, x ∈ scores.remove 82 → x ≠ s) :=
by
  sorry

end score_entered_twice_l225_225139


namespace rain_probability_l225_225039

theorem rain_probability :
  let PM : ℝ := 0.62
  let PT : ℝ := 0.54
  let PMcTc : ℝ := 0.28
  let PMT : ℝ := PM + PT - (1 - PMcTc)
  PMT = 0.44 :=
by
  sorry

end rain_probability_l225_225039


namespace divide_estate_l225_225001

theorem divide_estate (total_estate : ℕ) (son_share : ℕ) (daughter_share : ℕ) (wife_share : ℕ) :
  total_estate = 210 →
  son_share = (4 / 7) * total_estate →
  daughter_share = (1 / 7) * total_estate →
  wife_share = (2 / 7) * total_estate →
  son_share + daughter_share + wife_share = total_estate :=
by
  intros
  sorry

end divide_estate_l225_225001


namespace find_a_l225_225930

def E (a b c : ℤ) : ℤ := a * b * b + c

theorem find_a (a : ℤ) : E a 3 1 = E a 5 11 → a = -5 / 8 := 
by sorry

end find_a_l225_225930


namespace odd_function_value_l225_225688

theorem odd_function_value (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fx : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) :
  f 1 = -3 := 
sorry

end odd_function_value_l225_225688


namespace Watson_class_student_count_l225_225473

def num_kindergartners : ℕ := 14
def num_first_graders : ℕ := 24
def num_second_graders : ℕ := 4

def total_students : ℕ := num_kindergartners + num_first_graders + num_second_graders

theorem Watson_class_student_count : total_students = 42 := 
by
    sorry

end Watson_class_student_count_l225_225473


namespace find_reciprocal_square_sum_of_roots_l225_225221

theorem find_reciprocal_square_sum_of_roots :
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (a^3 - 6 * a^2 - a + 3 = 0) ∧ 
    (b^3 - 6 * b^2 - b + 3 = 0) ∧ 
    (c^3 - 6 * c^2 - c + 3 = 0) ∧ 
    (a + b + c = 6) ∧
    (a * b + b * c + c * a = -1) ∧
    (a * b * c = -3)) 
    → (1 / a^2 + 1 / b^2 + 1 / c^2 = 37 / 9) :=
sorry

end find_reciprocal_square_sum_of_roots_l225_225221


namespace problem1_l225_225588

theorem problem1 :
  (15 * (-3 / 4) + (-15) * (3 / 2) + 15 / 4) = -30 :=
by
  sorry

end problem1_l225_225588


namespace remaining_cooking_time_l225_225896

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l225_225896


namespace substitutions_made_in_first_half_l225_225831

-- Definitions based on given problem conditions
def total_players : ℕ := 24
def starters : ℕ := 11
def non_players : ℕ := 7
def first_half_substitutions (S : ℕ) : ℕ := S
def second_half_substitutions (S : ℕ) : ℕ := 2 * S
def total_players_played (S : ℕ) := starters + first_half_substitutions S + second_half_substitutions S
def remaining_players : ℕ := total_players - non_players

-- Proof problem statement
theorem substitutions_made_in_first_half (S : ℕ) (h : total_players_played S = remaining_players) : S = 2 :=
by
  sorry

end substitutions_made_in_first_half_l225_225831


namespace number_of_total_flowers_l225_225614

theorem number_of_total_flowers :
  let n_pots := 141
  let flowers_per_pot := 71
  n_pots * flowers_per_pot = 10011 :=
by
  sorry

end number_of_total_flowers_l225_225614


namespace add_fractions_l225_225195

theorem add_fractions: (2 / 5) + (3 / 8) = 31 / 40 := 
by 
  sorry

end add_fractions_l225_225195


namespace probability_one_side_is_side_of_decagon_l225_225913

theorem probability_one_side_is_side_of_decagon :
  let decagon_vertices := 10
  let total_triangles := Nat.choose decagon_vertices 3
  let favorable_one_side :=
    decagon_vertices * (decagon_vertices - 3) / 2
  let favorable_two_sides := decagon_vertices
  let favorable_outcomes := favorable_one_side + favorable_two_sides
  let probability := favorable_outcomes / total_triangles
  total_triangles = 120 ∧ favorable_outcomes = 60 ∧ probability = 1 / 2 := 
by
  sorry

end probability_one_side_is_side_of_decagon_l225_225913


namespace chickens_and_rabbits_l225_225038

theorem chickens_and_rabbits (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chickens_and_rabbits_l225_225038


namespace evaluate_expression_at_two_l225_225750

theorem evaluate_expression_at_two: 
  (3 * 2^2 - 4 * 2 + 2) = 6 := 
by 
  sorry

end evaluate_expression_at_two_l225_225750


namespace vector_CB_correct_l225_225929

-- Define the vectors AB and AC
def AB : ℝ × ℝ := (2, 3)
def AC : ℝ × ℝ := (-1, 2)

-- Define the vector CB as the difference of AB and AC
def CB (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Prove that CB = (3, 1) given AB and AC
theorem vector_CB_correct : CB AB AC = (3, 1) :=
by
  sorry

end vector_CB_correct_l225_225929


namespace vectors_projection_l225_225017

noncomputable def p := (⟨-44 / 53, 154 / 53⟩ : ℝ × ℝ)

theorem vectors_projection :
  let u := (⟨-4, 2⟩ : ℝ × ℝ)
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let w := (⟨7, 2⟩ : ℝ × ℝ)
  (⟨(7 * (24 / 53)) - 4, (2 * (24 / 53)) + 2⟩ : ℝ × ℝ) = p :=
by {
  -- proof skipped
  sorry
}

end vectors_projection_l225_225017


namespace age_sum_squares_l225_225193

theorem age_sum_squares (a b c : ℕ) (h1 : 5 * a + 2 * b = 3 * c) (h2 : 3 * c^2 = 4 * a^2 + b^2) (h3 : Nat.gcd (Nat.gcd a b) c = 1) : a^2 + b^2 + c^2 = 18 :=
sorry

end age_sum_squares_l225_225193


namespace arcsin_cos_arcsin_rel_arccos_sin_arccos_l225_225925

theorem arcsin_cos_arcsin_rel_arccos_sin_arccos (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
    let α := Real.arcsin (Real.cos (Real.arcsin x))
    let β := Real.arccos (Real.sin (Real.arccos x))
    (Real.arcsin x + Real.arccos x = π / 2) → α + β = π / 2 :=
by
  let α := Real.arcsin (Real.cos (Real.arcsin x))
  let β := Real.arccos (Real.sin (Real.arccos x))
  intro h_arcsin_arccos_eq
  sorry

end arcsin_cos_arcsin_rel_arccos_sin_arccos_l225_225925


namespace markov_coprime_squares_l225_225736

def is_coprime (x y : ℕ) : Prop :=
Nat.gcd x y = 1

theorem markov_coprime_squares (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  x^2 + y^2 + z^2 = 3 * x * y * z →
  ∃ a b c: ℕ, (a, b, c) = (2, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∧ 
  (a ≠ 1 → ∃ p q : ℕ, is_coprime p q ∧ a = p^2 + q^2) :=
sorry

end markov_coprime_squares_l225_225736


namespace even_func_decreasing_on_neg_interval_l225_225944

variable {f : ℝ → ℝ}

theorem even_func_decreasing_on_neg_interval
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ (a b : ℝ), 3 ≤ a → a < b → b ≤ 7 → f a < f b)
  (h_min_val : ∀ x, 3 ≤ x → x ≤ 7 → f x ≥ 2) :
  (∀ (a b : ℝ), -7 ≤ a → a < b → b ≤ -3 → f a > f b) ∧ (∀ x, -7 ≤ x → x ≤ -3 → f x ≤ 2) :=
by
  sorry

end even_func_decreasing_on_neg_interval_l225_225944


namespace max_lateral_surface_area_of_pyramid_l225_225590

theorem max_lateral_surface_area_of_pyramid (a h : ℝ) (r : ℝ) (h_eq : 2 * a^2 + h^2 = 4) (r_eq : r = 1) :
  ∃ (a : ℝ), (a = 1) :=
by
sorry

end max_lateral_surface_area_of_pyramid_l225_225590


namespace julie_can_print_100_newspapers_l225_225472

def num_boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

theorem julie_can_print_100_newspapers :
  (num_boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end julie_can_print_100_newspapers_l225_225472


namespace max_bees_in_largest_beehive_l225_225567

def total_bees : ℕ := 2000000
def beehives : ℕ := 7
def min_ratio : ℚ := 0.7

theorem max_bees_in_largest_beehive (B_max : ℚ) : 
  (6 * (min_ratio * B_max) + B_max = total_bees) → 
  B_max <= 2000000 / 5.2 ∧ B_max.floor = 384615 :=
by
  sorry

end max_bees_in_largest_beehive_l225_225567


namespace train_speed_including_stoppages_l225_225729

theorem train_speed_including_stoppages (s : ℝ) (t : ℝ) (running_time_fraction : ℝ) :
  s = 48 ∧ t = 1/4 ∧ running_time_fraction = (1 - t) → (s * running_time_fraction = 36) :=
by
  sorry

end train_speed_including_stoppages_l225_225729


namespace arithmetic_mean_of_fractions_l225_225476

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 8 + (5 : ℚ) / 12 / 2 = 19 / 48 := by
  sorry

end arithmetic_mean_of_fractions_l225_225476


namespace abs_diff_61st_terms_l225_225823

noncomputable def seq_C (n : ℕ) : ℤ := 20 + 15 * (n - 1)
noncomputable def seq_D (n : ℕ) : ℤ := 20 - 15 * (n - 1)

theorem abs_diff_61st_terms :
  |seq_C 61 - seq_D 61| = 1800 := sorry

end abs_diff_61st_terms_l225_225823


namespace age_of_john_l225_225057

theorem age_of_john (J S : ℕ) 
  (h1 : S = 2 * J)
  (h2 : S + (50 - J) = 60) :
  J = 10 :=
sorry

end age_of_john_l225_225057


namespace sqrt_144000_simplified_l225_225907

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplified_l225_225907


namespace unoccupied_cylinder_volume_l225_225638

theorem unoccupied_cylinder_volume (r h : ℝ) (V_cylinder V_cone : ℝ) :
  r = 15 ∧ h = 30 ∧ V_cylinder = π * r^2 * h ∧ V_cone = (1/3) * π * r^2 * (r / 2) →
  V_cylinder - 2 * V_cone = 4500 * π :=
by
  intros h1
  sorry

end unoccupied_cylinder_volume_l225_225638


namespace original_number_l225_225310

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l225_225310


namespace inlet_pipe_rate_l225_225768

theorem inlet_pipe_rate (capacity : ℕ) (t_empty : ℕ) (t_with_inlet : ℕ) (R_out : ℕ) :
  capacity = 6400 →
  t_empty = 10 →
  t_with_inlet = 16 →
  R_out = capacity / t_empty →
  (R_out - (capacity / t_with_inlet)) / 60 = 4 :=
by
  intros h1 h2 h3 h4 
  sorry

end inlet_pipe_rate_l225_225768


namespace problem_I_problem_II_l225_225268

-- Problem (I)
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) : 
  (f (x + 8) ≥ 10 - f x) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

-- Problem (II)
theorem problem_II (x y : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x + 1|) 
(h_abs_x : |x| > 1) (h_abs_y : |y| < 1) :
  f y < |x| * f (y / x^2) :=
sorry

end problem_I_problem_II_l225_225268


namespace B_days_to_complete_work_l225_225771

theorem B_days_to_complete_work (B : ℕ) (hB : B ≠ 0)
  (A_work_days : ℕ := 9) (combined_days : ℕ := 6)
  (work_rate_A : ℚ := 1 / A_work_days) (work_rate_combined : ℚ := 1 / combined_days):
  (1 / B : ℚ) = work_rate_combined - work_rate_A → B = 18 :=
by
  intro h
  sorry

end B_days_to_complete_work_l225_225771


namespace adam_change_l225_225185

theorem adam_change : 
  let amount : ℝ := 5.00
  let cost : ℝ := 4.28
  amount - cost = 0.72 :=
by
  -- proof goes here
  sorry

end adam_change_l225_225185


namespace line_intersects_parabola_once_l225_225212

theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, -3 * y^2 + 2 * y + 7 = k) ↔ k = 22 / 3 :=
by {
  sorry
}

end line_intersects_parabola_once_l225_225212


namespace max_sides_of_convex_polygon_l225_225137

theorem max_sides_of_convex_polygon (n : ℕ) 
  (h_convex : n ≥ 3) 
  (h_angles: ∀ (a : Fin 4), (100 : ℝ) ≤ a.val) 
  : n ≤ 8 :=
sorry

end max_sides_of_convex_polygon_l225_225137


namespace question_l225_225008

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l225_225008


namespace a_2016_value_l225_225556

def S (n : ℕ) : ℕ := n^2 - 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_2016_value : a 2016 = 4031 := by
  sorry

end a_2016_value_l225_225556


namespace find_PA_PB_sum_2sqrt6_l225_225409

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.cos θ - 6 * Real.sin θ + 1 / ρ = 0

noncomputable def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + 1 / 2 * t ∧ y = 3 + Real.sqrt 3 / 2 * t

def point_P (x y : ℝ) : Prop :=
  x = 3 ∧ y = 3

theorem find_PA_PB_sum_2sqrt6 :
  (∃ ρ θ t₁ t₂, polar_equation ρ θ ∧ parametric_line t₁ 3 3 ∧ parametric_line t₂ 3 3 ∧
  point_P 3 3 ∧ |t₁| + |t₂| = 2 * Real.sqrt 6) := sorry

end find_PA_PB_sum_2sqrt6_l225_225409


namespace find_a_l225_225042

-- Define the conditions for the lines l1 and l2
def line1 (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def line2 (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - (3/2) = 0

-- Define the condition for parallel lines
def parallel (a : ℝ) : Prop := a^2 - 4 * ((3/4) * a + 1) = 0 ∧ 4 * (-3/2) - 6 * a ≠ 0

-- Define the condition for perpendicular lines
def perpendicular (a : ℝ) : Prop := a * ((3/4) * a + 1) + 4 * a = 0

-- The theorem to prove values of a for which l1 is parallel or perpendicular to l2
theorem find_a (a : ℝ) :
  (parallel a → a = 4) ∧ (perpendicular a → a = 0 ∨ a = -20/3) :=
by
  sorry

end find_a_l225_225042


namespace num_subsets_of_abc_eq_eight_l225_225539

theorem num_subsets_of_abc_eq_eight : 
  (∃ (s : Finset ℕ), s = {1, 2, 3} ∧ s.powerset.card = 8) :=
sorry

end num_subsets_of_abc_eq_eight_l225_225539


namespace tetrahedron_sphere_surface_area_l225_225722

-- Define the conditions
variables (a : ℝ) (mid_AB_C : ℝ → Prop) (S : ℝ)
variables (h1 : a > 0)
variables (h2 : mid_AB_C a)
variables (h3 : S = 3 * Real.sqrt 2)

-- Theorem statement
theorem tetrahedron_sphere_surface_area (h1 : a = 2 * Real.sqrt 3) : 
  4 * Real.pi * ( (Real.sqrt 6 / 4) * a )^2 = 18 * Real.pi := by
  sorry

end tetrahedron_sphere_surface_area_l225_225722


namespace expensive_feed_cost_l225_225995

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed worth 0.36 dollars per pound by mixing one kind worth 0.18 dollars per pound with another kind. They used 17 pounds of the cheaper kind in the mix. What is the cost per pound of the more expensive kind of feed? --/
theorem expensive_feed_cost 
  (total_feed : ℝ := 35) 
  (avg_cost : ℝ := 0.36) 
  (cheaper_feed : ℝ := 17) 
  (cheaper_cost : ℝ := 0.18) 
  (total_cost : ℝ := total_feed * avg_cost) 
  (cheaper_total_cost : ℝ := cheaper_feed * cheaper_cost) 
  (expensive_feed : ℝ := total_feed - cheaper_feed) : 
  (total_cost - cheaper_total_cost) / expensive_feed = 0.53 :=
by
  sorry

end expensive_feed_cost_l225_225995


namespace youngest_son_trips_l225_225876

theorem youngest_son_trips 
  (p : ℝ) (n_oldest : ℝ) (c : ℝ) (Y : ℝ)
  (h1 : p = 100)
  (h2 : n_oldest = 35)
  (h3 : c = 4)
  (h4 : p / c = Y) :
  Y = 25 := sorry

end youngest_son_trips_l225_225876


namespace probability_of_events_l225_225625

-- Define the sets of tiles in each box
def boxA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 25}
def boxB : Set ℕ := {n | 15 ≤ n ∧ n ≤ 40}

-- Define the specific conditions
def eventA (tile : ℕ) : Prop := tile ≤ 20
def eventB (tile : ℕ) : Prop := (Odd tile ∨ tile > 35)

-- Define the probabilities as calculations
def prob_eventA : ℚ := 20 / 25
def prob_eventB : ℚ := 15 / 26

-- The final probability given independence
def combined_prob : ℚ := prob_eventA * prob_eventB

-- The theorem statement we want to prove
theorem probability_of_events :
  combined_prob = 6 / 13 := 
by 
  -- proof details would go here
  sorry

end probability_of_events_l225_225625


namespace total_budget_l225_225070

-- Define the conditions for the problem
def fiscal_months : ℕ := 12
def total_spent_at_six_months : ℕ := 6580
def over_budget_at_six_months : ℕ := 280

-- Calculate the total budget for the project
theorem total_budget (budget : ℕ) 
  (h : 6 * (total_spent_at_six_months - over_budget_at_six_months) * 2 = budget) 
  : budget = 12600 := 
  by
    -- Proof will be here
    sorry

end total_budget_l225_225070


namespace tan_half_angles_l225_225730

theorem tan_half_angles (a b : ℝ) (ha : 3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0) :
  ∃ z : ℝ, z = Real.tan (a / 2) * Real.tan (b / 2) ∧ (z = Real.sqrt (6 / 13) ∨ z = -Real.sqrt (6 / 13)) :=
by
  sorry

end tan_half_angles_l225_225730


namespace cracker_calories_l225_225327

theorem cracker_calories (cc : ℕ) (hc1 : ∀ (n : ℕ), n = 50 → cc = 50) (hc2 : ∀ (n : ℕ), n = 7 → 7 * 50 = 350) (hc3 : ∀ (n : ℕ), n = 10 * cc → 10 * cc = 10 * cc) (hc4 : 350 + 10 * cc = 500) : cc = 15 :=
by
  sorry

end cracker_calories_l225_225327


namespace updated_mean_166_l225_225769

/-- The mean of 50 observations is 200. Later, it was found that there is a decrement of 34 
from each observation. Prove that the updated mean of the observations is 166. -/
theorem updated_mean_166
  (mean : ℝ) (n : ℕ) (decrement : ℝ) (updated_mean : ℝ)
  (h1 : mean = 200) (h2 : n = 50) (h3 : decrement = 34) (h4 : updated_mean = 166) :
  mean - (decrement * n) / n = updated_mean :=
by
  sorry

end updated_mean_166_l225_225769


namespace water_tank_capacity_l225_225145

theorem water_tank_capacity (x : ℝ)
  (h1 : (2 / 3) * x - (1 / 3) * x = 20) : x = 60 := 
  sorry

end water_tank_capacity_l225_225145


namespace biased_coin_probability_l225_225631

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability mass function for a binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- Define the problem conditions
def problem_conditions : Prop :=
  let p := 1 / 3
  binomial_pmf 5 1 p = binomial_pmf 5 2 p ∧ p ≠ 0 ∧ (1 - p) ≠ 0

-- The target probability to prove
def target_probability := 40 / 243

-- The theorem statement
theorem biased_coin_probability : problem_conditions → binomial_pmf 5 3 (1 / 3) = target_probability :=
by
  intro h
  sorry

end biased_coin_probability_l225_225631


namespace cats_remaining_l225_225529

theorem cats_remaining 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 := 
by
  sorry

end cats_remaining_l225_225529


namespace part1_part2_l225_225204

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem part1 (x : ℝ) : |f (-x)| + |f x| ≥ 4 * |x| := 
by
  sorry

theorem part2 (x a : ℝ) (h : |x - a| < 1 / 2) : |f x - f a| < |a| + 5 / 4 := 
by
  sorry

end part1_part2_l225_225204


namespace find_k_l225_225413

theorem find_k 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-3, 0))
  (hB : B = (0, -3))
  (hX : X = (0, 9))
  (Yx : Y.1 = 15)
  (hXY_parallel : (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1)) :
  Y.2 = -6 := by
  -- proofs are omitted as per the requirements
  sorry

end find_k_l225_225413


namespace problem_statement_l225_225820

theorem problem_statement :
  (-2010)^2011 = - (2010 ^ 2011) :=
by
  -- proof to be filled in
  sorry

end problem_statement_l225_225820


namespace largest_common_term_l225_225535

theorem largest_common_term (a : ℕ) (k l : ℕ) (hk : a = 4 + 5 * k) (hl : a = 5 + 10 * l) (h : a < 300) : a = 299 :=
by {
  sorry
}

end largest_common_term_l225_225535


namespace reflection_identity_l225_225749

-- Define the reflection function
def reflect (O P : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Given three points and a point P
variables (O1 O2 O3 P : ℝ × ℝ)

-- Define the sequence of reflections
def sequence_reflection (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect O3 (reflect O2 (reflect O1 P))

-- Lean 4 statement to prove the mathematical theorem
theorem reflection_identity :
  sequence_reflection O1 O2 O3 (sequence_reflection O1 O2 O3 P) = P :=
by sorry

end reflection_identity_l225_225749


namespace common_ratio_of_series_l225_225347

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l225_225347


namespace four_consecutive_numbers_l225_225664

theorem four_consecutive_numbers (numbers : List ℝ) (h_distinct : numbers.Nodup) (h_length : numbers.length = 100) :
  ∃ (a b c d : ℝ) (h_seq : ([a, b, c, d] ∈ numbers.cyclicPermutations)), b + c < a + d :=
by
  sorry

end four_consecutive_numbers_l225_225664


namespace minimum_value_F_l225_225827

noncomputable def minimum_value_condition (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 10 * (x + y)

noncomputable def F (x y : ℝ) : ℝ :=
  6 * y + 8 * x - 9

theorem minimum_value_F :
  (∃ x y : ℝ, minimum_value_condition x y) → ∃ x y : ℝ, minimum_value_condition x y ∧ F x y = 11 :=
sorry

end minimum_value_F_l225_225827


namespace total_height_of_sandcastles_l225_225346

structure Sandcastle :=
  (feet : Nat)
  (fraction_num : Nat)
  (fraction_den : Nat)

def janet : Sandcastle := ⟨3, 5, 6⟩
def sister : Sandcastle := ⟨2, 7, 12⟩
def tom : Sandcastle := ⟨1, 11, 20⟩
def lucy : Sandcastle := ⟨2, 13, 24⟩

-- a function to convert a Sandcastle to a common denominator
def convert_to_common_denominator (s : Sandcastle) : Sandcastle :=
  let common_den := 120 -- LCM of 6, 12, 20, 24
  ⟨s.feet, (s.fraction_num * (common_den / s.fraction_den)), common_den⟩

-- Definition of heights after conversion to common denominator
def janet_converted : Sandcastle := convert_to_common_denominator janet
def sister_converted : Sandcastle := convert_to_common_denominator sister
def tom_converted : Sandcastle := convert_to_common_denominator tom
def lucy_converted : Sandcastle := convert_to_common_denominator lucy

-- Proof problem
def total_height_proof_statement : Sandcastle :=
  let total_feet := janet.feet + sister.feet + tom.feet + lucy.feet
  let total_numerator := janet_converted.fraction_num + sister_converted.fraction_num + tom_converted.fraction_num + lucy_converted.fraction_num
  let total_denominator := 120
  ⟨total_feet + (total_numerator / total_denominator), total_numerator % total_denominator, total_denominator⟩

theorem total_height_of_sandcastles :
  total_height_proof_statement = ⟨10, 61, 120⟩ :=
by
  sorry

end total_height_of_sandcastles_l225_225346


namespace distance_on_dirt_road_l225_225443

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end distance_on_dirt_road_l225_225443


namespace power_division_l225_225632

theorem power_division : 3^18 / (27^3) = 19683 := by
  have h1 : 27 = 3^3 := by sorry
  have h2 : (3^3)^3 = 3^(3*3) := by sorry
  have h3 : 27^3 = 3^9 := by
    rw [h1]
    exact h2
  rw [h3]
  have h4 : 3^18 / 3^9 = 3^(18 - 9) := by sorry
  rw [h4]
  norm_num

end power_division_l225_225632


namespace probability_of_graduate_degree_l225_225610

-- Define the conditions as Lean statements
variable (k m : ℕ)
variable (G := 1 * k) 
variable (C := 2 * m) 
variable (N1 := 8 * k) -- from the ratio G:N = 1:8
variable (N2 := 3 * m) -- from the ratio C:N = 2:3

-- Least common multiple (LCM) of 8 and 3 is 24
-- Therefore, determine specific values for G, C, and N
-- Given these updates from solution steps we set:
def G_scaled : ℕ := 3
def C_scaled : ℕ := 16
def N_scaled : ℕ := 24

-- Total number of college graduates
def total_college_graduates : ℕ := G_scaled + C_scaled

-- Probability q of picking a college graduate with a graduate degree
def q : ℚ := G_scaled / total_college_graduates

-- Lean proof statement for equivalence
theorem probability_of_graduate_degree : 
  q = 3 / 19 := by
sorry

end probability_of_graduate_degree_l225_225610


namespace similar_triangle_shortest_side_l225_225884

theorem similar_triangle_shortest_side {a b c : ℝ} (h₁ : a = 24) (h₂ : b = 32) (h₃ : c = 80) :
  let hypotenuse₁ := Real.sqrt (a ^ 2 + b ^ 2)
  let scale_factor := c / hypotenuse₁
  let shortest_side₂ := scale_factor * a
  shortest_side₂ = 48 :=
by
  sorry

end similar_triangle_shortest_side_l225_225884


namespace remainder_div_84_l225_225954

def a := (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)

theorem remainder_div_84 (a : ℕ) (h : a = (5 : ℕ) * 10 ^ 2015 + (5 : ℕ)) : a % 84 = 63 := 
by 
  -- Placeholder for the actual steps to prove
  sorry

end remainder_div_84_l225_225954


namespace percentage_fractions_l225_225893

theorem percentage_fractions : (3 / 8 / 100) * (160 : ℚ) = 3 / 5 :=
by
  sorry

end percentage_fractions_l225_225893


namespace equivalent_annual_rate_approx_l225_225272

noncomputable def annual_rate : ℝ := 0.045
noncomputable def days_in_year : ℝ := 365
noncomputable def daily_rate : ℝ := annual_rate / days_in_year
noncomputable def equivalent_annual_rate : ℝ := (1 + daily_rate) ^ days_in_year - 1

theorem equivalent_annual_rate_approx :
  abs (equivalent_annual_rate - 0.0459) < 0.0001 :=
by sorry

end equivalent_annual_rate_approx_l225_225272


namespace total_amount_paid_l225_225385

-- Define the quantities and rates as constants
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost functions
def cost_grapes (q : ℕ) (r : ℕ) : ℕ := q * r
def cost_mangoes (q : ℕ) (r : ℕ) : ℕ := q * r

-- Define the total cost function
def total_cost (c1 : ℕ) (c2 : ℕ) : ℕ := c1 + c2

-- State the proof problem
theorem total_amount_paid :
  total_cost (cost_grapes quantity_grapes rate_grapes) (cost_mangoes quantity_mangoes rate_mangoes) = 1055 :=
by
  sorry

end total_amount_paid_l225_225385


namespace clean_per_hour_l225_225432

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l225_225432


namespace harkamal_grapes_purchase_l225_225782

-- Define the conditions as parameters and constants
def cost_per_kg_grapes := 70
def kg_mangoes := 9
def cost_per_kg_mangoes := 45
def total_payment := 965

-- The theorem stating Harkamal purchased 8 kg of grapes
theorem harkamal_grapes_purchase : 
  ∃ G : ℕ, (cost_per_kg_grapes * G + cost_per_kg_mangoes * kg_mangoes = total_payment) ∧ G = 8 :=
by
  use 8
  unfold cost_per_kg_grapes cost_per_kg_mangoes kg_mangoes total_payment
  show 70 * 8 + 45 * 9 = 965 ∧ 8 = 8
  sorry

end harkamal_grapes_purchase_l225_225782


namespace simplified_expression_eq_l225_225758

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l225_225758


namespace triangle_centroid_property_l225_225469

def distance_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem triangle_centroid_property
  (A B C P : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hG : G = ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )) :
  distance_sq A P + distance_sq B P + distance_sq C P = 
  distance_sq A G + distance_sq B G + distance_sq C G + 3 * distance_sq G P :=
by
  sorry

end triangle_centroid_property_l225_225469


namespace john_bought_3_tshirts_l225_225953

theorem john_bought_3_tshirts (T : ℕ) (h : 20 * T + 50 = 110) : T = 3 := 
by 
  sorry

end john_bought_3_tshirts_l225_225953


namespace largest_base_5_five_digits_base_10_value_l225_225304

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l225_225304


namespace distinct_even_numbers_between_100_and_999_l225_225785

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end distinct_even_numbers_between_100_and_999_l225_225785


namespace point_quadrant_l225_225952

theorem point_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : b < 0 ∧ a > 0 := 
by {
  sorry
}

end point_quadrant_l225_225952


namespace rate_per_sq_meter_l225_225960

theorem rate_per_sq_meter
  (length : Float := 9)
  (width : Float := 4.75)
  (total_cost : Float := 38475)
  : (total_cost / (length * width)) = 900 := 
by
  sorry

end rate_per_sq_meter_l225_225960


namespace martin_distance_l225_225163

-- Define the given conditions
def speed : ℝ := 12.0
def time : ℝ := 6.0

-- State the theorem we want to prove
theorem martin_distance : speed * time = 72.0 := by
  sorry

end martin_distance_l225_225163


namespace functional_equation_solution_l225_225450

theorem functional_equation_solution (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f (f (m + n)) = f m + f n) :
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) := by
  sorry

end functional_equation_solution_l225_225450


namespace arrangement_count_l225_225608

theorem arrangement_count (students : Fin 6) (teacher : Bool) :
  (teacher = true) ∧
  ∀ (A B : Fin 6), 
    A ≠ 0 ∧ B ≠ 5 →
    A ≠ B →
    (Sorry) = 960 := sorry

end arrangement_count_l225_225608


namespace cube_root_of_neg_27_l225_225723

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end cube_root_of_neg_27_l225_225723


namespace y_relation_l225_225650

noncomputable def f (x : ℝ) : ℝ := -2 * x + 5

theorem y_relation (x1 y1 y2 y3 : ℝ) (h1 : y1 = f x1) (h2 : y2 = f (x1 - 2)) (h3 : y3 = f (x1 + 3)) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end y_relation_l225_225650


namespace log_monotonic_increasing_l225_225391

noncomputable def f (a x : ℝ) := Real.log x / Real.log a

theorem log_monotonic_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 1 < a) :
  f a (a + 1) > f a 2 := 
by
  -- Here the actual proof will be added.
  sorry

end log_monotonic_increasing_l225_225391


namespace remainder_base12_div_9_l225_225220

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l225_225220


namespace max_xy_max_xy_value_l225_225186

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y ≤ 3 :=
sorry

theorem max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y = 3 → x = 3 / 2 ∧ y = 2 :=
sorry

end max_xy_max_xy_value_l225_225186


namespace largest_k_inequality_l225_225022

noncomputable def k : ℚ := 39 / 2

theorem largest_k_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b + c)^3 ≥ (5 / 2) * (a^3 + b^3 + c^3) + k * a * b * c := 
sorry

end largest_k_inequality_l225_225022


namespace num_ducks_l225_225041

variable (D G : ℕ)

theorem num_ducks (h1 : D + G = 8) (h2 : 2 * D + 4 * G = 24) : D = 4 := by
  sorry

end num_ducks_l225_225041


namespace six_divides_p_plus_one_l225_225517

theorem six_divides_p_plus_one 
  (p : ℕ) 
  (prime_p : Nat.Prime p) 
  (gt_three_p : p > 3) 
  (prime_p_plus_two : Nat.Prime (p + 2)) 
  (gt_three_p_plus_two : p + 2 > 3) : 
  6 ∣ (p + 1) := 
sorry

end six_divides_p_plus_one_l225_225517


namespace y_coordinate_of_point_l225_225633

theorem y_coordinate_of_point (x y : ℝ) (m : ℝ)
  (h₁ : x = 10)
  (h₂ : y = m * x + -2)
  (m_def : m = (0 - (-4)) / (4 - (-4)))
  (h₃ : y = 3) : y = 3 :=
sorry

end y_coordinate_of_point_l225_225633


namespace exists_base_for_1994_no_base_for_1993_l225_225053

-- Problem 1: Existence of a base for 1994 with identical digits
theorem exists_base_for_1994 :
  ∃ b : ℕ, 1 < b ∧ b < 1993 ∧ (∃ a : ℕ, ∀ n : ℕ, 1994 = a * ((b ^ n - 1) / (b - 1)) ∧ a = 2) :=
sorry

-- Problem 2: Non-existence of a base for 1993 with identical digits
theorem no_base_for_1993 :
  ¬∃ b : ℕ, 1 < b ∧ b < 1992 ∧ (∃ a : ℕ, ∀ n : ℕ, 1993 = a * ((b ^ n - 1) / (b - 1))) :=
sorry

end exists_base_for_1994_no_base_for_1993_l225_225053


namespace sum_of_roots_l225_225502

theorem sum_of_roots (a b c : ℚ) (h_eq : 6 * a^3 + 7 * a^2 - 12 * a = 0) (h_eq_b : 6 * b^3 + 7 * b^2 - 12 * b = 0) (h_eq_c : 6 * c^3 + 7 * c^2 - 12 * c = 0) : 
  a + b + c = -7/6 := 
by
  -- Insert proof steps here
  sorry

end sum_of_roots_l225_225502


namespace divisibility_expression_l225_225732

variable {R : Type*} [CommRing R] (x a b : R)

theorem divisibility_expression :
  ∃ k : R, (x + a + b) ^ 3 - x ^ 3 - a ^ 3 - b ^ 3 = (x + a) * (x + b) * k :=
sorry

end divisibility_expression_l225_225732


namespace food_requirement_l225_225609

/-- Peter has six horses. Each horse eats 5 pounds of oats, three times a day, and 4 pounds of grain twice a day. -/
def totalFoodRequired (horses : ℕ) (days : ℕ) (oatsMeal : ℕ) (oatsMealsPerDay : ℕ) (grainMeal : ℕ) (grainMealsPerDay : ℕ) : ℕ :=
  let dailyOats := oatsMeal * oatsMealsPerDay
  let dailyGrain := grainMeal * grainMealsPerDay
  let dailyFood := dailyOats + dailyGrain
  let totalDailyFood := dailyFood * horses
  totalDailyFood * days

theorem food_requirement :
  totalFoodRequired 6 5 5 3 4 2 = 690 :=
by sorry

end food_requirement_l225_225609


namespace maximum_value_sum_l225_225271

theorem maximum_value_sum (a b c d : ℕ) (h1 : a + c = 1000) (h2 : b + d = 500) :
  ∃ a b c d, a + c = 1000 ∧ b + d = 500 ∧ (a = 1 ∧ c = 999 ∧ b = 499 ∧ d = 1) ∧ 
  ((a : ℝ) / b + (c : ℝ) / d = (1 / 499) + 999) := 
  sorry

end maximum_value_sum_l225_225271


namespace distance_between_foci_of_ellipse_l225_225551

-- Define the three given points
structure Point where
  x : ℝ
  y : ℝ

def p1 : Point := ⟨1, 3⟩
def p2 : Point := ⟨5, -1⟩
def p3 : Point := ⟨10, 3⟩

-- Define the statement that the distance between the foci of the ellipse they define is 2 * sqrt(4.25)
theorem distance_between_foci_of_ellipse : 
  ∃ (c : ℝ) (f : ℝ), f = 2 * Real.sqrt 4.25 ∧ 
  (∃ (ellipse : Point → Prop), ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
sorry

end distance_between_foci_of_ellipse_l225_225551


namespace find_difference_l225_225948

theorem find_difference (x y : ℕ) (hx : ∃ k : ℕ, x = k^2) (h_sum_prod : x + y = x * y - 2006) : y - x = 666 :=
sorry

end find_difference_l225_225948


namespace problem1_calculation_l225_225635

theorem problem1_calculation :
  (2 * Real.tan (Real.pi / 4) + (-1 / 2) ^ 0 + |Real.sqrt 3 - 1|) = 2 + Real.sqrt 3 :=
by
  sorry

end problem1_calculation_l225_225635


namespace robert_elizabeth_age_difference_l225_225182

theorem robert_elizabeth_age_difference 
  (patrick_age_1_5_times_robert : ∀ (robert_age : ℝ), ∃ (patrick_age : ℝ), patrick_age = 1.5 * robert_age)
  (elizabeth_born_after_richard : ∀ (richard_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = richard_age - 7 / 12)
  (elizabeth_younger_by_4_5_years : ∀ (patrick_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = patrick_age - 4.5)
  (robert_will_be_30_3_after_2_5_years : ∃ (robert_age_current : ℝ), robert_age_current = 30.3 - 2.5) :
  ∃ (years : ℤ) (months : ℤ), years = 9 ∧ months = 4 := by
  sorry

end robert_elizabeth_age_difference_l225_225182


namespace find_function_satisfying_property_l225_225166

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2 * x * y)

theorem find_function_satisfying_property (f : ℝ → ℝ) (h : ∀ x, 0 ≤ f x) (hf : example_function f) :
  ∃ a : ℝ, 0 ≤ a ∧ ∀ x : ℝ, f x = a * x^2 :=
sorry

end find_function_satisfying_property_l225_225166


namespace min_sum_abc_l225_225510

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l225_225510


namespace scientific_notation_110_billion_l225_225870

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ 110 * 10^8 = a * 10^n

theorem scientific_notation_110_billion :
  ∃ (a : ℝ) (n : ℤ), scientific_notation_form a n ∧ a = 1.1 ∧ n = 10 :=
by
  sorry

end scientific_notation_110_billion_l225_225870


namespace inequality_proof_l225_225138

theorem inequality_proof (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) (h5 : a * d = b * c) :
  (a - d) ^ 2 ≥ 4 * d + 8 := 
sorry

end inequality_proof_l225_225138


namespace final_position_correct_l225_225557

structure Position :=
(base : ℝ × ℝ)
(stem : ℝ × ℝ)

def initial_position : Position :=
{ base := (0, -1),
  stem := (1, 0) }

def reflect_x (p : Position) : Position :=
{ base := (p.base.1, -p.base.2),
  stem := (p.stem.1, -p.stem.2) }

def rotate_90_ccw (p : Position) : Position :=
{ base := (-p.base.2, p.base.1),
  stem := (-p.stem.2, p.stem.1) }

def half_turn (p : Position) : Position :=
{ base := (-p.base.1, -p.base.2),
  stem := (-p.stem.1, -p.stem.2) }

def reflect_y (p : Position) : Position :=
{ base := (-p.base.1, p.base.2),
  stem := (-p.stem.1, p.stem.2) }

def final_position : Position :=
reflect_y (half_turn (rotate_90_ccw (reflect_x initial_position)))

theorem final_position_correct : final_position = { base := (1, 0), stem := (0, 1) } :=
sorry

end final_position_correct_l225_225557


namespace find_unit_prices_minimal_cost_l225_225790

-- Definitions for part 1
def unitPrices (x y : ℕ) : Prop :=
  20 * x + 30 * y = 2920 ∧ x - y = 11 

-- Definitions for part 2
def costFunction (m : ℕ) : ℕ :=
  52 * m + 48 * (40 - m)

def additionalPurchase (m : ℕ) : Prop :=
  m ≥ 40 / 3

-- Statement for unit prices proof
theorem find_unit_prices (x y : ℕ) (h1 : 20 * x + 30 * y = 2920) (h2 : x - y = 11) : x = 65 ∧ y = 54 := 
  sorry

-- Statement for minimal cost proof
theorem minimal_cost (m : ℕ) (x y : ℕ) 
  (hx : 20 * x + 30 * y = 2920) 
  (hy : x - y = 11)
  (hx_65 : x = 65)
  (hy_54 : y = 54)
  (hm : m ≥ 40 / 3) : 
  costFunction m = 1976 ∧ m = 14 :=
  sorry

end find_unit_prices_minimal_cost_l225_225790


namespace solution_set_of_inequality_l225_225267

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x - 2 else if x < 0 then -(x - 2) else 0

theorem solution_set_of_inequality :
  {x : ℝ | f x < 1 / 2} =
  {x : ℝ | (0 ≤ x ∧ x < 5 / 2) ∨ x < -3 / 2} :=
by
  sorry

end solution_set_of_inequality_l225_225267


namespace problem1_problem2_l225_225191

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Problem 1: If ¬p is true, find the range of values for x
theorem problem1 {x : ℝ} (h : ¬ p x) : x > 2 ∨ x < -1 :=
by
  -- Proof omitted
  sorry

-- Problem 2: If ¬q is a sufficient but not necessary condition for ¬p, find the range of values for m
theorem problem2 {m : ℝ} (h : ∀ x : ℝ, ¬ q x m → ¬ p x) : m > 1 ∨ m < -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l225_225191


namespace find_amount_l225_225838

-- Given conditions
variables (x A : ℝ)

theorem find_amount :
  (0.65 * x = 0.20 * A) → (x = 190) → (A = 617.5) :=
by
  intros h1 h2
  sorry

end find_amount_l225_225838


namespace solve_system_of_equations_simplify_expression_l225_225811

-- Statement for system of equations
theorem solve_system_of_equations (s t : ℚ) 
  (h1 : 2 * s + 3 * t = 2) 
  (h2 : 2 * s - 6 * t = -1) :
  s = 1 / 2 ∧ t = 1 / 3 :=
sorry

-- Statement for simplifying the expression
theorem simplify_expression (x y : ℚ) :
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y :=
sorry

end solve_system_of_equations_simplify_expression_l225_225811


namespace max_cubes_submerged_l225_225035

noncomputable def cylinder_radius (diameter: ℝ) : ℝ := diameter / 2

noncomputable def water_volume (radius height: ℝ) : ℝ := Real.pi * radius^2 * height

noncomputable def cube_volume (edge: ℝ) : ℝ := edge^3

noncomputable def height_of_cubes (edge n: ℝ) : ℝ := edge * n

theorem max_cubes_submerged (diameter height water_height edge: ℝ) 
  (h1: diameter = 2.9)
  (h2: water_height = 4)
  (h3: edge = 2):
  ∃ max_n: ℝ, max_n = 5 := 
  sorry

end max_cubes_submerged_l225_225035


namespace time_to_cover_length_l225_225695

def escalator_rate : ℝ := 12 -- rate of the escalator in feet per second
def person_rate : ℝ := 8 -- rate of the person in feet per second
def escalator_length : ℝ := 160 -- length of the escalator in feet

theorem time_to_cover_length : escalator_length / (escalator_rate + person_rate) = 8 := by
  sorry

end time_to_cover_length_l225_225695


namespace circle_equation_l225_225682

def circle_center : (ℝ × ℝ) := (1, 2)
def radius : ℝ := 3

theorem circle_equation : 
  (∀ x y : ℝ, (x - circle_center.1) ^ 2 + (y - circle_center.2) ^ 2 = radius ^ 2 ↔ 
  (x - 1) ^ 2 + (y - 2) ^ 2 = 9) := 
by
  sorry

end circle_equation_l225_225682


namespace maximize_det_l225_225546

theorem maximize_det (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  (Matrix.det ![
    ![a, 1],
    ![1, b]
  ]) ≤ 0 :=
sorry

end maximize_det_l225_225546


namespace rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l225_225420

variable (a : ℂ)

theorem rationalize (h : a = 1 / (Real.sqrt 2 - 1)) : a = Real.sqrt 2 + 1 := by
  sorry

theorem value_of_a2_minus_2a (h : a = Real.sqrt 2 + 1) : a ^ 2 - 2 * a = 1 := by
  sorry

theorem value_of_2a3_minus_4a2_minus_1 (h : a = Real.sqrt 2 + 1) : 2 * a ^ 3 - 4 * a ^ 2 - 1 = 2 * Real.sqrt 2 + 1 := by
  sorry

end rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l225_225420


namespace train_length_is_150_l225_225616

noncomputable def train_length (v_km_hr : ℝ) (t_sec : ℝ) : ℝ :=
  let v_m_s := v_km_hr * (5 / 18)
  v_m_s * t_sec

theorem train_length_is_150 :
  train_length 122 4.425875438161669 = 150 :=
by
  -- It follows directly from the given conditions and known conversion factor
  -- The actual proof steps would involve arithmetic simplifications.
  sorry

end train_length_is_150_l225_225616


namespace question1_question2_l225_225033

namespace MathProofs

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

-- Definitions based on conditions
def isA := ∀ x, A x ↔ (-3 < x ∧ x < 2)
def isB := ∀ x, B x ↔ (Real.exp (x - 1) ≥ 1)
def isCuA := ∀ x, (U \ A) x ↔ (x ≤ -3 ∨ x ≥ 2)

-- Proof of Question 1
theorem question1 : (∀ x, (A ∪ B) x ↔ (x > -3)) := by
  sorry

-- Proof of Question 2
theorem question2 : (∀ x, ((U \ A) ∩ B) x ↔ (x ≥ 2)) := by
  sorry

end MathProofs

end question1_question2_l225_225033


namespace range_of_m_l225_225417

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + 1 > 0) ↔ (-2 < m ∧ m < 2) :=
  sorry

end range_of_m_l225_225417


namespace billy_age_l225_225921

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 64) : B = 48 :=
by
  sorry

end billy_age_l225_225921


namespace parallel_lines_intersect_hyperbola_l225_225595

noncomputable def point_A : (ℝ × ℝ) := (0, 14)
noncomputable def point_B : (ℝ × ℝ) := (0, 4)
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

theorem parallel_lines_intersect_hyperbola (k : ℝ)
  (x_K x_L x_M x_N : ℝ) 
  (hAK : hyperbola x_K = k * x_K + 14) (hAL : hyperbola x_L = k * x_L + 14)
  (hBM : hyperbola x_M = k * x_M + 4) (hBN : hyperbola x_N = k * x_N + 4)
  (vieta1 : x_K + x_L = -14 / k) (vieta2 : x_M + x_N = -4 / k) :
  (AL - AK) / (BN - BM) = 3.5 :=
by
  sorry

end parallel_lines_intersect_hyperbola_l225_225595


namespace third_box_nuts_l225_225162

theorem third_box_nuts
  (A B C : ℕ)
  (h1 : A = B + C - 6)
  (h2 : B = A + C - 10) :
  C = 8 :=
by
  sorry

end third_box_nuts_l225_225162


namespace unique_three_digit_numbers_count_l225_225544

theorem unique_three_digit_numbers_count :
  ∃ l : List Nat, (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    l = [230, 203, 302, 320] ∧ l.length = 4 := 
by
  sorry

end unique_three_digit_numbers_count_l225_225544


namespace man_climbs_out_of_well_in_65_days_l225_225776

theorem man_climbs_out_of_well_in_65_days (depth climb slip net_days last_climb : ℕ) 
  (h_depth : depth = 70)
  (h_climb : climb = 6)
  (h_slip : slip = 5)
  (h_net_days : net_days = 64)
  (h_last_climb : last_climb = 1) :
  ∃ days : ℕ, days = net_days + last_climb ∧ days = 65 := by
  sorry

end man_climbs_out_of_well_in_65_days_l225_225776


namespace snake_count_l225_225207

def neighborhood : Type := {n : ℕ // n = 200}

def percentage (total : ℕ) (percent : ℕ) : ℕ := total * percent / 100

def owns_only_dogs (total : ℕ) : ℕ := percentage total 13
def owns_only_cats (total : ℕ) : ℕ := percentage total 10
def owns_only_snakes (total : ℕ) : ℕ := percentage total 5
def owns_only_rabbits (total : ℕ) : ℕ := percentage total 7
def owns_only_birds (total : ℕ) : ℕ := percentage total 3
def owns_only_exotic (total : ℕ) : ℕ := percentage total 6
def owns_dogs_and_cats (total : ℕ) : ℕ := percentage total 8
def owns_dogs_cats_exotic (total : ℕ) : ℕ := percentage total 9
def owns_cats_and_snakes (total : ℕ) : ℕ := percentage total 4
def owns_cats_and_birds (total : ℕ) : ℕ := percentage total 2
def owns_snakes_and_rabbits (total : ℕ) : ℕ := percentage total 5
def owns_snakes_and_birds (total : ℕ) : ℕ := percentage total 3
def owns_rabbits_and_birds (total : ℕ) : ℕ := percentage total 1
def owns_all_except_snakes (total : ℕ) : ℕ := percentage total 2
def owns_all_except_birds (total : ℕ) : ℕ := percentage total 1
def owns_three_with_exotic (total : ℕ) : ℕ := percentage total 11
def owns_only_chameleons (total : ℕ) : ℕ := percentage total 3
def owns_only_hedgehogs (total : ℕ) : ℕ := percentage total 2

def exotic_pet_owners (total : ℕ) : ℕ :=
  owns_only_exotic total + owns_dogs_cats_exotic total + owns_all_except_snakes total +
  owns_all_except_birds total + owns_three_with_exotic total + owns_only_chameleons total +
  owns_only_hedgehogs total

def exotic_pet_owners_with_snakes (total : ℕ) : ℕ :=
  percentage (exotic_pet_owners total) 25

def total_snake_owners (total : ℕ) : ℕ :=
  owns_only_snakes total + owns_cats_and_snakes total +
  owns_snakes_and_rabbits total + owns_snakes_and_birds total +
  exotic_pet_owners_with_snakes total

theorem snake_count (nh : neighborhood) : total_snake_owners (nh.val) = 51 :=
by
  sorry

end snake_count_l225_225207


namespace simplify_and_compute_l225_225416

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l225_225416


namespace system_has_three_solutions_l225_225976

theorem system_has_three_solutions (a : ℝ) :
  (a = 4 ∨ a = 64 ∨ a = 51 + 10 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), 
    (x = abs (y - Real.sqrt a) + Real.sqrt a - 4 
    ∧ (abs x - 6)^2 + (abs y - 8)^2 = 100) 
        ∧ (∃! x1 y1 : ℝ, (x1 = abs (y1 - Real.sqrt a) + Real.sqrt a - 4 
        ∧ (abs x1 - 6)^2 + (abs y1 - 8)^2 = 100)) :=
by
  sorry

end system_has_three_solutions_l225_225976


namespace packs_sold_in_other_villages_l225_225936

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages_l225_225936


namespace adam_has_more_apples_l225_225104

-- Define the number of apples Jackie has
def JackiesApples : Nat := 9

-- Define the number of apples Adam has
def AdamsApples : Nat := 14

-- Statement of the problem: Prove that Adam has 5 more apples than Jackie
theorem adam_has_more_apples :
  AdamsApples - JackiesApples = 5 :=
by
  sorry

end adam_has_more_apples_l225_225104


namespace range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l225_225703

/-- There exists a real number x such that 2x^2 + (m-1)x + 1/2 ≤ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1 / 2 ≤ 0

/-- The curve C1: x^2/m^2 + y^2/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def proposition_q (m : ℝ) : Prop :=
  m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0

/-- The curve C2: x^2/(m-t) + y^2/(m-t-1) = 1 represents a hyperbola -/
def proposition_s (m t : ℝ) : Prop :=
  (m - t) * (m - t - 1) < 0

/-- Find the range of values for m if p and q are true -/
theorem range_of_m_if_p_and_q_true (m : ℝ) :
  proposition_p m ∧ proposition_q m ↔ (-4 < m ∧ m < -2) ∨ m > 4 :=
  sorry

/-- Find the range of values for t if q is a necessary but not sufficient condition for s -/
theorem range_of_t_if_q_necessary_for_s (m t : ℝ) :
  (∀ m, proposition_q m → proposition_s m t) ∧ ¬(proposition_s m t → proposition_q m) ↔ 
  (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
  sorry

end range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l225_225703


namespace gnollish_valid_sentence_count_is_48_l225_225795

-- Define the problem parameters
def gnollish_words : List String := ["word1", "word2", "splargh", "glumph", "kreeg"]

def valid_sentence_count : Nat :=
  let total_sentences := 4 * 4 * 4
  let invalid_sentences :=
    4 +         -- (word) splargh glumph
    4 +         -- splargh glumph (word)
    4 +         -- (word) splargh kreeg
    4           -- splargh kreeg (word)
  total_sentences - invalid_sentences

-- Prove that the number of valid 3-word sentences is 48
theorem gnollish_valid_sentence_count_is_48 : valid_sentence_count = 48 := by
  sorry

end gnollish_valid_sentence_count_is_48_l225_225795


namespace simplify_fraction_l225_225222

theorem simplify_fraction : 
  1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_fraction_l225_225222


namespace find_a_l225_225967

-- Given conditions as definitions.
def f (a x : ℝ) := a * x^3
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 3 * x + a - 3

-- Problem statement in Lean 4.
theorem find_a (a : ℝ) (h_tangent : ∀ x : ℝ, f a 1 = 1 ∧ f a 1 = tangent_line a 1) : a = 1 := 
by sorry

end find_a_l225_225967


namespace find_m_plus_n_l225_225613

noncomputable def overlapping_points (A B: ℝ × ℝ) (C D: ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let M_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let axis_slope := - 1 / k_AB
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  let M_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  k_CD = axis_slope ∧ (M_CD.2 - M_AB.2) = axis_slope * (M_CD.1 - M_AB.1)

theorem find_m_plus_n : 
  ∃ (m n: ℝ), overlapping_points (0, 2) (4, 0) (7, 3) (m, n) ∧ m + n = 34 / 5 :=
sorry

end find_m_plus_n_l225_225613


namespace at_least_one_not_less_than_100_l225_225129

-- Defining the original propositions
def p : Prop := ∀ (A_score : ℕ), A_score ≥ 100
def q : Prop := ∀ (B_score : ℕ), B_score < 100

-- Assertion to be proved in Lean
theorem at_least_one_not_less_than_100 (h1 : p) (h2 : q) : p ∨ ¬q := 
sorry

end at_least_one_not_less_than_100_l225_225129


namespace possible_values_l225_225889

theorem possible_values (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  ∃ S : Set ℝ, S = {x : ℝ | 4 ≤ x} ∧ (1 / a + 1 / b) ∈ S :=
by
  sorry

end possible_values_l225_225889


namespace find_k_intersection_on_line_l225_225817

theorem find_k_intersection_on_line (k : ℝ) :
  (∃ (x y : ℝ), x - 2 * y - 2 * k = 0 ∧ 2 * x - 3 * y - k = 0 ∧ 3 * x - y = 0) → k = 0 :=
by
  sorry

end find_k_intersection_on_line_l225_225817


namespace math_problem_l225_225717

theorem math_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 * x + y - x * y = 0) : 
  ((9 * x + y) * (9 / y + 1 / x) = x * y) ∧ ¬ ((x / 9) + y = 10) ∧ 
  ((x + y = 16) ↔ (x = 4 ∧ y = 12)) ∧ 
  ((x * y = 36) ↔ (x = 2 ∧ y = 18)) :=
by {
  sorry
}

end math_problem_l225_225717


namespace longer_subsegment_length_l225_225187

-- Define the given conditions and proof goal in Lean 4
theorem longer_subsegment_length {DE EF DF DG GF : ℝ} (h1 : 3 * EF < 4 * EF) (h2 : 4 * EF < 5 * EF)
  (ratio_condition : DE / EF = 4 / 5) (DF_length : DF = 12) :
  DG + GF = DF ∧ DE / EF = DG / GF ∧ GF = (5 * 12 / 9) :=
by
  sorry

end longer_subsegment_length_l225_225187


namespace four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l225_225244

-- Number of four-digit numbers greater than 3999 such that the product of the middle two digits > 12 is 4260
theorem four_digit_numbers_greater_3999_with_middle_product_exceeding_12
  {d1 d2 d3 d4 : ℕ}
  (h1 : 4 ≤ d1 ∧ d1 ≤ 9)
  (h2 : 0 ≤ d4 ∧ d4 ≤ 9)
  (h3 : 1 ≤ d2 ∧ d2 ≤ 9)
  (h4 : 1 ≤ d3 ∧ d3 ≤ 9)
  (h5 : d2 * d3 > 12) :
  (6 * 71 * 10 = 4260) :=
by
  sorry

end four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l225_225244


namespace original_number_of_people_l225_225335

theorem original_number_of_people (x : ℕ) 
  (h1 : (x / 2) - ((x / 2) / 3) = 12) : 
  x = 36 :=
sorry

end original_number_of_people_l225_225335


namespace mid_point_between_fractions_l225_225299

theorem mid_point_between_fractions : (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end mid_point_between_fractions_l225_225299


namespace rate_of_current_l225_225645

variable (c : ℝ)

-- Define the given conditions
def speed_still_water : ℝ := 4.5
def time_ratio : ℝ := 2

-- Define the effective speeds
def speed_downstream : ℝ := speed_still_water + c
def speed_upstream : ℝ := speed_still_water - c

-- Define the condition that it takes twice as long to row upstream as downstream
def rowing_equation : Prop := 1 / speed_upstream = 2 * (1 / speed_downstream)

-- The Lean theorem stating the problem we need to prove
theorem rate_of_current (h : rowing_equation) : c = 1.5 := by
  sorry

end rate_of_current_l225_225645


namespace how_many_months_to_buy_tv_l225_225112

-- Definitions based on given conditions
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500

def total_expenses := food_expenses + utilities_expenses + other_expenses
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000
def monthly_savings := monthly_income - total_expenses

-- Theorem statement based on the problem
theorem how_many_months_to_buy_tv 
    (H_income : monthly_income = 30000)
    (H_food : food_expenses = 15000)
    (H_utilities : utilities_expenses = 5000)
    (H_other : other_expenses = 2500)
    (H_savings : current_savings = 10000)
    (H_tv_cost : tv_cost = 25000)
    : (tv_cost - current_savings) / monthly_savings = 2 :=
by
  sorry

end how_many_months_to_buy_tv_l225_225112


namespace drive_time_is_eleven_hours_l225_225603

-- Define the distances and speed as constants
def distance_salt_lake_to_vegas : ℕ := 420
def distance_vegas_to_los_angeles : ℕ := 273
def average_speed : ℕ := 63

-- Calculate the total distance
def total_distance : ℕ := distance_salt_lake_to_vegas + distance_vegas_to_los_angeles

-- Calculate the total time required
def total_time : ℕ := total_distance / average_speed

-- Theorem stating Andy wants to complete the drive in 11 hours
theorem drive_time_is_eleven_hours : total_time = 11 := sorry

end drive_time_is_eleven_hours_l225_225603


namespace find_n_l225_225824

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 := by
  intros h
  sorry

end find_n_l225_225824


namespace expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l225_225917

-- Definitions to add parentheses in the given expressions to achieve the desired results.
def expr1 := 7 * (9 + 12 / 3)
def expr2 := (7 * 9 + 12) / 3
def expr3 := 7 * (9 + 12) / 3
def expr4 := (48 * 6) / (48 * 6)

-- Proof statements
theorem expr1_is_91 : expr1 = 91 := 
by sorry

theorem expr2_is_25 : expr2 = 25 :=
by sorry

theorem expr3_is_49 : expr3 = 49 :=
by sorry

theorem expr4_is_1 : expr4 = 1 :=
by sorry

end expr1_is_91_expr2_is_25_expr3_is_49_expr4_is_1_l225_225917


namespace petes_average_speed_l225_225830

theorem petes_average_speed
    (map_distance : ℝ := 5) 
    (time_taken : ℝ := 1.5) 
    (map_scale : ℝ := 0.05555555555555555) :
    (map_distance / map_scale) / time_taken = 60 := 
by
    sorry

end petes_average_speed_l225_225830


namespace complement_intersection_U_l225_225232

-- Definitions of the sets based on the given conditions
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to another set
def complement (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Statement asserting the equivalence
theorem complement_intersection_U :
  complement U (M ∩ N) = {1, 4} :=
by
  sorry

end complement_intersection_U_l225_225232


namespace waiters_hired_l225_225698

theorem waiters_hired (W H : ℕ) (h1 : 3 * W = 90) (h2 : 3 * (W + H) = 126) : H = 12 :=
sorry

end waiters_hired_l225_225698


namespace hat_value_in_rice_l225_225183

variables (f l r h : ℚ)

theorem hat_value_in_rice :
  (4 * f = 3 * l) →
  (l = 5 * r) →
  (5 * f = 7 * h) →
  h = (75 / 28) * r :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end hat_value_in_rice_l225_225183


namespace M_inter_N_eq_l225_225793

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_inter_N_eq : (M ∩ N) = Set.Ico 0 (1/2) := 
by
  sorry

end M_inter_N_eq_l225_225793


namespace curve_cartesian_equation_max_value_3x_plus_4y_l225_225397

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem curve_cartesian_equation :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∀ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) → (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

theorem max_value_3x_plus_4y :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∃ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) ∧ (∀ ϴ : ℝ, 3 * (3 * Real.cos ϴ) + 4 * (2 * Real.sin ϴ) ≤ Real.sqrt 145) :=
sorry

end curve_cartesian_equation_max_value_3x_plus_4y_l225_225397


namespace solution_to_inequality_system_l225_225124

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_to_inequality_system_l225_225124


namespace legacy_earnings_l225_225107

theorem legacy_earnings 
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (earnings_per_hour : ℕ)
  (total_floors : floors = 4)
  (total_rooms_per_floor : rooms_per_floor = 10)
  (time_per_room : hours_per_room = 6)
  (rate_per_hour : earnings_per_hour = 15) :
  floors * rooms_per_floor * hours_per_room * earnings_per_hour = 3600 := 
by
  sorry

end legacy_earnings_l225_225107


namespace sum_of_two_numbers_is_147_l225_225086

theorem sum_of_two_numbers_is_147 (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) :
  A + B = 147 :=
by
  sorry

end sum_of_two_numbers_is_147_l225_225086


namespace hyperbola_center_l225_225320

theorem hyperbola_center :
  ∃ c : ℝ × ℝ, (c = (3, 4) ∧ ∀ x y : ℝ, 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0 ↔ (x - 3)^2 / 4 - (y - 4)^2 / 1 = 1) :=
sorry

end hyperbola_center_l225_225320


namespace probability_correct_l225_225570
noncomputable def probability_no_2_in_id : ℚ :=
  let total_ids := 5000
  let valid_ids := 2916
  valid_ids / total_ids

theorem probability_correct : probability_no_2_in_id = 729 / 1250 := by
  sorry

end probability_correct_l225_225570


namespace action_figures_added_l225_225306

-- Definitions according to conditions
def initial_action_figures : ℕ := 4
def books_on_shelf : ℕ := 22 -- This information is not necessary for proving the action figures added
def total_action_figures_after_adding : ℕ := 10

-- Theorem to prove given the conditions
theorem action_figures_added : (total_action_figures_after_adding - initial_action_figures) = 6 := by
  sorry

end action_figures_added_l225_225306


namespace evaluate_expression_l225_225295

variable (x y z : ℤ)

theorem evaluate_expression :
  x = 3 → y = 2 → z = 4 → 3 * x - 4 * y + 5 * z = 21 :=
by
  intros hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l225_225295


namespace tan_add_tan_105_eq_l225_225130

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l225_225130


namespace olivia_earnings_l225_225206

-- Define Olivia's hourly wage
def wage : ℕ := 9

-- Define the hours worked on each day
def hours_monday : ℕ := 4
def hours_wednesday : ℕ := 3
def hours_friday : ℕ := 6

-- Define the total hours worked
def total_hours : ℕ := hours_monday + hours_wednesday + hours_friday

-- Define the total earnings
def total_earnings : ℕ := total_hours * wage

-- State the theorem
theorem olivia_earnings : total_earnings = 117 :=
by
  sorry

end olivia_earnings_l225_225206


namespace quad_vertex_transform_l225_225577

theorem quad_vertex_transform :
  ∀ (x y : ℝ) (h : y = -2 * x^2) (new_x new_y : ℝ) (h_translation : new_x = x + 3 ∧ new_y = y - 2),
  new_y = -2 * (new_x - 3)^2 + 2 :=
by
  intros x y h new_x new_y h_translation
  sorry

end quad_vertex_transform_l225_225577


namespace case_one_case_two_l225_225789

theorem case_one (n : ℝ) (h : n > -1) : n^3 + 1 > n^2 + n :=
sorry

theorem case_two (n : ℝ) (h : n < -1) : n^3 + 1 < n^2 + n :=
sorry

end case_one_case_two_l225_225789


namespace time_to_cross_pole_is_2_5_l225_225147

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end time_to_cross_pole_is_2_5_l225_225147


namespace find_digit_B_l225_225964

theorem find_digit_B (A B : ℕ) (h1 : 100 * A + 78 - (210 + B) = 364) : B = 4 :=
by sorry

end find_digit_B_l225_225964


namespace percentage_microphotonics_l225_225593

noncomputable def percentage_home_electronics : ℝ := 24
noncomputable def percentage_food_additives : ℝ := 20
noncomputable def percentage_GMO : ℝ := 29
noncomputable def percentage_industrial_lubricants : ℝ := 8
noncomputable def angle_basic_astrophysics : ℝ := 18

theorem percentage_microphotonics : 
  ∀ (home_elec food_additives GMO industrial_lub angle_bas_astro : ℝ),
  home_elec = 24 →
  food_additives = 20 →
  GMO = 29 →
  industrial_lub = 8 →
  angle_bas_astro = 18 →
  (100 - (home_elec + food_additives + GMO + industrial_lub + ((angle_bas_astro / 360) * 100))) = 14 :=
by
  intros _ _ _ _ _
  sorry

end percentage_microphotonics_l225_225593


namespace min_x_prime_sum_l225_225530

theorem min_x_prime_sum (x y : ℕ) (h : 3 * x^2 = 5 * y^4) :
  ∃ a b c d : ℕ, x = a^b * c^d ∧ (a + b + c + d = 11) := 
by sorry

end min_x_prime_sum_l225_225530


namespace find_x_l225_225504

theorem find_x (x : ℝ) (h₀ : ⌊x⌋ * x = 162) : x = 13.5 :=
sorry

end find_x_l225_225504


namespace reciprocals_expression_value_l225_225691

theorem reciprocals_expression_value (a b : ℝ) (h : a * b = 1) : a^2 * b - (a - 2023) = 2023 := 
by 
  sorry

end reciprocals_expression_value_l225_225691


namespace jeff_corrected_mean_l225_225962

def initial_scores : List ℕ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℕ := [85, 90, 92, 93, 89, 89, 88]

noncomputable def arithmetic_mean (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / (scores.length : ℝ)

theorem jeff_corrected_mean :
  arithmetic_mean corrected_scores = 89.42857142857143 := 
by
  sorry

end jeff_corrected_mean_l225_225962


namespace M_inter_N_l225_225693

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {-1, 0}

theorem M_inter_N :
  M ∩ N = {0} :=
by
  sorry

end M_inter_N_l225_225693


namespace quadrilateral_area_l225_225394

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 300 := 
by
  sorry

end quadrilateral_area_l225_225394


namespace cone_to_cylinder_water_height_l225_225428

theorem cone_to_cylinder_water_height :
  let r_cone := 15 -- radius of the cone
  let h_cone := 24 -- height of the cone
  let r_cylinder := 18 -- radius of the cylinder
  let V_cone := (1 / 3: ℝ) * Real.pi * r_cone^2 * h_cone -- volume of the cone
  let h_cylinder := V_cone / (Real.pi * r_cylinder^2) -- height of the water in the cylinder
  h_cylinder = 8.33 := by
  sorry

end cone_to_cylinder_water_height_l225_225428


namespace pyramid_base_side_length_l225_225694

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l225_225694


namespace jason_total_amount_l225_225910

def shorts_price : ℝ := 14.28
def jacket_price : ℝ := 4.74
def shoes_price : ℝ := 25.95
def socks_price : ℝ := 6.80
def tshirts_price : ℝ := 18.36
def hat_price : ℝ := 12.50
def swimsuit_price : ℝ := 22.95
def sunglasses_price : ℝ := 45.60
def wristbands_price : ℝ := 9.80

def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount)

def total_discounted_price : ℝ := 
  (discounted_price shorts_price discount1) + 
  (discounted_price jacket_price discount1) + 
  (discounted_price hat_price discount1) + 
  (discounted_price shoes_price discount2) + 
  (discounted_price socks_price discount2) + 
  (discounted_price tshirts_price discount2) + 
  (discounted_price swimsuit_price discount2) + 
  (discounted_price sunglasses_price discount2) + 
  (discounted_price wristbands_price discount2)

def total_with_tax : ℝ := total_discounted_price + (total_discounted_price * sales_tax_rate)

theorem jason_total_amount : total_with_tax = 153.07 := by
  sorry

end jason_total_amount_l225_225910


namespace smallest_mu_ineq_l225_225607

theorem smallest_mu_ineq (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    a^2 + b^2 + c^2 + d^2 + 2 * a * d ≥ 2 * (a * b + b * c + c * d) := by {
    sorry
}

end smallest_mu_ineq_l225_225607


namespace last_digit_p_minus_q_not_5_l225_225883

theorem last_digit_p_minus_q_not_5 (p q : ℕ) (n : ℕ) 
  (h1 : p * q = 10^n) 
  (h2 : ¬ (p % 10 = 0))
  (h3 : ¬ (q % 10 = 0))
  (h4 : p > q) : (p - q) % 10 ≠ 5 :=
by sorry

end last_digit_p_minus_q_not_5_l225_225883


namespace painting_time_l225_225972

theorem painting_time (n₁ t₁ n₂ t₂ : ℕ) (h1 : n₁ = 8) (h2 : t₁ = 12) (h3 : n₂ = 6) (h4 : n₁ * t₁ = n₂ * t₂) : t₂ = 16 :=
by
  sorry

end painting_time_l225_225972


namespace shortest_distance_l225_225150

-- The initial position of the cowboy.
def initial_position : ℝ × ℝ := (-2, -6)

-- The position of the cabin relative to the cowboy's initial position.
def cabin_position : ℝ × ℝ := (10, -15)

-- The equation of the stream flowing due northeast.
def stream_equation : ℝ → ℝ := id  -- y = x

-- Function to calculate the distance between two points (x1, y1) and (x2, y2).
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Calculate the reflection point of C over y = x.
def reflection_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Main proof statement: shortest distance the cowboy can travel.
theorem shortest_distance : distance initial_position (reflection_point initial_position) +
                            distance (reflection_point initial_position) cabin_position = 8 +
                            Real.sqrt 545 :=
by
  sorry

end shortest_distance_l225_225150


namespace nilpotent_matrix_squared_zero_l225_225051

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l225_225051


namespace factorize_polynomial_l225_225778

theorem factorize_polynomial (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2 * y) ^ 2 := 
by sorry

end factorize_polynomial_l225_225778


namespace part1_part2_part3_l225_225463

-- Definitions of conditions
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_even_between (a b : ℕ) : ℕ := sum_even b - sum_even a

-- Problem 1: Prove that for n = 8, S = 72
theorem part1 (n : ℕ) (h : n = 8) : sum_even n = 72 := by
  rw [h]
  exact rfl

-- Problem 2: Prove the general formula for the sum of the first n consecutive even numbers
theorem part2 (n : ℕ) : sum_even n = n * (n + 1) := by
  exact rfl

-- Problem 3: Prove the sum of 102 to 212 is 8792 using the formula
theorem part3 : sum_even_between 50 106 = 8792 := by
  sorry

end part1_part2_part3_l225_225463


namespace negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l225_225040

-- Definitions based on the conditions in the problem:
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b
def MonotonicFunction (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The proposition that 'All linear functions are monotonic functions'
def AllLinearAreMonotonic : Prop := ∀ (f : ℝ → ℝ), LinearFunction f → MonotonicFunction f

-- The correct answer to the question:
def SomeLinearAreNotMonotonic : Prop := ∃ (f : ℝ → ℝ), LinearFunction f ∧ ¬ MonotonicFunction f

-- The proof problem:
theorem negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic : 
  ¬ AllLinearAreMonotonic ↔ SomeLinearAreNotMonotonic :=
by
  sorry

end negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l225_225040


namespace equations_have_different_graphs_l225_225398

theorem equations_have_different_graphs :
  ¬(∀ x : ℝ, (2 * (x - 3)) / (x + 3) = 2 * (x - 3) ∧ 
              (x + 3) * ((2 * x^2 - 18) / (x + 3)) = 2 * x^2 - 18 ∧
              (2 * x - 3) = (2 * (x - 3)) ∧ 
              (2 * x - 3) = (2 * x - 3)) :=
by
  sorry

end equations_have_different_graphs_l225_225398


namespace max_f_l225_225697

noncomputable def S_n (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℚ :=
  S_n n / ((n + 32) * S_n (n + 1))

theorem max_f (n : ℕ) : f n ≤ 1 / 50 := sorry

-- Verify the bound is achieved for n = 8
example : f 8 = 1 / 50 := by
  unfold f S_n
  norm_num

end max_f_l225_225697


namespace selling_price_of_article_l225_225594

theorem selling_price_of_article (CP : ℝ) (L_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 600) 
  (h2 : L_percent = 50) 
  : SP = 300 := 
by
  sorry

end selling_price_of_article_l225_225594


namespace quadratic_condition_l225_225210

noncomputable def quadratic_sufficiency (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + x + m = 0 → m < 1/4

noncomputable def quadratic_necessity (m : ℝ) : Prop :=
  (∃ (x : ℝ), x^2 + x + m = 0) → m ≤ 1/4

theorem quadratic_condition (m : ℝ) : 
  (m < 1/4 → quadratic_sufficiency m) ∧ ¬ quadratic_necessity m := 
sorry

end quadratic_condition_l225_225210


namespace largest_digit_B_divisible_by_3_l225_225666

-- Define the six-digit number form and the known digits sum.
def isIntegerDivisibleBy3 (b : ℕ) : Prop :=
  b < 10 ∧ (b + 30) % 3 = 0

-- The main theorem: Find the largest digit B such that the number 4B5,894 is divisible by 3.
theorem largest_digit_B_divisible_by_3 : ∃ (B : ℕ), isIntegerDivisibleBy3 B ∧ ∀ (b' : ℕ), isIntegerDivisibleBy3 b' → b' ≤ B := by
  -- Notice the existential and universal quantifiers involved in finding the largest B.
  sorry

end largest_digit_B_divisible_by_3_l225_225666


namespace average_of_data_set_l225_225599

theorem average_of_data_set :
  (7 + 5 + (-2) + 5 + 10) / 5 = 5 :=
by sorry

end average_of_data_set_l225_225599


namespace circles_are_intersecting_l225_225900

-- Define the circles and the distances given
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 5
def distance_O1O2 : ℝ := 2

-- Define the positional relationships
inductive PositionalRelationship
| externally_tangent
| intersecting
| internally_tangent
| contained_within_each_other

open PositionalRelationship

-- State the theorem to be proved
theorem circles_are_intersecting :
  distance_O1O2 > 0 ∧ distance_O1O2 < (radius_O1 + radius_O2) ∧ distance_O1O2 > abs (radius_O1 - radius_O2) →
  PositionalRelationship := 
by
  intro h
  exact PositionalRelationship.intersecting

end circles_are_intersecting_l225_225900


namespace germination_rate_sunflower_l225_225380

variable (s_d s_s f_d f_s p : ℕ) (g_d g_f : ℚ)

-- Define the conditions
def conditions :=
  s_d = 25 ∧ s_s = 25 ∧ g_d = 0.60 ∧ g_f = 0.80 ∧ p = 28 ∧ f_d = 12 ∧ f_s = 16

-- Define the statement to be proved
theorem germination_rate_sunflower (h : conditions s_d s_s f_d f_s p g_d g_f) : 
  (f_s / (g_f * (s_s : ℚ))) > 0.0 ∧ (f_s / (g_f * (s_s : ℚ)) * 100) = 80 := 
by
  sorry

end germination_rate_sunflower_l225_225380


namespace asher_speed_l225_225773

theorem asher_speed :
  (5 * 60 ≠ 0) → (6600 / (5 * 60) = 22) :=
by
  intros h
  sorry

end asher_speed_l225_225773


namespace necessary_but_not_sufficient_l225_225355

theorem necessary_but_not_sufficient (x : ℝ) :
  (x - 1) * (x + 2) = 0 → (x = 1 ∨ x = -2) ∧ (x = 1 → (x - 1) * (x + 2) = 0) ∧ ¬((x - 1) * (x + 2) = 0 ↔ x = 1) :=
by
  sorry

end necessary_but_not_sufficient_l225_225355


namespace evaluate_expression_l225_225975

theorem evaluate_expression : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) + Int.ceil (4 / 5 : ℚ) + Int.floor (-4 / 5 : ℚ) = 0 :=
by
  sorry

end evaluate_expression_l225_225975


namespace ham_and_bread_percentage_l225_225328

-- Defining the different costs as constants
def cost_of_bread : ℝ := 50
def cost_of_ham : ℝ := 150
def cost_of_cake : ℝ := 200

-- Defining the total cost of the items
def total_cost : ℝ := cost_of_bread + cost_of_ham + cost_of_cake

-- Defining the combined cost of ham and bread
def combined_cost_ham_and_bread : ℝ := cost_of_bread + cost_of_ham

-- The theorem stating that the combined cost of ham and bread is 50% of the total cost
theorem ham_and_bread_percentage : (combined_cost_ham_and_bread / total_cost) * 100 = 50 := by
  sorry  -- Proof to be provided

end ham_and_bread_percentage_l225_225328


namespace time_between_last_two_rings_l225_225576

variable (n : ℕ) (x y : ℝ)

noncomputable def timeBetweenLastTwoRings : ℝ :=
  x + (n - 3) * y

theorem time_between_last_two_rings :
  timeBetweenLastTwoRings n x y = x + (n - 3) * y :=
by
  sorry

end time_between_last_two_rings_l225_225576


namespace birgit_time_to_travel_8km_l225_225369

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l225_225369


namespace ages_total_l225_225325

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l225_225325


namespace sin_polar_circle_l225_225841

theorem sin_polar_circle (t : ℝ) (θ : ℝ) (r : ℝ) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) :
  t = Real.pi := 
by
  sorry

end sin_polar_circle_l225_225841


namespace correct_operation_l225_225373

theorem correct_operation :
  (2 * a - a ≠ 2) ∧ ((a - 1) * (a - 1) ≠ a ^ 2 - 1) ∧ (a ^ 6 / a ^ 3 ≠ a ^ 2) ∧ ((-2 * a ^ 3) ^ 2 = 4 * a ^ 6) :=
by
  sorry

end correct_operation_l225_225373


namespace sum_of_possible_b_values_l225_225798

noncomputable def g (x b : ℝ) : ℝ := x^2 - b * x + 3 * b

theorem sum_of_possible_b_values :
  (∀ (x₀ x₁ : ℝ), g x₀ x₁ = 0 → g x₀ x₁ = (x₀ - x₁) * (x₀ - 3)) → ∃ b : ℝ, b = 12 ∨ b = 16 :=
sorry

end sum_of_possible_b_values_l225_225798


namespace range_of_a_l225_225887

def A := {x : ℝ | x * (4 - x) ≥ 3}
def B (a : ℝ) := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a < 1) := by
  sorry

end range_of_a_l225_225887


namespace average_marks_physics_chemistry_l225_225190

theorem average_marks_physics_chemistry
  (P C M : ℕ)
  (h1 : (P + C + M) / 3 = 60)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 140) :
  (P + C) / 2 = 70 :=
by
  sorry

end average_marks_physics_chemistry_l225_225190


namespace simplify_expression_l225_225087

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ( ((a ^ (4 / 3 / 5)) ^ (3 / 2)) / ((a ^ (4 / 1 / 5)) ^ 3) ) /
  ( ((a * (a ^ (2 / 3) * b ^ (1 / 3))) ^ (1 / 2)) ^ 4) * 
  (a ^ (1 / 4) * b ^ (1 / 8)) ^ 6 = 1 / ((a ^ (2 / 12)) * (b ^ (1 / 12))) :=
by
  sorry

end simplify_expression_l225_225087


namespace find_sum_l225_225918

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem find_sum (h₁ : a * b = 2 * (a + b))
                (h₂ : b * c = 3 * (b + c))
                (h₃ : c * a = 4 * (a + c))
                (ha : a ≠ 0)
                (hb : b ≠ 0)
                (hc : c ≠ 0) 
                : a + b + c = 1128 / 35 :=
by
  sorry

end find_sum_l225_225918


namespace greatest_three_digit_number_l225_225027

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l225_225027


namespace exists_ab_odd_n_exists_ab_odd_n_gt3_l225_225135

-- Define the required conditions
def gcd_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define a helper function to identify odd positive integers
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem exists_ab_odd_n (n : ℕ) (h : is_odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n :=
sorry

theorem exists_ab_odd_n_gt3 (n : ℕ) (h1 : is_odd n) (h2 : n > 3) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n ∧ n ∣ (a - b) = false :=
sorry

end exists_ab_odd_n_exists_ab_odd_n_gt3_l225_225135


namespace cranes_in_each_flock_l225_225958

theorem cranes_in_each_flock (c : ℕ) (h1 : ∃ n : ℕ, 13 * n = 221)
  (h2 : ∃ n : ℕ, c * n = 221) :
  c = 221 :=
by sorry

end cranes_in_each_flock_l225_225958


namespace operation_hash_12_6_l225_225329

axiom operation_hash (r s : ℝ) : ℝ

-- Conditions
axiom condition_1 : ∀ r : ℝ, operation_hash r 0 = r
axiom condition_2 : ∀ r s : ℝ, operation_hash r s = operation_hash s r
axiom condition_3 : ∀ r s : ℝ, operation_hash (r + 2) s = (operation_hash r s) + 2 * s + 2

-- Proof statement
theorem operation_hash_12_6 : operation_hash 12 6 = 168 :=
by
  sorry

end operation_hash_12_6_l225_225329


namespace paving_cost_l225_225435

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 1000
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost :
  cost = 20625 := by sorry

end paving_cost_l225_225435


namespace percentage_problem_l225_225374

variable (x : ℝ)

theorem percentage_problem (h : 0.4 * x = 160) : 240 / x = 0.6 :=
by sorry

end percentage_problem_l225_225374


namespace expected_value_twelve_sided_die_l225_225423

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l225_225423


namespace number_of_blue_fish_l225_225411

def total_fish : ℕ := 22
def goldfish : ℕ := 15
def blue_fish : ℕ := total_fish - goldfish

theorem number_of_blue_fish : blue_fish = 7 :=
by
  -- proof goes here
  sorry

end number_of_blue_fish_l225_225411


namespace find_q_l225_225324

variable (p q : ℝ)

theorem find_q (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end find_q_l225_225324


namespace quadrilateral_area_l225_225111

theorem quadrilateral_area 
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P = (1, 1 / 4))
  (focus : ℝ × ℝ) (hfocus : focus = (0, 1))
  (directrix : ℝ → Prop) (hdirectrix : ∀ y, directrix y ↔ y = 1)
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (M : ℝ × ℝ) (hM : M = (0, 1))
  (Q : ℝ × ℝ) 
  (PQ : ℝ)
  (area : ℝ) 
  (harea : area = 13 / 8) :
  ∃ (PQMF : ℝ), PQMF = 13 / 8 :=
sorry

end quadrilateral_area_l225_225111


namespace avg_visitors_is_correct_l225_225196

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average number of visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Define the number of Sundays in the month
def sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors on Sundays
def total_visitors_sundays : ℕ := sundays_in_month * avg_visitors_sunday

-- Define the total visitors on other days
def total_visitors_other_days : ℕ := other_days_in_month * avg_visitors_other_days

-- Define the total number of visitors in the month
def total_visitors : ℕ := total_visitors_sundays + total_visitors_other_days

-- Define the average number of visitors per day
def avg_visitors_per_day : ℕ := total_visitors / days_in_month

-- The theorem to prove
theorem avg_visitors_is_correct : avg_visitors_per_day = 276 := by
  sorry

end avg_visitors_is_correct_l225_225196


namespace sequence_is_constant_l225_225537

theorem sequence_is_constant
  (a : ℕ+ → ℝ)
  (S : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, S n + S (n + 1) = a (n + 1))
  : ∀ n : ℕ+, a n = 0 :=
by
  sorry

end sequence_is_constant_l225_225537


namespace isabella_hair_length_l225_225479

theorem isabella_hair_length (h : ℕ) (g : h + 4 = 22) : h = 18 := by
  sorry

end isabella_hair_length_l225_225479


namespace complement_intersection_l225_225009

universe u

def U : Finset Int := {-3, -2, -1, 0, 1}
def A : Finset Int := {-2, -1}
def B : Finset Int := {-3, -1, 0}

def complement_U (A : Finset Int) (U : Finset Int) : Finset Int :=
  U.filter (λ x => x ∉ A)

theorem complement_intersection :
  (complement_U A U) ∩ B = {-3, 0} :=
by
  sorry

end complement_intersection_l225_225009


namespace product_of_solutions_l225_225078

theorem product_of_solutions : (∃ x : ℝ, |x| = 3*(|x| - 2)) → (x = 3 ∨ x = -3) → 3 * -3 = -9 :=
by sorry

end product_of_solutions_l225_225078


namespace anthony_ate_total_l225_225699

def slices := 16

def ate_alone := 1 / slices
def shared_with_ben := (1 / 2) * (1 / slices)
def shared_with_chris := (1 / 2) * (1 / slices)

theorem anthony_ate_total :
  ate_alone + shared_with_ben + shared_with_chris = 1 / 8 :=
by
  sorry

end anthony_ate_total_l225_225699


namespace solve_inequality_1_range_of_m_l225_225179

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 3) + m

theorem solve_inequality_1 : {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} := sorry

theorem range_of_m (m : ℝ) (h : m > 4) : ∃ x : ℝ, f x < g x m := sorry

end solve_inequality_1_range_of_m_l225_225179


namespace ratio_of_cards_lost_l225_225779

-- Definitions based on the conditions
def purchases_per_week : ℕ := 20
def weeks_per_year : ℕ := 52
def cards_left : ℕ := 520

-- Main statement to be proved
theorem ratio_of_cards_lost (total_cards : ℕ := purchases_per_week * weeks_per_year)
                            (cards_lost : ℕ := total_cards - cards_left) :
                            (cards_lost : ℚ) / total_cards = 1 / 2 :=
by
  sorry

end ratio_of_cards_lost_l225_225779


namespace solve_for_x_l225_225241

theorem solve_for_x (x : ℝ) (h : (15 - 2 + (x / 1)) / 2 * 8 = 77) : x = 6.25 :=
by
  sorry

end solve_for_x_l225_225241


namespace period_cosine_l225_225354

noncomputable def period_of_cosine_function : ℝ := 2 * Real.pi / 3

theorem period_cosine (x : ℝ) : ∃ T, ∀ x, Real.cos (3 * x - Real.pi) = Real.cos (3 * (x + T) - Real.pi) :=
  ⟨period_of_cosine_function, by sorry⟩

end period_cosine_l225_225354


namespace expensive_time_8_l225_225850

variable (x : ℝ) -- x represents the time to pick an expensive handcuff lock

-- Conditions
def cheap_time := 6
def total_time := 42
def cheap_pairs := 3
def expensive_pairs := 3

-- Total time for cheap handcuffs
def total_cheap_time := cheap_pairs * cheap_time

-- Total time for expensive handcuffs
def total_expensive_time := total_time - total_cheap_time

-- Equation relating x to total_expensive_time
def expensive_equation := expensive_pairs * x = total_expensive_time

-- Proof goal
theorem expensive_time_8 : expensive_equation x -> x = 8 := by
  sorry

end expensive_time_8_l225_225850


namespace percentage_of_ginger_is_correct_l225_225580

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end percentage_of_ginger_is_correct_l225_225580


namespace find_number_satisfy_equation_l225_225288

theorem find_number_satisfy_equation (x : ℝ) :
  9 - x / 7 * 5 + 10 = 13.285714285714286 ↔ x = -20 := sorry

end find_number_satisfy_equation_l225_225288


namespace power_half_mod_prime_l225_225738

-- Definitions of odd prime and coprime condition
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1
def coprime (a p : ℕ) : Prop := Nat.gcd a p = 1

-- Main statement
theorem power_half_mod_prime (p a : ℕ) (hp : is_odd_prime p) (ha : coprime a p) :
  a ^ ((p - 1) / 2) % p = 1 ∨ a ^ ((p - 1) / 2) % p = p - 1 := 
  sorry

end power_half_mod_prime_l225_225738


namespace equation_of_plane_l225_225218

noncomputable def parametric_form (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - s + 2 * t, 1 - 3 * s - t)

theorem equation_of_plane (x y z : ℝ) : 
  (∃ s t : ℝ, parametric_form s t = (x, y, z)) → 5 * x + 11 * y + 7 * z - 61 = 0 :=
by
  sorry

end equation_of_plane_l225_225218


namespace garden_dimensions_l225_225757

variable {w l x : ℝ}

-- Definition of the problem conditions
def garden_length_eq_three_times_width (w l : ℝ) : Prop := l = 3 * w
def combined_area_eq (w x : ℝ) : Prop := (w + 2 * x) * (3 * w + 2 * x) = 432
def walkway_area_eq (w x : ℝ) : Prop := 8 * w * x + 4 * x^2 = 108

-- The main theorem statement
theorem garden_dimensions (w l x : ℝ)
  (h1 : garden_length_eq_three_times_width w l)
  (h2 : combined_area_eq w x)
  (h3 : walkway_area_eq w x) :
  w = 6 * Real.sqrt 3 ∧ l = 18 * Real.sqrt 3 :=
sorry

end garden_dimensions_l225_225757


namespace range_of_a_l225_225052

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - 2 * a * x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_of_a_l225_225052


namespace units_digit_G1000_l225_225442

def Gn (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G1000 : (Gn 1000) % 10 = 2 :=
by sorry

end units_digit_G1000_l225_225442


namespace savings_calculation_l225_225639

theorem savings_calculation (income expenditure savings : ℕ) (ratio_income ratio_expenditure : ℕ)
  (h_ratio : ratio_income = 10) (h_ratio2 : ratio_expenditure = 7) (h_income : income = 10000)
  (h_expenditure : 10 * expenditure = 7 * income) :
  savings = income - expenditure :=
by
  sorry

end savings_calculation_l225_225639


namespace minimize_abs_difference_and_product_l225_225937

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end minimize_abs_difference_and_product_l225_225937


namespace smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l225_225100

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f_is_pi : 
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ ε > 0, ε < Real.pi → ∃ x, f (x + ε) ≠ f x) :=
by
  sorry

theorem f_at_pi_over_2_not_sqrt_3_over_2 : f (Real.pi / 2) ≠ Real.sqrt 3 / 2 :=
by
  sorry

theorem max_value_of_f_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 1 :=
by
  sorry

end smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l225_225100


namespace flower_pattern_perimeter_l225_225873

theorem flower_pattern_perimeter (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 45) : 
    let arc_length := (360 - θ) / 360 * 2 * π * r
    let total_perimeter := arc_length + 2 * r
    total_perimeter = (21 / 4 * π) + 6 := 
by
  -- Definitions from conditions
  let arc_length := (360 - θ) / 360 * 2 * π * r
  let total_perimeter := arc_length + 2 * r

  -- Assertions to reach the target conclusion
  have h_arc_length: arc_length = (21 / 4 * π) :=
    by
      sorry

  -- Incorporate the radius
  have h_total: total_perimeter = (21 / 4 * π) + 6 :=
    by
      sorry

  exact h_total

end flower_pattern_perimeter_l225_225873


namespace remainder_when_15_plus_y_div_31_l225_225359

theorem remainder_when_15_plus_y_div_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + y) % 31 = 24 :=
by
  sorry

end remainder_when_15_plus_y_div_31_l225_225359


namespace range_of_a_fixed_point_l225_225322

open Function

def f (x a : ℝ) := x^3 - a * x

theorem range_of_a (a : ℝ) (h1 : 0 < a) : 0 < a ∧ a ≤ 3 ↔ ∀ x ≥ 1, 3 * x^2 - a > 0 :=
sorry

theorem fixed_point (a x0 : ℝ) (h_a : 0 < a) (h_b : a ≤ 3)
  (h1 : x0 ≥ 1) (h2 : f x0 a ≥ 1) (h3 : f (f x0 a) a = x0) (strict_incr : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f x a < f y a) :
  f x0 a = x0 :=
sorry

end range_of_a_fixed_point_l225_225322


namespace solve_equation_l225_225099

theorem solve_equation (x y : ℝ) : 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end solve_equation_l225_225099


namespace fraction_addition_l225_225301

theorem fraction_addition : (3 / 4 : ℚ) + (5 / 6) = 19 / 12 :=
by
  sorry

end fraction_addition_l225_225301


namespace arithmetic_sequence_problem_l225_225630

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 = 15)
  (h2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 :=
sorry

end arithmetic_sequence_problem_l225_225630


namespace four_x_plus_t_odd_l225_225955

theorem four_x_plus_t_odd (x t : ℤ) (hx : 2 * x - t = 11) : ¬(∃ n : ℤ, 4 * x + t = 2 * n) :=
by
  -- Since we need to prove the statement, we start a proof block
  sorry -- skipping the actual proof part for this statement

end four_x_plus_t_odd_l225_225955


namespace positive_number_is_25_l225_225134

theorem positive_number_is_25 {a x : ℝ}
(h1 : x = (3 * a + 1)^2)
(h2 : x = (-a - 3)^2)
(h_sum : 3 * a + 1 + (-a - 3) = 0) :
x = 25 :=
sorry

end positive_number_is_25_l225_225134


namespace perimeter_triangle_ABC_is_correct_l225_225482

noncomputable def semicircle_perimeter_trianlge_ABC : ℝ :=
  let BE := (1 : ℝ)
  let EF := (24 : ℝ)
  let FC := (3 : ℝ)
  let BC := BE + EF + FC
  let r := EF / 2
  let x := 71.5
  let AB := x + BE
  let AC := x + FC
  AB + BC + AC

theorem perimeter_triangle_ABC_is_correct : semicircle_perimeter_trianlge_ABC = 175 := by
  sorry

end perimeter_triangle_ABC_is_correct_l225_225482


namespace dog_speed_correct_l225_225264

-- Definitions of the conditions
def football_field_length_yards : ℕ := 200
def total_football_fields : ℕ := 6
def yards_to_feet_conversion : ℕ := 3
def time_to_fetch_minutes : ℕ := 9

-- The goal is to find the dog's speed in feet per minute
def dog_speed_feet_per_minute : ℕ :=
  (total_football_fields * football_field_length_yards * yards_to_feet_conversion) / time_to_fetch_minutes

-- Statement for the proof
theorem dog_speed_correct : dog_speed_feet_per_minute = 400 := by
  sorry

end dog_speed_correct_l225_225264


namespace reciprocal_eq_self_is_one_or_neg_one_l225_225968

/-- If a rational number equals its own reciprocal, then the number is either 1 or -1. -/
theorem reciprocal_eq_self_is_one_or_neg_one (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 := 
by
  sorry

end reciprocal_eq_self_is_one_or_neg_one_l225_225968


namespace parabola_equation_line_tangent_to_fixed_circle_l225_225146

open Real

def parabola_vertex_origin_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -2

def point_on_directrix (l: ℝ) (t : ℝ) : Prop :=
  t ≠ 0 ∧ l = 3 * t - 1 / t

def point_on_y_axis (q : ℝ) (t : ℝ) : Prop :=
  q = 2 * t

theorem parabola_equation (p : ℝ) : 
  parabola_vertex_origin_directrix 4 →
  y^2 = 8 * x :=
by
  sorry

theorem line_tangent_to_fixed_circle (t : ℝ) (x0 : ℝ) (r : ℝ) :
  t ≠ 0 →
  point_on_directrix (-2) t →
  point_on_y_axis (2 * t) t →
  (x0 = 2 ∧ r = 2) →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
by
  sorry

end parabola_equation_line_tangent_to_fixed_circle_l225_225146


namespace intersection_in_fourth_quadrant_l225_225202

theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ (-6 < k) ∧ (k < -2) :=
by
  sorry

end intersection_in_fourth_quadrant_l225_225202


namespace school_total_payment_l225_225487

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l225_225487


namespace range_of_x_l225_225998

theorem range_of_x (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x / (Real.sqrt (x + 2)) :=
by {
  sorry
}

end range_of_x_l225_225998


namespace sum_of_numbers_with_lcm_and_ratio_l225_225089

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) (h_lcm : Nat.lcm a b = 60) (h_ratio : a = 2 * b / 3) : a + b = 50 := 
by
  sorry

end sum_of_numbers_with_lcm_and_ratio_l225_225089


namespace outlinedSquareDigit_l225_225236

-- We define the conditions for three-digit powers of 2 and 3
def isThreeDigitPowerOf (base : ℕ) (n : ℕ) : Prop :=
  let power := base ^ n
  power >= 100 ∧ power < 1000

-- Define the sets of three-digit powers of 2 and 3
def threeDigitPowersOf2 : List ℕ := [128, 256, 512]
def threeDigitPowersOf3 : List ℕ := [243, 729]

-- Define the condition that the digit in the outlined square should be common as a last digit in any power of 2 and 3 that's three-digit long
def commonLastDigitOfPowers (a b : List ℕ) : Option ℕ :=
  let aLastDigits := a.map (λ x => x % 10)
  let bLastDigits := b.map (λ x => x % 10)
  (aLastDigits.inter bLastDigits).head?

theorem outlinedSquareDigit : (commonLastDigitOfPowers threeDigitPowersOf2 threeDigitPowersOf3) = some 3 :=
by
  sorry

end outlinedSquareDigit_l225_225236


namespace seashells_given_to_brothers_l225_225495

theorem seashells_given_to_brothers :
  ∃ B : ℕ, 180 - 40 - B = 2 * 55 ∧ B = 30 := by
  sorry

end seashells_given_to_brothers_l225_225495


namespace largest_possible_average_l225_225612

noncomputable def ten_test_scores (a b c d e f g h i j : ℤ) : ℤ :=
  a + b + c + d + e + f + g + h + i + j

theorem largest_possible_average
  (a b c d e f g h i j : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 100)
  (h2 : 0 ≤ b ∧ b ≤ 100)
  (h3 : 0 ≤ c ∧ c ≤ 100)
  (h4 : 0 ≤ d ∧ d ≤ 100)
  (h5 : 0 ≤ e ∧ e ≤ 100)
  (h6 : 0 ≤ f ∧ f ≤ 100)
  (h7 : 0 ≤ g ∧ g ≤ 100)
  (h8 : 0 ≤ h ∧ h ≤ 100)
  (h9 : 0 ≤ i ∧ i ≤ 100)
  (h10 : 0 ≤ j ∧ j ≤ 100)
  (h11 : a + b + c + d ≤ 190)
  (h12 : b + c + d + e ≤ 190)
  (h13 : c + d + e + f ≤ 190)
  (h14 : d + e + f + g ≤ 190)
  (h15 : e + f + g + h ≤ 190)
  (h16 : f + g + h + i ≤ 190)
  (h17 : g + h + i + j ≤ 190)
  : ((ten_test_scores a b c d e f g h i j : ℚ) / 10) ≤ 44.33 := sorry

end largest_possible_average_l225_225612


namespace find_point_B_coordinates_l225_225254

theorem find_point_B_coordinates (a : ℝ) : 
  (∀ (x y : ℝ), x^2 - 4*x + y^2 = 0 → (x - a)^2 + y^2 = 4 * ((x - 1)^2 + y^2)) →
  a = -2 :=
by
  sorry

end find_point_B_coordinates_l225_225254


namespace radius_of_circumscribed_sphere_l225_225689

-- Condition: SA = 2
def SA : ℝ := 2

-- Condition: SB = 4
def SB : ℝ := 4

-- Condition: SC = 4
def SC : ℝ := 4

-- Condition: The three side edges are pairwise perpendicular.
def pairwise_perpendicular : Prop := true -- This condition is described but would require geometric definition.

-- To prove: Radius of circumscribed sphere is 3
theorem radius_of_circumscribed_sphere : 
  ∀ (SA SB SC : ℝ) (pairwise_perpendicular : Prop), SA = 2 → SB = 4 → SC = 4 → pairwise_perpendicular → 
  (3 : ℝ) = 3 := by 
  intros SA SB SC pairwise_perpendicular h1 h2 h3 h4
  sorry

end radius_of_circumscribed_sphere_l225_225689


namespace inequality_for_positive_reals_l225_225617

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ ((a + b + c) ^ 2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a))) :=
by
  sorry

end inequality_for_positive_reals_l225_225617


namespace unique_solution_l225_225949

theorem unique_solution (n : ℕ) (h1 : n > 0) (h2 : n^2 ∣ 3^n + 1) : n = 1 :=
sorry

end unique_solution_l225_225949


namespace original_length_of_tape_l225_225851

-- Given conditions
variables (L : Real) (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
          (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4)

-- The theorem to prove
theorem original_length_of_tape (L : Real) 
  (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
  (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4) :
  L = 7.5 :=
by
  sorry

end original_length_of_tape_l225_225851


namespace totalMilkConsumption_l225_225068

-- Conditions
def regularMilk (week: ℕ) : ℝ := 0.5
def soyMilk (week: ℕ) : ℝ := 0.1

-- Theorem statement
theorem totalMilkConsumption : regularMilk 1 + soyMilk 1 = 0.6 := 
by 
  sorry

end totalMilkConsumption_l225_225068


namespace min_third_side_of_right_triangle_l225_225425

theorem min_third_side_of_right_triangle (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ c : ℕ, (min c (4 + 5 - 3) - (4 - 3)) = 3 :=
sorry

end min_third_side_of_right_triangle_l225_225425


namespace total_cost_for_photos_l225_225456

def total_cost (n : ℕ) (f : ℝ) (c : ℝ) : ℝ :=
  f + (n - 4) * c

theorem total_cost_for_photos :
  total_cost 54 24.5 2.3 = 139.5 :=
by
  sorry

end total_cost_for_photos_l225_225456


namespace g_zero_g_one_l225_225344

variable (g : ℤ → ℤ)

axiom condition1 (x : ℤ) : g (x + 5) - g x = 10 * x + 30
axiom condition2 (x : ℤ) : g (x^2 - 2) = (g x - x)^2 + x^2 - 4

theorem g_zero_g_one : (g 0, g 1) = (-4, 1) := 
by 
  sorry

end g_zero_g_one_l225_225344


namespace father_age_l225_225464

variable (F S x : ℕ)

-- Conditions
axiom h1 : F + S = 75
axiom h2 : F = 8 * (S - x)
axiom h3 : F - x = S

-- Theorem to prove
theorem father_age : F = 48 :=
sorry

end father_age_l225_225464


namespace rectangle_area_l225_225404

variable (w l A P : ℝ)
variable (h1 : l = w + 6)
variable (h2 : A = w * l)
variable (h3 : P = 2 * (w + l))
variable (h4 : A = 2 * P)
variable (h5 : w = 3)

theorem rectangle_area
  (w l A P : ℝ)
  (h1 : l = w + 6)
  (h2 : A = w * l)
  (h3 : P = 2 * (w + l))
  (h4 : A = 2 * P)
  (h5 : w = 3) :
  A = 27 := 
sorry

end rectangle_area_l225_225404


namespace product_of_differences_l225_225899

theorem product_of_differences (p q p' q' α β α' β' : ℝ)
  (h1 : α + β = -p) (h2 : α * β = q)
  (h3 : α' + β' = -p') (h4 : α' * β' = q') :
  ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q' * p - p' * q)) :=
sorry

end product_of_differences_l225_225899


namespace xy_value_l225_225020

structure Point (R : Type) := (x : R) (y : R)

def A : Point ℝ := ⟨2, 7⟩ 
def C : Point ℝ := ⟨4, 3⟩ 

def is_midpoint (A B C : Point ℝ) : Prop :=
  (C.x = (A.x + B.x) / 2) ∧ (C.y = (A.y + B.y) / 2)

theorem xy_value (x y : ℝ) (B : Point ℝ := ⟨x, y⟩) (H : is_midpoint A B C) :
  x * y = -6 := 
sorry

end xy_value_l225_225020


namespace problem_1_problem_2_problem_3_l225_225670

def range_1 : Set ℝ :=
  { y | ∃ x : ℝ, y = 1 / (x - 1) ∧ x ≠ 1 }

def range_2 : Set ℝ :=
  { y | ∃ x : ℝ, y = x^2 + 4 * x - 1 }

def range_3 : Set ℝ :=
  { y | ∃ x : ℝ, y = x + Real.sqrt (x + 1) ∧ x ≥ 0 }

theorem problem_1 : range_1 = {y | y < 0 ∨ y > 0} :=
by 
  sorry

theorem problem_2 : range_2 = {y | y ≥ -5} :=
by 
  sorry

theorem problem_3 : range_3 = {y | y ≥ -1} :=
by 
  sorry

end problem_1_problem_2_problem_3_l225_225670


namespace other_number_is_300_l225_225629

theorem other_number_is_300 (A B : ℕ) (h1 : A = 231) (h2 : lcm A B = 2310) (h3 : gcd A B = 30) : B = 300 := by
  sorry

end other_number_is_300_l225_225629


namespace new_mixture_alcohol_percentage_l225_225152

/-- 
Given: 
  - a solution with 15 liters containing 26% alcohol
  - 5 liters of water added to the solution
Prove:
  The percentage of alcohol in the new mixture is 19.5%
-/
theorem new_mixture_alcohol_percentage 
  (original_volume : ℝ) (original_percent_alcohol : ℝ) (added_water_volume : ℝ) :
  original_volume = 15 → 
  original_percent_alcohol = 26 →
  added_water_volume = 5 →
  (original_volume * (original_percent_alcohol / 100) / (original_volume + added_water_volume)) * 100 = 19.5 :=
by 
  intros h1 h2 h3
  sorry

end new_mixture_alcohol_percentage_l225_225152


namespace find_y_l225_225806

theorem find_y (y : ℝ) (h : (8 + 15 + 22 + 5 + y) / 5 = 12) : y = 10 :=
by
  -- the proof is skipped
  sorry

end find_y_l225_225806


namespace initial_bags_of_rice_l225_225016

theorem initial_bags_of_rice (sold restocked final initial : Int) 
  (h1 : sold = 23)
  (h2 : restocked = 132)
  (h3 : final = 164) 
  : ((initial - sold) + restocked = final) ↔ initial = 55 :=
by 
  have eq1 : ((initial - sold) + restocked = final) ↔ initial - 23 + 132 = 164 := by rw [h1, h2, h3]
  simp [eq1]
  sorry

end initial_bags_of_rice_l225_225016


namespace total_value_correct_l225_225911

-- Define conditions
def import_tax_rate : ℝ := 0.07
def tax_paid : ℝ := 109.90
def tax_exempt_value : ℝ := 1000

-- Define total value
def total_value (V : ℝ) : Prop :=
  V - tax_exempt_value = tax_paid / import_tax_rate

-- Theorem stating that the total value is $2570
theorem total_value_correct : total_value 2570 := by
  sorry

end total_value_correct_l225_225911


namespace vanessa_points_l225_225801

theorem vanessa_points (total_points : ℕ) (num_other_players : ℕ) (avg_points_other : ℕ) 
  (h1 : total_points = 65) (h2 : num_other_players = 7) (h3 : avg_points_other = 5) :
  ∃ vp : ℕ, vp = 30 :=
by
  sorry

end vanessa_points_l225_225801


namespace stuffed_animals_total_l225_225933

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l225_225933


namespace circle_radius_l225_225865

theorem circle_radius (x y : ℝ) : x^2 + 8*x + y^2 - 10*y + 32 = 0 → ∃ r : ℝ, r = 3 :=
by
  sorry

end circle_radius_l225_225865


namespace smallest_whole_number_l225_225932

theorem smallest_whole_number (a b c d : ℤ)
  (h₁ : a = 3 + 1 / 3)
  (h₂ : b = 4 + 1 / 4)
  (h₃ : c = 5 + 1 / 6)
  (h₄ : d = 6 + 1 / 8)
  (h₅ : a + b + c + d - 2 > 16)
  (h₆ : a + b + c + d - 2 < 17) :
  17 > 16 + (a + b + c + d - 18) - 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 8 :=
  sorry

end smallest_whole_number_l225_225932


namespace unique_hexagon_angles_sides_identity_1_identity_2_l225_225781

noncomputable def lengths_angles_determined 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : Prop :=
  -- Assuming this is the expression we need to handle:
  ∀ (δ ε ζ : ℝ),
    δ = 180 - α ∧
    ε = 180 - β ∧
    ζ = 180 - γ →
  ∃ (angles_determined : Prop),
    angles_determined

theorem unique_hexagon_angles_sides 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : 
  lengths_angles_determined a b c d e f α β γ h₀ :=
sorry

theorem identity_1 
  (a b c d : ℝ) 
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  (1 / a + 1 / c = 1 / b + 1 / d) :=
sorry

theorem identity_2 
  (a b c d e f : ℝ) 
  (h₀ : true) : 
  ((a + f) * (b + d) * (c + e) = (a + e) * (b + f) * (c + d)) :=
sorry

end unique_hexagon_angles_sides_identity_1_identity_2_l225_225781


namespace second_printer_cost_l225_225103

theorem second_printer_cost (p1_cost : ℕ) (num_units : ℕ) (total_spent : ℕ) (x : ℕ) 
  (h1 : p1_cost = 375) 
  (h2 : num_units = 7) 
  (h3 : total_spent = p1_cost * num_units) 
  (h4 : total_spent = x * num_units) : 
  x = 375 := 
sorry

end second_printer_cost_l225_225103


namespace gain_percentage_l225_225023

theorem gain_percentage (selling_price gain : ℝ) (h1 : selling_price = 225) (h2 : gain = 75) : 
  (gain / (selling_price - gain) * 100) = 50 :=
by
  sorry

end gain_percentage_l225_225023


namespace root_exists_between_a_and_b_l225_225362

variable {α : Type*} [LinearOrderedField α]

theorem root_exists_between_a_and_b (a b p q : α) (h₁ : a^2 + p * a + q = 0) (h₂ : b^2 - p * b - q = 0) (h₃ : q ≠ 0) :
  ∃ c, a < c ∧ c < b ∧ (c^2 + 2 * p * c + 2 * q = 0) := by
  sorry

end root_exists_between_a_and_b_l225_225362


namespace double_inequality_pos_reals_equality_condition_l225_225649

theorem double_inequality_pos_reals (x y z : ℝ) (x_pos: 0 < x) (y_pos: 0 < y) (z_pos: 0 < z):
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ (1 / 8) :=
  sorry

theorem equality_condition (x y z : ℝ) :
  ((1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) = (1 / 8)) ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
  sorry

end double_inequality_pos_reals_equality_condition_l225_225649


namespace complement_B_intersection_A_complement_B_l225_225352

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x < 0}
noncomputable def B : Set ℝ := {x | x > 1}

theorem complement_B :
  (U \ B) = {x | x ≤ 1} := by
  sorry

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | x < 0} := by
  sorry

end complement_B_intersection_A_complement_B_l225_225352


namespace part1_part2_l225_225628

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem part1 (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : |f a x| ≤ 5/4 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x ∈ Set.Icc (-1:ℝ) (1:ℝ), f a x = 17/8) : a = -2 :=
by
  sorry

end part1_part2_l225_225628


namespace derivative_at_pi_over_six_l225_225661

-- Define the function f(x) = cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem: the derivative of f at π/6 is -1/2
theorem derivative_at_pi_over_six : deriv f (Real.pi / 6) = -1 / 2 :=
by sorry

end derivative_at_pi_over_six_l225_225661


namespace friends_in_group_l225_225144

theorem friends_in_group : 
  ∀ (total_chicken_wings cooked_wings additional_wings chicken_wings_per_person : ℕ), 
    cooked_wings = 8 →
    additional_wings = 10 →
    chicken_wings_per_person = 6 →
    total_chicken_wings = cooked_wings + additional_wings →
    total_chicken_wings / chicken_wings_per_person = 3 :=
by
  intros total_chicken_wings cooked_wings additional_wings chicken_wings_per_person hcooked hadditional hperson htotal
  sorry

end friends_in_group_l225_225144


namespace system1_solution_system2_solution_l225_225338

-- System 1 Definitions
def eq1 (x y : ℝ) : Prop := 3 * x - 2 * y = 9
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y = 19

-- System 2 Definitions
def eq3 (x y : ℝ) : Prop := (2 * x + 1) / 5 - 1 = (y - 1) / 3
def eq4 (x y : ℝ) : Prop := 2 * (y - x) - 3 * (1 - y) = 6

-- Theorem Statements
theorem system1_solution (x y : ℝ) : eq1 x y ∧ eq2 x y ↔ x = 5 ∧ y = 3 := by
  sorry

theorem system2_solution (x y : ℝ) : eq3 x y ∧ eq4 x y ↔ x = 4 ∧ y = 17 / 5 := by
  sorry

end system1_solution_system2_solution_l225_225338


namespace general_term_l225_225317

def S (n : ℕ) : ℤ := n^2 - 4*n

noncomputable def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 5) := by
  sorry

end general_term_l225_225317


namespace complex_equation_solution_l225_225555

theorem complex_equation_solution (x : ℝ) (i : ℂ) (h_imag_unit : i * i = -1) (h_eq : (x + 2 * i) * (x - i) = 6 + 2 * i) : x = 2 :=
by
  sorry

end complex_equation_solution_l225_225555


namespace solve_for_x_l225_225775

theorem solve_for_x (x : ℝ) (h : (x / 4) / 2 = 4 / (x / 2)) : x = 8 ∨ x = -8 :=
by
  sorry

end solve_for_x_l225_225775


namespace tenth_term_ar_sequence_l225_225006

-- Variables for the first term and common difference
variables (a1 d : ℕ) (n : ℕ)

-- Specific given values
def a1_fixed := 3
def d_fixed := 2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) := a1 + (n - 1) * d

-- The statement to prove
theorem tenth_term_ar_sequence : a_n 10 = 21 := by
  -- Definitions for a1 and d
  let a1 := a1_fixed
  let d := d_fixed
  -- The rest of the proof
  sorry

end tenth_term_ar_sequence_l225_225006


namespace correct_option_l225_225700

def monomial_structure_same (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

def monomial1 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 3ab^2
| 1 => 2 -- Exponent of b in 3ab^2
| _ => 0

def monomial2 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 4ab^2
| 1 => 2 -- Exponent of b in 4ab^2
| _ => 0

theorem correct_option :
  monomial_structure_same monomial1 monomial2 := sorry

end correct_option_l225_225700


namespace scalene_polygon_exists_l225_225897

theorem scalene_polygon_exists (n: ℕ) (a: Fin n → ℝ) (h: ∀ i, 1 ≤ a i ∧ a i ≤ 2013) (h_geq: n ≥ 13):
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ a A + a B > a C ∧ a A + a C > a B ∧ a B + a C > a A :=
sorry

end scalene_polygon_exists_l225_225897


namespace total_face_value_of_notes_l225_225742

theorem total_face_value_of_notes :
  let face_value := 5
  let number_of_notes := 440 * 10^6
  face_value * number_of_notes = 2200000000 := 
by
  sorry

end total_face_value_of_notes_l225_225742


namespace janet_overtime_multiple_l225_225029

theorem janet_overtime_multiple :
  let hourly_rate := 20
  let weekly_hours := 52
  let regular_hours := 40
  let car_price := 4640
  let weeks_needed := 4
  let normal_weekly_earning := regular_hours * hourly_rate
  let overtime_hours := weekly_hours - regular_hours
  let required_weekly_earning := car_price / weeks_needed
  let overtime_weekly_earning := required_weekly_earning - normal_weekly_earning
  let overtime_rate := overtime_weekly_earning / overtime_hours
  (overtime_rate / hourly_rate = 1.5) :=
by
  sorry

end janet_overtime_multiple_l225_225029


namespace firm_partners_l225_225071

theorem firm_partners
  (P A : ℕ)
  (h1 : P / A = 2 / 63)
  (h2 : P / (A + 35) = 1 / 34) :
  P = 14 :=
by
  sorry

end firm_partners_l225_225071


namespace min_max_x_l225_225106

-- Definitions for the initial conditions and surveys
def students : ℕ := 100
def like_math_initial : ℕ := 50
def dislike_math_initial : ℕ := 50
def like_math_final : ℕ := 60
def dislike_math_final : ℕ := 40

-- Variables for the students' responses
variables (a b c d : ℕ)

-- Conditions based on the problem statement
def initial_survey : Prop := a + d = like_math_initial ∧ b + c = dislike_math_initial
def final_survey : Prop := a + c = like_math_final ∧ b + d = dislike_math_final

-- Definition of x as the number of students who changed their answer
def x : ℕ := c + d

-- Prove the minimum and maximum value of x with given conditions
theorem min_max_x (a b c d : ℕ) 
  (initial_cond : initial_survey a b c d)
  (final_cond : final_survey a b c d)
  : 10 ≤ (x c d) ∧ (x c d) ≤ 90 :=
by
  -- This is where the proof would go, but we'll simply state sorry for now.
  sorry

end min_max_x_l225_225106


namespace hyperbola_eccentricity_range_l225_225619

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_B: ∃ A B : ℝ, x = -c ∧ |AF| = b^2 / a ∧ |CF| = a + c) :
  e > 2 :=
by
  sorry

end hyperbola_eccentricity_range_l225_225619


namespace solution_set_of_inequality_l225_225243

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x | 3 * a < x ∧ x < -a} :=
sorry

end solution_set_of_inequality_l225_225243


namespace numbers_sum_and_difference_l225_225173

variables (a b : ℝ)

theorem numbers_sum_and_difference (h : a / b = -1) : a + b = 0 ∧ (a - b = 2 * b ∨ a - b = -2 * b) :=
by {
  sorry
}

end numbers_sum_and_difference_l225_225173


namespace correct_relation_l225_225934

-- Define the set A
def A : Set ℤ := { x | x^2 - 4 = 0 }

-- The statement that 2 is an element of A
theorem correct_relation : 2 ∈ A :=
by 
    -- We skip the proof here
    sorry

end correct_relation_l225_225934


namespace car_speed_l225_225480

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l225_225480


namespace smallest_four_digit_congruent_one_mod_17_l225_225963

theorem smallest_four_digit_congruent_one_mod_17 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n % 17 = 1 ∧ n = 1003 :=
by
sorry

end smallest_four_digit_congruent_one_mod_17_l225_225963


namespace find_tangent_line_equation_l225_225837

noncomputable def tangent_line_equation (f : ℝ → ℝ) (perp_line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  let y₀ := f x₀
  let slope_perp_to_tangent := -2
  let slope_tangent := -1 / 2
  slope_perp_to_tangent = -1 / (deriv f x₀) ∧
  x₀ = 1 ∧ y₀ = 1 ∧
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3

theorem find_tangent_line_equation :
  tangent_line_equation (fun (x : ℝ) => Real.sqrt x) (fun (x : ℝ) => -2 * x - 4) 1 := by
  sorry

end find_tangent_line_equation_l225_225837


namespace ratio_ac_l225_225585

variable {a b c d : ℝ}

-- Given the conditions
axiom ratio_ab : a / b = 5 / 4
axiom ratio_cd : c / d = 4 / 3
axiom ratio_db : d / b = 1 / 5

-- The statement to prove
theorem ratio_ac : a / c = 75 / 16 :=
  by sorry

end ratio_ac_l225_225585


namespace probability_first_ge_second_l225_225375

-- Define the number of faces
def faces : ℕ := 10

-- Define the total number of outcomes excluding the duplicates
def total_outcomes : ℕ := faces * faces - faces

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ := 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_first_ge_second :
  probability = 11 / 18 :=
sorry

end probability_first_ge_second_l225_225375


namespace part1_part2_l225_225209

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 4 - abs (x - 4)} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 5} :=
by
  sorry

theorem part2 (set_is : {x : ℝ | 1 ≤ x ∧ x ≤ 2}) : 
  ∃ a : ℝ, 
    (∀ x : ℝ, abs (f (2*x + a) a - 2*f x a) ≤ 2 → (1 ≤ x ∧ x ≤ 2)) ∧ 
    a = 3 :=
by
  sorry

end part1_part2_l225_225209


namespace max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l225_225508

/-- Define the given function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

/-- The maximum value of the function f(x) is sqrt(2) -/
theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 := 
sorry

/-- The smallest positive period of the function f(x) -/
theorem smallest_positive_period_of_f :
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = Real.pi :=
sorry

/-- The set of values x that satisfy f(x) ≥ 1 -/
theorem values_of_x_satisfying_f_ge_1 :
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4 :=
sorry

end max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l225_225508


namespace remainder_when_sum_divided_by_30_l225_225336

theorem remainder_when_sum_divided_by_30 (x y z : ℕ) (hx : x % 30 = 14) (hy : y % 30 = 5) (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 :=
by
  sorry

end remainder_when_sum_divided_by_30_l225_225336


namespace minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l225_225151

theorem minimum_value_x_plus_four_over_x (x : ℝ) (h : x ≥ 2) : 
  x + 4 / x ≥ 4 :=
by sorry

theorem minimum_value_occurs_at_x_eq_2 : ∀ (x : ℝ), x ≥ 2 → (x + 4 / x = 4 ↔ x = 2) :=
by sorry

end minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l225_225151


namespace perimeter_of_polygon_is_15_l225_225431

-- Definitions for the problem conditions
def side_length_of_square : ℕ := 5
def fraction_of_square_occupied (n : ℕ) : ℚ := 3 / 4

-- Problem statement: Prove that the perimeter of the polygon is 15 units
theorem perimeter_of_polygon_is_15 :
  4 * side_length_of_square * (fraction_of_square_occupied side_length_of_square) = 15 := 
by
  sorry

end perimeter_of_polygon_is_15_l225_225431


namespace total_distance_is_correct_l225_225140

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l225_225140


namespace congruent_semicircles_ratio_l225_225363

theorem congruent_semicircles_ratio (N : ℕ) (r : ℝ) (hN : N > 0) 
    (A : ℝ) (B : ℝ) (hA : A = (N * π * r^2) / 2)
    (hB : B = (π * N^2 * r^2) / 2 - (N * π * r^2) / 2)
    (h_ratio : A / B = 1 / 9) : 
    N = 10 :=
by
  -- The proof will be filled in here.
  sorry

end congruent_semicircles_ratio_l225_225363


namespace sequence_not_generated_l225_225858

theorem sequence_not_generated (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (a 2 = 0) ∧ (a 3 = 2) ∧ (a 4 = 0) → 
  (∀ n, a n ≠ (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)) :=
by sorry

end sequence_not_generated_l225_225858


namespace third_median_length_l225_225800

theorem third_median_length (a b: ℝ) (h_a: a = 5) (h_b: b = 8)
  (area: ℝ) (h_area: area = 6 * Real.sqrt 15) (m: ℝ):
  m = 3 * Real.sqrt 6 :=
sorry

end third_median_length_l225_225800


namespace increasing_interval_implication_l225_225345

theorem increasing_interval_implication (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2) 2, (1 / x + 2 * a * x > 0)) → a > -1 / 8 :=
by
  intro h
  sorry

end increasing_interval_implication_l225_225345


namespace like_terms_exponents_l225_225085

theorem like_terms_exponents (m n : ℕ) (x y : ℝ) (h : 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) : m = 4 ∧ n = 3 :=
by 
  sorry

end like_terms_exponents_l225_225085


namespace total_surface_area_of_pyramid_l225_225547

noncomputable def base_length_ab : ℝ := 8 -- Length of side AB
noncomputable def base_length_ad : ℝ := 6 -- Length of side AD
noncomputable def height_pf : ℝ := 15 -- Perpendicular height from peak P to the base's center F

noncomputable def base_area : ℝ := base_length_ab * base_length_ad
noncomputable def fm_distance : ℝ := Real.sqrt ((base_length_ab / 2)^2 + (base_length_ad / 2)^2)
noncomputable def slant_height_pm : ℝ := Real.sqrt (height_pf^2 + fm_distance^2)

noncomputable def lateral_area_ab : ℝ := 2 * (0.5 * base_length_ab * slant_height_pm)
noncomputable def lateral_area_ad : ℝ := 2 * (0.5 * base_length_ad * slant_height_pm)
noncomputable def total_surface_area : ℝ := base_area + lateral_area_ab + lateral_area_ad

theorem total_surface_area_of_pyramid :
  total_surface_area = 48 + 55 * Real.sqrt 10 := by
  sorry

end total_surface_area_of_pyramid_l225_225547


namespace all_values_are_equal_l225_225654

theorem all_values_are_equal
  (f : ℤ × ℤ → ℕ)
  (h : ∀ x y : ℤ, f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1))
  (hf_pos : ∀ x y : ℤ, 0 < f (x, y)) : 
  ∀ x y x' y' : ℤ, f (x, y) = f (x', y') :=
by
  sorry

end all_values_are_equal_l225_225654


namespace min_sine_difference_l225_225658

theorem min_sine_difference (N : ℕ) (hN : 0 < N) :
  ∃ (n k : ℕ), (1 ≤ n ∧ n ≤ N + 1) ∧ (1 ≤ k ∧ k ≤ N + 1) ∧ (n ≠ k) ∧ 
    (|Real.sin n - Real.sin k| < 2 / N) := 
sorry

end min_sine_difference_l225_225658


namespace total_accessories_correct_l225_225665

-- Definitions
def dresses_first_period := 10 * 4
def dresses_second_period := 3 * 5
def total_dresses := dresses_first_period + dresses_second_period
def accessories_per_dress := 3 + 2 + 1
def total_accessories := total_dresses * accessories_per_dress

-- Theorem statement
theorem total_accessories_correct : total_accessories = 330 := by
  sorry

end total_accessories_correct_l225_225665


namespace property1_property2_l225_225326

/-- Given sequence a_n defined as a_n = 3(n^2 + n) + 7 -/
def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

/-- Property 1: Out of any five consecutive terms in the sequence, only one term is divisible by 5. -/
theorem property1 (n : ℕ) : (∃ k : ℕ, a (5 * k + 2) % 5 = 0) ∧ (∀ k : ℕ, ∀ r : ℕ, r ≠ 2 → a (5 * k + r) % 5 ≠ 0) :=
by
  sorry

/-- Property 2: None of the terms in this sequence is a cube of an integer. -/
theorem property2 (n : ℕ) : ¬(∃ t : ℕ, a n = t^3) :=
by
  sorry

end property1_property2_l225_225326


namespace inscribed_square_area_ratio_l225_225507

theorem inscribed_square_area_ratio (side_length : ℝ) (h_pos : side_length > 0) :
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  (inscribed_square_area / large_square_area) = (1 / 4) :=
by
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  sorry

end inscribed_square_area_ratio_l225_225507


namespace find_m_for_positive_integer_x_l225_225175

theorem find_m_for_positive_integer_x :
  ∃ (m : ℤ), (2 * m * x - 8 = (m + 2) * x) → ∀ (x : ℤ), x > 0 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 :=
sorry

end find_m_for_positive_integer_x_l225_225175


namespace fraction_torn_off_l225_225971

theorem fraction_torn_off (P: ℝ) (A_remaining: ℝ) (fraction: ℝ):
  P = 32 → 
  A_remaining = 48 → 
  fraction = 1 / 4 :=
by 
  sorry

end fraction_torn_off_l225_225971


namespace volume_of_sphere_in_cone_l225_225489

theorem volume_of_sphere_in_cone :
  let r_base := 9
  let h_cone := 9
  let diameter_sphere := 9 * Real.sqrt 2
  let radius_sphere := diameter_sphere / 2
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere ^ 3
  volume_sphere = (1458 * Real.sqrt 2 / 4) * Real.pi :=
by
  sorry

end volume_of_sphere_in_cone_l225_225489


namespace trees_total_count_l225_225706

theorem trees_total_count (D P : ℕ) 
  (h1 : D = 350 ∨ P = 350)
  (h2 : 300 * D + 225 * P = 217500) :
  D + P = 850 :=
by
  sorry

end trees_total_count_l225_225706


namespace present_age_of_A_l225_225528

theorem present_age_of_A (A B C : ℕ) 
  (h1 : A + B + C = 57)
  (h2 : B - 3 = 2 * (A - 3))
  (h3 : C - 3 = 3 * (A - 3)) :
  A = 11 :=
sorry

end present_age_of_A_l225_225528


namespace faith_change_l225_225780

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l225_225780


namespace lottery_numbers_bound_l225_225919

theorem lottery_numbers_bound (s : ℕ) (k : ℕ) (num_tickets : ℕ) (num_numbers : ℕ) (nums_per_ticket : ℕ)
  (h_tickets : num_tickets = 100) (h_numbers : num_numbers = 90) (h_nums_per_ticket : nums_per_ticket = 5)
  (h_s : s = num_tickets) (h_k : k = 49) :
  ∃ n : ℕ, n ≤ 10 :=
by
  sorry

end lottery_numbers_bound_l225_225919


namespace board_divisible_into_hexominos_l225_225759

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end board_divisible_into_hexominos_l225_225759


namespace cottage_cost_per_hour_l225_225285

-- Define the conditions
def jack_payment : ℝ := 20
def jill_payment : ℝ := 20
def total_payment : ℝ := jack_payment + jill_payment
def rental_duration : ℝ := 8

-- Define the theorem to be proved
theorem cottage_cost_per_hour : (total_payment / rental_duration) = 5 := by
  sorry

end cottage_cost_per_hour_l225_225285


namespace time_addition_and_sum_l225_225713

noncomputable def time_after_addition (hours_1 minutes_1 seconds_1 hours_2 minutes_2 seconds_2 : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := seconds_1 + seconds_2
  let extra_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := minutes_1 + minutes_2 + extra_minutes
  let extra_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := hours_1 + hours_2 + extra_hours
  let result_hours := total_hours % 12
  (result_hours, result_minutes, result_seconds)

theorem time_addition_and_sum :
  let current_hours := 3
  let current_minutes := 0
  let current_seconds := 0
  let add_hours := 300
  let add_minutes := 55
  let add_seconds := 30
  let (final_hours, final_minutes, final_seconds) := time_after_addition current_hours current_minutes current_seconds add_hours add_minutes add_seconds
  final_hours + final_minutes + final_seconds = 88 :=
by
  sorry

end time_addition_and_sum_l225_225713


namespace parallel_vectors_implies_x_l225_225458

-- a definition of the vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

-- a definition for vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- a definition for scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- a definition for vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- the theorem statement
theorem parallel_vectors_implies_x (x : ℝ) (h : 
  vector_add vector_a (vector_b x) = ⟨3, 1 + x⟩ ∧
  vector_sub (scalar_mul 2 vector_a) (vector_b x) = ⟨3, 2 - x⟩ ∧
  ∃ k : ℝ, vector_add vector_a (vector_b x) = scalar_mul k (vector_sub (scalar_mul 2 vector_a) (vector_b x))
  ) : x = 1 / 2 :=
sorry

end parallel_vectors_implies_x_l225_225458


namespace cost_per_tissue_box_l225_225855

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l225_225855


namespace roots_of_equation_l225_225516

theorem roots_of_equation {x : ℝ} :
  (12 * x^2 - 31 * x - 6 = 0) →
  (x = (31 + Real.sqrt 1249) / 24 ∨ x = (31 - Real.sqrt 1249) / 24) :=
by
  sorry

end roots_of_equation_l225_225516


namespace largest_smallest_difference_l225_225170

theorem largest_smallest_difference (a b c d : ℚ) (h₁ : a = 2.5) (h₂ : b = 22/13) (h₃ : c = 0.7) (h₄ : d = 32/33) :
  max (max a b) (max c d) - min (min a b) (min c d) = 1.8 := by
  sorry

end largest_smallest_difference_l225_225170


namespace al_original_amount_l225_225448

theorem al_original_amount : 
  ∃ (a b c : ℝ), 
    a + b + c = 1200 ∧ 
    (a - 200 + 3 * b + 4 * c) = 1800 ∧ 
    b = 2800 - 3 * a ∧ 
    c = 1200 - a - b ∧ 
    a = 860 := by
  sorry

end al_original_amount_l225_225448


namespace exists_unique_circle_l225_225294

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

def diametrically_opposite_points (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (cx, cy) := C.center
  let (px, py) := P
  (px - cx) ^ 2 + (py - cy) ^ 2 = (C.radius ^ 2)

def intersects_at_diametrically_opposite_points (K A : Circle) : Prop :=
  ∃ P₁ P₂ : ℝ × ℝ, diametrically_opposite_points A P₁ ∧ diametrically_opposite_points A P₂ ∧
  P₁ ≠ P₂ ∧ diametrically_opposite_points K P₁ ∧ diametrically_opposite_points K P₂

theorem exists_unique_circle (A B C : Circle) :
  ∃! K : Circle, intersects_at_diametrically_opposite_points K A ∧
  intersects_at_diametrically_opposite_points K B ∧
  intersects_at_diametrically_opposite_points K C := sorry

end exists_unique_circle_l225_225294


namespace snickers_cost_l225_225492

variable (S : ℝ)

def cost_of_snickers (n : ℝ) : Prop :=
  2 * n + 3 * (2 * n) = 12

theorem snickers_cost (h : cost_of_snickers S) : S = 1.50 :=
by
  sorry

end snickers_cost_l225_225492


namespace fixed_cost_calculation_l225_225931

theorem fixed_cost_calculation (TC MC n FC : ℕ) (h1 : TC = 16000) (h2 : MC = 200) (h3 : n = 20) (h4 : TC = FC + MC * n) : FC = 12000 :=
by
  sorry

end fixed_cost_calculation_l225_225931


namespace fraction_black_part_l225_225825

theorem fraction_black_part (L : ℝ) (blue_part : ℝ) (white_part_fraction : ℝ) 
  (h1 : L = 8) (h2 : blue_part = 3.5) (h3 : white_part_fraction = 0.5) : 
  (8 - (3.5 + 0.5 * (8 - 3.5))) / 8 = 9 / 32 :=
by
  sorry

end fraction_black_part_l225_225825


namespace number_of_students_l225_225460

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N) (h2 : (T - 250) / (N - 5) = 90) : N = 20 :=
sorry

end number_of_students_l225_225460


namespace solve_for_b_l225_225957

theorem solve_for_b (b : ℝ) (m : ℝ) (h : b > 0)
  (h1 : ∀ x : ℝ, x^2 + b * x + 54 = (x + m) ^ 2 + 18) : b = 12 :=
by
  sorry

end solve_for_b_l225_225957


namespace cube_volume_l225_225880

theorem cube_volume (s : ℝ) (h1 : 6 * s^2 = 1734) : s^3 = 4913 := by
  sorry

end cube_volume_l225_225880


namespace marked_elements_duplicate_l225_225074

open Nat

def table : Matrix (Fin 4) (Fin 10) ℕ := ![
  ![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
  ![9, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
  ![8, 9, 0, 1, 2, 3, 4, 5, 6, 7], 
  ![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
]

theorem marked_elements_duplicate 
  (marked : Fin 4 → Fin 10) 
  (h_marked_unique_row : ∀ i1 i2, i1 ≠ i2 → marked i1 ≠ marked i2)
  (h_marked_unique_col : ∀ j, ∃ i, marked i = j) :
  ∃ i1 i2, i1 ≠ i2 ∧ table i1 (marked i1) = table i2 (marked i2) := sorry

end marked_elements_duplicate_l225_225074


namespace shift_down_two_units_l225_225387

def original_function (x : ℝ) : ℝ := 2 * x + 1

def shifted_function (x : ℝ) : ℝ := original_function x - 2

theorem shift_down_two_units :
  ∀ x : ℝ, shifted_function x = 2 * x - 1 :=
by 
  intros x
  simp [shifted_function, original_function]
  sorry

end shift_down_two_units_l225_225387


namespace root_inverse_cubes_l225_225752

theorem root_inverse_cubes (a b c r s : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) :
  (1 / r^3) + (1 / s^3) = (-b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end root_inverse_cubes_l225_225752


namespace canoes_to_kayaks_ratio_l225_225333

theorem canoes_to_kayaks_ratio
  (canoe_cost kayak_cost total_revenue canoes_more_than_kayaks : ℕ)
  (H1 : canoe_cost = 14)
  (H2 : kayak_cost = 15)
  (H3 : total_revenue = 288)
  (H4 : ∃ C K : ℕ, C = K + canoes_more_than_kayaks ∧ 14 * C + 15 * K = 288) :
  ∃ (r : ℚ), r = 3 / 2 := by
  sorry

end canoes_to_kayaks_ratio_l225_225333


namespace room_length_l225_225935

theorem room_length (L : ℝ) (width height door_area window_area cost_per_sq_ft total_cost : ℝ) 
    (num_windows : ℕ) (door_w window_w door_h window_h : ℝ)
    (h_width : width = 15) (h_height : height = 12) 
    (h_cost_per_sq_ft : cost_per_sq_ft = 9)
    (h_door_area : door_area = door_w * door_h)
    (h_window_area : window_area = window_w * window_h)
    (h_num_windows : num_windows = 3)
    (h_door_dim : door_w = 6 ∧ door_h = 3)
    (h_window_dim : window_w = 4 ∧ window_h = 3)
    (h_total_cost : total_cost = 8154) :
    (2 * height * (L + width) - (door_area + num_windows * window_area)) * cost_per_sq_ft = total_cost →
    L = 25 := 
by
  intros h_cost_eq
  sorry

end room_length_l225_225935


namespace vector_addition_simplification_l225_225439

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_simplification
  (AB BC AC DC CD : V)
  (h1 : AB + BC = AC)
  (h2 : - DC = CD) :
  AB + BC - AC - DC = CD :=
by
  -- Placeholder for the proof
  sorry

end vector_addition_simplification_l225_225439


namespace fractions_sum_equals_one_l225_225686

variable {a b c x y z : ℝ}

variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 29 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

theorem fractions_sum_equals_one (a b c x y z : ℝ) 
  (h1 : 17 * x + b * y + c * z = 0)
  (h2 : a * x + 29 * y + c * z = 0)
  (h3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 := by 
  sorry

end fractions_sum_equals_one_l225_225686


namespace cubes_not_touching_tin_foil_volume_l225_225991

-- Definitions for the conditions given
variables (l w h : ℕ)
-- Condition 1: Width is twice the length
def width_twice_length := w = 2 * l
-- Condition 2: Width is twice the height
def width_twice_height := w = 2 * h
-- Condition 3: The adjusted width for the inner structure in inches
def adjusted_width := w = 8

-- The theorem statement to prove the final answer
theorem cubes_not_touching_tin_foil_volume : 
  width_twice_length l w → 
  width_twice_height w h →
  adjusted_width w →
  l * w * h = 128 :=
by
  intros h1 h2 h3
  sorry

end cubes_not_touching_tin_foil_volume_l225_225991


namespace basketball_not_table_tennis_l225_225370

-- Definitions and conditions
def total_students := 30
def like_basketball := 15
def like_table_tennis := 10
def do_not_like_either := 8
def like_both (x : ℕ) := x

-- Theorem statement
theorem basketball_not_table_tennis (x : ℕ) (H : (like_basketball - x) + (like_table_tennis - x) + x + do_not_like_either = total_students) : (like_basketball - x) = 12 :=
by
  sorry

end basketball_not_table_tennis_l225_225370


namespace shaded_trapezoids_perimeter_l225_225985

theorem shaded_trapezoids_perimeter :
  let l := 8
  let w := 6
  let half_diagonal_1 := (l^2 + w^2) / 2
  let perimeter := 2 * (w + (half_diagonal_1 / l))
  let total_perimeter := perimeter + perimeter + half_diagonal_1
  total_perimeter = 48 :=
by 
  sorry

end shaded_trapezoids_perimeter_l225_225985


namespace factorize_m_factorize_x_factorize_xy_l225_225405

theorem factorize_m (m : ℝ) : m^2 + 7 * m - 18 = (m - 2) * (m + 9) := 
sorry

theorem factorize_x (x : ℝ) : x^2 - 2 * x - 8 = (x + 2) * (x - 4) :=
sorry

theorem factorize_xy (x y : ℝ) : (x * y)^2 - 7 * (x * y) + 10 = (x * y - 2) * (x * y - 5) := 
sorry

end factorize_m_factorize_x_factorize_xy_l225_225405


namespace min_value_reciprocals_l225_225704

open Real

theorem min_value_reciprocals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 3 * b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3 * y = 1 → (1 / x + 1 / y) ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
sorry

end min_value_reciprocals_l225_225704


namespace problem_solution_l225_225674

-- Definitions of sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def B : Set ℝ := {-2, -1, 1, 2}

-- Complement of set A in reals
def C_A : Set ℝ := {x | x < 0}

-- Lean theorem statement
theorem problem_solution : (C_A ∩ B) = {-2, -1} :=
by sorry

end problem_solution_l225_225674


namespace two_candidates_solve_all_problems_l225_225049

-- Definitions for the conditions and problem context
def candidates : Nat := 200
def problems : Nat := 6 
def solved_by (p : Nat) : Nat := 120 -- at least 120 participants solve each problem.

-- The main theorem representing the proof problem
theorem two_candidates_solve_all_problems :
  (∃ c1 c2 : Fin candidates, ∀ p : Fin problems, (solved_by p ≥ 120)) :=
by
  sorry

end two_candidates_solve_all_problems_l225_225049


namespace tom_watching_days_l225_225709

def show_a_season_1_time : Nat := 20 * 22
def show_a_season_2_time : Nat := 18 * 24
def show_a_season_3_time : Nat := 22 * 26
def show_a_season_4_time : Nat := 15 * 30

def show_b_season_1_time : Nat := 24 * 42
def show_b_season_2_time : Nat := 16 * 48
def show_b_season_3_time : Nat := 12 * 55

def show_c_season_1_time : Nat := 10 * 60
def show_c_season_2_time : Nat := 13 * 58
def show_c_season_3_time : Nat := 15 * 50
def show_c_season_4_time : Nat := 11 * 52
def show_c_season_5_time : Nat := 9 * 65

def show_a_total_time : Nat :=
  show_a_season_1_time + show_a_season_2_time +
  show_a_season_3_time + show_a_season_4_time

def show_b_total_time : Nat :=
  show_b_season_1_time + show_b_season_2_time + show_b_season_3_time

def show_c_total_time : Nat :=
  show_c_season_1_time + show_c_season_2_time +
  show_c_season_3_time + show_c_season_4_time +
  show_c_season_5_time

def total_time : Nat := show_a_total_time + show_b_total_time + show_c_total_time

def daily_watch_time : Nat := 120

theorem tom_watching_days : (total_time + daily_watch_time - 1) / daily_watch_time = 64 := sorry

end tom_watching_days_l225_225709


namespace trigonometric_identity_l225_225237

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 :=
sorry

end trigonometric_identity_l225_225237


namespace nods_per_kilometer_l225_225290

theorem nods_per_kilometer
  (p q r s t u : ℕ)
  (h1 : p * q = q * p)
  (h2 : r * s = s * r)
  (h3 : t * u = u * t) : 
  (1 : ℕ) = qts/pru :=
by
  sorry

end nods_per_kilometer_l225_225290


namespace probability_relationship_l225_225263

def total_outcomes : ℕ := 36

def P1 : ℚ := 1 / total_outcomes
def P2 : ℚ := 2 / total_outcomes
def P3 : ℚ := 3 / total_outcomes

theorem probability_relationship :
  P1 < P2 ∧ P2 < P3 :=
by
  sorry

end probability_relationship_l225_225263


namespace xyz_value_l225_225252

theorem xyz_value
  (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 :=
  sorry

end xyz_value_l225_225252


namespace percentage_stock_sold_l225_225523

/-!
# Problem Statement
Given:
1. The cash realized on selling a certain percentage stock is Rs. 109.25.
2. The brokerage is 1/4%.
3. The cash after deducting the brokerage is Rs. 109.

Prove:
The percentage of the stock sold is 100%.
-/

noncomputable def brokerage_fee (S : ℝ) : ℝ :=
  S * 0.0025

noncomputable def selling_price (realized_cash : ℝ) (fee : ℝ) : ℝ :=
  realized_cash + fee

theorem percentage_stock_sold (S : ℝ) (realized_cash : ℝ) (cash_after_brokerage : ℝ)
  (h1 : realized_cash = 109.25)
  (h2 : cash_after_brokerage = 109)
  (h3 : brokerage_fee S = S * 0.0025) :
  S = 109.25 :=
by
  sorry

end percentage_stock_sold_l225_225523


namespace sufficient_but_not_necessary_l225_225189

theorem sufficient_but_not_necessary (x : ℝ) (h1 : x > 1 → x > 0) (h2 : ¬ (x > 0 → x > 1)) : 
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) := 
by 
  sorry

end sufficient_but_not_necessary_l225_225189


namespace hydrated_aluminum_iodide_props_l225_225348

noncomputable def Al_mass : ℝ := 26.98
noncomputable def I_mass : ℝ := 126.90
noncomputable def H2O_mass : ℝ := 18.015
noncomputable def AlI3_mass (mass_AlI3: ℝ) : ℝ := 26.98 + 3 * 126.90

noncomputable def mass_percentage_iodine (mass_AlI3 mass_sample: ℝ) : ℝ :=
  (mass_AlI3 * (3 * I_mass / (Al_mass + 3 * I_mass)) / mass_sample) * 100

noncomputable def value_x (mass_H2O mass_AlI3: ℝ) : ℝ :=
  (mass_H2O / H2O_mass) / (mass_AlI3 / (Al_mass + 3 * I_mass))

theorem hydrated_aluminum_iodide_props (mass_AlI3 mass_H2O mass_sample: ℝ)
    (h_sample: mass_AlI3 + mass_H2O = mass_sample) :
    ∃ (percentage: ℝ) (x: ℝ), percentage = mass_percentage_iodine mass_AlI3 mass_sample ∧
                                      x = value_x mass_H2O mass_AlI3 :=
by
  sorry

end hydrated_aluminum_iodide_props_l225_225348


namespace difference_rabbits_antelopes_l225_225761

variable (A R H W L : ℕ)
variable (x : ℕ)

def antelopes := 80
def rabbits := antelopes + x
def hyenas := (antelopes + rabbits) - 42
def wild_dogs := hyenas + 50
def leopards := rabbits / 2
def total_animals := 605

theorem difference_rabbits_antelopes
  (h1 : antelopes = 80)
  (h2 : rabbits = antelopes + x)
  (h3 : hyenas = (antelopes + rabbits) - 42)
  (h4 : wild_dogs = hyenas + 50)
  (h5 : leopards = rabbits / 2)
  (h6 : antelopes + rabbits + hyenas + wild_dogs + leopards = total_animals) : rabbits - antelopes = 70 := 
by
  -- Proof goes here
  sorry

end difference_rabbits_antelopes_l225_225761


namespace remaining_payment_l225_225176
noncomputable def total_cost (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / percentage

noncomputable def remaining_amount (deposit : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost - deposit

theorem remaining_payment (deposit : ℝ) (percentage : ℝ) (total_cost : ℝ) (remaining_amount : ℝ) :
  deposit = 140 → percentage = 0.1 → total_cost = deposit / percentage → remaining_amount = total_cost - deposit → remaining_amount = 1260 :=
by
  intros
  sorry

end remaining_payment_l225_225176


namespace circle_center_coordinates_l225_225034

-- Definition of the circle's equation
def circle_eq : Prop := ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 3

-- Proof of the circle's center coordinates
theorem circle_center_coordinates : ∃ h k : ℝ, (h, k) = (2, -1) := 
sorry

end circle_center_coordinates_l225_225034


namespace base_conversion_problem_l225_225969

theorem base_conversion_problem (n d : ℕ) (hn : 0 < n) (hd : d < 10) 
  (h1 : 3 * n^2 + 2 * n + d = 263) (h2 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) : 
  n + d = 11 :=
by
  sorry

end base_conversion_problem_l225_225969


namespace perfect_square_expression_l225_225739
open Real

theorem perfect_square_expression (x : ℝ) :
  (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = (12.86 + 0.14)^2) → x = 0.28 :=
by
  sorry

end perfect_square_expression_l225_225739


namespace sin_4theta_l225_225623

theorem sin_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5) :
  Real.sin (4 * θ) = (144 * Real.sqrt 7) / 625 := by
  sorry

end sin_4theta_l225_225623


namespace remainder_when_divided_by_15_l225_225117

theorem remainder_when_divided_by_15 (N : ℕ) (k : ℤ) (h1 : N = 60 * k + 49) : (N % 15) = 4 :=
sorry

end remainder_when_divided_by_15_l225_225117


namespace smallest_b_l225_225090

theorem smallest_b (a b : ℕ) (pos_a : 0 < a) (pos_b : 0 < b)
    (h1 : a - b = 4)
    (h2 : gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : b = 2 :=
sorry

end smallest_b_l225_225090


namespace M_is_listed_correctly_l225_225231

noncomputable def M : Set ℕ := { m | ∃ n : ℕ+, 3 / (5 - m : ℝ) = n }

theorem M_is_listed_correctly : M = { 2, 4 } :=
by
  sorry

end M_is_listed_correctly_l225_225231


namespace fewest_posts_required_l225_225445

def dimensions_garden : ℕ × ℕ := (32, 72)
def post_spacing : ℕ := 8

theorem fewest_posts_required
  (d : ℕ × ℕ := dimensions_garden)
  (s : ℕ := post_spacing) :
  d = (32, 72) ∧ s = 8 → 
  ∃ N, N = 26 := 
by 
  sorry

end fewest_posts_required_l225_225445


namespace milan_total_bill_correct_l225_225853

-- Define the monthly fee, the per minute rate, and the number of minutes used last month
def monthly_fee : ℝ := 2
def per_minute_rate : ℝ := 0.12
def minutes_used : ℕ := 178

-- Define the total bill calculation
def total_bill : ℝ := minutes_used * per_minute_rate + monthly_fee

-- The proof statement
theorem milan_total_bill_correct :
  total_bill = 23.36 := 
by
  sorry

end milan_total_bill_correct_l225_225853


namespace weekly_earnings_l225_225571

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end weekly_earnings_l225_225571


namespace solve_for_y_l225_225055

def diamond (a b : ℕ) : ℕ := 2 * a + b

theorem solve_for_y (y : ℕ) (h : diamond 4 (diamond 3 y) = 17) : y = 3 :=
by sorry

end solve_for_y_l225_225055


namespace factorial_sum_mod_30_l225_225822

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map factorial |>.sum

def remainder_when_divided_by (m k : Nat) : Nat :=
  m % k

theorem factorial_sum_mod_30 : remainder_when_divided_by (sum_of_factorials 100) 30 = 3 :=
by
  sorry

end factorial_sum_mod_30_l225_225822


namespace problem_x2_minus_y2_l225_225534

-- Problem statement: Given the conditions, prove x^2 - y^2 = 5 / 1111
theorem problem_x2_minus_y2 (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 101) :
  x^2 - y^2 = 5 / 1111 :=
by
  sorry

end problem_x2_minus_y2_l225_225534


namespace find_abs_of_y_l225_225217

theorem find_abs_of_y (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := 
sorry

end find_abs_of_y_l225_225217


namespace first_train_travels_more_l225_225554

-- Define the conditions
def velocity_first_train := 50 -- speed of the first train in km/hr
def velocity_second_train := 40 -- speed of the second train in km/hr
def distance_between_P_and_Q := 900 -- distance between P and Q in km

-- Problem statement
theorem first_train_travels_more :
  ∃ t : ℝ, (velocity_first_train * t + velocity_second_train * t = distance_between_P_and_Q)
          → (velocity_first_train * t - velocity_second_train * t = 100) :=
by sorry

end first_train_travels_more_l225_225554


namespace sin_300_eq_neg_sqrt3_div_2_l225_225430

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l225_225430


namespace lcm_5_6_10_15_l225_225011

theorem lcm_5_6_10_15 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 10 15) = 30 := 
by
  sorry

end lcm_5_6_10_15_l225_225011


namespace intersection_of_A_and_B_l225_225509

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {4} :=
by
  sorry

end intersection_of_A_and_B_l225_225509


namespace expand_expression_l225_225110

theorem expand_expression : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := 
by
  sorry

end expand_expression_l225_225110


namespace train_speed_is_correct_l225_225273

-- Definitions for conditions
def train_length : ℝ := 150  -- length of the train in meters
def time_to_cross_pole : ℝ := 3  -- time to cross the pole in seconds

-- Proof statement
theorem train_speed_is_correct : (train_length / time_to_cross_pole) = 50 := by
  sorry

end train_speed_is_correct_l225_225273


namespace number_of_sides_on_die_l225_225756

theorem number_of_sides_on_die (n : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : (∃ k : ℕ, k = 5) → (5 : ℚ) / (n ^ 2 : ℚ) = (5 : ℚ) / (36 : ℚ)) 
  : n = 6 :=
sorry

end number_of_sides_on_die_l225_225756


namespace max_abc_value_l225_225004

variables (a b c : ℕ)

theorem max_abc_value : 
  (a > 0) → (b > 0) → (c > 0) → a + 2 * b + 3 * c = 100 → abc ≤ 6171 := 
by sorry

end max_abc_value_l225_225004


namespace general_formula_sequence_l225_225156

theorem general_formula_sequence (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h_rec : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4^n - 1 :=
by 
  sorry

end general_formula_sequence_l225_225156


namespace trackball_mice_count_l225_225526

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l225_225526


namespace cultural_festival_recommendation_schemes_l225_225013

theorem cultural_festival_recommendation_schemes :
  (∃ (females : Finset ℕ) (males : Finset ℕ),
    females.card = 3 ∧ males.card = 2 ∧
    ∃ (dance : Finset ℕ) (singing : Finset ℕ) (instruments : Finset ℕ),
      dance.card = 2 ∧ dance ⊆ females ∧
      singing.card = 2 ∧ singing ∩ females ≠ ∅ ∧
      instruments.card = 1 ∧ instruments ⊆ males ∧
      (females ∪ males).card = 5) → 
  ∃ (recommendation_schemes : ℕ), recommendation_schemes = 18 :=
by
  sorry

end cultural_festival_recommendation_schemes_l225_225013


namespace probability_x_plus_y_lt_3_in_rectangle_l225_225169

noncomputable def probability_problem : ℚ :=
let rect_area := (4 : ℚ) * 3
let tri_area := (1 / 2 : ℚ) * 3 * 3
tri_area / rect_area

theorem probability_x_plus_y_lt_3_in_rectangle :
  probability_problem = 3 / 8 :=
sorry

end probability_x_plus_y_lt_3_in_rectangle_l225_225169


namespace total_spent_by_pete_and_raymond_l225_225219

def pete_initial_amount := 250
def pete_spending_on_stickers := 4 * 5
def pete_spending_on_candy := 3 * 10
def pete_spending_on_toy_car := 2 * 25
def pete_spending_on_keychain := 5
def pete_total_spent := pete_spending_on_stickers + pete_spending_on_candy + pete_spending_on_toy_car + pete_spending_on_keychain
def raymond_initial_amount := 250
def raymond_left_dimes := 7 * 10
def raymond_left_quarters := 4 * 25
def raymond_left_nickels := 5 * 5
def raymond_left_pennies := 3 * 1
def raymond_total_left := raymond_left_dimes + raymond_left_quarters + raymond_left_nickels + raymond_left_pennies
def raymond_total_spent := raymond_initial_amount - raymond_total_left
def total_spent := pete_total_spent + raymond_total_spent

theorem total_spent_by_pete_and_raymond : total_spent = 157 := by
  have h1 : pete_total_spent = 105 := sorry
  have h2 : raymond_total_spent = 52 := sorry
  exact sorry

end total_spent_by_pete_and_raymond_l225_225219


namespace max_value_of_m_l225_225847

theorem max_value_of_m :
  (∃ (t : ℝ), ∀ (x : ℝ), 2 ≤ x ∧ x ≤ m → (x + t)^2 ≤ 2 * x) → m ≤ 8 :=
sorry

end max_value_of_m_l225_225847


namespace battery_current_l225_225000

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l225_225000


namespace translation_coordinates_l225_225815

theorem translation_coordinates (A : ℝ × ℝ) (T : ℝ × ℝ) (A' : ℝ × ℝ) 
  (hA : A = (-4, 3)) (hT : T = (2, 0)) (hA' : A' = (A.1 + T.1, A.2 + T.2)) : 
  A' = (-2, 3) := sorry

end translation_coordinates_l225_225815


namespace climbing_difference_l225_225997

theorem climbing_difference (rate_matt rate_jason time : ℕ) (h_rate_matt : rate_matt = 6) (h_rate_jason : rate_jason = 12) (h_time : time = 7) : 
  rate_jason * time - rate_matt * time = 42 :=
by
  sorry

end climbing_difference_l225_225997


namespace survey_students_l225_225981

theorem survey_students (S F : ℕ) (h1 : F = 20 + 60) (h2 : F = 40 * S / 100) : S = 200 :=
by
  sorry

end survey_students_l225_225981


namespace solve_for_x_l225_225673

theorem solve_for_x : 
  (∃ x : ℝ, (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) 
  ∧ x = 4.5) := by
{
  sorry
}

end solve_for_x_l225_225673


namespace value_of_x_squared_plus_y_squared_l225_225283

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h : |x - 1/2| + (2*y + 1)^2 = 0) : 
  x^2 + y^2 = 1/2 :=
sorry

end value_of_x_squared_plus_y_squared_l225_225283


namespace halfway_fraction_l225_225938

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l225_225938


namespace total_wheels_at_park_l225_225153

-- Conditions as definitions
def number_of_adults := 6
def number_of_children := 15
def wheels_per_bicycle := 2
def wheels_per_tricycle := 3

-- To prove: total number of wheels = 57
theorem total_wheels_at_park : 
  (number_of_adults * wheels_per_bicycle) + (number_of_children * wheels_per_tricycle) = 57 :=
by
  sorry

end total_wheels_at_park_l225_225153


namespace train_crossing_time_l225_225490

noncomputable def length_of_train : ℝ := 120 -- meters
noncomputable def speed_of_train_kmh : ℝ := 27 -- kilometers per hour
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- converted to meters per second
noncomputable def time_to_cross : ℝ := length_of_train / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross = 16 :=
by
  -- proof goes here
  sorry

end train_crossing_time_l225_225490


namespace child_ticket_cost_l225_225318

theorem child_ticket_cost 
    (x : ℝ)
    (adult_ticket_cost : ℝ := 5)
    (total_sales : ℝ := 178)
    (total_tickets_sold : ℝ := 42)
    (child_tickets_sold : ℝ := 16) 
    (adult_tickets_sold : ℝ := total_tickets_sold - child_tickets_sold)
    (total_adult_sales : ℝ := adult_tickets_sold * adult_ticket_cost)
    (sales_equation : total_adult_sales + child_tickets_sold * x = total_sales) : 
    x = 3 :=
by
  sorry

end child_ticket_cost_l225_225318


namespace exists_two_numbers_l225_225316

theorem exists_two_numbers (x : Fin 7 → ℝ) :
  ∃ i j, 0 ≤ (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) ≤ 1 / Real.sqrt 3 :=
sorry

end exists_two_numbers_l225_225316


namespace tan_alpha_plus_pi_over_3_sin_cos_ratio_l225_225312

theorem tan_alpha_plus_pi_over_3
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  Real.tan (α + Real.pi / 3) = (48 - 25 * Real.sqrt 3) / 11 := 
sorry

theorem sin_cos_ratio
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17 :=
sorry

end tan_alpha_plus_pi_over_3_sin_cos_ratio_l225_225312


namespace solutionSet_l225_225905

def passesThroughQuadrants (a b : ℝ) : Prop :=
  a > 0

def intersectsXAxisAt (a b : ℝ) : Prop :=
  b = 2 * a

theorem solutionSet (a b x : ℝ) (hq : passesThroughQuadrants a b) (hi : intersectsXAxisAt a b) :
  (a * x > b) ↔ (x > 2) :=
by
  sorry

end solutionSet_l225_225905


namespace B_completion_time_l225_225747

-- Definitions based on the conditions
def A_work : ℚ := 1 / 24
def B_work : ℚ := 1 / 16
def C_work : ℚ := 1 / 32  -- Since C takes twice the time as B, C_work = B_work / 2

-- Combined work rates based on the conditions
def combined_ABC_work := A_work + B_work + C_work
def combined_AB_work := A_work + B_work

-- Question: How long does B take to complete the job alone?
-- Answer: 16 days

theorem B_completion_time : 
  (combined_ABC_work = 1 / 8) ∧ 
  (combined_AB_work = 1 / 12) ∧ 
  (A_work = 1 / 24) ∧ 
  (C_work = B_work / 2) → 
  (1 / B_work = 16) := 
by 
  sorry

end B_completion_time_l225_225747


namespace parallel_lines_find_m_l225_225122

theorem parallel_lines_find_m :
  (∀ (m : ℝ), ∀ (x y : ℝ), (2 * x + (m + 1) * y + 4 = 0) ∧ (m * x + 3 * y - 2 = 0) → (m = -3 ∨ m = 2)) := 
sorry

end parallel_lines_find_m_l225_225122


namespace selection_ways_l225_225584

def ways_to_select_president_and_secretary (n : Nat) : Nat :=
  n * (n - 1)

theorem selection_ways :
  ways_to_select_president_and_secretary 5 = 20 :=
by
  sorry

end selection_ways_l225_225584


namespace num_divisors_of_36_l225_225992

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end num_divisors_of_36_l225_225992


namespace sum_of_primes_between_1_and_20_l225_225983

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l225_225983


namespace decompose_two_over_eleven_decompose_two_over_n_l225_225382

-- Problem 1: Decompose 2/11
theorem decompose_two_over_eleven : (2 : ℚ) / 11 = (1 / 6) + (1 / 66) :=
  sorry

-- Problem 2: General form for 2/n for odd n >= 5
theorem decompose_two_over_n (n : ℕ) (hn : n ≥ 5) (odd_n : n % 2 = 1) :
  (2 : ℚ) / n = (1 / ((n + 1) / 2)) + (1 / (n * (n + 1) / 2)) :=
  sorry

end decompose_two_over_eleven_decompose_two_over_n_l225_225382


namespace function_properties_l225_225455

-- Define the function and conditions
def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f (10 + x) = f (10 - x)
axiom condition2 (x : ℝ) : f (20 - x) = -f (20 + x)

-- Lean statement to encapsulate the question and expected result
theorem function_properties (x : ℝ) : (f (-x) = -f x) ∧ (f (x + 40) = f x) :=
sorry

end function_properties_l225_225455


namespace two_digit_number_count_four_digit_number_count_l225_225659

-- Defining the set of digits
def digits : Finset ℕ := {1, 2, 3, 4}

-- Problem 1 condition and question
def two_digit_count := Nat.choose 4 2 * 2

-- Problem 2 condition and question
def four_digit_count := Nat.choose 4 4 * 24

-- Theorem statement for Problem 1
theorem two_digit_number_count : two_digit_count = 12 :=
sorry

-- Theorem statement for Problem 2
theorem four_digit_number_count : four_digit_count = 24 :=
sorry

end two_digit_number_count_four_digit_number_count_l225_225659


namespace distance_a_beats_b_l225_225030

noncomputable def time_a : ℕ := 90 -- A's time in seconds 
noncomputable def time_b : ℕ := 180 -- B's time in seconds 
noncomputable def distance : ℝ := 4.5 -- distance in km

theorem distance_a_beats_b : distance = (distance / time_a) * (time_b - time_a) :=
by
  -- sorry placeholder for proof
  sorry

end distance_a_beats_b_l225_225030


namespace determine_top_5_median_required_l225_225767

theorem determine_top_5_median_required (scores : Fin 9 → ℝ) (unique_scores : ∀ (i j : Fin 9), i ≠ j → scores i ≠ scores j) :
  ∃ median,
  (∀ (student_score : ℝ), 
    (student_score > median ↔ ∃ (idx_top : Fin 5), student_score = scores ⟨idx_top.1, sorry⟩)) :=
sorry

end determine_top_5_median_required_l225_225767


namespace fuel_tank_initial_capacity_l225_225521

variables (fuel_consumption : ℕ) (journey_distance remaining_fuel initial_fuel : ℕ)

-- Define conditions
def fuel_consumption_rate := 12      -- liters per 100 km
def journey := 275                  -- km
def remaining := 14                 -- liters
def fuel_converted := (fuel_consumption_rate * journey) / 100

-- Define the proposition to be proved
theorem fuel_tank_initial_capacity :
  initial_fuel = fuel_converted + remaining :=
sorry

end fuel_tank_initial_capacity_l225_225521


namespace friend_jogging_time_l225_225816

theorem friend_jogging_time (D : ℝ) (my_time : ℝ) (friend_speed : ℝ) :
  my_time = 3 * 60 →
  friend_speed = 2 * (D / my_time) →
  (D / friend_speed) = 90 :=
by
  sorry

end friend_jogging_time_l225_225816


namespace unique_triple_primes_l225_225741

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end unique_triple_primes_l225_225741


namespace smallest_n_exists_l225_225821

theorem smallest_n_exists (G : Type) [Fintype G] [DecidableEq G] (connected : G → G → Prop)
  (distinct_naturals : G → ℕ) :
  (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 = 1) ∧
  (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 > 1) →
  (∀ n : ℕ, 
    (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) n = 1) ∧
    (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) n > 1) →
    15 ≤ n) :=
sorry

end smallest_n_exists_l225_225821


namespace avg_price_of_pencil_l225_225589

theorem avg_price_of_pencil 
  (total_pens : ℤ) (total_pencils : ℤ) (total_cost : ℤ)
  (avg_cost_pen : ℤ) (avg_cost_pencil : ℤ) :
  total_pens = 30 → 
  total_pencils = 75 → 
  total_cost = 690 → 
  avg_cost_pen = 18 → 
  (total_cost - total_pens * avg_cost_pen) / total_pencils = avg_cost_pencil → 
  avg_cost_pencil = 2 :=
by
  intros
  sorry

end avg_price_of_pencil_l225_225589


namespace evaluate_polynomial_at_6_l225_225671

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

theorem evaluate_polynomial_at_6 : polynomial 6 = 3658 :=
by 
  sorry

end evaluate_polynomial_at_6_l225_225671


namespace special_number_is_square_l225_225474

-- Define the special number format
def special_number (n : ℕ) : ℕ :=
  3 * (10^n - 1)/9 + 4

theorem special_number_is_square (n : ℕ) :
  ∃ k : ℕ, k * k = special_number n := by
  sorry

end special_number_is_square_l225_225474


namespace find_constant_k_l225_225224

theorem find_constant_k (k : ℤ) :
    (∀ x : ℝ, -x^2 - (k + 7) * x - 8 = - (x - 2) * (x - 4)) → k = -13 :=
by 
    intros h
    sorry

end find_constant_k_l225_225224


namespace min_rainfall_on_fourth_day_l225_225486

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end min_rainfall_on_fourth_day_l225_225486


namespace side_lengths_of_triangle_l225_225245

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem side_lengths_of_triangle (m : ℝ) (a b c : ℝ) 
  (h1 : f m a > 0) 
  (h2 : f m b > 0) 
  (h3 : f m c > 0) 
  (h4 : f m a + f m b > f m c)
  (h5 : f m a + f m c > f m b)
  (h6 : f m b + f m c > f m a) :
  m ∈ Set.Ioo (7/5 : ℝ) 5 :=
sorry

end side_lengths_of_triangle_l225_225245


namespace three_digit_integers_sat_f_n_eq_f_2005_l225_225527

theorem three_digit_integers_sat_f_n_eq_f_2005 
  (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m + n) = f (f m + n))
  (h2 : f 6 = 2)
  (h3 : f 6 ≠ f 9)
  (h4 : f 6 ≠ f 12)
  (h5 : f 6 ≠ f 15)
  (h6 : f 9 ≠ f 12)
  (h7 : f 9 ≠ f 15)
  (h8 : f 12 ≠ f 15) :
  ∃! n, 100 ≤ n ∧ n ≤ 999 ∧ f n = f 2005 → n = 225 := 
  sorry

end three_digit_integers_sat_f_n_eq_f_2005_l225_225527


namespace pet_store_problem_l225_225015

theorem pet_store_problem 
  (initial_puppies : ℕ) 
  (sold_day1 : ℕ) 
  (sold_day2 : ℕ) 
  (sold_day3 : ℕ) 
  (sold_day4 : ℕ)
  (sold_day5 : ℕ) 
  (puppies_per_cage : ℕ)
  (initial_puppies_eq : initial_puppies = 120) 
  (sold_day1_eq : sold_day1 = 25) 
  (sold_day2_eq : sold_day2 = 10) 
  (sold_day3_eq : sold_day3 = 30) 
  (sold_day4_eq : sold_day4 = 15) 
  (sold_day5_eq : sold_day5 = 28) 
  (puppies_per_cage_eq : puppies_per_cage = 6) : 
  (initial_puppies - (sold_day1 + sold_day2 + sold_day3 + sold_day4 + sold_day5)) / puppies_per_cage = 2 := 
by 
  sorry

end pet_store_problem_l225_225015


namespace count_solutions_l225_225579

theorem count_solutions : 
  (∃ (n : ℕ), ∀ (x : ℕ), (x + 17) % 43 = 71 % 43 ∧ x < 150 → n = 4) := 
sorry

end count_solutions_l225_225579


namespace solve_system_l225_225683

theorem solve_system (X Y : ℝ) : 
  (X + (X + 2 * Y) / (X^2 + Y^2) = 2 ∧ Y + (2 * X - Y) / (X^2 + Y^2) = 0) ↔ (X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1) :=
by
  sorry

end solve_system_l225_225683


namespace count_not_divisible_by_5_or_7_l225_225977

theorem count_not_divisible_by_5_or_7 :
  let n := 1000
  let count_divisible_by (m : ℕ) := Nat.floor (999 / m)
  (999 - count_divisible_by 5 - count_divisible_by 7 + count_divisible_by 35) = 686 :=
by
  sorry

end count_not_divisible_by_5_or_7_l225_225977


namespace negation_of_exists_proposition_l225_225229

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 * x > 2) ↔ (∀ x : ℝ, x^2 - 2 * x ≤ 2) :=
by
  sorry

end negation_of_exists_proposition_l225_225229


namespace sequence_general_term_l225_225088

theorem sequence_general_term {a : ℕ → ℝ} (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 4 * a n - 3) :
  a n = (4/3)^(n-1) :=
sorry

end sequence_general_term_l225_225088


namespace find_value_of_m_l225_225839

/-- Given the universal set U, set A, and the complement of A in U, we prove that m = -2. -/
theorem find_value_of_m (m : ℤ) (U : Set ℤ) (A : Set ℤ) (complement_U_A : Set ℤ) 
  (h1 : U = {2, 3, m^2 + m - 4})
  (h2 : A = {m, 2})
  (h3 : complement_U_A = {3}) 
  (h4 : U = A ∪ complement_U_A) 
  (h5 : A ∩ complement_U_A = ∅) 
  : m = -2 :=
sorry

end find_value_of_m_l225_225839


namespace average_price_blankets_l225_225426

theorem average_price_blankets :
  let cost_blankets1 := 3 * 100
  let cost_blankets2 := 5 * 150
  let cost_blankets3 := 550
  let total_cost := cost_blankets1 + cost_blankets2 + cost_blankets3
  let total_blankets := 3 + 5 + 2
  total_cost / total_blankets = 160 :=
by
  sorry

end average_price_blankets_l225_225426


namespace janice_purchase_l225_225341

theorem janice_purchase : 
  ∃ (a b c : ℕ), a + b + c = 50 ∧ 50 * a + 400 * b + 500 * c = 10000 ∧ a = 23 :=
by
  sorry

end janice_purchase_l225_225341


namespace parabola_equation_l225_225912

variables (x y : ℝ)

def parabola_passes_through_point (x y : ℝ) : Prop :=
(x = 2 ∧ y = 7)

def focus_x_coord_five (x : ℝ) : Prop :=
(x = 5)

def axis_of_symmetry_parallel_to_y : Prop := True

def vertex_lies_on_x_axis (x y : ℝ) : Prop :=
(x = 5 ∧ y = 0)

theorem parabola_equation
  (h1 : parabola_passes_through_point x y)
  (h2 : focus_x_coord_five x)
  (h3 : axis_of_symmetry_parallel_to_y)
  (h4 : vertex_lies_on_x_axis x y) :
  49 * x + 3 * y^2 - 245 = 0
:= sorry

end parabola_equation_l225_225912


namespace rate_of_second_batch_l225_225178

-- Define the problem statement
theorem rate_of_second_batch
  (rate_first : ℝ)
  (weight_first weight_second weight_total : ℝ)
  (rate_mixture : ℝ)
  (profit_multiplier : ℝ) 
  (total_selling_price : ℝ) :
  rate_first = 11.5 →
  weight_first = 30 →
  weight_second = 20 →
  weight_total = weight_first + weight_second →
  rate_mixture = 15.12 →
  profit_multiplier = 1.20 →
  total_selling_price = weight_total * rate_mixture →
  (rate_first * weight_first + (weight_second * x) * profit_multiplier = total_selling_price) →
  x = 14.25 :=
by
  intros
  sorry

end rate_of_second_batch_l225_225178


namespace verify_b_c_sum_ten_l225_225269

theorem verify_b_c_sum_ten (a b c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) (hc : 1 ≤ c ∧ c < 10) 
    (h_eq : (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a ^ 2) : b + c = 10 :=
by
  sorry

end verify_b_c_sum_ten_l225_225269


namespace Geraldine_more_than_Jazmin_l225_225846

-- Define the number of dolls Geraldine and Jazmin have
def Geraldine_dolls : ℝ := 2186.0
def Jazmin_dolls : ℝ := 1209.0

-- State the theorem we need to prove
theorem Geraldine_more_than_Jazmin :
  Geraldine_dolls - Jazmin_dolls = 977.0 := 
by
  sorry

end Geraldine_more_than_Jazmin_l225_225846


namespace max_radius_of_sector_l225_225477

def sector_perimeter_area (r : ℝ) : ℝ := -r^2 + 10 * r

theorem max_radius_of_sector (R A : ℝ) (h : 2 * R + A = 20) : R = 5 :=
by
  sorry

end max_radius_of_sector_l225_225477


namespace division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l225_225149

theorem division_to_fraction : (7 / 9) = 7 / 9 := by
  sorry

theorem fraction_to_division : 12 / 7 = 12 / 7 := by
  sorry

theorem mixed_to_improper_fraction : (3 + 5 / 8) = 29 / 8 := by
  sorry

theorem whole_to_fraction : 6 = 66 / 11 := by
  sorry

end division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l225_225149


namespace simplify_expr_l225_225253

theorem simplify_expr : 
  (576:ℝ)^(1/4) * (216:ℝ)^(1/2) = 72 := 
by 
  have h1 : 576 = (2^4 * 36 : ℝ) := by norm_num
  have h2 : 36 = (6^2 : ℝ) := by norm_num
  have h3 : 216 = (6^3 : ℝ) := by norm_num
  sorry

end simplify_expr_l225_225253


namespace angle_alpha_range_l225_225898

/-- Given point P (tan α, sin α - cos α) is in the first quadrant, 
and 0 ≤ α ≤ 2π, then the range of values for angle α is (π/4, π/2) ∪ (π, 5π/4). -/
theorem angle_alpha_range (α : ℝ) 
  (h0 : 0 ≤ α) (h1 : α ≤ 2 * Real.pi) 
  (h2 : Real.tan α > 0) (h3 : Real.sin α - Real.cos α > 0) : 
  (Real.pi / 4 < α ∧ α < Real.pi / 2) ∨ 
  (Real.pi < α ∧ α < 5 * Real.pi / 4) :=
sorry

end angle_alpha_range_l225_225898


namespace number_of_Cl_atoms_l225_225019

/-- 
Given a compound with 1 aluminum atom and a molecular weight of 132 g/mol,
prove that the number of chlorine atoms in the compound is 3.
--/
theorem number_of_Cl_atoms 
  (weight_Al : ℝ) 
  (weight_Cl : ℝ) 
  (molecular_weight : ℝ)
  (ha : weight_Al = 26.98)
  (hc : weight_Cl = 35.45)
  (hm : molecular_weight = 132) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_Cl_atoms_l225_225019


namespace average_interest_rate_correct_l225_225726

-- Constants representing the conditions
def totalInvestment : ℝ := 5000
def rateA : ℝ := 0.035
def rateB : ℝ := 0.07

-- The condition that return from investment at 7% is twice that at 3.5%
def return_condition (x : ℝ) : Prop := 0.07 * x = 2 * 0.035 * (5000 - x)

-- The average rate of interest formula
noncomputable def average_rate_of_interest (x : ℝ) : ℝ := 
  (0.07 * x + 0.035 * (5000 - x)) / 5000

-- The theorem to prove the average rate is 5.25%
theorem average_interest_rate_correct : ∃ (x : ℝ), return_condition x ∧ average_rate_of_interest x = 0.0525 := 
by
  sorry

end average_interest_rate_correct_l225_225726


namespace sequence_value_2023_l225_225298

theorem sequence_value_2023 (a : ℕ → ℕ) (h₁ : a 1 = 3)
  (h₂ : ∀ m n : ℕ, a (m + n) = a m + a n) : a 2023 = 6069 := by
  sorry

end sequence_value_2023_l225_225298


namespace ThreePowerTowerIsLarger_l225_225826

-- original power tower definitions
def A : ℕ := 3^(3^(3^3))
def B : ℕ := 2^(2^(2^(2^2)))

-- reduced forms given from the conditions
def reducedA : ℕ := 3^(3^27)
def reducedB : ℕ := 2^(2^16)

theorem ThreePowerTowerIsLarger : reducedA > reducedB := by
  sorry

end ThreePowerTowerIsLarger_l225_225826


namespace turnover_threshold_l225_225669

-- Definitions based on the problem conditions
def valid_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def daily_turnover (x : ℝ) : ℝ := 20 * (10 - x) * (50 + 8 * x)

-- Lean 4 statement equivalent to mathematical proof problem
theorem turnover_threshold (x : ℝ) (hx : valid_domain x) (h_turnover : daily_turnover x ≥ 10260) :
  x ≥ 1 / 2 ∧ x ≤ 2 :=
sorry

end turnover_threshold_l225_225669


namespace basketball_team_starters_l225_225289

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  choose 4 2 * choose 14 4 = 6006 := by
  sorry

end basketball_team_starters_l225_225289


namespace exists_non_prime_form_l225_225095

theorem exists_non_prime_form (n : ℕ) : ∃ n : ℕ, ¬Nat.Prime (n^2 + n + 41) :=
sorry

end exists_non_prime_form_l225_225095


namespace rabbit_weight_l225_225653

variable (k r p : ℝ)

theorem rabbit_weight :
  k + r + p = 39 →
  r + p = 3 * k →
  r + k = 1.5 * p →
  r = 13.65 :=
by
  intros h1 h2 h3
  sorry

end rabbit_weight_l225_225653


namespace melanie_cats_l225_225733

theorem melanie_cats (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ) 
  (h_jacob : jacob_cats = 90)
  (h_annie : annie_cats = jacob_cats / 3)
  (h_melanie : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end melanie_cats_l225_225733


namespace fraction_product_eq_one_l225_225909

theorem fraction_product_eq_one :
  (7 / 4 : ℚ) * (8 / 14) * (21 / 12) * (16 / 28) * (49 / 28) * (24 / 42) * (63 / 36) * (32 / 56) = 1 := by
  sorry

end fraction_product_eq_one_l225_225909


namespace linda_total_distance_l225_225155

theorem linda_total_distance :
  ∃ x: ℕ, 
    (x > 0) ∧ (60 % x = 0) ∧
    ((x + 5) > 0) ∧ (60 % (x + 5) = 0) ∧
    ((x + 10) > 0) ∧ (60 % (x + 10) = 0) ∧
    ((x + 15) > 0) ∧ (60 % (x + 15) = 0) ∧
    (60 / x + 60 / (x + 5) + 60 / (x + 10) + 60 / (x + 15) = 25) :=
by
  sorry

end linda_total_distance_l225_225155


namespace problem_statement_l225_225132

-- Define rational number representations for points A, B, and C
def a : ℚ := (-4)^2 - 8

-- Define that B and C are opposites
def are_opposites (b c : ℚ) : Prop := b = -c

-- Define the distance condition
def distance_is_three (a c : ℚ) : Prop := |c - a| = 3

-- Main theorem statement
theorem problem_statement :
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -74) ∨
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -86) :=
sorry

end problem_statement_l225_225132


namespace quadratic_equation_factored_form_correct_l225_225725

theorem quadratic_equation_factored_form_correct :
  ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intros x h
  sorry

end quadratic_equation_factored_form_correct_l225_225725


namespace triangle_side_length_l225_225999

theorem triangle_side_length (A : ℝ) (AC BC AB : ℝ) 
  (hA : A = 60)
  (hAC : AC = 4)
  (hBC : BC = 2 * Real.sqrt 3) :
  AB = 2 :=
sorry

end triangle_side_length_l225_225999


namespace number_of_divisors_of_n_l225_225916

def n : ℕ := 2^3 * 3^4 * 5^3 * 7^2

theorem number_of_divisors_of_n : ∃ d : ℕ, d = 240 ∧ ∀ k : ℕ, k ∣ n ↔ ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 3 ∧ 0 ≤ d ∧ d ≤ 2 := 
sorry

end number_of_divisors_of_n_l225_225916


namespace elastic_collision_ball_speed_l225_225807

open Real

noncomputable def final_ball_speed (v_car v_ball : ℝ) : ℝ :=
  let relative_speed := v_ball + v_car
  relative_speed + v_car

theorem elastic_collision_ball_speed :
  let v_car := 5
  let v_ball := 6
  final_ball_speed v_car v_ball = 16 := 
by
  sorry

end elastic_collision_ball_speed_l225_225807


namespace marks_fathers_gift_l225_225031

noncomputable def total_spent (books : ℕ) (cost_per_book : ℕ) : ℕ :=
  books * cost_per_book

noncomputable def total_money_given (spent : ℕ) (left_over : ℕ) : ℕ :=
  spent + left_over

theorem marks_fathers_gift :
  total_money_given (total_spent 10 5) 35 = 85 := by
  sorry

end marks_fathers_gift_l225_225031


namespace tan_sum_half_l225_225915

theorem tan_sum_half (a b : ℝ) (h1 : Real.cos a + Real.cos b = 3/5) (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1 / 3 := 
by
  sorry

end tan_sum_half_l225_225915


namespace range_f_3_l225_225939

section

variables (a c : ℝ) (f : ℝ → ℝ)
def quadratic_function := ∀ x, f x = a * x^2 - c

-- Define the constraints given in the problem
axiom h1 : -4 ≤ f 1 ∧ f 1 ≤ -1
axiom h2 : -1 ≤ f 2 ∧ f 2 ≤ 5

-- Prove that the correct range for f(3) is -1 ≤ f(3) ≤ 20
theorem range_f_3 (a c : ℝ) (f : ℝ → ℝ) (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5):
  -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end

end range_f_3_l225_225939


namespace find_square_l225_225048

theorem find_square (y : ℝ) (h : (y + 5)^(1/3) = 3) : (y + 5)^2 = 729 := 
sorry

end find_square_l225_225048


namespace line_tangent_circle_iff_m_l225_225522

/-- Definition of the circle and the line -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- Prove that the line is tangent to the circle if and only if m = -3 or m = -13 -/
theorem line_tangent_circle_iff_m (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y m) ↔ m = -3 ∨ m = -13 :=
by
  sorry

end line_tangent_circle_iff_m_l225_225522


namespace total_basketballs_l225_225501

theorem total_basketballs (soccer_balls : ℕ) (soccer_balls_with_holes : ℕ) (basketballs_with_holes : ℕ) (balls_without_holes : ℕ) 
  (h1 : soccer_balls = 40) 
  (h2 : soccer_balls_with_holes = 30) 
  (h3 : basketballs_with_holes = 7) 
  (h4 : balls_without_holes = 18)
  (soccer_balls_without_holes : ℕ) 
  (basketballs_without_holes : ℕ) 
  (total_basketballs : ℕ)
  (h5 : soccer_balls_without_holes = soccer_balls - soccer_balls_with_holes)
  (h6 : basketballs_without_holes = balls_without_holes - soccer_balls_without_holes)
  (h7 : total_basketballs = basketballs_without_holes + basketballs_with_holes) : 
  total_basketballs = 15 := 
sorry

end total_basketballs_l225_225501


namespace units_digit_17_pow_2007_l225_225563

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l225_225563


namespace correct_calculation_is_d_l225_225105

theorem correct_calculation_is_d :
  (-7) + (-7) ≠ 0 ∧
  ((-1 / 10) - (1 / 10)) ≠ 0 ∧
  (0 + (-101)) ≠ 101 ∧
  (1 / 3 + -1 / 2 = -1 / 6) :=
by
  sorry

end correct_calculation_is_d_l225_225105


namespace perfect_square_fraction_l225_225852

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem perfect_square_fraction (a b : ℕ) 
  (h_pos_a: 0 < a) 
  (h_pos_b: 0 < b) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  is_perfect_square ((a^2 + b^2) / (a * b + 1)) := 
sorry

end perfect_square_fraction_l225_225852


namespace silver_cube_price_l225_225286

theorem silver_cube_price
  (price_2inch_cube : ℝ := 300) (side_length_2inch : ℝ := 2) (side_length_4inch : ℝ := 4) : 
  price_4inch_cube = 2400 := 
by 
  sorry

end silver_cube_price_l225_225286


namespace new_acute_angle_l225_225906

/- Definitions -/
def initial_angle_A (ACB : ℝ) (angle_CAB : ℝ) := angle_CAB = 40
def rotation_degrees (rotation : ℝ) := rotation = 480

/- Theorem Statement -/
theorem new_acute_angle (ACB : ℝ) (angle_CAB : ℝ) (rotation : ℝ) :
  initial_angle_A angle_CAB ACB ∧ rotation_degrees rotation → angle_CAB = 80 := 
by
  intros h
  -- This is where you'd provide the proof steps, but we use 'sorry' to indicate the proof is skipped.
  sorry

end new_acute_angle_l225_225906


namespace solution_set_of_inequality_l225_225518

theorem solution_set_of_inequality:
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_inequality_l225_225518


namespace simplest_square_root_l225_225076

noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def inv_sqrt2 : ℝ := 1 / Real.sqrt 2
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt_inv2 : ℝ := Real.sqrt (1 / 2)

theorem simplest_square_root : sqrt2 = Real.sqrt 2 := 
  sorry

end simplest_square_root_l225_225076


namespace apples_left_total_l225_225970

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_l225_225970


namespace gum_ratio_correct_l225_225247

variable (y : ℝ)
variable (cherry_pieces : ℝ := 30)
variable (grape_pieces : ℝ := 40)
variable (pieces_per_pack : ℝ := y)

theorem gum_ratio_correct:
  ((cherry_pieces - 2 * pieces_per_pack) / grape_pieces = cherry_pieces / (grape_pieces + 4 * pieces_per_pack)) ↔ y = 5 :=
by
  sorry

end gum_ratio_correct_l225_225247


namespace maximum_term_of_sequence_l225_225444

noncomputable def a (n : ℕ) : ℝ := n * (3 / 4)^n

theorem maximum_term_of_sequence : ∃ n : ℕ, a n = a 3 ∧ ∀ m : ℕ, a m ≤ a 3 :=
by sorry

end maximum_term_of_sequence_l225_225444


namespace isosceles_triangle_base_length_l225_225446

theorem isosceles_triangle_base_length :
  ∀ (p_equilateral p_isosceles side_equilateral : ℕ), 
  p_equilateral = 60 → 
  side_equilateral = p_equilateral / 3 →
  p_isosceles = 55 →
  ∀ (base_isosceles : ℕ),
  side_equilateral + side_equilateral + base_isosceles = p_isosceles →
  base_isosceles = 15 :=
by
  intros p_equilateral p_isosceles side_equilateral h1 h2 h3 base_isosceles h4
  sorry

end isosceles_triangle_base_length_l225_225446


namespace integral_eval_l225_225540

theorem integral_eval : ∫ x in (1:ℝ)..(2:ℝ), (2*x + 1/x) = 3 + Real.log 2 := by
  sorry

end integral_eval_l225_225540


namespace no_very_convex_function_exists_l225_225399

-- Definition of very convex function
def very_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

-- Theorem stating the non-existence of very convex functions
theorem no_very_convex_function_exists : ¬∃ f : ℝ → ℝ, very_convex f :=
by {
  sorry
}

end no_very_convex_function_exists_l225_225399


namespace quadratic_has_one_real_root_positive_value_of_m_l225_225845

theorem quadratic_has_one_real_root (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 0 ∨ m = 1/4 := by
  sorry

theorem positive_value_of_m (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 1/4 := by
  have root_cases := quadratic_has_one_real_root m h
  cases root_cases
  · exfalso
    -- We know m = 0 cannot be the positive m we are looking for.
    sorry
  · assumption

end quadratic_has_one_real_root_positive_value_of_m_l225_225845


namespace wade_final_profit_l225_225240

theorem wade_final_profit :
  let tips_per_customer_friday := 2.00
  let customers_friday := 28
  let tips_per_customer_saturday := 2.50
  let customers_saturday := 3 * customers_friday
  let tips_per_customer_sunday := 1.50
  let customers_sunday := 36
  let cost_ingredients_per_hotdog := 1.25
  let price_per_hotdog := 4.00
  let truck_maintenance_daily_cost := 50.00
  let total_taxes := 150.00
  let revenue_tips_friday := tips_per_customer_friday * customers_friday
  let revenue_hotdogs_friday := customers_friday * price_per_hotdog
  let cost_ingredients_friday := customers_friday * cost_ingredients_per_hotdog
  let revenue_friday := revenue_tips_friday + revenue_hotdogs_friday
  let total_costs_friday := cost_ingredients_friday + truck_maintenance_daily_cost
  let profit_friday := revenue_friday - total_costs_friday
  let revenue_tips_saturday := tips_per_customer_saturday * customers_saturday
  let revenue_hotdogs_saturday := customers_saturday * price_per_hotdog
  let cost_ingredients_saturday := customers_saturday * cost_ingredients_per_hotdog
  let revenue_saturday := revenue_tips_saturday + revenue_hotdogs_saturday
  let total_costs_saturday := cost_ingredients_saturday + truck_maintenance_daily_cost
  let profit_saturday := revenue_saturday - total_costs_saturday
  let revenue_tips_sunday := tips_per_customer_sunday * customers_sunday
  let revenue_hotdogs_sunday := customers_sunday * price_per_hotdog
  let cost_ingredients_sunday := customers_sunday * cost_ingredients_per_hotdog
  let revenue_sunday := revenue_tips_sunday + revenue_hotdogs_sunday
  let total_costs_sunday := cost_ingredients_sunday + truck_maintenance_daily_cost
  let profit_sunday := revenue_sunday - total_costs_sunday
  let total_profit := profit_friday + profit_saturday + profit_sunday
  let final_profit := total_profit - total_taxes
  final_profit = 427.00 :=
by
  sorry

end wade_final_profit_l225_225240


namespace tractor_planting_rate_l225_225018

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l225_225018


namespace arithmetic_sequence_formula_l225_225097

theorem arithmetic_sequence_formula (x : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = x - 1) (h2 : a 2 = x + 1) (h3 : a 3 = 2 * x + 3) :
  ∃ c d : ℤ, (∀ n : ℕ, a n = c + d * (n - 1)) ∧ ∀ n : ℕ, a n = 2 * n - 3 :=
by {
  sorry
}

end arithmetic_sequence_formula_l225_225097


namespace family_vacation_rain_days_l225_225058

theorem family_vacation_rain_days (r_m r_a : ℕ) 
(h_rain_days : r_m + r_a = 13)
(clear_mornings : r_a = 11)
(clear_afternoons : r_m = 12) : 
r_m + r_a = 23 := 
by 
  sorry

end family_vacation_rain_days_l225_225058


namespace xy_relationship_l225_225454

theorem xy_relationship (x y : ℝ) (h : y = 2 * x - 1 - Real.sqrt (y^2 - 2 * x * y + 3 * x - 2)) :
  (x ≠ 1 → y = 2 * x - 1.5) ∧ (x = 1 → y ≤ 1) :=
by
  sorry

end xy_relationship_l225_225454


namespace non_congruent_non_square_rectangles_count_l225_225708

theorem non_congruent_non_square_rectangles_count :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → 2 * (x.1 + x.2) = 80) ∧
    S.card = 19 ∧
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 ≠ x.2) ∧
    (∀ (x y : ℕ × ℕ), x ∈ S → y ∈ S → x ≠ y → x.1 = y.2 → x.2 = y.1) :=
sorry

end non_congruent_non_square_rectangles_count_l225_225708


namespace no_integer_solutions_l225_225676

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l225_225676


namespace xyz_inequality_l225_225675

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l225_225675


namespace div_by_7_or_11_l225_225728

theorem div_by_7_or_11 (z x y : ℕ) (hx : x < 1000) (hz : z = 1000 * y + x) (hdiv7 : (x - y) % 7 = 0 ∨ (x - y) % 11 = 0) :
  z % 7 = 0 ∨ z % 11 = 0 :=
by
  sorry

end div_by_7_or_11_l225_225728


namespace total_pizzas_bought_l225_225192

theorem total_pizzas_bought (slices_small : ℕ) (slices_medium : ℕ) (slices_large : ℕ) 
                            (num_small : ℕ) (num_medium : ℕ) (total_slices : ℕ) :
  slices_small = 6 → 
  slices_medium = 8 → 
  slices_large = 12 → 
  num_small = 4 → 
  num_medium = 5 → 
  total_slices = 136 → 
  (total_slices = num_small * slices_small + num_medium * slices_medium + 72) →
  15 = num_small + num_medium + 6 :=
by
  intros
  sorry

end total_pizzas_bought_l225_225192


namespace phase_shift_equivalence_l225_225755

noncomputable def y_original (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def y_target (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1
noncomputable def phase_shift : ℝ := 5 * Real.pi / 12

theorem phase_shift_equivalence : 
  ∀ x : ℝ, y_original x = y_target (x - phase_shift) :=
sorry

end phase_shift_equivalence_l225_225755


namespace band_section_student_count_l225_225447

theorem band_section_student_count :
  (0.5 * 500) + (0.12 * 500) + (0.23 * 500) + (0.08 * 500) = 465 :=
by 
  sorry

end band_section_student_count_l225_225447


namespace part_a_part_b_l225_225863

/-- Part (a) statement: -/
theorem part_a (x : Fin 100 → ℕ) :
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) →
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) :=
by
  sorry

/-- Part (b) statement: -/
theorem part_b (x : Fin 100 → ℕ) :
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) →
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) :=
by
  sorry

end part_a_part_b_l225_225863


namespace ordered_pairs_bound_l225_225390

variable (m n : ℕ) (a b : ℕ → ℝ)

theorem ordered_pairs_bound
  (h_m : m ≥ n)
  (h_n : n ≥ 2022)
  : (∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ i j, (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ |a i + b j - (i * j)| ≤ m) ∧
      pairs.card ≤ 3 * n * Real.sqrt (m * Real.log (n))) := 
  sorry

end ordered_pairs_bound_l225_225390


namespace hexagonal_prism_sum_maximum_l225_225740

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum_l225_225740


namespace part_one_part_two_l225_225293

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem part_one :
  ∀ x m : ℕ, f x ≤ -m^2 + 6 * m → 1 ≤ m ∧ m ≤ 5 := 
by
  sorry

theorem part_two (a b c : ℝ) (h : 3 * a + 4 * b + 5 * c = 1) :
  (a^2 + b^2 + c^2) ≥ (1 / 50) :=
by
  sorry

end part_one_part_two_l225_225293


namespace unsatisfactory_tests_l225_225401

theorem unsatisfactory_tests {n k : ℕ} (h1 : n < 50) 
  (h2 : n % 7 = 0) 
  (h3 : n % 3 = 0) 
  (h4 : n % 2 = 0)
  (h5 : n = 7 * (n / 7) + 3 * (n / 3) + 2 * (n / 2) + k) : 
  k = 1 := 
by 
  sorry

end unsatisfactory_tests_l225_225401


namespace arithmetic_expression_equals_fraction_l225_225434

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l225_225434


namespace smallest_four_digit_divisible_by_4_and_5_l225_225927

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l225_225927


namespace soccer_games_per_month_l225_225541

theorem soccer_games_per_month (total_games : ℕ) (months : ℕ) (h1 : total_games = 27) (h2 : months = 3) : total_games / months = 9 :=
by 
  sorry

end soccer_games_per_month_l225_225541


namespace solve_quadratic_roots_l225_225340

theorem solve_quadratic_roots (x : ℝ) : (x - 3) ^ 2 = 3 - x ↔ x = 3 ∨ x = 2 :=
by
  sorry

end solve_quadratic_roots_l225_225340


namespace at_least_one_not_less_than_two_l225_225642

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l225_225642


namespace cashier_amount_l225_225069

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l225_225069


namespace scientific_notation_of_300_million_l225_225043

theorem scientific_notation_of_300_million : 
  300000000 = 3 * 10^8 := 
by
  sorry

end scientific_notation_of_300_million_l225_225043


namespace inequality_proof_l225_225624

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (habc : a * b * (1 / (a * b)) = 1) :
  a^2 + b^2 + (1 / (a * b))^2 + 3 ≥ 2 * (1 / a + 1 / b + a * b) := 
by sorry

end inequality_proof_l225_225624


namespace half_radius_y_l225_225533

theorem half_radius_y (r_x r_y : ℝ) (hx : 2 * Real.pi * r_x = 12 * Real.pi) (harea : Real.pi * r_x ^ 2 = Real.pi * r_y ^ 2) : r_y / 2 = 3 := by
  sorry

end half_radius_y_l225_225533


namespace sec_120_eq_neg_2_l225_225118

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2_l225_225118


namespace simplify_and_evaluate_expression_l225_225604

theorem simplify_and_evaluate_expression :
  (2 * (-1/2) + 3 * 1)^2 - (2 * (-1/2) + 1) * (2 * (-1/2) - 1) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l225_225604


namespace rate_per_sq_meter_is_900_l225_225230

/-- The length of the room L is 7 (meters). -/
def L : ℝ := 7

/-- The width of the room W is 4.75 (meters). -/
def W : ℝ := 4.75

/-- The total cost of paving the floor is Rs. 29,925. -/
def total_cost : ℝ := 29925

/-- The rate per square meter for the slabs is Rs. 900. -/
theorem rate_per_sq_meter_is_900 :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_is_900_l225_225230


namespace first_percentage_increase_l225_225498

theorem first_percentage_increase (x : ℝ) :
  (1 + x / 100) * 1.4 = 1.82 → x = 30 := 
by 
  intro h
  -- start your proof here
  sorry

end first_percentage_increase_l225_225498


namespace quadratic_roots_interlace_l225_225364

variable (p1 p2 q1 q2 : ℝ)

theorem quadratic_roots_interlace
(h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + p1 * r1 + q1 = 0 ∧ r2^2 + p1 * r2 + q1 = 0)) ∧
  (∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + p2 * s1 + q2 = 0 ∧ s2^2 + p2 * s2 + q2 = 0)) ∧
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
  (a^2 + p1*a + q1 = 0 ∧ b^2 + p2*b + q2 = 0 ∧ c^2 + p1*c + q1 = 0 ∧ d^2 + p2*d + q2 = 0)) := 
sorry

end quadratic_roots_interlace_l225_225364


namespace ajax_weight_after_two_weeks_l225_225036

/-- Initial weight of Ajax in kilograms. -/
def initial_weight_kg : ℝ := 80

/-- Conversion factor from kilograms to pounds. -/
def kg_to_pounds : ℝ := 2.2

/-- Weight lost per hour of each exercise type. -/
def high_intensity_loss_per_hour : ℝ := 4
def moderate_intensity_loss_per_hour : ℝ := 2.5
def low_intensity_loss_per_hour : ℝ := 1.5

/-- Ajax's weekly exercise routine. -/
def weekly_high_intensity_hours : ℝ := 1 * 3 + 1.5 * 1
def weekly_moderate_intensity_hours : ℝ := 0.5 * 5
def weekly_low_intensity_hours : ℝ := 1 * 2 + 0.5 * 1

/-- Calculate the total weight loss in pounds per week. -/
def total_weekly_weight_loss_pounds : ℝ :=
  weekly_high_intensity_hours * high_intensity_loss_per_hour +
  weekly_moderate_intensity_hours * moderate_intensity_loss_per_hour +
  weekly_low_intensity_hours * low_intensity_loss_per_hour

/-- Calculate the total weight loss in pounds for two weeks. -/
def total_weight_loss_pounds_for_two_weeks : ℝ :=
  total_weekly_weight_loss_pounds * 2

/-- Calculate Ajax's initial weight in pounds. -/
def initial_weight_pounds : ℝ :=
  initial_weight_kg * kg_to_pounds

/-- Calculate Ajax's new weight after two weeks. -/
def new_weight_pounds : ℝ :=
  initial_weight_pounds - total_weight_loss_pounds_for_two_weeks

/-- Prove that Ajax's new weight in pounds is 120 after following the workout schedule for two weeks. -/
theorem ajax_weight_after_two_weeks :
  new_weight_pounds = 120 :=
by
  sorry

end ajax_weight_after_two_weeks_l225_225036


namespace percentage_of_nine_hundred_l225_225680

theorem percentage_of_nine_hundred : (45 * 8 = 360) ∧ ((360 / 900) * 100 = 40) :=
by
  have h1 : 45 * 8 = 360 := by sorry
  have h2 : (360 / 900) * 100 = 40 := by sorry
  exact ⟨h1, h2⟩

end percentage_of_nine_hundred_l225_225680


namespace problem_statement_l225_225184

theorem problem_statement : 
  (∀ (base : ℤ) (exp : ℕ), (-3) = base ∧ 2 = exp → (base ^ exp ≠ -9)) :=
by
  sorry

end problem_statement_l225_225184


namespace total_eyes_l225_225274

def boys := 23
def girls := 18
def cats := 10
def spiders := 5

def boy_eyes := 2
def girl_eyes := 2
def cat_eyes := 2
def spider_eyes := 8

theorem total_eyes : (boys * boy_eyes) + (girls * girl_eyes) + (cats * cat_eyes) + (spiders * spider_eyes) = 142 := by
  sorry

end total_eyes_l225_225274


namespace correct_calculation_l225_225707

theorem correct_calculation (N : ℤ) (h : 41 - N = 12) : 41 + N = 70 := 
by 
  sorry

end correct_calculation_l225_225707


namespace average_value_l225_225025

variable (z : ℝ)

theorem average_value : (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 :=
by
  sorry

end average_value_l225_225025


namespace proof_problem_l225_225667

def otimes (a b : ℕ) : ℕ := (a^2 - b) / (a - b)

theorem proof_problem : otimes (otimes 7 5) 2 = 24 := by
  sorry

end proof_problem_l225_225667


namespace team_win_percentage_remaining_l225_225618

theorem team_win_percentage_remaining (won_first_30: ℝ) (total_games: ℝ) (total_wins: ℝ)
  (h1: won_first_30 = 0.40 * 30)
  (h2: total_games = 120)
  (h3: total_wins = 0.70 * total_games) :
  (total_wins - won_first_30) / (total_games - 30) * 100 = 80 :=
by
  sorry


end team_win_percentage_remaining_l225_225618


namespace least_number_to_add_l225_225161

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 1100 → d = 23 → r = n % d → (r ≠ 0) → (d - r) = 4 :=
by
  intros h₀ h₁ h₂ h₃
  simp [h₀, h₁] at h₂
  sorry

end least_number_to_add_l225_225161


namespace teacher_selection_l225_225987

/-- A school has 150 teachers, including 15 senior teachers, 45 intermediate teachers, 
and 90 junior teachers. By stratified sampling, 30 teachers are selected to 
participate in the teachers' representative conference. 
--/

def total_teachers : ℕ := 150
def senior_teachers : ℕ := 15
def intermediate_teachers : ℕ := 45
def junior_teachers : ℕ := 90

def total_selected_teachers : ℕ := 30
def selected_senior_teachers : ℕ := 3
def selected_intermediate_teachers : ℕ := 9
def selected_junior_teachers : ℕ := 18

def ratio (a b : ℕ) : ℕ × ℕ := (a / (gcd a b), b / (gcd a b))

theorem teacher_selection :
  ratio senior_teachers (gcd senior_teachers total_teachers) = ratio intermediate_teachers (gcd intermediate_teachers total_teachers) ∧
  ratio intermediate_teachers (gcd intermediate_teachers total_teachers) = ratio junior_teachers (gcd junior_teachers total_teachers) →
  selected_senior_teachers / selected_intermediate_teachers / selected_junior_teachers = 1 / 3 / 6 → 
  selected_senior_teachers + selected_intermediate_teachers + selected_junior_teachers = 30 :=
sorry

end teacher_selection_l225_225987


namespace reach_14_from_458_l225_225543

def double (n : ℕ) : ℕ :=
  n * 2

def erase_last_digit (n : ℕ) : ℕ :=
  n / 10

def can_reach (start target : ℕ) (ops : List (ℕ → ℕ)) : Prop :=
  ∃ seq : List (ℕ → ℕ), seq = ops ∧
    seq.foldl (fun acc f => f acc) start = target

-- The proof problem statement
theorem reach_14_from_458 : can_reach 458 14 [double, erase_last_digit, double, double, erase_last_digit, double, double, erase_last_digit] :=
  sorry

end reach_14_from_458_l225_225543


namespace probability_of_target_hit_l225_225214

theorem probability_of_target_hit  :
  let A_hits := 0.9
  let B_hits := 0.8
  ∃ (P_A P_B : ℝ), 
  P_A = A_hits ∧ P_B = B_hits ∧ 
  (∀ events_independent : Prop, 
   events_independent → P_A * P_B = (0.1) * (0.2)) →
  1 - (0.1 * 0.2) = 0.98
:= 
  sorry

end probability_of_target_hit_l225_225214


namespace pump_no_leak_fill_time_l225_225774

noncomputable def pump_fill_time (P t l : ℝ) :=
  1 / P - 1 / l = 1 / t

theorem pump_no_leak_fill_time :
  ∃ P : ℝ, pump_fill_time P (13 / 6) 26 ∧ P = 2 :=
by
  sorry

end pump_no_leak_fill_time_l225_225774


namespace nate_cooking_for_people_l225_225802

/-- Given that 8 jumbo scallops weigh one pound, scallops cost $24.00 per pound, Nate is pairing 2 scallops with a corn bisque per person, and he spends $48 on scallops. We want to prove that Nate is cooking for 8 people. -/
theorem nate_cooking_for_people :
  (8 : ℕ) = 8 →
  (24 : ℕ) = 24 →
  (2 : ℕ) = 2 →
  (48 : ℕ) = 48 →
  let scallops_per_pound := 8
  let cost_per_pound := 24
  let scallops_per_person := 2
  let money_spent := 48
  let pounds_of_scallops := money_spent / cost_per_pound
  let total_scallops := scallops_per_pound * pounds_of_scallops
  let people := total_scallops / scallops_per_person
  people = 8 :=
by
  sorry

end nate_cooking_for_people_l225_225802


namespace farmer_crops_saved_l225_225882

noncomputable def average_corn_per_row := (10 + 14) / 2
noncomputable def average_potato_per_row := (35 + 45) / 2
noncomputable def average_wheat_per_row := (55 + 65) / 2

noncomputable def avg_reduction_corn := (40 + 60 + 25) / 3 / 100
noncomputable def avg_reduction_potato := (50 + 30 + 60) / 3 / 100
noncomputable def avg_reduction_wheat := (20 + 55 + 35) / 3 / 100

noncomputable def saved_corn_per_row := average_corn_per_row * (1 - avg_reduction_corn)
noncomputable def saved_potato_per_row := average_potato_per_row * (1 - avg_reduction_potato)
noncomputable def saved_wheat_per_row := average_wheat_per_row * (1 - avg_reduction_wheat)

def rows_corn := 30
def rows_potato := 24
def rows_wheat := 36

noncomputable def total_saved_corn := saved_corn_per_row * rows_corn
noncomputable def total_saved_potatoes := saved_potato_per_row * rows_potato
noncomputable def total_saved_wheat := saved_wheat_per_row * rows_wheat

noncomputable def total_crops_saved := total_saved_corn + total_saved_potatoes + total_saved_wheat

theorem farmer_crops_saved : total_crops_saved = 2090 := by
  sorry

end farmer_crops_saved_l225_225882


namespace sally_last_10_shots_made_l225_225602

def sally_initial_shots : ℕ := 30
def sally_initial_success_rate : ℝ := 0.60
def sally_additional_shots : ℕ := 10
def sally_final_success_rate : ℝ := 0.65

theorem sally_last_10_shots_made (x : ℕ) 
  (h1 : sally_initial_success_rate * sally_initial_shots = 18)
  (h2 : sally_final_success_rate * (sally_initial_shots + sally_additional_shots) = 26) :
  x = 8 :=
by
  sorry

end sally_last_10_shots_made_l225_225602


namespace line_intersects_ellipse_slopes_l225_225668

theorem line_intersects_ellipse_slopes (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (1/5)) ∨ m ∈ Set.Ici (Real.sqrt (1/5)) :=
by
  sorry

end line_intersects_ellipse_slopes_l225_225668


namespace ammonium_chloride_reacts_with_potassium_hydroxide_l225_225200

/-- Prove that 1 mole of ammonium chloride is required to react with 
    1 mole of potassium hydroxide to form 1 mole of ammonia, 
    1 mole of water, and 1 mole of potassium chloride, 
    given the balanced chemical equation:
    NH₄Cl + KOH → NH₃ + H₂O + KCl
-/
theorem ammonium_chloride_reacts_with_potassium_hydroxide :
    ∀ (NH₄Cl KOH NH₃ H₂O KCl : ℕ), 
    (NH₄Cl + KOH = NH₃ + H₂O + KCl) → 
    (NH₄Cl = 1) → 
    (KOH = 1) → 
    (NH₃ = 1) → 
    (H₂O = 1) → 
    (KCl = 1) → 
    NH₄Cl = 1 :=
by
  intros
  sorry

end ammonium_chloride_reacts_with_potassium_hydroxide_l225_225200


namespace expected_value_of_fair_6_sided_die_l225_225805

noncomputable def fair_die_expected_value : ℝ :=
  (1/6) * 1 + (1/6) * 2 + (1/6) * 3 + (1/6) * 4 + (1/6) * 5 + (1/6) * 6

theorem expected_value_of_fair_6_sided_die : fair_die_expected_value = 3.5 := by
  sorry

end expected_value_of_fair_6_sided_die_l225_225805


namespace F_3_f_5_eq_24_l225_225234

def f (a : ℤ) : ℤ := a - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem F_3_f_5_eq_24 : F 3 (f 5) = 24 := by
  sorry

end F_3_f_5_eq_24_l225_225234


namespace min_sum_xy_l225_225959

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l225_225959


namespace min_value_x_2y_l225_225792

theorem min_value_x_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y + 2*x*y = 8) : x + 2*y ≥ 4 :=
sorry

end min_value_x_2y_l225_225792


namespace g_1000_is_1820_l225_225902

-- Definitions and conditions from the problem
def g (n : ℕ) : ℕ := sorry -- exact definition is unknown, we will assume conditions

-- Conditions as given
axiom g_g (n : ℕ) : g (g n) = 3 * n
axiom g_3n_plus_1 (n : ℕ) : g (3 * n + 1) = 3 * n + 2

-- Statement to prove
theorem g_1000_is_1820 : g 1000 = 1820 :=
by
  sorry

end g_1000_is_1820_l225_225902


namespace power_function_half_value_l225_225856

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_half_value (a : ℝ) (h : (f 4 a) / (f 2 a) = 3) :
  f (1 / 2) a = 1 / 3 :=
by
  sorry  -- Proof goes here

end power_function_half_value_l225_225856


namespace sum_of_positive_integers_l225_225928

theorem sum_of_positive_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 272) : x + y = 32 := 
by 
  sorry

end sum_of_positive_integers_l225_225928


namespace perimeter_of_figure_composed_of_squares_l225_225552

theorem perimeter_of_figure_composed_of_squares
  (n : ℕ)
  (side_length : ℝ)
  (square_perimeter : ℝ := 4 * side_length)
  (total_squares : ℕ := 7)
  (total_perimeter_if_independent : ℝ := square_perimeter * total_squares)
  (meet_at_vertices : ∀ i j : ℕ, i ≠ j → ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ¬(s1 = s2))
  : total_perimeter_if_independent = 28 :=
by sorry

end perimeter_of_figure_composed_of_squares_l225_225552


namespace younger_age_is_12_l225_225724

theorem younger_age_is_12 
  (y elder : ℕ)
  (h_diff : elder = y + 20)
  (h_past : elder - 7 = 5 * (y - 7)) :
  y = 12 :=
by
  sorry

end younger_age_is_12_l225_225724


namespace geom_series_sum_l225_225044

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end geom_series_sum_l225_225044


namespace total_eggs_needed_l225_225808

-- Define the conditions
def eggsFromAndrew : ℕ := 155
def eggsToBuy : ℕ := 67

-- Define the total number of eggs
def totalEggs : ℕ := eggsFromAndrew + eggsToBuy

-- The theorem to be proven
theorem total_eggs_needed : totalEggs = 222 := by
  sorry

end total_eggs_needed_l225_225808


namespace gross_profit_percentage_l225_225646

theorem gross_profit_percentage (sales_price gross_profit cost : ℝ) 
  (h1 : sales_price = 81) 
  (h2 : gross_profit = 51) 
  (h3 : cost = sales_price - gross_profit) : 
  (gross_profit / cost) * 100 = 170 := 
by
  simp [h1, h2, h3]
  sorry

end gross_profit_percentage_l225_225646


namespace smallest_M_satisfying_conditions_l225_225652

theorem smallest_M_satisfying_conditions :
  ∃ M : ℕ, M > 0 ∧ M = 250 ∧
    ( (M % 125 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 8 = 0)) ∨
      (M % 8 = 0 ∧ ((M + 1) % 125 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 125 = 0)) ∨
      (M % 9 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 125 = 0) ∨ ((M + 1) % 125 = 0 ∧ (M + 2) % 8 = 0)) ) :=
by
  sorry

end smallest_M_satisfying_conditions_l225_225652


namespace min_time_to_shoe_horses_l225_225705

-- Definitions based on the conditions
def n_blacksmiths : ℕ := 48
def n_horses : ℕ := 60
def t_hoof : ℕ := 5 -- minutes per hoof
def n_hooves : ℕ := n_horses * 4
def total_time : ℕ := n_hooves * t_hoof
def t_min : ℕ := total_time / n_blacksmiths

-- The theorem states that the minimal time required is 25 minutes
theorem min_time_to_shoe_horses : t_min = 25 := by
  sorry

end min_time_to_shoe_horses_l225_225705


namespace determine_c_l225_225427

theorem determine_c (c : ℝ) (r : ℝ) (h1 : 2 * r^2 - 8 * r - c = 0) (h2 : r ≠ 0) (h3 : 2 * (r + 5.5)^2 + 5 * (r + 5.5) = c) :
  c = 12 :=
sorry

end determine_c_l225_225427


namespace emily_lives_total_l225_225784

variable (x : ℤ)

def total_lives_after_stages (x : ℤ) : ℤ :=
  let lives_after_stage1 := x + 25
  let lives_after_stage2 := lives_after_stage1 + 24
  let lives_after_stage3 := lives_after_stage2 + 15
  lives_after_stage3

theorem emily_lives_total : total_lives_after_stages x = x + 64 := by
  -- The proof will go here
  sorry

end emily_lives_total_l225_225784


namespace tax_free_amount_correct_l225_225198

-- Definitions based on the problem conditions
def total_value : ℝ := 1720
def tax_paid : ℝ := 78.4
def tax_rate : ℝ := 0.07

-- Definition of the tax-free amount we need to prove
def tax_free_amount : ℝ := 600

-- Main theorem to prove
theorem tax_free_amount_correct : 
  ∃ X : ℝ, 0.07 * (total_value - X) = tax_paid ∧ X = tax_free_amount :=
by 
  use 600
  simp
  sorry

end tax_free_amount_correct_l225_225198


namespace sum_of_cubes_of_roots_l225_225643

theorem sum_of_cubes_of_roots :
  ∀ (x1 x2 : ℝ), (2 * x1^2 - 5 * x1 + 1 = 0) ∧ (2 * x2^2 - 5 * x2 + 1 = 0) →
  (x1 + x2 = 5 / 2) ∧ (x1 * x2 = 1 / 2) →
  (x1^3 + x2^3 = 95 / 8) :=
by
  sorry

end sum_of_cubes_of_roots_l225_225643


namespace sculpture_height_l225_225357

theorem sculpture_height (base_height : ℕ) (total_height_ft : ℝ) (inches_per_foot : ℕ) 
  (h1 : base_height = 8) (h2 : total_height_ft = 3.5) (h3 : inches_per_foot = 12) : 
  (total_height_ft * inches_per_foot - base_height) = 34 := 
by
  sorry

end sculpture_height_l225_225357


namespace parallel_lines_condition_l225_225491

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → (a - 2) * x + 3 * y + 2 * a = 0 → False) ↔ a = -1 :=
sorry

end parallel_lines_condition_l225_225491


namespace vector_addition_l225_225279

def a : ℝ × ℝ := (5, -3)
def b : ℝ × ℝ := (-6, 4)

theorem vector_addition : a + b = (-1, 1) := by
  rw [a, b]
  sorry

end vector_addition_l225_225279


namespace polynomial_solution_l225_225947

theorem polynomial_solution (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) : x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1 = 0 := 
by {
  sorry
}

end polynomial_solution_l225_225947


namespace integer_solutions_of_quadratic_eq_l225_225503

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l225_225503


namespace compute_x_over_w_l225_225514

theorem compute_x_over_w (w x y z : ℚ) (hw : w ≠ 0)
  (h1 : (x + 6 * y - 3 * z) / (-3 * x + 4 * w) = 2 / 3)
  (h2 : (-2 * y + z) / (x - w) = 2 / 3) :
  x / w = 2 / 3 :=
sorry

end compute_x_over_w_l225_225514


namespace relationship_between_M_and_N_l225_225300
   
   variable (x : ℝ)
   def M := 2*x^2 - 12*x + 15
   def N := x^2 - 8*x + 11
   
   theorem relationship_between_M_and_N : M x ≥ N x :=
   by
     sorry
   
end relationship_between_M_and_N_l225_225300


namespace arithmetic_sequence_a5_value_l225_225323

variable (a : ℕ → ℝ)
variable (a_2 a_5 a_8 : ℝ)
variable (h1 : a 2 + a 8 = 15 - a 5)

/-- In an arithmetic sequence {a_n}, given that a_2 + a_8 = 15 - a_5, prove that a_5 equals 5. -/ 
theorem arithmetic_sequence_a5_value (h1 : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l225_225323


namespace simplify_expression_l225_225356

theorem simplify_expression (x y : ℤ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := 
by 
  sorry 

end simplify_expression_l225_225356


namespace total_items_deleted_l225_225941

-- Define the initial conditions
def initial_apps : Nat := 17
def initial_files : Nat := 21
def remaining_apps : Nat := 3
def remaining_files : Nat := 7
def transferred_files : Nat := 4

-- Prove the total number of deleted items
theorem total_items_deleted : (initial_apps - remaining_apps) + (initial_files - (remaining_files + transferred_files)) = 24 :=
by
  sorry

end total_items_deleted_l225_225941


namespace smallest_number_of_contestants_solving_all_problems_l225_225235

theorem smallest_number_of_contestants_solving_all_problems
    (total_contestants : ℕ)
    (solve_first : ℕ)
    (solve_second : ℕ)
    (solve_third : ℕ)
    (solve_fourth : ℕ)
    (H1 : total_contestants = 100)
    (H2 : solve_first = 90)
    (H3 : solve_second = 85)
    (H4 : solve_third = 80)
    (H5 : solve_fourth = 75)
  : ∃ n, n = 30 := by
  sorry

end smallest_number_of_contestants_solving_all_problems_l225_225235


namespace part_one_solution_set_part_two_m_range_l225_225028

theorem part_one_solution_set (m : ℝ) (x : ℝ) (h : m = 0) : ((m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

theorem part_two_m_range (m : ℝ) : (∀ x : ℝ, (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0) ↔ (1 ≤ m ∧ m < 9) :=
by
  sorry

end part_one_solution_set_part_two_m_range_l225_225028


namespace train_length_proof_l225_225368

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end train_length_proof_l225_225368


namespace gcd_poly_l225_225014

-- Defining the conditions as stated in part a:
def is_even_multiple_of_1171 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 1171 * k * 2

-- Stating the main theorem based on the conditions and required proof in part c:
theorem gcd_poly (b : ℤ) (h : is_even_multiple_of_1171 b) : Int.gcd (3 * b ^ 2 + 47 * b + 79) (b + 17) = 1 := by
  sorry

end gcd_poly_l225_225014


namespace line_does_not_pass_through_second_quadrant_l225_225812

theorem line_does_not_pass_through_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ x - y - a^2 = 0 := 
by
  sorry

end line_does_not_pass_through_second_quadrant_l225_225812


namespace montoya_food_budget_l225_225598

theorem montoya_food_budget (g t e : ℝ) (h1 : g = 0.6) (h2 : t = 0.8) : e = 0.2 :=
by sorry

end montoya_food_budget_l225_225598


namespace division_of_fraction_simplified_l225_225923

theorem division_of_fraction_simplified :
  12 / (2 / (5 - 3)) = 12 := 
by
  sorry

end division_of_fraction_simplified_l225_225923


namespace even_numbers_average_l225_225194

theorem even_numbers_average (n : ℕ) (h1 : 2 * (n * (n + 1)) = 22 * n) : n = 10 :=
by
  sorry

end even_numbers_average_l225_225194


namespace large_pizza_slices_l225_225002

variable (L : ℕ)

theorem large_pizza_slices :
  (2 * L + 2 * 8 = 48) → (L = 16) :=
by 
  sorry

end large_pizza_slices_l225_225002


namespace reciprocal_of_negative_2023_l225_225678

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l225_225678


namespace sufficient_but_not_necessary_l225_225720

-- Let's define the conditions and the theorem to be proved in Lean 4
theorem sufficient_but_not_necessary : ∀ x : ℝ, (x > 1 → x > 0) ∧ ¬(∀ x : ℝ, x > 0 → x > 1) := by
  sorry

end sufficient_but_not_necessary_l225_225720


namespace sum_of_interior_angles_of_remaining_polygon_l225_225484

theorem sum_of_interior_angles_of_remaining_polygon (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 5) :
  (n - 2) * 180 ≠ 270 :=
by 
  sorry

end sum_of_interior_angles_of_remaining_polygon_l225_225484


namespace license_plate_count_l225_225395

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let second_char_options := letters - 1 + digits
  let third_char_options := digits - 1
  letters * second_char_options * third_char_options = 8190 :=
by
  sorry

end license_plate_count_l225_225395


namespace johns_running_hours_l225_225711

-- Define the conditions
variable (x : ℕ) -- let x represent the number of hours at 8 mph and 6 mph
variable (total_hours : ℕ) (total_distance : ℕ)
variable (speed_8 : ℕ) (speed_6 : ℕ) (speed_5 : ℕ)
variable (distance_8 : ℕ := speed_8 * x)
variable (distance_6 : ℕ := speed_6 * x)
variable (distance_5 : ℕ := speed_5 * (total_hours - 2 * x))

-- Total hours John completes the marathon
axiom h1: total_hours = 15

-- Total distance John completes in miles
axiom h2: total_distance = 95

-- Speed factors
axiom h3: speed_8 = 8
axiom h4: speed_6 = 6
axiom h5: speed_5 = 5

-- Distance equation
axiom h6: distance_8 + distance_6 + distance_5 = total_distance

-- Prove the number of hours John ran at each speed
theorem johns_running_hours : x = 5 :=
by
  sorry

end johns_running_hours_l225_225711


namespace milk_cost_correct_l225_225940

-- Definitions of the given conditions
def bagelCost : ℝ := 0.95
def orangeJuiceCost : ℝ := 0.85
def sandwichCost : ℝ := 4.65
def lunchExtraCost : ℝ := 4.0

-- Total cost of breakfast
def breakfastCost : ℝ := bagelCost + orangeJuiceCost

-- Total cost of lunch
def lunchCost : ℝ := breakfastCost + lunchExtraCost

-- Cost of milk
def milkCost : ℝ := lunchCost - sandwichCost

-- Theorem to prove the cost of milk
theorem milk_cost_correct : milkCost = 1.15 :=
by
  sorry

end milk_cost_correct_l225_225940


namespace sqrt_expression_l225_225996

theorem sqrt_expression : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end sqrt_expression_l225_225996


namespace angles_sum_correct_l225_225818

-- Definitions from the problem conditions
def identicalSquares (n : Nat) := n = 13

variable (α β γ δ ε ζ η θ : ℝ) -- Angles of interest

def anglesSum :=
  (α + β + γ + δ) + (ε + ζ + η + θ)

-- Lean 4 statement
theorem angles_sum_correct
  (h₁ : identicalSquares 13)
  (h₂ : α = 90) (h₃ : β = 90) (h₄ : γ = 90) (h₅ : δ = 90)
  (h₆ : ε = 90) (h₇ : ζ = 90) (h₈ : η = 45) (h₉ : θ = 45) :
  anglesSum α β γ δ ε ζ η θ = 405 :=
by
  simp [anglesSum]
  sorry

end angles_sum_correct_l225_225818


namespace find_a_b_find_extreme_point_g_num_zeros_h_l225_225512

-- (1) Proving the values of a and b
theorem find_a_b (a b : ℝ)
  (h1 : (3 + 2 * a + b = 0))
  (h2 : (3 - 2 * a + b = 0)) : 
  a = 0 ∧ b = -3 :=
sorry

-- (2) Proving the extreme points of g(x)
theorem find_extreme_point_g (x : ℝ) : 
  x = -2 :=
sorry

-- (3) Proving the number of zeros of h(x)
theorem num_zeros_h (c : ℝ) (h : -2 ≤ c ∧ c ≤ 2) :
  (|c| = 2 → ∃ y, y = 5) ∧ (|c| < 2 → ∃ y, y = 9) :=
sorry

end find_a_b_find_extreme_point_g_num_zeros_h_l225_225512


namespace cost_of_painting_new_room_l225_225926

theorem cost_of_painting_new_room
  (L B H : ℝ)    -- Dimensions of the original room
  (c : ℝ)        -- Cost to paint the original room
  (h₁ : c = 350) -- Given that the cost of painting the original room is Rs. 350
  (A : ℝ)        -- Area of the walls of the original room
  (h₂ : A = 2 * (L + B) * H) -- Given the area calculation for the original room
  (newA : ℝ)     -- Area of the walls of the new room
  (h₃ : newA = 18 * (L + B) * H) -- Given the area calculation for the new room
  : (350 / (2 * (L + B) * H)) * (18 * (L + B) * H) = 3150 :=
by
  sorry

end cost_of_painting_new_room_l225_225926


namespace sum_of_squares_of_coeffs_l225_225021

theorem sum_of_squares_of_coeffs :
  let p := 3 * (x^5 + 5 * x^3 + 2 * x + 1)
  let coeffs := [3, 15, 6, 3]
  coeffs.map (λ c => c^2) |>.sum = 279 := by
  sorry

end sum_of_squares_of_coeffs_l225_225021


namespace unique_prime_triple_l225_225979

/-- A prime is an integer greater than 1 whose only positive integer divisors are itself and 1. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- Prove that the only triple of primes (p, q, r), such that p = q + 2 and q = r + 2 is (7, 5, 3). -/
theorem unique_prime_triple (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  (p = q + 2) ∧ (q = r + 2) → (p = 7 ∧ q = 5 ∧ r = 3) := by
  sorry

end unique_prime_triple_l225_225979


namespace polynomial_zero_iff_divisibility_l225_225582

theorem polynomial_zero_iff_divisibility (P : Polynomial ℤ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, P.eval (2^n) = n * k) ↔ P = 0 :=
by sorry

end polynomial_zero_iff_divisibility_l225_225582


namespace find_ABC_l225_225684

variables (A B C D : ℕ)

-- Conditions
def non_zero_distinct_digits_less_than_7 : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def ab_c_seven : Prop := 
  (A * 7 + B) + C = C * 7

def ab_ba_dc_seven : Prop :=
  (A * 7 + B) + (B * 7 + A) = D * 7 + C

-- Theorem to prove
theorem find_ABC 
  (h1 : non_zero_distinct_digits_less_than_7 A B C) 
  (h2 : ab_c_seven A B C) 
  (h3 : ab_ba_dc_seven A B C D) : 
  A * 100 + B * 10 + C = 516 :=
sorry

end find_ABC_l225_225684


namespace problem_solution_l225_225054

variable (a : ℝ)

theorem problem_solution (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end problem_solution_l225_225054


namespace solve_for_nonzero_x_l225_225746

open Real

theorem solve_for_nonzero_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 :=
by
  sorry

end solve_for_nonzero_x_l225_225746


namespace complement_intersect_A_B_range_of_a_l225_225372

-- Definitions for sets A and B
def setA : Set ℝ := {x | -2 < x ∧ x < 0}
def setB : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- First statement to prove
theorem complement_intersect_A_B : (setAᶜ ∩ setB) = {x | x ≥ 0} :=
  sorry

-- Definition for set C
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

-- Second statement to prove
theorem range_of_a (a : ℝ) : (setC a ⊆ setA) ↔ (a ≤ -1) ∨ (-1 ≤ a ∧ a ≤ -1 / 2) :=
  sorry

end complement_intersect_A_B_range_of_a_l225_225372


namespace equation_of_rotated_translated_line_l225_225655

theorem equation_of_rotated_translated_line (x y : ℝ) :
  (∀ x, y = 3 * x → y = x / -3 + 1 / -3) →
  (∀ x, y = -1/3 * (x - 1)) →
  y = -1/3 * x + 1/3 :=
sorry

end equation_of_rotated_translated_line_l225_225655


namespace sqrt_real_domain_l225_225843

theorem sqrt_real_domain (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 := 
sorry

end sqrt_real_domain_l225_225843


namespace determine_a_l225_225640

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem determine_a (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end determine_a_l225_225640


namespace unique_solution_for_4_circ_20_l225_225894

def operation (x y : ℝ) : ℝ := 3 * x - 2 * y + 2 * x * y

theorem unique_solution_for_4_circ_20 : ∃! y : ℝ, operation 4 y = 20 :=
by 
  sorry

end unique_solution_for_4_circ_20_l225_225894


namespace quincy_more_stuffed_animals_l225_225587

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end quincy_more_stuffed_animals_l225_225587


namespace average_problem_l225_225197

noncomputable def avg2 (a b : ℚ) := (a + b) / 2
noncomputable def avg3 (a b c : ℚ) := (a + b + c) / 3

theorem average_problem :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 :=
by
  sorry

end average_problem_l225_225197


namespace problem_a_problem_d_l225_225303

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d_l225_225303


namespace find_segment_XY_length_l225_225943

theorem find_segment_XY_length (A B C D X Y : Type) 
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq X] [DecidableEq Y]
  (line_l : Type) (BX : ℝ) (DY : ℝ) (AB : ℝ) (BC : ℝ) (l : line_l)
  (hBX : BX = 4) (hDY : DY = 10) (hBC : BC = 2 * AB) :
  XY = 13 :=
  sorry

end find_segment_XY_length_l225_225943


namespace positive_difference_eq_250_l225_225056

-- Definition of the sum of the first n positive even integers
def sum_first_n_evens (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2)

-- Definition of the sum of the first n positive odd integers
def sum_first_n_odds (n : ℕ) : ℕ :=
  n * n

-- Definition of the positive difference between the sum of the first 25 positive even integers
-- and the sum of the first 20 positive odd integers
def positive_difference : ℕ :=
  (sum_first_n_evens 25) - (sum_first_n_odds 20)

-- The theorem we need to prove
theorem positive_difference_eq_250 : positive_difference = 250 :=
  by
    -- Sorry allows us to skip the proof while ensuring the code compiles.
    sorry

end positive_difference_eq_250_l225_225056


namespace value_of_f_g_3_l225_225488

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x + 2

theorem value_of_f_g_3 : f (g 3) = 83 := by
  sorry

end value_of_f_g_3_l225_225488


namespace meaning_of_a2_add_b2_ne_zero_l225_225468

theorem meaning_of_a2_add_b2_ne_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end meaning_of_a2_add_b2_ne_zero_l225_225468


namespace parabola_cubic_intersection_points_l225_225142

def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15

def cubic (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem parabola_cubic_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    p1 = (-1, 0) ∧ p2 = (1, -24) ∧ p3 = (9, 162) ∧
    parabola p1.1 = p1.2 ∧ cubic p1.1 = p1.2 ∧
    parabola p2.1 = p2.2 ∧ cubic p2.1 = p2.2 ∧
    parabola p3.1 = p3.2 ∧ cubic p3.1 = p3.2 :=
by {
  -- This is the statement
  sorry
}

end parabola_cubic_intersection_points_l225_225142


namespace new_average_score_l225_225270

theorem new_average_score (average_initial : ℝ) (total_practices : ℕ) (highest_score lowest_score : ℝ) :
  average_initial = 87 → 
  total_practices = 10 → 
  highest_score = 95 → 
  lowest_score = 55 → 
  ((average_initial * total_practices - highest_score - lowest_score) / (total_practices - 2)) = 90 :=
by
  intros h_avg h_total h_high h_low
  sorry

end new_average_score_l225_225270


namespace find_uncommon_cards_l225_225799

def numRare : ℕ := 19
def numCommon : ℕ := 30
def costRare : ℝ := 1
def costUncommon : ℝ := 0.50
def costCommon : ℝ := 0.25
def totalCostDeck : ℝ := 32

theorem find_uncommon_cards (U : ℕ) (h : U * costUncommon + numRare * costRare + numCommon * costCommon = totalCostDeck) : U = 11 := by
  sorry

end find_uncommon_cards_l225_225799


namespace gcd_3375_9180_l225_225891

-- Definition of gcd and the problem condition
theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry -- Proof can be filled in with the steps using the Euclidean algorithm

end gcd_3375_9180_l225_225891


namespace find_minimum_value_of_f_l225_225092

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l225_225092


namespace model_height_l225_225681

noncomputable def H_actual : ℝ := 50
noncomputable def A_actual : ℝ := 25
noncomputable def A_model : ℝ := 0.025

theorem model_height : 
  let ratio := (A_actual / A_model)
  ∃ h : ℝ, h = H_actual / (Real.sqrt ratio) ∧ h = 5 * Real.sqrt 10 := 
by 
  sorry

end model_height_l225_225681


namespace divide_coal_l225_225433

noncomputable def part_of_pile (whole: ℚ) (parts: ℕ) := whole / parts
noncomputable def part_tons (total_tons: ℚ) (fraction: ℚ) := total_tons * fraction

theorem divide_coal (total_tons: ℚ) (parts: ℕ) (h: total_tons = 3 ∧ parts = 5):
  (part_of_pile 1 parts = 1/parts) ∧ (part_tons total_tons (1/parts) = total_tons / parts) :=
by
  sorry

end divide_coal_l225_225433


namespace exists_unique_i_l225_225760

-- Let p be an odd prime number.
variable {p : ℕ} [Fact (Nat.Prime p)] (odd_prime : p % 2 = 1)

-- Let a be an integer in the sequence {2, 3, 4, ..., p-3, p-2}
variable (a : ℕ) (a_range : 2 ≤ a ∧ a ≤ p - 2)

-- Prove that there exists a unique i such that i * a ≡ 1 (mod p) and i ≠ a
theorem exists_unique_i (h1 : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → Nat.gcd k p = 1) :
  ∃! (i : ℕ), 1 ≤ i ∧ i ≤ p - 1 ∧ i * a % p = 1 ∧ i ≠ a :=
by 
  sorry

end exists_unique_i_l225_225760


namespace prove_m_plus_n_eq_one_l225_225869

-- Define coordinates of points A and B
def A (m n : ℝ) : ℝ × ℝ := (1 + m, 1 - n)
def B : ℝ × ℝ := (-3, 2)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  symmetric_about_y_axis (A m n) B

-- Statement to prove
theorem prove_m_plus_n_eq_one (m n : ℝ) (h : conditions m n) : m + n = 1 := 
by 
  sorry

end prove_m_plus_n_eq_one_l225_225869


namespace perimeter_of_equilateral_triangle_l225_225126

theorem perimeter_of_equilateral_triangle (a : ℕ) (h1 : a = 12) (h2 : ∀ sides, sides = 3) : 
  3 * a = 36 := 
by
  sorry

end perimeter_of_equilateral_triangle_l225_225126


namespace number_of_members_l225_225615

variable (n : ℕ)

-- Conditions
def each_member_contributes_n_cents : Prop := n * n = 64736

-- Theorem that relates to the number of members being 254
theorem number_of_members (h : each_member_contributes_n_cents n) : n = 254 :=
sorry

end number_of_members_l225_225615


namespace cos_alpha_add_beta_over_two_l225_225890

theorem cos_alpha_add_beta_over_two (
  α β : ℝ) 
  (h1 : 0 < α ∧ α < (Real.pi / 2)) 
  (h2 : - (Real.pi / 2) < β ∧ β < 0) 
  (hcos1 : Real.cos (α + (Real.pi / 4)) = 1 / 3) 
  (hcos2 : Real.cos ((β / 2) - (Real.pi / 4)) = Real.sqrt 3 / 3) : 
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_two_l225_225890


namespace max_min_of_f_in_M_l225_225803

noncomputable def domain (x : ℝ) : Prop := 3 - 4*x + x^2 > 0

def M : Set ℝ := { x | domain x }

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem max_min_of_f_in_M :
  ∃ (xₘ xₘₐₓ : ℝ), xₘ ∈ M ∧ xₘₐₓ ∈ M ∧ 
  (∀ x ∈ M, f xₘₐₓ ≥ f x) ∧ 
  (∀ x ∈ M, f xₘ ≠ f xₓₐₓ) :=
by
  sorry

end max_min_of_f_in_M_l225_225803


namespace unique_a_for_system_solution_l225_225386

-- Define the variables
variables (a b x y : ℝ)

-- Define the system of equations
def system_has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, 2^(b * x) + (a + 1) * b * y^2 = a^2 ∧ (a-1) * x^3 + y^3 = 1

-- Main theorem statement
theorem unique_a_for_system_solution :
  a = -1 ↔ ∀ b : ℝ, system_has_solution a b :=
sorry

end unique_a_for_system_solution_l225_225386


namespace lion_to_leopard_ratio_l225_225249

variable (L P E : ℕ)

axiom lion_count : L = 200
axiom total_population : L + P + E = 450
axiom elephants_relation : E = (1 / 2 : ℚ) * (L + P)

theorem lion_to_leopard_ratio : L / P = 2 :=
by
  sorry

end lion_to_leopard_ratio_l225_225249


namespace simplify_expression_l225_225864

variable (a b c x y z : ℝ)

theorem simplify_expression :
  (cz * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + bz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cz + bz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cz * a^3 * y^3 + 3 * bz * c^3 * x^3) / (cz + bz) :=
by
  sorry

end simplify_expression_l225_225864


namespace find_a_b_of_solution_set_l225_225046

theorem find_a_b_of_solution_set :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + (a + 1) * x + a * b = 0 ↔ x = -1 ∨ x = 4) → a + b = -3 :=
by
  sorry

end find_a_b_of_solution_set_l225_225046


namespace evaluate_fraction_l225_225180

theorem evaluate_fraction : (8 / 29) - (5 / 87) = (19 / 87) := sorry

end evaluate_fraction_l225_225180


namespace beef_weight_after_processing_l225_225172

theorem beef_weight_after_processing
  (initial_weight : ℝ)
  (weight_loss_percentage : ℝ)
  (processed_weight : ℝ)
  (h1 : initial_weight = 892.31)
  (h2 : weight_loss_percentage = 0.35)
  (h3 : processed_weight = initial_weight * (1 - weight_loss_percentage)) :
  processed_weight = 579.5015 :=
by
  sorry

end beef_weight_after_processing_l225_225172


namespace min_value_x_plus_inv_x_l225_225101

open Real

theorem min_value_x_plus_inv_x (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2 := by
  sorry

end min_value_x_plus_inv_x_l225_225101


namespace find_C_monthly_income_l225_225965

theorem find_C_monthly_income (A_m B_m C_m : ℝ) (h1 : A_m / B_m = 5 / 2) (h2 : B_m = 1.12 * C_m) (h3 : 12 * A_m = 504000) : C_m = 15000 :=
sorry

end find_C_monthly_income_l225_225965


namespace find_b_l225_225026

-- Definitions from the conditions
variables (a b : ℝ)

-- Theorem statement using the conditions and the correct answer
theorem find_b (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
by
  sorry

end find_b_l225_225026


namespace agreed_upon_service_period_l225_225037

theorem agreed_upon_service_period (x : ℕ) (hx : 900 + 100 = 1000) 
(assumed_service : x * 1000 = 9 * (650 + 100)) :
  x = 12 :=
by {
  sorry
}

end agreed_upon_service_period_l225_225037


namespace complete_square_l225_225315

theorem complete_square :
  (∀ x: ℝ, 2 * x^2 - 4 * x + 1 = 2 * (x - 1)^2 - 1) := 
by
  intro x
  sorry

end complete_square_l225_225315


namespace incorrect_reasoning_form_l225_225199

-- Define what it means to be a rational number
def is_rational (x : ℚ) : Prop := true

-- Define what it means to be a fraction
def is_fraction (x : ℚ) : Prop := true

-- Define what it means to be an integer
def is_integer (x : ℤ) : Prop := true

-- State the premises as hypotheses
theorem incorrect_reasoning_form (h1 : ∃ x : ℚ, is_rational x ∧ is_fraction x)
                                 (h2 : ∀ z : ℤ, is_rational z) :
  ¬ (∀ z : ℤ, is_fraction z) :=
by
  -- We are stating the conclusion as a hypothesis that needs to be proven incorrect
  sorry

end incorrect_reasoning_form_l225_225199


namespace cos_alpha_value_l225_225513

theorem cos_alpha_value (α β : Real) (hα1 : 0 < α) (hα2 : α < π / 2) 
    (hβ1 : π / 2 < β) (hβ2 : β < π) (hcosβ : Real.cos β = -1/3)
    (hsin_alpha_beta : Real.sin (α + β) = 1/3) : 
    Real.cos α = 4 * Real.sqrt 2 / 9 := by
  sorry

end cos_alpha_value_l225_225513


namespace consecutive_ints_prod_square_l225_225121

theorem consecutive_ints_prod_square (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
sorry

end consecutive_ints_prod_square_l225_225121


namespace intersection_of_A_and_B_l225_225311

noncomputable def A := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_l225_225311


namespace sulfuric_acid_moles_l225_225059

-- Definitions based on the conditions
def iron_moles := 2
def hydrogen_moles := 2

-- The reaction equation in the problem
def reaction (Fe H₂SO₄ : ℕ) : Prop :=
  Fe + H₂SO₄ = hydrogen_moles

-- Goal: prove the number of moles of sulfuric acid used is 2
theorem sulfuric_acid_moles (Fe : ℕ) (H₂SO₄ : ℕ) (h : reaction Fe H₂SO₄) :
  H₂SO₄ = 2 :=
sorry

end sulfuric_acid_moles_l225_225059


namespace area_of_inscribed_hexagon_in_square_is_27sqrt3_l225_225496

noncomputable def side_length_of_triangle : ℝ := 6
noncomputable def radius_of_circle (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2
noncomputable def side_length_of_square (r : ℝ) : ℝ := 2 * r
noncomputable def side_length_of_hexagon_in_square (s : ℝ) : ℝ := s / (Real.sqrt 2)
noncomputable def area_of_hexagon (side_hexagon : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side_hexagon^2

theorem area_of_inscribed_hexagon_in_square_is_27sqrt3 :
  ∀ (a r s side_hex : ℝ), 
    a = side_length_of_triangle →
    r = radius_of_circle a →
    s = side_length_of_square r →
    side_hex = side_length_of_hexagon_in_square s →
    area_of_hexagon side_hex = 27 * Real.sqrt 3 :=
by
  intros a r s side_hex h_a h_r h_s h_side_hex
  sorry

end area_of_inscribed_hexagon_in_square_is_27sqrt3_l225_225496


namespace sara_total_spent_l225_225586

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end sara_total_spent_l225_225586


namespace fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l225_225201

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

theorem fixed_point_when_a_2_b_neg2 :
  (∃ x : ℝ, f 2 (-2) x = x) → (x = -1 ∨ x = 2) :=
sorry

theorem range_of_a_for_two_fixed_points (a : ℝ) :
  (∀ b : ℝ, a ≠ 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2)) → (0 < a ∧ a < 2) :=
sorry

end fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l225_225201


namespace inequality_has_no_solution_l225_225116

theorem inequality_has_no_solution (x : ℝ) : -x^2 + 2*x - 2 > 0 → false :=
by
  sorry

end inequality_has_no_solution_l225_225116


namespace find_y_when_x_is_7_l225_225005

theorem find_y_when_x_is_7
  (x y : ℝ)
  (h1 : x * y = 384)
  (h2 : x + y = 40)
  (h3 : x - y = 8)
  (h4 : x = 7) :
  y = 384 / 7 :=
by
  sorry

end find_y_when_x_is_7_l225_225005


namespace convex_power_function_l225_225248

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end convex_power_function_l225_225248


namespace jack_morning_emails_l225_225098

-- Define the conditions as constants
def totalEmails : ℕ := 10
def emailsAfternoon : ℕ := 3
def emailsEvening : ℕ := 1

-- Problem statement to prove emails in the morning
def emailsMorning : ℕ := totalEmails - (emailsAfternoon + emailsEvening)

-- The theorem to prove
theorem jack_morning_emails : emailsMorning = 6 := by
  sorry

end jack_morning_emails_l225_225098


namespace grayson_unanswered_l225_225901

noncomputable def unanswered_questions : ℕ :=
  let total_questions := 200
  let first_set_questions := 50
  let first_set_time := first_set_questions * 1 -- 1 minute per question
  let second_set_questions := 50
  let second_set_time := second_set_questions * (90 / 60) -- convert 90 seconds to minutes
  let third_set_questions := 25
  let third_set_time := third_set_questions * 2 -- 2 minutes per question
  let total_answered_time := first_set_time + second_set_time + third_set_time
  let total_time_available := 4 * 60 -- 4 hours in minutes 
  let unanswered := total_questions - (first_set_questions + second_set_questions + third_set_questions)
  unanswered

theorem grayson_unanswered : unanswered_questions = 75 := 
by 
  sorry

end grayson_unanswered_l225_225901


namespace area_comparison_l225_225844

def point := (ℝ × ℝ)

def quadrilateral_I_vertices : List point := [(0, 0), (2, 0), (2, 2), (0, 2)]

def quadrilateral_I_area : ℝ := 4

def quadrilateral_II_vertices : List point := [(1, 0), (4, 0), (4, 4), (1, 3)]

noncomputable def quadrilateral_II_area : ℝ := 10.5

theorem area_comparison :
  quadrilateral_I_area < quadrilateral_II_area :=
  by
    sorry

end area_comparison_l225_225844


namespace find_K_l225_225449

theorem find_K (Z K : ℕ)
  (hZ1 : 700 < Z)
  (hZ2 : Z < 1500)
  (hK : K > 1)
  (hZ_eq : Z = K^4)
  (hZ_perfect : ∃ n : ℕ, Z = n^6) :
  K = 3 :=
by
  sorry

end find_K_l225_225449


namespace smallest_arith_prog_l225_225974

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end smallest_arith_prog_l225_225974


namespace find_paintings_l225_225032

noncomputable def cost_painting (P : ℕ) : ℝ := 40 * P
noncomputable def cost_toy : ℝ := 20 * 8
noncomputable def total_cost (P : ℕ) : ℝ := cost_painting P + cost_toy

noncomputable def sell_painting (P : ℕ) : ℝ := 36 * P
noncomputable def sell_toy : ℝ := 17 * 8
noncomputable def total_sell (P : ℕ) : ℝ := sell_painting P + sell_toy

noncomputable def total_loss (P : ℕ) : ℝ := total_cost P - total_sell P

theorem find_paintings : ∀ (P : ℕ), total_loss P = 64 → P = 10 :=
by
  intros P h
  sorry

end find_paintings_l225_225032


namespace slope_of_line_l225_225125

theorem slope_of_line : ∀ (x y : ℝ), (6 * x + 10 * y = 30) → (y = -((3 / 5) * x) + 3) :=
by
  -- Proof needs to be filled out
  sorry

end slope_of_line_l225_225125


namespace probability_A2_l225_225515

-- Define events and their probabilities
variable (A1 : Prop) (A2 : Prop) (B1 : Prop)
variable (P : Prop → ℝ)
variable [MeasureTheory.MeasureSpace ℝ]

-- Conditions given in the problem
axiom P_A1 : P A1 = 0.5
axiom P_B1 : P B1 = 0.5
axiom P_A2_given_A1 : P (A2 ∧ A1) / P A1 = 0.7
axiom P_A2_given_B1 : P (A2 ∧ B1) / P B1 = 0.8

-- Theorem statement to prove
theorem probability_A2 : P A2 = 0.75 :=
by
  -- Skipping the proof as per instructions
  sorry

end probability_A2_l225_225515


namespace sum_of_n_and_k_l225_225520

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l225_225520


namespace total_amount_l225_225047

-- Define the conditions in Lean
variables (X Y Z: ℝ)
variable (h1 : Y = 0.75 * X)
variable (h2 : Z = (2/3) * X)
variable (h3 : Y = 48)

-- The theorem stating that the total amount of money is Rs. 154.67
theorem total_amount (X Y Z : ℝ) (h1 : Y = 0.75 * X) (h2 : Z = (2/3) * X) (h3 : Y = 48) : 
  X + Y + Z = 154.67 := 
by
  sorry

end total_amount_l225_225047


namespace train_crossing_time_l225_225377

def train_length : ℕ := 100  -- length of the train in meters
def bridge_length : ℕ := 180  -- length of the bridge in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph

theorem train_crossing_time 
  (TL : ℕ := train_length) 
  (BL : ℕ := bridge_length) 
  (TSK : ℕ := train_speed_kmph) : 
  (TL + BL) / ((TSK * 1000) / 3600) = 28 := by
  sorry

end train_crossing_time_l225_225377


namespace cubical_tank_water_volume_l225_225796

theorem cubical_tank_water_volume 
    (s : ℝ) -- side length of the cube in feet
    (h_fill : 1 / 4 * s = 1) -- tank is filled to 0.25 of its capacity, water level is 1 foot
    (h_volume_water : 0.25 * (s ^ 3) = 16) -- 0.25 of the tank's total volume is the volume of water
    : s ^ 3 = 64 := 
by
  sorry

end cubical_tank_water_volume_l225_225796


namespace total_weight_of_arrangement_l225_225892

def original_side_length : ℤ := 4
def original_weight : ℤ := 16
def larger_side_length : ℤ := 10

theorem total_weight_of_arrangement :
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  total_weight = 96 :=
by
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  sorry

end total_weight_of_arrangement_l225_225892


namespace domain_of_sqrt_tan_x_minus_sqrt_3_l225_225734

noncomputable def domain_of_function : Set Real :=
  {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

theorem domain_of_sqrt_tan_x_minus_sqrt_3 :
  { x : Real | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2 } = domain_of_function :=
by
  sorry

end domain_of_sqrt_tan_x_minus_sqrt_3_l225_225734


namespace find_ivans_number_l225_225421

theorem find_ivans_number :
  ∃ (a b c d e f g h i j k l : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    1000 ≤ e ∧ e < 10000 ∧
    (a * 10^10 + b * 10^8 + c * 10^6 + d * 10^4 + e) = 132040530321 := sorry

end find_ivans_number_l225_225421


namespace inequality_proof_l225_225003

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + x + 2 * x^2) * (2 + 3 * y + y^2) * (4 + z + z^2) ≥ 60 * x * y * z :=
by
  sorry

end inequality_proof_l225_225003


namespace evaluate_expression_l225_225211

theorem evaluate_expression : (10^9) / ((2 * 10^6) * 3) = 500 / 3 :=
by sorry

end evaluate_expression_l225_225211


namespace total_elephants_l225_225461

-- Define the conditions in Lean
def G (W : ℕ) : ℕ := 3 * W
def N (G : ℕ) : ℕ := 5 * G
def W : ℕ := 70

-- Define the statement to prove
theorem total_elephants :
  G W + W + N (G W) = 1330 :=
by
  -- Proof to be filled in
  sorry

end total_elephants_l225_225461


namespace find_value_of_x_l225_225281

theorem find_value_of_x (w : ℕ) (x y z : ℕ) (h₁ : x = y / 3) (h₂ : y = z / 6) (h₃ : z = 2 * w) (hw : w = 45) : x = 5 :=
by
  sorry

end find_value_of_x_l225_225281


namespace initial_fliers_l225_225457

variable (F : ℕ) -- Initial number of fliers

-- Conditions
axiom morning_send : F - (1 / 5) * F = (4 / 5) * F
axiom afternoon_send : (4 / 5) * F - (1 / 4) * ((4 / 5) * F) = (3 / 5) * F
axiom final_count : (3 / 5) * F = 600

theorem initial_fliers : F = 1000 := by
  sorry

end initial_fliers_l225_225457


namespace six_digit_number_contains_7_l225_225239

theorem six_digit_number_contains_7
  (a b k : ℤ)
  (h1 : 100 ≤ 7 * a + k ∧ 7 * a + k < 1000)
  (h2 : 100 ≤ 7 * b + k ∧ 7 * b + k < 1000) :
  7 ∣ (1000 * (7 * a + k) + (7 * b + k)) :=
by
  sorry

end six_digit_number_contains_7_l225_225239


namespace average_candies_correct_l225_225066

def candy_counts : List ℕ := [16, 22, 30, 26, 18, 20]
def num_members : ℕ := 6
def total_candies : ℕ := List.sum candy_counts
def average_candies : ℕ := total_candies / num_members

theorem average_candies_correct : average_candies = 22 := by
  -- Proof is omitted, as per instructions
  sorry

end average_candies_correct_l225_225066


namespace sequence_sum_l225_225133

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end sequence_sum_l225_225133


namespace popsicle_count_l225_225396

-- Define the number of each type of popsicles
def num_grape_popsicles : Nat := 2
def num_cherry_popsicles : Nat := 13
def num_banana_popsicles : Nat := 2

-- Prove the total number of popsicles
theorem popsicle_count : num_grape_popsicles + num_cherry_popsicles + num_banana_popsicles = 17 := by
  sorry

end popsicle_count_l225_225396


namespace auditorium_total_chairs_l225_225367

theorem auditorium_total_chairs 
  (n : ℕ)
  (h1 : 2 + 5 - 1 = n)   -- n is the number of rows which is equal to 6
  (h2 : 3 + 4 - 1 = n)   -- n is the number of chairs per row which is also equal to 6
  : n * n = 36 :=        -- the total number of chairs is 36
by
  sorry

end auditorium_total_chairs_l225_225367


namespace fraction_addition_l225_225783

theorem fraction_addition :
  (1 / 6) + (1 / 3) + (5 / 9) = 19 / 18 :=
by
  sorry

end fraction_addition_l225_225783


namespace proportion_solution_l225_225296

theorem proportion_solution (x: ℕ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_solution_l225_225296


namespace unique_solution_l225_225381

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
    (∀ x, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
    (∀ x, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

theorem unique_solution (f : ℝ → ℝ) :
    is_solution f → (∀ x, x ≥ 1 → f x = x + 1) := 
sorry

end unique_solution_l225_225381


namespace triangle_third_side_l225_225862

theorem triangle_third_side (x : ℕ) : 
  (3 < x) ∧ (x < 17) → 
  (x = 11) :=
by
  sorry

end triangle_third_side_l225_225862


namespace find_x_l225_225284

theorem find_x (x : ℕ) (h1 : 8^x = 2^9) (h2 : 8 = 2^3) : x = 3 := by
  sorry

end find_x_l225_225284


namespace probability_different_colors_l225_225989

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end probability_different_colors_l225_225989


namespace solution_set_of_gx_lt_0_l225_225578

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f_inv (1 - x) - f_inv (1 + x)

theorem solution_set_of_gx_lt_0 : { x : ℝ | g x < 0 } = Set.Ioo 0 1 := by
  sorry

end solution_set_of_gx_lt_0_l225_225578


namespace units_digit_of_17_pow_3_mul_24_l225_225389

def unit_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_3_mul_24 :
  unit_digit (17^3 * 24) = 2 :=
by
  sorry

end units_digit_of_17_pow_3_mul_24_l225_225389


namespace even_function_zeros_l225_225662

noncomputable def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

theorem even_function_zeros (m : ℝ) (h : ∀ x : ℝ, f x m = f (-x) m ) : 
  m = 1 ∧ (∀ x : ℝ, f x m = 0 → (x = 1 ∨ x = -1)) := by
  sorry

end even_function_zeros_l225_225662


namespace smallest_n_for_multiples_of_7_l225_225424

theorem smallest_n_for_multiples_of_7 (x y : ℤ) (h1 : x ≡ 4 [ZMOD 7]) (h2 : y ≡ 5 [ZMOD 7]) :
  ∃ n : ℕ, 0 < n ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7]) ∧ ∀ m : ℕ, 0 < m ∧ (x^2 + x * y + y^2 + m ≡ 0 [ZMOD 7]) → n ≤ m :=
by
  sorry

end smallest_n_for_multiples_of_7_l225_225424


namespace Abhay_takes_1_hour_less_than_Sameer_l225_225226

noncomputable def Sameer_speed := 42 / (6 - 2)
noncomputable def Abhay_time_doubled_speed := 42 / (2 * 7)
noncomputable def Sameer_time := 42 / Sameer_speed

theorem Abhay_takes_1_hour_less_than_Sameer
  (distance : ℝ := 42)
  (Abhay_speed : ℝ := 7)
  (Sameer_speed : ℝ := Sameer_speed)
  (time_Sameer : ℝ := distance / Sameer_speed)
  (time_Abhay_doubled_speed : ℝ := distance / (2 * Abhay_speed)) :
  time_Sameer - time_Abhay_doubled_speed = 1 :=
by
  sorry

end Abhay_takes_1_hour_less_than_Sameer_l225_225226


namespace solve_equation_l225_225660

theorem solve_equation :
  ∀ x : ℝ, 4 * x * (6 * x - 1) = 1 - 6 * x ↔ (x = 1/6 ∨ x = -1/4) := 
by
  sorry

end solve_equation_l225_225660


namespace domain_of_function_l225_225606

theorem domain_of_function : 
  {x : ℝ | 0 < x ∧ 4 - x^2 > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end domain_of_function_l225_225606


namespace find_sum_of_squares_of_roots_l225_225063

theorem find_sum_of_squares_of_roots (a b c : ℝ) (h_ab : a < b) (h_bc : b < c)
  (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 2 * x^2 - 3 * x + 4)
  (h_eq : f a = f b ∧ f b = f c) :
  a^2 + b^2 + c^2 = 10 :=
sorry

end find_sum_of_squares_of_roots_l225_225063


namespace total_number_of_animals_l225_225636

theorem total_number_of_animals 
  (rabbits ducks chickens : ℕ)
  (h1 : chickens = 5 * ducks)
  (h2 : ducks = rabbits + 12)
  (h3 : rabbits = 4) : 
  chickens + ducks + rabbits = 100 :=
by
  sorry

end total_number_of_animals_l225_225636


namespace tangent_line_at_e_l225_225797

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e : ∀ x y : ℝ, (x = Real.exp 1) → (y = f x) → (y = 2 * x - Real.exp 1) :=
by
  intros x y hx hy
  sorry

end tangent_line_at_e_l225_225797


namespace cos_double_angle_l225_225572

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 := by
  sorry

end cos_double_angle_l225_225572


namespace radius_of_shorter_cone_l225_225096

theorem radius_of_shorter_cone {h : ℝ} (h_ne_zero : h ≠ 0) :
  ∀ r : ℝ, ∀ V_taller V_shorter : ℝ,
   (V_taller = (1/3) * π * (5 ^ 2) * (4 * h)) →
   (V_shorter = (1/3) * π * (r ^ 2) * h) →
   V_taller = V_shorter →
   r = 10 :=
by
  intros
  sorry

end radius_of_shorter_cone_l225_225096


namespace sum_midpoint_x_coords_l225_225569

theorem sum_midpoint_x_coords (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
    (a + (a - 3)) / 2 + (a + c) / 2 + ((a - 3) + c) / 2 = 15 := 
by 
  sorry

end sum_midpoint_x_coords_l225_225569


namespace solve_fraction_equation_l225_225371

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ x = -2 := by
sorry

end solve_fraction_equation_l225_225371


namespace jasmine_stops_at_S_l225_225158

-- Definitions of the given conditions
def circumference : ℕ := 60
def total_distance : ℕ := 5400
def quadrants : ℕ := 4
def laps (distance circumference : ℕ) := distance / circumference
def isMultiple (a b : ℕ) := b ∣ a
def onSamePoint (distance circumference : ℕ) := (distance % circumference) = 0

-- The theorem to be proved: Jasmine stops at point S after running the total distance
theorem jasmine_stops_at_S 
  (circumference : ℕ) (total_distance : ℕ) (quadrants : ℕ)
  (h1 : circumference = 60) 
  (h2 : total_distance = 5400)
  (h3 : quadrants = 4)
  (h4 : laps total_distance circumference = 90)
  (h5 : isMultiple total_distance circumference)
  : onSamePoint total_distance circumference := 
  sorry

end jasmine_stops_at_S_l225_225158


namespace find_f_at_one_l225_225872

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * x^3 + n * x + 1

theorem find_f_at_one (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : f (-1) m n = 5) : f (1) m n = 7 :=
by
  -- proof goes here
  sorry

end find_f_at_one_l225_225872


namespace geometric_concepts_cases_l225_225255

theorem geometric_concepts_cases :
  (∃ x y, x = "rectangle" ∧ y = "rhombus") ∧ 
  (∃ x y z, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "acute_triangle") ∧ 
  (∃ x y z u, x = "parallelogram" ∧ y = "rectangle" ∧ z = "square" ∧ u = "acute_angled_rhombus") ∧ 
  (∃ x y z u t, x = "polygon" ∧ y = "triangle" ∧ z = "isosceles_triangle" ∧ u = "equilateral_triangle" ∧ t = "right_triangle") ∧ 
  (∃ x y z u, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "obtuse_triangle" ∧ u = "scalene_triangle") :=
by {
  sorry
}

end geometric_concepts_cases_l225_225255


namespace find_b_l225_225007

def has_exactly_one_real_solution (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

theorem find_b (b : ℝ) :
  (∃! (x : ℝ), x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) ↔ b < 2 :=
by
  sorry

end find_b_l225_225007


namespace x_cubed_inverse_cubed_l225_225136

theorem x_cubed_inverse_cubed (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 :=
by sorry

end x_cubed_inverse_cubed_l225_225136


namespace number_of_distinct_configurations_l225_225250

-- Define the conditions
def numConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 else n + 1

-- Theorem statement
theorem number_of_distinct_configurations (n : ℕ) : 
  numConfigurations n = if n % 2 = 1 then 2 else n + 1 :=
by
  sorry -- Proof intentionally left out

end number_of_distinct_configurations_l225_225250


namespace mortgage_loan_amount_l225_225334

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end mortgage_loan_amount_l225_225334


namespace initial_garrison_men_l225_225859

theorem initial_garrison_men (M : ℕ) (H1 : ∃ provisions : ℕ, provisions = M * 60)
  (H2 : ∃ provisions_15 : ℕ, provisions_15 = M * 45)
  (H3 : ∀ provisions_15 (new_provisions: ℕ), (provisions_15 = M * 45 ∧ new_provisions = 20 * (M + 1250)) → provisions_15 = new_provisions) :
  M = 1000 :=
by
  sorry

end initial_garrison_men_l225_225859


namespace larger_number_of_two_with_conditions_l225_225313

theorem larger_number_of_two_with_conditions (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
by
  sorry

end larger_number_of_two_with_conditions_l225_225313


namespace product_xyz_l225_225073

noncomputable def x : ℚ := 97 / 12
noncomputable def n : ℚ := 8 * x
noncomputable def y : ℚ := n + 7
noncomputable def z : ℚ := n - 11

theorem product_xyz 
  (h1: x + y + z = 190)
  (h2: n = 8 * x)
  (h3: n = y - 7)
  (h4: n = z + 11) : 
  x * y * z = (97 * 215 * 161) / 108 := 
by 
  sorry

end product_xyz_l225_225073


namespace equal_sum_seq_value_at_18_l225_225605

-- Define what it means for a sequence to be an equal-sum sequence with a common sum
def equal_sum_seq (a : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_seq_value_at_18
  (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : equal_sum_seq a 5) :
  a 18 = 3 :=
sorry

end equal_sum_seq_value_at_18_l225_225605


namespace polygon_sides_l225_225119

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l225_225119


namespace original_cube_volume_eq_216_l225_225538

theorem original_cube_volume_eq_216 (a : ℕ)
  (h1 : ∀ (a : ℕ), ∃ V_orig V_new : ℕ, 
    V_orig = a^3 ∧ 
    V_new = (a + 1) * (a + 1) * (a - 2) ∧ 
    V_orig = V_new + 10) : 
  a = 6 → a^3 = 216 := 
by
  sorry

end original_cube_volume_eq_216_l225_225538


namespace tommys_profit_l225_225453

-- Definitions of the conditions
def crateA_cost : ℕ := 220
def crateB_cost : ℕ := 375
def crateC_cost : ℕ := 180

def crateA_count : ℕ := 2
def crateB_count : ℕ := 3
def crateC_count : ℕ := 1

def crateA_capacity : ℕ := 20
def crateB_capacity : ℕ := 25
def crateC_capacity : ℕ := 30

def crateA_rotten : ℕ := 4
def crateB_rotten : ℕ := 5
def crateC_rotten : ℕ := 3

def crateA_price_per_kg : ℕ := 5
def crateB_price_per_kg : ℕ := 6
def crateC_price_per_kg : ℕ := 7

-- Calculations based on the conditions
def total_cost : ℕ := crateA_cost + crateB_cost + crateC_cost

def sellable_weightA : ℕ := crateA_count * crateA_capacity - crateA_rotten
def sellable_weightB : ℕ := crateB_count * crateB_capacity - crateB_rotten
def sellable_weightC : ℕ := crateC_count * crateC_capacity - crateC_rotten

def revenueA : ℕ := sellable_weightA * crateA_price_per_kg
def revenueB : ℕ := sellable_weightB * crateB_price_per_kg
def revenueC : ℕ := sellable_weightC * crateC_price_per_kg

def total_revenue : ℕ := revenueA + revenueB + revenueC

def profit : ℕ := total_revenue - total_cost

-- The theorem we want to verify
theorem tommys_profit : profit = 14 := by
  sorry

end tommys_profit_l225_225453


namespace probability_shattering_l225_225485

theorem probability_shattering (total_cars : ℕ) (shattered_windshields : ℕ) (p : ℚ) 
  (h_total : total_cars = 20000) 
  (h_shattered: shattered_windshields = 600) 
  (h_p : p = shattered_windshields / total_cars) : 
  p = 0.03 := 
by 
  -- skipped proof
  sorry

end probability_shattering_l225_225485


namespace tan_pink_violet_probability_l225_225772

noncomputable def probability_tan_pink_violet_consecutive_order : ℚ :=
  let num_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5)
  let total_ways := Nat.factorial 12
  num_ways / total_ways

theorem tan_pink_violet_probability :
  probability_tan_pink_violet_consecutive_order = 1 / 27720 := by
  sorry

end tan_pink_violet_probability_l225_225772


namespace necessary_but_not_sufficient_l225_225278

-- Define the sets A and B
def A (x : ℝ) : Prop := x > 2
def B (x : ℝ) : Prop := x > 1

-- Prove that B (necessary condition x > 1) does not suffice for A (x > 2)
theorem necessary_but_not_sufficient (x : ℝ) (h : B x) : A x ∨ ¬A x :=
by
  -- B x is a necessary condition for A x
  have h1 : x > 1 := h
  -- A x is not necessarily implied by B x
  sorry

end necessary_but_not_sufficient_l225_225278


namespace train_length_l225_225412

theorem train_length (v : ℝ) (t : ℝ) (conversion_factor : ℝ) : v = 45 → t = 16 → conversion_factor = 1000 / 3600 → (v * (conversion_factor) * t) = 200 :=
  by
  intros hv ht hcf
  rw [hv, ht, hcf]
  -- Proof steps skipped
  sorry

end train_length_l225_225412


namespace value_of_abcg_defh_l225_225672

theorem value_of_abcg_defh
  (a b c d e f g h: ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4) :
  abcg / defh = 5 / 48 :=
by
  sorry

end value_of_abcg_defh_l225_225672


namespace ratio_friday_to_monday_l225_225791

-- Definitions from conditions
def rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def total_toys : ℕ := 3 * rabbits

-- Definition to represent the number of toys bought on Friday
def friday_toys : ℕ := total_toys - (monday_toys + wednesday_toys + saturday_toys)

-- Theorem to prove the ratio is 4:1
theorem ratio_friday_to_monday : friday_toys / monday_toys = 4 := by
  -- Placeholder for the proof
  sorry

end ratio_friday_to_monday_l225_225791


namespace alex_shirts_l225_225611

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l225_225611


namespace milk_water_equal_l225_225361

theorem milk_water_equal (a : ℕ) :
  let glass_a_initial := a
  let glass_b_initial := a
  let mixture_in_a := glass_a_initial + 1
  let milk_portion_in_a := 1 / mixture_in_a
  let water_portion_in_a := glass_a_initial / mixture_in_a
  let water_in_milk_glass := water_portion_in_a
  let milk_in_water_glass := milk_portion_in_a
  water_in_milk_glass = milk_in_water_glass := by
  sorry

end milk_water_equal_l225_225361
