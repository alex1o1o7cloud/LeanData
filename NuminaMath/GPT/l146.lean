import Mathlib

namespace NUMINAMATH_GPT_sum_of_edges_l146_14602

theorem sum_of_edges (a r : ℝ) 
  (h_vol : (a / r) * a * (a * r) = 432) 
  (h_surf_area : 2 * ((a * a) / r + (a * a) * r + a * a) = 384) 
  (h_geom_prog : r ≠ 1) :
  4 * ((6 * Real.sqrt 2) / r + 6 * Real.sqrt 2 + (6 * Real.sqrt 2) * r) = 72 * (Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_sum_of_edges_l146_14602


namespace NUMINAMATH_GPT_angle_complement_supplement_l146_14646

theorem angle_complement_supplement (x : ℝ) (h : 90 - x = 3 / 4 * (180 - x)) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_angle_complement_supplement_l146_14646


namespace NUMINAMATH_GPT_trigonometric_identity_l146_14695

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 2 + α) = - 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l146_14695


namespace NUMINAMATH_GPT_moles_of_NaCl_formed_l146_14628

-- Define the conditions
def moles_NaOH : ℕ := 3
def moles_HCl : ℕ := 3

-- Define the balanced chemical equation as a relation
def reaction (NaOH HCl NaCl H2O : ℕ) : Prop :=
  NaOH = HCl ∧ HCl = NaCl ∧ H2O = NaCl

-- Define the proof problem
theorem moles_of_NaCl_formed :
  ∀ (NaOH HCl NaCl H2O : ℕ), NaOH = 3 → HCl = 3 → reaction NaOH HCl NaCl H2O → NaCl = 3 :=
by
  intros NaOH HCl NaCl H2O hNa hHCl hReaction
  sorry

end NUMINAMATH_GPT_moles_of_NaCl_formed_l146_14628


namespace NUMINAMATH_GPT_candies_in_box_more_than_pockets_l146_14620

theorem candies_in_box_more_than_pockets (x : ℕ) : 
  let initial_pockets := 2 * x
  let pockets_after_return := 2 * (x - 6)
  let candies_returned_to_box := 12
  let total_candies_after_return := initial_pockets + candies_returned_to_box
  (total_candies_after_return - pockets_after_return) = 24 :=
by
  sorry

end NUMINAMATH_GPT_candies_in_box_more_than_pockets_l146_14620


namespace NUMINAMATH_GPT_sum_of_first_10_terms_l146_14640

def general_term (n : ℕ) : ℕ := 2 * n + 1

def sequence_sum (n : ℕ) : ℕ := n / 2 * (general_term 1 + general_term n)

theorem sum_of_first_10_terms : sequence_sum 10 = 120 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_10_terms_l146_14640


namespace NUMINAMATH_GPT_find_m_l146_14641

open Set Real

noncomputable def setA : Set ℝ := {x | x < 2}
noncomputable def setB : Set ℝ := {x | x > 4}
noncomputable def setC (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

theorem find_m (m : ℝ) : setC m ⊆ (setA ∪ setB) → m < 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l146_14641


namespace NUMINAMATH_GPT_cost_of_six_dozen_l146_14664

variable (cost_of_four_dozen : ℕ)
variable (dozens_to_purchase : ℕ)

theorem cost_of_six_dozen :
  cost_of_four_dozen = 24 →
  dozens_to_purchase = 6 →
  (dozens_to_purchase * (cost_of_four_dozen / 4)) = 36 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_of_six_dozen_l146_14664


namespace NUMINAMATH_GPT_calculation_proof_l146_14613

theorem calculation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end NUMINAMATH_GPT_calculation_proof_l146_14613


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l146_14608

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iic (-2) → (x^2 + 2 * a * x - 2) ≤ ((x - 1)^2 + 2 * a * (x - 1) - 2)) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l146_14608


namespace NUMINAMATH_GPT_allocation_schemes_count_l146_14679

open BigOperators -- For working with big operator notations
open Finset -- For working with finite sets
open Nat -- For natural number operations

-- Define the number of students and dormitories
def num_students : ℕ := 7
def num_dormitories : ℕ := 2

-- Define the constraint for minimum students in each dormitory
def min_students_in_dormitory : ℕ := 2

-- Compute the number of ways to allocate students given the conditions
noncomputable def number_of_allocation_schemes : ℕ :=
  (Nat.choose num_students 3) * (Nat.choose 4 2) + (Nat.choose num_students 2) * (Nat.choose 5 2)

-- The theorem stating the total number of allocation schemes
theorem allocation_schemes_count :
  number_of_allocation_schemes = 112 :=
  by sorry

end NUMINAMATH_GPT_allocation_schemes_count_l146_14679


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l146_14659

theorem megatek_manufacturing_percentage (angle_manufacturing : ℝ) (full_circle : ℝ) 
  (h1 : angle_manufacturing = 162) (h2 : full_circle = 360) :
  (angle_manufacturing / full_circle) * 100 = 45 :=
by
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l146_14659


namespace NUMINAMATH_GPT_probability_of_three_cards_l146_14651

-- Conditions
def deck_size : ℕ := 52
def spades : ℕ := 13
def spades_face_cards : ℕ := 3
def face_cards : ℕ := 12
def diamonds : ℕ := 13

-- Probability of drawing specific cards
def prob_first_spade_non_face : ℚ := 10 / 52
def prob_second_face_given_first_spade_non_face : ℚ := 12 / 51
def prob_third_diamond_given_first_two : ℚ := 13 / 50

def prob_first_spade_face : ℚ := 3 / 52
def prob_second_face_given_first_spade_face : ℚ := 9 / 51

-- Final probability
def final_probability := 
  (prob_first_spade_non_face * prob_second_face_given_first_spade_non_face * prob_third_diamond_given_first_two) +
  (prob_first_spade_face * prob_second_face_given_first_spade_face * prob_third_diamond_given_first_two)

theorem probability_of_three_cards :
  final_probability = 1911 / 132600 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_three_cards_l146_14651


namespace NUMINAMATH_GPT_sqrt_calculation_l146_14662

theorem sqrt_calculation : Real.sqrt (36 * Real.sqrt 16) = 12 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_calculation_l146_14662


namespace NUMINAMATH_GPT_total_income_l146_14681

def ron_ticket_price : ℝ := 2.00
def kathy_ticket_price : ℝ := 4.50
def total_tickets : ℕ := 20
def ron_tickets_sold : ℕ := 12

theorem total_income : ron_tickets_sold * ron_ticket_price + (total_tickets - ron_tickets_sold) * kathy_ticket_price = 60.00 := by
  sorry

end NUMINAMATH_GPT_total_income_l146_14681


namespace NUMINAMATH_GPT_find_certain_number_l146_14647

theorem find_certain_number (N : ℝ) 
  (h : 3.6 * N * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001)
  : N = 0.48 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l146_14647


namespace NUMINAMATH_GPT_max_value_ab_l146_14626

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end NUMINAMATH_GPT_max_value_ab_l146_14626


namespace NUMINAMATH_GPT_find_a_l146_14682

theorem find_a (a : ℝ) (h1 : ∀ (x y : ℝ), ax + 2*y - 2 = 0 → (x + y) = 0)
  (h2 : ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 6 → (∃ A B : ℝ × ℝ, A ≠ B ∧ (A = (x, y) ∧ B = (-x, -y))))
  : a = -2 := 
sorry

end NUMINAMATH_GPT_find_a_l146_14682


namespace NUMINAMATH_GPT_sufficient_not_necessary_l146_14615

theorem sufficient_not_necessary (x : ℝ) :
  (|x - 1| < 2 → x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l146_14615


namespace NUMINAMATH_GPT_yellow_tickets_needed_l146_14698

def yellow_from_red (r : ℕ) : ℕ := r / 10
def red_from_blue (b : ℕ) : ℕ := b / 10
def blue_needed (current_blue : ℕ) (additional_blue : ℕ) : ℕ := current_blue + additional_blue
def total_blue_from_tickets (y : ℕ) (r : ℕ) (b : ℕ) : ℕ := (y * 10 * 10) + (r * 10) + b

theorem yellow_tickets_needed (y r b additional_blue : ℕ) (h : total_blue_from_tickets y r b + additional_blue = 1000) :
  yellow_from_red (red_from_blue (total_blue_from_tickets y r b + additional_blue)) = 10 := 
by
  sorry

end NUMINAMATH_GPT_yellow_tickets_needed_l146_14698


namespace NUMINAMATH_GPT_solve_system_of_equations_l146_14666

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - y = 2 ∧ 3 * x + y = 4 ∧ x = 1.5 ∧ y = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l146_14666


namespace NUMINAMATH_GPT_solve_xy_eq_yx_l146_14668

theorem solve_xy_eq_yx (x y : ℕ) (hxy : x ≠ y) : x^y = y^x ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_xy_eq_yx_l146_14668


namespace NUMINAMATH_GPT_complement_union_eq_l146_14697

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)})
variable (B : Set ℝ := {x | -2 ≤ x ∧ x < 4})

theorem complement_union_eq : (U \ A) ∪ B = {x | x ≥ -2} := by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l146_14697


namespace NUMINAMATH_GPT_combined_total_l146_14633

variable (Jane Jean : ℕ)

theorem combined_total (h1 : Jean = 3 * Jane) (h2 : Jean = 57) : Jane + Jean = 76 := by
  sorry

end NUMINAMATH_GPT_combined_total_l146_14633


namespace NUMINAMATH_GPT_intersection_A_B_l146_14609

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection (S T : Set ℝ) : Set ℝ := {x | x ∈ S ∧ x ∈ T}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l146_14609


namespace NUMINAMATH_GPT_janet_stuffies_l146_14677

theorem janet_stuffies (total_stuffies kept_stuffies given_away_stuffies janet_stuffies : ℕ) 
 (h1 : total_stuffies = 60)
 (h2 : kept_stuffies = total_stuffies / 3)
 (h3 : given_away_stuffies = total_stuffies - kept_stuffies)
 (h4 : janet_stuffies = given_away_stuffies / 4) : 
 janet_stuffies = 10 := 
sorry

end NUMINAMATH_GPT_janet_stuffies_l146_14677


namespace NUMINAMATH_GPT_square_nonneg_of_nonneg_l146_14667

theorem square_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x^2 :=
sorry

end NUMINAMATH_GPT_square_nonneg_of_nonneg_l146_14667


namespace NUMINAMATH_GPT_white_balls_count_l146_14671

theorem white_balls_count (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) (W : ℕ)
    (h_total : total_balls = 100)
    (h_green : green_balls = 30)
    (h_yellow : yellow_balls = 10)
    (h_red : red_balls = 37)
    (h_purple : purple_balls = 3)
    (h_prob : prob_neither_red_nor_purple = 0.6)
    (h_computation : W = total_balls * prob_neither_red_nor_purple - (green_balls + yellow_balls)) :
    W = 20 := 
sorry

end NUMINAMATH_GPT_white_balls_count_l146_14671


namespace NUMINAMATH_GPT_erica_blank_question_count_l146_14669

variable {C W B : ℕ}

theorem erica_blank_question_count
  (h1 : C + W + B = 20)
  (h2 : 7 * C - 4 * W = 100) :
  B = 1 :=
by
  sorry

end NUMINAMATH_GPT_erica_blank_question_count_l146_14669


namespace NUMINAMATH_GPT_simplify_fraction_l146_14617

variable {R : Type*} [Field R]
variables (x y z : R)

theorem simplify_fraction : (6 * x * y / (5 * z ^ 2)) * (10 * z ^ 3 / (9 * x * y)) = (4 * z) / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l146_14617


namespace NUMINAMATH_GPT_total_amount_after_5_months_l146_14639

-- Definitions from the conditions
def initial_deposit : ℝ := 100
def monthly_interest_rate : ℝ := 0.0036  -- 0.36% expressed as a decimal

-- Definition of the function relationship y with respect to x
def total_amount (x : ℕ) : ℝ := initial_deposit + initial_deposit * monthly_interest_rate * x

-- Prove the total amount after 5 months is 101.8
theorem total_amount_after_5_months : total_amount 5 = 101.8 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_after_5_months_l146_14639


namespace NUMINAMATH_GPT_probability_of_point_within_two_units_l146_14687

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let area_of_circle := 4 * Real.pi
  let area_of_square := 36
  area_of_circle / area_of_square

theorem probability_of_point_within_two_units :
  probability_within_two_units_of_origin = Real.pi / 9 := 
by
  -- The proof steps are omitted as per the requirements
  sorry

end NUMINAMATH_GPT_probability_of_point_within_two_units_l146_14687


namespace NUMINAMATH_GPT_impossible_to_transport_stones_l146_14665

-- Define the conditions of the problem
def stones : List ℕ := List.range' 370 (468 - 370 + 2 + 1) 2
def truck_capacity : ℕ := 3000
def number_of_trucks : ℕ := 7
def number_of_stones : ℕ := 50

-- Prove that it is impossible to transport the stones using the given trucks
theorem impossible_to_transport_stones :
  stones.length = number_of_stones →
  (∀ weights ∈ stones.sublists, (weights.sum ≤ truck_capacity → List.length weights ≤ number_of_trucks)) → 
  false :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_transport_stones_l146_14665


namespace NUMINAMATH_GPT_denominator_of_expression_l146_14642

theorem denominator_of_expression (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end NUMINAMATH_GPT_denominator_of_expression_l146_14642


namespace NUMINAMATH_GPT_find_f_neg2_l146_14616

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem find_f_neg2 : f (-2) = -8 := by
  sorry

end NUMINAMATH_GPT_find_f_neg2_l146_14616


namespace NUMINAMATH_GPT_sheep_ratio_l146_14694

theorem sheep_ratio (s : ℕ) (h1 : s = 400) (h2 : s / 4 + 150 = s - s / 4) : (s / 4 * 3 - 150) / 150 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_sheep_ratio_l146_14694


namespace NUMINAMATH_GPT_z_share_in_profit_l146_14660

noncomputable def investment_share (investment : ℕ) (months : ℕ) : ℕ := investment * months

noncomputable def profit_share (profit : ℕ) (share : ℚ) : ℚ := (profit : ℚ) * share

theorem z_share_in_profit 
  (investment_X : ℕ := 36000)
  (investment_Y : ℕ := 42000)
  (investment_Z : ℕ := 48000)
  (months_X : ℕ := 12)
  (months_Y : ℕ := 12)
  (months_Z : ℕ := 8)
  (total_profit : ℕ := 14300) :
  profit_share total_profit (investment_share investment_Z months_Z / 
            (investment_share investment_X months_X + 
             investment_share investment_Y months_Y + 
             investment_share investment_Z months_Z)) = 2600 := 
by
  sorry

end NUMINAMATH_GPT_z_share_in_profit_l146_14660


namespace NUMINAMATH_GPT_white_pairs_coincide_l146_14688

def num_red : Nat := 4
def num_blue : Nat := 4
def num_green : Nat := 2
def num_white : Nat := 6
def red_pairs : Nat := 3
def blue_pairs : Nat := 2
def green_pairs : Nat := 1 
def red_white_pairs : Nat := 2
def green_blue_pairs : Nat := 1

theorem white_pairs_coincide :
  (num_red = 4) ∧ 
  (num_blue = 4) ∧ 
  (num_green = 2) ∧ 
  (num_white = 6) ∧ 
  (red_pairs = 3) ∧ 
  (blue_pairs = 2) ∧ 
  (green_pairs = 1) ∧ 
  (red_white_pairs = 2) ∧ 
  (green_blue_pairs = 1) → 
  4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l146_14688


namespace NUMINAMATH_GPT_find_N_l146_14629

theorem find_N (a b c N : ℚ) (h_sum : a + b + c = 84)
    (h_a : a - 7 = N) (h_b : b + 7 = N) (h_c : c / 7 = N) : 
    N = 28 / 3 :=
sorry

end NUMINAMATH_GPT_find_N_l146_14629


namespace NUMINAMATH_GPT_first_candidate_percentage_l146_14657

theorem first_candidate_percentage (P : ℝ) 
    (total_votes : ℝ) (votes_second : ℝ)
    (h_total_votes : total_votes = 1200)
    (h_votes_second : votes_second = 480) :
    (P / 100) * total_votes + votes_second = total_votes → P = 60 := 
by
  intro h
  rw [h_total_votes, h_votes_second] at h
  sorry

end NUMINAMATH_GPT_first_candidate_percentage_l146_14657


namespace NUMINAMATH_GPT_cistern_fill_time_l146_14607

-- Let F be the rate at which the first tap fills the cistern (cisterns per hour)
def F : ℚ := 1 / 4

-- Let E be the rate at which the second tap empties the cistern (cisterns per hour)
def E : ℚ := 1 / 5

-- Prove that the time it takes to fill the cistern is 20 hours given the rates F and E
theorem cistern_fill_time : (1 / (F - E)) = 20 := 
by
  -- Insert necessary proofs here
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l146_14607


namespace NUMINAMATH_GPT_abs_sum_lt_abs_diff_l146_14644

theorem abs_sum_lt_abs_diff (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end NUMINAMATH_GPT_abs_sum_lt_abs_diff_l146_14644


namespace NUMINAMATH_GPT_buns_left_l146_14605

theorem buns_left (buns_initial : ℕ) (h1 : buns_initial = 15)
                  (x : ℕ) (h2 : 13 * x ≤ buns_initial)
                  (buns_taken_by_bimbo : ℕ) (h3 : buns_taken_by_bimbo = x)
                  (buns_taken_by_little_boy : ℕ) (h4 : buns_taken_by_little_boy = 3 * x)
                  (buns_taken_by_karlsson : ℕ) (h5 : buns_taken_by_karlsson = 9 * x)
                  :
                  buns_initial - (buns_taken_by_bimbo + buns_taken_by_little_boy + buns_taken_by_karlsson) = 2 :=
by
  sorry

end NUMINAMATH_GPT_buns_left_l146_14605


namespace NUMINAMATH_GPT_sum_of_squares_l146_14693

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : ab + bc + ca = 5) : a^2 + b^2 + c^2 = 390 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_l146_14693


namespace NUMINAMATH_GPT_proof_l146_14645

-- Define the equation and its conditions
def equation (x m : ℤ) : Prop := (3 * x - 1) / 2 + m = 3

-- Part 1: Prove that for m = 5, the corresponding x must be 1
def part1 : Prop :=
  ∃ x : ℤ, equation x 5 ∧ x = 1

-- Part 2: Prove that if the equation has a positive integer solution, the positive integer m must be 2
def part2 : Prop :=
  ∃ m x : ℤ, m > 0 ∧ x > 0 ∧ equation x m ∧ m = 2

theorem proof : part1 ∧ part2 :=
  by
    sorry

end NUMINAMATH_GPT_proof_l146_14645


namespace NUMINAMATH_GPT_minimum_value_inequality_l146_14649

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 1) * (y^2 + 5 * y + 1) * (z^2 + 5 * y + 1) / (x * y * z) ≥ 343 :=
by sorry

end NUMINAMATH_GPT_minimum_value_inequality_l146_14649


namespace NUMINAMATH_GPT_union_M_N_l146_14636

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end NUMINAMATH_GPT_union_M_N_l146_14636


namespace NUMINAMATH_GPT_joe_saves_6000_l146_14699

-- Definitions based on the conditions
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

-- Total expenses
def total_expenses : ℕ := flight_cost + hotel_cost + food_cost

-- Total savings
def total_savings : ℕ := total_expenses + money_left

-- The proof statement
theorem joe_saves_6000 : total_savings = 6000 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_joe_saves_6000_l146_14699


namespace NUMINAMATH_GPT_savings_wednesday_l146_14624

variable (m t s w : ℕ)

theorem savings_wednesday :
  m = 15 → t = 28 → s = 28 → 2 * s = 56 → 
  m + t + w = 56 → w = 13 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_savings_wednesday_l146_14624


namespace NUMINAMATH_GPT_problem1_problem2_l146_14638

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = -3/4) :
  (Real.cos ((π / 2) + α) * Real.sin (-π - α)) /
  (Real.cos ((11 * π) / 2 - α) * Real.sin ((9 * π) / 2 + α)) = -3 / 4 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : Real.sin α + Real.cos α = 1 / 5) :
  Real.cos (2 * α - π / 4) = -31 * Real.sqrt 2 / 50 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l146_14638


namespace NUMINAMATH_GPT_beavers_help_l146_14619

theorem beavers_help (initial final : ℝ) (h_initial : initial = 2.0) (h_final : final = 3) : final - initial = 1 :=
  by
    sorry

end NUMINAMATH_GPT_beavers_help_l146_14619


namespace NUMINAMATH_GPT_matrix_det_l146_14684

def matrix := ![
  ![2, -4, 2],
  ![0, 6, -1],
  ![5, -3, 1]
]

theorem matrix_det : Matrix.det matrix = -34 := by
  sorry

end NUMINAMATH_GPT_matrix_det_l146_14684


namespace NUMINAMATH_GPT_option_A_correct_l146_14604

theorem option_A_correct (y x : ℝ) : y * x - 2 * (x * y) = - (x * y) :=
by
  sorry

end NUMINAMATH_GPT_option_A_correct_l146_14604


namespace NUMINAMATH_GPT_acrobat_count_l146_14601

theorem acrobat_count (a e c : ℕ) (h1 : 2 * a + 4 * e + 2 * c = 88) (h2 : a + e + c = 30) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_acrobat_count_l146_14601


namespace NUMINAMATH_GPT_pureGalaTrees_l146_14630

theorem pureGalaTrees {T F C : ℕ} (h1 : F + C = 204) (h2 : F = (3 / 4 : ℝ) * T) (h3 : C = (1 / 10 : ℝ) * T) : (0.15 * T : ℝ) = 36 :=
by
  sorry

end NUMINAMATH_GPT_pureGalaTrees_l146_14630


namespace NUMINAMATH_GPT_kenneth_past_finish_line_l146_14614

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_kenneth_past_finish_line_l146_14614


namespace NUMINAMATH_GPT_students_left_during_year_l146_14661

theorem students_left_during_year (initial_students : ℕ) (new_students : ℕ) (final_students : ℕ) (students_left : ℕ) :
  initial_students = 4 →
  new_students = 42 →
  final_students = 43 →
  students_left = initial_students + new_students - final_students →
  students_left = 3 :=
by
  intro h_initial h_new h_final h_students_left
  rw [h_initial, h_new, h_final] at h_students_left
  exact h_students_left

end NUMINAMATH_GPT_students_left_during_year_l146_14661


namespace NUMINAMATH_GPT_fewer_cubes_needed_l146_14648

variable (cubeVolume : ℕ) (length : ℕ) (width : ℕ) (depth : ℕ) (TVolume : ℕ)

theorem fewer_cubes_needed : 
  cubeVolume = 5 → 
  length = 7 → 
  width = 7 → 
  depth = 6 → 
  TVolume = 3 → 
  (length * width * depth - TVolume = 291) :=
by
  intros hc hl hw hd ht
  sorry

end NUMINAMATH_GPT_fewer_cubes_needed_l146_14648


namespace NUMINAMATH_GPT_min_value_expr_l146_14658

theorem min_value_expr (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, m > 0 ∧ (forall (n : ℕ), 0 < n → (n/2 + 50/n : ℝ) ≥ 10) ∧ 
           (n = 10) → (n/2 + 50/n : ℝ) = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l146_14658


namespace NUMINAMATH_GPT_scientific_notation_l146_14663

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_scientific_notation_l146_14663


namespace NUMINAMATH_GPT_express_in_scientific_notation_l146_14683

def scientific_notation_of_160000 : Prop :=
  160000 = 1.6 * 10^5

theorem express_in_scientific_notation : scientific_notation_of_160000 :=
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l146_14683


namespace NUMINAMATH_GPT_sufficient_condition_for_reciprocal_inequality_l146_14623

variable (a b : ℝ)

theorem sufficient_condition_for_reciprocal_inequality 
  (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_reciprocal_inequality_l146_14623


namespace NUMINAMATH_GPT_stephanie_falls_l146_14635

theorem stephanie_falls 
  (steven_falls : ℕ := 3)
  (sonya_falls : ℕ := 6)
  (h1 : sonya_falls = 6)
  (h2 : ∃ S : ℕ, sonya_falls = (S / 2) - 2 ∧ S > steven_falls) :
  ∃ S : ℕ, S - steven_falls = 13 :=
by
  sorry

end NUMINAMATH_GPT_stephanie_falls_l146_14635


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l146_14672

-- Condition for the quadratic equation having two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)) ↔ (k ≥ 3 / 2) :=
sorry

-- Condition linking the roots of the equation and the properties of the rectangle
theorem roots_form_rectangle_with_diagonal (k : ℝ) 
  (h : k ≥ 3 / 2) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)
  ∧ (x1^2 + x2^2 = 5)) ↔ (k = 2) :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l146_14672


namespace NUMINAMATH_GPT_sum_of_squares_and_product_l146_14627

theorem sum_of_squares_and_product (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y = Real.sqrt 202 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_product_l146_14627


namespace NUMINAMATH_GPT_swap_correct_l146_14632

variable (a b c : ℕ)

noncomputable def swap_and_verify (a : ℕ) (b : ℕ) : Prop :=
  let c := b
  let b := a
  let a := c
  a = 2012 ∧ b = 2011

theorem swap_correct :
  ∀ a b : ℕ, a = 2011 → b = 2012 → swap_and_verify a b :=
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_swap_correct_l146_14632


namespace NUMINAMATH_GPT_find_M_M_superset_N_M_intersection_N_l146_14634

-- Define the set M as per the given condition
def M : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

-- Define the set N based on parameters a and b
def N (a b : ℝ) : Set ℝ := { x : ℝ | a < x ∧ x < b }

-- Prove that M = (-1, 2)
theorem find_M : M = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Prove that if M ⊇ N, then a ≥ -1
theorem M_superset_N (a b : ℝ) (h : M ⊇ N a b) : -1 ≤ a :=
sorry

-- Prove that if M ∩ N = M, then b ≥ 2
theorem M_intersection_N (a b : ℝ) (h : M ∩ (N a b) = M) : 2 ≤ b :=
sorry

end NUMINAMATH_GPT_find_M_M_superset_N_M_intersection_N_l146_14634


namespace NUMINAMATH_GPT_four_digit_number_count_l146_14643

theorem four_digit_number_count :
  (∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ 
    ((n / 1000 < 5 ∧ (n / 100) % 10 < 5) ∨ (n / 1000 > 5 ∧ (n / 100) % 10 > 5)) ∧ 
    (((n % 100) / 10 < 5 ∧ n % 10 < 5) ∨ ((n % 100) / 10 > 5 ∧ n % 10 > 5))) →
    ∃ (count : ℕ), count = 1681 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_count_l146_14643


namespace NUMINAMATH_GPT_nat_prime_p_and_5p_plus_1_is_prime_l146_14622

theorem nat_prime_p_and_5p_plus_1_is_prime (p : ℕ) (hp : Nat.Prime p) (h5p1 : Nat.Prime (5 * p + 1)) : p = 2 := 
by 
  -- Sorry is added to skip the proof
  sorry 

end NUMINAMATH_GPT_nat_prime_p_and_5p_plus_1_is_prime_l146_14622


namespace NUMINAMATH_GPT_minimum_k_exists_l146_14691

theorem minimum_k_exists :
  ∀ (s : Finset ℝ), s.card = 3 → (∀ (a b : ℝ), a ∈ s → b ∈ s → (|a - b| ≤ (1.5 : ℝ) ∨ |(1 / a) - (1 / b)| ≤ 1.5)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_k_exists_l146_14691


namespace NUMINAMATH_GPT_linear_equation_must_be_neg2_l146_14621

theorem linear_equation_must_be_neg2 {m : ℝ} (h1 : |m| - 1 = 1) (h2 : m ≠ 2) : m = -2 :=
sorry

end NUMINAMATH_GPT_linear_equation_must_be_neg2_l146_14621


namespace NUMINAMATH_GPT_maximum_area_of_rectangular_playground_l146_14652

theorem maximum_area_of_rectangular_playground (P : ℕ) (A : ℕ) (h : P = 150) :
  ∃ (x y : ℕ), x + y = 75 ∧ A ≤ x * y ∧ A = 1406 :=
sorry

end NUMINAMATH_GPT_maximum_area_of_rectangular_playground_l146_14652


namespace NUMINAMATH_GPT_quadratic_roots_l146_14654

theorem quadratic_roots (a b c : ℝ) :
  ∃ x y : ℝ, (x ≠ y ∧ (x^2 - (a + b) * x + (ab - c^2) = 0) ∧ (y^2 - (a + b) * y + (ab - c^2) = 0)) ∧
  (x = y ↔ a = b ∧ c = 0) := sorry

end NUMINAMATH_GPT_quadratic_roots_l146_14654


namespace NUMINAMATH_GPT_k_is_square_l146_14625

theorem k_is_square (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (k : ℕ) (h_k : k > 0)
    (h : (a^2 + b^2) = k * (a * b + 1)) : ∃ (n : ℕ), n^2 = k :=
sorry

end NUMINAMATH_GPT_k_is_square_l146_14625


namespace NUMINAMATH_GPT_generalized_inequality_combinatorial_inequality_l146_14618

-- Part 1: Generalized Inequality
theorem generalized_inequality (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (b i)^2 / (a i))) ≥
  ((Finset.univ.sum (fun i => b i))^2 / (Finset.univ.sum (fun i => a i))) :=
sorry

-- Part 2: Combinatorial Inequality
theorem combinatorial_inequality (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => (2 * k + 1) / (Nat.choose n k)) ≥
  ((n + 1)^3 / (2^n : ℝ)) :=
sorry

end NUMINAMATH_GPT_generalized_inequality_combinatorial_inequality_l146_14618


namespace NUMINAMATH_GPT_surface_area_of_large_cube_is_486_cm_squared_l146_14650

noncomputable def surfaceAreaLargeCube : ℕ :=
  let small_box_count := 27
  let edge_small_box := 3
  let edge_large_cube := (small_box_count^(1/3)) * edge_small_box
  6 * edge_large_cube^2

theorem surface_area_of_large_cube_is_486_cm_squared :
  surfaceAreaLargeCube = 486 := 
sorry

end NUMINAMATH_GPT_surface_area_of_large_cube_is_486_cm_squared_l146_14650


namespace NUMINAMATH_GPT_students_taking_neither_l146_14680

-- Defining given conditions as Lean definitions
def total_students : ℕ := 70
def students_math : ℕ := 42
def students_physics : ℕ := 35
def students_chemistry : ℕ := 25
def students_math_physics : ℕ := 18
def students_math_chemistry : ℕ := 10
def students_physics_chemistry : ℕ := 8
def students_all_three : ℕ := 5

-- Define the problem to prove
theorem students_taking_neither : total_students
  - (students_math - students_math_physics - students_math_chemistry + students_all_three
    + students_physics - students_math_physics - students_physics_chemistry + students_all_three
    + students_chemistry - students_math_chemistry - students_physics_chemistry + students_all_three
    + students_math_physics - students_all_three
    + students_math_chemistry - students_all_three
    + students_physics_chemistry - students_all_three
    + students_all_three) = 0 := by
  sorry

end NUMINAMATH_GPT_students_taking_neither_l146_14680


namespace NUMINAMATH_GPT_num_5_letter_words_with_at_least_one_A_l146_14631

theorem num_5_letter_words_with_at_least_one_A :
  let total := 6 ^ 5
  let without_A := 5 ^ 5
  total - without_A = 4651 := by
sorry

end NUMINAMATH_GPT_num_5_letter_words_with_at_least_one_A_l146_14631


namespace NUMINAMATH_GPT_area_union_of_reflected_triangles_l146_14610

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 7)
def C : ℝ × ℝ := (6, 2)
def A' : ℝ × ℝ := (3, 2)
def B' : ℝ × ℝ := (7, 5)
def C' : ℝ × ℝ := (2, 6)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_union_of_reflected_triangles :
  let area_ABC := triangle_area A B C
  let area_A'B'C' := triangle_area A' B' C'
  area_ABC + area_A'B'C' = 19 := by
  sorry

end NUMINAMATH_GPT_area_union_of_reflected_triangles_l146_14610


namespace NUMINAMATH_GPT_lab_tech_items_l146_14696

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end NUMINAMATH_GPT_lab_tech_items_l146_14696


namespace NUMINAMATH_GPT_hypotenuse_length_l146_14611

noncomputable def hypotenuse_of_30_60_90_triangle (r : ℝ) : ℝ :=
  let a := (r * 3) / Real.sqrt 3
  2 * a

theorem hypotenuse_length (r : ℝ) (h : r = 3) : hypotenuse_of_30_60_90_triangle r = 6 * Real.sqrt 3 :=
  by sorry

end NUMINAMATH_GPT_hypotenuse_length_l146_14611


namespace NUMINAMATH_GPT_circumference_of_smaller_circle_l146_14603

variable (R : ℝ)
variable (A_shaded : ℝ)

theorem circumference_of_smaller_circle :
  (A_shaded = (32 / π) ∧ 3 * (π * R ^ 2) - π * R ^ 2 = A_shaded) → 
  2 * π * R = 4 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_smaller_circle_l146_14603


namespace NUMINAMATH_GPT_time_difference_l146_14670

-- Define the conditions
def time_to_nile_delta : Nat := 4
def number_of_alligators : Nat := 7
def combined_walking_time : Nat := 46

-- Define the mathematical statement we want to prove
theorem time_difference (x : Nat) :
  4 + 7 * (time_to_nile_delta + x) = combined_walking_time → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_time_difference_l146_14670


namespace NUMINAMATH_GPT_tip_percentage_correct_l146_14600

def lunch_cost := 50.20
def total_spent := 60.24
def tip_percentage := ((total_spent - lunch_cost) / lunch_cost) * 100

theorem tip_percentage_correct : tip_percentage = 19.96 := 
by
  sorry

end NUMINAMATH_GPT_tip_percentage_correct_l146_14600


namespace NUMINAMATH_GPT_number_of_prime_factors_30_factorial_l146_14690

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_prime_factors_30_factorial_l146_14690


namespace NUMINAMATH_GPT_distance_AB_l146_14678

def A : ℝ := -1
def B : ℝ := 2023

theorem distance_AB : |B - A| = 2024 := by
  sorry

end NUMINAMATH_GPT_distance_AB_l146_14678


namespace NUMINAMATH_GPT_sum_of_squares_eight_l146_14675

theorem sum_of_squares_eight (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := 
  sorry

end NUMINAMATH_GPT_sum_of_squares_eight_l146_14675


namespace NUMINAMATH_GPT_train_speed_l146_14676

theorem train_speed (length : ℝ) (time_seconds : ℝ) (speed : ℝ) :
  length = 320 → time_seconds = 16 → speed = 72 :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l146_14676


namespace NUMINAMATH_GPT_bucket_full_weight_l146_14637

theorem bucket_full_weight (x y c d : ℝ) 
  (h1 : x + (3/4) * y = c)
  (h2 : x + (3/5) * y = d) :
  x + y = (5/3) * c - (5/3) * d :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l146_14637


namespace NUMINAMATH_GPT_trip_to_museum_l146_14685

theorem trip_to_museum (x y z w : ℕ) 
  (h2 : y = 2 * x) 
  (h3 : z = 2 * x - 6) 
  (h4 : w = x + 9) 
  (htotal : x + y + z + w = 75) : 
  x = 12 := 
by 
  sorry

end NUMINAMATH_GPT_trip_to_museum_l146_14685


namespace NUMINAMATH_GPT_distance_between_neg5_and_neg1_l146_14655

theorem distance_between_neg5_and_neg1 : 
  dist (-5 : ℝ) (-1) = 4 := by
sorry

end NUMINAMATH_GPT_distance_between_neg5_and_neg1_l146_14655


namespace NUMINAMATH_GPT_fraction_to_decimal_l146_14673

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l146_14673


namespace NUMINAMATH_GPT_solve_for_x_l146_14686

-- Define the custom operation for real numbers
def custom_op (a b c d : ℝ) : ℝ := a * c - b * d

-- The theorem to prove
theorem solve_for_x (x : ℝ) (h : custom_op (-x) 3 (x - 2) (-6) = 10) :
  x = 4 ∨ x = -2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l146_14686


namespace NUMINAMATH_GPT_right_triangle_satisfies_pythagorean_l146_14689

-- Definition of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove
theorem right_triangle_satisfies_pythagorean :
  a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_satisfies_pythagorean_l146_14689


namespace NUMINAMATH_GPT_contrapositive_of_square_sum_zero_l146_14692

theorem contrapositive_of_square_sum_zero (a b : ℝ) :
  (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_square_sum_zero_l146_14692


namespace NUMINAMATH_GPT_KatieMarbles_l146_14606

variable {O P : ℕ}

theorem KatieMarbles :
  13 + O + P = 33 → P = 4 * O → 13 - O = 9 :=
by
  sorry

end NUMINAMATH_GPT_KatieMarbles_l146_14606


namespace NUMINAMATH_GPT_number_of_green_balls_l146_14674

theorem number_of_green_balls
  (total_balls white_balls yellow_balls red_balls purple_balls : ℕ)
  (prob : ℚ)
  (H_total : total_balls = 100)
  (H_white : white_balls = 50)
  (H_yellow : yellow_balls = 10)
  (H_red : red_balls = 7)
  (H_purple : purple_balls = 3)
  (H_prob : prob = 0.9) :
  ∃ (green_balls : ℕ), 
    (white_balls + green_balls + yellow_balls) / total_balls = prob ∧ green_balls = 30 := by
  sorry

end NUMINAMATH_GPT_number_of_green_balls_l146_14674


namespace NUMINAMATH_GPT_time_for_trains_to_cross_l146_14653

def length_train1 := 500 -- 500 meters
def length_train2 := 750 -- 750 meters
def speed_train1 := 60 * 1000 / 3600 -- 60 km/hr to m/s
def speed_train2 := 40 * 1000 / 3600 -- 40 km/hr to m/s
def relative_speed := speed_train1 + speed_train2 -- relative speed in m/s
def combined_length := length_train1 + length_train2 -- sum of lengths of both trains

theorem time_for_trains_to_cross :
  (combined_length / relative_speed) = 45 := 
by
  sorry

end NUMINAMATH_GPT_time_for_trains_to_cross_l146_14653


namespace NUMINAMATH_GPT_cos_pi_minus_2alpha_l146_14656

theorem cos_pi_minus_2alpha {α : ℝ} (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_2alpha_l146_14656


namespace NUMINAMATH_GPT_vector_addition_proof_l146_14612

def vector_add (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 + b.1, a.2 + b.2)

theorem vector_addition_proof :
  let a := (2, 0)
  let b := (-1, -2)
  vector_add a b = (1, -2) :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_proof_l146_14612
