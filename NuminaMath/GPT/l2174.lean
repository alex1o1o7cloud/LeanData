import Mathlib

namespace combined_area_win_bonus_l2174_217487

theorem combined_area_win_bonus (r : ℝ) (P_win P_bonus : ℝ) : 
  r = 8 → P_win = 1 / 4 → P_bonus = 1 / 8 → 
  (P_win * (Real.pi * r^2) + P_bonus * (Real.pi * r^2) = 24 * Real.pi) :=
by
  intro h_r h_Pwin h_Pbonus
  rw [h_r, h_Pwin, h_Pbonus]
  -- Calculation is skipped as per the instructions
  sorry

end combined_area_win_bonus_l2174_217487


namespace large_square_pattern_l2174_217415

theorem large_square_pattern :
  999999^2 = 1000000 * 999998 + 1 :=
by sorry

end large_square_pattern_l2174_217415


namespace wedding_cost_l2174_217446

theorem wedding_cost (venue_cost food_drink_cost guests_john : ℕ) 
  (guest_increment decorations_base decorations_per_guest transport_couple transport_per_guest entertainment_cost surchage_rate discount_thresh : ℕ) (discount_rate : ℕ) :
  let guests_wife := guests_john + (guests_john * guest_increment / 100)
  let venue_total := venue_cost + (venue_cost * surchage_rate / 100)
  let food_drink_total := if guests_wife > discount_thresh then (food_drink_cost * guests_wife) * (100 - discount_rate) / 100 else food_drink_cost * guests_wife
  let decorations_total := decorations_base + (decorations_per_guest * guests_wife)
  let transport_total := transport_couple + (transport_per_guest * guests_wife)
  (venue_total + food_drink_total + decorations_total + transport_total + entertainment_cost = 56200) :=
by {
  -- Constants given in the conditions
  let venue_cost := 10000
  let food_drink_cost := 500
  let guests_john := 50
  let guest_increment := 60
  let decorations_base := 2500
  let decorations_per_guest := 10
  let transport_couple := 200
  let transport_per_guest := 15
  let entertainment_cost := 4000
  let surchage_rate := 15
  let discount_thresh := 75
  let discount_rate := 10
  sorry
}

end wedding_cost_l2174_217446


namespace eval_f_3_minus_f_neg_3_l2174_217463

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 7 * x

-- State the theorem
theorem eval_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 := by
  sorry

end eval_f_3_minus_f_neg_3_l2174_217463


namespace increase_80_by_135_percent_l2174_217497

theorem increase_80_by_135_percent : 
  let original := 80 
  let increase := 1.35 
  original + (increase * original) = 188 := 
by
  sorry

end increase_80_by_135_percent_l2174_217497


namespace percentage_of_smoking_teens_l2174_217405

theorem percentage_of_smoking_teens (total_students : ℕ) (hospitalized_percentage : ℝ) (non_hospitalized_count : ℕ) 
  (h_total_students : total_students = 300)
  (h_hospitalized_percentage : hospitalized_percentage = 0.70)
  (h_non_hospitalized_count : non_hospitalized_count = 36) : 
  (non_hospitalized_count / (total_students * (1 - hospitalized_percentage))) * 100 = 40 := 
by 
  sorry

end percentage_of_smoking_teens_l2174_217405


namespace integer_roots_of_poly_l2174_217426

-- Define the polynomial
def poly (x : ℤ) (b1 b2 : ℤ) : ℤ :=
  x^3 + b2 * x ^ 2 + b1 * x + 18

-- The list of possible integer roots
def possible_integer_roots := [-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18]

-- Statement of the theorem
theorem integer_roots_of_poly (b1 b2 : ℤ) :
  ∀ x : ℤ, poly x b1 b2 = 0 → x ∈ possible_integer_roots :=
sorry

end integer_roots_of_poly_l2174_217426


namespace wendy_furniture_time_l2174_217476

variable (chairs tables pieces minutes total_time : ℕ)

theorem wendy_furniture_time (h1 : chairs = 4) (h2 : tables = 4) (h3 : pieces = chairs + tables) (h4 : minutes = 6) (h5 : total_time = pieces * minutes) : total_time = 48 :=
by
  sorry

end wendy_furniture_time_l2174_217476


namespace problem_statement_l2174_217432

open Nat

theorem problem_statement (n a : ℕ) 
  (hn : n > 1) 
  (ha : a > n^2)
  (H : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k, a + i = (n^2 + i) * k) :
  a > n^4 - n^3 := 
sorry

end problem_statement_l2174_217432


namespace where_they_meet_l2174_217411

/-- Define the conditions under which Petya and Vasya are walking. -/
structure WalkingCondition (n : ℕ) where
  lampposts : ℕ
  start_p : ℕ
  start_v : ℕ
  position_p : ℕ
  position_v : ℕ

/-- Initial conditions based on the problem statement. -/
def initialCondition : WalkingCondition 100 := {
  lampposts := 100,
  start_p := 1,
  start_v := 100,
  position_p := 22,
  position_v := 88
}

/-- Prove Petya and Vasya will meet at the 64th lamppost. -/
theorem where_they_meet (cond : WalkingCondition 100) : 64 ∈ { x | x = 64 } :=
  -- The formal proof would go here.
  sorry

end where_they_meet_l2174_217411


namespace find_code_l2174_217454

theorem find_code (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 11 * (A + B + C) = 242) :
  A = 5 ∧ B = 8 ∧ C = 9 ∨ A = 5 ∧ B = 9 ∧ C = 8 :=
by
  sorry

end find_code_l2174_217454


namespace fraction_walk_is_three_twentieths_l2174_217424

-- Define the various fractions given in the conditions
def fraction_bus : ℚ := 1 / 2
def fraction_auto : ℚ := 1 / 4
def fraction_bicycle : ℚ := 1 / 10

-- Defining the total fraction for students that do not walk
def total_not_walk : ℚ := fraction_bus + fraction_auto + fraction_bicycle

-- The remaining fraction after subtracting from 1
def fraction_walk : ℚ := 1 - total_not_walk

-- The theorem we want to prove that fraction_walk is 3/20
theorem fraction_walk_is_three_twentieths : fraction_walk = 3 / 20 := by
  sorry

end fraction_walk_is_three_twentieths_l2174_217424


namespace eval_expression_l2174_217478

-- Define the expression to evaluate
def expression : ℚ := 2 * 3 + 4 - (5 / 6)

-- Prove the equivalence of the evaluated expression to the expected result
theorem eval_expression : expression = 37 / 3 :=
by
  -- The detailed proof steps are omitted (relying on sorry)
  sorry

end eval_expression_l2174_217478


namespace smaller_angle_is_70_l2174_217433

def measure_of_smaller_angle (x : ℕ) : Prop :=
  (x + (x + 40) = 180) ∧ (2 * x - 60 = 80)

theorem smaller_angle_is_70 {x : ℕ} : measure_of_smaller_angle x → x = 70 :=
by
  sorry

end smaller_angle_is_70_l2174_217433


namespace feet_of_wood_required_l2174_217453

def rung_length_in_inches : ℤ := 18
def spacing_between_rungs_in_inches : ℤ := 6
def height_to_climb_in_feet : ℤ := 50

def feet_per_rung := rung_length_in_inches / 12
def rungs_per_foot := 12 / spacing_between_rungs_in_inches
def total_rungs := height_to_climb_in_feet * rungs_per_foot
def total_feet_of_wood := total_rungs * feet_per_rung

theorem feet_of_wood_required :
  total_feet_of_wood = 150 :=
by
  sorry

end feet_of_wood_required_l2174_217453


namespace score_of_B_is_correct_l2174_217490

theorem score_of_B_is_correct (A B C D E : ℝ)
  (h1 : (A + B + C + D + E) / 5 = 90)
  (h2 : (A + B + C) / 3 = 86)
  (h3 : (B + D + E) / 3 = 95) : 
  B = 93 := 
by 
  sorry

end score_of_B_is_correct_l2174_217490


namespace ratio_of_volumes_of_tetrahedrons_l2174_217442

theorem ratio_of_volumes_of_tetrahedrons (a b : ℝ) (h : a / b = 1 / 2) : (a^3) / (b^3) = 1 / 8 :=
by
-- proof goes here
sorry

end ratio_of_volumes_of_tetrahedrons_l2174_217442


namespace baker_sold_more_cakes_than_pastries_l2174_217471

theorem baker_sold_more_cakes_than_pastries (cakes_sold pastries_sold : ℕ) 
  (h_cakes_sold : cakes_sold = 158) (h_pastries_sold : pastries_sold = 147) : 
  (cakes_sold - pastries_sold) = 11 := by
  sorry

end baker_sold_more_cakes_than_pastries_l2174_217471


namespace maya_lift_increase_l2174_217404

def initial_lift_America : ℕ := 240
def peak_lift_America : ℕ := 300

def initial_lift_Maya (a_lift : ℕ) : ℕ := a_lift / 4
def peak_lift_Maya (p_lift : ℕ) : ℕ := p_lift / 2

def lift_difference (initial_lift : ℕ) (peak_lift : ℕ) : ℕ := peak_lift - initial_lift

theorem maya_lift_increase :
  lift_difference (initial_lift_Maya initial_lift_America) (peak_lift_Maya peak_lift_America) = 90 :=
by
  -- Proof is skipped with sorry
  sorry

end maya_lift_increase_l2174_217404


namespace yellow_balls_count_l2174_217450

theorem yellow_balls_count {R B Y G : ℕ} 
  (h1 : R + B + Y + G = 531)
  (h2 : R + B = Y + G + 31)
  (h3 : Y = G + 22) : 
  Y = 136 :=
by
  -- The proof is skipped, as requested.
  sorry

end yellow_balls_count_l2174_217450


namespace interchanged_digits_subtraction_l2174_217409

theorem interchanged_digits_subtraction (a b k : ℤ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) :=
by sorry

end interchanged_digits_subtraction_l2174_217409


namespace women_science_majors_is_30_percent_l2174_217462

noncomputable def percentage_women_science_majors (ns_percent : ℝ) (m_percent : ℝ) (m_sci_percent : ℝ) : ℝ :=
  let w_percent := 1 - m_percent
  let m_sci_total := m_percent * m_sci_percent
  let total_sci := 1 - ns_percent
  let w_sci_total := total_sci - m_sci_total
  (w_sci_total / w_percent) * 100

theorem women_science_majors_is_30_percent :
  percentage_women_science_majors 0.60 0.40 0.55 = 30 := by
  sorry

end women_science_majors_is_30_percent_l2174_217462


namespace shoes_cost_l2174_217447

theorem shoes_cost (S : ℝ) : 
  let suit := 430
  let discount := 100
  let total_paid := 520
  suit + S - discount = total_paid -> 
  S = 190 :=
by 
  intro h
  sorry

end shoes_cost_l2174_217447


namespace find_k_l2174_217403

theorem find_k : ∃ k : ℚ, (k = (k + 4) / 4) ∧ k = 4 / 3 :=
by
  sorry

end find_k_l2174_217403


namespace minimum_value_a_l2174_217443

noncomputable def f (a b x : ℝ) := a * Real.log x - (1 / 2) * x^2 + b * x

theorem minimum_value_a (h : ∀ b x : ℝ, x > 0 → f a b x > 0) : a ≥ -Real.exp 3 := 
sorry

end minimum_value_a_l2174_217443


namespace find_m_l2174_217434

def circle1 (x y m : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

theorem find_m (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, 
    circle1 x1 y1 m ∧ 
    circle2 x2 y2 m ∧ 
    (m + 2)^2 + (-1 - m)^2 = 25 → 
    m = 2 :=
by
  sorry

end find_m_l2174_217434


namespace simplify_expression_l2174_217451

noncomputable def sqrt' (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_expression :
  (3 * sqrt' 8 / (sqrt' 2 + sqrt' 3 + sqrt' 7)) = (sqrt' 2 + sqrt' 3 - sqrt' 7) := 
by
  sorry

end simplify_expression_l2174_217451


namespace distance_inequality_l2174_217406

theorem distance_inequality 
  (A B C D : Point)
  (dist : Point → Point → ℝ)
  (h_dist_pos : ∀ P Q : Point, dist P Q ≥ 0)
  (AC BD AD BC AB CD : ℝ)
  (hAC : AC = dist A C)
  (hBD : BD = dist B D)
  (hAD : AD = dist A D)
  (hBC : BC = dist B C)
  (hAB : AB = dist A B)
  (hCD : CD = dist C D) :
  AC^2 + BD^2 + AD^2 + BC^2 ≥ AB^2 + CD^2 := 
by
  sorry

end distance_inequality_l2174_217406


namespace jessica_initial_withdrawal_fraction_l2174_217410

variable {B : ℝ} -- this is the initial balance

noncomputable def initial_withdrawal_fraction (B : ℝ) : Prop :=
  let remaining_balance := B - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 → (400 / B) = 2 / 5

-- Our goal is to prove the statement given conditions.
theorem jessica_initial_withdrawal_fraction : 
  ∃ B : ℝ, initial_withdrawal_fraction B :=
sorry

end jessica_initial_withdrawal_fraction_l2174_217410


namespace distance_traveled_l2174_217480

theorem distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 12) : D = 100 := 
sorry

end distance_traveled_l2174_217480


namespace trigonometric_ratio_sum_l2174_217445

open Real

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h₁ : sin x / sin y = 2) 
  (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 41 / 57 := 
by
  sorry

end trigonometric_ratio_sum_l2174_217445


namespace min_max_pieces_three_planes_l2174_217461

theorem min_max_pieces_three_planes : 
  ∃ (min max : ℕ), (min = 4) ∧ (max = 8) := by
  sorry

end min_max_pieces_three_planes_l2174_217461


namespace a_eq_zero_l2174_217427

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, ax + 2 ≠ 0) : a = 0 := by
  sorry

end a_eq_zero_l2174_217427


namespace M_minus_N_positive_l2174_217416

variable (a b : ℝ)

def M : ℝ := 10 * a^2 + b^2 - 7 * a + 8
def N : ℝ := a^2 + b^2 + 5 * a + 1

theorem M_minus_N_positive : M a b - N a b ≥ 3 := by
  sorry

end M_minus_N_positive_l2174_217416


namespace find_b_l2174_217481

def oscillation_period (a b c d : ℝ) (oscillations : ℝ) : Prop :=
  oscillations = 5 * (2 * Real.pi) / b

theorem find_b
  (a b c d : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0)
  (osc_complexity: oscillation_period a b c d 5):
  b = 5 := by
  sorry

end find_b_l2174_217481


namespace digit_d_is_six_l2174_217422

theorem digit_d_is_six (d : ℕ) (h_even : d % 2 = 0) (h_digits_sum : 7 + 4 + 8 + 2 + d % 9 = 0) : d = 6 :=
by 
  sorry

end digit_d_is_six_l2174_217422


namespace calculate_correctly_l2174_217413

theorem calculate_correctly (n : ℕ) (h1 : n - 21 = 52) : n - 40 = 33 := 
by 
  sorry

end calculate_correctly_l2174_217413


namespace directrix_parabola_y_eq_2x2_l2174_217499

theorem directrix_parabola_y_eq_2x2 : (∃ y : ℝ, y = 2 * x^2) → (∃ y : ℝ, y = -1/8) :=
by
  sorry

end directrix_parabola_y_eq_2x2_l2174_217499


namespace train_speed_incl_stoppages_l2174_217439

theorem train_speed_incl_stoppages
  (speed_excl_stoppages : ℝ)
  (stoppage_time_minutes : ℝ)
  (h1 : speed_excl_stoppages = 42)
  (h2 : stoppage_time_minutes = 21.428571428571423)
  : ∃ speed_incl_stoppages, speed_incl_stoppages = 27 := 
sorry

end train_speed_incl_stoppages_l2174_217439


namespace domain_of_composite_function_l2174_217407

theorem domain_of_composite_function
    (f : ℝ → ℝ)
    (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → f (x + 1) ∈ (Set.Icc (-2:ℝ) (3:ℝ))):
    ∃ s : Set ℝ, s = Set.Icc 0 (5/2) ∧ (∀ x, x ∈ s ↔ f (2 * x - 1) ∈ Set.Icc (-1) 4) :=
by
  sorry

end domain_of_composite_function_l2174_217407


namespace base_square_eq_l2174_217436

theorem base_square_eq (b : ℕ) (h : (3*b + 3)^2 = b^3 + 2*b^2 + 3*b) : b = 9 :=
sorry

end base_square_eq_l2174_217436


namespace solve_for_P_l2174_217429

theorem solve_for_P (P : Real) (h : (P ^ 4) ^ (1 / 3) = 9 * 81 ^ (1 / 9)) : P = 3 ^ (11 / 6) :=
by
  sorry

end solve_for_P_l2174_217429


namespace find_dividend_l2174_217425

-- Define the conditions
def divisor : ℕ := 20
def quotient : ℕ := 8
def remainder : ℕ := 6

-- Lean 4 statement to prove the dividend
theorem find_dividend : (divisor * quotient + remainder) = 166 := by
  sorry

end find_dividend_l2174_217425


namespace real_number_identity_l2174_217444

theorem real_number_identity (a : ℝ) (h : a^2 - a - 1 = 0) : a^8 + 7 * a^(-(4:ℝ)) = 48 := by
  sorry

end real_number_identity_l2174_217444


namespace balance_scale_equation_l2174_217452

theorem balance_scale_equation 
  (G Y B W : ℝ)
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 6 * B)
  (h3 : 2 * B = 3 * W) : 
  3 * G + 4 * Y + 3 * W = 16 * B :=
by
  sorry

end balance_scale_equation_l2174_217452


namespace sum_of_roots_even_l2174_217475

theorem sum_of_roots_even (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
    (h_distinct : ∃ x y : ℤ, x ≠ y ∧ (x^2 - 2 * p * x + (p * q) = 0) ∧ (y^2 - 2 * p * y + (p * q) = 0)) :
    Even (2 * p) :=
by 
  sorry

end sum_of_roots_even_l2174_217475


namespace part1_part2_l2174_217460

-- Definitions
def A (x : ℝ) : Prop := (x + 2) / (x - 3 / 2) < 0
def B (x : ℝ) (m : ℝ) : Prop := x^2 - (m + 1) * x + m ≤ 0

-- Part (1): when m = 2, find A ∪ B
theorem part1 :
  (∀ x, A x ∨ B x 2) ↔ ∀ x, -2 < x ∧ x ≤ 2 := sorry

-- Part (2): find the range of real number m
theorem part2 :
  (∀ x, A x → B x m) ↔ (-2 < m ∧ m < 3 / 2) := sorry

end part1_part2_l2174_217460


namespace abs_expression_equals_l2174_217418

theorem abs_expression_equals (h : Real.pi < 12) : 
  abs (Real.pi - abs (Real.pi - 12)) = 12 - 2 * Real.pi := 
by
  sorry

end abs_expression_equals_l2174_217418


namespace hyperbola_transformation_l2174_217496

def equation_transform (x y : ℝ) : Prop :=
  y = (1 - 3 * x) / (2 * x - 1)

def coordinate_shift (x y X Y : ℝ) : Prop :=
  X = x - 0.5 ∧ Y = y + 1.5

theorem hyperbola_transformation (x y X Y : ℝ) :
  equation_transform x y →
  coordinate_shift x y X Y →
  (X * Y = -0.25) :=
by
  sorry

end hyperbola_transformation_l2174_217496


namespace solution_set_of_quadratic_inequality_l2174_217455

theorem solution_set_of_quadratic_inequality (a : ℝ) (x : ℝ) :
  (∀ x, 0 < x - 0.5 ∧ x < 2 → ax^2 + 5 * x - 2 > 0) ∧ a = -2 →
  (∀ x, -3 < x ∧ x < 0.5 → a * x^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end solution_set_of_quadratic_inequality_l2174_217455


namespace problem_l2174_217459

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end problem_l2174_217459


namespace base_seven_to_ten_l2174_217470

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l2174_217470


namespace no_prime_sum_10003_l2174_217493

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l2174_217493


namespace art_museum_visitors_l2174_217474

theorem art_museum_visitors 
  (V : ℕ)
  (H1 : ∃ (d : ℕ), d = 130)
  (H2 : ∃ (e u : ℕ), e = u)
  (H3 : ∃ (x : ℕ), x = (3 * V) / 4)
  (H4 : V = (3 * V) / 4 + 130) :
  V = 520 :=
sorry

end art_museum_visitors_l2174_217474


namespace cleanup_drive_weight_per_mile_per_hour_l2174_217428

theorem cleanup_drive_weight_per_mile_per_hour :
  let duration := 4
  let lizzie_group := 387
  let second_group := lizzie_group - 39
  let third_group := 560 / 16
  let total_distance := 8
  let total_garbage := lizzie_group + second_group + third_group
  total_garbage / total_distance / duration = 24.0625 := 
by {
  sorry
}

end cleanup_drive_weight_per_mile_per_hour_l2174_217428


namespace investment_value_l2174_217448

-- Define the compound interest calculation
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Given values
def P : ℝ := 8000
def r : ℝ := 0.05
def n : ℕ := 7

-- The theorem statement in Lean 4
theorem investment_value :
  round (compound_interest P r n) = 11257 :=
by
  sorry

end investment_value_l2174_217448


namespace tom_batteries_used_total_l2174_217489

def batteries_used_in_flashlights : Nat := 2 * 3
def batteries_used_in_toys : Nat := 4 * 5
def batteries_used_in_controllers : Nat := 2 * 6
def total_batteries_used : Nat := batteries_used_in_flashlights + batteries_used_in_toys + batteries_used_in_controllers

theorem tom_batteries_used_total : total_batteries_used = 38 :=
by
  sorry

end tom_batteries_used_total_l2174_217489


namespace rectangular_prism_edges_vertices_faces_sum_l2174_217420

theorem rectangular_prism_edges_vertices_faces_sum (a b c : ℕ) (h1: a = 2) (h2: b = 3) (h3: c = 4) : 
  12 + 8 + 6 = 26 :=
by
  sorry

end rectangular_prism_edges_vertices_faces_sum_l2174_217420


namespace keith_total_spent_l2174_217482

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tire_cost : ℝ := 112.46
def num_tires : ℕ := 4
def printer_cable_cost : ℝ := 14.85
def num_printer_cables : ℕ := 2
def blank_cd_pack_cost : ℝ := 0.98
def num_blank_cds : ℕ := 10
def sales_tax_rate : ℝ := 0.0825

theorem keith_total_spent : 
  speakers_cost +
  cd_player_cost +
  (num_tires * tire_cost) +
  (num_printer_cables * printer_cable_cost) +
  (num_blank_cds * blank_cd_pack_cost) *
  (1 + sales_tax_rate) = 827.87 := 
sorry

end keith_total_spent_l2174_217482


namespace clothing_store_earnings_l2174_217492

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l2174_217492


namespace find_fraction_of_cistern_l2174_217484

noncomputable def fraction_initially_full (x : ℝ) : Prop :=
  let rateA := (1 - x) / 12
  let rateB := (1 - x) / 8
  let combined_rate := 1 / 14.4
  combined_rate = rateA + rateB

theorem find_fraction_of_cistern {x : ℝ} (h : fraction_initially_full x) : x = 2 / 3 :=
by
  sorry

end find_fraction_of_cistern_l2174_217484


namespace hurdle_distance_l2174_217491

theorem hurdle_distance (d : ℝ) : 
  50 + 11 * d + 55 = 600 → d = 45 := by
  sorry

end hurdle_distance_l2174_217491


namespace evaluate_expression_l2174_217483

theorem evaluate_expression :
  500 * 997 * 0.0997 * 10^2 = 5 * (997:ℝ)^2 :=
by
  sorry

end evaluate_expression_l2174_217483


namespace base_number_is_2_l2174_217414

open Real

noncomputable def valid_x (x : ℝ) (n : ℕ) := sqrt (x^n) = 64

theorem base_number_is_2 (x : ℝ) (n : ℕ) (h : valid_x x n) (hn : n = 12) : x = 2 := 
by 
  sorry

end base_number_is_2_l2174_217414


namespace banana_price_reduction_l2174_217421

theorem banana_price_reduction (P_r : ℝ) (P : ℝ) (n : ℝ) (m : ℝ) (h1 : P_r = 3) (h2 : n = 40) (h3 : m = 64) 
  (h4 : 160 = (n / P_r) * 12) 
  (h5 : 96 = 160 - m) 
  (h6 : (40 / 8) = P) :
  (P - P_r) / P * 100 = 40 :=
by
  sorry

end banana_price_reduction_l2174_217421


namespace problem_statement_l2174_217488

def approx_digit_place (num : ℕ) : ℕ :=
if num = 3020000 then 0 else sorry

theorem problem_statement :
  approx_digit_place (3 * 10^6 + 2 * 10^4) = 0 :=
by
  sorry

end problem_statement_l2174_217488


namespace calculate_material_needed_l2174_217419

theorem calculate_material_needed (area : ℝ) (pi_approx : ℝ) (extra_material : ℝ) (r : ℝ) (C : ℝ) : 
  area = 50.24 → pi_approx = 3.14 → extra_material = 4 → pi_approx * r ^ 2 = area → 
  C = 2 * pi_approx * r →
  C + extra_material = 29.12 :=
by
  intros h_area h_pi h_extra h_area_eq h_C_eq
  sorry

end calculate_material_needed_l2174_217419


namespace part_I_part_II_l2174_217431

variable (x : ℝ)

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define complement of B in real numbers
def neg_RB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Part I: Statement for a = -2
theorem part_I (a : ℝ) (h : a = -2) : A a ∩ neg_RB = {x | -1 ≤ x ∧ x ≤ 1} := by
  sorry

-- Part II: Statement for A ∪ B = B
theorem part_II (a : ℝ) (h : ∀ x, A a x -> B x) : a < -4 ∨ a > 5 := by
  sorry

end part_I_part_II_l2174_217431


namespace polynomial_roots_l2174_217477

theorem polynomial_roots : (∃ x : ℝ, (4 * x ^ 4 + 11 * x ^ 3 - 37 * x ^ 2 + 18 * x = 0) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = 3 / 2 ∨ x = -6)) :=
by 
  sorry

end polynomial_roots_l2174_217477


namespace length_of_AB_l2174_217457

-- Conditions:
-- The radius of the inscribed circle is 6 cm.
-- The triangle is a right triangle with a 60 degree angle at one vertex.
-- Question: Prove that the length of AB is 12 + 12√3 cm.

theorem length_of_AB (r : ℝ) (angle : ℝ) (h_radius : r = 6) (h_angle : angle = 60) :
  ∃ (AB : ℝ), AB = 12 + 12 * Real.sqrt 3 :=
by
  sorry

end length_of_AB_l2174_217457


namespace rate_times_base_eq_9000_l2174_217494

noncomputable def Rate : ℝ := 0.00015
noncomputable def BaseAmount : ℝ := 60000000

theorem rate_times_base_eq_9000 :
  Rate * BaseAmount = 9000 := 
  sorry

end rate_times_base_eq_9000_l2174_217494


namespace sum_of_reciprocals_of_roots_l2174_217473

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (hroots : ∀ x, (x = p ∨ x = q ∨ x = r) ↔ (30*x^3 - 50*x^2 + 22*x - 1 = 0)) 
  (h0 : 0 < p ∧ p < 1) (h1 : 0 < q ∧ q < 1) (h2 : 0 < r ∧ r < 1) 
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - r)) = 12 := 
by 
  sorry

end sum_of_reciprocals_of_roots_l2174_217473


namespace proof_G_eq_BC_eq_D_eq_AB_AC_l2174_217408

-- Let's define the conditions of the problem first
variables (A B C O D G F E : Type) [Field A] [Field B] [Field C] [Field O] [Field D] [Field G] [Field F] [Field E]

-- Given triangle ABC with circumcenter O
variable {triangle_ABC: Prop}

-- Given point D on line segment BC
variable (D_on_BC : Prop)

-- Given circle Gamma with diameter OD
variable (circle_Gamma : Prop)

-- Given circles Gamma_1 and Gamma_2 are circumcircles of triangles ABD and ACD respectively
variable (circle_Gamma1 : Prop)
variable (circle_Gamma2 : Prop)

-- Given points F and E as intersection points
variable (intersect_F : Prop)
variable (intersect_E : Prop)

-- Given G as the second intersection point of the circumcircles of triangles BED and DFC
variable (second_intersect_G : Prop)

-- Prove that the condition for point G to be equidistant from points B and C is that point D is equidistant from lines AB and AC
theorem proof_G_eq_BC_eq_D_eq_AB_AC : 
  triangle_ABC ∧ D_on_BC ∧ circle_Gamma ∧ circle_Gamma1 ∧ circle_Gamma2 ∧ intersect_F ∧ intersect_E ∧ second_intersect_G → 
  G_dist_BC ↔ D_dist_AB_AC :=
by
  sorry

end proof_G_eq_BC_eq_D_eq_AB_AC_l2174_217408


namespace ball_travel_distance_fourth_hit_l2174_217472

theorem ball_travel_distance_fourth_hit :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let distances := [initial_height, 
                    initial_height * rebound_ratio, 
                    initial_height * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio]
  distances.sum = 294 + 1 / 3 := by
  sorry

end ball_travel_distance_fourth_hit_l2174_217472


namespace color_of_85th_bead_l2174_217430

def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def bead_color (n : ℕ) : String :=
  bead_pattern.get! (n % bead_pattern.length)

theorem color_of_85th_bead : bead_color 84 = "yellow" := 
by
  sorry

end color_of_85th_bead_l2174_217430


namespace triangle_isosceles_l2174_217495

theorem triangle_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) :
  b = c → IsoscelesTriangle := 
by
  sorry

end triangle_isosceles_l2174_217495


namespace students_transferred_l2174_217479

theorem students_transferred (students_before : ℕ) (total_students : ℕ) (students_equal : ℕ) 
  (h1 : students_before = 23) (h2 : total_students = 50) (h3 : students_equal = total_students / 2) : 
  (∃ x : ℕ, students_equal = students_before + x) → (∃ x : ℕ, x = 2) :=
by
  -- h1: students_before = 23
  -- h2: total_students = 50
  -- h3: students_equal = total_students / 2
  -- to prove: ∃ x : ℕ, students_equal = students_before + x → ∃ x : ℕ, x = 2
  sorry

end students_transferred_l2174_217479


namespace greatest_prime_factor_of_factorial_sum_l2174_217441

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p, Prime p ∧ p > 11 ∧ (∀ q, Prime q ∧ q > 11 → q ≤ 61) ∧ p = 61 :=
by
  sorry

end greatest_prime_factor_of_factorial_sum_l2174_217441


namespace sum_series_eq_two_l2174_217466

noncomputable def series_term (n : ℕ) : ℚ := (3 * n - 2) / (n * (n + 1) * (n + 2))

theorem sum_series_eq_two :
  ∑' n : ℕ, series_term (n + 1) = 2 :=
sorry

end sum_series_eq_two_l2174_217466


namespace problem_l2174_217412

variable (p q : Prop)

theorem problem (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end problem_l2174_217412


namespace evaluate_expression_at_2_l2174_217468

-- Define the quadratic and linear components of the expression
def quadratic (x : ℝ) := 3 * x ^ 2 - 8 * x + 5
def linear (x : ℝ) := 4 * x - 7

-- State the proposition to evaluate the given expression at x = 2
theorem evaluate_expression_at_2 : quadratic 2 * linear 2 = 1 := by
  -- The proof is skipped by using sorry
  sorry

end evaluate_expression_at_2_l2174_217468


namespace gcd_12a_20b_min_value_l2174_217485

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0

def gcd_condition (a b d : ℕ) : Prop := gcd a b = d

-- State the problem
theorem gcd_12a_20b_min_value (a b : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_gcd_ab : gcd_condition a b 10) :
  ∃ (k : ℕ), k = gcd (12 * a) (20 * b) ∧ k = 40 :=
by
  sorry

end gcd_12a_20b_min_value_l2174_217485


namespace percentage_shaded_in_square_l2174_217437

theorem percentage_shaded_in_square
  (EFGH : Type)
  (square : EFGH → Prop)
  (side_length : EFGH → ℝ)
  (area : EFGH → ℝ)
  (shaded_area : EFGH → ℝ)
  (P : EFGH)
  (h_square : square P)
  (h_side_length : side_length P = 8)
  (h_area : area P = side_length P * side_length P)
  (h_small_shaded : shaded_area P = 4)
  (h_large_shaded : shaded_area P + 7 = 11) :
  (shaded_area P / area P) * 100 = 17.1875 :=
by
  sorry

end percentage_shaded_in_square_l2174_217437


namespace steven_apples_peaches_difference_l2174_217469

def steven_apples := 19
def jake_apples (steven_apples : ℕ) := steven_apples + 4
def jake_peaches (steven_peaches : ℕ) := steven_peaches - 3

theorem steven_apples_peaches_difference (P : ℕ) :
  19 - P = steven_apples - P :=
by
  sorry

end steven_apples_peaches_difference_l2174_217469


namespace calculate_selling_price_l2174_217423

-- Define the conditions
def purchase_price : ℝ := 900
def repair_cost : ℝ := 300
def gain_percentage : ℝ := 0.10

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define the gain
def gain : ℝ := gain_percentage * total_cost

-- Define the selling price
def selling_price : ℝ := total_cost + gain

-- The theorem to prove
theorem calculate_selling_price : selling_price = 1320 := by
  sorry

end calculate_selling_price_l2174_217423


namespace triangle_inequality_x_not_2_l2174_217464

theorem triangle_inequality_x_not_2 (x : ℝ) (h1 : 2 < x) (h2 : x < 8) : x ≠ 2 :=
by 
  sorry

end triangle_inequality_x_not_2_l2174_217464


namespace carla_marbles_l2174_217458

theorem carla_marbles (before now bought : ℝ) (h_before : before = 187.0) (h_now : now = 321) : bought = 134 :=
by
  sorry

end carla_marbles_l2174_217458


namespace jose_bottle_caps_l2174_217467

def jose_start : ℕ := 7
def rebecca_gives : ℕ := 2
def final_bottle_caps : ℕ := 9

theorem jose_bottle_caps :
  jose_start + rebecca_gives = final_bottle_caps :=
by
  sorry

end jose_bottle_caps_l2174_217467


namespace least_integer_value_x_l2174_217498

theorem least_integer_value_x (x : ℤ) : (3 * |2 * (x : ℤ) - 1| + 6 < 24) → x = -2 :=
by
  sorry

end least_integer_value_x_l2174_217498


namespace tony_fever_temperature_above_threshold_l2174_217417

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end tony_fever_temperature_above_threshold_l2174_217417


namespace max_min_value_of_f_l2174_217440

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_min_value_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f (Real.pi / 2) ≤ f x) :=
by
  sorry

end max_min_value_of_f_l2174_217440


namespace trig_identity_example_l2174_217449

theorem trig_identity_example : 4 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 := 
by
  -- The statement "π/12" is mathematically equivalent to 15 degrees.
  sorry

end trig_identity_example_l2174_217449


namespace inequality_a_neg_one_inequality_general_a_l2174_217400

theorem inequality_a_neg_one : ∀ x : ℝ, (x^2 + x - 2 > 0) ↔ (x < -2 ∨ x > 1) :=
by { sorry }

theorem inequality_general_a : 
∀ (a x : ℝ), ax^2 - (a + 2)*x + 2 < 0 ↔ 
  if a = 0 then x > 1
  else if a < 0 then x < (2 / a) ∨ x > 1
  else if 0 < a ∧ a < 2 then 1 < x ∧ x < (2 / a)
  else if a = 2 then False
  else (2 / a) < x ∧ x < 1 :=
by { sorry }

end inequality_a_neg_one_inequality_general_a_l2174_217400


namespace part1_part2_part3_l2174_217456

variables (a b c : ℤ)
-- Condition: For all integer values of x, (ax^2 + bx + c) is a square number 
def quadratic_is_square_for_any_x (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

-- Question (1): Prove that 2a, 2b, c are all integers
theorem part1 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ m n : ℤ, 2 * a = m ∧ 2 * b = n ∧ ∃ k₁ : ℤ, c = k₁ :=
sorry

-- Question (2): Prove that a, b, c are all integers, and c is a square number
theorem part2 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2 :=
sorry

-- Question (3): Prove that if (2) holds, it does not necessarily mean that 
-- for all integer values of x, (ax^2 + bx + c) is always a square number.
theorem part3 (a b c : ℤ) (h : ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2) : 
  ¬ quadratic_is_square_for_any_x a b c :=
sorry

end part1_part2_part3_l2174_217456


namespace original_deck_card_count_l2174_217402

theorem original_deck_card_count (r b : ℕ) 
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
sorry

end original_deck_card_count_l2174_217402


namespace share_price_increase_l2174_217465

theorem share_price_increase
  (P : ℝ)
  -- At the end of the first quarter, the share price was 20% higher than at the beginning of the year.
  (end_of_first_quarter : ℝ := 1.20 * P)
  -- The percent increase from the end of the first quarter to the end of the second quarter was 25%.
  (percent_increase_second_quarter : ℝ := 0.25)
  -- At the end of the second quarter, the share price
  (end_of_second_quarter : ℝ := end_of_first_quarter + percent_increase_second_quarter * end_of_first_quarter) :
  -- What is the percent increase in share price at the end of the second quarter compared to the beginning of the year?
  end_of_second_quarter = 1.50 * P :=
by
  sorry

end share_price_increase_l2174_217465


namespace distinct_triple_identity_l2174_217401

theorem distinct_triple_identity (p q r : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ r) 
  (h3 : r ≠ p)
  (h : (p / (q - r)) + (q / (r - p)) + (r / (p - q)) = 3) : 
  (p^2 / (q - r)^2) + (q^2 / (r - p)^2) + (r^2 / (p - q)^2) = 3 :=
by 
  sorry

end distinct_triple_identity_l2174_217401


namespace gcd_546_210_l2174_217435

theorem gcd_546_210 : Nat.gcd 546 210 = 42 := by
  sorry -- Proof is required to solve

end gcd_546_210_l2174_217435


namespace parabola_chord_solution_l2174_217438

noncomputable def parabola_chord : Prop :=
  ∃ x_A x_B : ℝ, (140 = 5 * x_B^2 + 2 * x_A^2) ∧ 
  ((x_A = -5 * Real.sqrt 2 ∧ x_B = 2 * Real.sqrt 2) ∨ 
   (x_A = 5 * Real.sqrt 2 ∧ x_B = -2 * Real.sqrt 2))

theorem parabola_chord_solution : parabola_chord := 
sorry

end parabola_chord_solution_l2174_217438


namespace part1_part2_l2174_217486

noncomputable def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -1 < x ∧ x < (5:ℝ)/3 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l2174_217486
