import Mathlib

namespace christian_age_in_eight_years_l134_13462

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end christian_age_in_eight_years_l134_13462


namespace product_of_b_l134_13497

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (y + 4) / 3

theorem product_of_b (b : ℝ) :
  g b 3 = g_inv b (b + 2) → b = 3 := 
by
  sorry

end product_of_b_l134_13497


namespace find_omega_l134_13445

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem find_omega (ω : ℝ) (h₁ : ∀ x₁ x₂, (-ω < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * ω) → f ω x₁ < f ω x₂)
  (h₂ : ∀ x, f ω x = f ω (-2 * ω - x)) :
  ω = Real.sqrt (3 * Real.pi) / 3 :=
by
  sorry

end find_omega_l134_13445


namespace coles_average_speed_l134_13450

theorem coles_average_speed (t_work : ℝ) (t_round : ℝ) (s_return : ℝ) (t_return : ℝ) (d : ℝ) (t_work_min : ℕ) :
  t_work_min = 72 ∧ t_round = 2 ∧ s_return = 90 ∧ 
  t_work = t_work_min / 60 ∧ t_return = t_round - t_work ∧ d = s_return * t_return →
  d / t_work = 60 := 
by
  intro h
  sorry

end coles_average_speed_l134_13450


namespace test_total_points_l134_13438

theorem test_total_points (computation_points_per_problem : ℕ) (word_points_per_problem : ℕ) (total_problems : ℕ) (computation_problems : ℕ) :
  computation_points_per_problem = 3 →
  word_points_per_problem = 5 →
  total_problems = 30 →
  computation_problems = 20 →
  (computation_problems * computation_points_per_problem + 
  (total_problems - computation_problems) * word_points_per_problem) = 110 :=
by
  intros h1 h2 h3 h4
  sorry

end test_total_points_l134_13438


namespace complex_addition_l134_13498

namespace ComplexProof

def B := (3 : ℂ) + (2 * Complex.I)
def Q := (-5 : ℂ)
def R := (2 * Complex.I)
def T := (3 : ℂ) + (5 * Complex.I)

theorem complex_addition :
  B - Q + R + T = (1 : ℂ) + (9 * Complex.I) := 
by
  sorry

end ComplexProof

end complex_addition_l134_13498


namespace find_A_l134_13486

theorem find_A (A7B : ℕ) (H1 : (A7B % 100) / 10 = 7) (H2 : A7B + 23 = 695) : (A7B / 100) = 6 := 
  sorry

end find_A_l134_13486


namespace average_weight_of_11_children_l134_13478

theorem average_weight_of_11_children (b: ℕ) (g: ℕ) (avg_b: ℕ) (avg_g: ℕ) (hb: b = 8) (hg: g = 3) (havg_b: avg_b = 155) (havg_g: avg_g = 115) : 
  (b * avg_b + g * avg_g) / (b + g) = 144 :=
by {
  sorry
}

end average_weight_of_11_children_l134_13478


namespace y_intercept_of_line_l134_13434

theorem y_intercept_of_line (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) (h_slope : m = -3) (h_intercept : (x₀, y₀) = (7, 0)) : (0, 21) = (0, (y₀ - m * x₀)) :=
by
  sorry

end y_intercept_of_line_l134_13434


namespace find_certain_number_l134_13453

theorem find_certain_number (x : ℝ) (h : 34 = (4/5) * x + 14) : x = 25 :=
by
  sorry

end find_certain_number_l134_13453


namespace price_after_two_reductions_l134_13487

variable (orig_price : ℝ) (m : ℝ)

def current_price (orig_price : ℝ) (m : ℝ) : ℝ :=
  orig_price * (1 - m) * (1 - m)

theorem price_after_two_reductions (h1 : orig_price = 100) (h2 : 0 ≤ m ∧ m ≤ 1) :
  current_price orig_price m = 100 * (1 - m) ^ 2 := by
    sorry

end price_after_two_reductions_l134_13487


namespace two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l134_13460

theorem two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one :
  (2.85 = 2850 * 0.001) := by
  sorry

end two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l134_13460


namespace player_jump_height_to_dunk_l134_13412

/-- Definitions given in the conditions -/
def rim_height : ℕ := 120
def player_height : ℕ := 72
def player_reach_above_head : ℕ := 22

/-- The statement to be proven -/
theorem player_jump_height_to_dunk :
  rim_height - (player_height + player_reach_above_head) = 26 :=
by
  sorry

end player_jump_height_to_dunk_l134_13412


namespace triangle_perimeter_l134_13409

theorem triangle_perimeter (side1 side2 side3 : ℕ) (h1 : side1 = 40) (h2 : side2 = 50) (h3 : side3 = 70) : 
  side1 + side2 + side3 = 160 :=
by 
  sorry

end triangle_perimeter_l134_13409


namespace student_ticket_cost_l134_13452

theorem student_ticket_cost :
  ∀ (S : ℤ),
  (525 - 388) * S + 388 * 6 = 2876 → S = 4 :=
by
  sorry

end student_ticket_cost_l134_13452


namespace number_line_problem_l134_13430

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l134_13430


namespace roadsters_paving_company_total_cement_l134_13471

noncomputable def cement_lexi : ℝ := 10
noncomputable def cement_tess : ℝ := cement_lexi + 0.20 * cement_lexi
noncomputable def cement_ben : ℝ := cement_tess - 0.10 * cement_tess
noncomputable def cement_olivia : ℝ := 2 * cement_ben

theorem roadsters_paving_company_total_cement :
  cement_lexi + cement_tess + cement_ben + cement_olivia = 54.4 := by
  sorry

end roadsters_paving_company_total_cement_l134_13471


namespace HCF_of_numbers_l134_13457

theorem HCF_of_numbers (a b : ℕ) (h₁ : a * b = 84942) (h₂ : Nat.lcm a b = 2574) : Nat.gcd a b = 33 :=
by
  sorry

end HCF_of_numbers_l134_13457


namespace number_being_divided_l134_13443

theorem number_being_divided (divisor quotient remainder number : ℕ) 
  (h_divisor : divisor = 3) 
  (h_quotient : quotient = 7) 
  (h_remainder : remainder = 1)
  (h_number : number = divisor * quotient + remainder) : 
  number = 22 :=
by
  rw [h_divisor, h_quotient, h_remainder] at h_number
  exact h_number

end number_being_divided_l134_13443


namespace c_payment_l134_13475

theorem c_payment 
  (A_rate : ℝ) (B_rate : ℝ) (days : ℝ) (total_payment : ℝ) (C_fraction : ℝ) 
  (hA : A_rate = 1 / 6) 
  (hB : B_rate = 1 / 8) 
  (hdays : days = 3) 
  (hpayment : total_payment = 3200)
  (hC_fraction : C_fraction = 1 / 8) :
  total_payment * C_fraction = 400 :=
by {
  -- The proof would go here
  sorry
}

end c_payment_l134_13475


namespace total_time_taken_l134_13496

theorem total_time_taken 
  (R : ℝ) -- Rickey's speed
  (T_R : ℝ := 40) -- Rickey's time
  (T_P : ℝ := (40 * (4 / 3))) -- Prejean's time derived from given conditions
  (P : ℝ := (3 / 4) * R) -- Prejean's speed
  (k : ℝ := 40 * R) -- constant k for distance
 
  (h1 : T_R = 40)
  (h2 : T_P = 40 * (4 / 3))
  -- Main goal: Prove total time taken equals 93.33 minutes
  : (T_R + T_P) = 93.33 := 
  sorry

end total_time_taken_l134_13496


namespace total_money_shared_l134_13422

theorem total_money_shared (A B C : ℕ) (rA rB rC : ℕ) (bens_share : ℕ) 
  (h_ratio : rA = 2 ∧ rB = 3 ∧ rC = 8)
  (h_ben : B = bens_share)
  (h_bensShareGiven : bens_share = 60) : 
  (rA * (bens_share / rB)) + bens_share + (rC * (bens_share / rB)) = 260 :=
by
  -- sorry to skip the proof
  sorry

end total_money_shared_l134_13422


namespace unqualified_weight_l134_13414

theorem unqualified_weight (w : ℝ) (upper_limit lower_limit : ℝ) 
  (h1 : upper_limit = 10.1) 
  (h2 : lower_limit = 9.9) 
  (h3 : w = 9.09 ∨ w = 9.99 ∨ w = 10.01 ∨ w = 10.09) :
  ¬ (9.09 ≥ lower_limit ∧ 9.09 ≤ upper_limit) :=
by
  sorry

end unqualified_weight_l134_13414


namespace days_left_in_year_is_100_l134_13482

noncomputable def days_left_in_year 
    (daily_average_rain_before : ℝ) 
    (total_rainfall_so_far : ℝ) 
    (average_rain_needed : ℝ) 
    (total_days_in_year : ℕ) : ℕ :=
    sorry

theorem days_left_in_year_is_100 :
    days_left_in_year 2 430 3 365 = 100 := 
sorry

end days_left_in_year_is_100_l134_13482


namespace sum_of_variables_l134_13410

theorem sum_of_variables (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) (ha : a = 2 * Real.sqrt 6) (hb : b = 3 * Real.sqrt 6) (hc : c = 6 * Real.sqrt 6) : 
  a + b + c = 11 * Real.sqrt 6 :=
by
  sorry

end sum_of_variables_l134_13410


namespace find_fraction_value_l134_13484

noncomputable section

open Real

theorem find_fraction_value (α : ℝ) (h : sin (α / 2) - 2 * cos (α / 2) = 1) :
  (1 + sin α + cos α) / (1 + sin α - cos α) = 1 :=
sorry

end find_fraction_value_l134_13484


namespace rational_powers_implies_rational_a_rational_powers_implies_rational_b_l134_13483

open Real

theorem rational_powers_implies_rational_a (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^7 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

theorem rational_powers_implies_rational_b (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^9 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

end rational_powers_implies_rational_a_rational_powers_implies_rational_b_l134_13483


namespace scientific_notation_219400_l134_13418

def scientific_notation (n : ℝ) (m : ℝ) : Prop := n = m * 10^5

theorem scientific_notation_219400 : scientific_notation 219400 2.194 := 
by
  sorry

end scientific_notation_219400_l134_13418


namespace f_neg_a_l134_13439

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 2

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 2 := by
  sorry

end f_neg_a_l134_13439


namespace width_of_canal_at_bottom_l134_13473

theorem width_of_canal_at_bottom (h : Real) (b : Real) : 
  (A = 1/2 * (top_width + b) * d) ∧ 
  (A = 840) ∧ 
  (top_width = 12) ∧ 
  (d = 84) 
  → b = 8 := 
by
  intros
  sorry

end width_of_canal_at_bottom_l134_13473


namespace find_number_type_l134_13455

-- Definitions of the problem conditions
def consecutive (a b c d : ℤ) : Prop := (b = a + 2) ∧ (c = a + 4) ∧ (d = a + 6)
def sum_is_52 (a b c d : ℤ) : Prop := a + b + c + d = 52
def third_number_is_14 (c : ℤ) : Prop := c = 14

-- The proof problem statement
theorem find_number_type (a b c d : ℤ) 
                         (h1 : consecutive a b c d) 
                         (h2 : sum_is_52 a b c d) 
                         (h3 : third_number_is_14 c) :
  (∃ (k : ℤ), a = 2 * k ∧ b = 2 * k + 2 ∧ c = 2 * k + 4 ∧ d = 2 * k + 6) 
  := sorry

end find_number_type_l134_13455


namespace brokerage_percentage_l134_13435

theorem brokerage_percentage
  (cash_realized : ℝ)
  (cash_before_brokerage : ℝ)
  (h₁ : cash_realized = 109.25)
  (h₂ : cash_before_brokerage = 109) :
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100 = 0.23 := 
by
  sorry

end brokerage_percentage_l134_13435


namespace men_absent_is_5_l134_13413

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5_l134_13413


namespace seating_arrangement_l134_13403

theorem seating_arrangement :
  let total_arrangements := Nat.factorial 8
  let adjacent_arrangements := Nat.factorial 7 * 2
  total_arrangements - adjacent_arrangements = 30240 :=
by
  sorry

end seating_arrangement_l134_13403


namespace nontrivial_solution_fraction_l134_13458

theorem nontrivial_solution_fraction (x y z : ℚ)
  (h₁ : x - 6 * y + 3 * z = 0)
  (h₂ : 3 * x - 6 * y - 2 * z = 0)
  (h₃ : x + 6 * y - 5 * z = 0)
  (hne : x ≠ 0) :
  (y * z) / (x^2) = 2 / 3 :=
by
  sorry

end nontrivial_solution_fraction_l134_13458


namespace star_example_l134_13495

def star (a b : ℤ) : ℤ := a * b^3 - 2 * b + 2

theorem star_example : star 2 3 = 50 := by
  sorry

end star_example_l134_13495


namespace new_students_correct_l134_13456

variable 
  (students_start_year : Nat)
  (students_left : Nat)
  (students_end_year : Nat)

def new_students (students_start_year students_left students_end_year : Nat) : Nat :=
  students_end_year - (students_start_year - students_left)

theorem new_students_correct :
  ∀ (students_start_year students_left students_end_year : Nat),
  students_start_year = 10 →
  students_left = 4 →
  students_end_year = 48 →
  new_students students_start_year students_left students_end_year = 42 :=
by
  intros students_start_year students_left students_end_year h1 h2 h3
  rw [h1, h2, h3]
  unfold new_students
  norm_num

end new_students_correct_l134_13456


namespace sum_of_all_possible_values_l134_13408

theorem sum_of_all_possible_values (x y : ℝ) (h : x * y - x^2 - y^2 = 4) :
  (x - 2) * (y - 2) = 4 :=
sorry

end sum_of_all_possible_values_l134_13408


namespace lunks_needed_for_20_apples_l134_13492

-- Define the conditions as given in the problem
def lunks_to_kunks (lunks : ℤ) : ℤ := (4 * lunks) / 7
def kunks_to_apples (kunks : ℤ) : ℤ := (5 * kunks) / 3

-- Define the target function to calculate the number of lunks needed for given apples
def apples_to_lunks (apples : ℤ) : ℤ := 
  let kunks := (3 * apples) / 5
  let lunks := (7 * kunks) / 4
  lunks

-- Prove the given problem
theorem lunks_needed_for_20_apples : apples_to_lunks 20 = 21 := by
  sorry

end lunks_needed_for_20_apples_l134_13492


namespace dhoni_initial_toys_l134_13474

theorem dhoni_initial_toys (x : ℕ) (T : ℕ) 
    (h1 : T = 10 * x) 
    (h2 : T + 16 = 66) : x = 5 := by
  sorry

end dhoni_initial_toys_l134_13474


namespace number_of_sweaters_l134_13433

theorem number_of_sweaters 
(total_price_shirts : ℝ)
(total_shirts : ℕ)
(total_price_sweaters : ℝ)
(price_difference : ℝ) :
total_price_shirts = 400 ∧ total_shirts = 25 ∧ total_price_sweaters = 1500 ∧ price_difference = 4 →
(total_price_sweaters / ((total_price_shirts / total_shirts) + price_difference) = 75) :=
by
  intros
  sorry

end number_of_sweaters_l134_13433


namespace find_mother_age_l134_13404

-- Definitions for the given conditions
def serena_age_now := 9
def years_in_future := 6
def serena_age_future := serena_age_now + years_in_future
def mother_age_future (M : ℕ) := 3 * serena_age_future

-- The main statement to prove
theorem find_mother_age (M : ℕ) (h1 : M = mother_age_future M - years_in_future) : M = 39 :=
by
  sorry

end find_mother_age_l134_13404


namespace equal_charge_at_250_l134_13416

/-- Define the monthly fee for Plan A --/
def planA_fee (x : ℕ) : ℝ :=
  0.4 * x + 50

/-- Define the monthly fee for Plan B --/
def planB_fee (x : ℕ) : ℝ :=
  0.6 * x

/-- Prove that the charges for Plan A and Plan B are equal when the call duration is 250 minutes --/
theorem equal_charge_at_250 : planA_fee 250 = planB_fee 250 :=
by
  sorry

end equal_charge_at_250_l134_13416


namespace gus_buys_2_dozen_l134_13407

-- Definitions from conditions
def dozens_to_golf_balls (d : ℕ) : ℕ := d * 12
def total_golf_balls : ℕ := 132
def golf_balls_per_dozen : ℕ := 12
def dan_buys : ℕ := 5
def chris_buys_golf_balls : ℕ := 48

-- The number of dozens Gus buys
noncomputable def gus_buys (total_dozens dan_dozens chris_dozens : ℕ) : ℕ := total_dozens - dan_dozens - chris_dozens

theorem gus_buys_2_dozen : gus_buys (total_golf_balls / golf_balls_per_dozen) dan_buys (chris_buys_golf_balls / golf_balls_per_dozen) = 2 := by
  sorry

end gus_buys_2_dozen_l134_13407


namespace rotation_result_l134_13451

def initial_vector : ℝ × ℝ × ℝ := (3, -1, 1)

def rotate_180_z (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match v with
  | (x, y, z) => (-x, -y, z)

theorem rotation_result :
  rotate_180_z initial_vector = (-3, 1, 1) :=
by
  sorry

end rotation_result_l134_13451


namespace driver_speed_ratio_l134_13488

theorem driver_speed_ratio (V1 V2 x : ℝ) (h : V1 > 0 ∧ V2 > 0 ∧ x > 0)
  (meet_halfway : ∀ t1 t2, t1 = x / (2 * V1) ∧ t2 = x / (2 * V2))
  (earlier_start : ∀ t1 t2, t1 = t2 + x / (2 * (V1 + V2))) :
  V2 / V1 = (1 + Real.sqrt 5) / 2 := by
  sorry

end driver_speed_ratio_l134_13488


namespace inequality_solution_set_l134_13426

theorem inequality_solution_set :
  { x : ℝ | -3 < x ∧ x < 2 } = { x : ℝ | abs (x - 1) + abs (x + 2) < 5 } :=
by
  sorry

end inequality_solution_set_l134_13426


namespace range_of_function_l134_13442

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_function : Set.Icc (-2 : ℝ) 7 = Set.image f (Set.Icc (-3 : ℝ) 2) :=
by
  sorry

end range_of_function_l134_13442


namespace nonagon_arithmetic_mean_property_l134_13423

def is_equilateral_triangle (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  (j = (i + 3) % 9) ∧ (k = (i + 6) % 9)

def is_arithmetic_mean (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  A j = (A i + A k) / 2

theorem nonagon_arithmetic_mean_property :
  ∀ (A : Fin 9 → ℤ),
    (∀ i, A i = 2016 + i) →
    (∀ i j k : Fin 9, is_equilateral_triangle A i j k → is_arithmetic_mean A i j k) :=
by
  intros
  sorry

end nonagon_arithmetic_mean_property_l134_13423


namespace quadratic_factorization_l134_13447

theorem quadratic_factorization (C D : ℤ) (h : (15 * y^2 - 74 * y + 48) = (C * y - 16) * (D * y - 3)) :
  C * D + C = 20 :=
sorry

end quadratic_factorization_l134_13447


namespace x_cubed_inverse_cubed_l134_13406

theorem x_cubed_inverse_cubed (x : ℝ) (hx : x + 1/x = 3) : x^3 + 1/x^3 = 18 :=
by
  sorry

end x_cubed_inverse_cubed_l134_13406


namespace root_exponent_equiv_l134_13463

theorem root_exponent_equiv :
  (7 ^ (1 / 2)) / (7 ^ (1 / 4)) = 7 ^ (1 / 4) := by
  sorry

end root_exponent_equiv_l134_13463


namespace eventB_is_not_random_l134_13448

def eventA := "The sun rises in the east and it rains in the west"
def eventB := "It's not cold when it snows but cold when it melts"
def eventC := "It rains continuously during the Qingming festival"
def eventD := "It's sunny every day when the plums turn yellow"

def is_random_event (event : String) : Prop :=
  event = eventA ∨ event = eventC ∨ event = eventD

theorem eventB_is_not_random : ¬ is_random_event eventB :=
by
  unfold is_random_event
  sorry

end eventB_is_not_random_l134_13448


namespace determine_a_l134_13429

theorem determine_a (a : ℕ) (p1 p2 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : 2 * p1 * p2 = a) (h4 : p1 + p2 = 15) : 
  a = 52 :=
by
  sorry

end determine_a_l134_13429


namespace parking_lot_vehicle_spaces_l134_13432

theorem parking_lot_vehicle_spaces
  (total_spaces : ℕ)
  (spaces_per_caravan : ℕ)
  (num_caravans : ℕ)
  (remaining_spaces : ℕ) :
  total_spaces = 30 →
  spaces_per_caravan = 2 →
  num_caravans = 3 →
  remaining_spaces = total_spaces - (spaces_per_caravan * num_caravans) →
  remaining_spaces = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end parking_lot_vehicle_spaces_l134_13432


namespace simplify_fraction_l134_13427

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l134_13427


namespace min_f_value_l134_13466

open Real

theorem min_f_value (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
    ∃ (x : ℝ), (∀ y : ℝ, (|y - a| + |y - b| + |y - c| + |y - d| + |y - e|) ≥ -a - b + d + e) ∧ 
    (|x - a| + |x - b| + |x - c| + |x - d| + |x - e| = -a - b + d + e) :=
sorry

end min_f_value_l134_13466


namespace roster_method_A_l134_13420

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem roster_method_A :
  A = {1, 2} :=
by
  sorry

end roster_method_A_l134_13420


namespace k_value_if_divisible_l134_13424

theorem k_value_if_divisible :
  ∀ k : ℤ, (x^2 + k * x - 3) % (x - 1) = 0 → k = 2 :=
by
  intro k
  sorry

end k_value_if_divisible_l134_13424


namespace largest_partner_share_l134_13469

def total_profit : ℕ := 48000
def partner_ratios : List ℕ := [3, 4, 4, 6, 7]
def value_per_part : ℕ := total_profit / partner_ratios.sum
def largest_share : ℕ := 7 * value_per_part

theorem largest_partner_share :
  largest_share = 14000 := by
  sorry

end largest_partner_share_l134_13469


namespace passenger_capacity_passenger_capacity_at_5_max_profit_l134_13419

section SubwayProject

-- Define the time interval t and the passenger capacity function p(t)
def p (t : ℕ) : ℕ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit function Q(t)
def Q (t : ℕ) : ℚ :=
  if 2 ≤ t ∧ t < 10 then (8 * p t - 2656) / t - 60
  else if 10 ≤ t ∧ t ≤ 20 then (1344 : ℚ) / t - 60
  else 0

-- Statement 1: Prove the correct expression for p(t) and its value at t = 5
theorem passenger_capacity (t : ℕ) (ht1 : 2 ≤ t) (ht2 : t ≤ 20) :
  (p t = if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2 else 500) :=
sorry

theorem passenger_capacity_at_5 : p 5 = 450 :=
sorry

-- Statement 2: Prove the time interval t and the maximum value of Q(t)
theorem max_profit : ∃ t : ℕ, 2 ≤ t ∧ t ≤ 10 ∧ Q t = 132 ∧ (∀ u : ℕ, 2 ≤ u ∧ u ≤ 10 → Q u ≤ Q t) :=
sorry

end SubwayProject

end passenger_capacity_passenger_capacity_at_5_max_profit_l134_13419


namespace correct_calculation_l134_13428

theorem correct_calculation (a : ℝ) :
  2 * a^4 * 3 * a^5 = 6 * a^9 :=
by
  sorry

end correct_calculation_l134_13428


namespace shaded_area_T_shape_l134_13441

theorem shaded_area_T_shape (a b c d e: ℕ) (square_side_length rect_length rect_width: ℕ)
  (h_side_lengths: ∀ x, x = 2 ∨ x = 4) (h_square: square_side_length = 6) 
  (h_rect_dim: rect_length = 4 ∧ rect_width = 2)
  (h_areas: [a, b, c, d, e] = [4, 4, 4, 8, 4]) :
  a + b + d + e = 20 :=
by
  sorry

end shaded_area_T_shape_l134_13441


namespace geometric_sequence_terms_sum_l134_13494

theorem geometric_sequence_terms_sum :
  ∀ (a_n : ℕ → ℝ) (q : ℝ),
    (∀ n, a_n (n + 1) = a_n n * q) ∧ a_n 1 = 3 ∧
    (a_n 1 + a_n 2 + a_n 3) = 21 →
    (a_n (1 + 2) + a_n (1 + 3) + a_n (1 + 4)) = 84 :=
by
  intros a_n q h
  sorry

end geometric_sequence_terms_sum_l134_13494


namespace smallest_base_10_integer_l134_13459

-- Given conditions
def is_valid_base (a b : ℕ) : Prop := a > 2 ∧ b > 2

def base_10_equivalence (a b n : ℕ) : Prop := (2 * a + 1 = n) ∧ (b + 2 = n)

-- The smallest base-10 integer represented as 21_a and 12_b
theorem smallest_base_10_integer :
  ∃ (a b n : ℕ), is_valid_base a b ∧ base_10_equivalence a b n ∧ n = 7 :=
by
  sorry

end smallest_base_10_integer_l134_13459


namespace quadratic_nonneg_iff_l134_13425

variable {a b c : ℝ}

theorem quadratic_nonneg_iff :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4 * a * c ≤ 0) :=
by sorry

end quadratic_nonneg_iff_l134_13425


namespace brother_growth_is_one_l134_13444

-- Define measurements related to Stacy's height.
def Stacy_previous_height : ℕ := 50
def Stacy_current_height : ℕ := 57

-- Define the condition that Stacy's growth is 6 inches more than her brother's growth.
def Stacy_growth := Stacy_current_height - Stacy_previous_height
def Brother_growth := Stacy_growth - 6

-- Prove that Stacy's brother grew 1 inch.
theorem brother_growth_is_one : Brother_growth = 1 :=
by
  sorry

end brother_growth_is_one_l134_13444


namespace initial_tomatoes_l134_13499

theorem initial_tomatoes (T : ℕ) (picked : ℕ) (remaining_total : ℕ) (potatoes : ℕ) :
  potatoes = 12 →
  picked = 53 →
  remaining_total = 136 →
  T + picked = remaining_total - potatoes →
  T = 71 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_tomatoes_l134_13499


namespace bike_cost_l134_13461

theorem bike_cost (price_per_apple repairs_share remaining_share apples_sold earnings repairs_cost bike_cost : ℝ) :
  price_per_apple = 1.25 →
  repairs_share = 0.25 →
  remaining_share = 1/5 →
  apples_sold = 20 →
  earnings = apples_sold * price_per_apple →
  repairs_cost = earnings * 4/5 →
  repairs_cost = bike_cost * repairs_share →
  bike_cost = 80 :=
by
  intros;
  sorry

end bike_cost_l134_13461


namespace find_certain_number_l134_13449

theorem find_certain_number (x certain_number : ℕ) (h1 : certain_number + x = 13200) (h2 : x = 3327) : certain_number = 9873 :=
by
  sorry

end find_certain_number_l134_13449


namespace infinite_series_sum_eq_l134_13489

theorem infinite_series_sum_eq : 
  (∑' n : ℕ, if n = 0 then 0 else ((1 : ℝ) / (n * (n + 3)))) = (11 / 18 : ℝ) :=
sorry

end infinite_series_sum_eq_l134_13489


namespace total_flowers_l134_13465

noncomputable def yellow_flowers : ℕ := 10
noncomputable def purple_flowers : ℕ := yellow_flowers + (80 * yellow_flowers) / 100
noncomputable def green_flowers : ℕ := (25 * (yellow_flowers + purple_flowers)) / 100
noncomputable def red_flowers : ℕ := (35 * (yellow_flowers + purple_flowers + green_flowers)) / 100

theorem total_flowers :
  yellow_flowers + purple_flowers + green_flowers + red_flowers = 47 :=
by
  -- Insert proof here
  sorry

end total_flowers_l134_13465


namespace largest_square_side_length_l134_13402

theorem largest_square_side_length (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : 
  ∃ x : ℝ, x = (a * b) / (a + b) := 
sorry

end largest_square_side_length_l134_13402


namespace range_of_m_l134_13464

def A (x : ℝ) : Prop := x^2 - x - 6 > 0
def B (x m : ℝ) : Prop := (x - m) * (x - 2 * m) ≤ 0
def is_disjoint (A B : ℝ → Prop) : Prop := ∀ x, ¬ (A x ∧ B x)

theorem range_of_m (m : ℝ) : 
  is_disjoint (A) (B m) ↔ -1 ≤ m ∧ m ≤ 3 / 2 := by
  sorry

end range_of_m_l134_13464


namespace probability_two_dice_sum_seven_l134_13415

theorem probability_two_dice_sum_seven (z : ℕ) (w : ℚ) (h : z = 2) : w = 1 / 6 :=
by sorry

end probability_two_dice_sum_seven_l134_13415


namespace haley_initial_trees_l134_13405

theorem haley_initial_trees (dead_trees trees_left initial_trees : ℕ) 
    (h_dead: dead_trees = 2)
    (h_left: trees_left = 10)
    (h_initial: initial_trees = trees_left + dead_trees) : 
    initial_trees = 12 := 
by sorry

end haley_initial_trees_l134_13405


namespace right_triangle_example_find_inverse_450_mod_3599_l134_13491

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a b m : ℕ) : Prop :=
  (a * b) % m = 1

theorem right_triangle_example : is_right_triangle 60 221 229 :=
by
  sorry

theorem find_inverse_450_mod_3599 : ∃ n, 0 ≤ n ∧ n < 3599 ∧ multiplicative_inverse 450 n 3599 :=
by
  use 8
  sorry

end right_triangle_example_find_inverse_450_mod_3599_l134_13491


namespace janet_used_clips_correct_l134_13493

-- Define the initial number of paper clips
def initial_clips : ℕ := 85

-- Define the remaining number of paper clips
def remaining_clips : ℕ := 26

-- Define the number of clips Janet used
def used_clips (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

-- The theorem to state the correctness of the calculation
theorem janet_used_clips_correct : used_clips initial_clips remaining_clips = 59 :=
by
  -- Lean proof goes here
  sorry

end janet_used_clips_correct_l134_13493


namespace loot_box_cost_l134_13476

variable (C : ℝ) -- Declare cost of each loot box as a real number

-- Conditions (average value of items, money spent, loss)
def avg_value : ℝ := 3.5
def money_spent : ℝ := 40
def avg_loss : ℝ := 12

-- Derived equation
def equation := avg_value * (money_spent / C) = money_spent - avg_loss

-- Statement to prove
theorem loot_box_cost : equation C → C = 5 := by
  sorry

end loot_box_cost_l134_13476


namespace polynomial_evaluation_l134_13490

noncomputable def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_evaluation : f 2 = 123 := by
  sorry

end polynomial_evaluation_l134_13490


namespace sum_of_roots_of_quadratic_l134_13470

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l134_13470


namespace wenlock_olympian_games_first_held_year_difference_l134_13436

theorem wenlock_olympian_games_first_held_year_difference :
  2012 - 1850 = 162 :=
sorry

end wenlock_olympian_games_first_held_year_difference_l134_13436


namespace cos_double_angle_l134_13421

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by 
  sorry

end cos_double_angle_l134_13421


namespace middle_card_is_five_l134_13477

section card_numbers

variables {a b c : ℕ}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def sum_fifteen (a b c : ℕ) : Prop := a + b + c = 15
def sum_two_smallest_less_than_ten (a b : ℕ) : Prop := a + b < 10
def ascending_order (a b c : ℕ) : Prop := a < b ∧ b < c 

-- Main theorem statement
theorem middle_card_is_five 
  (h1 : distinct a b c)
  (h2 : sum_fifteen a b c)
  (h3 : sum_two_smallest_less_than_ten a b) 
  (h4 : ascending_order a b c)
  (h5 : ∀ x, (x = a → (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten x b ∧ ascending_order x b c ∧ ¬ (b = 5 ∧ c = 10))) →
           (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten b c ∧ ascending_order x b c ∧ ¬ (b = 2 ∧ c = 7)))
  (h6 : ∀ x, (x = c → (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 1 ∧ b = 4))) →
           (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 2 ∧ b = 6)))
  (h7 : ∀ x, (x = b → (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 9 ∨ a = 2 ∧ c = 8))) →
           (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 6 ∨ a = 2 ∧ c = 5)))
  : b = 5 := sorry

end card_numbers

end middle_card_is_five_l134_13477


namespace contrapositive_proposition_l134_13431

theorem contrapositive_proposition {a b : ℝ} :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) → (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_proposition_l134_13431


namespace appropriate_word_count_l134_13400

-- Define the conditions of the problem
def min_minutes := 40
def max_minutes := 55
def words_per_minute := 120

-- Define the bounds for the number of words
def min_words := min_minutes * words_per_minute
def max_words := max_minutes * words_per_minute

-- Define the appropriate number of words
def appropriate_words (words : ℕ) : Prop :=
  words >= min_words ∧ words <= max_words

-- The specific numbers to test
def words1 := 5000
def words2 := 6200

-- The main proof statement
theorem appropriate_word_count : 
  appropriate_words words1 ∧ appropriate_words words2 :=
by
  -- We do not need to provide the proof steps, just state the theorem
  sorry

end appropriate_word_count_l134_13400


namespace spadesuit_value_l134_13454

def spadesuit (a b : ℤ) : ℤ :=
  |a^2 - b^2|

theorem spadesuit_value :
  spadesuit 3 (spadesuit 5 2) = 432 :=
by
  sorry

end spadesuit_value_l134_13454


namespace crossing_time_l134_13446

-- Define the conditions
def walking_speed_kmh : Float := 10
def bridge_length_m : Float := 1666.6666666666665

-- Convert the man's walking speed to meters per minute
def walking_speed_mpm : Float := walking_speed_kmh * (1000 / 60)

-- State the theorem we want to prove
theorem crossing_time 
  (ws_kmh : Float := walking_speed_kmh)
  (bl_m : Float := bridge_length_m)
  (ws_mpm : Float := walking_speed_mpm) :
  bl_m / ws_mpm = 10 :=
by
  sorry

end crossing_time_l134_13446


namespace solve_for_x_l134_13472

theorem solve_for_x (x : ℚ) : ((1/3 - x) ^ 2 = 4) → (x = -5/3 ∨ x = 7/3) :=
by
  sorry

end solve_for_x_l134_13472


namespace order_of_even_function_l134_13485

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem order_of_even_function {f : ℝ → ℝ}
  (h_even : is_even f)
  (h_mono_inc : is_monotonically_increasing_on_nonneg f) :
  f (-π) > f (3) ∧ f (3) > f (-2) :=
sorry

end order_of_even_function_l134_13485


namespace max_mark_is_600_l134_13411

-- Define the conditions
def forty_percent (M : ℝ) : ℝ := 0.40 * M
def student_score : ℝ := 175
def additional_marks_needed : ℝ := 65

-- The goal is to prove that the maximum mark is 600
theorem max_mark_is_600 (M : ℝ) :
  forty_percent M = student_score + additional_marks_needed → M = 600 := 
by 
  sorry

end max_mark_is_600_l134_13411


namespace solve_inequality_system_l134_13417

theorem solve_inequality_system (x : ℝ) (h1 : 2 * x + 1 < 5) (h2 : 2 - x ≤ 1) : 1 ≤ x ∧ x < 2 :=
by
  sorry

end solve_inequality_system_l134_13417


namespace positive_integer_pairs_count_l134_13480

theorem positive_integer_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a > 0 ∧ b > 0 ∧ (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 2021) ∧ 
    pairs.length = 4 :=
by sorry

end positive_integer_pairs_count_l134_13480


namespace polynomial_sum_eq_l134_13437

-- Definitions of the given polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- The theorem to prove
theorem polynomial_sum_eq (x : ℝ) : 
  p x + q x + r x + s x = -x^2 + 10 * x - 11 :=
by 
  -- Proof steps are omitted here
  sorry

end polynomial_sum_eq_l134_13437


namespace children_left_on_bus_l134_13479

-- Definitions based on the conditions
def initial_children := 43
def children_got_off := 22

-- The theorem we want to prove
theorem children_left_on_bus (initial_children children_got_off : ℕ) : 
  initial_children - children_got_off = 21 :=
by
  sorry

end children_left_on_bus_l134_13479


namespace max_ratio_of_sequence_l134_13440

theorem max_ratio_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, S n = (n + 2) / 3 * a n) :
  ∃ n : ℕ, ∀ m : ℕ, (n = 2 → m ≠ 1) → (a n / a (n - 1)) ≤ (a m / a (m - 1)) :=
by
  sorry

end max_ratio_of_sequence_l134_13440


namespace not_possible_d_count_l134_13401

open Real

theorem not_possible_d_count (t s d : ℝ) (h1 : 3 * t - 4 * s = 1989) (h2 : t - s = d) (h3 : 4 * s > 0) :
  ∃ k : ℕ, k = 663 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ k → d ≠ n :=
by
  sorry

end not_possible_d_count_l134_13401


namespace savings_percentage_first_year_l134_13481

noncomputable def savings_percentage (I S : ℝ) : ℝ := (S / I) * 100

theorem savings_percentage_first_year (I S : ℝ) (h1 : S = 0.20 * I) :
  savings_percentage I S = 20 :=
by
  unfold savings_percentage
  rw [h1]
  field_simp
  norm_num
  sorry

end savings_percentage_first_year_l134_13481


namespace functional_equation_true_l134_13467

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : f x > 0
axiom f_property (a b : ℝ) : f a * f b = f (a + b)

theorem functional_equation_true :
  (f 0 = 1) ∧ 
  (∀ a, f (-a) = 1 / f a) ∧ 
  (∀ a, f a = (f (4 * a)) ^ (1 / 4)) ∧ 
  (∀ a, f (a^2) = (f a)^2) :=
by {
  sorry
}

end functional_equation_true_l134_13467


namespace ImpossibleNonConformists_l134_13468

open Int

def BadPairCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (pairs : Finset (ℤ × ℤ)), 
    pairs.card ≤ ⌊0.001 * (n.natAbs^2 : ℝ)⌋₊ ∧ 
    ∀ (x y : ℤ), (x, y) ∈ pairs → max (abs x) (abs y) ≤ n ∧ f (x + y) ≠ f x + f y

def NonConformistCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (conformists : Finset ℤ), 
    conformists.card > n ∧ 
    ∀ (a : ℤ), abs a ≤ n → (f a ≠ a * f 1 → a ∈ conformists)

theorem ImpossibleNonConformists (f : ℤ → ℤ) :
  (∀ (n : ℤ), n ≥ 0 → BadPairCondition f n) → 
  ¬ ∃ (n : ℤ), n ≥ 0 ∧ NonConformistCondition f n :=
  by 
    intros h_cond h_ex
    sorry

end ImpossibleNonConformists_l134_13468
