import Mathlib

namespace NUMINAMATH_GPT_find_original_number_of_men_l2274_227408

variable (M : ℕ) (W : ℕ)

-- Given conditions translated to Lean
def condition1 := M * 10 = W -- M men complete work W in 10 days
def condition2 := (M - 10) * 20 = W -- (M - 10) men complete work W in 20 days

theorem find_original_number_of_men (h1 : condition1 M W) (h2 : condition2 M W) : M = 20 :=
sorry

end NUMINAMATH_GPT_find_original_number_of_men_l2274_227408


namespace NUMINAMATH_GPT_calculate_f_at_2_l2274_227420

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem calculate_f_at_2
  (a b : ℝ)
  (h_extremum : 3 + 2 * a + b = 0)
  (h_f1 : f 1 a b = 10) :
  f 2 a b = 18 :=
sorry

end NUMINAMATH_GPT_calculate_f_at_2_l2274_227420


namespace NUMINAMATH_GPT_rotated_square_vertical_distance_is_correct_l2274_227417

-- Define a setup with four 1-inch squares in a straight line
-- and the second square rotated 45 degrees around its center

-- Noncomputable setup
noncomputable def rotated_square_vert_distance : ℝ :=
  let side_length := 1
  let diagonal := side_length * Real.sqrt 2
  -- Calculate the required vertical distance according to given conditions
  Real.sqrt 2 + side_length / 2

-- Theorem statement confirming the calculated vertical distance
theorem rotated_square_vertical_distance_is_correct :
  rotated_square_vert_distance = Real.sqrt 2 + 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rotated_square_vertical_distance_is_correct_l2274_227417


namespace NUMINAMATH_GPT_vanya_four_times_faster_l2274_227495

-- We let d be the total distance, and define the respective speeds
variables (d : ℝ) (v_m v_v : ℝ)

-- Conditions from the problem
-- 1. Vanya starts after Masha
axiom start_after_masha : ∀ t : ℝ, t > 0

-- 2. Vanya overtakes Masha at one-third of the distance
axiom vanya_overtakes_masha : ∀ t : ℝ, (v_v * t) = d / 3

-- 3. When Vanya reaches the school, Masha still has half of the way to go
axiom masha_halfway : ∀ t : ℝ, (v_m * t) = d / 2

-- Goal to prove
theorem vanya_four_times_faster : v_v = 4 * v_m :=
sorry

end NUMINAMATH_GPT_vanya_four_times_faster_l2274_227495


namespace NUMINAMATH_GPT_Malou_first_quiz_score_l2274_227473

variable (score1 score2 score3 : ℝ)

theorem Malou_first_quiz_score (h1 : score1 = 90) (h2 : score2 = 92) (h_avg : (score1 + score2 + score3) / 3 = 91) : score3 = 91 := by
  sorry

end NUMINAMATH_GPT_Malou_first_quiz_score_l2274_227473


namespace NUMINAMATH_GPT_find_b_l2274_227429

noncomputable def complex_b_value (i : ℂ) (b : ℝ) : Prop :=
(1 + b * i) * i = 1 + i

theorem find_b (i : ℂ) (b : ℝ) (hi : i^2 = -1) (h : complex_b_value i b) : b = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_l2274_227429


namespace NUMINAMATH_GPT_mirror_area_correct_l2274_227475

noncomputable def width_of_mirror (frame_width : ℕ) (side_width : ℕ) : ℕ :=
  frame_width - 2 * side_width

noncomputable def height_of_mirror (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  frame_height - 2 * side_width

noncomputable def area_of_mirror (frame_width : ℕ) (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  width_of_mirror frame_width side_width * height_of_mirror frame_height side_width

theorem mirror_area_correct :
  area_of_mirror 50 70 7 = 2016 :=
by
  sorry

end NUMINAMATH_GPT_mirror_area_correct_l2274_227475


namespace NUMINAMATH_GPT_students_in_cars_l2274_227405

theorem students_in_cars (total_students : ℕ := 396) (buses : ℕ := 7) (students_per_bus : ℕ := 56) :
  total_students - (buses * students_per_bus) = 4 := by
  sorry

end NUMINAMATH_GPT_students_in_cars_l2274_227405


namespace NUMINAMATH_GPT_vector_linear_combination_l2274_227498

open Matrix

theorem vector_linear_combination :
  let v1 := ![3, -9]
  let v2 := ![2, -8]
  let v3 := ![1, -6]
  4 • v1 - 3 • v2 + 2 • v3 = ![8, -24] :=
by sorry

end NUMINAMATH_GPT_vector_linear_combination_l2274_227498


namespace NUMINAMATH_GPT_certain_number_l2274_227423

theorem certain_number (x y z : ℕ) 
  (h1 : x + y = 15) 
  (h2 : y = 7) 
  (h3 : 3 * x = z * y - 11) : 
  z = 5 :=
by sorry

end NUMINAMATH_GPT_certain_number_l2274_227423


namespace NUMINAMATH_GPT_lever_equilibrium_min_force_l2274_227468

noncomputable def lever_minimum_force (F L : ℝ) : Prop :=
  (F * L = 49 + 2 * (L^2))

theorem lever_equilibrium_min_force : ∃ F : ℝ, ∃ L : ℝ, L = 7 → lever_minimum_force F L :=
by
  sorry

end NUMINAMATH_GPT_lever_equilibrium_min_force_l2274_227468


namespace NUMINAMATH_GPT_current_short_trees_l2274_227478

theorem current_short_trees (S : ℕ) (S_planted : ℕ) (S_total : ℕ) 
  (H1 : S_planted = 105) 
  (H2 : S_total = 217) 
  (H3 : S + S_planted = S_total) :
  S = 112 :=
by
  sorry

end NUMINAMATH_GPT_current_short_trees_l2274_227478


namespace NUMINAMATH_GPT_total_intersections_l2274_227412

def north_south_streets : ℕ := 10
def east_west_streets : ℕ := 10

theorem total_intersections :
  (north_south_streets * east_west_streets = 100) :=
by
  sorry

end NUMINAMATH_GPT_total_intersections_l2274_227412


namespace NUMINAMATH_GPT_arith_seq_sum_first_four_terms_l2274_227450

noncomputable def sum_first_four_terms_arith_seq (a1 : ℤ) (d : ℤ) : ℤ :=
  4 * a1 + 6 * d

theorem arith_seq_sum_first_four_terms (a1 a3 : ℤ) 
  (h1 : a3 = a1 + 2 * 3)
  (h2 : a1 + a3 = 8) 
  (d : ℤ := 3) :
  sum_first_four_terms_arith_seq a1 d = 22 := by
  unfold sum_first_four_terms_arith_seq
  sorry

end NUMINAMATH_GPT_arith_seq_sum_first_four_terms_l2274_227450


namespace NUMINAMATH_GPT_Michael_made_97_dollars_l2274_227418

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end NUMINAMATH_GPT_Michael_made_97_dollars_l2274_227418


namespace NUMINAMATH_GPT_weather_condition_l2274_227476

theorem weather_condition (T : ℝ) (windy : Prop) (kites_will_fly : Prop) 
  (h1 : (T > 25 ∧ windy) → kites_will_fly) 
  (h2 : ¬ kites_will_fly) : T ≤ 25 ∨ ¬ windy :=
by 
  sorry

end NUMINAMATH_GPT_weather_condition_l2274_227476


namespace NUMINAMATH_GPT_impossible_arrangement_of_300_numbers_in_circle_l2274_227439

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end NUMINAMATH_GPT_impossible_arrangement_of_300_numbers_in_circle_l2274_227439


namespace NUMINAMATH_GPT_least_value_of_x_l2274_227421

theorem least_value_of_x (x p : ℕ) (h1 : (x / (11 * p)) = 3) (h2 : x > 0) (h3 : Nat.Prime p) : x = 66 := by
  sorry

end NUMINAMATH_GPT_least_value_of_x_l2274_227421


namespace NUMINAMATH_GPT_cats_not_eating_either_l2274_227409

/-- In a shelter with 80 cats, 15 cats like tuna, 60 cats like chicken, 
and 10 like both tuna and chicken, prove that 15 cats do not eat either. -/
theorem cats_not_eating_either (total_cats : ℕ) (like_tuna : ℕ) (like_chicken : ℕ) (like_both : ℕ)
    (h1 : total_cats = 80) (h2 : like_tuna = 15) (h3 : like_chicken = 60) (h4 : like_both = 10) :
    (total_cats - (like_tuna - like_both + like_chicken - like_both + like_both) = 15) := 
by
    sorry

end NUMINAMATH_GPT_cats_not_eating_either_l2274_227409


namespace NUMINAMATH_GPT_triangle_inequality_l2274_227493

noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def r (a b c : ℝ) : ℝ := 
  let p := p a b c
  let x := p - a
  let y := p - b
  let z := p - c
  Real.sqrt ((x * y * z) / (x + y + z))

noncomputable def x (a b c : ℝ) : ℝ := p a b c - a
noncomputable def y (a b c : ℝ) : ℝ := p a b c - b
noncomputable def z (a b c : ℝ) : ℝ := p a b c - c

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 / (x a b c)^2 + 1 / (y a b c)^2 + 1 / (z a b c)^2 ≥ (x a b c + y a b c + z a b c) / ((x a b c) * (y a b c) * (z a b c)) := by
    sorry

end NUMINAMATH_GPT_triangle_inequality_l2274_227493


namespace NUMINAMATH_GPT_largest_fraction_is_36_l2274_227489

theorem largest_fraction_is_36 : 
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  A < E ∧ B < E ∧ C < E ∧ D < E :=
by
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  sorry

end NUMINAMATH_GPT_largest_fraction_is_36_l2274_227489


namespace NUMINAMATH_GPT_computer_sale_price_percent_l2274_227479

theorem computer_sale_price_percent (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) :
  original_price = 500 ∧ discount1 = 0.25 ∧ discount2 = 0.10 ∧ discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price * 100 = 64.13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_computer_sale_price_percent_l2274_227479


namespace NUMINAMATH_GPT_fraction_of_foreign_males_l2274_227445

theorem fraction_of_foreign_males
  (total_students : ℕ)
  (female_ratio : ℚ)
  (non_foreign_males : ℕ)
  (foreign_male_fraction : ℚ)
  (h1 : total_students = 300)
  (h2 : female_ratio = 2/3)
  (h3 : non_foreign_males = 90) :
  foreign_male_fraction = 1/10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_foreign_males_l2274_227445


namespace NUMINAMATH_GPT_car_turns_proof_l2274_227451

def turns_opposite_direction (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180

theorem car_turns_proof
  (angle1 angle2 : ℝ)
  (h1 : (angle1 = 50 ∧ angle2 = 130) ∨ (angle1 = -50 ∧ angle2 = 130) ∨ 
       (angle1 = 50 ∧ angle2 = -130) ∨ (angle1 = 30 ∧ angle2 = -30)) :
  turns_opposite_direction angle1 angle2 ↔ (angle1 = 50 ∧ angle2 = 130) :=
by
  sorry

end NUMINAMATH_GPT_car_turns_proof_l2274_227451


namespace NUMINAMATH_GPT_laura_owes_amount_l2274_227465

noncomputable def calculate_amount_owed (P R T : ℝ) : ℝ :=
  let I := P * R * T 
  P + I

theorem laura_owes_amount (P : ℝ) (R : ℝ) (T : ℝ) (hP : P = 35) (hR : R = 0.09) (hT : T = 1) :
  calculate_amount_owed P R T = 38.15 := by
  -- Prove that the total amount owed calculated by the formula matches the correct answer
  sorry

end NUMINAMATH_GPT_laura_owes_amount_l2274_227465


namespace NUMINAMATH_GPT_system_solutions_l2274_227422

theorem system_solutions (a b : ℝ) :
  (∃ (x y : ℝ), x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := 
sorry

end NUMINAMATH_GPT_system_solutions_l2274_227422


namespace NUMINAMATH_GPT_range_of_y_under_conditions_l2274_227441

theorem range_of_y_under_conditions :
  (∀ x : ℝ, (x - y) * (x + y) < 1) → (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
  intro h
  have h' : ∀ x : ℝ, (x - y) * (1 - x - y) < 1 := by
    sorry
  have g_min : ∀ x : ℝ, y^2 - y < x^2 - x + 1 := by
    sorry
  have min_value : y^2 - y < 3/4 := by
    sorry
  have range_y : (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
    sorry
  exact range_y

end NUMINAMATH_GPT_range_of_y_under_conditions_l2274_227441


namespace NUMINAMATH_GPT_percentage_transform_l2274_227426

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end NUMINAMATH_GPT_percentage_transform_l2274_227426


namespace NUMINAMATH_GPT_profit_percentage_correct_l2274_227482

def SP : ℝ := 900
def P : ℝ := 100

theorem profit_percentage_correct : (P / (SP - P)) * 100 = 12.5 := sorry

end NUMINAMATH_GPT_profit_percentage_correct_l2274_227482


namespace NUMINAMATH_GPT_schools_participation_l2274_227416

-- Definition of the problem conditions
def school_teams : ℕ := 3

-- Paula's rank p must satisfy this
def total_participants (p : ℕ) : ℕ := 2 * p - 1

-- Predicate indicating the number of participants condition:
def participants_condition (p : ℕ) : Prop := total_participants p ≥ 75

-- Translation of number of participants to number of schools
def number_of_schools (n : ℕ) : ℕ := 3 * n

-- The statement to prove:
theorem schools_participation : ∃ (n p : ℕ), participants_condition p ∧ p = 38 ∧ number_of_schools n = total_participants p ∧ n = 25 := 
by 
  sorry

end NUMINAMATH_GPT_schools_participation_l2274_227416


namespace NUMINAMATH_GPT_find_k_l2274_227438

theorem find_k (k : ℕ) : (1 / 3)^32 * (1 / 125)^k = 1 / 27^32 → k = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_l2274_227438


namespace NUMINAMATH_GPT_find_prime_triplet_l2274_227497

def is_geometric_sequence (x y z : ℕ) : Prop :=
  (y^2 = x * z)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_triplet :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_sequence (a + 1) (b + 1) (c + 1) ∧
  (a = 17 ∧ b = 23 ∧ c = 31) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_triplet_l2274_227497


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l2274_227494

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ}, 
  (3 * x - 4 * y + 1 = 0) → (3 * x - 4 * y + 7 = 0) → 
  ∃ d, d = (6 : ℝ) / 5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l2274_227494


namespace NUMINAMATH_GPT_quadratic_decreasing_right_of_axis_of_symmetry_l2274_227446

theorem quadratic_decreasing_right_of_axis_of_symmetry :
  ∀ x : ℝ, -2 * (x - 1)^2 < -2 * (x + 1 - 1)^2 →
  (∀ x' : ℝ, x' > 1 → -2 * (x' - 1)^2 < -2 * (x + 1 - 1)^2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_decreasing_right_of_axis_of_symmetry_l2274_227446


namespace NUMINAMATH_GPT_original_number_of_people_l2274_227430

theorem original_number_of_people (x : ℕ) (h1 : 3 ∣ x) (h2 : 6 ∣ x) (h3 : (x / 2) = 18) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_people_l2274_227430


namespace NUMINAMATH_GPT_percent_first_shift_participating_l2274_227457

variable (total_employees_in_company : ℕ)
variable (first_shift_employees : ℕ)
variable (second_shift_employees : ℕ)
variable (third_shift_employees : ℕ)
variable (second_shift_percent_participating : ℚ)
variable (third_shift_percent_participating : ℚ)
variable (overall_percent_participating : ℚ)
variable (first_shift_percent_participating : ℚ)

theorem percent_first_shift_participating :
  total_employees_in_company = 150 →
  first_shift_employees = 60 →
  second_shift_employees = 50 →
  third_shift_employees = 40 →
  second_shift_percent_participating = 0.40 →
  third_shift_percent_participating = 0.10 →
  overall_percent_participating = 0.24 →
  first_shift_percent_participating = (12 / 60) →
  first_shift_percent_participating = 0.20 := 
by 
  intros t_e f_s_e s_s_e t_s_e s_s_p_p t_s_p_p o_p_p f_s_p_p
  -- Sorry, here would be the place for the actual proof
  sorry

end NUMINAMATH_GPT_percent_first_shift_participating_l2274_227457


namespace NUMINAMATH_GPT_depression_comparative_phrase_l2274_227462

def correct_comparative_phrase (phrase : String) : Prop :=
  phrase = "twice as…as"

theorem depression_comparative_phrase :
  correct_comparative_phrase "twice as…as" :=
by
  sorry

end NUMINAMATH_GPT_depression_comparative_phrase_l2274_227462


namespace NUMINAMATH_GPT_ratio_of_ducks_to_total_goats_and_chickens_l2274_227407

theorem ratio_of_ducks_to_total_goats_and_chickens 
    (goats chickens ducks pigs : ℕ) 
    (h1 : goats = 66)
    (h2 : chickens = 2 * goats)
    (h3 : pigs = ducks / 3)
    (h4 : goats = pigs + 33) :
    (ducks : ℚ) / (goats + chickens : ℚ) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_ducks_to_total_goats_and_chickens_l2274_227407


namespace NUMINAMATH_GPT_second_rice_price_l2274_227470

theorem second_rice_price (P : ℝ) 
  (price_first : ℝ := 3.10) 
  (price_mixture : ℝ := 3.25) 
  (ratio_first_to_second : ℝ := 3 / 7) :
  (3 * price_first + 7 * P) / 10 = price_mixture → 
  P = 3.3142857142857145 :=
by
  sorry

end NUMINAMATH_GPT_second_rice_price_l2274_227470


namespace NUMINAMATH_GPT_determine_n_l2274_227440

theorem determine_n (n : ℕ) (hn : 0 < n) :
  (∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 2 * y + z = n) → (n = 15 ∨ n = 16) :=
  by sorry

end NUMINAMATH_GPT_determine_n_l2274_227440


namespace NUMINAMATH_GPT_find_sum_of_squares_l2274_227419

theorem find_sum_of_squares (x y z : ℝ)
  (h1 : x^2 + 3 * y = 8)
  (h2 : y^2 + 5 * z = -9)
  (h3 : z^2 + 7 * x = -16) : x^2 + y^2 + z^2 = 20.75 :=
sorry

end NUMINAMATH_GPT_find_sum_of_squares_l2274_227419


namespace NUMINAMATH_GPT_Isabella_hair_length_l2274_227410

-- Define the conditions using variables
variables (h_current h_cut_off h_initial : ℕ)

-- The proof problem statement
theorem Isabella_hair_length :
  h_current = 9 → h_cut_off = 9 → h_initial = h_current + h_cut_off → h_initial = 18 :=
by
  intros hc hc' hi
  rw [hc, hc'] at hi
  exact hi


end NUMINAMATH_GPT_Isabella_hair_length_l2274_227410


namespace NUMINAMATH_GPT_number_of_integers_in_sequence_l2274_227486

theorem number_of_integers_in_sequence 
  (a_0 : ℕ) 
  (h_0 : a_0 = 8820) 
  (seq : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, seq (n + 1) = seq n / 3) :
  ∃ n : ℕ, seq n = 980 ∧ n + 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_in_sequence_l2274_227486


namespace NUMINAMATH_GPT_stella_annual_income_l2274_227456

-- Define the conditions
def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def total_months : ℕ := 12

-- The question: What is Stella's annual income last year?
def annual_income (monthly_income : ℕ) (worked_months : ℕ) : ℕ :=
  monthly_income * worked_months

-- Prove that Stella's annual income last year was $49190
theorem stella_annual_income : annual_income monthly_income (total_months - unpaid_leave_months) = 49190 :=
by
  sorry

end NUMINAMATH_GPT_stella_annual_income_l2274_227456


namespace NUMINAMATH_GPT_seventh_fifth_tiles_difference_l2274_227413

def side_length (n : ℕ) : ℕ := 2 * n - 1
def number_of_tiles (n : ℕ) : ℕ := (side_length n) ^ 2
def tiles_difference (n m : ℕ) : ℕ := number_of_tiles n - number_of_tiles m

theorem seventh_fifth_tiles_difference : tiles_difference 7 5 = 88 := by
  sorry

end NUMINAMATH_GPT_seventh_fifth_tiles_difference_l2274_227413


namespace NUMINAMATH_GPT_part1_part2_l2274_227436

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x
noncomputable def h (x : ℝ) : ℝ := -3 * Real.log x + x^3 + (2 * x^2 - 4 * x) * Real.exp x + 7

theorem part1 (a : ℤ) : 
  (∀ x, (a : ℝ) < x ∧ x < a + 5 → ∀ y, (a : ℝ) < y ∧ y < a + 5 → f x ≤ f y) →
  a = -6 ∨ a = -5 ∨ a = -4 :=
sorry

theorem part2 (x : ℝ) (hx : 0 < x) : 
  f x < h x :=
sorry

end NUMINAMATH_GPT_part1_part2_l2274_227436


namespace NUMINAMATH_GPT_find_k_l2274_227427

theorem find_k : ∃ k : ℝ, (3 * k - 4) / (k + 7) = 2 / 5 ∧ k = 34 / 13 :=
by
  use 34 / 13
  sorry

end NUMINAMATH_GPT_find_k_l2274_227427


namespace NUMINAMATH_GPT_koala_fiber_intake_l2274_227448

theorem koala_fiber_intake 
  (absorption_rate : ℝ) 
  (absorbed_fiber : ℝ) 
  (eaten_fiber : ℝ) 
  (h1 : absorption_rate = 0.40) 
  (h2 : absorbed_fiber = 16)
  (h3 : absorbed_fiber = absorption_rate * eaten_fiber) :
  eaten_fiber = 40 := 
  sorry

end NUMINAMATH_GPT_koala_fiber_intake_l2274_227448


namespace NUMINAMATH_GPT_avg_ABC_l2274_227400

variables (A B C : Set ℕ) -- Sets of people
variables (a b c : ℕ) -- Numbers of people in sets A, B, and C respectively
variables (sum_A sum_B sum_C : ℕ) -- Sums of the ages of people in sets A, B, and C respectively

-- Given conditions
axiom avg_A : sum_A / a = 30
axiom avg_B : sum_B / b = 20
axiom avg_C : sum_C / c = 45

axiom avg_AB : (sum_A + sum_B) / (a + b) = 25
axiom avg_AC : (sum_A + sum_C) / (a + c) = 40
axiom avg_BC : (sum_B + sum_C) / (b + c) = 32

theorem avg_ABC : (sum_A + sum_B + sum_C) / (a + b + c) = 35 :=
by
  sorry

end NUMINAMATH_GPT_avg_ABC_l2274_227400


namespace NUMINAMATH_GPT_circumferences_ratio_l2274_227403

theorem circumferences_ratio (r1 r2 : ℝ) (h : (π * r1 ^ 2) / (π * r2 ^ 2) = 49 / 64) : r1 / r2 = 7 / 8 :=
sorry

end NUMINAMATH_GPT_circumferences_ratio_l2274_227403


namespace NUMINAMATH_GPT_green_more_than_blue_l2274_227471

variable (B Y G : ℕ)

theorem green_more_than_blue
  (h_sum : B + Y + G = 126)
  (h_ratio : ∃ k : ℕ, B = 3 * k ∧ Y = 7 * k ∧ G = 8 * k) :
  G - B = 35 := by
  sorry

end NUMINAMATH_GPT_green_more_than_blue_l2274_227471


namespace NUMINAMATH_GPT_speed_of_goods_train_l2274_227444

theorem speed_of_goods_train 
  (t₁ t₂ v_express : ℝ)
  (h1 : v_express = 90) 
  (h2 : t₁ = 6) 
  (h3 : t₂ = 4)
  (h4 : v_express * t₂ = v * (t₁ + t₂)) : 
  v = 36 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_goods_train_l2274_227444


namespace NUMINAMATH_GPT_earnings_difference_is_200_l2274_227496

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end NUMINAMATH_GPT_earnings_difference_is_200_l2274_227496


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_ellipse_l2274_227464

def is_ellipse (m : ℝ) : Prop := 
  1 < m ∧ m < 3 ∧ m ≠ 2

theorem necessary_but_not_sufficient_ellipse (m : ℝ) :
  (1 < m ∧ m < 3) → (m ≠ 2) → is_ellipse m :=
by
  intros h₁ h₂
  have h : 1 < m ∧ m < 3 ∧ m ≠ 2 := ⟨h₁.left, h₁.right, h₂⟩
  exact h

end NUMINAMATH_GPT_necessary_but_not_sufficient_ellipse_l2274_227464


namespace NUMINAMATH_GPT_range_of_n_l2274_227472

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end NUMINAMATH_GPT_range_of_n_l2274_227472


namespace NUMINAMATH_GPT_betty_age_l2274_227411

-- Define the constants and conditions
variables (A M B : ℕ)
variables (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 8)

-- Define the theorem to prove Betty's age
theorem betty_age : B = 4 :=
by sorry

end NUMINAMATH_GPT_betty_age_l2274_227411


namespace NUMINAMATH_GPT_number_of_sequences_with_at_least_two_reds_l2274_227461

theorem number_of_sequences_with_at_least_two_reds (n : ℕ) (h : n ≥ 2) :
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2 :=
by
  intros
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  show T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2
  sorry

end NUMINAMATH_GPT_number_of_sequences_with_at_least_two_reds_l2274_227461


namespace NUMINAMATH_GPT_ellipse_focus_and_axes_l2274_227492

theorem ellipse_focus_and_axes (m : ℝ) :
  (∃ a b : ℝ, (a > b) ∧ (mx^2 + y^2 = 1) ∧ (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * a = 3 * 2 * b)) → 
  m = 4 / 9 :=
by
  intro h
  rcases h with ⟨a, b, hab, h_eq, ha, hb, ha_b_eq⟩
  sorry

end NUMINAMATH_GPT_ellipse_focus_and_axes_l2274_227492


namespace NUMINAMATH_GPT_sequence_value_at_20_l2274_227447

open Nat

def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 4

theorem sequence_value_at_20 (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 20 = 77 :=
sorry

end NUMINAMATH_GPT_sequence_value_at_20_l2274_227447


namespace NUMINAMATH_GPT_nickys_pace_l2274_227477

theorem nickys_pace (distance : ℝ) (head_start_time : ℝ) (cristina_pace : ℝ) 
    (time_before_catchup : ℝ) (nicky_distance : ℝ) :
    distance = 100 ∧ head_start_time = 12 ∧ cristina_pace = 5 
    ∧ time_before_catchup = 30 ∧ nicky_distance = 90 →
    nicky_distance / time_before_catchup = 3 :=
by
  sorry

end NUMINAMATH_GPT_nickys_pace_l2274_227477


namespace NUMINAMATH_GPT_problem_statement_l2274_227404

-- Problem statement in Lean 4
theorem problem_statement (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2274_227404


namespace NUMINAMATH_GPT_sales_tax_difference_l2274_227406

-- Definitions for the price and tax rates
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.065
def tax_rate2 : ℝ := 0.06
def tax_rate3 : ℝ := 0.07

-- Sales tax amounts derived from the given rates and item price
def tax_amount (rate : ℝ) (price : ℝ) : ℝ := rate * price

-- Calculate the individual tax amounts
def tax_amount1 : ℝ := tax_amount tax_rate1 item_price
def tax_amount2 : ℝ := tax_amount tax_rate2 item_price
def tax_amount3 : ℝ := tax_amount tax_rate3 item_price

-- Proposition stating the proof problem
theorem sales_tax_difference :
  max tax_amount1 (max tax_amount2 tax_amount3) - min tax_amount1 (min tax_amount2 tax_amount3) = 0.50 :=
by 
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l2274_227406


namespace NUMINAMATH_GPT_trigonometric_identity_l2274_227453

theorem trigonometric_identity
  (α β : Real)
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2274_227453


namespace NUMINAMATH_GPT_inequality_positive_reals_l2274_227415

open Real

variable (x y : ℝ)

theorem inequality_positive_reals (hx : 0 < x) (hy : 0 < y) : x^2 + (8 / (x * y)) + y^2 ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_positive_reals_l2274_227415


namespace NUMINAMATH_GPT_range_of_a_l2274_227431

theorem range_of_a (a : ℝ) (x : ℝ) : (x^2 + 2*x > 3) → (x > a) → (¬ (x^2 + 2*x > 3) → ¬ (x > a)) → a ≥ 1 :=
by
  intros hp hq hr
  sorry

end NUMINAMATH_GPT_range_of_a_l2274_227431


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l2274_227485

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_of_quadratic_l2274_227485


namespace NUMINAMATH_GPT_train_length_is_120_l2274_227481

-- Definitions based on conditions
def bridge_length : ℕ := 600
def total_time : ℕ := 30
def on_bridge_time : ℕ := 20

-- Proof statement
theorem train_length_is_120 (x : ℕ) (speed1 speed2 : ℕ) :
  (speed1 = (bridge_length + x) / total_time) ∧
  (speed2 = bridge_length / on_bridge_time) ∧
  (speed1 = speed2) →
  x = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_120_l2274_227481


namespace NUMINAMATH_GPT_replace_star_l2274_227437

theorem replace_star (x : ℕ) : 2 * 18 * 14 = 6 * x * 7 → x = 12 :=
sorry

end NUMINAMATH_GPT_replace_star_l2274_227437


namespace NUMINAMATH_GPT_eval_expression_at_neg_one_l2274_227442

variable (x : ℤ)

theorem eval_expression_at_neg_one : x = -1 → 3 * x ^ 2 + 2 * x - 1 = 0 := by
  intro h
  rw [h]
  show 3 * (-1) ^ 2 + 2 * (-1) - 1 = 0
  sorry

end NUMINAMATH_GPT_eval_expression_at_neg_one_l2274_227442


namespace NUMINAMATH_GPT_number_of_standing_demons_l2274_227484

variable (N : ℕ)
variable (initial_knocked_down : ℕ)
variable (initial_standing : ℕ)
variable (current_knocked_down : ℕ)
variable (current_standing : ℕ)

axiom initial_condition : initial_knocked_down = (3 * initial_standing) / 2
axiom condition_after_changes : current_knocked_down = initial_knocked_down + 2
axiom condition_after_changes_2 : current_standing = initial_standing - 10
axiom final_condition : current_standing = (5 * current_knocked_down) / 4

theorem number_of_standing_demons : current_standing = 35 :=
sorry

end NUMINAMATH_GPT_number_of_standing_demons_l2274_227484


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2274_227424

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2274_227424


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l2274_227483

variable (a : ℕ → ℕ)

def arithmetic_seq (a₁ d : ℕ) : ℕ → ℕ :=
  λ n => a₁ + n * d

theorem arithmetic_seq_problem (a₁ d : ℕ)
  (h_cond : (arithmetic_seq a₁ d 1) + 2 * (arithmetic_seq a₁ d 5) + (arithmetic_seq a₁ d 9) = 120)
  : (arithmetic_seq a₁ d 2) + (arithmetic_seq a₁ d 8) = 60 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l2274_227483


namespace NUMINAMATH_GPT_Brian_traveled_60_miles_l2274_227454

theorem Brian_traveled_60_miles (mpg gallons : ℕ) (hmpg : mpg = 20) (hgallons : gallons = 3) :
    mpg * gallons = 60 := by
  sorry

end NUMINAMATH_GPT_Brian_traveled_60_miles_l2274_227454


namespace NUMINAMATH_GPT_median_of_consecutive_integers_l2274_227425

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end NUMINAMATH_GPT_median_of_consecutive_integers_l2274_227425


namespace NUMINAMATH_GPT_problem_solution_l2274_227499

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
  1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
  1 / (3 - Real.sqrt 2)

theorem problem_solution : S = 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l2274_227499


namespace NUMINAMATH_GPT_negation_of_every_student_is_punctual_l2274_227480

variable (Student : Type) (student punctual : Student → Prop)

theorem negation_of_every_student_is_punctual :
  ¬ (∀ x, student x → punctual x) ↔ ∃ x, student x ∧ ¬ punctual x := by
sorry

end NUMINAMATH_GPT_negation_of_every_student_is_punctual_l2274_227480


namespace NUMINAMATH_GPT_solve_equation_l2274_227443

def equation (x : ℝ) : Prop := (2 / x + 3 * (4 / x / (8 / x)) = 1.2)

theorem solve_equation : 
  ∃ x : ℝ, equation x ∧ x = - 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2274_227443


namespace NUMINAMATH_GPT_gumball_problem_l2274_227469

/--
A gumball machine contains 10 red, 6 white, 8 blue, and 9 green gumballs.
The least number of gumballs a person must buy to be sure of getting four gumballs of the same color is 13.
-/
theorem gumball_problem
  (red white blue green : ℕ)
  (h_red : red = 10)
  (h_white : white = 6)
  (h_blue : blue = 8)
  (h_green : green = 9) :
  ∃ n, n = 13 ∧ (∀ gumballs : ℕ, gumballs ≥ 13 → (∃ color_count : ℕ, color_count ≥ 4 ∧ (color_count = red ∨ color_count = white ∨ color_count = blue ∨ color_count = green))) :=
sorry

end NUMINAMATH_GPT_gumball_problem_l2274_227469


namespace NUMINAMATH_GPT_maximize_sum_of_sides_l2274_227487

theorem maximize_sum_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_b : b = 2) (h_B : B = (Real.pi / 3)) (h_law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) :
  a + c ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_maximize_sum_of_sides_l2274_227487


namespace NUMINAMATH_GPT_three_digit_number_division_l2274_227432

theorem three_digit_number_division :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, 10 ≤ m ∧ m < 100 ∧ n / m = 8 ∧ n % m = 6) → n = 342 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_division_l2274_227432


namespace NUMINAMATH_GPT_complement_U_A_l2274_227434

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_U_A : (U \ A) = {3, 9} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l2274_227434


namespace NUMINAMATH_GPT_find_all_f_l2274_227466

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_all_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x + 2 * y^2) →
  ∃ a c : ℝ, (∀ x : ℝ, f x = x^2 + a * x + c) ∧ (a^2 - 4 * c ≤ 0) := sorry

end NUMINAMATH_GPT_find_all_f_l2274_227466


namespace NUMINAMATH_GPT_Sarah_is_26_l2274_227401

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end NUMINAMATH_GPT_Sarah_is_26_l2274_227401


namespace NUMINAMATH_GPT_fruit_difference_l2274_227491

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end NUMINAMATH_GPT_fruit_difference_l2274_227491


namespace NUMINAMATH_GPT_max_value_f_on_interval_l2274_227474

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 - 1 := sorry

end NUMINAMATH_GPT_max_value_f_on_interval_l2274_227474


namespace NUMINAMATH_GPT_base_k_number_to_decimal_l2274_227467

theorem base_k_number_to_decimal (k : ℕ) (h : 4 ≤ k) : 1 * k^2 + 3 * k + 2 = 30 ↔ k = 4 := by
  sorry

end NUMINAMATH_GPT_base_k_number_to_decimal_l2274_227467


namespace NUMINAMATH_GPT_total_earnings_proof_l2274_227490

noncomputable def total_earnings (x y : ℝ) : ℝ :=
  let earnings_a := (18 * x * y) / 100
  let earnings_b := (20 * x * y) / 100
  let earnings_c := (20 * x * y) / 100
  earnings_a + earnings_b + earnings_c

theorem total_earnings_proof (x y : ℝ) (h : 2 * x * y = 15000) :
  total_earnings x y = 4350 := by
  sorry

end NUMINAMATH_GPT_total_earnings_proof_l2274_227490


namespace NUMINAMATH_GPT_graph_of_equation_l2274_227458

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end NUMINAMATH_GPT_graph_of_equation_l2274_227458


namespace NUMINAMATH_GPT_trapezoid_combined_area_correct_l2274_227433

noncomputable def combined_trapezoid_area_proof : Prop :=
  let EF : ℝ := 60
  let GH : ℝ := 40
  let altitude_EF_GH : ℝ := 18
  let trapezoid_EFGH_area : ℝ := (1 / 2) * (EF + GH) * altitude_EF_GH

  let IJ : ℝ := 30
  let KL : ℝ := 25
  let altitude_IJ_KL : ℝ := 10
  let trapezoid_IJKL_area : ℝ := (1 / 2) * (IJ + KL) * altitude_IJ_KL

  let combined_area : ℝ := trapezoid_EFGH_area + trapezoid_IJKL_area

  combined_area = 1175

theorem trapezoid_combined_area_correct : combined_trapezoid_area_proof := by
  sorry

end NUMINAMATH_GPT_trapezoid_combined_area_correct_l2274_227433


namespace NUMINAMATH_GPT_five_star_three_eq_ten_l2274_227452

def operation (a b : ℝ) : ℝ := b^2 + 1

theorem five_star_three_eq_ten : operation 5 3 = 10 := by
  sorry

end NUMINAMATH_GPT_five_star_three_eq_ten_l2274_227452


namespace NUMINAMATH_GPT_find_a_l2274_227459

-- Define the variables
variables (m d a b : ℝ)

-- State the main theorem with conditions
theorem find_a (h : m = d * a * b / (a - b)) (h_ne : m ≠ d * b) : a = m * b / (m - d * b) :=
sorry

end NUMINAMATH_GPT_find_a_l2274_227459


namespace NUMINAMATH_GPT_green_beads_in_pattern_l2274_227402

noncomputable def G : ℕ := 3
def P : ℕ := 5
def R (G : ℕ) : ℕ := 2 * G
def total_beads (G : ℕ) (P : ℕ) (R : ℕ) : ℕ := 3 * (G + P + R) + 10 * 5 * (G + P + R)

theorem green_beads_in_pattern :
  total_beads 3 5 (R 3) = 742 :=
by
  sorry

end NUMINAMATH_GPT_green_beads_in_pattern_l2274_227402


namespace NUMINAMATH_GPT_ratio_is_l2274_227435

noncomputable def volume_dodecahedron (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s ^ 3

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := Real.sqrt 2 / 12 * ((Real.sqrt 3 / 2) * s) ^ 3

noncomputable def ratio_volumes (s : ℝ) : ℝ := volume_dodecahedron s / volume_tetrahedron s

theorem ratio_is (s : ℝ) : ratio_volumes s = (60 + 28 * Real.sqrt 5) / Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_is_l2274_227435


namespace NUMINAMATH_GPT_inequality_proof_l2274_227463

theorem inequality_proof (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2274_227463


namespace NUMINAMATH_GPT_total_time_simultaneous_l2274_227455

def total_time_bread1 : Nat := 30 + 120 + 20 + 120 + 10 + 30 + 30 + 15
def total_time_bread2 : Nat := 90 + 15 + 20 + 25 + 10
def total_time_bread3 : Nat := 40 + 100 + 5 + 110 + 15 + 5 + 25 + 20

theorem total_time_simultaneous :
  max (max total_time_bread1 total_time_bread2) total_time_bread3 = 375 :=
by
  sorry

end NUMINAMATH_GPT_total_time_simultaneous_l2274_227455


namespace NUMINAMATH_GPT_right_triangle_area_inscribed_circle_l2274_227428

theorem right_triangle_area_inscribed_circle (r a b c : ℝ)
  (h_c : c = 6 + 7)
  (h_a : a = 6 + r)
  (h_b : b = 7 + r)
  (h_pyth : (6 + r)^2 + (7 + r)^2 = 13^2):
  (1 / 2) * (a * b) = 42 :=
by
  -- The necessary calculations have already been derived and verified
  sorry

end NUMINAMATH_GPT_right_triangle_area_inscribed_circle_l2274_227428


namespace NUMINAMATH_GPT_find_number_l2274_227449

theorem find_number (x : ℝ) : (x + 1) / (x + 5) = (x + 5) / (x + 13) → x = 3 :=
sorry

end NUMINAMATH_GPT_find_number_l2274_227449


namespace NUMINAMATH_GPT_find_n_l2274_227488

theorem find_n (n : ℕ) (hn_pos : 0 < n) (hn_greater_30 : 30 < n) 
  (divides : (4 * n - 1) ∣ 2002 * n) : n = 36 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l2274_227488


namespace NUMINAMATH_GPT_log_40_cannot_be_directly_calculated_l2274_227414

theorem log_40_cannot_be_directly_calculated (log_3 log_5 : ℝ) (h1 : log_3 = 0.4771) (h2 : log_5 = 0.6990) : 
  ¬ (exists (log_40 : ℝ), (log_40 = (log_3 + log_5) + log_40)) :=
by {
  sorry
}

end NUMINAMATH_GPT_log_40_cannot_be_directly_calculated_l2274_227414


namespace NUMINAMATH_GPT_find_original_price_l2274_227460

-- Define the conditions provided in the problem
def original_price (P : ℝ) : Prop :=
  let first_discount := 0.90 * P
  let second_discount := 0.85 * first_discount
  let taxed_price := 1.08 * second_discount
  taxed_price = 450

-- State and prove the main theorem
theorem find_original_price (P : ℝ) (h : original_price P) : P = 544.59 :=
  sorry

end NUMINAMATH_GPT_find_original_price_l2274_227460
