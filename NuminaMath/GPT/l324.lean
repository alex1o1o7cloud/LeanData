import Mathlib

namespace find_one_third_of_product_l324_32437

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l324_32437


namespace subset_relation_l324_32459

def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

theorem subset_relation : N ⊆ M := by
  sorry

end subset_relation_l324_32459


namespace jen_visits_exactly_two_countries_l324_32448

noncomputable def probability_of_visiting_exactly_two_countries (p_chile p_madagascar p_japan p_egypt : ℝ) : ℝ :=
  let p_chile_madagascar := (p_chile * p_madagascar) * (1 - p_japan) * (1 - p_egypt)
  let p_chile_japan := (p_chile * p_japan) * (1 - p_madagascar) * (1 - p_egypt)
  let p_chile_egypt := (p_chile * p_egypt) * (1 - p_madagascar) * (1 - p_japan)
  let p_madagascar_japan := (p_madagascar * p_japan) * (1 - p_chile) * (1 - p_egypt)
  let p_madagascar_egypt := (p_madagascar * p_egypt) * (1 - p_chile) * (1 - p_japan)
  let p_japan_egypt := (p_japan * p_egypt) * (1 - p_chile) * (1 - p_madagascar)
  p_chile_madagascar + p_chile_japan + p_chile_egypt + p_madagascar_japan + p_madagascar_egypt + p_japan_egypt

theorem jen_visits_exactly_two_countries :
  probability_of_visiting_exactly_two_countries 0.4 0.35 0.2 0.15 = 0.2432 :=
by
  sorry

end jen_visits_exactly_two_countries_l324_32448


namespace ab_sum_pow_eq_neg_one_l324_32474

theorem ab_sum_pow_eq_neg_one (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) : (a + b) ^ 2003 = -1 := 
by
  sorry

end ab_sum_pow_eq_neg_one_l324_32474


namespace blue_notes_per_red_note_l324_32493

-- Given conditions
def total_red_notes : ℕ := 5 * 6
def additional_blue_notes : ℕ := 10
def total_notes : ℕ := 100
def total_blue_notes := total_notes - total_red_notes

-- Proposition that needs to be proved
theorem blue_notes_per_red_note (x : ℕ) : total_red_notes * x + additional_blue_notes = total_blue_notes → x = 2 := by
  intro h
  sorry

end blue_notes_per_red_note_l324_32493


namespace white_area_of_sign_remains_l324_32430

theorem white_area_of_sign_remains (h1 : (6 * 18 = 108))
  (h2 : 9 = 6 + 3)
  (h3 : 7.5 = 5 + 3 - 0.5)
  (h4 : 13 = 9 + 4)
  (h5 : 9 = 6 + 3)
  (h6 : 38.5 = 9 + 7.5 + 13 + 9)
  : 108 - 38.5 = 69.5 := by
  sorry

end white_area_of_sign_remains_l324_32430


namespace multiplication_distributive_example_l324_32405

theorem multiplication_distributive_example : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end multiplication_distributive_example_l324_32405


namespace range_of_a_l324_32439

theorem range_of_a (a : ℝ) (e : ℝ) (x : ℝ) (ln : ℝ → ℝ) :
  (∀ x, (1 / e) ≤ x ∧ x ≤ e → (a - x^2 = -2 * ln x)) →
  (1 ≤ a ∧ a ≤ (e^2 - 2)) :=
by
  sorry

end range_of_a_l324_32439


namespace max_product_l324_32480

noncomputable def max_of_product (x y : ℝ) : ℝ := x * y

theorem max_product (x y : ℝ) (h1 : x ∈ Set.Ioi 0) (h2 : y ∈ Set.Ioi 0) (h3 : x + 4 * y = 1) :
  max_of_product x y ≤ 1 / 16 := sorry

end max_product_l324_32480


namespace imaginary_part_of_fraction_l324_32465

theorem imaginary_part_of_fraction (i : ℂ) (h : i^2 = -1) : ( (i^2) / (2 * i - 1) ).im = (2 / 5) :=
by
  sorry

end imaginary_part_of_fraction_l324_32465


namespace problem1_problem2_l324_32481

-- Problem 1: Prove that x = ±7/2 given 4x^2 - 49 = 0
theorem problem1 (x : ℝ) : 4 * x^2 - 49 = 0 → x = 7 / 2 ∨ x = -7 / 2 := 
by
  sorry

-- Problem 2: Prove that x = 2 given (x + 1)^3 - 27 = 0
theorem problem2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 := 
by
  sorry

end problem1_problem2_l324_32481


namespace milk_needed_for_one_batch_l324_32492

-- Define cost of one batch given amount of milk M
def cost_of_one_batch (M : ℝ) : ℝ := 1.5 * M + 6

-- Define cost of three batches
def cost_of_three_batches (M : ℝ) : ℝ := 3 * cost_of_one_batch M

theorem milk_needed_for_one_batch : ∃ M : ℝ, cost_of_three_batches M = 63 ∧ M = 10 :=
by
  sorry

end milk_needed_for_one_batch_l324_32492


namespace orchestra_member_count_l324_32404

theorem orchestra_member_count :
  ∃ x : ℕ, 150 ≤ x ∧ x ≤ 250 ∧ 
           x % 4 = 2 ∧
           x % 5 = 3 ∧
           x % 8 = 4 ∧
           x % 9 = 5 :=
sorry

end orchestra_member_count_l324_32404


namespace parallel_lines_slope_l324_32441

theorem parallel_lines_slope (d : ℝ) (h : 3 = 4 * d) : d = 3 / 4 :=
by
  sorry

end parallel_lines_slope_l324_32441


namespace A_days_to_complete_work_l324_32487

noncomputable def work (W : ℝ) (A_work_per_day B_work_per_day : ℝ) (days_A days_B days_B_alone : ℝ) : ℝ :=
  A_work_per_day * days_A + B_work_per_day * days_B

theorem A_days_to_complete_work 
  (W : ℝ)
  (A_work_per_day B_work_per_day : ℝ)
  (days_A days_B days_B_alone : ℝ)
  (h1 : days_A = 5)
  (h2 : days_B = 12)
  (h3 : days_B_alone = 18)
  (h4 : B_work_per_day = W / days_B_alone)
  (h5 : work W A_work_per_day B_work_per_day days_A days_B days_B_alone = W) :
  W / A_work_per_day = 15 := 
sorry

end A_days_to_complete_work_l324_32487


namespace scientific_notation_41600_l324_32477

theorem scientific_notation_41600 : (4.16 * 10^4) = 41600 := by
  sorry

end scientific_notation_41600_l324_32477


namespace range_of_uv_sq_l324_32482

theorem range_of_uv_sq (u v w : ℝ) (h₀ : 0 ≤ u) (h₁ : 0 ≤ v) (h₂ : 0 ≤ w) (h₃ : u + v + w = 2) :
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 :=
sorry

end range_of_uv_sq_l324_32482


namespace find_angle_A_find_area_triangle_l324_32452

-- Definitions for the triangle and the angles
def triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- Given conditions
variables (a b c A B C : ℝ)
variables (hTriangle : triangle A B C)
variables (hEq : 2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C)
variables (hAngleB : B = Real.pi / 6)
variables (hMedianAM : Real.sqrt 7 = Real.sqrt (b^2 + (b / 2)^2 - 2 * b * (b / 2) * Real.cos (2 * Real.pi / 3)))

-- Proof statements
theorem find_angle_A : A = Real.pi / 6 :=
sorry

theorem find_area_triangle : (1/2) * b^2 * Real.sin C = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_triangle_l324_32452


namespace contradiction_proof_l324_32407

theorem contradiction_proof (x y : ℝ) (h1 : x + y ≤ 0) (h2 : x > 0) (h3 : y > 0) : false :=
by
  sorry

end contradiction_proof_l324_32407


namespace dot_product_result_parallelism_condition_l324_32433

-- Definitions of the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

-- 1. Prove the dot product result
theorem dot_product_result :
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_2b := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  a_plus_b.1 * a_minus_2b.1 + a_plus_b.2 * a_minus_2b.2 = -14 :=
by
  sorry

-- 2. Prove parallelism condition
theorem parallelism_condition (k : ℝ) :
  let k_a_plus_b := (k * a.1 + b.1, k * a.2 + b.2)
  let a_minus_3b := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  k = -1/3 → k_a_plus_b.1 * a_minus_3b.2 = k_a_plus_b.2 * a_minus_3b.1 :=
by
  sorry

end dot_product_result_parallelism_condition_l324_32433


namespace john_must_deliver_1063_pizzas_l324_32415

-- Declare all the given conditions
def car_cost : ℕ := 8000
def maintenance_cost : ℕ := 500
def pizza_income (p : ℕ) : ℕ := 12 * p
def gas_cost (p : ℕ) : ℕ := 4 * p

-- Define the function that returns the net earnings
def net_earnings (p : ℕ) := pizza_income p - gas_cost p

-- Define the total expenses
def total_expenses : ℕ := car_cost + maintenance_cost

-- Define the minimum number of pizzas John must deliver
def minimum_pizzas (p : ℕ) : Prop := net_earnings p ≥ total_expenses

-- State the theorem that needs to be proved
theorem john_must_deliver_1063_pizzas : minimum_pizzas 1063 := by
  sorry

end john_must_deliver_1063_pizzas_l324_32415


namespace december_sales_multiple_l324_32417

   noncomputable def find_sales_multiple (A : ℝ) (x : ℝ) :=
     x * A = 0.3888888888888889 * (11 * A + x * A)

   theorem december_sales_multiple (A : ℝ) (x : ℝ) (h : find_sales_multiple A x) : x = 7 :=
   by 
     sorry
   
end december_sales_multiple_l324_32417


namespace find_radius_l324_32413

def radius_of_circle (d : ℤ) (PQ : ℕ) (QR : ℕ) (r : ℕ) : Prop := 
  let PR := PQ + QR
  (PQ * PR = (d - r) * (d + r)) ∧ (d = 15) ∧ (PQ = 11) ∧ (QR = 8) ∧ (r = 4)

-- Now stating the theorem to prove the radius r given the conditions
theorem find_radius (r : ℕ) : radius_of_circle 15 11 8 r := by
  sorry

end find_radius_l324_32413


namespace not_possible_last_digit_l324_32471

theorem not_possible_last_digit :
  ∀ (S : ℕ) (a : Fin 111 → ℕ),
  (∀ i, a i ≤ 500) →
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i, (a i) % 10 = (S - a i) % 10) →
  False :=
by
  intro S a h1 h2 h3
  sorry

end not_possible_last_digit_l324_32471


namespace triangle_area_l324_32463

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area_l324_32463


namespace square_b_perimeter_l324_32422

theorem square_b_perimeter (a b : ℝ) 
  (ha : a^2 = 65) 
  (prob : (65 - b^2) / 65 = 0.7538461538461538) : 
  4 * b = 16 :=
by 
  sorry

end square_b_perimeter_l324_32422


namespace roof_shingle_width_l324_32434

theorem roof_shingle_width (L A W : ℕ) (hL : L = 10) (hA : A = 70) (hArea : A = L * W) : W = 7 :=
by
  sorry

end roof_shingle_width_l324_32434


namespace prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l324_32473

def num_outcomes := 36

def same_points_events := 6
def less_than_seven_events := 15
def greater_than_or_equal_eleven_events := 3

def prob_same_points := (same_points_events : ℚ) / num_outcomes
def prob_less_than_seven := (less_than_seven_events : ℚ) / num_outcomes
def prob_greater_or_equal_eleven := (greater_than_or_equal_eleven_events : ℚ) / num_outcomes

theorem prob_same_points_eq : prob_same_points = 1 / 6 := by
  sorry

theorem prob_less_than_seven_eq : prob_less_than_seven = 5 / 12 := by
  sorry

theorem prob_greater_or_equal_eleven_eq : prob_greater_or_equal_eleven = 1 / 12 := by
  sorry

end prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l324_32473


namespace jim_saves_money_by_buying_gallon_l324_32424

theorem jim_saves_money_by_buying_gallon :
  let gallon_price := 8
  let bottle_price := 3
  let ounces_per_gallon := 128
  let ounces_per_bottle := 16
  (ounces_per_gallon / ounces_per_bottle) * bottle_price - gallon_price = 16 :=
by
  sorry

end jim_saves_money_by_buying_gallon_l324_32424


namespace time_saved_by_taking_route_B_l324_32464

-- Defining the times for the routes A and B
def time_route_A_one_way : ℕ := 5
def time_route_B_one_way : ℕ := 2

-- The total round trip times
def time_route_A_round_trip : ℕ := 2 * time_route_A_one_way
def time_route_B_round_trip : ℕ := 2 * time_route_B_one_way

-- The statement to prove
theorem time_saved_by_taking_route_B :
  time_route_A_round_trip - time_route_B_round_trip = 6 :=
by
  -- Proof would go here
  sorry

end time_saved_by_taking_route_B_l324_32464


namespace genuine_product_probability_l324_32476

-- Define the probabilities as constants
def P_second_grade := 0.03
def P_third_grade := 0.01

-- Define the total probability (outcome must be either genuine or substandard)
def P_substandard := P_second_grade + P_third_grade
def P_genuine := 1 - P_substandard

-- The statement to be proved
theorem genuine_product_probability :
  P_genuine = 0.96 :=
sorry

end genuine_product_probability_l324_32476


namespace find_a_b_l324_32467

noncomputable def parabola_props (a b : ℝ) : Prop :=
a ≠ 0 ∧ 
∀ x : ℝ, a * x^2 + b * x - 4 = (1 / 2) * x^2 + x - 4

theorem find_a_b {a b : ℝ} (h1 : parabola_props a b) : 
a = 1 / 2 ∧ b = -1 :=
sorry

end find_a_b_l324_32467


namespace tangent_line_to_circle_range_mn_l324_32484

theorem tangent_line_to_circle_range_mn (m n : ℝ) 
  (h1 : (m + 1) * (m + 1) + (n + 1) * (n + 1) = 4) :
  (m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end tangent_line_to_circle_range_mn_l324_32484


namespace tom_father_time_saved_correct_l324_32491

def tom_father_jog_time_saved : Prop :=
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 5
  let daily_distance := 3
  let hours_to_minutes := 60

  let monday_time := daily_distance / monday_speed
  let tuesday_time := daily_distance / tuesday_speed
  let thursday_time := daily_distance / thursday_speed
  let saturday_time := daily_distance / saturday_speed

  let total_time_original := monday_time + tuesday_time + thursday_time + saturday_time
  let always_5mph_time := 4 * (daily_distance / 5)
  let time_saved := total_time_original - always_5mph_time

  let time_saved_minutes := time_saved * hours_to_minutes

  time_saved_minutes = 3

theorem tom_father_time_saved_correct : tom_father_jog_time_saved := by
  sorry

end tom_father_time_saved_correct_l324_32491


namespace bisect_segment_l324_32490

variables {A B C D E P : Point}
variables {α β γ δ ε : Real} -- angles in degrees
variables {BD CE : Line}

-- Geometric predicates
def Angle (x y z : Point) : Real := sorry -- calculates the angle ∠xyz

def isMidpoint (M A B : Point) : Prop := sorry -- M is the midpoint of segment AB

-- Given Conditions
variables (h1 : convex_pentagon A B C D E)
          (h2 : Angle B A C = Angle C A D ∧ Angle C A D = Angle D A E)
          (h3 : Angle A B C = Angle A C D ∧ Angle A C D = Angle A D E)
          (h4 : intersects BD CE P)

-- Conclusion to be proved
theorem bisect_segment : isMidpoint P C D :=
by {
  sorry -- proof to be filled in
}

end bisect_segment_l324_32490


namespace hyperbola_equation_l324_32486

noncomputable def hyperbola : Prop :=
  ∃ (a b : ℝ), 
    (2 : ℝ) * a = (3 : ℝ) * b ∧
    ∀ (x y : ℝ), (4 * x^2 - 9 * y^2 = -32) → (x = 1) ∧ (y = 2)

theorem hyperbola_equation (a b : ℝ) :
  (2 * a = 3 * b) ∧ (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = -32 → x = 1 ∧ y = 2) → 
  (9 / 32 * y^2 - x^2 / 8 = 1) :=
by
  sorry

end hyperbola_equation_l324_32486


namespace rhombus_longer_diagonal_length_l324_32410

theorem rhombus_longer_diagonal_length
  (side_length : ℕ) (shorter_diagonal : ℕ) 
  (side_length_eq : side_length = 53) 
  (shorter_diagonal_eq : shorter_diagonal = 50) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 94 := by
  sorry

end rhombus_longer_diagonal_length_l324_32410


namespace remainder_div_14_l324_32423

def S : ℕ := 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077

theorem remainder_div_14 : S % 14 = 7 :=
by
  sorry

end remainder_div_14_l324_32423


namespace geometric_sequence_product_l324_32421

theorem geometric_sequence_product {a : ℕ → ℝ} 
(h₁ : a 1 = 2) 
(h₂ : a 5 = 8) 
(h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
a 2 * a 3 * a 4 = 64 := 
sorry

end geometric_sequence_product_l324_32421


namespace ratio_of_pieces_l324_32431

-- Define the total length of the wire.
def total_length : ℕ := 14

-- Define the length of the shorter piece.
def shorter_piece_length : ℕ := 4

-- Define the length of the longer piece.
def longer_piece_length : ℕ := total_length - shorter_piece_length

-- Define the expected ratio of the lengths.
def ratio : ℚ := shorter_piece_length / longer_piece_length

-- State the theorem to prove.
theorem ratio_of_pieces : ratio = 2 / 5 := 
by {
  -- skip the proof
  sorry
}

end ratio_of_pieces_l324_32431


namespace find_salary_May_l324_32479

-- Define the salaries for each month as variables
variables (J F M A May : ℝ)

-- Declare the conditions as hypotheses
def avg_salary_Jan_to_Apr := (J + F + M + A) / 4 = 8000
def avg_salary_Feb_to_May := (F + M + A + May) / 4 = 8100
def salary_Jan := J = 6100

-- The theorem stating the salary for the month of May
theorem find_salary_May (h1 : avg_salary_Jan_to_Apr J F M A) (h2 : avg_salary_Feb_to_May F M A May) (h3 : salary_Jan J) :
  May = 6500 :=
  sorry

end find_salary_May_l324_32479


namespace john_volunteer_hours_l324_32402

noncomputable def total_volunteer_hours :=
  let first_six_months_hours := 2 * 3 * 6
  let next_five_months_hours := 1 * 2 * 4 * 5
  let december_hours := 3 * 2
  first_six_months_hours + next_five_months_hours + december_hours

theorem john_volunteer_hours : total_volunteer_hours = 82 := by
  sorry

end john_volunteer_hours_l324_32402


namespace rancher_steers_cows_solution_l324_32435

theorem rancher_steers_cows_solution :
  ∃ (s c : ℕ), s > 0 ∧ c > 0 ∧ (30 * s + 31 * c = 1200) ∧ (s = 9) ∧ (c = 30) :=
by
  sorry

end rancher_steers_cows_solution_l324_32435


namespace parabola_line_intersection_l324_32445

theorem parabola_line_intersection (p : ℝ) (hp : p > 0) 
  (line_eq : ∃ b : ℝ, ∀ x : ℝ, 2 * x + b = 2 * x - p/2) 
  (focus := (p / 4, 0))
  (point_A := (0, -p / 2))
  (area_OAF : 1 / 2 * (p / 4) * (p / 2) = 1) : 
  p = 4 :=
sorry

end parabola_line_intersection_l324_32445


namespace bucket_capacity_l324_32485

-- Given Conditions
variable (C : ℝ)
variable (h : (2 / 3) * C = 9)

-- Goal
theorem bucket_capacity : C = 13.5 := by
  sorry

end bucket_capacity_l324_32485


namespace factor_x_squared_minus_144_l324_32499

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) :=
by
  sorry

end factor_x_squared_minus_144_l324_32499


namespace find_certain_number_l324_32438

theorem find_certain_number (x certain_number : ℕ) (h: x = 3) (h2: certain_number = 5 * x + 4) : certain_number = 19 :=
by
  sorry

end find_certain_number_l324_32438


namespace percent_of_workday_in_meetings_l324_32406

theorem percent_of_workday_in_meetings (h1 : 9 > 0) (m1 m2 : ℕ) (h2 : m1 = 45) (h3 : m2 = 2 * m1) : 
  (135 / 540 : ℚ) * 100 = 25 := 
by
  -- Just for structure, the proof should go here
  sorry

end percent_of_workday_in_meetings_l324_32406


namespace total_students_is_46_l324_32495

-- Define the constants for the problem
def students_in_history : ℕ := 19
def students_in_math : ℕ := 14
def students_in_english : ℕ := 26
def students_in_all_three : ℕ := 3
def students_in_exactly_two : ℕ := 7

-- The total number of students as per the inclusion-exclusion principle
def total_students : ℕ :=
  students_in_history + students_in_math + students_in_english
  - students_in_exactly_two - 2 * students_in_all_three + students_in_all_three

theorem total_students_is_46 : total_students = 46 :=
  sorry

end total_students_is_46_l324_32495


namespace last_row_number_l324_32478

/-
Given:
1. Each row forms an arithmetic sequence.
2. The common differences of the rows are:
   - 1st row: common difference = 1
   - 2nd row: common difference = 2
   - 3rd row: common difference = 4
   - ...
   - 2015th row: common difference = 2^2014
3. The nth row starts with \( (n+1) \times 2^{n-2} \).

Prove:
The number in the last row (2016th row) is \( 2017 \times 2^{2014} \).
-/
theorem last_row_number
  (common_diff : ℕ → ℕ)
  (h1 : common_diff 1 = 1)
  (h2 : common_diff 2 = 2)
  (h3 : common_diff 3 = 4)
  (h_general : ∀ n, common_diff n = 2^(n-1))
  (first_number_in_row : ℕ → ℕ)
  (first_number_in_row_def : ∀ n, first_number_in_row n = (n + 1) * 2^(n - 2)) :
  first_number_in_row 2016 = 2017 * 2^2014 := by
    sorry

end last_row_number_l324_32478


namespace xy_identity_l324_32451

theorem xy_identity (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) : (x^2 + y^2) * (x + y) = 803 := by
  sorry

end xy_identity_l324_32451


namespace largest_divisor_of_n_l324_32429

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n^2 = 18 * k) : ∃ l : ℕ, n = 6 * l :=
sorry

end largest_divisor_of_n_l324_32429


namespace sine_ratio_comparison_l324_32494

theorem sine_ratio_comparison : (Real.sin (1 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) < (Real.sin (3 * Real.pi / 180) / Real.sin (4 * Real.pi / 180)) :=
sorry

end sine_ratio_comparison_l324_32494


namespace num_girls_l324_32411

theorem num_girls (boys girls : ℕ) (h1 : girls = boys + 228) (h2 : boys = 469) : girls = 697 :=
sorry

end num_girls_l324_32411


namespace kittens_price_l324_32472

theorem kittens_price (x : ℕ) 
  (h1 : 2 * x + 5 = 17) : x = 6 := by
  sorry

end kittens_price_l324_32472


namespace solve_for_x_l324_32460

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end solve_for_x_l324_32460


namespace find_second_term_l324_32466

theorem find_second_term 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h_sum : ∀ n, S n = n * (2 * n + 1))
  (h_S1 : S 1 = a 1) 
  (h_S2 : S 2 = a 1 + a 2) 
  (h_a1 : a 1 = 3) : 
  a 2 = 7 := 
sorry

end find_second_term_l324_32466


namespace bob_needs_8_additional_wins_to_afford_puppy_l324_32427

variable (n : ℕ) (grand_prize_per_win : ℝ) (total_cost : ℝ)

def bob_total_wins_to_afford_puppy : Prop :=
  total_cost = 1000 ∧ grand_prize_per_win = 100 ∧ n = (total_cost / grand_prize_per_win) - 2

theorem bob_needs_8_additional_wins_to_afford_puppy :
  bob_total_wins_to_afford_puppy 8 100 1000 :=
by {
  sorry
}

end bob_needs_8_additional_wins_to_afford_puppy_l324_32427


namespace max_value_of_expression_l324_32418

noncomputable def maximum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : x^2 - x*y + 2*y^2 = 8) : ℝ :=
  x^2 + x*y + 2*y^2

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^2 - x*y + 2*y^2 = 8) : maximum_value hx hy h = (72 + 32 * Real.sqrt 2) / 7 :=
by
  sorry

end max_value_of_expression_l324_32418


namespace line_contains_point_l324_32449

theorem line_contains_point (k : ℝ) : 
  let x := (1 : ℝ) / 3
  let y := -2 
  let line_eq := (3 : ℝ) - 3 * k * x = 4 * y
  line_eq → k = 11 :=
by
  intro h
  sorry

end line_contains_point_l324_32449


namespace ratio_first_term_to_common_difference_l324_32442

theorem ratio_first_term_to_common_difference
  (a d : ℝ)
  (S_n : ℕ → ℝ)
  (hS_n : ∀ n, S_n n = (n / 2) * (2 * a + (n - 1) * d))
  (h : S_n 15 = 3 * S_n 10) :
  a / d = -2 :=
by
  sorry

end ratio_first_term_to_common_difference_l324_32442


namespace find_M_N_l324_32454

-- Define positive integers less than 10
def is_pos_int_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Main theorem to prove M = 5 and N = 6 given the conditions
theorem find_M_N (M N : ℕ) (hM : is_pos_int_lt_10 M) (hN : is_pos_int_lt_10 N) 
  (h : 8 * (10 ^ 7) * M + 420852 * 9 = N * (10 ^ 7) * 9889788 * 11) : 
  M = 5 ∧ N = 6 :=
by {
  sorry
}

end find_M_N_l324_32454


namespace carpet_needed_for_room_l324_32400

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end carpet_needed_for_room_l324_32400


namespace remainder_of_sum_l324_32488

open Nat

theorem remainder_of_sum :
  (12345 + 12347 + 12349 + 12351 + 12353 + 12355 + 12357) % 16 = 9 :=
by 
  sorry

end remainder_of_sum_l324_32488


namespace simplify_expression_l324_32409

theorem simplify_expression :
  6^6 + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 :=
by sorry

end simplify_expression_l324_32409


namespace value_of_m_l324_32416

theorem value_of_m 
    (x : ℝ) (m : ℝ) 
    (h : 0 < x)
    (h_eq : (2 / (x - 2)) - ((2 * x - m) / (2 - x)) = 3) : 
    m = 6 := 
sorry

end value_of_m_l324_32416


namespace total_value_of_item_l324_32403

theorem total_value_of_item (V : ℝ) (h1 : 0.07 * (V - 1000) = 87.50) :
  V = 2250 :=
by
  sorry

end total_value_of_item_l324_32403


namespace effect_on_revenue_l324_32461

variables (P Q : ℝ)

def original_revenue : ℝ := P * Q
def new_price : ℝ := 1.60 * P
def new_quantity : ℝ := 0.80 * Q
def new_revenue : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue (h1 : new_price P = 1.60 * P) (h2 : new_quantity Q = 0.80 * Q) :
  new_revenue P Q - original_revenue P Q = 0.28 * original_revenue P Q :=
by
  sorry

end effect_on_revenue_l324_32461


namespace ellipse_equation_with_foci_l324_32457

theorem ellipse_equation_with_foci (M N P : ℝ × ℝ)
  (area_triangle : Real) (tan_M tan_N : ℝ)
  (h₁ : area_triangle = 1)
  (h₂ : tan_M = 1 / 2)
  (h₃ : tan_N = -2) :
  ∃ (a b : ℝ), (4 * x^2) / (15 : ℝ) + y^2 / (3 : ℝ) = 1 :=
by
  -- Definitions to meet given conditions would be here
  sorry

end ellipse_equation_with_foci_l324_32457


namespace perimeter_of_regular_pentagon_is_75_l324_32426

-- Define the side length and the property of the figure
def side_length : ℝ := 15
def is_regular_pentagon : Prop := true  -- assuming this captures the regular pentagon property

-- Define the perimeter calculation based on the conditions
def perimeter (n : ℕ) (side_length : ℝ) := n * side_length

-- The theorem to prove
theorem perimeter_of_regular_pentagon_is_75 :
  is_regular_pentagon → perimeter 5 side_length = 75 :=
by
  intro _ -- We don't need to use is_regular_pentagon directly
  rw [side_length]
  norm_num
  sorry

end perimeter_of_regular_pentagon_is_75_l324_32426


namespace additional_charge_fraction_of_mile_l324_32483

-- Conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_mile_fraction : ℝ := 0.15
def total_charge (distance : ℝ) : ℝ := 2.25 + 0.15 * distance
def trip_distance : ℝ := 3.6
def total_cost : ℝ := 3.60

-- Question
theorem additional_charge_fraction_of_mile :
  ∃ f : ℝ, total_cost = initial_fee + additional_charge_per_mile_fraction * 3.6 ∧ f = 1 / 9 :=
by
  sorry

end additional_charge_fraction_of_mile_l324_32483


namespace find_a_l324_32425

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x ^ 2 + a * Real.cos (Real.pi * x) else 2

theorem find_a (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → f 1 a = 2 → a = - 3 :=
by
  sorry

end find_a_l324_32425


namespace smallest_N_divisible_by_p_l324_32455

theorem smallest_N_divisible_by_p (p : ℕ) (hp : Nat.Prime p)
    (N1 : ℕ) (N2 : ℕ) :
  (∃ N1 N2, 
    (N1 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N1 % n = 1) ∧
    (N2 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N2 % n = n - 1)
  ) :=
sorry

end smallest_N_divisible_by_p_l324_32455


namespace sum_of_series_l324_32475

theorem sum_of_series (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_gt : a > b) :
  ∑' n, 1 / ( ((n - 1) * a + (n - 2) * b) * (n * a + (n - 1) * b)) = 1 / ((a + b) * b) :=
by
  sorry

end sum_of_series_l324_32475


namespace Margo_total_distance_walked_l324_32456

theorem Margo_total_distance_walked :
  ∀ (d : ℝ),
  (5 * (d / 5) + 3 * (d / 3) = 1) →
  (2 * d = 3.75) :=
by
  sorry

end Margo_total_distance_walked_l324_32456


namespace harmonic_mean_of_4_and_5040_is_8_closest_l324_32412

noncomputable def harmonicMean (a b : ℕ) : ℝ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_of_4_and_5040_is_8_closest :
  abs (harmonicMean 4 5040 - 8) < 1 :=
by
  -- The proof process would go here
  sorry

end harmonic_mean_of_4_and_5040_is_8_closest_l324_32412


namespace volume_tetrahedron_l324_32428

variables (AB AC AD : ℝ) (β γ D : ℝ)
open Real

/-- Prove that the volume of tetrahedron ABCD is equal to 
    (AB * AC * AD * sin β * sin γ * sin D) / 6,
    where β and γ are the plane angles at vertex A opposite to edges AB and AC, 
    and D is the dihedral angle at edge AD. 
-/
theorem volume_tetrahedron (h₁: β ≠ 0) (h₂: γ ≠ 0) (h₃: D ≠ 0):
  (AB * AC * AD * sin β * sin γ * sin D) / 6 =
    abs (AB * AC * AD * sin β * sin γ * sin D) / 6 :=
by sorry

end volume_tetrahedron_l324_32428


namespace no_nonzero_solution_l324_32446

theorem no_nonzero_solution (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
by 
  sorry

end no_nonzero_solution_l324_32446


namespace find_dividend_l324_32444

theorem find_dividend (R D Q V : ℤ) (hR : R = 5) (hD1 : D = 3 * Q) (hD2 : D = 3 * R + 3) : V = D * Q + R → V = 113 :=
by 
  sorry

end find_dividend_l324_32444


namespace sector_central_angle_l324_32450

theorem sector_central_angle (r θ : ℝ) 
  (h1 : 1 = (1 / 2) * 2 * r) 
  (h2 : 2 = θ * r) : θ = 2 := 
sorry

end sector_central_angle_l324_32450


namespace no_such_number_exists_l324_32470

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

/-- Define the number N as a sequence of digits a_n a_{n-1} ... a_0 -/
def number (a b : ℕ) (n : ℕ) : ℕ := a * 10^n + b

theorem no_such_number_exists :
  ¬ ∃ (N a_n b : ℕ) (n : ℕ), is_digit a_n ∧ a_n ≠ 0 ∧ b < 10^n ∧
    N = number a_n b n ∧
    b = N / 57 :=
sorry

end no_such_number_exists_l324_32470


namespace sum_of_last_digits_l324_32497

theorem sum_of_last_digits (num : Nat → Nat) (a b : Nat) :
  (∀ i, 1 ≤ i ∧ i < 2000 → (num i * 10 + num (i + 1)) % 17 = 0 ∨ (num i * 10 + num (i + 1)) % 23 = 0) →
  num 1 = 3 →
  (num 2000 = a ∨ num 2000 = b) →
  a = 2 →
  b = 5 →
  a + b = 7 :=
by 
  sorry

end sum_of_last_digits_l324_32497


namespace ratio_X_N_l324_32469

-- Given conditions as definitions
variables (P Q M N X : ℝ)
variables (hM : M = 0.40 * Q)
variables (hQ : Q = 0.30 * P)
variables (hN : N = 0.60 * P)
variables (hX : X = 0.25 * M)

-- Prove that X / N == 1 / 20
theorem ratio_X_N : X / N = 1 / 20 :=
by
  sorry

end ratio_X_N_l324_32469


namespace hexagon_perimeter_l324_32489

-- Defining the side lengths of the hexagon
def side_lengths : List ℕ := [7, 10, 8, 13, 11, 9]

-- Defining the perimeter calculation
def perimeter (sides : List ℕ) : ℕ := sides.sum

-- The main theorem stating the perimeter of the given hexagon
theorem hexagon_perimeter :
  perimeter side_lengths = 58 := by
  -- Skipping proof here
  sorry

end hexagon_perimeter_l324_32489


namespace solve_inequality_l324_32447

theorem solve_inequality :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
sorry

end solve_inequality_l324_32447


namespace teachers_like_at_least_one_l324_32436

theorem teachers_like_at_least_one (T C B N: ℕ) 
    (total_teachers : T + C + N = 90)  -- Total number of teachers plus neither equals 90
    (tea_teachers : T = 66)           -- Teachers who like tea is 66
    (coffee_teachers : C = 42)        -- Teachers who like coffee is 42
    (both_beverages : B = 3 * N)      -- Teachers who like both is three times neither
    : T + C - B = 81 :=               -- Teachers who like at least one beverage
by 
  sorry

end teachers_like_at_least_one_l324_32436


namespace geom_seq_m_value_l324_32401

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end geom_seq_m_value_l324_32401


namespace charlie_extra_fee_l324_32414

-- Conditions
def data_limit_week1 : ℕ := 2 -- in GB
def data_limit_week2 : ℕ := 3 -- in GB
def data_limit_week3 : ℕ := 2 -- in GB
def data_limit_week4 : ℕ := 1 -- in GB

def additional_fee_week1 : ℕ := 12 -- dollars per GB
def additional_fee_week2 : ℕ := 10 -- dollars per GB
def additional_fee_week3 : ℕ := 8 -- dollars per GB
def additional_fee_week4 : ℕ := 6 -- dollars per GB

def data_used_week1 : ℕ := 25 -- in 0.1 GB
def data_used_week2 : ℕ := 40 -- in 0.1 GB
def data_used_week3 : ℕ := 30 -- in 0.1 GB
def data_used_week4 : ℕ := 50 -- in 0.1 GB

-- Additional fee calculation
def extra_data_fee := 
  let extra_data_week1 := max (data_used_week1 - data_limit_week1 * 10) 0
  let extra_fee_week1 := extra_data_week1 * additional_fee_week1 / 10
  let extra_data_week2 := max (data_used_week2 - data_limit_week2 * 10) 0
  let extra_fee_week2 := extra_data_week2 * additional_fee_week2 / 10
  let extra_data_week3 := max (data_used_week3 - data_limit_week3 * 10) 0
  let extra_fee_week3 := extra_data_week3 * additional_fee_week3 / 10
  let extra_data_week4 := max (data_used_week4 - data_limit_week4 * 10) 0
  let extra_fee_week4 := extra_data_week4 * additional_fee_week4 / 10
  extra_fee_week1 + extra_fee_week2 + extra_fee_week3 + extra_fee_week4

-- The math proof problem
theorem charlie_extra_fee : extra_data_fee = 48 := sorry

end charlie_extra_fee_l324_32414


namespace original_number_divisible_l324_32498

theorem original_number_divisible (N M R : ℕ) (n : ℕ) (hN : N = 1000 * M + R)
  (hDiff : (M - R) % n = 0) (hn : n = 7 ∨ n = 11 ∨ n = 13) : N % n = 0 :=
by
  sorry

end original_number_divisible_l324_32498


namespace sqrt_combination_l324_32462

theorem sqrt_combination : 
    ∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 8) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 3))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 12))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 0.2))) :=
by
  sorry

end sqrt_combination_l324_32462


namespace evaluate_fractions_l324_32458

theorem evaluate_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := 
by
  sorry

end evaluate_fractions_l324_32458


namespace find_four_numbers_l324_32496

theorem find_four_numbers (a b c d : ℚ) :
  ((a + b = 1) ∧ (a + c = 5) ∧ 
   ((a + d = 8 ∧ b + c = 9) ∨ (a + d = 9 ∧ b + c = 8)) ) →
  ((a = -3/2 ∧ b = 5/2 ∧ c = 13/2 ∧ d = 19/2) ∨ 
   (a = -1 ∧ b = 2 ∧ c = 6 ∧ d = 10)) :=
  by
    sorry

end find_four_numbers_l324_32496


namespace even_function_l324_32443

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := (x + 2)^2 + (2 * x - 1)^2

theorem even_function : is_even_function f :=
by
  sorry

end even_function_l324_32443


namespace yi_successful_shots_l324_32432

-- Defining the basic conditions
variables {x y : ℕ} -- Number of successful shots made by Jia and Yi respectively

-- Each hit gains 20 points and each miss deducts 12 points.
-- Both person A (Jia) and person B (Yi) made 10 shots each.
def total_shots (x y : ℕ) : Prop := 
  (20 * x - 12 * (10 - x)) + (20 * y - 12 * (10 - y)) = 208 ∧ x + y = 14 ∧ x - y = 2

theorem yi_successful_shots (x y : ℕ) (h : total_shots x y) : y = 6 := 
  by sorry

end yi_successful_shots_l324_32432


namespace min_blocks_for_wall_l324_32408

noncomputable def min_blocks_needed (length height : ℕ) (block_sizes : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem min_blocks_for_wall :
  min_blocks_needed 120 8 [(1, 3), (1, 2), (1, 1)] = 404 := by
  sorry

end min_blocks_for_wall_l324_32408


namespace probability_at_least_one_girl_l324_32453

theorem probability_at_least_one_girl 
  (boys girls : ℕ) 
  (total : boys + girls = 7) 
  (combinations_total : ℕ := Nat.choose 7 2) 
  (combinations_boys : ℕ := Nat.choose 4 2) 
  (prob_no_girls : ℚ := combinations_boys / combinations_total) 
  (prob_at_least_one_girl : ℚ := 1 - prob_no_girls) :
  boys = 4 ∧ girls = 3 → prob_at_least_one_girl = 5 / 7 := 
by
  intro h
  cases h
  sorry

end probability_at_least_one_girl_l324_32453


namespace petya_time_comparison_l324_32420

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l324_32420


namespace distance_from_A_to_directrix_l324_32468

open Real

noncomputable def distance_from_point_to_directrix (p : ℝ) : ℝ :=
  1 + p / 2

theorem distance_from_A_to_directrix : 
  ∃ (p : ℝ), (sqrt 5)^2 = 2 * p ∧ distance_from_point_to_directrix p = 9 / 4 :=
by 
  sorry

end distance_from_A_to_directrix_l324_32468


namespace spherical_coordinates_convert_l324_32419

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l324_32419


namespace fabric_cost_equation_l324_32440

theorem fabric_cost_equation (x : ℝ) :
  (3 * x + 5 * (138 - x) = 540) :=
sorry

end fabric_cost_equation_l324_32440
