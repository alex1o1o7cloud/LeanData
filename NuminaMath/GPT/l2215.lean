import Mathlib

namespace incorrect_operation_in_list_l2215_221592

open Real

theorem incorrect_operation_in_list :
  ¬ (abs ((-2)^2) = -2) :=
by
  -- Proof will be added here
  sorry

end incorrect_operation_in_list_l2215_221592


namespace base_conversion_l2215_221517

theorem base_conversion (b : ℝ) (h : 2 * b^2 + 3 = 51) : b = 2 * Real.sqrt 6 :=
by
  sorry

end base_conversion_l2215_221517


namespace sequence_sum_l2215_221525

theorem sequence_sum :
  (3 + 13 + 23 + 33 + 43 + 53) + (5 + 15 + 25 + 35 + 45 + 55) = 348 := by
  sorry

end sequence_sum_l2215_221525


namespace hens_to_roosters_multiplier_l2215_221513

def totalChickens : ℕ := 75
def numHens : ℕ := 67

-- Given the total number of chickens and a certain relationship
theorem hens_to_roosters_multiplier
  (numRoosters : ℕ) (multiplier : ℕ)
  (h1 : totalChickens = numHens + numRoosters)
  (h2 : numHens = multiplier * numRoosters - 5) :
  multiplier = 9 :=
by sorry

end hens_to_roosters_multiplier_l2215_221513


namespace find_k_l2215_221532

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Vectors expressions
def k_a_add_b (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def a_sub_3b : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)

-- Condition of collinearity
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 = 0 ∨ v2.1 = 0 ∨ v1.1 * v2.2 = v1.2 * v2.1)

-- Statement to prove
theorem find_k :
  collinear (k_a_add_b (-1/3)) a_sub_3b :=
sorry

end find_k_l2215_221532


namespace parabola_vertex_and_point_l2215_221546

theorem parabola_vertex_and_point (a b c : ℝ) (h_vertex : (1, -2) = (1, a * 1^2 + b * 1 + c))
  (h_point : (3, 7) = (3, a * 3^2 + b * 3 + c)) : a = 3 := 
by {
  sorry
}

end parabola_vertex_and_point_l2215_221546


namespace emails_in_inbox_l2215_221529

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l2215_221529


namespace tangent_line_b_value_l2215_221576

noncomputable def b_value : ℝ := Real.log 2 - 1

theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x > 0, (fun x => Real.log x) x = (1/2) * x + b → ∃ c : ℝ, c = b) → b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_b_value_l2215_221576


namespace simple_interest_rate_l2215_221554

theorem simple_interest_rate (P T SI : ℝ) (hP : P = 10000) (hT : T = 1) (hSI : SI = 400) :
    (SI = P * 0.04 * T) := by
  rw [hP, hT, hSI]
  sorry

end simple_interest_rate_l2215_221554


namespace average_first_21_multiples_of_8_l2215_221552

noncomputable def average_of_multiples (n : ℕ) (a : ℕ) : ℕ :=
  let sum := (n * (a + a * n)) / 2
  sum / n

theorem average_first_21_multiples_of_8 : average_of_multiples 21 8 = 88 :=
by
  sorry

end average_first_21_multiples_of_8_l2215_221552


namespace abc_zero_l2215_221570

theorem abc_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) :
  a * b * c = 0 := 
sorry

end abc_zero_l2215_221570


namespace part1_part2_l2215_221563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part1 (a : ℝ) (x : ℝ) (hx1 : 1 ≤ x) (hx2 : x ≤ Real.exp 1) :
  a = 1 →
  (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 + (Real.exp 1)^2 / 2) ∧ (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 / 2) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part2 (a : ℝ) :
  (-1/2 ≤ a ∧ a ≤ 1/2) ↔
  ∀ x, 1 < x → g a x < 0 :=
sorry

end part1_part2_l2215_221563


namespace range_of_sum_of_products_l2215_221551

theorem range_of_sum_of_products (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)
  (h_sum : a + b + c = (Real.sqrt 3) / 2) :
  0 < (a * b + b * c + c * a) ∧ (a * b + b * c + c * a) ≤ 1 / 4 :=
by
  sorry

end range_of_sum_of_products_l2215_221551


namespace clock_90_degree_angle_times_l2215_221516

noncomputable def first_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 90

noncomputable def second_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 270

theorem clock_90_degree_angle_times :
  ∃ t₁ t₂ : ℝ,
  first_time_90_degree_angle t₁ ∧ 
  second_time_90_degree_angle t₂ ∧ 
  t₁ = (180 / 11 : ℝ) ∧ 
  t₂ = (540 / 11 : ℝ) :=
by
  sorry

end clock_90_degree_angle_times_l2215_221516


namespace intersection_A_B_union_A_B_diff_A_B_diff_B_A_l2215_221583

def A : Set Real := {x | -1 < x ∧ x < 2}
def B : Set Real := {x | 0 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x < 2} :=
sorry

theorem union_A_B :
  A ∪ B = {x | -1 < x ∧ x < 4} :=
sorry

theorem diff_A_B :
  A \ B = {x | -1 < x ∧ x ≤ 0} :=
sorry

theorem diff_B_A :
  B \ A = {x | 2 ≤ x ∧ x < 4} :=
sorry

end intersection_A_B_union_A_B_diff_A_B_diff_B_A_l2215_221583


namespace suggested_bacon_students_l2215_221584

-- Definitions based on the given conditions
def students_mashed_potatoes : ℕ := 330
def students_tomatoes : ℕ := 76
def difference_bacon_mashed_potatoes : ℕ := 61

-- Lean 4 statement to prove the correct answer
theorem suggested_bacon_students : ∃ (B : ℕ), students_mashed_potatoes = B + difference_bacon_mashed_potatoes ∧ B = 269 := 
by
  sorry

end suggested_bacon_students_l2215_221584


namespace subset_implies_a_geq_4_l2215_221574

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + 3 ≤ 0}

theorem subset_implies_a_geq_4 (a : ℝ) :
  A ⊆ B a → a ≥ 4 := sorry

end subset_implies_a_geq_4_l2215_221574


namespace small_beaker_salt_fraction_l2215_221573

theorem small_beaker_salt_fraction
  (S L : ℝ) 
  (h1 : L = 5 * S)
  (h2 : L * (1 / 5) = S)
  (h3 : L * 0.3 = S * 1.5)
  : (S * 0.5) / S = 0.5 :=
by 
  sorry

end small_beaker_salt_fraction_l2215_221573


namespace cannot_form_figureB_l2215_221575

-- Define the pieces as terms
inductive Piece
| square : Piece
| rectangle : Π (h w : ℕ), Piece   -- h: height, w: width

-- Define the available pieces in a list (assuming these are predefined somewhere)
def pieces : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

-- Define the figures that can be formed
def figureA : List Piece := [Piece.square, Piece.square, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

def figureC : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, 
                             Piece.square, Piece.square]

def figureD : List Piece := [Piece.rectangle 2 2, Piece.square, Piece.square, Piece.square,
                              Piece.square]

def figureE : List Piece := [Piece.rectangle 3 1, Piece.square, Piece.square, Piece.square]

-- Define the figure B that we need to prove cannot be formed
def figureB : List Piece := [Piece.rectangle 5 1, Piece.square, Piece.square, Piece.square,
                              Piece.square]

theorem cannot_form_figureB :
  ¬(∃ arrangement : List Piece, arrangement ⊆ pieces ∧ arrangement = figureB) :=
sorry

end cannot_form_figureB_l2215_221575


namespace ram_pairs_sold_correct_l2215_221514

-- Define the costs
def graphics_card_cost := 600
def hard_drive_cost := 80
def cpu_cost := 200
def ram_pair_cost := 60

-- Define the number of items sold
def graphics_cards_sold := 10
def hard_drives_sold := 14
def cpus_sold := 8
def total_earnings := 8960

-- Calculate earnings from individual items
def earnings_graphics_cards := graphics_cards_sold * graphics_card_cost
def earnings_hard_drives := hard_drives_sold * hard_drive_cost
def earnings_cpus := cpus_sold * cpu_cost

-- Calculate total earnings from graphics cards, hard drives, and CPUs
def earnings_other_items := earnings_graphics_cards + earnings_hard_drives + earnings_cpus

-- Calculate earnings from RAM
def earnings_from_ram := total_earnings - earnings_other_items

-- Calculate number of RAM pairs sold
def ram_pairs_sold := earnings_from_ram / ram_pair_cost

-- The theorem to be proven
theorem ram_pairs_sold_correct : ram_pairs_sold = 4 :=
by
  sorry

end ram_pairs_sold_correct_l2215_221514


namespace largest_n_for_factorable_polynomial_l2215_221527

theorem largest_n_for_factorable_polynomial :
  ∃ (n : ℤ), (∀ A B : ℤ, 7 * A * B = 56 → n ≤ 7 * B + A) ∧ n = 393 :=
by {
  sorry
}

end largest_n_for_factorable_polynomial_l2215_221527


namespace root_zero_implies_m_eq_6_l2215_221599

theorem root_zero_implies_m_eq_6 (m : ℝ) (h : ∃ x : ℝ, 3 * (x^2) + m * x + m - 6 = 0) : m = 6 := 
by sorry

end root_zero_implies_m_eq_6_l2215_221599


namespace central_cell_value_l2215_221566

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l2215_221566


namespace route_length_is_140_l2215_221536

-- Conditions of the problem
variable (D : ℝ)  -- Length of the route
variable (Vx Vy t : ℝ)  -- Speeds of Train X and Train Y, and time to meet

-- Given conditions
axiom train_X_trip_time : D / Vx = 4
axiom train_Y_trip_time : D / Vy = 3
axiom train_X_distance_when_meet : Vx * t = 60
axiom total_distance_covered_on_meeting : Vx * t + Vy * t = D

-- Goal: Prove that the length of the route is 140 kilometers
theorem route_length_is_140 : D = 140 := by
  -- Proof omitted
  sorry

end route_length_is_140_l2215_221536


namespace range_of_a_l2215_221544

open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7 * x - 18 < 0}

theorem range_of_a (a : ℝ) : A a ⊆ B → (-2 : ℝ) ≤ a ∧ a ≤ 9 :=
by sorry

end range_of_a_l2215_221544


namespace complement_intersection_l2215_221586

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3, 4}
def A_complement : Set ℕ := U \ A

theorem complement_intersection :
  (A_complement ∩ B) = {2, 4} :=
by 
  sorry

end complement_intersection_l2215_221586


namespace inequality_proof_l2215_221538

theorem inequality_proof
  (a b c d : ℝ)
  (hpos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (hcond: (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2 * a + b + c) * (2 * b + c + d) * (2 * c + d + a) * (2 * d + a + b) * (a * b * c * d) ^ 2 ≤ 1 / 16 := 
by
  sorry

end inequality_proof_l2215_221538


namespace proof_problem_l2215_221507

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ f (-3) = -2

theorem proof_problem (f : ℝ → ℝ) (h : given_function f) : f 3 + f 0 = -2 :=
by sorry

end proof_problem_l2215_221507


namespace base_of_triangle_is_24_l2215_221526

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l2215_221526


namespace probability_of_dice_outcome_l2215_221594

theorem probability_of_dice_outcome : 
  let p_one_digit := 3 / 4
  let p_two_digit := 1 / 4
  let comb := Nat.choose 5 3
  (comb * (p_one_digit^3) * (p_two_digit^2)) = 135 / 512 := 
by
  sorry

end probability_of_dice_outcome_l2215_221594


namespace inequality_solution_set_l2215_221569

noncomputable def solution_set (a b : ℝ) := {x : ℝ | 2 < x ∧ x < 3}

theorem inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (ax^2 + 5 * x + b > 0)) →
  (∀ x : ℝ, (-6) * x^2 - 5 * x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by
  sorry

end inequality_solution_set_l2215_221569


namespace retailer_mark_up_l2215_221503

theorem retailer_mark_up (R C M S : ℝ) 
  (hC : C = 0.7 * R)
  (hS : S = C / 0.7)
  (hSm : S = 0.9 * M) : 
  M = 1.111 * R :=
by 
  sorry

end retailer_mark_up_l2215_221503


namespace min_expression_value_l2215_221589

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end min_expression_value_l2215_221589


namespace arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l2215_221598

-- Definitions based on conditions in A)
def students : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def A : Char := 'A'
def B : Char := 'B'
def C : Char := 'C'
def D : Char := 'D'
def E : Char := 'E'
def F : Char := 'F'
def G : Char := 'G'

-- Holistic theorem statements for each question derived from the correct answers in B)
theorem arrangement_A_and_B_adjacent :
  ∃ (n : ℕ), n = 1440 := sorry

theorem arrangement_A_B_and_C_adjacent :
  ∃ (n : ℕ), n = 720 := sorry

theorem arrangement_A_and_B_adjacent_C_not_ends :
  ∃ (n : ℕ), n = 960 := sorry

theorem arrangement_ABC_and_DEFG_units :
  ∃ (n : ℕ), n = 288 := sorry

end arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l2215_221598


namespace ratio_equality_proof_l2215_221543

theorem ratio_equality_proof
  (m n k a b c x y z : ℝ)
  (h : x / (m * (n * b + k * c - m * a)) = y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = k / (z * (a * x + b * y - c * z)) :=
by
  sorry

end ratio_equality_proof_l2215_221543


namespace parametric_equation_of_line_passing_through_M_l2215_221533

theorem parametric_equation_of_line_passing_through_M (
  t : ℝ
) : 
    ∃ x y : ℝ, 
      x = 1 + (t * (Real.cos (Real.pi / 3))) ∧ 
      y = 5 + (t * (Real.sin (Real.pi / 3))) ∧ 
      x = 1 + (1/2) * t ∧ 
      y = 5 + (Real.sqrt 3 / 2) * t := 
by
  sorry

end parametric_equation_of_line_passing_through_M_l2215_221533


namespace find_distance_of_post_office_from_village_l2215_221545

-- Conditions
def rate_to_post_office : ℝ := 12.5
def rate_back_village : ℝ := 2
def total_time : ℝ := 5.8

-- Statement of the theorem
theorem find_distance_of_post_office_from_village (D : ℝ) 
  (travel_time_to : D / rate_to_post_office = D / 12.5) 
  (travel_time_back : D / rate_back_village = D / 2)
  (journey_time_total : D / 12.5 + D / 2 = total_time) : 
  D = 10 := 
sorry

end find_distance_of_post_office_from_village_l2215_221545


namespace evaluate_expression_l2215_221556

theorem evaluate_expression : (-2)^3 - (-3)^2 = -17 :=
by sorry

end evaluate_expression_l2215_221556


namespace workers_new_daily_wage_l2215_221522

def wage_before : ℝ := 25
def increase_percentage : ℝ := 0.40

theorem workers_new_daily_wage : wage_before * (1 + increase_percentage) = 35 :=
by
  -- sorry will be replaced by the actual proof steps
  sorry

end workers_new_daily_wage_l2215_221522


namespace sum_of_intersections_l2215_221518

theorem sum_of_intersections :
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (∀ x y : ℝ, y = (x - 2)^2 ↔ x + 1 = (y - 2)^2) ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 20) :=
sorry

end sum_of_intersections_l2215_221518


namespace find_rate_of_interest_l2215_221580

/-- At what rate percent on simple interest will Rs. 25,000 amount to Rs. 34,500 in 5 years? 
    Given Principal (P) = Rs. 25,000, Amount (A) = Rs. 34,500, Time (T) = 5 years. 
    We need to find the Rate (R). -/
def principal : ℝ := 25000
def amount : ℝ := 34500
def time : ℝ := 5

theorem find_rate_of_interest (P A T : ℝ) : 
  P = principal → 
  A = amount → 
  T = time → 
  ∃ R : ℝ, R = 7.6 :=
by
  intros hP hA hT
  -- proof goes here
  sorry

end find_rate_of_interest_l2215_221580


namespace kelly_chris_boxes_ratio_l2215_221524

theorem kelly_chris_boxes_ratio (X : ℝ) (h : X > 0) :
  (0.4 * X) / (0.6 * X) = 2 / 3 :=
by sorry

end kelly_chris_boxes_ratio_l2215_221524


namespace combined_total_cost_is_correct_l2215_221550

-- Define the number and costs of balloons for each person
def Fred_yellow_count : ℕ := 5
def Fred_red_count : ℕ := 3
def Fred_yellow_cost_per : ℕ := 3
def Fred_red_cost_per : ℕ := 4

def Sam_yellow_count : ℕ := 6
def Sam_red_count : ℕ := 4
def Sam_yellow_cost_per : ℕ := 4
def Sam_red_cost_per : ℕ := 5

def Mary_yellow_count : ℕ := 7
def Mary_red_count : ℕ := 5
def Mary_yellow_cost_per : ℕ := 5
def Mary_red_cost_per : ℕ := 6

def Susan_yellow_count : ℕ := 4
def Susan_red_count : ℕ := 6
def Susan_yellow_cost_per : ℕ := 6
def Susan_red_cost_per : ℕ := 7

def Tom_yellow_count : ℕ := 10
def Tom_red_count : ℕ := 8
def Tom_yellow_cost_per : ℕ := 2
def Tom_red_cost_per : ℕ := 3

-- Formula to calculate total cost for a given person
def total_cost (yellow_count red_count yellow_cost_per red_cost_per : ℕ) : ℕ :=
  (yellow_count * yellow_cost_per) + (red_count * red_cost_per)

-- Total costs for each person
def Fred_total_cost := total_cost Fred_yellow_count Fred_red_count Fred_yellow_cost_per Fred_red_cost_per
def Sam_total_cost := total_cost Sam_yellow_count Sam_red_count Sam_yellow_cost_per Sam_red_cost_per
def Mary_total_cost := total_cost Mary_yellow_count Mary_red_count Mary_yellow_cost_per Mary_red_cost_per
def Susan_total_cost := total_cost Susan_yellow_count Susan_red_count Susan_yellow_cost_per Susan_red_cost_per
def Tom_total_cost := total_cost Tom_yellow_count Tom_red_count Tom_yellow_cost_per Tom_red_cost_per

-- Combined total cost
def combined_total_cost : ℕ :=
  Fred_total_cost + Sam_total_cost + Mary_total_cost + Susan_total_cost + Tom_total_cost

-- Lean statement to prove
theorem combined_total_cost_is_correct : combined_total_cost = 246 :=
by
  dsimp [combined_total_cost, Fred_total_cost, Sam_total_cost, Mary_total_cost, Susan_total_cost, Tom_total_cost, total_cost]
  sorry

end combined_total_cost_is_correct_l2215_221550


namespace fraction_comparison_l2215_221535

theorem fraction_comparison (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) :=
by
  sorry

end fraction_comparison_l2215_221535


namespace number_of_square_free_odd_integers_between_1_and_200_l2215_221555

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end number_of_square_free_odd_integers_between_1_and_200_l2215_221555


namespace emma_average_speed_last_segment_l2215_221549

open Real

theorem emma_average_speed_last_segment :
  ∀ (d1 d2 d3 : ℝ) (t1 t2 t3 : ℝ),
    d1 + d2 + d3 = 120 →
    t1 + t2 + t3 = 2 →
    t1 = 2 / 3 → t2 = 2 / 3 → 
    t1 = d1 / 50 → t2 = d2 / 55 → 
    ∃ x : ℝ, t3 = d3 / x ∧ x = 75 := 
by
  intros d1 d2 d3 t1 t2 t3 h1 h2 ht1 ht2 hs1 hs2
  use 75 / (2 / 3)
  -- skipped proof for simplicity
  sorry

end emma_average_speed_last_segment_l2215_221549


namespace num_convex_quadrilateral_angles_arith_prog_l2215_221559

theorem num_convex_quadrilateral_angles_arith_prog :
  ∃ (S : Finset (Finset ℤ)), S.card = 29 ∧
    ∀ {a b c d : ℤ}, {a, b, c, d} ∈ S →
      a + b + c + d = 360 ∧
      a < b ∧ b < c ∧ c < d ∧
      ∃ (m d_diff : ℤ), 
        m - d_diff = a ∧
        m = b ∧
        m + d_diff = c ∧
        m + 2 * d_diff = d ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end num_convex_quadrilateral_angles_arith_prog_l2215_221559


namespace RachelFurnitureAssemblyTime_l2215_221521

/-- Rachel bought seven new chairs and three new tables for her house.
    She spent four minutes on each piece of furniture putting it together.
    Prove that it took her 40 minutes to finish putting together all the furniture. -/
theorem RachelFurnitureAssemblyTime :
  let chairs := 7
  let tables := 3
  let time_per_piece := 4
  let total_time := (chairs + tables) * time_per_piece
  total_time = 40 := by
    sorry

end RachelFurnitureAssemblyTime_l2215_221521


namespace perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l2215_221505

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Problem statement for part (I)
theorem perp_bisector_eq : ∃ (k m: ℝ), 3 * k - 4 * m - 23 = 0 :=
sorry

-- Problem statement for part (II)
theorem parallel_line_eq : ∃ (k m: ℝ), 4 * k + 3 * m + 1 = 0 :=
sorry

-- Problem statement for part (III)
theorem reflected_ray_eq : ∃ (k m: ℝ), 11 * k + 27 * m + 74 = 0 :=
sorry

end perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l2215_221505


namespace negation_universal_prop_l2215_221557

theorem negation_universal_prop:
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
  sorry

end negation_universal_prop_l2215_221557


namespace map_line_segments_l2215_221578

def point : Type := ℝ × ℝ

def transformation (f : point → point) (p q : point) : Prop := f p = q

def counterclockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

def clockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

theorem map_line_segments :
  (transformation counterclockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation counterclockwise_rotation_180 (2, -5) (-2, 5)) ∨
  (transformation clockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation clockwise_rotation_180 (2, -5) (-2, 5)) :=
by
  sorry

end map_line_segments_l2215_221578


namespace two_marbles_different_colors_probability_l2215_221561

-- Definitions
def red_marbles : Nat := 3
def green_marbles : Nat := 4
def white_marbles : Nat := 5
def blue_marbles : Nat := 3
def total_marbles : Nat := red_marbles + green_marbles + white_marbles + blue_marbles

-- Combinations of different colored marbles
def red_green : Nat := red_marbles * green_marbles
def red_white : Nat := red_marbles * white_marbles
def red_blue : Nat := red_marbles * blue_marbles
def green_white : Nat := green_marbles * white_marbles
def green_blue : Nat := green_marbles * blue_marbles
def white_blue : Nat := white_marbles * blue_marbles

-- Total favorable outcomes
def total_favorable : Nat := red_green + red_white + red_blue + green_white + green_blue + white_blue

-- Total outcomes when drawing 2 marbles from the jar
def total_outcomes : Nat := Nat.choose total_marbles 2

-- Probability calculation
noncomputable def probability_different_colors : Rat := total_favorable / total_outcomes

-- Proof that the probability is 83/105
theorem two_marbles_different_colors_probability :
  probability_different_colors = 83 / 105 := by
  sorry

end two_marbles_different_colors_probability_l2215_221561


namespace son_l2215_221565

-- Define the context of the problem with conditions
variables (S M : ℕ)

-- Condition 1: The man is 28 years older than his son
def condition1 : Prop := M = S + 28

-- Condition 2: In two years, the man's age will be twice the son's age
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The final statement to prove the son's present age
theorem son's_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 26 :=
by
  sorry

end son_l2215_221565


namespace evaluate_expression_l2215_221579

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a * b^2 = 59 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l2215_221579


namespace min_value_of_f_l2215_221508

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 3)

theorem min_value_of_f (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx1_distinct : x1 ≠ x2) (hx2_pos : 0 < x2)
  (h_f_eq : f x1 = f x2) : (1 / x1 + 9 / x2) = 2 / 3 :=
by
  sorry

end min_value_of_f_l2215_221508


namespace div_by_5_factor_l2215_221591

theorem div_by_5_factor {x y z : ℤ} (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * 5 * (y - z) * (z - x) * (x - y) :=
sorry

end div_by_5_factor_l2215_221591


namespace makeup_set_cost_l2215_221593

theorem makeup_set_cost (initial : ℕ) (gift : ℕ) (needed : ℕ) (total_cost : ℕ) :
  initial = 35 → gift = 20 → needed = 10 → total_cost = initial + gift + needed → total_cost = 65 :=
by
  intros h_init h_gift h_needed h_cost
  sorry

end makeup_set_cost_l2215_221593


namespace surface_area_sphere_l2215_221548

-- Definitions based on conditions
def SA : ℝ := 3
def SB : ℝ := 4
def SC : ℝ := 5
def vertices_perpendicular : Prop := ∀ (a b c : ℝ), (a = SA ∧ b = SB ∧ c = SC) → (a * b * c = SA * SB * SC)

-- Definition of the theorem based on problem and correct answer
theorem surface_area_sphere (h1 : vertices_perpendicular) : 
  4 * Real.pi * ((Real.sqrt (SA^2 + SB^2 + SC^2)) / 2)^2 = 50 * Real.pi :=
by
  -- skip the proof
  sorry

end surface_area_sphere_l2215_221548


namespace missy_total_watching_time_l2215_221504

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l2215_221504


namespace total_marks_of_all_candidates_l2215_221500

theorem total_marks_of_all_candidates 
  (average_marks : ℕ) 
  (num_candidates : ℕ) 
  (average : average_marks = 35) 
  (candidates : num_candidates = 120) : 
  average_marks * num_candidates = 4200 :=
by
  -- The proof will be written here
  sorry

end total_marks_of_all_candidates_l2215_221500


namespace solution_to_equation1_solution_to_equation2_l2215_221530

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := 3 * x^3 + 4 = -20

-- State the theorems with the correct answers
theorem solution_to_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = -3) :=
by
  sorry

theorem solution_to_equation2 (x : ℝ) : equation2 x ↔ (x = -2) :=
by
  sorry

end solution_to_equation1_solution_to_equation2_l2215_221530


namespace min_value_range_of_a_l2215_221582

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp (-2 * x) + a * (2 * x + 1) * Real.exp (-x) + x^2 + x

theorem min_value_range_of_a (a : ℝ) (h : a > 0)
  (min_f : ∃ x : ℝ, f a x = Real.log a ^ 2 + 3 * Real.log a + 2) :
  a ∈ Set.Ici (Real.exp (-3 / 2)) :=
by
  sorry

end min_value_range_of_a_l2215_221582


namespace solve_cryptarithm_l2215_221510

def cryptarithm_puzzle (K I C : ℕ) : Prop :=
  K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K + I + C < 30 ∧  -- Ensuring each is a single digit (0-9)
  (10 * K + I + C) + (10 * K + 10 * C + I) = 100 + 10 * I + 10 * C + K

theorem solve_cryptarithm :
  ∃ K I C, cryptarithm_puzzle K I C ∧ K = 4 ∧ I = 9 ∧ C = 5 :=
by
  use 4, 9, 5
  sorry 

end solve_cryptarithm_l2215_221510


namespace option_c_correct_l2215_221585

-- Statement of the problem: Prove that (x-3)^2 = x^2 - 6x + 9

theorem option_c_correct (x : ℝ) : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=
by
  sorry

end option_c_correct_l2215_221585


namespace books_count_is_8_l2215_221502

theorem books_count_is_8
  (k a p_k p_a : ℕ)
  (h1 : k = a + 6)
  (h2 : k * p_k = 1056)
  (h3 : a * p_a = 56)
  (h4 : p_k > p_a + 100) :
  k = 8 := 
sorry

end books_count_is_8_l2215_221502


namespace increase_factor_l2215_221520

noncomputable def old_plates : ℕ := 26 * 10^3
noncomputable def new_plates : ℕ := 26^4 * 10^4
theorem increase_factor : (new_plates / old_plates) = 175760 := by
  sorry

end increase_factor_l2215_221520


namespace expression_calculates_to_l2215_221588

noncomputable def mixed_number : ℚ := 3 + 3 / 4

noncomputable def decimal_to_fraction : ℚ := 2 / 10

noncomputable def given_expression : ℚ := ((mixed_number * decimal_to_fraction) / 135) * 5.4

theorem expression_calculates_to : given_expression = 0.03 := by
  sorry

end expression_calculates_to_l2215_221588


namespace sum_reciprocal_l2215_221531

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l2215_221531


namespace union_eq_l2215_221523

-- Define the sets M and N
def M : Finset ℕ := {0, 3}
def N : Finset ℕ := {1, 2, 3}

-- Define the proof statement
theorem union_eq : M ∪ N = {0, 1, 2, 3} := 
by
  sorry

end union_eq_l2215_221523


namespace total_bees_count_l2215_221558

-- Definitions
def initial_bees : ℕ := 16
def additional_bees : ℕ := 7

-- Problem statement to prove
theorem total_bees_count : initial_bees + additional_bees = 23 := by
  -- The proof will be given here
  sorry

end total_bees_count_l2215_221558


namespace findMonthlyIncome_l2215_221560

-- Variables and conditions
variable (I : ℝ) -- Raja's monthly income
variable (saving : ℝ) (r1 r2 r3 r4 r5 : ℝ) -- savings and monthly percentages

-- Conditions
def condition1 : r1 = 0.45 := by sorry
def condition2 : r2 = 0.12 := by sorry
def condition3 : r3 = 0.08 := by sorry
def condition4 : r4 = 0.15 := by sorry
def condition5 : r5 = 0.10 := by sorry
def conditionSaving : saving = 5000 := by sorry

-- Define the main equation
def mainEquation (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) : Prop :=
  (r1 * I) + (r2 * I) + (r3 * I) + (r4 * I) + (r5 * I) + saving = I

-- Main theorem to prove
theorem findMonthlyIncome (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) 
  (h1 : r1 = 0.45) (h2 : r2 = 0.12) (h3 : r3 = 0.08) (h4 : r4 = 0.15) (h5 : r5 = 0.10) (hSaving : saving = 5000) :
  mainEquation I r1 r2 r3 r4 r5 saving → I = 50000 :=
  by sorry

end findMonthlyIncome_l2215_221560


namespace grid_path_theorem_l2215_221512

open Nat

variables (m n : ℕ)
variables (A B C : ℕ)

def conditions (m n : ℕ) : Prop := m ≥ 4 ∧ n ≥ 4

noncomputable def grid_path_problem (m n A B C : ℕ) : Prop :=
  conditions m n ∧
  ((m - 1) * (n - 1) = A + (B + C)) ∧
  A = B - C + m + n - 1

theorem grid_path_theorem (m n A B C : ℕ) (h : grid_path_problem m n A B C) : 
  A = B - C + m + n - 1 :=
  sorry

end grid_path_theorem_l2215_221512


namespace alpha_plus_beta_l2215_221590

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π)
variable (hβ : 0 < β ∧ β < π)
variable (h1 : Real.sin (α - β) = 3 / 4)
variable (h2 : Real.tan α / Real.tan β = -5)

theorem alpha_plus_beta (h3 : α + β = 5 * π / 6) : α + β = 5 * π / 6 :=
by
  sorry

end alpha_plus_beta_l2215_221590


namespace fraction_of_white_surface_area_l2215_221581

def larger_cube_edge : ℕ := 4
def number_of_smaller_cubes : ℕ := 64
def number_of_white_cubes : ℕ := 8
def number_of_red_cubes : ℕ := 56
def total_surface_area : ℕ := 6 * (larger_cube_edge * larger_cube_edge)
def minimized_white_surface_area : ℕ := 7

theorem fraction_of_white_surface_area :
  minimized_white_surface_area % total_surface_area = 7 % 96 :=
by
  sorry

end fraction_of_white_surface_area_l2215_221581


namespace find_angle_CDE_l2215_221595

-- Definition of the angles and their properties
variables {A B C D E : Type}

-- Hypotheses
def angleA_is_right (angleA: ℝ) : Prop := angleA = 90
def angleB_is_right (angleB: ℝ) : Prop := angleB = 90
def angleC_is_right (angleC: ℝ) : Prop := angleC = 90
def angleAEB_value (angleAEB : ℝ) : Prop := angleAEB = 40
def angleBED_eq_angleBDE (angleBED angleBDE : ℝ) : Prop := angleBED = angleBDE

-- The theorem to be proved
theorem find_angle_CDE 
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ) (angleAEB : ℝ) (angleBED angleBDE : ℝ) (angleCDE : ℝ) :
  angleA_is_right angleA → 
  angleB_is_right angleB → 
  angleC_is_right angleC → 
  angleAEB_value angleAEB → 
  angleBED_eq_angleBDE angleBED angleBDE →
  angleBED = 45 →
  angleCDE = 95 :=
by
  intros
  sorry


end find_angle_CDE_l2215_221595


namespace waiter_earned_total_tips_l2215_221564

def tips (c1 c2 c3 c4 c5 : ℝ) := c1 + c2 + c3 + c4 + c5

theorem waiter_earned_total_tips :
  tips 1.50 2.75 3.25 4.00 5.00 = 16.50 := 
by 
  sorry

end waiter_earned_total_tips_l2215_221564


namespace trigonometric_identity_l2215_221553

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = 10 / 3 :=
sorry

end trigonometric_identity_l2215_221553


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l2215_221572

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l2215_221572


namespace discriminant_of_quadratic_eq_l2215_221540

/-- The discriminant of a quadratic equation -/
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_eq : discriminant 1 3 (-1) = 13 := by
  sorry

end discriminant_of_quadratic_eq_l2215_221540


namespace degrees_of_remainder_is_correct_l2215_221501

noncomputable def degrees_of_remainder (P D : Polynomial ℤ) : Finset ℕ :=
  if D.degree = 3 then {0, 1, 2} else ∅

theorem degrees_of_remainder_is_correct
(P : Polynomial ℤ) :
  degrees_of_remainder P (Polynomial.C 3 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = {0, 1, 2} :=
by
  -- Proof omitted
  sorry

end degrees_of_remainder_is_correct_l2215_221501


namespace dice_probability_same_face_l2215_221528

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l2215_221528


namespace truth_values_of_p_and_q_l2215_221541

theorem truth_values_of_p_and_q (p q : Prop) (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬p) : ¬p ∧ q :=
by
  sorry

end truth_values_of_p_and_q_l2215_221541


namespace inverse_g_of_neg_92_l2215_221539

noncomputable def g (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 1

theorem inverse_g_of_neg_92 : g (-3) = -92 :=
by 
-- This would be the proof but we are skipping it as requested
sorry

end inverse_g_of_neg_92_l2215_221539


namespace number_of_arrangements_BANANA_l2215_221587

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l2215_221587


namespace sum_of_three_numbers_eq_zero_l2215_221515

theorem sum_of_three_numbers_eq_zero (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : (a + b + c) / 3 = a + 20) (h3 : (a + b + c) / 3 = c - 10) (h4 : b = 10) : 
  a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l2215_221515


namespace least_positive_integer_y_l2215_221571

theorem least_positive_integer_y (x k y: ℤ) (h1: 24 * x + k * y = 4) (h2: ∃ x: ℤ, ∃ y: ℤ, 24 * x + k * y = 4) : y = 4 :=
sorry

end least_positive_integer_y_l2215_221571


namespace cookies_in_box_l2215_221562

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l2215_221562


namespace radius_ratio_l2215_221519

noncomputable def volume_large_sphere : ℝ := 432 * Real.pi

noncomputable def volume_small_sphere : ℝ := 0.08 * volume_large_sphere

noncomputable def radius_large_sphere : ℝ :=
  (3 * volume_large_sphere / (4 * Real.pi)) ^ (1 / 3)

noncomputable def radius_small_sphere : ℝ :=
  (3 * volume_small_sphere / (4 * Real.pi)) ^ (1 / 3)

theorem radius_ratio (V_L V_s : ℝ) (hL : V_L = 432 * Real.pi) (hS : V_s = 0.08 * V_L) :
  (radius_small_sphere / radius_large_sphere) = (2/5)^(1/3) :=
by
  sorry

end radius_ratio_l2215_221519


namespace num_pos_three_digit_div_by_seven_l2215_221577

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l2215_221577


namespace harper_water_intake_l2215_221567

theorem harper_water_intake
  (cases_cost : ℕ := 12)
  (cases_count : ℕ := 24)
  (total_spent : ℕ)
  (days : ℕ)
  (total_days_spent : ℕ := 240)
  (total_money_spent: ℕ := 60)
  (total_water: ℕ := 5 * 24)
  (water_per_day : ℝ := 0.5):
  total_spent = total_money_spent ->
  days = total_days_spent ->
  water_per_day = (total_water : ℝ) / total_days_spent :=
by
  sorry

end harper_water_intake_l2215_221567


namespace intersection_A_B_l2215_221506
open Set

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by sorry

end intersection_A_B_l2215_221506


namespace final_exam_mean_score_l2215_221597

theorem final_exam_mean_score (μ σ : ℝ) 
  (h1 : 55 = μ - 1.5 * σ)
  (h2 : 75 = μ - 2 * σ)
  (h3 : 85 = μ + 1.5 * σ)
  (h4 : 100 = μ + 3.5 * σ) :
  μ = 115 :=
by
  sorry

end final_exam_mean_score_l2215_221597


namespace Winnie_the_Pooh_stationary_escalator_steps_l2215_221509

theorem Winnie_the_Pooh_stationary_escalator_steps
  (u v L : ℝ)
  (cond1 : L * u / (u + v) = 55)
  (cond2 : L * u / (u - v) = 1155) :
  L = 105 := by
  sorry

end Winnie_the_Pooh_stationary_escalator_steps_l2215_221509


namespace cost_of_600_pages_l2215_221596

def cost_per_5_pages := 10 -- 10 cents for 5 pages
def pages_to_copy := 600
def expected_cost := 12 * 100 -- 12 dollars in cents

theorem cost_of_600_pages : pages_to_copy * (cost_per_5_pages / 5) = expected_cost := by
  sorry

end cost_of_600_pages_l2215_221596


namespace range_of_a_l2215_221547

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + a * x - 2 < 0) → a < -1 :=
by
  sorry

end range_of_a_l2215_221547


namespace range_a_I_range_a_II_l2215_221537

variable (a: ℝ)

-- Define the proposition p and q
def p := (Real.sqrt (a^2 + 13) > Real.sqrt 17)
def q := ∀ x, (0 < x ∧ x < 3) → (x^2 - 2 * a * x - 2 = 0)

-- Prove question (I): If proposition p is true, find the range of the real number $a$
theorem range_a_I (h_p : p a) : a < -2 ∨ a > 2 :=
by sorry

-- Prove question (II): If both the proposition "¬q" and "p ∧ q" are false, find the range of the real number $a$
theorem range_a_II (h_neg_q : ¬ q a) (h_p_and_q : ¬ (p a ∧ q a)) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_a_I_range_a_II_l2215_221537


namespace theorem1_theorem2_theorem3_l2215_221542

-- Given conditions as definitions
variables {x y p q : ℝ}

-- Condition definitions
def condition1 : x + y = -p := sorry
def condition2 : x * y = q := sorry

-- Theorems to be proved
theorem theorem1 (h1 : x + y = -p) (h2 : x * y = q) : x^2 + y^2 = p^2 - 2 * q := sorry

theorem theorem2 (h1 : x + y = -p) (h2 : x * y = q) : x^3 + y^3 = -p^3 + 3 * p * q := sorry

theorem theorem3 (h1 : x + y = -p) (h2 : x * y = q) : x^4 + y^4 = p^4 - 4 * p^2 * q + 2 * q^2 := sorry

end theorem1_theorem2_theorem3_l2215_221542


namespace not_all_inequalities_hold_l2215_221511

theorem not_all_inequalities_hold (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end not_all_inequalities_hold_l2215_221511


namespace instructors_meeting_l2215_221568

theorem instructors_meeting (R P E M : ℕ) (hR : R = 5) (hP : P = 8) (hE : E = 10) (hM : M = 9) :
  Nat.lcm (Nat.lcm R P) (Nat.lcm E M) = 360 :=
by
  rw [hR, hP, hE, hM]
  sorry

end instructors_meeting_l2215_221568


namespace work_problem_l2215_221534

theorem work_problem (A B C : ℝ) (hB : B = 3) (h1 : 1 / B + 1 / C = 1 / 2) (h2 : 1 / A + 1 / C = 1 / 2) : A = 3 := by
  sorry

end work_problem_l2215_221534
