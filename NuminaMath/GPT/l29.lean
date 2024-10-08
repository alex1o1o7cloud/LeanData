import Mathlib

namespace proof_problem_l29_29675

variables {a b c d : ℝ} (h1 : a ≠ -2) (h2 : b ≠ -2) (h3 : c ≠ -2) (h4 : d ≠ -2)
variable (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
variable (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω)

theorem proof_problem : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 :=
sorry

end proof_problem_l29_29675


namespace number_of_permutations_l29_29474

theorem number_of_permutations (readers : Fin 8 → Type) : ∃! (n : ℕ), n = 40320 :=
by
  sorry

end number_of_permutations_l29_29474


namespace question_equals_answer_l29_29038

def heartsuit (a b : ℤ) : ℤ := |a + b|

theorem question_equals_answer : heartsuit (-3) (heartsuit 5 (-8)) = 0 := 
by
  sorry

end question_equals_answer_l29_29038


namespace area_of_WXYZ_l29_29703

structure Quadrilateral (α : Type _) :=
  (W : α) (X : α) (Y : α) (Z : α)
  (WZ ZW' WX XX' XY YY' YZ Z'W : ℝ)
  (area_WXYZ : ℝ)

theorem area_of_WXYZ' (WXYZ : Quadrilateral ℝ) 
  (h1 : WXYZ.WZ = 10) 
  (h2 : WXYZ.ZW' = 10)
  (h3 : WXYZ.WX = 6)
  (h4 : WXYZ.XX' = 6)
  (h5 : WXYZ.XY = 7)
  (h6 : WXYZ.YY' = 7)
  (h7 : WXYZ.YZ = 12)
  (h8 : WXYZ.Z'W = 12)
  (h9 : WXYZ.area_WXYZ = 15) : 
  ∃ area_WXZY' : ℝ, area_WXZY' = 45 :=
sorry

end area_of_WXYZ_l29_29703


namespace orchestra_musicians_l29_29772

theorem orchestra_musicians : ∃ (m n : ℕ), (m = n^2 + 11) ∧ (m = n * (n + 5)) ∧ m = 36 :=
by {
  sorry
}

end orchestra_musicians_l29_29772


namespace range_of_x_add_y_l29_29302

noncomputable def floor_not_exceeding (z : ℝ) : ℤ := ⌊z⌋

theorem range_of_x_add_y (x y : ℝ) (h1 : y = 3 * floor_not_exceeding x + 4) 
    (h2 : y = 4 * floor_not_exceeding (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) : 
    40 < x + y ∧ x + y < 41 :=
by 
  sorry 

end range_of_x_add_y_l29_29302


namespace largest_base_conversion_l29_29233

theorem largest_base_conversion :
  let a := (3: ℕ)
  let b := (1 * 2^1 + 1 * 2^0: ℕ)
  let c := (3 * 8^0: ℕ)
  let d := (1 * 3^1 + 1 * 3^0: ℕ)
  max a (max b (max c d)) = d :=
by
  sorry

end largest_base_conversion_l29_29233


namespace weeks_saved_l29_29292

theorem weeks_saved (w : ℕ) :
  (10 * w / 2) - ((10 * w / 2) / 4) = 15 → 
  w = 4 := 
by
  sorry

end weeks_saved_l29_29292


namespace books_added_is_10_l29_29536

-- Define initial number of books on the shelf
def initial_books : ℕ := 38

-- Define the final number of books on the shelf
def final_books : ℕ := 48

-- Define the number of books that Marta added
def books_added : ℕ := final_books - initial_books

-- Theorem stating that Marta added 10 books
theorem books_added_is_10 : books_added = 10 :=
by
  sorry

end books_added_is_10_l29_29536


namespace moles_of_NaOH_combined_l29_29501

-- Define the reaction conditions
variable (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ)

-- Define a proof problem that asserts the number of moles of NaOH combined
theorem moles_of_NaOH_combined
  (h1 : moles_NH4NO3 = 3)  -- 3 moles of NH4NO3 are combined
  (h2 : moles_NaNO3 = 3)  -- 3 moles of NaNO3 are formed
  : ∃ moles_NaOH : ℕ, moles_NaOH = 3 :=
by {
  -- Proof skeleton to be filled
  sorry
}

end moles_of_NaOH_combined_l29_29501


namespace final_segment_distance_l29_29754

theorem final_segment_distance :
  let north_distance := 2
  let east_distance := 1
  let south_distance := 1
  let net_north := north_distance - south_distance
  let net_east := east_distance
  let final_distance := Real.sqrt (net_north ^ 2 + net_east ^ 2)
  final_distance = Real.sqrt 2 :=
by
  sorry

end final_segment_distance_l29_29754


namespace least_possible_value_of_quadratic_l29_29511

theorem least_possible_value_of_quadratic (p q : ℝ) (hq : ∀ x : ℝ, x^2 + p * x + q ≥ 0) : q = (p^2) / 4 :=
sorry

end least_possible_value_of_quadratic_l29_29511


namespace calculate_solution_volume_l29_29054

theorem calculate_solution_volume (V : ℝ) (h : 0.35 * V = 1.4) : V = 4 :=
sorry

end calculate_solution_volume_l29_29054


namespace percentage_increase_l29_29523

theorem percentage_increase (original new : ℝ) (h₁ : original = 50) (h₂ : new = 80) :
  ((new - original) / original) * 100 = 60 :=
by
  sorry

end percentage_increase_l29_29523


namespace solution_set_abs_inequality_l29_29174

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l29_29174


namespace value_of_a_l29_29318

theorem value_of_a (a : ℝ) (h₁ : ∀ x : ℝ, (2 * x - (1/3) * a ≤ 0) → (x ≤ 2)) : a = 12 :=
sorry

end value_of_a_l29_29318


namespace perpendicular_parallel_l29_29747

variables {a b : Line} {α : Plane}

-- Definition of perpendicular and parallel relations should be available
-- since their exact details were not provided, placeholder functions will be used for demonstration

-- Placeholder definitions for perpendicular and parallel (they should be accurately defined elsewhere)
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_parallel {a b : Line} {α : Plane}
    (a_perp_alpha : perp a α)
    (b_perp_alpha : perp b α)
    : parallel a b :=
sorry

end perpendicular_parallel_l29_29747


namespace mouse_shortest_path_on_cube_l29_29037

noncomputable def shortest_path_length (edge_length : ℝ) : ℝ :=
  2 * edge_length * Real.sqrt 2

theorem mouse_shortest_path_on_cube :
  shortest_path_length 2 = 4 * Real.sqrt 2 :=
by
  sorry

end mouse_shortest_path_on_cube_l29_29037


namespace total_books_l29_29180

/-- Define Tim’s and Sam’s number of books. -/
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52

/-- Prove that together they have 96 books. -/
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l29_29180


namespace tan_alpha_l29_29439

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 3/5) (h2 : Real.pi / 2 < α ∧ α < Real.pi) : Real.tan α = -3/4 := 
  sorry

end tan_alpha_l29_29439


namespace reaches_school_early_l29_29856

theorem reaches_school_early (R : ℝ) (T : ℝ) (F : ℝ) (T' : ℝ)
    (h₁ : F = (6/5) * R)
    (h₂ : T = 24)
    (h₃ : R * T = F * T')
    : T - T' = 4 := by
  -- All the given conditions are set; fill in the below placeholder with the proof.
  sorry

end reaches_school_early_l29_29856


namespace plants_remaining_l29_29758

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l29_29758


namespace day_of_week_306_2003_l29_29117

-- Note: Definitions to support the conditions and the proof
def day_of_week (n : ℕ) : ℕ := n % 7

-- Theorem statement: Given conditions lead to the conclusion that the 306th day of the year 2003 falls on a Sunday
theorem day_of_week_306_2003 :
  (day_of_week (15) = 2) → (day_of_week (306) = 0) :=
by sorry

end day_of_week_306_2003_l29_29117


namespace max_value_expression_l29_29395

theorem max_value_expression (a b c : ℝ) (h : a * b * c + a + c - b = 0) : 
  ∃ m, (m = (1/(1+a^2) - 1/(1+b^2) + 1/(1+c^2))) ∧ (m = 5 / 4) :=
by 
  sorry

end max_value_expression_l29_29395


namespace original_decimal_number_l29_29368

theorem original_decimal_number (x : ℝ) (h : 10 * x - x / 10 = 23.76) : x = 2.4 :=
sorry

end original_decimal_number_l29_29368


namespace angle_in_third_quadrant_l29_29549

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) (h : π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π) :
  ∃ m : ℤ, -π - 2 * m * π < π / 2 - α ∧ (π / 2 - α) < -π / 2 - 2 * m * π :=
by
  -- Lean users note: The proof isn't required here, just setting up the statement as instructed.
  sorry

end angle_in_third_quadrant_l29_29549


namespace rope_segments_l29_29524

theorem rope_segments (total_length : ℝ) (n : ℕ) (h1 : total_length = 3) (h2 : n = 7) :
  (∃ segment_fraction : ℝ, segment_fraction = 1 / n ∧
   ∃ segment_length : ℝ, segment_length = total_length / n) :=
sorry

end rope_segments_l29_29524


namespace translate_vertex_l29_29695

/-- Given points A and B and their translations, verify the translated coordinates of B --/
theorem translate_vertex (A A' B B' : ℝ × ℝ)
  (hA : A = (0, 2))
  (hA' : A' = (-1, 0))
  (hB : B = (2, -1))
  (h_translation : A' = (A.1 - 1, A.2 - 2)) :
  B' = (B.1 - 1, B.2 - 2) :=
by
  sorry

end translate_vertex_l29_29695


namespace cistern_fill_time_l29_29206

variable (A_rate : ℚ) (B_rate : ℚ) (C_rate : ℚ)
variable (total_rate : ℚ := A_rate + C_rate - B_rate)

theorem cistern_fill_time (hA : A_rate = 1/7) (hB : B_rate = 1/9) (hC : C_rate = 1/12) :
  (1/total_rate) = 252/29 :=
by
  rw [hA, hB, hC]
  sorry

end cistern_fill_time_l29_29206


namespace remainder_problem_l29_29127

def rem (x y : ℚ) := x - y * (⌊x / y⌋ : ℤ)

theorem remainder_problem :
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  rem x y = (-19 : ℚ) / 63 :=
by
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  sorry

end remainder_problem_l29_29127


namespace sequence_m_value_l29_29315

theorem sequence_m_value (m : ℕ) (a : ℕ → ℝ) (h₀ : a 0 = 37) (h₁ : a 1 = 72)
  (hm : a m = 0) (h_rec : ∀ k, 1 ≤ k ∧ k < m → a (k + 1) = a (k - 1) - 3 / a k) : m = 889 :=
sorry

end sequence_m_value_l29_29315


namespace arithmetic_progression_square_l29_29151

theorem arithmetic_progression_square (a b c : ℝ) (h : b - a = c - b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by
  sorry

end arithmetic_progression_square_l29_29151


namespace distance_between_bars_l29_29494

theorem distance_between_bars (d V v : ℝ) 
  (h1 : x = 2 * d - 200)
  (h2 : d = P * V)
  (h3 : d - 200 = P * v)
  (h4 : V = (d - 200) / 4)
  (h5 : v = d / 9)
  (h6 : P = 4 * d / (d - 200))
  (h7 : P * (d - 200) = 8)
  (h8 : P * d = 18) :
  x = 1000 := by
  sorry

end distance_between_bars_l29_29494


namespace tangent_line_circle_sol_l29_29016

theorem tangent_line_circle_sol (r : ℝ) (h_pos : r > 0)
  (h_tangent : ∀ x y : ℝ, x^2 + y^2 = 2 * r → x + 2 * y = r) : r = 10 := 
sorry

end tangent_line_circle_sol_l29_29016


namespace least_multiple_of_17_gt_450_l29_29966

def least_multiple_gt (n x : ℕ) (k : ℕ) : Prop :=
  k * n > x ∧ ∀ m : ℕ, m * n > x → m ≥ k

theorem least_multiple_of_17_gt_450 : ∃ k : ℕ, least_multiple_gt 17 450 k :=
by
  use 27
  sorry

end least_multiple_of_17_gt_450_l29_29966


namespace base_conversion_zero_l29_29664

theorem base_conversion_zero (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 8 * A + B = 6 * B + A) : 8 * A + B = 0 :=
by
  sorry

end base_conversion_zero_l29_29664


namespace chickens_count_l29_29616

def total_animals := 13
def total_legs := 44
def legs_per_chicken := 2
def legs_per_buffalo := 4

theorem chickens_count : 
  (∃ c b : ℕ, c + b = total_animals ∧ legs_per_chicken * c + legs_per_buffalo * b = total_legs ∧ c = 4) :=
by
  sorry

end chickens_count_l29_29616


namespace nth_equation_proof_l29_29967

theorem nth_equation_proof (n : ℕ) (h : n ≥ 1) :
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1)) = 1 / n := 
sorry

end nth_equation_proof_l29_29967


namespace probability_cheryl_same_color_l29_29042

theorem probability_cheryl_same_color :
  let total_marble_count := 12
  let marbles_per_color := 3
  let carol_draw := 3
  let claudia_draw := 3
  let cheryl_draw := total_marble_count - carol_draw - claudia_draw
  let num_colors := 4

  0 < marbles_per_color ∧ marbles_per_color * num_colors = total_marble_count ∧
  0 < carol_draw ∧ carol_draw <= total_marble_count ∧
  0 < claudia_draw ∧ claudia_draw <= total_marble_count - carol_draw ∧
  0 < cheryl_draw ∧ cheryl_draw <= total_marble_count - carol_draw - claudia_draw ∧
  num_colors * (num_colors - 1) > 0
  →
  ∃ (p : ℚ), p = 2 / 55 := 
sorry

end probability_cheryl_same_color_l29_29042


namespace division_problem_l29_29858

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end division_problem_l29_29858


namespace range_of_f_4_l29_29003

theorem range_of_f_4 {a b c d : ℝ} 
  (h1 : 1 ≤ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ∧ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ≤ 2) 
  (h2 : 1 ≤ a*1^3 + b*1^2 + c*1 + d ∧ a*1^3 + b*1^2 + c*1 + d ≤ 3) 
  (h3 : 2 ≤ a*2^3 + b*2^2 + c*2 + d ∧ a*2^3 + b*2^2 + c*2 + d ≤ 4) 
  (h4 : -1 ≤ a*3^3 + b*3^2 + c*3 + d ∧ a*3^3 + b*3^2 + c*3 + d ≤ 1) :
  -21.75 ≤ a*4^3 + b*4^2 + c*4 + d ∧ a*4^3 + b*4^2 + c*4 + d ≤ 1 :=
sorry

end range_of_f_4_l29_29003


namespace ratio_a_c_l29_29570

theorem ratio_a_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 1) 
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 32 := 
by sorry

end ratio_a_c_l29_29570


namespace least_positive_integer_solution_l29_29584

theorem least_positive_integer_solution : 
  ∃ x : ℕ, x + 3567 ≡ 1543 [MOD 14] ∧ x = 6 := 
by
  -- proof goes here
  sorry

end least_positive_integer_solution_l29_29584


namespace upper_bound_of_expression_l29_29495

theorem upper_bound_of_expression (n : ℤ) (h1 : ∀ (n : ℤ), 4 * n + 7 > 1 ∧ 4 * n + 7 < 111) :
  ∃ U, (∀ (n : ℤ), 4 * n + 7 < U) ∧ 
       (∀ (n : ℤ), 4 * n + 7 < U ↔ 4 * n + 7 < 111) ∧ 
       U = 111 :=
by
  sorry

end upper_bound_of_expression_l29_29495


namespace mia_spent_total_l29_29402

theorem mia_spent_total (sibling_cost parent_cost : ℕ) (num_siblings num_parents : ℕ)
    (h1 : sibling_cost = 30)
    (h2 : parent_cost = 30)
    (h3 : num_siblings = 3)
    (h4 : num_parents = 2) :
    sibling_cost * num_siblings + parent_cost * num_parents = 150 :=
by
  sorry

end mia_spent_total_l29_29402


namespace time_for_grid_5x5_l29_29268

-- Definition for the 3x7 grid conditions
def grid_3x7_minutes := 26
def grid_3x7_total_length := 4 * 7 + 8 * 3
def time_per_unit_length := grid_3x7_minutes / grid_3x7_total_length

-- Definition for the 5x5 grid total length
def grid_5x5_total_length := 6 * 5 + 6 * 5

-- Theorem stating that the time it takes to trace all lines of a 5x5 grid is 30 minutes
theorem time_for_grid_5x5 : (time_per_unit_length * grid_5x5_total_length) = 30 := by
  sorry

end time_for_grid_5x5_l29_29268


namespace expand_expression_l29_29839

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end expand_expression_l29_29839


namespace train_length_l29_29725

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmph = 60) 
  (h_time : time_sec = 7.199424046076314) 
  (h_length : length_m = 120)
  : speed_kmph * (1000 / 3600) * time_sec = length_m :=
by 
  sorry

end train_length_l29_29725


namespace minimum_reciprocal_sum_l29_29882

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

theorem minimum_reciprocal_sum (a m n : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : f a (-1) = -1) (h₄ : m + n = 2) (h₅ : 0 < m) (h₆ : 0 < n) :
  (1 / m) + (1 / n) = 2 :=
by
  sorry

end minimum_reciprocal_sum_l29_29882


namespace initial_order_cogs_l29_29551

theorem initial_order_cogs (x : ℕ) (h : (x + 60 : ℚ) / (x / 36 + 1) = 45) : x = 60 := 
sorry

end initial_order_cogs_l29_29551


namespace floor_S_proof_l29_29817

noncomputable def floor_S (a b c d: ℝ) : ℝ :=
⌊a + b + c + d⌋

theorem floor_S_proof (a b c d : ℝ)
  (h1 : a ^ 2 + 2 * b ^ 2 = 2016)
  (h2 : c ^ 2 + 2 * d ^ 2 = 2016)
  (h3 : a * c = 1024)
  (h4 : b * d = 1024)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : floor_S a b c d = 129 := 
sorry

end floor_S_proof_l29_29817


namespace king_total_payment_l29_29303

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l29_29303


namespace part1_values_correct_estimated_students_correct_l29_29890

def students_data : List ℕ :=
  [30, 60, 70, 10, 30, 115, 70, 60, 75, 90, 15, 70, 40, 75, 105, 80, 60, 30, 70, 45]

def total_students := 200

def categorized_counts := (2, 5, 10, 3) -- (0 ≤ t < 30, 30 ≤ t < 60, 60 ≤ t < 90, 90 ≤ t < 120)

def mean := 60

def median := 65

def mode := 70

theorem part1_values_correct :
  let a := 5
  let b := 3
  let c := 65
  let d := 70
  categorized_counts = (2, a, 10, b) ∧ mean = 60 ∧ median = c ∧ mode = d := by {
  -- Proof will be provided here
  sorry
}

theorem estimated_students_correct :
  let at_least_avg := 130
  at_least_avg = (total_students * 13 / 20) := by {
  -- Proof will be provided here
  sorry
}

end part1_values_correct_estimated_students_correct_l29_29890


namespace total_samples_correct_l29_29803

-- Define the conditions as constants
def samples_per_shelf : ℕ := 65
def number_of_shelves : ℕ := 7

-- Define the total number of samples and the expected result
def total_samples : ℕ := samples_per_shelf * number_of_shelves
def expected_samples : ℕ := 455

-- State the theorem to be proved
theorem total_samples_correct : total_samples = expected_samples := by
  -- Proof to be filled in
  sorry

end total_samples_correct_l29_29803


namespace divide_polynomials_l29_29121

theorem divide_polynomials (n : ℕ) (h : ∃ (k : ℤ), n^2 + 3*n + 51 = 13 * k) : 
  ∃ (m : ℤ), 21*n^2 + 89*n + 44 = 169 * m := by
  sorry

end divide_polynomials_l29_29121


namespace find_S16_l29_29555

theorem find_S16 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 12 = -8)
  (h2 : S 9 = -9)
  (h_sum : ∀ n, S n = (n * (a 1 + a n) / 2)) :
  S 16 = -72 := 
by
  sorry

end find_S16_l29_29555


namespace find_W_from_conditions_l29_29528

theorem find_W_from_conditions :
  ∀ (x y : ℝ), (y = 1 / x ∧ y = |x| + 1) → (x + y = Real.sqrt 5) :=
by
  sorry

end find_W_from_conditions_l29_29528


namespace kevin_speed_first_half_l29_29215

-- Let's define the conditions as variables and constants
variable (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ)
variable (time_20mph : ℝ) (time_8mph : ℝ) (distance_first_half : ℕ)
variable (speed_first_half : ℝ)

-- Conditions from the problem
def conditions (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ) : Prop :=
  total_distance = 17 ∧ 
  distance_20mph = 20 * 1 / 2 ∧
  distance_8mph = 8 * 1 / 4

-- Proof objective based on conditions and correct answer
theorem kevin_speed_first_half (
  h : conditions total_distance distance_20mph distance_8mph
) : speed_first_half = 10 := by
  sorry

end kevin_speed_first_half_l29_29215


namespace train_speed_identification_l29_29130

-- Define the conditions
def train_length : ℕ := 300
def crossing_time : ℕ := 30

-- Define the speed calculation
def calculate_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The target theorem stating the speed of the train
theorem train_speed_identification : calculate_speed train_length crossing_time = 10 := 
by 
  sorry

end train_speed_identification_l29_29130


namespace weight_order_l29_29819

variable {P Q R S T : ℕ}

theorem weight_order
    (h1 : Q + S = 1200)
    (h2 : R + T = 2100)
    (h3 : Q + T = 800)
    (h4 : Q + R = 900)
    (h5 : P + T = 700)
    (hP : P < 1000)
    (hQ : Q < 1000)
    (hR : R < 1000)
    (hS : S < 1000)
    (hT : T < 1000) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
sorry

end weight_order_l29_29819


namespace find_s_l29_29719

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l29_29719


namespace marcus_savings_l29_29120

theorem marcus_savings
  (running_shoes_price : ℝ)
  (running_shoes_discount : ℝ)
  (cashback : ℝ)
  (running_shoes_tax_rate : ℝ)
  (athletic_socks_price : ℝ)
  (athletic_socks_tax_rate : ℝ)
  (bogo : ℝ)
  (performance_tshirt_price : ℝ)
  (performance_tshirt_discount : ℝ)
  (performance_tshirt_tax_rate : ℝ)
  (total_budget : ℝ)
  (running_shoes_final_price : ℝ)
  (athletic_socks_final_price : ℝ)
  (performance_tshirt_final_price : ℝ) :
  running_shoes_price = 120 →
  running_shoes_discount = 30 / 100 →
  cashback = 10 →
  running_shoes_tax_rate = 8 / 100 →
  athletic_socks_price = 25 →
  athletic_socks_tax_rate = 6 / 100 →
  bogo = 2 →
  performance_tshirt_price = 55 →
  performance_tshirt_discount = 10 / 100 →
  performance_tshirt_tax_rate = 7 / 100 →
  total_budget = 250 →
  running_shoes_final_price = (running_shoes_price * (1 - running_shoes_discount) - cashback) * (1 + running_shoes_tax_rate) →
  athletic_socks_final_price = (athletic_socks_price * bogo) * (1 + athletic_socks_tax_rate) / bogo →
  performance_tshirt_final_price = (performance_tshirt_price * (1 - performance_tshirt_discount)) * (1 + performance_tshirt_tax_rate) →
  total_budget - (running_shoes_final_price + athletic_socks_final_price + performance_tshirt_final_price) = 103.86 :=
sorry

end marcus_savings_l29_29120


namespace total_earnings_correct_l29_29969

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l29_29969


namespace num_distinct_combinations_l29_29237

-- Define the conditions
def num_dials : Nat := 4
def digits : List Nat := List.range 10  -- Digits from 0 to 9

-- Define what it means for a combination to have distinct digits
def distinct_digits (comb : List Nat) : Prop :=
  comb.length = num_dials ∧ comb.Nodup

-- The main statement for the theorem
theorem num_distinct_combinations : 
  ∃ (n : Nat), n = 5040 ∧ ∀ comb : List Nat, distinct_digits comb → comb.length = num_dials →
  (List.permutations digits).length = n :=
by
  sorry

end num_distinct_combinations_l29_29237


namespace minimum_digits_for_divisibility_l29_29141

theorem minimum_digits_for_divisibility :
  ∃ n : ℕ, (10 * 2013 + n) % 2520 = 0 ∧ n < 1000 :=
sorry

end minimum_digits_for_divisibility_l29_29141


namespace total_students_is_48_l29_29548

-- Definitions according to the given conditions
def boys'_row := 24
def girls'_row := 24

-- Theorem based on the question and the correct answer
theorem total_students_is_48 :
  boys'_row + girls'_row = 48 :=
by
  sorry

end total_students_is_48_l29_29548


namespace weekly_sales_correct_l29_29359

open Real

noncomputable def cost_left_handed_mouse (cost_normal_mouse : ℝ) : ℝ :=
  cost_normal_mouse * 1.3

noncomputable def cost_left_handed_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  cost_normal_keyboard * 1.2

noncomputable def cost_left_handed_scissors (cost_normal_scissors : ℝ) : ℝ :=
  cost_normal_scissors * 1.5

noncomputable def daily_sales_mouse (cost_normal_mouse : ℝ) : ℝ :=
  25 * cost_left_handed_mouse cost_normal_mouse

noncomputable def daily_sales_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  10 * cost_left_handed_keyboard cost_normal_keyboard

noncomputable def daily_sales_scissors (cost_normal_scissors : ℝ) : ℝ :=
  15 * cost_left_handed_scissors cost_normal_scissors

noncomputable def bundle_price (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  (cost_left_handed_mouse cost_normal_mouse + cost_left_handed_keyboard cost_normal_keyboard + cost_left_handed_scissors cost_normal_scissors) * 0.9

noncomputable def daily_sales_bundle (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  5 * bundle_price cost_normal_mouse cost_normal_keyboard cost_normal_scissors

noncomputable def weekly_sales (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  3 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors) +
  1.5 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors)

theorem weekly_sales_correct :
  weekly_sales 120 80 30 = 29922.25 := sorry

end weekly_sales_correct_l29_29359


namespace total_percentage_of_samplers_l29_29970

theorem total_percentage_of_samplers :
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  (pA + pA_not_caught + pB + pB_not_caught + pC + pC_not_caught + pD + pD_not_caught) = 54 :=
by
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  sorry

end total_percentage_of_samplers_l29_29970


namespace tens_digit_of_sum_l29_29540

theorem tens_digit_of_sum (a b c : ℕ) (h : a = c + 3) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) :
    ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ (202 * c + 20 * b + 303) % 100 = t ∧ t / 10 = 1 :=
by
  use (20 * b + 3)
  sorry

end tens_digit_of_sum_l29_29540


namespace sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l29_29388

-- Define a convex n-gon and prove that the sum of its interior angles is (n-2) * 180 degrees
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (sum_of_angles : ℝ), sum_of_angles = (n-2) * 180 :=
sorry

-- Define a convex n-gon and prove that the number of triangles formed by dividing with non-intersecting diagonals is n-2
theorem number_of_triangles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (num_of_triangles : ℕ), num_of_triangles = n-2 :=
sorry

end sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l29_29388


namespace twenty_five_percent_of_five_hundred_l29_29414

theorem twenty_five_percent_of_five_hundred : 0.25 * 500 = 125 := 
by 
  sorry

end twenty_five_percent_of_five_hundred_l29_29414


namespace find_2alpha_minus_beta_l29_29283

theorem find_2alpha_minus_beta (α β : ℝ) (tan_diff : Real.tan (α - β) = 1 / 2) 
  (cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) (α_range : 0 < α ∧ α < Real.pi) 
  (β_range : 0 < β ∧ β < Real.pi) : 2 * α - β = -3 * Real.pi / 4 :=
sorry

end find_2alpha_minus_beta_l29_29283


namespace intersection_A_B_union_A_compB_l29_29378

-- Define the sets A and B
def A : Set ℝ := { x | x^2 + 3 * x - 10 < 0 }
def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define the complement of B in the universal set
def comp_B : Set ℝ := { x | ¬ B x }

-- 1. Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_A_B :
  A ∩ B = { x | -5 < x ∧ x ≤ -1 } :=
by 
  sorry

-- 2. Prove that A ∪ (complement of B) = {x | -5 < x ∧ x < 3}
theorem union_A_compB :
  A ∪ comp_B = { x | -5 < x ∧ x < 3 } :=
by 
  sorry

end intersection_A_B_union_A_compB_l29_29378


namespace problem_solution_l29_29252

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end problem_solution_l29_29252


namespace remaining_miles_to_be_built_l29_29417

-- Definitions from problem conditions
def current_length : ℕ := 200
def target_length : ℕ := 650
def first_day_miles : ℕ := 50
def second_day_miles : ℕ := 3 * first_day_miles

-- Lean theorem statement
theorem remaining_miles_to_be_built : 
  (target_length - current_length) - (first_day_miles + second_day_miles) = 250 := 
by 
  sorry

end remaining_miles_to_be_built_l29_29417


namespace find_number_l29_29377

theorem find_number (N : ℕ) (h1 : ∃ k : ℤ, N = 13 * k + 11) (h2 : ∃ m : ℤ, N = 17 * m + 9) : N = 89 := 
sorry

end find_number_l29_29377


namespace necessary_sufficient_condition_l29_29171

theorem necessary_sufficient_condition (A B C : ℝ)
    (h : ∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) :
    |A - B + C| ≤ 2 * Real.sqrt (A * C) := 
by sorry

end necessary_sufficient_condition_l29_29171


namespace number_of_arrangements_is_48_l29_29870

noncomputable def number_of_arrangements (students : List String) (boy_not_at_ends : String) (adjacent_girls : List String) : Nat :=
  sorry

theorem number_of_arrangements_is_48 : number_of_arrangements ["A", "B1", "B2", "G1", "G2", "G3"] "B1" ["G1", "G2", "G3"] = 48 :=
by
  sorry

end number_of_arrangements_is_48_l29_29870


namespace cases_in_1990_is_correct_l29_29282

-- Define the initial and final number of cases.
def initial_cases : ℕ := 600000
def final_cases : ℕ := 200

-- Define the years and time spans.
def year_1970 : ℕ := 1970
def year_1985 : ℕ := 1985
def year_2000 : ℕ := 2000

def span_1970_to_1985 : ℕ := year_1985 - year_1970 -- 15 years
def span_1985_to_2000 : ℕ := year_2000 - year_1985 -- 15 years

-- Define the rate of decrease from 1970 to 1985 as r cases per year.
-- Define the rate of decrease from 1985 to 2000 as (r / 2) cases per year.
def rate_of_decrease_1 (r : ℕ) := r
def rate_of_decrease_2 (r : ℕ) := r / 2

-- Define the intermediate number of cases in 1985.
def cases_in_1985 (r : ℕ) : ℕ := initial_cases - (span_1970_to_1985 * rate_of_decrease_1 r)

-- Define the number of cases in 1990.
def cases_in_1990 (r : ℕ) : ℕ := cases_in_1985 r - (5 * rate_of_decrease_2 r) -- 5 years from 1985 to 1990

-- Total decrease in cases over 30 years.
def total_decrease : ℕ := initial_cases - final_cases

-- Formalize the proof that the number of cases in 1990 is 133,450.
theorem cases_in_1990_is_correct : 
  ∃ (r : ℕ), 15 * rate_of_decrease_1 r + 15 * rate_of_decrease_2 r = total_decrease ∧ cases_in_1990 r = 133450 := 
by {
  sorry
}

end cases_in_1990_is_correct_l29_29282


namespace parabola_and_hyperbola_focus_equal_l29_29710

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) :=
(2, 0)

noncomputable def hyperbola_focus : (ℝ × ℝ) :=
(2, 0)

theorem parabola_and_hyperbola_focus_equal
  (p : ℝ)
  (h_parabola : parabola_focus p = (2, 0))
  (h_hyperbola : hyperbola_focus = (2, 0)) :
  p = 4 := by
  sorry

end parabola_and_hyperbola_focus_equal_l29_29710


namespace debt_payments_l29_29609

noncomputable def average_payment (total_amount : ℕ) (payments : ℕ) : ℕ := total_amount / payments

theorem debt_payments (x : ℕ) :
  8 * x + 44 * (x + 65) = 52 * 465 → x = 410 :=
by
  intros h
  sorry

end debt_payments_l29_29609


namespace ratio_of_cost_to_selling_price_l29_29415

-- Define the given conditions
def cost_price (CP : ℝ) := CP
def selling_price (CP : ℝ) : ℝ := CP + 0.25 * CP

-- Lean statement for the problem
theorem ratio_of_cost_to_selling_price (CP SP : ℝ) (h1 : SP = selling_price CP) : CP / SP = 4 / 5 :=
by
  sorry

end ratio_of_cost_to_selling_price_l29_29415


namespace polynomial_identity_l29_29986

noncomputable def p (x : ℝ) : ℝ := x 

theorem polynomial_identity (p : ℝ → ℝ) (h : ∀ q : ℝ → ℝ, ∀ x : ℝ, p (q x) = q (p x)) : 
  (∀ x : ℝ, p x = x) :=
by
  sorry

end polynomial_identity_l29_29986


namespace inverse_h_l29_29613

-- definitions of f, g, and h
def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

-- statement of the problem
theorem inverse_h (x : ℝ) : (∃ y : ℝ, h y = x) ∧ ∀ y : ℝ, h y = x → y = (x - 23) / 12 :=
by
  sorry

end inverse_h_l29_29613


namespace find_x_when_y_neg4_l29_29311

variable {x y : ℝ}
variable (k : ℝ)

-- Condition: x is inversely proportional to y
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop :=
  x * y = k

theorem find_x_when_y_neg4 (h : inversely_proportional 5 10 50) :
  inversely_proportional x (-4) 50 → x = -25 / 2 :=
by sorry

end find_x_when_y_neg4_l29_29311


namespace total_cards_traded_l29_29983

-- Define the total number of cards traded in both trades
def total_traded (p1_t: ℕ) (r1_t: ℕ) (p2_t: ℕ) (r2_t: ℕ): ℕ :=
  (p1_t + r1_t) + (p2_t + r2_t)

-- Given conditions as definitions
def padma_trade1 := 2   -- Cards Padma traded in the first trade
def robert_trade1 := 10  -- Cards Robert traded in the first trade
def padma_trade2 := 15  -- Cards Padma traded in the second trade
def robert_trade2 := 8   -- Cards Robert traded in the second trade

-- Theorem stating the total number of cards traded is 35
theorem total_cards_traded : 
  total_traded padma_trade1 robert_trade1 padma_trade2 robert_trade2 = 35 :=
by
  sorry

end total_cards_traded_l29_29983


namespace zero_not_in_range_of_g_l29_29893

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l29_29893


namespace similar_triangle_leg_l29_29913

theorem similar_triangle_leg (x : Real) : 
  (12 / x = 9 / 7) → x = 84 / 9 := by
  intro h
  sorry

end similar_triangle_leg_l29_29913


namespace effective_weight_lowered_l29_29924

theorem effective_weight_lowered 
    (num_weight_plates : ℕ) 
    (weight_per_plate : ℝ) 
    (increase_percentage : ℝ) 
    (total_weight_without_technology : ℝ) 
    (additional_weight : ℝ) 
    (effective_weight_lowering : ℝ) 
    (h1 : num_weight_plates = 10)
    (h2 : weight_per_plate = 30)
    (h3 : increase_percentage = 0.20)
    (h4 : total_weight_without_technology = num_weight_plates * weight_per_plate)
    (h5 : additional_weight = increase_percentage * total_weight_without_technology)
    (h6 : effective_weight_lowering = total_weight_without_technology + additional_weight) :
    effective_weight_lowering = 360 := 
by
  sorry

end effective_weight_lowered_l29_29924


namespace monthly_rent_l29_29020

-- Definition
def total_amount_saved := 2225
def extra_amount_needed := 775
def deposit := 500

-- Total amount required
def total_amount_required := total_amount_saved + extra_amount_needed
def total_rent_plus_deposit (R : ℝ) := 2 * R + deposit

-- The statement to prove
theorem monthly_rent (R : ℝ) : total_rent_plus_deposit R = total_amount_required → R = 1250 :=
by
  intros h
  exact sorry -- Proof is omitted.

end monthly_rent_l29_29020


namespace function_decreasing_interval_l29_29735

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 * (a * x + b)

theorem function_decreasing_interval :
  (deriv (f a b) 2 = 0) ∧ (deriv (f a b) 1 = -3) →
  ∃ (a b : ℝ), (deriv (f a b) x < 0) ↔ (0 < x ∧ x < 2) := sorry

end function_decreasing_interval_l29_29735


namespace smallest_number_of_lawyers_l29_29185

/-- Given that:
- n is the number of delegates, where 220 < n < 254
- m is the number of economists, so the number of lawyers is n - m
- Each participant played with each other participant exactly once.
- A match winner got one point, the loser got none, and in case of a draw, both participants received half a point each.
- By the end of the tournament, each participant gained half of all their points from matches against economists.

Prove that the smallest number of lawyers participating in the tournament is 105. -/
theorem smallest_number_of_lawyers (n m : ℕ) (h1 : 220 < n) (h2 : n < 254)
  (h3 : m * (m - 1) + (n - m) * (n - m - 1) = n * (n - 1))
  (h4 : m * (m - 1) = 2 * (n * (n - 1)) / 4) :
  n - m = 105 :=
sorry

end smallest_number_of_lawyers_l29_29185


namespace interest_rate_l29_29880

theorem interest_rate (part1_amount part2_amount total_amount total_income : ℝ) (interest_rate1 interest_rate2 : ℝ) :
  part1_amount = 2000 →
  part2_amount = total_amount - part1_amount →
  interest_rate2 = 6 →
  total_income = (part1_amount * interest_rate1 / 100) + (part2_amount * interest_rate2 / 100) →
  total_amount = 2500 →
  total_income = 130 →
  interest_rate1 = 5 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end interest_rate_l29_29880


namespace how_many_years_older_is_a_than_b_l29_29408

variable (a b c : ℕ)

theorem how_many_years_older_is_a_than_b
  (hb : b = 4)
  (hc : c = b / 2)
  (h_ages_sum : a + b + c = 12) :
  a - b = 2 := by
  sorry

end how_many_years_older_is_a_than_b_l29_29408


namespace fitted_bowling_ball_volume_l29_29826

theorem fitted_bowling_ball_volume :
  let r_bowl := 20 -- radius of the bowling ball in cm
  let r_hole1 := 1 -- radius of the first hole in cm
  let r_hole2 := 2 -- radius of the second hole in cm
  let r_hole3 := 2 -- radius of the third hole in cm
  let depth := 10 -- depth of each hole in cm
  let V_bowl := (4/3) * Real.pi * r_bowl^3
  let V_hole1 := Real.pi * r_hole1^2 * depth
  let V_hole2 := Real.pi * r_hole2^2 * depth
  let V_hole3 := Real.pi * r_hole3^2 * depth
  let V_holes := V_hole1 + V_hole2 + V_hole3
  let V_fitted := V_bowl - V_holes
  V_fitted = (31710 / 3) * Real.pi :=
by sorry

end fitted_bowling_ball_volume_l29_29826


namespace sequence_1_formula_sequence_2_formula_sequence_3_formula_l29_29239

theorem sequence_1_formula (n : ℕ) (hn : n > 0) : 
  (∃ a : ℕ → ℚ, (a 1 = 1/2) ∧ (a 2 = 1/6) ∧ (a 3 = 1/12) ∧ (a 4 = 1/20) ∧ (∀ n, a n = 1/(n*(n+1)))) :=
by
  sorry

theorem sequence_2_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (∀ n, a n = 2^(n-1))) :=
by
  sorry

theorem sequence_3_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℚ, (a 1 = 4/5) ∧ (a 2 = 1/2) ∧ (a 3 = 4/11) ∧ (a 4 = 2/7) ∧ (∀ n, a n = 4/(3*n + 2))) :=
by
  sorry

end sequence_1_formula_sequence_2_formula_sequence_3_formula_l29_29239


namespace instantaneous_velocity_at_3s_l29_29267

theorem instantaneous_velocity_at_3s (t s v : ℝ) (hs : s = t^3) (hts : t = 3*s) : v = 27 :=
by
  sorry

end instantaneous_velocity_at_3s_l29_29267


namespace solve_first_equation_solve_second_equation_l29_29576

theorem solve_first_equation (x : ℤ) : 4 * x + 3 = 5 * x - 1 → x = 4 :=
by
  intros h
  sorry

theorem solve_second_equation (x : ℤ) : 4 * (x - 1) = 1 - x → x = 1 :=
by
  intros h
  sorry

end solve_first_equation_solve_second_equation_l29_29576


namespace original_pencils_l29_29363

-- Definition of the conditions
def pencils_initial := 115
def pencils_added := 100
def pencils_total := 215

-- Theorem stating the problem to be proved
theorem original_pencils :
  pencils_initial + pencils_added = pencils_total :=
by
  sorry

end original_pencils_l29_29363


namespace cube_root_of_5_irrational_l29_29849

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l29_29849


namespace greatest_b_not_in_range_l29_29901

theorem greatest_b_not_in_range (b : ℤ) : ∀ x : ℝ, ¬ (x^2 + (b : ℝ) * x + 20 = -9) ↔ b ≤ 10 :=
by
  sorry

end greatest_b_not_in_range_l29_29901


namespace problem_inequality_l29_29586

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x - y + z) * (y - z + x) * (z - x + y) ≤ x * y * z := sorry

end problem_inequality_l29_29586


namespace two_pow_ge_two_mul_l29_29792

theorem two_pow_ge_two_mul (n : ℕ) : 2^n ≥ 2 * n :=
sorry

end two_pow_ge_two_mul_l29_29792


namespace average_speed_l29_29324

-- Define the speeds and times
def speed1 : ℝ := 120 -- km/h
def time1 : ℝ := 1 -- hour

def speed2 : ℝ := 150 -- km/h
def time2 : ℝ := 2 -- hours

def speed3 : ℝ := 80 -- km/h
def time3 : ℝ := 0.5 -- hour

-- Define the conversion factor
def km_to_miles : ℝ := 0.62

-- Calculate total distance (in kilometers)
def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time2
def distance3 : ℝ := speed3 * time3

def total_distance_km : ℝ := distance1 + distance2 + distance3

-- Convert total distance to miles
def total_distance_miles : ℝ := total_distance_km * km_to_miles

-- Calculate total time (in hours)
def total_time : ℝ := time1 + time2 + time3

-- Final proof statement for average speed
theorem average_speed : total_distance_miles / total_time = 81.49 := by {
  sorry
}

end average_speed_l29_29324


namespace shaded_area_in_6x6_grid_l29_29296

def total_shaded_area (grid_size : ℕ) (triangle_squares : ℕ) (num_triangles : ℕ)
  (trapezoid_squares : ℕ) (num_trapezoids : ℕ) : ℕ :=
  (triangle_squares * num_triangles) + (trapezoid_squares * num_trapezoids)

theorem shaded_area_in_6x6_grid :
  total_shaded_area 6 2 2 3 4 = 16 :=
by
  -- Proof omitted for demonstration purposes
  sorry

end shaded_area_in_6x6_grid_l29_29296


namespace age_of_b_l29_29654

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 32) : b = 12 :=
by sorry

end age_of_b_l29_29654


namespace total_packages_of_gum_l29_29396

theorem total_packages_of_gum (R_total R_extra R_per_package A_total A_extra A_per_package : ℕ) 
  (hR1 : R_total = 41) (hR2 : R_extra = 6) (hR3 : R_per_package = 7)
  (hA1 : A_total = 23) (hA2 : A_extra = 3) (hA3 : A_per_package = 5) :
  (R_total - R_extra) / R_per_package + (A_total - A_extra) / A_per_package = 9 :=
by
  sorry

end total_packages_of_gum_l29_29396


namespace parrots_in_each_cage_l29_29736

theorem parrots_in_each_cage (P : ℕ) (h : 9 * P + 9 * 6 = 72) : P = 2 :=
sorry

end parrots_in_each_cage_l29_29736


namespace find_a_l29_29715

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l29_29715


namespace cylinder_volume_l29_29503

theorem cylinder_volume (V_sphere : ℝ) (V_cylinder : ℝ) (R H : ℝ) 
  (h1 : V_sphere = 4 * π / 3) 
  (h2 : (4 * π * R ^ 3) / 3 = V_sphere) 
  (h3 : H = 2 * R) 
  (h4 : R = 1) : V_cylinder = 2 * π :=
by
  sorry

end cylinder_volume_l29_29503


namespace distance_between_fourth_and_work_l29_29107

theorem distance_between_fourth_and_work (x : ℝ) (h₁ : x > 0) :
  let total_distance := x + 0.5 * x + 2 * x
  let to_fourth := (1 / 3) * total_distance
  let total_to_fourth := total_distance + to_fourth
  3 * total_to_fourth = 14 * x :=
by
  sorry

end distance_between_fourth_and_work_l29_29107


namespace comparison_of_A_and_B_l29_29804

noncomputable def A (m : ℝ) : ℝ := Real.sqrt (m + 1) - Real.sqrt m
noncomputable def B (m : ℝ) : ℝ := Real.sqrt m - Real.sqrt (m - 1)

theorem comparison_of_A_and_B (m : ℝ) (h : m > 1) : A m < B m :=
by
  sorry

end comparison_of_A_and_B_l29_29804


namespace find_n_in_arithmetic_sequence_l29_29562

noncomputable def arithmetic_sequence_n : ℕ :=
  sorry

theorem find_n_in_arithmetic_sequence (a : ℕ → ℕ) (d n : ℕ) :
  (a 3) + (a 4) = 10 → (a (n-3) + a (n-2)) = 30 → n * (a 1 + a n) / 2 = 100 → n = 10 :=
  sorry

end find_n_in_arithmetic_sequence_l29_29562


namespace n_mul_n_plus_one_even_l29_29951

theorem n_mul_n_plus_one_even (n : ℤ) : Even (n * (n + 1)) := 
sorry

end n_mul_n_plus_one_even_l29_29951


namespace find_f_2015_l29_29840

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2015
  (h1 : ∀ x, f (-x) = -f x) -- f is an odd function
  (h2 : ∀ x, f (x + 2) = -f x) -- f(x+2) = -f(x)
  (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) -- f(x) = 2x^2 for x in (0, 2)
  : f 2015 = -2 :=
sorry

end find_f_2015_l29_29840


namespace ribbon_per_box_l29_29175

theorem ribbon_per_box (total_ribbon : ℚ) (total_boxes : ℕ) (same_ribbon_each_box : ℚ) 
  (h1 : total_ribbon = 5/12) (h2 : total_boxes = 5) : 
  same_ribbon_each_box = 1/12 :=
sorry

end ribbon_per_box_l29_29175


namespace original_speed_correct_l29_29242

variables (t m s : ℝ)

noncomputable def original_speed (t m s : ℝ) : ℝ :=
  ((t * m + Real.sqrt (t^2 * m^2 + 4 * t * m * s)) / (2 * t))

theorem original_speed_correct (t m s : ℝ) (ht : 0 < t) : 
  original_speed t m s = (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t) :=
by
  sorry

end original_speed_correct_l29_29242


namespace intersection_of_lines_l29_29873

theorem intersection_of_lines : ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 5 = 15 * x - 2 ∧ x = 1 / 3 ∧ y = 0 :=
by
  sorry

end intersection_of_lines_l29_29873


namespace polynomial_evaluation_l29_29372

theorem polynomial_evaluation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end polynomial_evaluation_l29_29372


namespace fraction_of_paint_used_l29_29989

theorem fraction_of_paint_used 
  (total_paint : ℕ)
  (paint_used_first_week : ℚ)
  (total_paint_used : ℕ)
  (paint_fraction_first_week : ℚ)
  (remaining_paint : ℚ)
  (paint_used_second_week : ℚ)
  (paint_fraction_second_week : ℚ)
  (h1 : total_paint = 360)
  (h2 : paint_fraction_first_week = 2/3)
  (h3 : paint_used_first_week = paint_fraction_first_week * total_paint)
  (h4 : remaining_paint = total_paint - paint_used_first_week)
  (h5 : remaining_paint = 120)
  (h6 : total_paint_used = 264)
  (h7 : paint_used_second_week = total_paint_used - paint_used_first_week)
  (h8 : paint_fraction_second_week = paint_used_second_week / remaining_paint):
  paint_fraction_second_week = 1/5 := 
by 
  sorry

end fraction_of_paint_used_l29_29989


namespace amelia_painted_faces_l29_29040

def faces_of_cuboid : ℕ := 6
def number_of_cuboids : ℕ := 6

theorem amelia_painted_faces : faces_of_cuboid * number_of_cuboids = 36 :=
by {
  sorry
}

end amelia_painted_faces_l29_29040


namespace scramble_time_is_correct_l29_29532

-- Define the conditions
def sausages : ℕ := 3
def fry_time_per_sausage : ℕ := 5
def eggs : ℕ := 6
def total_time : ℕ := 39

-- Define the time to scramble each egg
def scramble_time_per_egg : ℕ :=
  let frying_time := sausages * fry_time_per_sausage
  let scrambling_time := total_time - frying_time
  scrambling_time / eggs

-- The theorem stating the main question and desired answer
theorem scramble_time_is_correct : scramble_time_per_egg = 4 := by
  sorry

end scramble_time_is_correct_l29_29532


namespace log_neg_inequality_l29_29526

theorem log_neg_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  Real.log (-a) > Real.log (-b) := 
sorry

end log_neg_inequality_l29_29526


namespace points_6_units_away_from_neg1_l29_29308

theorem points_6_units_away_from_neg1 (A : ℝ) (h : A = -1) :
  { x : ℝ | abs (x - A) = 6 } = { -7, 5 } :=
by
  sorry

end points_6_units_away_from_neg1_l29_29308


namespace smallest_three_digit_number_l29_29484

theorem smallest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧
  (x % 2 = 0) ∧
  ((x + 1) % 3 = 0) ∧
  ((x + 2) % 4 = 0) ∧
  ((x + 3) % 5 = 0) ∧
  ((x + 4) % 6 = 0) ∧
  x = 122 :=
by
  sorry

end smallest_three_digit_number_l29_29484


namespace solution_set_inequality_l29_29136

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom increasing_on_positive : ∀ {x y : ℝ}, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_inequality :
  {x : ℝ | (f x) / x < 0} = {x : ℝ | x < -1} ∪ {x | 0 < x ∧ x < 1} := sorry

end solution_set_inequality_l29_29136


namespace volume_of_given_wedge_l29_29358

noncomputable def volume_of_wedge (d : ℝ) (angle : ℝ) : ℝ := 
  let r := d / 2
  let height := d
  let cos_angle := Real.cos angle
  (r^2 * height * Real.pi / 2) * cos_angle

theorem volume_of_given_wedge :
  volume_of_wedge 20 (Real.pi / 6) = 1732 * Real.pi :=
by {
  -- The proof logic will go here.
  sorry
}

end volume_of_given_wedge_l29_29358


namespace valid_range_of_x_l29_29529

theorem valid_range_of_x (x : ℝ) (h1 : 2 - x ≥ 0) (h2 : x + 1 ≠ 0) : x ≤ 2 ∧ x ≠ -1 :=
sorry

end valid_range_of_x_l29_29529


namespace find_sqrt_abc_sum_l29_29137

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l29_29137


namespace square_inequality_l29_29455

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality_l29_29455


namespace range_of_a_minus_b_l29_29546

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : 1 < b) (h₄ : b < 3) : 
  -4 < a - b ∧ a - b < 0 := by
  sorry

end range_of_a_minus_b_l29_29546


namespace Monica_books_next_year_l29_29743

-- Definitions for conditions
def books_last_year : ℕ := 25
def books_this_year (bl_year: ℕ) : ℕ := 3 * bl_year
def books_next_year (bt_year: ℕ) : ℕ := 3 * bt_year + 7

-- Theorem statement
theorem Monica_books_next_year : books_next_year (books_this_year books_last_year) = 232 :=
by
  sorry

end Monica_books_next_year_l29_29743


namespace ratio_of_linear_combination_l29_29722

theorem ratio_of_linear_combination (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 :=
by
  sorry

end ratio_of_linear_combination_l29_29722


namespace percentage_increase_l29_29383

theorem percentage_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) : ((M - N) / N) * 100 = P :=
by
  sorry

end percentage_increase_l29_29383


namespace polar_to_rectangular_l29_29832

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 3 * Real.sqrt 2 → θ = (3 * Real.pi) / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (-3, 3) :=
by
  intro r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l29_29832


namespace sum_of_x_intersections_l29_29889

theorem sum_of_x_intersections (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 + x) = f (3 - x))
  (m : ℕ) (xs : Fin m → ℝ) (ys : Fin m → ℝ)
  (h_intersection : ∀ i : Fin m, f (xs i) = |(xs i)^2 - 4 * (xs i) - 3|) :
  (Finset.univ.sum fun i => xs i) = 2 * m :=
by
  sorry

end sum_of_x_intersections_l29_29889


namespace Thabo_books_l29_29297

theorem Thabo_books :
  ∃ (H : ℕ), ∃ (P : ℕ), ∃ (F : ℕ), 
  (H + P + F = 220) ∧ 
  (P = H + 20) ∧ 
  (F = 2 * P) ∧ 
  (H = 40) :=
by
  -- Here will be the formal proof, which is not required for this task.
  sorry

end Thabo_books_l29_29297


namespace elephant_weight_l29_29258

theorem elephant_weight :
  ∃ (w : ℕ), ∀ i : Fin 15, (i.val ≤ 13 → w + 2 * w = 15000) ∧ ((0:ℕ) < w → w = 5000) :=
by
  sorry

end elephant_weight_l29_29258


namespace average_cost_price_per_meter_l29_29384

noncomputable def average_cost_per_meter (total_cost total_meters : ℝ) : ℝ :=
  total_cost / total_meters

theorem average_cost_price_per_meter :
  let silk_cost := 416.25
  let silk_meters := 9.25
  let cotton_cost := 337.50
  let cotton_meters := 7.5
  let wool_cost := 378.0
  let wool_meters := 6.0
  let total_cost := silk_cost + cotton_cost + wool_cost
  let total_meters := silk_meters + cotton_meters + wool_meters
  average_cost_per_meter total_cost total_meters = 49.75 := by
  sorry

end average_cost_price_per_meter_l29_29384


namespace no_three_integers_exist_l29_29061

theorem no_three_integers_exist (x y z : ℤ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  ((x^2 - 1) % y = 0) ∧ ((x^2 - 1) % z = 0) ∧
  ((y^2 - 1) % x = 0) ∧ ((y^2 - 1) % z = 0) ∧
  ((z^2 - 1) % x = 0) ∧ ((z^2 - 1) % y = 0) → false :=
by
  sorry

end no_three_integers_exist_l29_29061


namespace common_tangent_curves_l29_29320

theorem common_tangent_curves (s t a : ℝ) (e : ℝ) (he : e > 0) :
  (t = (1 / (2 * e)) * s^2) →
  (t = a * Real.log s) →
  (s / e = a / s) →
  a = 1 :=
by
  intro h1 h2 h3
  sorry

end common_tangent_curves_l29_29320


namespace Nick_riding_speed_l29_29365

theorem Nick_riding_speed (Alan_speed Maria_ratio Nick_ratio : ℝ) 
(h1 : Alan_speed = 6) (h2 : Maria_ratio = 3/4) (h3 : Nick_ratio = 4/3) : 
Nick_ratio * (Maria_ratio * Alan_speed) = 6 := 
by 
  sorry

end Nick_riding_speed_l29_29365


namespace matrix_determinant_equiv_l29_29745

variable {x y z w : ℝ}

theorem matrix_determinant_equiv (h : x * w - y * z = 7) :
    (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by
    sorry

end matrix_determinant_equiv_l29_29745


namespace monotonically_decreasing_implies_a_geq_3_l29_29504

noncomputable def f (x a : ℝ): ℝ := x^3 - a * x - 1

theorem monotonically_decreasing_implies_a_geq_3 : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x a ≤ f x 3) →
  a ≥ 3 := 
sorry

end monotonically_decreasing_implies_a_geq_3_l29_29504


namespace min_product_of_three_l29_29557

theorem min_product_of_three :
  ∀ (list : List Int), 
    list = [-9, -7, -1, 2, 4, 6, 8] →
    ∃ (a b c : Int), a ∈ list ∧ b ∈ list ∧ c ∈ list ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ (x y z : Int), x ∈ list → y ∈ list → z ∈ list → x ≠ y → y ≠ z → x ≠ z → x * y * z ≥ a * b * c) ∧
    a * b * c = -432 :=
by
  sorry

end min_product_of_three_l29_29557


namespace mean_equality_l29_29552

theorem mean_equality (y : ℝ) : 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 :=
by
  intros h
  sorry

end mean_equality_l29_29552


namespace solution_set_l29_29927

theorem solution_set :
  {p : ℝ × ℝ | (p.1^2 + 3 * p.1 * p.2 + 2 * p.2^2) * (p.1^2 * p.2^2 - 1) = 0} =
  {p : ℝ × ℝ | p.2 = -p.1 / 2} ∪
  {p : ℝ × ℝ | p.2 = -p.1} ∪
  {p : ℝ × ℝ | p.2 = -1 / p.1} ∪
  {p : ℝ × ℝ | p.2 = 1 / p.1} :=
by sorry

end solution_set_l29_29927


namespace UPOMB_position_l29_29963

-- Define the set of letters B, M, O, P, and U
def letters : List Char := ['B', 'M', 'O', 'P', 'U']

-- Define the word UPOMB
def word := "UPOMB"

-- Define a function that calculates the factorial of a number
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the position of a word in the alphabetical permutations of a list of characters
def word_position (w : String) (chars : List Char) : Nat :=
  let rec aux (w : List Char) (remaining : List Char) : Nat :=
    match w with
    | [] => 1
    | c :: cs =>
      let before_count := remaining.filter (· < c) |>.length
      let rest_count := factorial (remaining.length - 1)
      before_count * rest_count + aux cs (remaining.erase c)
  aux w.data chars

-- The desired theorem statement
theorem UPOMB_position : word_position word letters = 119 := by
  sorry

end UPOMB_position_l29_29963


namespace count_valid_n_l29_29109

theorem count_valid_n : 
  ∃ (count : ℕ), count = 88 ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2000 ∧ 
   (∃ (a b : ℤ), a + b = -2 ∧ a * b = -n) ↔ 
   ∃ m, 1 ≤ m ∧ m ≤ 2000 ∧ (∃ a, a * (a + 2) = m)) := 
sorry

end count_valid_n_l29_29109


namespace probability_diff_colors_l29_29138

theorem probability_diff_colors (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_balls = 4 ∧ white_balls = 3 ∧ black_balls = 1 ∧ drawn_balls = 2 ∧ 
  total_outcomes = Nat.choose 4 2 ∧ favorable_outcomes = Nat.choose 3 1 * Nat.choose 1 1
  → favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_diff_colors_l29_29138


namespace maximum_distance_value_of_m_l29_29394

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := y = m * x - m - 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the problem statement
theorem maximum_distance_value_of_m :
  ∃ (m : ℝ), (∀ x y : ℝ, circle_eq x y → ∃ P : ℝ × ℝ, line_eq m P.fst P.snd) →
  m = -0.5 :=
sorry

end maximum_distance_value_of_m_l29_29394


namespace a_plus_b_l29_29328

open Complex

theorem a_plus_b (a b : ℝ) (h : (a - I) * I = -b + 2 * I) : a + b = 1 := by
  sorry

end a_plus_b_l29_29328


namespace find_number_l29_29354

theorem find_number :
  ∃ x : ℚ, x * (-1/2) = 1 ↔ x = -2 := 
sorry

end find_number_l29_29354


namespace initial_number_correct_l29_29126

-- Define the relevant values
def x : ℝ := 53.33
def initial_number : ℝ := 319.98

-- Define the conditions in Lean with appropriate constraints
def conditions (n : ℝ) (x : ℝ) : Prop :=
  x = n / 2 / 3

-- Theorem stating that 319.98 divided by 2 and then by 3 results in 53.33
theorem initial_number_correct : conditions initial_number x :=
by
  unfold conditions
  sorry

end initial_number_correct_l29_29126


namespace combined_weight_of_three_l29_29547

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l29_29547


namespace acuteAnglesSum_l29_29129

theorem acuteAnglesSum (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2) (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end acuteAnglesSum_l29_29129


namespace find_S_9_l29_29800

variable (a : ℕ → ℝ)

def arithmetic_sum_9 (S_9 : ℝ) : Prop :=
  (a 1 + a 3 + a 5 = 39) ∧ (a 5 + a 7 + a 9 = 27) ∧ (S_9 = (9 * (a 3 + a 7)) / 2)

theorem find_S_9 
  (h1 : a 1 + a 3 + a 5 = 39)
  (h2 : a 5 + a 7 + a 9 = 27) :
  ∃ S_9, arithmetic_sum_9 a S_9 ∧ S_9 = 99 := 
by
  sorry

end find_S_9_l29_29800


namespace total_legs_of_animals_l29_29721

def num_kangaroos := 23
def num_goats := 3 * num_kangaroos
def legs_per_kangaroo := 2
def legs_per_goat := 4

def total_legs := (num_kangaroos * legs_per_kangaroo) + (num_goats * legs_per_goat)

theorem total_legs_of_animals : total_legs = 322 := by
  sorry

end total_legs_of_animals_l29_29721


namespace number_of_petri_dishes_l29_29780

def germs_in_lab : ℕ := 3700
def germs_per_dish : ℕ := 25
def num_petri_dishes : ℕ := germs_in_lab / germs_per_dish

theorem number_of_petri_dishes : num_petri_dishes = 148 :=
by
  sorry

end number_of_petri_dishes_l29_29780


namespace problem_l29_29929

theorem problem (a b : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 2 → ax^2 - bx + 2 < 0) : a + b = 4 :=
sorry

end problem_l29_29929


namespace range_of_independent_variable_l29_29821

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y : ℝ, y = 2 * x / (x - 1)) ↔ x ≠ 1 :=
by sorry

end range_of_independent_variable_l29_29821


namespace average_of_remaining_numbers_l29_29327

theorem average_of_remaining_numbers (S : ℕ) (h1 : S = 12 * 90) :
  ((S - 65 - 75 - 85) / 9) = 95 :=
by
  sorry

end average_of_remaining_numbers_l29_29327


namespace least_multiple_of_seven_not_lucky_is_14_l29_29122

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven_not_lucky (n : ℕ) : Prop :=
  n % 7 = 0 ∧ ¬ is_lucky_integer n

theorem least_multiple_of_seven_not_lucky_is_14 : 
  ∃ n : ℕ, is_multiple_of_seven_not_lucky n ∧ ∀ m, (is_multiple_of_seven_not_lucky m → n ≤ m) :=
⟨ 14, 
  by {
    -- Proof is provided here, but for now, we use "sorry"
    sorry
  }⟩

end least_multiple_of_seven_not_lucky_is_14_l29_29122


namespace number_of_girls_l29_29635

theorem number_of_girls
  (B G : ℕ)
  (h1 : B = (8 * G) / 5)
  (h2 : B + G = 351) :
  G = 135 :=
sorry

end number_of_girls_l29_29635


namespace cost_of_5_dozen_l29_29923

noncomputable def price_per_dozen : ℝ :=
  24 / 3

noncomputable def cost_before_tax (num_dozen : ℝ) : ℝ :=
  num_dozen * price_per_dozen

noncomputable def cost_after_tax (num_dozen : ℝ) : ℝ :=
  (1 + 0.10) * cost_before_tax num_dozen

theorem cost_of_5_dozen :
  cost_after_tax 5 = 44 := 
sorry

end cost_of_5_dozen_l29_29923


namespace value_of_b_l29_29621

noncomputable def k := 675

theorem value_of_b (a b : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) (h4 : a = -12) :
  b = -56.25 := by
  sorry

end value_of_b_l29_29621


namespace passengers_from_other_continents_l29_29512

theorem passengers_from_other_continents :
  (∀ (n NA EU AF AS : ℕ),
     NA = n / 4 →
     EU = n / 8 →
     AF = n / 12 →
     AS = n / 6 →
     96 = n →
     n - (NA + EU + AF + AS) = 36) :=
by
  sorry

end passengers_from_other_continents_l29_29512


namespace neg_prop_p_l29_29859

theorem neg_prop_p :
  (¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end neg_prop_p_l29_29859


namespace find_values_of_a_l29_29433

theorem find_values_of_a :
  ∃ (a : ℝ), 
    (∀ x y, (|y + 2| + |x - 11| - 3) * (x^2 + y^2 - 13) = 0 ∧ 
             (x - 5)^2 + (y + 2)^2 = a) ↔ 
    a = 9 ∨ a = 42 + 2 * Real.sqrt 377 :=
sorry

end find_values_of_a_l29_29433


namespace min_value_f_l29_29087

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end min_value_f_l29_29087


namespace circle_represents_circle_iff_a_nonzero_l29_29276

-- Define the equation given in the problem
def circleEquation (a x y : ℝ) : Prop :=
  a*x^2 + a*y^2 - 4*(a-1)*x + 4*y = 0

-- State the required theorem
theorem circle_represents_circle_iff_a_nonzero (a : ℝ) :
  (∃ c : ℝ, ∃ h k : ℝ, ∀ x y : ℝ, circleEquation a x y ↔ (x - h)^2 + (y - k)^2 = c)
  ↔ a ≠ 0 :=
by
  sorry

end circle_represents_circle_iff_a_nonzero_l29_29276


namespace frank_candy_bags_l29_29341

theorem frank_candy_bags (total_candies : ℕ) (candies_per_bag : ℕ) (bags : ℕ) 
  (h1 : total_candies = 22) (h2 : candies_per_bag = 11) : bags = 2 :=
by
  sorry

end frank_candy_bags_l29_29341


namespace y_equals_px_div_5x_p_l29_29429

variable (p x y : ℝ)

theorem y_equals_px_div_5x_p (h : p = 5 * x * y / (x - y)) : y = p * x / (5 * x + p) :=
sorry

end y_equals_px_div_5x_p_l29_29429


namespace land_per_person_l29_29017

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l29_29017


namespace students_not_good_at_either_l29_29755

theorem students_not_good_at_either (total good_at_english good_at_chinese both_good : ℕ) 
(h₁ : total = 45) 
(h₂ : good_at_english = 35) 
(h₃ : good_at_chinese = 31) 
(h₄ : both_good = 24) : total - (good_at_english + good_at_chinese - both_good) = 3 :=
by sorry

end students_not_good_at_either_l29_29755


namespace inverse_36_mod_53_l29_29440

theorem inverse_36_mod_53 (h : 17 * 26 ≡ 1 [MOD 53]) : 36 * 27 ≡ 1 [MOD 53] :=
sorry

end inverse_36_mod_53_l29_29440


namespace bob_walking_rate_is_12_l29_29096

-- Definitions for the problem
def yolanda_distance := 24
def yolanda_rate := 3
def bob_distance_when_met := 12
def time_yolanda_walked := 2

-- The theorem we need to prove
theorem bob_walking_rate_is_12 : 
  (bob_distance_when_met / (time_yolanda_walked - 1) = 12) :=
by sorry

end bob_walking_rate_is_12_l29_29096


namespace third_height_of_triangle_l29_29648

theorem third_height_of_triangle 
  (a b c ha hb hc : ℝ)
  (h_abc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_heights : ∃ (h1 h2 h3 : ℕ), h1 = 3 ∧ h2 = 10 ∧ h3 ≠ h1 ∧ h3 ≠ h2) :
  ∃ (h3 : ℕ), h3 = 4 :=
by
  sorry

end third_height_of_triangle_l29_29648


namespace union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l29_29092

open Set

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | x^2 - 12*x + 20 < 0 }
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem union_of_A_and_B :
  A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
sorry

theorem complement_of_A_intersect_B :
  ((univ \ A) ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
sorry

theorem intersection_of_A_and_C (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a > 3 :=
sorry

end union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l29_29092


namespace cube_faces_sum_39_l29_29350

theorem cube_faces_sum_39 (a b c d e f g h : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0)
    (vertex_sum : (a*e*b*h + a*e*c*h + a*f*b*h + a*f*c*h + d*e*b*h + d*e*c*h + d*f*b*h + d*f*c*h) = 2002) :
    (a + b + c + d + e + f + g + h) = 39 := 
sorry

end cube_faces_sum_39_l29_29350


namespace find_prime_between_20_and_35_with_remainder_7_l29_29667

theorem find_prime_between_20_and_35_with_remainder_7 : 
  ∃ p : ℕ, Nat.Prime p ∧ 20 ≤ p ∧ p ≤ 35 ∧ p % 11 = 7 ∧ p = 29 := 
by 
  sorry

end find_prime_between_20_and_35_with_remainder_7_l29_29667


namespace xy_system_l29_29481

theorem xy_system (x y : ℚ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 :=
by
  sorry

end xy_system_l29_29481


namespace simple_interest_calculation_l29_29179

theorem simple_interest_calculation (P R T : ℝ) (H₁ : P = 8925) (H₂ : R = 9) (H₃ : T = 5) : 
  P * R * T / 100 = 4016.25 :=
by
  sorry

end simple_interest_calculation_l29_29179


namespace find_a_l29_29709

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x / (3 * x + 4)

theorem find_a (a : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : (f a) (f a x) = x → a = -2 := by
  unfold f
  -- Remaining proof steps skipped
  sorry

end find_a_l29_29709


namespace joe_cars_after_getting_more_l29_29947

-- Defining the initial conditions as Lean variables
def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

-- Stating the proof problem
theorem joe_cars_after_getting_more : initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_after_getting_more_l29_29947


namespace men_in_second_group_l29_29794

theorem men_in_second_group (M : ℕ) (h1 : 16 * 30 = 480) (h2 : M * 24 = 480) : M = 20 :=
by
  sorry

end men_in_second_group_l29_29794


namespace matches_C_won_l29_29575

variable (A_wins B_wins D_wins total_matches wins_C : ℕ)

theorem matches_C_won 
  (hA : A_wins = 3)
  (hB : B_wins = 1)
  (hD : D_wins = 0)
  (htot : total_matches = 6)
  (h_sum_wins: A_wins + B_wins + D_wins + wins_C = total_matches)
  : wins_C = 2 :=
by
  sorry

end matches_C_won_l29_29575


namespace intersection_A_B_l29_29134

-- Define the conditions of set A and B using the given inequalities and constraints
def set_A : Set ℤ := {x | -2 < x ∧ x < 3}
def set_B : Set ℤ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the proof problem translating conditions and question to Lean
theorem intersection_A_B : (set_A ∩ set_B) = {0, 1, 2} := by
  sorry

end intersection_A_B_l29_29134


namespace abcd_product_l29_29975

noncomputable def A := (Real.sqrt 3000 + Real.sqrt 3001)
noncomputable def B := (-Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def C := (Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def D := (Real.sqrt 3001 - Real.sqrt 3000)

theorem abcd_product :
  A * B * C * D = -1 :=
by
  sorry

end abcd_product_l29_29975


namespace find_a_l29_29443

theorem find_a (a : ℝ) (h : a * (1 : ℝ)^2 - 6 * 1 + 3 = 0) : a = 3 :=
by
  sorry

end find_a_l29_29443


namespace red_trace_larger_sphere_area_l29_29916

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end red_trace_larger_sphere_area_l29_29916


namespace complete_the_square_b_26_l29_29169

theorem complete_the_square_b_26 :
  ∃ (a b : ℝ), (∀ x : ℝ, x^2 + 10 * x - 1 = 0 ↔ (x + a)^2 = b) ∧ b = 26 :=
sorry

end complete_the_square_b_26_l29_29169


namespace find_a_2016_l29_29618

theorem find_a_2016 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 1, a (n + 1) = 3 * S n)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)):
  a 2016 = 3 * 4 ^ 2014 := 
by 
  sorry

end find_a_2016_l29_29618


namespace chairs_built_in_10_days_l29_29168

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l29_29168


namespace fraction_bounds_l29_29434

theorem fraction_bounds (x y : ℝ) (h : x^2 * y^2 + x * y + 1 = 3 * y^2) : 
0 ≤ (y - x) / (x + 4 * y) ∧ (y - x) / (x + 4 * y) ≤ 4 := 
sorry

end fraction_bounds_l29_29434


namespace hydropump_output_l29_29556

theorem hydropump_output :
  ∀ (rate : ℕ) (time_hours : ℚ), 
    rate = 600 → 
    time_hours = 1.5 → 
    rate * time_hours = 900 :=
by
  intros rate time_hours rate_cond time_cond 
  sorry

end hydropump_output_l29_29556


namespace part1_69_part1_97_not_part2_difference_numbers_in_range_l29_29280

def is_difference_number (n : ℕ) : Prop :=
  (n % 7 = 6) ∧ (n % 5 = 4)

theorem part1_69 : is_difference_number 69 :=
sorry

theorem part1_97_not : ¬is_difference_number 97 :=
sorry

theorem part2_difference_numbers_in_range :
  {n : ℕ | is_difference_number n ∧ 500 < n ∧ n < 600} = {524, 559, 594} :=
sorry

end part1_69_part1_97_not_part2_difference_numbers_in_range_l29_29280


namespace recommended_water_intake_l29_29411

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end recommended_water_intake_l29_29411


namespace find_a_from_conditions_l29_29990

theorem find_a_from_conditions (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 9) 
  (h3 : c = 4) : 
  a = -1 := 
by 
  sorry

end find_a_from_conditions_l29_29990


namespace probability_of_disease_given_positive_test_l29_29987

-- Define the probabilities given in the problem
noncomputable def pr_D : ℝ := 1 / 1000
noncomputable def pr_Dc : ℝ := 1 - pr_D
noncomputable def pr_T_given_D : ℝ := 1
noncomputable def pr_T_given_Dc : ℝ := 0.05

-- Define the total probability of a positive test using the law of total probability
noncomputable def pr_T := 
  pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Using Bayes' theorem
noncomputable def pr_D_given_T := 
  pr_T_given_D * pr_D / pr_T

-- Theorem to prove the desired probability
theorem probability_of_disease_given_positive_test : 
  pr_D_given_T = 1 / 10 :=
by
  sorry

end probability_of_disease_given_positive_test_l29_29987


namespace problem_l29_29025

variable {R : Type} [Field R]

def f1 (a b c d : R) : R := a + b + c + d
def f2 (a b c d : R) : R := (1 / a) + (1 / b) + (1 / c) + (1 / d)
def f3 (a b c d : R) : R := (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) + (1 / (1 - d))

theorem problem (a b c d : R) (h1 : f1 a b c d = 2) (h2 : f2 a b c d = 2) : f3 a b c d = 2 :=
by sorry

end problem_l29_29025


namespace revenue_difference_l29_29854

theorem revenue_difference {x z : ℕ} (hx : 10 ≤ x ∧ x ≤ 96) (hz : z = x + 3) :
  1000 * z + 10 * x - (1000 * x + 10 * z) = 2920 :=
by
  sorry

end revenue_difference_l29_29854


namespace negative_subtraction_result_l29_29052

theorem negative_subtraction_result : -2 - 1 = -3 := 
by
  -- The proof is not required by the prompt, so we use "sorry" to indicate the unfinished proof.
  sorry

end negative_subtraction_result_l29_29052


namespace power_of_3_l29_29632

theorem power_of_3 {y : ℕ} (h : 3^y = 81) : 3^(y + 3) = 2187 :=
by
  sorry

end power_of_3_l29_29632


namespace ethanol_in_full_tank_l29_29343

theorem ethanol_in_full_tank:
  ∀ (capacity : ℕ) (vol_A : ℕ) (vol_B : ℕ) (eth_A_perc : ℝ) (eth_B_perc : ℝ) (eth_A : ℝ) (eth_B : ℝ),
  capacity = 208 →
  vol_A = 82 →
  vol_B = (capacity - vol_A) →
  eth_A_perc = 0.12 →
  eth_B_perc = 0.16 →
  eth_A = vol_A * eth_A_perc →
  eth_B = vol_B * eth_B_perc →
  eth_A + eth_B = 30 :=
by
  intros capacity vol_A vol_B eth_A_perc eth_B_perc eth_A eth_B h1 h2 h3 h4 h5 h6 h7
  sorry

end ethanol_in_full_tank_l29_29343


namespace value_of_a2012_l29_29347

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem value_of_a2012 (a : ℕ → ℤ) (h : seq a) : a 2012 = 2012 * 2011 :=
by 
  sorry

end value_of_a2012_l29_29347


namespace ratio_diagonals_of_squares_l29_29219

variable (d₁ d₂ : ℝ)

theorem ratio_diagonals_of_squares (h : ∃ k : ℝ, d₂ = k * d₁) (h₁ : 1 < k ∧ k < 9) : 
  (∃ k : ℝ, 4 * (d₂ / Real.sqrt 2) = k * 4 * (d₁ / Real.sqrt 2)) → k = 5 := by
  sorry

end ratio_diagonals_of_squares_l29_29219


namespace complex_cube_root_identity_l29_29834

theorem complex_cube_root_identity (a b c : ℂ) (ω : ℂ)
  (h1 : ω^3 = 1)
  (h2 : 1 + ω + ω^2 = 0) :
  (a + b * ω + c * ω^2) * (a + b * ω^2 + c * ω) = a^2 + b^2 + c^2 - ab - ac - bc :=
by
  sorry

end complex_cube_root_identity_l29_29834


namespace coin_exchange_proof_l29_29566

/-- Prove the coin combination that Petya initially had -/
theorem coin_exchange_proof (x y z : ℕ) (hx : 20 * x + 15 * y + 10 * z = 125) : x = 0 ∧ y = 1 ∧ z = 11 :=
by
  sorry

end coin_exchange_proof_l29_29566


namespace sin_cos_special_l29_29527

def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem sin_cos_special (x : ℝ) : 
  special_operation (Real.sin (x / 12)) (Real.cos (x / 12)) = -(1 + 2 * Real.sqrt 3) / 4 :=
  sorry

end sin_cos_special_l29_29527


namespace expression_simplifies_to_36_l29_29917

theorem expression_simplifies_to_36 (x : ℝ) : (x + 1)^2 + 2 * (x + 1) * (5 - x) + (5 - x)^2 = 36 :=
by
  sorry

end expression_simplifies_to_36_l29_29917


namespace distance_between_A_and_B_l29_29446

-- Definitions according to the problem's conditions
def speed_train_A : ℕ := 50
def speed_train_B : ℕ := 60
def distance_difference : ℕ := 100

-- The main theorem statement to prove
theorem distance_between_A_and_B
  (x : ℕ) -- x is the distance traveled by the first train
  (distance_train_A := x)
  (distance_train_B := x + distance_difference)
  (total_distance := distance_train_A + distance_train_B)
  (meet_condition : distance_train_A / speed_train_A = distance_train_B / speed_train_B) :
  total_distance = 1100 := 
sorry

end distance_between_A_and_B_l29_29446


namespace total_votes_l29_29696

theorem total_votes (total_votes : ℕ) (brenda_votes : ℕ) (fraction : ℚ) (h : brenda_votes = fraction * total_votes) (h_fraction : fraction = 1 / 5) (h_brenda : brenda_votes = 15) : 
  total_votes = 75 := 
by
  sorry

end total_votes_l29_29696


namespace randy_trip_total_distance_l29_29058

-- Definition of the problem condition
def randy_trip_length (x : ℝ) : Prop :=
  x / 3 + 20 + x / 5 = x

-- The total length of Randy's trip
theorem randy_trip_total_distance : ∃ x : ℝ, randy_trip_length x ∧ x = 300 / 7 :=
by
  sorry

end randy_trip_total_distance_l29_29058


namespace geometric_sequence_common_ratio_l29_29449

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 3 * a 2 - 5 * a 1)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a n < a (n + 1))
  (h4 : ∀ n, a (n + 1) = a n * q) : 
  q = 5 :=
  sorry

end geometric_sequence_common_ratio_l29_29449


namespace water_jugs_problem_l29_29163

-- Definitions based on the conditions
variables (m n : ℕ) (relatively_prime_m_n : Nat.gcd m n = 1)
variables (k : ℕ) (hk : 1 ≤ k ∧ k ≤ m + n)

-- Statement of the theorem
theorem water_jugs_problem : 
    ∃ (x y z : ℕ), 
    (x = m ∨ x = n ∨ x = m + n) ∧ 
    (y = m ∨ y = n ∨ y = m + n) ∧ 
    (z = m ∨ z = n ∨ z = m + n) ∧ 
    (x ≤ m + n) ∧ 
    (y ≤ m + n) ∧ 
    (z ≤ m + n) ∧ 
    x + y + z = m + n ∧ 
    (x = k ∨ y = k ∨ z = k) :=
sorry

end water_jugs_problem_l29_29163


namespace not_always_possible_to_predict_winner_l29_29608

def football_championship (teams : Fin 16 → ℕ) : Prop :=
  ∃ i j : Fin 16, i ≠ j ∧ teams i = teams j ∧
  ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧
               teams (pairs k).fst ≠ teams (pairs k).snd) ∨
  ∃ k : Fin 8, (pairs k).fst = i ∧ (pairs k).snd = j

theorem not_always_possible_to_predict_winner :
  ∀ teams : Fin 16 → ℕ, (∃ i j : Fin 16, i ≠ j ∧ teams i = teams j) →
  ∃ pairs : Fin 16 → Fin 16 × Fin 16,
  (∃ k : Fin 8, teams (pairs k).fst = 15 ∧ teams (pairs k).snd = 15) ↔
  ¬ ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧ teams (pairs k).fst ≠ teams (pairs k).snd) :=
by
  sorry

end not_always_possible_to_predict_winner_l29_29608


namespace integer_solutions_range_l29_29124

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l29_29124


namespace trig_expression_simplify_l29_29312

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l29_29312


namespace mary_principal_amount_l29_29633

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount_l29_29633


namespace sector_area_l29_29424

theorem sector_area
  (r : ℝ) (s : ℝ) (h_r : r = 1) (h_s : s = 1) : 
  (1 / 2) * r * s = 1 / 2 := by
  sorry

end sector_area_l29_29424


namespace stickers_distribution_l29_29920

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end stickers_distribution_l29_29920


namespace two_numbers_and_sum_l29_29717

theorem two_numbers_and_sum (x y : ℕ) (hx : x * y = 18) (hy : x - y = 4) : x + y = 10 :=
sorry

end two_numbers_and_sum_l29_29717


namespace average_speed_is_70_l29_29583

theorem average_speed_is_70 
  (distance1 distance2 : ℕ) (time1 time2 : ℕ)
  (h1 : distance1 = 80) (h2 : distance2 = 60)
  (h3 : time1 = 1) (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 70 := 
by 
  sorry

end average_speed_is_70_l29_29583


namespace find_missing_number_l29_29892

theorem find_missing_number (x : ℝ)
  (h1 : (x + 42 + 78 + 104) / 4 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) :
  x = 74 :=
sorry

end find_missing_number_l29_29892


namespace no_solution_equation_l29_29018

theorem no_solution_equation (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) = (x - m) / (x - 8) → false) ↔ m = 7 :=
by
  sorry

end no_solution_equation_l29_29018


namespace students_play_football_l29_29450

theorem students_play_football (total_students : ℕ) (C : ℕ) (B : ℕ) (neither : ℕ) (F : ℕ)
  (h1 : total_students = 460)
  (h2 : C = 175)
  (h3 : B = 90)
  (h4 : neither = 50)
  (h5 : total_students = neither + F + C - B) : 
  F = 325 :=
by 
  sorry

end students_play_football_l29_29450


namespace sin_pi_over_six_l29_29881

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 :=
sorry

end sin_pi_over_six_l29_29881


namespace smallest_logarithmic_term_l29_29582

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_logarithmic_term (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_interval : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) := 
by
  sorry

end smallest_logarithmic_term_l29_29582


namespace length_of_AB_l29_29505

noncomputable def ratio3to5 (AP PB : ℝ) : Prop := AP / PB = 3 / 5
noncomputable def ratio4to5 (AQ QB : ℝ) : Prop := AQ / QB = 4 / 5
noncomputable def pointDistances (P Q : ℝ) : Prop := P - Q = 3

theorem length_of_AB (A B P Q : ℝ) (P_on_AB : P > A ∧ P < B) (Q_on_AB : Q > A ∧ Q < B)
  (middle_side : P < (A + B) / 2 ∧ Q < (A + B) / 2)
  (h1 : ratio3to5 (P - A) (B - P))
  (h2 : ratio4to5 (Q - A) (B - Q))
  (h3 : pointDistances P Q) : B - A = 43.2 := 
sorry

end length_of_AB_l29_29505


namespace square_garden_area_l29_29437

theorem square_garden_area (P A : ℕ)
  (h1 : P = 40)
  (h2 : A = 2 * P + 20) :
  A = 100 :=
by
  rw [h1] at h2 -- Substitute h1 (P = 40) into h2 (A = 2P + 20)
  norm_num at h2 -- Normalize numeric expressions in h2
  exact h2 -- Conclude by showing h2 (A = 100) holds

-- The output should be able to build successfully without solving the proof.

end square_garden_area_l29_29437


namespace geometric_sequence_problem_l29_29597

theorem geometric_sequence_problem
  (q : ℝ) (h_q : |q| ≠ 1) (m : ℕ)
  (a : ℕ → ℝ)
  (h_a1 : a 1 = -1)
  (h_am : a m = a 1 * a 2 * a 3 * a 4 * a 5) 
  (h_gseq : ∀ n, a (n + 1) = a n * q) :
  m = 11 :=
by
  sorry

end geometric_sequence_problem_l29_29597


namespace number_of_intersection_points_l29_29976

-- Definitions of the given lines
def line1 (x y : ℝ) : Prop := 6 * y - 4 * x = 2
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := -4 * x + 6 * y = 3

-- Definitions of the intersection points
def intersection1 (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def intersection2 (x y : ℝ) : Prop := line2 x y ∧ line3 x y

-- Definition of the problem
theorem number_of_intersection_points : 
  (∃ x y : ℝ, intersection1 x y) ∧
  (∃ x y : ℝ, intersection2 x y) ∧
  (¬ ∃ x y : ℝ, line1 x y ∧ line3 x y) →
  (∃ z : ℕ, z = 2) :=
sorry

end number_of_intersection_points_l29_29976


namespace rhombus_perimeter_l29_29785

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) 
  (h3 : ∀ {x y : ℝ}, (x = d1 / 2 ∧ y = d2 / 2) → (x^2 + y^2 = (d1 / 2)^2 + (d2 / 2)^2)) : 
  4 * (Real.sqrt ((d1/2)^2 + (d2/2)^2)) = 156 :=
by 
  rw [h1, h2]
  simp
  sorry

end rhombus_perimeter_l29_29785


namespace point_M_coordinates_l29_29959

open Real

theorem point_M_coordinates (θ : ℝ) (h_tan : tan θ = -4 / 3) (h_theta : π / 2 < θ ∧ θ < π) :
  let x := 5 * cos θ
  let y := 5 * sin θ
  (x, y) = (-3, 4) := 
by 
  sorry

end point_M_coordinates_l29_29959


namespace edward_total_money_l29_29230

-- define the amounts made and spent
def money_made_spring : ℕ := 2
def money_made_summer : ℕ := 27
def money_spent_supplies : ℕ := 5

-- total money left is calculated by adding what he made and subtracting the expenses
def total_money_end (m_spring m_summer m_supplies : ℕ) : ℕ :=
  m_spring + m_summer - m_supplies

-- the theorem to prove
theorem edward_total_money :
  total_money_end money_made_spring money_made_summer money_spent_supplies = 24 :=
by
  sorry

end edward_total_money_l29_29230


namespace find_7c_plus_7d_l29_29564

noncomputable def f (c d x : ℝ) : ℝ := c * x + d
noncomputable def h (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 1

theorem find_7c_plus_7d (c d : ℝ) (h_def : ∀ x, h x = f_inv x - 5) (f_def : ∀ x, f c d x = c * x + d) (f_inv_def : ∀ x, f_inv x = 7 * x - 1) : 7 * c + 7 * d = 2 := by
  sorry

end find_7c_plus_7d_l29_29564


namespace mixing_ratios_indeterminacy_l29_29259

theorem mixing_ratios_indeterminacy (x : ℝ) (a b : ℝ) (h1 : a + b = 50) (h2 : 0.40 * a + (x / 100) * b = 25) : False :=
sorry

end mixing_ratios_indeterminacy_l29_29259


namespace p_p_values_l29_29498

def p (x y : ℤ) : ℤ :=
if 0 ≤ x ∧ 0 ≤ y then x + 2*y
else if x < 0 ∧ y < 0 then x - 3*y
else 4*x + y

theorem p_p_values : p (p 2 (-2)) (p (-3) (-1)) = 6 :=
by
  sorry

end p_p_values_l29_29498


namespace circle_inscribed_angles_l29_29698

theorem circle_inscribed_angles (O : Type) (circle : Set O) (A B C D E F G H I J K L : O) 
  (P : ℕ) (n : ℕ) (x_deg_sum y_deg_sum : ℝ)  
  (h1 : n = 12) 
  (h2 : x_deg_sum = 45) 
  (h3 : y_deg_sum = 75) :
  x_deg_sum + y_deg_sum = 120 :=
by
  /- Proof steps are not required -/
  apply sorry

end circle_inscribed_angles_l29_29698


namespace trig_identity_example_l29_29787

open Real

noncomputable def tan_alpha_eq_two_tan_pi_fifths (α : ℝ) :=
  tan α = 2 * tan (π / 5)

theorem trig_identity_example (α : ℝ) (h : tan_alpha_eq_two_tan_pi_fifths α) :
  (cos (α - 3 * π / 10) / sin (α - π / 5)) = 3 :=
sorry

end trig_identity_example_l29_29787


namespace combination_10_3_l29_29658

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l29_29658


namespace max_daily_sales_l29_29659

def f (t : ℕ) : ℝ := -2 * t + 200
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30
  else 45

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales : ∃ t, 1 ≤ t ∧ t ≤ 50 ∧ S t = 54600 := 
  sorry

end max_daily_sales_l29_29659


namespace sum_of_squares_l29_29977

theorem sum_of_squares : 
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  (squares.sum = 195) := 
by
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  have h : squares.sum = 195 := sorry
  exact h

end sum_of_squares_l29_29977


namespace total_cost_sandwiches_sodas_l29_29638

theorem total_cost_sandwiches_sodas (cost_per_sandwich cost_per_soda : ℝ) 
  (num_sandwiches num_sodas : ℕ) (discount_rate : ℝ) (total_items : ℕ) :
  cost_per_sandwich = 4 → 
  cost_per_soda = 3 → 
  num_sandwiches = 6 → 
  num_sodas = 7 → 
  discount_rate = 0.10 → 
  total_items = num_sandwiches + num_sodas → 
  total_items > 10 → 
  (num_sandwiches * cost_per_sandwich + num_sodas * cost_per_soda) * (1 - discount_rate) = 40.5 :=
by
  intros
  sorry

end total_cost_sandwiches_sodas_l29_29638


namespace total_amount_is_33_l29_29665

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end total_amount_is_33_l29_29665


namespace largest_possible_b_l29_29182

theorem largest_possible_b (b : ℚ) (h : (3 * b + 7) * (b - 2) = 9 * b) : b ≤ 2 :=
sorry

end largest_possible_b_l29_29182


namespace isosceles_triangle_area_l29_29724

theorem isosceles_triangle_area (x : ℤ) (h1 : x > 2) (h2 : x < 4) 
  (h3 : ∃ (a b : ℤ), a = x ∧ b = 8 - 2 * x ∧ a = b) :
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end isosceles_triangle_area_l29_29724


namespace proof_problem_l29_29507

variable (A B C : ℕ)

-- Defining the conditions
def condition1 : Prop := A + B + C = 700
def condition2 : Prop := B + C = 600
def condition3 : Prop := C = 200

-- Stating the proof problem
theorem proof_problem (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 C) : A + C = 300 :=
sorry

end proof_problem_l29_29507


namespace blue_lights_count_l29_29726

def num_colored_lights := 350
def num_red_lights := 85
def num_yellow_lights := 112
def num_green_lights := 65
def num_blue_lights := num_colored_lights - (num_red_lights + num_yellow_lights + num_green_lights)

theorem blue_lights_count : num_blue_lights = 88 := by
  sorry

end blue_lights_count_l29_29726


namespace smallest_three_digit_divisible_by_3_and_6_l29_29670

theorem smallest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → n ≤ m) ∧ n = 102 := 
by {sorry}

end smallest_three_digit_divisible_by_3_and_6_l29_29670


namespace cost_of_four_books_l29_29310

theorem cost_of_four_books
  (H : 2 * book_cost = 36) :
  4 * book_cost = 72 :=
by
  sorry

end cost_of_four_books_l29_29310


namespace initial_oranges_l29_29357

variable (x : ℕ)
variable (total_oranges : ℕ := 8)
variable (oranges_from_joyce : ℕ := 3)

theorem initial_oranges (h : total_oranges = x + oranges_from_joyce) : x = 5 := by
  sorry

end initial_oranges_l29_29357


namespace quilt_shading_fraction_l29_29316

/-- 
Statement:
Given a quilt block made from nine unit squares, where two unit squares are divided diagonally into triangles, 
and one unit square is divided into four smaller equal squares with one of the smaller squares shaded, 
the fraction of the quilt that is shaded is \( \frac{5}{36} \).
-/
theorem quilt_shading_fraction : 
  let total_area := 9 
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2 
  shaded_area / total_area = 5 / 36 :=
by
  -- Definitions based on conditions
  let total_area := 9
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2
  -- The proof statement (fraction of shaded area)
  have h : shaded_area / total_area = 5 / 36 := sorry
  exact h

end quilt_shading_fraction_l29_29316


namespace solve_for_y_l29_29746

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end solve_for_y_l29_29746


namespace simplify_and_multiply_expression_l29_29090

variable (b : ℝ)

theorem simplify_and_multiply_expression :
  (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 :=
by
  sorry

end simplify_and_multiply_expression_l29_29090


namespace abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l29_29254

theorem abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one (x : ℝ) :
  |x| < 1 → x^3 < 1 ∧ (x^3 < 1 → |x| < 1 → False) :=
by
  sorry

end abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l29_29254


namespace nine_chapters_coins_l29_29367

theorem nine_chapters_coins (a d : ℚ)
  (h1 : (a - 2 * d) + (a - d) = a + (a + d) + (a + 2 * d))
  (h2 : (a - 2 * d) + (a - d) + a + (a + d) + (a + 2 * d) = 5) :
  a - d = 7 / 6 :=
by 
  sorry

end nine_chapters_coins_l29_29367


namespace cakes_served_dinner_l29_29925

def total_cakes_today : Nat := 15
def cakes_served_lunch : Nat := 6

theorem cakes_served_dinner : total_cakes_today - cakes_served_lunch = 9 :=
by
  -- Define what we need to prove
  sorry -- to skip the proof

end cakes_served_dinner_l29_29925


namespace tony_average_time_l29_29419

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l29_29419


namespace susan_strawberries_l29_29047

def strawberries_picked (total_in_basket : ℕ) (handful_size : ℕ) (eats_per_handful : ℕ) : ℕ :=
  let strawberries_per_handful := handful_size - eats_per_handful
  (total_in_basket / strawberries_per_handful) * handful_size

theorem susan_strawberries : strawberries_picked 60 5 1 = 75 := by
  sorry

end susan_strawberries_l29_29047


namespace no_real_b_for_line_to_vertex_of_parabola_l29_29921

theorem no_real_b_for_line_to_vertex_of_parabola : 
  ¬ ∃ b : ℝ, ∃ x : ℝ, y = x + b ∧ y = x^2 + b^2 + 1 :=
by
  sorry

end no_real_b_for_line_to_vertex_of_parabola_l29_29921


namespace floor_of_ten_times_expected_value_of_fourth_largest_l29_29910

theorem floor_of_ten_times_expected_value_of_fourth_largest : 
  let n := 90
  let m := 5
  let k := 4
  let E := (k * (n + 1)) / (m + 1)
  ∀ (X : Fin m → Fin n) (h : ∀ i j : Fin m, i ≠ j → X i ≠ X j), 
  Nat.floor (10 * E) = 606 := 
by
  sorry

end floor_of_ten_times_expected_value_of_fourth_largest_l29_29910


namespace artworks_per_student_in_first_half_l29_29338

theorem artworks_per_student_in_first_half (x : ℕ) (h1 : 10 = 10) (h2 : 20 = 20) (h3 : 5 * x + 5 * 4 = 35) : x = 3 := by
  sorry

end artworks_per_student_in_first_half_l29_29338


namespace sum_of_reciprocals_l29_29704

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + y = 3 * x * y) (h2 : x - y = 2) : (1/x + 1/y) = 4/3 :=
by
  -- Proof omitted
  sorry

end sum_of_reciprocals_l29_29704


namespace even_square_is_even_l29_29515

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l29_29515


namespace min_sum_4410_l29_29697

def min_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem min_sum_4410 :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 4410 ∧ min_sum a b c d = 69 :=
sorry

end min_sum_4410_l29_29697


namespace find_person_10_number_l29_29778

theorem find_person_10_number (n : ℕ) (a : ℕ → ℕ)
  (h1 : n = 15)
  (h2 : 2 * a 10 = a 9 + a 11)
  (h3 : 2 * a 3 = a 2 + a 4)
  (h4 : a 10 = 8)
  (h5 : a 3 = 7) :
  a 10 = 8 := 
by sorry

end find_person_10_number_l29_29778


namespace infinite_coprime_pairs_with_divisibility_l29_29869

theorem infinite_coprime_pairs_with_divisibility :
  ∃ (A : ℕ → ℕ) (B : ℕ → ℕ), (∀ n, gcd (A n) (B n) = 1) ∧
    ∀ n, (A n ∣ (B n)^2 - 5) ∧ (B n ∣ (A n)^2 - 5) :=
sorry

end infinite_coprime_pairs_with_divisibility_l29_29869


namespace netCaloriesConsumedIs1082_l29_29740

-- Given conditions
def caloriesPerCandyBar : ℕ := 347
def candyBarsEatenInAWeek : ℕ := 6
def caloriesBurnedInAWeek : ℕ := 1000

-- Net calories calculation
def netCaloriesInAWeek (calsPerBar : ℕ) (barsPerWeek : ℕ) (calsBurned : ℕ) : ℕ :=
  calsPerBar * barsPerWeek - calsBurned

-- The theorem to prove
theorem netCaloriesConsumedIs1082 :
  netCaloriesInAWeek caloriesPerCandyBar candyBarsEatenInAWeek caloriesBurnedInAWeek = 1082 :=
by
  sorry

end netCaloriesConsumedIs1082_l29_29740


namespace isosceles_triangle_l29_29066

theorem isosceles_triangle 
  (α β γ : ℝ) 
  (a b : ℝ) 
  (h_sum : a + b = (Real.tan (γ / 2)) * (a * (Real.tan α) + b * (Real.tan β)))
  (h_sum_angles : α + β + γ = π) 
  (zero_lt_γ : 0 < γ ∧ γ < π) 
  (zero_lt_α : 0 < α ∧ α < π / 2) 
  (zero_lt_β : 0 < β ∧ β < π / 2) : 
  α = β := 
sorry

end isosceles_triangle_l29_29066


namespace boatman_current_speed_and_upstream_time_l29_29602

variables (v : ℝ) (v_T : ℝ) (t_up : ℝ) (t_total : ℝ) (dist : ℝ) (d1 : ℝ) (d2 : ℝ)

theorem boatman_current_speed_and_upstream_time
  (h1 : dist = 12.5)
  (h2 : d1 = 3)
  (h3 : d2 = 5)
  (h4 : t_total = 8)
  (h5 : ∀ t, t = d1 / (v - v_T))
  (h6 : ∀ t, t = d2 / (v + v_T))
  (h7 : dist / (v - v_T) + dist / (v + v_T) = t_total) :
  v_T = 5 / 6 ∧ t_up = 5 := by
  sorry

end boatman_current_speed_and_upstream_time_l29_29602


namespace find_a_plus_b_l29_29132

-- Definitions for the conditions
variables {a b : ℝ} (i : ℂ)
def imaginary_unit : Prop := i * i = -1

-- Given condition
def given_equation (a b : ℝ) (i : ℂ) : Prop := (a + 2 * i) / i = b + i

-- Theorem statement
theorem find_a_plus_b (h1 : imaginary_unit i) (h2 : given_equation a b i) : a + b = 1 := 
sorry

end find_a_plus_b_l29_29132


namespace f_9_over_2_l29_29104

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l29_29104


namespace box_tape_length_l29_29968

variable (L S : ℕ)
variable (tape_total : ℕ)
variable (num_boxes : ℕ)
variable (square_side : ℕ)

theorem box_tape_length (h1 : num_boxes = 5) (h2 : square_side = 40) (h3 : tape_total = 540) :
  tape_total = 5 * (L + 2 * S) + 2 * 3 * square_side → L = 60 - 2 * S := 
by
  sorry

end box_tape_length_l29_29968


namespace eighth_group_number_correct_stratified_sampling_below_30_correct_l29_29452

noncomputable def systematic_sampling_eighth_group_number 
  (total_employees : ℕ) (sample_size : ℕ) (groups : ℕ) (fifth_group_number : ℕ) : ℕ :=
  let interval := total_employees / groups
  let initial_number := fifth_group_number - 4 * interval
  initial_number + 7 * interval

theorem eighth_group_number_correct :
  systematic_sampling_eighth_group_number 200 40 40 22 = 37 :=
  sorry

noncomputable def stratified_sampling_below_30_persons 
  (total_employees : ℕ) (sample_size : ℕ) (percent_below_30 : ℕ) : ℕ :=
  (percent_below_30 * sample_size) / 100

theorem stratified_sampling_below_30_correct :
  stratified_sampling_below_30_persons 200 40 40 = 16 :=
  sorry

end eighth_group_number_correct_stratified_sampling_below_30_correct_l29_29452


namespace find_x_l29_29653

variable (A B x : ℝ)
variable (h1 : A > 0) (h2 : B > 0)
variable (h3 : A = (x / 100) * B)

theorem find_x : x = 100 * (A / B) :=
by
  sorry

end find_x_l29_29653


namespace calculate_expression_l29_29482

theorem calculate_expression : -1^2021 + 1^2022 = 0 := by
  sorry

end calculate_expression_l29_29482


namespace frequency_of_rolling_six_is_0_point_19_l29_29404

theorem frequency_of_rolling_six_is_0_point_19 :
  ∀ (total_rolls number_six_appeared : ℕ), total_rolls = 100 → number_six_appeared = 19 → 
  (number_six_appeared : ℝ) / (total_rolls : ℝ) = 0.19 := 
by 
  intros total_rolls number_six_appeared h_total_rolls h_number_six_appeared
  sorry

end frequency_of_rolling_six_is_0_point_19_l29_29404


namespace plums_in_basket_l29_29145

theorem plums_in_basket (initial : ℕ) (added : ℕ) (total : ℕ) (h_initial : initial = 17) (h_added : added = 4) : total = 21 := by
  sorry

end plums_in_basket_l29_29145


namespace albania_inequality_l29_29559

variable (a b c r R s : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)
variable (h4 : r > 0)
variable (h5 : R > 0)
variable (h6 : s = (a + b + c) / 2)

theorem albania_inequality :
    1 / (a + b) + 1 / (a + c) + 1 / (b + c) ≤ r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s) :=
sorry

end albania_inequality_l29_29559


namespace arithmetic_sequence_terms_l29_29033

theorem arithmetic_sequence_terms
  (a : ℕ → ℝ)
  (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 20)
  (h2 : a (n-2) + a (n-1) + a n = 130)
  (h3 : (n * (a 1 + a n)) / 2 = 200) :
  n = 8 := 
sorry

end arithmetic_sequence_terms_l29_29033


namespace parallelogram_circumference_l29_29693

-- Defining the conditions
def isParallelogram (a b : ℕ) := a = 18 ∧ b = 12

-- The theorem statement to prove
theorem parallelogram_circumference (a b : ℕ) (h : isParallelogram a b) : 2 * (a + b) = 60 :=
  by
  -- Extract the conditions from hypothesis
  cases h with
  | intro hab' hab'' =>
    sorry

end parallelogram_circumference_l29_29693


namespace smallest_whole_number_l29_29217

theorem smallest_whole_number :
  ∃ a : ℕ, a % 3 = 2 ∧ a % 5 = 3 ∧ a % 7 = 3 ∧ ∀ b : ℕ, (b % 3 = 2 ∧ b % 5 = 3 ∧ b % 7 = 3 → a ≤ b) :=
sorry

end smallest_whole_number_l29_29217


namespace range_of_a_for_local_maximum_l29_29375

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a_for_local_maximum {a : ℝ} (hf_max : ∀ x : ℝ, f' a x = 0 → ∀ y : ℝ, y ≠ x → f' a y ≤ f' a x) :
  -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_local_maximum_l29_29375


namespace double_square_area_l29_29013

theorem double_square_area (a k : ℝ) (h : (k * a) ^ 2 = 2 * a ^ 2) : k = Real.sqrt 2 := 
by 
  -- Our goal is to prove that k = sqrt(2)
  sorry

end double_square_area_l29_29013


namespace annual_return_l29_29010

theorem annual_return (initial_price profit : ℝ) (h₁ : initial_price = 5000) (h₂ : profit = 400) : 
  ((profit / initial_price) * 100 = 8) := by
  -- Lean's substitute for proof
  sorry

end annual_return_l29_29010


namespace div_by_3_l29_29086

theorem div_by_3 (a b : ℤ) : 
  (∃ (k : ℤ), a = 3 * k) ∨ 
  (∃ (k : ℤ), b = 3 * k) ∨ 
  (∃ (k : ℤ), a + b = 3 * k) ∨ 
  (∃ (k : ℤ), a - b = 3 * k) :=
sorry

end div_by_3_l29_29086


namespace compute_expression_l29_29805

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem compute_expression : 
  f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry

end compute_expression_l29_29805


namespace no_infinite_arithmetic_progression_divisible_l29_29309

-- Definitions based on the given condition
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

def product_divisible_by_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
(a n * a (n+1) * a (n+2) * a (n+3) * a (n+4) * a (n+5) * a (n+6) * a (n+7) * a (n+8) * a (n+9)) %
(a n + a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6) + a (n+7) + a (n+8) + a (n+9)) = 0

-- Final statement to be proven
theorem no_infinite_arithmetic_progression_divisible :
  ¬ ∃ (a : ℕ → ℕ), is_arithmetic_progression a ∧ ∀ n : ℕ, product_divisible_by_sum a n := 
sorry

end no_infinite_arithmetic_progression_divisible_l29_29309


namespace avg_speed_is_20_l29_29091

-- Define the total distance and total time
def total_distance : ℕ := 100
def total_time : ℕ := 5

-- Define the average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The theorem to prove the average speed given the distance and time
theorem avg_speed_is_20 : average_speed total_distance total_time = 20 :=
by
  sorry

end avg_speed_is_20_l29_29091


namespace area_ratio_l29_29085

variable (A_shape A_triangle : ℝ)

-- Condition: The area ratio given.
axiom ratio_condition : A_shape / A_triangle = 2

-- Theorem statement
theorem area_ratio (A_shape A_triangle : ℝ) (h : A_shape / A_triangle = 2) : A_shape / A_triangle = 2 :=
by
  exact h

end area_ratio_l29_29085


namespace total_wheels_combined_l29_29059

-- Define the counts of vehicles and wheels per vehicle in each storage area
def bicycles_A : ℕ := 16
def tricycles_A : ℕ := 7
def unicycles_A : ℕ := 10
def four_wheelers_A : ℕ := 5

def bicycles_B : ℕ := 12
def tricycles_B : ℕ := 5
def unicycles_B : ℕ := 8
def four_wheelers_B : ℕ := 3

def wheels_bicycle : ℕ := 2
def wheels_tricycle : ℕ := 3
def wheels_unicycle : ℕ := 1
def wheels_four_wheeler : ℕ := 4

-- Calculate total wheels in Storage Area A
def total_wheels_A : ℕ :=
  bicycles_A * wheels_bicycle + tricycles_A * wheels_tricycle + unicycles_A * wheels_unicycle + four_wheelers_A * wheels_four_wheeler
  
-- Calculate total wheels in Storage Area B
def total_wheels_B : ℕ :=
  bicycles_B * wheels_bicycle + tricycles_B * wheels_tricycle + unicycles_B * wheels_unicycle + four_wheelers_B * wheels_four_wheeler

-- Theorem stating that the combined total number of wheels in both storage areas is 142
theorem total_wheels_combined : total_wheels_A + total_wheels_B = 142 := by
  sorry

end total_wheels_combined_l29_29059


namespace distinct_values_in_expression_rearrangement_l29_29600

theorem distinct_values_in_expression_rearrangement : 
  ∀ (exp : ℕ), exp = 3 → 
  (∃ n : ℕ, n = 3 ∧ 
    let a := exp ^ (exp ^ exp)
    let b := exp ^ ((exp ^ exp) ^ exp)
    let c := ((exp ^ exp) ^ exp) ^ exp
    let d := (exp ^ (exp ^ exp)) ^ exp
    let e := (exp ^ exp) ^ (exp ^ exp)
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :=
by
  sorry

end distinct_values_in_expression_rearrangement_l29_29600


namespace rate_of_interest_l29_29591

theorem rate_of_interest (R : ℝ) (h : 5000 * 2 * R / 100 + 3000 * 4 * R / 100 = 2200) : R = 10 := by
  sorry

end rate_of_interest_l29_29591


namespace intersection_of_A_and_B_l29_29331

theorem intersection_of_A_and_B :
  let A := {0, 1, 2, 3, 4}
  let B := {x | ∃ n ∈ A, x = 2 * n}
  A ∩ B = {0, 2, 4} :=
by
  sorry

end intersection_of_A_and_B_l29_29331


namespace total_sides_of_cookie_cutters_l29_29973

theorem total_sides_of_cookie_cutters :
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  total_sides = 75 :=
by
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  show total_sides = 75
  sorry

end total_sides_of_cookie_cutters_l29_29973


namespace trajectory_of_point_l29_29095

theorem trajectory_of_point (P : ℝ × ℝ) 
  (h1 : dist P (0, 3) = dist P (x1, -3)) :
  ∃ p > 0, (P.fst)^2 = 2 * p * P.snd ∧ p = 6 :=
by {
  sorry
}

end trajectory_of_point_l29_29095


namespace isosceles_triangle_area_l29_29565

noncomputable def area_of_isosceles_triangle (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20) : ℝ :=
  1 / 2 * (2 * b) * 10

theorem isosceles_triangle_area (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20)
  (h3 : 2 * s + 2 * b = 40) : area_of_isosceles_triangle b s h1 h2 = 75 :=
sorry

end isosceles_triangle_area_l29_29565


namespace arithmetic_sequence_sum_l29_29997

/-
In an arithmetic sequence, if the sum of terms \( a_2 + a_3 + a_4 + a_5 + a_6 = 90 \), 
prove that \( a_1 + a_7 = 36 \).
-/

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_sum : a 2 + a 3 + a 4 + a 5 + a 6 = 90) :
  a 1 + a 7 = 36 := by
  sorry

end arithmetic_sequence_sum_l29_29997


namespace students_didnt_like_food_l29_29933

theorem students_didnt_like_food (total_students : ℕ) (liked_food : ℕ) (didnt_like_food : ℕ) 
  (h1 : total_students = 814) (h2 : liked_food = 383) 
  : didnt_like_food = total_students - liked_food := 
by 
  rw [h1, h2]
  sorry

end students_didnt_like_food_l29_29933


namespace pink_cookies_eq_fifty_l29_29435

-- Define the total number of cookies
def total_cookies : ℕ := 86

-- Define the number of red cookies
def red_cookies : ℕ := 36

-- The property we want to prove
theorem pink_cookies_eq_fifty : (total_cookies - red_cookies = 50) :=
by
  sorry

end pink_cookies_eq_fifty_l29_29435


namespace xy_difference_l29_29486

theorem xy_difference (x y : ℚ) (h1 : 3 * x - 4 * y = 17) (h2 : x + 3 * y = 5) : x - y = 73 / 13 :=
by
  sorry

end xy_difference_l29_29486


namespace quadratic_real_root_m_l29_29846

theorem quadratic_real_root_m (m : ℝ) (h : 4 - 4 * m ≥ 0) : m = 0 ∨ m = 2 ∨ m = 4 ∨ m = 6 ↔ m = 0 :=
by
  sorry

end quadratic_real_root_m_l29_29846


namespace compute_pounds_of_cotton_l29_29855

theorem compute_pounds_of_cotton (x : ℝ) :
  (5 * 30 + 10 * x = 640) → (x = 49) := by
  intro h
  sorry

end compute_pounds_of_cotton_l29_29855


namespace multiple_of_six_as_four_cubes_integer_as_five_cubes_l29_29911

-- Part (a)
theorem multiple_of_six_as_four_cubes (n : ℤ) : ∃ a b c d : ℤ, 6 * n = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 :=
by
  sorry

-- Part (b)
theorem integer_as_five_cubes (k : ℤ) : ∃ a b c d e : ℤ, k = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 + e ^ 3 :=
by
  have h := multiple_of_six_as_four_cubes
  sorry

end multiple_of_six_as_four_cubes_integer_as_five_cubes_l29_29911


namespace chameleons_to_blue_l29_29319

-- Define a function that simulates the biting between chameleons and their resulting color changes
def color_transition (color_biter : ℕ) (color_bitten : ℕ) : ℕ :=
  if color_bitten = 1 then color_biter + 1
  else if color_bitten = 2 then color_biter + 2
  else if color_bitten = 3 then color_biter + 3
  else if color_bitten = 4 then color_biter + 4
  else 5  -- Once it reaches color 5 (blue), it remains blue.

-- Define the main theorem statement that given 5 red chameleons, all can be turned to blue.
theorem chameleons_to_blue : ∀ (red_chameleons : ℕ), red_chameleons = 5 → 
  ∃ (sequence_of_bites : ℕ → (ℕ × ℕ)), (∀ (c : ℕ), c < 5 → color_transition c (sequence_of_bites c).fst = 5) :=
by sorry

end chameleons_to_blue_l29_29319


namespace infinitely_many_a_l29_29157

theorem infinitely_many_a (n : ℕ) : ∃ (a : ℕ), ∃ (k : ℕ), ∀ n : ℕ, n^6 + 3 * (3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3) = (n^2 + 3 * k)^3 :=
by
  sorry

end infinitely_many_a_l29_29157


namespace pencils_placed_by_Joan_l29_29829

variable (initial_pencils : ℕ)
variable (total_pencils : ℕ)

theorem pencils_placed_by_Joan 
  (h1 : initial_pencils = 33) 
  (h2 : total_pencils = 60)
  : total_pencils - initial_pencils = 27 := 
by
  sorry

end pencils_placed_by_Joan_l29_29829


namespace red_flowers_needed_l29_29082

-- Define the number of white and red flowers
def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

-- Define the problem statement.
theorem red_flowers_needed : red_flowers + 208 = white_flowers := by
  -- The proof goes here.
  sorry

end red_flowers_needed_l29_29082


namespace arithmetic_series_sum_l29_29857

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 56
  let n := 19
  let pairs_sum := (n-1) / 2 * (-3)
  let single_term := 56
  2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 = 29 :=
by
  sorry

end arithmetic_series_sum_l29_29857


namespace cyclic_determinant_zero_l29_29075

open Matrix

-- Define the roots of the polynomial and the polynomial itself.
variables {α β γ δ : ℂ} -- We assume the roots are complex numbers.
variable (p q r : ℂ) -- Coefficients of the polynomial x^4 + px^2 + qx + r = 0

-- Define the matrix whose determinant we want to compute
def cyclic_matrix (α β γ δ : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![
    ![α, β, γ, δ],
    ![β, γ, δ, α],
    ![γ, δ, α, β],
    ![δ, α, β, γ]
  ]

-- Statement of the theorem
theorem cyclic_determinant_zero :
  ∀ (α β γ δ : ℂ) (p q r : ℂ),
  (∀ x : ℂ, x ^ 4 + p * x ^ 2 + q * x + r = 0 → x = α ∨ x = β ∨ x = γ ∨ x = δ) →
  det (cyclic_matrix α β γ δ) = 0 :=
by
  intros α β γ δ p q r hRoots
  sorry

end cyclic_determinant_zero_l29_29075


namespace julian_comic_book_l29_29002

theorem julian_comic_book : 
  ∀ (total_frames frames_per_page : ℕ),
    total_frames = 143 →
    frames_per_page = 11 →
    total_frames / frames_per_page = 13 ∧ total_frames % frames_per_page = 0 :=
by
  intros total_frames frames_per_page
  intros h_total_frames h_frames_per_page
  sorry

end julian_comic_book_l29_29002


namespace problem_1_problem_2_problem_3_l29_29360

-- First proof statement
theorem problem_1 : 2017^2 - 2016 * 2018 = 1 :=
by
  sorry

-- Definitions for the second problem
variables {a b : ℤ}

-- Second proof statement
theorem problem_2 (h1 : a + b = 7) (h2 : a * b = -1) : (a + b)^2 = 49 :=
by
  sorry

-- Third proof statement (part of the second problem)
theorem problem_3 (h1 : a + b = 7) (h2 : a * b = -1) : a^2 - 3 * a * b + b^2 = 54 :=
by
  sorry

end problem_1_problem_2_problem_3_l29_29360


namespace three_people_on_staircase_l29_29510

theorem three_people_on_staircase (A B C : Type) (steps : Finset ℕ) (h1 : steps.card = 7) 
  (h2 : ∀ step ∈ steps, step ≤ 2) : 
  ∃ (total_ways : ℕ), total_ways = 336 :=
by {
  sorry
}

end three_people_on_staircase_l29_29510


namespace insurance_covers_80_percent_l29_29067

def total_cost : ℝ := 300
def out_of_pocket_cost : ℝ := 60
def insurance_coverage : ℝ := 0.8  -- Representing 80%

theorem insurance_covers_80_percent :
  (total_cost - out_of_pocket_cost) / total_cost = insurance_coverage := by
  sorry

end insurance_covers_80_percent_l29_29067


namespace expected_coincidences_l29_29422

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l29_29422


namespace average_sleep_hours_l29_29069

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l29_29069


namespace lisa_flew_distance_l29_29807

-- Define the given conditions
def speed := 32  -- speed in miles per hour
def time := 8    -- time in hours

-- Define the derived distance
def distance := speed * time  -- using the formula Distance = Speed × Time

-- Prove that the calculated distance is 256 miles
theorem lisa_flew_distance : distance = 256 :=
by
  sorry

end lisa_flew_distance_l29_29807


namespace area_of_similar_rectangle_l29_29116

theorem area_of_similar_rectangle:
  ∀ (R1 : ℝ → ℝ → Prop) (R2 : ℝ → ℝ → Prop),
  (∀ a b, R1 a b → a = 3 ∧ a * b = 18) →
  (∀ a b c d, R1 a b → R2 c d → c / d = a / b) →
  (∀ a b, R2 a b → a^2 + b^2 = 400) →
  ∃ areaR2, (∀ a b, R2 a b → a * b = areaR2) ∧ areaR2 = 160 :=
by
  intros R1 R2 hR1 h_similar h_diagonal
  use 160
  sorry

end area_of_similar_rectangle_l29_29116


namespace students_liking_both_l29_29046

theorem students_liking_both (total_students sports_enthusiasts music_enthusiasts neither : ℕ)
  (h1 : total_students = 55)
  (h2: sports_enthusiasts = 43)
  (h3: music_enthusiasts = 34)
  (h4: neither = 4) : 
  ∃ x, ((sports_enthusiasts - x) + x + (music_enthusiasts - x) = total_students - neither) ∧ (x = 22) :=
by
  sorry -- Proof omitted

end students_liking_both_l29_29046


namespace log_base_change_l29_29797

-- Define the conditions: 8192 = 2 ^ 13 and change of base formula
def x : ℕ := 8192
def a : ℕ := 2
def n : ℕ := 13
def b : ℕ := 5

theorem log_base_change (log : ℕ → ℕ → ℝ) 
  (h1 : x = a ^ n) 
  (h2 : ∀ (x b c: ℕ), c ≠ 1 → log x b = (log x c) / (log b c) ): 
  log x b = 13 / (log 5 2) :=
by
  sorry

end log_base_change_l29_29797


namespace arrangement_non_adjacent_l29_29420

theorem arrangement_non_adjacent :
  let total_arrangements := Nat.factorial 30
  let adjacent_arrangements := 2 * Nat.factorial 29
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements = 28 * Nat.factorial 29 :=
by
  sorry

end arrangement_non_adjacent_l29_29420


namespace percentage_half_day_students_l29_29235

theorem percentage_half_day_students
  (total_students : ℕ)
  (full_day_students : ℕ)
  (h_total : total_students = 80)
  (h_full_day : full_day_students = 60) :
  ((total_students - full_day_students) / total_students : ℚ) * 100 = 25 := 
by
  sorry

end percentage_half_day_students_l29_29235


namespace product_of_fractions_l29_29815

theorem product_of_fractions :
  (1 / 5) * (3 / 7) = 3 / 35 :=
sorry

end product_of_fractions_l29_29815


namespace ZYX_syndrome_diagnosis_l29_29099

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l29_29099


namespace sqrt_of_4_l29_29620

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l29_29620


namespace distance_of_canteen_from_each_camp_l29_29249

noncomputable def distanceFromCanteen (distGtoRoad distBtoG : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (distGtoRoad ^ 2 + distBtoG ^ 2)
  hypotenuse / 2

theorem distance_of_canteen_from_each_camp :
  distanceFromCanteen 360 800 = 438.6 :=
by
  sorry -- The proof is omitted but must show that this statement is valid.

end distance_of_canteen_from_each_camp_l29_29249


namespace smallest_number_of_set_s_l29_29848

theorem smallest_number_of_set_s : 
  ∀ (s : Set ℕ),
    (∃ n : ℕ, s = {k | ∃ m : ℕ, k = 5 * (m+n) ∧ m < 45}) ∧ 
    (275 ∈ s) → 
      (∃ min_elem : ℕ, min_elem ∈ s ∧ min_elem = 55) 
  :=
by
  sorry

end smallest_number_of_set_s_l29_29848


namespace part_I_part_II_l29_29031

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (∃ x : ℝ, (x * (f a (1/x))) = 4 * x - 3 ∧ ∀ y, x = y → (x * (f a (1/x))) = 4 * x - 3) →
  a = 2 :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := 2 / x - x

theorem part_II : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f2 x1 > f2 x2 :=
sorry

end part_I_part_II_l29_29031


namespace mike_planted_50_l29_29436

-- Definitions for conditions
def mike_morning (M : ℕ) := M
def ted_morning (M : ℕ) := 2 * M
def mike_afternoon := 60
def ted_afternoon := 40
def total_planted (M : ℕ) := mike_morning M + ted_morning M + mike_afternoon + ted_afternoon

-- Statement to prove
theorem mike_planted_50 (M : ℕ) (h : total_planted M = 250) : M = 50 :=
by
  sorry

end mike_planted_50_l29_29436


namespace abs_diff_of_two_numbers_l29_29412

theorem abs_diff_of_two_numbers (x y : ℝ) (h_sum : x + y = 42) (h_prod : x * y = 437) : |x - y| = 4 :=
sorry

end abs_diff_of_two_numbers_l29_29412


namespace value_of_x_minus_y_l29_29184

theorem value_of_x_minus_y (x y : ℝ) 
  (h1 : |x| = 2) 
  (h2 : y^2 = 9) 
  (h3 : x + y < 0) : 
  x - y = 1 ∨ x - y = 5 := 
by 
  sorry

end value_of_x_minus_y_l29_29184


namespace Cody_spent_25_tickets_on_beanie_l29_29899

-- Introducing the necessary definitions and assumptions
variable (x : ℕ)

-- Define the conditions translated from the problem statement
def initial_tickets := 49
def tickets_left (x : ℕ) := initial_tickets - x + 6

-- State the main problem as Theorem
theorem Cody_spent_25_tickets_on_beanie (H : tickets_left x = 30) : x = 25 := by
  sorry

end Cody_spent_25_tickets_on_beanie_l29_29899


namespace paul_money_duration_l29_29641

theorem paul_money_duration (earn1 earn2 spend : ℕ) (h1 : earn1 = 3) (h2 : earn2 = 3) (h_spend : spend = 3) : 
  (earn1 + earn2) / spend = 2 :=
by
  sorry

end paul_money_duration_l29_29641


namespace students_number_l29_29224

theorem students_number (C P S : ℕ) : C = 315 ∧ 121 + C = P * S -> S = 4 := by
  sorry

end students_number_l29_29224


namespace Dalton_saved_amount_l29_29241

theorem Dalton_saved_amount (total_cost uncle_contribution additional_needed saved_from_allowance : ℕ) 
  (h_total_cost : total_cost = 7 + 12 + 4)
  (h_uncle_contribution : uncle_contribution = 13)
  (h_additional_needed : additional_needed = 4)
  (h_current_amount : total_cost - additional_needed = 19)
  (h_saved_amount : 19 - uncle_contribution = saved_from_allowance) :
  saved_from_allowance = 6 :=
sorry

end Dalton_saved_amount_l29_29241


namespace auntie_em_parking_l29_29247

theorem auntie_em_parking (total_spaces cars : ℕ) (probability_can_park : ℚ) :
  total_spaces = 20 →
  cars = 15 →
  probability_can_park = 232/323 :=
by
  sorry

end auntie_em_parking_l29_29247


namespace leaves_blew_away_l29_29672

theorem leaves_blew_away (initial_leaves : ℕ) (leaves_left : ℕ) (blew_away : ℕ) 
  (h1 : initial_leaves = 356) (h2 : leaves_left = 112) (h3 : blew_away = initial_leaves - leaves_left) :
  blew_away = 244 :=
by
  sorry

end leaves_blew_away_l29_29672


namespace sum_of_squares_l29_29094

theorem sum_of_squares (x : ℚ) (hx : 7 * x = 15) : 
  (x^2 + (2 * x)^2 + (4 * x)^2 = 4725 / 49) := by
  sorry

end sum_of_squares_l29_29094


namespace milkshake_cost_proof_l29_29595

-- Define the problem
def milkshake_cost (total_money : ℕ) (hamburger_cost : ℕ) (n_hamburgers : ℕ)
                   (n_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_hamburgers_cost := n_hamburgers * hamburger_cost
  let money_after_hamburgers := total_money - total_hamburgers_cost
  let milkshake_cost := (money_after_hamburgers - remaining_money) / n_milkshakes
  milkshake_cost

-- Statement to prove
theorem milkshake_cost_proof : milkshake_cost 120 4 8 6 70 = 3 :=
by
  -- we skip the proof steps as the problem statement does not require it
  sorry

end milkshake_cost_proof_l29_29595


namespace problem_l29_29908

theorem problem (a : ℝ) :
  (∀ x : ℝ, (x > 1 ↔ (x - 1 > 0 ∧ 2 * x - a > 0))) → a ≤ 2 :=
by
  sorry

end problem_l29_29908


namespace part_a_part_b_l29_29885

-- Define the system of equations
def system_of_equations (x y z p : ℝ) :=
  x^2 - 3 * y + p = z ∧ y^2 - 3 * z + p = x ∧ z^2 - 3 * x + p = y

-- Part (a) proof problem statement
theorem part_a (p : ℝ) (hp : p ≥ 4) :
  (p > 4 → ¬ ∃ (x y z : ℝ), system_of_equations x y z p) ∧
  (p = 4 → ∀ (x y z : ℝ), system_of_equations x y z 4 → x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

-- Part (b) proof problem statement
theorem part_b (p : ℝ) (hp : 1 < p ∧ p < 4) :
  ∀ (x y z : ℝ), system_of_equations x y z p → x = y ∧ y = z :=
by sorry

end part_a_part_b_l29_29885


namespace expression_of_f_f_increasing_on_interval_inequality_solution_l29_29444

noncomputable def f (x : ℝ) : ℝ := (x / (1 + x^2))

-- 1. Proving f(x) is the given function
theorem expression_of_f (x : ℝ) (h₁ : f x = (a*x + b) / (1 + x^2)) (h₂ : (∀ x, f (-x) = -f x)) (h₃ : f (1/2) = 2/5) :
  f x = x / (1 + x^2) :=
sorry

-- 2. Prove f(x) is increasing on (-1,1)
theorem f_increasing_on_interval {x₁ x₂ : ℝ} (h₁ : -1 < x₁ ∧ x₁ < 1) (h₂ : -1 < x₂ ∧ x₂ < 1) (h₃ : x₁ < x₂) :
  f x₁ < f x₂ :=
sorry

-- 3. Solve the inequality f(t-1) + f(t) < 0 on (0, 1/2)
theorem inequality_solution (t : ℝ) (h₁ : 0 < t) (h₂ : t < 1/2) :
  f (t - 1) + f t < 0 :=
sorry

end expression_of_f_f_increasing_on_interval_inequality_solution_l29_29444


namespace train_speed_l29_29878

/-- Define the lengths of the train and the bridge and the time taken to cross the bridge. --/
def len_train : ℕ := 360
def len_bridge : ℕ := 240
def time_minutes : ℕ := 4
def time_seconds : ℕ := 240 -- 4 minutes converted to seconds

/-- Define the speed calculation based on the given domain. --/
def total_distance : ℕ := len_train + len_bridge
def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

/-- The statement to prove that the speed of the train is 2.5 m/s. --/
theorem train_speed :
  speed total_distance time_seconds = 2.5 := sorry

end train_speed_l29_29878


namespace line_tangent_to_parabola_j_eq_98_l29_29346

theorem line_tangent_to_parabola_j_eq_98 (j : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 7 * y + j = 0 → x ≠ 0) →
  j = 98 :=
by
  sorry

end line_tangent_to_parabola_j_eq_98_l29_29346


namespace tan_five_pi_over_four_l29_29569

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l29_29569


namespace contrapositive_l29_29300

theorem contrapositive (q p : Prop) (h : q → p) : ¬p → ¬q :=
by
  -- Proof will be filled in later.
  sorry

end contrapositive_l29_29300


namespace not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l29_29764

theorem not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C (A B C : ℝ) (h1 : A = 2 * C) (h2 : B = 2 * C) (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := 
by 
  sorry

end not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l29_29764


namespace sequence_general_formula_l29_29053

theorem sequence_general_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) > a n)
  (h3 : ∀ n : ℕ, n > 0 → (a (n + 1))^2 - 2 * a n * a (n + 1) + (a n)^2 = 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
by 
  sorry

end sequence_general_formula_l29_29053


namespace verify_triangle_inequality_l29_29554

-- Conditions of the problem
variables (L : ℕ → ℕ)
-- The rods lengths are arranged in increasing order
axiom rods_in_order : ∀ i : ℕ, L i ≤ L (i + 1)

-- Define the critical check
def critical_check : Prop :=
  L 98 + L 99 > L 100

-- Prove that verifying the critical_check is sufficient
theorem verify_triangle_inequality (h : critical_check L) :
  ∀ i j k : ℕ, 1 ≤ i → i < j → j < k → k ≤ 100 → L i + L j > L k :=
by
  sorry

end verify_triangle_inequality_l29_29554


namespace usual_time_is_12_l29_29904

variable (S T : ℕ)

theorem usual_time_is_12 (h1: S > 0) (h2: 5 * (T + 3) = 4 * T) : T = 12 := 
by 
  sorry

end usual_time_is_12_l29_29904


namespace cost_of_bananas_l29_29845

theorem cost_of_bananas (A B : ℝ) (n : ℝ) (Tcost: ℝ) (Acost: ℝ): 
  (A * n + B = Tcost) → (A * (1 / 2 * n) + B = Acost) → (Tcost = 7) → (Acost = 5) → B = 3 :=
by
  intros hTony hArnold hTcost hAcost
  sorry

end cost_of_bananas_l29_29845


namespace misha_current_dollars_l29_29497

variable (x : ℕ)

def misha_needs_more : ℕ := 13
def total_amount : ℕ := 47

theorem misha_current_dollars : x = total_amount - misha_needs_more → x = 34 :=
by
  sorry

end misha_current_dollars_l29_29497


namespace paint_cost_per_quart_l29_29568

theorem paint_cost_per_quart
  (total_cost : ℝ)
  (coverage_per_quart : ℝ)
  (side_length : ℝ)
  (cost_per_quart : ℝ) 
  (h1 : total_cost = 192)
  (h2 : coverage_per_quart = 10)
  (h3 : side_length = 10) 
  (h4 : cost_per_quart = total_cost / ((6 * side_length ^ 2) / coverage_per_quart))
  : cost_per_quart = 3.20 := 
by 
  sorry

end paint_cost_per_quart_l29_29568


namespace double_acute_angle_is_less_than_180_degrees_l29_29195

theorem double_acute_angle_is_less_than_180_degrees (alpha : ℝ) (h : 0 < alpha ∧ alpha < 90) : 2 * alpha < 180 :=
sorry

end double_acute_angle_is_less_than_180_degrees_l29_29195


namespace midpoint_coordinates_l29_29816

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end midpoint_coordinates_l29_29816


namespace savings_percentage_l29_29170

variable (I : ℝ) -- First year's income
variable (S : ℝ) -- Amount saved in the first year

-- Conditions
axiom condition1 (h1 : S = 0.05 * I) : Prop
axiom condition2 (h2 : S + 0.05 * I = 2 * S) : Prop
axiom condition3 (h3 : (I - S) + 1.10 * (I - S) = 2 * (I - S)) : Prop

-- Theorem that proves the man saved 5% of his income in the first year
theorem savings_percentage : S = 0.05 * I :=
by
  sorry -- Proof goes here

end savings_percentage_l29_29170


namespace find_a_b_tangent_line_at_zero_l29_29287

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_a_b :
  ∃ a b : ℝ, (a ≠ 0) ∧ (∀ x, f' a b x = 2 * x - 8) := 
sorry

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x + x^2 - 8 * x + 3
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x + 2 * x - 8

theorem tangent_line_at_zero :
  g' 0 = -7 ∧ g 0 = 3 ∧ (∀ y, y = 3 + (-7) * x) := 
sorry

end find_a_b_tangent_line_at_zero_l29_29287


namespace Robert_ate_10_chocolates_l29_29872

def chocolates_eaten_by_Nickel : Nat := 5
def difference_between_Robert_and_Nickel : Nat := 5
def chocolates_eaten_by_Robert := chocolates_eaten_by_Nickel + difference_between_Robert_and_Nickel

theorem Robert_ate_10_chocolates : chocolates_eaten_by_Robert = 10 :=
by
  -- Proof omitted
  sorry

end Robert_ate_10_chocolates_l29_29872


namespace hyperbola_asymptotes_and_parabola_l29_29706

-- Definitions for hyperbola and parabola
noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
noncomputable def focus_of_hyperbola (focus : ℝ × ℝ) : Prop := focus = (5, 0)
noncomputable def asymptote_of_hyperbola (y x : ℝ) : Prop := y = (4 / 3) * x ∨ y = - (4 / 3) * x
noncomputable def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

-- Main statement
theorem hyperbola_asymptotes_and_parabola :
  (∀ x y, hyperbola x y → asymptote_of_hyperbola y x) ∧
  (∀ y x, focus_of_hyperbola (5, 0) → parabola y x 10) :=
by
  -- To be proved
  sorry

end hyperbola_asymptotes_and_parabola_l29_29706


namespace ring_toss_total_earnings_l29_29702

theorem ring_toss_total_earnings :
  let earnings_first_ring_day1 := 761
  let days_first_ring_day1 := 88
  let earnings_first_ring_day2 := 487
  let days_first_ring_day2 := 20
  let earnings_second_ring_day1 := 569
  let days_second_ring_day1 := 66
  let earnings_second_ring_day2 := 932
  let days_second_ring_day2 := 15

  let total_first_ring := (earnings_first_ring_day1 * days_first_ring_day1) + (earnings_first_ring_day2 * days_first_ring_day2)
  let total_second_ring := (earnings_second_ring_day1 * days_second_ring_day1) + (earnings_second_ring_day2 * days_second_ring_day2)
  let total_earnings := total_first_ring + total_second_ring

  total_earnings = 128242 :=
by
  sorry

end ring_toss_total_earnings_l29_29702


namespace maximum_value_of_function_l29_29605

theorem maximum_value_of_function :
  ∀ (x : ℝ), -2 < x ∧ x < 0 → x + 1 / x ≤ -2 :=
by
  sorry

end maximum_value_of_function_l29_29605


namespace bamboo_sections_length_l29_29106

variable {n d : ℕ} (a : ℕ → ℕ)
variable (h_arith : ∀ k, a (k + 1) = a k + d)
variable (h_top : a 1 = 10)
variable (h_sum_last_three : a n + a (n - 1) + a (n - 2) = 114)
variable (h_geom_6 : (a 6) ^ 2 = a 1 * a n)

theorem bamboo_sections_length : n = 16 := 
by 
  sorry

end bamboo_sections_length_l29_29106


namespace final_S_is_correct_l29_29306

/-- Define a function to compute the final value of S --/
def final_value_of_S : ℕ :=
  let S := 0
  let I_values := List.range' 1 27 3 -- generate list [1, 4, 7, ..., 28]
  I_values.foldl (fun S I => S + I) 0  -- compute the sum of the list

/-- Theorem stating the final value of S is 145 --/
theorem final_S_is_correct : final_value_of_S = 145 := by
  sorry

end final_S_is_correct_l29_29306


namespace sum_of_two_numbers_l29_29074

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 :=
by
  sorry

end sum_of_two_numbers_l29_29074


namespace solve_for_k_l29_29825

theorem solve_for_k : {k : ℕ | ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} :=
sorry

end solve_for_k_l29_29825


namespace find_d_l29_29539

theorem find_d (c d : ℝ) (h1 : c / d = 5) (h2 : c = 18 - 7 * d) : d = 3 / 2 := by
  sorry

end find_d_l29_29539


namespace roots_are_distinct_l29_29454

theorem roots_are_distinct (a x1 x2 : ℝ) (h : x1 ≠ x2) :
  (∀ x, x^2 - a*x - 2 = 0 → x = x1 ∨ x = x2) → x1 ≠ x2 := sorry

end roots_are_distinct_l29_29454


namespace total_ways_to_choose_president_and_vice_president_of_same_gender_l29_29535

theorem total_ways_to_choose_president_and_vice_president_of_same_gender :
  let boys := 12
  let girls := 12
  (boys * (boys - 1) + girls * (girls - 1)) = 264 :=
by
  sorry

end total_ways_to_choose_president_and_vice_president_of_same_gender_l29_29535


namespace ellipse_eq_max_area_AEBF_l29_29464

open Real

section ellipse_parabola_problem

variables {a b : ℝ} (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (x y k : ℝ) {M : ℝ × ℝ} {AO BO : ℝ} 
  (b_pos : 0 < b) (a_gt_b : b < a) (MF1_dist : abs (y - 1) = 5 / 3) (M_on_parabola : x^2 = 4 * y)
  (M_on_ellipse : (y / a)^2 + (x / b)^2 = 1) (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ)
  (E F : ℝ × ℝ) (A_on_x : A.1 = b ∧ A.2 = 0) (B_on_y : B.1 = 0 ∧ B.2 = a)
  (D_intersect : D.2 = k * D.1) (E_on_ellipse : (E.2 / a)^2 + (E.1 / b)^2 = 1) 
  (F_on_ellipse : (F.2 / a)^2 + (F.1 / b)^2 = 1)
  (k_pos : 0 < k)

theorem ellipse_eq :
  a = 2 ∧ b = sqrt 3 → (y^2 / (2:ℝ)^2 + x^2 / (sqrt 3:ℝ)^2 = 1) :=
sorry

theorem max_area_AEBF :
  (a = 2 ∧ b = sqrt 3) →
  ∃ max_area : ℝ, max_area = 2 * sqrt 6 :=
sorry

end ellipse_parabola_problem

end ellipse_eq_max_area_AEBF_l29_29464


namespace inequality_solution_set_l29_29240

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : (x - 1) / x > 1 ↔ x < 0 :=
by
  sorry

end inequality_solution_set_l29_29240


namespace solution_set_of_inequality_l29_29133

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - x + 2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_of_inequality_l29_29133


namespace common_points_l29_29705

variable {R : Type*} [LinearOrderedField R]

def eq1 (x y : R) : Prop := x - y + 2 = 0
def eq2 (x y : R) : Prop := 3 * x + y - 4 = 0
def eq3 (x y : R) : Prop := x + y - 2 = 0
def eq4 (x y : R) : Prop := 2 * x - 5 * y + 7 = 0

theorem common_points : ∃ s : Finset (R × R), 
  (∀ p ∈ s, eq1 p.1 p.2 ∨ eq2 p.1 p.2) ∧ (∀ p ∈ s, eq3 p.1 p.2 ∨ eq4 p.1 p.2) ∧ s.card = 6 :=
by
  sorry

end common_points_l29_29705


namespace sequence_properties_l29_29490

-- Definitions from conditions
def S (n : ℕ) := n^2 - n
def a (n : ℕ) := if n = 1 then 0 else 2 * (n - 1)
def b (n : ℕ) := 2^(n - 1)
def c (n : ℕ) := a n * b n
def T (n : ℕ) := (n - 2) * 2^(n + 1) + 4

-- Theorem statement proving the required identities
theorem sequence_properties {n : ℕ} (hn : n ≠ 0) :
  (a n = (if n = 1 then 0 else 2 * (n - 1))) ∧ 
  (b 2 = a 2) ∧ 
  (b 4 = a 5) ∧ 
  (T n = (n - 2) * 2^(n + 1) + 4) := by
  sorry

end sequence_properties_l29_29490


namespace n_four_minus_n_squared_l29_29178

theorem n_four_minus_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
by 
  sorry

end n_four_minus_n_squared_l29_29178


namespace slices_left_for_phill_correct_l29_29886

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l29_29886


namespace projectiles_meet_in_90_minutes_l29_29837

theorem projectiles_meet_in_90_minutes
  (d : ℝ) (v1 : ℝ) (v2 : ℝ) (time_in_minutes : ℝ)
  (h_d : d = 1455)
  (h_v1 : v1 = 470)
  (h_v2 : v2 = 500)
  (h_time : time_in_minutes = 90) :
  d / (v1 + v2) * 60 = time_in_minutes :=
by
  sorry

end projectiles_meet_in_90_minutes_l29_29837


namespace identity_proof_l29_29488

theorem identity_proof (A B C A1 B1 C1 : ℝ) :
  (A^2 + B^2 + C^2) * (A1^2 + B1^2 + C1^2) - (A * A1 + B * B1 + C * C1)^2 =
    (A * B1 + A1 * B)^2 + (A * C1 + A1 * C)^2 + (B * C1 + B1 * C)^2 :=
by
  sorry

end identity_proof_l29_29488


namespace arithmetic_mean_is_b_l29_29938

variable (x a b : ℝ)
variable (hx : x ≠ 0)
variable (hb : b ≠ 0)

theorem arithmetic_mean_is_b : (1 / 2 : ℝ) * ((x * b + a) / x + (x * b - a) / x) = b :=
by
  sorry

end arithmetic_mean_is_b_l29_29938


namespace probability_of_three_draws_l29_29708

noncomputable def box_chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_first_two_draws (a b : ℕ) : Prop :=
  a + b <= 7

def prob_three_draws_to_exceed_seven : ℚ :=
  1 / 6

theorem probability_of_three_draws :
  (∃ (draws : List ℕ), (draws.length = 3) ∧ (draws.sum > 7)
    ∧ (∀ x ∈ draws, x ∈ box_chips)
    ∧ (∀ (a b : ℕ), (a ∈ box_chips ∧ b ∈ box_chips) → valid_first_two_draws a b))
  → prob_three_draws_to_exceed_seven = 1 / 6 :=
sorry

end probability_of_three_draws_l29_29708


namespace frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l29_29487

-- Part (a): Prove the number of ways to reach vertex C from A in n jumps when n is even
theorem frog_reaches_C_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = (4^n/2 - 1) / 3 := by sorry

-- Part (b): Prove the number of ways to reach vertex C from A in n jumps without jumping to D when n is even
theorem frog_reaches_C_no_D_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = 3^(n/2 - 1) := by sorry

-- Part (c): Prove the probability the frog is alive after n jumps with a mine at D
theorem frog_alive_probability (n : ℕ) (k : ℕ) (h_n : n = 2*k - 1 ∨ n = 2*k) : 
    ∃ p : ℝ, p = (3/4)^(k-1) := by sorry

-- Part (d): Prove the average lifespan of the frog in the presence of a mine at D
theorem frog_average_lifespan : 
    ∃ t : ℝ, t = 9 := by sorry

end frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l29_29487


namespace domain_of_function_l29_29707

noncomputable def function_domain := {x : ℝ | x * (3 - x) ≥ 0 ∧ x - 1 ≥ 0 }

theorem domain_of_function: function_domain = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end domain_of_function_l29_29707


namespace compute_modulo_l29_29111

theorem compute_modulo :
    (2015 % 7) = 3 ∧ (2016 % 7) = 4 ∧ (2017 % 7) = 5 ∧ (2018 % 7) = 6 →
    (2015 * 2016 * 2017 * 2018) % 7 = 3 :=
by
  intros h
  have h1 := h.left
  have h2 := h.right.left
  have h3 := h.right.right.left
  have h4 := h.right.right.right
  sorry

end compute_modulo_l29_29111


namespace person_dining_minutes_l29_29782

theorem person_dining_minutes
  (initial_angle : ℕ)
  (final_angle : ℕ)
  (time_spent : ℕ)
  (minute_angle_per_minute : ℕ)
  (hour_angle_per_minute : ℕ)
  (h1 : initial_angle = 110)
  (h2 : final_angle = 110)
  (h3 : minute_angle_per_minute = 6)
  (h4 : hour_angle_per_minute = minute_angle_per_minute / 12)
  (h5 : time_spent = (final_angle - initial_angle) / (minute_angle_per_minute / (minute_angle_per_minute / 12) - hour_angle_per_minute)) :
  time_spent = 40 := sorry

end person_dining_minutes_l29_29782


namespace f_diff_ineq_l29_29079

variable {f : ℝ → ℝ}
variable (deriv_f : ∀ x > 0, x * (deriv f x) > 1)

theorem f_diff_ineq (h : ∀ x > 0, x * (deriv f x) > 1) : f 2 - f 1 > Real.log 2 := by 
  sorry

end f_diff_ineq_l29_29079


namespace simplify_fraction_l29_29522

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end simplify_fraction_l29_29522


namespace solve_equation_l29_29656

theorem solve_equation (x y : ℝ) (k : ℤ) :
  x^2 - 2 * x * Real.sin (x * y) + 1 = 0 ↔ (x = 1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) ∨ (x = -1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) :=
by
  -- Logical content will be filled here, sorry is used because proof steps are not required.
  sorry

end solve_equation_l29_29656


namespace units_digit_uniform_l29_29118

-- Definitions
def domain : Finset ℕ := Finset.range 15

def pick : Type := { n // n ∈ domain }

def uniform_pick : pick := sorry

-- Statement of the theorem
theorem units_digit_uniform :
  ∀ (J1 J2 K : pick), 
  ∃ d : ℕ, d < 10 ∧ (J1.val + J2.val + K.val) % 10 = d
:= sorry

end units_digit_uniform_l29_29118


namespace find_a_and_b_l29_29265

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l29_29265


namespace algebra_expression_value_l29_29594

theorem algebra_expression_value (m : ℝ) (h : m^2 - 3 * m - 1 = 0) : 2 * m^2 - 6 * m + 5 = 7 := by
  sorry

end algebra_expression_value_l29_29594


namespace parallel_and_through_point_l29_29926

-- Defining the given line
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Defining the target line passing through the point (0, 4)
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the point (0, 4)
def point : ℝ × ℝ := (0, 4)

-- Prove that line2 passes through the point (0, 4) and is parallel to line1
theorem parallel_and_through_point (x y : ℝ) 
  (h1 : line1 x y) 
  : line2 (point.fst) (point.snd) := by
  sorry

end parallel_and_through_point_l29_29926


namespace milk_replacement_problem_l29_29508

theorem milk_replacement_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 90)
  (h3 : (90 - x) - ((90 - x) * x / 90) = 72.9) : x = 9 :=
sorry

end milk_replacement_problem_l29_29508


namespace least_N_bench_sections_l29_29008

-- First, define the problem conditions
def bench_capacity_adult (N : ℕ) : ℕ := 7 * N
def bench_capacity_child (N : ℕ) : ℕ := 11 * N

-- Define the problem statement to be proven
theorem least_N_bench_sections :
  ∃ N : ℕ, (N > 0) ∧ (bench_capacity_adult N = bench_capacity_child N → N = 77) :=
sorry

end least_N_bench_sections_l29_29008


namespace find_a_l29_29781

-- Define given parameters and conditions
def parabola_eq (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

def shifted_parabola_eq (a : ℝ) (x : ℝ) : ℝ := parabola_eq a x - 3 * |a|

-- Define axis of symmetry function
def axis_of_symmetry (a : ℝ) : ℝ := 1

-- Conditions: a ≠ 0
variable (a : ℝ)
variable (h : a ≠ 0)

-- Define value for discriminant check
def discriminant (a : ℝ) (c : ℝ) : ℝ := (-2 * a)^2 - 4 * a * c

-- Problem statement
theorem find_a (ha : a ≠ 0) : 
  (axis_of_symmetry a = 1) ∧ (discriminant a (3 - 3 * |a|) = 0 → (a = 3 / 4 ∨ a = -3 / 2)) := 
by
  sorry -- proof to be filled in

end find_a_l29_29781


namespace paint_leftover_l29_29421

theorem paint_leftover (containers total_walls tiles_wall paint_ceiling : ℕ) 
  (h_containers : containers = 16) 
  (h_total_walls : total_walls = 4) 
  (h_tiles_wall : tiles_wall = 1) 
  (h_paint_ceiling : paint_ceiling = 1) : 
  containers - ((total_walls - tiles_wall) * (containers / total_walls)) - paint_ceiling = 3 :=
by 
  sorry

end paint_leftover_l29_29421


namespace inv_eq_self_l29_29114

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem inv_eq_self (m : ℝ) :
  (∀ x : ℝ, g m x = g m (g m x)) ↔ m ∈ Set.Iic (-9 / 4) ∪ Set.Ici (-9 / 4) :=
by
  sorry

end inv_eq_self_l29_29114


namespace range_of_d_largest_S_n_l29_29159

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d a_1 : ℝ)

-- Conditions
axiom a_3_eq_12 : a_n 3 = 12
axiom S_12_pos : S_n 12 > 0
axiom S_13_neg : S_n 13 < 0
axiom arithmetic_sequence : ∀ n, a_n n = a_1 + (n - 1) * d
axiom sum_of_terms : ∀ n, S_n n = n * a_1 + (n * (n - 1)) / 2 * d

-- Problems
theorem range_of_d : -24/7 < d ∧ d < -3 := sorry

theorem largest_S_n : (∀ m, m > 0 ∧ m < 13 → (S_n 6 >= S_n m)) := sorry

end range_of_d_largest_S_n_l29_29159


namespace range_of_a_l29_29828

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 4^x - (a + 3) * 2^x + 1 = 0) → a ≥ -1 := sorry

end range_of_a_l29_29828


namespace solve_inequality_l29_29876

noncomputable def inequality_solution : Set ℝ :=
  { x | x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4 }

theorem solve_inequality :
  inequality_solution = { x | -2 < x ∧ x < 2 } ∪ { x | 3 ≤ x } :=
by
  sorry

end solve_inequality_l29_29876


namespace slower_train_speed_l29_29761

theorem slower_train_speed (v : ℝ) (faster_train_speed : ℝ) (time_pass : ℝ) (train_length : ℝ) :
  (faster_train_speed = 46) →
  (time_pass = 36) →
  (train_length = 50) →
  (v = 36) :=
by
  intro h1 h2 h3
  -- Formal proof goes here
  sorry

end slower_train_speed_l29_29761


namespace inequality_solution_sets_l29_29952

theorem inequality_solution_sets (a b m : ℝ) (h_sol_set : ∀ x, x^2 - a * x - 2 > 0 ↔ x < -1 ∨ x > b) (hb : b > -1) (hm : m > -1 / 2) :
  a = 1 ∧ b = 2 ∧ 
  (if m > 0 then ∀ x, (x < -1/m ∨ x > 2) ↔ (mx + 1) * (x - 2) > 0 
   else if m = 0 then ∀ x, x > 2 ↔ (mx + 1) * (x - 2) > 0 
   else ∀ x, (2 < x ∧ x < -1/m) ↔ (mx + 1) * (x - 2) > 0) :=
by
  sorry

end inequality_solution_sets_l29_29952


namespace mod_product_example_l29_29272

theorem mod_product_example :
  ∃ m : ℤ, 256 * 738 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 ∧ m = 53 :=
by
  use 53
  sorry

end mod_product_example_l29_29272


namespace telepathic_connection_correct_l29_29348

def telepathic_connection_probability : ℚ := sorry

theorem telepathic_connection_correct :
  telepathic_connection_probability = 7 / 25 := sorry

end telepathic_connection_correct_l29_29348


namespace negation_of_existential_l29_29516

theorem negation_of_existential :
  ¬ (∃ x : ℝ, x^2 - 2 * x - 3 < 0) ↔ ∀ x : ℝ, x^2 - 2 * x - 3 ≥ 0 :=
by sorry

end negation_of_existential_l29_29516


namespace central_angle_of_sector_l29_29851

theorem central_angle_of_sector (R θ l : ℝ) (h1 : 2 * R + l = π * R) : θ = π - 2 := 
by
  sorry

end central_angle_of_sector_l29_29851


namespace indigo_restaurant_total_reviews_l29_29737

-- Define the number of 5-star reviews
def five_star_reviews : Nat := 6

-- Define the number of 4-star reviews
def four_star_reviews : Nat := 7

-- Define the number of 3-star reviews
def three_star_reviews : Nat := 4

-- Define the number of 2-star reviews
def two_star_reviews : Nat := 1

-- Define the total number of reviews
def total_reviews : Nat := five_star_reviews + four_star_reviews + three_star_reviews + two_star_reviews

-- Proof that the total number of customer reviews is 18
theorem indigo_restaurant_total_reviews : total_reviews = 18 :=
by
  -- Direct calculation
  sorry

end indigo_restaurant_total_reviews_l29_29737


namespace factorize_expression_l29_29499

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l29_29499


namespace kylie_total_apples_l29_29293

-- Define the conditions as given in the problem.
def first_hour_apples : ℕ := 66
def second_hour_apples : ℕ := 2 * first_hour_apples
def third_hour_apples : ℕ := first_hour_apples / 3

-- Define the mathematical proof problem.
theorem kylie_total_apples : 
  first_hour_apples + second_hour_apples + third_hour_apples = 220 :=
by
  -- Proof goes here
  sorry

end kylie_total_apples_l29_29293


namespace man_was_absent_for_days_l29_29902

theorem man_was_absent_for_days
  (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 10 * x - 2 * y = 216) :
  y = 7 :=
by
  sorry

end man_was_absent_for_days_l29_29902


namespace equal_tuesdays_thursdays_l29_29802

theorem equal_tuesdays_thursdays (days_in_month : ℕ) (tuesdays : ℕ) (thursdays : ℕ) : (days_in_month = 30) → (tuesdays = thursdays) → (∃ (start_days : Finset ℕ), start_days.card = 2) :=
by
  sorry

end equal_tuesdays_thursdays_l29_29802


namespace geometric_sequence_value_l29_29281

theorem geometric_sequence_value (a : ℝ) (h_pos : 0 < a) 
    (h_geom1 : ∃ r, 25 * r = a)
    (h_geom2 : ∃ r, a * r = 7 / 9) : 
    a = 5 * Real.sqrt 7 / 3 :=
by
  sorry

end geometric_sequence_value_l29_29281


namespace min_sum_product_l29_29166

theorem min_sum_product (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 9/n = 1) :
  m * n = 48 :=
sorry

end min_sum_product_l29_29166


namespace inequality_addition_l29_29972

-- Definitions and Conditions
variables (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c > d)

-- Theorem statement: Prove that a + c > b + d
theorem inequality_addition (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := 
sorry

end inequality_addition_l29_29972


namespace top_cell_pos_cases_l29_29561

-- Define the rule for the cell sign propagation
def cell_sign (a b : ℤ) : ℤ := 
  if a = b then 1 else -1

-- The pyramid height
def pyramid_height : ℕ := 5

-- Define the final condition for the top cell in the pyramid to be "+"
def top_cell_sign (a b c d e : ℤ) : ℤ :=
  a * b * c * d * e

-- Define the proof statement
theorem top_cell_pos_cases :
  (∃ a b c d e : ℤ,
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    top_cell_sign a b c d e = 1) ∧
  (∃ n, n = 11) :=
by
  sorry

end top_cell_pos_cases_l29_29561


namespace largest_x_value_l29_29688

theorem largest_x_value (x : ℝ) :
  (x ≠ 9) ∧ (x ≠ -4) ∧ ((x ^ 2 - x - 72) / (x - 9) = 5 / (x + 4)) → x = -3 :=
sorry

end largest_x_value_l29_29688


namespace inequality_holds_for_all_x_l29_29994

theorem inequality_holds_for_all_x (m : ℝ) (h : ∀ x : ℝ, |x + 5| ≥ m + 2) : m ≤ -2 :=
sorry

end inequality_holds_for_all_x_l29_29994


namespace product_of_midpoint_l29_29942

-- Define the coordinates of the endpoints
def x1 := 5
def y1 := -4
def x2 := 1
def y2 := 14

-- Define the formulas for the midpoint coordinates
def xm := (x1 + x2) / 2
def ym := (y1 + y2) / 2

-- Define the product of the midpoint coordinates
def product := xm * ym

-- Now state the theorem
theorem product_of_midpoint :
  product = 15 := 
by
  -- Optional: detailed steps can go here if necessary
  sorry

end product_of_midpoint_l29_29942


namespace grapes_total_sum_l29_29376

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l29_29376


namespace MN_eq_l29_29423

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}
def operation (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

theorem MN_eq : operation M N = {1, 4} :=
sorry

end MN_eq_l29_29423


namespace find_range_m_l29_29027

-- Definitions of the conditions
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x + y - m = 0) ∧ ((x - 1)^2 + y^2 = 1)
def q (m : ℝ) : Prop := ∃ x : ℝ, (x^2 - x + m - 4 = 0) ∧ x ≠ 0 ∧ ∀ y : ℝ, (y^2 - y + m - 4 = 0) → x * y < 0

theorem find_range_m (m : ℝ) : (p m ∨ q m) ∧ ¬p m → (m ≤ 1 - Real.sqrt 2 ∨ 1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by
  sorry

end find_range_m_l29_29027


namespace geometric_sequence_product_l29_29427

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_product (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
  (h : a 3 = -1) : a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by
  sorry

end geometric_sequence_product_l29_29427


namespace option_C_holds_l29_29056

theorem option_C_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a - b / a > b - a / b := 
  sorry

end option_C_holds_l29_29056


namespace frequency_of_middle_group_l29_29687

theorem frequency_of_middle_group
    (num_rectangles : ℕ)
    (middle_area : ℝ)
    (other_areas_sum : ℝ)
    (sample_size : ℕ)
    (total_area_norm : ℝ)
    (h1 : num_rectangles = 11)
    (h2 : middle_area = other_areas_sum)
    (h3 : sample_size = 160)
    (h4 : middle_area + other_areas_sum = total_area_norm)
    (h5 : total_area_norm = 1):
    160 * (middle_area / total_area_norm) = 80 :=
by
  sorry

end frequency_of_middle_group_l29_29687


namespace symmetric_points_y_axis_l29_29152

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : ∃ M N : ℝ × ℝ, M = (a, 3) ∧ N = (4, b) ∧ M.1 = -N.1 ∧ M.2 = N.2) :
  (a + b) ^ 2012 = 1 :=
by 
  sorry

end symmetric_points_y_axis_l29_29152


namespace thread_length_l29_29442

def side_length : ℕ := 13

def perimeter (s : ℕ) : ℕ := 4 * s

theorem thread_length : perimeter side_length = 52 := by
  sorry

end thread_length_l29_29442


namespace minimum_degree_g_l29_29571

open Polynomial

theorem minimum_degree_g (f g h : Polynomial ℝ) 
  (h_eq : 5 • f + 2 • g = h)
  (deg_f : f.degree = 11)
  (deg_h : h.degree = 12) : 
  ∃ d : ℕ, g.degree = d ∧ d >= 12 := 
sorry

end minimum_degree_g_l29_29571


namespace middle_number_l29_29949

theorem middle_number {a b c : ℕ} (h1 : a + b = 12) (h2 : a + c = 17) (h3 : b + c = 19) (h4 : a < b) (h5 : b < c) : b = 7 :=
sorry

end middle_number_l29_29949


namespace solve_for_x_l29_29416

theorem solve_for_x (x y : ℝ) (h : (x + 1) / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 5)) : 
  x = (y^2 + 3 * y - 1) / 7 := 
by 
  sorry

end solve_for_x_l29_29416


namespace ineq_triples_distinct_integers_l29_29028

theorem ineq_triples_distinct_integers 
  (x y z : ℤ) (h₁ : x ≠ y) (h₂ : y ≠ z) (h₃ : z ≠ x) : 
  ( ( (x - y)^7 + (y - z)^7 + (z - x)^7 - (x - y) * (y - z) * (z - x) * ((x - y)^4 + (y - z)^4 + (z - x)^4) )
  / ( (x - y)^5 + (y - z)^5 + (z - x)^5 ) ) ≥ 3 :=
sorry

end ineq_triples_distinct_integers_l29_29028


namespace find_original_number_l29_29397

theorem find_original_number (x : ℕ) 
    (h1 : (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183) : x = 32 := 
by
  sorry

end find_original_number_l29_29397


namespace arithmetic_progression_l29_29939

-- Define the general formula for the nth term of an arithmetic progression
def nth_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the conditions given in the problem
def condition1 (a1 d : ℤ) : Prop := nth_term a1 d 13 = 3 * nth_term a1 d 3
def condition2 (a1 d : ℤ) : Prop := nth_term a1 d 18 = 2 * nth_term a1 d 7 + 8

-- The main proof problem statement
theorem arithmetic_progression (a1 d : ℤ) (h1 : condition1 a1 d) (h2 : condition2 a1 d) : a1 = 12 ∧ d = 4 :=
by
  sorry

end arithmetic_progression_l29_29939


namespace customers_left_l29_29004

theorem customers_left (original_customers remaining_tables people_per_table customers_left : ℕ)
  (h1 : original_customers = 44)
  (h2 : remaining_tables = 4)
  (h3 : people_per_table = 8)
  (h4 : original_customers - remaining_tables * people_per_table = customers_left) :
  customers_left = 12 :=
by
  sorry

end customers_left_l29_29004


namespace find_n_l29_29060

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l29_29060


namespace ages_sum_13_and_product_72_l29_29950

theorem ages_sum_13_and_product_72 (g b s : ℕ) (h1 : b < g) (h2 : g < s) (h3 : b * g * s = 72) : b + g + s = 13 :=
sorry

end ages_sum_13_and_product_72_l29_29950


namespace students_just_passed_l29_29100

theorem students_just_passed (total_students first_div_percent second_div_percent : ℝ)
  (h_total_students: total_students = 300)
  (h_first_div_percent: first_div_percent = 0.29)
  (h_second_div_percent: second_div_percent = 0.54)
  (h_no_failures : total_students = 300) :
  ∃ passed_students, passed_students = total_students - (first_div_percent * total_students + second_div_percent * total_students) ∧ passed_students = 51 :=
by
  sorry

end students_just_passed_l29_29100


namespace tan_cos_solution_count_l29_29353

theorem tan_cos_solution_count : 
  ∃ (n : ℕ), n = 5 ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (2 * x) = Real.cos (x / 2) → x ∈ Set.Icc 0 (2 * Real.pi) :=
sorry

end tan_cos_solution_count_l29_29353


namespace solve_for_x_l29_29349

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 14.7 -> x = 105 := by
  sorry

end solve_for_x_l29_29349


namespace compute_c_minus_d_squared_eq_0_l29_29203

-- Defining conditions
def multiples_of_n_under_m (n m : ℕ) : ℕ :=
  (m - 1) / n

-- Defining the specific values
def c : ℕ := multiples_of_n_under_m 9 60
def d : ℕ := multiples_of_n_under_m 9 60  -- Since every multiple of 9 is a multiple of 3

theorem compute_c_minus_d_squared_eq_0 : (c - d) ^ 2 = 0 := by
  sorry

end compute_c_minus_d_squared_eq_0_l29_29203


namespace sum_of_roots_equals_18_l29_29022

-- Define the conditions
variable (f : ℝ → ℝ)
variable (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x))
variable (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0))

-- The theorem statement
theorem sum_of_roots_equals_18 (f : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, f (3 + x) = f (3 - x)) 
  (h_distinct_roots : ∃ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0)) :
  ∀ xs : Finset ℝ, xs.card = 6 ∧ (∀ x ∈ xs, f x = 0) → xs.sum id = 18 :=
by
  sorry

end sum_of_roots_equals_18_l29_29022


namespace intersection_eq_l29_29113

def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x ≥ 1}
def CU_N : Set ℝ := {x : ℝ | x < 1}

theorem intersection_eq : M ∩ CU_N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l29_29113


namespace right_triangle_area_l29_29751

theorem right_triangle_area (a b c r : ℝ) (h1 : a = 15) (h2 : r = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_right : a ^ 2 + b ^ 2 = c ^ 2) (h_incircle : r = (a + b - c) / 2) : 
  1 / 2 * a * b = 60 :=
by
  sorry

end right_triangle_area_l29_29751


namespace one_thirds_of_nine_halfs_l29_29223

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l29_29223


namespace basketball_game_half_points_l29_29871

noncomputable def eagles_geometric_sequence (a r : ℕ) (n : ℕ) : ℕ :=
  a * r ^ n

noncomputable def lions_arithmetic_sequence (b d : ℕ) (n : ℕ) : ℕ :=
  b + n * d

noncomputable def total_first_half_points (a r b d : ℕ) : ℕ :=
  eagles_geometric_sequence a r 0 + eagles_geometric_sequence a r 1 +
  lions_arithmetic_sequence b d 0 + lions_arithmetic_sequence b d 1

theorem basketball_game_half_points (a r b d : ℕ) (h1 : a + a * r = b + (b + d)) (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2*d) + (b + 3*d)) :
  total_first_half_points a r b d = 8 :=
by sorry

end basketball_game_half_points_l29_29871


namespace geometric_seq_value_l29_29588

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_seq_value_l29_29588


namespace exists_sequence_satisfying_conditions_l29_29810

def F : ℕ → ℕ := sorry

theorem exists_sequence_satisfying_conditions :
  (∀ n, ∃ k, F k = n) ∧ 
  (∀ n, ∃ m > n, F m = n) ∧ 
  (∀ n ≥ 2, F (F (n ^ 163)) = F (F n) + F (F 361)) :=
sorry

end exists_sequence_satisfying_conditions_l29_29810


namespace smallest_coprime_gt_one_l29_29783

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l29_29783


namespace ted_age_solution_l29_29329

theorem ted_age_solution (t s : ℝ) (h1 : t = 3 * s - 10) (h2 : t + s = 60) : t = 42.5 :=
by {
  sorry
}

end ted_age_solution_l29_29329


namespace area_of_triangle_is_sqrt3_l29_29808

theorem area_of_triangle_is_sqrt3
  (a b c : ℝ)
  (B : ℝ)
  (h_geom_prog : b^2 = a * c)
  (h_b : b = 2)
  (h_B : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 := 
by
  sorry

end area_of_triangle_is_sqrt3_l29_29808


namespace geom_progression_n_eq_6_l29_29398

theorem geom_progression_n_eq_6
  (a r : ℝ)
  (h_r : r = 6)
  (h_ratio : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217) :
  n = 6 :=
by
  sorry

end geom_progression_n_eq_6_l29_29398


namespace sin_30_eq_half_l29_29841

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l29_29841


namespace quadrilateral_smallest_angle_l29_29677

theorem quadrilateral_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + 2 * d) = 160)
  (h2 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) :
  a = 60 :=
by
  sorry

end quadrilateral_smallest_angle_l29_29677


namespace unique_number_encoding_l29_29345

-- Defining participants' score ranges 
def score_range := {x : ℕ // x ≤ 5}

-- Defining total score
def total_score (s1 s2 s3 s4 s5 s6 : score_range) : ℕ := 
  s1.val + s2.val + s3.val + s4.val + s5.val + s6.val

-- Main statement to encode participant's scores into a unique number
theorem unique_number_encoding (s1 s2 s3 s4 s5 s6 : score_range) :
  ∃ n : ℕ, ∃ s : ℕ, 
    s = total_score s1 s2 s3 s4 s5 s6 ∧ 
    n = s * 10^6 + s1.val * 10^5 + s2.val * 10^4 + s3.val * 10^3 + s4.val * 10^2 + s5.val * 10 + s6.val := 
sorry

end unique_number_encoding_l29_29345


namespace B_investment_amount_l29_29626

-- Definitions based on given conditions
variable (A_investment : ℕ := 300) -- A's investment in dollars
variable (B_investment : ℕ)        -- B's investment in dollars
variable (A_time : ℕ := 12)        -- Time A's investment was in the business in months
variable (B_time : ℕ := 6)         -- Time B's investment was in the business in months
variable (profit : ℕ := 100)       -- Total profit in dollars
variable (A_share : ℕ := 75)       -- A's share of the profit in dollars

-- The mathematically equivalent proof problem to prove that B invested $200
theorem B_investment_amount (h : A_share * (A_investment * A_time + B_investment * B_time) / profit = A_investment * A_time) : 
  B_investment = 200 := by
  sorry

end B_investment_amount_l29_29626


namespace seahawks_final_score_l29_29470

def num_touchdowns : ℕ := 4
def num_field_goals : ℕ := 3
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3

theorem seahawks_final_score : (num_touchdowns * points_per_touchdown) + (num_field_goals * points_per_fieldgoal) = 37 := by
  sorry

end seahawks_final_score_l29_29470


namespace find_clubs_l29_29197

theorem find_clubs (S D H C : ℕ) (h1 : S + D + H + C = 13)
  (h2 : S + C = 7) 
  (h3 : D + H = 6) 
  (h4 : D = 2 * S) 
  (h5 : H = 2 * D) 
  : C = 6 :=
by
  sorry

end find_clubs_l29_29197


namespace largest_circle_area_rounded_to_nearest_int_l29_29244

theorem largest_circle_area_rounded_to_nearest_int
  (x : Real)
  (hx : 3 * x^2 = 180) :
  let r := (16 * Real.sqrt 15) / (2 * Real.pi)
  let area_of_circle := Real.pi * r^2
  round (area_of_circle) = 306 :=
by
  sorry

end largest_circle_area_rounded_to_nearest_int_l29_29244


namespace boat_distance_against_stream_in_one_hour_l29_29651

-- Define the conditions
def speed_in_still_water : ℝ := 4 -- speed of the boat in still water (km/hr)
def downstream_distance_in_one_hour : ℝ := 6 -- distance traveled along the stream in one hour (km)

-- Define the function to compute the speed of the stream
def speed_of_stream (downstream_distance : ℝ) (boat_speed_still_water : ℝ) : ℝ :=
  downstream_distance - boat_speed_still_water

-- Define the effective speed against the stream
def effective_speed_against_stream (boat_speed_still_water : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water - stream_speed

-- Prove that the boat travels 2 km against the stream in one hour given the conditions
theorem boat_distance_against_stream_in_one_hour :
  effective_speed_against_stream speed_in_still_water (speed_of_stream downstream_distance_in_one_hour speed_in_still_water) * 1 = 2 := 
by
  sorry

end boat_distance_against_stream_in_one_hour_l29_29651


namespace sum_of_first_n_terms_l29_29577

variable (a_n : ℕ → ℝ) -- Sequence term
variable (S_n : ℕ → ℝ) -- Sum of first n terms

-- Conditions given in the problem
axiom sum_first_term : a_n 1 = 2
axiom sum_first_two_terms : a_n 1 + a_n 2 = 7
axiom sum_first_three_terms : a_n 1 + a_n 2 + a_n 3 = 18

-- Expected result to prove
theorem sum_of_first_n_terms 
  (h1 : S_n 1 = 2)
  (h2 : S_n 2 = 7)
  (h3 : S_n 3 = 18) :
  S_n n = (3/2) * ((n * (n + 1) * (2 * n + 1) / 6) - (n * (n + 1) / 2) + 2 * n) :=
sorry

end sum_of_first_n_terms_l29_29577


namespace range_of_m_minimum_value_l29_29009

theorem range_of_m (m n : ℝ) (h : 2 * m - n = 3) (ineq : |m| + |n + 3| ≥ 9) : 
  m ≤ -3 ∨ m ≥ 3 := 
sorry

theorem minimum_value (m n : ℝ) (h : 2 * m - n = 3) : 
  ∃ c, c = 3 ∧ c = |(5 / 3) * m - (1 / 3) * n| + |(1 / 3) * m - (2 / 3) * n| := 
sorry

end range_of_m_minimum_value_l29_29009


namespace kitchen_upgrade_total_cost_l29_29590

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l29_29590


namespace elsa_final_marbles_l29_29461

def start_marbles : ℕ := 40
def lost_breakfast : ℕ := 3
def given_susie : ℕ := 5
def new_marbles : ℕ := 12
def returned_marbles : ℕ := 2 * given_susie

def final_marbles : ℕ :=
  start_marbles - lost_breakfast - given_susie + new_marbles + returned_marbles

theorem elsa_final_marbles : final_marbles = 54 := by
  sorry

end elsa_final_marbles_l29_29461


namespace juan_ran_80_miles_l29_29895

def speed : Real := 10 -- miles per hour
def time : Real := 8   -- hours

theorem juan_ran_80_miles :
  speed * time = 80 := 
by
  sorry

end juan_ran_80_miles_l29_29895


namespace inequality_proof_l29_29070

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l29_29070


namespace find_a3_plus_a9_l29_29964

noncomputable def arithmetic_sequence (a : ℕ → ℕ) : Prop := 
∀ n m : ℕ, a (n + m) = a n + a m

theorem find_a3_plus_a9 (a : ℕ → ℕ) 
  (is_arithmetic : arithmetic_sequence a)
  (h : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 :=
sorry

end find_a3_plus_a9_l29_29964


namespace fraction_unchanged_l29_29823

-- Define the digit rotation
def rotate (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => d  -- for completeness, though we assume d only takes {0, 1, 6, 8, 9}

-- Define the condition for a fraction to be unchanged when flipped
def unchanged_when_flipped (numerator denominator : ℕ) : Prop :=
  let rotated_numerator := rotate numerator
  let rotated_denominator := rotate denominator
  rotated_numerator * denominator = rotated_denominator * numerator

-- Define the specific fraction 6/9
def specific_fraction_6_9 : Prop :=
  unchanged_when_flipped 6 9 ∧ 6 < 9

-- Theorem stating 6/9 is unchanged when its digits are flipped and it's a valid fraction
theorem fraction_unchanged : specific_fraction_6_9 :=
by
  sorry

end fraction_unchanged_l29_29823


namespace evaluate_g_l29_29192

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l29_29192


namespace g_composition_evaluation_l29_29956

def g (x : ℤ) : ℤ :=
  if x < 5 then x^3 + x^2 - 6 else 2 * x - 18

theorem g_composition_evaluation : g (g (g 16)) = 2 := by
  sorry

end g_composition_evaluation_l29_29956


namespace find_opposite_endpoint_l29_29438

/-- A utility function to model coordinate pairs as tuples -/
def coord_pair := (ℝ × ℝ)

-- Define the center and one endpoint
def center : coord_pair := (4, 6)
def endpoint1 : coord_pair := (2, 1)

-- Define the expected endpoint
def expected_endpoint2 : coord_pair := (6, 11)

/-- Definition of the opposite endpoint given the center and one endpoint -/
def opposite_endpoint (c : coord_pair) (p : coord_pair) : coord_pair :=
  let dx := c.1 - p.1
  let dy := c.2 - p.2
  (c.1 + dx, c.2 + dy)

/-- The proof statement for the problem -/
theorem find_opposite_endpoint :
  opposite_endpoint center endpoint1 = expected_endpoint2 :=
sorry

end find_opposite_endpoint_l29_29438


namespace shares_difference_l29_29390

-- conditions: the ratio is 3:7:12, and the difference between q and r's share is Rs. 3000
theorem shares_difference (x : ℕ) (h : 12 * x - 7 * x = 3000) : 7 * x - 3 * x = 2400 :=
by
  -- simply skip the proof since it's not required in the prompt
  sorry

end shares_difference_l29_29390


namespace conditional_probability_l29_29521

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability_l29_29521


namespace miles_run_by_harriet_l29_29818

def miles_run_by_all_runners := 285
def miles_run_by_katarina := 51
def miles_run_by_adriana := 74
def miles_run_by_tomas_tyler_harriet (total_run: ℝ) := (total_run - (miles_run_by_katarina + miles_run_by_adriana))

theorem miles_run_by_harriet : (miles_run_by_tomas_tyler_harriet miles_run_by_all_runners) / 3 = 53.33 := by
  sorry

end miles_run_by_harriet_l29_29818


namespace isosceles_triangle_side_length_l29_29144

theorem isosceles_triangle_side_length (n : ℕ) : 
  (∃ a b : ℕ, a ≠ 4 ∧ b ≠ 4 ∧ (a = b ∨ a = 4 ∨ b = 4) ∧ 
  a^2 - 6*a + n = 0 ∧ b^2 - 6*b + n = 0) → 
  (n = 8 ∨ n = 9) := 
by
  sorry

end isosceles_triangle_side_length_l29_29144


namespace rainfall_on_thursday_l29_29266

theorem rainfall_on_thursday
  (monday_am : ℝ := 2)
  (monday_pm : ℝ := 1)
  (tuesday_factor : ℝ := 2)
  (wednesday : ℝ := 0)
  (thursday : ℝ)
  (weekly_avg : ℝ := 4)
  (days_in_week : ℕ := 7)
  (total_weekly_rain : ℝ := days_in_week * weekly_avg) :
  2 * (monday_am + monday_pm + tuesday_factor * (monday_am + monday_pm) + thursday) 
    = total_weekly_rain
  → thursday = 5 :=
by
  sorry

end rainfall_on_thursday_l29_29266


namespace simplify_sqrt_sum_l29_29760

noncomputable def sqrt_72 : ℝ := Real.sqrt 72
noncomputable def sqrt_32 : ℝ := Real.sqrt 32
noncomputable def sqrt_27 : ℝ := Real.sqrt 27
noncomputable def result : ℝ := 10 * Real.sqrt 2 + 3 * Real.sqrt 3

theorem simplify_sqrt_sum :
  sqrt_72 + sqrt_32 + sqrt_27 = result :=
by
  sorry

end simplify_sqrt_sum_l29_29760


namespace total_spent_l29_29142

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l29_29142


namespace problem_statement_l29_29624

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 65 / 2) + 5 / 2)

theorem problem_statement :
  ∃ a b c : ℕ, (x ^ 100 = 2 * x ^ 98 + 16 * x ^ 96 + 13 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 42) ∧ (a + b + c = 337) :=
by
  sorry

end problem_statement_l29_29624


namespace smallest_product_is_298150_l29_29960

def digits : List ℕ := [5, 6, 7, 8, 9, 0]

theorem smallest_product_is_298150 :
  ∃ (a b c : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c = 298150) :=
sorry

end smallest_product_is_298150_l29_29960


namespace klinker_daughter_age_l29_29150

-- Define the conditions in Lean
variable (D : ℕ) -- ℕ is the natural number type in Lean

-- Define the theorem statement
theorem klinker_daughter_age (h1 : 35 + 15 = 50)
    (h2 : 50 = 2 * (D + 15)) : D = 10 := by
  sorry

end klinker_daughter_age_l29_29150


namespace no_solution_exists_l29_29678

theorem no_solution_exists (a b : ℤ) : ∃ c : ℤ, ∀ m n : ℤ, m^2 + a * m + b ≠ 2 * n^2 + 2 * n + c :=
by {
  -- Insert correct proof here
  sorry
}

end no_solution_exists_l29_29678


namespace simplify_and_evaluate_l29_29385

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l29_29385


namespace longer_side_of_rectangle_l29_29409

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
noncomputable def rectangle_area : ℝ := 3 * circle_area
noncomputable def shorter_side : ℝ := 2 * circle_radius

theorem longer_side_of_rectangle :
    ∃ (l : ℝ), l = rectangle_area / shorter_side ∧ l = 9 * Real.pi :=
by
  sorry

end longer_side_of_rectangle_l29_29409


namespace tenth_term_is_correct_l29_29290

-- Define the conditions
def first_term : ℚ := 3
def last_term : ℚ := 88
def num_terms : ℕ := 30
def common_difference : ℚ := (last_term - first_term) / (num_terms - 1)

-- Define the function for the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℚ := first_term + (n - 1) * common_difference

-- Prove that the 10th term is 852/29
theorem tenth_term_is_correct : nth_term 10 = 852 / 29 := 
by 
  -- Add the proof later, the statement includes the setup and conditions
  sorry

end tenth_term_is_correct_l29_29290


namespace no_real_solution_l29_29822

theorem no_real_solution : ∀ x : ℝ, ¬ ((2*x - 3*x + 7)^2 + 4 = -|2*x|) :=
by
  intro x
  have h1 : (2*x - 3*x + 7)^2 + 4 ≥ 4 := by
    sorry
  have h2 : -|2*x| ≤ 0 := by
    sorry
  -- The main contradiction follows from comparing h1 and h2
  sorry

end no_real_solution_l29_29822


namespace area_of_table_l29_29791

-- Definitions of the given conditions
def free_side_conditions (L W : ℝ) : Prop :=
  (L = 2 * W) ∧ (2 * W + L = 32)

-- Statement to prove the area of the rectangular table
theorem area_of_table {L W : ℝ} (h : free_side_conditions L W) : L * W = 128 := by
  sorry

end area_of_table_l29_29791


namespace cylinder_lateral_surface_area_l29_29285

theorem cylinder_lateral_surface_area 
  (diameter height : ℝ) 
  (h1 : diameter = 2) 
  (h2 : height = 2) : 
  2 * Real.pi * (diameter / 2) * height = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l29_29285


namespace average_words_written_l29_29088

def total_words : ℕ := 50000
def total_hours : ℕ := 100
def average_words_per_hour : ℕ := total_words / total_hours

theorem average_words_written :
  average_words_per_hour = 500 := 
by
  sorry

end average_words_written_l29_29088


namespace ellipse_equation_l29_29165

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (M : ℝ × ℝ)
  (h1 : 2 * a = 4)
  (h2 : 2 * b = 2 * a / 2)
  (h3 : M = (2, 1))
  (line_eq : ∀ k : ℝ, (y = 1 + k * (x - 2))) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 4 + y^2 = 1)) ∧
  (∃ k : ℝ, (k = -1/2) ∧ (∀ x y : ℝ, (y - 1 = k * (x - 2)) → (x + 2*y - 4 = 0))) :=
by
  sorry

end ellipse_equation_l29_29165


namespace hyperbola_eccentricity_l29_29766

-- Define the conditions and parameters for the problem
variables (m : ℝ) (c a e : ℝ)

-- Given conditions
def hyperbola_eq (m : ℝ) := ∀ x y : ℝ, (x^2 / m^2 - y^2 = 4)
def focal_distance : Prop := c = 4
def standard_hyperbola_form : Prop := a^2 = 4 * m^2 ∧ 4 = 4

-- Eccentricity definition
def eccentricity : Prop := e = c / a

-- Main theorem
theorem hyperbola_eccentricity (m : ℝ) (h_pos : 0 < m) (h_foc_dist : focal_distance c) (h_form : standard_hyperbola_form a m) :
  eccentricity e a c :=
by
  sorry

end hyperbola_eccentricity_l29_29766


namespace sin_sum_leq_3_sqrt_3_div_2_l29_29081

theorem sin_sum_leq_3_sqrt_3_div_2 (A B C : ℝ) (h_sum : A + B + C = Real.pi) (h_pos : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_leq_3_sqrt_3_div_2_l29_29081


namespace brokerage_percentage_calculation_l29_29339

theorem brokerage_percentage_calculation
  (face_value : ℝ)
  (discount_percentage : ℝ)
  (cost_price : ℝ)
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 6)
  (h_cost_price : cost_price = 94.2) :
  ((cost_price - (face_value - (discount_percentage / 100 * face_value))) / cost_price * 100) = 0.2124 := 
by
  sorry

end brokerage_percentage_calculation_l29_29339


namespace ratio_of_boys_to_girls_l29_29756

def boys_girls_ratio (b g : ℕ) : ℚ := b / g

theorem ratio_of_boys_to_girls (b g : ℕ) (h1 : b = g + 6) (h2 : g + b = 40) :
  boys_girls_ratio b g = 23 / 17 :=
by
  sorry

end ratio_of_boys_to_girls_l29_29756


namespace ab_cd_l29_29386

theorem ab_cd {a b c d : ℕ} {w x y z : ℕ}
  (hw : Prime w) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (horder : w < x ∧ x < y ∧ y < z)
  (hprod : w^a * x^b * y^c * z^d = 660) :
  (a + b) - (c + d) = 1 :=
by
  sorry

end ab_cd_l29_29386


namespace total_pairs_of_shoes_l29_29739

-- Conditions as Definitions
def blue_shoes := 540
def purple_shoes := 355
def green_shoes := purple_shoes  -- The number of green shoes is equal to the number of purple shoes

-- The theorem we need to prove
theorem total_pairs_of_shoes : blue_shoes + green_shoes + purple_shoes = 1250 := by
  sorry

end total_pairs_of_shoes_l29_29739


namespace max_distance_circle_ellipse_l29_29164

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 10 + p.2^2 = 1}
  ∀ (P Q : ℝ × ℝ), P ∈ circle → Q ∈ ellipse → 
  dist P Q ≤ 6 * Real.sqrt 2 :=
by
  intro circle ellipse P Q hP hQ
  sorry

end max_distance_circle_ellipse_l29_29164


namespace max_A_l29_29077

theorem max_A (A : ℝ) : (∀ (x y : ℕ), 0 < x → 0 < y → 3 * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) ↔ A ≤ 5 / 3 := by
  sorry

end max_A_l29_29077


namespace parents_without_full_time_jobs_l29_29877

theorem parents_without_full_time_jobs
  {total_parents mothers fathers : ℕ}
  (h_total_parents : total_parents = 100)
  (h_mothers_percentage : mothers = 60)
  (h_fathers_percentage : fathers = 40)
  (h_mothers_full_time : ℕ)
  (h_fathers_full_time : ℕ)
  (h_mothers_ratio : h_mothers_full_time = (5 * mothers) / 6)
  (h_fathers_ratio : h_fathers_full_time = (3 * fathers) / 4) :
  ((total_parents - (h_mothers_full_time + h_fathers_full_time)) * 100 / total_parents = 20) := sorry

end parents_without_full_time_jobs_l29_29877


namespace max_sum_of_digits_l29_29767

theorem max_sum_of_digits (X Y Z : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 1 ≤ Y ∧ Y ≤ 9) (hZ : 1 ≤ Z ∧ Z ≤ 9) (hXYZ : X > Y ∧ Y > Z) : 
  10 * X + 11 * Y + Z ≤ 185 :=
  sorry

end max_sum_of_digits_l29_29767


namespace imaginary_part_of_z_l29_29734

theorem imaginary_part_of_z {z : ℂ} (h : (1 + z) / I = 1 - z) : z.im = 1 := 
sorry

end imaginary_part_of_z_l29_29734


namespace books_bought_l29_29838

theorem books_bought (math_price : ℕ) (hist_price : ℕ) (total_cost : ℕ) (math_books : ℕ) (hist_books : ℕ) 
  (H : math_price = 4) (H1 : hist_price = 5) (H2 : total_cost = 396) (H3 : math_books = 54) 
  (H4 : math_books * math_price + hist_books * hist_price = total_cost) :
  math_books + hist_books = 90 :=
by sorry

end books_bought_l29_29838


namespace inequality_holds_l29_29669

theorem inequality_holds (a b : ℝ) (h : a ≠ b) : a^4 + 6 * a^2 * b^2 + b^4 > 4 * a * b * (a^2 + b^2) := 
by
  sorry

end inequality_holds_l29_29669


namespace evaluate_g_at_2_l29_29763

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem evaluate_g_at_2 : g 2 = 5 :=
by
  sorry

end evaluate_g_at_2_l29_29763


namespace workers_together_time_l29_29000

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 10) :
  let rateA := 1 / A_time
  let rateB := 1 / B_time
  let combined_rate := rateA + rateB
  combined_rate * (40 / 9) = 1 :=
by 
  sorry

end workers_together_time_l29_29000


namespace sounds_meet_at_x_l29_29369

theorem sounds_meet_at_x (d c s : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : 0 < s) :
  ∃ x : ℝ, x = d / 2 * (1 + s / c) ∧ x <= d ∧ x > 0 :=
by
  sorry

end sounds_meet_at_x_l29_29369


namespace sufficient_but_not_necessary_condition_l29_29264

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ ((1 / a < 1) → (a > 1 ∨ a < 0)) → 
  (∀ (P Q : Prop), (P → Q) → (Q → P ∨ False) → P ∧ ¬Q → False) :=
by
  sorry

end sufficient_but_not_necessary_condition_l29_29264


namespace can_form_set_l29_29593

-- Define each group of objects based on given conditions
def famous_movie_stars : Type := sorry
def small_rivers_in_our_country : Type := sorry
def students_2012_senior_class_Panzhihua : Type := sorry
def difficult_high_school_math_problems : Type := sorry

-- Define the property of having well-defined elements
def has_definite_elements (T : Type) : Prop := sorry

-- The groups in terms of propositions
def group_A : Prop := ¬ has_definite_elements famous_movie_stars
def group_B : Prop := ¬ has_definite_elements small_rivers_in_our_country
def group_C : Prop := has_definite_elements students_2012_senior_class_Panzhihua
def group_D : Prop := ¬ has_definite_elements difficult_high_school_math_problems

-- We need to prove that group C can form a set
theorem can_form_set : group_C :=
by
  sorry

end can_form_set_l29_29593


namespace towel_bleach_volume_decrease_l29_29005

theorem towel_bleach_volume_decrease :
  ∀ (L B T : ℝ) (L' B' T' : ℝ),
  (L' = L * 0.75) →
  (B' = B * 0.70) →
  (T' = T * 0.90) →
  (L * B * T = 1000000) →
  ((L * B * T - L' * B' * T') / (L * B * T) * 100) = 52.75 :=
by
  intros L B T L' B' T' hL' hB' hT' hV
  sorry

end towel_bleach_volume_decrease_l29_29005


namespace triangle_angle_C_right_l29_29943

theorem triangle_angle_C_right {a b c A B C : ℝ}
  (h1 : a / Real.sin B + b / Real.sin A = 2 * c) 
  (h2 : a / Real.sin A = b / Real.sin B) 
  (h3 : b / Real.sin B = c / Real.sin C) : 
  C = Real.pi / 2 :=
by sorry

end triangle_angle_C_right_l29_29943


namespace find_initial_number_l29_29567

theorem find_initial_number (x : ℝ) (h : x + 12.808 - 47.80600000000004 = 3854.002) : x = 3889 := by
  sorry

end find_initial_number_l29_29567


namespace square_free_condition_l29_29083

/-- Define square-free integer -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Define the problem in Lean -/
theorem square_free_condition (p : ℕ) (hp : p ≥ 3 ∧ Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q ∧ q < p → square_free (p - (p / q) * q)) ↔
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13 := by
  sorry

end square_free_condition_l29_29083


namespace equivalent_problem_l29_29830

def f (x : ℤ) : ℤ := 9 - x

def g (x : ℤ) : ℤ := x - 9

theorem equivalent_problem : g (f 15) = -15 := sorry

end equivalent_problem_l29_29830


namespace remainder_problem_l29_29629

theorem remainder_problem {x y z : ℤ} (h1 : x % 102 = 56) (h2 : y % 154 = 79) (h3 : z % 297 = 183) :
  x % 19 = 18 ∧ y % 22 = 13 ∧ z % 33 = 18 :=
by
  sorry

end remainder_problem_l29_29629


namespace at_least_one_corner_square_selected_l29_29051

theorem at_least_one_corner_square_selected :
  let total_squares := 16
  let total_corners := 4
  let total_non_corners := 12
  let ways_to_select_3_from_total := Nat.choose total_squares 3
  let ways_to_select_3_from_non_corners := Nat.choose total_non_corners 3
  let probability_no_corners := (ways_to_select_3_from_non_corners : ℚ) / ways_to_select_3_from_total
  let probability_at_least_one_corner := 1 - probability_no_corners
  probability_at_least_one_corner = (17 / 28 : ℚ) :=
by
  sorry

end at_least_one_corner_square_selected_l29_29051


namespace perfect_square_trinomial_l29_29728

theorem perfect_square_trinomial {m : ℝ} :
  (∃ (a : ℝ), x^2 + 2 * m * x + 9 = (x + a)^2) → (m = 3 ∨ m = -3) :=
sorry

end perfect_square_trinomial_l29_29728


namespace spadesuit_calculation_l29_29226

def spadesuit (x y : ℝ) : ℝ := (x + 2 * y) ^ 2 * (x - y)

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 2 3) = 1046875 :=
by
  sorry

end spadesuit_calculation_l29_29226


namespace jake_has_one_more_balloon_than_allan_l29_29666

def balloons_allan : ℕ := 6
def balloons_jake_initial : ℕ := 3
def balloons_jake_additional : ℕ := 4

theorem jake_has_one_more_balloon_than_allan :
  (balloons_jake_initial + balloons_jake_additional - balloons_allan) = 1 :=
by
  sorry

end jake_has_one_more_balloon_than_allan_l29_29666


namespace original_price_correct_percentage_growth_rate_l29_29366

-- Definitions and conditions
def original_price := 45
def sale_discount := 15
def price_after_discount := original_price - sale_discount

def initial_cost_before_event := 90
def final_cost_during_event := 120
def ratio_of_chickens := 2

def initial_buyers := 50
def increase_percentage := 20
def total_sales := 5460
def time_slots := 2  -- 1 hour = 2 slots of 30 minutes each

-- The problem: Prove the original price and growth rate
theorem original_price_correct (x : ℕ) : (120 / (x - 15) = 2 * (90 / x) → x = original_price) :=
by
  sorry

theorem percentage_growth_rate (m : ℕ) :
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = total_sales / (original_price - sale_discount) →
  m = increase_percentage) :=
by
  sorry

end original_price_correct_percentage_growth_rate_l29_29366


namespace tabs_per_window_l29_29466

def totalTabs (browsers windowsPerBrowser tabsOpened : Nat) : Nat :=
  tabsOpened / (browsers * windowsPerBrowser)

theorem tabs_per_window : totalTabs 2 3 60 = 10 := by
  sorry

end tabs_per_window_l29_29466


namespace total_children_on_bus_after_stop_l29_29847

theorem total_children_on_bus_after_stop (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 18) (h2 : additional = 7) : total = 25 :=
by sorry

end total_children_on_bus_after_stop_l29_29847


namespace taxi_ride_total_cost_l29_29026

theorem taxi_ride_total_cost :
  let base_fee := 1.50
  let cost_per_mile := 0.25
  let distance1 := 5
  let distance2 := 8
  let distance3 := 3
  let cost1 := base_fee + distance1 * cost_per_mile
  let cost2 := base_fee + distance2 * cost_per_mile
  let cost3 := base_fee + distance3 * cost_per_mile
  cost1 + cost2 + cost3 = 8.50 := sorry

end taxi_ride_total_cost_l29_29026


namespace sin_double_angle_l29_29333

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_l29_29333


namespace no_two_digit_multiples_of_3_5_7_l29_29485

theorem no_two_digit_multiples_of_3_5_7 : ∀ n : ℕ, 10 ≤ n ∧ n < 100 → ¬ (3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := 
by
  intro n
  intro h
  intro h_div
  sorry

end no_two_digit_multiples_of_3_5_7_l29_29485


namespace PropositionA_PropositionD_l29_29036

-- Proposition A: a > 1 is a sufficient but not necessary condition for 1/a < 1.
theorem PropositionA (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by sorry

-- PropositionD: a ≠ 0 is a necessary but not sufficient condition for ab ≠ 0.
theorem PropositionD (a b : ℝ) (h : a ≠ 0) : a * b ≠ 0 :=
by sorry
 
end PropositionA_PropositionD_l29_29036


namespace solve_y_l29_29245

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l29_29245


namespace find_natural_numbers_l29_29733

theorem find_natural_numbers (n : ℕ) (x : ℕ) (y : ℕ) (hx : n = 10 * x + y) (hy : 10 * x + y = 14 * x) : n = 14 ∨ n = 28 :=
by
  sorry

end find_natural_numbers_l29_29733


namespace lilly_daily_savings_l29_29604

-- Conditions
def days_until_birthday : ℕ := 22
def flowers_to_buy : ℕ := 11
def cost_per_flower : ℕ := 4

-- Definition we want to prove
def total_cost : ℕ := flowers_to_buy * cost_per_flower
def daily_savings : ℕ := total_cost / days_until_birthday

theorem lilly_daily_savings : daily_savings = 2 := by
  sorry

end lilly_daily_savings_l29_29604


namespace product_greater_than_constant_l29_29269

noncomputable def f (x m : ℝ) := Real.log x - (m + 1) * x + (1 / 2) * m * x ^ 2
noncomputable def g (x m : ℝ) := Real.log x - (m + 1) * x

variables {x1 x2 m : ℝ} 
  (h1 : g x1 m = 0)
  (h2 : g x2 m = 0)
  (h3 : x2 > Real.exp 1 * x1)

theorem product_greater_than_constant :
  x1 * x2 > 2 / (Real.exp 1 - 1) :=
sorry

end product_greater_than_constant_l29_29269


namespace never_return_to_start_l29_29789

variable {City : Type} [MetricSpace City]

-- Conditions
variable (C : ℕ → City)  -- C is the sequence of cities
variable (dist : City → City → ℝ)  -- distance function
variable (furthest : City → City)  -- function that maps each city to the furthest city from it
variable (start : City)  -- initial city

-- Assuming C satisfies the properties in the problem statement
axiom initial_city : C 1 = start
axiom furthest_city_step : ∀ n, C (n + 1) = furthest (C n)
axiom no_ambiguity : ∀ c1 c2, (dist c1 (furthest c1) > dist c1 c2 ↔ c2 ≠ furthest c1)

-- Define the problem to prove that if C₁ ≠ C₃, then ∀ n ≥ 4, Cₙ ≠ C₁
theorem never_return_to_start (h : C 1 ≠ C 3) : ∀ n ≥ 4, C n ≠ start := sorry

end never_return_to_start_l29_29789


namespace candy_bar_cost_l29_29671

def cost_soft_drink : ℕ := 2
def num_candy_bars : ℕ := 5
def total_spent : ℕ := 27
def cost_per_candy_bar (C : ℕ) : Prop := cost_soft_drink + num_candy_bars * C = total_spent

-- The theorem we want to prove
theorem candy_bar_cost (C : ℕ) (h : cost_per_candy_bar C) : C = 5 :=
by sorry

end candy_bar_cost_l29_29671


namespace max_intersections_quadrilateral_l29_29531

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end max_intersections_quadrilateral_l29_29531


namespace problem1_problem2_l29_29158

noncomputable def vec (α : ℝ) (β : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (Real.cos α, Real.sin α, Real.cos β, -Real.sin β)

theorem problem1 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : (Real.sqrt ((Real.cos α - Real.cos β) ^ 2 + (Real.sin α + Real.sin β) ^ 2)) = (Real.sqrt 10) / 5) :
  Real.cos (α + β) = 4 / 5 :=
by
  sorry

theorem problem2 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = 4 / 5) :
  Real.cos β = 24 / 25 :=
by
  sorry

end problem1_problem2_l29_29158


namespace find_x_of_equation_l29_29646

theorem find_x_of_equation (x : ℝ) (hx : x ≠ 0) : (7 * x)^4 = (14 * x)^3 → x = 8 / 7 :=
by
  intro h
  sorry

end find_x_of_equation_l29_29646


namespace lemonade_glasses_from_fruit_l29_29496

noncomputable def lemons_per_glass : ℕ := 2
noncomputable def oranges_per_glass : ℕ := 1
noncomputable def total_lemons : ℕ := 18
noncomputable def total_oranges : ℕ := 10
noncomputable def grapefruits : ℕ := 6
noncomputable def lemons_per_grapefruit : ℕ := 2
noncomputable def oranges_per_grapefruit : ℕ := 1

theorem lemonade_glasses_from_fruit :
  (total_lemons / lemons_per_glass) = 9 →
  (total_oranges / oranges_per_glass) = 10 →
  min (total_lemons / lemons_per_glass) (total_oranges / oranges_per_glass) = 9 →
  (grapefruits * lemons_per_grapefruit = 12) →
  (grapefruits * oranges_per_grapefruit = 6) →
  (9 + grapefruits) = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lemonade_glasses_from_fruit_l29_29496


namespace new_mean_after_adding_constant_l29_29662

theorem new_mean_after_adding_constant (S : ℝ) (average : ℝ) (n : ℕ) (a : ℝ) :
  n = 15 → average = 40 → a = 15 → S = n * average → (S + n * a) / n = 55 :=
by
  intros hn haverage ha hS
  sorry

end new_mean_after_adding_constant_l29_29662


namespace correct_equation_l29_29689

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l29_29689


namespace num_boys_l29_29289

-- Definitions as per the conditions
def boys (d : ℕ) := 2 * d
def reducedGirls (d : ℕ) := d - 1

-- Lean statement for the proof problem
theorem num_boys (d b : ℕ) 
  (h1 : b = boys d)
  (h2 : b = reducedGirls d + 8) : b = 14 :=
by {
  sorry
}

end num_boys_l29_29289


namespace minimum_rubles_to_reverse_order_of_chips_100_l29_29080

noncomputable def minimum_rubles_to_reverse_order_of_chips (n : ℕ) : ℕ :=
if n = 100 then 61 else 0

theorem minimum_rubles_to_reverse_order_of_chips_100 :
  minimum_rubles_to_reverse_order_of_chips 100 = 61 :=
by sorry

end minimum_rubles_to_reverse_order_of_chips_100_l29_29080


namespace number_of_possible_values_of_a_l29_29774

theorem number_of_possible_values_of_a :
  ∃ (a_values : Finset ℕ), 
    (∀ a ∈ a_values, 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27 ∧ 0 < a) ∧
    a_values.card = 2 :=
by
  sorry

end number_of_possible_values_of_a_l29_29774


namespace total_time_spent_l29_29614

variable (B I E M EE ST ME : ℝ)

def learn_basic_rules : ℝ := B
def learn_intermediate_level : ℝ := I
def learn_expert_level : ℝ := E
def learn_master_level : ℝ := M
def endgame_exercises : ℝ := EE
def middle_game_strategy_tactics : ℝ := ST
def mentoring : ℝ := ME

theorem total_time_spent :
  B = 2 →
  I = 75 * B →
  E = 50 * (B + I) →
  M = 30 * E →
  EE = 0.25 * I →
  ST = 2 * EE →
  ME = 0.5 * E →
  B + I + E + M + EE + ST + ME = 235664.5 :=
by
  intros hB hI hE hM hEE hST hME
  rw [hB, hI, hE, hM, hEE, hST, hME]
  sorry

end total_time_spent_l29_29614


namespace james_car_purchase_l29_29193

/-- 
James sold his $20,000 car for 80% of its value, 
then bought a $30,000 sticker price car, 
and he was out of pocket $11,000. 
James bought the new car for 90% of its value. 
-/
theorem james_car_purchase (V_1 P_1 V_2 O P : ℝ)
  (hV1 : V_1 = 20000)
  (hP1 : P_1 = 80)
  (hV2 : V_2 = 30000)
  (hO : O = 11000)
  (hSaleOld : (P_1 / 100) * V_1 = 16000)
  (hDiff : 16000 + O = 27000)
  (hPurchase : (P / 100) * V_2 = 27000) :
  P = 90 := 
sorry

end james_car_purchase_l29_29193


namespace add_in_base6_l29_29875

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end add_in_base6_l29_29875


namespace original_population_divisor_l29_29884

theorem original_population_divisor (a b c : ℕ) (ha : ∃ a, ∃ b, ∃ c, a^2 + 120 = b^2 ∧ b^2 + 80 = c^2) :
  7 ∣ a :=
by
  sorry

end original_population_divisor_l29_29884


namespace quadratic_roots_inequality_solution_set_l29_29898

-- Problem 1 statement
theorem quadratic_roots : 
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := 
by
  sorry

-- Problem 2 statement
theorem inequality_solution_set :
  (∀ x : ℝ, (x - 2 * (x - 1) ≤ 1 ∧ (1 + x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 2) :=
by
  sorry

end quadratic_roots_inequality_solution_set_l29_29898


namespace geometric_sequence_a2_a6_l29_29045

theorem geometric_sequence_a2_a6 (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = r * a n) (h₄ : a 4 = 4) :
  a 2 * a 6 = 16 :=
sorry

end geometric_sequence_a2_a6_l29_29045


namespace fibonacci_money_problem_l29_29777

variable (x : ℕ)

theorem fibonacci_money_problem (h : 0 < x - 6) (eq_amounts : 90 / (x - 6) = 120 / x) : 
    90 / (x - 6) = 120 / x :=
sorry

end fibonacci_money_problem_l29_29777


namespace vertices_of_regular_hexagonal_pyramid_l29_29213

-- Define a structure for a regular hexagonal pyramid
structure RegularHexagonalPyramid where
  baseVertices : Nat
  apexVertices : Nat

-- Define a specific regular hexagonal pyramid with given conditions
def regularHexagonalPyramid : RegularHexagonalPyramid :=
  { baseVertices := 6, apexVertices := 1 }

-- The theorem stating the number of vertices of the pyramid
theorem vertices_of_regular_hexagonal_pyramid : regularHexagonalPyramid.baseVertices + regularHexagonalPyramid.apexVertices = 7 := 
  by
  sorry

end vertices_of_regular_hexagonal_pyramid_l29_29213


namespace sum_of_three_numbers_l29_29932

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 :=
by sorry

end sum_of_three_numbers_l29_29932


namespace distance_BF_l29_29275

-- Given the focus F of the parabola y^2 = 4x
def focus_of_parabola : (ℝ × ℝ) := (1, 0)

-- Points A and B lie on the parabola y^2 = 4x
def point_A (x y : ℝ) := y^2 = 4 * x
def point_B (x y : ℝ) := y^2 = 4 * x

-- The line through F intersects the parabola at points A and B, and |AF| = 2
def distance_AF : ℝ := 2

-- Prove that |BF| = 2
theorem distance_BF : ∀ (A B F : ℝ × ℝ), 
  A = (1, F.2) → 
  B = (1, -F.2) → 
  F = (1, 0) → 
  |A.1 - F.1| + |A.2 - F.2| = distance_AF → 
  |B.1 - F.1| + |B.2 - F.2| = 2 :=
by
  intros A B F hA hB hF d_AF
  sorry

end distance_BF_l29_29275


namespace integer_sum_l29_29055

theorem integer_sum {p q r s : ℤ} 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 22 := 
sorry

end integer_sum_l29_29055


namespace equation_of_perpendicular_line_l29_29188

theorem equation_of_perpendicular_line (x y : ℝ) (l1 : 2*x - 3*y + 4 = 0) (pt : x = -2 ∧ y = -3) :
  3*(-2) + 2*(-3) + 12 = 0 := by
  sorry

end equation_of_perpendicular_line_l29_29188


namespace sequence_properties_l29_29599

def f (x : ℝ) : ℝ := x^3 + 3 * x

variables {a_5 a_8 : ℝ}
variables {S_12 : ℝ}

axiom a5_condition : (a_5 - 1)^3 + 3 * a_5 = 4
axiom a8_condition : (a_8 - 1)^3 + 3 * a_8 = 2

theorem sequence_properties : (a_5 > a_8) ∧ (S_12 = 12) :=
by {
  sorry
}

end sequence_properties_l29_29599


namespace power_expression_simplify_l29_29216

theorem power_expression_simplify :
  (1 / (-5^2)^3) * (-5)^8 * Real.sqrt 5 = 5^(5/2) :=
by
  sorry

end power_expression_simplify_l29_29216


namespace find_number_l29_29448

theorem find_number (x: ℝ) (h: (6 * x) / 2 - 5 = 25) : x = 10 :=
by
  sorry

end find_number_l29_29448


namespace problem_1163_prime_and_16424_composite_l29_29021

theorem problem_1163_prime_and_16424_composite :
  let x := 1910 * 10000 + 1112
  let a := 1163
  let b := 16424
  x = a * b →
  Prime a ∧ ¬ Prime b :=
by
  intros h
  sorry

end problem_1163_prime_and_16424_composite_l29_29021


namespace div_condition_l29_29615

theorem div_condition (N : ℤ) : (∃ k : ℤ, N^2 - 71 = k * (7 * N + 55)) ↔ (N = 57 ∨ N = -8) := 
by
  sorry

end div_condition_l29_29615


namespace parabola_vertex_coordinate_l29_29692

theorem parabola_vertex_coordinate :
  ∀ x_P : ℝ, 
  (P : ℝ × ℝ) → 
  (P = (x_P, 1/2 * x_P^2)) → 
  (dist P (0, 1/2) = 3) →
  P.2 = 5 / 2 :=
by sorry

end parabola_vertex_coordinate_l29_29692


namespace prime_square_mod_12_l29_29865

theorem prime_square_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 
  (p ^ 2) % 12 = 1 :=
sorry

end prime_square_mod_12_l29_29865


namespace required_speed_is_85_l29_29944

-- Definitions based on conditions
def speed1 := 60
def time1 := 3
def total_time := 5
def average_speed := 70

-- Derived conditions
def distance1 := speed1 * time1
def total_distance := average_speed * total_time
def remaining_distance := total_distance - distance1
def remaining_time := total_time - time1
def required_speed := remaining_distance / remaining_time

-- Theorem statement
theorem required_speed_is_85 : required_speed = 85 := by
    sorry

end required_speed_is_85_l29_29944


namespace john_total_spent_is_correct_l29_29610

noncomputable def john_spent_total (original_cost : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_cost := original_cost - (discount_rate / 100 * original_cost)
  let cost_with_tax := discounted_cost + (sales_tax_rate / 100 * discounted_cost)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_cost_with_tax := lightsaber_cost + (sales_tax_rate / 100 * lightsaber_cost)
  cost_with_tax + lightsaber_cost_with_tax

theorem john_total_spent_is_correct :
  john_spent_total 1200 20 8 = 3628.80 :=
by
  sorry

end john_total_spent_is_correct_l29_29610


namespace exercise_l29_29221

-- Define the given expression.
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- Define the general form expression.
def g (x h k : ℝ) (a : ℝ) := a * (x - h)^2 + k

-- Prove that a + h + k = 6 when expressing f(x) in the form a(x-h)^2 + k.
theorem exercise : ∃ a h k : ℝ, (∀ x : ℝ, f x = g x h k a) ∧ (a + h + k = 6) :=
by
  sorry

end exercise_l29_29221


namespace midpoint_distance_trapezoid_l29_29460

theorem midpoint_distance_trapezoid (x : ℝ) : 
  let AD := x
  let BC := 5
  PQ = (|x - 5| / 2) :=
sorry

end midpoint_distance_trapezoid_l29_29460


namespace lcm_24_36_45_l29_29798

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l29_29798


namespace alpha_beta_range_l29_29611

theorem alpha_beta_range (α β : ℝ) (h1 : - (π / 2) < α) (h2 : α < β) (h3 : β < π) : 
- 3 * (π / 2) < α - β ∧ α - β < 0 :=
by
  sorry

end alpha_beta_range_l29_29611


namespace ratio_of_sides_product_of_areas_and_segments_l29_29814

variable (S S' S'' : ℝ) (a a' : ℝ)

-- Given condition
axiom proportion_condition : S / S'' = a / a'

-- Proofs that need to be verified
theorem ratio_of_sides (S S' : ℝ) (a a' : ℝ) (h : S / S'' = a / a') :
  S / a = S' / a' :=
sorry

theorem product_of_areas_and_segments (S S' : ℝ) (a a' : ℝ) (h: S / S'' = a / a') :
  S * a' = S' * a :=
sorry

end ratio_of_sides_product_of_areas_and_segments_l29_29814


namespace brad_read_more_books_l29_29176

-- Definitions based on the given conditions
def books_william_read_last_month : ℕ := 6
def books_brad_read_last_month : ℕ := 3 * books_william_read_last_month
def books_brad_read_this_month : ℕ := 8
def books_william_read_this_month : ℕ := 2 * books_brad_read_this_month

-- Totals
def total_books_brad_read : ℕ := books_brad_read_last_month + books_brad_read_this_month
def total_books_william_read : ℕ := books_william_read_last_month + books_william_read_this_month

-- The statement to prove
theorem brad_read_more_books : total_books_brad_read = total_books_william_read + 4 := by
  sorry

end brad_read_more_books_l29_29176


namespace f_one_f_a_f_f_a_l29_29617

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem f_one : f 1 = 5 := by
  sorry

theorem f_a (a : ℝ) : f a = 2 * a + 3 := by
  sorry

theorem f_f_a (a : ℝ) : f (f a) = 4 * a + 9 := by
  sorry

end f_one_f_a_f_f_a_l29_29617


namespace inverse_proportion_symmetric_l29_29232

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l29_29232


namespace range_for_a_l29_29543

theorem range_for_a (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  sorry

end range_for_a_l29_29543


namespace Events_B_and_C_mutex_l29_29642

-- Definitions of events based on scores
def EventA (score : ℕ) := score ≥ 1 ∧ score ≤ 10
def EventB (score : ℕ) := score > 5 ∧ score ≤ 10
def EventC (score : ℕ) := score > 1 ∧ score < 6
def EventD (score : ℕ) := score > 0 ∧ score < 6

-- Mutually exclusive definition:
def mutually_exclusive (P Q : ℕ → Prop) := ∀ (x : ℕ), ¬ (P x ∧ Q x)

-- The proof statement:
theorem Events_B_and_C_mutex : mutually_exclusive EventB EventC :=
by
  sorry

end Events_B_and_C_mutex_l29_29642


namespace complement_U_M_l29_29048

noncomputable def U : Set ℝ := {x : ℝ | x > 0}

noncomputable def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}

theorem complement_U_M : (U \ M) = {x : ℝ | x ≥ 2} := 
by
  sorry

end complement_U_M_l29_29048


namespace measure_of_unknown_angle_in_hexagon_l29_29852

theorem measure_of_unknown_angle_in_hexagon :
  let a1 := 135
  let a2 := 105
  let a3 := 87
  let a4 := 120
  let a5 := 78
  let total_internal_angles := 180 * (6 - 2)
  let known_sum := a1 + a2 + a3 + a4 + a5
  let Q := total_internal_angles - known_sum
  Q = 195 :=
by
  sorry

end measure_of_unknown_angle_in_hexagon_l29_29852


namespace derivative_and_value_l29_29796

-- Given conditions
def eqn (x y : ℝ) : Prop := 10 * x^3 + 4 * x^2 * y + y^2 = 0

-- The derivative y'
def y_prime (x y y' : ℝ) : Prop := y' = (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

-- Specific values derivatives
def y_prime_at_x_neg2_y_4 (y' : ℝ) : Prop := y' = -7 / 3

-- The main theorem
theorem derivative_and_value (x y y' : ℝ) 
  (h1 : eqn x y) (x_neg2 : x = -2) (y_4 : y = 4) : 
  y_prime x y y' ∧ y_prime_at_x_neg2_y_4 y' :=
sorry

end derivative_and_value_l29_29796


namespace product_equals_eight_l29_29894

theorem product_equals_eight : (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7) = 8 := 
sorry

end product_equals_eight_l29_29894


namespace f_zero_eq_zero_l29_29084

-- Define the problem conditions
variable {f : ℝ → ℝ}
variables (h_odd : ∀ x : ℝ, f (-x) = -f (x))
variables (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variables (h_eq : ∀ x : ℝ, f (1 - x) - f (1 + x) + 2 * x = 0)
variables (h_mono : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem
theorem f_zero_eq_zero : f 0 = 0 :=
by sorry

end f_zero_eq_zero_l29_29084


namespace trailing_zeros_1_to_100_l29_29619

def count_multiples (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def trailing_zeros_in_range (n : ℕ) : ℕ :=
  let multiples_of_5 := count_multiples n 5
  let multiples_of_25 := count_multiples n 25
  multiples_of_5 + multiples_of_25

theorem trailing_zeros_1_to_100 : trailing_zeros_in_range 100 = 24 := by
  sorry

end trailing_zeros_1_to_100_l29_29619


namespace total_toys_given_l29_29996

theorem total_toys_given (toys_for_boys : ℕ) (toys_for_girls : ℕ) (h1 : toys_for_boys = 134) (h2 : toys_for_girls = 269) : 
  toys_for_boys + toys_for_girls = 403 := 
by 
  sorry

end total_toys_given_l29_29996


namespace number_times_half_squared_is_eight_l29_29579

noncomputable def num : ℝ := 32

theorem number_times_half_squared_is_eight :
  (num * (1 / 2) ^ 2 = 2 ^ 3) :=
by
  sorry

end number_times_half_squared_is_eight_l29_29579


namespace calculate_total_selling_price_l29_29335

noncomputable def total_selling_price (cost_price1 cost_price2 cost_price3 profit_percent1 profit_percent2 profit_percent3 : ℝ) : ℝ :=
  let sp1 := cost_price1 + (profit_percent1 / 100 * cost_price1)
  let sp2 := cost_price2 + (profit_percent2 / 100 * cost_price2)
  let sp3 := cost_price3 + (profit_percent3 / 100 * cost_price3)
  sp1 + sp2 + sp3

theorem calculate_total_selling_price :
  total_selling_price 550 750 1000 30 25 20 = 2852.5 :=
by
  -- proof omitted
  sorry

end calculate_total_selling_price_l29_29335


namespace wrapping_paper_per_present_l29_29068

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l29_29068


namespace larger_number_of_product_56_and_sum_15_l29_29607

theorem larger_number_of_product_56_and_sum_15 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := 
by
  sorry

end larger_number_of_product_56_and_sum_15_l29_29607


namespace jon_toaster_total_cost_l29_29558

def total_cost_toaster (MSRP : ℝ) (std_ins_pct : ℝ) (premium_upgrade_cost : ℝ) (state_tax_pct : ℝ) (environmental_fee : ℝ) : ℝ :=
  let std_ins_cost := std_ins_pct * MSRP
  let premium_ins_cost := std_ins_cost + premium_upgrade_cost
  let subtotal_before_tax := MSRP + premium_ins_cost
  let state_tax := state_tax_pct * subtotal_before_tax
  let total_before_env_fee := subtotal_before_tax + state_tax
  total_before_env_fee + environmental_fee

theorem jon_toaster_total_cost :
  total_cost_toaster 30 0.2 7 0.5 5 = 69.5 :=
by
  sorry

end jon_toaster_total_cost_l29_29558


namespace intercept_condition_slope_condition_l29_29196

theorem intercept_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 - 2 * m - 3) * -3 + (2 * m^2 + m - 1) * 0 + (-2 * m + 6) = 0 → 
  m = -5 / 3 := 
  sorry

theorem slope_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 + 2 * m - 4) = 0 → 
  m = 4 / 3 := 
  sorry

end intercept_condition_slope_condition_l29_29196


namespace unit_prices_max_helmets_A_l29_29445

open Nat Real

-- Given conditions
variables (x y : ℝ)
variables (m : ℕ)

def wholesale_price_A := 30
def wholesale_price_B := 20
def price_difference := 15
def revenue_A := 450
def revenue_B := 600
def total_helmets := 100
def budget := 2350

-- Part 1: Prove the unit prices of helmets A and B
theorem unit_prices :
  ∃ (price_A price_B : ℝ), 
    (price_A = price_B + price_difference) ∧ 
    (revenue_B / price_B = 2 * revenue_A / price_A) ∧
    (price_B = 30) ∧
    (price_A = 45) :=
by
  sorry

-- Part 2: Prove the maximum number of helmets of type A that can be purchased
theorem max_helmets_A :
  ∃ (m : ℕ), 
    (30 * m + 20 * (total_helmets - m) ≤ budget) ∧
    (m ≤ 35) :=
by
  sorry

end unit_prices_max_helmets_A_l29_29445


namespace remainder_of_3x_minus_2y_mod_30_l29_29861

theorem remainder_of_3x_minus_2y_mod_30
  (p q : ℤ) (x y : ℤ)
  (hx : x = 60 * p + 53)
  (hy : y = 45 * q + 28) :
  (3 * x - 2 * y) % 30 = 13 :=
by 
  sorry

end remainder_of_3x_minus_2y_mod_30_l29_29861


namespace cost_prices_sum_l29_29337

theorem cost_prices_sum
  (W B : ℝ)
  (h1 : 0.9 * W + 196 = 1.04 * W)
  (h2 : 1.08 * B - 150 = 1.02 * B) :
  W + B = 3900 := 
sorry

end cost_prices_sum_l29_29337


namespace sin_negative_300_eq_l29_29172

theorem sin_negative_300_eq : Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
by
  -- Periodic property of sine function: sin(theta) = sin(theta + 360 * n)
  have periodic_property : ∀ θ n : ℤ, Real.sin θ = Real.sin (θ + n * 2 * Real.pi) :=
    by sorry
  -- Known value: sin(60 degrees) = sqrt(3)/2
  have sin_60 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  -- Apply periodic_property to transform sin(-300 degrees) to sin(60 degrees)
  sorry

end sin_negative_300_eq_l29_29172


namespace find_range_of_function_l29_29974

variable (a : ℝ) (x : ℝ)

def func (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem find_range_of_function (a : ℝ) :
  if a < 0 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -1 ≤ y ∧ y ≤ 3 - 4*a
  else if 0 ≤ a ∧ a ≤ 1 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ 3 - 4*a
  else if 1 < a ∧ a ≤ 2 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ -1
  else
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ 3 - 4*a ≤ y ∧ y ≤ -1
:= sorry

end find_range_of_function_l29_29974


namespace share_of_a_l29_29935

variables (A B C : ℝ)

def conditions :=
  A = (2 / 3) * (B + C) ∧
  B = (2 / 3) * (A + C) ∧
  A + B + C = 700

theorem share_of_a (h : conditions A B C) : A = 280 :=
by { sorry }

end share_of_a_l29_29935


namespace sum_of_remainders_l29_29248

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l29_29248


namespace average_price_mixed_sugar_l29_29891

def average_selling_price_per_kg (weightA weightB weightC costA costB costC : ℕ) := 
  (costA * weightA + costB * weightB + costC * weightC) / (weightA + weightB + weightC : ℚ)

theorem average_price_mixed_sugar : 
  average_selling_price_per_kg 3 2 5 28 20 12 = 18.4 := 
by
  sorry

end average_price_mixed_sugar_l29_29891


namespace union_of_sets_l29_29352

open Set

theorem union_of_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} := by
  sorry

end union_of_sets_l29_29352


namespace fraction_meaningful_domain_l29_29403

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l29_29403


namespace simplify_expression_l29_29294

theorem simplify_expression : 1 + 3 / (2 + 5 / 6) = 35 / 17 := 
  sorry

end simplify_expression_l29_29294


namespace intersection_S_T_eq_T_l29_29050

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l29_29050


namespace fraction_red_after_tripling_l29_29836

-- Define the initial conditions
def initial_fraction_blue : ℚ := 4 / 7
def initial_fraction_red : ℚ := 1 - initial_fraction_blue
def triple_red_fraction (initial_red : ℚ) : ℚ := 3 * initial_red

-- Theorem statement
theorem fraction_red_after_tripling :
  let x := 1 -- Any number since it will cancel out
  let initial_red_marble := initial_fraction_red * x
  let total_marble := x
  let new_red_marble := triple_red_fraction initial_red_marble
  let new_total_marble := initial_fraction_blue * x + new_red_marble
  (new_red_marble / new_total_marble) = 9 / 13 :=
by
  sorry

end fraction_red_after_tripling_l29_29836


namespace value_of_a_l29_29410

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end value_of_a_l29_29410


namespace largest_whole_number_less_than_100_l29_29625

theorem largest_whole_number_less_than_100 (x : ℕ) (h1 : 7 * x < 100) (h_max : ∀ y : ℕ, 7 * y < 100 → y ≤ x) :
  x = 14 := 
sorry

end largest_whole_number_less_than_100_l29_29625


namespace acute_triangle_l29_29694

-- Given the lengths of three line segments
def length1 : ℝ := 5
def length2 : ℝ := 6
def length3 : ℝ := 7

-- Conditions (C): The lengths of the three line segments
def triangle_inequality : Prop :=
  length1 + length2 > length3 ∧
  length1 + length3 > length2 ∧
  length2 + length3 > length1

-- Question (Q) and Answer (A): They form an acute triangle
theorem acute_triangle (h : triangle_inequality) : (length1^2 + length2^2 - length3^2 > 0) :=
by
  sorry

end acute_triangle_l29_29694


namespace basis_group1_basis_group2_basis_group3_basis_l29_29984

def vector (α : Type*) := α × α

def is_collinear (v1 v2: vector ℝ) : Prop :=
  v1.1 * v2.2 - v2.1 * v1.2 = 0

def group1_v1 : vector ℝ := (-1, 2)
def group1_v2 : vector ℝ := (5, 7)

def group2_v1 : vector ℝ := (3, 5)
def group2_v2 : vector ℝ := (6, 10)

def group3_v1 : vector ℝ := (2, -3)
def group3_v2 : vector ℝ := (0.5, 0.75)

theorem basis_group1 : ¬ is_collinear group1_v1 group1_v2 :=
by sorry

theorem basis_group2 : is_collinear group2_v1 group2_v2 :=
by sorry

theorem basis_group3 : ¬ is_collinear group3_v1 group3_v2 :=
by sorry

theorem basis : (¬ is_collinear group1_v1 group1_v2) ∧ (is_collinear group2_v1 group2_v2) ∧ (¬ is_collinear group3_v1 group3_v2) :=
by sorry

end basis_group1_basis_group2_basis_group3_basis_l29_29984


namespace find_largest_number_among_three_l29_29962

noncomputable def A (B : ℝ) := 2 * B - 43
noncomputable def C (A : ℝ) := 0.5 * A + 5

-- The main statement to be proven
theorem find_largest_number_among_three : 
  ∃ (A B C : ℝ), 
  A + B + C = 50 ∧ 
  A = 2 * B - 43 ∧ 
  C = 0.5 * A + 5 ∧ 
  max A (max B C) = 27.375 :=
by
  sorry

end find_largest_number_among_three_l29_29962


namespace total_divisions_is_48_l29_29480

-- Definitions based on the conditions
def initial_cells := 1
def final_cells := 1993
def cells_added_division_42 := 41
def cells_added_division_44 := 43

-- The main statement we want to prove
theorem total_divisions_is_48 (a b : ℕ) 
  (h1 : cells_added_division_42 = 41)
  (h2 : cells_added_division_44 = 43)
  (h3 : cells_added_division_42 * a + cells_added_division_44 * b = final_cells - initial_cells) :
  a + b = 48 := 
sorry

end total_divisions_is_48_l29_29480


namespace relationship_between_x_x_squared_and_x_cubed_l29_29596

theorem relationship_between_x_x_squared_and_x_cubed (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : x < x^3 ∧ x^3 < x^2 :=
by
  sorry

end relationship_between_x_x_squared_and_x_cubed_l29_29596


namespace find_positive_x_l29_29545

theorem find_positive_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z) (h3 : x * z = 40 - 5 * x - 3 * z) :
  x = 3 :=
by sorry

end find_positive_x_l29_29545


namespace power_identity_l29_29097

theorem power_identity (x : ℝ) : (x ^ 10 = 25 ^ 5) → x = 5 := by
  sorry

end power_identity_l29_29097


namespace simplify_expression_l29_29243

variable (x : ℝ) (h : x ≠ 0)

theorem simplify_expression : (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) :=
by
  sorry

end simplify_expression_l29_29243


namespace tuning_day_method_pi_l29_29866

variable (x : ℝ)

-- Initial bounds and approximations
def initial_bounds (π : ℝ) := 31 / 10 < π ∧ π < 49 / 15

-- Definition of the "Tuning Day Method"
def tuning_day_method (a b c d : ℕ) (a' b' : ℝ) := a' = (b + d) / (a + c)

theorem tuning_day_method_pi :
  ∀ π : ℝ, initial_bounds π →
  (31 / 10 < π ∧ π < 16 / 5) ∧ 
  (47 / 15 < π ∧ π < 63 / 20) ∧
  (47 / 15 < π ∧ π < 22 / 7) →
  22 / 7 = 22 / 7 :=
by
  sorry

end tuning_day_method_pi_l29_29866


namespace green_peaches_per_basket_l29_29598

/-- Define the conditions given in the problem. -/
def n_baskets : ℕ := 7
def n_red_each : ℕ := 10
def n_green_total : ℕ := 14

/-- Prove that there are 2 green peaches in each basket. -/
theorem green_peaches_per_basket : n_green_total / n_baskets = 2 := by
  sorry

end green_peaches_per_basket_l29_29598


namespace intersection_complement_eq_l29_29637

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def P : Finset ℕ := {1, 2, 3, 4}
def Q : Finset ℕ := {3, 4, 5}
def U_complement_Q : Finset ℕ := U \ Q

theorem intersection_complement_eq : P ∩ U_complement_Q = {1, 2} :=
by {
  sorry
}

end intersection_complement_eq_l29_29637


namespace machine_worked_minutes_l29_29525

theorem machine_worked_minutes
  (shirts_today : ℕ)
  (rate : ℕ)
  (h1 : shirts_today = 8)
  (h2 : rate = 2) :
  (shirts_today / rate) = 4 :=
by
  sorry

end machine_worked_minutes_l29_29525


namespace circle_through_origin_and_point_l29_29380

theorem circle_through_origin_and_point (a r : ℝ) :
  (∃ a r : ℝ, (a^2 + (5 - 3 * a)^2 = r^2) ∧ ((a - 3)^2 + (3 * a - 6)^2 = r^2)) →
  a = 5/3 ∧ r^2 = 25/9 :=
sorry

end circle_through_origin_and_point_l29_29380


namespace greatest_drop_in_price_l29_29537

theorem greatest_drop_in_price (jan feb mar apr may jun : ℝ)
  (h_jan : jan = -0.50)
  (h_feb : feb = 2.00)
  (h_mar : mar = -2.50)
  (h_apr : apr = 3.00)
  (h_may : may = -0.50)
  (h_jun : jun = -2.00) :
  mar = -2.50 ∧ (mar ≤ jan ∧ mar ≤ may ∧ mar ≤ jun) :=
by
  sorry

end greatest_drop_in_price_l29_29537


namespace determine_no_conditionals_l29_29690

def problem_requires_conditionals (n : ℕ) : Prop :=
  n = 3 ∨ n = 4

theorem determine_no_conditionals :
  problem_requires_conditionals 1 = false ∧
  problem_requires_conditionals 2 = false ∧
  problem_requires_conditionals 3 = true ∧
  problem_requires_conditionals 4 = true :=
by sorry

end determine_no_conditionals_l29_29690


namespace k_h_neg3_l29_29093

-- Definitions of h and k
def h (x : ℝ) : ℝ := 4 * x^2 - 12

variable (k : ℝ → ℝ) -- function k with range an ℝ

-- Given k(h(3)) = 16
axiom k_h_3 : k (h 3) = 16

-- Prove that k(h(-3)) = 16
theorem k_h_neg3 : k (h (-3)) = 16 :=
sorry

end k_h_neg3_l29_29093


namespace point_A_symmetric_to_B_about_l_l29_29167

variables {A B : ℝ × ℝ} {l : ℝ → ℝ → Prop}

-- define point B
def point_B := (1, 2)

-- define the line equation x + y + 3 = 0 as a property
def line_l (x y : ℝ) := x + y + 3 = 0

-- define that A is symmetric to B about line l
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  (∀ x y : ℝ, l x y → ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = -(x + y)))
  ∧ ((A.2 - B.2) / (A.1 - B.1) * -1 = -1)

theorem point_A_symmetric_to_B_about_l :
  A = (-5, -4) →
  symmetric_about A B line_l →
  A = (-5, -4) := by
  intros _ sym
  sorry

end point_A_symmetric_to_B_about_l_l29_29167


namespace pyramid_height_l29_29948

theorem pyramid_height (lateral_edge : ℝ) (h : ℝ) (equilateral_angles : ℝ × ℝ × ℝ) (lateral_edge_length : lateral_edge = 3)
  (lateral_faces_are_equilateral : equilateral_angles = (60, 60, 60)) :
  h = 3 / 4 := by
  sorry

end pyramid_height_l29_29948


namespace orig_polygon_sides_l29_29673

theorem orig_polygon_sides (n : ℕ) (S : ℕ) :
  (n - 1 > 2) ∧ S = 1620 → (n = 10 ∨ n = 11 ∨ n = 12) :=
by
  sorry

end orig_polygon_sides_l29_29673


namespace tan_theta_eq_1_over_3_l29_29456

noncomputable def unit_circle_point (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := Real.sin θ
  (x^2 + y^2 = 1) ∧ (θ = Real.arccos ((4*x + 3*y) / 5))

theorem tan_theta_eq_1_over_3 (θ : ℝ) (h : unit_circle_point θ) : Real.tan θ = 1 / 3 := 
by
  sorry

end tan_theta_eq_1_over_3_l29_29456


namespace solve_for_x_l29_29413

-- Define the conditions as mathematical statements in Lean
def conditions (x y : ℝ) : Prop :=
  (2 * x - 3 * y = 10) ∧ (y = -x)

-- State the theorem that needs to be proven
theorem solve_for_x : ∃ x : ℝ, ∃ y : ℝ, conditions x y ∧ x = 2 :=
by 
  -- Provide a sketch of the proof to show that the statement is well-formed
  sorry

end solve_for_x_l29_29413


namespace stewart_farm_sheep_count_l29_29896

theorem stewart_farm_sheep_count 
  (S H : ℕ) 
  (ratio : S * 7 = 4 * H)
  (food_per_horse : H * 230 = 12880) : 
  S = 32 := 
sorry

end stewart_farm_sheep_count_l29_29896


namespace pencils_left_l29_29212

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l29_29212


namespace smoothie_combinations_l29_29674

theorem smoothie_combinations :
  let flavors := 5
  let supplements := 8
  (flavors * Nat.choose supplements 3) = 280 :=
by
  sorry

end smoothie_combinations_l29_29674


namespace tysons_speed_in_ocean_l29_29211

theorem tysons_speed_in_ocean
  (speed_lake : ℕ) (half_races_lake : ℕ) (total_races : ℕ) (race_distance : ℕ) (total_time : ℕ)
  (speed_lake_val : speed_lake = 3)
  (half_races_lake_val : half_races_lake = 5)
  (total_races_val : total_races = 10)
  (race_distance_val : race_distance = 3)
  (total_time_val : total_time = 11) :
  ∃ (speed_ocean : ℚ), speed_ocean = 2.5 := 
by
  sorry

end tysons_speed_in_ocean_l29_29211


namespace trig_identity_cosine_powers_l29_29773

theorem trig_identity_cosine_powers :
  12 * (Real.cos (Real.pi / 8)) ^ 4 + 
  (Real.cos (3 * Real.pi / 8)) ^ 4 + 
  (Real.cos (5 * Real.pi / 8)) ^ 4 + 
  (Real.cos (7 * Real.pi / 8)) ^ 4 = 
  3 / 2 := 
  sorry

end trig_identity_cosine_powers_l29_29773


namespace solve_system_of_equations_l29_29101

variables {a1 a2 a3 a4 : ℝ}

theorem solve_system_of_equations (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ (x1 x2 x3 x4 : ℝ),
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / (a1 - a4)) ∧ (x2 = 0) ∧ (x3 = 0) ∧ (x4 = 1 / (a1 - a4)) :=
sorry

end solve_system_of_equations_l29_29101


namespace germination_relative_frequency_l29_29187

theorem germination_relative_frequency {n m : ℕ} (h₁ : n = 1000) (h₂ : m = 1000 - 90) : 
  (m : ℝ) / (n : ℝ) = 0.91 := by
  sorry

end germination_relative_frequency_l29_29187


namespace no_integer_solutions_for_trapezoid_bases_l29_29518

theorem no_integer_solutions_for_trapezoid_bases :
  ∃ (A h : ℤ) (b1_b2 : ℤ → Prop),
    A = 2800 ∧ h = 80 ∧
    (∀ m n : ℤ, b1_b2 (12 * m) ∧ b1_b2 (12 * n) → (12 * m + 12 * n = 70) → false) :=
by
  sorry

end no_integer_solutions_for_trapezoid_bases_l29_29518


namespace parallelogram_area_15_l29_29630

def point := (ℝ × ℝ)

def base_length (p1 p2 : point) : ℝ :=
  abs (p2.1 - p1.1)

def height_length (p3 p4 : point) : ℝ :=
  abs (p3.2 - p4.2)

def parallelogram_area (p1 p2 p3 p4 : point) : ℝ :=
  base_length p1 p2 * height_length p1 p3

theorem parallelogram_area_15 :
  parallelogram_area (0, 0) (3, 0) (1, 5) (4, 5) = 15 := by
  sorry

end parallelogram_area_15_l29_29630


namespace problem_statement_l29_29957

theorem problem_statement (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
(a + b = 2) ∧ ¬( (a^2 + a > 2) ∧ (b^2 + b > 2) ) := by
  sorry

end problem_statement_l29_29957


namespace Todd_time_correct_l29_29914

theorem Todd_time_correct :
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  Todd_time = 88 :=
by
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  sorry

end Todd_time_correct_l29_29914


namespace sampling_methods_match_l29_29307

inductive SamplingMethod
| simple_random
| stratified
| systematic

open SamplingMethod

def commonly_used_sampling_methods : List SamplingMethod := 
  [simple_random, stratified, systematic]

def option_C : List SamplingMethod := 
  [simple_random, stratified, systematic]

theorem sampling_methods_match : commonly_used_sampling_methods = option_C := by
  sorry

end sampling_methods_match_l29_29307


namespace triangle_sides_from_rhombus_l29_29612

variable (m p q : ℝ)

def is_triangle_side_lengths (BC AC AB : ℝ) :=
  (BC = p + q) ∧
  (AC = m * (p + q) / p) ∧
  (AB = m * (p + q) / q)

theorem triangle_sides_from_rhombus :
  ∃ BC AC AB : ℝ, is_triangle_side_lengths m p q BC AC AB :=
by
  use p + q
  use m * (p + q) / p
  use m * (p + q) / q
  sorry

end triangle_sides_from_rhombus_l29_29612


namespace joe_sold_50_cookies_l29_29236

theorem joe_sold_50_cookies :
  ∀ (x : ℝ), (1.20 = 1 + 0.20 * 1) → (60 = 1.20 * x) → x = 50 :=
by
  intros x h1 h2
  sorry

end joe_sold_50_cookies_l29_29236


namespace check_correct_conditional_expression_l29_29387
-- importing the necessary library for basic algebraic constructions and predicates

-- defining a predicate to denote the symbolic representation of conditional expressions validity
def valid_conditional_expression (expr: String) : Prop :=
  expr = "x <> 1" ∨ expr = "x > 1" ∨ expr = "x >= 1" ∨ expr = "x < 1" ∨ expr = "x <= 1" ∨ expr = "x = 1"

-- theorem to check for the valid conditional expression among the given options
theorem check_correct_conditional_expression :
  (valid_conditional_expression "1 < x < 2") = false ∧ 
  (valid_conditional_expression "x > < 1") = false ∧ 
  (valid_conditional_expression "x <> 1") = true ∧ 
  (valid_conditional_expression "x ≤ 1") = true :=
by sorry

end check_correct_conditional_expression_l29_29387


namespace number_of_white_cats_l29_29123

theorem number_of_white_cats (total_cats : ℕ) (percent_black : ℤ) (grey_cats : ℕ) : 
  total_cats = 16 → 
  percent_black = 25 →
  grey_cats = 10 → 
  (total_cats - (total_cats * percent_black / 100 + grey_cats)) = 2 :=
by
  intros
  sorry

end number_of_white_cats_l29_29123


namespace height_of_Joaos_salary_in_kilometers_l29_29162

def real_to_cruzados (reais: ℕ) : ℕ := reais * 2750000000

def stacks (cruzados: ℕ) : ℕ := cruzados / 100

def height_in_cm (stacks: ℕ) : ℕ := stacks * 15

noncomputable def height_in_km (height_cm: ℕ) : ℕ := height_cm / 100000

theorem height_of_Joaos_salary_in_kilometers :
  height_in_km (height_in_cm (stacks (real_to_cruzados 640))) = 264000 :=
by
  sorry

end height_of_Joaos_salary_in_kilometers_l29_29162


namespace smallest_n_l29_29039

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l29_29039


namespace simultaneous_equations_solution_l29_29628

theorem simultaneous_equations_solution (x y : ℚ) :
  3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1 ↔ 
  (x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5) :=
by
  sorry

end simultaneous_equations_solution_l29_29628


namespace arc_length_of_circle_l29_29757

theorem arc_length_of_circle (r θ : ℝ) (h_r : r = 2) (h_θ : θ = 120) : 
  (θ / 180 * r * Real.pi) = (4 / 3) * Real.pi := by
  sorry

end arc_length_of_circle_l29_29757


namespace total_population_is_700_l29_29768

-- Definitions for the problem conditions
def L : ℕ := 200
def P : ℕ := L / 2
def E : ℕ := (L + P) / 2
def Z : ℕ := E + P

-- Proof statement (with sorry)
theorem total_population_is_700 : L + P + E + Z = 700 :=
by
  sorry

end total_population_is_700_l29_29768


namespace total_legs_among_animals_l29_29691

def legs (chickens sheep grasshoppers spiders : Nat) (legs_chicken legs_sheep legs_grasshopper legs_spider : Nat) : Nat :=
  (chickens * legs_chicken) + (sheep * legs_sheep) + (grasshoppers * legs_grasshopper) + (spiders * legs_spider)

theorem total_legs_among_animals :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let legs_chicken := 2
  let legs_sheep := 4
  let legs_grasshopper := 6
  let legs_spider := 8
  legs chickens sheep grasshoppers spiders legs_chicken legs_sheep legs_grasshopper legs_spider = 118 :=
by
  sorry

end total_legs_among_animals_l29_29691


namespace geometric_sequence_fourth_term_l29_29742

theorem geometric_sequence_fourth_term (a : ℝ) (r : ℝ) (h : a = 512) (h1 : a * r^5 = 125) :
  a * r^3 = 1536 :=
by
  sorry

end geometric_sequence_fourth_term_l29_29742


namespace quadratic_smallest_root_a_quadratic_smallest_root_b_l29_29298

-- For Part (a)
theorem quadratic_smallest_root_a (a : ℝ) 
  (h : a^2 - 9 * a - 10 = 0 ∧ ∀ x, x^2 - 9 * x - 10 = 0 → x ≥ a) : 
  a^4 - 909 * a = 910 :=
by sorry

-- For Part (b)
theorem quadratic_smallest_root_b (b : ℝ) 
  (h : b^2 - 9 * b + 10 = 0 ∧ ∀ x, x^2 - 9 * x + 10 = 0 → x ≥ b) : 
  b^4 - 549 * b = -710 :=
by sorry

end quadratic_smallest_root_a_quadratic_smallest_root_b_l29_29298


namespace a_beats_b_by_10_seconds_l29_29790

theorem a_beats_b_by_10_seconds :
  ∀ (T_A T_B D_A D_B : ℕ),
    T_A = 615 →
    D_A = 1000 →
    D_A - D_B = 16 →
    T_B = (D_A * T_A) / D_B →
    T_B - T_A = 10 :=
by
  -- Placeholder to ensure the theorem compiles
  intros T_A T_B D_A D_B h1 h2 h3 h4
  sorry

end a_beats_b_by_10_seconds_l29_29790


namespace lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l29_29538

def lassis_per_three_mangoes := 15
def smoothies_per_mango := 1
def bananas_per_smoothie := 2

-- proving the number of lassis Caroline can make with eighteen mangoes
theorem lassis_with_eighteen_mangoes :
  (18 / 3) * lassis_per_three_mangoes = 90 :=
by 
  sorry

-- proving the number of smoothies Caroline can make with eighteen mangoes and thirty-six bananas
theorem smoothies_with_eighteen_mangoes_and_thirtysix_bananas :
  min (18 / smoothies_per_mango) (36 / bananas_per_smoothie) = 18 :=
by 
  sorry

end lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l29_29538


namespace number_of_routes_l29_29844

structure RailwayStation :=
  (A B C D E F G H I J K L M : ℕ)

def initialize_station : RailwayStation :=
  ⟨1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 9, 9, 18⟩

theorem number_of_routes (station : RailwayStation) : station.M = 18 :=
  by sorry

end number_of_routes_l29_29844


namespace how_long_to_grow_more_l29_29220

def current_length : ℕ := 14
def length_to_donate : ℕ := 23
def desired_length_after_donation : ℕ := 12

theorem how_long_to_grow_more : 
  (desired_length_after_donation + length_to_donate - current_length) = 21 := 
by
  -- Leave the proof part for later
  sorry

end how_long_to_grow_more_l29_29220


namespace largest_real_number_mu_l29_29426

noncomputable def largest_mu : ℝ := 13 / 2

theorem largest_real_number_mu (
  a b c d : ℝ
) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) :
  (a^2 + b^2 + c^2 + d^2) ≥ (largest_mu * a * b + b * c + 2 * c * d) :=
sorry

end largest_real_number_mu_l29_29426


namespace inverse_proportion_range_l29_29231

theorem inverse_proportion_range (k : ℝ) (x : ℝ) :
  (∀ x : ℝ, (x < 0 -> (k - 1) / x > 0) ∧ (x > 0 -> (k - 1) / x < 0)) -> k < 1 :=
by
  sorry

end inverse_proportion_range_l29_29231


namespace fg_of_5_eq_163_l29_29945

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end fg_of_5_eq_163_l29_29945


namespace new_sum_after_decrease_l29_29336

theorem new_sum_after_decrease (a b : ℕ) (h₁ : a + b = 100) (h₂ : a' = a - 48) : a' + b = 52 := by
  sorry

end new_sum_after_decrease_l29_29336


namespace digit_theta_l29_29030

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta_l29_29030


namespace units_digit_x_pow_75_plus_6_eq_9_l29_29370

theorem units_digit_x_pow_75_plus_6_eq_9 (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9)
  (h3 : (x ^ 75 + 6) % 10 = 9) : x = 3 :=
sorry

end units_digit_x_pow_75_plus_6_eq_9_l29_29370


namespace r_exceeds_s_by_six_l29_29786

theorem r_exceeds_s_by_six (x y : ℚ) (h1 : 3 * x + 2 * y = 16) (h2 : x + 3 * y = 26 / 5) :
  x - y = 6 := by
  sorry

end r_exceeds_s_by_six_l29_29786


namespace sin_theta_value_l29_29344

theorem sin_theta_value (θ : ℝ) (h₁ : 8 * (Real.tan θ) = 3 * (Real.cos θ)) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 := 
by sorry

end sin_theta_value_l29_29344


namespace problem_k_value_l29_29988

theorem problem_k_value (k x1 x2 : ℝ) 
  (h_eq : 8 * x1^2 + 2 * k * x1 + k - 1 = 0) 
  (h_eq2 : 8 * x2^2 + 2 * k * x2 + k - 1 = 0) 
  (h_sum_sq : x1^2 + x2^2 = 1) : 
  k = -2 :=
sorry

end problem_k_value_l29_29988


namespace set_listing_method_l29_29519

theorem set_listing_method :
  {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 5} = {0, 1, 2} :=
by
  sorry

end set_listing_method_l29_29519


namespace number_of_unanswered_questions_l29_29581

theorem number_of_unanswered_questions (n p q : ℕ) (h1 : p = 8) (h2 : q = 5) (h3 : n = 20)
(h4: ∃ s, s % 13 = 0) (hy : y = 0 ∨ y = 13) : 
  ∃ k, k = 20 ∨ k = 7 := by
  sorry

end number_of_unanswered_questions_l29_29581


namespace part1_inequality_part2_inequality_l29_29683

-- Problem Part 1
def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)

theorem part1_inequality (x : ℝ) : f x ≤ 1 ↔ 0 ≤ x :=
by sorry

-- Problem Part 2
def max_f_value : ℝ := 3
def a : ℝ := sorry  -- Define in context
def b : ℝ := sorry  -- Define in context
def c : ℝ := sorry  -- Define in context

-- Prove √a + √b + √c ≤ 3 given a + b + c = 3
theorem part2_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = max_f_value) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 :=
by sorry

end part1_inequality_part2_inequality_l29_29683


namespace fire_fighting_max_saved_houses_l29_29860

noncomputable def max_houses_saved (n c : ℕ) : ℕ :=
  n^2 + c^2 - n * c - c

theorem fire_fighting_max_saved_houses (n c : ℕ) (h : c ≤ n / 2) :
    ∃ k, k = max_houses_saved n c :=
    sorry

end fire_fighting_max_saved_houses_l29_29860


namespace tax_percentage_l29_29351

theorem tax_percentage (total_pay take_home_pay: ℕ) (h1 : total_pay = 650) (h2 : take_home_pay = 585) :
  ((total_pay - take_home_pay) * 100 / total_pay) = 10 :=
by
  -- Assumptions
  have hp1 : total_pay = 650 := h1
  have hp2 : take_home_pay = 585 := h2
  -- Calculate tax paid
  let tax_paid := total_pay - take_home_pay
  -- Calculate tax percentage
  let tax_percentage := (tax_paid * 100) / total_pay
  -- Prove the tax percentage is 10%
  sorry

end tax_percentage_l29_29351


namespace bob_speed_before_construction_l29_29064

theorem bob_speed_before_construction:
  ∀ (v : ℝ),
    (1.5 * v + 2 * 45 = 180) →
    v = 60 :=
by
  intros v h
  sorry

end bob_speed_before_construction_l29_29064


namespace smallest_non_lucky_multiple_of_8_correct_l29_29647

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def smallest_non_lucky_multiple_of_8 := 16

theorem smallest_non_lucky_multiple_of_8_correct :
  smallest_non_lucky_multiple_of_8 = 16 ∧
  is_lucky smallest_non_lucky_multiple_of_8 = false :=
by
  sorry

end smallest_non_lucky_multiple_of_8_correct_l29_29647


namespace angle_bisector_length_l29_29601

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end angle_bisector_length_l29_29601


namespace mixing_solutions_l29_29007

theorem mixing_solutions (Vx : ℝ) :
  (0.10 * Vx + 0.30 * 900 = 0.25 * (Vx + 900)) ↔ Vx = 300 := by
  sorry

end mixing_solutions_l29_29007


namespace students_with_all_three_pets_correct_l29_29119

noncomputable def students_with_all_three_pets (total_students dog_owners cat_owners bird_owners dog_and_cat_owners cat_and_bird_owners dog_and_bird_owners : ℕ) : ℕ :=
  total_students - (dog_owners + cat_owners + bird_owners - dog_and_cat_owners - cat_and_bird_owners - dog_and_bird_owners)

theorem students_with_all_three_pets_correct : 
  students_with_all_three_pets 50 30 35 10 8 5 3 = 7 :=
by
  rw [students_with_all_three_pets]
  norm_num
  sorry

end students_with_all_three_pets_correct_l29_29119


namespace proof_problem_l29_29305

def problem : Prop :=
  ∃ (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004

theorem proof_problem : 
  problem → 
  ∃! (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004 :=
sorry

end proof_problem_l29_29305


namespace expansion_abs_coeff_sum_l29_29907

theorem expansion_abs_coeff_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - x)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 32 :=
by
  sorry

end expansion_abs_coeff_sum_l29_29907


namespace min_distance_equals_sqrt2_over_2_l29_29716

noncomputable def min_distance_from_point_to_line (m n : ℝ) : ℝ :=
  (|m + n + 10|) / Real.sqrt (1^2 + 1^2)

def circle_eq (m n : ℝ) : Prop :=
  (m - 1 / 2)^2 + (n - 1 / 2)^2 = 1 / 2

theorem min_distance_equals_sqrt2_over_2 (m n : ℝ) (h1 : circle_eq m n) :
  min_distance_from_point_to_line m n = 1 / (Real.sqrt 2) :=
sorry

end min_distance_equals_sqrt2_over_2_l29_29716


namespace john_friends_count_l29_29273

-- Define the initial conditions
def initial_amount : ℚ := 7.10
def cost_of_sweets : ℚ := 1.05
def amount_per_friend : ℚ := 1.00
def remaining_amount : ℚ := 4.05

-- Define the intermediate values
def after_sweets : ℚ := initial_amount - cost_of_sweets
def given_away : ℚ := after_sweets - remaining_amount

-- Define the final proof statement
theorem john_friends_count : given_away / amount_per_friend = 2 :=
by
  sorry

end john_friends_count_l29_29273


namespace woman_working_days_l29_29288

-- Define the conditions
def man_work_rate := 1 / 6
def boy_work_rate := 1 / 18
def combined_work_rate := 1 / 4

-- Question statement in Lean 4
theorem woman_working_days :
  ∃ W : ℚ, (man_work_rate + W + boy_work_rate = combined_work_rate) ∧ (1 / W = 1296) :=
sorry

end woman_working_days_l29_29288


namespace valid_root_l29_29992

theorem valid_root:
  ∃ x : ℚ, 
    (3 * x^2 + 5) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 ∧ x = 2 / 3 := 
by
  sorry

end valid_root_l29_29992


namespace initial_calculated_average_was_23_l29_29506

theorem initial_calculated_average_was_23 (S : ℕ) (incorrect_sum : ℕ) (n : ℕ)
  (correct_sum : ℕ) (correct_average : ℕ) (wrong_read : ℕ) (correct_read : ℕ) :
  (n = 10) →
  (wrong_read = 26) →
  (correct_read = 36) →
  (correct_average = 24) →
  (correct_sum = n * correct_average) →
  (incorrect_sum = correct_sum - correct_read + wrong_read) →
  S = incorrect_sum →
  S / n = 23 :=
by
  intros
  sorry

end initial_calculated_average_was_23_l29_29506


namespace simplify_expression_l29_29879

theorem simplify_expression (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (hx2 : x ≠ 2) (hy3 : y ≠ 3) (hz5 : z ≠ 5) :
  ( ( (x - 2) / (3 - z) * ( (y - 3) / (5 - x) ) * ( (z - 5) / (2 - y) ) ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l29_29879


namespace roots_quadratic_sum_squares_l29_29330

theorem roots_quadratic_sum_squares :
  (∃ a b : ℝ, (∀ x : ℝ, x^2 - 4 * x + 4 = 0 → (x = a ∨ x = b)) ∧ a^2 + b^2 = 8) :=
by
  sorry

end roots_quadratic_sum_squares_l29_29330


namespace color_set_no_arith_prog_same_color_l29_29824

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1987}

def colors : Fin 4 := sorry  -- Color indexing set (0, 1, 2, 3)

def valid_coloring (c : ℕ → Fin 4) : Prop :=
  ∀ (a d : ℕ) (h₁ : a ∈ M) (h₂ : d ≠ 0) (h₃ : ∀ k, a + k * d ∈ M ∧ k < 10), 
  ¬ ∀ k, c (a + k * d) = c a

theorem color_set_no_arith_prog_same_color :
  ∃ (c : ℕ → Fin 4), valid_coloring c :=
sorry

end color_set_no_arith_prog_same_color_l29_29824


namespace right_angled_triangle_k_values_l29_29476

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def BC (k : ℝ) : ℝ × ℝ := (1, k - 1)

theorem right_angled_triangle_k_values (k : ℝ) :
  (dot_product AB (AC k) = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔ (k = -6 ∨ k = -1) :=
sorry

end right_angled_triangle_k_values_l29_29476


namespace solve_inequality_system_l29_29936

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l29_29936


namespace simon_practice_hours_l29_29465

theorem simon_practice_hours (x : ℕ) (h : (12 + 16 + 14 + x) / 4 ≥ 15) : x = 18 := 
by {
  -- placeholder for the proof
  sorry
}

end simon_practice_hours_l29_29465


namespace negative_implies_neg_reciprocal_positive_l29_29971

theorem negative_implies_neg_reciprocal_positive {x : ℝ} (h : x < 0) : -x⁻¹ > 0 :=
sorry

end negative_implies_neg_reciprocal_positive_l29_29971


namespace teresa_age_at_michiko_birth_l29_29015

noncomputable def Teresa_age_now : ℕ := 59
noncomputable def Morio_age_now : ℕ := 71
noncomputable def Morio_age_at_Michiko_birth : ℕ := 38

theorem teresa_age_at_michiko_birth :
  (Teresa_age_now - (Morio_age_now - Morio_age_at_Michiko_birth)) = 26 := 
by
  sorry

end teresa_age_at_michiko_birth_l29_29015


namespace calculate_value_is_neg_seventeen_l29_29432

theorem calculate_value_is_neg_seventeen : -3^2 + (-2)^3 = -17 :=
by
  sorry

end calculate_value_is_neg_seventeen_l29_29432


namespace single_shot_percentage_decrease_l29_29483

theorem single_shot_percentage_decrease
  (initial_salary : ℝ)
  (final_salary : ℝ := initial_salary * 0.95 * 0.90 * 0.85) :
  ((1 - final_salary / initial_salary) * 100) = 27.325 := by
  sorry

end single_shot_percentage_decrease_l29_29483


namespace limit_leq_l29_29731

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem limit_leq {a_n b_n : ℕ → α} {a b : α}
  (ha : Filter.Tendsto a_n Filter.atTop (nhds a))
  (hb : Filter.Tendsto b_n Filter.atTop (nhds b))
  (h_leq : ∀ n, a_n n ≤ b_n n)
  : a ≤ b :=
by
  -- Proof will be constructed here
  sorry

end limit_leq_l29_29731


namespace contrapositive_l29_29850

variable {α : Type} (M : α → Prop) (a b : α)

theorem contrapositive (h : (M a → ¬ M b)) : (M b → ¬ M a) := 
by
  sorry

end contrapositive_l29_29850


namespace problem1_problem2_l29_29864

-- For Problem (1)
theorem problem1 (x : ℝ) : 2 * x - 3 > x + 1 → x > 4 := 
by sorry

-- For Problem (2)
theorem problem2 (a b : ℝ) (h : a^2 + 3 * a * b = 5) : (a + b) * (a + 2 * b) - 2 * b^2 = 5 := 
by sorry

end problem1_problem2_l29_29864


namespace find_number_l29_29589

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end find_number_l29_29589


namespace triangle_angle_solution_exists_l29_29542

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (A = 120 ∨ B = 120 ∨ C = 120) ∧
  (
    ((A = 40 ∧ B = 20) ∨ (A = 20 ∧ B = 40)) ∨
    ((A = 45 ∧ B = 15) ∨ (A = 15 ∧ B = 45))
  )
  
theorem triangle_angle_solution_exists :
  ∃ A B C : ℝ, possible_angles A B C :=
sorry

end triangle_angle_solution_exists_l29_29542


namespace age_of_oldest_child_l29_29812

theorem age_of_oldest_child
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 8)
  (h3 : c = 10)
  (h4 : (a + b + c + d) / 4 = 9) :
  d = 12 :=
sorry

end age_of_oldest_child_l29_29812


namespace split_coins_l29_29278

theorem split_coins (p n d q : ℕ) (hp : p % 5 = 0) 
  (h_total : p + 5 * n + 10 * d + 25 * q = 10000) :
  ∃ (p1 n1 d1 q1 p2 n2 d2 q2 : ℕ),
    (p1 + 5 * n1 + 10 * d1 + 25 * q1 = 5000) ∧
    (p2 + 5 * n2 + 10 * d2 + 25 * q2 = 5000) ∧
    (p = p1 + p2) ∧ (n = n1 + n2) ∧ (d = d1 + d2) ∧ (q = q1 + q2) :=
sorry

end split_coins_l29_29278


namespace tangent_lines_ln_l29_29262

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l29_29262


namespace part1_tangent_line_at_x2_part2_inequality_l29_29809

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

theorem part1_tangent_line_at_x2 (a : ℝ) (h_a : a = 2) :
  ∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = Real.exp 2 - 2 ∧ b = -(2 * Real.exp 2 - 7) := by
  sorry

theorem part2_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x a ≥ (7 / 4) * x^2) → a ≤ Real.exp 2 - 7 := by
  sorry

end part1_tangent_line_at_x2_part2_inequality_l29_29809


namespace red_paint_four_times_blue_paint_total_painted_faces_is_1625_l29_29592

/-- Given a structure of twenty-five layers of cubes -/
def structure_layers := 25

/-- The number of painted faces from each vertical view -/
def vertical_faces_per_view : ℕ :=
  (structure_layers * (structure_layers + 1)) / 2

/-- The total number of red-painted faces (4 vertical views) -/
def total_red_faces : ℕ :=
  4 * vertical_faces_per_view

/-- The total number of blue-painted faces (1 top view) -/
def total_blue_faces : ℕ :=
  vertical_faces_per_view

theorem red_paint_four_times_blue_paint :
  total_red_faces = 4 * total_blue_faces :=
by sorry

theorem total_painted_faces_is_1625 :
  (4 * vertical_faces_per_view + vertical_faces_per_view) = 1625 :=
by sorry

end red_paint_four_times_blue_paint_total_painted_faces_is_1625_l29_29592


namespace sum_formula_l29_29541

open Nat

/-- The sequence a_n defined as (-1)^n * (2 * n - 1) -/
def a_n (n : ℕ) : ℤ :=
  (-1) ^ n * (2 * n - 1)

/-- The partial sum S_n of the first n terms of the sequence a_n -/
def S_n : ℕ → ℤ
| 0     => 0
| (n+1) => S_n n + a_n (n + 1)

/-- The main theorem: For all n in natural numbers, S_n = (-1)^n * n -/
theorem sum_formula (n : ℕ) : S_n n = (-1) ^ n * n := by
  sorry

end sum_formula_l29_29541


namespace music_player_and_concert_tickets_l29_29156

theorem music_player_and_concert_tickets (n : ℕ) (h1 : 35 % 5 = 0) (h2 : 35 % n = 0) (h3 : ∀ m : ℕ, m < 35 → (m % 5 ≠ 0 ∨ m % n ≠ 0)) : n = 7 :=
sorry

end music_player_and_concert_tickets_l29_29156


namespace max_value_expression_l29_29135

theorem max_value_expression (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 :=
sorry

end max_value_expression_l29_29135


namespace original_classes_l29_29261

theorem original_classes (x : ℕ) (h1 : 280 % x = 0) (h2 : 585 % (x + 6) = 0) : x = 7 :=
sorry

end original_classes_l29_29261


namespace legally_drive_after_hours_l29_29771

theorem legally_drive_after_hours (n : ℕ) :
  (∀ t ≥ n, 0.8 * (0.5 : ℝ) ^ t ≤ 0.2) ↔ n = 2 :=
by
  sorry

end legally_drive_after_hours_l29_29771


namespace polygon_diagonals_regions_l29_29110

theorem polygon_diagonals_regions (n : ℕ) (hn : n ≥ 3) :
  let D := n * (n - 3) / 2
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24
  let R := D + P + 1
  R = n * (n - 1) * (n - 2) * (n - 3) / 24 + n * (n - 3) / 2 + 1 :=
by
  sorry

end polygon_diagonals_regions_l29_29110


namespace square_area_4900_l29_29186

/-- If one side of a square is increased by 3.5 times and the other side is decreased by 30 cm, resulting in a rectangle that has twice the area of the square, then the area of the square is 4900 square centimeters. -/
theorem square_area_4900 (x : ℝ) (h1 : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 :=
sorry

end square_area_4900_l29_29186


namespace juniors_to_freshmen_ratio_l29_29029

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end juniors_to_freshmen_ratio_l29_29029


namespace compute_expression_l29_29732

theorem compute_expression : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end compute_expression_l29_29732


namespace responses_needed_750_l29_29874

section Responses
  variable (q_min : ℕ) (response_rate : ℝ)

  def responses_needed : ℝ := response_rate * q_min

  theorem responses_needed_750 (h1 : q_min = 1250) (h2 : response_rate = 0.60) : responses_needed q_min response_rate = 750 :=
  by
    simp [responses_needed, h1, h2]
    sorry
end Responses

end responses_needed_750_l29_29874


namespace equivalent_operation_l29_29752

theorem equivalent_operation (x : ℚ) : (x * (2 / 5)) / (4 / 7) = x * (7 / 10) :=
by
  sorry

end equivalent_operation_l29_29752


namespace triangle_circumradius_l29_29477

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) : 
  ∃ r : ℝ, r = 5 :=
by
  sorry

end triangle_circumradius_l29_29477


namespace domain_ln_l29_29520

theorem domain_ln (x : ℝ) (h : x - 1 > 0) : x > 1 := 
sorry

end domain_ln_l29_29520


namespace tom_trip_cost_l29_29580

-- Definitions of hourly rates
def rate_6AM_to_10AM := 10
def rate_10AM_to_2PM := 12
def rate_2PM_to_6PM := 15
def rate_6PM_to_10PM := 20

-- Definitions of trip start times and durations
def first_trip_start := 8
def second_trip_start := 14
def third_trip_start := 20

-- Function to calculate the cost for each trip segment
def cost (start_hour : Nat) (duration : Nat) : Nat :=
  if start_hour >= 6 ∧ start_hour < 10 then duration * rate_6AM_to_10AM
  else if start_hour >= 10 ∧ start_hour < 14 then duration * rate_10AM_to_2PM
  else if start_hour >= 14 ∧ start_hour < 18 then duration * rate_2PM_to_6PM
  else if start_hour >= 18 ∧ start_hour < 22 then duration * rate_6PM_to_10PM
  else 0

-- Function to calculate the total trip cost
def total_cost : Nat :=
  cost first_trip_start 2 + cost (first_trip_start + 2) 2 +
  cost second_trip_start 4 +
  cost third_trip_start 4

-- Proof statement
theorem tom_trip_cost : total_cost = 184 := by
  -- The detailed steps of the proof would go here. Replaced with 'sorry' presently to indicate incomplete proof.
  sorry

end tom_trip_cost_l29_29580


namespace numbers_in_ratio_l29_29361

theorem numbers_in_ratio (a b c : ℤ) :
  (∃ x : ℤ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x) ∧ (a * a + b * b + c * c = 725) →
  (a = 10 ∧ b = 15 ∧ c = 20 ∨ a = -10 ∧ b = -15 ∧ c = -20) :=
by
  sorry

end numbers_in_ratio_l29_29361


namespace rancher_cows_l29_29965

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end rancher_cows_l29_29965


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l29_29770

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l29_29770


namespace probability_male_female_ratio_l29_29304

theorem probability_male_female_ratio :
  let total_possibilities := Nat.choose 9 5
  let specific_scenarios := Nat.choose 5 2 * Nat.choose 4 3 + Nat.choose 5 3 * Nat.choose 4 2
  let probability := specific_scenarios / (total_possibilities : ℚ)
  probability = 50 / 63 :=
by 
  sorry

end probability_male_female_ratio_l29_29304


namespace weekly_cost_l29_29862

def cost_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7
def number_of_bodyguards : ℕ := 2

theorem weekly_cost :
  (cost_per_hour * hours_per_day * number_of_bodyguards * days_per_week) = 2240 := by
  sorry

end weekly_cost_l29_29862


namespace present_population_l29_29930

-- Definitions
def initial_population : ℕ := 1200
def first_year_increase_rate : ℝ := 0.25
def second_year_increase_rate : ℝ := 0.30

-- Problem Statement
theorem present_population (initial_population : ℕ) 
    (first_year_increase_rate second_year_increase_rate : ℝ) : 
    initial_population = 1200 → 
    first_year_increase_rate = 0.25 → 
    second_year_increase_rate = 0.30 →
    ∃ current_population : ℕ, current_population = 1950 :=
by
  intros h₁ h₂ h₃
  sorry

end present_population_l29_29930


namespace find_sides_of_triangle_ABC_find_angle_A_l29_29795

variable (a b c A B C : ℝ)

-- Part (Ⅰ)
theorem find_sides_of_triangle_ABC
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hArea : 1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3) :
  a = 2 ∧ b = 2 := sorry

-- Part (Ⅱ)
theorem find_angle_A
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hTrig : Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 := sorry

end find_sides_of_triangle_ABC_find_angle_A_l29_29795


namespace other_candidate_valid_votes_l29_29833

noncomputable def validVotes (totalVotes invalidPct : ℝ) : ℝ :=
  totalVotes * (1 - invalidPct)

noncomputable def otherCandidateVotes (validVotes oneCandidatePct : ℝ) : ℝ :=
  validVotes * (1 - oneCandidatePct)

theorem other_candidate_valid_votes :
  let totalVotes := 7500
  let invalidPct := 0.20
  let oneCandidatePct := 0.55
  validVotes totalVotes invalidPct = 6000 ∧
  otherCandidateVotes (validVotes totalVotes invalidPct) oneCandidatePct = 2700 :=
by
  sorry

end other_candidate_valid_votes_l29_29833


namespace find_k_l29_29108

theorem find_k (k : ℝ) (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0)
  (h3 : r / s = 3) (h4 : r + s = 4) (h5 : r * s = k) : k = 3 :=
sorry

end find_k_l29_29108


namespace projection_of_c_onto_b_l29_29140

open Real

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := sqrt (b.1^2 + b.2^2)
  let scalar := dot_product / magnitude_b
  (scalar * b.1 / magnitude_b, scalar * b.2 / magnitude_b)

theorem projection_of_c_onto_b :
  let a := (2, 3)
  let b := (-4, 7)
  let c := (-a.1, -a.2)
  vector_projection c b = (-sqrt 65 / 5, -sqrt 65 / 5) :=
by sorry

end projection_of_c_onto_b_l29_29140


namespace num_dogs_with_spots_l29_29407

variable (D P : ℕ)

theorem num_dogs_with_spots (h1 : D / 2 = D / 2) (h2 : D / 5 = P) : (5 * P) / 2 = D / 2 := 
by
  have h3 : 5 * P = D := by
    sorry
  have h4 : (5 * P) / 2 = D / 2 := by
    rw [h3]
  exact h4

end num_dogs_with_spots_l29_29407


namespace symmetric_points_origin_l29_29657

theorem symmetric_points_origin (a b : ℝ)
  (h1 : (-2 : ℝ) = -a)
  (h2 : (b : ℝ) = -3) : a - b = 5 :=
by
  sorry

end symmetric_points_origin_l29_29657


namespace proof_correct_judgments_l29_29827

def terms_are_like (t1 t2 : Expr) : Prop := sorry -- Define like terms
def is_polynomial (p : Expr) : Prop := sorry -- Define polynomial
def is_quadratic_trinomial (p : Expr) : Prop := sorry -- Define quadratic trinomial
def constant_term (p : Expr) : Expr := sorry -- Define extraction of constant term

theorem proof_correct_judgments :
  let t1 := (2 * Real.pi * (a ^ 2) * b)
  let t2 := ((1 / 3) * (a ^ 2) * b)
  let p1 := (5 * a + 4 * b - 1)
  let p2 := (x - 2 * x * y + y)
  let p3 := ((x + y) / 4)
  let p4 := (x / 2 + 1)
  let p5 := (a / 4)
  terms_are_like t1 t2 ∧ 
  constant_term p1 = 1 = False ∧
  is_quadratic_trinomial p2 ∧
  is_polynomial p3 ∧ is_polynomial p4 ∧ is_polynomial p5
  → ("①③④" = "C") :=
by
  sorry

end proof_correct_judgments_l29_29827


namespace find_other_polynomial_l29_29699

variables {a b c d : ℤ}

theorem find_other_polynomial (h : ∀ P Q : ℤ, P - Q = c^2 * d^2 - a^2 * b^2) 
  (P : ℤ) (hP : P = a^2 * b^2 + c^2 * d^2 - 2 * a * b * c * d) : 
  (∃ Q : ℤ, Q = 2 * c^2 * d^2 - 2 * a * b * c * d) ∨ 
  (∃ Q : ℤ, Q = 2 * a^2 * b^2 - 2 * a * b * c * d) :=
by {
  sorry
}

end find_other_polynomial_l29_29699


namespace three_digit_numbers_with_properties_l29_29762

noncomputable def valid_numbers_with_properties : List Nat :=
  [179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959]

theorem three_digit_numbers_with_properties (N : ℕ) :
  N >= 100 ∧ N < 1000 ∧ 
  N ≡ 1 [MOD 2] ∧
  N ≡ 2 [MOD 3] ∧
  N ≡ 3 [MOD 4] ∧
  N ≡ 4 [MOD 5] ∧
  N ≡ 5 [MOD 6] ↔ N ∈ valid_numbers_with_properties :=
by
  sorry

end three_digit_numbers_with_properties_l29_29762


namespace no_solution_iff_m_range_l29_29572

theorem no_solution_iff_m_range (m : ℝ) : 
  ¬ ∃ x : ℝ, |x-1| + |x-m| < 2*m ↔ (0 < m ∧ m < 1/3) := sorry

end no_solution_iff_m_range_l29_29572


namespace composite_function_l29_29769

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

theorem composite_function : ∀ (x : ℝ), f (g x) = 2 * x + 1 :=
by
  intro x
  sorry

end composite_function_l29_29769


namespace find_x_squared_plus_y_squared_l29_29475

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 :=
sorry

end find_x_squared_plus_y_squared_l29_29475


namespace composite_sum_pow_l29_29103

theorem composite_sum_pow (a b c d : ℕ) (h_pos : a > b ∧ b > c ∧ c > d)
    (h_div : (a + b - c + d) ∣ (a * c + b * d)) (m : ℕ) (h_m_pos : 0 < m) 
    (n : ℕ) (h_n_odd : n % 2 = 1) : ∃ k : ℕ, k > 1 ∧ k ∣ (a ^ n * b ^ m + c ^ m * d ^ n) :=
by
  sorry

end composite_sum_pow_l29_29103


namespace prism_volume_is_25_l29_29753

noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

noncomputable def prism_volume (base_area height : ℝ) : ℝ := base_area * height

theorem prism_volume_is_25 :
  let a := Real.sqrt 5
  let base_area := triangle_area a a
  let volume := prism_volume base_area 10
  volume = 25 :=
by
  intros
  sorry

end prism_volume_is_25_l29_29753


namespace factorization_analysis_l29_29820

variable (a b c : ℝ)

theorem factorization_analysis : a^2 - 2 * a * b + b^2 - c^2 = (a - b + c) * (a - b - c) := 
sorry

end factorization_analysis_l29_29820


namespace minimum_expression_value_l29_29089

noncomputable def expr (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  (2 * (Real.sin x₁)^2 + 1 / (Real.sin x₁)^2) *
  (2 * (Real.sin x₂)^2 + 1 / (Real.sin x₂)^2) *
  (2 * (Real.sin x₃)^2 + 1 / (Real.sin x₃)^2) *
  (2 * (Real.sin x₄)^2 + 1 / (Real.sin x₄)^2)

theorem minimum_expression_value :
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  x₁ + x₂ + x₃ + x₄ = Real.pi →
  expr x₁ x₂ x₃ x₄ ≥ 81 := sorry

end minimum_expression_value_l29_29089


namespace polygon_sides_l29_29260

theorem polygon_sides (n : ℕ) (h : n * (n - 3) / 2 = 20) : n = 8 :=
by
  sorry

end polygon_sides_l29_29260


namespace ellipse_foci_coordinates_l29_29012

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (y^2 / 3 + x^2 / 2 = 1) → (x, y) = (0, -1) ∨ (x, y) = (0, 1) :=
by
  sorry

end ellipse_foci_coordinates_l29_29012


namespace monomial_2023rd_l29_29198

theorem monomial_2023rd : ∀ (x : ℝ), (2 * 2023 + 1) / 2023 * x ^ 2023 = (4047 / 2023) * x ^ 2023 :=
by
  intro x
  sorry

end monomial_2023rd_l29_29198


namespace area_of_red_flowers_is_54_l29_29623

noncomputable def total_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def red_yellow_area (total : ℝ) : ℝ :=
  total / 2

noncomputable def red_area (red_yellow : ℝ) : ℝ :=
  red_yellow / 2

theorem area_of_red_flowers_is_54 :
  total_area 18 12 / 2 / 2 = 54 := 
  by
    sorry

end area_of_red_flowers_is_54_l29_29623


namespace union_A_B_eq_C_l29_29035

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end union_A_B_eq_C_l29_29035


namespace joy_tape_deficit_l29_29173

noncomputable def tape_needed_field (width length : ℕ) : ℕ :=
2 * (length + width)

noncomputable def tape_needed_trees (num_trees circumference : ℕ) : ℕ :=
num_trees * circumference

def tape_total_needed (tape_field tape_trees : ℕ) : ℕ :=
tape_field + tape_trees

theorem joy_tape_deficit (tape_has : ℕ) (tape_field tape_trees: ℕ) : ℤ :=
tape_has - (tape_field + tape_trees)

example : joy_tape_deficit 180 (tape_needed_field 35 80) (tape_needed_trees 3 5) = -65 := by
sorry

end joy_tape_deficit_l29_29173


namespace apprentice_daily_output_l29_29587

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end apprentice_daily_output_l29_29587


namespace cara_age_is_40_l29_29393

-- Defining the conditions
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Proving the question
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end cara_age_is_40_l29_29393


namespace shaded_area_is_110_l29_29049

-- Definitions based on conditions
def equilateral_triangle_area : ℕ := 10
def num_triangles_small : ℕ := 1
def num_triangles_medium : ℕ := 3
def num_triangles_large : ℕ := 7

-- Total area calculation
def total_area : ℕ := (num_triangles_small + num_triangles_medium + num_triangles_large) * equilateral_triangle_area

-- The theorem statement
theorem shaded_area_is_110 : total_area = 110 := 
by 
  sorry

end shaded_area_is_110_l29_29049


namespace shelby_stars_yesterday_l29_29744

-- Define the number of stars earned yesterday
def stars_yesterday : ℕ := sorry

-- Condition 1: In all, Shelby earned 7 gold stars
def stars_total : ℕ := 7

-- Condition 2: Today, she earned 3 more gold stars
def stars_today : ℕ := 3

-- The proof statement that combines the conditions 
-- and question to the correct answer
theorem shelby_stars_yesterday (y : ℕ) (h1 : y + stars_today = stars_total) : y = 4 := 
by
  -- Placeholder for the actual proof
  sorry

end shelby_stars_yesterday_l29_29744


namespace tan_difference_identity_l29_29788

theorem tan_difference_identity (a b : ℝ) (h1 : Real.tan a = 2) (h2 : Real.tan b = 3 / 4) :
  Real.tan (a - b) = 1 / 2 :=
sorry

end tan_difference_identity_l29_29788


namespace no_unique_p_l29_29284

-- Define the probabilities P_1 and P_2 given p
def P1 (p : ℝ) : ℝ := 3 * p^2 - 2 * p^3
def P2 (p : ℝ) : ℝ := 3 * p^2 - 3 * p^3

-- Define the expected value E(xi)
def E_xi (p : ℝ) : ℝ := P1 p + P2 p

-- Prove that there does not exist a unique p in (0, 1) such that E(xi) = 1.5
theorem no_unique_p (p : ℝ) (h : 0 < p ∧ p < 1) : E_xi p ≠ 1.5 := by
  sorry

end no_unique_p_l29_29284


namespace length_of_platform_l29_29700

theorem length_of_platform {train_length : ℕ} {time_to_cross_pole : ℕ} {time_to_cross_platform : ℕ} 
  (h1 : train_length = 300) 
  (h2 : time_to_cross_pole = 18) 
  (h3 : time_to_cross_platform = 45) : 
  ∃ platform_length : ℕ, platform_length = 450 :=
by
  sorry

end length_of_platform_l29_29700


namespace parking_lot_total_spaces_l29_29076

theorem parking_lot_total_spaces (ratio_fs_cc : ℕ) (ratio_cc_fs : ℕ) (fs_spaces : ℕ) (total_spaces : ℕ) 
  (h1 : ratio_fs_cc = 11) (h2 : ratio_cc_fs = 4) (h3 : fs_spaces = 330) :
  total_spaces = 450 :=
by
  sorry

end parking_lot_total_spaces_l29_29076


namespace polygon_encloses_250_square_units_l29_29238

def vertices : List (ℕ × ℕ) := [(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to calculate the area of the given polygon
  sorry

theorem polygon_encloses_250_square_units : polygon_area vertices = 250 := by
  -- Proof that the area of the polygon is 250 square units
  sorry

end polygon_encloses_250_square_units_l29_29238


namespace sin_cos_identity_l29_29749

theorem sin_cos_identity : (Real.sin (65 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) 
  - Real.cos (65 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_cos_identity_l29_29749


namespace circle_cut_by_parabolas_l29_29246

theorem circle_cut_by_parabolas (n : ℕ) (h : n = 10) : 
  2 * n ^ 2 + 1 = 201 :=
by
  sorry

end circle_cut_by_parabolas_l29_29246


namespace optimal_optimism_coefficient_l29_29313

theorem optimal_optimism_coefficient (a b : ℝ) (x : ℝ) (h_b_gt_a : b > a) (h_x : 0 < x ∧ x < 1) 
  (h_c : ∀ (c : ℝ), c = a + x * (b - a) → (c - a) * (c - a) = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end optimal_optimism_coefficient_l29_29313


namespace todd_ate_cupcakes_l29_29379

theorem todd_ate_cupcakes :
  let C := 38   -- Total cupcakes baked by Sarah
  let P := 3    -- Number of packages made
  let c := 8    -- Number of cupcakes per package
  let L := P * c  -- Total cupcakes left after packaging
  C - L = 14 :=  -- Cupcakes Todd ate is 14
by
  sorry

end todd_ate_cupcakes_l29_29379


namespace mass_of_cork_l29_29909

theorem mass_of_cork (ρ_p ρ_w ρ_s : ℝ) (m_p x : ℝ) :
  ρ_p = 2.15 * 10^4 → 
  ρ_w = 2.4 * 10^2 →
  ρ_s = 4.8 * 10^2 →
  m_p = 86.94 →
  x = 2.4 * 10^2 * (m_p / ρ_p) →
  x = 85 :=
by
  intros
  sorry

end mass_of_cork_l29_29909


namespace beth_crayon_packs_l29_29392

theorem beth_crayon_packs (P : ℕ) (h1 : 10 * P + 6 = 46) : P = 4 :=
by
  sorry

end beth_crayon_packs_l29_29392


namespace f_is_neither_odd_nor_even_l29_29684

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^2 + 6 * x

-- Defining the concept of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Defining the concept of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

-- The goal is to prove that f is neither odd nor even
theorem f_is_neither_odd_nor_even : ¬ is_odd f ∧ ¬ is_even f :=
by
  sorry

end f_is_neither_odd_nor_even_l29_29684


namespace range_of_a_l29_29534

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ Real.exp 1) ↔ ∀ x : ℝ, 0 < x → a * Real.log (a * x) ≤ Real.exp x := 
by 
  sorry

end range_of_a_l29_29534


namespace alyssas_weekly_allowance_l29_29271

-- Define the constants and parameters
def spent_on_movies (A : ℝ) := 0.5 * A
def spent_on_snacks (A : ℝ) := 0.2 * A
def saved_for_future (A : ℝ) := 0.25 * A

-- Define the remaining allowance after expenses
def remaining_allowance_after_expenses (A : ℝ) := A - spent_on_movies A - spent_on_snacks A - saved_for_future A

-- Define Alyssa's allowance given the conditions
theorem alyssas_weekly_allowance : ∀ (A : ℝ), 
  remaining_allowance_after_expenses A = 12 → 
  A = 240 :=
by
  -- Proof omitted
  sorry

end alyssas_weekly_allowance_l29_29271


namespace find_added_number_l29_29573

theorem find_added_number (R D Q X : ℕ) (hR : R = 5) (hD : D = 3 * Q) (hDiv : 113 = D * Q + R) (hD_def : D = 3 * R + X) : 
  X = 3 :=
by
  -- Provide the conditions as assumptions
  sorry

end find_added_number_l29_29573


namespace conditional_probability_l29_29322

theorem conditional_probability :
  let P_B : ℝ := 0.15
  let P_A : ℝ := 0.05
  let P_A_and_B : ℝ := 0.03
  let P_B_given_A := P_A_and_B / P_A
  P_B_given_A = 0.6 :=
by
  sorry

end conditional_probability_l29_29322


namespace num_lighting_methods_l29_29382

-- Definitions of the problem's conditions
def total_lights : ℕ := 15
def lights_off : ℕ := 6
def lights_on : ℕ := total_lights - lights_off
def available_spaces : ℕ := lights_on - 1

-- Statement of the mathematically equivalent proof problem
theorem num_lighting_methods : Nat.choose available_spaces lights_off = 28 := by
  sorry

end num_lighting_methods_l29_29382


namespace ensure_nonempty_intersection_l29_29447

def M (x : ℝ) : Prop := x ≤ 1
def N (x : ℝ) (p : ℝ) : Prop := x > p

theorem ensure_nonempty_intersection (p : ℝ) : (∃ x : ℝ, M x ∧ N x p) ↔ p < 1 :=
by
  sorry

end ensure_nonempty_intersection_l29_29447


namespace solve_prime_equation_l29_29999

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end solve_prime_equation_l29_29999


namespace derivative_at_zero_l29_29155

def f (x : ℝ) : ℝ := (x + 1)^4

theorem derivative_at_zero : deriv f 0 = 4 :=
by
  sorry

end derivative_at_zero_l29_29155


namespace second_offset_length_l29_29953

theorem second_offset_length (d h1 area : ℝ) (h_diagonal : d = 28) (h_offset1 : h1 = 8) (h_area : area = 140) :
  ∃ x : ℝ, area = (1/2) * d * (h1 + x) ∧ x = 2 :=
by
  sorry

end second_offset_length_l29_29953


namespace quadratic_coefficient_c_l29_29513

theorem quadratic_coefficient_c (b c: ℝ) 
  (h_sum: 12 = b) (h_prod: 20 = c) : 
  c = 20 := 
by sorry

end quadratic_coefficient_c_l29_29513


namespace smallest_and_largest_values_l29_29912

theorem smallest_and_largest_values (x : ℕ) (h : x < 100) :
  (x ≡ 2 [MOD 3]) ∧ (x ≡ 2 [MOD 4]) ∧ (x ≡ 2 [MOD 5]) ↔ (x = 2 ∨ x = 62) :=
by
  sorry

end smallest_and_largest_values_l29_29912


namespace result_l29_29177

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l29_29177


namespace markup_percentage_l29_29863

-- Definitions coming from conditions
variables (C : ℝ) (M : ℝ) (S : ℝ)
-- Markup formula
def markup_formula : Prop := M = 0.10 * C
-- Selling price formula
def selling_price_formula : Prop := S = C + M

-- Given the conditions, we need to prove that the markup is 9.09% of the selling price
theorem markup_percentage (h1 : markup_formula C M) (h2 : selling_price_formula C M S) :
  (M / S) * 100 = 9.09 :=
sorry

end markup_percentage_l29_29863


namespace amount_after_two_years_l29_29374

/-- Defining given conditions. -/
def initial_value : ℤ := 65000
def first_year_increase : ℚ := 12 / 100
def second_year_increase : ℚ := 8 / 100

/-- The main statement that needs to be proved. -/
theorem amount_after_two_years : 
  let first_year_amount := initial_value + (initial_value * first_year_increase)
  let second_year_amount := first_year_amount + (first_year_amount * second_year_increase)
  second_year_amount = 78624 := 
by 
  sorry

end amount_after_two_years_l29_29374


namespace sqrt_12_bounds_l29_29194

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l29_29194


namespace sequence_fraction_l29_29634

-- Definitions for arithmetic and geometric sequences
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def isGeometricSeq (a b c : ℝ) :=
  b^2 = a * c

-- Given conditions
variables {a : ℕ → ℝ} {d : ℝ}

-- a is an arithmetic sequence with common difference d ≠ 0
axiom h1 : isArithmeticSeq a d
axiom h2 : d ≠ 0

-- a_2, a_3, a_9 form a geometric sequence
axiom h3 : isGeometricSeq (a 2) (a 3) (a 9)

-- Goal: prove the value of the given expression
theorem sequence_fraction {a : ℕ → ℝ} {d : ℝ} (h1 : isArithmeticSeq a d) (h2 : d ≠ 0) (h3 : isGeometricSeq (a 2) (a 3) (a 9)) :
  (a 2 + a 3 + a 4) / (a 4 + a 5 + a 6) = 3 / 8 :=
by
  sorry

end sequence_fraction_l29_29634


namespace six_digit_numbers_with_zero_l29_29489

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l29_29489


namespace number_of_pieces_l29_29676

def pan_length : ℕ := 24
def pan_width : ℕ := 30
def brownie_length : ℕ := 3
def brownie_width : ℕ := 4

def area (length : ℕ) (width : ℕ) : ℕ := length * width

theorem number_of_pieces :
  (area pan_length pan_width) / (area brownie_length brownie_width) = 60 := by
  sorry

end number_of_pieces_l29_29676


namespace solve_for_b_l29_29765

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_b (b : ℝ) (i_is_imag_unit : ∀ (z : ℂ), i * z = z * i):
  is_imaginary (i * (b * i + 1)) → b = 0 :=
by
  sorry

end solve_for_b_l29_29765


namespace johns_remaining_money_l29_29190

theorem johns_remaining_money (H1 : ∃ (n : ℕ), n = 5376) (H2 : 5376 = 5 * 8^3 + 3 * 8^2 + 7 * 8^1 + 6) :
  (2814 - 1350 = 1464) :=
by {
  sorry
}

end johns_remaining_money_l29_29190


namespace number_of_divisors_M_l29_29001

def M : ℕ := 2^5 * 3^4 * 5^2 * 7^3 * 11^1

theorem number_of_divisors_M : (M.factors.prod.divisors.card = 720) :=
sorry

end number_of_divisors_M_l29_29001


namespace pages_read_on_wednesday_l29_29057

theorem pages_read_on_wednesday (W : ℕ) (h : 18 + W + 23 = 60) : W = 19 :=
by {
  sorry
}

end pages_read_on_wednesday_l29_29057


namespace f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l29_29514

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (k x : ℝ) : ℝ := x^2 + k * x
noncomputable def a (x1 x2 : ℝ) : ℝ := (f x1 - f x2) / (x1 - x2)
noncomputable def b (z1 z2 k : ℝ) : ℝ := (g k z1 - g k z2) / (z1 - z2)

theorem f_is_increasing (x1 x2 : ℝ) (h : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) : a x1 x2 > 0 := by
  sorry

theorem exists_ratio_two (k : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = 2 * a x1 x2 := by
  sorry

theorem range_k_for_negative_two_ratio (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = -2 * a x1 x2) → k < -4 := by
  sorry

end f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l29_29514


namespace number_of_pairs_of_shoes_size_40_to_42_200_pairs_l29_29463

theorem number_of_pairs_of_shoes_size_40_to_42_200_pairs 
  (total_pairs_sample : ℕ)
  (freq_3rd_group : ℝ)
  (freq_1st_group : ℕ)
  (freq_2nd_group : ℕ)
  (freq_4th_group : ℕ)
  (total_pairs_200 : ℕ)
  (scaled_pairs_size_40_42 : ℕ)
: total_pairs_sample = 40 ∧ freq_3rd_group = 0.25 ∧ freq_1st_group = 6 ∧ freq_2nd_group = 7 ∧ freq_4th_group = 9 ∧ total_pairs_200 = 200 ∧ scaled_pairs_size_40_42 = 40 :=
sorry

end number_of_pairs_of_shoes_size_40_to_42_200_pairs_l29_29463


namespace eval_sin_570_l29_29430

theorem eval_sin_570:
  2 * Real.sin (570 * Real.pi / 180) = -1 := 
by sorry

end eval_sin_570_l29_29430


namespace find_h_l29_29937

theorem find_h (h : ℝ) :
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ -(x - h)^2 = -1) → (h = 2 ∨ h = 8) :=
by sorry

end find_h_l29_29937


namespace largest_common_divisor_l29_29686

theorem largest_common_divisor (d h m s : ℕ) : 
  40 ∣ (1000000 * d + 10000 * h + 100 * m + s - (86400 * d + 3600 * h + 60 * m + s)) :=
by
  sorry

end largest_common_divisor_l29_29686


namespace batches_of_muffins_l29_29663

-- Definitions of the costs and savings
def cost_blueberries_6oz : ℝ := 5
def cost_raspberries_12oz : ℝ := 3
def ounces_per_batch : ℝ := 12
def total_savings : ℝ := 22

-- The proof problem is to show the number of batches Bill plans to make
theorem batches_of_muffins : (total_savings / (2 * cost_blueberries_6oz - cost_raspberries_12oz)) = 3 := 
by 
  sorry  -- Proof goes here

end batches_of_muffins_l29_29663


namespace inscribed_square_area_l29_29391

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = (a ^ 2 + b ^ 2) ^ (1 / 2)

def square_area (s : ℝ) : ℝ := s * s

theorem inscribed_square_area
  (a b c : ℝ) (s₁ s₂ : ℝ)
  (ha : a = 16 * 2) -- Leg lengths equal to 2 * 16 cm
  (hb : b = 16 * 2)
  (hc : c = 32 * Real.sqrt 2) -- Hypotenuse of the triangle
  (hiso : isosceles_right_triangle a b c)
  (harea₁ : square_area 16 = 256) -- Given square area
  (hS : s₂ = 16 * Real.sqrt 2 - 8) -- Side length of the new square
  : square_area s₂ = 576 - 256 * Real.sqrt 2 := sorry

end inscribed_square_area_l29_29391


namespace find_smaller_number_l29_29225

def one_number_is_11_more_than_3times_another (x y : ℕ) : Prop :=
  y = 3 * x + 11

def their_sum_is_55 (x y : ℕ) : Prop :=
  x + y = 55

theorem find_smaller_number (x y : ℕ) (h1 : one_number_is_11_more_than_3times_another x y) (h2 : their_sum_is_55 x y) :
  x = 11 :=
by
  -- The proof will be inserted here
  sorry

end find_smaller_number_l29_29225


namespace amount_given_to_beggar_l29_29161

variable (X : ℕ)
variable (pennies_initial : ℕ := 42)
variable (pennies_to_farmer : ℕ := 22)
variable (pennies_after_farmer : ℕ := 20)

def amount_to_boy (X : ℕ) : ℕ :=
  (20 - X) / 2 + 3

theorem amount_given_to_beggar : 
  (X = 12) →  (pennies_initial - pennies_to_farmer - X) / 2 + 3 + 1 = pennies_initial - pennies_to_farmer - X :=
by
  intro h
  subst h
  sorry

end amount_given_to_beggar_l29_29161


namespace find_number_l29_29727

theorem find_number (x : ℝ) (h: 9999 * x = 4690910862): x = 469.1 :=
by
  sorry

end find_number_l29_29727


namespace angle_in_third_quadrant_l29_29472

-- Definitions for quadrants
def in_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360
def in_third_quadrant (β : ℝ) : Prop := 180 < β ∧ β < 270

theorem angle_in_third_quadrant (α : ℝ) (h : in_fourth_quadrant α) : in_third_quadrant (180 - α) :=
by
  -- Proof goes here
  sorry

end angle_in_third_quadrant_l29_29472


namespace number_of_children_l29_29843

-- Definition of the conditions
def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 30

-- Theorem statement
theorem number_of_children (n : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  n = total_pencils / pencils_per_child :=
by
  have h : n = 30 / 2 := sorry
  exact h

end number_of_children_l29_29843


namespace max_sin_a_l29_29014

theorem max_sin_a (a b c : ℝ) (h1 : Real.cos a = Real.tan b) 
                                  (h2 : Real.cos b = Real.tan c) 
                                  (h3 : Real.cos c = Real.tan a) : 
  Real.sin a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
by
  sorry

end max_sin_a_l29_29014


namespace area_at_stage_8_l29_29063

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end area_at_stage_8_l29_29063


namespace darcy_folded_shorts_l29_29544

-- Define the conditions
def total_shirts : Nat := 20
def total_shorts : Nat := 8
def folded_shirts : Nat := 12
def remaining_pieces : Nat := 11

-- Expected result to prove
def folded_shorts : Nat := 5

-- The statement to prove
theorem darcy_folded_shorts : total_shorts - (remaining_pieces - (total_shirts - folded_shirts)) = folded_shorts :=
by
  sorry

end darcy_folded_shorts_l29_29544


namespace slices_dinner_l29_29291

variable (lunch_slices : ℕ) (total_slices : ℕ)
variable (h1 : lunch_slices = 7) (h2 : total_slices = 12)

theorem slices_dinner : total_slices - lunch_slices = 5 :=
by sorry

end slices_dinner_l29_29291


namespace moles_of_C6H6_l29_29928

def balanced_reaction (a b c d : ℕ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ a + b + c + d = 4

theorem moles_of_C6H6 (a b c d : ℕ) (h_balanced : balanced_reaction a b c d) :
  a = 1 := 
by 
  sorry

end moles_of_C6H6_l29_29928


namespace tims_seashells_now_l29_29981

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tims_seashells_now : (initial_seashells - seashells_given_away) = 507 :=
by
  sorry

end tims_seashells_now_l29_29981


namespace sufficient_not_necessary_l29_29191

theorem sufficient_not_necessary (x : ℝ) : (x > 3) → (abs (x - 3) > 0) ∧ (¬(abs (x - 3) > 0) → (¬(x > 3))) :=
by
  sorry

end sufficient_not_necessary_l29_29191


namespace arithmetic_sequence_sum_l29_29931

-- Given {a_n} is an arithmetic sequence, and a_2 + a_3 + a_{10} + a_{11} = 40, prove a_6 + a_7 = 20
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 40) :
  a 6 + a 7 = 20 :=
sorry

end arithmetic_sequence_sum_l29_29931


namespace min_value_y_l29_29998

theorem min_value_y (x : ℝ) : ∃ x : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ z : ℝ, (y = z^2 + 16 * z + 20) → y ≥ -44 := 
sorry

end min_value_y_l29_29998


namespace variance_of_scores_l29_29934

open Real

def scores : List ℝ := [30, 26, 32, 27, 35]
noncomputable def average (s : List ℝ) : ℝ := s.sum / s.length
noncomputable def variance (s : List ℝ) : ℝ :=
  (s.map (λ x => (x - average s) ^ 2)).sum / s.length

theorem variance_of_scores :
  variance scores = 54 / 5 := 
by
  sorry

end variance_of_scores_l29_29934


namespace jack_total_plates_after_smashing_and_buying_l29_29530

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1
def new_polka_dotted_plates : ℕ := initial_checked_plates * initial_checked_plates

theorem jack_total_plates_after_smashing_and_buying : 
  initial_flower_plates - smashed_flower_plates
  + initial_checked_plates
  + initial_striped_plates - smashed_striped_plates
  + new_polka_dotted_plates = 96 := 
by {
  -- calculation proof here
  sorry
}

end jack_total_plates_after_smashing_and_buying_l29_29530


namespace john_swimming_improvement_l29_29299

theorem john_swimming_improvement :
  let initial_lap_time := 35 / 15 -- initial lap time in minutes per lap
  let current_lap_time := 33 / 18 -- current lap time in minutes per lap
  initial_lap_time - current_lap_time = 1 / 9 := 
by
  -- Definition of initial and current lap times are implied in Lean.
  sorry

end john_swimming_improvement_l29_29299


namespace number_at_two_units_right_of_origin_l29_29738

theorem number_at_two_units_right_of_origin : 
  ∀ (n : ℝ), (n = 0) →
  ∀ (x : ℝ), (x = n + 2) →
  x = 2 := 
by
  sorry

end number_at_two_units_right_of_origin_l29_29738


namespace triangle_cosine_sum_l29_29023

theorem triangle_cosine_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hsum : A + B + C = π) : 
  (Real.cos A + Real.cos B + Real.cos C > 1) :=
sorry

end triangle_cosine_sum_l29_29023


namespace regular_polygons_cover_plane_l29_29208

theorem regular_polygons_cover_plane (n : ℕ) (h_n_ge_3 : 3 ≤ n)
    (h_angle_eq : ∀ n, (180 * (1 - (2 / n)) : ℝ) = (internal_angle : ℝ))
    (h_summation_eq : ∃ k : ℕ, k * internal_angle = 360) :
    n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_polygons_cover_plane_l29_29208


namespace ribbons_purple_l29_29915

theorem ribbons_purple (total_ribbons : ℕ) (yellow_ribbons purple_ribbons orange_ribbons black_ribbons : ℕ)
  (h1 : yellow_ribbons = total_ribbons / 4)
  (h2 : purple_ribbons = total_ribbons / 3)
  (h3 : orange_ribbons = total_ribbons / 6)
  (h4 : black_ribbons = 40)
  (h5 : yellow_ribbons + purple_ribbons + orange_ribbons + black_ribbons = total_ribbons) :
  purple_ribbons = 53 :=
by
  sorry

end ribbons_purple_l29_29915


namespace negative_product_implies_negatives_l29_29775

theorem negative_product_implies_negatives (a b c : ℚ) (h : a * b * c < 0) :
  (∃ n : ℕ, n = 1 ∨ n = 3 ∧ (n = 1 ↔ (a < 0 ∧ b > 0 ∧ c > 0 ∨ a > 0 ∧ b < 0 ∧ c > 0 ∨ a > 0 ∧ b > 0 ∧ c < 0)) ∨ 
                                n = 3 ∧ (n = 3 ↔ (a < 0 ∧ b < 0 ∧ c < 0 ∨ a < 0 ∧ b < 0 ∧ c > 0 ∨ a < 0 ∧ b > 0 ∧ c < 0 ∨ a > 0 ∧ b < 0 ∧ c < 0))) :=
  sorry

end negative_product_implies_negatives_l29_29775


namespace find_common_difference_l29_29645

def arithmetic_sequence (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)

theorem find_common_difference (S_n : ℕ → ℝ) (d : ℝ) (h : ∀n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)) 
    (h_condition : S_n 3 / 3 - S_n 2 / 2 = 1) :
  d = 2 :=
sorry

end find_common_difference_l29_29645


namespace coin_same_side_probability_l29_29574

noncomputable def probability_same_side_5_tosses (p : ℚ) := (p ^ 5) + (p ^ 5)

theorem coin_same_side_probability : probability_same_side_5_tosses (1/2) = 1/16 := by
  sorry

end coin_same_side_probability_l29_29574


namespace emery_family_first_hour_distance_l29_29355

noncomputable def total_time : ℝ := 4
noncomputable def remaining_distance : ℝ := 300
noncomputable def first_hour_distance : ℝ := 100

theorem emery_family_first_hour_distance :
  (remaining_distance / (total_time - 1)) = first_hour_distance :=
sorry

end emery_family_first_hour_distance_l29_29355


namespace solution_l29_29993

-- Definition of the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * m * x + 1

-- Statement of the problem
theorem solution (m : ℝ) (x : ℝ) (h : quadratic_equation m x = (m - 2) * x^2 + 3 * m * x + 1) : m ≠ 2 :=
by
  sorry

end solution_l29_29993


namespace inequality_selection_l29_29533

theorem inequality_selection (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) 
  (h₃ : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 ∧ (∀ x, |x + a| + |x - b| + c = 4 → x = (a - b)/2) ∧ (a = 8 / 7 ∧ b = 18 / 7 ∧ c = 2 / 7) :=
by
  sorry

end inequality_selection_l29_29533


namespace find_min_value_l29_29918

theorem find_min_value (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y = 2 → c = 1 / 2 ∧ (8 / ((x + 2) * (y + 4))) ≥ c) :=
  sorry

end find_min_value_l29_29918


namespace find_m_l29_29389

theorem find_m (x : ℝ) (m : ℝ) (h1 : x > 2) (h2 : x - 3 * m + 1 > 0) : m = 1 :=
sorry

end find_m_l29_29389


namespace sisterPassesMeInOppositeDirection_l29_29806

noncomputable def numberOfPasses (laps_sister : ℕ) : ℕ :=
if laps_sister > 1 then 2 * laps_sister else 0

theorem sisterPassesMeInOppositeDirection
  (my_laps : ℕ) (laps_sister : ℕ) (passes_in_same_direction : ℕ) :
  my_laps = 1 ∧ passes_in_same_direction = 2 ∧ laps_sister > 1 →
  passes_in_same_direction * 2 = 4 :=
by intros; sorry

end sisterPassesMeInOppositeDirection_l29_29806


namespace teams_played_same_matches_l29_29473

theorem teams_played_same_matches (n : ℕ) (h : n = 30)
  (matches_played : Fin n → ℕ) :
  ∃ (i j : Fin n), i ≠ j ∧ matches_played i = matches_played j :=
by
  sorry

end teams_played_same_matches_l29_29473


namespace henry_finishes_on_thursday_l29_29553

theorem henry_finishes_on_thursday :
  let total_days := 210
  let start_day := 4  -- Assume Thursday is 4th day of the week in 0-indexed (0=Sunday, 1=Monday, ..., 6=Saturday)
  (start_day + total_days) % 7 = start_day :=
by
  sorry

end henry_finishes_on_thursday_l29_29553


namespace gray_region_area_l29_29631

theorem gray_region_area (r R : ℝ) (hR : R = 3 * r) (h_diff : R - r = 3) :
  π * (R^2 - r^2) = 18 * π :=
by
  -- The proof goes here
  sorry

end gray_region_area_l29_29631


namespace min_value_expression_l29_29980

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
sorry

end min_value_expression_l29_29980


namespace least_number_to_add_for_divisibility_by_nine_l29_29748

theorem least_number_to_add_for_divisibility_by_nine : ∃ x : ℕ, (4499 + x) % 9 = 0 ∧ x = 1 :=
by
  sorry

end least_number_to_add_for_divisibility_by_nine_l29_29748


namespace tennis_balls_in_each_container_l29_29652

theorem tennis_balls_in_each_container :
  let total_balls := 100
  let given_away := total_balls / 2
  let remaining := total_balls - given_away
  let containers := 5
  remaining / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l29_29652


namespace second_group_children_is_16_l29_29985

def cases_purchased : ℕ := 13
def bottles_per_case : ℕ := 24
def camp_days : ℕ := 3
def first_group_children : ℕ := 14
def third_group_children : ℕ := 12
def bottles_per_child_per_day : ℕ := 3
def additional_bottles_needed : ℕ := 255

def fourth_group_children (x : ℕ) : ℕ := (14 + x + 12) / 2
def total_initial_bottles : ℕ := cases_purchased * bottles_per_case
def total_children (x : ℕ) : ℕ := 14 + x + 12 + fourth_group_children x 

def total_consumption (x : ℕ) : ℕ := (total_children x) * bottles_per_child_per_day * camp_days
def total_bottles_needed : ℕ := total_initial_bottles + additional_bottles_needed

theorem second_group_children_is_16 :
  ∃ x : ℕ, total_consumption x = total_bottles_needed ∧ x = 16 :=
by
  sorry

end second_group_children_is_16_l29_29985


namespace man_can_lift_one_box_each_hand_l29_29381

theorem man_can_lift_one_box_each_hand : 
  ∀ (people boxes : ℕ), people = 7 → boxes = 14 → (boxes / people) / 2 = 1 :=
by
  intros people boxes h_people h_boxes
  sorry

end man_can_lift_one_box_each_hand_l29_29381


namespace average_cd_e_l29_29906

theorem average_cd_e (c d e : ℝ) (h : (4 + 6 + 9 + c + d + e) / 6 = 20) : 
    (c + d + e) / 3 = 101 / 3 :=
by
  sorry

end average_cd_e_l29_29906


namespace sets_produced_and_sold_is_500_l29_29679

-- Define the initial conditions as constants
def initial_outlay : ℕ := 10000
def manufacturing_cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def total_profit : ℕ := 5000

-- The proof goal
theorem sets_produced_and_sold_is_500 (x : ℕ) : 
  (total_profit = selling_price_per_set * x - (initial_outlay + manufacturing_cost_per_set * x)) → 
  x = 500 :=
by 
  sorry

end sets_produced_and_sold_is_500_l29_29679


namespace nested_fraction_expression_l29_29371

theorem nested_fraction_expression : 
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := 
by sorry

end nested_fraction_expression_l29_29371


namespace cut_problem_l29_29979

theorem cut_problem (n : ℕ) : (1 / 2 : ℝ) ^ n = 1 / 64 ↔ n = 6 :=
by
  sorry

end cut_problem_l29_29979


namespace cos_squared_pi_over_4_minus_alpha_l29_29441

theorem cos_squared_pi_over_4_minus_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 3 / 4) :
  Real.cos (Real.pi / 4 - α) ^ 2 = 9 / 25 :=
by
  sorry

end cos_squared_pi_over_4_minus_alpha_l29_29441


namespace determine_m_even_function_l29_29146

theorem determine_m_even_function (m : ℤ) :
  (∀ x : ℤ, (x^2 + (m-1)*x) = (x^2 - (m-1)*x)) → m = 1 :=
by
    sorry

end determine_m_even_function_l29_29146


namespace find_angle_l29_29479

theorem find_angle (x : ℝ) (h1 : 90 - x = (1/2) * (180 - x)) : x = 90 :=
by
  sorry

end find_angle_l29_29479


namespace compare_neg_fractions_l29_29334

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l29_29334


namespace translate_statement_to_inequality_l29_29640

theorem translate_statement_to_inequality (y : ℝ) : (1/2) * y + 5 > 0 ↔ True := 
sorry

end translate_statement_to_inequality_l29_29640


namespace cuboid_dimensions_exist_l29_29958

theorem cuboid_dimensions_exist (l w h : ℝ) 
  (h1 : l * w = 5) 
  (h2 : l * h = 8) 
  (h3 : w * h = 10) 
  (h4 : l * w * h = 200) : 
  ∃ (l w h : ℝ), l = 4 ∧ w = 2.5 ∧ h = 2 := 
sorry

end cuboid_dimensions_exist_l29_29958


namespace combined_cost_is_3490_l29_29148

-- Definitions for the quantities of gold each person has and their respective prices per gram
def Gary_gold_grams : ℕ := 30
def Gary_gold_price_per_gram : ℕ := 15

def Anna_gold_grams : ℕ := 50
def Anna_gold_price_per_gram : ℕ := 20

def Lisa_gold_grams : ℕ := 40
def Lisa_gold_price_per_gram : ℕ := 18

def John_gold_grams : ℕ := 60
def John_gold_price_per_gram : ℕ := 22

-- Combined cost
def combined_cost : ℕ :=
  Gary_gold_grams * Gary_gold_price_per_gram +
  Anna_gold_grams * Anna_gold_price_per_gram +
  Lisa_gold_grams * Lisa_gold_price_per_gram +
  John_gold_grams * John_gold_price_per_gram

-- Proof that the combined cost is equal to $3490
theorem combined_cost_is_3490 : combined_cost = 3490 :=
  by
  -- proof skipped
  sorry

end combined_cost_is_3490_l29_29148


namespace complex_square_sum_eq_zero_l29_29222

theorem complex_square_sum_eq_zero (i : ℂ) (h : i^2 = -1) : (1 + i)^2 + (1 - i)^2 = 0 :=
sorry

end complex_square_sum_eq_zero_l29_29222


namespace grade_students_difference_condition_l29_29199

variables (G1 G2 G5 : ℕ)

theorem grade_students_difference_condition (h : G1 + G2 = G2 + G5 + 30) : G1 - G5 = 30 :=
sorry

end grade_students_difference_condition_l29_29199


namespace cost_price_of_computer_table_l29_29102

/-- The cost price \(C\) of a computer table is Rs. 7000 -/
theorem cost_price_of_computer_table : 
  ∃ (C : ℝ), (S = 1.20 * C) ∧ (S = 8400) → C = 7000 := 
by 
  sorry

end cost_price_of_computer_table_l29_29102


namespace solve_for_x_l29_29149

theorem solve_for_x : ∀ x : ℝ, 3^(2 * x) = Real.sqrt 27 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l29_29149


namespace rate_of_painting_per_sq_m_l29_29643

def length_of_floor : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def ratio_of_length_to_breadth : ℝ := 3

theorem rate_of_painting_per_sq_m :
  ∃ (rate : ℝ), rate = 3 :=
by
  let B := length_of_floor / ratio_of_length_to_breadth
  let A := length_of_floor * B
  let rate := total_cost / A
  use rate
  sorry  -- Skipping proof as instructed

end rate_of_painting_per_sq_m_l29_29643


namespace evaluate_expression_l29_29183

def expression (x y : ℤ) : ℤ :=
  y * (y - 2 * x) ^ 2

theorem evaluate_expression : 
  expression 4 2 = 72 :=
by
  -- Proof will go here
  sorry

end evaluate_expression_l29_29183


namespace length_segment_MN_l29_29204

open Real

noncomputable def line (x : ℝ) : ℝ := x + 2

def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem length_segment_MN :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ →
    on_circle x₂ y₂ →
    (line x₁ = y₁ ∧ line x₂ = y₂) →
    dist (x₁, y₁) (x₂, y₂) = 2 * sqrt 3 :=
by
  sorry

end length_segment_MN_l29_29204


namespace TruckloadsOfSand_l29_29364

theorem TruckloadsOfSand (S : ℝ) (totalMat dirt cement : ℝ) 
  (h1 : totalMat = 0.67) 
  (h2 : dirt = 0.33) 
  (h3 : cement = 0.17) 
  (h4 : totalMat = S + dirt + cement) : 
  S = 0.17 := 
  by 
    sorry

end TruckloadsOfSand_l29_29364


namespace abe_job_time_l29_29793

theorem abe_job_time (A G C: ℕ) : G = 70 → C = 21 → (1 / G + 1 / A = 1 / C) → A = 30 := by
sorry

end abe_job_time_l29_29793


namespace collin_savings_l29_29218

theorem collin_savings :
  let cans_home := 12
  let cans_grandparents := 3 * cans_home
  let cans_neighbor := 46
  let cans_dad := 250
  let total_cans := cans_home + cans_grandparents + cans_neighbor + cans_dad
  let money_per_can := 0.25
  let total_money := total_cans * money_per_can
  let savings := total_money / 2
  savings = 43 := 
  by 
  sorry

end collin_savings_l29_29218


namespace max_distance_traveled_l29_29517

theorem max_distance_traveled (fare: ℝ) (x: ℝ) :
  fare = 17.2 → 
  x > 3 →
  1.4 * (x - 3) + 6 ≤ fare → 
  x ≤ 11 := by
  sorry

end max_distance_traveled_l29_29517


namespace exists_n_le_2500_perfect_square_l29_29105

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_squares_segment (n : ℕ) : ℚ :=
  ((26 * n^3 + 12 * n^2 + n) / 3)

theorem exists_n_le_2500_perfect_square :
  ∃ (n : ℕ), n ≤ 2500 ∧ ∃ (k : ℚ), k^2 = (sum_of_squares n) * (sum_of_squares_segment n) :=
sorry

end exists_n_le_2500_perfect_square_l29_29105


namespace mapping_f_correct_l29_29682

theorem mapping_f_correct (a1 a2 a3 a4 b1 b2 b3 b4 : ℤ) :
  (∀ (x : ℤ), x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4 = (x + 1)^4 + b1 * (x + 1)^3 + b2 * (x + 1)^2 + b3 * (x + 1) + b4) →
  a1 = 4 → a2 = 3 → a3 = 2 → a4 = 1 →
  b1 = 0 → b1 + b2 + b3 + b4 = 0 →
  (b1, b2, b3, b4) = (0, -3, 4, -1) :=
by
  intros
  sorry

end mapping_f_correct_l29_29682


namespace grade3_trees_count_l29_29143

-- Declare the variables and types
variables (x y : ℕ)

-- Given conditions as definitions
def students_equation := (2 * x + y = 100)
def trees_equation := (9 * x + (13 / 2) * y = 566)
def avg_trees_grade3 := 4

-- Assert the problem statement
theorem grade3_trees_count (hx : students_equation x y) (hy : trees_equation x y) : 
  (avg_trees_grade3 * x = 84) :=
sorry

end grade3_trees_count_l29_29143


namespace bees_process_2_77_kg_nectar_l29_29072

noncomputable def nectar_to_honey : ℝ :=
  let percent_other_in_nectar : ℝ := 0.30
  let other_mass_in_honey : ℝ := 0.83
  other_mass_in_honey / percent_other_in_nectar

theorem bees_process_2_77_kg_nectar :
  nectar_to_honey = 2.77 :=
by
  sorry

end bees_process_2_77_kg_nectar_l29_29072


namespace range_of_m_l29_29723

theorem range_of_m (m : ℝ) : 
  ((m + 3) * (m - 4) < 0) → 
  (m^2 - 4 * (m + 3) ≤ 0) → 
  (-2 ≤ m ∧ m < 4) :=
by 
  intro h1 h2
  sorry

end range_of_m_l29_29723


namespace const_sequence_l29_29228

theorem const_sequence (x y : ℝ) (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ n, a n - a (n + 1) = (a n ^ 2 - 1) / (a n + a (n - 1)))
  (h2 : ∀ n, a n = a (n + 1) → a n ^ 2 = 1 ∧ a n ≠ -a (n - 1))
  (h_init : a 1 = y ∧ a 0 = x)
  (hx : |x| = 1 ∧ y ≠ -x) :
  (∃ n0, ∀ n ≥ n0, a n = 1 ∨ a n = -1) := sorry

end const_sequence_l29_29228


namespace sum_of_new_dimensions_l29_29718

theorem sum_of_new_dimensions (s : ℕ) (h₁ : s^2 = 36) (h₂ : s' = s - 1) : s' + s' + s' = 15 :=
sorry

end sum_of_new_dimensions_l29_29718


namespace range_of_a_for_empty_solution_set_l29_29401

theorem range_of_a_for_empty_solution_set : 
  (∀ a : ℝ, (∀ x : ℝ, |x - 4| + |3 - x| < a → false) ↔ a ≤ 1) := 
sorry

end range_of_a_for_empty_solution_set_l29_29401


namespace first_spade_second_king_prob_l29_29509

-- Definitions and conditions of the problem
def total_cards := 52
def total_spades := 13
def total_kings := 4
def spades_excluding_king := 12 -- Number of spades excluding the king of spades
def remaining_kings_after_king_spade := 3

-- Calculate probabilities for each case
def first_non_king_spade_prob := spades_excluding_king / total_cards
def second_king_after_non_king_spade_prob := total_kings / (total_cards - 1)
def case1_prob := first_non_king_spade_prob * second_king_after_non_king_spade_prob

def first_king_spade_prob := 1 / total_cards
def second_king_after_king_spade_prob := remaining_kings_after_king_spade / (total_cards - 1)
def case2_prob := first_king_spade_prob * second_king_after_king_spade_prob

def combined_prob := case1_prob + case2_prob

-- The proof statement
theorem first_spade_second_king_prob :
  combined_prob = 1 / total_cards := by
  sorry

end first_spade_second_king_prob_l29_29509


namespace eggs_in_larger_omelette_l29_29982

theorem eggs_in_larger_omelette :
  ∀ (total_eggs : ℕ) (orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette : ℕ),
    total_eggs = 84 →
    orders_3_eggs_first_hour = 5 →
    orders_3_eggs_third_hour = 3 →
    orders_large_eggs_second_hour = 7 →
    orders_large_eggs_last_hour = 8 →
    num_eggs_per_3_omelette = 3 →
    (total_eggs - (orders_3_eggs_first_hour * num_eggs_per_3_omelette + orders_3_eggs_third_hour * num_eggs_per_3_omelette)) / (orders_large_eggs_second_hour + orders_large_eggs_last_hour) = 4 :=
by
  intros total_eggs orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette
  sorry

end eggs_in_larger_omelette_l29_29982


namespace lucy_50_cent_items_l29_29799

theorem lucy_50_cent_items :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 50 * a + 150 * b + 300 * c = 4500 ∧ a = 6 :=
by
  sorry

end lucy_50_cent_items_l29_29799


namespace total_precious_stones_is_305_l29_29234

theorem total_precious_stones_is_305 :
  let agate := 25
  let olivine := agate + 5
  let sapphire := 2 * olivine
  let diamond := olivine + 11
  let amethyst := sapphire + diamond
  let ruby := diamond + 7
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 :=
by
  sorry

end total_precious_stones_is_305_l29_29234


namespace number_of_kids_at_circus_l29_29991

theorem number_of_kids_at_circus (K A : ℕ) 
(h1 : ∀ x, 5 * x = 1 / 2 * 10 * x)
(h2 : 5 * K + 10 * A = 50) : K = 2 :=
sorry

end number_of_kids_at_circus_l29_29991


namespace number_of_five_dollar_bills_l29_29314

theorem number_of_five_dollar_bills (total_money denomination expected_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : denomination = 5) 
  (h3 : expected_bills = total_money / denomination) : 
  expected_bills = 9 :=
by
  sorry

end number_of_five_dollar_bills_l29_29314


namespace machine_worked_yesterday_l29_29098

noncomputable def shirts_made_per_minute : ℕ := 3
noncomputable def shirts_made_yesterday : ℕ := 9

theorem machine_worked_yesterday : 
  (shirts_made_yesterday / shirts_made_per_minute) = 3 :=
sorry

end machine_worked_yesterday_l29_29098


namespace exists_solution_real_l29_29044

theorem exists_solution_real (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3 / 2 :=
by
  sorry

end exists_solution_real_l29_29044


namespace square_area_l29_29362

theorem square_area (p : ℝ) (h : p = 20) : (p / 4) ^ 2 = 25 :=
by
  sorry

end square_area_l29_29362


namespace floor_neg_seven_fourths_l29_29468

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l29_29468


namespace range_f_l29_29712

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_f : Set.range f = Set.Ioi 0 ∪ Set.Iio 0 := by
  sorry

end range_f_l29_29712


namespace class_A_scores_more_uniform_l29_29563

-- Define the variances of the test scores for classes A and B
def variance_A := 13.2
def variance_B := 26.26

-- Theorem: Prove that the scores of the 10 students from class A are more uniform than those from class B
theorem class_A_scores_more_uniform :
  variance_A < variance_B :=
  by
    -- Assume the given variances and state the comparison
    have h : 13.2 < 26.26 := by sorry
    exact h

end class_A_scores_more_uniform_l29_29563


namespace exponent_m_n_add_l29_29492

variable (a : ℝ) (m n : ℕ)

theorem exponent_m_n_add (h1 : a ^ m = 2) (h2 : a ^ n = 3) : a ^ (m + n) = 6 := by
  sorry

end exponent_m_n_add_l29_29492


namespace value_of_a5_l29_29332

variable (a_n : ℕ → ℝ)
variable (a1 a9 a5 : ℝ)

-- Given conditions
axiom a1_plus_a9_eq_10 : a1 + a9 = 10
axiom arithmetic_sequence : ∀ n, a_n n = a1 + (n - 1) * (a_n 2 - a1)

-- Prove that a5 = 5
theorem value_of_a5 : a5 = 5 :=
by
  sorry

end value_of_a5_l29_29332


namespace problem_statement_l29_29154

theorem problem_statement (f : ℕ → ℕ) (h1 : f 1 = 4) (h2 : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4) :
  f 2 + f 5 = 125 :=
by
  sorry

end problem_statement_l29_29154


namespace fraction_spent_on_fruits_l29_29251

theorem fraction_spent_on_fruits (M : ℕ) (hM : M = 24) :
  (M - (M / 3 + M / 6) - 6) / M = 1 / 4 :=
by
  sorry

end fraction_spent_on_fruits_l29_29251


namespace point_relationship_l29_29711

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -(x - 1) ^ 2 + c

noncomputable def y1_def (c : ℝ) : ℝ := quadratic_function (-3) c
noncomputable def y2_def (c : ℝ) : ℝ := quadratic_function (-1) c
noncomputable def y3_def (c : ℝ) : ℝ := quadratic_function 5 c

theorem point_relationship (c : ℝ) :
  y2_def c > y1_def c ∧ y1_def c = y3_def c :=
by
  sorry

end point_relationship_l29_29711


namespace product_of_solutions_abs_eq_40_l29_29714

theorem product_of_solutions_abs_eq_40 :
  (∃ x1 x2 : ℝ, (|3 * x1 - 5| = 40) ∧ (|3 * x2 - 5| = 40) ∧ ((x1 * x2) = -175)) :=
by
  sorry

end product_of_solutions_abs_eq_40_l29_29714


namespace fraction_of_female_participants_is_correct_l29_29147

-- defining conditions
def last_year_males : ℕ := 30
def male_increase_rate : ℚ := 1.1
def female_increase_rate : ℚ := 1.25
def overall_increase_rate : ℚ := 1.2

-- the statement to prove
theorem fraction_of_female_participants_is_correct :
  ∀ (y : ℕ), 
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  total_this_year = males_this_year + females_this_year →
  (females_this_year / total_this_year) = (25 / 36) :=
by
  intros y
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  intro h
  sorry

end fraction_of_female_participants_is_correct_l29_29147


namespace deepak_present_age_l29_29550

def present_age_rahul (x : ℕ) : ℕ := 4 * x
def present_age_deepak (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age : ∀ (x : ℕ), 
  (present_age_rahul x + 22 = 26) →
  present_age_deepak x = 3 := 
by
  intros x h
  sorry

end deepak_present_age_l29_29550


namespace find_y_l29_29903

theorem find_y (y : ℕ) (h : 4 ^ 12 = 64 ^ y) : y = 4 :=
sorry

end find_y_l29_29903


namespace doughnut_machine_completion_time_l29_29603

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end doughnut_machine_completion_time_l29_29603


namespace indeterminate_equation_solution_exists_l29_29125

theorem indeterminate_equation_solution_exists
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * c = b^2 + b + 1) :
  ∃ x y : ℤ, a * x^2 - (2 * b + 1) * x * y + c * y^2 = 1 := by
  sorry

end indeterminate_equation_solution_exists_l29_29125


namespace sum_of_z_values_l29_29286

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem sum_of_z_values (z1 z2 : ℝ) (hz1 : f (3 * z1) = 11) (hz2 : f (3 * z2) = 11) :
  z1 + z2 = - (2 / 9) :=
sorry

end sum_of_z_values_l29_29286


namespace isosceles_right_triangle_square_ratio_l29_29955

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := Real.sqrt 2 / 2

theorem isosceles_right_triangle_square_ratio :
  x / y = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_square_ratio_l29_29955


namespace sequence_contradiction_l29_29467

open Classical

variable {α : Type} (a : ℕ → α) [PartialOrder α]

theorem sequence_contradiction {a : ℕ → ℝ} :
  (∀ n, a n < 2) ↔ ¬ ∃ k, a k ≥ 2 := 
by sorry

end sequence_contradiction_l29_29467


namespace breadth_of_rectangular_plot_l29_29978

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : 3 * b * b = 972) : b = 18 :=
sorry

end breadth_of_rectangular_plot_l29_29978


namespace find_m_eq_5_l29_29457

-- Definitions for the problem conditions
def f (x m : ℝ) := 2 * x + m

theorem find_m_eq_5 (m : ℝ) (a b : ℝ) :
  (a = f 0 m) ∧ (b = f m m) ∧ ((b - a) = (m - 0 + 5)) → m = 5 :=
by
  sorry

end find_m_eq_5_l29_29457


namespace initial_animal_types_l29_29636

theorem initial_animal_types (x : ℕ) (h1 : 6 * (x + 4) = 54) : x = 5 := 
sorry

end initial_animal_types_l29_29636


namespace possible_values_of_C_l29_29585

variable {α : Type} [LinearOrderedField α]

-- Definitions of points A, B and C
def pointA (a : α) := a
def pointB (b : α) := b
def pointC (c : α) := c

-- Given condition
def given_condition (a b : α) : Prop := (a + 3) ^ 2 + |b - 1| = 0

-- Function to determine if the folding condition is met
def folding_number_line (A B C : α) : Prop :=
  (C = 2 * A - B ∨ C = 2 * B - A ∨ (A + B) / 2 = C)

-- Theorem to prove the possible values of C
theorem possible_values_of_C (a b : α) (h : given_condition a b) :
  ∃ C : α, folding_number_line (pointA a) (pointB b) (pointC C) ∧ (C = -7 ∨ C = 5 ∨ C = -1) :=
sorry

end possible_values_of_C_l29_29585


namespace prime_divisor_exponent_l29_29399

theorem prime_divisor_exponent (a n : ℕ) (p : ℕ) 
    (ha : a ≥ 2)
    (hn : n ≥ 1) 
    (hp : Nat.Prime p) 
    (hdiv : p ∣ a^(2^n) + 1) :
    2^(n+1) ∣ (p-1) :=
by
  sorry

end prime_divisor_exponent_l29_29399


namespace odd_function_example_l29_29115

theorem odd_function_example (f : ℝ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_neg : ∀ x, x < 0 → f x = x + 2) : f 0 + f 3 = 1 :=
by
  sorry

end odd_function_example_l29_29115


namespace min_x_plus_y_of_positive_l29_29883

open Real

theorem min_x_plus_y_of_positive (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_of_positive_l29_29883


namespace find_an_find_n_l29_29995

noncomputable def a_n (n : ℕ) : ℤ := 12 + (n - 1) * 2

noncomputable def S_n (n : ℕ) : ℤ := n * 12 + (n * (n - 1) / 2) * 2

theorem find_an (n : ℕ) : a_n n = 2 * n + 10 :=
by sorry

theorem find_n (n : ℕ) (S_n : ℤ) : S_n = 242 → n = 11 :=
by sorry

end find_an_find_n_l29_29995


namespace necessary_but_not_sufficient_l29_29255

theorem necessary_but_not_sufficient (x : ℝ) : 
  (0 < x ∧ x < 2) → (x^2 - x - 6 < 0) ∧ ¬ ((x^2 - x - 6 < 0) → (0 < x ∧ x < 2)) :=
by
  sorry

end necessary_but_not_sufficient_l29_29255


namespace total_cans_collected_l29_29326

theorem total_cans_collected 
  (bags_saturday : ℕ) 
  (bags_sunday : ℕ) 
  (cans_per_bag : ℕ) 
  (h1 : bags_saturday = 6) 
  (h2 : bags_sunday = 3) 
  (h3 : cans_per_bag = 8) : 
  bags_saturday + bags_sunday * cans_per_bag = 72 := 
by 
  simp [h1, h2, h3]; -- Simplify using the given conditions
  sorry -- Placeholder for the computation proof

end total_cans_collected_l29_29326


namespace arithmetic_sequence_common_difference_l29_29811

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℤ) (h_arithmetic : ∀ n, (a_n (n + 1) = a_n n + d)) 
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99) : 
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l29_29811


namespace part1_part2_l29_29024

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem part1 : A ∩ B = {3, 5} := by
  sorry

theorem part2 : (U \ A) ∪ B = {3, 4, 5, 6} := by
  sorry

end part1_part2_l29_29024


namespace problem_AD_l29_29606

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

open Real

theorem problem_AD :
  (∀ x, 0 < x ∧ x < π / 4 → f x < f (x + 0.01) ∧ g x < g (x + 0.01)) ∧
  (∃ x, x = π / 4 ∧ f x + g x = 1 / 2 + sqrt 2) :=
by
  sorry

end problem_AD_l29_29606


namespace division_dividend_l29_29006

/-- In a division sum, the quotient is 40, the divisor is 72, and the remainder is 64. We need to prove that the dividend is 2944. -/
theorem division_dividend : 
  let Q := 40
  let D := 72
  let R := 64
  (D * Q + R = 2944) :=
by
  sorry

end division_dividend_l29_29006


namespace smaller_angle_at_8_15_l29_29356

noncomputable def hour_hand_position (h m : ℕ) : ℝ := (↑h % 12) * 30 + (↑m / 60) * 30

noncomputable def minute_hand_position (m : ℕ) : ℝ := ↑m / 60 * 360

noncomputable def angle_between_hands (h m : ℕ) : ℝ :=
  let θ := |hour_hand_position h m - minute_hand_position m|
  min θ (360 - θ)

theorem smaller_angle_at_8_15 : angle_between_hands 8 15 = 157.5 := by
  sorry

end smaller_angle_at_8_15_l29_29356


namespace number_of_birdhouses_l29_29900

-- Definitions for the conditions
def cost_per_nail : ℝ := 0.05
def cost_per_plank : ℝ := 3.0
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def total_cost : ℝ := 88.0

-- Total cost calculation per birdhouse
def cost_per_birdhouse := planks_per_birdhouse * cost_per_plank + nails_per_birdhouse * cost_per_nail

-- Proving that the number of birdhouses is 4
theorem number_of_birdhouses : total_cost / cost_per_birdhouse = 4 := by
  sorry

end number_of_birdhouses_l29_29900


namespace find_product_of_roots_l29_29431

noncomputable def equation (x : ℝ) : ℝ := (Real.sqrt 2023) * x^3 - 4047 * x^2 + 3

theorem find_product_of_roots (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : equation x1 = 0) (h4 : equation x2 = 0) (h5 : equation x3 = 0) :
  x2 * (x1 + x3) = 3 :=
by
  sorry

end find_product_of_roots_l29_29431


namespace sum_of_arithmetic_sequence_9_terms_l29_29922

-- Define the odd function and its properties
variables {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = -f (x)) 
          (h2 : ∀ x y, x < y → f x < f y)

-- Define the shifted function g
noncomputable def g (x : ℝ) := f (x - 5)

-- Define the arithmetic sequence with non-zero common difference
variables {a : ℕ → ℝ} (d : ℝ) (h3 : d ≠ 0) 
          (h4 : ∀ n, a (n + 1) = a n + d)

-- Condition given by the problem
variable (h5 : g (a 1) + g (a 9) = 0)

-- Proof obligation
theorem sum_of_arithmetic_sequence_9_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 45 :=
sorry

end sum_of_arithmetic_sequence_9_terms_l29_29922


namespace surface_area_of_sphere_given_cube_volume_8_l29_29139

theorem surface_area_of_sphere_given_cube_volume_8 
  (volume_of_cube : ℝ)
  (h₁ : volume_of_cube = 8) :
  ∃ (surface_area_of_sphere : ℝ), 
  surface_area_of_sphere = 12 * Real.pi :=
by
  sorry

end surface_area_of_sphere_given_cube_volume_8_l29_29139


namespace root_triple_condition_l29_29295

theorem root_triple_condition (a b c α β : ℝ)
  (h_eq : a * α^2 + b * α + c = 0)
  (h_β_eq : β = 3 * α)
  (h_vieta_sum : α + β = -b / a)
  (h_vieta_product : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
by
  sorry

end root_triple_condition_l29_29295


namespace find_r_l29_29469

theorem find_r (f g : ℝ → ℝ) (monic_f : ∀x, f x = (x - r - 2) * (x - r - 8) * (x - a))
  (monic_g : ∀x, g x = (x - r - 4) * (x - r - 10) * (x - b)) (h : ∀ x, f x - g x = r):
  r = 32 :=
by
  sorry

end find_r_l29_29469


namespace unit_triangle_count_bound_l29_29406

variable {L : ℝ} (L_pos : L > 0)
variable {n : ℕ}

/--
  Let \( \Delta \) be an equilateral triangle with side length \( L \), and suppose that \( n \) unit 
  equilateral triangles are drawn inside \( \Delta \) with non-overlapping interiors and each having 
  sides parallel to \( \Delta \) but with opposite orientation. Then,
  we must have \( n \leq \frac{2}{3} L^2 \).
-/
theorem unit_triangle_count_bound (L_pos : L > 0) (n : ℕ) :
  n ≤ (2 / 3) * (L ^ 2) := 
sorry

end unit_triangle_count_bound_l29_29406


namespace find_g_1_l29_29131

noncomputable def g (x : ℝ) : ℝ := sorry -- express g(x) as a 4th degree polynomial with unknown coefficients

-- Conditions given in the problem
axiom cond1 : |g (-1)| = 15
axiom cond2 : |g (0)| = 15
axiom cond3 : |g (2)| = 15
axiom cond4 : |g (3)| = 15
axiom cond5 : |g (4)| = 15

-- The statement we need to prove
theorem find_g_1 : |g 1| = 11 :=
sorry

end find_g_1_l29_29131


namespace negation_statement_l29_29741

open Set

variable {S : Set ℝ}

theorem negation_statement (h : ∀ x ∈ S, 3 * x - 5 > 0) : ∃ x ∈ S, 3 * x - 5 ≤ 0 :=
sorry

end negation_statement_l29_29741


namespace find_fraction_l29_29250

variable (x y z : ℂ) -- All complex numbers
variable (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) -- Non-zero conditions
variable (h2 : x + y + z = 10) -- Sum condition
variable (h3 : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) -- Given equation condition

theorem find_fraction 
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
    (h2 : x + y + z = 10)
    (h3 : 2 * ((x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2) = x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := 
sorry -- Proof yet to be completed

end find_fraction_l29_29250


namespace right_triangle_perimeter_5_shortest_altitude_1_l29_29661

-- Definition of a right-angled triangle's sides with given perimeter and altitude
def right_angled_triangle (a b c : ℚ) : Prop :=
a^2 + b^2 = c^2 ∧ a + b + c = 5 ∧ a * b = c

-- Statement of the theorem to prove the side lengths of the triangle
theorem right_triangle_perimeter_5_shortest_altitude_1 :
  ∃ (a b c : ℚ), right_angled_triangle a b c ∧ (a = 5 / 3 ∧ b = 5 / 4 ∧ c = 25 / 12) ∨ (a = 5 / 4 ∧ b = 5 / 3 ∧ c = 25 / 12) :=
by
  sorry

end right_triangle_perimeter_5_shortest_altitude_1_l29_29661


namespace circle_positions_n_l29_29405

theorem circle_positions_n (n : ℕ) (h1 : n ≥ 23) (h2 : (23 - 7) * 2 + 2 = n) : n = 32 :=
sorry

end circle_positions_n_l29_29405


namespace discount_percentage_l29_29502

theorem discount_percentage (cost_price marked_price : ℝ) (profit_percentage : ℝ) 
  (h_cost_price : cost_price = 47.50)
  (h_marked_price : marked_price = 65)
  (h_profit_percentage : profit_percentage = 0.30) :
  ((marked_price - (cost_price + (profit_percentage * cost_price))) / marked_price) * 100 = 5 :=
by
  sorry

end discount_percentage_l29_29502


namespace combined_area_l29_29019

noncomputable def diagonal : ℝ := 12 * Real.sqrt 2

noncomputable def side_of_square (d : ℝ) : ℝ := d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

theorem combined_area (d : ℝ) (h : d = diagonal) :
  let s := side_of_square d
  let area_sq := area_of_square s
  let r := radius_of_circle d
  let area_circ := area_of_circle r
  area_sq + area_circ = 144 + 72 * Real.pi :=
by
  sorry

end combined_area_l29_29019


namespace more_than_half_millet_on_day_three_l29_29750

-- Definition of the initial conditions
def seeds_in_feeder (n: ℕ) : ℝ :=
  1 + n

def millet_amount (n: ℕ) : ℝ :=
  0.6 * (1 - (0.5)^n)

-- The theorem we want to prove
theorem more_than_half_millet_on_day_three :
  ∀ n, n = 3 → (millet_amount n) / (seeds_in_feeder n) > 0.5 :=
by
  intros n hn
  rw [hn, seeds_in_feeder, millet_amount]
  sorry

end more_than_half_millet_on_day_three_l29_29750


namespace joe_initial_cars_l29_29560

theorem joe_initial_cars (x : ℕ) (h : x + 12 = 62) : x = 50 :=
by {
  sorry
}

end joe_initial_cars_l29_29560


namespace right_triangle_area_l29_29685

theorem right_triangle_area (a b c p S : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a^2 + b^2 = c^2)
  (h4 : p = (a + b + c) / 2) (h5 : S = a * b / 2) :
  p * (p - c) = S ∧ (p - a) * (p - b) = S :=
sorry

end right_triangle_area_l29_29685


namespace mother_hen_heavier_l29_29317

-- Define the weights in kilograms
def weight_mother_hen : ℝ := 2.3
def weight_baby_chick : ℝ := 0.4

-- State the theorem with the final correct answer
theorem mother_hen_heavier :
  weight_mother_hen - weight_baby_chick = 1.9 :=
by
  sorry

end mother_hen_heavier_l29_29317


namespace maximum_capacity_of_smallest_barrel_l29_29720

theorem maximum_capacity_of_smallest_barrel : 
  ∃ (A B C D E F : ℕ), 
    8 ≤ A ∧ A ≤ 16 ∧
    8 ≤ B ∧ B ≤ 16 ∧
    8 ≤ C ∧ C ≤ 16 ∧
    8 ≤ D ∧ D ≤ 16 ∧
    8 ≤ E ∧ E ≤ 16 ∧
    8 ≤ F ∧ F ≤ 16 ∧
    (A + B + C + D + E + F = 72) ∧
    ((C + D) / 2 = 14) ∧ 
    (F = 11 ∨ F = 13) ∧
    (∀ (A' : ℕ), 8 ≤ A' ∧ A' ≤ 16 ∧
      ∃ (B' C' D' E' F' : ℕ), 
      8 ≤ B' ∧ B' ≤ 16 ∧
      8 ≤ C' ∧ C' ≤ 16 ∧
      8 ≤ D' ∧ D' ≤ 16 ∧
      8 ≤ E' ∧ E' ≤ 16 ∧
      8 ≤ F' ∧ F' ≤ 16 ∧
      (A' + B' + C' + D' + E' + F' = 72) ∧
      ((C' + D') / 2 = 14) ∧ 
      (F' = 11 ∨ F' = 13) → A' ≤ A ) :=
sorry

end maximum_capacity_of_smallest_barrel_l29_29720


namespace henrys_distance_from_start_l29_29784

noncomputable def meters_to_feet (x : ℝ) : ℝ := x * 3.281
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem henrys_distance_from_start :
  let west_walk_feet := meters_to_feet 15
  let north_walk_feet := 60
  let east_walk_feet := 156
  let south_walk_meter_backwards := 30
  let south_walk_feet_backwards := 12
  let total_south_feet := meters_to_feet south_walk_meter_backwards + south_walk_feet_backwards
  let net_south_feet := total_south_feet - north_walk_feet
  let net_east_feet := east_walk_feet - west_walk_feet
  distance 0 0 net_east_feet (-net_south_feet) = 118 := 
by
  sorry

end henrys_distance_from_start_l29_29784


namespace line_param_func_l29_29202

theorem line_param_func (t : ℝ) : 
    ∃ f : ℝ → ℝ, (∀ t, (20 * t - 14) = 2 * (f t) - 30) ∧ (f t = 10 * t + 8) := by
  sorry

end line_param_func_l29_29202


namespace erasers_difference_l29_29062

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l29_29062


namespace number_mod_conditions_l29_29205

theorem number_mod_conditions :
  ∃ N, (N % 10 = 9) ∧ (N % 9 = 8) ∧ (N % 8 = 7) ∧ (N % 7 = 6) ∧
       (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧
       N = 2519 :=
by
  sorry

end number_mod_conditions_l29_29205


namespace simplify_expression_l29_29207

variable (p : ℤ)

-- Defining the given expression
def initial_expression : ℤ := ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9)

-- Statement asserting the simplification
theorem simplify_expression : initial_expression p = 13 * p - 30 := 
sorry

end simplify_expression_l29_29207


namespace total_profit_calculation_l29_29078

theorem total_profit_calculation (A B C : ℕ) (C_share total_profit : ℕ) 
  (hA : A = 27000) 
  (hB : B = 72000) 
  (hC : C = 81000) 
  (hC_share : C_share = 36000) 
  (h_ratio : C_share * 20 = total_profit * 9) :
  total_profit = 80000 := by
  sorry

end total_profit_calculation_l29_29078


namespace no_prime_divisible_by_56_l29_29214

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divisible_by_56 (n : ℕ) : Prop := 56 ∣ n

theorem no_prime_divisible_by_56 : ¬ ∃ p, is_prime p ∧ divisible_by_56 p := 
  sorry

end no_prime_divisible_by_56_l29_29214


namespace randy_piggy_bank_balance_l29_29660

def initial_amount : ℕ := 200
def store_trip_cost : ℕ := 2
def trips_per_month : ℕ := 4
def extra_cost_trip : ℕ := 1
def extra_trip_interval : ℕ := 3
def months_in_year : ℕ := 12
def weekly_income : ℕ := 15
def internet_bill_per_month : ℕ := 20
def birthday_gift : ℕ := 100
def weeks_in_year : ℕ := 52

-- To be proved
theorem randy_piggy_bank_balance : 
  initial_amount 
  + (weekly_income * weeks_in_year) 
  + birthday_gift 
  - ((store_trip_cost * trips_per_month * months_in_year)
  + (months_in_year / extra_trip_interval) * extra_cost_trip
  + (internet_bill_per_month * months_in_year))
  = 740 :=
by
  sorry

end randy_piggy_bank_balance_l29_29660


namespace find_A_l29_29256

theorem find_A (A B : ℝ) 
  (h1 : A - 3 * B = 303.1)
  (h2 : 10 * B = A) : 
  A = 433 :=
by
  sorry

end find_A_l29_29256


namespace find_machines_l29_29229

theorem find_machines (R : ℝ) : 
  (N : ℕ) -> 
  (H1 : N * R * 6 = 1) -> 
  (H2 : 4 * R * 12 = 1) -> 
  N = 8 :=
by
  sorry

end find_machines_l29_29229


namespace work_done_by_student_l29_29961

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end work_done_by_student_l29_29961


namespace line_parallel_not_coincident_l29_29905

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l29_29905


namespace m_not_in_P_l29_29578

noncomputable def m : ℝ := Real.sqrt 3
def P : Set ℝ := { x | x^2 - Real.sqrt 2 * x ≤ 0 }

theorem m_not_in_P : m ∉ P := by
  sorry

end m_not_in_P_l29_29578


namespace discount_percentage_is_20_l29_29919

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end discount_percentage_is_20_l29_29919


namespace num_people_for_new_avg_l29_29321

def avg_salary := 430
def old_supervisor_salary := 870
def new_supervisor_salary := 870
def num_workers := 8
def total_people_before := num_workers + 1
def total_salary_before := total_people_before * avg_salary
def workers_salary := total_salary_before - old_supervisor_salary
def total_salary_after := workers_salary + new_supervisor_salary

theorem num_people_for_new_avg :
    ∃ (x : ℕ), x * avg_salary = total_salary_after ∧ x = 9 :=
by
  use 9
  field_simp
  sorry

end num_people_for_new_avg_l29_29321


namespace problem_DE_length_l29_29425

theorem problem_DE_length
  (AB AD : ℝ)
  (AB_eq : AB = 7)
  (AD_eq : AD = 10)
  (area_eq : 7 * CE = 140)
  (DC CE DE : ℝ)
  (DC_eq : DC = 7)
  (CE_eq : CE = 20)
  : DE = Real.sqrt 449 :=
by
  sorry

end problem_DE_length_l29_29425


namespace catch_up_distance_l29_29279

def v_a : ℝ := 10 -- A's speed in kmph
def v_b : ℝ := 20 -- B's speed in kmph
def t : ℝ := 10 -- Time in hours when B starts after A

theorem catch_up_distance : v_b * t + v_a * t = 200 :=
by sorry

end catch_up_distance_l29_29279


namespace value_of_f_at_log_l29_29277

noncomputable def f : ℝ → ℝ := sorry -- We will define this below

-- Conditions as hypotheses
axiom odd_f : ∀ x : ℝ, f (-x) = - f (x)
axiom periodic_f : ∀ x : ℝ, f (x + 2) + f (x) = 0
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = 2^x - 1

-- Theorem statement
theorem value_of_f_at_log : f (Real.logb (1/8) 125) = 1 / 4 :=
sorry

end value_of_f_at_log_l29_29277


namespace Rajesh_Spend_Salary_on_Food_l29_29842

theorem Rajesh_Spend_Salary_on_Food
    (monthly_salary : ℝ)
    (percentage_medicines : ℝ)
    (savings_percentage : ℝ)
    (savings : ℝ) :
    monthly_salary = 15000 ∧
    percentage_medicines = 0.20 ∧
    savings_percentage = 0.60 ∧
    savings = 4320 →
    (32 : ℝ) = ((monthly_salary * percentage_medicines + monthly_salary * (1 - (percentage_medicines + savings_percentage))) / monthly_salary) * 100 :=
by
  sorry

end Rajesh_Spend_Salary_on_Food_l29_29842


namespace rectangle_perimeter_l29_29493

theorem rectangle_perimeter (a b : ℤ) (h1 : a ≠ b) (h2 : 2 * (2 * a + 2 * b) - a * b = 12) : 2 * (a + b) = 26 :=
sorry

end rectangle_perimeter_l29_29493


namespace total_money_divided_l29_29043

theorem total_money_divided (A B C : ℝ) (h1 : A = (1 / 2) * B) (h2 : B = (1 / 2) * C) (h3 : C = 208) :
  A + B + C = 364 := 
sorry

end total_money_divided_l29_29043


namespace Peter_finishes_all_tasks_at_5_30_PM_l29_29868

-- Definitions representing the initial conditions
def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
def task_durations : List ℕ :=
  [30, 30, 60, 120, 240] -- Durations of the 5 tasks in minutes
  
-- Statement for the proof problem
theorem Peter_finishes_all_tasks_at_5_30_PM :
  let total_duration := task_durations.sum 
  let finish_time := start_time + total_duration
  finish_time = 17 * 60 + 30 := -- 5:30 PM in minutes
  sorry

end Peter_finishes_all_tasks_at_5_30_PM_l29_29868


namespace average_snack_sales_per_ticket_l29_29650

theorem average_snack_sales_per_ticket :
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  (total_sales / movie_tickets = 2.79) :=
by
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  show total_sales / movie_tickets = 2.79
  sorry

end average_snack_sales_per_ticket_l29_29650


namespace distance_from_point_to_x_axis_l29_29342

theorem distance_from_point_to_x_axis (x y : ℤ) (h : (x, y) = (5, -12)) : |y| = 12 :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end distance_from_point_to_x_axis_l29_29342


namespace find_alpha_l29_29649

noncomputable def angle_in_interval (α : ℝ) : Prop :=
  370 < α ∧ α < 520 

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = 1 / 2) (h_interval: angle_in_interval α) : α = 420 :=
sorry

end find_alpha_l29_29649


namespace sales_percentage_l29_29813

theorem sales_percentage (pens_sales pencils_sales notebooks_sales : ℕ) 
  (h1 : pens_sales = 25)
  (h2 : pencils_sales = 20)
  (h3 : notebooks_sales = 30) :
  100 - (pens_sales + pencils_sales + notebooks_sales) = 25 :=
by
  sorry

end sales_percentage_l29_29813


namespace sandy_paints_area_l29_29897

-- Definition of the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 5
def door_height : ℝ := 1
def door_length : ℝ := 6.5

-- Areas computation
def wall_area : ℝ := wall_height * wall_length
def window_area : ℝ := window_height * window_length
def door_area : ℝ := door_height * door_length

-- Area to be painted
def area_not_painted : ℝ := window_area + door_area
def area_to_be_painted : ℝ := wall_area - area_not_painted

-- The theorem to prove
theorem sandy_paints_area : area_to_be_painted = 128.5 := by
  -- The proof is omitted
  sorry

end sandy_paints_area_l29_29897


namespace max_digit_sum_l29_29189

-- Define the condition for the hours and minutes digits
def is_valid_hour (h : ℕ) := 0 ≤ h ∧ h < 24
def is_valid_minute (m : ℕ) := 0 ≤ m ∧ m < 60

-- Define the function to calculate the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main statement: Prove that the maximum sum of the digits in the display is 24
theorem max_digit_sum : ∃ h m: ℕ, is_valid_hour h ∧ is_valid_minute m ∧ 
  sum_of_digits h + sum_of_digits m = 24 :=
sorry

end max_digit_sum_l29_29189


namespace soccer_ball_cost_l29_29428

theorem soccer_ball_cost (F S : ℝ) 
  (h1 : 3 * F + S = 155) 
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 := 
sorry

end soccer_ball_cost_l29_29428


namespace new_average_commission_is_250_l29_29622

-- Definitions based on the problem conditions
def C : ℝ := 1000
def n : ℝ := 6
def increase_in_average_commission : ℝ := 150

-- Theorem stating the new average commission is $250
theorem new_average_commission_is_250 (x : ℝ) (h1 : x + increase_in_average_commission = (5 * x + C) / n) :
  x + increase_in_average_commission = 250 := by
  sorry

end new_average_commission_is_250_l29_29622


namespace relationship_among_abc_l29_29373

noncomputable def a : ℝ := 4^(1/3 : ℝ)
noncomputable def b : ℝ := Real.log 1/7 / Real.log 3
noncomputable def c : ℝ := (1/3 : ℝ)^(1/5 : ℝ)

theorem relationship_among_abc : a > c ∧ c > b := 
by 
  sorry

end relationship_among_abc_l29_29373


namespace algebraic_expression_value_l29_29639

theorem algebraic_expression_value (m n : ℤ) (h : n - m = 2):
  (m^2 - n^2) / m * (2 * m / (m + n)) = -4 :=
sorry

end algebraic_expression_value_l29_29639


namespace arccos_cos_three_l29_29831

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end arccos_cos_three_l29_29831


namespace elderly_sample_correct_l29_29887

-- Conditions
def young_employees : ℕ := 300
def middle_aged_employees : ℕ := 150
def elderly_employees : ℕ := 100
def total_employees : ℕ := young_employees + middle_aged_employees + elderly_employees
def sample_size : ℕ := 33
def elderly_sample (total : ℕ) (elderly : ℕ) (sample : ℕ) : ℕ := (sample * elderly) / total

-- Statement to prove
theorem elderly_sample_correct :
  elderly_sample total_employees elderly_employees sample_size = 6 := 
by
  sorry

end elderly_sample_correct_l29_29887


namespace largest_number_among_list_l29_29257

theorem largest_number_among_list :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  sorry

end largest_number_among_list_l29_29257


namespace clever_seven_year_count_l29_29041

def isCleverSevenYear (y : Nat) : Bool :=
  let d1 := y / 1000
  let d2 := (y % 1000) / 100
  let d3 := (y % 100) / 10
  let d4 := y % 10
  d1 + d2 + d3 + d4 = 7

theorem clever_seven_year_count : 
  ∃ n, n = 21 ∧ ∀ y, 2000 ≤ y ∧ y ≤ 2999 → isCleverSevenYear y = true ↔ n = 21 :=
by 
  sorry

end clever_seven_year_count_l29_29041


namespace common_root_cubic_polynomials_l29_29325

open Real

theorem common_root_cubic_polynomials (a b c : ℝ)
  (h1 : ∃ α : ℝ, α^3 - a * α^2 + b = 0 ∧ α^3 - b * α^2 + c = 0)
  (h2 : ∃ β : ℝ, β^3 - b * β^2 + c = 0 ∧ β^3 - c * β^2 + a = 0)
  (h3 : ∃ γ : ℝ, γ^3 - c * γ^2 + a = 0 ∧ γ^3 - a * γ^2 + b = 0)
  : a = b ∧ b = c :=
sorry

end common_root_cubic_polynomials_l29_29325


namespace sector_area_maximized_l29_29453

noncomputable def maximize_sector_area (r θ : ℝ) : Prop :=
  2 * r + θ * r = 20 ∧
  (r > 0 ∧ θ > 0) ∧
  ∀ (r' θ' : ℝ), (2 * r' + θ' * r' = 20 ∧ r' > 0 ∧ θ' > 0) → (1/2 * θ' * r'^2 ≤ 1/2 * θ * r^2)

theorem sector_area_maximized : maximize_sector_area 5 2 :=
by
  sorry

end sector_area_maximized_l29_29453


namespace solve_for_m_l29_29801

theorem solve_for_m (m : ℝ) (h : (4 * m + 6) * (2 * m - 5) = 159) : m = 5.3925 :=
sorry

end solve_for_m_l29_29801


namespace number_of_middle_managers_selected_l29_29459

-- Definitions based on conditions
def total_employees := 1000
def senior_managers := 50
def middle_managers := 150
def general_staff := 800
def survey_size := 200

-- Proposition to state the question and correct answer formally
theorem number_of_middle_managers_selected:
  200 * (150 / 1000) = 30 :=
by
  sorry

end number_of_middle_managers_selected_l29_29459


namespace pascal_fifth_number_in_row_15_l29_29940

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l29_29940


namespace graphs_intersection_count_l29_29112

theorem graphs_intersection_count (g : ℝ → ℝ) (hg : Function.Injective g) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (x^3) = g (x^5)) ∧ S.card = 3 :=
by
  sorry

end graphs_intersection_count_l29_29112


namespace find_number_l29_29713

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l29_29713


namespace ribbon_segment_length_l29_29071

theorem ribbon_segment_length :
  ∀ (ribbon_length : ℚ) (segments : ℕ), ribbon_length = 4/5 → segments = 3 → 
  (ribbon_length / segments) = 4/15 :=
by
  intros ribbon_length segments h1 h2
  sorry

end ribbon_segment_length_l29_29071


namespace stratified_sampling_pines_l29_29888

def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

theorem stratified_sampling_pines :
  sample_size * pine_saplings / total_saplings = 20 := by
  sorry

end stratified_sampling_pines_l29_29888


namespace white_squares_in_20th_row_l29_29032

def num_squares_in_row (n : ℕ) : ℕ :=
  3 * n

def num_white_squares (n : ℕ) : ℕ :=
  (num_squares_in_row n - 2) / 2

theorem white_squares_in_20th_row: num_white_squares 20 = 30 := by
  -- Proof skipped
  sorry

end white_squares_in_20th_row_l29_29032


namespace intersection_of_sets_l29_29730

def set_M : Set ℝ := { x : ℝ | (x + 2) * (x - 1) < 0 }
def set_N : Set ℝ := { x : ℝ | x + 1 < 0 }
def intersection (A B : Set ℝ) : Set ℝ := { x : ℝ | x ∈ A ∧ x ∈ B }

theorem intersection_of_sets :
  intersection set_M set_N = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry

end intersection_of_sets_l29_29730


namespace first_representation_second_representation_third_representation_l29_29776

theorem first_representation :
  1 + 2 + 3 + 4 + 5 + 6 + 7 + (8 * 9) = 100 := 
by 
  sorry

theorem second_representation:
  1 + 2 + 3 + 47 + (5 * 6) + 8 + 9 = 100 :=
by
  sorry

theorem third_representation:
  1 + 2 + 3 + 4 + 5 - 6 - 7 + 8 + 92 = 100 := 
by
  sorry

end first_representation_second_representation_third_representation_l29_29776


namespace average_score_l29_29458

theorem average_score (N : ℕ) (p3 p2 p1 p0 : ℕ) (n : ℕ) 
  (H1 : N = 3)
  (H2 : p3 = 30)
  (H3 : p2 = 50)
  (H4 : p1 = 10)
  (H5 : p0 = 10)
  (H6 : n = 20)
  (H7 : p3 + p2 + p1 + p0 = 100) :
  (3 * (p3 * n / 100) + 2 * (p2 * n / 100) + 1 * (p1 * n / 100) + 0 * (p0 * n / 100)) / n = 2 :=
by 
  sorry

end average_score_l29_29458


namespace remainder_when_divided_by_8_l29_29011

theorem remainder_when_divided_by_8 (x : ℤ) (h : ∃ k : ℤ, x = 72 * k + 19) : x % 8 = 3 :=
by
  sorry

end remainder_when_divided_by_8_l29_29011


namespace mirasol_balance_l29_29655

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l29_29655


namespace cole_avg_speed_back_home_l29_29073

noncomputable def avg_speed_back_home 
  (speed_to_work : ℚ) 
  (total_round_trip_time : ℚ) 
  (time_to_work : ℚ) 
  (time_in_minutes : ℚ) :=
  let time_to_work_hours := time_to_work / time_in_minutes
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_back_home := total_round_trip_time - time_to_work_hours
  distance_to_work / time_back_home

theorem cole_avg_speed_back_home :
  avg_speed_back_home 75 1 (35/60) 60 = 105 := 
by 
  -- The proof is omitted
  sorry

end cole_avg_speed_back_home_l29_29073


namespace base8_to_base10_conversion_l29_29471

theorem base8_to_base10_conversion : 
  let n := 432
  let base := 8
  let result := 282
  (2 * base^0 + 3 * base^1 + 4 * base^2) = result := 
by
  let n := 2 * 8^0 + 3 * 8^1 + 4 * 8^2
  have h1 : n = 2 + 24 + 256 := by sorry
  have h2 : 2 + 24 + 256 = 282 := by sorry
  exact Eq.trans h1 h2


end base8_to_base10_conversion_l29_29471


namespace proof_problem_l29_29853

variable {a b : ℝ}

theorem proof_problem (h₁ : a < b) (h₂ : b < 0) : (b/a) + (a/b) > 2 :=
by 
  sorry

end proof_problem_l29_29853


namespace simplify_fraction_l29_29946

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l29_29946


namespace howard_items_l29_29153

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end howard_items_l29_29153


namespace diophantine_eq_unique_solutions_l29_29491

theorem diophantine_eq_unique_solutions (x y : ℕ) (hx_positive : x > 0) (hy_positive : y > 0) :
  x^y = y^x + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end diophantine_eq_unique_solutions_l29_29491


namespace intersection_A_B_l29_29941

-- Defining set A condition
def A : Set ℝ := {x | x - 1 < 2}

-- Defining set B condition
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- The goal to prove
theorem intersection_A_B : {x | x > 0 ∧ x < 3} = (A ∩ { x | 0 < x ∧ x < 8 }) :=
by
  sorry

end intersection_A_B_l29_29941


namespace probability_of_four_twos_in_five_rolls_l29_29253

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end probability_of_four_twos_in_five_rolls_l29_29253


namespace seq_2011_l29_29759

-- Definition of the sequence
def seq (a : ℕ → ℤ) := (a 1 = a 201) ∧ a 201 = 2 ∧ ∀ n : ℕ, a n + a (n + 1) = 0

-- The main theorem to prove that a_2011 = 2
theorem seq_2011 : ∀ a : ℕ → ℤ, seq a → a 2011 = 2 :=
by
  intros a h
  let seq := h
  sorry

end seq_2011_l29_29759


namespace find_number_l29_29644

noncomputable def solve_N (x : ℝ) (N : ℝ) : Prop :=
  ((N / x) / (3.6 * 0.2) = 2)

theorem find_number (x : ℝ) (N : ℝ) (h1 : x = 12) (h2 : solve_N x N) : N = 17.28 :=
  by
  sorry

end find_number_l29_29644


namespace problem_statement_l29_29263

variable {x y : ℝ}

theorem problem_statement (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : y - 2 / x ≠ 0) :
  (2 * x - 3 / y) / (3 * y - 2 / x) = (2 * x * y - 3) / (3 * x * y - 2) :=
sorry

end problem_statement_l29_29263


namespace probability_of_specific_combination_l29_29701

def count_all_clothes : ℕ := 6 + 7 + 8 + 3
def choose4_out_of_24 : ℕ := Nat.choose 24 4
def choose1_shirt : ℕ := 6
def choose1_pair_shorts : ℕ := 7
def choose1_pair_socks : ℕ := 8
def choose1_hat : ℕ := 3
def favorable_outcomes : ℕ := choose1_shirt * choose1_pair_shorts * choose1_pair_socks * choose1_hat
def probability_of_combination : ℚ := favorable_outcomes / choose4_out_of_24

theorem probability_of_specific_combination :
  probability_of_combination = 144 / 1815 := by
sorry

end probability_of_specific_combination_l29_29701


namespace trigonometric_identity_l29_29201

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (3 * Real.cos x - Real.sin x) = 3 := 
by
  sorry

end trigonometric_identity_l29_29201


namespace min_value_proof_l29_29867

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end min_value_proof_l29_29867


namespace compute_expression_l29_29227

theorem compute_expression :
  (1 / 36) / ((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) + 
  (((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) / (1 / 36)) = -10 / 3 :=
by
  sorry

end compute_expression_l29_29227


namespace division_of_decimals_l29_29954

theorem division_of_decimals : (0.5 : ℝ) / (0.025 : ℝ) = 20 := 
sorry

end division_of_decimals_l29_29954


namespace pauline_bought_2_pounds_of_meat_l29_29462

theorem pauline_bought_2_pounds_of_meat :
  ∀ (cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent : ℝ) 
    (num_bell_peppers : ℕ),
  cost_taco_shells = 5 →
  cost_bell_pepper = 1.5 →
  cost_meat_per_pound = 3 →
  total_spent = 17 →
  num_bell_peppers = 4 →
  (total_spent - (cost_taco_shells + (num_bell_peppers * cost_bell_pepper))) / cost_meat_per_pound = 2 :=
by
  intros cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent num_bell_peppers 
         h1 h2 h3 h4 h5
  sorry

end pauline_bought_2_pounds_of_meat_l29_29462


namespace arithmetic_sequence_m_value_l29_29181

theorem arithmetic_sequence_m_value (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) 
  (h_seq : ∀ n : ℕ, S n = (n + 1) / 2 * (2 * a₁ + n * d)) :
  m = 5 :=
by
  sorry

end arithmetic_sequence_m_value_l29_29181


namespace two_digit_ab_divisible_by_11_13_l29_29065

theorem two_digit_ab_divisible_by_11_13 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 11 = 0)
  (h4 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 13 = 0) :
  10 * a + b = 48 :=
sorry

end two_digit_ab_divisible_by_11_13_l29_29065


namespace sixth_root_of_unity_l29_29451

/- Constants and Variables -/
variable (p q r s t k : ℂ)
variable (nz_p : p ≠ 0) (nz_q : q ≠ 0) (nz_r : r ≠ 0) (nz_s : s ≠ 0) (nz_t : t ≠ 0)
variable (hk1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
variable (hk2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0)

/- Theorem to prove -/
theorem sixth_root_of_unity : k^6 = 1 :=
by sorry

end sixth_root_of_unity_l29_29451


namespace cistern_wet_surface_area_l29_29270

def cistern (length : ℕ) (width : ℕ) (water_height : ℝ) : ℝ :=
  (length * width : ℝ) + 2 * (water_height * length) + 2 * (water_height * width)

theorem cistern_wet_surface_area :
  cistern 7 5 1.40 = 68.6 :=
by
  sorry

end cistern_wet_surface_area_l29_29270


namespace gecko_insects_eaten_l29_29210

theorem gecko_insects_eaten
    (G : ℕ)  -- Number of insects each gecko eats
    (H1 : 5 * G + 3 * (2 * G) = 66) :  -- Total insects eaten condition
    G = 6 :=  -- Expected number of insects each gecko eats
by
  sorry

end gecko_insects_eaten_l29_29210


namespace min_x_y_l29_29274

theorem min_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 2) :
  x + y ≥ 9 / 2 := 
by 
  sorry

end min_x_y_l29_29274


namespace restore_arithmetic_operations_l29_29478

/--
Given the placeholders \(A, B, C, D, E\) for operations in the equations:
1. \(4 A 2 = 2\)
2. \(8 = 4 C 2\)
3. \(2 D 3 = 5\)
4. \(4 = 5 E 1\)

Prove that:
(a) \(A = ÷\)
(b) \(B = =\)
(c) \(C = ×\)
(d) \(D = +\)
(e) \(E = -\)
-/
theorem restore_arithmetic_operations {A B C D E : String} (h1 : B = "=") 
    (h2 : "4" ++ A  ++ "2" ++ B ++ "2" = "4" ++ "÷" ++ "2" ++ "=" ++ "2")
    (h3 : "8" ++ "=" ++ "4" ++ C ++ "2" = "8" ++ "=" ++ "4" ++ "×" ++ "2")
    (h4 : "2" ++ D ++ "3" ++ "=" ++ "5" = "2" ++ "+" ++ "3" ++ "=" ++ "5")
    (h5 : "4" ++ "=" ++ "5" ++ E ++ "1" = "4" ++ "=" ++ "5" ++ "-" ++ "1") :
  (A = "÷") ∧ (B = "=") ∧ (C = "×") ∧ (D = "+") ∧ (E = "-") := by
    sorry

end restore_arithmetic_operations_l29_29478


namespace vector_dot_product_parallel_l29_29323

theorem vector_dot_product_parallel (m : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (m, -4))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (a.1 * b.1 + a.2 * b.2) = -10 := by
  sorry

end vector_dot_product_parallel_l29_29323


namespace chocolate_truffles_sold_l29_29034

def fudge_sold_pounds : ℕ := 20
def price_per_pound_fudge : ℝ := 2.50
def price_per_truffle : ℝ := 1.50
def pretzels_sold_dozen : ℕ := 3
def price_per_pretzel : ℝ := 2.00
def total_revenue : ℝ := 212.00

theorem chocolate_truffles_sold (dozens_of_truffles_sold : ℕ) :
  let fudge_revenue := (fudge_sold_pounds : ℝ) * price_per_pound_fudge
  let pretzels_revenue := (pretzels_sold_dozen : ℝ) * 12 * price_per_pretzel
  let truffles_revenue := total_revenue - fudge_revenue - pretzels_revenue
  let num_truffles_sold := truffles_revenue / price_per_truffle
  let dozens_of_truffles_sold := num_truffles_sold / 12
  dozens_of_truffles_sold = 5 :=
by
  sorry

end chocolate_truffles_sold_l29_29034


namespace symmetric_angles_y_axis_l29_29680

theorem symmetric_angles_y_axis (α β : ℝ) (k : ℤ)
  (h : ∃ k : ℤ, β = 2 * k * π + (π - α)) :
  α + β = (2 * k + 1) * π ∨ α = -β + (2 * k + 1) * π :=
by sorry

end symmetric_angles_y_axis_l29_29680


namespace value_of_w_over_y_l29_29200

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3.25) : w / y = 0.75 :=
sorry

end value_of_w_over_y_l29_29200


namespace probability_x_lt_y_l29_29627

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l29_29627


namespace range_f_pos_l29_29835

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y
axiom f_at_neg_one : f (-1) = 0

theorem range_f_pos : {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := 
by
  sorry

end range_f_pos_l29_29835


namespace divide_number_l29_29418

theorem divide_number (x : ℝ) (h : 0.3 * x = 0.2 * (80 - x) + 10) : min x (80 - x) = 28 := 
by 
  sorry

end divide_number_l29_29418


namespace arithmetic_seq_sum_mod_9_l29_29301

def sum_arithmetic_seq := 88230 + 88231 + 88232 + 88233 + 88234 + 88235 + 88236 + 88237 + 88238 + 88239 + 88240

theorem arithmetic_seq_sum_mod_9 : 
  sum_arithmetic_seq % 9 = 0 :=
by
-- proof will be provided here
sorry

end arithmetic_seq_sum_mod_9_l29_29301


namespace arithmetic_seq_and_general_formula_find_Tn_l29_29779

-- Given definitions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ := sorry

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → n * S n.succ = (n+1) * S n + n^2 + n

-- Problem 1: Prove and derive general formula for Sₙ
theorem arithmetic_seq_and_general_formula (n : ℕ) (h : n > 0) :
  ∃ S : ℕ → ℕ, (∀ n : ℕ, n > 0 → (S (n+1)) / (n+1) - (S n) / n = 1) ∧ (S n = n^2) := sorry

-- Problem 2: Given bₙ and Tₙ, find Tₙ
def b (n : ℕ) : ℕ := 1 / (a n * a (n+1))
def T : ℕ → ℕ := sorry

axiom b1 : ∀ n : ℕ, n > 0 → b 1 = 1
axiom b2 : ∀ n : ℕ, n > 0 → T n = 1 / (2 * n + 1)

theorem find_Tn (n : ℕ) (h : n > 0) : T n = n / (2 * n + 1) := sorry

end arithmetic_seq_and_general_formula_find_Tn_l29_29779


namespace balls_in_boxes_l29_29668

theorem balls_in_boxes :
  ∃ (f : Fin 5 → Fin 3), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ b : Fin 3, ∃ i, f i = b) ∧
    f 0 ≠ f 1 :=
  sorry

end balls_in_boxes_l29_29668


namespace find_x_l29_29160

theorem find_x (x y : ℝ) (h₁ : x - y = 10) (h₂ : x + y = 14) : x = 12 :=
by
  sorry

end find_x_l29_29160


namespace desks_built_by_carpenters_l29_29500

theorem desks_built_by_carpenters (h : 2 * 2.5 * r ≥ 2 * r) : 4 * 5 * r ≥ 8 * r :=
by
  sorry

end desks_built_by_carpenters_l29_29500


namespace total_money_l29_29128

variable (Sally Jolly Molly : ℕ)

-- Conditions
def condition1 (Sally : ℕ) : Prop := Sally - 20 = 80
def condition2 (Jolly : ℕ) : Prop := Jolly + 20 = 70
def condition3 (Molly : ℕ) : Prop := Molly + 30 = 100

-- The theorem to prove
theorem total_money (h1: condition1 Sally)
                    (h2: condition2 Jolly)
                    (h3: condition3 Molly) :
  Sally + Jolly + Molly = 220 :=
by
  sorry

end total_money_l29_29128


namespace number_of_people_l29_29340

def totalCups : ℕ := 10
def cupsPerPerson : ℕ := 2

theorem number_of_people {n : ℕ} (h : n = totalCups / cupsPerPerson) : n = 5 := by
  sorry

end number_of_people_l29_29340


namespace bobby_initial_candy_l29_29209

theorem bobby_initial_candy (initial_candy : ℕ) (remaining_candy : ℕ) (extra_candy : ℕ) (total_eaten : ℕ)
  (h_candy_initial : initial_candy = 36)
  (h_candy_remaining : remaining_candy = 4)
  (h_candy_extra : extra_candy = 15)
  (h_candy_total_eaten : total_eaten = initial_candy - remaining_candy) :
  total_eaten - extra_candy = 17 :=
by
  sorry

end bobby_initial_candy_l29_29209


namespace sum_mod_13_l29_29729

theorem sum_mod_13 (a b c d e : ℤ) (ha : a % 13 = 3) (hb : b % 13 = 5) (hc : c % 13 = 7) (hd : d % 13 = 9) (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by
  -- The proof can be constructed here
  sorry

end sum_mod_13_l29_29729


namespace ball_reaches_top_left_pocket_l29_29681

-- Definitions based on the given problem
def table_width : ℕ := 26
def table_height : ℕ := 1965
def pocket_start : (ℕ × ℕ) := (0, 0)
def pocket_end : (ℕ × ℕ) := (0, table_height)
def angle_of_release : ℝ := 45

-- The goal is to prove that the ball will reach the top left pocket after reflections
theorem ball_reaches_top_left_pocket :
  ∃ reflections : ℕ, (reflections * table_width, reflections * table_height) = pocket_end :=
sorry

end ball_reaches_top_left_pocket_l29_29681


namespace max_ab_bc_cd_l29_29400

theorem max_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
    ab + bc + cd ≤ 10000 := by
  sorry

end max_ab_bc_cd_l29_29400
