import Mathlib

namespace original_price_of_coat_l2327_232753

theorem original_price_of_coat (P : ℝ) (h : 0.40 * P = 200) : P = 500 :=
by {
  sorry
}

end original_price_of_coat_l2327_232753


namespace seventh_term_of_geometric_sequence_l2327_232773

theorem seventh_term_of_geometric_sequence (r : ℝ) 
  (h1 : 3 * r^5 = 729) : 3 * r^6 = 2187 :=
sorry

end seventh_term_of_geometric_sequence_l2327_232773


namespace equation_solution_system_of_inequalities_solution_l2327_232766

theorem equation_solution (x : ℝ) : (3 / (x - 1) = 1 / (2 * x + 3)) ↔ (x = -2) :=
by
  sorry

theorem system_of_inequalities_solution (x : ℝ) : ((3 * x - 1 ≥ x + 1) ∧ (x + 3 > 4 * x - 2)) ↔ (1 ≤ x ∧ x < 5 / 3) :=
by
  sorry

end equation_solution_system_of_inequalities_solution_l2327_232766


namespace derivative_of_f_is_l2327_232739

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2

theorem derivative_of_f_is (x : ℝ) : deriv f x = 2 * x + 2 :=
by
  sorry

end derivative_of_f_is_l2327_232739


namespace cat_mouse_position_after_moves_l2327_232791

-- Define the total number of moves
def total_moves : ℕ := 360

-- Define cat's cycle length and position calculation
def cat_cycle_length : ℕ := 5
def cat_final_position := total_moves % cat_cycle_length

-- Define mouse's cycle length and actual moves per cycle
def mouse_cycle_length : ℕ := 10
def mouse_effective_moves_per_cycle : ℕ := 9
def total_mouse_effective_moves := (total_moves / mouse_cycle_length) * mouse_effective_moves_per_cycle
def mouse_final_position := total_mouse_effective_moves % mouse_cycle_length

theorem cat_mouse_position_after_moves :
  cat_final_position = 0 ∧ mouse_final_position = 4 :=
by
  sorry

end cat_mouse_position_after_moves_l2327_232791


namespace prob_rain_at_least_one_day_l2327_232708

noncomputable def prob_rain_saturday := 0.35
noncomputable def prob_rain_sunday := 0.45

theorem prob_rain_at_least_one_day : 
  (1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)) * 100 = 64.25 := 
by 
  sorry

end prob_rain_at_least_one_day_l2327_232708


namespace rabbit_shape_area_l2327_232768

theorem rabbit_shape_area (A_ear : ℝ) (h1 : A_ear = 10) (h2 : A_ear = (1/8) * A_total) :
  A_total = 80 :=
by
  sorry

end rabbit_shape_area_l2327_232768


namespace move_line_down_l2327_232752

theorem move_line_down (x y : ℝ) : (y = -3 * x + 5) → (y = -3 * x + 2) :=
by
  sorry

end move_line_down_l2327_232752


namespace percent_shaded_of_square_l2327_232756

theorem percent_shaded_of_square (side_len : ℤ) (first_layer_side : ℤ) 
(second_layer_outer_side : ℤ) (second_layer_inner_side : ℤ)
(third_layer_outer_side : ℤ) (third_layer_inner_side : ℤ)
(h_side : side_len = 7) (h_first : first_layer_side = 2) 
(h_second_outer : second_layer_outer_side = 5) (h_second_inner : second_layer_inner_side = 3) 
(h_third_outer : third_layer_outer_side = 7) (h_third_inner : third_layer_inner_side = 6) : 
  (4 + (25 - 9) + (49 - 36)) / (side_len * side_len : ℝ) = 33 / 49 :=
by
  -- Sorry is used as we are only required to construct the statement, not the proof.
  sorry

end percent_shaded_of_square_l2327_232756


namespace solve_range_m_l2327_232725

variable (m : ℝ)
def p := m < 0
def q := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem solve_range_m (hpq : p m ∧ q m) : -2 < m ∧ m < 0 := 
  sorry

end solve_range_m_l2327_232725


namespace roots_quadratic_expr_l2327_232715

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end roots_quadratic_expr_l2327_232715


namespace range_of_sum_l2327_232798

theorem range_of_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b + 1 / a + 9 / b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 :=
sorry

end range_of_sum_l2327_232798


namespace extreme_value_sum_l2327_232784

noncomputable def f (m n x : ℝ) : ℝ := x^3 + 3 * m * x^2 + n * x + m^2

theorem extreme_value_sum (m n : ℝ) (h1 : f m n (-1) = 0) (h2 : (deriv (f m n)) (-1) = 0) : m + n = 11 := 
sorry

end extreme_value_sum_l2327_232784


namespace solution_set_to_coeff_properties_l2327_232764

theorem solution_set_to_coeff_properties 
  (a b c : ℝ) 
  (h : ∀ x, (2 < x ∧ x < 3) → ax^2 + bx + c > 0) 
  : 
  (a < 0) 
  ∧ (b * c < 0) 
  ∧ (b + c = a) :=
sorry

end solution_set_to_coeff_properties_l2327_232764


namespace courses_selection_l2327_232776

-- Definition of the problem
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways person A can choose 2 courses from 4
def total_ways : ℕ := C 4 2 * C 4 2

-- Number of ways both choose exactly the same courses
def same_ways : ℕ := C 4 2

-- Prove the number of ways they can choose such that there is at least one course different
theorem courses_selection :
  total_ways - same_ways = 30 := by
  sorry

end courses_selection_l2327_232776


namespace percent_non_condiments_l2327_232718

def sandwich_weight : ℕ := 150
def condiment_weight : ℕ := 45
def non_condiment_weight (total: ℕ) (condiments: ℕ) : ℕ := total - condiments
def percentage (num denom: ℕ) : ℕ := (num * 100) / denom

theorem percent_non_condiments : 
  percentage (non_condiment_weight sandwich_weight condiment_weight) sandwich_weight = 70 :=
by
  sorry

end percent_non_condiments_l2327_232718


namespace area_under_pressure_l2327_232737

theorem area_under_pressure (F : ℝ) (S : ℝ) (p : ℝ) (hF : F = 100) (hp : p > 1000) (hpressure : p = F / S) :
  S < 0.1 :=
by
  sorry

end area_under_pressure_l2327_232737


namespace slope_of_line_l2327_232742

theorem slope_of_line (θ : ℝ) (h : θ = 30) :
  ∃ k, k = Real.tan (60 * (π / 180)) ∨ k = Real.tan (120 * (π / 180)) := by
    sorry

end slope_of_line_l2327_232742


namespace total_amount_received_correct_l2327_232770

variable (total_won : ℝ) (fraction : ℝ) (students : ℕ)
variable (portion_per_student : ℝ := total_won * fraction)
variable (total_given : ℝ := portion_per_student * students)

theorem total_amount_received_correct :
  total_won = 555850 →
  fraction = 3 / 10000 →
  students = 500 →
  total_given = 833775 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_amount_received_correct_l2327_232770


namespace exists_four_distinct_numbers_with_equal_half_sum_l2327_232796

theorem exists_four_distinct_numbers_with_equal_half_sum (S : Finset ℕ) (h_card : S.card = 10) (h_range : ∀ x ∈ S, x ≤ 23) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a + b = c + d) :=
by
  sorry

end exists_four_distinct_numbers_with_equal_half_sum_l2327_232796


namespace more_people_joined_l2327_232748

def initial_people : Nat := 61
def final_people : Nat := 83

theorem more_people_joined :
  final_people - initial_people = 22 := by
  sorry

end more_people_joined_l2327_232748


namespace molecular_weight_of_NH4I_l2327_232723

-- Define the conditions in Lean
def molecular_weight (moles grams: ℕ) : Prop :=
  grams / moles = 145

-- Statement of the proof problem
theorem molecular_weight_of_NH4I :
  molecular_weight 9 1305 :=
by
  -- Proof is omitted 
  sorry

end molecular_weight_of_NH4I_l2327_232723


namespace geometric_sequence_quadratic_roots_l2327_232711

theorem geometric_sequence_quadratic_roots
    (a b : ℝ)
    (h_geometric : ∃ q : ℝ, b = 2 * q ∧ a = 2 * q^2) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + (1 / 3) = 0 ∧ a * x2^2 + b * x2 + (1 / 3) = 0) :=
by
  sorry

end geometric_sequence_quadratic_roots_l2327_232711


namespace problem_statement_l2327_232702

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 2)
    (h3 : 0 < b) (h4 : b < 2) (h5 : 0 < c) (h6 : c < 2) :
    ¬ ((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) :=
by
  sorry

end problem_statement_l2327_232702


namespace quiz_passing_condition_l2327_232727

theorem quiz_passing_condition (P Q : Prop) :
  (Q → P) → 
    (¬P → ¬Q) ∧ 
    (¬Q → ¬P) ∧ 
    (P → Q) :=
by sorry

end quiz_passing_condition_l2327_232727


namespace fish_to_rice_equivalence_l2327_232762

variable (f : ℚ) (l : ℚ)

theorem fish_to_rice_equivalence (h1 : 5 * f = 3 * l) (h2 : l = 6) : f = 18 / 5 := by
  sorry

end fish_to_rice_equivalence_l2327_232762


namespace votes_cast_46800_l2327_232757

-- Define the election context
noncomputable def total_votes (v : ℕ) : Prop :=
  let percentage_a := 0.35
  let percentage_b := 0.40
  let vote_diff := 2340
  (percentage_b - percentage_a) * (v : ℝ) = (vote_diff : ℝ)

-- Theorem stating the total number of votes cast in the election
theorem votes_cast_46800 : total_votes 46800 :=
by
  sorry

end votes_cast_46800_l2327_232757


namespace length_of_square_side_l2327_232731

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem length_of_square_side
  (time_seconds : ℝ)
  (speed_km_per_hr : ℝ)
  (distance_m : ℝ)
  (side_length : ℝ)
  (h1 : time_seconds = 72)
  (h2 : speed_km_per_hr = 10)
  (h3 : distance_m = speed_km_per_hr_to_m_per_s speed_km_per_hr * time_seconds)
  (h4 : distance_m = perimeter_of_square side_length) :
  side_length = 50 :=
sorry

end length_of_square_side_l2327_232731


namespace polynomial_root_condition_l2327_232755

theorem polynomial_root_condition (a : ℝ) :
  (∃ x1 x2 x3 : ℝ,
    (x1^3 - 6 * x1^2 + a * x1 + a = 0) ∧
    (x2^3 - 6 * x2^2 + a * x2 + a = 0) ∧
    (x3^3 - 6 * x3^2 + a * x3 + a = 0) ∧
    ((x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0)) →
  a = -9 :=
by
  sorry

end polynomial_root_condition_l2327_232755


namespace inequality_proof_l2327_232781

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l2327_232781


namespace number_of_workers_in_second_group_l2327_232728

theorem number_of_workers_in_second_group (w₁ w₂ d₁ d₂ : ℕ) (total_wages₁ total_wages₂ : ℝ) (daily_wage : ℝ) :
  w₁ = 15 ∧ d₁ = 6 ∧ total_wages₁ = 9450 ∧ 
  w₂ * d₂ * daily_wage = total_wages₂ ∧ d₂ = 5 ∧ total_wages₂ = 9975 ∧ 
  daily_wage = 105 
  → w₂ = 19 :=
by
  sorry

end number_of_workers_in_second_group_l2327_232728


namespace express_B_using_roster_l2327_232769

open Set

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem express_B_using_roster :
  B = {4, 9, 16} := by
  sorry

end express_B_using_roster_l2327_232769


namespace time_per_flash_l2327_232750

def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60
def light_flashes_in_three_fourths_hour : ℕ := 180

-- Converting ¾ of an hour to minutes and then to seconds
def seconds_in_three_fourths_hour : ℕ := (3 * minutes_per_hour / 4) * seconds_per_minute

-- Proving that the time taken for one flash is 15 seconds
theorem time_per_flash : (seconds_in_three_fourths_hour / light_flashes_in_three_fourths_hour) = 15 :=
by
  sorry

end time_per_flash_l2327_232750


namespace find_value_of_f_neg_3_over_2_l2327_232749

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_f_neg_3_over_2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
    (h2 : ∀ x : ℝ, f (x + 3/2) = -f x) : 
    f (- 3 / 2) = 0 := 
sorry

end find_value_of_f_neg_3_over_2_l2327_232749


namespace true_weight_third_object_proof_l2327_232786

noncomputable def true_weight_third_object (A a B b C : ℝ) : ℝ :=
  let h := Real.sqrt ((a - b) / (A - B))
  let k := (b * A - a * B) / ((A - B) * (h + 1))
  h * C + k

theorem true_weight_third_object_proof (A a B b C : ℝ) (h := Real.sqrt ((a - b) / (A - B))) (k := (b * A - a * B) / ((A - B) * (h + 1))) :
  true_weight_third_object A a B b C = h * C + k := by
  sorry

end true_weight_third_object_proof_l2327_232786


namespace candy_pieces_per_pile_l2327_232712

theorem candy_pieces_per_pile :
  ∀ (total_candies eaten_candies num_piles pieces_per_pile : ℕ),
    total_candies = 108 →
    eaten_candies = 36 →
    num_piles = 8 →
    pieces_per_pile = (total_candies - eaten_candies) / num_piles →
    pieces_per_pile = 9 :=
by
  intros total_candies eaten_candies num_piles pieces_per_pile
  sorry

end candy_pieces_per_pile_l2327_232712


namespace scott_sold_40_cups_of_smoothies_l2327_232714

theorem scott_sold_40_cups_of_smoothies
  (cost_smoothie : ℕ)
  (cost_cake : ℕ)
  (num_cakes : ℕ)
  (total_revenue : ℕ)
  (h1 : cost_smoothie = 3)
  (h2 : cost_cake = 2)
  (h3 : num_cakes = 18)
  (h4 : total_revenue = 156) :
  ∃ x : ℕ, (cost_smoothie * x + cost_cake * num_cakes = total_revenue ∧ x = 40) := 
sorry

end scott_sold_40_cups_of_smoothies_l2327_232714


namespace find_angle_l2327_232779

theorem find_angle :
  ∃ (x : ℝ), (90 - x = 0.4 * (180 - x)) → x = 30 :=
by
  sorry

end find_angle_l2327_232779


namespace smallest_root_of_g_l2327_232765

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- The main statement: proving the smallest root of g(x) is -sqrt(7/5)
theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y := 
sorry

end smallest_root_of_g_l2327_232765


namespace symmetric_point_coordinates_l2327_232771

structure Point : Type where
  x : ℝ
  y : ℝ

def symmetric_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def P : Point := { x := -10, y := -1 }

def P1 : Point := symmetric_y P

def P2 : Point := symmetric_x P1

theorem symmetric_point_coordinates :
  P2 = { x := 10, y := 1 } := by
  sorry

end symmetric_point_coordinates_l2327_232771


namespace equal_real_roots_of_quadratic_eq_l2327_232759

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l2327_232759


namespace total_area_correct_at_stage_5_l2327_232701

def initial_side_length := 3

def side_length (n : ℕ) : ℕ := initial_side_length + n

def area (side : ℕ) : ℕ := side * side

noncomputable def total_area_at_stage_5 : ℕ :=
  (area (side_length 0)) + (area (side_length 1)) + (area (side_length 2)) + (area (side_length 3)) + (area (side_length 4))

theorem total_area_correct_at_stage_5 : total_area_at_stage_5 = 135 :=
by
  sorry

end total_area_correct_at_stage_5_l2327_232701


namespace kids_went_home_l2327_232743

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℝ) (went_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : remaining_kids = 8.0) : went_home = 14.0 :=
by 
  sorry

end kids_went_home_l2327_232743


namespace intersection_of_M_and_N_l2327_232703

-- Define sets M and N
def M : Set ℕ := {0, 2, 3, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

-- State the problem as a theorem
theorem intersection_of_M_and_N : (M ∩ N) = {0, 4} :=
by
    sorry

end intersection_of_M_and_N_l2327_232703


namespace systematic_sampling_second_invoice_l2327_232740

theorem systematic_sampling_second_invoice 
  (N : ℕ) 
  (valid_invoice : N ≥ 10)
  (first_invoice : Fin 10) :
  ¬ (∃ k : ℕ, k ≥ 1 ∧ first_invoice.1 + k * 10 = 23) := 
by 
  -- Proof omitted
  sorry

end systematic_sampling_second_invoice_l2327_232740


namespace problem_l2327_232729

theorem problem (a b c d : ℝ) (h1 : b + c = 7) (h2 : c + d = 5) (h3 : a + d = 2) : a + b = 4 :=
sorry

end problem_l2327_232729


namespace Moe_has_least_amount_of_money_l2327_232775

variables (Money : Type) [LinearOrder Money]
variables (Bo Coe Flo Jo Moe Zoe : Money)
variables (Bo_lt_Flo : Bo < Flo) (Jo_lt_Flo : Jo < Flo)
variables (Moe_lt_Bo : Moe < Bo) (Moe_lt_Coe : Moe < Coe)
variables (Moe_lt_Jo : Moe < Jo) (Jo_lt_Bo : Jo < Bo)
variables (Moe_lt_Zoe : Moe < Zoe) (Zoe_lt_Jo : Zoe < Jo)

theorem Moe_has_least_amount_of_money : ∀ x, x ≠ Moe → Moe < x := by
  sorry

end Moe_has_least_amount_of_money_l2327_232775


namespace max_point_f_l2327_232747

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end max_point_f_l2327_232747


namespace sin_vertex_angle_isosceles_triangle_l2327_232777

theorem sin_vertex_angle_isosceles_triangle (α β : ℝ) (h_isosceles : β = 2 * α) (tan_base_angle : Real.tan α = 2 / 3) :
  Real.sin β = 12 / 13 := 
sorry

end sin_vertex_angle_isosceles_triangle_l2327_232777


namespace regular_pyramid_sufficient_condition_l2327_232793

-- Define the basic structure of a pyramid
structure Pyramid :=
  (lateral_face_is_equilateral_triangle : Prop)  
  (base_is_square : Prop)  
  (apex_angles_of_lateral_face_are_45_deg : Prop)  
  (projection_of_vertex_at_intersection_of_base_diagonals : Prop)
  (is_regular : Prop)

-- Define the hypothesis conditions
variables 
  (P : Pyramid)
  (h1 : P.lateral_face_is_equilateral_triangle)
  (h2 : P.base_is_square)
  (h3 : P.apex_angles_of_lateral_face_are_45_deg)
  (h4 : P.projection_of_vertex_at_intersection_of_base_diagonals)

-- Define the statement of the proof
theorem regular_pyramid_sufficient_condition :
  (P.lateral_face_is_equilateral_triangle → P.is_regular) ∧ 
  (¬(P.lateral_face_is_equilateral_triangle) → ¬P.is_regular) ↔
  (P.lateral_face_is_equilateral_triangle ∧ ¬P.base_is_square ∧ ¬P.apex_angles_of_lateral_face_are_45_deg ∧ ¬P.projection_of_vertex_at_intersection_of_base_diagonals) := 
by { sorry }


end regular_pyramid_sufficient_condition_l2327_232793


namespace pq_eq_real_nums_l2327_232780

theorem pq_eq_real_nums (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := 
by 
  sorry

end pq_eq_real_nums_l2327_232780


namespace percentage_reduction_is_correct_l2327_232790

-- Definitions and initial conditions
def initial_price_per_model := 100
def models_for_kindergarten := 2
def models_for_elementary := 2 * models_for_kindergarten
def total_models := models_for_kindergarten + models_for_elementary
def total_cost_without_reduction := total_models * initial_price_per_model
def total_cost_paid := 570

-- Goal statement in Lean 4
theorem percentage_reduction_is_correct :
  (total_models > 5) →
  total_cost_paid = 570 →
  models_for_kindergarten = 2 →
  (total_cost_without_reduction - total_cost_paid) / total_models / initial_price_per_model * 100 = 5 :=
by
  -- sorry to skip the proof
  sorry

end percentage_reduction_is_correct_l2327_232790


namespace amount_lent_by_A_to_B_l2327_232782

theorem amount_lent_by_A_to_B
  (P : ℝ)
  (H1 : P * 0.115 * 3 - P * 0.10 * 3 = 1125) :
  P = 25000 :=
by
  sorry

end amount_lent_by_A_to_B_l2327_232782


namespace option_c_correct_l2327_232705

theorem option_c_correct (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y :=
by {
  sorry
}

end option_c_correct_l2327_232705


namespace find_y_interval_l2327_232799

theorem find_y_interval (y : ℝ) (h : y^2 - 8 * y + 12 < 0) : 2 < y ∧ y < 6 :=
sorry

end find_y_interval_l2327_232799


namespace line_through_origin_tangent_lines_line_through_tangents_l2327_232741

section GeomProblem

variables {A : ℝ × ℝ} {C : ℝ × ℝ → Prop}

def is_circle (C : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∀ (P : ℝ × ℝ), C P ↔ (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

theorem line_through_origin (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ m : ℝ, ∀ P : ℝ × ℝ, C P → abs ((m * P.1 - P.2) / Real.sqrt (m ^ 2 + 1)) = 1)
    ↔ m = 0 :=
sorry

theorem tangent_lines (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P : ℝ × ℝ, C P → (P.2 - 2 * Real.sqrt 3) = k * (P.1 - 1))
    ↔ (∀ P : ℝ × ℝ, C P → (Real.sqrt 3 * P.1 - 3 * P.2 + 5 * Real.sqrt 3 = 0 ∨ P.1 = 1)) :=
sorry

theorem line_through_tangents (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P D E : ℝ × ℝ, C P → (Real.sqrt 3 * D.1 - 3 * D.2 + 5 * Real.sqrt 3 = 0 ∧
                                      (E.1 - 1 = 0 ∨ Real.sqrt 3 * E.1 - 3 * E.2 + 5 * Real.sqrt 3 = 0)) →
    (D.1 + Real.sqrt 3 * D.2 - 1 = 0 ∧ E.1 + Real.sqrt 3 * E.2 - 1 = 0)) :=
sorry

end GeomProblem

end line_through_origin_tangent_lines_line_through_tangents_l2327_232741


namespace integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l2327_232744

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l2327_232744


namespace not_divisible_by_n_plus_4_l2327_232788

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 8*n + 15 = k * (n + 4) :=
sorry

end not_divisible_by_n_plus_4_l2327_232788


namespace mathematics_equivalent_proof_l2327_232733

noncomputable def distinctRealNumbers (a b c d : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d

theorem mathematics_equivalent_proof (a b c d : ℝ)
  (H₀ : distinctRealNumbers a b c d)
  (H₁ : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 :=
sorry

end mathematics_equivalent_proof_l2327_232733


namespace sasha_sequence_eventually_five_to_100_l2327_232758

theorem sasha_sequence_eventually_five_to_100 :
  ∃ (n : ℕ), 
  (5 ^ 100) = initial_value + n * (3 ^ 100) - m * (2 ^ 100) ∧ 
  (initial_value + n * (3 ^ 100) - m * (2 ^ 100) > 0) :=
by
  let initial_value := 1
  let threshold := 2 ^ 100
  let increment := 3 ^ 100
  let decrement := 2 ^ 100
  sorry

end sasha_sequence_eventually_five_to_100_l2327_232758


namespace range_of_x_l2327_232730

noncomputable def f (x : ℝ) : ℝ := (5 / (x^2)) - (3 * (x^2)) + 2

theorem range_of_x :
  { x : ℝ | f 1 < f (Real.log x / Real.log 3) } = { x : ℝ | (1 / 3) < x ∧ x < 1 ∨ 1 < x ∧ x < 3 } :=
by
  sorry

end range_of_x_l2327_232730


namespace quadratic_ineq_solution_set_l2327_232785

theorem quadratic_ineq_solution_set (a b c : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 3 < x → x < 6 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, x < (1 / 6) ∨ x > (1 / 3) → cx^2 + bx + a < 0 := by 
  sorry

end quadratic_ineq_solution_set_l2327_232785


namespace sum_of_number_and_reverse_l2327_232783

theorem sum_of_number_and_reverse (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
(h : (10 * a + b) - (10 * b + a) = 3 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 33 := 
sorry

end sum_of_number_and_reverse_l2327_232783


namespace sqrt_meaningful_condition_l2327_232732

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l2327_232732


namespace inverse_variation_example_l2327_232726

theorem inverse_variation_example
  (k : ℝ)
  (h1 : ∀ (c d : ℝ), (c^2) * (d^4) = k)
  (h2 : ∃ (c : ℝ), c = 8 ∧ (∀ (d : ℝ), d = 2 → (c^2) * (d^4) = k)) : 
  (∀ (d : ℝ), d = 4 → (∃ (c : ℝ), (c^2) = 4)) := 
by 
  sorry

end inverse_variation_example_l2327_232726


namespace flight_duration_is_four_hours_l2327_232774

def convert_to_moscow_time (local_time : ℕ) (time_difference : ℕ) : ℕ :=
  (local_time - time_difference) % 24

def flight_duration (departure_time arrival_time : ℕ) : ℕ :=
  (arrival_time - departure_time) % 24

def duration_per_flight (total_flight_time : ℕ) (number_of_flights : ℕ) : ℕ :=
  total_flight_time / number_of_flights

theorem flight_duration_is_four_hours :
  let MoscowToBishkekTimeDifference := 3
  let departureMoscowTime := 12
  let arrivalBishkekLocalTime := 18
  let departureBishkekLocalTime := 8
  let arrivalMoscowTime := 10
  let outboundArrivalMoscowTime := convert_to_moscow_time arrivalBishkekLocalTime MoscowToBishkekTimeDifference
  let returnDepartureMoscowTime := convert_to_moscow_time departureBishkekLocalTime MoscowToBishkekTimeDifference
  let outboundDuration := flight_duration departureMoscowTime outboundArrivalMoscowTime
  let returnDuration := flight_duration returnDepartureMoscowTime arrivalMoscowTime
  let totalFlightTime := outboundDuration + returnDuration
  duration_per_flight totalFlightTime 2 = 4 := by
  sorry

end flight_duration_is_four_hours_l2327_232774


namespace more_volunteers_needed_l2327_232707

theorem more_volunteers_needed
    (required_volunteers : ℕ)
    (students_per_class : ℕ)
    (num_classes : ℕ)
    (teacher_volunteers : ℕ)
    (total_volunteers : ℕ) :
    required_volunteers = 50 →
    students_per_class = 5 →
    num_classes = 6 →
    teacher_volunteers = 13 →
    total_volunteers = (students_per_class * num_classes) + teacher_volunteers →
    (required_volunteers - total_volunteers) = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end more_volunteers_needed_l2327_232707


namespace sequence_general_formula_l2327_232724

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 12)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) :
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 12 :=
sorry

end sequence_general_formula_l2327_232724


namespace quadratic_real_roots_range_l2327_232797

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l2327_232797


namespace faster_speed_l2327_232789

theorem faster_speed (x : ℝ) (h1 : 10 ≠ 0) (h2 : 5 * 10 = 50) (h3 : 50 + 20 = 70) (h4 : 5 = 70 / x) : x = 14 :=
by
  -- proof steps go here
  sorry

end faster_speed_l2327_232789


namespace factor_expression_l2327_232713

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) :=
sorry

end factor_expression_l2327_232713


namespace sheepdog_rounded_up_percentage_l2327_232720

/-- Carla's sheepdog rounded up a certain percentage of her sheep. We know the remaining 10% of the sheep  wandered off into the hills, which is 9 sheep out in the wilderness. There are 81 sheep in the pen. We need to prove that the sheepdog rounded up 90% of the total number of sheep. -/
theorem sheepdog_rounded_up_percentage (total_sheep pen_sheep wilderness_sheep : ℕ) 
  (h1 : wilderness_sheep = 9) 
  (h2 : pen_sheep = 81) 
  (h3 : wilderness_sheep = total_sheep / 10) :
  (pen_sheep * 100 / total_sheep) = 90 :=
sorry

end sheepdog_rounded_up_percentage_l2327_232720


namespace yoghurt_cost_1_l2327_232704

theorem yoghurt_cost_1 :
  ∃ y : ℝ,
  (∀ (ice_cream_cartons yoghurt_cartons : ℕ) (ice_cream_cost_one_carton : ℝ) (yoghurt_cost_one_carton : ℝ),
    ice_cream_cartons = 19 →
    yoghurt_cartons = 4 →
    ice_cream_cost_one_carton = 7 →
    (19 * 7 = 133) →  -- total ice cream cost
    (133 - 129 = 4) → -- Total yogurt cost
    (4 = 4 * y) →    -- Yoghurt cost equation
    y = 1) :=
sorry

end yoghurt_cost_1_l2327_232704


namespace integer_solutions_set_l2327_232760

theorem integer_solutions_set :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} :=
by {
  sorry
}

end integer_solutions_set_l2327_232760


namespace range_of_y_l2327_232706

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 120) : y ∈ Set.Ioo (-11 : ℝ) (-10 : ℝ) :=
sorry

end range_of_y_l2327_232706


namespace factorize_difference_of_squares_l2327_232778

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l2327_232778


namespace original_number_exists_l2327_232772

theorem original_number_exists (x : ℤ) (h1 : x * 16 = 3408) (h2 : 0.016 * 2.13 = 0.03408) : x = 213 := 
by 
  sorry

end original_number_exists_l2327_232772


namespace length_of_c_l2327_232746

theorem length_of_c (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h_triangle : 0 < c) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) → c = 3 :=
by
  intros h_ineq
  sorry

end length_of_c_l2327_232746


namespace radius_increase_of_pizza_l2327_232763

/-- 
Prove that the percent increase in radius from a medium pizza to a large pizza is 20% 
given the following conditions:
1. The radius of the large pizza is some percent larger than that of a medium pizza.
2. The percent increase in area between a medium and a large pizza is approximately 44%.
3. The area of a circle is given by the formula A = π * r^2.
--/
theorem radius_increase_of_pizza
  (r R : ℝ) -- r and R are the radii of the medium and large pizza respectively
  (h1 : R = (1 + k) * r) -- The radius of the large pizza is some percent larger than that of a medium pizza
  (h2 : π * R^2 = 1.44 * π * r^2) -- The percent increase in area between a medium and a large pizza is approximately 44%
  : k = 0.2 := 
sorry

end radius_increase_of_pizza_l2327_232763


namespace work_day_meeting_percent_l2327_232722

open Nat

theorem work_day_meeting_percent :
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 35 := 
by
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  sorry

end work_day_meeting_percent_l2327_232722


namespace mr_johnson_total_volunteers_l2327_232751

theorem mr_johnson_total_volunteers (students_per_class : ℕ) (classes : ℕ) (teachers : ℕ) (additional_volunteers : ℕ) :
  students_per_class = 5 → classes = 6 → teachers = 13 → additional_volunteers = 7 →
  (students_per_class * classes + teachers + additional_volunteers) = 50 :=
by intros; simp [*]

end mr_johnson_total_volunteers_l2327_232751


namespace equation_of_line_passing_through_points_l2327_232795

-- Definition of the points
def point1 : ℝ × ℝ := (-2, -3)
def point2 : ℝ × ℝ := (4, 7)

-- The statement to prove
theorem equation_of_line_passing_through_points :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (forall (x y : ℝ), 
  y + 3 = (5 / 3) * (x + 2) → 3 * y - 5 * x = 1) := sorry

end equation_of_line_passing_through_points_l2327_232795


namespace division_multiplication_l2327_232716

-- Given a number x, we want to prove that (x / 6) * 12 = 2 * x under basic arithmetic operations.

theorem division_multiplication (x : ℝ) : (x / 6) * 12 = 2 * x := 
by
  sorry

end division_multiplication_l2327_232716


namespace max_sum_of_digits_l2327_232745

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_sum_of_digits : ∃ h m : ℕ, h < 24 ∧ m < 60 ∧
  sum_of_digits h + sum_of_digits m = 24 :=
by
  sorry

end max_sum_of_digits_l2327_232745


namespace cos_300_eq_one_half_l2327_232754

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l2327_232754


namespace triangle_height_l2327_232735

theorem triangle_height (base height : ℝ) (area : ℝ) (h_base : base = 4) (h_area : area = 12) (h_area_eq : area = (base * height) / 2) :
  height = 6 :=
by
  sorry

end triangle_height_l2327_232735


namespace valid_outfit_selections_l2327_232792

-- Definitions based on the given conditions
def num_shirts : ℕ := 6
def num_pants : ℕ := 5
def num_hats : ℕ := 6
def num_colors : ℕ := 6

-- The total number of outfits without restrictions
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- The theorem statement to prove the final answer
theorem valid_outfit_selections : total_outfits = 150 :=
by
  have h1 : total_outfits = 6 * 5 * 6 := rfl
  have h2 : 6 * 5 * 6 = 180 := by norm_num
  have h3 : 180 = 150 := sorry -- Here you need to differentiate the invalid outfits using provided restrictions
  exact h3

end valid_outfit_selections_l2327_232792


namespace polar_to_cartesian_l2327_232738

theorem polar_to_cartesian :
  ∃ (x y : ℝ), x = 2 * Real.cos (Real.pi / 6) ∧ y = 2 * Real.sin (Real.pi / 6) ∧ 
  (x, y) = (Real.sqrt 3, 1) :=
by
  use (2 * Real.cos (Real.pi / 6)), (2 * Real.sin (Real.pi / 6))
  -- The proof will show the necessary steps
  sorry

end polar_to_cartesian_l2327_232738


namespace difference_in_ages_l2327_232709

/-- Definitions: --/
def sum_of_ages (B J : ℕ) := B + J = 70
def jennis_age (J : ℕ) := J = 19

/-- Theorem: --/
theorem difference_in_ages : ∀ (B J : ℕ), sum_of_ages B J → jennis_age J → B - J = 32 :=
by
  intros B J hsum hJ
  rw [jennis_age] at hJ
  rw [sum_of_ages] at hsum
  sorry

end difference_in_ages_l2327_232709


namespace amount_of_first_alloy_used_is_15_l2327_232721

-- Definitions of percentages and weights
def chromium_percentage_first_alloy : ℝ := 0.12
def chromium_percentage_second_alloy : ℝ := 0.08
def weight_second_alloy : ℝ := 40
def chromium_percentage_new_alloy : ℝ := 0.0909090909090909
def total_weight_new_alloy (x : ℝ) : ℝ := x + weight_second_alloy
def chromium_content_first_alloy (x : ℝ) : ℝ := chromium_percentage_first_alloy * x
def chromium_content_second_alloy : ℝ := chromium_percentage_second_alloy * weight_second_alloy
def total_chromium_content (x : ℝ) : ℝ := chromium_content_first_alloy x + chromium_content_second_alloy

-- The proof problem
theorem amount_of_first_alloy_used_is_15 :
  ∃ x : ℝ, total_chromium_content x = chromium_percentage_new_alloy * total_weight_new_alloy x ∧ x = 15 :=
by
  sorry

end amount_of_first_alloy_used_is_15_l2327_232721


namespace simplify_expression_l2327_232794

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l2327_232794


namespace max_constant_C_all_real_numbers_l2327_232761

theorem max_constant_C_all_real_numbers:
  ∀ (x1 x2 x3 x4 x5 x6 : ℝ), 
  (x1 + x2 + x3 + x4 + x5 + x6)^2 ≥ 
  3 * (x1 * (x2 + x3) + x2 * (x3 + x4) + x3 * (x4 + x5) + x4 * (x5 + x6) + x5 * (x6 + x1) + x6 * (x1 + x2)) := 
by 
  sorry

end max_constant_C_all_real_numbers_l2327_232761


namespace find_X_l2327_232717

def operation (X Y : Int) : Int := X + 2 * Y 

lemma property_1 (X : Int) : operation X 0 = X := 
by simp [operation]

lemma property_2 (X Y : Int) : operation X (Y - 1) = (operation X Y) - 2 := 
by simp [operation]; linarith

lemma property_3 (X Y : Int) : operation X (Y + 1) = (operation X Y) + 2 := 
by simp [operation]; linarith

theorem find_X (X : Int) : operation X X = -2019 ↔ X = -673 :=
by sorry

end find_X_l2327_232717


namespace simplify_polynomial_l2327_232787

theorem simplify_polynomial :
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  (x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10) :=
by
  sorry

end simplify_polynomial_l2327_232787


namespace problem1_problem2_problem3_l2327_232767

-- Problem 1
theorem problem1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) (h : m - n = 2) : 2 * (n - m) - 4 * m + 4 * n - 3 = -15 :=
by sorry

-- Problem 3
theorem problem3 (m n : ℝ) (h1 : m^2 + 2 * m * n = -2) (h2 : m * n - n^2 = -4) : 
  3 * m^2 + (9 / 2) * m * n + (3 / 2) * n^2 = 0 :=
by sorry

end problem1_problem2_problem3_l2327_232767


namespace positive_integers_mod_l2327_232710

theorem positive_integers_mod (n : ℕ) (h : n > 0) :
  ∃! (x : ℕ), x < 10^n ∧ x^2 % 10^n = x % 10^n :=
sorry

end positive_integers_mod_l2327_232710


namespace distinct_xy_values_l2327_232734

theorem distinct_xy_values : ∃ (xy_values : Finset ℕ), 
  (∀ (x y : ℕ), (0 < x ∧ 0 < y) → (1 / Real.sqrt x + 1 / Real.sqrt y = 1 / Real.sqrt 20) → (xy_values = {8100, 6400})) ∧
  (xy_values.card = 2) :=
by
  sorry

end distinct_xy_values_l2327_232734


namespace calculate_down_payment_l2327_232700

def loan_period_years : ℕ := 5
def monthly_payment : ℝ := 250.0
def car_price : ℝ := 20000.0
def months_in_year : ℕ := 12

def total_loan_period_months : ℕ := loan_period_years * months_in_year
def total_amount_paid : ℝ := monthly_payment * total_loan_period_months
def down_payment : ℝ := car_price - total_amount_paid

theorem calculate_down_payment : down_payment = 5000 :=
by 
  simp [loan_period_years, monthly_payment, car_price, months_in_year, total_loan_period_months, total_amount_paid, down_payment]
  sorry

end calculate_down_payment_l2327_232700


namespace firecracker_confiscation_l2327_232736

variables
  (F : ℕ)   -- Total number of firecrackers bought
  (R : ℕ)   -- Number of firecrackers remaining after confiscation
  (D : ℕ)   -- Number of defective firecrackers
  (G : ℕ)   -- Number of good firecrackers before setting off half
  (C : ℕ)   -- Number of firecrackers confiscated

-- Define the conditions:
def conditions := 
  F = 48 ∧
  D = R / 6 ∧
  G = 2 * 15 ∧
  R - D = G ∧
  F - R = C

-- The theorem to prove:
theorem firecracker_confiscation (h : conditions F R D G C) : C = 12 := 
  sorry

end firecracker_confiscation_l2327_232736


namespace both_boys_and_girls_selected_probability_l2327_232719

theorem both_boys_and_girls_selected_probability :
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) :=
by
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  have h : (only_girls_ways / total_ways : ℚ) = (1 / 10 : ℚ) := sorry
  have h1 : (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) := by rw [h]; norm_num
  exact h1

end both_boys_and_girls_selected_probability_l2327_232719
