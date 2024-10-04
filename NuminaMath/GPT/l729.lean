import Mathlib

namespace NiE_parallel_AD_l729_729184

variables {A B C D O1 O2 O' Ni E : Type*} [EuclideanGeometry A B C] 

-- Given the conditions stated in the problem.
variables (tri_ABC : Triangle A B C)
variable (D_on_BC : PointOnSegment D B C)
variable (circumcenter_ABD : Circumcenter O1 (Triangle A B D))
variable (circumcenter_ACD : Circumcenter O2 (Triangle A C D))
variable (circumcircle_AO1O2 : Circumcircle O' (points A O1 O2))
variable (nine_point_center_ABC : NinePointCenter Ni (Triangle A B C))
variable (O'E_perp_BC : Perpendicular O' E (Line B C))

-- Statement of the theorem to prove.
theorem NiE_parallel_AD :
  Parallel (Line Ni E) (Line A D) :=
sorry

end NiE_parallel_AD_l729_729184


namespace evaluate_expression_l729_729845

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729845


namespace aiyanna_more_cookies_than_alyssa_l729_729730

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end aiyanna_more_cookies_than_alyssa_l729_729730


namespace train_speed_time_relationship_l729_729728

def original_time (x : ℝ) : ℝ := 400 / x
def new_time (x : ℝ) : ℝ := 400 / (x + 20)
def time_difference (x : ℝ) : ℝ := original_time x - new_time x

theorem train_speed_time_relationship (x : ℝ) (h : x > 0) :
  time_difference x = 0.5 :=
sorry

end train_speed_time_relationship_l729_729728


namespace product_infeasibility_l729_729755

noncomputable def product_infeasible : Prop :=
  ∏ n in Finset.range 15, (factorial n * (n + 1) : ℝ) / ((n + 3) ^ 3 : ℝ) → False

theorem product_infeasibility : product_infeasible := by
  sorry

end product_infeasibility_l729_729755


namespace octagon_angle_sum_eq_1080_l729_729655

-- Definitions
def octagon_interior_angle_sum : ℕ → ℕ := 
  λ n, (n - 2) * 180

-- Theorem statement
theorem octagon_angle_sum_eq_1080 : octagon_interior_angle_sum 8 = 1080 :=
sorry

end octagon_angle_sum_eq_1080_l729_729655


namespace standard_equation_of_ellipse_l729_729534

theorem standard_equation_of_ellipse
  (a b c : ℝ)
  (h_major_minor : 2 * a = 6 * b)
  (h_focal_distance : 2 * c = 8)
  (h_ellipse_relation : a^2 = b^2 + c^2) :
  (∀ x y : ℝ, (x^2 / 18 + y^2 / 2 = 1) ∨ (y^2 / 18 + x^2 / 2 = 1)) :=
by {
  sorry
}

end standard_equation_of_ellipse_l729_729534


namespace tourist_punctuality_l729_729601

def wealth_disparity (wealth : ℕ → ℕ) (group_size : ℕ) (tourist_id : ℕ) : Prop := sorry
def excursion_cost (group_size : ℕ) (cost_per_person : ℕ) : Prop := sorry
def perceived_importance (payment : ℕ) (group_size : ℕ) (tourist_id : ℕ) : Prop := sorry

theorem tourist_punctuality (group_size : ℕ) (wealth : ℕ → ℕ) (cost_per_person : ℕ) (payment : ℕ) :
  (∀ N, N = group_size ∧ N > 1 → ∀ tourist, wealth_disparity wealth group_size tourist ∧ 
       excursion_cost group_size cost_per_person ∧
       perceived_importance payment group_size tourist → tourist_is_punctual tourist) ∧
  (∀ N, N = group_size ∧ N = 1 → ∃ tourist, wealth_disparity wealth group_size tourist ∨ 
       excursion_cost group_size cost_per_person ∨
       perceived_importance payment group_size tourist ∧ ¬tourist_is_punctual tourist) :=
sorry

end tourist_punctuality_l729_729601


namespace compute_all_m_f_l729_729432

noncomputable def f_exists (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, differentiable ℝ (n • f) ∧ continuous (n • f)

def f_cond (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = deriv (deriv (deriv f x))) ∧
  (f 0 + deriv f 0 + deriv (deriv f 0) = 0) ∧
  (f 0 = deriv f 0)

def m_f (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

theorem compute_all_m_f (f : ℝ → ℝ) (hof : f_exists f) (hcf : f_cond f) :
  ∃ x : ℝ, m_f f x ∧ (x = 0 ∨ x = 2 * π * Real.sqrt 3 / 3) :=
sorry

end compute_all_m_f_l729_729432


namespace find_sum_l729_729998

variables {f : ℝ → ℝ}

-- Given conditions
axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom shifted_odd_function : ∀ x : ℝ, f(x + 1) = -f(-1 + x)
axiom f_at_2 : f(2) = -1

-- Proof obligation
theorem find_sum : f(1) + (∑ i in finset.range 2011, f(i + 2)) = -1 :=
by {
  sorry
}

end find_sum_l729_729998


namespace a_plus_b_plus_c_at_2_l729_729637

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def maximum_value (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x = 75

def passes_through (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  quadratic a b c p1.1 = p1.2 ∧ quadratic a b c p2.1 = p2.2

theorem a_plus_b_plus_c_at_2 
  (a b c : ℝ)
  (hmax : maximum_value a b c)
  (hpoints : passes_through a b c (-3, 0) (3, 0))
  (hvertex : ∀ x : ℝ, quadratic a 0 c x ≤ quadratic a (2 * b) c 0) : 
  quadratic a b c 2 = 125 / 3 :=
sorry

end a_plus_b_plus_c_at_2_l729_729637


namespace axis_of_symmetry_vertex_on_x_axis_range_of_values_l729_729511

open Real

noncomputable def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : - (-2 * a) / (2 * a) = 1 :=
by
  have h1 : 2 * a ≠ 0 := by linarith
  field_simp
  exact rfl

theorem vertex_on_x_axis (a : ℝ) (h : a ≠ 0) : 
    let a1 := (2 * a)^2 - 4 * a * (3 - 3 * |a|)
    a1 = 0 ↔ a = 3 / 4 ∨ a = -3 / 2 :=
by sorry

theorem range_of_values (a : ℝ) (h : a > 0) (h1 : y1 > y2) (P : parabola a a = y1) (Q : parabola a 2 = y2) : 
    y1 > y2 → a > 2 :=
by sorry

end axis_of_symmetry_vertex_on_x_axis_range_of_values_l729_729511


namespace evaluate_expression_l729_729837

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729837


namespace concave_three_digit_count_l729_729172

noncomputable def is_concave_digit (h t u : ℕ) : Prop :=
t < h ∧ t < u ∧ h ≠ u ∧ h ≠ t ∧ t ≠ u

noncomputable def three_digit_number (h t u : ℕ) : ℕ :=
100 * h + 10 * t + u

theorem concave_three_digit_count : 
  (Finset.filter (λ n, ∃ h t u, three_digit_number h t u = n ∧ is_concave_digit h t u)
  (Finset.range 1000)).card = 448 :=
sorry

end concave_three_digit_count_l729_729172


namespace count_powers_of_2_not_4_l729_729164

theorem count_powers_of_2_not_4 (n : ℕ) (h : n = 500000) : 
  (∑ k in finset.range 20, ite ((¬ (∃ m, 2 ^ (2 * m) = 2 ^ k)) ∧ (2 ^ k < n)) 1 0) = 9 := 
by
  sorry

end count_powers_of_2_not_4_l729_729164


namespace parabola_reflection_translation_l729_729398

open Real

noncomputable def f (a b c x : ℝ) : ℝ := a * (x - 4)^2 + b * (x - 4) + c
noncomputable def g (a b c x : ℝ) : ℝ := -a * (x + 4)^2 - b * (x + 4) - c
noncomputable def fg_x (a b c x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation (a b c x : ℝ) (ha : a ≠ 0) :
  fg_x a b c x = -16 * a * x :=
by
  sorry

end parabola_reflection_translation_l729_729398


namespace CE_parallel_AO_l729_729219

theorem CE_parallel_AO (A B C O J P E : Point)
  (h1 : O ∈ interior_triangle A B C)
  (h2 : line_through O ∥ line_through B C)
  (h3 : line_through J O ∈ line_through A B)
  (h4 : line_through P O ∈ line_through A C)
  (h5 : line_through P E ∈ line_through P ∥ line_through A B)
  (h6 : line_through E ∈ extension_of_line_through B O) :
  line_through C E ∥ line_through A O :=
sorry

end CE_parallel_AO_l729_729219


namespace rhombus_construction_exists_l729_729433

open EuclideanGeometry

variables (A B C D : Point)
variables (BD AD : Line)

def isRhombus (A B C D: Point) : Prop :=
  -- statement of the rhombus properties
  ∃ M N : Point, BD ≠ 0 ∧
  -- Diagonals bisect each other
  midpoint BD M ∧
  midpoint AC N ∧
  -- Diagonals perpendicular
  perp BD AC ∧
  -- 4 equal sides
  dist A B = dist B C ∧
  dist B C = dist C D ∧
  dist C D = dist D A ∧
  -- one diagonal condition
  dist B D = 8 ∧
  -- distance from vertex B to the line AD
  dist_pt_line B AD = 5

theorem rhombus_construction_exists : 
  ∃ (A B C D : Point), isRhombus A B C D BD AD :=
sorry

end rhombus_construction_exists_l729_729433


namespace eval_expression_l729_729952

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729952


namespace min_value_h10_l729_729393

noncomputable def h : ℕ → ℕ := sorry

def is_tenuous (h : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, 0 < x → 0 < y → h(x) + h(y) > (x + y)^2

def min_sum (h : ℕ → ℕ) : Prop :=
  (∀ g : ℕ → ℕ, is_tenuous g → (∑ i in range 1 16, h(i)) ≤ (∑ i in range 1 16, g(i)))

theorem min_value_h10 (h : ℕ → ℕ) (ht : is_tenuous h) (hs : min_sum h) : h(10) = 196 :=
sorry

end min_value_h10_l729_729393


namespace evaluate_expression_l729_729863

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729863


namespace largest_reciprocal_l729_729349

theorem largest_reciprocal :
  let a := (1 : ℚ) / 4
  let b := (3 : ℚ) / 8
  let c := 0 -- reciprocal undefined
  let d := -2
  let e := 4
  a ≠ 0 →
  b ≠ 0 → 
  d ≠ 0 →
  e ≠ 0 →
  (1 / a > (1 / b)) ∧ (1 / a > 0) ∧ (1 / a > (1 / d)) ∧ (1 / a > (1 / e)) :=
by
  intros ha hb hd he
  have h1 : 1 / a = 4, from by norm_num,
  have h2 : 1 / b = 8 / 3, from by norm_num,
  have h3 : 1 / d = -1 / 2, from by norm_num,
  have h4 : 1 / e = 1 / 4, from by norm_num,
  split,
  { rw h1, rw h2, linarith, },
  split,
  { rw h1, linarith, },
  split,
  { rw h1, rw h3, linarith, },
  { rw h1, rw h4, linarith, },

end largest_reciprocal_l729_729349


namespace evaluate_expression_l729_729872

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729872


namespace area_of_region_R_l729_729336

variables (A B E D : Point) (E_interior : E ∈ interior (square A B D)) 

variables (h1 : ∠ A E B = 90)
variables (h2 : ∀ (P : Point), (distance_from_line P A D) ∈ (1/4, 1/2))

theorem area_of_region_R 
  (h_square : area (square A B D) = 1)
  (h_triangle : area (triangle A B E) = 1/8) :
  area (region_R A B E D) = 1/8 := 
sorry

end area_of_region_R_l729_729336


namespace modulus_of_z_l729_729175

-- Define the complex number z
def z : ℂ := Complex.i / (1 + Complex.i)

-- State the theorem to be proved
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 := 
by 
  sorry -- We skip the proof

end modulus_of_z_l729_729175


namespace only_n_equal_3_exists_pos_solution_l729_729464

theorem only_n_equal_3_exists_pos_solution :
  ∀ (n : ℕ), (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := 
by
  sorry

end only_n_equal_3_exists_pos_solution_l729_729464


namespace jack_turn_in_correct_amount_l729_729209

-- Definition of the conditions
def exchange_rate_euro : ℝ := 1.18
def exchange_rate_pound : ℝ := 1.39

def till_usd_total : ℝ := (2 * 100) + (1 * 50) + (5 * 20) + (3 * 10) + (7 * 5) + (27 * 1) + (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def till_euro_total : ℝ := 20 * 5
def till_pound_total : ℝ := 25 * 10

def till_usd : ℝ := till_usd_total + (till_euro_total * exchange_rate_euro) + (till_pound_total * exchange_rate_pound)

def leave_in_till_notes : ℝ := 300
def leave_in_till_coins : ℝ := (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def leave_in_till_total : ℝ := leave_in_till_notes + leave_in_till_coins

def turn_in_to_office : ℝ := till_usd - leave_in_till_total

theorem jack_turn_in_correct_amount : turn_in_to_office = 607.50 := by
  sorry

end jack_turn_in_correct_amount_l729_729209


namespace points_form_ellipse_l729_729330

noncomputable def point (α : Type) := (α × α)
noncomputable def distance (P Q : point ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def fixed_points := { AB_dist : ℝ // AB_dist = 10 }
def P_condition (A B P : point ℝ) (AB_dist : fixed_points) :=
  distance P A + distance P B = 2 * AB_dist.1

theorem points_form_ellipse (A B P : point ℝ)
  (hAB_dist : fixed_points) 
  (hP: P_condition A B P hAB_dist) :
  ∃ (e : point ℝ → Prop), (e P ∧ ∀ Q, e Q → distance Q A + distance Q B = 20) := sorry

end points_form_ellipse_l729_729330


namespace bob_needs_improvement_approx_12_97_percent_l729_729036

noncomputable def bob_time : ℕ := (10 * 60 + 40)
noncomputable def sister_time : ℕ := (9 * 60 + 17)
noncomputable def time_difference : ℕ := bob_time - sister_time
noncomputable def percentage_improvement : ℚ := (time_difference * 100) / bob_time

theorem bob_needs_improvement_approx_12_97_percent :
  percentage_improvement ≈ 12.97 :=
sorry

end bob_needs_improvement_approx_12_97_percent_l729_729036


namespace color_square_l729_729425

noncomputable def exists_same_color_points (c : ℝ × ℝ → ℕ) : Prop :=
  (∀ p q : ℝ × ℝ, c p = c q → ∥p - q∥ ≥ real.sqrt (65 / 64)) 

theorem color_square (c : ℝ × ℝ → ℕ) :
  (∀ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) →
  (∀ p : ℝ × ℝ, c p ∈ {1, 2, 3}) →
  ∃ p q : ℝ × ℝ, c p = c q ∧ ∥p - q∥ ≥ real.sqrt (65 / 64) :=
sorry

end color_square_l729_729425


namespace puppy_sleep_duration_l729_729770

-- Definitions based on the given conditions
def connor_sleep_hours : ℕ := 6
def luke_sleep_hours : ℕ := connor_sleep_hours + 2
def puppy_sleep_hours : ℕ := 2 * luke_sleep_hours

-- Theorem stating the puppy's sleep duration
theorem puppy_sleep_duration : puppy_sleep_hours = 16 :=
by
  -- ( Proof goes here )
  sorry

end puppy_sleep_duration_l729_729770


namespace ratio_after_injury_l729_729722

variable (initial_speed : ℝ) (injured_speed : ℝ)
variable (total_distance : ℝ := 40)
variable (half_distance : ℝ := 20)
variable (second_half_time : ℝ := 10)
variable (time_difference : ℝ := 5)

-- Definition of the initial speed
def v1 : ℝ := half_distance / (second_half_time - time_difference)

-- Definition of the injured speed
def v2 : ℝ := half_distance / second_half_time

-- The main statement to be proved
theorem ratio_after_injury (v1 v2 : ℝ) :
  v1 = 4 → v2 = 2 → v2 / v1 = 1 / 2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end ratio_after_injury_l729_729722


namespace anna_chargers_l729_729744

theorem anna_chargers (P L: ℕ) (h1: L = 5 * P) (h2: P + L = 24): P = 4 := by
  sorry

end anna_chargers_l729_729744


namespace Aiyanna_has_more_cookies_l729_729732

theorem Aiyanna_has_more_cookies : 
  let Alyssa_cookies := 129 in
  let Aiyanna_cookies := 140 in
  Aiyanna_cookies - Alyssa_cookies = 11 :=
by
  sorry

end Aiyanna_has_more_cookies_l729_729732


namespace combined_work_days_l729_729360

variable (p q : Type)
variable [Inhabited p] [Inhabited q]

-- Define the efficiencies and work completion times
def efficiency_p := 1.1
def days_to_complete (efficiency : ℝ) := 21 / efficiency
def work_rate (days : ℝ) := 1 / days

-- Given conditions
def p_rate := work_rate 21
def q_rate := work_rate (21 / efficiency_p)

-- Combined work rate of p and q
def combined_rate := p_rate + q_rate

-- Proof problem statement
theorem combined_work_days : combined_rate = 1 / 11 := sorry

end combined_work_days_l729_729360


namespace sides_parallel_l729_729485

-- Define a structure for a regular tetrahedron
structure Tetrahedron :=
(O A B C : Point)
(equal_sides : dist O A = dist O B ∧ dist O A = dist O C ∧ dist O A = dist A B)
(equal_angles : ∀ X Y Z : Point, angle X Y Z = π / 3)

-- Define the problem in Lean terms
theorem sides_parallel
  (T : Tetrahedron)
  (P Q R : Point)
  (hP : P ≠ T.O ∧ P ≠ T.A)
  (hQ : Q ≠ T.O ∧ Q ≠ T.B)
  (hR : R ≠ T.O ∧ R ≠ T.C)
  (hPQR_eq : dist P Q = dist Q R ∧ dist Q R = dist R P)
  (hPQR_angle : ∀ X Y Z : Point, X ≠ Y → Y ≠ Z → Z ≠ X → angle X Y Z = π / 3) :
  parallel (line P Q) (line T.A T.B) ∧
  parallel (line Q R) (line T.B T.C) ∧
  parallel (line R P) (line T.C T.A) :=
by {
  sorry
}

end sides_parallel_l729_729485


namespace red_marbles_in_A_l729_729323

-- Define the number of marbles in baskets A, B, and C
variables (R : ℕ)
def basketA := R + 2 -- Basket A: R red, 2 yellow
def basketB := 6 + 1 -- Basket B: 6 green, 1 yellow
def basketC := 3 + 9 -- Basket C: 3 white, 9 yellow

-- Define the greatest difference condition
def greatest_difference (A B C : ℕ) := max (max (A - B) (B - C)) (max (A - C) (C - B))

-- Define the hypothesis based on the conditions
axiom H1 : greatest_difference 3 9 0 = 6

-- The theorem we need to prove: The number of red marbles in Basket A is 8
theorem red_marbles_in_A : R = 8 := 
by {
  -- The proof would go here, but we'll use sorry to skip it
  sorry
}

end red_marbles_in_A_l729_729323


namespace distance_between_parallel_lines_l729_729457

theorem distance_between_parallel_lines :
  let A := 4
  let B := 3
  let C1 := -1
  let C2 := 3 / 2
  let d := (|C2 - C1|) / Real.sqrt (A^2 + B^2)
  in d = 1 / 2 :=
by
  sorry

end distance_between_parallel_lines_l729_729457


namespace octagon_diagonals_l729_729152

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l729_729152


namespace smallest_twice_perfect_square_three_times_perfect_cube_l729_729983

theorem smallest_twice_perfect_square_three_times_perfect_cube :
  ∃ n : ℕ, (∃ k : ℕ, n = 2 * k^2) ∧ (∃ m : ℕ, n = 3 * m^3) ∧ n = 648 :=
by
  sorry

end smallest_twice_perfect_square_three_times_perfect_cube_l729_729983


namespace max_value_a5_a10_l729_729482

noncomputable def max_a5_a10 (a : ℕ → ℕ) (P : ∑ i in finset.range 10, a i = 2000 ∧ ∀ i j : fin 10, i < j → a i ≤ a j) : Prop :=
  let a5 := a 4 in
  let a10 := a 9 in
  (a5 = 329) ∧ (a10 = 335)

theorem max_value_a5_a10 (a : ℕ → ℕ) (P : ∑ i in finset.range 10, a i = 2000 ∧ ∀ i j : fin 10, i < j → a i ≤ a j) :
  max_a5_a10 a P :=
by {
  sorry
}

end max_value_a5_a10_l729_729482


namespace find_x_l729_729359

theorem find_x (x : ℕ) (h : 2^x - 2^(x-2) = 3 * 2^(12)) : x = 14 :=
sorry

end find_x_l729_729359


namespace eval_expression_l729_729827

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729827


namespace coefficient_x17_x18_l729_729051

noncomputable def coefficient_x_pow (c : ℕ) (n : ℕ) : ℕ :=
(multichoose (λ k, if k ∈ {5, 7} then 1 else 0) n).filter (λ s, s.sum = c).card 

theorem coefficient_x17_x18 : 
  coefficient_x_pow 17 20 = 3420 ∧ coefficient_x_pow 18 20 = 0 := 
by 
  sorry

end coefficient_x17_x18_l729_729051


namespace evaluate_expression_l729_729851

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729851


namespace eventually_non_negative_l729_729289

-- The problem conditions and goal
theorem eventually_non_negative :
  ∀ (x : Fin 5 → ℤ),
    (0 < (∑ i, x i)) →
    (∃ n, ∀ m ≥ n, ∀ i, nth_operation x m i ≥ 0)
:= by
  sorry

-- Function describing the transformation rule in Lean
noncomputable def nth_operation (x : Fin 5 → ℤ) (n : ℕ) (i : Fin 5) : ℤ := 
  if x i < 0 then
    match i with
    | ⟨0, _⟩    => x ⟨4, sorry⟩ + x i
    | ⟨1, _⟩    => x ⟨0, sorry⟩ + x i
    | ⟨2, _⟩    => -x i
    | ⟨3, _⟩    => x ⟨2, sorry⟩ + x i
    | ⟨4, _⟩    => x ⟨3, sorry⟩ + x i
  else x i

end eventually_non_negative_l729_729289


namespace limit_sequences_l729_729274

noncomputable def initial_triple : ℝ × ℝ × ℝ := (1007 * real.sqrt 2, 2014 * real.sqrt 2, 1007 * real.sqrt 14)

def recurrence (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (real.sqrt (x * (y + z - x)), real.sqrt (y * (z + x - y)), real.sqrt (z * (x + y - z)))

def seq_x_y_z (n : ℕ) : ℝ × ℝ × ℝ :=
  nat.rec_on n initial_triple (λ n' p, recurrence p.1 p.2 p.3)

theorem limit_sequences :
  ∃ l : ℝ, (tendsto (λ n, (seq_x_y_z n).1) at_top (𝓝 l)) ∧
           (tendsto (λ n, (seq_x_y_z n).2) at_top (𝓝 l)) ∧
           (tendsto (λ n, (seq_x_y_z n).3) at_top (𝓝 l)) ∧
           l = 2014 :=
by
  sorry

end limit_sequences_l729_729274


namespace total_possible_orders_correct_l729_729194

noncomputable def total_possible_orders : Nat := sorry

theorem total_possible_orders_correct :
  total_possible_orders = 2! * 3! * 4! * 1! := by
  sorry

end total_possible_orders_correct_l729_729194


namespace solve_x_l729_729556

-- Define the structure of the pyramid
def pyramid (x : ℕ) : Prop :=
  let level1 := [x + 4, 12, 15, 18]
  let level2 := [x + 16, 27, 33]
  let level3 := [x + 43, 60]
  let top := x + 103
  top = 120

theorem solve_x : ∃ x : ℕ, pyramid x → x = 17 :=
by
  -- Proof omitted
  sorry

end solve_x_l729_729556


namespace sum_of_squares_lt_nine_l729_729368

-- Define the set of points and their properties
variables {P : Type} [metric_space P]
def is_in_circle (S : set P) (C : P) (r : ℝ) :=
∀ p ∈ S, dist C p ≤ r

def is_center_point (S : set P) (C : P) :=
C ∈ S

-- Define the distances x_i
def nearest_distance (S : set P) (P_i : P) (h : P_i ∈ S) : ℝ :=
Inf {dist P_i P_j | P_j ∈ S ∧ P_j ≠ P_i}

noncomputable def distances (S : set P) 
  (hS : is_in_circle S C 1) 
  (hC : is_center_point S C) : list ℝ :=
(list.map (λ P_i, nearest_distance S P_i sorry) (finset.to_list S.to_finset))

-- Establishing the proof problem in Lean
theorem sum_of_squares_lt_nine 
  {S : set P} 
  (hS : is_in_circle S C 1) 
  (hC : is_center_point S C) 
  (h_count : S.to_finset.card = 2000)
  (x_i : distances S hS hC)
  :
  list.sum (list.map (λ x, x^2) x_i) < 9 := 
begin
  sorry
end

end sum_of_squares_lt_nine_l729_729368


namespace average_between_15_and_55_div_by_4_l729_729357

open Finset

-- Define the conditions
def isDivisibleBy4 (n : ℕ) : Prop := n % 4 = 0

-- Define the range and the filtered set of numbers
def numsBetween15And55Div4 : Finset ℕ := (Finset.range (55 - 15 + 1)).filter (λ x, isDivisibleBy4 (x + 15))

-- Calculating sum and count
def numsSum : ℕ := Finset.sum numsBetween15And55Div4 id
def numsCount : ℕ := Finset.card numsBetween15And55Div4

-- The average calculation
def average := (numsSum : ℚ) / numsCount

theorem average_between_15_and_55_div_by_4 : average = 34 := 
by 
  sorry

end average_between_15_and_55_div_by_4_l729_729357


namespace eyes_that_saw_the_plane_l729_729322

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end eyes_that_saw_the_plane_l729_729322


namespace small_fries_number_l729_729013

variables (L S : ℕ)

axiom h1 : L + S = 24
axiom h2 : L = 5 * S

theorem small_fries_number : S = 4 :=
by sorry

end small_fries_number_l729_729013


namespace regression_is_appropriate_l729_729676

-- Definitions for the different analysis methods
inductive AnalysisMethod
| ResidualAnalysis : AnalysisMethod
| RegressionAnalysis : AnalysisMethod
| IsoplethBarChart : AnalysisMethod
| IndependenceTest : AnalysisMethod

-- Relating height and weight with an appropriate analysis method
def appropriateMethod (method : AnalysisMethod) : Prop :=
  method = AnalysisMethod.RegressionAnalysis

-- Stating the theorem that regression analysis is the appropriate method
theorem regression_is_appropriate : appropriateMethod AnalysisMethod.RegressionAnalysis :=
by sorry

end regression_is_appropriate_l729_729676


namespace eval_expression_l729_729955

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729955


namespace octagon_diagonals_l729_729157

theorem octagon_diagonals : 
  let n := 8 in
  (n * (n - 3)) / 2 = 20 := 
by 
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2 : by rfl
                    ... = 40 / 2           : by norm_num
                    ... = 20               : by norm_num

end octagon_diagonals_l729_729157


namespace emma_height_in_meters_l729_729448

-- Define the conditions as Lean 4 statements
def emma_height_inches := 67
def inches_to_cm := 2.54
def cm_to_meter := 100

-- Define the main theorem statement, proving the given height in meters
theorem emma_height_in_meters : 
  (emma_height_inches * inches_to_cm) / cm_to_meter = 1.7 :=
by
  sorry

end emma_height_in_meters_l729_729448


namespace check_quadratic_function_l729_729677

def isQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem check_quadratic_function : 
  (isQuadraticFunction (λ x, x^2 / 3) ∧ ¬ isQuadraticFunction (λ x, (x^2 - 4) ^ (1/2)) ∧ 
   ¬ isQuadraticFunction (λ x, 1 / (x^2 - 3)) ∧ ¬ isQuadraticFunction (λ x, x - 3)) :=
by
  repeat { sorry }

end check_quadratic_function_l729_729677


namespace eval_expression_l729_729938

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729938


namespace complex_conjugate_quadrant_l729_729283

def z (c : ℂ) : Prop := (c - 3) * (2 - complex.I) = 5 * complex.I

def z_conjugate_in_fourth_quadrant (c : ℂ) : Prop :=
  c.im < 0 ∧ c.re > 0

theorem complex_conjugate_quadrant (c : ℂ) (h : z c) :
  z_conjugate_in_fourth_quadrant c.conj :=
sorry

end complex_conjugate_quadrant_l729_729283


namespace eval_expression_l729_729823

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729823


namespace evaluate_expression_l729_729857

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729857


namespace evaluate_expression_l729_729842

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729842


namespace trainA_time_correct_trainB_time_correct_trainB_faster_time_l729_729523

-- Definitions based on the conditions
def trainALength: ℕ := 150
def trainBLength: ℕ := 120
def bridgeXLength: ℕ := 300
def bridgeYLength: ℕ := 250
def trainASpeedKmph: ℕ := 45
def trainBSpeedKmph: ℕ := 60
def kmphToMps: ℕ := 1000 / 3600

def trainASpeedMps: ℕ := trainASpeedKmph * kmphToMps
def trainBSpeedMps: ℕ := trainBSpeedKmph * kmphToMps

def totalDistanceA: ℕ := trainALength + bridgeXLength
def totalDistanceB: ℕ := trainBLength + bridgeYLength

def timeTrainACrossBridgeX: ℕ := totalDistanceA / trainASpeedMps
def timeTrainBCrossBridgeY: ℕ := totalDistanceB / trainBSpeedMps

-- Proofs
theorem trainA_time_correct : 
timeTrainACrossBridgeX = 36 := 
by 
sorry

theorem trainB_time_correct : 
timeTrainBCrossBridgeY = 22.2 := 
by 
sorry

theorem trainB_faster_time : 
(timeTrainACrossBridgeX - timeTrainBCrossBridgeY) = 13.8 := 
by 
sorry

end trainA_time_correct_trainB_time_correct_trainB_faster_time_l729_729523


namespace train_cross_time_approx_24_seconds_l729_729522

open Real

noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_h : ℝ) (man_speed_km_h : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_h * (1000 / 3600)
  let man_speed_m_s := man_speed_km_h * (1000 / 3600)
  let relative_speed := train_speed_m_s - man_speed_m_s
  train_length / relative_speed

theorem train_cross_time_approx_24_seconds : 
  abs (time_to_cross 400 63 3 - 24) < 1 :=
by
  sorry

end train_cross_time_approx_24_seconds_l729_729522


namespace quadratic_equation_in_terms_of_x_l729_729678

def is_quadratic_eq (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def equation_A : ℝ → ℝ := λ x, 1 / x^2 - x - 1
def equation_B (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c
def equation_C : ℝ → ℝ := λ x, (x + 1) * (x - 2) - x^2
def equation_D : ℝ → ℝ := λ x, 3 * x^2 + 1

theorem quadratic_equation_in_terms_of_x :
  ¬ is_quadratic_eq equation_A ∧
  ¬ is_quadratic_eq (equation_B 0 0 0) ∧
  ¬ is_quadratic_eq equation_C ∧
  is_quadratic_eq equation_D :=
by {
  sorry
}

end quadratic_equation_in_terms_of_x_l729_729678


namespace circle_center_radius_l729_729977

theorem circle_center_radius (x y : ℝ) : 
    x^2 - 6*x + y^2 + 2*y - 12 = 0 →
    (∃ c : ℝ × ℝ, ∃ r : ℝ, (c = (3, -1)) ∧ (r = sqrt 22) ∧ (x - c.1)^2 + (y - c.2)^2 = r^2) :=
by
  sorry

end circle_center_radius_l729_729977


namespace min_inverse_sum_l729_729510

theorem min_inverse_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 4) : 1 ≤ (1/a) + (1/b) :=
by
  sorry

end min_inverse_sum_l729_729510


namespace min_value_of_u_l729_729483

variable (a b c : ℝ)
variable (cond1 : 0 < a) (cond2 : 0 < b) (cond3 : 0 < c)
variable (sum_cond : a + b + c = 1)

def u (a b c : ℝ) : ℝ :=
  (3 * a^2 - a) / (1 + a^2) + (3 * b^2 - b) / (1 + b^2) + (3 * c^2 - c) / (1 + c^2)

theorem min_value_of_u : u a b c = 0 :=
by
  sorry

end min_value_of_u_l729_729483


namespace coefficient_x_term_in_expansion_l729_729625

-- Definitions for the problem
def polynomial : ℤ[X] := (1 - X) * (X + 1)^5

-- Theorem: The coefficient of the x term in the expansion of (1-x)(1+x)^5 is 4
theorem coefficient_x_term_in_expansion : polynomial.coeff 1 = 4 := by
  sorry

end coefficient_x_term_in_expansion_l729_729625


namespace triangle_side_count_l729_729291

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l729_729291


namespace octagon_diagonals_l729_729158

theorem octagon_diagonals : 
  let n := 8 in
  (n * (n - 3)) / 2 = 20 := 
by 
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2 : by rfl
                    ... = 40 / 2           : by norm_num
                    ... = 20               : by norm_num

end octagon_diagonals_l729_729158


namespace probability_correct_l729_729390

-- Definition of the cube size
def cube_size := 5

-- Definition of the total number of unit cubes
def total_unit_cubes := cube_size^3

-- Definition of the number of unit cubes with two painted faces
def cubes_with_two_painted_faces := 9

-- Definition of the number of unit cubes with no painted faces
def cubes_with_no_painted_faces := 80

-- Definition of the total number of ways to select 2 cubes out of 125
noncomputable def total_selections := (nat.choose total_unit_cubes 2)

-- Definition of the number of successful selections: one cube with two painted faces and one with no painted faces
def successful_selections := cubes_with_two_painted_faces * cubes_with_no_painted_faces

-- Calculation of the probability
noncomputable def probability := (successful_selections : ℚ) / total_selections

-- The statement to prove the probability is 24/258
theorem probability_correct :
  probability = (24 / 258 : ℚ) :=
by
  sorry

end probability_correct_l729_729390


namespace theorem_positive_int_n_l729_729974

theorem theorem_positive_int_n (n : ℕ) (h_pos : 0 < n) :
  (∃ k : ℤ, (n^{3 * n - 2} - 3 * n + 1) = k * (3 * n - 2)) ↔ n = 1 := 
sorry

end theorem_positive_int_n_l729_729974


namespace circumscribed_circle_around_quadrilateral_l729_729198

noncomputable theory

open EuclideanGeometry

-- Definitions for the points and sides in the problem
variables {A B C D E F G H K L M N : Point}
variable {square_ABCD : square A B C D}
variable {E_on_BC : on_line_segment B C E}
variable {F_on_CD : on_line_segment C D F}
variable {G_on_CD : on_line_segment C D G}
variable {H_on_AD : on_line_segment A D H}
variable {E_eq_CF : (dist C E) = (dist C F)}
variable {DG_eq_DH : (dist D G) = (dist D H)}
variable {KLMN_vertices : quadrilateral K L M N}
variable {K_intersections : (angle_of_intersection H B G K L M)}
variable {M_intersections : (angle_of_intersection E A F M L N)}

-- Statement of the problem in Lean 4
theorem circumscribed_circle_around_quadrilateral :
  cyclic_quadrilateral K L M N :=
sorry

end circumscribed_circle_around_quadrilateral_l729_729198


namespace sequence_an_correct_l729_729486

theorem sequence_an_correct (S_n : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S_n n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n, n ≥ 2 → a n = 2 * n - 1) :=
by
  -- We assume S_n is defined such that S_n = n^2 + 1
  -- From this, we have to show that:
  -- for n = 1, a_1 = 2,
  -- and for n ≥ 2, a_n = 2n - 1
  sorry

end sequence_an_correct_l729_729486


namespace domain_of_g_l729_729504

noncomputable def f (x : ℝ) : ℝ := sorry

def g (x : ℝ) : ℝ := f (x / 2) + f (x - 1)

theorem domain_of_g :
  (∀ x, f x ≠ ∅ → x ∈ (-1 : ℝ, 1)) →
  (∀ x, g x ≠ ∅ → x ∈ (0 : ℝ, 2)) :=
by
  intros h x hg
  have h1 : -1 < x / 2 ∧ x / 2 < 1 := sorry
  have h2 : -1 < x - 1 ∧ x - 1 < 1 := sorry
  sorry

end domain_of_g_l729_729504


namespace evaluate_expression_l729_729859

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729859


namespace count_ordered_pairs_l729_729437

def P : Set (Set ℕ) := {{S}, {T}}
def proper_subsets (s : Set (Set ℕ)) : Set (Set ℕ) := s.filter (λ t, t ⊂ P)
def ordered_pairs (S T : Set (Set ℕ)) : Set (Set (Set ℕ) × Set (Set ℕ)) := 
  (S.product T).filter (λ pair, ¬(pair.1 ⊂ pair.2) ∧ ¬(pair.2 ⊂ pair.1) ∧
                                ∀ x ∈ pair.1, ∀ y ∈ pair.2, ¬(x ⊂ y) ∧ ¬(y ⊂ x))

open Set

theorem count_ordered_pairs : 
  ∃ (n : ℕ), n = 7 ∧ ∃ S T, S ⊆ proper_subsets P ∧ T ⊆ proper_subsets P ∧
  (ordered_pairs S T).card = n := by 
sorry

end count_ordered_pairs_l729_729437


namespace peter_percentage_books_read_l729_729249

-- Definitions
def total_books : ℕ := 20
def brother_books_read (total_books : ℕ) : ℕ := (10 * total_books) / 100
def peter_books_read (brother_books_read : ℕ) : ℕ := brother_books_read + 6

-- Theorem to prove
theorem peter_percentage_books_read : 
  let P := (peter_books_read (brother_books_read total_books)).toRat / total_books * 100 in
  P = 40 :=
by
  sorry

end peter_percentage_books_read_l729_729249


namespace eval_expression_l729_729960

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729960


namespace expression_evaluation_l729_729168

theorem expression_evaluation (a b : ℤ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 :=
by
  sorry

end expression_evaluation_l729_729168


namespace evaluation_of_expression_l729_729899

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729899


namespace eval_expression_l729_729825

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729825


namespace diagonal_contains_all_numbers_l729_729693

def symmetric_table (n : ℕ) (table : matrix (fin n) (fin n) ℕ) : Prop :=
  ∀ i j : fin n, table i j = table j i

def all_numbers_in_row (n : ℕ) (table : matrix (fin n) (fin n) ℕ) : Prop :=
  ∀ i : fin n, ∀ m : ℕ, m > 0 ∧ m ≤ n → ∃ j : fin n, table i j = m

theorem diagonal_contains_all_numbers :
  ∀ (table : matrix (fin 25) (fin 25) ℕ),
    symmetric_table 25 table →
    all_numbers_in_row 25 table →
    ∀ m : ℕ, m > 0 ∧ m ≤ 25 → ∃ i : fin 25, table i i = m := 
by { sorry }

end diagonal_contains_all_numbers_l729_729693


namespace sum_of_valid_x_values_l729_729326

theorem sum_of_valid_x_values (x y : ℕ) (hxy : x * y = 360) (hx : x ≥ 20) (hy : y ≥ 12) : 
  { x | ∃ y, x * y = 360 ∧ x ≥ 20 ∧ y ≥ 12 }.sum = 74 :=
by
  sorry

end sum_of_valid_x_values_l729_729326


namespace circumradius_remains_constant_l729_729518

theorem circumradius_remains_constant
  (l1 l2: Line)
  (h_perp : Perpendicular l1 l2)
  (A B C D E: Point)
  (h_A_on_l1 : OnLine A l1)
  (h_B_on_l1 : OnLine B l1)
  (h_C_on_l1 : OnLine C l1)
  (h_D_on_l2 : OnLine D l2)
  (h_E_on_l2 : OnLine E l2)
  (h_sequence : (PositionOrder A B C))
  (h_angle_AEC_90 : Angle A E C = 90)
  (h_angle_ADB_90 : Angle A D B = 90):
  (circumradius (triangle A D E)).radius = (circumradius (triangle (translate_parallel A B l2).D (translate_parallel A B l2).E)).radius :=
sorry

end circumradius_remains_constant_l729_729518


namespace sum_of_smallest_and_largest_prime_between_1_and_50_l729_729778

theorem sum_of_smallest_and_largest_prime_between_1_and_50 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  in primes.head + primes.reverse.head = 49 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  exact rfl

end sum_of_smallest_and_largest_prime_between_1_and_50_l729_729778


namespace polynomial_root_set_unbounded_l729_729698

noncomputable def polynomial_two_variables := {P : (ℝ × ℝ) → ℝ // ∃ d : ℕ, ∀ (x y : ℝ), P (x, y) = ∑ i in finset.range d, (coe_fn (mv_polynomial.X i) x, coe_fn (mv_polynomial.X i) y)}

def highest_degree_sum (P : (ℝ × ℝ) → ℝ) : (ℝ × ℝ) → ℝ :=
  λ xy, finset.sum (finset.filter (λ m, mv_polynomial.degree (unop m) = mv_polynomial.total_degree P)
      (mv_polynomial.support P)) (λ m, mv_polynomial.coeff m P * (mv_polynomial.eval xy m))

theorem polynomial_root_set_unbounded (P : (ℝ × ℝ) → ℝ) (Q : (ℝ × ℝ) → ℝ)
  (hP : polynomial_two_variables P) (hQ : Q = highest_degree_sum P)
  (x1 y1 x2 y2 : ℝ) (hQ_positive : Q (x1, y1) > 0) (hQ_negative : Q (x2, y2) < 0) :
  ¬(bounded (λ xy : ℝ × ℝ, P xy = 0)) :=
begin
  sorry
end

end polynomial_root_set_unbounded_l729_729698


namespace problem_solution_l729_729470

def star (x y : ℝ) (h : x ≠ y) : ℝ := (x + y) / (x - y)

theorem problem_solution :
  (star (star 3 5 (by norm_num)) 8 (by norm_num)) = -1 / 3 :=
sorry

end problem_solution_l729_729470


namespace triangle_side_count_l729_729293

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l729_729293


namespace octagon_diagonals_l729_729159

theorem octagon_diagonals : 
  let n := 8 in
  (n * (n - 3)) / 2 = 20 := 
by 
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2 : by rfl
                    ... = 40 / 2           : by norm_num
                    ... = 20               : by norm_num

end octagon_diagonals_l729_729159


namespace sum_possible_n_k_l729_729629

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l729_729629


namespace eval_expression_l729_729905

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729905


namespace travel_time_from_B_to_C_to_A_to_D_l729_729132

variable (v v_r x y : ℝ)
variable (h₁ : 0 < v)
variable (h₂ : 0 < v_r)
variable (hACeqCD : ∀ (A C D: ℝ), AC = CD)
variable (tAC : x / (v - v_r) + y / v = 6)
variable (tBC : y / v + x / (v + v_r) = 8), (tCB : x / (v - v_r) = 5 )

theorem travel_time_from_B_to_C_to_A_to_D :
  12 + 1 / 3 :=
  sorry

end travel_time_from_B_to_C_to_A_to_D_l729_729132


namespace probability_of_divisible_by_4_l729_729334

-- Define the set of outcomes for rolling a standard 6-sided die
def die_outcomes := {1, 2, 3, 4, 5, 6}

-- Define the set of all possible outcomes when rolling 7 dice
def all_outcomes := set.prod die_outcomes die_outcomes die_outcomes die_outcomes die_outcomes die_outcomes die_outcomes

-- Probability that the product of numbers is divisible by 4
def probability_divisible_by_4 (outcome : all_outcomes) : Prop :=
  ∃ p : ℕ, p * 4 = outcome.val.prod

theorem probability_of_divisible_by_4 :
  probability_divisible_by_4 = 187/192 := 
sorry

end probability_of_divisible_by_4_l729_729334


namespace intersection_points_l729_729235

theorem intersection_points (l1 l2 : Type) [fintype l1] [fintype l2] (h_l1_points : ∃ s : set l1, s.card = 5) (h_l2_points : ∃ t : set l2, t.card = 10) (parallel : parallel_lines l1 l2) (no_three_intersections : no_three_segments_intersect l1 l2) :
  ∃ i : ℕ, i = 450 :=
by {
  sorry
}

end intersection_points_l729_729235


namespace evaluate_expression_l729_729797

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729797


namespace evaluate_expression_l729_729866

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729866


namespace circle_radius_l729_729075

theorem circle_radius (x y : ℝ) : x^2 + 8*x + y^2 - 10*y + 32 = 0 → ∃ r : ℝ, r = 3 :=
by
  sorry

end circle_radius_l729_729075


namespace evaluate_expression_l729_729883

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729883


namespace square_product_third_sides_l729_729257

/-- Define the relevant properties of triangles T1 and T2 --/
variables (T1 T2 : Type)
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- Conditions from the problem
def area_T1 (a1 b1 : ℝ) := 0.5 * a1 * b1 = 8
def area_T2 (a2 b2 : ℝ) := 0.5 * a2 * b1 = 18
def side_congruence1 := a2 = c1
def hypotenuse_leg_congruence := a1 = b2
def larger_angle := b2 = a2 * (Real.sqrt 3 / 2)

theorem square_product_third_sides
  (h1 : area_T1 a1 b1)
  (h2 : area_T2 a2 b2)
  (h3 : side_congruence1)
  (h4 : hypotenuse_leg_congruence)
  (h5 : larger_angle) :
  (c1 * b2) ^ 2 = 576 :=
sorry

end square_product_third_sides_l729_729257


namespace find_constant_a_l729_729180

theorem find_constant_a (a : ℚ) (S : ℕ → ℚ) (hS : ∀ n, S n = (a - 2) * 3^(n + 1) + 2) : a = 4 / 3 :=
by
  sorry

end find_constant_a_l729_729180


namespace find_number_l729_729358

theorem find_number (x : ℝ) (h : x - (3/5 : ℝ) * x = 60) : x = 150 :=
sorry

end find_number_l729_729358


namespace problem_statement_l729_729514

noncomputable def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 1 }

noncomputable def f (p : ℝ × ℝ) (h : p ∈ M) : ℝ × ℝ := (2^p.1, 2^p.2)

def N : Set (ℝ × ℝ) := { q | q.1 * q.2 = 2 ∧ q.1 > 0 ∧ q.2 > 0 }

theorem problem_statement : ∀ p ∈ M, f p p.2 ∈ N :=
by
  sorry

end problem_statement_l729_729514


namespace eval_expr_l729_729812

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729812


namespace evaluate_expression_l729_729882

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729882


namespace diagonals_in_octagon_l729_729148

theorem diagonals_in_octagon (n : ℕ) (h : n = 8) : (nat.choose n 2) - n = 20 :=
by
  rw [h, nat.choose]
  sorry

end diagonals_in_octagon_l729_729148


namespace pepperoni_fraction_covered_by_pepperoni_l729_729591

-- Definitions of conditions
def pizza_diameter : ℝ := 18
def num_pepperoni_diameter_fits : ℕ := 9
def total_pepperoni_circles : ℕ := 36

-- Definitions derived from conditions
def pepperoni_diameter : ℝ := pizza_diameter / num_pepperoni_diameter_fits
def pepperoni_radius : ℝ := pepperoni_diameter / 2
def pepperoni_area : ℝ := π * (pepperoni_radius) ^ 2
def total_pepperoni_area : ℝ := total_pepperoni_circles * pepperoni_area
def pizza_radius : ℝ := pizza_diameter / 2
def pizza_area : ℝ := π * (pizza_radius) ^ 2
def fraction_covered : ℚ := total_pepperoni_area / pizza_area

-- Theorem statement
theorem pepperoni_fraction_covered_by_pepperoni :
  fraction_covered = 4 / 9 := by
  -- Proof omitted
  sorry

end pepperoni_fraction_covered_by_pepperoni_l729_729591


namespace ways_to_pay_100_l729_729045

theorem ways_to_pay_100 :
  let denominations := [1, 2, 10, 20, 50]
  (number_of_ways_to_pay (denominations) 100) = 784 := sorry

end ways_to_pay_100_l729_729045


namespace anna_chargers_l729_729745

theorem anna_chargers (P L: ℕ) (h1: L = 5 * P) (h2: P + L = 24): P = 4 := by
  sorry

end anna_chargers_l729_729745


namespace particle_speed_is_sqrt_34_l729_729714

def path_formula (t : ℝ) : ℝ × ℝ :=
  (3 * t + 9, 5 * t - 20)

-- Conditions as functions in Lean
def particle_coordinates (t : ℝ) : ℝ × ℝ := path_formula t

def particle_speed : ℝ :=
  let (x1, y1) := particle_coordinates 2
  let (x2, y2) := particle_coordinates 3
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- The theorem we want to prove
theorem particle_speed_is_sqrt_34 : particle_speed = Real.sqrt 34 := by
  sorry

end particle_speed_is_sqrt_34_l729_729714


namespace logarithmic_inequality_l729_729477

-- Define the problem
theorem logarithmic_inequality (a b c : ℝ) (h₁ : a > 1) (h₂ : c > 0) (h₃ : log b a > log a (b * c)) : a < b ∧ a * c < b * c ∧ a^c < b^c := 
sorry

end logarithmic_inequality_l729_729477


namespace transformation_result_l729_729478

variables {ω A a ϕ : ℝ}
variables (h_ω : ω > 0) (h_A : A > 0) (h_a : a > 0) (h_ϕ : 0 < ϕ ∧ ϕ < π)
variable (h_transformation : ∀ x, 3 * sin(2 * x - π / 6) + 1 = A * sin(ω * (x + ϕ)) + a)

theorem transformation_result :
  A + a + ω + ϕ = 16 / 3 + 11 * π / 12 :=
sorry

end transformation_result_l729_729478


namespace die_rolls_multiple_of_5_l729_729407

open Probability

def probability_product_multiple_of_5 : ℚ :=
  1 - (5/6)^8

theorem die_rolls_multiple_of_5 :
  probability_product_multiple_of_5 = 1288991 / 1679616 :=
by
  sorry

end die_rolls_multiple_of_5_l729_729407


namespace length_BP_length_QT_l729_729197

-- Conditions
variables (A B C D P T S Q R : Type) [metric_space A] [metric_space B] 
[metric_space C] [metric_space D] [metric_space P] [metric_space T] [metric_space S] [metric_space Q] [metric_space R]

-- Definitions from conditions
variable [h1 : rect A B C D]
variable [h2 : on P B]
variable [h3 : angle APD = 90]
variable [h4 : perpendicular TS BC]
variable [h5 : BP = PT]
variable [h6 : T <> S] -- ensure T and S are different since TS is a line
variable [h7 : intersects PD TS Q]
variable [h8 : on R C]
variable [h9 : passes RA Q]
variable [h10 : triangle PQA (PA = 24) (AQ = 30) (QP = 18)]

-- Desired proofs
theorem length_BP : BP = 8 * sqrt 5 := 
begin
  sorry, -- Placeholder for the proof
end

theorem length_QT : QT = 2 := 
begin
  sorry, -- Placeholder for the proof
end

end length_BP_length_QT_l729_729197


namespace eval_expression_l729_729914

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729914


namespace num_triangles_with_perimeter_20_l729_729325

theorem num_triangles_with_perimeter_20 : 
  ∃ (n : ℕ), n = 8 ∧ 
  ∀ (a b c : ℕ), a + b + c = 20 → 
  a + b > c ∧ a + c > b ∧ b + c > a → 
  list.mem (a, b, c) [(9, 9, 2), (8, 8, 4), (7, 7, 6), (6, 6, 8), (9, 6, 5), (9, 7, 4), (9, 8, 3), (8, 7, 5)] = tt :=
by {
  -- proof to be provided
  sorry
}

end num_triangles_with_perimeter_20_l729_729325


namespace standard_eq_line_cartesian_eq_curve_max_distance_point_to_line_l729_729199

/-- Definition of parametric equation for line l -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3 - t, 1 + t)

/-- Definition of polar equation for curve C -/
def polar_curve (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

/-- Theorem 1: Standard equation of line l -/
theorem standard_eq_line :
  ∃ t : ℝ, ∀ (x y : ℝ), (x, y) = parametric_line t → x + y - 4 = 0 :=
sorry

/-- Theorem 2: Cartesian coordinate equation of curve C -/
theorem cartesian_eq_curve :
  ∀ (ρ θ : ℝ), ρ = polar_curve θ → ρ ^ 2 = 2 * ρ * Real.cos θ + 2 * ρ * Real.sin θ -/
  ∃ (x y : ℝ), (Real.sqrt (x^2 + y^2) = ρ) ∧ (ρ * Real.cos θ = x) ∧ (ρ * Real.sin θ = y) ∧ (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

/-- Theorem 3: Maximum distance from a point on curve C to line l -/
theorem max_distance_point_to_line :
  ∃ (P : ℝ × ℝ), (let (x, y) := P in
  ((x - 1)^2 + (y - 1)^2 = 2) → (∃ t : ℝ, parametric_line t = (x, y)) → 2 * Real.sqrt 2 = Real.abs ((x + y - 4) / Real.sqrt 2)) :=
sorry

end standard_eq_line_cartesian_eq_curve_max_distance_point_to_line_l729_729199


namespace prob_five_coins_heads_or_one_tail_l729_729467

theorem prob_five_coins_heads_or_one_tail : 
  (∃ (H T : ℚ), H = 1/32 ∧ T = 31/32 ∧ H + T = 1) ↔ 1 = 1 :=
by sorry

end prob_five_coins_heads_or_one_tail_l729_729467


namespace probability_even_heads_after_60_tosses_l729_729742

noncomputable def P : ℕ → ℝ
| 0     := 1
| (n+1) := 3/4 - 1/2 * P n

theorem probability_even_heads_after_60_tosses :
  P 60 = 1/2 * (1 + 1/(2^60)) :=
sorry

end probability_even_heads_after_60_tosses_l729_729742


namespace polyhedron_volume_inscribed_sphere_l729_729253

-- Define the inputs: a polyhedron with an inscribed sphere
variables {P : Type} [Polyhedron P] (R : ℝ) -- R is the radius of the inscribed sphere
variables (S : ℝ) -- S is the surface area of the polyhedron

-- The theorem statement
theorem polyhedron_volume_inscribed_sphere (h : has_inscribed_sphere P R) : 
  volume P = (1 / 3) * R * S := sorry

end polyhedron_volume_inscribed_sphere_l729_729253


namespace find_first_divisor_l729_729066

theorem find_first_divisor (x : ℕ) (k m : ℕ) (h₁ : 282 = k * x + 3) (h₂ : 282 = 9 * m + 3) : x = 31 :=
sorry

end find_first_divisor_l729_729066


namespace quadratic_function_count_is_100_l729_729131

noncomputable def countQuadraticFunctions : Nat :=
  let possibilitiesA := {0, 1, 2, 3, 4}.erase 0 -- removing 0 from the set for a
  let possibilitiesB := {0, 1, 2, 3, 4}
  let possibilitiesC := {0, 1, 2, 3, 4}
  possibilitiesA.card * possibilitiesB.card * possibilitiesC.card

theorem quadratic_function_count_is_100 : countQuadraticFunctions = 100 := by
  sorry

end quadratic_function_count_is_100_l729_729131


namespace eval_expr_l729_729801

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729801


namespace four_digit_integers_count_l729_729521

-- Define conditions for the digits
def is_valid_first_two_digits (d1 d2 : ℕ) : Prop :=
  (d1 = 2 ∨ d1 = 3 ∨ d1 = 6) ∧ (d2 = 2 ∨ d2 = 3 ∨ d2 = 6) ∧ (d1 + d2) % 2 = 0

def is_valid_last_two_digits (d3 d4 : ℕ) : Prop :=
  (d3 = 5 ∨ d3 = 7 ∨ d3 = 8) ∧ (d4 = 5 ∨ d4 = 7 ∨ d4 = 8) ∧ d3 ≠ d4

def count_valid_4_digit_integers : ℕ :=
  (Σ d1 (h1 : d1 = 2 ∨ d1 = 3 ∨ d1 = 6),
  Σ d2 (h2 : d2 = 2 ∨ d2 = 3 ∨ d2 = 6),
  Σ d3 (h3 : d3 = 5 ∨ d3 = 7 ∨ d3 = 8),
  Σ d4 (h4 : d4 = 5 ∨ d4 = 7 ∨ d4 = 8),
    if (is_valid_first_two_digits d1 d2 ∧ is_valid_last_two_digits d3 d4)
         then 1 else 0) 

-- The theorem stating the total number of 4-digit integers
theorem four_digit_integers_count : count_valid_4_digit_integers = 18 :=
sorry

end four_digit_integers_count_l729_729521


namespace minimum_distance_minimum_distance_achieved_l729_729118

theorem minimum_distance (a : ℝ) (h : a > 0) : abs ((16 / a) + a^2) ≥ 12 :=
by {
  sorry
}

theorem minimum_distance_achieved : ∃ a : ℝ, a > 0 ∧ abs ((16 / a) + a^2) = 12 :=
by {
  use 2,
  split,
  { exact by norm_num },
  { exact by norm_num }
}

end minimum_distance_minimum_distance_achieved_l729_729118


namespace value_standard_deviations_less_than_mean_l729_729281

-- Definitions of the given conditions
def mean : ℝ := 15
def std_dev : ℝ := 1.5
def value : ℝ := 12

-- Lean 4 statement to prove the question
theorem value_standard_deviations_less_than_mean :
  (mean - value) / std_dev = 2 := by
  sorry

end value_standard_deviations_less_than_mean_l729_729281


namespace correct_statement_C_l729_729684

def V_m_rho_relation (V m ρ : ℝ) : Prop :=
  V = m / ρ

theorem correct_statement_C (V m ρ : ℝ) (h : ρ ≠ 0) : 
  ((∃ k : ℝ, k = ρ ∧ ∀ V' m' : ℝ, V' = m' / k → V' ≠ V) ∧ 
  (∃ v_var v_var', v_var = V ∧ v_var' = m ∧ V = m / ρ) →
  (∃ ρ_const : ℝ, ρ_const = ρ)) :=
by
  sorry

end correct_statement_C_l729_729684


namespace number_of_scalene_triangles_with_prime_side_l729_729068

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_scalene_triangle_with_prime_side (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ is_prime a ∨ is_prime b ∨ is_prime c ∧
  a + b + c < 16 ∧ a + b > c ∧ b + c > a ∧ a + c > b

theorem number_of_scalene_triangles_with_prime_side :
  { (a, b, c) : ℕ × ℕ × ℕ // is_scalene_triangle_with_prime_side a b c }.card = 4 :=
sorry

end number_of_scalene_triangles_with_prime_side_l729_729068


namespace min_value_a_l729_729176

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, a + cos x ≥ 0) → a ≥ 1 :=
by 
  sorry

end min_value_a_l729_729176


namespace evaluate_expression_l729_729928

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729928


namespace triangle_altitude_theorem_l729_729285

variable {A B C A' B' C' A'' B'' C'' : Type}
variable [metric_space A] [metric_space B] [metric_space C] 
variable [metric_space A'] [metric_space B'] [metric_space C'] 
variable [metric_space A''] [metric_space B''] [metric_space C'']
variable [has_dist A] [has_dist B] [has_dist C]
variable [has_dist A'] [has_dist B'] [has_dist C']
variable [has_dist A''] [has_dist B''] [has_dist C'']

-- Providing definitions to hold distances (these could be switched to specific types if required)
variables (dist_AB dist_AC dist_BC : ℝ)
variables (dist_A'B' dist_A'C' dist_B'C' : ℝ)
variables (dist_A''B'' dist_A''C'' dist_B''C'' : ℝ)

def dist {X : Type} [has_dist X] (x y : X) : ℝ := sorry

-- Condition definitions
def altitude_feet (ABC : Type) (A' B' C' : Type) : Prop := sorry
def symmetric_points (A' B' C' : Type) (A'' B'' C'' : Type) : Prop := sorry

theorem triangle_altitude_theorem
  (h1 : altitude_feet A B C A' B' C')
  (h2 : symmetric_points A' B' C' A'' B'' C'') :
  (5 * (dist A' B')^2 - (dist A'' B'')^2) / (dist A B)^2 +
  (5 * (dist A' C')^2 - (dist A'' C'')^2) / (dist A C)^2 +
  (5 * (dist B' C')^2 - (dist B'' C'')^2) / (dist B C)^2 = 3 :=
sorry

end triangle_altitude_theorem_l729_729285


namespace max_elements_in_R_l729_729229

open Complex

-- Define the conditions as Lean 4 definitions
def nonzero_complex (z : ℂ) : Prop := z ≠ 0

def not_real_ratio (a b : ℂ) : Prop := ¬ ((a / b).im = 0)

def L (a b : ℂ) : set ℂ := { z | ∃ r s : ℤ, z = r • a + s • b }

def R (a b : ℂ) : set ℂ := { z | nonzero_complex z ∧ L a b = L (z • a) (z • b) }

-- The theorem statement
theorem max_elements_in_R (a b : ℂ) (h1 : nonzero_complex a) (h2 : nonzero_complex b) (h3 : not_real_ratio a b) :
  ∃ n : ℕ, n = finset.card (R a b) ∧ (∀ m : ℕ, m = finset.card (R a b) → m ≤ 6) ∧ (n = 6) :=
sorry

end max_elements_in_R_l729_729229


namespace value_of_2_neg_y_l729_729171

theorem value_of_2_neg_y (y : ℚ) (h : 128 ^ 3 = 16 ^ y) :
  2 ^ (-y) = 1 / 2 ^ (21/4) :=
sorry

end value_of_2_neg_y_l729_729171


namespace complete_square_correct_l729_729270

theorem complete_square_correct (x : ℝ) : (x^2 - 6*x + 1 = 0) → ((x - 3)^2 = 8) :=
begin
  sorry
end

end complete_square_correct_l729_729270


namespace bear_food_per_victor_l729_729384

theorem bear_food_per_victor (bear_food_per_day : ℕ) (victor_weight : ℕ) (days_in_week : ℕ) (weeks : ℕ)
  (H1 : bear_food_per_day = 90)
  (H2 : victor_weight = 126)
  (H3 : days_in_week = 7)
  (H4 : weeks = 3) :
  (bear_food_per_day * (days_in_week * weeks)) / victor_weight = 15 :=
by
  rw [H1, H2, H3, H4]
  norm_num
  sorry

end bear_food_per_victor_l729_729384


namespace eval_expression_l729_729908

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729908


namespace line_segment_no_intersection_l729_729533

theorem line_segment_no_intersection (a : ℝ) :
  (¬ ∃ t : ℝ, (0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * (3 : ℝ) + t * (1 : ℝ) = 2 ∧ (1 - t) * (1 : ℝ) + t * (2 : ℝ) = (2 - (1 - t) * (3 : ℝ)) / a)) ->
  (a < -1 ∨ a > 0.5) :=
by
  sorry

end line_segment_no_intersection_l729_729533


namespace sum_n_k_eq_eight_l729_729634

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l729_729634


namespace a_minus_b_plus_c_eq_five_l729_729139

theorem a_minus_b_plus_c_eq_five
(a b c : ℝ)
(h1 : a + b + c = 1)
(h2 : 3 * (4 * a + 2 * b + c) = 15)
(h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a - b + c = 5 := 
by 
  sorry

end a_minus_b_plus_c_eq_five_l729_729139


namespace max_weight_of_flock_l729_729280

def MaxWeight (A E Af: ℕ): ℕ := A * 5 + E * 10 + Af * 15

theorem max_weight_of_flock :
  ∀ (A E Af: ℕ),
    A = 2 * E →
    Af = 3 * A →
    A + E + Af = 120 →
    MaxWeight A E Af = 1415 :=
by
  sorry

end max_weight_of_flock_l729_729280


namespace evaluate_expression_l729_729875

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729875


namespace least_product_ab_l729_729100

theorem least_product_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (1 : ℚ) / a + 1 / (3 * b) = 1 / 6) : a * b ≥ 48 :=
by
  sorry

end least_product_ab_l729_729100


namespace Tiffany_time_less_than_Mary_l729_729376

-- Define the problem variables and conditions as Lean statements
variables (Mary Susan Jen Tiffany : ℕ)

-- Conditions based on the problem statement
axiom cond1 : Mary = 2 * Susan
axiom cond2 : Susan = Jen + 10
axiom cond3 : Jen = 30
axiom cond4 : Mary + Susan + Jen + Tiffany = 223

-- Define the theorem
theorem Tiffany_time_less_than_Mary : (Mary - Tiffany) = 7 :=
by
  -- Including all axioms as assumptions
  have h1 : Susan = 30 + 10 := cond2,
  have h2 : Mary = 2 * (30 + 10) := cond1,
  have h3 : Tiffany = 223 - (Mary + Susan + 30) := cond4,
  have Mary_time: Mary = 80 := by -- Tactic to compute Mary's time
  {
      rw [h1, cond3],
      norm_num,
  },
  have Tiffany_time: Tiffany = 73 := by -- Tactic to compute Tiffany's time
  {
      rw [h3, Mary_time, h1, cond3],
      norm_num,
  },
  rw [Mary_time, Tiffany_time],
  norm_num,
  sorry

end Tiffany_time_less_than_Mary_l729_729376


namespace triangle_angles_division_l729_729196

theorem triangle_angles_division (A B C D A' : Point)
  (h1 : altitude A D)
  (h2 : median A A')
  (h3 : angle_bisector A D)
  (h4 : angle_DIVISION (BAC) = 4 * α) :
  α = π / 8 :=
begin
  sorry
end

end triangle_angles_division_l729_729196


namespace puppy_sleep_duration_l729_729768

-- Definitions based on conditions
def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def puppy_sleep : ℕ := 2 * luke_sleep

-- Theorem stating that the puppy sleeps for 16 hours
theorem puppy_sleep_duration : puppy_sleep = 16 := by
  sorry

end puppy_sleep_duration_l729_729768


namespace sharks_counted_l729_729588

variables (day1_fish day2_fish day3_fish day4_fish total_fish sharks: ℕ)
variables (shark_percentage tuna_percentage eel_percentage marlin_percentage: ℝ)
variables (day1 day2 day3 day4: ℕ)

-- Constants based on the problem statement
noncomputable def day1_fish : ℕ := 24
noncomputable def day2_fish : ℕ := 2 * day1_fish
noncomputable def day3_fish : ℕ := (3 * day1_fish) / 2
noncomputable def day4_fish : ℕ := (36 * 65) / 100 -- Using integer division to approximate

-- Totals and percentages
noncomputable def total_fish : ℕ := day1_fish + day2_fish + day3_fish + day4_fish
noncomputable def shark_percentage : ℝ := 0.184
noncomputable def sharks : ℕ := (shark_percentage * total_fish).round

theorem sharks_counted :
  sharks = 24 :=
by
  sorry

end sharks_counted_l729_729588


namespace eval_expression_l729_729943

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729943


namespace exists_x_gg_eq_3_l729_729626

noncomputable def g (x : ℝ) : ℝ :=
if x < -3 then -0.5 * x^2 + 3
else if x < 2 then 1
else 0.5 * x^2 - 1.5 * x + 3

theorem exists_x_gg_eq_3 : ∃ x : ℝ, x = -5 ∨ x = 5 ∧ g (g x) = 3 :=
by
  sorry

end exists_x_gg_eq_3_l729_729626


namespace parallel_condition_true_for_1_and_4_l729_729224

theorem parallel_condition_true_for_1_and_4 (x y z : Type) :
  ( (x = "line" ∧ y = "line" ∧ z = "line") ∨ 
    (x = "plane" ∧ y = "plane" ∧ z = "plane") ) →
  ( x ∥ z ∧ y ∥ z → ¬(x ∥ y) ) → 
  ((x = "line" ∧ y = "line" ∧ z = "line") 
   ∨ (x = "plane" ∧ y = "plane" ∧ z = "plane") :=
by
  sorry

end parallel_condition_true_for_1_and_4_l729_729224


namespace a_b_in_E_basis_a_b_F_in_E_and_F_is_Fibonacci_l729_729694

-- Definition of the space E
def E : Type :=
{u : ℕ → ℝ // ∀ n, u(n+2) = u(n+1) + u(n)}

-- Definitions of sequences a_n and b_n
def a : ℕ → ℝ := λ n, (1 + Real.sqrt 5) / 2 ^ n
def b : ℕ → ℝ := λ n, (1 - Real.sqrt 5) / 2 ^ n

-- Statement 1: a_n and b_n belong to E
theorem a_b_in_E : (a ∈ E) ∧ (b ∈ E) := sorry

-- Statement 2: a_n and b_n form a basis of E, thus dim(E) = 2
theorem basis_a_b (u : E) : ∃ λ μ : ℝ, u = λ • a + μ • b := sorry

-- Statement 3: Determine the general term of the sequence F_n in E with F_0 = 0 and F_1 = 1
def F : ℕ → ℝ
| 0 := 0
| 1 := 1
| (n + 2) := F(n + 1) + F(n)

theorem F_in_E_and_F_is_Fibonacci : ∀ n, (F n) ∈ E ∧ F = fibonacci_sequence := sorry

end a_b_in_E_basis_a_b_F_in_E_and_F_is_Fibonacci_l729_729694


namespace cost_for_flour_for_two_cakes_l729_729762

theorem cost_for_flour_for_two_cakes 
    (packages_per_cake : ℕ)
    (cost_per_package : ℕ)
    (cakes : ℕ) 
    (total_cost : ℕ)
    (H1 : packages_per_cake = 2)
    (H2 : cost_per_package = 3)
    (H3 : cakes = 2)
    (H4 : total_cost = 12) :
    total_cost = cakes * packages_per_cake * cost_per_package := 
by 
    rw [H1, H2, H3]
    sorry

end cost_for_flour_for_two_cakes_l729_729762


namespace frustum_volume_l729_729025

/-- The volume of the frustum formed by cutting a square pyramid with 
    a base edge of 12 cm and altitude 8 cm into a smaller pyramid with 
    a base edge of 6 cm and altitude 4 cm is 336 cubic centimeters. -/
theorem frustum_volume (base_edge_orig alt_orig base_edge_small alt_small : ℝ) 
  (h_base_edge_orig : base_edge_orig = 12) (h_alt_orig : alt_orig = 8) 
  (h_base_edge_small : base_edge_small = 6) (h_alt_small : alt_small = 4) : 
  let volume_orig := (1 / 3) * (base_edge_orig ^ 2) * alt_orig,
      volume_small := (1 / 3) * (base_edge_small ^ 2) * alt_small,
      volume_frustum := volume_orig - volume_small in
  volume_frustum = 336 :=
  sorry

end frustum_volume_l729_729025


namespace nature_of_a_l729_729284

variable {a m n p q : ℤ}
variable {x : ℤ}

/-- Given expression 15x^2 + ax + 15 can be factored into linear binomials with integer coefficients -/
theorem nature_of_a (h1 : ∃ (m n p q : ℤ), (15 = m * p) ∧ (15 = n * q) ∧ (a = m * q + n * p)) : 
  ∃ k : ℤ, a = 2 * k :=
by sorry

end nature_of_a_l729_729284


namespace tangent_line_sine_at_origin_l729_729409

theorem tangent_line_sine_at_origin :
  tangent_line_at_point (λ x, Real.sin x) (0, 0) = λ x, x :=
sorry

end tangent_line_sine_at_origin_l729_729409


namespace eval_expression_l729_729912

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729912


namespace find_f_prime_zero_l729_729087

def f (x : ℝ) : ℝ :=
  x^2 + 2 * x * f'(-1)

theorem find_f_prime_zero
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + 2 * x * f'(-1))
  : f'(-1) = 2 → f'(0) = 4 :=
by
  sorry

end find_f_prime_zero_l729_729087


namespace perimeter_isosceles_triangle_l729_729287

theorem perimeter_isosceles_triangle (a : ℝ) (A B C M H K : Point)
  (isosceles_ABC : is_isosceles_triangle A B C)
  (midpoint_M : midpoint M A B)
  (perpendicular_AH : height B C A H)
  (perpendicular_MK : height M A K)
  (AK_eq_a : distance A K = a)
  (AH_eq_MK : distance A H = distance M K) :
  perimeter A B C = 20 * a :=
by
  sorry

end perimeter_isosceles_triangle_l729_729287


namespace parabola_through_point_l729_729602

theorem parabola_through_point (a b : ℝ) (ha : 0 < a) :
  ∃ f : ℝ → ℝ, (∀ x, f x = -a*x^2 + b*x + 1) ∧ f 0 = 1 :=
by
  -- We are given a > 0
  -- We need to show there exists a parabola of the form y = -a*x^2 + b*x + 1 passing through (0,1)
  sorry

end parabola_through_point_l729_729602


namespace find_x_for_f_of_one_fourth_l729_729231

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then 2^(-x) else Real.log x / Real.log 4 

-- Define the proof problem
theorem find_x_for_f_of_one_fourth : 
  ∃ x : ℝ, (f x = 1 / 4) ∧ (x = Real.sqrt 2)  :=
sorry

end find_x_for_f_of_one_fourth_l729_729231


namespace decreasing_power_function_l729_729133

variable (m : ℝ)

theorem decreasing_power_function (h₁ : m^2 - 3 = 1) (h₂ : m^2 + m - 3 < 0) : m = -2 :=
  sorry

end decreasing_power_function_l729_729133


namespace initial_blue_balls_proof_l729_729017

-- Define the main problem parameters and condition
def initial_jars (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :=
  total_balls = 18 ∧
  removed_blue = 3 ∧
  remaining_balls = total_balls - removed_blue ∧
  probability = 1/5 → 
  (initial_blue_balls - removed_blue) / remaining_balls = probability

-- Define the proof problem
theorem initial_blue_balls_proof (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :
  initial_jars total_balls initial_blue_balls removed_blue probability remaining_balls →
  initial_blue_balls = 6 :=
by
  sorry

end initial_blue_balls_proof_l729_729017


namespace subset_intersects_all_l729_729573

variable {S : Type*} [Fintype S]
variable {A : Fin 50 → Finset S}

theorem subset_intersects_all
  (h : ∀ i,  ∃ (S: Finset S), (|S| > (Fintype.card S / 2))):
  ∃ (B : Finset S), |B| ≤ 5 ∧ (∀ i, ∃ a ∈ B, a ∈ A i) :=
by
  sorry

end subset_intersects_all_l729_729573


namespace sin_six_arcsin_one_third_l729_729973

open Real

theorem sin_six_arcsin_one_third :
  sin (6 * arcsin (1 / 3)) = (191 * real.sqrt 2) / 729 :=
by
  sorry

end sin_six_arcsin_one_third_l729_729973


namespace evaluate_expression_l729_729921

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729921


namespace negation_of_proposition_l729_729251

-- Define the proposition P(x)
def P (x : ℝ) : Prop := x + Real.log x > 0

-- Translate the problem into lean
theorem negation_of_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_proposition_l729_729251


namespace cosine_identity_max_triangle_area_l729_729539

-- Part 1: Prove the equation involving cosine.
theorem cosine_identity (cos_A : ℝ) (h : cos_A = 1/3) :
  (cos ((B + C) / 2))^2 + cos (2 * A) = -4 / 9 := by
  sorry

-- Part 2: Prove the maximum area of the triangle with given conditions.
theorem max_triangle_area (a : ℝ) (cos_A : ℝ) (h1 : cos_A = 1/3) (h2 : a = sqrt 3) :
  ∃ S, S = 3 * sqrt 2 / 4 := by
  sorry

end cosine_identity_max_triangle_area_l729_729539


namespace solve_for_u_l729_729472

noncomputable def value_of_u (x : ℝ) (u : ℝ) : Prop :=
  4 * x^2 + 31 * x + u = 0

theorem solve_for_u :
  value_of_u ((-31 - real.sqrt 621) / 10) (85 / 4) :=
by
  sorry

end solve_for_u_l729_729472


namespace product_units_digit_l729_729994

theorem product_units_digit (s : Finset ℕ) (h1 : ∀ x ∈ s, x % 2 = 1) (h2 : s.card = 40) : 
  ∃ d : ℕ, d ∈ {1, 5} ∧ (∏ x in s, x) % 10 = d :=
by
  sorry

end product_units_digit_l729_729994


namespace integral_f_eq_l729_729127

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then (x + 1)^2
  else if 0 < x ∧ x ≤ 1 then real.sqrt (1 - x^2)
  else 0

theorem integral_f_eq :
  ∫ x in -1..1, f x = 1/3 + real.pi / 4 :=
  sorry

end integral_f_eq_l729_729127


namespace problem_statement_l729_729528

theorem problem_statement :
  let x := (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * (1 + 2^16) * (1 + 2^32) * (1 + 2^64) * (1 + 2^128) * (1 + 2^256)
  in x + 1 = 2 ^ 512 :=
by
  sorry

end problem_statement_l729_729528


namespace eval_expr_l729_729813

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729813


namespace average_of_numbers_l729_729363

theorem average_of_numbers : 
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1 + 1252140 + 2345) / 11 = 114391 :=
by
  sorry

end average_of_numbers_l729_729363


namespace equation_of_line_l729_729314

theorem equation_of_line (m : ℝ) (b : ℝ) (x₀ y₀ : ℝ) (line_eq : ℝ → ℝ → Prop) (h_slope : m = 2) (h_point : x₀ = 0 ∧ y₀ = 3) :
  line_eq (λ x, 2 * x + 3) (λ y, y) :=
by
  sorry

end equation_of_line_l729_729314


namespace position_of_2018_in_T100_l729_729748

def T (n : ℕ) : ℕ × ℕ → ℕ
  | (1, j) => j
  | (i, 1) => 4 * (n - (i - 1)) + 1
  | (i, n) => 4 * ((n - 1) - (i - 1) / 2) + i
  | (n, j) => 4 * ((j / 2) - 1) + n
  | (i, j) => (i - 1) * n + j

theorem position_of_2018_in_T100 (i j : ℕ) (h : i = 34 ∧ j = 95) : T 100 (34, 95) = 2018 :=
by {
  sorry
}

end position_of_2018_in_T100_l729_729748


namespace parabola_equation_proof_l729_729397

noncomputable def parabola_equation (m : ℝ) (a : ℝ) (x y : ℝ) : Prop :=
  y^2 = -4 * a * x

noncomputable def parabola_vertex (vertex_x vertex_y : ℝ) : Prop :=
  vertex_x = 0 ∧ vertex_y = 0

noncomputable def axis_of_symmetry (is_x_axis : Bool) : Prop :=
  is_x_axis = true

noncomputable def point_on_parabola (x y a : ℝ) : Prop :=
  y^2 = -4 * a * x

noncomputable def distance_to_focus (x1 y1 a : ℝ) (dist : ℝ) : Prop :=
  dist = sqrt ((x1 + a)^2 + y1^2)

theorem parabola_equation_proof : 
  ∀ (a m : ℝ),
  (parabola_vertex 0 0) ∧ 
  (axis_of_symmetry true) ∧
  (point_on_parabola (-5) m a) ∧ 
  (distance_to_focus (-5) m a 6) → 
  parabola_equation m a 1 (-4) := 
by
  sorry

end parabola_equation_proof_l729_729397


namespace attendees_on_monday_is_10_l729_729272

-- Define the given conditions
def attendees_tuesday : ℕ := 15
def attendees_wed_thru_fri : ℕ := 10
def days_wed_thru_fri : ℕ := 3
def average_attendance : ℕ := 11
def total_days : ℕ := 5

-- Define the number of people who attended class on Monday
def attendees_tuesday_to_friday : ℕ := attendees_tuesday + attendees_wed_thru_fri * days_wed_thru_fri
def total_attendance : ℕ := average_attendance * total_days
def attendees_monday : ℕ := total_attendance - attendees_tuesday_to_friday

-- State the theorem
theorem attendees_on_monday_is_10 : attendees_monday = 10 :=
by
  -- Proof omitted
  sorry

end attendees_on_monday_is_10_l729_729272


namespace determine_c_l729_729053

theorem determine_c (c : ℝ) : 
  (- ∀ x : ℝ, (-x^2 + c*x + 3 < 0 ↔ x ∈ set.Iio (-3) ∨ x ∈ set.Ioi 2)) → c = 5 := 
by
  sorry

end determine_c_l729_729053


namespace area_of_shaded_region_l729_729202

theorem area_of_shaded_region (t : ℝ) (hpos : 0 < t)
  (h_eq : ∃ A B C D : ℝ × ℝ, ¬collinear A B C ∧ circle_diagonal A C radius := t / 2 
    ∧ rect ABCD ∧ dist CD = 2 * dist AD ) : 
  let r := t / 2,
      x := sqrt (t^2 / 5),
      area_circle := π * r^2,
      area_rect := if h_rect : ∃ AD CD, dist CD = 2 * dist AD then 2 * x^2 else 0,
      area_shaded := area_circle - area_rect 
  in area_shaded = 9 * π - (72 / 5) :=
begin
  sorry
end

end area_of_shaded_region_l729_729202


namespace length_of_chord_AB_l729_729141

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

theorem length_of_chord_AB :
  ∃ A B : ℝ × ℝ, 
    circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
    circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
    dist A B = sqrt 95 := sorry

end length_of_chord_AB_l729_729141


namespace speed_of_second_train_l729_729332

def length_train : ℝ := 210
def speed_train_1_kmh : ℝ := 90
def time_to_pass_seconds : ℝ := 8.64
def total_distance : ℝ := 2 * length_train
def relative_speed_mps : ℝ := total_distance / time_to_pass_seconds
def conversion_factor : ℝ := 3.6
def relative_speed_kmh : ℝ := relative_speed_mps * conversion_factor

theorem speed_of_second_train :
  let speed_train_2_kmh := relative_speed_kmh - speed_train_1_kmh in
  speed_train_2_kmh = 85 := by
    sorry  

end speed_of_second_train_l729_729332


namespace candies_problem_l729_729400

theorem candies_problem (N : ℕ) : 
  (N ≡ 5 [MOD 2]) ∧ 
  (N ≡ 2 [MOD 3]) ∧ 
  (N ≡ 3 [MOD 5]) → 
  N = 53 :=
begin
  sorry
end

end candies_problem_l729_729400


namespace evaluate_expression_l729_729853

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729853


namespace smallest_possible_value_of_EF_minus_DE_l729_729328

theorem smallest_possible_value_of_EF_minus_DE :
  ∃ (DE EF FD : ℤ), DE + EF + FD = 2010 ∧ DE < EF ∧ EF ≤ FD ∧ 1 = EF - DE ∧ DE > 0 ∧ EF > 0 ∧ FD > 0 ∧ 
  DE + EF > FD ∧ DE + FD > EF ∧ EF + FD > DE :=
by {
  sorry
}

end smallest_possible_value_of_EF_minus_DE_l729_729328


namespace Jan_older_than_Cindy_l729_729043

noncomputable def Cindy_age : ℕ := 5
noncomputable def Greg_age : ℕ := 16

variables (Marcia_age Jan_age : ℕ)

axiom Greg_and_Marcia : Greg_age = Marcia_age + 2
axiom Marcia_and_Jan : Marcia_age = 2 * Jan_age

theorem Jan_older_than_Cindy : (Jan_age - Cindy_age) = 2 :=
by
  -- Insert proof here
  sorry

end Jan_older_than_Cindy_l729_729043


namespace plywood_cut_perimeter_difference_l729_729701

theorem plywood_cut_perimeter_difference :
  let l := 6
  let w := 9
  let num_rects := 6
  let area := l * w
  ((exists a b : ℕ, a * b = num_rects ∧ (area / b) * (area / a) = area ∧ ((2 * (area / b + (area / a))) = 18)) ∧
   (exists a b : ℕ, a * b = num_rects ∧ (area / b) * (area / a) = area ∧ ((2 * (area / b + (area / a))) = 12))) →
    18 - 12 = 6 :=
by solver sorry

end plywood_cut_perimeter_difference_l729_729701


namespace tourist_punctuality_l729_729600

def wealth_disparity (wealth : ℕ → ℕ) (group_size : ℕ) (tourist_id : ℕ) : Prop := sorry
def excursion_cost (group_size : ℕ) (cost_per_person : ℕ) : Prop := sorry
def perceived_importance (payment : ℕ) (group_size : ℕ) (tourist_id : ℕ) : Prop := sorry

theorem tourist_punctuality (group_size : ℕ) (wealth : ℕ → ℕ) (cost_per_person : ℕ) (payment : ℕ) :
  (∀ N, N = group_size ∧ N > 1 → ∀ tourist, wealth_disparity wealth group_size tourist ∧ 
       excursion_cost group_size cost_per_person ∧
       perceived_importance payment group_size tourist → tourist_is_punctual tourist) ∧
  (∀ N, N = group_size ∧ N = 1 → ∃ tourist, wealth_disparity wealth group_size tourist ∨ 
       excursion_cost group_size cost_per_person ∨
       perceived_importance payment group_size tourist ∧ ¬tourist_is_punctual tourist) :=
sorry

end tourist_punctuality_l729_729600


namespace max_gcd_14m_plus_4_9m_plus_2_l729_729414

theorem max_gcd_14m_plus_4_9m_plus_2 (m : ℕ) (h : m > 0) : ∃ M, M = 8 ∧ ∀ k, gcd (14 * m + 4) (9 * m + 2) = k → k ≤ M :=
by
  sorry

end max_gcd_14m_plus_4_9m_plus_2_l729_729414


namespace transformation_composition_l729_729192

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition_l729_729192


namespace ptolemy_hexagon_l729_729218

-- Define the variables
variables (A B C D E F : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F]

-- Assume AD, BE, CF, AB, DE, BC, EF, CD, FA are segments of a hexagon inscribed in a circle
variables (AD BE CF AB DE BC EF CD FA : ℝ)

-- Theorem statement
theorem ptolemy_hexagon (h: ∃ A B C D E F : Type, true):
  (AD * BE * CF = AB * DE * CF + BC * EF * AD + CD * FA * BE + AB * CD * EF + BC * DE * FA) :=
sorry

end ptolemy_hexagon_l729_729218


namespace part1_part2_l729_729126

def f (x a : ℝ) := abs (x - a) + abs (2 * x - 1)

theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≤ 2 → x ∈ Icc (0 : ℝ) (4 / 3) :=
sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ∈ Icc (1 / 2) 1 → f x a ≤ abs (2 * x + 1)) → a ∈ Icc (-1) (5 / 2) :=
sorry

end part1_part2_l729_729126


namespace smallest_integer_gt_neg_seven_thirds_l729_729342

theorem smallest_integer_gt_neg_seven_thirds : ∃ n : ℤ, n > -7/3 ∧ ∀ m : ℤ, m > -7/3 → n ≤ m :=
begin
  use -2,
  split,
  { norm_num },
  { intros m hm,
    norm_num at hm,
    linarith }
end

end smallest_integer_gt_neg_seven_thirds_l729_729342


namespace inequality_proof_l729_729109

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^4 * b^b * c^c ≥ a⁻¹ * b⁻¹ * c⁻¹ :=
sorry

end inequality_proof_l729_729109


namespace bond_selling_price_approx_l729_729568

noncomputable def bond_selling_price (face_value : ℝ) 
                                      (interest_rate : ℝ) 
                                      (tax_rate : ℝ) 
                                      (inflation_rate : ℝ) 
                                      (net_interest_rate : ℝ) : ℝ :=
  let interest_earned := face_value * interest_rate
  let tax_paid := interest_earned * tax_rate
  let net_interest_after_taxes := interest_earned - tax_paid
  let real_value_net_interest := net_interest_after_taxes / (1 + inflation_rate)
  real_value_net_interest / net_interest_rate

theorem bond_selling_price_approx (face_value : ℝ) 
                                   (interest_rate : ℝ) 
                                   (tax_rate : ℝ) 
                                   (inflation_rate : ℝ) 
                                   (net_interest_rate : ℝ) :
  face_value = 10000 → 
  interest_rate = 0.07 →
  tax_rate = 0.25 →
  inflation_rate = 0.035 →
  net_interest_rate = 0.045 →
  bond_selling_price face_value interest_rate tax_rate inflation_rate net_interest_rate ≈ 11272.22 :=
by
  intros
  simp only [bond_selling_price]
  sorry

end bond_selling_price_approx_l729_729568


namespace parabola_sequence_properties_l729_729594

theorem parabola_sequence_properties (p : ℝ) (y_1 y_2 : ℝ) (C1 C2 : ℝ × ℝ)
  (hC1: C1 = (-(p/2), y_1))
  (hC2: C2 = (-(p/2), y_2))
  (hy1y2: y_1 * y_2 = -p^2) :
  ∀ n : ℕ, (if n % 2 = 0 then (-(p/2), y_2) else (-(p/2), y_1)) = if n % 2 = 0 then C2 else C1 :=
begin
  assume n,
  split_ifs,
  { rw hC2 },
  { rw hC1 },
end

end parabola_sequence_properties_l729_729594


namespace robin_initial_gum_is_18_l729_729258

-- Defining the conditions as given in the problem
def given_gum : ℝ := 44
def total_gum : ℝ := 62

-- Statement to prove that the initial number of pieces of gum Robin had is 18
theorem robin_initial_gum_is_18 : total_gum - given_gum = 18 := by
  -- Proof goes here
  sorry

end robin_initial_gum_is_18_l729_729258


namespace dividend_is_correct_l729_729339

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct_l729_729339


namespace integer_values_x_possible_l729_729304

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l729_729304


namespace P_lies_on_diagonal_BD_l729_729706

variables (A B C D P : Type) 
[geometry.point A] [geometry.point B] [geometry.point C] [geometry.point D] [geometry.point P]
(circle : Type) [geometry.circle circle]

-- Assuming the geometric conditions
axiom condition1 : ∃ (parallelogram : Type), geometry.parallelogram parallelogram A B C D
axiom condition2 : ∃ (tangent_circle : circle), geometry.tangent_to_rays tangent_circle (geometry.ray B A) (geometry.ray B C)
axiom condition3 : geometry.externally_tangent_at tangent_circle P

-- The goal is to prove that P lies on the diagonal BD
theorem P_lies_on_diagonal_BD : geometry.on_diagonal P (geometry.diagonal B D) :=
sorry

end P_lies_on_diagonal_BD_l729_729706


namespace three_digit_numbers_no_repetition_l729_729669

theorem three_digit_numbers_no_repetition :
  let digits := {1, 2, 3, 4, 5} in
  (finset.filter (λ x, x.to_list.nodup) (finset.image (λ l, list.to_digits l) (finset.permutations {1, 2, 3, 4, 5}.to_list))).card = 60 := 
by
  sorry

end three_digit_numbers_no_repetition_l729_729669


namespace range_of_log_function_l729_729309

noncomputable theory
open Real BigOperators

theorem range_of_log_function :
  let f : ℝ → ℝ := λ x, log (1/2) (x^2 - 6*x + 17)
  ∃ (L U : ℝ), U = 3 ∧ (∀y, y ∈ (set.range f) ↔ y ≤ U) :=
by
  let f := λ x, log (1 / 2) (x^2 - 6*x + 17)
  use 3
  split
  { refl }
  { intro y
    split
    { intro hy
      sorry
    }
    { intro hy
      cases classic.em (∃ x, f x = y) with hx hn
      { exact set.mem_range_self _ hx}
      { sorry }
    }
  }

end range_of_log_function_l729_729309


namespace inequality_always_holds_l729_729107

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
by 
  sorry

end inequality_always_holds_l729_729107


namespace coin_toss_frequency_approaches_half_l729_729019

noncomputable def frequency_limit (n : ℕ) (m : ℕ) (P : ℝ) : Prop :=
  P = (m : ℝ) / (n : ℝ) →

open_locale classical
open filter

theorem coin_toss_frequency_approaches_half : 
  ∀ (P : ℕ → ℝ) (n : ℕ),
    (∀ n, P n = n / 2) → 
    tendsto P at_top (nhds (1 / 2)) :=
begin
  intros P n hP,
  sorry
end

end coin_toss_frequency_approaches_half_l729_729019


namespace committee_selection_l729_729084

-- Definitions corresponding to the conditions:
-- Representation of Jiǎ, Yǐ, Bǐng, and Dīng as A, B, C, and D respectively.
-- Total number of students is 9, number of students to select is 5, 
-- with given constraints on selection.

variable (A B C D : Type) -- Representing the students A, B, C, D
variable (students : Finset (Type)) -- Representing the set of all students
variable (nine_students : students.card = 9) -- There are 9 students in total
variable (committee_size : ℕ := 5) -- Committee size is 5 students
variable (exclude_CD : ∀ selection : Finset students, C ∈ selection ∧ D ∈ selection → False)
variable (in_or_out_AB : ∀ selection : Finset students, A ∈ selection ↔ B ∈ selection)

-- The goal is to find the number of valid ways to form the committee which is 41.
theorem committee_selection : Finset.card {s : Finset students | s.card = committee_size ∧
  (A ∈ s ↔ B ∈ s) ∧ (C ∈ s ∧ D ∈ s → False)} = 41 := 
  sorry

end committee_selection_l729_729084


namespace eval_expr_l729_729805

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729805


namespace evaluate_expression_l729_729884

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729884


namespace average_weight_b_c_l729_729282

theorem average_weight_b_c (A B C : ℝ) (h1 : A + B + C = 126) (h2 : A + B = 80) (h3 : B = 40) : 
  (B + C) / 2 = 43 := 
by 
  -- Proof would go here, but is left as sorry as per instructions
  sorry

end average_weight_b_c_l729_729282


namespace triangle_side_count_l729_729290

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l729_729290


namespace curve_equation_and_m_range_l729_729503

-- Define the distance conditions given in the problem
def point_on_curve_condition (P : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  let x := P.1 in
  let y := P.2 in
  let Fx := F.1 in
  let Fy := F.2 in
  (real.sqrt (x^2 + (y - Fy)^2)) - |y| = 1

-- Define curve C
def curve_C (P : ℝ × ℝ) : Prop :=
  let x := P.1 in
  let y := P.2 in
  (y >= 0 ∧ x^2 = 4 * y) ∨ (y < 0 ∧ x = 0)

-- Define the range of m such that for any k in R, the dot product of FA and FB is less than 0
def valid_m_range (m : ℝ) : Prop :=
  3 - 2 * real.sqrt 2 < m ∧ m < 3 + 2 * real.sqrt 2

-- The Lean theorem statement encompassing the math proofs
theorem curve_equation_and_m_range (F : ℝ × ℝ) (m : ℝ) (k : ℝ) (P : ℝ × ℝ) :
  F = (0, 1) →
  (∀ P : ℝ × ℝ, point_on_curve_condition P F → curve_C P) →
  (m > 0 ∧ ∀ k : ℝ, ∃ P1 P2 : ℝ × ℝ, P1.2 = k * P1.1 + m ∧ P2.2 = k * P2.1 + m ∧ curve_C P1 ∧ curve_C P2 ∧
    ((P1.1 - F.1) * (P2.1 - F.1) + (P1.2 - F.2) * (P2.2 - F.2) < 0)) →
  valid_m_range m :=
begin
  intros hF hCurve hDotProduct,
  sorry -- Proof goes here
end

end curve_equation_and_m_range_l729_729503


namespace frequency_not_equal_probability_l729_729394

theorem frequency_not_equal_probability
  (N : ℕ) -- Total number of trials
  (N1 : ℕ) -- Number of times student A is selected
  (hN : N > 0) -- Ensure the number of trials is positive
  (rand_int_gen : ℕ → ℕ) -- A function generating random integers from 1 to 6
  (h_gen : ∀ n, 1 ≤ rand_int_gen n ∧ rand_int_gen n ≤ 6) -- Generator produces numbers between 1 to 6
: (N1/N : ℚ) ≠ (1/6 : ℚ) := 
sorry

end frequency_not_equal_probability_l729_729394


namespace contest_sequences_l729_729545

-- Define the conditions
def total_contestants : ℕ := 5
def females : ℕ := 3
def males : ℕ := 2

-- Conditions:
def males_non_consecutive : Prop := sorry  -- Needs a precise mathematical definition or formulation.
def female_A_not_first : Prop := sorry    -- Needs a precise mathematical definition or formulation.

-- Define the proof problem
theorem contest_sequences:
    total_contestants = 5 →
    females = 3 →
    males = 2 →
    males_non_consecutive →
    female_A_not_first →
    (Σ (ways : ℕ), ways = 60) :=
by
  sorry

end contest_sequences_l729_729545


namespace evaluate_expression_l729_729858

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729858


namespace evaluate_expression_l729_729854

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729854


namespace C_share_of_profit_l729_729727

-- Definitions corresponding to the conditions in the problem
def investment_relation_A_B (x : ℝ) : ℝ := 5 * x
def investment_relation_A_C (x : ℝ) : ℝ := (3 / 5) * (investment_relation_A_B x)

def A_share (x : ℝ) : ℝ := (investment_relation_A_B x) * 6
def B_share (x : ℝ) : ℝ := x * 9
def C_share (x : ℝ) : ℝ := (investment_relation_A_C x) * 12

def total_ratio (x : ℝ) : ℝ := A_share x + B_share x + C_share x

-- The proof problem as a Lean 4 statement
theorem C_share_of_profit (x : ℝ) (profit : ℝ) (hx : x ≠ 0) (hp : profit = 110000) :
  (C_share x / total_ratio x) * profit ≈ 79136.57 := by
  sorry

end C_share_of_profit_l729_729727


namespace consecutive_sum_ways_l729_729547

theorem consecutive_sum_ways (S : ℕ) (hS : S = 385) :
  ∃! n : ℕ, ∃! k : ℕ, n ≥ 2 ∧ S = n * (2 * k + n - 1) / 2 :=
sorry

end consecutive_sum_ways_l729_729547


namespace no_good_integers_l729_729740

theorem no_good_integers : ∀ (n : ℕ), n ≥ 1 → ¬ (∀ (k : ℕ), (k > 0 → (∀ i ∈ (Finset.range 9).image (λ x, n + x + 1), i ∣ k) → (n + 10) ∣ k)) :=
by
  assume n hn,
  assume h : ∀ (k : ℕ), k > 0 → (∀ i ∈ (Finset.range 9).image (λ x, n + x + 1), i ∣ k) → (n + 10) ∣ k,
  have hc : ¬ (∃ m, m ∈ (Finset.range 9).image (λ x, n + x + 1) ∧ m = n + 10),
  { simp only [Finset.mem_image, exists_prop, Finset.mem_range],
    assume ⟨m, hm1, hm2⟩,
    have hm : m < 9, from hm1,
    linarith, },
  sorry

end no_good_integers_l729_729740


namespace sum_b_n_l729_729513

def seq_a : ℕ → ℚ
| 1     := 1 / 2
| 2     := 1 / 3 + 2 / 3
| 3     := 1 / 4 + 2 / 4 + 3 / 4
| 4     := 1 / 5 + 2 / 5 + 3 / 5 + 4 / 5
| (n+1) := (1 + 2 + ... + n) / (n + 1) := sorry  -- For demonstration; need formal def.

def seq_b : ℕ → ℚ
| n := 1 / (seq_a n * seq_a (n + 1))

def sum_b (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, seq_b (k + 1)

theorem sum_b_n (n : ℕ) : sum_b n = 4 * (n / (n + 1)) :=
by
  unfold sum_b seq_b seq_a
  sorry

end sum_b_n_l729_729513


namespace probability_three_heads_in_eight_tosses_l729_729005

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729005


namespace digits_right_of_decimal_l729_729161

theorem digits_right_of_decimal :
  let expr := (2^7 * 3^5) / (6^4 * 5^3 * 7^2 : ℚ) in
  nat_digits_right_of_decimal expr = 9 :=
by sorry

def nat_digits_right_of_decimal (x : ℚ) : ℕ := 
  sorry

end digits_right_of_decimal_l729_729161


namespace anna_phone_chargers_l729_729746

theorem anna_phone_chargers (p l : ℕ) (h₁ : l = 5 * p) (h₂ : l + p = 24) : p = 4 :=
by
  sorry

end anna_phone_chargers_l729_729746


namespace find_B_l729_729343

-- Definitions
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

-- Statement of the problem in Lean
theorem find_B : 
  ∀ (A7B B : ℕ), 
    A7B + 23 = 695 ∧ is_three_digit (27 * B) ∧ (A7B % 10 = 2) → B = 2 :=
by 
  intros A7B B h,
  cases h,
  cases h_right,
  cases h_right_right,
  sorry

end find_B_l729_729343


namespace sum_max_value_l729_729469

theorem sum_max_value:
  ∀ (x : Fin 60 → ℝ),
  (∀ i, x i ∈ set.Icc (-1 : ℝ) 1) →
  (x 0 = x 59 ∧ x 60 = x 1) →
  ∃ (y : ℝ), y = 40 ∧ 
  ∀ (s : ℝ), (s = ∑ i in Finset.range 60, (x i)^2 * (x (i+1) mod 60 - x (i+59) mod 60)) →
  s ≤ y :=
by
  sorry

end sum_max_value_l729_729469


namespace evaluate_expression_l729_729931

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729931


namespace son_age_l729_729391

theorem son_age {x : ℕ} {father son : ℕ} 
  (h1 : father = 4 * son)
  (h2 : (son - 10) + (father - 10) = 60)
  (h3 : son = x)
  : x = 16 := 
sorry

end son_age_l729_729391


namespace PS_length_l729_729207

noncomputable def length_PS (P Q R S : Point)
  (hPQ : distance P Q = 4)
  (hPR : distance P R = 3)
  (hRight : right_angle P Q R)
  (hBisect : is_angle_bisector Q R P Q S) : ℚ :=
distance P S = 12 / 7

theorem PS_length (P Q R S : Point)
  (hPQ: distance P Q = 4)
  (hPR: distance P R = 3)
  (hAngle: right_angle P Q R)
  (hBisector: is_angle_bisector Q R P Q S) :
  distance P S = 12 / 7 := 
sorry

end PS_length_l729_729207


namespace largest_power_of_5_in_factorial_sum_l729_729987

theorem largest_power_of_5_in_factorial_sum :
  ∀ (M : ℕ), 
  M > 0 → 
  (M! + (M+1)! + (M+2)!) = 5^n * k → 
  k % 5 ≠ 0 → 
  M = 105 → 
  n = 25 :=
by
  sorry

end largest_power_of_5_in_factorial_sum_l729_729987


namespace race_times_l729_729193

theorem race_times :
  ∃ (patrick_time manu_time amy_time olivia_time : ℕ),
    patrick_time = 60 ∧
    manu_time = patrick_time + 12 ∧
    amy_time = manu_time / 2 ∧
    olivia_time = 2 * amy_time / 3 ∧
    manu_time = 72 ∧
    amy_time = 36 ∧
    olivia_time = 24 :=
by
  let patrick_time := 60
  have h1 : manu_time = patrick_time + 12 := by simp [patrick_time]
  let manu_time := 72
  have h2 : amy_time = manu_time / 2 := by simp [manu_time]
  let amy_time := 36
  have h3 : olivia_time = 2 * amy_time / 3 := by simp [amy_time]
  let olivia_time := 24
  simp [patrick_time, manu_time, amy_time, olivia_time]
  sorry

end race_times_l729_729193


namespace abs_diff_roots_l729_729050

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Conditions from the problem
variables (u v : ℝ)
axiom roots_properties :
  quadratic_eq 1 (-7) 9 u ∧ quadratic_eq 1 (-7) 9 v ∧ (u + v = 7) ∧ (u * v = 9)

-- The statement to prove
theorem abs_diff_roots :
  |u - v| = real.sqrt 13 :=
by
  sorry

end abs_diff_roots_l729_729050


namespace correct_conclusions_l729_729125

theorem correct_conclusions:
  (∀ a : ℝ, a < 0 → (a^2)^(3/2) = a^3) ∨
  (∀ (n: ℕ) (a: ℝ), n > 1 ∧ n % 2 = 0 → n * a^n = |a|) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≠ 7 / 3 → (x - 2)^(1/2) - (3 * x - 7)^0 ∈ {x | x ≥ 2 ∧ x ≠ 7 / 3}) ∧
  (exists x y : ℝ, 2^x = 16 ∧ 3^y = 1/27 → x + y = 7) → 
  ({2, 3} := true) :=
by
  sorry

end correct_conclusions_l729_729125


namespace probability_three_heads_in_eight_tosses_l729_729007

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729007


namespace eval_expr_l729_729806

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729806


namespace cost_of_gravelling_roads_l729_729403

theorem cost_of_gravelling_roads :
  let lawn_length := 70
  let lawn_breadth := 30
  let road_width := 5
  let cost_per_sqm := 4
  let area_road_length := lawn_length * road_width
  let area_road_breadth := lawn_breadth * road_width
  let area_intersection := road_width * road_width
  let total_area_to_be_graveled := (area_road_length + area_road_breadth) - area_intersection
  let total_cost := total_area_to_be_graveled * cost_per_sqm
  total_cost = 1900 :=
by
  sorry

end cost_of_gravelling_roads_l729_729403


namespace eval_expression_l729_729819

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729819


namespace color_line_plane_l729_729113

theorem color_line_plane (colors : Fin 5 → Point) (distinct_colors : ∀ i j : Fin 5, i ≠ j → colors i ≠ colors j) : 
  (∃ l : Line, ∃ c₁ c₂ c₃ : Color, c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃ ∧ 
  ∀ p : Point, on_line p l → color_of p = c₁ ∨ color_of p = c₂ ∨ color_of p = c₃) ∧
  (∃ p : Plane, ∃ c₁ c₂ c₃ c₄ : Color, c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄ ∧ 
  ∀ q : Point, on_plane q p → color_of q = c₁ ∨ color_of q = c₂ ∨ color_of q = c₃ ∨ color_of q = c₄) :=
sorry

end color_line_plane_l729_729113


namespace equal_circumcenter_distances_l729_729570

theorem equal_circumcenter_distances
  (ABC : Triangle)
  (O : Point)
  (P : Point)
  (D E F : Point)
  (R S T : Point)
  (Q : Point)
  (h1 : circumcenter ABC = O)
  (h2 : P ∈ interior ABC)
  (h3 : on_line_segment D BC)
  (h4 : on_line_segment E AC)
  (h5 : on_line_segment F AB)
  (h6 : miquel_point ABC D E F = P)
  (h7 : reflection_over_midpoint D BC = R)
  (h8 : reflection_over_midpoint E AC = S)
  (h9 : reflection_over_midpoint F AB = T)
  (h10 : miquel_point ABC R S T = Q) :
  distance O P = distance O Q := sorry

end equal_circumcenter_distances_l729_729570


namespace find_C_coordinates_l729_729603

noncomputable def point := (ℝ × ℝ)

variables A B D C : point

theorem find_C_coordinates (A B D : point) (hA : A = (8, 7)) (hB : B = (-1, 2)) (hD : D = (-4, 5)) (hMidpoint : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) : 
  C = (-7, 8) := 
sorry

end find_C_coordinates_l729_729603


namespace eval_expr_l729_729811

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729811


namespace triangle_sides_and_angles_l729_729039

theorem triangle_sides_and_angles (R : ℝ) (alpha : ℝ) (h_diff : ℝ) :
  R = 8.5 → alpha = 47 * (π / 180) → h_diff = 2.7 →
  ∃ (a b c : ℝ) (alpha beta gamma : ℝ),
    a = 12.433 ∧
    b = 13.156 ∧
    c = 16.847 ∧
    alpha = 47 * (π / 180) ∧
    beta = (50 + 41 / 60 + 56 / 3600) * (π / 180) ∧
    gamma = (82 + 18 / 60 + 4 / 3600) * (π / 180) :=
begin
  sorry
end

end triangle_sides_and_angles_l729_729039


namespace fibonacci_mod_9_l729_729049

def A : ℕ → ℕ
| 0       := 2
| 1       := 5
| (n + 2) := A n + A (n + 1)

theorem fibonacci_mod_9 : (A 49) % 9 = 5 := 
by
  sorry

end fibonacci_mod_9_l729_729049


namespace evaluate_expression_l729_729848

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729848


namespace exists_number_divisible_by_24_and_cube_root_in_range_l729_729453

noncomputable def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

theorem exists_number_divisible_by_24_and_cube_root_in_range :
  ∃ n : ℕ, is_divisible_by n 24 ∧ (8 < real.cbrt n) ∧ (real.cbrt n < 8.2) ∧ (n = 528) :=
begin
  sorry
end

end exists_number_divisible_by_24_and_cube_root_in_range_l729_729453


namespace range_of_z2_minus_z1_l729_729500

noncomputable def complex_norm (z : Complex) := Complex.abs z

theorem range_of_z2_minus_z1
    (x y : Real)
    (hx : x ∈ Set.Icc (-Real.sqrt (Real.pi / 2)) (Real.sqrt (Real.pi / 2)))
    (hy : y ∈ Set.Icc (-Real.sqrt (Real.pi / 2)) (Real.sqrt (Real.pi / 2)))
    (hxy : x^2 + y^2 = Real.pi / 2) :
    let z1 := Complex.mk (Real.cos (x^2) / Real.sin (y^2)) (Real.cos (y^2) / Real.sin (x^2))
    let z2 := Complex.mk x y
    (complex_norm z1 = Real.sqrt 2) →
    complex_norm (z2 - z1) ∈
    Set.union (Set.union
        (Set.Icc (Real.sqrt 2 - Real.sqrt (Real.pi / 2)) (Real.sqrt (2 - Real.sqrt (2 * Real.pi) + Real.pi / 2)))
        (Set.Icc (Real.sqrt (2 - Real.sqrt (2 * Real.pi) + Real.pi / 2)) (Real.sqrt (2 + Real.sqrt (2 * Real.pi) + Real.pi / 2))))
        (Set.Icc (Real.sqrt (2 + Real.sqrt (2 * Real.pi) + Real.pi / 2)) (Real.sqrt 2 + Real.sqrt (Real.pi / 2)))) :=
  sorry

end range_of_z2_minus_z1_l729_729500


namespace eval_expression_l729_729834

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729834


namespace only_setB_forms_triangle_l729_729681

/-- Check if three line segments can form a triangle using the triangle inequality theorem -/
def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Options as sets of three line segments -/
def setA := (2, 4, 6)
def setB := (3, 5, 7)
def setC := (4, 5, 10)
def setD := (3, 3, 8)

/-- Prove that among the sets given, only setB can form a triangle -/
theorem only_setB_forms_triangle :
  canFormTriangle setB.1 setB.2 setB.3 ∧
  ¬ canFormTriangle setA.1 setA.2 setA.3 ∧
  ¬ canFormTriangle setC.1 setC.2 setC.3 ∧
  ¬ canFormTriangle setD.1 setD.2 setD.3 :=
by
  sorry

end only_setB_forms_triangle_l729_729681


namespace prime_sum_probability_l729_729029

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_probability :
  let primes := {2, 3, 5, 7, 11, 13}
  let pairs := ({(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (3, 5), (3, 7), (3, 11), (3, 13), (5, 7), 
                  (5, 11), (5, 13), (7, 11), (7, 13), (11, 13)} : set (ℕ × ℕ))
  let favorable_pairs := { (2, 3), (2, 5), (2, 11) }
  ∃ (P : ℝ), P = ↑(favorable_pairs.to_finset.card) / ↑(pairs.to_finset.card) ∧ P = 1 / 5 :=
by
  sorry

end prime_sum_probability_l729_729029


namespace initial_number_of_men_l729_729617

theorem initial_number_of_men (W : ℝ) (M : ℝ) (h1 : (M * 15) = W / 2) (h2 : ((M - 2) * 25) = W / 2) : M = 5 :=
sorry

end initial_number_of_men_l729_729617


namespace domain_of_function_l729_729780

noncomputable def function_domain : Set ℝ :=
  {x : ℝ | (1 - x^2 ≥ 0) ∧ (2 * x^2 - 3 * x - 2 ≠ 0)}

theorem domain_of_function :
  function_domain = {x : ℝ | (-1 ≤ x ∧ x ≤ 1) ∧ (x ≠ 2) ∧ (x ≠ -1 / 2)} :=
begin
  -- proof goes here
  sorry
end

end domain_of_function_l729_729780


namespace probability_three_heads_in_eight_tosses_l729_729001

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l729_729001


namespace elmer_more_than_penelope_l729_729245

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l729_729245


namespace powers_of_2_not_powers_of_4_l729_729162

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end powers_of_2_not_powers_of_4_l729_729162


namespace range_of_eccentricity_l729_729122

theorem range_of_eccentricity 
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2))
  (N : ℝ × ℝ) (hN : N = (3 * c / 2, Real.sqrt(2) * c / 2))
  (M : ℝ × ℝ) (hM : M ∈ {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1 })
  (h_condition : ∀ M, (M ∈ {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1 }) → (dist M (F1 a b c) + dist M N) < 4 * Real.sqrt(3) * c)
  (e : ℝ) (he : e = c / a) : 
  (4 * Real.sqrt(3) / 21 < e) ∧ (e < Real.sqrt(3) / 3) :=
sorry

end range_of_eccentricity_l729_729122


namespace find_dot_product_l729_729232

variable {V : Type*} [InnerProductSpace ℝ V]

variables (a b : V)
-- Given conditions
def magnitude_a_eq_2 : ∥a∥ = 2 := sorry
def dot_a_b_eq_1 : ⟪a, b⟫ = 1 := sorry

-- The theorem to prove
theorem find_dot_product : ⟪a, (2 • a + b)⟫ = 9 :=
by
  have mag_sq_a : ⟪a, a⟫ = 4 := by
    rw [←norm_sq_eq_inner]
    simp [magnitude_a_eq_2]
  rw [inner_add_left, inner_smul_left, dot_a_b_eq_1, mag_sq_a]
  norm_num

end find_dot_product_l729_729232


namespace eval_expression_l729_729951

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729951


namespace existence_of_lambda_smallest_lambda_l729_729575

variables {a b c : ℝ}

def d (a b c : ℝ) : ℝ := min ((a - b) ^ 2) (min ((b - c) ^ 2) ((c - a) ^ 2))

noncomputable def exists_lambda (a b c : ℝ) (d : ℝ) : Prop :=
    ∃ (λ : ℝ), 0 < λ ∧ λ < 1 ∧ d ≤ λ * (a^2 + b^2 + c^2)

noncomputable def lambda_min (a b c : ℝ) (d : ℝ) : ℝ :=
    if h : d ≤ (1/5) * (a^2 + b^2 + c^2) then (1/5 : ℝ) else d 

theorem existence_of_lambda (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) : 
  exists (λ : ℝ), 0 < λ ∧ λ < 1 ∧ d a b c ≤ λ * (a^2 + b^2 + c^2) :=
sorry

theorem smallest_lambda (a b c : ℝ) (d : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) : 
  lambda_min a b c d = 1 / 5 :=
sorry

end existence_of_lambda_smallest_lambda_l729_729575


namespace number_of_elements_in_A_l729_729585

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def A : Set (ℕ × ℕ × ℕ) :=
  { p | p.1 ∈ M ∧ p.2 ∈ M ∧ p.3 ∈ M ∧ (x^3 + y^3 + z^3) ∣ 9 }

theorem number_of_elements_in_A : 
  (#A = 243) :=
sorry

end number_of_elements_in_A_l729_729585


namespace remainder_when_closest_even_to_sum_S_divided_by_5_is_1_l729_729420

noncomputable def sum_S : ℝ :=
  ∑ k in (Finset.range 670).filter (λ n, (3 * n + 2) ≤ 2012),
    2014 / ((3 * k + 2) * (3 * k + 5))

theorem remainder_when_closest_even_to_sum_S_divided_by_5_is_1 :
  ↑( (int.to_nat (real.to_int (sum_S))) ) % 5 = 1 := by sorry

end remainder_when_closest_even_to_sum_S_divided_by_5_is_1_l729_729420


namespace find_cos_beta_l729_729105

noncomputable def cos_beta (α β : ℝ) : ℝ :=
  - (6 * Real.sqrt 2 + 4) / 15

theorem find_cos_beta (α β : ℝ)
  (h0 : α ∈ Set.Ioc 0 (Real.pi / 2))
  (h1 : β ∈ Set.Ioc (Real.pi / 2) Real.pi)
  (h2 : Real.cos α = 1 / 3)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.cos β = cos_beta α β :=
by
  sorry

end find_cos_beta_l729_729105


namespace dodecagon_formation_l729_729597

variable (A B C D K L M N : Point)
variable (AB BC CD DA : Segment)
variable (ABCD : Square A B C D)
variable (ABK : Triangle A B K)
variable (BCL : Triangle B C L)
variable (CDM : Triangle C D M)
variable (DAN : Triangle D A N)
variable (midpoints : List Point)
variable (KL LM MN NK : Segment)

def is_equilateral (T : Triangle) : Prop := 
  (T.AB = T.BC) ∧ (T.BC = T.CA)

def is_midpoint (P : Point) (S : Segment) : Prop :=
  dist P S.A = dist P S.B

def regular_dodecagon (points : List Point) : Prop :=
  points.length = 12 ∧ ∀ i j, dist points[i] points[(i + 1) % 12] = dist points[j] points[(j + 1) % 12]

theorem dodecagon_formation :
  is_square ABCD →
  is_equilateral ABK ∧ is_equilateral BCL ∧ is_equilateral CDM ∧ is_equilateral DAN →
  (∀ P ∈ midpoints, ∃ (S : Segment), is_midpoint P S) →
  (∃ KL_mid LM_mid MN_mid NK_mid, 
    KL_mid ∈ midpoints ∧ LM_mid ∈ midpoints ∧ MN_mid ∈ midpoints ∧ NK_mid ∈ midpoints ∧
    are_midpoints KL LM MN NK KL_mid LM_mid MN_mid NK_mid) →
  regular_dodecagon midpoints :=
begin
  sorry
end

end dodecagon_formation_l729_729597


namespace smallest_n_inequality_l729_729076

theorem smallest_n_inequality:
  ∃ n : ℤ, (∀ x y z : ℝ, (x^2 + 2 * y^2 + z^2)^2 ≤ n * (x^4 + 3 * y^4 + z^4)) ∧ n = 4 :=
by
  sorry

end smallest_n_inequality_l729_729076


namespace eval_expression_l729_729918

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729918


namespace no_unique_solution_for_x_l729_729399

theorem no_unique_solution_for_x (x : ℝ) : 
  ∃ B : ℝ × ℝ, 
    let O := (0,0) in 
    let A := (-x,0) in 
    B = (A.1 + 3 * real.cos (150 * real.pi / 180), A.2 + 3 * real.sin (150 * real.pi / 180)) ∧ 
    real.dist O B = real.sqrt 3 → 
      ¬ ∃! x, x = x :=
begin
  sorry
end

end no_unique_solution_for_x_l729_729399


namespace evaluate_expression_l729_729871

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729871


namespace cost_price_toy_l729_729688

theorem cost_price_toy (selling_price_total : ℝ) (total_toys : ℕ) (gain_toys : ℕ) (sp_per_toy : ℝ) (general_cost : ℝ) :
  selling_price_total = 27300 →
  total_toys = 18 →
  gain_toys = 3 →
  sp_per_toy = selling_price_total / total_toys →
  general_cost = sp_per_toy * total_toys - (sp_per_toy * gain_toys / total_toys) →
    general_cost = 1300 := 
by 
  sorry

end cost_price_toy_l729_729688


namespace olivia_gives_each_sister_two_euros_l729_729242

/-- Olivia has 20 euros. Each of her four sisters has 10 euros. Prove that Olivia needs to give
     each of her sisters 2 euros so that each of the five girls has the same amount of money. -/
theorem olivia_gives_each_sister_two_euros (O S G : ℕ) (HO : O = 20) (HS : S = 10) (HG : G = 5) : 
  let total_amount := O + 4 * S in
  let equal_share := total_amount / G in
  equal_share - S = 2 := 
by
  sorry

end olivia_gives_each_sister_two_euros_l729_729242


namespace probability_between_lines_l729_729182

theorem probability_between_lines :
  let l := λ x : ℝ, -2 * x + 8
  let m := λ x : ℝ, -3 * x + 8
  let area_l := (4 * 8) / 2
  let area_m := ((8 / 3) * 8) / 2
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.33 :=
by
  let l := λ x : ℝ, -2 * x + 8
  let m := λ x : ℝ, -3 * x + 8
  let area_l := (4 * 8) / 2
  let area_m := ((8 / 3) * 8) / 2
  let area_between := area_l - area_m
  let probability := area_between / area_l
  sorry

end probability_between_lines_l729_729182


namespace evaluate_expression_l729_729930

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729930


namespace ducks_in_smaller_pond_l729_729191

theorem ducks_in_smaller_pond (x : ℝ) (h1 : 50 > 0) 
  (h2 : 0.20 * x > 0) (h3 : 0.12 * 50 > 0) (h4 : 0.15 * (x + 50) = 0.20 * x + 0.12 * 50) 
  : x = 30 := 
sorry

end ducks_in_smaller_pond_l729_729191


namespace evaluate_expression_l729_729879

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729879


namespace complex_solution_exists_l729_729659

theorem complex_solution_exists :
  ∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ (x + y * Complex.i) ^ 3 = -26 + d * Complex.i :=
by
  use 1
  use 3
  split
  { exact zero_lt_one }
  split
  { exact lt_of_lt_of_le zero_lt_three (le_refl 3) }
  sorry -- Proof of the computation that (1 + 3i)^3 = -26 + (some integer) * i

end complex_solution_exists_l729_729659


namespace lim_n_div_x_n_lim_n2_div_x_n_l729_729230

def sequence : ℕ → ℝ
| 0       := 1
| (n + 1) := sequence n + 3 * sqrt (sequence n) + n / sqrt (sequence n)

theorem lim_n_div_x_n : (tendsto (λ n, n / (sequence n)) at_top (𝓝 0)) :=
sorry

theorem lim_n2_div_x_n : (tendsto (λ n, (n ^ 2) / (sequence n)) at_top (𝓝 (4 / 9))) :=
sorry

end lim_n_div_x_n_lim_n2_div_x_n_l729_729230


namespace evaluate_sqrt_expression_l729_729971

noncomputable def evaluatedExpression : ℚ := 
    (Real.sqrt (9 / 4)) - (Real.sqrt (8 / 9)) + (Real.sqrt 1)

theorem evaluate_sqrt_expression :
    evaluatedExpression = (15 - 4 * Real.sqrt 2) / 6 :=
by
    sorry

end evaluate_sqrt_expression_l729_729971


namespace units_digit_2_1501_5_1602_11_1703_l729_729766

theorem units_digit_2_1501_5_1602_11_1703 : 
  (2 ^ 1501 * 5 ^ 1602 * 11 ^ 1703) % 10 = 0 :=
  sorry

end units_digit_2_1501_5_1602_11_1703_l729_729766


namespace evaluate_expression_l729_729835

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729835


namespace integer_values_x_possible_l729_729303

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l729_729303


namespace necessarily_positive_y_plus_z_squared_l729_729610

variable (x y z : ℝ)

def conditions : Prop :=
  0 < x ∧ x < 1 ∧ -1 < y ∧ y < 0 ∧ 2 < z ∧ z < 3

theorem necessarily_positive_y_plus_z_squared : conditions x y z → y + z^2 > 0 :=
by
  intros h
  have hx : 0 < x ∧ x < 1 := ⟨h.1, h.2⟩
  have hy : -1 < y ∧ y < 0 := ⟨h.3, h.4⟩
  have hz : 2 < z ∧ z < 3 := ⟨h.5, h.6⟩
  sorry

end necessarily_positive_y_plus_z_squared_l729_729610


namespace rational_values_of_expressions_l729_729370

theorem rational_values_of_expressions {x : ℚ} :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by
  sorry

end rational_values_of_expressions_l729_729370


namespace margin_expression_l729_729177

variable (n : ℕ) (C S M : ℝ)

theorem margin_expression (H1 : M = (1 / n) * C) (H2 : C = S - M) : 
  M = (1 / (n + 1)) * S := 
by
  sorry

end margin_expression_l729_729177


namespace greatest_distance_A_B_l729_729201

-- Define the sets A and B based on given conditions
def setA : set ℂ := {z | z^4 = 16}
def setB : set ℂ := {z | z^4 - 16*z^3 - 16*z + 256 = 0}

-- Statement of the proof problem
theorem greatest_distance_A_B : 
  (∀ z1 ∈ setA, ∀ z2 ∈ setB, dist z1 z2 ≤ 2 * Real.sqrt 65) ∧
  (∃ z1 ∈ setA, ∃ z2 ∈ setB, dist z1 z2 = 2 * Real.sqrt 65) := 
sorry

end greatest_distance_A_B_l729_729201


namespace five_families_occupancy_sequence_l729_729662

theorem five_families_occupancy_sequence :
  ∃ A B C D E : ℕ,
    -- Zhao's condition: 3rd to move in, opposite of 1st.
    (C = 105 ∧ A = 101 ∧ A ≠ C) ∧
    -- Qian's condition: only one on the top floor.
    (B = 109) ∧
    -- Sun's condition: occupants on the floor above and below.
    (D = 107) ∧
    -- Li's condition: last one, the floor below empty.
    (E = 103) ∧
    -- Zhou's condition: lives in 106, 104 and 108 are empty.
    (C ≠ 106 ∧ B ≠ 106 ∧ D ≠ 106 ∧ E ≠ 106 ∧ C ≠ 104 ∧ B ≠ 104 ∧ D ≠ 104 ∧ E ≠ 104 ∧ 
     C ≠ 108 ∧ B ≠ 108 ∧ D ≠ 108 ∧ E ≠ 108 ) ∧    
    -- Assert the formed number.
    (10^4 * A + 10^3 * B + 10^2 * C + 10^1 * D + E = 69573) :=
begin
  sorry
end

end five_families_occupancy_sequence_l729_729662


namespace vertical_asymptotes_polynomial_l729_729169

theorem vertical_asymptotes_polynomial (a b : ℝ) (h₁ : -3 * 2 = b) (h₂ : -3 + 2 = a) : a + b = -5 := by
  sorry

end vertical_asymptotes_polynomial_l729_729169


namespace solutions_count_eq_33_l729_729458

noncomputable def count_solutions : ℝ :=
  33

theorem solutions_count_eq_33 :
  ∀ x ∈ Icc (-50 : ℝ) 50, (x / 50 = sin x) → x = count_solutions :=
by
  sorry

end solutions_count_eq_33_l729_729458


namespace paint_per_large_canvas_l729_729030

-- Define the conditions
variables (L : ℕ) (paint_large paint_small total_paint : ℕ)

-- Given conditions
def large_canvas_paint := 3 * L
def small_canvas_paint := 4 * 2
def total_paint_used := large_canvas_paint + small_canvas_paint

-- Statement that needs to be proven
theorem paint_per_large_canvas :
  total_paint_used = 17 → L = 3 :=
by
  intro h
  sorry

end paint_per_large_canvas_l729_729030


namespace eval_expr_l729_729802

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729802


namespace triangle_side_count_l729_729294

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l729_729294


namespace collinear_E_H_N_l729_729140

-- Definitions of points from the problem
variables (A B C E F G H N : Point)
variables (ABC : Triangle A B C)

-- Hypothesis for the excircle and its touchpoints
variables (touches_AB : excircle_touches ABC AB E)
variables (touches_BC : excircle_touches ABC BC F)
variables (touches_CA : excircle_touches ABC CA G)

-- Intersection of lines AF and BG
variables (intersect_AF_BG : line_intersection (line_through_points A F) (line_through_points B G) H)

-- Definition of medial triangle and incircle touchpoint
variables (IN : incircle_touch_medial_triangle ABC AB N)

-- Conjecture: Collinearity of points E, H, and N
theorem collinear_E_H_N
  (h_excircle_touches : touches_AB ∧ touches_BC ∧ touches_CA)
  (h_intersection : intersect_AF_BG)
  (h_incircle_touch : IN) :
  collinear E H N :=
sorry

end collinear_E_H_N_l729_729140


namespace evaluate_expression_l729_729847

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729847


namespace cake_cut_sum_l729_729719

theorem cake_cut_sum (length width height : ℝ) (c s : ℝ) (T M : ℝ × ℝ) :
  length = 4 ∧ width = 3 ∧ height = 2 ∧ 
  T = (2, 1.5) ∧ M = (0, 1.5) ∧ 
  c = 12 ∧ s = 26 → c + s = 38 := by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with hT h_rest
  cases h_rest with hM h_rest
  cases h_rest with hc hs
  rw [hc, hs]
  simp

end cake_cut_sum_l729_729719


namespace eval_expr_l729_729804

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729804


namespace sum_of_digits_of_n_is_19_l729_729520

def decimal_to_hexadecimal (n : ℕ) : List (Fin 16) :=
  if n = 0 then [0] else
  let rec loop (n : ℕ) (acc : List (Fin 16)) : List (Fin 16) :=
    if n = 0 then acc else loop (n / 16) ((Fin.ofNat (n % 16)) :: acc)
  loop n []

def contains_only_numeric_digits (l : List (Fin 16)) : Bool :=
  l.all (λ d, d.val < 10)

def count_hex_numbers_with_only_numeric_digits (bound : ℕ) : ℕ :=
  (List.range bound).count (λ n, contains_only_numeric_digits (decimal_to_hexadecimal n))

theorem sum_of_digits_of_n_is_19 : 
  count_hex_numbers_with_only_numeric_digits 500 = 199 ∧ (1 + 9 + 9 = 19) :=
by
  sorry

end sum_of_digits_of_n_is_19_l729_729520


namespace factory_output_decrease_l729_729689

theorem factory_output_decrease :
  ∀ (original_output : ℝ) (initial_increase_percent : ℝ) (holiday_increase_percent : ℝ),
  original_output > 0 →
  initial_increase_percent = 10 →
  holiday_increase_percent = 40 →
  let increased_output := original_output * (1 + initial_increase_percent / 100) in
  let final_output := increased_output * (1 + holiday_increase_percent / 100) in
  let decrease_needed := (final_output - original_output) / final_output * 100 in
  decrease_needed ≈ 35.06 := by
  sorry

end factory_output_decrease_l729_729689


namespace range_of_a_l729_729128

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3)*x^3 + x^2 + a*x
def g (x : ℝ) : ℝ := (1/ℯ)^x

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ set.Icc (1/2 : ℝ) 2, ∃ x₂ ∈ set.Icc (1/2 : ℝ) 2, (f a x₁).derivative > g x₂) ->
  a ∈ set.Ioi (ℯ^(-2) - (5/4)) :=
by
  sorry

end range_of_a_l729_729128


namespace arithmetic_sequence_general_term_l729_729652

def S (n : ℕ) (a : ℕ → ℤ) : ℤ := n * (a 1 + a n) / 2

theorem arithmetic_sequence_general_term :
  (∀ n, S n (λ n, -8 + (n - 1) * 2) = (n * (2 * -8 + (n-1) * 2)) / 2) →
  S 10 (λ n, -8 + (n - 1) * 2) = 10 →
  S 20 (λ n, -8 + (n - 1) * 2) = 220 →
  (∀ n, (λ n, -8 + (n - 1) * 2) n = 2 * n - 10) :=
by
sorry

end arithmetic_sequence_general_term_l729_729652


namespace integer_solution_count_l729_729783

theorem integer_solution_count : 
  ∃ (n : ℕ), 
    n = { x : ℤ | -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24 }.to_finset.card :=
by sorry

end integer_solution_count_l729_729783


namespace Kenny_running_to_basketball_ratio_l729_729213

theorem Kenny_running_to_basketball_ratio (basketball_hours trumpet_hours running_hours : ℕ) 
    (h1 : basketball_hours = 10)
    (h2 : trumpet_hours = 2 * running_hours)
    (h3 : trumpet_hours = 40) :
    running_hours = 20 ∧ basketball_hours = 10 ∧ (running_hours / basketball_hours = 2) :=
by
  sorry

end Kenny_running_to_basketball_ratio_l729_729213


namespace evaluate_expression_l729_729929

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729929


namespace chipped_marbles_bag_contains_24_l729_729465

theorem chipped_marbles_bag_contains_24 :
  ∃ (A B C D E : ℕ), 
  (A = 15 ∧ B = 18 ∧ C = 22 ∧ D = 24 ∧ E = 30) ∧
  (∃ (x y z w v : ℕ), 
    (x = A ∧ y = B ∧ z = C ∧ w = D ∧ v = E) ∧ 
    (x + y + z + w + v = 109) ∧ 
    (∃ (J_bags G_bags : set ℕ), 
      J_bags = {z, w} ∧ G_bags = {x, y} ∧ (J_bags ∪ G_bags ⊆ {A, B, C, D, E}) ∧ 
      (|{A, B, C, D, E} \ (J_bags ∪ G_bags)| = 1) ∧
      ∃ J_total G_total, 
        (J_total = z + w) ∧ (G_total = x + y) ∧ (J_total = 3 * G_total))
  ) →
  ∃ x, x = 24 := sorry

end chipped_marbles_bag_contains_24_l729_729465


namespace expand_expression_l729_729449

theorem expand_expression (y : ℚ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end expand_expression_l729_729449


namespace speedster_convertibles_l729_729687

-- Definitions of conditions
def isSpeedster (v : Vehicle) : Prop := sorry
def isConvertible (v : Vehicle) : Prop := sorry
def inventory : Set Vehicle := sorry
def speedsterInventory : Set Vehicle := {v ∈ inventory | isSpeedster v}
def convertibleInventory : Set Vehicle := {v ∈ inventory | isSpeedster v ∧ isConvertible v}
def nonSpeedsterInventory : Set Vehicle := {v ∈ inventory | ¬ isSpeedster v}

-- Assumptions based on given conditions
axiom speedster_fraction : ∃ T : ℕ, speedsterInventory.card = (2 / 3) * T
axiom convertible_fraction : ∀ v ∈ speedsterInventory, (4 / 5) * speedsterInventory.card = convertibleInventory.card
axiom non_speedster_count : nonSpeedsterInventory.card = 40

-- Proof statement
theorem speedster_convertibles : ∃ T : ℕ, (4 / 5) * (2 / 3) * T = 64 :=
by sorry

end speedster_convertibles_l729_729687


namespace evaluation_of_expression_l729_729886

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729886


namespace area_of_figure_M_l729_729593

noncomputable def figure_M_area : Real :=
  sorry

theorem area_of_figure_M :
  figure_M_area = 3 :=
  sorry

end area_of_figure_M_l729_729593


namespace probability_three_heads_in_eight_tosses_l729_729008

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729008


namespace eval_expression_l729_729916

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729916


namespace evaluate_expression_l729_729934

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729934


namespace calculate_tan_product_l729_729418

-- Define the problem statement and conditions
theorem calculate_tan_product :
  ∏ x in finset.range 89, (1 + real.tan (real.pi / 180 * (x + 1))) = 2 ^ 45 := sorry

end calculate_tan_product_l729_729418


namespace remainder_division_123456789012_by_112_l729_729782

-- Define the conditions
def M : ℕ := 123456789012
def m7 : ℕ := M % 7
def m16 : ℕ := M % 16

-- State the proof problem
theorem remainder_division_123456789012_by_112 : M % 112 = 76 :=
by
  -- Conditions
  have h1 : m7 = 3 := by sorry
  have h2 : m16 = 12 := by sorry
  -- Conclusion
  sorry

end remainder_division_123456789012_by_112_l729_729782


namespace gcd_lcm_product_24_36_proof_l729_729072

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l729_729072


namespace eval_expression_l729_729970

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729970


namespace eval_expression_l729_729950

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729950


namespace height_difference_l729_729044

def height_of_graces_tower := 40
def multiplier := 8
def height_of_clydes_tower := height_of_graces_tower / multiplier

theorem height_difference (height_of_graces_tower : ℕ) (multiplier : ℕ) :
  height_of_graces_tower = 40 →
  multiplier = 8 →
  height_of_graces_tower - (height_of_graces_tower / multiplier) = 35 :=
by
  intros h_grace h_mult
  rw [h_grace, h_mult]
  have h_clyde : height_of_graces_tower / multiplier = 5
  { simp [h_grace, h_mult] }
  rw [h_clyde]
  simp
  exact sorry

end height_difference_l729_729044


namespace eval_expression_l729_729940

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729940


namespace math_problem_find_n_l729_729524

variable {a b : ℂ}
variable (x : ℝ)

theorem math_problem 
  (h1 : a^2 + b^2 = 5) 
  (h2 : a^3 + b^3 = 7) 
  (hx : x = a + b) :
  x = ( -1 + Real.sqrt 57 ) / 2 := sorry

# Prove n = 57
theorem find_n : 57 = 57 := rfl

end math_problem_find_n_l729_729524


namespace eval_expression_l729_729953

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729953


namespace angle_between_vectors_perpendicular_vectors_l729_729108

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (h_a : ∥a∥ = 2)
variables (h_b : ∥b∥ = 2)
variables (h_a_b : ∥a + b∥ = 2)

-- Statement a: Prove the angle between a and b is 2π/3
theorem angle_between_vectors (a b : V) (h_a : ∥a∥ = 2) (h_b : ∥b∥ = 2) 
  (h_a_b : ∥a + b∥ = 2) : real.angle a b = 2 * real.pi / 3 :=
sorry

-- Statement b: Prove that (a + 2b) is perpendicular to a
theorem perpendicular_vectors (a b : V) (h_a : ∥a∥ = 2) (h_b : ∥b∥ = 2) 
  (h_a_b : ∥a + b∥ = 2) : ⟪a + 2 • b, a⟫ = 0 :=
sorry

end angle_between_vectors_perpendicular_vectors_l729_729108


namespace find_n_l729_729492

theorem find_n (x n : ℝ) (h1 : Real.log10 (Real.sin x) + Real.log10 (Real.cos x) = -2)
  (h2 : Real.log10 ((Real.sin x + Real.cos x) ^ 2) = Real.log10 n + 1) : 
  n = 0.102 :=
sorry

end find_n_l729_729492


namespace evaluation_of_expression_l729_729894

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729894


namespace find_period_l729_729018

variable (Rs_3500 : ℝ := 3500)
variable (interest_A_rate : ℝ := 0.10) -- 10% per annum
variable (interest_C_rate : ℝ := 0.15) -- 15% per annum
variable (total_gain_B : ℝ := 525) -- Rs. 525 in total

theorem find_period :
  let interest_to_A := interest_A_rate * Rs_3500,
      interest_from_C := interest_C_rate * Rs_3500,
      gain_per_annum := interest_from_C - interest_to_A in
  total_gain_B = gain_per_annum * 3 →
  3 = n :=
by
  intros,
  sorry

end find_period_l729_729018


namespace smallest_number_of_students_in_debate_club_l729_729647

-- Define conditions
def ratio_8th_to_6th (x₈ x₆ : ℕ) : Prop := 7 * x₆ = 4 * x₈
def ratio_8th_to_7th (x₈ x₇ : ℕ) : Prop := 6 * x₇ = 5 * x₈
def ratio_8th_to_9th (x₈ x₉ : ℕ) : Prop := 9 * x₉ = 2 * x₈

-- Problem statement
theorem smallest_number_of_students_in_debate_club 
  (x₈ x₆ x₇ x₉ : ℕ) 
  (h₁ : ratio_8th_to_6th x₈ x₆) 
  (h₂ : ratio_8th_to_7th x₈ x₇) 
  (h₃ : ratio_8th_to_9th x₈ x₉) : 
  x₈ + x₆ + x₇ + x₉ = 331 := 
sorry

end smallest_number_of_students_in_debate_club_l729_729647


namespace evaluation_of_expression_l729_729888

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729888


namespace lies_on_new_ellipse_lies_on_new_hyperbola_l729_729094

variable (x y c d a : ℝ)

def new_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Definition for new ellipse.
def is_new_ellipse (E : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  new_distance E F1 + new_distance E F2 = 2 * a

-- Definition for new hyperbola.
def is_new_hyperbola (H : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  |new_distance H F1 - new_distance H F2| = 2 * a

-- The point E lies on the new ellipse.
theorem lies_on_new_ellipse
  (E F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_ellipse E F1 F2 a :=
by sorry

-- The point H lies on the new hyperbola.
theorem lies_on_new_hyperbola
  (H F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_hyperbola H F1 F2 a :=
by sorry

end lies_on_new_ellipse_lies_on_new_hyperbola_l729_729094


namespace evaluate_expression_l729_729864

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729864


namespace calc_expression_l729_729423

theorem calc_expression : (sqrt 5) ^ 2 + abs (-3) - (Real.pi + sqrt 3) ^ 0 = 7 := by
  sorry

end calc_expression_l729_729423


namespace boxes_sold_in_first_five_days_actual_sales_volume_for_week_reached_planned_quantity_farmer_earnings_for_week_l729_729012

-- Define the deviations for each day
def deviations : List Int := [4, -3, -5, +7, -8, +21, -6]

-- Define the planned number of boxes per day
def planned_boxes_per_day := 10

-- Define the selling price and shipping fee per box
def selling_price_per_box := 80
def shipping_fee_per_box := 7

-- Define the correct answers
def boxes_sold_first_five_days := 45
def actual_sales_reached_planned := True
def total_earnings := 5840

theorem boxes_sold_in_first_five_days :
  List.sum (List.take deviations 5) + planned_boxes_per_day * 5 = boxes_sold_first_five_days := by
  sorry

theorem actual_sales_volume_for_week_reached_planned_quantity :
  (List.sum deviations + planned_boxes_per_day * List.length deviations) = 80 := by
  sorry

theorem farmer_earnings_for_week :
  (80 * selling_price_per_box) - (80 * shipping_fee_per_box) = total_earnings := by
  sorry

end boxes_sold_in_first_five_days_actual_sales_volume_for_week_reached_planned_quantity_farmer_earnings_for_week_l729_729012


namespace octagon_diagonals_l729_729151

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l729_729151


namespace part_1_part_2_l729_729056

section trisomic_cell
variable (n : ℕ)
variable (gamete_genotypes : Type)
variable (disease_resistance : gamete_genotypes → Prop)
variable (trisomic_gen: gamete_genotypes → Prop)
variable (cross : gamete_genotypes → gamete_genotypes → list (gamete_genotypes × ℕ))

def meiosis (parent_with_three_chromosomes: gamete_genotypes): list (gamete_genotypes × ℕ) :=
  sorry

theorem part_1 (bbb BB: gamete_genotypes)
  (Htrisomic_bbb: trisomic_gen bbb)
  (Hnormal_BB: ¬ trisomic_gen BB) :
  let F1: list (gamete_genotypes × ℕ) := cross bbb BB in
  F1 = [(Bbb, 2), (Bb, 1)] → 
  cross (Bbb : gamete_genotypes) (bb : gamete_genotypes) = 
  [(B, 1), (b, 2)] → 
  list.sum (list.map prod.snd (cross B (bb : gamete_genotypes))) = 2 ∧
  list.sum (list.map prod.snd (cross b bb)) = 1 → 
  true :=
sorry

theorem part_2 
  (dis_resistant: gamete_genotypes)
  (trisomic_susceptible: gamete_genotypes)
  (susceptible : gamete_genotypes) 
  (Htrisomic: trisomic_gen trisomic_susceptible)
  (Hdis_resistant: disease_resistance dis_resistant) 
  (cross1: list (gamete_genotypes × ℕ))
  (cross2: list (gamete_genotypes × ℕ))
  (Hcross1: cross trisomic_susceptible dis_resistant = cross1)
  (Hcross2: cross1 = cross dis_resistant susceptible)
  (F2ratio : float)
  :
  (F2ratio = 1) ∨ (F2ratio = 2) → 
  (F2ratio = 1 → ¬ trisomic_gen dis_resistant) ∧ 
  (F2ratio = 2 → trisomic_gen dis_resistant) :=
sorry
end trisomic_cell

end part_1_part_2_l729_729056


namespace constant_term_is_negative_thirty_two_l729_729481

noncomputable def n : ℝ := ∫ x in 0..2, x^3

theorem constant_term_is_negative_thirty_two :
  let expr := (x - (2 / (x ^ (1 / 3))))^n
  in constant_term_of_binomial_expansion expr = -32 := 
by 
  let expr := (x - (2 / (x ^ (1 / 3))))^n
  have h1 : n = 4 := by simp [n,integral_div_by_fundamental_theorem_of_calculus]
  sorry

end constant_term_is_negative_thirty_two_l729_729481


namespace evaluate_expression_l729_729865

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729865


namespace evaluate_expression_l729_729920

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729920


namespace ratio_sum_product_is_constant_l729_729699

variables {p a : ℝ} (h_a : 0 < a)
theorem ratio_sum_product_is_constant
    (k : ℝ) (h_k : k ≠ 0)
    (x₁ x₂ : ℝ) (h_intersection : x₁ * (2 * p * (x₂ - a)) = 2 * p * (x₁ - a) ∧ x₂ * (2 * p * (x₁ - a)) = 2 * p * (x₂ - a)) :
  (x₁ + x₂) / (x₁ * x₂) = 1 / a := by
  sorry

end ratio_sum_product_is_constant_l729_729699


namespace evaluate_expression_l729_729785

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729785


namespace round_robin_total_points_l729_729721

theorem round_robin_total_points :
  let points_per_match := 2
  let total_matches := 3
  (total_matches * points_per_match) = 6 :=
by
  sorry

end round_robin_total_points_l729_729721


namespace joan_total_spent_on_clothing_l729_729211

theorem joan_total_spent_on_clothing :
  let shorts_cost := 15.00
  let jacket_cost := 14.82
  let shirt_cost := 12.51
  let shoes_cost := 21.67
  let hat_cost := 8.75
  let belt_cost := 6.34
  shorts_cost + jacket_cost + shirt_cost + shoes_cost + hat_cost + belt_cost = 79.09 :=
by
  sorry

end joan_total_spent_on_clothing_l729_729211


namespace eval_expr_l729_729808

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729808


namespace find_real_solutions_l729_729976

theorem find_real_solutions (x : ℝ) :
  x^4 + (3 - x)^4 = 146 ↔ x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 :=
by
  sorry

end find_real_solutions_l729_729976


namespace point_in_transformed_plane_l729_729576

-- Define a structure for Point
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

-- Define a structure for Plane
structure Plane := (A B C D : ℝ)

-- Definition of point A
def A : Point := { x := 3, y := 2, z := 4 }

-- Definition of initial plane a
def a : Plane := { A := 2, B := -3, C := 1, D := -6 }

-- Definition of similarity transformation coefficient k
def k : ℝ := 2 / 3

-- Definition of the transformed plane
def a_transformed (p : Plane) (k : ℝ) : Plane := 
  { A := p.A, B := p.B, C := p.C, D := k * p.D }

-- Target: Prove that point A lies on the transformed plane
theorem point_in_transformed_plane : 
  let trans_plane := a_transformed a k in
  trans_plane.A * A.x + trans_plane.B * A.y + trans_plane.C * A.z + trans_plane.D = 0 :=
by
  let trans_plane := a_transformed a k 
  show trans_plane.A * A.x + trans_plane.B * A.y + trans_plane.C * A.z + trans_plane.D = 0
  sorry

end point_in_transformed_plane_l729_729576


namespace evaluate_expression_l729_729881

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729881


namespace find_a_l729_729121

noncomputable def z1 := 2 + Complex.i
noncomputable def z2 (a : ℝ) := a - Complex.i

theorem find_a (a : ℝ) : (z1 * z2 a).im = 0 → a = 2 := by
  simp [z1, z2, Complex.mul_eq, Complex.add_im, Complex.sub_im, Complex.one_im, Complex.i_im, Complex.i_re, Complex.zero_im, Complex.zero_re, Complex.one_re]
  intro h
  linarith

end find_a_l729_729121


namespace evaluation_of_expression_l729_729890

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729890


namespace seven_does_not_always_divide_l729_729348

theorem seven_does_not_always_divide (n : ℤ) :
  ¬(7 ∣ (n ^ 2225 - n ^ 2005)) :=
by sorry

end seven_does_not_always_divide_l729_729348


namespace abs_fraction_complex_eq_one_l729_729498

theorem abs_fraction_complex_eq_one : 
  abs ((1 - complex.I) / (1 + complex.I)) = 1 := 
by 
  sorry

end abs_fraction_complex_eq_one_l729_729498


namespace fraction_difference_l729_729525

variable (x y : ℝ)

theorem fraction_difference (h : x / y = 2) : (x - y) / y = 1 :=
  sorry

end fraction_difference_l729_729525


namespace tim_fewer_apples_l729_729664

theorem tim_fewer_apples (martha_apples : ℕ) (harry_apples : ℕ) (tim_apples : ℕ) (H1 : martha_apples = 68) (H2 : harry_apples = 19) (H3 : harry_apples * 2 = tim_apples) : martha_apples - tim_apples = 30 :=
by
  sorry

end tim_fewer_apples_l729_729664


namespace same_function_l729_729680

-- Definitions based on conditions in the problem
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := if x ≥ 0 then x else -x

-- Lean statement that proves f(x) = g(x) for all x
theorem same_function : ∀ (x : ℝ), f x = g x :=
by
  intro x
  sorry

end same_function_l729_729680


namespace quadratic_solution_l729_729074

noncomputable def g (x : ℝ) : ℝ := x^2 + 2021 * x + 18

theorem quadratic_solution : ∀ x : ℝ, g (g x + x + 1) / g x = x^2 + 2023 * x + 2040 :=
by
  intros
  sorry

end quadratic_solution_l729_729074


namespace ali_baba_can_open_cave_l729_729416

theorem ali_baba_can_open_cave (n : ℕ) : 
  (∃ (f : set ℕ → set ℕ), (∀ S : set ℕ, S ⊆ finset.range n → (f S).subset (finset.range n) ∧ f (f S) = S)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end ali_baba_can_open_cave_l729_729416


namespace integer_values_x_possible_l729_729301

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l729_729301


namespace food_additives_percentage_l729_729387

variable (microphotonics_pct home_electronics_pct gmo_pct industrial_lubricants_pct astrophysics_degrees : ℝ)

-- Define the given percentages as constants
def MicrophotonicsPercentage : ℝ := 13
def HomeElectronicsPercentage : ℝ := 24
def GMOPercentage : ℝ := 29
def IndustrialLubricantsPercentage : ℝ := 8

-- Define the arc for basic astrophysics research
def BasicAstrophysicsDegrees : ℝ := 39.6
def FullCircleDegrees : ℝ := 360

-- Prove the percentage allocation for food additives
theorem food_additives_percentage :
  microphotonics_pct = MicrophotonicsPercentage ∧
  home_electronics_pct = HomeElectronicsPercentage ∧
  gmo_pct = GMOPercentage ∧
  industrial_lubricants_pct = IndustrialLubricantsPercentage ∧
  astrophysics_degrees = BasicAstrophysicsDegrees →
  let basicAstrophysicsPercentage := (astrophysics_degrees / FullCircleDegrees) * 100
  in (100 - (microphotonics_pct + home_electronics_pct + gmo_pct + industrial_lubricants_pct + basicAstrophysicsPercentage) = 15) :=
by
  sorry

end food_additives_percentage_l729_729387


namespace investment_final_amount_l729_729032

theorem investment_final_amount :
  let principal := 15000
  let semi_annual_interest := 0.04
  let periods := 10
  let annual_contribution := 1000
  let contribution_periods := 5
  (principal * (1 + semi_annual_interest)^periods +
  ∑ k in (Finset.range contribution_periods), 
    annual_contribution * (1 + semi_annual_interest)^(periods - 2 * k)) = 28540 :=
by
  let principal := 15000
  let semi_annual_interest := 0.04
  let periods := 10
  let annual_contribution := 1000
  let contribution_periods := 5
  let total_initial_investment := principal * (1 + semi_annual_interest)^periods
  let total_contributions := ∑ k in (Finset.range contribution_periods),
                              annual_contribution * (1 + semi_annual_interest)^(periods - 2 * k)
  let final_amount := total_initial_investment + total_contributions
  have : final_amount = 28540 := sorry
  exact this

end investment_final_amount_l729_729032


namespace sum_of_complex_numbers_l729_729580

noncomputable def complex_numbers_with_properties (n : ℕ) (z : ℕ → ℂ) : Prop :=
  (∀ k, k < n → complex.abs (z k) = 1) ∧
  (∀ k, k < n → (complex.arg(z k) ≥ 0 ∧ complex.arg(z k) ≤ π))

theorem sum_of_complex_numbers (n : ℕ) 
  (z : ℕ → ℂ)
  (hn : n % 2 = 1)
  (h : complex_numbers_with_properties n z) :
  complex.abs (finset.univ.sum (λ k, z k)) ≥ 1 :=
sorry

end sum_of_complex_numbers_l729_729580


namespace puppy_sleep_duration_l729_729767

-- Definitions based on conditions
def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def puppy_sleep : ℕ := 2 * luke_sleep

-- Theorem stating that the puppy sleeps for 16 hours
theorem puppy_sleep_duration : puppy_sleep = 16 := by
  sorry

end puppy_sleep_duration_l729_729767


namespace common_difference_arithmetic_seq_l729_729651

theorem common_difference_arithmetic_seq (S n a1 d : ℕ) (h_sum : S = 650) (h_n : n = 20) (h_a1 : a1 = 4) :
  S = (n / 2) * (2 * a1 + (n - 1) * d) → d = 3 := by
  intros h_formula
  sorry

end common_difference_arithmetic_seq_l729_729651


namespace Fred_earned_4_dollars_l729_729214

-- Conditions are translated to definitions
def initial_amount_Fred : ℕ := 111
def current_amount_Fred : ℕ := 115

-- Proof problem in Lean 4 statement
theorem Fred_earned_4_dollars : current_amount_Fred - initial_amount_Fred = 4 := by
  sorry

end Fred_earned_4_dollars_l729_729214


namespace x_minus_y_eq_eight_l729_729324

theorem x_minus_y_eq_eight (x y : ℝ) (hx : 3 = 0.15 * x) (hy : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end x_minus_y_eq_eight_l729_729324


namespace f_neg_a_eq_minus_10_l729_729130

def f (x : ℝ) : ℝ := (2022 * x^3 + 2 * x^2 + 3 * x + 6) / (x^2 + 3)

variable (a : ℝ)
variable (h : f a = 14)

-- Statement to be proved
theorem f_neg_a_eq_minus_10 : f (-a) = -10 := 
by {
  -- proof goes here but is replaced with sorry.
  sorry 
}

end f_neg_a_eq_minus_10_l729_729130


namespace mary_daily_tasks_l729_729238

theorem mary_daily_tasks :
  ∃ (x y : ℕ), (x + y = 15) ∧ (4 * x + 7 * y = 85) ∧ (y = 8) :=
by
  sorry

end mary_daily_tasks_l729_729238


namespace number_of_distinct_cubes_l729_729709

theorem number_of_distinct_cubes (w b : ℕ) (total_cubes : ℕ) (dim : ℕ) :
  w + b = total_cubes ∧ total_cubes = 8 ∧ dim = 2 ∧ w = 6 ∧ b = 2 →
  (number_of_distinct_orbits : ℕ) = 1 :=
by
  -- Conditions
  intros h
  -- Translation of conditions into a useful form
  let num_cubes := 8
  let distinct_configurations := 1
  -- Burnside's Lemma applied to find the distinct configurations
  sorry

end number_of_distinct_cubes_l729_729709


namespace simplify_expression_eq_log_l729_729266

noncomputable def expression (a b : ℝ) : ℝ :=
  (1 - (Real.log a b) ^ 3) / ((Real.log a b + (1 / Real.log a b) + 1) * (1 - Real.log a b))

theorem simplify_expression_eq_log (a b : ℝ) (hab : a > 0 ∧ a ≠ 1) (hbb : b > 0) :
  expression a b = Real.log a b :=
by
  sorry

end simplify_expression_eq_log_l729_729266


namespace diagonals_of_octagon_l729_729154

theorem diagonals_of_octagon : 
  let n := 8 in 
  let total_line_segments := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_line_segments - sides in
  diagonals = 20 := 
  by 
    let n := 8
    let total_line_segments := (n * (n - 1)) / 2
    let sides := n
    let diagonals := total_line_segments - sides
    have h : diagonals = 20 := sorry
    exact h

end diagonals_of_octagon_l729_729154


namespace find_x_l729_729143

variables (x : ℝ) (a b : ℝ × ℝ)

def a := (2, 0) : ℝ × ℝ
def b := (x, 2 * real.sqrt 3) : ℝ × ℝ
def theta := 2 * real.pi / 3

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_x
(h : dot_product a b = magnitude a * magnitude b * real.cos theta) :
x = -2 :=
sorry

end find_x_l729_729143


namespace sum_n_k_l729_729633

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l729_729633


namespace quadrant_of_conjugate_z_l729_729506

-- Define the given complex number
def z : ℂ := 5 / (-2 + I)

-- State the main theorem
theorem quadrant_of_conjugate_z : (z.conj.re < 0) ∧ (z.conj.im > 0) :=
sorry

end quadrant_of_conjugate_z_l729_729506


namespace geometric_series_sum_l729_729765

theorem geometric_series_sum : 
  let a := 1
  let r := 2
  let n := 21
  a * ((r^n - 1) / (r - 1)) = 2097151 :=
by
  sorry

end geometric_series_sum_l729_729765


namespace eval_expression_l729_729821

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729821


namespace eval_expression_l729_729949

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729949


namespace set_representation_l729_729972

theorem set_representation :
  {p : ℕ × ℕ | 2 * p.1 + 3 * p.2 = 16} = {(2, 4), (5, 2), (8, 0)} :=
by
  sorry

end set_representation_l729_729972


namespace apple_equals_pear_l729_729243

-- Define the masses of the apple and pear.
variable (A G : ℝ)

-- The equilibrium condition on the balance scale.
axiom equilibrium_condition : A + 2 * G = 2 * A + G

-- Prove the mass of an apple equals the mass of a pear.
theorem apple_equals_pear (A G : ℝ) (h : A + 2 * G = 2 * A + G) : A = G :=
by
  -- Proof goes here, but we use sorry to indicate the proof's need.
  sorry

end apple_equals_pear_l729_729243


namespace find_n_expression_l729_729124

noncomputable def solve_for_n (s P k c : ℝ) (n : ℝ) : Prop :=
  P = s / ((1 + k) ^ n + c) → 
  n = log ((s / P) - c) / log (1 + k)

theorem find_n_expression (s P k c n : ℝ) : 
  solve_for_n s P k c n :=
by
  sorry

end find_n_expression_l729_729124


namespace percentage_carnations_l729_729014

variable (F : ℕ)
variable (H1 : F ≠ 0) -- Non-zero flowers
variable (H2 : ∀ (y : ℕ), 5 * y = F → 2 * y ≠ 0) -- Two fifths of the pink flowers are roses.
variable (H3 : ∀ (z : ℕ), 7 * z = 3 * (F - F / 2 - F / 5) → 6 * z ≠ 0) -- Six sevenths of the red flowers are carnations.
variable (H4 : ∀ (w : ℕ), 5 * w = F → w ≠ 0) -- One fifth of the flowers are yellow tulips.
variable (H5 : 2 * F / 2 = F) -- Half of the flowers are pink.
variable (H6 : ∀ (c : ℕ), 10 * c = F → c ≠ 0) -- Total flowers in multiple of 10

theorem percentage_carnations :
  (exists (pc rc : ℕ), 70 * (pc + rc) = 55 * F) :=
sorry

end percentage_carnations_l729_729014


namespace evaluate_expression_l729_729790

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729790


namespace remainder_of_7_9_power_2008_mod_64_l729_729648

theorem remainder_of_7_9_power_2008_mod_64 :
  (7^2008 + 9^2008) % 64 = 2 := 
sorry

end remainder_of_7_9_power_2008_mod_64_l729_729648


namespace barbie_can_carry_four_coconuts_l729_729753

variable (B : ℕ)

theorem barbie_can_carry_four_coconuts
  (total_coconuts : ℕ)
  (bruno_coconuts_per_trip : ℕ)
  (total_trips : ℕ)
  (total_coconuts = 144)
  (bruno_coconuts_per_trip = 8)
  (total_trips = 12)
  (total_coconuts = total_trips * (bruno_coconuts_per_trip + B)) :
  B = 4 :=
by
  sorry

end barbie_can_carry_four_coconuts_l729_729753


namespace diagonals_of_octagon_l729_729153

theorem diagonals_of_octagon : 
  let n := 8 in 
  let total_line_segments := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_line_segments - sides in
  diagonals = 20 := 
  by 
    let n := 8
    let total_line_segments := (n * (n - 1)) / 2
    let sides := n
    let diagonals := total_line_segments - sides
    have h : diagonals = 20 := sorry
    exact h

end diagonals_of_octagon_l729_729153


namespace find_BC_l729_729536

-- Define the triangle vertices and initial conditions.
variables (A B C X : Point)

-- Define distances as given in the problem.
def AB : ℝ := 95
def AC : ℝ := 115
def AX : ℝ := 10

-- Define the conditions for BX and CX being integers.
def BX : ℕ
def CX : ℕ

-- State the problem.
theorem find_BC (h_AB : dist A B = 95) (h_AC : dist A C = 115)
    (h_AX : dist A X = 10) (h_BX_int : ∃ (n : ℕ), dist B X = n)
    (h_CX_int : ∃ (m : ℕ), dist C X = m) :
  dist B C = 120 :=
sorry

end find_BC_l729_729536


namespace ways_to_select_four_doctors_l729_729704

def num_ways_to_select_doctors (num_internists : ℕ) (num_surgeons : ℕ) (team_size : ℕ) : ℕ :=
  (Nat.choose num_internists 1 * Nat.choose num_surgeons (team_size - 1)) + 
  (Nat.choose num_internists 2 * Nat.choose num_surgeons (team_size - 2)) + 
  (Nat.choose num_internists 3 * Nat.choose num_surgeons (team_size - 3))

theorem ways_to_select_four_doctors : num_ways_to_select_doctors 5 6 4 = 310 := 
by
  sorry

end ways_to_select_four_doctors_l729_729704


namespace octal_subtraction_l729_729756

noncomputable def octal_452 := nat.of_digits 8 [4, 5, 2]
noncomputable def octal_317 := nat.of_digits 8 [3, 1, 7]
noncomputable def octal_135 := nat.of_digits 8 [1, 3, 5]

theorem octal_subtraction : octal_452 - octal_317 = octal_135 := by
  sorry

end octal_subtraction_l729_729756


namespace days_to_shovel_l729_729174

-- Defining conditions as formal statements
def original_task_time := 10
def original_task_people := 10
def original_task_weight := 10000
def new_task_weight := 40000
def new_task_people := 5

-- Definition of rate in terms of weight, people and time
def rate_per_person (total_weight : ℕ) (total_people : ℕ) (total_time : ℕ) : ℕ :=
  total_weight / total_people / total_time

-- Theorem statement to prove
theorem days_to_shovel (t : ℕ) :
  (rate_per_person original_task_weight original_task_people original_task_time) * new_task_people * t = new_task_weight := sorry

end days_to_shovel_l729_729174


namespace geometric_sequence_l729_729135

-- Define the set and its properties
variable (A : Set ℕ) (a : ℕ → ℕ) (n : ℕ)
variable (h1 : 1 ≤ a 1) 
variable (h2 : ∀ (i : ℕ), 1 ≤ i → i < n → a i < a (i + 1))
variable (h3 : n ≥ 5)
variable (h4 : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → (a i) * (a j) ∈ A ∨ (a i) / (a j) ∈ A)

-- Statement to prove that the sequence forms a geometric sequence
theorem geometric_sequence : 
  ∃ (c : ℕ), c > 1 ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ n → a i = c^(i-1) := sorry

end geometric_sequence_l729_729135


namespace induction_inequality_proof_l729_729335

theorem induction_inequality_proof (n k : ℕ) (h1 : n > 1) (h2 : n = k + 1) :
  (∑ i in finset.range (2 * n + 1), (if (n + 1 ≤ i) ∧ (i ≤ 2 * n) then 1 / ↑i else 0)) > 13 / 24 :=
by sorry

end induction_inequality_proof_l729_729335


namespace number_of_groups_l729_729724

def groups_needed (range class_width : ℕ) : ℕ :=
  (range.toNat div class_width.toNat).succ

theorem number_of_groups :
  ∃ n, groups_needed 35 4 = n ∧ n = 9 := 
by
  use 9
  sorry

end number_of_groups_l729_729724


namespace ranking_arrangements_l729_729984

open Finset

theorem ranking_arrangements (students : Finset ℕ) (A B : ℕ) (ranking : ℕ → ℕ) :
  students = {1, 2, 3, 4, 5} →
  A ∉ {ranking 1} →
  B ∉ {ranking 5} →
  ∃ possible_rankings, possible_rankings.card = 78 :=
by
  sorry

end ranking_arrangements_l729_729984


namespace probability_two_heads_one_tail_l729_729710

noncomputable theory

open MeasureTheory

def fair_coin := probability_measure (pmf.bool (1/2))

def toss_coin (n : ℕ) := 
  repeat (pmf.bind fair_coin (λ b, pmf.pure b)) n

theorem probability_two_heads_one_tail :
  let p := toss_coin 3 in
  P {s | (s.count true = 2) ∧ (s.count false = 1)} = 3 / 8 :=
begin
  sorry
end

end probability_two_heads_one_tail_l729_729710


namespace dividend_calculation_l729_729233

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86) (hd : d = 52.7) (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 :=
by 
  sorry

end dividend_calculation_l729_729233


namespace trigonometric_identity_l729_729479

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 6 / 5 := 
sorry

end trigonometric_identity_l729_729479


namespace angle_DAC_is_45_degrees_l729_729554

-- Definitions for the conditions
variables {A B C E D : Type} [InnerProductSpace ℝ Type] -- Use ℝ for real numbers
variable {AB : A ≃ B}
variable {BC : B ≃ C}
variable {AE : A ≃ E}
variable {ED : E ≃ D}

-- Given Conditions
variable (isosceles_ABC : AB.angle = BC.angle)
variable (E_on_AB : E ∈ LineSegment A B)
variable (perpendicular_ED_BC : ED ⊥ BC)
variable (equal_AE_ED : AE.angle = ED.angle)

-- The goal is to find the angle DAC
theorem angle_DAC_is_45_degrees 
  (isosceles_ABC : isosceles_ABC)
  (E_on_AB : E_on_AB)
  (perpendicular_ED_BC : perpendicular_ED_BC)
  (equal_AE_ED : equal_AE_ED) : 
  angle A D C = 45 :=
begin
  sorry
end

end angle_DAC_is_45_degrees_l729_729554


namespace gcd_lcm_product_l729_729070

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l729_729070


namespace product_evaluation_l729_729058

noncomputable def evaluate_product : ℂ :=
  let w := Complex.exp (Complex.I * 2 * Real.pi / 11) in
  List.prod ((List.range 10).map (λ k => 3 - w^(k+1)))

theorem product_evaluation : evaluate_product = 88573 := by
  sorry

end product_evaluation_l729_729058


namespace probability_of_a_in_set_l729_729484

def quadratic_ineq_set : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

theorem probability_of_a_in_set (a : ℝ) (h : -2 ≤ a ∧ a ≤ 5) :
  Real.MeasurableSet (quadratic_ineq_set) ∧ (interval_integral (λ _ , 1) (-1) 3 / interval_integral (λ _ , 1) (-2) 5) = 4 / 7 :=
  sorry

end probability_of_a_in_set_l729_729484


namespace monotonic_functions_on_interval_l729_729679

section

-- Define the functions
def f_A (x : ℝ) := log x - x
def f_B (x : ℝ) := -x^3 / 3 + 3 * x^2 - 8 * x + 1
def f_C (x : ℝ) := sqrt 3 * sin x - 2 * x - cos x
def f_D (x : ℝ) := x / exp x

-- Define the derivatives
def f_A' (x : ℝ) := 1 / x - 1
def f_B' (x : ℝ) := -x^2 + 6 * x - 8
def f_C' (x : ℝ) := sqrt 3 * cos x - 2 + sin x
def f_D' (x : ℝ) := (1 - x) / exp x

-- The statement of the problem
theorem monotonic_functions_on_interval :
  (∀ x, 0 < x ∧ x < 1 → f_A' x > 0) ∧
  (∀ x, 0 < x ∧ x < 1 → f_B' x < 0) ∧
  (∀ x, 0 < x ∧ x < 1 → f_C' x ≤ 0) ∧
  (∀ x, 0 < x ∧ x < 1 → f_D' x > 0) :=
sorry

end

end monotonic_functions_on_interval_l729_729679


namespace perimeter_ratio_area_ratio_l729_729329

namespace TriangleRatio

variables {A a : ℝ} (h : A = 2 * a)

def perimeter_triangle_I (A : ℝ) : ℝ := 3 * A
def perimeter_triangle_II (a : ℝ) : ℝ := 3 * a
def area_triangle_I (A a : ℝ) (h : A = 2 * a) : ℝ := (A * A * Real.sqrt 3) / 4
def area_triangle_II (a : ℝ) : ℝ := (a * a * Real.sqrt 3) / 4

theorem perimeter_ratio (h : A = 2 * a) :
  perimeter_triangle_I A / perimeter_triangle_II a = 2 :=
by
  sorry

theorem area_ratio (h : A = 2 * a) :
  area_triangle_I A a h / area_triangle_II a = 4 :=
by
  sorry

end TriangleRatio

end perimeter_ratio_area_ratio_l729_729329


namespace diagonals_in_octagon_l729_729145

theorem diagonals_in_octagon (n : ℕ) (h : n = 8) : (nat.choose n 2) - n = 20 :=
by
  rw [h, nat.choose]
  sorry

end diagonals_in_octagon_l729_729145


namespace volume_calculation_l729_729422

noncomputable def volume_of_box
  (l w h : ℝ)
  (h1 : l * w = 18)
  (h2 : w * h = 50)
  (h3 : l * h = 45) : ℝ :=
  l * w * h

theorem volume_calculation
  (l w h : ℝ)
  (h1 : l * w = 18)
  (h2 : w * h = 50)
  (h3 : l * h = 45) :
  volume_of_box l w h h1 h2 h3 = 30 * real.sqrt 5 :=
sorry

end volume_calculation_l729_729422


namespace michael_average_speed_l729_729589

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed_l729_729589


namespace angle_BIC_115_l729_729537

theorem angle_BIC_115 (O I A B C : Type) [triangle ABC] 
  (hO_circumcenter: is_circumcenter O A B C) 
  (hI_incenter: is_incenter I A B C) 
  (hangle_BOC: angle B O C = 100) 
  : angle B I C = 115 := 
by 
  sorry

end angle_BIC_115_l729_729537


namespace max_f_l729_729979

noncomputable def f (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 + Real.sin θ)

theorem max_f : ∀ (θ : ℝ), 0 < θ ∧ θ < π → f θ ≤ (4 * Real.sqrt 3) / 9 :=
by
  sorry

end max_f_l729_729979


namespace maximum_term_of_sequence_l729_729134

noncomputable def a (n : ℕ) : ℝ := n * (3 / 4)^n

theorem maximum_term_of_sequence : ∃ n : ℕ, a n = a 3 ∧ ∀ m : ℕ, a m ≤ a 3 :=
by sorry

end maximum_term_of_sequence_l729_729134


namespace tangent_line_eq_AB_l729_729055

noncomputable def curve_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0

noncomputable def point_P := (3 : ℝ, 2 : ℝ)

def equation_line_AB : ℝ → ℝ → Prop := λ x y, 2 * x + 2 * y - 3 = 0

theorem tangent_line_eq_AB :
  ∀ (A B : ℝ × ℝ),
  (curve_eq A.1 A.2) ∧ (curve_eq B.1 B.2) ∧ (∃ t₁ t₂ : ℝ, t₁ * (A.1 - 3) + t₂ * (A.2 - 2) = 0 ∧ t₁ * (B.1 - 3) + t₂ * (B.2 - 2) = 0) →
  equation_line_AB (A.1) (A.2) =
  equation_line_AB (B.1) (B.2) :=
sorry

end tangent_line_eq_AB_l729_729055


namespace total_breakfast_cost_l729_729083

-- Define the costs of individual items.
def muffin_cost : ℝ := 2
def fruit_cup_cost : ℝ := 3
def coffee_cost : ℝ := 1.5

-- Define the discount on the combination of 2 muffins and 1 fruit cup.
def discount_rate : ℝ := 0.10

-- Define the counts of items for Francis and Kiera.
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def francis_coffee : ℕ := 1

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1
def kiera_coffee : ℕ := 2

-- Calculate the total cost for both Francis and Kiera, considering the discount.
theorem total_breakfast_cost :
  let
    francis_cost := (francis_muffins * muffin_cost) + (francis_fruit_cups * fruit_cup_cost) + (francis_coffee * coffee_cost),
    kiera_cost_before_discount := (kiera_muffins * muffin_cost) + (kiera_fruit_cup * fruit_cup_cost) + (kiera_coffee * coffee_cost),
    discount_value := discount_rate * (2 * muffin_cost + fruit_cup_cost),
    kiera_cost_after_discount := kiera_cost_before_discount - discount_value,
    total_cost := francis_cost + kiera_cost_after_discount
  in
    total_cost = 20.80 :=
by
    sorry

end total_breakfast_cost_l729_729083


namespace right_angle_triangle_exists_l729_729735

theorem right_angle_triangle_exists : 
    (∃ (a b c : ℕ), (a, b, c) ∈ {(1, 2, 3), (3, 4, 5), (6, 8, 10), (5, 10, 12)} ∧ a^2 + b^2 = c^2) ↔ 
    ((1, 2, 3) = (1, 2, 3)) :=
by
  sorry

end right_angle_triangle_exists_l729_729735


namespace evaluate_expression_l729_729922

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729922


namespace intersection_A_B_l729_729572

definition A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
definition B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} :=
by
  sorry

end intersection_A_B_l729_729572


namespace integral_value_l729_729316

theorem integral_value :
  ∫ x in 0..1, (2 * x + Real.exp x) = Real.exp 1 :=
by 
    sorry

end integral_value_l729_729316


namespace ship_speed_in_still_water_eq_25_l729_729725

-- Definitions and conditions
variable (x : ℝ) (h1 : 81 / (x + 2) = 69 / (x - 2)) (h2 : x ≠ -2) (h3 : x ≠ 2)

-- Theorem statement
theorem ship_speed_in_still_water_eq_25 : x = 25 :=
by
  sorry

end ship_speed_in_still_water_eq_25_l729_729725


namespace artist_paints_total_exposed_surface_area_l729_729737

def num_cubes : Nat := 18
def edge_length : Nat := 1

-- Define the configuration of cubes
def bottom_layer_grid : Nat := 9 -- Number of cubes in the 3x3 grid (bottom layer)
def top_layer_cross : Nat := 9 -- Number of cubes in the cross shape (top layer)

-- Exposed surfaces in bottom layer
def bottom_layer_exposed_surfaces : Nat :=
  let top_surfaces := 9 -- 9 top surfaces for 9 cubes
  let corner_cube_sides := 4 * 3 -- 4 corners, 3 exposed sides each
  let edge_cube_sides := 4 * 2 -- 4 edge (non-corner) cubes, 2 exposed sides each
  top_surfaces + corner_cube_sides + edge_cube_sides

-- Exposed surfaces in top layer
def top_layer_exposed_surfaces : Nat :=
  let top_surfaces := 5 -- 5 top surfaces for 5 cubes in the cross
  let side_surfaces_of_cross_arms := 4 * 3 -- 4 arms, 3 exposed sides each
  top_surfaces + side_surfaces_of_cross_arms

-- Total exposed surface area
def total_exposed_surface_area : Nat :=
  bottom_layer_exposed_surfaces + top_layer_exposed_surfaces

-- Problem statement
theorem artist_paints_total_exposed_surface_area :
  total_exposed_surface_area = 46 := by
    sorry

end artist_paints_total_exposed_surface_area_l729_729737


namespace modulus_of_z_l729_729552

-- Define the imaginary unit i
def i : ℂ := Complex.i

-- Define the complex number z as given
def z : ℂ := Complex.cos 3 + Complex.sin 3 * i

-- State the proof problem
theorem modulus_of_z : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l729_729552


namespace division_of_negatives_l729_729427

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end division_of_negatives_l729_729427


namespace number_of_students_playing_soccer_l729_729203

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer_l729_729203


namespace eval_expression_l729_729961

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729961


namespace find_n_l729_729065

noncomputable def angles_periodic_mod_eq (n : ℤ) : Prop :=
  -100 < n ∧ n < 100 ∧ Real.tan (n * Real.pi / 180) = Real.tan (216 * Real.pi / 180)

theorem find_n (n : ℤ) (h : angles_periodic_mod_eq n) : n = 36 :=
  sorry

end find_n_l729_729065


namespace john_age_l729_729380

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l729_729380


namespace find_number_l729_729079

theorem find_number (x : ℝ) 
  (h : x^3 * 9^3 / 679 = 549.7025036818851) : 
  x ≈ 8 :=
  sorry

end find_number_l729_729079


namespace max_triangle_area_of_perimeter_12_l729_729306

-- Definitions and constraints
def is_integer_side_lengths (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

def area (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Problem statement: proving the maximum area given conditions
theorem max_triangle_area_of_perimeter_12 :
  ∃ (a b c : ℕ), is_integer_side_lengths a b c ∧ is_triangle a b c ∧ perimeter a b c = 12 ∧ 
    area a b c = 4 * Real.sqrt 3 :=
  sorry

end max_triangle_area_of_perimeter_12_l729_729306


namespace eval_expression_l729_729942

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729942


namespace limit_seq_l729_729364

/-- Define the sequence a(n) as given in the problem -/ 
def seq (n : ℕ) := (3 - 4 * n : ℝ)^2 / ((n - 3 : ℝ)^3 - (n + 3 : ℝ)^3)

/-- The limit statement to prove -/
theorem limit_seq : 
  tendsto (fun n => seq n) at_top (𝓝 (-8/9 : ℝ)) :=
sorry

end limit_seq_l729_729364


namespace dick_jane_savings_1989_l729_729444

-- Definitions
variables (D J : ℝ)

-- Conditions
def dick_savings_1990 := 1.15 * D
def jane_savings_1990 := 0.85 * J
def total_savings_1990 := dick_savings_1990 + jane_savings_1990 = 2000

-- Goal to prove
theorem dick_jane_savings_1989 : dick_savings_1990 + jane_savings_1990 = 2000 → J = 1000 → D = J := 
begin
  intros h1 h2,
  rw h2 at h1,
  have h3 : 1.15 * J + 0.85 * J = 2000,
  { simp [dick_savings_1990, jane_savings_1990, h2] },
  linarith,
end

end dick_jane_savings_1989_l729_729444


namespace units_digit_n_l729_729462

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31^8) (h2 : m % 10 = 7) : n % 10 = 3 := 
sorry

end units_digit_n_l729_729462


namespace probability_of_choosing_a_quarter_l729_729395

theorem probability_of_choosing_a_quarter (total_value_quarters : ℝ) (total_value_nickels : ℝ) (total_value_pennies : ℝ) (value_of_quarter : ℝ) (value_of_nickel : ℝ) (value_of_penny : ℝ) :
  (total_value_quarters = 10) →
  (total_value_nickels = 5) →
  (total_value_pennies = 15) →
  (value_of_quarter = 0.25) →
  (value_of_nickel = 0.05) →
  (value_of_penny = 0.01) →
  let number_of_quarters := total_value_quarters / value_of_quarter in
  let number_of_nickels := total_value_nickels / value_of_nickel in
  let number_of_pennies := total_value_pennies / value_of_penny in
  let total_number_of_coins := number_of_quarters + number_of_nickels + number_of_pennies in
  (number_of_quarters / total_number_of_coins = 1 / 41) :=
begin
  intros,
  sorry
end

end probability_of_choosing_a_quarter_l729_729395


namespace tangent_line_at_point_l729_729627

-- Define the function
def curve (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

-- Define the tangent line equation
def tangent_line_eq (x y : ℝ) : Prop := 30 * x - y + 63 = 0

-- Point of tangency
def point_of_tangency : ℝ × ℝ := (-2, 3)

-- Define the slope of the curve at a given point
def slope_of_curve_at (x : ℝ) : ℝ := (6 * x^2 - 12 * x - 18)

-- Statement to prove that the tangent line at (-2, 3) has the given equation
theorem tangent_line_at_point :
  ∃ m b : ℝ, m = slope_of_curve_at (-2) ∧ b = 30 * (-2) + 3 +
  (∀ x y : ℝ, curve x = y → tangent_line_eq x y) :=
sorry

end tangent_line_at_point_l729_729627


namespace Elmer_eats_more_than_Penelope_l729_729248

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l729_729248


namespace integer_solutions_count_l729_729082

theorem integer_solutions_count :
  (∃ n : ℕ, n = { x : ℤ | (2 * x - 3)^2 + (-2 * x - 3)^2 ≤ 100 }.toFinset.card)
  ∧ n = 7 :=
by
  sorry

end integer_solutions_count_l729_729082


namespace imaginary_part_one_l729_729530

-- Given condition
def condition (z : ℂ) : Prop := (1 + z) * (1 - complex.I) = 2

-- The imaginary part of a complex number
def imag_part (z : ℂ) : ℝ := z.im

-- Prove that if the condition holds, the imaginary part of z is 1
theorem imaginary_part_one (z : ℂ) (h : condition z) : imag_part z = 1 :=
by
  sorry

end imaginary_part_one_l729_729530


namespace length_of_QS_l729_729558

theorem length_of_QS
    (P Q R S : Type) -- introducing triangular points and bisector point S
    (PQ QR PR : ℝ) -- lengths of the triangle sides
    (PQ_eq : PQ = 10) -- PQ = 10
    (QR_eq : QR = 24) -- QR = 24
    (PR_eq : PR = 26) -- PR = 26
    (QS_bisects_angle : angle_bisector Q QS) -- QS is the angle bisector of ∠Q
    : length QS = 102 / 9 :=
sorry

end length_of_QS_l729_729558


namespace eval_expression_l729_729907

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729907


namespace probability_all_same_color_l729_729385

theorem probability_all_same_color :
  let total_marbles := 5 + 6 + 7 in
  let total_ways := Nat.choose total_marbles 4 in
  let red_ways := 5.choose 4 in
  let white_ways := 6.choose 4 in
  let blue_ways := 7.choose 4 in
  let prob_all_same := (red_ways + white_ways + blue_ways) / total_ways in
  prob_all_same = 11 / 612 :=
by
  sorry

end probability_all_same_color_l729_729385


namespace P_c_eq_S_c_div_S_l729_729046

noncomputable def geometric_sequence (b q : ℝ) (n : ℕ) : List ℝ :=
(List.range n).map (λ k, b * q^k)

noncomputable def cubes (seq : List ℝ) : List ℝ :=
seq.map (λ x, x^3)

noncomputable def P_c (cubes: List ℝ) : ℝ :=
cubes.prod

noncomputable def S_c (cubes: List ℝ) : ℝ :=
cubes.sum

noncomputable def S'_c (cubes: List ℝ) : ℝ :=
(cubes.map (λ x, x⁻¹)).sum

theorem P_c_eq_S_c_div_S'_c {b q : ℝ} {n : ℕ} :
  let seq := geometric_sequence b q n
  let cubes_seq := cubes seq
  P_c cubes_seq = (S_c cubes_seq / S'_c cubes_seq)^(n / 2) :=
by
  -- Proof steps go here
  sorry

end P_c_eq_S_c_div_S_l729_729046


namespace proof_expr_is_neg_four_ninths_l729_729758

noncomputable def example_expr : ℚ := (-3 / 2) ^ 2021 * (2 / 3) ^ 2023

theorem proof_expr_is_neg_four_ninths : example_expr = (-4 / 9) := 
by 
  -- Here the proof would be placed
  sorry

end proof_expr_is_neg_four_ninths_l729_729758


namespace total_students_count_l729_729544

variable (T : ℕ)
variable (J : ℕ) (S : ℕ) (F : ℕ) (Sn : ℕ)

-- Given conditions:
-- 1. 26 percent are juniors.
def percentage_juniors (T J : ℕ) : Prop := J = 26 * T / 100
-- 2. 75 percent are not sophomores.
def percentage_sophomores (T S : ℕ) : Prop := S = 25 * T / 100
-- 3. There are 160 seniors.
def seniors_count (Sn : ℕ) : Prop := Sn = 160
-- 4. There are 32 more freshmen than sophomores.
def freshmen_sophomore_relationship (F S : ℕ) : Prop := F = S + 32

-- Question: Prove the total number of students is 800.
theorem total_students_count
  (hJ : percentage_juniors T J)
  (hS : percentage_sophomores T S)
  (hSn : seniors_count Sn)
  (hF : freshmen_sophomore_relationship F S) :
  F + S + J + Sn = T → T = 800 := by
  sorry

end total_students_count_l729_729544


namespace probability_three_heads_in_eight_tosses_l729_729011

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729011


namespace probability_three_heads_in_eight_tosses_l729_729000

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l729_729000


namespace range_of_a_l729_729117

variables {ℝ : Type} [linear_ordered_field ℝ] (f : ℝ → ℝ) (a : ℝ)

-- Conditions from a)
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x
def condition (f : ℝ → ℝ) (a : ℝ) := f a ≥ f (-2)

-- Proof problem in c)
theorem range_of_a (h₁ : even_function f) (h₂ : decreasing_on_nonneg f) (h₃ : condition f a) : -2 ≤ a ∧ a ≤ 2 :=
  sorry

end range_of_a_l729_729117


namespace area_correct_l729_729718

noncomputable def area_inside_circle_outside_rectangle : ℝ :=
  let length := 2
  let width := 1
  let radius := 1
  let diagonal := Real.sqrt (length^2 + width^2)
  let half_diagonal := diagonal / 2
  let area_circle := Real.pi * radius^2
  let area_rectangle := length * width
  let area_intersection := Real.pi - (Real.sqrt 5 - 2)
  area_intersection

theorem area_correct : 
  area_inside_circle_outside_rectangle = Real.pi - (Real.sqrt 5 - 2) :=
begin
  -- Proof omitted here
  sorry
end

end area_correct_l729_729718


namespace add_and_round_l729_729411

theorem add_and_round (a b : ℝ) (H1 : a = 49.213) (H2 : b = 27.569) : Float.round (a + b) = 77 := 
by
  sorry

end add_and_round_l729_729411


namespace eval_expression_l729_729906

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729906


namespace total_triangles_F15_l729_729428

-- Definition of the sequence of triangles
def T : ℕ → ℕ
| 1          := 1 -- Base case for F_1
| (n + 1)    := T n + 3*(n + 1) + 3 -- Recursive case for F_(n+1)

-- Theorem statement to prove
theorem total_triangles_F15 : T 15 = 400 :=
by
  sorry

end total_triangles_F15_l729_729428


namespace evaluate_expression_l729_729839

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729839


namespace cost_for_flour_for_two_cakes_l729_729761

theorem cost_for_flour_for_two_cakes 
    (packages_per_cake : ℕ)
    (cost_per_package : ℕ)
    (cakes : ℕ) 
    (total_cost : ℕ)
    (H1 : packages_per_cake = 2)
    (H2 : cost_per_package = 3)
    (H3 : cakes = 2)
    (H4 : total_cost = 12) :
    total_cost = cakes * packages_per_cake * cost_per_package := 
by 
    rw [H1, H2, H3]
    sorry

end cost_for_flour_for_two_cakes_l729_729761


namespace speed_of_slower_train_l729_729668

theorem speed_of_slower_train 
  (length_each_train : ℕ) 
  (speed_faster_train : ℕ) 
  (time_overtake : ℕ) 
  (distance_overtake : ℕ) 
  (conversion_factor : ℝ) 
  (V : ℕ) : 
  length_each_train = 50 → 
  speed_faster_train = 46 → 
  time_overtake = 36 → 
  distance_overtake = 100 → 
  conversion_factor = (5 / 18) → 
  100 = (46 - V) * 10 → 
  V = 36 :=
by {
  intros,
  sorry
}

end speed_of_slower_train_l729_729668


namespace exists_triangles_with_60_deg_angle_l729_729660

theorem exists_triangles_with_60_deg_angle : 
  ∃ (a b c : ℕ), gcd (gcd a b) c = 1 ∧ (is_triangle a b c) ∧ (one_angle_60_deg a b c) :=
begin
  sorry
end

-- Definitions to be used in the theorem (these would be defined in the library or can be defined as needed).

-- is_triangle means the three given side lengths form a valid triangle.
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- one_angle_60_deg means that one and only one angle of the triangle is 60 degrees.
def one_angle_60_deg (a b c : ℕ) : Prop :=
  ∃ (x y z : ℕ), (a = (x - y) / 2) ∧ (b = (x + y) / 2) ∧ (c = z / 2) ∧ (x^2 + 3 * (y^2) = z^2)

end exists_triangles_with_60_deg_angle_l729_729660


namespace average_daily_sales_price_reduction_for_target_profit_l729_729022

-- Statement of the conditions as definitions
def initial_sales := 20
def profit_per_item := 40
def additional_sales_per_dollar := 2
def minimum_profit_per_item := 25

-- Question 1: Proving the expression for average daily sales after the price reduction
theorem average_daily_sales (a : ℕ) : initial_sales + additional_sales_per_dollar * a = 20 + 2 * a :=
by sorry

-- Question 2: Proving the price reduction leads to the target daily profit
theorem price_reduction_for_target_profit : ∃ (x : ℕ), (40 - x) * (20 + 2 * x) = 1200 ∧ (40 - x) ≥ minimum_profit_per_item :=
by { use 10, split, sorry, sorry }

end average_daily_sales_price_reduction_for_target_profit_l729_729022


namespace octagon_angle_sum_eq_1080_l729_729656

-- Definitions
def octagon_interior_angle_sum : ℕ → ℕ := 
  λ n, (n - 2) * 180

-- Theorem statement
theorem octagon_angle_sum_eq_1080 : octagon_interior_angle_sum 8 = 1080 :=
sorry

end octagon_angle_sum_eq_1080_l729_729656


namespace evaluate_fraction_l729_729759

theorem evaluate_fraction : (3 : ℚ) / (2 - (3 / 4)) = (12 / 5) := 
by
  sorry

end evaluate_fraction_l729_729759


namespace claire_flour_cost_l729_729764

def num_cakes : ℕ := 2
def flour_per_cake : ℕ := 2
def cost_per_flour : ℕ := 3
def total_cost (num_cakes flour_per_cake cost_per_flour : ℕ) : ℕ := 
  num_cakes * flour_per_cake * cost_per_flour

theorem claire_flour_cost : total_cost num_cakes flour_per_cake cost_per_flour = 12 := by
  sorry

end claire_flour_cost_l729_729764


namespace evaluation_of_expression_l729_729902

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729902


namespace triangle_largest_smallest_angle_sum_l729_729185

theorem triangle_largest_smallest_angle_sum {A B C : Point} (BC AC AB : ℝ)
  (hBC : BC = 5) (hAC : AC = 7) (hAB : AB = 8) :
  let α := angle A BC
  let β := angle A C B
  let γ := angle B C A
  (α + β + γ = π) →
  α + β = 2 * π / 3 :=
by
  sorry

end triangle_largest_smallest_angle_sum_l729_729185


namespace chord_length_l729_729388

theorem chord_length (r d: ℝ) (h1: r = 5) (h2: d = 4) : ∃ EF, EF = 6 := by
  sorry

end chord_length_l729_729388


namespace order_of_variables_l729_729495

noncomputable def log0_1_0_2 := Real.log 0.2 / Real.log 0.1
noncomputable def log1_1_0_2 := Real.log 0.2 / Real.log 1.1
noncomputable def pow1_2_0_2 := (1.2 : ℝ)^(0.2 : ℝ)
noncomputable def pow1_1_0_2 := (1.1 : ℝ)^(0.2 : ℝ)

theorem order_of_variables :
  let a := log0_1_0_2
  let b := log1_1_0_2
  let c := pow1_2_0_2
  let d := pow1_1_0_2
  in c > d ∧ d > a ∧ a > b := 
by
  sorry

end order_of_variables_l729_729495


namespace interest_rate_is_10_percent_l729_729341

variable (P : Type) [Field P]

def simple_interest (principal rate : P) (time : ℕ) : P :=
  principal * rate * (time : P) / 100 

theorem interest_rate_is_10_percent 
  (principal : P) 
  (interest : P) 
  (time : ℕ) 
  (rate : P)
  (h_interest : simple_interest principal rate time = 400) 
  (h_time : time = 4) 
  (h_rate : rate = 10) : 
  rate = 10 :=
by
  sorry

end interest_rate_is_10_percent_l729_729341


namespace evaluate_expression_l729_729800

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729800


namespace eval_expr_l729_729815

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729815


namespace work_days_l729_729355

theorem work_days (A B C : ℝ)
  (h1 : A + B = 1 / 20)
  (h2 : B + C = 1 / 30)
  (h3 : A + C = 1 / 30) :
  (1 / (A + B + C)) = 120 / 7 := 
by 
  sorry

end work_days_l729_729355


namespace sum_of_interior_angles_of_octagon_l729_729654

theorem sum_of_interior_angles_of_octagon : 
  let n := 8 in
  (n - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_of_octagon_l729_729654


namespace find_alpha_l729_729551

variables (β θ t α : Real)
noncomputable def curve_C1 (β : Real) : Real × Real := (1 + cos β, sin β)
noncomputable def curve_C2 (θ : Real) : Real := 4 * cos θ
noncomputable def line_l (α t : Real) : Real × Real := (t * cos α, t * sin α)

theorem find_alpha (t ≠ 0) (h1 : (1 + cos β - t * cos α)^2 + (sin β - t * sin α)^2 = 3) 
                  (h2 : curve_C1 β = (1 + cos β, sin β)) (h3 : curve_C2 θ = 4 * cos θ) 
                  (h4 : line_l α t = (t * cos α, t * sin α))
                  (h5 : (π / 2 < α) ∧ (α < π)) :
  α = 5 * π / 6 :=
by 
  sorry

end find_alpha_l729_729551


namespace first_group_person_count_l729_729273

theorem first_group_person_count
  (P : ℕ)
  (h1 : P * 24 * 5 = 30 * 26 * 6) : 
  P = 39 :=
by
  sorry

end first_group_person_count_l729_729273


namespace evaluate_expression_l729_729787

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729787


namespace evaluation_of_expression_l729_729893

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729893


namespace anna_has_4_twenty_cent_coins_l729_729743

theorem anna_has_4_twenty_cent_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 59 - 3 * x = 24) : y = 4 :=
by {
  -- evidence based on the established conditions would be derived here
  sorry
}

end anna_has_4_twenty_cent_coins_l729_729743


namespace simplify_fraction_l729_729351

theorem simplify_fraction (c : ℚ) : (⟦5 + 6 * c⟧ / 9) + 3 = (⟦32 + 6 * c⟧ / 9) :=
sorry

end simplify_fraction_l729_729351


namespace eval_expression_l729_729828

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729828


namespace complex_pairs_l729_729980

-- Define the complex numbers x and y and the conditions in Lean
theorem complex_pairs : ∃ (f : Fin 24 → ℂ × ℂ), ∀ p : ℂ × ℂ, 
  (p ∈ set.range f → (p.1 ^ 4 * p.2 ^ 6 = 1 ∧ p.1 ^ 8 * p.2 ^ 3 = 1)) ∧
  ∀ q: ℂ × ℂ, (q.1 ^ 4 * q.2 ^ 6 = 1 ∧ q.1 ^ 8 * q.2 ^ 3 = 1) →
  ∃ i, f i = q :=
sorry

end complex_pairs_l729_729980


namespace defect_rate_product_l729_729276

theorem defect_rate_product (P1_defect P2_defect : ℝ) (h1 : P1_defect = 0.10) (h2 : P2_defect = 0.03) : 
  ((1 - P1_defect) * (1 - P2_defect)) = 0.873 → (1 - ((1 - P1_defect) * (1 - P2_defect)) = 0.127) :=
by
  intro h
  sorry

end defect_rate_product_l729_729276


namespace bug_total_distance_l729_729386

theorem bug_total_distance :
  let start := 3
  let mid1 := -4
  let mid2 := 7
  let end := 0
  abs(mid1 - start) + abs(mid2 - mid1) + abs(end - mid2) = 25 := by
  -- define the points
  let start := 3
  let mid1 := -4
  let mid2 := 7
  let end := 0

  -- calculate distances
  have h1 : abs(mid1 - start) = 7 := by
    apply abs_eq;
    norm_num,
  
  have h2 : abs(mid2 - mid1) = 11 := by
    apply abs_eq;
    norm_num,

  have h3 : abs(end - mid2) = 7 := by
    apply abs_eq;
    norm_num,

  -- sum the distances
  calc
    abs(mid1 - start) + abs(mid2 - mid1) + abs(end - mid2)
      = 7 + 11 + 7 : by rw [h1, h2, h3]
  ... = 25 : by norm_num

end bug_total_distance_l729_729386


namespace evaluate_expression_l729_729870

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729870


namespace evaluation_of_expression_l729_729895

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729895


namespace problem_statement_l729_729480

-- Defining the sequence of functions based on their derivatives
noncomputable def f : ℕ → (ℝ → ℝ)
| 1 := λ x, Real.cos x
| (n+1) := λ x, (f n)' x

-- The proof problem statement: we want to prove that f 2015 equals -cos for any x
theorem problem_statement : ∀ x, f 2015 x = -Real.cos x :=
by
  sorry

end problem_statement_l729_729480


namespace walmart_pot_stacking_l729_729035

theorem walmart_pot_stacking :
  ∀ (total_pots pots_per_set shelves : ℕ),
    total_pots = 60 →
    pots_per_set = 5 →
    shelves = 4 →
    (total_pots / pots_per_set / shelves) = 3 :=
by 
  intros total_pots pots_per_set shelves h1 h2 h3
  sorry

end walmart_pot_stacking_l729_729035


namespace lowry_earnings_l729_729587

def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def small_bonsai_sold : ℕ := 3
def big_bonsai_sold : ℕ := 5

def total_earnings (small_cost : ℕ) (big_cost : ℕ) (small_sold : ℕ) (big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

theorem lowry_earnings :
  total_earnings small_bonsai_cost big_bonsai_cost small_bonsai_sold big_bonsai_sold = 190 := 
by
  sorry

end lowry_earnings_l729_729587


namespace total_cost_is_225_l729_729621

def total_tickets : ℕ := 29
def cost_7_dollar_ticket : ℕ := 7
def cost_9_dollar_ticket : ℕ := 9
def number_of_9_dollar_tickets : ℕ := 11
def number_of_7_dollar_tickets : ℕ := total_tickets - number_of_9_dollar_tickets
def total_cost : ℕ := (number_of_9_dollar_tickets * cost_9_dollar_ticket) + (number_of_7_dollar_tickets * cost_7_dollar_ticket)

theorem total_cost_is_225 : total_cost = 225 := by
  sorry

end total_cost_is_225_l729_729621


namespace tan_theta_given_sin_theta_l729_729996

theorem tan_theta_given_sin_theta (θ : ℝ) (h1 : sin θ = 1/3) (h2 : θ ∈ set.Ioo (π / 2) π) : 
  tan θ = - (real.sqrt 2 / 4) :=
by sorry

end tan_theta_given_sin_theta_l729_729996


namespace length_GG_l729_729215

noncomputable def SegmentLength : Type := ℝ

structure Triangle :=
  (A B C : Point)
  (AB BC CA : SegmentLength)
  (AB_length : AB = 9)
  (BC_length : BC = 10)
  (CA_length : CA = 17)

structure Reflection :=
  (B B' : Point)
  (CA : SegmentLength)
  (reflection : reflect B' over_line CA = B)

structure Centroid :=
  (G : Point)
  (ABC : Triangle)
  (centroid : is_centroid G ABC)

structure ReflectionCentroid :=
  (G' : Point)
  (AB'C : Triangle)
  (centroid : is_centroid G' AB'C)

theorem length_GG' (ABC : Triangle) (CA : SegmentLength)
  (Breflection : Reflection ABC.B B' CA)
  (Gcentroid : Centroid ABC.G ABC)
  (G'reflectionCentroid : ReflectionCentroid G' AB'C) :
  segment_len (GG' ABC.G G' AB'C) = 48 / 17 :=
sorry

end length_GG_l729_729215


namespace ten_percent_of_x_is_17_85_l729_729375

-- Define the conditions and the proof statement
theorem ten_percent_of_x_is_17_85 :
  ∃ x : ℝ, (3 - (1/4) * 2 - (1/3) * 3 - (1/7) * x = 27) ∧ (0.10 * x = 17.85) := sorry

end ten_percent_of_x_is_17_85_l729_729375


namespace problem1_problem2_l729_729760

variable (x : ℝ)

theorem problem1 : 
  (3 * x + 1) * (3 * x - 1) - (3 * x + 1)^2 = -6 * x - 2 :=
sorry

theorem problem2 : 
  (6 * x^4 - 8 * x^3) / (-2 * x^2) - (3 * x + 2) * (1 - x) = 3 * x - 2 :=
sorry

end problem1_problem2_l729_729760


namespace sum_possible_n_k_l729_729628

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l729_729628


namespace number_of_redbirds_on_island_l729_729985

theorem number_of_redbirds_on_island (total_birds : ℕ) (bluebird_fraction : ℚ)
  (h_total : total_birds = 120) (h_fraction : bluebird_fraction = 5/6) :
  let redbird_fraction := 1 - bluebird_fraction in
  let redbirds := redbird_fraction * total_birds in
  redbirds = 20 := by
  sorry

end number_of_redbirds_on_island_l729_729985


namespace find_integer_divisible_by_24_and_cube_root_between_8_and_8_2_l729_729455

theorem find_integer_divisible_by_24_and_cube_root_between_8_and_8_2 : 
  ∃ (n : ℕ), (n > 0) ∧ (n % 24 = 0) ∧ (8 < real.cbrt n ∧ real.cbrt n < 8.2) ∧ n = 528 :=
by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_8_and_8_2_l729_729455


namespace eval_expression_l729_729962

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729962


namespace limit_seq_l729_729365

/-- Define the sequence a(n) as given in the problem -/ 
def seq (n : ℕ) := (3 - 4 * n : ℝ)^2 / ((n - 3 : ℝ)^3 - (n + 3 : ℝ)^3)

/-- The limit statement to prove -/
theorem limit_seq : 
  tendsto (fun n => seq n) at_top (𝓝 (-8/9 : ℝ)) :=
sorry

end limit_seq_l729_729365


namespace arithmetic_sequence_difference_l729_729277

noncomputable def arithmetic_sequence : ℕ → ℝ := sorry -- Placeholder for the actual arithmetic sequence definition

theorem arithmetic_sequence_difference (a : ℕ → ℝ) (d : ℝ) (h₁ : ∑ i in finset.range 150, a i = 150)
  (h₂ : ∑ i in finset.range 150 200, a i = 300) :
  (a 51 + d) - a 51 = 3 / 349 :=
  sorry

end arithmetic_sequence_difference_l729_729277


namespace problem_II_problem_III_l729_729487

section
variables {A : Type*} [linear_order A] [add_comm_group A]

def has_property_P (s : finset A) : Prop :=
∀ k ∈ s, 2 ≤ k → ∃ i j ∈ s, i ≤ j ∧ k = i + j

theorem problem_II (a1 a2 a3 a4 : A)
  (h1 : 1 = a1) (h2 : a1 < a2) (h3 : a2 < a3) (h4 : a3 < a4)
  (P : has_property_P {a1, a2, a3, a4}) : a4 ≤ 2 * a1 + a2 + a3 :=
by
  sorry 

theorem problem_III (A : finset A) (hA : A ∈ all_finsets)
  (P : has_property_P A) : find_minimum_value_n A :=
by
  sorry

end

end problem_II_problem_III_l729_729487


namespace range_of_a_l729_729123

open Real

theorem range_of_a (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |2^x₁ - a| = 1 ∧ |2^x₂ - a| = 1) ↔ 1 < a :=
by 
    sorry

end range_of_a_l729_729123


namespace eval_expr_l729_729817

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729817


namespace segment_AB_length_l729_729550

noncomputable def distance_between_points (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem segment_AB_length :
  ∃ (A B : ℝ × ℝ),
  (B.1 - B.2 = -3 / 2) ∧ 
  (B.2 ^ 2 = 8 * B.1) ∧
  distance_between_points A B = 4 * real.sqrt 2 := 
sorry

end segment_AB_length_l729_729550


namespace length_of_ab_equals_length_of_pq_l729_729089

-- Define the necessary structures and geometric entities
-- Point, Circle, and related predicates
noncomputable def Point := ℝ × ℝ
def Circle (center : Point) (radius : ℝ) := {p : Point // dist p center = radius}

-- Define the intersection points and lines

-- Given conditions translated to assumptions
variables {O C P Q A B : Point}
variables {r1 r2 : ℝ}
variables (h1 : C ∈ Circle O r1)
variables (h2 : (Circle O r2) ∩ (Circle C r1) = {P, Q})
variables (h3 : A ∈ Circle O r2 ∧ B ∈ Circle O r2)
variables (h4 : same_line CP A)(h5 : same_line CQ B)

-- Now state the necessary theorem
theorem length_of_ab_equals_length_of_pq : dist A B = dist P Q :=
sorry

end length_of_ab_equals_length_of_pq_l729_729089


namespace find_width_l729_729461

-- Definitions based on conditions
def length := 10
def height := 2
def surface_area := 136

-- Formula for surface area of the rectangular prism
def surface_area_formula (width : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

-- Statement to prove
theorem find_width (w : ℕ) (h : ℕ) (l : ℕ) (sa : ℕ) 
  (h_l : l = 10) (h_h : h = 2) (h_sa : sa = 136) :
  w = 4 := 
begin
  sorry -- Proof required
end

end find_width_l729_729461


namespace countable_and_uncountable_l729_729262

-- Definitions of the sets
def Z := Int
def Q := Rat
def QX := Polynomial Rat
def R := Real

-- Theorem stating that Z, Q, and QX are countable but R is not countable.
theorem countable_and_uncountable :
  (countable Z) ∧ (countable Q) ∧ (countable QX) ∧ ¬(countable R) := by
  sorry

end countable_and_uncountable_l729_729262


namespace distribution_X_P_X_eq_1_P_X_eq_0_distribution_series_l729_729064

variable {Ω : Type} -- the sample space
variable {P : MeasureTheory.ProbabilityMeasure Ω} -- the probability measure
variable (A₆ : Set Ω) -- event of rolling a 6

def X : Ω → ℕ := fun ω => if ω ∈ A₆ then 1 else 0

-- Assume A₆ has probability 1/6
axiom prob_A₆ : P A₆ = 1/6

-- X can take values 0 or 1
theorem distribution_X :
  ∀ ω, X ω = 0 ∨ X ω = 1 :=
by intros; dunfold X; split_ifs; simp

-- Calculate P(X=1)
theorem P_X_eq_1 : P {ω | X ω = 1} = 1/6 :=
by
  dunfold X
  rw ← Set.setOf_mem_eq
  rw MeasureTheory.Measure.map_apply
  sorry

-- Calculate P(X=0)
theorem P_X_eq_0 : P {ω | X ω = 0} = 5/6 :=
by 
  have : {ω | X ω = 0} = A₆ᶜ := by
    ext ω
    dunfold X
    split_ifs
    { simp [h] }
    { simp [h] }
  rw this
  rw prob_A₆
  simp
  sorry

-- Combined statement for the distribution of X
theorem distribution_series :
  P {ω | X ω = 0} = 5/6 ∧ P {ω | X ω = 1} = 1/6 :=
⟨P_X_eq_0, P_X_eq_1⟩

end distribution_X_P_X_eq_1_P_X_eq_0_distribution_series_l729_729064


namespace perpendicular_planes_l729_729114

-- Define planes and lines
variables (α β γ : Plane) (l m : Line)

-- Conditions of the problem
-- α is perpendicular to β
-- Intersection of α and β is the line l
-- γ is perpendicular to l
def conditions : Prop := α.perp β ∧ Plane.inter α β = l ∧ γ.perp l

-- Conclusion to prove: γ is perpendicular to β
theorem perpendicular_planes (h : conditions α β γ l) : γ.perp β :=
sorry

end perpendicular_planes_l729_729114


namespace students_with_uncool_parents_correct_l729_729317

def total_students : ℕ := 30
def cool_dads : ℕ := 12
def cool_moms : ℕ := 15
def cool_both : ℕ := 9

def students_with_uncool_parents : ℕ :=
  total_students - (cool_dads + cool_moms - cool_both)

theorem students_with_uncool_parents_correct :
  students_with_uncool_parents = 12 := by
  sorry

end students_with_uncool_parents_correct_l729_729317


namespace tracy_customers_l729_729327

theorem tracy_customers
  (total_customers : ℕ)
  (customers_bought_two_each : ℕ)
  (customers_bought_one_each : ℕ)
  (customers_bought_four_each : ℕ)
  (total_paintings_sold : ℕ)
  (h1 : total_customers = 20)
  (h2 : customers_bought_one_each = 12)
  (h3 : customers_bought_four_each = 4)
  (h4 : total_paintings_sold = 36)
  (h5 : 2 * customers_bought_two_each + customers_bought_one_each + 4 * customers_bought_four_each = total_paintings_sold) :
  customers_bought_two_each = 4 :=
by
  sorry

end tracy_customers_l729_729327


namespace rectangle_area_eq_l729_729402

theorem rectangle_area_eq (d : ℝ) (w : ℝ) (h1 : w = d / (2 * (5 : ℝ) ^ (1/2))) (h2 : 3 * w = (3 * d) / (2 * (5 : ℝ) ^ (1/2))) : 
  (3 * w^2) = (3 / 10) * d^2 := 
by sorry

end rectangle_area_eq_l729_729402


namespace eval_expression_l729_729903

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729903


namespace evaluate_expression_l729_729885

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729885


namespace prizeMoney_l729_729256

-- Define the condition that Rica got 3/8 of the prize money
def ricaShare (P : ℝ) : ℝ := (3 / 8) * P

-- Define the condition that she spent 1/5 of her prize money
def amountSpent (P : ℝ) : ℝ := (1 / 5) * ricaShare P

-- Define the remaining money condition
def remainingMoney (P : ℝ) : ℝ := ricaShare P - amountSpent P

-- Prove that the prize money P equals 1000 given the conditions
theorem prizeMoney (h : remainingMoney P = 300) : P = 1000 := 
by
  sorry

end prizeMoney_l729_729256


namespace intersection_is_neg2_l729_729581

-- Define sets A and B
def A : Set ℤ := {-3, -2, -1, 0, 1}
def B : Set ℤ := {x | x * x - 4 = 0}

-- Goal: Prove that the intersection of A and B is {-2}
theorem intersection_is_neg2 : A ∩ B = {-2} :=
by
  sorry

end intersection_is_neg2_l729_729581


namespace find_x_l729_729078

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x (x : ℝ) : 
  (sqrt x / sqrt 0.81 + sqrt 1.44 / sqrt 0.49 = 3.0751133491652576) → 
  x = 1.5 :=
by { sorry }

end find_x_l729_729078


namespace infinitely_many_non_representable_naturals_l729_729607

theorem infinitely_many_non_representable_naturals :
  ∃^∞ a : ℕ, ∀ (p : ℕ) (h : p.Prime) (n k : ℤ), a^2 ≠ p + n^(2*k) :=
sorry

end infinitely_many_non_representable_naturals_l729_729607


namespace series_sum_eq_six_statements_correctness_l729_729773

noncomputable def a : ℕ → ℝ
| 0       => 3
| (n + 1) => (a n) * 0.5

def sum_series (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (a k)

theorem series_sum_eq_six : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sum_series a n - 6| < ε :=
begin
  sorry -- proof of series sum convergence to 6
end

theorem statements_correctness : 
  (series_sum_eq_six → (∀ ε > 0, ∃ N : ℕ, ∀ k ≥ N, |a k - 0| < ε) ∧ 
   (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sum_series a n - 6| < ε) ∧ 
   (∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sum_series a n - L| < ε)) :=
begin
  sorry -- verification of statements 3, 4, and 5
end

end series_sum_eq_six_statements_correctness_l729_729773


namespace eval_expression_l729_729948

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729948


namespace evaluate_expression_l729_729925

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729925


namespace eval_expression_l729_729947

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729947


namespace find_angle_ACB_l729_729557

variables (A B C D E F : Type)
variables [triangle ABC]
variables (H1 : length AB = 3 * length AC)
variables (H2 : angle BAE = angle ACD)
variables (H3 : isosceles_triangle CFE CF FE)

theorem find_angle_ACB (A B C D E F : Point) :
  ∠ACB = 50 :=
sorry

end find_angle_ACB_l729_729557


namespace evaluation_of_expression_l729_729896

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729896


namespace angle_between_sum_and_difference_l729_729517

open Real EuclideanSpace

variables {V : Type*} [inner_product_space ℝ V]

/-- Definition of the condition -/
def magnitude_condition {v₁ v₂ : V} (hoc : ∥v₁ + v₂∥ = 2 * ∥v₁∥) (hom : ∥v₁ - v₂∥ = 2 * ∥v₁∥) := true

/-- Main theorem -/
theorem angle_between_sum_and_difference
  {v₁ v₂ : V}
  (h₀ : v₁ ≠ 0)
  (h₁ : v₂ ≠ 0)
  (h₂ : ∥v₁ + v₂∥ = 2 * ∥v₁∥)
  (h₃ : ∥v₁ - v₂∥ = 2 * ∥v₁∥) :
  angle (v₁ + v₂) (v₁ - v₂) = 2 * π / 3 :=
by {
  sorry
}

end angle_between_sum_and_difference_l729_729517


namespace eval_expression_l729_729968

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729968


namespace triangle_inequality_valid_x_values_l729_729297

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l729_729297


namespace number_of_subsets_of_five_element_set_is_32_l729_729308

theorem number_of_subsets_of_five_element_set_is_32 (M : Finset ℕ) (h : M.card = 5) :
    (2 : ℕ) ^ 5 = 32 :=
by
  sorry

end number_of_subsets_of_five_element_set_is_32_l729_729308


namespace eval_expression_l729_729919

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729919


namespace evaluate_expression_l729_729869

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729869


namespace faye_candies_final_count_l729_729081

def initialCandies : ℕ := 47
def candiesEaten : ℕ := 25
def candiesReceived : ℕ := 40

theorem faye_candies_final_count : (initialCandies - candiesEaten + candiesReceived) = 62 :=
by
  sorry

end faye_candies_final_count_l729_729081


namespace evaluate_expression_l729_729877

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729877


namespace solve_for_a8_l729_729639

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = a n + a (n + 1)

theorem solve_for_a8 (a : ℕ → ℕ) (h1 : sequence a) (h2 : a 7 = 120) : a 8 = 194 :=
by
  sorry

end solve_for_a8_l729_729639


namespace powers_of_2_not_powers_of_4_l729_729163

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end powers_of_2_not_powers_of_4_l729_729163


namespace tangent_line_through_point_l729_729978

-- Define the circle equation and the given point
def circle (x y : ℝ) := x^2 + y^2 = 5
def point_P := (-1, 2)

-- Statement of the proof problem
theorem tangent_line_through_point :
  ∃ k : ℝ, (∀ (x y : ℝ), circle x y → y - 2 = k * (x + 1)) ∧ (circle 0 0 → circle 0 0) → k = 1/2 →
  ∀ (x y : ℝ), y - 2 = 1/2 * (x + 1) → x - 2*y + 5 = 0 :=
sorry

end tangent_line_through_point_l729_729978


namespace factorize_x4_plus_81_l729_729059

noncomputable def factorize_poly (x : ℝ) : (ℝ × ℝ) :=
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  (p, q)

theorem factorize_x4_plus_81 : ∀ x : ℝ, (x^4 + 81) = (factorize_poly x).fst * (factorize_poly x).snd := by
  intro x
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  have h : x^4 + 81 = p * q
  { sorry }
  exact h

end factorize_x4_plus_81_l729_729059


namespace valid_zorg_sentences_count_l729_729200

/--
In the Zorgian language, there are 4 words: "zor", "glib", "mek", and "troz".
In a sentence, "zor" cannot come directly before "glib", and "mek" cannot come directly before "troz"; all other sentences are grammatically correct (including sentences with repeated words).
We want to determine the number of valid 3-word sentences.
-/
def zorg_words : list string := ["zor", "glib", "mek", "troz"]

def is_valid_zorg_sentence (sentence : list string) : Prop :=
  match sentence with
  | [_, "zor", "glib"] => false
  | ["zor", "glib", _] => false
  | [_, "mek", "troz"] => false
  | ["mek", "troz", _] => false
  | _ => true
  end

def count_valid_zorg_sentences : ℕ :=
  (list.permutations zorg_words).count is_valid_zorg_sentence

theorem valid_zorg_sentences_count : count_valid_zorg_sentences = 48 :=
by
  sorry

end valid_zorg_sentences_count_l729_729200


namespace smallest_divisor_28_l729_729459

theorem smallest_divisor_28 : ∃ (d : ℕ), d > 0 ∧ d ∣ 28 ∧ ∀ (d' : ℕ), d' > 0 ∧ d' ∣ 28 → d ≤ d' := by
  sorry

end smallest_divisor_28_l729_729459


namespace find_cost_price_l729_729690

-- Let CP (cost price) be a variable of type real.
variable (CP : ℝ)

-- Define SP (selling price) in terms of CP given the conditions.
def selling_price (CP : ℝ) : ℝ := 1.60 * CP

-- Define the equation given in the problem.
def equation (CP : ℝ) : Prop := selling_price CP = 2000

-- Main theorem to prove the cost price CP is Rs. 1250.
theorem find_cost_price : CP = 1250 :=
by
  have h : equation CP,
  sorry

end find_cost_price_l729_729690


namespace probability_winning_precisely_second_draw_expected_number_of_wins_replacement_l729_729540

-- Definitions for part (I)
def draw_two_white_first : ℕ := 2
def without_replacement_probability_first : ℚ := (4.choose draw_two_white_first) / (7.choose draw_two_white_first)

def remaining_red_balls_after_first_draw : ℕ := 3
def remaining_total_balls_after_first_draw : ℕ := 5
def at_least_one_red_second_draw : ℚ := ((3.choose 2) + ((3.choose 1) * (2.choose 1))) / (5.choose 2)

-- Problem (I)
theorem probability_winning_precisely_second_draw :
  without_replacement_probability_first * at_least_one_red_second_draw = 9 / 35 :=
sorry

-- Definitions for part (II)
def draw_with_replacement_probability: ℚ := ((3.choose 2) + ((3.choose 1) * (4.choose 1))) / (7.choose 2)
def trials : ℕ := 4

-- Problem (II)
theorem expected_number_of_wins_replacement :
  trials * draw_with_replacement_probability = 20 / 7 :=
sorry

end probability_winning_precisely_second_draw_expected_number_of_wins_replacement_l729_729540


namespace sum_of_extreme_primes_l729_729776

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes_upto_50 : List ℕ :=
  List.filter is_prime (List.range 51)

def smallest_prime_50 : ℕ :=
  List.minimum primes_upto_50

def largest_prime_50 : ℕ :=
  List.maximum primes_upto_50

theorem sum_of_extreme_primes : smallest_prime_50 + largest_prime_50 = 49 :=
  sorry

end sum_of_extreme_primes_l729_729776


namespace correct_statements_l729_729582

def f (x : ℝ) : ℝ := cos (x + π / 6)

theorem correct_statements :
  (∀ k : ℤ, k ≠ 0 → f(x) = f(x + -2 * π * k)) ∧
  (∀ x : ℝ, x = 5 * π / 6 → f(x) = f(π - x)) ∧
  (f(π / 3 + π) = 0) ∧
  (∀ x, π / 2 < x → x < π → ∀ x', x' ∈ Ioo (π / 2) π → ∀ x, f(x') < f(x)) :=
by
  sorry

end correct_statements_l729_729582


namespace person_walk_rate_proof_l729_729739

noncomputable def person_walk_rate : ℝ :=
  let v := 3 in
  v

theorem person_walk_rate_proof (escalator_rate : ℝ) (escalator_length : ℝ) (time : ℝ) (v : ℝ) :
  escalator_rate = 11 → escalator_length = 140 → time = 10 → 
  (escalator_length = (v + escalator_rate) * time) → v = 3 :=
by
  intros h1 h2 h3 h4
  -- Placeholder for proof
  sorry

end person_walk_rate_proof_l729_729739


namespace diagonals_in_octagon_l729_729147

theorem diagonals_in_octagon (n : ℕ) (h : n = 8) : (nat.choose n 2) - n = 20 :=
by
  rw [h, nat.choose]
  sorry

end diagonals_in_octagon_l729_729147


namespace max_vector_norm_diff_l729_729167

variables (a b c : Euclidean_Space ℝ) -- assuming Euclidean_Space ℝ as vector space
variables (h₁ : ∥a∥ = 1) (h₂ : ∥b∥ = 1) (h₃ : ∥c∥ = 1) -- unit vectors condition
variables (h₄ : inner a b = 0) -- a • b = 0
variables (h₅ : inner a c + inner b c = 1 / 2) -- a • c + b • c = 1/2

theorem max_vector_norm_diff : 
  (∥a - b∥ ^ 2) + (∥a - c∥ ^ 2) + (∥b - c∥ ^ 2) = 5 :=
by 
  sorry

end max_vector_norm_diff_l729_729167


namespace increasing_interval_sin_l729_729641

theorem increasing_interval_sin:
  ∀ (k : ℤ), 
  let y := λ x : ℝ, sin (π / 4 - 3 * x) in
  increasing_on y (set.Icc ((2 * k * π / 3) + (π / 4)) ((2 * k * π / 3) + (7 * π / 12))) :=
begin
  sorry
end

end increasing_interval_sin_l729_729641


namespace eval_expr_l729_729809

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729809


namespace oak_trees_initially_in_park_l729_729658

def initialOakTrees (new_oak_trees total_oak_trees_after: ℕ) : ℕ :=
  total_oak_trees_after - new_oak_trees

theorem oak_trees_initially_in_park (new_oak_trees total_oak_trees_after initial_oak_trees : ℕ) 
  (h_new_trees : new_oak_trees = 2) 
  (h_total_after : total_oak_trees_after = 11) 
  (h_correct : initial_oak_trees = 9) : 
  initialOakTrees new_oak_trees total_oak_trees_after = initial_oak_trees := 
by 
  rw [h_new_trees, h_total_after, h_correct]
  sorry

end oak_trees_initially_in_park_l729_729658


namespace circles_C1_C2_intersect_l729_729781

noncomputable def C1_center : (ℝ × ℝ) := (-1, -2)
noncomputable def C1_radius : ℝ := 2

noncomputable def C2_center : (ℝ × ℝ) := (1, -1)
noncomputable def C2_radius : ℝ := 3

noncomputable def distance_between_centers (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circles_intersect (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  let d := distance_between_centers c1 c2
  r2 - r1 < d ∧ d < r1 + r2

theorem circles_C1_C2_intersect :
  circles_intersect C1_center C2_center C1_radius C2_radius :=
by
  -- Proof goes here
  sorry

end circles_C1_C2_intersect_l729_729781


namespace area_triangle_ABC_l729_729204

noncomputable def side_length_of_square (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ :=
  (1 / 2) * leg * leg

theorem area_triangle_ABC :
  ∀ (A B C D E : Type)
  (AC BC : ℝ)
  (angle_ACB : ℝ)
  (area_square : ℝ),
  ACDE_is_square : Prop
  ∧ AC = BC
  ∧ angle_ACB = π / 2
  ∧ area_square = 10
  →
  area_of_isosceles_right_triangle (side_length_of_square area_square) = 5 :=
begin
  sorry
end

end area_triangle_ABC_l729_729204


namespace division_of_negatives_l729_729426

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end division_of_negatives_l729_729426


namespace concyclic_points_of_square_l729_729596

theorem concyclic_points_of_square 
  (A B C D K N L M : Point) 
  (hSquare : is_square A B C D) 
  (hK : K ∈ segment A B) 
  (hN : N ∈ segment A D) 
  (hRelation : dist2 A K * dist2 A N = 2 * dist2 B K * dist2 D N) 
  (hL : L ∈ intersection (line_through C K) (diagonal B D)) 
  (hM : M ∈ intersection (line_through C N) (diagonal B D)) :
  concyclic { K, L, M, N, A } :=
sorry

end concyclic_points_of_square_l729_729596


namespace matchsticks_left_l729_729041

def initial_matchsticks : ℕ := 30
def matchsticks_needed_2 : ℕ := 5
def matchsticks_needed_0 : ℕ := 6
def num_2s : ℕ := 3
def num_0s : ℕ := 1

theorem matchsticks_left : 
  initial_matchsticks - (num_2s * matchsticks_needed_2 + num_0s * matchsticks_needed_0) = 9 :=
by sorry

end matchsticks_left_l729_729041


namespace integer_values_x_possible_l729_729300

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l729_729300


namespace find_a_value_l729_729638

def quadratic_vertex_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = 2) → (y = 5) →
  a * (x - 2)^2 + 5 = y

def quadratic_passing_point_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = -1) → (y = -20) →
  a * (x - 2)^2 + 5 = y

theorem find_a_value : ∃ a : ℚ, quadratic_vertex_condition a ∧ quadratic_passing_point_condition a ∧ a = (-25)/9 := 
by 
  sorry

end find_a_value_l729_729638


namespace PQ_parallel_to_AB_3_times_l729_729205

-- Definitions for the problem
structure Rectangle :=
  (A B C D : Type)
  (AB AD : ℝ)
  (P Q : ℝ → ℝ)
  (P_speed Q_speed : ℝ)
  (time : ℝ)

noncomputable def rectangle_properties (R : Rectangle) : Prop :=
  R.AB = 4 ∧
  R.AD = 12 ∧
  ∀ t, 0 ≤ t → t ≤ 12 → R.P t = t ∧  -- P moves from A to D at 1 cm/s
  R.Q_speed = 3 ∧                     -- Q moves at 3 cm/s
  ∀ t, R.Q t = R.Q_speed * t ∧             -- Q moves from C to B and back
  ∃ s1 s2 s3, R.P s1 = 4 ∧ R.P s2 = 8 ∧ R.P s3 = 12 ∧
  (R.Q s1 = 3 ∨ R.Q s1 = 1) ∧
  (R.Q s2 = 6 ∨ R.Q s2 = 2) ∧
  (R.Q s3 = 9 ∨ R.Q s3 = 0)

theorem PQ_parallel_to_AB_3_times : 
  ∀ (R : Rectangle), rectangle_properties R → 
  ∃ (times : ℕ), times = 3 :=
by
  sorry

end PQ_parallel_to_AB_3_times_l729_729205


namespace evaluate_expression_l729_729927

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729927


namespace find_k_l729_729096

-- Define the sequence
def a (n : ℕ) : ℝ := n^2 + n * k + 2

-- Prove the inequality
theorem find_k (k : ℝ) (n : ℕ) (hn : 0 < n) : a (n + 1) > a n ↔ k > -3 :=
by
  -- Express the problem in Lean
  sorry

end find_k_l729_729096


namespace evaluate_expression_l729_729856

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729856


namespace evaluate_expression_l729_729874

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729874


namespace evaluate_expression_l729_729841

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729841


namespace statement_a_statement_c_l729_729142

-- Define the vectors a, b, c
def vector_a := (2, 1)
def vector_b := (1, -1)
variable (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
def vector_c := (m - 2, n)

-- Define dot product and perpendicular condition
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Statement A
theorem statement_a : dot_product vector_a vector_b = 1 := 
  sorry

-- Statement C
theorem statement_c (h : perpendicular (2 - 1, 1 + 1) vector_c) : m + 2 * n = 2 := 
  sorry

end statement_a_statement_c_l729_729142


namespace volume_of_P_ABC_correct_equation_of_line_through_M_correct_l729_729553

noncomputable def volume_of_triangular_pyramid (AB AC AP : ℝ × ℝ × ℝ) : ℝ :=
  let normal_vector := (1, 2, 0) in
  let h := (|normal_vector.fst * AP.fst + normal_vector.snd * AP.snd + normal_vector.trd * AP.trd|) /
    (Real.sqrt (normal_vector.fst ^ 2 + normal_vector.snd ^ 2 + normal_vector.trd ^ 2)) in
  let cosA := (AB.fst * AC.fst + AB.snd * AC.snd + AB.trd * AC.trd) /
    ((Real.sqrt (AB.fst ^ 2 + AB.snd ^ 2 + AB.trd ^ 2) * Real.sqrt (AC.fst ^ 2 + AC.snd ^ 2 + AC.trd ^ 2))) in
  let sinA := Real.sqrt (1 - cosA ^ 2) in
  let area_ABC := 0.5 * (Real.sqrt (AB.fst ^ 2 + AB.snd ^ 2 + AB.trd ^ 2) * Real.sqrt (AC.fst ^ 2 + AC.snd ^ 2 + AC.trd ^ 2) * sinA) in
  (1/3) * area_ABC * h

theorem volume_of_P_ABC_correct : volume_of_triangular_pyramid (2, -1, 3) (-2, 1, 0) (3, -1, 4) = 1/2 := by
  sorry

noncomputable def equation_of_line_through_M (M AB : ℝ × ℝ × ℝ) : String :=
  "⟨(1 - x) / " ++ toString AB.fst ++ ", (1 - y) / " ++ toString -AB.snd ++ ", (1 - z) / " ++ toString AB.trd ++ "⟩"

theorem equation_of_line_through_M_correct : equation_of_line_through_M (1, 1, 1) (2, -1, 3) = "⟨(1 - x) / 2, (1 - y) / -1, (1 - z) / 3⟩" := by 
  sorry

end volume_of_P_ABC_correct_equation_of_line_through_M_correct_l729_729553


namespace find_p_at_0_l729_729713

noncomputable def p (x : ℝ) : ℝ := sorry

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d, p = λ x, x^4 + a * x^3 + b * x^2 + c * x + d

theorem find_p_at_0
    (h_monic : is_monic_quartic p)
    (h1 : p (-2) = -4)
    (h2 : p (1) = -1)
    (h3 : p (3) = -9)
    (h4 : p (5) = -25) :
  p 0 = -30 := 
sorry

end find_p_at_0_l729_729713


namespace hyperbola_m_value_l729_729288

theorem hyperbola_m_value
  (m : ℝ)
  (h1 : 3 * m * x^2 - m * y^2 = 3)
  (focus : ∃ c, (0, c) = (0, 2)) :
  m = -1 :=
sorry

end hyperbola_m_value_l729_729288


namespace find_T_41216_l729_729315

variables {V : Type*} [normed_group V] [normed_space ℝ V]

def T (v : V) : V := sorry  -- The transformation T

axiom T_linearity : ∀ (a b : ℝ) (v w : V), T(a • v + b • w) = a • T(v) + b • T(w)
axiom T_cross_product : ∀ (v w : V), T(v × w) = T(v) × T(w)
axiom T_74 : T ⟨7, 8, 4⟩ = ⟨5, -2, 9⟩
axiom T_748 : T ⟨-7, 4, 8⟩ = ⟨5, 9, -2⟩

theorem find_T_41216 : T ⟨4, 12, 16⟩ = ⟨15, -3, 10⟩ := sorry

end find_T_41216_l729_729315


namespace repeating_decimal_fraction_l729_729450

noncomputable def repeating_decimal := 4.66666 -- Assuming repeating forever

theorem repeating_decimal_fraction : repeating_decimal = 14 / 3 :=
by 
  sorry

end repeating_decimal_fraction_l729_729450


namespace evaluate_expression_l729_729795

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729795


namespace determine_y_l729_729779

def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem determine_y (y : ℝ) (h : diamond 4 y = 30) : y = 5 / 3 :=
by sorry

end determine_y_l729_729779


namespace jack_needs_additional_money_l729_729563

noncomputable def pair_of_socks_cost : Float := 9.50
noncomputable def number_of_pairs_of_socks : Int := 2
noncomputable def shoes_cost : Float := 92
noncomputable def soccer_ball_cost : Float := 25
noncomputable def sports_bag_cost : Float := 35
noncomputable def discount_shoes : Float := 0.10
noncomputable def discount_bag : Float := 0.20
noncomputable def current_money : Float := 40

noncomputable def total_cost_of_items : Float :=
  number_of_pairs_of_socks * pair_of_socks_cost +
  shoes_cost * (1 - discount_shoes) +
  soccer_ball_cost +
  sports_bag_cost * (1 - discount_bag)

noncomputable def additional_money_needed : Float :=
  total_cost_of_items - current_money

theorem jack_needs_additional_money : additional_money_needed = 114.80 := by
  sorry -- proof placeholder

end jack_needs_additional_money_l729_729563


namespace sum_of_first_n_terms_l729_729460

theorem sum_of_first_n_terms (n : ℕ) : 
  (∑ i in Finset.range n, (i + 1 : ℝ) + (1 / 2) ^ (i + 1)) = (n * (n + 1) / 2 : ℝ) + 1 - (1 / 2) ^ n :=
by
  sorry

end sum_of_first_n_terms_l729_729460


namespace arithmetic_sequence_max_value_l729_729771

theorem arithmetic_sequence_max_value 
  (S : ℕ → ℤ)
  (k : ℕ)
  (h1 : 2 ≤ k)
  (h2 : S (k - 1) = 8)
  (h3 : S k = 0)
  (h4 : S (k + 1) = -10) :
  ∃ n, S n = 20 ∧ (∀ m, S m ≤ 20) :=
sorry

end arithmetic_sequence_max_value_l729_729771


namespace part1_part2_l729_729372

-- Part 1
theorem part1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

-- Part 2
theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := sorry

end part1_part2_l729_729372


namespace system_of_equations_solution_l729_729649

theorem system_of_equations_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 3) : 
  x = 4 ∧ y = 1 :=
by
  sorry

end system_of_equations_solution_l729_729649


namespace min_value_of_function_l729_729106

theorem min_value_of_function (α : ℝ) (hα : 0 < α ∧ α < Real.pi) :
  (∃ y_min, ∀ t : ℝ, -1 < t ∧ t < 1 → y_min ≤ 2 * t^2 - 3 * t + 5) ∧ y_min = 31/8 :=
by
  let y := -2 * sin α ^ 2 - 3 * cos α + 7
  have y_eq : y = 2 * cos α ^ 2 - 3 * cos α + 5 := by
    calc
      y = -2 * sin α ^ 2 - 3 * cos α + 7 : rfl
      ... = -2 * (1 - cos α ^ 2) - 3 * cos α + 7 : by rw [← Real.sin_sq]
      ... = 2 * cos α ^ 2 - 3 * cos α + 5 : by ring
  sorry

end min_value_of_function_l729_729106


namespace geom_seq_ration_l729_729468

variable {α : Type*} [LinearOrderedField α]
variables (a1 q : α)
variable (n : ℕ)

def geom_seq := λ n : ℕ, a1 * q^n
def all_pos (s : ℕ → α) := ∀ n, 0 < s n
def common_ratio (q : α) := q^2 = 4

theorem geom_seq_ration : all_pos (geom_seq a1 q) → common_ratio q →
  (geom_seq a1 q 2 + geom_seq a1 q 3) / (geom_seq a1 q 4 + geom_seq a1 q 5) = (1 : α) / 4 :=
by sorry

end geom_seq_ration_l729_729468


namespace richard_distance_l729_729611

noncomputable theory

variables (v t : ℝ) (d : ℝ)

-- Condition 1: Initial distance equation
def initial_distance_eq : Prop := d = v * t

-- Condition 2: Distance after first speed change
def first_speed_change_eq : Prop := d = (2 / 3 * v) + (v - 1) * t

-- Condition 3: Distance after second speed change
def second_speed_change_eq : Prop := d = (2 / 3 * v) + (3 / 4 * (v - 1)) + (v - 2) * t

-- Main theorem
theorem richard_distance : 
  initial_distance_eq v t d ∧ 
  first_speed_change_eq v t d ∧ 
  second_speed_change_eq v t d → 
  d = 54 :=
by
  /- We have set up the conditions and need to prove the distance -/
  sorry

end richard_distance_l729_729611


namespace positive_integer_solution_eq_l729_729440

theorem positive_integer_solution_eq :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (xyz + 2 * x + 3 * y + 6 * z = xy + 2 * xz + 3 * yz) ∧ (x, y, z) = (4, 3, 1) := 
by
  sorry

end positive_integer_solution_eq_l729_729440


namespace aiyanna_more_cookies_than_alyssa_l729_729731

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end aiyanna_more_cookies_than_alyssa_l729_729731


namespace range_of_f_over_x_l729_729497

-- Definitions based on conditions
variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable (h1 : ∀ x, f x = f (-x)) -- f is even
variable (h2 : ∀ x, f' x = deriv f x) -- derivative exists
variable (h3 : ∀ x ∈ Iic (0 : ℝ), monotone_on f (Iic (0 : ℝ)) ∧ f (-3) = 0) -- unique zero point at -3 for x ∈ (-∞, 0]
variable (h4 : ∀ x, x * f' x < f (-x)) -- the given inequality

-- The statement to prove
theorem range_of_f_over_x (h1 : ∀ x, f x = f (-x))
    (h2 : ∀ x, f' x = deriv f x)
    (h3 : ∀ x, x ∈ Iic (0 : ℝ) → f (-3) = 0)
    (h4 : ∀ x, x * f' x < f (-x)) :
    {x : ℝ | f x / x ≤ 0} = ↑(Iic (-3)) ∪ ↑(Ici (3)) := 
by {
  sorry
}

end range_of_f_over_x_l729_729497


namespace eval_expression_l729_729967

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729967


namespace exists_four_digit_number_with_property_l729_729561

-- Definition: A four-digit number with distinct non-zero digits
def isFourDigitNaturalNumberWithDistinctNonZeroDigits (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  let digits := (List.map digitChar $ n.digitsDec) in
  List.Nodup (List.filter ((!=) 0) digits) ∧ 
  List.Forall (λ c, (c ≠ '0')) digits

-- Definition: Reverse a four-digit number
def reverseNumber (n : ℕ) : ℕ :=
  let digits := n.digitsDec in
  let revDigits := digits.reverse in
  revDigits.foldr (λ (d : ℕ) (acc : ℕ), acc * 10 + d) 0

-- Theorem Statement
theorem exists_four_digit_number_with_property :
  ∃ n : ℕ, isFourDigitNaturalNumberWithDistinctNonZeroDigits n ∧ 
           101 ∣ (n + reverseNumber n) :=
by
  sorry

end exists_four_digit_number_with_property_l729_729561


namespace M_is_on_ray_AB_l729_729099

variables (x1 y1 x2 y2 λ : ℝ)
def M_coord := (1 - λ) • (x1, y1) + λ • (x2, y2)

theorem M_is_on_ray_AB (Hλ : 0 ≤ λ) :
  ∃ (x0 y0 : ℝ), (x0, y0) = M_coord x1 y1 x2 y2 λ :=
by {
  use (1 - λ) * x1 + λ * x2,
  use (1 - λ) * y1 + λ * y2,
  refl,
}

end M_is_on_ray_AB_l729_729099


namespace solve_for_c_l729_729268

theorem solve_for_c (c : ℚ) :
  (c - 35) / 14 = (2 * c + 9) / 49 →
  c = 1841 / 21 :=
by
  sorry

end solve_for_c_l729_729268


namespace minimum_value_proof_l729_729088

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1 / a + 2 / b

theorem minimum_value_proof :
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (a + b = 1) ∧ (minimum_value a b (by sorry) (by sorry) (by sorry) = 3 + 2 * Real.sqrt 2) :=
by
  use (1 / (1 + Real.sqrt 2))
  use (1 / (2 * (1 + Real.sqrt 2)))
  split
  -- Proof for 0 < a
  sorry
  split
  -- Proof for 0 < b
  sorry
  split
  -- Proof for a + b = 1
  sorry
  -- Proof for minimum_value = 3 + 2 * sqrt 2
  sorry

end minimum_value_proof_l729_729088


namespace johns_age_l729_729378

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l729_729378


namespace leaves_blew_away_correct_l729_729590

-- Define the initial number of leaves Mikey had.
def initial_leaves : ℕ := 356

-- Define the number of leaves Mikey has left.
def leaves_left : ℕ := 112

-- Define the number of leaves that blew away.
def leaves_blew_away : ℕ := initial_leaves - leaves_left

-- Prove that the number of leaves that blew away is 244.
theorem leaves_blew_away_correct : leaves_blew_away = 244 :=
by sorry

end leaves_blew_away_correct_l729_729590


namespace area_after_reduction_l729_729586

-- Define the initial dimensions of the card
def initial_length : ℕ := 5
def initial_width : ℕ := 3

-- Define the new dimensions when both sides are shortened by 2 inches
def new_length := initial_length - 2
def new_width := initial_width - 2

-- Prove that the new area is 3 square inches
theorem area_after_reduction : new_length * new_width = 3 := by
  -- Calculation
  have length_eq : new_length = 3 := by rfl
  have width_eq : new_width = 1 := by rfl
  rw [length_eq, width_eq]
  norm_num

end area_after_reduction_l729_729586


namespace ratio_black_white_l729_729278

-- Definitions of the parameters
variables (B W : ℕ)
variables (h1 : B + W = 200)
variables (h2 : 30 * B + 25 * W = 5500)

theorem ratio_black_white (B W : ℕ) (h1 : B + W = 200) (h2 : 30 * B + 25 * W = 5500) :
  B = W :=
by
  -- Proof omitted
  sorry

end ratio_black_white_l729_729278


namespace intersection_point_AB_CD_l729_729546

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def pointA : Point3D := ⟨8, -5, 5⟩
noncomputable def pointB : Point3D := ⟨18, -15, 10⟩
noncomputable def pointC : Point3D := ⟨1, 5, -7⟩
noncomputable def pointD : Point3D := ⟨3, -3, 13⟩

noncomputable def intersection_point (A B C D : Point3D) : Point3D :=
  let t := -12 / 5 in
  ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩

theorem intersection_point_AB_CD 
  (A B C D : Point3D) :
  intersection_point A B C D = ⟨-16, 7, -7⟩ :=
  sorry

end intersection_point_AB_CD_l729_729546


namespace matrix_vector_evaluation_l729_729220

variable {α : Type*} [Field α] {m n : Type*} [Fintype m] [Fintype n]

def u : Matrix m n α := ![3, -2]
def z : Matrix m n α := ![-1, 4]
def M : Matrix n m α := sorry  -- Placeholder for the matrix M

theorem matrix_vector_evaluation (a b c d : α) :
  (M ⬝ ![u, z]) = ![a, b, c, d] →
  M ⬝ (3 • u - 2 • z) = ![11, -14] :=
by
  sorry

end matrix_vector_evaluation_l729_729220


namespace simplify_expr_1_simplify_expr_2_l729_729267

-- The first problem
theorem simplify_expr_1 (a : ℝ) : 2 * a^2 - 3 * a - 5 * a^2 + 6 * a = -3 * a^2 + 3 * a := 
by
  sorry

-- The second problem
theorem simplify_expr_2 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l729_729267


namespace evaluate_expression_l729_729840

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729840


namespace hotel_flat_fee_l729_729016

theorem hotel_flat_fee :
  ∃ (f n : ℝ), f + 3 * n = 205 ∧ f + 6 * n = 350 ∧ f = 60 :=
by
  -- Definitions based on conditions
  existsi (60 : ℝ)
  existsi (48.33̅: ℝ)
  split
  { sorry }
  { split
    { sorry }
    { refl }
  }

end hotel_flat_fee_l729_729016


namespace abs_expression_equals_one_l729_729463

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one_l729_729463


namespace angle_CAG_measures_15_degrees_l729_729543

-- Definitions of the geometric objects and properties
variables {α : Type*} [euclidean_plane α]
variables {A B C F G : α} 

-- Conditions
axiom eqt_triangle : equilateral_triangle A B C
axiom common_vertex_B : B ∈ triangle_vertices (equilateral_triangle.to_simplex eqt_triangle)
axiom sq_BCFG : square B C F G

-- Theorem
theorem angle_CAG_measures_15_degrees :
  ∠ C A G = 15 := 
sorry

end angle_CAG_measures_15_degrees_l729_729543


namespace problem_m_property_l729_729531

def has_m_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → e^(x) * f(x) < e^(y) * f(y)

theorem problem_m_property : has_m_property (λ x : ℝ, 2^x) :=
sorry

end problem_m_property_l729_729531


namespace absolute_product_slopes_l729_729033

variable (e : ℝ) (k1 k2 : ℝ)
variable (C1 C2 : set ℝ)
variable (A B : ℝ × ℝ)

-- Assume C1 and C2 are ellipses with eccentricity e
-- Points A (right vertex) and B (top vertex) on C2
-- Tangents l1 and l2 to C1 are drawn through A and B respectively.
-- Slopes of l1 and l2 are k1 and k2

theorem absolute_product_slopes (A B : ℝ × ℝ)
  (C1 C2 : set ℝ) (hC1 : ∃ e, C1 = {p : ℝ | p = e}) 
  (hC2 : ∃ e, C2 = {p : ℝ | p = e}) 
  (hA : A = (e, 0)) 
  (hB : B = (0, e)) 
  (hk1 : k1 = - e / e) 
  (hk2 : k2 = - e / e) :
  |k1 * k2| = 1 - e^2 :=
sorry

end absolute_product_slopes_l729_729033


namespace eval_expression_l729_729911

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729911


namespace minimum_value_expression_l729_729101

theorem minimum_value_expression {a b c : ℝ} :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 13 → 
  (∃ x, x = (a^2 + b^3 + c^4 + 2019) / (10 * b + 123 * c + 26) ∧ ∀ y, y ≤ x) →
  x = 4 :=
by
  sorry

end minimum_value_expression_l729_729101


namespace evaluate_expression_l729_729793

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729793


namespace circles_different_radii_no_three_common_tangents_l729_729666

theorem circles_different_radii_no_three_common_tangents (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ)
  (h_diff : r1 ≠ r2) :
  ¬ (∃ t : ℕ, t = 3 ∧ (are_common_tangents t (circle c1 r1) (circle c2 r2))) := 
sorry

end circles_different_radii_no_three_common_tangents_l729_729666


namespace max_sum_of_inverses_l729_729577

open Set

theorem max_sum_of_inverses 
  (n : ℕ) (h : n ≥ 5)
  (a : Fin n → ℕ) 
  (ha : Function.Injective a)
  (hb : ∀ A B : Finset (Fin n), A ≠ B → A.nonempty → B.nonempty → (A.sum (λ i => a i) ≠ B.sum (λ i => a i))) :
  (Finset.univ.sum (λ i => (1 : ℚ) / a i) ≤ 2 - 1 / 2 ^ (n - 1)) :=
by
  sorry

end max_sum_of_inverses_l729_729577


namespace total_wattage_new_l729_729405

def light_a_wattage_new (orig : ℝ) (increase_perc : ℝ) : ℝ :=
  orig + (orig * increase_perc / 100)

def light_b_wattage_new (orig : ℝ) (increase_perc : ℝ) : ℝ :=
  orig + (orig * increase_perc / 100)

def light_c_wattage_new (orig : ℝ) (increase_perc : ℝ) : ℝ :=
  orig + (orig * increase_perc / 100)

def light_d_wattage_new (orig : ℝ) (increase_perc : ℝ) : ℝ :=
  orig + (orig * increase_perc / 100)

theorem total_wattage_new :
  let light_a_orig := 60,
      light_b_orig := 40,
      light_c_orig := 50,
      light_d_orig := 80,
      light_a_perc := 12,
      light_b_perc := 20,
      light_c_perc := 15,
      light_d_perc := 10
  light_a_wattage_new light_a_orig light_a_perc +
  light_b_wattage_new light_b_orig light_b_perc +
  light_c_wattage_new light_c_orig light_c_perc +
  light_d_wattage_new light_d_orig light_d_perc = 260.7 := by
  sorry

end total_wattage_new_l729_729405


namespace midpoint_and_trisection_paths_l729_729331

-- Define the problem setting

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}

def trisection_point (A B : Point) : Point :=
  {x := (2 * A.x + B.x) / 3, y := (2 * A.y + B.y) / 3}

def point_on_ray_from (O : Point) (θ : ℝ) (t : ℝ) : Point :=
  {x := O.x + t * cos θ, y := O.y + t * sin θ}

-- The main theorem
theorem midpoint_and_trisection_paths (O A B : Point) (θ : ℝ)
  (t : ℝ → ℝ) (equal_speed : ∀ (s1 s2 : ℝ), s1 = s2 → t s1 = t s2) :
  ∃ (F H f_direction : Point), 
    (∀ t : ℝ, midpoint (point_on_ray_from O θ (t t)) (point_on_ray_from O (θ + π) (t t)) = 
     {x := F.x + t * cos θ, y := F.y + t * sin θ}) ∧
    (∀ t : ℝ, trisection_point (point_on_ray_from O θ (t t)) (point_on_ray_from O (θ + π) (t t)) =
     {x := H.x + t * f_direction.x, y := H.y + t * f_direction.y}) :=
sorry

end midpoint_and_trisection_paths_l729_729331


namespace hilary_stalks_l729_729144

-- Define the given conditions
def ears_per_stalk : ℕ := 4
def kernels_per_ear_first_half : ℕ := 500
def kernels_per_ear_second_half : ℕ := 600
def total_kernels : ℕ := 237600

-- Average number of kernels per ear
def average_kernels_per_ear : ℕ := (kernels_per_ear_first_half + kernels_per_ear_second_half) / 2

-- Total number of ears based on total kernels
noncomputable def total_ears : ℕ := total_kernels / average_kernels_per_ear

-- Total number of stalks based on total ears
noncomputable def total_stalks : ℕ := total_ears / ears_per_stalk

-- The main theorem to prove
theorem hilary_stalks : total_stalks = 108 :=
by
  sorry

end hilary_stalks_l729_729144


namespace sufficient_not_necessary_l729_729129

noncomputable def f (x a : ℝ) := x^2 - 2*a*x + 1

def no_real_roots (a : ℝ) : Prop := 4*a^2 - 4 < 0

def non_monotonic_interval (a m : ℝ) : Prop := m < a ∧ a < m + 3

def A := {a : ℝ | -1 < a ∧ a < 1}
def B (m : ℝ) := {a : ℝ | m < a ∧ a < m + 3}

theorem sufficient_not_necessary (x : ℝ) (m : ℝ) :
  (x ∈ A → x ∈ B m) → (A ⊆ B m) ∧ (exists a : ℝ, a ∈ B m ∧ a ∉ A) →
  -2 ≤ m ∧ m ≤ -1 := by 
  sorry

end sufficient_not_necessary_l729_729129


namespace eval_expression_l729_729917

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729917


namespace evaluate_expression_l729_729849

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729849


namespace students_choices_l729_729318

theorem students_choices (n_students n_lectures : ℕ) (h_students : n_students = 5) (h_lectures : n_lectures = 3) : 
  (n_lectures ^ n_students = 3 ^ 5) :=
by
  rw [h_students, h_lectures]
  exact rfl

end students_choices_l729_729318


namespace diagonals_in_octagon_l729_729146

theorem diagonals_in_octagon (n : ℕ) (h : n = 8) : (nat.choose n 2) - n = 20 :=
by
  rw [h, nat.choose]
  sorry

end diagonals_in_octagon_l729_729146


namespace explain_punctuality_behavior_l729_729599

-- Define the conditions for the problem
def large_group_punctuality (n : ℕ) : Prop :=
  n = 50 → ∀ t ∈ tourists(n), shows_up_on_time t

def small_group_punctuality (n : ℕ) : Prop :=
  n < 50 → ∃ t ∈ tourists(n), ¬ shows_up_on_time t

-- Define economic arguments function
-- These functions are mocked here as abbreviation as the original context is textual explanation 
def wealth_levels_argument : Prop := sorry
def value_perception_argument : Prop := sorry

-- Main theorem statement
theorem explain_punctuality_behavior (n : ℕ) :
  large_group_punctuality n ∧ small_group_punctuality n ↔ 
  wealth_levels_argument ∧ value_perception_argument := 
sorry

end explain_punctuality_behavior_l729_729599


namespace more_cats_than_dogs_l729_729685

theorem more_cats_than_dogs (total_animals : ℕ) (cats : ℕ) (h1 : total_animals = 60) (h2 : cats = 40) : (cats - (total_animals - cats)) = 20 :=
by 
  sorry

end more_cats_than_dogs_l729_729685


namespace triangle_inequality_l729_729110

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end triangle_inequality_l729_729110


namespace oil_truck_radius_l729_729392

theorem oil_truck_radius
  (r_stationary : ℝ) (h_stationary : ℝ) (h_drop : ℝ) 
  (h_truck : ℝ)
  (V_pumped : ℝ) (π : ℝ) (r_truck : ℝ) :
  r_stationary = 100 → h_stationary = 25 → h_drop = 0.064 → h_truck = 10 →
  V_pumped = π * r_stationary^2 * h_drop →
  V_pumped = π * r_truck^2 * h_truck →
  r_truck = 8 := 
by 
  intros r_stationary_eq h_stationary_eq h_drop_eq h_truck_eq V_pumped_eq1 V_pumped_eq2
  sorry

end oil_truck_radius_l729_729392


namespace octagon_diagonals_l729_729150

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l729_729150


namespace downstream_distance_l729_729361

theorem downstream_distance (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ) (distance : ℝ) :
  speed_boat = 20 ∧ speed_current = 5 ∧ time_minutes = 24 ∧ distance = 10 →
  (speed_boat + speed_current) * (time_minutes / 60) = distance :=
by
  sorry

end downstream_distance_l729_729361


namespace rectangle_area_equals_392_over_9_l729_729401

theorem rectangle_area_equals_392_over_9 :
  let triangle_perimeter := 7.5 + 9.3 + 11.2
  let rectangle_perimeter := triangle_perimeter
  ∃ l w : ℝ, 2 * l + 2 * w = rectangle_perimeter ∧ l = 2 * w ∧ (l * w = 392 / 9) :=
by
  let triangle_perimeter := 7.5 + 9.3 + 11.2
  let rectangle_perimeter := triangle_perimeter
  have : 2 * (2 * w) + 2 * w = 28 := sorry -- Placeholder for detailed steps.
  have : w = 14 / 3 := sorry -- Placeholder for detailed steps.
  have : l = 2 * (14 / 3) := sorry -- Placeholder for detailed steps.
  have : (l * w = 392 / 9) := sorry -- Conclusion drawn from previous steps.
  existsi (14 / 3), (28 / 3), this

end rectangle_area_equals_392_over_9_l729_729401


namespace func_properties_l729_729373

-- Define the conditions
variables {f : ℝ → ℝ}

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- f is decreasing on (0, ∞)
def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f y ≤ f x

-- f has a minimum value of 2
def has_minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ y, f y = m

-- Prove the desired statement
theorem func_properties
  (h1 : is_even f)
  (h2 : is_decreasing f {x | 0 < x})
  (h3 : has_minimum_value f 2) :
  (∀ x y ∈ {x | x < 0}, x < y → f x ≤ f y) ∧ (∀ x, x < 0 → f x ≥ 2) :=
sorry

end func_properties_l729_729373


namespace Aiyanna_has_more_cookies_l729_729733

theorem Aiyanna_has_more_cookies : 
  let Alyssa_cookies := 129 in
  let Aiyanna_cookies := 140 in
  Aiyanna_cookies - Alyssa_cookies = 11 :=
by
  sorry

end Aiyanna_has_more_cookies_l729_729733


namespace evaluate_expression_l729_729855

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729855


namespace probability_three_heads_in_eight_tosses_l729_729009

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729009


namespace least_positive_angle_l729_729439

noncomputable def cos15 : ℝ := Real.cos (15 * Real.pi / 180)
noncomputable def sin35 : ℝ := Real.sin (35 * Real.pi / 180)

theorem least_positive_angle (θ : ℝ) (h : cos15 = sin35 + Real.sin θ) : θ = 55 * Real.pi / 180 :=
  sorry

end least_positive_angle_l729_729439


namespace conic_section_is_ellipse_l729_729442

def is_conic_section_ellipse (x y : ℝ) : Prop :=
  sqrt (x^2 + (y + 2)^2) + sqrt ((x - 4)^2 + (y - 3)^2) = 8

def distance_between_foci : ℝ :=
  sqrt ((4 - 0)^2 + (3 + 2)^2)

theorem conic_section_is_ellipse :
  distance_between_foci = sqrt 41 →
  sqrt 41 < 8 →
  ∀ (x y : ℝ), is_conic_section_ellipse x y → True :=
by {
  assume _ _ _,
  exact trivial,
}

end conic_section_is_ellipse_l729_729442


namespace sufficiency_and_non_necessity_l729_729098

variable {x y : ℝ}

def condition_p : Prop := x > 1 ∧ y > 1
def condition_q : Prop := x + y > 2 ∧ x * y > 1

theorem sufficiency_and_non_necessity : condition_p → condition_q ∧ ¬(condition_q → condition_p) := by
  sorry

end sufficiency_and_non_necessity_l729_729098


namespace eval_expression_l729_729830

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729830


namespace gas_cost_per_gallon_l729_729436

-- Definitions of the given conditions
def fuel_efficiency := 32 : ℝ
def distance := 464 : ℝ
def total_cost := 58 : ℝ

-- The property we need to prove
theorem gas_cost_per_gallon : (total_cost / (distance / fuel_efficiency)) = 4 :=
by
  sorry

end gas_cost_per_gallon_l729_729436


namespace stairs_left_to_climb_l729_729027

def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

theorem stairs_left_to_climb : total_stairs - climbed_stairs = 22 := by
  sorry

end stairs_left_to_climb_l729_729027


namespace divides_expression_for_any_integer_l729_729263

theorem divides_expression_for_any_integer (n: ℤ): (n - 1) ∣ (n ^ (n^n + n^3 + 3^n) + 4 * n - n^3 + n^2 + 6) :=
  sorry

end divides_expression_for_any_integer_l729_729263


namespace num_four_digit_even_numbers_l729_729657

theorem num_four_digit_even_numbers : 
  let cards := [1, 2, 3, 3, 4, 5],
      n := array.length cards,
      condition_1 := ∀ (l: List ℕ), length l = 4 → 
                    (∀ d, d ∈ l → d ∈ cards) → 
                    even (l.getLastOrElse 0)
  in   (∃ (l: List ℕ), length l = 4 ∧ 
         (∀ d, d ∈ l → d ∈ cards) ∧ 
         even (l.getLastOrElse 0)) → 
       (|{ l : List ℕ | length l = 4 ∧ 
          (∀ d, d ∈ l → d ∈ cards) ∧ 
          even (l.getLastOrElse 0)}| = 66) :=
sorry

end num_four_digit_even_numbers_l729_729657


namespace evaluate_expression_l729_729878

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729878


namespace evaluate_expression_l729_729784

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729784


namespace evaluate_expression_l729_729791

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729791


namespace field_dimension_m_l729_729711

theorem field_dimension_m (m : ℝ) (h : (3 * m + 8) * (m - 3) = 80) : m = 6.057 := by
  sorry

end field_dimension_m_l729_729711


namespace correct_statements_l729_729683

open MeasureTheory ProbabilityTheory

noncomputable def Binomial : ProbabilityMassFunction ℕ := sorry
noncomputable def Normal : ProbabilityMassFunction ℝ := sorry

variables
  (X : ℕ → ℝ) -- Random variable X for the binomial distribution
  (Y : ℝ → ℝ) -- Random variable Y for the normal distribution

-- Defining the conditions for binomial distribution B(4, 1/3)
axiom binomial_dist : X follows Binomial

-- Defining the conditions for normal distribution N(3, σ^2)
axiom normal_dist : Y follows Normal

-- Probability value P(X ≤ 5) = 0.85 for normal distribution case
axiom normal_prob_5 : P((λ x, x ≤ 5) Y) = 0.85

-- Variance of a random variable X
variable D : ℝ → ℝ

-- Defining the condition about variance
axiom variance_property : D(Y) = D(X)

-- Defining the mutually exclusive condition
axiom mutually_exclusive (A B : Event) : A ∩ B = ∅

-- Defining the theorem
theorem correct_statements :
  B_correct : (∃ σ^2, (P((1 <ᵣ Y ∧ Y ≤ 3)) = 0.35)) ∧ 
  D_correct : mutually_exclusive → complementary events :=
by
  sorry

end correct_statements_l729_729683


namespace largest_sum_l729_729052

theorem largest_sum :
  max (
    {1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/8} : finset ℚ
  ) = 5/6 :=
by
  sorry

end largest_sum_l729_729052


namespace octagon_diagonals_l729_729160

theorem octagon_diagonals : 
  let n := 8 in
  (n * (n - 3)) / 2 = 20 := 
by 
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2 : by rfl
                    ... = 40 / 2           : by norm_num
                    ... = 20               : by norm_num

end octagon_diagonals_l729_729160


namespace pentagon_area_l729_729419

-- Given definitions from the problem condition
def is_inscribed (polygon : Type*) (circle : Type*) : Prop := sorry
def radius (circle : Type*) : ℝ := 4
def is_regular_pentagon (polygon : Type*) : Prop := sorry

-- Statement of the problem
theorem pentagon_area (P : Type*) (C : Type*) 
  (h_inscribed : is_inscribed P C) 
  (h_radius : radius C = 4) 
  (h_regular : is_regular_pentagon P) :
  ∃ a, a = 40 * real.sin (72 * real.pi / 180) := 
sorry

end pentagon_area_l729_729419


namespace evaluate_expression_l729_729862

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729862


namespace num_ordered_triples_l729_729104

def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

theorem num_ordered_triples :
  ∃ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 ∧
  ∃ (count : ℕ), count = 70 :=
by
  sorry

end num_ordered_triples_l729_729104


namespace arithmetic_sequence_sum_l729_729494

theorem arithmetic_sequence_sum 
  (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n : ℕ, a n = 2 + (n - 5)) 
  (ha5 : a 5 = 2) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9) := 
by 
  sorry

end arithmetic_sequence_sum_l729_729494


namespace system_solution_l729_729650

theorem system_solution (x y: ℝ) 
  (h1: x + y = 2) 
  (h2: 3 * x + y = 4) : 
  x = 1 ∧ y = 1 :=
sorry

end system_solution_l729_729650


namespace red_apples_count_l729_729592

-- Definitions based on conditions
def green_apples : ℕ := 2
def yellow_apples : ℕ := 14
def total_apples : ℕ := 19

-- Definition of red apples as a theorem to be proven
theorem red_apples_count :
  green_apples + yellow_apples + red_apples = total_apples → red_apples = 3 :=
by
  -- You would need to prove this using Lean
  sorry

end red_apples_count_l729_729592


namespace prime_divisors_of_29_pow_p_plus_1_l729_729337

open Nat

theorem prime_divisors_of_29_pow_p_plus_1 :
  ∀ p : ℕ, Prime p → p ∣ 29^p + 1 → p = 2 ∨ p = 3 ∨ p = 5 :=
by
  sorry

end prime_divisors_of_29_pow_p_plus_1_l729_729337


namespace positive_expression_l729_729241

theorem positive_expression (x y : ℝ) : (x^2 - 4 * x + y^2 + 13) > 0 := by
  sorry

end positive_expression_l729_729241


namespace eval_expression_l729_729956

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729956


namespace gcd_lcm_product_l729_729071

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l729_729071


namespace n_squared_plus_n_plus_1_is_odd_l729_729252

theorem n_squared_plus_n_plus_1_is_odd (n : ℤ) : Odd (n^2 + n + 1) :=
sorry

end n_squared_plus_n_plus_1_is_odd_l729_729252


namespace find_m_l729_729491

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
by
  sorry

end find_m_l729_729491


namespace triangle_inequality_valid_x_values_l729_729295

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l729_729295


namespace simplify_expression_1_simplify_expression_2_l729_729614

-- Problem (1)
theorem simplify_expression_1 (α : ℝ) : 
  (sin (π + α) * sin (2 * π - α) * cos (-π - α)) / (sin (3 * π + α) * cos (π - α) * cos (3 * π / 2 + α)) = -1 := 
  sorry

-- Problem (2)
theorem simplify_expression_2 : 
  cos 20 + cos 160 + sin 1866 - sin 606 = 0 := 
  sorry

end simplify_expression_1_simplify_expression_2_l729_729614


namespace Suzanna_total_distance_l729_729170

-- Conditions
def rate := 1.5 -- miles per 10 minutes
def time := 40 -- minutes

-- Question: Prove Suzanna rides 6 miles in 40 minutes at the given rate.
theorem Suzanna_total_distance :
  (time / 10) * rate = 6 := by
  sorry

end Suzanna_total_distance_l729_729170


namespace eval_expression_l729_729904

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729904


namespace prove_unattainable_y_l729_729431

noncomputable def unattainable_y : Prop :=
  ∀ (x y : ℝ), x ≠ -4 / 3 → y = (2 - x) / (3 * x + 4) → y ≠ -1 / 3

theorem prove_unattainable_y : unattainable_y :=
by
  intro x y h1 h2
  sorry

end prove_unattainable_y_l729_729431


namespace sum_to_infinity_l729_729986

noncomputable def harmonic (n : ℕ) : ℝ :=
  if n = 0 then 0 else (finset.range n).sum (λ k, 1 / (k + 1 : ℝ))

theorem sum_to_infinity (a : ℝ) (p: ℕ) (hp: 0 < p) (ha: a ≠ 0):
  (∑' n from p to ⊤, (n : ℝ) / (a * (n + 1) * harmonic n * harmonic (n + 1))) = 1 / (a * harmonic p) :=
by
  sorry

end sum_to_infinity_l729_729986


namespace h_h_of_2_l729_729223

def h (x : ℝ) : ℝ := 4 * x^2 - 8

theorem h_h_of_2 : h (h 2) = 248 := by
  -- Proof goes here
  sorry

end h_h_of_2_l729_729223


namespace probability_getting_wet_l729_729708

theorem probability_getting_wet 
  (P_R : ℝ := 1/2)
  (P_notT : ℝ := 1/2)
  (h1 : 0 ≤ P_R ∧ P_R ≤ 1)
  (h2 : 0 ≤ P_notT ∧ P_notT ≤ 1) 
  : P_R * P_notT = 1/4 := 
by
  -- Proof that the probability of getting wet equals 1/4
  sorry

end probability_getting_wet_l729_729708


namespace quadratic_root_exists_l729_729417

theorem quadratic_root_exists (a b c : ℝ) (ha : a ≠ 0)
  (h1 : a * (0.6 : ℝ)^2 + b * 0.6 + c = -0.04)
  (h2 : a * (0.7 : ℝ)^2 + b * 0.7 + c = 0.19) :
  ∃ x : ℝ, 0.6 < x ∧ x < 0.7 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_exists_l729_729417


namespace crystal_run_distance_l729_729435

def distance (x1 y1 x2 y2 : ℝ) := (real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2))

-- Define the coordinates of each point on the path
def A := (0, 0 : ℝ × ℝ)
def B := (0, 2)
def C := (real.sqrt 2, 2 + real.sqrt 2)
def D := (real.sqrt 2 + (3 * real.sqrt 2) / 2, 2 + real.sqrt 2 - (3 * real.sqrt 2) / 2)

-- Final position should be equal to the starting point
theorem crystal_run_distance :
  distance D.1 D.2 A.1 A.2 = 4.5 :=
sorry

end crystal_run_distance_l729_729435


namespace points_colorable_l729_729092

noncomputable def colorable_points (points : Finset (ℤ × ℤ)) : Prop :=
  ∃ color : (ℤ × ℤ) → ℕ, 
    (∀ (L : ℤ) (pts : Finset (ℤ × ℤ)), 
      (pts ⊆ points ∧
        (∀ p ∈ pts, p.1 = L ∨ p.2 = L)) → 
      abs (pts.count (λ p, color p = 0) - pts.count (λ p, color p = 1)) ≤ 1)

theorem points_colorable (points : Finset (ℤ × ℤ)) : colorable_points points :=
sorry

end points_colorable_l729_729092


namespace area_triangle_AOB_constant_eqn_circle_c_min_val_PB_plus_PQ_l729_729090

noncomputable def circle_c (t : ℝ) (ht : t > 0) := 
  (λ (x y : ℝ), (x - t)^2 + (y - 2 / t)^2 = t^2 + (2 / t)^2)

theorem area_triangle_AOB_constant (t : ℝ) (ht : t > 0) : 
  let A := (2 * t, 0),
      B := (0, 4 / t) in
  1 / 2 * |2 * t| * |4 / t| = 4 :=
by
  sorry

theorem eqn_circle_c (t : ℝ) (ht : t = 2) : 
  circle_c 2 (by linarith) = (λ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 5) :=
by
  sorry

theorem min_val_PB_plus_PQ :
  ∃ P Q : ℝ × ℝ, line_l P ∧ circle_c P Q ∧ 
  minimum_value |PB| + |PQ| = 2 * sqrt 5 ∧ P = (-4 / 3, -2 / 3) :=
by
  sorry

end area_triangle_AOB_constant_eqn_circle_c_min_val_PB_plus_PQ_l729_729090


namespace leo_homework_assignments_l729_729408

theorem leo_homework_assignments :
  let
    first_four := 4 * 1,
    next_seven := 7 * 3,
    grades_12_to_15 := 4 * 5,
    grades_16_to_19 := 4 * 7,
    grades_20_to_23 := 4 * 9,
    grades_24_to_27 := 4 * 11
  in
    first_four + next_seven + grades_12_to_15 + grades_16_to_19 + grades_20_to_23 + grades_24_to_27 = 153 :=
by
  sorry

end leo_homework_assignments_l729_729408


namespace diana_principal_charge_l729_729443

theorem diana_principal_charge :
  ∃ P : ℝ, P > 0 ∧ (P + P * 0.06 = 63.6) ∧ P = 60 :=
by
  use 60
  sorry

end diana_principal_charge_l729_729443


namespace restaurant_total_cost_l729_729751

theorem restaurant_total_cost :
  let adult_meal_cost := 12
  let kids_meal_cost := 0
  let adult_drink_cost := 2.50
  let kids_drink_cost := 1.50
  let dessert_cost := 4
  let adult_count := 7
  let kids_count := 4
  let exclusive_dish_charge := 3 * 3
  let total_people := 11
  let discount_rate := 0.10
  let sales_tax_rate := 0.075
  let service_charge_rate := 0.15
  let adult_cost := adult_count * (adult_meal_cost + adult_drink_cost + dessert_cost)
  let kids_cost := kids_count * (kids_meal_cost + kids_drink_cost + dessert_cost)
  let subtotal := adult_cost + kids_cost + exclusive_dish_charge
  let discount := subtotal * discount_rate
  let discounted_subtotal := subtotal - discount
  let sales_tax := discounted_subtotal * sales_tax_rate
  let total_with_tax := discounted_subtotal + sales_tax
  let service_charge := total_with_tax * service_charge_rate
  let total_cost := total_with_tax + service_charge
  in total_cost = 178.57 := 
by
  sorry

end restaurant_total_cost_l729_729751


namespace inequality_product_lt_zero_l729_729496

theorem inequality_product_lt_zero (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 :=
  sorry

end inequality_product_lt_zero_l729_729496


namespace evaluate_expression_l729_729867

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729867


namespace card_product_even_l729_729612

theorem card_product_even (a b : Fin 7 → ℕ) (h1 : ∀ i, (a i) ∈ {1, 2, 3, 4, 5, 6, 7})
  (h2 : ∀ j, (b j) ∈ {1, 2, 3, 4, 5, 6, 7}) :
  ∃ i, (a i + b i) % 2 = 0 :=
begin
  have h, from Set.toFinset_subset_univ (h1 : a.to_set ⊆ {1, 2, 3, 4, 5, 6, 7}) ▸ h2,
  sorry
end

end card_product_even_l729_729612


namespace Amy_age_2005_l729_729038

def Amy_age_in_2005 (Amy_age_in_2000 Grandfather_age_in_2000 : ℕ) : ℕ :=
  Amy_age_in_2000 + 5

axiom conditions (y : ℕ) :
  y * 4 = 4000 - 3900 ∧ Amy_age_in_2000 = y ∧ Grandfather_age_in_2000 = 3 * y

theorem Amy_age_2005 (y : ℕ) (h : conditions y) : Amy_age_in_2005 y (3 * y) = 30 := 
by sorry

end Amy_age_2005_l729_729038


namespace probability_scoring_80_or_above_probability_failing_exam_l729_729189

theorem probability_scoring_80_or_above (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C) = 0.69 :=
by
  intros hB hC hD hE
  sorry

theorem probability_failing_exam (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C ∪ D ∪ E) = 0.93 →
  1 - P (B ∪ C ∪ D ∪ E) = 0.07 :=
by
  intros hB hC hD hE hBCDE
  sorry

end probability_scoring_80_or_above_probability_failing_exam_l729_729189


namespace jessica_number_of_pies_l729_729566

theorem jessica_number_of_pies :
  (∀ (each_serving_requires : ℕ → ℝ) (number_of_guests : ℕ) (servings_per_pie : ℕ) (average_apples_per_guest : ℝ),
    (each_serving_requires 1 = 1.5) →
    (number_of_guests = 12) →
    (servings_per_pie = 8) →
    (average_apples_per_guest = 3) →
    (number_of_guests * (average_apples_per_guest / each_serving_requires 1) / servings_per_pie) = 3) :=
by
  intros each_serving_requires number_of_guests servings_per_pie average_apples_per_guest h1 h2 h3 h4 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jessica_number_of_pies_l729_729566


namespace faucet_draining_time_l729_729717

theorem faucet_draining_time 
  (all_faucets_drain_time : ℝ)
  (n : ℝ) 
  (first_faucet_time : ℝ) 
  (last_faucet_time : ℝ) 
  (avg_drain_time : ℝ)
  (condition_1 : all_faucets_drain_time = 24)
  (condition_2 : last_faucet_time = first_faucet_time / 7)
  (condition_3 : avg_drain_time = (first_faucet_time + last_faucet_time) / 2)
  (condition_4 : avg_drain_time = 24) : 
  first_faucet_time = 42 := 
by
  sorry

end faucet_draining_time_l729_729717


namespace centroid_triangle_set_l729_729674

theorem centroid_triangle_set (ABC : Type) [triangle ABC] :
  ∃ (R : set (point ABC)), ∀ (P : point R),
    ∃ (triangle : ABC → Type), 
      (vertex1 triangle) ∈ side AB ∧ 
      (vertex2 triangle) ∈ side BC ∧ 
      (vertex3 triangle) ∈ side AC ∧ 
      (centroid triangle) = P :=
sorry

end centroid_triangle_set_l729_729674


namespace subtract_correctly_l729_729354

theorem subtract_correctly (x : ℕ) (h : x + 35 = 77) : x - 35 = 7 :=
sorry

end subtract_correctly_l729_729354


namespace transformation_correct_l729_729507

theorem transformation_correct :
  ∀ (x : ℝ), (cos (2 * (x / 2) - π / 12)) = cos (2 * x + π / 6) :=
begin
  assume (x : ℝ),
  calc
    cos (2 * (x / 2) - π / 12)
        = cos (x - π / 12) : by rw mul_div_cancel' _ two_ne_zero
    ... = cos (x - π / 12 + 2 * π / 6 - 2 * π / 6) : by ring
    ... = cos ((x + π / 6 - π / 12) + 2 * π / 6) : by ring
    ... = cos (2 * x + π / 6) : sorry
end

end transformation_correct_l729_729507


namespace evaluate_expression_l729_729843

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729843


namespace average_multiples_of_10_from_10_to_100_l729_729362

theorem average_multiples_of_10_from_10_to_100 :
  let multiples := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] in
  (List.sum multiples : ℝ) / (List.length multiples) = 55 := 
by
  sorry

end average_multiples_of_10_from_10_to_100_l729_729362


namespace volunteer_selection_count_l729_729261

open Nat

theorem volunteer_selection_count :
  let boys : ℕ := 5
  let girls : ℕ := 2
  let total_ways := choose girls 1 * choose boys 2 + choose girls 2 * choose boys 1
  total_ways = 25 :=
by
  sorry

end volunteer_selection_count_l729_729261


namespace part_a_constant_part_b_inequality_l729_729244

open Real

noncomputable def cubic_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem part_a_constant (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1 * x2 / x3^2) + cubic_root (x2 * x3 / x1^2) + cubic_root (x3 * x1 / x2^2)) = 
  const_value := sorry

theorem part_b_inequality (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1^2 / (x2 * x3)) + cubic_root (x2^2 / (x3 * x1)) + cubic_root (x3^2 / (x1 * x2))) < (-15 / 4) := sorry

end part_a_constant_part_b_inequality_l729_729244


namespace pentagonal_number_sixth_l729_729413

theorem pentagonal_number_sixth :
  let P : ℕ → ℕ := 
  λ n, match n with
    | 1 => 1
    | 2 => 5
    | 3 => 12
    | 4 => 22
    | 5 => 35
    | 6 => 51
    | _ => 0 -- We only need the first six values
  in
  P 6 = 51 :=
by
  -- Conditions given
  let P : ℕ → ℕ :=
    λ n, match n with
      | 1 => 1
      | 2 => 5
      | 3 => 12
      | 4 => 22
      | 5 => 35
      | 6 => 51
      | _ => 0 -- We only need the first six values
  -- We need to prove P(6) = 51
  sorry

end pentagonal_number_sixth_l729_729413


namespace problem_conditions_implies_a_ge_5_l729_729499

theorem problem_conditions_implies_a_ge_5
    (a : ℝ)
    (h1 : 0 < a) (h2 : a ≠ 1)
    (h3 : ∀ x1 x2 : ℝ, x1 < x2 → a^x1 < a^x2)
    (h4 : ¬ (∃ x y : ℝ, 3 * x + 4 * y + a = 0 ∧ x^2 + y^2 = 1)) :
    5 ≤ a :=
begin
  sorry
end

end problem_conditions_implies_a_ge_5_l729_729499


namespace imaginary_part_one_l729_729529

-- Given condition
def condition (z : ℂ) : Prop := (1 + z) * (1 - complex.I) = 2

-- The imaginary part of a complex number
def imag_part (z : ℂ) : ℝ := z.im

-- Prove that if the condition holds, the imaginary part of z is 1
theorem imaginary_part_one (z : ℂ) (h : condition z) : imag_part z = 1 :=
by
  sorry

end imaginary_part_one_l729_729529


namespace evaluate_expression_l729_729844

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729844


namespace eval_expression_l729_729829

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729829


namespace eval_expression_l729_729944

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729944


namespace ratio_is_three_to_one_l729_729034

-- Definitions
def total_vegetables : ℕ := 280
def num_cucumbers : ℕ := 70

-- Calculate the number of tomatoes
def num_tomatoes : ℕ := total_vegetables - num_cucumbers

-- State the ratio
def ratio_tomatoes_to_cucumbers : ℕ × ℕ :=
  (num_tomatoes / num_cucumbers, num_cucumbers / num_cucumbers)

-- Proof that the ratio is 3:1
theorem ratio_is_three_to_one :
  num_tomatoes = 210 ∧ ratio_tomatoes_to_cucumbers = (3, 1) :=
by
  have num_tomatoes_calc : num_tomatoes = 280 - 70 := rfl
  have ratio_calc : ratio_tomatoes_to_cucumbers = (210 / 70, 70 / 70) := rfl
  simp [num_tomatoes_calc, ratio_calc]
  sorry

end ratio_is_three_to_one_l729_729034


namespace decimal_to_binary_24_l729_729774

theorem decimal_to_binary_24 :
  ∀ x : ℕ, x = 24 → (to_binary x) = "11000" :=
by
  intro x
  intro hx
  rw hx
  sorry

end decimal_to_binary_24_l729_729774


namespace diagonals_of_octagon_l729_729156

theorem diagonals_of_octagon : 
  let n := 8 in 
  let total_line_segments := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_line_segments - sides in
  diagonals = 20 := 
  by 
    let n := 8
    let total_line_segments := (n * (n - 1)) / 2
    let sides := n
    let diagonals := total_line_segments - sides
    have h : diagonals = 20 := sorry
    exact h

end diagonals_of_octagon_l729_729156


namespace arithmetic_seq_proof_l729_729993

theorem arithmetic_seq_proof
  (x : ℕ → ℝ)
  (h : ∀ n ≥ 3, x (n-1) = (x n + x (n-1) + x (n-2)) / 3):
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry

end arithmetic_seq_proof_l729_729993


namespace evaluate_expression_l729_729926

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729926


namespace distance_traveled_when_meeting_l729_729692

noncomputable def trainDistance (a b t_A t_B d : ℝ) : ℝ :=
  (d / t_A) * (d / (d / t_A + d / t_B))

theorem distance_traveled_when_meeting
  (d : ℝ) (t_A : ℝ) (t_B : ℝ)
  (d_pos : d > 0)
  (t_A_pos : t_A > 0)
  (t_B_pos : t_B > 0) :
  trainDistance 0 0 t_A t_B d = 50 :=
by
  let speed_A := d / t_A
  let speed_B := d / t_B
  let t := d / (speed_A + speed_B)
  have h1 : speed_A = 125 / 12 := rfl
  have h2 : speed_B = 125 / 8 := rfl
  exact sorry

end distance_traveled_when_meeting_l729_729692


namespace max_area_triangle_correct_l729_729186

def max_area_triangle (a b : ℝ) (h1 : a + b = 4) (C : ℝ) (h2 : C = 30) : Prop :=
  let area := (1 / 2) * a * b * Real.sin (C * Real.pi / 180)
  area ≤ 1

theorem max_area_triangle_correct (a b : ℝ) (h1 : a + b = 4) (C : ℝ) (h2 : C = 30) :
  max_area_triangle a b h1 C h2 := 
begin
  -- Proof goes here
  sorry
end

end max_area_triangle_correct_l729_729186


namespace sum_of_interior_angles_of_octagon_l729_729653

theorem sum_of_interior_angles_of_octagon : 
  let n := 8 in
  (n - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_of_octagon_l729_729653


namespace disproving_implication_l729_729992

theorem disproving_implication :
  ∃ a b c : ℝ, 
   (a^2 + b^2) / (b^2 + c^2) = a / c ∧ a / b ≠ b / c :=
by
  use [1, 2, 1]
  simp
  sorry

end disproving_implication_l729_729992


namespace evaluate_expression_l729_729836

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729836


namespace digit_in_hundredths_place_of_7_div_20_is_5_l729_729672

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ :=
  num / den

def hundredths_place_digit (num : ℕ) (den : ℕ) : ℕ :=
  let dec := fraction_to_decimal num den in
  let dec_str := dec.repr in
  let dec_str_parts := dec_str.splitOn "." in
  if dec_str_parts.length > 1 then
    let decimal_part := dec_str_parts[1] in
    if decimal_part.length >= 2 then
      decimal_part.get 1 - '0'
    else
      0
  else
    0
  
theorem digit_in_hundredths_place_of_7_div_20_is_5 :
  hundredths_place_digit 7 20 = 5 := 
by sorry

end digit_in_hundredths_place_of_7_div_20_is_5_l729_729672


namespace proof_expr_is_neg_four_ninths_l729_729757

noncomputable def example_expr : ℚ := (-3 / 2) ^ 2021 * (2 / 3) ^ 2023

theorem proof_expr_is_neg_four_ninths : example_expr = (-4 / 9) := 
by 
  -- Here the proof would be placed
  sorry

end proof_expr_is_neg_four_ninths_l729_729757


namespace evaluate_expression_l729_729786

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729786


namespace category_dishes_count_l729_729705

theorem category_dishes_count (x y : ℕ) (hx : x + y = 10) (hc : 5 + 3.5 * x + 2.5 * y = 36) : x = 6 ∧ y = 4 :=
by
  sorry

end category_dishes_count_l729_729705


namespace eval_expression_l729_729964

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729964


namespace find_length_BC_l729_729749

section
variable {O M N A B C D : Type}

-- Assuming given distances
variable (AB_len AD_len AN_len : ℝ)
variable (AB_is_diameter : A ⟶ B)
variable (midpoint_CD : M = midpoint C D)
variable (MN_perp_AB : perpendicular M N AB)
variable (AB_eq : AB_len = 10)
variable (AD_eq : AD_len = 3)
variable (AN_eq : AN_len = 3)

-- Proving that the length of BC is 7.
theorem find_length_BC 
  (AB_len = 10) 
  (AD_len = 3) 
  (AN_len = 3) 
  (AB_is_diameter : A ⟶ B) 
  (midpoint_CD : M = midpoint C D) 
  (MN_perp_AB : perpendicular M N AB) : 
  let BC : ℝ := sorry in
  BC = 7 :=
begin
  -- The theorem statement, proof is not necessary.
  sorry
end

end find_length_BC_l729_729749


namespace frank_reads_pages_per_day_l729_729474

theorem frank_reads_pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) : pages_per_book / days_per_book = 83 :=
by {
  sorry
}

end frank_reads_pages_per_day_l729_729474


namespace problem_statement_l729_729526

theorem problem_statement (a b : ℝ) (h : a < b) : a - b < 0 :=
sorry

end problem_statement_l729_729526


namespace eval_expression_l729_729909

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729909


namespace f_256_l729_729501

noncomputable def f (x : ℕ) : ℕ := sorry

axiom f_recurrence (n : ℕ) (h : 2 ≤ n) : f(n^2) = f(n) + 2
axiom f_initial : f(2) = 1

theorem f_256 : f(256) = 7 :=
by
  sorry

end f_256_l729_729501


namespace eval_expression_l729_729826

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729826


namespace evaluate_expression_l729_729868

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729868


namespace new_monthly_savings_l729_729712

-- Definitions based on conditions
def monthly_salary := 4166.67
def initial_savings_percent := 0.20
def expense_increase_percent := 0.10

-- Calculations
def initial_savings := initial_savings_percent * monthly_salary
def initial_expenses := (1 - initial_savings_percent) * monthly_salary
def increased_expenses := initial_expenses + expense_increase_percent * initial_expenses
def new_savings := monthly_salary - increased_expenses

-- Lean statement to prove the question equals the answer given conditions
theorem new_monthly_savings :
  new_savings = 499.6704 := 
by
  sorry

end new_monthly_savings_l729_729712


namespace eval_expression_l729_729941

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729941


namespace determine_side_length_of_equilateral_triangle_l729_729275

noncomputable def triangle_side_length (A B C P : Point) (s : ℝ) : Prop :=
  equilateral_triangle A B C ∧ 
  inside_triangle A B C P ∧ 
  dist A P = 2 ∧ 
  dist B P = 2 * Real.sqrt 3 ∧ 
  dist C P = 4

theorem determine_side_length_of_equilateral_triangle 
  (A B C P : Point) (s : ℝ) 
  (h : triangle_side_length A B C P s) :
  s = Real.sqrt 14 :=
sorry

end determine_side_length_of_equilateral_triangle_l729_729275


namespace harvest_season_duration_l729_729234

theorem harvest_season_duration (weekly_rent : ℕ) (total_rent_paid : ℕ) : 
    (weekly_rent = 388) →
    (total_rent_paid = 527292) →
    (total_rent_paid / weekly_rent = 1360) :=
by
  intros h1 h2
  sorry

end harvest_season_duration_l729_729234


namespace eyes_that_saw_airplane_l729_729319

theorem eyes_that_saw_airplane (students : ℕ) (looked_up_fraction : ℚ) (eyes_per_student : ℕ) :
  students = 200 → looked_up_fraction = 3/4 → eyes_per_student = 2 → looked_up_fraction * students * eyes_per_student = 300 :=
by
  intros hstudents hlooked_up_fraction heyes_per_student
  rw [hstudents, hlooked_up_fraction, heyes_per_student]
  norm_num
  sorry

end eyes_that_saw_airplane_l729_729319


namespace heptagon_diagonals_l729_729389

theorem heptagon_diagonals (n : ℕ) (h : n = 7) : (n * (n - 3)) / 2 = 14 := by
  sorry

end heptagon_diagonals_l729_729389


namespace evaluate_expression_l729_729932

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729932


namespace eval_expression_l729_729969

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729969


namespace johns_age_l729_729379

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l729_729379


namespace eval_expression_l729_729824

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729824


namespace inequality_proof_l729_729527

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : (n / m) + (m / n) > 2 :=
sorry

end inequality_proof_l729_729527


namespace find_multiplier_l729_729023

theorem find_multiplier (x y: ℤ) (h1: x = 127)
  (h2: x * y - 152 = 102): y = 2 :=
by
  sorry

end find_multiplier_l729_729023


namespace margaret_spends_on_croissants_l729_729991

theorem margaret_spends_on_croissants :
  (∀ (people : ℕ) (sandwiches_per_person : ℕ) (croissants_per_sandwich : ℕ) (croissants_per_set : ℕ) (cost_per_set : ℝ),
    people = 24 →
    sandwiches_per_person = 2 →
    croissants_per_sandwich = 1 →
    croissants_per_set = 12 →
    cost_per_set = 8 →
    (people * sandwiches_per_person * croissants_per_sandwich) / croissants_per_set * cost_per_set = 32) := sorry

end margaret_spends_on_croissants_l729_729991


namespace sum_n_k_eq_eight_l729_729636

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l729_729636


namespace geometric_mean_proof_l729_729613

theorem geometric_mean_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let G := sqrt (a * b),
      A := (a + b) / 2,
      H := (2 * a * b) / (a + b) in
  G^2 = H * A ∧ G = sqrt (a * b) :=
by
  sorry

end geometric_mean_proof_l729_729613


namespace sequence_inequality_l729_729311

theorem sequence_inequality (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)
  (S : ℕ → ℝ) (T : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n + 2 * a (n + 1) = 0) →
  (∀ n : ℕ, b n = 2 * n - 1) →
  (∀ n : ℕ, S n = ∑ i in Finset.range (n + 1), a (i + 1)) →
  (∀ n : ℕ, c n = 1 / (Real.sqrt (b n) * Real.sqrt (b (n + 1)) * (Real.sqrt (b n) + Real.sqrt (b (n + 1))))) →
  (∀ n : ℕ, T n = ∑ i in Finset.range (n + 1), c (i + 1)) →
  ∀ (m k : ℕ), m > 0 → k > 0 → S m > T k :=
begin
  sorry
end

end sequence_inequality_l729_729311


namespace lisa_goal_impossible_l729_729236

theorem lisa_goal_impossible (total_quizzes : ℕ) (completed_quizzes earned_As : ℕ) (goal_percentage : ℕ) (remaining_quizzes needed_As : ℕ)
  (H1 : total_quizzes = 60)
  (H2 : completed_quizzes = 40)
  (H3 : earned_As = 30)
  (H4 : goal_percentage = 90)
  (H5 : remaining_quizzes = total_quizzes - completed_quizzes)
  (H6 : needed_As = (goal_percentage * total_quizzes / 100) - earned_As) :
  needed_As > remaining_quizzes := 
begin
  sorry
end

end lisa_goal_impossible_l729_729236


namespace smallest_prime_with_composite_odd_sum_of_digits_l729_729608

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits.sum

def is_composite (n : ℕ) : Prop :=
2 ≤ n ∧ ∃ m k, 2 ≤ m ∧ 2 ≤ k ∧ m * k = n

def is_composite_odd (n : ℕ) : Prop :=
n % 2 = 1 ∧ is_composite n

theorem smallest_prime_with_composite_odd_sum_of_digits :
  ∃ p : ℕ, Prime p ∧ is_composite_odd (sum_of_digits p) ∧ p = 997 :=
sorry

end smallest_prime_with_composite_odd_sum_of_digits_l729_729608


namespace eval_expression_l729_729831

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729831


namespace real_coeffs_with_even_expression_are_integers_l729_729750

theorem real_coeffs_with_even_expression_are_integers
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h : ∀ x y : ℤ, (∃ k1 : ℤ, a1 * x + b1 * y + c1 = 2 * k1) ∨ (∃ k2 : ℤ, a2 * x + b2 * y + c2 = 2 * k2)) :
  (∃ (i1 j1 k1 : ℤ), a1 = i1 ∧ b1 = j1 ∧ c1 = k1) ∨
  (∃ (i2 j2 k2 : ℤ), a2 = i2 ∧ b2 = j2 ∧ c2 = k2) := by
  sorry

end real_coeffs_with_even_expression_are_integers_l729_729750


namespace total_number_of_cards_l729_729565

theorem total_number_of_cards (groups : ℕ) (cards_per_group : ℕ) (h_groups : groups = 9) (h_cards_per_group : cards_per_group = 8) : groups * cards_per_group = 72 := by
  sorry

end total_number_of_cards_l729_729565


namespace evaluation_of_expression_l729_729897

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729897


namespace log_216_eq_3_log_6_l729_729696

theorem log_216_eq_3_log_6 :
  let n := 216 in
  n = 6^3 →
  Real.log n = 3 * Real.log 6 := by
  intro n_eq
  sorry

end log_216_eq_3_log_6_l729_729696


namespace solve_trig_eq_l729_729981

theorem solve_trig_eq (x : ℝ) (h₁ : x ∈ Ioo (-π/2) 0) (h₂ : (sqrt 3) / (Real.sin x) + 1 / (Real.cos x) = 4) : 
  x = -4 * π / 9 :=
by
  -- Proof is omitted
  sorry

end solve_trig_eq_l729_729981


namespace original_couch_price_l729_729212

def chair_price : ℝ := sorry
def table_price := 3 * chair_price
def couch_price := 5 * table_price
def bookshelf_price := 0.5 * couch_price

def discounted_chair_price := 0.8 * chair_price
def discounted_couch_price := 0.9 * couch_price
def total_price_before_tax := discounted_chair_price + table_price + discounted_couch_price + bookshelf_price
def total_price_after_tax := total_price_before_tax * 1.08

theorem original_couch_price (budget : ℝ) (h_budget : budget = 900) : 
  total_price_after_tax = budget → couch_price = 503.85 :=
by
  sorry

end original_couch_price_l729_729212


namespace inequality_in_triangle_l729_729555

noncomputable def median_length (a b c A α : ℝ) : Prop :=
  2 * a ≥ (b + c) * (Real.cos (α / 2))

-- Define the conditions
variables {A B C M : Point}
variables {a b c α : ℝ} -- lengths of sides and angle α considered here

-- Ensuring that AM is a median
def is_median (A M B C : Point) : Prop :=
  sorry

-- Theorem statement
theorem inequality_in_triangle (h1 : is_median A M B C) :
  median_length (distance A M) b c A α :=
sorry

end inequality_in_triangle_l729_729555


namespace evaluation_of_expression_l729_729898

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729898


namespace transformed_roots_eqn_l729_729583

theorem transformed_roots_eqn {a b c r s : ℝ} (h1 : r + s = -b / a) (h2 : r * s = c / a) :
  ∃ g : ℝ → ℝ, g = λ y, y^2 - b * y + a * c :=
sorry

end transformed_roots_eqn_l729_729583


namespace range_of_m_l729_729102

variable (m : ℝ)

def prop_p : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + m*x1 + 1 = 0) ∧ (x2^2 + m*x2 + 1 = 0)

def prop_q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h₁ : prop_p m) (h₂ : ¬prop_q m) : m < -2 ∨ m ≥ 3 :=
sorry

end range_of_m_l729_729102


namespace johns_initial_weekly_salary_l729_729567

theorem johns_initial_weekly_salary : ∃ x : ℝ, (1.1076923076923077 * x = 72 ∧ x ≈ 64.99) :=
by {
  use (72 / 1.1076923076923077),
  split,
  { norm_num, },
  { apply real.norm_num_approx,
    norm_num,
  }
}

end johns_initial_weekly_salary_l729_729567


namespace area_of_triangle_ABC_l729_729187

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

noncomputable def trisection_point (A C : Point) : Point :=
  (2 * A + C) / 3

variable {ABC : Triangle}
variable (M : Point) (N : Point) (K : Point)
variable hM : M = midpoint ABC.A ABC.B
variable hN : N = trisection_point ABC.A ABC.C
variable hK : collinear ABC.C M K ∧ collinear ABC.B N K
variable hAreaBCK : area ABC.B ABC.C K = 1

theorem area_of_triangle_ABC : area ABC.A ABC.B ABC.C = 4 :=
by
  sorry

end area_of_triangle_ABC_l729_729187


namespace evaluation_of_expression_l729_729901

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729901


namespace circle_equation_l729_729115

-- Define conditions
def on_parabola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 = 4 * y

def tangent_to_y_axis (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x, _) := M
  abs x = r

def tangent_to_axis_of_symmetry (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (_, y) := M
  abs (1 + y) = r

-- Main theorem statement
theorem circle_equation (M : ℝ × ℝ) (r : ℝ) (x y : ℝ)
  (h1 : on_parabola M)
  (h2 : tangent_to_y_axis M r)
  (h3 : tangent_to_axis_of_symmetry M r) :
  (x - M.1)^2 + (y - M.2)^2 = r^2 ↔
  x^2 + y^2 + 4 * M.1 * x - 2 * M.2 * y + 1 = 0 := 
sorry

end circle_equation_l729_729115


namespace collinear_points_l729_729505

variables {R : Type} [OrderedRing R]

variables {a b : R}
variables (λ μ k : R)
variables (A B C : R)

noncomputable def collinear (A B C R: Type) [OrderedRing R] : Prop :=
∃ k : R, A=k*B ∧ B = k*C

theorem collinear_points (h₁ : collinear A B C R)
                         (h₂ : a ≠ 0) 
                         (h₃ : b ≠ 0) 
                         (h₄ : B = λ * a + b) 
                         (h₅ : C = a + μ * b) :
                         λ*μ = 1 :=
begin
  sorry
end

end collinear_points_l729_729505


namespace distance_midpoint_example_l729_729063

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem distance_midpoint_example :
  let A := (-3, -1)
  let B := (9, 4)
  distance A B = 13 ∧ midpoint A B = (3, 1.5) :=
by {
  let A := (-3, -1)
  let B := (9, 4)
  sorry
}

end distance_midpoint_example_l729_729063


namespace cookies_per_bag_l729_729353

theorem cookies_per_bag (total_bags : ℕ) (total_cookies : ℕ) (num_cookies_per_bag : ℕ) 
  (h1 : total_bags = 26) (h2 : total_cookies = 52) (h3 : total_cookies = total_bags * num_cookies_per_bag) : 
  num_cookies_per_bag = 2 :=
begin
  sorry
end

end cookies_per_bag_l729_729353


namespace AM_GEOM_Square_Min_Value_l729_729374

-- Definition for question 1
theorem AM_GEOM_Square (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ 2 * real.sqrt (a * b) := 
by 
  sorry

-- Definition for question 2
theorem Min_Value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 := 
by 
  sorry

end AM_GEOM_Square_Min_Value_l729_729374


namespace theta_value_l729_729999

variable (a b θ : ℝ)
variable (h₁ : f(x) = a * Real.cos (x + 2 * θ) + b * x + 3)
variable (h₂ : f 1 = 5)
variable (h₃ : f (-1) = 1)

noncomputable def correct_theta : Prop :=
  θ = Real.pi / 4

theorem theta_value (a b : ℝ) (h₁ : ∀ x, f x = a * Real.cos (x + 2 * θ) + b * x + 3)
  (h₂ : f 1 = 5) (h₃ : f (-1) = 1) : correct_theta θ := sorry

end theta_value_l729_729999


namespace border_area_l729_729720

theorem border_area (h_photo : ℕ) (w_photo : ℕ) (border : ℕ) (h : h_photo = 8) (w : w_photo = 10) (b : border = 2) :
  (2 * (border + h_photo) * (border + w_photo) - h_photo * w_photo) = 88 :=
by
  rw [h, w, b]
  sorry

end border_area_l729_729720


namespace simplify_expression_l729_729264

theorem simplify_expression :
  (sqrt 800 / sqrt 50) - (sqrt 288 / sqrt 72) = 2 :=
by
  sorry

end simplify_expression_l729_729264


namespace triangle_properties_l729_729020

noncomputable def right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def triangle_circumradius (a b c : ℕ) [right_triangle a b c] : ℝ :=
  c / 2

noncomputable def triangle_inradius (a b c : ℕ) [right_triangle a b c] : ℝ :=
  let s := (a + b + c) / 2
  let K := (a * b) / 2
  K / s

noncomputable def distance_circumcenter_incenter (a b c : ℕ) [right_triangle a b c] : ℝ :=
  let R := c / 2
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  real.sqrt (r^2 + (R - (s - b))^2)

theorem triangle_properties :
  ∀ a b c : ℕ, right_triangle a b c → a = 8 → b = 15 → c = 17 → 
  triangle_circumradius a b c = 8.5 ∧ 
  distance_circumcenter_incenter a b c = real.sqrt 85 / 2 := 
by
  intros a b c ht ha hb hc
  sorry

end triangle_properties_l729_729020


namespace count_large_prime_factors_l729_729057

def has_large_prime_factor (n : ℕ) : Prop := 
  ∃ p : ℕ, p.prime ∧ p ∣ (n^2 + 1) ∧ p > n

theorem count_large_prime_factors :
  (finset.card (finset.filter has_large_prime_factor (finset.range (10^6 + 1)))) = 757575 :=
by
  sorry

end count_large_prime_factors_l729_729057


namespace sum_n_k_eq_eight_l729_729635

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l729_729635


namespace find_index_arithmetic_seq_l729_729112

def arithmetic_seq (a_1 d: ℤ) (n: ℕ) : ℤ := a_1 + (n - 1) * d

theorem find_index_arithmetic_seq :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (∀ n, a n = arithmetic_seq 1 (-2) n) →
    a n = -15 → n = 9 :=
by {
  intros a n h_seq h_an,
  sorry
}

end find_index_arithmetic_seq_l729_729112


namespace largest_number_of_permutations_l729_729578

theorem largest_number_of_permutations (Q : set (perm (fin 100)))
  (h : ∀ (π₁ π₂ ∈ Q) (a b : fin 100), (π₁(a) < π₁(b) ∧ π₁(a + 1) = π₁(b)) → (π₂(a) > π₂(b))) :
  Q.to_finset.card ≤ 100 :=
sorry

end largest_number_of_permutations_l729_729578


namespace evaluation_of_expression_l729_729887

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729887


namespace polynomial_inequality_conditions_l729_729061

theorem polynomial_inequality_conditions (P : ℝ → ℝ) (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → P(x + y) ≥ P(x) + P(y)) ∧
  (P = λ x, a * x ^ 3 + b * x ^ 2 + c * x + d) →
  a > 0 ∧ d ≤ 0 ∧ 8 * b ^ 3 ≥ 243 * a ^ 2 * d :=
by sorry

end polynomial_inequality_conditions_l729_729061


namespace anna_phone_chargers_l729_729747

theorem anna_phone_chargers (p l : ℕ) (h₁ : l = 5 * p) (h₂ : l + p = 24) : p = 4 :=
by
  sorry

end anna_phone_chargers_l729_729747


namespace semicircle_circumference_given_perimeters_l729_729691

noncomputable def semicircle_circumference (π : ℝ) (d : ℝ) : ℝ := (π * (d / 2)) / 2 + d

theorem semicircle_circumference_given_perimeters :
  let rectangle_length := 24
  let rectangle_breadth := 16
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_diameter := square_side
  let semicircle_circumference_approx := Real.floor ((semicircle_circumference 3.14 semicircle_diameter) * 100) / 100
  in semicircle_circumference_approx = 51.40 := by
  sorry

end semicircle_circumference_given_perimeters_l729_729691


namespace find_coordinates_perpendicular_tangent_l729_729116

theorem find_coordinates_perpendicular_tangent :
  ∃ (m n : ℝ), (f m = n) ∧ ((f' m = -(1/4)) → (m = 1 ∧ n = 0) ∨ (m = -1 ∧ n = -4)) :=
by
  let f : ℝ → ℝ := λ x, x^3 + x - 2
  let f' : ℝ → ℝ := λ x, 3*x^2 + 1
  existsi (1 : ℝ, 0 : ℝ)
  existsi (-1 : ℝ, -4 : ℝ)
  sorry

end find_coordinates_perpendicular_tangent_l729_729116


namespace evaluate_expression_l729_729794

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729794


namespace main_theorem_l729_729226

-- Define the necessary components: NonEmpty, 2-basis definition, and the problem statement
def is_2_basis (A : Set ℕ) : Prop :=
  ∀ (n : ℕ), ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ n = a + b

noncomputable def exists_2_basis_constant : Prop :=
  ∃ (A : Set ℕ) (C : ℝ), NonEmpty A ∧ is_2_basis A ∧ (∀ (x : ℝ), x ≥ 1 → (size (A ∩ Set.Iic ⌊x⌋) : ℝ) ≤ C * Real.sqrt x)

theorem main_theorem : exists_2_basis_constant :=
sorry

end main_theorem_l729_729226


namespace tan_fraction_cos_condition_l729_729995

theorem tan_fraction_cos_condition (x : ℝ) (hx : x > π ∧ x < 3 * π / 2)
  (h : cos (π / 4 + x) = -3 / 5) :
  (1 + tan x) / (1 - tan x) = 4 / 3 :=
by
  sorry

end tan_fraction_cos_condition_l729_729995


namespace count_leftmost_8_in_R_l729_729227

noncomputable def num_elements_with_leftmost_8 : ℕ :=
let R := {k : ℕ | k ≤ 2000} in
let log_10_8 := Real.log10 8 in
if (Real.log10 (8^2000)).floor + 1 = 1909 ∧ Nat.digits 10 (8^2000) ≠ [] ∧ Nat.digits 10 (8^2000).head = 8 then
  91
else
  0

theorem count_leftmost_8_in_R :
  let R := {8^k | k : ℕ, k ≤ 2000} in
  (∀ k ∈ R, (Nat.digits 10 (8^k)).head = 8) ↔ num_elements_with_leftmost_8 = 91 :=
by sorry

end count_leftmost_8_in_R_l729_729227


namespace shape_D_not_possible_l729_729021

-- Definitions to describe the conditions
inductive Shape
| A : Shape
| B : Shape
| C : Shape
| D : Shape
| E : Shape

-- Defining the condition of shapes obtained by folding a square twice
def can_result_from_two_folds : Shape → Prop
| Shape.A := true
| Shape.B := true
| Shape.C := true
| Shape.D := false
| Shape.E := true

-- The theorem we need to prove:
theorem shape_D_not_possible : ¬ can_result_from_two_folds Shape.D :=
by {
  sorry
}

end shape_D_not_possible_l729_729021


namespace magic_square_x_value_l729_729188

theorem magic_square_x_value (a b c d e S x : ℕ) (h1 : S = x + 91)
  (h2 : c = x + 84) (h3 : d + e = 7) (h4 : a + 154 = x + 91) : x = 133 :=
by
  -- Add necessary assumptions derived from conditions without the labelling from the solution steps
  have h5 : x + 21 + 70 = S, from h1,
  have h6 : x + 84 + a + 70 = x + 91, from h4,
  have h7 : x + 84 + d + e = x + 91, from h3,
  sorry

end magic_square_x_value_l729_729188


namespace trajectory_is_line_segment_l729_729516

-- Definitions for the fixed points and the distance conditions
def F1 : Point := sorry  -- Fixed point F1
def F2 : Point := sorry  -- Fixed point F2
def M : Point := sorry   -- Moving point M
def d (p1 p2 : Point) : ℝ := sorry  -- Distance function

-- Given conditions
axiom dist_F1_F2 : d F1 F2 = 8
axiom dist_M_F1_F2 : ∀ M, d M F1 + d M F2 = 8

-- The statement to prove
theorem trajectory_is_line_segment : 
  ∀ M, d M F1 + d M F2 = 8 ↔ ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ M = a • F1 + (1 - a) • F2 :=
sorry

end trajectory_is_line_segment_l729_729516


namespace elmer_more_than_penelope_l729_729246

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l729_729246


namespace points_parallel_within_plane_l729_729178

/-- If the projections of several points in space onto the same plane lie on a straight line, 
then the positions of these points in space are parallel within a plane. -/
theorem points_parallel_within_plane {P : Type*} [EuclideanSpace P] (points : list P) (plane : Plane P) :
  (∀ p ∈ points, ∃ q ∈ (project_onto_plane p plane), q ∈ line (project_onto_plane (points.head) plane, project_onto_plane (points.nth 1) plane)) →
  (∃ parallel_plane : Plane P, ∀ p ∈ points, p ∈ parallel_plane ∧ ∃ direction : Vector, ∀ p1 p2 ∈ points, p1 -ᵥ p2 = direction) := 
sorry

end points_parallel_within_plane_l729_729178


namespace quotient_of_x6_plus_5x4_plus_3_by_x_minus_2_l729_729982

noncomputable def polynomial_division (p q : Polynomial ℝ) :=
  Polynomial.divModByMonic p q

theorem quotient_of_x6_plus_5x4_plus_3_by_x_minus_2 :
  polynomial_division (Polynomial.X ^ 6 + 5 * Polynomial.X ^ 4 + 3) (Polynomial.X - 2) = 
  ((Polynomial.X ^ 5 + 2 * Polynomial.X ^ 4 + 9 * Polynomial.X ^ 3 + 18 * Polynomial.X ^ 2 + 36 * Polynomial.X + 72), 147) := 
by
  sorry

end quotient_of_x6_plus_5x4_plus_3_by_x_minus_2_l729_729982


namespace evaluate_expression_l729_729799

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729799


namespace function_properties_l729_729734

theorem function_properties (x : ℝ) : 
  (∀ x ∈ Ioo 0 (π / 2), 0 ≤ (x : ℝ) → is_increasing (λ x, abs (sin x))) ∧ (∀ x, abs (sin (-x)) = abs (sin x)) ∧ (∀ x, abs (sin (x + π)) = abs (sin x)) := 
by
  sorry

end function_properties_l729_729734


namespace evaluate_expression_l729_729796

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729796


namespace quadratic_no_real_roots_l729_729313

theorem quadratic_no_real_roots 
  (a b c : ℝ) 
  (h : a = 1 ∧ b = -4 * real.sqrt 2 ∧ c = 9)
  (discriminant : b^2 - 4*a*c = -4) :
  ¬ ∃ x : ℝ, a*x^2 + b*x + c = 0 :=
by 
  sorry

end quadratic_no_real_roots_l729_729313


namespace yield_calc_l729_729382

def stock := Real

def par_value : stock := 100
def market_value : stock := 125
def dividend (par_value : stock) : stock := par_value * 0.10

def yield_percentage (dividend : stock) (market_value : stock) : Real :=
  (dividend / market_value) * 100

theorem yield_calc : yield_percentage (dividend par_value) market_value = 8 := by
  sorry

end yield_calc_l729_729382


namespace parallel_lines_l729_729549

-- Definitions and conditions
variables {A B C D E F M : Type} [ordered_ring A] [add_comm_group B] [add_comm_group C] [add_comm_group D] 
  [add_comm_group E] [add_comm_group F] [add_comm_group M] [module A B] [module A C] [module A D] 
  [module A E] [module A F] [module A M] 

-- Quadrilateral ABCD with AB = AD
variable (AB AD : A)
variable (h1 : AB = AD)

-- BC perpendicular to AB
variable (BC : A)
variable (h2 : BC ⊥ AB)

-- E is the intersection of the angle bisector of ∠DCB with AB
variable (E : A)
variable (h3 : E ∈ intersection_of_angle_bisector D C B AB)

-- Line perpendicular to CD passing through A intersects DE at F
variable (CD : A)
variable (h4 : CD ⊥ A ∧ A ∈ intersection_line DE F)

-- M is the midpoint of BD
variable (BD : A)
variable (h5 : midpoint M B D)

-- Prove FM parallel to EC
theorem parallel_lines (h1 : AB = AD) (h2 : BC ⊥ AB) (h3 : E ∈ intersection_of_angle_bisector D C B AB)
  (h4 : CD ⊥ A ∧ A ∈ intersection_line DE F) (h5 : midpoint M B D) 
  : FM ∥ EC :=
sorry

end parallel_lines_l729_729549


namespace eval_expression_l729_729937

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729937


namespace max_sum_of_squares_proof_l729_729430

noncomputable
def max_sum_of_squares (a : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  a^2 / (2 * (sin (α / 2))^2)

theorem max_sum_of_squares_proof (a : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∀ (b c : ℝ), acute (triangle.mk a b c α) → b^2 + c^2 ≤ max_sum_of_squares a α hα := sorry

end max_sum_of_squares_proof_l729_729430


namespace unique_solution_nat_numbers_l729_729060

theorem unique_solution_nat_numbers (a b c : ℕ) (h : 2^a + 9^b = 2 * 5^c + 5) : 
  (a, b, c) = (1, 0, 0) :=
sorry

end unique_solution_nat_numbers_l729_729060


namespace problem_solution_l729_729620

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)
variables (h3 : 3 * log 101 ((1030301 - a - b) / (3 * a * b)) = 3 - 2 * log 101 (a * b))

theorem problem_solution : 101 - (a)^(1/3) - (b)^(1/3) = 0 :=
by
  sorry

end problem_solution_l729_729620


namespace initial_bottles_calculation_l729_729237

theorem initial_bottles_calculation (maria_bottles : ℝ) (sister_bottles : ℝ) (left_bottles : ℝ) 
  (H₁ : maria_bottles = 14.0) (H₂ : sister_bottles = 8.0) (H₃ : left_bottles = 23.0) :
  maria_bottles + sister_bottles + left_bottles = 45.0 :=
by
  sorry

end initial_bottles_calculation_l729_729237


namespace area_of_triangle_ABC_l729_729560

noncomputable def area_of_triangle {a b c : ℝ} (A B C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

theorem area_of_triangle_ABC :
  ∀ (a b c : ℝ), b = 6 → a = 2 * c → B = Real.pi / 3 → area_of_triangle a c B = 6 * Real.sqrt 3 :=
by
  intros a b c hb ha hB
  sorry

end area_of_triangle_ABC_l729_729560


namespace sam_has_8_marbles_l729_729260

theorem sam_has_8_marbles :
  ∀ (steve sam sally : ℕ),
  sam = 2 * steve →
  sally = sam - 5 →
  steve + 3 = 10 →
  sam - 6 = 8 :=
by
  intros steve sam sally
  intros h1 h2 h3
  sorry

end sam_has_8_marbles_l729_729260


namespace problem_solution_l729_729031

variable {n : ℕ}
variable (q a₁ a₂ a₃ a₆ : ℝ)
variable (b : ℕ → ℝ)

-- Conditions
-- 1. The sequence {a_n} is exponential and consists of positive terms.
-- 2. 2a₁ + 3a₂ = 1
axiom cond1 : 2 * a₁ + 3 * a₂ = 1

-- 3. a₃² = 9a₂a₆ 
axiom cond2 : a₃^2 = 9 * a₂ * a₆

-- Exponential sequence definition
def exponential_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^(n-1)

-- General formula for the sequence {a_n}
def a_n := exponential_sequence a₁ (1 / 3) n

-- Logarithm summation sequence definition
def b_n := (∑ i in finset.range n, real.log (a_n i)) / real.log 3

-- Sum of first n terms of {1 / b_n}
def sum_inverse_b (n : ℕ) : ℝ := ∑ k in finset.range n, 1 / b k

-- Theorem to prove
theorem problem_solution : 
  (a₁ = 1 / 3) ∧ (q = 1 / 3) ∧ (∀ n, a_n = 1 / 3^n) ∧ 
  (sum_inverse_b n = -2 * (n / (n + 1 : ℝ))) :=
by
  -- Proof of the theorem goes here.
  sorry

end problem_solution_l729_729031


namespace graph_transformation_correct_l729_729665

theorem graph_transformation_correct :
  (∀ x : ℝ, 3 * sin (2 * x + π / 5) = 3 * sin (2 * (x + π / 10))) :=
sorry

end graph_transformation_correct_l729_729665


namespace find_angle_E_l729_729415

open Real

-- Define the coordinates of points A, B, C, and D
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 3)
def C : ℝ × ℝ := (0, 3)
def D : ℝ × ℝ := (3, 0)

-- Define the slope of a line given two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the slope of line AB
def slope_AB : ℝ := slope A B

-- Define the slope of line CD
def slope_CD : ℝ := slope C D

-- Define the angle between two lines given their slopes
def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  atan (abs ((m1 - m2) / (1 + m1 * m2)))

-- Define the expected angle at E
def angle_E_expected : ℝ := 45

-- The main theorem
theorem find_angle_E :
  let m1 := slope_AB in
  let m2 := slope_CD in
  let angle_E := angle_between_lines m1 m2 in
  angle_E = angle_E_expected :=
sorry

end find_angle_E_l729_729415


namespace probability_three_heads_in_eight_tosses_l729_729004

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729004


namespace tan_identity_proof_l729_729111

-- Conditions
variable (α : ℝ)
variable (h1 : ∀ (a b c : ℝ), a + b + c = π → a = α → b < c → b < π/3)

-- Given conditions
axiom h_largest_angle (h2 : ∃ (a b c : ℝ), a = α ∧ h1 a b c ∧ α > π/3 ∧ α < π) : 
  ∃ (a b c : ℝ), a = α ∧ h1 a b c ∧ ∀ t, cos (2 * t) = 1/2 → t * 2 < 2 * α

axiom h_cos_2alpha (h3 : ℝ) (h4 : cos (2 * α) = 1/2) : α = h3

-- Prove
theorem tan_identity_proof (h2 : ∃ (a b c : ℝ), a = α ∧ h1 a b c ∧ α > π/3 ∧ α < π) 
  (h3 : ∀ x : ℝ, cos (2 * x) = 1/2 → ∀ t, x = α ∧ t = 2 * x → t < 2 * α) : 
  (1 - tan α) / (1 + tan α) = 2 + √3 := by sorry

end tan_identity_proof_l729_729111


namespace evaluate_expression_l729_729880

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729880


namespace wang_heng_birth_l729_729670

theorem wang_heng_birth : ∃ (x y : ℕ), 1900 < x ∧ x < 2000 ∧ 1 ≤ y ∧ y ≤ 12 ∧ 100 * y + x = 2088 ∧ x = 1998 ∧ y = 1 :=
by
  exists 1998
  exists 1
  simp
  -- Simplifying the conditions
  have : 1900 < 1998 := by norm_num
  have : 1998 < 2000 := by norm_num
  have : 1 ≤ 1 := le_refl 1
  have : 1 ≤ 12 := by norm_num
  have : 100 * 1 + 1998 = 2088 := by norm_num
  tauto

end wang_heng_birth_l729_729670


namespace evaluate_expression_l729_729873

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729873


namespace eval_expression_l729_729957

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729957


namespace max_area_l729_729618

noncomputable theory

namespace Playground

def perimeter : ℕ := 150

def side1 (x : ℕ) : ℕ := x
def side2 (x : ℕ) : ℕ := perimeter / 2 - x

def area (x : ℕ) : ℕ := side1 x * side2 x

theorem max_area : ∃ x : ℕ, area x = 1406 ∧ (0 < x ∧ x < perimeter / 2) := 
by 
  sorry

end Playground

end max_area_l729_729618


namespace number_of_books_before_purchase_l729_729305

theorem number_of_books_before_purchase (x : ℕ) (h1 : x + 140 = (27 / 25) * x) : x = 1750 :=
by
  sorry

end number_of_books_before_purchase_l729_729305


namespace claire_flour_cost_l729_729763

def num_cakes : ℕ := 2
def flour_per_cake : ℕ := 2
def cost_per_flour : ℕ := 3
def total_cost (num_cakes flour_per_cake cost_per_flour : ℕ) : ℕ := 
  num_cakes * flour_per_cake * cost_per_flour

theorem claire_flour_cost : total_cost num_cakes flour_per_cake cost_per_flour = 12 := by
  sorry

end claire_flour_cost_l729_729763


namespace line_intersects_parabola_exactly_one_point_l729_729077

theorem line_intersects_parabola_exactly_one_point (k : ℝ) :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 10 = k) ∧
  (∀ y z : ℝ, -3 * y^2 - 4 * y + 10 = k ∧ -3 * z^2 - 4 * z + 10 = k → y = z) 
  → k = 34 / 3 :=
by
  sorry

end line_intersects_parabola_exactly_one_point_l729_729077


namespace irrational_of_series_l729_729619

/-- 
  Suppose that \( f(x) = \sum_{i=0}^\infty c_i x^i \) is a power series for which each coefficient \( c_i \) is 0 or 1. 
  Show that if \( f(2/3) = 3/2 \), then \( f(1/2) \) must be irrational.
-/
theorem irrational_of_series (c : ℕ → ℕ) (h_c : ∀ i, c i = 0 ∨ c i = 1)
  (h_sum_23 : ∑' i, (c i : ℚ) * (2 / 3) ^ i = 3 / 2) :
  ¬ rational (∑' i, (c i : ℚ) * (1 / 2) ^ i) :=
by
  sorry

end irrational_of_series_l729_729619


namespace evaluate_expression_l729_729876

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  calc
    (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4)
    = 4 * (3^2 + 1^2) * (3^4 + 1^4) : by rw [add_eq (add_nat (add_eq (add_nat 1) (add_nat 3)))]
    = 4 * 10 * (3^4 + 1^4) : by rw [pow2_add_pow2, pow2_add_pow2 (pow_nat 3 2) (pow_nat 1 1)]
    = 4 * 10 * 82 : by rw [pow4_add_pow4, pow4_add_pow4 (pow_nat 3 4) (pow_nat 1 1)]
    = 3280 : by norm_num

end evaluate_expression_l729_729876


namespace math_problem_l729_729228

theorem math_problem (k : ℕ) (p : ℕ) (h1 : k > 14) (h2 : p < k) (h3 : p ≥ 3 * k / 4) 
(h4 : ∀ q, q < k → (Prime q → q ≤ p)) :
  ¬ (2 * p ∣ (factorial (2 * p - k))) ∧ (∀ n, n > 2 * p ∧ ¬ (Prime n) → n ∣ (factorial (n - k))) :=
by
  sorry

end math_problem_l729_729228


namespace determine_M_l729_729048

-- Definitions used directly from the conditions
def first_hyperbola_asymptotes_slope : ℝ := 4/3
def second_hyperbola_asymptotes_slope (M : ℝ) : ℝ := 5 / Real.sqrt M

-- Statement of the problem in Lean
theorem determine_M (M_correct : M = 225 / 16) :
  second_hyperbola_asymptotes_slope M = first_hyperbola_asymptotes_slope := by
  sorry

end determine_M_l729_729048


namespace eval_expression_l729_729913

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729913


namespace solve_for_a_l729_729166

theorem solve_for_a (a : ℝ) (h : 4 * a + 9 + (3 * a + 5) = 0) : a = -2 :=
by
  sorry

end solve_for_a_l729_729166


namespace avg_of_14_23_y_is_21_l729_729622

theorem avg_of_14_23_y_is_21 (y : ℝ) (h : (14 + 23 + y) / 3 = 21) : y = 26 :=
by
  sorry

end avg_of_14_23_y_is_21_l729_729622


namespace measure_1kg_of_sugar_l729_729661

-- Define the conditions
def times_equal_ratio (left right : ℕ) : Prop :=
  3 * right = 4 * left

theorem measure_1kg_of_sugar (left_weight right_weight kg_iron : ℕ) [weightless_bags : Prop] 
    (balance_ratio : times_equal_ratio left_weight right_weight) : 
    ∃ (sugar : ℕ), sugar = 1 :=
by
  -- Details omitted
  sorry 

end measure_1kg_of_sugar_l729_729661


namespace range_of_m_l729_729179

theorem range_of_m (x m : ℝ) :
  (∀ x, (x - 1) / 2 ≥ (x - 2) / 3 → 2 * x - m ≥ x → x ≥ m) ↔ m ≥ -1 := by
  sorry

end range_of_m_l729_729179


namespace carrie_bought_t_shirts_l729_729424

theorem carrie_bought_t_shirts (total_spent : ℝ) (cost_each : ℝ) (n : ℕ) 
    (h_total : total_spent = 199) (h_cost : cost_each = 9.95) 
    (h_eq : n = total_spent / cost_each) : n = 20 := 
by
sorry

end carrie_bought_t_shirts_l729_729424


namespace eccentricity_of_ellipse_l729_729097

variables (x y a b c : ℝ)

def ellipse (a b : ℝ) (x y : ℝ) :=
  (x^2 / a^2 + y^2 / b^2 = 1)

def foci_distance(a : ℝ) := 2 * real.sqrt(a^2 - (a^2 / 4))

noncomputable def ellipse_eccentricity (a : ℝ) (c : ℝ) := c / a

theorem eccentricity_of_ellipse
  (h_ellipse : ∀ x y, ellipse a b x y → a > b ∧ b > 0)
  (h_point_on_ellipse : ∃ x y, ellipse a b x y)
  (h_perimeter : ∀ F1 F2 P, F1 • F2 = 2 * a ∧ P • F1 + P • F2 = 18)
  (h_major_axis : 2 * a = 5) :
  ellipse_eccentricity a (foci_distance a) = 4 / 5 :=
sorry

end eccentricity_of_ellipse_l729_729097


namespace evaluate_expression_l729_729788

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729788


namespace polynomial_arithmetic_roots_l729_729975

theorem polynomial_arithmetic_roots (a : ℝ) :
  (∃ r d : ℂ, r ∈ ℝ ∧ d ≠ 0 ∧
  (r - d) * r * (r + d) = -a ∧ 
  (r - d) + r + (r + d) = 9 ∧
  3 * (3 - d) + 3 * (3 + d) + (3 - d) * (3 + d) = 35) →
  a = -9 :=
sorry

end polynomial_arithmetic_roots_l729_729975


namespace evaluate_expression_l729_729923

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729923


namespace sum_n_k_l729_729632

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l729_729632


namespace find_x_minus_y_l729_729138

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + 3 * y = 14) (h2 : x + 4 * y = 11) : x - y = 3 := by
  sorry

end find_x_minus_y_l729_729138


namespace evaluate_expression_l729_729846

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729846


namespace gcd_lcm_product_24_36_proof_l729_729073

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l729_729073


namespace eval_expression_l729_729946

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729946


namespace triangle_side_count_l729_729292

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l729_729292


namespace vec_expression_l729_729519

def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 5)

theorem vec_expression : 2 • vec_a + vec_b = (5, 1) := by
  sorry

end vec_expression_l729_729519


namespace alley_width_l729_729190

theorem alley_width (ℓ : ℝ) (m : ℝ) (n : ℝ): ℓ * (1 / 2 + Real.cos (70 * Real.pi / 180)) = ℓ * (Real.cos (60 * Real.pi / 180)) + ℓ * (Real.cos (70 * Real.pi / 180)) := by
  sorry

end alley_width_l729_729190


namespace eval_expression_l729_729939

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729939


namespace solve_diophantine_eq1_l729_729615

theorem solve_diophantine_eq1 (x y : ℤ) : x^2 - x * y - y^2 = 1 → ∃ n : ℤ, (x, y) = (Fibonacci (2 * n + 1), Fibonacci (2 * n)) ∨ (x, y) = -(Fibonacci (2 * n + 1), Fibonacci (2 * n)) :=
sorry

end solve_diophantine_eq1_l729_729615


namespace intersection_curve_length_l729_729091

-- Define the cube with edge length 1
structure Cube where
  edge_length : ℝ

-- Define the sphere with a given center and radius
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Example cube and sphere as described in the problem
def cube : Cube := { edge_length := 1 }
def sphere : Sphere := { center := (0, 0, 0), radius := 2 * Real.sqrt 3 / 3 }

-- Problem statement in Lean
theorem intersection_curve_length (c : Cube) (s : Sphere) (h₁ : c.edge_length = 1) (h₂ : s.center = (0, 0, 0)) (h₃ : s.radius = 2 * Real.sqrt 3 / 3) : 
  curve_length (intersection s c) = 5 * Real.sqrt 3 * Real.pi / 6 :=
sorry

end intersection_curve_length_l729_729091


namespace range_of_x0_l729_729120

section
variables {a b c x0 y0 y1 y2 : ℝ}
variables {A : ℝ × ℝ} {B : ℝ × ℝ} {C : ℝ × ℝ}

def parabola (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
def conditions : Prop :=
  A = (-5, y1) ∧ B = (-1, y2) ∧ (y1 > y2 ∧ y2 ≥ y0) ∧ parabola (-5) = y1 ∧ parabola (-1) = y2 

theorem range_of_x0 (h : conditions) : x0 > -3 :=
sorry
end

end range_of_x0_l729_729120


namespace sum_digits_single_digit_l729_729471

theorem sum_digits_single_digit (n : ℕ) (h : n = 2^100) : (n % 9) = 7 := 
sorry

end sum_digits_single_digit_l729_729471


namespace num_undefined_values_l729_729988

theorem num_undefined_values : 
  -- Conditions
  ∀ x : ℝ, 
    let denom := (x^2 + 3 * x - 4) * (x - 4) 
  in -- Question and correct answer
     (denom = 0) ↔ (x = -4 ∨ x = 1 ∨ x = 4) ∧ 
     (finset.card (finset.filter (λ x, denom = 0) (finset.Icc (-10) 10))) = 3 := 
     sorry

end num_undefined_values_l729_729988


namespace rahim_books_from_first_shop_l729_729609

def rahim_books (x : ℕ) : Prop :=
  let total_amount := 600 + 240 in
  let total_books := x + 20 in
  total_amount / total_books = 14 ∧ total_amount = 840

theorem rahim_books_from_first_shop : ∃ x : ℕ, rahim_books x ∧ x = 40 :=
by
  use 40
  unfold rahim_books
  split
  · simp
  · exact sorry

end rahim_books_from_first_shop_l729_729609


namespace eyes_that_saw_airplane_l729_729320

theorem eyes_that_saw_airplane (students : ℕ) (looked_up_fraction : ℚ) (eyes_per_student : ℕ) :
  students = 200 → looked_up_fraction = 3/4 → eyes_per_student = 2 → looked_up_fraction * students * eyes_per_student = 300 :=
by
  intros hstudents hlooked_up_fraction heyes_per_student
  rw [hstudents, hlooked_up_fraction, heyes_per_student]
  norm_num
  sorry

end eyes_that_saw_airplane_l729_729320


namespace average_of_last_20_students_l729_729707

theorem average_of_last_20_students 
  (total_students : ℕ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (total_average : ℕ) (first_group_average : ℕ) (second_group_average : ℕ) 
  (total_students_eq : total_students = 50) 
  (first_group_size_eq : first_group_size = 30)
  (second_group_size_eq : second_group_size = 20)
  (total_average_eq : total_average = 92) 
  (first_group_average_eq : first_group_average = 90) :
  second_group_average = 95 :=
by
  sorry

end average_of_last_20_students_l729_729707


namespace probability_of_sum_of_dice_le_5_l729_729667

-- Defining the sample space as pairs of points on two dice.
def sample_space : set (ℕ × ℕ) := { p | p.1 ∈ {1, 2, 3, 4, 5, 6} ∧ p.2 ∈ {1, 2, 3, 4, 5, 6} }

-- Defining the event where the sum of the dice is less than or equal to 5.
def event_sum_le_5 : set (ℕ × ℕ) := { p | p.1 + p.2 ≤ 5 }

-- Calculating the probability of the event.
def probability_event_sum_le_5 : ℚ :=
  (event_sum_le_5 ∩ sample_space).card.to_rat / sample_space.card.to_rat

theorem probability_of_sum_of_dice_le_5 :
  probability_event_sum_le_5 = 5 / 18 := by
  sorry

end probability_of_sum_of_dice_le_5_l729_729667


namespace evaluate_expression_l729_729838

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729838


namespace john_age_l729_729381

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end john_age_l729_729381


namespace altitude_of_triangle_l729_729473

theorem altitude_of_triangle (x y : ℝ) :
  let A := (-5, 3)
  let B := (3, 7)
  let C := (4, -1)
  (x - 8 * y + 29 = 0) ↔
  ∃ D : ℝ × ℝ, D ∈ line B C ∧ D ∈ line_perp_through A :=
sorry

end altitude_of_triangle_l729_729473


namespace evaluate_expression_l729_729936

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729936


namespace eval_expression_l729_729945

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729945


namespace eval_expression_l729_729966

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729966


namespace earliest_meeting_time_l729_729042

theorem earliest_meeting_time
    (charlie_lap : ℕ := 5)
    (ben_lap : ℕ := 8)
    (laura_lap_effective : ℕ := 11) :
    lcm (lcm charlie_lap ben_lap) laura_lap_effective = 440 := by
  sorry

end earliest_meeting_time_l729_729042


namespace lcm_12_18_30_l729_729340

-- Definitions of the numbers 12, 18, and 30
def num1 := 12
def num2 := 18
def num3 := 30

-- Define the least common multiple function (for the sake of completeness)
def lcm (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

-- State the theorem that needs to be proven
theorem lcm_12_18_30 : lcm num1 (lcm num2 num3) = 180 := by
  sorry

end lcm_12_18_30_l729_729340


namespace smallest_N_last_digit_is_five_l729_729026

def ada_sequence (n : ℕ) : List ℕ :=
  let rec sequence n acc :=
    if n = 0 then acc
    else
      let k := Nat.sqrt n
      sequence (n - k * k) (acc ++ [n])
  sequence n []

noncomputable def sequence_length (n : ℕ) : Nat :=
(ada_sequence n).length

theorem smallest_N_last_digit_is_five :
  ∃ N, sequence_length N = 10 ∧ N % 10 = 5 :=
sorry

end smallest_N_last_digit_is_five_l729_729026


namespace probability_three_even_dice_l729_729754

open_locale big_operators

noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

theorem probability_three_even_dice :
  (3 : ℝ) / (12 : ℝ) = 1 / 2 →
  -- Given that the probability of rolling an even number is 1/2
  let p_even : ℝ := 1 / 2 in
  let p_odd : ℝ := 1 / 2 in
  let comb : ℕ := choose 6 3 in
  -- Calculate the number of ways to choose 3 out of 6 dice
  p_even = 1 / 2 →
  p_odd = 1 / 2 →
  -- Probability that each individual die is either even or odd is 1/2
  (20 * (p_even ^ 3 * p_odd ^ 3) = 5 / 16) :=
by sorry

end probability_three_even_dice_l729_729754


namespace quadratic_function_even_and_point_inequality_solution_set_l729_729512

theorem quadratic_function_even_and_point (a b : ℝ) (h_even : ∀ x : ℝ, 2 * x^2 + a * x + b = 2 * x^2 - a * x + b)
    (h_point : 2 * (1:ℝ) ^ 2 + b = -3) :
    (a = 0 ∧ b = -5) :=
by
  have h_a_zero : a = 0,
  { specialize h_even 1,
    simp at h_even,
    exact eq_zero_of_add_eq_zero h_even.symm, },  
  have h_b_value : b = -5,
  { linarith },
  exact ⟨h_a_zero, h_b_value⟩

theorem inequality_solution_set :
    ∀ x : ℝ, 2 * x^2 - 5 ≥ 3 * x + 4 ↔ x ∈ Set.Iic (-3 / 2) ∪ Set.Ici 3 := sorry

end quadratic_function_even_and_point_inequality_solution_set_l729_729512


namespace eval_expression_l729_729958

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729958


namespace original_price_of_the_bag_l729_729702

theorem original_price_of_the_bag (P : ℝ) :
  P - (P * 0.95 * 0.96) = 44 → P = 500 := 
by
  intro h
  calc
    P = 44 / 0.088 := sorry
  have h1 : 0.95 * 0.96 = 0.912 := sorry
  rw [h1] at h 
  sorry

end original_price_of_the_bag_l729_729702


namespace center_sum_of_coordinates_l729_729645

-- Define the given points
def point1 : ℝ × ℝ := (8, -4)
def point2 : ℝ × ℝ := (-2, 0)

-- Define the midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the statement to prove
theorem center_sum_of_coordinates :
  let center := midpoint point1 point2
  in center.1 + center.2 = 1 :=
by
  sorry

end center_sum_of_coordinates_l729_729645


namespace round_to_nearest_integer_l729_729259

theorem round_to_nearest_integer :
  let x := 7254382.673849 in Int.round x = 7254383 := 
by
  --
  sorry

end round_to_nearest_integer_l729_729259


namespace range_of_f_l729_729509

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + π / 3)

theorem range_of_f : Set.Icc (-3 : ℝ) (3 / 2 : ℝ) = Set.range (λ x, f x) ∩ Set.Icc 0 (π / 3) :=
by
  sorry

end range_of_f_l729_729509


namespace point_symmetric_after_folding_l729_729080

/-- Given a Cartesian coordinate system, point (2, 0) coincides with point (-2, 4) after folding.
    We need to prove that the point (5, 8) coincides with the point (6, 7) after the same folding. -/
theorem point_symmetric_after_folding :
  let A := (2, 0) in
  let A' := (-2, 4) in
  let M := ((A.1 + A'.1) / 2, (A.2 + A'.2) / 2) in
  let B := (5, 8) in
  let B' := (6, 7) in
  (M.2 - M.1 * B.1 + B.2 = 0) ∧ (((B'.2 - M.2) / (B'.1 - M.1)) * (-1) = -1) →
  B' = (6, 7) :=
by
  intros A A' M B B' h
  have k := (A.2 - A'.2) / (A.1 - A'.1) -- slope of line AA'
  have k_perpendicular := -1 / k -- slope of perpendicular bisector
  have line_eq := (M.2 - k_perpendicular * M.1) -- Equation of the line l
  let c := M.1 - M.2
  have symmetric_eq := ((B.2 - (c * B.1 + M.2)) * (-1)) = -1
  have midpoint_symmetric := (B.1 + B'.1) / 2 = (M.2 + c * B.1) / 2 ∧ (B.2 + B'.2) / 2 = (k_perpendicular * B.1 + M.2) / 2
  have slope_symmetric := ((B'.2 - B.2) / (B'.1 - B.1)) = -1
  have final := equiv (B', (6, 7))
  { sorry }

/- The theorem states that the point (5, 8) will coincide with the point (6, 7) after the described folding. -/

end point_symmetric_after_folding_l729_729080


namespace max_angle_BAC_l729_729371

-- Definitions and conditions
variables {A B C D : Type}
variables [has_midpoint A C D]  -- D is the midpoint of AC
variables [triangle A B C]
variables [perp_bisectors A B C D]  -- The bisectors of ∠ACB and ∠ABD are perpendicular.

-- The theorem statement
theorem max_angle_BAC : ∃ α : ℝ, angle A B C = α ∧ α ≤ 45 :=
sorry

end max_angle_BAC_l729_729371


namespace sum_of_extreme_primes_l729_729775

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes_upto_50 : List ℕ :=
  List.filter is_prime (List.range 51)

def smallest_prime_50 : ℕ :=
  List.minimum primes_upto_50

def largest_prime_50 : ℕ :=
  List.maximum primes_upto_50

theorem sum_of_extreme_primes : smallest_prime_50 + largest_prime_50 = 49 :=
  sorry

end sum_of_extreme_primes_l729_729775


namespace sum_possible_n_k_l729_729630

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l729_729630


namespace sum_equality_l729_729489

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def falling_factorial (x n : ℕ) : ℕ :=
  if n = 0 then 1 else x * falling_factorial (x - 1) (n - 1)

def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else factorial n / (factorial k * factorial (n - k))

def sum_expression (k m n : ℕ) : ℕ :=
  ∑ i in range (n + 1),
    (-1 : ℕ) ^ i * (factorial (m + n + i) / (factorial i * factorial (n - i) * factorial (m + i))) / (n + k + i)

def right_hand_side (k m n : ℕ) : ℕ :=
  falling_factorial (m - k) n / factorial (n + k)

theorem sum_equality (k m n : ℕ) (hk : 1 ≤ k) (hkm : k ≤ m) (hmn : m ≤ n) :
  sum_expression k m n = right_hand_side k m n :=
by
  sorry

end sum_equality_l729_729489


namespace part_a_impossible_2011_blue_squares_part_b_possible_2010_blue_squares_l729_729446

def initially_red_table (n : ℕ) : (ℕ × ℕ) → bool :=
  λ _, false

def step (table : (ℕ × ℕ) → bool) (flip : ℕ) (is_row : bool) : (ℕ × ℕ) → bool :=
  if is_row then
    λ (i, j), if i = flip then ¬table (i, j) else table (i, j)
  else
    λ (i, j), if j = flip then ¬table (i, j) else table (i, j)

def count_blue (table : (ℕ × ℕ) → bool) (n : ℕ) : ℕ :=
  finset.card (finset.filter (λ pos, table pos) (finset.range n ×ˢ finset.range n))

def exists_sequence_of_steps (final_blue_count : ℕ) (n : ℕ) : Prop :=
  ∃ steps : list (ℕ × bool),
    count_blue (steps.foldl (λ t st, step t (st.1) (st.2)) (initially_red_table n)) n = final_blue_count

theorem part_a_impossible_2011_blue_squares : 
  ¬ exists_sequence_of_steps 2011 50 :=
by sorry

theorem part_b_possible_2010_blue_squares : 
  exists_sequence_of_steps 2010 50 :=
by sorry

end part_a_impossible_2011_blue_squares_part_b_possible_2010_blue_squares_l729_729446


namespace convert_to_base7_l729_729434

theorem convert_to_base7 : 3589 = 1 * 7^4 + 3 * 7^3 + 3 * 7^2 + 1 * 7^1 + 5 * 7^0 :=
by
  sorry

end convert_to_base7_l729_729434


namespace scalene_triangle_third_median_length_l729_729195

theorem scalene_triangle_third_median_length 
  (a b c m₁ m₂ m₃ : ℝ) 
  (h₀ : a ≠ b) (h₁: a ≠ c) (h₂: b ≠ c)
  (h₃ : m₁ = 4) (h₄ : m₂ = 8)
  (h₅ : 1 / 2 * a * b * sin (c) = 4 * sqrt 30) :
  m₃ = 3 * sqrt 60 / 8 :=
by
  sorry

end scalene_triangle_third_median_length_l729_729195


namespace odd_positive_int_divisible_by_24_l729_729606

theorem odd_positive_int_divisible_by_24 (n : ℕ) (hn : n % 2 = 1 ∧ n > 0) : 24 ∣ (n ^ n - n) :=
sorry

end odd_positive_int_divisible_by_24_l729_729606


namespace probability_three_heads_in_eight_tosses_l729_729003

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729003


namespace evaluation_of_expression_l729_729891

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729891


namespace zero_is_neither_positive_nor_negative_l729_729682

theorem zero_is_neither_positive_nor_negative :
  ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  sorry

end zero_is_neither_positive_nor_negative_l729_729682


namespace polynomial_solution_l729_729574

def Q (x : ℝ) : ℝ := Q(0) + Q(1) * x + Q(2) * x^2 + Q(3) * x^3

theorem polynomial_solution :
  (Q(-2) = 2) →
  (∀ x, Q(x) = 4 - x + x^2) :=
by
  -- Assuming Q(x) = Q(0) + Q(1) * x + Q(2) * x^2 + Q(3) * x^3 and Q(-2) = 2, we need to prove Q(x) = 4 - x + x^2
  assume h1,
  sorry

end polynomial_solution_l729_729574


namespace race_order_l729_729736

inductive Position where
| First | Second | Third | Fourth | Fifth
deriving DecidableEq, Repr

structure Statements where
  amy1 : Position → Prop
  amy2 : Position → Prop
  bruce1 : Position → Prop
  bruce2 : Position → Prop
  chris1 : Position → Prop
  chris2 : Position → Prop
  donna1 : Position → Prop
  donna2 : Position → Prop
  eve1 : Position → Prop
  eve2 : Position → Prop

def trueStatements : Statements := {
  amy1 := fun p => p = Position.Second,
  amy2 := fun p => p = Position.Third,
  bruce1 := fun p => p = Position.Second,
  bruce2 := fun p => p = Position.Fourth,
  chris1 := fun p => p = Position.First,
  chris2 := fun p => p = Position.Second,
  donna1 := fun p => p = Position.Third,
  donna2 := fun p => p = Position.Fifth,
  eve1 := fun p => p = Position.Fourth,
  eve2 := fun p => p = Position.First,
}

theorem race_order (f : Statements) :
  f.amy1 Position.Second ∧ f.amy2 Position.Third ∧
  f.bruce1 Position.First ∧ f.bruce2 Position.Fourth ∧
  f.chris1 Position.Fifth ∧ f.chris2 Position.Second ∧
  f.donna1 Position.Fourth ∧ f.donna2 Position.Fifth ∧
  f.eve1 Position.Fourth ∧ f.eve2 Position.First :=
by
  sorry

end race_order_l729_729736


namespace distance_between_A_and_C_l729_729447

noncomputable def average_speed (d : ℝ) (t : ℝ) := d / t

theorem distance_between_A_and_C:
  let Eddy_time := 3
  let Freddy_time := 4
  let distance_AB := 900
  let speed_ratio := 4
  ∃ distance_AC,
    let Eddy_speed := average_speed distance_AB Eddy_time in
    let Freddy_speed := Eddy_speed / speed_ratio in
    let distance_AC := Freddy_speed * Freddy_time in
    distance_AC = 300 :=
by
  sorry

end distance_between_A_and_C_l729_729447


namespace maximize_ecological_benefits_l729_729703

noncomputable def ecological_benefits : ℝ → ℝ :=
  λ x, (27 * x) / (10 + x) - (3 * x) / 10 + 30

def investment_distribution : ℝ × ℝ :=
  (20, 80)

theorem maximize_ecological_benefits :
  ∀ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 100 →
    ecological_benefits x = ecological_benefits 20 →
    investment_distribution = (20, 80) :=
by
  sorry

end maximize_ecological_benefits_l729_729703


namespace water_needed_for_reaction_l729_729062

namespace Chemistry

inductive Molecule
| NaH
| H2O
| NaOH
| H2

open Molecule

def balanced_reaction : List (List (Molecule × ℕ)) :=
[
    [(NaH, 1), (H2O, 1)],
    [(NaOH, 1), (H2, 1)]
]

theorem water_needed_for_reaction (n : ℕ) 
    (h : balanced_reaction = [[(NaH, 1), (H2O, 1)], [(NaOH, 1), (H2, 1)]]) :
        n * (H2O) = n :=
by sorry

end Chemistry

end water_needed_for_reaction_l729_729062


namespace only_square_in_seq_l729_729438

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = (∑ i in Finset.range n, a (i+1)^2) + n)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m^2 = n

theorem only_square_in_seq (a : ℕ → ℕ) [sequence a] :
  ∀ n, is_perfect_square (a n) ↔ n = 1 :=
begin
  sorry
end

end only_square_in_seq_l729_729438


namespace evaluate_expression_l729_729933

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729933


namespace sum_of_smallest_and_largest_prime_between_1_and_50_l729_729777

theorem sum_of_smallest_and_largest_prime_between_1_and_50 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  in primes.head + primes.reverse.head = 49 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  exact rfl

end sum_of_smallest_and_largest_prime_between_1_and_50_l729_729777


namespace sum_of_remainders_l729_729346

theorem sum_of_remainders (a b c d : ℕ)
  (ha : a % 17 = 3) (hb : b % 17 = 5) (hc : c % 17 = 7) (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 :=
by
  sorry

end sum_of_remainders_l729_729346


namespace min_value_of_f_l729_729640

def f (x : Real) : Real := -3 * Real.sin x + 4 * Real.cos x

theorem min_value_of_f :
  ∃ x : Real, f x = -5 :=
sorry

end min_value_of_f_l729_729640


namespace evaluate_expression_l729_729935

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729935


namespace eval_expression_l729_729910

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729910


namespace eval_expr_l729_729816

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729816


namespace amount_per_person_is_correct_l729_729352

-- Define the total amount and the number of people
def total_amount : ℕ := 2400
def number_of_people : ℕ := 9

-- State the main theorem to be proved
theorem amount_per_person_is_correct : total_amount / number_of_people = 266 := 
by sorry

end amount_per_person_is_correct_l729_729352


namespace sin_35_pos_cos_167_neg_tan_3_neg_cot_neg_1_5_neg_l729_729421

theorem sin_35_pos : sin (35 * Real.pi / 180) > 0 := sorry

theorem cos_167_neg : cos (167 * Real.pi / 180) < 0 := sorry

theorem tan_3_neg : tan 3 < 0 := sorry

theorem cot_neg_1_5_neg : cot (-1.5) < 0 := sorry

end sin_35_pos_cos_167_neg_tan_3_neg_cot_neg_1_5_neg_l729_729421


namespace business_hours_correct_l729_729697

-- Define the business hours
def start_time : ℕ := 8 * 60 + 30   -- 8:30 in minutes
def end_time : ℕ := 22 * 60 + 30    -- 22:30 in minutes

-- Calculate total business hours in minutes and convert it to hours
def total_business_hours : ℕ := (end_time - start_time) / 60

-- State the business hour condition (which says the total business hour is 15 hours).
def business_hour_claim : ℕ := 15

-- Formulate the statement to prove: the claim that the total business hours are 15 hours is false.
theorem business_hours_correct : total_business_hours ≠ business_hour_claim := by
  sorry

end business_hours_correct_l729_729697


namespace largest_domain_of_g_l729_729286

variable (g : ℝ → ℝ)

-- Ensure the conditions on g are included in the Lean statement
def domain_condition_1 (x : ℝ) : Prop := ∀ (x ∈ domain g), -x ∈ domain g
def func_eq_condition (x : ℝ) : Prop := g(x) + g(-x) = x^2

theorem largest_domain_of_g :
  (∀ x : ℝ, domain_condition_1 g x → func_eq_condition g x) →
  { x : ℝ | x ∈ domain g } = set.univ :=
by
  intros
  sorry

end largest_domain_of_g_l729_729286


namespace explain_punctuality_behavior_l729_729598

-- Define the conditions for the problem
def large_group_punctuality (n : ℕ) : Prop :=
  n = 50 → ∀ t ∈ tourists(n), shows_up_on_time t

def small_group_punctuality (n : ℕ) : Prop :=
  n < 50 → ∃ t ∈ tourists(n), ¬ shows_up_on_time t

-- Define economic arguments function
-- These functions are mocked here as abbreviation as the original context is textual explanation 
def wealth_levels_argument : Prop := sorry
def value_perception_argument : Prop := sorry

-- Main theorem statement
theorem explain_punctuality_behavior (n : ℕ) :
  large_group_punctuality n ∧ small_group_punctuality n ↔ 
  wealth_levels_argument ∧ value_perception_argument := 
sorry

end explain_punctuality_behavior_l729_729598


namespace correct_statement_B_l729_729350

theorem correct_statement_B : (∃ x, x * x = 25 ∧ x = -5) ∧ ¬ (∀ y, y * y = 0.4 ∧ y = 0.2)
  ∧ ¬ (∀ z1 z2, (z1 = sqrt 81 ∧ z2 * z2 = (sqrt 81) ∧ z2 = 9) ∧ z2 = 3)
  ∧ ¬ (∃ w, w * w = 16 ∧ w = 4) :=
by {
  sorry
}

end correct_statement_B_l729_729350


namespace number_of_lines_at_least_n_l729_729367

theorem number_of_lines_at_least_n (n : ℕ) (points : Finset (ℝ × ℝ)) :
  points.card = n → ¬ ∃ l : Line ℝ, ∀ p ∈ points, p ∈ l → (∃ lines : Finset (Line ℝ), 
  ∀ l ∈ lines, ∃ p₁ p₂ ∈ points, p₁ ≠ p₂ ∧ l.pass_through p₁ ∧ l.pass_through p₂) 
  → lines.card ≥ n :=
by sorry

end number_of_lines_at_least_n_l729_729367


namespace shaded_region_area_l729_729069

section

-- Define points and shapes
structure point := (x : ℝ) (y : ℝ)
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length * square_side_length

-- Points defining the square and triangles within it
def point_O : point := ⟨0, 0⟩
def point_A : point := ⟨15, 0⟩
def point_B : point := ⟨40, 25⟩
def point_C : point := ⟨40, 40⟩
def point_D1 : point := ⟨25, 40⟩
def point_E : point := ⟨0, 15⟩

-- Function to calculate the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

-- Areas of individual triangles
def triangle1_area : ℝ := triangle_area 15 15
def triangle2_area : ℝ := triangle_area 25 25
def triangle3_area : ℝ := triangle_area 15 15

-- Total area of the triangles
def total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area

-- Shaded area calculation
def shaded_area : ℝ := square_area - total_triangles_area

-- Statement of the theorem to be proven
theorem shaded_region_area : shaded_area = 1062.5 := by sorry

end

end shaded_region_area_l729_729069


namespace angle_PQB_l729_729538

-- Definitions and conditions
variables {A B C D E P Q : Type}
variables {a b c : ℝ} -- Angles in degrees
variables [decidable_eq A] [decidable_eq B] [decidable_eq C]
variables {AD BE AB : ℝ}
variables (PQ_perpendicular_to_AB : PQ ⟂ AB) -- The property we're proving

-- Assume conditions given in the problem
axiom angle_A : a = 50
axiom angle_B : b = 40
axiom length_AB : AB = 12
axiom length_AD : AD = 2
axiom length_BE : BE = 2
axiom midpoint_P : midpoint A B P
axiom midpoint_Q : midpoint D E Q

-- Statement: we need to prove PQB forms a 90 degree angle.
theorem angle_PQB : angle PQ AB = 90 :=
by sorry

end angle_PQB_l729_729538


namespace equation_1_roots_equation_2_roots_l729_729616

open Real

theorem equation_1_roots :
  ∀ x, 2 * x^2 - 3 * x - 2 = 0 ↔ (x = -1 / 2 ∨ x = 2) :=
by
  sorry

theorem equation_2_roots :
  ∀ x, 2 * x^2 - 3 * x - 1 = 0 ↔ (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  sorry

end equation_1_roots_equation_2_roots_l729_729616


namespace households_3_houses_proportion_l729_729541

noncomputable def total_households : ℕ := 100000
noncomputable def ordinary_households : ℕ := 99000
noncomputable def high_income_households : ℕ := 1000

noncomputable def sampled_ordinary_households : ℕ := 990
noncomputable def sampled_high_income_households : ℕ := 100

noncomputable def sampled_ordinary_3_houses : ℕ := 40
noncomputable def sampled_high_income_3_houses : ℕ := 80

noncomputable def proportion_3_houses : ℝ := (sampled_ordinary_3_houses / sampled_ordinary_households * ordinary_households + sampled_high_income_3_houses / sampled_high_income_households * high_income_households) / total_households

theorem households_3_houses_proportion : proportion_3_houses = 0.048 := 
by
  sorry

end households_3_houses_proportion_l729_729541


namespace Elmer_eats_more_than_Penelope_l729_729247

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l729_729247


namespace complex_transform_l729_729333

theorem complex_transform :
  let z := -4 - 6 * complex.I
  let rotation := complex.of_real (real.sqrt 3 / 2) + complex.I * complex.of_real (1 / 2)
  let dilation := complex.of_real (real.sqrt 3)
  let combined := (rotation * dilation)
  (z * combined) = (-6 - 3 * (real.sqrt 3) - (2 * (real.sqrt 3) + 9) * complex.I) :=
by
  let z := -4 - 6 * complex.I
  let rotation := complex.of_real (real.sqrt 3 / 2) + complex.I * complex.of_real (1 / 2)
  let dilation := complex.of_real (real.sqrt 3)
  let combined := (rotation * dilation)
  have res : (z * combined) = (-6 - 3 * (real.sqrt 3) - (2 * (real.sqrt 3) + 9) * complex.I) := sorry
  exact res

end complex_transform_l729_729333


namespace tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l729_729085

open Real

-- Part 1
theorem tan_x_min_x_div_x_min_sin_x_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) :
  (tan x - x) / (x - sin x) > 2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x + 2 * sin x - a * x > 0) → a ≤ 3 :=
sorry

end tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l729_729085


namespace shaded_white_ratio_proof_l729_729673

def shaded_white_area_ratio (n : ℕ) : ℚ :=
  if n = 1 then 5 / 3 else shaded_white_area_ratio (n-1) * 5 / 3

theorem shaded_white_ratio_proof :
  shaded_white_area_ratio 1 = 5 / 3 :=
by
  -- the proof is omitted
  sorry

end shaded_white_ratio_proof_l729_729673


namespace triangle_inequality_valid_x_values_l729_729299

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l729_729299


namespace symmetric_iff_m_eq_neg2_l729_729054

-- Definitions from conditions
def f (x : ℝ) (m : ℝ) := x^2 + m * x + 1

-- Definitions to prove symmetry about x=1
def symmetric_about_x1 (f : ℝ → ℝ) := ∀ x, f (2 - x) = f x

-- The statement to prove
theorem symmetric_iff_m_eq_neg2 : 
  (symmetric_about_x1 (f (fun x => x) m)) ↔ (m = -2) :=
sorry

end symmetric_iff_m_eq_neg2_l729_729054


namespace find_area_of_triangle_BOC_l729_729206

-- Define the conditions given in the problem
def Trapezoid := ...
constant A B C D O : Point
axiom AD_perpendicular_to_bases : Perpendicular (Line A D) bases
axiom AD_eq_9 : length (Segment A D) = 9
axiom CD_eq_12 : length (Segment C D) = 12
axiom AO_eq_6 : length (Segment A O) = 6
axiom O_diagonal_intersection : DiagonalIntersection A B C D O

-- Define the main theorem to find the area of triangle BOC
theorem find_area_of_triangle_BOC (A B C D O : Point) : 
  Trapezoid A B C D →
  Perpendicular (Line A D) bases →
  length (Segment A D) = 9 →
  length (Segment C D) = 12 →
  length (Segment A O) = 6 →
  DiagonalIntersection A B C D O →
  area (Triangle B O C) = 108 / 5 :=
by
  sorry

end find_area_of_triangle_BOC_l729_729206


namespace selling_price_of_car_l729_729254

theorem selling_price_of_car (purchase_price repair_cost : ℝ) (profit_percent : ℝ) 
    (h1 : purchase_price = 42000) (h2 : repair_cost = 8000) (h3 : profit_percent = 29.8) :
    (purchase_price + repair_cost) * (1 + profit_percent / 100) = 64900 := 
by 
  -- The proof will go here
  sorry

end selling_price_of_car_l729_729254


namespace probability_no_distinct_positive_real_roots_l729_729741

theorem probability_no_distinct_positive_real_roots :
  let pairs := (Finset.Icc (-6 : ℤ) 6).product (Finset.Icc (-6 : ℤ) 6) in
  let total_pairs := pairs.card in
  let discriminant_nonpositive_or_non_positive_roots (b c : ℤ) :=
    b^2 - 4 * c ≤ 0 ∨ ( -b + Real.sqrt (b^2 - 4 * c)) / 2 ≤ 0 ∨ ( -b - Real.sqrt (b^2 - 4 * c)) / 2 ≤ 0 in
  let invalid_pairs := pairs.filter (λ p, discriminant_nonpositive_or_non_positive_roots p.1 p.2) in
  total_pairs = 169 ∧ invalid_pairs.card = 154 → 
  (invalid_pairs.card : ℚ) / total_pairs = 154 / 169 :=
by 
  sorry

end probability_no_distinct_positive_real_roots_l729_729741


namespace smallest_positive_period_of_f_max_min_values_of_f_l729_729508

-- Define the function f
def f (x : ℝ) : ℝ := cos x * sin (x + π / 3) - sqrt 3 * cos x ^ 2 + sqrt 3 / 4

-- Statement (1): Smallest positive period
theorem smallest_positive_period_of_f : 
  (∀ x, f (x + π) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T >= π) := sorry

-- Statement (2): Maximum and minimum values on the given interval
theorem max_min_values_of_f : 
  (∀ x ∈ Icc (-π/4) (π/4), -1/2 <= f x ∧ f x <= 1/4) 
  ∧ (f (-π/12) = -1/2) 
  ∧ (f (π/4) = 1/4) := sorry

end smallest_positive_period_of_f_max_min_values_of_f_l729_729508


namespace sum_of_values_l729_729103

theorem sum_of_values (θ φ : ℝ) 
  (h₁ : (sin θ) ^ 6 / (sin φ) ^ 3 + (cos θ) ^ 6 / (cos φ) ^ 3 = 1) : 
  (sin φ) ^ 6 / (sin θ) ^ 3 + (cos φ) ^ 6 / (cos θ) ^ 3 = 1 :=
sorry

end sum_of_values_l729_729103


namespace eval_expression_l729_729915

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l729_729915


namespace find_a_l729_729579

theorem find_a (a : ℕ) : 
  (a >= 100 ∧ a <= 999) ∧ 7 ∣ (504000 + a) ∧ 9 ∣ (504000 + a) ∧ 11 ∣ (504000 + a) ↔ a = 711 :=
by {
  sorry
}

end find_a_l729_729579


namespace remainder_of_sum_l729_729663

theorem remainder_of_sum (a b c : ℕ) (h₁ : a % 15 = 11) (h₂ : b % 15 = 12) (h₃ : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
by 
  sorry

end remainder_of_sum_l729_729663


namespace equation1_solutions_equation2_solutions_l729_729271

theorem equation1_solutions (x : ℝ) :
  x ^ 2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := by
  sorry

theorem equation2_solutions (x : ℝ) :
  2 * x ^ 2 - 2 * x = 1 ↔ x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2 := by
  sorry

end equation1_solutions_equation2_solutions_l729_729271


namespace triangle_area_integer_l729_729429

theorem triangle_area_integer (x2 y2 x3 y3 : ℤ) (hx2 : x2 % 2 = 1) (hy2 : y2 % 2 = 1) (hx3 : x3 % 2 = 1) (hy3 : y3 % 2 = 1) :
  ∃ k : ℤ, (1 * (y2 - y3) + x2 * (y3 - 1) + x3 * (1 - y2)) = 2 * k :=
begin
  sorry
end

end triangle_area_integer_l729_729429


namespace evaluate_expression_l729_729861

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729861


namespace inclination_angle_x_eq_one_l729_729532

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one_l729_729532


namespace platform_length_l729_729383

noncomputable def train_length : ℕ := 1200
noncomputable def time_to_cross_tree : ℕ := 120
noncomputable def time_to_pass_platform : ℕ := 230

theorem platform_length
  (v : ℚ)
  (h1 : v = train_length / time_to_cross_tree)
  (total_distance : ℚ)
  (h2 : total_distance = v * time_to_pass_platform)
  (platform_length : ℚ)
  (h3 : total_distance = train_length + platform_length) :
  platform_length = 1100 := by 
  sorry

end platform_length_l729_729383


namespace eval_expression_l729_729832

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729832


namespace probability_three_heads_in_eight_tosses_l729_729002

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729002


namespace eval_expr_l729_729807

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729807


namespace depth_of_first_hole_l729_729377

theorem depth_of_first_hole :
  (45 * 8 * (80 * 6 * 40) / (45 * 8) : ℝ) = 53.33 := by
  -- This is where you would provide the proof, but it will be skipped with 'sorry'
  sorry

end depth_of_first_hole_l729_729377


namespace girls_25_percent_less_false_l729_729037

theorem girls_25_percent_less_false (g b : ℕ) (h : b = g * 125 / 100) : (b - g) / b ≠ 25 / 100 := by
  sorry

end girls_25_percent_less_false_l729_729037


namespace simplify_frac_1_eq_simplify_frac_2_eq_xy_val_eq_series_sum_eq_l729_729671

noncomputable def simplify_frac_1 : ℝ := 1 / (Real.sqrt 5 + 2)
noncomputable def simplify_frac_2 : ℝ := 1 / (Real.sqrt 6 - Real.sqrt 5)
noncomputable def x_def : ℝ := 1 / (3 + 2 * Real.sqrt 2)
noncomputable def y_def : ℝ := 1 / (3 - 2 * Real.sqrt 2)
noncomputable def series_sum : ℝ := (Finset.range 2022).sum (λ n, 1 / (Real.sqrt (n + 2) + Real.sqrt (n + 1)))

theorem simplify_frac_1_eq : simplify_frac_1 = Real.sqrt 5 - 2 :=
by sorry

theorem simplify_frac_2_eq : simplify_frac_2 = Real.sqrt 6 + Real.sqrt 5 :=
by sorry

theorem xy_val_eq : (x_def - y_def) ^ 2 - x_def * y_def = 31 :=
by sorry

theorem series_sum_eq : series_sum = Real.sqrt 2022 - 1 :=
by sorry

end simplify_frac_1_eq_simplify_frac_2_eq_xy_val_eq_series_sum_eq_l729_729671


namespace integer_values_x_possible_l729_729302

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l729_729302


namespace evaluate_expression_l729_729789

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729789


namespace evaluate_expression_l729_729792

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729792


namespace quadratic_inequality_solution_l729_729441

theorem quadratic_inequality_solution :
  {x : ℝ | 3 * x^2 + 5 * x < 8} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
sorry

end quadratic_inequality_solution_l729_729441


namespace cats_left_in_store_l729_729715

theorem cats_left_in_store :
  ∀ (init_siamese init_house init_persian total_sold : ℕ)
    (percent_siamese percent_house percent_persian sold_siamese sold_house sold_persian : ℕ),
    init_siamese = 15 →
    init_house = 49 →
    init_persian = 21 →
    total_sold = 19 →
    percent_siamese = 40 →
    percent_house = 35 →
    percent_persian = 25 →
    sold_siamese = 7 →
    sold_house = 7 →
    sold_persian = 5 →
    let remaining_siamese := init_siamese - sold_siamese in
    let remaining_house := init_house - sold_house in
    let remaining_persian := init_persian - sold_persian in
    remaining_siamese + remaining_house + remaining_persian = 66 :=
begin
  intros,
  sorry
end

end cats_left_in_store_l729_729715


namespace vector_magnitude_l729_729493

variables {α : Type*} [inner_product_space ℝ α]

theorem vector_magnitude
  (a b : α)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (angle_ab : real.angle a b = real.pi / 3) :
  ∥3 • a + b∥ = sqrt 13 := 
sorry

end vector_magnitude_l729_729493


namespace no_true_statements_l729_729989

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * sin (2 * x) + sin x ^ 2

theorem no_true_statements :
  ¬((∃ c : ℝ, ∃ k : ℤ, c = (k * π / 2 + θ / 2) ∧ f (π / 12) = 0) ∨
    (∃ p > 0, ∀ x, f (x + p) = f x ∧ p = 2 * π) ∨
    (∀ x ∈ Icc (-π / 6) (π / 3), f' x > 0) ∨
    ( ∃ k : ℤ, x = k * π / 2 ∧ x = π / 3)) := 
sorry

end no_true_statements_l729_729989


namespace sum_exponents_simplified_outside_cube_root_l729_729265

theorem sum_exponents_simplified_outside_cube_root : 
  ∀ (a b c : ℝ), 
  let x := real.cbrt (40 * a^5 * b^8 * c^14) in
  let simplified_x := (2 * a * b^2 * c^4) * real.cbrt (5 * a * b^2 * c^2) in
  (x = simplified_x) → (1 + 2 + 4 = 7)
:= 
begin
  intros,
  sorry
end

end sum_exponents_simplified_outside_cube_root_l729_729265


namespace selection_methods_count_l729_729476

-- Definition of terms for the conditions
def students : Finset ℕ := {1, 2, 3, 4, 5, 6}
def total_students := 6
def selected_students := 4
def students_must_include := {1, 2} -- assuming 1 represents A and 2 represents B

-- Theorem stating the number of different methods of selection
theorem selection_methods_count :
  (∃ (x y : Finset ℕ), 
    x ⊆ students ∧
    y ⊆ students ∧
    students_must_include ⊆ x ∧
    (x.card = 4) ∧
    -- B can only receive the baton directly from A in 3 ways
    (∀ p ∈ (x : finset ℕ).powerset.filter (λ s, s.card = 2), 
      ∃ (a b c : ℕ), 
        ({a, b} = students_must_include) ∧
        ({a, c} = students_must_include)
       )) → 
  3 * (4.choose 2) * (2.factorial) = 36 :=
by 
  sorry

end selection_methods_count_l729_729476


namespace jason_initial_money_l729_729210

theorem jason_initial_money (M : ℝ) 
  (h1 : M - (M / 4 + 10 + (2 / 5 * (3 / 4 * M - 10) + 8)) = 130) : 
  M = 320 :=
by
  sorry

end jason_initial_money_l729_729210


namespace nat_pairs_solution_l729_729456

theorem nat_pairs_solution (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) :=
by
  sorry

end nat_pairs_solution_l729_729456


namespace initial_apples_l729_729564

-- Define the number of initial fruits
def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def fruits_given : ℕ := 40
def fruits_left : ℕ := 15

-- Define the equation for the initial number of fruits
def initial_total_fruits (A : ℕ) : Prop :=
  initial_plums + initial_guavas + A = fruits_left + fruits_given

-- Define the proof problem to find the number of apples
theorem initial_apples : ∃ A : ℕ, initial_total_fruits A ∧ A = 21 :=
  by
    sorry

end initial_apples_l729_729564


namespace evaluate_expression_l729_729850

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l729_729850


namespace cyclist_total_time_l729_729356

def distance : ℝ := 400 -- Length of the hill in meters
def speed_kmh : ℝ := 7.2 -- Speed while climbing in km/h
def speed_descending_factor : ℝ := 2 -- Factor by which descending speed is greater

-- Conversion from km/h to m/s
def speed_climbing : ℝ := speed_kmh * 1000 / 3600 -- Climbing speed in m/s

-- Time to climb in seconds
def time_to_climb : ℝ := distance / speed_climbing

-- Speed while descending in m/s
def speed_descending : ℝ := speed_descending_factor * speed_climbing

-- Time to descend in seconds
def time_to_descend : ℝ := distance / speed_descending

-- Total time to climb and descend in seconds
def total_time : ℝ := time_to_climb + time_to_descend

theorem cyclist_total_time :
  total_time = 300 :=
by
  sorry

end cyclist_total_time_l729_729356


namespace max_dot_product_of_octagon_vectors_l729_729312

open Real

-- Conditions for the regular octagon
noncomputable def vertices : Fin 8 → ℝ × ℝ
| ⟨k, _⟩ => (cos ((2 * k) * π / 8), sin ((2 * k) * π / 8))

-- Definition for the vector between two vertices
def vec (i j : Fin 8) := (vertices j).1 - (vertices i).1, (vertices j).2 - (vertices i).2

-- Unit vector A1A2
def A₁A₂ := vec 0 1

-- Dot product definition
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- The goal of the proof
theorem max_dot_product_of_octagon_vectors :
  ∃ (i j : Fin 8), dot_product (vec i j) A₁A₂ = sqrt(2) + 1 := 
sorry

end max_dot_product_of_octagon_vectors_l729_729312


namespace find_largest_m_l729_729095

variables (a b c t : ℝ)
def f (x : ℝ) := a * x^2 + b * x + c

theorem find_largest_m (a_ne_zero : a ≠ 0)
  (cond1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x) ∧ f a b c x ≥ x)
  (cond2 : ∀ x : ℝ, 0 < x ∧ x < 2 → f a b c x ≤ ((x + 1) / 2)^2)
  (cond3 : ∃ x : ℝ, ∀ y : ℝ, f a b c y ≥ f a b c x ∧ f a b c x = 0) :
  ∃ m : ℝ, 1 < m ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x) ∧ m = 9 := sorry

end find_largest_m_l729_729095


namespace evaluate_expression_l729_729860

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729860


namespace problem_a_problem_b_problem_c_problem_d_l729_729086

variable {a b : ℝ}

theorem problem_a (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : ab ≤ 1 / 8 := sorry

theorem problem_b (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (1 / a) + (8 / b) ≥ 25 := sorry

theorem problem_c (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 := sorry

theorem problem_d (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 - b^2 > -1 / 4 := sorry

end problem_a_problem_b_problem_c_problem_d_l729_729086


namespace evaluation_of_expression_l729_729892

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729892


namespace table_last_number_l729_729024

theorem table_last_number:
  (∃ (table : ℕ → ℕ → ℕ), 
    (∀ i : ℕ, i < 100 → table 0 i = i + 1) ∧
    (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ j < i → table i j = table (i - 1) (j - 1) + table (i - 1) j) ∧
    (∀ i : ℕ, i > 0 → table i i = 2 * table (i - 1) (i - 1)))
    → ∃ k : ℕ, table 99 0 = 101 * 2 ^ 98) :=
sorry

end table_last_number_l729_729024


namespace seating_arrangements_l729_729548

theorem seating_arrangements (n : ℕ) (h : n = 6) :
  ∃ (m : ℕ), (m = 48) ∧ (∃ (a b : ℕ) (c : Finset (Fin 5)),
  (a = 2! ∧ b = 4! ∧ m = a * b)) :=
by
  have h2 : 2! = 2 := rfl
  have h4 : 4! = 24 := rfl
  use 48
  split
  · rfl
  · use 2!, 4!, Finset.univ
    split
    · exact h2
    split
    · exact h4
    · rw [h2, h4]
      rfl
  sorry

end seating_arrangements_l729_729548


namespace evaluation_of_expression_l729_729900

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729900


namespace vertical_line_divides_triangle_equal_area_l729_729047

theorem vertical_line_divides_triangle_equal_area :
  let A : (ℝ × ℝ) := (1, 2)
  let B : (ℝ × ℝ) := (1, 1)
  let C : (ℝ × ℝ) := (10, 1)
  let area_ABC := (1 / 2 : ℝ) * (C.1 - A.1) * (A.2 - B.2)
  let a : ℝ := 5.5
  let area_left_triangle := (1 / 2 : ℝ) * (a - A.1) * (A.2 - B.2)
  let area_right_triangle := (1 / 2 : ℝ) * (C.1 - a) * (A.2 - B.2)
  area_left_triangle = area_right_triangle :=
by
  sorry

end vertical_line_divides_triangle_equal_area_l729_729047


namespace evaluate_expression_l729_729924

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l729_729924


namespace yolanda_free_throws_per_game_l729_729686

theorem yolanda_free_throws_per_game (
  total_points : ℕ,
  total_games : ℕ,
  twos_per_game : ℕ,
  threes_per_game : ℕ
) (h1 : total_points = 345) 
  (h2 : total_games = 15) 
  (h3 : twos_per_game = 5) 
  (h4 : threes_per_game = 3) : 
  (total_points / total_games) - ((twos_per_game * 2) + (threes_per_game * 3)) = 4 := 
sorry

end yolanda_free_throws_per_game_l729_729686


namespace vector_addition_example_l729_729451

theorem vector_addition_example : 
  (\begin{array}{c} -5 \\ 1 \\ -4 \end{array} : ℝ^3) + (\begin{array}{c} 0 \\ 8 \\ -4 \end{array} : ℝ^3) = (\begin{array}{c} -5 \\ 9 \\ -8 \end{array} : ℝ^3) :=
sorry

end vector_addition_example_l729_729451


namespace cos_angle_between_planes_l729_729221

noncomputable def cos_theta : ℝ :=
  let n1 : ℝ × ℝ × ℝ := (3, -2, 1)
  let n2 : ℝ × ℝ × ℝ := (4, 1, -3)
  let dot_product : ℝ := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude_n1 : ℝ := real.sqrt (n1.1^2 + n1.2^2 + n1.3^2)
  let magnitude_n2 : ℝ := real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)
  dot_product / (magnitude_n1 * magnitude_n2)

theorem cos_angle_between_planes :
  cos_theta = 7 / (2 * real.sqrt 91) :=
by
  sorry

end cos_angle_between_planes_l729_729221


namespace probability_two_draws_l729_729406

def probability_first_red_second_kd (total_cards : ℕ) (red_cards : ℕ) (king_of_diamonds : ℕ) : ℚ :=
  (red_cards / total_cards) * (king_of_diamonds / (total_cards - 1))

theorem probability_two_draws :
  let total_cards := 52
  let red_cards := 26
  let king_of_diamonds := 1
  probability_first_red_second_kd total_cards red_cards king_of_diamonds = 1 / 102 :=
by {
  sorry
}

end probability_two_draws_l729_729406


namespace eval_expression_l729_729818

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729818


namespace pa_pb_pc_qa_qb_qc_on_circle_l729_729369

-- Here, we define the geometric entities and their properties.
variables {A B C I G P_A Q_A P_B Q_B P_C Q_C : Point}
variable {foot_of_perpendicular : Point -> Line -> Point}

-- Define our conditions
def triangle_conditions (A B C I G P_A Q_A P_B Q_B P_C Q_C : Point) : Prop :=
  let exterior_angle_bisector_B := line_of_angle_bisector_exterior B
  let exterior_angle_bisector_C := line_of_angle_bisector_exterior C in
  P_A = foot_of_perpendicular C exterior_angle_bisector_B ∧
  Q_A = foot_of_perpendicular B exterior_angle_bisector_C ∧
  P_B = foot_of_perpendicular A exterior_angle_bisector_C ∧
  Q_B = foot_of_perpendicular C exterior_angle_bisector_A ∧
  P_C = foot_of_perpendicular B exterior_angle_bisector_A ∧
  Q_C = foot_of_perpendicular A exterior_angle_bisector_B

-- Define the statement to be proven
theorem pa_pb_pc_qa_qb_qc_on_circle {A B C I G P_A Q_A P_B Q_B P_C Q_C : Point} :
  triangle_conditions A B C I G P_A Q_A P_B Q_B P_C Q_C →
  ∃ O : Point, (circle_through_points P_A P_B P_C Q_A Q_B Q_C O) ∧ line_contains_point I G O :=
sorry

end pa_pb_pc_qa_qb_qc_on_circle_l729_729369


namespace exists_number_divisible_by_24_and_cube_root_in_range_l729_729452

noncomputable def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

theorem exists_number_divisible_by_24_and_cube_root_in_range :
  ∃ n : ℕ, is_divisible_by n 24 ∧ (8 < real.cbrt n) ∧ (real.cbrt n < 8.2) ∧ (n = 528) :=
begin
  sorry
end

end exists_number_divisible_by_24_and_cube_root_in_range_l729_729452


namespace junior_scores_89_l729_729542

theorem junior_scores_89 (n : ℕ) (hn : n > 0) 
  (juniors seniors : ℕ)
  (hj : juniors = 0.2 * n) (hs : seniors = 0.8 * n)
  (overall_avg_score : ℝ) 
  (hoas : overall_avg_score = 85)
  (avg_senior_score : ℝ)
  (has : avg_senior_score = 84)
  (total_score : ℕ → ℝ) 
  (h_total_score : total_score n = 85 * n) :
  (∀ j, j ∈ (finset.range juniors) → total_score j = 17.8 * n / 0.2 * n) ↔ (∀ junior_score, junior_score = 89) :=
by
  sorry

end junior_scores_89_l729_729542


namespace tourists_walking_speed_l729_729015

-- Define the conditions
def tourists_start_time := 3 + 10 / 60 -- 3:10 A.M.
def bus_pickup_time := 5 -- 5:00 A.M.
def bus_speed := 60 -- 60 km/h
def early_arrival := 20 / 60 -- 20 minutes earlier

-- This is the Lean 4 theorem statement
theorem tourists_walking_speed : 
  (bus_speed * (10 / 60) / (100 / 60)) = 6 := 
by
  sorry

end tourists_walking_speed_l729_729015


namespace figure_with_two_axes_has_center_l729_729208

def figure_has_axes_of_symmetry (F : Type) : Prop :=
  ∃ l₁ l₂ : ℝ → ℝ → Prop, (∀ x y, l₁ x y ↔ l₂ x y) ∧ (l₁ ≠ l₂)

def figure_has_center_of_symmetry (F : Type) : Prop :=
  ∃ C : ℝ × ℝ, ∀ x y, (x - C.1, y - C.2) ∈ F ↔ (-x - C.1, -y - C.2) ∈ F

theorem figure_with_two_axes_has_center (F : Type) :
  figure_has_axes_of_symmetry F → figure_has_center_of_symmetry F :=
sorry

end figure_with_two_axes_has_center_l729_729208


namespace eval_expression_l729_729965

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729965


namespace AmyWins_l729_729412

structure GamePosition where
  red : Nat
  blue : Nat
  deriving DecidableEq

def legalMove (p : GamePosition) : List GamePosition :=
  match p with
  | ⟨r, b⟩ =>
    let moves :=
      if r ≥ 2 then [{ red := r - 2, blue := b }] else []
      ++ if b ≥ 2 then [{ red := r, blue := b - 2 }] else []
      ++ if r ≥ 1 then [{ red := r - 1, blue := b + 1 }] else []
    moves.filter (λ p => p.red ≥ 0 ∧ p.blue >= 0)

theorem AmyWins : ∀ (p : GamePosition), p = ⟨20, 14⟩ -> (exists! m ∈ legalMove p, ∀ p', p' ∈ legalMove m → p'.red = 0 ∧ p'.blue ≤ 1) :=
by
  sorry

end AmyWins_l729_729412


namespace average_weight_increase_l729_729623

-- Define the initial conditions as given in the problem
def W_old : ℕ := 53
def W_new : ℕ := 71
def N : ℕ := 10

-- Average weight increase after replacing one oarsman
theorem average_weight_increase : (W_new - W_old : ℝ) / N = 1.8 := by
  sorry

end average_weight_increase_l729_729623


namespace impossible_result_l729_729990

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬ (∃ f1 f_1 : ℤ, f1 = a * Real.sin 1 + b + c ∧ f_1 = -a * Real.sin 1 - b + c ∧ (f1 = 1 ∧ f_1 = 2)) :=
by
  sorry

end impossible_result_l729_729990


namespace union_sets_l729_729515

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem union_sets :
  M ∪ N = {x | x ≤ 1} :=
by
  sorry

end union_sets_l729_729515


namespace sum_n_k_l729_729631

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l729_729631


namespace geometric_series_inequality_l729_729604

theorem geometric_series_inequality (a : ℝ) (n : ℕ) (h1 : a > 0) (h2 : n > 1) :
  (1 + a + a^2 + ... + a^n) / (a + a^2 + ... + a^(n-1)) ≥ (n + 1) / (n - 1)
:= 
sorry

end geometric_series_inequality_l729_729604


namespace exists_a_9_range_of_m_l729_729772

def f (x : ℝ) (a : ℝ) : ℝ := x + a / x

theorem exists_a_9 : 
  ∃ a : ℝ, a = 9 ∧ (∀ x : ℝ, (0 < x ∧ x ≤ 3) → (1 - a / x^2) < 0) ∧ (∀ x : ℝ, (3 ≤ x) → (1 - a / x^2) > 0) :=
by
  sorry

theorem range_of_m (m a : ℝ) (h_a : a = 9) (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x + a / x - m ≤ 0) : 
  if a ≤ 4 then m ≥ 4 + 9 / 4 else m ≥ 10 :=
by
  sorry

end exists_a_9_range_of_m_l729_729772


namespace quotient_of_23_div_6_l729_729675

theorem quotient_of_23_div_6 :
  ∃ A : ℤ, 23 = 6 * A + 5 ∧ A = 3 :=
begin
  sorry
end

end quotient_of_23_div_6_l729_729675


namespace distance_from_city_l729_729695

-- Define the conditions
variables (d_a d_b : ℝ) 
           (v_b : ℝ) 
           (v_car : ℝ := 2 * (1.5 * v_b))
           (v_truck : ℝ := 1.5 * v_car)
           (time_in_city : ℝ)

-- Given conditions
def conditions := 
  d_a = d_b + 3 ∧
  time_in_city = (d_b / 2) / v_b + (d_b / 2) / v_truck ∧
  time_in_city = (d_a / 2) / (1.5 * v_b) + (d_a / 2) / v_car

-- Theorem to prove
theorem distance_from_city (h : conditions d_a d_b v_b time_in_city) : d_a = 16.5 :=
by
  sorry

end distance_from_city_l729_729695


namespace eyes_that_saw_the_plane_l729_729321

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end eyes_that_saw_the_plane_l729_729321


namespace triangle_larger_angle_longer_side_l729_729559

-- We define a structure for a triangle with sides opposite the corresponding angles
structure Triangle := 
  (A B C : Type)
  (a b c : ℝ) -- lengths of sides opposites to angles A, B, and C respectively
  (angle : A → ℝ)
  (ineq : ℝ → ℝ → Prop)
  (angle_ineq : ∀ {A B : A}, (angle A < angle B) ↔ (ineq A B))

-- Assuming a triangle with the major premise that a larger angle is opposite a longer side
theorem triangle_larger_angle_longer_side :
  ∀ {T : Triangle}, (T.angle A < T.angle B) → (T.ineq a b) :=
by 
  sorry

end triangle_larger_angle_longer_side_l729_729559


namespace eval_expression_l729_729954

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729954


namespace perp_midpoints_to_side_l729_729595

open EuclideanGeometry
open Classical

noncomputable def triangle := {p1 p2 p3 : Point}

variables {A B C M I₁ J₁ I₂ J₂ : Point} {ABC : triangle}

-- Given a triangle ABC with point M on side AB
def on_side (t : triangle) (M : Point) : Prop :=
  ∃ (A B : Point), M ∈ Segment A B ∧ A ∈ t ∧ B ∈ t

-- I₁ is the incenter of triangle ACM
def incenter_acm (A C M I₁ : Point) : Prop :=
  IsIncenter I₁ (Triangle A C M)

-- J₁ is the excenter opposite CM in triangle ACM
def excenter_acm (A C M J₁ : Point) : Prop :=
  IsExcenter J₁ (Triangle A C M) (opposite_side C M)

-- I₂ is the incenter of triangle BCM
def incenter_bcm (B C M I₂ : Point) : Prop :=
  IsIncenter I₂ (Triangle B C M)

-- J₂ is the excenter opposite CM in triangle BCM
def excenter_bcm (B C M J₂ : Point) : Prop :=
  IsExcenter J₂ (Triangle B C M) (opposite_side C M)

-- Statement: Prove that the line passing through the midpoints of segments I₁I₂ and J₁J₂ is perpendicular to AB
theorem perp_midpoints_to_side (h₁ : on_side ABC M) 
  (h₂ : incenter_acm A C M I₁) 
  (h₃ : excenter_acm A C M J₁) 
  (h₄ : incenter_bcm B C M I₂) 
  (h₅ : excenter_bcm B C M J₂): 
  let I₁I₂_mid := midpoint I₁ I₂ 
  let J₁J₂_mid := midpoint J₁ J₂ 
  let L := line_through I₁I₂_mid J₁J₂_mid 
  perp_to_side L (Segment A B) :=
sorry 

end perp_midpoints_to_side_l729_729595


namespace mrs_hilt_chapters_read_l729_729239

def number_of_books : ℝ := 4.0
def chapters_per_book : ℝ := 4.25
def total_chapters_read : ℝ := number_of_books * chapters_per_book

theorem mrs_hilt_chapters_read : total_chapters_read = 17 :=
by
  unfold total_chapters_read
  norm_num
  sorry

end mrs_hilt_chapters_read_l729_729239


namespace matrix_problem_l729_729502

variables {n : Type*} [fintype n] [decidable_eq n]
variables {α : Type*} [field α]
variables (A : matrix n n α)

theorem matrix_problem (A_invertible : invertible A)
  (h : (A - 3 • (1 : matrix n n α)) ⬝ (A - 5 • (1 : matrix n n α)) = 0) :
  A + 10 • (A⁻¹) = 8 • (1 : matrix n n α) :=
sorry

end matrix_problem_l729_729502


namespace tv_cost_difference_correct_l729_729345

def TV1 := {width := 24, height := 16, original_cost := 840, discount_rate := 0.10, tax_rate := 0.05}
def TV2 := {width := 48, height := 32, original_cost := 1800, discount1_rate := 0.20, discount2_rate := 0.15, tax_rate := 0.08}

noncomputable def cost_with_discount tax (price discount_rate) := 
  let discounted_price := price - (price * discount_rate)
  let total_cost := discounted_price + (discounted_price * tax)
  total_cost

noncomputable def cost_per_square_inch width height total_cost := 
  let area := width * height
  total_cost / area

theorem tv_cost_difference_correct : 
  let cost_tv1 := cost_with_discount TV1.tax_rate TV1.original_cost TV1.discount_rate
  let cost_tv2 := cost_with_discount TV2.tax_rate (cost_with_discount 0 TV2.original_cost TV2.discount1_rate) TV2.discount2_rate
  let cost_per_sq_inch_tv1 := cost_per_square_inch TV1.width TV1.height cost_tv1
  let cost_per_sq_inch_tv2 := cost_per_square_inch TV2.width TV2.height cost_tv2
  cost_per_sq_inch_tv1 - cost_per_sq_inch_tv2 = 1.2073 := 
sorry

end tv_cost_difference_correct_l729_729345


namespace diagonals_of_octagon_l729_729155

theorem diagonals_of_octagon : 
  let n := 8 in 
  let total_line_segments := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_line_segments - sides in
  diagonals = 20 := 
  by 
    let n := 8
    let total_line_segments := (n * (n - 1)) / 2
    let sides := n
    let diagonals := total_line_segments - sides
    have h : diagonals = 20 := sorry
    exact h

end diagonals_of_octagon_l729_729155


namespace fixed_point_always_on_line_l729_729222

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line_l729_729222


namespace rhombus_properties_l729_729404

noncomputable def side_length (d1 d2 : ℝ) : ℝ :=
  Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

noncomputable def area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_properties :
  let d1 := 13
  let d2 := 20
  side_length d1 d2 ≈ 11.93 ∧ area d1 d2 = 130 :=
by
  let d1 := 13
  let d2 := 20
  have length_correct : side_length d1 d2 ≈ 11.93 := sorry
  have area_correct : area d1 d2 = 130 := sorry
  exact ⟨length_correct, area_correct⟩

end rhombus_properties_l729_729404


namespace difference_between_max_and_min_z_l729_729225

variables (x y z : ℝ)

theorem difference_between_max_and_min_z :
  x + y + z = 5 →
  x^2 + y^2 + z^2 = 20 →
  x * y = 2 →
  let zs := {z | ∃ x y : ℝ, (x + y + z = 5 ∧ x^2 + y^2 + z^2 = 20 ∧ x * y = 2)} in
  (sup zs - inf zs = 6) :=
by
  intros h1 h2 h3 zs_def
  sorry

end difference_between_max_and_min_z_l729_729225


namespace tangent_angle_l729_729216

theorem tangent_angle (ω1 ω2 : Circle) (S T Q X A B P: Point) (r1 r2 : ℝ) 
  (h1 : ω1.radius = 3) 
  (h2 : ω2.radius = 12) 
  (h3 : ω1.tangent ω2 P) -- externally tangent at P
  (h4 : external_tangent ω1 ω2 S T) -- S on ω1 and T on ω2
  (h5 : internal_tangent_intersect Q ω1 ω2) -- internal tangent intersects at Q
  (h6 : on_ray QP X) -- X lies on the ray QP
  (h7 : dist Q X = 10) 
  (h8 : second_intersection XS ω1 A) -- XS intersects ω1 second time at A
  (h9 : second_intersection XT ω2 B) -- XT intersects ω2 second time at B
  : tan_angle A P B = 2 / 3 := 
sorry

end tangent_angle_l729_729216


namespace count_correct_derivatives_l729_729643

noncomputable def f₁ (x : ℝ) : ℝ := 3^x
noncomputable def f₂ (x : ℝ) : ℝ := log x / log 2
noncomputable def f₃ (x : ℝ) : ℝ := exp x
noncomputable def f₄ (x : ℝ) : ℝ := 1 / log x
noncomputable def f₅ (x : ℝ) : ℝ := x * exp x

theorem count_correct_derivatives :
  let c1 := (deriv f₁ x ≠ 3^x * log 3)
      c2 := (deriv f₂ x = 1 / (x * log 2))
      c3 := (deriv f₃ x = exp x)
      c4 := (deriv (deriv f₄) x ≠ x)
      c5 := (deriv f₅ x ≠ exp x - x * exp x)
  in c1 ∧ c2 ∧ c3 ∧ c4 ∧ c5 →
     ([c1, c2, c3, c4, c5].count (λ x, x) = 3) := by
  sorry

end count_correct_derivatives_l729_729643


namespace negation_proposition_l729_729642

theorem negation_proposition : 
  ¬(∀ x : ℝ, 0 ≤ x → 2^x > x^2) ↔ ∃ x : ℝ, 0 ≤ x ∧ 2^x ≤ x^2 := by
  sorry

end negation_proposition_l729_729642


namespace intersection_of_A_and_B_is_empty_l729_729137

definition A : set ℝ := { x : ℝ | x > 1 }
definition B : set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_of_A_and_B_is_empty_l729_729137


namespace find_integer_divisible_by_24_and_cube_root_between_8_and_8_2_l729_729454

theorem find_integer_divisible_by_24_and_cube_root_between_8_and_8_2 : 
  ∃ (n : ℕ), (n > 0) ∧ (n % 24 = 0) ∧ (8 < real.cbrt n ∧ real.cbrt n < 8.2) ∧ n = 528 :=
by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_8_and_8_2_l729_729454


namespace find_sum_a_b_l729_729644

noncomputable def parabola_kite_area (a b : ℝ) : Prop :=
  -- Define the x-intercepts of the parabolas.
  let x_intercepts_y_neg2 := sqrt (2 / a)
  let x_intercepts_y_4 := sqrt (4 / b)
  
  -- Kite area condition.
  let kite_area := 12
  
  -- The diagonals of the kite in terms of x-intercepts and y-intercepts.
  let d1 := 2 * x_intercepts_y_neg2
  let d2 := 6 -- Distance from -2 to 4 is 6
  
  -- Calculate area of the kite.
  (1 / 2) * d1 * d2 = kite_area

-- Define the proof problem.
theorem find_sum_a_b : ∃ (a b : ℝ), parabola_kite_area a b ∧ a + b = 1.5 := 
  sorry

end find_sum_a_b_l729_729644


namespace car_travel_distance_l729_729624

-- Definitions of conditions
def speed_kmph : ℝ := 27 -- 27 kilometers per hour
def time_sec : ℝ := 50 -- 50 seconds

-- Equivalent in Lean 4 for car moving distance in meters
theorem car_travel_distance : (speed_kmph * 1000 / 3600) * time_sec = 375 := by
  sorry

end car_travel_distance_l729_729624


namespace correct_option_l729_729347

theorem correct_option :
  (∀ a : ℝ, a ≠ 0 → (a ^ 0 = 1)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → (a^6 / a^3 = a^2)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → ((a^2)^3 = a^5)) ∧
  ¬(∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / (a + b)^2 + b / (a + b)^2 = a + b)) :=
by {
  sorry
}

end correct_option_l729_729347


namespace puppy_sleep_duration_l729_729769

-- Definitions based on the given conditions
def connor_sleep_hours : ℕ := 6
def luke_sleep_hours : ℕ := connor_sleep_hours + 2
def puppy_sleep_hours : ℕ := 2 * luke_sleep_hours

-- Theorem stating the puppy's sleep duration
theorem puppy_sleep_duration : puppy_sleep_hours = 16 :=
by
  -- ( Proof goes here )
  sorry

end puppy_sleep_duration_l729_729769


namespace sequence_sum_l729_729723

theorem sequence_sum :
  let a : ℕ → ℚ := (λ n, if n = 1 then 2 else if n = 2 then 2 else (2/5) * a (n - 1) + (1/6) * a (n - 2)) in
  let S : ℚ := ∑' n : ℕ, a (n + 1) in
  S = 96 / 13 :=
  sorry

end sequence_sum_l729_729723


namespace eval_expression_l729_729822

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729822


namespace octagon_diagonals_l729_729149

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l729_729149


namespace prob_bob_before_12_45_given_alice_after_bob_l729_729028

theorem prob_bob_before_12_45_given_alice_after_bob :
  let total_area := 60 * 60 / 2 in
  let interested_area := 45 * 45 / 2 in
  (interested_area / total_area = 0.5625) :=
by
  let total_area := 60 * 60 / 2
  let interested_area := 45 * 45 / 2
  have h1 : interested_area / total_area = 0.5625 := by norm_num
  exact h1

end prob_bob_before_12_45_given_alice_after_bob_l729_729028


namespace exists_x0_in_interval_l729_729269

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem exists_x0_in_interval :
  ∃ x0 ∈ Ioo 2 3, f x0 = 0 :=
sorry

end exists_x0_in_interval_l729_729269


namespace sum_of_digits_of_8_pow_351_eq_11_l729_729338

theorem sum_of_digits_of_8_pow_351_eq_11 :
  let a := 8^351 in
  let last_two_digits := a % 100 in
  (last_two_digits / 10) + (last_two_digits % 10) = 11 :=
by
  sorry

end sum_of_digits_of_8_pow_351_eq_11_l729_729338


namespace minimum_lambda_ineq_l729_729067

theorem minimum_lambda_ineq (n : ℕ) (hn : n ≥ 2) (a : Fin n → ℝ)
  (ha_pos : ∀ i, 0 < a i) (ha_sum : (∑ i, a i) = n) :
  ∃ (e : ℝ), 
  (∀ λ : ℝ, (∀ i, 0 < a i) → 
  (∑ i, (1 / a i) - λ * (∏ i, 1 / a i) ≤ n - λ) ↔ λ = Real.exp 1) :=
begin
  sorry
end

end minimum_lambda_ineq_l729_729067


namespace problem_l729_729584

theorem problem
  (M : Set ℝ) (N : Set ℝ)
  (hM : M = {x | x^2 = x})
  (hN : N = {x | log x ≤ 0}) :
  M ∪ N = Set.Icc 0 1 :=
by
  sorry

end problem_l729_729584


namespace decrease_in_profit_due_to_idle_loom_correct_l729_729726

def loom_count : ℕ := 80
def total_sales_value : ℕ := 500000
def monthly_manufacturing_expenses : ℕ := 150000
def establishment_charges : ℕ := 75000
def efficiency_level_idle_loom : ℕ := 100
def sales_per_loom : ℕ := total_sales_value / loom_count
def expenses_per_loom : ℕ := monthly_manufacturing_expenses / loom_count
def profit_contribution_idle_loom : ℕ := sales_per_loom - expenses_per_loom

def decrease_in_profit_due_to_idle_loom : ℕ := 4375

theorem decrease_in_profit_due_to_idle_loom_correct :
  profit_contribution_idle_loom = decrease_in_profit_due_to_idle_loom :=
by sorry

end decrease_in_profit_due_to_idle_loom_correct_l729_729726


namespace arithmetic_mean_of_divisors_is_integer_l729_729217

open Nat

theorem arithmetic_mean_of_divisors_is_integer (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : p ≠ q) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 
  (∃ n = (p ^ a * q ^ b), 
     (∑ d in divisors n, d) / (divisors n).card) ∈ ℕ := by
  sorry

end arithmetic_mean_of_divisors_is_integer_l729_729217


namespace teresa_colored_pencils_l729_729279

-- Definition of the conditions
def black_pencils : ℕ := 35
def keep_pencils : ℕ := 10
def sibling_pencils : ℕ := 13
def siblings : ℕ := 3

-- Theorem stating the problem
theorem teresa_colored_pencils : 
  let total_pencils := siblings * sibling_pencils + keep_pencils in
  let colored_pencils := total_pencils - black_pencils in
  colored_pencils = 14 := 
by sorry

end teresa_colored_pencils_l729_729279


namespace eval_expression_l729_729963

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729963


namespace eval_expr_l729_729814

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729814


namespace find_first_number_in_first_set_l729_729307

-- Given conditions represented as hypotheses
theorem find_first_number_in_first_set :
  ∃ (x : ℝ), 
    (x + 42 + 78 + 104) / 4 = 62 ∧
    (48 + 62 + 98 + 124 + x) / 5 = 78 → 
    x = 24 :=
by
  intro h
  sorry

end find_first_number_in_first_set_l729_729307


namespace triangle_inequality_valid_x_values_l729_729296

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l729_729296


namespace eval_expression_l729_729959

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l729_729959


namespace distance_of_tangent_circles_l729_729173

noncomputable def distance_between_centers (r1 r2 : ℝ) : ℝ :=
  r1 + r2

theorem distance_of_tangent_circles :
  ∀ (r1 r2 : ℝ), r1 = 8 → r2 = 3 → 
  distance_between_centers r1 r2 = 11 :=
begin
  intros r1 r2 hr1 hr2,
  rw [hr1, hr2],
  simp [distance_between_centers],
  norm_num,
end

end distance_of_tangent_circles_l729_729173


namespace general_term_sequence_value_of_n_minimizing_sum_l729_729535

theorem general_term_sequence (a : ℕ → ℤ)
  (h1 : ∀ n ≥ 2, a n - a (n-1) = 3)
  (h2 : a 3 = -10) :
  ∀ n, a n = 3 * n - 19 :=
by
  sorry

theorem value_of_n_minimizing_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n ≥ 2, a n - a (n-1) = 3) 
  (h2 : a 3 = -10) 
  (h3 : ∀ n, S n = ∑ i in range (n + 1), a i) :
  ∃ n, S n = min (S n) := 
by
  sorry

end general_term_sequence_value_of_n_minimizing_sum_l729_729535


namespace probability_no_stable_snow_cover_next_two_years_probability_stable_snow_cover_at_least_once_next_two_years_l729_729646

def probability_stable_snow_cover : ℙ := 0.2
def probability_no_stable_snow_cover : ℙ := 0.8

theorem probability_no_stable_snow_cover_next_two_years :
  (probability_no_stable_snow_cover * probability_no_stable_snow_cover) = 0.64 := by sorry

theorem probability_stable_snow_cover_at_least_once_next_two_years :
  (1 - (probability_no_stable_snow_cover * probability_no_stable_snow_cover)) = 0.36 := by sorry

end probability_no_stable_snow_cover_next_two_years_probability_stable_snow_cover_at_least_once_next_two_years_l729_729646


namespace profit_percent_l729_729396

-- Definitions based on the conditions in the problem
def marked_price_per_pen := ℝ
def total_pens := 52
def cost_equivalent_pens := 46
def discount_percentage := 1 / 100

-- Values calculated from conditions
def cost_price (P : ℝ) := cost_equivalent_pens * P
def selling_price_per_pen (P : ℝ) := P * (1 - discount_percentage)
def total_selling_price (P : ℝ) := total_pens * selling_price_per_pen P

-- The proof statement
theorem profit_percent (P : ℝ) (hP : P > 0) :
  ((total_selling_price P - cost_price P) / (cost_price P)) * 100 = 11.91 := by
    sorry

end profit_percent_l729_729396


namespace greatest_number_of_unit_segments_l729_729738

-- Define the conditions
def is_equilateral (n : ℕ) : Prop := n > 0

-- Define the theorem
theorem greatest_number_of_unit_segments (n : ℕ) (h : is_equilateral n) : 
  -- Prove the greatest number of unit segments such that no three of them form a single triangle
  ∃(m : ℕ), m = n * (n + 1) := 
sorry

end greatest_number_of_unit_segments_l729_729738


namespace correct_multiplication_l729_729410

theorem correct_multiplication :
  ∃ (n : ℕ), 98765 * n = 888885 ∧ (98765 * n = 867559827931 → n = 9) :=
by
  sorry

end correct_multiplication_l729_729410


namespace eval_expr_l729_729803

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729803


namespace probability_three_heads_in_eight_tosses_l729_729006

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729006


namespace probability_of_selecting_a_l729_729255

noncomputable def probability_of_condition : ℝ :=
  let interval_total_length := 10
  let valid_interval_length := 3
  valid_interval_length / interval_total_length

theorem probability_of_selecting_a :
  (let a ∈ Icc (-5 : ℝ) 5 → (1 : ℝ) ∈ {x | 2 * x^2 + a * x - a^2 > 0}) →
  probability_of_condition = 0.3 :=
by
  sorry

end probability_of_selecting_a_l729_729255


namespace point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l729_729093

-- Question (1): Proving that the point (-2,0) lies on the graph
theorem point_on_graph (k : ℝ) (hk : k ≠ 0) : k * (-2 + 2) = 0 := 
by sorry

-- Question (2): Finding the value of k given a shifted graph passing through a point
theorem find_k_shifted_graph_passing (k : ℝ) : (k * (1 + 2) + 2 = -2) → k = -4/3 := 
by sorry

-- Question (3): Proving the range of k for the function's y-intercept within given limits
theorem y_axis_intercept_range (k : ℝ) (hk : -2 < 2 * k ∧ 2 * k < 0) : -1 < k ∧ k < 0 := 
by sorry

end point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l729_729093


namespace curve_equation_slope_range_u_range_l729_729488

/-- Given points A and B and curve C with the property that the dot product of vectors AP and BP is -3,
prove that the equation of the curve C is x^2 + y^2 = 1. --/
theorem curve_equation :
  ∀ (x y : ℝ), ((x + 2) * (x - 2) + y * y = -3) ↔ (x^2 + y^2 = 1) :=
by
  sorry

/-- Given a fixed point M(0, -2) and a line l passing through M and intersects curve C,
prove that the range of the slope k is (-∞, -√3] ∪ [√3, +∞). --/
theorem slope_range :
  ∀ (k : ℝ), ((∃ (x y : ℝ), (y = k * x - 2) ∧ (x^2 + y^2 = 1))) ↔ ((k ∈ Iic (-real.sqrt 3)) ∨ (k ∈ Ici (real.sqrt 3))) :=
by
  sorry

/-- Given a moving point Q on curve C and calculating u = (y + 2) / (x - 1),
prove that the range of u is (-∞, -3/4]. --/
theorem u_range :
  ∀ (x y : ℝ), ((x^2 + y^2 = 1) → ((y + 2) / (x - 1) ∈ (Iic (-3/4)))) :=
by
  sorry

end curve_equation_slope_range_u_range_l729_729488


namespace find_ellipse_equation_verify_dot_product_constant_l729_729716

noncomputable def M := (Real.sqrt 2, 1 : ℝ × ℝ)

def is_point_on_ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def sum_distances_to_foci (x y a b : ℝ) : ℝ :=
  Real.sqrt ((x - a)^2 + y^2) + Real.sqrt ((x + a)^2 + y^2)

def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (3 * y^2 / a^2) = 1

theorem find_ellipse_equation : 
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  is_point_on_ellipse (Real.sqrt 2) 1 a b ∧
  sum_distances_to_foci (Real.sqrt 2) 1 a b = 2 * Real.sqrt 5 ∧
  ellipse_equation a b :=
sorry

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def intersects_ellipse (k a b : ℝ) : Prop :=
  let line (x : ℝ) := k * (x + 1) in
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  line x₁ = y₁ ∧ line x₂ = y₂ ∧
  is_point_on_ellipse x₁ y₁ a b ∧ 
  is_point_on_ellipse x₂ y₂ a b ∧
  dot_product ((x₁ - (-⁷/₃)), y₁) ((x₂ - (-⁷/₃)), y₂) = 4 / 9

theorem verify_dot_product_constant : 
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
  is_point_on_ellipse (Real.sqrt 2) 1 a b ∧ 
  sum_distances_to_foci (Real.sqrt 2) 1 a b = 2 * Real.sqrt 5 →
  ∀ k : ℝ, intersects_ellipse k a b :=
sorry

end find_ellipse_equation_verify_dot_product_constant_l729_729716


namespace subset_condition_m_eq_1_l729_729136

theorem subset_condition_m_eq_1 (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {3, m ^ 2}) 
  (hB : B = {-1, 3, 2 * m - 1}) 
  (hSub : A ⊆ B) : m = 1 :=
begin
  sorry
end

end subset_condition_m_eq_1_l729_729136


namespace evaluation_of_expression_l729_729889

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l729_729889


namespace max_isosceles_good_sides_l729_729571

def is_good_diagonal (n : ℕ) (i j : ℕ) : Prop :=
  (i ≠ j) ∧ ((even (j - i) ∧ odd (i + n - j)) ∨ (odd (j - i) ∧ even (i + n - j)))

def is_good_side (n : ℕ) (i : ℕ) : Prop :=
  true -- sides are always good

noncomputable def P := 2006

theorem max_isosceles_good_sides :
  ∃ (triangulation : list (ℕ × ℕ × ℕ)), ∃ T,
      set.to_finset { t | set.subset t (set.filter (λ d, is_good_diagonal P d.1 d.2 ∨ is_good_side P d.1) T) }
      ∧ triangulation.card = 2004
      ∧ ∃ (isosceles : finset (ℕ × ℕ × ℕ)),
        ∀ triangle ∈ isosceles, is_good_diagonal P triangle.1 triangle.2 ∧ is_good_diagonal P triangle.1 triangle.3 
        ∧ isosceles.card = 1003
:= by
  sorry

end max_isosceles_good_sides_l729_729571


namespace find_x_eq_e_l729_729700

noncomputable def f (x : ℝ) : ℝ := x + x * (Real.log x) ^ 2

noncomputable def f' (x : ℝ) : ℝ :=
  1 + (Real.log x) ^ 2 + 2 * Real.log x

theorem find_x_eq_e : ∃ (x : ℝ), (x * f' x = 2 * f x) ∧ (x = Real.exp 1) :=
by
  sorry

end find_x_eq_e_l729_729700


namespace nice_people_count_l729_729729

/-- Variables representing the number of people. --/
variables (Barry Kevin Julie Joe Alex Lauren Chris Taylor Morgan Casey : ℕ)

definition total_nice_people (Barry Kevin Julie Joe Alex Lauren Chris Taylor Morgan Casey : ℕ) : ℕ :=
  let nice_Barry := 1 * Barry in
  let nice_Kevin := 45 * Kevin / 100 in
  let nice_Julie := 3 * Julie / 5 in
  let nice_Joe := 1 * Joe / 8 in
  let nice_Alex := 7 * Alex / 8 in
  let nice_Lauren := 5 * Lauren / 9 in
  let nice_Chris := 3 * Chris / 8 in
  let nice_Taylor := 37 * Taylor / 40 in
  let nice_Morgan := 27 * Morgan / 35 in
  let nice_Casey := 4 * Casey / 7 in
  nice_Barry + nice_Kevin + nice_Julie + nice_Joe + nice_Alex + nice_Lauren + nice_Chris + nice_Taylor + nice_Morgan + nice_Casey

theorem nice_people_count :
  total_nice_people 70 60 300 180 220 135 120 150 105 140 = 913 :=
by norm_num
-- sorry, replace "by norm_num" with "sorry" to skip the proof

end nice_people_count_l729_729729


namespace problem_1_l729_729250

theorem problem_1 
  (P_moving_on_C1 : ∀ P : ℝ × ℝ, (P.1 - 2)^2 + P.2^2 = 4)
  (P_rotated_to_Q : ∀ P Q : ℝ × ℝ, 
    let θ_P := real.atan2 P.2 (P.1 - 2),
        ρ_P := (P.1 - 2)^2 + P.2^2,
        θ_Q := θ_P + π / 2 in
    Q = (ρ_P * real.cos θ_Q, ρ_P * real.sin θ_Q))
  (M : ℝ × ℝ := (2, 0))
  (A B : ℝ × ℝ)
  (θ : ℝ := π / 3)
  (intersection_A : A = (4 * real.cos θ, θ))
  (intersection_B : B = (4 * real.sin θ, θ)) :
  (∀ θ ρ, (ρ = 4 * real.cos θ) → (ρ = 4 * real.sin (θ - π / 2)) → true) ∧
  (let d := 2 * real.sin (θ / 2),
       AB := 4 * (real.sin θ - real.cos θ),
       area := 1 / 2 * AB * d in
    area = 3 - real.sqrt 3) := 
  begin
    sorry, -- proofs needed here
  end

end problem_1_l729_729250


namespace crayons_total_l729_729181

noncomputable def totalCrayons (blue : ℕ) (red : ℕ) : ℕ := blue + red

theorem crayons_total (h₁ : ∃ b : ℕ, b = 3)
                     (h₂ : ∀ b : ℕ, ∃ r : ℕ, r = 4 * b) :
  ∃ t : ℕ, t = 15 := by
  obtain ⟨b, hb⟩ := h₁
  obtain ⟨r, hr⟩ := h₂ b
  have total := totalCrayons b r
  rw [hb, hr, totalCrayons]
  exact ⟨15, by norm_num⟩

end crayons_total_l729_729181


namespace union_A_B_l729_729490

-- Define them as sets
def A : Set ℝ := {x | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Statement of the theorem
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l729_729490


namespace num_integer_terms_in_sequence_l729_729310

theorem num_integer_terms_in_sequence : 
  ∃ n, (n = 6) ∧ (∀ m < n, ∃ k, 9720 / (3^m) = k ∧ k ∈ ℤ) ∧ (∃ m > n, ∀ k, 9720 / (3^m) ≠ k) := 
sorry

end num_integer_terms_in_sequence_l729_729310


namespace classify_bundles_l729_729605

-- Definitions based on the conditions

-- Condition for Elliptical Bundle
def elliptical_bundle (circles : Set Circle) : Prop :=
  ∀ c ∈ circles, c.intersects_radical_axis_two_fixed_points ∧ c.radius > 0

-- Condition for Parabolic Bundle
def parabolic_bundle (circles : Set Circle) : Prop :=
  ∃ p, ∀ c ∈ circles, c.tangent_to_radical_axis_at p

-- Condition for Hyperbolic Bundle
def hyperbolic_bundle (circles : Set Circle) : Prop :=
  ∃ k > 0, ∀ c ∈ circles, c.power_of_point > k

-- Limiting points
def limiting_points (circles : Set Circle) : Set Point :=
  { p | ∃ c ∈ circles, c.radius = 0 ∧ c.center = p }

-- Main theorem statement
theorem classify_bundles (circles : Set Circle) : 
  (elliptical_bundle circles → limiting_points circles = ∅) ∧
  (parabolic_bundle circles → ∃ p, limiting_points circles = {p}) ∧
  (hyperbolic_bundle circles → ∃ p1 p2, p1 ≠ p2 ∧ limiting_points circles = {p1, p2}) :=
by
  sorry

end classify_bundles_l729_729605


namespace eval_expression_l729_729820

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729820


namespace exists_vector_not_in_any_L_of_A_l729_729569

open Classical

variables (H : Type) [HilbertSpace H] (A : Set (Subspace ℝ H))

def TotalOrderWithRespectToInclusion : Prop :=
  ∀ L1 L2 ∈ A, L1 ⊆ L2 ∨ L2 ⊆ L1

theorem exists_vector_not_in_any_L_of_A
  (hA_nonempty : A.Nonempty)
  (hA_proper : ∀ L ∈ A, L ≠ ⊤)
  (hA_total_order : TotalOrderWithRespectToInclusion H A) :
  ∃ (x : H), ∀ L ∈ A, x ∉ L :=
by
  sorry

end exists_vector_not_in_any_L_of_A_l729_729569


namespace solve_for_ab_l729_729344

def f (a b : ℚ) (x : ℚ) : ℚ := a * x^3 - 4 * x^2 + b * x - 3

theorem solve_for_ab : 
  ∃ a b : ℚ, 
    f a b 1 = 3 ∧ 
    f a b (-2) = -47 ∧ 
    (a, b) = (4 / 3, 26 / 3) := 
by
  sorry

end solve_for_ab_l729_729344


namespace cube_labelings_count_l729_729445

-- We need to define a cube with labeled edges. In Lean, we can represent the labels using a list, vector or similar structure.
-- Define a structure for a cube and what it means to have labels 0 or 1 on edges such that each face sums to 3.

structure CubeLabeling where
  labels : Vector (Fin 2) 12 -- each edge is labeled with a 0 or 1

def valid_face_sum (cube : CubeLabeling) : Prop :=
  ∀ face ∈ cube_faces, (∑ edge in face, cube.labels[edge]) = 3
  where 
    cube_faces : List (List Nat) := 
      [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 4, 8, 5], [1, 5, 9, 6], [2, 6, 10, 7], [3, 7, 11, 4]]

def valid_cube_labelings : List CubeLabeling :=
  -- here we would generate all valid configuration
  sorry

theorem cube_labelings_count : 
  valid_cube_labelings.length = 8 := 
sorry

end cube_labelings_count_l729_729445


namespace triangle_inequality_valid_x_values_l729_729298

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l729_729298


namespace eval_expr_l729_729810

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end eval_expr_l729_729810


namespace evaluate_expression_l729_729798

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end evaluate_expression_l729_729798


namespace eval_expression_l729_729833

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l729_729833


namespace sin_cos_value_l729_729119

-- Definitions based on conditions:
def theta (k : ℝ) (h : k < 0) : ℝ := 
  let x := -4 * k
  let y := 3 * k
  atan2 y x

def sin_cos_expr (k : ℝ) (h : k < 0) : ℝ :=
  let θ := theta k h
  2 * Real.sin θ + Real.cos θ

-- Proof statement:
theorem sin_cos_value (k : ℝ) (h : k < 0) : sin_cos_expr k h = - (2 / 5) :=
sorry

end sin_cos_value_l729_729119


namespace evaluate_expression_l729_729852

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l729_729852


namespace calc_expression_l729_729040

theorem calc_expression :
  (∛(-27 / 8) + sqrt 16 - abs (sqrt 2 - 1) + sqrt ((-1 / 2) ^ 2) = 4 - sqrt 2) :=
by sorry

end calc_expression_l729_729040


namespace solve_cubic_system_l729_729183

noncomputable def cubic_polynomial (a b c : ℝ) : ℝ → ℝ :=
  λ x, x^3 + a * x^2 + b * x + c

theorem solve_cubic_system (a b c : ℝ) (x y : ℝ) (h1 : x = cubic_polynomial a b c y)
  (h2 : y = cubic_polynomial a b c x) :
  ∃ s t : ℝ, s = x + y ∧ t = x * y ∧ (x, y) ∈ {z : ℝ × ℝ | ∃ s t : ℝ, z = (s + sqrt (s^2 - 4 * t)) / 2 ∨ z = (s - sqrt (s^2 - 4 * t)) / 2} :=
sorry

end solve_cubic_system_l729_729183


namespace cost_of_milk_l729_729475

theorem cost_of_milk (x : ℝ) (h1 : 10 * 0.1 = 1) (h2 : 11 = 1 + x + 3 * x) : x = 2.5 :=
by 
  sorry

end cost_of_milk_l729_729475


namespace prove_card_A_l729_729466

def Card : Type := 
  | A
  | S
  | Five
  | Eight
  | Seven

def isOdd (n : ℕ) : Prop := 
  n % 2 = 1

def isVowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def isConsonant (c : Char) : Prop := 
  ¬isVowel c

def reverseCard : Card → ℕ ⊕ Char
  | Card.A := Sum.inl 5
  | Card.S := Sum.inl 8
  | Card.Five := Sum.inr 'A'
  | Card.Eight := Sum.inr 'S'
  | Card.Seven := Sum.inr 'O'

theorem prove_card_A : (∃ c, reverseCard c = Sum.inr 'A') →
  reverseCard (Card.A) = Sum.inl 5 :=
begin
  intro h,
  assumption,
sorry -- Proof here

end prove_card_A_l729_729466


namespace count_powers_of_2_not_4_l729_729165

theorem count_powers_of_2_not_4 (n : ℕ) (h : n = 500000) : 
  (∑ k in finset.range 20, ite ((¬ (∃ m, 2 ^ (2 * m) = 2 ^ k)) ∧ (2 ^ k < n)) 1 0) = 9 := 
by
  sorry

end count_powers_of_2_not_4_l729_729165


namespace max_minus_min_l729_729997

variable {a b : ℝ}

theorem max_minus_min {a b : ℝ} (h : a^2 + a * b + b^2 = 6) : 
  let t := a^2 - a * b + b^2
  let M := 12
  let m := -3
  in M - m = 15 := 
by
  -- detailed proof goes here
  sorry

end max_minus_min_l729_729997


namespace min_c_value_l729_729240

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (unique_sol : ∃ ! x, ∃ y, 2 * x + y = 2037 ∧ y = |x - a| + |x - b| + |x - c|) : c = 1019 :=
sorry

end min_c_value_l729_729240


namespace travel_speed_is_four_l729_729562
-- Import the required library

-- Define the conditions
def jacksSpeed (x : ℝ) : ℝ := x^2 - 13 * x - 26
def jillsDistance (x : ℝ) : ℝ := x^2 - 5 * x - 66
def jillsTime (x : ℝ) : ℝ := x + 8

-- Prove the equivalent statement
theorem travel_speed_is_four (x : ℝ) (h : x = 15) :
  jillsDistance x / jillsTime x = 4 ∧ jacksSpeed x = 4 := 
by sorry

end travel_speed_is_four_l729_729562


namespace reservoir_ratio_proof_l729_729752

variable total_capacity normal_level end_of_month_capacity : ℝ
variable h1 : end_of_month_capacity = 14 -- 14 million gallons
variable h2 : end_of_month_capacity = 0.70 * total_capacity
variable h3 : normal_level = total_capacity - 10 -- 10 million gallons

theorem reservoir_ratio_proof :
  end_of_month_capacity / normal_level = 1.4 := by
  sorry

end reservoir_ratio_proof_l729_729752


namespace probability_three_heads_in_eight_tosses_l729_729010

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l729_729010


namespace tangent_line_at_neg2_l729_729366

noncomputable def y (x : ℝ) : ℝ := x / (x^2 + 1)
noncomputable def y' (x : ℝ) : ℝ := (1 - x^2) / (x^2 + 1)^2

theorem tangent_line_at_neg2 : 
    y = λ x, x / (x^2 + 1) ∧
    y' = λ x, (1 - x^2) / (x^2 + 1)^2 ∧
    y' (-2) = -3 / 25 ∧
    y (-2) = -2 / 5 →
    ∃ (m b : ℝ), (m = -3 / 25 ∧ b = -16 / 25) ∧ ∀ (x : ℝ),
    y = λ x, m * x + b :=
by
  sorry

end tangent_line_at_neg2_l729_729366
