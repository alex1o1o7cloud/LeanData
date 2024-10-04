import Mathlib

namespace circle_area_l351_351100

theorem circle_area (d : ℝ) (r : ℝ) (A : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : A = π * r^2) : 
  A = 25 * π :=
by
  -- Introduce variables and assumptions
  have h_radius : r = 5 := by
    rw [h2, h1]
    norm_num
  -- Introduce the calculation of area
  have h_area : A = π * (5^2) := by
    rw [h3, h_radius]
  -- Simplify the expression
  rw [h_area]
  norm_num  
  rfl

end circle_area_l351_351100


namespace prime_ratio_l351_351722

def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def median (a b c : ℕ) : ℕ := b

theorem prime_ratio :
  ∃ b c : ℕ, is_prime b ∧ is_prime c ∧ 2 < b ∧ b < c ∧ b ≠ 3 ∧ ((2 + b + c) / 3 = 6 * median 2 b c) ∧ c / b = 83 / 5 := 
by
  sorry

end prime_ratio_l351_351722


namespace solve_equation_l351_351716

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l351_351716


namespace problem_1_problem_2_l351_351078

/-- Circular Card Passing Game Setup -/
structure card_game (n : ℕ) :=
  (students : ℕ)
  (initial_cards : ℕ)
  (students_count : students = 1994)
  (cards_count : initial_cards = n)
  
def can_continue_indefinitely : Prop :=
  ∀ (g : card_game n), n ≥ 1994 → ∃ k, k > 0 ∧ ∃ s : ℕ → ℕ, (s k > 1)

def must_end_eventually : Prop :=
  ∀ (g : card_game n), n < 1994 → ∃ k, k > 0 ∧ ∀ s : ℕ → ℕ, ∀ j, j > k → s j ≤ 1

theorem problem_1 (n : ℕ) : n ≥ 1994 → can_continue_indefinitely :=
  sorry

theorem problem_2 (n : ℕ) : n < 1994 → must_end_eventually :=
  sorry

end problem_1_problem_2_l351_351078


namespace sum_coordinates_A_l351_351598

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351598


namespace last_two_digits_sum_factorials_l351_351421

theorem last_two_digits_sum_factorials : 
  (Finset.sum (Finset.range 2003) (λ n, n.factorial) % 100) = 13 := 
by 
  sorry

end last_two_digits_sum_factorials_l351_351421


namespace cos_330_eq_sqrt3_div_2_l351_351272

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351272


namespace triangle_smallest_angle_l351_351525

noncomputable def smallest_angle (a b c: ℝ) (ha: a = 7) (hb: b = 4 * Real.sqrt 3) (hc: c = Real.sqrt 13) : ℝ :=
    let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
    Real.acos cos_C

theorem triangle_smallest_angle (a b c: ℝ) (ha: a = 7) (hb: b = 4 * Real.sqrt 3) (hc: c = Real.sqrt 13) :
    smallest_angle a b c ha hb hc = π / 6 :=
sorry

end triangle_smallest_angle_l351_351525


namespace has_no_common_side_l351_351091

def can_create_colored_grid (cards : ℕ → ℕ) : Prop :=
  ∀ (arrangement : Fin (10 * 10) → ℕ),
    (∀ n, n < 3 → ∑ i in Fin (10 * 10), if arrangement i = n then 1 else 0 = cards n) →
    (∀ i j, adjacent i j → arrangement i ≠ arrangement j)

theorem has_no_common_side (cards : Nat → Nat) (h1 : cards 0 + cards 1 + cards 2 = 100)
    (h2 : ∀ n, n < 3 → cards n ≤ 50) :
  can_create_colored_grid cards :=
sorry

end has_no_common_side_l351_351091


namespace relationship_y1_y2_l351_351466

theorem relationship_y1_y2 (a y1 y2 m n : ℝ) 
  (hA : y1 = (m^2 + 1) * (2 * a - 1) + 2 * n)
  (hB : y2 = (m^2 + 1) * (a^2 + 1) + 2 * n) :
  y1 < y2 :=
by {
  have pos_slope : (m^2 + 1) > 0,
  { 
    have sq_nonneg : m^2 ≥ 0 := sq_nonneg m, 
    linarith,
  },

  have ha2_pos : (a^2 - 2 * a + 2) > 0,
  { 
    have sq_nonneg : (a - 1)^2 ≥ 0 := sq_nonneg (a - 1),
    linarith,
  },

  have x_diff : (a^2 + 1) > (2 * a - 1) := by linarith,
  
  have y_diff: y2 = (m^2 + 1) * (a^2 + 1) + 2 * n,
    rw [hB, hA],
    linarith, sorry
}

end relationship_y1_y2_l351_351466


namespace gcf_75_100_l351_351818

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351818


namespace where_they_meet_l351_351158

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

end where_they_meet_l351_351158


namespace division_addition_problem_l351_351953

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l351_351953


namespace frog_jump_positions_l351_351659

theorem frog_jump_positions
  (p q : ℕ) (coprime : Nat.gcd p q = 1)
  (returns_to_zero : ∃ N : ℕ, (Σ (i : ℕ), p * i - q * (N - i) = 0)) :
  ∀ d : ℕ, d < p + q → ∃ (i j : ℕ), i ≠ j ∧ |(λ k, if k % 2 = 0 then k / 2 * p else k / 2 * q) i - (λ k, if k % 2 = 0 then k / 2 * p else k / 2 * q) j| = d :=
by
  sorry

end frog_jump_positions_l351_351659


namespace cos_330_eq_sqrt3_div_2_l351_351256

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351256


namespace area_PQR_l351_351084

noncomputable def area_of_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem area_PQR : 
  let P := (2 : ℝ, 1 : ℝ)
  let Q := (5 : ℝ, 4 : ℝ)
  let R := (4 : ℝ, 0 : ℝ)
  line_eq := ∀ (x y : ℝ), (x + y = 4) → (R.1 + R.2 = 4) 
  area_of_triangle P Q R = 4.5 :=
by
  sorry

end area_PQR_l351_351084


namespace amy_owes_thirty_l351_351029

variable (A D : ℝ)

theorem amy_owes_thirty
  (total_pledged remaining_owed sally_carl_owe derek_half_amys_owes : ℝ)
  (h1 : total_pledged = 285)
  (h2 : remaining_owed = 400 - total_pledged)
  (h3 : sally_carl_owe = 35 + 35)
  (h4 : derek_half_amys_owes = A / 2)
  (h5 : remaining_owed - sally_carl_owe = 45)
  (h6 : 45 = A + (A / 2)) :
  A = 30 :=
by
  -- Proof steps skipped
  sorry

end amy_owes_thirty_l351_351029


namespace parabola_vertex_point_l351_351721

theorem parabola_vertex_point (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c → 
  ∃ k : ℝ, ∃ h : ℝ, y = a * (x - h)^2 + k ∧ h = 2 ∧ k = -1 ∧ 
  (∃ y₀ : ℝ, 7 = a * (0 - h)^2 + k) ∧ y₀ = 7) 
  → (a = 2 ∧ b = -8 ∧ c = 7) := by
  sorry

end parabola_vertex_point_l351_351721


namespace distance_between_ellipse_foci_l351_351919

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351919


namespace similarity_of_triangles_l351_351526

-- Definitions of points as complex numbers
variables (z1 z2 z3 t1 t2 t3 z1' z2' z3' : ℂ)

-- Given similarity conditions for constructed triangles
variables (h1 : t2 - z1' = (z3 - t1) * (z2 - z3)⁻¹ * (t2 - z1'))
variables (h2 : t3 - z1' = (z2 - t1) * (z3 - z1) * (t3 - z1'))
variables (h3 : t1 * z1 = (t2 * z2 - t3 * z3 - t1 * t2 + t1 * t3)⁻¹ * z1')

theorem similarity_of_triangles :
  (z2' - z1') / (z3' - z1') = (z2 - z1) / (z3 - z1) → 
  ((z1' z1' ∙ z2' z2' ∙ z3' z3' ∙ 0) ∼ (z1 z1 ∙ z2 z2 ∙ z3 z3 ∙ 0)) :=
by sorry

end similarity_of_triangles_l351_351526


namespace cream_ratio_l351_351635

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l351_351635


namespace cos_330_eq_sqrt3_div_2_l351_351330

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351330


namespace min_questionnaires_l351_351048

theorem min_questionnaires 
  (resp_rate_A : ℝ) (resp_rate_B : ℝ) (resp_rate_C : ℝ)
  (min_responses : ℕ) :
  resp_rate_A = 0.6 →
  resp_rate_B = 0.75 →
  resp_rate_C = 0.80 →
  let x := ⌈min_responses / resp_rate_A⌉ in
  let y := ⌈min_responses / resp_rate_B⌉ in
  let z := min_responses / resp_rate_C in
  x = 167 ∧ y = 134 ∧ z = 125 ∧ x + y + z = 426 :=
begin
  intros,
  sorry
end

end min_questionnaires_l351_351048


namespace farmer_tomatoes_left_l351_351126

theorem farmer_tomatoes_left 
  (initial_tomatoes : ℕ)
  (picked_yesterday : ℕ)
  (picked_today : ℕ) :
  initial_tomatoes = 171 →
  picked_yesterday = 134 →
  picked_today = 30 →
  initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
  intros h_initial h_yesterday h_today
  rw [h_initial, h_yesterday, h_today]
  norm_num
  sorry

end farmer_tomatoes_left_l351_351126


namespace geometric_sequence_value_a6_l351_351539

theorem geometric_sequence_value_a6
    (q a1 : ℝ) (a : ℕ → ℝ)
    (h1 : ∀ n, a n = a1 * q ^ (n - 1))
    (h2 : a 2 = 1)
    (h3 : a 8 = a 6 + 2 * a 4)
    (h4 : q > 0)
    (h5 : ∀ n, a n > 0) : 
    a 6 = 4 :=
by
  sorry

end geometric_sequence_value_a6_l351_351539


namespace cos_330_eq_sqrt3_over_2_l351_351212

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351212


namespace domain_sqrt_function_domain_fractional_function_l351_351987

-- Problem (1): Domain of y = sqrt(2x + 1) + sqrt(3 - 4x)
theorem domain_sqrt_function (x : ℝ) : 
  (-1 / 2 ≤ x ∧ x ≤ 3 / 4) ↔ (0 ≤ 2 * x + 1 ∧ 0 ≤ 3 - 4 * x) := 
begin
  sorry
end

-- Problem (2): Domain of y = 1 / (|x + 2| - 1)
theorem domain_fractional_function (x : ℝ) :
  (x ≠ -1 ∧ x ≠ -3) ↔ (| x + 2 | ≠ 1) := 
begin
  sorry
end

end domain_sqrt_function_domain_fractional_function_l351_351987


namespace merchant_profit_percentage_l351_351885

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

end merchant_profit_percentage_l351_351885


namespace evaluate_log_base_4_l351_351975

def log_base_4_256_minus_log_base_4_16 : Prop :=
  let a := Real.log 256 / Real.log 4 in
  let b := Real.log 16 / Real.log 4 in
  a - b = 2

theorem evaluate_log_base_4 : log_base_4_256_minus_log_base_4_16 :=
by
  sorry

end evaluate_log_base_4_l351_351975


namespace cos_330_cos_30_val_answer_l351_351228

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351228


namespace find_coordinates_l351_351726

open Real

theorem find_coordinates (x y : ℝ) : 
  (x^2 + y^2 = 289) → ((x - 16)^2 + y^2 = 289) → (x = 8 ∧ (y = 15 ∨ y = -15)) := by
  sorry

end find_coordinates_l351_351726


namespace cream_ratio_l351_351631

-- Define the initial conditions for Joe and JoAnn
def initial_coffee : ℕ := 15
def initial_cup_size : ℕ := 20
def cream_added : ℕ := 3
def coffee_drank_by_joe : ℕ := 3
def mixture_stirred_by_joann : ℕ := 3

-- Define the resulting amounts of cream in Joe and JoAnn's coffee
def cream_in_joe : ℕ := cream_added
def cream_in_joann : ℝ := cream_added - (cream_added * (mixture_stirred_by_joann / (initial_coffee + cream_added)))

-- Prove the ratio of the amount of cream in Joe's coffee to that in JoAnn's coffee
theorem cream_ratio :
  (cream_in_joe : ℝ) / cream_in_joann = 6 / 5 :=
by
  -- The code is just a statement; the proof detail is omitted with sorry, and variables are straightforward math.
  sorry

end cream_ratio_l351_351631


namespace Marley_fruits_total_is_31_l351_351672

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l351_351672


namespace petya_password_count_l351_351695

theorem petya_password_count :
  let digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  in (∑ d1 in digits, ∑ d2 in digits, ∑ d3 in digits, ∑ d4 in digits,
        (if d1 = d2 ∨ d1 = d3 ∨ d1 = d4 ∨ d2 = d3 ∨ d2 = d4 ∨ d3 = d4 then 1 else 0))
     = 3537 :=
by
  sorry

end petya_password_count_l351_351695


namespace product_sets_not_identical_l351_351807

theorem product_sets_not_identical :
  ∀ (M : array (array nat 10) 10),
  (∀ i j, 101 ≤ M[i][j] ∧ M[i][j] ≤ 200) →
  let row_products := array.map (array.foldr (*) 1) M in
  let col_products := array.map (array.foldr (*) 1) (array.transpose M) in
  row_products ≠ col_products :=
by
  sorry

end product_sets_not_identical_l351_351807


namespace required_hours_in_seventh_week_l351_351079

noncomputable theory
open_locale classical

def hoursInWeeks := [10, 13, 9, 14, 8, 0]

theorem required_hours_in_seventh_week (x : ℕ) :
  (list.sum hoursInWeeks + x) / 7 = 12 → x = 30 :=
by {
  assume h,
  sorry
}

end required_hours_in_seventh_week_l351_351079


namespace typing_problem_l351_351514

theorem typing_problem (a b m n : ℕ) (h1 : 60 = a * b) (h2 : 540 = 75 * n) (h3 : n = 3 * m) :
  a = 25 :=
by {
  -- sorry placeholder where the proof would go
  sorry
}

end typing_problem_l351_351514


namespace product_mod_7_l351_351745

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351745


namespace sum_of_cubes_of_roots_eq_302_l351_351955

-- Define the polynomial and roots
noncomputable def poly := (y : ℝ) ^ 3 - 8 * (y : ℝ) ^ 2 + 9 * (y : ℝ) - 2

-- Define the roots r, s, t which satisfy Vieta's formulas
variables (r s t : ℝ)

-- All roots are real and nonnegative
def roots_real_nonnegative : Prop := 
  (poly r = 0 ∧ poly s = 0 ∧ poly t = 0) ∧ 
  r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ 
  r + s + t = 8 ∧ 
  r * s + s * t + t * r = 9 ∧ 
  r * s * t = 2

theorem sum_of_cubes_of_roots_eq_302 
  (h : roots_real_nonnegative r s t) : 
  r^3 + s^3 + t^3 = 302 := 
sorry

end sum_of_cubes_of_roots_eq_302_l351_351955


namespace calculate_delta_nabla_l351_351441

-- Define the operations Δ and ∇
def delta (a b : ℤ) : ℤ := 3 * a + 2 * b
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Formalize the theorem
theorem calculate_delta_nabla : delta 3 (nabla 2 1) = 23 := 
by 
  -- Placeholder for proof, not required by the question
  sorry

end calculate_delta_nabla_l351_351441


namespace sum_first_10_common_elements_in_ap_gp_l351_351434

/-- To find the sum of the first 10 elements that appear in both the arithmetic progression (AP) 
  {5, 8, 11, 14, ...} and the geometric progression (GP) {10, 20, 40, 80, ...}, we follow these steps:
-/
theorem sum_first_10_common_elements_in_ap_gp 
  (a_n : ℕ → ℕ := λ n, 5 + 3 * n)
  (b_k : ℕ → ℕ := λ k, 10 * 2^k)
  (common_elements : ℕ → ℕ := λ m, 20 * 4^m) :
  (Finset.range 10).sum (λ i, common_elements i) = 6990500 := 
by
  -- Set up the common_elements based on the given progressions
  -- Calculate the sum of the first 10 terms of the geometric progression
  sorry

end sum_first_10_common_elements_in_ap_gp_l351_351434


namespace cos_330_eq_sqrt3_div_2_l351_351285

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351285


namespace ellipse_foci_distance_l351_351911

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351911


namespace min_value_frac_sum_l351_351519

theorem min_value_frac_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1): 
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_frac_sum_l351_351519


namespace parking_spaces_remaining_l351_351791

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l351_351791


namespace faye_money_left_is_30_l351_351979

-- Definitions and conditions
def initial_money : ℝ := 20
def mother_gave (initial : ℝ) : ℝ := 2 * initial
def cost_of_cupcakes : ℝ := 10 * 1.5
def cost_of_cookies : ℝ := 5 * 3

-- Calculate the total money Faye has left
def total_money_left (initial : ℝ) (mother_gave_ : ℝ) (cost_cupcakes : ℝ) (cost_cookies : ℝ) : ℝ :=
  initial + mother_gave_ - (cost_cupcakes + cost_cookies)

-- Theorem stating the money left
theorem faye_money_left_is_30 :
  total_money_left initial_money (mother_gave initial_money) cost_of_cupcakes cost_of_cookies = 30 :=
by sorry

end faye_money_left_is_30_l351_351979


namespace complex_problem_l351_351853

noncomputable def z : ℂ := -((1 - complex.i) / complex.sqrt 2)

theorem complex_problem : z ^ 2016 + z ^ 50 - 1 = -complex.i := by
  sorry

end complex_problem_l351_351853


namespace cos_330_cos_30_val_answer_l351_351238

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351238


namespace enrique_commission_l351_351402

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l351_351402


namespace evaluates_all_true_l351_351964

variables (p q r : Prop)

def statement1 : Prop := p ∧ q ∧ r
def statement2 : Prop := ¬p ∧ q ∧ ¬r
def statement3 : Prop := p ∧ ¬q ∧ ¬r
def statement4 : Prop := ¬p ∧ ¬q ∧ r

def implication (s : Prop) : Prop := s → ((p ∧ q) → r)

def evaluates_to_true (s : Prop) : Prop := implication s

theorem evaluates_all_true : 
  evaluates_to_true statement1 ∧
  evaluates_to_true statement2 ∧
  evaluates_to_true statement3 ∧
  evaluates_to_true statement4 :=
by {
  sorry
}

end evaluates_all_true_l351_351964


namespace cos_330_eq_sqrt_3_div_2_l351_351373

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351373


namespace evaluate_expression_l351_351977

variable (a b c d : ℝ)

theorem evaluate_expression :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c :=
sorry

end evaluate_expression_l351_351977


namespace product_mod_7_l351_351743

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351743


namespace company_pays_300_per_month_l351_351123

theorem company_pays_300_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box_per_month : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box_per_month = 0.5) :
  (total_volume / (length * width * height)) * cost_per_box_per_month = 300 := by
  sorry

end company_pays_300_per_month_l351_351123


namespace cos_330_is_sqrt3_over_2_l351_351294

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351294


namespace cost_of_each_top_l351_351936

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l351_351936


namespace area_of_circle_with_diameter_10_l351_351097

theorem area_of_circle_with_diameter_10 (d : ℝ) (π : ℝ) (h : d = 10): 
  ∃ A, A = π * ((d / 2) ^ 2) ∧ A = 25 * π :=
begin
  use π * ((10 / 2) ^ 2),
  split,
  { rw h, },
  { ring }
end

end area_of_circle_with_diameter_10_l351_351097


namespace remainder_of_product_l351_351770

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351770


namespace water_added_l351_351122

theorem water_added (C : ℝ) (initial_percent : ℝ) (final_fraction : ℝ) (added_water : ℝ) :
    (initial_percent = 0.30) →
    (final_fraction = 3/4) →
    (C = 80) →
    added_water = (final_fraction * C) - (initial_percent * C) →
    added_water = 36 := by
  intros h_initial_percent h_final_fraction h_C h_added_water
  rw [h_initial_percent, h_final_fraction, h_C] at h_added_water
  simp at h_added_water
  exact h_added_water

-- Theorem water_added states that the added_water is 36 liters 
-- under the given conditions (initial percentage, final fraction, and total capacity).

end water_added_l351_351122


namespace trapezoid_area_l351_351156

variable (A B C D M : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M]

-- Define point objects
variable [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint D] [IsPoint M]

-- Conditions
variable (AB CD : Line)
variable (midpoint_M_AD : M = midpoint A D)
variable (angle_MCB : Measure = 150)
variable (length_BC : ℝ)
variable (length_MC : ℝ)

-- Hypotheses
hypothesis (AB_parallel_CD : Parallel AB CD)
hypothesis (M_midpoint_AD : midpoint_M_AD)

noncomputable def area_trapezoid_ABCD (x y : ℝ) : ℝ := 
  if (midpoint M A D) 
  then x * y / 2
  else 0

theorem trapezoid_area (BC x : ℝ) (MC y : ℝ) 
  (M : Point) (M_midpoint_AD : M = midpoint A D) 
  (angle_MCB : Measure = 150) 
  : area_trapezoid_ABCD BC MC = x * y / 2 := 
  sorry

end trapezoid_area_l351_351156


namespace exact_current_time_is_8_01_AM_l351_351626

theorem exact_current_time_is_8_01_AM
  (t : ℝ) -- t is the current time in minutes after 8:00 AM
  (h_t_range : 0 ≤ t ∧ t < 60) :
  eight_minutes_from_now_opposite_hour_hand_four_minutes_ago t →
  t = 1.82 :=
by 
  -- Definitions
  let minute_hand_position := (t + 8) * 6
  let hour_hand_position := 240 + 0.5 * (t - 4)
  have opposite_condition := abs (minute_hand_position - hour_hand_position) = 180
  -- Applying the conditions
  assume condition : abs (6 * (t + 8) - (238 + 0.5 * t)) = 180
  sorry

end exact_current_time_is_8_01_AM_l351_351626


namespace remainder_of_product_l351_351777

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351777


namespace ellipse_foci_distance_l351_351909

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351909


namespace intersection_points_l351_351021

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem intersection_points :
  ∃ p q : ℝ × ℝ, 
    (p = (0, c) ∨ p = (-1, a - b + c)) ∧ 
    (q = (0, c) ∨ q = (-1, a - b + c)) ∧
    p ≠ q ∧
    (∃ x : ℝ, (x, ax^2 + bx + c) = p) ∧
    (∃ x : ℝ, (x, -ax^3 + bx + c) = q) :=
by
  sorry

end intersection_points_l351_351021


namespace sum_first_10_common_elements_in_ap_gp_l351_351432

/-- To find the sum of the first 10 elements that appear in both the arithmetic progression (AP) 
  {5, 8, 11, 14, ...} and the geometric progression (GP) {10, 20, 40, 80, ...}, we follow these steps:
-/
theorem sum_first_10_common_elements_in_ap_gp 
  (a_n : ℕ → ℕ := λ n, 5 + 3 * n)
  (b_k : ℕ → ℕ := λ k, 10 * 2^k)
  (common_elements : ℕ → ℕ := λ m, 20 * 4^m) :
  (Finset.range 10).sum (λ i, common_elements i) = 6990500 := 
by
  -- Set up the common_elements based on the given progressions
  -- Calculate the sum of the first 10 terms of the geometric progression
  sorry

end sum_first_10_common_elements_in_ap_gp_l351_351432


namespace tobias_shoveled_six_driveways_l351_351083

def money_from_allowance (months : ℕ) (allowance_per_month : ℚ) : ℚ :=
  months * allowance_per_month

def money_from_mowing_lawns (lawns : ℕ) (charge_per_lawn : ℚ) : ℚ :=
  lawns * charge_per_lawn

def money_from_part_time_job (hours : ℕ) (wage_per_hour : ℚ) : ℚ :=
  hours * wage_per_hour

def total_money_before_shoes 
  (allowance_money : ℚ) 
  (mowing_money : ℚ) 
  (part_time_job_money : ℚ) : ℚ :=
  allowance_money + mowing_money + part_time_job_money

def money_from_shoveling_driveways 
  (total_before_shoes : ℚ) 
  (change_after_shoes : ℚ) 
  (cost_of_shoes : ℚ) 
  (total_from_other_sources : ℚ) : ℚ :=
  total_before_shoes - (total_from_other_sources - change_after_shoes)

def number_of_driveways_shoveled (total_money_from_shoveling : ℚ) (charge_per_driveway : ℚ) : ℕ :=
  (total_money_from_shoveling / charge_per_driveway).toNat

theorem tobias_shoveled_six_driveways :
  let cost_of_shoes := 95
  let allowance_per_month := 5
  let total_allowance_months := 3
  let charge_per_lawn := 15
  let lawns_mowed := 4
  let wage_per_hour := 8
  let hours_worked := 10
  let charge_per_driveway := 7
  let change_after_shoes := 15
  let total_before_shoes := 110
  
  let allowance_money := money_from_allowance total_allowance_months allowance_per_month
  let mowing_money := money_from_mowing_lawns lawns_mowed charge_per_lawn
  let part_time_job_money := money_from_part_time_job hours_worked wage_per_hour
  let total_money_other_sources := total_money_before_shoes allowance_money mowing_money part_time_job_money

  number_of_driveways_shoveled 
    (total_money_other_sources - (total_before_shoes - cost_of_shoes + change_after_shoes)) 
    charge_per_driveway = 6 := by 
      sorry

end tobias_shoveled_six_driveways_l351_351083


namespace cos_330_eq_sqrt3_div_2_l351_351286

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351286


namespace cos_330_eq_sqrt3_div_2_l351_351280

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351280


namespace sum_of_products_of_roots_l351_351006

noncomputable def poly : Polynomial ℝ := 5 * Polynomial.X^3 - 10 * Polynomial.X^2 + 17 * Polynomial.X - 7

theorem sum_of_products_of_roots :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ poly.eval p = 0 ∧ poly.eval q = 0 ∧ poly.eval r = 0) →
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ ((p * q + p * r + q * r) = 17 / 5)) :=
by
  sorry

end sum_of_products_of_roots_l351_351006


namespace probability_digit_three_in_six_sevenths_l351_351022

theorem probability_digit_three_in_six_sevenths :
  let decimal_rep := "857142" in
  (∀ d ∈ decimal_rep.toList, d ≠ '3') →
  ∃ prob : ℚ, (prob = 0) :=
by
  sorry

end probability_digit_three_in_six_sevenths_l351_351022


namespace cos_330_is_sqrt3_over_2_l351_351289

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351289


namespace product_remainder_mod_7_l351_351766

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351766


namespace find_a_l351_351478

noncomputable def complex_z (a : ℝ) := a + ((Complex.I - 1) / (1 + Complex.I))
theorem find_a (a : ℝ) (h : complex_z a.re = complex_z a.im) : a = 1 :=
by
  sorry

end find_a_l351_351478


namespace coupon_discount_l351_351872

theorem coupon_discount (total_before_coupon : ℝ) (amount_paid_per_friend : ℝ) (number_of_friends : ℕ) :
  total_before_coupon = 100 ∧ amount_paid_per_friend = 18.8 ∧ number_of_friends = 5 →
  ∃ discount_percentage : ℝ, discount_percentage = 6 :=
by
  sorry

end coupon_discount_l351_351872


namespace gcf_75_100_l351_351819

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351819


namespace rotated_line_equation_l351_351887

-- Define the given conditions
def P : Point := (3, 4) -- Point P with x-coordinate 3 on the line

def original_line (x y : ℝ) : Prop := x - y + 1 = 0 -- Original line equation

-- Define the perpendicular line obtained by rotating the original line 90° counterclockwise
def perpendicular_line (x y : ℝ) : Prop := x + y - 7 = 0 -- Perpendicular line equation

-- Theorem statement: Given conditions, prove the equation of the rotated line
theorem rotated_line_equation : 
  (∀ (x y : ℝ), original_line x y → (x = 3 → y = 4)) →
  (∀ (x y : ℝ), perpendicular_line x y ↔ original_line x y → x = 3 ∧ y = 4)
:=
begin
  sorry
end

end rotated_line_equation_l351_351887


namespace parallel_lines_m_value_l351_351733

noncomputable def m_value_parallel (m : ℝ) : Prop :=
  (m-1) / 2 = 1 / -3

theorem parallel_lines_m_value :
  ∀ (m : ℝ), (m_value_parallel m) → m = 1 / 3 :=
by
  intro m
  intro h
  sorry

end parallel_lines_m_value_l351_351733


namespace cos_330_eq_sqrt3_div_2_l351_351262

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351262


namespace segments_covered_by_q_half_lines_l351_351456

theorem segments_covered_by_q_half_lines (q : ℕ) 
(halflines : ℕ → set ℝ) -- each halflines index represents a half-line
-- Condition: Each half-line is either right-infinite or left-infinite
(right_infinite : ∀ n, ∃ b : ℝ, (∀ x : ℝ, x ≥ b → x ∈ halflines n) ∨ (∀ x : ℝ, x ≤ b → x ∈ halflines n))
-- Segment structure based on endpoints defined by half-lines
(segments : set ℝ → set ℝ) -- function that maps endpoints to segments
-- Hypothesis: segment count function
(segment_count : ℕ → ℕ) -- function counting segments covered by exactly q half-lines
:  
segment_count q ≤ q + 1 := 
sorry

end segments_covered_by_q_half_lines_l351_351456


namespace jane_reading_speed_second_half_l351_351628

-- Definitions from the problem's conditions
def total_pages : ℕ := 500
def first_half_pages : ℕ := total_pages / 2
def first_half_speed : ℕ := 10
def total_days : ℕ := 75

-- The number of days spent reading the first half
def first_half_days : ℕ := first_half_pages / first_half_speed

-- The number of days spent reading the second half
def second_half_days : ℕ := total_days - first_half_days

-- The number of pages in the second half
def second_half_pages : ℕ := total_pages - first_half_pages

-- The actual theorem stating that Jane's reading speed for the second half was 5 pages per day
theorem jane_reading_speed_second_half :
  second_half_pages / second_half_days = 5 :=
by
  sorry

end jane_reading_speed_second_half_l351_351628


namespace evaluate_log_base_4_l351_351976

def log_base_4_256_minus_log_base_4_16 : Prop :=
  let a := Real.log 256 / Real.log 4 in
  let b := Real.log 16 / Real.log 4 in
  a - b = 2

theorem evaluate_log_base_4 : log_base_4_256_minus_log_base_4_16 :=
by
  sorry

end evaluate_log_base_4_l351_351976


namespace cos_330_eq_sqrt3_div_2_l351_351258

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351258


namespace cyclist_average_speed_l351_351882

theorem cyclist_average_speed 
  (d : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = 20) 
  (h2 : v1 * 1.2 = v2) 
  (h3 : v2 = 24) :
  let t1 := d / 60 
  let t2 := d / 36 
  let T := t1 + t2 
  (v_avg : ℝ) := d / T in
  v_avg = 22.5 := 
by 
  sorry

end cyclist_average_speed_l351_351882


namespace remainder_of_product_mod_7_l351_351761

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351761


namespace isosceles_triangle_tangent_theorem_l351_351656

variables {A B C D E F M N : Point}
variables {AB AC BC : Line}
variables {α : Angle}

-- Definitions for points and lines
def isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist A C

def midpoint (D B C : Point) : Prop :=
  dist B D = dist C D

def perpendicular_projection (D P : Point) (AB : Line) : Prop :=
  ∃ E, E ∈ AB ∧ angle D E P = 90 

def circle_center (D : Point) (E : Point) : set Point :=
  { P : Point | dist D P = dist D E }

def tangent_to_circle (MN : Line) (circle : set Point) (E : Point) : Prop :=
  ∃ M N, MN = Line.mk M N ∧ M ∈ circle ∧ N ∈ circle ∧ (MN).is_tangent circle

-- Given conditions
def conditions : Prop :=
  isosceles_triangle A B C ∧
  midpoint D B C ∧
  perpendicular_projection D E AB ∧
  perpendicular_projection D F AC ∧
  ∀ P, P ∈ (circle_center D E) ↔ P = E ∧ P = F ∧
  ∃ MN, tangent_to_circle (Line.mk M N) (circle_center D E) E

-- The statement to be proved
theorem isosceles_triangle_tangent_theorem : conditions → (dist B D)^2 = (dist B M) * (dist C N) :=
sorry

end isosceles_triangle_tangent_theorem_l351_351656


namespace rectangle_breadth_approx_1_1_l351_351065

theorem rectangle_breadth_approx_1_1 (s b : ℝ) (h1 : 4 * s = 2 * (16 + b))
  (h2 : abs ((π * s / 2) + s - 21.99) < 0.01) : abs (b - 1.1) < 0.01 :=
sorry

end rectangle_breadth_approx_1_1_l351_351065


namespace log_diff_example_l351_351971

theorem log_diff_example : 
  log 4 256 - log 4 16 = 2 :=
sorry

end log_diff_example_l351_351971


namespace sum_of_coordinates_A_l351_351605

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351605


namespace cos_330_eq_sqrt3_div_2_l351_351271

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351271


namespace triangle_area_l351_351152

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Define the property of being a right triangle via the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of a right triangle given base and height
def area_right_triangle (a b : ℕ) : ℕ := (a * b) / 2

-- The main theorem, stating that the area of the triangle with sides 9, 12, 15 is 54
theorem triangle_area : is_right_triangle a b c → area_right_triangle a b = 54 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l351_351152


namespace problem_statement_l351_351476

open Set

noncomputable def U := ℝ

def A : Set ℝ := { x | 0 < 2 * x + 4 ∧ 2 * x + 4 < 10 }
def B : Set ℝ := { x | x < -4 ∨ x > 2 }
def C (a : ℝ) (h : a < 0) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem problem_statement (a : ℝ) (ha : a < 0) :
    A ∪ B = { x | x < -4 ∨ x > -2 } ∧
    compl (A ∪ B) ⊆ C a ha → -2 < a ∧ a < -4 / 3 :=
sorry

end problem_statement_l351_351476


namespace percentage_of_one_pair_repeated_digits_l351_351173

theorem percentage_of_one_pair_repeated_digits (n : ℕ) (h1 : 10000 ≤ n) (h2 : n ≤ 99999) :
  ∃ (percentage : ℝ), percentage = 56.0 :=
by
  sorry

end percentage_of_one_pair_repeated_digits_l351_351173


namespace four_valid_pairs_of_squares_l351_351500

theorem four_valid_pairs_of_squares (m n : ℕ) (h₀ : m ≥ n) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m ^ 2 - n ^ 2 = 120) :
  (∃ (m n : ℕ), m ≥ n ∧ m > 0 ∧ n > 0 ∧ m ^ 2 - n ^ 2 = 120) ↔ (finset.card ({p : ℕ × ℕ | p.1 ≥ p.2 ∧ p.1 ^ 2 - p.2 ^ 2 = 120}.to_finset) = 4) :=
sorry

end four_valid_pairs_of_squares_l351_351500


namespace cos_330_eq_sqrt3_div_2_l351_351186

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351186


namespace remainder_product_l351_351751

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351751


namespace sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l351_351524

-- Given conditions for the triangle ABC
variables {A B C a b c : ℝ}
axiom angle_C_eq_two_pi_over_three : C = 2 * Real.pi / 3
axiom c_squared_eq_five_a_squared_plus_ab : c^2 = 5 * a^2 + a * b

-- Proof statements
theorem sin_B_over_sin_A_eq_two (hAC: C = 2 * Real.pi / 3) (hCond: c^2 = 5 * a^2 + a * b) :
  Real.sin B / Real.sin A = 2 :=
sorry

theorem max_value_sin_A_sin_B (hAC: C = 2 * Real.pi / 3) :
  ∃ A B : ℝ, 0 < A ∧ A < Real.pi / 3 ∧ B = (Real.pi / 3 - A) ∧ Real.sin A * Real.sin B ≤ 1 / 4 :=
sorry

end sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l351_351524


namespace point_transformations_l351_351066

theorem point_transformations (a b : ℝ) (h : (a ≠ 2 ∨ b ≠ 3))
  (H1 : ∃ x y : ℝ, (x, y) = (2 - (b - 3), 3 + (a - 2)) ∧ (y, x) = (-4, 2)) :
  b - a = -6 :=
by
  sorry

end point_transformations_l351_351066


namespace match_scheduling_ways_l351_351808

noncomputable def schedule_ways : Nat := 8!

theorem match_scheduling_ways : schedule_ways = 40320 :=
by
  -- Conditions interpretation and the proof structuring steps if necessary.
  sorry

end match_scheduling_ways_l351_351808


namespace x_2_eq_2_x_3_eq_9_by_4_x_4_eq_793_by_324_x_strictly_increasing_y_greater_than_y_n_minus_1_plus_3_exists_x_N_greater_than_2016_l351_351114

noncomputable def x : ℕ → ℝ
| 1       := 1
| (n + 1) := x n + 1 / (x n)^2

def y (n : ℕ) : ℝ := (x n) ^ 3

theorem x_2_eq_2 : x 2 = 2 :=
sorry

theorem x_3_eq_9_by_4 : x 3 = 9 / 4 :=
sorry

theorem x_4_eq_793_by_324 : x 4 = 793 / 324 :=
sorry

theorem x_strictly_increasing (n : ℕ) (h : n ≥ 2) : x (n + 1) > x n :=
sorry

theorem y_greater_than_y_n_minus_1_plus_3 (n : ℕ) (h : n ≥ 2) : y (n + 1) > y n + 3 :=
sorry

theorem exists_x_N_greater_than_2016 : ∃ N : ℕ, x N > 2016 :=
sorry

end x_2_eq_2_x_3_eq_9_by_4_x_4_eq_793_by_324_x_strictly_increasing_y_greater_than_y_n_minus_1_plus_3_exists_x_N_greater_than_2016_l351_351114


namespace find_x_in_acute_triangle_l351_351463

-- Definition of an acute triangle with given segment lengths due to altitudes
def acute_triangle_with_segments (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) : Prop :=
  BC = 4 + x ∧ AE = x ∧ BE = 8 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- The theorem to prove
theorem find_x_in_acute_triangle (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) 
  (h : acute_triangle_with_segments A B C D E BC AE BE x) : 
  x = 4 :=
by
  -- As the focus is on the statement, we add sorry to skip the proof.
  sorry

end find_x_in_acute_triangle_l351_351463


namespace smallest_positive_a_with_conditions_l351_351102

noncomputable def smallest_a : ℤ :=
  4

theorem smallest_positive_a_with_conditions :
  ∃ b c : ℤ, ∀ r1 r2 : ℝ, r1 ≠ r2 ∧ 0 < r1 ∧ r1 < 1 ∧ 0 < r2 ∧ r2 < 1 →
    let a := smallest_a in
    (r1 + r2 = -b / a ∧ r1 * r2 = c / a) :=
  sorry

end smallest_positive_a_with_conditions_l351_351102


namespace problem_positive_l351_351508

theorem problem_positive : ∀ x : ℝ, x < 0 → -3 * x⁻¹ > 0 :=
by 
  sorry

end problem_positive_l351_351508


namespace at_least_3_students_same_score_l351_351798

-- Conditions
def initial_points : ℕ := 6
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def num_questions : ℕ := 6
def num_students : ℕ := 51

-- Question
theorem at_least_3_students_same_score :
  ∃ score : ℤ, ∃ students_with_same_score : ℕ, students_with_same_score ≥ 3 :=
by
  sorry

end at_least_3_students_same_score_l351_351798


namespace part_1_part_2_l351_351477

noncomputable def z : ℂ := (1 + 2 * Complex.i) / (2 - Complex.i)

theorem part_1 : z + (1 / z) = 0 :=
by
  sorry

theorem part_2 : z * Complex.conj(z) = 1 :=
by
  sorry

end part_1_part_2_l351_351477


namespace solve_equation_l351_351714

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l351_351714


namespace smallest_positive_integer_m_l351_351005

noncomputable def fractional_part (x : ℝ) := x - (Real.floor x : ℝ)

noncomputable def g (x : ℝ) := abs (3 * fractional_part x - 1.5)

def m_smallest (m : ℕ) : Prop :=
  (∀ x, abs (3 * fractional_part x - 1.5) ≥ 0) → 
  (∃ n : ℕ, n = 3000 ∧ (∀ x : ℝ, m * g (x * g x) = x → n ≤ 3000))

theorem smallest_positive_integer_m : ∃ m : ℕ, m_smallest m := sorry

end smallest_positive_integer_m_l351_351005


namespace passing_grade_fraction_l351_351111

theorem passing_grade_fraction (A B C D F : ℚ) (hA : A = 1/4) (hB : B = 1/2) (hC : C = 1/8) (hD : D = 1/12) (hF : F = 1/24) : 
  A + B + C = 7/8 :=
by
  sorry

end passing_grade_fraction_l351_351111


namespace sum_of_cubes_first_six_l351_351017

theorem sum_of_cubes_first_six {n : ℕ} (h : n = 6) :
  (∑ k in Finset.range (n + 1), k^3) = (∑ k in Finset.range (n + 1), k)^2 :=
by
  sorry

end sum_of_cubes_first_six_l351_351017


namespace centroid_of_PQR_eq_K_l351_351116

open EuclideanGeometry

theorem centroid_of_PQR_eq_K (A B C S P Q R K : Point ℝ) (hS : centroid ℝ ⟨A, B, C⟩ = S) (hK : circumcenter ℝ ⟨A, B, C⟩ = K)
  (hP : circumcenter ℝ ⟨B, C, S⟩ = P) (hQ : circumcenter ℝ ⟨C, A, S⟩ = Q) (hR : circumcenter ℝ ⟨A, B, S⟩ = R) :
  centroid ℝ ⟨P, Q, R⟩ = K := sorry

end centroid_of_PQR_eq_K_l351_351116


namespace tan_600_eq_sqrt3_l351_351505

theorem tan_600_eq_sqrt3 : (Real.tan (600 * Real.pi / 180)) = Real.sqrt 3 := 
by 
  -- sorry to skip the actual proof steps
  sorry

end tan_600_eq_sqrt3_l351_351505


namespace range_of_m_l351_351044

theorem range_of_m (m : ℝ) (n : ℕ) :
  m = 2017 * (∏ k in Finset.range n, (1 - 1 / (k + 2 : ℝ))) → 2 < m ∧ m ≤ 3 := by
  sorry

end range_of_m_l351_351044


namespace curve_intersection_and_locus_l351_351052

theorem curve_intersection_and_locus
  (m : ℝ)
  (G3 : ℝ → ℝ := λ x, 2*x^3 - 8*x)
  (G2 : ℝ → ℝ := λ x, m*x^2 - 8*x)
  (A : ℝ × ℝ := (m / 2, (m^3 / 4) - 4 * m))
  (line_e_slope : ℝ := (m^2 - 16) / 2)
  (line_e : ℝ → ℝ := λ x, line_e_slope * x)
  (M1 : ℝ × ℝ := (1, 0))
  (M2 : ℝ × ℝ := (-1, 0))
  (zero_points_G3 := [0, 2, -2])
  (locus_M1 : ℝ × ℝ := (1, 0))
  (locus_M2 : ℝ × ℝ := (-1, 0))
  (circle_eq1 : ℝ × ℝ → Prop := λ p, (p.1 - 1)^2 + p.2^2 = 1)
  (circle_eq2 : ℝ × ℝ → Prop := λ p, (p.1 + 1)^2 + p.2^2 = 1) : 
  (line_e (A.1) = A.2) ∧ (circle_eq1 (1, 0)) ∧ (circle_eq2 (-1, 0)) :=
by
  sorry

end curve_intersection_and_locus_l351_351052


namespace consecutive_odd_integers_sum_l351_351060

theorem consecutive_odd_integers_sum (x : ℤ) (h1 : ¬ even x) (h2 : ¬ even (x + 4)) (h3 : x + 4 = 5 * x) :
  x + (x + 4) = 6 :=
by
  sorry

end consecutive_odd_integers_sum_l351_351060


namespace triangle_is_right_triangle_l351_351693

-- Define Noncomputable because we're working with real numbers and trigonometric functions
noncomputable def triangle_right_triangle (a : ℝ) (c : ℝ) : Prop :=
  let b := 2 * a in
  let gamma := real.pi / 3 in -- 60° in radians
  -- Law of Cosines for the side opposite the 60° angle
  let c_squared := a^2 + b^2 - 2 * a * b * real.cos(gamma) in
  -- Calculated length using the sides and cosine of 60°
  c = real.sqrt c_squared
  
theorem triangle_is_right_triangle (a : ℝ) :
  ∀ c, triangle_right_triangle a c → ∃ α, 0 ≤ α ∧ α < real.pi ∧ real.cos α = 0 :=
by {
  sorry
}

end triangle_is_right_triangle_l351_351693


namespace maximize_g_in_interval_l351_351435

noncomputable def g (x : ℝ) : ℝ := Real.sin(2 * x + Real.pi / 4)

theorem maximize_g_in_interval : 
  ∃ x ∈ Icc 0 (Real.pi / 2), ∀ y ∈ Icc 0 (Real.pi / 2), g x ≥ g y ∧ x = Real.pi / 8 := 
by
  sorry

end maximize_g_in_interval_l351_351435


namespace cos_330_eq_sqrt3_div_2_l351_351249

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351249


namespace product_remainder_mod_7_l351_351762

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351762


namespace cos_330_eq_sqrt3_div_2_l351_351312

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351312


namespace product_of_two_numbers_l351_351804

theorem product_of_two_numbers :
  ∃ (a b : ℚ), (∀ k : ℚ, a = k + b) ∧ (∀ k : ℚ, a + b = 8 * k) ∧ (∀ k : ℚ, a * b = 40 * k) ∧ (a * b = 6400 / 63) :=
by {
  sorry
}

end product_of_two_numbers_l351_351804


namespace cos_330_eq_sqrt3_div_2_l351_351281

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351281


namespace cos_330_eq_sqrt3_over_2_l351_351211

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351211


namespace cos_330_eq_sqrt_3_div_2_l351_351377

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351377


namespace cos_330_eq_sqrt3_div_2_l351_351324

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351324


namespace cos_330_eq_sqrt3_over_2_l351_351210

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351210


namespace gcd_75_100_l351_351842

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351842


namespace line_passes_through_fixed_point_equal_intercepts_line_equation_l351_351481

open Real

theorem line_passes_through_fixed_point (m : ℝ) : ∃ P : ℝ × ℝ, P = (4, 1) ∧ (m + 2) * P.1 - (m + 1) * P.2 - 3 * m - 7 = 0 := 
sorry

theorem equal_intercepts_line_equation (m : ℝ) :
  ((3 * m + 7) / (m + 2) = -(3 * m + 7) / (m + 1)) → (m = -3 / 2) → 
  (∀ (x y : ℝ), (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0 → x + y - 5 = 0) := 
sorry

end line_passes_through_fixed_point_equal_intercepts_line_equation_l351_351481


namespace sum_of_coordinates_A_l351_351553

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351553


namespace concurrency_of_lines_l351_351535

variables {O : Type*} [EuclideanSpace ℝ O]
variables (C C1 C2 : circline O) (O_c O1_c O2_c : O) (r r1 r2 : ℝ)
variables (A A1 A2 : O)

-- Circle C with center O_c and radius r
def is_circle_C := is_circle O_c r C

-- Circle C1 with center O1_c and radius r1, tangent to C at A1
def is_circle_C1 := is_circle O1_c r1 C1 ∧ is_tangent C C1 A1

-- Circle C2 with center O2_c and radius r2, tangent to C at A2
def is_circle_C2 := is_circle O2_c r2 C2 ∧ is_tangent C C2 A2

-- C1 and C2 are externally tangent to each other at A
def is_ext_tangent_C1_C2 := is_tangent_external C1 C2 A

-- Lines O A, O1 A2, O2 A1 are concurrent
theorem concurrency_of_lines
  (hC : is_circle_C)
  (hC1 : is_circle_C1)
  (hC2 : is_circle_C2)
  (h_ext_tangent : is_ext_tangent_C1_C2) :
  concurrent (line_through O_c A) (line_through O1_c A2) (line_through O2_c A1) :=
sorry

end concurrency_of_lines_l351_351535


namespace cos_330_eq_sqrt3_div_2_l351_351336

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351336


namespace complex_problem_l351_351008

open Complex

theorem complex_problem (z : ℂ) (h : 10 * norm_sq z = 2 * norm_sq (z + 3) + norm_sq (z^2 + 16) + 40) : 
  z + 9 / z = -3 / 17 :=
by
  sorry

end complex_problem_l351_351008


namespace cos_330_is_sqrt3_over_2_l351_351290

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351290


namespace radius_of_circle_in_spherical_coords_l351_351781

theorem radius_of_circle_in_spherical_coords :
  ∀ θ : ℝ, (∃ ρ φ : ℝ, ρ = 2 ∧ φ = π/4) →
  ∃ r : ℝ, r = ∥(2 * sin (π/4) * cos θ, 2 * sin (π/4) * sin θ, 2 * cos (π/4)).1.1 ∥ :=
begin
  intro θ,
  intro h,
  use sqrt 2,
  have : ∥(sqrt 2 * cos θ, sqrt 2 * sin θ, sqrt 2).1.1∥ = sqrt 2,
  sorry -- Skip the actual proof
end

end radius_of_circle_in_spherical_coords_l351_351781


namespace cos_330_eq_sqrt3_div_2_l351_351282

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351282


namespace circumcenters_form_regular_pentagon_l351_351645

theorem circumcenters_form_regular_pentagon (ABCDE : Type*) [convex_pentagon ABCDE] 
  (F : point_in_pentagon ABCDE)
  (P Q R S T : point)
  (hP : circumcenter (△ ABCDE.ABCDE.ABCDE.ABCDE.ABCDE.ABCDE.AF))
  (hQ : circumcenter (△ ABCDE.BF))
  (hR : circumcenter (△ ABCDE.CF))
  (hS : circumcenter (△ ABCDE.DF))
  (hT : circumcenter (△ ABCDE.EF))
  (hE : equiangular_at F ABCDE) :
  is_regular_pentagon (polygon.mk [P, Q, R, S, T]) :=
sorry

end circumcenters_form_regular_pentagon_l351_351645


namespace moles_of_KCl_formed_l351_351423

variables (NaCl KNO3 KCl NaNO3 : Type) 

-- Define the moles of each compound
variables (moles_NaCl moles_KNO3 moles_KCl moles_NaNO3 : ℕ)

-- Initial conditions
axiom initial_NaCl_condition : moles_NaCl = 2
axiom initial_KNO3_condition : moles_KNO3 = 2

-- Reaction definition
axiom reaction : moles_KCl = moles_NaCl

theorem moles_of_KCl_formed :
  moles_KCl = 2 :=
by sorry

end moles_of_KCl_formed_l351_351423


namespace part1_part2_l351_351475

-- Conditions and constants
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)
variables (h1 : a 1 = 1) 
variables (h2 : ∀ n, a n ≠ 0) 
variables (h3 : ∀ n, a n * a (n + 1) = λ * S n - 1)

-- Question 1: Proof that a_{n+2} - a_{n} = λ
theorem part1 (a S : ℕ → ℝ) (λ : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n ≠ 0)
  (h3 : ∀ n, a n * a (n + 1) = λ * S n - 1) :
  ∀ n, a (n + 2) - a n = λ :=
sorry

-- Additional condition for arithmetic sequence
hypothesis (ha : ∀ n, a (n + 1) = a n + (a 2 - a 1))
-- Sum conditions for T_n 
def T (n : ℕ) : ℝ := ∑ i in finset.range n, (a i / 3 ^ i)

-- Question 2: Proof that T_n < 1
theorem part2 (a S : ℕ → ℝ) (λ : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n ≠ 0)
  (h3 : ∀ n, a n * a (n + 1) = λ * S n - 1)
  (ha : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  ∀ n, T n < 1 :=
sorry

end part1_part2_l351_351475


namespace batsman_average_increase_l351_351861

theorem batsman_average_increase (A : ℕ) (H1 : 16 * A + 85 = 17 * (A + 3)) : A + 3 = 37 :=
by {
  sorry
}

end batsman_average_increase_l351_351861


namespace ellipse_major_axis_length_l351_351137

theorem ellipse_major_axis_length (r : ℝ) (h_r : r = 2) (k : ℝ) (h_k : k = 0.75) :
  let m := 2 * r in
  let M := (1 + k) * m in
  M = 7 :=
by
  sorry

end ellipse_major_axis_length_l351_351137


namespace cos_330_eq_sqrt3_div_2_l351_351197

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351197


namespace ellipse_foci_distance_l351_351922

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351922


namespace distance_between_foci_of_given_ellipse_l351_351905

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351905


namespace ratio_of_area_PSQT_QYR_eq_one_l351_351551

open Real

noncomputable def ratio_area_PSQT_QYR : Real :=
  let P := (0, 15) -- coordinates of P
  let Q := (0, 0) -- coordinates of Q
  let R := (20, 0) -- coordinates of R
  let S := ((P.1 + Q.1)/2, (P.2 + Q.2)/2) -- midpoint of PQ
  let T := ((P.1 + R.1)/2, (P.2 + R.2)/2) -- midpoint of PR
  let midpoint x y := ((fst x + fst y) / 2, (snd x + snd y) / 2)
  let area (a b c : (Real × Real)) :=
    abs $ (fst a * (snd b - snd c) + fst b * (snd c - snd a) + fst c * (snd a - snd b)) / 2
  let triangle_PQR_area := area P Q R
  let quad_PSQT_area := area P S T + area S T Q
  let triangle_QYR_area := area Q Y R
  (quad_PSQT_area / triangle_QYR_area)

theorem ratio_of_area_PSQT_QYR_eq_one :
  ratio_area_PSQT_QYR = 1 :=
    begin
      let P := (0, 15), -- coordinates of P
      let Q := (0, 0), -- coordinates of Q
      let R := (20, 0), -- coordinates of R
      let S := ((P.1 + Q.1)/2, (P.2 + Q.2)/2), -- midpoint of PQ
      let T := ((P.1 + R.1)/2, (P.2 + R.2)/2), -- midpoint of PR
      let midpoint x y := ((fst x + fst y) / 2, (snd x + snd y) / 2),
      let area (a b c : (Real × Real)) := abs ((fst a * (snd b - snd c) + fst b * (snd c - snd a) + fst c * (snd a - snd b)) / 2),
      let triangle_PQR_area := area P Q R,
      let quad_PSQT_area := area P S T + area S T Q,
      let triangle_QYR_area := area Q Y R,
      have H : ratio_area_PSQT_QYR = (quad_PSQT_area / triangle_QYR_area), by sorry,
      exact H
    end

end ratio_of_area_PSQT_QYR_eq_one_l351_351551


namespace derivative_f1_derivative_f2_l351_351985

-- Define the first problem's function and its derivative
def f1 (x : ℝ) : ℝ := (x^3 - 1) / (sin x)
def df1 (x : ℝ) : ℝ :=
  (3 * x^2 * (sin x) - (x^3 - 1) * (cos x)) / (sin x)^2

theorem derivative_f1 (x : ℝ) (h : sin x ≠ 0) : deriv f1 x = df1 x :=
by sorry

-- Define the second problem's function and its derivative
def f2 (x : ℝ) : ℝ := 2 * x * sin (2 * x + 5)
def df2 (x : ℝ) : ℝ :=
  4 * x * cos (2 * x + 5) + 2 * sin (2 * x + 5)

theorem derivative_f2 (x : ℝ) : deriv f2 x = df2 x :=
by sorry

end derivative_f1_derivative_f2_l351_351985


namespace cos_330_eq_sqrt3_div_2_l351_351351

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351351


namespace sum_of_coordinates_A_l351_351555

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351555


namespace minimum_even_numbers_l351_351536

/-- In a circular arrangement of 101 natural numbers, given that among any 5 consecutive numbers, there are at least two even numbers, the minimum number of even numbers is 41. -/
theorem minimum_even_numbers (n : ℕ) (h₁ : n = 101) 
    (h₂ : ∀ (nums : fin n → ℕ), 
      (∀ i : fin n, 2 ≤ (finset.card (finset.filter (λ x, (nums (i + x)) % 2 = 0) (finset.range 5)))) 
        → ∃ evens : finset (fin n), finset.card evens = 41) : 
      ∃ evens : finset (fin n), finset.card evens = 41 :=
by
  sorry

end minimum_even_numbers_l351_351536


namespace cos_330_is_sqrt3_over_2_l351_351293

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351293


namespace tangent_line_k_value_l351_351997

theorem tangent_line_k_value : ∃ k : ℝ, (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → k = 49) :=
begin
  sorry
end

end tangent_line_k_value_l351_351997


namespace total_pencils_is_54_l351_351683

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l351_351683


namespace gcd_75_100_l351_351831

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351831


namespace sum_of_coordinates_of_A_l351_351586

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351586


namespace sequence_sum_S15_l351_351461

theorem sequence_sum_S15 : 
  (∀ (n : ℕ), n > 1 → S (n + 1) + S (n - 1) = 2 * (S n + S 1)) →
  S 1 = 1 →
  S 2 = 2 →
  S 15 = 120 :=
by
  intros h_seq h_S1 h_S2
  sorry

end sequence_sum_S15_l351_351461


namespace cos_330_eq_sqrt3_div_2_l351_351283

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351283


namespace distance_between_ellipse_foci_l351_351915

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351915


namespace seating_arrangement_7_people_l351_351548

theorem seating_arrangement_7_people (n : Nat) (h1 : n = 7) :
  let m := n - 1
  (m.factorial / m) * 2 = 240 :=
by
  sorry

end seating_arrangement_7_people_l351_351548


namespace trajectory_passes_through_incenter_l351_351647

variables {Point : Type}
variables [inhabited Point] [add_comm_group Point] [vector_space ℝ Point] [metric_space Point]

def vector (A B : Point) : Point := sorry -- placeholder for vector between points A and B
def norm (v : Point) : ℝ := sorry -- placeholder for the norm of a vector

noncomputable def trajectory (O A B C P : Point) (λ : ℝ) (hλ : λ > 0) : Prop :=
  vector O P = vector O A + λ * (vector A B / norm (vector A B) + vector A C / norm (vector A C))

def is_incenter (P A B C : Point) : Prop := sorry -- placeholder for checking if P is the incenter of ΔABC

theorem trajectory_passes_through_incenter
  (O A B C P : Point) (λ : ℝ) (hλ : λ > 0)
  (h : trajectory O A B C P λ hλ) : is_incenter P A B C :=
sorry

end trajectory_passes_through_incenter_l351_351647


namespace factorial_base_b4_l351_351381

theorem factorial_base_b4 (b1 b2 b3 b4 b5 : ℕ) (h1 : 0 ≤ b1 ∧ b1 ≤ 1)
    (h2 : 0 ≤ b2 ∧ b2 ≤ 2) (h3 : 0 ≤ b3 ∧ b3 ≤ 3) (h4 : 0 ≤ b4 ∧ b4 ≤ 4)
    (h5 : 0 ≤ b5 ∧ b5 ≤ 5) (h : 762 = b1 * 1! + b2 * 2! + b3 * 3! + b4 * 4! + b5 * 5!) :
    b4 = 1 :=
    sorry

end factorial_base_b4_l351_351381


namespace gcd_75_100_l351_351830

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351830


namespace cos_330_eq_sqrt3_div_2_l351_351276

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351276


namespace sum_of_coordinates_A_l351_351567

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351567


namespace cos_330_eq_sqrt3_div_2_l351_351317

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351317


namespace probability_is_seven_over_fifteen_l351_351734

-- Define the finite set of numbers from 1 to 30.
def cardSet := Finset.range 30

-- Define the property to check whether a number is a multiple of 3 or 5.
def isMultipleOfThreeOrFive (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 5 = 0

-- Count the number of elements in the set that satisfy the given property.
def countMultiplesOfThreeOrFive : ℕ :=
  (cardSet.filter isMultipleOfThreeOrFive).card

-- Calculate the total number of elements in the set (from 1 to 30).
def totalElements : ℕ := cardSet.card

-- Calculate the probability as a ratio of the count of multiples to the total elements.
def probabilityOfMultipleOfThreeOrFive : ℚ :=
  countMultiplesOfThreeOrFive / totalElements

-- The theorem statement that we aim to prove.
theorem probability_is_seven_over_fifteen : probabilityOfMultipleOfThreeOrFive = 7 / 15 :=
  sorry

end probability_is_seven_over_fifteen_l351_351734


namespace pool_percentage_filled_after_addition_l351_351143

def pool_total_capacity := 1529.4117647058824
def additional_water := 300
def percentage_increase := 30 / 100

theorem pool_percentage_filled_after_addition :
  (pool_total_capacity / pool_total_capacity) * 100 = 40.38 :=
by
  -- Using the given conditions to construct the proof
  have calculation1 : percentage_increase * pool_total_capacity = 458.8235294117647 := by sorry
  have initial_water := 158.8235294117647 := by sorry
  have initial_percentage_filled := (initial_water / pool_total_capacity) * 100 := by sorry
  have final_percentage_filled := initial_percentage_filled + 30 := by sorry
  have final_percentage_filled = 40.38 := by sorry
  apply final_percentage_filled
  sorry

end pool_percentage_filled_after_addition_l351_351143


namespace x_one_sub_f_eq_one_l351_351007

theorem x_one_sub_f_eq_one:
  let α := (2 + Real.sqrt 5) in
  let x := α ^ 500 in
  let n := ⌊x⌋ in
  let f := x - n in
  x * (1 - f) = 1 :=
by 
  sorry

end x_one_sub_f_eq_one_l351_351007


namespace area_under_arccos_cos_l351_351983

theorem area_under_arccos_cos : 
  ∫ x in 0..(2 * π), real.arccos (real.cos x) = π^2 := sorry

end area_under_arccos_cos_l351_351983


namespace enrique_commission_l351_351406

theorem enrique_commission :
  let commission_rate : ℚ := 0.15
  let suits_sold : ℚ := 2
  let suits_price : ℚ := 700
  let shirts_sold : ℚ := 6
  let shirts_price : ℚ := 50
  let loafers_sold : ℚ := 2
  let loafers_price : ℚ := 150
  let total_sales := suits_sold * suits_price + shirts_sold * shirts_price + loafers_sold * loafers_price
  let commission := commission_rate * total_sales
  commission = 300 := by
begin
  sorry
end

end enrique_commission_l351_351406


namespace determine_x_value_l351_351010

variable {a b x r : ℝ}
variable (b_nonzero : b ≠ 0)

theorem determine_x_value (h1 : r = (3 * a)^(3 * b)) (h2 : r = a^b * x^b) : x = 27 * a^2 :=
by
  sorry

end determine_x_value_l351_351010


namespace correct_statements_incorrect_statement_l351_351666

noncomputable def f (x b c : ℝ) : ℝ :=
  |x| * x + b * x + c

theorem correct_statements :
  ( ∀ c > 0, f 0 0 c = 0 → ∃! x : ℝ, f x 0 c = 0 ) ∧
  ( ∀ x b, f x b 0 = -f (-x) b 0 ) ∧
  ( ∀ x y c, ∃ p : ℝ, f p 0 c = y → p = 0 ∧ y = c )
 :=
by
  sorry

theorem incorrect_statement :
  ( ∀ b c, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) → false )
 :=
by
  sorry

end correct_statements_incorrect_statement_l351_351666


namespace cyclist_energized_time_l351_351398

-- Definitions based on the conditions in a)
def total_time : ℝ := 9
def total_distance : ℝ := 154
def speed_energized : ℝ := 22
def speed_exhausted : ℝ := 15

-- To be shown
theorem cyclist_energized_time 
  (x : ℝ) 
  (h1 : total_distance = speed_energized * x + speed_exhausted * (total_time - x)) :
  x = 19 / 7 :=
by
  have h2 : 22 * x + 15 * (9 - x) = 154 from h1
  sorry

end cyclist_energized_time_l351_351398


namespace Marley_fruit_count_l351_351680

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l351_351680


namespace sum_of_coords_A_l351_351571

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351571


namespace gcd_75_100_l351_351829

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351829


namespace cos_330_cos_30_val_answer_l351_351240

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351240


namespace cos_330_cos_30_val_answer_l351_351230

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351230


namespace cos_330_eq_sqrt_3_div_2_l351_351219

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351219


namespace period_f_monotonic_intervals_max_f_on_interval_min_f_on_interval_l351_351483

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin x ^ 2

theorem period_f : ∀ x, f (x + π) = f x :=
by sorry

theorem monotonic_intervals :
  ∀ k : ℤ, ∀ x1 x2 : ℝ, (k * π - π / 6) ≤ x1 → x1 ≤ x2 → x2 ≤ (k * π + π / 3) → f x1 ≤ f x2 :=
by sorry

theorem max_f_on_interval : ∀ x ∈ [0, π / 2], f x ≤ 2 :=
by sorry

theorem min_f_on_interval : ∀ x ∈ [0, π / 2], f x ≥ 1 / 2 :=
by sorry

end period_f_monotonic_intervals_max_f_on_interval_min_f_on_interval_l351_351483


namespace product_remainder_mod_7_l351_351763

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351763


namespace gcd_75_100_l351_351815

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351815


namespace count_distinct_house_numbers_l351_351397

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def two_digit_primes_between_40_and_60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_distinct_house_numbers : 
  let primes := two_digit_primes_between_40_and_60 in
  let WXYZs := { WXYZ : ℕ // ∀ n ∈ primes, ∃ WX YZ : ℕ, WX ≠ YZ ∧ WXYZ = WX * 100 + YZ } in
  WXYZs.card = 20 := 
by
  sorry

end count_distinct_house_numbers_l351_351397


namespace typeA_selling_price_maximize_profit_l351_351948

theorem typeA_selling_price (sales_last_year : ℝ) (sales_increase_rate : ℝ) (price_increase : ℝ) 
                            (cars_sold_last_year : ℝ) : 
                            (sales_last_year = 32000) ∧ (sales_increase_rate = 1.25) ∧ 
                            (price_increase = 400) ∧ 
                            (sales_last_year / cars_sold_last_year = (sales_last_year * sales_increase_rate) / (cars_sold_last_year + price_increase)) → 
                            (cars_sold_last_year = 1600) :=
by
  sorry

theorem maximize_profit (typeA_price : ℝ) (typeB_price : ℝ) (typeA_cost : ℝ) (typeB_cost : ℝ) 
                        (total_cars : ℕ) :
                        (typeA_price = 2000) ∧ (typeB_price = 2400) ∧ 
                        (typeA_cost = 1100) ∧ (typeB_cost = 1400) ∧ 
                        (total_cars = 50) ∧ 
                        (∀ m : ℕ, m ≤ 50 / 3) → 
                        ∃ m : ℕ, (m = 17) ∧ (50 - m * 2 ≤ 33) :=
by
  sorry

end typeA_selling_price_maximize_profit_l351_351948


namespace only_event_B_is_random_l351_351104

def roll_dice_sum (x y : ℕ) : ℕ := x + y

def event_A (x y : ℕ) : Prop := roll_dice_sum x y = 1
def event_B (x y : ℕ) : Prop := roll_dice_sum x y = 6
def event_C (x y : ℕ) : Prop := roll_dice_sum x y > 12
def event_D (x y : ℕ) : Prop := roll_dice_sum x y < 13

def random_event (x y : ℕ) (e : (ℕ → ℕ → Prop)) : Prop :=
  match x, y with
  | x, y := e x y ∧ 1 <= x ∧ x <= 6 ∧ 1 <= y ∧ y <= 6

theorem only_event_B_is_random :
  ∀ x y : ℕ,
  (random_event x y event_A = false) ∧
  (random_event x y event_B = true) ∧
  (random_event x y event_C = false) ∧
  (random_event x y event_D = false) :=
by
  sorry

end only_event_B_is_random_l351_351104


namespace five_neg_two_l351_351179

theorem five_neg_two : 5^(-2) = (1 / 25) := 
by sorry

end five_neg_two_l351_351179


namespace find_larger_number_l351_351054

theorem find_larger_number (x y : ℝ) (h1 : x - y = 1860) (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 :=
by
  sorry

end find_larger_number_l351_351054


namespace gcd_75_100_l351_351813

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351813


namespace sum_of_coordinates_of_A_l351_351587

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351587


namespace cos_330_eq_sqrt3_over_2_l351_351209

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351209


namespace almost_perfect_number_is_square_of_odd_l351_351013

def is_sum_of_divisors (σ : ℕ → ℕ) (N : ℕ) : Prop :=
  σ N = (Finset.range (N + 1)).filter (λ d, d ≠ 0 ∧ N % d = 0).sum id

def is_almost_perfect_number (σ : ℕ → ℕ) (N : ℕ) : Prop :=
  σ N = 2 * N + 1

theorem almost_perfect_number_is_square_of_odd (σ : ℕ → ℕ) (N : ℕ) :
  is_sum_of_divisors σ N →
  is_almost_perfect_number σ N →
  ∃ M : ℕ, N = M * M ∧ M % 2 = 1 :=
sorry

end almost_perfect_number_is_square_of_odd_l351_351013


namespace product_remainder_mod_7_l351_351769

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351769


namespace sand_pile_paves_road_l351_351880

noncomputable def volume_cone (A h : ℝ) : ℝ :=
  (1/3) * A * h

noncomputable def length_road_paved (V w t : ℝ) : ℝ :=
  V / (w * t)

theorem sand_pile_paves_road :
  ∀ (A h w t : ℝ), A = 45.9 → h = 1.2 → w = 12 → t = 0.03 →
   length_road_paved (volume_cone A h) w t ≈ 11.5 :=
by
  intros A h w t A_eq h_eq w_eq t_eq
  unfold volume_cone
  unfold length_road_paved
  rw [A_eq, h_eq, w_eq, t_eq]
  sorry

end sand_pile_paves_road_l351_351880


namespace fg_of_minus_three_l351_351658

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x * x + 4

-- The theorem to prove
theorem fg_of_minus_three : f (g (-3)) = 25 := by
  sorry

end fg_of_minus_three_l351_351658


namespace marley_fruits_l351_351677

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l351_351677


namespace bertha_descendants_no_daughters_l351_351168

theorem bertha_descendants_no_daughters :
  let (d, s) := (7, 3)
  (some_children_have_3_daughters : ∃ x, 3 * x = 18) (remaining_children_have_no_daughters : ∃ y, y * 0 = 0) 
  (total_descendants_without_ggdaughters : 40 = 7 + 18 + 15)
  (no_great_great_granddaughters : 0 = 0)
  (answer := 28),
  d + s = 10 →
  total_descendants_without_ggdaughters →
  no_great_great_granddaughters →
  answer = (4 + 9 + 15) :=
by
  sorry

end bertha_descendants_no_daughters_l351_351168


namespace problem_l351_351003

-- Definitions
variables {a b : ℝ}
def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

-- Root condition using the given equation
def quadratic_eq (x : ℝ) : ℝ := (x - 3) * (2 * x + 7) - (x^2 - 11 * x + 28)

-- Statement to prove
theorem problem (ha : is_root quadratic_eq a) (hb : is_root quadratic_eq b) (h_distinct : a ≠ b):
  (a + 2) * (b + 2) = -66 :=
sorry

end problem_l351_351003


namespace inequality_I_inequality_II_inequality_III_l351_351479

variable {a b c x y z : ℝ}

-- Assume the conditions
def conditions (a b c x y z : ℝ) : Prop :=
  x^2 < a ∧ y^2 < b ∧ z^2 < c

-- Prove the first inequality
theorem inequality_I (h : conditions a b c x y z) : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a * b + b * c + c * a :=
sorry

-- Prove the second inequality
theorem inequality_II (h : conditions a b c x y z) : x^4 + y^4 + z^4 < a^2 + b^2 + c^2 :=
sorry

-- Prove the third inequality
theorem inequality_III (h : conditions a b c x y z) : x^2 * y^2 * z^2 < a * b * c :=
sorry

end inequality_I_inequality_II_inequality_III_l351_351479


namespace Marley_fruit_count_l351_351678

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l351_351678


namespace cream_ratio_Joe_JoAnn_l351_351638

def Joe_initial_coffee := 15
def Joe_drank_coffee := 3
def Joe_added_cream := 3

def JoAnn_initial_coffee := 15
def JoAnn_added_cream := 3
def JoAnn_drank_total := 3

theorem cream_ratio_Joe_JoAnn :
  let Joe_final_cream := Joe_added_cream,
      JoAnn_total_volume := JoAnn_initial_coffee + JoAnn_added_cream,
      JoAnn_cream_concentration := (JoAnn_added_cream : ℚ) / JoAnn_total_volume,
      JoAnn_drank_cream := (JoAnn_drank_total : ℚ) * JoAnn_cream_concentration,
      JoAnn_remaining_cream := (JoAnn_added_cream : ℚ) - JoAnn_drank_cream,
      Joe_cream_amount := Joe_final_cream,
      JoAnn_cream_amount := JoAnn_remaining_cream
  in (Joe_cream_amount : ℚ) / JoAnn_cream_amount = 6 / 5 :=
by
  sorry

end cream_ratio_Joe_JoAnn_l351_351638


namespace product_simplification_l351_351177

theorem product_simplification : 
  (∏ k in Finset.range 99, (1 - (1 / (k + 2)))) = (1 / 100) :=
by
  sorry

end product_simplification_l351_351177


namespace gcd_75_100_l351_351839

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351839


namespace cos_330_eq_sqrt3_div_2_l351_351316

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351316


namespace lindsey_savings_december_l351_351671

variables (s : ℕ → ℝ) -- s represents Lindsey's savings at the end of each month
def january_to_march : ℕ → ℝ
| 0 => 100
| 1 => 100
| 2 => 100

def from_april (n : ℕ) := 0.75 * s (n - 1)
def september_bonus := 0.25 * (s 7)
def november_expense (total : ℝ) := 0.45 * total
def december_savings (n : ℝ) := n + (1.5 * n)

-- Mathematical problem statement in Lean
theorem lindsey_savings_december :
  (s 12 = 417.83935546875) :=
by
  have s_0 := january_to_march 0,
  have s_1 := january_to_march 1,
  have s_2 := january_to_march 2,
  have s_3 := s_0 + s_1 + s_2,

  -- Savings from January to March
  have march_savings : s 3 = s_3,
  -- April to August using recurrence relation
  have april_savings : s 4 = s 3 + from_april 4,
  have may_savings : s 5 = s 4 + from_april 5,
  have june_savings : s 6 = s 5 + from_april 6,
  have july_savings : s 7 = s 6 + from_april 7,
  have august_savings : s 8 = s 7 + from_april 8,
  
  -- September with bonus
  have sept_bonus : s 9 = s 8 + september_bonus,
  
  -- October calculation following rule in September
  have oct_savings : s 10 = s 9 + from_april 10,

  -- November expense on laptop
  have nov_savings : s 11 = s 10 - november_expense (s 10),
  
  -- December savings
  have dec_savings_change : s 11 = s 10 * 0.75,
  have dec_savings : s 12 = s 11 + december_savings (s 11 * 0.75),

  -- Proof that the total savings at the end of December is 417.83935546875
  sorry

end lindsey_savings_december_l351_351671


namespace value_of_x_l351_351851

theorem value_of_x (x : ℝ) : (3 * 8^12 = 2^x) → x = Real.log 3 / Real.log 2 + 36 :=
by
  intro h
  have h1 : 8 = 2^3 := by norm_num
  have h2 : 8^12 = (2^3)^12 := by rw [h1]
  have h3 : (2^3)^12 = 2^(3*12) := by rw [Real.rpow_nat_cast 2 3, Real.rpow_nat_cast 2 12, Real.rpow_mul]
  rw [h3] at h2
  have h4 : 3 * 2^(3*12) = 3 * 2^36 := by norm_num
  rw [h2,h4] at h
  have h5 : 3 = 2^(Real.log 3 / Real.log 2) := by rw [←Real.rpow_nat_cast 3, Real.log_nat]
  rw [h5,Real.rpow_add] at h
  rw [Real.expmulInner] at h
  sorry

end value_of_x_l351_351851


namespace shooting_range_l351_351854

theorem shooting_range (initial_amount : ℝ) (final_amount : ℝ)
  (n : ℕ) (succ_shots : ℕ) (miss_shots : ℕ) :
  initial_amount = 100 ∧ final_amount = 80.19 ∧ succ_shots = 1 ∧ miss_shots = 3  →
  initial_amount * (1.1 ^ succ_shots) * (0.9 ^ miss_shots) = final_amount :=
begin
  intro h,
  have h1 : initial_amount = 100 := h.1,
  have h2 : final_amount = 80.19 := h.2.1,
  have h3 : succ_shots = 1 := h.2.2.1,
  have h4 : miss_shots = 3 := h.2.2.2,
  rw [h1, h2, h3, h4],
  norm_num,
end

end shooting_range_l351_351854


namespace cream_ratio_l351_351632

-- Define the initial conditions for Joe and JoAnn
def initial_coffee : ℕ := 15
def initial_cup_size : ℕ := 20
def cream_added : ℕ := 3
def coffee_drank_by_joe : ℕ := 3
def mixture_stirred_by_joann : ℕ := 3

-- Define the resulting amounts of cream in Joe and JoAnn's coffee
def cream_in_joe : ℕ := cream_added
def cream_in_joann : ℝ := cream_added - (cream_added * (mixture_stirred_by_joann / (initial_coffee + cream_added)))

-- Prove the ratio of the amount of cream in Joe's coffee to that in JoAnn's coffee
theorem cream_ratio :
  (cream_in_joe : ℝ) / cream_in_joann = 6 / 5 :=
by
  -- The code is just a statement; the proof detail is omitted with sorry, and variables are straightforward math.
  sorry

end cream_ratio_l351_351632


namespace cos_alpha_plus_pi_six_l351_351451

theorem cos_alpha_plus_pi_six (α : ℝ) (hα_in_interval : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 3 / 3) :
  Real.cos (α + π / 6) = (3 - Real.sqrt 6) / 6 := 
by
  sorry

end cos_alpha_plus_pi_six_l351_351451


namespace cos_330_eq_sqrt3_div_2_l351_351248

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351248


namespace product_mod_7_l351_351740

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351740


namespace solve_equation_l351_351715

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l351_351715


namespace foci_of_ellipse_l351_351725

theorem foci_of_ellipse (θ : Real) :
  ∃ c : Real, (\begin {cases} 
    x = 3 * Real.cos θ, 
    y = 5 * Real.sin θ 
  end) → c = 4 :=
sorry

end foci_of_ellipse_l351_351725


namespace distance_A_B_l351_351986

-- Defining the points A and B in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- The two points A and B
def A : Point3D := { x := 1, y := 5, z := 2 }
def B : Point3D := { x := 4, y := 1, z := 5 }

-- Distance formula in 3D
def distance (A B : Point3D) : ℝ := 
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2 + (B.z - A.z)^2)

-- Theorem to prove that the distance between points A and B is sqrt(34)
theorem distance_A_B : distance A B = Real.sqrt 34 := 
  by 
  sorry

end distance_A_B_l351_351986


namespace game_outcome_1010_game_outcome_1011_l351_351181

-- Declare the players and their strategies
inductive Player
| chris
| alex

-- Define their strategies in general terms
def strategy (p : Player) : ℕ → Prop
| Player.chris := λ n, n % 3 = 0 ∨ n % 5 = 0
| Player.alex := λ n, n % 1 = 0 ∨ n % 3 = 0

-- Define a function to determine the winner given a starting number of coins
def winner (n : ℕ) : Player :=
if n % 4 = 0 then Player.alex else Player.chris

theorem game_outcome_1010 :
  winner 1010 = Player.alex :=
by sorry

theorem game_outcome_1011 :
  winner 1011 = Player.chris :=
by sorry

end game_outcome_1010_game_outcome_1011_l351_351181


namespace cos_330_eq_sqrt3_div_2_l351_351311

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351311


namespace arithmetic_sequence_nth_term_l351_351058

theorem arithmetic_sequence_nth_term (x n : ℕ) (a1 a2 a3 : ℚ) (a_n : ℕ) :
  a1 = 3 * x - 5 ∧ a2 = 7 * x - 17 ∧ a3 = 4 * x + 3 ∧ a_n = 4033 →
  n = 641 :=
by sorry

end arithmetic_sequence_nth_term_l351_351058


namespace radius_of_circle_spherical_coords_l351_351786

theorem radius_of_circle_spherical_coords :
  ∀ (θ : ℝ), ∃ r : ℝ, r = √2 ∧ ∀ (ρ : ℝ) (ϕ : ℝ), ρ = 2 → ϕ = π / 4 →
    (let x := ρ * real.sin ϕ * real.cos θ in
     let y := ρ * real.sin ϕ * real.sin θ in
     x^2 + y^2 = r^2) :=
by
  sorry

end radius_of_circle_spherical_coords_l351_351786


namespace remainder_of_product_l351_351774

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351774


namespace expand_product_l351_351414

theorem expand_product (x : ℝ) :
  (x + 4) * (x - 5) = x^2 - x - 20 :=
by
  -- The proof will use algebraic identities and simplifications.
  sorry

end expand_product_l351_351414


namespace remainder_of_product_l351_351776

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351776


namespace circle_area_l351_351099

theorem circle_area (d : ℝ) (r : ℝ) (A : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : A = π * r^2) : 
  A = 25 * π :=
by
  -- Introduce variables and assumptions
  have h_radius : r = 5 := by
    rw [h2, h1]
    norm_num
  -- Introduce the calculation of area
  have h_area : A = π * (5^2) := by
    rw [h3, h_radius]
  -- Simplify the expression
  rw [h_area]
  norm_num  
  rfl

end circle_area_l351_351099


namespace mutually_exclusive_not_complementary_l351_351119

theorem mutually_exclusive_not_complementary :
  let balls := {b | b = "red" ∨ b = "black"}
  let pocket := {b ∈ balls | b = "red"} ∪ {b ∈ balls | b = "black"}
  let draw_two := ({x : Set String | Finset.card x = 2} : Finset (Set String))
  -- Define event C1: Having exactly one black ball
  let event_C1 := {x ∈ draw_two | Finset.card (x ∩ {"black"}) = 1}
  -- Define event C2: Having exactly two red balls
  let event_C2 := {x ∈ draw_two | x = {"red", "red"}}
  -- Prove the events are mutually exclusive but not complementary
  (event_C1 ∩ event_C2 = ∅) ∧ (event_C1 ∪ event_C2 ≠ draw_two) :=
by
  -- Event definitions
  sorry

end mutually_exclusive_not_complementary_l351_351119


namespace cos_330_eq_sqrt3_div_2_l351_351332

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351332


namespace sums_of_coordinates_of_A_l351_351616

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351616


namespace rationalize_denominator_sum_of_coefficients_l351_351704
open Real

theorem rationalize_denominator :
  (2 : ℝ) / (3 * sqrt 7 + 2 * sqrt 13) = (6 * sqrt 7 - 4 * sqrt 13) / 11 :=
by sorry

theorem sum_of_coefficients :
  let A := 6
  let B := 7
  let C := -4
  let D := 13
  let E := 11
  A + B + C + D + E = 33 :=
by {
  have hA : A = 6 := rfl
  have hB : B = 7 := rfl
  have hC : C = -4 := rfl
  have hD : D = 13 := rfl
  have hE : E = 11 := rfl
  calc A + B + C + D + E
      = 6 + 7 + (-4) + 13 + 11 : by rw [hA, hB, hC, hD, hE]
  ... = 33 : by norm_num
}

end rationalize_denominator_sum_of_coefficients_l351_351704


namespace problem_l351_351703

theorem problem
  (circle : Type)
  (A B C D : circle)
  (inscribed : InscribedQuadrilateral circle A B C D)
  (angle_BAC : ∠ B A C = 80)
  (angle_ADB : ∠ A D B = 35)
  (AD BC : ℝ)
  (AD_equals_5 : AD = 5)
  (BC_equals_7 : BC = 7):
  length (segment A C) = 7 * (real.sin 65) / (real.sin 80) :=
sorry

end problem_l351_351703


namespace cos_330_eq_sqrt3_div_2_l351_351273

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351273


namespace area_of_triangle_ABC_l351_351736

variables (A B C : ℝ × ℝ)

-- Define point A
def point_A := (3, 4 : ℝ)

-- Define reflection over y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define reflection over the line y = -x
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define points B and C using the reflections
def point_B := reflect_y_axis point_A
def point_C := reflect_y_eq_neg_x point_B

-- Function to calculate the area of a triangle given vertices A, B, and C
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Proof statement
theorem area_of_triangle_ABC : triangle_area point_A point_B point_C = 3 :=
by
  sorry

end area_of_triangle_ABC_l351_351736


namespace distance_between_foci_of_given_ellipse_l351_351902

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351902


namespace cos_330_eq_sqrt3_div_2_l351_351243

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351243


namespace sums_of_coordinates_of_A_l351_351609

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351609


namespace magnitude_of_complex_number_l351_351410

noncomputable def complex_magnitude (c : ℂ) : ℝ :=
  complex.abs c

theorem magnitude_of_complex_number :
  complex_magnitude (7/4 - 3*complex.I) = real.sqrt 193 / 4 := 
by 
  sorry

end magnitude_of_complex_number_l351_351410


namespace symmetric_point_y_axis_l351_351051

theorem symmetric_point_y_axis (A : ℝ × ℝ) (hA : A = (2, -1)) :
  ∃ B : ℝ × ℝ, B = (-2, -1) ∧ (B.1 = -A.1 ∧ B.2 = A.2) :=
by
  use (-2, -1)
  split
  · rfl
  · rw [hA]
    simp

end symmetric_point_y_axis_l351_351051


namespace pure_imaginary_condition_l351_351515

theorem pure_imaginary_condition (m : ℝ) (h : (m^2 - 3 * m) = 0) : (m = 0) :=
by
  sorry

end pure_imaginary_condition_l351_351515


namespace calculation_is_correct_l351_351950

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l351_351950


namespace sums_of_coordinates_of_A_l351_351613

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351613


namespace value_of_expression_when_x_is_3_l351_351176

theorem value_of_expression_when_x_is_3 : 
  let x := 3 in x + x * (x^3 - x) = 75 := 
by 
  sorry

end value_of_expression_when_x_is_3_l351_351176


namespace cos_330_eq_sqrt3_over_2_l351_351204

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351204


namespace circle_area_l351_351098

theorem circle_area (d : ℝ) (r : ℝ) (A : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : A = π * r^2) : 
  A = 25 * π :=
by
  -- Introduce variables and assumptions
  have h_radius : r = 5 := by
    rw [h2, h1]
    norm_num
  -- Introduce the calculation of area
  have h_area : A = π * (5^2) := by
    rw [h3, h_radius]
  -- Simplify the expression
  rw [h_area]
  norm_num  
  rfl

end circle_area_l351_351098


namespace gcd_75_100_l351_351825

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351825


namespace calc_hash_80_l351_351965

def hash (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem calc_hash_80 : hash (hash (hash 80)) = 17.28 :=
by 
  sorry

end calc_hash_80_l351_351965


namespace sum_of_coordinates_A_l351_351561

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351561


namespace remainder_of_product_mod_7_l351_351754

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351754


namespace find_varphi_l351_351059

theorem find_varphi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π)
(h_symm : ∃ k : ℤ, ϕ = k * π + 2 * π / 3) :
ϕ = 2 * π / 3 :=
sorry

end find_varphi_l351_351059


namespace range_of_x_l351_351494

theorem range_of_x (x : ℝ) (M : Set ℝ) (hM : M = {x ^ 2, 1}) : x ≠ 1 ∧ x ≠ -1 := by
  sorry

end range_of_x_l351_351494


namespace cos_330_eq_sqrt_3_div_2_l351_351366

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351366


namespace smallest_possible_norm_l351_351669

noncomputable def vector_u := ℝ × ℝ

noncomputable def condition (u : vector_u) : Prop :=
  ‖(u.1 + 5, u.2 + 2)‖ = 10

noncomputable def answer : ℝ :=
  10 - real.sqrt 29

theorem smallest_possible_norm (u : vector_u) (h : condition u) :
  ‖u‖ = answer := 
sorry

end smallest_possible_norm_l351_351669


namespace sum_of_squares_of_cosines_l351_351178

theorem sum_of_squares_of_cosines : 
  (∑ k in finset.range 16, real.cos ((k + 1) * real.pi / 17) ^ 2) = 15 / 2 :=
sorry

end sum_of_squares_of_cosines_l351_351178


namespace cos_330_eq_sqrt_3_div_2_l351_351226

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351226


namespace greatest_product_of_slopes_of_intersecting_lines_l351_351803

theorem greatest_product_of_slopes_of_intersecting_lines :
  ∃ m1 m2 : ℝ, (tan (real.pi / 6) = abs ((m2 - m1) / (1 + m1 * m2))) ∧ (m2 = 4 * m1) ∧
    m1 * m2 = (38 - 6 * real.sqrt 33) / 16 :=
begin
  sorry
end

end greatest_product_of_slopes_of_intersecting_lines_l351_351803


namespace total_sum_120_l351_351707

def is_adjacent (a b : ℕ × ℕ) : Prop :=
(abs a.1 - b.1 + abs a.2 - b.2 = 1)

def write_number (grid : array (fin 6) (array (fin 6) (option ℕ))) (pos : ℕ × ℕ) : ℕ :=
grid.foldl (λ acc i, acc + (grid[i.1][i.2].is_some && is_adjacent i pos ? 1 : 0)) 0

noncomputable def total_sum (color_order : list (ℕ × ℕ)) : ℕ :=
(color_order.foldl (λ grid pos,
  let n := write_number grid pos in
  grid.update_nth pos (some n))
  (array.mk [array.mk [none] (fin 6) val := 6]))).foldl (λ acc i, acc + (i.val.foldl (λ acc j, acc + j.val.get_or_else 0) 0)) 0

theorem total_sum_120 (color_order : list (ℕ × ℕ)) : total_sum color_order = 120 := by
  sorry

end total_sum_120_l351_351707


namespace cos_330_eq_sqrt3_div_2_l351_351270

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351270


namespace cos_330_eq_sqrt3_div_2_l351_351255

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351255


namespace cos_330_eq_sqrt_3_div_2_l351_351224

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351224


namespace average_output_l351_351161

theorem average_output (t1 t2: ℝ) (cogs1 cogs2 : ℕ) (h1 : t1 = cogs1 / 36) (h2 : t2 = cogs2 / 60) (h_sum_cogs : cogs1 = 60) (h_sum_more_cogs : cogs2 = 60) (h_sum_time : t1 + t2 = 60 / 36 + 60 / 60) : 
  (cogs1 + cogs2) / (t1 + t2) = 45 := by
  sorry

end average_output_l351_351161


namespace ellipse_equation_and_slopes_product_l351_351464

theorem ellipse_equation_and_slopes_product (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : 2 * a = 2 * sqrt 3) (h4 : c / a = sqrt 3 / 3) (h5 : a^2 = b^2 + c^2) :
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∀ y0 y1 x1 : ℝ, 3 ∙ x1 ≠ 0 → 
     let k_pq := y1 / x1 in
     let k_oq := (y1 - y0) / (x1 - 3) in
     k_pq * k_oq = -2 / 3) :=
sorry

end ellipse_equation_and_slopes_product_l351_351464


namespace trucks_rented_out_l351_351805

theorem trucks_rented_out
  (T : ℕ)
  (H1 : T = 30)
  (H2 : ∀ R : ℕ, R ≤ T ∧ (T - 15) ≤ 0.40 * R) :
  ∃ R : ℕ, R = 37 := by
  sorry

end trucks_rented_out_l351_351805


namespace complement_of_M_is_correct_l351_351523

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | real.log10 (x - 1) < 0 }
def complement_of_M := { x | x ≤ 1 } ∪ { x | x ≥ 2 }

theorem complement_of_M_is_correct :
  (U \ M) = complement_of_M := by
  sorry

end complement_of_M_is_correct_l351_351523


namespace cos_330_eq_sqrt3_div_2_l351_351348

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351348


namespace find_integers_k_l351_351657

theorem find_integers_k (n : ℕ) (cn : fin n → ℝ) (h_n : n ≥ 2) 
      (h_sum : 0 ≤ ∑ i, cn i ∧ ∑ i, cn i ≤ n) :
  ∃ k : fin n → ℤ, (∑ i, k i = 0) ∧ (∀ i, 1 - (n : ℝ) ≤ cn i + (n : ℝ) * (k i) ∧ cn i + (n : ℝ) * (k i) ≤ n) := 
sorry

end find_integers_k_l351_351657


namespace radius_of_circle_spherical_coords_l351_351784

theorem radius_of_circle_spherical_coords :
  ∀ (θ : ℝ), ∃ r : ℝ, r = √2 ∧ ∀ (ρ : ℝ) (ϕ : ℝ), ρ = 2 → ϕ = π / 4 →
    (let x := ρ * real.sin ϕ * real.cos θ in
     let y := ρ * real.sin ϕ * real.sin θ in
     x^2 + y^2 = r^2) :=
by
  sorry

end radius_of_circle_spherical_coords_l351_351784


namespace sum_of_coordinates_A_l351_351556

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351556


namespace petya_password_count_l351_351696

theorem petya_password_count :
  let digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  in (∑ d1 in digits, ∑ d2 in digits, ∑ d3 in digits, ∑ d4 in digits,
        (if d1 = d2 ∨ d1 = d3 ∨ d1 = d4 ∨ d2 = d3 ∨ d2 = d4 ∨ d3 = d4 then 1 else 0))
     = 3537 :=
by
  sorry

end petya_password_count_l351_351696


namespace school_to_park_distance_l351_351694

theorem school_to_park_distance : 
  ∃ (m n : ℕ), Nat.Coprime m n ∧ ((m : ℚ) / (n : ℚ)) = (let t := 14 / 5 in let v := 18 / 5 in v * t) ∧ (m + n = 277) := 
by 
  sorry

end school_to_park_distance_l351_351694


namespace ellipse_foci_distance_l351_351923

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351923


namespace sum_of_c_l351_351994

def is_rational_root (a b c n : ℤ) : Prop :=
  let discriminant := b * b - 4 * a * c in
  ∃ k: ℤ, discriminant = k * k

def problem (c : ℤ) : Prop :=
  c ≤ 100 ∧ is_rational_root 1 (-6) (-c) (36 + 4 * c)

theorem sum_of_c : 
  ∑ c in { c : ℤ | problem c }.toFinset, c = 308 :=
by
  sorry

end sum_of_c_l351_351994


namespace remainder_of_product_l351_351771

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351771


namespace compute_expr_l351_351956

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l351_351956


namespace distance_between_foci_of_ellipse_l351_351927

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351927


namespace max_diagonals_no_perpendicular_l351_351847

-- Define the concept of a regular n-gon and its properties
def regular_polygon (n : ℕ) := { vertices : finset ℝ // vertices.card = n }

-- Define the number of diagonals function for an n-gon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the concept of perpendicular diagonals within a regular n-gon
def perpendicular_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2 / 2

-- Define the maximum number of non-perpendicular diagonals
def max_non_perpendicular_diagonals (n : ℕ) : ℕ :=
  num_diagonals n - perpendicular_diagonals n

-- Main theorem to be proved
theorem max_diagonals_no_perpendicular (n : ℕ) (h : n = 12) : 
  max_non_perpendicular_diagonals n = 24 :=
begin
  rw h,
  sorry
end

end max_diagonals_no_perpendicular_l351_351847


namespace roof_length_width_difference_l351_351070

theorem roof_length_width_difference
  {w l : ℝ} 
  (h_area : l * w = 576) 
  (h_length : l = 4 * w) 
  (hw_pos : w > 0) :
  l - w = 36 :=
by 
  sorry

end roof_length_width_difference_l351_351070


namespace volume_pyramid_eq_20sqrt489_l351_351026

def volume_of_pyramid (AB AD PA PC : ℝ) : ℝ :=
  let area_base := AB * AD
  let height := PA
  (1 / 3) * area_base * height

theorem volume_pyramid_eq_20sqrt489 :
  ∀ (AB AD PA PC : ℝ),
  AB = 10 → AD = 6 → PC = 25 → PA = Real.sqrt (25^2 - 10^2) →
  volume_of_pyramid AB AD PA PC = 20 * Real.sqrt 489 :=
by
  intros AB AD PA PC h_AB h_AD h_PC h_PA
  have h_base_area : AB * AD = 60 := by
    rw [h_AB, h_AD]
    norm_num
  have h_height : PA = Real.sqrt (25^2 - 10^2) := h_PA
  have volume : volume_of_pyramid AB AD PA PC = (1 / 3) * 60 * Real.sqrt 489 := by
    rw [volume_of_pyramid, h_base_area, h_height]
    norm_num
  exact volume

end volume_pyramid_eq_20sqrt489_l351_351026


namespace petya_password_count_l351_351697

theorem petya_password_count : 
  let all_digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  let total_passwords := 9^4
  let choose_4 := Nat.choose 9 4
  let arrange_4 := factorial 4
  let different_digits := choose_4 * arrange_4
  let passwords_with_identical_digits := total_passwords - different_digits
  in passwords_with_identical_digits = 3537 := by
  -- Definitions
  let all_digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  let total_passwords := 9^4
  let choose_4 := Nat.choose 9 4
  let arrange_4 := factorial 4
  let different_digits := choose_4 * arrange_4
  let passwords_with_identical_digits := total_passwords - different_digits
  -- Proof (to be filled in)
  sorry

end petya_password_count_l351_351697


namespace cos_330_eq_sqrt3_over_2_l351_351203

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351203


namespace cos_330_eq_sqrt3_over_2_l351_351198

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351198


namespace sum_coordinates_A_l351_351593

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351593


namespace sum_of_first_10_common_elements_l351_351429

noncomputable def sum_first_common_elements : ℕ :=
  let common_elements := (20 : ℕ) :: (80 : ℕ) :: (320 : ℕ) :: (1280 : ℕ) :: (5120 : ℕ) ::
                        (20480 : ℕ) :: (81920 : ℕ) :: (327680 : ℕ) :: (1310720 : ℕ) :: (5242880 : ℕ) :: []
  in common_elements.sum

theorem sum_of_first_10_common_elements :
  sum_first_common_elements = 6990500 :=
by
  -- Insert mathematical proof here
  sorry

end sum_of_first_10_common_elements_l351_351429


namespace OC_eq_l351_351452

variable {V : Type} [AddCommGroup V]

-- Given vectors a and b
variables (a b : V)

-- Conditions given in the problem
def OA := a + b
def AB := 3 • (a - b)
def CB := 2 • a + b

-- Prove that OC = 2a - 3b
theorem OC_eq : (a + b) + (3 • (a - b)) + (- (2 • a + b)) = 2 • a - 3 • b :=
by
  -- write your proof here
  sorry

end OC_eq_l351_351452


namespace gcd_75_100_l351_351824

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351824


namespace hyperbola_s_squared_l351_351884

theorem hyperbola_s_squared 
  (s : ℝ) 
  (h1 : ∃ (b : ℝ) (b > 0), ∀ (x y : ℝ), (x, y) ≠ (3, 0) → x^2 / 9 - y^2 / b^2 = 1)
  (h2 : ∀ (x y : ℝ), (x, y) = (5, -6) → x^2 / 9 - y^2 / (81/4) = 1)
  (h3 : ∀ x y : ℝ, (x, y) = (s, -3) → x^2 / 9 - 4*y^2 / 81 = 1)
  : s^2 = 12 :=
sorry

end hyperbola_s_squared_l351_351884


namespace cos_330_eq_sqrt3_over_2_l351_351199

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351199


namespace proposition_3_is_false_l351_351482

variables (m l n : Line) (α β : Plane) (A : Point)

-- Condition definitions
def cond1 : Prop := m ⊂ α ∧ l ∩ α = A ∧ A ∉ m → ¬coplanar l m
def cond2 : Prop := skew l m ∧ l ∥ α ∧ m ∥ α ∧ n ⟂ l ∧ n ⟂ m → n ⟂ α
def cond3 : Prop := l ∥ α ∧ m ∥ β ∧ α ∥ β → l ⟂ m
def cond4 : Prop := l ⊂ α ∧ m ⊂ α ∧ l ∩ m = A ∧ l ∥ β ∧ m ∥ β → α ∥ β

-- Equivalent proof problem statement
theorem proposition_3_is_false (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : ¬ cond3 := 
begin
    sorry -- placeholder for the proof
end

end proposition_3_is_false_l351_351482


namespace correct_expression_l351_351900

-- Definitions for the problem options.
def optionA (m n : ℕ) : ℕ := 2 * m + n
def optionB (m n : ℕ) : ℕ := m + 2 * n
def optionC (m n : ℕ) : ℕ := 2 * (m + n)
def optionD (m n : ℕ) : ℕ := (m + n) ^ 2

-- Statement for the proof problem.
theorem correct_expression (m n : ℕ) : optionB m n = m + 2 * n :=
by sorry

end correct_expression_l351_351900


namespace cos_330_cos_30_val_answer_l351_351233

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351233


namespace max_liars_5x5_grid_l351_351533

theorem max_liars_5x5_grid : ∃ m : ℕ, m = 13 ∧
  ∀ (grid : ℕ × ℕ → bool), 
  (∀ i j, (0 ≤ i) ∧ (i < 5) ∧ (0 ≤ j) ∧ (j < 5) → 
    (grid (i, j) = tt → 
      (∃ k l, (abs (i - k) + abs (j - l) = 1) ∧ 
        (0 ≤ k) ∧ (k < 5) ∧ (0 ≤ l) ∧ (l < 5) ∧ grid (k, l) = ff))) →
  m = ∑ i j, if grid (i, j) = tt then 1 else 0 :=
begin
  sorry
end

end max_liars_5x5_grid_l351_351533


namespace exp_value_l351_351448

theorem exp_value (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2 * n) = 18 := 
by
  sorry

end exp_value_l351_351448


namespace equilateral_triangle_l351_351002

open Complex

noncomputable def λ := (1 + Real.sqrt 33) / 2

theorem equilateral_triangle (ω : ℂ) (hω : abs ω = 3) (hλ : λ > 1) :
  ∃ λ, (λ = (1 + Real.sqrt 33) / 2) ∧ (abs ω = 3 ∧ λ > 1) :=
by
  use λ
  split
  { refl }
  split
  { exact hω }
  { exact hλ }

end equilateral_triangle_l351_351002


namespace sum_coordinates_A_l351_351596

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351596


namespace sum_of_coordinates_A_l351_351562

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351562


namespace max_number_of_liars_l351_351531

noncomputable def max_liars (grid : Fin 5 × Fin 5 → Prop) : ℕ :=
  ∑ i in Finset.univ, if grid i = liar then 1 else 0

theorem max_number_of_liars : ∃ grid : (Fin 5 × Fin 5 → Prop), 
  (∀ i, grid i = liar → (∃ j, j ≠ i ∧ adjacent i j ∧ grid j ≠ liar)) ∧ 
  max_liars grid = 13 :=
begin
  sorry
end

-- A function to describe adjacency condition in 5×5 grid
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ a.2 = b.2) ∨ (abs (a.2 - b.2) = 1 ∧ a.1 = b.1)

-- Define liar and knight
inductive Person
| liar : Person
| knight : Person

open Person

end max_number_of_liars_l351_351531


namespace gcd_75_100_l351_351827

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351827


namespace product_remainder_mod_7_l351_351768

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351768


namespace remainder_product_l351_351747

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351747


namespace base_satisfying_eq_l351_351995

theorem base_satisfying_eq : ∃ a : ℕ, (11 < a) ∧ (293 * a^2 + 9 * a + 3 + (4 * a^2 + 6 * a + 8) = 7 * a^2 + 3 * a + 11) ∧ (a = 12) :=
by
  sorry

end base_satisfying_eq_l351_351995


namespace cost_of_each_top_l351_351935

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l351_351935


namespace cos_330_eq_sqrt_3_div_2_l351_351222

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351222


namespace real_values_of_c_l351_351390

theorem real_values_of_c : ∃ cs : Finset ℝ, cs.card = 2 ∧ ∀ c ∈ cs, |(1/3 : ℂ) - c * Complex.I| = 1/2 := 
by
  -- conditions and statement here
  sorry

end real_values_of_c_l351_351390


namespace sophia_purchase_cost_l351_351719

-- Define the variables involved
variables (f g p φ : ℝ) -- prices of fruit, bunches of grapes, pineapple, and pack of figs, respectively

-- Conditions provided in the problem
def condition1 : Prop := 3 * f + 2 * g + p + φ = 36
def condition2 : Prop := φ = 3 * f
def condition3 : Prop := p = f + g

-- The statement we need to prove
theorem sophia_purchase_cost :
  condition1 f g p φ ∧ condition2 f g p φ ∧ condition3 f g p φ →
  2 * g + p = (15 * g + 36) / 7 :=
by
  sorry

end sophia_purchase_cost_l351_351719


namespace seq_div_by_k_l351_351462

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + a (nat.floor (real.sqrt n))

theorem seq_div_by_k (a : ℕ → ℕ) :
  seq a →
  ∀ k : ℕ, k > 0 → ∃ n : ℕ, a n % k = 0 :=
by
  intro h_seq k h_k
  sorry

end seq_div_by_k_l351_351462


namespace sum_of_coordinates_of_A_l351_351589

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351589


namespace sum_first_10_common_elements_l351_351426

noncomputable def arithmetic_progression (n : ℕ) : ℕ :=
  5 + 3 * n

noncomputable def geometric_progression (k : ℕ) : ℕ :=
  10 * (2 ^ k)

theorem sum_first_10_common_elements : 
  (finset.sum (finset.range 10) (λ i, 20 * (4^i))) = 6990500 :=
by
  sorry

end sum_first_10_common_elements_l351_351426


namespace part_I_solution_part_II_solution_l351_351491

open real

def inequality_1 (x : ℝ) : Prop := 2*abs(x-3) + abs(x-4) < 2
def solution_set_1 : set ℝ := {x | 8/3 < x ∧ x < 4}

theorem part_I_solution : 
  {x : ℝ | inequality_1 x} = solution_set_1 :=
by
  -- The proof would go here.
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 4 then 3*x - 10
  else if 3 < x ∧ x < 4 then x - 2
  else 10 - 3*x

def inequality_2 (x : ℝ) (a : ℝ) : Prop := 2*abs(x-3) + abs(x-4) < 2*a
def non_empty_solution_set (a : ℝ) : Prop := (∃ x : ℝ, inequality_2 x a)

theorem part_II_solution {a : ℝ} : 
  (non_empty_solution_set a) ↔ (a > 1/2) :=
by
  -- The proof would go here.
  sorry

end part_I_solution_part_II_solution_l351_351491


namespace distance_between_foci_of_given_ellipse_l351_351906

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351906


namespace cos_330_eq_sqrt3_div_2_l351_351246

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351246


namespace test_subjects_count_l351_351893

theorem test_subjects_count 
  (n k : ℕ)
  (hk : k = 4)
  (hn : n = 8)
  (unidentified : ℕ)
  (hunidentified : unidentified = 35) :
  let num_combinations := Nat.choose n k in
  num_combinations + unidentified = 105 :=
by
  have hnum_combinations : Nat.choose n k = 70,
  { rw [hn, hk],
    norm_num,
    exact Nat.choose_eq_70},
  rw [hnum_combinations, hunidentified],
  norm_num

end test_subjects_count_l351_351893


namespace sum_of_coordinates_A_l351_351606

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351606


namespace compute_expr_l351_351957

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l351_351957


namespace range_of_s_l351_351488

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |Real.log x| else 2 * x + 3
  
theorem range_of_s (m : ℝ) (h_m : 0 < m ∧ m ≤ 3) :
  ∃ x1 x2 x3 : ℝ,
    (2 * x1 + 3 = m) ∧
    (Real.log x2 = -m) ∧
    (x2 * x3 = 1) ∧
    (x1 + x2 * x3 ∈ Icc (-1/2 : ℝ) 1) :=
sorry

end range_of_s_l351_351488


namespace product_remainder_mod_7_l351_351767

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351767


namespace cos_330_eq_sqrt3_div_2_l351_351254

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351254


namespace cos_330_cos_30_val_answer_l351_351234

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351234


namespace aunt_wang_time_after_10_paper_cuts_l351_351947

def cuttingTime (cuts : Nat) : Nat := (cuts - 1) * 4 + 3

def startTime : Time := ⟨9, 40⟩

theorem aunt_wang_time_after_10_paper_cuts :
  let final_time := startTime + (cuttingTime 10)
  final_time = ⟨10, 19⟩ :=
by
  sorry

end aunt_wang_time_after_10_paper_cuts_l351_351947


namespace pq_conjunction_false_perp_condition_false_contrapositive_correct_l351_351391

section Proposition1

variable {x : ℝ}

def p := ∃ x : ℝ, Real.tan x = 1
def q := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_conjunction_false : ¬ (p ∧ ¬q) :=
by { sorry }

end Proposition1

section PerpendicularLines

variables {a b : ℝ}

def l1 (a : ℝ) := λ x y : ℝ, a * x + 3 * y - 1
def l2 (b : ℝ) := λ x y : ℝ, x + b * y + 1

theorem perp_condition_false : ¬ (∀ a b : ℝ, (a / b = -3) ↔ (l1 a ⟂ l2 b)) :=
by { sorry }

end PerpendicularLines

section Contrapositive

variable {x : ℝ}

theorem contrapositive_correct :
  (x^2 - 3*x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by { sorry }

end Contrapositive

end pq_conjunction_false_perp_condition_false_contrapositive_correct_l351_351391


namespace cos_330_eq_sqrt_3_div_2_l351_351364

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351364


namespace cos_330_eq_sqrt3_div_2_l351_351196

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351196


namespace greatest_k_for_quadratic_roots_diff_l351_351071

theorem greatest_k_for_quadratic_roots_diff (k : ℝ)
  (H : ∀ x: ℝ, (x^2 + k * x + 8 = 0) → (∃ a b : ℝ, a ≠ b ∧ (a - b)^2 = 84)) :
  k = 2 * Real.sqrt 29 :=
by
  sorry

end greatest_k_for_quadratic_roots_diff_l351_351071


namespace cos_330_is_sqrt3_over_2_l351_351302

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351302


namespace division_addition_problem_l351_351952

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l351_351952


namespace car_speed_l351_351944

theorem car_speed
  (gasoline_per_mile : ℝ)
  (full_tank : ℝ)
  (hours_travelled : ℝ)
  (fraction_of_tank_used : ℝ)
  (amount_of_gasoline_used : ℝ)
  (distance_travelled : ℝ)
  (speed : ℝ) :
  gasoline_per_mile = 1 / 30 →
  full_tank = 12 →
  hours_travelled = 5 →
  fraction_of_tank_used = 0.8333333333333334 →
  amount_of_gasoline_used = fraction_of_tank_used * full_tank →
  distance_travelled = amount_of_gasoline_used / gasoline_per_mile →
  speed = distance_travelled / hours_travelled →
  speed = 60 :=
begin
  sorry
end

end car_speed_l351_351944


namespace cos_330_eq_sqrt_3_div_2_l351_351221

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351221


namespace greatest_common_divisor_456_108_lt_60_l351_351844

theorem greatest_common_divisor_456_108_lt_60 : 
  let divisors_456 := {d : ℕ | d ∣ 456}
  let divisors_108 := {d : ℕ | d ∣ 108}
  let common_divisors := divisors_456 ∩ divisors_108
  let common_divisors_lt_60 := {d ∈ common_divisors | d < 60}
  ∃ d, d ∈ common_divisors_lt_60 ∧ ∀ e ∈ common_divisors_lt_60, e ≤ d ∧ d = 12 := by {
    sorry
  }

end greatest_common_divisor_456_108_lt_60_l351_351844


namespace right_triangle_angles_l351_351057

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : 
  (α = 58.5 ∧ β = 31.5) :=
by sorry

end right_triangle_angles_l351_351057


namespace new_ratio_l351_351737

theorem new_ratio (J: ℝ) (F: ℝ) (F_new: ℝ): 
  J = 59.99999999999997 → 
  F / J = 3 / 2 → 
  F_new = F + 10 → 
  F_new / J = 5 / 3 :=
by
  intros hJ hF hF_new
  sorry

end new_ratio_l351_351737


namespace part_one_part_two_l351_351453

open Real

-- Definitions of circle C and line l
def Circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def Line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Fixed point P
def FixedPoint (x y : ℝ) : Prop := x = 1 ∧ y = 1

-- Chord AB and ratio condition
def ChordDivision (x1 y1 x2 y2 : ℝ) : Prop :=
  FixedPoint (1,1) → (1 - x1) = (1/2) * (x2 - 1)

-- Main theorems
theorem part_one (m : ℝ) : ∀ (x y : ℝ), Circle x y → Line m x y → x ≠ y ∧ y ≠ m := by sorry

theorem part_two (m x y : ℝ) : 
  FixedPoint 1 1 ∧ ChordDivision 1 1 ((3 - 2 * 1) / (1 + m^2)) ((m^2 - 3) / (1 + m^2)) →
  Line (1,1) (x - y = 0 ∨ x + y - 2 = 0) := by sorry

end part_one_part_two_l351_351453


namespace inscribed_circle_radius_l351_351708

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (sector_angle : ℝ) (circle_radius : ℝ) (tangent_points : ℕ),
    sector_angle = 120 ∧ circle_radius = 5 ∧ tangent_points = 3 → 
    r = 5 * (Real.sqrt 3 - 1) / 2 :=
by
  sorry

end inscribed_circle_radius_l351_351708


namespace distance_between_foci_of_ellipse_l351_351931

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351931


namespace cos_330_is_sqrt3_over_2_l351_351292

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351292


namespace cos_330_eq_sqrt_3_div_2_l351_351372

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351372


namespace parking_spaces_remaining_l351_351792

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l351_351792


namespace solve_quadratic_equation_l351_351718

theorem solve_quadratic_equation (m : ℝ) : 9 * m^2 - (2 * m + 1)^2 = 0 → m = 1 ∨ m = -1/5 :=
by
  intro h
  sorry

end solve_quadratic_equation_l351_351718


namespace cos_330_eq_sqrt3_div_2_l351_351360

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351360


namespace sum_first_10_common_elements_l351_351427

noncomputable def arithmetic_progression (n : ℕ) : ℕ :=
  5 + 3 * n

noncomputable def geometric_progression (k : ℕ) : ℕ :=
  10 * (2 ^ k)

theorem sum_first_10_common_elements : 
  (finset.sum (finset.range 10) (λ i, 20 * (4^i))) = 6990500 :=
by
  sorry

end sum_first_10_common_elements_l351_351427


namespace DiameterCond_l351_351115

open Real

-- Define the circle and points on the circle
variable {O : Point}
variable {P : ℕ → Point}
variable {n : ℕ}

-- Assume the conditions
axiom OnCircle (Γ : Circle) : ∀ i : ℕ, IsOnCircle (P i) Γ
axiom AngleDivision (h : n > 2)
    : ∀ i : ℕ, 2 ≤ i ∧ i ≤ n → ∠ P 1 O (P (n+1)) / n = ∠ P i O (P i+1)

-- Define the proof statement
theorem DiameterCond
    (P1Pn1Diameter : Diameter (P 1) (P (n+1)))
    (Gamma : Circle O)
    : ∀ i : ℕ, 2 ≤ i ∧ i ≤ n →
      (Distance O (P 2) + Distance O (P n)) 
      - (Distance O (P 1) + Distance O (P (n+1))) 
      = 
      4 * (sin^2 (π / (4 * n))) * (∑ i in (range (n-1)), Distance O (P (i+2)))  :=
by sorry

end DiameterCond_l351_351115


namespace cos_330_eq_sqrt_3_div_2_l351_351220

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351220


namespace smallest_square_area_l351_351620

variable (M N : ℝ)

/-- Given that the largest square has an area of 1 cm^2, the middle square has an area M cm^2, and the smallest square has a vertex on the side of the middle square, prove that the area of the smallest square N is equal to ((1 - M) / 2)^2. -/
theorem smallest_square_area (h1 : 1 ≥ 0)
  (h2 : 0 ≤ M ∧ M ≤ 1)
  (h3 : 0 ≤ N) :
  N = (1 - M) ^ 2 / 4 := sorry

end smallest_square_area_l351_351620


namespace opponent_choice_is_random_l351_351621

-- Define the possible outcomes in the game
inductive Outcome
| rock
| paper
| scissors

-- Defining the opponent's choice set
def opponent_choice := {outcome : Outcome | outcome = Outcome.rock ∨ outcome = Outcome.paper ∨ outcome = Outcome.scissors}

-- The event where the opponent chooses "scissors"
def event_opponent_chooses_scissors := Outcome.scissors ∈ opponent_choice

-- Proving that the event of opponent choosing "scissors" is a random event
theorem opponent_choice_is_random : ¬(∀outcome ∈ opponent_choice, outcome = Outcome.scissors) ∧ (∃ outcome ∈ opponent_choice, outcome = Outcome.scissors) → event_opponent_chooses_scissors := 
sorry

end opponent_choice_is_random_l351_351621


namespace cos_330_eq_sqrt3_div_2_l351_351358

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351358


namespace line_intersects_circle_l351_351067

open Real

def circle_center : ℝ × ℝ := (3/2, 0)
def circle_radius : ℝ := sqrt (13/4)

def line (k x : ℝ) : ℝ := k * (x - 1)

noncomputable def distance_from_point_to_line (k : ℝ) (P : ℝ × ℝ) : ℝ :=
  abs (k * (P.1 - 1)) / sqrt (1 + k^2)

theorem line_intersects_circle (k : ℝ) :
  distance_from_point_to_line k circle_center < circle_radius :=
by
  sorry

end line_intersects_circle_l351_351067


namespace find_A_coordinates_sum_l351_351578

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351578


namespace Marley_fruits_total_is_31_l351_351673

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l351_351673


namespace two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l351_351394

theorem two_pow_1000_mod_3 : 2^1000 % 3 = 1 := sorry
theorem two_pow_1000_mod_5 : 2^1000 % 5 = 1 := sorry
theorem two_pow_1000_mod_11 : 2^1000 % 11 = 1 := sorry
theorem two_pow_1000_mod_13 : 2^1000 % 13 = 3 := sorry

end two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l351_351394


namespace sums_of_coordinates_of_A_l351_351614

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351614


namespace distance_product_maximized_l351_351699

theorem distance_product_maximized 
  (A B C : Type) [metric_space C] [inner_product_space ℝ C]
  (angleACB_bisector_perp : Prop)
  (distance_from_A_to_line : C → ℝ)
  (distance_from_B_to_line : C → ℝ)
  (lineL : set C) :
  (∀ (l : set C), is_line l ∧ through_vertex l C ∧ ¬ perpendicular_to_angle_bisctor l 
    → (distance_from_A_to_line A) * (distance_from_B_to_line B) < (distance_from_A_to_line A) * (distance_from_B_to_line B)) :=
sorry

end distance_product_maximized_l351_351699


namespace total_pencils_correct_l351_351685

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l351_351685


namespace cos_330_cos_30_val_answer_l351_351236

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351236


namespace bianca_points_l351_351169

theorem bianca_points : 
  let a := 5; let b := 8; let c := 10;
  let A1 := 10; let P1 := 5; let G1 := 5;
  let A2 := 3; let P2 := 2; let G2 := 1;
  (A1 * a - A2 * a) + (P1 * b - P2 * b) + (G1 * c - G2 * c) = 99 := 
by
  sorry

end bianca_points_l351_351169


namespace cos_330_eq_sqrt_3_div_2_l351_351216

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351216


namespace red_pairs_calculation_l351_351945

-- Define total numbers of students and pairs, and specific pairs wearing green shirts
variable (total_students : ℕ)
variable (green_students : ℕ)
variable (red_students : ℕ)
variable (total_pairs : ℕ)
variable (green_green_pairs : ℕ)

-- Given conditions
def conditions : Prop :=
  total_students = 150 ∧
  green_students = 65 ∧
  red_students = 85 ∧
  total_pairs = 75 ∧
  green_green_pairs = 30

-- The number of pairs where both students wear red shirts
def red_red_pairs (total_students green_students red_students total_pairs green_green_pairs : ℕ) : ℕ :=
  (red_students - (green_students - green_green_pairs * 2)) / 2

-- The theorem stating the problem and expected outcome
theorem red_pairs_calculation (h : conditions) :
  red_red_pairs total_students green_students red_students total_pairs green_green_pairs = 40 := sorry

end red_pairs_calculation_l351_351945


namespace number_of_substitution_ways_mod_1000_l351_351144

theorem number_of_substitution_ways_mod_1000 :
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  total_ways % 1000 = 573 := by
  -- Definition
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  -- Proof is omitted
  sorry

end number_of_substitution_ways_mod_1000_l351_351144


namespace marley_fruits_l351_351675

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l351_351675


namespace profit_percentage_l351_351110

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : (S - C) / C * 100 = 6.25 := by
  sorry

end profit_percentage_l351_351110


namespace sum_of_coordinates_A_l351_351608

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351608


namespace abcd_value_l351_351641

noncomputable def abcd_eval (a b c d : ℂ) : ℂ := a * b * c * d

theorem abcd_value (a b c d : ℂ) 
  (h1 : a + b + c + d = 5)
  (h2 : (5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125)
  (h3 : (a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205)
  (h4 : a^4 + b^4 + c^4 + d^4 = 25) : 
  abcd_eval a b c d = 70 := 
sorry

end abcd_value_l351_351641


namespace gcf_75_100_l351_351821

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351821


namespace midpoint_of_chord_through_focus_l351_351130

theorem midpoint_of_chord_through_focus
  (A B : ℝ × ℝ)
  (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4 * P.1 ↔ P ∈ {A, B})
  (h_line : ∀ P : ℝ × ℝ, P.2 = P.1 - 1 ↔ P ∈ {A, B})
  (h_focus : (1, 0) ∈ {A, B}) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in M = (3, 2) :=
by
  sorry

end midpoint_of_chord_through_focus_l351_351130


namespace circle_equation_l351_351420

theorem circle_equation :
  (∀ (a b r : ℝ), a = b ∧
    (-1 - a) ^ 2 + (2 - b) ^ 2 = r ^ 2 ∧
    a ^ 2 + (2 * √2) ^ 2 = r ^ 2 →
    (x - 3) ^ 2 + (y - 3) ^ 2 = 17) :=
sorry

end circle_equation_l351_351420


namespace vessel_capacity_is_8_l351_351897

-- Definitions based on the conditions provided
def vessel1_capacity : ℝ := 2
def vessel1_alcohol_percentage : ℝ := 0.30
def vessel1_alcohol_amount : ℝ := vessel1_alcohol_percentage * vessel1_capacity

def vessel2_capacity : ℝ := 6
def vessel2_alcohol_percentage : ℝ := 0.40
def vessel2_alcohol_amount : ℝ := vessel2_alcohol_percentage * vessel2_capacity

def total_mixture_volume : ℝ := 8
def new_concentration : ℝ := 0.30000000000000004
def existing_alcohol_amount : ℝ := vessel1_alcohol_amount + vessel2_alcohol_amount

theorem vessel_capacity_is_8 :
  let expected_alcohol_amount := new_concentration * total_mixture_volume in
  expected_alcohol_amount <= existing_alcohol_amount →
  total_mixture_volume = 8 :=
sorry

end vessel_capacity_is_8_l351_351897


namespace least_positive_integer_l351_351846

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 1) ∧ (a % 3 = 2) ∧ (a % 4 = 3) ∧ (a % 5 = 4) → a = 59 :=
by
  sorry

end least_positive_integer_l351_351846


namespace vecs_coplanar_l351_351439

open_locale classical

variables {V : Type*} [add_comm_group V] [module ℝ V]

def coplanar (a b c : V) : Prop := 
  ∃ r s : ℝ, c = r • a + s • b

theorem vecs_coplanar (a b : V) : coplanar a b (2 • a + 4 • b) :=
begin
  use [2, 4],
  exact rfl,
end

end vecs_coplanar_l351_351439


namespace cos_330_eq_sqrt3_div_2_l351_351257

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351257


namespace marley_fruits_l351_351676

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l351_351676


namespace cos_330_eq_sqrt3_div_2_l351_351247

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351247


namespace total_votes_proof_l351_351690

noncomputable def total_votes (A : ℝ) (T : ℝ) := 0.40 * T = A
noncomputable def votes_in_favor (A : ℝ) := A + 68
noncomputable def total_votes_calc (T : ℝ) (Favor : ℝ) (A : ℝ) := T = Favor + A

theorem total_votes_proof (A T : ℝ) (Favor : ℝ) 
  (hA : total_votes A T) 
  (hFavor : votes_in_favor A = Favor) 
  (hT : total_votes_calc T Favor A) : 
  T = 340 :=
by
  sorry

end total_votes_proof_l351_351690


namespace train_cross_platform_time_l351_351896

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_man : ℝ := 18
noncomputable def length_of_platform : ℝ := 280
noncomputable def length_of_train : ℝ := train_speed_mps * time_to_cross_man
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def speed_of_train : ℝ := 20

theorem train_cross_platform_time :
  (total_distance_to_cover / speed_of_train) = 32 :=
by
  calc
    total_distance_to_cover / speed_of_train = 640 / 20 : by sorry
    ... = 32 : by sorry

end train_cross_platform_time_l351_351896


namespace sin_cos_sum_l351_351446

theorem sin_cos_sum (α : ℝ) (h1 : sin α * cos α = 1 / 8) (h2 : 0 < α ∧ α < π / 2) : 
  sin α + cos α = sqrt 5 / 2 :=
by
  sorry

end sin_cos_sum_l351_351446


namespace geometry_proof_l351_351878

variables {A B C D E F M : Point}
variables {circle : Circle}
variables {triangleABC : Triangle A B C}

-- Given conditions
def circle_passes_through_A_and_B (circle : Circle) (A B : Point) : Prop := 
  circle.contains A ∧ circle.contains B

def circle_intersects_AC_at_D (circle : Circle) (A C D : Point) : Prop :=
  circle.contains D ∧ segment A C D

def circle_intersects_BC_at_E (circle : Circle) (B C E : Point) : Prop :=
  circle.contains E ∧ segment B C E

def BA_and_ED_intersect_at_F (A B E D F : Point) : Prop :=
  intersect (line A B) (line E D) = F

def BD_and_CF_intersect_at_M (B D C F M : Point) : Prop :=
  intersect (line B D) (line C F) = M

-- Statement to prove
def problem_statement (A B C D E F M : Point) (circle : Circle) 
  (triangleABC : Triangle A B C) : Prop :=
  circle_passes_through_A_and_B circle A B ∧
  circle_intersects_AC_at_D circle A C D ∧
  circle_intersects_BC_at_E circle B C E ∧
  BA_and_ED_intersect_at_F  A B E D F ∧
  BD_and_CF_intersect_at_M  B D C F M ∧
  distance M F = distance M C ↔ distance M B * distance M D = distance M C^2

-- Lean statement
theorem geometry_proof (A B C D E F M : Point) (circle : Circle)
  (triangleABC : Triangle A B C) : problem_statement A B C D E F M circle triangleABC :=
by {
  sorry
}

end geometry_proof_l351_351878


namespace unique_solution_condition_l351_351507

theorem unique_solution_condition (p q : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by
  sorry

end unique_solution_condition_l351_351507


namespace total_students_sampled_l351_351128

theorem total_students_sampled (senior: ℕ) (junior: ℕ) (freshman: ℕ) (sampled_freshman: ℕ) 
    (Hsenior: senior = 1000) 
    (Hjunior: junior = 1200) 
    (Hfreshman: freshman = 1500) 
    (Hsampled_freshman: sampled_freshman = 75) 
    (Hstratified_sampling: ∀ (s t: ℕ), s / t = sampled_freshman / freshman) : 
    ∃ total_sampled: ℕ, total_sampled = 185 := 
by 
  let total_students := senior + junior + freshman 
  let sampling_ratio := sampled_freshman / freshman 
  have Htotal_students : total_students = 3700 := by 
    rw [Hsenior, Hjunior, Hfreshman] 
  have Hsampling_ratio : sampling_ratio = 1 / 20 := by 
    rw [Hsampled_freshman, Hfreshman] 
  let total_sampled := total_students * sampling_ratio 
  have Htotal_sampled : total_sampled = 185 := by 
    exact Hsampling_ratio * 3700 
  use total_sampled 
  exact Htotal_sampled 

end total_students_sampled_l351_351128


namespace shadow_area_l351_351898

theorem shadow_area (y : ℝ) (cube_side : ℝ) (shadow_excl_area : ℝ) 
  (h₁ : cube_side = 2) 
  (h₂ : shadow_excl_area = 200)
  (h₃ : ((14.28 - 2) / 2 = y)) :
  ⌊1000 * y⌋ = 6140 :=
by
  sorry

end shadow_area_l351_351898


namespace cos_330_eq_sqrt3_div_2_l351_351263

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351263


namespace calculation_is_correct_l351_351951

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l351_351951


namespace functional_equation_solution_l351_351050

noncomputable def f : ℝ → ℝ 
  := sorry

variable (f : ℝ → ℝ)
variable (c : ℝ)

theorem functional_equation_solution
  (h_cont : Continuous f)
  (h_func_eq : ∀ x y : ℝ, c^2 * f(x + y) = f(x) * f(y))
  (h_f_one : f 1 = c)
  (h_c_pos : c > 0)
  : ∀ x : ℝ, f x = c ^ x :=
by
  sorry

end functional_equation_solution_l351_351050


namespace cos_330_cos_30_val_answer_l351_351231

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351231


namespace maximize_profit_l351_351124

noncomputable def additional_cost (x : ℕ) (h : 0 < x) : ℚ :=
if x < 80 then
  (1/3 : ℚ) * x^2 + 10 * x
else
  51 * x + (10000 : ℚ) / x - 1550

noncomputable def profit (x : ℕ) (h : 0 < x) : ℚ :=
if x < 80 then
  -(1/3 : ℚ) * x^2 + 40 * x - 150
else
  1400 - (x + (10000 : ℚ) / x)

theorem maximize_profit : ∃ x, 0 < x ∧ profit x (by linarith) = 1200 :=
by
  use 100
  split
  { linarith }
  { have : profit 100 _ = 1200 := sorry
    exact this }
  sorry

end maximize_profit_l351_351124


namespace remainder_of_product_l351_351773

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351773


namespace find_parabola_equation_l351_351014

def parabola_equation (p : ℝ) (h : p > 0) : Prop :=
  let F := (p / 2, 0)
  let M := (0 : ℝ, 2)
  let y_axis := (x : ℝ) → 0
  (∃ (P Q : ℝ × ℝ),
   P.1 ^ 2 = 2 * p * P.1 ∧
   Q.2 = 4 ∧ 
   (Q.1, Q.2) = (-p / 2, 4) ∧
   (∃ (FQ : ℝ → ℝ), FQ (0 : ℝ) = 2 ∧ FQ Q.1 = Q.2)) ∧ 
   let area_PQF := 1/2 * (Q.1 - P.1) * (P.2 - F.2) = 10 in 
  y^2 = 2 * p * x

theorem find_parabola_equation : 
  parabola_equation 2 ∨ parabola_equation 8 ∧
  (∀ e, e = parabola_equation 2 ∨ e = parabola_equation 8) :=
sorry

end find_parabola_equation_l351_351014


namespace sharon_total_distance_l351_351709

/-- Problem Conditions -/
def usual_time_minutes := 240
def half_distance (d : ℕ) := d / 2
def normal_speed (d : ℕ) := d / usual_time_minutes
def reduced_speed (d : ℕ) := normal_speed d - 0.5
def total_journey_time := 330

/-- Mathematically equivalent problem -/
theorem sharon_total_distance (d : ℕ) (h1 : usual_time_minutes = 240)
  (h2 : 120 + ((half_distance d) / reduced_speed d) = total_journey_time):
  d = 280 := by
  sorry

end sharon_total_distance_l351_351709


namespace calc_f_f_21_over_4_l351_351652

noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then 4 * x^2 - 2
  else if 0 < x ∧ x < 1 then x
  else f (x - 3)

theorem calc_f_f_21_over_4 : f (f (21 / 4)) = 1 / 4 :=
by
  sorry

end calc_f_f_21_over_4_l351_351652


namespace ellipse_foci_distance_l351_351921

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351921


namespace remainder_of_product_mod_7_l351_351756

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351756


namespace intersection_count_l351_351457

noncomputable def P_trajectory_part1 : Set (ℝ × ℝ) := { p | p.2 = 0 ∧ p.1 ≤ 0 }
noncomputable def P_trajectory_part2 : Set (ℝ × ℝ) := { p | p.2 ^ 2 = 4 * p.1 ∧ p.1 ≥ 0 }
noncomputable def trajectory_C : Set (ℝ × ℝ) := P_trajectory_part1 ∪ P_trajectory_part2

noncomputable def line_l : Set (ℝ × ℝ) := { p | 2 * p.1 - 3 * p.2 + 4 = 0 }

theorem intersection_count :
  ∃ (P : Set (ℝ × ℝ)), P = line_l ∩ trajectory_C ∧ Finset.card (P.to_finset) = 3 := sorry

end intersection_count_l351_351457


namespace det_matrix_is_82_l351_351117

theorem det_matrix_is_82 :
  let M := ![[3, 1, 2], [-1, 2, 5], [0, -4, 2]] in
  Matrix.det M = 82 :=
by
  sorry

end det_matrix_is_82_l351_351117


namespace small_stick_length_l351_351513

theorem small_stick_length 
  (x : ℝ) 
  (hx1 : 3 < x) 
  (hx2 : x < 9) 
  (hx3 : 3 + 6 > x) : 
  x = 4 := 
by 
  sorry

end small_stick_length_l351_351513


namespace distinguishable_squares_count_l351_351799

theorem distinguishable_squares_count :
  let colors := 5  -- Number of different colors
  let total_corner_sets :=
    5 + -- All four corners the same color
    5 * 4 + -- Three corners the same color
    Nat.choose 5 2 * 2 + -- Two pairs of corners with the same color
    5 * 4 * 3 * 2 -- All four corners different
  let total_corner_together := total_corner_sets
  let total := 
    (4 * 5 + -- One corner color used
    3 * (5 * 4 + Nat.choose 5 2 * 2) + -- Two corner colors used
    2 * (5 * 4 * 3 * 2) + -- Three corner colors used
    1 * (5 * 4 * 3 * 2)) -- Four corner colors used
  total_corner_together * colors / 10
= 540 :=
by
  sorry

end distinguishable_squares_count_l351_351799


namespace cos_330_eq_sqrt3_div_2_l351_351357

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351357


namespace cos_330_eq_sqrt3_div_2_l351_351307

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351307


namespace cos_330_cos_30_val_answer_l351_351229

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351229


namespace sum_first_10_common_elements_l351_351428

noncomputable def arithmetic_progression (n : ℕ) : ℕ :=
  5 + 3 * n

noncomputable def geometric_progression (k : ℕ) : ℕ :=
  10 * (2 ^ k)

theorem sum_first_10_common_elements : 
  (finset.sum (finset.range 10) (λ i, 20 * (4^i))) = 6990500 :=
by
  sorry

end sum_first_10_common_elements_l351_351428


namespace cos_330_eq_sqrt3_div_2_l351_351284

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351284


namespace cos_330_cos_30_val_answer_l351_351241

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351241


namespace greatest_x_l351_351866

-- Define conditions from the problem
def is_greatest_x (x : ℤ) : Prop :=
  2.134 * 10 ^ x < 240000 ∧ ∀ y : ℤ, 2.134 * 10 ^ y < 240000 → y ≤ x

-- State the main theorem
theorem greatest_x : is_greatest_x 5 :=
by
  sorry

end greatest_x_l351_351866


namespace min_queries_needed_l351_351023

/- Person A writes 100 natural numbers in a specific sequence -/
def sequence_A := List.range (100)

/- Person B can ask Person A to reveal the order of any subset of 50 numbers -/
def query (subset : Finset ℕ) : List ℕ :=
  subset.to_list.sorted_with respect to A's original order 

/- Person B needs to determine the entire sequence -/
def determine_sequence : List (Finset ℕ) → List ℕ
  | [] => []
  | (subset :: remaining_queries) =>  sorry -- we skip the actual implementation

/- Proving that the minimum number of queries required is 5 -/
theorem min_queries_needed :
  ∃ queries : List (Finset ℕ), List.length queries = 5 ∧ determine_sequence queries = sequence_A :=
  sorry -- this is the statement we need to prove

end min_queries_needed_l351_351023


namespace gcf_75_100_l351_351820

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351820


namespace cos_330_eq_sqrt3_div_2_l351_351192

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351192


namespace roots_of_equation_l351_351960

theorem roots_of_equation :
  ∀ x : ℚ, (3 * x^2 / (x - 2) - (5 * x + 10) / 4 + (9 - 9 * x) / (x - 2) + 2 = 0) ↔ 
           (x = 6 ∨ x = 17/3) := 
sorry

end roots_of_equation_l351_351960


namespace cos_330_eq_sqrt3_over_2_l351_351208

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351208


namespace polar_coordinates_of_circle_center_l351_351480

theorem polar_coordinates_of_circle_center:
  let x (θ : Real) := 1 + Real.cos θ
  let y (θ : Real) := 1 + Real.sin θ
  let r := Real.sqrt(2)
  let φ := Real.pi / 4
  (∃ θ, x θ = 1 + Real.cos θ ∧ y θ = 1 + Real.sin θ) →
  (r, φ) = (Real.sqrt(2), Real.pi / 4) := by
  sorry

end polar_coordinates_of_circle_center_l351_351480


namespace distance_between_foci_of_ellipse_l351_351928

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351928


namespace complex_binomial_expression_sum_equals_zero_l351_351665

def x : ℂ := (2 * complex.I) / (1 - complex.I)

theorem complex_binomial_expression_sum_equals_zero :
  ∑ k in finset.range 2016 \{0}, nat.choose 2016 (k + 1) * x^(k + 1) = 0 :=
sorry

end complex_binomial_expression_sum_equals_zero_l351_351665


namespace min_value_x_l351_351546

theorem min_value_x (m : ℕ) (x : ℕ) (x_i : Fin m → ℕ) :
  (∀ j : Fin 10, ∑ i : Fin m, ite (x_i i > 0) 1 0 = 3) →
  (∀ a b : Fin 10, a ≠ b → ∃ i : Fin m, x_i i ≥ 1) →
  ∑ i : Fin m, x_i i = 30 →
  x = Finset.max' (Finset.univ.map (Finset.SortedBy (x_i))) _ →
  x ≥ 5 :=
sorry

end min_value_x_l351_351546


namespace ellipse_foci_distance_l351_351925

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351925


namespace cos_330_eq_sqrt3_div_2_l351_351315

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351315


namespace roots_seventh_sum_l351_351660

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l351_351660


namespace cos_330_eq_sqrt3_div_2_l351_351190

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351190


namespace find_polynomial_B_correct_result_of_A_minus_B_l351_351858

def A : ℤ[X] := 2 * X^2 - 5 * X + 6
def incorrect_result : ℤ[X] := 4 * X^2 - 4 * X + 6

theorem find_polynomial_B : ∃ B : ℤ[X], A + B = incorrect_result ∧ B = 2 * X^2 + X := by
  sorry

theorem correct_result_of_A_minus_B : ∀ B : ℤ[X], B = 2 * X^2 + X → A - B = -6 * X + 6 := by
  sorry

end find_polynomial_B_correct_result_of_A_minus_B_l351_351858


namespace gcd_75_100_l351_351835

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351835


namespace problem_complex_conjugate_power_l351_351469

-- Define the complex number z and its conjugate \overline{z}
def z : ℂ := 1/2 - complex.I

-- Statement of the proof problem
theorem problem_complex_conjugate_power :
  (z - complex.conj z) ^ 2016 = 2 ^ 2016 :=
sorry

end problem_complex_conjugate_power_l351_351469


namespace remainder_of_product_mod_7_l351_351760

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351760


namespace finite_additive_not_countably_additive_l351_351649

variable (Ω : Type)
variable [Countable Ω]

noncomputable def μ (A : Set Ω) : ℝ≥0∞ := if A.Finite then 0 else ⊤

theorem finite_additive_not_countably_additive :
  (∀ (A B : Set Ω), A ∩ B = ∅ → μ Ω (A ∪ B) = μ Ω A + μ Ω B) ∧
  ¬ (∀ (A : ℕ → Set Ω), Pairwise (λ i j, A i ∩ A j = ∅) → μ Ω (⋃ i, A i) = ∑' i, μ Ω (A i)) :=
sorry

end finite_additive_not_countably_additive_l351_351649


namespace cos_330_cos_30_val_answer_l351_351239

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351239


namespace odd_function_sin_cos_product_l351_351517

-- Prove that if the function f(x) = sin(x + α) - 2cos(x - α) is an odd function, then sin(α) * cos(α) = 2/5
theorem odd_function_sin_cos_product (α : ℝ)
  (hf : ∀ x, Real.sin (x + α) - 2 * Real.cos (x - α) = -(Real.sin (-x + α) - 2 * Real.cos (-x - α))) :
  Real.sin α * Real.cos α = 2 / 5 :=
  sorry

end odd_function_sin_cos_product_l351_351517


namespace cube_surface_divisible_into_12_squares_l351_351881

theorem cube_surface_divisible_into_12_squares (a : ℝ) :
  (∃ b : ℝ, b = a / Real.sqrt 2 ∧
  ∀ cube_surface_area: ℝ, cube_surface_area = 6 * a^2 →
  ∀ smaller_square_area: ℝ, smaller_square_area = b^2 →
  12 * smaller_square_area = cube_surface_area) :=
sorry

end cube_surface_divisible_into_12_squares_l351_351881


namespace find_A_coordinates_sum_l351_351582

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351582


namespace second_train_speed_l351_351151

theorem second_train_speed (len1 len2 dist t : ℕ) (h1 : len1 = 100) (h2 : len2 = 150) (h3 : dist = 50) (h4 : t = 60) : 
  (len1 + len2 + dist) / t = 5 := 
  by
  -- Definitions from conditions
  have h_len1 : len1 = 100 := h1
  have h_len2 : len2 = 150 := h2
  have h_dist : dist = 50 := h3
  have h_time : t = 60 := h4
  
  -- Proof deferred
  sorry

end second_train_speed_l351_351151


namespace find_special_integer_l351_351416

theorem find_special_integer
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ∀ (d : ℕ), d ∣ n → d > 1 → ∃ (a r : ℕ), a > 0 ∧ r > 1 ∧ d = a^r + 1) :
  n = 10 :=
begin
  sorry
end

end find_special_integer_l351_351416


namespace maximum_self_intersections_hexagon_l351_351380

theorem maximum_self_intersections_hexagon (vertices : Fin 6 → ℝ × ℝ) (h_circle : ∀ i j : Fin 6, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  : ∃ N ≤ 7, ∀ hexagon : Fin 6 → ℝ × ℝ, closed_polygon hexagon ∧ all_vertices_on_circle hexagon = maximum_self_intersections hexagon = N :=
sorry

end maximum_self_intersections_hexagon_l351_351380


namespace number_of_zeros_of_f_l351_351064

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 9

theorem number_of_zeros_of_f : ∃ (z : ℕ), z = 2 ∧ ∀ x : ℝ, (f x = 0 → x = -3 ∨ x = -2 / 3 ∨ x = 1 ∨ x = 3) := 
sorry

end number_of_zeros_of_f_l351_351064


namespace number_of_distinct_paintings_l351_351619

noncomputable def binom : ℕ → ℕ → ℕ
  | n, k :=
    if h : k ≤ n
    then nat.choose n k
    else 0

def total_colorings_without_symmetry : ℕ :=
  binom 8 4 * binom 4 3

def fixed_points_identity : ℕ :=
  total_colorings_without_symmetry

def fixed_points_vertex_reflections : ℕ :=
  4 * 8 -- Placeholder, in the problem we assume each contributes 8

def fixed_points_side_reflections : ℕ :=
  3 * 6 -- Placeholder, in the problem we assume each contributes 6

def fixed_points_rotations : ℕ :=
  4 * 0 -- Placeholder, in the problem we assume each contributes 0

def total_fixed_points : ℕ :=
  fixed_points_identity + fixed_points_vertex_reflections + fixed_points_side_reflections + fixed_points_rotations

theorem number_of_distinct_paintings : (total_fixed_points / 12) = 26 :=
  by sorry

end number_of_distinct_paintings_l351_351619


namespace profit_percentage_is_correct_l351_351862

-- Define the given conditions
def selling_price : ℝ := 850
def profit : ℝ := 205

-- Define the cost price based on selling price and profit
def cost_price : ℝ := selling_price - profit

-- Define the profit percentage
def profit_percentage : ℝ := (profit / cost_price) * 100

-- Prove that the profit percentage is 31.78%
theorem profit_percentage_is_correct : profit_percentage = 31.78 := 
by 
  -- Skip proof 
  sorry

end profit_percentage_is_correct_l351_351862


namespace num_possible_values_and_sum_l351_351012

def S : Set ℝ := {x | x ≠ 0}
def f (x : S) : S

axiom functional_equation (x y : S) (h : x + y = 1) : 
  f x + f y = f (x * y * f 1)

theorem num_possible_values_and_sum (n : ℕ) (s : ℝ) :
  (∃ n s : ℕ, s = \1/3 ∧ n = 1 ∧ n * s = 1/3) := by
  sorry

end num_possible_values_and_sum_l351_351012


namespace inequality_proof_l351_351552

variables {AB AD BC CD AP BP h : ℝ}

-- Suppose we have a convex quadrilateral ABCD with the given conditions.
-- AB = AD + BC holds
-- AP = h + AD holds 
-- BP = h + BC holds
-- h is the distance from point P to line CD

theorem inequality_proof (h : ℝ) (AD BC : ℝ) :
  (1 / Real.sqrt h) ≥ (1 / Real.sqrt AD) + (1 / Real.sqrt BC) :=
by
  assume (AD BC : ℝ)
  sorry

end inequality_proof_l351_351552


namespace greatest_prime_factor_of_expression_l351_351094

theorem greatest_prime_factor_of_expression : 
  ∃ p : ℕ, prime p ∧ p = 31 ∧ 
  (∀ q : ℕ, (q ∣ (4 ^ 17 - 2 ^ 29) → prime q) → q ≤ 31) :=
by
  sorry

end greatest_prime_factor_of_expression_l351_351094


namespace sum_of_coordinates_A_l351_351554

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351554


namespace segment_area_approx_l351_351419

def area_of_remaining_segment (r : ℝ) (θ1 θ2 : ℝ) : ℝ :=
  (θ1 / 360) * π * r^2 - (θ2 / 360) * π * r^2

theorem segment_area_approx :
  area_of_remaining_segment 12 42 18 ≈ 30.159 := 
sorry

end segment_area_approx_l351_351419


namespace rabbit_position_after_10_exchanges_l351_351443

def initial_positions : (ℕ → ℕ) :=
  λ n, if n = 1 then 1
       else if n = 2 then 2
       else if n = 3 then 3
       else 4

def swap_1_swap_3 (p : ℕ → ℕ) : (ℕ → ℕ) :=
  λ n, if n = 1 then p 3
       else if n = 2 then p 4
       else if n = 3 then p 1
       else p 2

def swap_2_swap_4 (p : ℕ → ℕ) : (ℕ → ℕ) :=
  λ n, if n = 1 then p 2
       else if n = 2 then p 1
       else if n = 3 then p 4
       else p 3

def nth_swap (n : ℕ) : (ℕ → ℕ) :=
  λ p, if n % 2 = 0 then swap_2_swap_4 p else swap_1_swap_3 p

def positions_after_n_exchanges (n : ℕ) : (ℕ → ℕ) :=
  (λ p, (List.range n).foldl (λ p i, nth_swap (i + 1) p) p) initial_positions

theorem rabbit_position_after_10_exchanges : (positions_after_n_exchanges 10 3) = 2 :=
  sorry

end rabbit_position_after_10_exchanges_l351_351443


namespace least_total_bananas_l351_351081

theorem least_total_bananas :
  ∃ (d : ℕ), 
    ∃ (a b c : ℕ), 
    (d = a + b + c) ∧ 
    (∃ (n1 n2 n3 : ℕ), n1 = 3 / 4 * a + 1 / 8 * b + 11 / 24 * c ∧ n2 = 1 / 8 * a + 1 / 4 * b + 11 / 24 * c ∧ n3 = 1 / 8 * a + 3 / 8 * b + 1 / 12 * c ∧
    (n1, n2, n3) = (3 * k, 2 * k, 1 * k) for some k in ℕ) ∧
    (n1 % 1 = 0 ∧ n2 % 1 = 0 ∧ n3 % 1 = 0) ∧ 
    d = 51 :=
begin
  sorry
end

end least_total_bananas_l351_351081


namespace bullet_train_speed_l351_351874

theorem bullet_train_speed 
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_cross : ℝ)
  (combined_length : ℝ)
  (time_cross_hours : ℝ)
  (relative_speed : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 270 → 
  length_train2 = 230.04 →
  speed_train2 = 80 →
  time_cross = 9 →
  combined_length = (length_train1 + length_train2) / 1000 →
  time_cross_hours = time_cross / 3600 →
  relative_speed = combined_length / time_cross_hours →
  relative_speed = speed_train1 + speed_train2 →
  speed_train1 = 120.016 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bullet_train_speed_l351_351874


namespace range_of_a_l351_351520

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x - a ≤ -3) → a ∈ Set.Iic (-6) ∪ Set.Ici 2 :=
by
  intro h
  sorry

end range_of_a_l351_351520


namespace max_liars_5x5_grid_l351_351532

theorem max_liars_5x5_grid : ∃ m : ℕ, m = 13 ∧
  ∀ (grid : ℕ × ℕ → bool), 
  (∀ i j, (0 ≤ i) ∧ (i < 5) ∧ (0 ≤ j) ∧ (j < 5) → 
    (grid (i, j) = tt → 
      (∃ k l, (abs (i - k) + abs (j - l) = 1) ∧ 
        (0 ≤ k) ∧ (k < 5) ∧ (0 ≤ l) ∧ (l < 5) ∧ grid (k, l) = ff))) →
  m = ∑ i j, if grid (i, j) = tt then 1 else 0 :=
begin
  sorry
end

end max_liars_5x5_grid_l351_351532


namespace inequality_a2_b2_c2_geq_abc_l351_351664

theorem inequality_a2_b2_c2_geq_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_cond: a + b + c ≥ a * b * c) :
  a^2 + b^2 + c^2 ≥ a * b * c := 
sorry

end inequality_a2_b2_c2_geq_abc_l351_351664


namespace gcd_75_100_l351_351828

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351828


namespace complex_parts_sum_l351_351650

-- Definitions of the complex parts and the summation
def a : ℂ := (i⁻¹).im
def b : ℂ := ((1 + i)^2).re

theorem complex_parts_sum :
  a + b = -1 := by
  sorry

end complex_parts_sum_l351_351650


namespace problem1_problem2_l351_351870

-- Problem (1)
theorem problem1 (α : ℝ) (h : cos (α + π / 6) - sin α = 3 * sqrt 3 / 5) : sin (α + 5 * π / 6) = 3 / 5 :=
by
  sorry

-- Problem (2)
theorem problem2 (α β : ℝ) (h1 : sin α + sin β = 1 / 2) (h2 : cos α + cos β = sqrt 2 / 2) : cos (α - β) = -5 / 8 :=
by
  sorry

end problem1_problem2_l351_351870


namespace linear_eq_k_l351_351471

theorem linear_eq_k (k : ℕ) : (∀ x : ℝ, x^(k-1) + 3 = 0 ↔ k = 2) :=
by
  sorry

end linear_eq_k_l351_351471


namespace ratio_of_new_triangle_area_l351_351379

variables (T : ℝ) -- area of the original triangle
-- condition 2: Points divide the sides into segments with the ratio 1:2
-- condition 3: These points are connected to form a new triangle inside the original one

theorem ratio_of_new_triangle_area (T : ℝ) : 
  let T_new := T / 9
  in (T_new / T) = 1 / 9 :=
sorry

end ratio_of_new_triangle_area_l351_351379


namespace Samia_walked_distance_l351_351031

theorem Samia_walked_distance 
  (bike_speed : ℝ) (bike_time : ℝ) (walk_speed : ℝ) (total_time : ℝ)
  (h_bike_speed : bike_speed = 15)
  (h_bike_time : bike_time = 0.5)
  (h_walk_speed : walk_speed = 4)
  (h_total_time : total_time = 1.5) :
  let distance_biked := bike_speed * bike_time,
      walk_time := total_time - bike_time,
      distance_walked := walk_speed * walk_time 
  in distance_walked = 4.0 := 
by 
  -- Proof will be provided here
  sorry

end Samia_walked_distance_l351_351031


namespace num_pairs_satisfying_eq_l351_351393

theorem num_pairs_satisfying_eq : 
  (∃! (pairs : Finset (Real × Real)), (
    ∀ (pair : Real × Real),
      pair ∈ pairs ↔ 
      (0 ≤ pair.1 ∧ pair.1 ≤ π / 8 ∧ 0 ≤ pair.2 ∧ pair.2 ≤ π / 8 ∧ 
      cos(1000 * pair.1)^6 - sin(1000 * pair.2)^6 = 1)) ∧
    pairs.card = 15876) :=
by
  sorry

end num_pairs_satisfying_eq_l351_351393


namespace sum_coordinates_A_l351_351595

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351595


namespace sum_of_coordinates_A_l351_351603

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351603


namespace find_f2_l351_351450

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f(x) + 9

-- The main theorem
theorem find_f2 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : g f (-2) = 3) : f 2 = 6 :=
by
  sorry

end find_f2_l351_351450


namespace area_of_rectangle_l351_351141

def length_fence (x : ℝ) : ℝ := 2 * x + 2 * x

theorem area_of_rectangle (x : ℝ) (h : length_fence x = 150) : x * 2 * x = 2812.5 :=
by
  sorry

end area_of_rectangle_l351_351141


namespace square_in_ellipse_area_l351_351895

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 8 = 1

-- Define the square's side length
def square_side_length := 2 * (2 * real.sqrt(6) / 3)

-- Define the area of the square
def square_area := square_side_length ^ 2

-- The square is inscribed in the ellipse
theorem square_in_ellipse_area :
  square_area = 32 / 3 := by
  sorry

end square_in_ellipse_area_l351_351895


namespace cos_330_cos_30_val_answer_l351_351237

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351237


namespace find_x_l351_351425

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem find_x (x : ℝ) (h : log_base 3 (x - 3) + log_base (Real.sqrt 3) (x^3 - 3) + log_base (1/3) (x - 3) = 3 ∧ x > 3) : 
  x = Real.cbrt 12 :=
begin
  sorry
end

end find_x_l351_351425


namespace derivative_of_f_sqrt_f_at_inv_sqrt2_f_plus_f_sqrt_l351_351437

open Real

noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, 1 / sqrt (1 - t^2)

variables (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1)

theorem derivative_of_f_sqrt (h : 0 < x ∧ x < 1) : 
  deriv (λ x, f (sqrt (1 - x ^ 2))) x = -1 / sqrt (1 - x^2) :=
sorry

theorem f_at_inv_sqrt2 : 
  f (1 / sqrt 2) = π / 4 :=
sorry

theorem f_plus_f_sqrt : 
  (f x + f (sqrt (1 - x^2)) = π / 2) :=
sorry

end derivative_of_f_sqrt_f_at_inv_sqrt2_f_plus_f_sqrt_l351_351437


namespace total_packs_equiv_117_l351_351686

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end total_packs_equiv_117_l351_351686


namespace ellipse_foci_distance_l351_351924

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351924


namespace cos_330_eq_sqrt3_div_2_l351_351353

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351353


namespace unique_solution_to_functional_eq_l351_351991

theorem unique_solution_to_functional_eq :
  (∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2) :=
by
  sorry

end unique_solution_to_functional_eq_l351_351991


namespace min_sum_distances_from_parabola_point_l351_351465

theorem min_sum_distances_from_parabola_point :
  ∀ (P : ℝ × ℝ), (P.1, P.2)^2 = (P.1, 4 * P.1) →
    let d₁ := (4 * P.1 - 3 * P.2 + 6) / real.sqrt(4^2 + (-3)^2),
        d₂ := abs(P.1 - (-1))
    in d₁ + d₂ = 2 := sorry

end min_sum_distances_from_parabola_point_l351_351465


namespace sum_of_coords_A_l351_351570

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351570


namespace omega_value_a_value_graph_properties_l351_351015

noncomputable def f (x : ℝ) (ω a : ℝ) := sin (2 * ω * x + π / 3) + sqrt 3 / 2 + a
noncomputable def g (x : ℝ) (ω a : ℝ) := f x ω a - a

theorem omega_value : ∃ ω, (∀ x, (2 * ω * x + π / 3 = π / 2) -> (x = π / 6)) -> ω = 1 / 2 :=
begin
  sorry
end

theorem a_value (ω : ℝ) (hω: ω = 1 / 2) : ∃ a, (∀ x ∈ Icc (-π / 3) (5 * π / 6), f x  ω a = √3) -> a = (√3 + 1) / 2 :=
begin
  sorry
end 

theorem graph_properties (ω : ℝ) (a : ℝ) (hω: ω = 1 / 2) (ha: a = (√3 + 1) / 2) :
  (∀ x, g x ω a = sin (x + π / 3) + sqrt 3 / 2) ∧ 
  (∀ k : ℤ, symm_metric_axis (g (ω:=1 / 2) (a:=(√3 + 1) / 2)) (x = π / 6 + k * π)) ∧  
  (∀ k : ℤ, center_of_symmetry (g (ω:=1 / 2) (a:=(√3 + 1) / 2)) ((-π / 3 + k * π), sqrt 3 / 2))
 :=
begin
  sorry
end

end omega_value_a_value_graph_properties_l351_351015


namespace integer_solutions_log_inequality_l351_351072

theorem integer_solutions_log_inequality :
  {x : ℤ | 2 < Real.log2 (x + 5) ∧ Real.log2 (x + 5) < 3} = {0, 1, 2} :=
by
  -- The proof will be provided here
  sorry

end integer_solutions_log_inequality_l351_351072


namespace cos_330_eq_sqrt_3_div_2_l351_351215

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351215


namespace wisdom_number_1998th_l351_351888

definition is_wisdom_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ n = x^2 - y^2

def odd_wisdom_number_formula (k : ℕ) : ℕ :=
  (k + 1)^2 - k^2

def even_wisdom_number_formula (k : ℕ) : ℕ :=
  (k + 1)^2 - (k - 1)^2

theorem wisdom_number_1998th : ∃ n : ℕ, nth_wisdom_number n 1998 = 2667 :=
sorry

noncomputable def nth_wisdom_number (n : ℕ) (m : ℕ) : ℕ :=
sorry

end wisdom_number_1998th_l351_351888


namespace cos_330_eq_sqrt3_div_2_l351_351321

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351321


namespace remainder_of_product_mod_7_l351_351757

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351757


namespace sum_of_coordinates_A_l351_351560

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351560


namespace cos_330_eq_sqrt3_div_2_l351_351340

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351340


namespace cos_330_eq_sqrt3_div_2_l351_351260

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351260


namespace angle_ABC_correct_l351_351011

noncomputable def A : ℝ × ℝ × ℝ := (-3, 1, 5)
noncomputable def B : ℝ × ℝ × ℝ := (-4, 0, 2)
noncomputable def C : ℝ × ℝ × ℝ := (-5, 0, 3)

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

noncomputable def AB := distance A B
noncomputable def AC := distance A C
noncomputable def BC := distance B C

noncomputable def cos_angle_ABC : ℝ :=
  (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

noncomputable def angle_ABC : ℝ :=
  Real.arccos cos_angle_ABC

theorem angle_ABC_correct : angle_ABC ≈ 75.7 :=
  sorry

end angle_ABC_correct_l351_351011


namespace quadratic_roots_identity_l351_351000

variable (α β : ℝ)
variable (h1 : α^2 + 3*α - 7 = 0)
variable (h2 : β^2 + 3*β - 7 = 0)

-- The problem is to prove that α^2 + 4*α + β = 4
theorem quadratic_roots_identity :
  α^2 + 4*α + β = 4 :=
sorry

end quadratic_roots_identity_l351_351000


namespace remainder_product_l351_351752

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351752


namespace number_of_distinct_collections_l351_351691

def mathe_matical_letters : Multiset Char :=
  {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

def vowels : Multiset Char :=
  {'A', 'A', 'A', 'E', 'I'}

def consonants : Multiset Char :=
  {'M', 'T', 'H', 'M', 'T', 'C', 'L', 'C'}

def indistinguishable (s : Multiset Char) :=
  (s.count 'A' = s.count 'A' ∧
   s.count 'E' = 1 ∧
   s.count 'I' = 1 ∧
   s.count 'M' = 2 ∧
   s.count 'T' = 2 ∧
   s.count 'H' = 1 ∧
   s.count 'C' = 2 ∧
   s.count 'L' = 1)

theorem number_of_distinct_collections :
  5 * 16 = 80 :=
by
  -- proof would go here
  sorry

end number_of_distinct_collections_l351_351691


namespace number_of_special_divisors_l351_351388

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end number_of_special_divisors_l351_351388


namespace general_term_formula_l351_351493

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 then 0 else 
  let rec aux (m : ℕ) : ℝ :=
    match m with 
    | 1 => 0
    | k+1 => 5 * aux k + (24 * (aux k)^2 + 1).sqrt
  in aux n

theorem general_term_formula (n : ℕ) (hn : n ≥ 1) : 
  sequence n = (√6 / 24) * ((5 + 2 * √6) ^ n - (5 - 2 * √6) ^ n) :=
begin
  sorry
end

end general_term_formula_l351_351493


namespace triangle_area_l351_351155

noncomputable def area_of_right_triangle (a b c : ℝ) (h : a ^ 2 + b ^ 2 = c ^ 2) : ℝ :=
  (1 / 2) * a * b

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_of_right_triangle a b c h4 = 54 := by
  rw [h1, h2, h3]
  sorry

end triangle_area_l351_351155


namespace cos_330_eq_sqrt3_div_2_l351_351322

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351322


namespace train_crossing_time_l351_351132

-- Define the length of the train
def train_length : ℝ := 120

-- Define the speed of the train
def train_speed : ℝ := 15

-- Define the target time to cross the man
def target_time : ℝ := 8

-- Proposition to prove
theorem train_crossing_time :
  target_time = train_length / train_speed :=
by
  sorry

end train_crossing_time_l351_351132


namespace cos_330_eq_sqrt3_div_2_l351_351341

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351341


namespace gcd_75_100_l351_351836

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351836


namespace three_digit_numbers_divisible_by_45_with_arithmetic_sequence_l351_351857

theorem three_digit_numbers_divisible_by_45_with_arithmetic_sequence :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 45 ∣ n ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ b = (a + c) / 2 ∧ (a - b = b - c))} =
  {135, 630, 765} :=
sorry

end three_digit_numbers_divisible_by_45_with_arithmetic_sequence_l351_351857


namespace find_A_coordinates_sum_l351_351580

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351580


namespace sum_of_coordinates_A_l351_351564

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351564


namespace catchup_time_l351_351166

-- Define speeds for Xiaoming and Father
def v_X := 15 -- Xiaoming's speed in km/h
def v_F := 30 -- Father's speed in km/h

-- Define times in hours
def starting_time_X := 8 + 8 / 60 -- 8:08 AM in hours
def starting_time_F := 8 + 16 / 60 -- 8:16 AM in hours

-- Define distances in kilometers
def first_catchup_dist := 4 -- 4 km
def second_catchup_dist := 8 -- 8 km

-- Helper function to convert hours to time in HH:MM format
def hours_to_time (h : ℝ) : String :=
  let h_int : ℕ := h.to_nat
  let m_int : ℕ := ((h - h_int.to_real) * 60).to_nat
  let hour_str : String := if h_int < 10 then "0" ++ toString h_int else toString h_int
  let min_str : String := if m_int < 10 then "0" ++ toString m_int else toString m_int
  hour_str ++ ":" ++ min_str

-- Function to calculate the total time it took for the second catch-up.
def total_time_second_catchup : ℝ :=
  first_catchup_dist / v_F + -- Time to first catchup
  first_catchup_dist / v_F + -- Time to return home
  3 / 60 + -- 3 minute wait in hours
  (second_catchup_dist - first_catchup_dist) / v_F -- Time to second catchup

-- Final time when father catches Xiaoming the second time
def final_time : String :=
  hours_to_time (starting_time_X + total_time_second_catchup)

-- Lean theorem statement
theorem catchup_time : final_time = "08:32" :=
sorry

end catchup_time_l351_351166


namespace sum_of_coordinates_of_A_l351_351588

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351588


namespace sum_of_first_10_common_elements_l351_351431

noncomputable def sum_first_common_elements : ℕ :=
  let common_elements := (20 : ℕ) :: (80 : ℕ) :: (320 : ℕ) :: (1280 : ℕ) :: (5120 : ℕ) ::
                        (20480 : ℕ) :: (81920 : ℕ) :: (327680 : ℕ) :: (1310720 : ℕ) :: (5242880 : ℕ) :: []
  in common_elements.sum

theorem sum_of_first_10_common_elements :
  sum_first_common_elements = 6990500 :=
by
  -- Insert mathematical proof here
  sorry

end sum_of_first_10_common_elements_l351_351431


namespace john_monthly_paintball_cost_l351_351640

theorem john_monthly_paintball_cost (sessions_per_month boxes_per_session cost_per_box : ℕ)
  (h₁ : sessions_per_month = 3) (h₂ : boxes_per_session = 3) (h₃ : cost_per_box = 25) :
  sessions_per_month * (boxes_per_session * cost_per_box) = 225 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end john_monthly_paintball_cost_l351_351640


namespace cos_330_eq_sqrt3_div_2_l351_351188

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351188


namespace solution_set_of_quadratic_inequality_2_l351_351787

-- Definitions
variables {a b c x : ℝ}
def quadratic_inequality_1 (a b c x : ℝ) := a * x^2 + b * x + c < 0
def quadratic_inequality_2 (a b c x : ℝ) := a * x^2 - b * x + c > 0

-- Conditions
axiom condition_1 : ∀ x, quadratic_inequality_1 a b c x ↔ (x < -2 ∨ x > -1/2)
axiom condition_2 : a < 0
axiom condition_3 : ∃ x, a * x^2 + b * x + c = 0 ∧ (x = -2 ∨ x = -1/2)
axiom condition_4 : b = 5 * a / 2
axiom condition_5 : c = a

-- Proof Problem
theorem solution_set_of_quadratic_inequality_2 : ∀ x, quadratic_inequality_2 a b c x ↔ (1/2 < x ∧ x < 2) :=
by
  -- Proof goes here
  sorry

end solution_set_of_quadratic_inequality_2_l351_351787


namespace value_of_k_l351_351618

theorem value_of_k (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = (k - 1) * x + k^2 - 1)
  (h2 : ∃ m : ℝ, y = m * x)
  (h3 : k ≠ 1) :
  k = -1 :=
by
  sorry

end value_of_k_l351_351618


namespace roots_on_unit_circle_l351_351001

open Complex

noncomputable def P (z : ℂ) : ℂ := sorry -- Placeholder for the polynomial P(z)

theorem roots_on_unit_circle
  (n : ℕ)
  (P_roots_on_unit_circle : ∀ z : ℂ, P z = 0 → abs z = 1)
  (c : ℝ)
  (hc : c ≥ 0) :
  ∀ z : ℂ, (2 * z * (z - 1) * (P' z) + ((c - n : ℂ) * z + (c + n : ℂ)) * P z) = 0 → abs z = 1 :=
sorry

end roots_on_unit_circle_l351_351001


namespace total_savings_during_sale_l351_351444

theorem total_savings_during_sale :
  let regular_price_fox := 15
  let regular_price_pony := 20
  let pairs_fox := 3
  let pairs_pony := 2
  let total_discount := 22
  let discount_pony := 18.000000000000014
  let regular_total := (pairs_fox * regular_price_fox) + (pairs_pony * regular_price_pony)
  let discount_fox := total_discount - discount_pony
  (discount_fox / 100 * (pairs_fox * regular_price_fox)) + (discount_pony / 100 * (pairs_pony * regular_price_pony)) = 9 := by
  sorry

end total_savings_during_sale_l351_351444


namespace domain_of_f_l351_351988

noncomputable def f (x : ℝ) := real.cbrt (x - 5) + real.sqrt (9 - x)

theorem domain_of_f : ∀ x : ℝ, (x ≤ 9) ↔ ∃ y, f y = f x :=
by
  intros
  sorry

end domain_of_f_l351_351988


namespace jenny_games_l351_351630

theorem jenny_games (M : ℕ) 
  (h1 : ∀ J : ℕ, J = 2 * M)
  (h2 : ∀ JW : ℕ, JW = 2 * M - 0.75 * 2 * M)
  (h3 : ∀ MW : ℕ, MW = M - 1)
  (h4 : MW + JW = 14) 
  : M = 30 := sorry

end jenny_games_l351_351630


namespace product_remainder_mod_7_l351_351765

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351765


namespace jennifer_fruits_left_l351_351629

open Nat

theorem jennifer_fruits_left :
  (p o a g : ℕ) → p = 10 → o = 20 → a = 2 * p → g = 2 → (p - g) + (o - g) + (a - g) = 44 :=
by
  intros p o a g h_p h_o h_a h_g
  rw [h_p, h_o, h_a, h_g]
  sorry

end jennifer_fruits_left_l351_351629


namespace cost_per_top_l351_351938
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l351_351938


namespace remainder_of_product_mod_7_l351_351759

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351759


namespace books_left_to_read_l351_351859

theorem books_left_to_read (total_books books_read : ℕ) (h_total_books : total_books = 14) (h_books_read : books_read = 8) :
  total_books - books_read = 6 :=
by
  rw [h_total_books, h_books_read]
  exact rfl

end books_left_to_read_l351_351859


namespace distribution_methods_correctness_l351_351795

theorem distribution_methods_correctness : ∀ (graduates classes : ℕ),
  graduates = 5 →
  classes = 3 →
  ∃! (n : ℕ), (n = 150) ∧ (∃ (distributions : list (list ℕ)),
    distributions = [[1, 1, 3], [1, 2, 2]] ∧
    ∀ d ∈ distributions, (d.sum = graduates) ∧ (d.length = classes)) :=
by
  intros graduates classes h_graduates h_classes
  use 150
  split
  sorry

end distribution_methods_correctness_l351_351795


namespace distance_between_centers_l351_351934

theorem distance_between_centers (r1 r2 d x : ℝ) (h1 : r1 = 10) (h2 : r2 = 6) (h3 : d = 30) :
  x = 2 * Real.sqrt 229 := 
sorry

end distance_between_centers_l351_351934


namespace negative_number_in_options_l351_351160

theorem negative_number_in_options :
  ∀ (x : ℤ), x ∈ { -( -3), | -3|, (-3)^2, (-3)^3 } → x < 0 ↔ x = (-3)^3 :=
by
  intros x hx
  rw [Set.mem_insert_iff, Set.mem_insert_iff, Set.mem_insert_iff, Set.mem_singleton_iff] at hx
  cases hx with ha ha
  { sorry }   -- case hx is -( -3) = 3, which is not < 0
  cases ha with hb hb
  { sorry }   -- case hx is | -3| = 3, which is not < 0
  cases hb with hc hc
  { sorry }   -- case hx is (-3)^2 = 9, which is not < 0
  { sorry }   -- case hx is (-3)^3 = -27, which is < 0

end negative_number_in_options_l351_351160


namespace cos_330_cos_30_val_answer_l351_351235

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351235


namespace evaluate_Y_l351_351503

def Y (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2 + 3

theorem evaluate_Y : Y 2 5 = 2 :=
by
  sorry

end evaluate_Y_l351_351503


namespace number_of_orders_l351_351545

open Nat

theorem number_of_orders (total_targets : ℕ) (targets_A : ℕ) (targets_B : ℕ) (targets_C : ℕ)
  (h1 : total_targets = 10)
  (h2 : targets_A = 4)
  (h3 : targets_B = 3)
  (h4 : targets_C = 3)
  : total_orders = 80 :=
sorry

end number_of_orders_l351_351545


namespace a_4_is_minus_2_over_13_l351_351382

noncomputable def sequence_a : ℕ → ℚ
| 1       := 2
| 2       := 1 / 3
| (n + 3) := (sequence_a (n + 1) * sequence_a (n + 2)) / (sequence_a (n + 1) - 3 * sequence_a (n + 2))

theorem a_4_is_minus_2_over_13 : sequence_a 4 = - (2 : ℚ) / 13 :=
sorry

end a_4_is_minus_2_over_13_l351_351382


namespace prime_mod3_q_l351_351667

theorem prime_mod3_q (p q m n : ℕ) (h_prime: Prime p)
  (h_pmod3: p % 3 = 1)
  (h_q: q = 2 * (p / 3))
  (h_sum: ∑ i in Finset.range ((q / 2) - 1), (1 : ℚ) / ((2 * i + 1) * (2 * i + 2)) = (m : ℚ) / n)
  (h_coprime: Nat.coprime m n): p ∣ m :=
by
  sorry

end prime_mod3_q_l351_351667


namespace cos_330_eq_sqrt_3_div_2_l351_351365

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351365


namespace cos_330_eq_sqrt3_div_2_l351_351354

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351354


namespace number_sum_of_two_positive_integer_squares_l351_351033

theorem number_sum_of_two_positive_integer_squares (n : ℕ) (hn : n > 0) :
  (∃ k m : ℕ, k > 0 ∧ m > 0 ∧ n * (n + 1) = k * (k + 1) + m * (m + 1)) ↔
  (¬nat.prime (2 * n^2 + 2 * n + 1)) :=
by sorry

end number_sum_of_two_positive_integer_squares_l351_351033


namespace math_problem_l351_351504

noncomputable def log_8 := Real.log 8
noncomputable def log_27 := Real.log 27
noncomputable def expr := (9 : ℝ) ^ (log_8 / log_27) + (2 : ℝ) ^ (log_27 / log_8)

theorem math_problem : expr = 7 := by
  sorry

end math_problem_l351_351504


namespace abe_mia_matching_jelly_bean_probability_l351_351899

-- Definitions based on the conditions
def abe_jelly_beans : List (String × ℕ) := [("green", 2), ("blue", 1)]
def mia_jelly_beans : List (String × ℕ) := [("green", 2), ("yellow", 2), ("blue", 3)]
def total_jelly_beans (jelly_beans : List (String × ℕ)) : ℕ := jelly_beans.foldl (λ acc jb, acc + jb.snd) 0

-- Compute probabilities
def probability_of (jelly_beans : List (String × ℕ)) (color : String) : ℚ :=
  let color_count := jelly_beans.find (λjb, jb.fst = color) |>.map (λjb, jb.snd) |>.getD 0
  color_count / total_jelly_beans jelly_beans

-- Main theorem statement
theorem abe_mia_matching_jelly_bean_probability :
  (probability_of abe_jelly_beans "green" * probability_of mia_jelly_beans "green" +
   probability_of abe_jelly_beans "blue" * probability_of mia_jelly_beans "blue") = 1 / 3 :=
by 
  sorry

end abe_mia_matching_jelly_bean_probability_l351_351899


namespace cos_330_eq_sqrt_3_div_2_l351_351363

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351363


namespace number_of_participants_is_square_l351_351534

theorem number_of_participants_is_square
  (n k : ℕ)
  (h1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∨ 1 ≤ j ∧ j ≤ k → 1 ≤ i ∧ i ≤ n ∨ 1 ≤ j ∧ j ≤ k)
  (h2 : ∀ i j : ℕ, (i ∈ finset.range (n + k)) ∧ (j ∈ finset.range (n + k)) ∧ (i ≠ j) → 1 = 1)
  (h3 : ∀ i : ℕ, (i ∈ finset.range (n + k)) ∧ (1 ≤ i ∧ i ≤ n ∨ 1 ≤ i ∧ i ≤ k) → ∃ b : ℕ, b = (n + k) / 2)
: ∃ m : ℕ, (n + k) = m^2 := 
sorry

end number_of_participants_is_square_l351_351534


namespace geometric_sequence_term_count_l351_351540

def a1 : ℝ := 1 / 2
def q : ℝ := 1 / 2
def an : ℝ := 1 / 32

theorem geometric_sequence_term_count (n : ℕ) 
  (h : an = a1 * q ^ (n - 1)) : n = 5 := 
by
  -- With the given conditions, we need to prove the statement.
  sorry

end geometric_sequence_term_count_l351_351540


namespace evaluate_expression_l351_351961

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end evaluate_expression_l351_351961


namespace tangent_line_k_value_l351_351996

theorem tangent_line_k_value : ∃ k : ℝ, (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → k = 49) :=
begin
  sorry
end

end tangent_line_k_value_l351_351996


namespace sector_bisection_smaller_area_l351_351142

theorem sector_bisection_smaller_area (r : ℝ) (O A C A' : ℝ) 
  (h₁ : ∠AOC = 60) 
  (h₂ : ∃ k, k = (OA + arc_length AC) / 2 ∧ equal_perimeters (part₁ k) (part₂ k)) :
  has_smaller_area (part_containing_center O) (other_part O) :=
by
  sorry -- proof is omitted

end sector_bisection_smaller_area_l351_351142


namespace cos_330_eq_sqrt3_div_2_l351_351352

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351352


namespace number_of_roots_l351_351075

theorem number_of_roots (a : ℝ) (h : ∀ f : ℝ → ℝ, (4^f - 4^(-f) = 2 * cos (a * f)) → (∃ n, n = 2007)) :
  ∃ m, m = 4014 :=
sorry

end number_of_roots_l351_351075


namespace remainder_product_l351_351748

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351748


namespace cos_330_eq_sqrt3_over_2_l351_351206

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351206


namespace distinct_students_27_l351_351942

variable (students_euler : ℕ) (students_fibonacci : ℕ) (students_gauss : ℕ) (overlap_euler_fibonacci : ℕ)

-- Conditions
def conditions : Prop := 
  students_euler = 12 ∧ 
  students_fibonacci = 10 ∧ 
  students_gauss = 11 ∧ 
  overlap_euler_fibonacci = 3

-- Question and correct answer
def distinct_students (students_euler students_fibonacci students_gauss overlap_euler_fibonacci : ℕ) : ℕ :=
  (students_euler + students_fibonacci + students_gauss) - overlap_euler_fibonacci

theorem distinct_students_27 : conditions students_euler students_fibonacci students_gauss overlap_euler_fibonacci →
  distinct_students students_euler students_fibonacci students_gauss overlap_euler_fibonacci = 27 :=
by
  sorry

end distinct_students_27_l351_351942


namespace ab_cd_zero_l351_351492

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1)
  (h3 : a * c + b * d = 0) : 
  a * b + c * d = 0 := 
by sorry

end ab_cd_zero_l351_351492


namespace cos_330_eq_sqrt3_div_2_l351_351183

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351183


namespace cos_330_eq_sqrt3_div_2_l351_351328

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351328


namespace area_of_triangle_is_correct_l351_351984

noncomputable def calculate_area : ℝ :=
  let A := (0, 8, 11)
  let B := (-2, 7, 7)
  let C := (-5, 10, 7)
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let BC := (C.1 - B.1, C.2 - B.2, C.3 - B.3)
  let cross_product := 
    (AB.2 * BC.3 - AB.3 * BC.2, 
     AB.3 * BC.1 - AB.1 * BC.3, 
     AB.1 * BC.2 - AB.2 * BC.1)
  let magnitude := 
    Math.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  magnitude / 2

theorem area_of_triangle_is_correct :
  let A := (0, 8, 11)
  let B := (-2, 7, 7)
  let C := (-5, 10, 7)
  calculate_area = Math.sqrt 513 / 2 :=
begin
  sorry
end

end area_of_triangle_is_correct_l351_351984


namespace cos_330_eq_sqrt3_div_2_l351_351259

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351259


namespace gcf_75_100_l351_351816

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351816


namespace sequences_count_l351_351028

open Function

-- Define the vertices of the rectangle
def A : (ℝ × ℝ) := (2, 1)
def B : (ℝ × ℝ) := (-2, 1)
def C : (ℝ × ℝ) := (-2, -1)
def D : (ℝ × ℝ) := (2, -1)

-- Define the transformations L, H, V
def L : (ℝ × ℝ) → (ℝ × ℝ)
| (x, y) => (-x, -y)

def H : (ℝ × ℝ) → (ℝ × ℝ)
| (x, y) => (x, -y)

def V : (ℝ × ℝ) → (ℝ × ℝ)
| (x, y) => (-x, y)

-- Prove that the number of sequences of 10 transformations chosen from {L, H, V} that return all labeled vertices to their original positions is 21.
theorem sequences_count : 
  (∃ L H V : (ℝ × ℝ) → (ℝ × ℝ), 
   L (A) = C ∧ L (B) = D ∧ L (C) = A ∧ L (D) = B ∧  
   H (A) = D ∧ H (B) = C ∧ H (C) = B ∧ H (D) = A ∧ 
   V (A) = B ∧ V (B) = A ∧ V (C) = D ∧ V (D) = C) →
   ∑ (x y z : ℕ) in {k | k ≤ 10 ∧ k % 2 = 0}, (x + y + z = 10) → k = 21 :=
sorry

end sequences_count_l351_351028


namespace alternating_draws_probability_l351_351875

noncomputable def probability_alternating_draws : ℚ :=
  let total_draws := 11
  let white_balls := 5
  let black_balls := 6
  let successful_sequences := 1
  let total_sequences := @Nat.choose total_draws black_balls
  successful_sequences / total_sequences

theorem alternating_draws_probability :
  probability_alternating_draws = 1 / 462 := by
  sorry

end alternating_draws_probability_l351_351875


namespace water_bottles_needed_l351_351970

-- Define the conditions as constants
def num_people : ℕ := 10
def water_per_person_per_hour : ℚ := 1 / 2
def total_hours : ℕ := 24

-- Define the assertion to prove
theorem water_bottles_needed : num_people * water_per_person_per_hour * total_hours = 120 := 
by 
  calc 
    num_people * water_per_person_per_hour * ↑total_hours
    = 10 * (1 / 2) * 24 : by simp [num_people, water_per_person_per_hour, total_hours]
    ... = 10 * 12 : by norm_num
    ... = 120 : by norm_num

end water_bottles_needed_l351_351970


namespace n_squared_sum_of_squares_l351_351627

theorem n_squared_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) : 
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
by 
  sorry

end n_squared_sum_of_squares_l351_351627


namespace cos_330_eq_sqrt3_div_2_l351_351337

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351337


namespace expression_value_l351_351735

theorem expression_value (a b : ℝ) (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1 / 3 := 
by 
  sorry

end expression_value_l351_351735


namespace dot_product_result_l351_351497

open scoped BigOperators

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 2)

-- Define the addition of two vectors
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_result : dot_product (vector_add a b) a = 5 := by
  sorry

end dot_product_result_l351_351497


namespace find_m_for_opposite_solutions_l351_351521

theorem find_m_for_opposite_solutions (x y m : ℝ) 
  (h1 : x = -y)
  (h2 : 3 * x + 5 * y = 2)
  (h3 : 2 * x + 7 * y = m - 18) : 
  m = 23 :=
sorry

end find_m_for_opposite_solutions_l351_351521


namespace hyperbola_equation_l351_351455

noncomputable def hyperbola_standard_eq (a b : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, C x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def problem_conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ b = sqrt 3 * a ∧ b = 2 * sqrt 3

theorem hyperbola_equation :
  ∃ a b : ℝ, problem_conditions a b ∧ hyperbola_standard_eq 2 (2 * sqrt 3) (λ x y, (x^2 / 4) - (y^2 / 12) = 1) :=
begin
  sorry
end

end hyperbola_equation_l351_351455


namespace cos_330_eq_sqrt3_div_2_l351_351327

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351327


namespace evaluate_log_base_4_l351_351974

def log_base_4_256_minus_log_base_4_16 : Prop :=
  let a := Real.log 256 / Real.log 4 in
  let b := Real.log 16 / Real.log 4 in
  a - b = 2

theorem evaluate_log_base_4 : log_base_4_256_minus_log_base_4_16 :=
by
  sorry

end evaluate_log_base_4_l351_351974


namespace common_chord_of_circles_l351_351727

theorem common_chord_of_circles : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 2*x = 0 ∧ x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := 
by 
  sorry

end common_chord_of_circles_l351_351727


namespace overall_average_marks_is_57_l351_351544

-- Define the number of students and average mark per class
def students_class_A := 26
def avg_marks_class_A := 40

def students_class_B := 50
def avg_marks_class_B := 60

def students_class_C := 35
def avg_marks_class_C := 55

def students_class_D := 45
def avg_marks_class_D := 65

-- Define the total marks per class
def total_marks_class_A := students_class_A * avg_marks_class_A
def total_marks_class_B := students_class_B * avg_marks_class_B
def total_marks_class_C := students_class_C * avg_marks_class_C
def total_marks_class_D := students_class_D * avg_marks_class_D

-- Define the grand total of marks
def grand_total_marks := total_marks_class_A + total_marks_class_B + total_marks_class_C + total_marks_class_D

-- Define the total number of students
def total_students := students_class_A + students_class_B + students_class_C + students_class_D

-- Define the overall average marks
def overall_avg_marks := grand_total_marks / total_students

-- The target theorem we want to prove
theorem overall_average_marks_is_57 : overall_avg_marks = 57 := by
  sorry

end overall_average_marks_is_57_l351_351544


namespace radius_of_circle_in_spherical_coords_l351_351783

theorem radius_of_circle_in_spherical_coords :
  ∀ θ : ℝ, (∃ ρ φ : ℝ, ρ = 2 ∧ φ = π/4) →
  ∃ r : ℝ, r = ∥(2 * sin (π/4) * cos θ, 2 * sin (π/4) * sin θ, 2 * cos (π/4)).1.1 ∥ :=
begin
  intro θ,
  intro h,
  use sqrt 2,
  have : ∥(sqrt 2 * cos θ, sqrt 2 * sin θ, sqrt 2).1.1∥ = sqrt 2,
  sorry -- Skip the actual proof
end

end radius_of_circle_in_spherical_coords_l351_351783


namespace prob_at_least_one_interested_theorem_l351_351157

noncomputable def prob_at_least_one_interested 
    (total_members : ℕ) 
    (fraction_interested : ℚ)
    (students_chosen : ℕ) : ℚ :=
begin
  assume htotal : total_members = 20,
  assume hfrac : fraction_interested = 3/4,
  assume hchosen : students_chosen = 2,
  
  let interested := fraction_interested * total_members,
  have h_interested : interested = 15 := by norm_num [htotal, hfrac],

  let not_interested := total_members - interested,
  have h_not_interested : not_interested = 5 := by norm_num [h_interested, htotal],

  let prob_none_interested := (not_interested / total_members) * ((not_interested - 1) / (total_members - 1)),
  have h_prob_none_interested : prob_none_interested = 1/19 := by norm_num [not_interested, total_members],

  let prob_at_least_one := 1 - prob_none_interested,
  have h_prob_at_least_one : prob_at_least_one = 18 / 19 := by norm_num [prob_none_interested],

  exact h_prob_at_least_one
end

theorem prob_at_least_one_interested_theorem :
  prob_at_least_one_interested 20 (3/4) 2 = 18 / 19 := 
by sorry

end prob_at_least_one_interested_theorem_l351_351157


namespace sum_coordinates_A_l351_351600

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351600


namespace cos_330_eq_sqrt3_div_2_l351_351339

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351339


namespace area_of_circle_with_diameter_10_l351_351096

theorem area_of_circle_with_diameter_10 (d : ℝ) (π : ℝ) (h : d = 10): 
  ∃ A, A = π * ((d / 2) ^ 2) ∧ A = 25 * π :=
begin
  use π * ((10 / 2) ^ 2),
  split,
  { rw h, },
  { ring }
end

end area_of_circle_with_diameter_10_l351_351096


namespace product_mod_7_l351_351738

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351738


namespace parabola_focus_l351_351516

theorem parabola_focus (a : ℝ) (h : a ≠ 0) (h_directrix : ∀ x y : ℝ, y^2 = a * x → x = -1) : 
    ∃ x y : ℝ, (y = 0 ∧ x = 1 ∧ y^2 = a * x) :=
sorry

end parabola_focus_l351_351516


namespace cos_330_eq_sqrt3_div_2_l351_351362

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351362


namespace radius_of_circle_is_sqrt_two_l351_351780

-- Definitions based on the conditions given in the problem
def rho : ℝ := 2
def phi : ℝ := Real.pi / 4

-- Lean statement of the proof problem
theorem radius_of_circle_is_sqrt_two (theta : ℝ) :
  let x := rho * Real.sin phi * Real.cos theta
  let y := rho * Real.sin phi * Real.sin theta
  sqrt (x^2 + y^2) = sqrt 2 :=
sorry

end radius_of_circle_is_sqrt_two_l351_351780


namespace sums_of_coordinates_of_A_l351_351610

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351610


namespace mike_becomes_champion_l351_351681

-- Define player and game outcomes
inductive Player
| Mike
| Alain

-- Define the game's win condition and initial state
def win_condition (mike_wins alain_wins : ℕ) : bool :=
  if mike_wins = 3 then true
  else if alain_wins = 3 then false
  else false

def initial_game : (mike_wins : ℕ) × (alain_wins : ℕ) := (1, 0)

-- Specify the probability of winning for each player
def game_probability : Rational := 1 / 2

-- Define sequences and their probabilities
def sequence_prob : list Player → Rational
| []          := 1
| (Player.Mike :: xs) := game_probability * sequence_prob xs
| (Player.Alain :: xs) := game_probability * sequence_prob xs

-- List all possible winning paths for Mike
def winning_sequences : list (list Player) := [
  [Player.Mike, Player.Mike],              -- MMM
  [Player.Mike, Player.Alain, Player.Mike], -- MMAM
  [Player.Mike, Player.Mike, Player.Alain, Player.Mike], -- MMAAM
  [Player.Alain, Player.Mike, Player.Mike], -- MAMM
  [Player.Alain, Player.Mike, Player.Alain, Player.Mike], -- MAMAM
  [Player.Alain, Player.Alain, Player.Mike, Player.Mike]  -- MAAMM
]

-- Calculate the total probability
def total_prob : Rational :=
  (winning_sequences.map sequence_prob).sum

-- The theorem statement
theorem mike_becomes_champion :
  total_prob = 11 / 16 :=
by sorry

end mike_becomes_champion_l351_351681


namespace train_pass_time_l351_351864

-- Define the conditions
def length_of_train : ℝ := 110  -- in meters
def speed_of_train : ℝ := 24    -- in km/hr
def speed_of_man : ℝ := 6       -- in km/hr
def conversion_factor : ℝ := 5 / 18 -- to convert km/hr to m/s

-- Define the relative speed, converted to m/s
def relative_speed : ℝ := (speed_of_train + speed_of_man) * conversion_factor

-- Define the time taken for the train to pass the man
def time_to_pass : ℝ := length_of_train / relative_speed

-- The statement to be proved
theorem train_pass_time : time_to_pass ≈ 13.20 := by
  sorry

end train_pass_time_l351_351864


namespace distance_between_ellipse_foci_l351_351917

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351917


namespace absolute_value_equation_sum_l351_351850

theorem absolute_value_equation_sum (x1 x2 : ℝ) (h1 : 3 * x1 - 12 = 6) (h2 : 3 * x2 - 12 = -6) : x1 + x2 = 8 := 
sorry

end absolute_value_equation_sum_l351_351850


namespace prob_identical_painted_dice_l351_351086

theorem prob_identical_painted_dice :
  let total_ways_per_die := 3^6 in
  let total_ways_two_dice := total_ways_per_die * total_ways_per_die in
  let identical_ways := 3 + 216 + 540 + 360 in
  (identical_ways : ℚ) / total_ways_two_dice = 1119 / 531441 := by
  sorry

end prob_identical_painted_dice_l351_351086


namespace taxi_ride_cost_l351_351941

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l351_351941


namespace v_closed_only_under_multiplication_l351_351668

/-- Define the set v, consisting of cubes of positive integers -/
def v : Set ℕ := { x | ∃ n : ℕ, x = n^3 }

/-- Prove the set v is closed under specific operations -/
theorem v_closed_only_under_multiplication :
  (∀ a b ∈ v, a * b ∈ v) ∧
  ¬(∀ a b ∈ v, a + b ∈ v) ∧
  ¬(∀ a b ∈ v, a / b ∈ v) ∧
  ¬(∀ a ∈ v, ∃ n : ℕ, n^3 = a) :=
by
  sorry

end v_closed_only_under_multiplication_l351_351668


namespace cos_330_eq_sqrt3_div_2_l351_351309

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351309


namespace log_bounds_l351_351788

theorem log_bounds (a b : ℕ) (log_a log_b : ℝ) (x : ℕ) (log_x : ℝ)
  (h1 : a < x) (h2 : x < b)
  (hl_a : log_a = real.log10 a) 
  (hl_b : log_b = real.log10 b) 
  (hl_x : log_x = real.log10 x) :
  4 < log_x ∧ log_x < 5 → 4 + 5 = 9 := 
by
  sorry

end log_bounds_l351_351788


namespace cos_330_eq_sqrt3_div_2_l351_351335

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351335


namespace simplify_sqrt_forth_root_l351_351038

/-- Given expressions and their powers --/
def exp1 := 2 ^ 9 * 3 ^ 5
def exp2 := 16 * (486 ^ (1 / 4 : ℝ))

/-- Main theorem statement with the final sum of a and b --/
theorem simplify_sqrt_forth_root : (exp2 + 486) = 502 :=
by 
  sorry

end simplify_sqrt_forth_root_l351_351038


namespace ex1_simplified_ex2_simplified_l351_351180

-- Definitions and problem setup
def ex1 (a : ℝ) : ℝ := ((-a^3)^2 * a^3 - 4 * a^2 * a^7)
def ex2 (a : ℝ) : ℝ := (2 * a + 1) * (-2 * a + 1)

-- Proof goals
theorem ex1_simplified (a : ℝ) : ex1 a = -3 * a^9 :=
by sorry

theorem ex2_simplified (a : ℝ) : ex2 a = 4 * a^2 - 1 :=
by sorry

end ex1_simplified_ex2_simplified_l351_351180


namespace max_rectangles_in_triangle_l351_351020

theorem max_rectangles_in_triangle : 
  (∃ (n : ℕ), n = 192 ∧ 
  ∀ (i j : ℕ), i + j < 7 → ∀ (a b : ℕ), a ≤ 6 - i ∧ b ≤ 6 - j → 
  ∃ (rectangles : ℕ), rectangles = (6 - i) * (6 - j)) :=
sorry

end max_rectangles_in_triangle_l351_351020


namespace gcf_75_100_l351_351817

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351817


namespace transform_quadratic_function_l351_351963

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 24 * x - 45

theorem transform_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function x = a * (x + b)^2 + c) ∧ a + b + c = 4 :=
by {
  use [-3, 4, 3],
  split,
  { intro x,
    calc
      quadratic_function x = -3 * x^2 + 24 * x - 45 : rfl
                      ... = -3 * (x^2 - 8 * x + 16 - 16) - 45 : by rw [sub_add_cancel]
                      ... = -3 * ((x - 4)^2 - 16) - 45 : by ring
                      ... = -3 * (x - 4)^2 + 48 - 45 : by ring
                      ... = -3 * (x - 4)^2 + 3 : by ring },
  { norm_num }
}

end transform_quadratic_function_l351_351963


namespace sum_of_coordinates_A_l351_351559

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351559


namespace infinitely_many_primes_dividing_2k_minus_3_l351_351701

theorem infinitely_many_primes_dividing_2k_minus_3 :
  ∃ᶠ p : ℕ in filter.at_top, ∃ k : ℕ, k > 0 ∧ p.prime ∧ p ∣ (2^k - 3) := sorry

end infinitely_many_primes_dividing_2k_minus_3_l351_351701


namespace find_A_coordinates_sum_l351_351577

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351577


namespace parents_rating_needs_improvement_l351_351146

-- Define the conditions as variables and expressions
def total_parents : ℕ := 120
def percentage_excellent : ℕ := 15
def percentage_very_satisfactory : ℕ := 60
def percentage_satisfactory_of_remaining : ℕ := 80

-- Define the calculation as a Lean theorem
theorem parents_rating_needs_improvement : 
  let num_excellent := (percentage_excellent * total_parents) / 100
  let num_very_satisfactory := (percentage_very_satisfactory * total_parents) / 100
  let remaining_parents := total_parents - num_excellent - num_very_satisfactory
  let num_satisfactory := (percentage_satisfactory_of_remaining * remaining_parents) / 100
  let num_needs_improvement := remaining_parents - num_satisfactory
  in num_needs_improvement = 6 :=
by 
  -- Defer the proof
  sorry

end parents_rating_needs_improvement_l351_351146


namespace modulus_of_complex_l351_351409

-- Define the complex number z as (7/4) - 3i
def z : ℂ := (7 / 4 : ℝ) - 3 * complex.I

-- Statement to prove
theorem modulus_of_complex : complex.abs z = real.sqrt 193 / 4 :=
by
  sorry

end modulus_of_complex_l351_351409


namespace increasing_quadratic_l351_351728

theorem increasing_quadratic (k : ℝ) : 
  (∀ x y ∈ set.Icc (0 : ℝ) (14 : ℝ), x < y → f x ≤ f y) ↔ k ≤ 0 :=
begin
  sorry
end where f (x : ℝ) : ℝ := x^2 - 2 * k * x - 8

end increasing_quadratic_l351_351728


namespace find_number_l351_351852

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end find_number_l351_351852


namespace angle_between_unit_vectors_l351_351496

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def unit_vector_angle (a b : ℝ × ℝ) : Prop :=
  norm a = 1 ∧ norm b = 1 ∧ norm ⟨a.1 + b.1, a.2 + b.2⟩ = 1 → 
  real.arccos (dot_product a b / (norm a * norm b)) = 2 * real.pi / 3

theorem angle_between_unit_vectors (a b : ℝ × ℝ) :
  unit_vector_angle a b :=
sorry

end angle_between_unit_vectors_l351_351496


namespace geometric_mean_EF_l351_351625

open EuclideanGeometry

variables {A B C D H E F : Point}

-- conditions: A triangle ABC
variable (triangle_ABC : Triangle A B C)

-- D is the point where the angle bisector from B intersects AC
variable (isAngleBisectorBD : isAngleBisector B D (Side A C) (Side A B))

-- H is the point where the altitude from B intersects AC
variable (isAltitudeBH : isAltitude B H (Side A C))

-- E is the point where the incircle touches AC
variable (isTangentPointE : isTangentPoint (incircle_triangle_ABC) E (Side A C))

-- F is the midpoint of AC
variable (isMidpointF : isMidpoint A C F)

-- Proof statement
theorem geometric_mean_EF (triangle_ABC : Triangle A B C)
  (isAngleBisectorBD : isAngleBisector B D (Side A C) (Side A B))
  (isAltitudeBH : isAltitude B H (Side A C))
  (isTangentPointE : isTangentPoint (incircle_triangle_ABC) E (Side A C))
  (isMidpointF : isMidpoint A C F) :
  distance E F * distance E F = distance F D * distance F H :=
by
  sorry

end geometric_mean_EF_l351_351625


namespace stream_speed_l351_351131

theorem stream_speed (v : ℝ) (h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v)))) : 
  v = 5 / 3 :=
by
  -- Variables and assumptions
  have h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v))) := sorry
  -- To prove
  sorry

end stream_speed_l351_351131


namespace remainder_product_l351_351753

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351753


namespace cos_330_eq_sqrt3_div_2_l351_351274

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351274


namespace cos_330_is_sqrt3_over_2_l351_351299

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351299


namespace cos_330_eq_sqrt3_div_2_l351_351329

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351329


namespace nero_wolfe_can_solve_case_l351_351385

def witnesses_and_criminals_problem (people : Fin 80) (days : Fin 12) : Prop :=
  ∃ w c : Fin 80, w ≠ c ∧
  (∀ i : Fin (people + 1), ∃ invitees : Set (Fin 80),
   invitees ⊆ (Fin 80).elems ∧
   (w ∈ invitees ∧ c ∉ invitees → ∃ day : Fin 12, day < days))

theorem nero_wolfe_can_solve_case : witnesses_and_criminals_problem 80 12 :=
sorry

end nero_wolfe_can_solve_case_l351_351385


namespace cos_330_eq_sqrt3_div_2_l351_351194

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351194


namespace cos_330_eq_sqrt3_div_2_l351_351320

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351320


namespace enrique_commission_l351_351401

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l351_351401


namespace treasure_in_heaviest_bag_l351_351090

theorem treasure_in_heaviest_bag (A B C D : ℝ) (h1 : A + B < C)
                                        (h2 : A + C = D)
                                        (h3 : A + D > B + C) : D > A ∧ D > B ∧ D > C :=
by 
  sorry

end treasure_in_heaviest_bag_l351_351090


namespace cream_ratio_l351_351633

-- Define the initial conditions for Joe and JoAnn
def initial_coffee : ℕ := 15
def initial_cup_size : ℕ := 20
def cream_added : ℕ := 3
def coffee_drank_by_joe : ℕ := 3
def mixture_stirred_by_joann : ℕ := 3

-- Define the resulting amounts of cream in Joe and JoAnn's coffee
def cream_in_joe : ℕ := cream_added
def cream_in_joann : ℝ := cream_added - (cream_added * (mixture_stirred_by_joann / (initial_coffee + cream_added)))

-- Prove the ratio of the amount of cream in Joe's coffee to that in JoAnn's coffee
theorem cream_ratio :
  (cream_in_joe : ℝ) / cream_in_joann = 6 / 5 :=
by
  -- The code is just a statement; the proof detail is omitted with sorry, and variables are straightforward math.
  sorry

end cream_ratio_l351_351633


namespace log_diff_example_l351_351973

theorem log_diff_example : 
  log 4 256 - log 4 16 = 2 :=
sorry

end log_diff_example_l351_351973


namespace tiles_cover_floor_l351_351139

/--
A rectangular floor is covered with congruent square tiles.
Each diagonal of the rectangle covers a whole number of tiles.
If the total number of tiles that lie on the two diagonals is 45,
and the ratio of length to width of the rectangle is 3:2,
prove that the total number of square tiles covering the floor is 245.
-/
theorem tiles_cover_floor :
  ∃ (a b : ℕ), 
    (a + b = 45) ∧ 
    (3 * b = 2 * a) ∧
    (a % 13 = 0) ∧
    (∃ (x : ℕ), a = 9 * x ∧ b = 4 * x) ∧
    let area := 2 * (a / 9) * 3 * (a / 9) in area = 245 := 
sorry

end tiles_cover_floor_l351_351139


namespace good_numbers_count_l351_351512

def is_good_number (n : ℕ) : Prop :=
  n / 10 % 10 = 1 ∧ 
  ((n / 10 / 10 / 10 = n / 10 / 10 % 10 ∧ n / 10 / 10 / 10 = n % 10) ∨ 
   (n / 10 / 10 = n / 10 % 10 × 100 + n / 10 % 10 × 10 + 1))

theorem good_numbers_count :
  {n : ℕ | is_good_number n}.to_finset.card = 12 := 
sorry

end good_numbers_count_l351_351512


namespace cos_330_cos_30_val_answer_l351_351242

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351242


namespace train_speeds_l351_351800

theorem train_speeds :
  ∃ (v w : ℕ), 
    let u := 45 in
    let t := 5 in
    let d := 450 in
    (2 * d) / 2 = 225 ∧
    v + (v + 6) = d / t ∧
    u = 225 / t ∧
    v = 42 ∧
    w = 48 :=
by
  sorry

end train_speeds_l351_351800


namespace cream_ratio_l351_351634

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l351_351634


namespace cos_330_eq_sqrt3_div_2_l351_351195

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351195


namespace evaluate_expression_l351_351040

noncomputable def expression (a : ℚ) : ℚ := 
  (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2 * a)

theorem evaluate_expression (a : ℚ) (ha : a = -1/3) : expression a = -2 :=
by 
  rw [expression, ha]
  sorry

end evaluate_expression_l351_351040


namespace balls_arrangement_is_correct_select_balls_is_correct_divide_balls_is_correct_l351_351077

open Finset

-- (1) Arrangement Proof
noncomputable def arrange_balls : ℕ := 
  let black_ways := factorial 4
  let red_ways := factorial 2
  let yellow_ways := combinatorial_conditions  -- to be detailed
  24 * 2 * 2 * 6

-- asserting our arrangement result
theorem balls_arrangement_is_correct : arrange_balls = 576 := sorry

-- (2) Selection Proof
noncomputable def select_balls : ℕ :=
  let scenario_1 := choose 4 1 * choose 2 1 * choose 2 2
  let scenario_2 := choose 4 1 * choose 2 2 * choose 2 1
  let scenario_3 := choose 4 2 * choose 2 1 * choose 2 1
  scenario_1 + scenario_2 + scenario_3

-- asserting our selection result
theorem select_balls_is_correct : select_balls = 40 := sorry

-- (3) Division Proof
noncomputable def divide_balls : ℕ :=
  let case1 := (choose 8 2 * choose 6 2) / factorial 2
  let case2 := (choose 8 2 * choose 6 3) / factorial 2
  case1 + case2

-- asserting our division result
theorem divide_balls_is_correct : divide_balls = 490 := sorry

end balls_arrangement_is_correct_select_balls_is_correct_divide_balls_is_correct_l351_351077


namespace remainder_of_product_l351_351772

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351772


namespace number_of_special_divisors_l351_351389

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end number_of_special_divisors_l351_351389


namespace cos_330_eq_sqrt3_div_2_l351_351313

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351313


namespace dumbbell_weight_l351_351670

theorem dumbbell_weight (bands_count : ℕ) (resistance_per_band total_weight : ℕ) :
  bands_count = 2 → resistance_per_band = 5 → total_weight = 30 → 
  (total_weight - bands_count * resistance_per_band = 20) :=
by
  intros bands_count_eq resistance_per_band_eq total_weight_eq
  rw [bands_count_eq, resistance_per_band_eq, total_weight_eq]
  simp
  sorry

end dumbbell_weight_l351_351670


namespace cos_330_eq_sqrt_3_div_2_l351_351367

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351367


namespace maximum_value_of_transformed_function_l351_351062

theorem maximum_value_of_transformed_function (a b : ℝ) (h_max : ∀ x : ℝ, a * (Real.cos x) + b ≤ 1)
  (h_min : ∀ x : ℝ, a * (Real.cos x) + b ≥ -7) : 
  ∃ ab : ℝ, (ab = 3 + a * b * (Real.sin x)) ∧ (∀ x : ℝ, ab ≤ 15) :=
by
  sorry

end maximum_value_of_transformed_function_l351_351062


namespace solve_inequality_l351_351417

-- Define the inequality
def inequality (x : ℚ) : Prop :=
  (7 / 30 : ℚ) + | x - (13 / 60 : ℚ) | < (11 / 20 : ℚ)

-- Define the interval
def interval (x : ℚ) : Prop :=
  (-1 / 5 : ℚ) < x ∧ x < (8 / 15 : ℚ)

-- The theorem statement
theorem solve_inequality (x : ℚ) : inequality x → interval x :=
  sorry

end solve_inequality_l351_351417


namespace technician_round_trip_l351_351863

theorem technician_round_trip (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  (traveled_distance / round_trip * 100) = 65 := by
  -- Definitions based on the given conditions
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  
  -- Placeholder for the proof to satisfy Lean syntax.
  sorry

end technician_round_trip_l351_351863


namespace gcd_75_100_l351_351809

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351809


namespace sqrt_product_simplification_l351_351949

variable (q : ℝ)

theorem sqrt_product_simplification (hq : q ≥ 0) : 
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by
  sorry

end sqrt_product_simplification_l351_351949


namespace cos_330_eq_sqrt3_div_2_l351_351334

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351334


namespace father_age_when_sum_100_l351_351016

/-- Given the current ages of the mother and father, prove that the father's age will be 51 years old when the sum of their ages is 100. -/
theorem father_age_when_sum_100 (M F : ℕ) (hM : M = 42) (hF : F = 44) :
  ∃ X : ℕ, (M + X) + (F + X) = 100 ∧ F + X = 51 :=
by
  sorry

end father_age_when_sum_100_l351_351016


namespace cos_330_eq_sqrt3_div_2_l351_351361

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351361


namespace find_angle_A_l351_351528

theorem find_angle_A (a b c A B C : ℝ)
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) :
  A = Real.pi / 6 :=
sorry

end find_angle_A_l351_351528


namespace distribution_methods_l351_351790

theorem distribution_methods :
  let volunteers := ["A", "B", "C", "M1", "M2"] in
  let communities := [c1, c2, c3] in
  let conditions := 
    ∀ c1 c2 c3 : set string, 
      ("A" ∈ c1 ∧ "B" ∈ c1) ∧ 
      ("M1" ∈ c2 ∨ "M1" ∈ c3) ∧ 
      ("M2" ∈ c1 ∨ "M2" ∈ c2 ∨ "M2" ∈ c3) ∧ 
      (∀ x y : string, x ≠ y → c1 ≠ c2 ∧ c2 ≠ c3) ∧
      (∀ x : string, c1 ∪ c2 ∪ c3 = {"A", "B", "C", "M1", "M2"}) ∧ 
      (∀ c : set string, 1 ≤ c.card ∧ c.card ≤ 2) in
  ∃ (number_of_methods : ℕ), number_of_methods = 12 :=
sorry

end distribution_methods_l351_351790


namespace total_carrots_l351_351706

theorem total_carrots (sandy_carrots: Nat) (sam_carrots: Nat) (h1: sandy_carrots = 6) (h2: sam_carrots = 3) : sandy_carrots + sam_carrots = 9 :=
by
  sorry

end total_carrots_l351_351706


namespace remainder_when_divided_by_x_plus_2_l351_351103

def q (x D E F : ℝ) : ℝ := D*x^4 + E*x^2 + F*x - 2

theorem remainder_when_divided_by_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 14) : q (-2) D E F = -18 := 
by 
     sorry

end remainder_when_divided_by_x_plus_2_l351_351103


namespace cost_of_gravelling_path_is_720_l351_351140

-- Definitions from the problem conditions
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65
def width_path : ℝ := 2.5
def cost_per_sq_meter_paise : ℝ := 80
def conversion_rate : ℝ := 1 / 100

-- Calculate the dimensions of the larger rectangle including the path
def length_larger_rectangle : ℝ := length_grassy_plot + 2 * width_path
def width_larger_rectangle : ℝ := width_grassy_plot + 2 * width_path

-- Calculate the areas
def area_larger_rectangle : ℝ := length_larger_rectangle * width_larger_rectangle
def area_grassy_plot : ℝ := length_grassy_plot * width_grassy_plot
def area_gravel_path : ℝ := area_larger_rectangle - area_grassy_plot

-- Convert cost from paise to rupees
def cost_per_sq_meter_rupees : ℝ := cost_per_sq_meter_paise * conversion_rate

-- Calculate total cost
def total_cost : ℝ := area_gravel_path * cost_per_sq_meter_rupees

-- Proof
theorem cost_of_gravelling_path_is_720 :
  total_cost = 720 :=
by
  sorry

end cost_of_gravelling_path_is_720_l351_351140


namespace sum_of_coordinates_A_l351_351558

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351558


namespace find_definite_SUM_l351_351654

noncomputable def d := 43
noncomputable def e := 10
noncomputable def f := 33
def x := (Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2))

theorem find_definite_SUM (x : ℝ) :
  (x = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2))
  → (x ^ 100 = 3 * x ^ 98 + 18 * x ^ 96 + 13 * x ^ 94 - x ^ 50 + d * x ^ 46 + e * x ^ 44 + f * x ^ 40)
  → d + e + f = 86 :=
by
  intros h1 h2
  have h_d := (43 : ℝ)
  have h_e := (10 : ℝ)
  have h_f := (33 : ℝ)
  sorry

end find_definite_SUM_l351_351654


namespace sums_of_coordinates_of_A_l351_351615

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351615


namespace ellipse_foci_distance_l351_351910

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351910


namespace value_of_x_squared_plus_y_squared_l351_351511

theorem value_of_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

end value_of_x_squared_plus_y_squared_l351_351511


namespace find_f_2011_l351_351474

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom symmetric_shift : ∀ x : ℝ, f(2 + x) = f(2 - x)
axiom given_condition : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f(x) = Real.log 2 (1 - x)

theorem find_f_2011 : f 2011 = 1 :=
by
  sorry

end find_f_2011_l351_351474


namespace ellipse_foci_distance_l351_351908

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351908


namespace find_function_l351_351982

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (x : ℝ) (h1 : x = 3) (h2 : x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  -- Given x = 3 and x + 17 = 60 * f x
  rw h1 at h2
  -- Showing the target f 3
  have h3 : 3 + 17 = 20 := by norm_num
  rw h3 at h2
  have h4 : 60 * f 3 = 20 := by rw h2
  -- Solving explicitly for f 3
  sorry

end find_function_l351_351982


namespace function_inequality_l351_351415

noncomputable def f (c : ℝ) (h_c : c > 1) (x : ℝ) (h_x : x > 1) := c ^ (1 / Real.log x)

theorem function_inequality (c : ℝ) (h_c : c > 1) 
  (x y : ℝ) (h_x : x > 1) (h_y : y > 1)
  (u v : ℝ) (h_u : u > 0) (h_v : v > 0) :
  f c h_c (x ^ u * y ^ v) (mul_pos (Real.rpow_pos_of_pos h_x u) (Real.rpow_pos_of_pos h_y v))
  ≤ (f c h_c x h_x) ^ (1 / (4 * u)) * (f c h_c y h_y) ^ (1 / (4 * v)) :=
sorry

end function_inequality_l351_351415


namespace symmetric_about_neg_pi_div_4_l351_351032
-- Import necessary Lean 4 libraries

-- Define the function f
def f (x : ℝ) : ℝ := cos (2 * x)

-- State the problem as a Lean theorem
theorem symmetric_about_neg_pi_div_4 :
  ∃ (x₀ : ℝ), ∀ x, f(x₀ - x) = f(x₀ + x) ∧ x₀ = - (π / 4) :=
sorry

end symmetric_about_neg_pi_div_4_l351_351032


namespace roots_seventh_sum_l351_351661

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l351_351661


namespace sum_of_coordinates_A_l351_351601

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351601


namespace gcd_75_100_l351_351810

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351810


namespace cos_330_eq_sqrt_3_div_2_l351_351375

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351375


namespace extremum_points_opposite_signs_l351_351093

theorem extremum_points_opposite_signs (p q : ℝ) ((q / 2)^2 + (p / 3)^3 < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 * f x2 < 0 :=
by
  let f := λ x : ℝ, x^3 + p * x + q
  sorry

end extremum_points_opposite_signs_l351_351093


namespace total_packs_is_117_l351_351688

-- Defining the constants based on the conditions
def nancy_cards : ℕ := 540
def melanie_cards : ℕ := 620
def mary_cards : ℕ := 480
def alyssa_cards : ℕ := 720

def nancy_cards_per_pack : ℝ := 18.5
def melanie_cards_per_pack : ℝ := 22.5
def mary_cards_per_pack : ℝ := 15.3
def alyssa_cards_per_pack : ℝ := 24

-- Calculating the number of packs each person has
def nancy_packs := (nancy_cards : ℝ) / nancy_cards_per_pack
def melanie_packs := (melanie_cards : ℝ) / melanie_cards_per_pack
def mary_packs := (mary_cards : ℝ) / mary_cards_per_pack
def alyssa_packs := (alyssa_cards : ℝ) / alyssa_cards_per_pack

-- Rounding down the number of packs
def nancy_packs_rounded := nancy_packs.toNat
def melanie_packs_rounded := melanie_packs.toNat
def mary_packs_rounded := mary_packs.toNat
def alyssa_packs_rounded := alyssa_packs.toNat

-- Summing the total number of packs
def total_packs : ℕ := nancy_packs_rounded + melanie_packs_rounded + mary_packs_rounded + alyssa_packs_rounded

-- Proposition stating that the total number of packs is 117
theorem total_packs_is_117 : total_packs = 117 := by
  sorry

end total_packs_is_117_l351_351688


namespace inequalities_not_equivalent_l351_351164

theorem inequalities_not_equivalent :
  ¬(∀ x, (sqrt (x - 1) < sqrt (2 - x) ↔ (x - 1 < 2 - x))) :=
by {
  have h_domain_first : ∀ x, (1 ≤ x ∧ x ≤ 2) → (sqrt (x - 1) < sqrt (2 - x)),
    sorry,
  have h_domain_second : ∀ x, x < 3 / 2,
    sorry,
  intro h,
  specialize h (-5),
  -- use some counterexample or contradiction here
  sorry
}

end inequalities_not_equivalent_l351_351164


namespace initial_points_l351_351030

theorem initial_points (total_points_after_scoring : ℕ) (additional_points : ℕ) (h : total_points_after_scoring = 95) (h_add : additional_points = 3) : 
  total_points_after_scoring - additional_points = 92 :=
by
  rw [h, h_add]
  exact (nat.sub_eq_of_eq_add h.symm).symm.sorry

end initial_points_l351_351030


namespace expected_jumps_l351_351445

noncomputable def E : ℕ → ℕ
| 24 := 0
| 0 := 1 + (2 * E 0 + E 1) / 3
| (n + 1) := 1 + (E (n + 2) + E n + 2 * E (n + 1)) / 4

theorem expected_jumps (y : ℕ) : y = 21 → E y = 273 :=
by sorry

end expected_jumps_l351_351445


namespace cos_330_eq_sqrt_3_div_2_l351_351214

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351214


namespace determine_a_value_l351_351967

theorem determine_a_value :
  ∀ (a b c d : ℕ), 
  (a = b + 3) →
  (b = c + 6) →
  (c = d + 15) →
  (d = 50) →
  a = 74 :=
by
  intros a b c d h1 h2 h3 h4
  sorry

end determine_a_value_l351_351967


namespace cos_330_eq_sqrt3_div_2_l351_351331

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351331


namespace gwen_shelves_mystery_books_l351_351498

theorem gwen_shelves_mystery_books
  (total_books : ℕ)
  (books_per_shelf : ℕ)
  (shelves_picture_books : ℕ)
  (shelves_mystery_books : ℕ) :
  total_books = 72 →
  books_per_shelf = 9 →
  shelves_picture_books = 5 →
  shelves_mystery_books = (total_books - shelves_picture_books * books_per_shelf) / books_per_shelf →
  shelves_mystery_books = 3 :=
begin
  sorry
end

end gwen_shelves_mystery_books_l351_351498


namespace face_opposite_C_is_E_l351_351883

-- Define the types for the labels A, B, C, D, E, F
inductive Label
  | A | B | C | D | E | F

open Label

-- Conditions as definitions, proceed to prove the final statement
def adjacent (x y : Label) : Prop := 
  -- A simple adjacency relation placeholder
  sorry

axiom adj_1 : adjacent B A
axiom adj_2 : adjacent C B ∧ ¬ adjacent C A
axiom adj_3 : adjacent D A ∧ adjacent D F

theorem face_opposite_C_is_E : ∀ faces, faces ∈ {A, B, C, D, E, F} → adjacent C faces → ¬ adjacent faces E → faces = E :=
by sorry

end face_opposite_C_is_E_l351_351883


namespace sum_of_coords_A_l351_351572

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351572


namespace log_diff_example_l351_351972

theorem log_diff_example : 
  log 4 256 - log 4 16 = 2 :=
sorry

end log_diff_example_l351_351972


namespace zero_in_interval_l351_351487

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.logb 2 x

theorem zero_in_interval : (f 2 > 0) ∧ (f 4 < 0) → ∃ c ∈ Ioo 2 4, f c = 0 :=
by
  intros h
  have f_2_pos : f 2 > 0 := h.1
  have f_4_neg : f 4 < 0 := h.2
  -- Use the intermediate value theorem
  sorry

end zero_in_interval_l351_351487


namespace fraction_identity_l351_351522

theorem fraction_identity (x y z v : ℝ) (hy : y ≠ 0) (hv : v ≠ 0)
    (h : x / y + z / v = 1) : x / y - z / v = (x / y) ^ 2 - (z / v) ^ 2 := by
  sorry

end fraction_identity_l351_351522


namespace range_values_of_sum_l351_351489

def f (x : ℝ) : ℝ := abs (log x / log 10)

theorem range_values_of_sum {a b : ℝ} (h₁ : a ≠ b) (h₂ : f a = f b) :
  a + b > 2 :=
sorry

end range_values_of_sum_l351_351489


namespace parents_rated_needs_improvement_l351_351147

@[inline] def percentage(p: ℕ) (total: ℕ) : ℕ := (total * p) / 100

theorem parents_rated_needs_improvement :
  (respondents excellent very_satisfactory satisfactory needs_improvement : ℕ) 
  (h_total : respondents = 120)
  (h_excellent : percentage 15 respondents = excellent)
  (h_very_satisfactory : percentage 60 respondents = very_satisfactory)
  (h_rem : respondents - excellent - very_satisfactory = satisfactory + needs_improvement)
  (h_satisfactory : percentage 80 (satisfactory + needs_improvement) = satisfactory) :
  needs_improvement = 6 :=
by
  sorry

end parents_rated_needs_improvement_l351_351147


namespace three_digit_numbers_count_l351_351499

theorem three_digit_numbers_count :
  let digits := [1, 2, 2, 3, 4, 4, 4]
  let unique_digits := (digits.eraseDuplicates.length)
  let count_2 := (digits.filter (λ x => x = 2)).length
  let count_4 := (digits.filter (λ x => x = 4)).length
  -- All three different digits
  let case1 := unique_digits.choose 3 * 6 -- permutations
  -- Two digits the same, one different
  let case2_2 := (unique_digits - 1).choose 1 * 3  -- using digit '2' twice
  let case2_4 := (unique_digits - 1).choose 1 * 3  -- using digit '4' twice
  -- All three digits the same
  let case3 := if count_4 >= 3 then 1 else 0
  -- Total cases
  case1 + case2_2 + case2_4 + case3 = 43 := 
by
  sorry

end three_digit_numbers_count_l351_351499


namespace power_sum_roots_l351_351662

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l351_351662


namespace blocks_with_one_face_painted_l351_351501

def larger_cube_side : ℝ := 10
def small_block_side : ℝ := 2
def number_of_faces : ℕ := 6
def number_of_edges_per_face : ℕ := 4
def number_of_blocks_per_edge := (larger_cube_side / small_block_side).to_nat
def number_of_blocks_per_face := number_of_blocks_per_edge * number_of_blocks_per_edge
def total_surface_blocks := number_of_faces * number_of_blocks_per_face
def corner_blocks := 8
def edge_blocks_per_face : ℕ := (number_of_blocks_per_edge * number_of_edges_per_face) - 4
def total_edge_blocks := edge_blocks_per_face * number_of_faces

theorem blocks_with_one_face_painted :
  (total_surface_blocks - total_edge_blocks) = 54 :=
by
  sorry

end blocks_with_one_face_painted_l351_351501


namespace prime_distinct_integers_l351_351009

theorem prime_distinct_integers (p : ℕ) (hp : Nat.Prime p) 
  (a : ℕ → ℕ) (ha : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (n : ℕ) (hn : n = p + 1) : 
  ∃ i j : ℕ, 1 ≤ i → i < j → j ≤ n → (Nat.lcm (a i) (a j)) / Nat.gcd (a i) (a j) ≥ p + 1 :=
begin
  sorry
end

end prime_distinct_integers_l351_351009


namespace cos_330_eq_sqrt3_div_2_l351_351338

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351338


namespace sum_of_coordinates_A_l351_351607

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351607


namespace enrique_commission_l351_351407

theorem enrique_commission :
  let commission_rate : ℚ := 0.15
  let suits_sold : ℚ := 2
  let suits_price : ℚ := 700
  let shirts_sold : ℚ := 6
  let shirts_price : ℚ := 50
  let loafers_sold : ℚ := 2
  let loafers_price : ℚ := 150
  let total_sales := suits_sold * suits_price + shirts_sold * shirts_price + loafers_sold * loafers_price
  let commission := commission_rate * total_sales
  commission = 300 := by
begin
  sorry
end

end enrique_commission_l351_351407


namespace A_add_B_l351_351644

def four_digit_numbers := {n : ℕ // 1000 ≤ n ∧ n < 10000}

def odd_and_divisible_by_3 (n : ℕ) : Prop :=
(n % 2 = 1) ∧ (n % 3 = 0)

def divisible_by_4 (n : ℕ) : Prop :=
n % 4 = 0

def A : ℕ :=
set.card {n ∈ four_digit_numbers | odd_and_divisible_by_3 n}

def B : ℕ :=
set.card {n ∈ four_digit_numbers | divisible_by_4 n}

theorem A_add_B : A + B = 3750 := by
  sorry

end A_add_B_l351_351644


namespace radius_of_circle_is_sqrt_two_l351_351779

-- Definitions based on the conditions given in the problem
def rho : ℝ := 2
def phi : ℝ := Real.pi / 4

-- Lean statement of the proof problem
theorem radius_of_circle_is_sqrt_two (theta : ℝ) :
  let x := rho * Real.sin phi * Real.cos theta
  let y := rho * Real.sin phi * Real.sin theta
  sqrt (x^2 + y^2) = sqrt 2 :=
sorry

end radius_of_circle_is_sqrt_two_l351_351779


namespace range_of_f_l351_351485

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 ^ x else abs (Real.log x / Real.log 2)

theorem range_of_f (a : ℝ) :
  (f a < 1 / 2) ↔ (a ∈ Set.Iio (-1) ∪ Set.Ioo (Real.sqrt 2⁻¹⁄²) (Real.sqrt 2)) :=
by
  sorry

end range_of_f_l351_351485


namespace cos_330_eq_sqrt3_div_2_l351_351326

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351326


namespace max_abs_x_sub_2y_add_1_l351_351440

theorem max_abs_x_sub_2y_add_1 (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : |x - 2y + 1| ≤ 5 :=
sorry

end max_abs_x_sub_2y_add_1_l351_351440


namespace cream_ratio_l351_351636

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l351_351636


namespace distinct_paths_to_B_and_C_l351_351876

def paths_to_red_arrows : ℕ × ℕ := (1, 2)
def paths_from_first_red_to_blue : ℕ := 3 * 2
def paths_from_second_red_to_blue : ℕ := 4 * 2
def total_paths_to_blue_arrows : ℕ := paths_from_first_red_to_blue + paths_from_second_red_to_blue

def paths_from_first_two_blue_to_green : ℕ := 5 * 4
def paths_from_third_and_fourth_blue_to_green : ℕ := 6 * 4
def total_paths_to_green_arrows : ℕ := paths_from_first_two_blue_to_green + paths_from_third_and_fourth_blue_to_green

def paths_to_B : ℕ := total_paths_to_green_arrows * 3
def paths_to_C : ℕ := total_paths_to_green_arrows * 4
def total_paths : ℕ := paths_to_B + paths_to_C

theorem distinct_paths_to_B_and_C :
  total_paths = 4312 := 
by
  -- all conditions can be used within this proof
  sorry

end distinct_paths_to_B_and_C_l351_351876


namespace not_unique_y_20_paise_l351_351134

theorem not_unique_y_20_paise (x y z w : ℕ) : 
  x + y + z + w = 750 → 10 * x + 20 * y + 50 * z + 100 * w = 27500 → ∃ (y₁ y₂ : ℕ), y₁ ≠ y₂ :=
by 
  intro h1 h2
  -- Without additional constraints on x, y, z, w,
  -- suppose that there are at least two different solutions satisfying both equations,
  -- demonstrating the non-uniqueness of y.
  sorry

end not_unique_y_20_paise_l351_351134


namespace geometry_problem_l351_351869

-- Define the points A, B, C, F, G
variables {A B C F G X : Type}

-- Definition of intersection condition:
def intersection_point (A B F G : Type) : Prop :=
  ∃ (l : Type), line_through F l ∧ line_parallel_to BC l ∧ G ∈ l ∧ G ∈ AB

-- Theorem statement representing the hypothesis and what we need to prove:
theorem geometry_problem (h : intersection_point A B F G) : 
  length B G = length A X :=
begin
  sorry, -- proof to be inserted
end

end geometry_problem_l351_351869


namespace prove_equal_distance_l351_351447

axiom circumcircle (O : Type) (A B C : O) (ABC_eq : ∀ {A' B' C'}, A' = B' ∧ B' = C' ∧ C' = A' := by sorry) : O
axiom diameter (D : Type) (O : Type) {A : O} : Type := by sorry
axiom incenter (E F : O) (P A B C : O) : (∀ (T : Type) (T_eq_PAB T_eq_PAC : Type), E = incenter T_eq_PAB ∧ F = incenter T_eq_PAC) := by sorry
axiom arc_not_B_not_C (P B C : O) : (P ≠ B) ∧ (P ≠ C) := by sorry

def distance (x y : Type) : Type := by sorry

theorem prove_equal_distance (O : Type) (A B C D P E F : O) (h1 : circumcircle O A B C) (h2 : diameter D O A) (h3 : incenter E F P A B C) (h4 : arc_not_B_not_C P B C) : (distance P D) = abs (distance P E - distance P F) := by sorry

end prove_equal_distance_l351_351447


namespace cupcakes_eaten_right_away_l351_351702

theorem cupcakes_eaten_right_away : 
  let cost_per_cupcake := 0.75
  let price_per_cupcake := 2.00
  let burnt_cupcakes := 2 * 12
  let perfect_cupcakes := 2 * 12
  let later_cupcakes := 2 * 12
  let total_cost := (burnt_cupcakes + perfect_cupcakes + later_cupcakes) * cost_per_cupcake
  let net_profit := 24
  let total_revenue := net_profit + total_cost
  let cupcakes_sold := total_revenue / price_per_cupcake
  let total_made := burnt_cupcakes + perfect_cupcakes + later_cupcakes
  let cupcakes_left_after_burn := total_made - burnt_cupcakes
  let cupcakes_left_after_eating_later := cupcakes_left_after_burn - 4
  in cupcakes_left_after_eating_later - cupcakes_sold = 5 :=
by 
  -- Proof omitted
  sorry

end cupcakes_eaten_right_away_l351_351702


namespace mean_of_eight_numbers_l351_351074

theorem mean_of_eight_numbers (sum_of_numbers : ℚ) (h : sum_of_numbers = 3/4) : 
  sum_of_numbers / 8 = 3/32 := by
  sorry

end mean_of_eight_numbers_l351_351074


namespace cos_330_eq_sqrt3_div_2_l351_351346

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351346


namespace find_xy_l351_351980

theorem find_xy:
  ∃ x y : ℤ, (√5 (119287 - 48682 * real.sqrt 6) = x + y * real.sqrt 6) ∧ (x = 7) ∧ (y = -2) :=
sorry

end find_xy_l351_351980


namespace find_A_coordinates_sum_l351_351581

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351581


namespace hexagon_triangle_area_ratio_l351_351127

theorem hexagon_triangle_area_ratio :
  ∀ (side_length : ℝ), (side_length > 0) →
  let center_to_vertex := side_length
  let center_to_midpoint := side_length * (Real.sqrt 3 / 2)
  let area_larger_triangle := (1 / 2) * center_to_vertex * center_to_midpoint
  let area_smaller_triangle := (1 / 2) * center_to_midpoint * (center_to_midpoint / 2)
  ratio (area_larger_triangle / area_smaller_triangle) = (4 * Real.sqrt 3) / 3 :=
by 
  intros side_length side_length_pos
  have center_to_vertex : ℝ := side_length
  have center_to_midpoint : ℝ := side_length * (Real.sqrt 3 / 2)
  have area_larger_triangle : ℝ := (1 / 2) * center_to_vertex * center_to_midpoint
  have area_smaller_triangle : ℝ := (1 / 2) * center_to_midpoint * (center_to_midpoint / 2)
  show area_larger_triangle / area_smaller_triangle = (4 * Real.sqrt 3) / 3
  sorry

end hexagon_triangle_area_ratio_l351_351127


namespace cos_330_eq_sqrt3_div_2_l351_351344

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351344


namespace train_stations_distance_l351_351802

/-- Given:
1. Trains A and B start simultaneously.
2. Trains A and B travel on adjacent parallel tracks towards each other.
3. Both trains travel at a constant speed of 50 miles per hour.
4. Train A has traveled 225 miles when the trains pass each other.
  
Prove: The total distance between the stations where trains A and B started is 450 miles. -/
theorem train_stations_distance :
  ∀ (start_time : ℝ) (distance_A distance_B : ℝ) (speed : ℝ),
  start_time ≥ 0 →
  speed = 50 →
  distance_A = 225 →
  distance_B = 225 →
  (distance_A + distance_B) = 450 :=
by
  intros start_time distance_A distance_B speed h₁ h₂ h₃ h₄
  rw [h₃, h₄]
  exact rfl

end train_stations_distance_l351_351802


namespace probability_parabola_l351_351106

-- Define the range of values for x and y
def values := {1, 2, 3, 4, 5, 6}

-- Define the parabolic condition
def on_parabola (x y : ℕ) : Prop :=
  y = - x^2 + 4 * x

-- The main statement of the problem
theorem probability_parabola :
  ((λ (x y : ℕ), (x, y)) '' (values ×ˢ values)).count on_parabola = 1 / 12 := 
by
  sorry

end probability_parabola_l351_351106


namespace ellipse_foci_distance_l351_351913

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351913


namespace cos_330_eq_sqrt3_div_2_l351_351252

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351252


namespace keychain_arrangement_l351_351547

/-- 
Prove that the number of distinct ways to arrange six keys on a keychain,
given the house key is adjacent to the car key and the office key is adjacent
to the mailbox key, and considering rotations and reflections as equivalent,
is 24. 
-/
theorem keychain_arrangement : 
  let keys := ["House", "Car", "Office", "Mailbox", "Key1", "Key2"] in
  let grouped_units := [["House", "Car"], ["Office", "Mailbox"], "Key1", "Key2"] in
  let distinct_arrangements := (factorial 3) * 2 * 2 in
  distinct_arrangements = 24 :=
by
  sorry  -- Proof here

end keychain_arrangement_l351_351547


namespace distance_between_foci_of_given_ellipse_l351_351907

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351907


namespace cos_330_eq_sqrt3_over_2_l351_351200

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351200


namespace product_mod_7_l351_351741

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351741


namespace books_remainder_l351_351541

theorem books_remainder (total_books new_books_per_section sections : ℕ) 
  (h1 : total_books = 1521) 
  (h2 : new_books_per_section = 45) 
  (h3 : sections = 41) : 
  (total_books * sections) % new_books_per_section = 36 :=
by
  sorry

end books_remainder_l351_351541


namespace two_trains_clearing_time_l351_351089

noncomputable def length_train1 : ℝ := 100  -- Length of Train 1 in meters
noncomputable def length_train2 : ℝ := 160  -- Length of Train 2 in meters
noncomputable def speed_train1 : ℝ := 42 * 1000 / 3600  -- Speed of Train 1 in m/s
noncomputable def speed_train2 : ℝ := 30 * 1000 / 3600  -- Speed of Train 2 in m/s
noncomputable def total_distance : ℝ := length_train1 + length_train2  -- Total distance to be covered
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2  -- Relative speed

theorem two_trains_clearing_time : total_distance / relative_speed = 13 := by
  sorry

end two_trains_clearing_time_l351_351089


namespace simplify_rationalize_expr_l351_351711

theorem simplify_rationalize_expr : 
  (1 / (2 + 1 / (Real.sqrt 5 - 2))) = (4 - Real.sqrt 5) / 11 := 
by 
  sorry

end simplify_rationalize_expr_l351_351711


namespace cos_330_eq_sqrt_3_div_2_l351_351223

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351223


namespace sibling_pizza_order_l351_351436

theorem sibling_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 3 / 10
  let Ella := 1 - Alex - Beth - Cyril - Dan
  let siblings := [Beth, Cyril, Dan, Alex, Ella]
  siblings = [2 / 5, 1 / 3, 3 / 10, 1 / 6, (1 - 1 / 6 - 2 / 5 - 1 / 3 - 3 / 10)] →
  siblings.sorted (λ a b => a > b) = [2 / 5, 1 / 3, 3 / 10, 1 / 6, (1 - 1 / 6 - 2 / 5 - 1 / 3 - 3 / 10)]  := 
by 
  sorry

end sibling_pizza_order_l351_351436


namespace gcd_75_100_l351_351838

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351838


namespace area_of_square_l351_351796

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end area_of_square_l351_351796


namespace cos_330_eq_sqrt_3_div_2_l351_351374

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351374


namespace vlecks_in_straight_angle_l351_351019

theorem vlecks_in_straight_angle (V : Type) [LinearOrderedField V] (full_circle_vlecks : V) (h1 : full_circle_vlecks = 600) :
  (full_circle_vlecks / 2) = 300 :=
by
  sorry

end vlecks_in_straight_angle_l351_351019


namespace Marley_fruits_total_is_31_l351_351674

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l351_351674


namespace product_mod_7_l351_351739

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351739


namespace no_consecutive_even_fibonacci_l351_351035

-- Fibonacci sequence definition
def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

-- Statement to prove: No two consecutive even Fibonacci numbers exist
theorem no_consecutive_even_fibonacci : ∀ n : ℕ, ¬ (even (fibonacci n) ∧ even (fibonacci (n + 1))) :=
by
  sorry

end no_consecutive_even_fibonacci_l351_351035


namespace arrangement_count_l351_351939

/-- The number of different arrangements of 3 male students and 4 female students
in a row where two specific individuals A and B must stand at the two ends is 120 -/
theorem arrangement_count (n_male n_female n_ends : ℕ) (A B : Type) :
  n_male = 3 → n_female = 4 → n_ends = 2 →
  (n_ends * nat.factorial (n_male + n_female - n_ends)) = 120 :=
by
  intros h_male h_female h_ends
  rw [h_male, h_female, h_ends]
  -- The proof would go here
  sorry

end arrangement_count_l351_351939


namespace calculate_expression_l351_351962

/-- Define the operation -/
def star (a b : ℝ) : ℝ := a + 2 / b

/-- Stating the problem as a Lean theorem -/
theorem calculate_expression : 
  (star (star 3 4) 5) - (star 3 (star 4 5)) = 49 / 110 := 
sorry

end calculate_expression_l351_351962


namespace part1_part2_l351_351490

noncomputable def f (x : ℝ) : ℝ := cos (x + π / 12) ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 + 1 / 2 * sin (2 * x)
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem part1 (k : ℤ) (x0 : ℝ) (h_eq : x0 = (k * π - π / 6) / 2) :
  (k % 2 = 0 → g x0 = 3 / 4) ∧ (k % 2 = 1 → g x0 = 5 / 4) := sorry

theorem part2 (k : ℤ) (x : ℝ) :
  (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) → 0 < deriv h x := sorry

end part1_part2_l351_351490


namespace cos_330_eq_sqrt_3_div_2_l351_351213

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351213


namespace sums_of_coordinates_of_A_l351_351612

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351612


namespace sum_of_coordinates_of_A_l351_351590

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351590


namespace cos_330_eq_sqrt3_div_2_l351_351319

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351319


namespace gcd_75_100_l351_351811

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351811


namespace estimated_total_fish_population_l351_351082

-- Definitions of the initial conditions
def tagged_fish_in_first_catch : ℕ := 100
def total_fish_in_second_catch : ℕ := 300
def tagged_fish_in_second_catch : ℕ := 15

-- The theorem to prove the estimated number of total fish in the pond
theorem estimated_total_fish_population (tagged_fish_in_first_catch : ℕ) (total_fish_in_second_catch : ℕ) (tagged_fish_in_second_catch : ℕ) : ℕ :=
  2000

-- Assertion of the theorem with actual numbers
example : estimated_total_fish_population tagged_fish_in_first_catch total_fish_in_second_catch tagged_fish_in_second_catch = 2000 := by
  sorry

end estimated_total_fish_population_l351_351082


namespace quadrilateral_angle_contradiction_l351_351105

theorem quadrilateral_angle_contradiction (a b c d : ℝ)
  (h : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)
  (sum_eq_360 : a + b + c + d = 360) :
  (¬ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) → (90 < a ∧ 90 < b ∧ 90 < c ∧ 90 < d) :=
sorry

end quadrilateral_angle_contradiction_l351_351105


namespace find_A_coordinates_sum_l351_351579

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351579


namespace lines_parallel_l351_351473

variables {α β γ : Type}
variables [Plane α] [Plane β] [Plane γ]
variables (a b : Line)
variables h1 : Parallel α β
variables h2 : α ∩ γ = a
variables h3 : β ∩ γ = b

theorem lines_parallel : Parallel a b :=
sorry

end lines_parallel_l351_351473


namespace symmetric_function_x_axis_l351_351729

theorem symmetric_function_x_axis (x : ℝ) :
  let f := λ x : ℝ, x^2 - 3 * x,
      g := λ x : ℝ, - (x^2 - 3 * x) 
  in g x = -x^2 + 3 :=
by
  -- Original function definition
  let f := λ x : ℝ, x^2 - 3 * x
  -- Symmetry about the x-axis
  let g := λ x : ℝ, - f x
  -- Prove the transformed function equals the expected result
  show g x = -x^2 + 3
  sorry

end symmetric_function_x_axis_l351_351729


namespace total_packs_equiv_117_l351_351687

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end total_packs_equiv_117_l351_351687


namespace sum_of_lines_is_19_l351_351053

-- Define the problem in Lean 4
theorem sum_of_lines_is_19 :
  let numbers := List.range' 1 14 in
  let total_sum := List.sum numbers in
  let repeated_sum := (total_sum + List.sum (List.range' 1 8) - 1) / 2 in
  let total := total_sum + repeated_sum in
  (total / 7) = 19 :=
by
  sorry

end sum_of_lines_is_19_l351_351053


namespace rectangle_perimeter_l351_351061

theorem rectangle_perimeter (L W : ℕ) (hL : L = 6) (hW : W = 4) : 2 * (L + W) = 20 := by
  rw [hL, hW]
  norm_num
  sorry

end rectangle_perimeter_l351_351061


namespace remainder_product_l351_351750

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351750


namespace swimmers_meet_l351_351087

theorem swimmers_meet (pool_length : ℕ) (speed1 : ℕ) (speed2 : ℕ) (time_minutes : ℕ) 
    (no_loss_turns: true) (pool_length = 120) (speed1 = 4) (speed2 = 3) (time_minutes = 20) : 
    ∃ n, n = 30 ∧ (number_of_meetings pool_length speed1 speed2 time_minutes = n) :=
sorry

end swimmers_meet_l351_351087


namespace sum_coordinates_A_l351_351594

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351594


namespace product_remainder_mod_7_l351_351764

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l351_351764


namespace questions_ratio_l351_351027

theorem questions_ratio (R A : ℕ) (H₁ : R + 6 + A = 24) :
  (R, 6, A) = (R, 6, A) :=
sorry

end questions_ratio_l351_351027


namespace questionnaire_C_count_l351_351149

theorem questionnaire_C_count : 
  let a_n := λ n : ℕ, 30 * n - 26
  in (finset.range 33).filter (λ n, 721 ≤ a_n n ∧ a_n n ≤ 960 ∧ 1 ≤ n).card = 8 := 
by
  sorry

end questionnaire_C_count_l351_351149


namespace enrique_commission_l351_351404

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l351_351404


namespace calculate_expression_l351_351175

theorem calculate_expression : (64 / 27 : ℝ) ^ (1 / 3) + real.logb 3 (10 / 9) + real.logb 3 (9 / 10) = 4 / 3 :=
by 
  -- Proof omitted
  sorry

end calculate_expression_l351_351175


namespace distance_between_ellipse_foci_l351_351918

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351918


namespace distance_point_to_line_l351_351623

noncomputable def point := (1, 0) : ℝ × ℝ
noncomputable def line := λ (x y : ℝ), x - y

theorem distance_point_to_line :
  let d := λ (x1 y1 : ℝ) (f : ℝ → ℝ → ℝ), |f x1 y1| / real.sqrt (f 1 1 ^ 2 + f 0 1 ^ 2)
  in d 1 0 line = real.sqrt 2 / 2 :=
by
  sorry

end distance_point_to_line_l351_351623


namespace cos_330_eq_sqrt3_div_2_l351_351269

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351269


namespace sums_of_coordinates_of_A_l351_351611

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l351_351611


namespace cos_330_eq_sqrt_3_div_2_l351_351370

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351370


namespace cos_330_eq_sqrt3_div_2_l351_351323

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351323


namespace magnitude_of_w_l351_351653

theorem magnitude_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + (3 / w) = s) : |w| = 1.5 := by
  sorry

end magnitude_of_w_l351_351653


namespace cos_330_eq_sqrt3_div_2_l351_351251

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351251


namespace cos_330_eq_sqrt3_div_2_l351_351304

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351304


namespace cos_330_is_sqrt3_over_2_l351_351301

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351301


namespace fraction_of_students_with_buddies_l351_351543

theorem fraction_of_students_with_buddies
  (s n : ℕ)
  (h : n / 4 = s / 2) :
  (n / 4 + s / 2) / (n + s) = 1 / 3 :=
by
  -- from the condition n / 4 = s / 2, we can derive n = 2s
  have : n = 2 * s := by
    calc
      n = 2 * s := sorry,
  -- substituting n = 2s in the required equation
  calc
    (n / 4 + s / 2) / (n + s)
        = ((2 * s) / 4 + s / 2) / ((2 * s) + s) := by rw this
    ... = (s / 2 + s / 2) / (3 * s)             := by norm_num
    ... = (s / s) * (1 / 3)                     := by rw [←div_add_div, div_self, div_eq_mul_inv]
    ... = 1 / 3                                 := by norm_num
  sorry

end fraction_of_students_with_buddies_l351_351543


namespace radius_of_circle_spherical_coords_l351_351785

theorem radius_of_circle_spherical_coords :
  ∀ (θ : ℝ), ∃ r : ℝ, r = √2 ∧ ∀ (ρ : ℝ) (ϕ : ℝ), ρ = 2 → ϕ = π / 4 →
    (let x := ρ * real.sin ϕ * real.cos θ in
     let y := ρ * real.sin ϕ * real.sin θ in
     x^2 + y^2 = r^2) :=
by
  sorry

end radius_of_circle_spherical_coords_l351_351785


namespace min_value_of_quartic_function_l351_351422

theorem min_value_of_quartic_function : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → x^4 + (1 - x)^4 ≤ y^4 + (1 - y)^4) ∧ (x^4 + (1 - x)^4 = 1 / 8) :=
by
  sorry

end min_value_of_quartic_function_l351_351422


namespace simplify_and_evaluate_l351_351039

variable (x : ℝ)

-- Define the expressions and conditions
def expr1 : ℝ := (x^2 - 1) / (x^2 - 2 * x + 1)
def expr2 : ℝ := 1 / (x - 1)
def expr3 : ℝ := (x + 2) / (x - 1)
def condition : ℝ := real.sqrt 27 + abs (-2) - 3 * real.tan (real.pi / 3)

theorem simplify_and_evaluate (h : x = condition) : 
  ((expr1 - expr2) / expr3) = 1 / 2 :=
by 
  sorry

end simplify_and_evaluate_l351_351039


namespace cos_330_eq_sqrt3_div_2_l351_351356

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351356


namespace num_divisors_with_divisor_count_1806_l351_351387

/-
Define the prime factorization of 1806 and its power.
-/
def prime_factors_1806 : List (ℕ × ℕ) := [(2, 1), (3, 2), (101, 1)]

/-
Define the prime factorization of 1806^1806.
-/
def prime_factors_1806_pow : List (ℕ × ℕ) := [(2, 1806), (3, 3612), (101, 1806)]

/-
Define a positive divisor of 1806^1806.
-/
def is_divisor (d : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    d = 2^a * 3^b * 101^c ∧ (a + 1) * (b + 1) * (c + 1) = 1806

/-
Define the problem statement.
-/
theorem num_divisors_with_divisor_count_1806 :
  (Finset.univ.filter is_divisor).card = 36 :=
begin
  sorry
end

end num_divisors_with_divisor_count_1806_l351_351387


namespace distance_CD_l351_351073

-- Conditions
variable (width_small : ℝ) 
variable (length_small : ℝ := 2 * width_small) 
variable (perimeter_small : ℝ := 2 * (width_small + length_small))
variable (width_large : ℝ := 3 * width_small)
variable (length_large : ℝ := 2 * length_small)
variable (area_large : ℝ := width_large * length_large)

-- Condition assertions
axiom smaller_rectangle_perimeter : perimeter_small = 6
axiom larger_rectangle_area : area_large = 12

-- Calculating distance hypothesis
theorem distance_CD (CD_x CD_y : ℝ) (width_small length_small width_large length_large : ℝ) 
  (smaller_rectangle_perimeter : 2 * (width_small + length_small) = 6)
  (larger_rectangle_area : (3 * width_small) * (2 * length_small) = 12)
  (CD_x_def : CD_x = 2 * length_small)
  (CD_y_def : CD_y = 2 * width_large - width_small)
  : Real.sqrt ((CD_x) ^ 2 + (CD_y) ^ 2) = Real.sqrt 45 := 
sorry

end distance_CD_l351_351073


namespace sum_of_coords_A_l351_351576

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351576


namespace cos_330_eq_sqrt_3_div_2_l351_351369

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351369


namespace parking_lot_vehicle_spaces_l351_351794

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

end parking_lot_vehicle_spaces_l351_351794


namespace cos_330_eq_sqrt3_div_2_l351_351359

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351359


namespace cos_330_eq_sqrt3_div_2_l351_351343

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351343


namespace gcd_75_100_l351_351840

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351840


namespace train_speed_l351_351860

-- Define the conditions given in the problem
def train_length : ℝ := 160
def time_to_cross_man : ℝ := 4

-- Define the statement to be proved
theorem train_speed (H1 : train_length = 160) (H2 : time_to_cross_man = 4) : train_length / time_to_cross_man = 40 :=
by
  sorry

end train_speed_l351_351860


namespace evaluate_expression_l351_351413

-- Define the integers a and b
def a := 2019
def b := 2020

-- The main theorem stating the equivalence
theorem evaluate_expression :
  (a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3 + 6) / (a * b) = 5 / (a * b) := 
by
  sorry

end evaluate_expression_l351_351413


namespace find_x_l351_351458

-- Define the conditions
def A_on_plane_alpha := True
def P_not_on_plane_alpha := True
def vec_PA := (- (Real.sqrt 3 / 2), 1 / 2, x : Real)
def vec_PA_magnitude := Real.sqrt 3
def normal_vector := (0, - 1 / 2, - (Real.sqrt 2))

-- Define the proof problem
theorem find_x (x : Real) (h1 : A_on_plane_alpha) (h2 : P_not_on_plane_alpha) (h3 : vec_PA = (- (Real.sqrt 3 / 2), 1 / 2, x)) (h4 : vec_PA_magnitude = Real.sqrt 3) (h5 : normal_vector = (0, - 1 / 2, - (Real.sqrt 2))) : 
  x = Real.sqrt 2 := by
  sorry

end find_x_l351_351458


namespace remainder_3x3_minus_4x2_plus_17x_plus_34_div_x_minus_7_l351_351848

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 17 * x + 34

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := x - 7

-- State the theorem that the remainder when P(x) is divided by D(x) is 986
theorem remainder_3x3_minus_4x2_plus_17x_plus_34_div_x_minus_7:
  let r := P(7) in -- Since r = P(c) when dividing by x - c
  r = 986 :=
by
  -- Await proof
  sorry

end remainder_3x3_minus_4x2_plus_17x_plus_34_div_x_minus_7_l351_351848


namespace cos_330_eq_sqrt3_div_2_l351_351278

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351278


namespace cone_height_l351_351879

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

theorem cone_height (h r : ℝ) (H1: volume_of_cone r h = 9720 * real.pi) (H2: h = r) :
  h = 30.8 :=
by
  sorry

end cone_height_l351_351879


namespace gcd_75_100_l351_351834

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351834


namespace max_A_min_A_l351_351136

-- Define the problem and its conditions and question

def A_max (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

def A_min (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

theorem max_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_max B h1 h2 h3 = 999999998 := sorry

theorem min_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_min B h1 h2 h3 = 122222224 := sorry

end max_A_min_A_l351_351136


namespace cos_330_eq_sqrt3_div_2_l351_351266

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351266


namespace product_mod_7_l351_351744

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351744


namespace gcd_75_100_l351_351843

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351843


namespace smallest_next_divisor_l351_351080

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor (m : ℕ) (h_even : is_even m)
  (h_four_digit : is_four_digit m)
  (h_div_437 : is_divisor 437 m) :
  ∃ next_div : ℕ, next_div > 437 ∧ is_divisor next_div m ∧ 
  ∀ d, d > 437 ∧ is_divisor d m → next_div ≤ d :=
sorry

end smallest_next_divisor_l351_351080


namespace total_pencils_is_54_l351_351682

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l351_351682


namespace QC_bisects_AQB_l351_351801

open Set

variable (P A B C Q O : Point)
variable (circle : Circle)
variable [RealCoordinateSpace]

-- Conditions:
axiom P_exterior : ¬P ∈ circle
axiom secant_intersects : Segment AP ∩ circle = {A, B}
axiom tangent_touch : tangent_to_circle C = circle ∩ line_through C P
axiom projection_Q : Q = projection C (diameter_through P)

-- Proof statement:
theorem QC_bisects_AQB (hQCd : distance_squared Q C = 0)
  (hprojection : Q ∈ diameter_through P)
  (htangent : is_tangent_to C circle)
  (hA B : line_through P ∩ circle = {A, B})
  (hP_ext : P ∉ circle)
  (hOC : O ∈ circle.center)
  (same_side : same_side_of_diameter P A B C) :
  bisects (angle_between Q C) (angle_between A Q B) :=
by
  sorry

end QC_bisects_AQB_l351_351801


namespace ratio_of_areas_triangle_to_rectangle_l351_351550

variables (A B C D E F : Type)
variables [ordered_ring A] [ordered_ring B] [ordered_ring C] [ordered_ring D] [ordered_ring E] [ordered_ring F]
variables (AB BC CD DA EF : ℝ)

-- Condition 1: Rectangle ABCD with DC = 3 * CB
def rectangle_ABCD (AB BC CD DA : ℝ) : Prop :=
  CD = 3 * BC ∧ DA * BC = CD * AB ∧ AB = DA -- AB || DC, AD || BC

-- Condition 2: Points E and F lie on AB
def points_on_AB (E F AB : ℝ) : Prop :=
  E < AB ∧ F < AB ∧ E ≠ F

-- Condition 3: ED and FD trisect ∠ADC
def trisect_angle_ADC (ED FD : ℝ) : Prop :=
  true -- Placeholder as this condition is inherently geometric and complex to represent

-- Main statement to prove
theorem ratio_of_areas_triangle_to_rectangle
  (h1 : rectangle_ABCD AB BC CD DA)
  (h2 : points_on_AB E F AB)
  (h3 : trisect_angle_ADC E F) :
  (area_of_triangle DEF / area_of_rectangle ABCD) = (real.sqrt 3 / 18) :=
sorry

end ratio_of_areas_triangle_to_rectangle_l351_351550


namespace cos_arithmetic_sequence_l351_351459

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_arith : ∀ n m, a (n + m) = a n + m * (a 2 - a 1)) (h_sum : a 1 + a 8 + a 15 = real.pi) :
  real.cos (a 4 + a 12) = -1 / 2 :=
sorry

end cos_arithmetic_sequence_l351_351459


namespace cos_330_eq_sqrt3_over_2_l351_351201

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351201


namespace cos_330_eq_sqrt3_div_2_l351_351287

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351287


namespace cos_330_eq_sqrt3_div_2_l351_351308

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351308


namespace parents_rating_needs_improvement_l351_351145

-- Define the conditions as variables and expressions
def total_parents : ℕ := 120
def percentage_excellent : ℕ := 15
def percentage_very_satisfactory : ℕ := 60
def percentage_satisfactory_of_remaining : ℕ := 80

-- Define the calculation as a Lean theorem
theorem parents_rating_needs_improvement : 
  let num_excellent := (percentage_excellent * total_parents) / 100
  let num_very_satisfactory := (percentage_very_satisfactory * total_parents) / 100
  let remaining_parents := total_parents - num_excellent - num_very_satisfactory
  let num_satisfactory := (percentage_satisfactory_of_remaining * remaining_parents) / 100
  let num_needs_improvement := remaining_parents - num_satisfactory
  in num_needs_improvement = 6 :=
by 
  -- Defer the proof
  sorry

end parents_rating_needs_improvement_l351_351145


namespace enrique_commission_l351_351405

theorem enrique_commission :
  let commission_rate : ℚ := 0.15
  let suits_sold : ℚ := 2
  let suits_price : ℚ := 700
  let shirts_sold : ℚ := 6
  let shirts_price : ℚ := 50
  let loafers_sold : ℚ := 2
  let loafers_price : ℚ := 150
  let total_sales := suits_sold * suits_price + shirts_sold * shirts_price + loafers_sold * loafers_price
  let commission := commission_rate * total_sales
  commission = 300 := by
begin
  sorry
end

end enrique_commission_l351_351405


namespace height_of_equilateral_triangle_l351_351162

theorem height_of_equilateral_triangle (s : ℝ) (h : ℝ) 
  (eq_areas : (s ^ 2) = (sqrt 3 / 4) * (s ^ 2)) :
  h = (sqrt 3 / 2) * s := 
sorry

end height_of_equilateral_triangle_l351_351162


namespace petya_password_count_l351_351698

theorem petya_password_count : 
  let all_digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  let total_passwords := 9^4
  let choose_4 := Nat.choose 9 4
  let arrange_4 := factorial 4
  let different_digits := choose_4 * arrange_4
  let passwords_with_identical_digits := total_passwords - different_digits
  in passwords_with_identical_digits = 3537 := by
  -- Definitions
  let all_digits := {0, 1, 2, 3, 4, 5, 6, 8, 9}
  let total_passwords := 9^4
  let choose_4 := Nat.choose 9 4
  let arrange_4 := factorial 4
  let different_digits := choose_4 * arrange_4
  let passwords_with_identical_digits := total_passwords - different_digits
  -- Proof (to be filled in)
  sorry

end petya_password_count_l351_351698


namespace distance_between_foci_of_given_ellipse_l351_351903

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351903


namespace remainder_of_product_mod_7_l351_351758

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351758


namespace find_values_of_a_to_make_lines_skew_l351_351418

noncomputable def lines_are_skew (t u a : ℝ) : Prop :=
  ∀ t u,
    (1 + 2 * t = 4 + 5 * u ∧
     2 + 3 * t = 1 + 2 * u ∧
     a + 4 * t = u) → false

theorem find_values_of_a_to_make_lines_skew :
  ∀ a : ℝ, ¬ a = 3 ↔ lines_are_skew t u a :=
by
  sorry

end find_values_of_a_to_make_lines_skew_l351_351418


namespace cos_330_cos_30_val_answer_l351_351232

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l351_351232


namespace final_lamps_l351_351047

variable (L : Type) -- Type for lamps
variables (A B C D E F G : L) -- The individual lamps
variable (initial_state : L → Bool) -- Initial state of the lamps (True for on, False for off)
variable (toggle : L → L) -- Operation simulating the toggle

def initial_lamp_state : L → Bool
| A | C | E | G := true
| B | D | F := false

def final_state (n : Nat) (state : L → Bool) : L → Bool :=
  let toggle_n_times := (n / 7) + if n % 7 ≠ 0 then 1 else 0
  fun l => if toggle_n_times % 2 == 0 then state l else not (state l)

theorem final_lamps (h : initial_lamp_state A ∧ initial_lamp_state C ∧ initial_lamp_state E ∧ initial_lamp_state G ∧
                       ¬initial_lamp_state B ∧ ¬initial_lamp_state D ∧ ¬initial_lamp_state F) :
  final_state 1999 initial_lamp_state A ∧ 
  final_state 1999 initial_lamp_state C ∧ 
  ¬final_state 1999 initial_lamp_state E ∧ 
  ¬final_state 1999 initial_lamp_state G ∧ 
  ¬final_state 1999 initial_lamp_state B ∧ 
  ¬final_state 1999 initial_lamp_state D ∧ 
  final_state 1999 initial_lamp_state F :=
by
  sorry

end final_lamps_l351_351047


namespace gcd_75_100_l351_351812

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351812


namespace number_of_pencil_boxes_l351_351789

open Nat

def books_per_box : Nat := 46
def num_book_boxes : Nat := 19
def pencils_per_box : Nat := 170
def total_books_and_pencils : Nat := 1894

theorem number_of_pencil_boxes :
  (total_books_and_pencils - (num_book_boxes * books_per_box)) / pencils_per_box = 6 := 
by
  sorry

end number_of_pencil_boxes_l351_351789


namespace enrique_commission_l351_351399

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l351_351399


namespace cos_330_is_sqrt3_over_2_l351_351296

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351296


namespace cos_330_eq_sqrt3_div_2_l351_351349

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351349


namespace sum_of_coordinates_A_l351_351604

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351604


namespace sum_of_coords_A_l351_351569

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351569


namespace problem1_problem2_problem3_problem4_l351_351954

theorem problem1 : (1 * (-7)) - (-10) + (-8) - (+2) = -7 :=
by sorry

theorem problem2 : ((1 / 4) - (1 / 2) + (1 / 6)) * 12 = -1 :=
by sorry

theorem problem3 : (-3 * abs (-2)) + ((-28) / (-7)) = -2 :=
by sorry

theorem problem4 : (-3^2) - ((-2)^3 / 4) = -7 :=
by sorry

end problem1_problem2_problem3_problem4_l351_351954


namespace cos_330_eq_sqrt3_div_2_l351_351265

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351265


namespace triangle_area_l351_351153

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Define the property of being a right triangle via the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of a right triangle given base and height
def area_right_triangle (a b : ℕ) : ℕ := (a * b) / 2

-- The main theorem, stating that the area of the triangle with sides 9, 12, 15 is 54
theorem triangle_area : is_right_triangle a b c → area_right_triangle a b = 54 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l351_351153


namespace perimeter_of_specific_figure_l351_351886

-- Define the grid size and additional column properties as given in the problem
structure Figure :=
  (rows : ℕ)
  (cols : ℕ)
  (additionalCols : ℕ)
  (additionalRows : ℕ)

-- The specific figure properties from the problem statement
def specificFigure : Figure := {
  rows := 3,
  cols := 4,
  additionalCols := 1,
  additionalRows := 2
}

-- Define the perimeter computation
def computePerimeter (fig : Figure) : ℕ :=
  2 * (fig.rows + fig.cols + fig.additionalCols) + fig.additionalRows

theorem perimeter_of_specific_figure : computePerimeter specificFigure = 13 :=
by
  sorry

end perimeter_of_specific_figure_l351_351886


namespace total_pencils_correct_l351_351684

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l351_351684


namespace minimum_k_to_draw_red_ball_l351_351529

theorem minimum_k_to_draw_red_ball (red white black : ℕ)
    (h_red : red = 10) (h_white : white = 8) (h_black : black = 7) :
    ∃ k, (∀ draws : list ℕ, (length draws ≥ k) → (∃ n < k, draws.nth n = some 1)) ∧ k = 16 :=
by 
  sorry

end minimum_k_to_draw_red_ball_l351_351529


namespace maximize_take_home_pay_l351_351537

-- Define the tax system condition
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay condition
def take_home_pay (y : ℝ) : ℝ := 100 * y^2 - tax y

-- The theorem to prove the maximum take-home pay is achieved at a specific income level
theorem maximize_take_home_pay : 
  ∃ y : ℝ, take_home_pay y = 100 * 50^2 - 50^3 := sorry

end maximize_take_home_pay_l351_351537


namespace median_of_dataset_l351_351622

theorem median_of_dataset (x : ℕ) (h_mode_eq_mean : mode [9, 9, x, 7] = mean [9, 9, x, 7]) :
  median [9, 9, x, 7] = 9 := 
by 
  -- Since we need to provide the proof, we put a placeholder (sorry)
  sorry

end median_of_dataset_l351_351622


namespace monotonic_decreasing_interval_l351_351063

noncomputable def y (x : ℝ) : ℝ := (real.sqrt 3 - real.tan x) / (1 + real.sqrt 3 * real.tan x)

theorem monotonic_decreasing_interval :
  ∀ (k : ℤ), ∀ (x : ℝ), (k * real.pi - real.pi / 6 < x) ∧ (x < k * real.pi + 5 * real.pi / 6) → 
  ∀ y, y = (real.sqrt 3 - real.tan x) / (1 + real.sqrt 3 * real.tan x) → 
  ∃ x, x ∈ set.Ioo (k * real.pi - real.pi / 6) (k * real.pi + 5 * real.pi / 6) :=
begin
  sorry
end

end monotonic_decreasing_interval_l351_351063


namespace radius_of_circle_in_spherical_coords_l351_351782

theorem radius_of_circle_in_spherical_coords :
  ∀ θ : ℝ, (∃ ρ φ : ℝ, ρ = 2 ∧ φ = π/4) →
  ∃ r : ℝ, r = ∥(2 * sin (π/4) * cos θ, 2 * sin (π/4) * sin θ, 2 * cos (π/4)).1.1 ∥ :=
begin
  intro θ,
  intro h,
  use sqrt 2,
  have : ∥(sqrt 2 * cos θ, sqrt 2 * sin θ, sqrt 2).1.1∥ = sqrt 2,
  sorry -- Skip the actual proof
end

end radius_of_circle_in_spherical_coords_l351_351782


namespace cos_330_eq_sqrt3_div_2_l351_351187

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351187


namespace even_function_not_monotonic_l351_351472

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem even_function_not_monotonic (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) → ¬ MonotonicOn (f m) (Set.Ioo (-(∞ : ℝ)) 3) :=
by
  intros h_even
  have h0: m = 0, from sorry,
  rw h0,
  let f0 := λ x:ℝ, -x^2 + 3,
  have h_derivative: ∀ x: ℝ, deriv f0 x = -2 * x, from sorry,
  -- show that it is increasing on (-∞, 0) and decreasing on (0, ∞)
  have h_increasing: ∀ x < 0, deriv f0 x > 0, from sorry,
  have h_decreasing: ∀ x > 0, deriv f0 x < 0, from sorry,
  -- thus, it is not monotonic on (-∞,3)
  have h_non_monotonic: ¬ MonotonicOn f0 (Set.Ioo (-(∞ : ℝ)) 3), from sorry
  exact h_non_monotonic

end even_function_not_monotonic_l351_351472


namespace even_function_definition_l351_351056

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x * (1 + x) else if x < 0 then x * (x - 1) else 0

theorem even_function_definition : ∀ x : ℝ, f(x) = f(-x) :=
by
  intro x
  cases x
  case neg hx => sorry
  case pos hx => sorry
  case zero => sorry

end even_function_definition_l351_351056


namespace mean_of_solutions_l351_351990

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^3 + 5*x^2 - 2*x - 8

-- Define the solutions
noncomputable def solution_1 : ℝ := -1
noncomputable def solution_2 : ℝ := -2 + 2*real.sqrt 3
noncomputable def solution_3 : ℝ := -2 - 2*real.sqrt 3

-- Prove the required mean
theorem mean_of_solutions :
  poly(solution_1) = 0 ∧ poly(solution_2) = 0 ∧ poly(solution_3) = 0 →
  (solution_1 + solution_2 + solution_3) / 3 = -5/3 :=
by
  sorry

end mean_of_solutions_l351_351990


namespace general_formula_sum_of_squares_l351_351460

-- Define the sequence a_n with initial condition and recurrence relation
def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 1 / (2 * n : ℝ)

-- Initial condition a_1 = 1 / 2
def a1_initial_condition : Prop :=
  sequence 1 = 1 / 2

-- Recurrence relation 1 / a_(n+1) = 1 / a_n + 2
def recurrence_relation (n : ℕ) : Prop :=
  n > 0 → 1 / sequence (n + 1) = 1 / sequence n + 2

-- Prove that a_n = 1 / (2 * n)
theorem general_formula (n : ℕ) (h₁ : a1_initial_condition) (h₂ : ∀ n, recurrence_relation n) : 
  sequence n = 1 / (2 * n) := 
sorry

-- Prove that a_1^2 + a_2^2 + ... + a_n^2 < 1 / 2
theorem sum_of_squares (n : ℕ) (h₁ : a1_initial_condition) (h₂ : ∀ n, recurrence_relation n) : 
  ∑ k in range n, (sequence (k+1))^2 < 1 / 2 :=
sorry

end general_formula_sum_of_squares_l351_351460


namespace divisible_by_12_l351_351966

theorem divisible_by_12 (n : ℤ) : 12 ∣ (n^4 - n^2) := sorry

end divisible_by_12_l351_351966


namespace cos_330_eq_sqrt3_div_2_l351_351318

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351318


namespace remainder_of_product_l351_351775

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l351_351775


namespace min_value_of_reciprocal_sum_l351_351518

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 1) : 
  (\frac{1}{a} + \frac{1}{b}) = 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l351_351518


namespace cos_330_eq_sqrt3_div_2_l351_351191

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351191


namespace range_of_m_l351_351068

open Real

theorem range_of_m (a b m : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 9 / b = 1) :
  a + b ≥ -x^2 + 4 * x + 18 - m ↔ m ≥ 6 :=
by sorry

end range_of_m_l351_351068


namespace distance_between_foci_of_ellipse_l351_351929

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351929


namespace amusement_park_trip_distance_l351_351159

theorem amusement_park_trip_distance :
    let part1_distance := 40 * 1.5 in
    let part2_distance := 50 * 1 in
    let detour_distance := 10 in
    let part4_distance := 30 * 2.25 in
    part1_distance + part2_distance + detour_distance + part4_distance = 187.5 :=
by
    let part1_distance := 40 * 1.5
    let part2_distance := 50 * 1
    let detour_distance := 10
    let part4_distance := 30 * 2.25
    have : part1_distance + part2_distance + detour_distance + part4_distance = 187.5 := sorry
    exact this

end amusement_park_trip_distance_l351_351159


namespace distance_between_ellipse_foci_l351_351916

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351916


namespace sum_of_coordinates_A_l351_351565

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351565


namespace max_handshakes_30_men_l351_351871

theorem max_handshakes_30_men : ∃ k, k = 435 ∧ ∀ (n : ℕ), n = 30 → (n * (n - 1)) / 2 = k :=
by { use 435, intros n hn, rw hn, norm_num, }

end max_handshakes_30_men_l351_351871


namespace cos_330_eq_sqrt3_div_2_l351_351184

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351184


namespace series_sum_equals_one_third_l351_351174

-- Define the function for the general term of the series
def general_term (k : ℕ) : ℚ := 2^k / (8^k - 1)

-- Define the infinite series sum
def series_sum := ∑' k, general_term (k + 1)

-- State the theorem
theorem series_sum_equals_one_third : series_sum = 1 / 3 := 
sorry

end series_sum_equals_one_third_l351_351174


namespace product_mod_7_l351_351742

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l351_351742


namespace construction_of_line_through_point_in_angle_l351_351806

-- Definitions for Lean syntax
variables {A B O M X Y : Point}
variables {m n : ℕ}
variables {angleAOB : Angle O A B}

def ratio (a b : ℝ) : ℝ := a / b

-- The Lean 4 theorem statement
theorem construction_of_line_through_point_in_angle
(O M A B X Y : Point) 
(angleAOB : Angle O A B) 
(m n : ℕ)
(hM : M ∈ interior angleAOB)
(hX : X ∈ segment O A)
(hY : Y ∈ segment O B)
(hXY : line_through M X Y) :
ratio (distance O X) (distance O Y) = ratio m n :=
sorry

end construction_of_line_through_point_in_angle_l351_351806


namespace cos_330_eq_sqrt3_over_2_l351_351202

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351202


namespace pair_d_same_function_l351_351901

theorem pair_d_same_function : ∀ x : ℝ, x = (x ^ 5) ^ (1 / 5) := 
by
  intro x
  sorry

end pair_d_same_function_l351_351901


namespace cos_330_eq_sqrt3_div_2_l351_351185

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351185


namespace edges_less_than_bound_l351_351076

noncomputable section

def number_of_people := 2000

def max_calls_between_two_people := 1

def disjoint_sets_property (A B : Finset ℕ) : Prop :=
  A ∩ B = ∅ → ∃ a ∈ A, ∃ b ∈ B, a ≠ b ∧ (a, b) ∉ called_edges

def called_edges := {e : ℕ × ℕ | e.1 < e.2 ∧ called e.1 e.2}

def called (a b : ℕ) : Bool := sorry

theorem edges_less_than_bound (called_edges : Finset (ℕ × ℕ)) :
  called_edges.card < 201000 :=
sorry

end edges_less_than_bound_l351_351076


namespace simplify_trig_expr_l351_351712

theorem simplify_trig_expr (α : ℝ) :
  (sin(π - α) ^ 2 * cos(2 * π - α) * tan(-π + α)) / (sin(-π + α) * tan(-α + 3 * π)) = sin(α) * cos(α) :=
by
  sorry

end simplify_trig_expr_l351_351712


namespace cos_330_is_sqrt3_over_2_l351_351291

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351291


namespace dodgeball_tournament_l351_351018

theorem dodgeball_tournament (N : ℕ) (points : ℕ) :
  points = 1151 →
  (∀ {G : ℕ}, G = N * (N - 1) / 2 →
    (∃ (win_points loss_points tie_points : ℕ), 
      win_points = 15 * (N * (N - 1) / 2 - tie_points) ∧ 
      tie_points = 11 * tie_points ∧ 
      points = win_points + tie_points + loss_points)) → 
  N = 12 :=
by
  intro h_points h_games
  sorry

end dodgeball_tournament_l351_351018


namespace radius_of_sphere_l351_351892

-- Define the right triangular prism and conditions
structure RightTriangularPrism :=
(edge_length : ℝ)
(vertex_on_sphere : ∀ (v : ℝ × ℝ × ℝ), v ∈ sphere O r)

-- Define the sphere with radius r
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ × ℝ) :=
  { p | dist p center = radius }

-- Given conditions
def O : (ℝ × ℝ × ℝ) := (0, 0, 0)
def prism : RightTriangularPrism := { edge_length := 3, vertex_on_sphere := sorry }

-- Radius of the sphere O
def r : ℝ := \frac{\sqrt{21}}{2}

-- Goal: Prove that the radius of sphere O is \(\frac{\sqrt{21}}{2}\)
theorem radius_of_sphere : r = \frac{\sqrt{21}}{2} :=
by sorry

end radius_of_sphere_l351_351892


namespace sum_of_first_10_common_elements_l351_351430

noncomputable def sum_first_common_elements : ℕ :=
  let common_elements := (20 : ℕ) :: (80 : ℕ) :: (320 : ℕ) :: (1280 : ℕ) :: (5120 : ℕ) ::
                        (20480 : ℕ) :: (81920 : ℕ) :: (327680 : ℕ) :: (1310720 : ℕ) :: (5242880 : ℕ) :: []
  in common_elements.sum

theorem sum_of_first_10_common_elements :
  sum_first_common_elements = 6990500 :=
by
  -- Insert mathematical proof here
  sorry

end sum_of_first_10_common_elements_l351_351430


namespace cos_330_is_sqrt3_over_2_l351_351288

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351288


namespace gcd_75_100_l351_351832

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351832


namespace cos_330_eq_sqrt_3_div_2_l351_351371

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351371


namespace determine_x_value_l351_351509

theorem determine_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y ^ 3) (h2 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by 
  sorry

end determine_x_value_l351_351509


namespace divisible_by_3_l351_351025

theorem divisible_by_3 (x y : ℤ) (h : (x^2 + y^2) % 3 = 0) : x % 3 = 0 ∧ y % 3 = 0 :=
sorry

end divisible_by_3_l351_351025


namespace cos_330_eq_sqrt3_div_2_l351_351279

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351279


namespace solve_for_k_l351_351713

-- Define the conditions
theorem solve_for_k (k : ℂ) :
  (∀ x : ℂ, (x / (x + 3) + x / (x - 3) = k * x) → (x = 0 ∨ k * x^2 - 2 * x - 9 * k = 0)) →
  (∃! x : ℂ, ∀ y : ℂ, (y = x) ∨ (y ≠ x ∧ k * y^2 - 2 * y - 9 * k = 0) → (discriminant (polynomial.C k * polynomial.X ^ 2 - polynomial.C 2 * polynomial.X - polynomial.C (9 * k)) = 0) →
  (k = (0, (1 / 3) * complex.I) ∨ k = (0, -(1 / 3) * complex.I))) :=
begin
  sorry
end

end solve_for_k_l351_351713


namespace speed_of_stream_l351_351133

-- Conditions
variables (b s : ℝ)

-- Downstream and upstream conditions
def downstream_speed := 150 = (b + s) * 5
def upstream_speed := 75 = (b - s) * 7

-- Goal statement
theorem speed_of_stream (h1 : downstream_speed b s) (h2 : upstream_speed b s) : s = 135/14 :=
by sorry

end speed_of_stream_l351_351133


namespace actual_yield_H2O_l351_351392

def balanced_equation : Prop :=
  (2 * CH4 + 5 * CO2 + C3H8) = (4 * C2H5OH + 4 * H2O)

def conditions (CH4 CO2 C3H8 : ℕ) : Prop :=
  CH4 = 3 ∧ CO2 = 4 ∧ C3H8 = 5

def theoretical_yield : ℝ := 0.95

theorem actual_yield_H2O (CH4 CO2 C3H8 : ℕ) (H2O : ℝ) :
  conditions CH4 CO2 C3H8 →
  balanced_equation →
  CO2 = 4 →
  H2O = 3.04 := by
  intros h_cond h_bal h_CO2
  sorry

end actual_yield_H2O_l351_351392


namespace close_functions_interval_l351_351651

def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

theorem close_functions_interval (a b : ℝ) (h : ∀ x ∈ set.Icc a b, |f x - g x| ≤ 1) : a = 2 ∧ b = 3 :=
by
  sorry

end close_functions_interval_l351_351651


namespace sufficient_but_not_necessary_condition_l351_351646

theorem sufficient_but_not_necessary_condition (a : ℝ) (M : set ℝ) (N : set ℝ) (hM : M = {1, 2}) (hN : N = {a^2}) :
  (a = -1 → N ⊆ M) ∧ ¬(∀ x : ℝ, N ⊆ M → x = -1) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l351_351646


namespace cos_330_eq_sqrt_3_div_2_l351_351218

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351218


namespace cos_330_eq_sqrt3_div_2_l351_351306

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351306


namespace gcd_75_100_l351_351823

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351823


namespace equation_of_other_line_l351_351797

theorem equation_of_other_line
  (x_line : ∀ y : ℝ, y = -5 → ∃ x : ℝ, x = -5)
  (y_line : ∀ x : ℝ, y = x)
  (area_formed : ∃ (base height : ℝ), (1 / 2) * base * height = 12.5)
  : y_line ∈ { y | ∀ x : ℝ, y = x } := by
  sorry

end equation_of_other_line_l351_351797


namespace gcd_72_120_180_is_12_l351_351731

theorem gcd_72_120_180_is_12 : Int.gcd (Int.gcd 72 120) 180 = 12 := by
  sorry

end gcd_72_120_180_is_12_l351_351731


namespace max_balls_guaranteed_l351_351024

theorem max_balls_guaranteed (n k : ℕ) (hn : Odd n) (hballs : ∑ i in range n, balls i = 2013) :
  ∃ k, (∀ b : ℕ, b < n → (∑ i in selected_boxes b n, balls i) = 2012) :=
sorry

end max_balls_guaranteed_l351_351024


namespace radius_of_circle_is_sqrt_two_l351_351778

-- Definitions based on the conditions given in the problem
def rho : ℝ := 2
def phi : ℝ := Real.pi / 4

-- Lean statement of the proof problem
theorem radius_of_circle_is_sqrt_two (theta : ℝ) :
  let x := rho * Real.sin phi * Real.cos theta
  let y := rho * Real.sin phi * Real.sin theta
  sqrt (x^2 + y^2) = sqrt 2 :=
sorry

end radius_of_circle_is_sqrt_two_l351_351778


namespace remainder_of_product_mod_7_l351_351755

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l351_351755


namespace simplify_expression_l351_351710

theorem simplify_expression (t : ℝ) : (t ^ 5 * t ^ 3) / t ^ 2 = t ^ 6 :=
by
  sorry

end simplify_expression_l351_351710


namespace cos_330_eq_sqrt3_div_2_l351_351244

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351244


namespace function_evaluation_example_l351_351959

noncomputable def g : ℕ → ℕ
| 1 := 3
| 2 := 4
| 3 := 6
| 4 := 8
| 5 := 9
| _ := 0  -- default to 0 if not defined

axiom g_invertible : Function.LeftInverse g g ∧ ∀ x, g x ≠ 0

theorem function_evaluation_example :
  let g_inv (y : ℕ) := y
  -- Define the inverse values explicitly for known values
  have g_inv_6 : g_inv (g 3) = 3, by rfl,
  have g_inv_3 : g_inv 3 = 1, from rfl,
  g(g(3)) = 0 ∧ g(g(g_inv 3)) + g_inv(g_inv 6) = 4 :=
begin
  sorry
end

end function_evaluation_example_l351_351959


namespace sum_of_coordinates_of_A_l351_351591

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351591


namespace sum_first_10_common_elements_in_ap_gp_l351_351433

/-- To find the sum of the first 10 elements that appear in both the arithmetic progression (AP) 
  {5, 8, 11, 14, ...} and the geometric progression (GP) {10, 20, 40, 80, ...}, we follow these steps:
-/
theorem sum_first_10_common_elements_in_ap_gp 
  (a_n : ℕ → ℕ := λ n, 5 + 3 * n)
  (b_k : ℕ → ℕ := λ k, 10 * 2^k)
  (common_elements : ℕ → ℕ := λ m, 20 * 4^m) :
  (Finset.range 10).sum (λ i, common_elements i) = 6990500 := 
by
  -- Set up the common_elements based on the given progressions
  -- Calculate the sum of the first 10 terms of the geometric progression
  sorry

end sum_first_10_common_elements_in_ap_gp_l351_351433


namespace engineering_department_men_count_l351_351617

theorem engineering_department_men_count
  (total_students : ℕ)
  (women_count : ℕ)
  (men_percentage : ℝ)
  (h1 : women_count = 180)
  (h2 : men_percentage = 0.70)
  (h3 : 0.30 * total_students = women_count)
  : 0.70 * total_students = 420 :=
by 
  sorry

end engineering_department_men_count_l351_351617


namespace probability_answered_within_first_four_rings_l351_351855

theorem probability_answered_within_first_four_rings 
  (P1 P2 P3 P4 : ℝ) (h1 : P1 = 0.1) (h2 : P2 = 0.3) (h3 : P3 = 0.4) (h4 : P4 = 0.1) :
  (1 - ((1 - P1) * (1 - P2) * (1 - P3) * (1 - P4))) = 0.9 := 
sorry

end probability_answered_within_first_four_rings_l351_351855


namespace gambler_received_max_2240_l351_351107

def largest_amount_received_back (x y l : ℕ) : ℕ :=
  if 2 * l + 2 = 14 ∨ 2 * l - 2 = 14 then 
    let lost_value_1 := (6 * 100 + 8 * 20)
    let lost_value_2 := (8 * 100 + 6 * 20)
    max (3000 - lost_value_1) (3000 - lost_value_2)
  else 0

theorem gambler_received_max_2240 {x y : ℕ} (hx : 20 * x + 100 * y = 3000)
  (hl : ∃ l : ℕ, (l + (l + 2) = 14 ∨ l + (l - 2) = 14)) :
  largest_amount_received_back x y 6 = 2240 ∧ largest_amount_received_back x y 8 = 2080 := by
  sorry

end gambler_received_max_2240_l351_351107


namespace basketball_team_win_rate_l351_351120

theorem basketball_team_win_rate (won_initial : ℕ) (total_initial : ℕ) (total_remaining : ℕ)
  (desired_percentage : ℚ) :
  total_initial = 60 → won_initial = 45 → total_remaining = 50 → desired_percentage = 3 / 4 →
  ∃ wins_remaining: ℕ, wins_remaining = 38 :=
by
  intros _ _ _ _
  use 38
  sorry

end basketball_team_win_rate_l351_351120


namespace area_quadrilateral_ABDM_l351_351958

-- Definition of the 16-sided polygon with specified properties
structure Polygon16 :=
  (A P : Type)
  (side_length : ℝ)
  (side_length_eq_5 : side_length = 5)
  (angles_right : ∀ (i : ℕ), i < 16 → right_angle (angle (A i) (A (i+1))))

-- Definitions of the points and lines
def points_and_lines (P : Polygon16) :=
  exists 
    (A' J' D' N' M' : P.A)
    (AJ' DN' : P.A → P.A → line)
    (intersect_at_M' : AJ' A' J' = DN' D' N' ∧ intersects_at AJ' DN' M')

-- Theorem statement
theorem area_quadrilateral_ABDM (P : Polygon16) (h : points_and_lines P) : 
  (area (quadrilateral (P.A 0) (P.A 1) (P.A 3) M) = 50) :=
sorry

end area_quadrilateral_ABDM_l351_351958


namespace sum_of_coords_A_l351_351573

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351573


namespace range_of_a_l351_351486

theorem range_of_a (a : ℝ) 
  (f : ℝ → ℝ := λ x, if x > 1 then a^x else (6 - a) * x) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) : 
  3 ≤ a ∧ a < 6 :=
  sorry

end range_of_a_l351_351486


namespace smallest_m_l351_351655

variable (S : Type) [Fintype S] [DecidableEq S]
def P (F : set (S → S)) (k : ℕ) : Prop :=
  ∀ x y : S, ∃ f : Fin k → S → S, (∀ i, f i ∈ F) ∧
    f (k-1) (f (k-2) (...(f 0 x)...)) = f (k-1) (f (k-2) (...(f 0 y)...))

theorem smallest_m (S : Finset (Fin 35)) (F : finset (S → S)) (h : P F 2019) : P F 595 :=
sorry

end smallest_m_l351_351655


namespace cos_330_eq_sqrt3_over_2_l351_351207

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351207


namespace distance_point_C_to_line_is_2_inch_l351_351442

/-- 
Four 2-inch squares are aligned in a straight line. The second square from the left is rotated 90 degrees, 
and then shifted vertically downward until it touches the adjacent squares. Prove that the distance from 
point C, the top vertex of the rotated square, to the original line on which the bases of the squares were 
placed is 2 inches.
-/
theorem distance_point_C_to_line_is_2_inch :
  ∀ (squares : Fin 4 → ℝ) (rotation : ℝ) (vertical_shift : ℝ) (C_position : ℝ),
  (∀ n : Fin 4, squares n = 2) →
  rotation = 90 →
  vertical_shift = 0 →
  C_position = 2 →
  C_position = 2 :=
by
  intros squares rotation vertical_shift C_position
  sorry

end distance_point_C_to_line_is_2_inch_l351_351442


namespace remainder_product_l351_351746

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351746


namespace number_of_diagonals_excluding_dividing_diagonals_l351_351172

theorem number_of_diagonals_excluding_dividing_diagonals (n : ℕ) (h1 : n = 150) :
  let totalDiagonals := n * (n - 3) / 2
  let dividingDiagonals := n / 2
  totalDiagonals - dividingDiagonals = 10950 :=
by
  sorry

end number_of_diagonals_excluding_dividing_diagonals_l351_351172


namespace geometric_sequence_a10_l351_351538

theorem geometric_sequence_a10
  (a : ℕ → ℝ)
  (h_geo : ∃ r, ∀ n, a(n+1) = r * a(n))
  (h_pos : ∀ n, a(n) > 0)
  (h_root1 : x^2 - 10*x + 16 = 0 → x = a(1) ∨ x = a(19)) :
  a(10) = 4 :=
by
  sorry

end geometric_sequence_a10_l351_351538


namespace cos_330_eq_sqrt3_div_2_l351_351347

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351347


namespace total_packs_is_117_l351_351689

-- Defining the constants based on the conditions
def nancy_cards : ℕ := 540
def melanie_cards : ℕ := 620
def mary_cards : ℕ := 480
def alyssa_cards : ℕ := 720

def nancy_cards_per_pack : ℝ := 18.5
def melanie_cards_per_pack : ℝ := 22.5
def mary_cards_per_pack : ℝ := 15.3
def alyssa_cards_per_pack : ℝ := 24

-- Calculating the number of packs each person has
def nancy_packs := (nancy_cards : ℝ) / nancy_cards_per_pack
def melanie_packs := (melanie_cards : ℝ) / melanie_cards_per_pack
def mary_packs := (mary_cards : ℝ) / mary_cards_per_pack
def alyssa_packs := (alyssa_cards : ℝ) / alyssa_cards_per_pack

-- Rounding down the number of packs
def nancy_packs_rounded := nancy_packs.toNat
def melanie_packs_rounded := melanie_packs.toNat
def mary_packs_rounded := mary_packs.toNat
def alyssa_packs_rounded := alyssa_packs.toNat

-- Summing the total number of packs
def total_packs : ℕ := nancy_packs_rounded + melanie_packs_rounded + mary_packs_rounded + alyssa_packs_rounded

-- Proposition stating that the total number of packs is 117
theorem total_packs_is_117 : total_packs = 117 := by
  sorry

end total_packs_is_117_l351_351689


namespace cos_330_is_sqrt3_over_2_l351_351300

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351300


namespace power_sum_roots_l351_351663

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l351_351663


namespace element_in_set_l351_351643

open Set

theorem element_in_set : -7 ∈ ({1, -7} : Set ℤ) := by
  sorry

end element_in_set_l351_351643


namespace math_problem_l351_351484

-- Definitions
def f (x : ℝ) := log 2 (x^2 - 2 * x + 1)
def g (a b x : ℝ) := if x ≤ 0 then x + b else a^x - 4

-- Mathematical equivalent proof problem in Lean
theorem math_problem (a : ℝ) (b : ℝ) (ha : a > 0) :
  (∀ x, f x = 0 ↔ x = 0 ∨ x = 2) ∧
  (f 0 = 0 ∧ f 2 = 0) →
  b = 0 ∧ a = 2 ∧ (∀ y ∈ set.range (λ x, g a b x), y ∈ set.univ) :=
by sorry

end math_problem_l351_351484


namespace high_school_heralds_loss_percentage_is_19_percent_l351_351069

theorem high_school_heralds_loss_percentage_is_19_percent
  (ratio_won_lost : ℚ := 13 / 3)
  (total_games : ℕ := 64)
  (games_won := 52)
  (games_lost := 12)
  : ((games_lost : ℚ) / total_games) * 100 ≈ 19 :=
by
  sorry

end high_school_heralds_loss_percentage_is_19_percent_l351_351069


namespace min_value_of_PQ_plus_PO_l351_351468

noncomputable def min_distance (P Q O : Point) : Real :=
  Real.dist P Q + (Real.sqrt 2 / 2) * Real.dist P O

theorem min_value_of_PQ_plus_PO :
  ∀ (P Q : Point),
  (P ∈ C ∧ Q ∈ D ∧ O = (0, 0)) →
  (∃ P Q, min_distance P Q (0, 0) = Real.sqrt 5) :=
sorry

end min_value_of_PQ_plus_PO_l351_351468


namespace cos_330_is_sqrt3_over_2_l351_351298

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351298


namespace log_intersection_problem_l351_351129

theorem log_intersection_problem (k a b : ℝ) (h1 : k = a + Real.sqrt b) (h2 : |Real.logBase 3 k - Real.logBase 3 (k + 4)| = 1) 
  (ha : a ∈ Set.Ioo (-1 : ℝ) 3) (hb : b ∈ Set.Ici 0 ) :
  a + b = 2 := 
sorry

end log_intersection_problem_l351_351129


namespace rectangles_equidecomposable_of_equal_area_l351_351109

theorem rectangles_equidecomposable_of_equal_area
  {ABCD A1B1C1D1 : ℝ × ℝ}  -- Representing the side lengths of the rectangles.
  (hArea : (ABCD.fst * ABCD.snd) = (A1B1C1D1.fst * A1B1C1D1.snd)) :
  is_equidecomposable_rectangles ABCD A1B1C1D1 := 
sorry

end rectangles_equidecomposable_of_equal_area_l351_351109


namespace sum_maximization_l351_351004

def a (n : ℕ) : ℤ := -n^2 + 10 * n + 11

noncomputable def S (n : ℕ) : ℤ := ∑ k in Finset.range (n + 1), a k

theorem sum_maximization : (S 10 = S 11) ∧ (∀ n, n ≠ 10 ∨ n ≠ 11 → S n ≤ S 10) := sorry

end sum_maximization_l351_351004


namespace cos_330_eq_sqrt3_div_2_l351_351303

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351303


namespace sum_of_coordinates_A_l351_351566

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351566


namespace asymptote_of_hyperbola_l351_351467

-- Definition of the hyperbola and related variables
def hyperbola_equation (x y a : ℝ) : Prop := (x^2 / a^2) - (y^2 / 2) = 1
def distance_between_foci (a : ℝ) : ℝ := a * sqrt (3 / 2)
def line_equation (x c : ℝ) : Prop := x = -c
def y_coordinates (a c y : ℝ) : Prop := y = sqrt(c^2 / a^2 - 1) * sqrt(2) ∨ y = -sqrt(c^2 / a^2 - 1) * sqrt(2)
def area_of_triangle (y c : ℝ) : Prop := y * 2 * c = 2 * sqrt 6

-- Proof problem statement
theorem asymptote_of_hyperbola (a c y : ℝ) (h_a_nonneg: 0 < a)
  (h_hyperbola : hyperbola_equation (-c) y a)
  (h_line : line_equation (-c) c)
  (h_y_coords : y_coordinates a c y)
  (h_area : area_of_triangle y c) :
  ∃ m : ℝ, m = sqrt 2 / 2 := 
sorry

end asymptote_of_hyperbola_l351_351467


namespace distance_between_foci_of_ellipse_l351_351930

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351930


namespace prism_volume_l351_351890

-- Define the dimensions and areas
variables (l w h : ℝ)

-- Given conditions
axiom area1 : l * w = 15
axiom area2 : w * h = 20
axiom area3 : l * h = 30

-- To prove the volume
theorem prism_volume : l * w * h = 60 * real.sqrt 10 :=
by sorry

end prism_volume_l351_351890


namespace num_divisors_with_divisor_count_1806_l351_351386

/-
Define the prime factorization of 1806 and its power.
-/
def prime_factors_1806 : List (ℕ × ℕ) := [(2, 1), (3, 2), (101, 1)]

/-
Define the prime factorization of 1806^1806.
-/
def prime_factors_1806_pow : List (ℕ × ℕ) := [(2, 1806), (3, 3612), (101, 1806)]

/-
Define a positive divisor of 1806^1806.
-/
def is_divisor (d : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    d = 2^a * 3^b * 101^c ∧ (a + 1) * (b + 1) * (c + 1) = 1806

/-
Define the problem statement.
-/
theorem num_divisors_with_divisor_count_1806 :
  (Finset.univ.filter is_divisor).card = 36 :=
begin
  sorry
end

end num_divisors_with_divisor_count_1806_l351_351386


namespace cos_330_is_sqrt3_over_2_l351_351295

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351295


namespace cos_330_eq_sqrt_3_div_2_l351_351227

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351227


namespace eval_expression_l351_351978

theorem eval_expression : (2023 - 1984)^2 / 144 = 10 := by
  have h1 : 2023 - 1984 = 39 := rfl
  have h2 : (2023 - 1984)^2 = 39^2 := by rw [h1]
  have h3 : 39^2 = 1521 := by norm_num
  have h4 : 1521 / 144 = 10 := by norm_num
  rw [h2]
  exact h4

end eval_expression_l351_351978


namespace enrique_commission_l351_351403

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l351_351403


namespace cos_330_eq_sqrt3_div_2_l351_351345

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351345


namespace cost_price_of_apple_l351_351112

theorem cost_price_of_apple (SP : ℝ) (hSP : SP = 20) (loss : ℝ) (CP : ℝ) (h_loss : loss = CP / 6) 
  (h_eq : SP = CP - loss) : CP = 24 :=
by 
  have h1 : 20 = CP - CP / 6 := by rw [hSP, h_loss, h_eq]
  have h2 : 20 = (5/6) * CP := by linarith 
  have h3 : 20 * (6/5) = CP := by linarith
  rw [h3]
  norm_num
  sorry

end cost_price_of_apple_l351_351112


namespace find_w_l351_351049

variable (x y z w : ℝ)

theorem find_w (h : (x + y + z) / 3 = (y + z + w) / 3 + 10) : w = x - 30 := by 
  sorry

end find_w_l351_351049


namespace domain_log_base_2_l351_351055

theorem domain_log_base_2 (x : ℝ) : x > 3 ↔ ∃ y, y = log (x - 3) / log 2 :=
by
  sorry

end domain_log_base_2_l351_351055


namespace cos_330_eq_sqrt3_div_2_l351_351305

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351305


namespace remainder_product_l351_351749

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l351_351749


namespace triangle_area_l351_351154

noncomputable def area_of_right_triangle (a b c : ℝ) (h : a ^ 2 + b ^ 2 = c ^ 2) : ℝ :=
  (1 / 2) * a * b

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_of_right_triangle a b c h4 = 54 := by
  rw [h1, h2, h3]
  sorry

end triangle_area_l351_351154


namespace gcd_75_100_l351_351826

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l351_351826


namespace range_of_a_l351_351383

/--
Define: [A] represents the number of elements in set A, 
A ⊗ B = 
  if [A] ≥ [B] then [A] - [B] 
  else [B] - [A].
Given set M = {1, 2}, set A = {x | x ⊆ M}, 
set B = {x | x(x^2 - 1)(x^2 - ax + 4) = 0}.
If A ⊗ B = 1, then the range of values for a is {a | a ≠ ± 4 and a ≠ ± 5}.
-/

def M : set ℕ := {1, 2}
def A : set (set ℕ) := {x | x ⊆ M}
def B (a : ℝ) : set ℝ := {x | x * (x^2 - 1) * (x^2 - a * x + 4) = 0}
def card (s : set (set ℕ)) : ℕ := set.finite.to_finset s.finite_to_set.card
def A_ox_B (A B : set (set ℚ)) : ℕ := 
  if card A ≥ card B then card A - card B else card B - card A

theorem range_of_a (a : ℝ) (h : A_ox_B A (B a) = 1) :
  a ≠ 4 ∧ a ≠ -4 ∧ a ≠ 5 ∧ a ≠ -5 :=
sorry

end range_of_a_l351_351383


namespace cos_330_eq_sqrt3_div_2_l351_351250

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351250


namespace taxi_ride_cost_l351_351940

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l351_351940


namespace cos_330_eq_sqrt3_div_2_l351_351350

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351350


namespace cycles_bijections_l351_351642

theorem cycles_bijections (k : ℕ) (h_pos : k > 0) : 
  let n := 2^k in 
  let N := finset.range (n + 1) in 
  let bijections := {f : N → N // function.bijective f} in 
  ∃ (f_set : finset bijections), 
  f_set.card ≥ n! / 2 ∧ 
  ∀ f ∈ f_set, (finset_univ.count_cycles f).length ≤ 2 * k - 1 :=
begin
  let n := 2^k,
  have hN : ∀ x ∈ N, x > 0, from sorry,
  sorry
end

end cycles_bijections_l351_351642


namespace cos_330_eq_sqrt_3_div_2_l351_351368

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351368


namespace equal_angles_proof_l351_351720

/-- Proof Problem: After how many minutes will the hour and minute hands form equal angles with their positions at 12 o'clock? -/
noncomputable def equal_angle_time (x : ℝ) : Prop :=
  -- Defining the conditions for the problem
  let minute_hand_speed := 6 -- degrees per minute
  let hour_hand_speed := 0.5 -- degrees per minute
  let total_degrees := 360 * x -- total degrees of minute hand till time x
  let hour_hand_degrees := 30 * (x / 60) -- total degrees of hour hand till time x

  -- Equation for equal angles formed with respect to 12 o'clock
  30 * (x / 60) = 360 - 360 * (x / 60)

theorem equal_angles_proof :
  ∃ (x : ℝ), equal_angle_time x ∧ x = 55 + 5/13 :=
sorry

end equal_angles_proof_l351_351720


namespace correct_system_l351_351969

-- Define the conditions
variables (x y : ℕ)
def total_items := x + y = 20
def total_cost := 4 * x + 3 * y = 72

-- Define the target system of equations corresponding to option B
def system_b := total_items ∧ total_cost

-- The theorem stating the equivalence of the given conditions with the correct choice
theorem correct_system : system_b := by
  sorry

end correct_system_l351_351969


namespace triangle_properties_l351_351527

noncomputable theory

open Real

theorem triangle_properties 
  (A B C : ℝ)
  (B : ℝ)
  (BC : ℝ)
  (h1 : sqrt 3 * sin (2 * B) = 1 - cos (2 * B))
  (h2 : BC = 2)
  (h3 : A = π / 4) :
  (B = π / 6) ∧ (0.5 * (BC * (√6 * BC * sin (π - A - B) / sin B)) * sin (π - A - B) = (3 + sqrt 3) / 2) :=
begin
  sorry,
end

end triangle_properties_l351_351527


namespace simplify_fraction_l351_351037

theorem simplify_fraction (a b gcd : ℕ) (h1 : a = 72) (h2 : b = 108) (h3 : gcd = Nat.gcd a b) : (a / gcd) / (b / gcd) = 2 / 3 :=
by
  -- the proof is omitted here
  sorry

end simplify_fraction_l351_351037


namespace sum_of_coords_A_l351_351574

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351574


namespace weight_of_third_student_l351_351723

-- Definitions for the conditions
def average_weight_29_students : ℝ := 28
def average_weight_32_students : ℝ := 27.3
def first_new_student_weight : ℝ := 20
def second_new_student_weight : ℝ := 30

-- Main proof statement
theorem weight_of_third_student :
  let total_weight_29_students := 29 * average_weight_29_students,
      total_weight_32_students := 32 * average_weight_32_students,
      total_weight_new_students := total_weight_32_students - total_weight_29_students,
      third_new_student_weight := total_weight_new_students - first_new_student_weight - second_new_student_weight
  in third_new_student_weight = 11.6 :=
by
  sorry

end weight_of_third_student_l351_351723


namespace sum_of_coordinates_A_l351_351563

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351563


namespace gcd_75_100_l351_351833

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l351_351833


namespace average_runs_next_7_matches_l351_351121

def batsman_average_25_matches := 45
def batsman_average_32_matches := 38.4375
def number_of_first_25_matches := 25
def number_of_next_7_matches := 7

theorem average_runs_next_7_matches :
  (batsman_average_32_matches * (number_of_first_25_matches + number_of_next_7_matches) - batsman_average_25_matches * number_of_first_25_matches) / number_of_next_7_matches = 15 :=
by
  sorry

end average_runs_next_7_matches_l351_351121


namespace cos_330_eq_sqrt3_div_2_l351_351355

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351355


namespace topics_assignment_l351_351873

theorem topics_assignment (students groups arrangements : ℕ) (h1 : students = 6) (h2 : groups = 3) (h3 : arrangements = 90) :
  let T := arrangements / (students * (students - 1) / 2 * (4 * 3 / 2 * 1))
  T = 1 :=
by
  sorry

end topics_assignment_l351_351873


namespace count_leap_years_l351_351135

def is_leap_year_ending_in_double_zeros_new_rule (y : ℕ) : Prop :=
  (y % 900 = 200 ∨ y % 900 = 600) ∧ (y % 100 = 0)

def count_leap_years_ending_in_double_zeros_new_rule (start finish : ℕ) : ℕ :=
  (List.range' start (finish - start)).count (is_leap_year_ending_in_double_zeros_new_rule)

theorem count_leap_years : count_leap_years_ending_in_double_zeros_new_rule 2000 5000 = 7 := 
by 
  sorry

end count_leap_years_l351_351135


namespace sum_of_coords_A_l351_351575

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l351_351575


namespace total_profit_correct_l351_351877

-- We define the conditions
variables (a m : ℝ)

-- The item's cost per piece
def cost_per_piece : ℝ := a
-- The markup percentage
def markup_percentage : ℝ := 0.20
-- The discount percentage
def discount_percentage : ℝ := 0.10
-- The number of pieces sold
def pieces_sold : ℝ := m

-- Definitions derived from conditions
def selling_price_markup : ℝ := cost_per_piece a * (1 + markup_percentage)
def selling_price_discount : ℝ := selling_price_markup a * (1 - discount_percentage)
def profit_per_piece : ℝ := selling_price_discount a - cost_per_piece a
def total_profit : ℝ := profit_per_piece a * pieces_sold m

theorem total_profit_correct (a m : ℝ) : total_profit a m = 0.08 * a * m :=
by sorry

end total_profit_correct_l351_351877


namespace cos_330_eq_sqrt3_div_2_l351_351314

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351314


namespace gcf_75_100_l351_351822

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l351_351822


namespace find_relation_l351_351495

theorem find_relation (x y : ℕ) :
  (x = 0 ∧ y = 200) ∨
  (x = 1 ∧ y = 150) ∨
  (x = 2 ∧ y = 100) ∨
  (x = 3 ∧ y = 50) ∨
  (x = 4 ∧ y = 0) →
  y = -50 * x + 200 :=
by
  intro h
  cases h
  case inl h₀ { rw [h₀.left, h₀.right] }
  case inr h₁ { cases h₁
    case inl h₁₀ { rw [h₁₀.left, h₁₀.right] }
    case inr h₂ { cases h₂
      case inl h₂₀ { rw [h₂₀.left, h₂₀.right] }
      case inr h₃ { cases h₃
        case inl h₃₀ { rw [h₃₀.left, h₃₀.right] }
        case inr h₄ { rw [h₄.left, h₄.right] } } } } }
  sorry

end find_relation_l351_351495


namespace moles_of_CaCO3_formed_l351_351992

theorem moles_of_CaCO3_formed (m n : ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : ∀ m n : ℕ, (m = n) → (m = 3) → (n = 3) → moles_of_CaCO3 = m) : 
  moles_of_CaCO3 = 3 := by
  sorry

end moles_of_CaCO3_formed_l351_351992


namespace solve_oplus_bicirc_l351_351384

theorem solve_oplus_bicirc :
  let (a : ℕ) ⊕ (b : ℕ) := a * b
  let (c : ℕ) ⊙ (d : ℕ) := d ^ c
  (5 ⊕ 8) ⊕ (3 ⊙ 7) = 13720 :=
by
  -- The proof is omitted.
  sorry

end solve_oplus_bicirc_l351_351384


namespace probability_divisible_is_15_over_44_l351_351732
-- Import the entire Mathlib library

noncomputable def count_divisible_pairs : ℕ :=
  let r_values := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let k_values := [1, 2, 3, 4, 5, 6, 7, 8]
  (r_values.product k_values).count (λ (rk : ℤ × ℤ), rk.1 % rk.2 = 0)

noncomputable def total_possible_pairs : ℕ :=
  let r_values := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let k_values := [1, 2, 3, 4, 5, 6, 7, 8]
  r_values.length * k_values.length

noncomputable def probability_divisible : ℚ :=
  count_divisible_pairs / total_possible_pairs

theorem probability_divisible_is_15_over_44 :
  probability_divisible = 15 / 44 := sorry

end probability_divisible_is_15_over_44_l351_351732


namespace line_tangent_to_parabola_l351_351998

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end line_tangent_to_parabola_l351_351998


namespace gender_related_to_judgment_l351_351868

theorem gender_related_to_judgment
    (male_participants : ℕ) (male_opposite_view : ℕ)
    (female_participants : ℕ) (female_opposite_view : ℕ)
    (correct_method : String) :
    male_participants = 2600 →
    male_opposite_view = 1560 →
    female_participants = 2400 →
    female_opposite_view = 1118 →
    correct_method = "Independence test" :=
by
  intros h1 h2 h3 h4
  exact "Independence test"

end gender_related_to_judgment_l351_351868


namespace radius_of_new_circle_l351_351085

theorem radius_of_new_circle
  (r1 r2 : ℝ) (r1_eq : r1 = 24) (r2_eq : r2 = 34)
  : ∃ r : ℝ, π * r ^ 2 = (π * r2 ^ 2 - π * r1 ^ 2) ∧ r = 2 * Real.sqrt 145 :=
by
  have A_outer := π * r2 ^ 2,
  have A_inner := π * r1 ^ 2,
  have A_shaded := A_outer - A_inner,
  have r := Real.sqrt (A_shaded / π),
  use r,
  sorry

end radius_of_new_circle_l351_351085


namespace sum_of_coordinates_A_l351_351557

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l351_351557


namespace cos_330_eq_sqrt3_div_2_l351_351310

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351310


namespace isosceles_triangle_side_length_l351_351724

theorem isosceles_triangle_side_length (base : ℝ) (area : ℝ) (congruent_side : ℝ) 
  (h_base : base = 30) (h_area : area = 60) : congruent_side = Real.sqrt 241 :=
by 
  sorry

end isosceles_triangle_side_length_l351_351724


namespace find_A_coordinates_sum_l351_351583

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351583


namespace cos_330_eq_sqrt3_div_2_l351_351261

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351261


namespace whole_2_config_is_m_separable_l351_351034

-- Definitions based on the conditions
def is_2_configuration (C : set (α × α)) (A : set α) (As : list (set α)) : Prop :=
  (∀ (i j : ℕ) (hi : i < As.length) (hj : j < As.length),
    i ≠ j → ∀ a b, a ∈ As.nth_le i hi → b ∈ As.nth_le j hj → (a, b) ∉ C)

def m_separable (A : set α) (f : A → fin m) : Prop :=
  ∀ a b, a ∈ A → b ∈ A → a ≠ b → f a ≠ f b

-- Main statement
theorem whole_2_config_is_m_separable
  {α : Type*} {m : ℕ} (A : set α) (C : set (α × α)) (As : list (set α))
  (hconfig : is_2_configuration C A As)
  (hsep : ∀ i (hi : i < As.length), ∃ fi : As.nth_le i hi → fin m, m_separable (As.nth_le i hi) fi) :
  ∃ f : A → fin m, m_separable A f :=
sorry

end whole_2_config_is_m_separable_l351_351034


namespace smallest_number_grouped_equally_l351_351101

theorem smallest_number_grouped_equally (x : ℕ) :
  (∀ x, (x % 18 = 0) ∧ (x % 60 = 0) → x ≥ 180) :=
by
  intro x
  have h1 : x % 18 = 0
  have h2 : x % 60 = 0
  sorry

example : (∀ x, (x % 18 = 0) ∧ (x % 60 = 0) → x = 180) :=
by
  have hx : ∃ x, (x % 18 = 0) ∧ (x % 60 = 0)
  sift
  rw [Nat.gcd, Nat.mul, Nat.lcm]
  refine hx_inl
  sorry

end smallest_number_grouped_equally_l351_351101


namespace ellipse_foci_distance_l351_351912

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l351_351912


namespace ellipse_foci_distance_l351_351920

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l351_351920


namespace sum_of_coordinates_A_l351_351568

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351568


namespace cos_330_eq_sqrt3_div_2_l351_351275

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351275


namespace volume_of_increased_cube_l351_351125

def edge_length_original := 10
def increase_percentage := 0.1
def edge_length_new := edge_length_original * (1 + increase_percentage)
def volume_cube (a : ℝ) := a^3

theorem volume_of_increased_cube :
  volume_cube edge_length_new = 1331 := by
  sorry

end volume_of_increased_cube_l351_351125


namespace train_distance_covered_l351_351150

-- Definitions based on the given conditions
def average_speed := 3   -- in meters per second
def total_time := 9      -- in seconds

-- Theorem statement: Given the average speed and total time, the total distance covered is 27 meters
theorem train_distance_covered : average_speed * total_time = 27 := 
by
  sorry

end train_distance_covered_l351_351150


namespace area_of_circle_with_diameter_10_l351_351095

theorem area_of_circle_with_diameter_10 (d : ℝ) (π : ℝ) (h : d = 10): 
  ∃ A, A = π * ((d / 2) ^ 2) ∧ A = 25 * π :=
begin
  use π * ((10 / 2) ^ 2),
  split,
  { rw h, },
  { ring }
end

end area_of_circle_with_diameter_10_l351_351095


namespace cos_330_eq_sqrt3_div_2_l351_351325

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351325


namespace distance_between_foci_of_given_ellipse_l351_351904

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l351_351904


namespace smallest_integer_k_l351_351849

theorem smallest_integer_k : ∀ (k : ℕ), (64^k > 4^16) → k ≥ 6 :=
by
  sorry

end smallest_integer_k_l351_351849


namespace even_difference_exists_l351_351113

theorem even_difference_exists :
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2010) ∧ (1 ≤ j ∧ j ≤ 2010) ∧ (i ≠ j) ∧ (abs (i - j) % 2 = 0) := by
sorry

end even_difference_exists_l351_351113


namespace arithmetic_mean_after_removal_l351_351932

theorem arithmetic_mean_after_removal 
  (mean_original : ℝ) (num_original : ℕ) 
  (nums_removed : List ℝ) (mean_new : ℝ)
  (h1 : mean_original = 50) 
  (h2 : num_original = 60) 
  (h3 : nums_removed = [60, 65, 70, 40]) 
  (h4 : mean_new = 49.38) :
  let sum_original := mean_original * num_original
  let num_remaining := num_original - nums_removed.length
  let sum_removed := List.sum nums_removed
  let sum_new := sum_original - sum_removed
  
  mean_new = sum_new / num_remaining :=
sorry

end arithmetic_mean_after_removal_l351_351932


namespace range_of_m_l351_351449

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔ -4 < m ∧ m < 0 :=
by sorry

end range_of_m_l351_351449


namespace cos_330_eq_sqrt_3_div_2_l351_351225

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351225


namespace sum_of_coordinates_A_l351_351602

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l351_351602


namespace triangle_circle_area_l351_351163

open Real

theorem triangle_circle_area :
  ∀ (A B C O : Point) (r R : ℝ),
    is_triangle ABC ∧
    is_isosceles_triangle ABC A B C ∧
    dist A B = 5 * sqrt 6 ∧
    dist A C = 5 * sqrt 6 ∧
    r = 7 * sqrt 2 ∧
    is_tangent_circle O r A B ∧
    is_tangent_circle O r A C ∧
    passes_through_circle A B C R →
    ∃ π, area_circle R = 108 * π := 
by
  sorry

end triangle_circle_area_l351_351163


namespace distance_between_foci_of_ellipse_l351_351926

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l351_351926


namespace angle_DEC_90_degrees_l351_351891

theorem angle_DEC_90_degrees
  (ABC : Type*)
  [circumscribed ABC circle]
  (BCA_right : ∀ a b c : ABC, right_triangle a b c)
  (D_collinear_BC : ∀ d b c : ABC, collinear d b c)
  (AC_eq_BD : ∀ a c d b : ABC, dist a c = dist d b)
  (E_midpoint_arc_AB : ∀ e a b c : ABC, midpoint (arc a b) e ∧ contains_point (arc a b) c e)
  : ∀ d e : ABC, angle DEC = π / 2 := sorry

end angle_DEC_90_degrees_l351_351891


namespace product_modulo_25_l351_351045

theorem product_modulo_25 : 
  (123 ≡ 3 [MOD 25]) → 
  (456 ≡ 6 [MOD 25]) → 
  (789 ≡ 14 [MOD 25]) → 
  (123 * 456 * 789 ≡ 2 [MOD 25]) := 
by 
  intros h1 h2 h3 
  sorry

end product_modulo_25_l351_351045


namespace gcd_75_100_l351_351837

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351837


namespace modulus_of_complex_l351_351408

-- Define the complex number z as (7/4) - 3i
def z : ℂ := (7 / 4 : ℝ) - 3 * complex.I

-- Statement to prove
theorem modulus_of_complex : complex.abs z = real.sqrt 193 / 4 :=
by
  sorry

end modulus_of_complex_l351_351408


namespace factor_expression_l351_351182

theorem factor_expression (x : ℝ) : 
  ((4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8)) = 2 * x^2 * (5 * x + 31) := 
by sorry

end factor_expression_l351_351182


namespace digits_of_squares_l351_351968

theorem digits_of_squares :
  ∃ (d1 d2 d3 : ℕ), (d1 ≠ 0) ∧ (d2 ≠ 0) ∧ (d3 ≠ 0) ∧
  (∀ n : ℕ, ∃ m : ℕ, m^2 = n → (digit_in_square m d1) ∧ (digit_in_square m d2) ∧ (digit_in_square m d3)) :=
sorry

end digits_of_squares_l351_351968


namespace parents_rated_needs_improvement_l351_351148

@[inline] def percentage(p: ℕ) (total: ℕ) : ℕ := (total * p) / 100

theorem parents_rated_needs_improvement :
  (respondents excellent very_satisfactory satisfactory needs_improvement : ℕ) 
  (h_total : respondents = 120)
  (h_excellent : percentage 15 respondents = excellent)
  (h_very_satisfactory : percentage 60 respondents = very_satisfactory)
  (h_rem : respondents - excellent - very_satisfactory = satisfactory + needs_improvement)
  (h_satisfactory : percentage 80 (satisfactory + needs_improvement) = satisfactory) :
  needs_improvement = 6 :=
by
  sorry

end parents_rated_needs_improvement_l351_351148


namespace parking_lot_vehicle_spaces_l351_351793

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

end parking_lot_vehicle_spaces_l351_351793


namespace options_correct_l351_351510

theorem options_correct (z1 z2 : ℂ) :
  (¬ (z1.im ≠ 0 → z1^2 > 0)) ∧
  (¬ ((z1^2 + z2^2 = 0) → (z1 = 0 ∧ z2 = 0))) ∧
  ((abs z1 = abs z2) → (z1 * conj(z1) = z2 * conj(z2))) ∧
  ((abs z1 = 1) → ((λ b : ℝ, ∃ a : ℝ, z1 = a + b * I) → ∃ b : ℝ, (b ∈ [-1, 1]) → (abs (z1 + 2 * I) = 3))) := 
by sorry

end options_correct_l351_351510


namespace largest_median_l351_351845

theorem largest_median (x : ℕ) (h_pos : x > 0) :
  ∃ m, (m = 5) ∧ 
       let s := multiset.sort (≤) {x, 2*x, 3*x, 5, 4, 7, 1}.to_multiset 
       in s.nth 3 = some m :=
by sorry

end largest_median_l351_351845


namespace rectangle_area_solution_l351_351395

theorem rectangle_area_solution (x : ℝ) (h1 : (x + 3) * (2*x - 1) = 12*x + 5) : 
  x = (7 + Real.sqrt 113) / 4 :=
by 
  sorry

end rectangle_area_solution_l351_351395


namespace line_tangent_to_parabola_l351_351999

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end line_tangent_to_parabola_l351_351999


namespace geometric_sequence_sum_eq_five_l351_351470

/-- Given that {a_n} is a geometric sequence where each a_n > 0
    and the equation a_2 * a_4 + 2 * a_3 * a_5 + a_4 * a_6 = 25 holds,
    we want to prove that a_3 + a_5 = 5. -/
theorem geometric_sequence_sum_eq_five
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a n = a 1 * r ^ (n - 1))
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : a 3 + a 5 = 5 :=
sorry

end geometric_sequence_sum_eq_five_l351_351470


namespace sum_coordinates_A_l351_351597

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351597


namespace part_a_part_b_l351_351108

/-- Part (a): Prove the product of the sequence -/
def productSeq :=
  (∏ k in Finset.range 2006 + 2, (1 + 1 / (k: ℝ))) = 1004

theorem part_a : productSeq := by
  sorry

/-- Part (b): Prove the sum of all products of 2, 4, ..., 2006 different elements from set A -/
def setA := { k : ℝ | ∃ n: ℕ, 2 ≤ n ∧ n ≤ 2007 ∧ k = 1 / n }

def sumEvenProducts (s : Set ℝ) : ℝ :=
  let n := s.toFinset.card
  ((Finset.powersetLen n).sum (λ t, ∏ x in t, x)) 

def sumOfEvenProducts := 
  let A := setA.finset
  sumEvenProducts A ∩ set.range (λ (i : ℕ), 2 * i) = 1003

theorem part_b : sumOfEvenProducts := by
  sorry

end part_a_part_b_l351_351108


namespace cos_330_eq_sqrt3_div_2_l351_351264

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351264


namespace simplify_expression_l351_351041

variable {a b : ℝ}

theorem simplify_expression {a b : ℝ} (h : |2 - a + b| + (ab + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end simplify_expression_l351_351041


namespace cream_ratio_Joe_JoAnn_l351_351637

def Joe_initial_coffee := 15
def Joe_drank_coffee := 3
def Joe_added_cream := 3

def JoAnn_initial_coffee := 15
def JoAnn_added_cream := 3
def JoAnn_drank_total := 3

theorem cream_ratio_Joe_JoAnn :
  let Joe_final_cream := Joe_added_cream,
      JoAnn_total_volume := JoAnn_initial_coffee + JoAnn_added_cream,
      JoAnn_cream_concentration := (JoAnn_added_cream : ℚ) / JoAnn_total_volume,
      JoAnn_drank_cream := (JoAnn_drank_total : ℚ) * JoAnn_cream_concentration,
      JoAnn_remaining_cream := (JoAnn_added_cream : ℚ) - JoAnn_drank_cream,
      Joe_cream_amount := Joe_final_cream,
      JoAnn_cream_amount := JoAnn_remaining_cream
  in (Joe_cream_amount : ℚ) / JoAnn_cream_amount = 6 / 5 :=
by
  sorry

end cream_ratio_Joe_JoAnn_l351_351637


namespace find_value_of_expression_l351_351502

theorem find_value_of_expression
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 2 * y - 7 * z = 0)
  (hz : z ≠ 0) :
  (x^2 - 2 * x * y) / (y^2 + 4 * z^2) = -0.252 := 
sorry

end find_value_of_expression_l351_351502


namespace Marley_fruit_count_l351_351679

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l351_351679


namespace cos_330_eq_sqrt3_div_2_l351_351268

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351268


namespace compute_volume_of_rotated_solid_l351_351378

noncomputable def volume_of_solid_rotation {a b : ℝ} (f : ℝ → ℝ) : ℝ :=
  π * ∫ x in a..b, (f x)^2

theorem compute_volume_of_rotated_solid :
  volume_of_solid_rotation (λ x, Real.exp (1 - x)) 0 1 = (π * Real.exp 2 / 2) * (1 - Real.exp (-2)) :=
by
  sorry

end compute_volume_of_rotated_solid_l351_351378


namespace max_vehicles_and_quotient_l351_351692

-- Define the conditions
def speed_limit : ℕ := 80
def vehicle_length_m : ℕ := 5
def unit_length_front_to_front (n : ℕ) : ℕ := 5 * (n + 1)
def max_vehicle_lengths : ℕ := speed_limit / 10
def total_distance_m : ℕ := 80000

-- Define the statement
theorem max_vehicles_and_quotient :
  let max_vehicles := total_distance_m / unit_length_front_to_front max_vehicle_lengths in
  (max_vehicles = 1777) ∧ (max_vehicles / 10 = 177) :=
by
  sorry

end max_vehicles_and_quotient_l351_351692


namespace cos_330_eq_sqrt3_div_2_l351_351267

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l351_351267


namespace cos_330_eq_sqrt3_div_2_l351_351333

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351333


namespace integral_problem_1_integral_problem_2_integral_problem_3_integral_problem_4_l351_351989

-- The first integral
theorem integral_problem_1 (C : ℝ) :
  ∫ (λ x : ℝ, (3 * x^2 + 8) / (x^3 + 4 * x^2 + 4 * x)) = 
  (2 * Real.log (Real.abs x) + Real.log (Real.abs (x + 2)) - 10/(x + 2) + C) :=
sorry

-- The second integral
theorem integral_problem_2 (C : ℝ) :
  ∫ (λ x : ℝ, (2 * x^5 + 6 * x^3 + 1) / (x^4 + 3 * x^2)) = 
  (x^2 - 1/(3 * x) - 1/(3 * Real.sqrt 3) * Real.arctan (x / (Real.sqrt 3)) + C) :=
sorry

-- The third integral
theorem integral_problem_3 (C : ℝ) :
  ∫ (λ x : ℝ, (x^3 + 4 * x^2 - 2 * x + 1) / (x^4 + x)) = 
  (Real.log (Real.abs x) - 2 * Real.log (Real.abs (x + 1)) + Real.log (Real.abs (x^2 - x + 1)) + C) :=
sorry

-- The fourth integral
theorem integral_problem_4 (C : ℝ) :
  ∫ (λ x : ℝ, (x^3 - 3) / (x^4 + 10 * x^2 + 25)) = 
  (1/2 * Real.log (x^2 + 5) + (5 / 2) * (1 / ((x^2 + 5) + C)) :=
sorry

end integral_problem_1_integral_problem_2_integral_problem_3_integral_problem_4_l351_351989


namespace distance_between_ellipse_foci_l351_351914

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l351_351914


namespace cos_330_eq_sqrt_3_div_2_l351_351217

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l351_351217


namespace measure_angle_PSR_l351_351549

-- Definition of parallelogram
structure parallelogram (A B C D : Type) :=
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (angle_D : ℝ)
  (angle_sum_A_B : angle_A + angle_B = 180)
  (angle_sum_B_C : angle_B + angle_C = 180)

-- Given conditions
variables {P Q R S : Type}

def angle_PQR := 2 * angle_QRS

-- Proof statement to be verified
theorem measure_angle_PSR (h : parallelogram P Q R S)
  (h_cond : angle_PQR = 2 * angle_QRS) :
  angle_PSR = 120 :=
begin
  sorry
end

end measure_angle_PSR_l351_351549


namespace ram_krish_together_l351_351867

theorem ram_krish_together (K : ℕ → ℚ) (R : ℕ → ℚ) 
  (h1 : ∀ t, R t = 1/) (27:℀ฺq), 
  (h2 : ∀ t, K t = 2 × R t)  
:
by sorry end

end ram_krish_together_l351_351867


namespace cream_ratio_Joe_JoAnn_l351_351639

def Joe_initial_coffee := 15
def Joe_drank_coffee := 3
def Joe_added_cream := 3

def JoAnn_initial_coffee := 15
def JoAnn_added_cream := 3
def JoAnn_drank_total := 3

theorem cream_ratio_Joe_JoAnn :
  let Joe_final_cream := Joe_added_cream,
      JoAnn_total_volume := JoAnn_initial_coffee + JoAnn_added_cream,
      JoAnn_cream_concentration := (JoAnn_added_cream : ℚ) / JoAnn_total_volume,
      JoAnn_drank_cream := (JoAnn_drank_total : ℚ) * JoAnn_cream_concentration,
      JoAnn_remaining_cream := (JoAnn_added_cream : ℚ) - JoAnn_drank_cream,
      Joe_cream_amount := Joe_final_cream,
      JoAnn_cream_amount := JoAnn_remaining_cream
  in (Joe_cream_amount : ℚ) / JoAnn_cream_amount = 6 / 5 :=
by
  sorry

end cream_ratio_Joe_JoAnn_l351_351639


namespace distance_between_foci_of_hyperbola_l351_351730

theorem distance_between_foci_of_hyperbola :
  ∀ {x y : ℝ}, x^2 - y^2 = 1 → ∃ (c : ℝ), c = 2 * Real.sqrt 2 :=
by
  intros x y hyp
  use 2 * Real.sqrt 2
  sorry

end distance_between_foci_of_hyperbola_l351_351730


namespace sum_coordinates_A_l351_351599

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l351_351599


namespace valid_conclusions_in_space_l351_351856

variables (L₁ L₂ L₃ : Type) [Line L₁] [Line L₂] [Line L₃]

-- Condition 1: Two lines parallel to the same line are parallel.
def parallel_to_same_line (L₁ L₂ L₃ : Type) [Line L₁] [Line L₂] [Line L₃] : Prop :=
  (Parallel L₁ L₃) ∧ (Parallel L₂ L₃) → (Parallel L₁ L₂)

-- Condition 2: If a line is perpendicular to one of two parallel lines, it must also be perpendicular to the other.
def perpendicular_to_parallel_lines (L₁ L₂ L₃ : Type) [Line L₁] [Line L₂] [Line L₃] : Prop :=
  (Perpendicular L₁ L₂) ∧ (Parallel L₁ L₃) → (Perpendicular L₁ L₃)

theorem valid_conclusions_in_space : 
  (parallel_to_same_line L₁ L₂ L₃) ∧ (perpendicular_to_parallel_lines L₁ L₂ L₃) → 
  ((Parallel L₁ L₂) ∧ (Perpendicular L₁ L₃)) :=
sorry

end valid_conclusions_in_space_l351_351856


namespace mean_median_change_l351_351542

def initial_data : List ℕ := [24, 30, 19, 25, 19]

def corrected_data : List ℕ := [24, 31, 20, 25, 19]

noncomputable def mean (data : List ℕ) : ℚ :=
  data.sum / data.length

def median (data : List ℕ) : ℕ :=
  let sorted := data.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem mean_median_change :
  mean corrected_data - mean initial_data = 0.4 ∧
  median corrected_data = median initial_data :=
by
  sorry

end mean_median_change_l351_351542


namespace complement_intersect_l351_351648

def U : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def A : Set ℤ := {x | x^2 - 1 ≤ 0}
def B : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def C : Set ℤ := {x | x ∉ A ∧ x ∈ U} -- complement of A in U

theorem complement_intersect (U A B : Set ℤ) :
  (C ∩ B) = {2, 3} :=
by
  sorry

end complement_intersect_l351_351648


namespace cos_330_eq_sqrt3_div_2_l351_351189

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351189


namespace trains_clear_time_approx_l351_351088

noncomputable def time_to_clear (length1 length2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let speed1_meters_per_sec := speed1_kmh * (1000 / 3600)
  let speed2_meters_per_sec := speed2_kmh * (1000 / 3600)
  let relative_speed := speed1_meters_per_sec + speed2_meters_per_sec
  total_distance / relative_speed

theorem trains_clear_time_approx :
  time_to_clear 300 235 120 90 ≈ 3.06 :=
by
  -- We will provide the proof steps here.
  sorry

end trains_clear_time_approx_l351_351088


namespace cos_330_eq_sqrt3_div_2_l351_351193

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l351_351193


namespace cos_330_eq_sqrt3_div_2_l351_351342

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l351_351342


namespace cos_330_eq_sqrt3_div_2_l351_351245

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351245


namespace cos_330_eq_sqrt_3_div_2_l351_351376

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l351_351376


namespace gcd_75_100_l351_351814

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l351_351814


namespace ones_digit_19_power_l351_351424

theorem ones_digit_19_power (n : ℕ) (hn : n = 19 * (13 ^ 13)) : 
  (19^n % 10 = 9) := 
by 
  have h_one_digit_19_n_eq_9_n : ∀ n : ℕ, (19^n % 10 = 9^n % 10) := sorry
  have h_9_power_cycle : ∀ k : ℕ, (9^ (2 * k + 1) % 10 = 9) := sorry
  
  have h_odd_power : ∃ k : ℕ, n = 2 * k + 1 := 
  by 
    have h13_odd := nat.odd_pow (13 : ℕ) 13 (nat.odd_succ 6)
    have h := nat.odd_mul (nat.odd_succ 9) h13_odd
    exact nat.exists_sq h 
  
  have h_9_is_odd := h_9_power_cycle
  
  exact eq.trans (h_one_digit_19_n_eq_9_n n) (h_9_is_odd _)
  sorry

end ones_digit_19_power_l351_351424


namespace Maya_donut_holes_covered_l351_351943

noncomputable def radius_Niraek : ℝ := 5
noncomputable def radius_Theo : ℝ := 7
noncomputable def radius_Akshaj : ℝ := 11
noncomputable def radius_Maya : ℝ := 9

def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def lcm (a b c d : ℝ) : ℝ := Real.lcm (Real.lcm (Real.lcm a b) c) d

noncomputable def LCM : ℝ := lcm (surface_area radius_Niraek) (surface_area radius_Theo) 
                                (surface_area radius_Akshaj) (surface_area radius_Maya)

theorem Maya_donut_holes_covered : LCM / surface_area radius_Maya = 33075 :=
by 
  -- This is the statement problem; the proof part is omitted
  sorry

end Maya_donut_holes_covered_l351_351943


namespace cos_330_eq_sqrt3_div_2_l351_351253

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351253


namespace cos_330_eq_sqrt3_div_2_l351_351277

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l351_351277


namespace sum_of_coordinates_of_A_l351_351585

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351585


namespace domain_and_width_of_g_l351_351046

theorem domain_and_width_of_g
  (h : ℝ → ℝ)
  (dom_h : ∀ x, -12 ≤ x ∧ x ≤ 6 → ∃ y, h(y) = x)
  (g : ℝ → ℝ := λ x, h (2 * x / 3)) :
  (∀ x, -18 ≤ x ∧ x ≤ 9 → ∃ y, g(y) = x) ∧ (9 - (-18) = 27) :=
by
  sorry

end domain_and_width_of_g_l351_351046


namespace samia_walked_3_point_6_kilometers_l351_351705

noncomputable def samia_walking_distance (total_time : ℝ) (bike_speed : ℝ) (wait_time : ℝ) (walk_speed : ℝ) (x : ℝ) : ℝ :=
  (total_time = (x / bike_speed + wait_time + 0.5 * x)) → (2 * x = 3.6)

theorem samia_walked_3_point_6_kilometers (x : ℝ) : samia_walking_distance 1.25 20 0.25 4 x :=
  by
    sorry

end samia_walked_3_point_6_kilometers_l351_351705


namespace exists_six_consecutive_lcm_l351_351396

theorem exists_six_consecutive_lcm :
  ∃ n : ℕ, Nat.lcm (n) (n+1) (n+2) > Nat.lcm (n+3) (n+4) (n+5) := by
  sorry

end exists_six_consecutive_lcm_l351_351396


namespace find_function_l351_351981

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (x : ℝ) (h1 : x = 3) (h2 : x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  -- Given x = 3 and x + 17 = 60 * f x
  rw h1 at h2
  -- Showing the target f 3
  have h3 : 3 + 17 = 20 := by norm_num
  rw h3 at h2
  have h4 : 60 * f 3 = 20 := by rw h2
  -- Solving explicitly for f 3
  sorry

end find_function_l351_351981


namespace quadrilateral_perimeter_l351_351138

noncomputable def perimeter_remaining_quadrilateral (DB EB : ℝ) (DB_square_plus_EB_square_eq : DB^2 + EB^2 = 5) 
  (AC_length : ℝ = 5) (CE_and_DA_length : ℝ = 3) : ℝ :=
  AC_length + CE_and_DA_length + sqrt 5 + CE_and_DA_length

theorem quadrilateral_perimeter (DB EB : ℝ) (H1 : DB = 1) (H2 : EB = 2) 
  (H3 : DB^2 + EB^2 = 5) (H4 : ∀ AC CE DA : ℝ, AC = 5 ∧ CE = 3 ∧ DA = 3) : perimeter_remaining_quadrilateral DB EB H3 5 3 = 11 + sqrt 5 :=
by sorry

end quadrilateral_perimeter_l351_351138


namespace smallest_positive_period_of_f_fx_ge_neg_half_l351_351454

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f(x + T) = f(x) :=
begin
  use Real.pi,
  sorry
end

theorem fx_ge_neg_half (x : ℝ) :
  0 ≤ x ∧ x ≤ Real.pi / 2 → f(x) ≥ -1 / 2 :=
begin
  sorry
end

end smallest_positive_period_of_f_fx_ge_neg_half_l351_351454


namespace ratio_of_circumscribed_areas_l351_351889

theorem ratio_of_circumscribed_areas (P : ℝ) (A B : ℝ) 
  (hA : A = π * (P / 6) ^ 2) 
  (hB : B = π * (P * real.sqrt 2 / 8) ^ 2) : 
  A / B = 8 / 9 :=
by {
  -- Definitions from the conditions and solution are combined to form the necessary relationships
  rw [hA, hB],
  field_simp [π, real.square, div_eq_inv_mul, mul_assoc],
  norm_num,
  }

end ratio_of_circumscribed_areas_l351_351889


namespace axis_of_symmetry_of_graph_l351_351506

variable {R : Type*} [LinearOrderedField R]

def symmetric_function (g : R → R) (c : R) := 
  ∀ x : R, g x = g (c - x)

theorem axis_of_symmetry_of_graph (g : ℝ → ℝ) (h : symmetric_function g 3) : 
  ∃ c : ℝ, c = 1.5 ∧ ∀ x : ℝ, g x = g (2 * c - x) := 
begin
  use 1.5,
  split,
  {
    refl,
  },
  {
    intro x,
    specialize h x,
    exact h,
  }
end

end axis_of_symmetry_of_graph_l351_351506


namespace range_of_x_l351_351118

open Function

variable {α : Type*}
variable {β : Type*} [OrderedAddCommGroup β] [DecidableLinearOrder β]

-- Definitions based on the conditions
def even_function (f : α → β) : Prop := ∀ x, f x = f (-x)
def monotone_decreasing_on_nonneg (f : α → β) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

-- The statement to prove
theorem range_of_x 
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_mono : monotone_decreasing_on_nonneg f)
  (hf_val : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) : 
  ∃ a b : ℝ, a = -1 ∧ b = 3 :=
by
  -- proof goes here
  sorry

end range_of_x_l351_351118


namespace points_on_parabola_count_l351_351993

noncomputable def is_natural_point_on_parabola (x y : ℕ) : Prop :=
  y = (- x^2 / 4 : ℚ) + (3 * x : ℚ) + (253 / 4 : ℚ)

theorem points_on_parabola_count : 
  (∀ (x y : ℕ), is_natural_point_on_parabola x y → (1 ≤ x ∧ x ≤ 22 ∧ 
    (y = (- (x : ℚ)^2 / 4) + (3 * (x : ℚ)) + (253 / 4): ℚ) )) → 
  ∃ n : ℕ, n = 11 := 
begin
  sorry
end

end points_on_parabola_count_l351_351993


namespace max_number_of_liars_l351_351530

noncomputable def max_liars (grid : Fin 5 × Fin 5 → Prop) : ℕ :=
  ∑ i in Finset.univ, if grid i = liar then 1 else 0

theorem max_number_of_liars : ∃ grid : (Fin 5 × Fin 5 → Prop), 
  (∀ i, grid i = liar → (∃ j, j ≠ i ∧ adjacent i j ∧ grid j ≠ liar)) ∧ 
  max_liars grid = 13 :=
begin
  sorry
end

-- A function to describe adjacency condition in 5×5 grid
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ a.2 = b.2) ∨ (abs (a.2 - b.2) = 1 ∧ a.1 = b.1)

-- Define liar and knight
inductive Person
| liar : Person
| knight : Person

open Person

end max_number_of_liars_l351_351530


namespace cost_per_top_l351_351937
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l351_351937


namespace calculation_result_l351_351170

theorem calculation_result : 7 * (9 + 2 / 5) + 3 = 68.8 :=
by
  sorry

end calculation_result_l351_351170


namespace area_intersection_triangle_lt_face_l351_351700

theorem area_intersection_triangle_lt_face (A B C D M : Point) 
  (H1: is_tetrahedron A B C D) 
  (H2: M ∈ line_segment C D) 
  (H3: plane_passes_through AB M) 
  : 
  let face_areas := [area (triangle A B C), area (triangle A B D)]
  in area (triangle_intersection_plane A B M) < max face_areas :=
sorry

end area_intersection_triangle_lt_face_l351_351700


namespace expression_evaluation_l351_351171

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end expression_evaluation_l351_351171


namespace cos_330_is_sqrt3_over_2_l351_351297

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l351_351297


namespace solve_equation_l351_351717

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l351_351717


namespace cos_330_eq_sqrt3_over_2_l351_351205

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l351_351205


namespace solve_positive_integer_equation_l351_351042

theorem solve_positive_integer_equation :
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ y^3 = x^3 + 8 * x^2 - 6 * x + 8 → x = 9 ∧ y = 11 := 
by 
  intros x y h,
  sorry

end solve_positive_integer_equation_l351_351042


namespace possible_values_n_l351_351624

theorem possible_values_n (n : ℕ) (h1 : ∠A > ∠B ∧ ∠B > ∠C) (h2 : AB = 3 * n + 18) (h3 : AC = 3 * n + 10) (h4 : BC = 4 * n - 4) : 
  ∃ (n : ℕ), 4 ≤ n ∧ n ≤ 13 := sorry

end possible_values_n_l351_351624


namespace cube_vertices_even_impossible_l351_351946

theorem cube_vertices_even_impossible : 
  (∃ f : fin 8 → ℤ, f 0 = 2021 ∧ (∀ i, i ≠ 0 → f i = 0) ∧ 
  (∀ f', (∀ v1 v2, v1 ≠ v2 → f' v1 = f v1 + 1 ∧ f' v2 = f v2 + 1 ∨ ∃ u, f' u = f u) → 
  ∀ i, f' i % 2 = 0)) → false := 
sorry

end cube_vertices_even_impossible_l351_351946


namespace solution_correct_l351_351043

noncomputable def a := 3 + 3 * Real.sqrt 2
noncomputable def b := 3 - 3 * Real.sqrt 2

theorem solution_correct (h : a ≥ b) : 3 * a + 2 * b = 15 + 3 * Real.sqrt 2 :=
by sorry

end solution_correct_l351_351043


namespace gcd_75_100_l351_351841

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l351_351841


namespace night_crew_load_fraction_l351_351167

theorem night_crew_load_fraction (D N B : ℝ) 
  (h1: N = 3/4 * D) 
  (h2: B > 0)
  (h3: D > 0)
  (h4: N > 0) 
  (h5: D ≠ 0)  
  (h6: 0 < 0.64 ∧ 0.64 < 1)
  (h7: 0 < 0.36 ∧ 0.36 < 1)
  (day_load_fraction: 0.64 * B)
  (night_load_fraction : 0.36 * B): 
  ((0.36 * B) / (3/4 * D)) / ((0.64 * B) / D) = 3/4 :=
by
  sorry

end night_crew_load_fraction_l351_351167


namespace enrique_commission_l351_351400

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l351_351400


namespace complex_multiplication_example_l351_351412

theorem complex_multiplication_example : (3 - 4 * Complex.i) * (-2 + 6 * Complex.i) = -30 + 26 * Complex.i := by
  sorry

end complex_multiplication_example_l351_351412


namespace alice_has_winning_strategy_l351_351092

def circular_table : Type :=
  unit -- Placeholder for circular table representation

def is_valid_move (table : circular_table) (pos : ℝ × ℝ) : Prop :=
  -- Placeholder for the validity of the coin placement (within the circle boundaries)
  sorry

def symmetric_pos (pos : ℝ × ℝ) : ℝ × ℝ :=
  (-pos.1, -pos.2) -- Reflecting position through the center of the table

def alice_winning_strategy (table : circular_table) (moves_alice bob_moves : list (ℝ × ℝ)) : Prop :=
  -- Formalize Alice's strategy and show it's always winning if proper turns are followed
  sorry

theorem alice_has_winning_strategy : ∀ (table : circular_table),
  (∃ (moves_alice bob_moves : list (ℝ × ℝ)),
   alice_winning_strategy table moves_alice bob_moves) :=
by
  -- Proof of Alice's winning strategy
  sorry

end alice_has_winning_strategy_l351_351092


namespace sum_of_coordinates_of_A_l351_351592

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l351_351592


namespace total_pencils_l351_351865

theorem total_pencils (pencils_per_child : ℕ) (number_of_children : ℕ) 
(h1 : pencils_per_child = 2) 
(h2 : number_of_children = 15) : 
(pencils_per_child * number_of_children = 30) := 
by
  rw [h1, h2]
  norm_num

end total_pencils_l351_351865


namespace brooke_total_jumping_jacks_l351_351036

def sj1 : Nat := 20
def sj2 : Nat := 36
def sj3 : Nat := 40
def sj4 : Nat := 50
def Brooke_jumping_jacks : Nat := 3 * (sj1 + sj2 + sj3 + sj4)

theorem brooke_total_jumping_jacks : Brooke_jumping_jacks = 438 := by
  sorry

end brooke_total_jumping_jacks_l351_351036


namespace find_S_5_l351_351438

variable {a : ℕ → ℝ} -- a_n is represented as a function from natural numbers to reals
variable {S : ℕ → ℝ} -- S_n is also a function from natural numbers to reals

-- Define an arithmetic sequence:
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of a sequence:
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ k in Finset.range n, a (k + 1)

-- Given conditions:
def conditions (a S : ℕ → ℝ) : Prop :=
is_arithmetic_sequence a ∧ S 2 = sum_first_n_terms a 2 ∧ S 2 = 3 ∧ S 3 = sum_first_n_terms a 3 ∧ S 3 = 3

-- Prove S_5 = 0
theorem find_S_5 (a S : ℕ → ℝ) (h : conditions a S) : S 5 = 0 := by
  sorry

end find_S_5_l351_351438


namespace find_A_coordinates_sum_l351_351584

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l351_351584


namespace magnitude_of_complex_number_l351_351411

noncomputable def complex_magnitude (c : ℂ) : ℝ :=
  complex.abs c

theorem magnitude_of_complex_number :
  complex_magnitude (7/4 - 3*complex.I) = real.sqrt 193 / 4 := 
by 
  sorry

end magnitude_of_complex_number_l351_351411


namespace find_f_2023_l351_351933

-- Definitions based on conditions
def f (a : ℝ) (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then a * x^3 + 2 * x + a + 1 else 0

-- Given conditions
axiom odd_function (a : ℝ) : ∀ x : ℝ, f a (-x) = -f a x
axiom f_two_minus_x (a : ℝ) : ∀ x : ℝ, f a x = f a (2 - x)

-- Proof problem statement
theorem find_f_2023 (a : ℝ) (ha : a = -1) : f a 2023 = -1 := by
  sorry

end find_f_2023_l351_351933


namespace remaining_solid_edges_l351_351894

/-- A solid cube of side length 3 has eight smaller cubes of side length 1 removed from its corners.
    Prove that the number of edges of the resulting solid is equal to 84. -/
theorem remaining_solid_edges : 
  let original_cube_edges := 12
  let small_cube_edges := 12
  let number_of_corners := 8
  let removed_cube_edges_contribute := 72 -- (8 * 9 since it was recalculated)
  let final_correction := 8 -- since each edge originally at a corner is shared by three cubes
  let total_edges := original_cube_edges + (removed_cube_edges_contribute - final_correction) 
  in total_edges = 84 :=
by {
  sorry
}

end remaining_solid_edges_l351_351894


namespace probability_at_least_one_boy_and_one_girl_l351_351165

theorem probability_at_least_one_boy_and_one_girl 
  (prob_boy prob_girl : ℝ)
  (h_eq_prob : prob_boy = 1/2 ∧ prob_girl = 1/2)
  (n_children : ℕ)
  (h_n_children : n_children = 4) :
  let prob_all_boys := prob_boy ^ n_children in
  let prob_all_girls := prob_girl ^ n_children in
  (1 - (prob_all_boys + prob_all_girls) = 7 / 8) :=
by
  let prob_all_boys := prob_boy ^ n_children
  let prob_all_girls := prob_girl ^ n_children
  sorry

end probability_at_least_one_boy_and_one_girl_l351_351165
