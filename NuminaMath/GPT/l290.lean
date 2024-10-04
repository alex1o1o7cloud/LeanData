import Mathlib

namespace find_a_value_l290_290016

open Real BigOperators

noncomputable def constant_term_value (a : ℝ) : ℝ :=
  let r := 2
  let C_8_2 := Nat.choose 8 r
  (-a)^r * C_8_2

theorem find_a_value (a : ℝ) (h : constant_term_value a = 14) : a = (Math.sqrt 2) / 2 ∨ a = -(Math.sqrt 2) / 2 :=
by
  -- Proof omitted
  sorry

end find_a_value_l290_290016


namespace remainder_of_product_mod_5_l290_290282

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l290_290282


namespace annual_interest_rate_approx_l290_290546

-- Definitions of the variables
def FV : ℝ := 1764    -- Face value of the bill
def TD : ℝ := 189     -- True discount
def PV : ℝ := FV - TD -- Present value, calculated as per the problem statement

-- Simple interest formula components
def P : ℝ := PV       -- Principal
def T : ℝ := 9 / 12   -- Time period in years

-- Given conditions as definitions:
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Statement to prove that the annual interest rate R equals 16%
theorem annual_interest_rate_approx : ∃ R : ℝ, simple_interest P R T = TD ∧ R ≈ 16 := by
  use 16
  sorry

end annual_interest_rate_approx_l290_290546


namespace find_m_value_l290_290297

theorem find_m_value (x : ℝ) (hx1 : ∃ n : ℤ, x ≠ n * 3 * π) (hx2 : ∃ n : ℤ, x ≠ n * π / 2) :
  (real.cot (x / 3) - real.cot (2 * x)) = (real.sin (5 * x / 3) / (real.sin (x / 3) * real.sin (2 * x))) :=
by sorry

end find_m_value_l290_290297


namespace max_AF2_BF2_area_triangle_ABF2_l290_290328

-- Definitions for the given conditions
def ellipse (a b : ℝ) (h : (a > b) ∧ (b > 0)) : set (ℝ × ℝ) :=
  {p | (p.1)^2 / a^2 + (p.2)^2 / b^2 = 1}

def foci (a b : ℝ) : ℝ × ℝ :=
  let c := real.sqrt (a^2 - b^2) in (c, 0)

def M_and_N (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (0, -b, 0, b)  -- Assuming M = (0, -b) and N = (0, b)

def minor_axis_quadrilateral (a b : ℝ) : Prop :=
  let (M1, M2, N1, N2) := M_and_N a b in
  2 * b = 4 / 2  -- Perimeter condition simplified to a = 1

-- Conditions
variables (a b : ℝ) (h : (a > b) ∧ (b > 0)) (M N : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (l : set (ℝ × ℝ)) (A B : ℝ × ℝ)
  (h_foci : F1 = foci a b ∧ F2 = (-foci a b).1)
  (h_MN : (M = (0, -b)) ∧ (N = (0, b)))
  (h_perimeter : minor_axis_quadrilateral a b)
  (h_line : ∃ x, ∀ p, p ∈ l ↔ p = (F1.1 + x * F1.2, F1.2 + x * (F2.1 - F1.1)))
  (h_intersect : A ∈ ellipse a b h ∧ B ∈ ellipse a b h ∧ |A.1 - B.1| = 4 / 3)

-- Question 1
theorem max_AF2_BF2 (h_AF2_BF2 : ∃ x y, x = real.sqrt ((4 - (4 / 3))^2 / 4) ∧ y = 4 - (4 / 3) - x) :
  (|A.1 - B.1| = (4 / 3) →  (x * y ≤ (4 / 3) * (2 / 3)) :=
sorry

-- Conditions for Question 2
variable (slope_45 : l = {(x, y) | y = x + (real.sqrt 1 / b^2)})

-- Question 2
theorem area_triangle_ABF2 : 
  (slope_45 → real.abs ((A.2 - B.2) / (A.1 - B.1) - 1) = real.sqrt 2 / 2):
  ∃ S, 
    S = (1 / 2) * (4 / 3) * 1 →
    S = (2 / 3)
:=
sorry

end max_AF2_BF2_area_triangle_ABF2_l290_290328


namespace number_of_fish_l290_290430

theorem number_of_fish (initial_fish : ℕ) (double_day : ℕ → ℕ → ℕ) (remove_fish : ℕ → ℕ → ℕ) (add_fish : ℕ → ℕ → ℕ) :
  (initial_fish = 6) →
  (∀ n m, double_day n m = n * 2) →
  (∀ n d m, d = 3 ∨ d = 5 → remove_fish n d = n - n / m) →
  (∀ n d, d = 7 → add_fish n d = n + 15) →
  (double_day 6 1 = 12) →
  (double_day 12 2 = 24) →
  (remove_fish 24 3 = 16) →
  (double_day 16 4 = 32) →
  (double_day 32 5 = 64) →
  (remove_fish 64 5 = 48) →
  (double_day 48 6 = 96) →
  (double_day 96 7 = 192) →
  (add_fish 192 7 = 207) →
  207 = 207 :=
begin
  intros,
  -- Proof omitted since it's not required
  sorry,
end

end number_of_fish_l290_290430


namespace new_quadratic_equation_l290_290975

-- Given: The roots of the equation x^2 + p * x + q are x1 and x2
variables {x1 x2 p q : ℝ}

-- Definition indicating that x1 and x2 are roots of the given quadratic equation
def original_roots : Prop :=
  (x1^2 + p * x1 + q = 0) ∧ (x2^2 + p * x2 + q = 0)

-- We need to prove the new quadratic equation with roots (x1 + 1) and (x2 + 1) is x^2 + (p - 2) * x + (q - p + 1) = 0
theorem new_quadratic_equation (h : original_roots) : 
  ∃ r s : ℝ, (r = p - 2) ∧ (s = q - p + 1) ∧ ∀ x : ℝ, 
    (x - (x1 + 1)) * (x - (x2 + 1)) = x^2 + r * x + s :=
by
  sorry

end new_quadratic_equation_l290_290975


namespace man_work_rate_l290_290664

theorem man_work_rate (W : ℝ) (M S : ℝ)
  (h1 : (M + S) * 3 = W)
  (h2 : S * 5.25 = W) :
  M * 7 = W :=
by 
-- The proof steps will be filled in here.
sorry

end man_work_rate_l290_290664


namespace alpha_eq_beta_plus2_l290_290262

-- Definitions
def compositions_with_1s_and_2s (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else compositions_with_1s_and_2s (n - 1) + compositions_with_1s_and_2s (n - 2)

def compositions_with_greater_than_1s (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 1
  else compositions_with_greater_than_1s (n - 1) + compositions_with_greater_than_1s (n - 2)

-- Statement
theorem alpha_eq_beta_plus2 (n : ℕ) : compositions_with_1s_and_2s n = compositions_with_greater_than_1s (n + 2) := by
  sorry

end alpha_eq_beta_plus2_l290_290262


namespace inconsistent_money_l290_290228

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 300

theorem inconsistent_money : A + B + C = 250 → false :=
by
  intro h
  have h1 : C = 300 := rfl
  linarith

end inconsistent_money_l290_290228


namespace rectangular_paper_width_l290_290696

def wall_photo_width := 2
def paper_length := 8
def wall_photo_area := 96

theorem rectangular_paper_width : ∃ w : ℝ, (w + 4) * 12 = wall_photo_area ∧ w = 4 :=
by
  existsi (4 : ℝ)
  split
  . simp [wall_photo_area]
  . rfl

-- The theorem proves the width of the rectangular paper is 4 inches.

end rectangular_paper_width_l290_290696


namespace overtime_hours_l290_290583

-- Definitions of given conditions
def regularPay (rp hr : ℝ) := rp * hr
def overtimeRate (rp : ℝ) := 2 * rp
def overtimePay (tp regularPay : ℝ) := tp - regularPay

-- The theorem we aim to prove
theorem overtime_hours 
  (rp tp : ℝ)
  (hr : ℕ) 
  (regularPay : ℝ)
  (or : ℝ) 
  (otPay : ℝ) 
  (otHours : ℝ) :
  rp = 3 →
  tp = 180 →
  hr = 40 →
  regularPay = regularPay rp hr →
  or = overtimeRate rp →
  otPay = overtimePay tp regularPay →
  otHours = otPay / or →
  otHours = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end overtime_hours_l290_290583


namespace probability_of_v_w_l290_290057

open Complex Real

noncomputable def probability_satisfying_condition : ℝ :=
  let n := 2023
  let theta (k : ℕ) := 2 * π * k / n
  let roots : Fin n.succ → ℂ := λ k, Complex.exp (I * theta k)
  let val (k : ℕ) :=  2 + 2 * Real.cos (theta k)
  let condition (k : ℕ) : Bool := 4 + Real.sqrt 5 ≤ val k
  let count := Fintype.card {k // k < n ∧ condition k}
  count / n

theorem probability_of_v_w : probability_satisfying_condition = 505.5 / 2022 :=
  sorry

end probability_of_v_w_l290_290057


namespace center_of_inscribed_circle_on_midpoint_line_l290_290499
-- Definitions
variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- The given conditions for the tangential quadrilateral
def tangential_quadrilateral (Q : Quadrilateral ABCD) :=
  inscribed_circle Q ∧ pitot_theorem Q

-- The midpoint function
def midpoint (x y : Type) [metric_space x] [metric_space y] : Type := sorry

-- Lean statement of the problem
theorem center_of_inscribed_circle_on_midpoint_line 
  (ABCD : Quadrilateral ABCD) (O : Point)
  (M : Point) (N : Point)
  (h₁ : tangential_quadrilateral ABCD)
  (hM : M = midpoint A C)
  (hN : N = midpoint B D)
  (h2 : satisfies_pitot_theorem ABCD) :
  collinear {O, M, N} :=
sorry

end center_of_inscribed_circle_on_midpoint_line_l290_290499


namespace total_fruits_l290_290139

theorem total_fruits (cucumbers : ℕ) (watermelons : ℕ) 
  (h1 : cucumbers = 18) 
  (h2 : watermelons = cucumbers + 8) : 
  cucumbers + watermelons = 44 := 
by {
  sorry
}

end total_fruits_l290_290139


namespace parabola_directrix_l290_290272

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end parabola_directrix_l290_290272


namespace children_count_l290_290597

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290597


namespace children_count_l290_290596

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290596


namespace compute_DF_l290_290133

-- Define points A, B, and C in a triangle such that AB = 4, BC = 5, and AC = 6.
variable {A B C : Type}
variable [metric_space A] [metric_space B] [metric_space C]
variable (AB AC BC : ℝ)

-- Define the lengths of the sides of the triangle.
def triangle_sides : Prop := (AB = 4) ∧ (BC = 5) ∧ (AC = 6)

-- Define D as the intersection of the angle bisector of ∠A with side BC.
variable {D E F : Type}
def angle_bisector_D : Prop := ∃ (BD DC : ℝ), BC = BD + DC ∧ BD / DC = AB / AC

-- Define E as the foot of the perpendicular from B to the angle bisector of ∠A.
def foot_perpendicular_E : Prop := ∃ (BE : ℝ), ⊥B⊥D

-- Define F as the intersection of the line through E parallel to AC with side BC.
def line_through_E_parallel_AC : Prop := ∃ (EF : ℝ), EF ∥ AC

-- Prove that DF = 0.5 given the above conditions.
theorem compute_DF (AB AC BC : ℝ) (H1 : triangle_sides AB BC AC) 
  (H2 : angle_bisector_D BC AB AC) 
  (H3 : foot_perpendicular_E) 
  (H4 : line_through_E_parallel_AC) : DF = 0.5 :=
begin
  sorry
end

end compute_DF_l290_290133


namespace surface_area_of_box_l290_290672

theorem surface_area_of_box (a b c : ℝ) (h1 : a + b + c = 40) (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 975 := 
by
  have H : (a + b + c)^2 = 1600 := by linarith
  rw [h1] at H
  have : 1600 = a^2 + b^2 + c^2 + 2 * (a * b + b * c + c * a) := by
    rw [← add_assoc, pow_two, add_sq, add_sq, add_sq]
  rw [h2] at this
  linarith

end surface_area_of_box_l290_290672


namespace find_x_l290_290419

-- Definitions based on the problem conditions
def angle_CDE : ℝ := 90 -- angle CDE in degrees
def angle_ECB : ℝ := 68 -- angle ECB in degrees

-- Theorem statement
theorem find_x (x : ℝ) 
  (h1 : angle_CDE = 90) 
  (h2 : angle_ECB = 68) 
  (h3 : angle_CDE + x + angle_ECB = 180) : 
  x = 22 := 
by
  sorry

end find_x_l290_290419


namespace find_original_price_l290_290943

-- Define the conditions provided in the problem
def original_price (P : ℝ) : Prop :=
  let first_discount := 0.90 * P
  let second_discount := 0.85 * first_discount
  let taxed_price := 1.08 * second_discount
  taxed_price = 450

-- State and prove the main theorem
theorem find_original_price (P : ℝ) (h : original_price P) : P = 544.59 :=
  sorry

end find_original_price_l290_290943


namespace unique_plane_l290_290834

variable {α : Type*}
variable [EuclideanGeometry α]

-- Definition that the three points A, B, and C are not collinear.
def non_collinear (A B C : α) : Prop :=
  ¬ (∃ (ℓ : Line α), A ∈ ℓ ∧ B ∈ ℓ ∧ C ∈ ℓ)

-- Definition of planes through points
def unique_plane_through_points (A B C : α) : Prop :=
  ∃! (π : Plane α), A ∈ π ∧ B ∈ π ∧ C ∈ π 

-- Problem statement
theorem unique_plane (A B C : α) (h : non_collinear A B C) : unique_plane_through_points A B C :=
sorry

end unique_plane_l290_290834


namespace parabola_directrix_l290_290210

theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (O D : ℝ × ℝ) :
  A ≠ B →
  O = (0, 0) →
  D = (1, 2) →
  (∃ k, k = ((2:ℝ) - 0) / ((1:ℝ) - 0) ∧ k = 2) →
  (∃ k, k = - 1 / 2) →
  (∀ x y, y^2 = 2 * p * x) →
  p = 5 / 2 →
  O.1 * A.1 + O.2 * A.2 = 0 →
  O.1 * B.1 + O.2 * B.2 = 0 →
  A.1 * B.1 + A.2 * B.2 = 0 →
  (∃ k, (y - 2) = k * (x - 1) ∧ (A.1 * B.1) = 25 ∧ (A.1 + B.1) = 10 + 8 * p) →
  ∃ dir_eq, dir_eq = -5 / 4 :=
by
  sorry

end parabola_directrix_l290_290210


namespace total_towels_folded_in_one_hour_l290_290037

-- Define the conditions for folding rates and breaks of each person
def Jane_folding_rate (minutes : ℕ) : ℕ :=
  if minutes % 8 < 5 then 5 * (minutes / 8 + 1) else 5 * (minutes / 8)

def Kyla_folding_rate (minutes : ℕ) : ℕ :=
  if minutes < 30 then 12 * (minutes / 10 + 1) else 36 + 6 * ((minutes - 30) / 10)

def Anthony_folding_rate (minutes : ℕ) : ℕ :=
  if minutes <= 40 then 14 * (minutes / 20)
  else if minutes <= 50 then 28
  else 28 + 14 * ((minutes - 50) / 20)

def David_folding_rate (minutes : ℕ) : ℕ :=
  let sets := minutes / 15
  let additional := sets / 3
  4 * (sets - additional) + 5 * additional

-- Definitions are months passing given in the questions
def hours_fold_towels (minutes : ℕ) : ℕ :=
  Jane_folding_rate minutes + Kyla_folding_rate minutes + Anthony_folding_rate minutes + David_folding_rate minutes

theorem total_towels_folded_in_one_hour : hours_fold_towels 60 = 134 := sorry

end total_towels_folded_in_one_hour_l290_290037


namespace ramanujan_number_l290_290850

open Complex

theorem ramanujan_number (r h : ℂ) (h_eq : h = 3 + 4 * I )
  (product_eq : r * h = 24 - 10 * I) : 
  r = (112 / 25) - (126 / 25) * I :=
by 
  sorry

end ramanujan_number_l290_290850


namespace incorrect_major_premise_l290_290946

theorem incorrect_major_premise : 
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), (f' x₀ = 0) → ((∀ x, x > x₀ → f' x ≥ 0) ∧ (∀ x, x < x₀ → f' x ≤ 0)) → (is_extremum f x₀)) := 
sorry

end incorrect_major_premise_l290_290946


namespace circle_standard_equation_l290_290204

-- Define the conditions
def point_on_line_y_eq_neg_x (a : ℝ) := (a, -a)
def tangent_point := (-1 : ℝ, -1 : ℝ)
def line_y_eq_1_plus_2x (x : ℝ) := 1 + 2 * x

-- Define the function for the distance from a point to a line
def distance_from_point_to_line (p : ℝ × ℝ) (A B C : ℝ) :=
  abs (A * p.1 + B * p.2 + C) / sqrt (A^2 + B^2)

-- Define the function for distance between points
def distance_between_points (p1 p2 : ℝ × ℝ) :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The standard equation of a circle with center (h, k) and radius r
def circle_equation (h k r : ℝ) (x y : ℝ) := (x - h)^2 + (y - k)^2 = r^2

theorem circle_standard_equation :
  ∃ h k r,
  (h, k) = (3, -3) ∧
  r = sqrt 20 ∧
  ∀ x y, circle_equation h k r x y = (x - 3)^2 + (y + 3)^2 = 20 :=
begin
  existsi (3, -3),
  existsi sqrt 20,
  split,
  exact rfl, -- (h, k) = (3, -3)
  split,
  exact rfl, -- r = sqrt 20
  intros x y,
  exact rfl, -- circle_equation 3 -3 (sqrt 20) x y = (x - 3)^2 + (y + 3)^2 = 20
end

end circle_standard_equation_l290_290204


namespace min_value_cos2_sin_l290_290530

theorem min_value_cos2_sin (x : ℝ) : 
  cos x ^ 2 - 2 * sin x ≥ -2 :=
by
  sorry

end min_value_cos2_sin_l290_290530


namespace grasshopper_frog_jump_difference_l290_290524

theorem grasshopper_frog_jump_difference :
  let grasshopper_jump := 19
  let frog_jump := 15
  grasshopper_jump - frog_jump = 4 :=
by
  let grasshopper_jump := 19
  let frog_jump := 15
  sorry

end grasshopper_frog_jump_difference_l290_290524


namespace sum_of_solutions_eq_zero_l290_290164

theorem sum_of_solutions_eq_zero : 
  let f : ℝ → ℝ := λ x, (6 * x / 24) - (8 / x)
  ∀ x : ℝ, f x = 0 → x = (4 * Real.sqrt 2) ∨ x = -(4 * Real.sqrt 2) ∧ (4 * Real.sqrt 2) + (-(4 * Real.sqrt 2)) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l290_290164


namespace mary_flour_requirement_l290_290481

theorem mary_flour_requirement (total_flour : ℕ) (added_flour : ℕ) (remaining_flour : ℕ) 
  (h1 : total_flour = 7) 
  (h2 : added_flour = 2) 
  (h3 : remaining_flour = total_flour - added_flour) : 
  remaining_flour = 5 :=
sorry

end mary_flour_requirement_l290_290481


namespace pam_total_apples_l290_290494

theorem pam_total_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) (pam_apples_per_bag : ℕ) (gerald_bags_for_pam_bag : ℕ) :
  pam_bags = 10 →
  gerald_apples_per_bag = 40 →
  gerald_bags_for_pam_bag = 3 →
  pam_apples_per_bag = gerald_bags_for_pam_bag * gerald_apples_per_bag →
  pam_bags * pam_apples_per_bag = 1200 := 
by
  intros h_pam_bags h_gerald_apples h_gerald_bags_for_pam h_pam_apples
  rw [h_pam_bags, h_gerald_apples, h_gerald_bags_for_pam, h_pam_apples]
  calc
    10 * (3 * 40) = 10 * 120 : by rfl
               ... = 1200 : by rfl
  done

end pam_total_apples_l290_290494


namespace non_chihuahua_males_l290_290136

theorem non_chihuahua_males (total_dogs : ℕ) (male_fraction chihuahua_fraction : ℚ)
  (h1 : total_dogs = 32)
  (h2 : male_fraction = 5 / 8)
  (h3 : chihuahua_fraction = 3 / 4) :
  (let male_dogs := male_fraction * total_dogs;
       chihuahua_males := chihuahua_fraction * male_dogs
  in male_dogs - chihuahua_males) = 5 :=
by
  -- proof can go here
  sorry

end non_chihuahua_males_l290_290136


namespace non_overlapping_lines_in_same_plane_l290_290967

-- Define the problem conditions and the question
def lines_relationship (l1 l2 : Line) (h1 : same_plane l1 l2) (h2 : non_overlapping l1 l2) : Prop :=
  ∃ r : Relationship, r = Relationship.Parallel ∨ r = Relationship.Intersecting

theorem non_overlapping_lines_in_same_plane {l1 l2 : Line} (h1 : same_plane l1 l2) (h2 : non_overlapping l1 l2) :
  lines_relationship l1 l2 h1 h2 :=
sorry

end non_overlapping_lines_in_same_plane_l290_290967


namespace least_natural_number_to_create_palindrome_l290_290159

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem least_natural_number_to_create_palindrome :
  ∃ n : ℕ, n + 52351 = 53335 ∧ is_palindrome (n + 52351) :=
by 
  use 984
  split
  -- Proof that 984 + 52351 = 53335
  sorry,
  -- Proof that 53335 is a palindrome
  sorry

end least_natural_number_to_create_palindrome_l290_290159


namespace find_added_number_l290_290551

def S₁₅ := 15 * 17
def S₁₆ := 16 * 20
def added_number := S₁₆ - S₁₅

theorem find_added_number : added_number = 65 :=
by
  sorry

end find_added_number_l290_290551


namespace royal_children_l290_290620

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290620


namespace square_root_condition_l290_290840

-- Define the condition
def meaningful_square_root (x : ℝ) : Prop :=
  x - 5 ≥ 0

-- Define the theorem that x must be greater than or equal to 5 for the square root to be meaningful
theorem square_root_condition (x : ℝ) : meaningful_square_root x ↔ x ≥ 5 := by
  sorry

end square_root_condition_l290_290840


namespace domain_g_l290_290394

noncomputable def f (x : ℝ) : ℝ := sorry

def g (x : ℝ) : ℝ := f (x + 1) / (x - 2)

theorem domain_g :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x ≠ 0) →
  {x : ℝ | ∃ y, g y = x} = set.Icc (-1 : ℝ) 2 := 
by {
  intro h,
  -- Proof omitted
  sorry
}

end domain_g_l290_290394


namespace find_annual_interest_rate_l290_290544

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end find_annual_interest_rate_l290_290544


namespace train_length_is_correct_l290_290684

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l290_290684


namespace log_equation_solution_l290_290130

theorem log_equation_solution (x : ℝ) 
  (h1 : x + 2 > 0) 
  (h2 : 3 * x - 4 > 0)
  (h3 : x - 2 > 0) 
  (h4 : Real.log 5 (x + 2) - Real.log 5 (3 * x - 4) = - Real.log 5 (x - 2)) : 
  x = 3 :=
sorry

end log_equation_solution_l290_290130


namespace license_plates_count_with_9_and_div_by_9_l290_290021

/-- 
Given a range of license plates from "10000" to "99999". 
We need to count how many of them contain at least one '9' 
and have the sum of their digits divisible by 9.
--/
def digit_sum (n : ℕ) : ℕ := 
  (toString n).toList.map (λ c, c.toNat - '0'.toNat).sum

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  (toString n).toList.any (λ c, c.toNat - '0'.toNat = d)

theorem license_plates_count_with_9_and_div_by_9 :
  (Finset.filter (λ n, contains_digit 9 n ∧ digit_sum n % 9 = 0) (Finset.range 100000)).card.toNat 
  - (Finset.filter (λ n, contains_digit 9 n ∧ digit_sum n % 9 = 0) (Finset.range 10000)).card.toNat 
  = 84168 := 
sorry

end license_plates_count_with_9_and_div_by_9_l290_290021


namespace cp_value_in_right_triangle_l290_290097

theorem cp_value_in_right_triangle 
  (A B C : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [Tri : Triangle A B C]
  (AC BC AB : ℝ) 
  (hAC : AC = Real.sqrt 85) 
  (hAB : AB = 7) 
  (right_angle_at_B : isRightAngle B)
  (circle_center_core : ∃ (O : A), isCircle (Segment AB O) (TangentTo AC BC)) 
  (tangent_P_to_AC : ∃ (P : A), isTangent P → TangentTo AC P) : 

end cp_value_in_right_triangle_l290_290097


namespace oscar_leap_longer_l290_290408

theorem oscar_leap_longer (h_markers : 51) (total_distance : 7920) (elmer_strides_per_gap : 66)
(oscar_leaps_per_gap : 18) :
  let gaps := h_markers - 1 in
  let elmer_total_strides := elmer_strides_per_gap * gaps in
  let oscar_total_leaps := oscar_leaps_per_gap * gaps in
  let elmer_stride_length := (total_distance : ℝ) / elmer_total_strides in
  let oscar_leap_length := (total_distance : ℝ) / oscar_total_leaps in
  oscar_leap_length - elmer_stride_length = 6.4 :=
by
  have gaps : ℕ := h_markers - 1
  have elmer_total_strides : ℕ := elmer_strides_per_gap * gaps
  have oscar_total_leaps : ℕ := oscar_leaps_per_gap * gaps
  have elmer_stride_length : ℝ := (total_distance : ℝ) / elmer_total_strides
  have oscar_leap_length : ℝ := (total_distance : ℝ) / oscar_total_leaps
  have difference : ℝ := oscar_leap_length - elmer_stride_length
  show difference = 6.4
  sorry

end oscar_leap_longer_l290_290408


namespace AC_parallel_BK_l290_290879

-- Given definitions and conditions:
variables (A B C M K : Point)
variable (between_M_A_B : Between M A B)
variable (equilateral_ABC : Equilateral A B C)
variable (equilateral_MKC : Equilateral M K C)
variable (M_K_diff_halfplanes : DifferentHalfPlanes M K B C)

-- The goal is to prove that lines AC and BK are parallel.
theorem AC_parallel_BK (A B C M K : Point)
  (between_M_A_B : Between M A B)
  (equilateral_ABC : Equilateral A B C)
  (equilateral_MKC : Equilateral M K C)
  (M_K_diff_halfplanes : DifferentHalfPlanes M K B C) :
  Parallel (Line.mk A C) (Line.mk B K) :=
sorry

end AC_parallel_BK_l290_290879


namespace royal_children_count_l290_290623

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290623


namespace triangle_problem_l290_290440

theorem triangle_problem (T : ℕ) (ABC : Triangle ℝ)
  (D E F : Point)
  (P : Point)
  (p : ℝ)
  (AP PD : ℝ) 
  (BE CF : ℝ) 
  (S : ℕ) 
  (m n : ℕ)
  (hT : T = 1801)
  (hS : S = 10)
  (ha_bisectors : are_angle_bisectors A B C D E F)
  (hP : P = intersection AD BE)
  (hAP : AP = 3 * PD)
  (hBE : BE = 9 )
  (hCF : CF = 9)
  (hRat : ∃ m n : ℕ, (AD / p : ℝ) = (real.sqrt m) / n
    ∧ ¬ ∃ (k : ℕ), (2 ≤ k ∧ k * k ∣ m)) :
  m + n = 18 :=
sorry

end triangle_problem_l290_290440


namespace max_value_of_f_on_interval_l290_290962

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

def interval : set ℝ := set.Icc (-2 : ℝ) 2

theorem max_value_of_f_on_interval : 
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 2 := 
sorry

end max_value_of_f_on_interval_l290_290962


namespace circumscribed_circle_l290_290103

open EuclideanGeometry

variable {P Q R S : Point}

-- Conditions for the quadrilateral ABCD:
def is_convex_quadrilateral (P Q R S : Point) : Prop := 
  convex_polygon [P, Q, R, S]

def bisectors_meet_at_incenter (P Q R S O : Point) : Prop :=
  is_incenter O P Q R S
  ∧ (∠ P Q O = 2 * ∠ P Q O)
  ∧ (∠ Q R O = 2 * ∠ Q R O)
  ∧ (∠ R S O = 2 * ∠ R S O)
  ∧ (∠ S P O = 2 * ∠ S P O)

def bisectors_are_perpendicular (P Q R S : Point) : Prop :=
  let bisector_A := bisector (get_angle P Q R)
  let bisector_C := bisector (get_angle R S P)
  let bisector_B := bisector (get_angle Q R S)
  let bisector_D := bisector (get_angle S P Q)
  ⟪bisector_A, bisector_C⟫ = 0
  ∧ ⟪bisector_B, bisector_D⟫ = 0

theorem circumscribed_circle (P Q R S : Point) (O : Point) :
  is_convex_quadrilateral P Q R S →
  bisectors_meet_at_incenter P Q R S O →
  bisectors_are_perpendicular P Q R S →
  can_circumscribe_a_circle P Q R S :=
sorry

end circumscribed_circle_l290_290103


namespace problem_statement_l290_290449

noncomputable def a : ℕ → ℤ
| 0     := 1
| (n+1) := if n % 2 = 0 then - (a n + 1) else a n + 1

def S : ℕ → ℤ
| 0     := a 0
| (n+1) := S n + a (n + 1)

theorem problem_statement : S 2013 = -1005 :=
by
  sorry

end problem_statement_l290_290449


namespace change_in_preference_l290_290240

theorem change_in_preference:
  (start_hist_pref end_hist_pref start_geog_pref end_geog_pref : ℝ)
  (start_hist_pref = 0.6) → (end_hist_pref = 0.8) →
  (start_geog_pref = 0.4) → (end_geog_pref = 0.2) →
  ∃ (y_diff : ℝ), y_diff = 0.4 :=
by
  intros
  use 0.4
  sorry

end change_in_preference_l290_290240


namespace fixed_point_of_f_l290_290116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  calc
    a^(1 - 1) + 4 = 1 + 4 : by rw [pow_zero a]
    _ = 5 : by norm_num

end fixed_point_of_f_l290_290116


namespace qevian_concurrent_triples_l290_290441

/-- Let ABC be a triangle with vertices located at the center of masses of three houses, 
with the three points not collinear.
Let N = 2017, and define the A-ntipodes to be the points A_1,...,A_N 
on segment BC such that BA_1 = A_1A_2 = ... = A_{N-1}A_N = A_NC.
Similarly define the B-ntipodes and C-ntipodes.
A line ℓ_A through A is called a qevian if it passes through an A-ntipode. 
Similarly, define qevians through B and C.

Theorem: The number of ordered triples (ℓ_A, ℓ_B, ℓ_C) of concurrent qevians through 
A, B, and C, respectively, is 2017^3 - 2. -/
theorem qevian_concurrent_triples : 
  let N := 2017 in 
  let n := N := 2017
  ∃ A B C : Type,
  ∃ (A_isntipodes : list (set (A×A×A))) 
    (B_isntipodes : list (set (B×B×B))) 
    (C_isntipodes : list (set (C×C×C))),
  (∀ x ∈ A_isntipodes, ∀ y ∈ B_isntipodes, ∀ z ∈ C_isntipodes, 
  (set.size (⋂ u ∈ x | ∀ n = y ∩ ⋂ w ∈ z)) = 2017^3 - 2) :=
  sorry

end qevian_concurrent_triples_l290_290441


namespace hawks_score_l290_290020

theorem hawks_score (a b : ℕ) (h1 : a + b = 58) (h2 : a - b = 12) : b = 23 :=
by
  sorry

end hawks_score_l290_290020


namespace sum_of_abc_l290_290254

theorem sum_of_abc :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   (∀ x : ℝ, sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 →
    cos (a * x) * cos (b * x) * cos (c * x) = 0) ∧
   a + b + c = 14) := 
sorry

end sum_of_abc_l290_290254


namespace find_common_difference_l290_290027

variable (a : ℕ → ℤ)  -- define the arithmetic sequence as a function from ℕ to ℤ
variable (d : ℤ)      -- define the common difference

-- Define the conditions
def conditions := (a 5 = 10) ∧ (a 12 = 31)

-- Define the formula for the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) := a 1 + d * (n - 1)

-- Prove that the common difference d is 3 given the conditions
theorem find_common_difference (h : conditions a) : d = 3 :=
sorry

end find_common_difference_l290_290027


namespace product_end_digit_3_mod_5_l290_290287

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l290_290287


namespace fish_count_seventh_day_l290_290432

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end fish_count_seventh_day_l290_290432


namespace karen_average_speed_l290_290040

noncomputable def total_distance : ℚ := 198
noncomputable def start_time : ℚ := (9 * 60 + 40) / 60
noncomputable def end_time : ℚ := (13 * 60 + 20) / 60
noncomputable def total_time : ℚ := end_time - start_time
noncomputable def average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed :
  average_speed total_distance total_time = 54 := by
  sorry

end karen_average_speed_l290_290040


namespace exists_three_digit_number_divisible_by_1_to_8_l290_290036

theorem exists_three_digit_number_divisible_by_1_to_8 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ x ∈ {1, 2, 3, 4, 5, 6, 7, 8}, x ∣ n :=
by
  use 840
  sorry

end exists_three_digit_number_divisible_by_1_to_8_l290_290036


namespace five_digit_palindromes_count_l290_290738

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l290_290738


namespace parabola_focus_ratio_l290_290448

theorem parabola_focus_ratio
  (P : ℝ → ℝ) (Q : ℝ → ℝ)
  (V1 V2 F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (hP : ∀ x, P x = (1 / 2) * x ^ 2)
  (hV1 : V1 = (0, 0))
  (hA : A = (a, (1 / 2) * a ^ 2))
  (hB : B = (b, (1 / 2) * b ^ 2))
  (hab : a * b = -2)
  (locus_midpoint : ∀ x, Q x = (1 / 2) * x ^ 2 + 1)
  (hF1 : F1 = (0, 1))
  (hV2 : V2 = (0, 1))
  (hF2 : F2 = (0, 1.5)) :
  (euclidean_distance F1 F2) / (euclidean_distance V1 V2) = 1 / 2 := 
sorry

end parabola_focus_ratio_l290_290448


namespace minimum_cells_to_determine_covering_l290_290656

def cells_per_domino : ℕ := 10
def chessboard_size : ℕ := 1000
def total_cells : ℕ := chessboard_size * chessboard_size
def total_dominoes : ℕ := total_cells / cells_per_domino

theorem minimum_cells_to_determine_covering :
  ∀ (N : ℕ), (N = total_dominoes) → (N = 100000) :=
by
  assume N,
  intro h,
  rw total_dominoes_def at h,
  sorry

end minimum_cells_to_determine_covering_l290_290656


namespace equivalent_problem_l290_290338

variable (x : ℕ → ℝ) (n : ℕ)

def sample1_avg := (∑ i in finset.range n, 1 + x i) / n
def sample1_var := (∑ i in finset.range n, (1 + x i - 10) ^ 2) / n
def sample2_avg := (∑ i in finset.range n, 2 + x i) / n
def sample2_var := (∑ i in finset.range n, (2 + x i - 11) ^ 2) / n

theorem equivalent_problem (h1 : sample1_avg x n = 10)
  (h2 : sample1_var x n = 2) : sample2_avg x n = 11 ∧ sample2_var x n = 2 := by
  sorry

end equivalent_problem_l290_290338


namespace gcd_problem_l290_290896

-- Define the conditions
def a (d : ℕ) : ℕ := d - 3
def b (d : ℕ) : ℕ := d - 2
def c (d : ℕ) : ℕ := d - 1

-- Define the number formed by digits in the specific form
def abcd (d : ℕ) : ℕ := 1000 * a d + 100 * b d + 10 * c d + d
def dcba (d : ℕ) : ℕ := 1000 * d + 100 * c d + 10 * b d + a d

-- Summing the two numbers
def num_sum (d : ℕ) : ℕ := abcd d + dcba d

-- The GCD of all num_sum(d) where d ranges from 3 to 9
def gcd_of_nums : ℕ := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (num_sum 3) (num_sum 4)) (num_sum 5)) (num_sum 6)) (Nat.gcd (num_sum 7) (Nat.gcd (num_sum 8) (num_sum 9)))

theorem gcd_problem : gcd_of_nums = 1111 := sorry

end gcd_problem_l290_290896


namespace power_of_power_l290_290574

theorem power_of_power {a : ℝ} : (a^2)^3 = a^6 := 
by
  sorry

end power_of_power_l290_290574


namespace probability_of_triangle_l290_290018

open Nat

noncomputable def num_ways_select3 (n : ℕ) : ℕ :=
  Nat.choose n 3

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triplets (segments : List ℕ) : List (ℕ × ℕ × ℕ) :=
  segments.pairs.pairs_filter (fun (a, (b, c)) => (a < b) ∧ (b < c) ∧ satisfies_triangle_inequality a b c)

def probability_to_form_triangle : ℚ :=
  let segments := [1, 3, 7, 8, 9]
  let total_ways := num_ways_select3 (length segments)
  let successful_ways := length (valid_triplets segments)
  successful_ways / total_ways

theorem probability_of_triangle :
  probability_to_form_triangle = 2 / 5 :=
sorry

end probability_of_triangle_l290_290018


namespace evaluate_g_l290_290901

def g (a b c d : ℤ) : ℚ := (d * (c + 2 * a)) / (c + b)

theorem evaluate_g : g 4 (-1) (-8) 2 = 0 := 
by 
  sorry

end evaluate_g_l290_290901


namespace train_length_l290_290691

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l290_290691


namespace h_inverse_correct_l290_290066

noncomputable def f (x : ℝ) := 4 * x + 7
noncomputable def g (x : ℝ) := 3 * x - 2
noncomputable def h (x : ℝ) := f (g x)
noncomputable def h_inv (y : ℝ) := (y + 1) / 12

theorem h_inverse_correct : ∀ x : ℝ, h_inv (h x) = x :=
by
  intro x
  sorry

end h_inverse_correct_l290_290066


namespace line_tangent_through_A_l290_290580

theorem line_tangent_through_A {A : ℝ × ℝ} (hA : A = (1, 2)) : 
  ∃ m b : ℝ, (b = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ (∀ y x : ℝ, y^2 = 4*x → y = 2) :=
by
  sorry

end line_tangent_through_A_l290_290580


namespace sum_of_cosines_l290_290933

theorem sum_of_cosines (m : ℕ) :  
    Finset.sum (Finset.range m) (λ k, Real.cos (2 * (k + 1) * Real.pi / (2 * m + 1))) = -1/2 :=
sorry

end sum_of_cosines_l290_290933


namespace product_end_digit_3_mod_5_l290_290284

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l290_290284


namespace circumcenter_of_BIC_lies_on_circumcircle_of_ABC_l290_290463

theorem circumcenter_of_BIC_lies_on_circumcircle_of_ABC
  {A B C I : Type}
  [triangle : IsTriangle A B C] -- Assuming IsTriangle is a typeclass verifying three points form a triangle
  [is_incenter I A B C]         -- I being the incenter of triangle ABC
  (circumcenter_BIC_lies_on_circumcircle_ABC : IsCircumcenter (BIC) (Circumcircle ABC)) -- Assuming IsCircumcenter verifies the circumcenter property
  : IsOnCircumcircle (Circumcenter BIC) (Circumcircle ABC) := sorry

end circumcenter_of_BIC_lies_on_circumcircle_of_ABC_l290_290463


namespace max_value_inequality_l290_290014

theorem max_value_inequality (x y : ℝ) (h : x^2 + y^2 = 20) : 
  ∃ x y : ℝ, x^2 + y^2 = 20 ∧ xy + 8x + y ≤ 42 :=
by {
  sorry
}

end max_value_inequality_l290_290014


namespace perp_proof_l290_290825

open Real

def vector (α : Type) := prod α α

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def add_vectors (v1 v2 : vector ℝ) : vector ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def is_perp (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

def vec_a : vector ℝ := (2, 0)
def vec_b : vector ℝ := (-1, 1)
def vec_c : vector ℝ := add_vectors vec_a vec_b

theorem perp_proof : is_perp vec_b vec_c := by
  sorry

end perp_proof_l290_290825


namespace find_remainder_of_trailing_zeros_mod_500_l290_290445

noncomputable def num_trailing_zeros_in_factorial_product (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), ∑ j in Finset.range i, (j + 1).factorial.factors.count 5

theorem find_remainder_of_trailing_zeros_mod_500 :
  (num_trailing_zeros_in_factorial_product 50) % 500 = 15 :=
by
  have M := num_trailing_zeros_in_factorial_product 50
  show M % 500 = 15
  sorry

end find_remainder_of_trailing_zeros_mod_500_l290_290445


namespace wire_division_l290_290368

theorem wire_division (total_length : ℕ) (num_parts : ℕ)
  (h_total_length : total_length = 49) (h_num_parts : num_parts = 7) :
  total_length / num_parts = 7 :=
by
  rw [h_total_length, h_num_parts]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num)
  sorry

end wire_division_l290_290368


namespace expectation_inequality_l290_290451

variables {X Y : ℝ → ℝ} (c : ℝ × ℝ → ℝ) 
  [MeasureTheory.BorelSpace ℝ] [ProbabilityTheory.IsIndependent X Y]

-- Hypotheses
def bounded_borel (c : ℝ × ℝ → ℝ) :=
  ∀ x y, MeasureTheory.Borel (c (x, y))

def expectation_X2 : ℝ :=
  Real.sqrt (ProbabilityTheory.variance X) = 1

def expectation_Y2 : ℝ :=
  Real.sqrt (ProbabilityTheory.variance Y) = 1

-- Theorem Statement
theorem expectation_inequality (h1 : bounded_borel c) (h2 : expectation_X2)
  (h3 : expectation_Y2) : 
  ∃ supEx, ∃ supEy, 
    ∣ ProbabilityTheory.variance (λ xy : ℝ×ℝ, c xy * (fst xy) * (snd xy)) ∣^2 ≤
        (supEx * supEy) :=
sorry

end expectation_inequality_l290_290451


namespace max_rectangles_from_square_l290_290870

theorem max_rectangles_from_square (side_length : ℝ) (rect_length : ℝ) (rect_width : ℝ) (h_side_length : side_length = 14) (h_rect_length : rect_length = 8) (h_rect_width : rect_width = 2) : 
  let A_square := side_length * side_length,
      A_rectangle := rect_length * rect_width,
      max_rectangles := ⌊A_square / A_rectangle⌋ in
  max_rectangles = 12 :=
sorry

end max_rectangles_from_square_l290_290870


namespace weighted_average_score_is_correct_l290_290091

-- Defining the marks in different subjects
def marks_math := 76
def marks_science := 65
def marks_social := 82
def marks_english := 62
def marks_biology := 85

-- Defining the weightages of different subjects
def weightage_math := 0.20
def weightage_science := 0.15
def weightage_social := 0.25
def weightage_english := 0.25
def weightage_biology := 0.15

-- Defining the weighted scores
def weighted_math := marks_math * weightage_math
def weighted_science := marks_science * weightage_science
def weighted_social := marks_social * weightage_social
def weighted_english := marks_english * weightage_english
def weighted_biology := marks_biology * weightage_biology

-- Sum of all weighted scores
def weighted_sum := 
    weighted_math + 
    weighted_science + 
    weighted_social + 
    weighted_english + 
    weighted_biology

-- Prove that the weighted sum is 73.7
theorem weighted_average_score_is_correct : weighted_sum = 73.7 := 
by sorry

end weighted_average_score_is_correct_l290_290091


namespace simplify_fractional_exponents_l290_290242

theorem simplify_fractional_exponents :
  (5 ^ (1/6) * 5 ^ (1/2)) / 5 ^ (1/3) = 5 ^ (1/6) :=
by
  sorry

end simplify_fractional_exponents_l290_290242


namespace value_of_a_minus_b_l290_290330

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : 2 * a - b = 1) : a - b = -1 :=
by
  sorry

end value_of_a_minus_b_l290_290330


namespace complementary_angle_ratio_l290_290992

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l290_290992


namespace min_b_plus_c_l290_290334

theorem min_b_plus_c (b c : ℕ) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h0 : ∃ k = 0, discriminant (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑c - polynomial.C ↑k) ≥ 0 
  ∧ ∀ (x : ℤ), ¬ polynomial.eval x (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑c - polynomial.C ↑k) = 0)
  (h1 : ∃ k = 1, discriminant (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑c - polynomial.C ↑k) ≥ 0
  ∧ ∀ (x : ℤ), ¬ polynomial.eval x (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑k = 0))
  (h2 : ∃ k = 2, discriminant (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑c - polynomial.C ↑k) ≥ 0
  ∧ ∀ (x : ℤ), ¬ polynomial.eval x (polynomial.X^2 + (polynomial.C ↑b)*polynomial.X + polynomial.C ↑k = 0)) :
  b + c = 8 := 
  sorry


end min_b_plus_c_l290_290334


namespace total_distance_is_10_miles_l290_290708

noncomputable def total_distance_back_to_town : ℕ :=
  let distance1 := 3
  let distance2 := 3
  let distance3 := 4
  distance1 + distance2 + distance3

theorem total_distance_is_10_miles :
  total_distance_back_to_town = 10 :=
by
  sorry

end total_distance_is_10_miles_l290_290708


namespace proof_problem_l290_290921

namespace GeometryProof

variables {A1 A2 A3 A4 O1 O2 O3 O4 O : Point}
variables (S1 S2 S3 S4 : Circle) (hA4_ortho : IsOrthocenter A4 A1 A2 A3)
variables (centerS1 : Center S1 = O1) (centerS2 : Center S2 = O2)
variables (centerS3 : Center S3 = O3) (centerS4 : Center S4 = O4)
variables (circumS4 : CircumCircle A1 A2 A3 = S4)
variables (circumS3 : CircumCircle A1 A2 A4 = S3)
variables (circumS2 : CircumCircle A1 A3 A4 = S2)
variables (circumS1 : CircumCircle A2 A3 A4 = S1)

theorem proof_problem :
  (IsOrthocenter A1 A2 A3 A4 ∧ IsOrthocenter A2 A1 A3 A4 ∧ IsOrthocenter A3 A1 A2 A4) ∧
  (IsCongruent S1 S2 ∧ IsCongruent S2 S3 ∧ IsCongruent S3 S4) ∧
  (IsSymmetricQuadrilateral O1 O2 O3 O4 A1 A2 A3 A4 O).
sorry

end GeometryProof

end proof_problem_l290_290921


namespace sum_of_valid_N_l290_290253

theorem sum_of_valid_N : 
  let N_values := {N | N < 1000 ∧ (N + 2^2015) % 257 = 0}
  ∑ n in N_values, n = 2058 := 
by
  sorry

end sum_of_valid_N_l290_290253


namespace five_digit_palindromes_count_l290_290734

theorem five_digit_palindromes_count : 
  let a_values := {a : ℕ | 1 ≤ a ∧ a ≤ 9}
  let b_values := {b : ℕ | 0 ≤ b ∧ b ≤ 9}
  let c_values := {c : ℕ | 0 ≤ c ∧ c ≤ 9}
  a_values.card * b_values.card * c_values.card = 900 := 
by 
  -- a has 9 possible values
  have a_card : a_values.card = 9 := sorry
  -- b has 10 possible values
  have b_card : b_values.card = 10 := sorry
  -- c has 10 possible values
  have c_card : c_values.card = 10 := sorry
  -- solve the multiplication
  sorry

end five_digit_palindromes_count_l290_290734


namespace find_c2013_l290_290362

theorem find_c2013 :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ),
    (a 1 = 3) →
    (b 1 = 3) →
    (∀ n : ℕ, 1 ≤ n → a (n+1) - a n = 3) →
    (∀ n : ℕ, 1 ≤ n → b (n+1) = 3 * b n) →
    (∀ n : ℕ, c n = b (a n)) →
    c 2013 = 27^2013 := by
  sorry

end find_c2013_l290_290362


namespace MrAllenSpeed_l290_290483

noncomputable def timeToOffice (d : ℝ) (v : ℝ) : ℝ := d / v

theorem MrAllenSpeed (d t : ℝ) :
  let t_ideal := t * 60 in  -- convert hours to minutes
  let t1 := t_ideal + 5 in  -- late by 5 minutes
  let t2 := t_ideal - 5 in  -- early by 5 minutes
  d = 30 * (t + 5 / 60) →    -- distance equation for 30 mph
  d = 50 * (t - 5 / 60) →    -- distance equation for 50 mph
  t = 1 / 3 →                -- derived ideal time
  float.eq (timeToOffice d t) 37.5 := -- necessary speed for on-time arrival
sorry

end MrAllenSpeed_l290_290483


namespace floor_of_summation_l290_290475

noncomputable def T : ℝ :=
  (∑ i in finset.range 1000, (sqrt(1 + 1/(i+1:ℝ)^3 + 1/((i+1+1):ℝ)^3))^3)

theorem floor_of_summation :
  ∃ (n : ℤ), n = ⌊T⌋ := sorry

end floor_of_summation_l290_290475


namespace max_min_y_over_x_max_min_distance_l290_290799

theorem max_min_y_over_x (x y : ℝ) (h : (x - 3) ^ 2 + (y - 3) ^ 2 = 6) :
  (max (y / x) = 3 + 2 * Real.sqrt 2) ∧ (min (y / x) = 3 - 2 * Real.sqrt 2) := sorry

theorem max_min_distance (x y : ℝ) (h : (x - 3) ^ 2 + (y - 3) ^ 2 = 6) :
  (max (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = Real.sqrt 10 + Real.sqrt 6) ∧ 
  (min (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = Real.sqrt 10 - Real.sqrt 6) := sorry

end max_min_y_over_x_max_min_distance_l290_290799


namespace ian_saves_per_day_l290_290381

-- Let us define the given conditions
def total_saved : ℝ := 0.40 -- Ian saved a total of $0.40
def days : ℕ := 40 -- Ian saved for 40 days

-- Now, we need to prove that Ian saved 0.01 dollars/day
theorem ian_saves_per_day (h : total_saved = 0.40 ∧ days = 40) : total_saved / days = 0.01 :=
by
  sorry

end ian_saves_per_day_l290_290381


namespace true_propositions_l290_290361

-- Definitions of lines, planes, and their relations
variable (Line : Type) (Plane : Type)
variable (m n l : Line) (α β : Plane)

-- Propositions as hypotheses
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Given conditions and hypotheses
variable (h1 : m ≠ n) -- non-coincident lines
variable (h2 : α ≠ β) -- non-coincident planes

-- Propositions:
variable (p1 : ∀ m n α, parallel m n → subset n α → parallel m α)
variable (p2 : ∀ l α m β, perpendicular l α → perpendicular m β → parallel l m → parallel_planes α β)
variable (p3 : ∀ m α n β, subset m α → subset n α → parallel m β → parallel n β → parallel_planes α β)
variable (p4 : ∀ α β m n, perpendicular_planes α β → intersection α β = m → subset n β → perpendicular n m → perpendicular n α)

-- Proof that propositions 2 and 4 are true
theorem true_propositions 
  (h_p2 : ∀ l α m β, perpendicular l α → perpendicular m β → parallel l m → parallel_planes α β)
  (h_p4 : ∀ α β m n, perpendicular_planes α β → intersection α β = m → subset n β → perpendicular n m → perpendicular n α) :
  True :=
begin
  -- Skipping the actual proof with sorry
  sorry
end

end true_propositions_l290_290361


namespace ratioOAOB_range_l290_290854

noncomputable def polarEquationLineL : ℝ → ℝ → Prop := λ (ρ θ : ℝ), ρ * Real.cos θ = 4

noncomputable def parametricCurveC (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 2 * Real.cos φ, 1 + Real.sqrt 2 * Real.sin φ)

noncomputable def cartesianCurveC : ℝ × ℝ → Prop := λ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2

noncomputable def polarEquationCurveC : ℝ → ℝ → Prop := λ (ρ θ : ℝ),
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

noncomputable def ratioOAOB (α : ℝ) : ℝ :=
  let ρ₁ := 2 * Real.cos α + 2 * Real.sin α in
  let ρ₂ := 4 / Real.cos α in
  ρ₁ / ρ₂

theorem ratioOAOB_range (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 4) :
  1 / 2 < ratioOAOB α ∧ ratioOAOB α ≤ 1 + Real.sqrt 2 / 4 := sorry

end ratioOAOB_range_l290_290854


namespace current_trees_in_park_l290_290553

-- Definitions based on the conditions
def trees_planted_today : ℕ := 3
def trees_planted_tomorrow : ℕ := 2
def total_trees_after_planting : ℕ := 12
def trees_planted_total : ℕ := trees_planted_today + trees_planted_tomorrow

-- The proof statement
theorem current_trees_in_park (X : ℕ) (h : X + trees_planted_total = total_trees_after_planting) : X = 7 :=
by
  have trees_planted_total_eq : trees_planted_total = 5 := by
    rw [trees_planted_today, trees_planted_tomorrow]
    exact rfl

  rw [← trees_planted_total_eq] at h
  linarith

end current_trees_in_park_l290_290553


namespace probability_digit_9_in_3_over_11_is_zero_l290_290491

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end probability_digit_9_in_3_over_11_is_zero_l290_290491


namespace polygons_no_common_interiors_l290_290154

open EuclideanGeometry

/-- 
Given two polygons on a plane, if the distance between 
any two vertices of the same polygon is at most 1 and 
the distance between any two vertices of different 
polygons is at least 1 / sqrt(2), then the two polygons 
do not share any common interior points. 
-/
theorem polygons_no_common_interiors
  (P1 P2 : Set Point)
  (h1 : ∀ (v1 v2 : Point), v1 ∈ P1 → v2 ∈ P1 → dist v1 v2 ≤ 1)
  (h2 : ∀ (v1 : Point) (v2 : Point), v1 ∈ P1 → v2 ∈ P2 → dist v1 v2 ≥ 1/Real.sqrt 2) :
  ¬ ∃ (p : Point), p ∈ interior P1 ∧ p ∈ interior P2 :=
by 
  sorry

end polygons_no_common_interiors_l290_290154


namespace total_games_played_l290_290746

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l290_290746


namespace boat_upstream_time_is_1_5_hours_l290_290193

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l290_290193


namespace complementary_angle_ratio_l290_290990

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l290_290990


namespace sum_ge_sequence_l290_290076

variable (C D : Finset ℕ)
variable [Nonempty (C ∩ D)] -- Ensures C and D are non-empty sets

noncomputable def a (n : ℕ) : ℕ := 3^(n - 1)

noncomputable def S (A : Finset ℕ) : ℕ := A.sum a

theorem sum_ge_sequence 
  (hC : C ⊆ Finset.range 100) 
  (hD : D ⊆ Finset.range 100) 
  (hCD : S a C ≥ S a D) :
  S a C + S a (C ∩ D) ≥ 2 * S a D := sorry

end sum_ge_sequence_l290_290076


namespace range_of_vector_magnitude_l290_290409

noncomputable def unit_vector {α : Type*} [normed_group α] (e : α) : Prop :=
  ‖e‖ = 1

theorem range_of_vector_magnitude {α : Type*} [inner_product_space ℝ α] {a e : α}
  (h_e : unit_vector e) (h_dot : ⟪a, e⟫ = 2)
  (h_ineq : ∀ t : ℝ, ‖a‖^2 ≤ 5 * ‖a + t • e‖) :
  ∃ (l u : ℝ), l = Real.sqrt 5 ∧ u = 2 * Real.sqrt 5 ∧ l ≤ ‖a‖ ∧ ‖a‖ ≤ u :=
sorry

end range_of_vector_magnitude_l290_290409


namespace stratified_sampling_male_athletes_l290_290226

theorem stratified_sampling_male_athletes
  (num_males : ℕ) (num_females : ℕ) (total_sample : ℕ)
  (total_population : num_males + num_females = 56)
  (sample_size : total_sample = 14) :
  let x := (num_males * total_sample) / (num_males + num_females)
  in x = 8 :=
by
  sorry

end stratified_sampling_male_athletes_l290_290226


namespace Dave_guitar_strings_replacement_l290_290727

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l290_290727


namespace three_digit_numbers_with_2_without_4_l290_290376

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l290_290376


namespace acute_triangle_angle_l290_290411

theorem acute_triangle_angle (A B C H : Point) (α β : ℝ) (h_triangle : acute_triangle A B C)
  (h_altitude : altitude A H B C) (ratio_condition : ratio BH HC AH = 2/3/6)
  (tan_alpha : tan α = 1/3) (tan_beta : tan β = 1/2) :
  ∠A = π / 4 := by
  sorry

end acute_triangle_angle_l290_290411


namespace number_of_mappings_number_of_surjective_mappings_l290_290887

open Finset

variables {X Y : Type} [Fintype X] [Fintype Y]
variable n : ℕ
variable m : ℕ
variable H : Fintype.card X = n
variable G : Fintype.card Y = m

-- Part (i): Number of mappings from X to Y
theorem number_of_mappings (H : Fintype.card X = n) (G : Fintype.card Y = m) :
  (Fintype.card (X → Y)) = m^n :=
sorry

-- Part (ii): Number of surjective mappings from X to Y
theorem number_of_surjective_mappings (H : Fintype.card X = n) (G : Fintype.card Y = m) :
  (Fintype.card {f : X → Y // Function.Surjective f}) =
  m^n - ∑ k in range m, (-1)^k * (m.choose k) * (m-k)^n :=
sorry

end number_of_mappings_number_of_surjective_mappings_l290_290887


namespace number_of_customers_l290_290013

theorem number_of_customers
  (nails_per_person : ℕ)
  (total_sounds : ℕ)
  (trimmed_nails_per_person : nails_per_person = 20)
  (produced_sounds : total_sounds = 100) :
  total_sounds / nails_per_person = 5 :=
by
  -- This is offered as a placeholder to indicate where a Lean proof goes.
  sorry

end number_of_customers_l290_290013


namespace seating_arrangements_count_l290_290230

open Nat

/-- Define the characters involved in the seating arrangement. -/
inductive Person
  | Alice | Bob | Carla | Fiona | George
  deriving DecidableEq

/-- Define the seating arrangement as a list of Persons. -/
def seating_arrangement := List Person

/-- A function to check the conditions for a valid seating. -/
def is_valid_seating (arr: seating_arrangement) : Prop :=
  -- Condition: Alice not in the first chair
  arr.head ≠ Person.Alice ∧
  -- Condition: Alice not next to Bob or Carla
  ∀ i, i < 4 → (arr.get?  i = some Person.Alice → arr.get? (i+1) ≠ some Person.Bob ∧ arr.get? (i+1) ≠ some Person.Carla) ∧
  (arr.get? (i+1) = some Person.Alice → arr.get? i ≠ some Person.Bob ∧ arr.get? i ≠ some Person.Carla) ∧
  -- Condition: Fiona not next to George
  (arr.get?  i = some Person.Fiona → arr.get?  (i+1) ≠ some Person.George) ∧ 
  (arr.get? (i+1) = some Person.Fiona → arr.get?  i ≠ some Person.George)

/-- Count valid seating arrangements. -/
def count_valid_seatings : Nat :=
  List.permutations [Person.Alice, Person.Bob, Person.Carla, Person.Fiona, Person.George]
  |>.countP is_valid_seating

/-- Prove the number of valid arrangements is 12. -/
theorem seating_arrangements_count : count_valid_seatings = 12 :=
by
  sorry

end seating_arrangements_count_l290_290230


namespace no_solution_for_inequality_system_l290_290945

theorem no_solution_for_inequality_system (x : ℝ) : 
  ¬ ((2 * x + 3 ≥ x + 11) ∧ (((2 * x + 5) / 3 - 1) < (2 - x))) :=
by
  sorry

end no_solution_for_inequality_system_l290_290945


namespace number_is_4_less_than_opposite_l290_290643

-- Define the number and its opposite relationship
def opposite_relation (x : ℤ) : Prop := x = -x + (-4)

-- Theorem stating that the given number is 4 less than its opposite
theorem number_is_4_less_than_opposite (x : ℤ) : opposite_relation x :=
sorry

end number_is_4_less_than_opposite_l290_290643


namespace income_is_20000_l290_290961

-- Definitions from conditions
def income (x : ℕ) : ℕ := 4 * x
def expenditure (x : ℕ) : ℕ := 3 * x
def savings : ℕ := 5000

-- Theorem to prove the income
theorem income_is_20000 (x : ℕ) (h : income x - expenditure x = savings) : income x = 20000 :=
by
  sorry

end income_is_20000_l290_290961


namespace eg_over_es_400_l290_290853

variables (E F G H Q R S : Point)
variable [affine_space ℝ Point]

-- Definitions based on conditions
def is_parallelogram (E F G H : Point) : Prop :=
  collinear (lseg E F) (lseg G H) ∧ collinear (lseg E H) (lseg F G)

def eq_EQ_EF_ratio (E F Q : Point) : Prop :=
  dist E Q / dist E F = 3 / 700

def eq_ER_EH_ratio (E H R : Point) : Prop :=
  dist E R / dist E H = 3 / 500

def point_of_intersection (E G Q R S : Point) : Prop :=
  same_line E G S ∧ same_line Q R S

-- The final theorem to prove
theorem eg_over_es_400
  (EFGH_parallelogram : is_parallelogram E F G H)
  (EQ_EF_ratio : eq_EQ_EF_ratio E F Q)
  (ER_EH_ratio : eq_ER_EH_ratio E H R)
  (S_intersection : point_of_intersection E G Q R S) :
  dist E G / dist E S = 400 := 
sorry

end eg_over_es_400_l290_290853


namespace part_I_part_II_l290_290343

-- Define the ellipse C and the point M
def ellipse_C (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def point_M := (0 : ℝ, 1 : ℝ)
def point_N := (0 : ℝ, 1 / 2 : ℝ)

-- Define midpoint property for points on a line
def is_midpoint (P A M : (ℝ × ℝ)) : Prop :=
  P.1 = (A.1 + M.1) / 2 ∧ P.2 = (A.2 + M.2) / 2

-- Define the lines found in part (I)
def line_4sqrt3_minus3y_plus3 (x y : ℝ) : Prop := 4 * real.sqrt 3 * x - 3 * y + 3 = 0
def line_4sqrt3_plus3y_minus3 (x y : ℝ) : Prop := 4 * real.sqrt 3 * x + 3 * y - 3 = 0

-- Prove part (I)
theorem part_I (A B : (ℝ × ℝ)) :
  (∃ l : ℝ → ℝ, (∀ x, l x = (A.2 - 1) / (A.1 - 0) * (x - A.1) + A.2) ∧ 
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧ 
  is_midpoint (0, l 0) A point_M) →
  (line_4sqrt3_minus3y_plus3 = true) ∨ (line_4sqrt3_plus3y_minus3 = true) :=
sorry

-- Define vector addition & modulo
def vector_add_mod (v1 v2 : ℝ × ℝ) : ℝ × ℝ := 
  (v1.1 + v2.1, v1.2 + v2.2 - 1)

def magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Prove part (II)
theorem part_II (A B : (ℝ × ℝ)) :
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 → 
  ∀ N, N = point_N → magnitude (vector_add_mod (A.1, A.2 - 1/2) (B.1, B.2 - 1/2)) ≤ 1 :=
sorry

end part_I_part_II_l290_290343


namespace cookies_per_child_l290_290845

theorem cookies_per_child 
  (total_cookies : ℕ) 
  (children : ℕ) 
  (x : ℚ) 
  (adults_fraction : total_cookies * x = total_cookies / 4) 
  (remaining_cookies : total_cookies - total_cookies * x = 180) 
  (correct_fraction : x = 1 / 4) 
  (correct_children : children = 6) :
  (total_cookies - total_cookies * x) / children = 30 := by
  sorry

end cookies_per_child_l290_290845


namespace intersection_of_sets_A_B_l290_290358

def set_A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }
def set_B : Set ℝ := { x : ℝ | -2 < x ∧ x ≤ 2 }
def set_intersection : Set ℝ := { x : ℝ | -2 < x ∧ x < -1 }

theorem intersection_of_sets_A_B :
  (set_A ∩ set_B) = set_intersection :=
  sorry

end intersection_of_sets_A_B_l290_290358


namespace amount_with_r_l290_290587

theorem amount_with_r (p q r : ℕ) (h1 : p + q + r = 7000) (h2 : r = (2 * (p + q)) / 3) : r = 2800 :=
sorry

end amount_with_r_l290_290587


namespace max_gcd_d_n_l290_290251

noncomputable def a (n : ℕ) : ℕ := 101 + n^2 + 3 * n
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_d_n : ∃ m : ℕ, ∀ n : ℕ, n > 0 → d n ≤ m ∧ m = 4 :=
by
  sorry

end max_gcd_d_n_l290_290251


namespace ellipse_equation_trajectory_equation_maximum_area_triangle_l290_290795

theorem ellipse_equation :
  ∃ (a b : ℝ), a = 2 ∧ b^2 = 2 ∧
  ∀ (x y : ℝ),
  (let C1 : set (ℝ × ℝ) := λ p : ℝ × ℝ, (p.1)^2 / (a^2) + (p.2)^2 / (b^2) = 1 in
  (x, y) ∈ C1 ↔ x^2 / 4 + y^2 / 2 = 1) := sorry

theorem trajectory_equation : ∀ (P Q : ℝ × ℝ),
  let A : ℝ × ℝ := (-√2, 1),
      foci_F1 := (-√2, 0),
      foci_F2 := (√2, 0),
      B : ℝ × ℝ := (√2, -1),
      C1 : set (ℝ × ℝ) := λ p, p.1^2 / 4 + p.2^2 / 2 = 1
  in (P.1^2 / 4 + P.2^2 / 2 = 1) ∧ (A.1 - Q.1) * (A.1 - P.1) + (A.2 - Q.2) * (A.2 - P.2) = 0
  ∧ (B.1 - Q.1) * (B.1 - P.1) + (B.2 - Q.2) * (B.2 - P.2) = 0
  → (2 * Q.1^2 + Q.2^2 = 5) := sorry

theorem maximum_area_triangle : ∀ (Q : ℝ × ℝ),
  let A : ℝ × ℝ := (-√2, 1),
      B : ℝ × ℝ := (√2, -1),
      line_AB : ℝ × ℝ := (1, √2, 0),
      C1 : set (ℝ × ℝ) := λ p, p.1^2 / 4 + p.2^2 / 2 = 1
  in ¬ collinear A B Q
  ∧ Q.1^2 * 2 + Q.2^2 = 5
  → (area_of_triangle A B Q ≤ 5√2 / 2) := sorry

end ellipse_equation_trajectory_equation_maximum_area_triangle_l290_290795


namespace sum_of_roots_is_zero_l290_290810

-- Definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem Statement
theorem sum_of_roots_is_zero (f : ℝ → ℝ) (h_even : is_even f) (h_intersects : ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) : 
  x1 + x2 + x3 + x4 = 0 :=
by 
  sorry -- Proof can be provided here

end sum_of_roots_is_zero_l290_290810


namespace trajectory_eq_distance_const_lines_thru_fixed_point_l290_290788

-- Define the points and slopes
variables {x y k1 k2 : ℝ}
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Given condition k1 * k2 = -1/4
variables (h : k1 * k2 = -1 / 4)

-- Problem statements
-- (1) Prove the trajectory equation of point P
theorem trajectory_eq : x^2 / 4 + y^2 = 1 ↔ x^2 + 4 * y^2 = 4 :=
by sorry

-- (2) Line l intersects curve C at points M and N
variables {k m : ℝ}
def l (x : ℝ) : ℝ := k * x + m
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- (i) If OM ⊥ ON, prove the distance from O to line l is constant
theorem distance_const (OM_ON_ortho : (0, (l 0)) = (0, m) → False) :
  ∃ d : ℝ, ∀ x, C x (l x) → d = 2 * sqrt 5 / 5 :=
by sorry

-- (ii) If slopes BM and BN satisfy k_BM * k_BN = -1/4
variables {x1 y1 x2 y2 : ℝ} -- Coordinates for points M and N
def BM (x y : ℝ) : ℝ := (y - 0) / (x - 2)
def BN (x y : ℝ) : ℝ := (y - 0) / (x + 2)

theorem lines_thru_fixed_point (hBM : BM x1 (l x1) * BM x2 (l x2) = -1 / 4)
  (hBN : BN x1 (l x1) * BN x2 (l x2) = -1 / 4) :
  ∃ P : ℝ × ℝ, P = (0, 0) :=
by sorry

end trajectory_eq_distance_const_lines_thru_fixed_point_l290_290788


namespace conference_center_distance_l290_290936

theorem conference_center_distance
  (d : ℝ)  -- total distance to the conference center
  (t : ℝ)  -- total on-time duration
  (h1 : d = 40 * (t + 1.5))  -- condition from initial speed and late time
  (h2 : d - 40 = 60 * (t - 1.75))  -- condition from increased speed and early arrival
  : d = 310 := 
sorry

end conference_center_distance_l290_290936


namespace apples_oranges_ratio_l290_290987

-- Define the problem conditions.
def apples_oranges_ratio_problem : Prop :=
  ∃ (A O B : ℕ), (B = 5) ∧ (O = 2 * B) ∧ (A + O + B = 35) ∧ (A / gcd A O = 2) ∧ (O / gcd A O = 1)

-- The statement to prove the ratio of apples to oranges is 2:1.
theorem apples_oranges_ratio : apples_oranges_ratio_problem :=
begin
  -- Exists three natural numbers A, O, and B.
  use 20, use 10, use 5,
  -- Given the conditions:
  split, exact rfl,        -- B = 5
  split, exact rfl,        -- O = 2 * B
  split, exact rfl,        -- A + O + B = 35
  split,
    -- A / gcd A O = 2
    exact nat.div_eq_of_eq_mul_right (gcd_pos_of_pos_right 20 10) rfl,
  -- O / gcd A O = 1
  exact nat.div_eq_of_eq_mul_right (gcd_pos_of_pos_left 10 20) (by norm_num),
end

end apples_oranges_ratio_l290_290987


namespace total_games_played_l290_290745

theorem total_games_played (won lost total_games : ℕ) 
  (h1 : won = 18)
  (h2 : lost = won + 21)
  (h3 : total_games = won + lost) : total_games = 57 :=
by sorry

end total_games_played_l290_290745


namespace min_selected_athletes_l290_290135

theorem min_selected_athletes (n : ℕ) (m : ℕ) (condition₁ : ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n → a * b ≠ m) : n ≥ 2002 → m = 43 :=
by
  intros hn hcondition₁
  sorry

end min_selected_athletes_l290_290135


namespace max_value_f_in_interval_l290_290940

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem max_value_f_in_interval 
  (φ : ℝ) (h : |φ| < π / 2) 
  (h_sym : ∀ x : ℝ, f (x - π / 3) (2 * (π / 3) + φ) = f (-x - π / 3) (2 * (π / 3) + φ))
  : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ f x φ = 1 :=
sorry

end max_value_f_in_interval_l290_290940


namespace playground_area_l290_290952

theorem playground_area (L B : ℕ) 
  (h1 : B = 8 * L) 
  (h2 : B = 480) 
  (h3 : ∀ total_area : ℕ, total_area = L * B → ∀ A_playground : ℕ, A_playground = total_area / 9) : 
  ∃ A_playground : ℕ, A_playground = 3200 :=
by
  -- Setup the conditions and proof context
  have total_area := L * B,
  have A_playground := total_area / 9,
  sorry

end playground_area_l290_290952


namespace survey_total_participants_l290_290025

noncomputable def total_participants (wrong_thinkers : ℕ) : ℕ :=
  let harmful_believers := wrong_thinkers / 0.538
  (harmful_believers / 0.883).ceil

theorem survey_total_participants : total_participants 28 = 59 := by
  sorry

end survey_total_participants_l290_290025


namespace proj_b_l290_290890

open Matrix Real

-- Definition of orthogonality
def orthogonal (a b : Vector ℝ 2) : Prop :=
  dot_product a b = 0

-- Projections
def proj (u v : Vector ℝ 2) : Vector ℝ 2 :=
  (dot_product u v / dot_product u u) • u

theorem proj_b (a b v : Vector ℝ 2) 
    (h_orthog : orthogonal a b)
    (h_proj_a : proj a v = ⟨[-4/5, -8/5]⟩) :
    proj b v = ⟨[24/5, -12/5]⟩ := 
by 
  sorry

end proj_b_l290_290890


namespace no_positive_integer_solutions_l290_290770

theorem no_positive_integer_solutions (k : ℤ) (h_k_pos : 0 < k) (h_k_large : 10^20 < k) :
  let a := k - 1 in
  let b := (k + 1) * (k^2 - 3) in
  ¬ ∃ x y : ℤ, 0 < x ∧ 0 < y ∧ (a * x^2 - b * y^2 = 1 ∨ a * x^2 - b * y^2 = -1) :=
sorry

end no_positive_integer_solutions_l290_290770


namespace rate_of_volume_change_l290_290190

variable (S : ℝ → ℝ) (V : ℝ → ℝ) (a t : ℝ)

def surface_area (t : ℝ) : ℝ := 6 * t

def side_length (s : ℝ) : ℝ := Real.sqrt (s / 6)

def volume (t : ℝ) : ℝ := (side_length (surface_area t))^3

theorem rate_of_volume_change (h : surface_area t = 144) : deriv volume t = 3 * Real.sqrt 6 := by
  sorry

end rate_of_volume_change_l290_290190


namespace sufficient_condition_parallel_planes_l290_290096

-- Definitions of planes and lines
variables (α β : Plane) (m n : Line) (l1 l2 : Line)

-- Conditions given in the problem
axiom m_in_α : m ∈ α
axiom n_in_α : n ∈ α
axiom l1_in_β : l1 ∈ β
axiom l2_in_β : l2 ∈ β
axiom l1_intersects_l2 : ∃ (p : Point), p ∈ l1 ∧ p ∈ l2

-- To prove the statement
theorem sufficient_condition_parallel_planes (h1 : m ∥ l1) (h2 : n ∥ l2) : α ∥ β :=
sorry

end sufficient_condition_parallel_planes_l290_290096


namespace line_parabola_no_intersect_l290_290885

theorem line_parabola_no_intersect (m : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ) (a b : ℝ) :
  (P = fun x => x^2) ∧ (Q = (12, 8)) ∧ (a = 1.665) ∧ (b = 47.335) →
  (1.665 < m ∧ m < 47.335 ↔ ¬∃ x : ℝ, P x = m * x + 8 - 12 * m) ∧ a + b = 49 :=
begin
  sorry
end

end line_parabola_no_intersect_l290_290885


namespace basketball_team_first_competition_games_l290_290192

-- Definitions given the conditions
def first_competition_games (x : ℕ) := x
def second_competition_games (x : ℕ) := (5 * x) / 8
def third_competition_games (x : ℕ) := x + (5 * x) / 8
def total_games (x : ℕ) := x + (5 * x) / 8 + (x + (5 * x) / 8)

-- Lean 4 statement to prove the correct answer
theorem basketball_team_first_competition_games : 
  ∃ x : ℕ, total_games x = 130 ∧ first_competition_games x = 40 :=
by
  sorry

end basketball_team_first_competition_games_l290_290192


namespace number_of_children_l290_290634

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290634


namespace sum_of_ratios_l290_290125

theorem sum_of_ratios (x y z : ℕ) 
  (h1 : (x, y, z) = (5, 6, 16))
  (h2 : real.sqrt (75 / 128) = (x * real.sqrt y) / z) : 
  x + y + z = 27 := 
by 
  sorry

end sum_of_ratios_l290_290125


namespace student_groups_arrangements_l290_290186

theorem student_groups_arrangements :
  let students := 6
  let groups := 3
  let students_per_group := 2
  let choose_group_ways (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let permute_topics (n : ℕ) := Nat.factorial n
  in (choose_group_ways students students_per_group)
     * (choose_group_ways (students - students_per_group) students_per_group)
     * (choose_group_ways (students - 2 * students_per_group) students_per_group)
     * (permute_topics groups) = 540 :=
by sorry

end student_groups_arrangements_l290_290186


namespace royal_children_count_l290_290611

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290611


namespace mary_sheep_remaining_l290_290480

noncomputable def initial_sheep : ℕ := 1500
noncomputable def percentage_sister : ℚ := 0.25
noncomputable def percentage_brother : ℚ := 0.30
noncomputable def fraction_cousin : ℚ := 1/7

theorem mary_sheep_remaining : 
  (remaining_sheep : ℕ) :=
  let sheep_after_sister := initial_sheep - (percentage_sister * initial_sheep).toInt in
  let sheep_after_brother := sheep_after_sister - (percentage_brother * sheep_after_sister).toInt in
  let sheep_after_cousin := sheep_after_brother - (fraction_cousin * sheep_after_brother).toInt in
  sheep_after_cousin = 676 :=
sorry

end mary_sheep_remaining_l290_290480


namespace max_elements_subset_l290_290051

theorem max_elements_subset (S : Finset ℕ) (hS : ∀ x ∈ S, x ∈ Finset.range 1001) 
  (h_diff : ∀ x y ∈ S, x ≠ y → |x - y| ≠ 3 ∧ |x - y| ≠ 5) : S.card ≤ 375 :=
sorry

end max_elements_subset_l290_290051


namespace series_sum_inequality_2019_l290_290920

theorem series_sum_inequality_2019 :
  1 + (∑ k in Finset.range 2019, 1 / (k + 2)^2) < 4039 / 2020 := by
  sorry

end series_sum_inequality_2019_l290_290920


namespace exactly_one_three_digit_perfect_cube_divisible_by_25_l290_290005

theorem exactly_one_three_digit_perfect_cube_divisible_by_25 :
  ∃! (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 25 = 0 :=
sorry

end exactly_one_three_digit_perfect_cube_divisible_by_25_l290_290005


namespace tangent_lines_to_circle_passing_through_point_l290_290744

theorem tangent_lines_to_circle_passing_through_point :
  ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 1 → ((x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = -1)) :=
by
  sorry

end tangent_lines_to_circle_passing_through_point_l290_290744


namespace career_preference_degrees_l290_290589

variable (M F : ℕ)
variable (h1 : M / F = 2 / 3)
variable (preferred_males : ℚ := M / 4)
variable (preferred_females : ℚ := F / 2)
variable (total_students : ℚ := M + F)
variable (preferred_career_students : ℚ := preferred_males + preferred_females)
variable (career_fraction : ℚ := preferred_career_students / total_students)
variable (degrees : ℚ := 360 * career_fraction)

theorem career_preference_degrees :
  degrees = 144 :=
sorry

end career_preference_degrees_l290_290589


namespace children_count_l290_290594

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290594


namespace five_digit_palindromes_count_l290_290739

theorem five_digit_palindromes_count :
  (∃ (A B C : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  ∃ (count : ℕ), count = 900 :=
by {
  intro h,
  use 900,
  sorry        -- Proof is omitted
}

end five_digit_palindromes_count_l290_290739


namespace orthogonal_projection_l290_290888

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end orthogonal_projection_l290_290888


namespace locus_of_intersection_is_straight_l290_290082

theorem locus_of_intersection_is_straight (A B M : Point) (l : Line) (k : ℝ) :
  on_line A B l →
  on_line M l →
  (∃ (P Q : Point), on_line P (line_through M) ∧ on_line Q (line_through M) ∧ P ≠ Q ∧ (dist M P) = k * (dist M Q)) →
  ∃ (L : Line), ∀ (P Q : Point), (on_line P (line_through M) ∧ on_line Q (line_through M) ∧ (dist M P) = k * (dist M Q)) →
  let K := intersection (line_through A P) (line_through B Q) in on_line K L :=
sorry

end locus_of_intersection_is_straight_l290_290082


namespace problem_statement_l290_290115

theorem problem_statement : 
  (∀ x y : ℤ, y = 2 * x^2 - 3 * x + 4 ∧ y = 6 ∧ x = 2) → (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  sorry

end problem_statement_l290_290115


namespace lonely_numbers_less_than_2021_l290_290832

theorem lonely_numbers_less_than_2021 : 
  {n : ℕ // n < 2021 ∧ ∃ k a : ℕ, k ≥ 2 ∧ n = k * (2 * a + k - 1) / 2 }.card = 10 :=
sorry -- no proof required, placeholder

end lonely_numbers_less_than_2021_l290_290832


namespace find_y_l290_290831

theorem find_y (x k m y : ℤ) 
  (h1 : x = 82 * k + 5) 
  (h2 : x + y = 41 * m + 12) : 
  y = 7 := 
sorry

end find_y_l290_290831


namespace julia_total_spend_l290_290872

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l290_290872


namespace children_count_l290_290598

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290598


namespace pam_total_apples_l290_290492

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end pam_total_apples_l290_290492


namespace not_necessarily_periodic_difference_l290_290205

-- Definition of periodic function
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f(x + p) = f(x)

-- Given Conditions
variable (g h : ℝ → ℝ)
axiom g_periodic : is_periodic g 6
axiom h_periodic : is_periodic h (2 * Real.pi)

-- The proof problem
theorem not_necessarily_periodic_difference : ¬∃ p > 0, is_periodic (λ x, g(x) - h(x)) p := sorry

end not_necessarily_periodic_difference_l290_290205


namespace problem_1_intervals_of_monotonicity_problem_2_minimum_value_l290_290811

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - |x| + 2 * a - 1

-- Problem 1: Given a = 1, find intervals of monotonicity for f(x)
theorem problem_1_intervals_of_monotonicity (x : ℝ) : 
  let a := 1 in
  (x ∈ Ioi (1/2) ∨ x ∈ Iio 0) ∨ (x ∈ Ioi (-1/2) ∧ x < 0) ∨ (x ∈ Iio (-1/2) ∧ x < 0) ∨ (x ∈ Ioi 1/2) :=
sorry

-- Problem 2: Given a > 0, find the minimum value of f(x) in [1, 2] as g(a)
noncomputable def g (a : ℝ) : ℝ := 
  if 0 < a ∧ a < 1/4 then 6 * a - 3 
  else if 1/4 ≤ a ∧ a ≤ 1/2 then 2 * a - 1/(4 * a) - 1 
  else if a > 1/2 then 3 * a - 2 
  else 0

theorem problem_2_minimum_value (a : ℝ) (h : a > 0) : g(a) = 
  if 0 < a ∧ a < 1/4 then 6 * a - 3 
  else if 1/4 ≤ a ∧ a ≤ 1/2 then 2 * a - 1/(4 * a) - 1 
  else if a > 1/2 then 3 * a - 2 
  else 0 :=
sorry

end problem_1_intervals_of_monotonicity_problem_2_minimum_value_l290_290811


namespace maximum_m_l290_290796

open Finset
open Set

variable (n : ℕ) (hn : n > 1)

noncomputable def S_m (m : ℕ) := {1..(m*n)}

def conditions (m : ℕ) (S : Set (Finset ℕ)) := 
  (|S| = 2 * n) ∧ 
  (∀ s ∈ S, s.card = m) ∧
  (∀ (s1 ∈ S) (s2 ∈ S), s1 ≠ s2 → (s1 ∩ s2).card ≤ 1) ∧
  (∀ x ∈ S_m m, (filter (λ s, x ∈ s) S).card = 2)

theorem maximum_m (m : ℕ) (S : Set (Finset ℕ)) (hS : conditions n m S) : m = n := 
sorry

end maximum_m_l290_290796


namespace slope_BT_l290_290541

theorem slope_BT 
  (x1 y1 x2 y2 : ℝ)
  (hA : x1^2 / 25 + y1^2 / 9 = 1)
  (hB : 4^2 / 25 + (9/5)^2 / 9 = 1)
  (hC : x2^2 / 25 + y2^2 / 9 = 1)
  (F := (4, 0))
  (hFocus_distance : 2 * real.sqrt ( (4 - 4)^2 + (9/5 - 0)^2 ) = 
                      real.sqrt ( (4 - x1)^2 + (9/5 - y1)^2 ) + 
                      real.sqrt ( (4 - x2)^2 + (9/5 - y2)^2 ))
  (T := (64/25, 0)) :
  let B := (4, 9/5) in
  let k := (9/5 - 0) / (4 - 64/25) in
  k = 5 / 4 :=
sorry

end slope_BT_l290_290541


namespace sunflower_cans_l290_290868

theorem sunflower_cans (seeds : ℝ) (seeds_per_can : ℝ) (total_cans : ℝ) 
  (h1 : seeds = 54.0) (h2 : seeds_per_can = 6) : total_cans = seeds / seeds_per_can :=
by
  have h3 : total_cans = 54.0 / 6 := by sorry
  exact h3

end sunflower_cans_l290_290868


namespace construct_triangle_from_given_side_radii_l290_290724

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Given conditions
variables (AB : A) (r1 r2 : ℝ) -- Radii of the two circles touching sides AC and BC respectively

-- Definitions
def is_defined_circle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (r : ℝ) : Prop := sorry -- Placeholder definition for circles
def triangle_exists_construction (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (AB : ℝ) (r1 r2 : ℝ) : Prop := 
    ∃ (F H : A) (K M : B), 
    (FH = AB) ∧ -- The segment FH equals the given side AB
    is_defined_circle A B C r1 ∧
    is_defined_circle A B C r2 ∧
    -- Additional properties to establish the vertices of triangle correctly
    ∃ (vertex_constructor : C), by sorry

-- Statement to be proved
theorem construct_triangle_from_given_side_radii 
    (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB : ℝ) (r1 r2 : ℝ) : 
    triangle_exists_construction A B C AB r1 r2 := sorry

end construct_triangle_from_given_side_radii_l290_290724


namespace circle_equation_solution_l290_290761

theorem circle_equation_solution (x y : ℝ) (λ : ℝ) :
  (x - 2*y = 0) → 
  ((x-1)^2 + (y-1)^2 + λ * (x^2 + y^2 - 2*x - 4*y + 4) = 0) → 
  x = 1 → y = (1 + 2*λ)/(1 + λ) → λ = -1/3 →
  (x^2 + y^2 - 2*x - y + 1 = 0) :=
by
  intros h_center_line h_tangent_circle h_x h_y h_lambda
  sorry

end circle_equation_solution_l290_290761


namespace royal_family_children_l290_290603

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290603


namespace largest_product_term_l290_290033

def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a1 * q ^ (n - 1)

def product_of_first_n_terms (a : ℕ → ℝ) (M : ℕ → ℝ) : Prop :=
  ∀ n, M n = ∏ i in finset.range(n), a (i + 1)

theorem largest_product_term 
  (a : ℕ → ℝ) (M : ℕ → ℝ)
  (h_geom_seq : geometric_sequence a 512 (-1/2))
  (h_product : product_of_first_n_terms a M) :
  ∃ n, M n = M 9 ∧ (∀ m, M m ≤ M n) :=
by {
  sorry
}

end largest_product_term_l290_290033


namespace paving_cost_l290_290175

theorem paving_cost (length width rate : ℝ) (h_length : length = 5.5) (h_width : width = 3.75) (h_rate : rate = 400) :
  length * width * rate = 8250 :=
by
  rw [h_length, h_width, h_rate]
  -- Here we calculate 5.5 * 3.75 * 400, which should give 8250
  have : 5.5 * 3.75 = 20.625 := by norm_num
  rw this
  norm_num
  -- Sorry is used here just to indicate the place where the proof completes
  sorry

end paving_cost_l290_290175


namespace calculator_display_after_100_presses_l290_290098

theorem calculator_display_after_100_presses :
  let special_key_op (x : ℝ) := 1 / (1 - x)
  let cycle_length := 3
  let initial_display := 5
  let nth_display (n : ℕ) := Nat.recOn n initial_display 
    (λ _ previous, special_key_op previous)
  (100 % cycle_length = 1) →
  nth_display 100 = -1 / 4 :=
by
  sorry

end calculator_display_after_100_presses_l290_290098


namespace find_AX_l290_290263

theorem find_AX (A B C X : Type) [has_distance A] [has_distance B] [has_distance C] [has_distance X] 
  (hCA : distance C A = 21) (hCB : distance C B = 28) (hBX : distance B X = 24) :
  distance A X = 18 :=
by sorry

end find_AX_l290_290263


namespace total_mile_times_l290_290148

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l290_290148


namespace segment_PS_length_l290_290413

-- Define our points and distances using structures and their properties
structure Quadrilateral :=
(P Q R S : ℝ × ℝ)
(PQ_length QR_length RS_length PS_length : ℝ)
(angle_Q angle_R : ℝ)

-- Define the main theorem
theorem segment_PS_length
  (quad : Quadrilateral)
  (hPQ : quad.PQ_length = 7)
  (hQR : quad.QR_length = 12)
  (hRS : quad.RS_length = 25)
  (h_angle_Q : quad.angle_Q = real.pi / 2)
  (h_angle_R : quad.angle_R = real.pi / 2) :
  quad.PS_length = real.sqrt 313 := 
sorry

end segment_PS_length_l290_290413


namespace chord_length_eq_2_l290_290527

theorem chord_length_eq_2 : 
  ∀ (x y : ℝ),
  (x^2 + y^2 = 4) ∧ (sqrt 3 * x + y = 2 * sqrt 3) →
  (2 * sqrt (4 - (abs (-2 * sqrt 3) / sqrt (1 + 3))^2) = 2) :=
by
  sorry

end chord_length_eq_2_l290_290527


namespace number_of_triplets_l290_290459

theorem number_of_triplets (n : ℕ) (h_prime : Prime n) (h_n_ge_3 : n ≥ 3) :
  let E := λ n : ℕ, nat.card { t : ℕ × ℕ × ℕ // t.1 < t.2 ∧ t.2 < t.3 ∧ t.1 + t.2 + t.3 = n + 2 ∧ 
                                ∀ i : ℤ, 1 ≤ i ∧ i ≤ n → (t.1 = i ∨ t.2 = i ∨ t.3 = i)} in
  E n = (n - 1) * (n - 2) / 2 := sorry

end number_of_triplets_l290_290459


namespace original_players_l290_290985

variable (n : ℕ)
variable (W : ℕ)
variable (a1: ℕ)
variable (a2: ℕ)
variable (n1: ℕ)
variable (n2: ℕ)
variable (avg1: ℕ)
variable (avg2: ℕ)

-- Conditions
def condition1 : Prop := avg1 = 121
def condition2 : Prop := a1 = 110
def condition3 : Prop := a2 = 60
def condition4 : Prop := avg2 = 113
def condition5 : Prop := n >= 1
def original_weight : Prop := W = n * avg1
def total_weight_with_new_players : Prop := (W + a1 + a2) = (n1 + n2) * avg2
def n1_def : Prop := n1 = n
def n2_def : Prop := n2 = 2

-- Theorem to prove
theorem original_players (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) (h6 : original_weight) (h7 : total_weight_with_new_players) (h8 : n1_def) (h9 : n2_def) : n = 7 := by
  sorry

end original_players_l290_290985


namespace find_integers_l290_290759

theorem find_integers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) : 
  (a, b, c, d) = (1, 2, 3, 4) ∨ (a, b, c, d) = (1, 2, 4, 3) ∨ (a, b, c, d) = (1, 3, 2, 4) ∨ (a, b, c, d) = (1, 3, 4, 2) ∨ (a, b, c, d) = (1, 4, 2, 3) ∨ (a, b, c, d) = (1, 4, 3, 2) ∨ (a, b, c, d) = (2, 1, 3, 4) ∨ (a, b, c, d) = (2, 1, 4, 3) ∨ (a, b, c, d) = (2, 3, 1, 4) ∨ (a, b, c, d) = (2, 3, 4, 1) ∨ (a, b, c, d) = (2, 4, 1, 3) ∨ (a, b, c, d) = (2, 4, 3, 1) ∨ (a, b, c, d) = (3, 1, 2, 4) ∨ (a, b, c, d) = (3, 1, 4, 2) ∨ (a, b, c, d) = (3, 2, 1, 4) ∨ (a, b, c, d) = (3, 2, 4, 1) ∨ (a, b, c, d) = (3, 4, 1, 2) ∨ (a, b, c, d) = (3, 4, 2, 1) ∨ (a, b, c, d) = (4, 1, 2, 3) ∨ (a, b, c, d) = (4, 1, 3, 2) ∨ (a, b, c, d) = (4, 2, 1, 3) ∨ (a, b, c, d) = (4, 2, 3, 1) ∨ (a, b, c, d) = (4, 3, 1, 2) ∨ (a, b, c, d) = (4, 3, 2, 1) :=
sorry

end find_integers_l290_290759


namespace find_b_l290_290398

noncomputable def tangent_line_b : ℝ :=
  let f : ℝ → ℝ := λ x, Real.log x
  let df := deriv f
  let slope := 1 / 2
  let x0 := 2
  let y0 := f x0
  let tangent := λ x, slope * x + ((1 / 2) * x0)
  (-1 + Real.log 2)

theorem find_b : tangent_line_b = -1 + Real.log 2 := 
by
  sorry

end find_b_l290_290398


namespace intervals_of_monotonicity_range_of_a_for_monotonicity_l290_290899

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp x / (1 + a * x^2)

theorem intervals_of_monotonicity (x : ℝ) : 
  ∀ a : ℝ, a = (4/3) → 
  (∃ (I1 I2 I3 : set ℝ), I1 = set.Iio 0.5 ∧ I2 = set.Ioo 0.5 1.5 ∧ I3 = set.Ioi 1.5 
  ∧ (∀ x ∈ I1, ∀ y ∈ I1, x < y → f x a < f y a) 
  ∧ (∀ x ∈ I2, ∀ y ∈ I2, x < y → f x a > f y a)
  ∧ (∀ x ∈ I3, ∀ y ∈ I3, x < y → f x a < f y a)) := sorry

theorem range_of_a_for_monotonicity (x : ℝ) : 
  ∀ a : ℝ, 
  (∀ x y : ℝ, x < y → f x a ≤ f y a ∨ f x a ≥ f y a) → 
  a ∈ set.Ioo 0 1 := sorry

end intervals_of_monotonicity_range_of_a_for_monotonicity_l290_290899


namespace find_a_b_l290_290384

theorem find_a_b (a b : ℝ) (h : (a - 2) ^ 2 + |b + 4| = 0) : a + b = -2 :=
sorry

end find_a_b_l290_290384


namespace speed_of_mans_train_l290_290215

theorem speed_of_mans_train (time_to_pass : ℝ) (length_of_goods_train : ℝ) (speed_of_goods_train_kmh : ℝ) : ℝ :=
let speed_of_goods_train_ms := speed_of_goods_train_kmh * 1000 / 3600 in
let relative_speed := length_of_goods_train / time_to_pass in
let speed_of_mans_train_ms := relative_speed - speed_of_goods_train_ms in
let speed_of_mans_train_kmh := speed_of_mans_train_ms * 3600 / 1000 in
speed_of_mans_train_kmh

example : speed_of_mans_train 9 280 62 = 50 := by
  sorry

end speed_of_mans_train_l290_290215


namespace inequality_addition_l290_290895

-- Definitions and Conditions
variables (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c > d)

-- Theorem statement: Prove that a + c > b + d
theorem inequality_addition (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := 
sorry

end inequality_addition_l290_290895


namespace increasing_function_condition_l290_290812

def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x + 1) / (x + 2)

theorem increasing_function_condition (a : ℝ) :
  (∀ x1 x2 : ℝ, -2 < x1 → -2 < x2 → x1 < x2 → f a x1 < f a x2) ↔ a > 1 / 2 :=
by
  sorry

end increasing_function_condition_l290_290812


namespace distinct_square_roots_l290_290385

theorem distinct_square_roots (m : ℝ) (h : 2 * m - 4 ≠ 3 * m - 1) : ∃ n : ℝ, (2 * m - 4) * (2 * m - 4) = n ∧ (3 * m - 1) * (3 * m - 1) = n ∧ n = 4 :=
by
  sorry

end distinct_square_roots_l290_290385


namespace area_triangle_AOB_l290_290817

-- Definitions for geometric objects
def parabola : Type := {p : ℝ × ℝ // p.2 ^ 2 = 4 * p.1}

def focus : parabola := ⟨(1, 0), by norm_num⟩

def line_through_focus (k : ℝ) : Type := 
  {P : ℝ × ℝ // P.2 = k * (P.1 - 1)}

def intersects_parabola (L : line_through_focus k) (P : parabola) : Prop :=
  L.1.1 ≥ 0 ∧ P.1.1 ≥ 0 ∧ (L.1 = P.1)

def AB_length_six (A B : parabola) : Prop :=
  dist A.1 B.1 = 6

-- Main theorem statement
theorem area_triangle_AOB (A B : parabola) (hAB : AB_length_six A B) :
  ∃ k : ℝ, 
    (∃ (L : line_through_focus k), intersects_parabola L A ∧ intersects_parabola L B) ∧
    area (triangle (0, 0) A.1 B.1) = sqrt 6 := 
sorry

end area_triangle_AOB_l290_290817


namespace number_of_fish_l290_290429

theorem number_of_fish (initial_fish : ℕ) (double_day : ℕ → ℕ → ℕ) (remove_fish : ℕ → ℕ → ℕ) (add_fish : ℕ → ℕ → ℕ) :
  (initial_fish = 6) →
  (∀ n m, double_day n m = n * 2) →
  (∀ n d m, d = 3 ∨ d = 5 → remove_fish n d = n - n / m) →
  (∀ n d, d = 7 → add_fish n d = n + 15) →
  (double_day 6 1 = 12) →
  (double_day 12 2 = 24) →
  (remove_fish 24 3 = 16) →
  (double_day 16 4 = 32) →
  (double_day 32 5 = 64) →
  (remove_fish 64 5 = 48) →
  (double_day 48 6 = 96) →
  (double_day 96 7 = 192) →
  (add_fish 192 7 = 207) →
  207 = 207 :=
begin
  intros,
  -- Proof omitted since it's not required
  sorry,
end

end number_of_fish_l290_290429


namespace knights_and_liars_l290_290923

-- Define the conditions: 
variables (K L : ℕ) 

-- Total number of council members is 101
def total_members : Prop := K + L = 101

-- Inequality conditions
def knight_inequality : Prop := L > (K + L - 1) / 2
def liar_inequality : Prop := K <= (K + L - 1) / 2

-- The theorem we need to prove
theorem knights_and_liars (K L : ℕ) (h1 : total_members K L) (h2 : knight_inequality K L) (h3 : liar_inequality K L) : K = 50 ∧ L = 51 :=
by {
  sorry
}

end knights_and_liars_l290_290923


namespace product_mod_5_l290_290289

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l290_290289


namespace rational_iff_geometric_progression_l290_290941

theorem rational_iff_geometric_progression :
  (∃ x a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + a)*(x + c) = (x + b)^2) ↔
  (∃ x : ℚ, ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + (a : ℚ))*(x + (c : ℚ)) = (x + (b : ℚ))^2) :=
sorry

end rational_iff_geometric_progression_l290_290941


namespace parabola_conditions_l290_290955

noncomputable def parabola_eq (a x : ℝ) := a * (x - 1) * (x - 4)

theorem parabola_conditions {a : ℝ} :
  (∃ (y : ℝ) (x : ℝ), parabola_eq a x = y ∧ y = 2 * x) →
  (∃ (y : ℝ), (parabola_eq a 1 = 0 ∧ parabola_eq a 4 = 0)) →
  (parabola_eq a x ∈ {(x, y) : ℝ × ℝ | y = -((2 / 9)) * (x -1) * (x -4 )} ∨
   parabola_eq a x ∈ {(x, y) : ℝ × ℝ | y = -2 * (x -1) * (x -4 )}) :=
sorry

end parabola_conditions_l290_290955


namespace gcd_12m_18n_with_gcd_mn_18_l290_290008

theorem gcd_12m_18n_with_gcd_mn_18 (m n : ℕ) (hm : Nat.gcd m n = 18) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  Nat.gcd (12 * m) (18 * n) = 108 :=
by sorry

end gcd_12m_18n_with_gcd_mn_18_l290_290008


namespace area_of_hexagon_l290_290068

-- Definitions and conditions
variables {A B C P : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]

variables (AB BC CA PA1 PA2 PB1 PB2 PC1 PC2 : ℝ)
variables (A1 A2 B1 B2 C1 C2 : Type)
  [MetricSpace A1] [MetricSpace A2] [MetricSpace B1] [MetricSpace B2] [MetricSpace C1] [MetricSpace C2]

def is_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B ≠ 0 ∧ dist B C ≠ 0 ∧ dist A C ≠ 0 ∧ dist A B + dist B C > dist A C ∧ dist B C + dist A C > dist A B ∧ dist A C + dist A B > dist B C

variables (hΔABC : is_triangle A B C)
variables (hAB : dist A B = 3) (hBC : dist B C = 4) (hCA : dist A C = 5)

def is_equilateral_triangle (P A1 A2 : Type) [MetricSpace P] [MetricSpace A1] [MetricSpace A2] : Prop :=
  dist P A1 = dist A1 A2 ∧ dist A1 A2 = dist A2 P

variables (hPA1A2 : is_equilateral_triangle P A1 A2)
variables (hPB1B2 : is_equilateral_triangle P B1 B2)
variables (hPC1C2 : is_equilateral_triangle P C1 C2)

def hex_area (A1 A2 B1 B2 C1 C2 : Type) [MetricSpace A1] [MetricSpace A2] [MetricSpace B1] [MetricSpace B2] [MetricSpace C1] [MetricSpace C2] : ℝ := sorry

-- Required statement to prove
theorem area_of_hexagon :
  @hex_area A1 A2 B1 B2 C1 C2 _ _ _ _ _ _ = (12 + 22 * Real.sqrt 3) / 15 :=
begin
  sorry
end

end area_of_hexagon_l290_290068


namespace hundredth_term_in_sequence_l290_290919

theorem hundredth_term_in_sequence : ∃ n, (∑ i in range (n + 1), i) ≥ 100 ∧ n = 14 := by
  sorry

end hundredth_term_in_sequence_l290_290919


namespace tangent_circles_radii_l290_290821

open Real

-- Here's the Lean statement of the equivalent proof problem.

theorem tangent_circles_radii (r1 r2 r3 : ℝ)
  (h1 : r1 > 0) (h2 : r2 > 0) (h3 : r3 > 0)
  (tangent_condition : (* external tangents condition *)):
  r1^2 = 4 * r2 * r3 :=
sorry

end tangent_circles_radii_l290_290821


namespace find_XW_l290_290866

variable {X Y Z W : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace W]
variable {XY XZ XW YW ZW : ℝ}

noncomputable def XW : ℝ := (42 / real.sqrt 7)

theorem find_XW (h1 : XY = 15) (h2 : XZ = 26) (h3 : ∃ (W : X → YZ), is_perpendicular (X, YZ)) (h4 : YW / ZW = 3 / 4) :
  XW = 42 / real.sqrt 7 :=
sorry

end find_XW_l290_290866


namespace evaluate_statements_l290_290332

variable {α : Type} [LinearOrderedField α] [DecidableEq α]
variable (a b c : α)

def condition1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ ¬(a = b ∧ b = c)

def statement1 : Prop := (a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0
def statement2 : Prop := (a > b) ∨ (a < b) ∨ (a = b)
def statement3 : Prop := ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b)

theorem evaluate_statements (h : condition1) : statement1 ∧ statement2 ∧ ¬statement3 :=
by
  sorry

end evaluate_statements_l290_290332


namespace complex_quadrant_l290_290858

theorem complex_quadrant (x y: ℝ) (h : x = 1 ∧ y = 2) : x > 0 ∧ y > 0 :=
by
  sorry

end complex_quadrant_l290_290858


namespace squares_problem_l290_290676

-- Define the concept of a square
structure Square (P Q R S : Type) := 
  (is_square : true) -- Placeholder for square property

-- Define the concept of larger_square containing smaller squares and points
structure LargerSquare (A B C D P Q R S : Type) :=
  (small_square : Square P Q R S)
  (A_on_extension_PQ : true) -- Placeholder for condition
  (B_on_extension_QR : true) -- Placeholder for condition
  (C_on_extension_RS : true) -- Placeholder for condition
  (D_on_extension_SP : true) -- Placeholder for condition

-- The main problem statement
theorem squares_problem (A B C D P Q R S : Type) 
  (large_square : LargerSquare A B C D P Q R S) : 
  (AC = BD) ∧ (AC ⊥ BD) :=
by
  intros
  tauto -- Placeholder to show the absence of code surrounding the proof.
  sorry

end squares_problem_l290_290676


namespace probability_abs_diff_gt_half_l290_290935

noncomputable def coin_flip := (0, 0)     -- represents two heads
  | (1, 1)     -- represents two tails
  | [ (1, 0), (0, 1)]       -- represents one head followed by a tail or vice versa

-- Define the process of obtaining x and y
noncomputable def select_number (flip : (ℕ, ℕ)) : ℝ :=
  if flip = (0, 0) then 0
  else if flip = (1, 1) then 1
  else if flip = (1, 0) then (1 : ℝ) / 2 -- For simplicity, assume a fixed representative of the uniform interval
  else if flip = (0, 1) then (3 : ℝ) / 4 -- For simplicity, assume a fixed representative of the uniform interval
  else 0 -- This case should never happen

axiom uniform_independence (flip1 flip2 : (ℕ, ℕ)) : Prop           -- The flips are independent
axiom uniform_distribution (flip : (ℕ, ℕ)) : flip ∈ [(0,0), (1,1), (1,0), (0,1)] -- Each flip result has the uniform probability

noncomputable def x := select_number coin_flip
noncomputable def y := select_number coin_flip

theorem probability_abs_diff_gt_half : Pr (|x - y| > 1 / 2) = 1 / 8 := sorry

end probability_abs_diff_gt_half_l290_290935


namespace length_of_DE_l290_290425

-- Definitions for side lengths and angle
def BC : ℝ := 20 * Real.sqrt 3
def angle_C : ℝ := 30 -- in degrees

-- Midpoint calculation
def midpoint_length (length : ℝ) : ℝ := length / 2

-- Length of CD
def CD : ℝ := midpoint_length BC

-- Function to compute length DE in a 30-60-90 triangle
def length_DE (CD_length : ℝ) := CD_length / Real.sqrt 3

-- The main statement we aim to prove
theorem length_of_DE : ∀ (BC CD DE : ℝ) (angle_C : ℝ),
  BC = 20 * Real.sqrt 3 →
  angle_C = 30 →
  CD = midpoint_length BC →
  DE = length_DE CD →
  DE = 10 :=
by
  intros BC CD DE angle_C hBC h_angleC hCD hDE
  rw [hBC, h_angleC, hCD, hDE]
  sorry

end length_of_DE_l290_290425


namespace range_of_a_for_no_extreme_points_l290_290959

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x^2 + (a + 1) * x

theorem range_of_a_for_no_extreme_points : 
  (∀ x : ℝ, (f a x)' = 0 → false) ↔ (0 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_for_no_extreme_points_l290_290959


namespace part1_not_H1_seq_part2_find_t_and_geom_seq_part3_compare_ln_an_and_prove_ineq_l290_290399

variable {t : ℤ}
noncomputable def H_seq (a : ℕ → ℤ) (t : ℤ) : Prop :=
  ∀ n, a (n + 1) - ∏ i in range (n + 1), a (i + 1) = t

noncomputable def geom_seq (b : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * q

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in range n, a (i + 1)

theorem part1_not_H1_seq : ¬H_seq (fun n => [1, 2, 3, 8, 49].nth n.get_or_else 0) 1 := sorry

theorem part2_find_t_and_geom_seq
  (a : ℕ → ℤ) (b : ℕ → ℤ)
  (geom_b : geom_seq b 2)
  (a_1_eq_2 : a 1 = 2)
  (H_a_t : H_seq a t)
  (sum_a_sq_eq_log_b : ∀ n, ∑ i in range n, (a (i + 1)) ^ 2 = ∏ i in range n, a (i + 1) + Int.log 2 (b n))
  : t = -1 ∧ ∀ n, b n = 2^(n+1) := sorry

theorem part3_compare_ln_an_and_prove_ineq
  (a : ℕ → ℤ)
  (H_a_t : H_seq a t)
  (a1_gt_1 : a 1 > 1)
  (t_gt_0 : t > 0)
  (ln_an_lt_an_minus_1 : ∀ n, (Real.log (a (n + 1)) : ℝ) < (a (n + 1) : ℤ) - 1)
  : ∀ n, t > S a (n + 1) - S a n - Real.exp (S a n - n : ℝ) := sorry

end part1_not_H1_seq_part2_find_t_and_geom_seq_part3_compare_ln_an_and_prove_ineq_l290_290399


namespace triangle_AB_AP_BQ_perimeter_4_l290_290487

noncomputable def triangle_perimeter (A B C P Q I : Point) (incircle : Circle) : ℝ :=
  let AB := dist A B
  let AP := dist A P
  let BQ := dist B Q
  if AB = 1 ∧ AP = 1 ∧ BQ = 1 ∧
     (AQ ∩ BP).to_set = incircle.to_set then
    perimeter A B C
  else
    0

theorem triangle_AB_AP_BQ_perimeter_4
  (A B C P Q M : Point) (incircle : Circle) (hP_on_AC : on_line P A C)
  (hQ_on_BC : on_line Q B C)
  (hAB_eq_1 : dist A B = 1)
  (hAP_eq_1 : dist A P = 1)
  (hBQ_eq_1 : dist B Q = 1)
  (hM_on_incircle : M ∈ incircle) : perimeter A B C = 4 := by
  sorry

end triangle_AB_AP_BQ_perimeter_4_l290_290487


namespace extremum_points_sum_l290_290452

def f (x a : ℝ) : ℝ := x^2 - 4 * a * x + a * Real.log x

theorem extremum_points_sum (a x1 x2 : ℝ) (h1 : a > 1/2) (h2 : f' x1 a = 0) (h3 : f' x2 a = 0) (h4 : x1 ≠ x2) : 
  f x1 a + f x2 a < -2 := 
sorry

end extremum_points_sum_l290_290452


namespace average_age_union_l290_290582

theorem average_age_union (students_A students_B students_C : ℕ)
  (sumA sumB sumC : ℕ) (avgA avgB avgC avgAB avgAC avgBC : ℚ)
  (hA : avgA = (sumA : ℚ) / students_A)
  (hB : avgB = (sumB : ℚ) / students_B)
  (hC : avgC = (sumC : ℚ) / students_C)
  (hAB : avgAB = (sumA + sumB) / (students_A + students_B))
  (hAC : avgAC = (sumA + sumC) / (students_A + students_C))
  (hBC : avgBC = (sumB + sumC) / (students_B + students_C))
  (h_avgA: avgA = 34)
  (h_avgB: avgB = 25)
  (h_avgC: avgC = 45)
  (h_avgAB: avgAB = 30)
  (h_avgAC: avgAC = 42)
  (h_avgBC: avgBC = 36) :
  (sumA + sumB + sumC : ℚ) / (students_A + students_B + students_C) = 33 := 
  sorry

end average_age_union_l290_290582


namespace factorize_expr_l290_290757

theorem factorize_expr (a : ℝ) : a^2 - 8 * a = a * (a - 8) :=
sorry

end factorize_expr_l290_290757


namespace f_cos_eq_l290_290830

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given condition
axiom f_sin_eq : f (Real.sin x) = 3 - Real.cos (2 * x)

-- The statement we want to prove
theorem f_cos_eq : f (Real.cos x) = 3 + Real.cos (2 * x) := 
by
  sorry

end f_cos_eq_l290_290830


namespace list_price_of_article_l290_290120

theorem list_price_of_article (P : ℝ) 
  (first_discount second_discount final_price : ℝ)
  (h1 : first_discount = 0.10)
  (h2 : second_discount = 0.08235294117647069)
  (h3 : final_price = 56.16)
  (h4 : P * (1 - first_discount) * (1 - second_discount) = final_price) : P = 68 :=
sorry

end list_price_of_article_l290_290120


namespace find_x_value_l290_290187

theorem find_x_value (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
sorry

end find_x_value_l290_290187


namespace password_recovery_l290_290989

-- Define the alphabet and letter encoding
constant alphabet : Fin 32 → Char 

-- Define the ordinal number representation function
def ordinal_repr (c: Char) : Fin 32 :=
  if c = 'ё' then 5 else 
  ⟨(c.to_nat - 'а'.to_nat + 1) % 32, sorry⟩ -- Placeholder for underlying proof if needed

-- Define conditions
constant a b : ℕ
constant x0 : ℕ

-- Define the sequence generating function
def r10 (n: ℕ) : ℕ := n % 10

-- Define the original sequence of digits y_i representing password
constant y : List (Fin 32)

-- Define the generated sequence (x_i)
def x : ℕ → ℕ
| 0     := x0
| (n+1) := r10 (a * x n + b)

-- Define the computed sequence (c_i)
def c (n : ℕ) := r10 (x n + (ordinal_repr (alphabet (y.nth n).get_or_else 0)).val)

-- Given stored sequence c_i
constant c_stored : List ℕ := [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7]

-- Problem to prove: the recovered password is "яхта"
theorem password_recovery :
  let password := [⟨30, sorry⟩ /*я*/, ⟨21, sorry⟩ /*х*/, ⟨20, sorry⟩ /*т*/, ⟨01, sorry⟩ /*а*/]
  in y.take 4 = password ∧ y.drop 4 = password := 
sorry -- Proof part to be filled in

end password_recovery_l290_290989


namespace three_digit_integers_l290_290378

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≠ 4

def contains_two (n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def does_not_contain_four (n : ℕ) : Prop :=
  ∀ k, n / 10^k % 10 ≠ 4

theorem three_digit_integers : {n : ℕ | is_three_digit n ∧ contains_two n ∧ does_not_contain_four n}.card = 200 := by
  sorry

end three_digit_integers_l290_290378


namespace five_digit_palindromes_count_l290_290735

theorem five_digit_palindromes_count : 
  let a_values := {a : ℕ | 1 ≤ a ∧ a ≤ 9}
  let b_values := {b : ℕ | 0 ≤ b ∧ b ≤ 9}
  let c_values := {c : ℕ | 0 ≤ c ∧ c ≤ 9}
  a_values.card * b_values.card * c_values.card = 900 := 
by 
  -- a has 9 possible values
  have a_card : a_values.card = 9 := sorry
  -- b has 10 possible values
  have b_card : b_values.card = 10 := sorry
  -- c has 10 possible values
  have c_card : c_values.card = 10 := sorry
  -- solve the multiplication
  sorry

end five_digit_palindromes_count_l290_290735


namespace train_length_l290_290688

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l290_290688


namespace jessica_candy_distribution_l290_290434

theorem jessica_candy_distribution :
  ∃ pieces_to_remove : ℕ,
  pieces_to_remove = 2 ∧ (30 - pieces_to_remove) % 4 = 0 := 
begin
  use 2,
  split,
  { refl, },
  { norm_num, },
end

end jessica_candy_distribution_l290_290434


namespace general_formulas_and_sum_l290_290856

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

-- Main statements translating the proof problem
theorem general_formulas_and_sum (n : ℕ) : 
  a 1 = 1 ∧ b 1 = 1 ∧ a 2 = b 2 ∧ 2 + a 4 = b 3 ∧
  (a n = 2 * n - 1) ∧ (b n = 3 ^ (n - 1)) ∧
  (∑ k in Finset.range n, a (k + 1) + b (k + 1)) = n^2 + (3^n - 1) / 2 := 
by
  -- Conditions to start the proof
  split,
  -- Assertions for initial conditions
  case a_1_eq_1
    exact rfl,
  case b_1_eq_1
    exact rfl,
  case a_2_eq_b_2
    -- Arithmetic and geometric sequence conditions here
    have a_eq_2k_minus_1 : ∀ k, a k = 2 * k - 1 := sorry,
    have b_eq_3k_minus_1 : ∀ k, b k = 3 ^ (k - 1) := sorry,
    show a 2 = b 2, by
      simp [a_eq_2k_minus_1, b_eq_3k_minus_1],
      sorry,
  case two_plus_a4_eq_b3
    show 2 + a 4 = b 3, by
      simp [a_eq_2k_minus_1, b_eq_3k_minus_1],
      sorry,
  case general_formulas
    split,
    -- Formula for a_n
    show a n = 2 * n - 1, by
      simp [a_eq_2k_minus_1],
      sorry,
    -- Formula for b_n  
    show b n = 3 ^ (n - 1), by 
      simp [b_eq_3k_minus_1],
      sorry,
  case sum_formula
    show (∑ k in Finset.range n, a (k + 1) + b (k + 1)) = n^2 + (3^n - 1) / 2, by
      -- Handle the sum of the sequences here
      sorry

end general_formulas_and_sum_l290_290856


namespace royal_family_children_l290_290600

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290600


namespace angle_AOA_l290_290478

-- Definitions for the problem scenario
variable (l m : Line) (O A A' A'' : Point)
variable (θ : Real) 

-- Hypotheses
hypothesis (intersect_at_O : ∃ O, l intersects m at O)
hypothesis (A_reflected_over_l_to_A' : reflect_over A l = A')
hypothesis (A'_reflected_over_m_to_A'' : reflect_over A' m = A'')
hypothesis (angle_between_l_m : angle_between l m = θ)

-- Theorem stating the proof objective
theorem angle_AOA''_equals_2θ
  (intersect_at_O : ∃ O, l intersects m at O)
  (A_reflected_over_l_to_A' : reflect_over A l = A')
  (A'_reflected_over_m_to_A'' : reflect_over A' m = A'')
  (angle_between_l_m : angle_between l m = θ) :
  angle A O A'' = 2 * θ := by
  sorry

end angle_AOA_l290_290478


namespace royal_children_l290_290618

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290618


namespace ned_did_not_wash_10_items_l290_290917

theorem ned_did_not_wash_10_items :
  let short_sleeve_shirts := 9
  let long_sleeve_shirts := 21
  let pairs_of_pants := 15
  let jackets := 8
  let total_items := short_sleeve_shirts + long_sleeve_shirts + pairs_of_pants + jackets
  let washed_items := 43
  let not_washed_Items := total_items - washed_items
  not_washed_Items = 10 := by
sorry

end ned_did_not_wash_10_items_l290_290917


namespace magnitude_vector_AB_l290_290325

-- Define the points A and B
def A : ℝ × ℝ := (-1, -6)
def B : ℝ × ℝ := (2, -2)

-- Define the vector AB as the difference between B and A
def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude function for a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The theorem states that the magnitude of vector_AB is 5
theorem magnitude_vector_AB : magnitude vector_AB = 5 :=
sorry

end magnitude_vector_AB_l290_290325


namespace royal_children_count_l290_290613

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290613


namespace problem_k_bound_l290_290179

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Define the conditions of the problem
axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_one : f 1 = 2

-- Define the goal statement
theorem problem_k_bound (k : ℝ) : (∃ t > 2, f(k * log 2 t) + f((log 2 t)^2 - log 2 t - 2) < 0) → k < 2 :=
sorry

end problem_k_bound_l290_290179


namespace smaller_angle_measure_l290_290995

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l290_290995


namespace sum_of_smallest_and_largest_l290_290950

variable (m b z : ℤ) -- Defining variables m, b, and z as integers

-- Conditions stated as definitions
def even (n : ℤ) : Prop := ∃ k, n = 2 * k -- Define what it means to be even

def mean_calculation (m b z : ℤ) : Prop := 
  even m ∧ z = b + m - 1

-- Lean statement for the proof problem
theorem sum_of_smallest_and_largest (m b z : ℤ) 
  (hm : mean_calculation m b z) : 2 * z = b + (b + 2 * (m - 1)) :=
begin
  sorry -- Proof not required
end

end sum_of_smallest_and_largest_l290_290950


namespace percentage_greater_than_88_l290_290402

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h : x = 88 + percentage * 88) (hx : x = 132) : 
  percentage = 0.5 :=
by
  sorry

end percentage_greater_than_88_l290_290402


namespace sum_of_distances_equals_twice_radius_and_inradius_l290_290504

variables {A B C M : Point} [Triangle A B C]

-- Assumptions
def orthocenter (M : Point) (ABC : Triangle) : Prop :=
  ∀ (X Y Z : Point), 
    (Altitude A B C = X) ∧ 
    (Altitude B A C = Y) ∧ 
    (Altitude C A B = Z) ∧ 
    (Intersection X Y Z = M)

def circumradius (ABC : Triangle) : ℝ := sorry
def inradius (ABC : Triangle) : ℝ := sorry

-- Condition definitions
axiom M_is_orthocenter : orthocenter M (Triangle.mk A B C)
axiom R_is_circumradius : circumradius (Triangle.mk A B C) = R
axiom r_is_inradius : inradius (Triangle.mk A B C) = r

-- Proof statement
theorem sum_of_distances_equals_twice_radius_and_inradius
  (A B C M : Point) [Triangle A B C]
  (M_is_orthocenter : orthocenter M (Triangle.mk A B C))
  (R_is_circumradius : circumradius (Triangle.mk A B C) = R)
  (r_is_inradius : inradius (Triangle.mk A B C) = r) :
  distance A M + distance B M + distance C M = 2 * R + 2 * r :=
sorry

end sum_of_distances_equals_twice_radius_and_inradius_l290_290504


namespace distance_between_incircle_and_excircle_centers_l290_290894

noncomputable def distance_between_circle_centers (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z] (XY XZ YZ : ℝ) (incircle_center excircle_center : X) : ℝ :=
  let side_lengths := (XY, XZ, YZ) in
  match side_lengths with
  | (15, 20, 25) => dist incircle_center excircle_center
  | _ => 0

theorem distance_between_incircle_and_excircle_centers :
  let X Y Z : Type := ℝ,
      XY : ℝ := 15,
      XZ : ℝ := 20,
      YZ : ℝ := 25,
      (incircle_center, excircle_center) : ℝ × ℝ := (5, 30)
  in
  distance_between_circle_centers X Y Z XY XZ YZ incircle_center excircle_center = 25 :=
by
  sorry

end distance_between_incircle_and_excircle_centers_l290_290894


namespace constant_term_l290_290514

theorem constant_term (x : ℝ) (h : x ≠ 0) :
  let T (r : ℕ) := Nat.choose 8 r * (-2) ^ r * x ^ (8 - 3 * r / 2)
  (1 + x) * (x - 2 / (sqrt x)) ^ 8 = 1792 := 
by
  sorry

end constant_term_l290_290514


namespace compounded_interest_correct_l290_290760

noncomputable def continuously_compounded_interest (P : ℝ) (r1 r2 r3 r4 r5 : ℝ) : ℝ :=
  let A1 := P * Real.exp(r1)
  let A2 := A1 * Real.exp(r2)
  let A3 := A2 * Real.exp(r3)
  let A4 := A3 * Real.exp(r4)
  let A5 := A4 * Real.exp(r5)
  A5 - P

theorem compounded_interest_correct :
  continuously_compounded_interest 10000 0.04 0.03 0.05 0.06 0.02 ≈ 2218.27 := sorry

end compounded_interest_correct_l290_290760


namespace leonardo_initial_money_l290_290438

theorem leonardo_initial_money (chocolate_cost : ℝ) (borrowed_amount : ℝ) (needed_amount : ℝ)
  (h_chocolate_cost : chocolate_cost = 5)
  (h_borrowed_amount : borrowed_amount = 0.59)
  (h_needed_amount : needed_amount = 0.41) :
  chocolate_cost + borrowed_amount + needed_amount - (chocolate_cost - borrowed_amount) = 4.41 :=
by
  rw [h_chocolate_cost, h_borrowed_amount, h_needed_amount]
  norm_num
  -- Continue with the proof, eventually obtaining the value 4.41
  sorry

end leonardo_initial_money_l290_290438


namespace prob_9_in_decimal_rep_of_3_over_11_l290_290489

def decimal_rep_of_3_over_11 : List ℕ := [2, 7]  -- decimal representation of 3/11 is 0.272727...

theorem prob_9_in_decimal_rep_of_3_over_11 : 
  (1 / (2 : ℚ)) * (decimal_rep_of_3_over_11.count 9) = 0 := by
  have h : 9 ∉ decimal_rep_of_3_over_11 := by simp only [decimal_rep_of_3_over_11, List.mem_cons, List.mem_nil, not_false_iff]; exact dec_trivial
  rw List.count_eq_zero_of_not_mem h
  norm_num
  sorry

end prob_9_in_decimal_rep_of_3_over_11_l290_290489


namespace sk_eq_sl_l290_290462

variables {A B C X Y K L S : Type} [euclidean_geometry A B C X Y K L S]
variables (ω : circle A B C X Y)

-- Definitions of circle, points, and bisectors
variables (circumcircle : ω.circumcircle ABC)
variables (bisector_BC : is_internal_angle_bisector ω (AB B C))
variables (bisector_AC : is_internal_angle_bisector ω (AC A B))

-- Definitions of points on the circle
variables (X_on_circle : ω.on_circle X) (Y_on_circle : ω.on_circle Y)
variables (X_ne_B : X ≠ B) (Y_ne_C : Y ≠ C)

-- Definitions of angle conditions
variables (K_on_CX : CX.contains K) (L_on_BY : BY.contains L)
variables (angle_KAC : angle K A C = 90) (angle_LAB : angle L A B = 90)

-- Definition midpoint of arc
variables (midpoint_arc_S : ω.midpoint_arc_CAB S)

-- The main theorem
theorem sk_eq_sl : dist S K = dist S L :=
sorry

end sk_eq_sl_l290_290462


namespace inequality_inequality_l290_290383

theorem inequality_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b) ^ 2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b) ^ 2 / (8 * b) :=
sorry

end inequality_inequality_l290_290383


namespace royal_children_count_l290_290626

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290626


namespace isosceles_trapezoid_area_l290_290339

theorem isosceles_trapezoid_area (m h : ℝ) (hg : h = 3) (mg : m = 15) : 
  (m * h = 45) :=
by
  simp [hg, mg]
  sorry

end isosceles_trapezoid_area_l290_290339


namespace product_of_two_numbers_l290_290561

theorem product_of_two_numbers (x y : ℚ) 
  (h1 : x + y = 8 * (x - y)) 
  (h2 : x * y = 15 * (x - y)) : 
  x * y = 100 / 7 := 
by 
  sorry

end product_of_two_numbers_l290_290561


namespace min_value_f_min_value_achieved_l290_290265

noncomputable def f (x y : ℝ) : ℝ :=
  (x^4 / y^4) + (y^4 / x^4) - (x^2 / y^2) - (y^2 / x^2) + (x / y) + (y / x)

theorem min_value_f :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → f x y ≥ 2 :=
sorry

theorem min_value_achieved :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → (f x y = 2) ↔ (x = y) :=
sorry

end min_value_f_min_value_achieved_l290_290265


namespace f_le_2x_f_not_le_1_9x_l290_290908

-- Define the function f and conditions
def f : ℝ → ℝ := sorry

axiom non_neg_f : ∀ x, 0 ≤ x → 0 ≤ f x
axiom f_at_1 : f 1 = 1
axiom f_additivity : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Proof for part (1): f(x) ≤ 2x for all x in [0, 1]
theorem f_le_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x := 
by
  sorry

-- Part (2): The inequality f(x) ≤ 1.9x does not hold for all x
theorem f_not_le_1_9x : ¬ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 1.9 * x) := 
by
  sorry

end f_le_2x_f_not_le_1_9x_l290_290908


namespace countIntegersLessThan1000_l290_290004

open Nat

def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem countIntegersLessThan1000 :
  (Finset.filter (λ n : ℕ, n ≤ 999 ∧ n > 0 ∧ n = 7 * sumOfDigits n) (Finset.range 1000)).card = 2 :=
by
  sorry

end countIntegersLessThan1000_l290_290004


namespace length_DE_l290_290512

theorem length_DE (ABC : Triangle) (A B C : Point)
  (h_base : ABC.base_length = 15)
  (h_creasw_parallel : ABC.crease_parallel_DE)
  (h_area_fraction : ABC.area_fraction_below_base = 0.25) :
  ABC.length_DE = 7.5 :=
sorry

end length_DE_l290_290512


namespace scientist_born_on_saturday_l290_290948

noncomputable def day_of_week := List String

noncomputable def calculate_day := 
  let days_in_regular_years := 113
  let days_in_leap_years := 2 * 37
  let total_days_back := days_in_regular_years + days_in_leap_years
  total_days_back % 7

theorem scientist_born_on_saturday :
  let anniversary_day := 4  -- 0=Sunday, 1=Monday, ..., 4=Thursday
  calculate_day = 5 → 
  let birth_day := (anniversary_day + 7 - calculate_day) % 7 
  birth_day = 6 := sorry

end scientist_born_on_saturday_l290_290948


namespace three_digit_integers_with_at_least_one_two_but_no_four_l290_290374

-- Define the properties
def is_three_digit (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000
def contains_digit (n: ℕ) (d: ℕ) : Prop := ∃ i, i < 3 ∧ d = n / 10^i % 10
def no_four (n: ℕ) : Prop := ¬ contains_digit n 4

-- Define the sets A and B
def setA (n: ℕ) : Prop := is_three_digit n ∧ no_four n
def setB (n: ℕ) : Prop := setA n ∧ ¬ contains_digit n 2

-- The final theorem statement
theorem three_digit_integers_with_at_least_one_two_but_no_four : 
  {n : ℕ | contains_digit n 2 ∧ setA n}.card = 200 :=
sorry

end three_digit_integers_with_at_least_one_two_but_no_four_l290_290374


namespace base_8_numbers_with_6_or_7_l290_290002

theorem base_8_numbers_with_6_or_7 : 
  ∃ (count : ℕ), count = 128 - 72 ∧
  ∀ n ∈ (finset.range 128).filter (λ n, 
    let digits := nat.digits 8 n in
    digits.contains 6 ∨ digits.contains 7), 
  count = 56 := 
by 
  sorry

end base_8_numbers_with_6_or_7_l290_290002


namespace differential_eq_solution_initial_condition_l290_290763

noncomputable def particular_solution (x : ℝ) : ℝ := 
  sqrt (1 + 2 * log ((1 + exp x) / 2))

theorem differential_eq_solution_initial_condition :
  ∃ y : ℝ → ℝ, (∀ x : ℝ, (1 + exp x) * y x * (y x)' = exp x) ∧ (y 0 = 1) ∧ (∀ x, y x = sqrt (1 + 2 * log ((1 + exp x) / 2))) :=
begin
  use particular_solution,
  sorry
end

end differential_eq_solution_initial_condition_l290_290763


namespace sum_of_4th_and_12th_term_l290_290412

variable (a d : ℝ)

-- Definition of the nth term in an arithmetic progression
def arithmetic_term (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Definition of the sum of the first n terms in an arithmetic progression
def arithmetic_sum (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable (h_sum_first_15_terms : arithmetic_sum a d 15 = 225)

-- Theorem to prove
theorem sum_of_4th_and_12th_term : arithmetic_term a d 4 + arithmetic_term a d 12 = 30 :=
  sorry

end sum_of_4th_and_12th_term_l290_290412


namespace train_crossing_time_l290_290173

def length_of_train : ℕ := 120
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 660

def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℕ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℕ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 52 :=
sorry

end train_crossing_time_l290_290173


namespace floor_sqrt_five_minus_three_l290_290241

theorem floor_sqrt_five_minus_three : ⌊real.sqrt 5 - 3⌋ = -1 := by
  sorry

end floor_sqrt_five_minus_three_l290_290241


namespace weighted_average_germination_rate_correct_l290_290768

noncomputable def total_seeds : ℕ := 300 + 200 + 500 + 400 + 600
noncomputable def total_germinated_seeds : ℕ := (0.25 * 300) + (0.40 * 200) + (0.30 * 500) + (0.35 * 400) + (0.20 * 600)
noncomputable def weighted_average_germination_rate : ℝ := (total_germinated_seeds / total_seeds) * 100

theorem weighted_average_germination_rate_correct :
  weighted_average_germination_rate = 28.25 := by
  sorry

end weighted_average_germination_rate_correct_l290_290768


namespace find_smallest_w_l290_290585

theorem find_smallest_w (w : ℕ) (h : 0 < w) : 
  (∀ k, k = 2^5 ∨ k = 3^3 ∨ k = 12^2 → (k ∣ (936 * w))) ↔ w = 36 := by 
  sorry

end find_smallest_w_l290_290585


namespace donation_amount_per_person_l290_290407

theorem donation_amount_per_person (m n : ℕ) 
  (h1 : m + 11 = n + 9) 
  (h2 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (m + 11)) 
  (h3 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (n + 9))
  : ∃ k : ℕ, k = 25 ∨ k = 47 :=
by
  sorry

end donation_amount_per_person_l290_290407


namespace three_digit_numbers_with_2_without_4_l290_290377

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l290_290377


namespace polynomial_division_quotient_l290_290764

theorem polynomial_division_quotient :
  polynomial.quotient (x^5 + 8) (x + 2) = x^4 - 2x^3 + 4x^2 - 8x + 16 :=
sorry

end polynomial_division_quotient_l290_290764


namespace sufficient_but_not_necessary_l290_290169

theorem sufficient_but_not_necessary (x : ℝ) :
  (x^2 > 1) → (1 / x < 1) ∧ ¬(1 / x < 1 → x^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_l290_290169


namespace percentage_of_kittens_is_67_l290_290482

def number_of_cats := 6
def half_of_cats := number_of_cats / 2
def kittens_per_female := 7
def total_kittens := half_of_cats * kittens_per_female
def kittens_sold := 9
def kittens_remaining := total_kittens - kittens_sold
def remaining_total_cats := number_of_cats + kittens_remaining
def percentage_kittens := (kittens_remaining / remaining_total_cats) * 100

theorem percentage_of_kittens_is_67 :
  round percentage_kittens = 67 :=
sorry

end percentage_of_kittens_is_67_l290_290482


namespace terminal_side_angle_set_l290_290128

noncomputable def angle_set := { α : ℝ | ∃ k : ℤ, α = k * real.pi + real.pi / 3 }

theorem terminal_side_angle_set :
  ∀ α : ℝ, (∃ k : ℤ, α = k * real.pi + real.pi / 3) ↔ α ∈ angle_set :=
by sorry

end terminal_side_angle_set_l290_290128


namespace area_of_triangle_l290_290497

def point_in_triangle (A B C O : ℝ × ℝ) : Prop :=
  isosceles_right_triangle A B C ∧
  dist O A = 6 ∧ dist O B = 9 ∧ dist O C = 3

theorem area_of_triangle (A B C O : ℝ × ℝ) (h : point_in_triangle A B C O) :
  ∃ (area : ℝ), area = (45 / 2) + 9 * √2 :=
sorry

end area_of_triangle_l290_290497


namespace license_plate_palindrome_l290_290074

theorem license_plate_palindrome :
  let p_digit_palindrome := (1 : ℚ) / 100,
      p_letter_palindrome := (1 : ℚ) / 676,
      combined_probability := p_digit_palindrome + p_letter_palindrome - p_digit_palindrome * p_letter_palindrome,
      m := 31,
      n := 2704 in
  combined_probability = (m : ℚ) / n ∧ m + n = 2735 :=
by
  -- The assumptions and definitions as stated suffice to skip the actual proof here.
  sorry

end license_plate_palindrome_l290_290074


namespace problem_statement_l290_290802

variable (α : ℝ)

theorem problem_statement
  (h₀ : 0 < α)
  (h₁ : α < π / 2)
  (h₂ : cos α - sin α = -sqrt 5 / 5) :
  (sin α * cos α = 2 / 5) ∧
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (2 * sin α * cos α - cos α + 1) / (1 - tan α) = (-9 + sqrt 5) / 5 :=
by
  sorry

end problem_statement_l290_290802


namespace polynomial_remainder_l290_290573

theorem polynomial_remainder (z : ℂ) :
  let dividend := 4*z^3 - 5*z^2 - 17*z + 4
  let divisor := 4*z + 6
  let quotient := z^2 - 4*z + (1/4 : ℝ)
  let remainder := 5*z^2 + 6*z + (5/2 : ℝ)
  dividend = divisor * quotient + remainder := sorry

end polynomial_remainder_l290_290573


namespace C_investment_value_is_correct_l290_290697

noncomputable def C_investment_contribution 
  (A_investment B_investment total_profit A_profit_share : ℝ) : ℝ :=
  let C_investment := 
    (A_profit_share * (A_investment + B_investment) - A_investment * total_profit) / 
    (total_profit - A_profit_share)
  C_investment

theorem C_investment_value_is_correct : 
  C_investment_contribution 6300 4200 13600 4080 = 10500 := 
by
  unfold C_investment_contribution
  norm_num
  sorry

end C_investment_value_is_correct_l290_290697


namespace train_length_250_05_l290_290694

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l290_290694


namespace variance_comparison_l290_290444

variables (x1 x2 x3 x4 x5 : ℝ)
variables (ξ1 ξ2 : fin 5 → ℝ)

-- Conditions
axiom h1 : 10 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 ≤ 10^4
axiom h2 : x5 = 10^5
axiom h3 : ∀ i, (ξ1 i = x1 ∨ ξ1 i = x2 ∨ ξ1 i = x3 ∨ ξ1 i = x4 ∨ ξ1 i = x5) ∧ ξ1 i = 0.2
axiom h4 : ∀ i, (ξ2 i = (x1 + x2) / 2 ∨ ξ2 i = (x2 + x3) / 2 ∨ ξ2 i = (x3 + x4) / 2 ∨ ξ2 i = (x4 + x5) / 2 ∨ ξ2 i = (x5 + x1) / 2) ∧ ξ2 i = 0.2

-- Variances definition
noncomputable def D (ξ : fin 5 → ℝ) : ℝ :=
  0.2 * (finset.univ.sum (λ i : fin 5, (ξ i - (0.2 * finset.univ.sum (λ j : fin 5, ξ j))) ^ 2))

-- Proof problem
theorem variance_comparison : D ξ1 > D ξ2 :=
sorry

end variance_comparison_l290_290444


namespace number_of_children_l290_290630

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290630


namespace sum_of_coefficients_l290_290767

theorem sum_of_coefficients (A B C D : ℤ) :
  (∀ x : ℤ, (x - 3) * (2 * x^2 + 3 * x - 4) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = -2 := 
begin
  sorry
end

end sum_of_coefficients_l290_290767


namespace incorrect_statements_l290_290577

noncomputable def A_statement (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) → f 0 = 0

noncomputable def B_statement : Prop :=
  (∀ f : ℝ → ℝ, set.Icc (-2 : ℝ) 2 ⊆ set.univ) →
  (∀ (x : ℝ), x ∈ set.Icc (-(1 / 2) : ℝ) (3 / 2) ↔ (2 * x - 1) ∈ set.Icc (-2 : ℝ) 2)

noncomputable def C_statement (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 1 < x ∧ x ≤ y ∧ y ≤ 2 → f x ≤ f y)
  → ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≤ f y

noncomputable def D_statement (a : ℝ) : Prop :=
  (∀ x, x > 1 → (x^2 - 2 * x + a) ≥ (x - 2) * x) ∧
  (∀ x, x ≤ 1 → ((1 - 2 * a) * x - 1) ≥ (-(a + 1)) * x) →
  a < 1 / 2 ∧ a ≥ 1 / 3

theorem incorrect_statements :
  ∃ (f : ℝ → ℝ) (a : ℝ),
  ¬ A_statement f ∧ ¬ C_statement f ∧ ¬ D_statement a :=
by sorry

end incorrect_statements_l290_290577


namespace translate_A_to_B_l290_290417

def point : Type := ℤ × ℤ

def translate_right (p : point) (d : ℤ) : point :=
  (p.1 + d, p.2)

def translate_up (p : point) (d : ℤ) : point :=
  (p.1, p.2 + d)

theorem translate_A_to_B :
  let A : point := (-1, 4)
  let B : point := (4, 7)
  translate_up (translate_right A 5) 3 = B :=
by
  let A : point := (-1, 4)
  let B : point := (4, 7)
  have h1 : translate_right A 5 = (4, 4) := rfl
  have h2 : translate_up (4, 4) 3 = B := rfl
  rw [h1, h2]

end translate_A_to_B_l290_290417


namespace chess_group_players_l290_290986

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
by
  sorry

end chess_group_players_l290_290986


namespace polynomial_division_l290_290775

noncomputable def poly1 : Polynomial ℤ := Polynomial.X ^ 13 - Polynomial.X + 100
noncomputable def poly2 : Polynomial ℤ := Polynomial.X ^ 2 + Polynomial.X + 2

theorem polynomial_division : ∃ q : Polynomial ℤ, poly1 = poly2 * q :=
by 
  sorry

end polynomial_division_l290_290775


namespace general_term_l290_290357

open Nat

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 1 / 3 ∧ 
  ∀ n, n ≥ 2 → a n * a (n - 1) + 2 * a n * a (n + 1) = 3 * a (n - 1) * a (n + 1)

theorem general_term (a : ℕ → ℝ) (h : sequence a) : 
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end general_term_l290_290357


namespace total_people_in_cars_by_end_of_race_l290_290982

-- Define the initial conditions and question
def initial_num_cars : ℕ := 20
def initial_num_passengers_per_car : ℕ := 2
def initial_num_drivers_per_car : ℕ := 1
def extra_passengers_per_car : ℕ := 1

-- Define the number of people per car initially
def initial_people_per_car : ℕ := initial_num_passengers_per_car + initial_num_drivers_per_car

-- Define the number of people per car after gaining extra passenger
def final_people_per_car : ℕ := initial_people_per_car + extra_passengers_per_car

-- The statement to be proven
theorem total_people_in_cars_by_end_of_race : initial_num_cars * final_people_per_car = 80 := by
  -- Prove the theorem
  sorry

end total_people_in_cars_by_end_of_race_l290_290982


namespace royal_children_count_l290_290621

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290621


namespace children_count_l290_290595

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290595


namespace vasya_correct_l290_290929

noncomputable def consecutive_pairs := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6) : list (ℕ × ℕ)]

theorem vasya_correct : 
  ∀ (n : fin 7 → fin 7), 
  (∀ i : fin 6, i + 1 = n i) →
  ∃ (f1 f2 f3 f4 : fin 6), 
  (f1 ≠ f3) ∧ (f2 ≠ f4) ∧
  (list.has_mem (min n.to_fun n.to_fun) consecutive_pairs) ∧ 
  (list.has_mem (max n.to_fun n.to_fun) consecutive_pairs) 
  ∧ (f1 ≠ f2) ∧ (f3 ≠ f4) := 
sorry

end vasya_correct_l290_290929


namespace locus_definition_error_l290_290168

theorem locus_definition_error (A B C D E : String) :
  (A = "Every point satisfying the conditions is on the locus, and every point on the locus satisfies the conditions." ∧
   B = "Every point not on the locus satisfies the conditions, and every point satisfying the conditions is on the locus." ∧
   C = "No point not on the locus satisfies the conditions, and every point on the locus satisfies the conditions." ∧
   D = "No point on the locus fails to satisfy the conditions, and no point off the locus satisfies the conditions." ∧
   E = "Every point on the locus satisfies the conditions, and no point not satisfying the conditions is on the locus.") →
  ∃ A_error : Prop, 
  (A_error = ¬B) := sorry

end locus_definition_error_l290_290168


namespace sum_even_numbers_1_to_10_l290_290163

theorem sum_even_numbers_1_to_10 : (∑ n in (Finset.filter (λ n, n % 2 = 0) (Finset.range 11)), n) = 30 :=
by
  sorry

end sum_even_numbers_1_to_10_l290_290163


namespace solveNumberOfWaysToChooseSeats_l290_290079

/--
Define the problem of professors choosing their seats among 9 chairs with specific constraints.
-/
noncomputable def numberOfWaysToChooseSeats : ℕ :=
  let totalChairs := 9
  let endChairChoices := 2 * (7 * (7 - 2))  -- (2 end chairs, 7 for 2nd prof, 5 for 3rd prof)
  let middleChairChoices := 7 * (6 * (6 - 2))  -- (7 non-end chairs, 6 for 2nd prof, 4 for 3rd prof)
  endChairChoices + middleChairChoices

/--
The final result should be 238
-/
theorem solveNumberOfWaysToChooseSeats : numberOfWaysToChooseSeats = 238 := by
  sorry

end solveNumberOfWaysToChooseSeats_l290_290079


namespace largest_integer_l290_290182

theorem largest_integer (n : ℕ) : n ^ 200 < 5 ^ 300 → n <= 11 :=
by
  sorry

end largest_integer_l290_290182


namespace number_of_digits_of_x_l290_290828

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem number_of_digits_of_x (x : ℝ) (h : Real.log 2 (Real.log 2 (Real.log 2 x)) = 3) :
  let d := Real.log10 x
  77 ≤ d ∧ d < 78 :=
by
  sorry

end number_of_digits_of_x_l290_290828


namespace relationship_among_y_values_l290_290931

theorem relationship_among_y_values (c y1 y2 y3 : ℝ) :
  (-1)^2 - 2 * (-1) + c = y1 →
  (3)^2 - 2 * 3 + c = y2 →
  (5)^2 - 2 * 5 + c = y3 →
  y1 = y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  sorry

end relationship_among_y_values_l290_290931


namespace inequality_holds_for_all_x_l290_290776

variable (p : ℝ)
variable (x : ℝ)

theorem inequality_holds_for_all_x (h : -3 < p ∧ p < 6) : 
  -9 < (3*x^2 + p*x - 6) / (x^2 - x + 1) ∧ (3*x^2 + p*x - 6) / (x^2 - x + 1) < 6 := by
  sorry

end inequality_holds_for_all_x_l290_290776


namespace smallest_a_exists_l290_290450

theorem smallest_a_exists : ∃ (a : ℕ), a > 0 ∧ 
  (∀ P : ℤ[X], (P.eval 1 = a) ∧ (P.eval 3 = a) ∧ (P.eval 5 = a) ∧ (P.eval 7 = a) ∧ 
               (P.eval 2 = -a) ∧ (P.eval 4 = -a) ∧ (P.eval 6 = -a) ∧ (P.eval 8 = -a) ∧ (P.eval 10 = -a)) 
  → a = 945 := 
by
  -- proof to be filled here
  sorry

end smallest_a_exists_l290_290450


namespace travel_time_from_C_to_A_l290_290485

-- Assume the distances and timings
variables (d1 d2 d3 : ℝ) -- distances in km
variables (t1 t2 : ℝ) -- times in minutes

-- Assume velocities of boat and current
variables (v_b v_c : ℝ) -- speeds in km/min

-- Define distances
def dist_A_confluence := 1
def dist_B_confluence := 1
def dist_C_confluence := 2

-- Define travel times
def time_AB := 30
def time_BC := 18

-- Define conditions for velocities
noncomputable def travel_time := sorry

-- Define time taken from C to A
def time_CA (v_b v_c : ℝ) : ℝ :=
  let time_downstream := 2 / (v_b + v_c) in
  let time_upstream := 1 / (v_b - v_c) in
  time_downstream + time_upstream

-- The theorem to prove
theorem travel_time_from_C_to_A  (v_b v_c : ℝ) 
  (h1 : d1 = 1) (h2 : d2 = 1) (h3 : d3 = 2)
  (h4 : t1 = 30) (h5 : t2 = 18)
  (h6 : time_CA v_b v_c = 24 ∨ time_CA v_b v_c = 72) 
  : exist [(λ (t : ℝ), t = 24) ∨ t = 72] sorry :=
begin
  -- Proof to be provided
  sorry,
end

end travel_time_from_C_to_A_l290_290485


namespace total_fat_served_l290_290660

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l290_290660


namespace inequality_solution_l290_290267

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) * (x + 1) / (x - 2)

theorem inequality_solution :
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | 2 < x ∧ x ≤ 8/3 } ∪ { x : ℝ | 4 ≤ x } :=
by sorry

end inequality_solution_l290_290267


namespace parabola_directrix_l290_290271

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end parabola_directrix_l290_290271


namespace kopecks_to_rubles_l290_290141

noncomputable def exchangeable_using_coins (total : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (x y z t u v w : ℕ), 
    total = x * 1 + y * 2 + z * 5 + t * 10 + u * 20 + v * 50 + w * 100 ∧ 
    num_coins = x + y + z + t + u + v + w

theorem kopecks_to_rubles (A B : ℕ)
  (h : exchangeable_using_coins A B) : exchangeable_using_coins (100 * B) A :=
sorry

end kopecks_to_rubles_l290_290141


namespace true_propositions_l290_290454

-- Definitions for planes and lines in space
variables (m n : Line) (α β : Plane)

-- Conditions as functions on the propositions
def prop1 : Prop := (m ∥ α) ∧ (m ∥ β) → (α ∥ β)
def prop2 : Prop := (m ⊥ α) ∧ (m ⊥ β) → (α ∥ β)
def prop3 : Prop := (m ∥ α) ∧ (n ∥ α) → (m ∥ n)
def prop4 : Prop := (m ⊥ α) ∧ (n ⊥ α) → (m ∥ n)

-- Theorem proving the truth of propositions 2 and 4, and disproving 1 and 3
theorem true_propositions : prop2 ∧ prop4 ∧ ¬prop1 ∧ ¬prop3 := by
  sorry

end true_propositions_l290_290454


namespace paul_cookies_batch_size_l290_290777

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def area_parallelogram (base height : ℝ) : ℝ := base * height

variable (roger_batch_cookies : ℕ) (roger_length roger_width paul_base paul_height : ℝ)

axiom roger_cookies : roger_length = 5 ∧ roger_width = 4
axiom paul_cookies : paul_base = 4 ∧ paul_height = 3
axiom roger_batch : roger_batch_cookies = 10

theorem paul_cookies_batch_size :
  ∀ roger_length roger_width paul_base paul_height : ℝ,
  roger_cookies →
  paul_cookies →
  roger_batch →
  let roger_total_area := (roger_batch_cookies : ℝ) * (area_rectangle roger_length roger_width) in
  let paul_cookie_area := area_parallelogram paul_base paul_height in
  let paul_batch_cookies := roger_total_area / paul_cookie_area in
  paul_batch_cookies = 17 :=
by
  sorry

end paul_cookies_batch_size_l290_290777


namespace ellipse_probability_l290_290501

-- Define that m is within the interval [1,5] 
def m_in_interval (m : ℝ) : Prop :=
  1 ≤ m ∧ m ≤ 5

-- Define the condition for the equation to represent an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  4 > m^2

-- Define the probability calculation function
def probability_of_event (event : ℝ → Prop) : ℝ :=
  (∫ x in set.Icc (1 : ℝ) 2, if event x then 1 else 0) / (∫ x in set.Icc (1 : ℝ) 5, 1)

-- The main statement to prove
theorem ellipse_probability : probability_of_event is_ellipse_with_foci_on_y_axis = 1 / 4 :=
  by sorry

end ellipse_probability_l290_290501


namespace children_count_l290_290593

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290593


namespace find_AB_in_right_triangle_l290_290844

theorem find_AB_in_right_triangle
  (A B C : Type)
  (angle_A_90 : ∠A = 90)
  (sin_B : sin B = 5/13)
  (AC : AC = 52) : 
  AB = 20 := 
by
  sorry

end find_AB_in_right_triangle_l290_290844


namespace root_of_polynomial_l290_290006

theorem root_of_polynomial (k : ℝ) (h : (3 : ℝ) ^ 4 + k * (3 : ℝ) ^ 2 + 27 = 0) : k = -12 :=
by
  sorry

end root_of_polynomial_l290_290006


namespace starting_number_is_100_l290_290549

theorem starting_number_is_100 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, k = 10 ∧ n = 1000 - (k - 1) * 100) :
  n = 100 := by
  sorry

end starting_number_is_100_l290_290549


namespace number_of_children_l290_290633

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290633


namespace circle_area_circumference_l290_290498

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_circumference :
  let C : (ℝ × ℝ) := (1, 1)
  let D : (ℝ × ℝ) := (8, 6)
  let diameter := dist (C.1) (C.2) (D.1) (D.2)
  let radius := diameter / 2
  let area := π * radius^2
  let circumference := 2 * π * radius
  area = 74 * π / 4 ∧ circumference = real.sqrt 74 * π :=
by
  -- prove the theorem
  sorry

end circle_area_circumference_l290_290498


namespace count_even_3_digit_integers_divisible_by_4_no_digit_4_l290_290001

theorem count_even_3_digit_integers_divisible_by_4_no_digit_4 : 
  let digits := [0, 1, 2, 3, 5, 6, 7, 8, 9] in
  let suitable_c := [0, 2, 6, 8] in
  let is_valid (n : ℕ) : Prop :=
    n >= 100 ∧ n < 1000 ∧ n % 2 = 0 ∧ n % 4 = 0 ∧ '4' ∉ n.digits 10 in
  (∃ (count : ℕ), (count = 88) ∧
    (count = List.length (List.filter is_valid (List.range 900)) - 100)) :=
by
  sorry

end count_even_3_digit_integers_divisible_by_4_no_digit_4_l290_290001


namespace sum_of_products_not_zero_l290_290852

def grid_25x25 := Fin 25 × Fin 25 → ℤ

def is_valid_value (n : ℤ) : Prop := n = 1 ∨ n = -1

def product_of_row (g : grid_25x25) (i : Fin 25) : ℤ :=
  ∏ j in Finset.univ, g (i, j)

def product_of_column (g : grid_25x25) (j : Fin 25) : ℤ :=
  ∏ i in Finset.univ, g (i, j)

def sum_of_products (g : grid_25x25) : ℤ :=
  (∑ j in Finset.univ, product_of_column g j) +
  (∑ i in Finset.univ, product_of_row g i)

theorem sum_of_products_not_zero (g : grid_25x25)
  (hv : ∀ p, is_valid_value (g p)) : sum_of_products g ≠ 0 := 
by
  sorry

end sum_of_products_not_zero_l290_290852


namespace periodic_difference_not_necessarily_periodic_l290_290208

theorem periodic_difference_not_necessarily_periodic (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 6) = g x) 
  (hh : ∀ x, h (x + 2 * Real.pi) = h x) : 
  ¬ ∃ p > 0, ∀ x, (g - h) (x + p) = (g - h) x :=
begin
  sorry,
end

end periodic_difference_not_necessarily_periodic_l290_290208


namespace number_of_ways_to_complete_journey_l290_290022

-- Definitions for conditions
def face : Type := Nat
def top_ring : list face := [1, 2, 3, 4, 5]
def bottom_ring : list face := [1, 2, 3, 4, 5]
def forbidden_moves (f : face) : face := f

-- Function to count the number of valid journeys
def count_journeys : Nat := 369

-- The theorem statement
theorem number_of_ways_to_complete_journey :
    count_journeys = 369 :=
by
  sorry

end number_of_ways_to_complete_journey_l290_290022


namespace royal_children_l290_290617

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290617


namespace Alice_max_odd_integers_l290_290699

theorem Alice_max_odd_integers (l : List ℕ) (h_len : l.length = 6) (h_prod : l.prod % 2 = 1) (h_sum : l.sum % 2 = 0) : l.count (λ n : ℕ, n % 2 = 1) = 6 :=
by
  sorry

end Alice_max_odd_integers_l290_290699


namespace max_value_expression_l290_290864

theorem max_value_expression (a b c d : ℕ) 
  (ha : a ∈ {1, 2, 3, 4})
  (hb : b ∈ {1, 2, 3, 4})
  (hc : c ∈ {1, 2, 3, 4})
  (hd : d ∈ {1, 2, 3, 4})
  (h_distinct : (a, b, c, d).val.nodup) :
  c * a^b - d ≤ 127 :=
sorry

end max_value_expression_l290_290864


namespace proof_problem_l290_290884

noncomputable def M : Set ℝ := { x | x ≥ 2 }
noncomputable def a : ℝ := Real.pi

theorem proof_problem : a ∈ M ∧ {a} ⊂ M :=
by
  sorry

end proof_problem_l290_290884


namespace f_of_5_eq_1_l290_290900

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_5_eq_1
    (h1 : ∀ x : ℝ, f (-x) = -f x)
    (h2 : ∀ x : ℝ, f (-x) + f (x + 3) = 0)
    (h3 : f (-1) = 1) :
    f 5 = 1 :=
sorry

end f_of_5_eq_1_l290_290900


namespace part_I_part_II_part_III_l290_290823

open Real

section Problem

noncomputable def a : ℕ → ℝ
| 1       := 1 / 4
| (n + 1) := 1 - b n

noncomputable def b : ℕ → ℝ
| 1       := 3 / 4
| (n + 1) := b n / ((1 - a n) * (1 + a n))

noncomputable def c (n : ℕ) : ℝ := 1 / (b n - 1)

-- a, b, c sequences are defined based on given conditions
def c_sequence_arithmetic := ∀ n : ℕ, c (n + 1) = c n - 1

def inequality_holds (a : ℝ) (n : ℕ) : Prop :=
  4 * a * (finset.range n).sum (λ i, a i * a (i + 1)) < b n

def a_condition (a : ℝ) : Prop := ∀ n : ℕ, inequality_holds a n

theorem part_I : b 1 = 3 / 4 ∧ b 2 = 4 / 5 ∧ b 3 = 5 / 6 ∧ b 4 = 6 / 7 :=
sorry

theorem part_II : c_sequence_arithmetic :=
sorry

theorem part_III : ∀ a : ℝ, a_condition a → a ≤ 1 :=
sorry

end Problem

end part_I_part_II_part_III_l290_290823


namespace brown_eyed_brunettes_count_l290_290256

-- Definitions based on problem conditions
def total_girls : ℕ := 60
def blue_eyed_blondes : ℕ := 16
def brunettes : ℕ := 36
def brown_eyed : ℕ := 25

-- Problem statement to prove the number of brown-eyed brunettes is 17
theorem brown_eyed_brunettes_count : 
  ∃ (brown_eyed_brunettes : ℕ), brown_eyed_brunettes = 17 := 
by 
  let blondes := total_girls - brunettes
  let brown_eyed_blondes := blondes - blue_eyed_blondes
  let brown_eyed_brunettes := brown_eyed - brown_eyed_blondes
  have h_brown_eyed_brunettes : brown_eyed_brunettes = 17, by sorry
  exact ⟨brown_eyed_brunettes, h_brown_eyed_brunettes⟩

end brown_eyed_brunettes_count_l290_290256


namespace part1_part2a_part2b_part3_l290_290476

def f (x : ℝ) := Real.log x

def g (a x : ℝ) := a * x + (a - 1) / x - 3

def φ (a x : ℝ) := f x + g a x

def h (x : ℝ) := f x * g 1 x

theorem part1 (x : ℝ) : g 2 (Real.exp x) = 0 ↔ x = -Real.log 2 ∨ x = 0 := by
  sorry

theorem part2a (a : ℝ) (ha : a ≤ 1) : ∀ x > 0, φ a x = (0 : ℝ) → x > 0 := by
  sorry

theorem part2b (a : ℝ) (ha : 1 < a) : ∀ x > (a - 1) / a, φ a x = (0 : ℝ) → x > (a - 1) / a := by
  sorry

theorem part3 : ∀ λ : ℤ, ¬ ∃ x : ℝ, 2 * λ ≥ h x := by
  sorry

end part1_part2a_part2b_part3_l290_290476


namespace number_of_people_in_group_l290_290511

theorem number_of_people_in_group :
  ∀ (N : ℕ), (75 - 35) = 5 * N → N = 8 :=
by
  intros N h
  sorry

end number_of_people_in_group_l290_290511


namespace royal_children_l290_290615

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290615


namespace seeds_per_flowerbed_l290_290927

theorem seeds_per_flowerbed (total_seeds flowerbeds : ℕ) (h1 : total_seeds = 32) (h2 : flowerbeds = 8) :
  total_seeds / flowerbeds = 4 :=
by {
  sorry
}

end seeds_per_flowerbed_l290_290927


namespace polynomial_inequality_solution_l290_290774

theorem polynomial_inequality_solution :
  { x : ℝ | x * (x - 5) * (x - 10)^2 > 0 } = { x : ℝ | 0 < x ∧ x < 5 ∨ 10 < x } :=
by
  sorry

end polynomial_inequality_solution_l290_290774


namespace monthly_average_growth_rate_price_reduction_for_profit_l290_290654

-- Part 1: Monthly average growth rate of sales volume
theorem monthly_average_growth_rate (x : ℝ) : 
  256 * (1 + x) ^ 2 = 400 ↔ x = 0.25 :=
by
  sorry

-- Part 2: Price reduction to achieve profit of $4250
theorem price_reduction_for_profit (m : ℝ) : 
  (40 - m - 25) * (400 + 5 * m) = 4250 ↔ m = 5 :=
by
  sorry

end monthly_average_growth_rate_price_reduction_for_profit_l290_290654


namespace pam_total_apples_l290_290493

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end pam_total_apples_l290_290493


namespace zog_words_count_l290_290924

-- Defining the number of letters in the Zoggian alphabet
def num_letters : ℕ := 6

-- Function to calculate the number of words with n letters
def words_with_n_letters (n : ℕ) : ℕ := num_letters ^ n

-- Definition to calculate the total number of words with at most 4 letters
def total_words : ℕ :=
  (words_with_n_letters 1) +
  (words_with_n_letters 2) +
  (words_with_n_letters 3) +
  (words_with_n_letters 4)

-- Theorem statement
theorem zog_words_count : total_words = 1554 := by
  sorry

end zog_words_count_l290_290924


namespace total_sheep_flock_l290_290078

-- Definitions and conditions based on the problem description
def crossing_rate : ℕ := 3 -- Sheep per minute
def sleep_duration : ℕ := 90 -- Duration of sleep in minutes
def sheep_counted_before_sleep : ℕ := 42 -- Sheep counted before falling asleep

-- Total sheep that crossed while Nicholas was asleep
def sheep_during_sleep := crossing_rate * sleep_duration 

-- Total sheep that crossed when Nicholas woke up
def total_sheep_after_sleep := sheep_counted_before_sleep + sheep_during_sleep

-- Prove the total number of sheep in the flock
theorem total_sheep_flock : (2 * total_sheep_after_sleep) = 624 :=
by
  sorry

end total_sheep_flock_l290_290078


namespace kitchen_upgrade_cost_l290_290556

def total_kitchen_upgrade_cost (num_knobs : ℕ) (cost_per_knob : ℝ) (num_pulls : ℕ) (cost_per_pull : ℝ) : ℝ :=
  (num_knobs * cost_per_knob) + (num_pulls * cost_per_pull)

theorem kitchen_upgrade_cost : total_kitchen_upgrade_cost 18 2.50 8 4.00 = 77.00 :=
  by
    sorry

end kitchen_upgrade_cost_l290_290556


namespace ratio_of_hours_l290_290142

theorem ratio_of_hours (x y z : ℕ) 
  (h1 : x + y + z = 157) 
  (h2 : z = y - 8) 
  (h3 : z = 56) 
  (h4 : y = x + 10) : 
  (y / gcd y x) = 32 ∧ (x / gcd y x) = 27 := 
by 
  sorry

end ratio_of_hours_l290_290142


namespace Jim_runs_total_distance_l290_290435

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end Jim_runs_total_distance_l290_290435


namespace parabola_vertex_coordinates_l290_290515

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, -x^2 + 15 ≥ -x^2 + 15 :=
by
  sorry

end parabola_vertex_coordinates_l290_290515


namespace tree_width_iff_contains_bramble_of_order_greater_than_l290_290902

open Classical

variables {G : Type} [Graph G]
variable {k : ℕ}

noncomputable def tree_width (G : Graph) : ℕ := sorry

def contains_bramble_of_order_greater_than (G : Graph) (k : ℕ) : Prop :=
  ∃ (B : set (set G.V)), 
    (∀ b ∈ B, ∃ (u v : G.V), u ≠ v ∧ u ∈ b ∧ v ∈ b) ∧
    (∀ (b1 b2 ∈ B), b1 ∩ b2 ≠ ∅ ∨ ∃ (u ∈ b1) (v ∈ b2), G.connected u v) ∧
    (∃ (X : set G.V), X.card > k ∧ ∀ b ∈ B, b ∩ X ≠ ∅)

theorem tree_width_iff_contains_bramble_of_order_greater_than 
    (G : Graph) (k : ℕ) (hk : 0 ≤ k) : 
    tree_width G ≥ k ↔ contains_bramble_of_order_greater_than G k := 
sorry

end tree_width_iff_contains_bramble_of_order_greater_than_l290_290902


namespace exists_nonzero_B_l290_290045

open Matrix

variable {n : ℕ} (hn : n ≥ 2)
variable {A : Matrix (Fin n) (Fin n) ℂ} (hA : rank A ≠ rank (A ⬝ A))

theorem exists_nonzero_B : 
  ∃ B : Matrix (Fin n) (Fin n) ℂ, B ≠ 0 ∧ A ⬝ B = 0 ∧ B ⬝ A = 0 ∧ B ⬝ B = 0 :=
sorry

end exists_nonzero_B_l290_290045


namespace divergence_part_a_divergence_part_b_l290_290270

-- Using noncomputable theory for divisions and square roots
noncomputable theory

-- Definitions for the divergence operator and the given conditions

-- Define vector field A for part (a)
def vecA_a (x y z : ℝ) : ℝ × ℝ × ℝ := 
  let r := (x^2 + y^2 + z^2).sqrt
  (x / r, y / r, z / r)

-- Define the divergence of vecA_a
def divA_a (x y z : ℝ) : ℝ := 
  (∂ (λ x, (vecA_a x y z).1) / ∂ x + 
   ∂ (λ y, (vecA_a x y z).2) / ∂ y + 
   ∂ (λ z, (vecA_a x y z).3) / ∂ z)

-- Define the mathematical statement to be proven for part (a)
theorem divergence_part_a (x y z : ℝ) : 
  let r := (x^2 + y^2 + z^2).sqrt 
  in divA_a x y z = 2 / r := sorry

-- Define vector field A for part (b)
def vecA_b (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let denom := (x^2 + y^2).sqrt
  (-x / denom, y / denom, z / denom)

-- Define the divergence of vecA_b
def divA_b (x y z : ℝ) : ℝ := 
  (∂ (λ x, (vecA_b x y z).1) / ∂ x + 
   ∂ (λ y, (vecA_b x y z).2) / ∂ y + 
   ∂ (λ z, (vecA_b x y z).3) / ∂ z)

-- Define the mathematical statement to be proven for part (b)
theorem divergence_part_b (x y z : ℝ) : 
  divA_b x y z = (x^2 - y^2) / (x^2 + y^2)^(3/2) := sorry

end divergence_part_a_divergence_part_b_l290_290270


namespace sum_of_prime_divisors_of_least_N_l290_290077

theorem sum_of_prime_divisors_of_least_N :
  ∃ (N : ℕ), 
    N % 16 = 0 ∧ 
    N % 15 = 0 ∧ 
    N % 14 = 0 ∧ 
    (∃ (x y z : ℕ), 0 < x ∧ x < y ∧ y < z ∧ z < 14 ∧ 
      N % x = 3 ∧ 
      N % y = 3 ∧ 
      N % z = 3) ∧
    Nat.sum (Nat.factors (Nat.lcm (16, Nat.lcm (15, 14)))) = 148 
    :=
sorry

end sum_of_prime_divisors_of_least_N_l290_290077


namespace Eddy_travel_time_l290_290257

-- Define the given conditions
def T_f : ℝ := 4
def D_AC : ℝ := 460
def D_AB : ℝ := 600
def speed_ratio : ℝ := 1.7391304347826086
def V_f : ℝ := D_AC / T_f
def V_e : ℝ := speed_ratio * V_f

-- State and prove the main theorem
theorem Eddy_travel_time : (D_AB / V_e) = 3 := 
by
  -- What follows would normally be the proof, but we skip it as per instructions.
  sorry

end Eddy_travel_time_l290_290257


namespace list_price_is_45_l290_290700

variables (x : ℝ)
variables (alice_commission bob_commission : ℝ)

def alice_selling_price := x - 15
def bob_selling_price := x - 25

def alice_commission := 0.10 * alice_selling_price
def bob_commission := 0.15 * bob_selling_price

theorem list_price_is_45
  (h : alice_commission = bob_commission) :
  x = 45 := by
  sorry

end list_price_is_45_l290_290700


namespace find_y_when_x_is_neg6_l290_290532

variables (x y : ℝ)

-- Define the initial conditions
def x_y_sum_is_60 := x + y = 60
def x_is_3y := x = 3 * y
def x_y_inverse_proportional := ∃ k, x * y = k

-- Define the constant k
def k_value := (45 : ℝ) * (15 : ℝ)
-- Theorem to prove y when x is -6
theorem find_y_when_x_is_neg6
  (hx_sum : x_y_sum_is_60 x y)
  (hx_3y : x_is_3y x y)
  (hx_inv : x_y_inverse_proportional x y) :
  y = -112.5 :=
begin
  sorry
end

end find_y_when_x_is_neg6_l290_290532


namespace oil_leak_l290_290235

theorem oil_leak (a b c : ℕ) (h₁ : a = 6522) (h₂ : b = 11687) (h₃ : c = b - a) : c = 5165 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end oil_leak_l290_290235


namespace solve_for_x_l290_290053

def star (a b : ℝ) : ℝ := (Real.sqrt (a + b)) / (Real.sqrt (a - b))

theorem solve_for_x (x : ℝ) (h : star x 20 = 3) : x = 25 :=
by
  unfold star at h
  sorry

end solve_for_x_l290_290053


namespace sum_of_largest_and_smallest_angles_l290_290540

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 7
noncomputable def c : ℝ := 8

theorem sum_of_largest_and_smallest_angles :
  ∀ (α β γ : ℝ),
    α + β + γ = 180 ∧
    a^2 = b^2 + c^2 - 2 * b * c * ℝ.cos α ∧
    b^2 = a^2 + c^2 - 2 * a * c * ℝ.cos β ∧
    c^2 = a^2 + b^2 - 2 * a * b * ℝ.cos γ →
    (α + γ = 120 ∨ β + γ = 120 ∨ α + β = 120) :=
by sorry

end sum_of_largest_and_smallest_angles_l290_290540


namespace smallest_period_and_range_l290_290350

open Real

def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + cos (π / 2 - 2 * x)

theorem smallest_period_and_range : (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧ (∀ y, 1 - sqrt 2 ≤ y ∧ y ≤ 1 + sqrt 2 ↔ ∃ x, f x = y) :=
by
  sorry

end smallest_period_and_range_l290_290350


namespace area_of_scalene_right_triangle_l290_290095

noncomputable def area_of_triangle_DEF (DE EF : ℝ) (h1 : DE > 0) (h2 : EF > 0) (h3 : DE / EF = 3) (h4 : DE^2 + EF^2 = 16) : ℝ :=
1 / 2 * DE * EF

theorem area_of_scalene_right_triangle (DE EF : ℝ) 
  (h1 : DE > 0)
  (h2 : EF > 0)
  (h3 : DE / EF = 3)
  (h4 : DE^2 + EF^2 = 16) :
  area_of_triangle_DEF DE EF h1 h2 h3 h4 = 2.4 :=
sorry

end area_of_scalene_right_triangle_l290_290095


namespace no_snow_probability_l290_290968

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l290_290968


namespace hank_total_spending_l290_290707

variables
  (cost_apples_per_dozen : ℕ) (cost_pears_per_dozen : ℕ) (cost_oranges_per_dozen : ℕ) (cost_grapes_per_dozen : ℕ)
  (discount_apples : ℚ) (discount_pears : ℚ) (discount_oranges : ℚ) (discount_grapes : ℚ)
  (dozen_apples : ℕ) (dozen_pears : ℕ) (dozen_oranges : ℕ) (dozen_grapes : ℕ)

def total_cost
  (cost_apples_per_dozen : ℕ)
  (cost_pears_per_dozen : ℕ)
  (cost_oranges_per_dozen : ℕ)
  (cost_grapes_per_dozen : ℕ)
  (discount_apples : ℚ)
  (discount_pears : ℚ)
  (discount_oranges : ℚ)
  (discount_grapes : ℚ)
  (dozen_apples : ℕ)
  (dozen_pears : ℕ)
  (dozen_oranges : ℕ)
  (dozen_grapes : ℕ) : ℚ :=
  let apples_cost := dozen_apples * cost_apples_per_dozen in
  let pears_cost := dozen_pears * cost_pears_per_dozen in
  let oranges_cost := dozen_oranges * cost_oranges_per_dozen in
  let grapes_cost := dozen_grapes * cost_grapes_per_dozen in
  let apples_total := apples_cost - (discount_apples * apples_cost) in
  let pears_total := pears_cost - (discount_pears * pears_cost) in
  let oranges_total := oranges_cost - (discount_oranges * oranges_cost) in
  let grapes_total := grapes_cost - (discount_grapes * grapes_cost) in
  apples_total + pears_total + oranges_total + grapes_total

theorem hank_total_spending :
  total_cost 40 50 30 60 0.10 0.05 0.15 0 14 18 10 8 = 2094 := by
  sorry

end hank_total_spending_l290_290707


namespace a_plus_b_l290_290347

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem a_plus_b (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 :=
by
  sorry

end a_plus_b_l290_290347


namespace derivative_of_power_l290_290086

theorem derivative_of_power (a : ℝ) (x : ℝ) (hx : 0 < x) : 
  deriv (λ x : ℝ, x ^ a) x = a * x ^ (a - 1) :=
sorry

end derivative_of_power_l290_290086


namespace royal_family_children_l290_290601

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290601


namespace no_all_positive_signs_possible_l290_290567

theorem no_all_positive_signs_possible (grid : matrix (fin 8) (fin 8) bool) :
(∀ i j, grid i j = tt) → false :=
by sorry

end no_all_positive_signs_possible_l290_290567


namespace fraction_area_of_smaller_octagon_l290_290220

open Real

-- Define the regular octagon ABCDEFGH and its properties
def is_regular_octagon (A B C D E F G H : Point) : Prop :=
  -- This part is a placeholder for the actual definition.
  -- We assume it includes properties such as:
  -- A-B, B-C, ..., H-A have equal lengths (side lengths of a regular octagon)
  -- All internal angles are 135 degrees
  sorry

-- In Lean, we can define the midpoints of the sides of the octagon
def midpoint (P Q : Point) : Point :=
  -- Placeholder for the actual midpoint calculation
  sorry

-- Define midpoints for each side of the larger octagon to form the smaller octagon
def is_smaller_octagon_subset_of_larger (A B C D E F G H R S T U V W X Y : Point) : Prop :=
  R = midpoint A B ∧ S = midpoint B C ∧ T = midpoint C D ∧ U = midpoint D E ∧
  V = midpoint E F ∧ W = midpoint F G ∧ X = midpoint G H ∧ Y = midpoint H A

-- Define the function to calculate the area of a regular octagon
noncomputable def area_octagon (A B C D E F G H : Point) (h : is_regular_octagon A B C D E F G H) : ℝ :=
  -- Placeholder for the actual area calculation
  sorry

-- Define our main theorem statement
theorem fraction_area_of_smaller_octagon {A B C D E F G H R S T U V W X Y : Point}
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : is_smaller_octagon_subset_of_larger A B C D E F G H R S T U V W X Y) :
  area_octagon R S T U V W X Y h2 = 1 / 2 * area_octagon A B C D E F G H h1 :=
sorry

end fraction_area_of_smaller_octagon_l290_290220


namespace pell_negative_solution_exists_l290_290470

def is_prime_of_the_form_4k_plus_1 (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k + 1

def is_legendre_symbol_equal_neg1 (a p : ℕ) [fact (nat.prime p)] : Prop :=
  Nat.legendreSym a p = -1

theorem pell_negative_solution_exists
  (r : ℕ) (hr : r = 2 ∨ odd r)
  (p : Fin r → ℕ)
  (hprime : ∀ i, Nat.Prime (p i) ∧ is_prime_of_the_form_4k_plus_1 (p i))
  (hleger : ∀ i j, i ≠ j → is_legendre_symbol_equal_neg1 (p i) (p j)) :
  ∃ x y : ℤ, x^2 - (∏ i, p i : ℕ) * (y^2) = -1 := by
  sorry

end pell_negative_solution_exists_l290_290470


namespace product_of_cosines_l290_290905

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (1, -2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def cos_theta (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b
  dot_product a b / (magnitude a * magnitude b)

theorem product_of_cosines (x : ℝ) (h : magnitude (vector_a x + vector_b) = real.sqrt 5) :
  (x = -3 ∨ x = 1) →
  (cos_theta (-3) * cos_theta 1 = real.sqrt 5 / 10) :=
by
  sorry

end product_of_cosines_l290_290905


namespace problem1_problem2_l290_290356

noncomputable def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)
noncomputable def condition1 (a : ℝ) : Prop := 
  a = -1 ∧ (∃ x, p x a ∨ q x)

noncomputable def condition2 (a : ℝ) : Prop :=
  ∀ x, ¬ p x a → ¬ q x

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : condition1 a) : -6 ≤ x ∧ x < -3 ∨ 1 < x ∧ x ≤ 12 := 
sorry

theorem problem2 (a : ℝ) (h₂ : condition2 a) : -4 ≤ a ∧ a ≤ -2 :=
sorry

end problem1_problem2_l290_290356


namespace train_length_250_05_l290_290693

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l290_290693


namespace complex_sum_499_l290_290911

noncomputable theory

open Complex

theorem complex_sum_499 (x : ℂ) (h1 : x ^ 1001 = 1) (h2 : x ≠ 1) :
  ∑ k in Finset.range 1000 | (k > 0), x ^ (2 * (k + 1)) / (x ^ (k + 1) - 1) = 499 :=
begin
  sorry
end

end complex_sum_499_l290_290911


namespace unique_zero_function_l290_290264

theorem unique_zero_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f(x + y) = 2 * f(x) + f(y)) : 
  ∀ x : ℝ, f(x) = 0 := 
by
  -- The proof steps will go here
  sorry

end unique_zero_function_l290_290264


namespace triangle_equilateral_l290_290883

variables (A B C F E : Type)
variables [triangle : is_triangle ABC] [is_median AF] [is_median CE]
variables [angle_BAF_eq_30 : ∠ BAF = 30] [angle_BCE_eq_30 : ∠ BCE = 30]

theorem triangle_equilateral (ABC : Type) [is_triangle ABC] 
  (AF CE : Type) [is_median AF] [is_median CE]
  (BAF BCE : Type) [angle_BAF_eq_30 : ∠ BAF = 30] [angle_BCE_eq_30 : ∠ BCE = 30] :
  AB = BC ∧ BC = AC ∧ AC = AB :=
sorry

end triangle_equilateral_l290_290883


namespace hours_per_week_is_40_l290_290428

-- Definitions
def current_job_hourly_rate := 30
def freelancer_hourly_rate := 40
def ficataxes_per_week := 25
def healthcare_premium_per_month := 400
def increase_per_month := 1100
def weeks_per_month := 4

-- Theorem stating that Janet works 40 hours per week
theorem hours_per_week_is_40 :
  let weekly_healthcare_premium := healthcare_premium_per_month / weeks_per_month in
  let desired_weekly_increase := increase_per_month / weeks_per_month in
  ∃ H, (freelancer_hourly_rate * H - ficataxes_per_week - weekly_healthcare_premium = current_job_hourly_rate * H + desired_weekly_increase) ∧ 
       H = 40 :=
by
  sorry

end hours_per_week_is_40_l290_290428


namespace set_count_B_satisfying_union_condition_l290_290819

theorem set_count_B_satisfying_union_condition :
  let A := {1, 2}
  in {B : set ℕ | A ∪ B = A}.finite ∧ {B : set ℕ | A ∪ B = A}.to_finset.card = 4 :=
by
  let A := {1, 2}
  have hB : {B : set ℕ | A ∪ B = A} = {A, ∅, {1}, {2}} :=
    sorry -- This is where the proof would go.
  exact ⟨finite_mem_set_eq_finite hB, finite_mem_set_eq_card hB 4⟩

end set_count_B_satisfying_union_condition_l290_290819


namespace train_length_l290_290689

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l290_290689


namespace cylinder_cut_is_cylinder_l290_290252

-- Define what it means to be a cylinder
structure Cylinder (r h : ℝ) : Prop :=
(r_pos : r > 0)
(h_pos : h > 0)

-- Define the condition of cutting a cylinder with two parallel planes
def cut_by_parallel_planes (c : Cylinder r h) (d : ℝ) : Prop :=
d > 0 ∧ d < h

-- Prove that the part between the parallel planes is still a cylinder
theorem cylinder_cut_is_cylinder (r h d : ℝ) (c : Cylinder r h) (H : cut_by_parallel_planes c d) :
  ∃ r' h', Cylinder r' h' :=
sorry

end cylinder_cut_is_cylinder_l290_290252


namespace correct_proposition_l290_290336

variables {α β : Type} [plane α] [plane β] (l m : line) 

-- Define the conditions
def is_perpendicular_to (l : line) (α : plane) : Prop := sorry  -- l ⟂ α
def is_subset_of (m : line) (β : plane) : Prop := m ⊆ β

-- Define the proposition
def proposition_2 := is_perpendicular_to α β → is_perpendicular_to l m

-- Reformulate the proof problem in Lean 4 statement
theorem correct_proposition (hl : is_perpendicular_to l α) (hm : is_subset_of m β) : 
  proposition_2 :=
sorry

end correct_proposition_l290_290336


namespace number_of_children_l290_290628

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290628


namespace simplify_sum_l290_290592

theorem simplify_sum (n k : ℕ) : 
  (∑ i in Finset.range n, (∏ j in Finset.range k, (i + j))) = 
  ((n * (n + 1) * ... * (n + k)) / (k + 1)) :=
by sorry

end simplify_sum_l290_290592


namespace avg_score_assigned_day_l290_290405

theorem avg_score_assigned_day
  (total_students : ℕ)
  (exam_assigned_day_students_perc : ℕ)
  (exam_makeup_day_students_perc : ℕ)
  (avg_makeup_day_score : ℕ)
  (total_avg_score : ℕ)
  : exam_assigned_day_students_perc = 70 → 
    exam_makeup_day_students_perc = 30 → 
    avg_makeup_day_score = 95 → 
    total_avg_score = 74 → 
    total_students = 100 → 
    (70 * 65 + 30 * 95 = 7400) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_score_assigned_day_l290_290405


namespace remaining_number_is_divisible_by_divisor_l290_290571

def initial_number : ℕ := 427398
def subtracted_number : ℕ := 8
def remaining_number : ℕ := initial_number - subtracted_number
def divisor : ℕ := 10

theorem remaining_number_is_divisible_by_divisor :
  remaining_number % divisor = 0 :=
by {
  sorry
}

end remaining_number_is_divisible_by_divisor_l290_290571


namespace annual_interest_rate_approx_l290_290547

-- Definitions of the variables
def FV : ℝ := 1764    -- Face value of the bill
def TD : ℝ := 189     -- True discount
def PV : ℝ := FV - TD -- Present value, calculated as per the problem statement

-- Simple interest formula components
def P : ℝ := PV       -- Principal
def T : ℝ := 9 / 12   -- Time period in years

-- Given conditions as definitions:
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Statement to prove that the annual interest rate R equals 16%
theorem annual_interest_rate_approx : ∃ R : ℝ, simple_interest P R T = TD ∧ R ≈ 16 := by
  use 16
  sorry

end annual_interest_rate_approx_l290_290547


namespace royal_family_children_l290_290606

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290606


namespace area_of_triangle_ABC_l290_290143

theorem area_of_triangle_ABC :
  let s := 1
  let area_of_equilateral_triangle := (fun (s : ℝ) => (sqrt 3 / 4) * s^2)
  let height_of_equilateral_triangle := (fun (s : ℝ) => (sqrt 3 / 2) * s)
  let distance_between_centers := (2 / 3) * height_of_equilateral_triangle s
  let side_length_of_triangle_ABC := 2 * distance_between_centers
  let area_of_ABC := area_of_equilateral_triangle side_length_of_triangle_ABC
  area_of_ABC = sqrt 3 / 3 :=
by {
  let s := 1
  let height_of_equilateral_triangle := (sqrt 3 / 2) * s
  let distance_between_centers := (2 / 3) * height_of_equilateral_triangle
  let side_length_of_triangle_ABC := 2 * distance_between_centers
  let area_of_equilateral_triangle := (sqrt 3 / 4) * side_length_of_triangle_ABC^2
  rw [←mul_assoc, ←(@div_mul_eq_mul_div ℝ _ _ _ (4:ℝ))],
  norm_num,
  linarith,
  sorry
}

end area_of_triangle_ABC_l290_290143


namespace WangLei_is_13_l290_290010

-- We need to define the conditions and question in Lean 4
def WangLei_age (x : ℕ) : Prop :=
  3 * x - 8 = 31

theorem WangLei_is_13 : ∃ x : ℕ, WangLei_age x ∧ x = 13 :=
by
  use 13
  unfold WangLei_age
  sorry

end WangLei_is_13_l290_290010


namespace particle_position_2023_minutes_l290_290667

def position_after_time (n : ℕ) : ℕ × ℕ :=
  let step_length := fun k => if k % 2 = 0 then k / 2 + 1 else (k + 1) / 2;
  let direction := fun k => if k % 2 = 0 then (-1, -1) else (1, 1);
  let move := fun (pos : ℕ × ℕ) (k : ℕ) =>
    let (dx, dy) := direction k;
    (pos.1 + dx * step_length k, pos.2 + dy * step_length k);
  (0, 2)
  |> fun start => (List.range n).foldl move start

theorem particle_position_2023_minutes : position_after_time 2023 = (44, 1) :=
sorry

end particle_position_2023_minutes_l290_290667


namespace variance_of_η_l290_290947

noncomputable def ξ : Type := sorry -- Define ξ as a random variable of type Binomial(16, 1/2)

-- Defining η in terms of ξ
def η := 5 * ξ - 1

-- The theorem to prove
theorem variance_of_η (ξ : Type) [isBinomial ξ 16 (1/2)] : Dη = 100 := by
  -- Assume all necessary properties and definitions are included within the context
  -- Prove the theorem using given properties and conditions
  sorry

end variance_of_η_l290_290947


namespace find_price_per_craft_l290_290366

-- Definitions based on conditions
def price_per_craft (x : ℝ) : Prop :=
  let crafts_sold := 3
  let extra_money := 7
  let deposit := 18
  let remaining_money := 25
  let total_before_deposit := 43
  3 * x + extra_money = total_before_deposit

-- Statement of the problem to prove x = 12 given conditions
theorem find_price_per_craft : ∃ x : ℝ, price_per_craft x ∧ x = 12 :=
by
  sorry

end find_price_per_craft_l290_290366


namespace wife_late_duration_l290_290214

noncomputable def time_late_in_minutes (speed_man speed_wife : ℝ) (meeting_time : ℝ) (distance_man : ℝ) : ℝ :=
  let t := (distance_man - speed_wife * meeting_time) / (speed_wife - speed_man) in
  t * 60

theorem wife_late_duration :
  let speed_man := 40
  let speed_wife := 50
  let meeting_time := 2
  let distance_man := speed_man * meeting_time in
  time_late_in_minutes speed_man speed_wife meeting_time distance_man = 24 :=
by
  sorry

end wife_late_duration_l290_290214


namespace fg_sqrt2_eq_neg5_l290_290389

noncomputable def f (x : ℝ) : ℝ := 4 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

theorem fg_sqrt2_eq_neg5 : f (g (Real.sqrt 2)) = -5 := by
  sorry

end fg_sqrt2_eq_neg5_l290_290389


namespace prob_product_less_than_36_l290_290926

-- Define the probability calculation setting
def uniform_prob (s : Finset ℕ) (p : ℕ → Prop) : ℚ :=
  s.filter p).card * ((s.card)⁻¹)

-- Define Manu's and Paco's number sets
def paco_set := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
def manu_set := Finset.range 16 \ {0}

-- Define the event definition
def event (p m : ℕ) := p * m < 36

-- Define the combined probability calculation given the independent probabilities
noncomputable def combined_probability : ℚ :=
  (∑ p in paco_set, ∑ m in manu_set, if event p m then 1 else 0) *
  ((paco_set.card * manu_set.card)⁻¹)

-- The main theorem statement that the combined probability of the event is 11/15
theorem prob_product_less_than_36 :
  combined_probability = 11 / 15 :=
sorry

end prob_product_less_than_36_l290_290926


namespace real_part_of_z_l290_290342

def imaginary_unit : ℂ := complex.I

noncomputable def complex_number : ℂ := imaginary_unit * (3 - imaginary_unit)

def real_part_of_complex (z : ℂ) : ℝ := z.re

theorem real_part_of_z : real_part_of_complex complex_number = 1 :=
by
  sorry

end real_part_of_z_l290_290342


namespace length_of_fence_l290_290988

theorem length_of_fence (side_length : ℕ) (h : side_length = 28) : 4 * side_length = 112 :=
by
  sorry

end length_of_fence_l290_290988


namespace circle_reflection_l290_290107

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l290_290107


namespace quinton_total_fruit_trees_l290_290090

-- Define the given conditions
def num_apple_trees := 2
def width_apple_tree_ft := 10
def space_between_apples_ft := 12
def width_peach_tree_ft := 12
def space_between_peaches_ft := 15
def total_space_ft := 71

-- Definition that calculates the total number of fruit trees Quinton wants to plant
def total_fruit_trees : ℕ := 
  let space_apple_trees := num_apple_trees * width_apple_tree_ft + space_between_apples_ft
  let space_remaining_for_peaches := total_space_ft - space_apple_trees
  1 + space_remaining_for_peaches / (width_peach_tree_ft + space_between_peaches_ft) + num_apple_trees

-- The statement to prove
theorem quinton_total_fruit_trees : total_fruit_trees = 4 := by
  sorry

end quinton_total_fruit_trees_l290_290090


namespace remainder_problem_l290_290913

theorem remainder_problem {x y z : ℤ} (h1 : x % 102 = 56) (h2 : y % 154 = 79) (h3 : z % 297 = 183) :
  x % 19 = 18 ∧ y % 22 = 13 ∧ z % 33 = 18 :=
by
  sorry

end remainder_problem_l290_290913


namespace num_of_nine_painted_l290_290223

-- Definitions based on the conditions
def house_numbers : List Nat := List.range' 1 70

-- The theorem stating the mathematical equivalence to our problem
theorem num_of_nine_painted : 
  (house_numbers.filter (λ n, ∃ i : Fin 1, n = 9 * (1 + i))).length = 7 := 
sorry

end num_of_nine_painted_l290_290223


namespace constant_term_expansion_l290_290109

theorem constant_term_expansion :
  let x := x;
  let f := (x - (1/x)) * (2*x + (1/x))^5
  in constant_term f = -40 :=
by
  sorry

end constant_term_expansion_l290_290109


namespace part_1_part_2_l290_290319

open Nat

def a : ℕ → ℕ 
| 0     := 0 -- a_0 is not defined in original sequence
| 1     := 1
| (n+2) := 2 * a (n + 1) + n + 1

theorem part_1 (n : ℕ) (hn : n > 0) : a n + n = 2^n := by
  sorry

theorem part_2 (n : ℕ) (hn : n > 0) : 
  (∑ k in range n, a (k + 1)) = 2^(n + 1) - 2 - (n * (n + 1)) / 2 := by
  sorry

end part_1_part_2_l290_290319


namespace books_from_second_shop_l290_290500

theorem books_from_second_shop (x : ℕ) (h₁ : 6500 + 2000 = 8500)
    (h₂ : 85 = 8500 / (65 + x)) : x = 35 :=
by
  -- proof goes here
  sorry

end books_from_second_shop_l290_290500


namespace train_length_correct_l290_290680

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l290_290680


namespace product_mod_5_l290_290279

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l290_290279


namespace fraction_cube_square_eq_30_l290_290245

/--
Prove that \(\frac{(0.3)^3}{(0.03)^2} = 30\) given the conditions:
1. \(0.3 = 3 \times 10^{-1}\)
2. \(0.03 = 3 \times 10^{-2}\)
-/
theorem fraction_cube_square_eq_30 :
  let a := 0.3
  let b := 0.03
  (a = 3 * 10⁻¹) →
  (b = 3 * 10⁻²) →
  (a^3 / b^2 = 30) :=
by
  intros ha hb
  sorry

end fraction_cube_square_eq_30_l290_290245


namespace max_value_correct_l290_290897

open Real

noncomputable def max_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  Real.sup {y | ∃ x, y = 2 * (a - x) * (x + sqrt (x^2 + b^2 + c))}

theorem max_value_correct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_value a b c ha hb hc = a^2 + b^2 + c := 
sorry

end max_value_correct_l290_290897


namespace log_ratio_l290_290510

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : log_base 4 a = log_base 6 b)
  (h4 : log_base 6 b = log_base 9 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := sorry

end log_ratio_l290_290510


namespace children_count_l290_290599

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l290_290599


namespace line_passes_center_of_circle_l290_290534

theorem line_passes_center_of_circle :
  let center := (-1 : ℝ, 0 : ℝ)
  ∃ (x y : ℝ), ((x + 1)^2 + y^2 = 1) ∧ (x = -1 ∧ y = 0) ∧ (x - y + 1 = 0) :=
by
  sorry

end line_passes_center_of_circle_l290_290534


namespace arithmetic_sequence_a1_a6_l290_290857

theorem arithmetic_sequence_a1_a6
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) : a 1 * a 6 = 14 :=
sorry

end arithmetic_sequence_a1_a6_l290_290857


namespace sqrt6_op_sqrt6_l290_290174

variable (x y : ℝ)

noncomputable def op (x y : ℝ) := (x + y)^2 - (x - y)^2

theorem sqrt6_op_sqrt6 : ∀ (x y : ℝ), op (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt6_op_sqrt6_l290_290174


namespace total_pairs_purchased_l290_290778

-- Define the conditions as hypotheses
def foxPrice : ℝ := 15
def ponyPrice : ℝ := 18
def totalSaved : ℝ := 8.91
def foxPairs : ℕ := 3
def ponyPairs : ℕ := 2
def sumDiscountRates : ℝ := 0.22
def ponyDiscountRate : ℝ := 0.10999999999999996

-- Prove that the total number of pairs of jeans purchased is 5
theorem total_pairs_purchased : foxPairs + ponyPairs = 5 := by
  sorry

end total_pairs_purchased_l290_290778


namespace measure_of_smaller_angle_l290_290998

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l290_290998


namespace royal_family_children_l290_290636

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290636


namespace minimum_additional_bailing_rate_l290_290579

-- Define the given conditions
def boat_distance_from_shore : ℝ := 2 -- miles
def water_intake_rate : ℝ := 15 -- gallons per minute
def max_gallons_before_sink : ℝ := 80 -- gallons
def rowing_speed : ℝ := 5 -- miles per hour
def initial_bailing_rate : ℝ := 5 -- gallons per minute

-- Calculate the time to reach shore in minutes
def time_to_shore : ℝ := (boat_distance_from_shore / rowing_speed) * 60 -- minutes

-- Define the proof that the minimum additional bailing rate is 7 gallons per minute
theorem minimum_additional_bailing_rate : ∀ r : ℝ, 
  rowing_speed > 0 →  -- ensure rowing speed is positive
  (water_intake_rate - (initial_bailing_rate + r)) * time_to_shore ≤ max_gallons_before_sink → 
  r ≥ 7 :=
by
  intros r h1 h2
  sorry

end minimum_additional_bailing_rate_l290_290579


namespace digit_4_more_than_digit_8_l290_290750

def count_digit (n : Nat) (d : Nat) : Nat :=
  (n.toString.toList.filter (fun c => c = d.digitChar)).length

def count_all_digit (pages : List Nat) (d : Nat) : Nat :=
  pages.foldr (fun x acc => acc + count_digit x d) 0

theorem digit_4_more_than_digit_8 :
  let pages := List.range' 1 520 1
  count_all_digit pages 4 - count_all_digit pages 8 = 100 := 
by 
  sorry

end digit_4_more_than_digit_8_l290_290750


namespace rectangular_prism_cut_l290_290237

theorem rectangular_prism_cut
  (x y : ℕ)
  (original_volume : ℕ := 15 * 5 * 4) 
  (remaining_volume : ℕ := 120) 
  (cut_out_volume_eq : original_volume - remaining_volume = 5 * x * y) 
  (x_condition : 1 < x) 
  (x_condition_2 : x < 4) 
  (y_condition : 1 < y) 
  (y_condition_2 : y < 15) : 
  x + y = 15 := 
sorry

end rectangular_prism_cut_l290_290237


namespace parallel_line_through_point_l290_290273

theorem parallel_line_through_point :
  ∃ c : ℝ, ∀ x y : ℝ, (x = -1) → (y = 3) → (x - 2*y + 3 = 0) → (x - 2*y + c = 0) :=
sorry

end parallel_line_through_point_l290_290273


namespace sweater_cost_proof_l290_290083

-- Definitions based on conditions
constant dress_shirt_price : ℝ := 15.00
constant pants_price : ℝ := 40.00
constant suit_price : ℝ := 150.00
constant num_dress_shirts : ℕ := 4
constant num_pants : ℕ := 2
constant num_sweaters : ℕ := 2
constant total_spent : ℝ := 252.00
constant store_discount : ℝ := 0.20
constant coupon_discount : ℝ := 0.10

-- The statement to prove the cost of each sweater
theorem sweater_cost_proof (s : ℝ) :
  let total_cost := (num_dress_shirts * dress_shirt_price) + (num_pants * pants_price) + suit_price + (num_sweaters * s),
      after_store_discount := total_cost - store_discount * total_cost,
      final_cost := after_store_discount - coupon_discount * after_store_discount
  in final_cost = total_spent → s = 30 :=
by
  intros total_cost after_store_discount final_cost h1
  sorry

end sweater_cost_proof_l290_290083


namespace heather_oranges_l290_290365

theorem heather_oranges (initial_oranges additional_oranges : ℝ) (h1 : initial_oranges = 60.5) (h2 : additional_oranges = 35.8) :
  initial_oranges + additional_oranges = 96.3 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end heather_oranges_l290_290365


namespace find_sum_of_y_l290_290815

open Real

-- Define the list of numbers
def numbers (y : ℝ) : List ℝ := [8, 3, 6, 3, 7, 3, y]

-- Define the mean of the list
def mean (y : ℝ) : ℝ := (30 + y) / 7

-- Define the mode of the list
def mode : ℝ := 3

-- Define the median of the list
def median (y : ℝ) : ℝ :=
  if y ≤ 3 then 3
  else if y < 6 then y
  else 6

-- Define the arithmetic progression condition
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the main problem statement
theorem find_sum_of_y : (∃ y : ℝ, median y = 3 ∧ mean y = 3 ∧ y ≤ 3) ∨
                        (∃ y : ℝ, median y = 6 ∧ mean y = 9 ∧ y ≥ 6) →
                        ∑ y in {y | (median y = 3 ∧ mean y = 3 ∧ y ≤ 3) ∨
                                   (median y = 6 ∧ mean y = 9 ∧ y ≥ 6)}, y = 24 :=
  sorry

end find_sum_of_y_l290_290815


namespace quadratic_solution_l290_290833

theorem quadratic_solution (a b : ℚ) (h : a * 1^2 + b * 1 + 1 = 0) : 3 - a - b = 4 := 
by
  sorry

end quadratic_solution_l290_290833


namespace convex_ngon_triangle_count_l290_290080

theorem convex_ngon_triangle_count (n : ℕ) 
  (h_convex: ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i \\ -- Convexity condition in terms of diagonals' intersection points
  h_no_three_diagonals_intersect: ∀ i j k l m n,/
  (i ≠ j ∧ j ≠ k ∧ k ≠ i) → (l ≠ m ∧ m ≠ n ∧ n ≠ l) 
  ∧ (intersection of diagonals formed by choosing i, j, k 
  with l, m, n doesn't coincide possibly with other diagonals ")) : 
  ℕ := 
begin
-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := nat.choose n k
suite 
  steps according convergence for final triangle count prove assertion
⟨sorry⟩, -- Skipping the proof
end

end convex_ngon_triangle_count_l290_290080


namespace time_upstream_is_correct_l290_290197

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l290_290197


namespace equilateral_triangle_dot_product_l290_290860

-- Define the condition that triangle ABC is equilateral with side length 2
def is_equilateral_triangle (A B C : Type*) [normed_group A] [normed_space ℝ A] (P Q R : A) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- Main theorem statement
theorem equilateral_triangle_dot_product {A : Type*} [inner_product_space ℝ A] (P Q R : A) 
  (h : is_equilateral_triangle A B C P Q R) (hPQ : dist P Q = 2) :
  ⟪Q - P, R - P⟫ = 2 :=
sorry

end equilateral_triangle_dot_product_l290_290860


namespace apple_price_l290_290705

theorem apple_price :
  ∀ (l q : ℝ), 
    (10 * l = 3.62) →
    (30 * l + 3 * q = 11.67) →
    (30 * l + 6 * q = 12.48) :=
by
  intros l q h₁ h₂
  -- The proof would go here with the steps, but for now we use sorry.
  sorry

end apple_price_l290_290705


namespace time_upstream_is_correct_l290_290196

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l290_290196


namespace find_value_of_fraction_l290_290062

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l290_290062


namespace five_digit_palindromes_count_l290_290737

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l290_290737


namespace sum_of_first_five_terms_arith_seq_l290_290167

/-
An arithmetic sequence where the first term is 6 and the common difference is 4.
We aim to prove that the sum of the first five terms is 70.
-/

def a : ℕ → ℕ
| 0     := 6
| (n+1) := a n + 4

theorem sum_of_first_five_terms_arith_seq : 
  (a 0 + a 1 + a 2 + a 3 + a 4) = 70 :=
by
  sorry

end sum_of_first_five_terms_arith_seq_l290_290167


namespace prove_sum_of_f_l290_290309

theorem prove_sum_of_f :
  (∀ x, f (3 ^ x) = 4 * x * log 2 3) →
  (f 2 + f 4 + f 8 + f 16 + f 32 + f 64 + f 128 + f 256 = 2008) :=
  sorry

end prove_sum_of_f_l290_290309


namespace ladder_distance_base_wall_l290_290188

theorem ladder_distance_base_wall
  (hypotenuse : ℝ) (height : ℝ) (base : ℝ)
  (h1 : hypotenuse = 15)
  (h2 : height = 9)
  (h3 : hypotenuse^2 = base^2 + height^2) :
  base = 12 :=
by {
  rw [h1, h2] at h3;
  norm_num at h3;
  exact eq_of_sq_eq_sq _ _ h3 (by norm_num),
}

end ladder_distance_base_wall_l290_290188


namespace power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l290_290310

theorem power_function_condition (m : ℝ) : m^2 + 2 * m = 1 ↔ m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

theorem direct_proportionality_condition (m : ℝ) : (m^2 + m - 1 = 1 ∧ m^2 + 3 * m ≠ 0) ↔ m = 1 :=
by sorry

theorem inverse_proportionality_condition (m : ℝ) : (m^2 + m - 1 = -1 ∧ m^2 + 3 * m ≠ 0) ↔ m = -1 :=
by sorry

end power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l290_290310


namespace train_length_l290_290690

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmh = 60 → time_s = 15 → (60 * 1000 / 3600) * 15 = length_m → length_m = 250 :=
by { intros, sorry }

end train_length_l290_290690


namespace minimize_triangle_area_minimize_product_PA_PB_l290_290209

-- Define the initial conditions and geometry setup
def point (x y : ℝ) := (x, y)
def line_eq (a b : ℝ) := ∀ x y : ℝ, x / a + y / b = 1

-- Point P
def P := point 2 1

-- Condition: the line passes through point P and intersects the axes
def line_through_P (a b : ℝ) := line_eq a b ∧ (2 / a + 1 / b = 1) ∧ a > 2 ∧ b > 1

-- Prove that the line minimizing the area of triangle AOB is x + 2y - 4 = 0
theorem minimize_triangle_area (a b : ℝ) (h : line_through_P a b) :
  a = 4 ∧ b = 2 → line_eq 4 2 := 
sorry

-- Prove that the line minimizing the product |PA||PB| is x + y - 3 = 0
theorem minimize_product_PA_PB (a b : ℝ) (h : line_through_P a b) :
  a = 3 ∧ b = 3 → line_eq 3 3 := 
sorry

end minimize_triangle_area_minimize_product_PA_PB_l290_290209


namespace positive_difference_perimeters_l290_290999

theorem positive_difference_perimeters :
  let w1 := 3
  let h1 := 2
  let w2 := 6
  let h2 := 1
  let P1 := 2 * (w1 + h1)
  let P2 := 2 * (w2 + h2)
  P2 - P1 = 4 := by
  sorry

end positive_difference_perimeters_l290_290999


namespace constant_term_in_quadratic_eq_l290_290422

theorem constant_term_in_quadratic_eq : 
  ∀ (x : ℝ), (x^2 - 5 * x = 2) → (∃ a b c : ℝ, a = 1 ∧ a * x^2 + b * x + c = 0 ∧ c = -2) :=
by
  sorry

end constant_term_in_quadratic_eq_l290_290422


namespace count_correct_expressions_l290_290232

noncomputable def expressions_correct_count : ℕ := 1

theorem count_correct_expressions :
  (¬ ({1} ∈ {1, 2}) ∧  -- ① is incorrect
  (∅ ⊆ {0}) ∧            -- ② is correct    
  ¬ ({3, 2, 1} ⊂ {1, 2, 3}) ∧  -- ③ is incorrect
  ¬ ({y | y = x} ⊆ {(x, y) | y = x}))  -- ④ is incorrect
  → expressions_correct_count = 1 :=
by
  intros,
  sorry

end count_correct_expressions_l290_290232


namespace number_of_factors_of_n_l290_290456

noncomputable def n : ℕ := 2^4 * 3^5 * 5^6 * 7^7

theorem number_of_factors_of_n : ∃ k : ℕ, k = 1680 ∧ (∀ d : ℕ, d ∣ n ↔ d ∈ finset.Icc (1 : ℕ) n ∧ d.factorization ⊆ n.factorization) :=
by
  existsi 1680
  split
  case left =>
    sorry
  case right =>
    sorry

end number_of_factors_of_n_l290_290456


namespace non_unique_line_when_right_angle_l290_290822

variables {Point : Type} [metric_space Point]

def is_right_angle (A B C : Point) : Prop := 
  ∃ (line_AB : line_segment Point) (line_AC : line_segment Point) (line_BC : line_segment Point),
    (A ∈ line_AB ∧ B ∈ line_AB) ∧ 
    (A ∈ line_AC ∧ C ∈ line_AC) ∧ 
    (B ∈ line_BC ∧ C ∈ line_BC) ∧ 
    (line_AB.is_perpendicular_to line_BC)

def unique_max_product_line (A B C : Point) (L : set (line_segment Point)) : Prop :=
  ∀ (L1 L2 : line_segment Point),
    (L1 ∈ L ∧ L2 ∈ L) → 
    ((prod_of_distances L1 A B > prod_of_distances L2 A B ∨ 
      prod_of_distances L2 A B > prod_of_distances L1 A B)  ∨ 
    (L1 = L2))

theorem non_unique_line_when_right_angle (A B C : Point) :
  is_right_angle A B C → 
  ¬ unique_max_product_line A B C {line | contains_point line C ∧ is_max_product_line line A B C} :=
sorry

end non_unique_line_when_right_angle_l290_290822


namespace horizontal_asymptote_l290_290390

theorem horizontal_asymptote :
  ∀ (x : ℝ), (∀ ε > 0, ∃ M, ∀ x > M, |(7 * x^5 + 2 * x^3 + 3 * x^2 + 8 * x + 4) / (8 * x^5 + 5 * x^3 + 4 * x^2 + 6 * x + 2) - 7 / 8| < ε) :=
begin
  sorry
end

end horizontal_asymptote_l290_290390


namespace complementary_sets_5904_l290_290756

-- Definitions according to problem conditions
structure Card where
  shape : Fin 3 -- circle, square, triangle
  color : Fin 3 -- red, blue, green
  shade : Fin 3 -- light, medium, dark
  size : Fin 3  -- small, medium, large

def deck : Finset Card := 
  Finset.univ  -- Assuming a universe of all possible Cards

def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∨ c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c3.shape ≠ c1.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c3.color ≠ c1.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∨ c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c3.shade ≠ c1.shade) ∧
  (c1.size = c2.size ∧ c2.size = c3.size ∨ c1.size ≠ c2.size ∧ c2.size ≠ c3.size ∧ c3.size ≠ c1.size)

noncomputable def complementary_set_count : Nat :=
  (deck.to_list.combinations 3).filter (λ triple, match triple with
    | [c1, c2, c3] => is_complementary c1 c2 c3
    | _ => false).length

-- Proof goal
theorem complementary_sets_5904 : complementary_set_count = 5904 := sorry

end complementary_sets_5904_l290_290756


namespace ratio_equality_l290_290646

-- Definitions based on the problem's conditions
variables {k : Type} [metric_space k] [normed_group k] [normed_space ℝ k] {O : k} {r : ℝ}
variables {A B C D P X Y : k}

-- Conditions
def is_diameter (O A B : k) : Prop := dist O A = dist O B ∧ dist A B = 2 * dist O A
def is_perpendicular (A B C D : k) : Prop := ⟪A - B, C - D⟫ = 0 -- where ⟪ , ⟫ denotes the inner product

-- Given conditions
axiom AB_diameter : is_diameter O A B
axiom CD_diameter : is_diameter O C D
axiom CD_perpendicular : is_perpendicular A B C D
axiom P_on_circle : dist O P = r
axiom X_intersection : ∃ X, P ∈ line O C ∧ P ∈ line X B
axiom Y_intersection : ∃ Y, P ∈ line O D ∧ P ∈ line Y A

-- Required proof
theorem ratio_equality : (dist A X / dist A Y) = (dist B X / dist B Y) :=
sorry

end ratio_equality_l290_290646


namespace largest_prime_factor_of_expression_l290_290243

noncomputable def largest_prime_factor (n : ℤ) : ℤ :=
  let factors := (prime_factors n).to_finset
  factors.max' (nonempty_of_ne_empty (to_finset_prime_factors_ne_empty n))

theorem largest_prime_factor_of_expression :
  largest_prime_factor (25^2 + 35^3 - 10^5) = 113 :=
by
  -- Basic computations according to given conditions
  let term1 := 25^2
  let term2 := 35^3
  let term3 := 10^5
  let expression := term1 + term2 - term3

  -- Verifying the expression equals -56500
  have expr_eq : expression = -56500 := by
    have h1 : term1 = 625 := by norm_num
    have h2 : term2 = 125 * 343 := by norm_num
    have h3 : term2 = 42875 := by norm_num at h2
    have h4 : term3 = 32 * 3125 := by norm_num
    have h5 : term3 = 100000 := by norm_num at h4
    have h_sum := calc 
      625 + 42875 - 100000 = 43500 - 100000 := by norm_num
      ... = -56500 := by norm_num
    exact h_sum

  -- Proof step to indicate expression clearly
  rw [expr_eq]

  -- Proving the largest prime factor of -56500, which is 113
  have factorization : prime_factors (-56500) = [2, 2, 5, 5, 5, 5, 113, -1] := by sorry
  have largest_prime := 113
  have largest_prime_in_factors := list.mem_cons_self 113 [2, 2, 5, 5, 5, 5]
  exact largest_prime_in_factors

end largest_prime_factor_of_expression_l290_290243


namespace cosine_angle_PC_PAB_eq_one_third_l290_290502

open Real EuclideanGeometry Vector

theorem cosine_angle_PC_PAB_eq_one_third (P A B C : Point) (PA PB PC : Vector) :
  (∠ P A B = 60 ∧ ∠ P B C = 60 ∧ ∠ P A C = 60) →
  (cosine_of_angle_between_vector_line_and_plane PC PA PB = 1 / 3) :=
by
  sorry

end cosine_angle_PC_PAB_eq_one_third_l290_290502


namespace students_in_band_or_sports_l290_290023

theorem students_in_band_or_sports (total_students B S B_inter_S : ℕ) 
  (h_total : total_students = 320)
  (h_B : B = 85)
  (h_S : S = 200)
  (h_B_inter_S : B_inter_S = 60) : 
  B ∪ S = 225 := 
by
  sorry

end students_in_band_or_sports_l290_290023


namespace intersection_M_N_l290_290359

def M : Set ℝ := { x | x^2 - x - 6 ≤ 0 }
def N : Set ℝ := { x | -2 < x ∧ x ≤ 4 }

theorem intersection_M_N : (M ∩ N) = { x | -2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_M_N_l290_290359


namespace five_digit_palindromes_count_l290_290733

theorem five_digit_palindromes_count : 
  let a_values := {a : ℕ | 1 ≤ a ∧ a ≤ 9}
  let b_values := {b : ℕ | 0 ≤ b ∧ b ≤ 9}
  let c_values := {c : ℕ | 0 ≤ c ∧ c ≤ 9}
  a_values.card * b_values.card * c_values.card = 900 := 
by 
  -- a has 9 possible values
  have a_card : a_values.card = 9 := sorry
  -- b has 10 possible values
  have b_card : b_values.card = 10 := sorry
  -- c has 10 possible values
  have c_card : c_values.card = 10 := sorry
  -- solve the multiplication
  sorry

end five_digit_palindromes_count_l290_290733


namespace sqrt_expression_equals_l290_290714

theorem sqrt_expression_equals :
  sqrt ((16 ^ 12 + 8 ^ 14) / (16 ^ 5 + 8 ^ 16 + 2 ^ 24)) 
  = 2 ^ 11 * sqrt (65 / 17) :=
by
  let a := 16
  let b := 8
  let a_pow_12 := a ^ 12
  let b_pow_14 := b ^ 14
  let a_pow_5 := a ^ 5
  let b_pow_16 := b ^ 16
  let two_pow_24 := 2 ^ 24
  have eq_a : a = 2 ^ 4 := by norm_num
  have eq_b : b = 2 ^ 3 := by norm_num
  sorry

end sqrt_expression_equals_l290_290714


namespace hyperbola_quadrilateral_area_l290_290352

theorem hyperbola_quadrilateral_area
  (a : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h_hyperbola : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / 3) = 1)
  (h_foci : F1 = (sqrt 6, 0) ∧ F2 = (-sqrt 6, 0))
  (h_circle : ∀ (x y : ℝ), (x - F1.1)^2 + y^2 = a^2 ∨ (x - F2.1)^2 + y^2 = a^2)
  (h_asymptotes_tangent : ∀ (x y : ℝ), y = (sqrt 3 / a) * x - sqrt 2 →
                                   (x - F1.1)^2 + y^2 = a^2 ∧ (x - F2.1)^2 + y^2 = a^2)
  (h_pointA : A = (sqrt 6 / 2, sqrt 6 / 2))
  (h_pointB : B = (-sqrt 6 / 2, sqrt 6 / 2)) :
  let quadrilateral_area := 2 * (2 * sqrt 6) * sqrt 3
  in quadrilateral_area / 2 = 6 := by
  sorry

end hyperbola_quadrilateral_area_l290_290352


namespace common_chord_condition_l290_290528

theorem common_chord_condition 
    (h d1 d2 : ℝ) (C1 C2 D1 D2 : ℝ) 
    (hyp_len : (C1 * D1 = C2 * D2)) : 
    (C1 * D1 = C2 * D2) ↔ (1 / h^2 = 1 / d1^2 + 1 / d2^2) :=
by
  sorry

end common_chord_condition_l290_290528


namespace gcd_sequence_inequality_l290_290477

-- Add your Lean 4 statement here
theorem gcd_sequence_inequality {n : ℕ} 
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → Nat.gcd n k < Nat.gcd n (k+1)) : 
  Nat.gcd n 35 < Nat.gcd n 36 := 
sorry

end gcd_sequence_inequality_l290_290477


namespace diamond_contains_conditional_structure_l290_290841

-- Define what a decision box is in terms of logical structure
def is_decision_box (shape : Type) : Prop := 
  shape = "◇"

-- Define what a conditional structure is
def is_conditional_structure : Prop := 
  True -- We will assume this as a basic property

-- Proof statement that any diamond shape (◇) definitely contains a conditional structure
theorem diamond_contains_conditional_structure (shape: Type) (h: is_decision_box shape) : is_conditional_structure :=
by
  sorry

end diamond_contains_conditional_structure_l290_290841


namespace fg_difference_l290_290806

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference : f (g 3) - g (f 3) = 59 :=
by
  sorry

end fg_difference_l290_290806


namespace divisor_in_first_division_l290_290484

theorem divisor_in_first_division
  (N : ℕ)
  (D : ℕ)
  (Q : ℕ)
  (h1 : N = 8 * D)
  (h2 : N % 5 = 4) :
  D = 3 := 
sorry

end divisor_in_first_division_l290_290484


namespace line_intersects_segment_l290_290855

theorem line_intersects_segment (a : ℝ) (A B : ℝ × ℝ)
  (hA : A = (1, a)) (hB : B = (2, 4))
  (line : ℝ → ℝ → Prop := λ x y, x - y + 1 = 0) :
  (line 1 a) = line 2 4 → (a > 2) := 
by
  sorry

end line_intersects_segment_l290_290855


namespace correct_circle_property_l290_290231

theorem correct_circle_property (A B C D X Y Z : Point) (circle : Set Point)
  (inscribed : inscribed_quadrilateral A B C D X Y circle) :
  (diagonals_supplementary A C B D ↔ true) := 
sorry

end correct_circle_property_l290_290231


namespace possible_values_of_b_l290_290932

theorem possible_values_of_b (b : ℝ) : (¬ ∃ x : ℝ, x^2 + b * x + 1 ≤ 0) → -2 < b ∧ b < 2 :=
by
  intro h
  sorry

end possible_values_of_b_l290_290932


namespace royal_children_count_l290_290607

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290607


namespace total_fish_at_wedding_l290_290953

def num_tables : ℕ := 32
def fish_per_table_except_one : ℕ := 2
def fish_on_special_table : ℕ := 3
def number_of_special_tables : ℕ := 1
def number_of_regular_tables : ℕ := num_tables - number_of_special_tables

theorem total_fish_at_wedding : 
  (number_of_regular_tables * fish_per_table_except_one) + (number_of_special_tables * fish_on_special_table) = 65 :=
by
  sorry

end total_fish_at_wedding_l290_290953


namespace num_correct_propositions_is_two_l290_290351

definition f (x : ℝ) : ℝ := Real.sin (2 * x + π / 6)

theorem num_correct_propositions_is_two :
  (¬ (∀ x, f (-π / 12) = f (-π / 12 + x))) ∧
  (∀ x, f (5 * π / 12) = 0) ∧
  (¬ (∀ x, f (x - π / 6) = Real.sin (2 * x + π / 3))) ∧
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  2 = 2 := by
  sorry

end num_correct_propositions_is_two_l290_290351


namespace remainder_of_product_mod_5_l290_290281

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l290_290281


namespace intersection_complementA_setB_eq_expected_result_l290_290801

-- Definition of sets A and B
def setA : Set ℝ := {x : ℝ | 2^x > 4}
def setB : Set ℤ := {x : ℤ | Real.logBase 2 (x : ℝ) < 3}

-- Define complement of set A in the real numbers
def complementA : Set ℝ := {x : ℝ | ¬ (2^x > 4)}

-- Define intersection of complementA and setB
def intersection_complementA_setB : Set ℤ := {x : ℤ | (x : ℝ) ≤ 2 ∧ x ∈ setB}

-- Expected Result
def expected_result : Set ℤ := {1, 2}

-- Lean statement to prove the intersection equals expected result
theorem intersection_complementA_setB_eq_expected_result :
  intersection_complementA_setB = expected_result :=
by
  sorry

end intersection_complementA_setB_eq_expected_result_l290_290801


namespace how_much_money_per_tshirt_l290_290101

def money_made_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) : Prop :=
  total_money_tshirts / number_tshirts = 62

theorem how_much_money_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) 
  (h1 : total_money_tshirts = 11346) 
  (h2 : number_tshirts = 183) : 
  money_made_per_tshirt total_money_tshirts number_tshirts := 
by 
  sorry

end how_much_money_per_tshirt_l290_290101


namespace angle_BMA_60_l290_290922

theorem angle_BMA_60 {A B C M : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M]
  (h1 : ∃ (M : M), ∃ (A C : A), AM = BM + MC)
  (h2 : ∠BMA = ∠MBC + ∠BAC) :
  ∠BMA = 60 :=
sorry

end angle_BMA_60_l290_290922


namespace sum_to_130_mod_7_sum_to_130_mod_4_l290_290160

theorem sum_to_130_mod_7 : (finset.sum (finset.range 131) id) % 7 = 3 := sorry

theorem sum_to_130_mod_4 : (finset.sum (finset.range 131) id) % 4 = 3 := sorry

end sum_to_130_mod_7_sum_to_130_mod_4_l290_290160


namespace parents_give_per_year_l290_290916

def Mikail_age (x : ℕ) : Prop :=
  x = 3 * (x - 3)

noncomputable def money_per_year (total_money : ℕ) (age : ℕ) : ℕ :=
  total_money / age

theorem parents_give_per_year 
  (x : ℕ) (hx : Mikail_age x) : 
  money_per_year 45 x = 5 :=
sorry

end parents_give_per_year_l290_290916


namespace distance_between_bars_l290_290591

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

end distance_between_bars_l290_290591


namespace SURE_to_9612_l290_290132

/-- 
  GREAT FUNDS coding scheme where 
  G = 0, R = 1, E = 2, A = 3, T = 4, F = 5,
  U = 6, N = 7, D = 8, S = 9.
  Prove that the code word "SURE" translates to 9612.
-/
def coding_scheme : char → ℕ 
| 'G' := 0 
| 'R' := 1 
| 'E' := 2 
| 'A' := 3 
| 'T' := 4 
| 'F' := 5 
| 'U' := 6 
| 'N' := 7 
| 'D' := 8 
| 'S' := 9 
| _   := 0 -- default case for other characters, not needed for this proof

theorem SURE_to_9612 : 
  let SURE := ['S', 'U', 'R', 'E'] in 
  (coding_scheme SURE[0]) * 1000 + 
  (coding_scheme SURE[1]) * 100 + 
  (coding_scheme SURE[2]) * 10 + 
  (coding_scheme SURE[3]) = 9612 := 
by 
  intro SURE 
  simp only [coding_scheme, SURE] 
  have h_S := coding_scheme 'S' 
  have h_U := coding_scheme 'U' 
  have h_R := coding_scheme 'R' 
  have h_E := coding_scheme 'E' 
  simp [h_S, h_U, h_R, h_E] 
  sorry

end SURE_to_9612_l290_290132


namespace Jim_runs_total_distance_l290_290436

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end Jim_runs_total_distance_l290_290436


namespace total_fat_l290_290661

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l290_290661


namespace calc_x_power_2x_l290_290827

theorem calc_x_power_2x (x : ℝ) (h : 8^x - 8^(x - 1) = 448) : x^(2 * x) = 729 :=
sorry

end calc_x_power_2x_l290_290827


namespace distance_between_x_intercepts_l290_290211

theorem distance_between_x_intercepts 
  (s1 s2 : ℝ) (P : ℝ × ℝ)
  (h1 : s1 = 2) 
  (h2 : s2 = -4) 
  (hP : P = (8, 20)) :
  let l1_x_intercept := (0 - (20 - P.2)) / s1 + P.1
  let l2_x_intercept := (0 - (20 - P.2)) / s2 + P.1
  abs (l1_x_intercept - l2_x_intercept) = 15 := 
sorry

end distance_between_x_intercepts_l290_290211


namespace travel_time_Carville_to_Pikville_l290_290716

theorem travel_time_Carville_to_Pikville :
  let distance_CN := 50 * 5,
      distance_NP := 60 * 3,
      distance_CP := distance_CN + distance_NP,
      time_CP := distance_CP / 55 
  in Real.floor (100 * time_CP) / 100 = 7.82 :=
by
  let distance_CN := 50 * 5
  let distance_NP := 60 * 3
  let distance_CP := distance_CN + distance_NP
  let time_CP := distance_CP / 55 
  have eq_distances : distance_CN = 250 := by sorry
  have eq_distances' : distance_NP = 180 := by sorry
  have eq_distances_total : distance_CP = 430 := by sorry
  have eq_time : time_CP ≈ 430 / 55 := by sorry
  have eq_rounded : Real.floor (100 * time_CP) / 100 ≈ 7.81818 := by sorry
  exact eq_rounded

end travel_time_Carville_to_Pikville_l290_290716


namespace points_in_different_half_spaces_l290_290965

-- Definition of the plane equation and the two points
def plane : ℝ × ℝ × ℝ → ℝ := λ p => p.1 + 2 * p.2 + 3 * p.3
def point1 : ℝ × ℝ × ℝ := (1, 2, -2)
def point2 : ℝ × ℝ × ℝ := (2, 1, -1)

-- Problem statement: proving that the points lie in different half-spaces
theorem points_in_different_half_spaces : plane point1 < 0 ∧ plane point2 > 0 :=
by
  -- Proof here
  sorry

end points_in_different_half_spaces_l290_290965


namespace sequence_formula_l290_290791

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n + 1) :
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2 * n) :=
by
  sorry

end sequence_formula_l290_290791


namespace max_expression_value_eq_32_l290_290862

theorem max_expression_value_eq_32 :
  ∃ (a b c d : ℕ), a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ d ∈ {1, 2, 3, 4} ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (c * a^b - d = 32) :=
sorry

end max_expression_value_eq_32_l290_290862


namespace problem_solution_exists_six_values_n_l290_290003

open Int

theorem problem_solution_exists_six_values_n : 
  let condition := λ n : ℕ, (n > 0) ∧ (⌊Real.sqrt (n:ℝ)⌋ = (n + 1000) / 70) in 
  ∃ (S : Set ℕ), S = {n | condition n} ∧ S.card = 6 :=
by
  sorry

end problem_solution_exists_six_values_n_l290_290003


namespace stable_table_quadruples_l290_290320

theorem stable_table_quadruples (n : ℕ) :
  let count_pairs (m : ℕ) : ℕ :=
    if m ≤ n then m + 1 else 2 * n - m + 1 in
  let num_quadruples : ℕ :=
    (finset.range (n + 1)).sum (λ m, count_pairs m) +
    (finset.range' (n + 1) (2 * n + 1)).sum (λ m, count_pairs m) in
  num_quadruples = (n + 1) * (n + 1) :=
sorry

end stable_table_quadruples_l290_290320


namespace carrie_total_spent_l290_290715

-- Definitions based on conditions
def regularPrice : ℝ := 9.95
def tshirtsBought : ℝ := 20
def discountRate : ℝ := 0.15
def taxRate : ℝ := 0.05

-- Goal: prove that the final cost is $177.61
theorem carrie_total_spent : 
    let discount := regularPrice * discountRate;
    let discountedPrice := regularPrice - discount;
    let totalCostBeforeTax := tshirtsBought * discountedPrice;
    let tax := totalCostBeforeTax * taxRate;
    let totalCost := totalCostBeforeTax + tax;
    (Float.round (totalCost * 100) / 100) = 177.61 :=
by
    -- This will be filled with the proof steps to show the evaluation.
    sorry

end carrie_total_spent_l290_290715


namespace base_eight_conversion_l290_290157

theorem base_eight_conversion :
  (1 * 8^2 + 3 * 8^1 + 2 * 8^0 = 90) := by
  sorry

end base_eight_conversion_l290_290157


namespace find_first_number_l290_290212

noncomputable def sequence (a_1 a_2 : ℚ) : ℕ → ℚ
| 0     := a_1
| 1     := a_2
| (n+2) := sequence n * sequence (n+1)

theorem find_first_number (a : ℕ → ℚ) : 
  a 6 = 16 → a 7 = 64 → a 8 = 1024 → 
  (∀ n, n ≥ 2 → a (n+2) = a (n+1) * a n) → 
  a 0 = 1 / 4 :=
by
  sorry

end find_first_number_l290_290212


namespace sin_cos_fourth_quadrant_l290_290804

variables {α : ℝ} (h1 : (0 > sin α ∧ cos α > 0)) (h2 : tan α = -3)

theorem sin_cos_fourth_quadrant (α : ℝ) (h1 : (0 > sin α ∧ cos α > 0)) (h2 : tan α = -3) :
  sin α = -3 * (cos α) ∧ cos α = (1 / 10) * real.sqrt 10 :=
sorry

end sin_cos_fourth_quadrant_l290_290804


namespace total_area_of_hexagon_is_693_l290_290663

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end total_area_of_hexagon_is_693_l290_290663


namespace constant_term_of_expansion_l290_290397

theorem constant_term_of_expansion (n : ℕ) (h : (2 : ℕ)^n = 32) : 
  ∑ k in finset.range (n + 1), nat.choose n k * ((2:ℤ)^(n - 3 * k)) = 10 := 
by
  sorry

end constant_term_of_expansion_l290_290397


namespace sqrt_inequality_l290_290648

theorem sqrt_inequality (a : ℝ) (h : a > 6) : 
  sqrt (a - 3) - sqrt (a - 4) < sqrt (a - 5) - sqrt (a - 6) :=
by 
  sorry

end sqrt_inequality_l290_290648


namespace max_feet_to_catch_up_l290_290229

theorem max_feet_to_catch_up (total_distance : ℕ) (initial_even_distance : ℕ) (alex_uphill_gain : ℕ) 
  (max_downhill_gain : ℕ) (alex_mixed_terrain_gain : ℕ) : 
  (total_distance = 5000) → 
  (initial_even_distance = 200) → 
  (alex_uphill_gain = 300) → 
  (max_downhill_gain = 170) → 
  (alex_mixed_terrain_gain = 440) → 
  (alex_uphill_gain - max_downhill_gain + alex_mixed_terrain_gain = 570) := 
by
  intros
  rw [alex_uphill_gain, max_downhill_gain, alex_mixed_terrain_gain]
  exact rfl

end max_feet_to_catch_up_l290_290229


namespace cube_root_nested_l290_290387

theorem cube_root_nested (N : ℝ) (h : N > 1) : real.cbrt (N * real.cbrt (N * real.cbrt N)) = N^(13/27) := 
sorry

end cube_root_nested_l290_290387


namespace royal_family_children_l290_290637

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290637


namespace problem_statement_l290_290052

def S : Set ℝ := { x | x ≠ 0 }

noncomputable def f (x : ℝ) : ℝ := sorry -- \( f \) is a function from \( S \) to \( S \)

axiom prop1 (x : ℝ) (h : x ∈ S) : f (1 / x) = x^2 * f x
axiom prop2 (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x + y ∈ S) : f (1 / x) + f (1 / y) = 2 + f (1 / (x + y))

theorem problem_statement : 
  let n := 1, s := 4 in n * s = 4 := 
by sorry

end problem_statement_l290_290052


namespace john_spent_on_sweets_l290_290073

def initial_amount := 7.10
def amount_given_per_friend := 1.00
def amount_left := 4.05
def amount_spent_on_friends := 2 * amount_given_per_friend
def amount_remaining_after_friends := initial_amount - amount_spent_on_friends
def amount_spent_on_sweets := amount_remaining_after_friends - amount_left

theorem john_spent_on_sweets : amount_spent_on_sweets = 1.05 := 
by
  sorry

end john_spent_on_sweets_l290_290073


namespace num_valid_start_days_for_30_day_month_is_4_l290_290665

def number_of_valid_starting_days (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 4 else 0

theorem num_valid_start_days_for_30_day_month_is_4 :
  number_of_valid_starting_days 30 = 4 :=
by
  unfold number_of_valid_starting_days
  simp [number_of_valid_starting_days]
  sorry

end num_valid_start_days_for_30_day_month_is_4_l290_290665


namespace granola_bars_per_kid_l290_290041

-- Definitions based on the conditions:
def kids : ℕ := 30
def bars_per_box : ℕ := 12
def boxes : ℕ := 5

-- The proof problem statement:
theorem granola_bars_per_kid : (boxes * bars_per_box) / kids = 2 :=
by simp [kids, bars_per_box, boxes]; exact sorry

end granola_bars_per_kid_l290_290041


namespace flight_duration_l290_290042

theorem flight_duration (h m : ℕ) (H1 : 0 < m ∧ m < 60) 
    (H2 : h = 2) (H3 : m = 44) : h + m = 46 :=
by
  rw [H2, H3]
  simp
  sorry

end flight_duration_l290_290042


namespace bug_final_position_l290_290294

-- Define the initial setup
def circle_points := {1, 2, 3, 4, 5}
def initial_position : ℕ := 3
def final_jumps : ℕ := 1995

-- Define the movement function
def move (pos : ℕ) : ℕ :=
  if pos % 2 = 1 then (pos + 1) % 5 else (pos + 3) % 5

-- Recursive function to compute the final position after n jumps
def final_position (pos : ℕ) (jumps : ℕ) : ℕ :=
  match jumps with
  | 0 => pos
  | n + 1 => final_position (move pos) n

-- The main theorem to prove
theorem bug_final_position : final_position initial_position final_jumps = 1 :=
  sorry

end bug_final_position_l290_290294


namespace mixture_weight_l290_290590

theorem mixture_weight (wa wb wc : ℕ) (ra rb rc : ℕ) (total_vol : ℕ) (wg : ℚ) :
  wa = 900 ∧ wb = 700 ∧ wc = 800 ∧ ra = 3 ∧ rb = 2 ∧ rc = 1 ∧ total_vol = 6 →
  wg = (wa * ra + wb * rb + wc * rc) / total_vol / 1000 :=
begin
  intros h,
  obtain ⟨hwa, hwb, hwc, hra, hrb, hrc, htv⟩ := h,
  rw [hwa, hwb, hwc, hra, hrb, hrc],
  ring,
end

end mixture_weight_l290_290590


namespace remainder_when_x_squared_divided_by_20_l290_290386

theorem remainder_when_x_squared_divided_by_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] :=
sorry

end remainder_when_x_squared_divided_by_20_l290_290386


namespace find_length_AB_l290_290043

variable (K A B C D : Point)
variable (AB BC BK DK : ℝ)
variable (b d : ℝ)

-- Given the conditions
axiom cyclic_quadrilateral : cyclic_quadrilateral ABCD
axiom K_is_intersection : K_is_intersection_of_diagonals ABCD K
axiom AB_eq_BC : AB = BC
axiom BK_val : BK = b
axiom DK_val : DK = d

noncomputable def length_AB : ℝ := sqrt (b^2 + b * d)

theorem find_length_AB :
  AB = length_AB :=
sorry

end find_length_AB_l290_290043


namespace problem_2016th_number_l290_290642

def sequence_rule (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 + 2 else n * 2 - 2

def sequence (start : ℕ) : ℕ → ℕ
| 0     => start
| (n+1) => sequence_rule (sequence n)

lemma sequence_cycle : ∀ n, sequence 130 (7 + n) = [8, 6, 5].nth ((n + 2) % 3)
| n => by sorry

theorem problem_2016th_number : sequence 130 2016 = 6 :=
by
  have h : sequence 130 (7 + 2009) = sequence 130 2016 := by rfl
  rw [←h, sequence_cycle]
  have h2 : (2009 + 2) % 3 = 2 := by decide
  rw [h2]
  norm_num
  sorry

end problem_2016th_number_l290_290642


namespace train_length_is_correct_l290_290686

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l290_290686


namespace hyperbola_eccentricity_l290_290787

theorem hyperbola_eccentricity (a b : ℝ) (h_ab : 0 < a ∧ 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_parabola : ∀ x y : ℝ, y^2 = 8 * x)
  (h_common_focus : ∀ x y : ℝ, (x, y) = (2, 0))
  (h_intersection : ∀ x y : ℝ, y = 4 ∨ y = -4)
  (h_distance : ∀ x y : ℝ, x = 2 ∧ (y = 4 ∨ y = -4) → (x - 2)^2 + y^2 = 16) :
  let c := 2 in
  let e := c / a in
  e = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l290_290787


namespace greatest_N_consecutive_n_free_integers_l290_290046

theorem greatest_N_consecutive_n_free_integers {n : ℤ} (hn : n ≥ 2) :
  ∃ N, (N = 2^n - 1) ∧ (∀ N', N' > N → ¬(∃ inf_many_ways_of_consecutive_n_free_integers N' n)) :=
sorry

end greatest_N_consecutive_n_free_integers_l290_290046


namespace y_coordinate_of_C_l290_290084

def Point : Type := (ℤ × ℤ)

def A : Point := (0, 0)
def B : Point := (0, 4)
def D : Point := (4, 4)
def E : Point := (4, 0)

def PentagonArea (C : Point) : ℚ :=
  let triangleArea : ℚ := (1/2 : ℚ) * 4 * ((C.2 : ℚ) - 4)
  let squareArea : ℚ := 4 * 4
  triangleArea + squareArea

theorem y_coordinate_of_C (h : ℤ) (C : Point := (2, h)) : PentagonArea C = 40 → C.2 = 16 :=
by
  sorry

end y_coordinate_of_C_l290_290084


namespace distances_from_D_to_X_and_Y_are_equal_l290_290525

open EuclideanGeometry

-- Definitions for the points and lines:
variables {A B C D E F E' F' X Y : Point}

-- Conditions describing the problem setup:
axiom inscribed_circle_tangent_to_sides : circle_tangent_to_sides A B C D E F
axiom reflection_E_across_DF : E' = reflection E (line D F)
axiom reflection_F_across_DE : F' = reflection F (line D E)
axiom line_EF_intersects_circumcircle_AE'F'_at_XY :
  intersect_at_two_points (line E F) (circumcircle A E' F') X Y

-- The proof goal is to show:
theorem distances_from_D_to_X_and_Y_are_equal :
  distance D X = distance D Y :=
sorry -- Proof omitted

end distances_from_D_to_X_and_Y_are_equal_l290_290525


namespace total_price_of_hats_l290_290566

variables (total_hats : ℕ) (blue_hat_cost : ℕ) (green_hat_cost : ℕ) (green_hats : ℕ) (total_price : ℕ)

def total_number_of_hats := 85
def cost_per_blue_hat := 6
def cost_per_green_hat := 7
def number_of_green_hats := 30

theorem total_price_of_hats :
  (number_of_green_hats * cost_per_green_hat) + ((total_number_of_hats - number_of_green_hats) * cost_per_blue_hat) = 540 :=
sorry

end total_price_of_hats_l290_290566


namespace qin_jiushao_algorithm_l290_290958

theorem qin_jiushao_algorithm (n v x: ℕ) (h_n : n = 5) (h_v : v = 1) (h_x : x = 2) :
  let v_final := (n + 1).fold (λ v _, 2 * v + 1) v in
  v_final = 2^5 + 2^4 + 2^3 + 2^2 + 2 + 1 :=
by
  sorry

end qin_jiushao_algorithm_l290_290958


namespace circumcircle_radii_equal_l290_290657

theorem circumcircle_radii_equal 
  {A B C E F G M N P Q : Point} 
  (hK1 : Circle (B, C))
  (hK2 : (E ≠ C) ∧ (F ≠ B))
  (hE : E ∈ Line (C, A)) 
  (hF : F ∈ Line (A, B))
  (hG : G = LineIntersection (Line (B, E), Line (C, F)))
  (hM : M = SymmetricPoint (A, F))
  (hN : N = SymmetricPoint (A, E))
  (hP : P = Reflection (C, Line (A, G)))
  (hQ : Q = Reflection (B, Line (A, G))) :
  circumcircle_radius (Triangle (B, P, M)) = circumcircle_radius (Triangle (C, Q, N)) := 
sorry

end circumcircle_radii_equal_l290_290657


namespace surface_area_of_sliced_prism_l290_290678

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

structure Prism :=
  (P Q R S T U : Point)
  (height : ℝ)
  (base_side : ℝ)
  (P_to_R : distance P R = base_side)
  (R_to_Q : distance R Q = base_side)
  (Q_to_P : distance Q P = base_side)
  (P_to_S : distance P S = height)
  (Q_to_T : distance Q T = height)
  (R_to_U : distance R U = height)

def V := midpoint Prism.P Prism.R
def W := midpoint Prism.R Prism.Q
def X := midpoint Prism.R Prism.T

theorem surface_area_of_sliced_prism (h : Prism.height = 20) (b : Prism.base_side = 10) :
  surface_area (sliced_prism Prism.P Prism.Q Prism.R Prism.S Prism.T Prism.U V W X) =
    50 + (25 * (Real.sqrt 3) / 4) + (5 * (Real.sqrt 118.75) / 2) :=
by
  sorry

end surface_area_of_sliced_prism_l290_290678


namespace exists_program_l290_290178

-- Define the labyrinth and movement commands
structure Labyrinth :=
  (size : ℕ := 8)
  (partitions : set (ℕ × ℕ))

inductive Command
| Right
| Left
| Up
| Down

def move (lab : Labyrinth) (pos : ℕ × ℕ) (cmd : Command) : ℕ × ℕ :=
  match cmd with
  | Command.Right => if pos.1 < (lab.size - 1) ∧ (pos.1 + 1, pos.2) ∉ lab.partitions then (pos.1 + 1, pos.2) else pos
  | Command.Left  => if pos.1 > 0 ∧ (pos.1 - 1, pos.2) ∉ lab.partitions then (pos.1 - 1, pos.2) else pos
  | Command.Up    => if pos.2 > 0 ∧ (pos.1, pos.2 - 1) ∉ lab.partitions then (pos.1, pos.2 - 1) else pos
  | Command.Down  => if pos.2 < (lab.size - 1) ∧ (pos.1, pos.2 + 1) ∉ lab.partitions then (pos.1, pos.2 + 1) else pos

-- Define the path taken by the rook given a program
def traverse (lab : Labyrinth) (pos : ℕ × ℕ) (program : list Command) : set (ℕ × ℕ) :=
  program.foldl (λ visited cmd, visited ∪ {move lab (pos) cmd}) {pos}

-- Prove the existence of a program that visits all accessible squares
theorem exists_program : ∀ (lab : Labyrinth) (initial_pos : ℕ × ℕ), ∃ (program : list Command), 
  traverse lab initial_pos program = {pos | pos.1 < lab.size ∧ pos.2 < lab.size ∧ pos ∉ lab.partitions ∪ {initial_pos}} :=
by
  sorry

end exists_program_l290_290178


namespace f_f_0_eq_3_pi_squared_minus_4_l290_290396

def f : ℝ → ℝ :=
λ x, if x > 0 then 3 * x^2 - 4 else if x = 0 then Real.pi else 0

theorem f_f_0_eq_3_pi_squared_minus_4 : f (f 0) = 3 * Real.pi^2 - 4 := by 
  sorry

end f_f_0_eq_3_pi_squared_minus_4_l290_290396


namespace common_chord_length_correct_l290_290153

noncomputable def findCommonChordLength : ℝ := 
let R := 13
let r := 5
let d := 12

def commonChordLength (R : ℝ) (r : ℝ) (d : ℝ) : ℝ :=
  let alpha_cos := (R^2 + r^2 - d^2) / (2 * R * r)
  let alpha_sin := sqrt (1 - alpha_cos^2)
  2 * R * alpha_sin

theorem common_chord_length_correct : commonChordLength 13 5 12 = 24 := by
  sorry

end common_chord_length_correct_l290_290153


namespace julia_total_spend_l290_290873

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l290_290873


namespace orthogonal_projection_l290_290889

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end orthogonal_projection_l290_290889


namespace problem_statement_l290_290085

theorem problem_statement (m n p q : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q) :
    let t := (m + n + p + q) / 2 in
    (m / (t + n + p + q) + n / (t + p + q + m) + p / (t + q + m + n) + q / (t + m + n + p)) ≥ 4 / 5 := 
sorry

end problem_statement_l290_290085


namespace problem_part1_problem_part2_l290_290324

noncomputable def f (x : ℝ) := x^2 + 4 * x + 2
noncomputable def g (x : ℝ) := Real.exp (x) * (2 * x + 2)

theorem problem_part1 : 
  ∀ (a b c d : ℝ), 
  (f 0 = 2) ∧ (g 0 = 2) ∧ 
  (f' (idFun) 0 = 4) ∧ (LinearMap.hasDerivAt g' 0 (4 * 1)) 
    → (a = 4) ∧ (b = 2) ∧ (c = 2) ∧ (d = 2) := 
by
  sorry

theorem problem_part2 :
  ∀ k : ℝ, 
  (∀ x ≥ -2, f x ≤ k * g x) ↔ 1 ≤ k ∧ k ≤ Real.exp 2 :=
by
  sorry

end problem_part1_problem_part2_l290_290324


namespace inequality_sum_of_positive_real_numbers_l290_290089

open BigOperators -- for summation notation
open Classical -- for classical logic

variable {α : Type*} [LinearOrderedField α]

theorem inequality_sum_of_positive_real_numbers {n : ℕ} 
  (a : Fin n → α) 
  (h : ∀ i, 0 < a i) : 
  ∑ i in Finset.range n, ∑ j in Finset.range n, i < j ∧ (a i * a j) / (a i + a j) 
      ≤ (n / (2 * ∑ i in Finset.range n, a i)) * ∑ i in Finset.range n, ∑ j in Finset.range n, i < j ∧ (a i * a j) := 
sorry

end inequality_sum_of_positive_real_numbers_l290_290089


namespace part1_part2_l290_290809

def f : (ℝ × ℝ) → ℝ

axiom domain_f : ∀ x y : ℝ, xy ≠ 0 → f (x, y) > 0

axiom axiom1 : ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → f (xy, z) = f (x, z) * f (y, z)
axiom axiom2 : ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → f (x, yz) = f (x, y) * f (x, z)
axiom axiom3 : ∀ x : ℝ, x ≠ 0 → x ≠ 1 → f (x, 1 - x) = 1

theorem part1 (x : ℝ) (h1 : x ≠ 0) : f (x, x) = 1 ∧ f (x, -x) = 1 := 
sorry

theorem part2 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (f (x, y) * f (y, x) = 1) := 
sorry

end part1_part2_l290_290809


namespace three_digit_integers_l290_290379

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≠ 4

def contains_two (n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def does_not_contain_four (n : ℕ) : Prop :=
  ∀ k, n / 10^k % 10 ≠ 4

theorem three_digit_integers : {n : ℕ | is_three_digit n ∧ contains_two n ∧ does_not_contain_four n}.card = 200 := by
  sorry

end three_digit_integers_l290_290379


namespace four_digit_divisible_by_12_l290_290303

theorem four_digit_divisible_by_12 (n : ℕ) : 3150 + n = 3156 → n = 6 :=
by {
    intro h,
    linarith,
    sorry
}

end four_digit_divisible_by_12_l290_290303


namespace g_7_l290_290520

noncomputable theory

variable (g : ℝ → ℝ)

-- Conditions
axiom add_property : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_6 : g 6 = 8

-- To prove
theorem g_7 : g 7 = 28 / 3 :=
sorry

end g_7_l290_290520


namespace root_sum_of_polynomial_l290_290898

noncomputable def polynomial : Polynomial ℂ := Polynomial.C 1 -- λ (x: ℂ), x^4 + a*x^3 + b*x^2 + c*x + d

theorem root_sum_of_polynomial (a b c d : ℝ) (h : (2 + complex.I) ∈ (polynomial.roots (x^4 + a*x^3 + b*x^2 + c*x + Polynomial.C d))) :
  a + b + c + d = 10 :=
sorry

end root_sum_of_polynomial_l290_290898


namespace three_digit_integers_with_at_least_one_two_but_no_four_l290_290372

-- Define the properties
def is_three_digit (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000
def contains_digit (n: ℕ) (d: ℕ) : Prop := ∃ i, i < 3 ∧ d = n / 10^i % 10
def no_four (n: ℕ) : Prop := ¬ contains_digit n 4

-- Define the sets A and B
def setA (n: ℕ) : Prop := is_three_digit n ∧ no_four n
def setB (n: ℕ) : Prop := setA n ∧ ¬ contains_digit n 2

-- The final theorem statement
theorem three_digit_integers_with_at_least_one_two_but_no_four : 
  {n : ℕ | contains_digit n 2 ∧ setA n}.card = 200 :=
sorry

end three_digit_integers_with_at_least_one_two_but_no_four_l290_290372


namespace a4_b4_in_base3_l290_290472

noncomputable def a_seq : ℕ → ℤ
| 0       := 2
| (n + 1) := a_seq n * Int.sqrt(1 + (a_seq n)^2 + (b_seq n)^2) - b_seq n

noncomputable def b_seq : ℕ → ℤ
| 0       := 2
| (n + 1) := b_seq n * Int.sqrt(1 + (a_seq n)^2 + (b_seq n)^2) + a_seq n

theorem a4_b4_in_base3 : 
  (a_seq 4 = 1000001100111222) ∧ (b_seq 4 = 2211100110000012) :=
by
  sorry

end a4_b4_in_base3_l290_290472


namespace arithmetic_sequence_terms_l290_290976

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, a 1 + d = a 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_terms 
  (a : ℕ → ℝ)
  (n : ℕ)
  (seq_arith : arithmetic_sequence a)
  (sum_odd : ∑ i in finset.range n, a (2 * i + 1) = 24)
  (sum_even : ∑ i in finset.range n, a (2 * (i + 1)) = 30)
  (last_term_eq : a (2 * n) = a 1 + 21 / 2) :
  2 * n = 8 :=
begin
  sorry
end

end arithmetic_sequence_terms_l290_290976


namespace remainder_1493827_div_4_l290_290161

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := 
by
  sorry

end remainder_1493827_div_4_l290_290161


namespace coefficient_x_neg2_expansion_l290_290108

noncomputable def binomial_expansion_coefficient : ℤ := -10

theorem coefficient_x_neg2_expansion :
  let general_term (n r : ℕ) (x : ℚ) := ( -1)^r * (Binomial.binom n r) * x^( (5 / 2 : ℚ) - (3 * r / 2 : ℚ)) in
  ∃ r : ℕ, (5 / 2 : ℚ) - (3 * r / 2 : ℚ) = (-2 : ℚ) ∧ 
           general_term 5 r x = (-10 : ℤ) :=
begin
  sorry,
end

end coefficient_x_neg2_expansion_l290_290108


namespace product_mod_5_l290_290276

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l290_290276


namespace graph_no_4_cycles_bound_l290_290467

open Nat

theorem graph_no_4_cycles_bound (G : Type) [Fintype G] [Graph G] (n m : ℕ)
  (h_vertex_count : G.vertex_count = n)
  (h_edge_count : G.edge_count = m)
  (h_no_4_cycles : ∀ (v1 v2 v3 v4 : G), 
    Graph.path v1 v2 v3 ∧ Graph.path v2 v3 v4 ∧ Graph.path v3 v4 v1 ∧ Graph.path v4 v1 v2 → false) :
  m ≤ n / 4 * (1 + sqrt (4 * n - 3)) := by
  sorry

end graph_no_4_cycles_bound_l290_290467


namespace union_is_correct_l290_290185

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}
def union_set : Set ℤ := {-1, 0, 1, 2}

theorem union_is_correct : M ∪ N = union_set :=
  by sorry

end union_is_correct_l290_290185


namespace max_g_equals_six_l290_290826

def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x - Real.sqrt 3 * Real.sin x)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

def g (x a : ℝ) : ℝ :=
  f (x - Real.pi / 6) + a * f (x / 2 - Real.pi / 6) - a * f (x / 2 + Real.pi / 12)

theorem max_g_equals_six (a : ℝ) : 
  (∃ x : ℝ, x ∈ set.Icc 0 Real.pi ∧ g x a = 6) → a = -3 ∨ a = 2 * Real.sqrt 2 :=
sorry

end max_g_equals_six_l290_290826


namespace distance_equality_implies_z_l290_290851

-- Definitions of points A, B, and C
def A : ℝ × ℝ × ℝ := (1, 0, 2)
def B : ℝ × ℝ × ℝ := (1, 1, 1)
def C (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)

-- Function to calculate the Euclidean distance between two points in 3D space
def dist (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Theorem: Given points A, B and a point C on the z-axis, prove that the distances from C to A 
-- and from C to B being equal implies z = 1/2.
theorem distance_equality_implies_z :
  ∀ z : ℝ, dist (C z) A = dist (C z) B → z = 1 / 2 := by
  sorry

end distance_equality_implies_z_l290_290851


namespace area_of_square_l290_290156

-- Given conditions
def diagonal : ℝ := 26
def side_length (d : ℝ) : ℝ := d / Real.sqrt 2

-- Theorem statement
theorem area_of_square : (side_length diagonal) ^ 2 = 338.0625 :=
begin
  sorry
end

end area_of_square_l290_290156


namespace circle_radius_is_sqrt2_l290_290816
open Real

noncomputable def parabola_tangent_circle_radius (x y r : ℝ) : Prop :=
  (x^2 = 4 * y) ∧ 
  ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) ∧
  ∃ (x0 : ℝ), P = (x0, 1/4 * x0 ^ 2) ∧
  let k := (1 / 2) * x0 in
  let kPC := (1/4 * x0 ^ 2 - 2) / (x0 - 1) in
  k * kPC = 1

theorem circle_radius_is_sqrt2 : ∀ (x y r : ℝ), parabola_tangent_circle_radius x y r → r = sqrt 2 :=
by
  sorry

end circle_radius_is_sqrt2_l290_290816


namespace minimum_k_l290_290129

def is_non_decreasing (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i j : ℕ, i < j → i < digits.length → j < digits.length → digits.nth i ≤ digits.nth j

def has_matching_digit_at_same_place (N : ℕ) (M : ℕ) : Prop :=
  let N_digits := N.digits 10 in
  let M_digits := M.digits 10 in
  ∃ i : ℕ, i < 5 ∧ N_digits.nth i = M_digits.nth i

theorem minimum_k
  (N1 N2 : ℕ)
  (hN1 : N1 = 13579)
  (hN2 : N2 = 12468) :
  ∃ k : ℕ, k = 2 ∧
    ∀ M : ℕ, is_non_decreasing M ∧ 10000 ≤ M ∧ M < 100000 →
      (has_matching_digit_at_same_place N1 M ∨ has_matching_digit_at_same_place N2 M) :=
by
  sorry

end minimum_k_l290_290129


namespace sum_Sn_first_10_l290_290792

noncomputable def a_n : ℕ → ℕ
| 0        => 0
| 1        => 1
| (n + 1)  => 2^(n - 1)

noncomputable def S_n : ℕ → ℕ
| n        => ∑ i in Finset.range(n), a_n (i+1)

theorem sum_Sn_first_10 :
  (Finset.range 10).sum S_n = 512 :=
sorry

end sum_Sn_first_10_l290_290792


namespace equilateral_triangle_exists_l290_290087

def is_equilateral_convex_pentagon (P : Type) [ConvexPolygon P] : Prop :=
  P.has_sides 5 ∧ ∀ (i j : Fin 5), P.side_length i = P.side_length j

theorem equilateral_triangle_exists
    (P : Type) [ConvexPolygon P] 
    (hp : is_equilateral_convex_pentagon P) :
    ∃ T : Triangle, is_equilateral T ∧ (∃ (i : Fin 5), T.side 1 = P.side_length i) ∧ T ⊆ P :=
sorry

end equilateral_triangle_exists_l290_290087


namespace shaffiq_steps_to_one_in_five_l290_290094

def shaffiq (n: ℕ): ℕ := n / 3

theorem shaffiq_steps_to_one_in_five :
  let procedure := λ n, List.length (List.unfold (λ x, if x = 1 then none else some (x, shaffiq x)) 250) in
  procedure 250 = 5 := by
  sorry

end shaffiq_steps_to_one_in_five_l290_290094


namespace time_for_B_alone_to_paint_l290_290652

noncomputable def rate_A := 1 / 4
noncomputable def rate_BC := 1 / 3
noncomputable def rate_AC := 1 / 2
noncomputable def rate_DB := 1 / 6

theorem time_for_B_alone_to_paint :
  (1 / (rate_BC - (rate_AC - rate_A))) = 12 := by
  sorry

end time_for_B_alone_to_paint_l290_290652


namespace unfolded_paper_has_four_symmetrical_holes_l290_290236

structure Paper :=
  (width : ℤ) (height : ℤ) (hole_x : ℤ) (hole_y : ℤ)

structure Fold :=
  (direction : String) (fold_line : ℤ)

structure UnfoldedPaper :=
  (holes : List (ℤ × ℤ))

-- Define the initial paper, folds, and punching
def initial_paper : Paper := {width := 4, height := 6, hole_x := 2, hole_y := 1}
def folds : List Fold := 
  [{direction := "bottom_to_top", fold_line := initial_paper.height / 2}, 
   {direction := "left_to_right", fold_line := initial_paper.width / 2}]
def punch : (ℤ × ℤ) := (initial_paper.hole_x, initial_paper.hole_y)

-- The theorem to prove the resulting unfolded paper
theorem unfolded_paper_has_four_symmetrical_holes (p : Paper) (fs : List Fold) (punch : ℤ × ℤ) :
  UnfoldedPaper :=
  { holes := [(1, 1), (1, 5), (3, 1), (3, 5)] } -- Four symmetrically placed holes.

end unfolded_paper_has_four_symmetrical_holes_l290_290236


namespace white_marbles_multiple_of_8_l290_290717

-- Definitions based on conditions
def blue_marbles : ℕ := 16
def num_groups : ℕ := 8

-- Stating the problem
theorem white_marbles_multiple_of_8 (white_marbles : ℕ) :
  (blue_marbles + white_marbles) % num_groups = 0 → white_marbles % num_groups = 0 :=
by
  sorry

end white_marbles_multiple_of_8_l290_290717


namespace simplify_expression_and_evaluate_evaluate_at_neg_one_l290_290507

theorem simplify_expression_and_evaluate :
  ∀ (x : ℝ), x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 2 → (1 - 1 / (x - 1)) / ((x^2 - 4) / (x - 1)) = 1 / (x + 2) :=
begin
  intro x,
  rintros ⟨h1, h2, h3⟩,
  -- Proof would go here
  sorry
end

theorem evaluate_at_neg_one :
  (1 - 1 / (-1 - 1)) / ((-1^2 - 4) / (-1 - 1)) = 1 :=
begin
  -- Proof would go here
  sorry
end

end simplify_expression_and_evaluate_evaluate_at_neg_one_l290_290507


namespace ellipse_foci_distance_l290_290772

theorem ellipse_foci_distance 
  (a b : ℝ) 
  (h_a : a = 8) 
  (h_b : b = 3) : 
  2 * (Real.sqrt (a^2 - b^2)) = 2 * Real.sqrt 55 := 
by
  rw [h_a, h_b]
  sorry

end ellipse_foci_distance_l290_290772


namespace abs_eq_case_l290_290743

theorem abs_eq_case (x : ℝ) (h : |x - 3| = |x + 2|) : x = 1/2 :=
by
  sorry

end abs_eq_case_l290_290743


namespace midpoint_locus_l290_290363

variables {a d : ℝ} (a_nonzero : a ≠ 0) (d_positive : d > 0) (d_large : d > a)

theorem midpoint_locus :
  ∃ (S : set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ S ↔ 
    (z = a / 2 ∧ x^2 + y^2 = (d^2 - a^2) / 4)) ∧
    S = {p | ∃ (x y : ℝ), p = (x, y, a / 2) ∧ x^2 + y^2 = (d^2 - a^2) / 4} :=
by {
  sorry
}

end midpoint_locus_l290_290363


namespace no_x_exists_l290_290466

def X : Set ℚ := { x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 }

def f (x : ℚ) : ℚ := x - 1 / x

def f_n : ℕ → (ℚ → ℚ)
| 1 => f
| (n + 1) => f ∘ f_n n

theorem no_x_exists : ¬ ∃ (x : ℚ) (hx : x ∈ X), ∀ n > 0, ∃ y ∈ X, f_n n y = x := 
by
  sorry

end no_x_exists_l290_290466


namespace line_ellipse_common_points_l290_290119

def point (P : Type*) := P → ℝ × ℝ

theorem line_ellipse_common_points
  (m n : ℝ)
  (no_common_points_with_circle : ∀ (x y : ℝ), mx + ny - 3 = 0 → x^2 + y^2 ≠ 3) :
  ∀ (Px Py : ℝ), (Px = m ∧ Py = n) →
  (∃ (x1 y1 x2 y2 : ℝ), ((x1^2 / 7) + (y1^2 / 3) = 1 ∧ (x2^2 / 7) + (y2^2 / 3) = 1) ∧ (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end line_ellipse_common_points_l290_290119


namespace sequence_sum_2006_l290_290818

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, a n * a (n+1) * a (n+2) = a n + a (n+1) + a (n+2)) ∧
  (∀ n, a (n+1) * a (n+2) ≠ 1)

theorem sequence_sum_2006 (a : ℕ → ℕ) (h : sequence a) : 
  (∑ i in Finset.range 2006, a (i + 1)) = 4011 :=
sorry

end sequence_sum_2006_l290_290818


namespace area_comparison_of_triangles_l290_290843

-- Let's state the given conditions as assumptions in Lean.
variables {A B C P Q D : Type}
variables [h1 : P ∈ segment A C]
variables [h2 : Q = midpoint B P]
variables [h3 : D = intersection (line C Q) (line A B)]

-- Define the areas of the triangles
variables [S_ABP S_ACD : ℝ]

-- Assuming the correct relationship
axiom area_triangle_relationship : S_ABP < S_ACD

-- The goal is to state the proof problem
theorem area_comparison_of_triangles (h1 : P ∈ segment A C) 
                                      (h2 : Q = midpoint B P) 
                                      (h3 : D = intersection (line C Q) (line A B))
                                      (S_ABP S_ACD : ℝ)
                                      (area_triangle_relationship : S_ABP < S_ACD) : 
  S_ABP < S_ACD := 
sorry

end area_comparison_of_triangles_l290_290843


namespace part_1_part_2_l290_290538

-- Definition of the sequence {a_n}
def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 4 * (2 ^ (n - 2))

-- The sum of the first n terms of sequence {a_n}
def S (n : ℕ) : ℕ :=
  seq_a (n + 1) - 2

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℤ :=
  -20 + Int.log2 (seq_a (4 * n))

-- Sum of the first n terms of {c_n} and the minimum value
def T (n : ℕ) : ℤ :=
  2 * n^2 - 18 * n

theorem part_1 (n : ℕ) :
  seq_a n = 2 ^ n :=
by
  sorry

theorem part_2 (T_min : ℤ) :
  (T_min = T 4 ∨ T_min = T 5) ∧ T_min = -40 :=
by
  sorry

end part_1_part_2_l290_290538


namespace sqrt_multiplication_l290_290711

theorem sqrt_multiplication (x : ℝ) : (sqrt (50 * x^2) * sqrt (18 * x^3) * sqrt (98 * x) = 210 * x^3) :=
by 
  sorry

end sqrt_multiplication_l290_290711


namespace large_circle_diameter_proof_l290_290753

-- Define the conditions
def radius_small_circle : ℝ := 4
def number_of_small_circles : ℕ := 8

-- Define a property about the circles' setup
def are_tangent (R r : ℝ) (n : ℕ) :=
  ∀ i : ℕ, i < n → ∀ j : ℕ, j < n → ((dist R r) = 0 ∨ (dist r r) = 2 * radius_small_circle)

-- Calculate the required dimensions
def side_length_inner_octagon : ℝ := 2 * radius_small_circle
def radius_inner_octagon (s : ℝ) (n : ℕ) : ℝ := s / (2 * Real.tan (Real.pi / n))
def radius_large_circle (r_in : ℝ) : ℝ := r_in + radius_small_circle
def diameter_large_circle (R_large : ℝ) : ℝ := 2 * R_large

noncomputable def large_circle_diameter
  (R : ℝ) (r : ℝ) (s : ℝ) (n : ℕ) [Fact (r = radius_inner_octagon s n)] 
  [Fact (s = side_length_inner_octagon)] [Fact (n = number_of_small_circles)] : Prop :=
diameter_large_circle (radius_large_circle r) = 27.32

-- Main statement to be proved
theorem large_circle_diameter_proof : large_circle_diameter _ _ _ _ :=
by sorry

end large_circle_diameter_proof_l290_290753


namespace john_billed_minutes_l290_290298

theorem john_billed_minutes
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (total_bill : ℝ := 12.02) :
  ∃ (minutes : ℕ), minutes = 28 :=
by
  have amount_for_minutes := total_bill - monthly_fee
  have minutes_float := amount_for_minutes / cost_per_minute
  have minutes := floor minutes_float
  use minutes
  have : 28 = (floor minutes_float : ℕ) := sorry
  exact this.symm

end john_billed_minutes_l290_290298


namespace minimum_phi_l290_290017

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

theorem minimum_phi (φ : ℝ) (h1 : ∀ x : ℝ, sin(2 * (x - φ) + π / 4) = sin(2 * x + π / 4 - 2 * φ))
  (h2 : ∀ x : ℝ, sin(2 * x + π / 4 - 2 * φ) = sin(-2 * x - π / 4 + 2 * φ)) :
  (φ = 3 * π / 8) :=
sorry

end minimum_phi_l290_290017


namespace total_games_played_l290_290747

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l290_290747


namespace shooting_frequency_l290_290191

theorem shooting_frequency (total_shots successful_shots : ℕ) (h1 : total_shots = 90) (h2 : successful_shots = 63) :
  (successful_shots : ℝ) / (total_shots : ℝ) = 0.7 :=
by
  rw [h1, h2]
  norm_num
  sorry

end shooting_frequency_l290_290191


namespace train_length_250_05_l290_290695

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l290_290695


namespace tangent_line_at_point_l290_290518

open Real

theorem tangent_line_at_point (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (∀ x, f x = x^3 + a * x) →
  f 1 = 2 →
  (∀ x, deriv f x = 3 * x^2 + a) →
  (∀ x y, tangent_line f 1 2 x y = 4 * x - 2) :=
by
  sorry

end tangent_line_at_point_l290_290518


namespace max_possible_value_l290_290909

noncomputable def maximum_Z (n : ℕ) : ℝ :=
  max (λ x : Fin (2 * n) → ℝ, ∑ 1 ≤ r < s < 2 * n, (s - r - n) * x r * x s)
       {x | ∀ i, -1 ≤ x i ∧ x i ≤ 1}

theorem max_possible_value (n : ℕ) (h : 0 < n):
  maximum_Z n = n * (n - 1) := by
    sorry

end max_possible_value_l290_290909


namespace sequence_2017_l290_290839

theorem sequence_2017 (a : ℕ → ℚ) 
  (h1 : a 1 = 12)
  (h2 : ∀ n ≥ 1, (finset.range n).sum (λ k => (k + 1) * a (k + 1)) = n^2 * a n) :
  a 2017 = 12 / 2017 := 
sorry

end sequence_2017_l290_290839


namespace solution_l290_290331

noncomputable def problem_statement (a b : ℝ) : Prop :=
  a > b ∧ b > 1 ∧
  log a b + log b a = 5 / 2 ∧ 
  a^b = b^a → 
  a / (b + 2) = 1

theorem solution : ∃ a b : ℝ, problem_statement a b :=
begin
  sorry
end

end solution_l290_290331


namespace continuous_zero_solution_l290_290780

variable {R : Type*} [LinearOrderedField R] [TopologicalSpace R] 

noncomputable def f : R → R := sorry

theorem continuous_zero_solution (f : R → R) (n : ℕ)
  (h_cont : Continuous f)
  (h_eq : ∀ x : R, ∑ k in Finset.range (n+1), (Nat.choose n k) * f (x ^ (2^k)) = 0) :
  f = 0 :=
by
  sorry

end continuous_zero_solution_l290_290780


namespace count_divisors_l290_290370

theorem count_divisors :
  let odd_divisors := finset.filter (fun n => (nat.sqrt n)^2 = n) (finset.range 100)
  let even_divisors := finset.filter (fun n => (nat.sqrt n)^2 ≠ n) (finset.range 100)
  odd_divisors.card = 9 ∧ even_divisors.card = 90 :=
by
  let odd_divisors := finset.filter (fun n => (nat.sqrt n)^2 = n) (finset.range 100)
  let even_divisors := finset.filter (fun n => (nat.sqrt n)^2 ≠ n) (finset.range 100)
  have h1 : odd_divisors.card = 9 := sorry
  have h2 : even_divisors.card = 90 := sorry
  exact ⟨h1, h2⟩

end count_divisors_l290_290370


namespace correct_fraction_of_day_l290_290650

-- Define the defective clock properties
def defective_clock (hours : ℕ) (minutes : ℕ) : Prop :=
  ∀ h m, (h == 1 ∨ h == 10 ∨ h == 11 ∨ h == 12 ∨
          m / 10 == 1 ∨ m % 10 == 1) → 
         (h == 7 ∨ h == 70 ∨ h == 71 ∨ h == 72 ∨
          m / 10 == 7 ∨ m % 10 == 7)

-- Define the correct fraction for hours and minutes
def correct_fraction_hours : ℚ := 2 / 3
def correct_fraction_minutes : ℚ := 3 / 4

-- The fraction of the day the clock is correct
theorem correct_fraction_of_day : 
  (correct_fraction_hours * correct_fraction_minutes) = 1 / 2 :=
by {
  have h : correct_fraction_hours = 2 / 3, from rfl,
  have m : correct_fraction_minutes = 3 / 4, from rfl,
  calc
    correct_fraction_hours * correct_fraction_minutes
      = (2 / 3) * (3 / 4) : by rw [h, m]
  ... = (2 * 3) / (3 * 4) : by rw rat.mul_def
  ... = 6 / 12 : by norm_num
  ... = 1 / 2 : by norm_num
}

end correct_fraction_of_day_l290_290650


namespace royal_children_count_l290_290609

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290609


namespace sufficient_not_necessary_condition_l290_290311

variable (x a : ℝ)

def p := x ≤ -1
def q := a ≤ x ∧ x < a + 2

-- If q is sufficient but not necessary for p, then the range of a is (-∞, -3]
theorem sufficient_not_necessary_condition : 
  (∀ x, q x a → p x) ∧ ∃ x, p x ∧ ¬ q x a → a ≤ -3 :=
by
  sorry

end sufficient_not_necessary_condition_l290_290311


namespace both_badminton_and_tennis_l290_290410

namespace SportsClub

def total_members : ℕ := 150
def badminton_players : ℕ := 75
def tennis_players : ℕ := 60
def neither_players : ℕ := 25

theorem both_badminton_and_tennis :
  ∃ X : ℕ, (badminton_players + tennis_players - X = total_members - neither_players) ∧ X = 10 :=
by
  use 10
  split
  · calc
      badminton_players + tennis_players - 10
        = 75 + 60 - 10 : rfl
    ... = 135 - 10 : by norm_num
    ... = 125 : by norm_num
    ... = total_members - neither_players : by norm_num
  · rfl

end both_badminton_and_tennis_l290_290410


namespace series_equality_l290_290064

open Classical

noncomputable def series_sum (x y : ℝ) (hx : 1 < x) (hy : 1 < y) : ℝ :=
  ∑' n : ℕ, 1 / (x^(3^n) - y^(-3^n))

theorem series_equality (x y : ℝ) (hx : 1 < x) (hy : 1 < y) :
  series_sum x y hx hy = 1 / (x * y - 1) :=
sorry

end series_equality_l290_290064


namespace find_a_of_tangent_line_l290_290517

theorem find_a_of_tangent_line (y : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, y x = a ^ x)
    (h2 : ∀ x, x * real.log 2 + y 0 - 1 = 0) : 
    a = 1 / 2 :=
by
  sorry

end find_a_of_tangent_line_l290_290517


namespace probability_shot_condition_l290_290218

noncomputable def probability_of_hitting (p : ℝ) : Prop :=
  let miss := 1 - p in
  (miss ^ 2) * p = 3 / 16

theorem probability_shot_condition :
  probability_of_hitting (3 / 4) :=
by
  sorry

end probability_shot_condition_l290_290218


namespace prime_divides_sequence_term_l290_290881

theorem prime_divides_sequence_term (k : ℕ) (h_prime : Nat.Prime k) (h_ne_two : k ≠ 2) (h_ne_five : k ≠ 5) :
  ∃ n ≤ k, k ∣ (Nat.ofDigits 10 (List.replicate n 1)) :=
by
  sorry

end prime_divides_sequence_term_l290_290881


namespace find_value_of_fraction_l290_290063

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l290_290063


namespace boat_upstream_time_is_1_5_hours_l290_290194

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l290_290194


namespace div_eq_frac_l290_290709

theorem div_eq_frac : 250 / (5 + 12 * 3^2) = 250 / 113 :=
by
  sorry

end div_eq_frac_l290_290709


namespace remainder_of_product_mod_5_l290_290280

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l290_290280


namespace problem_solution_l290_290460

theorem problem_solution
  (h : ∃ (x : ℝ), 
        ( 5 / (x - 4) + 6 / (x - 6) + 14 / (x - 14) + 15 / (x - 15) = x^2 - 13 * x - 6 )
        ∧ (∀ d e f : ℕ, d + sqrt (e + sqrt f) = x) )
: (∃ d e f : ℕ, d + e + f = 3565) :=
sorry

end problem_solution_l290_290460


namespace samuel_teacups_left_l290_290937

-- Define the initial conditions
def total_boxes := 60
def pans_boxes := 12
def decoration_fraction := 1 / 4
def decoration_trade := 3
def trade_gain := 1
def teacups_per_box := 6 * 4 * 2
def broken_per_pickup := 4

-- Calculate the number of boxes initially containing teacups
def remaining_boxes := total_boxes - pans_boxes
def decoration_boxes := decoration_fraction * remaining_boxes
def initial_teacup_boxes := remaining_boxes - decoration_boxes

-- Adjust the number of teacup boxes after the trade
def teacup_boxes := initial_teacup_boxes + trade_gain

-- Calculate total number of teacups and the number of teacups broken
def total_teacups := teacup_boxes * teacups_per_box
def total_broken := teacup_boxes * broken_per_pickup

-- Calculate the number of teacups left
def teacups_left := total_teacups - total_broken

-- State the theorem
theorem samuel_teacups_left : teacups_left = 1628 := by
  sorry

end samuel_teacups_left_l290_290937


namespace intersection_area_correct_l290_290560

noncomputable def circle_center_1 : ℝ × ℝ := (0, 0)
noncomputable def radius_1 : ℝ := 1

noncomputable def circle_center_2 : ℝ × ℝ := (2, 0)
noncomputable def radius_2 : ℝ := 2

-- The distance between the circle centers
noncomputable def distance : ℝ := dist circle_center_1 circle_center_2

-- The area of the intersection of the two circles
noncomputable def intersection_area (r1 r2 d : ℝ) : ℝ :=
  r1^2 * real.arccos((d^2 + r1^2 - r2^2) / (2 * d * r1)) +
  r2^2 * real.arccos((d^2 + r2^2 - r1^2) / (2 * d * r2)) -
  0.5 * real.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

theorem intersection_area_correct :
  intersection_area radius_1 radius_2 distance = 0.904 :=
sorry

end intersection_area_correct_l290_290560


namespace royal_family_children_l290_290641

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290641


namespace area_of_field_l290_290673

theorem area_of_field (L W A : ℝ) (hL : L = 20) (hP : L + 2 * W = 25) : A = 50 :=
by
  sorry

end area_of_field_l290_290673


namespace birds_count_is_30_l290_290219

def total_animals : ℕ := 77
def number_of_kittens : ℕ := 32
def number_of_hamsters : ℕ := 15

def number_of_birds : ℕ := total_animals - number_of_kittens - number_of_hamsters

theorem birds_count_is_30 : number_of_birds = 30 := by
  sorry

end birds_count_is_30_l290_290219


namespace smaller_angle_measure_l290_290994

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l290_290994


namespace find_k_value_l290_290030

/-- 
 Given the ellipse \( C \) defined by the equation \( x^2 + \frac{y^2}{4} = 1 \), 
 and a line \( y = kx + 1 \) intersecting \( C \) at points \( A \) and \( B \) such that 
 \( \overrightarrow{OA} \perp \overrightarrow{OB} \),
 prove that \( k = \pm \frac{1}{2} \).
 -/
theorem find_k_value :
  ∀ k : ℝ, (∃ A B : ℝ × ℝ, 
    (A ≠ B) ∧ 
    (A.2 = k * A.1 + 1) ∧ 
    (B.2 = k * B.1 + 1) ∧ 
    (A.1^2 + (A.2^2) / 4 = 1) ∧ 
    (B.1^2 + (B.2^2) / 4 = 1) ∧ 
    (A.1 * B.1 + A.2 * B.2 = 0)) 
  ∃ (k_val : ℝ), k_val = 1 / 2 ∨ k_val = -1 / 2 := 
sorry

end find_k_value_l290_290030


namespace smallest_n_for_n_cubed_ends_in_888_l290_290056

/-- Proof Problem: Prove that 192 is the smallest positive integer \( n \) such that the last three digits of \( n^3 \) are 888. -/
theorem smallest_n_for_n_cubed_ends_in_888 : ∃ n : ℕ, n > 0 ∧ (n^3 % 1000 = 888) ∧ ∀ m : ℕ, 0 < m ∧ (m^3 % 1000 = 888) → n ≤ m :=
by
  sorry

end smallest_n_for_n_cubed_ends_in_888_l290_290056


namespace sum_of_eight_smallest_multiples_of_6_times_2_eq_432_l290_290166

theorem sum_of_eight_smallest_multiples_of_6_times_2_eq_432 :
  (∑ i in finset.range 8, 2 * 6 * (i + 1)) = 432 :=
begin
  sorry
end

end sum_of_eight_smallest_multiples_of_6_times_2_eq_432_l290_290166


namespace simplify_expression_l290_290506

variable (x : Int)

theorem simplify_expression : 3 * x + 5 * x + 7 * x = 15 * x :=
  by
  sorry

end simplify_expression_l290_290506


namespace solve_fraction_equation_l290_290092

theorem solve_fraction_equation (x : ℚ) (h : (x + 7) / (x - 4) = (x - 5) / (x + 3)) : x = -1 / 19 := 
sorry

end solve_fraction_equation_l290_290092


namespace quadratic_condition_interval_length_l290_290268

theorem quadratic_condition_interval_length (a : ℝ) (f : ℝ → ℝ) :
  (∀ x ∈ set.Icc (-2:ℝ) 0, |f x| ≤ 3) →
  (f = λ x, a*x^2 + 2*a*x - 1) →
  (a ∈ set.Icc (-4:ℝ) 0 ∨ a ∈ set.Ioc 0 2) →
  true :=
begin
  sorry
end

end quadratic_condition_interval_length_l290_290268


namespace find_cos_F1PF2_l290_290329

noncomputable def cos_angle_P_F1_F2 : ℝ :=
  let F1 := (-(4:ℝ), 0)
  let F2 := ((4:ℝ), 0)
  let a := (5:ℝ)
  let b := (3:ℝ)
  let P : ℝ × ℝ := sorry -- P is a point on the ellipse
  let area_triangle : ℝ := 3 * Real.sqrt 3
  let cos_angle : ℝ := 1 / 2
  cos_angle

def cos_angle_F1PF2_lemma (F1 F2 : ℝ × ℝ) (ellipse_Area : ℝ) (cos_angle : ℝ) : Prop :=
  cos_angle = 1/2

theorem find_cos_F1PF2 (a b : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Area_PF1F2 : ℝ) :
  (F1 = (-(4:ℝ), 0) ∧ F2 = ((4:ℝ), 0)) ∧ (Area_PF1F2 = 3 * Real.sqrt 3) ∧
  (P.1^2 / (a^2) + P.2^2 / (b^2) = 1) → cos_angle_F1PF2_lemma F1 F2 Area_PF1F2 (cos_angle_P_F1_F2)
:=
  sorry

end find_cos_F1PF2_l290_290329


namespace five_digit_palindromes_count_l290_290736

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l290_290736


namespace circle_area_from_polar_l290_290521

theorem circle_area_from_polar :
  (∀ (r θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ + 5 → 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
  center = (-3 / 2, -2) ∧ radius = 3 / 2 ∧ 
  real.pi * radius^2 = (9 / 4) * real.pi) :=
sorry

end circle_area_from_polar_l290_290521


namespace vector_dot_product_l290_290400

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end vector_dot_product_l290_290400


namespace time_upstream_is_correct_l290_290198

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l290_290198


namespace infinite_series_sum_l290_290755

theorem infinite_series_sum :
  (∑ n : ℕ, (3^n : ℝ)/(7^(2^n : ℕ) + 1)) = (1/6 : ℝ) :=
sorry

end infinite_series_sum_l290_290755


namespace function_symmetric_about_l290_290345

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem function_symmetric_about (x : ℝ) : 
  f (x) = f (Real.pi / 4 - x) :=
sorry

end function_symmetric_about_l290_290345


namespace ratio_of_B_to_C_l290_290171

variables (A B C : ℕ)

-- Conditions from the problem
axiom h1 : A = B + 2
axiom h2 : A + B + C = 12
axiom h3 : B = 4

-- Goal: Prove that the ratio of B's age to C's age is 2
theorem ratio_of_B_to_C : B / C = 2 :=
by {
  sorry
}

end ratio_of_B_to_C_l290_290171


namespace greatest_four_digit_number_divisible_by_6_and_12_l290_290569

theorem greatest_four_digit_number_divisible_by_6_and_12 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧ (n % 12 = 0) ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 6 = 0) ∧ (m % 12 = 0) → m ≤ n) ∧
  n = 9996 := 
by
  sorry

end greatest_four_digit_number_divisible_by_6_and_12_l290_290569


namespace measure_of_smaller_angle_l290_290996

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l290_290996


namespace max_value_expression_l290_290863

theorem max_value_expression (a b c d : ℕ) 
  (ha : a ∈ {1, 2, 3, 4})
  (hb : b ∈ {1, 2, 3, 4})
  (hc : c ∈ {1, 2, 3, 4})
  (hd : d ∈ {1, 2, 3, 4})
  (h_distinct : (a, b, c, d).val.nodup) :
  c * a^b - d ≤ 127 :=
sorry

end max_value_expression_l290_290863


namespace greatest_number_is_2040_l290_290555

theorem greatest_number_is_2040 (certain_number : ℕ) : 
  (∀ d : ℕ, d ∣ certain_number ∧ d ∣ 2037 → d ≤ 1) ∧ 
  (certain_number % 1 = 10) ∧ 
  (2037 % 1 = 7) → 
  certain_number = 2040 :=
by
  sorry

end greatest_number_is_2040_l290_290555


namespace intersection_correct_l290_290800

-- Define the sets A and B
def A := {-1, 0, 1, 2}
def B := {x | -1 < x ∧ x < 2}

-- Define the intersection of A and B
def Intersection := {x | x ∈ A ∧ x ∈ B}

-- The theorem we need to prove
theorem intersection_correct : Intersection = {0, 1} :=
sorry

end intersection_correct_l290_290800


namespace gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l290_290779

def gervais_distance_miles_per_day : Real := 315
def gervais_days : Real := 3
def gervais_km_per_mile : Real := 1.60934

def henri_total_miles : Real := 1250
def madeleine_distance_miles_per_day : Real := 100
def madeleine_days : Real := 5

def gervais_total_km := gervais_distance_miles_per_day * gervais_days * gervais_km_per_mile
def henri_total_km := henri_total_miles * gervais_km_per_mile
def madeleine_total_km := madeleine_distance_miles_per_day * madeleine_days * gervais_km_per_mile

def combined_total_km := gervais_total_km + henri_total_km + madeleine_total_km

theorem gervais_km_correct : gervais_total_km = 1520.82405 := sorry
theorem henri_km_correct : henri_total_km = 2011.675 := sorry
theorem madeleine_km_correct : madeleine_total_km = 804.67 := sorry
theorem total_km_correct : combined_total_km = 4337.16905 := sorry
theorem henri_drove_farthest : henri_total_km = 2011.675 := sorry

end gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l290_290779


namespace nth_term_of_sequence_l290_290360

theorem nth_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 * 2^n - 3) →
  a 1 = S 1 →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) →
  (∀ n : ℕ, n ≥ 1 → a n = 3 * 2 ^ (n - 1)) :=
by
  intros hS h1 h2 n hn
  induction n with pn hpn
  case zero =>
    simp at hn
  case succ pn =>
    cases pn
    case zero =>
      simp [h1, hS]
    case succ pn =>
      simp [h2, hS, hpn]
  sorry

end nth_term_of_sequence_l290_290360


namespace minimal_integers_seq_l290_290877

theorem minimal_integers_seq :
    ∃ (x : ℕ → ℝ), 
    x 0 = 0 ∧ 
    x 2 = real.cbrt 2 * x 1 ∧ 
    (∀ n ≥ 2, x (n + 1) = (1 / real.cbrt 4) * x n + real.cbrt 4 * x (n - 1) + (1 / 2) * x (n - 2)) ∧ 
    x 3 ∈ ℤ ∧ 
    x 3 > 0 ∧ 
    (∀ i, i ≠ 0 ∧ i ≠ 3 ∧ ¬(∃ k : ℕ, i = 6 * k ∨ i = 6 * k + 1 ∨ i = 6 * k + 2 ∨ i = 6 * k + 3 ∨ i = 6 * k + 4 \∨ i = 6 * k + 5) → x i ∉ ℤ) :=
sorry

end minimal_integers_seq_l290_290877


namespace percentage_students_went_on_trip_l290_290238

theorem percentage_students_went_on_trip
  (total_students : ℕ)
  (students_march : ℕ)
  (students_march_more_than_100 : ℕ)
  (students_june : ℕ)
  (students_june_more_than_100 : ℕ)
  (total_more_than_100_either_trip : ℕ) :
  total_students = 100 → students_march = 20 → students_march_more_than_100 = 7 →
  students_june = 15 → students_june_more_than_100 = 6 →
  70 * total_more_than_100_either_trip = 7 * 100 →
  (students_march + students_june) * 100 / total_students = 35 :=
by
  intros h_total h_march h_march_100 h_june h_june_100 h_total_100
  sorry

end percentage_students_went_on_trip_l290_290238


namespace measure_of_smaller_angle_l290_290997

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l290_290997


namespace cos_sin_power_sixty_l290_290718

noncomputable def cos_135 := Real.cos (135 * Real.pi / 180)
noncomputable def sin_135 := Real.sin (135 * Real.pi / 180)

theorem cos_sin_power_sixty : (complex.mk cos_135 sin_135) ^ 60 = -1 :=
sorry

end cos_sin_power_sixty_l290_290718


namespace arithmetic_sequence_properties_l290_290321

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, S n = ∑ i in finset.range (n + 1), a i) →
  (S 3 + S 4 = S 5) →
  (2 * S 5 = S 4 + S 4) →
  (a 5 = 3 * a 2 + 2 * a 1 - 2) →
  (a n = 2 * n - 1) →
  (b n = 2 ^ (n - 1)) →
  (∀ n, T n = ∑ i in finset.range n, (a i) / (b i)) →
  (T n = 6 - (2 * n + 3) / 2 ^ (n - 1)) :=
by
  intros hS hSeq hS5 hEq a_n_formula b_n_formula hT_target
  -- Proof here
  sorry

end arithmetic_sequence_properties_l290_290321


namespace fraction_sum_correct_l290_290165

noncomputable def fraction_sum : ℝ :=
  ∑ n in finset.range 2009, 2 / (n+1) / (n+3)

theorem fraction_sum_correct : abs (fraction_sum - 1.499) < 0.001 := 
  sorry

end fraction_sum_correct_l290_290165


namespace optimal_X_for_Hagrid_l290_290100

noncomputable def optimal_distance (d_island d_shore v_shore v_sea : ℝ) : ℝ :=
  let p := 1 / v_shore
  let q := 1 / v_sea
  let sqrt_term := (d_shore * d_shore - d_island * d_island * (q * q - p * p)) / (d_island * d_island)
  let x := if sqrt_term > 0 then real.sqrt sqrt_term else 0
  x

theorem optimal_X_for_Hagrid :
  optimal_distance 9 15 50 40 = 3 :=
by sorry

end optimal_X_for_Hagrid_l290_290100


namespace evaluate_polynomial_at_minus_two_l290_290260

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_polynomial_at_minus_two :
  P (-2) = -18 :=
by
  sorry

end evaluate_polynomial_at_minus_two_l290_290260


namespace quadratic_equation_with_given_roots_l290_290340

theorem quadratic_equation_with_given_roots :
  (∃ (x : ℝ), (x - 3) * (x + 4) = 0 ↔ x = 3 ∨ x = -4) :=
by
  sorry

end quadratic_equation_with_given_roots_l290_290340


namespace pam_total_apples_l290_290495

theorem pam_total_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) (pam_apples_per_bag : ℕ) (gerald_bags_for_pam_bag : ℕ) :
  pam_bags = 10 →
  gerald_apples_per_bag = 40 →
  gerald_bags_for_pam_bag = 3 →
  pam_apples_per_bag = gerald_bags_for_pam_bag * gerald_apples_per_bag →
  pam_bags * pam_apples_per_bag = 1200 := 
by
  intros h_pam_bags h_gerald_apples h_gerald_bags_for_pam h_pam_apples
  rw [h_pam_bags, h_gerald_apples, h_gerald_bags_for_pam, h_pam_apples]
  calc
    10 * (3 * 40) = 10 * 120 : by rfl
               ... = 1200 : by rfl
  done

end pam_total_apples_l290_290495


namespace bucket_full_weight_l290_290651

variables (p q x y : ℝ)

theorem bucket_full_weight :
  (x + (1 / 4) * y = p) →
  (x + (3 / 4) * y = q) →
  (x + y = - (1 / 2) * p + (3 / 2) * q) :=
by
  intros h1 h2
  have h := sub_eq_of_eq_add' h1 h2
  have h3 : (3 / 4) * y - (1 / 4) * y = q - p, from h
  sorry

end bucket_full_weight_l290_290651


namespace isosceles_right_triangle_area_l290_290028

theorem isosceles_right_triangle_area (A B C D : Type)
  [triangle A B C] [right_angle ∠B] [angle ∠A = 45°] [angle ∠C = 45°]
  (BD : line_segment B D) (hBD : BD.length = 1)
  (isosceles_right_triangle : isosceles_right_triangle A B C ∠B)
  : area (triangle A B C) = 1 :=
sorry

end isosceles_right_triangle_area_l290_290028


namespace quadratic_function_value_at_neg_one_l290_290349

theorem quadratic_function_value_at_neg_one (b c : ℝ) 
  (h1 : (1:ℝ) ^ 2 + b * 1 + c = 0) 
  (h2 : (3:ℝ) ^ 2 + b * 3 + c = 0) : 
  ((-1:ℝ) ^ 2 + b * (-1) + c = 8) :=
by
  sorry

end quadratic_function_value_at_neg_one_l290_290349


namespace chipped_marbles_bag_count_l290_290552

variables (j g : ℕ) (marbles_bags : Set ℕ)
  (bags : Finset ℕ)
  (bag_contents : ∀ b : ℕ, b ∈ bags → b ∈ marbles_bags)
  (total_marbles : ℕ := 221)
  (picked_bags : bags.card = 6)
  (remaining_bags : bags.sdiff (finset.sum [ j, 3 * g ]))

theorem chipped_marbles_bag_count 
  (h1 : marbles_bags = {17, 20, 22, 24, 26, 35, 37, 40})
  (h2 : total_marbles = 221)
  (h3 : ∃ j g : ℕ, j = 3 * g)
  (h4 : ∀ remaining_bags, remaining_bags.card = 2)
  (h5 : ∃ a b : ℕ, a + b = total_marbles - (j + g) ∧ a ∈ marbles_bags ∧ b ∈ marbles_bags)
  : ∃ c : ℕ, c = 40 := sorry

end chipped_marbles_bag_count_l290_290552


namespace n_value_for_315n_divisible_by_12_l290_290306

theorem n_value_for_315n_divisible_by_12 :
  ∃ n : ℕ, (n < 10) ∧ ((10 + n) % 4 = 0) ∧ ((9 + n) % 3 = 0) ∧ n = 6 :=
by
  use 6
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end n_value_for_315n_divisible_by_12_l290_290306


namespace repeating_decimal_to_fraction_l290_290570

theorem repeating_decimal_to_fraction :
  let x := 0.\overline{246}
  let y := 0.\overline{135}
  let z := 0.\overline{369}
  x + y == \frac{381}{999} → 
  (x + y) * z == \frac{140529}{998001} := 
by
  sorry

end repeating_decimal_to_fraction_l290_290570


namespace log_inequality_domain_constraints_l290_290093

theorem log_inequality (x : Real) (k : Int) (hw : 0 < cos x ∧ cos x < 1) (hpx : x^2 - 6 * x - 2 > 0) :
    log (cos x) (x^2 - 6 * x - 2) > 2 / log 5 (cos x) := sorry

-- Domain conditions
theorem domain_constraints (x : Real) (k : Int) :
    (x^2 - 6 * x - 2 > 0) ->
    (2 * k * π - π / 2 < x ∧ x < 2 * k * π) ∨ (2 * k * π < x ∧ x < 2 * k * π + π / 2) :=
sorry

end log_inequality_domain_constraints_l290_290093


namespace coat_price_reduction_l290_290535

theorem coat_price_reduction:
  ∀ (original_price reduction_amount : ℕ),
  original_price = 500 →
  reduction_amount = 350 →
  (reduction_amount : ℝ) / original_price * 100 = 70 :=
by
  intros original_price reduction_amount h1 h2
  sorry

end coat_price_reduction_l290_290535


namespace percentage_of_rotten_bananas_l290_290675

theorem percentage_of_rotten_bananas :
  ∀ (total_oranges total_bananas : ℕ) 
    (percent_rotten_oranges : ℝ) 
    (percent_good_fruits : ℝ), 
  total_oranges = 600 → total_bananas = 400 → 
  percent_rotten_oranges = 0.15 → percent_good_fruits = 0.89 → 
  (100 - (((percent_good_fruits * (total_oranges + total_bananas)) - 
  ((1 - percent_rotten_oranges) * total_oranges)) / total_bananas) * 100) = 5 := 
by
  intros total_oranges total_bananas percent_rotten_oranges percent_good_fruits 
  intro ho hb hro hpf 
  sorry

end percentage_of_rotten_bananas_l290_290675


namespace find_value_of_fraction_l290_290061

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l290_290061


namespace sum_of_first_2007_terms_is_zero_l290_290222

def b : ℕ → ℤ
| 0       := 0  -- Not defined in the problem, assumed to be zero for index 0
| 1       := u
| 2       := v
| n+3     := 2 * b (n+2) - b (n+1)

def sum_seq : ℕ → ℤ
| 0       := 0
| n+1     := sum_seq n + b (n+1)

theorem sum_of_first_2007_terms_is_zero (u v : ℤ) 
  (h1 : sum_seq 2011 = 0) 
  (h2 : sum_seq 2017 = 0) : 
  sum_seq 2007 = 0 := 
sorry

end sum_of_first_2007_terms_is_zero_l290_290222


namespace conner_average_speed_and_fuel_efficiency_l290_290719

/-- 
  Conner's average speed and fuel consumption in the desert.
  Assume the following conditions:
  1. Speed on flat sand: 60 mph.
  2. Speed on downhill slopes: 12 mph faster than on flat sand.
  3. Speed on uphill slopes: 18 mph slower than on flat sand.
  4. Wind resistance impact: -5% when headwind, +5% when tailwind.
  5. Fuel efficiency changes: +20% more fuel on uphill, -10% less fuel on downhill.
  6. One-third of the time on each terrain: flat sand, uphill, downhill.
  7. Headwind while uphill, tailwind while downhill, no wind impact on flat sand.
-/
theorem conner_average_speed_and_fuel_efficiency :
  let flat_speed := 60
  let downhill_speed := flat_speed + 12
  let uphill_speed := flat_speed - 18
  
  let downhill_speed_tailwind := downhill_speed * 1.05
  let uphill_speed_headwind := uphill_speed * 0.95
  
  let average_speed := (flat_speed + downhill_speed_tailwind + uphill_speed_headwind) / 3
  
  let x_mpg := 1
  let downhill_fuel_eff := x_mpg * 1.1
  let uphill_fuel_eff := x_mpg * 0.8
  
  let average_fuel_eff := (x_mpg + downhill_fuel_eff + uphill_fuel_eff) / 3
  
  average_speed = 58.5 ∧ average_fuel_eff ≈ 0.9667 * x_mpg :=
by
  sorry 

end conner_average_speed_and_fuel_efficiency_l290_290719


namespace tan_subtraction_l290_290012

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 55 :=
by
  sorry

end tan_subtraction_l290_290012


namespace base_4_representation_of_101010101_l290_290568

theorem base_4_representation_of_101010101 :
  ∀ (n : ℕ), n = 101010101_2.to_nat → BaseRepresentation n 4 = "11111" := 
by
  intro n h
  rw [← h]
  have : 101010101_2.to_nat = 341 := sorry
  rw [this]
  show BaseRepresentation 341 4 = "11111"
  sorry

end base_4_representation_of_101010101_l290_290568


namespace dennis_floor_l290_290731

theorem dennis_floor :
  ∀ (Dennis Charlie Frank : ℕ), 
    Frank = 16 →
    Charlie = Frank / 4 →
    Dennis = Charlie + 2 →
    Dennis = 6 :=
by
  intro Dennis Charlie Frank
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end dennis_floor_l290_290731


namespace circle_reflection_l290_290106

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l290_290106


namespace largest_4_digit_congruent_17_mod_24_l290_290158

theorem largest_4_digit_congruent_17_mod_24 :
  ∃ (k : ℕ), (k < 10000) ∧ (1000 ≤ k) ∧ (k % 24 = 17) ∧ (∀ (m : ℕ), (m < 10000) ∧ (1000 ≤ m) ∧ (m % 24 = 17) → m ≤ k) :=
begin
  sorry
end

end largest_4_digit_congruent_17_mod_24_l290_290158


namespace wrench_force_l290_290519

theorem wrench_force (F L k: ℝ) (h_inv: ∀ F L, F * L = k) (h_given: F * 12 = 240 * 12) : 
  (∀ L, (L = 16) → (F = 180)) ∧ (∀ L, (L = 8) → (F = 360)) := by 
sorry

end wrench_force_l290_290519


namespace three_digit_integer_conditions_l290_290371

theorem three_digit_integer_conditions:
  ∃ n : ℕ, 
    n % 5 = 3 ∧ 
    n % 7 = 4 ∧ 
    n % 4 = 2 ∧
    100 ≤ n ∧ n < 1000 ∧ 
    n = 548 :=
sorry

end three_digit_integer_conditions_l290_290371


namespace parabola_y_intercepts_zero_l290_290367

theorem parabola_y_intercepts_zero : ∀ (y : ℝ), 3 * y^2 - 4 * y + 8 ≠ 0 → (∃ x : ℝ, x = 3 * y^2 - 4 * y + 8) → false :=
by
  intro y h h'
  unfold gcd
  exfalso
  sorry

end parabola_y_intercepts_zero_l290_290367


namespace parabola_count_l290_290315

theorem parabola_count :
  let vertex_x_axis := true in
  let directrix_y_axis := true in
  let point_A := (4, 0) in
  let distance_A_to_parabola := 2 in
  ∃ p : ℝ, p > 0 ∧ (∃ a : ℝ, ∃ parabola_count : ℕ, parabola_count = 3) :=
sorry

end parabola_count_l290_290315


namespace number_of_points_on_ellipse_l290_290529

noncomputable def line_eq (x y : ℝ) : Prop :=
  (x / 4) + (y / 3) = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 9) = 1

noncomputable def intersection_points (x y : ℝ) : Prop :=
  line_eq x y ∧ ellipse_eq x y

noncomputable def area_of_triangle (A B P : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := P in
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

theorem number_of_points_on_ellipse :
  let A := (4, 0): ℝ × ℝ -- Placeholder values
  let B := (0, 3): ℝ × ℝ -- Placeholder values
  ∃ (P : ℝ × ℝ), ellipse_eq P.1 P.2 ∧ area_of_triangle A B P = 3 :=
begin
  sorry
end

end number_of_points_on_ellipse_l290_290529


namespace shaded_perimeter_l290_290418

/-- O is the center of a circle with radii OR and OS, both of length 7.
    Arc RS forms 5/8 of the circle's circumference. Prove that the perimeter
    of the shaded region is 14 + 35/4 * π. -/
theorem shaded_perimeter (O R S : Point) (radius : ℕ) 
  (hO : Center O (Circle radius))
  (hR : Radius O R radius)
  (hS : Radius O S radius)
  (arc_ratio : Real := 5/8) :
  radius = 7 ∧ arc_ratio = 5/8 →
  Perimeter O R S = 14 + 35/4 * Real.pi :=
begin
  sorry
end

end shaded_perimeter_l290_290418


namespace pure_imaginary_complex_l290_290333

theorem pure_imaginary_complex (m : ℝ) (i : ℂ) (h : i^2 = -1) :
    (∃ (y : ℂ), (2 - m * i) / (1 + i) = y * i) ↔ m = 2 :=
by
  sorry

end pure_imaginary_complex_l290_290333


namespace octal_sum_is_correct_l290_290766

-- Definitions for the octal to decimal conversions
def octalToDecimal (n : List Nat) : Nat :=
  List.foldr (λ (d acc : Nat), 8 * acc + d) 0 n

-- Conditions
def n1 := [4, 4, 4] -- 444_8
def n2 := [4, 4]    -- 44_8
def n3 := [4]       -- 4_8

-- Conversion of the sum back to octal
def decimalToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc else 
        let (q, r) := n / 8, n % 8
        aux q (r :: acc)
    aux n []

-- Proof statement
theorem octal_sum_is_correct : 
  decimalToOctal ((octalToDecimal n1) + (octalToDecimal n2) + (octalToDecimal n3)) = [5, 1, 4] := 
by
  -- Proof would go here
  sorry

end octal_sum_is_correct_l290_290766


namespace smallest_m_for_R_eq_l_l290_290479

-- Definitions and Hypotheses
def line_angle1 : ℝ := Real.pi / 70
def line_angle2 : ℝ := Real.pi / 54
def slope_l : ℝ := 19 / 92
def angle_l : ℝ := Real.atan slope_l
def R (theta : ℝ) : ℝ := theta + 8 * Real.pi / 945

-- Main theorem statement
theorem smallest_m_for_R_eq_l :
  ∃ m : ℕ, m > 0 ∧ (R^[m] angle_l = angle_l) := 
begin
  use 945,
  sorry
end

end smallest_m_for_R_eq_l_l290_290479


namespace product_of_distances_l290_290317

def point_on_hyperbola (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def distance_to_asymptotes_product (x y : ℝ) 
  (hx : point_on_hyperbola x y) : ℝ :=
  ((abs (sqrt(3) * x - y)) / 2) * ((abs (sqrt(3) * x + y)) / 2)

theorem product_of_distances (x y : ℝ) (hx : point_on_hyperbola x y) :
  distance_to_asymptotes_product x y hx = 3 :=
sorry

end product_of_distances_l290_290317


namespace quadratic_rewrite_b_value_l290_290721

theorem quadratic_rewrite_b_value (b n : ℝ)
  (h₁ : ∃ n, (x : ℝ) -> (x + n)^2 + 1/4 = x^2 + b * x + 2/3)
  (h₂ : b < 0)
  (h₃ : n^2 + 1/4 = 2/3) :
  b = - real.sqrt 15 / 3 :=
by
  sorry

end quadratic_rewrite_b_value_l290_290721


namespace distinct_gcd_numbers_l290_290550

theorem distinct_gcd_numbers (nums : Fin 100 → ℕ) (h_distinct : Function.Injective nums) :
  ¬ ∃ a b c : Fin 100, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (nums a + Nat.gcd (nums b) (nums c) = nums b + Nat.gcd (nums a) (nums c)) ∧ 
    (nums b + Nat.gcd (nums a) (nums c) = nums c + Nat.gcd (nums a) (nums b)) := 
sorry

end distinct_gcd_numbers_l290_290550


namespace A_n_eq_B_n_if_even_A_n_gt_B_n_if_odd_l290_290443

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def prime_factors_count (n : ℕ) : ℕ :=
  (Nat.factorization n).support.card

def A_n (n : ℕ) : Finset ℕ :=
  {k ∈ Finset.range n | prime_factors_count (Nat.gcd n k) % 2 = 0}

def B_n (n : ℕ) : Finset ℕ :=
  {k ∈ Finset.range n | prime_factors_count (Nat.gcd n k) % 2 = 1}

theorem A_n_eq_B_n_if_even (n : ℕ) (h : is_even n) : A_n n.card = B_n n.card :=
sorry

theorem A_n_gt_B_n_if_odd (n : ℕ) (h : is_odd n) : A_n n.card > B_n n.card :=
sorry

end A_n_eq_B_n_if_even_A_n_gt_B_n_if_odd_l290_290443


namespace Dennis_floor_proof_l290_290730

def floor_of_Dennis : ℕ :=
  let charlie_floor : ℕ := 1 / 4 * 16
  let dennis_floor := charlie_floor + 2
  dennis_floor
  -- The correct floor on which Dennis lives is 6

theorem Dennis_floor_proof :
  (∀ (F C D : ℕ), (C = F / 4) ∧ (D = C + 2) ∧ (F = 16) → D = 6) :=
by
  intros F C D h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw h3 at h1,
  rw h1,
  rw h2,
  norm_num,
  sorry

end Dennis_floor_proof_l290_290730


namespace Dave_guitar_strings_replacement_l290_290728

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l290_290728


namespace major_premise_is_wrong_l290_290644

theorem major_premise_is_wrong : ¬ (∀ x : ℤ, x ∈ ℕ) :=
by
  sorry

end major_premise_is_wrong_l290_290644


namespace P_at_7_eq_69120_l290_290069

theorem P_at_7_eq_69120 (a b c d e f : ℝ) :
  (∃ Q R : Polynomial ℂ, 
    Q = 3*Polynomial.C 1 * Polynomial.X^4 - 39*Polynomial.C 1 * Polynomial.X^3 + a*Polynomial.C 1 * Polynomial.X^2 + b*Polynomial.C 1 * Polynomial.X + c*Polynomial.C 1 ∧
    R = 4*Polynomial.C 1 * Polynomial.X^4 - 64*Polynomial.C 1 * Polynomial.X^3 + d*Polynomial.C 1 * Polynomial.X^2 + e*Polynomial.C 1 * Polynomial.X + f*Polynomial.C 1 ∧
    Polynomial.roots (Q * R) = {1, 2, 3, 4, 6}) →
  (let P := 12 * (Polynomial.X - 1) * (Polynomial.X - 2) * (Polynomial.X - 3)^2 * (Polynomial.X - 4) * (Polynomial.X - 5)^2 * (Polynomial.X - 6) in
   P.eval 7 = 69120) :=
by
  intros
  let Q := 3*Polynomial.C 1 * Polynomial.X^4 - 39*Polynomial.C 1 * Polynomial.X^3 + a*Polynomial.C 1 * Polynomial.X^2 + b*Polynomial.C 1 * Polynomial.X + c*Polynomial.C 1
  let R := 4*Polynomial.C 1 * Polynomial.X^4 - 64*Polynomial.C 1 * Polynomial.X^3 + d*Polynomial.C 1 * Polynomial.X^2 + e*Polynomial.C 1 * Polynomial.X + f*Polynomial.C 1
  have h1 : Polynomial.roots (Q * R) = {1, 2, 3, 4, 6} := sorry
  have h2 : P = 12 * (Polynomial.X - 1) * (Polynomial.X - 2) * (Polynomial.X - 3)^2 * (Polynomial.X - 4) * (Polynomial.X - 5)^2 * (Polynomial.X - 6) := sorry
  have h3 : P.eval 7 = 12 * 6 * 5 * 16 * 4 * 3 := sorry
  exact h3

end P_at_7_eq_69120_l290_290069


namespace find_k_l290_290015

theorem find_k (k : ℝ) : 
  (∃ (line_eq : ℝ → ℝ), (line_eq = λ x, k * x + 2) ∧ 
  (1/2 * |(-2/k)| * 2 = 6)) → 
  (k = 1/3 ∨ k = -1/3) :=
by 
  sorry

end find_k_l290_290015


namespace royal_children_count_l290_290612

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290612


namespace projections_concyclic_l290_290906

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def A' : Point := orthogonal_projection A (line_through B D)
noncomputable def C' : Point := orthogonal_projection C (line_through B D)
noncomputable def B' : Point := orthogonal_projection B (line_through A C)
noncomputable def D' : Point := orthogonal_projection D (line_through A C)

axiom concyclic_points (a b c d : Point) : circle_exists_containing a b c d

theorem projections_concyclic :
  concyclic_points A B C D →
  concyclic_points A' B' C' D' :=
by
  intros concyclic
  sorry

end projections_concyclic_l290_290906


namespace tom_swim_time_l290_290145

theorem tom_swim_time (t : ℝ) :
  (2 * t + 4 * t = 12) → t = 2 :=
by
  intro h
  have eq1 : 6 * t = 12 := by linarith
  linarith

end tom_swim_time_l290_290145


namespace calc_expression1_calc_expression2_l290_290713

-- Prove the first calculation
theorem calc_expression1 :
  |1 + log 10 0.001| + sqrt ((log 10 (1 / 3))^2 - 4 * log 10 3 + 4) + log 10 6 - log 10 0.02 = 7 :=
sorry

-- Prove the second calculation
theorem calc_expression2 :
  ((-27/8)^(-2/3)) + 0.002^(-1/2) - 10 * ((sqrt 5 - 2)^(-1)) + (2 - sqrt 3)^0 = 4/9 + 10 * sqrt 2 - 10 * sqrt 5 - 19 :=
sorry

end calc_expression1_calc_expression2_l290_290713


namespace royal_children_count_l290_290610

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290610


namespace train_length_correct_l290_290683

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l290_290683


namespace five_digit_palindromes_count_l290_290740

theorem five_digit_palindromes_count :
  (∃ (A B C : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  ∃ (count : ℕ), count = 900 :=
by {
  intro h,
  use 900,
  sorry        -- Proof is omitted
}

end five_digit_palindromes_count_l290_290740


namespace units_digit_of_n_cubed_minus_n_squared_l290_290468

-- Define n for the purpose of the problem
def n : ℕ := 9867

-- Prove that the units digit of n^3 - n^2 is 4
theorem units_digit_of_n_cubed_minus_n_squared : ∃ d : ℕ, d = (n^3 - n^2) % 10 ∧ d = 4 := by
  sorry

end units_digit_of_n_cubed_minus_n_squared_l290_290468


namespace negation_of_prop1_l290_290575

-- Define the statements involved in B
variable α : ℝ
variable π : ℝ -- Assuming π is the constant representing pi

def prop1 := (α = π / 6) → (Real.sin α = 1 / 2)
def neg_prop1 := (α ≠ π / 6) → (Real.sin α ≠ 1 / 2)

-- Our goal is to prove that the negation of prop1 is neg_prop1
theorem negation_of_prop1 : ¬prop1 = neg_prop1 :=
sorry

end negation_of_prop1_l290_290575


namespace total_eggs_l290_290137

def e0 : ℝ := 47.0
def ei : ℝ := 5.0

theorem total_eggs : e0 + ei = 52.0 := by
  sorry

end total_eggs_l290_290137


namespace problem_I_problem_II_l290_290415

noncomputable def curve_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem problem_I : 
  ∀ (x y : ℝ), 
  (∃ (M N : ℝ), 
    M * M + N * N = 8 ∧ 
    (∃ (P : ℝ × ℝ), 
      P = (x, y) ∧ 
      ((P.2 - (⟨⌊M/√3⌋div 3 ∂⟩)) ∧ (P.2 - (⟨⌊N/√3⌋div 3 ∂⟩))
    ⟹ curve_equation x y)) sorry

theorem problem_II : 
  ∀ (A B C D : ℝ×ℝ) (Q : ℝ×ℝ),
  Q = (1, 1) ∧ 
  (curve_equation A.1 A.2) ∧ 
  (curve_equation B.1 B.2) ∧ 
  (curve_equation C.1 C.2) ∧ 
  (curve_equation D.1 D.2) ∧ 
  (parallel (A,B) (C,D)) ∧ 
  (∃ (λ : ℝ), A - Q = λ * (C - Q) ∧ B - Q = λ * (D - Q)) 
  ⟹ slope (A,B) = -3/4 := sorry

end problem_I_problem_II_l290_290415


namespace part_i_part_ii_l290_290442

-- Part (i)
variable (n : ℕ)
def M : Finset ℕ := Finset.range n

def phi_bij (φ : M → M) : Prop :=
  Function.Bijective φ

theorem part_i (φ : M → M) (hφ : phi_bij φ) :
  ∃ (φ1 φ2 : M → M), (Function.Bijective φ1) ∧ (Function.Bijective φ2) ∧
    (∀ x, φ (φ1 (φ2 x)) = x) ∧ (∀ x, φ1 (φ1 x) = x) ∧ (∀ x, φ2 (φ2 x) = x) := sorry

-- Part (ii)
variable (M_inf : Type) [Inhabited M_inf] [Infinite M_inf]
def phi_inf_bij (φ : M_inf → M_inf) : Prop :=
  Function.Bijective φ

theorem part_ii (φ : M_inf → M_inf) (hφ : phi_inf_bij φ) :
  ∃ (φ1 φ2 : M_inf → M_inf), (Function.Bijective φ1) ∧ (Function.Bijective φ2) ∧
    (∀ x, φ (φ1 (φ2 x)) = x) ∧ (∀ x, φ1 (φ1 x) = x) ∧ (∀ x, φ2 (φ2 x) = x) := sorry

end part_i_part_ii_l290_290442


namespace solution_set_f_x_plus_1_gt_0_l290_290522

noncomputable def f : ℝ → ℝ := sorry  -- Define the function f with the conditions provided

theorem solution_set_f_x_plus_1_gt_0 :
  (∀ x, f (2 - x) = f (x + 2)) →  -- Symmetry about the line x = 1
  (∀ x y, x ≤ y → y ≥ 1 → f y ≤ f x) →  -- Monotonic decreasing on [1, +∞)
  f 0 = 0 →  -- f(0) = 0
  {x : ℝ | f (x + 1) > 0} = set.Ioo (-1) 1 :=  -- The solution set of f(x + 1) > 0 is (-1,1)
by
  intros sym mon f0
  sorry  -- Proof omitted

end solution_set_f_x_plus_1_gt_0_l290_290522


namespace reflected_circle_center_l290_290105

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l290_290105


namespace total_mile_times_l290_290147

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l290_290147


namespace angle_B_pi_div_3_triangle_perimeter_l290_290034

-- Problem 1: Prove that B = π / 3 given the condition.
theorem angle_B_pi_div_3 (A B C : ℝ) (hTriangle : A + B + C = Real.pi) 
  (hCos : Real.cos B = Real.cos ((A + C) / 2)) : 
  B = Real.pi / 3 :=
sorry

-- Problem 2: Prove the perimeter given the conditions.
theorem triangle_perimeter (a b c : ℝ) (m : ℝ) 
  (altitude : ℝ) 
  (hSides : 8 * a = 3 * c) 
  (hAltitude : altitude = 12 * Real.sqrt 3 / 7) 
  (hAngleB : ∃ B, B = Real.pi / 3) :
  a + b + c = 18 := 
sorry

end angle_B_pi_div_3_triangle_perimeter_l290_290034


namespace length_of_EF_l290_290414

/-- A proof that, given a rectangle ABCD with AB = 4 and BC = 8, folding the rectangle such that AD and BC overlap 
with C coinciding with point A results in a pentagon ABEFD where the length of segment EF is 4. -/
theorem length_of_EF {A B C D E F : Point} :  
  distance A B = 4 → distance B C = 8 → 
  C = A → 
  F = D →
  B.x = 0 → B.y = 4 →
  D.x = 8 → D.y = 0 →
  E = B → 
  distance E F = 4 := 
sorry

end length_of_EF_l290_290414


namespace ganesh_ram_sohan_work_time_l290_290307

theorem ganesh_ram_sohan_work_time (G R S : ℝ)
  (H1 : G + R = 1 / 24)
  (H2 : S = 1 / 48) : (G + R + S = 1 / 16) ∧ (1 / (G + R + S) = 16) :=
by
  sorry

end ganesh_ram_sohan_work_time_l290_290307


namespace decompose_vector_l290_290170

-- Definitions of the given vectors
def x : ℝ^3 := ![-1, 7, 0]
def p : ℝ^3 := ![0, 3, 1]
def q : ℝ^3 := ![1, -1, 2]
def r : ℝ^3 := ![2, -1, 0]

-- The theorem we want to prove
theorem decompose_vector :
  ∃ (α β γ : ℝ), x = α • p + β • q + γ • r ∧ α = 2 ∧ β = -1 ∧ γ = 0 :=
by {
  use [2, -1, 0],
  simp [x, p, q, r],
  sorry
}

end decompose_vector_l290_290170


namespace royal_family_children_l290_290635

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290635


namespace james_budget_spent_on_food_l290_290867

theorem james_budget_spent_on_food :
  ∀ (budget : ℕ) (spend_accom_perc spend_enter_perc : ℝ) (spend_materials : ℕ),
  budget = 1000 ∧ spend_accom_perc = 0.15 ∧ spend_enter_perc = 0.25 ∧ spend_materials = 300 
  → ((budget - (spend_accom_perc * budget).toNat - (spend_enter_perc * budget).toNat - spend_materials) / (budget : ℝ) * 100) = 30 :=
by
intros budget spend_accom_perc spend_enter_perc spend_materials
intro h
cases' h with hb h1
cases' h1 with ha h2
cases' h2 with he hm
rw [hb, ha, he, hm]
norm_num
sorry

end james_budget_spent_on_food_l290_290867


namespace different_values_expression_l290_290369

theorem different_values_expression : 
  let seq := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_list seq →
  (2:ℚ ≠ 0) →
  (3:ℚ ≠ 0) →
  seq.head = 2 →
  seq.tail.head = 3 →
  (256 = 2 ^ ( ( seq.length - 4 ) / 2+1)) :=
by
  assume seq
  assume Hprime Hneq2 Hneq3 Hhead Htail
  sorry

end different_values_expression_l290_290369


namespace car_travel_distance_l290_290653

noncomputable def velocity (t : ℝ) : ℝ := 7 - 3 * t + 25 / (1 + t)

noncomputable def stopping_time : ℝ :=
  let solve_quadratic := λ (a b c : ℝ), -- Solves ax^2 + bx + c = 0 for x
    let disc := b ^ 2 - 4 * a * c in
    if disc >= 0 then [(-b + Real.sqrt disc) / (2 * a), (-b - Real.sqrt disc) / (2 * a)]
    else [] in
  (solve_quadratic 3 (-4) (-32)).find (λ t, t > 0).getOrElse 0

theorem car_travel_distance :
  ∫ t in 0..stopping_time, velocity t = 4 + 25 * Real.log 5 :=
by
  unfold velocity stopping_time
  sorry

end car_travel_distance_l290_290653


namespace john_billed_for_28_minutes_l290_290300

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end john_billed_for_28_minutes_l290_290300


namespace true_discount_correct_l290_290588

noncomputable def true_discount (banker_gain : ℝ) (average_rate : ℝ) (time_years : ℝ) : ℝ :=
  let r := average_rate
  let t := time_years
  let exp_factor := Real.exp (-r * t)
  let face_value := banker_gain / (1 - exp_factor)
  face_value - (face_value * exp_factor)

theorem true_discount_correct : 
  true_discount 15.8 0.145 5 = 15.8 := 
by
  sorry

end true_discount_correct_l290_290588


namespace time_to_complete_together_l290_290295

-- Definitions for the given conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Theorem statement for the mathematically equivalent proof problem
theorem time_to_complete_together (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
   (1 : ℝ) / ((1 / x) + (1 / y)) = x * y / (x + y) :=
sorry

end time_to_complete_together_l290_290295


namespace infinite_perpendicular_lines_infinite_parallel_lines_l290_290029

-- Definitions for clarity
def perpendicular_lines_through_point (P : Point) (L : Line) : Set Line := 
  { l : Line | l.contains P ∧ l.perpendicular_to L }

def parallel_lines_through_point_outside_plane (P : Point) (Π : Plane) : Set Line := 
  { l : Line | l.contains P ∧ l.parallel_to Π }

-- Proof problem
theorem infinite_perpendicular_lines (P : Point) (L : Line) : 
  set.infinite (perpendicular_lines_through_point P L) := 
sorry

theorem infinite_parallel_lines (P : Point) (Π : Plane) : 
  set.infinite (parallel_lines_through_point_outside_plane P Π) := 
sorry

end infinite_perpendicular_lines_infinite_parallel_lines_l290_290029


namespace shift_length_is_eight_l290_290216

noncomputable def shift_length (h : ℕ) : Prop :=
  (20 * 12 * h + 20 * 100 + 1000) + 9080 = 14000

theorem shift_length_is_eight : shift_length 8 :=
by
  unfold shift_length
  rw [mul_add, add_assoc, ←add_assoc 2000 1000]
  norm_num
  sorry

end shift_length_is_eight_l290_290216


namespace divisibility_problem_l290_290904

theorem divisibility_problem (q : ℕ) (hq : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬((q + 2)^(q - 3) + 1) % (q - 4) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % q = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 6) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 3) = 0 := sorry

end divisibility_problem_l290_290904


namespace investment_relationship_l290_290928

theorem investment_relationship :
  ∀ (A B C : ℝ),
  let A1 := A * 1.10 in
  let B1 := B * 0.80 in
  let C1 := C * 1.05 in
  let final_A := A1 * 0.95 in
  let final_B := B1 * 1.15 in
  let final_C := C1 * 0.90 in
  A = 150 → B = 100 → C = 200 →
  final_B < final_A ∧ final_A < final_C :=
begin
  intros A B C,
  rw [A1, B1, C1, final_A, final_B, final_C],
  assume hA : A = 150,
  assume hB : B = 100,
  assume hC : C = 200,
  sorry
end

end investment_relationship_l290_290928


namespace train_length_correct_l290_290682

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l290_290682


namespace other_root_of_quadratic_eq_l290_290009

theorem other_root_of_quadratic_eq (b : ℝ) (h : (1 : ℝ) ^ 2 + b * 1 + 2 = 0) :
  ∃ x2 : ℝ, (1 * x2 = 2) ∧ (x2 = 2) := by
  use 2
  split
  · exact Eq.symm (by norm_num)
  · rfl

end other_root_of_quadratic_eq_l290_290009


namespace upstream_travel_time_l290_290199

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l290_290199


namespace simplify_expression_l290_290000

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) : (a - 2) * (b - 2) = -2 * m := 
by
  sorry

end simplify_expression_l290_290000


namespace find_a3_l290_290978

-- Define the conditions of the arithmetic sequence and the sum of the first n terms
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variable (n : ℕ)

-- Define the common difference and the sum of the first three terms
def common_difference := 2
def sum_first_three_terms := S 3 = 12

-- State the equation for the sum of the first n terms of an arithmetic sequence
def sum_formula (a_1 d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given the conditions, prove that a₃ = 6
theorem find_a3 (h1 : ∀ n, S n = sum_formula a 2 n) (h2 : sum_first_three_terms) : a 3 = 6 :=
sorry -- proof is not needed

end find_a3_l290_290978


namespace royal_family_children_l290_290640

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290640


namespace _l290_290548
noncomputable theorem inverse_variation (x y : ℝ) (k : ℝ) (h1 : y = k / x^2) (h2 : y = 2) (h3 : x = 3) : x = 3/2 :=
by
-- Given that y varies inversely as x^2, k is the constant of proportionality.
  let k := y * x^2
  have k_def : k = 18 := by
-- The value k is determined by the initial condition x = 3, y = 2
    calc k = y * x^2   : by intros
       ... = 2 * 3^2  : by { rw [h2, h3], unfold pow }
       ... = 18       : by norm_num
  have hx : x^2 = k / y := by
-- From the inverse variation relationship, we substitute y = 8
    calc x^2 = k / y   : by unfold pow
       ... = 18 / 8   : by {rw [k_def, h1], norm_num}
       ... = 9/4      : by norm_num
  calc x = sqrt (x^2) : by rw [sqrt_eq_iff_sq_eq, hx]
     ... = sqrt (9/4) : by rw [hx]
     ... = 3/2        : by { rw sqrt_div, norm_num }

end _l290_290548


namespace number_of_factors_of_n_l290_290455

noncomputable def n : ℕ := 2^4 * 3^5 * 5^6 * 7^7

theorem number_of_factors_of_n : ∃ k : ℕ, k = 1680 ∧ (∀ d : ℕ, d ∣ n ↔ d ∈ finset.Icc (1 : ℕ) n ∧ d.factorization ⊆ n.factorization) :=
by
  existsi 1680
  split
  case left =>
    sorry
  case right =>
    sorry

end number_of_factors_of_n_l290_290455


namespace log_equation_l290_290246

theorem log_equation : 
    (log 10 2 + log 10 5 + 2 * log 5 10 - log 5 20) = 2 := 
by 
  sorry

end log_equation_l290_290246


namespace smallest_consecutive_even_sum_140_l290_290176

theorem smallest_consecutive_even_sum_140 :
  ∃ (x : ℕ), (x % 2 = 0) ∧ (x + (x + 2) + (x + 4) + (x + 6) = 140) ∧ (x = 32) :=
by
  sorry

end smallest_consecutive_even_sum_140_l290_290176


namespace emily_eggs_collection_l290_290258

-- Define the initial conditions and arithmetic sequence parameters
def a1 : ℕ := 25
def d : ℕ := 5
def n : ℕ := 12

-- Define the problem statement
theorem emily_eggs_collection : ∑ k in (finset.range n), (a1 + k * d) = 630 :=
by sorry

end emily_eggs_collection_l290_290258


namespace julias_total_spending_l290_290875

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l290_290875


namespace simplify_expression_l290_290508

variable (x y : ℝ)

theorem simplify_expression :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 :=
by
  sorry

end simplify_expression_l290_290508


namespace line_circle_intersection_a_value_l290_290354

theorem line_circle_intersection_a_value (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + A.2 = a ∧ A.1^2 + A.2^2 = 2) ∧ 
    (B.1 + B.2 = a ∧ B.1^2 + B.2^2 = 2) ∧
    let OA := (A.1, A.2), OB := (B.1, B.2) in 
    ∥2 * OA.1 - 3 * OB.1, 2 * OA.2 - 3 * OB.2∥ = ∥2 * OA.1 + 3 * OB.1, 2 * OA.2 + 3 * OB.2∥) 
    ↔ (a = sqrt 2 ∨ a = -sqrt 2) :=
sorry

end line_circle_intersection_a_value_l290_290354


namespace num_factors_of_n_l290_290458

theorem num_factors_of_n (n : ℕ) (h : n = 2^4 * 3^5 * 5^6 * 7^7) : 
  ∃ k, k = 1680 ∧ (number_of_factors n) = k := 
sorry

end num_factors_of_n_l290_290458


namespace area_ratio_5_l290_290050

variables {D E F Q : Type*}
variables [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F] [EuclideanGeometry Q]
variables (d e f q : ℝ^3)
variables (QD QE QF : ℝ^3)

def vector_eq (QD QE QF : ℝ^3) : Prop :=
  QD + 3 * QE + 4 * QF = 0

noncomputable def ratio_of_areas (A B C P : Type*) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry P]
  (area_DEF area_DQF : ℝ) : Prop :=
  area_DEF / area_DQF = 5

axiom area_DEF (DEF : Triangle D E F) : ℝ
axiom area_DQF (DQF : Triangle D Q F) : ℝ

theorem area_ratio_5 (DEF : Triangle D E F) (DQF : Triangle D Q F) 
  (h_vector_eq : vector_eq QD QE QF) : ratio_of_areas D E F Q (area_DEF DEF) (area_DQF DQF) :=
sorry

end area_ratio_5_l290_290050


namespace n_gon_diagonal_regions_l290_290406

theorem n_gon_diagonal_regions (n : ℕ) (h : n ≥ 4) :
  let D := n * (n - 3) / 2,
      P := n * (n - 1) * (n - 2) * (n - 3) / 24,
      R := D + P + 1 in
  R = n * (n - 3) / 2 + n * (n - 1) * (n - 2) * (n - 3) / 24 + 1 :=
by
  sorry

end n_gon_diagonal_regions_l290_290406


namespace periodic_difference_not_necessarily_periodic_l290_290207

theorem periodic_difference_not_necessarily_periodic (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 6) = g x) 
  (hh : ∀ x, h (x + 2 * Real.pi) = h x) : 
  ¬ ∃ p > 0, ∀ x, (g - h) (x + p) = (g - h) x :=
begin
  sorry,
end

end periodic_difference_not_necessarily_periodic_l290_290207


namespace deriv_function1_deriv_function2_deriv_function3_l290_290269

noncomputable def function1 (x : ℝ) : ℝ := (1 + Real.cos x) / (1 - Real.cos x)
noncomputable def function2 (x : ℝ) : ℝ := Real.sin x - Real.cos x
noncomputable def function3 (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

theorem deriv_function1 : deriv function1 = λ x, (-2 * Real.sin x) / (1 - Real.cos x)^2 := sorry

theorem deriv_function2 : deriv function2 = λ x, Real.cos x + Real.sin x := sorry

theorem deriv_function3 : deriv function3 = λ x, 3*x^2 + 6*x := sorry

end deriv_function1_deriv_function2_deriv_function3_l290_290269


namespace lines_intersect_on_circumcircle_l290_290447

theorem lines_intersect_on_circumcircle 
  (A B C D E G H L M : Point) 
  (Γ : Circle) 
  (on_circumcircle : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ) 
  (circle_ac : Circle) 
  (passes_through_ac : A ∈ circle_ac ∧ C ∈ circle_ac) 
  (D_on_bc : D ∈ BC) 
  (E_on_ba : E ∈ BA) 
  (G_on_ad : G ∈ AD ∧ G ∈ Γ) 
  (H_on_ce : H ∈ CE ∧ H ∈ Γ) 
  (L_on_tangent_A : tangent_to (Γ, A, L) ∧ L ∈ DE) 
  (M_on_tangent_C : tangent_to (Γ, C, M) ∧ M ∈ DE)
  (intersects_at_Γ : LH ∩ MG ∈ Γ) : 
  LH ∩ MG ∈ Γ := 
  sorry

end lines_intersect_on_circumcircle_l290_290447


namespace solve_equation_l290_290649

theorem solve_equation : (x : ℚ) → 
  ((x = 0) ∨ (x = -5/2)) ↔ 
  (frac ((x + 1)^2 + 1) (x + 1) + frac ((x + 4)^2 + 4) (x + 4) = frac ((x + 2)^2 + 2) (x + 2) + frac ((x + 3)^2 + 3) (x + 3)) :=
by
  sorry

end solve_equation_l290_290649


namespace find_jack_money_l290_290259

noncomputable def Jack_money : ℕ :=
  let J := 26 in
  J

theorem find_jack_money (J : ℕ) 
  (Ben_has : ℕ := J - 9)
  (Eric_has : ℕ := J - 19)
  (total : J + Ben_has + Eric_has = 50) : 
  J = 26 :=
by
  have h1 : Ben_has = J - 9 := rfl
  have h2 : Eric_has = J - 19 := rfl
  rw [h1, h2] at total
  linarith

end find_jack_money_l290_290259


namespace compute_z_to_the_sixth_l290_290439

noncomputable def z : ℂ := (sqrt 3 + complex.i) / 2

theorem compute_z_to_the_sixth :
  z^6 = (1 + sqrt 3) / 4 - (sqrt 3 + 1) / 8 * complex.i :=
by
  sorry

end compute_z_to_the_sixth_l290_290439


namespace probability_digit_9_in_3_over_11_is_zero_l290_290490

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end probability_digit_9_in_3_over_11_is_zero_l290_290490


namespace congruence_of_prime_squared_sum_l290_290469

theorem congruence_of_prime_squared_sum (p a b : ℕ) (hp : nat.prime p)
  (h_eq : p = a^2 + b^2) :
  ∃ x ∈ {a, -a, b, -b}, x ≡ 1/2 * nat.choose ((p-1)/2) ((p-1)/4) [MOD p] :=
by sorry

end congruence_of_prime_squared_sum_l290_290469


namespace find_unit_prices_calculate_cost_range_l290_290239

-- Definitions based on provided conditions
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 5 * y = 210 ∧ 4 * x + 10 * y = 380

def cost_for_12 (a : ℕ) : ℕ :=
  -a^2 + 10 * a + 240

-- Statements to prove based on the problem
theorem find_unit_prices : ∃ x y, unit_prices x y ∧ x = 20 ∧ y = 30 :=
begin
  -- solution steps to solve the theorem should be added here
  sorry
end

theorem calculate_cost_range : (229 ≤ cost_for_12 11) ∧ (cost_for_12 5 ≤ 265) :=
begin
  -- solution steps to solve the theorem should be added here
  sorry
end

end find_unit_prices_calculate_cost_range_l290_290239


namespace coefficient_x2y_in_expansion_l290_290954

open BigOperators

theorem coefficient_x2y_in_expansion :
  ∃ (c : ℚ), (∀ (x y : ℚ), (x - 2 * y + 1)^5 = c * x^2 * y + ...) ∧ c = -60 :=
by
  sorry

end coefficient_x2y_in_expansion_l290_290954


namespace number_of_children_l290_290631

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290631


namespace parabola_focus_vertex_ratio_l290_290893

open Real

-- Definitions of parabolas, their vertices, and foci
def parabola₁ := {p : ℝ × ℝ | p.2 = 4 * p.1^2}
def V₁ := (0 : ℝ, 0 : ℝ)
def F₁ := (0 : ℝ, 1 / 16 : ℝ)

-- Angle condition imposed on points A and B on parabola₁
def A (a : ℝ) := (a, 4 * a^2)
def B (b : ℝ) := (b, 4 * b^2)
def orthogonal_condition (a b : ℝ) : Prop := a * b = -1 / 4

-- Midpoint definition and resultant parabola
def midpoint (a b : ℝ) := ((a + b) / 2, (a + b)^2 + 1 / 2)
def parabola₂ := {p : ℝ × ℝ | p.2 = p.1^2 / 2 + 1 / 2}
def V₂ := (0 : ℝ, 1 / 2 : ℝ)
def F₂ := (0 : ℝ, 5 / 8 : ℝ)

-- Distance formulas
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  (sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))

noncomputable def ratio_of_distances : ℝ :=
  (distance F₁ F₂) / (distance V₁ V₂)

theorem parabola_focus_vertex_ratio : 
  ∀ (a b : ℝ), orthogonal_condition a b -> 
  ratio_of_distances = 9 / 8 :=
by
  sorry

end parabola_focus_vertex_ratio_l290_290893


namespace total_fat_l290_290662

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end total_fat_l290_290662


namespace hyperbola_focus_k_value_l290_290118

theorem hyperbola_focus_k_value :
  (∃ k : ℝ, 8 * k * x^2 - k * y^2 = 8 ∧ (0, -3) is_a_focus k ∧ k = -1) := sorry

end hyperbola_focus_k_value_l290_290118


namespace john_last_segment_speed_l290_290871

/-- John drove 150 miles in 120 minutes. 
  His average speed during the first 40 minutes was 50 mph, 
  and his average speed during the second 40 minutes was 55 mph. 
  Prove that his average speed during the last 40 minutes was 120 mph.
-/
theorem john_last_segment_speed :
  ∃ x : ℚ, 
    let total_distance := 150 in
    let total_time := 2 in -- Total time in hours
    let distance1 := 50 * (2 / 3) in -- Distance covered in the first 40 minutes
    let distance2 := 55 * (2 / 3) in -- Distance covered in the second 40 minutes
    let remaining_distance := total_distance - (distance1 + distance2) in
    let remaining_time := 2 / 3 in -- Remaining time in hours
    x = remaining_distance / remaining_time → x = 120 :=
begin
  sorry
end

end john_last_segment_speed_l290_290871


namespace find_value_of_fraction_l290_290060

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l290_290060


namespace number_of_children_l290_290629

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290629


namespace num_quadrilaterals_with_parallel_sides_l290_290318

theorem num_quadrilaterals_with_parallel_sides (M : Type) [regular_polygon M 16] :
  (count_quadrilaterals_with_parallel_sides M) = 364 := sorry

end num_quadrilaterals_with_parallel_sides_l290_290318


namespace range_of_a_l290_290395

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ :=
  if h : 0 < b ∧ b ≠ 1 ∧ 0 < x then log x / log b else 0

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 0 → log_base (2 * a) (x + 1) > 0) → 
  0 < a ∧ a < 1/2 := 
by 
  sorry

end range_of_a_l290_290395


namespace total_mile_times_l290_290149

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l290_290149


namespace royal_children_l290_290619

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290619


namespace triangle_PQD_angles_60_l290_290849

variables {A B C D P Q : Type*} [metric_space M] [AffineSpace M M]

def rhombus (A B C D : M) : Prop :=
  dist A B = dist B C ∧
  dist B C = dist C D ∧
  dist C D = dist D A

def angle_120 (A B C : M) : Prop :=
  angle A B C = 120

def AP_eq_BQ (A B C P Q : M) : Prop :=
  dist A P = dist B Q

theorem triangle_PQD_angles_60 (A B C D P Q : M) [rhombus A B C D] [angle_120 A B C] [AP_eq_BQ A B C P Q] :
  angle P Q D = 60 ∧ angle Q D P = 60 ∧ angle D P Q = 60 :=
sorry

end triangle_PQD_angles_60_l290_290849


namespace pyramid_volume_pyramid_surface_area_l290_290951

noncomputable def volume_of_pyramid (l : ℝ) := (l^3 * Real.sqrt 2) / 12

noncomputable def surface_area_of_pyramid (l : ℝ) := (l^2 * (2 + Real.sqrt 2)) / 2

theorem pyramid_volume (l : ℝ) :
  volume_of_pyramid l = (l^3 * Real.sqrt 2) / 12 :=
sorry

theorem pyramid_surface_area (l : ℝ) :
  surface_area_of_pyramid l = (l^2 * (2 + Real.sqrt 2)) / 2 :=
sorry

end pyramid_volume_pyramid_surface_area_l290_290951


namespace quadratic_roots_l290_290292

theorem quadratic_roots (z : ℂ) : (z^2 - 2*z = 4 - 3 * complex.I) ↔ 
  (z = 1 + real.sqrt 6 - (1 / 2) * real.sqrt 6 * complex.I) ∨ 
  (z = 1 - real.sqrt 6 + (1 / 2) * real.sqrt 6 * complex.I) :=
by 
  sorry

end quadratic_roots_l290_290292


namespace quadrilateral_perimeter_l290_290670

/-
A quadrilateral has vertices at (0, 0), (2, 5), (5, 5), and (6, 2).
Find its perimeter and express it in the form c * sqrt(p) + d * sqrt(q),
where c, d, p, and q are integers. Determine c + d = 4.
-/
noncomputable def perimeter_expression (c d p q : ℤ) : Prop :=
  ∃ (c d p q : ℤ), p = 29 ∧ q = 10 ∧ c = 1 ∧ d = 3 ∧ c + d = 4

theorem quadrilateral_perimeter :
  ∃ (c d p q : ℤ), perimeter_expression c d p q :=
  begin
    use [1, 3, 29, 10],
    split, refl,
    split, refl,
    split, refl,
    refl,
  end

end quadrilateral_perimeter_l290_290670


namespace sequence_formula_l290_290539

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 5)
  (h3 : ∀ n > 1, a (n + 1) = 2 * a n - a (n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end sequence_formula_l290_290539


namespace shaded_square_percentage_l290_290572

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 16) (h2 : shaded_squares = 8) : 
  (shaded_squares : ℚ) / total_squares * 100 = 50 :=
by
  sorry

end shaded_square_percentage_l290_290572


namespace wisdom_number_2006_l290_290392

/-- A positive integer is called a Wisdom Number if it can be expressed as the difference
    of squares of two positive integers. -/
def isWisdomNumber (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > b ∧ n = a^2 - b^2

/-- Define the sequence of Wisdom Numbers -/
def wisdomNumbers (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 5
  else if n = 3 then 7
  else let k := (n - 2) / 3 in
    match n % 3 with
    | 1 => 4 * (k + 2)
    | 2 => 4 * (k + 2) + 1
    | _ => 4 * (k + 2) + 2

/-- Prove that the 2006th Wisdom Number is 2677. -/
theorem wisdom_number_2006 : wisdomNumbers 2006 = 2677 :=
  sorry

end wisdom_number_2006_l290_290392


namespace sum_abs_ai_l290_290771

def P (x : ℝ) := 1 - (1/2) * x + (1/4) * x^2
def Q (x : ℝ) := P(x) * P(x^2) * P(x^4) * P(x^6)

theorem sum_abs_ai : (∑ i in Finset.range 21, |(Q(0 : ℝ).coeff i)|) = 189 / 256 := by
  sorry

end sum_abs_ai_l290_290771


namespace royal_children_count_l290_290627

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290627


namespace age_ratio_in_4_years_l290_290563

-- Definitions based on the conditions
def pete_age (years_ago : ℕ) (p : ℕ) (c : ℕ) : Prop :=
match years_ago with
  | 2 => p - 2 = 3 * (c - 2)
  | 4 => p - 4 = 4 * (c - 4)
  | _ => true
end

-- Question: In how many years will the ratio of their ages be 2:1?
def age_ratio (years : ℕ) (p : ℕ) (c : ℕ) : Prop :=
(p + years) / (c + years) = 2

-- Proof problem
theorem age_ratio_in_4_years {p c : ℕ} (h1 : pete_age 2 p c) (h2 : pete_age 4 p c) : 
  age_ratio 4 p c :=
sorry

end age_ratio_in_4_years_l290_290563


namespace complex_z_pow_condition_l290_290391

theorem complex_z_pow_condition (z : ℂ) (h : z + z⁻¹ = 2 * real.sqrt 2) : z^100 + z^(-100) = -2 := by
  sorry

end complex_z_pow_condition_l290_290391


namespace remainder_of_product_mod_5_l290_290283

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l290_290283


namespace train_length_250_05_l290_290692

noncomputable def length_of_train (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  speed_m_s * time_s

theorem train_length_250_05 : length_of_train 60 15 = 250.05 :=
by
  -- Definitions from the problem
  let speed_km_hr := 60
  let time_s := 15
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let distance := speed_m_s * time_s
  -- The proven assertion
  show distance = 250.05
  sorry

end train_length_250_05_l290_290692


namespace royal_family_children_l290_290639

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290639


namespace royal_children_count_l290_290622

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290622


namespace correct_statement_A_l290_290576

theorem correct_statement_A : True :=
by {
  -- Define the dataset
  let dataset := [0, 1, 2, 1, 1],
  
  -- Calculate the mode and median
  let mode := (1 : ℕ), -- Mode is the most frequent element, which is 1 here.
  let median := (1 : ℕ), -- Median is the middle element when sorted, which is 1 here.
  
  -- Prove that both are equal to 1
  have mode_property : mode = 1 := rfl,
  have median_property : median = 1 := rfl,

  -- Since the problem context claims correctness when both mode and median are 1
  trivial
}

end correct_statement_A_l290_290576


namespace scientific_notation_sesame_seed_mass_l290_290698

-- Define the mass of the sesame seed
def sesame_seed_mass : ℝ := 0.00000201

-- Prove that the scientific notation of sesame_seed_mass is 2.01 * 10^(-6)
theorem scientific_notation_sesame_seed_mass : sesame_seed_mass = 2.01 * 10^(-6) :=
by
  sorry

end scientific_notation_sesame_seed_mass_l290_290698


namespace abs_inequality_holds_l290_290296

theorem abs_inequality_holds (m x : ℝ) (h : -1 ≤ m ∧ m ≤ 6) : 
  |x - 2| + |x + 4| ≥ m^2 - 5 * m :=
sorry

end abs_inequality_holds_l290_290296


namespace coeff_x2_expansion_l290_290420

theorem coeff_x2_expansion (a b : ℕ) :
  polynomial.coeff (polynomial.expand 4 (polynomial.X - 3 * polynomial.X)) 2 = 120 := by
sorry

end coeff_x2_expansion_l290_290420


namespace hexagon_area_correctness_l290_290722

-- Define the conditions
def side_length : ℝ := 3
def individual_triangle_area : ℝ :=
  (sqrt 3 / 4) * (side_length ^ 2)

def hexagon_area : ℝ :=
  6 * individual_triangle_area

-- The key property to prove
theorem hexagon_area_correctness :
  ∃ m n : ℤ, hexagon_area = 3 * sqrt ↑m + n ∧ m + n = 27 :=
by
  -- Given that each side of the hexagon is 3, calculate
  have h1 : individual_triangle_area = (9 * sqrt 3) / 4,
    sorry  -- Calculation of an individual triangle area
  
  -- Total area of the hexagon
  have h2 : hexagon_area = 6 * ((9 * sqrt 3) / 4),
    sorry  -- Calculation of the entire hexagon’s area

  -- Express in the requested form
  use [27, 0],
  split,
  sorry, -- Proof of equality in the form
  simp,
  norm_num,

end hexagon_area_correctness_l290_290722


namespace probability_not_snow_l290_290972

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l290_290972


namespace number_of_arrangements_l290_290138

-- Define the conditions
def students : ℕ := 5
def teachers : ℕ := 2
def total_people : ℕ := students + teachers

-- Define the problem statement
theorem number_of_arrangements : 
  ∃ n : ℕ, n = 960 ∧ 
  ∀ (people : list ℕ) (orderings : list (list ℕ)),
    (people.length = total_people) → 
    (∀ people, people ⊆ (list.range total_people)) →
    (∃ sublist, (sublist.length = 4) ∧ sublist.countp (λ x, x < teachers) = 2) →
    (∀ (p1 p2 : ℕ), p1 ≠ p2 → (p1 ∈ people) → (p2 ∈ people) → 
      (∃ subindices, (subindices.length = 2) ∧ subindices.nodup = true ∧ 
        ∃ (i j : ℕ), (people.nth i = some p1 ∧ people.nth j = some p2) ∧ abs (i - j - 2) = 3)) :=
begin
  sorry
end

end number_of_arrangements_l290_290138


namespace max_minus_min_depends_on_a_not_b_l290_290837

def quadratic_function (a b x : ℝ) : ℝ := x^2 + a * x + b

theorem max_minus_min_depends_on_a_not_b (a b : ℝ) :
  let f := quadratic_function a b
  let M := max (f 0) (f 1)
  let m := min (f 0) (f 1)
  M - m == |a| :=
sorry

end max_minus_min_depends_on_a_not_b_l290_290837


namespace sum_f_eq_39_l290_290769

def f (n : ℕ) : ℝ :=
if (∃ k, (log 4 n = k / 2)) then log 4 n else 0

theorem sum_f_eq_39 : (∑ n in Finset.range 4095.succ, f n) = 39 := by
  sorry

end sum_f_eq_39_l290_290769


namespace four_digit_divisible_by_12_l290_290304

theorem four_digit_divisible_by_12 (n : ℕ) : 3150 + n = 3156 → n = 6 :=
by {
    intro h,
    linarith,
    sorry
}

end four_digit_divisible_by_12_l290_290304


namespace find_reflection_point_l290_290671

noncomputable def point (α : Type) [Field α] := (α × α × α)

def A : (ℚ × ℚ × ℚ) := (-3, 9, 11)
def B : (ℚ × ℚ × ℚ) := (-5/3, 16/3, 25/3)
def C : (ℚ × ℚ × ℚ) := (3, 5, 9)
def plane (p : (ℚ × ℚ × ℚ)) : Prop := p.1 + p.2 + p.3 = 12

theorem find_reflection_point :
  ∃ (B : ℚ × ℚ × ℚ), B = (-5/3, 16/3, 25/3) ∧
  plane B ∧
  collinear A B C :=
begin
  sorry
end

end find_reflection_point_l290_290671


namespace sum_sequence_l290_290424

noncomputable def a : ℕ → ℚ
| 1 := 1
| (n+1) := (↑((n+1)^2) / (↑((n+1)^2 - 1))) * a n

noncomputable def T_n (n : ℕ) : ℚ :=
(∑ i in Finset.range n, a (i + 1) / (↑(i + 1) ^ 2))

theorem sum_sequence (n : ℕ) :
T_n n = (2 * ↑n) / (↑n + 1) := sorry

end sum_sequence_l290_290424


namespace gain_percent_l290_290393

-- Definitions and Conditions
variables {C S : ℝ}
variable (h : 65 * C = 50 * S)

-- Goal: Prove the gain percent
theorem gain_percent (h : 65 * C = 50 * S) : ((15 / 50) * 100 = 30) :=
by
  sorry

end gain_percent_l290_290393


namespace not_necessarily_periodic_difference_l290_290206

-- Definition of periodic function
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f(x + p) = f(x)

-- Given Conditions
variable (g h : ℝ → ℝ)
axiom g_periodic : is_periodic g 6
axiom h_periodic : is_periodic h (2 * Real.pi)

-- The proof problem
theorem not_necessarily_periodic_difference : ¬∃ p > 0, is_periodic (λ x, g(x) - h(x)) p := sorry

end not_necessarily_periodic_difference_l290_290206


namespace people_count_l290_290486

theorem people_count (wheels_per_person total_wheels : ℕ) (h1 : wheels_per_person = 4) (h2 : total_wheels = 320) :
  total_wheels / wheels_per_person = 80 :=
sorry

end people_count_l290_290486


namespace regular_permutation_condition_l290_290789

-- Define the condition of a legal transposition
def is_legal_transposition (a : List ℕ) (i j : ℕ) : Prop :=
  i > 0 ∧ a[i] = 0 ∧ a[i - 1] + 1 = a[j]

-- Define what it means for a permutation to be regular
def is_regular_permutation (a : List ℕ) (n : ℕ) : Prop :=
  ∃ b : List ℕ, (∀ i j, is_legal_transposition a i j → b = (List.range (n + 1).tail ++ [0])) ∧
  b = (List.range (n + 1).filter (≠ 0) ++ [0])

-- Define the main theorem
theorem regular_permutation_condition (n : ℕ) : 
  is_regular_permutation [1] |>.append ([n, n-1, ..., 3, 2, 0]) n ↔
  n = 2 ∨ ∃ k : ℕ, n = 2^k - 1 := sorry

end regular_permutation_condition_l290_290789


namespace unique_zero_in_interval_l290_290829

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

theorem unique_zero_in_interval (a : ℝ) (h : a > 2) :
  ∃! x ∈ Ioo (0 : ℝ) (2 : ℝ), f a x = 0 :=
begin
  sorry
end

end unique_zero_in_interval_l290_290829


namespace domain_F_eq_4_5_l290_290011

def f (x : ℝ) : ℝ := 5 * x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem domain_F_eq_4_5 :
  {x : ℝ | x ∈ {0, 2, 3, 4, 5} ∧ f x > 0 ∧ g x > 0} = {4, 5} :=
by {
  sorry
}

end domain_F_eq_4_5_l290_290011


namespace Carla_total_counts_l290_290749

def Monday_counts := (60 * 2) + (120 * 2) + (10 * 2)
def Tuesday_counts := (60 * 3) + (120 * 2) + (10 * 1)
def Wednesday_counts := (80 * 4) + (24 * 5)
def Thursday_counts := (60 * 1) + (80 * 2) + (120 * 3) + (10 * 4) + (24 * 5)
def Friday_counts := (60 * 1) + (120 * 2) + (80 * 2) + (10 * 3) + (24 * 3)

def total_counts := Monday_counts + Tuesday_counts + Wednesday_counts + Thursday_counts + Friday_counts

theorem Carla_total_counts : total_counts = 2552 :=
by 
  sorry

end Carla_total_counts_l290_290749


namespace seating_arrangement_l290_290751

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 6 * y = 57) : x = 1 :=
sorry

end seating_arrangement_l290_290751


namespace triangle_ratios_equal_l290_290471

theorem triangle_ratios_equal {A B C A' B' C' : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace A'] [MetricSpace B'] [MetricSpace C'] 
  (triangle1 : Triangle A B C) 
  (triangle2 : Triangle A' B' C') 
  (h1 : ∠ A B C = ∠ B' A' C') 
  (h2 : ∠ A C B = ∠ A' C' B') 
  (h3 : ∠ C B A = ∠ C' B' A') :
  (dist A B / dist A C) = (dist A' B' / dist A' C') :=
by sorry

end triangle_ratios_equal_l290_290471


namespace simon_sand_dollars_l290_290505

theorem simon_sand_dollars (S G P : ℕ) (h1 : G = 3 * S) (h2 : P = 5 * G) (h3 : S + G + P = 190) : S = 10 := by
  sorry

end simon_sand_dollars_l290_290505


namespace vector_dot_product_l290_290401

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end vector_dot_product_l290_290401


namespace octahedron_hexagon_cover_l290_290035
  
  -- Definitions for conditions
  def is_octahedron_surface (shape : Type) := 
    ∃ f : shape → set ℝ^3, (f.shape = 8 ∧ ∀ face, shape.face.is_triangle)

  def can_cover_with_hexagons (surface : set ℝ^3) := 
    ∃ hexagons : set (set ℝ^3), (∀ hexagon ∈ hexagons, is_regular_hexagon hexagon) ∧
    (⋃ h ∈ hexagons, h) = surface ∧ 
    (∀ h₁ h₂ ∈ hexagons, h₁ ≠ h₂ → h₁ ∩ h₂ = ∅)

  -- Main theorem
  theorem octahedron_hexagon_cover : 
    ∀ surface, is_octahedron_surface surface → can_cover_with_hexagons surface :=
  by
    sorry
  
end octahedron_hexagon_cover_l290_290035


namespace sum_possible_amounts_l290_290915

def isValidChange (amount : ℕ) : Prop :=
  amount < 200

def conditionHalfDollar (amount : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 3 ∧ (amount = 50 * k + 20)

def conditionQuarters (amount : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 7 ∧ (amount = 25 * k + 35)

def isValidAmount (amount : ℕ) : Prop :=
  isValidChange amount ∧ conditionHalfDollar amount ∧ conditionQuarters amount

theorem sum_possible_amounts : 
  ∑ x in {amount : ℕ | isValidAmount amount}.toFinset, x = 190 :=
by
  sorry

end sum_possible_amounts_l290_290915


namespace John_lost_socks_l290_290437

theorem John_lost_socks (initial_socks remaining_socks : ℕ) (H1 : initial_socks = 20) (H2 : remaining_socks = 14) : initial_socks - remaining_socks = 6 :=
by
-- Proof steps can be skipped
sorry

end John_lost_socks_l290_290437


namespace area_of_XQRY_l290_290152

theorem area_of_XQRY (P Q R M N X Y : Type) 
  (PQ PR : ℝ) (area_PQR : ℝ) (M_midpoint : (M = midpoint P Q)) 
  (N_midpoint : (N = midpoint P R)) (angle_bisector_X : (X = intersection (angle_bisector Q P R) (line M N))) 
  (angle_bisector_Y : (Y = intersection (angle_bisector Q P R) (line Q R)))
  (PQ_val : PQ = 40) (PR_val : PR = 20) (area_PQR_val : area_PQR = 160) :
  ∃ (area_XQRY : ℝ), area_XQRY = 64 :=
sorry

end area_of_XQRY_l290_290152


namespace start_difference_A_B_l290_290847

-- Definitions for the conditions
def start_difference_A_C : ℝ := 200
def start_difference_B_C : ℝ := 120.87912087912093

-- The theorem statement
theorem start_difference_A_B : true :=
  show ((start_difference_A_C : ℝ) = (79.12087912087907 + start_difference_B_C)) from sorry

end start_difference_A_B_l290_290847


namespace number_of_elements_in_B_l290_290882

open Set

def A : Set ℤ := {x | x ∈ {x | abs x ≤ 3}}
def B : Set ℕ := {y | ∃ x ∈ A, y = x^2 + 1}

theorem number_of_elements_in_B : (B.to_finset.card = 4) :=
by
  sorry

end number_of_elements_in_B_l290_290882


namespace symmetric_points_power_l290_290416

theorem symmetric_points_power 
  (a b : ℝ) 
  (h1 : 2 * a = 8) 
  (h2 : 2 = a + b) :
  a^b = 1/16 := 
by sorry

end symmetric_points_power_l290_290416


namespace pencils_left_l290_290704

theorem pencils_left (anna_pencils : ℕ) (harry_pencils : ℕ)
  (h_anna : anna_pencils = 50) (h_harry : harry_pencils = 2 * anna_pencils)
  (lost_pencils : ℕ) (h_lost : lost_pencils = 19) :
  harry_pencils - lost_pencils = 81 :=
by
  sorry

end pencils_left_l290_290704


namespace num_cute_6_digit_integers_l290_290703

def is_cute (n : ℕ) : Prop :=
  ∃ (digits : Finset ℕ) (hd : digits.to_list.length = 6), 
    digits = {1, 2, 3, 4, 5, 6} ∧ (∀ (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 6), 
    (digits.to_list.take k).foldl (λ acc d, acc * 10 + d) 0 % k = 0)

def is_valid_cute_count (count : ℕ) : Prop :=
  ∃ (cute_nums : Finset ℕ), cute_nums.card = count ∧ (∀ n ∈ cute_nums, is_cute n)

theorem num_cute_6_digit_integers : is_valid_cute_count 2 :=
sorry

end num_cute_6_digit_integers_l290_290703


namespace no_integer_root_l290_290880

theorem no_integer_root (P : ℤ[X]) (h1 : P.coeff 0 = 4) (h2 : P.eval 1 = 10) (h3 : P.eval (-1) = 22) :
  ¬ ∃ r : ℤ, P.eval r = 0 :=
sorry

end no_integer_root_l290_290880


namespace max_and_min_ones_l290_290247

noncomputable def max_num_ones (n : ℕ) (h : n ≥ 3) : ℕ :=
  if n = 3 then 1 else (n - 2) ^ 2 - 1

noncomputable def min_num_ones (n : ℕ) (h : n ≥ 3) : ℕ :=
  n - 2

theorem max_and_min_ones (n : ℕ) (h : n ≥ 3) :
  (∃ table : matrix (fin n) (fin n) ℤ, 
    (∀ i, (i = 0 ∨ i = n-1) → table i 0 = -1 ∧ table i (n-1) = -1 ∧ table 0 i = -1 ∧ table (n-1) i = -1) ∧
    (∀ i j, (0 < i ∧ i < n-1 ∧ 0 < j ∧ j < n-1) → 
      table i j = (table (i-1) j * table (i+1) j) ∨ table i j = (table i (j-1) * table i (j+1))) ∧
    let ones_count := finset.univ.filter (λ ij, table ij.1 ij.2 = 1) in
    finset.card ones_count = max_num_ones n h ∨ finset.card ones_count = min_num_ones n h) :=
sorry

end max_and_min_ones_l290_290247


namespace evaluate_expression_l290_290754

theorem evaluate_expression (a : ℕ) (h : a = 2023) : 
  (2023^3 - 3 * 2023^2 * 2024 + 5 * 2023 * 2024^2 - 2024^3 + 5) / (2023 * 2024) = 4048 :=
by
  rw [h]
  sorry

end evaluate_expression_l290_290754


namespace sum_F_arithmetic_sequence_sum_an_bn_l290_290346

-- Define the function F.
def F (x : ℝ) : ℝ := (3 * x - 2) / (2 * x - 1)

-- Define the sequence a_n.
def a : ℕ → ℝ
| 1     := 2
| (n+1) := F (a n)

-- Define the sequence b_n.
def b (n : ℕ) : ℝ := (2 * n - 1) / 2^n

-- Define the sum S_n of the sequence {a_n b_n}.
def S (n : ℕ) : ℝ := ∑ k in Finset.range n, (a (k + 1) * b (k + 1))

-- Prove that F(1/2010) + F(2/2010) + ... + F(2009/2010) = 6027/2.
theorem sum_F : ∑ k in Finset.range 2009, F ((k + 1) / 2010) = 6027 / 2 :=
sorry

-- Prove that the sequence {1 / (a_n - 1)} is an arithmetic sequence.
theorem arithmetic_sequence : ∀ n : ℕ, (1 / (a (n + 1) - 1)) = (1 + n * 2) :=
sorry

-- Prove that the sum of the first n terms of {a_n b_n} is S_n = 4 - (2 + n) / 2^(n-1).
theorem sum_an_bn : ∀ n : ℕ, S n = 4 - (2 + n) / 2^(n - 1) :=
sorry

end sum_F_arithmetic_sequence_sum_an_bn_l290_290346


namespace no_snow_probability_l290_290970

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l290_290970


namespace angle_B_measure_l290_290019

noncomputable section

-- Definitions
variables {A B C G : Type*} [metric_space G] [normed_group G] [add_comm_group G]
  [vector_space ℝ G] [has_dist G G] -- Basic space and vector assumptions

-- Given conditions
def is_centroid (G : G) (A B C : G) : Prop := 
  ∃ n, G = (n • A + n • B + n • C) / 3

-- The vector equation given in problem
def vector_equation (a b c : ℝ) (GA GB GC : G) : Prop :=
  56 * a • GA + 40 * b • GB + 35 * c • GC = 0

-- The main theorem to prove
theorem angle_B_measure (A B C G : G) (a b c : ℝ) 
  (h1 : is_centroid G A B C)
  (h2 : vector_equation a b c (G - A) (G - B) (G - C)) :
  ∠ B = 60 := 
sorry

end angle_B_measure_l290_290019


namespace average_xy_l290_290121

theorem average_xy (x y : ℝ) 
  (h : (4 + 6 + 9 + x + y) / 5 = 20) : (x + y) / 2 = 40.5 :=
sorry

end average_xy_l290_290121


namespace ordinary_eq_from_param_eq_l290_290533

theorem ordinary_eq_from_param_eq (α : ℝ) :
  (∃ (x y : ℝ), x = 3 * Real.cos α + 1 ∧ y = - Real.cos α → x + 3 * y - 1 = 0 ∧ (-2 ≤ x ∧ x ≤ 4)) := 
sorry

end ordinary_eq_from_param_eq_l290_290533


namespace rearrange_pegs_l290_290225

-- Define the board as a 7x7 grid
def board : Set (ℕ × ℕ) := { (i, j) | i ∈ Finset.range 7 ∧ j ∈ Finset.range 7 }

-- The initial positions of the 10 pegs
def initial_pegs : Set (ℕ × ℕ) := 
  -- specify the initial positions of the 10 pegs
  sorry

-- The positions to which the 3 pegs should be moved
def new_positions : Set (ℕ × ℕ) := 
  -- specify the new positions for the 3 pegs
  sorry

-- The condition to verify i.e., there exist 5 rows each containing exactly 4 pegs
def valid_configuration (pegs : Set (ℕ × ℕ)) : Prop :=
  -- Check if there are 5 rows each containing exactly 4 pegs
  sorry

-- The Lean statement to prove
theorem rearrange_pegs (pegs : Set (ℕ × ℕ)) (initial_pegs_subset : initial_pegs ⊆ board) 
  (new_positions_subset : new_positions ⊆ board) : 
  valid_configuration (initial_pegs \ new_positions ∪ new_positions) :=
by 
  sorry

end rearrange_pegs_l290_290225


namespace movie_original_length_l290_290189

-- Conditions
def final_length (x : ℕ) := x = 52
def cut_scene (y : ℕ) := y = 8
def original_length (x y z : ℕ) := z = x + y

-- Proof statement
theorem movie_original_length (x y z : ℕ) (h1 : final_length x) (h2 : cut_scene y) (h3 : original_length x y z) : z = 60 :=
by {
  simp [final_length, cut_scene, original_length] at *,
  sorry
}

end movie_original_length_l290_290189


namespace ellipse_and_parabola_properties_l290_290794

def C1_is_ellipse (a b : ℝ) (h : 0 < b ∧ b < a) (e : ℝ) (h_e : e = 1 / 2) : Prop :=
  let ellipse_eq := (x : ℝ ) (y : ℝ) => (x^2) / (a^2) + (y^2) / (b^2) = 1
  (a = 4) ∧ (b = 2 * sqrt 3) ∧ (ellipse_eq (0) (2 * sqrt 3))

noncomputable def C2_is_parabola (p : ℝ) (c : ℝ) (a : ℝ) (h_c_a : c = a / 2) : Prop :=
  let parabola_eq := (y : ℝ) (x : ℝ) => y^2 = 2 * p * x
  (p = 4) ∧ (parabola_eq (2 * c))

def minimum_area (k : ℝ) (h_c_a: k^2 + 2 + 1 / (k^2) ≥ 4) : ℝ :=
  let area := 16 * sqrt ((k^2 + 2 + 1 / k^2) * (2 * (k^2) + 5 + 2 / k^2))
  if k = 1 ∨ k = -1 then area else area / √(8)

theorem ellipse_and_parabola_properties (a b p c : ℝ)
  (h1: C1_is_ellipse a b (and.intro (by assumption) (by assumption)) (1/2))
  (h2: C2_is_parabola p c a)
  (h3: minimum_area (1) (by simp : 1^2 + 2 + 1 / 1^2 ≥ 4) = 96) :
  (a = 4) ∧ (b = 2 * sqrt 3) ∧ (p = 4) ∧ (h3 = 96) := sorry

end ellipse_and_parabola_properties_l290_290794


namespace fraction_of_coins_l290_290938

theorem fraction_of_coins (total_states joining_states : ℕ) 
  (h₁ : total_states = 32) 
  (h₂ : joining_states = 7) :  
  (joining_states:ℚ) / total_states = 7 / 32 :=
by 
  -- We skip the proof using sorry
  sorry

end fraction_of_coins_l290_290938


namespace polynomial_P_at_3_l290_290446

noncomputable def P (x : ℝ) : ℝ := ∑ i in (Finset.range (n + 1)), (coeffs i) * x^i

theorem polynomial_P_at_3 (P : ℝ → ℝ) (coeffs : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, (0 ≤ coeffs i) ∧ (coeffs i < 5))
  (h2 : P (sqrt 5) = 50 + 21 * (sqrt 5)) :
  P 3 = 273 :=
sorry

end polynomial_P_at_3_l290_290446


namespace find_a_l290_290960

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f(x) = 1 / x + log 2 ((1 + a * x) / (1 - x)) ∧ f (-x) = -f (x)) : a = 1 :=
sorry

end find_a_l290_290960


namespace complex_multiplication_correct_l290_290070

def z1 : ℂ := 2 + complex.i
def z2 : ℂ := 2 - 3 * complex.i

theorem complex_multiplication_correct : z1 * z2 = (7 - 4 * complex.i) := 
by
  sorry

end complex_multiplication_correct_l290_290070


namespace radical_center_exists_l290_290177

theorem radical_center_exists 
  (A B C D O1 O2 T1 T2 : Type) 
  [affine_space A] [plane_space B] [plane_space C] [plane_space D]
  [circle_inscribed_in_triangle O1 ABC T1] [circle_inscribed_in_triangle O2 ABD T2]
  (O1O2_midpoint : midpoint T1 T2 = O1O2)
  (πAB ⊥ O1O2) (πAC ⊥ O3O4) (πBC ⊥ O5O6) (πAD ⊥ O7O8) (πBD ⊥ O9O10) (πCD ⊥ O11O12)
  (midpoint_Basis : B ⊥ O1O2 ∧ C ⊥ O3O4 ∧ D ⊥ O5O6 ∧ A ⊥ O7O8 ∧ B ⊥ O9O10 ∧ C ⊥ O11O12)
  : ∃ (P : Type), πAB ≠ ⊥ P ∧ πAC ≠ ⊥ P ∧ πBC ≠ ⊥ P ∧ πAD ≠ ⊥ P ∧ πBD ≠ ⊥ P ∧ πCD ≠ ⊥ P :=
sorry

end radical_center_exists_l290_290177


namespace smaller_angle_measure_l290_290993

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l290_290993


namespace part1_part2_l290_290798

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x : ℝ) : ℝ := min (f x) (g x)

theorem part1 :
  ∃ ! x ∈ set.Ioo 1 ⊤, F x = 0 :=
sorry

theorem part2 (c : ℝ) (x₁ x₂ x₀ : ℝ) (h₁ : x₀ ∈ set.Ioo 1 ⊤) (h₂ : x₁ < x₂) 
               (h₃ : x₁ ∈ set.Ioo 1 x₀) (h₄ : x₂ ∈ set.Ioo x₀ ⊤) (h₅ : m x₁ = c) 
               (h₆ : m x₂ = c) (h₇ : F x₀ = 0) :
  (x₁ + x₂) / 2 > x₀ :=
sorry

end part1_part2_l290_290798


namespace probability_of_forming_number_between_20_and_30_l290_290099

def is_valid_number_formed_by_dice (d1 d2 : ℕ) : Prop :=
  (10 * d1 + d2 >= 20 ∧ 10 * d1 + d2 <= 30) ∨ (10 * d2 + d1 >= 20 ∧ 10 * d2 + d1 <= 30)

def successful_outcomes (dice : list (ℕ × ℕ)) : list (ℕ × ℕ) :=
  dice.filter (λ d, is_valid_number_formed_by_dice d.1 d.2)

theorem probability_of_forming_number_between_20_and_30 :
  let dice_rolls := [(d1, d2) | d1 ← [1, 2, 3, 4, 5, 6], d2 ← [1, 2, 3, 4, 5, 6]] in
  let successful := successful_outcomes dice_rolls in
  (successful.length : ℚ) / (dice_rolls.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_forming_number_between_20_and_30_l290_290099


namespace john_billed_minutes_l290_290299

theorem john_billed_minutes
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (total_bill : ℝ := 12.02) :
  ∃ (minutes : ℕ), minutes = 28 :=
by
  have amount_for_minutes := total_bill - monthly_fee
  have minutes_float := amount_for_minutes / cost_per_minute
  have minutes := floor minutes_float
  use minutes
  have : 28 = (floor minutes_float : ℕ) := sorry
  exact this.symm

end john_billed_minutes_l290_290299


namespace probability_of_scatterbrained_pills_l290_290939

namespace ScatterbrainedScientist

def R : Prop := "The Scientist took the pills from the jar for scattering treatment."
def A : Prop := "The knee pain stopped."
def B : Prop := "The scattering disappeared."

variable (prob_R : ℝ) (prob_not_R : ℝ) (prob_A_given_R : ℝ) (prob_B_given_R : ℝ)
variable (prob_A_given_not_R : ℝ) (prob_B_given_not_R : ℝ)
variable (prob_R_given_A_and_B : ℝ)

-- Conditions: initial probabilities and conditional probabilities
axiom prob_R : prob_R = 0.5
axiom prob_not_R : prob_not_R = 0.5
axiom prob_B_given_R : prob_B_given_R = 0.80
axiom prob_A_given_R : prob_A_given_R = 0.05
axiom prob_A_given_not_R : prob_A_given_not_R = 0.90
axiom prob_B_given_not_R : prob_B_given_not_R = 0.02

-- The probability to be proven
axiom calculated_prob_R_given_A_and_B :
  (prob_R * prob_B_given_R * prob_A_given_R) /
  ((prob_R * prob_B_given_R * prob_A_given_R) + (prob_not_R * prob_A_given_not_R * prob_B_given_not_R)) = 0.69

theorem probability_of_scatterbrained_pills :
  prob_R_given_A_and_B = 0.69 :=
sorry

end ScatterbrainedScientist

end probability_of_scatterbrained_pills_l290_290939


namespace total_mile_times_l290_290146

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l290_290146


namespace sequence_count_l290_290886

def Triangle : Type :=
  { vertices : Fin 3 → ℝ × ℝ // set (vertices (Fin.mk 0 _)) = {(0, 0)} ∧ 
                                   set (vertices (Fin.mk 1 _)) = {(3, 0)} ∧ 
                                   set (vertices (Fin.mk 2 _)) = {(0, 2)} }

inductive Transformation : Type
| Rot90
| Rot180
| Rot270
| ReflectX
| ReflectYeqX

open Transformation

def apply_transformation (T : Triangle) (t : Transformation) : Triangle := sorry

def apply_transformations (T : Triangle) (ts : List Transformation) : Triangle :=
  ts.foldl apply_transformation T

def returns_to_original (T : Triangle) (ts : List Transformation) : Prop :=
  apply_transformations T ts = T

theorem sequence_count (T : Triangle) (seq_count : set (List Transformation).card = 125) :
  (∃ sequences : set (List Transformation), 
    sequences.card = 12 ∧ 
    ∀ ts ∈ sequences, returns_to_original T ts) := sorry

end sequence_count_l290_290886


namespace total_mile_times_l290_290150

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l290_290150


namespace kathleen_spent_on_school_supplies_l290_290876

theorem kathleen_spent_on_school_supplies :
  let june_savings := 21
      july_savings := 46
      august_savings := 45
      clothes_spent := 54
      total_left := 46
      total_savings := june_savings + july_savings + august_savings
      S := total_savings - total_left - clothes_spent in
  S = 12 :=
by
  let june_savings := 21
  let july_savings := 46
  let august_savings := 45
  let clothes_spent := 54
  let total_left := 46
  let total_savings := june_savings + july_savings + august_savings
  let S := total_savings - total_left - clothes_spent
  show S = 12
  sorry

end kathleen_spent_on_school_supplies_l290_290876


namespace dave_guitar_strings_l290_290726

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l290_290726


namespace num_pos_ints_satisfying_cond_l290_290742

theorem num_pos_ints_satisfying_cond : 
  {n : ℕ // (n + 1500) % 90 = 0 ∧ ((n + 1500) / 90 = (⌊real.nthRoot 3 (n:ℝ)⌋).toNat)}.card = 2 := by
  sorry

end num_pos_ints_satisfying_cond_l290_290742


namespace DHQ_perpendicular_l290_290327

-- Definitions of points in the Euclidean plane
variables {A B C D P Q H : Type*} [EuclideanGeometry.Point A] [EuclideanGeometry.Point B]
[EuclideanGeometry.Point C] [EuclideanGeometry.Point D] [EuclideanGeometry.Point P]
[EuclideanGeometry.Point Q] [EuclideanGeometry.Point H]

-- Given the square ABCD, conditions P and Q on AB and BC, respectivly, BP = BQ, and H is the foot of the perpendicular from B to PC,
-- we aim to prove that angle DHQ is 90 degrees.
theorem DHQ_perpendicular (h_square : EuclideanGeometry.Square A B C D)
  (h_P_on_AB : EuclideanGeometry.OnLine P A B)
  (h_Q_on_BC : EuclideanGeometry.OnLine Q B C)
  (h_BP_BQ_equal : EuclideanGeometry.dist B P = EuclideanGeometry.dist B Q)
  (h_H_perpendicular : EuclideanGeometry.Perpendicular B H (EuclideanGeometry.Segment P C))
  : EuclideanGeometry.Angle D H Q = 90 := 
by sorry

end DHQ_perpendicular_l290_290327


namespace deepak_meet_time_l290_290526

theorem deepak_meet_time
  (track_circumference : ℕ)
  (deepak_speed_kmh : ℕ)
  (wife_speed_kmh : ℕ)
  (deepak_speed_mpm : ℚ)
  (wife_speed_mpm : ℚ)
  (relative_speed_mpm : ℚ)
  (time_to_meet : ℚ)
  (h1 : track_circumference = 1000)
  (h2 : deepak_speed_kmh = 20)
  (h3 : wife_speed_kmh = 13)
  (h4 : deepak_speed_mpm = (deepak_speed_kmh * 1000) / 60)
  (h5 : wife_speed_mpm = (wife_speed_kmh * 1000) / 60)
  (h6 : relative_speed_mpm = deepak_speed_mpm + wife_speed_mpm)
  (h7 : time_to_meet = track_circumference / relative_speed_mpm) :
  time_to_meet ≈ 1.82 :=
by
  sorry

end deepak_meet_time_l290_290526


namespace sqrt_1_0201_eq_1_01_l290_290007

theorem sqrt_1_0201_eq_1_01 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 :=
by 
  sorry

end sqrt_1_0201_eq_1_01_l290_290007


namespace perimeter_PQRS_l290_290859

-- Points Q and S are given specific coordinates
def Q : ℝ × ℝ := (0, 0)
def S : ℝ × ℝ := (0, 4)
-- Distance QR and PS are given as 3 units each
def QR : ℝ := 3
def PS : ℝ := 3

-- Use the Pythagorean theorem to determine that RS equals 5
lemma QR_eq_3 : dist (Q, S) = QR then QR = 3 := sorry
lemma PS_eq_3 : dist (Q, S) = PS then PS = 3 := sorry
lemma Pythagorean_RS : sqrt (QR^2 + QS^2) = 5 := sorry

-- Prove that the perimeter of the parallelogram is 16
theorem perimeter_PQRS : 2 * (QR + RS) + 2 * (PS + PQ) = 16 :=
by
have h1 : dist(R,S) = 5 from Pythagorean_RS
have h2 : dist(P,Q) = dist(R,S)
arb_calc 
(3 + 5) = 16
by
simp

end perimeter_PQRS_l290_290859


namespace fifth_color_marbles_l290_290848

theorem fifth_color_marbles :
  let red := 25
  let green := 3 * red
  let yellow := 20 * green / 100
  let blue := 2 * yellow
  let total := 4 * green
  in total - (red + green + yellow + blue) = 155 :=
by {
  let red := 25
  let green := 3 * red
  let yellow := 20 * green / 100
  let blue := 2 * yellow
  let total := 4 * green
  have h : total - (red + green + yellow + blue) = 155 := by {
    sorry
  }
  exact h
}

end fifth_color_marbles_l290_290848


namespace product_mod_5_l290_290291

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l290_290291


namespace male_athletes_sampled_l290_290679

def number_of_male_athletes := 32
def total_number_of_athletes := 56
def sample_size := 14
def sample_fraction := sample_size.toRational / total_number_of_athletes.toRational

theorem male_athletes_sampled : (number_of_male_athletes * sample_fraction).toInt = 8 := by
  sorry

end male_athletes_sampled_l290_290679


namespace difference_between_wins_and_losses_l290_290655

noncomputable def number_of_wins (n m : ℕ) : Prop :=
  0 ≤ n ∧ 0 ≤ m ∧ n + m ≤ 42 ∧ n + (42 - n - m) / 2 = 30 / 1

theorem difference_between_wins_and_losses (n m : ℕ) (h : number_of_wins n m) : n - m = 18 :=
sorry

end difference_between_wins_and_losses_l290_290655


namespace product_end_digit_3_mod_5_l290_290286

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l290_290286


namespace royal_family_children_l290_290638

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l290_290638


namespace tan_alpha_solution_l290_290781

-- Define the conditions and goal according to the problem statement
theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 4) = -3 / 5) : 
  tan α = 7 := 
by 
  sorry

end tan_alpha_solution_l290_290781


namespace sum_factorial_mod_7_l290_290244

theorem sum_factorial_mod_7 :
  (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10!) % 7 = 5 :=
sorry

end sum_factorial_mod_7_l290_290244


namespace n_value_for_315n_divisible_by_12_l290_290305

theorem n_value_for_315n_divisible_by_12 :
  ∃ n : ℕ, (n < 10) ∧ ((10 + n) % 4 = 0) ∧ ((9 + n) % 3 = 0) ∧ n = 6 :=
by
  use 6
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end n_value_for_315n_divisible_by_12_l290_290305


namespace prob_9_in_decimal_rep_of_3_over_11_l290_290488

def decimal_rep_of_3_over_11 : List ℕ := [2, 7]  -- decimal representation of 3/11 is 0.272727...

theorem prob_9_in_decimal_rep_of_3_over_11 : 
  (1 / (2 : ℚ)) * (decimal_rep_of_3_over_11.count 9) = 0 := by
  have h : 9 ∉ decimal_rep_of_3_over_11 := by simp only [decimal_rep_of_3_over_11, List.mem_cons, List.mem_nil, not_false_iff]; exact dec_trivial
  rw List.count_eq_zero_of_not_mem h
  norm_num
  sorry

end prob_9_in_decimal_rep_of_3_over_11_l290_290488


namespace probability_white_or_red_l290_290202

theorem probability_white_or_red :
  let white_balls := 7
  let black_balls := 8
  let red_balls := 5
  let total_balls := white_balls + black_balls + red_balls
  let favorable_outcomes := white_balls + red_balls
  in favorable_outcomes / total_balls = 3 / 5 :=
by
  let white_balls := 7
  let black_balls := 8
  let red_balls := 5
  let total_balls := white_balls + black_balls + red_balls
  let favorable_outcomes := white_balls + red_balls
  show favorable_outcomes / total_balls = 3 / 5
  sorry

end probability_white_or_red_l290_290202


namespace sum_solutions_l290_290293

theorem sum_solutions (h : ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → (1 / Real.sin x + 1 / Real.cos x = 4)) :
  ∃ s, (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → (1 / Real.sin x + 1 / Real.cos x = 4) → s = x) ∧ s = 3 * Real.pi / 2 := 
begin
  sorry
end

end sum_solutions_l290_290293


namespace min_n_binomial_expansion_non_zero_constant_term_l290_290113

theorem min_n_binomial_expansion_non_zero_constant_term :
  ∃ (n : ℕ), (2 : ℤ)^4 * x^4^n - (1 : ℤ)/(3 * x^3)^n → 4*n - (7*r) = 0 → n ≥ 1 → n = 7 :=
by
  sorry

end min_n_binomial_expansion_non_zero_constant_term_l290_290113


namespace royal_children_count_l290_290624

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290624


namespace probability_not_snow_l290_290973

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l290_290973


namespace sin_half_pi_plus_alpha_l290_290337

-- Definitions based on the conditions
def point_P : ℝ × ℝ := (6, -8)
def length_OP : ℝ := real.sqrt (6^2 + (-8)^2)
def cos_alpha : ℝ := 6 / length_OP

-- The resulting theorem
theorem sin_half_pi_plus_alpha : sin (π / 2 + real.acos (cos_alpha)) = cos_alpha :=
by {
  -- TODO: provide proof
  sorry
}

end sin_half_pi_plus_alpha_l290_290337


namespace phase_and_initial_phase_theorem_l290_290964

open Real

noncomputable def phase_and_initial_phase (x : ℝ) : ℝ := 3 * sin (-x + π / 6)

theorem phase_and_initial_phase_theorem :
  ∃ φ : ℝ, ∃ ψ : ℝ,
    ∀ x : ℝ, phase_and_initial_phase x = 3 * sin (x + φ) ∧
    (φ = 5 * π / 6) ∧ (ψ = φ) :=
sorry

end phase_and_initial_phase_theorem_l290_290964


namespace water_remaining_in_canteen_l290_290364

theorem water_remaining_in_canteen
  (initial_water : ℕ)
  (leak_rate : ℕ)
  (time : ℕ)
  (drink_rate_mile_1_to_6 : ℕ)
  (drink_last_mile : ℕ)
  (miles_1_to_6 : ℕ)
  (total_miles : ℕ) :
  initial_water = 11 →
  leak_rate = 1 →
  time = 3 →
  drink_rate_mile_1_to_6 = 1 → -- Representing 0.5 as 1 cup per 2 miles for integer calcs
  drink_last_mile = 3 →
  miles_1_to_6 = 6 →
  total_miles = 7 →
  let water_drunk_first_6_miles := (drink_rate_mile_1_to_6 * miles_1_to_6) / 2 in
  let water_drunk_last_mile := drink_last_mile in
  let water_leaked := leak_rate * time in
  initial_water - (water_drunk_first_6_miles + water_drunk_last_mile + water_leaked) = 2 :=
begin
  intros,
  unfold let,
  calc initial_water - ((drink_rate_mile_1_to_6 * miles_1_to_6) / 2 + drink_last_mile + leak_rate * time)
    = 11 - ((1 * 6) / 2 + 3 + 1 * 3) : by simp [initial_water, drink_rate_mile_1_to_6, drink_last_mile, leak_rate, time]
    -- Further calculations and proof steps (to be done in Lean)
    ... = 2 : sorry
end

end water_remaining_in_canteen_l290_290364


namespace integral_eq_solution_l290_290712

noncomputable def indefiniteIntegral : ∫ (x^3 - 5 * x^2 + 5 * x + 23) / ((x - 1) * (x + 1) * (x - 5)) := sorry

theorem integral_eq_solution :
  indefiniteIntegral = x - 3 * log (abs (x - 1)) + log (abs (x + 1)) + 2 * log (abs (x - 5)) + C :=
sorry

end integral_eq_solution_l290_290712


namespace incorrect_statementD_l290_290578

-- Definitions based on conditions in the problem
def conditionA : Prop :=
  ∀ (c_CH3COOH c_CH3COO : ℝ), (c_CH3COOH + c_CH3COO = 0.1)

def conditionB : Prop :=
  ∀ (c_OH c_H c_HS c_H2S : ℝ), (c_OH = c_H + c_HS + 2 * c_H2S)

def conditionC : Prop :=
  ∀ (FeCl3 HydrochloricAcid : Type), (hydrolyzes FeCl3 → HydrochloricAcid ≠ 0)

def conditionD : Prop :=
  ∀ (c : ℝ), heating_burning (Al2_SO4_3 + 6 * H2O) → Al2_O3

-- Incorporating the correct answer
def incorrect_statement : Prop := 
  ∀ (c : ℝ), heating_burning (Al2_SO4_3 + 6 * H2O) → Al2_SO4_3

-- The theorem to prove
theorem incorrect_statementD (condA : conditionA)
                             (condB : conditionB)
                             (condC : conditionC)
                             (condD : conditionD) :
  incorrect_statement := 
  sorry

end incorrect_statementD_l290_290578


namespace plane_equation_through_point_and_parallel_l290_290762

theorem plane_equation_through_point_and_parallel (P : ℝ × ℝ × ℝ) (D : ℝ)
  (normal_vector : ℝ × ℝ × ℝ) (A B C : ℝ)
  (h1 : normal_vector = (2, -1, 3))
  (h2 : P = (2, 3, -1))
  (h3 : A = 2) (h4 : B = -1) (h5 : C = 3)
  (hD : A * 2 + B * 3 + C * -1 + D = 0) :
  A * x + B * y + C * z + D = 0 :=
by
  sorry

end plane_equation_through_point_and_parallel_l290_290762


namespace region_geometry_l290_290974

/--
Given a line segment CD in three-dimensional space, all points within 4 units of this segment form a cylindrical shape with hemispheres capping each end. 

If the total surface area of this shape (including the curved surface of the cylinder and the outer surfaces of the hemispheres) is 400π square units, then the length of CD is 42 units.

Additionally, the total volume of the region thus formed is 806π cubic units.
-/
theorem region_geometry
  (r : ℝ) (L : ℝ) (S : ℝ)
  (h1 : r = 4)
  (h2 : S = 400 * real.pi)
  (h3 : 8 * real.pi * L + 64 * real.pi = S) :
  L = 42 ∧ (real.pi * r^2 * L + 2 * (2/3) * real.pi * r^3 = 806 * real.pi) :=
by
  sorry

end region_geometry_l290_290974


namespace geometric_sequence_ratio_l290_290313

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (r : ℤ) (h1 : r = -2) (h2 : ∀ n, S n = a1 * (1 - r ^ n) / (1 - r)) :
  S 4 / S 2 = 5 :=
by
  -- Placeholder for proof steps
  sorry

end geometric_sequence_ratio_l290_290313


namespace problem_statement_l290_290423

open Real

-- Conditions
def line_eq (x y : ℝ) : Prop := y = 2 * x + sqrt 5
def polar_eq (rho theta : ℝ) : Prop := rho^2 * cos (2 * theta) + 4 = 0
def curve_eq (x y : ℝ) : Prop := y^2 - x^2 = 4
def point_A : (ℝ × ℝ) := (0, sqrt 5)

-- Proof problem
theorem problem_statement :
  (∀ x y : ℝ, polar_eq (sqrt (x^2 + y^2)) (atan2 y x) ↔ curve_eq x y) ∧
  (∀ (M N : ℝ × ℝ), line_eq M.1 M.2 ∧ line_eq N.1 N.2 ∧ curve_eq M.1 M.2 ∧ curve_eq N.1 N.2 →
    let AM := dist point_A M in
    let AN := dist point_A N in
    1 / |AM| + 1 / |AN| = 4) :=
by sorry

end problem_statement_l290_290423


namespace math_problem_l290_290537

open Real

variable (x : ℝ)
variable (h : x + 1 / x = sqrt 3)

theorem math_problem : x^7 - 3 * x^5 + x^2 = -5 * x + 4 * sqrt 3 :=
by sorry

end math_problem_l290_290537


namespace fraction_of_boys_among_attendees_l290_290930

def boys : ℕ := sorry
def girls : ℕ := boys
def teachers : ℕ := boys / 2

def boys_attending : ℕ := (4 * boys) / 5
def girls_attending : ℕ := girls / 2
def teachers_attending : ℕ := teachers / 10

theorem fraction_of_boys_among_attendees :
  (boys_attending : ℚ) / (boys_attending + girls_attending + teachers_attending) = 16 / 27 := sorry

end fraction_of_boys_among_attendees_l290_290930


namespace boat_upstream_time_is_1_5_hours_l290_290195

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l290_290195


namespace line_passes_through_fixed_point_l290_290355

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y = k * x + 2 * k + 1 :=
by
  sorry

end line_passes_through_fixed_point_l290_290355


namespace product_mod_5_l290_290290

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l290_290290


namespace sixth_smallest_number_l290_290765

def valid_digits : Finset ℕ := {0, 4, 6, 7, 8}

def is_valid_number (n : ℕ) : Prop :=
let digits := n.digits 10 in digits.to_finset ⊆ valid_digits ∧ digits.length = 5 ∧ 0 ∉ digits.reverse.tail

def sixth_smallest : ℕ := 40876

theorem sixth_smallest_number : ∃ n, is_valid_number n ∧ (finset.sort (λ x y, x < y) {n | is_valid_number n}).nth 5 = some sixth_smallest :=
sorry

end sixth_smallest_number_l290_290765


namespace translate_sine_curve_left_l290_290797

theorem translate_sine_curve_left (x : ℝ) : 
  (sin 2 x) = (sin (2 (x + π / 8))) ↔ (sin (2 x + π / 4)) := sorry

end translate_sine_curve_left_l290_290797


namespace sum_neg_one_binom_eq_l290_290956

theorem sum_neg_one_binom_eq (m n : ℕ) : 
  (∑ k in finset.range (n+1), if k > 0 then (-1)^k * (nat.choose n k) * (nat.choose m (n-k)) else 0) = 
  if m = n then 1 else 0 :=
sorry

end sum_neg_one_binom_eq_l290_290956


namespace three_digit_integers_l290_290380

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≠ 4

def contains_two (n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def does_not_contain_four (n : ℕ) : Prop :=
  ∀ k, n / 10^k % 10 ≠ 4

theorem three_digit_integers : {n : ℕ | is_three_digit n ∧ contains_two n ∧ does_not_contain_four n}.card = 200 := by
  sorry

end three_digit_integers_l290_290380


namespace product_mod_5_l290_290288

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l290_290288


namespace determine_cd_l290_290126

theorem determine_cd (c d : ℝ) 
  (h : (fin.mk 3 ⟨3, c, -7⟩) × (fin.mk 3 ⟨9, 4, d⟩) = 0) : 
  c = 4 / 3 ∧ d = -21 :=
by {
  sorry
}

end determine_cd_l290_290126


namespace union_of_sets_l290_290049

def M : Set ℝ := {x | x^2 + 2 * x = 0}

def N : Set ℝ := {x | x^2 - 2 * x = 0}

theorem union_of_sets : M ∪ N = {x | x = -2 ∨ x = 0 ∨ x = 2} := sorry

end union_of_sets_l290_290049


namespace directly_proportional_function_l290_290701

-- Definitions for functions
def fA (x : ℝ) : ℝ := -0.1 * x
def fB (x : ℝ) : ℝ := 2 * x^2
def fC (x : ℝ) : ℝ := (4 * x)^(1/2)
def fD (x : ℝ) : ℝ := 2 * x + 1

-- Assumption that y is directly proportional to x
def directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

theorem directly_proportional_function :
  directly_proportional fA :=
by
  sorry

end directly_proportional_function_l290_290701


namespace total_mile_times_l290_290151

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l290_290151


namespace seating_arrangements_l290_290752

/-- Define the problem setup with conditions -/
def is_valid_arrangement (seating : Fin 8 → Fin 4 × Bool) : Prop :=
  ∀ i : Fin 8, 
    seating (i + 1) % 2 = seating i % 2 
    ∧ (seating (i + 1)).fst ≠ (seating i).fst
    ∧ (seating (i + 4)).fst ≠ (seating i).fst

theorem seating_arrangements : ∃ arrangements : Finset (Fin 8 → Fin 4 × Bool),
  ∀ f ∈ arrangements, is_valid_arrangement f ∧ arrangements.card = 48 :=
sorry

end seating_arrangements_l290_290752


namespace expression_evaluation_l290_290255

theorem expression_evaluation : (2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1) := 
by 
  sorry

end expression_evaluation_l290_290255


namespace johns_fifth_race_time_l290_290039

theorem johns_fifth_race_time (a b c d : ℕ) (h1 : a = 95) (h2 : b = 102) (h3 : c = 107) (h4 : d = 110) (e : ℕ) (h5 : e = 103) :
  median_of_list [a, b, c, d, e] = 103 :=
by
  -- skip the proof
  sorry

end johns_fifth_race_time_l290_290039


namespace defective_units_shipped_for_sale_l290_290586

theorem defective_units_shipped_for_sale (d p : ℝ) (h1 : d = 0.09) (h2 : p = 0.04) : (d * p * 100 = 0.36) :=
by 
  -- Assuming some calculation steps 
  sorry

end defective_units_shipped_for_sale_l290_290586


namespace exists_distinct_abc_sum_l290_290910

theorem exists_distinct_abc_sum (n : ℕ) (h : n ≥ 1) (X : Finset ℤ)
  (h_card : X.card = n + 2)
  (h_abs : ∀ x ∈ X, abs x ≤ n) :
  ∃ (a b c : ℤ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end exists_distinct_abc_sum_l290_290910


namespace train_length_is_correct_l290_290685

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l290_290685


namespace sum_inequality_l290_290067

-- Define the conditions for the problem
variables {n : ℕ} {x : ℕ → ℝ}
hypothesis h1 : n ≥ 2
hypothesis h2 : ∀ j : ℕ, 1 ≤ j ∧ j ≤ n → x j > -1
hypothesis h3 : (∑ j in Finset.range n, x j) = n

-- Statement to prove
theorem sum_inequality :
  (∑ j in Finset.range n, 1 / (1 + x j)) ≥ (∑ j in Finset.range n, x j / (1 + (x j) ^ 2)) :=
sorry

end sum_inequality_l290_290067


namespace largest_alpha_and_sequence_l290_290464

noncomputable def sequence_x (N : ℕ) (α : ℝ) : ℕ → ℝ
| 0     := 0
| 1     := 1
| (k+2) := (α * (sequence_x N α (k+1)) - (N - k) * (sequence_x N α k)) / (k + 1)

theorem largest_alpha_and_sequence (N : ℕ) (hN : 0 < N) :
  ∃α : ℝ, (α = N - 1) ∧ (sequence_x N α (N + 1) = 0) ∧
    (∀ k : ℕ, k ≤ N → sequence_x N α k = if k = 0 then 0 else if k = 1 then 1 else real.binom (N-1) (k-1)) ∧
    (∀ k : ℕ, k > N → sequence_x N α k = 0) :=
by
  let α := (N - 1 : ℝ)
  use α
  sorry

end largest_alpha_and_sequence_l290_290464


namespace proj_b_l290_290891

open Matrix Real

-- Definition of orthogonality
def orthogonal (a b : Vector ℝ 2) : Prop :=
  dot_product a b = 0

-- Projections
def proj (u v : Vector ℝ 2) : Vector ℝ 2 :=
  (dot_product u v / dot_product u u) • u

theorem proj_b (a b v : Vector ℝ 2) 
    (h_orthog : orthogonal a b)
    (h_proj_a : proj a v = ⟨[-4/5, -8/5]⟩) :
    proj b v = ⟨[24/5, -12/5]⟩ := 
by 
  sorry

end proj_b_l290_290891


namespace find_x_l290_290032

variables (A B C D X : Prop)
variables (angle_AXB angle_BAX angle_ABX angle_CYX x : ℝ)

def given_conditions : Prop :=
  angle_AXB = 180 ∧ AB && CD ∧ angle_BAX = 55 ∧ angle_ABX = 65

theorem find_x (h : given_conditions) : x = 75 :=
sorry

end find_x_l290_290032


namespace age_ratio_in_years_l290_290564

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end age_ratio_in_years_l290_290564


namespace age_ratio_in_4_years_l290_290562

-- Definitions based on the conditions
def pete_age (years_ago : ℕ) (p : ℕ) (c : ℕ) : Prop :=
match years_ago with
  | 2 => p - 2 = 3 * (c - 2)
  | 4 => p - 4 = 4 * (c - 4)
  | _ => true
end

-- Question: In how many years will the ratio of their ages be 2:1?
def age_ratio (years : ℕ) (p : ℕ) (c : ℕ) : Prop :=
(p + years) / (c + years) = 2

-- Proof problem
theorem age_ratio_in_4_years {p c : ℕ} (h1 : pete_age 2 p c) (h2 : pete_age 4 p c) : 
  age_ratio 4 p c :=
sorry

end age_ratio_in_4_years_l290_290562


namespace bananas_bought_l290_290111

theorem bananas_bought (O P B : Nat) (x : Nat) 
  (h1 : P - O = B)
  (h2 : O + P = 120)
  (h3 : P = 90)
  (h4 : 60 * x + 30 * (2 * x) = 24000) : 
  x = 200 := by
  sorry

end bananas_bought_l290_290111


namespace sequence_equality_proof_l290_290907

def a_sequence : ℕ → ℝ
| 1 := 0.202
| 2 := (0.2021) ^ (a_sequence 1)
| (k + 1) := if k % 2 = 1 
             then (0 + 0.202 * 10 ^ (-k - 3)) ^ (a_sequence k)  -- Alternating base sequence
             else (0 + 0.202 * 10 ^ (-k - 2)) ^ (a_sequence k)  -- Alternating base sequence

def b_sequence : list ℝ := list.sort (≥) (list.of_fn (fun n => a_sequence (n + 1)))

noncomputable def indexes_sum : ℕ :=
(list.range 101).filter (λ k => (a_sequence (k + 1)) = (b_sequence.nth_le k (sorry))).sum

theorem sequence_equality_proof :
  indexes_sum = 2550 :=
sorry

end sequence_equality_proof_l290_290907


namespace correct_propositions_count_l290_290785

def α : Type := Plane
def β : Type := Plane
def m : Type := Line
def n : Type := Line

variables (α β : Plane) (m n : Line)

-- Conditions related to each proposition
def proposition_1 : Prop := α ⊓ β = ∅ ∧ m ∈ α → m ⊓ α = ∅
def proposition_2 : Prop := m ∈ α ∧ n ∈ α ∧ m ∥ β ∧ n ∥ β → α ∥ β
def proposition_3 : Prop := m ∈ α ∧ n ∉ α ∧ m.skew n → (n ⊓ α ≠ ∅ ∨ n ∥ α)
def proposition_4 : Prop := α ⊓ β = m ∧ (n ∥ m ∧ n ∉ α ∧ n ∉ β) → (n ∥ α ∧ n ∥ β)

-- The Lean statement that verifies the number of correct propositions
theorem correct_propositions_count :
  (proposition_1 α β m n ∧ proposition_2 α β m n ∧ proposition_3 α β m n ∧ proposition_4 α β m n) → 1 := 
sorry

end correct_propositions_count_l290_290785


namespace evaluate_expression_at_values_l290_290942

theorem evaluate_expression_at_values (x y : ℤ) (h₁ : x = 1) (h₂ : y = -2) :
  (-2 * x ^ 2 + 2 * x - y) = 2 :=
by
  subst h₁
  subst h₂
  sorry

end evaluate_expression_at_values_l290_290942


namespace sin_A_value_c_value_l290_290403

variables (A B C : ℝ) (a b c : ℝ)

-- Given conditions
def triangle_ABC : Prop :=
  C = 2 * Real.pi / 3 ∧ a = 6

-- Proof of part 1
theorem sin_A_value (h : triangle_ABC) (h1 : c = 14) : 
  sin A = 3 / 14 * Real.sqrt 3 :=
sorry

-- Given condition for part 2
def area_ABC (S : ℝ) : Prop :=
  S = 3 * Real.sqrt 3

-- Proof of part 2
theorem c_value (h : triangle_ABC) (h2 : area_ABC (1/2 * a * b * sin C)) : 
  c = 2 * Real.sqrt 13 :=
sorry

end sin_A_value_c_value_l290_290403


namespace dennis_floor_l290_290732

theorem dennis_floor :
  ∀ (Dennis Charlie Frank : ℕ), 
    Frank = 16 →
    Charlie = Frank / 4 →
    Dennis = Charlie + 2 →
    Dennis = 6 :=
by
  intro Dennis Charlie Frank
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end dennis_floor_l290_290732


namespace royal_children_l290_290616

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290616


namespace infinite_solutions_natural_numbers_l290_290122

theorem infinite_solutions_natural_numbers (x y : ℕ) (h : (x = 3 ∧ y = 2 ∧ (x + y * real.sqrt 2) * (x - y * real.sqrt 2) = 1)):
  ∃ (inf_in_ns: ∀ n : ℕ, ∃ a b : ℕ, (a + b * real.sqrt 2) * (a - b * real.sqrt 2) = 1) :=
begin
  sorry
end

end infinite_solutions_natural_numbers_l290_290122


namespace max_black_cells_in_5x5_square_l290_290404

def max_black_cells (n : ℕ) : ℕ :=
  9  -- The predefined result for the 5x5 grid problem

theorem max_black_cells_in_5x5_square :
  ∀ (grid : fin 5 × fin 5 → bool), 
    (∀ x y, grid x = tt → grid y = tt → ∃ z, grid z = ff ∧ (z = x ∨ z = y)) →
    (∑ x, if grid x then 1 else 0 ≤ max_black_cells 5) :=
by
  sorry

end max_black_cells_in_5x5_square_l290_290404


namespace range_of_q_eq_eight_inf_l290_290250

noncomputable def q (x : ℝ) : ℝ := (x^2 + 2)^3

theorem range_of_q_eq_eight_inf (x : ℝ) : 0 ≤ x → ∃ y, y = q x ∧ 8 ≤ y := sorry

end range_of_q_eq_eight_inf_l290_290250


namespace proof_geometric_models_l290_290702

def interval := set.Icc (-5 : ℝ) 5
def abs_interval := set.Icc (-1 : ℝ) 1
def integer_interval := set.Icc (-5 : ℤ) 5
def square_side := 5
def circle_radius := 1

def is_geometric_probability_model (model : ℕ) : Prop :=
  match model with
  | 1 => set.infinite interval
  | 2 => set.infinite interval ∧ set.infinite abs_interval
  | 3 => ¬ set.infinite integer_interval
  | 4 => set.infinite (set.Icc (-square_side/2 : ℝ) (square_side/2)) ∧ set.infinite (set.Icc (-(circle_radius : ℝ)) (circle_radius))
  | _ => false

theorem proof_geometric_models (models : list ℕ) :
  (models = [1, 2, 4]) ↔
  (∀ model ∈ models, is_geometric_probability_model model) :=
by
  sorry

end proof_geometric_models_l290_290702


namespace exists_special_number_l290_290914

theorem exists_special_number :
  ∃ N : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ 149 → k ∣ N) ∨ (k + 1 ∣ N) = false) :=
sorry

end exists_special_number_l290_290914


namespace extremum_g_range_m_l290_290814

noncomputable def f (x m a : ℝ) := m * x - a * Real.log x - m
def g (x : ℝ) := Real.exp x / Real.exp x

theorem extremum_g : ∀ x, g(x) ≤ 1 := sorry

theorem range_m 
  (a : ℝ) (h : a = 2) 
  (t1 t2 x0 : ℝ) (hx0 : 0 < x0) (hx0_upper : x0 ≤ Real.exp 1)
  (ht1 : 0 < t1) (ht1_upper : t1 ≤ Real.exp 1)
  (ht2 : 0 < t2) (ht2_upper : t2 ≤ Real.exp 1) (ht1_ne_ht2 : t1 ≠ t2)
  (h_eq : f t1 1 a = f t2 1 a = g x0) : 
  ∃ m, m ≥ 3 / (Real.exp 1 - 1) := sorry

end extremum_g_range_m_l290_290814


namespace range_of_a_l290_290071

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) (h1 : 2 ≤ a) (h2 : (∀ x, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11)) : 4 ≤ a ∧ a ≤ 8 := by
  sorry

end range_of_a_l290_290071


namespace freshPermCount_ge_l290_290668

def isFreshPermutation (m : ℕ) (perm : List ℕ) : Prop :=
  ∀ k, k < m → ¬(perm.take k = List.range k)

def freshPermCount (m : ℕ) : ℕ :=
  List.permutations (List.range m).count (isFreshPermutation m)

theorem freshPermCount_ge (n : ℕ) (h : n ≥ 3) : freshPermCount n ≥ n * freshPermCount (n - 1) := 
by
  sorry

end freshPermCount_ge_l290_290668


namespace staffing_arrangements_count_l290_290221

theorem staffing_arrangements_count :
  let teachers := {A, B, C, D, E}
  let schools := {s1, s2, s3}
  (∀ t ∈ teachers, ∃ s ∈ schools, t ∈ s) ∧ -- Each school gets at least one teacher
  (A ∈ s1 → B ∈ s1) ∧ (A ∈ s1 → C ∉ s1) ∧ -- A and B together, A and C separate
  (A ∈ s2 → B ∈ s2) ∧ (A ∈ s2 → C ∉ s2) ∧
  (A ∈ s3 → B ∈ s3) ∧ (A ∈ s3 → C ∉ s3) → 
  ∃ n : ℕ, n = 30 :=
by sorry

end staffing_arrangements_count_l290_290221


namespace arithmetic_sequence_num_terms_l290_290977

theorem arithmetic_sequence_num_terms (a_1 d S_n n : ℕ) 
  (h1 : a_1 = 4) (h2 : d = 3) (h3 : S_n = 650)
  (h4 : S_n = (n / 2) * (2 * a_1 + (n - 1) * d)) : n = 20 := by
  sorry

end arithmetic_sequence_num_terms_l290_290977


namespace train_length_is_correct_l290_290687

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l290_290687


namespace range_of_a_l290_290820

theorem range_of_a (M N : Set ℝ) (a : ℝ) 
(hM : M = {x : ℝ | x < 2}) 
(hN : N = {x : ℝ | x < a}) 
(hSubset : M ⊆ N) : 
  2 ≤ a := 
sorry

end range_of_a_l290_290820


namespace problem_statement_l290_290348

open Real

noncomputable def f (ω x : ℝ) : ℝ := (sin (ω * x) * cos (ω * x) - cos (ω * x) ^ 2)

theorem problem_statement (ω : ℝ) (ω_pos : ω > 0) :
  (∃ T > 0, ∀ x, f ω (x + T) = f ω x) → ω = 2 ∧ 
  ∀ (a b c : ℝ) (h : b^2 = a * c) (x : angle B),
    sin (4 * x - π / 2) - 1 ∈ set.Icc (-2 : ℝ) 0  :=
by
  sorry

end problem_statement_l290_290348


namespace sum_F_1_to_1024_l290_290055

def F (m : ℕ) : ℕ := int.natAbs (⌊real.log2 (m : ℝ)⌋)

theorem sum_F_1_to_1024 : (∑ m in Finset.range 1024 \ quast-F.one
, F (m + 1)) = 8192 := sorry with sorry 

end sum_F_1_to_1024_l290_290055


namespace ellipse_focal_length_l290_290344

theorem ellipse_focal_length (k : ℝ) :
  (∀ x y : ℝ, x^2 / k + y^2 / 2 = 1) →
  (∃ c : ℝ, 2 * c = 2 ∧ (k = 1 ∨ k = 3)) :=
by
  -- Given condition: equation of ellipse and focal length  
  intro h  
  sorry

end ellipse_focal_length_l290_290344


namespace solve_for_x_l290_290388

theorem solve_for_x (x : ℝ) (h :  9 / x^2 = x / 25) : x = 5 :=
by 
  sorry

end solve_for_x_l290_290388


namespace maximal_altitudes_product_l290_290723

theorem maximal_altitudes_product {A B C : Type*}
  (has_fixed_base : Segment AB) -- representation of fixed base
  (fixed_height : Altitude h from C to AB) -- representation of fixed altitude
  : ∃ (ABC : Triangle) (max_condition : (ABC.is_isosceles ∨ ∠ ACB = π/2)), 
    ∀ (t : Triangle) (same_base : t.base = has_fixed_base)
      (same_alt : t.altitude_from_C = fixed_height), 
      product_of_altitudes t ≤ product_of_altitudes ABC :=
sorry

end maximal_altitudes_product_l290_290723


namespace angle_BAC_measure_l290_290426

-- Lean 4 Statement for the given problem
theorem angle_BAC_measure {A B C X Z Y : Type} [IsTriangle A B C] 
    (h1 : PointOnLineSegment X A B) (h2 : PointOnLineSegment Z A X) (h3 : PointOnLineSegment Z X Y) 
    (h4 : PointOnLineSegment Y X B) (h5 : PointOnLineSegment Y B Z) (h6 : PointOnLineSegment Z B C)
    (h7 : PointOnLineSegment Z C Y) (h8 : MeasureAngle B A C = 140) 
    (AX_eq_XZ : AX = XZ) (XZ_eq_ZY : XZ = ZY) (ZY_eq_YB : ZY = YB) 
    (BZ_eq_ZC : BZ = ZC) (ZC_eq_CY : ZC = CY) : 
    ∃ t : ℝ, MeasureAngle B A C = t ∧ t = 40 := 
by 
  sorry

end angle_BAC_measure_l290_290426


namespace select_rows_and_columns_l290_290786

theorem select_rows_and_columns (n : Nat) (pieces : Fin (2 * n) × Fin (2 * n) → Bool) :
  (∃ rows cols : Finset (Fin (2 * n)),
    rows.card = n ∧ cols.card = n ∧
    (∀ r c, r ∈ rows → c ∈ cols → pieces (r, c))) :=
sorry

end select_rows_and_columns_l290_290786


namespace total_distance_crawled_l290_290224

theorem total_distance_crawled :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let pos4 := 0
  abs (pos2 - pos1) + abs (pos3 - pos2) + abs (pos4 - pos3) = 29 :=
by
  sorry

end total_distance_crawled_l290_290224


namespace triangle_inequality_example_l290_290865

theorem triangle_inequality_example :
  ∀ (y : ℚ), (9 / 4 < y ∧ y < 5) → (5 - 9 / 4 = 11 / 4) :=
by {
  assume y hyp,
  have h1 : 5 - 9 / 4 = (20 / 4) - (9 / 4), from sorry,
  rw h1,
  show 11 / 4 = 11 / 4, from eq.refl (11 / 4),
  sorry
}

end triangle_inequality_example_l290_290865


namespace find_d1_l290_290903

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  (∃ d_3 d_2 d_1 d_0 : ℤ, ∀ n : ℕ, n ≥ 7 → n % 2 = 1 → E(n) = d_3 * n^3 + d_2 * n^2 + d_1 * n + d_0) → d_1 = 6 :=
sorry

end find_d1_l290_290903


namespace roots_of_polynomial_l290_290266

noncomputable def p (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial : {x : ℝ | p x = 0} = {1, -1, 3} :=
by
  sorry

end roots_of_polynomial_l290_290266


namespace royal_children_count_l290_290608

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l290_290608


namespace parallelogram_sides_and_diagonals_l290_290217

theorem parallelogram_sides_and_diagonals
  (AE EC: ℝ) 
  (BC_diff_AB BC: ℝ) 
  (BC EC: ℝ)
  (_0 : AE = 6)
  (_1 : EC = 15)
  (_2 : BC - AB = 7) :
  (AB = 10 ∧ BC = 17 ∧ AC = 21 ∧ BD = real.sqrt 337) :=
begin
  sorry
end

end parallelogram_sides_and_diagonals_l290_290217


namespace set_of_m_values_l290_290835

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {y | m * y + 2 = 0}

theorem set_of_m_values (m : ℝ) :
  (A ∪ B m = A) ↔ m ∈ {0, -1, -2/3} := 
by 
  sorry

end set_of_m_values_l290_290835


namespace train_length_correct_l290_290681

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l290_290681


namespace sufficient_but_not_necessary_condition_l290_290302

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 1) (hy : y > 2) : (x + y > 3) :=
by {
  have h1 : 1 < x := hx,
  have h2 : 2 < y := hy,
  have h : x + y > 1 + 2 := add_lt_add h1 h2,
  exact h
}

end sufficient_but_not_necessary_condition_l290_290302


namespace ratio_of_sums_l290_290503

open Nat

def sum_multiples_of_3 (n : Nat) : Nat :=
  let m := n / 3
  m * (3 + 3 * m) / 2

def sum_first_n_integers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem ratio_of_sums :
  (sum_multiples_of_3 600) / (sum_first_n_integers 300) = 4 / 3 :=
by
  sorry

end ratio_of_sums_l290_290503


namespace propositions_count_correct_l290_290233

theorem propositions_count_correct :
  (∀ (a b : ℝ → ℝ) (h1 : |a| = 1) (h2 : b ∥ a) (h3: |b| = 1), a ≠ b) ∧
  (∀ (k : ℝ), k * 0 = (0 : ℝ)) ∧
  (∀ (a b : ℝ → ℝ) (h : b ∥ a), |b| ≠ |a|) ∧ 
  (∀ (k : ℝ) (a : ℝ → ℝ), k * a = (0 : ℝ) → k = 0) ∧
  (∀ (a : ℝ → ℝ) (h : |a| = 0), a = (0 : ℝ → ℝ)) :=
  sorry

end propositions_count_correct_l290_290233


namespace monotonically_decreasing_interval_l290_290836

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x - 3 * Real.log x

theorem monotonically_decreasing_interval :
  ∀ x, 0 < x ∧ x < 3 → f' x < 0 :=
by
  sorry

end monotonically_decreasing_interval_l290_290836


namespace greater_number_is_84_l290_290140

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : x + y - (x - y) = 64) :
  x = 84 :=
by sorry

end greater_number_is_84_l290_290140


namespace royal_children_count_l290_290625

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l290_290625


namespace inequality_solution_range_l290_290353

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ≥ 2 :=
by
  sorry

end inequality_solution_range_l290_290353


namespace sum_of_complex_numbers_on_real_axis_l290_290784

theorem sum_of_complex_numbers_on_real_axis (a : ℝ) (z1 z2 : ℂ) (h1 : z1 = 2 + complex.i) (h2 : z2 = 3 + a * complex.i) :
  (z1 + z2).im = 0 → a = -1 :=
by
  sorry

end sum_of_complex_numbers_on_real_axis_l290_290784


namespace distance_from_goal_after_4_weeks_distance_from_goal_l290_290248

variable (P : ℝ)

def daily_steps (initial_steps : ℝ) (weekly_increase : ℝ) (week : ℕ) : ℝ :=
  initial_steps * (1 + weekly_increase) ^ week

def total_steps_over_weeks (initial_steps : ℝ) (weekly_increase : ℝ) (weeks : ℕ) : ℝ :=
  (List.range weeks).sum (λ week, daily_steps initial_steps weekly_increase week)

theorem distance_from_goal_after_4_weeks (initial_steps : ℝ) (weekly_increase : ℝ) (target_steps : ℝ) :
  total_steps_over_weeks initial_steps weekly_increase 4 = 7 * initial_steps * (1 + (1 + weekly_increase) + (1 + weekly_increase)^2 + (1 + weekly_increase)^3) :=
by
  sorry

theorem distance_from_goal (initial_steps weekly_increase : ℝ) : 
  distance_from_goal_after_4_weeks initial_steps weekly_increase 100000 = 100000 - 7 * initial_steps * (1 + (1 + weekly_increase) + (1 + weekly_increase)^2 + (1 + weekly_increase)^3) :=
by
  sorry

end distance_from_goal_after_4_weeks_distance_from_goal_l290_290248


namespace total_dance_hours_l290_290557

theorem total_dance_hours (t : ℕ) (h : ℕ) (w : ℕ) (y : ℕ) (weeks_in_year : ℕ) (num_years : ℕ)
  (Ht : t = 4) (Hh : h = 2) (Hw : weeks_in_year = 52) (Hy : num_years = 10) :
  t * h * weeks_in_year * num_years = 4160 := by
  rw [Ht, Hh, Hw, Hy]
  norm_num
  sorry

end total_dance_hours_l290_290557


namespace jeremy_can_win_in_4_turns_l290_290433

noncomputable def game_winnable_in_4_turns (left right : ℕ) : Prop :=
∃ n1 n2 n3 n4 : ℕ,
  n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧
  (left + n1 + n2 + n3 + n4 = right * n1 * n2 * n3 * n4)

theorem jeremy_can_win_in_4_turns (left right : ℕ) (hleft : left = 17) (hright : right = 5) : game_winnable_in_4_turns left right :=
by
  rw [hleft, hright]
  sorry

end jeremy_can_win_in_4_turns_l290_290433


namespace linear_function_quadrants_passing_through_l290_290523

theorem linear_function_quadrants_passing_through :
  ∀ (x : ℝ) (y : ℝ), (y = 2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end linear_function_quadrants_passing_through_l290_290523


namespace repeatable_triangle_process_iff_equilateral_l290_290793

theorem repeatable_triangle_process_iff_equilateral {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (∀ n : ℕ, let s := (a + b + c) / 2 in 
    0 < s - a ∧ 0 < s - b ∧ 0 < s - c ∧ 
    (s - a) + (s - b) > (s - c) ∧ 
    (s - a) + (s - c) > (s - b) ∧ 
    (s - b) + (s - c) > (s - a)) ↔ a = b ∧ b = c := 
begin
  sorry
end

end repeatable_triangle_process_iff_equilateral_l290_290793


namespace probability_of_region_l290_290669

def within_bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5
def region (x y : ℝ) : Prop := x + y ≤ 6

theorem probability_of_region : 
  (∃ (x y : ℕ), within_bounds x y ∧ region x y) / (∃ (x y : ℕ), within_bounds x y) = 3/4 := 
  sorry

end probability_of_region_l290_290669


namespace ay_bz_cx_lt_S_squared_l290_290465

theorem ay_bz_cx_lt_S_squared 
  (S : ℝ) (a b c x y z : ℝ) 
  (hS : 0 < S) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a + x = S) 
  (h2 : b + y = S) 
  (h3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := 
sorry

end ay_bz_cx_lt_S_squared_l290_290465


namespace complementary_angle_ratio_l290_290991

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l290_290991


namespace max_wx_xy_yz_zt_l290_290058

theorem max_wx_xy_yz_zt {w x y z t : ℕ} (h_sum : w + x + y + z + t = 120)
  (hnn_w : 0 ≤ w) (hnn_x : 0 ≤ x) (hnn_y : 0 ≤ y) (hnn_z : 0 ≤ z) (hnn_t : 0 ≤ t) :
  wx + xy + yz + zt ≤ 3600 := 
sorry

end max_wx_xy_yz_zt_l290_290058


namespace julias_total_spending_l290_290874

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l290_290874


namespace pairs_sums_prime_1_to_10_l290_290584

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ k : ℕ, k > 1 → k < n → n % k ≠ 0

def pair_sums_prime (s : list (ℕ × ℕ)) : Prop :=
  (∀ (a, b) ∈ s, is_prime (a + b)) ∧ s.length = 5 ∧ ∀ (a, b) ∈ s, a ≠ b ∧ a ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ b ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧
  (∀ (a₁, b₁) (a₂, b₂) ∈ s, a₁ + b₁ ≠ a₂ + b₂ ∨ (a₁ = a₂ ∧ b₁ = b₂))

theorem pairs_sums_prime_1_to_10 :
  ∃ (s : list (ℕ × ℕ)), pair_sums_prime s :=
by
  sorry

end pairs_sums_prime_1_to_10_l290_290584


namespace semicircular_paper_cone_surface_area_l290_290674

noncomputable def cone_surface_area (r : ℝ) : ℝ :=
  let base_radius := r * π / (2 * π)
  let base_area := real.pi * base_radius^2
  let lateral_surface_area := 0.5 * (2 * real.pi * base_radius) * r
  base_area + lateral_surface_area

theorem semicircular_paper_cone_surface_area :
  cone_surface_area 10 = 75 * real.pi :=
sorry

end semicircular_paper_cone_surface_area_l290_290674


namespace find_f_of_3_l290_290473

theorem find_f_of_3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * f y - y) = x * y - f y) 
  (h2 : f 0 = 0) (h3 : ∀ x : ℝ, f (-x) = -f x) : f 3 = 3 :=
sorry

end find_f_of_3_l290_290473


namespace total_games_played_l290_290748

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l290_290748


namespace triangle_angle_inequality_l290_290382

theorem triangle_angle_inequality
  (x y z x_1 y_1 z_1 : Real)
  (hx : x + y + z = Real.pi)
  (hx1 : x_1 + y_1 + z_1 = Real.pi) :
  (Real.cos x_1 / Real.sin x) + 
  (Real.cos y_1 / Real.sin y) + 
  (Real.cos z_1 / Real.sin z) ≤ 
  (Real.cot x) + 
  (Real.cot y) + 
  (Real.cot z) :=
by
  sorry

end triangle_angle_inequality_l290_290382


namespace upstream_travel_time_l290_290200

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l290_290200


namespace find_starting_positions_l290_290892

noncomputable def hyperbola := {p : ℝ × ℝ // p.2 ^ 2 - p.1 ^ 2 = 1}

def P0_P2008_condition (P : ℝ) (n : ℕ) : Prop :=
  ∃ x0 : ℝ,
    P = x0 ∧
    (∀ k : ℕ, k < n → P_seq k = orthogonal_projection ((line_through P_seq k).intersection hyperbola)),
    P_seq n = P

theorem find_starting_positions :
  ∃ count : ℕ,
    count = 2 ^ 2008 - 2 ∧
    (∀ P : ℝ, P0_P2008_condition P 2008 → P ∈ set.range (λ k, (k * π) / (2 ^ 2008 - 1))) 

end find_starting_positions_l290_290892


namespace number_of_children_l290_290632

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l290_290632


namespace royal_family_children_l290_290604

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290604


namespace cost_of_birthday_gift_l290_290918

theorem cost_of_birthday_gift 
  (boss_contrib : ℕ)
  (todd_contrib : ℕ)
  (employee_contrib : ℕ)
  (num_employees : ℕ)
  (h1 : boss_contrib = 15)
  (h2 : todd_contrib = 2 * boss_contrib)
  (h3 : employee_contrib = 11)
  (h4 : num_employees = 5) :
  boss_contrib + todd_contrib + num_employees * employee_contrib = 100 := by
  sorry

end cost_of_birthday_gift_l290_290918


namespace unique_similar_triangles_l290_290180

noncomputable def triangle_similar_sides (a b c a' b' c' : ℕ) : Prop :=
  a < a' ∧  a < b ∧ b < c ∧ a = 8 ∧
  (a * a' = b * b') ∧ (a * a' = c * c') ∧
  a * a' < (b * b' + c * c')

theorem unique_similar_triangles :
  ∃! (a b c a' b' c' : ℕ), triangle_similar_sides a b c a' b' c' :=
exists_unique.intro 8 12 18 12 18 27
begin
    unfold triangle_similar_sides,
    split,
    { split,
      { exact dec_trivial },
      { split,
        { exact dec_trivial },
        { split,
          { exact dec_trivial },
          { split,
            { refl },
            { split,
              {exact dec_trivial},
              {exact dec_trivial}}}}}},
    intros x hx,
    cases hx with xA hx,
    cases hx with xb hc,
    cases hc with cruc_A  cru_b,
    cases cru_b with cru_B cru_c,
    cases cru_c with ax bx,
    have eq_a: xA = 8, from eq.symm ax,
    have eq_b: xb = 12 , from _,
    tauto
end

end unique_similar_triangles_l290_290180


namespace fish_count_seventh_day_l290_290431

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end fish_count_seventh_day_l290_290431


namespace probability_not_snow_l290_290971

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l290_290971


namespace inequality_solution_l290_290275

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, -1/2 < x ∧ x ≤ 1 → 2^x - real.arccos x > a) ↔ 
  a < (real.sqrt 2 / 2 - 2 * real.pi / 3) :=
sorry

end inequality_solution_l290_290275


namespace number_of_possible_m_values_l290_290461

theorem number_of_possible_m_values :
  ∃ m_set : Finset ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 40 → (x1 + x2) ∈ m_set) ∧ m_set.card = 8 :=
sorry

end number_of_possible_m_values_l290_290461


namespace angle_bisector_angle_bac_l290_290047

open EuclideanGeometry

variable {A B C P Q R S : Point}

noncomputable theory

-- Define the conditions
def is_on_segment (p a b : Point) : Prop := ∃ λ, 0 ≤ λ ∧ λ ≤ 1 ∧ p = (1 - λ) *: a + λ *: b
def same_distance (a b c d : Point) : Prop := dist a b = dist c d
def is_intersecting (l1 l2 : line) (p : Point) : Prop := p ∈ l1 ∧ p ∈ l2
def is_on_circumcircle (a b c d : Point) : Prop := cyclic_quad a b c d

-- Define the theorem
theorem angle_bisector_angle_bac
    (h_seg_P : is_on_segment P A B)
    (h_seg_Q : is_on_segment Q A C)
    (h_same_dist : same_distance B P C Q)
    (h_intersect_R : is_intersecting (line B Q) (line C P) R)
    (h_circum_S_BP : is_on_circumcircle B P R S)
    (h_circum_S_CQ : is_on_circumcircle C Q R S) :
    lies_on_angle_bisector A B C S :=
sorry

end angle_bisector_angle_bac_l290_290047


namespace M_midpoint_DF_l290_290846

-- Definitions of the points and lines in the problem
noncomputable def triangle_ABC (A B C : Point) : Prop :=
  collinear A B C ∧ (distance A B = distance A C)

noncomputable def perpendicular (P Q R : Point) : Prop :=
  ∃ (line1 : Line), ∃ (line2 : Line),
  line1.contains P ∧ line1.contains Q ∧
  line2.contains Q ∧ line2.contains R ∧
  line1.is_perpendicular line2

noncomputable def midpoint (M P Q : Point) : Prop :=
  distance P M = distance M Q

-- Main theorem statement
theorem M_midpoint_DF (A B C D F E M : Point) :
  triangle_ABC A B C →
  perpendicular A D B C →
  perpendicular D F A B →
  perpendicular A E C F →
  intersects A E D F M →
  midpoint M D F :=
by
  sorry

end M_midpoint_DF_l290_290846


namespace geometric_sequence_b_value_l290_290980

theorem geometric_sequence_b_value (b : ℝ) (r : ℝ) (h1 : 210 * r = b) (h2 : b * r = 35 / 36) (hb : b > 0) : 
  b = Real.sqrt (7350 / 36) :=
by
  sorry

end geometric_sequence_b_value_l290_290980


namespace tournament_players_l290_290026

noncomputable def total_players_in_tournament : ℕ :=
  let n := 25 in n

theorem tournament_players :
  let n := total_players_in_tournament in
  let x := n - 10 in
  (10 + x = 25) ∧
  (∑ i in (finset.range 10).powerset, 1 = 45) ∧
  (10 * x - 45 = ∑ i in (finset.range x).powerset, 1 / 2) ∧
  (2 * (10 * x - 45) = x * (x - 1) / 2) :=
begin
  sorry
end

end tournament_players_l290_290026


namespace prove_missing_exponent_l290_290183

-- Conditions (given problem)
def problem_condition (x : ℝ) : Prop :=
  (9^x * 9^10.3) / 9^2.56256 = 9^13.33744

-- Prove that the correct x is 5.6
theorem prove_missing_exponent : problem_condition 5.6 :=
by
  -- sorry is a placeholder for the proof
  sorry

end prove_missing_exponent_l290_290183


namespace center_of_circle_l290_290112

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem center_of_circle (x1 y1 x2 y2 : ℝ) :
  (x1, y1) = (2, -3) → (x2, y2) = (8, 9) → midpoint (x1, y1) (x2, y2) = (5, 3) := 
by 
  intros h1 h2
  rw [h1, h2]
  simp [midpoint]
  sorry

end center_of_circle_l290_290112


namespace constant_r_l290_290554

theorem constant_r
  (p q a b : ℝ)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = p)
  (h₂ : x₁ * x₂ = q - b + a * p)
  : ∃ r : ℝ, 
    r = 
    1 / (Real.sqrt ((x₁ - a)^2 + (x₁^2 + p * x₁ + q - b)^2)) +
    1 / (Real.sqrt ((x₂ - a)^2 + (x₂^2 + p * x₂ + q - b)^2)) :=
begin
  sorry
end

end constant_r_l290_290554


namespace range_of_m_l290_290807

theorem range_of_m 
  (x y : ℝ)
  (h1 : ∀ (x y : ℝ), max (|x - y^2 + 4|) (|2y^2 - x + 8|) ≥ 6)
  (h2 : ∀ (x y : ℝ), max (|x - y^2 + 4|) (|2y^2 - x + 8|) ≥ m^2 - 2 * m) :
  1 - real.sqrt 7 ≤ m ∧ m ≤ 1 + real.sqrt 7 :=
sorry

end range_of_m_l290_290807


namespace cross_section_area_is_correct_l290_290803

noncomputable def cross_section_area_tetrahedron (M : ℝ × ℝ × ℝ) (S : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) (O : ℝ × ℝ × ℝ) : ℝ :=
  let l := 1 in
  let SO := (S.1 - O.1, S.2 - O.2, S.3 - O.3) in
  let M := ((S.1 + O.1) / 2, (S.2 + O.2) / 2, (S.3 + O.3) / 2) in
  let h := real.sqrt(2 / 3) * l in
  let SM := h / 2 in
  let area := 1 / 6 in
  area

theorem cross_section_area_is_correct (M : ℝ × ℝ × ℝ) (S : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) (O : ℝ × ℝ × ℝ) :
  cross_section_area_tetrahedron M S A B C O = 1 / 6 := 
sorry

end cross_section_area_is_correct_l290_290803


namespace min_value_ineq_l290_290474

theorem min_value_ineq (x y z : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) 
  (h2 : x^2 + y^2 + z^2 = 1) : 
  (1 / (x^2 + y^2) + 1 / (x^2 + z^2) + 1 / (y^2 + z^2)) ≥ 9 / 2 :=
begin
  sorry
end

end min_value_ineq_l290_290474


namespace age_ratio_in_years_l290_290565

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end age_ratio_in_years_l290_290565


namespace probability_event_A_l290_290981

def chariot : Type := Unit
def horse : Type := Unit
def cannon : Type := Unit
def color : Type := Unit

def arrangement (pieces : List (Unit × Unit)) : Prop :=
  ∃ (redChariot blueChariot redHorse blueHorse redCannon blueCannon : Unit), 
    pieces = [(redChariot, ()), (blueChariot, ()), (redHorse, ()), (blueHorse, ()), (redCannon, ()), (blueCannon, ())] ∧
    redChariot < blueChariot ∧ redHorse < blueHorse ∧ redCannon < blueCannon

theorem probability_event_A : 
  let total_arrangements := 2^3 in
  let favorable_arrangements := 1 in
  total_arrangements > 0 → (favorable_arrangements / total_arrangements) = (1 / 8) :=
by sorry

end probability_event_A_l290_290981


namespace revenue_decrease_percent_l290_290979

theorem revenue_decrease_percent (T C : ℝ) (hT_pos : T > 0) (hC_pos : C > 0) :
  let new_T := 0.75 * T
  let new_C := 1.10 * C
  let original_revenue := T * C
  let new_revenue := new_T * new_C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 17.5 := 
by {
  sorry
}

end revenue_decrease_percent_l290_290979


namespace distance_of_parallel_lines_l290_290783

theorem distance_of_parallel_lines (m : ℝ) 
  (l₁ : ℝ × ℝ → ℝ) (l₂ : ℝ × ℝ → ℝ) 
  (h₁ : ∀ p, l₁ p = sqrt 3 * p.1 - p.2 + 7)
  (h₂ : ∀ p, l₂ p = m * p.1 + p.2 - 1)
  (parallel : ∀ x₁ y₁ x₂ y₂, sqrt 3 * y₁ - x₁ = m * y₂ - x₂ → y₁ = y₂) : 
  ∃ d, d = 3 ∧ (distance_between_lines l₁ l₂ = d) :=
sorry

end distance_of_parallel_lines_l290_290783


namespace circle_locus_properties_l290_290658

theorem circle_locus_properties :
  let F := (0, 1) in
  let tangent_line := λ y, y = -1 in
  let locus_M := λ x y, x^2 = 4 * y in
  let A := λ x0, (-x0, (1/4) * x0^2) in
  let D := λ x0, (x0, (1/4) * x0^2) in
  let l := λ k x0, k = (1/2) * x0 in
  let B := λ x1, (x1, (1/4) * x1^2) in
  let C := λ x2, (x2, (1/4) * x2^2) in
  let slope_BC := λ x0 x1 x2, (x1 + x2) / 4 = (1/2) * x0 in
  x^2 = 4 * y ∧
  ∀ (x0 : ℝ), let A_coords := A x0 in
    let D_coords := D x0 in
    let k_BC := (1/2) * x0 in
    let slope_A := ((1/4) * x1^2 - (1/4) * x2^2) / (x1 - x2) in
    (∠ BAD = ∠ CAD) ∧
    let dist_AD := 2 * sqrt(2) * (x0 - 2) in
    let area_ABC := 1/2 * 2 * sqrt(2) * (x0 + 2) in
    (y - (1/4) * x0^2 = -1 * (x + x0)) ∧
    6 * x - 4 * y + 7 = 0 ∨ 6 * x + 4 * y - 7 = 0 :=
sorry

end circle_locus_properties_l290_290658


namespace three_digit_numbers_with_2_without_4_l290_290375

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l290_290375


namespace proof_sum_of_adjacent_to_11_is_275_l290_290124

-- Definitions based on conditions
def is_divisor (a b : Nat) : Prop := b % a = 0

def has_common_factor_greater_than_one (a b : Nat) : Prop :=
  ∃ d:Nat, d > 1 ∧ is_divisor d a ∧ is_divisor d b

def adjacent_in_circle (l : List Nat) : Prop :=
  ∀ i, let n := l.length in has_common_factor_greater_than_one (l.get (i % n)) (l.get ((i + 1) % n))

-- Specific to problem conditions
def divisors_of_220 : List Nat := [2, 4, 5, 10, 11, 20, 22, 44, 55, 110, 220]

-- Directly testing the sum of integers adjacent to 11
def sum_of_adjacent_to_11_is_275 : Prop :=
  ∃ l, List.Perm l divisors_of_220 ∧ adjacent_in_circle l ∧ 
       ∃ a b, has_common_factor_greater_than_one a 11 ∧ has_common_factor_greater_than_one b 11 ∧ 
              (a + b = 275) ∧ (a ≠ 11 ∧ b ≠ 11)

theorem proof_sum_of_adjacent_to_11_is_275 : sum_of_adjacent_to_11_is_275 :=
sorry

end proof_sum_of_adjacent_to_11_is_275_l290_290124


namespace chord_length_of_concentric_circles_l290_290949

theorem chord_length_of_concentric_circles
  (R r : ℝ) (h : R^2 - r^2 = 20) :
  ∃ c, c = 4 * Real.sqrt 5 ∧
  (∀ P, let d := dist P (0, 0) in 
    (d = R → dist P (0, 1) = r → dist P (x, 0) < c → False) → 
    dist P (0, 1) = r) :=
  sorry

end chord_length_of_concentric_circles_l290_290949


namespace upstream_travel_time_l290_290201

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l290_290201


namespace max_expression_value_eq_32_l290_290861

theorem max_expression_value_eq_32 :
  ∃ (a b c d : ℕ), a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ d ∈ {1, 2, 3, 4} ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (c * a^b - d = 32) :=
sorry

end max_expression_value_eq_32_l290_290861


namespace find_QS_l290_290509

theorem find_QS (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 3 / 5) (h2 : RS = 10) (h3 : cosR = QR / RS) (h4: QR ^ 2 + QS ^ 2 = RS ^ 2) : QS = 8 :=
by 
  sorry

end find_QS_l290_290509


namespace arithmetic_sequence_relation_l290_290102

noncomputable def calc_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 5 / S 2 = -11

theorem arithmetic_sequence_relation
  (a : ℕ → ℝ) -- ℝ is used as a generic real number sequence
  (S : ℕ → ℝ) -- ℝ is used as the sum type
  (h1 : a 2 + 8 * a 5 = 0) -- Given condition on the arithmetic sequence
  (h2 : ∀ n, S n = (n * (2 * (2 ^ n - 1)) / 2) * (1 / a 1)) -- Definition of sum of the sequence {1/a_n}
: calc_ratio a S :=
begin
  sorry
end

end arithmetic_sequence_relation_l290_290102


namespace find_AM_MB_ratio_l290_290496

noncomputable def cube_coordinates : Type := { 
  A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ 
  // Coordinates for the points forming the cube
}

noncomputable def is_point_on_edge (P AB: ℝ × ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • AB

noncomputable def is_rectangle_inscribed (M N K L: ℝ × ℝ × ℝ) (A B C D: ℝ × ℝ × ℝ) : Prop :=
  -- Check if vertices M, N, K, L form a rectangle inscribed in square ABCD
  M ∈ {A, B} ∧ 
  N ∈ {B, C} ∧ 
  K ∈ {C, D} ∧ 
  L ∈ {D, A}

noncomputable def is_orthogonal_projection (M1 N1 L1 K1: ℝ × ℝ × ℝ) (M N K L: ℝ × ℝ × ℝ) : Prop :=
  -- Check if M1, N1, L1, K1 are the orthogonal projections of M, N, K, L respectively
  ∀ (P P1 : ℝ × ℝ × ℝ), P1 = (P.1, P.2, M1.3)

noncomputable def are_diagonals_perpendicular (M K1 L1 N: ℝ × ℝ × ℝ) : Prop :=
  -- Check if the diagonals of the quadrilateral MK1L1N are perpendicular
  let diag1 := (M.1 - K1.1, M.2 - K1.2, M.3 - K1.3) in
  let diag2 := (L1.1 - N.1, L1.2 - N.2, L1.3 - N.3) in
  diag1.1 * diag2.1 + diag1.2 * diag2.2 + diag1.3 * diag2.3 = 0

theorem find_AM_MB_ratio : 
  ∀ (A B C D A1 B1 C1 D1 M N K L M1 N1 L1 K1: ℝ × ℝ × ℝ),
  (cube_coordinates) →
  (is_point_on_edge M (A - B)) →
  (is_rectangle_inscribed M N K L A B C D) →
  (is_orthogonal_projection M1 N1 L1 K1 M N K L A1 B1 C1 D1) →
  (are_diagonals_perpendicular M K1 L1 N) →
  ((M.1 / B.1) = 1 / 2) :=
sorry

end find_AM_MB_ratio_l290_290496


namespace find_derivative_value_l290_290813

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x

theorem find_derivative_value 
  (a : ℝ) 
  (h : ∀ x : ℝ, deriv (f a) x = 2 * x + 2 * a) :
  a = 2 / 3 :=
by
  sorry

end find_derivative_value_l290_290813


namespace inequality_system_solution_l290_290944

theorem inequality_system_solution {x : ℝ} :
  (2 * x + 1 < 5) ∧ (3 - x > 2) → x < 1 :=
by
  intro h
  cases h with h1 h2
  sorry

end inequality_system_solution_l290_290944


namespace five_digit_palindromes_count_l290_290741

theorem five_digit_palindromes_count :
  (∃ (A B C : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) → 
  ∃ (count : ℕ), count = 900 :=
by {
  intro h,
  use 900,
  sorry        -- Proof is omitted
}

end five_digit_palindromes_count_l290_290741


namespace evaluate_fraction_l290_290261

theorem evaluate_fraction (a b : ℕ) (ha : a = 5) (hb : b = 6) : (3 * b) / (a + b) = 18 / 11 :=
by {
  // We provide an assumption to satisfy the example
  have h1: a + b = 11 := by rw [ha, hb]; exact rfl,
  have h2: 3 * b = 18 := by rw [hb]; exact rfl,
  rw h1,
  rw h2,
  norm_num,
  exact rfl,
  sorry
}

end evaluate_fraction_l290_290261


namespace new_average_production_l290_290773

theorem new_average_production (n : ℕ) (daily_avg : ℕ) (today_prod : ℕ) (new_avg : ℕ) 
  (h1 : daily_avg = 50) 
  (h2 : today_prod = 95) 
  (h3 : n = 8) 
  (h4 : new_avg = (daily_avg * n + today_prod) / (n + 1)) : 
  new_avg = 55 := 
sorry

end new_average_production_l290_290773


namespace integral_result_l290_290758

noncomputable def integrate_rational_fraction (x : ℝ) : ℝ :=
  ∫ t in 0..x, (7 * t^3 - 4 * t^2 - 32 * t - 37) / ((t + 2) * (2 * t - 1) * (t^2 + 2 * t + 3))

theorem integral_result :
  ∀ x : ℝ, integrate_rational_fraction x =
    3 * log |x + 2| - 5 / 2 * log |2 * x - 1| +
    3 / 2 * log |x^2 + 2 * x + 3| -
    4 / real.sqrt 2 * real.arctan ((x + 1) / real.sqrt 2) + 0 := 
  by
  sorry

end integral_result_l290_290758


namespace sequence_problem_l290_290308

theorem sequence_problem :
  (∀ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, (1 + a n = (1 + a 1) * 3 ^ (n - 1))) → a 4 = 80) :=
begin
  sorry
end

end sequence_problem_l290_290308


namespace comparison_of_maximal_root_l290_290326

theorem comparison_of_maximal_root
  {a1 a2 a3 b1 b2 b3 : ℝ}
  (h1 : a1 ≤ a2) (h2 : a2 ≤ a3)
  (h3 : b1 ≤ b2) (h4 : b2 ≤ b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_prod : a1 * a2 + a2 * a3 + a1 * a3 = b1 * b2 + b2 * b3 + b1 * b3)
  (h_min : a1 ≤ b1) :
  a3 ≤ b3 :=
begin
  sorry
end

end comparison_of_maximal_root_l290_290326


namespace min_value_expression_l290_290274

theorem min_value_expression (x y : ℝ) : 
  (∃ (x_min y_min : ℝ), 
  (x_min = 1/2 ∧ y_min = 0) ∧ 
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39/4) :=
by
  sorry

end min_value_expression_l290_290274


namespace find_a_if_line_passes_through_center_l290_290838

-- Define the given circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the given line equation
def line_eqn (x y a : ℝ) : Prop := 3*x + y + a = 0

-- The coordinates of the center of the circle
def center_of_circle : (ℝ × ℝ) := (-1, 2)

-- Prove that a = 1 if the line passes through the center of the circle
theorem find_a_if_line_passes_through_center (a : ℝ) :
  line_eqn (-1) 2 a → a = 1 :=
by
  sorry

end find_a_if_line_passes_through_center_l290_290838


namespace royal_family_children_l290_290602

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290602


namespace sum_reciprocal_roots_eq_757_l290_290054

noncomputable def polynomial_roots (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a : ℕ → ℂ), (∀ i, 1 ≤ i ∧ i ≤ n → is_root (λ x, ∑ i : ℕ in finset.range (k + 1), x ^ i - 670) (a i))

theorem sum_reciprocal_roots_eq_757 :
  polynomial_roots 1010 1010 →
  ∑ i in finset.range 1010, (1 : ℂ) / (1 - polynomial_roots.some i) = 757 := 
sorry

end sum_reciprocal_roots_eq_757_l290_290054


namespace arithmetic_sequence_sum_l290_290710

-- Definitions for the conditions
def a := 70
def d := 3
def n := 10
def l := 97

-- Sum of the arithmetic series
def S := (n / 2) * (a + l)

-- Final calculation
theorem arithmetic_sequence_sum :
  3 * (70 + 73 + 76 + 79 + 82 + 85 + 88 + 91 + 94 + 97) = 2505 :=
by
  -- Lean will calculate these interactively when proving.
  sorry

end arithmetic_sequence_sum_l290_290710


namespace circle_with_diameter_A_B_l290_290516

-- Defining the points A and B
variable (A B : ℝ × ℝ)

-- The positions of points A and B
def point_A : A = (1, 4) := by sorry
def point_B : B = (3, -2) := by sorry

-- Definition to state the center calculation
def circle_midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Definition for the distance squared
def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the circle equation based on center and radius
def circle_equation (center : ℝ × ℝ) (radius_squared : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius_squared

-- Theorem stating the given condition and the expected result
theorem circle_with_diameter_A_B :
  let center := circle_midpoint A B,
      radius_squared := distance_squared A center,
      x := 2,
      y := 1 in
  point_A A →
  point_B B →
  circle_equation (2, 1) 10 (x) (y) :=
by sorry

end circle_with_diameter_A_B_l290_290516


namespace example_theorem_l290_290983

noncomputable def second_largest_divided_by_smallest (numbers : List ℕ) : ℚ :=
  let sorted_numbers := List.sort (≤) numbers
  let second_largest := sorted_numbers[sorted_numbers.length - 2]
  let smallest := sorted_numbers.head!
  rat.of_int second_largest / rat.of_int smallest

theorem example_theorem :
  second_largest_divided_by_smallest [10, 11, 12, 13, 14] = 1.3 := by
  sorry

end example_theorem_l290_290983


namespace number_of_true_statements_l290_290782

theorem number_of_true_statements 
  (a b c : ℝ) 
  (Hc : c ≠ 0) : 
  ((a > b → a * c^2 > b * c^2) ∧ (a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
  ¬((a * c^2 > b * c^2 → a > b) ∨ (a ≤ b → a * c^2 ≤ b * c^2)) :=
by
  sorry

end number_of_true_statements_l290_290782


namespace find_number_of_terms_l290_290131

variable {n : ℕ} {a : ℕ → ℤ}
variable (a_seq : ℕ → ℤ)

def sum_first_three_terms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3

def sum_last_three_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  a (n-2) + a (n-1) + a n

def sum_all_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem find_number_of_terms (h1 : sum_first_three_terms a_seq = 20)
    (h2 : sum_last_three_terms n a_seq = 130)
    (h3 : sum_all_terms n a_seq = 200) : n = 8 :=
sorry

end find_number_of_terms_l290_290131


namespace song_arrangements_count_l290_290234

-- We define the participants and pairs
inductive Person | Amy | Beth | Jo | Meg

open Person

def Pair := {p : Person × Person // p.1 ≠ p.2}

-- Define a finite set of five songs
def Song : Type := Fin 5

-- Define like function mapping pairs to finite sets of songs
def likes : Pair → Finset Song

-- Conditions
axiom cond_no_all_four : ∀ s : Song, ¬ (likes ⟨(Amy, Beth), by simp⟩ s ∧ 
                                          likes ⟨(Amy, Jo), by simp⟩ s ∧ 
                                          likes ⟨(Amy, Meg), by simp⟩ s ∧ 
                                          likes ⟨(Beth, Jo), by simp⟩ s ∧
                                          likes ⟨(Beth, Meg), by simp⟩ s ∧
                                          likes ⟨(Jo, Meg), by simp⟩ s)

axiom cond_each_pair : ∀ p : Pair, ∃ s : Song, likes p s

-- Statement to be proved
theorem song_arrangements_count : 
  (∃ (likes : Pair → Finset Song), cond_no_all_four ∧ cond_each_pair) → 48 :=
sorry

end song_arrangements_count_l290_290234


namespace math_metropolis_intersections_l290_290075

theorem math_metropolis_intersections :
  let streets := 10
  let intersections := (streets - 1) * streets / 2
  intersections = 45 := by
  let streets := 10
  let intersections := (streets - 1) * streets / 2
  show intersections = 45
  by sorry

end math_metropolis_intersections_l290_290075


namespace tan_C_eq_neg_2_sqrt_2_l290_290805

theorem tan_C_eq_neg_2_sqrt_2 (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 + 2 * a * b = 0) :
  (tan (real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -2 * real.sqrt 2) :=
by
  sorry

end tan_C_eq_neg_2_sqrt_2_l290_290805


namespace triangle_perimeter_l290_290227

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_perimeter {a b c : ℕ} (h : is_triangle 15 11 19) : 15 + 11 + 19 = 45 := by
  sorry

end triangle_perimeter_l290_290227


namespace pure_imaginary_solutions_l290_290155
-- Import the entire Mathlib library to ensure all necessary
-- functions and theorems are available.

-- Define the polynomial.
def poly (x : ℂ) : ℂ := x^4 - 4*x^3 + 6*x^2 - 40*x - 64

-- Statement of the theorem.
theorem pure_imaginary_solutions :
  ∃ k : ℝ, k = real.sqrt 10 ∨ k = -real.sqrt 10 ∧ poly (k * complex.I) = 0 := 
by
  sorry

end pure_imaginary_solutions_l290_290155


namespace minimize_g_l290_290790

/-- Math proof problem statement -/
theorem minimize_g : 
  let f (x : ℝ) := x^2 - x,
      l (t : ℝ) := t^2 - t,
      S1 (t : ℝ) := ∫ x in 0..t, (f x - l t),
      S2 (t : ℝ) := ∫ x in (1/2)..t, (l t - f x),
      g (t : ℝ) := S1 t + 1 / 2 * S2 t in
  0 < t ∧ t < 1/2 → g t = g (1/4) → t = 1/4 :=
by
  sorry

end minimize_g_l290_290790


namespace volleyball_tournament_equation_l290_290925

-- Definitions (conditions)
def total_matches (x : ℕ) : ℕ := x * (x - 1) / 2
def matches := 28

-- Theorem that needs to be proved
theorem volleyball_tournament_equation (x : ℕ) : total_matches x = matches → (x * (x - 1)) / 2 = 28 :=
by sorry

end volleyball_tournament_equation_l290_290925


namespace dave_guitar_strings_l290_290725

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l290_290725


namespace length_of_lk_equals_ba_l290_290048

-- Abstracting the problem into conditions and required proof
theorem length_of_lk_equals_ba
  (A B C D E L K: Point)
  (circular_triangle: Triangle ABC)
  (circumcircle: Circle (Triangle.circumcenter circular_triangle) (Triangle.circumradius circular_triangle))
  (DE_perp_AC: is_perpendicular DE AC)
  (D_circum: D ∈ circumcircle)
  (E_circum: E ∈ circumcircle)
  (proj_D_L: is_projection D L BC)
  (proj_E_K: is_projection E K BC)
  :
  length LK = length BA := 
sorry

end length_of_lk_equals_ba_l290_290048


namespace complex_number_quadrant_l290_290335

theorem complex_number_quadrant (i : ℂ) (hi : i * i = -1) : 
  let z := i * (2 + i) in 
  (Complex.re z < 0) ∧ (Complex.im z > 0) :=
by
  let z := i * (2 + i)
  sorry

end complex_number_quadrant_l290_290335


namespace even_cycle_exists_l290_290720

open Nat

-- Define the conditions
variables (A : Type) [Plane A] (n : ℕ) (h_n : n ≥ 4)
variables (points : Fin n → A)
variable (connected : (Fin n) → (Fin n) → Prop)
variable (h_collinear : ∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → ¬Collinear {points i, points j, points k})
variable (h_connected : ∀ i : Fin n, ∃ j1 j2 j3 : Fin n, j1 ≠ i ∧ j2 ≠ i ∧ j3 ≠ i ∧ connected i j1 ∧ connected i j2 ∧ connected i j3)

-- Define the goal
theorem even_cycle_exists : ∃ k > 1, ∃ X : Fin (2 * k) → Fin n, (∀ i : Fin (2 * k - 1), connected (X i) (X (i + 1)))
∧ connected (X ⟨2 * k - 1, Nat.pred_lt (mul_lt_mul_of_pos_left (show 1 < 2, by linarith [h_n]) (by linarith)⟩)) (X ⟨0, by linarith⟩) := sorry

end even_cycle_exists_l290_290720


namespace three_digit_integers_with_at_least_one_two_but_no_four_l290_290373

-- Define the properties
def is_three_digit (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000
def contains_digit (n: ℕ) (d: ℕ) : Prop := ∃ i, i < 3 ∧ d = n / 10^i % 10
def no_four (n: ℕ) : Prop := ¬ contains_digit n 4

-- Define the sets A and B
def setA (n: ℕ) : Prop := is_three_digit n ∧ no_four n
def setB (n: ℕ) : Prop := setA n ∧ ¬ contains_digit n 2

-- The final theorem statement
theorem three_digit_integers_with_at_least_one_two_but_no_four : 
  {n : ℕ | contains_digit n 2 ∧ setA n}.card = 200 :=
sorry

end three_digit_integers_with_at_least_one_two_but_no_four_l290_290373


namespace ticket_door_price_l290_290134

theorem ticket_door_price
  (total_attendance : ℕ)
  (tickets_before : ℕ)
  (price_before : ℚ)
  (total_receipts : ℚ)
  (tickets_bought_before : ℕ)
  (price_door : ℚ)
  (h_attendance : total_attendance = 750)
  (h_price_before : price_before = 2)
  (h_receipts : total_receipts = 1706.25)
  (h_tickets_before : tickets_bought_before = 475)
  (h_total_receipts : (tickets_bought_before * price_before) + (((total_attendance - tickets_bought_before) : ℕ) * price_door) = total_receipts) :
  price_door = 2.75 :=
by
  sorry

end ticket_door_price_l290_290134


namespace area_transformed_function_l290_290127

variable (f : ℝ → ℝ)
variable (a b : ℝ)
variable (h_integral : ∫ x in a..b, f x = 12)

theorem area_transformed_function :
  ∫ x in a..b, (2 * f (x - 1) + 4) = 24 :=
by
  sorry

end area_transformed_function_l290_290127


namespace ellipse_and_circle_properties_l290_290912

noncomputable def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : set (ℝ × ℝ) :=
{ p | ∃ x y, p = (x, y) ∧ (x^2)/(a^2) + (y^2)/(b^2) = 1 }

def related_circle (a b : ℝ) : set (ℝ × ℝ) :=
{ p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = (a^2 * b^2) / (a^2 + b^2) }

def parabola_focus : ℝ × ℝ := (1, 0)

def ellipse_foci (a b : ℝ) : set (ℝ × ℝ) :=
{ (sqrt(a^2 - b^2), 0), (-sqrt(a^2 - b^2), 0) }

theorem ellipse_and_circle_properties
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_focus : parabola_focus ∈ ellipse_foci a b)
  (h_minor_axis : ∃ x y, (x = 0 ∧ (y = b ∨ y = -b)) ∧ (0, y) forms a right triangle with the foci) :
  ellipse a b ha hb hab = {p | ∃ x y, p = (x, y) ∧ (x^2)/2 + y^2 = 1 } ∧
  related_circle a b = {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 2/3 } ∧
  (∀ (P : ℝ × ℝ), P ∈ related_circle a b → eq_angle (line_through P (0, 0)) (ellipse a b ha hb hab) (π / 2)) ∧
  (∀ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ related_circle a b → Q ∈ related_circle a b →
    extends_to_other_point (line_through P (0, 0)) Q (triangle_area_range ((ellipse a b ha hb hab) = [4/3, sqrt 2])) :=
sorry

end ellipse_and_circle_properties_l290_290912


namespace find_m_l290_290824

open Real

def a : ℝ × ℝ × ℝ := (2, -1, 1)
def b (m : ℝ) : ℝ × ℝ × ℝ := (m, -1, 1)

theorem find_m (m : ℝ) (h : ∃ λ : ℝ, a = (λ * b m).1) : m = 2 := by
  sorry

end find_m_l290_290824


namespace prod_z_eq_five_l290_290341

-- Define the complex number z
def z : ℂ := 1 + 2 * complex.I

-- Define the conjugate of z
def z_conjugate := conj z

-- Define the product of z and its conjugate
def z_product := z * z_conjugate

-- The theorem statement
theorem prod_z_eq_five : z_product = 5 := by
  -- proof goes here
  sorry

end prod_z_eq_five_l290_290341


namespace min_f_eq_zero_log_a_b_gt_one_g_critical_points_l290_290323

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x + 1
noncomputable def g (x m : ℝ) : ℝ := m * Real.log x + Real.exp (-x)

theorem min_f_eq_zero :
  ∃ x : ℝ, x > 0 ∧ f x = 0 := sorry

theorem log_a_b_gt_one (a b : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : b * Real.exp ((1 - a) / a) = 1) :
  Real.log a b > 1 := sorry

theorem g_critical_points (m x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : g x₁ m = g x₂ m) :
  |g x₁ m - g x₂ m| < 1 := sorry

end min_f_eq_zero_log_a_b_gt_one_g_critical_points_l290_290323


namespace correct_phrase_for_egg_consumption_l290_290427

-- Define the conditions
def comparing_eggs (today : ℕ) (seventies : ℕ) : Prop :=
  today > 2 * seventies

-- Define the main statement
theorem correct_phrase_for_egg_consumption (today seventies : ℕ) (h : comparing_eggs today seventies) : 
  "The correct comparative phrase is 'more than twice as many'" :=
sorry

end correct_phrase_for_egg_consumption_l290_290427


namespace KLMN_is_rectangle_l290_290044

variables (f : ℝ × ℝ → ℝ) [∀ x, 0 < f x]

-- Conditions for the function f
axiom f_cond (A B C D : ℝ × ℝ) (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (h2 : (A.1 = B.1 ∧ C.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2) ∨ (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)) :
  f A + f C = f B + f D

variables (K L M N : ℝ × ℝ)

axiom f_quad (h_quad : f K + f M = f L + f N)

theorem KLMN_is_rectangle : 
  ((K.1 = L.1 ∧ M.1 = N.1 ∧ K.2 = N.2 ∧ L.2 = M.2) ∨ 
   (K.1 = N.1 ∧ L.1 = M.1 ∧ K.2 = L.2 ∧ M.2 = N.2)) :=
sorry

end KLMN_is_rectangle_l290_290044


namespace annual_increase_P_correct_l290_290536

-- Definitions based on conditions
def initial_price_P (year : ℕ) : ℕ :=
  if year = 2001 then 420 else 0

def initial_price_Q (year : ℕ) : ℕ :=
  if year = 2001 then 630 else 0

def annual_increase_Q : ℕ := 15

def price_in_year_Q (initial_price : ℕ) (years_passed : ℕ) : ℕ := 
  initial_price + annual_increase_Q * years_passed

-- Lean Statement
theorem annual_increase_P_correct :
  ∃ (x : ℕ), ∀ years_passed : ℕ,
  let price_P_2001 := initial_price_P 2001 in
  let price_Q_2001 := initial_price_Q 2001 in
  let price_Q_2011 := price_in_year_Q price_Q_2001 10 in
  price_Q_2011 + 40 = price_P_2001 + x * years_passed → 
  x = 40 := 
sorry

end annual_increase_P_correct_l290_290536


namespace john_billed_for_28_minutes_l290_290301

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end john_billed_for_28_minutes_l290_290301


namespace ellipse_probability_l290_290421

-- Define the interval and the function representing the ellipse condition
def interval : Set ℝ := set.Icc (-1.0) 5.0
def f (m : ℝ) : Prop := 0 < m ∧ m < 2

-- The theorem stating the required probability
theorem ellipse_probability :
  (measure_theory.measure_of_interval (set_of f) interval) = (1 / 3 : ℝ) :=
sorry

end ellipse_probability_l290_290421


namespace vacation_cost_eq_l290_290542

theorem vacation_cost_eq (C : ℕ) (h : C / 3 - C / 5 = 50) : C = 375 :=
sorry

end vacation_cost_eq_l290_290542


namespace javelin_throw_sum_l290_290869

-- Definitions according to the conditions
def first_throw : ℝ := 300
def second_throw : ℝ := first_throw / 2
def third_throw : ℝ := first_throw * 2

-- The sum of all three throws
def total_throw : ℝ := first_throw + second_throw + third_throw

-- Prove that the total sum of throws is 1050 meters
theorem javelin_throw_sum : total_throw = 1050 := by
  -- Proof to be filled in
  sorry

end javelin_throw_sum_l290_290869


namespace sequence_remainder_mod8_l290_290024

theorem sequence_remainder_mod8 :
  ∃ a : ℕ → ℕ, 
  (∀ n, 3 ≤ n → (a n) - (a (n-1)) - (a (n-2)) % 100 = 0) ∧ 
  a 1 = 19 ∧ 
  a 2 = 99 ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + ∑ n in finset.range 1997, a (n + 3) ^ 2) % 8 = 1 :=
sorry

end sequence_remainder_mod8_l290_290024


namespace palindrome_product_2016_l290_290666

def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

theorem palindrome_product_2016 : ∃ (a b : ℕ), is_palindrome a ∧ is_palindrome b ∧ a * b = 2016 ∧ (a = 8 ∧ b = 252) :=
by 
  -- Solution not required, so we leave it as sorry
  sorry

end palindrome_product_2016_l290_290666


namespace triangle_PA_PB_PD_l290_290543

open EuclideanGeometry

-- Definitions of the given problem.
variable {A B C P D : Point}
variable {circumcircle : Circumcircle}
variable (ABC : Triangle)
variable (CA CB : Length)
variable [Circumcircle ABC P]
variable [LengthEqual CA CB : CA = CB]
variable [FootPerpendicular : PerpendicularFoot D C P B]

-- Theorem statement.
theorem triangle_PA_PB_PD :
  PA + PB = 2 * PD :=
by
  sorry

end triangle_PA_PB_PD_l290_290543


namespace monotonic_decreasing_interval_l290_290963

def f (x: ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem monotonic_decreasing_interval :
  ∃ a b: ℝ, a < b ∧ f' x < 0 → x ∈ (set.Ioo a b) :=
sorry

end monotonic_decreasing_interval_l290_290963


namespace coin_toss_10_times_random_event_l290_290558

theorem coin_toss_10_times_random_event :
  (∃ (f : ℕ → ℝ), (∀ n, f n = 1/2) ∧ (10.choose 5) * (1/2)^10 * (1/2)^10 > 0) :=
sorry

end coin_toss_10_times_random_event_l290_290558


namespace floor_area_not_greater_than_10_l290_290117

theorem floor_area_not_greater_than_10 (L W H : ℝ) (h_height : H = 3)
  (h_more_paint_wall1 : L * 3 > L * W)
  (h_more_paint_wall2 : W * 3 > L * W) :
  L * W ≤ 9 :=
by
  sorry

end floor_area_not_greater_than_10_l290_290117


namespace joan_total_spent_l290_290038

theorem joan_total_spent (cost_basketball cost_racing total_spent : ℝ) 
  (h1 : cost_basketball = 5.20) 
  (h2 : cost_racing = 4.23) 
  (h3 : total_spent = cost_basketball + cost_racing) : 
  total_spent = 9.43 := 
by 
  sorry

end joan_total_spent_l290_290038


namespace angle_TPU_is_100_degrees_l290_290966

theorem angle_TPU_is_100_degrees
  (P Q R S T U : Point)
  (h₁ : QS = QU)
  (h₂ : RS = RT)
  (h₃ : ∠ TSU = 40) : ∠ TPU = 100 := by
  sorry

end angle_TPU_is_100_degrees_l290_290966


namespace arithmetic_sum_of_11_terms_l290_290031

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (a₁ : α) (d : α) : Prop :=
∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(n + 1) * (a 0 + a n) / 2

theorem arithmetic_sum_of_11_terms
  (a₁ d : α)
  (a : ℕ → α)
  (h_seq : arithmetic_sequence a a₁ d)
  (h_cond : a 8 = (1 / 2) * a 11 + 3) :
  sum_first_n_terms a 10 = 66 := by
  sorry

end arithmetic_sum_of_11_terms_l290_290031


namespace hyperbola_focal_length_l290_290957

noncomputable def a : ℝ := Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length : ℝ := 2 * c

theorem hyperbola_focal_length :
  focal_length = 4 * Real.sqrt 3 := by
  sorry

end hyperbola_focal_length_l290_290957


namespace coefficient_x3_in_expansion_sum_of_coefficients_evaluated_at_1_number_of_4digit_numbers_l290_290184

-- Definition and proofs of binomial theorem and combinatorial selections are assumed to be in Mathlib

-- Proving the coefficient of x^3 in the expansion of (1/2 - x)^5 is -5/2
theorem coefficient_x3_in_expansion : 
  binomial_expansion (1/2 : ℝ) (-x) 5 3 = -5/2 := 
sorry

-- Proving the sum of all coefficients in expansion of (1/2 - x)^5 evaluated at x=1 is -1/32
theorem sum_of_coefficients_evaluated_at_1 : 
  evaluate_sum_of_coefficients (1/2 - 1)^5 = -1/32 := 
sorry

-- Proving there are 300 four-digit numbers from {0, 2, 3, 4, 5, 6} without repeating digits and not starting with 0
theorem number_of_4digit_numbers : 
  count_4digit_numbers [0, 2, 3, 4, 5, 6] 4 ≠ starting_with 0 = 300 := 
sorry

end coefficient_x3_in_expansion_sum_of_coefficients_evaluated_at_1_number_of_4digit_numbers_l290_290184


namespace problem1_problem2_problem3_l290_290312

-- Define the function f and its properties
variables {f : ℝ → ℝ}

-- Conditions given in the problem statement
axiom condition1 (x1 x2 : ℝ) : f (x1 / x2) = f x1 - f x2
axiom condition2 (x : ℝ) (h : x > 1) : 0 < f x
axiom condition3 : f 3 = 1

-- Problem 1: Prove that f is increasing
theorem problem1 (x1 x2 : ℝ) (h1 : 0 < x2) (h2 : x1 > x2) : f x1 > f x2 := 
sorry

-- Problem 2: Solve the inequality f(3x + 6) + f(1/x) > 2 for x
theorem problem2 (x : ℝ) : (0 < x ∧ x < 1) ↔ f (3*x + 6) + f (1/x) > 2 :=
sorry

-- Problem 3: Find the range of the real number m
theorem problem3 {m : ℝ} (h : ∀ x ∈ set.Ioo 0 3, ∀ a ∈ set.Icc (-1) 1, f x ≤ m^2 - 2*a*m + 1) :
    m ≤ -2 ∨ m = 0 ∨ 2 ≤ m :=
sorry

end problem1_problem2_problem3_l290_290312


namespace GrishaHatColorDetermination_l290_290581

universe u

namespace HatProblem

-- Defining the boys
inductive Boy
| Zhenya 
| Lyova 
| Grisha
deriving DecidableEq, Inhabited

open Boy

-- Defining the possible hat colors
inductive HatColor
| Black 
| White 
deriving DecidableEq, Inhabited

open HatColor

-- Conditions
def seatingArrangement (canSee : Boy → Boy → Prop) : Prop :=
  (canSee Zhenya Lyova) ∧ (canSee Zhenya Grisha) ∧ (canSee Lyova Grisha) ∧ ¬(canSee Lyova Zhenya) ∧ ¬(canSee Grisha Zhenya) ∧ ¬(canSee Grisha Lyova)

def hatDistribution (hatsInBag : Multiset HatColor) :=
  hatsInBag.count Black = 3 ∧ hatsInBag.count White = 2

-- Statement that captures Zhenya and Lyova's responses
def ZhenyaResponse (visibleHats : Boy → HatColor) : Prop :=
  visibleHats Lyova ≠ White ∨ visibleHats Grisha ≠ White

def LyovaResponse (visibleHats : Boy → HatColor) : Prop :=
  visibleHats Grisha ≠ White

-- Formal proof problem statement given the conditions
theorem GrishaHatColorDetermination
  (canSee : Boy → Boy → Prop)
  (hatsInBag : Multiset HatColor)
  (hatsOnBoys : Boy → HatColor)
  (H_seating : seatingArrangement canSee)
  (H_hats : hatDistribution hatsInBag)
  (H_Zhenya : ZhenyaResponse (fun b => if (canSee Zhenya b) then hatsOnBoys b else sorry))
  (H_Lyova : LyovaResponse (fun b => if (canSee Lyova b) then hatsOnBoys b else sorry)) :
  hatsOnBoys Grisha = Black :=
sorry

end HatProblem

end GrishaHatColorDetermination_l290_290581


namespace length_of_AB_l290_290808

theorem length_of_AB {A B : ℝ × ℝ} 
  (ellipse_origin : (0,0))
  (eccentricity : ℝ)
  (focus_parabola : ℝ × ℝ)
  (latus_rectum_parabola : ∀ y: ℝ, (-2, y) ∈ {p : ℝ × ℝ | p.1 = -2}) 
  (intersection_points : A = (-2, 3) ∧ B = (-2, -3))
  (eccentricity : 1/2)
  (focus_parabola : (2,0))
  : |A.2 - B.2| = 6 := 
by 
  sorry

end length_of_AB_l290_290808


namespace probability_stopping_after_three_draws_l290_290984

def draws : List (List ℕ) := [
  [2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
  [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]
]

def favorable_sequences (seqs : List (List ℕ)) : List (List ℕ) :=
  seqs.filter (λ seq => 0 ∈ seq ∧ 1 ∈ seq)

def probability_of_drawing_zhong_hua (seqs : List (List ℕ)) : ℚ :=
  (favorable_sequences seqs).length / seqs.length

theorem probability_stopping_after_three_draws :
  probability_of_drawing_zhong_hua draws = 5 / 18 := by
sorry

end probability_stopping_after_three_draws_l290_290984


namespace sum_even_odd_divisors_eq_zero_l290_290314

def H : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37

/-- 
  Beneath each divisor, we write 1 if the number has an even number of prime factors, and -1 if it has an 
  odd number of prime factors. We want to prove that the sum of this resulting sequence is 0.
-/
theorem sum_even_odd_divisors_eq_zero : 
  (Finset.sum (Finset.divisors H) (λ d, if nat.totient (d).factors.length.even then 1 else -1)) = 0 :=
by
   sorry

end sum_even_odd_divisors_eq_zero_l290_290314


namespace total_right_handed_players_is_correct_l290_290081

-- Definitions of conditions
variables (total_players : ℕ) (throwers_ratio batters_all_rounders_ratio : ℚ) 
          (left_handed_ratio : ℚ)

-- Conditions
def conditions : Prop :=
  total_players = 120 ∧ 
  throwers_ratio = 3/5 ∧
  left_handed_ratio = 2/7

-- Number of throwers
def num_throwers (total : ℕ) (ratio : ℚ) : ℕ :=
  (ratio * total).to_int -- convert to integer

-- Number of batters and all-rounders
def num_batters_all_rounders (total num_throwers : ℕ) : ℕ :=
  total - num_throwers

-- Number of left-handed players among batters and all-rounders
def num_left_handed (num : ℕ) (ratio : ℚ) : ℕ :=
  (ratio * num).to_int -- convert to integer

-- Number of right-handed players among batters and all-rounders
def num_right_handed (num num_left : ℕ) : ℕ :=
  num - num_left

-- Total number of right-handed players in the team
def total_right_handed (num_throwers num_right_handed : ℕ) : ℕ :=
  num_throwers + num_right_handed

-- Main theorem statement
theorem total_right_handed_players_is_correct :
  conditions total_players throwers_ratio left_handed_ratio →
  let num_t := num_throwers total_players throwers_ratio in
  let num_ba := num_batters_all_rounders total_players num_t in
  let num_lh := num_left_handed num_ba left_handed_ratio in
  let num_rh := num_right_handed num_ba num_lh in
  total_right_handed num_t num_rh = 106 :=
by
  sorry

end total_right_handed_players_is_correct_l290_290081


namespace reflected_circle_center_l290_290104

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l290_290104


namespace equal_sum_of_segments_l290_290513

-- Define the points O1, O2, O3 as vertices of a triangle
variables {O1 O2 O3 : Point}

-- Assume O1, O2, O3 are the centers of non-intersecting circles all with equal radii R
variables {r : ℝ} (hr : 0 < r)

-- Definitions for tangents drawn from the points O1, O2, O3.
-- Convex hexagon formed by these tangents
structure Hexagon extends ConvexHull ℝ Point where
  red_edge : ℕ → LineSegment ℝ Point
  blue_edge : ℕ → LineSegment ℝ Point
  -- Ensure alternating coloring of edges
  alternating_colors : ∀ n, (n mod 2 = 0 → (red_edge n).length ≠ 0) ∧ (n mod 2 ≠ 0 → (blue_edge n).length ≠ 0)

-- The goal is to prove the equality of sums of lengths of red and blue segments
theorem equal_sum_of_segments (H : Hexagon) :
  ∑ n in (finset.range 6).filter (λ n, n % 2 = 0), (H.red_edge n).length =
  ∑ n in (finset.range 6).filter (λ n, n % 2 ≠ 0), (H.blue_edge n).length := sorry

end equal_sum_of_segments_l290_290513


namespace simplify_expression_solve_inequality_system_l290_290645

-- Problem 1
theorem simplify_expression (m n : ℝ) (h1 : 3 * m - 2 * n ≠ 0) (h2 : 3 * m + 2 * n ≠ 0) (h3 : 9 * m ^ 2 - 4 * n ^ 2 ≠ 0) :
  ((1 / (3 * m - 2 * n) - 1 / (3 * m + 2 * n)) / (m * n / (9 * m ^ 2 - 4 * n ^ 2))) = (4 / m) :=
sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) (h1 : 3 * x + 10 > 5 * x - 2 * (5 - x)) (h2 : (x + 3) / 5 > 1 - x) :
  1 / 3 < x ∧ x < 5 :=
sorry

end simplify_expression_solve_inequality_system_l290_290645


namespace Dennis_floor_proof_l290_290729

def floor_of_Dennis : ℕ :=
  let charlie_floor : ℕ := 1 / 4 * 16
  let dennis_floor := charlie_floor + 2
  dennis_floor
  -- The correct floor on which Dennis lives is 6

theorem Dennis_floor_proof :
  (∀ (F C D : ℕ), (C = F / 4) ∧ (D = C + 2) ∧ (F = 16) → D = 6) :=
by
  intros F C D h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw h3 at h1,
  rw h1,
  rw h2,
  norm_num,
  sorry

end Dennis_floor_proof_l290_290729


namespace number_of_valid_subsets_l290_290065

def T : Set ℕ := {n | ∃ a b : ℕ, 0 ≤ a ∧ a ≤ 5 ∧ 0 ≤ b ∧ b ≤ 5 ∧ n = 2^a * 3^b}

def valid_subset (S : Set ℕ) : Prop :=
  ∀ n ∈ S, ∀ d ∣ n, d > 0 → d ∈ S

theorem number_of_valid_subsets :
  {S : Set ℕ // valid_subset S ∧ S ⊆ T}.to_finset.card = 924 :=
sorry

end number_of_valid_subsets_l290_290065


namespace tetrahedron_medians_intersect_l290_290172

noncomputable def center_of_mass_tetrahedron (tetra : {x y z w : ℝ × ℝ × ℝ}) : ℝ × ℝ × ℝ :=
  let (x, y, z, w) := tetra in
  ((x.1 + y.1 + z.1 + w.1)/4, (x.2 + y.2 + z.2 + w.2)/4, (x.3 + y.3 + z.3 + w.3)/4)

def median_ratio (tetra : {x y z w : ℝ × ℝ × ℝ}) : Prop :=
  let c := center_of_mass_tetrahedron tetra in
  ∀ v ∈ {tetra.1, tetra.2, tetra.3, tetra.4},
  let m := ((tetra.1 + tetra.2 + tetra.3) - v) / 3 in
  ∃ p, p = (3*c + v) / 4 ∧ p = c

def midpoints_bisection (tetra : {x y z w : ℝ × ℝ × ℝ}) : Prop :=
  let c := center_of_mass_tetrahedron tetra in
  (∃ p, p = (tetra.1 + tetra.2) / 2 ∧ p = c) ∧
  (∃ q, q = (tetra.3 + tetra.4) / 2 ∧ q = c)

theorem tetrahedron_medians_intersect (tetra : {x y z w : ℝ × ℝ × ℝ}) :
  median_ratio tetra ∧
  midpoints_bisection tetra :=
by { sorry }

end tetrahedron_medians_intersect_l290_290172


namespace BE_parallel_DF_l290_290878

variables (A B C E D F : Type) [NonTrivial D F]
variables [Point A] [Point B] [Point C] [Point E] [Point D] [Point F]

-- Assumptions for the points
axiom A_non_collinear_B_C : ¬collinear A B C
axiom E_not_on_AC : ¬on_line E A C
axiom E_not_B : E ≠ B

-- Defining the parallelograms
axiom parallelogram_ABCD : parallelogram A B C D
axiom parallelogram_AECF : parallelogram A E C F

-- Properties of the parallelograms in terms of parallel sides
axiom AB_parallel_CD : line_parallel (line_through A B) (line_through C D)
axiom AD_parallel_BC : line_parallel (line_through A D) (line_through B C)
axiom AE_parallel_CF : line_parallel (line_through A E) (line_through C F)
axiom AC_parallel_EF : line_parallel (line_through A C) (line_through E F)

theorem BE_parallel_DF : parallel (line_through B E) (line_through D F) :=
sorry

end BE_parallel_DF_l290_290878


namespace find_A_plus_B_l290_290072

/-- Let A, B, C, and D be distinct digits such that 0 ≤ A, B, C, D ≤ 9.
    C and D are non-zero, and A ≠ B ≠ C ≠ D.
    If (A+B)/(C+D) is an integer and C+D is minimized,
    then prove that A + B = 15. -/
theorem find_A_plus_B
  (A B C D : ℕ)
  (h_digits : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_range : 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9)
  (h_nonzero_CD : C ≠ 0 ∧ D ≠ 0)
  (h_integer : (A + B) % (C + D) = 0)
  (h_min_CD : ∀ C' D', (C' ≠ C ∨ D' ≠ D) → (C' ≠ 0 ∧ D' ≠ 0 → (C + D ≤ C' + D'))) :
  A + B = 15 := 
sorry

end find_A_plus_B_l290_290072


namespace smallest_four_digit_product_is_12_l290_290162

theorem smallest_four_digit_product_is_12 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
           (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 12 ∧ a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6) ∧
           (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 →
                     (∃ a' b' c' d' : ℕ, m = 1000 * a' + 100 * b' + 10 * c' + d' ∧ a' * b' * c' * d' = 12) →
                     n ≤ m) :=
by
  sorry

end smallest_four_digit_product_is_12_l290_290162


namespace inverse_f_l290_290114

variables {X Y Z W : Type}
variable (p : X → Y) (q : Y → Z) (r : W → X)

-- Assume p, q, and r are invertible
variable [invertible p] [invertible q] [invertible r]

noncomputable def f : W → Z := q ∘ p ∘ r

theorem inverse_f :
  (inverse f) = (inverse r) ∘ (inverse p) ∘ (inverse q) :=
by
  sorry

end inverse_f_l290_290114


namespace total_fat_served_l290_290659

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l290_290659


namespace num_factors_of_n_l290_290457

theorem num_factors_of_n (n : ℕ) (h : n = 2^4 * 3^5 * 5^6 * 7^7) : 
  ∃ k, k = 1680 ∧ (number_of_factors n) = k := 
sorry

end num_factors_of_n_l290_290457


namespace tom_paid_correct_amount_l290_290144

-- Define the conditions given in the problem
def kg_apples : ℕ := 8
def rate_apples : ℕ := 70
def kg_mangoes : ℕ := 9
def rate_mangoes : ℕ := 45

-- Define the cost calculations
def cost_apples : ℕ := kg_apples * rate_apples
def cost_mangoes : ℕ := kg_mangoes * rate_mangoes
def total_amount : ℕ := cost_apples + cost_mangoes

-- The proof problem statement
theorem tom_paid_correct_amount : total_amount = 965 :=
by
  -- The proof steps are omitted and replaced with sorry
  sorry

end tom_paid_correct_amount_l290_290144


namespace volume_of_given_solid_l290_290677

noncomputable def volume_of_solid (s : ℝ) (h : ℝ) : ℝ :=
  (h / 3) * (s^2 + (s * (3 / 2))^2 + (s * (3 / 2)) * s)

theorem volume_of_given_solid : volume_of_solid 8 10 = 3040 / 3 :=
by
  sorry

end volume_of_given_solid_l290_290677


namespace probability_of_real_solutions_l290_290934

theorem probability_of_real_solutions :
  let interval_total := -1 ≤ (a : ℝ) ∧ a ≤ 2
      interval_solutions := -1 ≤ a ∧ a ≤ 0
  in (measure {a ∈ Icc (-1 : ℝ) (2 : ℝ) | -1 ≤ a ∧ a ≤ 0}).toReal =
     (measure (Icc (-1 : ℝ) 2)).toReal :=
  sorry

end probability_of_real_solutions_l290_290934


namespace find_annual_interest_rate_l290_290545

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end find_annual_interest_rate_l290_290545


namespace problem1_problem2_l290_290181

-- Proof for Problem 1
theorem problem1 : (99^2 + 202*99 + 101^2) = 40000 := 
by {
  -- proof
  sorry
}

-- Proof for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : ((1 / (x - 1) - 2) / ((2 * x - 3) / (x^2 - 1))) = -x - 1 :=
by {
  -- proof
  sorry
}

end problem1_problem2_l290_290181


namespace compare_mixed_decimal_l290_290249

def mixed_number_value : ℚ := -2 - 1 / 3  -- Representation of -2 1/3 as a rational number
def decimal_value : ℚ := -2.3             -- Representation of -2.3 as a rational number

theorem compare_mixed_decimal : mixed_number_value < decimal_value :=
sorry

end compare_mixed_decimal_l290_290249


namespace abs_expression_value_l290_290059

theorem abs_expression_value (x : ℤ) (h : x = -2023) : abs (abs (x - 100) - abs (x + 100) - abs x) - x = 3846 :=
by
  rw h
  sorry

end abs_expression_value_l290_290059


namespace no_snow_probability_l290_290969

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l290_290969


namespace tangents_secant_intersect_l290_290706

variable {A B C O1 P Q R : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (AB AC : Set (MetricSpace A)) (t : Tangent AB) (s : Tangent AC)

variable (BC : line ( Set A))
variable (APQ : secant A P Q) 

theorem tangents_secant_intersect { AR AP AQ : ℝ } :
  2 / AR = 1 / AP + 1 / AQ :=
by
  sorry

end tangents_secant_intersect_l290_290706


namespace circle_equation_l290_290203

theorem circle_equation 
    (a : ℝ)
    (x y : ℝ)
    (tangent_lines : x + y = 0 ∧ x + y = 4)
    (center_line : x - y = a)
    (center_point : ∃ (a : ℝ), x = a ∧ y = a) :
    ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

end circle_equation_l290_290203


namespace product_end_digit_3_mod_5_l290_290285

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l290_290285


namespace royal_children_l290_290614

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l290_290614


namespace line_circle_position_l290_290123

theorem line_circle_position (x0 y0 a : ℝ) (h1 : x0^2 + y0^2 < a^2) (h2 : a > 0) :
  x0 ≠ 0 ∨ y0 ≠ 0 → 
  (∃ (d : ℝ), d = |a^2| / ((x0^2 + y0^2)^(1/2)) ∧ d > a) :=
begin
  sorry
end

end line_circle_position_l290_290123


namespace triangle_side_length_l290_290842

theorem triangle_side_length (A : ℝ) (AC BC AB : ℝ) 
  (hA : A = 60)
  (hAC : AC = 4)
  (hBC : BC = 2 * Real.sqrt 3) :
  AB = 2 :=
sorry

end triangle_side_length_l290_290842


namespace infinite_common_divisor_l290_290088

theorem infinite_common_divisor (n : ℕ) : ∃ᶠ n in at_top, Nat.gcd (2 * n - 3) (3 * n - 2) > 1 := 
sorry

end infinite_common_divisor_l290_290088


namespace radius_of_other_circle_is_5_l290_290647

/-
Two circles are externally tangent to each other, and the distance between their centers is 8 cm.
The radius of one circle is 3 cm.

We want to prove that the radius of the other circle is 5 cm.
-/

def distance_between_centers : ℕ := 8
def radius_of_first_circle : ℕ := 3

theorem radius_of_other_circle_is_5 :
  ∃ (r : ℕ), distance_between_centers = radius_of_first_circle + r ∧ r = 5 :=
by
  use 5
  split
  · rfl
  · rfl
#align radius_of_other_circle_is_5 the radius of the other circle is 5 cm

end radius_of_other_circle_is_5_l290_290647


namespace product_mod_5_l290_290278

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l290_290278


namespace number_of_real_solutions_l290_290531

theorem number_of_real_solutions : 
  ∀ x : ℝ, (2 ^ (6 * x + 3)) * (4 ^ (3 * x + 6)) = 8 ^ (4 * x + 5) ↔ (∃ n : ℕ, n > 3) :=
by
  sorry

end number_of_real_solutions_l290_290531


namespace midpoint_locus_circle_line_l290_290322

theorem midpoint_locus_circle_line 
  (C : set (ℝ × ℝ))
  (hC : ∀ {x y : ℝ}, (x, y) ∈ C ↔ x^2 + (y - 2)^2 = 5)
  (line : ℝ → ℝ)
  (A B : ℝ × ℝ)
  (hA : (A.1, line A.1) ∈ C ∧ line A.1 = A.2)
  (hB : (B.1, line B.1) ∈ C ∧ line B.1 = B.2)
  (M : ℝ × ℝ) 
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  M.1^2 + (M.2 - 3/2)^2 = 1/4 :=
by
  sorry

end midpoint_locus_circle_line_l290_290322


namespace product_mod_5_l290_290277

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l290_290277


namespace inequality_of_f_l290_290453

variable {α : Type} [LinearOrder α]

def is_even_function (f : α → α) := ∀ x, f x = f (-x)
def is_increasing_on (f : α → α) (a b : α) := ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x < y → f x < f y
def is_symmetric_about (f : α → α) (c : α) := ∀ x, f (c + x) = f (c - x)

theorem inequality_of_f {f : ℝ → ℝ} (h1 : is_even_function f) (h2 : is_increasing_on f 0 3)
  (h3 : is_symmetric_about f 3) : f 6.5 < f 1.5 ∧ f 1.5 < f 3.5 :=
by
  sorry

end inequality_of_f_l290_290453


namespace royal_family_children_l290_290605

theorem royal_family_children (n d : ℕ) (h_age_king_queen : 35 + 35 = 70)
  (h_children_age : 35 = 35) (h_age_combine : 70 + 2*n = 35 + (d + 3)*n)
  (h_children_limit : d + 3 ≤ 20) : d + 3 = 7 ∨ d + 3 = 9 := by 
s

end royal_family_children_l290_290605


namespace perfume_price_l290_290213

variable (P : ℝ)

theorem perfume_price (h_increase : 1.10 * P = P + 0.10 * P)
    (h_decrease : 0.935 * P = 1.10 * P - 0.15 * 1.10 * P)
    (h_final_price : P - 0.935 * P = 78) : P = 1200 := 
by
  sorry

end perfume_price_l290_290213


namespace range_of_a_l290_290316

theorem range_of_a {a : ℝ} (h : (a^2) / 4 + 1 / 2 < 1) : -Real.sqrt 2 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_l290_290316


namespace scientific_notation_l290_290110

theorem scientific_notation (d : ℝ) (h : d = 0.0000065) : d = 6.5 * 10^(-6) :=
by
  rw [h]
  norm_num

end scientific_notation_l290_290110


namespace segment_area_proof_l290_290559

-- Define the problem parameters
def radius (R: ℝ) := R = 10
def distance_from_center (d: ℝ) := d = 6
def angle_between_chords (θ: ℝ) := θ = 30 * (Math.pi / 180)

-- Define the result (area of the segment cut off by the chord)
def area_of_segment (A: ℝ) := A = 5.93

-- Define the relationship that needs to be proved
theorem segment_area_proof
  (R d θ: ℝ)
  (hR: radius R)
  (hd: distance_from_center d)
  (hθ: angle_between_chords θ):
  area_of_segment (A: ℝ) :=
  sorry

end segment_area_proof_l290_290559
