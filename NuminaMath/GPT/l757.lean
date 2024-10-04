import Mathlib

namespace square_free_odd_integers_count_l757_757658

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757658


namespace coefficient_x2_term_l757_757972

-- Define the polynomial expressions
def P (x : ℝ) := 2 * x ^ 3 + 4 * x ^ 2 - 3 * x + 1
def Q (x : ℝ) := 3 * x ^ 2 - 2 * x - 5

-- Statement of the problem
theorem coefficient_x2_term :
  let product := P * Q in
  (coeff product 2) = -11 :=
by
  sorry

end coefficient_x2_term_l757_757972


namespace circle_center_radius_l757_757368

theorem circle_center_radius :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y - 6 = 0) →
  (∃ (C : ℝ × ℝ) (r : ℝ), C = (-1, 2) ∧ r = sqrt 11 ∧ ((x - C.1)^2 + (y - C.2)^2 = r^2)) :=
by
  intros x y h
  use (-1, 2), sqrt 11
  split
  { refl }
  split
  { refl }
  sorry

end circle_center_radius_l757_757368


namespace find_omega_range_f_x_sum_of_roots_l757_757621

open Real

-- Condition definitions
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 4)
def interval_1 : Set ℝ := Icc (π / 6) (π / 2)
def interval_2 : Set ℝ := Icc 0 (2 * π)
def a_interval : Set ℝ := Ioo 0 1

-- Question 1: Prove ω = 3 given T = 2π/3 and T = 2π/ω
theorem find_omega (T : ℝ) (h1 : T = 2 * π / 3) (h2 : T = 2 * π / ω) : ω = 3 :=
by
  sorry

-- Question 2: Prove the range of f(x) when x ∈ [π/6, π/2] is [-√2/2, 1]
theorem range_f_x (ω : ℝ) (h1 : ω = 3) (x : ℝ) (hx : x ∈ interval_1) : (f ω x) ∈ Icc (-sqrt 2 / 2) 1 :=
by
  sorry

-- Question 3: Prove the sum of all real roots of f(x) = a in [0, 2π] is 11π/2 given 0 < a < 1
theorem sum_of_roots (ω : ℝ) (h1 : ω = 3) (a : ℝ) (ha : a ∈ a_interval) : 
  (set_of (λ x, x ∈ interval_2 ∧ f ω x = a)).sum ∈ Icc (11 * π / 2) (11 * π / 2) :=
by
  sorry

end find_omega_range_f_x_sum_of_roots_l757_757621


namespace can_be_midpoint_of_AB_l757_757185

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757185


namespace polar_coordinates_correct_l757_757947

-- Define the rectangular coordinates
def rectangular_point : ℝ × ℝ := (3, -3)

-- Define the condition for converting to polar coordinates
def polar_coordinates (p : ℝ × ℝ) : ℝ × ℝ :=
  let r := real.sqrt (p.1^2 + p.2^2) in
  let theta := if p.1 > 0 ∧ p.2 < 0 then 2 * real.pi - real.atan (real.abs (p.2 / p.1)) else 0 in
  (r, theta)

-- Define the required polar coordinates
def expected_polar_coordinates : ℝ × ℝ := (3 * real.sqrt 2, 7 * real.pi / 4)

-- Assert that the computed polar coordinates are as expected
theorem polar_coordinates_correct : polar_coordinates rectangular_point = expected_polar_coordinates := by
  sorry

end polar_coordinates_correct_l757_757947


namespace weights_in_pile_l757_757885

theorem weights_in_pile (a b c : ℕ) (h1 : a + b + c = 100) (h2 : a + 10 * b + 50 * c = 500) : 
  a = 60 ∧ b = 39 ∧ c = 1 :=
sorry

end weights_in_pile_l757_757885


namespace hyperbola_midpoint_l757_757145

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757145


namespace solution_set_of_inequality_l757_757553

theorem solution_set_of_inequality (a : ℝ) :
  (a > 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x < a + 1}) ∧
  (a < 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x > a + 1}) ∧
  (a = 1 → {x : ℝ | ax + 1 < a^2 + x} = ∅) := 
  sorry

end solution_set_of_inequality_l757_757553


namespace num_five_ruble_coins_l757_757333

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757333


namespace original_price_before_discounts_l757_757459

theorem original_price_before_discounts (P : ℝ) 
  (h : 0.75 * (0.75 * P) = 18) : P = 32 :=
by
  sorry

end original_price_before_discounts_l757_757459


namespace solve_quadratic_complete_square_l757_757007

theorem solve_quadratic_complete_square :
  ∃ b c : ℤ, (∀ x : ℝ, (x + b)^2 = c ↔ x^2 + 6 * x - 9 = 0) ∧ b + c = 21 := by
  sorry

end solve_quadratic_complete_square_l757_757007


namespace find_coordinates_of_P_l757_757637

noncomputable def pointP_minimizes_dot_product : Prop :=
  let OA := (2, 2)
  let OB := (4, 1)
  let AP x := (x - 2, -2)
  let BP x := (x - 4, -1)
  let dot_product x := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2
  ∃ x, (dot_product x = (x - 3) ^ 2 + 1) ∧ (∀ y, dot_product y ≥ dot_product x) ∧ (x = 3)

theorem find_coordinates_of_P : pointP_minimizes_dot_product :=
  sorry

end find_coordinates_of_P_l757_757637


namespace parallelogram_not_symmetrical_l757_757452

def is_symmetrical (shape : String) : Prop :=
  shape = "Circle" ∨ shape = "Rectangle" ∨ shape = "Isosceles Trapezoid"

theorem parallelogram_not_symmetrical : ¬ is_symmetrical "Parallelogram" :=
by
  sorry

end parallelogram_not_symmetrical_l757_757452


namespace area_of_OPF_eq_sqrt_2_div_2_l757_757631

noncomputable def area_of_triangle_OPF : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  if (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) then
    let base := dist O F
    let height := Real.sqrt 2
    (1 / 2) * base * height
  else
    0

theorem area_of_OPF_eq_sqrt_2_div_2 : 
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) →
  let base := dist O F
  let height := Real.sqrt 2
  area_of_triangle_OPF = Real.sqrt 2 / 2 := 
by 
  sorry

end area_of_OPF_eq_sqrt_2_div_2_l757_757631


namespace max_rooms_tour_l757_757807

-- Definitions based on conditions
def rooms := Fin 16  -- There are 16 rooms in the museum
def paintings : Finset rooms := {0, 2, 4, 6, 8, 10, 12, 14}  -- 8 rooms display paintings
def sculptures : Finset rooms := {1, 3, 5, 7, 9, 11, 13, 15}  -- 8 rooms exhibit sculptures

def adjacent (a b : rooms) : Prop :=
  (a.1 % 4 == b.1 % 4 ∧ (a.1 - b.1).abs = 4) ∨  -- vertically adjacent
  (a.1 / 4 == b.1 / 4 ∧ (a.1 - b.1).abs = 1)    -- horizontally adjacent

def tour (path : List rooms) : Prop :=
  path.head = some 0 ∧  -- starts at room A (room 0)
  path.last = some 1 ∧  -- ends at room B (room 1)
  (∀ i ∈ path.init.tail.zip (path.tail), adjacent i.fst i.snd) ∧  -- all consecutive rooms are adjacent
  (∀ i ∈ List.zip path (List.zip (path.init.tail) path.tail), 
    (i.snd.fst ∈ paintings ↔ i.snd.snd ∈ sculptures))  -- alternate between paintings and sculptures

theorem max_rooms_tour : ∃ (path : List rooms), tour path ∧ path.length = 15 := sorry

end max_rooms_tour_l757_757807


namespace period_f_mono_increasing_intervals_max_min_f_l757_757620

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin x ^ 2

theorem period_f : (∃ k : ℤ, ∀ x : ℝ, f (x + k * π) = f x) :=
sorry

theorem mono_increasing_intervals :
  ∀ k : ℤ, ∀ x1 x2 : ℝ, (k * π - π / 6 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ k * π + π / 3) → f x1 ≤ f x2 :=
sorry

theorem max_min_f :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = 2) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 1 / 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = 1 / 2) :=
sorry

end period_f_mono_increasing_intervals_max_min_f_l757_757620


namespace study_time_difference_l757_757016

def Kwame_study_hours := 2.5
def Connor_study_hours := 1.5
def Lexia_study_minutes := 97

def hours_to_minutes (h : ℕ) : ℕ := h * 60

theorem study_time_difference :
  hours_to_minutes Kwame_study_hours + hours_to_minutes Connor_study_hours - Lexia_study_minutes = 143 :=
sorry

end study_time_difference_l757_757016


namespace number_of_days_worked_l757_757569

-- Define the conditions
def hours_per_day := 8
def total_hours := 32

-- Define the proof statement
theorem number_of_days_worked : total_hours / hours_per_day = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_days_worked_l757_757569


namespace min_sum_is_25_over_72_l757_757260

theorem min_sum_is_25_over_72: ∀ (P Q R S : ℕ), 
  P ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  Q ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  R ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  S ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  P < Q ∧ Q < R ∧ R < S →
  (P * 9 + Q * 8) / (R * S) = 25 / 72 :=
begin
  sorry
end

end min_sum_is_25_over_72_l757_757260


namespace solve_inequality_l757_757777

def polynomial_fraction (x : ℝ) : ℝ :=
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem solve_inequality (x : ℝ) :
  -2 < polynomial_fraction x ∧ polynomial_fraction x < 2 ↔ 11.57 < x :=
sorry

end solve_inequality_l757_757777


namespace positive_difference_l757_757397

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l757_757397


namespace line_BQ_bisects_CD_at_midpoint_l757_757588

open EuclideanGeometry

variables (A B C D P M Q N O : Point) (circle : Circle) (diameter : line_segment A B)
          (trapezoid: trapezoid A B C D) (angle_NBC angle_PMQ : ℝ)

-- Assuming the conditions provided in the problem
axiom h_CD_perpendicular_AD : ⊥ CD AD
axiom h_CD_perpendicular_BC : ⊥ CD BC
axiom h_circle_diameter_AB : circle.diameter = diameter
axiom h_AD_intersect_circle_P : P ∈ (AD ∩ circle)
axiom h_P_distinct_A : P ≠ A
axiom h_tangent_P_intersect_CD_at_M : tangent (circle, P) ∩ CD = {M}
axiom h_second_tangent_M_circle_touch_Q : tangent (circle, M) ∩ circle = {Q}

-- The main statement to prove
theorem line_BQ_bisects_CD_at_midpoint :
  (BQ ∩ CD = {N}) → (N = midpoint C D) :=
begin
  sorry
end

end line_BQ_bisects_CD_at_midpoint_l757_757588


namespace second_year_students_selected_l757_757707

/-
Conditions:
- There are 450 students in the first year of high school.
- There are 750 students in the second year.
- There are 600 students in the third year.
- Each student has a probability of 0.02 of being selected.

Prove:
- The number of students that should be drawn from the second year is 15.
-/

def students_in_first_year : ℕ := 450
def students_in_second_year : ℕ := 750
def students_in_third_year : ℕ := 600
def selection_probability : ℝ := 0.02

theorem second_year_students_selected :
  real.to_nat (students_in_second_year * selection_probability) = 15 :=
by 
  sorry

end second_year_students_selected_l757_757707


namespace total_earnings_l757_757509

theorem total_earnings (L A J M : ℝ) 
  (hL : L = 2000) 
  (hA : A = 0.70 * L) 
  (hJ : J = 1.50 * A) 
  (hM : M = 0.40 * J) 
  : L + A + J + M = 6340 := 
  by 
    sorry

end total_earnings_l757_757509


namespace mod_inverse_sum_l757_757825

theorem mod_inverse_sum (a b : ℤ) (h1 : 5 * a ≡ 1 [MOD 17]) (h2 : 25 * b ≡ 1 [MOD 17]) :
  (a + b) % 17 = 14 :=
by
  sorry

end mod_inverse_sum_l757_757825


namespace solution_set_l757_757574

def f (x : ℝ) : ℝ :=
  if x > 1 then 2 else -1

theorem solution_set (x : ℝ) : (x + 2 * x * f (x + 1) > 5) ↔ (x < -5 ∨ x > 1) :=
by
  sorry

end solution_set_l757_757574


namespace diameter_increase_is_correct_l757_757269

noncomputable def increase_in_diameter 
  (original_distance : ℝ) 
  (new_distance : ℝ) 
  (original_diameter : ℝ) 
  (mile_to_inch : ℝ) : ℝ :=
  let r := original_diameter / 2 in
  let original_circumference := 2 * Mathlib.Data.Real.Basic.pi * r in
  let distance_per_rotation := original_circumference / mile_to_inch in
  let num_rotations := original_distance / distance_per_rotation in
  let new_distance_per_rotation := new_distance / num_rotations in
  let new_diameter := new_distance_per_rotation * mile_to_inch / Mathlib.Data.Real.Basic.pi in
  new_diameter - original_diameter

theorem diameter_increase_is_correct
  (d1 d2 : ℝ)
  (od : ℝ)
  (C_m : ℝ)
  (h1 : d1 = 120)
  (h2 : d2 = 118)
  (h3 : od = 26)
  (h4 : C_m = 63360) :
  increase_in_diameter d1 d2 od C_m = 0.67 :=
by
  rw [h1, h2, h3, h4]
  unfold increase_in_diameter
  conv_lhs {
    -- Reducing constants if needed
    norm_num
  }
  sorry

end diameter_increase_is_correct_l757_757269


namespace probability_more_heads_than_tails_l757_757833

/-- Julia flips 12 fair coins. The probability that she gets more heads than tails is equal 
    to 3172/8192. -/
theorem probability_more_heads_than_tails : 
  let p_heads := (1 : ℝ) / 2 in
  let p_tails := (1 : ℝ) / 2 in
  let n := 12 in
  let prob_more_heads := (3172 : ℝ) / 8192 in
  (∀ i, p_heads = p_tails = 1/2) →
  prob_more_heads =
  (∑ k in finset.Icc (n/2 + 1) n, nat.choose n k * (p_heads ^ k) * (p_tails ^ (n - k))) :=
sorry

end probability_more_heads_than_tails_l757_757833


namespace commission_rate_is_2_5_l757_757512

-- Define the given conditions
def CommissionEarned : ℝ := 21
def TotalSales : ℝ := 840

-- Define the formula for the commission rate
def CommissionRate (commissionEarned totalSales : ℝ) : ℝ := (commissionEarned / totalSales) * 100

-- State the main theorem
theorem commission_rate_is_2_5 :
  CommissionRate CommissionEarned TotalSales = 2.5 := 
by
  sorry

end commission_rate_is_2_5_l757_757512


namespace james_veg_consumption_l757_757712

-- Define the given conditions in Lean
def asparagus_per_day : ℝ := 0.25
def broccoli_per_day : ℝ := 0.25
def days_in_week : ℝ := 7
def weeks : ℝ := 2
def kale_per_week : ℝ := 3

-- Define the amount of vegetables (initial, doubled, and added kale)
def initial_veg_per_day := asparagus_per_day + broccoli_per_day
def initial_veg_per_week := initial_veg_per_day * days_in_week
def double_veg_per_week := initial_veg_per_week * weeks
def total_veg_per_week_after_kale := double_veg_per_week + kale_per_week

-- Statement of the proof problem
theorem james_veg_consumption :
  total_veg_per_week_after_kale = 10 := by 
  sorry

end james_veg_consumption_l757_757712


namespace greatest_visible_sum_l757_757811

/-- 
Three cubes with distinct number patterns are each formed and then stacked on top of each other. 
The number patterns for three cubes are:
- Cube A: 1, 2, 3, 4, 5, 6
- Cube B: 7, 8, 9, 10, 11, 12
- Cube C: 13, 14, 15, 16, 17, 18
They are stacked in such a way that the 13 visible numbers have the greatest possible sum. 
Prove that the greatest possible sum of the 13 visible numbers is 138.
-/
theorem greatest_visible_sum 
  (A : Finset ℕ) (B : Finset ℕ) (C : Finset ℕ) 
  (hA : A = {1, 2, 3, 4, 5, 6}) 
  (hB : B = {7, 8, 9, 10, 11, 12}) 
  (hC : C = {13, 14, 15, 16, 17, 18}) 
  : (20 + 38 + 80) = 138 := by
  have hA_visible : {2, 3, 4, 5, 6} ⊆ A := by rw hA; exact Finset.subset_insert 1 _
  have hB_visible : {8, 9, 10, 11} ⊆ B := by rw hB; exact Finset.subset_insert 7 _
  have hC_visible : {14, 15, 16, 17, 18} ⊆ C := by rw hC; exact Finset.subset_insert 13 _
  have sumA := 2 + 3 + 4 + 5 + 6
  have sumB := 8 + 9 + 10 + 11
  have sumC := 14 + 15 + 16 + 17 + 18
  have sum_total := sumA + sumB + sumC
  show sum_total = 138 from calc
    sum_total = (20 + 38 + 80) := by sorry
    ... = 138 := eq.refl 138

end greatest_visible_sum_l757_757811


namespace graph_triangle_bound_l757_757763

theorem graph_triangle_bound (n m : ℕ) (G : SimpleGraph (Fin n)) (h_edges : G.edge_count = m) :
  ∃ t : ℕ, t ≥ (m * (4 * m - n^2)) / (3 * n) :=
by sorry

end graph_triangle_bound_l757_757763


namespace petya_five_ruble_coins_count_l757_757320

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757320


namespace find_strictly_increasing_intervals_l757_757539

noncomputable def strictly_increasing_interval (k : ℤ) : Set ℝ :=
  {x | (7 * Real.pi / 12 + k * Real.pi) ≤ x ∧ x ≤ (13 * Real.pi / 12 + k * Real.pi)}

theorem find_strictly_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, (strictly_increasing_interval k).indicator 1 x = 1 ↔
    (7 * Real.pi / 12 + k * Real.pi) ≤ x ∧ x ≤ (13 * Real.pi / 12 + k * Real.pi) :=
by {
  intros,
  sorry
}

end find_strictly_increasing_intervals_l757_757539


namespace eq_circle_C_eq_tangent_line_eq_area_OAB_l757_757629

section geometric_problem

variables {x y : ℝ}

-- Definitions based on conditions
def line_l := 4 * x + 3 * y - 8 = 0
def circle_C (a : ℝ) := x^2 + y^2 - a * x = 0
def point_P := (1, real.sqrt 3)
def center_C := (2: ℝ, 0: ℝ)
def distance_from_origin_to_l := 8 / real.sqrt(4^2 + 3^2)
def length_AB := 4
def area_OAB := (1/2) * length_AB * distance_from_origin_to_l

-- Proof statements
theorem eq_circle_C : circle_C 4 := by sorry
theorem eq_tangent_line : ∀ P ∈ circle_C 4, point_P = P → (x - real.sqrt 3 * y + 2 = 0) := by sorry
theorem eq_area_OAB : area_OAB = 16 / 5 := by sorry

end geometric_problem

end eq_circle_C_eq_tangent_line_eq_area_OAB_l757_757629


namespace conjugate_of_quotient_l757_757973

def conjugate (z : ℂ) : ℂ := z.re - z.im * Complex.i

theorem conjugate_of_quotient : conjugate (i / (1 - i)) = -1/2 - (1/2) * Complex.i := sorry

end conjugate_of_quotient_l757_757973


namespace petya_five_ruble_coins_l757_757304

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757304


namespace sequence_property_l757_757532

theorem sequence_property
  (b : ℝ) (h₀ : b > 0)
  (u : ℕ → ℝ)
  (h₁ : u 1 = b)
  (h₂ : ∀ n ≥ 1, u (n + 1) = 1 / (2 - u n)) :
  u 10 = (4 * b - 3) / (6 * b - 5) :=
by
  sorry

end sequence_property_l757_757532


namespace hayden_ironing_time_l757_757520

def shirt : ℕ := 5
def pants : ℕ := 3
def days_per_week : ℕ := 5
def weeks : ℕ := 4

theorem hayden_ironing_time : 
  let daily_minutes := shirt + pants in
  let weekly_minutes := daily_minutes * days_per_week in
  let total_minutes := weekly_minutes * weeks in
  total_minutes = 160 := 
by 
  -- unwrapping the let bindings
  let daily_minutes := shirt + pants
  let weekly_minutes := daily_minutes * days_per_week
  let total_minutes := weekly_minutes * weeks
  -- proving the theorem
  sorry

end hayden_ironing_time_l757_757520


namespace larger_integer_value_l757_757391

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l757_757391


namespace num_unbounded_sequences_450_l757_757535

noncomputable def f1 (n : ℕ) : ℕ :=
if n = 1 then 1
else let prime_factors : List (ℕ × ℕ) := (uniqueFactorizationMonoid.factors n).toList.map (λ p, (p, (uniqueFactorizationMonoid.count p (uniqueFactorizationMonoid.factors n))));
foldr (λ (pe : ℕ × ℕ) (acc : ℕ), acc * (2 * pe.1 + 1) ^ pe.2) 1 prime_factors

def f (m n : ℕ) : ℕ :=
nat.rec n (λ m prev, f1 prev) m

def unbound_sequence_values (n : ℕ) (bound : ℕ) : bool :=
∃ m : ℕ, f m n > bound

def count_unbounded_sequences (bound : ℕ) : ℕ :=
(range bound).count (λ n, unbound_sequence_values (n + 1) bound)

theorem num_unbounded_sequences_450 : count_unbounded_sequences 450 = 21 := 
sorry

end num_unbounded_sequences_450_l757_757535


namespace find_larger_integer_l757_757387

noncomputable def larger_integer (a b : ℕ) : Prop :=
  a * b = 189 ∧ (b = (7 * a) / 3⁷) / 3

theorem find_larger_integer (a b : ℕ) (h1 : a * b = 189) (h2 : a * 7 = 3 * b) :
  b = 21 :=
by
  sorry

end find_larger_integer_l757_757387


namespace midpoint_of_hyperbola_l757_757165

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757165


namespace midpoint_of_line_segment_on_hyperbola_l757_757026

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757026


namespace integral_sin_power_l757_757921

theorem integral_sin_power (h : ∫ x in 0..π, 2^4 * sin x ^ 8 = ∫ x in 0..π, 35 / 8 * π ) :
  true := by
  sorry

end integral_sin_power_l757_757921


namespace midpoint_of_hyperbola_segment_l757_757087

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757087


namespace midpoint_on_hyperbola_l757_757076

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757076


namespace part_a_total_time_part_b_average_time_part_c_probability_l757_757290

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l757_757290


namespace cost_of_candies_l757_757868

variable (cost_per_box : ℚ) (candies_per_box : ℕ) (total_candies : ℕ)
variables (h1 : cost_per_box = 7.5) (h2 : candies_per_box = 30) (h3 : total_candies = 450)

theorem cost_of_candies : 15 * 7.50 = 112.50 :=
by
  have boxes_needed : ℚ := total_candies / candies_per_box
  have total_cost : ℚ := boxes_needed * cost_per_box
  show total_cost = 112.50
  sorry

end cost_of_candies_l757_757868


namespace parabola_focus_l757_757991

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757991


namespace part_a_l757_757856

theorem part_a (a b c : ℕ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 := 
by sorry

end part_a_l757_757856


namespace hyperbola_midpoint_exists_l757_757239

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757239


namespace distance_origin_to_line_l757_757826

-- Define the origin point
def origin : Point := ⟨0, 0⟩

-- Define the line equation coefficients
def line (x y : ℝ) : ℝ := x + sqrt 3 * y - 2

-- Define the distance function from point to line
def distance_from_point_to_line (p : Point) : ℝ :=
  abs (line p.1 p.2) / sqrt (1^2 + (sqrt 3)^2)

-- Define the distance from the origin to the line
def distance_from_origin_to_line : ℝ := distance_from_point_to_line origin

-- Theorem statement
theorem distance_origin_to_line : distance_from_origin_to_line = 1 :=
  sorry

end distance_origin_to_line_l757_757826


namespace square_free_odd_integers_count_l757_757654

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757654


namespace probability_of_region_C_l757_757476

theorem probability_of_region_C (pA pB pC : ℚ) 
  (h1 : pA = 1/2) 
  (h2 : pB = 1/5) 
  (h3 : pA + pB + pC = 1) : 
  pC = 3/10 := 
sorry

end probability_of_region_C_l757_757476


namespace polynomial_root_constraints_l757_757968

theorem polynomial_root_constraints (P : Polynomial ℝ) (n : ℕ) (an : ℕ → ℤ)
  (hP : P = (Factorial factorial n) * (X ^ n) + ∑ i in Finset.range n, (an i) * (X ^ i) + (-1) ^ n * (n + 1)) :
  (∃ (roots : Fin n → ℝ), ∀ k, k ∈ Finset.range n → k ≤ roots k ∧ roots k ≤ k + 1) ↔ n = 1 ∧ P = X - 2 :=
sorry

end polynomial_root_constraints_l757_757968


namespace parabola_focus_l757_757993

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757993


namespace bryan_bought_4_pairs_of_pants_l757_757522

def number_of_tshirts : Nat := 5
def total_cost : Nat := 1500
def cost_per_tshirt : Nat := 100
def cost_per_pants : Nat := 250

theorem bryan_bought_4_pairs_of_pants : (total_cost - number_of_tshirts * cost_per_tshirt) / cost_per_pants = 4 := by
  sorry

end bryan_bought_4_pairs_of_pants_l757_757522


namespace number_of_square_free_odds_l757_757642

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757642


namespace hyperbola_midpoint_l757_757116

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757116


namespace congruence_solution_exists_l757_757734

theorem congruence_solution_exists {p n a : ℕ} (hp : Prime p) (hn : n % p ≠ 0) (ha : a % p ≠ 0)
  (hx : ∃ x : ℕ, x^n % p = a % p) :
  ∀ r : ℕ, ∃ x : ℕ, x^n % (p^(r + 1)) = a % (p^(r + 1)) :=
by
  intros r
  sorry

end congruence_solution_exists_l757_757734


namespace number_divided_by_three_l757_757447

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l757_757447


namespace prism_definition_l757_757862

-- Definitions
structure Polyhedron where
  faces : List (List (Fin 4 → Point))
  is_parallel : ∀ (f1 f2 : List (Fin 4 → Point)), f1 ∈ faces → f2 ∈ faces → Bool
  has_parallel_edges : ∀ (f : List (Fin 4 → Point)), f ∈ faces → ∀ (i j : Fin 4), i ≠ j → (∃ (v : Vector), Vector.is_parallel (f i) (f j)) 

-- Conditions
def is_prism (p : Polyhedron) : Prop :=
  ∃ (f1 f2 : List (Fin 4 → Point)), f1 ∈ p.faces ∧ f2 ∈ p.faces ∧
  p.is_parallel f1 f2 = true ∧ 
  ∀ (f : List (Fin 4 → Point)), f ∈ p.faces → ∀ (i j : Fin 4), i ≠ j → 
  p.has_parallel_edges f i j

-- Problem Statement
theorem prism_definition (p : Polyhedron) : is_prism p ↔ 
  (∃ (f1 f2 : List (Fin 4 → Point)), f1 ∈ p.faces ∧ f2 ∈ p.faces ∧
  p.is_parallel f1 f2 = true ∧ 
  ∀ (f : List (Fin 4 → Point)), f ∈ p.faces → (∀ (i j : Fin 4), i ≠ j → 
  p.has_parallel_edges f i j)) := by
  sorry

end prism_definition_l757_757862


namespace correct_midpoint_l757_757208

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757208


namespace gcd_of_sum_of_evens_l757_757858

noncomputable def x : ℕ := (25 / 2) * (14 + 62)
noncomputable def y : ℕ := (62 - 14) / 2 + 1

theorem gcd_of_sum_of_evens :
  let x := 950 in
  let y := 25 in
  Nat.gcd x y = 25 :=
by
  let x := 950
  let y := 25
  sorry

end gcd_of_sum_of_evens_l757_757858


namespace square_free_odd_integers_count_l757_757651

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757651


namespace debony_extra_time_needed_l757_757456

-- Conditions
def driving_time_minutes := 45
def driving_speed_mph := 40
def biking_distance_fraction := 0.8
def min_biking_speed_mph := 12

-- Derived quantities
def driving_time_hours := driving_time_minutes / 60
def driving_distance_miles := driving_time_hours * driving_speed_mph
def biking_distance_miles := biking_distance_fraction * driving_distance_miles
def biking_time_hours := biking_distance_miles / min_biking_speed_mph
def biking_time_minutes := biking_time_hours * 60
def extra_time_needed := biking_time_minutes - driving_time_minutes

-- Theorem
theorem debony_extra_time_needed :
  extra_time_needed = 75 := by
  sorry

end debony_extra_time_needed_l757_757456


namespace correct_midpoint_l757_757210

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757210


namespace product_sequence_l757_757851

theorem product_sequence : 
  ∃ (a : ℕ), a = ∏ i in (finset.range 10).map (λ n, n + 1), 
    (if i = 1 then 2001 else (i / a (i - 2))) ∧ a = 3840 :=
sorry

end product_sequence_l757_757851


namespace quadrilateral_perpendicular_diagonals_l757_757580

-- Define the given condition for the quadrilateral
def is_cyclic_quadrilateral (a b c d R : ℝ) :=
  a^2 + b^2 + c^2 + d^2 = 8 * R^2

-- Define the statement we wish to prove
theorem quadrilateral_perpendicular_diagonals
  (a b c d R : ℝ)
  (h : is_cyclic_quadrilateral a b c d R) :
  ⦃ABCD : Type⦄ [is_quadrilateral ABCD a b c d] [is_circumradius ABCD R] → 
  (diagonals_perpendicular ABCD) :=
sorry

end quadrilateral_perpendicular_diagonals_l757_757580


namespace find_x_l757_757355

theorem find_x (x : ℕ) :
  (3 * x > 91 ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ ¬(x < 120) ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ ¬(x < 27) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7)) →
  x = 9 :=
sorry

end find_x_l757_757355


namespace area_A_area_A_l757_757453

variables {A B C A' B' C': Type} [acute_triangle : triangle A B C] [midpoint_B' : is_midpoint B' A C]
variable {point_A'_on_BC : on_side A' B C}
variable {point_C'_on_AB : on_side C' A B}

-- Theorem 1: The area of triangle A'B'C' is less than or equal to half the area of triangle ABC.
theorem area_A'B'C'_le_half_area_ABC :
  area (triangle A' B' C') ≤ (1 / 2) * area (triangle A B C) :=
sorry

-- Theorem 2: The area of triangle A'B'C' is exactly one-fourth the area of triangle ABC if and only if
-- at least one point A' or C' is the midpoint of sides BC or AB, respectively.
theorem area_A'B'C'_eq_one_fourth_area_ABC_iff_midpoint_condition :
  area (triangle A' B' C') = (1 / 4) * area (triangle A B C) ↔
  (is_midpoint A' B C) ∨ (is_midpoint C' A B) :=
sorry

end area_A_area_A_l757_757453


namespace gcf_60_90_150_l757_757827

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l757_757827


namespace find_tangent_line_at_neg1_l757_757548

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x^2 + 3

theorem find_tangent_line_at_neg1 :
  let x := -1
  let m := 4 * x
  let y := 2 * x^2 + 3
  let tangent := y + m * (x - x)
  tangent = -4 * x + 1 :=
by
  sorry

end find_tangent_line_at_neg1_l757_757548


namespace general_term_correct_sum_first_n_terms_correct_l757_757632

open Nat

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n : ℕ, 0 < n → a (n + 1) = 1 / 2 * a n + (2 * n + 3) / 2^(n + 1)

def general_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n^2 + 2 * n - 2) / 2^n

theorem general_term_correct (a : ℕ → ℝ) (h : seq a) (n : ℕ) :
  a n = general_term a n :=
sorry

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ k in range (n + 1), a k

theorem sum_first_n_terms_correct (a : ℕ → ℝ) (S : ℕ → ℝ) (h : seq a) (hS : sum_first_n_terms a S) (n : ℕ) :
  S n = 8 - (n^2 + 6 * n + 8) / 2^n :=
sorry

end general_term_correct_sum_first_n_terms_correct_l757_757632


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757283

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757283


namespace geometric_sequence_a2_l757_757679

theorem geometric_sequence_a2 (a1 a2 a3 : ℝ) (h1 : 1 * (1/a1) = a1)
  (h2 : a1 * (1/a2) = a2) (h3 : a2 * (1/a3) = a3) (h4 : a3 * (1/4) = 4)
  (h5 : a2 > 0) : a2 = 2 := sorry

end geometric_sequence_a2_l757_757679


namespace midpoint_of_hyperbola_segment_l757_757089

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757089


namespace collinear_of_similar_and_ratios_l757_757399

open BigOperators

-- Given similar triangles' coordinates (you might need your own setup for geometry context)
variables {α : Type*} [linear_ordered_field α] [metric_space α]

structure Triangle (α : Type*) := 
(A B C : α)

def similar (T₁ T₂ : Triangle α) : Prop :=
  ∃ k : α, k ≠ 0 ∧
    dist T₁.A T₁.B / dist T₂.A T₂.B = k ∧
    dist T₁.B T₁.C / dist T₂.B T₂.C = k ∧
    dist T₁.C T₁.A / dist T₂.C T₂.A = k

-- Define the homothety (scaling transformation)
def homothety (k : α) (O A : α) : α :=
O + k * (A - O)

variables (ABC A1B1C1 : Triangle α)
variables (BC B1C1 : α)
variables (k : α)

-- Define the conditions
axiom similarity : similar ABC A1B1C1
axiom A'_ratio : (λ AA' : α, homothety k ABC.A A1B1C1.A' = A'_ratio)
axiom B'_ratio : (λ BB' : α, homothety k ABC.B A1B1C1.B' = B'_ratio)
axiom C'_ratio : (λ CC' : α, homothety k ABC.C A1B1C1.C' = C'_ratio)

-- Define the points A', B' and C' using the homothety construction
def A' : α := homothety k ABC.A A1B1C1.A'
def B' : α := homothety k ABC.B A1B1C1.B'
def C' : α := homothety k ABC.C A1B1C1.C'

-- The theorem to prove collinearity
theorem collinear_of_similar_and_ratios :
  collinear A' B' C' := sorry

end collinear_of_similar_and_ratios_l757_757399


namespace clock_angle_at_3_20_l757_757909

theorem clock_angle_at_3_20 : 
  let h := 3
  let m := 20
  abs ((60 * h - 11 * m) / 2) = 20 := sorry

end clock_angle_at_3_20_l757_757909


namespace biographies_increased_by_388_89_percent_l757_757460

noncomputable def percentageIncreaseInBiographies (B N : ℝ) : ℝ :=
  let newBiographies := 0.20 * B + N
  let newTotalBooks := B + N
  if h : newBiographies = 0.55 * newTotalBooks then
    (N / (0.20 * B)) * 100
  else
    0 -- this handles the case where the condition does not hold, though should not happen

theorem biographies_increased_by_388_89_percent (B N : ℝ) (h : N = (7/9) * B) 
  (h_new_collection : 0.20 * B + N = 0.55 * (B + N))  :
  percentageIncreaseInBiographies B N ≈ 388.89 :=
by
  -- Sorry to skip the proof
  sorry

end biographies_increased_by_388_89_percent_l757_757460


namespace ming_dynasty_wine_problem_l757_757696

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end ming_dynasty_wine_problem_l757_757696


namespace tan_neg_405_eq_neg1_l757_757932

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l757_757932


namespace call_processing_ratio_l757_757478

variables (A B C : ℝ)
variable (total_calls : ℝ)
variable (calls_processed_by_A_per_member calls_processed_by_B_per_member : ℝ)

-- Given conditions
def team_A_agents_ratio : Prop := A = (5 / 8) * B
def team_B_calls_ratio : Prop := calls_processed_by_B_per_member * B = (4 / 7) * total_calls
def team_A_calls_ratio : Prop := calls_processed_by_A_per_member * A = (3 / 7) * total_calls

-- Proving the ratio of calls processed by each member
theorem call_processing_ratio
    (hA : team_A_agents_ratio A B)
    (hB_calls : team_B_calls_ratio B total_calls calls_processed_by_B_per_member)
    (hA_calls : team_A_calls_ratio A total_calls calls_processed_by_A_per_member) :
  calls_processed_by_A_per_member / calls_processed_by_B_per_member = 6 / 5 :=
by
  sorry

end call_processing_ratio_l757_757478


namespace hyperbola_midpoint_exists_l757_757238

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757238


namespace net_hourly_rate_correct_l757_757487

noncomputable def net_hourly_rate
    (hours : ℕ) 
    (speed : ℕ) 
    (fuel_efficiency : ℕ) 
    (earnings_per_mile : ℝ) 
    (cost_per_gallon : ℝ) 
    (distance := speed * hours) 
    (gasoline_used := distance / fuel_efficiency) 
    (earnings := earnings_per_mile * distance) 
    (cost_of_gasoline := cost_per_gallon * gasoline_used) 
    (net_earnings := earnings - cost_of_gasoline) : ℝ :=
  net_earnings / hours

theorem net_hourly_rate_correct : 
  net_hourly_rate 3 45 25 0.6 1.8 = 23.76 := 
by 
  unfold net_hourly_rate
  norm_num
  sorry

end net_hourly_rate_correct_l757_757487


namespace expected_bullets_correct_l757_757499

noncomputable def P_hit := 0.6
noncomputable def P_miss := 1 - P_hit
def num_bullets := 4

open Probability

def remaining_bullets_distribution : List (ℕ × ℝ) :=
  [(3, P_hit),
   (2, P_miss * P_hit),
   (1, P_miss^2 * P_hit),
   (0, P_miss^3 * P_hit)]

def expected_remaining_bullets (dist : List (ℕ × ℝ)) : ℝ :=
  dist.foldl (λ acc (x : ℕ × ℝ), acc + x.1 * x.2) 0

theorem expected_bullets_correct :
  expected_remaining_bullets remaining_bullets_distribution = 2.376 :=
by
  sorry

end expected_bullets_correct_l757_757499


namespace complex_norm_one_conjugate_in_set_l757_757722

variable (n : ℕ) (A : FinSet ℂ)
variable (z : ℕ → ℂ)
variable (h1 : 2 ≤ n)
variable (h2 : ∀ i : Fin n, (Finset.image (λ j => z i * z j) (Finset.range n)) = A)
variable (h3 : A = Finset.range n .map z)

-- Prove |z_i| = 1 for all i ∈ {1, 2, ..., n}
theorem complex_norm_one (i : Fin n) : ∥ z i ∥ = 1 :=
sorry

-- Prove that if z ∈ A, then conjugate(z) ∈ A
theorem conjugate_in_set (z' : ℂ) (hz : z' ∈ A) : conj z' ∈ A :=
sorry

end complex_norm_one_conjugate_in_set_l757_757722


namespace group_2009_in_A45_l757_757386

-- Define the grouping as a function
def group (n : ℕ) : ℕ → Prop
| k := let cumulative_size := n * n in 
       cumulative_size - (2 * n - 1) < k ∧ k <= cumulative_size

-- Formalizing the main problem
theorem group_2009_in_A45 :
  ∃ n, group n 2009 ↔ n = 45 :=
begin
  -- This will be proved by showing that 2009 belongs to the 45th group
  sorry
end

end group_2009_in_A45_l757_757386


namespace number_of_five_ruble_coins_l757_757326

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757326


namespace hyperbola_midpoint_l757_757122

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757122


namespace part_one_part_two_l757_757624

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 4 - x

-- Problem set (I)
theorem part_one (x : ℝ) : inequality_condition x ↔ (x ≤ -3 ∨ x ≥ 1) :=
sorry

-- Define range conditions for a and b
def range_condition (a b : ℝ) : Prop := a ≥ 3 ∧ b ≥ 3

-- Problem set (II)
theorem part_two (a b : ℝ) (h : range_condition a b) : 2 * (a + b) < a * b + 4 :=
sorry

end part_one_part_two_l757_757624


namespace number_divided_by_3_equals_subtract_3_l757_757436

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l757_757436


namespace minimize_s_over_r_perimeter_l757_757717

-- Define the conditions
def T : ℕ := 7
def BC : ℕ := 100 * T - 4
def a (ABC : Triangle) : ℕ := some_value -- placeholder
def b : ℕ := BC
def c (ABC : Triangle) : ℕ := some_value -- placeholder, satisfying a^2 + BC^2 = c^2
def s (ABC : Triangle) : ℕ := (a ABC + b + c ABC) / 2
def r (ABC : Triangle) : ℕ := some_value -- placeholder for inradius

-- Define the main theorem statement
theorem minimize_s_over_r_perimeter : ∀ (ABC : Triangle), 
  (right_angle ABC) ∧ (integer_sides ABC) ∧ (side_length BC) → 
  perimeter ABC = 1624 :=
by
  intros
  sorry

noncomputable theory -- needed as the problem involves non-computable steps like finding divisors

end minimize_s_over_r_perimeter_l757_757717


namespace distribute_tickets_l757_757956

/-- The number of different ways to distribute 3 different movie tickets among 3 out of 10 people, with each person receiving one ticket, is 720. -/
theorem distribute_tickets : 
  ∃ (ways : ℕ), ways = 10 * 9 * 8 ∧ ways = 720 :=
by
  existsi 720
  split
  -- proof omitted
  sorry
  rfl

end distribute_tickets_l757_757956


namespace strap_pieces_l757_757808

/-
  Given the conditions:
  1. The sum of the lengths of the two straps is 64 cm.
  2. The longer strap is 48 cm longer than the shorter strap.
  
  Prove that the number of pieces of strap that equal the length of the shorter strap 
  that can be cut from the longer strap is 7.
-/

theorem strap_pieces (S L : ℕ) (h1 : S + L = 64) (h2 : L = S + 48) :
  L / S = 7 :=
by
  sorry

end strap_pieces_l757_757808


namespace find_y_coordinate_l757_757491

def point1 : ℝ × ℝ × ℝ := (3, 3, 2)
def point2 : ℝ × ℝ × ℝ := (7, 2, -1)
def target_x : ℝ := 6

theorem find_y_coordinate :
  ∃ y : ℝ, ∃ t : ℝ, (point1.1 + t * (point2.1 - point1.1) = target_x ∧ 
                     y = point1.2 + t * (point2.2 - point1.2) ∧
                     point1.3 + t * (point2.3 - point1.3) = 2 - 3 * (3 / 4)) ∧
                     y = 2.25 :=
begin
  sorry
end

end find_y_coordinate_l757_757491


namespace parabola_focus_l757_757982

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757982


namespace find_annual_interest_rate_l757_757874

theorem find_annual_interest_rate (P0 P1 P2 : ℝ) (r1 r : ℝ) :
  P0 = 12000 →
  r1 = 10 →
  P1 = P0 * (1 + (r1 / 100) / 2) →
  P1 = 12600 →
  P2 = 13260 →
  P1 * (1 + (r / 200)) = P2 →
  r = 10.476 :=
by
  intros hP0 hr1 hP1 hP1val hP2 hP1P2
  sorry

end find_annual_interest_rate_l757_757874


namespace probability_at_most_one_girl_l757_757878

theorem probability_at_most_one_girl (boys girls : ℕ) (total_selected : ℕ)
  (hb : boys = 3) (hg : girls = 2) (hts : total_selected = 2) : 
  let n := Nat.choose (boys + girls) total_selected in
  let m := Nat.choose boys total_selected + (Nat.choose boys 1 * Nat.choose girls 1) in
  m / n = 9 / 10 :=
by 
  exact sorry

end probability_at_most_one_girl_l757_757878


namespace can_be_midpoint_of_AB_l757_757178

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757178


namespace _l757_757690

-- Define the problem conditions and the theorem statement
noncomputable def triangle_collinear (A B C D X M N P O : Point) 
  (h_angle_C : ∠ C = 45)
  (h_AD_altitude : Altitude A D B C)
  (h_X_on_AD : OnLine X A D)
  (h_angle_XBC : ∠XBC = 90 - ∠B)
  (h_D_circ : CircleIntersection A D C M)
  (h_CX_circ : CircleIntersection C X A N)
  (h_tangent : TangentToCircleAt M P)
  (h_tangent_AN : TangentLineIntersectionAtTangent A N P X)
  (h_circumcenter : IsCircumcenter O A B C)
  : Collinear P B O :=
begin
  sorry
end

-- Definitions and axioms used in the problem
constant Point : Type
constant Angle : Type
constant Line : Type

constant ∠ : Point → Angle
constant OnLine : Point → Point → Point → Prop
constant Altitude : Point → Point → Point → Point → Prop
constant CircleIntersection : Point → Point → Point → Point → Prop
constant TangentToCircleAt : Point → Point → Prop
constant TangentLineIntersectionAtTangent : Point → Point → Point → Point → Prop
constant IsCircumcenter : Point → Point → Point → Point → Prop
constant Collinear : Point → Point → Point → Prop

-- Definitions for specific angles    
axiom angle_C_45 : ∠ C = 45
axiom angle_XBC : ∠XBC = 90 - ∠B

end _l757_757690


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757279

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757279


namespace part_a_part_b_l757_757854

-- Definitions for part a: 
structure Square :=
  (rectangles : Set Rectangle)

def is_chain (S : Side) (K : Set Rectangle) : Prop :=
  (∀ r ∈ K, projects_to_side r S) ∧
  (∀ p : Point, covers_side S p → ∃! r ∈ K, p ∈ projection r S)

theorem part_a (sq : Square) (rect1 rect2 : Rectangle) (H1 : rect1 ∈ sq.rectangles) (H2 : rect2 ∈ sq.rectangles) :
  ∃ K, is_chain some_side K ∧ rect1 ∈ K ∧ rect2 ∈ K := sorry

-- Definitions for part b:
structure Cube :=
  (parallelepipeds : Set Parallelepiped)

def is_chain_cube (E : Edge) (K : Set Parallelepiped) : Prop :=
  (∀ p ∈ K, projects_to_edge p E) ∧
  (∀ p : Point, covers_edge E p → ∃! p ∈ K, p ∈ projection p E)

theorem part_b (cu : Cube) (paral1 paral2 : Parallelepiped) (H1 : paral1 ∈ cu.parallelepipeds) (H2 : paral2 ∈ cu.parallelepipeds) :
  ∃ K, is_chain_cube some_edge K ∧ paral1 ∈ K ∧ paral2 ∈ K := sorry

end part_a_part_b_l757_757854


namespace second_square_larger_l757_757586

variables (a b c m : ℝ) (m_pos : 0 < m) (m_eq : m = a * b / c) (c_eq : c = Real.sqrt (a^2 + b^2))

def x1 := (a * b / (m + c))
def x2 := (a * b / (a + b))

theorem second_square_larger (h : m + c > a + b) : x1 < x2 := 
by
  unfold x1 x2
  sorry

end second_square_larger_l757_757586


namespace new_students_count_l757_757785

theorem new_students_count (x : ℕ) (avg_age_group new_avg_age avg_new_students : ℕ)
  (h1 : avg_age_group = 14) (h2 : new_avg_age = 15) (h3 : avg_new_students = 17)
  (initial_students : ℕ) (initial_avg_age : ℕ)
  (h4 : initial_students = 10) (h5 : initial_avg_age = initial_students * avg_age_group)
  (h6 : new_avg_age * (initial_students + x) = initial_avg_age + (x * avg_new_students)) :
  x = 5 := 
by
  sorry

end new_students_count_l757_757785


namespace midpoint_of_hyperbola_l757_757057

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757057


namespace clubsuit_sum_l757_757557

-- Define the function ♣(x), which is the average of x^3 and x^4
def clubsuit (x : ℝ) : ℝ := (x^3 + x^4) / 2

-- State the theorem that we want to prove
theorem clubsuit_sum :
  clubsuit 2 + clubsuit 3 = 66 :=
  by sorry

end clubsuit_sum_l757_757557


namespace period_is_3_years_l757_757489

def gain_of_B_per_annum (principal : ℕ) (rate_A rate_B : ℚ) : ℚ := 
  (rate_B - rate_A) * principal

def period (principal : ℕ) (rate_A rate_B : ℚ) (total_gain : ℚ) : ℚ := 
  total_gain / gain_of_B_per_annum principal rate_A rate_B

theorem period_is_3_years :
  period 1500 (10 / 100) (11.5 / 100) 67.5 = 3 :=
by
  sorry

end period_is_3_years_l757_757489


namespace moles_HCl_combination_l757_757970

-- Define the conditions:
def moles_HCl (C5H12O: ℕ) (H2O: ℕ) : ℕ :=
  if H2O = 18 then 18 else 0

-- The main statement to prove:
theorem moles_HCl_combination :
  moles_HCl 1 18 = 18 :=
sorry

end moles_HCl_combination_l757_757970


namespace increase_500_by_30_l757_757471

theorem increase_500_by_30 :
  let original := 500
  let percent := 0.30
  let increase := original * percent
  let final := original + increase
  final = 650 :=
by {
  let original := 500
  let percent := 0.30
  let increase := original * percent
  let final := original + increase
  have h : final = 650 := sorry,
  exact h
}

end increase_500_by_30_l757_757471


namespace odd_square_free_count_l757_757661

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757661


namespace width_of_lawn_is_60_l757_757889

-- Define the problem conditions in Lean
def length_of_lawn : ℕ := 70
def road_width : ℕ := 10
def total_road_cost : ℕ := 3600
def cost_per_sq_meter : ℕ := 3

-- Define the proof problem
theorem width_of_lawn_is_60 (W : ℕ) 
  (h1 : (road_width * W) + (road_width * length_of_lawn) - (road_width * road_width) 
        = total_road_cost / cost_per_sq_meter) : 
  W = 60 := 
by 
  sorry

end width_of_lawn_is_60_l757_757889


namespace midpoint_hyperbola_l757_757100

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757100


namespace proof_time_to_run_square_field_l757_757458

def side : ℝ := 40
def speed_kmh : ℝ := 9
def perimeter (side : ℝ) : ℝ := 4 * side

noncomputable def speed_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def time_to_run (perimeter : ℝ) (speed_mps : ℝ) : ℝ := perimeter / speed_mps

theorem proof_time_to_run_square_field :
  time_to_run (perimeter side) (speed_mps speed_kmh) = 64 :=
by
  sorry

end proof_time_to_run_square_field_l757_757458


namespace midpoint_of_hyperbola_segment_l757_757085

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757085


namespace molecular_weight_equimolar_mixture_l757_757954

def molecular_weight_carbon : ℝ := 12.01
def molecular_weight_hydrogen : ℝ := 1.008
def molecular_weight_oxygen : ℝ := 16.00

def molecular_weight_acetic_acid : ℝ :=
  2 * molecular_weight_carbon + 
  4 * molecular_weight_hydrogen + 
  2 * molecular_weight_oxygen

def molecular_weight_ethanol : ℝ :=
  2 * molecular_weight_carbon + 
  6 * molecular_weight_hydrogen + 
  molecular_weight_oxygen 

def average_molecular_weight_mixture (mw1 mw2 : ℝ) : ℝ :=
  (mw1 + mw2) / 2

theorem molecular_weight_equimolar_mixture :
    average_molecular_weight_mixture molecular_weight_acetic_acid molecular_weight_ethanol = 53.060 :=
by
  -- Here, you would provide the proof that the above statement holds true
  -- based on the definitions and given conditions.
  sorry  -- Placeholder for the actual proof

end molecular_weight_equimolar_mixture_l757_757954


namespace distance_O₁_to_DE_l757_757859

variables (O₁ O₂ A B C D E : Type)
variables [metric_space O₁] [metric_space O₂] [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

-- defining circles, their properties, and given distances
def circle (center : Type) (radius : ℝ) := {p : Type // dist center p = radius}
def intersect (c1 c2 : Type) := ∃ A B, A ∈ c1 ∧ A ∈ c2 ∧ B ∈ c1 ∧ B ∈ c2
def line_of (p1 p2 : Type) := λ x, x = p1 ∨ x = p2
constant r₁ : ℝ
constant AC AD : ℝ
constant C_on_O₁ : Bool

axiom A_on_intersection : ∀ O₁ O₂, intersect (circle O₁ r₁) (circle O₂ r₁)
axiom C_on_O₁_true : C_on_O₁ = true
axiom AC_value : AC = 3
axiom AD_value : AD = 6
axiom radius_O₁ : r₁ = 2

theorem distance_O₁_to_DE
  (h1 : intersect (circle O₁ r₁) (circle O₂ r₁))
  (h2 : C_on_O₂ = false)
  (h3 : line_of C A = D)
  (h4 : line_of C B = E)
  (h5 : dist A C = 3)
  (h6 : dist A D = 6)
  (h7 : radius_O₁ = 2) :
  dist (O₁ : Type) (line_of D E) = 17/4 := sorry

end distance_O₁_to_DE_l757_757859


namespace find_100th_index_lt_zero_l757_757950

def sequence_a (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), Real.cos (k : ℝ)

theorem find_100th_index_lt_zero :
  ∃ n : ℕ, (∃ k : ℕ, n = ⌊2 * Real.pi * k⌋) ∧
  (n > 0) ∧
  ∀ m < n, (∃ i : ℕ, m = ⌊2 * Real.pi * i⌋ → m ≠ 100) →
  n = 628 :=
sorry

end find_100th_index_lt_zero_l757_757950


namespace distinct_arrangements_of_apples_l757_757639

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem distinct_arrangements_of_apples :
  let n := 6
  let freq_a := 1
  let freq_p := 2
  let freq_l := 1
  let freq_e := 1
  let freq_s := 1
  (factorial n) / (factorial freq_a * factorial freq_p * factorial freq_l * factorial freq_e * factorial freq_s) = 360 :=
begin
  sorry
end

end distinct_arrangements_of_apples_l757_757639


namespace prove_n_prime_l757_757556

theorem prove_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Prime p) (h1 : n > 0) (h2 : 3^n - 2^n = p^k) : Prime n :=
by {
  sorry
}

end prove_n_prime_l757_757556


namespace exist_indices_inequalities_l757_757465

open Nat

theorem exist_indices_inequalities (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  -- The proof is to be written here
  sorry

end exist_indices_inequalities_l757_757465


namespace find_ellipse_eq_dot_product_range_l757_757589

-- Define the conditions for the ellipse and additional conditions of the problem
variable (a b k m : ℝ)
variable (h1 : a > b) (h2 : b > 0) (h3 : b = 1) (h4 : (sqrt(2) / 2) = sqrt(a^2 - b^2) / a)
variable (area_eq : 2 * m^2 / (1 - 4 * k^2) = 2)
variable (k_nonzero : k ≠ 0)

-- Statement proving the equation of the ellipse
theorem find_ellipse_eq :
  ∃ a b : ℝ, (a > b ∧ b > 0) ∧ b = 1 ∧ (sqrt(2) / 2 = sqrt(a^2 - b^2) / a) → (a^2 = 2 ∧ ellipse_eq = (λ x y, (x^2 / 2) + y^2 = 1)) :=
sorry

-- Statement proving the range of values for the dot product given the conditions
theorem dot_product_range (x1 x2 y1 y2 : ℝ) :
  ∃ (x1 x2 y1 y2 k m : ℝ), (0 < k^2) ∧ (k^2 < 1 / 4) ∧ (1 - 4 * k^2 > 0) ∧ (2 * m^2 / (1 - 4 * k^2) = 2) ∧
    (m^2 - 2 * k^2) / (1 + 2 * k^2) + (2 * m^2 - 2) / (1 + 2 * k^2)) -> 
      (-5 / 3) < (x1 * x2 + y1 * y2) ∧ (x1 * x2 + y1 * y2) < 1 :=
sorry

end find_ellipse_eq_dot_product_range_l757_757589


namespace focus_of_parabola_l757_757994

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757994


namespace find_coordinates_l757_757615

def Z : ℂ := (2 + 4 * complex.I) / (1 + complex.I)

theorem find_coordinates :
  (Z.re, Z.im) = (3, 1) :=
sorry

end find_coordinates_l757_757615


namespace correct_midpoint_l757_757205

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757205


namespace number_of_square_free_odds_l757_757644

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757644


namespace min_button_presses_l757_757879

theorem min_button_presses :
  ∃ (a b : ℤ), 9 * a - 20 * b = 13 ∧  a + b = 24 := 
by
  sorry

end min_button_presses_l757_757879


namespace quadratic_solution_1_quadratic_solution_2_quadratic_solution_3_l757_757915

/- Proof problem: Given 2x^2 - 3x - 5 = 0, prove x = 5/2 or x = -1 -/
theorem quadratic_solution_1 (x : ℝ) (h : 2 * x^2 - 3 * x - 5 = 0) : x = 5 / 2 ∨ x = -1 := 
  sorry

/- Proof problem: Given x^2 + 2x - 3 = 0, prove x = 1 or x = -3 -/
theorem quadratic_solution_2 (x : ℕ) (h : x^2 + 2 * x - 3 = 0) : x = 1 ∨ x = -3 := 
  sorry

/- Proof problem: Given 3 * (x - 2)^2 = x^2 - 4, prove x = 2 or x = 4 -/
theorem quadratic_solution_3 (x : ℝ) (h : 3 * (x - 2)^2 = x^2 - 4) : x = 2 ∨ x = 4 := 
  sorry

end quadratic_solution_1_quadratic_solution_2_quadratic_solution_3_l757_757915


namespace total_books_l757_757565

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l757_757565


namespace number_of_ways_to_pick_two_cards_l757_757367

theorem number_of_ways_to_pick_two_cards (cards : Finset ℕ) (h : cards.cardinality = 104) : 
  (104 * 103 = 10692) :=
sorry

end number_of_ways_to_pick_two_cards_l757_757367


namespace sum_digits_of_3n_l757_757804

noncomputable def sum_digits (n : ℕ) : ℕ :=
sorry  -- Placeholder for a proper implementation of sum_digits

theorem sum_digits_of_3n (n : ℕ) 
  (h1 : sum_digits n = 100) 
  (h2 : sum_digits (44 * n) = 800) : 
  sum_digits (3 * n) = 300 := 
by
  sorry

end sum_digits_of_3n_l757_757804


namespace final_sale_price_is_12_25_percent_lower_l757_757410

-- Define original price x
variable (x : ℝ) (h₁ : x > 0)

-- Conditions
def increased_price : ℝ := 1.30 * x
def sale_price : ℝ := 0.75 * increased_price
def final_price : ℝ := 0.90 * sale_price

-- Statement to be proved
theorem final_sale_price_is_12_25_percent_lower (x : ℝ) (h₁ : x > 0) : 
  final_price x h₁ = 0.8775 * x :=
by
  unfold increased_price
  unfold sale_price
  unfold final_price
  sorry

end final_sale_price_is_12_25_percent_lower_l757_757410


namespace parabola_focus_l757_757992

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757992


namespace sum_of_modified_triangular_array_50th_row_l757_757529

theorem sum_of_modified_triangular_array_50th_row : 
    ∀ (f : ℕ → ℕ), (f 1 = 0) → (∀ n, f(n + 1) = 2 * f(n) + 4) → f(50) = 2^51 - 4 := by
  sorry

end sum_of_modified_triangular_array_50th_row_l757_757529


namespace square_free_odd_integers_count_l757_757670

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757670


namespace correct_midpoint_l757_757218

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757218


namespace midpoint_of_hyperbola_l757_757054

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757054


namespace midpoint_of_hyperbola_segment_l757_757088

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757088


namespace square_free_odd_integers_count_l757_757652

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757652


namespace midpoint_hyperbola_l757_757230

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757230


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757286

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757286


namespace solve_pascals_triangle_problem_l757_757792

def pascals_triangle (n : ℕ) : list (list ℕ) :=
  (List.range (n + 1)).map (λ r, (List.range (r + 1)).map (λ c, Nat.choose r c))

def row_has_property (row : list ℕ) : Prop :=
  ∀ (x : ℕ), (x ∈ row.init) ∧ (x ∈ row.tail.init) → x % 2 = 0

def count_rows_with_property (rows : list (list ℕ)) : ℕ :=
  rows.countp row_has_property

open Nat

theorem solve_pascals_triangle_problem : count_rows_with_property (pascals_triangle 20).drop 2 = 4 := sorry

end solve_pascals_triangle_problem_l757_757792


namespace problem1_problem2_l757_757922

-- Problem 1
theorem problem1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ≠ 1) : (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) := sorry

end problem1_problem2_l757_757922


namespace parabola_focus_l757_757984

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757984


namespace derivative_correct_l757_757546

noncomputable def y (x : ℝ) : ℝ := (1 / 6) * Real.log ((1 - Real.sinh(2 * x)) / (2 + Real.sinh(2 * x)))
noncomputable def y_derivative (x : ℝ) : ℝ := (1 / 6) * ((1 - Real.sinh(2 * x)) / (2 + Real.sinh(2 * x))).derivative
noncomputable def expected_y_derivative (x : ℝ) : ℝ := Real.cosh(2 * x) / (Real.sinh(2 * x)^2 + Real.sinh(2 * x) - 2)

theorem derivative_correct (x : ℝ) : y_derivative x = expected_y_derivative x :=
sorry

end derivative_correct_l757_757546


namespace lines_intersect_l757_757549

theorem lines_intersect :
  ∃ x y : ℚ, 
  8 * x - 5 * y = 40 ∧ 
  6 * x - y = -5 ∧ 
  x = 15 / 38 ∧ 
  y = 140 / 19 :=
by { sorry }

end lines_intersect_l757_757549


namespace tan_neg_405_eq_neg1_l757_757931

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l757_757931


namespace midpoint_hyperbola_l757_757106

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757106


namespace midpoint_hyperbola_l757_757224

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757224


namespace range_of_m_l757_757625

def g (x : ℝ) : ℝ :=
  Real.exp x - x + (1 / 2) * x^2

theorem range_of_m (m : ℝ) (x0 : ℝ) :
  (2 * m - 1 ≥ g x0) → m ≥ 1 :=
by
  intro h
  have g_min : g 0 = 1 := by
    -- The function g(x) has minimum value at g(0) which is 1
    sorry
  have h_m_min : 2 * m - 1 ≥ 1 := by
    -- Since 2m - 1 ≥ g(x_0) and g(x_0) >= g(0) = 1, we have 2 * m - 1 ≥ 1
    exact le_trans h (by exact g_min.ge)
  exact (sub_le_iff_le_add'.mp h_m_min)

end range_of_m_l757_757625


namespace num_valid_points_l757_757745

theorem num_valid_points :
  let P := {1, x}
  let Q := {1, 2, y}
  ∃ n : ℕ, n = 14 :=
by
  sorry

end num_valid_points_l757_757745


namespace midpoint_hyperbola_l757_757105

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757105


namespace num_ways_to_move_items_l757_757402

-- Definitions of initial conditions
def total_items : ℕ := 12
def upper_items : ℕ := 4
def lower_items : ℕ := 8
def selected_lower_items : ℕ := 2

-- The theorem we want to prove
theorem num_ways_to_move_items : 
  (nat.choose lower_items selected_lower_items) * 
  ((nat.factorial 5 / nat.factorial 3) + nat.choose 5 2) = 840 := 
by {
  -- Calculation steps can be filled in here
  sorry
}

end num_ways_to_move_items_l757_757402


namespace gcd_60_90_150_l757_757830

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l757_757830


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757293

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757293


namespace petya_five_ruble_coins_l757_757301

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757301


namespace inequality_solution_interval_l757_757774

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l757_757774


namespace coeff_x2_expansion_eq_14_l757_757370

theorem coeff_x2_expansion_eq_14 :
  let f := (1 + x)^7 * (1 - x),
      coeff_x2 := polynomial.coeff f 2 in
  coeff_x2 = 14 :=
by
  sorry

end coeff_x2_expansion_eq_14_l757_757370


namespace square_free_odd_integers_count_l757_757650

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757650


namespace midpoint_on_hyperbola_l757_757079

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757079


namespace time_to_cross_pole_l757_757504

noncomputable def train_length := 133.33333333333334 -- Length in meters
noncomputable def train_speed_kmh := 60 -- Speed in km/hr

-- Converting the speed to m/s
noncomputable def train_speed_ms := train_speed_kmh * (1000 / 3600)

-- Statement: Time to cross the pole
theorem time_to_cross_pole : train_length / train_speed_ms = 8 :=
by
  sorry

end time_to_cross_pole_l757_757504


namespace TB_eq_TC_l757_757407

variable (A B C D M N T : Type)
variables (AB AC AD BC CD BD : Real)
variables (h_trapezoid : AB * CD = AD * BC)

variable (circle : Set (Point A B C D))

-- Conditions related to the points and intersections
variables (h_on_circle_B : circle B)
variables (h_on_circle_C : circle C)
variables (h_on_circle_M : circle M)
variables (h_on_circle_N : circle N)
variables (h_intersection : ∃ T, T ∈ (Line (A, N)) ∩ (Line (D, M)) ∧ T ∈ circle)

-- The question/request to prove TB = TC
theorem TB_eq_TC : (dist T B = dist T C) :=
by
  sorry

end TB_eq_TC_l757_757407


namespace hyperbola_midpoint_l757_757118

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757118


namespace midpoint_hyperbola_l757_757229

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757229


namespace find_x_l757_757425

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l757_757425


namespace midpoint_of_hyperbola_segment_l757_757090

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757090


namespace perimeter_triangle_ABC_l757_757501

-- Definitions of A, B, C, D, and C' with given coordinates
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (2, 0)
def D : (ℝ × ℝ) := (0, 2)
def C : (ℝ × ℝ) := (2, 2)
def C' : (ℝ × ℝ) := (1.5, 0)

-- Definition of the distance function for ease of length calculations
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- The theorem to prove the perimeter of triangle ABC' is 4
theorem perimeter_triangle_ABC'_is_4 :
  dist A B + dist B C' + dist A C' = 4 := by
  sorry

end perimeter_triangle_ABC_l757_757501


namespace focus_of_parabola_l757_757996

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757996


namespace hyperbola_midpoint_l757_757126

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757126


namespace inverse_property_l757_757531

variable {X Y : Type}
variable (f : X → Y) (g : Y → X)
variable [Inhabited X] [Inhabited Y]

-- g is the inverse of f
def is_inverse (f : X → Y) (g : Y → X) := ∀ x : X, g (f x) = x

-- f(ab) = f(a) + f(b) for all a, b in the domain of f
def satisfies_property (f : X → Y) := ∀ a b : X, f (a * b) = f a + f b

-- Define the range of function f
def range (f : X → Y) := { y : Y | ∃ x : X, f x = y }

-- Prove that ∀ a, b in the range(f), g(a + b) = g(a) * g(b) 
theorem inverse_property (h1 : satisfies_property f) (h2 : is_inverse f g) :
  ∀ (a b : Y), a ∈ range f → b ∈ range f → g (a + b) = g a * g b :=
by
  sorry

end inverse_property_l757_757531


namespace hyperbola_midpoint_l757_757201

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757201


namespace arithmetic_sequence_geometric_ratio_l757_757611

theorem arithmetic_sequence_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n : ℕ, a (n+1) = a n + d)
  (h_nonzero_d : d ≠ 0)
  (h_geo : (a 2) * (a 9) = (a 3) ^ 2)
  : (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = (8 / 3) :=
by
  sorry

end arithmetic_sequence_geometric_ratio_l757_757611


namespace hyperbola_midpoint_l757_757149

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757149


namespace arithmetic_expression_l757_757524

theorem arithmetic_expression : (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
  sorry

end arithmetic_expression_l757_757524


namespace number_in_set_S_l757_757736

theorem number_in_set_S (n : ℕ) (S : Set ℕ) : 
  (S = {x | ∃ n : ℕ, x = 8 * n + 5}) → (∃ k : ℕ, k = 1000 → 8 * (k - 1) + 5 = 7997) :=
by
  intro h
  use 1000
  intro hk
  rw [←hk]
  calc
    8 * (1000 - 1) + 5 = 8 * 999 + 5 : by rw [Nat.sub_one]
                      ... = 7992 + 5 : by rw [Nat.mul_comm]
                      ... = 7997     : by norm_num
  done

end number_in_set_S_l757_757736


namespace volume_of_intersection_region_l757_757422

def abs (a : ℝ) : ℝ := if a < 0 then -a else a

def region_1 (x y z : ℝ) : Prop := abs x + abs y + abs (z - 1) <= 2
def region_2 (x y z : ℝ) : Prop := abs x + abs y + abs (z + 1) <= 2

theorem volume_of_intersection_region : 
  (∫ x in -2..2, ∫ y in -2..2, ∫ z in -2..2, 
    if region_1 x y z ∧ region_2 x y z then 1 else 0) = 8 / 3 :=
sorry

end volume_of_intersection_region_l757_757422


namespace tan_neg_405_eq_neg1_l757_757933

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l757_757933


namespace simplify_expr_l757_757352

-- Define the terms to be simplified
def term1 : ℝ := √300 / √75
def term2 : ℝ := √200 / √50

-- Prove that the given expression simplifies to 0
theorem simplify_expr : term1 - term2 = 0 :=
by sorry

end simplify_expr_l757_757352


namespace point_b_in_third_quadrant_l757_757902

-- Definitions of the points with their coordinates
def PointA : ℝ × ℝ := (2, 3)
def PointB : ℝ × ℝ := (-1, -4)
def PointC : ℝ × ℝ := (-4, 1)
def PointD : ℝ × ℝ := (5, -3)

-- Definition of a point being in the third quadrant
def inThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The main Theorem to prove that PointB is in the third quadrant
theorem point_b_in_third_quadrant : inThirdQuadrant PointB :=
by sorry

end point_b_in_third_quadrant_l757_757902


namespace line_intersects_ellipse_l757_757609

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ a = 2 ∧ b^2 = 3 ∧ 
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 ^ 2 / 4 + p.2 ^ 2 / 3 = 1))) 

theorem line_intersects_ellipse : ∀ m : ℝ, (∃ x y : ℝ, y = x + m ∧ (x^2 / 4) + (y^2 / 3) = 1) ↔ -√7 ≤ m ∧ m ≤ √7 :=
by sorry

end line_intersects_ellipse_l757_757609


namespace tangent_line_eq_2x_plus_1_at_0_1_l757_757790

noncomputable def tangent_line {α : Type*} [field α] [has_exp α] (f : α → α) (c x : α) (slope : α) :=
  f c + slope * (x - c)

theorem tangent_line_eq_2x_plus_1_at_0_1 :
  tangent_line (λ x : ℝ, Real.exp (2 * x)) 0 (λ x : ℝ, 2) = (λ x, 2 * x + 1) :=
by
  sorry

end tangent_line_eq_2x_plus_1_at_0_1_l757_757790


namespace bob_final_amount_l757_757916

noncomputable def final_amount (start: ℝ) : ℝ :=
  let day1 := start - (3/5) * start
  let day2 := day1 - (7/12) * day1
  let day3 := day2 - (2/3) * day2
  let day4 := day3 - (1/6) * day3
  let day5 := day4 - (5/8) * day4
  let day6 := day5 - (3/5) * day5
  day6

theorem bob_final_amount : final_amount 500 = 3.47 := by
  sorry

end bob_final_amount_l757_757916


namespace midpoint_on_hyperbola_l757_757140

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757140


namespace hyperbola_midpoint_exists_l757_757242

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757242


namespace hyperbola_midpoint_exists_l757_757235

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757235


namespace expression_value_l757_757525

theorem expression_value (a b c d : ℤ) (ha : a = 10) (hb : b = 15) (hc : c = 3) (hd : d = 2) :
    [ a * (b - c) ] - [ (a - b) * c ] + d = 137 :=
by
  rw [ha, hb, hc, hd]
  sorry

end expression_value_l757_757525


namespace probability_of_different_digits_correct_l757_757906

def is_two_digit (n : ℕ) : Prop := n > 9 ∧ n < 100

def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

def has_different_digits (n : ℕ) : Prop :=
  (n / 10) ≠ (n % 10)

noncomputable def probability_of_different_digits : ℚ :=
  let all_numbers := {n | is_two_digit n ∧ is_divisible_by_seven n}
  let count_all := all_numbers.to_finset.card
  let different_digit_numbers := {n | is_two_digit n ∧ is_divisible_by_seven n ∧ has_different_digits n}
  let count_different := different_digit_numbers.to_finset.card
  count_different / count_all

theorem probability_of_different_digits_correct :
  probability_of_different_digits = 12 / 13 := by sorry

end probability_of_different_digits_correct_l757_757906


namespace divisors_partition_100_l757_757002

theorem divisors_partition_100! :
  ∃ (A B : Multiset ℕ), A.card = B.card ∧ A.prod = B.prod ∧ A + B = finsupp.finset (100.factorial).divisors :=
by {
  sorry
}

end divisors_partition_100_l757_757002


namespace equivalence_volumes_l757_757488

noncomputable def radius (diameter : ℝ) : ℝ := diameter / 2

noncomputable def volume_of_cylinder (radius height : ℝ) : ℝ := Real.pi * radius^2 * height

-- Original dimensions
def diameter_original : ℝ := 18
def height_original : ℝ := 10

-- New dimensions
def diameter_new : ℝ := 12
def height_new : ℝ := 12

-- Volumes
def volume_original : ℝ := volume_of_cylinder (radius diameter_original) height_original
def volume_new : ℝ := volume_of_cylinder (radius diameter_new) height_new

-- Required number of new containers
noncomputable def required_new_containers : ℝ := 1000 * volume_original / volume_new

theorem equivalence_volumes :
  required_new_containers = 1875 :=
by
  sorry

end equivalence_volumes_l757_757488


namespace M_eq_interval_abs_sum_lt_abs_one_plus_mul_l757_757623

def f (x : ℝ) : ℝ := |x - (1 / 2)| + |x + (1 / 2)|

def M : set ℝ := { x | f x < 2 }

theorem M_eq_interval : M = set.Ioo (-1) 1 := sorry

theorem abs_sum_lt_abs_one_plus_mul {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := sorry

end M_eq_interval_abs_sum_lt_abs_one_plus_mul_l757_757623


namespace min_value_inequality_l757_757257

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / x + 1 / y) * (4 * x + y) ≥ 9 ∧ ((1 / x + 1 / y) * (4 * x + y) = 9 ↔ y / x = 2) :=
by
  sorry

end min_value_inequality_l757_757257


namespace a_2017_value_l757_757801

theorem a_2017_value (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2 * (n + 1) - 1) :
  a 2017 = 2 :=
by
  sorry

end a_2017_value_l757_757801


namespace printed_image_height_l757_757876

theorem printed_image_height : 
  let width_img1 := 13 -- width of first image in cm
  let height_img1 := 9 -- height of first image in cm
  let width_img2 := 14 -- width of second image in cm
  let height_img2 := 12 -- height of second image in cm
  let final_total_width := 18.8 -- final total width in cm
  let resizing_factor_height_img2 := 9 / 12 -- resizing factor for height of second image
  let new_width_img2 := (3 / 4) * width_img2 -- new width of second image
  let initial_total_width := width_img1 + new_width_img2 -- total initial width
  let resizing_factor_total_width := final_total_width / initial_total_width -- resizing factor for total width
  let final_height := resizing_factor_total_width * height_img1 -- final height
  in final_height = 7.2 :=
by sorry

end printed_image_height_l757_757876


namespace midpoint_on_hyperbola_l757_757046

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757046


namespace min_green_points_l757_757703

theorem min_green_points (total_points : ℕ) (points : Finset (ℝ × ℝ)) (black_points green_points : Finset (ℝ × ℝ)) : 
  total_points = 2020 ∧ 
  ( ∀ p ∈ black_points, ∃ g1 g2 ∈ green_points, dist p g1 = 2020 ∧ dist p g2 = 2020 ∧ g1 ≠ g2 ) ∧
  ( ∀ g ∈ green_points, ∃! b ∈ black_points, dist g b = 2020) 
  → ∃ m : ℕ, m = 45 ∧ green_points.card = m :=
by sorry

end min_green_points_l757_757703


namespace flagpole_height_l757_757492

theorem flagpole_height
  (bamboo_height : ℝ := 3)
  (bamboo_shadow : ℝ := 1.2)
  (flagpole_shadow : ℝ := 4.8) : 
  ∃ (flagpole_height : ℝ), flagpole_height = 12 :=
by
  let flagpole_height := (bamboo_height * flagpole_shadow) / bamboo_shadow
  have h : flagpole_height = 12 := by sorry
  use flagpole_height
  exact h

end flagpole_height_l757_757492


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757296

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757296


namespace midpoint_on_hyperbola_l757_757130

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757130


namespace pyramid_vol_5x7_15_l757_757887

def pyramid_volume (length width slant height : ℝ) : ℝ :=
  (1 / 3) * length * width * height

theorem pyramid_vol_5x7_15 (length width edge height : ℝ) 
  (h_length : length = 5)
  (h_width : width = 7)
  (h_edge : edge = 15)
  (h_volume : volume = (70 * Real.sqrt 47) / 3) : 
  pyramid_volume length width edge height = volume := 
  sorry

end pyramid_vol_5x7_15_l757_757887


namespace smaller_angle_measure_l757_757412

theorem smaller_angle_measure (α β : ℝ) (h1 : α + β = 90) (h2 : α = 4 * β) : β = 18 :=
by
  sorry

end smaller_angle_measure_l757_757412


namespace shirt_discount_l757_757265

theorem shirt_discount (original_price discounted_price : ℕ) 
  (h1 : original_price = 22) 
  (h2 : discounted_price = 16) : 
  original_price - discounted_price = 6 := 
by
  sorry

end shirt_discount_l757_757265


namespace unknown_cube_edge_length_l757_757789

theorem unknown_cube_edge_length :
  let a := 6
  let b := 8
  let c := 12
  let V1 := a^3
  let V2 := b^3
  let V_new := c^3
  ∃ x : ℕ, V_new = V1 + V2 + x^3 ∧ x = 10 :=
begin
  sorry
end

end unknown_cube_edge_length_l757_757789


namespace friends_left_after_removal_l757_757264

-- We'll define constants for the initial number of friends and the percentages.
def initial_friends : ℕ := 100
def keep_percentage : ℝ := 0.4
def respond_percentage : ℝ := 0.5

-- Calculate the number of friends Mark keeps initially.
def kept_friends : ℕ := (initial_friends : ℝ) * keep_percentage

-- Calculate the number of friends Mark contacts.
def contacted_friends : ℕ := initial_friends - kept_friends

-- Calculate the number of friends that respond.
def responded_friends : ℕ := (contacted_friends : ℝ) * respond_percentage

-- Calculate the total number of friends left after removal.
def total_friends_left : ℕ := kept_friends + responded_friends

-- Formalize the theorem to be proved by Lean.
theorem friends_left_after_removal : total_friends_left = 70 := by
  sorry

end friends_left_after_removal_l757_757264


namespace find_cost_price_l757_757857

variable (C : ℝ)

def profit_10_percent_selling_price := 1.10 * C

def profit_15_percent_with_150_more := 1.10 * C + 150

def profit_15_percent_selling_price := 1.15 * C

theorem find_cost_price
  (h : profit_15_percent_with_150_more C = profit_15_percent_selling_price C) :
  C = 3000 :=
by
  sorry

end find_cost_price_l757_757857


namespace comparison_arctan_l757_757726

theorem comparison_arctan (a b c : ℝ) (h : Real.arctan a + Real.arctan b + Real.arctan c + Real.pi / 2 = 0) :
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) :=
by
  sorry

end comparison_arctan_l757_757726


namespace hyperbola_midpoint_l757_757200

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757200


namespace courier_problem_l757_757485

variable (x : ℝ) -- Let x represent the specified time in minutes
variable (d : ℝ) -- Let d represent the total distance traveled in km

theorem courier_problem
  (h1 : 1.2 * (x - 10) = d)
  (h2 : 0.8 * (x + 5) = d) :
  x = 40 ∧ d = 36 :=
by
  -- This theorem statement encapsulates the conditions and the answer.
  sorry

end courier_problem_l757_757485


namespace hyperbola_midpoint_l757_757203

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757203


namespace quadratic_no_real_roots_l757_757846

theorem quadratic_no_real_roots :
  ∀ (a b c : ℝ), (a = 1 ∧ b = 1 ∧ c = 2) → (b^2 - 4 * a * c < 0) := by
  intros a b c H
  cases H with Ha Hac
  cases Hac with Hb Hc
  rw [Ha, Hb, Hc]
  simp
  linarith

end quadratic_no_real_roots_l757_757846


namespace midpoint_on_hyperbola_l757_757078

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757078


namespace midpoint_of_hyperbola_l757_757167

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757167


namespace find_square_sum_of_xy_l757_757964

theorem find_square_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y + x + y = 83) (h2 : x^2 * y + x * y^2 = 1056) : x^2 + y^2 = 458 :=
sorry

end find_square_sum_of_xy_l757_757964


namespace sum_powers_of_minus_one_l757_757961

theorem sum_powers_of_minus_one : 
  ∑ n in Finset.range 31, (-1 : ℤ) ^ (n - 15) = 0 :=
by
  sorry

end sum_powers_of_minus_one_l757_757961


namespace find_number_l757_757438

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l757_757438


namespace part1_b3_5_value_part2_general_formula_l757_757256

noncomputable def array_transform (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ) (m n i : ℕ) : ℕ :=
  if m = 1 then
    a i + a (i+1)
  else
    b (m-1) i + b (m-1) (i+1)

theorem part1_b3_5_value (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ)
  (hk : ∀ i, a i = i) (hn : n ≥ 2) : array_transform a b 3 5 = 52 :=
  sorry

theorem part2_general_formula (m n i : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ)
  (array_transform : a n+1 = a 1 ∧ ∀ m ≥ 2, b m-1 n+1 = b m-1 1) : 
  (∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → b m i = ∑ (j : ℕ) in finset.range (m + 1), a (i + j) * nat.choose m j) :=
  sorry

end part1_b3_5_value_part2_general_formula_l757_757256


namespace problem_transform_sum_of_solutions_l757_757955

theorem problem_transform (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 6) 
  (h2 : (x^2 - 4 * x + 3) ^ (x^2 - 5 * x + 6) = 1) : 
  x = 1 ∨ x = 2 ∨ x = 3 := sorry

theorem sum_of_solutions : 
  {x : ℝ | 1 ≤ x ∧ x ≤ 6 ∧ (x^2 - 4 * x + 3) ^ (x^2 - 5 * x + 6) = 1}.sum = 6 := 
by
  sorry

end problem_transform_sum_of_solutions_l757_757955


namespace frequency_count_l757_757497

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 1000) (h2 : f = 0.4) : n * f = 400 := by
  sorry

end frequency_count_l757_757497


namespace equivalence_of_expression_l757_757375

theorem equivalence_of_expression (x y : ℝ) :
  ( (x^2 + y^2 + xy) / (x^2 + y^2 - xy) ) - ( (x^2 + y^2 - xy) / (x^2 + y^2 + xy) ) =
  ( 4 * xy * (x^2 + y^2) ) / ( x^4 + y^4 ) :=
by sorry

end equivalence_of_expression_l757_757375


namespace sum_bounds_l757_757719

noncomputable section

variables {n : ℕ} (a : Fin n → ℝ) (k : Fin n → ℕ)

def valid_sequence (a : Fin n → ℝ) (k : Fin n → ℕ) : Prop :=
  ∀ i : Fin n, k i = ((a ((i - 1) % n) + a ((i + 1) % n)) / a i) ∧ k i > 0

theorem sum_bounds (n : ℕ) (a : Fin n → ℝ) (k : Fin n → ℕ)
    (h1 : 2 < n) 
    (h2 : valid_sequence a k) : 
    2 * (n : ℝ) ≤ ∑ i : Fin n, a i ∧ ∑ i : Fin n, a i ≤ 3 * (n : ℝ) :=
sorry

end sum_bounds_l757_757719


namespace part1_part2_l757_757573

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 = 0

theorem part1 (h : p a) : a ≤ 1 :=
sorry

theorem part2 (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ set.Icc (-Math.sqrt 2) 1 ∪ set.Ici (Math.sqrt 2) :=
sorry

end part1_part2_l757_757573


namespace mans_rate_in_still_water_l757_757852

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h1 : V_m + V_s = 20)
  (h2 : V_m - V_s = 4) :
  V_m = 12 :=
by
  sorry

end mans_rate_in_still_water_l757_757852


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757278

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757278


namespace correct_midpoint_l757_757209

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757209


namespace angle_ABC_40_degrees_l757_757910

theorem angle_ABC_40_degrees (ABC ABD CBD : ℝ) 
    (h1 : CBD = 90) 
    (h2 : ABD = 60)
    (h3 : ABC + ABD + CBD = 190) : 
    ABC = 40 := 
by {
  sorry
}

end angle_ABC_40_degrees_l757_757910


namespace midpoint_on_hyperbola_l757_757041

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757041


namespace parabola_focus_l757_757990

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757990


namespace midpoint_of_hyperbola_segment_l757_757091

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757091


namespace constant_area_of_triangles_in_an_ellipse_l757_757350

theorem constant_area_of_triangles_in_an_ellipse
  (E : Ellipse) (T : Triangle) (center_E : Point) (G : Point)
  (h1 : T.inscribed_in E) (h2 : G.centroid_of T = center_E) :
  ∃ A : ℝ, ∀ T₁ : Triangle,
  (T₁.inscribed_in E) ∧ (G.centroid_of T₁ = center_E) →
  T₁.area = A := by
  sorry

end constant_area_of_triangles_in_an_ellipse_l757_757350


namespace eigenvalues_of_a_l757_757630

theorem eigenvalues_of_a {a d : ℝ} (h1 : matrix.of ![![a, 3], ![2, d]] ⬝ ![1, 2] = ![8, 4]) :
  (∃ λ1 λ2, ((matrix.from_blocks ![![a-λ1, 3], ![2, d-λ1]]).det = 0) ∧
              ((matrix.from_blocks ![![a-λ2, 3], ![2, d-λ2]]).det = 0) ∧ 
              ((λ1 = 4 ∧ λ2 = -1) ∨ (λ1 = -1 ∧ λ2 = 4))) :=
by sorry

end eigenvalues_of_a_l757_757630


namespace surface_area_comparison_l757_757861

-- Define the conditions of the problem
variables {p x y : ℝ}

-- Condition of the parabola and the chord passing through the focus
def parabola := ∀ x y, y^2 = 2 * p * x
def chord_length_PQ := x + y

-- Directrix condition projection calculation
def projection_MN := 2 * real.sqrt(x * y)

-- Surface areas definitions
def surface_area_S1 := π * (x + y)^2
def surface_area_S2 := 4 * π * (real.sqrt(x * y))^2

-- Prove that S1 >= S2
theorem surface_area_comparison (h : 0 < x ∧ 0 < y) : 
  π * (x + y)^2 ≥ 4 * π * x * y := by
  apply mul_le_mul_left (lt_trans zero_lt_one pi_pos)
  apply (real.am_gm x y)
  .1
  .2
  .2

end

end surface_area_comparison_l757_757861


namespace number_of_yellow_parrots_l757_757755

theorem number_of_yellow_parrots (total_parrots : ℕ) (red_fraction : ℚ) 
  (h_total_parrots : total_parrots = 108) 
  (h_red_fraction : red_fraction = 5 / 6) : 
  ∃ (yellow_parrots : ℕ), yellow_parrots = total_parrots * (1 - red_fraction) ∧ yellow_parrots = 18 := 
by
  sorry

end number_of_yellow_parrots_l757_757755


namespace solution_set_inequality_l757_757626

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f e = 0) (h2 : ∀ x > 0, x * deriv f x < 2) :
    ∀ x, 0 < x → x ≤ e → f x + 2 ≥ 2 * log x :=
by
  sorry

end solution_set_inequality_l757_757626


namespace num_five_ruble_coins_l757_757314

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757314


namespace total_area_excluding_overlap_l757_757963

-- Definitions of the conditions
def width := 9.4   -- cm
def length := 3.7  -- cm
def overlap := 0.6 -- cm
def num_pieces := 15
def area_single_tape := width * length
def overlap_area_single := overlap * length
def total_area_without_overlap := num_pieces * area_single_tape
def total_overlap_area := (num_pieces - 1) * overlap_area_single

-- Statement of the goal
theorem total_area_excluding_overlap : 
  total_area_without_overlap - total_overlap_area = 490.62 := by
  sorry

end total_area_excluding_overlap_l757_757963


namespace rationalize_denominator_l757_757345

theorem rationalize_denominator :
  (2 / (Real.cbrt 3 - 2)) = -(2 * Real.cbrt 3 + 4) / 5 :=
by
  sorry

end rationalize_denominator_l757_757345


namespace jellybean_ratio_l757_757408

theorem jellybean_ratio (L Tino Arnold : ℕ) (h1 : Tino = L + 24) (h2 : Arnold = 5) (h3 : Tino = 34) :
  Arnold / L = 1 / 2 :=
by
  sorry

end jellybean_ratio_l757_757408


namespace petya_five_ruble_coins_l757_757302

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757302


namespace overdue_fine_day5_l757_757480

noncomputable def fine : ℕ → ℝ
| 0 := 0.05
| (n + 1) := min (fine n + 0.30) (2 * fine n)

theorem overdue_fine_day5 : fine 5 = 0.70 := by
  sorry

end overdue_fine_day5_l757_757480


namespace parabola_focus_l757_757988

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757988


namespace minimize_total_annual_cost_l757_757483

-- Define the conditions
def total_tonnage : ℝ := 400
def freight_cost_per_purchase : ℝ := 40000
def storage_cost_per_ton : ℝ := 40000

-- Define the function representing the total annual cost
def total_annual_cost (x : ℝ) : ℝ :=
  (total_tonnage / x) * freight_cost_per_purchase + storage_cost_per_ton * x

-- Define the problem statement: to minimize the total annual cost
theorem minimize_total_annual_cost : ∃ x : ℝ, x = 20 ∧ ∀ y : ℝ, y ≠ 20 → total_annual_cost 20 ≤ total_annual_cost y :=
sorry

end minimize_total_annual_cost_l757_757483


namespace steve_height_equiv_l757_757359

/-- 
  Steve's initial height in feet and inches.
  convert_height: converts feet and inches to total inches.
  grows further: additional height Steve grows.
  expected height: the expected total height after growing.
--/

def initial_height_feet := 5
def initial_height_inches := 6
def additional_height := 6

def convert_height(feet: Int, inches: Int): Int := 
  feet * 12 + inches

def expected_height(initial_feet: Int, initial_inches: Int, additional: Int): Int := 
  convert_height(initial_feet, initial_inches) + additional

theorem steve_height_equiv:
  expected_height initial_height_feet initial_height_inches additional_height = 72 :=
by
  sorry

end steve_height_equiv_l757_757359


namespace trigonometric_function_range_l757_757800

theorem trigonometric_function_range :
  let y (x : ℝ) := (sin x / |sin x|) + (|cos x| / cos x) + (tan x / |tan x|) + (|cot x| / cot x)
  ∃ (range_y : set ℝ), range_y = {-2, 0, 4} ∧ ∀ x, y x ∈ range_y :=
sorry

end trigonometric_function_range_l757_757800


namespace square_free_odd_integers_count_l757_757659

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757659


namespace hyperbola_midpoint_l757_757119

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757119


namespace initial_mean_l757_757796

theorem initial_mean (M : ℝ) (h1 : 50 * (36.5 : ℝ) - 23 = 50 * (36.04 : ℝ) + 23)
: M = 36.04 :=
by
  sorry

end initial_mean_l757_757796


namespace tan_minus_405_eq_neg1_l757_757937

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l757_757937


namespace assign_zero_ones_iff_4_divides_n_l757_757020

-- Define the problem using the conditions given
variables (n : ℕ) (A : fin (n + 2) → finset ℕ)
hypothesis h_even : even n
hypothesis h_size : ∀ i, (A i).card = n
hypothesis h_common : ∀ i j, i ≠ j → (A i ∩ A j).card = 1
hypothesis h_union : ∀ x ∈ (⋃ i, A i), ∃ i j, i ≠ j ∧ x ∈ A i ∧ x ∈ A j

definition can_assign_zero_ones (x : ℕ) : Prop :=
∃ f : ℕ → fin 2, (∀ i, (A i).filter (λ e, f e = 0)).card = n / 2

theorem assign_zero_ones_iff_4_divides_n :
  even n → (∀ i, (A i).card = n) → 
  (∀ i j, i ≠ j → (A i ∩ A j).card = 1) →
  (∀ x ∈ (⋃ i, A i), ∃ i j, i ≠ j ∧ x ∈ A i ∧ x ∈ A j) →
  (∃ x, can_assign_zero_ones x) ↔ 4 ∣ n :=
 by sorry

end assign_zero_ones_iff_4_divides_n_l757_757020


namespace find_coefficients_l757_757612

theorem find_coefficients 
  (solution_set_eq: set_of (λ x : ℝ, 5 - x > 7 * |x + 1|) = set_of (λ x : ℝ, -2 < x ∧ x < (-1/4)))
  (P : ℝ → ℝ := λ x, a * x^2 + b * x - 2)
  (roots : ∀ x : ℝ, P x > 0 ↔ (-2 < x ∧ x < (-1/4))) :
  (a = -4 ∧ b = -9) :=
by 
  sorry

end find_coefficients_l757_757612


namespace can_be_midpoint_of_AB_l757_757175

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757175


namespace number_of_five_ruble_coins_l757_757329

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757329


namespace correct_midpoint_l757_757213

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757213


namespace midpoint_hyperbola_l757_757113

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757113


namespace equal_chords_squared_sum_l757_757385

-- Condition: The points A1, A2, A3, A4, A5 divide a unit circle into five equal parts.
axiom points_on_circle (radius : ℝ) (split_points : ℕ → ℂ) : 
  radius = 1 → ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → abs (split_points n) = 1 ∧ split_points (n + 1) = split_points n * exp (2 * real.pi * complex.i / 5)

-- Define the specific points on the circle
noncomputable def A1 := complex.exp (0 * 2 * real.pi * complex.i / 5)
noncomputable def A2 := complex.exp (1 * 2 * real.pi * complex.i / 5)
noncomputable def A3 := complex.exp (2 * 2 * real.pi * complex.i / 5)

-- Define the distance functions for chords
def distance (z1 z2 : ℂ) : ℝ := abs (z1 - z2)
def chord_A1A2 := distance A1 A2
def chord_A1A3 := distance A1 A3

-- Proof statement
theorem equal_chords_squared_sum : chord_A1A2^2 + chord_A1A3^2 = 5 :=
sorry

end equal_chords_squared_sum_l757_757385


namespace midpoint_of_line_segment_on_hyperbola_l757_757032

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757032


namespace square_free_odd_integers_count_l757_757649

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757649


namespace hyperbola_midpoint_l757_757117

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757117


namespace find_b_l757_757727

def a : ℝ × ℝ × ℝ := (3, -6, 2)
def c : ℝ × ℝ × ℝ := (-1, 2, 0)
def c_new : ℝ × ℝ × ℝ := (0, 0, 1)
def b : ℝ × ℝ × ℝ := (0, 0, 0)

theorem find_b : (b.1 = 0 ∧ b.2 = 0 ∧ b.3 = 0) ∧
                 (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) ∧
                 (∃ t : ℝ, b = (t * c_new.1, t * c_new.2, t * c_new.3)) :=
by 
  -- proof omitted
  sorry

end find_b_l757_757727


namespace tan_neg405_deg_l757_757940

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l757_757940


namespace midpoint_of_hyperbola_segment_l757_757095

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757095


namespace definitely_incorrect_l757_757409

-- Definitions for conditions
def xor : ℕ → ℕ → ℕ
| 0, 0 => 0
| 0, 1 => 1
| 1, 0 => 1
| 1, 1 => 0

def h0 (a0 a1 : ℕ) : ℕ := xor a0 a1
def h1 (h0 a2 : ℕ) : ℕ := xor h0 a2

-- Encoded transmission
def transmission (a0 a1 a2 : ℕ) : List ℕ := [h0 a0 a1, a0, a1, a2, h1 (h0 a0 a1) a2]

-- Problem statement
theorem definitely_incorrect {a0 a1 a2 : ℕ} :
    ((a0, a1, a2) = (0, 1, 0) → transmission a0 a1 a2 = [1, 0, 1, 0, 1]) → transmission 0 1 0 ≠ [1, 0, 1, 0, 0] :=
by
  intros h
  -- Proof skipped
  sorry

end definitely_incorrect_l757_757409


namespace maximum_value_of_expression_l757_757496

theorem maximum_value_of_expression {a b c : ℝ} (hab : a > 0) (hbc : b > 0) (h : a^2 + b^2 = c^2) : 
  ∃ M, M = 1.5 ∧ ∀ x, x = (a^2 + b^2 + a * b) / c^2 → x ≤ M :=
begin
  sorry
end

end maximum_value_of_expression_l757_757496


namespace correct_midpoint_l757_757217

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757217


namespace evaluate_expression_l757_757526

theorem evaluate_expression : 
  -((5: ℤ) ^ 2) - (-(3: ℤ) ^ 3) * ((2: ℚ) / 9) - 9 * |((-(2: ℚ)) / 3)| = -25 := by
  sorry

end evaluate_expression_l757_757526


namespace study_time_difference_l757_757019

def hoursToMinutes (hours : ℝ) : ℝ := hours * 60

theorem study_time_difference :
  let kwame_study_time_hours : ℝ := 2.5;
  let connor_study_time_hours : ℝ := 1.5;
  let lexia_study_time_minutes : ℝ := 97;
  hoursToMinutes kwame_study_time_hours + hoursToMinutes connor_study_time_hours - lexia_study_time_minutes = 143 :=
by
  let kwame_study_time_minutes := hoursToMinutes 2.5
  let connor_study_time_minutes := hoursToMinutes 1.5
  let lexia_study_time_minutes := 97
  have h1 : kwame_study_time_minutes = 150 := by norm_num
  have h2 : connor_study_time_minutes = 90 := by norm_num
  have combined_study_time := h1 + h2
  have combined := combined_study_time - lexia_study_time_minutes
  show combined = 143 from by norm_num
  sorry

end study_time_difference_l757_757019


namespace midpoint_of_hyperbola_l757_757058

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757058


namespace positive_difference_l757_757394

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l757_757394


namespace monotonicity_max_value_l757_757628

/-- Define the function f(x) = x^2 * exp(ax) -/
def f (a x : ℝ) := x^2 * Real.exp (a * x)

theorem monotonicity (a : ℝ) (h : a ≤ 0) : 
  (∀ x > 0, f a x ≥ f a 0) ∨ 
  (∃ x₀, ∀ x, x ≥ x₀ → f a x ≤ f a x₀ ∧ ∀ x, x < x₀ → f a x > f a x₀) := 
sorry

theorem max_value (a : ℝ) (h : a ≤ 0) : 
  if a = 0 
  then ∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≤ f a 1 ∧ f a 1 = 1 
  else 
  if -2 < a ∧ a < 0 
  then ∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≤ f a 1 ∧ f a 1 = Real.exp a 
  else 
  a ≤ -2 → ∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≤ f a (-2/a) := 
sorry

end monotonicity_max_value_l757_757628


namespace room_height_l757_757534

-- Define the conditions
def total_curtain_length : ℕ := 101
def extra_material : ℕ := 5

-- Define the statement to be proven
theorem room_height : total_curtain_length - extra_material = 96 :=
by
  sorry

end room_height_l757_757534


namespace number_of_five_ruble_coins_l757_757330

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757330


namespace persons_complete_wall_in_days_l757_757864

noncomputable def numberOfDays (persons1 days1 length1 persons2 length2 : ℕ) : ℝ :=
  (persons1 * days1 * length2 : ℝ) / (length1 * persons2 : ℝ)

theorem persons_complete_wall_in_days :
    numberOfDays 8 8 140 30 100 ≈ 1.524 := sorry

end persons_complete_wall_in_days_l757_757864


namespace tan_minus_405_eq_neg1_l757_757935

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l757_757935


namespace hyperbola_midpoint_l757_757125

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757125


namespace polynomial_form_l757_757967

theorem polynomial_form (P : ℝ → ℝ) (h₁ : P 0 = 0) (h₂ : ∀ x, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_form_l757_757967


namespace floor_sqrt_150_l757_757959

theorem floor_sqrt_150 : (⌊Real.sqrt 150⌋ = 12) := 
by {
  have sqrt_144_eq_12 : Real.sqrt 144 = 12 := by sorry,
  have sqrt_169_eq_13 : Real.sqrt 169 = 13 := by sorry,
  have lt_144_150_169 : 144 < 150 ∧ 150 < 169 := by sorry,
  have h1 : Real.sqrt 144 < Real.sqrt 150 := by linarith,
  have h2 : Real.sqrt 150 < Real.sqrt 169 := by linarith,
  have h3 : 12 < Real.sqrt 150 := by linarith,
  have h4 : Real.sqrt 150 < 13 := by linarith,
  exact sorry
}

end floor_sqrt_150_l757_757959


namespace focus_of_parabola_l757_757998

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757998


namespace probability_X_eq_Y_correct_l757_757901

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := -20 * Real.pi
  let upper_bound := 20 * Real.pi
  let total_pairs := (upper_bound - lower_bound) * (upper_bound - lower_bound)
  let matching_pairs := 81
  matching_pairs / total_pairs

theorem probability_X_eq_Y_correct :
  probability_X_eq_Y = 81 / 1681 :=
by
  unfold probability_X_eq_Y
  sorry

end probability_X_eq_Y_correct_l757_757901


namespace sequence_0_is_arithmetic_not_geometric_l757_757802

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = 0

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def sequence_0 : ℕ → ℝ := λ n, 0

theorem sequence_0_is_arithmetic_not_geometric :
  is_arithmetic_sequence sequence_0 ∧ ¬ is_geometric_sequence sequence_0 :=
sorry

end sequence_0_is_arithmetic_not_geometric_l757_757802


namespace major_axis_of_ellipse_is_12_l757_757886

noncomputable def major_axis_length (r : ℝ) (h : r = 3) : ℝ :=
  let minor_axis := 2 * r in
  let major_axis := 2 * minor_axis in
  major_axis

theorem major_axis_of_ellipse_is_12 : major_axis_length 3 (by rfl) = 12 := 
sorry

end major_axis_of_ellipse_is_12_l757_757886


namespace find_center_and_shortest_chord_l757_757614

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x + 2 * y - 2 = 0

def center_of_circle (cx cy : ℝ) : Prop :=
  circle_equation (cx + 1) (cy + 1)

def shortest_chord_line (k : ℝ) (x y : ℝ) : Prop :=
  x + y = 0

theorem find_center_and_shortest_chord :
  (∃ cx cy, center_of_circle cx cy ∧ (cx, cy) = (-1, -1)) ∧
  (∃ k, k = -1 ∧ shortest_chord_line k)
:= by sorry

end find_center_and_shortest_chord_l757_757614


namespace part_a_part_b_part_c_l757_757276

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l757_757276


namespace parabola_focus_l757_757983

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757983


namespace lines_are_concurrent_l757_757793

-- Definitions for the given points and segments
variables {A B C D C1 A1 B1 A2 B2 C2 : Type*}
variables [plane A B C D]
variables [midpoint C1 A B]
variables [midpoint A1 B C]
variables [midpoint B1 C A]
variables [divides_in_ratio A2 D A 2 1]
variables [divides_in_ratio B2 D B 2 1]
variables [divides_in_ratio C2 D C 2 1]

-- Theorem statement
theorem lines_are_concurrent :
  ∃ (M : Type*), concurrent A1 A2 B1 B2 C1 C2 M :=
sorry

end lines_are_concurrent_l757_757793


namespace constant_term_of_product_l757_757416

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + 7
def poly2 (x : ℝ) : ℝ := 4 * x^4 + 2 * x^2 + 10

-- Main statement: Prove that the constant term in the expansion of poly1 * poly2 is 70
theorem constant_term_of_product : (poly1 0) * (poly2 0) = 70 :=
by
  -- The proof would go here
  sorry

end constant_term_of_product_l757_757416


namespace smallest_invertible_domain_g_l757_757739

theorem smallest_invertible_domain_g (g : ℝ → ℝ) :
  (∀ x, g x = (2 * x - 3)^2 - 4) →
  (∀ x ∈ (set.Ici (3 / 2 : ℝ)), strict_mono_on g (set.Ici x)) →
  is_invertible (set.Ici (3 / 2 : ℝ)) g :=
sorry

end smallest_invertible_domain_g_l757_757739


namespace square_free_odd_integers_count_l757_757666

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757666


namespace scientific_notation_63000_l757_757005

theorem scientific_notation_63000 : 63000 = 6.3 * 10^4 :=
by
  sorry

end scientific_notation_63000_l757_757005


namespace sum_in_base_4_l757_757507

theorem sum_in_base_4 : 
  let n1 := 2
  let n2 := 23
  let n3 := 132
  let n4 := 1320
  let sum := 20200
  n1 + n2 + n3 + n4 = sum := 
by
  sorry

end sum_in_base_4_l757_757507


namespace maggie_travel_distance_l757_757263

theorem maggie_travel_distance
  (mileage_per_gallon : ℕ)
  (price_per_gallon : ℕ)
  (total_money : ℕ) :
  mileage_per_gallon = 32 →
  price_per_gallon = 4 →
  total_money = 20 →
  (total_money / price_per_gallon) * mileage_per_gallon = 160 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have gallons := 20 / 4
  have distance := gallons * 32
  show distance = 160
  rw [←nat.mul_div_cancel 20 (nat.pos_of_ne_zero $ by norm_num)]
  norm_num
  rw [←mul_comm]
  rfl

end maggie_travel_distance_l757_757263


namespace hyperbola_midpoint_exists_l757_757248

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757248


namespace proposition1_proposition3_l757_757577

-- Definitions
variables {Plane : Type} {Line : Type}
variables (l : Line) (m : Line) (α : Plane) (β : Plane)

-- Conditions
variable (perpendicular_l_alpha : l ⊥ α)
variable (m_in_beta : m ∈ β)

-- Propositions to prove
theorem proposition1 (parallel_alpha_beta : α ∥ β) : l ⊥ m :=
sorry

theorem proposition3 (parallel_l_m : l ∥ m) : α ⊥ β :=
sorry

end proposition1_proposition3_l757_757577


namespace hyperbola_midpoint_l757_757120

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757120


namespace find_d_value_l757_757400

theorem find_d_value (d : ℝ) :
  let a := 1
  let b := -10
  let h := -b / (2 * a)
  let k := d - (b^2) / (4 * a)
  h = 5 ∧ k = 2 * h → d = 35 :=
by
  -- assume h = 5
  let h := 5
  -- condition for vertex lying on the line y = 2x
  have k_eq : k = 10 := by
    sorry
  -- actual proving step skipped
  sorry

end find_d_value_l757_757400


namespace num_five_ruble_coins_l757_757317

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757317


namespace midpoint_of_hyperbola_l757_757060

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757060


namespace highest_vertex_value_in_pyramid_value_vertex_specified_in_cube_value_vertex_A_in_octahedron_value_vertex_B_in_octahedron_l757_757791

-- Define the maximum value of the vertex of the pyramid
def max_vertex_value_pyramid (faces : List ℕ) : ℕ :=
  faces.foldl (+) 0

-- Define the vertex values at specific vertices in the solids
def vertex_value_cube (faces : List ℕ) : ℕ :=
  faces.foldl (+) 0

def vertex_value_octahedron (faces : List ℕ) : ℕ :=
  faces.foldl (+) 0

-- Statements to prove:

theorem highest_vertex_value_in_pyramid : 
  max_vertex_value_pyramid [2, 3, 4] = 9 := 
by
  sorry

theorem value_vertex_specified_in_cube : 
  vertex_value_cube [3, 6, 2] = 11 := 
by
  sorry

theorem value_vertex_A_in_octahedron : 
  vertex_value_octahedron [4, 5, 6, 7] = 22 := 
by
  sorry

theorem value_vertex_B_in_octahedron : 
  vertex_value_octahedron [1, 2, 4, 5] = 12 := 
by
  sorry

end highest_vertex_value_in_pyramid_value_vertex_specified_in_cube_value_vertex_A_in_octahedron_value_vertex_B_in_octahedron_l757_757791


namespace find_m_plus_nk_l757_757704

-- Definitions of points and other constants
def triangle_ABC (A B C : Type) [triangle ABC] :=
∀ A B C: ℝ, AB = 108 ∧ AC = 108 ∧ BC = 60

def circle_P (P : Type) [circle P] := 
∀ P, radius P = 18 ∧ tangent P (line AC) ∧ tangent P (line BC)

def circle_Q (Q : Type) [circle Q] := 
∀ Q, ∃ r : ℝ, 
  r = 48 - 6 * real.sqrt 39 ∧ externally_tangent Q P ∧ tangent Q (line AB) ∧ tangent Q (line BC) ∧ lies_within Q (triangle ABC)

theorem find_m_plus_nk : 
  (∃ m n k : ℕ , 
    ∀ Q : Type, [circle Q] ∧ radius Q = (m - n * real.sqrt k).to_real ∧ k = 39 ) 
  → (48 + 6 * 39 = 282):=
by
  sorry

end find_m_plus_nk_l757_757704


namespace hyperbola_midpoint_l757_757153

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757153


namespace no_adjacent_repeats_across_meetings_l757_757268

def meeting1 := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def meeting2 := [1, 3, 5, 7, 9, 4, 6, 2, 8]
def meeting3 := [1, 4, 7, 3, 8, 5, 2, 9, 6]
def meeting4 := [1, 5, 9, 3, 6, 8, 4, 2, 7]

def adjacent_pairs(lst : List Nat) : List (Nat × Nat) :=
  (lst.zip (lst.tail ++ [lst.head])).map (λ ⟨a, b⟩ => (a, b))

theorem no_adjacent_repeats_across_meetings :
  (∀ pair ∈ adjacent_pairs meeting1, pair ∉ adjacent_pairs meeting2) ∧
  (∀ pair ∈ adjacent_pairs meeting1, pair ∉ adjacent_pairs meeting3) ∧
  (∀ pair ∈ adjacent_pairs meeting1, pair ∉ adjacent_pairs meeting4) ∧
  (∀ pair ∈ adjacent_pairs meeting2, pair ∉ adjacent_pairs meeting3) ∧
  (∀ pair ∈ adjacent_pairs meeting2, pair ∉ adjacent_pairs meeting4) ∧
  (∀ pair ∈ adjacent_pairs meeting3, pair ∉ adjacent_pairs meeting4) :=
by
  sorry

end no_adjacent_repeats_across_meetings_l757_757268


namespace fraction_relationships_l757_757674

variables (a b c d : ℚ)

theorem fraction_relationships (h1 : a / b = 3) (h2 : b / c = 2 / 3) (h3 : c / d = 5) :
  d / a = 1 / 10 :=
by
  sorry

end fraction_relationships_l757_757674


namespace max_marks_for_test_l757_757853

theorem max_marks_for_test (M : ℝ) (h1: (0.30 * M) = 180) : M = 600 :=
by
  sorry

end max_marks_for_test_l757_757853


namespace parabola_focus_l757_757981

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757981


namespace midpoint_of_hyperbola_l757_757059

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757059


namespace find_larger_integer_l757_757388

noncomputable def larger_integer (a b : ℕ) : Prop :=
  a * b = 189 ∧ (b = (7 * a) / 3⁷) / 3

theorem find_larger_integer (a b : ℕ) (h1 : a * b = 189) (h2 : a * 7 = 3 * b) :
  b = 21 :=
by
  sorry

end find_larger_integer_l757_757388


namespace total_tomatoes_l757_757015

def tomatoes_first_plant : Nat := 2 * 12
def tomatoes_second_plant : Nat := (tomatoes_first_plant / 2) + 5
def tomatoes_third_plant : Nat := tomatoes_second_plant + 2

theorem total_tomatoes :
  (tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant) = 60 := by
  sorry

end total_tomatoes_l757_757015


namespace point_to_line_distance_correct_l757_757788

noncomputable def distance_from_point_to_line (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  let (x₁, y₁) := P in
  abs (A * x₁ + B * y₁ + C) / Real.sqrt (A^2 + B^2)

def point_P : ℝ × ℝ := (2, 0)

def line_standard_form : ℝ × ℝ × ℝ := (3, -4, 5) -- 3x - 4y + 5 = 0

theorem point_to_line_distance_correct :
  distance_from_point_to_line point_P line_standard_form.1 line_standard_form.2 line_standard_form.3 = 11 / 5 :=
by
  sorry

end point_to_line_distance_correct_l757_757788


namespace repeating_decimal_value_l757_757677

noncomputable def repeating_decimal_to_fraction (a : Real) :=
  let x := a
  let numerator := 3769.3769
  let denominator := 9999
  numerator / denominator

theorem repeating_decimal_value :
  (10 ^ 8 - 10 ^ 4) * 0.00003769 = 3765230.6231 :=
by
  have h1 : repeating_decimal_to_fraction 0.00003769 = 3769.3769 / 9999 := sorry
  have h2 : (10 ^ 8) * 0.00003769 = 3769000 := sorry
  have h3 : (10 ^ 4) * 0.00003769 = 3769.3769 := sorry
  rw [←mul_sub, h2, h3]
  norm_num
  sorry

end repeating_decimal_value_l757_757677


namespace number_divided_by_three_l757_757443

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l757_757443


namespace midpoint_on_hyperbola_l757_757082

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757082


namespace num_five_ruble_coins_l757_757312

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757312


namespace probability_of_winning_l757_757943

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def in_range (ns : List ℕ) : Prop :=
  ∀ n ∈ ns, 1 ≤ n ∧ n ≤ 30

def arithmetic_progression (ns : List ℕ) : Prop :=
  ∃ d m, ns = List.range' 5 (m + d) d

def sum_of_log_base_two_is_integer (ns : List ℕ) : Prop :=
  let log_sum := ns.map (λ x => Real.logb 2 x) |> List.sum
  log_sum ∈ Int

theorem probability_of_winning (ns : List ℕ) :
  in_range ns →
  arithmetic_progression ns →
  sum_of_log_base_two_is_integer ns →
  ∃ m : ℝ, m = 1 :=
by
  intro h1 h2 h3
  sorry

end probability_of_winning_l757_757943


namespace total_animals_made_it_to_shore_l757_757867

def boat (total_sheep total_cows total_dogs sheep_drowned cows_drowned dogs_saved : Nat) : Prop :=
  cows_drowned = sheep_drowned * 2 ∧
  dogs_saved = total_dogs ∧
  total_sheep + total_cows + total_dogs - sheep_drowned - cows_drowned = 35

theorem total_animals_made_it_to_shore :
  boat 20 10 14 3 6 14 :=
by
  sorry

end total_animals_made_it_to_shore_l757_757867


namespace five_ruble_coins_count_l757_757305

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757305


namespace lilly_daily_savings_l757_757747

-- Conditions
def days_until_birthday : ℕ := 22
def flowers_to_buy : ℕ := 11
def cost_per_flower : ℕ := 4

-- Definition we want to prove
def total_cost : ℕ := flowers_to_buy * cost_per_flower
def daily_savings : ℕ := total_cost / days_until_birthday

theorem lilly_daily_savings : daily_savings = 2 := by
  sorry

end lilly_daily_savings_l757_757747


namespace probability_product_odd_l757_757474

theorem probability_product_odd :
  let A := {1, 2, 3}
  let B := {0, 1, 3}
  let outcomes := 3 * 3
  let favorable_outcomes := 2 * 2
  (favorable_outcomes : ℚ) / outcomes = 4 / 9 := by
{
  -- Definitions for A and B
  let A := {1, 2, 3}
  let B := {0, 1, 3}
  -- Total number of outcomes
  let outcomes := 3 * 3
  -- Favorable outcomes where both a and b are odd
  let favorable_outcomes := 2 * 2
  -- Expected probability
  have h : (favorable_outcomes : ℚ) / outcomes = 4 / 9 := sorry
  exact h
}

end probability_product_odd_l757_757474


namespace largest_possible_m_l757_757383

theorem largest_possible_m (x y : ℕ) (h1 : x > y) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < 10) (hyy : y < 10) (h_prime_10xy : Nat.Prime (10 * x + y)) : ∃ m : ℕ, m = x * y * (10 * x + y) ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ ∀ n : ℕ, (n = x * y * (10 * x + y) ∧ 1000 ≤ n ∧ n ≤ 9999) → n ≤ 1533 :=
by
  sorry

end largest_possible_m_l757_757383


namespace midpoint_of_line_segment_on_hyperbola_l757_757029

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757029


namespace hyperbola_midpoint_l757_757194

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757194


namespace solve_system_of_equations_l757_757356

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - y = 2 ∧ 3 * x + y = 4 ∧ x = 1.5 ∧ y = -0.5 :=
by
  sorry

end solve_system_of_equations_l757_757356


namespace solveEquation_l757_757844

noncomputable def findNonZeroSolution (z : ℝ) : Prop :=
  (5 * z) ^ 10 = (20 * z) ^ 5 ∧ z ≠ 0

theorem solveEquation : ∃ z : ℝ, findNonZeroSolution z ∧ z = 4 / 5 := by
  exists 4 / 5
  simp [findNonZeroSolution]
  sorry

end solveEquation_l757_757844


namespace remainder_3_pow_19_mod_10_l757_757420

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_3_pow_19_mod_10_l757_757420


namespace symmetric_point_origin_l757_757798

theorem symmetric_point_origin (x y z : ℤ) (h : (x, y, z) = (-1, 2, -3)) :
  (-x, -y, -z) = (1, -2, 3) :=
by
  cases h
  simp
  sorry

end symmetric_point_origin_l757_757798


namespace speed_of_stream_l757_757880

-- Definitions
variable (b s : ℝ)
def downstream_distance : ℝ := 120
def downstream_time : ℝ := 4
def upstream_distance : ℝ := 90
def upstream_time : ℝ := 6

-- Equations
def downstream_eq : Prop := downstream_distance = (b + s) * downstream_time
def upstream_eq : Prop := upstream_distance = (b - s) * upstream_time

-- Main statement
theorem speed_of_stream (h₁ : downstream_eq b s) (h₂ : upstream_eq b s) : s = 7.5 :=
by
  sorry

end speed_of_stream_l757_757880


namespace regular_ngon_max_area_regular_ngon_max_perimeter_l757_757472

open Classical

-- Problem 57 (a)
theorem regular_ngon_max_area {n : ℕ} (n_gt_2 : n > 2) (R : ℝ) (hsr: R > 0) (ps : List (ℝ × ℝ)) (hips : ∀ (p : (ℝ × ℝ)), p ∈ ps → p.1 ^ 2 + p.2 ^ 2 = R^2) :
    ∃ (regular_ngon : List (ℝ × ℝ)), 
    (∀ (pi ∈ regular_ngon), ∃ k : ℤ, pi = (R * cos ((2 * k * π) / n), R * sin ((2 * k * π) / n))) ∧
    (∀ (ngon : List (ℝ × ℝ)), 
        (∀ (pi ∈ ngon), pi.1 ^ 2 + pi.2 ^ 2 = R^2) → 
        area regular_ngon ≥ area ngon) :=
sorry

-- Problem 57 (b)
theorem regular_ngon_max_perimeter {n : ℕ} (n_gt_2 : n > 2) (R : ℝ) (hsr: R > 0) (ps : List (ℝ × ℝ)) (hips : ∀ (p : (ℝ × ℝ)), p ∈ ps → p.1 ^ 2 + p.2 ^ 2 = R^2) :
    ∃ (regular_ngon : List (ℝ × ℝ)), 
    (∀ (pi ∈ regular_ngon), ∃ k : ℤ, pi = (R * cos ((2 * k * π) / n), R * sin ((2 * k * π) / n))) ∧
    (∀ (ngon : List (ℝ × ℝ)), 
        (∀ (pi ∈ ngon), pi.1 ^ 2 + pi.2 ^ 2 = R^2) → 
        perimeter regular_ngon ≥ perimeter ngon) :=
sorry

end regular_ngon_max_area_regular_ngon_max_perimeter_l757_757472


namespace solve_system_of_equations_l757_757778

theorem solve_system_of_equations :
  ∃ x y : ℝ, (log 2 (y - x) = logBase 8 (19 * y - 13 * x) ∧ x^2 + y^2 = 13) ↔
  (x = -real.sqrt 13 ∧ y = 0) ∨ (x = -3 ∧ y = -2) ∨ (x = -1 / real.sqrt 2 ∧ y = 5 / real.sqrt 2) :=
by
  sorry

end solve_system_of_equations_l757_757778


namespace find_x_l757_757424

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l757_757424


namespace chess_club_team_formation_l757_757684

theorem chess_club_team_formation (girls boys : ℕ) (total_team : ℕ) 
    (h_girls : girls = 2) (h_boys : boys = 7) (h_team : total_team = 4) :
    ∃ ways : ℕ, ways = 91 :=
by {
  have h1 : ∃ ways, ways = nat.choose 2 1 * nat.choose 7 3 + nat.choose 2 2 * nat.choose 7 2,
  exact ⟨91, by norm_num⟩,
  exact h1
}

end chess_club_team_formation_l757_757684


namespace midpoint_hyperbola_l757_757231

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757231


namespace find_concentric_tangent_circle_l757_757975

theorem find_concentric_tangent_circle :
  ∃ (r : ℝ) (h: r = √5), 
    (∀ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = r ^ 2) :=
begin
  sorry
end

end find_concentric_tangent_circle_l757_757975


namespace midpoint_on_hyperbola_l757_757071

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757071


namespace fraction_of_students_with_buddy_l757_757687

variables (f e : ℕ)
-- Given:
axiom H1 : e / 4 = f / 3

-- Prove:
theorem fraction_of_students_with_buddy : 
  (e / 4 + f / 3) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l757_757687


namespace square_free_odd_integers_count_l757_757668

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757668


namespace integer_solutions_count_l757_757374

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ x y, (x, y) ∈ S ↔ x^2 + x * y + 2 * y^2 = 29) ∧ 
  S.card = 4 := 
sorry

end integer_solutions_count_l757_757374


namespace petya_five_ruble_coins_l757_757298

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757298


namespace find_CM_l757_757481

theorem find_CM length_A_B (AB CD : ℝ) (c : ℝ) (hypotenuse : CD = sqrt (c^2 - CD^2))  
(CM block_size) where:
  - length_A_B: AB = c
  - hypotenuse: CD = sqrt (c^2 - CD^2)
  - block_size: CM = sqrt (CD^2 + (c / √2)^2) 
  --Given the congruent properties of triangle and distance, CM is calculated.
  --Calculation hold true with CD addition of altitude mark to midpoint
  - equal_CM: CM = c / 2 √2 :=
assume h1: length_A_B     -- prove length_A_B
  by sorry

assume h2: hypotenuse     -- prove hypotenuse is correct
  by sorry

have equal_CM: find_CM (CD: equal_CM) :=
--assert same solution to be true satisfying properties of completion.
assume h3: equal_CM where
 prove final equality.
by sorry

end find_CM_l757_757481


namespace height_difference_l757_757716

theorem height_difference (h_CN : ℕ) (diff_CN_SN : ℕ) (h_ET : ℕ)
    (h_CN_val : h_CN = 553)
    (diff_CN_SN_val : diff_CN_SN = 369)
    (h_ET_val : h_ET = 330) :
    h_ET - (h_CN - diff_CN_SN) = 146 := by
  rw [h_CN_val, diff_CN_SN_val, h_ET_val]
  dsimp
  norm_num

end height_difference_l757_757716


namespace find_number_l757_757440

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l757_757440


namespace AX_equals_XC_plus_CB_l757_757505

variables {A B C M X : Type*}
variables [triangle : Triangle ABC] (ACgtBC : AClength > BClength)
variables [circumcircle : Circumcircle ABC] (midpointM : MidpointArc M A B C)
variables (perpMXAC : Perpendicular MX AC)

theorem AX_equals_XC_plus_CB
  (HX : PointOnLine X AC perpMXAC):
  Alength X = XClength + CB :
  sorry

end AX_equals_XC_plus_CB_l757_757505


namespace problem_statement_l757_757848

theorem problem_statement 
  (regression_line : ∀ (x : ℝ), ∃ (b a : ℝ), ∀ (x̄ ȳ : ℝ), (x̄, ȳ) ∉ {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)})
  (correlation_coefficient : ∀ (r : ℝ), (abs r = 1) → (linear_correlation r))
  (binomial_distribution : ∀ (X : ℝ), (E_X = 30) ∧ (D_X = 20) → (p = 1 / 3))
  (normal_distribution : ∀ (ξ : ℝ), (P (ξ > 1) = p) → (P (-1 < ξ < 0) = (1 / 2) - p))
  : (correct_statement_B : ∀ (r : ℝ), (abs r = 1) → (linear_correlation r))
  ∧ (correct_statement_C : ∀ (X : ℝ), (E_X = 30) ∧ (D_X = 20) → (p = 1 / 3))
  ∧ (correct_statement_D : ∀ (ξ : ℝ), (P (ξ > 1) = p) → (P (-1 < ξ < 0) = (1 / 2) - p))
  ∧ ¬ (correct_statement_A : (∃ (b a : ℝ), (∀ (x̄ ȳ : ℝ), (x̄, ȳ) ∈ {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)})))
  :=
sorry

end problem_statement_l757_757848


namespace hyperbola_midpoint_l757_757157

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757157


namespace ads_minutes_l757_757903

-- Definitions and conditions
def videos_per_day : Nat := 2
def minutes_per_video : Nat := 7
def total_time_on_youtube : Nat := 17

-- The theorem to prove
theorem ads_minutes : (total_time_on_youtube - (videos_per_day * minutes_per_video)) = 3 :=
by
  sorry

end ads_minutes_l757_757903


namespace day_of_week_after_m_days_is_monday_l757_757255

theorem day_of_week_after_m_days_is_monday:
  let m := ∑ i in Finset.range (2003 + 1), i^2
  (m % 7 = 1) → "Monday" = "Monday" :=
by
  intros
  have h : m % 7 = 1 := sorry
  sorry

end day_of_week_after_m_days_is_monday_l757_757255


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757285

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757285


namespace exists_second_degree_polynomial_l757_757003

theorem exists_second_degree_polynomial :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), (x = 0 → y = 100) →
    (x = 1 → y = 90) →
    (x = 2 → y = 70) →
    (x = 3 → y = 40) →
    (x = 4 → y = 0) →
    (y = a * x^2 + b * x + c)) :=
begin
  use [-5, -5, 100],
  intros x y h0 h1 h2 h3 h4,
  split_ifs;
  {
    simp at *,
    assumption
  },
  sorry
end

end exists_second_degree_polynomial_l757_757003


namespace tan_theta_minus_pi_over_4_l757_757602

theorem tan_theta_minus_pi_over_4 (theta : ℝ) (h1 : π/2 < θ ∧ θ < 2π)
  (h2 : Real.sin (theta + π/4) = -3/5) : Real.tan (theta - π/4) = 4/3 :=
by
  sorry

end tan_theta_minus_pi_over_4_l757_757602


namespace midpoint_hyperbola_l757_757099

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757099


namespace midpoint_of_hyperbola_l757_757064

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757064


namespace joan_marbles_l757_757749

def mary_marbles : ℕ := 9
def total_marbles : ℕ := 12

theorem joan_marbles : total_marbles - mary_marbles = 3 :=
by
  simp [total_marbles, mary_marbles]
  norm_num

end joan_marbles_l757_757749


namespace midpoint_of_hyperbola_l757_757159

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757159


namespace find_larger_integer_l757_757389

noncomputable def larger_integer (a b : ℕ) : Prop :=
  a * b = 189 ∧ (b = (7 * a) / 3⁷) / 3

theorem find_larger_integer (a b : ℕ) (h1 : a * b = 189) (h2 : a * 7 = 3 * b) :
  b = 21 :=
by
  sorry

end find_larger_integer_l757_757389


namespace hyperbola_midpoint_l757_757114

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757114


namespace midpoint_of_hyperbola_segment_l757_757094

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757094


namespace hyperbola_midpoint_l757_757146

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757146


namespace midpoint_of_hyperbola_l757_757173

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757173


namespace midpoint_of_line_segment_on_hyperbola_l757_757033

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757033


namespace square_free_odd_integers_count_l757_757667

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757667


namespace gwen_more_money_from_mom_l757_757555

def dollars_received_from_mom : ℕ := 7
def dollars_received_from_dad : ℕ := 5

theorem gwen_more_money_from_mom :
  dollars_received_from_mom - dollars_received_from_dad = 2 :=
by
  sorry

end gwen_more_money_from_mom_l757_757555


namespace midpoint_of_hyperbola_l757_757062

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757062


namespace common_area_at_least_two_thirds_l757_757369

theorem common_area_at_least_two_thirds (T1 T2 : set ℝ) (h1 : is_equilateral_triangle T1) (h2 : is_equilateral_triangle T2) (h3 : area T1 = 1) (h4 : area T2 = 1) (h5 : centers_coincide T1 T2) : 
  2 / 3 ≤ area (T1 ∩ T2) := 
sorry

end common_area_at_least_two_thirds_l757_757369


namespace coefficient_x2_l757_757971

theorem coefficient_x2 : (coeff (polynomial.expand (1 - x) ^ 2 (1 - (polynomial.sqrt x)) ^ 4) 2) = 15 :=
by
suffices : tsum of flattened and sorted expr
sorry

end coefficient_x2_l757_757971


namespace hyperbola_midpoint_l757_757191

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757191


namespace can_be_midpoint_of_AB_l757_757183

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757183


namespace determine_r_plus_s_l757_757380

variable (r s : ℕ → ℕ)

/-- The conditions of the problem -/
def conditions : Prop :=
  ∃ r s : ℕ → ℕ, s 4 = 4 ∧ r 3 = 3 ∧ ∀ x, s x = x^3 - 3*x^2 + 2*x

theorem determine_r_plus_s :
  ∃ r s : ℕ → ℕ, s 4 = 4 ∧ r 3 = 3 ∧ (∀ x, (r x + s x = x^3 - 3*x^2 + 3*x)) :=
begin
  sorry,
end

end determine_r_plus_s_l757_757380


namespace find_number_l757_757432

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l757_757432


namespace quadratic_has_two_zeros_l757_757560

theorem quadratic_has_two_zeros {a b c : ℝ} (h : a * c < 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end quadratic_has_two_zeros_l757_757560


namespace parabola_chord_slope_angle_l757_757607

theorem parabola_chord_slope_angle 
  (x y : ℝ) 
  (y1 y2 x1 x2 : ℝ) 
  (h_focus_eq : (1.5, 0))
  (h_parabola : ∀ x y, y ^ 2 = 6 * x) 
  (h_chord_length : √((x1 - x2) ^ 2 + (y1 - y2)^2) = 12) 
  (h_midpoint : (x1 + x2) / 2 = 1.5 ∧ (y1 + y2) / 2 = 0) :
  tan (arctan ((y1 - y2) / (x1 - x2))) = 1/4 ∨ tan (arctan ((y1 - y2) / (x1 - x2))) = -1/4 :=
sorry

end parabola_chord_slope_angle_l757_757607


namespace number_divided_by_three_l757_757444

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l757_757444


namespace tan_neg405_deg_l757_757941

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l757_757941


namespace third_term_of_binomial_expansion_l757_757806

theorem third_term_of_binomial_expansion :
  ∀ (x : ℕ), (binomial_expansion : ℕ → ℕ → List ℕ),
  binomial_expansion(5, x + 2) !! 2 = 40 * x ^ 3 :=
by
  sorry

end third_term_of_binomial_expansion_l757_757806


namespace midpoint_on_hyperbola_l757_757042

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757042


namespace find_starting_number_l757_757870

theorem find_starting_number (x : ℝ) (h : ((x - 2 + 4) / 1) / 2 * 8 = 77) : x = 17.25 := by
  sorry

end find_starting_number_l757_757870


namespace midpoint_hyperbola_l757_757108

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757108


namespace thomas_additional_weight_cost_l757_757406

def initial_weight : ℝ := 60
def weight_increase_percentage : ℝ := 0.60
def ingot_weight : ℝ := 2
def ingot_cost : ℝ := 5

def discount (num_ingots : ℕ) : ℝ :=
  if num_ingots ≥ 11 ∧ num_ingots ≤ 20 then 0.20
  else if num_ingots ≥ 21 ∧ num_ingots ≤ 30 then 0.25
  else if num_ingots ≥ 31 then 0.30
  else 0

def sales_tax (num_ingots : ℕ) : ℝ :=
  if num_ingots ≤ 20 then 0.05
  else if num_ingots ≥ 21 ∧ num_ingots ≤ 30 then 0.03
  else if num_ingots ≥ 31 then 0.01
  else 0

def shipping_fee (total_weight : ℝ) : ℝ :=
  if total_weight ≤ 20 then 10
  else if total_weight ≥ 21 ∧ total_weight ≤ 40 then 15
  else 20

def additional_weight : ℝ := initial_weight * weight_increase_percentage
def num_ingots : ℕ := (additional_weight / ingot_weight).toNat

def initial_cost : ℝ := num_ingots * ingot_cost
def discounted_price : ℝ := initial_cost * (1 - discount(num_ingots))
def tax_amount : ℝ := discounted_price * sales_tax(num_ingots)
def price_after_tax : ℝ := discounted_price + tax_amount
def final_cost : ℝ := price_after_tax + shipping_fee(additional_weight)

theorem thomas_additional_weight_cost :
  final_cost = 90.60 := by
  sorry

end thomas_additional_weight_cost_l757_757406


namespace find_n_l757_757455

theorem find_n :
  let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b := (2 * n : ℕ)
  (a*a - b*b = 0) -> (n = 12) := 
by 
  let a := 24
  let b := 2*n
  sorry

end find_n_l757_757455


namespace steve_final_height_l757_757364

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l757_757364


namespace steve_height_equiv_l757_757360

/-- 
  Steve's initial height in feet and inches.
  convert_height: converts feet and inches to total inches.
  grows further: additional height Steve grows.
  expected height: the expected total height after growing.
--/

def initial_height_feet := 5
def initial_height_inches := 6
def additional_height := 6

def convert_height(feet: Int, inches: Int): Int := 
  feet * 12 + inches

def expected_height(initial_feet: Int, initial_inches: Int, additional: Int): Int := 
  convert_height(initial_feet, initial_inches) + additional

theorem steve_height_equiv:
  expected_height initial_height_feet initial_height_inches additional_height = 72 :=
by
  sorry

end steve_height_equiv_l757_757360


namespace total_age_of_siblings_in_10_years_l757_757815

theorem total_age_of_siblings_in_10_years (age_eldest : ℕ) (gap : ℕ) (h1 : age_eldest = 20) (h2 : gap = 5) :
  let age_second := age_eldest - gap,
      age_youngest := age_second - gap in
  age_eldest + 10 + (age_second + 10) + (age_youngest + 10) = 75 :=
by
  sorry

end total_age_of_siblings_in_10_years_l757_757815


namespace students_who_liked_both_l757_757685

theorem students_who_liked_both (total_students : ℕ) (liked_apple : ℕ) (liked_chocolate : ℕ) (liked_neither : ℕ) :
  total_students = 35 → liked_apple = 20 → liked_chocolate = 17 → liked_neither = 10 →
  (liked_apple + liked_chocolate - (total_students - liked_neither)) = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end students_who_liked_both_l757_757685


namespace algebra_books_cannot_be_determined_uniquely_l757_757490

theorem algebra_books_cannot_be_determined_uniquely (A H S M E : ℕ) (pos_A : A > 0) (pos_H : H > 0) (pos_S : S > 0) 
  (pos_M : M > 0) (pos_E : E > 0) (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ S ≠ M ∧ S ≠ E ∧ M ≠ E) 
  (cond1: S < A) (cond2: M > H) (cond3: A + 2 * H = S + 2 * M) : 
  E = 0 :=
sorry

end algebra_books_cannot_be_determined_uniquely_l757_757490


namespace correct_midpoint_l757_757215

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757215


namespace cosine_identity_l757_757673

theorem cosine_identity (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (π / 2 + α) = -1 / 3 := by
  sorry

end cosine_identity_l757_757673


namespace total_books_proof_l757_757561

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l757_757561


namespace number_of_boys_l757_757754

-- Definitions for the given conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := 20
def total_girls := 41
def happy_boys := 6
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

-- Define the total number of boys
def total_boys := total_children - total_girls

-- Proof statement
theorem number_of_boys : total_boys = 19 :=
  by
    sorry

end number_of_boys_l757_757754


namespace find_a5_l757_757583

variable {α : Type*} [Field α]

def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

theorem find_a5 (a q : α) 
  (h1 : geometric_seq a q 2 = 4)
  (h2 : geometric_seq a q 6 * geometric_seq a q 7 = 16 * geometric_seq a q 9) :
  geometric_seq a q 5 = 32 ∨ geometric_seq a q 5 = -32 :=
by
  -- Proof is omitted as per instructions
  sorry

end find_a5_l757_757583


namespace tan_neg_405_eq_neg1_l757_757930

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l757_757930


namespace domain_of_f_l757_757974

noncomputable def f (x : ℝ) : ℝ := sqrt ((log x - 2) * (x - log x - 1))

def g (x : ℝ) : ℝ := x - log x - 1

theorem domain_of_f : {x : ℝ | (0 < x ∧ (log x - 2) ≥ 0 ∧ (x - log x - 1) ≥ 0) ∨ x = 1} = {x : ℝ | x ≥ real.exp 2 ∨ x = 1} :=
by
  sorry

end domain_of_f_l757_757974


namespace range_of_odd_multiples_of_5_sum_180_l757_757688

theorem range_of_odd_multiples_of_5_sum_180 : 
  ∃ (S : Set ℕ), (∀ n ∈ S, n % 5 = 0 ∧ n % 2 = 1) ∧ (S.sum = 180) ∧ (S.range = 50) := by
  -- Proof here
  sorry

end range_of_odd_multiples_of_5_sum_180_l757_757688


namespace num_five_ruble_coins_l757_757334

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757334


namespace midpoint_hyperbola_l757_757225

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757225


namespace find_x_l757_757423

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l757_757423


namespace hyperbola_midpoint_l757_757123

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757123


namespace johnny_guitar_practice_l757_757012

theorem johnny_guitar_practice :
  ∃ x : ℕ, (∃ d : ℕ, d = 20 ∧ ∀ n : ℕ, (n = x - d ∧ n = x / 2)) ∧ (x + 80 = 3 * x) :=
by
  sorry

end johnny_guitar_practice_l757_757012


namespace b_k_divisible_by_11_for_1_to_100_l757_757728

def b_n (n : ℕ) : ℕ :=
  let digits := (List.finRange n).map (λ i => (i + 1).toString.data).join
  in digits.foldl (λ acc c => acc * 10 + c.toNat - '0'.toNat) 0

def g (n : ℕ) : ℕ :=
  let digits := n.toString.data.map (λ c => c.toNat - '0'.toNat)
  in digits.enum.map (λ ⟨i, d⟩ => if i % 2 = 0 then d else -d).sum

def number_of_b_k_divisible_by_11 (k : ℕ) : ℕ :=
  (List.finRange k).count (λ i => g (b_n (i + 1)) % 11 = 0)

theorem b_k_divisible_by_11_for_1_to_100 : number_of_b_k_divisible_by_11 100 = 18 :=
sorry

end b_k_divisible_by_11_for_1_to_100_l757_757728


namespace hyperbola_midpoint_l757_757154

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757154


namespace study_time_difference_l757_757018

def hoursToMinutes (hours : ℝ) : ℝ := hours * 60

theorem study_time_difference :
  let kwame_study_time_hours : ℝ := 2.5;
  let connor_study_time_hours : ℝ := 1.5;
  let lexia_study_time_minutes : ℝ := 97;
  hoursToMinutes kwame_study_time_hours + hoursToMinutes connor_study_time_hours - lexia_study_time_minutes = 143 :=
by
  let kwame_study_time_minutes := hoursToMinutes 2.5
  let connor_study_time_minutes := hoursToMinutes 1.5
  let lexia_study_time_minutes := 97
  have h1 : kwame_study_time_minutes = 150 := by norm_num
  have h2 : connor_study_time_minutes = 90 := by norm_num
  have combined_study_time := h1 + h2
  have combined := combined_study_time - lexia_study_time_minutes
  show combined = 143 from by norm_num
  sorry

end study_time_difference_l757_757018


namespace midpoint_of_hyperbola_l757_757164

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757164


namespace line_passes_through_quadrants_l757_757605

theorem line_passes_through_quadrants (a b c : ℝ) (h1 : a * c < 0) (h2 : b * c < 0) : 
  (a * b > 0 ∧ -c / b > 0 ∧ -a / b < 0 → "first, second, and fourth quadrants") := 
sorry

end line_passes_through_quadrants_l757_757605


namespace B_contribution_is_9000_l757_757502

def A_investment : ℝ := 3500
def A_time : ℝ := 12
def B_time : ℝ := 7
def profit_ratio : ℝ := 2 / 3

theorem B_contribution_is_9000 :
  ∃ x : ℝ, (A_investment * A_time) / (x * B_time) = profit_ratio ∧ x = 9000 :=
by
  sorry

end B_contribution_is_9000_l757_757502


namespace integer_solution_a_l757_757551

theorem integer_solution_a (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by
  sorry

end integer_solution_a_l757_757551


namespace hyperbola_midpoint_exists_l757_757240

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757240


namespace find_number_l757_757442

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l757_757442


namespace steve_final_height_l757_757365

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l757_757365


namespace midpoint_of_hyperbola_l757_757163

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757163


namespace midpoint_of_hyperbola_l757_757055

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757055


namespace total_books_proof_l757_757563

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l757_757563


namespace isosceles_necessary_not_sufficient_l757_757888

-- Definitions for the given conditions
def is_isosceles_pyramid (P A B C D : Type*) : Prop :=
-- Definition for the assertion that the pyramid is isosceles.
sorry

def is_rectangle_base_with_perpendicular (P A B C D : Type*) : Prop :=
-- Definition for the base being a rectangle and the line connecting the center of the base to vertex P being perpendicular.
sorry

-- Main theorem
theorem isosceles_necessary_not_sufficient 
  {P A B C D : Type*} 
  (hA : is_isosceles_pyramid P A B C D) 
  (hB : is_rectangle_base_with_perpendicular P A B C D) :
  (hA → hB) ∧ ¬(hA ← hB) :=
sorry

end isosceles_necessary_not_sufficient_l757_757888


namespace midpoint_hyperbola_l757_757101

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757101


namespace average_rate_of_interest_l757_757898

theorem average_rate_of_interest (total_investment : ℝ) (rate1 rate2 average_rate : ℝ) (amount1 amount2 : ℝ)
  (H1 : total_investment = 6000)
  (H2 : rate1 = 0.03)
  (H3 : rate2 = 0.07)
  (H4 : average_rate = 0.042)
  (H5 : amount1 + amount2 = total_investment)
  (H6 : rate1 * amount1 = rate2 * amount2) :
  (rate1 * amount1 + rate2 * amount2) / total_investment = average_rate := 
sorry

end average_rate_of_interest_l757_757898


namespace max_volume_cube_max_volume_sphere_l757_757511

-- Definitions for the conditions in the problem
def cuboid (length width height : ℝ) : Prop :=
  length > 0 ∧ width > 0 ∧ height > 0

def cube (side : ℝ) : Prop :=
  side > 0

def sphere (radius : ℝ) : Prop :=
  radius > 0

-- Surface area of cuboid and cube
def surface_area_cuboid (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

def surface_area_cube (side : ℝ) : ℝ :=
  6 * side ^ 2

def surface_area_sphere (radius : ℝ) : ℝ :=
  4 * real.pi * radius ^ 2

-- Volume of cuboid, cube, and sphere
def volume_cuboid (length width height : ℝ) : ℝ :=
  length * width * height

def volume_cube (side : ℝ) : ℝ :=
  side ^ 3

def volume_sphere (radius : ℝ) : ℝ :=
  (4 / 3) * real.pi * radius ^ 3

-- Lean 4 theorems to prove
theorem max_volume_cube (S : ℝ) (hlw : ∃ (length width height : ℝ), cuboid length width height ∧ surface_area_cuboid length width height = S) :
  ∃ (side : ℝ), cube side ∧ surface_area_cube side = S ∧ (∀ length (width height: ℝ), cuboid length width height → surface_area_cuboid length width height = S → volume_cuboid length width height ≤ volume_cube side) :=
sorry

theorem max_volume_sphere (S : ℝ) (hlw : ∃ (length width height : ℝ), cuboid length width height ∧ surface_area_cuboid length width height = S) :
  ∃ (radius : ℝ), sphere radius ∧ surface_area_sphere radius = S ∧ (∀ length (width height: ℝ), cuboid length width height → surface_area_cuboid length width height = S → volume_cuboid length width height ≤ volume_sphere radius) :=
sorry

end max_volume_cube_max_volume_sphere_l757_757511


namespace hyperbola_midpoint_l757_757144

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757144


namespace selection_of_books_l757_757498

theorem selection_of_books (n k : ℕ) (hn : n = 7) (hk : k = 4) : nat.choose n k = 35 :=
by 
  -- This part will contain actual proof based on combinatorial computation
  sorry

end selection_of_books_l757_757498


namespace five_ruble_coins_count_l757_757311

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757311


namespace find_sum_invested_l757_757454

noncomputable def sum_invested (interest_difference: ℝ) (rate1: ℝ) (rate2: ℝ) (time: ℝ): ℝ := 
  interest_difference * 100 / (time * (rate1 - rate2))

theorem find_sum_invested :
  let interest_difference := 600
  let rate1 := 18 / 100
  let rate2 := 12 / 100
  let time := 2
  sum_invested interest_difference rate1 rate2 time = 5000 :=
by
  sorry

end find_sum_invested_l757_757454


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757282

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757282


namespace can_be_midpoint_of_AB_l757_757184

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757184


namespace find_larger_number_l757_757457

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 := 
sorry

end find_larger_number_l757_757457


namespace tan_neg_405_eq_neg_1_l757_757926

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l757_757926


namespace petya_five_ruble_coins_l757_757300

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757300


namespace midpoint_hyperbola_l757_757232

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757232


namespace solution_correct_l757_757925

noncomputable def z1 (w k : ℂ) : ℂ := w + k
noncomputable def z2 (w k : ℂ) : ℂ := complex.I * w + k
noncomputable def z3 (w k : ℂ) : ℂ := -w + k
noncomputable def z4 (w k : ℂ) : ℂ := -complex.I * w + k
noncomputable def complex_sum (w k : ℂ) : ℂ := z1 w k + z2 w k + z3 w k + z4 w k
noncomputable def complex_expression (w k : ℂ) : ℂ :=
  z1 w k * z2 w k +
  z1 w k * z3 w k +
  z1 w k * z4 w k +
  z2 w k * z3 w k +
  z2 w k * z4 w k +
  z3 w k * z4 w k

theorem solution_correct :
  (∀ (z1 z2 z3 z4 : ℂ), (z1 - z2).abs = 10 ∧ (z2 - z3).abs = 10 ∧
    (z3 - z4).abs = 10 ∧ (z4 - z1).abs = 10 ∧ (z1 - z3).abs = 10 ∧ (z2 - z4).abs = 10) →
  (∀ (z1 z2 z3 z4 : ℂ), abs (z1 + z2 + z3 + z4) = 20) →
  (abs (complex_expression w k) = 25) :=
by
  intros
  sorry

end solution_correct_l757_757925


namespace minute_hand_travel_distance_l757_757873

theorem minute_hand_travel_distance :
  ∀ (r : ℝ), r = 8 → (45 / 60) * (2 * Real.pi * r) = 12 * Real.pi :=
by
  intros r r_eq
  sorry

end minute_hand_travel_distance_l757_757873


namespace midpoint_of_hyperbola_l757_757168

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757168


namespace hyperbola_midpoint_l757_757115

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757115


namespace num_five_ruble_coins_l757_757316

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757316


namespace largest_two_digit_number_with_1_in_ones_place_l757_757823

theorem largest_two_digit_number_with_1_in_ones_place {a b c : ℕ} 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
  (h2 : {a, b, c} ⊆ {5, 6, 9})
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  10 * (max a (max b c)) + 1 = 91 := 
sorry

end largest_two_digit_number_with_1_in_ones_place_l757_757823


namespace cost_of_refrigerator_l757_757346

theorem cost_of_refrigerator (R : ℝ) : 
  (∀ R : ℝ, 0.98 * R + 8800 = R + 8500 → R = 15000) :=
begin
  sorry
end

end cost_of_refrigerator_l757_757346


namespace gain_per_year_is_correct_l757_757882

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_per_year_is_correct :
  let borrowed_amount := 7000
  let borrowed_rate := 0.04
  let borrowed_time := 2
  let borrowed_compound_freq := 1 -- annually
  
  let lent_amount := 7000
  let lent_rate := 0.06
  let lent_time := 2
  let lent_compound_freq := 2 -- semi-annually
  
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_freq borrowed_time
  let amount_received := compound_interest lent_amount lent_rate lent_compound_freq lent_time
  let total_gain := amount_received - amount_owed
  let gain_per_year := total_gain / lent_time
  
  gain_per_year = 153.65 :=
by
  sorry

end gain_per_year_is_correct_l757_757882


namespace pi_approximation_l757_757761

theorem pi_approximation (n m : ℕ) (x y : Fin n → ℝ) (h1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1)
  (h2 : ∀ i, 0 ≤ y i ∧ y i ≤ 1) (h3 : m = ∑ i, if (x i)^2 + (y i)^2 < 1 then 1 else 0) :
  π = 4 * m / n := 
sorry

end pi_approximation_l757_757761


namespace sum_lambda_leq_n_l757_757758

theorem sum_lambda_leq_n (n : ℕ) (λ : ℕ → ℝ) :
  (∑ k in Finset.range n, λ (k + 1)) ≤ n :=
sorry

end sum_lambda_leq_n_l757_757758


namespace ratio_shaded_area_proof_l757_757850

noncomputable def ratio_shaded_area_to_circle (r : ℝ) : ℝ :=
  let r_squared := r * r
  let largest_semi_circle_area := (9 * (π * r_squared)) / 4
  let middle_semi_circle_area := (π * r_squared) / 2
  let smallest_semi_circle_area := (π * r_squared) / 8
  let shaded_area := largest_semi_circle_area - middle_semi_circle_area - smallest_semi_circle_area
  let cd := r * Real.sqrt 2
  let circle_area := π * (cd * cd)
  shaded_area / circle_area

theorem ratio_shaded_area_proof (r : ℝ) : ratio_shaded_area_to_circle r = 9 / 16 := by
  sorry

end ratio_shaded_area_proof_l757_757850


namespace midpoint_on_hyperbola_l757_757081

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757081


namespace midpoint_of_hyperbola_l757_757169

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757169


namespace initial_house_cats_l757_757493

/-- Conditions based on the problem statement -/
def number_of_siamese_cats : Nat := 12
def cats_sold : Nat := 20
def cats_left : Nat := 12

/-- Prove the number of house cats initially -/
theorem initial_house_cats (H : Nat) (total_cats_before_sale : Nat = number_of_siamese_cats + H) :
  (cats_left + cats_sold = total_cats_before_sale) → 
  (number_of_siamese_cats + H = 32) → 
  H = 20 := by
sorry

end initial_house_cats_l757_757493


namespace sum_real_parts_l757_757578

theorem sum_real_parts (x : ℝ) : 
  let z := (1 + complex.i * x) in 
  let sum_of_real_parts := ∑ i in (finset.range 51).filter (λ k, even k), 
    (z ^ 50).re :=
  sum_of_real_parts = 0 := sorry

end sum_real_parts_l757_757578


namespace odd_square_free_count_l757_757663

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757663


namespace square_free_odd_integers_count_l757_757655

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757655


namespace collinear_M_N_P_l757_757258

noncomputable theory
open_locale classical

variables (A B C I D E P M N : Type*)

-- Definitions for the problem setup
def is_triangle (A B C : Type*) : Prop := true  -- Placeholder; actual definition needed
def incenter (A B C : Type*) (I : Type*) : Prop := true  -- Placeholder
def incircle_touches (A B C : Type*) (D E : Type*) : Prop := true  -- Placeholder
def midpoint (A B C M : Type*) : Prop := true  -- Placeholder
def intersection (AI DE P : Type*) : Prop := true  -- Placeholder

-- Conditions
axiom triangle_ABC : is_triangle A B C
axiom incenter_I : incenter A B C I
axiom incircle_touch_D_E : incircle_touches A B C D ∧ incircle_touches A C B E
axiom midpoint_M : midpoint B C M
axiom midpoint_N : midpoint A B N
axiom intersection_P : intersection A I D E P

-- The theorem to prove
theorem collinear_M_N_P : collinear M N P := 
sorry

end collinear_M_N_P_l757_757258


namespace midpoint_on_hyperbola_l757_757051

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757051


namespace quadratic_rational_roots_l757_757272

theorem quadratic_rational_roots (a b : ℚ) 
  (h : ∃ x : ℝ, x^2 + (a:ℝ) * x + (b:ℝ) = 0 ∧ x = 1 + real.sqrt 3) : 
  a = -2 ∧ b = -2 := 
sorry

end quadratic_rational_roots_l757_757272


namespace midpoint_on_hyperbola_l757_757048

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757048


namespace find_angle4_l757_757686

-- Define angles as degrees
def angle := ℝ

-- Define the conditions
variables (angle1 angle2 angle3 angle4 angle5 : angle)

-- Given conditions
def condition1 : Prop := angle1 + angle2 = 180
def condition2 : Prop := angle4 = angle5
def condition3 : Prop := angle1 + angle3 + angle5 = 180

-- Given values
def given_values : Prop := (angle1 = 50) ∧ (angle3 = 60)

-- The statement to prove
theorem find_angle4 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : given_values) : angle4 = 70 :=
sorry

end find_angle4_l757_757686


namespace total_crayons_l757_757757

noncomputable def original_crayons : ℝ := 479.0
noncomputable def additional_crayons : ℝ := 134.0

theorem total_crayons : original_crayons + additional_crayons = 613.0 := by
  sorry

end total_crayons_l757_757757


namespace midpoint_on_hyperbola_l757_757073

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757073


namespace company_max_profit_l757_757479

theorem company_max_profit :
  (∀ x, (0 ≤ x ∧ x ≤ 400) → -0.5 * x^2 + 300 * x - 20000 ≤ f(300)) ∧
  (∀ x, (x > 400) → 60000 - 100 * x ≤ f(300)) ∧
  (f(300) = 25000) :=
by
  let f : ℝ → ℝ :=
    λ x, if (0 ≤ x ∧ x ≤ 400) then -0.5 * x^2 + 300 * x - 20000
                                else 60000 - 100 * x
  sorry

end company_max_profit_l757_757479


namespace number_of_square_free_odds_l757_757647

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757647


namespace slope_of_vertical_line_l757_757803

theorem slope_of_vertical_line :
  ∀ (x : ℝ), x = real.sqrt 3 → (∃ θ : ℝ, θ = 90) :=
by
  intro x hx
  use 90
  sorry

end slope_of_vertical_line_l757_757803


namespace find_number_l757_757430

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l757_757430


namespace midpoint_of_hyperbola_l757_757171

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757171


namespace row_lengths_count_90_l757_757872

def num_divisors_in_range (n : ℕ) (lower upper : ℕ) : ℕ :=
  (finset.filter (λ x, lower ≤ x ∧ x ≤ upper) (finset.divisors n)).card

theorem row_lengths_count_90 :
  num_divisors_in_range 90 6 15 = 4 :=
sorry

end row_lengths_count_90_l757_757872


namespace diff_highest_lowest_avg_speed_l757_757822

-- Assume participants and their distances
inductive Participant
| CyclistA
| CyclistB
| CarX
| CarY
| CarZ

open Participant

def distance_covered : Participant → ℝ
| CyclistA => 100
| CyclistB => 55
| CarX => 160
| CarY => 120
| CarZ => 200

-- Time taken for the race
def time_taken : ℝ := 8

-- Function to compute average speed
def average_speed (p : Participant) : ℝ := distance_covered p / time_taken

-- List of all participants
def participants : List Participant := [CyclistA, CyclistB, CarX, CarY, CarZ]

-- Function to find the highest and lowest average speeds
def highest_average_speed : ℝ := participants.map average_speed |> List.maximum?.getD 0
def lowest_average_speed : ℝ := participants.map average_speed |> List.minimum?.getD 0

-- Main theorem
theorem diff_highest_lowest_avg_speed :
  highest_average_speed - lowest_average_speed = 18.125 :=
by
  sorry

end diff_highest_lowest_avg_speed_l757_757822


namespace sum_b_l757_757559

noncomputable def b' (p : ℕ) : ℕ :=
choose (classical.some_spec (exists_unique.intro (λ k : ℕ, |k - real.sqrt p| < 1 / 3) sorry)) -- Use exists_unique.intro to express uniqueness

theorem sum_b'_1_to_100 : 
  ∑ p in (finset.range 100).image nat.succ, b' p = 513 + 1 / 3 := -- using image nat.succ to range over 1 to 100 
sorry -- Proof omitted

end sum_b_l757_757559


namespace hyperbola_midpoint_l757_757148

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757148


namespace sum_of_coefficients_l757_757376

theorem sum_of_coefficients : 
  ∃ (a b c d e f g h j k : ℤ), 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) → 
    (a + b + c + d + e + f + g + h + j + k = 92) :=
sorry

end sum_of_coefficients_l757_757376


namespace odd_square_free_count_l757_757665

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757665


namespace midpoint_on_hyperbola_l757_757133

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757133


namespace total_age_10_years_from_now_is_75_l757_757814

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end total_age_10_years_from_now_is_75_l757_757814


namespace log_equation_solution_l757_757766

open Real

theorem log_equation_solution :
  ∀ (x : ℝ), (x > 17 / 7) →
  log 5 (3 * x - 4) * log 5 (7 * x - 16) * (3 - log 5 (21 * x ^ 2 - 76 * x + 64)) = 1 ↔ x = 3 := 
by
  -- Proof omitted
  intro x hx
  sorry

end log_equation_solution_l757_757766


namespace circle_passing_through_O_to_line_circle_not_passing_through_O_to_circle_l757_757342

section inversion_problems

variables {O : Point}
-- Define inversion function
def inversion (P : Point) : Point := sorry

-- Define a (geometric) Circle structure in the plane
structure Circle :=
(center : Point)
(radius : ℝ)
(passes_through : Point → Prop)

-- Conditions 
def is_circle_passing_through (S : Circle) (O : Point) : Prop :=
∃ A B, S.passes_through A ∧ S.passes_through B ∧ A ≠ O ∧ B ≠ O

def is_circle_not_passing_through (S : Circle) (O : Point) : Prop :=
∀ P, S.passes_through P → P ≠ O

-- Prove that under inversion with center O
theorem circle_passing_through_O_to_line (S : Circle) (h : is_circle_passing_through S O) :
  (inversion S = Line) := sorry

-- Prove that under inversion with center O
theorem circle_not_passing_through_O_to_circle (S : Circle) (h : is_circle_not_passing_through S O) :
  (∃ (T : Circle), T = inversion S) := sorry

end inversion_problems

end circle_passing_through_O_to_line_circle_not_passing_through_O_to_circle_l757_757342


namespace midpoint_on_hyperbola_l757_757074

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757074


namespace tenth_term_arithmetic_seq_l757_757415

theorem tenth_term_arithmetic_seq : 
  ∀ (first_term common_diff : ℤ) (n : ℕ), 
    first_term = 10 → common_diff = -2 → n = 10 → 
    (first_term + (n - 1) * common_diff) = -8 :=
by
  sorry

end tenth_term_arithmetic_seq_l757_757415


namespace f_eq_f_inv_implies_x_eq_0_l757_757949

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
noncomputable def f_inv (x : ℝ) : ℝ := (-1 + Real.sqrt (3 * x + 4)) / 3

theorem f_eq_f_inv_implies_x_eq_0 (x : ℝ) : f x = f_inv x → x = 0 :=
by
  sorry

end f_eq_f_inv_implies_x_eq_0_l757_757949


namespace fraction_shaded_is_correct_l757_757340

-- Define the regular octagon with center O and vertices A, B, C, D, E, F, G, H
variables {A B C D E F G H O Y : Type}
variable [RegularOctagon A B C D E F G H O] -- assuming RegularOctagon is defined elsewhere

-- Define point Y on side AB such that AY:YB = 1:3
variable (AY_YB : ratio (SegmentLength Y A) (SegmentLength Y B) = 1 / 3)

-- Define the shaded triangles
variable (ShadedTriangles : Set (Triangle O))

-- Define the specific shaded triangles as given
def triangles_shaded : Set (Triangle O) := {BCO, CDO, AYO}

-- Define the total area of the octagon
variable (OctagonArea : Area (Polygon (vertices [A, B, C, D, E, F, G, H])))

-- Define the fraction of the area that is shaded
def fraction_shaded := Area (Union ShadedTriangles) / OctagonArea

-- The main theorem to prove
theorem fraction_shaded_is_correct :
  fraction_shaded = 9 / 32 :=
by
  sorry

end fraction_shaded_is_correct_l757_757340


namespace positive_difference_l757_757396

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l757_757396


namespace inequality_proof_l757_757732

variable {n : ℕ} 
variable {x : Fin n → ℝ}

theorem inequality_proof (h1 : 3 ≤ n) (h2 : ∀ i : Fin n, 0 < x i) :
  1 < (∑ i : Fin n, x i / (x i + x ((i + 1) % n))) ∧ 
      (∑ i : Fin n, x i / (x i + x ((i + 1) % n)) < n - 1) := 
sorry


end inequality_proof_l757_757732


namespace ming_dynasty_wine_problem_l757_757694

theorem ming_dynasty_wine_problem :
  ∃ x y : ℝ, x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by {
  -- Define the existence of variables x and y satisfying the conditions
  existsi (x : ℝ),
  existsi (y : ℝ),
  -- Conditions are given as premises to be satisfied
  split,
  -- First equation: x + y = 19
  exact x + y = 19,
  -- Second equation: 3x + (1/3)y = 33
  exact 3 * x + (1 / 3) * y = 33,
  -- Add placeholder to indicate where the actual proof would go
  sorry
}

end ming_dynasty_wine_problem_l757_757694


namespace complement_intersection_l757_757634

def P (x : ℝ) : Prop := x^2 - 2 * x ≥ 0
def Q (x : ℝ) : Prop := 1 < x ∧ x ≤ 2
def complement_R (P : ℝ → Prop) : ℝ → Prop := λ x, ¬ P x

theorem complement_intersection :
  (λ x, complement_R P x ∧ Q x) = (λ x, 1 < x ∧ x < 2) :=
by
  sorry

end complement_intersection_l757_757634


namespace thread_length_l757_757348

theorem thread_length (x : ℝ) (h : x + (3/4) * x = 21) : x = 12 :=
  sorry

end thread_length_l757_757348


namespace isabela_spent_2800_l757_757004

/-- Given:
1. Isabela bought twice as many cucumbers as pencils.
2. Both cucumbers and pencils cost $20 each.
3. Isabela got a 20% discount on the pencils.
4. She bought 100 cucumbers.
Prove that the total amount Isabela spent is $2800. -/
theorem isabela_spent_2800 :
  ∀ (pencils cucumbers : ℕ) (pencil_cost cucumber_cost : ℤ) (discount rate: ℚ)
    (total_cost pencils_cost cucumbers_cost discount_amount : ℤ),
  cucumbers = 100 →
  pencils * 2 = cucumbers →
  pencil_cost = 20 →
  cucumber_cost = 20 →
  rate = 20 / 100 →
  pencils_cost = pencils * pencil_cost →
  discount_amount = pencils_cost * rate →
  total_cost = pencils_cost - discount_amount + cucumbers * cucumber_cost →
  total_cost = 2800 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end isabela_spent_2800_l757_757004


namespace regular_price_of_Pony_jeans_l757_757568

theorem regular_price_of_Pony_jeans 
(Fox_price : ℝ) 
(Pony_price : ℝ) 
(savings : ℝ) 
(Fox_discount_rate : ℝ) 
(Pony_discount_rate : ℝ)
(h1 : Fox_price = 15)
(h2 : savings = 8.91)
(h3 : Fox_discount_rate + Pony_discount_rate = 0.22)
(h4 : Pony_discount_rate = 0.10999999999999996) : Pony_price = 18 := 
sorry

end regular_price_of_Pony_jeans_l757_757568


namespace find_common_difference_l757_757591

noncomputable def arithmetic_sequence_variance (d : ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := arithmetic_sequence_variance d n + d

theorem find_common_difference (d : ℝ) (a : ℕ → ℝ)
  (h1 : a = arithmetic_sequence_variance d)
  (h2 : ∀ n, a n ≤ a (n + 1))
  (h3 : variance (a 2, a 3, a 4, a 5, a 6) = 3) :
  d = real.sqrt 6 / 2 :=
sorry

end find_common_difference_l757_757591


namespace am_gm_minimum_l757_757735

noncomputable def minimum_value (p q r s t u v w : ℝ) : ℝ := (pt)^2 + (qu)^2 + (rv)^2 + (sw)^2

theorem am_gm_minimum (p q r s t u v w : ℝ)
  (h1 : p * q * r * s = 16)
  (h2 : t * u * v * w = 25) :
  (minimum_value p q r s t u v w) ≥ 80 :=
sorry

end am_gm_minimum_l757_757735


namespace age_of_15th_student_l757_757461

theorem age_of_15th_student : 
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  total_age_all_students - (total_age_first_group + total_age_second_group) = 16 :=
by
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  sorry

end age_of_15th_student_l757_757461


namespace midpoint_of_hyperbola_l757_757162

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757162


namespace integer_roots_of_polynomial_l757_757969

-- Defining the polynomial
def f (x : ℤ) : ℤ := x^3 - 4 * x^2 - 14 * x + 24

-- Defining what it means to be an integer root of the polynomial
def is_integer_root (x : ℤ) : Prop := f(x) = 0

-- Stating the theorem with the proven integer roots
theorem integer_roots_of_polynomial : {x : ℤ | is_integer_root x} = {-1, 3, 4} := by
  sorry

end integer_roots_of_polynomial_l757_757969


namespace floor_of_fraction_l757_757528

theorem floor_of_fraction : 
  (nat.floor ((2010! + 2008!) / (2011! + 2009!))) = 0 :=
by 
  sorry

end floor_of_fraction_l757_757528


namespace subsets_count_l757_757724

def M : Set ℤ := {x | x ^ 2 - 3 * x - 4 = 0}
def N : Set ℤ := {x | x ^ 2 - 16 = 0}
def union_set : Set ℤ := M ∪ N
def num_subsets (s : Set ℤ) : ℕ := 2 ^ s.to_finset.card

theorem subsets_count : num_subsets union_set = 8 := by
  sorry

end subsets_count_l757_757724


namespace midpoint_on_hyperbola_l757_757135

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757135


namespace can_be_midpoint_of_AB_l757_757176

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757176


namespace parabola_focus_l757_757986

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757986


namespace midpoint_product_is_ten_l757_757834

def product_of_midpoint_coordinates (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  let mz := (z1 + z2) / 2
  mx * my * mz

theorem midpoint_product_is_ten : product_of_midpoint_coordinates 3 -2 4 7 6 -2 = 10 :=
by
  sorry

end midpoint_product_is_ten_l757_757834


namespace part_a_total_time_part_b_average_time_part_c_probability_l757_757291

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l757_757291


namespace range_of_m_for_p_range_of_m_for_p_and_not_q_l757_757595

-- Definitions of the propositions p and q
def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * m * x - 3 * m > 0

def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4 * m * x + 1 < 0

-- The Lean theorem to prove the ranges of m based on the conditions
theorem range_of_m_for_p :
  ∀ m : ℝ, prop_p m → -3 < m ∧ m < 0 := sorry

theorem range_of_m_for_p_and_not_q :
  ∀ m : ℝ, prop_p m → ¬ prop_q m → -1/2 ≤ m ∧ m < 0 := sorry

end range_of_m_for_p_range_of_m_for_p_and_not_q_l757_757595


namespace symmetric_circle_eq_l757_757675

-- Definitions for the given problem
def circle_eq (x y : ℝ) : Type := x^2 + y^2 + 2*x = 0
def line_eq (x y : ℝ) : Type := x + y - 1 = 0

-- The given circle's equation
def given_circle_eq (x y : ℝ) : Prop := circle_eq x y

-- The line of symmetry
def line_of_symmetry_eq (x y : ℝ) : Prop := line_eq x y

-- Prove that the equation of the circle C is (x - 1)^2 + (y - 2)^2 = 1
theorem symmetric_circle_eq : 
  (∀ x y : ℝ, given_circle_eq x y → line_of_symmetry_eq x y → 
  (x - 1)^2 + (y - 2)^2 = 1) := 
sorry

end symmetric_circle_eq_l757_757675


namespace locus_of_midpoints_is_segment_O₁_O₂_l757_757250

-- Define the centers of the circles
variables {O₁ O₂ T S : Point}

-- Define the circles and their properties
axiom circles_tangent_at_T : tangent_circles O₁ O₂ T 1
axiom point_S_opposite_T : diametrically_opposite_point O₂ S T

-- Define points A and B and their respective motions
variables (A B : Point)
axiom point_A_moves_clockwise : moves_clockwise_on_circle A O₁ T 1
axiom point_B_moves_counterclockwise : moves_counterclockwise_on_circle B O₂ S 1
axiom points_move_same_speed : same_speed A B

-- Define the segment joining the centers
def segment_O₁_O₂ : Segment := ⟨O₁, O₂⟩

-- Proof statement
theorem locus_of_midpoints_is_segment_O₁_O₂ :
  locus (midpoint A B) = segment_O₁_O₂ := by
  sorry

end locus_of_midpoints_is_segment_O₁_O₂_l757_757250


namespace compound_interest_l757_757905

noncomputable def final_amount (P : ℕ) (r : ℚ) (t : ℕ) :=
  P * ((1 : ℚ) + r) ^ t

theorem compound_interest : 
  final_amount 20000 0.20 10 = 123834.73 := 
by 
  sorry

end compound_interest_l757_757905


namespace binary_to_decimal_conversion_l757_757533

theorem binary_to_decimal_conversion :
  let b := [1,0,1,1,0,0,1,1] in -- Representing 10 110 011_(2)
  (b[0] * 2^7 + b[1] * 2^6 + b[2] * 2^5 + b[3] * 2^4 + b[4] * 2^3 + b[5] * 2^2 + 
  b[6] * 2^1 + b[7] * 2^0) = 179 :=
by
  sorry

end binary_to_decimal_conversion_l757_757533


namespace arithmetic_sequence_sum_l757_757840

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l757_757840


namespace total_age_of_siblings_in_10_years_l757_757816

theorem total_age_of_siblings_in_10_years (age_eldest : ℕ) (gap : ℕ) (h1 : age_eldest = 20) (h2 : gap = 5) :
  let age_second := age_eldest - gap,
      age_youngest := age_second - gap in
  age_eldest + 10 + (age_second + 10) + (age_youngest + 10) = 75 :=
by
  sorry

end total_age_of_siblings_in_10_years_l757_757816


namespace midpoint_hyperbola_l757_757228

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757228


namespace john_reps_per_set_calculation_l757_757008

-- Definitions corresponding to the conditions
def weight_per_rep : ℕ := 15
def number_of_sets : ℕ := 3
def total_weight_moved : ℕ := 450

-- Definition corresponding to the answer
def reps_per_set : ℕ := 10

-- Proof statement
theorem john_reps_per_set_calculation :
  ((total_weight_moved / weight_per_rep) / number_of_sets) = reps_per_set :=
by
  calc ((total_weight_moved / weight_per_rep) / number_of_sets)
        = ((450 / 15) / 3) : by rfl
    ... = (30 / 3) : by rfl
    ... = 10 : by rfl

end john_reps_per_set_calculation_l757_757008


namespace exists_perpendicular_intersection_l757_757508

noncomputable def line (p1 p2 : EuclideanSpace ℝ (fin 3)) : Set (EuclideanSpace ℝ (fin 3)) :=
  {p | ∃ (t : ℝ), p = p1 + t • (p2 - p1)}

variables {l1 l2 : Set (EuclideanSpace ℝ (fin 3))}

def non_intersecting (l1 l2 : Set (EuclideanSpace ℝ (fin 3))) : Prop :=
  ∀ (p : EuclideanSpace ℝ (fin 3)), p ∈ l1 → p ∉ l2

def perpendicularly_intersect (l1 l2 l3 : Set (EuclideanSpace ℝ (fin 3))) : Prop :=
  ∃ (p1 p2 : EuclideanSpace ℝ (fin 3)), p1 ∈ l1 ∧ p2 ∈ l2 ∧ p1 ≠ p2 ∧ 
    ∀ (v1 v2 : EuclideanSpace ℝ (fin 3)), v1 ∈ l1 → v2 ∈ l2 → v1 - p1 ⊥ v2 - p2

theorem exists_perpendicular_intersection (l1 l2 : Set (EuclideanSpace ℝ (fin 3)))
  (h1 : non_intersecting l1 l2) :
  ∃ l3 : Set (EuclideanSpace ℝ (fin 3)), perpendicularly_intersect l1 l2 l3 := 
  sorry

end exists_perpendicular_intersection_l757_757508


namespace fair_die_proba_l757_757817
noncomputable def probability_of_six : ℚ := 1 / 6

theorem fair_die_proba : 
  (1 / 6 : ℚ) = probability_of_six :=
by
  sorry

end fair_die_proba_l757_757817


namespace tangent_line_equation_at_point_l757_757977

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem tangent_line_equation_at_point (x y : ℝ) (h : y = curve x) 
    (hx : 2) (hy : 5) (hpt : y = 5 ∧ x = 2) : 7 * x - y - 9 = 0 :=
by
  sorry

end tangent_line_equation_at_point_l757_757977


namespace midpoint_on_hyperbola_l757_757142

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757142


namespace tan_minus_405_eq_neg1_l757_757934

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l757_757934


namespace square_free_odd_integers_count_l757_757657

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757657


namespace find_door_height_l757_757372

theorem find_door_height :
  ∃ (h : ℝ), 
  let l := 25
  let w := 15
  let H := 12
  let A := 80 * H
  let W := 960 - (6 * h + 36)
  let cost := 4 * W
  cost = 3624 ∧ h = 3 := sorry

end find_door_height_l757_757372


namespace ellipse_center_and_major_axis_length_l757_757904

theorem ellipse_center_and_major_axis_length :
  let f1 : ℝ × ℝ := (3, -2)
  let f2 : ℝ × ℝ := (11, 6)
  let c : ℝ × ℝ := ((3 + 11) / 2, (-2 + 6) / 2)
  let major_axis_length := 20
  (c = (7, 2)) ∧ (major_axis_length = 20) :=
by
  let f1 : ℝ × ℝ := (3, -2)
  let f2 : ℝ × ℝ := (11, 6)
  let c : ℝ × ℝ := ((3 + 11) / 2, (-2 + 6) / 2)
  let major_axis_length := 20
  have center_eq : c = (7, 2) :=
    by
      calc c = ((3 + 11) / 2, (-2 + 6) / 2) : by rfl
         ... = (7, 2)                     : by rfl
  have axis_length_eq : major_axis_length = 20 :=
    by rfl
  exact ⟨center_eq, axis_length_eq⟩

end ellipse_center_and_major_axis_length_l757_757904


namespace imaginary_part_of_division_l757_757743

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Given complex number
def z : ℂ := (3 + 2 * i) / i

-- Objective: Prove that the imaginary part of z is -3
theorem imaginary_part_of_division :
  complex.im z = -3 :=
by
  sorry

end imaginary_part_of_division_l757_757743


namespace parabola_focus_l757_757980

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757980


namespace max_cities_visited_l757_757751

theorem max_cities_visited (n k : ℕ) : ∃ t, t = n - k :=
by
  sorry

end max_cities_visited_l757_757751


namespace midpoint_of_hyperbola_l757_757068

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757068


namespace midpoint_of_hyperbola_l757_757161

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757161


namespace distance_scenario_a_distance_scenario_b_distance_scenario_c_distance_scenario_d_l757_757373

-- Definitions: Initial distance, speeds, and time
def initial_distance : Real := 20 
def athos_speed : Real := 4
def aramis_speed : Real := 5
def time : Real := 1

-- Hypotheses including scenarios translated from problem
theorem distance_scenario_a : 
  x = initial_distance − (athos_speed + aramis_speed) * time → 
  x = 11 := by
  sorry

theorem distance_scenario_b : 
  y = initial_distance + (athos_speed + aramis_speed) * time → 
  y = 29 := by
  sorry

theorem distance_scenario_c : 
  z = initial_distance − (aramis_speed − athos_speed) * time → 
  z = 19 := by
  sorry

theorem distance_scenario_d : 
  w = initial_distance + (aramis_speed − athos_speed) * time → 
  w = 21 := by
  sorry

end distance_scenario_a_distance_scenario_b_distance_scenario_c_distance_scenario_d_l757_757373


namespace louisa_second_day_distance_l757_757271

-- Definitions based on conditions
def time_on_first_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_on_second_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

def condition (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) : Prop := 
  time_on_first_day distance_first_day speed + time_difference = time_on_second_day x speed

-- The proof statement
theorem louisa_second_day_distance (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) :
  distance_first_day = 240 → 
  speed = 60 → 
  time_difference = 3 → 
  condition distance_first_day speed time_difference x → 
  x = 420 :=
by
  intros h1 h2 h3 h4
  sorry

end louisa_second_day_distance_l757_757271


namespace equivalence_of_statements_l757_757451

theorem equivalence_of_statements (S X Y : Prop) : 
  (S → (¬ X ∧ ¬ Y)) ↔ ((X ∨ Y) → ¬ S) :=
by sorry

end equivalence_of_statements_l757_757451


namespace correct_midpoint_l757_757211

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757211


namespace can_be_midpoint_of_AB_l757_757174

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757174


namespace arrangement_without_A_at_head_tail_girls_next_to_each_other_ABC_not_next_to_each_other_A_not_head_B_not_tail_l757_757809

-- Definitions based on problem conditions
def boys : ℕ := 4
def girls : ℕ := 2

-- Question (I): Boy A does not stand at the head or the tail of the row
theorem arrangement_without_A_at_head_tail : 
  ∃ n : ℕ, n = 480 ∧ 
  ∀ A B C D G1 G2 : ℕ, A ≠ head ∧ A ≠ tail → number_of_arrangements(boys, girls) = 480 := 
sorry

-- Question (II): The two girls must stand next to each other
theorem girls_next_to_each_other : 
  ∃ n : ℕ, n = 240 ∧ 
  ∀ A B C D G1 G2 : ℕ, (G1 = G2 + 1 ∨ G2 = G1 + 1) → number_of_arrangements(boys, girls) = 240 := 
sorry

-- Question (III): Students A, B, and C are not next to each other
theorem ABC_not_next_to_each_other : 
  ∃ n : ℕ, n = 144 ∧ 
  ∀ A B C D G1 G2 : ℕ, (A ≠ B + 1 ∧ A ≠ C - 1 ∧ B ≠ C + 1) → number_of_arrangements(boys, girls) = 144 := 
sorry

-- Question (IV): A does not stand at the head, and B does not stand at the tail
theorem A_not_head_B_not_tail : 
  ∃ n : ℕ, n = 504 ∧ 
  ∀ A B C D G1 G2 : ℕ, (A ≠ head ∧ B ≠ tail) → number_of_arrangements(boys, girls) = 504 := 
sorry

end arrangement_without_A_at_head_tail_girls_next_to_each_other_ABC_not_next_to_each_other_A_not_head_B_not_tail_l757_757809


namespace midpoint_of_hyperbola_l757_757063

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757063


namespace dog_count_l757_757683

theorem dog_count 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (long_furred_brown : ℕ) 
  (total : ℕ) 
  (h1 : long_furred = 29) 
  (h2 : brown = 17) 
  (h3 : neither = 8) 
  (h4 : long_furred_brown = 9)
  (h5 : total = long_furred + brown - long_furred_brown + neither) : 
  total = 45 :=
by 
  sorry

end dog_count_l757_757683


namespace neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l757_757598

def p (x : ℝ) : Prop := (x^2 - x - 2) ≤ 0
def q (x m : ℝ) : Prop := (x^2 - x - m^2 - m) ≤ 0

theorem neg_p_range_of_x (x : ℝ) : ¬ p x → x > 2 ∨ x < -1 :=
by
-- proof steps here
sorry

theorem neg_q_sufficient_not_necessary_for_neg_p (m : ℝ) : 
  (∀ x, ¬ q x m → ¬ p x) ∧ (∃ x, p x → ¬ q x m) → m > 1 ∨ m < -2 :=
by
-- proof steps here
sorry

end neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l757_757598


namespace petya_five_ruble_coins_l757_757299

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757299


namespace edges_same_color_l757_757957

-- Define the vertices of the pentagons
inductive Vertex
| A : Fin 5 → Vertex
| B : Fin 5 → Vertex

open Vertex

-- Define colors
inductive Color
| Red : Color
| Blue : Color

-- Define the edge color function
def edgeColor : Vertex → Vertex → Color

-- Define the condition that no triangle is monochromatic
def noMonochromaticTriangle : Prop :=
  ∀ v1 v2 v3 : Vertex,
  v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
  ¬ (edgeColor v1 v2 = edgeColor v2 v3 ∧ edgeColor v2 v3 = edgeColor v1 v3)

-- Define the goal
def sameColorHorizontalEdges : Prop :=
  ∀ i j : Fin 5, edgeColor (A i) (A j) = edgeColor (B i) (B j)

theorem edges_same_color (h : noMonochromaticTriangle) : sameColorHorizontalEdges :=
  by
  sorry -- The proof itself is ommitted.

end edges_same_color_l757_757957


namespace maximum_value_of_function_l757_757953

noncomputable def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem maximum_value_of_function :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = 25 / 4 :=
by 
  sorry

end maximum_value_of_function_l757_757953


namespace can_be_midpoint_of_AB_l757_757182

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757182


namespace focus_of_parabola_l757_757995

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757995


namespace arithmetic_sequence_common_difference_l757_757700

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) 
    (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
    (h2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) : 
    ∃ d, d = 10 := 
by 
  sorry

end arithmetic_sequence_common_difference_l757_757700


namespace find_current_l757_757449

-- Define the variables and constants used in the problem
variables (Q I R t : ℝ)

-- State the conditions given in the problem
def given_conditions :=
  Q = 30 ∧ R = 5 ∧ t = 1

-- State the formula given for heat generation
def heat_equation :=
  Q = I^2 * R * t

-- State the main theorem to prove
theorem find_current (h_given : given_conditions) (h_heat : heat_equation) : I = real.sqrt 6 :=
  sorry

end find_current_l757_757449


namespace scramble_language_words_count_l757_757795

theorem scramble_language_words_count :
  let total_words (n : ℕ) := 25 ^ n
  let words_without_B (n : ℕ) := 24 ^ n
  let words_with_B (n : ℕ) := total_words n - words_without_B n
  words_with_B 1 + words_with_B 2 + words_with_B 3 + words_with_B 4 + words_with_B 5 = 1863701 :=
by
  sorry

end scramble_language_words_count_l757_757795


namespace complement_of_A_is_correct_l757_757262

-- Define the universal set U and the set A.
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U.
def A_complement : Set ℕ := {x ∈ U | x ∉ A}

-- The theorem statement that the complement of A in U is {2, 4}.
theorem complement_of_A_is_correct : A_complement = {2, 4} :=
sorry

end complement_of_A_is_correct_l757_757262


namespace midpoint_on_hyperbola_l757_757039

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757039


namespace base_subtraction_l757_757919

def convertBase (num : List ℕ) (base : ℕ) : ℕ :=
  num.foldr (λ (dig : ℕ) (acc : ℕ) => dig + acc * base) 0

theorem base_subtraction
  (h₁ : convertBase [5, 2, 1, 0, 3] 8 = 21571)
  (h₂ : convertBase [1, 4, 5, 2] 9 = 1100) :
  (21571 - 1100) = 20471 := 
by
  simp [h₁, h₂]
  done

end base_subtraction_l757_757919


namespace find_natural_numbers_l757_757965

theorem find_natural_numbers (n : ℕ) (h₁ : 2 ≤ n)
  (h₂ : ∃ (a : ℕ → ℝ), 
    {d | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ n ∧ d = |a i - a j|} = {1..finset.range (n*(n-1)/2).card}) :
  n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end find_natural_numbers_l757_757965


namespace number_of_five_ruble_coins_l757_757328

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757328


namespace part_a_part_b_part_c_l757_757273

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l757_757273


namespace geometric_series_sum_l757_757837

theorem geometric_series_sum : 
  let a := 1 
  let r := 2 
  let n := 11 
  let S_n := (a * (1 - r^n)) / (1 - r)
  S_n = 2047 := by
  -- The proof steps would normally go here.
  sorry

end geometric_series_sum_l757_757837


namespace solve_log_inequality_l757_757767

theorem solve_log_inequality (x : ℝ) :
  log 3 (abs (x - 1 / 3)) < -1 ↔ (0 < x ∧ x < 1 / 3) ∨ (1 / 3 < x ∧ x < 2 / 3) :=
by {
  -- Descriptive steps to be implemented in Lean proof
  sorry
}

end solve_log_inequality_l757_757767


namespace number_of_square_free_odds_l757_757645

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757645


namespace a6_value_l757_757585

noncomputable def a_n (d : ℝ) (n : ℕ) : ℝ := 
  if n = 0 then 0 else (1 : ℝ) / 4 + (n - 1) * d

theorem a6_value (d : ℝ) : 
  (∀ n, a_n d n > 0) → 
  (∀ n, ∑ i in range n, a_n d i = (1 / 4) + (↑n - 1) * d) → 
  (∀ n, sqrt (∑ i in range n, a_n d i) = (1 : ℝ) / 2 + (↑n - 1) * d) → 
  a_n d 6 = 11 / 4 :=
sorry

end a6_value_l757_757585


namespace hyperbola_midpoint_exists_l757_757241

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757241


namespace decrypt_ciphertext_l757_757023

def A : Set Char := {'a', 'b', 'c', ..., 'x', 'y', 'z'}

def f (c : Char) : Char :=
  if c = 'a' then 'b' else if c = 'b' then 'c' else if c = 'c' then 'd' else if c = 'd' then 'e'
  else if c = 'e' then 'f' else if c = 'f' then 'g' else if c = 'g' then 'h' else if c = 'h' then 'i'
  else if c = 'i' then 'j' else if c = 'j' then 'k' else if c = 'k' then 'l' else if c = 'l' then 'm'
  else if c = 'm' then 'n' else if c = 'n' then 'o' else if c = 'o' then 'p' else if c = 'p' then 'q'
  else if c = 'q' then 'r' else if c = 'r' then 's' else if c = 's' then 't' else if c = 't' then 'u'
  else if c = 'u' then 'v' else if c = 'v' then 'w' else if c = 'w' then 'x' else if c = 'x' then 'y'
  else if c = 'y' then 'z' else 'a'

theorem decrypt_ciphertext (ciphertext : String) (plaintext : String) : ciphertext = "nbui" → plaintext = "math" :=
  sorry

end decrypt_ciphertext_l757_757023


namespace arithmetic_sequence_sum_l757_757841

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l757_757841


namespace students_received_A_count_l757_757895

theorem students_received_A_count :
  ∀ (total : ℕ), (total = 39) →
  (C_fraction = 1 / 3) →
  (B_fraction = 5 / 13) →
  ((C_fraction + B_fraction) * total = 28) →
  (students_F = 1) →
  total - ((C_fraction + B_fraction) * total).nat_floor - students_F = 10 :=
by
  intros total h_total h_fraction_c h_fraction_b h_sum h_F
  sorry

end students_received_A_count_l757_757895


namespace first_sampled_individual_l757_757570

theorem first_sampled_individual (N n last_sampled first_sampled segment_size : ℕ)
  (hN : N = 8000)
  (hn : n = 50)
  (hlast : last_sampled = 7894)
  (hsegment : segment_size = N / n)
  (hlast_segment_start : hlast - (segment_size - 1) = first_sampled) :
  first_sampled = 735 := by
  sorry

end first_sampled_individual_l757_757570


namespace expected_value_of_flipped_coins_l757_757881

theorem expected_value_of_flipped_coins :
  let p := 1
  let n := 5
  let d := 10
  let q := 25
  let f := 50
  let prob := (1:ℝ) / 2
  let V := prob * p + prob * n + prob * d + prob * q + prob * f
  V = 45.5 :=
by
  sorry

end expected_value_of_flipped_coins_l757_757881


namespace f_lt_expression_l757_757729

noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.sqrt x - 1

theorem f_lt_expression (x : ℝ) (h : 1 < x) : 
  f(x) < (3 / 2) * (x - 1) := by
  sorry

end f_lt_expression_l757_757729


namespace length_of_spiraled_string_l757_757482

variable (circumference height : ℕ) (loops : ℕ) (vertical_rise horizontal_distance per_loop_length total_length : ℝ)

def length_of_string (circumference : ℕ) (height : ℕ) (loops : ℕ) : ℝ :=
  let vertical_rise := height / loops
  let horizontal_distance := circumference
  let per_loop_length := Real.sqrt(vertical_rise^2 + horizontal_distance^2)
  loops * per_loop_length

theorem length_of_spiraled_string 
  (circumference : ℕ) (height : ℕ) (loops : ℕ) 
  (h1 : circumference = 4) 
  (h2 : height = 16) 
  (h3 : loops = 5) 
  : length_of_string circumference height loops = 5 * Real.sqrt(26.24) :=
by
  sorry

end length_of_spiraled_string_l757_757482


namespace min_sticks_for_13_triangles_l757_757818

theorem min_sticks_for_13_triangles : 
  ∀ (n : ℕ), (n ≥ 1 → (n = 1 → 3) ∧ (n = 2 → 5) ∧ (n = 3 → 7)) → minimum_sticks 13 = 27 := 
by
  sorry

end min_sticks_for_13_triangles_l757_757818


namespace find_replaced_weight_l757_757786

-- Define the conditions and the hypothesis
def replaced_weight (W : ℝ) : Prop :=
  let avg_increase := 2.5
  let num_persons := 8
  let new_weight := 85
  (new_weight - W) = num_persons * avg_increase

-- Define the statement we aim to prove
theorem find_replaced_weight : replaced_weight 65 :=
by
  -- proof goes here
  sorry

end find_replaced_weight_l757_757786


namespace inequality_solution_l757_757770

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l757_757770


namespace part_a_part_b_part_c_l757_757275

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l757_757275


namespace midpoint_on_hyperbola_l757_757053

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757053


namespace midpoint_on_hyperbola_l757_757047

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757047


namespace solution_set_inequality_l757_757552

theorem solution_set_inequality (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ 2) ∨ (x = 6) := by
  sorry

end solution_set_inequality_l757_757552


namespace midpoint_on_hyperbola_l757_757043

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757043


namespace number_divided_by_3_equals_subtract_3_l757_757435

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l757_757435


namespace num_five_ruble_coins_l757_757318

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757318


namespace five_ruble_coins_count_l757_757310

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757310


namespace boat_travel_distance_downstream_l757_757477

-- Definitions of the given conditions
def boatSpeedStillWater : ℕ := 10 -- km/hr
def streamSpeed : ℕ := 8 -- km/hr
def timeDownstream : ℕ := 3 -- hours

-- Effective speed downstream
def effectiveSpeedDownstream : ℕ := boatSpeedStillWater + streamSpeed

-- Goal: Distance traveled downstream equals 54 km
theorem boat_travel_distance_downstream :
  effectiveSpeedDownstream * timeDownstream = 54 := 
by
  -- Since only the statement is needed, we use sorry to indicate the proof is skipped
  sorry

end boat_travel_distance_downstream_l757_757477


namespace quadratic_vertex_on_xaxis_l757_757680

theorem quadratic_vertex_on_xaxis (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0) → m = 1 / 4 :=
by
  assume h
  sorry

end quadratic_vertex_on_xaxis_l757_757680


namespace hyperbola_midpoint_l757_757155

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757155


namespace total_books_l757_757564

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l757_757564


namespace volleyball_team_selection_l757_757756

noncomputable def numberOfWaysToChooseStarters : ℕ :=
  (Nat.choose 13 4 * 3) + (Nat.choose 14 4 * 1)

theorem volleyball_team_selection :
  numberOfWaysToChooseStarters = 3146 := by
  sorry

end volleyball_team_selection_l757_757756


namespace midpoint_of_hyperbola_segment_l757_757092

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757092


namespace triangle_ADE_area_l757_757702

-- Given conditions
variables (ABC : Type) [triangle ABC]
variables (A B C D E : ABC)
variables (area_ABC : ℝ) (h_area_ABC : area_ABC = 1440)
variables (DE_parallel_AC : parallel DE AC)
variables (ratio_CE_EB : ℝ) (h_ratio : ratio_CE_EB = 1 / 3)

-- Required to prove
theorem triangle_ADE_area :
  ∃ x : ℝ, x = 270 ∧ area A D E = x :=
by sorry

end triangle_ADE_area_l757_757702


namespace cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l757_757001

theorem cannot_represent_1986_as_sum_of_squares_of_6_odd_integers
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 % 2 = 1) 
  (h2 : a2 % 2 = 1) 
  (h3 : a3 % 2 = 1) 
  (h4 : a4 % 2 = 1) 
  (h5 : a5 % 2 = 1) 
  (h6 : a6 % 2 = 1) : 
  ¬ (1986 = a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2) := 
by 
  sorry

end cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l757_757001


namespace midpoint_on_hyperbola_l757_757075

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757075


namespace midpoint_on_hyperbola_l757_757069

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757069


namespace square_free_odd_integers_count_l757_757648

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757648


namespace find_diameter_of_garden_roller_l757_757371

-- Define the given parameters
def length_roller : ℝ := 2
def total_area_covered : ℝ := 52.8
def number_of_revolutions : ℝ := 6
def pi_value : ℝ := 22 / 7

-- Define the area covered per revolution
def area_per_revolution : ℝ := total_area_covered / number_of_revolutions

-- State the theorem to find the diameter of the garden roller
theorem find_diameter_of_garden_roller : ∃ D : ℝ, (pi_value * D * length_roller = area_per_revolution) ∧ D = 1.4 :=
sorry

end find_diameter_of_garden_roller_l757_757371


namespace midpoint_on_hyperbola_l757_757045

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757045


namespace geometric_sequence_sum_six_l757_757603

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : a 3 * a 5 = 64)
  (h4 : ∀ n, a n = a 1 * q^(n-1))
  (h5 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) :
  S 6 = 63 := 
sorry

end geometric_sequence_sum_six_l757_757603


namespace hyperbola_midpoint_l757_757197

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757197


namespace solve_for_n_l757_757891

-- Define the conditions and the proof goal in Lean
theorem solve_for_n :
  ∃ n : ℕ, 
  let pentagon_angle := 108
  let pentagon_ext_angle := 180 - pentagon_angle
  5 * pentagon_ext_angle / n = 360 ∧ n ≥ 3 := 
by
sry

end solve_for_n_l757_757891


namespace abraham_correct_geometry_questions_l757_757506

theorem abraham_correct_geometry_questions :
  ∃ (A G T C_a C_g : ℕ) (P P_a : ℚ),
  A = 30 ∧ 
  G = 50 ∧ 
  T = 80 ∧ 
  P = 0.80 ∧ 
  P_a = 0.70 ∧ 
  C_a = (P_a * A).toNat ∧ 
  C_g = (P * T).toNat - C_a ∧ 
  C_g = 43 :=
by 
  sorry

end abraham_correct_geometry_questions_l757_757506


namespace jade_living_expenses_l757_757711

-- Definitions from the conditions
variable (income : ℝ) (insurance_fraction : ℝ) (savings : ℝ) (P : ℝ)

-- Constants from the given problem
noncomputable def jadeIncome : income = 1600 := by sorry
noncomputable def jadeInsuranceFraction : insurance_fraction = 1 / 5 := by sorry
noncomputable def jadeSavings : savings = 80 := by sorry

-- The proof problem statement
theorem jade_living_expenses :
    (P * 1600 + (1 / 5) * 1600 + 80 = 1600) → P = 3 / 4 := by
    intros h
    sorry

end jade_living_expenses_l757_757711


namespace circle_equation_slope_intercept_l757_757579
noncomputable theory

open_locale real

-- Definitions of points A and B
def A := (0 : ℝ, 2 : ℝ)
def B := (2 : ℝ, -2 : ℝ)

-- Condition: The center of the circle lies on the line x - y + 1 = 0.
def center_condition (center : ℝ × ℝ) : Prop := 
  center.fst - center.snd + 1 = 0

-- Question I: Find the standard equation of circle C.
theorem circle_equation (t : ℝ) (center := (t, t + 1)) (r := sqrt ((-3) ^ 2 + (-2 - 2) ^ 2)) 
  (h_center : center_condition center) (h_center_eq : center = (-3, -2)): 
  (r ^ 2 = 25) → 
  ((∀ (x y : ℝ), (x + 3) ^ 2 + (y + 2) ^ 2) = 25) :=
sorry

-- Question II: Find the slope-intercept equation of line m.
theorem slope_intercept (k : ℝ) (intercept := (43 : ℝ) / 12)
  (h1 : k = 5 / 12) :
  ((∀ (x y : ℝ), y = k * x + intercept)) :=
sorry

end circle_equation_slope_intercept_l757_757579


namespace midpoint_on_hyperbola_l757_757134

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757134


namespace complex_number_solution_l757_757606

theorem complex_number_solution (i : ℂ) (h : i^2 = -1) : (5 / (2 - i) - i = 2) :=
  sorry

end complex_number_solution_l757_757606


namespace tan_neg_405_eq_neg_1_l757_757927

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l757_757927


namespace problem1_problem2_l757_757469

-- Problem 1
theorem problem1 : 0.027 ^ (-1/3) + (Real.sqrt 8) ^ (4/3) - 3 ^ (-1) + (Real.sqrt 2 - 1) ^ 0 = 8 := 
by 
  sorry

-- Problem 2
theorem problem2 : (Real.log10 25 + Real.log10 4 + 7 ^ (Real.log 2 / Real.log 7) + (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 6 := 
by 
  sorry

end problem1_problem2_l757_757469


namespace midpoint_on_hyperbola_l757_757143

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757143


namespace imaginary_part_is_neg_two_l757_757381

open Complex

noncomputable def imaginary_part_of_square : ℂ := (1 - I)^2

theorem imaginary_part_is_neg_two : imaginary_part_of_square.im = -2 := by
  sorry

end imaginary_part_is_neg_two_l757_757381


namespace midpoint_hyperbola_l757_757226

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757226


namespace square_free_odd_integers_count_l757_757656

theorem square_free_odd_integers_count :
  let positiveOddIntegers := {n : ℕ | 1 < n ∧ n < 200 ∧ n % 2 = 1}
  let squareFree := λ x : ℕ, ∀ m : ℕ, m * m ∣ x → m = 1
  (∃ S : Finset ℕ, S.card = 82 ∧ ∀ n ∈ S, n ∈ positiveOddIntegers ∧ squareFree n) :=
sorry

end square_free_odd_integers_count_l757_757656


namespace midpoint_hyperbola_l757_757227

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757227


namespace functions_with_inverses_l757_757450

variable {A B C D E F G H} -- Define variables for all functions

-- Define the domains for all functions
def domain_a := Iic (3 : ℝ)
def domain_b := univ
def domain_c := Ioi (0 : ℝ)
def domain_d := Ici (3/2 : ℝ)
def domain_e := univ
def domain_f := univ
def domain_g := (Iio (-1 : ℝ)) ∪ (Ioi (-1 : ℝ))
def domain_h := Ico (-6 : ℝ) 9

-- Define the functions
def a (x : ℝ) : ℝ := -sqrt (3 - x)
def b (x : ℝ) : ℝ := x^3 + x
def c (x : ℝ) : ℝ := x - 2 / x
def d (x : ℝ) : ℝ := 4 * x^2 - 12 * x + 9
def e (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)
def f (x : ℝ) : ℝ := 2^x + exp x
def g (x : ℝ) : ℝ := (x - 1) / (x + 1)
def h (x : ℝ) : ℝ := -x / 3

-- The statement to prove functions having inverses
theorem functions_with_inverses :
  (bijective (λ x, a x) ∧ (∀ x ∈ domain_a, true)) ∧
  (¬ bijective (λ x, b x) ∧ (∀ x ∈ domain_b, true)) ∧
  (bijective (λ x, c x) ∧ (∀ x ∈ domain_c, true)) ∧
  (bijective (λ x, d x) ∧ (∀ x ∈ domain_d, true)) ∧
  (¬ bijective (λ x, e x) ∧ (∀ x ∈ domain_e, true)) ∧
  (bijective (λ x, f x) ∧ (∀ x ∈ domain_f, true)) ∧
  (bijective (λ x, g x) ∧ (∀ x ∈ domain_g, true)) ∧
  (bijective (λ x, h x) ∧ (∀ x ∈ domain_h, true)) :=
by
  sorry

end functions_with_inverses_l757_757450


namespace roots_unit_modulus_l757_757759

noncomputable def polynomial (z : ℂ) : ℂ := 11 * z^10 + 10 * complex.I * z^9 + 10 * complex.I * z - 11

theorem roots_unit_modulus (z : ℂ) (h : polynomial z = 0) : ∥z∥ = 1 :=
sorry

end roots_unit_modulus_l757_757759


namespace length_EH_l757_757343

theorem length_EH
  (inscribed : Quadrilateral E F G H)
  (angle_EFG : ∠EFG = 60)
  (angle_EHG : ∠EHG = 70)
  (length_EF : EF = 3)
  (length_FG : FG = 7) :
  EH = 3 * Real.sin 50 / Real.sin 70 :=
sorry

end length_EH_l757_757343


namespace min_energy_at_t2_l757_757398

-- Define the data and conditions as given in the problem
def Q_0 : ℝ := 10000 -- Initial energy in kJ
def M : ℝ := 60 -- Mass of athlete in kg
def v1 : ℝ := 30 -- Speed during stable phase in km/h
def ΔQ1 (t1 : ℝ) : ℝ := t1 * 2 * v1 -- Energy expended per kg during stable phase
def v2 (t2 : ℝ) : ℝ := 30 - 10 * t2 -- Speed during fatigue phase in km/h
def ΔQ2 (t2 : ℝ) : ℝ := (t2 * 2 * v2 t2) / (t2 + 1) -- Energy expended per kg during fatigue phase

-- Define the function Q(t)
def Q (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 1 then Q_0 - M * ΔQ1 t
  else if 1 < t ∧ t ≤ 4 then 400 + 1200 * t + 4800 / t
  else 0

-- Define the theorem we want to prove
theorem min_energy_at_t2 : ∃ t ∈ (1:ℝ) ─ (4:ℝ), Q t = 5200 ∧ ∀ s ∈ (1:ℝ) ─ (4:ℝ), Q t ≤ Q s := by
  sorry

end min_energy_at_t2_l757_757398


namespace midpoint_of_line_segment_on_hyperbola_l757_757028

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757028


namespace five_ruble_coins_count_l757_757309

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757309


namespace midpoint_of_line_segment_on_hyperbola_l757_757027

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757027


namespace num_five_ruble_coins_l757_757336

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757336


namespace midpoint_of_hyperbola_l757_757166

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757166


namespace function_sum_even_odd_l757_757764

theorem function_sum_even_odd (f : ℝ → ℝ) : 
  let g := λ x, (f x + f (-x)) / 2 in
  let h := λ x, (f x - f (-x)) / 2 in
  f = (λ x, g x + h x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h x = -h (-x)) :=
by
  sorry

end function_sum_even_odd_l757_757764


namespace smallest_positive_integer_a_l757_757378

theorem smallest_positive_integer_a :
  ∀ (a b c : ℤ), (∃ r s ∈ Ioo 0 1, r ≠ s ∧ a * (r * (1 - r) * s * (1 - s)) > 1) → a ≥ 5 :=
by
  sorry

end smallest_positive_integer_a_l757_757378


namespace perpendicular_bisectors_concurrent_altitudes_concurrent_l757_757536

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def midpoint (P Q : A) [PseudoMetricSpace A] : A := sorry
noncomputable def is_perpendicular_bisector (P Q : A) (R : B) : Prop := sorry
noncomputable def altitude (A B C : Type) [PseudoMetricSpace A] : Type := sorry

theorem perpendicular_bisectors_concurrent (A B C : A):
  ∃ O : A, is_perpendicular_bisector A B O ∧ is_perpendicular_bisector B C O ∧ is_perpendicular_bisector C A O :=
sorry

theorem altitudes_concurrent (A B C : A):
  ∃ H : A, altitude A B C = H :=
sorry

end perpendicular_bisectors_concurrent_altitudes_concurrent_l757_757536


namespace leftover_value_l757_757006

/--
James and Lindsay have a jar containing a collective sum of 100 quarters and 185 dimes. 
The capacity of a roll for quarters is 45, and for dimes it is 55. 
Determine the total dollar value of the quarters and dimes that cannot be rolled.
-/
theorem leftover_value :
  let quarters := 100
  let dimes := 185
  let roll_capacity_quarters := 45
  let roll_capacity_dimes := 55
  let leftover_quarters := quarters % roll_capacity_quarters
  let leftover_dimes := dimes % roll_capacity_dimes
  let value_per_quarter := 0.25
  let value_per_dime := 0.10
  (leftover_quarters * value_per_quarter + leftover_dimes * value_per_dime) = 4.50 := 
by
  sorry

end leftover_value_l757_757006


namespace range_of_m_l757_757575

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 + m * x + 1 = 0 → x ≠ 0) ∧ ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l757_757575


namespace tan_neg405_deg_l757_757939

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l757_757939


namespace steve_height_after_growth_l757_757363

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l757_757363


namespace period_f_achieve_extremes_min_pos_k_l757_757261

noncomputable def f (k : ℤ) (x : ℝ) : ℝ := sin (k * x / 5 + π / 3)

def max_value : ℝ := 1

def min_value : ℝ := -1

def period (k : ℤ) : ℝ := (10 * π) / k

theorem period_f (k : ℤ) (h : k ≠ 0) : period k > 0 := by
  rw [period]
  apply div_pos
  norm_num
  exact_mod_cast h

theorem achieve_extremes (x : ℝ) (k : ℤ) : 
  (f k x = max_value ∨ f k x = min_value) ↔ 
  ∃ (a b : ℤ), (a ≤ x ∧ x ≤ b ∧ (f k a = max_value ∨ f k a = min_value) ∧ (f k b = max_value ∨ f k b = min_value)) :=
sorry

noncomputable def proper_k : ℤ := 32

theorem min_pos_k : 
  ∀ (x : ℝ), 
    (∃ k : ℤ, k ≥ 32 ∧ 
      (f k x = max_value ∨ f k x = min_value)) :=
sorry

end period_f_achieve_extremes_min_pos_k_l757_757261


namespace jars_proof_l757_757013

def total_plums : ℕ := 240
def exchange_ratio : ℕ := 7
def mangoes_per_jar : ℕ := 5

def ripe_plums (total_plums : ℕ) := total_plums / 4
def unripe_plums (total_plums : ℕ) := 3 * total_plums / 4
def unripe_plums_kept : ℕ := 46

def plums_for_trade (total_plums unripe_plums_kept : ℕ) : ℕ :=
  ripe_plums total_plums + (unripe_plums total_plums - unripe_plums_kept)

def mangoes_received (plums_for_trade exchange_ratio : ℕ) : ℕ :=
  plums_for_trade / exchange_ratio

def jars_of_mangoes (mangoes_received mangoes_per_jar : ℕ) : ℕ :=
  mangoes_received / mangoes_per_jar

theorem jars_proof : jars_of_mangoes (mangoes_received (plums_for_trade total_plums unripe_plums_kept) exchange_ratio) mangoes_per_jar = 5 :=
by
  sorry

end jars_proof_l757_757013


namespace scientific_notation_of_tourists_l757_757899

theorem scientific_notation_of_tourists : 
  (23766400 : ℝ) = 2.37664 * 10^7 :=
by 
  sorry

end scientific_notation_of_tourists_l757_757899


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757295

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757295


namespace midpoint_hyperbola_l757_757219

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757219


namespace probability_of_heads_on_999th_toss_l757_757819

theorem probability_of_heads_on_999th_toss (fair_coin : Bool → ℝ) :
  (∀ (i : ℕ), fair_coin true = 1 / 2 ∧ fair_coin false = 1 / 2) →
  fair_coin true = 1 / 2 :=
by
  sorry

end probability_of_heads_on_999th_toss_l757_757819


namespace petya_five_ruble_coins_count_l757_757319

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757319


namespace hyperbola_midpoint_l757_757195

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757195


namespace fractional_part_exceeds_bound_l757_757721

noncomputable def x (a b : ℕ) : ℝ := Real.sqrt a + Real.sqrt b

theorem fractional_part_exceeds_bound
  (a b : ℕ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hx_not_int : ¬ (∃ n : ℤ, x a b = n))
  (hx_lt : x a b < 1976) :
    x a b % 1 > 3.24e-11 :=
sorry

end fractional_part_exceeds_bound_l757_757721


namespace distance_y_axis_l757_757693

def point_M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2 * m)

theorem distance_y_axis :
  ∀ m : ℝ, abs (2 - m) = 2 → (point_M m = (2, 1)) ∨ (point_M m = (-2, 9)) :=
by
  sorry

end distance_y_axis_l757_757693


namespace problem_statement_l757_757600

variables {A B C x y x0 y0 : ℝ}

noncomputable def line_L (x y : ℝ) : Prop := A * x + B * y + C = 0
noncomputable def point_outside_line (x0 y0 : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ A * x0 + B * y0 + C = k

theorem problem_statement (h : point_outside_line x0 y0) : 
  ∀ x y : ℝ, A * x + B * y + C + (A * x0 + B * y0 + C) = 0 ↔ 
  (A * x + B * y + C + (A * x0 + B * y0 + C) = 0) ∧ 
  (A * x + B * y + C + (A * x0 + B * y0 + C) ≠ A * x0 + B * y0 + C) ∧
  (A * x + B * y + C + (A * x0 + B * y0 + C) ≠ 0) :=
begin
  sorry
end

end problem_statement_l757_757600


namespace unique_pair_exists_l757_757951

theorem unique_pair_exists :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (a + b + (Nat.gcd a b)^2 = Nat.lcm a b) ∧
  (Nat.lcm a b = 2 * Nat.lcm (a - 1) b) ∧
  (a, b) = (6, 15) :=
sorry

end unique_pair_exists_l757_757951


namespace exists_positive_integer_k_l757_757341

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ¬ Nat.Prime (2^n * k + 1) ∧ 2^n * k + 1 > 1 :=
by
  sorry

end exists_positive_integer_k_l757_757341


namespace estimate_pi_l757_757484

open Real

noncomputable def approximate_pi (m n : ℕ) [fact (0 < n)] : ℝ :=
  (4 * m / n.to_real) + 2

theorem estimate_pi (m n : ℕ) [fact (0 < n)] :
  let pi := ∑ i in finset.range n, if (let x := (i:ℝ / n.to_real); let y := ((i.succ:ℝ) / n.to_real) in
                                       x ^ 2 + y ^ 2 < 1 ∧ x + y > 1) then 1 else 0 in
  pi.to_real / (n:ℝ) = 4 * m / n + 2 :=
sorry

end estimate_pi_l757_757484


namespace five_ones_make_100_l757_757849

noncomputable def concatenate (a b c : Nat) : Nat :=
  a * 100 + b * 10 + c

theorem five_ones_make_100 :
  let one := 1
  let x := concatenate one one one -- 111
  let y := concatenate one one 0 / 10 -- 11, concatenation of 1 and 1 treated as 110, divided by 10
  x - y = 100 :=
by
  sorry

end five_ones_make_100_l757_757849


namespace M_inter_N_eq_l757_757633

def M : set ℝ := { x | -1 < x ∧ x < 1 }
def N : set ℤ := { x | x^2 < 2 }

theorem M_inter_N_eq {x : ℤ} : (M ∩ N : set ℝ) = { (0 : ℤ) } :=
sorry

end M_inter_N_eq_l757_757633


namespace problem_statement_l757_757871

structure Pricing :=
  (price_per_unit_1 : ℕ) (threshold_1 : ℕ)
  (price_per_unit_2 : ℕ) (threshold_2 : ℕ)
  (price_per_unit_3 : ℕ)

def cost (units : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if units ≤ t1 then units * p1
  else if units ≤ t2 then t1 * p1 + (units - t1) * p2
  else t1 * p1 + (t2 - t1) * p2 + (units - t2) * p3 

def units_given_cost (c : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if c ≤ t1 * p1 then c / p1
  else if c ≤ t1 * p1 + (t2 - t1) * p2 then t1 + (c - t1 * p1) / p2
  else t2 + (c - t1 * p1 - (t2 - t1) * p2) / p3

def double_eleven_case (total_units total_cost : ℕ) (x_units : ℕ) (pricing : Pricing) : ℕ :=
  let y_units := total_units - x_units
  let case1_cost := cost x_units pricing + cost y_units pricing
  if case1_cost = total_cost then (x_units, y_units).fst
  else sorry

theorem problem_statement (pricing : Pricing):
  (cost 120 pricing = 420) ∧ 
  (cost 260 pricing = 868) ∧
  (units_given_cost 740 pricing = 220) ∧
  (double_eleven_case 400 1349 290 pricing = 290)
  := sorry

end problem_statement_l757_757871


namespace circle_diameter_C_l757_757527

theorem circle_diameter_C {D C : ℝ} (hD : D = 20) (h_ratio : (π * (D/2)^2 - π * (C/2)^2) / (π * (C/2)^2) = 4) : C = 4 * Real.sqrt 5 := 
sorry

end circle_diameter_C_l757_757527


namespace square_free_odd_integers_count_l757_757669

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757669


namespace square_free_odd_integers_count_l757_757671

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def count_square_free_odd_integers (lower upper : ℕ) : ℕ :=
  (List.range' (lower + 1) (upper - lower - 1)).filter (λ n, n % 2 = 1 ∧ is_square_free n).length

theorem square_free_odd_integers_count : count_square_free_odd_integers 1 200 = 79 := 
by
  unfold count_square_free_odd_integers
  unfold is_square_free
  sorry

end square_free_odd_integers_count_l757_757671


namespace midpoint_hyperbola_l757_757102

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757102


namespace part_a_part_b_part_c_l757_757948

-- Part (a)
theorem part_a (A B : ConvexPolyhedron) (HA : A.planesOfSymmetry = 2012) (HB : B.planesOfSymmetry = 2012) (HAB : A ∩ B = ∅) :
  (A ∪ B).planesOfSymmetry = 2013 := 
sorry

-- Part (b)
theorem part_b (A B : ConvexPolyhedron) (HA : A.planesOfSymmetry = 2012) (HB : B.planesOfSymmetry = 2013) (HAB : A ∩ B = ∅) :
  (A ∪ B).planesOfSymmetry = 2012 := 
sorry

-- Part (c)
theorem part_c (A B : ConvexPolyhedron) (HA : A.planesOfSymmetry = 2012) (HB : B.axesOfSymmetry = 2013) (HAB : A ∩ B = ∅) :
  (A ∪ B).axesOfSymmetry = 1 := 
sorry

end part_a_part_b_part_c_l757_757948


namespace midpoint_hyperbola_l757_757223

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757223


namespace steve_height_after_growth_l757_757362

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l757_757362


namespace integral_x_plus_one_over_x_l757_757960

open Real

theorem integral_x_plus_one_over_x :
  ∫ x in 1..e, (x + 1 / x) = (e ^ 2 + 1) / 2 :=
by
  sorry

end integral_x_plus_one_over_x_l757_757960


namespace palindrome_count_l757_757952

-- Define the properties of a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let d2 := n % 10 in
  let d1 := (n / 100) % 10 in
  d1 = d2

-- Define the set of three-digit numbers between 200 and 700 inclusive
def is_in_range (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 700

-- Define a predicate that a number is a three-digit palindrome and in the specified range
def is_valid_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ is_in_range n

-- Prove the number of integer palindromes between 200 and 700 is exactly 50
theorem palindrome_count : 
  (∑ n in finset.filter is_valid_palindrome (finset.range 701), 1) = 50 := 
sorry

end palindrome_count_l757_757952


namespace can_be_paired_1_to_10_can_be_paired_1_to_2014_l757_757855

-- Condition (a): Range 1 to 10
def pairs_1_to_10 (pair : (ℕ × ℕ)) : Prop :=
  let nums := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ p ∈ pairs, (p.1 ∈ nums) ∧ (p.2 ∈ nums) ∧ (p.1 - p.2 = 2 ∨ p.1 - p.2 = 3 ∨ p.2 - p.1 = 2 ∨ p.2 - p.1 = 3))
    ∧ list.to_finset (pairs.map prod.fst ∪ pairs.map prod.snd) = list.to_finset nums

theorem can_be_paired_1_to_10 : pairs_1_to_10 sorry := sorry

-- Condition (b): Range 1 to 2014
def pairs_1_to_2014 (pair : (ℕ × ℕ)) : Prop :=
  let nums := list.range' 1 2014
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ p ∈ pairs, (p.1 ∈ nums) ∧ (p.2 ∈ nums) ∧ (p.1 - p.2 = 2 ∨ p.1 - p.2 = 3 ∨ p.2 - p.1 = 2 ∨ p.2 - p.1 = 3))
    ∧ list.to_finset (pairs.map prod.fst ∪ pairs.map prod.snd) = list.to_finset nums

theorem can_be_paired_1_to_2014 : pairs_1_to_2014 sorry := sorry

end can_be_paired_1_to_10_can_be_paired_1_to_2014_l757_757855


namespace hyperbola_midpoint_l757_757190

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757190


namespace find_polynomials_l757_757259

theorem find_polynomials (P : ℤ → ℤ) : 
  (∀ a : ℕ → ℤ, (∀ n : ℤ, ∃ m, a m = n) → 
    ∃ i j k, i < j ∧ ∑ n in finset.Ico i j, a n = P k) ↔
  (∃ c d : ℤ, c ≠ 0 ∧ ∀ x, P x = c * x + d) :=
by
  sorry

end find_polynomials_l757_757259


namespace standard_deviation_of_set_l757_757613

/-- Given the average of four numbers 3, 5, x, 7 is 6,
prove that the standard deviation of this set of data is √5. -/
theorem standard_deviation_of_set :
  ∃ (x : ℝ), (3 + 5 + x + 7) / 4 = 6 → 
  let mean := 6 in 
  let set := {3, 5, x, 7} in
  let variance := (1 / 4) * ((3 - mean) ^ 2 + (5 - mean) ^ 2 + (x - mean) ^ 2 + (7 - mean) ^ 2) in
  real.sqrt variance = real.sqrt 5 := 
begin
  sorry
end

end standard_deviation_of_set_l757_757613


namespace find_c_degree3_l757_757945

-- Definition of the polynomials f and g.
def f (x : ℝ) : ℝ := 2 - 15 * x + 4 * x^2 - 5 * x^3 + 6 * x^4
def g (x : ℝ) : ℝ := 4 - 3 * x - 7 * x^3 + 10 * x^4

-- Assertion that the given value of c makes f(x) + c * g(x) a degree 3 polynomial.
theorem find_c_degree3 : (c : ℝ) = -3/5 → 
  ∀ x : ℝ, degree (polynomial.of_continuous_function (λ x, f x + c * (g x))) = 3 := sorry

end find_c_degree3_l757_757945


namespace ming_dynasty_wine_problem_l757_757697

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end ming_dynasty_wine_problem_l757_757697


namespace midpoint_on_hyperbola_l757_757080

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757080


namespace number_of_square_free_odds_l757_757643

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757643


namespace midpoint_on_hyperbola_l757_757137

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757137


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757281

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757281


namespace simplify_expression_l757_757353

open Real

theorem simplify_expression :
    (3 * (sqrt 5 + sqrt 7) / (4 * sqrt (3 + sqrt 5))) = sqrt (414 - 98 * sqrt 35) / 8 :=
by
  sorry

end simplify_expression_l757_757353


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757297

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757297


namespace perimeter_KLMN_geq_double_AC_l757_757784

theorem perimeter_KLMN_geq_double_AC
  (A B C D K L M N: ℝ^2)
  (h_convex: convex_quadrilateral A B C D)
  (h_not_acute_A: ¬acute_angle A B D)
  (h_not_acute_C: ¬acute_angle C D B)
  (h_K_on_AB: point_on_segment K A B)
  (h_L_on_BC: point_on_segment L B C)
  (h_M_on_CD: point_on_segment M C D)
  (h_N_on_DA: point_on_segment N D A) :
  perimeter (quadrilateral K L M N) ≥ 2 * dist A C :=
sorry

end perimeter_KLMN_geq_double_AC_l757_757784


namespace midpoint_hyperbola_l757_757112

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757112


namespace max_b_value_l757_757401

theorem max_b_value
  (a b c : ℕ)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = 240) : b = 10 :=
  sorry

end max_b_value_l757_757401


namespace find_x_l757_757426

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l757_757426


namespace parabola_focus_l757_757989

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757989


namespace intersection_is_correct_l757_757538

-- Define the sets M and N based on the given conditions
def M : Set ℝ := {x | log x / log 10 > 0 }
def N : Set ℝ := {x | |x| ≤ 2 }

-- Statement to prove that M ∩ N = (1, 2]
theorem intersection_is_correct : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_is_correct_l757_757538


namespace number_divided_by_3_equals_subtract_3_l757_757434

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l757_757434


namespace can_be_midpoint_of_AB_l757_757179

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757179


namespace num_five_ruble_coins_l757_757335

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757335


namespace hyperbola_midpoint_l757_757156

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757156


namespace find_k_l757_757638

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

end find_k_l757_757638


namespace daughter_percentage_younger_than_betty_l757_757521

-- Define the conditions
variables (B D G : ℕ)
assume h1 : B = 60
assume h2 : G = 12
assume h3 : G = D / 3

-- Define the percentage calculation
def percentage_younger (B D : ℕ) : ℕ := 100 * (B - D) / B

-- Theorem statement
theorem daughter_percentage_younger_than_betty : percentage_younger 60 36 = 40 := 
by sorry

end daughter_percentage_younger_than_betty_l757_757521


namespace find_m_l757_757379

-- Definition and conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def vertex_property (a b c : ℝ) : Prop := 
  (∀ x, quadratic a b c x ≤ quadratic a b c 2) ∧ quadratic a b c 2 = 4

noncomputable def passes_through_origin (a b c : ℝ) : Prop :=
  quadratic a b c 0 = -7

-- Main theorem statement
theorem find_m (a b c m : ℝ) 
  (h1 : vertex_property a b c) 
  (h2 : passes_through_origin a b c) 
  (h3 : quadratic a b c 5 = m) :
  m = -83/4 :=
sorry

end find_m_l757_757379


namespace symmetrical_transformation_l757_757617

theorem symmetrical_transformation 
    (t s : ℝ) :
  let C (x : ℝ) := x^3 - x in
  let C1 (x : ℝ) := (x - t)^3 - (x - t) + s in
  ∀ x y, (y = C x ↔ y = C1 (2 * (t / 2) - x)) :=
begin
  sorry
end

end symmetrical_transformation_l757_757617


namespace derivative_correct_l757_757547

noncomputable def y (x : ℝ) : ℝ := (1 / 6) * Real.log ((1 - Real.sinh(2 * x)) / (2 + Real.sinh(2 * x)))
noncomputable def y_derivative (x : ℝ) : ℝ := (1 / 6) * ((1 - Real.sinh(2 * x)) / (2 + Real.sinh(2 * x))).derivative
noncomputable def expected_y_derivative (x : ℝ) : ℝ := Real.cosh(2 * x) / (Real.sinh(2 * x)^2 + Real.sinh(2 * x) - 2)

theorem derivative_correct (x : ℝ) : y_derivative x = expected_y_derivative x :=
sorry

end derivative_correct_l757_757547


namespace parity_equivalence_l757_757740

open Nat

theorem parity_equivalence (p q : ℕ) : 
  (even (p^3 - q^3) ↔ even (p + q)) :=
by
  sorry

end parity_equivalence_l757_757740


namespace hyperbola_midpoint_l757_757158

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757158


namespace correct_midpoint_l757_757206

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757206


namespace hyperbola_midpoint_l757_757193

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757193


namespace polygon_coloring_l757_757691

open Nat

theorem polygon_coloring
  (n m : ℕ)
  (hn : n ≥ 3) (hm : 2 ≤ m ≤ n) :
  (choose n m) * (2^m + (-1)^m * 2) = 
  sorry

end polygon_coloring_l757_757691


namespace installation_cost_l757_757344

noncomputable def Ramesh_LP : ℝ := 13500 / 0.80
noncomputable def Ramesh_SP : ℝ := Ramesh_LP * 1.10
def given_SP : ℝ := 18975
def transport_cost : ℝ := 125

theorem installation_cost :
  let extra_amount := given_SP - Ramesh_SP in
  let installation_cost := extra_amount - transport_cost in
  installation_cost = 287.50 :=
by
  sorry

end installation_cost_l757_757344


namespace part1_part2_l757_757576

noncomputable theory
open_locale classical

-- Define the conditions p and q as predicates
def p (x : ℝ) : Prop := (x + 1) / (x - 2) > 2
def q (x a : ℝ) : Prop := x^2 - a * x + 5 > 0

-- Statement of part (1)
theorem part1 (x : ℝ) (h : p x) : 2 < x ∧ x < 5 := sorry

-- Statement of part (2)
theorem part2 (a : ℝ) (h : ∀ x, p x → q x a) : a < 2 * real.sqrt 5 := sorry

end part1_part2_l757_757576


namespace tangent_line_eq_l757_757979

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end tangent_line_eq_l757_757979


namespace negation_of_universal_proposition_l757_757597

variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)

theorem negation_of_universal_proposition :
  (∀ x1 x2 : R, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : R, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end negation_of_universal_proposition_l757_757597


namespace part_a_part_b_part_c_l757_757277

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l757_757277


namespace necessary_but_not_sufficient_condition_l757_757731

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem necessary_but_not_sufficient_condition (m : ℝ) (i : ℂ) (m_im : m ∈ ℝ) (i_im : i.im = 1) (i_re : i.re = 0) :
  (is_pure_imaginary (m * (m - 1) + i)) = (m = 1) :=
sorry

end necessary_but_not_sufficient_condition_l757_757731


namespace midpoint_hyperbola_l757_757220

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757220


namespace negation_of_proposition_l757_757596

theorem negation_of_proposition (p : ∀ x : ℝ, -x^2 + 4 * x + 3 > 0) :
  (∃ x : ℝ, -x^2 + 4 * x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l757_757596


namespace part_1_part_2_part_3_l757_757582

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3_l757_757582


namespace hyperbola_midpoint_l757_757128

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757128


namespace parallel_lines_distance_l757_757610

theorem parallel_lines_distance :
  ∀ (x y : ℝ),
  let l1 := 3 * x + 4 * y - 3,
      l2 := 6 * x + 8 * y + 14 in
  parallel l1 l2 → 
  distance l1 l2 = 2 :=
by
  sorry

end parallel_lines_distance_l757_757610


namespace hyperbola_midpoint_l757_757121

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757121


namespace total_books_l757_757566

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l757_757566


namespace add_second_largest_to_sum_l757_757843

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8

def form_number (d1 d2 d3 : ℕ) : ℕ := 100 * d1 + 10 * d2 + d3

def largest_number : ℕ := form_number 8 5 2
def smallest_number : ℕ := form_number 2 5 8
def second_largest_number : ℕ := form_number 8 2 5

theorem add_second_largest_to_sum : 
  second_largest_number + (largest_number + smallest_number) = 1935 := 
  sorry

end add_second_largest_to_sum_l757_757843


namespace HCl_mixture_l757_757640

theorem HCl_mixture (p : ℝ) (Hp : 0 < p ∧ p < 1) : 
  ∃ x : ℝ, 0 ≤ x ∧ (50 * 0.10 + x * 0.30) / (50 + x) = p :=
begin
  sorry
end

end HCl_mixture_l757_757640


namespace hyperbola_midpoint_l757_757202

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757202


namespace carl_candy_bars_l757_757962

theorem carl_candy_bars (earning_from_trash_per_week earning_from_dog_per_two_weeks earning_per_candy_bar monthly_gift_needed_to_buy goal_candy_bars : ℝ) (weeks_per_month days_per_week: ℝ) :
  earning_from_trash_per_week = 0.75 →
  earning_from_dog_per_two_weeks = 1.25 →
  earning_per_candy_bar = 0.50 →
  monthly_gift_needed_to_buy = 5 →
  goal_candy_bars = 30 →
  weeks_per_month = 4 →
  days_per_week = 7 →
  ∃ days_needed: ℝ, days_needed = 53 :=
begin
  sorry,
end

end carl_candy_bars_l757_757962


namespace odd_square_free_count_l757_757662

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757662


namespace midpoint_hyperbola_l757_757110

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757110


namespace steve_final_height_l757_757366

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l757_757366


namespace focus_of_parabola_l757_757999

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757999


namespace number_divided_by_3_equals_subtract_3_l757_757437

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l757_757437


namespace john_spent_fraction_at_arcade_l757_757010

theorem john_spent_fraction_at_arcade 
  (allowance : ℝ) (spent_arcade : ℝ) (spent_candy_store : ℝ) 
  (h1 : allowance = 3.45)
  (h2 : spent_candy_store = 0.92)
  (h3 : 3.45 - spent_arcade - (1/3) * (3.45 - spent_arcade) = spent_candy_store) :
  spent_arcade / allowance = 2.07 / 3.45 :=
by
  sorry

end john_spent_fraction_at_arcade_l757_757010


namespace doob_decomposition_l757_757741

-- Definitions for submartingale and necessary constructs.
variable (n : ℕ)
variable (ξ : Finₓ (n+1) → ℝ)
variable (ℱ : Finₓ (n+1) → Set (Set (ℝ)))
variable (A : Finₓ (n+1) → Set (ℝ))
variable (I : Set (ℝ) → ℝ)
variable (P : Set (ℝ) → Set (ℝ) → ℝ)

-- Assuming the conditions of the problem.
variable (h_submartingale : ∀ k : Finₓ n, ∑ m in Finset.range k, I (A m) ∈ ℱ k)
variable (h_def : ∀ k : Finₓ (n+1), ξ k = ∑ m in Finset.range k, I (A m))

-- The theorem we need to prove.
theorem doob_decomposition :
  ∀ k : Finₓ (n+1), ξ k = 
    (∑ i in Finset.range k, P (A i) (ℱ (i-1))) + 
    (ξ k - (∑ i in Finset.range k, P (A i) (ℱ (i-1)))) :=
by
  sorry

end doob_decomposition_l757_757741


namespace min_shaded_cells_for_L_tetromino_l757_757417

theorem min_shaded_cells_for_L_tetromino (B : matrix (fin 6) (fin 6) bool) :
  (∀ r c, (0 ≤ r ∧ r + 1 < 6 ∧ 0 ≤ c ∧ c + 2 < 6) → ∃ i j, (r ≤ i ∧ i ≤ r + 1 ∧ c ≤ j ∧ j ≤ c + 2 ∧ B i j = tt)) →
  (∀ r c, (0 ≤ r ∧ r + 2 < 6 ∧ 0 ≤ c ∧ c + 1 < 6) → ∃ i j, (r ≤ i ∧ i ≤ r + 2 ∧ c ≤ j ∧ j ≤ c + 1 ∧ B i j = tt)) →
  ∃ M ≤ 12, (count_tt_cells B = M) :=
by sorry

end min_shaded_cells_for_L_tetromino_l757_757417


namespace salon_revenue_l757_757911

noncomputable def revenue (num_customers first_visit second_visit third_visit : ℕ) (first_charge second_charge : ℕ) : ℕ :=
  num_customers * first_charge + second_visit * second_charge + third_visit * second_charge

theorem salon_revenue : revenue 100 100 30 10 10 8 = 1320 :=
by
  unfold revenue
  -- The proof will continue here.
  sorry

end salon_revenue_l757_757911


namespace odd_square_free_count_l757_757664

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757664


namespace marta_should_buy_84_ounces_l757_757748

/-- Definition of the problem's constants and assumptions --/
def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def bag_capacity : ℕ := 49
def num_bags : ℕ := 3

-- Marta wants to put the same number of apples and oranges in each bag
def equal_fruit (A O : ℕ) := A = O

-- Each bag should hold up to 49 ounces of fruit
def bag_limit (n : ℕ) := 4 * n + 3 * n ≤ 49

-- Marta's total apple weight based on the number of apples per bag and number of bags
def total_apple_weight (A : ℕ) : ℕ := (A * 3 * 4)

/-- Statement of the proof problem: 
Marta should buy 84 ounces of apples --/
theorem marta_should_buy_84_ounces : total_apple_weight 7 = 84 :=
by
  sorry

end marta_should_buy_84_ounces_l757_757748


namespace find_positive_integer_N_l757_757419

theorem find_positive_integer_N (N : ℕ) (h₁ : 33^2 * 55^2 = 15^2 * N^2) : N = 121 :=
by {
  sorry
}

end find_positive_integer_N_l757_757419


namespace proposition_l757_757593

variable {f g : ℝ → ℝ}
variable (h₁ : f 0 = g 0) 
variable (h₂ : f(0) > 0)
variable (h₃ : ∀ x ∈ (Set.Icc 0 1), f' x * sqrt (g' x) = 3)

theorem proposition (x : ℝ) (hx : x ∈ (Set.Icc 0 1)) : 2 * f x + 3 * g x > 9 * x :=
by
  sorry

end proposition_l757_757593


namespace midpoint_on_hyperbola_l757_757139

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757139


namespace min_positive_d_l757_757384

theorem min_positive_d (a b t d : ℤ) (h1 : 3 * t = 2 * a + 2 * b + 2016)
                                       (h2 : t - a = d)
                                       (h3 : t - b = 2 * d)
                                       (h4 : 2 * a + 2 * b > 0) :
    ∃ d : ℤ, d > 0 ∧ (505 ≤ d ∧ ∀ e : ℤ, e > 0 → 3 * (a + d) = 2 * (b + 2 * e) + 2016 → 505 ≤ e) := 
sorry

end min_positive_d_l757_757384


namespace curve_arc_length_l757_757464

noncomputable def arc_length_parametric (x y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in t1..t2, sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem curve_arc_length : 
  arc_length_parametric 
    (λ t, 1/2 * cos t - 1/4 * cos (2 * t)) 
    (λ t, 1/2 * sin t - 1/4 * sin (2 * t)) 
    (Real.pi / 2) (2 * Real.pi / 3) = sqrt 2 - 1 := 
sorry

end curve_arc_length_l757_757464


namespace unique_roots_of_system_l757_757537

theorem unique_roots_of_system {x y z : ℂ} 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end unique_roots_of_system_l757_757537


namespace hyperbola_midpoint_exists_l757_757236

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757236


namespace five_ruble_coins_count_l757_757306

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757306


namespace find_x_l757_757462

theorem find_x (x : ℤ) (h : (2 + 76 + x) / 3 = 5) : x = -63 := 
sorry

end find_x_l757_757462


namespace hyperbola_midpoint_exists_l757_757234

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757234


namespace num_five_ruble_coins_l757_757337

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757337


namespace largest_gcd_of_sum_1729_l757_757805

theorem largest_gcd_of_sum_1729 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1729) :
  ∃ g, g = Nat.gcd x y ∧ g = 247 := sorry

end largest_gcd_of_sum_1729_l757_757805


namespace count_two_digit_factors_of_n_l757_757672

-- Define the number n as 3^18 - 1
def n : ℕ := 3^18 - 1

-- Define a predicate to check if a number is a two-digit integer
def is_two_digit (x : ℕ) : Prop := x ≥ 10 ∧ x < 100

-- Define a predicate to check if a number divides n
def divides (x y : ℕ) : Prop := y % x = 0

-- Define the set of all two-digit factors of n
def two_digit_factors_of_n : Finset ℕ := (Finset.filter (λ x, divides x n ∧ is_two_digit x) (Finset.range (n + 1)))

-- State the theorem
theorem count_two_digit_factors_of_n : two_digit_factors_of_n.card = 3 :=
  sorry

end count_two_digit_factors_of_n_l757_757672


namespace infinite_pairs_exists_l757_757765

noncomputable def exists_infinite_pairs : Prop :=
  ∃ (a b : ℕ), (a + b ∣ a * b + 1) ∧ (a - b ∣ a * b - 1) ∧ b > 1 ∧ a > b * Real.sqrt 3 - 1

theorem infinite_pairs_exists : ∃ (count : ℕ) (a b : ℕ), ∀ n < count, exists_infinite_pairs :=
sorry

end infinite_pairs_exists_l757_757765


namespace midpoint_of_line_segment_on_hyperbola_l757_757034

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757034


namespace inequality_solution_l757_757768

noncomputable def bounds : Icc (-Real.sqrt 2) (Real.sqrt 2) := sorry

theorem inequality_solution (x : ℝ) (hx : x ∈ bounds) :
  (4 + x^2 + 2*x*Real.sqrt (2 - x^2) < 8*Real.sqrt (2 - x^2) + 5*x) ↔ (-1 < x ∧ x ≤ Real.sqrt 2) := sorry

end inequality_solution_l757_757768


namespace find_radius_l757_757821

-- Define the ellipse equation
def ellipse : ℝ → ℝ → ℝ := λ x y, 4 * x^2 + 6 * y^2 - 8

-- Define the circle equation shifted by radius r along the x-axis
def circle (r : ℝ) : ℝ → ℝ → ℝ := λ x y, (x - r)^2 + y^2 - r^2

-- Define the condition of the circle tangency to the ellipse at the center (r, 0)
def tangent_condition (r : ℝ) : Prop := 
  ∀ x y : ℝ, ellipse x y = 0 → circle r x y = 0

theorem find_radius (r : ℝ) : 
  (∀ x y : ℝ, ellipse x y = 0 → circle r x y = 0) ∧ (∀ x y : ℝ, ellipse (-x) y = 0 → circle r (-x) y = 0) → r = 2 / 3 :=
sorry

end find_radius_l757_757821


namespace petya_five_ruble_coins_count_l757_757322

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757322


namespace weight_of_3_moles_baf2_is_correct_l757_757418

noncomputable def atomic_weight_ba : ℝ := 137.33
noncomputable def atomic_weight_f : ℝ := 19.00

def molecular_weight_baf2 : ℝ := atomic_weight_ba + 2 * atomic_weight_f
def weight_of_3_moles_baf2 : ℝ := 3 * molecular_weight_baf2

theorem weight_of_3_moles_baf2_is_correct :
  weight_of_3_moles_baf2 = 525.99 := sorry

end weight_of_3_moles_baf2_is_correct_l757_757418


namespace find_number_l757_757431

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l757_757431


namespace island_problem_l757_757270

-- Define inhabitants A, B, C, and their respective roles
inductive Role
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Ordinary -- Can either tell the truth or lie

-- Define the inhabitant type with roles
inductive Inhabitant
| A : Role → Inhabitant
| B : Role → Inhabitant
| C : Role → Inhabitant

-- Rank according to roles
def rank : Role → ℕ
| Role.Knight := 3
| Role.Ordinary := 2
| Role.Liar := 1

-- Statements made by A and B
def statement_A (B_role C_role : Role) : Prop :=
  rank B_role > rank C_role

def statement_B (A_role C_role : Role) : Prop :=
  rank C_role > rank A_role

-- The proof problem related to C's indication
theorem island_problem (A_role B_role C_role : Role)
  (hA : Inhabitant.A A_role)
  (hB : Inhabitant.B B_role)
  (hC : Inhabitant.C C_role)
  (hA_statement : (A_role = Role.Knight ∧ statement_A B_role C_role) ∨ (A_role = Role.Ordinary ∨ A_role = Role.Liar ∧ ¬ statement_A B_role C_role))
  (hB_statement : (B_role = Role.Knight ∧ statement_B A_role C_role) ∨ (B_role = Role.Ordinary ∨ B_role = Role.Liar ∧ ¬ statement_B A_role C_role))
  : rank B_role > rank A_role :=
sorry

end island_problem_l757_757270


namespace arithmetic_sequence_sum_l757_757839

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l757_757839


namespace midpoint_on_hyperbola_l757_757049

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757049


namespace largest_subset_size_l757_757725

theorem largest_subset_size :
  ∃ S : set ℕ, S ⊆ {n | 1 ≤ n ∧ n ≤ 999} ∧
              (∀ x ∈ S, ∀ y ∈ S, x ≠ y → (x - y) % 3 ≠ 0 ∧ (x - y) % 6 ≠ 0) ∧
              (∃ k, set.card S = k) ∧ k = 416 :=
by sorry

end largest_subset_size_l757_757725


namespace person_age_in_1954_l757_757779

theorem person_age_in_1954 
  (x : ℤ)
  (cond1 : ∃ k1 : ℤ, 7 * x = 13 * k1 + 11)
  (cond2 : ∃ k2 : ℤ, 13 * x = 11 * k2 + 7)
  (input_year : ℤ) :
  input_year = 1954 → x = 1868 → input_year - x = 86 :=
by
  sorry

end person_age_in_1954_l757_757779


namespace different_integers_sum_of_three_distinct_members_l757_757946

open Set

theorem different_integers_sum_of_three_distinct_members :
  let S := {2, 5, 8, 11, 14, 17, 20}
  (∀ sums, (∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ sums = x + y + z) → sums ∈ Ico 15 52 ∧ sums % 3 = 0) →
  ∃ n, n = 13 :=
by
  let S := {2, 5, 8, 11, 14, 17, 20}
  intros h
  have h1 : ∃ sums, (∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ sums = x + y + z) :=
    sorry
  have h2 : ∀ sums, (∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ sums = x + y + z) → sums ∈ Ico 15 52 ∧ sums % 3 = 0 :=
    sorry
  have h3 : {sums | ∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ sums = x + y + z} = (Ico 15 52).filter (λ x, x % 3 = 0) :=
    sorry
  have h4 : ∃! sums, sums ∈ (Ico 15 52).filter (λ x, x % 3 = 0) :=
    sorry
  exact ⟨13, sorry⟩

end different_integers_sum_of_three_distinct_members_l757_757946


namespace avg_calculation_l757_757781

-- Define averages
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 0 2) 0 = 7 / 9 :=
  by
    sorry

end avg_calculation_l757_757781


namespace solve_vaccination_l757_757824

variable (A B : ℕ → ℕ) -- A functions for people vaccinated per day at points A and B
variable (m : ℕ) -- m value to be determined

-- Condition: A is 1.25 times B
def cond1 (Bavg : ℕ) : Prop := ∃ (Aavg : ℕ), Aavg = 5 * Bavg / 4

-- Condition: Time for point A to vaccinate 3000 people is 2 days less than time for point B to vaccinate 4000 people
def cond2 (Aavg Bavg : ℕ) : Prop := 3000 / Aavg = 4000 / Bavg - 2

-- Condition: Point B increases rate by 25%
def cond3 (Bavg : ℕ) : Prop := ∃ (Bnew : ℕ), Bnew = 5 * Bavg / 4

-- Condition: Point A decreases rate by 5m, but not less than 800 people.
def cond4 (Aavg : ℕ) : Prop := Aavg - 5 * m ≥ 800

-- Condition: B for (m + 15) days is 6000 more than A for (2m) days
def cond5 (Aavg Bnew : ℕ) : Prop := Bnew * (m + 15) = Aavg * 2 * m + 6000

theorem solve_vaccination (Bavg Aavg : ℕ) (Bnew : ℕ) :
  cond1 Bavg →
  cond2 Aavg Bavg →
  cond3 Bavg →
  cond4 Aavg →
  cond5 Aavg Bnew →
  Aavg = 1000 ∧ Bavg = 800 ∧ m = 10 :=
by {
  intros,
  sorry -- Proof details go here. 
}

end solve_vaccination_l757_757824


namespace thomas_final_amount_l757_757750

variables (michael_initial thomas_initial : ℝ)
variable michael_to_thomas : ℝ := 0.35 * michael_initial
variable thomas_after_candy : ℝ := (thomas_initial + michael_to_thomas) - 5
variable thomas_spent_on_books : ℝ := 0.25 * thomas_after_candy
noncomputable def thomas_final : ℝ := thomas_after_candy - thomas_spent_on_books

theorem thomas_final_amount : 
  michael_initial = 42 → thomas_initial = 17 → thomas_final = 20.02 :=
by
  intro h1 h2
  rw [h1, h2]
  have michael_to_thomas_value : michael_to_thomas = 0.35 * 42 := by simp [michael_to_thomas, h1]
  have thomas_after_candy_value : thomas_after_candy = (17 + 0.35 * 42) - 5 := by simp [thomas_after_candy, michael_to_thomas_value, h2]
  have thomas_spent_on_books_value : thomas_spent_on_books = 0.25 * ((17 + 0.35 * 42) - 5) := by simp [thomas_spent_on_books, thomas_after_candy_value]
  have thomas_final_calculated : thomas_final = ((17 + 0.35 * 42) - 5) - 0.25 * ((17 + 0.35 * 42) - 5) := 
    by simp [thomas_final, thomas_after_candy_value, thomas_spent_on_books_value]
  simp [thomas_final_calculated]
  -- Final calculation
  sorry

end thomas_final_amount_l757_757750


namespace can_be_midpoint_of_AB_l757_757181

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757181


namespace log_5_of_625_l757_757468

theorem log_5_of_625 : log 5 625 = 4 :=
by
  have h1: 625 = 5^4 := by norm_num
  rw [h1, log_pow]
  norm_num
  sorry

end log_5_of_625_l757_757468


namespace bridge_length_l757_757896

noncomputable def length_of_bridge (train_length speed time : ℝ) : ℝ :=
let distance := speed * time in
distance - train_length

theorem bridge_length (train_length speed time : ℝ) (h_train_length : train_length = 100)
  (h_speed : speed = 20) (h_time : time = 12.499) :
  length_of_bridge train_length speed time = 149.98 :=
by
  rw [length_of_bridge, h_train_length, h_speed, h_time]
  norm_num
  sorry

end bridge_length_l757_757896


namespace tan_minus_405_eq_neg1_l757_757936

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l757_757936


namespace b_minus_a_eq_two_l757_757799

theorem b_minus_a_eq_two (a b : ℤ) (h1 : b = 7) (h2 : a * b = 2 * (a + b) + 11) : b - a = 2 :=
by
  sorry

end b_minus_a_eq_two_l757_757799


namespace angle_between_vector_and_plane_l757_757572

theorem angle_between_vector_and_plane :
  let AB := ⟨0, 1, -1⟩
  let BE := ⟨2, -1, 2⟩
  ∃ θ : ℝ, θ = Real.pi / 4 ∧ 
    ∀ v : ℝ × ℝ × ℝ, v = AB → 
    ∀ n : ℝ × ℝ × ℝ, n = BE → 
    n ≠ 0 ∧ n ≠ v ∧ 
    BE ⟂ (vector one of the plane BCD) → θ = Real.pi / 4 := 
begin
  sorry
end

end angle_between_vector_and_plane_l757_757572


namespace Rikki_earnings_l757_757347

theorem Rikki_earnings
  (price_per_word : ℝ := 0.01)
  (words_per_5_minutes : ℕ := 25)
  (total_minutes : ℕ := 120)
  (earning : ℝ := 6)
  : price_per_word * (words_per_5_minutes * (total_minutes / 5)) = earning := by
  sorry

end Rikki_earnings_l757_757347


namespace cheaper_books_than_exactly_l757_757266

-- Define the cost function C(n) as per the given piecewise conditions.
def C (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 15 then 15 * n + 20
  else if n ≥ 16 ∧ n ≤ 30 then 13 * n
  else if n ≥ 31 ∧ n ≤ 45 then 11 * n + 50
  else 9 * n

-- Proposition to prove: There are exactly 4 values of n where buying more than n books is cheaper.
theorem cheaper_books_than_exactly (n: ℕ) : (finset.range 46).filter (λ n, C (n+1) < C n)).card = 4 :=
sorry

end cheaper_books_than_exactly_l757_757266


namespace perfect_cube_probability_l757_757893

theorem perfect_cube_probability :
  ∃ p q : ℕ, p + q = 288 ∧ Nat.Coprime p q ∧ 
  ∃ (P : ℚ), P = (p : ℚ) / (q : ℚ) ∧ 
  (∀ (dices : Fin 5 → Fin 6), 
    let product := ∏ i, dices i + 1 
    in (Nat.isCube product) ↔ ((dices 0 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 1 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 2 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 3 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 4 = ⟨5, Nat.lt_succ_self 5⟩) ∨ 
                            (natMod (product) 15 = 0))) :=
sorry

end perfect_cube_probability_l757_757893


namespace sum_vertices_not_equal_sum_edges_l757_757913

theorem sum_vertices_not_equal_sum_edges (a : Fin 8 → ℕ) (h_diff : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∑ i, a i ≠ ∑ (i : Fin 12), Nat.gcd (a (cube_edge_vertices i).1) (a (cube_edge_vertices i).2) := 
sorry

end sum_vertices_not_equal_sum_edges_l757_757913


namespace splay_sequence_problem_l757_757797

def is_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

def set_splay_sequences : Set (List ℝ) :=
  {C | ∀ c ∈ C, 0 < c ∧ c < 1 }

def power (C : List ℝ) : ℝ :=
  C.foldr (· * ·) 1  -- product of the elements in the list

def sum_powers (S : Set (List ℝ)) : ℚ :=
  ⟨S.toList.map power).sum, 1⟩ -- Q it to rational as \frac{sum}{1}

theorem splay_sequence_problem :
  ∃ (m n : ℕ), is_relatively_prime m n ∧ sum_powers set_splay_sequences = ⟨m, n⟩ ∧ 100 * m + n = 4817 :=
by
  sorry

end splay_sequence_problem_l757_757797


namespace midpoint_of_hyperbola_l757_757067

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757067


namespace range_of_a_l757_757622

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem range_of_a (f_def : ∀ x ≥ 1, f x ≥ a * x - 1) : a ≤ 1 :=
by
  have h := fun x hx => f_def x hx
  -- Remaining steps and doodles skipped
  sorry

end range_of_a_l757_757622


namespace flight_time_calc_l757_757907

theorem flight_time_calc :
  let reading_time := 2
  let movie_time := 4
  let eating_time := 0.5
  let listening_time := (2 / 3 : Real)
  let playing_time := (1 + 1 / 6 : Real)
  let nap_time := 3
  reading_time + movie_time + eating_time + listening_time + playing_time + nap_time = 11.34 :=
by
  -- Definitions
  let reading_time := 2
  let movie_time := 4
  let eating_time := 0.5
  let listening_time := (2 / 3 : Real)
  let playing_time := (1 + 1 / 6 : Real)
  let nap_time := 3

  -- Proof
  have h : reading_time + movie_time + eating_time + listening_time + playing_time + nap_time = 11.34 :=
    sorry

  exact h

end flight_time_calc_l757_757907


namespace chord_angle_interior_exterior_l757_757463

theorem chord_angle_interior_exterior {A B O O' : Point} {C : Circle} :
  is_chord C A B →
  is_point_in_segment C A B O →
  is_point_outside_circle_same_side C A B O' →
  (angle A O B > inscribed_angle A B) ∧ (angle A O' B < inscribed_angle A B) :=
by 
  sorry

end chord_angle_interior_exterior_l757_757463


namespace all_students_have_white_hats_l757_757860

theorem all_students_have_white_hats
  (students : Fin 3)
  (initial_hats : students → ℕ → Type)
  (num_hats : ∀ (h : Type), h = Fin 5 ∧ (h 0 = 3) ∧ (h 1 = 2))
  (observed_hats : students → ℕ → Type)
  (conditions : ∀ (i j : students), i ≠ j → observed_hats i j = initial_hats j) 
  (no_communication : ∀ (i j : students), i ≠ j → (observed_hats i j) ≠ (observed_hats j i))
  (eventually_conclude : ∀ (i : students), ∃ (color : Fin 3), initial_hats i color = 0)
  : ∀ (i : students), initial_hats i 0 = 0 :=
by
  sorry

end all_students_have_white_hats_l757_757860


namespace inequality_inequality_positive_integers_sum_pow_l757_757594

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {k : ℝ}

theorem inequality_inequality_positive_integers_sum_pow:
  (∀ i : Fin n, a i > 0) → 
  (∀ i j : Fin n, i < j → a i < a j) →
  (k ≥ 1) →
  ∑ i : Fin n, (a i)^(2*k + 1) ≥ (∑ i : Fin n, (a i)^k)^2 :=
by
  intros
  sorry

end inequality_inequality_positive_integers_sum_pow_l757_757594


namespace midpoint_of_hyperbola_l757_757170

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757170


namespace correct_midpoint_l757_757212

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757212


namespace products_B_correct_l757_757405

-- Define the total number of products
def total_products : ℕ := 4800

-- Define the sample size and the number of pieces from equipment A in the sample
def sample_size : ℕ := 80
def sample_A : ℕ := 50

-- Define the number of products produced by equipment A and B
def products_A : ℕ := 3000
def products_B : ℕ := total_products - products_A

-- The target number of products produced by equipment B
def target_products_B : ℕ := 1800

-- The theorem we need to prove
theorem products_B_correct :
  products_B = target_products_B := by
  sorry

end products_B_correct_l757_757405


namespace max_sum_squares_l757_757558

def arithmetic_mean (xs : List ℝ) : ℝ :=
  (xs.foldr (· + ·) 0) / (xs.length : ℝ)

theorem max_sum_squares (xs : List ℝ) (h₀ : ∀ x ∈ xs, 0 ≤ x ∧ x ≤ 1) (h₁ : xs.length = 2011) :
  let m := arithmetic_mean xs
  ∑ k in xs, (k - m) ^ 2 = (1005 * 1006) / 2011 :=
by
  sorry

end max_sum_squares_l757_757558


namespace odd_square_free_count_l757_757660

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 80 ∧ ∀ k : ℕ, (k > 1 ∧ k < 200 ∧ k % 2 = 1) → 
    (¬ ∃ a : ℕ, a > 1 ∧ a * a ∣ k) → k ∈ (1 :: List.range (200 // 2)).filter (λ x, x % 2 = 1) :=
by
  sorry

end odd_square_free_count_l757_757660


namespace molly_jake_divisor_count_l757_757752

theorem molly_jake_divisor_count : (setOf (λ x : ℕ, 1 ≤ x ∧ x ≤ 720 ∧ 720 % x = 0)).card = 30 := 
by
  -- skipping the actual proof
  sorry

end molly_jake_divisor_count_l757_757752


namespace midpoint_on_hyperbola_l757_757083

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757083


namespace hyperbola_midpoint_l757_757192

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757192


namespace sum_x_eq_one_l757_757923

variable {n : ℕ} (h_pos : 0 < n)
noncomputable def x : ℕ → ℚ
| 0       := 1 / n
| (j + 1) := 1 / (n - 1) * ∑ i in Finset.range (j + 1), x i

theorem sum_x_eq_one : ∑ j in Finset.range n, x j = 1 := sorry

end sum_x_eq_one_l757_757923


namespace water_for_chickens_l757_757924

theorem water_for_chickens : 
  ∃ (pigs horses pig_water horse_water total_water chicken_water: ℕ), 
    pigs = 8 ∧ 
    horses = 10 ∧ 
    pig_water = 3 ∧ 
    horse_water = 2 * pig_water ∧ 
    total_water = 114 ∧
    (total_water - (pigs * pig_water + horses * horse_water)) = chicken_water ∧
    chicken_water = 30 :=
by
  use [8, 10, 3, 6, 114, 30]
  unfold total_water pigs pig_water horses horse_water chicken_water
  sorry

end water_for_chickens_l757_757924


namespace g_of_f_of_3_eq_1902_l757_757253

def f (x : ℕ) := x^3 - 2
def g (x : ℕ) := 3 * x^2 + x + 2

theorem g_of_f_of_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_of_3_eq_1902_l757_757253


namespace tan_neg_405_eq_neg_1_l757_757929

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l757_757929


namespace apples_picked_l757_757267

theorem apples_picked (n_a : ℕ) (k_a : ℕ) (total : ℕ) (m_a : ℕ) (h_n : n_a = 3) (h_k : k_a = 6) (h_t : total = 16) :
  m_a = total - (n_a + k_a) →
  m_a = 7 :=
by
  sorry

end apples_picked_l757_757267


namespace minimum_n_for_80_intersections_l757_757787

-- Define what an n-sided polygon is and define the intersection condition
def n_sided_polygon (n : ℕ) : Type := sorry -- definition of n-sided polygon

-- Define the condition when boundaries of two polygons intersect at exactly 80 points
def boundaries_intersect_at (P Q : n_sided_polygon n) (k : ℕ) : Prop := sorry -- definition of boundaries intersecting at exactly k points

theorem minimum_n_for_80_intersections (n : ℕ) :
  (∃ (P Q : n_sided_polygon n), boundaries_intersect_at P Q 80) → (n ≥ 10) :=
sorry

end minimum_n_for_80_intersections_l757_757787


namespace arithmetic_sequence_sum_l757_757842

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l757_757842


namespace quadratic_no_real_roots_l757_757845

theorem quadratic_no_real_roots :
  ∀ (a b c : ℝ), (a = 1 ∧ b = 1 ∧ c = 2) → (b^2 - 4 * a * c < 0) := by
  intros a b c H
  cases H with Ha Hac
  cases Hac with Hb Hc
  rw [Ha, Hb, Hc]
  simp
  linarith

end quadratic_no_real_roots_l757_757845


namespace midpoint_of_line_segment_on_hyperbola_l757_757036

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757036


namespace tan_neg405_deg_l757_757938

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l757_757938


namespace triangle_AC_open_interval_l757_757351

theorem triangle_AC_open_interval (AB AC BC : ℝ) (BD CD : ℝ) :
  AB = 12 →
  CD = 4 →
  (∃ m n : ℝ, (∀ AC x : ℝ, 4 < x ∧ x < 24) ∧ m + n = 28) :=
by
    intros hAB hCD
    use [4, 24]
    split
    intro x hx
    exact sorry
    norm_num
    exact hAB
    exact hCD
    sorry

end triangle_AC_open_interval_l757_757351


namespace complement_intersection_l757_757636

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  (U \ A) ∩ B = {0} :=
  by
    sorry

end complement_intersection_l757_757636


namespace sadies_average_speed_l757_757349

def sadie_time : ℝ := 2
def ariana_speed : ℝ := 6
def ariana_time : ℝ := 0.5
def sarah_speed : ℝ := 4
def total_time : ℝ := 4.5
def total_distance : ℝ := 17

theorem sadies_average_speed :
  ((total_distance - ((ariana_speed * ariana_time) + (sarah_speed * (total_time - sadie_time - ariana_time)))) / sadie_time) = 3 := 
by sorry

end sadies_average_speed_l757_757349


namespace minimum_xyz_minimum_abcd_l757_757863

-- Problem 1 statement
theorem minimum_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 1) :
  ∃ m, (m = 1) ∧ (∀ xy_value, (xy_value = (x * y / z) + (y * z / x) + (z * x / y)) → xy_value ≥ m) :=
begin
  use 1,
  intros xy_value hx,
  sorry
end

-- Problem 2 statement
theorem minimum_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hd : 0 ≤ d) (hb : 0 < b) (hc : 0 < c) (h : b + c ≥ a + d) :
  ∃ m, (m = (Real.sqrt 2 - 1 / 2)) ∧ (∀ abcd_value, (abcd_value = (b / (b + d)) + (c / (a + b))) → abcd_value ≥ m) :=
begin
  use (Real.sqrt 2 - 1 / 2),
  intros abcd_value hb,
  sorry
end

end minimum_xyz_minimum_abcd_l757_757863


namespace distance_between_points_l757_757883

theorem distance_between_points :
  ∀ (D : ℝ), (10 + 2) * (5 / D) + (10 - 2) * (5 / D) = 24 ↔ D = 24 := 
sorry

end distance_between_points_l757_757883


namespace hyperbola_midpoint_exists_l757_757244

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757244


namespace hyperbola_midpoint_l757_757124

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757124


namespace proof_problem_l757_757738

def T := { x : ℝ // x ≠ 0 }  -- Define T as nonzero real numbers

def g (x : T) : T  
noncomputable def h : T → ℝ
noncomputable def possible_values_g1 : Set ℝ 
noncomputable def n : ℝ := ↑possible_values_g1.size
noncomputable def s : ℝ := possible_values_g1.sum id

axiom axiom1 (x : T) : g (⟨1 / x.val, by apply inst_nontrivial; exact x.prop⟩) = 3 * x.val * g x
axiom axiom2 (x y : T) (h : x.val + y.val ≠ 0) : g (⟨1 / x.val, by apply inst_nontrivial; exact x.prop⟩) + g (⟨1 / y.val, by apply inst_nontrivial; exact y.prop⟩) = 3 + g (⟨1 / (x.val + y.val), by apply inst_nontrivial; exact h⟩)

theorem proof_problem : n * s = 3 := sorry

end proof_problem_l757_757738


namespace midpoint_hyperbola_l757_757222

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757222


namespace part_a_part_b_part_c_l757_757274

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l757_757274


namespace hyperbola_midpoint_exists_l757_757245

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757245


namespace midpoint_of_line_segment_on_hyperbola_l757_757030

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757030


namespace petya_five_ruble_coins_l757_757303

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l757_757303


namespace find_x_l757_757427

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l757_757427


namespace midpoint_on_hyperbola_l757_757131

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757131


namespace proof_problem_l757_757592

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom h1 : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom h2 : g 2012 = a
axiom h3 : a > 0
axiom h4 : a ≠ 1
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

theorem proof_problem : f (-2012) = 2^(-2012) - 2^2012 := sorry

end proof_problem_l757_757592


namespace equation_of_ellipse_equation_of_line_l757_757590

-- Proof 1: Finding the equation of the ellipse
theorem equation_of_ellipse (a b : ℝ) (h1 : a > b ∧ b > 0) (e : ℝ) (h2 : e = (real.sqrt 3) / 2) (f : ℝ) (h3 : 2 * f = 2 * real.sqrt 3) :
  (a = 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 1) = 1) :=
by
  sorry

-- Proof 2: Finding the equation of the line
theorem equation_of_line (m : ℝ) (h1 : 1 < m ∧ m < real.sqrt 2) (h2 : m = 3 * (real.sqrt 5) / 5) :
  (∀ x y : ℝ, 5 * x + 10 * y - 6 * real.sqrt 5 = 0) :=
by
  sorry

end equation_of_ellipse_equation_of_line_l757_757590


namespace larger_integer_value_l757_757392

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l757_757392


namespace polynomial_sum_ineq_l757_757718

open Polynomial

theorem polynomial_sum_ineq {P : Polynomial ℝ} (nonneg_coeffs : ∀ n, 0 ≤ P.coeff n) 
  (k : ℕ) (hk : 0 < k) (x : Fin k → ℝ) (hx : ∏ i in Finset.univ, x i = 1) : 
  ∑ i in Finset.univ, P.eval (x i) ≥ k * P.eval 1 :=
sorry

end polynomial_sum_ineq_l757_757718


namespace part1_part2_part3_part4_part5_l757_757810

-- Given total students, boys, girls and their arrangement conditions, prove the various arrangement counts.
def total_students := 7
def boys := 4
def girls := 3
def arrangements := λ (students : Nat) => students.factorial
def student_A_not_front := total_students - 1  -- Given student A can't be at the front, so the remaining options

-- Proof statements as per the given solutions
theorem part1 : arrangements total_students = 5040 := sorry
theorem part2 : arrangements student_A_not_front = 4320 := sorry
-- For part 3 and subsequent, add more specific conditions as necessary
axiom cannot_stand_next_to : units := sorry -- Mock additional required units
axiom girl_ordering : units := sorry       -- Mock additional required units

theorem part3 : arrangements total_students = 1440 := sorry
theorem part4 : arrangements total_students = 840 := sorry
theorem part5 : arrangements total_students = 720 := sorry

end part1_part2_part3_part4_part5_l757_757810


namespace sum_odd_integers_1_to_49_l757_757421

theorem sum_odd_integers_1_to_49 : ∑ k in Finset.range 25, (2 * k + 1) = 625 := by
  sorry

end sum_odd_integers_1_to_49_l757_757421


namespace midpoint_on_hyperbola_l757_757129

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757129


namespace inequality_solution_interval_l757_757772

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l757_757772


namespace midpoint_on_hyperbola_l757_757070

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757070


namespace complex_division_l757_757616

open Complex

theorem complex_division :
  let z : ℂ := 1 - I
  in (z^2) / (z - 1) = 2 := by
  sorry

end complex_division_l757_757616


namespace cone_slice_volume_ratio_l757_757495

theorem cone_slice_volume_ratio
  (r h : ℝ) -- r is the radius, h is the height of the smallest cone
  (V : ℕ → ℝ) -- V is a function from number of pieces to volume
  (V_1 : ℝ) -- volume of the largest piece
  (V_2 : ℝ) -- volume of the second-largest piece
  (h_r_base : ∀ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) -- Slice considerations
  (slice_height : ∀ n : ℕ, n = 1 → V n = 1/3 * π * r^2 * h * (if n = 1 then 1 else n^3)) -- heights and volumes
  (largest_piece_vol : V_1 = 125/3 * π * r^2 * h - 64/3 * π * r^2 * h)
  (second_largest_piece_vol : V_2 = 64/3 * π * r^2 * h - 27/3 * π * r^2 * h)
  :
  V_2 / V_1 = 37 / 61 :=
by
  sorry

end cone_slice_volume_ratio_l757_757495


namespace find_a_l757_757254

theorem find_a (a : ℂ) (h : a / (1 - I) = (1 + I) / I) : a = -2 * I := 
by
  sorry

end find_a_l757_757254


namespace arrange_polynomial_l757_757908

theorem arrange_polynomial :
  ∀ (x y : ℝ), 2 * x^3 * y - 4 * y^2 + 5 * x^2 = 5 * x^2 + 2 * x^3 * y - 4 * y^2 :=
by
  sorry

end arrange_polynomial_l757_757908


namespace tangent_line_equation_at_point_l757_757976

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem tangent_line_equation_at_point (x y : ℝ) (h : y = curve x) 
    (hx : 2) (hy : 5) (hpt : y = 5 ∧ x = 2) : 7 * x - y - 9 = 0 :=
by
  sorry

end tangent_line_equation_at_point_l757_757976


namespace color_tv_cost_l757_757894

theorem color_tv_cost (x : ℝ) (y : ℝ) (z : ℝ)
  (h1 : y = x * 1.4)
  (h2 : z = y * 0.8)
  (h3 : z = 360 + x) :
  x = 3000 :=
sorry

end color_tv_cost_l757_757894


namespace range_of_vertical_coordinate_l757_757584

/-- If P is a point on the parabola y^2 = x, Q is another point on the parabola,
    and BP ⊥ PQ with B(1,1), then the vertical coordinate s of point Q(s^2, s)
    lies in the range (-∞, -1] ∪ [3, +∞). -/
theorem range_of_vertical_coordinate (t s : ℝ) (hP : t^2 = t)
  (hQ : s^2 = s) (B : ℝ × ℝ) (BP_PQ_orthogonal : (1 - t^2) * (s^2 - t^2) + (1 - t) * (s - t) = 0) :
    s ∈ set.Iic (-1) ∪ set.Ici 3 := 
sorry

end range_of_vertical_coordinate_l757_757584


namespace tangent_line_eq_l757_757978

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end tangent_line_eq_l757_757978


namespace hyperbola_midpoint_l757_757150

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757150


namespace second_car_mileage_l757_757515

theorem second_car_mileage (x : ℝ) : 
  (150 / 50) + (150 / x) + (150 / 15) = 56 / 2 → x = 10 :=
by
  intro h
  sorry

end second_car_mileage_l757_757515


namespace triangle_ratios_sum_l757_757701

theorem triangle_ratios_sum (P Q R S T : Type) [euclidean_geometry P Q R S T]
  (right_angle_at_Q : ∠ P Q R = 90)
  (PQ : dist P Q = 5)
  (QR : dist Q R = 12)
  (right_angle_at_P : ∠ P R S = 90)
  (PS: dist P S = 20)
  (Q_S_opposite_sides : ∃ M, M ∈ line PR ∧ point_on_different_sides Q S M)
  (parallel_line_through_s : ∃ T, line S T || line P Q ∧ T ∈ extended_line QR)
  (rel_prime : ∃ a b : ℕ, gcd a b = 1 ∧ ∃ (h : dist S T / dist S R = a / b), True) :
  let a := gcd_nat (dist S T) (dist S R) in
  let b := (dist S T / a) * (dist S R / a) in
  a + b = 138 :=
sorry

end triangle_ratios_sum_l757_757701


namespace total_time_pushing_car_l757_757917

theorem total_time_pushing_car :
  let d1 := 3
  let s1 := 6
  let d2 := 3
  let s2 := 3
  let d3 := 4
  let s3 := 8
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  (t1 + t2 + t3) = 2 :=
by
  sorry

end total_time_pushing_car_l757_757917


namespace circle_properties_l757_757554

structure Point :=
(x : ℝ)
(y : ℝ)

def circleThroughPoints (A B C : Point) (r_sq : ℝ) :=
(A.x - C.x)^2 + (A.y - C.y)^2 = r_sq ∧
(B.x - C.x)^2 + (B.y - C.y)^2 = r_sq

def centerOnLine (C : Point) :=
C.y = 0

def pointInsideCircle (P C : Point) (r_sq : ℝ) :=
(P.x - C.x)^2 + (P.y - C.y)^2 < r_sq

theorem circle_properties :
  ∀ (A B P C : Point) (r_sq : ℝ),
  A = Point.mk 1 4 →
  B = Point.mk 3 2 →
  C = Point.mk 4.5 0 →
  r_sq = 28.25 →
  circleThroughPoints A B C r_sq →
  centerOnLine C →
  ((P = Point.mk 2 4) → pointInsideCircle P C r_sq) :=
by
  intros A B P C r_sq hA hB hC hr hCircle hCenter hP
  rw [hA, hB, hC, hr, hP]
  -- expected proof steps
  sorry

end circle_properties_l757_757554


namespace puzzle_piece_total_l757_757009

theorem puzzle_piece_total :
  let p1 := 1000
  let p2 := p1 + 0.30 * p1
  let p3 := 2 * p2
  let p4 := (p1 + p3) + 0.50 * (p1 + p3)
  let p5 := 3 * p4
  let p6 := p1 + p2 + p3 + p4 + p5
  p1 + p2 + p3 + p4 + p5 + p6 = 55000
:= sorry

end puzzle_piece_total_l757_757009


namespace can_be_midpoint_of_AB_l757_757177

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757177


namespace area_GIM_eq_22_5_l757_757000

noncomputable def area_of_triangle (A B C : Point) : ℝ := sorry

structure Triangle :=
  (A B C : Point)

namespace Triangle

def G (T : Triangle) : Point := midpoint T.A T.B
def H (T : Triangle) : Point := midpoint T.A T.C
def I (T : Triangle) : Point := midpoint (G T) (H T)
def M (T : Triangle) : Point := midpoint T.B T.C

noncomputable def area_ABC_eq_180 (T : Triangle) (h : area_of_triangle T.A T.B T.C = 180) :=
  true

theorem area_GIM_eq_22_5 (T : Triangle) (h : area_of_triangle T.A T.B T.C = 180) : 
  area_of_triangle (G T) (I T) (M T) = 22.5 := sorry

end Triangle

end area_GIM_eq_22_5_l757_757000


namespace gcd_60_90_150_l757_757829

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l757_757829


namespace midpoint_of_hyperbola_l757_757160

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757160


namespace min_value_frac_inv_l757_757251

theorem min_value_frac_inv (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: a + 3 * b = 2) : 
  (2 + Real.sqrt 3) ≤ (1 / a + 1 / b) :=
sorry

end min_value_frac_inv_l757_757251


namespace midpoint_of_hyperbola_segment_l757_757098

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757098


namespace equivalent_single_discount_l757_757486

theorem equivalent_single_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  original_price = 50 → 
  first_discount = 0.3 → 
  second_discount = 0.2 →
  (original_price * (1 - first_discount) * (1 - second_discount) = original_price * (1 - 0.44)) :=
begin
  intros h_original h_first h_second,
  rw [h_original, h_first, h_second],
  -- Further steps to prove it, but we can skip proof by sorry
  sorry
end

end equivalent_single_discount_l757_757486


namespace find_angle_B_find_area_l757_757705

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def triangle_conditions (A B C a b c : ℝ) :=
  2 * b * Real.sin A = Real.sqrt 3 * a * Real.cos B + a * Real.sin B ∧
  b = Real.sqrt 13 ∧
  a + c = 5

-- Showing that the measure of angle B is π/3
theorem find_angle_B (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) : B = π / 3 :=
sorry

-- Showing that the area of triangle ABC is √3
theorem find_area (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) : 
  let s := 1/2 * a * c * Real.sin B in
  s = Real.sqrt 3 :=
sorry

end find_angle_B_find_area_l757_757705


namespace polynomial_not_33_l757_757760

theorem polynomial_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end polynomial_not_33_l757_757760


namespace petya_five_ruble_coins_count_l757_757324

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757324


namespace cost_per_semester_correct_l757_757011

variable (cost_per_semester total_cost : ℕ)
variable (years semesters_per_year : ℕ)

theorem cost_per_semester_correct :
    years = 13 →
    semesters_per_year = 2 →
    total_cost = 520000 →
    cost_per_semester = total_cost / (years * semesters_per_year) →
    cost_per_semester = 20000 := by
  sorry

end cost_per_semester_correct_l757_757011


namespace hyperbola_midpoint_l757_757151

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757151


namespace midpoint_hyperbola_l757_757221

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757221


namespace inequality_solution_l757_757769

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l757_757769


namespace midpoint_on_hyperbola_l757_757132

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757132


namespace petya_five_ruble_coins_count_l757_757323

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757323


namespace fraction_simplification_l757_757942

theorem fraction_simplification:
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 :=
by {
  -- Proof goes here
  sorry
}

end fraction_simplification_l757_757942


namespace study_time_difference_l757_757017

def Kwame_study_hours := 2.5
def Connor_study_hours := 1.5
def Lexia_study_minutes := 97

def hours_to_minutes (h : ℕ) : ℕ := h * 60

theorem study_time_difference :
  hours_to_minutes Kwame_study_hours + hours_to_minutes Connor_study_hours - Lexia_study_minutes = 143 :=
sorry

end study_time_difference_l757_757017


namespace midpoint_of_hyperbola_segment_l757_757084

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757084


namespace spot_area_l757_757780

/-- Proving the area of the accessible region outside the doghouse -/
theorem spot_area
  (pentagon_side : ℝ)
  (rope_length : ℝ)
  (accessible_area : ℝ) 
  (h1 : pentagon_side = 1) 
  (h2 : rope_length = 3)
  (h3 : accessible_area = (37 * π) / 5) :
  accessible_area = (π * (rope_length^2) * (288 / 360)) + 2 * (π * (pentagon_side^2) * (36 / 360)) := 
  sorry

end spot_area_l757_757780


namespace trajectory_of_P_range_of_S_div_k_l757_757692

theorem trajectory_of_P (x y : ℝ) :
  (∃ r : ℝ, (x + 1)^2 + y^2 = (r + 1)^2 ∧ (x - 1)^2 + y^2 = (3 - r)^2) →
  (x ≠ -2) →
  (x^2 / 4 + y^2 / 3 = 1) :=
sorry

theorem range_of_S_div_k (k : ℝ) (S : ℝ) :
  (k > 0) →
  (∃ M N : ℝ × ℝ, ((M.1 - (-2)) * (N.1 - (-2)) = (16 * k^2 - 12) / (3 + 4 * k^2)) ∧
                   (|M.1 - (-2)| * sqrt(1 + k^2) = sqrt(1 + k^2) * 12 / (3 + 4 * k^2)) ∧
                   (|N.1 - (-2)| * sqrt(1 + 1 / k^2) = sqrt(1 + 1 / k^2) * 12 * k^2 / (3 * k^2 + 4))) →
  0 < S / k ∧ S / k < 6 :=
sorry

end trajectory_of_P_range_of_S_div_k_l757_757692


namespace triangle_area_ratio_l757_757706

theorem triangle_area_ratio (A B C X : Type) (h : ∠ACB = 2 * ∠XCB) 
  (BC AC : ℝ) 
  (hBC : BC = 33) 
  (hAC : AC = 45)
  : ∃ (BX AX : ℝ), BX / AX = 11 / 15 ∧ (area (triangle B C X) / area (triangle A C X) = 11 / 15) :=
by
  have h_ratio : BX / AX = BC / AC := by sorry
  have h_area_ratio : area (triangle B C X) / area (triangle A C X) = BX / AX := by sorry
  have final_ratio : area (triangle B C X) / area (triangle A C X) = 11 / 15 := by sorry
  exists BX, AX, h_ratio, final_ratio
  sorry

end triangle_area_ratio_l757_757706


namespace midpoint_of_line_segment_on_hyperbola_l757_757025

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757025


namespace pyramid_volume_l757_757530

-- Define the key conditions for pyramid P-ABCD
def is_square_base (A B C D : Point) (AB : ℝ) := 
  AB = 2 ∧ dist A B = AB ∧ dist B C = AB ∧ dist C D = AB ∧ dist D A = AB ∧ 
  dist A C = dist B D ∧ is_diag_dist congruent

def is_vertex (P : Point) (A B C D : Point) := 
  dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D

def angle_APB (P A B : Point) := 
  angle P A B = π/3

-- Pyramid Volume proof context
theorem pyramid_volume (A B C D P : Point) (AB : ℝ) (Volume : ℝ):
  is_square_base A B C D AB →
  is_vertex P A B C D →
  angle_APB P A B →
  Volume = (1:ℝ) / 3 * (AB * AB) * 2 → 
  Volume = 8 / 3 :=
by
  intros h_base h_vertex h_angle h_volume
  sorry

end pyramid_volume_l757_757530


namespace trapezoid_two_heights_l757_757897

-- Define trivially what a trapezoid is, in terms of having two parallel sides.
structure Trapezoid :=
(base1 base2 : ℝ)
(height1 height2 : ℝ)
(has_two_heights : height1 = height2)

theorem trapezoid_two_heights (T : Trapezoid) : ∃ h1 h2 : ℝ, h1 = h2 :=
by
  use T.height1
  use T.height2
  exact T.has_two_heights

end trapezoid_two_heights_l757_757897


namespace division_from_multiplication_l757_757914

theorem division_from_multiplication (a b c : ℕ) : a * b = c → c / a = b ∧ c / b = a :=
by
  assume h : a * b = c
  have : c / a = b := sorry
  have : c / b = a := sorry
  exact ⟨this, this⟩

end division_from_multiplication_l757_757914


namespace linda_total_profit_is_50_l757_757746

def total_loaves : ℕ := 60
def loaves_sold_morning (total_loaves : ℕ) : ℕ := total_loaves / 3
def loaves_sold_afternoon (loaves_left_morning : ℕ) : ℕ := loaves_left_morning / 2
def loaves_sold_evening (loaves_left_afternoon : ℕ) : ℕ := loaves_left_afternoon

def price_per_loaf_morning : ℕ := 3
def price_per_loaf_afternoon : ℕ := 150 / 100 -- Representing $1.50 as 150 cents to use integer arithmetic
def price_per_loaf_evening : ℕ := 1

def cost_per_loaf : ℕ := 1

def calculate_profit (total_loaves loaves_sold_morning loaves_sold_afternoon loaves_sold_evening price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf : ℕ) : ℕ := 
  let revenue_morning := loaves_sold_morning * price_per_loaf_morning
  let loaves_left_morning := total_loaves - loaves_sold_morning
  let revenue_afternoon := loaves_sold_afternoon * price_per_loaf_afternoon
  let loaves_left_afternoon := loaves_left_morning - loaves_sold_afternoon
  let revenue_evening := loaves_sold_evening * price_per_loaf_evening
  let total_revenue := revenue_morning + revenue_afternoon + revenue_evening
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

theorem linda_total_profit_is_50 : calculate_profit total_loaves (loaves_sold_morning total_loaves) (loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) (total_loaves - loaves_sold_morning total_loaves - loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf = 50 := 
  by 
    sorry

end linda_total_profit_is_50_l757_757746


namespace rectangular_prism_vertex_labelling_l757_757715

open Function

theorem rectangular_prism_vertex_labelling :
  let vertices := {4, 5, 6, 7, 8, 9, 10, 11}
  -- Sum of all vertices
  ∑ (v : ℕ) in vertices, v = 60 
  -- Number of short faces and long faces
  -- Given that sums of short faces are equal and sums of long faces are equal
  -- S_short represents the sum of vertices for a short face
  -- S_long represents the sum of vertices for a long face
  -- Find S_short and S_long such that:
  -- 2 * S_short + 4 * S_long = 180
  let S_short := 46
  let S_long := 35
  -- Given all constraints, prove there are exactly 3 distinct labellings (considering rotations as the same)
  ∃ arrangements : Finset (Finset (Fin 8)), arrangements.card = 3
  := by
    sorry

end rectangular_prism_vertex_labelling_l757_757715


namespace count_monomials_l757_757698

def is_monomial (expr : ℕ → bool) : Prop :=
  ∀ n, expr n → ∃ c : ℚ, ∃ v : list (ℕ), ∃ p : list (ℕ), 
    v.length = p.length ∧ (∀ i ∈ list.finRange v.length, p[i] ≥ 0)

def expr1 : ℕ → bool := λ n, n = 1
def expr2 : ℕ → bool := λ n, n = 2
def expr3 : ℕ → bool := λ n, n = 3
def expr4 : ℕ → bool := λ n, n = 4
def expr5 : ℕ → bool := λ n, n = 5
def expr6 : ℕ → bool := λ n, n = 6
def expr7 : ℕ → bool := λ n, n = 7
def expr8 : ℕ → bool := λ n, n = 8

theorem count_monomials : 
  let monomials := [expr1, expr2, expr3, expr4, expr7] in
  let non_monomials := [expr5, expr6, expr8] in
  monomials.length = 5 :=
by 
  sorry

end count_monomials_l757_757698


namespace midpoint_of_hyperbola_l757_757061

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757061


namespace midpoint_of_line_segment_on_hyperbola_l757_757037

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757037


namespace beth_wins_optimally_l757_757514

def nim_value (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 2
  | 6 => 3
  | 7 => 0
  | 8 => 1
  | _ => 0    -- Placeholder for simplicity, use actual calculations in real cases

def nim_sum (n1 n2 n3 : Nat) : Nat :=
  n1 ⊕ n2 ⊕ n3

theorem beth_wins_optimally :
  ∀ (c1 c2 c3 : Nat),
  (nim_sum (nim_value c1) (nim_value c2) (nim_value c3) = 0) ↔ (c1 = 7 ∧ c2 = 4 ∧ c3 = 4) :=
by
  intros c1 c2 c3
  apply Iff.intro 
  sorry

end beth_wins_optimally_l757_757514


namespace yehudi_ate_100_black_candies_l757_757682

theorem yehudi_ate_100_black_candies
  (total_candies : ℕ)
  (initial_black_candies : ℕ)
  (initial_gold_candies : ℕ)
  (eaten_black_candies : ℕ)
  (remaining_black_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : total_candies = 200)
  (h2 : initial_black_candies = 0.9 * total_candies)
  (h3 : initial_gold_candies = total_candies - initial_black_candies)
  (h4 : remaining_candies = total_candies - eaten_black_candies)
  (h5 : remaining_black_candies = initial_black_candies - eaten_black_candies)
  (h6 : remaining_black_candies = 0.8 * remaining_candies) :
  eaten_black_candies = 100 := sorry

end yehudi_ate_100_black_candies_l757_757682


namespace barrels_needed_l757_757404

theorem barrels_needed (oil_mass : ℕ) (barrel_capacity : ℕ) : oil_mass = 250 ∧ barrel_capacity = 40 → (⟦oil_mass / barrel_capacity⟧ + 1) = 7 := 
by 
  assume h : oil_mass = 250 ∧ barrel_capacity = 40
  sorry

end barrels_needed_l757_757404


namespace find_number_of_dogs_l757_757918

-- Definitions based on conditions
def number_of_cats := 3
def number_of_birds := 4
def baths_per_year_per_dog := 24
def baths_per_year_per_cat := 12
def baths_per_year_per_bird := 3
def total_baths_per_year := 96

-- The Lean theorem statement
theorem find_number_of_dogs 
  (C : ℕ := number_of_cats)
  (B : ℕ := number_of_birds)
  (bath_d : ℕ := baths_per_year_per_dog)
  (bath_c : ℕ := baths_per_year_per_cat)
  (bath_b : ℕ := baths_per_year_per_bird)
  (total_baths : ℕ := total_baths_per_year)
  : ∃ D : ℕ, (bath_d * D + bath_c * C + bath_b * B = total_baths) ∧ D = 2 :=
begin
  sorry
end

end find_number_of_dogs_l757_757918


namespace midpoint_on_hyperbola_l757_757072

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757072


namespace A_finishes_work_in_8_days_l757_757866

-- Let W be the total amount of work
variables {W : ℝ} -- Note: Using real numbers for calculus of work done

-- Let A and B be the amount of work A and B can do in one day respectively
variable {A B : ℝ}

-- Given conditions:
-- 1. A and B can together finish the work in 40 days.
axiom cond1 : (A + B) * 40 = W
-- 2. They worked together for 10 days.
axiom cond2 : (A + B) * 10
-- 3. After B left, A worked alone for another 6 days to finish the remaining work.
axiom cond3 : A * 6

-- Prove that A alone can finish the job in 8 days.
theorem A_finishes_work_in_8_days : (W / A) = 8 :=
by
  sorry

end A_finishes_work_in_8_days_l757_757866


namespace school_trip_l757_757892

theorem school_trip (x : ℕ) (total_students : ℕ) :
  (28 * x + 13 = total_students) ∧ (32 * x - 3 = total_students) → 
  x = 4 ∧ total_students = 125 :=
by
  sorry

end school_trip_l757_757892


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757280

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l757_757280


namespace midpoint_on_hyperbola_l757_757141

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757141


namespace number_divided_by_three_l757_757445

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l757_757445


namespace ming_dynasty_wine_problem_l757_757695

theorem ming_dynasty_wine_problem :
  ∃ x y : ℝ, x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by {
  -- Define the existence of variables x and y satisfying the conditions
  existsi (x : ℝ),
  existsi (y : ℝ),
  -- Conditions are given as premises to be satisfied
  split,
  -- First equation: x + y = 19
  exact x + y = 19,
  -- Second equation: 3x + (1/3)y = 33
  exact 3 * x + (1 / 3) * y = 33,
  -- Add placeholder to indicate where the actual proof would go
  sorry
}

end ming_dynasty_wine_problem_l757_757695


namespace midpoint_hyperbola_l757_757233

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l757_757233


namespace least_women_required_l757_757354

def at_least_four_men_together_probability (n : ℕ) : ℚ :=
  (2 * (n + 1).choose 2 + (n + 1)) / (3 * (n + 1).choose 2 + (n + 1))

theorem least_women_required : ∃ (n : ℕ), at_least_four_men_together_probability n ≤ 1/100 ∧ n = 594 := 
by
  exists 594
  split
  · 
  sorry
  · 
  refl

end least_women_required_l757_757354


namespace handrail_length_l757_757500

theorem handrail_length (radius height : ℝ) (turn_degrees : ℝ) (h_radius : radius = 4) (h_height : height = 12) (h_turn_degrees : turn_degrees = 180) :
  let circumference := 2 * Real.pi * radius,
      arc_length := (turn_degrees / 360) * circumference,
      length := Real.sqrt (height^2 + arc_length^2)
  in  length ≈ 17.4 :=
by {
  -- Define the constants
  have hc : circumference = 8 * Real.pi, from by { rw [h_radius], norm_num },
  have ha : arc_length = 4 * Real.pi, from by { rw [h_turn_degrees, hc], norm_num },
  have hl : length = Real.sqrt (12^2 + (4 * Real.pi)^2), from by { rw [h_height, ha], norm_num },
  -- Prove the length approximation
  have approx : Real.sqrt (144 + 16 * Real.pi ^ 2) ≈ 17.4,
  { norm_num [Real.pi], linarith },
  rw ←approx,
  exact_eq_approx hl,
  sorry,
}

end handrail_length_l757_757500


namespace problem_statement_l757_757794

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem problem_statement (x : ℝ) : f (3^x) ≥ f (2^x) :=
begin
  -- Proof to be filled in
  sorry
end

end problem_statement_l757_757794


namespace midpoint_on_hyperbola_l757_757052

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757052


namespace can_be_midpoint_of_AB_l757_757186

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757186


namespace smallest_positive_period_maximum_value_and_points_range_in_interval_l757_757581

noncomputable def f (x : ℝ) : ℝ := (Mathlib.sin x + Mathlib.cos x) ^ 2 + 2 * (Mathlib.cos x) ^ 2 - 2

theorem smallest_positive_period (T : ℝ) (h : T = Real.pi) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x := sorry

theorem maximum_value_and_points : 
  ∃ M, M = Real.sqrt 2 ∧ ∀ x, (f x = M ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 8) := sorry

theorem range_in_interval : 
  ∀ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4)), 
    f x ∈ Set.Icc (-Real.sqrt 2) 1 := sorry

end smallest_positive_period_maximum_value_and_points_range_in_interval_l757_757581


namespace midpoint_on_hyperbola_l757_757044

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757044


namespace petya_five_ruble_coins_count_l757_757321

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757321


namespace find_number_l757_757441

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l757_757441


namespace evaluate_expression_l757_757541

theorem evaluate_expression : 
  (10^8 / (2.5 * 10^5) * 3) = 1200 :=
by
  sorry

end evaluate_expression_l757_757541


namespace number_divided_by_3_equals_subtract_3_l757_757433

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l757_757433


namespace midpoint_of_hyperbola_segment_l757_757096

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757096


namespace correct_statement_is_A_l757_757847

variable {V : Type} [AddCommGroup V] [Module ℝ V]

-- Definitions of the vectors
variables {A B C D : V}

-- Definitions of unit vectors
variable {a b : V}

-- Stating the problem
theorem correct_statement_is_A :
  (A ≠ B → (- (B - A)) = (A - B)) ∧
  (∀ a b : V, ∥a∥ = 1 → ∥b∥ = 1 → a = b) = False ∧
  (A ≠ B ∧ D ≠ C → A - B = D - C → ∃ P : V, A + D = P ∧ B + C = P) = False ∧
  (∀ u v : V, u = v ↔ ∃ (O T : V), u = T - O ∧ v = T - O) = False :=
by
  -- The proof steps would normally go here.
  sorry

end correct_statement_is_A_l757_757847


namespace midpoint_on_hyperbola_l757_757077

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l757_757077


namespace find_number_l757_757428

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l757_757428


namespace equal_contribution_each_friend_l757_757812

theorem equal_contribution_each_friend
  (tickets : ℕ) (ticket_cost : ℝ)
  (popcorn : ℕ) (popcorn_cost : ℝ)
  (milk_tea : ℕ) (milk_tea_cost : ℝ)
  (total_friends : ℕ) :
  tickets = 3 →
  ticket_cost = 7 →
  popcorn = 2 →
  popcorn_cost = 1.5 →
  milk_tea = 3 →
  milk_tea_cost = 3 →
  total_friends = 3 →
  (tickets * ticket_cost + popcorn * popcorn_cost + milk_tea * milk_tea_cost) / total_friends = 11 :=
by
  intros h_tickets h_ticket_cost h_popcorn h_popcorn_cost h_milk_tea h_milk_tea_cost h_total_friends
  rw [h_tickets, h_ticket_cost, h_popcorn, h_popcorn_cost, h_milk_tea, h_milk_tea_cost, h_total_friends]
  norm_num
  sorry

end equal_contribution_each_friend_l757_757812


namespace num_ways_to_stain_4x4_window_l757_757473

/-- 
    A 4 × 4 window is made out of 16 square windowpanes.
    Prove the number of ways to stain each of the windowpanes red, pink, or magenta,
    such that each windowpane is the same color as exactly two of its neighbors,
    is 24.
-/
theorem num_ways_to_stain_4x4_window :
  let k := 4 in
  let colors := {red, pink, magenta} in 
  { ways | ways : (Fin k → Fin k) → colors,
    ∀ (i j : Fin k), (ways i j) ≠ (ways i j.succ) ∧ (ways i j) ≠ (ways j i.succ) ∧
                     (ways (i.succ) j) = (ways i j) ∧ (ways i j) = (ways i j.pred) ∧
                     (ways j (i.succ)) = (ways i j) ∧ (ways j i.pred) = (ways i j)
  }.card = 24 :=
sorry

end num_ways_to_stain_4x4_window_l757_757473


namespace solution_set_of_inequality_l757_757393

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2 * x > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l757_757393


namespace correct_midpoint_l757_757204

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757204


namespace eval_f_l757_757618

noncomputable def f (x : ℝ) (a α β : ℝ) :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4

theorem eval_f:
  ∀ (a α β : ℝ), f 2010 a α β = 2 → f 2011 a α β = 6 := by
  intros a α β h_eq
  sorry

end eval_f_l757_757618


namespace area_triangle_CDE_l757_757920

-- Definitions of the geometric entities
variables {O A B C D E : Type}
variables [MetricSpace O] [MetricSpace A] [MetricSpace B] 
          [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Lengths provided as conditions
def length_OA : ℝ := 4
def length_OB : ℝ := 16
def length_DC : ℝ := 12
def length_EA : ℝ := length_DC / 4
def length_DE : ℝ := length_DC - length_EA

-- similarity condition between triangles
axiom triangle_similarity (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] 
  (h₁ : length_OA / length_OB = 1 / 4) : length_EA = length_DC / 4

-- The theorem to prove the area of triangle CDE is 54 square centimeters
theorem area_triangle_CDE : 
  length_DC = 12 ∧ length_DE = 9 → 
  (1 / 2 * length_DC * length_DE : ℝ) = 54 := 
by 
  intro h,
  simp [length_DC, length_DE] at h,
  exact sorry 

end area_triangle_CDE_l757_757920


namespace number_of_five_ruble_coins_l757_757332

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757332


namespace num_five_ruble_coins_l757_757315

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757315


namespace hyperbola_midpoint_exists_l757_757243

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757243


namespace range_of_a_l757_757627

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂)
                    (h₂ : -2 ≤ a + 1 ∧ a + 1 ≤ 4)
                    (h₃ : -2 ≤ 2 * a ∧ 2 * a ≤ 4)
                    (h₄ : f (a + 1) > f (2 * a)) : 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l757_757627


namespace tan_pink_violet_probability_l757_757475

noncomputable def probability_tan_pink_violet_consecutive_order : ℚ :=
  let num_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5)
  let total_ways := Nat.factorial 12
  num_ways / total_ways

theorem tan_pink_violet_probability :
  probability_tan_pink_violet_consecutive_order = 1 / 27720 := by
  sorry

end tan_pink_violet_probability_l757_757475


namespace green_tetrahedron_volume_l757_757782

theorem green_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_green_vertices_of_cube side_length = 512 / 3 :=
by
  sorry

end green_tetrahedron_volume_l757_757782


namespace translate_function_right_by_2_l757_757411

theorem translate_function_right_by_2 (x : ℝ) : 
  (∀ x, (x - 2) ^ 2 + (x - 2) = x ^ 2 - 3 * x + 2) := 
by 
  sorry

end translate_function_right_by_2_l757_757411


namespace fractions_equivalence_l757_757944

theorem fractions_equivalence (k : ℝ) (h : k ≠ -5) : (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := 
by 
  sorry

end fractions_equivalence_l757_757944


namespace bounds_for_f_l757_757744

def f (n : ℕ) : ℕ := -- Definition of f (it is the maximum number of elements from Example 1, unclear exact specifics from the given text)
sorry

theorem bounds_for_f (n : ℕ) : 
  1/6 * (n^2 - 4 * n) ≤ f(n) ∧ f(n) ≤ 1/6 * (n^2 - n) :=
by 
  sorry

end bounds_for_f_l757_757744


namespace prime_sum_inequality_l757_757720

open Real

theorem prime_sum_inequality {n : ℕ} (primes : Fin n → ℕ) (hprimes : ∀ i : Fin n, Nat.Prime (primes i)) (hn : 2 < n) :
  (∑ i : Fin n, (1 : ℝ) / (primes i)^2) + (1 / ∏ i : Fin n, primes i) < 1 / 2 := 
sorry

end prime_sum_inequality_l757_757720


namespace inequality_solution_interval_l757_757773

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l757_757773


namespace range_of_a_l757_757448

namespace InequalityProblem

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) → (x - 1)^2 < Real.log x / Real.log a) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end InequalityProblem

end range_of_a_l757_757448


namespace rose_flyers_l757_757710

theorem rose_flyers (total_flyers made: ℕ) (flyers_jack: ℕ) (flyers_left: ℕ) 
(h1 : total_flyers = 1236)
(h2 : flyers_jack = 120)
(h3 : flyers_left = 796)
: total_flyers - flyers_jack - flyers_left = 320 :=
by
  sorry

end rose_flyers_l757_757710


namespace prob_four_dice_product_div_by_8_l757_757414

noncomputable def prob_div_by_8 (n : ℕ) : ℚ :=
  let num_dice := 4 in
  let faces := 6 in
  1 - (1/2)^num_dice - (nat.choose num_dice 2 * (1/faces)^2 * (1/2)^2)

theorem prob_four_dice_product_div_by_8 : prob_div_by_8 4 = 43/48 := by
  sorry

end prob_four_dice_product_div_by_8_l757_757414


namespace midpoint_of_hyperbola_l757_757066

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757066


namespace num_five_ruble_coins_l757_757313

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l757_757313


namespace square_free_odd_integers_count_l757_757653

/-- Define the set of odd integers greater than 1 and less than 200 -/
def odd_integers := {n : ℕ | n > 1 ∧ n < 200 ∧ n % 2 = 1}

/-- Define a square-free predicate -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

/-- Define the set of square-free odd integers greater than 1 and less than 200 -/
def square_free_odd_integers := {n : ℕ | n ∈ odd_integers ∧ square_free n}

/-- The number of square-free odd integers between 1 and 200 is 79 -/
theorem square_free_odd_integers_count : 
  set.finite square_free_odd_integers ∧ set.card square_free_odd_integers = 79 :=
begin
  sorry
end

end square_free_odd_integers_count_l757_757653


namespace steve_height_equiv_l757_757358

/-- 
  Steve's initial height in feet and inches.
  convert_height: converts feet and inches to total inches.
  grows further: additional height Steve grows.
  expected height: the expected total height after growing.
--/

def initial_height_feet := 5
def initial_height_inches := 6
def additional_height := 6

def convert_height(feet: Int, inches: Int): Int := 
  feet * 12 + inches

def expected_height(initial_feet: Int, initial_inches: Int, additional: Int): Int := 
  convert_height(initial_feet, initial_inches) + additional

theorem steve_height_equiv:
  expected_height initial_height_feet initial_height_inches additional_height = 72 :=
by
  sorry

end steve_height_equiv_l757_757358


namespace table_height_l757_757503

theorem table_height (l w h : ℝ) (h1 : l + h - w = 38) (h2 : w + h - l = 34) : h = 36 :=
by
  sorry

end table_height_l757_757503


namespace max_value_of_x_l757_757022

theorem max_value_of_x (x y : ℝ) (h : x^2 + y^2 = 18 * x + 20 * y) : x ≤ 9 + Real.sqrt 181 :=
by
  sorry

end max_value_of_x_l757_757022


namespace midpoint_of_line_segment_on_hyperbola_l757_757031

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757031


namespace grocer_decaf_coffee_percentage_l757_757877

theorem grocer_decaf_coffee_percentage :
  let total_initial_coffee := 400
  let percent_initial_decaf := 0.20
  let additional_coffee := 100
  let percent_additional_decaf := 0.70

  let initial_decaf := percent_initial_decaf * total_initial_coffee
  let additional_decaf := percent_additional_decaf * additional_coffee
  let total_coffee := total_initial_coffee + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  let percent_decaf := (total_decaf / total_coffee) * 100

  percent_decaf = 30 :=
by
  unfold total_initial_coffee percent_initial_decaf additional_coffee percent_additional_decaf
  unfold initial_decaf additional_decaf total_coffee total_decaf percent_decaf
  sorry

end grocer_decaf_coffee_percentage_l757_757877


namespace midpoint_hyperbola_l757_757107

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757107


namespace five_ruble_coins_count_l757_757307

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757307


namespace arithmetic_series_sum_l757_757838

theorem arithmetic_series_sum :
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  sum = 418 := by {
  let a := 18
  let d := 4
  let l := 58
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  have h₁ : n = 11 := by sorry
  have h₂ : sum = 418 := by sorry
  exact h₂
}

end arithmetic_series_sum_l757_757838


namespace correct_midpoint_l757_757216

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757216


namespace inverse_function_passes_through_point_l757_757678

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_passes_through_point {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  f a⁻¹ 1 = -1 :=
sorry

end inverse_function_passes_through_point_l757_757678


namespace exists_f_eq_f_deriv_l757_757875

noncomputable def f_A (x : ℝ) : ℝ := 1 - x
noncomputable def f_B (x : ℝ) : ℝ := x
noncomputable def f_C (x : ℝ) : ℝ := Real.exp x
noncomputable def f_D (x : ℝ) : ℝ := 1

theorem exists_f_eq_f_deriv : ∃ (f : ℝ → ℝ), f = Real.exp :=
by {
  use f_C,
  apply funext,
  intro x,
  exact Eq.refl (Real.exp x)
}

end exists_f_eq_f_deriv_l757_757875


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757284

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757284


namespace triangle_min_diff_l757_757820

theorem triangle_min_diff (XZ XY YZ : ℕ) (h1 : XZ + XY + YZ = 3030) (h2 : XZ < XY) (h3 : XY ≤ YZ) (h4 : ∃ k, XY = 5 * k) :
  (XY - XZ) = 1 :=
begin
  sorry
end

end triangle_min_diff_l757_757820


namespace midpoint_of_hyperbola_l757_757065

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757065


namespace inequality_solution_l757_757771

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end inequality_solution_l757_757771


namespace tan_half_theta_l757_757571

theorem tan_half_theta (θ : ℝ) (h1 : Real.sin θ = -3 / 5) (h2 : 3 * Real.pi < θ ∧ θ < 7 / 2 * Real.pi) :
  Real.tan (θ / 2) = -3 :=
sorry

end tan_half_theta_l757_757571


namespace smallest_m_for_integral_solutions_l757_757836

theorem smallest_m_for_integral_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ x, x ∈ ℤ → 10 * x^2 - m * x + 1980 = 0 → some (x, y) ∈ [(1, 198), (2, 99), (3, 66), (6, 33), (9, 22), (11, 18)] ∧ m = 290) :=
sorry

end smallest_m_for_integral_solutions_l757_757836


namespace arun_completes_work_alone_in_70_days_l757_757516

def arun_days (A : ℕ) : Prop :=
  ∃ T : ℕ, (A > 0) ∧ (T > 0) ∧ 
           (∀ (work_done_by_arun_in_1_day work_done_by_tarun_in_1_day : ℝ),
            work_done_by_arun_in_1_day = 1 / A ∧
            work_done_by_tarun_in_1_day = 1 / T ∧
            (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day = 1 / 10) ∧
            (4 * (work_done_by_arun_in_1_day + work_done_by_tarun_in_1_day) = 4 / 10) ∧
            (42 * work_done_by_arun_in_1_day = 6 / 10) )

theorem arun_completes_work_alone_in_70_days : arun_days 70 :=
  sorry

end arun_completes_work_alone_in_70_days_l757_757516


namespace number_of_square_free_odds_l757_757646

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l757_757646


namespace sin_value_of_arithmetic_sequence_l757_757699

noncomputable def a4 (a : ℕ → ℝ) : ℝ := (3 * Real.pi) / 4

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ)
  (h_arith_seq : ∀ n m k, a n + a (n + k) = 2 * a (n + k / 2) ∨ (k + n ∈ set.univ) ∧ (k + m ∈ set.univ)) -- Arith sequence
  (h_sum : a 2 + a 6 = (3 * Real.pi) / 2) : 
  Real.sin (2 * a4 a - Real.pi / 3) = -1 / 2 := 
by {
    have h_a4 : a4 a = ((3 * Real.pi) / 4) := by sorry,
    have h_sin : 2 * a4 a - (Real.pi / 3) = Real.pi := by sorry,
    rw h_a4 at h_sin,
    rw Real.sin_pi,
    norm_num,
}

end sin_value_of_arithmetic_sequence_l757_757699


namespace tan_neg_405_eq_neg_1_l757_757928

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l757_757928


namespace midpoint_of_hyperbola_segment_l757_757093

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757093


namespace number_of_minibusses_l757_757709

def total_students := 156
def students_per_van := 10
def students_per_minibus := 24
def number_of_vans := 6

theorem number_of_minibusses : (total_students - number_of_vans * students_per_van) / students_per_minibus = 4 :=
by
  sorry

end number_of_minibusses_l757_757709


namespace area_of_quadrilateral_l757_757900

theorem area_of_quadrilateral (area_of_triangle : Real)
  (n_triangles : Nat) (n_quadrilaterals : Nat)
  (triangles_equal_area : ∀ t, t = area_of_triangle)
  (total_area : ∀ (quil : Real), total_area = 
    n_triangles * area_of_triangle + n_quadrilaterals * quil) :
  ∃ (A : Real), A = Real.sqrt 5 + 1 := by
  let area_of_triangle_val := 1 -- 1 cm^2
  let n_triangles_val := 4 -- four triangles
  let n_quadrilaterals_val := 3 -- three quadrilaterals
  have triangles_equal_area_val : ∀ t, t = area_of_triangle_val := by sorry
  have total_area_val : ∀ (A : Real), total_area = 
    n_triangles_val * area_of_triangle_val + n_quadrilaterals_val * A := by sorry
  use Real.sqrt 5 + 1
  sorry

end area_of_quadrilateral_l757_757900


namespace TOP_books_sold_correct_l757_757517

noncomputable def num_of_TOP_books_sold (price_TOP : ℕ) (price_ABC : ℕ) (num_ABC : ℕ) (diff : ℕ) : ℕ :=
  let T := (diff + num_ABC * price_ABC) / price_TOP in T

theorem TOP_books_sold_correct : 
  num_of_TOP_books_sold 8 23 4 12 = 13 := 
by
  unfold num_of_TOP_books_sold
  sorry

end TOP_books_sold_correct_l757_757517


namespace midpoint_hyperbola_l757_757104

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757104


namespace hyperbola_midpoint_l757_757127

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l757_757127


namespace hyperbola_midpoint_l757_757199

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757199


namespace petya_five_ruble_coins_count_l757_757325

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l757_757325


namespace joan_games_last_year_l757_757714

theorem joan_games_last_year (games_this_year : ℕ) (total_games : ℕ) (games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : total_games = 9) 
  (h3 : total_games = games_this_year + games_last_year) : 
  games_last_year = 5 := 
by
  sorry

end joan_games_last_year_l757_757714


namespace part_a_total_time_part_b_average_time_part_c_probability_l757_757292

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l757_757292


namespace part_a_total_time_part_b_average_time_part_c_probability_l757_757288

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l757_757288


namespace parabola_focus_l757_757985

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l757_757985


namespace hours_apart_l757_757713

-- Define the conditions according to the given problem.
variable (pills : ℕ) (mg_per_pill : ℕ) (mg_per_dose : ℕ) (weeks : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ)
variable (total_pills : pills = 112)
variable (mg_per_pill_eq : mg_per_pill = 500)
variable (mg_per_dose_eq : mg_per_dose = 1000)
variable (weeks_eq : weeks = 2)
variable (days_per_week_eq : days_per_week = 7)
variable (hours_per_day_eq : hours_per_day = 24)

-- State the theorem to be proven.
theorem hours_apart (h : pills = 112) (h1 : mg_per_pill = 500) (h2 : mg_per_dose = 1000) (h3: weeks = 2) (h4: days_per_week = 7) (h5: hours_per_day = 24) :
  let total_mg := pills * mg_per_pill in
  let doses := total_mg / mg_per_dose in
  let total_hours := weeks * days_per_week * hours_per_day in
  total_hours / doses = 6 :=
by
  sorry

end hours_apart_l757_757713


namespace eighth_graders_taller_rows_remain_ordered_l757_757467

-- Part (a)

theorem eighth_graders_taller {n : ℕ} (h8 : Fin n → ℚ) (h7 : Fin n → ℚ)
  (ordered8 : ∀ i j : Fin n, i ≤ j → h8 i ≤ h8 j)
  (ordered7 : ∀ i j : Fin n, i ≤ j → h7 i ≤ h7 j)
  (initial_condition : ∀ i : Fin n, h8 i > h7 i) :
  ∀ i : Fin n, h8 i > h7 i :=
sorry

-- Part (b)

theorem rows_remain_ordered {m n : ℕ} (h : Fin m → Fin n → ℚ)
  (row_ordered : ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k)
  (column_ordered_after : ∀ j : Fin n, ∀ i k : Fin m, i ≤ k → h i j ≤ h k j) :
  ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k :=
sorry

end eighth_graders_taller_rows_remain_ordered_l757_757467


namespace hyperbola_midpoint_l757_757189

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757189


namespace midpoint_of_line_segment_on_hyperbola_l757_757038

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757038


namespace midpoint_of_line_segment_on_hyperbola_l757_757024

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757024


namespace max_sum_of_products_cube_faces_l757_757510

theorem max_sum_of_products_cube_faces:
  ∃ (a b c d e f : ℝ), 
    {a, b, c, d, e, f} ⊆ {2, 3, 4, 6, 7, 8} ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧
    (
      (a + b = 10 ∧ c + d = 11 ∧ e + f = 11) ∨
      (a + b = 11 ∧ c + d = 10 ∧ e + f = 11) ∨
      (a + b = 11 ∧ c + d = 11 ∧ e + f = 10)
    ) ∧
    (a + b) * (c + d) * (e + f) = 1210 :=
begin
  sorry
end

end max_sum_of_products_cube_faces_l757_757510


namespace hyperbola_midpoint_exists_l757_757237

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757237


namespace center_of_incircle_on_MN_l757_757249

noncomputable theory

variables {A B C D M N : Type*}
variables [trapezoid_circumscribed A B C D] (incircle_triangle_ABC : circle) 

-- Ensure A, B, C, and D form a trapezoid with AB parallel to CD and AB > CD
def trapezoid_circumscribed (A B C D : Type*) : Prop :=
  parallel AB CD ∧ length AB > length CD

-- Define tangency points of the incircle of triangle ABC at M and N
def tangency_points (circle : Type*) (triangle : Type*) (A B C M N : Type*) : Prop :=
  is_tangent circle AB M ∧ is_tangent circle AC N

-- Define the center of the incircle of trapezoid ABCD
def center_of_incircle (A B C D : Type*) : Type* := sorry

-- State the main theorem
theorem center_of_incircle_on_MN 
  (h1 : trapezoid_circumscribed A B C D) 
  (h2 : tangency_points incircle_triangle_ABC ABC A B C M N) :
  lies_on (center_of_incircle A B C D) (line_through M N) := 
sorry

end center_of_incircle_on_MN_l757_757249


namespace five_ruble_coins_count_l757_757308

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l757_757308


namespace hyperbola_midpoint_l757_757198

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757198


namespace all_real_numbers_satisfy_property_l757_757544

theorem all_real_numbers_satisfy_property :
  ∀ (α : ℝ), (∀ (n : ℕ), n > 0 → ∃ (m : ℤ), abs (α - (m : ℝ)/n) < 1 / (3 * n)) :=
by {
  intro α,
  intro n,
  intro hn,
  -- sorry for the proof part
  sorry
}

end all_real_numbers_satisfy_property_l757_757544


namespace range_of_t_l757_757619

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc 0 1 then 3^x else
if x ∈ set.Ioc 1 3 then (9 / 2 - (3 / 2) * x) else 0

theorem range_of_t (t : ℝ) (h1 : t ∈ set.Icc 0 1) (h2 : f (f t) ∈ set.Icc 0 1) : 
    t ∈ set.Icc (Real.log 7 / 3) 1 :=
by
  sorry

end range_of_t_l757_757619


namespace find_m_range_l757_757635

variable {x y m : ℝ}

theorem find_m_range (h1 : x + 2 * y = m + 4) (h2 : 2 * x + y = 2 * m - 1)
    (h3 : x + y < 2) (h4 : x - y < 4) : m < 1 := by
  sorry

end find_m_range_l757_757635


namespace hyperbola_midpoint_exists_l757_757246

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757246


namespace angle_A_centroid_l757_757599

variable {A B C G : Point}
variable {a b c : ℝ}
variable {GA GB GC : Vector}
variable [Centroid G A B C]

theorem angle_A_centroid
  (h1 : a * GA + b * GB + (√3 / 3) * c * GC = 0)
  (h2 : GA + GB + GC = 0)
  (h3 : GC = - (GA + GB))
  (h4 : a = (√3 / 3) * c)
  (h5 : b = (√3 / 3) * c) :
  angle A = π / 6 :=
sorry

end angle_A_centroid_l757_757599


namespace relationship_among_a_b_c_l757_757604

noncomputable def a : ℝ := Real.logb 0.5 0.2
noncomputable def b : ℝ := Real.logb 2 0.2
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 2)

theorem relationship_among_a_b_c : b < c ∧ c < a :=
by
  sorry

end relationship_among_a_b_c_l757_757604


namespace num_five_ruble_coins_l757_757338

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757338


namespace smallest_primest_is_72_l757_757494

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).eraseDup.length

def is_primer (n : ℕ) : Prop :=
  is_prime (count_distinct_prime_factors n)

def count_distinct_primer_factors (n : ℕ) : ℕ :=
  (Nat.divisors n).filter (λ x => is_primer x).eraseDup.length

def is_primest (n : ℕ) : Prop :=
  is_primer (count_distinct_primer_factors n)

theorem smallest_primest_is_72 : ∀ n : ℕ, is_primest n → 72 ≤ n :=
by
  sorry

end smallest_primest_is_72_l757_757494


namespace max_value_x_plus_y_l757_757742

theorem max_value_x_plus_y : ∀ (x y : ℝ), 
  (5 * x + 3 * y ≤ 9) → 
  (3 * x + 5 * y ≤ 11) → 
  x + y ≤ 32 / 17 :=
by
  intros x y h1 h2
  -- proof steps go here
  sorry

end max_value_x_plus_y_l757_757742


namespace steve_height_after_growth_l757_757361

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l757_757361


namespace eccentricity_of_hyperbola_l757_757608

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ), (0 < a ∧ 0 < b ∧ 3 * a^2 = b^2) → 
  let e := (sqrt (1 + (b / a)^2)) in e = 2 := 
by
  intros a b h,
  sorry

end eccentricity_of_hyperbola_l757_757608


namespace positive_difference_l757_757395

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l757_757395


namespace can_be_midpoint_of_AB_l757_757187

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757187


namespace gcf_60_90_150_l757_757828

theorem gcf_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 :=
by
  sorry

end gcf_60_90_150_l757_757828


namespace can_be_midpoint_of_AB_l757_757180

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757180


namespace number_of_maas_and_maas_per_pib_l757_757587

axiom P1 (pib : Type) (maa : Type) : pib → Set maa
axiom P2 (pib : Type) (maa : Type) [Fintype pib] [DecidableEq pib] [Fintype maa] [DecidableEq maa] : 
  ∀ (p1 p2 : pib), p1 ≠ p2 → Fintype.card (P1 pib maa p1 ∩ P1 pib maa p2) = 2
axiom P3 (pib : Type) (maa : Type) [Fintype pib] [Fintype maa] : 
  ∀ (m : maa), Fintype.card {p : pib // m ∈ P1 pib maa p} = 3
axiom P4 : Fintype.card pib = 6

theorem number_of_maas_and_maas_per_pib (pib : Type) (maa : Type) [Fintype pib] [Fintype maa] :
  Fintype.card maa = 10 ∧ ∀ (p : pib), Fintype.card (P1 pib maa p) = 5 := 
by
  sorry

end number_of_maas_and_maas_per_pib_l757_757587


namespace number_divided_by_three_l757_757446

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l757_757446


namespace new_ratio_is_9_91_l757_757519

def initial_lifting_total : ℝ := 2200
def initial_body_weight : ℝ := 245
def gained_body_weight : ℝ := 8
def squat_gain_percent : ℝ := 0.12
def bench_press_gain_percent : ℝ := 0.10
def deadlift_gain_percent : ℝ := 0.18
def squat_proportion : ℝ := 4
def bench_press_proportion : ℝ := 3
def deadlift_proportion : ℝ := 5

noncomputable def initial_common_factor : ℝ := initial_lifting_total / (squat_proportion + bench_press_proportion + deadlift_proportion)

noncomputable def initial_squat : ℝ := squat_proportion * initial_common_factor
noncomputable def initial_bench_press : ℝ := bench_press_proportion * initial_common_factor
noncomputable def initial_deadlift : ℝ := deadlift_proportion * initial_common_factor

noncomputable def new_squat : ℝ := initial_squat * (1 + squat_gain_percent)
noncomputable def new_bench_press : ℝ := initial_bench_press * (1 + bench_press_gain_percent)
noncomputable def new_deadlift : ℝ := initial_deadlift * (1 + deadlift_gain_percent)

noncomputable def new_total_lifting : ℝ := new_squat + new_bench_press + new_deadlift
noncomputable def new_body_weight : ℝ := initial_body_weight + gained_body_weight

noncomputable def new_ratio : ℝ := new_total_lifting / new_body_weight

theorem new_ratio_is_9_91 :
  real.round (new_ratio * 100) / 100 = 9.91 :=
sorry

end new_ratio_is_9_91_l757_757519


namespace solution_set_of_f_g_l757_757730

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Theorem statement
theorem solution_set_of_f_g :
  (∀ x, f (-x) = -f x) ∧ -- f is odd function
  (∀ x, g (-x) = g x) ∧ -- g is even function
  (∀ x < 0, f' x * g x + f x * g' x > 0) ∧ -- condition for x < 0
  g 3 = 0 -- given g(3) = 0
  → {x | f x * g x < 0} = {x | x < -3} ∪ {x | 0 < x ∧ x < 3} :=
sorry

end solution_set_of_f_g_l757_757730


namespace sock_combination_count_l757_757403

noncomputable def numSockCombinations : Nat :=
  let striped := 4
  let solid := 4
  let checkered := 4
  let striped_and_solid := striped * solid
  let striped_and_checkered := striped * checkered
  striped_and_solid + striped_and_checkered

theorem sock_combination_count :
  numSockCombinations = 32 :=
by
  unfold numSockCombinations
  sorry

end sock_combination_count_l757_757403


namespace numberOfPolynomialsInH_l757_757723

def isPolynomialInH (Q : ℤ[X]) : Prop :=
  ∃ n : ℕ, ∃ (c : fin n → ℤ), 
    Q = ∑ i in range n, c i * X^(n-i) + 24 * X^0 ∧
    ∀ x : ℂ, is_root (Q.map (algebra_map ℤ ℂ)) x → 
    (∃ a b : ℤ, x = a + b * complex.I)

theorem numberOfPolynomialsInH : 
  fintype {Q : ℤ[X] // isPolynomialInH Q}.card = 56 :=
begin
  sorry
end

end numberOfPolynomialsInH_l757_757723


namespace midpoint_of_line_segment_on_hyperbola_l757_757035

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l757_757035


namespace midpoint_on_hyperbola_l757_757050

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757050


namespace nthEquation_l757_757753

noncomputable def alternatingSum (n : ℕ) : ℝ :=
  ∑ i in (Finset.range (2 * n)).filter(λ i => i % 2 = 0), (-1 : ℝ) ^ i * (1 / (i + 1))

noncomputable def harmonicSum (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n + 1), if i ≥ n then 1 / (i + 1) else 0

theorem nthEquation (n : ℕ) : 
  alternatingSum n = harmonicSum n := 
sorry

end nthEquation_l757_757753


namespace initial_percentage_of_alcohol_l757_757470

theorem initial_percentage_of_alcohol :
  ∃ P : ℝ, (P / 100 * 11) = (33 / 100 * 14) :=
by
  use 42
  sorry

end initial_percentage_of_alcohol_l757_757470


namespace hyperbola_midpoint_exists_l757_757247

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l757_757247


namespace tom_pie_share_l757_757912

theorem tom_pie_share :
  (∃ (x : ℚ), 4 * x = (5 / 8) ∧ x = 5 / 32) :=
by
  sorry

end tom_pie_share_l757_757912


namespace physics_class_size_l757_757518

-- Definitions based on the problem conditions
def total_students : ℕ := 50
def both_classes : ℕ := 6
def physics_twice_math (math only : ℕ) : Prop := (2 * (math only + both_classes) = (2 * math only) + 6)
def total_students_eq (physics only math only : ℕ) : Prop := (physics only + math only + both_classes = total_students)

-- Theorem to prove the size of the physics class
theorem physics_class_size : ∃ (physics only : ℕ), physics only + both_classes = 112 / 3 := by
  sorry

end physics_class_size_l757_757518


namespace necessary_and_sufficient_condition_extremum_l757_757382

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

theorem necessary_and_sufficient_condition_extremum (a : ℝ) :
  (∃ x, f a x = 0) ↔ -3 < a ∧ a < 4 :=
sorry

end necessary_and_sufficient_condition_extremum_l757_757382


namespace solve_inequality_l757_757776

def polynomial_fraction (x : ℝ) : ℝ :=
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem solve_inequality (x : ℝ) :
  -2 < polynomial_fraction x ∧ polynomial_fraction x < 2 ↔ 11.57 < x :=
sorry

end solve_inequality_l757_757776


namespace rain_next_tuesday_is_not_definite_l757_757708

theorem rain_next_tuesday_is_not_definite : 
  ∀ (next_tuesday : ℕ), is_future_time next_tuesday → weather_is_uncertain next_tuesday → ¬ will_definitely_rain next_tuesday :=
by
  intros next_tuesday H_future H_uncertain
  -- proof goes here
  sorry

end rain_next_tuesday_is_not_definite_l757_757708


namespace cos_value_l757_757601

-- Given condition
axiom sin_condition (α : ℝ) : Real.sin (Real.pi / 6 + α) = 2 / 3

-- The theorem we need to prove
theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) : 
  Real.cos (Real.pi / 3 - α) = 2 / 3 := 
by 
  sorry

end cos_value_l757_757601


namespace part_a_total_time_part_b_average_time_part_c_probability_l757_757289

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l757_757289


namespace ceil_sqrt_244_l757_757958

theorem ceil_sqrt_244 : (⌈Real.sqrt 244⌉ = 16) :=
by
  have h1 : 15 < Real.sqrt 244 := sorry
  have h2 : Real.sqrt 244 < 16 := sorry
  exact ceil_eq_of_lt h1 h2

end ceil_sqrt_244_l757_757958


namespace correct_midpoint_l757_757207

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757207


namespace midpoint_on_hyperbola_l757_757136

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757136


namespace focus_of_parabola_l757_757997

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l757_757997


namespace cotangent_ratio_l757_757252

-- Define the variables for the sides and angles of the triangle
variables (a b c : ℝ) (α β γ : ℝ)

-- Assume the sides a, b, c form a triangle
-- and α, β, γ are the corresponding opposite angles
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom triangle_angles : α + β + γ = π

-- Given condition a^2 + b^2 = 1989c^2
axiom side_condition : a^2 + b^2 = 1989 * c^2

-- Law of Sines and Cotangent relationship
noncomputable def cot (x : ℝ) : ℝ := cos x / sin x

-- The proof statement we have to show
theorem cotangent_ratio : 
  triangle_sides a b c → 
  triangle_angles α β γ → 
  side_condition a b c →
  a / sin α = b / sin β ∧ b / sin β = c / sin γ →
  cot γ / (cot α + cot β) = 994 :=
by
  intros
  sorry

end cotangent_ratio_l757_757252


namespace midpoint_of_hyperbola_l757_757056

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l757_757056


namespace larger_integer_value_l757_757390

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l757_757390


namespace sum_squares_and_cube_mod_13_eq_0_l757_757835

theorem sum_squares_and_cube_mod_13_eq_0 : 
  (∑ i in Finset.range 16, i ^ 2 + 15 ^ 3) % 13 = 0 :=
by
  sorry

end sum_squares_and_cube_mod_13_eq_0_l757_757835


namespace methane_reacts_with_oxygen_l757_757641

theorem methane_reacts_with_oxygen 
    (moles_of_O2 : ℝ)
    (reaction_ratio_CH4_O2 : ℝ)
    (reaction_ratio_CO2_CH4 : ℝ) 
    : (moles_of_CH4_required : ℝ) ∧ (moles_of_CO2_formed : ℝ) :=
begin
  let moles_of_CH4_required := moles_of_O2 * (1 / 2),
  let moles_of_CO2_formed := moles_of_CH4_required * (1 / 1),
  exact ⟨moles_of_CH4_required, moles_of_CO2_formed⟩,
  sorry
end

-- Given conditions
def moles_of_O2 := 2
def reaction_ratio_CH4_O2 := 1 / 2
def reaction_ratio_CO2_CH4 := 1 / 1

-- The correct answer
def correct_moles_of_CH4_required := 1
def correct_moles_of_CO2_formed := 1

example : methane_reacts_with_oxygen moles_of_O2 reaction_ratio_CH4_O2 reaction_ratio_CO2_CH4
  = ⟨correct_moles_of_CH4_required, correct_moles_of_CO2_formed⟩ :=
by sorry

end methane_reacts_with_oxygen_l757_757641


namespace fraction_addition_simplification_l757_757523

theorem fraction_addition_simplification :
  (2 / 5 : ℚ) + (3 / 15) = 3 / 5 :=
by
  sorry

end fraction_addition_simplification_l757_757523


namespace measure_angle_CDB_is_15_degrees_l757_757513

-- Definitions for conditions
def is_equilateral_triangle (A B C : Type) [EuclideanGeometry] : Prop :=
  equil_sides A B C ∧ equil_sides B C A ∧ equil_sides C A B

def is_square (A B C D : Type) [EuclideanGeometry] : Prop :=
  equil_sides A B C D ∧ right_angles A B C D

def shared_side (triangle1 triangle2 : Type) [EuclideanGeometry] : Prop :=
  equil_sides triangle1 triangle2

-- The triangle BCD is part of the conditions
def is_isosceles_triangle (B C D : Type) [EuclideanGeometry] : Prop :=
  side_eq B C ∧ side_eq C D

theorem measure_angle_CDB_is_15_degrees
    {square : Type} {A B C D : square}
    {triangle : Type} {E F G : triangle}
    [EuclideanGeometry : EuclideanGeometry]
    (H1 : is_square A B C D)
    (H2 : is_equilateral_triangle E F G)
    (H3 : shared_side triangle square)
    (H4 : is_isosceles_triangle B C D) :
    measure (angle C D B) = 15 :=
by sorry

end measure_angle_CDB_is_15_degrees_l757_757513


namespace find_number_l757_757429

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l757_757429


namespace midpoint_hyperbola_l757_757111

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757111


namespace total_books_proof_l757_757562

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l757_757562


namespace find_number_l757_757439

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l757_757439


namespace polynomial_function_unique_l757_757966

theorem polynomial_function_unique (f : ℝ → ℝ)
  (h1 : ∀ x, f (x^2) = (f x)^2)
  (h2 : ∀ x, deriv f x = 2 * f x)
  (hf : polynomialFunction f)
  (hf_deg : polynomialDegree f ≥ 1) :
  f = λ x, x^2 := 
sorry

end polynomial_function_unique_l757_757966


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757294

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l757_757294


namespace square_problem_l757_757357

theorem square_problem (O : Point) (A B C D E F : Point) (distance : ℝ)
  (hSquare : is_square A B C D O)
  (hAB : distance A B = 1200)
  (hEA_LT_BF : distance A E < distance B F)
  (hE_BT_F : is_between E A F)
  (hEOF : ∠ E O F = 60)
  (hEF : distance E F = 500)
  (p q r : ℕ)
  (hBF : distance B F = p + q * Real.sqrt r)
  (h_correct_p : p = 400) (h_correct_q : q = 100) (h_correct_r : r = 3) :
  p + q + r = 503 :=
sorry

end square_problem_l757_757357


namespace satisfies_differential_equation_l757_757466

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x

theorem satisfies_differential_equation (x : ℝ) (hx : x ≠ 0) : 
  x * (deriv (fun x => (Real.sin x) / x) x) + (Real.sin x) / x = Real.cos x := 
by
  -- the proof goes here
  sorry

end satisfies_differential_equation_l757_757466


namespace num_five_ruble_coins_l757_757339

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l757_757339


namespace ferris_wheel_seat_capacity_l757_757783

theorem ferris_wheel_seat_capacity (seats people : ℕ) (h_seats : seats = 4) (h_people : people = 20) : people / seats = 5 :=
by {
  -- We have:
  -- seats = 4
  -- people = 20
  -- 20 / 4 = 5
  rw [h_seats, h_people],
  norm_num,
  sorry -- proof to be completed
}

end ferris_wheel_seat_capacity_l757_757783


namespace nonreal_eigenvalues_of_matrix_l757_757021

variables {n : ℕ} 
variables (A : Matrix (Fin n) (Fin n) ℝ)

theorem nonreal_eigenvalues_of_matrix
  (hAn : A^n ≠ 0)
  (h_condition : ∀ i j, (A i j) * (A j i) ≤ 0) : 
  ∃ λ₁ λ₂ : ℂ, λ₁ ≠ λ₂ ∧ λ₁.im ≠ 0 ∧ λ₂.im ≠ 0 ∧ is_eigenvalue A λ₁ ∧ is_eigenvalue A λ₂ :=
sorry

end nonreal_eigenvalues_of_matrix_l757_757021


namespace value_of_a_l757_757676

noncomputable def n : ℝ := 2 ^ 0.1

def b : ℝ := 40.00000000000002

theorem value_of_a (a : ℤ) (H1 : n^b = a) : a = 16 :=
sorry

end value_of_a_l757_757676


namespace LCM_pairs_divisors_eq_l757_757733

open Nat

theorem LCM_pairs_divisors_eq (n : ℕ) : 
  let num_pairs : ℕ := nat.prod (λ p_i : ℕ, 2 * (nat.factorization n p_i) + 1)
  let num_divisors_n2 : ℕ := nat.prod (λ p_i: ℕ, 2 * (nat.factorization (n * n) p_i) + 1)
  num_pairs = num_divisors_n2 :=
sorry

end LCM_pairs_divisors_eq_l757_757733


namespace paper_folding_holes_l757_757890

theorem paper_folding_holes (initial_paper : ℝ × ℝ)
                            (folds : list (ℝ × ℝ → ℝ × ℝ))
                            (final_transformation : ℝ × ℝ → ℝ × ℝ)
                            (hole_position : ℝ × ℝ)
                            (unfold_transformations : list (ℝ × ℝ → ℝ × ℝ))
                            (final_state : set (ℝ × ℝ)) :
  initial_paper = (2, 4) →
  folds = [λ (p : ℝ × ℝ), (p.1, p.2 / 2),  -- First fold (bottom to top)
           λ (p : ℝ × ℝ), (p.1 / 2, p.2),   -- Second fold (left to right)
           λ (p : ℝ × ℝ), (p.1 / 2, p.2)] → -- Third fold (left to right)
  final_transformation = λ (p : ℝ × ℝ), (p.1 / 2, p.2 / 4) →
  hole_position = (0.25, 1) →   -- Center of the final folded piece
  unfold_transformations = [λ (p : ℝ × ℝ), (p.1 * 2, p.2),
                            λ (p : ℝ × ℝ), (p.1 * 2, p.2), 
                            λ (p : ℝ × ℝ), (p.1, p.2 * 2)] →
  final_state = {(0.25, 1), (0.75, 1), (1.25, 1), (1.75, 1), (0.25, 3), (0.75, 3), (1.25, 3), (1.75, 3)} →
  ∃ (result : set (ℝ × ℝ)), result = final_state :=
by
  sorry

end paper_folding_holes_l757_757890


namespace no_such_triples_l757_757545

noncomputable def no_triple_satisfy (a b c : ℤ) : Prop :=
  ∀ (x1 x2 x3 : ℤ), 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    Int.gcd x1 x2 = 1 ∧ Int.gcd x2 x3 = 1 ∧ Int.gcd x1 x3 = 1 ∧
    (x1^3 - a^2 * x1^2 + b^2 * x1 - a * b + 3 * c = 0) ∧ 
    (x2^3 - a^2 * x2^2 + b^2 * x2 - a * b + 3 * c = 0) ∧ 
    (x3^3 - a^2 * x3^2 + b^2 * x3 - a * b + 3 * c = 0) →
    False

theorem no_such_triples : ∀ (a b c : ℤ), no_triple_satisfy a b c :=
by
  intros
  sorry

end no_such_triples_l757_757545


namespace midpoint_of_hyperbola_l757_757172

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l757_757172


namespace hyperbola_midpoint_l757_757196

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l757_757196


namespace oldest_sibling_multiple_l757_757014

-- Definitions according to the conditions
def kay_age : Nat := 32
def youngest_sibling_age : Nat := kay_age / 2 - 5
def oldest_sibling_age : Nat := 44

-- The statement to prove
theorem oldest_sibling_multiple : oldest_sibling_age = 4 * youngest_sibling_age :=
by sorry

end oldest_sibling_multiple_l757_757014


namespace number_of_beautiful_arrangements_l757_757865

noncomputable def count_beautiful_arrangements : ℕ :=
  let total_ways := Nat.choose 10 5
  let invalid_ways := Nat.choose 9 4
  total_ways - invalid_ways

theorem number_of_beautiful_arrangements : count_beautiful_arrangements = 126 := by
  sorry

end number_of_beautiful_arrangements_l757_757865


namespace number_of_five_ruble_coins_l757_757331

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757331


namespace midpoint_on_hyperbola_l757_757138

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l757_757138


namespace calculate_hidden_dots_l757_757567

def sum_faces_of_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice : ℕ := 4
def total_sum_of_dots : ℕ := number_of_dice * sum_faces_of_die

def visible_faces : List (ℕ × String) :=
  [(1, "red"), (1, "none"), (2, "none"), (2, "blue"),
   (3, "none"), (4, "none"), (5, "none"), (6, "none")]

def adjust_face_value (value : ℕ) (color : String) : ℕ :=
  match color with
  | "red" => 2 * value
  | "blue" => 2 * value
  | _ => value

def visible_sum : ℕ :=
  visible_faces.foldl (fun acc (face) => acc + adjust_face_value face.1 face.2) 0

theorem calculate_hidden_dots :
  (total_sum_of_dots - visible_sum) = 57 :=
sorry

end calculate_hidden_dots_l757_757567


namespace hyperbola_midpoint_l757_757152

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757152


namespace total_students_began_contest_l757_757689

theorem total_students_began_contest (students_after_second_round : ℕ)
  (half_remaining_after_first_round : ℚ)
  (third_remaining_after_second_round : ℚ)
  (students_after_second_round_value : students_after_second_round = 24)
  (half_remaining_after_first_round_value : half_remaining_after_first_round = 1/2)
  (third_remaining_after_second_round_value : third_remaining_after_second_round = 1/3)
  (students_still_in_contest_after_both_rounds : half_remaining_after_first_round * third_remaining_after_second_round * (students_after_second_round * 6) = students_after_second_round)
  : ∃ total_students : ℕ, total_students = 144 :=
by
  existsi (students_after_second_round * 6)
  have h : 24 * 6 = 144 := by norm_num
  rw students_after_second_round_value at h
  exact h

end total_students_began_contest_l757_757689


namespace midpoint_hyperbola_l757_757103

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757103


namespace tyrone_gave_25_marbles_l757_757413

/-- Given that Tyrone initially had 97 marbles and Eric had 11 marbles, and after
    giving some marbles to Eric, Tyrone ended with twice as many marbles as Eric,
    we need to find the number of marbles Tyrone gave to Eric. -/
theorem tyrone_gave_25_marbles (x : ℕ) (t0 e0 : ℕ)
  (hT0 : t0 = 97)
  (hE0 : e0 = 11)
  (hT_end : (t0 - x) = 2 * (e0 + x)) :
  x = 25 := 
  sorry

end tyrone_gave_25_marbles_l757_757413


namespace find_f_63_l757_757377

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x y : ℝ, f(x * y) = x * f(y)
axiom cond2 : f(1) = 10
axiom cond3 : f(0) = 0

theorem find_f_63 : f(63) = 630 :=
by 
  -- proof
  sorry

end find_f_63_l757_757377


namespace can_be_midpoint_of_AB_l757_757188

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l757_757188


namespace midpoint_on_hyperbola_l757_757040

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l757_757040


namespace midpoint_of_hyperbola_segment_l757_757086

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757086


namespace midpoint_of_hyperbola_segment_l757_757097

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l757_757097


namespace only_square_with_two_nonzero_digits_one_being_three_l757_757543

theorem only_square_with_two_nonzero_digits_one_being_three :
  ∀ n : ℕ, (n > 0) → (has_square_with_digits n) → (has_two_nonzero_digits n) → (contains_digit n 3) → (n = 36) :=
by
  sorry

end only_square_with_two_nonzero_digits_one_being_three_l757_757543


namespace number_of_five_ruble_coins_l757_757327

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l757_757327


namespace hyperbola_midpoint_l757_757147

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l757_757147


namespace correct_midpoint_l757_757214

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l757_757214


namespace length_IK_greater_than_JK_l757_757737

/-- Definitions for the elements and problem-specific conditions -/
def is_center_of_inscribed_sphere (I : Point) (A B C D : Point) : Prop := sorry
def intersection_of_planes (J : Point) (A B C D : Point) : Prop := sorry
def intersection_with_circumscribed_sphere (K I J : Point) (A B C D : Point) : Prop := sorry

/-- Main theorem that needs proof -/
theorem length_IK_greater_than_JK
  (I J K A B C D : Point)
  (h1 : is_center_of_inscribed_sphere I A B C D)
  (h2 : intersection_of_planes J A B C D)
  (h3 : intersection_with_circumscribed_sphere K I J A B C D) :
  distance I K > distance J K := by
  sorry

end length_IK_greater_than_JK_l757_757737


namespace ordered_64_tuple_count_l757_757550

noncomputable def count_tuples : Nat :=
  (Finset.range (2017 - 64 + 1)).prod (λ k, 2017 - k)

theorem ordered_64_tuple_count :
  ∃ T : Nat, T = count_tuples :=
begin
  use (Finset.range (2017 - 64 + 1)).prod (λ k, 2017 - k),
  refl,
end

end ordered_64_tuple_count_l757_757550


namespace solve_inequality_l757_757775

def polynomial_fraction (x : ℝ) : ℝ :=
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem solve_inequality (x : ℝ) :
  -2 < polynomial_fraction x ∧ polynomial_fraction x < 2 ↔ 11.57 < x :=
sorry

end solve_inequality_l757_757775


namespace total_age_10_years_from_now_is_75_l757_757813

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end total_age_10_years_from_now_is_75_l757_757813


namespace rationalize_product_l757_757762

theorem rationalize_product (A B C : ℤ) (h : (A : ℚ) + B * real.sqrt C = (2 + real.sqrt 5 : ℚ) / (3 - real.sqrt 5)) :
  A * B * C = 50 :=
by {
  -- We would typically provide a detailed proof here, but to ensure the code can be built successfully:
  sorry
}

end rationalize_product_l757_757762


namespace largest_four_digit_multiple_of_9_with_digit_sum_27_l757_757831

theorem largest_four_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 0 ∧ (nat.digits 10 n).sum = 27 ∧
  ∀ (m : ℕ), 1000 ≤ m ∧ m < 10000 ∧ m % 9 = 0 ∧ (nat.digits 10 m).sum = 27 → n ≥ m :=
sorry

end largest_four_digit_multiple_of_9_with_digit_sum_27_l757_757831


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757287

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l757_757287


namespace midpoint_hyperbola_l757_757109

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l757_757109


namespace farmer_john_pairs_l757_757542

noncomputable def farmer_john_animals_pairing :
    Nat := 
  let cows := 5
  let pigs := 4
  let horses := 7
  let num_ways_cow_pig_pair := cows * pigs
  let num_ways_horses_remaining := Nat.factorial horses
  num_ways_cow_pig_pair * num_ways_horses_remaining

theorem farmer_john_pairs : farmer_john_animals_pairing = 100800 := 
by
  sorry

end farmer_john_pairs_l757_757542


namespace cost_of_candies_l757_757869

variable (cost_per_box : ℚ) (candies_per_box : ℕ) (total_candies : ℕ)
variables (h1 : cost_per_box = 7.5) (h2 : candies_per_box = 30) (h3 : total_candies = 450)

theorem cost_of_candies : 15 * 7.50 = 112.50 :=
by
  have boxes_needed : ℚ := total_candies / candies_per_box
  have total_cost : ℚ := boxes_needed * cost_per_box
  show total_cost = 112.50
  sorry

end cost_of_candies_l757_757869


namespace person_cannot_catch_up_l757_757884

theorem person_cannot_catch_up (v_person : ℝ) (d_initial : ℝ) (a_car : ℝ) :
  v_person = 6 → d_initial = 25 → a_car = 1 →
  let t := (List.solveQuadratic 1 (-12) 50).head
  in t.is_none ∧
  (∃ t : ℝ, 0 ≤ t ∧ let d_person := v_person * t in
                 let d_car := 25 + (1/2) * a_car * t^2 in
                 abs (d_person - d_car) = 7) :=
by 
  intros hp h₀ ha
  simp only [hp, h₀, ha]
  let t := (List.solveQuadratic 1 (-12) 50).head
  rw List.solveQuadratic at t
  split
  { simp [t] }, 
  existsi (6 : ℝ) 
  simp
  sorry

end person_cannot_catch_up_l757_757884


namespace percentage_increase_l757_757681

theorem percentage_increase (Z Y X : ℝ) (h1 : Y = 1.20 * Z) (h2 : Z = 250) (h3 : X + Y + Z = 925) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_l757_757681


namespace number_of_boys_decreased_l757_757540

variable (m : ℝ) -- initial number of boys
variable (total_students_initial : ℝ)
variable (total_students_final : ℝ)
variable (boys_final : ℝ)

-- Given conditions
def condition1 : Prop :=
  total_students_initial = 2 * m

def condition2 : Prop :=
  total_students_final = 0.9 * total_students_initial

def condition3 : Prop :=
  boys_final = 0.55 * total_students_final

-- Goal: Prove that the number of boys decreased
theorem number_of_boys_decreased (h1 : condition1) (h2 : condition2) (h3 : condition3) : boys_final < m :=
  by
  sorry

end number_of_boys_decreased_l757_757540


namespace parabola_focus_l757_757987

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l757_757987


namespace max_balls_in_cube_l757_757832

noncomputable def volume_of_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * real.pi * (r ^ 3)

theorem max_balls_in_cube :
  let s := 9
  let r := 3
  let V_cube := volume_of_cube s
  let V_sphere := volume_of_sphere r
  ⌊V_cube / V_sphere⌋ = 6 := by
    sorry

end max_balls_in_cube_l757_757832
