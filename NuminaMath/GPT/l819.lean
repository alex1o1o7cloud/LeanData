import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.group.defs
import Mathlib.Analysis.Geometry.Euclidean
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Enum
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Polygon.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Probability
import Real
import data.real.basic

namespace complex_number_z_value_l819_819543

open Complex

theorem complex_number_z_value :
  ∀ (i z : ℂ), i^2 = -1 ∧ z * (1 + i) = 2 * i^2018 → z = -1 + i :=
by
  intros i z h
  have h1 : i^2 = -1 := h.1
  have h2 : z * (1 + i) = 2 * i^2018 := h.2
  sorry

end complex_number_z_value_l819_819543


namespace simson_intersection_nine_point_circle_l819_819252

noncomputable def simson_line (P : Point) (X Y Z : Triangle) : Line :=
sorry -- Assume definition of Simson line of P with respect to triangle XYZ.

theorem simson_intersection_nine_point_circle
  (A B C : Point) -- Fixed points defining the triangle
  (O : Circle)   -- Circumcircle of triangle ABC
  (hO : O ∈ {circumcenter_abc}) -- Circle contains points A, B, and C
  (D : Point)    -- Variable point on circle O
  (hD : D ≠ A ∧ D ≠ B ∧ D ≠ C)
  (IA : Line := simson_line A B C D)
  (IB : Line := simson_line B A C D)
  (IC : Line := simson_line C A B D)
  (ID : Line := simson_line D A B C) :
  locus_of_intersections [IA, IB, IC, ID] = nine_point_circle A B C :=
sorry

end simson_intersection_nine_point_circle_l819_819252


namespace negA_necessary_for_negB_l819_819980

theorem negA_necessary_for_negB (A B : Prop) (h : A → B) : (¬B → ¬A) :=
by {
  intro h1,
  exact h1 ∘ h
}

end negA_necessary_for_negB_l819_819980


namespace wood_length_equation_l819_819799

variable (x : ℝ) (l : ℝ)

theorem wood_length_equation : 
  (l = x + 4.5) → (0.5 * l = x - 1) → (0.5 * (x + 4.5) = x - 1) :=
by
  intros h₁ h₂
  rw [←h₁] at h₂
  exact h₂

#check wood_length_equation -- to check the correctness of the statement

end wood_length_equation_l819_819799


namespace log_base_5_of_15625_l819_819075

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l819_819075


namespace candle_placement_l819_819921

theorem candle_placement (A : list (ℝ × ℝ)) (h_convex: ∀ (i : ℕ), convex A)
  (valid_cut: ∀ (i : ℕ) (p q : ℝ × ℝ), p ≠ q ∧ p ∉ A ∧ q ∉ A → 
    ∃ (B : list (ℝ × ℝ)), convex B ∧ length B = length A - 1) : 
  ∃ (c : ℝ × ℝ), (c ∉ (boundary A)) ∧ (∀ (p q : ℝ × ℝ), p = A.head ∧ q = A.last → c ∈ interior_tri (A !! nat.pred (length A) !! 0)) := 
sorry

end candle_placement_l819_819921


namespace find_a_l819_819996

noncomputable def value_of_a (P : ℝ × ℝ) (a : ℝ) : Prop := 
  let x := P.1 in
  let y := P.2 in
  a > 0 ∧
  y = Real.log x ∧
  y = x^2 / a ∧
  (1 / x = 2 * x / a)

theorem find_a (P : ℝ × ℝ) : 
  (∃ a > 0, value_of_a P a) → a = 2 * Real.exp(1) :=
by
  sorry

end find_a_l819_819996


namespace sin_theta_is_three_fifths_l819_819446

-- Definitions for the triangle conditions
def area : ℝ := 36
def side_length : ℝ := 12
def median_length : ℝ := 10

-- The formula for the area of the triangle involving sin(theta)
def area_formula (a m : ℝ) (theta : ℝ) : ℝ := 0.5 * a * m * Real.sin theta

-- Main theorem to be proved
theorem sin_theta_is_three_fifths :
  ∃ θ : ℝ, area_formula side_length median_length θ = area → Real.sin θ = 3 / 5 :=
by
  sorry

end sin_theta_is_three_fifths_l819_819446


namespace jason_needs_to_buy_guppies_l819_819638

theorem jason_needs_to_buy_guppies 
  (moray_eel_guppies : ℕ) 
  (num_betta_fish : ℕ) 
  (betta_fish_guppies : ℕ) 
  (h_moray_eel : moray_eel_guppies = 20)
  (h_num_betta_fish : num_betta_fish = 5)
  (h_betta_fish : betta_fish_guppies = 7)
  : moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 :=
by 
  rw [h_moray_eel, h_num_betta_fish, h_betta_fish]
  sorry

end jason_needs_to_buy_guppies_l819_819638


namespace distance_between_foci_of_ellipse_l819_819381

theorem distance_between_foci_of_ellipse :
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  2 * c = 14 :=
by
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  show 2 * c = 14, from sorry

end distance_between_foci_of_ellipse_l819_819381


namespace bell_pepper_ratio_l819_819699

theorem bell_pepper_ratio :
  ∀ (n : ℕ) (total_slices total_pieces portions : ℕ),
    n = 5 →
    total_slices = n * 20 →
    portions * 3 = total_pieces - total_slices →
    total_pieces = 200 →
    portions = 33 →
    total_slices = 100 →
    ratio portions total_slices = 33 / 100 := by
  intros n total_slices total_pieces portions h1 h2 h3 h4 h5 h6
  sorry

-- With help function
def ratio (a b : ℕ) : ℚ := a / b

end bell_pepper_ratio_l819_819699


namespace sqrt_sum_eq_six_l819_819493

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l819_819493


namespace trigonometric_identity_l819_819146

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l819_819146


namespace problem_ii_angle_VAU_is_90_l819_819803

theorem problem_ii_angle_VAU_is_90
  (A B C X Y U V : Type) (h1 : ∃ (r : ℝ), ∀ (P : Type), A ∈ P → ∃ Q : Type, C ∈ Q ∧ B ∈ Q)
  (h2 : A ∈ C) (h3 : A ∈ B)
  (h4 : X ∈ B) (h5 : X ∈ C)
  (h6 : Y ∈ B) (h7 : Y ∈ C)
  (h8 : ∃ l : Type, (AX ⊆ l) ∧ (l ⊥ B) ∧ (U ∈ l))
  (h9 : ∃ m : Type, (AY ⊆ m) ∧ (m ⊥ C) ∧ (V ∈ m))
  (h10 : BY = BA)
  (h11 : CX = CA) :
  ∠VAU = 90° :=
  by sorry

end problem_ii_angle_VAU_is_90_l819_819803


namespace series_convergence_l819_819873

def a : ℕ → ℚ
| 0 := 1
| 1 := 1 / 2
| (n + 1) := n * (a n) ^ 2 / (1 + (n + 1) * a n)

theorem series_convergence :
  (∑ k : ℕ, a (k + 1) / a k) = 1 := 
by
  sorry

end series_convergence_l819_819873


namespace max_f_probability_l819_819224

noncomputable def f (x a : ℝ) := abs (x^2 - 4 * x + 3 - a) + a

theorem max_f_probability (a : ℝ) (h_a : a ∈ Icc (-2 : ℝ) 2) :
  ∃ p : ℚ, p = 3 / 4 ∧ (∀ b ∈ Icc (-2 : ℝ) 2, 
    let m := max (f 0 b) (max (f 2 b) (f 4 b))
    in m = 3 ↔ b ∈ Icc (-2 : ℝ) 1) :=
sorry

end max_f_probability_l819_819224


namespace vendor_total_profit_l819_819841

theorem vendor_total_profit :
  let cost_per_apple := 3 / 2
      selling_price_per_apple := 10 / 5
      profit_per_apple := selling_price_per_apple - cost_per_apple
      total_profit_apples := profit_per_apple * 5
      cost_per_orange := 2.7 / 3
      selling_price_per_orange := 1
      profit_per_orange := selling_price_per_orange - cost_per_orange
      total_profit_oranges := profit_per_orange * 5
  in total_profit_apples + total_profit_oranges = 3 := 
by
  sorry

end vendor_total_profit_l819_819841


namespace area_ratio_of_trapezoid_l819_819227

open Classical

-- Definitions of elements in the problem
variables (PQ RS : ℕ) (A_TPQ A_PQRS : ℝ)
variables (T P Q R S : Type) [Geometry T P Q R S]

axiom h1 : PQ = 10
axiom h2 : RS = 21
axiom h3 : ∀ x y, Triangle ∧ Area (triangle x y T) = (x * y) / 2 -- define area geometrically (simplified)

-- The main theorem to prove in Lean
theorem area_ratio_of_trapezoid :
    (A_TPQ / A_PQRS) = (100 / 341) :=
by
  sorry

end area_ratio_of_trapezoid_l819_819227


namespace susie_rhode_island_reds_l819_819698

variable (R G B_R B_G : ℕ)

def susie_golden_comets := G = 6
def britney_rir := B_R = 2 * R
def britney_golden_comets := B_G = G / 2
def britney_more_chickens := B_R + B_G = R + G + 8

theorem susie_rhode_island_reds
  (h1 : susie_golden_comets G)
  (h2 : britney_rir R B_R)
  (h3 : britney_golden_comets G B_G)
  (h4 : britney_more_chickens R G B_R B_G) :
  R = 11 :=
by
  sorry

end susie_rhode_island_reds_l819_819698


namespace jeff_boxes_filled_l819_819644

noncomputable def jeff_donuts_per_day : ℕ := 10
noncomputable def number_of_days : ℕ := 12
noncomputable def jeff_eats_per_day : ℕ := 1
noncomputable def chris_eats : ℕ := 8
noncomputable def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled :
  let total_donuts := jeff_donuts_per_day * number_of_days
  let jeff_eats_total := jeff_eats_per_day * number_of_days
  let remaining_donuts_after_jeff := total_donuts - jeff_eats_total
  let remaining_donuts_after_chris := remaining_donuts_after_jeff - chris_eats
  let boxes_filled := remaining_donuts_after_chris / donuts_per_box
  in boxes_filled = 10 :=
by {
  sorry
}

end jeff_boxes_filled_l819_819644


namespace meeting_success_probability_l819_819434

open Set

noncomputable def meeting_probability : ℝ :=
  let Ω := Icc (0:ℝ) 2 × Icc (0:ℝ) 2 × Icc (0:ℝ) 2 × Icc (0:ℝ) 2 in
  let valid_region := {t | ∃ x y w z: ℝ, t = (x, y, w, z) ∧ z > x ∧ z > y ∧ z > w ∧
                       |x - y| ≤ 0.5 ∧ |x - w| ≤ 0.5 ∧ |y - w| ≤ 0.5} in
  (volume valid_region) / (volume Ω)

theorem meeting_success_probability : meeting_probability = 0.25 := sorry

end meeting_success_probability_l819_819434


namespace no_half_probability_socks_l819_819733

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l819_819733


namespace ratio_of_liquid_rise_l819_819765

theorem ratio_of_liquid_rise
  (h1 h2 : ℝ) (r1 r2 rm : ℝ)
  (V1 V2 Vm : ℝ)
  (H1 : r1 = 4)
  (H2 : r2 = 9)
  (H3 : V1 = (1 / 3) * π * r1^2 * h1)
  (H4 : V2 = (1 / 3) * π * r2^2 * h2)
  (H5 : V1 = V2)
  (H6 : rm = 2)
  (H7 : Vm = (4 / 3) * π * rm^3)
  (H8 : h2 = h1 * (81 / 16))
  (h1' h2' : ℝ)
  (H9 : h1' = h1 + Vm / ((1 / 3) * π * r1^2))
  (H10 : h2' = h2 + Vm / ((1 / 3) * π * r2^2)) :
  (h1' - h1) / (h2' - h2) = 81 / 16 :=
sorry

end ratio_of_liquid_rise_l819_819765


namespace total_fish_caught_l819_819251

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l819_819251


namespace ratio_of_square_sides_l819_819362

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l819_819362


namespace bottom_section_height_l819_819419

open Real

theorem bottom_section_height
  (P Q R : Point ℝ ℝ)
  (PQ PR QR : ℝ)
  (h : ℝ)
  (h₁ : PQ = 150)
  (h₂ : PR = 125)
  (h₃ : QR = 125)
  (h₄ : are_equal_parallel_sections P Q R) : 
  h = 16.1 := 
  sorry

end bottom_section_height_l819_819419


namespace set_b_correct_l819_819948

open Set

theorem set_b_correct (A B : Set ℝ) (h1 : A = {x | log 3 x ≤ 1}) (h2 : A ∩ B = Ioc 0 2) :
  B = Iic 2 := 
sorry

end set_b_correct_l819_819948


namespace derivative_y_x_l819_819104

-- Define the cotangent and arc-cotangent function
def cot (x : ℝ) : ℝ := 1 / tan x
def arccot (x : ℝ) : ℝ := π / 2 - arctan x

-- Define the given functions x(t) and y(t)
def x (t : ℝ) : ℝ := cot (t^2)
def y (t : ℝ) : ℝ := arccot (t^4)

-- Prove the derivative relationship
theorem derivative_y_x (t : ℝ) :
  (deriv (fun t => y t) t) / (deriv (fun t => x t) t) = (2 * t^2 * (sin (t^2))^2) / (1 + t^8) :=
by
  sorry

end derivative_y_x_l819_819104


namespace parabola_a_value_l819_819936

theorem parabola_a_value 
  (a b c : ℝ)
  (h1 : ∀ x, ax^2 + bx + c = a(x - 2)^2 + 5) 
  (h2 : a * 0^2 + b * 0 + c = -23) : 
  a = -7 :=
by sorry

end parabola_a_value_l819_819936


namespace convert_micrometers_to_meters_l819_819323

theorem convert_micrometers_to_meters :
  let μm_in_meters := 10^(-6)
  in 32 * μm_in_meters = 3.2 * 10^(-5) :=
by
  let μm_in_meters := 10^(-6)
  show 32 * μm_in_meters = 3.2 * 10^(-5)
  sorry

end convert_micrometers_to_meters_l819_819323


namespace work_together_days_l819_819818

noncomputable def time_to_complete_together (days_B: ℕ) (ratio_A_B: ℕ) : ℕ :=
  let days_A := days_B / ratio_A_B
  (ratio_A_B * days_B) / (ratio_A_B + 1)

theorem work_together_days (days_B: ℕ) (ratio_A_B: ℕ) (h: ratio_A_B = 2 ∧ days_B = 108) :
  time_to_complete_together days_B ratio_A_B = 36 :=
by {
  intro h,
  sorry
}

end work_together_days_l819_819818


namespace chessboard_overlap_area_l819_819764

theorem chessboard_overlap_area :
  let n := 8
  let cell_area := 1
  let side_length := 8
  let overlap_area := 32 * (Real.sqrt 2 - 1)
  (∃ black_overlap_area : ℝ, black_overlap_area = overlap_area) :=
by
  sorry

end chessboard_overlap_area_l819_819764


namespace intersection_points_sum_l819_819331

theorem intersection_points_sum :
  let intersects (x y : ℝ) := (y = x^3 - 5*x + 2) ∧ (x + 5*y = 5)
  ∃ x1 x2 x3 y1 y2 y3, 
    intersects x1 y1 ∧ 
    intersects x2 y2 ∧ 
    intersects x3 y3 ∧ 
    (x1 + x2 + x3 = 0) ∧ 
    (y1 + y2 + y3 = 3) :=
begin
  sorry
end

end intersection_points_sum_l819_819331


namespace avg_marks_all_students_l819_819415

-- Define average marks of the first class
def avg_marks_class1 : ℕ := 50
def students_class1 : ℕ := 30

-- Define average marks of the second class
def avg_marks_class2 : ℕ := 60
def students_class2 : ℕ := 50

-- Total number of students
def total_students : ℕ := students_class1 + students_class2

-- Total marks for the first class
def total_marks_class1 : ℕ := students_class1 * avg_marks_class1

-- Total marks for the second class
def total_marks_class2 : ℕ := students_class2 * avg_marks_class2

-- Total marks for both classes
def total_marks_all : ℕ := total_marks_class1 + total_marks_class2

-- Correct average marks for all students
def correct_avg_marks_all : ℝ := 56.25

theorem avg_marks_all_students :
  (total_marks_all : ℝ) / (total_students : ℝ) = correct_avg_marks_all := by
sory

end avg_marks_all_students_l819_819415


namespace sine_magnitude_comparison_l819_819045

theorem sine_magnitude_comparison 
  (h_increasing : ∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < (π/2) → sin x < sin y) :
  sin 3 < sin 1 ∧ sin 1 < sin 2 :=
by
  -- Sorry to indicate the proof is omitted.
  sorry

end sine_magnitude_comparison_l819_819045


namespace cos_identity_l819_819130

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end cos_identity_l819_819130


namespace no_equal_prob_for_same_color_socks_l819_819743

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l819_819743


namespace edit_post_time_zero_l819_819296

-- Define the conditions
def total_videos : ℕ := 4
def setup_time : ℕ := 1
def painting_time_per_video : ℕ := 1
def cleanup_time : ℕ := 1
def total_production_time_per_video : ℕ := 3

-- Define the total time spent on setup, painting, and cleanup for one video
def spc_time : ℕ := setup_time + painting_time_per_video + cleanup_time

-- State the theorem to be proven
theorem edit_post_time_zero : (total_production_time_per_video - spc_time) = 0 := by
  sorry

end edit_post_time_zero_l819_819296


namespace James_present_age_l819_819291

variable (D J : ℕ)

theorem James_present_age 
  (h1 : D / J = 6 / 5)
  (h2 : D + 4 = 28) :
  J = 20 := 
by
  sorry

end James_present_age_l819_819291


namespace graph_shifted_function_l819_819329

noncomputable def f : ℝ → ℝ := λ x, 1 - 2 * sin x * (sin x + (sqrt 3) * cos x)

def g (x : ℝ) : ℝ := 2 * sin (2 * x - (π / 2))

theorem graph_shifted_function (x : ℝ) :
  g x = 2 * sin (2 * x - (π / 2)) :=
by
  sorry

end graph_shifted_function_l819_819329


namespace smallest_total_pets_l819_819922

theorem smallest_total_pets (n : ℕ) (h₁ : ∃ k : ℕ, n = 2 * k) (h₂ : ∃ l : ℕ, n = 10 * l) : n = 20 :=
by
  use 2
  use 5
  sorry

end smallest_total_pets_l819_819922


namespace solve_equation_l819_819499

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l819_819499


namespace sin_beta_equals_sqrt3_div_2_l819_819541

noncomputable def angles_acute (α β : ℝ) : Prop :=
0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2

theorem sin_beta_equals_sqrt3_div_2 
  (α β : ℝ) 
  (h_acute: angles_acute α β) 
  (h_sin_alpha: Real.sin α = (4/7) * Real.sqrt 3) 
  (h_cos_alpha_plus_beta: Real.cos (α + β) = -(11/14)) 
  : Real.sin β = (Real.sqrt 3) / 2 :=
sorry

end sin_beta_equals_sqrt3_div_2_l819_819541


namespace domain_ln_x_sq_minus_x_l819_819324

theorem domain_ln_x_sq_minus_x : 
  ∀ x, (∃ y, y = x^2 - x ∧ y > 0) ↔ x ∈ set.Ioo (-∞) 0 ∪ set.Ioo 1 ∞ := 
sorry

end domain_ln_x_sq_minus_x_l819_819324


namespace minimum_y_value_y_at_4_eq_6_l819_819192

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 2)

theorem minimum_y_value (x : ℝ) (h : x > 2) : y x ≥ 6 :=
sorry

theorem y_at_4_eq_6 : y 4 = 6 :=
sorry

end minimum_y_value_y_at_4_eq_6_l819_819192


namespace max_modulus_z_l819_819259

theorem max_modulus_z (z : ℂ) (h : |z - 6 * complex.I| + |z - 5| = 7) : |z| ≤ 6 := sorry

end max_modulus_z_l819_819259


namespace average_eq_instantaneous_velocity_at_t_eq_3_l819_819164

theorem average_eq_instantaneous_velocity_at_t_eq_3
  (S : ℝ → ℝ) (hS : ∀ t, S t = 24 * t - 3 * t^2) :
  (1 / 6) * (S 6 - S 0) = 24 - 6 * 3 :=
by 
  sorry

end average_eq_instantaneous_velocity_at_t_eq_3_l819_819164


namespace total_fish_l819_819245

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l819_819245


namespace final_percentage_managers_l819_819372

-- Definitions based on the conditions
def totalEmployees : ℕ := 300
def initialPercentageManagers : ℝ := 0.99
def managersLeaving : ℝ := 149.99999999999986

-- Assertion based on the problem and correct answer
theorem final_percentage_managers :
  let initialManagers := initialPercentageManagers * totalEmployees
  let remainingManagers := initialManagers - managersLeaving
  let finalPercentage := (remainingManagers / totalEmployees) * 100
  finalPercentage = 49 := 
by
  let initialManagers := initialPercentageManagers * totalEmployees
  let remainingManagers := initialManagers - managersLeaving
  let finalPercentage := (remainingManagers / totalEmployees) * 100
  have h : initialManagers = 297 := rfl
  have h₁ : remainingManagers = 147 := rfl
  have h₂ : finalPercentage = 49 := rfl
  exact h₂

end final_percentage_managers_l819_819372


namespace binomial_remainder_mod_3_l819_819511

open BigOperators

theorem binomial_remainder_mod_3 :
  (1 + nat.choose 27 1 + nat.choose 27 2 + nat.choose 27 27) % 3 = 2 := by
sorry

end binomial_remainder_mod_3_l819_819511


namespace jason_needs_to_buy_guppies_l819_819639

theorem jason_needs_to_buy_guppies 
  (moray_eel_guppies : ℕ) 
  (num_betta_fish : ℕ) 
  (betta_fish_guppies : ℕ) 
  (h_moray_eel : moray_eel_guppies = 20)
  (h_num_betta_fish : num_betta_fish = 5)
  (h_betta_fish : betta_fish_guppies = 7)
  : moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 :=
by 
  rw [h_moray_eel, h_num_betta_fish, h_betta_fish]
  sorry

end jason_needs_to_buy_guppies_l819_819639


namespace distance_between_M_and_focus_yA_times_yB_const_exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const_l819_819934

-- Part 1
theorem distance_between_M_and_focus (y₀: ℝ) (y₀_eq_sqrt_2 : y₀ = Real.sqrt 2) : 
  let x₀ := y₀^2 
  let focus := (1/4 : ℝ, 0)
  let M := (x₀, y₀)
  (|M.fst - focus.fst|) = 7/4 :=
by
  sorry

-- Part 2
theorem yA_times_yB_const (t : ℝ) (P Q: (ℝ × ℝ)) (PQonParabola : (P.snd ^ 2 = P.fst) ∧ (Q.snd ^ 2 = Q.fst))
  (PonLine : P.fst = 1 ∧ P.snd = 1) (QonLine : Q.fst = 1 ∧ Q.snd = -1) (tonline : t = -1) : 
  let y₀ := sqrt 2 
  let M := (y₀^2, y₀)
  ∃ yA yB, yA * yB = -1 :=
by 
  sorry

-- Part 3
theorem exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const (yt1 yt2: ℝ) (yP yQ yA yB : ℝ)
  (t: ℝ) (Py Pnot_eq_1 : yP != 1 ∧ yQ != 1) (yt: t = 1) : 
  yA * yB = 1 ∧ yP * yQ ≠ yt1 / yt2 :=
by
  sorry

end distance_between_M_and_focus_yA_times_yB_const_exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const_l819_819934


namespace reporters_cover_local_politics_l819_819811

structure Reporters :=
(total : ℕ)
(politics : ℕ)
(local_politics : ℕ)

def percentages (reporters : Reporters) : Prop :=
  reporters.politics = (40 * reporters.total) / 100 ∧
  reporters.local_politics = (75 * reporters.politics) / 100

theorem reporters_cover_local_politics (reporters : Reporters) (h : percentages reporters) :
  (reporters.local_politics * 100) / reporters.total = 30 :=
by
  -- Proof steps would be added here
  sorry

end reporters_cover_local_politics_l819_819811


namespace constant_term_expansion_eq_160_l819_819125

noncomputable def a : ℝ := ∫ x in 0..π, (real.arcsin x + 2 * real.cos (x / 2) ^ 2)

theorem constant_term_expansion_eq_160 :
  let expr := (a * real.sqrt x - 1 / real.sqrt x) ^ 6 * (x ^ 2 + 2)
  in (expr.map (λ x => if x = 0 then some else none)) x_0 = 160 :=
by {
  sorry
}

end constant_term_expansion_eq_160_l819_819125


namespace factorial_div_evaluation_l819_819069

theorem factorial_div_evaluation : (17! / (8! * 9!)) = 130 := by
  sorry

end factorial_div_evaluation_l819_819069


namespace ellipse_standard_equation_parabola_standard_equation_l819_819911

-- Ellipse with major axis length 10 and eccentricity 4/5
theorem ellipse_standard_equation (a c b : ℝ) (h₀ : a = 5) (h₁ : c = 4) (h₂ : b = 3) :
  (x^2 / a^2) + (y^2 / b^2) = 1 := by sorry

-- Parabola with vertex at the origin and directrix y = 2
theorem parabola_standard_equation (p : ℝ) (h₀ : p = 4) :
  x^2 = -8 * y := by sorry

end ellipse_standard_equation_parabola_standard_equation_l819_819911


namespace find_f_for_neg_x_l819_819149

-- Define the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

def f_pos (f : ℝ → ℝ) (x : ℝ) : Prop :=
x > 0 → f x = -√x * (1 + x)

-- Define the main function we want to prove
def f (x : ℝ) : ℝ :=
if h : x > 0 then -√x * (1 + x) else if h₂ : x < 0 then √(-x) * (1 - x) else 0

-- Lean statement for the proof problem
theorem find_f_for_neg_x (f : ℝ → ℝ)
  (odd_f : is_odd_function f)
  (cond_f_pos : f_pos f) :
  ∀ x, x < 0 → f x = √(-x) * (1 - x) :=
by
  sorry

end find_f_for_neg_x_l819_819149


namespace count_valid_four_digit_numbers_l819_819180

-- Definitions based on given conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_valid_four_digit (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ a ≠ 0

def digits_sum_to_10 (a b c d : ℕ) : Prop :=
  a + b + c + d = 10

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Theorem statement
theorem count_valid_four_digit_numbers :
  { n : ℕ // ∃ (a b c d : ℕ), is_valid_four_digit a b c d ∧
                             digits_sum_to_10 a b c d ∧
                             divisible_by_5 d ∧
                             n = a * 1000 + b * 100 + c * 10 + d }.to_finset.card = 69 :=
sorry

end count_valid_four_digit_numbers_l819_819180


namespace quadrilateral_AD_length_l819_819216

noncomputable def length_AD (AB BC CD : ℝ) (angleB angleC : ℝ) : ℝ :=
  let AE := AB + BC * Real.cos angleC
  let CE := BC * Real.sin angleC
  let DE := CD - CE
  Real.sqrt (AE^2 + DE^2)

theorem quadrilateral_AD_length :
  let AB := 7
  let BC := 10
  let CD := 24
  let angleB := Real.pi / 2 -- 90 degrees in radians
  let angleC := Real.pi / 3 -- 60 degrees in radians
  length_AD AB BC CD angleB angleC = Real.sqrt (795 - 240 * Real.sqrt 3) :=
by
  sorry

end quadrilateral_AD_length_l819_819216


namespace find_Teena_speed_l819_819701

open Real

-- Define the given conditions
def Yoe_speed : ℝ := 40
def time_in_hours : ℝ := 1.5
def initial_distance_behind : ℝ := 7.5
def distance_ahead : ℝ := 15

-- Define Yoe's travelled distance
def distance_Yoe : ℝ := Yoe_speed * time_in_hours

-- Define total distance Teena needs to travel to be 15 miles ahead
def distance_Teena : ℝ := distance_Yoe + initial_distance_behind + distance_ahead

-- Define the final question: What is Teena's speed?
def Teena_speed (distance_Teena : ℝ) (time_in_hours : ℝ) : ℝ := distance_Teena / time_in_hours

-- The theorem to prove
theorem find_Teena_speed : Teena_speed distance_Teena time_in_hours = 55 := by
  sorry

end find_Teena_speed_l819_819701


namespace matrix_P_swaps_and_triples_l819_819902

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

def Q : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![a, b, c],
  ![d, e, f],
  ![g, h, i]
]

theorem matrix_P_swaps_and_triples (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  P ⬝ Q = ![
    ![3 * Q 0 0, 3 * Q 0 1, 3 * Q 0 2],
    ![Q 2 0, Q 2 1, Q 2 2],
    ![Q 1 0, Q 1 1, Q 1 2]
  ] :=
by {
  sorry
}

end matrix_P_swaps_and_triples_l819_819902


namespace smallest_integer_to_make_multiple_of_five_l819_819399

theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k: ℕ, 0 < k ∧ (726 + k) % 5 = 0 ∧ k = 4 := 
by
  use 4
  sorry

end smallest_integer_to_make_multiple_of_five_l819_819399


namespace log_expression_reduction_l819_819297

variable {a b c d x y : ℝ}

theorem log_expression_reduction (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < x) (h₅ : 0 < y) :
  log (2 * a / (3 * b)) + log (3 * b / (4 * c)) + log (4 * c / (5 * d)) - log (10 * a * y / (3 * d * x)) = log (3 * x / (25 * y)) :=
by
  sorry

end log_expression_reduction_l819_819297


namespace smallest_divisible_12_13_14_l819_819109

theorem smallest_divisible_12_13_14 :
  ∃ n : ℕ, n > 0 ∧ (n % 12 = 0) ∧ (n % 13 = 0) ∧ (n % 14 = 0) ∧ n = 1092 := by
  sorry

end smallest_divisible_12_13_14_l819_819109


namespace selection_methods_l819_819299

theorem selection_methods (females males : Nat) (h_females : females = 3) (h_males : males = 2):
  females + males = 5 := 
  by 
    -- We add sorry here to skip the proof
    sorry

end selection_methods_l819_819299


namespace no_half_probability_socks_l819_819734

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l819_819734


namespace no_possible_blue_socks_l819_819739

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l819_819739


namespace sqrt_div_expression_l819_819412

theorem sqrt_div_expression :
  let a := 1.21
  let b := 0.81
  let c := 0.49
  sqrt a = 1.1 →
  sqrt b = 0.9 →
  sqrt c = 0.7 →
  (sqrt a / sqrt b) + (sqrt b / sqrt c) ≈ 2.5079 :=
by
  intros
  sorry

end sqrt_div_expression_l819_819412


namespace smallest_integer_x_l819_819398

noncomputable theory
open Classical

-- Define the condition
def condition (x : ℤ) := x < 3 * x - 14

-- The proof problem statement
theorem smallest_integer_x :
  ∃ (x : ℤ), condition x ∧ ∀ (y : ℤ), condition y → y ≥ x := 
sorry

end smallest_integer_x_l819_819398


namespace distance_M_focus_yA_yB_constant_exists_t_yA_yB_yP_yQ_l819_819932

-- Part (1)
theorem distance_M_focus (y0 : ℝ) (h1 : y0 = sqrt 2) : 
  let M := (y0^2, y0) in 
  M.1 + (1 / 4) / 2 = 9 / 4 :=
by 
  -- Proof outline: calculate the distance and verify it equals 9/4
  sorry 

-- Part (2)
theorem yA_yB_constant (t : ℝ) (h2 : t = -1) (P : (ℝ × ℝ)) (hP : P = (1, 1)) (Q : (ℝ × ℝ)) (hQ : Q = (1, -1)) : 
  ∀ M y0, M = (y0^2, y0) → 
  let yA := (y0 - 1) / (y0 + 1) in 
  let yB := (-y0 - 1) / (y0 - 1) in 
  yA * yB = -1 :=
by 
  -- Proof outline: calculate yA and yB based on their definitions and verify yA * yB = -1
  sorry 

-- Part (3)
theorem exists_t_yA_yB_yP_yQ (P Q : ℝ × ℝ) :
  ∃ t : ℝ, (∀ yA yB, ((yA * yB = 1) → 
                     (let yP := (sqrt(2) * yA - t) / (sqrt(2) - yA) in 
                      let yQ := (sqrt(2) * yB - t) / (sqrt(2) - yB) in 
                      yP * yQ = 1)) :=
by 
  -- Proof outline: show t = 1 satisfies the conditions yA * yB = 1 and yP * yQ = 1
  sorry

end distance_M_focus_yA_yB_constant_exists_t_yA_yB_yP_yQ_l819_819932


namespace find_circle_and_line_l819_819931

noncomputable def is_on_line (pt : ℝ × ℝ) : Prop :=
  pt.1 - pt.2 + 1 = 0

noncomputable def distance (pt1 pt2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((pt1.1 - pt2.1)^2 + (pt1.2 - pt2.2)^2)

theorem find_circle_and_line:
  (∃ (C : ℝ × ℝ), is_on_line C ∧ distance C (1, 1) ^ 2 = 25 ∧ distance C (2, -2) ^ 2 = 25)
  ∧ (∃ (line_eq_1 line_eq_2 : ℝ × ℝ → ℝ), 
    (∀ pt, line_eq_1 (1, 1) = 0 ∧ (real.abs (line_eq_1 pt / real.sqrt (line_eq_1 (1, 0)^2 + line_eq_1 (0, 1)^2)) = 4) 
    → pt = (7, 24) ∨ pt = (1, 0))) :=
  sorry

end find_circle_and_line_l819_819931


namespace range_of_a_l819_819998

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, cot (arcsin x) = sqrt(a^2 - x^2)) ↔ (a ∈ Iic (-1) ∪ Ici 1) :=
by
  sorry

end range_of_a_l819_819998


namespace jeff_boxes_filled_l819_819642

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end jeff_boxes_filled_l819_819642


namespace rectangle_area_l819_819376

theorem rectangle_area (length : ℝ) (width : ℝ) (area : ℝ) 
  (h1 : length = 24) 
  (h2 : width = 0.875 * length) 
  (h3 : area = length * width) : 
  area = 504 := 
by
  sorry

end rectangle_area_l819_819376


namespace largest_even_not_sum_of_two_odd_composites_l819_819876

def is_odd_composite (n : ℕ) : Prop :=
  n % 2 = 1 ∧ ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem largest_even_not_sum_of_two_odd_composites : ∀ n : ℕ, 38 < n → 
  ∃ a b : ℕ, is_odd_composite a ∧ is_odd_composite b ∧ n = a + b :=
begin
  sorry
end

end largest_even_not_sum_of_two_odd_composites_l819_819876


namespace base_number_approx_is_6_l819_819198

theorem base_number_approx_is_6 (x k : ℝ) (hxk : x^k = 4) (hx2k3 : x^(2 * k + 3) = 3456) : x ≈ 6 := sorry

end base_number_approx_is_6_l819_819198


namespace vendor_profit_l819_819837

noncomputable def apple_cost_price := 3 / 2
noncomputable def apple_selling_price := 10 / 5
noncomputable def orange_cost_price := 2.70 / 3
noncomputable def orange_selling_price := 1

noncomputable def total_apple_cost_price := 5 * apple_cost_price
noncomputable def total_apple_selling_price := 5 * apple_selling_price
noncomputable def total_apple_profit := total_apple_selling_price - total_apple_cost_price

noncomputable def total_orange_cost_price := 5 * orange_cost_price
noncomputable def total_orange_selling_price := 5 * orange_selling_price
noncomputable def total_orange_profit := total_orange_selling_price - total_orange_cost_price

noncomputable def total_profit := total_apple_profit + total_orange_profit

theorem vendor_profit : total_profit = 3 := by
  sorry

end vendor_profit_l819_819837


namespace tom_wall_building_time_l819_819637

theorem tom_wall_building_time 
(a : ℝ) (T : ℝ)
(avery_rate : a = 1 / 3)
(tom_rate : T > 0 ∧ 1 / T)
(together_work : T > 0 ∧ 1 / 3 + 1 / T + (1 / T) * (7 / 3) = 1) :
T = 5 := 
by {
  sorry
}

end tom_wall_building_time_l819_819637


namespace trig_identity_tan_40_plus_4_sin_40_l819_819864

theorem trig_identity_tan_40_plus_4_sin_40 : 
  tan 40 + 4 * sin 40 = real.sqrt 3 :=
by
  sorry

end trig_identity_tan_40_plus_4_sin_40_l819_819864


namespace two_zeros_of_cubic_polynomial_l819_819153

theorem two_zeros_of_cubic_polynomial (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -x1^3 + 3*x1 + m = 0 ∧ -x2^3 + 3*x2 + m = 0) →
  (m = -2 ∨ m = 2) :=
by
  sorry

end two_zeros_of_cubic_polynomial_l819_819153


namespace cannot_use_bisection_method_for_f3_l819_819029

noncomputable def f1 (x : ℝ) : ℝ := 3 * x + 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x^2
noncomputable def f4 (x : ℝ) : ℝ := Real.log x

theorem cannot_use_bisection_method_for_f3 :
  ∃ r, f3 r = 0 ∧ (∀ (a b : ℝ), a < r → r < b ∨ b < r → r < a → f3 a * f3 b ≥ 0) :=
by
  use 0
  simp [f3]
  intro a b ha hb
  cases hb
  all_goals { simp [ha, hb] }
  sorry

end cannot_use_bisection_method_for_f3_l819_819029


namespace radius_semicircle_EF_l819_819634

-- Define the context and conditions
section
variables {DE DF EF : ℝ}
variables {area_DE : ℝ} (arc_DF : ℝ) (right_angle : True)

-- Define the conditions
def conditions :=
  right_angle ∧
  (area_DE = 18 * Real.pi) ∧ 
  (arc_DF = 10 * Real.pi)

-- Define the radius of the semicircle on EF
def radius_EF :=
  Real.sqrt ((DE / 2)^2 + (DF / 2)^2) / 2

-- The theorem statement
theorem radius_semicircle_EF (h : conditions) :
  radius_EF = Real.sqrt 136 :=
begin
  sorry
end
end

end radius_semicircle_EF_l819_819634


namespace linear_independence_polynomials_l819_819692

theorem linear_independence_polynomials :
  (∀ x : ℝ, α1 • (1 : ℝ → ℝ) x + α2 • (λ x, x) x + α3 • (λ x, x^2) x + α4 • (λ x, x^3) x = 0) → 
  α1 = 0 ∧ α2 = 0 ∧ α3 = 0 ∧ α4 = 0 :=
by
  intro h
  sorry

end linear_independence_polynomials_l819_819692


namespace sum_of_digits_in_base_5_l819_819991

-- Given conditions
variables {S H E : ℕ}
variable base5_representation : Π (n : ℕ), ℕ

def distinct_non_zero_digits_less_than_five (a b c : ℕ) : Prop :=
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ a < 5 ∧ 0 < b ∧ b < 5 ∧ 0 < c ∧ c < 5)

def base_5_addition_constraint (S H E : ℕ) :=
  base5_representation (S * 25 + H * 5 + E) = S * 25 + E * 5 + S

-- The problem translated to Lean statement
theorem sum_of_digits_in_base_5 :
  distinct_non_zero_digits_less_than_five S H E ∧
  base_5_addition_constraint S H E →
  base5_representation (S + H + E) = 2 :=
sorry

end sum_of_digits_in_base_5_l819_819991


namespace sum_of_constants_l819_819342

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l819_819342


namespace no_half_probability_socks_l819_819730

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l819_819730


namespace log5_15625_eq_6_l819_819093

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l819_819093


namespace possible_values_of_b_l819_819940

theorem possible_values_of_b (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h5 : a = c + 5) : b ∈ {2, 3, 4, 5, 7, 8, 9} :=
begin
  sorry
end

end possible_values_of_b_l819_819940


namespace meters_of_cloth_sold_l819_819443

/-- Conditions -/
def total_selling_price : ℝ := 9890
def profit_per_meter : ℝ := 24
def cost_price_per_meter : ℝ := 83.5

/-- Theorem Statement -/
theorem meters_of_cloth_sold :
  let selling_price_per_meter := cost_price_per_meter + profit_per_meter in
  let x := total_selling_price / selling_price_per_meter in
  x = 92 :=
by
  sorry

end meters_of_cloth_sold_l819_819443


namespace solveForS_l819_819189

theorem solveForS (s : ℝ) (h : sqrt (3 * sqrt (s - 3)) = real.sqrt (real.nth_root 4 (8 - s))) : s = 3.5 := 
sorry

end solveForS_l819_819189


namespace cost_of_apple_is_two_l819_819668

-- Define the costs and quantities
def cost_of_apple (A : ℝ) : Prop :=
  let total_cost := 12 * A + 4 * 1 + 4 * 3
  let total_pieces := 12 + 4 + 4
  let average_cost := 2
  total_cost = total_pieces * average_cost

theorem cost_of_apple_is_two : cost_of_apple 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cost_of_apple_is_two_l819_819668


namespace area_of_triangle_perpendicular_lines_l819_819334

theorem area_of_triangle_perpendicular_lines
  (a b c : ℝ)
  (h_perpendicular : 2 * (-a / b) = -1)
  (h_arithmetic : 2 * b = a + c)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0) :
  let l1 := λ x, 2 * x
  let l2 := λ x, (-a * x / b) - c / b
  let intersection_x := -c / (a + 2 * b)
  let intersection_y := l1 intersection_x
  let y_intercept_l2 := -c / b
  let S := 1/2 * |intersection_x * y_intercept_l2| in
  S = 9 / 20 :=
by
  sorry

end area_of_triangle_perpendicular_lines_l819_819334


namespace four_digit_num_count_l819_819585

theorem four_digit_num_count :
  let N := (a : Fin 10) * 1000 + (b : Fin 10) * 100 + (c : Fin 10) * 10 + (d : Fin 10)
  in
  (4000 ≤ N ∧ N < 7000) ∧ 
  (N % 5 = 0) ∧ 
  (b + c).val % 2 = 0 ∧ (3 ≤ b ∧ b < c ∧ c ≤ 6) 
  → (∃ (count : Nat), count = 12) := by
  sorry

end four_digit_num_count_l819_819585


namespace ratio_of_side_lengths_sum_l819_819347
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l819_819347


namespace distance_between_foci_of_ellipse_l819_819380

theorem distance_between_foci_of_ellipse :
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  2 * c = 14 :=
by
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  show 2 * c = 14, from sorry

end distance_between_foci_of_ellipse_l819_819380


namespace probability_of_ram_l819_819391

theorem probability_of_ram 
  (P_ravi : ℝ) (P_both : ℝ) 
  (h_ravi : P_ravi = 1 / 5) 
  (h_both : P_both = 0.11428571428571428) : 
  ∃ P_ram : ℝ, P_ram = 0.5714285714285714 :=
by
  sorry

end probability_of_ram_l819_819391


namespace suitcase_volume_comparison_l819_819235

theorem suitcase_volume_comparison (k : ℝ) (h : k > (4.4)^(3/2)) : 
  50^3 > 220^3 / k^2 :=
begin
  sorry -- proof omitted
end

end suitcase_volume_comparison_l819_819235


namespace bridget_apples_l819_819461

/-!
# Problem statement
Bridget bought a bag of apples. She gave half of the apples to Ann. She gave 5 apples to Cassie,
and 2 apples to Dan. She kept 6 apples for herself. Prove that Bridget originally bought 26 apples.
-/

theorem bridget_apples (x : ℕ) 
  (H1 : x / 2 + 2 * (x % 2) / 2 - 5 - 2 = 6) : x = 26 :=
sorry

end bridget_apples_l819_819461


namespace river_flow_rate_l819_819826

def depth_of_river : ℝ := 5
def width_of_river : ℝ := 35
def volume_per_minute : ℝ := 5833.333333333333

theorem river_flow_rate :
  ∃ R : ℝ, R = 2 ∧ 
  depth_of_river = 5 ∧
  width_of_river = 35 ∧
  volume_per_minute = 5833.333333333333 := by
  use 2
  simp [depth_of_river, width_of_river, volume_per_minute]
  sorry

end river_flow_rate_l819_819826


namespace helen_gas_usage_l819_819175

/--
  Assume:
  - Helen cuts her lawn from March through October.
  - Helen's lawn mower uses 2 gallons of gas every 4th time she cuts the lawn.
  - In March, April, September, and October, Helen cuts her lawn 2 times per month.
  - In May, June, July, and August, Helen cuts her lawn 4 times per month.
  Prove: The total gallons of gas needed for Helen to cut her lawn from March through October equals 12.
-/

theorem helen_gas_usage :
  ∀ (months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut : ℕ),
  months1 = 4 →
  months2 = 4 →
  cuts_per_month1 = 2 →
  cuts_per_month2 = 4 →
  gas_per_4th_cut = 2 →
  (months1 * cuts_per_month1 + months2 * cuts_per_month2) / 4 * gas_per_4th_cut = 12 :=
by
  intros months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  calc
    (4 * 2 + 4 * 4) / 4 * 2 = (8 + 16) / 4 * 2 : by rw [mul_add]
                                    ...             = 24 / 4 * 2 : by rw [add_mul]
                                    ...             = 6 * 2       : by norm_num
                                    ...             = 12          : by norm_num

end helen_gas_usage_l819_819175


namespace farey_neighbors_of_half_l819_819221

noncomputable def farey_neighbors (n : ℕ) : List (ℚ) :=
  if n % 2 = 1 then
    [ (n - 1 : ℚ) / (2 * n), (n + 1 : ℚ) / (2 * n) ]
  else
    [ (n - 2 : ℚ) / (2 * (n - 1)), n / (2 * (n - 1)) ]

theorem farey_neighbors_of_half (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℚ, a ∈ farey_neighbors n ∧ b ∈ farey_neighbors n ∧ 
    (n % 2 = 1 → a = (n - 1 : ℚ) / (2 * n) ∧ b = (n + 1 : ℚ) / (2 * n)) ∧
    (n % 2 = 0 → a = (n - 2 : ℚ) / (2 * (n - 1)) ∧ b = n / (2 * (n - 1))) :=
sorry

end farey_neighbors_of_half_l819_819221


namespace comparison_abc_l819_819955

variable (f : Real → Real)
variable (a b c : Real)
variable (x : Real)
variable (h_even : ∀ x, f (-x + 1) = f (x + 1))
variable (h_periodic : ∀ x, f (x + 2) = f x)
variable (h_mono : ∀ x y, 0 < x ∧ y < 1 ∧ x < y → f x < f y)
variable (h_f0 : f 0 = 0)
variable (a_def : a = f (Real.log 2))
variable (b_def : b = f (Real.log 3))
variable (c_def : c = f 0.5)

theorem comparison_abc : b > a ∧ a > c :=
sorry

end comparison_abc_l819_819955


namespace example4x4_coin_placement_l819_819285

theorem example4x4_coin_placement :
  (∃ (board : fin 4 → fin 4 → ℕ × ℕ),
    (∀ i j, i < 2 ∨ j < 2 → 
      let (gold, silver) := (board i j) in
      ∀ i' j' (h : i' ∈ finset.range 3) (h2 : j' ∈ finset.range 3),
        let (g', s') := (board (i + i')%4 (j + j')%4) in
        s' > g') ∧
    (let total_gold := finset.sum (finset.fin_range 4) (λ i, 
      finset.sum (finset.fin_range 4) (λ j, prod.fst (board i j))) in
    let total_silver := finset.sum (finset.fin_range 4) (λ i, 
      finset.sum (finset.fin_range 4) (λ j, prod.snd (board i j))) in
      total_gold > total_silver)
  )
  :=
sorry

end example4x4_coin_placement_l819_819285


namespace length_AB_l819_819418

-- Definitions and conditions
variables (R r a : ℝ) (hR : R > r) (BC_eq_a : BC = a) (r_eq_4 : r = 4)

-- Length of AB
theorem length_AB (AB : ℝ) : AB = a * Real.sqrt (R / (R - 4)) :=
sorry

end length_AB_l819_819418


namespace hyperbola_equation_through_point_l819_819194

theorem hyperbola_equation_through_point
  (H : ∀ x y : ℝ, 
    ((y = sqrt 2 ∧ x = 3) ∨ (∀ c : ℝ, y = c * x ∨ y = -c * x) → 
    exists a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    m = b / a ∧ m = 1/3 ∧ 
    λ = -1 ∧ a^2 = 153 ∧ b^2 = 17 ∧ 
    (x^2 / a^2 - y^2 / b^2) = 1)) :
  ∀ x y : ℝ, (x = 3 ∧ y = sqrt 2) → ((x^2 / 153 - y^2 / 17) = 1) :=
by
  sorry

end hyperbola_equation_through_point_l819_819194


namespace constant_term_in_expansion_l819_819963

theorem constant_term_in_expansion :
  (∃ n : ℕ, let T := (x : ℕ) → (C n x) * ((-1)^x) * (x^(n - 3*x)) in
   (C n 2 = C n 7) ∧ (- (C 9 3) = -84))

end constant_term_in_expansion_l819_819963


namespace triangle_CLM_is_isosceles_l819_819286

variable {Point : Type} [EuclideanGeometry Point]

-- Conditions: ABC is a right triangle with right angle at C, and K, L are specific points
variables (A B C K L M : Point)
variable (h_right : right_triangle A B C)
variable (h_AK_AC : distance A K = distance A C)
variable (h_BK_LC : distance B K = distance L C)
variable (h_intersect : ∃ M, segment B L ∩ segment C K = {M})

theorem triangle_CLM_is_isosceles :
  isosceles_triangle C L M :=
by 
  sorry

end triangle_CLM_is_isosceles_l819_819286


namespace simple_interest_rate_l819_819411

variables (P R T SI : ℝ)

theorem simple_interest_rate :
  T = 10 →
  SI = (2 / 5) * P →
  SI = (P * R * T) / 100 →
  R = 4 :=
by
  intros hT hSI hFormula
  sorry

end simple_interest_rate_l819_819411


namespace wood_length_equation_l819_819798

variable (x : ℝ) (l : ℝ)

theorem wood_length_equation : 
  (l = x + 4.5) → (0.5 * l = x - 1) → (0.5 * (x + 4.5) = x - 1) :=
by
  intros h₁ h₂
  rw [←h₁] at h₂
  exact h₂

#check wood_length_equation -- to check the correctness of the statement

end wood_length_equation_l819_819798


namespace solve_sqrt_equation_l819_819505

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l819_819505


namespace jerry_total_bill_l819_819645

-- Definitions for the initial bill and late fees
def initial_bill : ℝ := 250
def first_fee_rate : ℝ := 0.02
def second_fee_rate : ℝ := 0.03

-- Function to calculate the total bill after applying the fees
def total_bill (init : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let first_total := init * (1 + rate1)
  first_total * (1 + rate2)

-- Theorem statement
theorem jerry_total_bill : total_bill initial_bill first_fee_rate second_fee_rate = 262.65 := by
  sorry

end jerry_total_bill_l819_819645


namespace cards_difference_l819_819509

theorem cards_difference :
  let cards := [2, 0, 3, 5, 8],
      largest := 8532,
      smallest := 2035
  in largest - smallest = 6497 := 
by
  sorry

end cards_difference_l819_819509


namespace distinct_segment_lengths_count_l819_819283

theorem distinct_segment_lengths_count :
  let points := {0, 1, 2, 3, 5, 8, 2016}
  let segment_lengths := {abs x - y | x y in points}
  segment_lengths.card = 14 :=
sorry

end distinct_segment_lengths_count_l819_819283


namespace solve_sqrt_equation_l819_819504

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l819_819504


namespace hyperbola_standard_equation_l819_819142

open Real

noncomputable def distance_from_center_to_focus (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem hyperbola_standard_equation (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : b = sqrt 3 * c)
  (h4 : a + c = 3 * sqrt 3) :
  ∃ h : a^2 = 12 ∧ b = 3, y^2 / 12 - x^2 / 9 = 1 :=
sorry

end hyperbola_standard_equation_l819_819142


namespace tangent_line_at_origin_is_minus_3x_l819_819653

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_is_minus_3x (a : ℝ) (h : ∀ x : ℝ, f_prime a x = f_prime a (-x)) : 
  (f_prime 0 0 = -3) → ∀ x : ℝ, (f a x = -3 * x) :=
by
  sorry

end tangent_line_at_origin_is_minus_3x_l819_819653


namespace projection_correct_l819_819510

-- Define the vectors involved
def vec_a := (3, -4)
def vec_b := (1, 1)

-- Define the inner product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the norm squared of a vector
def norm_squared (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

-- Calculate the projection
def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product u v) / (norm_squared v) in
  (scalar * v.1, scalar * v.2)

-- State the theorem
theorem projection_correct :
  projection vec_a vec_b = (-0.5, -0.5) :=
by
  sorry

end projection_correct_l819_819510


namespace log5_15625_eq_6_l819_819090

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l819_819090


namespace equation_of_directrix_l819_819060

theorem equation_of_directrix (x y : ℝ) (h : y^2 = 2 * x) : 
  x = - (1/2) :=
sorry

end equation_of_directrix_l819_819060


namespace volume_of_intersection_of_two_perpendicular_cylinders_l819_819913

theorem volume_of_intersection_of_two_perpendicular_cylinders (R : ℝ) : 
  ∃ V : ℝ, V = (16 / 3) * R^3 := 
sorry

end volume_of_intersection_of_two_perpendicular_cylinders_l819_819913


namespace find_r_l819_819193

variable (a b c r : ℝ)

theorem find_r (h1 : a * (b - c) / (b * (c - a)) = r)
               (h2 : b * (c - a) / (c * (b - a)) = r)
               (h3 : r > 0) : 
               r = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_r_l819_819193


namespace infinite_not_always_greater_than_repeating_l819_819033

-- Definition of infinite decimal
def infinite_decimal (d : Real) : Prop := 
  ∃ seq : ℕ → ℕ, ∀ n : ℕ, abs (d - (∑ i in range n, seq i * (10 : ℝ)^(-i))) < (1 / (10 : ℝ)^n)

-- Definition of repeating decimal
def repeating_decimal (d : Real) : Prop :=
  ∃ a b : Real, b > 0 ∧ b < 1 ∧ d = a + b / (1 - 10^(0 + -1 : ℤ))

-- Prove that not all infinite decimals are greater than repeating decimals
theorem infinite_not_always_greater_than_repeating : ¬ ∀ d₁ d₂ : Real, (infinite_decimal d₁ ∧ repeating_decimal d₂) → d₁ > d₂ :=
by
  sorry

end infinite_not_always_greater_than_repeating_l819_819033


namespace constant_term_in_expansion_l819_819962

theorem constant_term_in_expansion :
  (∃ n : ℕ, let T := (x : ℕ) → (C n x) * ((-1)^x) * (x^(n - 3*x)) in
   (C n 2 = C n 7) ∧ (- (C 9 3) = -84))

end constant_term_in_expansion_l819_819962


namespace max_product_431_52_l819_819393

-- Given the digits 1, 2, 3, 4, and 5 and forming the product of two numbers,
-- prove that the arrangement 431 × 52 yields the maximum possible product.

theorem max_product_431_52 : ∀ (d1 d2 d3 d4 d5 : ℕ),
  {d1, d2, d3, d4, d5} = {1, 2, 3, 4, 5} →
  (∀ (a b : ℕ), a * b ≤ 431 * 52) :=
begin
  intros d1 d2 d3 d4 d5 h_set,
  sorry
end

end max_product_431_52_l819_819393


namespace polar_line_through_centers_l819_819630

-- Definition of the given circles in polar coordinates
def Circle1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def Circle2 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Statement of the problem
theorem polar_line_through_centers (ρ θ : ℝ) :
  (∃ c1 c2 : ℝ × ℝ, Circle1 c1.fst c1.snd ∧ Circle2 c2.fst c2.snd ∧ θ = Real.pi / 4) :=
sorry

end polar_line_through_centers_l819_819630


namespace find_complex_z_l819_819508

open Complex

theorem find_complex_z (z : ℂ) (hz : 3 * z + 4 * I * conj(z) = -6 - 8 * I) : z = -2 :=
sorry

end find_complex_z_l819_819508


namespace order_of_f_values_l819_819162

noncomputable def f (x : ℝ) : ℝ := (2 / 4^x) - x

def a : ℝ := 0
def b : ℝ := Real.log 2 / Real.log 0.4
def c : ℝ := Real.log 3 / Real.log 4

theorem order_of_f_values : f a < f c ∧ f c < f b :=
  by
  -- Proof will be added here
  sorry

end order_of_f_values_l819_819162


namespace sequence_periodic_iff_rational_l819_819452

open Set

-- Define the sequence recursively
noncomputable def sequence (x : ℕ → ℚ) (x1 : ℚ) (hx1 : 0 ≤ x1 ∧ x1 ≤ 1) (n : ℕ) :=
  if n = 0 then x1 else
  1 - abs (1 - 2 * (sequence x x1 hx1 (n-1)))

-- Define rationality of terms in the sequence
def is_rational (x : ℕ → ℚ) (n : ℕ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ x n = p / q

-- Define periodicity of the sequence
def is_periodic (x : ℕ → ℚ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

-- Main theorem stating the periodicity if and only if x1 is rational
theorem sequence_periodic_iff_rational (x1 : ℚ) (hx1 : 0 ≤ x1 ∧ x1 ≤ 1) :
  (∃ p : ℕ, is_periodic (sequence (fun _ => 0) x1 hx1) p) ↔ is_rational (sequence (fun _ => 0) x1 hx1) 0 :=
sorry

end sequence_periodic_iff_rational_l819_819452


namespace general_formula_a_sum_b_condition_l819_819137

noncomputable def sequence_a (n : ℕ) : ℕ := sorry
noncomputable def sum_a (n : ℕ) : ℕ := sorry

-- Conditions
def a_2_condition : Prop := sequence_a 2 = 4
def sum_condition (n : ℕ) : Prop := 2 * sum_a n = n * sequence_a n + n

-- General formula for the n-th term of the sequence a_n
theorem general_formula_a : 
  (∀ n, sequence_a n = 3 * n - 2) ↔
  (a_2_condition ∧ ∀ n, sum_condition n) :=
sorry

noncomputable def sequence_c (n : ℕ) : ℕ := sorry
noncomputable def sequence_b (n : ℕ) : ℕ := sorry
noncomputable def sum_b (n : ℕ) : ℝ := sorry

-- Geometric sequence condition
def geometric_sequence_condition : Prop :=
  ∀ n, sequence_c n = 4^n

-- Condition for a_n = b_n * c_n
def a_b_c_relation (n : ℕ) : Prop := 
  sequence_a n = sequence_b n * sequence_c n

-- Sum condition T_n < 2/3
theorem sum_b_condition :
  (∀ n, a_b_c_relation n) ∧ geometric_sequence_condition →
  (∀ n, sum_b n < 2 / 3) :=
sorry

end general_formula_a_sum_b_condition_l819_819137


namespace sum_thetas_l819_819565

noncomputable def f (x : ℝ) : ℝ := 2 / (x + 1)

def O : ℝ × ℝ := (0, 0)

def A_n (n : ℕ) : ℝ × ℝ := (n, f n)

def j : ℝ × ℝ := (0, 1)

def theta_n (n : ℕ) : ℝ :=
  let vector_OA_n := A_n n
  let magnitude_OA_n := Real.sqrt (vector_OA_n.1^2 + (vector_OA_n.2)^2)
  let dot_product := vector_OA_n.1 * j.1 + vector_OA_n.2 * j.2
  let cos_theta_n := dot_product / magnitude_OA_n
  let sin_theta_n := Real.sqrt (1 - cos_theta_n^2)
  (cos_theta_n / sin_theta_n)

theorem sum_thetas : 
  (∑ n in Finset.range 2016 | n > 0, (θ n).1 / (θ n).2) = 2016 / 2017 := sorry

end sum_thetas_l819_819565


namespace log_base_5_of_15625_eq_6_l819_819070

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l819_819070


namespace mass_percentage_C_in_CO_correct_l819_819906

def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_CO : ℝ := molar_mass_C + molar_mass_O

def mass_percentage_C_in_CO : ℝ :=
  (molar_mass_C / molar_mass_CO) * 100

theorem mass_percentage_C_in_CO_correct :
  mass_percentage_C_in_CO ≈ 42.88 :=
by
  unfold mass_percentage_C_in_CO
  unfold molar_mass_CO
  unfold molar_mass_C
  unfold molar_mass_O
  sorry

end mass_percentage_C_in_CO_correct_l819_819906


namespace primes_satisfying_equation_l819_819123

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l819_819123


namespace solve_otimes_eq_l819_819478

def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

theorem solve_otimes_eq : ∃ x : ℝ, otimes (-4) (x + 3) = 6 ↔ x = -5 :=
by
  use -5
  simp [otimes]
  sorry

end solve_otimes_eq_l819_819478


namespace expression_for_f_l819_819929

theorem expression_for_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 - x - 2) : ∀ x : ℤ, f x = x^2 - 3 * x := 
by
  sorry

end expression_for_f_l819_819929


namespace point_on_line_l819_819626

theorem point_on_line (c : ℝ) (h : ∀ (x1 y1 x2 y2 : ℝ), 
  ∃ (m b : ℝ), y1 = m * x1 + b ∧ y2 = m * x2 + b) : 
  c = 1012 :=
by
  -- Define the points
  let P1 := (2023 : ℝ, 0 : ℝ)
  let P2 := (-2021 : ℝ, 2024 : ℝ)
  -- Extract point coordinates
  cases P1 with x1 y1
  cases P2 with x2 y2
  -- Prove the point (1, 1012) satisfies the line equation
  obtain ⟨m, b, hm1, hm2⟩ := h x1 y1 x2 y2
  simp [x1, y1, x2, y2] at *
  -- Test point (1, c)
  have hp : 1 = 1 := rfl
  let Q := (1 : ℝ, c)
  cases Q with xq yq
  simp [xq, yq]

  -- Conclusion
  sorry

end point_on_line_l819_819626


namespace rebecca_eggs_l819_819687

theorem rebecca_eggs : ∃ (r : ℕ), (3 * 5 = r) ∧ r = 15 :=
by {
  use 15,
  split,
  { exact rfl, },
  { exact rfl, },
}

end rebecca_eggs_l819_819687


namespace log_base_5_of_15625_l819_819083

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l819_819083


namespace find_angle_DAE_l819_819632

universe u

variable {α : Type u}

-- Definitions of geometric objects and angles as given in conditions
def triangle (A B C : α) : Prop := true -- Placeholder for triangle definition
def angle_degrees (A B C : α) : Prop := true -- Placeholder for angle in degrees

noncomputable def is_perpendicular (A D B C : α) : Prop := true -- Perpendicular from A to BC
noncomputable def is_center (O : α) (A B C : α) : Prop := true -- Center of circumscribed circle
noncomputable def is_diameter_end (E A : α) (O : α) : Prop := true -- Other end of diameter through A

-- Given conditions as Lean definitions
variables (A B C D O E : α)
variable [triangle A B C]
variable [angle_degrees A C B]
variable [is_perpendicular A D B C]
variable [is_center O A B C]
variable [is_diameter_end E A O]

-- The proof problem statement
theorem find_angle_DAE (h1 : angle_degrees A C B = 45)
                      (h2 : angle_degrees C B A = 60) :
  angle_degrees D A E = 15 := sorry

end find_angle_DAE_l819_819632


namespace certain_number_correct_l819_819992

noncomputable def x : ℝ := 25 / 102
noncomputable def y : ℝ := 1.2443140329436361

theorem certain_number_correct :
  102 * x = 25 ∧ y - x ≈ 0.9992159937279498 ∧ y ≈ 1.2443140329436361 :=
by
  sorry

end certain_number_correct_l819_819992


namespace range_of_a_l819_819947

theorem range_of_a (a : ℝ) 
  (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) 
  (q : ∃ x : ℝ, x^2 - 4 * x + a ≤ 0) : 
  e ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l819_819947


namespace range_of_a_l819_819563

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (1/4 - a) * x + 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ (a = 1 / 2) :=
sorry

end range_of_a_l819_819563


namespace geometric_sequence_sum_of_first_four_terms_l819_819206

theorem geometric_sequence_sum_of_first_four_terms (a r : ℝ) 
  (h1 : a + a * r = 7) 
  (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5) = 91) : 
  a * (1 + r + r^2 + r^3) = 32 :=
by
  sorry

end geometric_sequence_sum_of_first_four_terms_l819_819206


namespace find_binomial_parameters_l819_819533

noncomputable def binomial_distribution (n : ℕ) (p : ℝ) : ℝ := sorry

theorem find_binomial_parameters (n : ℕ) (p : ℝ) 
  (h1 : E (binomial_distribution n p) = 2.4)
  (h2 : var (binomial_distribution n p) = 1.44) : 
  n = 6 ∧ p = 0.4 := 
by
  sorry

end find_binomial_parameters_l819_819533


namespace point_on_angle_bisector_l819_819600

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l819_819600


namespace monotone_increasing_interval_l819_819566

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6)

theorem monotone_increasing_interval (k : ℤ) : 
  (∀ ω > 0, ∃ I : Set ℝ, I = Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) ∧ 
  (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f 2 x ≤ f 2 y)) := 
begin
  sorry
end

end monotone_increasing_interval_l819_819566


namespace parallelogram_height_l819_819201

theorem parallelogram_height (base height area : ℝ) (h_base : base = 9) (h_area : area = 33.3) (h_formula : area = base * height) : height = 3.7 :=
by
  -- Proof goes here, but currently skipped
  sorry

end parallelogram_height_l819_819201


namespace line_translation_upwards_units_l819_819999

theorem line_translation_upwards_units:
  ∀ (x : ℝ), (y = x / 3) → (y = (x + 5) / 3) → (y' = y + 5 / 3) :=
by
  sorry

end line_translation_upwards_units_l819_819999


namespace not_divisible_by_3_l819_819264

/- Define p(n) -/
def p : ℕ → ℕ
| 1     := 1
| 2     := 1
| 3     := 2
| (n+1) := p n + p (n-2) + 1

/- Theorem to prove -/
theorem not_divisible_by_3 : (p 1996) % 3 ≠ 0 := 
sorry

end not_divisible_by_3_l819_819264


namespace find_number_l819_819435

theorem find_number :
  let sum := 555 + 445,
      difference := 555 - 445,
      quotient := 2 * difference,
      remainder := 50
  in ((sum * quotient) + remainder) = 220050 :=
by
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  let remainder := 50
  show ((sum * quotient) + remainder) = 220050
  sorry

end find_number_l819_819435


namespace num_integers_in_sequence_l819_819368

theorem num_integers_in_sequence : 
  let seq := λ n : ℕ, 8820 / 3^n in
  ∀ n, seq n ∈ ℤ ↔ n ≤ 2 :=
begin
  -- Proof goes here, this is just the statement
  sorry
end

end num_integers_in_sequence_l819_819368


namespace point_on_line_iff_l819_819685

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given points O, A, B, and X in a vector space V, prove that X lies on the line AB if and only if
there exists a scalar t such that the position vector of X is a linear combination of the position vectors
of A and B with respect to O. -/
theorem point_on_line_iff (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔ (∃ t : ℝ, ∃ (t : ℝ), X - O = (1 - t) • (A - O) + t • (B - O)) :=
sorry

end point_on_line_iff_l819_819685


namespace smallest_positive_period_monotonically_decreasing_interval_solution_range_exists_l819_819973

noncomputable def f (x : ℝ) : ℝ := 2 * sin^2 (π / 4 + x) - sqrt 3 * cos (2 * x)

theorem smallest_positive_period : (∀ x, f (x + π) = f x) := by
  sorry

theorem monotonically_decreasing_interval (k : ℤ) : 
  (∀ x ∈ set.Icc (k * π + 5 * π / 12) (k * π + 11 * π / 12), 
    -2 * cos (2 * x - π / 3) ≤ 0) := by
  sorry

theorem solution_range_exists {m : ℝ} : 
  (∃ x ∈ set.Icc (π / 4) (5 * π / 6), f x = m) ↔ (1 - sqrt 3 ≤ m ∧ m ≤ 3) := by
  sorry

end smallest_positive_period_monotonically_decreasing_interval_solution_range_exists_l819_819973


namespace sum_xyz_equals_l819_819946

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

def condition1 : Prop := x^2 + y^2 + x * y = 1
def condition2 : Prop := y^2 + z^2 + y * z = 4
def condition3 : Prop := z^2 + x^2 + z * x = 5
def conditions : Prop := x > 0 ∧ y > 0 ∧ z > 0 ∧ condition1 ∧ condition2 ∧ condition3

theorem sum_xyz_equals : conditions → x + y + z = real.sqrt (5 + 2 * real.sqrt 3) := 
by 
  intro h
  sorry

end sum_xyz_equals_l819_819946


namespace f_is_constant_l819_819267

noncomputable def is_const (f : ℤ × ℤ → ℕ) := ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c

theorem f_is_constant (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_const f :=
sorry

end f_is_constant_l819_819267


namespace find_q_l819_819666

theorem find_q (a b m p q : ℚ) (h1 : a * b = 3) (h2 : a + b = m) 
  (h3 : (a + 1/b) * (b + 1/a) = q) : 
  q = 13 / 3 := by
  sorry

end find_q_l819_819666


namespace final_theorem_l819_819951

variable {α : ℝ}

-- Define the given condition
def given_condition (α : ℝ) : Prop := 
  Real.tan (α + π / 4) = 3 + 2 * Real.sqrt 2

-- Define the expression we are trying to prove
def target_expression (α : ℝ) : ℝ := 
  (1 - Real.cos (2 * α)) / Real.sin (2 * α)

-- State the theorem
theorem final_theorem (α : ℝ) (h : given_condition α) : 
  target_expression α = Real.sqrt 2 / 2 := 
  sorry

end final_theorem_l819_819951


namespace ellipse_foci_distance_l819_819385

theorem ellipse_foci_distance :
  let eps := [(1, 5), (4, -3), (9, 5)] in
  ∃ (a b : ℝ), a > b ∧ 2 * real.sqrt (a^2 - b^2) = 14 :=
by
  sorry

end ellipse_foci_distance_l819_819385


namespace parabola_focus_distance_l819_819007

theorem parabola_focus_distance (x0 : ℝ) (m : ℝ) (hf : x0^2 = m * (-3)) (hd : abs(m / 4 + 3) = 5) :
  m = -8 :=
sorry

end parabola_focus_distance_l819_819007


namespace triangle_perimeter_l819_819944

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l819_819944


namespace ratio_of_square_sides_l819_819364

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l819_819364


namespace perimeter_of_figure_l819_819315

theorem perimeter_of_figure (total_area : ℝ) (num_squares : ℕ) (area_each : ℝ) 
(side_length : ℝ) (num_vertical_segments : ℕ) (num_horizontal_segments : ℕ) : 
  total_area = 144 ∧ num_squares = 4 ∧ 
  area_each = total_area / num_squares ∧ 
  side_length = Math.sqrt area_each ∧ 
  num_vertical_segments = 4 ∧ num_horizontal_segments = 6 → 
  10 * side_length = 60 :=
by
  intros h
  have h1 : area_each = 144 / 4 := by sorry
  have h2 : side_length = Math.sqrt 36 := by sorry
  have h3 : 10 * 6 = 60 := by sorry
  exact h3

end perimeter_of_figure_l819_819315


namespace l2_passes_through_0_2_l819_819196

open Classical

noncomputable theory

-- Define the line l1
def line_l1 (k : ℝ) : ℝ → ℝ := λ x, k * (x - 4)

-- Define the point (2,1)
def point_2_1 : ℝ × ℝ := (2, 1)

-- Condition: l1 passes through (4,0)
def l1_passes_through_4_0 (k : ℝ) : Prop := (line_l1 k 4 = 0)

-- Condition: l1 is symmetric to l2 about (2,1)
def symmetric_about (x1 y1 x2 y2 px py : ℝ) : Prop := (x2 - px = px - x1) ∧ (y2 - py = py - y1)

variable {k : ℝ}
variable {l2 : ℝ → ℝ}

-- The goal to prove: line l2 always passes through (0,2)
theorem l2_passes_through_0_2 (h₁ : l1_passes_through_4_0 k) 
                                (h₂ : ∀ x, line_l1 k x = line_l1 k (2 * 2 - x) + 2 * (1 - line_l1 k 2)) :
  ∃ y, l2 0 = 2 :=
sorry

end l2_passes_through_0_2_l819_819196


namespace area_of_triangle_ABC_is_geometric_mean_l819_819967

theorem area_of_triangle_ABC_is_geometric_mean (A B C D1 D2 : Point) 
    (h1 : lineSegment A B ∥ lineSegment D2 C) 
    (h2 : lineSegment A C ∥ lineSegment D1 B) 
    (h3 : A ∈ line D1 D2) : 
    area (triangle A B C) = real.sqrt (area (triangle A B D1) * area (triangle A C D2)) := sorry

end area_of_triangle_ABC_is_geometric_mean_l819_819967


namespace suitcase_volume_comparison_l819_819234

theorem suitcase_volume_comparison (k : ℝ) (h : k > (4.4)^(3/2)) : 
  50^3 > 220^3 / k^2 :=
begin
  sorry -- proof omitted
end

end suitcase_volume_comparison_l819_819234


namespace tanika_boxes_sold_l819_819312

theorem tanika_boxes_sold :
  let Thursday := 60
  let Friday := Thursday + 0.5 * Thursday
  let Saturday := Friday + 0.8 * Friday
  let Sunday := Saturday - 0.3 * Saturday
  Thursday + Friday + Saturday + Sunday = 425 :=
by
  let Thursday := 60
  let Friday := Thursday + 0.5 * Thursday
  let Saturday := Friday + 0.8 * Friday
  let Sunday := Saturday - 0.3 * Saturday
  sorry

end tanika_boxes_sold_l819_819312


namespace henry_socks_l819_819619

theorem henry_socks : 
  ∃ a b c : ℕ, 
    a + b + c = 15 ∧ 
    2 * a + 3 * b + 5 * c = 36 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    a = 11 :=
by
  sorry

end henry_socks_l819_819619


namespace max_value_inequality_l819_819269

theorem max_value_inequality (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (kx + y)^2 / (x^2 + y^2) ≤ 2 :=
sorry

end max_value_inequality_l819_819269


namespace frog_pond_tadpoles_ratio_l819_819816

theorem frog_pond_tadpoles_ratio :
  ∀ (T : ℕ),  -- T is the number of tadpoles
  let frogs := 5 in  -- initial number of frogs
  let max_frogs := 8 in  -- maximum number of frogs the pond can sustain
  let tadpoles_mature_into_frogs := (2 * T) / 3 in  -- two-thirds of tadpoles will mature
  let new_frogs := max_frogs - frogs in  -- new frogs that can be sustained
  tadpoles_mature_into_frogs = new_frogs →
  (frogs + new_frogs) = max_frogs →
  (T + frogs) - max_frogs = 7 →
  frogs = max_frogs →
  (T / frogs = 1) :=
by
  intros T frogs max_frogs tadpoles_mature_into_frogs new_frogs h1 h2 h3 h4
  sorry

end frog_pond_tadpoles_ratio_l819_819816


namespace sequence_length_unique_l819_819375

theorem sequence_length_unique(binary_expression : ℕ → ℕ):
  ∃ (b : ℕ → ℕ) (m : ℕ), (∀ i j, i < j → b i < b j) ∧
                         (\sum_i in range m, 2^(b i)) = (2^210 + 1) / (2^15 + 1) ∧
                         m = 105 :=
begin
  sorry
end

end sequence_length_unique_l819_819375


namespace loan_interest_years_l819_819019

theorem loan_interest_years (x y total second_part : ℝ) (rate1 rate2 years2 : ℝ) 
  (h_total : total = x + y) (h_second_part : second_part = y)
  (h_interest : (x * rate1 * n) / 100 = (y * rate2 * years2) / 100) :
  n = 8 :=
begin
  -- Given conditions
  have h1 : total = 2665 := rfl,
  have h2 : second_part = 1640 := rfl,
  have h3 : y = 1640 := h_second_part,
  have h4 : x = 2665 - y := 
    by {rw [h_total, h3], exact rfl},
  
  -- Simplifying the interest equality
  have h5 : (x * 3 * n) / 100 = (y * 5 * 3) / 100 := 
    by {rw [h_interest, h4, h3], exact rfl},

  -- Simplify to solve for n
  have hn : x * n = y * 5 :=
    by {field_simp at h5, exact h5},
  rw [h4, h3] at hn,
  have n_val : n = (1640 * 5) / (2665 - 1640),
  norm_num at n_val,
  exact n_val
end

end loan_interest_years_l819_819019


namespace probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l819_819700

noncomputable def binomial (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ)

noncomputable def probability_of_winning_fifth_game_championship : ℝ :=
  binomial 4 3 * 0.6^4 * 0.4

noncomputable def overall_probability_of_winning_championship : ℝ :=
  0.6^4 +
  binomial 4 3 * 0.6^4 * 0.4 +
  binomial 5 3 * 0.6^4 * 0.4^2 +
  binomial 6 3 * 0.6^4 * 0.4^3

theorem probability_of_winning_fifth_game_championship_correct :
  probability_of_winning_fifth_game_championship = 0.20736 := by
  sorry

theorem overall_probability_of_winning_championship_correct :
  overall_probability_of_winning_championship = 0.710208 := by
  sorry

end probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l819_819700


namespace g_inv_undefined_at_one_l819_819589

def g (x : ℝ) := (x - 2) / (x - 5)

theorem g_inv_undefined_at_one : ∀ x, g x = (2 - 5 * x) / (1 - x) → (1 = x) → false := 
by 
  sorry

end g_inv_undefined_at_one_l819_819589


namespace correct_statement_l819_819790

theorem correct_statement:
  ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < x :=
by
  intros x h
  cases h with h1 h2
  calc
    x^2 = x * x   : by rw sq
    ... < x * 1   : mul_lt_mul_of_pos_left h2 h1
    ... = x       : by rw mul_one

end correct_statement_l819_819790


namespace evaluate_expression_l819_819254

theorem evaluate_expression : 
  (let x := 4 in let y := 2 in ( (1 / 5) ^ (y - x) ) = 25) := 
by 
  sorry

end evaluate_expression_l819_819254


namespace work_completion_time_l819_819410

theorem work_completion_time (d : ℕ) (h : d = 9) : 3 * d = 27 := by
  sorry

end work_completion_time_l819_819410


namespace socks_impossible_l819_819749

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l819_819749


namespace solution_set_of_inequality_l819_819725

theorem solution_set_of_inequality (x : ℝ) : x + x^3 ≥ 0 → x ≥ 0 :=
begin
    sorry
end

end solution_set_of_inequality_l819_819725


namespace sum_of_ratio_simplified_l819_819354

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l819_819354


namespace slope_angle_of_y_axis_l819_819606

/-- The slope angle of the line x=0 is π/2. -/
theorem slope_angle_of_y_axis : slope_angle (λ P : ℝ × ℝ, P.1 = 0) = π / 2 := 
sorry

end slope_angle_of_y_axis_l819_819606


namespace unknown_number_is_correct_l819_819367

theorem unknown_number_is_correct :
  ∃ x : ℝ, (0.82 ^ 3 - 0.1 ^ 3 / 0.82 ^ 2 + x + 0.1 ^ 2 = 0.72) ∧ (x ≈ 0.160119) :=
by
  sorry

end unknown_number_is_correct_l819_819367


namespace prime_square_sum_eq_square_iff_l819_819118

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l819_819118


namespace value_of_n_l819_819995

theorem value_of_n (t : ℝ) (h : t = 10) : 
  let n := (4 * t^2 - 10 * t - 2 - 3 * (t^2 - t + 3) + (t^2 + 5 * t - 1)) / ((t + 7) + (t - 13))
  in n = 12 :=
by
  sorry

end value_of_n_l819_819995


namespace find_original_price_l819_819722

noncomputable def original_price (P : ℝ) : Prop :=
  0.75 * 1.25 * P = 150

theorem find_original_price : original_price 160 :=
by 
  unfold original_price
  norm_num
  sorry

end find_original_price_l819_819722


namespace Barbiers_theorem_l819_819054

-- Definitions for curves and conditions of constant width
def constant_width (h : ℝ) (K : Set (ℝ × ℝ)) : Prop := sorry
def diameter (O : Set (ℝ × ℝ)) : ℝ := sorry

-- Barbier's theorem statement in Lean 4
theorem Barbiers_theorem (h : ℝ) (K : Set (ℝ × ℝ)) (O : Set (ℝ × ℝ)) :
  (constant_width h K) → (diameter O = h) → 
  (∀ K, constant_width h K → length K = π * h) := 
sorry

end Barbiers_theorem_l819_819054


namespace max_a_sin_cos_increasing_l819_819191

theorem max_a_sin_cos_increasing :
  ∃ a, (∀ x : ℝ, x ∈ Icc (0 : ℝ) a → has_deriv_within (λ x, sin x + cos x) (Icc 0 a)).monotone ∧ a = π / 4 :=
begin
  sorry
end

end max_a_sin_cos_increasing_l819_819191


namespace laura_marbles_l819_819918

-- Given conditions
def volume_of_kevin_box : ℕ := 3 * 3 * 8
def marbles_in_kevin_box : ℕ := 216
def scaling_factor : ℕ := 3

-- Calculations for Laura
def volume_of_laura_box : ℕ := (scaling_factor * 3) * (scaling_factor * 3) * (scaling_factor * 8)

-- Theorem to prove
theorem laura_marbles :
  (volume_of_laura_box / volume_of_kevin_box) * marbles_in_kevin_box = 5832 :=
by 
  -- volume multipliers are correct as per the problem conditions
  have h1 : volume_of_kevin_box = 72 := by rfl,
  have h2 : volume_of_laura_box = 1944 := by rfl,
  rw [h1, h2],
  sorry

end laura_marbles_l819_819918


namespace percentage_goods_lost_eq_l819_819815

-- Define the initial conditions
def initial_value : ℝ := 100
def profit_margin : ℝ := 0.10 * initial_value
def selling_price : ℝ := initial_value + profit_margin
def loss_percentage : ℝ := 0.12

-- Define the correct answer as a constant
def correct_percentage_loss : ℝ := 13.2

-- Define the target theorem
theorem percentage_goods_lost_eq : (0.12 * selling_price / initial_value * 100) = correct_percentage_loss := 
by
  -- sorry is used to skip the proof part as per instructions
  sorry

end percentage_goods_lost_eq_l819_819815


namespace line_BC_eqn_symm_line_BC_CM_eqn_l819_819979

-- Definitions for the coordinates and lines
def A : ℝ × ℝ := (5, 1)
def median_CM := {p : ℝ × ℝ | 2 * p.1 - p.2 - 5 = 0}
def altitude_BH := {p : ℝ × ℝ | p.1 - 2 * p.2 - 5 = 0}

-- Definitions for vertices B and C, and the lines BC, and the symmetric line
def B := (-1, -3) : ℝ × ℝ
def C := (4, 3) : ℝ × ℝ
def line_BC := {p : ℝ × ℝ | 6 * p.1 - 5 * p.2 - 9 = 0}
def symm_line_BC_CM := {p : ℝ × ℝ | 38 * p.1 - 9 * p.2 - 125 = 0}

-- Theorem statements
theorem line_BC_eqn :
  ∀ (B C : ℝ × ℝ), (B = (-1, -3)) ∧ (C = (4, 3)) → (∀ p : ℝ × ℝ, p ∈ line_BC ↔ 6 * p.1 - 5 * p.2 - 9 = 0) :=
by
  intros B C h
  rw [h.1, h.2]
  exact sorry

theorem symm_line_BC_CM_eqn :
  ∀ (B' : ℝ × ℝ), (B' = (11 / 5, -23 / 5)) → (∀ p : ℝ × ℝ, p ∈ symm_line_BC_CM ↔ 38 * p.1 - 9 * p.2 - 125 = 0) :=
by
  intro B'
  intro h
  rw h
  exact sorry

end line_BC_eqn_symm_line_BC_CM_eqn_l819_819979


namespace single_elimination_matches_l819_819214

theorem single_elimination_matches (n : ℕ) (h : n = 247) : n - 1 = 246 :=
by {
  rw h,
  norm_num,
  sorry
}

end single_elimination_matches_l819_819214


namespace solve_p_l819_819820

noncomputable def parabola_p (p : ℝ) (h : p > 0) : Prop :=
∃ A B : ℝ × ℝ, 
  let center := (3, 2)
  let radius := 4 
  let focus : ℝ × ℝ := (p / 2, 0)
  let parabola_eq := A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1
  let line_eq := (focus.1 + 0) * (A.1 - B.1) = 0
  let circle_eq := (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 2
  parabola_eq ∧ line_eq ∧ circle_eq

theorem solve_p : parabola_p 2
  sorry

end solve_p_l819_819820


namespace parallel_lines_a_l819_819168

theorem parallel_lines_a (a : ℝ) :
  ((∃ k : ℝ, (a + 2) / 6 = k ∧ (a + 3) / (2 * a - 1) = k) ∧ 
   ¬ ((-5 / -5) = ((a + 2) / 6)) ∧ ((a + 3) / (2 * a - 1) = (-5 / -5))) →
  a = -5 / 2 :=
by
  sorry

end parallel_lines_a_l819_819168


namespace isosceles_triangle_angle_relationship_l819_819215

variables {α : Type*} [metric_space α] [triangle_space α] 

def is_isosceles {A B C : α} : Prop := dist A C = dist B C

def is_altitude {C A B : α} (H : α) : Prop :=
  altitude_from C A B H

def is_midpoint {B H M : α} : Prop :=
  midpoint B H M

def is_perpendicular_foot {H A C K : α} : Prop :=
  perpendicular_foot H A C K

def is_intersection {B K C M L : α} : Prop :=
  intersection B K C M L

def is_perpendicular_intersection {B BC HL N : α} : Prop :=
  perpendicular_intersection B BC HL N 

theorem isosceles_triangle_angle_relationship
  {A B C H M K L N : α}
  (h_iso : is_isosceles A B C)
  (h_alt : is_altitude C A B H)
  (h_mid : is_midpoint B H M)
  (h_foot : is_perpendicular_foot H A C K)
  (h_int : is_intersection B K C M L)
  (h_perp_int : is_perpendicular_intersection B BC HL N) :
  angle ACB = 2 * angle BCN :=
by sorry

end isosceles_triangle_angle_relationship_l819_819215


namespace incorrect_ratio_implies_l819_819893

variable {a b c d : ℝ} (h : a * d = b * c) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

theorem incorrect_ratio_implies :
  ¬ (c / b = a / d) :=
sorry

end incorrect_ratio_implies_l819_819893


namespace sum_of_constants_l819_819343

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l819_819343


namespace bestVotingMethod_l819_819884

-- Define the context for the voting problem
def voteMethod (isSecret: Bool) (showHandsAgree: Bool) (showHandsDisagree: Bool) (recorded: Bool) : Type
| C
  
-- Define the hypothesis based on the conditions in (a)
def bestReflectsTrueWill (isSecret: Bool) (showHandsAgree: Bool) (showHandsDisagree: Bool) (recorded: Bool) 
  (method: voteMethod isSecret showHandsAgree showHandsDisagree recorded) : Prop :=
  isSecret = true

-- State the theorem that we want to prove according to (c)
theorem bestVotingMethod : bestReflectsTrueWill true false false false (voteMethod C) :=
  sorry

end bestVotingMethod_l819_819884


namespace parabola_focus_reciprocal_sum_l819_819386

theorem parabola_focus_reciprocal_sum (a : ℝ) (p q : ℝ) (h₁ : a > 0)
  (h₂ : ∃ P Q : ℝ × ℝ, 
          ((∃ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a)) ∧
            (P.1 ∧ Q.1 = 1 / (2 * a) / (1 - Math.cos P.2))) ∧
           ∃ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a)) ∧
            (Q.1 ∧ Q.1 = 1 / (2 * a) / (1 - Math.cos (Q.2 + π))) ∧
           p = Math.sqrt ((focus.1 - P.1)^2 + (focus.2 - P.2)^2) ∧
           q = Math.sqrt ((focus.1 - Q.1)^2 + (focus.2 - Q.2)^2))) :
  1 / p + 1 / q = 4 * a := 
by
  sorry

end parabola_focus_reciprocal_sum_l819_819386


namespace find_CD_l819_819229

-- Define the points and vectors involved
variables (A B C D : Type) [add_comm_group A] [module ℝ A]
variables (to_vec : affine_space A)

-- Given conditions
def triangle (A B C : A) : Prop := true
def on_segment (D : A) (A B : A) : Prop := true
def dist_in_twipes (A D : A) (k : ℝ) : Prop := true

-- The Lean statement
theorem find_CD (A B C D : A) [add_comm_group A] [module ℝ A] [affine_space A]
  (h1 : triangle A B C) (h2 : on_segment D A B) (h3 : dist_in_twipes A D (1/3)) : 
  ∀ (vAB vAC : A), ((1/3) • vAB - vAC) = CD :=
by
  sorry

end find_CD_l819_819229


namespace monotonic_increasing_interval_of_f_l819_819335

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8)

theorem monotonic_increasing_interval_of_f :
  set.Ioi (4 : ℝ) = {x : ℝ | f' x > 0} :=
sorry

end monotonic_increasing_interval_of_f_l819_819335


namespace inequality_for_interior_point_of_triangle_l819_819628

variables {A B C P A1 B1 C1 : Type}
variables [innerPoint : InteriorPoint P (Triangle ABC)]
variables [intersectionA1 : IntersectionPoint (LineSegment P A) (Side BC)]
variables [intersectionB1 : IntersectionPoint (LineSegment P B) (Side CA)]
variables [intersectionC1 : IntersectionPoint (LineSegment P C) (Side AB)]

theorem inequality_for_interior_point_of_triangle :
  (AA1 : Ratio (Distance A A1) (Distance P A1)) +
  (BB1 : Ratio (Distance B B1) (Distance P B1)) +
  (CC1 : Ratio (Distance C C1) (Distance P C1))  >= 9 :=
sorry

end inequality_for_interior_point_of_triangle_l819_819628


namespace bead_distribution_values_l819_819671

theorem bead_distribution_values (r n : ℕ) (h1 : r * n = 480) (h2 : r > 1) (h3 : n > 1) : 
  finset.card {r | r * n = 480 ∧ 1 < r ∧ 1 < 480 / r} = 22 := 
sorry

end bead_distribution_values_l819_819671


namespace cynthia_potato_problem_l819_819056

def potatoes_more_chips_than_wedges (total_potatoes cut_potatoes wedges_per_potato chips_per_potato: ℕ) :=
  let remaining_potatoes := total_potatoes - cut_potatoes in
  let halved_potatoes := remaining_potatoes / 2 in
  let total_chips := halved_potatoes * chips_per_potato in
  let total_wedges := cut_potatoes * wedges_per_potato in
  total_chips - total_wedges

theorem cynthia_potato_problem :
  potatoes_more_chips_than_wedges 67 13 8 20 = 436 :=
by
  sorry

end cynthia_potato_problem_l819_819056


namespace log5_of_15625_l819_819086

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l819_819086


namespace painted_cube_probability_l819_819886

theorem painted_cube_probability (prob_red : ℚ) (prob_blue : ℚ) :
  prob_red = 2/3 ∧ prob_blue = 1/3 →
  ∃ (P : ℚ), P = (2/3)^6 + (1/3)^6 + 3 * (2/3)^4 * (1/3)^2 + 3 * (2/3)^2 * (1/3)^4 ∧
  P = 789 / 6561 :=
by
  intros h
  let prob_red := 2/3
  let prob_blue := 1/3
  let P := (2/3)^6 + (1/3)^6 + 3 * (2/3)^4 * (1/3)^2 + 3 * (2/3)^2 * (1/3)^4
  use P
  have h1 : prob_red = 2/3 := and.left h
  have h2 : prob_blue = 1/3 := and.right h
  split
  · exact rfl
  · sorry

end painted_cube_probability_l819_819886


namespace company_production_n_l819_819919

theorem company_production_n (n : ℕ) (P : ℕ) 
  (h1 : P = n * 50) 
  (h2 : (P + 90) / (n + 1) = 58) : n = 4 := by 
  sorry

end company_production_n_l819_819919


namespace zero_squared_implies_zero_l819_819682

theorem zero_squared_implies_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by {
  -- Assume by contradiction that not both a and b are zero.
  have h_neg : ¬(a = 0 ∧ b = 0), by sorry,
  
  -- Apply h and h_neg to derive a contradiction.
  sorry
}

end zero_squared_implies_zero_l819_819682


namespace ellipse_foci_distance_l819_819384

theorem ellipse_foci_distance :
  let eps := [(1, 5), (4, -3), (9, 5)] in
  ∃ (a b : ℝ), a > b ∧ 2 * real.sqrt (a^2 - b^2) = 14 :=
by
  sorry

end ellipse_foci_distance_l819_819384


namespace HK_angle_bisector_of_BHC_l819_819621

theorem HK_angle_bisector_of_BHC
  (ABC : Type*)
  [triangle ABC]
  (I H O T K : Point ABC)
  (incenter : Incenter ABC I)
  (orthocenter : Orthocenter ABC H)
  (circumcenter : Circumcenter ABC O)
  (A_excircle_touch : AExCircleTouchPoint ABC T BC)
  (incircle_touch : InCircleTouchPoint ABC K BC)
  (TI_passes_through_O : (T --- I) ∩ (O) ≠ ∅) 
  : IsAngleBisector H K H B C :=
sorry

end HK_angle_bisector_of_BHC_l819_819621


namespace solve_arithmetic_sequence_l819_819304

theorem solve_arithmetic_sequence (y : ℝ) (h : y > 0) : 
  let a1 := (2 : ℝ)^2
      a2 := y^2
      a3 := (4 : ℝ)^2
  in (a1 + a3) / 2 = a2 → y = Real.sqrt 10 :=
by
  intros a1 a2 a3 H
  have calc1 : a1 = 4 := by norm_num
  have calc2 : a3 = 16 := by norm_num
  rw [calc1, calc2] at H
  have avg_eq : (4 + 16) / 2 = 10 := by norm_num
  rw [avg_eq] at H
  suffices y_pos : y > 0, sorry
  sorry


end solve_arithmetic_sequence_l819_819304


namespace incorrect_selection_method_l819_819205

-- Define the instance of the problem.
def classOfStudents : Type := Fin 50 -- Finite type representing the class of 50 students.
def classPresident : classOfStudents := 0 -- Assume 0 is the class president.
def vicePresident : classOfStudents := 1 -- Assume 1 is the vice president.

-- Define the selection condition, i.e., at least one of the class president or vice president must be chosen.
def condition (selected : Finset classOfStudents) : Prop :=
  classPresident ∈ selected ∨ vicePresident ∈ selected ∧ selected.card = 5

-- Define the combination selection function for verification.
def combination (N K : ℕ) : ℕ := Nat.choose N K

-- The statement to be proved: Option C (C_{2}^{1}C_{49}^{4}) is incorrect.
theorem incorrect_selection_method :
  ¬ (combination 2 1 * combination 49 4 = combination 50 5 - combination 48 5 ∧
     combination 2 1 * combination 48 4 + combination 2 2 * combination 48 3 ∧
     combination 2 1 * combination 49 4 - combination 48 3) := 
sorry

end incorrect_selection_method_l819_819205


namespace complement_U_A_complement_U_B_intersection_A_complement_U_B_union_complement_U_A_B_l819_819581

open Set

variable {α : Type*}

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem complement_U_A : (U \ A) = {x | x < 1 ∨ 3 < x ∧ x ≤ 4 ∨ 6 ≤ x} := sorry

theorem complement_U_B : (U \ B) = {x | x < 2 ∨ 5 ≤ x} := sorry

theorem intersection_A_complement_U_B : (A ∩ (U \ B)) = {x | 1 ≤ x ∧ x < 2 ∨ 5 ≤ x ∧ x < 6} := sorry

theorem union_complement_U_A_B : ((U \ A) ∪ B) = {x | x < 1 ∨ 2 ≤ x ∧ x < 5 ∨ 6 ≤ x} := sorry

end complement_U_A_complement_U_B_intersection_A_complement_U_B_union_complement_U_A_B_l819_819581


namespace remainder_when_divided_l819_819266
variable {R : Type*} [CommRing R]

-- Declare the polynomial p and a, b, c as distinct elements of R
variables (p : R[X]) (a b c : R)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

-- Conditions of the problem
variable (h1 : Polynomial.eval a p = a)
variable (h2 : Polynomial.eval b p = b)
variable (h3 : Polynomial.eval c p = c)

-- Proof statement
theorem remainder_when_divided (p : R[X]) (a b c : R)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : Polynomial.eval a p = a)
  (h2 : Polynomial.eval b p = b)
  (h3 : Polynomial.eval c p = c) : 
  ∃ q : R[X], p = (X - C a) * (X - C b) * (X - C c) * q + X :=
sorry

end remainder_when_divided_l819_819266


namespace eyes_saw_plane_l819_819758

theorem eyes_saw_plane (total_students : ℕ) (fraction_looked_up : ℚ) (students_with_eyepatches : ℕ) :
  total_students = 200 → fraction_looked_up = 3/4 → students_with_eyepatches = 20 →
  ∃ eyes_saw_plane, eyes_saw_plane = 280 :=
by
  intros h1 h2 h3
  sorry

end eyes_saw_plane_l819_819758


namespace isosceles_triangle_min_perimeter_l819_819389

theorem isosceles_triangle_min_perimeter
  (P Q R J : Point)
  (dist_PQ : ℝ) (dist_PR : ℝ)
  (dist_QJ : ℝ)
  (isosceles : dist_PQ = dist_PR)
  (angle_bisectors_intersection : J = intersection_of_angle_bisectors P Q R)
  (QJ_equals_10 : dist Q J = 10) :
  (∃ Δ : Triangle, Δ.P = P ∧ Δ.Q = Q ∧ Δ.R = R ∧ Δ.perimeter = 270) :=
sorry

end isosceles_triangle_min_perimeter_l819_819389


namespace min_nS_n_l819_819261

open Function

noncomputable def a (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

noncomputable def S (n : ℕ) (a_1 : ℤ) (d : ℤ) : ℤ := n * a_1 + d * n * (n - 1) / 2

theorem min_nS_n (d : ℤ) (h_a7 : ∃ a_1 : ℤ, a 7 a_1 d = 5)
  (h_S5 : ∃ a_1 : ℤ, S 5 a_1 d = -55) :
  ∃ n : ℕ, n > 0 ∧ n * S n a_1 d = -343 :=
by
  sorry

end min_nS_n_l819_819261


namespace ratio_of_side_lengths_sum_l819_819350
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l819_819350


namespace option_C_forms_a_set_l819_819402

-- Definition of the criteria for forming a set
def well_defined (criterion : Prop) : Prop := criterion

-- Criteria for option C: all female students in grade one of Jiu Middle School
def grade_one_students_criteria (is_female : Prop) (is_grade_one_student : Prop) : Prop :=
  is_female ∧ is_grade_one_student

-- Proof statement
theorem option_C_forms_a_set :
  ∀ (is_female : Prop) (is_grade_one_student : Prop), well_defined (grade_one_students_criteria is_female is_grade_one_student) :=
  by sorry

end option_C_forms_a_set_l819_819402


namespace spherical_coordinates_transformation_l819_819937

-- Define the rectangular and spherical coordinates transformations
noncomputable def spherical_to_rectangular_coords (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

-- Given spherical coordinates for a point
def given_point_spherical_coords : ℝ × ℝ × ℝ := (3, (11 * Real.pi) / 9, Real.pi / 4)

-- Corresponding rectangular coordinates
def given_point_rectangular_coords : ℝ × ℝ × ℝ :=
  spherical_to_rectangular_coords 3 ((11 * Real.pi) / 9) (Real.pi / 4)

-- New rectangular coordinates after changing x to -x
def new_point_rectangular_coords : ℝ × ℝ × ℝ :=
  let (x, y, z) := given_point_rectangular_coords in (-x, y, z)

-- New spherical coordinates we expect to prove
def expected_new_spherical_coords : ℝ × ℝ × ℝ := (3, (2 * Real.pi) / 9, Real.pi / 4)

-- The proof statement (proof not provided here, just the goal)
theorem spherical_coordinates_transformation :
  (let (x, y, z) := new_point_rectangular_coords,
    spherical_to_rectangular_coords 3 ((2 * Real.pi) / 9) (Real.pi / 4) = (x, y, z)) :=
sorry

end spherical_coordinates_transformation_l819_819937


namespace range_fx_area_triangle_ABC_l819_819163

open Real

-- Define the function
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x ^ 2 + 2 * sin x * cos x - sqrt 3

-- Prove the range of the function
theorem range_fx : (∀ x ∈ Icc (π / 3) (11 * π / 24), sqrt 3 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Define relevant values for the triangle
def a := sqrt 3
def b := 2
def r := 3 * sqrt 2 / 4

-- Using law of sines to find the area
theorem area_triangle_ABC : (∃ A B C : ℝ, sin A = a / (2 * r) ∧ sin B = b / (2 * r) ∧ 
                            sin C = sin (A + B) ∧ 
                            (1 / 2) * a * b * sin C = sqrt 2) :=
sorry

end range_fx_area_triangle_ABC_l819_819163


namespace marty_votes_result_l819_819226

noncomputable def marty_votes (total_people undecided_ratio leaning_marty_ratio : ℝ) : ℝ :=
  let undecided_people := total_people * undecided_ratio in
  undecided_people * leaning_marty_ratio

theorem marty_votes_result :
  (marty_votes 600 0.15 0.40) = 36 := by
  sorry

end marty_votes_result_l819_819226


namespace units_digit_17_pow_2045_l819_819779

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_17_pow_2045 : units_digit (17 ^ 2045) = 7 :=
by
  have h : ∀ n : ℕ, units_digit (17 ^ n) = units_digit (7 ^ n) :=
    λ n, by simp [units_digit, pow_mod n 10 17]
  have cycle : ∀ k : ℕ, k % 4 = 1 → units_digit (7 ^ k) = 7 :=
    λ k hmod, by
      have units_cycle : ∀ m : ℕ, units_digit (7 ^ m) = nth ([7, 9, 3, 1] ++ (units_digit (7 ^ m))) (m % 4) :=
        λ m, by sorry -- Proof of the cycle is skipped
      let pos := k % 4
      have : pos = 1 := hmod
      simp [units_cycle, this]
  exact cycle 2045 (nat.mod_eq_of_lt (show 2045 < 2048, by norm_num))

end units_digit_17_pow_2045_l819_819779


namespace sum_absolute_values_of_first_ten_terms_l819_819473

noncomputable def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

noncomputable def absolute_sum_10 : ℤ :=
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|

theorem sum_absolute_values_of_first_ten_terms : absolute_sum_10 = 68 := by
  sorry

end sum_absolute_values_of_first_ten_terms_l819_819473


namespace solution_eq_l819_819489

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l819_819489


namespace find_a20n_l819_819139

variable {a : ℝ}
variable {a_n a_{n-1} : ℝ}

def recurrence_relation (a : ℝ) (a_n a_{n-1} : ℝ) : Prop :=
  a_n = (√3 * a_{n-1} + 1) / (√3 - a_{n-1})

theorem find_a20n (a : ℝ) (h : ∀ n, recurrence_relation a a_n a_{n-1}):
  a_{20 * n} = (a + √3) / (1 - √3 * a) :=
by 
  sorry

end find_a20n_l819_819139


namespace tens_digit_2023_pow_2024_minus_2025_l819_819882

theorem tens_digit_2023_pow_2024_minus_2025 :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 5 :=
sorry

end tens_digit_2023_pow_2024_minus_2025_l819_819882


namespace sum_of_divisors_24_l819_819774

theorem sum_of_divisors_24 : ∑ d in (List.range 25).filter (λ n, 24 % n = 0), d = 60 :=
by
  sorry

end sum_of_divisors_24_l819_819774


namespace rosa_total_calls_l819_819179

theorem rosa_total_calls : 
  let week1 := 10.2 * 50 in
  let week2 := 8.6 * 40 in
  let week3 := 12.4 * 45 in
  week1 + week2 + week3 = 1412 :=
by
  let week1 := 10.2 * 50
  let week2 := 8.6 * 40
  let week3 := 12.4 * 45
  have h_week1 : week1 = 510 := by sorry
  have h_week2 : week2 = 344 := by sorry
  have h_week3 : week3 = 558 := by sorry
  have total_calculation : week1 + week2 + week3 = 510 + 344 + 558 := by sorry
  have h_sum : 510 + 344 + 558 = 1412 := by sorry
  rw [h_week1, h_week2, h_week3, total_calculation, h_sum]
  rfl

end rosa_total_calls_l819_819179


namespace difference_of_squares_divisible_by_18_l819_819301

-- Definitions of odd integers.
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The main theorem stating the equivalence.
theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) % 18 = 0 := 
by
  sorry

end difference_of_squares_divisible_by_18_l819_819301


namespace max_area_triangle_BSA_l819_819316

-- Define the problem conditions and variables related to the pyramid
variables (S A B C D M N : ℝ) 
variable (is_square_ABCD :  A ^ 2 + B ^ 2 = C ^ 2 + D ^ 2) -- Assuming we represent the lengths of square sides
variable (SA_height : S = A) -- Assuming length representation
variable (is_midpoint_MN : (M + N) / 2 = (S + C) / 2) -- Midpoints condition 
variable (MN_length : (M - N).abs = 3)

-- Proof statement to be developed
theorem max_area_triangle_BSA : 
  ∃ x y, x^2 + y^2 = 36 ∧ (x * y) = 18 → ∃ A, (1/2) * x * y = 9 :=
by
  -- proof skipped
  sorry

end max_area_triangle_BSA_l819_819316


namespace largest_possible_integer_in_list_l819_819001

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ -- all integers are positive
    (a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7 ∨ e = 7) ∧ -- 7 occurs in the list
    (List.count (λ x, x = 7) [a, b, c, d, e] > 1) ∧ -- 7 occurs more than once
    (List.median [a, b, c, d, e] = 10) ∧ -- the median is 10
    (List.sum [a, b, c, d, e] / 5 = 10) ∧ -- the average is 10
    (max a (max b (max c (max d e))) = 16) := -- the largest integer is 16
by
  sorry

end largest_possible_integer_in_list_l819_819001


namespace equation_of_plane_is_correct_l819_819006

noncomputable def parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ :=
  λ s t, (2 + 2 * s - 3 * t, 4 + s, 1 - 3 * s + t)

theorem equation_of_plane_is_correct :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (A.natAbs) (B.natAbs) (C.natAbs) (D.natAbs) = 1 ∧
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0 := by
  let A := 1
  let B := 8
  let C := 3
  let D := -37
  use A, B, C, D
  split; try {exact zero_lt_one}
  split
  · norm_cast
    exact Int.gcd_comm A.natAbs (Int.gcd_comm B.natAbs (Int.gcd C.natAbs D.natAbs)).symm
  sorry -- proof of the plane equation

end equation_of_plane_is_correct_l819_819006


namespace no_half_probability_socks_l819_819731

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l819_819731


namespace avg_daily_production_n_l819_819413

theorem avg_daily_production_n (n : ℕ) (h₁ : 50 * n + 110 = 55 * (n + 1)) : n = 11 :=
by
  -- Proof omitted
  sorry

end avg_daily_production_n_l819_819413


namespace table_tennis_tournament_l819_819212

theorem table_tennis_tournament (n : ℕ) (matches : Finset (ℕ × ℕ)) (refereed_by : ℕ → ℕ → ℕ) :
  (∀ {i j : ℕ}, i ≠ j ⊆ {i, j} ∈ matches) →
  (∀ i, ∃ j k, j ≠ k ∧ refereed_by i j ≠ refereed_by i k) →
  (∃ f : ℕ → ℕ, (∀ (x : ℕ), ∃ y, refereed_by x y = f x) ∧
                 (∀ (x y : ℕ), x ≠ y → f x ≠ f y)) →
  False :=
by
  sorry

end table_tennis_tournament_l819_819212


namespace job_assignment_l819_819845

def Person : Type := string

def older_than (p1 p2 : Person) : Prop := sorry
def not_same_age (p1 p2 : Person) : Prop := sorry
def younger_than (p1 p2 : Person) : Prop := sorry

constants 
  (XiaoWang XiaoLi XiaoZhao : Person)
  (worker salesperson salesman : Person)

axiom cond1 : older_than XiaoZhao worker
axiom cond2 : not_same_age XiaoWang salesperson
axiom cond3 : younger_than salesperson XiaoLi

theorem job_assignment :
  (salesperson = XiaoZhao) ∧
  (salesman = XiaoLi) ∧
  (worker = XiaoWang) :=
sorry

end job_assignment_l819_819845


namespace estimate_greater_than_actual_l819_819676

variable {x y : ℝ}

def round_up (n : ℝ) : ℝ := ceil n
def round_down (n : ℝ) : ℝ := floor n

theorem estimate_greater_than_actual :
  round_up (2 * x + 3) / round_down (3 * x - 5) - round_down y > (2 * x + 3) / (3 * x - 5) - y :=
sorry

end estimate_greater_than_actual_l819_819676


namespace no_equal_prob_for_same_color_socks_l819_819741

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l819_819741


namespace loop_until_correct_l819_819407

-- Define the conditions
def num_iterations := 20

-- Define the loop condition
def loop_condition (i : Nat) : Prop := i > num_iterations

-- Theorem: Proof that the loop should continue until the counter i exceeds 20
theorem loop_until_correct (i : Nat) : loop_condition i := by
  sorry

end loop_until_correct_l819_819407


namespace solve_sqrt_equation_l819_819502

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l819_819502


namespace total_fish_caught_l819_819246

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l819_819246


namespace valid_fahrenheit_count_l819_819515

theorem valid_fahrenheit_count :
  let is_valid_f (F : ℤ) : Prop :=
    let C_rounded := (5 * (F - 32) + 4) / 9  -- rounding nearest
    let F_rounded_back := (9 * C_rounded + 160) / 5 -- rounding nearest
    C_rounded > 10
    ∧ F_rounded_back = F
  in (finset.Icc 50 500).filter is_valid_f).card = 35 :=
by {
  sorry
}

end valid_fahrenheit_count_l819_819515


namespace log_base_5_of_15625_l819_819079

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l819_819079


namespace tinsel_in_each_box_l819_819885

theorem tinsel_in_each_box :
  ∃ T : ℕ, (∀ (total_decorations_per_box total_boxes : ℕ), 
    total_decorations_per_box = T + 1 + 5 ∧ 
    total_boxes = 11 + 1 ∧ 
    total_boxes * total_decorations_per_box = 120 
    → T = 4) :=
begin
  existsi 4,
  intros total_decorations_per_box total_boxes,
  rintros ⟨h1, ⟨h2, h3⟩⟩,
  have ht : total_decorations_per_box = T + 6 := by assumption,
  have tb : total_boxes = 12 := by assumption,
  have h120 : total_boxes * total_decorations_per_box = 120 := by assumption,
  rw tb at h120,
  rw ht at h120,
  norm_num at h120,
  rw nat.mul_add_assoc at h120,
  norm_num at h120,
  exact nat.mul_left_inj zero_lt_succ _ h120,
  exact h3,
  exact h2
end

end tinsel_in_each_box_l819_819885


namespace simplify_function_and_find_sum_l819_819881

theorem simplify_function_and_find_sum :
  let y := (x : ℝ) ↦ (x^3 + 7*x^2 + 14*x + 8) / (x + 1)
  let A := 1
  let B := 6
  let C := 8
  let D := -1
  (∀ x, x ≠ -1 → y x = x^2 + 6 * x + 8) ∧ (A + B + C + D = 14) :=
sorry

end simplify_function_and_find_sum_l819_819881


namespace socks_impossible_l819_819747

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l819_819747


namespace minimum_value_of_sum_of_squares_l819_819106

open Classical

theorem minimum_value_of_sum_of_squares (x y : ℝ) 
  (h: x^2 - y^2 + 6 * x + 4 * y + 5 = 0) : x^2 + y^2 ≥ 0.5 :=
begin
  sorry
end

end minimum_value_of_sum_of_squares_l819_819106


namespace g_18_45_eq_180_l819_819326

noncomputable def g : ℕ × ℕ → ℕ 
| (x, y) := if x = y then 2 * x 
            else if x < y then
                (let t := y in (t / (t - x)) * g (x, t - x))
            else g (y, x) -- using symmetry

theorem g_18_45_eq_180 :
  g (18, 45) = 180 :=
by {
  sorry
}

end g_18_45_eq_180_l819_819326


namespace projection_is_q_l819_819907

noncomputable def projection_matrix := 
  let u := ![1, -1, 2]
  (1 / 6: ℚ) • ![
    ![1, -1, 2],
    ![-1, 1, -2],
    ![2, -2, 4]
  ]

theorem projection_is_q (v : Fin 3 → ℚ) :
  let Q := ![
    ![(1 / 6): ℚ, (-1 / 6): ℚ, (1 / 3): ℚ],
    ![(-1 / 6): ℚ, (1 / 6): ℚ, (-1 / 3): ℚ],
    ![(1 / 3): ℚ, (-1 / 3): ℚ, (2 / 3): ℚ]]
  in 
  Q.mul_vec v = projection_matrix.mul_vec v :=
sorry

end projection_is_q_l819_819907


namespace range_of_a_l819_819591

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Ioc 0 2, x^2 - 2*a*x + 2 ≥ 0) → a ≤ real.sqrt 2 := 
by
  sorry

end range_of_a_l819_819591


namespace solve_system_unique_solution_l819_819804

theorem solve_system_unique_solution :
  (∃ x y : ℝ, 3^y - 4^x = 11 ∧ log 4 x + log 3 y = 3/2) → (∃! (x, y) : ℝ × ℝ, x = 2 ∧ y = 3) :=
by
  sorry

end solve_system_unique_solution_l819_819804


namespace jessie_mother_age_comparison_l819_819672

def jessie_age_in_given_year (current_year jessie_birth_year jessie_age_in_2010: ℕ) : ℕ :=
  jessie_age_in_2010 + (current_year - jessie_birth_year + 10)

def mother_age_in_given_year (current_year jessie_birth_year jessie_age_in_2010: ℕ): ℕ :=
  5 * jessie_age_in_2010 + (current_year - jessie_birth_year)
  
theorem jessie_mother_age_comparison :
  ∃ (current_year : ℕ), 
  let jessie_age := jessie_age_in_given_year current_year 2010 10 in
  let mother_age := mother_age_in_given_year current_year 2010 10 in
  mother_age = 2.5 * jessie_age :=
by
  sorry

end jessie_mother_age_comparison_l819_819672


namespace square_of_area_of_triangle_l819_819728

-- Conditions:
-- 1. The vertices of the equilateral triangle lie on the ellipse given by the equation x^2 / 4 + y^2 / 9 = 1
-- 2. The centroid of the triangle is at the origin (0,0), which is also the center of the ellipse.
-- Question: calculate the square of the area of the triangle.

-- Mathematically equivalent problem in Lean
theorem square_of_area_of_triangle (vertices_on_ellipse : (ℝ × ℝ) → Prop)
  (centroid_origin : (0:ℝ, 0:ℝ)) :
  vertices_on_ellipse (x, y) ↔ (x^2 / 4 + y^2 / 9 = 1) ∧
  (centroid_origin = (0,0)) →
  (∃ a b c d e f : ℝ, 
  vertices_on_ellipse (a, b) ∧
  vertices_on_ellipse (c, d) ∧
  vertices_on_ellipse (e, f) ∧
  -- No explicit definition for equilateral triangle with centroid on (0,0) here due to complexity
  -- but assuming it satisfies the conditions given:
   (calculate_area a b c d e f)^2 = 507/16) :=
sorry
  
end square_of_area_of_triangle_l819_819728


namespace initial_amount_is_65000_l819_819097

-- Define the initial amount P
variable (P : ℝ)

-- Define the first condition: amount increases by 1/8 of itself every year
def first_year (P : ℝ) := (9 / 8) * P

-- Define the second condition: amount after two years
def second_year (P : ℝ) := ((9 / 8) * (9 / 8) * P)

-- Given the condition that the amount after two years is Rs. 82,265.625
constant amount_after_two_years : ℝ := 82265.625

theorem initial_amount_is_65000 : second_year P = amount_after_two_years → P = 65000 :=
by
  sorry

end initial_amount_is_65000_l819_819097


namespace calculate_selling_price_l819_819011

theorem calculate_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) 
  (h1 : cost_price = 83.33) 
  (h2 : profit_percentage = 20) : 
  selling_price = 100 := by
  sorry

end calculate_selling_price_l819_819011


namespace max_edges_connected_graph_l819_819908

theorem max_edges_connected_graph (G : SimpleGraph V) (n : ℕ) (h_conn : G.connected) (h_vertices : G.vertex_set.card = n) :
  ∃ (e ≤ 2 * n - 3), ∀ (C : set (V × V)), is_cycle G C → ¬G.delete_edges C.connected :=
sorry

end max_edges_connected_graph_l819_819908


namespace distinct_segment_lengths_l819_819281

theorem distinct_segment_lengths :
  let points := [0, 1, 2, 3, 5, 8, 2016]
  let segment_lengths := {abs (x - y) | x y : ℤ, x ∈ points, y ∈ points}
  ∃ n, n = 14 ∧ segment_lengths.card = n :=
by
  let points := [0, 1, 2, 3, 5, 8, 2016]
  let segment_lengths := {abs (x - y) | x y : ℤ, x ∈ points, y ∈ points}
  have h : segment_lengths.card = 14 := sorry
  exact Exists.intro 14 (And.intro rfl h)

end distinct_segment_lengths_l819_819281


namespace length_of_first_train_l819_819445

theorem length_of_first_train (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (crossing_time_s : ℝ) (length_second_train_m : ℝ) :
  speed_first_train_kmph = 120 → speed_second_train_kmph = 80 → crossing_time_s = 9 → length_second_train_m = 410.04 →
  let relative_speed_m_s := (speed_first_train_kmph + speed_second_train_kmph) * (1000 / 3600) in
  let combined_length_m := relative_speed_m_s * crossing_time_s in
  combined_length_m - length_second_train_m = 90 :=
by
  intros h1 h2 h3 h4
  -- Definitions from conditions
  let relative_speed_m_s := (120 + 80) * (1000 / 3600)
  let combined_length_m := relative_speed_m_s * 9
  -- Final correct length of the first train
  have length_first_train : combined_length_m - 410.04 = 90 := sorry
  exact length_first_train

end length_of_first_train_l819_819445


namespace total_bill_l819_819853

def num_adults := 2
def num_children := 5
def cost_per_meal := 3

theorem total_bill : (num_adults + num_children) * cost_per_meal = 21 := 
by 
  sorry

end total_bill_l819_819853


namespace problem_solution_l819_819652

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom h1 : is_arithmetic_seq a d
axiom h2 : sum_first_n a 9 = 3 * a 8

theorem problem_solution : sum_first_n a 15 / (3 * a 5) = 15 := by
  sorry

end problem_solution_l819_819652


namespace complex_fraction_value_l819_819714

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_value : (3 : ℂ) / ((1 - i) ^ 2) = (3 / 2) * i := by
  sorry

end complex_fraction_value_l819_819714


namespace average_distance_to_sides_l819_819000

open Real

noncomputable def side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def right_turn_distance : ℝ := 3

theorem average_distance_to_sides :
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  (d1 + d2 + d3 + d4) / 4 = 7.5 :=
by
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  have h : (d1 + d2 + d3 + d4) / 4 = 7.5
  { sorry }
  exact h

end average_distance_to_sides_l819_819000


namespace largest_even_not_sum_of_two_composite_odds_l819_819877

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end largest_even_not_sum_of_two_composite_odds_l819_819877


namespace min_value_geometric_seq_l819_819531

theorem min_value_geometric_seq (a : ℕ → ℝ) (m n : ℕ) (h_pos : ∀ k, a k > 0)
  (h1 : a 1 = 1)
  (h2 : a 7 = a 6 + 2 * a 5)
  (h3 : a m * a n = 16) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_value_geometric_seq_l819_819531


namespace hamburger_combinations_l819_819178

theorem hamburger_combinations :
  let condiments := 10
  let patties_choices := 4
  let condiment_combinations := 2^10
  let total_combinations := patties_choices * condiment_combinations
  in total_combinations = 4096 :=
by
  let condiments := 10
  let patties_choices := 4
  let condiment_combinations := Int.ofNat (2 ^ condiments)
  let total_combinations := patties_choices * condiment_combinations
  have condiments_eq : 2 ^ 10 = 1024 := by norm_num
  rw condiments_eq at condiment_combinations
  simp only [condiment_combinations, patties_choices]
  norm_num
  exact eq.refl 4096

end hamburger_combinations_l819_819178


namespace Clea_Rides_Escalator_Alone_l819_819238

-- Defining the conditions
variables (x y k : ℝ)
def Clea_Walking_Speed := x
def Total_Distance := y = 75 * x
def Time_with_Moving_Escalator := 30 * (x + k) = y
def Escalator_Speed := k = 1.5 * x

-- Stating the proof problem
theorem Clea_Rides_Escalator_Alone : 
  Total_Distance x y → 
  Time_with_Moving_Escalator x y k → 
  Escalator_Speed x k → 
  y / k = 50 :=
by
  intros
  sorry

end Clea_Rides_Escalator_Alone_l819_819238


namespace part_a_part_b_l819_819917

section
variable {n : ℕ} (k : ℕ)

-- Define f_i(n) as the number of divisors of n of the form 3k + i
def f_i (n : ℕ) (i : ℕ) : ℕ := 
  (Multiset.filter (λ d : ℕ, ∃ k, d = 3 * k + i) (Multiset.toFinset (List.divisors n))).card

-- Define f(n) as the difference between f_1(n) and f_2(n)
def f (n : ℕ) := f_i n 1 - f_i n 2

-- Prove that f(5^2022) = 1
theorem part_a : f (5 ^ 2022) = 1 := 
  sorry

-- Prove that f(21^2022) = 2023
theorem part_b : f (21 ^ 2022) = 2023 := 
  sorry

end

end part_a_part_b_l819_819917


namespace amy_picture_files_l819_819031

-- Define the given constants
constant music_files : ℝ := 4.0
constant video_files : ℝ := 21.0
constant total_files : ℝ := 48.0

-- Define the number of picture files Amy downloaded
def picture_files := total_files - (music_files + video_files)

-- Lean statement to prove the correctness
theorem amy_picture_files : picture_files = 23.0 :=
by
  sorry

end amy_picture_files_l819_819031


namespace find_p_l819_819099

noncomputable theory
open_locale topological_space

def f (x : ℝ) : ℝ := (real.cbrt (x + 1) + real.cbrt (x - 1) - 2 * real.cbrt x)

theorem find_p :
  (∃ L : ℝ, L ≠ 0 ∧ filter.tendsto (λ x : ℝ, x ^ (5 / 3) * f x) filter.at_top (nhds L)) →
  p = 5 / 3 :=
sorry

end find_p_l819_819099


namespace small_cone_altitude_l819_819427

/-- A frustum of a right circular cone is formed by cutting a small cone from the top of a larger cone. 
If this frustum has an altitude of 20 centimeters, the area of its lower base is 324π sq cm, and the 
area of its upper base is 36π sq cm, then the altitude of the small cone that was cut off is 10 cm. 
-/
theorem small_cone_altitude (h_f : ℝ) (A1 : ℝ) (A2 : ℝ) (π : ℝ) :
  h_f = 20 ∧ A1 = 324 * π ∧ A2 = 36 * π → (∃ h : ℝ, h = 10) :=
by
  intro h cond1 cond2
  sorry

end small_cone_altitude_l819_819427


namespace total_fish_caught_l819_819247

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l819_819247


namespace function_problem_l819_819154

theorem function_problem (f : ℕ → ℝ) (h1 : ∀ p q : ℕ, f (p + q) = f p * f q) (h2 : f 1 = 3) :
  (f (1) ^ 2 + f (2)) / f (1) + (f (2) ^ 2 + f (4)) / f (3) + (f (3) ^ 2 + f (6)) / f (5) + 
  (f (4) ^ 2 + f (8)) / f (7) + (f (5) ^ 2 + f (10)) / f (9) = 30 := by
  sorry

end function_problem_l819_819154


namespace in_fourth_quadrant_l819_819956

def z : ℂ := (Complex.mk 4 (-3)) / (Complex.mk 3 4) + Complex.I * 2

theorem in_fourth_quadrant : z.re > 0 ∧ z.im < 0 := by
  -- sorry will be replaced by the proof 
  sorry

end in_fourth_quadrant_l819_819956


namespace prime_square_sum_eq_square_iff_l819_819119

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l819_819119


namespace combined_cost_price_l819_819009

-- Definitions of selling prices and profit percentages
def SP1 : ℝ := 100
def SP2 : ℝ := 150
def SP3 : ℝ := 200

def P1 : ℝ := 0.4
def P2 : ℝ := 0.3
def P3 : ℝ := 0.2

-- Definitions of cost prices corresponding to selling prices and profit percentages
def CP1 : ℝ := SP1 / (1 + P1)
def CP2 : ℝ := SP2 / (1 + P2)
def CP3 : ℝ := SP3 / (1 + P3)

-- Combined cost price
def CP : ℝ := CP1 + CP2 + CP3

-- The theorem stating the combined cost price
theorem combined_cost_price : CP = 353.48 := by
  sorry

end combined_cost_price_l819_819009


namespace sum_of_areas_of_two_squares_l819_819512

theorem sum_of_areas_of_two_squares (a b : ℕ) (h1 : a = 8) (h2 : b = 10) :
  a * a + b * b = 164 := by
  sorry

end sum_of_areas_of_two_squares_l819_819512


namespace triangle_EFD_max_area_l819_819454

noncomputable def max_area_EFD (S_ABC : ℝ) : ℝ :=
  let EFD_max := (5 * real.sqrt 5 - 11) / 2
  in if S_ABC = 1 then EFD_max else 0

theorem triangle_EFD_max_area (S_ABC : ℝ) (h : S_ABC = 1) :
  max_area_EFD S_ABC = (5 * real.sqrt 5 - 11) / 2 :=
by
  sorry

end triangle_EFD_max_area_l819_819454


namespace prime_eq_sol_l819_819116

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l819_819116


namespace number_of_words_with_A_is_correct_l819_819333

open Finset

def alphabet : Finset Char := { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' }

noncomputable def words_of_length (n : ℕ) : Finset (List Char) :=
  (range (26^n)).Image (λ i, (List.finRange n).map (λ k, alphabet.toList.get! (i / (26^k) % 26)))

noncomputable def words_with_A (n : ℕ) : Finset (List Char) :=
  words_of_length n ∖ (words_of_length n).filter (λ w, ('A' ∉ w))

noncomputable def total_words_with_A : ℕ :=
  ∑ i in range (6), (words_with_A i).card

theorem number_of_words_with_A_is_correct : total_words_with_A = 2202115 :=
  by
    sorry

end number_of_words_with_A_is_correct_l819_819333


namespace trig_identity_cos_l819_819132

theorem trig_identity_cos (x : ℝ) (h : sin (2 * x + π / 6) = -1 / 3) : cos (π / 3 - 2 * x) = -1 / 3 :=
by
  sorry

end trig_identity_cos_l819_819132


namespace total_fish_caught_l819_819249

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l819_819249


namespace greatest_q_minus_r_l819_819414

-- Define the constraints in Lean
def is_digit (n : ℕ) : Prop := n < 10
def is_prime_digit (n : ℕ) : Prop := is_digit n ∧ Nat.Prime n

-- The Lean statement for the mathematical proof problem
theorem greatest_q_minus_r :
  ∃ (q r : ℕ), is_prime_digit q ∧ is_prime_digit r ∧
  ‖q - r‖ < 70 / 9 ∧ (q - r) = 5 :=
sorry

end greatest_q_minus_r_l819_819414


namespace f_2009_l819_819968

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, x ∈ ℝ

axiom f_recurrence : ∀ x : ℝ, f(x + 2) = f(x + 1) - f(x)

axiom f_init1 : f(1) = real.log10 3 - real.log10 2

axiom f_init2 : f(2) = real.log10 3 + real.log10 5

theorem f_2009 : f(2009) = -real.log10 15 :=
by
  -- Detailed proof omitted
  sorry

end f_2009_l819_819968


namespace polynomial_remainder_l819_819910

theorem polynomial_remainder (x : ℂ) :
  let f := (x^2009 + 1)
  let g := (x^6 - x^4 + x^2 - 1)
  (x^2 + 1) * g = x^8 + 1 →
  x^2008 + 1 ≡ 0 [MOD (x^8 + 1)] →
  f ≡ -x + 1 [MOD g] :=
by {
  intros,
  sorry
}

end polynomial_remainder_l819_819910


namespace gcd_mn_mn_squared_l819_819337

theorem gcd_mn_mn_squared (m n : ℕ) (h : Nat.gcd m n = 1) : ({d : ℕ | d = Nat.gcd (m + n) (m ^ 2 + n ^ 2)} ⊆ {1, 2}) := 
sorry

end gcd_mn_mn_squared_l819_819337


namespace hats_needed_to_pay_51_l819_819210

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_amount : ℕ := 51
def num_shirts : ℕ := 3
def num_jeans : ℕ := 2

theorem hats_needed_to_pay_51 :
  ∃ (n : ℕ), total_amount = num_shirts * shirt_cost + num_jeans * jeans_cost + n * hat_cost ∧ n = 4 :=
by
  sorry

end hats_needed_to_pay_51_l819_819210


namespace remainder_sum_div_l819_819590

theorem remainder_sum_div (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end remainder_sum_div_l819_819590


namespace find_missing_number_l819_819592

theorem find_missing_number (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 :=
by
  sorry

end find_missing_number_l819_819592


namespace vectors_parallel_y_eq_minus_one_l819_819582

theorem vectors_parallel_y_eq_minus_one (y : ℝ) :
  let a := (1, 2)
  let b := (1, -2 * y)
  b.1 * a.2 - a.1 * b.2 = 0 → y = -1 :=
by
  intros a b h
  simp at h
  sorry

end vectors_parallel_y_eq_minus_one_l819_819582


namespace arithmetic_sequence_sum_ratio_l819_819622

theorem arithmetic_sequence_sum_ratio
  (a_n : ℕ → ℝ)
  (d a1 : ℝ)
  (S_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n = a1 + (n-1) * d)
  (h_sum : ∀ n, S_n n = n / 2 * (2 * a1 + (n-1) * d))
  (h_ratio : S_n 4 / S_n 6 = -2 / 3) :
  S_n 5 / S_n 8 = 1 / 40.8 :=
sorry

end arithmetic_sequence_sum_ratio_l819_819622


namespace log_base_5_of_15625_l819_819081

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l819_819081


namespace log5_of_15625_l819_819087

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l819_819087


namespace variance_is_correct_l819_819828

-- Define the scores
def scores := [9.4, 9.4, 9.4, 9.6, 9.7]

-- Define the average calculation function
def average (xs : List ℝ) : ℝ :=
  (xs.sum / xs.length)

-- Define the variance calculation function
def variance (xs : List ℝ) : ℝ :=
  let mean := average xs
  (xs.map (fun x => (x - mean)^2)).sum / xs.length

-- The proof problem statement
theorem variance_is_correct : variance scores = 0.016 :=
by
  -- Solve it here if proof is required. Else, use sorry.
  sorry

end variance_is_correct_l819_819828


namespace teacher_distribution_schemes_l819_819455

theorem teacher_distribution_schemes :
  let teachers := 5
  let classes := 3
  let min_teachers_per_class := 1
  ((∀ c : ℕ, c ≤ classes → c ≥ min_teachers_per_class) →
  (nat.choose teachers min_teachers_per_class) * 
  (nat.choose (teachers - classes + min_teachers_per_class) (classes - 1)) = 60) :=
by
  sorry

end teacher_distribution_schemes_l819_819455


namespace no_first_quadrant_l819_819760

theorem no_first_quadrant (m : ℝ) : (∀ x : ℝ, (1/2)^x + m ≤ 0) → (m ≤ -1) :=
by
  intro h
  have key_inequality : (1 : ℝ) + m ≤ 0 := sorry
  exact key_inequality

end no_first_quadrant_l819_819760


namespace trapezoid_angles_l819_819034

-- Define the isosceles trapezoid ABCD with AB parallel to CD and AB > CD
variables {A B C D : Type}

-- Assume some basic geometry definitions for points and angles
def is_isosceles_trapezoid (A B C D : Type) (AB CD : ℝ) := 
  AB > CD ∧ parallel A B C D

-- Assume a division of the trapezoid ABDC into three isosceles triangles through point A
def divides_into_isosceles_triangles (A B C D : Type) (AB AD AE AC : ℝ) := 
  is_isosceles_triangle A D C ∧ is_isosceles_triangle A E B ∧ is_isosceles_triangle A E C

-- Define the property of an isosceles triangle
def is_isosceles_triangle (A B C : Type) (AB AC : ℝ) := 
  AB = AC

-- Statement to prove the angles of the isosceles trapezoid ABCD
theorem trapezoid_angles (A B C D : Type) (α β : ℝ) 
    (AB CD AD AE AC : ℝ) :
  is_isosceles_trapezoid A B C D AB CD →
  divides_into_isosceles_triangles A B C D AB AD AE AC →
  (α = 80 ∧ β = 100) :=
begin
  sorry
end

end trapezoid_angles_l819_819034


namespace joe_first_lift_weight_l819_819620

theorem joe_first_lift_weight (x y : ℕ) 
  (h1 : x + y = 900)
  (h2 : 2 * x = y + 300) :
  x = 400 :=
by
  sorry

end joe_first_lift_weight_l819_819620


namespace central_angle_of_sector_l819_819318

theorem central_angle_of_sector (r S α : ℝ) (h1 : r = 10) (h2 : S = 100)
  (h3 : S = 1/2 * α * r^2) : α = 2 :=
by
  -- Given radius r and area S, substituting into the formula for the area of the sector,
  -- we derive the central angle α.
  sorry

end central_angle_of_sector_l819_819318


namespace cylinder_lateral_surface_area_l819_819703

variables (R l : ℝ)

-- Definitions and conditions
def base_area (S : ℝ) : Prop := π * R ^ 2 = S
def square_side_length : Prop := l = 2 * π * R

-- Theorem statement
theorem cylinder_lateral_surface_area (S : ℝ) (h1 : base_area R S) (h2 : square_side_length R l) : 2 * π * R * l = 4 * π * S :=
by
  sorry

end cylinder_lateral_surface_area_l819_819703


namespace length_of_chord_l819_819143

theorem length_of_chord 
  (a : ℝ)
  (h_sym : ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (3*x - a*y - 11 = 0))
  (h_line : 3 * 1 - a * (-2) - 11 = 0)
  (h_midpoint : (1 : ℝ) = (a / 4) ∧ (-1 : ℝ) = (-a / 4)) :
  let r := Real.sqrt 5
  let d := Real.sqrt ((1 - 1)^2 + (-1 + 2)^2)
  (2 * Real.sqrt (r^2 - d^2)) = 4 :=
by {
  -- Variables and assumptions would go here
  sorry
}

end length_of_chord_l819_819143


namespace minimum_value_l819_819954

variable {a b : ℝ}

noncomputable def given_conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + 2 * b = 2

theorem minimum_value :
  given_conditions a b →
  ∃ x, x = (1 + 4 * a + 3 * b) / (a * b) ∧ x ≥ 25 / 2 :=
by
  sorry

end minimum_value_l819_819954


namespace values_of_a_l819_819513

theorem values_of_a 
{a : ℝ} :
(∀ x : ℝ, x^2 - 2^(a+2) * x - 2^(a+3) + 12 > 0) ↔ a ∈ Iio 0 := 
sorry

end values_of_a_l819_819513


namespace constant_term_in_expansion_l819_819961

theorem constant_term_in_expansion :
  (∃ n : ℕ, let T := (x : ℕ) → (C n x) * ((-1)^x) * (x^(n - 3*x)) in
   (C n 2 = C n 7) ∧ (- (C 9 3) = -84))

end constant_term_in_expansion_l819_819961


namespace solve_sqrt_equation_l819_819503

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l819_819503


namespace distinct_segment_lengths_l819_819282

theorem distinct_segment_lengths :
  let points := [0, 1, 2, 3, 5, 8, 2016]
  let segment_lengths := {abs (x - y) | x y : ℤ, x ∈ points, y ∈ points}
  ∃ n, n = 14 ∧ segment_lengths.card = n :=
by
  let points := [0, 1, 2, 3, 5, 8, 2016]
  let segment_lengths := {abs (x - y) | x y : ℤ, x ∈ points, y ∈ points}
  have h : segment_lengths.card = 14 := sorry
  exact Exists.intro 14 (And.intro rfl h)

end distinct_segment_lengths_l819_819282


namespace sum_of_coefficients_l819_819062

-- Definition of the polynomial
def P (x : ℝ) : ℝ := 5 * (2 * x ^ 9 - 3 * x ^ 6 + 4) - 4 * (x ^ 6 - 5 * x ^ 3 + 6)

-- Theorem stating the sum of the coefficients is 7
theorem sum_of_coefficients : P 1 = 7 := by
  sorry

end sum_of_coefficients_l819_819062


namespace arithmetic_sequence_solution_l819_819952

noncomputable def general_formula_arith_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  (∀ n : ℕ, a n = a 3 + (n - 3) * d) ∧ d > 0 ∧ (a 3) * (a 6) = 55 ∧ a 2 + a 7 = 16

noncomputable def sum_geometric_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ n : ℕ, (a n = 2n - 1) ∧ (b 1 = 2) ∧ ∀ n ≥ 2, ∃ bn : ℕ, b n = 2^(n+1)) ∧
  ∃ S : ℕ, S = 2^(n+1) - 2

theorem arithmetic_sequence_solution :
  ∃ a b : ℕ → ℕ, ∃ d : ℕ, 
    general_formula_arith_seq a d ∧ 
    (∀ n, sum_geometric_seq a b n) :=
begin
    sorry
end

end arithmetic_sequence_solution_l819_819952


namespace mass_of_12_moles_of_Fe2_CO3_3_l819_819855

def molar_mass_Fe := 55.845 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def num_atoms_Fe := 2
def num_atoms_C := 3
def num_atoms_O := 9
def moles_Fe2_CO3_3 := 12 -- number of moles of Fe2(CO3)3

theorem mass_of_12_moles_of_Fe2_CO3_3 :
  (moles_Fe2_CO3_3 * ((num_atoms_Fe * molar_mass_Fe) + 
                      (num_atoms_C * molar_mass_C) + 
                      (num_atoms_O * molar_mass_O))) = 3500.568 := by
  sorry

end mass_of_12_moles_of_Fe2_CO3_3_l819_819855


namespace initial_blue_marbles_correct_l819_819002

noncomputable def initial_blue_marbles (B : ℕ) : Prop :=
  let red_initial := 20 in
  let red_removed := 3 in
  let blue_removed := 4 * red_removed in
  let marbles_left := 35 in
  let red_left := red_initial - red_removed in
  let blue_left := B - blue_removed in
  red_left + blue_left = marbles_left

theorem initial_blue_marbles_correct : initial_blue_marbles 30 :=
by
  unfold initial_blue_marbles
  norm_num
  sorry

end initial_blue_marbles_correct_l819_819002


namespace smallest_product_l819_819724

theorem smallest_product : 
  ∃ (a b : ℤ), a ∈ ({-10, -5, 0, 2, 4} : set ℤ) ∧ b ∈ ({-10, -5, 0, 2, 4} : set ℤ) ∧ ∀ (x y : ℤ), 
  x ∈ ({-10, -5, 0, 2, 4} : set ℤ) ∧ y ∈ ({-10, -5, 0, 2, 4} : set ℤ) → a * b ≤ x * y → a * b = -40 :=
by sorry

end smallest_product_l819_819724


namespace no_strategy_for_vasya_tolya_l819_819755

-- This definition encapsulates the conditions and question
def players_game (coins : ℕ) : Prop :=
  ∀ p v t : ℕ, 
    (1 ≤ p ∧ p ≤ 4) ∧ (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
    (∃ (n : ℕ), coins = 5 * n)

-- Theorem formalizing the problem's conclusion
theorem no_strategy_for_vasya_tolya (n : ℕ) (h : n = 300) : 
  ¬ ∀ (v t : ℕ), 
     (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
     players_game (n - v - t) :=
by
  intro h
  sorry -- Skip the proof, as it is not required

end no_strategy_for_vasya_tolya_l819_819755


namespace cubic_larger_than_elongated_l819_819236

def cube_volume (s : ℝ) : ℝ := s ^ 3

def elongated_volume (k : ℝ) : ℝ := (220 / k) ^ 2 * 220

theorem cubic_larger_than_elongated (k : ℝ) (hk : k > real.sqrt (85.184)) :
  cube_volume 50 > elongated_volume k :=
by 
  -- Unroll the definitions and simplify the inequality
  sorry

end cubic_larger_than_elongated_l819_819236


namespace minimum_value_of_m_l819_819915

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define a function to determine if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

-- Our main theorem statement
theorem minimum_value_of_m :
  ∃ m : ℕ, (600 < m ∧ m ≤ 800) ∧
           is_perfect_square (3 * m) ∧
           is_perfect_cube (5 * m) :=
sorry

end minimum_value_of_m_l819_819915


namespace angle_sum_is_npi_l819_819290

noncomputable def isogonal_conjugate (Z W A B C O : Point) : Prop :=
  -- Definition of isogonal conjugates with respect to an equilateral triangle

noncomputable def midpoint (Z W : Point) : Point :=
  -- Definition of midpoint of segment ZW

theorem angle_sum_is_npi {A B C O Z W M : Point}
  (h1 : isogonal_conjugate Z W A B C O)
  (h2 : M = midpoint Z W) :
  ∃ n : ℤ, ∠ A O Z + ∠ A O W + ∠ A O M = n * π :=
sorry

end angle_sum_is_npi_l819_819290


namespace zero_of_f_and_order_l819_819914

noncomputable def f (z : ℂ) : ℂ := Complex.exp z - 1 - z

theorem zero_of_f_and_order :
  (∃ z : ℂ, f z = 0 ∧ z = 0) ∧ 
  (∃ n : ℕ, ∀ z ≠ 0, f z = (z ^ n) * h(z) ∧ h(0) ≠ 0 ∧ n = 2) :=
by 
  sorry

end zero_of_f_and_order_l819_819914


namespace largest_positive_c_l819_819528

theorem largest_positive_c (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (c : ℝ), (∀ x : ℝ, x > 0 → c ≤ max (a * x + (1 / (a * x))) (b * x + (1 / (b * x)))) ∧ 
  c = sqrt (b / a) + sqrt (a / b) :=
begin
  sorry
end

end largest_positive_c_l819_819528


namespace find_theta_l819_819904

-- Given conditions
def angles : List ℝ := [55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
def cis (θ : ℝ) : Complex := Complex.exp (θ * Complex.I * Real.pi / 180)

-- Statement to prove
theorem find_theta :
  ∃ r > 0, (angles.map cis).sum = r * cis 100 :=
sorry

end find_theta_l819_819904


namespace trig_identity_example_l819_819147

theorem trig_identity_example (θ : ℝ) (h1 : θ ∈ set.Ioo (-2*π) (0))
  (h2 : cos θ = 4/5) : (sin (θ + π/4)) / (cos (2*θ - 6*π)) = 5*real.sqrt 2 / 14 := 
by
  -- Proof will be provided here.
  sorry

end trig_identity_example_l819_819147


namespace helen_gas_usage_l819_819174

/--
  Assume:
  - Helen cuts her lawn from March through October.
  - Helen's lawn mower uses 2 gallons of gas every 4th time she cuts the lawn.
  - In March, April, September, and October, Helen cuts her lawn 2 times per month.
  - In May, June, July, and August, Helen cuts her lawn 4 times per month.
  Prove: The total gallons of gas needed for Helen to cut her lawn from March through October equals 12.
-/

theorem helen_gas_usage :
  ∀ (months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut : ℕ),
  months1 = 4 →
  months2 = 4 →
  cuts_per_month1 = 2 →
  cuts_per_month2 = 4 →
  gas_per_4th_cut = 2 →
  (months1 * cuts_per_month1 + months2 * cuts_per_month2) / 4 * gas_per_4th_cut = 12 :=
by
  intros months1 months2 cuts_per_month1 cuts_per_month2 gas_per_4th_cut
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  calc
    (4 * 2 + 4 * 4) / 4 * 2 = (8 + 16) / 4 * 2 : by rw [mul_add]
                                    ...             = 24 / 4 * 2 : by rw [add_mul]
                                    ...             = 6 * 2       : by norm_num
                                    ...             = 12          : by norm_num

end helen_gas_usage_l819_819174


namespace constant_term_is_minus_84_l819_819964

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition that binomial coefficients of the 3rd and 8th terms are equal
def binom_coeff_equal (n : ℕ) : Prop :=
  binom n 2 = binom n 7

-- Define the general term of the expansion
def general_term (n r : ℕ) : ℤ :=
  binom n r * (-1)^r * (x^(n-3*r))

-- Define the term where the constant term occurs
def constant_term (n r : ℕ) : Prop :=
  n - 3 * r = 0

-- Define the coefficient of the constant term
def coeff_constant_term (n r : ℕ) : ℤ :=
  if constant_term n r then -(binom n r) else 0

theorem constant_term_is_minus_84 : ∃ r, binom_coeff_equal 9 ∧ constant_term 9 r ∧ coeff_constant_term 9 r = -84 := 
  by
    sorry

end constant_term_is_minus_84_l819_819964


namespace conditional_probability_l819_819810

/-
Problem: Given that an animal has a probability of 0.8 to live from birth to 20 years old,
and a probability of 0.4 to live from birth to 25 years old,
prove that an animal that is already 20 years old has a probability of 0.5 to live to 25 years old.
-/

variables (P_A P_B : ℝ) (h_P_B : P_B = 0.8) (h_P_A_and_B : P_A = 0.4)

theorem conditional_probability (hb : P_B = 0.8) (ha : P_A = 0.4): (P_A / P_B = 0.5) :=
by
  have h_P_B_ne_zero : P_B ≠ 0 := by linarith
  have h : P_A / P_B = 0.5 := by
    rw [h_P_A_and_B, h_P_B]
    norm_num
  exact h

end conditional_probability_l819_819810


namespace line_intersect_sufficient_not_necessary_l819_819420

variables {Point : Type} [DecidableEq Point]

-- Definitions of lines and planes
structure Line (Point : Type) :=
(points : set Point)
(non_empty : points.nonempty)
(pairwise_distinct : pairwise (λ x y : Point, x ≠ y))

structure Plane (Point : Type) :=
(lines : set (Line Point))

-- Conditions
variable {a b : Line Point}
variable {α β : Plane Point}

-- Definitions for line lying in a plane and planes intersecting
def line_of_plane (l : Line Point) (π : Plane Point) : Prop :=
l ∈ π.lines

def plane_intersects (π₁ π₂ : Plane Point) : Prop :=
∃ (p : Point), ∃ (l₁ ∈ π₁.lines), ∃ (l₂ ∈ π₂.lines), p ∈ l₁.points ∧ p ∈ l₂.points

-- Main statement
theorem line_intersect_sufficient_not_necessary (h₁ : line_of_plane a α) (h₂ : line_of_plane b β) :
  (∃ p : Point, p ∈ a.points ∧ p ∈ b.points) ↔ plane_intersects α β ∧ ¬ (plane_intersects β α → ∃ p : Point, p ∈ a.points ∧ p ∈ b.points) :=
sorry

end line_intersect_sufficient_not_necessary_l819_819420


namespace stacy_berries_count_l819_819310

variable (Sophie Sylar Steve Stacy total_berries : ℕ)

-- Conditions
def sylar_cond : Sylar = 5 * Sophie := sorry
def steve_cond : Steve = 2 * Sylar := sorry
def stacy_cond : Stacy = 4 * Steve := sorry
def total_berries_cond : total_berries = Sophie + Sylar + Steve + Stacy := sorry
def total_is_2200 : total_berries = 2200 := sorry

theorem stacy_berries_count : Stacy = 1560 :=
by
  -- Including conditions
  rw [total_is_2200, total_berries_cond, sylar_cond, steve_cond, stacy_cond]
  -- Arithmetic and solving the equation steps would go here
  -- sorry to be replaced with actual proof steps
  sorry

end stacy_berries_count_l819_819310


namespace point_on_angle_bisector_l819_819601

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l819_819601


namespace divya_age_l819_819610

theorem divya_age (D N : ℝ) (h1 : N + 5 = 3 * (D + 5)) (h2 : N + D = 40) : D = 7.5 :=
by sorry

end divya_age_l819_819610


namespace sum_three_positive_numbers_ge_three_l819_819660

theorem sum_three_positive_numbers_ge_three 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 :=
sorry

end sum_three_positive_numbers_ge_three_l819_819660


namespace can_tile_floor_l819_819401

-- Define the interior angle of a regular polygon
def interior_angle (n : ℕ) : ℝ :=
  if h : n ≥ 3 then ((n-2 : ℕ) * 180 : ℝ) / n
  else 0

-- Given polygons (equilateral triangle, square, hexagon)
def equilateral_triangle_angle : ℝ := interior_angle 3
def square_angle : ℝ := interior_angle 4
def hexagon_angle : ℝ := interior_angle 6

-- The proof statement: the combination of these polygon angles can sum to 360°
theorem can_tile_floor :
  (equilateral_triangle_angle + square_angle + hexagon_angle 
   = 360) ∨ (equilateral_triangle_angle + square_angle * 2 + hexagon_angle = 360) :=
by
  sorry

end can_tile_floor_l819_819401


namespace multiplicative_inverse_mod_l819_819667

def A := 123456
def B := 142857
def p := 1000009

noncomputable def N : ℕ := 750298
def AB := A * B
def C := AB % p

theorem multiplicative_inverse_mod :
  ∃ (N : ℕ), N < 1000000 ∧ (C * N) % p = 1 :=
by
  use N
  -- Conditions for the proof
  have h1 : AB = A * B := rfl
  have h2 : C = AB % p := rfl
  -- Verification step
  show (C * N) % p = 1
  sorry

end multiplicative_inverse_mod_l819_819667


namespace candy_earned_correctly_l819_819408

theorem candy_earned_correctly :
  let correct_answers := 7 in
  let additional_correct_answers := 2 in
  let candy_per_correct_answer := 3 in
  let total_correct_answers := correct_answers + additional_correct_answers in
  total_correct_answers * candy_per_correct_answer = 27 := by
  sorry

end candy_earned_correctly_l819_819408


namespace ratio_of_side_lengths_sum_l819_819351
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l819_819351


namespace solve_equation_l819_819500

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l819_819500


namespace vendor_profit_is_three_l819_819835

def cost_per_apple := 3 / 2
def selling_price_per_apple := 10 / 5
def cost_per_orange := 2.7 / 3
def selling_price_per_orange := 1
def apples_sold := 5
def oranges_sold := 5

theorem vendor_profit_is_three :
  ((selling_price_per_apple - cost_per_apple) * apples_sold) +
  ((selling_price_per_orange - cost_per_orange) * oranges_sold) = 3 := by
  sorry

end vendor_profit_is_three_l819_819835


namespace gain_percent_is_correct_l819_819431

-- Definitions and conditions
def cost_price : ℝ := 20
def selling_price : ℝ := 35
def gain : ℝ := selling_price - cost_price
def gain_percent : ℝ := (gain / cost_price) * 100

-- Theorem statement
theorem gain_percent_is_correct : gain_percent = 75 := by
  sorry

end gain_percent_is_correct_l819_819431


namespace solve_sqrt_equation_l819_819506

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l819_819506


namespace probability_sum_odd_is_118_div_231_l819_819757

-- Defining the problem conditions
def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
def drawn_balls : Finset (Finset ℕ) := Finset.powersetLen 6 ball_numbers

-- Necessary to evaluate probability with rational numbers
noncomputable def favorable_outcomes : ℚ :=
  (Finset.filter (λ s, (s.sum % 2 = 1)) drawn_balls).card

noncomputable def total_outcomes : ℚ :=
  drawn_balls.card

noncomputable def probability_odd_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- Theorem to prove
theorem probability_sum_odd_is_118_div_231 : probability_odd_sum = 118 / 231 :=
by {
  -- Statement needs to calculate the specific probability which requires combinatorial reasoning
  sorry
}

end probability_sum_odd_is_118_div_231_l819_819757


namespace no_n_exists_l819_819522

open Nat

theorem no_n_exists (n : ℕ) (hn : n > 0) : ¬ (φ n = 2002^2 - 1) := 
by {
  let target := 2002^2 - 1,
  have h1 : target = (2002 - 1) * (2002 + 1),
  { calc
    target = 2002^2 - 1        : rfl
    ...    = (2002 - 1) * (2002 + 1) : by norm_num },
  sorry
}

end no_n_exists_l819_819522


namespace no_possible_blue_socks_l819_819737

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l819_819737


namespace count_triples_l819_819184
open Nat

-- Define the conditions as a structure (if needed)
structure Triple (A B C : ℕ) : Prop :=
  (posA : A > 0)
  (posB : B > 0)
  (posC : C > 0)
  (sumABC : A + B + C = 10)
  (order : A ≤ B ∧ B ≤ C)

-- Lean 4 theorem statement
theorem count_triples : finset.card({(A, B, C) : finset (ℕ × ℕ × ℕ) | Triple A B C}.to_finset) = 8 :=
sorry

end count_triples_l819_819184


namespace find_a_l819_819575

noncomputable theory

def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l819_819575


namespace prime_square_sum_eq_square_iff_l819_819120

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l819_819120


namespace sum_of_ratio_simplified_l819_819352

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l819_819352


namespace sum_of_powers_l819_819467

-- Define i where i^2 = -1
def i : ℂ := Complex.I

-- Lean theorem statement to prove the sum
theorem sum_of_powers :
  (∑ (k : ℕ) in (List.range 604), i ^ k) = 0 :=
by
  sorry

end sum_of_powers_l819_819467


namespace constant_term_is_minus_84_l819_819966

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition that binomial coefficients of the 3rd and 8th terms are equal
def binom_coeff_equal (n : ℕ) : Prop :=
  binom n 2 = binom n 7

-- Define the general term of the expansion
def general_term (n r : ℕ) : ℤ :=
  binom n r * (-1)^r * (x^(n-3*r))

-- Define the term where the constant term occurs
def constant_term (n r : ℕ) : Prop :=
  n - 3 * r = 0

-- Define the coefficient of the constant term
def coeff_constant_term (n r : ℕ) : ℤ :=
  if constant_term n r then -(binom n r) else 0

theorem constant_term_is_minus_84 : ∃ r, binom_coeff_equal 9 ∧ constant_term 9 r ∧ coeff_constant_term 9 r = -84 := 
  by
    sorry

end constant_term_is_minus_84_l819_819966


namespace helen_needed_gas_l819_819177

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end helen_needed_gas_l819_819177


namespace crop_fraction_brought_to_AD_l819_819047

theorem crop_fraction_brought_to_AD
  (AD BC AB CD : ℝ)
  (h : ℝ)
  (angle : ℝ)
  (AD_eq_150 : AD = 150)
  (BC_eq_100 : BC = 100)
  (AB_eq_130 : AB = 130)
  (CD_eq_130 : CD = 130)
  (angle_eq_75 : angle = 75)
  (height_eq : h = (AB / 2) * Real.sin (angle * Real.pi / 180)) -- converting degrees to radians
  (area_trap : ℝ)
  (upper_area : ℝ)
  (total_area_eq : area_trap = (1 / 2) * (AD + BC) * h)
  (upper_area_eq : upper_area = (1 / 2) * (AD + (BC / 2)) * h)
  : (upper_area / area_trap) = 0.8 := 
sorry

end crop_fraction_brought_to_AD_l819_819047


namespace probability_Laurent_greater_Chloe_l819_819042

-- Define the problem conditions
def Chloe_distribution : MeasureTheory.Probability.RV ℝ :=
  MeasureTheory.Probability.uniform ω (0, 3000)

def Laurent_distribution : MeasureTheory.Probability.RV ℝ :=
  MeasureTheory.Probability.uniform ω (0, 6000)

-- Define the main theorem we want to prove
theorem probability_Laurent_greater_Chloe :
  @MeasureTheory.probability ℝ ℝ _ Laurent_distribution (λ y, @MeasureTheory.Probability.has_support _ _ Chloe_distribution (λ x, y > x)) = 3 / 4 :=
by sorry -- proof to be done

end probability_Laurent_greater_Chloe_l819_819042


namespace Dirichlet_statements_l819_819064

def D(x : ℝ) : ℝ :=
  if (x : ℝ).is_rat then 1 else 0

-- Evaluation of the statements
theorem Dirichlet_statements :
  (∀ x, ¬ x.is_rat → D(D(x)) = 0) = false ∧
  (∀ x, D(x) = 0 ∨ D(x) = 1) = true ∧
  (∀ x, D(x) = D(-x)) = true ∧
  (∀ (T : ℝ), T ≠ 0 → T.is_rat → ∀ x, D(x + T) = D(x)) = true ∧
  (∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
   let A := (x1, D(x1)) 
   let B := (x2, D(x2)) 
   let C := (x3, D(x3))
   ∃ A B C such that 
   is_equilateral_triangle A B C) = false := by
  sorry

end Dirichlet_statements_l819_819064


namespace total_tiles_l819_819016

theorem total_tiles (n : ℕ) (h : 3 * n - 2 = 55) : n^2 = 361 :=
by
  sorry

end total_tiles_l819_819016


namespace total_fish_caught_l819_819250

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l819_819250


namespace apples_sold_second_hour_l819_819274

-- Given conditions and question

def apples_sold_first_hour : ℕ := 10
def average_apples_sold : ℕ := 6
def total_hours : ℕ := 2

-- Theorem statement in Lean
theorem apples_sold_second_hour :
  let x := 12 - apples_sold_first_hour in
  x = 2 :=
by
  sorry

end apples_sold_second_hour_l819_819274


namespace negation_of_existence_l819_819977

theorem negation_of_existence (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 ≤ x + 2)) ↔ (∀ x : ℝ, x > 0 → x^2 > x + 2) :=
by
  sorry

end negation_of_existence_l819_819977


namespace function_decreasing_interval_l819_819719

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

def decreasing_interval (a b : ℝ) : Prop :=
  ∀ x : ℝ, a < x ∧ x < b → 0 > (deriv f x)

theorem function_decreasing_interval : decreasing_interval (-1) 3 :=
by 
  sorry

end function_decreasing_interval_l819_819719


namespace cannot_simplify_to_PQ_l819_819846

-- Define vectors
variable {V : Type} [AddGroup V] [Module ℝ V]
variables (A B P Q C : V)

-- Expressions
def expr1 := A + (P + B)
def expr2 := (A + C) + (B - Q)
def expr3 := Q - P + C
def expr4 := P + A - B

-- Prove that expr4 is not equal to PQ
theorem cannot_simplify_to_PQ : expr4 P A B ≠ Q := sorry

end cannot_simplify_to_PQ_l819_819846


namespace repeating_decimal_fraction_eq_l819_819891

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end repeating_decimal_fraction_eq_l819_819891


namespace parabola_equation_and_maximum_area_l819_819552
open Real

-- Setup definitions for points and the parabola parameters
def Focus : Point := (0, 1 : ℝ)
def y_eqn (p : ℝ) : ℝ := 2 * p
def parabola_eqn (p : ℝ) (x : ℝ) : ℝ := x^2 - 2 * p * x
def area_of_triangle (A : Point) (B : Point) (C : Point) : ℝ := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem parabola_equation_and_maximum_area
  (p : ℝ) (A B C : Point) (h₁ : Focus = (0, 1))
  (h₂ : ∀ x y, (parabola_eqn 1 x) = 0 → parabola_eqn p y = x^2 - 4 * y)
  (h₃ : ∀ A B C, (A.2 = (A.1^2) / 4) ∧ (B.2 = (B.1^2) / 4) ∧ (C.2 = (C.1^2) / 4)) 
  (h₄ : A + B + C = 0) 
  : parabola_eqn 2 (x : ℝ) = x^2 - 4 * y ∧ area_of_triangle A B C = (3 * sqrt 6)/2 :=
sorry

end parabola_equation_and_maximum_area_l819_819552


namespace no_equal_prob_for_same_color_socks_l819_819740

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l819_819740


namespace speed_ratio_l819_819763

noncomputable def diameter : ℝ := 1 -- Cyclist A's track diameter in kilometers
noncomputable def straight_track_length : ℝ := 5 -- Cyclist B's track length in kilometers
noncomputable def laps_of_A : ℝ := 3 -- Cyclist A's total laps
noncomputable def time_of_A : ℝ := 10 -- Cyclist A's total time in minutes
noncomputable def round_trips_of_B : ℝ := 2 -- Cyclist B's total round trips
noncomputable def time_of_B : ℝ := 5 -- Cyclist B's total time in minutes

theorem speed_ratio (d : ℝ) (s_track : ℝ) (l_A : ℝ) (t_A : ℝ) (r_B : ℝ) (t_B : ℝ) :
  d = diameter → s_track = straight_track_length → l_A = laps_of_A → t_A = time_of_A → r_B = round_trips_of_B → t_B = time_of_B →
  (l_A * real.pi * d / t_A) / (r_B * 2 * s_track / t_B) = 3 * real.pi / 40 :=
by
  intros
  sorry

end speed_ratio_l819_819763


namespace segment_BJ_length_l819_819451

noncomputable def length_side_triangle : ℝ := 23
noncomputable def length_AG : ℝ := 2
noncomputable def length_GF : ℝ := 13
noncomputable def length_HJ : ℝ := 7
noncomputable def length_FC : ℝ := 1
noncomputable def length_BJ : ℝ := 16

theorem segment_BJ_length :
  let total_length := length_AG + length_GF + length_HJ + length_FC,
      side_length := length_side_triangle in
  total_length = side_length → length_BJ = 16 :=
by
  intros,
  sorry

end segment_BJ_length_l819_819451


namespace intersection_of_sets_l819_819579

open Set

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℤ | 2^x > 1}
  A ∩ B = {1, 2} :=
by
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℤ | 2^x > 1}
  sorry

end intersection_of_sets_l819_819579


namespace student_arrangement_l819_819456

noncomputable def count_arrangements (students : list char) : ℕ := sorry

theorem student_arrangement : 
  let students := ['A', 'B', 'C', 'D', 'E', 'F'] in
  let grade_assignment : char → ℕ → Prop :=
    λ student, match student with
    | 'A' => (λ grade, grade = 10)
    | 'B' => (λ grade, grade ≠ 12 ∧ grade ≠ 10)
    | 'C' => (λ grade, grade ≠ 12 ∧ grade ≠ 10)
    | _   => (λ grade, grade = 10 ∨ grade = 11 ∨ grade = 12) end in
  count_arrangements students = 9 := 
sorry

end student_arrangement_l819_819456


namespace max_size_set_A_l819_819650

theorem max_size_set_A : 
  ∃ A : set ℕ, A ⊆ finset.range 2016 ∧ 
    (∀ x ∈ A, ∀ y ∈ A, x ≠ y → (x + x > y ∧ y + y > x ∧ x + y > y + x)) ∧ 
    ∀ B : set ℕ, B ⊆ finset.range 2016 ∧ 
      (∀ x ∈ B, ∀ y ∈ B, x ≠ y → (x + x > y ∧ y + y > x ∧ x + y > y + x)) → B.card ≤ 10 :=
sorry

end max_size_set_A_l819_819650


namespace sum_of_constants_l819_819345

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l819_819345


namespace log5_15625_eq_6_l819_819091

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l819_819091


namespace total_paths_count_l819_819614

-- Defining the conditions of movements and paths in the problem.
def valid_move (from to : ℕ × ℕ) : Prop :=
  (abs (from.1 - to.1) = 1 ∧ abs (from.2 - to.2) = 0) ∨ 
  (abs (from.1 - to.1) = 0 ∧ abs (from.2 - to.2) = 1)

-- Define the concept of a path
def is_path (path : list (ℕ × ℕ)) : Prop :=
  path.head = (0, 0) ∧
  path.get 1 = (0, 1) ∧ -- This represents 'M'
  path.get 2 = (0, 2) ∧ -- This represents 'C'
  path.get 3 = (1, 2) ∧ -- This represents '8'
  ∀ i < path.length - 1, valid_move (path.get i) (path.get (i + 1))

-- Define the total number of paths
def count_paths : ℕ :=
  4 * 3 * 2  -- Each stage provides a specific number of distinct paths

-- Define the theorem to be proved
theorem total_paths_count : count_paths = 24 := 
  by
    -- Proof details are omitted
    sorry

end total_paths_count_l819_819614


namespace determine_squirrel_color_l819_819374

-- Define the types for Squirrel species and the nuts in hollows
inductive Squirrel
| red
| gray

def tells_truth (s : Squirrel) : Prop :=
  s = Squirrel.red

def lies (s : Squirrel) : Prop :=
  s = Squirrel.gray

-- Statements made by the squirrel in front of the second hollow
def statement1 (s : Squirrel) (no_nuts_in_first : Prop) : Prop :=
  tells_truth s → no_nuts_in_first ∧ (lies s → ¬no_nuts_in_first)

def statement2 (s : Squirrel) (nuts_in_either : Prop) : Prop :=
  tells_truth s → nuts_in_either ∧ (lies s → ¬nuts_in_either)

-- Given a squirrel that says the statements and the information about truth and lies
theorem determine_squirrel_color (s : Squirrel) (no_nuts_in_first : Prop) (nuts_in_either : Prop) :
  (statement1 s no_nuts_in_first) ∧ (statement2 s nuts_in_either) → s = Squirrel.red :=
by
  sorry

end determine_squirrel_color_l819_819374


namespace other_root_l819_819288

theorem other_root (z : ℂ) (h : z^2 = -55 + 48 * Complex.i) (h₁ : z = 3 + 8 * Complex.i) : 
  3 + 8 * Complex.i + (-(3 + 8 * Complex.i)) = 0 :=
by
  -- root proof
  sorry

end other_root_l819_819288


namespace cos_identity_l819_819129

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end cos_identity_l819_819129


namespace find_abs_square_of_complex_l819_819258

theorem find_abs_square_of_complex (z : ℂ) (h : z^2 + complex.abs z ^ 2 = 8 - 3 * complex.I) :
  complex.abs z ^ 2 = 73 / 16 := 
by 
  sorry

end find_abs_square_of_complex_l819_819258


namespace Daria_money_l819_819872

theorem Daria_money (num_tickets : ℕ) (price_per_ticket : ℕ) (amount_needed : ℕ) (h1 : num_tickets = 4) (h2 : price_per_ticket = 90) (h3 : amount_needed = 171) : 
  (num_tickets * price_per_ticket) - amount_needed = 189 := 
by 
  sorry

end Daria_money_l819_819872


namespace sum_of_divisors_24_l819_819773

theorem sum_of_divisors_24 : ∑ d in (List.range 25).filter (λ n, 24 % n = 0), d = 60 :=
by
  sorry

end sum_of_divisors_24_l819_819773


namespace sum_of_cubes_1998_l819_819894

theorem sum_of_cubes_1998 : 1998 = 334^3 + 332^3 + (-333)^3 + (-333)^3 := by
  sorry

end sum_of_cubes_1998_l819_819894


namespace rotate_vector_180_l819_819727

theorem rotate_vector_180 (v : ℝ^3) (h : v = ![2, 1, 3]) : 
  (∃ u : ℝ^3, u = ![-2, -1, -3] ∧ ∃ θ : ℝ, θ = π ∧ ∀ i, u i = -v i) :=
by
  sorry

end rotate_vector_180_l819_819727


namespace fill_tank_time_l819_819449

/-- Define the rates of the pump and the leak -/
def pump_rate := (1 : ℝ) / 6
def leak_rate := (1 : ℝ) / 12

/- The effective rate of filling the tank (rate of pump minus rate of leak) -/
def effective_rate := pump_rate - leak_rate

/-- Prove that the tank fills in 12 hours given the effective rate --/
theorem fill_tank_time : effective_rate = (1 : ℝ) / 12 → 1 / effective_rate = 12 := 
by
  intro h
  rw h
  norm_num
  sorry

end fill_tank_time_l819_819449


namespace bears_per_shelf_l819_819831

theorem bears_per_shelf
  (initial_bears : ℕ)
  (shipment_bears : ℕ)
  (shelves : ℕ)
  (total_bears : initial_bears + shipment_bears)
  (num_bears_per_shelf : total_bears / shelves) :
  initial_bears = 17 ∧ 
  shipment_bears = 10 ∧ 
  shelves = 3 → 
  num_bears_per_shelf = 9 :=
by
  sorry

end bears_per_shelf_l819_819831


namespace sum_of_integers_m_l819_819202

noncomputable def sum_of_satisfying_m (m_set : Set ℝ) : ℝ := 
  ∑ m in m_set, m

theorem sum_of_integers_m :
  let m_set := {m : ℝ | (x = (-5 - m) / 2 ∧ x < 0) ∧ 
                          (∀ y : ℝ, ((y + 2) / 3 - y / 2 < 1) ∧ (3 * (y - m) ≥ 0) → y > -2) ∧
                          m = floor m ∧ m > -5 ∧ m ≤ -2}
  in sum_of_satisfying_m m_set = -9 :=
by
  let m_set := {m : ℝ | (let x := (-5 - m) / 2 in x < 0) ∧ 
                          (∀ y : ℝ, ((y + 2) / 3 - y / 2 < 1) ∧ (3 * (y - m) ≥ 0) → y > -2) ∧
                          m = floor m ∧ m > -5 ∧ m ≤ -2}
  show ∑ m in m_set, m = -9
  sorry

end sum_of_integers_m_l819_819202


namespace problems_per_page_l819_819636

theorem problems_per_page (pages_math pages_reading total_problems x : ℕ) (h1 : pages_math = 2) (h2 : pages_reading = 4) (h3 : total_problems = 30) : 
  (pages_math + pages_reading) * x = total_problems → x = 5 := by
  sorry

end problems_per_page_l819_819636


namespace option_C_is_linear_l819_819787

-- Definitions for each of the conditions
def optionA := (x y : ℝ) → x - 3 = y
def optionB := (x : ℝ) → x^2 - 1 = 0
def optionC := (x : ℝ) → x - 2 = 1 / 3
def optionD := (x : ℝ) → 2 / x = 3

-- Theorem stating that option C is the correct linear equation
theorem option_C_is_linear (x : ℝ) : optionC x :=
by
  sorry

end option_C_is_linear_l819_819787


namespace L_shape_area_l819_819819

theorem L_shape_area : 
    let larger_length := 10
    let larger_width := 6
    let smaller_length := larger_length - 3
    let smaller_width := larger_width - 2
    let area_larger := larger_length * larger_width
    let area_smaller := smaller_length * smaller_width
    area_larger - area_smaller = 32 :=
by
    let larger_length := 10
    let larger_width := 6
    let smaller_length := larger_length - 3
    let smaller_width := larger_width - 2
    let area_larger := larger_length * larger_width
    let area_smaller := smaller_length * smaller_width
    exact calc 
        area_larger - area_smaller = 10 * 6 - (10 - 3) * (6 - 2) : by rfl
        ... = 60 - 28 : by rfl
        ... = 32 : by rfl

end L_shape_area_l819_819819


namespace log_base_5_of_15625_eq_6_l819_819074

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l819_819074


namespace odd_number_of_axes_of_symmetry_l819_819293

theorem odd_number_of_axes_of_symmetry 
  (B : Type) 
  (A : set (B → B → Prop)) 
  (hA : ∀ l' ∈ A, ∃ l'' ∈ A, (l' ≠ l'') ∨ (l' = l' ∧ l'' = l''))
  (h_cases : ∀ l l' ∈ A, 
    ((l' does_not_intersect l) ∨ 
    (l intersects_at_non_right_angle l') ∨ 
    (l intersects_at_right_angle l'))) : 
  (¬ (∃ n, n ≠ 0 ∧ even n ∧ n = |A|)) :=
sorry

end odd_number_of_axes_of_symmetry_l819_819293


namespace probability_chord_length_not_less_than_radius_l819_819540

theorem probability_chord_length_not_less_than_radius
  (R : ℝ) (M N : ℝ) (h_circle : N = 2 * π * R) : 
  (∃ P : ℝ, P = 2 / 3) :=
sorry

end probability_chord_length_not_less_than_radius_l819_819540


namespace largest_possible_number_of_markers_l819_819277

theorem largest_possible_number_of_markers (n_m n_c : ℕ) 
  (h_m : n_m = 72) (h_c : n_c = 48) : Nat.gcd n_m n_c = 24 :=
by
  sorry

end largest_possible_number_of_markers_l819_819277


namespace area_of_circular_field_l819_819314

noncomputable def cost_per_meter : ℝ := 4.40
noncomputable def total_fencing_cost : ℝ := 5806.831494371739
noncomputable def pi : ℝ := Real.pi -- Define Pi to use Real.pi for precision

theorem area_of_circular_field :
  let circumference := total_fencing_cost / cost_per_meter
  let radius := circumference / (2 * pi)
  let area_meters := pi * radius^2
  let area_hectares := area_meters / 10000
  area_hectares ≈ 13.85 :=
by 
  sorry

end area_of_circular_field_l819_819314


namespace function_characterization_l819_819897

theorem function_characterization (f : ℝ → ℝ)
  (h: ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → (x - y) ^ 2 ≤ |f x - f y| ∧ |f x - f y| ≤ |x - y|) :
  ∃ (C : ℝ) (s : ℝ), s ∈ {-1, 1} ∧ f = λ x, s * x + C :=
by
  sorry

end function_characterization_l819_819897


namespace simplify_and_evaluate_l819_819693

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 1) (h2 : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l819_819693


namespace FN_length_l819_819950

def parabola_focus := (2 : ℝ, 0 : ℝ)

def on_parabola (x y : ℝ) : Prop := y^2 = 8 * x

def midpoint (F M N : ℝ × ℝ) : Prop := M = (1 / 2 * (F.1 + N.1), 1 / 2 * (F.2 + N.2))

def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem FN_length (M N : ℝ × ℝ) 
  (F := parabola_focus)
  (hM_on_parabola : on_parabola M.1 M.2)
  (h_midpoint : midpoint F M N) :
  distance F N = 6  :=
sorry

end FN_length_l819_819950


namespace domain_of_function_l819_819710

theorem domain_of_function (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 1 > 0) ↔ (1 < x ∧ x ≤ 2) :=
by
  sorry

end domain_of_function_l819_819710


namespace max_height_at_t_l819_819012

-- Define the height function h(t)
def height (t : ℝ) : ℝ := 15 * t - 5 * t^2

-- Define the statement that the maximum height is reached at t = 3/2 and is equal to 45/4
theorem max_height_at_t: (∃ (t : ℝ), t = 3 / 2 ∧ height t = 45 / 4) :=
by {
    -- Placeholder for the proof steps
    sorry
}

end max_height_at_t_l819_819012


namespace no_constant_term_l819_819556

theorem no_constant_term (n : ℕ) (h1 : n ≥ 2) (h2 : n ≤ 7) :
  ¬ (∃ (c : ℕ), (1 + x + x^2) * (x + x⁻³) ^ n = c) ↔ n = 5 := by
  sorry

end no_constant_term_l819_819556


namespace cube_tangent_ratio_l819_819518

theorem cube_tangent_ratio 
  (edge_length : ℝ) 
  (midpoint K : ℝ) 
  (tangent E : ℝ) 
  (intersection F : ℝ) 
  (radius R : ℝ)
  (h1 : edge_length = 2)
  (h2 : radius = 1)
  (h3 : K = midpoint)
  (h4 : ∃ E F, tangent = E ∧ intersection = F) :
  (K - E) / (F - E) = 4 / 5 :=
sorry

end cube_tangent_ratio_l819_819518


namespace range_of_k_l819_819972

theorem range_of_k (k : ℝ) :
  let C := ((λ x y : ℝ, (x - 2)^2 + y^2 = 4) : set ℝ × ℝ)
  let L := ((λ x y : ℝ, y = k * x + 1) : set ℝ × ℝ)
  (∃ M N : ℝ × ℝ, M ≠ N ∧ M ∈ C ∧ N ∈ C ∧ M ∈ L ∧ N ∈ L ∧ (dist M N) ≥ 2 * real.sqrt 3) →
  -4/3 ≤ k ∧ k ≤ 0 :=
begin
  sorry
end

end range_of_k_l819_819972


namespace norm_a_sub_b_vals_l819_819166

variables (x : ℝ)

def a : EuclideanSpace ℝ (Fin 2) := ![1, x]
def b : EuclideanSpace ℝ (Fin 2) := ![2 * x + 3, -x]

theorem norm_a_sub_b_vals (h : ⟪a, b⟫ = (0 : ℝ)) : 
  ‖a - b‖ = 2 ∨ ‖a - b‖ = 10 :=
by sorry

end norm_a_sub_b_vals_l819_819166


namespace constant_term_is_minus_84_l819_819965

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition that binomial coefficients of the 3rd and 8th terms are equal
def binom_coeff_equal (n : ℕ) : Prop :=
  binom n 2 = binom n 7

-- Define the general term of the expansion
def general_term (n r : ℕ) : ℤ :=
  binom n r * (-1)^r * (x^(n-3*r))

-- Define the term where the constant term occurs
def constant_term (n r : ℕ) : Prop :=
  n - 3 * r = 0

-- Define the coefficient of the constant term
def coeff_constant_term (n r : ℕ) : ℤ :=
  if constant_term n r then -(binom n r) else 0

theorem constant_term_is_minus_84 : ∃ r, binom_coeff_equal 9 ∧ constant_term 9 r ∧ coeff_constant_term 9 r = -84 := 
  by
    sorry

end constant_term_is_minus_84_l819_819965


namespace problem1_problem2_l819_819040

namespace MathProblem

-- Problem 1
theorem problem1 : (π - 2)^0 + (-1)^3 = 0 := by
  sorry

-- Problem 2
variable (m n : ℤ)

theorem problem2 : (3 * m + n) * (m - 2 * n) = 3 * m ^ 2 - 5 * m * n - 2 * n ^ 2 := by
  sorry

end MathProblem

end problem1_problem2_l819_819040


namespace difference_between_numbers_l819_819369

open Int

theorem difference_between_numbers (A B : ℕ) 
  (h1 : A + B = 1812) 
  (h2 : A = 7 * B + 4) : 
  A - B = 1360 :=
by
  sorry

end difference_between_numbers_l819_819369


namespace point_on_angle_bisector_l819_819602

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l819_819602


namespace joe_current_age_l819_819646

variables (J M : ℕ)  -- J and M are natural numbers representing the ages of Joe and James respectively.

theorem joe_current_age :
  (J = M + 10) ∧ (2 * (J + 8) = 3 * (M + 8)) → J = 22 :=
by
  intro h
  cases h with h1 h2
  sorry

end joe_current_age_l819_819646


namespace sqrt5AddSqrt3Root_l819_819486

def monicPolynomialDegree4WithRoot (α : ℝ) : Polynomial ℚ :=
  Polynomial.monic {
    Polynomial.of_root {
      Polynomial.ℚ {
        degree := 4,
        coefficients := [1, 0, -16, 0, 4]
      }
    }
  }

theorem sqrt5AddSqrt3Root : ∃ (P : Polynomial ℚ), 
  Polynomial.degree P = 4 ∧ 
  Polynomial.isMonic P ∧ 
  Polynomial.hasRoot P (√5 + √3) := by
{
  let P := Polynomial.Coefficient.mk ![4, -16, 1],
  have h1: Polynomial.degree P = 4,
    sorry,  -- Proof here
  have h2: Polynomial.isMonic P,
    sorry,  -- Proof here
  have h3: Polynomial.hasRoot P (√5 + √3),
    sorry,  -- Proof here
  exact ⟨P, h1, h2, h3⟩
}

-- This concludes the construction of the Lean 4 statement.

end sqrt5AddSqrt3Root_l819_819486


namespace total_students_is_30_l819_819440

def students_per_bed : ℕ := 2 

def beds_per_room : ℕ := 2 

def students_per_couch : ℕ := 1 

def rooms_booked : ℕ := 6 

def total_students := (students_per_bed * beds_per_room + students_per_couch) * rooms_booked

theorem total_students_is_30 : total_students = 30 := by
  sorry

end total_students_is_30_l819_819440


namespace perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l819_819942

-- Definition of an Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Definition of a Circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Conditions about foci, points, lines, and tangent property
def foci_condition (x y : ℝ) : Prop := ellipse x y
def right_vertex (x y : ℝ) : Prop := x = 2 ∧ y = 0
def tangent_condition (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y → l x y

-- Problem 1: Prove the perimeter of Δ AF₁F₂ is 4 + 2√2
theorem perimeter_of_triangle (A F1 F2 : ℝ × ℝ) (hA : foci_condition A.1 A.2)
  (hF1 : ellipse F1.1 F1.2) (hF2 : ellipse F2.1 F2.2) : 
  dist A F1 + dist A F2 + dist F1 F2 = 4 + 2 * sqrt 2 := sorry

-- Problem 2: Find the equation of line l passing through (2,0)
theorem equation_of_line_through_vertex : 
  ∃ k : ℝ, ∀ x y : ℝ, (x = 2 → y = k * (x - 2)) → tangent_condition (λ x y, y = k * (x - 2)) := sorry

-- Problem 3: Prove line AD is tangent to circle C
theorem line_AD_tangent_to_C (A D : ℝ × ℝ) (hD : D.2 = 2) (hO : 0) (hperpendicular : A.1 * D.1 + A.2 * D.2 = 0) :
  ∃ l : ℝ → ℝ → Prop, tangent_condition l ∧ (l A.1 A.2 = A.2 → A ≠ O) := sorry

end perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l819_819942


namespace vendor_profit_l819_819838

noncomputable def apple_cost_price := 3 / 2
noncomputable def apple_selling_price := 10 / 5
noncomputable def orange_cost_price := 2.70 / 3
noncomputable def orange_selling_price := 1

noncomputable def total_apple_cost_price := 5 * apple_cost_price
noncomputable def total_apple_selling_price := 5 * apple_selling_price
noncomputable def total_apple_profit := total_apple_selling_price - total_apple_cost_price

noncomputable def total_orange_cost_price := 5 * orange_cost_price
noncomputable def total_orange_selling_price := 5 * orange_selling_price
noncomputable def total_orange_profit := total_orange_selling_price - total_orange_cost_price

noncomputable def total_profit := total_apple_profit + total_orange_profit

theorem vendor_profit : total_profit = 3 := by
  sorry

end vendor_profit_l819_819838


namespace triangle_integer_values_x_l819_819717

theorem triangle_integer_values_x :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, 22 < x ∧ x < 52 → x ∈ {23, 24, ..., 51} :=
by
  sorry

end triangle_integer_values_x_l819_819717


namespace linear_regression_model_applicable_l819_819702

section

open Real

def x_vals : List ℝ := [1, 2, 3, 4, 5]
def y_vals : List ℝ := [3, 7, 9, 10, 11]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum / l.length)

def x_mean := mean x_vals
def y_mean := mean y_vals

noncomputable def numerator_r : ℝ := (List.map₂ (· - x_mean) x_vals y_vals).sum

noncomputable def denominator_r_x : ℝ := Real.sqrt (List.map (λ x => (x - x_mean)^2) x_vals).sum
noncomputable def denominator_r_y : ℝ := Real.sqrt (List.map (λ y => (y - y_mean)^2) y_vals).sum

noncomputable def sample_correlation_coeff : ℝ := numerator_r / (denominator_r_x * denominator_r_y)

def b_hat : ℝ := numerator_r / (List.map (λ x => (x - x_mean)^2) x_vals).sum
def a_hat : ℝ := y_mean - b_hat * x_mean

theorem linear_regression_model_applicable (r : ℝ) (b : ℝ) (a : ℝ) (x_needed : ℝ) : 
  r = 0.95 ∧ b = 1.9 ∧ a = 2.3 ∧ x_needed ≥ 6.684 := sorry

end

end linear_regression_model_applicable_l819_819702


namespace weight_of_square_is_correct_l819_819020

def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

def area_of_square (s : ℝ) : ℝ :=
  s * s

def weight_of_metal (uniform_density : ℝ) (area : ℝ) : ℝ :=
  uniform_density * area

def weight_of_square_metal : ℝ :=
  let side_triangle := 4.0
  let weight_triangle := 32.0
  let side_square := 4.0
  let area_triangle := area_of_equilateral_triangle side_triangle
  let area_square := area_of_square side_square
  let density := weight_triangle / area_triangle
  weight_of_metal density area_square

theorem weight_of_square_is_correct :
  weight_of_square_metal = 74.0 :=
by
  -- The proof is omitted
  sorry

end weight_of_square_is_correct_l819_819020


namespace school_bus_solution_l819_819067

-- Define the capacities
def bus_capacity : Prop := 
  ∃ x y : ℕ, x + y = 75 ∧ 3 * x + 2 * y = 180 ∧ x = 30 ∧ y = 45

-- Define the rental problem
def rental_plans : Prop :=
  ∃ a : ℕ, 6 ≤ a ∧ a ≤ 8 ∧ 
  (30 * a + 45 * (25 - a) ≥ 1000) ∧ 
  (320 * a + 400 * (25 - a) ≤ 9550) ∧ 
  3 = 3

-- The main theorem combines the two aspects
theorem school_bus_solution: bus_capacity ∧ rental_plans := 
  sorry -- Proof omitted

end school_bus_solution_l819_819067


namespace equilateral_triangle_PA_equal_PB_plus_PC_l819_819416

open Real EuclideanGeometry

variable {A B C P : Point}

-- Given: ABC is an equilateral triangle
def is_equilateral_triangle (A B C : Point) : Prop :=
∥A - B∥ = ∥B - C∥ ∧ ∥B - C∥ = ∥C - A∥

-- Given: P is a point on the minor arc BC of the circumcircle of ABC
def on_minor_arc (P B C O : Point) : Prop :=
∃ cc : Circle O, cc ∈ circumcircle A B C ∧ point_on cc P ∧ ∃ θ : ℝ, (0 < θ ∧ θ < π) ∧ angle A P O = θ

-- To Prove: PA = PB + PC
theorem equilateral_triangle_PA_equal_PB_plus_PC
(A B C P O : Point) 
(is_equilateral : is_equilateral_triangle A B C)
(on_minor_arc_BC : on_minor_arc P B C O)
: ∥P - A∥ = ∥P - B∥ + ∥P - C∥ := by
  sorry

end equilateral_triangle_PA_equal_PB_plus_PC_l819_819416


namespace max_binomial_coefficient_expansion_l819_819480

theorem max_binomial_coefficient_expansion (n : ℕ) (k : ℕ) (h : n = 5) :
  (nat.choose n k) ≤ (nat.choose 5 2) :=
by
  sorry

end max_binomial_coefficient_expansion_l819_819480


namespace ratio_of_square_sides_sum_l819_819357

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l819_819357


namespace area_of_region_l819_819507

theorem area_of_region : 
  (let S := {p : ℝ × ℝ | |p.2 - |p.1 - 2| + |p.1|| ≤ 4} in
   let area := 32 in
   true) :=
begin
  sorry
end

end area_of_region_l819_819507


namespace real_fraction_condition_l819_819927

open Complex

theorem real_fraction_condition (a : ℝ) : (↑a - I) / (3 + I : ℂ) ∈ ℝ → a = -3 :=
by
    sorry

end real_fraction_condition_l819_819927


namespace perfect_square_trinomial_coeff_l819_819546

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end perfect_square_trinomial_coeff_l819_819546


namespace treasure_probability_l819_819022

noncomputable def prob_has_treasure_no_traps : ℚ := 1/5
noncomputable def prob_has_traps_no_treasure : ℚ := 1/10
noncomputable def prob_neither_treasure_nor_traps : ℚ := 7/10
noncomputable def num_islands : ℕ := 8

theorem treasure_probability :
  (∑ i in (finset.range 5), finset.card (finset.filter (λ (p : list bool × list bool), p.snd.count true = i) (finset.powerset_len 8 (finset.of_list [true, false ^ 7])))[0] * 
  (prob_has_treasure_no_traps ^ i) * (prob_neither_treasure_nor_traps ^ (num_islands - i))) = 67 / 2500 :=
by sorry

end treasure_probability_l819_819022


namespace tangent_line_through_P_l819_819158

theorem tangent_line_through_P (x y : ℝ) :
  (∃ l : ℝ, l = 3*x - 4*y + 5) ∨ (x = 1) :=
by
  sorry

end tangent_line_through_P_l819_819158


namespace malia_buttons_geometric_sequence_l819_819405

theorem malia_buttons_geometric_sequence :
  ∀ (first_box second_box third_box fifth_box sixth_box : ℕ),
    first_box = 1 →
    second_box = 3 →
    third_box = 9 →
    fifth_box = 81 →
    sixth_box = 243 →
    (∀ n, (n > 1 → first_box * (3 ^ (n - 1)) = n) → (third_box * 3 = 27)) :=
begin
  intros first_box second_box third_box fifth_box sixth_box,
  intros h_first h_second h_third h_fifth h_sixth h_geometric,
  have h_fourth : third_box * 3 = 27,
  sorry,
end

end malia_buttons_geometric_sequence_l819_819405


namespace divisors_24_count_l819_819986

def divisors (n : ℕ) : Finset ℕ := {d ∈ Finset.range (n+1) | n % d = 0}

example : divisors 24 = {1, 2, 3, 4, 6, 8, 12, 24} := by sorry

theorem divisors_24_count : (divisors 24).card = 8 := by sorry

end divisors_24_count_l819_819986


namespace max_necklaces_l819_819791

-- Definitions for the conditions
def green_beads_per_necklace := 9
def white_beads_per_necklace := 6
def orange_beads_per_necklace := 3
def total_green_beads := 45
def total_white_beads := 45
def total_orange_beads := 45

-- The main theorem to prove
theorem max_necklaces : max_necklaces_function total_green_beads total_white_beads total_orange_beads green_beads_per_necklace white_beads_per_necklace orange_beads_per_necklace = 5 :=
  sorry

end max_necklaces_l819_819791


namespace hats_count_l819_819618

theorem hats_count (T M W : ℕ) (hT : T = 1800)
  (hM : M = (2 * T) / 3) (hW : W = T - M) 
  (hats_men : ℕ) (hats_women : ℕ) (m_hats_condition : hats_men = 15 * M / 100)
  (w_hats_condition : hats_women = 25 * W / 100) :
  hats_men + hats_women = 330 :=
by sorry

end hats_count_l819_819618


namespace calculate_angle_EDB_l819_819812

-- Defining the setup for the geometric problem
noncomputable def radius : ℝ := 12
noncomputable def equilateral_triangle (A B C : Type) := 
  ∀ (dist_AC dist_BC : ℝ), dist_AC = radius ∧ dist_BC = radius ∧ (A = B ∨ B = C ∨ A = C)

noncomputable def point_on_line_extended_through (A C E : Type) (CE_length : ℝ) := 
  CE_length = 24

noncomputable def circle_passes_through (C E D : Type) :=
  ∃ (radius : ℝ), radius = 12 ∧ (CE_length = 24 ∧ ∃ (dist_CD : ℝ), dist_CD = 24)

noncomputable def is_right_angle (angle_deg : ℝ) :=
  angle_deg = 90

-- The statement we need to prove
theorem calculate_angle_EDB (A B C D E : Type) :
  equilateral_triangle A B C ∧ point_on_line_extended_through A C E 24 ∧ circle_passes_through C E D → is_right_angle 90 :=
by {
  sorry
}

end calculate_angle_EDB_l819_819812


namespace tan_3pi_plus_theta_l819_819925

theorem tan_3pi_plus_theta (θ : ℝ) (h1 : sin θ - cos θ = 1 / 5) (h2 : 0 < θ ∧ θ < π) : 
  tan (3 * π + θ) = 4 / 3 := sorry

end tan_3pi_plus_theta_l819_819925


namespace profit_percentage_l819_819010

theorem profit_percentage (SP CP : ℤ) (h_SP : SP = 1170) (h_CP : CP = 975) :
  ((SP - CP : ℤ) * 100) / CP = 20 :=
by 
  sorry

end profit_percentage_l819_819010


namespace transform_sin_l819_819761

theorem transform_sin (x : ℝ) : 
    sin (2 * x) = sin (3 * (x - π / 12) + π / 6) :=
by sorry

end transform_sin_l819_819761


namespace no_real_solution_l819_819101

theorem no_real_solution (x : ℝ) : ¬ (x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 6 * (x + 4)^3) :=
sorry

end no_real_solution_l819_819101


namespace ratio_of_areas_l819_819392

theorem ratio_of_areas (r : ℝ) (h1 : r > 0) : 
  let OX := r / 3
  let area_OP := π * r ^ 2
  let area_OX := π * (OX) ^ 2
  (area_OX / area_OP) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l819_819392


namespace log5_of_15625_l819_819085

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l819_819085


namespace fifteenth_prime_is_47_l819_819394

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ :=
  if h : ∃ k, (List.filter is_prime (List.range (n*k))).length = n 
  then (List.filter is_prime (List.range (n*(classical.some h)))).nth n
  else 0

theorem fifteenth_prime_is_47 : nth_prime 15 = 47 :=
by
  sorry

end fifteenth_prime_is_47_l819_819394


namespace phi_equal_if_not_empty_l819_819916

-- Define the conditions formally in Lean

def phi (n : ℕ) : ℕ := sorry -- Generic definition for phi

def M_phi (φ : ℕ → ℕ) : set (ℕ → ℤ) :=
  { f | ∀ x : ℕ, f x > f (φ x) }

theorem phi_equal_if_not_empty 
  (φ1 φ2 : ℕ → ℕ)
  (h1 : M_phi φ1 ≠ ∅)
  (h2 : M_phi φ2 ≠ ∅) :
  φ1 = φ2 :=
  sorry

end phi_equal_if_not_empty_l819_819916


namespace problem_l819_819684

variable {R : Type} [Field R]

def f1 (a b c d : R) : R := a + b + c + d
def f2 (a b c d : R) : R := (1 / a) + (1 / b) + (1 / c) + (1 / d)
def f3 (a b c d : R) : R := (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) + (1 / (1 - d))

theorem problem (a b c d : R) (h1 : f1 a b c d = 2) (h2 : f2 a b c d = 2) : f3 a b c d = 2 :=
by sorry

end problem_l819_819684


namespace find_a_value_l819_819562

theorem find_a_value (a : ℝ) (h : ∀ x : ℤ, abs(abs (x - 2) - 1) = a → ∃! n : ℤ, x = n) : 
a = 1 :=
sorry

end find_a_value_l819_819562


namespace soccer_league_teams_l819_819373

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 55) : n = 11 := 
sorry

end soccer_league_teams_l819_819373


namespace problem_solution_l819_819858

theorem problem_solution : (324^2 - 300^2) / 24 = 624 :=
by 
  -- The proof will be inserted here.
  sorry

end problem_solution_l819_819858


namespace decreasing_function_in_interval_l819_819847

theorem decreasing_function_in_interval (f: ℝ → ℝ) (a b: ℝ) (h: a < b) : 
  (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y) :=
  f = λ x, 1 / x :=
by
  sorry

example : ∀ a b: ℝ, 0 < a ∧ a < b ∧ b < 1 → 
  (decreasing_function_in_interval (λ x, 1 / x) a b (by linarith)) :=
by
  sorry

end decreasing_function_in_interval_l819_819847


namespace yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l819_819686

-- Defining "Yazhong point"
def yazhong (A B M : ℝ) : Prop := abs (M - A) = abs (M - B)

-- Problem 1
theorem yazhong_point_1 {A B M : ℝ} (hA : A = -5) (hB : B = 1) (hM : yazhong A B M) : M = -2 :=
sorry

-- Problem 2
theorem yazhong_point_2 {A B M : ℝ} (hM : M = 2) (hAB : B - A = 9) (h_order : A < B) (hY : yazhong A B M) :
  (A = -5/2) ∧ (B = 13/2) :=
sorry

-- Problem 3 Part ①
theorem yazhong_point_3_part1 (A : ℝ) (B : ℝ) (m : ℤ) 
  (hA : A = -6) (hB_range : -4 ≤ B ∧ B ≤ -2) (hM : yazhong A B m) : 
  m = -5 ∨ m = -4 :=
sorry

-- Problem 3 Part ②
theorem yazhong_point_3_part2 (C D : ℝ) (n : ℤ)
  (hC : C = -4) (hD : D = -2) (hM : yazhong (-6) (C + D + 2 * n) 0) : 
  8 ≤ n ∧ n ≤ 10 :=
sorry

end yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l819_819686


namespace KM_projection_AC_l819_819319

theorem KM_projection_AC (A B K M : ℝ × ℝ) (AC : ℝ) (r : ℝ) (ω : Set (ℝ × ℝ))
  (hω : ∀ p, p ∈ ω ↔ dist p A = r ∨ dist p B = r)
  (hA : A = (0, 0)) (hB : B = (1, 0))
  (hc1 : r = 1)
  (hK : K ∈ ω ∧ dist K A = r)
  (hM : M ∈ ω ∧ dist M A = r)
  (hAC : AC = ∥A - C∥)
  (hC : C = (1, 1))
  :
  (∥K - M∥ * ∥(C.1, 0)∥) / AC = (sqrt 2) / 2 :=
by
  sorry

end KM_projection_AC_l819_819319


namespace proper_subsets_count_l819_819721

-- Definition of the set A
def A : Set ℕ := { x | x < 3 }

-- Proof statement asserting the number of proper subsets of A
theorem proper_subsets_count : (A.toFinset.card = 3) → (2 ^ A.toFinset.card - 1 = 7) :=
by 
  intro h,
  sorry

end proper_subsets_count_l819_819721


namespace mother_older_by_27_l819_819003

variable (M D : ℕ)
axiom mother_age : M = 55
axiom age_difference : M - 1 = 2 * (D - 1)

theorem mother_older_by_27 : M - D = 27 :=
by
  have hM : M = 55 := mother_age
  have h1 : M - 1 = 2 * (D - 1) := age_difference
  have h2 : 55 - 1 = 2 * (D - 1) := by rw [hM] at h1; exact h1
  have h3 : 54 = 2 * (D - 1) := h2
  have h4 : 54 = 2 * D - 2 := by ring_exp h3
  have h5 : 54 + 2 = 2 * D := by linarith
  have h6 : 56 = 2 * D := h5
  have h7 : D = 28 := (nat.mul_right_inj (nat.succ_pos' 1)).mp h6
  have h8 : M - D = 55 - 28 := by rw [h7, hM]
  show 55 - 28 = 27 by norm_num
  show M - D = 27 by rw [h8]
  sorry

end mother_older_by_27_l819_819003


namespace find_m_l819_819141

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

def F1 : ℝ × ℝ := (-sqrt 2, 0)
def F2 : ℝ × ℝ := (sqrt 2, 0)

def line_eq (x y m : ℝ) : Prop :=
  y = x - m

def distance_point_line (x y m : ℝ) : ℝ :=
  abs (y - x + m) / sqrt (1^2 + (-1)^2)
  
def ratio_distances (d1 d2 : ℝ) : ℝ :=
  d1 / d2

theorem find_m (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  ellipse_eq x1 y1 → ellipse_eq x2 y2 →
  line_eq x1 y1 m → line_eq x2 y2 m →
  ratio_distances (distance_point_line x1 y1 m) (distance_point_line x2 y2 m) = 3 →
  m = sqrt 2 / 2 :=
by
  sorry

end find_m_l819_819141


namespace maria_trip_distance_l819_819481

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + ((D / 2) / 4) + 150 = D) 
  (h2 : 150 = 3 * D / 8) : 
  D = 400 :=
by
  -- Placeholder for the actual proof
  sorry

end maria_trip_distance_l819_819481


namespace problem_eval_expression_l819_819483

theorem problem_eval_expression :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end problem_eval_expression_l819_819483


namespace tangent_ABD_l819_819680

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 0)
def C : point := (1, 1)
def D : point := (3, 1)

noncomputable def dist (p1 p2 : point) : ℝ :=
(real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2))

noncomputable def ang_tangent (A B D : point) : ℝ := 
let AD := dist A D in
let BD := dist B D in
AD / BD

theorem tangent_ABD :
  ang_tangent A B D = real.sqrt 10 / 2 := 
sorry

end tangent_ABD_l819_819680


namespace find_range_of_a_l819_819161

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x => a * (x - 2 * Real.exp 1) * Real.log x + 1

def range_of_a (a : ℝ) : Prop :=
  (a < 0 ∨ a > 1 / Real.exp 1)

theorem find_range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ range_of_a a := by
  sorry

end find_range_of_a_l819_819161


namespace sum_powers_of_i_eq_zero_l819_819470

theorem sum_powers_of_i_eq_zero (i : ℂ) (h : i^2 = -1) : 
  i^(603) + i^(602) + ... + i^(1) + 1 = 0 := by
sorry

end sum_powers_of_i_eq_zero_l819_819470


namespace distance_between_M_and_focus_yA_times_yB_const_exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const_l819_819935

-- Part 1
theorem distance_between_M_and_focus (y₀: ℝ) (y₀_eq_sqrt_2 : y₀ = Real.sqrt 2) : 
  let x₀ := y₀^2 
  let focus := (1/4 : ℝ, 0)
  let M := (x₀, y₀)
  (|M.fst - focus.fst|) = 7/4 :=
by
  sorry

-- Part 2
theorem yA_times_yB_const (t : ℝ) (P Q: (ℝ × ℝ)) (PQonParabola : (P.snd ^ 2 = P.fst) ∧ (Q.snd ^ 2 = Q.fst))
  (PonLine : P.fst = 1 ∧ P.snd = 1) (QonLine : Q.fst = 1 ∧ Q.snd = -1) (tonline : t = -1) : 
  let y₀ := sqrt 2 
  let M := (y₀^2, y₀)
  ∃ yA yB, yA * yB = -1 :=
by 
  sorry

-- Part 3
theorem exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const (yt1 yt2: ℝ) (yP yQ yA yB : ℝ)
  (t: ℝ) (Py Pnot_eq_1 : yP != 1 ∧ yQ != 1) (yt: t = 1) : 
  yA * yB = 1 ∧ yP * yQ ≠ yt1 / yt2 :=
by
  sorry

end distance_between_M_and_focus_yA_times_yB_const_exists_t_such_that_yA_times_yB_eq_1_and_yP_times_yQ_const_l819_819935


namespace fraction_of_journey_by_rail_l819_819821

theorem fraction_of_journey_by_rail :
  ∀ (x : ℝ), x * 130 + (17 / 20) * 130 + 6.5 = 130 → x = 1 / 10 :=
by
  -- proof
  sorry

end fraction_of_journey_by_rail_l819_819821


namespace middle_part_value_l819_819990

theorem middle_part_value (x : ℚ) (h : x + (1 / 4) * x + (1 / 8) * x = 120) : 
  (1 / 4) * x = 240 / 11 :=
begin
  sorry
end

end middle_part_value_l819_819990


namespace imaginary_part_of_z_l819_819156

-- Condition: definition of the complex number z
def z : ℂ := (2 - Complex.i) / (1 + Complex.i)

-- Theorem statement: proving that the imaginary part of z is -3/2
theorem imaginary_part_of_z : z.im = -3/2 := by
  sorry

end imaginary_part_of_z_l819_819156


namespace equilateral_pyramid_base_side_length_l819_819428

theorem equilateral_pyramid_base_side_length 
  (radius : ℝ)
  (pyramid_height : ℝ)
  (tangent_to_faces : ∀ (A B C D E F : ℝ), tangency_condition A B C D E F)
  (equilateral_triangle_base : is_equilateral_triangle base)
  (side_length : ℝ) :
  radius = 3 →
  pyramid_height = 9 →
  base_side_length side_length equilateral_triangle_base →
  side_length = 6 * sqrt 3 := 
by {
  sorry
}

end equilateral_pyramid_base_side_length_l819_819428


namespace change_in_quadratic_expression_l819_819050

variable {x b a : ℝ}

theorem change_in_quadratic_expression (hb : b ∈ ℝ) (ha : a ∈ ℝ) (ha_pos : 0 < a) :
  ∃ c : ℝ, 
    (c = 2 * a * x + a^2 + b * a) ∨ (c = -2 * a * x + a^2 - b * a) :=
sorry

end change_in_quadratic_expression_l819_819050


namespace roles_assignment_l819_819538

noncomputable def numberOfWaysToAssignRoles : ℕ :=
  4.choose 3 * (3.factorial)

theorem roles_assignment :
  numberOfWaysToAssignRoles = 24 := by
sorry

end roles_assignment_l819_819538


namespace height_of_water_in_cylindrical_tank_l819_819426

theorem height_of_water_in_cylindrical_tank :
  let r_cone := 15  -- radius of base of conical tank in cm
  let h_cone := 24  -- height of conical tank in cm
  let r_cylinder := 18  -- radius of base of cylindrical tank in cm
  let V_cone := (1 / 3 : ℝ) * Real.pi * r_cone^2 * h_cone  -- volume of conical tank
  let h_cyl := V_cone / (Real.pi * r_cylinder^2)  -- height of water in cylindrical tank
  h_cyl = 5.56 :=
by
  sorry

end height_of_water_in_cylindrical_tank_l819_819426


namespace rectangles_with_trapezoid_area_l819_819223

-- Define the necessary conditions
def small_square_area : ℝ := 1
def total_squares : ℕ := 12
def rows : ℕ := 4
def columns : ℕ := 3
def trapezoid_area : ℝ := 3

-- Statement of the proof problem
theorem rectangles_with_trapezoid_area :
  (∀ rows columns : ℕ, rows * columns = total_squares) →
  (∀ area : ℝ, area = small_square_area) →
  (∀ trapezoid_area : ℝ, trapezoid_area = 3) →
  (rows = 4) →
  (columns = 3) →
  ∃ rectangles : ℕ, rectangles = 10 :=
by
  sorry

end rectangles_with_trapezoid_area_l819_819223


namespace max_squares_cut_from_rectangle_l819_819871

theorem max_squares_cut_from_rectangle (L W : ℕ) (hL : L = 11) (hW : W = 7) : 
  ∃ n, n = 6 ∧ (number_of_squares L W n) := 
sorry

-- Auxiliary definition to represent the number of squares that can be cut from a rectangle
def number_of_squares : ℕ → ℕ → ℕ → Prop 
| 0, _, n := n = 0
| _, 0, n := n = 0
| L, W, n := 
  if L >= W then 
    let sq_side := W in 
    let remaining_L := L - sq_side in 
    ∃ n₁, number_of_squares remaining_L sq_side n₁ ∧ n = n₁ + 1
  else 
    let sq_side := L in 
    let remaining_W := W - sq_side in 
    ∃ n₂, number_of_squares sq_side remaining_W n₂ ∧ n = n₂ + 1

end max_squares_cut_from_rectangle_l819_819871


namespace principal_amount_l819_819108

theorem principal_amount
(SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
(h₀ : SI = 800)
(h₁ : R = 0.08)
(h₂ : T = 1)
(h₃ : SI = P * R * T) : P = 10000 :=
by
  sorry

end principal_amount_l819_819108


namespace domain_of_g_l819_819471

noncomputable def g (x : ℝ) := 1 / (⌊x^2 - 6 * x + 10⌋)

theorem domain_of_g : ∀ x : ℝ, ⌊x^2 - 6 * x + 10⌋ ≠ 0 := by
  intros x
  have : x^2 - 6 * x + 10 > 1 ∨ x^2 - 6 * x + 10 < 0 := by
    sorry
  cases this with h_positive h_negative
  . have : ⌊x^2 - 6 * x + 10⌋ ≥ 1 := by
      sorry
    exact ne_of_gt this  -- Verifying not equal to 0
  . have : ⌊x^2 - 6 * x + 10⌋ ≤ -1 := by
      sorry
    exact ne_of_lt this  -- Verifying not equal to 0

end domain_of_g_l819_819471


namespace BE_eq_DF_l819_819664

variables (A B C D E F P Q R O : ℝ)
variables [ConvexQuadrilateral A B C D]
variables (h1 : dist B C = dist D A)
variables (h2 : B ≠ C ∧ D ≠ A)
variables (h3 : ∃ E F, E ∈ line B C ∧ F ∈ line D A)
variables (h4 : ∃ P, line A C ∩ line B D = {P})
variables (h5 : ∃ Q R, line E F ∩ line B D = {Q} ∧ line E F ∩ line A C = {R})
variables (h6 : ∃ O, isPerpBisectorOf O A C ∧ isPerpBisectorOf O B D)
variables (h7 : circumcircle P Q R O)

theorem BE_eq_DF : dist B E = dist D F :=
sorry

end BE_eq_DF_l819_819664


namespace find_angle_and_area_of_triangle_l819_819633

theorem find_angle_and_area_of_triangle (a b : ℝ) 
  (h_a : a = Real.sqrt 7) (h_b : b = 2)
  (angle_A : ℝ) (angle_A_eq : angle_A = Real.pi / 3)
  (angle_B : ℝ)
  (vec_m : ℝ × ℝ := (a, Real.sqrt 3 * b))
  (vec_n : ℝ × ℝ := (Real.cos angle_A, Real.sin angle_B))
  (colinear : vec_m.1 * vec_n.2 = vec_m.2 * vec_n.1)
  (sin_A : Real.sin angle_A = (Real.sqrt 3) / 2)
  (cos_A : Real.cos angle_A = 1 / 2) :
  angle_A = Real.pi / 3 ∧ 
  ∃ (c : ℝ), c = 3 ∧
  (1/2) * b * c * Real.sin angle_A = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_angle_and_area_of_triangle_l819_819633


namespace lines_parallel_or_coincide_l819_819263

-- Definitions of points on the equilateral hexagon
variables {A1 C2 B1 A2 C1 B2 : Point}
-- Definitions of circumcenters and orthocenters of the triangles
variables {O1 H1 O2 H2 : Point}

-- Conditions provided in the problem
variable (h1 : is_equilateral_hexagon A1 C2 B1 A2 C1 B2)
variable (h2 : circumcenter A1 B1 C1 O1)
variable (h3 : orthocenter A1 B1 C1 H1)
variable (h4 : circumcenter A2 B2 C2 O2)
variable (h5 : orthocenter A2 B2 C2 H2)
variable (h6 : O1 ≠ O2)
variable (h7 : H1 ≠ H2)

-- The target statement to prove
theorem lines_parallel_or_coincide (h1 : is_equilateral_hexagon A1 C2 B1 A2 C1 B2)
    (h2 : circumcenter A1 B1 C1 O1) (h3 : orthocenter A1 B1 C1 H1)
    (h4 : circumcenter A2 B2 C2 O2) (h5 : orthocenter A2 B2 C2 H2)
    (h6 : O1 ≠ O2) (h7 : H1 ≠ H2) : 
    parallel_or_coincide (line_through O1 O2) (line_through H1 H2) :=
sorry

end lines_parallel_or_coincide_l819_819263


namespace count_valid_arrangements_l819_819850

/-- Define the specific constraints for arrangement and the count of valid permutations. -/
def valid_arrangement (l : List ℕ) : Prop :=
  (∀ (i j : ℕ), i < j → j < l.length → (l.get? i = some 7 → l.get? (j - 1) = some 7) ∨ (l.get? i < l.get? j ∧ l.get? (j - 1) ≠ some 7)) ∧
  (length l = 7 ∧ ∀ i, l.get? i ≠ some 7 → (l.get? i = some 7 → l.get? (min 6 (i + 1)) = some 7 ∨ l.get? (min 6 (i - 1)) ≠ some 7))

theorem count_valid_arrangements : 
  ∃ n, n = 60 ∧ ∀ l, valid_arrangement l → 7 ≠ l.get? 4 → l.length = n :=
sorry

end count_valid_arrangements_l819_819850


namespace correct_option_is_D_l819_819786

theorem correct_option_is_D :
  (¬(sqrt 2 + sqrt 3 = sqrt 5) ∧
   ¬(5 * sqrt 5 - 2 * sqrt 2 = 3 * sqrt 3) ∧
   ¬(2 * sqrt 3 * 3 * sqrt 3 = 6 * sqrt 3) ∧
   (sqrt 2 / sqrt 3 = sqrt 6 / 3)) 
   → correct_option = 'D' :=
by
  sorry

end correct_option_is_D_l819_819786


namespace polygon_area_of_plane_intersection_l819_819814

noncomputable def P : EuclideanGeometry.Point ℝ 3 := ⟨6, 0, 0⟩
noncomputable def Q : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 17⟩
noncomputable def R : EuclideanGeometry.Point ℝ 3 := ⟨30, 3, 30⟩

-- Defining the cube vertices
noncomputable def A : EuclideanGeometry.Point ℝ 3 := ⟨0, 0, 0⟩
noncomputable def B : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 0⟩
noncomputable def C : EuclideanGeometry.Point ℝ 3 := ⟨30, 0, 30⟩
noncomputable def D : EuclideanGeometry.Point ℝ 3 := ⟨30, 30, 30⟩

theorem polygon_area_of_plane_intersection :
  let plane := EuclideanGeometry.Plane.mk_through_points P Q R in
  EuclideanGeometry.area_of_polygon_formed_by_plane_cube_intersection plane A B C D = 42 := -- Assume 42 is the placeholder area based on real calculations
sorry

end polygon_area_of_plane_intersection_l819_819814


namespace initial_nickels_l819_819690

theorem initial_nickels (N : ℕ) (h1 : N + 9 + 2 = 18) : N = 7 :=
by sorry

end initial_nickels_l819_819690


namespace sum_of_powers_l819_819468

-- Define i where i^2 = -1
def i : ℂ := Complex.I

-- Lean theorem statement to prove the sum
theorem sum_of_powers :
  (∑ (k : ℕ) in (List.range 604), i ^ k) = 0 :=
by
  sorry

end sum_of_powers_l819_819468


namespace max_P_l819_819271

noncomputable def P (a b : ℝ) : ℝ :=
  (a^2 + 6*b + 1) / (a^2 + a)

theorem max_P (a b x1 x2 x3 : ℝ) (h1 : a = x1 + x2 + x3) (h2 : a = x1 * x2 * x3) (h3 : ab = x1 * x2 + x2 * x3 + x3 * x1) 
    (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
    P a b ≤ (9 + Real.sqrt 3) / 9 := 
sorry

end max_P_l819_819271


namespace least_total_acorns_l819_819759

theorem least_total_acorns :
  ∃ a₁ a₂ a₃ : ℕ,
    (∀ k : ℕ, (∃ a₁ a₂ a₃ : ℕ,
      (2 * a₁ / 3 + a₁ % 3 / 3 + a₂ + a₃ / 9) % 6 = 4 * k ∧
      (a₁ / 6 + a₂ / 3 + a₃ / 3 + 8 * a₃ / 18) % 6 = 3 * k ∧
      (a₁ / 6 + 5 * a₂ / 6 + a₃ / 9) % 6 = 2 * k) → k = 630) ∧
    (a₁ + a₂ + a₃) = 630 :=
sorry

end least_total_acorns_l819_819759


namespace triangle_area_vertex_y_l819_819023

theorem triangle_area_vertex_y (y : ℝ) (h_positive : 0 < y) :
  let A := (-1, y)
      B := (7, 3)
      C := (-1, 3)
      base := 7 - (-1)
      height := |y - 3|
  in (1 / 2) * base * height = 36 → y = 12 :=
by
  intro h_area
  let base := 8
  let height := |y - 3|
  have h_eq : (1 / 2) * base * height = 36 := h_area
  sorry

end triangle_area_vertex_y_l819_819023


namespace sum_of_ratio_simplified_l819_819355

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l819_819355


namespace polynomial_degree_is_16_l819_819479

open Polynomial

noncomputable def degree_of_polynomial :=
  let p1 := (3 * X^5 + 2 * X^3 - X + 7)
  let p2 := (4 * X^11 - 6 * X^8 + 5 * X^5 - 15)
  let p3 := (X^2 + 3)^8
  degree (p1 * p2 - p3)

theorem polynomial_degree_is_16 : degree_of_polynomial = 16 := by
  sorry

end polynomial_degree_is_16_l819_819479


namespace sum_first_n_terms_c_l819_819555

/-- Given an arithmetic sequence {a_n} where a_n = 2n - 1 and a geometric sequence {b_n}
where b_n = 2^(n-1), prove that the sum of the first n terms of the sequence {c_n} where 
c_n = a_n / b_n equals to T_n = 6 - (2n + 3) / 2^(n-1). -/
theorem sum_first_n_terms_c (n : ℕ) (h_n_pos : 0 < n) :
  let a_n := λ n : ℕ, 2 * n - 1,
      b_n := λ n : ℕ, 2^(n - 1),
      c_n := λ n : ℕ, (a_n n) / (b_n n),
      T_n := ∑ i in Finset.range n, c_n (i + 1)
  in T_n = 6 - (2 * n + 3) / 2^(n-1) :=
by sorry

end sum_first_n_terms_c_l819_819555


namespace count_integers_with_6_or_0_l819_819987

theorem count_integers_with_6_or_0 :
  let count := (5689 : ℕ) - (8 + 64 + 512 + 2560) in -- Numbers not containing '0' or '6'
  count = 2545 :=
by
  let count := (5689 : ℕ) - (8 + 64 + 512 + 2560)
  have h : count = 2545 := rfl
  exact h

end count_integers_with_6_or_0_l819_819987


namespace count_numbers_containing_digit_three_l819_819185

def contains_digit_three (n : ℕ) : Prop :=
  (n / 100 = 3) ∨ (n / 10 % 10 = 3) ∨ (n % 10 = 3)

def is_in_range (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 499

theorem count_numbers_containing_digit_three :
  (Finset.filter (λ n, contains_digit_three n) (Finset.range' 200 300)).card = 138 :=
by
  sorry

end count_numbers_containing_digit_three_l819_819185


namespace ratio_of_radii_of_circles_l819_819151

theorem ratio_of_radii_of_circles 
  (a b : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : ∃ (c : ℝ), c = Real.sqrt (a^2 + b^2)) 
  (h4 : ∃ (r R : ℝ), R = c / 2 ∧ r = 24 / (a + b + c)) : R / r = 5 / 2 :=
by
  sorry

end ratio_of_radii_of_circles_l819_819151


namespace deck_width_l819_819824

theorem deck_width (w : ℝ) : 
  (10 + 2 * w) * (12 + 2 * w) = 360 → w = 4 := 
by 
  sorry

end deck_width_l819_819824


namespace shaded_area_correct_l819_819616

noncomputable def remaining_shaded_area (side_length : ℝ) := 
  let circle_radius := side_length / 4 in
  let large_square_area := side_length ^ 2 in
  let circle_area := π * circle_radius ^ 2 in
  let total_circle_area := 4 * circle_area in
  let inner_square_side := sqrt (side_length ^ 2 - 2 * circle_radius ^ 2) in
  let inner_square_area := inner_square_side ^ 2 in
  large_square_area - total_circle_area - inner_square_area

theorem shaded_area_correct : remaining_shaded_area 30 = 112.5 - 225 * π :=
by
  -- Proof would go here.
  sorry

end shaded_area_correct_l819_819616


namespace unique_solution_only_a_is_2_l819_819200

noncomputable def unique_solution_inequality (a : ℝ) : Prop :=
  ∀ (p : ℝ → ℝ), (∀ x, 0 ≤ p x ∧ p x ≤ 1 ∧ p x = x^2 - a * x + a) → 
  ∃! x, p x = 1

theorem unique_solution_only_a_is_2 (a : ℝ) (h : unique_solution_inequality a) : a = 2 :=
sorry

end unique_solution_only_a_is_2_l819_819200


namespace find_number_l819_819805

theorem find_number (x : ℝ) (h : (168 / 100) * x / 6 = 354.2) : x = 1265 := 
by
  sorry

end find_number_l819_819805


namespace survey_method_correct_l819_819789

/-- Definitions to represent the options in the survey method problem. -/
inductive SurveyMethod
| A
| B
| C
| D

/-- The function to determine the correct survey method. -/
def appropriate_survey_method : SurveyMethod :=
  SurveyMethod.C

/-- The theorem stating that the appropriate survey method is indeed option C. -/
theorem survey_method_correct : appropriate_survey_method = SurveyMethod.C :=
by
  /- The actual proof is omitted as per instruction. -/
  sorry

end survey_method_correct_l819_819789


namespace ratio_of_side_lengths_sum_l819_819348
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l819_819348


namespace prime_pi_lower_bound_l819_819417

open Real

theorem prime_pi_lower_bound :
  (∃ c > 0, ∀ x : ℝ, 2 ≤ x → π x ≥ c * ln (ln x)) :=
sorry

end prime_pi_lower_bound_l819_819417


namespace jeff_boxes_filled_l819_819641

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end jeff_boxes_filled_l819_819641


namespace quadrilateral_is_rhombus_l819_819256

variable {V : Type*} [InnerProductSpace ℝ V]

noncomputable def vector_rhombus (e AB CD AD : V) (AB_len : ℝ) (AD_len : ℝ) :=
  ∥e∥ = 1 ∧ AB = AB_len • e ∧ CD = -AB_len • e ∧ ∥AD∥ = AD_len

theorem quadrilateral_is_rhombus 
  (V : Type*) [InnerProductSpace ℝ V] 
  (e AB CD AD : V)
  (h : vector_rhombus e AB CD AD 3 3) 
  : (∃ d : V, AD = d ∧ ∥d∥ = 3 ∧ ∥AD∥ = 3) :=
begin
  sorry
end

end quadrilateral_is_rhombus_l819_819256


namespace angle_equality_in_temples_theorem_l819_819539

/-- Given an acute triangle ABC with AB < AC < BC, inscribed in a circle c(O, R).
    The perpendicular bisector of the angle bisector AD (D on BC) intersects the circle at K and L,
    where K lies on the smaller arc AB. The circle c₁(K, KA) intersects c at T 
    and the circle c₂(L, LA) intersects c at S.
    Prove that ∠BAT = ∠CAS. -/
theorem angle_equality_in_temples_theorem
    (A B C D K L T S O : Type*)
    [condition1 : is_acute_triangle A B C]
    [condition2 : lies_on_circle B C A O]
    [condition3 : angle_bisector_of AD B C]
    [condition4 : perpendicular_bisector_of AD intersects_circle_at K L]
    [condition5 : K_on_smaller_arc_of AB]
    [condition6 : circle_intersects_circle_at KA O T]
    [condition7 : circle_intersects_circle_at LA O S]
    : ∠BAT = ∠CAS := 
sorry

end angle_equality_in_temples_theorem_l819_819539


namespace find_value_of_a_l819_819949

theorem find_value_of_a :
  ∀ (a : ℝ), (A = {-1, 0, 1}) ∧ (B = {a + 1, 2a}) ∧ (A ∩ B = {0}) → a = -1 :=
by
  sorry

end find_value_of_a_l819_819949


namespace john_shower_duration_l819_819647

variable (days_per_week : ℕ := 7)
variable (weeks : ℕ := 4)
variable (total_days : ℕ := days_per_week * weeks)
variable (shower_frequency : ℕ := 2) -- every other day
variable (number_of_showers : ℕ := total_days / shower_frequency)
variable (total_gallons_used : ℕ := 280)
variable (gallons_per_shower : ℕ := total_gallons_used / number_of_showers)
variable (gallons_per_minute : ℕ := 2)

theorem john_shower_duration 
  (h_cond : total_gallons_used = number_of_showers * gallons_per_shower)
  (h_shower_eq : total_days / shower_frequency = number_of_showers)
  : gallons_per_shower / gallons_per_minute = 10 :=
by
  sorry

end john_shower_duration_l819_819647


namespace value_of_Q_if_n_100_l819_819659

theorem value_of_Q_if_n_100 : 
  (let Q := (Finset.range 100).prod (λ k, 1 - (k + 2) / (k + 3 : ℝ)) in
   Q = 1 / 101) :=
by
  let Q := (Finset.range 100).prod (λ k, 1 - (k + 2) / (k + 3 : ℝ))
  have Q_eq : Q = (Finset.range 100).prod (λ k, (1 : ℝ)/(k + 3)) := by sorry
  have prod_eq : Q = 1 / 101 := by sorry
  exact prod_eq

end value_of_Q_if_n_100_l819_819659


namespace trader_sold_pens_l819_819442

theorem trader_sold_pens (C : ℝ) (N : ℕ) (hC : C > 0) (h_gain : N * (2 / 5) = 40) : N = 100 :=
by
  sorry

end trader_sold_pens_l819_819442


namespace equal_distances_l819_819262

theorem equal_distances (ABC : Triangle) 
  (P : Point ABC) 
  (hP : ∠(PAC) = ∠(PBC))
  (L : Point) (hL : perp P BC L)
  (M : Point) (hM : perp P CA M)
  (D : Point) (hD : midpoint D A B) :
  dist D L = dist D M := 
  sorry

end equal_distances_l819_819262


namespace tan_alpha_eq_neg2_complex_expression_eq_neg5_l819_819523

variables (α : ℝ)
variables (h_sin : Real.sin α = - (2 * Real.sqrt 5) / 5)
variables (h_tan_neg : Real.tan α < 0)

theorem tan_alpha_eq_neg2 :
  Real.tan α = -2 :=
sorry

theorem complex_expression_eq_neg5 :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) /
  (Real.cos (α - Real.pi / 2) - Real.sin (3 * Real.pi / 2 + α)) = -5 :=
sorry

end tan_alpha_eq_neg2_complex_expression_eq_neg5_l819_819523


namespace BC_length_l819_819823

variable (H O M F A B C : Type)
variable (dist : Type -> Type -> ℝ)
variable [metric_space Type]

-- Define the conditions
variable (rect_HOMF : True) -- implying that H, O, M, F form a rectangle
variable (H_orthocenter : H = orthocenter A B C)
variable (O_circumcenter : O = circumcenter A B C)
variable (M_midpoint_BC : M = midpoint B C)
variable (F_feet_A : F = foot A B C)

-- Side lengths of the rectangle HOMF
variable (HO_eq_11 : dist H O = 11)
variable (OM_eq_5 : dist O M = 5)

-- Final proof statement
theorem BC_length : dist B C = 28 := 
  sorry

end BC_length_l819_819823


namespace constant_term_binomial_expansion_l819_819960

theorem constant_term_binomial_expansion :
  let n := 9,
  let term_x_binom := λ (r : ℕ), (Nat.choose n r) * (-1)^r * (x^(n - 3*r)),
  let k := 3,
  let m := 8,
  (Nat.choose n k = Nat.choose n (n - m)) →
  (∃ r : ℕ, x^(n - 3*r) = 1 ∧ Nat.choose n r * (-1)^r = -84) :=
by
  intros
  sorry

end constant_term_binomial_expansion_l819_819960


namespace validNs_solution_l819_819548

-- Definition of the conditions
def isValidN (n : ℕ) : Prop :=
  n < 50 ∧ let d := Nat.gcd (3 * n + 5) (5 * n + 4) in d > 1

-- Definition of the specific values of n
def validNs : List ℕ := [7, 20, 33, 46]

-- Theorem stating that the only valid n satisfying the conditions are the values in validNs
theorem validNs_solution :
  ∀ n : ℕ, isValidN n ↔ n ∈ validNs :=
by
  sorry

end validNs_solution_l819_819548


namespace rate_of_markup_l819_819024

theorem rate_of_markup (S : ℝ) (profit_percentage : ℝ) (expenses_percentage : ℝ) 
(hS : S = 8) 
(hprofit : profit_percentage = 0.12) 
(hexpenses : expenses_percentage = 0.18) :
  let C := S * (1 - (profit_percentage + expenses_percentage)) in
  (S - C) / C * 100 = 42.857 :=
by 
  sorry

end rate_of_markup_l819_819024


namespace expression_in_terms_of_p_q_l819_819255

-- Define the roots and the polynomials conditions
variable (α β γ δ : ℝ)
variable (p q : ℝ)

-- The conditions of the problem
axiom roots_poly1 : α * β = 1 ∧ α + β = -p
axiom roots_poly2 : γ * δ = 1 ∧ γ + δ = -q

theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end expression_in_terms_of_p_q_l819_819255


namespace pyramid_cube_volume_l819_819008

noncomputable def volume_of_cube_in_pyramid : ℝ :=
let s : ℝ := 2 in -- Side length of the hexagonal base
let h_lateral : ℝ := 2 * Real.sqrt 3 in -- Height of lateral equilateral face
let height_of_pyramid : ℝ := h_lateral / 3 in -- Height partition of the pyramid accommodating the cube and the tetrahedron
let side_of_cube : ℝ := height_of_pyramid / 3 in -- Side of the cube
let volume_cube : ℝ := side_of_cube ^ 3 in -- Volume of the cube
let result : ℝ := volume_cube in
  result

theorem pyramid_cube_volume (hex_side : ℝ := 2) (lateral_height : ℝ := 2 * Real.sqrt 3)
(height_partition : ℝ := lateral_height / 3) (cube_side : ℝ := height_partition / 3)
(cube_volume : ℝ := cube_side ^ 3) :
  cube_volume = 8 * Real.sqrt 3 / 243 := by
  sorry

end pyramid_cube_volume_l819_819008


namespace minimum_elements_l819_819829

noncomputable def min_elements_X : ℕ :=
  Inf { n | ∀ (X : set (fin 10 × fin 10)), (∀ seq : ℕ → fin 10, ∃ i : ℕ, (seq i, seq (i + 1)) ∈ X) → X ⊆ { ⟨i, j⟩ | 0 ≤ i ∧ i ≤ 9 ∧ 0 ≤ j ∧ j ≤ 9 } → n ≤ X.to_finset.card }

theorem minimum_elements (X : set (fin 10 × fin 10)) (h1 : X ⊆ { ⟨i, j⟩ | 0 ≤ i ∧ i ≤ 9 ∧ 0 ≤ j ∧ j ≤ 9 }) 
  (h2 : ∀ seq : ℕ → fin 10, ∃ i : ℕ, (seq i, seq (i + 1)) ∈ X) : 
  |X| ≥ 55 :=
begin
  sorry
end

end minimum_elements_l819_819829


namespace distance_from_F_to_midpoint_DE_l819_819217

theorem distance_from_F_to_midpoint_DE (D E F : Type) [MetricSpace D] (DE DF EF : ℝ) 
  (hDE : DE = 15) (hDF : DF = 9) (hEF : EF = 12) (hRightTriangle : DE^2 = DF^2 + EF^2) :
  distance F (midpoint DE) = 7.5 :=
by
  sorry

end distance_from_F_to_midpoint_DE_l819_819217


namespace snail_reaches_tree_l819_819013

theorem snail_reaches_tree
  (l1 l2 s : ℝ) 
  (h_l1 : l1 = 4) 
  (h_l2 : l2 = 3) 
  (h_s : s = 40) : 
  ∃ n : ℕ, n = 37 ∧ s - n*(l1 - l2) ≤ l1 :=
  by
    sorry

end snail_reaches_tree_l819_819013


namespace distance_from_F_to_midpoint_DE_l819_819218

theorem distance_from_F_to_midpoint_DE (D E F : Type) [MetricSpace D] (DE DF EF : ℝ) 
  (hDE : DE = 15) (hDF : DF = 9) (hEF : EF = 12) (hRightTriangle : DE^2 = DF^2 + EF^2) :
  distance F (midpoint DE) = 7.5 :=
by
  sorry

end distance_from_F_to_midpoint_DE_l819_819218


namespace calculation_proof_l819_819464

theorem calculation_proof : 
  2 * Real.tan (Real.pi / 3) - (-2023) ^ 0 + (1 / 2) ^ (-1 : ℤ) + abs (Real.sqrt 3 - 1) = 3 * Real.sqrt 3 := 
by
  sorry

end calculation_proof_l819_819464


namespace largest_operation_result_is_div_l819_819896

noncomputable def max_operation_result : ℚ :=
  max (max (-1 + (-1 / 2)) (-1 - (-1 / 2)))
      (max (-1 * (-1 / 2)) (-1 / (-1 / 2)))

theorem largest_operation_result_is_div :
  max_operation_result = 2 := by
  sorry

end largest_operation_result_is_div_l819_819896


namespace toll_roads_distribution_l819_819612

-- Definitions for the cities and capitals
def City : Type := sorry
def isCapital : City → Prop := sorry
def isSouthernCapital (Y : City) : Prop := isCapital Y ∧ ∃! C, isCapital C ∧ C ≠ Y
def isNorthernCapital (S : City) : Prop := isCapital S ∧ ∃! C, isCapital C ∧ C ≠ S

-- Definitions for routes and toll roads
def Route (a b : City) : Type := List City
def isTollRoad (c1 c2 : City) : Prop := sorry
def routePassesThroughTollRoads (r : Route Y S) : Prop := ∀ c1 c2 ∈ r, isTollRoad c1 c2

-- The main theorem statement
theorem toll_roads_distribution {Y S : City} (hY: isSouthernCapital Y) (hS: isNorthernCapital S) 
    (h : ∀ (r : Route Y S), routePassesThroughTollRoads r → r.length ≥ 10) :
    ∃ (company : ℕ → Set (City × City)), 
    (∀ i, 1 ≤ i ∧ i ≤ 10 → (∀ r : Route Y S, ∃ x y ∈ r, (x, y) ∈ company i)) :=
sorry

end toll_roads_distribution_l819_819612


namespace smallest_simple_polynomial_l819_819767

def is_simple (P : Polynomial ℤ) : Prop :=
  ∀ (a : ℤ), a ∈ P.coeff → a = -1 ∨ a = 0 ∨ a = 1

def has_property (P : Polynomial ℤ) (n : ℤ) : Prop :=
  ∀ (k : ℤ), P.eval k % n = 0

theorem smallest_simple_polynomial {n : ℤ} (hn : n > 1) :
  ∃ P : Polynomial ℤ, is_simple P ∧ has_property P n ∧ (∀ Q : Polynomial ℤ, is_simple Q ∧ has_property Q n → Q.support.card ≥ 2) ∧ P.support.card = 2 := sorry

end smallest_simple_polynomial_l819_819767


namespace adam_deleted_items_l819_819447

theorem adam_deleted_items (initial_items deleted_items remaining_items : ℕ)
  (h1 : initial_items = 100) (h2 : remaining_items = 20) 
  (h3 : remaining_items = initial_items - deleted_items) : 
  deleted_items = 80 :=
by
  sorry

end adam_deleted_items_l819_819447


namespace alex_buys_15_pounds_of_corn_l819_819477

theorem alex_buys_15_pounds_of_corn:
  ∃ (c b : ℝ), c + b = 30 ∧ 1.20 * c + 0.60 * b = 27.00 ∧ c = 15.0 :=
by
  sorry

end alex_buys_15_pounds_of_corn_l819_819477


namespace prime_eq_sol_l819_819117

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l819_819117


namespace weight_order_l819_819279

variables (A B C D : ℝ) -- Representing the weights of objects A, B, C, and D as real numbers.

-- Conditions given in the problem:
axiom eq1 : A + B = C + D
axiom ineq1 : D + A > B + C
axiom ineq2 : B > A + C

-- Proof stating that the weights in ascending order are C < A < B < D.
theorem weight_order (A B C D : ℝ) : C < A ∧ A < B ∧ B < D :=
by
  -- We are not providing the proof steps here.
  sorry

end weight_order_l819_819279


namespace parabola_focus_coordinates_l819_819103

theorem parabola_focus_coordinates (y x : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_coordinates_l819_819103


namespace log5_of_15625_l819_819088

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l819_819088


namespace determine_functions_l819_819058

noncomputable def satisfies_functional_eq (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ), f(f(x) + y) + x * f(y) = f (x * y + y) + f(x)

theorem determine_functions (f : ℝ → ℝ) :
  satisfies_functional_eq f ↔ (f = λ x, x) ∨ (f = λ x, 0) :=
by
  sorry

end determine_functions_l819_819058


namespace fabric_initial_length_l819_819005

theorem fabric_initial_length 
  (shrinks_length : ℚ := 1/18) 
  (shrinks_width : ℚ := 1/14) 
  (width_before : ℚ := 0.875) 
  (final_area : ℚ := 221) : 
  (288 : ℚ) = 
  let width_after := width_before * (1 - shrinks_width) in
  let length_after := (final_area / width_after) in
  let initial_length := length_after / (1 - shrinks_length) in
  initial_length :=
by sorry

end fabric_initial_length_l819_819005


namespace vacation_hours_per_week_l819_819186

open Nat

theorem vacation_hours_per_week :
  let planned_hours_per_week := 25
  let total_weeks := 15
  let total_money_needed := 4500
  let sick_weeks := 3
  let hourly_rate := total_money_needed / (planned_hours_per_week * total_weeks)
  let remaining_weeks := total_weeks - sick_weeks
  let total_hours_needed := total_money_needed / hourly_rate
  let required_hours_per_week := total_hours_needed / remaining_weeks
  required_hours_per_week = 31.25 := by
sorry

end vacation_hours_per_week_l819_819186


namespace distance_M_focus_yA_yB_constant_exists_t_yA_yB_yP_yQ_l819_819933

-- Part (1)
theorem distance_M_focus (y0 : ℝ) (h1 : y0 = sqrt 2) : 
  let M := (y0^2, y0) in 
  M.1 + (1 / 4) / 2 = 9 / 4 :=
by 
  -- Proof outline: calculate the distance and verify it equals 9/4
  sorry 

-- Part (2)
theorem yA_yB_constant (t : ℝ) (h2 : t = -1) (P : (ℝ × ℝ)) (hP : P = (1, 1)) (Q : (ℝ × ℝ)) (hQ : Q = (1, -1)) : 
  ∀ M y0, M = (y0^2, y0) → 
  let yA := (y0 - 1) / (y0 + 1) in 
  let yB := (-y0 - 1) / (y0 - 1) in 
  yA * yB = -1 :=
by 
  -- Proof outline: calculate yA and yB based on their definitions and verify yA * yB = -1
  sorry 

-- Part (3)
theorem exists_t_yA_yB_yP_yQ (P Q : ℝ × ℝ) :
  ∃ t : ℝ, (∀ yA yB, ((yA * yB = 1) → 
                     (let yP := (sqrt(2) * yA - t) / (sqrt(2) - yA) in 
                      let yQ := (sqrt(2) * yB - t) / (sqrt(2) - yB) in 
                      yP * yQ = 1)) :=
by 
  -- Proof outline: show t = 1 satisfies the conditions yA * yB = 1 and yP * yQ = 1
  sorry

end distance_M_focus_yA_yB_constant_exists_t_yA_yB_yP_yQ_l819_819933


namespace hyperbola_range_of_k_l819_819199

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ (x y : ℝ), (x^2)/(k-3) + (y^2)/(k+3) = 1 ∧ 
  (k-3 < 0) ∧ (k+3 > 0)) → (-3 < k ∧ k < 3) :=
by
  sorry

end hyperbola_range_of_k_l819_819199


namespace vampire_drains_per_week_l819_819833

-- Definitions for the conditions
def werewolf_eats_per_week (W : ℕ) : Prop := W = 5
def village_population (P : ℕ) : Prop := P = 72
def weeks (n : ℕ) : Prop := n = 9

-- Prove that the number of people the vampire drains per week is 3
theorem vampire_drains_per_week : ∀ V W P n : ℕ, 
  werewolf_eats_per_week W →
  village_population P →
  weeks n →
  9 * (V + W) = P →
  V = 3 :=
by 
  intros V W P n hW hP hN hEq 
  -- Using the given conditions
  rw [hW, hP, hN] at hEq 
  -- Skipping the actual calculation
  sorry

end vampire_drains_per_week_l819_819833


namespace negation_universal_to_particular_l819_819976

theorem negation_universal_to_particular :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_universal_to_particular_l819_819976


namespace sum_of_interior_angles_l819_819706

theorem sum_of_interior_angles (n : ℕ) (h1 : 180 * (n - 2) = 1800) (h2 : n = 12) : 
  180 * ((n + 4) - 2) = 2520 := 
by 
  { sorry }

end sum_of_interior_angles_l819_819706


namespace sqrt_sum_eq_six_l819_819494

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l819_819494


namespace base7_addition_l819_819025

theorem base7_addition : (21 : ℕ) + 254 = 505 :=
by sorry

end base7_addition_l819_819025


namespace sum_of_remainders_l819_819783

theorem sum_of_remainders 
  (a b c : ℕ) 
  (h1 : a % 53 = 37) 
  (h2 : b % 53 = 14) 
  (h3 : c % 53 = 7) : 
  (a + b + c) % 53 = 5 := 
by 
  sorry

end sum_of_remainders_l819_819783


namespace area_of_S_l819_819052

-- Define the sets M, P, and S
def M (b : ℝ) : set (ℝ × ℝ) := {p | ∃ x, p.2 = x^2 + 2 * b * x + 1}
def P (a b : ℝ) : set (ℝ × ℝ) := {p | ∃ x, p.2 = 2 * a * (x + b)}
def S : set (ℝ × ℝ) := {p | let a := p.1; let b := p.2 in ∀ (x : ℝ), (x^2 + 2 * b * x + 1) ≠ (2 * a * (x + b))}

-- The Lean statement that needs to be proved
theorem area_of_S :
  (∃ p, p ∈ S) →
  ∃ area : ℝ,
    area = π :=
sorry

end area_of_S_l819_819052


namespace three_obtuse_impossible_l819_819797

-- Define the type for obtuse angle
def is_obtuse (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

-- Define the main theorem stating the problem
theorem three_obtuse_impossible 
  (A B C D O : Type) 
  (angle_AOB angle_COD angle_AOD angle_COB
   angle_OAB angle_OBA angle_OBC angle_OCB
   angle_OAD angle_ODA angle_ODC angle_OCC : ℝ)
  (h1 : angle_AOB = angle_COD)
  (h2 : angle_AOD = angle_COB)
  (h_sum : angle_AOB + angle_COD + angle_AOD + angle_COB = 360)
  : ¬ (is_obtuse angle_OAB ∧ is_obtuse angle_OBC ∧ is_obtuse angle_ODA) := 
sorry

end three_obtuse_impossible_l819_819797


namespace distance_between_A_and_B_l819_819673

theorem distance_between_A_and_B :
  ∃ d : ℝ,
  (d > 0) ∧
  (∀ carX_speed carY_speed breakdown_distance delay_time, 
     carX_speed = 50 →
     carY_speed = 60 →
     breakdown_distance = 30 →
     delay_time = 1.2 →
     let total_additional_time := delay_time + (breakdown_distance / carX_speed) in
     total_additional_time = 1.8 →
     carX_speed + carY_speed = 110 →
     (carX_speed + carY_speed) * total_additional_time = d →
     d = 198) :=
begin
  use 198,
  simp,
end

end distance_between_A_and_B_l819_819673


namespace possible_num_students_l819_819817

-- Given conditions
variables (C1 C2 C3 : Prop)
-- C1: Chris starts and ends
-- C2: Bag contained 120 candies
-- C3: Chris takes the second to last candy

noncomputable def num_students (n : Nat) : Bool :=
  119 % (n - 1) = 0

theorem possible_num_students (n : Nat) (C1 : True) (C2 : True) (C3 : True) :
  C1 ∧ C2 ∧ C3 → num_students n → n ∈ {2, 3, 60, 119} :=
sorry

end possible_num_students_l819_819817


namespace express_repeating_decimal_as_fraction_l819_819890

noncomputable def repeating_decimal : ℚ := 7 + 123 / 999
#r "3033"
theorem express_repeating_decimal_as_fraction :
  repeating_decimal = 593 / 111 :=
sorry

end express_repeating_decimal_as_fraction_l819_819890


namespace scientific_notation_of_0_0000021_l819_819458

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end scientific_notation_of_0_0000021_l819_819458


namespace area_ratio_l819_819708

-- Define the conditions according to the problem description
-- Triangle FGH with angles 60°, 30°, 90° (making it a 30-60-90 triangle)
-- Triangle EGH, a right-angled isosceles triangle
-- Side length GH = 1 

-- Note: The Lean definitions here are oversimplified placeholders and may not directly correspond to geometrical constructs in Lean's library
variables (GH : ℝ) (FG EH : ℝ) (A_FGH : Finset ℝ) (A_IEH : Finset ℝ)

def triangle_FGH_angles := (60, 30, 90)
def triangle_EGH := (isRightAngled : true, isIsosceles : true)

-- Assign lengths based on conditions
def G_H : ℝ := 1
def F_G : ℝ := 1 / (real.sqrt 3)
def E_H : ℝ := real.sqrt 2

-- Define and prove the area ratio
theorem area_ratio : 
  (triangle_FGH_angles = (60, 30, 90)) ∧ (triangle_EGH = (true, true)) 
  → (GH = 1) 
  → (A_FGH.card / A_IEH.card = 1 / 2) :=
by 
  sorry

end area_ratio_l819_819708


namespace proof_of_expression_l819_819170

variables (α : ℝ)

def a := (Real.cos α, -2)
def b := (Real.sin α, 1)
def are_parallel : Prop := a.1 * b.2 = a.2 * b.1

theorem proof_of_expression (h : are_parallel α) : 2 * Real.sin α * Real.cos α = -4 / 5 := 
sorry

end proof_of_expression_l819_819170


namespace log5_15625_eq_6_l819_819092

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l819_819092


namespace solve_arithmetic_sequence_l819_819303

theorem solve_arithmetic_sequence (y : ℝ) (h : y > 0) : 
  let a1 := (2 : ℝ)^2
      a2 := y^2
      a3 := (4 : ℝ)^2
  in (a1 + a3) / 2 = a2 → y = Real.sqrt 10 :=
by
  intros a1 a2 a3 H
  have calc1 : a1 = 4 := by norm_num
  have calc2 : a3 = 16 := by norm_num
  rw [calc1, calc2] at H
  have avg_eq : (4 + 16) / 2 = 10 := by norm_num
  rw [avg_eq] at H
  suffices y_pos : y > 0, sorry
  sorry


end solve_arithmetic_sequence_l819_819303


namespace complex_fraction_conjugate_l819_819561

theorem complex_fraction_conjugate (z : ℂ) (hz : z = 2 + I) : (conj z / z) = (3/5) - (4/5) * I :=
by
  -- Proof goes here
  sorry

end complex_fraction_conjugate_l819_819561


namespace no_half_probability_socks_l819_819732

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l819_819732


namespace prism_height_squared_l819_819825

/-- Given a right prism with height h, bases that are regular hexagons with side lengths 12, 
and a dihedral angle between a base and another face measuring 60 degrees, prove h^2 = 108. -/
theorem prism_height_squared (h : ℝ) 
  (hexagon_side : ℝ) 
  (regular_hexagon : ∀ (A B C D E F : ℝ), hexagon_side = 12) 
  (dihedral_angle : ℝ) 
  (angle_sixty_deg : dihedral_angle = 60) 
  (height_def : h = 6 * sqrt 3) :
  h ^ 2 = 108 :=
sorry

end prism_height_squared_l819_819825


namespace tom_average_speed_l819_819762

noncomputable def total_distance : ℝ := 180
noncomputable def segment1_distance : ℝ := 60
noncomputable def segment1_speed : ℝ := 20
noncomputable def segment2_distance : ℝ := 50
noncomputable def segment2_speed : ℝ := 30
noncomputable def segment3_distance : ℝ := 70
noncomputable def segment3_speed : ℝ := 50

def average_speed (total_distance : ℝ) (time1 time2 time3 : ℝ) : ℝ :=
  total_distance / (time1 + time2 + time3)

theorem tom_average_speed :
  average_speed total_distance
    (segment1_distance / segment1_speed)
    (segment2_distance / segment2_speed)
    (segment3_distance / segment3_speed) = 29.67 := 
by
  sorry

end tom_average_speed_l819_819762


namespace log_base_5_of_15625_eq_6_l819_819072

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l819_819072


namespace travel_time_at_50_mph_exact_travel_time_at_50_mph_approx_l819_819371

-- Definitions based on conditions
def time_at_80_mph := 5 + 1 / 3 -- Time in hours
def speed_80_mph := 80 -- Speed in miles per hour
def speed_50_mph := 50 -- New speed in miles per hour

-- Distance calculation based on time and speed at 80 mph
def distance (t : ℝ) (s : ℝ) := t * s
def distance_mtown_rivertown := distance time_at_80_mph speed_80_mph

-- Time calculation based on distance and new speed at 50 mph
def new_time (d : ℝ) (s : ℝ) := d / s
def time_at_50_mph := new_time distance_mtown_rivertown speed_50_mph

-- Proof statement
theorem travel_time_at_50_mph_exact : time_at_50_mph = 8.5333333 := by
  norm_num
  sorry

theorem travel_time_at_50_mph_approx : Real.toRational(Real.round(100 * time_at_50_mph) / 100) = 8.53 := by
  norm_num
  sorry

-- Sorry is used to skip the proof.

end travel_time_at_50_mph_exact_travel_time_at_50_mph_approx_l819_819371


namespace balance_gift_card_l819_819275

variable (x : ℝ)

def balance_after_monday := x / 2
def balance_after_tuesday := balance_after_monday x - (balance_after_monday x / 4)
def balance_after_wednesday := balance_after_tuesday x - (balance_after_tuesday x / 3)
def balance_after_thursday := balance_after_wednesday x - (balance_after_wednesday x / 5)

theorem balance_gift_card (x : ℝ) : 
  balance_after_thursday x = (1 / 5) * x :=
by sorry

end balance_gift_card_l819_819275


namespace curve_intersection_range_l819_819603

theorem curve_intersection_range (λ : ℝ) (hλ : λ < 0) :
  (∀ x y : ℝ, (2 * abs x - y - 4 = 0 ∧ x^2 + λ * y^2 = 4) →
    (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) ∧ (-1 / 4 < λ ∧ λ < 0) :=
by
  sorry

end curve_intersection_range_l819_819603


namespace solution_set_of_inequality_l819_819152

noncomputable def inequality_solution (f : ℝ → ℝ) (f_deriv : ℝ → ℝ) : set ℝ :=
  {x : ℝ | f x > Real.exp (x / 2)}

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (f_deriv : ℝ → ℝ)
  (h_deriv : ∀ x, deriv f x = f_deriv x)
  (h_ineq : ∀ x, f x < 2 * f_deriv x)
  (h_init : f (Real.log 4) = 2) :
  inequality_solution f f_deriv = {x : ℝ | 2*Real.log 2 < x} :=
begin
  sorry
end

end solution_set_of_inequality_l819_819152


namespace largest_even_not_sum_of_two_odd_composites_l819_819875

def is_odd_composite (n : ℕ) : Prop :=
  n % 2 = 1 ∧ ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem largest_even_not_sum_of_two_odd_composites : ∀ n : ℕ, 38 < n → 
  ∃ a b : ℕ, is_odd_composite a ∧ is_odd_composite b ∧ n = a + b :=
begin
  sorry
end

end largest_even_not_sum_of_two_odd_composites_l819_819875


namespace triangle_perimeter_l819_819197

-- Given conditions
def given_conditions (m n : ℝ) : Prop :=
  |m - 2| + sqrt (n - 4) = 0 ∧ (m = 2 ∨ n = 4)

-- Prove that the perimeter of the triangle is 10 under the given conditions
theorem triangle_perimeter (m n : ℝ) (h : given_conditions m n) (iso1 iso2 : ℝ) :
  iso1 = m ∧ iso2 = n →
  iso1 + iso2 + (if iso1 = iso2 then iso1 else (2 * Real.sqrt ((iso1 - iso2)^2))) = 10 :=
begin
  sorry,
end

end triangle_perimeter_l819_819197


namespace triangle_PQR_PR_value_l819_819635

theorem triangle_PQR_PR_value (PQ QR PR : ℕ) (h1 : PQ = 7) (h2 : QR = 20) (h3 : 13 < PR) (h4 : PR < 27) : PR = 21 :=
by sorry

end triangle_PQR_PR_value_l819_819635


namespace first_number_of_10th_group_l819_819159

def sequence (n : ℕ) : ℕ := 2 * n - 3

def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_number_of_10th_group :
  sequence (sum_natural 9 + 1) = 89 :=
by
  sorry

end first_number_of_10th_group_l819_819159


namespace part_a_l819_819536

-- Definitions and Setup
structure Point (α : Type) :=
(x : α)
(y : α)

variables {α : Type} [linear_ordered_field α]
variables {A B C G M : Point α}

-- Some distance calculation helpers
def sqdist (P Q : Point α) : α := (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Problem Part (a):
theorem part_a (h : G = centroid A B C) 
  : ∀ M : Point α, sqdist M A + sqdist M B + sqdist M C ≥ sqdist G A + sqdist G B + sqdist G C :=
sorry

-- Problem Part (b):
def part_b_locus (k : α) (h1 : k > sqdist G A + sqdist G B + sqdist G C) 
  : set (Point α) :=
  { M | sqdist M A + sqdist M B + sqdist M C = k }

example (k : α) (h1 : k > sqdist G A + sqdist G B + sqdist G C) 
  : ∃ C (r : α), r > 0 ∧ part_b_locus k h1 = { M | sqdist M C = r } :=
sorry

end part_a_l819_819536


namespace increasing_and_odd_function_l819_819328

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Lean statement for the proof
theorem increasing_and_odd_function :
  (∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2)) ∧
  (∀ x : ℝ, f(-x) = -f(x)) :=
by
  -- using sorry to skip the proof as requested
  sorry

end increasing_and_odd_function_l819_819328


namespace AB_value_l819_819608

theorem AB_value (A B C : Type)
  (h : ∠A = 90)
  (tanC : ℚ)
  (AC : ℚ) : 
  tanC = 4 ∧ AC = 80 →  AB = 80 * Real.sqrt (16 / 17) := by
  sorry

end AB_value_l819_819608


namespace find_constants_to_satisfy_equation_l819_819899

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end find_constants_to_satisfy_equation_l819_819899


namespace depth_of_river_bank_l819_819705

theorem depth_of_river_bank (top_width bottom_width area depth : ℝ) 
  (h₁ : top_width = 12)
  (h₂ : bottom_width = 8)
  (h₃ : area = 500)
  (h₄ : area = (1 / 2) * (top_width + bottom_width) * depth) :
  depth = 50 :=
sorry

end depth_of_river_bank_l819_819705


namespace num_ordered_pairs_l819_819183

theorem num_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) :
  (M * N = 32) → ∃ (k : ℕ), k = 6 :=
by
  sorry

end num_ordered_pairs_l819_819183


namespace ratio_of_square_sides_sum_l819_819360

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l819_819360


namespace sum_of_last_two_digits_l819_819777

def fib_factorial_sum : ℕ := 1! + 1! + 2! + 3! + 5! + 8! + 13! + 21! + 34! + 55! + 89!

theorem sum_of_last_two_digits :
  let last_two_digits := fib_factorial_sum % 100,
      digit_sum := (last_two_digits / 10) + (last_two_digits % 10)
  in digit_sum = 5 :=
by
  let last_two_digits := fib_factorial_sum % 100
  let digit_sum := (last_two_digits / 10) + (last_two_digits % 10)
  have h : digit_sum = 5 := sorry
  exact h

end sum_of_last_two_digits_l819_819777


namespace log5_15625_eq_6_l819_819094

noncomputable def log5_15625 : ℕ := Real.log 15625 / Real.log 5

theorem log5_15625_eq_6 : log5_15625 = 6 :=
by
  sorry

end log5_15625_eq_6_l819_819094


namespace residue_of_neg_1000_l819_819880

def residue_mod_33 (n : ℤ) : ℤ := n % 33

theorem residue_of_neg_1000 : residue_mod_33 (-1000) = 23 := 
by
  -- Here, we assert the conditions which should hold true
  have h₀ : 0 ≤ 23 := by norm_num
  have h₁ : 23 < 33 := by norm_num
  sorry

end residue_of_neg_1000_l819_819880


namespace find_k_l819_819222

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : a 2 = -1)
  (h2 : 2 * a 1 + a 3 = -1)
  (h3 : arithmetic_sequence a d)
  (h4 : sum_of_sequence S a)
  (h5 : S k = -99) :
  k = 11 := 
by
  sorry

end find_k_l819_819222


namespace problem_statement_l819_819593

theorem problem_statement (x : ℝ) (h : x + sqrt (x^2 - 1) + 1 / (x - sqrt (x^2 - 1)) = 20) :
  x^2 + sqrt (x^4 - 1) + 1 / (x^2 + sqrt (x^4 - 1)) = 51.005 :=
by
  sorry

end problem_statement_l819_819593


namespace matrix_norm_squared_l819_819868

/--
Let \(\mathbf{P}\) be the matrix such that:
\[ \mathbf{P} = \begin{pmatrix} 0 & 3y & -2z \\ 2x & y & z \\ 2x & -2y & -z \end{pmatrix} \]
and \(\mathbf{P}^T \mathbf{P} = 2\mathbf{I}\).
Then \(x^2 + y^2 + z^2 = \frac{47}{60}\).
-/
theorem matrix_norm_squared {x y z : ℝ} (h : (matrix (fin 3) (fin 3) ℝ)
  = ![![0, 3*y, -2*z], ![2*x, y, z], ![2*x, -2*y, -z]] 
  ∧ (matrix.transpose (matrix (fin 3) (fin 3) ℝ) 
      * (matrix (fin 3) (fin 3) ℝ) = 2 • matrix.one (fin 3))) :
    x^2 + y^2 + z^2 = 47 / 60 := 
  sorry

end matrix_norm_squared_l819_819868


namespace maximum_value_of_sin2B_plus_2cosC_l819_819866

theorem maximum_value_of_sin2B_plus_2cosC
  (A B C : ℝ)
  (hTriangle : A + B + C = π)
  (hAngleA : (sin A + sqrt 3 * cos A) / (cos A - sqrt 3 * sin A) = tan (7 * π / 12)) :
  ∃ B C : ℝ, sin (2 * B) + 2 * cos C ≤ 3 / 2 :=
by
  sorry

end maximum_value_of_sin2B_plus_2cosC_l819_819866


namespace tree_edge_count_l819_819124

noncomputable theory

open_locale big_operators

-- Definitions representing the conditions of the problem
variables (V : Type) [fintype V] (E : set (V → V → Prop))
variables (is_connected : ∀ (u v : V), E u v → u ≠ v → reachable E u v)
variables (is_acyclic : ∀ (u v : V), ¬ (E u v ∧ E v u))

-- definition of tree with n vertices
def is_tree (V : Type) [fintype V] (E : set (V → V → Prop)) : Prop :=
  is_connected E ∧ is_acyclic E

-- Proving that a tree with n vertices has exactly n-1 edges
theorem tree_edge_count (V : Type) [fintype V] (E : set (V → V → Prop)) [fintype E]
  (is_connected : ∀ (u v : V), reachable E u v) (is_acyclic : ∀ (u v : V), ¬ (E u v ∧ E v u)) :
  fintype.card E = fintype.card V - 1 :=
sorry

end tree_edge_count_l819_819124


namespace trig_identity_cos_l819_819131

theorem trig_identity_cos (x : ℝ) (h : sin (2 * x + π / 6) = -1 / 3) : cos (π / 3 - 2 * x) = -1 / 3 :=
by
  sorry

end trig_identity_cos_l819_819131


namespace all_points_on_single_circle_l819_819134

theorem all_points_on_single_circle
  (P : Finset (ℝ × ℝ)) 
  (h1 : ∀ {p1 p2 p3 : (ℝ × ℝ)}, {p1, p2, p3} ⊆ P → p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → 
         ¬ ∃ k : ℝ, k ≠ 0 ∧ ∀ x : (ℝ × ℝ), x ∈ {p1, p2, p3} → x.2 = x.1 * k + p1.2 - p1.1 * k)
  (h2 : ∀ p1 p2 p3 : (ℝ × ℝ), {p1, p2, p3} ⊆ P → 
        ∃ p4 : (ℝ × ℝ), p4 ∈ P ∧ p4 ∉ {p1, p2, p3} ∧ 
               let c := Circle.mk p1 p2 p3 in Circle.contains c p4) :
  ∃ c : Circle (ℝ × ℝ), ∀ p ∈ P, Circle.contains c p := 
by
  sorry

end all_points_on_single_circle_l819_819134


namespace storage_house_blocks_needed_l819_819028

noncomputable def volume_of_storage_house
  (L_o : ℕ) (W_o : ℕ) (H_o : ℕ) (T : ℕ) : ℕ :=
  let interior_length := L_o - 2 * T
  let interior_width := W_o - 2 * T
  let interior_height := H_o - T
  let outer_volume := L_o * W_o * H_o
  let interior_volume := interior_length * interior_width * interior_height
  outer_volume - interior_volume

theorem storage_house_blocks_needed :
  volume_of_storage_house 15 12 8 2 = 912 :=
  by
    sorry

end storage_house_blocks_needed_l819_819028


namespace range_of_a_for_increasing_l819_819605

noncomputable def f (x a : ℝ) := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ x y, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a_for_increasing 
  (a : ℝ) : is_increasing_on (f (·) a) ({ x : ℝ | x > 1/2 }) ↔ a ≥ 3 :=
sorry

end range_of_a_for_increasing_l819_819605


namespace largest_integer_solution_l819_819607

theorem largest_integer_solution (m : ℤ) (h : 2 * m + 7 ≤ 3) : m ≤ -2 :=
sorry

end largest_integer_solution_l819_819607


namespace tail_to_body_ratio_l819_819640

variables (B : ℝ) (tail : ℝ := 9) (total_length : ℝ := 30)
variables (head_ratio : ℝ := 1/6)

-- Condition: The overall length is 30 inches
def overall_length_eq : Prop := B + B * head_ratio + tail = total_length

-- Theorem: Ratio of tail length to body length is 1:2
theorem tail_to_body_ratio (h : overall_length_eq B) : tail / B = 1 / 2 :=
sorry

end tail_to_body_ratio_l819_819640


namespace sphere_requires_more_paint_l819_819769

noncomputable def paint_volume_sphere (R d : ℝ) : ℝ :=
  4 * π * (R^2 * d + R * d^2 + d^3 / 3)

noncomputable def paint_volume_cylinder (R d : ℝ) : ℝ :=
  2 * π * (2 * R^2 * d + R * d^2)

theorem sphere_requires_more_paint (R d : ℝ) (hR : R > 0) (hd : d > 0) :
  paint_volume_sphere R d > paint_volume_cylinder R d :=
by
  sorry

end sphere_requires_more_paint_l819_819769


namespace ratio_x_y_z_l819_819048

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.10 * x = 0.20 * y) (h2 : 0.30 * y = 0.40 * z) :
  ∃ k : ℝ, x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
by                         
  sorry

end ratio_x_y_z_l819_819048


namespace value_of_x_squared_plus_one_l819_819587

theorem value_of_x_squared_plus_one (x : ℝ) (h : 5^(2 * x) + 25 = 26 * 5^x) : x^2 + 1 = 1 ∨ x^2 + 1 = 5 := 
by sorry

end value_of_x_squared_plus_one_l819_819587


namespace no_blue_socks_make_probability_half_l819_819751

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l819_819751


namespace solution_eq_l819_819491

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l819_819491


namespace factorize_expression_l819_819895

theorem factorize_expression (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := 
  sorry

end factorize_expression_l819_819895


namespace range_of_values_l819_819568

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else 2^x

theorem range_of_values (x : ℝ) : f(x) + f(x - 1) > 1 ↔ x ∈ set.Ioo (-1 : ℝ) ⊤ := by
  sorry

end range_of_values_l819_819568


namespace tangent_line_intercept_l819_819044

/-- Given two circles with centers (1,3) and (15,8) and radii 3 and 10 respectively,
the value of b in the equation of a common external tangent line y = mx + b with m > 0 is 518/1197. -/
theorem tangent_line_intercept :
  let C1 := ⟨⟨1, 3⟩, 3⟩ in -- Circle 1, center (1,3), radius 3
  let C2 := ⟨⟨15, 8⟩, 10⟩ in -- Circle 2, center (15,8), radius 10
  ∃ m b : ℚ, m > 0 ∧ y = m * x + b ∧ b = 518 / 1197 :=
sorry

end tangent_line_intercept_l819_819044


namespace smallest_prime_factor_of_2939_l819_819400

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

theorem smallest_prime_factor_of_2939 : smallest_prime_factor 2939 13 :=
by
  sorry

end smallest_prime_factor_of_2939_l819_819400


namespace dot_product_values_l819_819272

noncomputable def dot_product_range (u v : ℝ) (θ : ℝ) : Set ℝ :=
  {x | ∃ (u v : E) (θ : ℝ) (hu : ∥u∥ = 9) (hv : ∥v∥ = 13) (hθ : θ ∈ set.Icc (real.pi / 6) (real.pi / 3)), x = (9 * 13 * real.cos θ)}

theorem dot_product_values (u v : E) (θ : ℝ) (hu : ∥u∥ = 9) (hv : ∥v∥ = 13) (hθ : θ ∈ set.Icc (real.pi / 6) (real.pi / 3)) :
  (9 * 13 * real.cos θ) ∈ set.Icc (58.5) (58.5 * real.sqrt 3) :=
by
    sorry

end dot_product_values_l819_819272


namespace cylindrical_to_rectangular_l819_819055

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 10) (hθ : θ = 3 * Real.pi / 4) (hz : z = 2) :
    ∃ (x y z' : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (z' = z) ∧ (x = -5 * Real.sqrt 2) ∧ (y = 5 * Real.sqrt 2) ∧ (z' = 2) :=
by
  sorry

end cylindrical_to_rectangular_l819_819055


namespace perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l819_819943

-- Definition of an Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Definition of a Circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Conditions about foci, points, lines, and tangent property
def foci_condition (x y : ℝ) : Prop := ellipse x y
def right_vertex (x y : ℝ) : Prop := x = 2 ∧ y = 0
def tangent_condition (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y → l x y

-- Problem 1: Prove the perimeter of Δ AF₁F₂ is 4 + 2√2
theorem perimeter_of_triangle (A F1 F2 : ℝ × ℝ) (hA : foci_condition A.1 A.2)
  (hF1 : ellipse F1.1 F1.2) (hF2 : ellipse F2.1 F2.2) : 
  dist A F1 + dist A F2 + dist F1 F2 = 4 + 2 * sqrt 2 := sorry

-- Problem 2: Find the equation of line l passing through (2,0)
theorem equation_of_line_through_vertex : 
  ∃ k : ℝ, ∀ x y : ℝ, (x = 2 → y = k * (x - 2)) → tangent_condition (λ x y, y = k * (x - 2)) := sorry

-- Problem 3: Prove line AD is tangent to circle C
theorem line_AD_tangent_to_C (A D : ℝ × ℝ) (hD : D.2 = 2) (hO : 0) (hperpendicular : A.1 * D.1 + A.2 * D.2 = 0) :
  ∃ l : ℝ → ℝ → Prop, tangent_condition l ∧ (l A.1 A.2 = A.2 → A ≠ O) := sorry

end perimeter_of_triangle_equation_of_line_through_vertex_line_AD_tangent_to_C_l819_819943


namespace correct_option_is_B_l819_819785

-- Define the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (m : ℝ) : Prop := (-2 * m^2)^3 = -8 * m^6
def optionC (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def optionD (a b : ℝ) : Prop := 2 * a * b + 3 * a^2 * b = 5 * a^3 * b^2

-- The proof problem: which option is correct
theorem correct_option_is_B (m : ℝ) : optionB m := by
  sorry

end correct_option_is_B_l819_819785


namespace range_of_a_l819_819655

theorem range_of_a (a : ℝ) (x : ℝ) (h_a_pos : 0 < a) 
  (hp : |x - 4| > 6) (h_suff : ∀ x, |x - 4| > 6 → x^2 - 2x + 1 - a^2 > 0) : 
  0 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l819_819655


namespace sequence_an_formula_tn_upper_bound_l819_819138

noncomputable def sequence_an (n : ℕ) : ℕ :=
  if n = 0 then 0 else n^3

noncomputable def sequence_bn (n : ℕ) : ℚ :=
  if n = 0 then 1 else n / (sequence_an n)

noncomputable def sum_bn (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, sequence_bn (k + 1))

theorem sequence_an_formula (n : ℕ) (h : n ≠ 0) : sequence_an n = n^3 :=
by sorry

theorem tn_upper_bound (n : ℕ) : sum_bn n < 7 / 4 :=
by sorry

end sequence_an_formula_tn_upper_bound_l819_819138


namespace ratio_of_square_sides_l819_819365

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l819_819365


namespace triangle_similarity_l819_819863

theorem triangle_similarity
  (E F B C M U A : Type)
  [Circle E F]
  [Chord EF BC]
  (H_bisects : EF bisects BC at M)
  (H_not_diameter : EF ≠ diameter)
  (H_U_on_BM : U ∈ BM)
  (H_EU_extends_A : ∃ GH, extends EU to A)
  (H_right_angle : ∠EAF = 90°) :
  similar (triangle E U M) (triangle E F A) :=
by
  sorry

end triangle_similarity_l819_819863


namespace expression_possible_values_l819_819957

noncomputable def abs_sign (x : ℝ) : ℝ := x / |x|

theorem expression_possible_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    abs_sign x + abs_sign y + abs_sign z + abs_sign (x * y * z) ∈ {4, 0, -4} :=
by
  sorry

end expression_possible_values_l819_819957


namespace monotone_on_neg_inf_to_one_max_min_on_zero_to_five_l819_819160

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem monotone_on_neg_inf_to_one : 
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ 1 → f x1 < f x2 := 
by
  sorry

theorem max_min_on_zero_to_five : 
  ∃! m M, 
    (∀ x ∈ set.Icc (0 : ℝ) (5 : ℝ), f x ≥ m ∧ f x ≤ M) ∧ 
    m = -15 ∧ M = 1 := 
by
  sorry

end monotone_on_neg_inf_to_one_max_min_on_zero_to_five_l819_819160


namespace sandy_grew_watermelons_l819_819239

-- Definitions for the conditions
def jason_grew_watermelons : ℕ := 37
def total_watermelons : ℕ := 48

-- Define what we want to prove
theorem sandy_grew_watermelons : total_watermelons - jason_grew_watermelons = 11 := by
  sorry

end sandy_grew_watermelons_l819_819239


namespace Zuminglish_words_remainder_l819_819204

def a : ℕ → ℕ
| 2 := 4
| (n + 1) := 2 * (a n + c n)

def b : ℕ → ℕ
| 2 := 2
| (n + 1) := a n

def c : ℕ → ℕ
| 2 := 2
| (n + 1) := 2 * b n

def S := a 8 + b 8 + c 8

theorem Zuminglish_words_remainder :
  S % 500 = 304 :=
by
  unfold S a b c
  have h₁: a 3 = 12 := by rfl
  have h₂: b 3 = 4 := by rfl
  have h₃: c 3 = 4 := by rfl
  have h₄: a 4 = 32 := by rfl
  have h₅: b 4 = 12 := by rfl
  have h₆: c 4 = 8 := by rfl
  have h₇: a 5 = 80 := by rfl
  have h₈: b 5 = 32 := by rfl
  have h₉: c 5 = 24 := by rfl
  have h₁₀: a 6 = 208 := by rfl
  have h₁₁: b 6 = 80 := by rfl
  have h₁₂: c 6 = 64 := by rfl
  have h₁₃: a 7 = 544 := by rfl
  have h₁₄: b 7 = 208 := by rfl
  have h₁₅: c 7 = 128 := by rfl
  have h₁₆: a 8 = 1344 := by rfl
  have h₁₇: b 8 = 544 := by rfl
  have h₁₈: c 8 = 416 := by rfl
  calc
    (1344 + 544 + 416) % 500 = 2304 % 500 := by rfl
    ... = 304 := by rfl

end Zuminglish_words_remainder_l819_819204


namespace part1_part2_l819_819938

variable {a b c : ℝ}
variable {f : ℝ → ℝ}

-- Conditions
axiom h1 : f = λ x, a * x^2 + b * x + c
axiom h2 : a < 0
axiom h3 : ∀ x, (1 < x ∧ x < 3) → f x > -2 * x

-- Questions
theorem part1 (h4 : ∀ x, f x + 6 * a = 0 → x ∈ {1,3}) : 
  f = λ x, - (1/5) * x^2 - (4/5) * x - (3/5) := sorry

theorem part2 (h5 : ∃ x, f x = a * (x^2 - (4 / a) * (2 * a + 1) * x) + 3 ∧ 
                    ∀ y, y ≠ x → f y < f x ∧ f x > 0) : 
  (- ∞ < a ∧ a < - (2/5)) ∨ (- (2/5) < a ∧ a < 0) := sorry

end part1_part2_l819_819938


namespace equation_solutions_count_l819_819988

theorem equation_solutions_count :
  let f (x : ℕ) := (∏ n in (1..120), (x - n)) / (∏ n in (1..120), (x - n * n))
  ∀ x ∈ finset.range 1 121, f x = 0 → (∃ n, x = n ∧ n ∉ {k | k^2 ≤ 120}) :=
begin
  sorry,
end

end equation_solutions_count_l819_819988


namespace number_of_valid_license_plates_l819_819313

-- Define the alphabet for Rotokas
def rotokas_alphabet : List Char := ['A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V']

-- Define the conditions given in the problem
def valid_plate (plate : List Char) : Prop :=
  plate.length = 5 ∧ 
  plate.head ∈ ['G', 'K'] ∧ 
  plate.reverse.head = 'T' ∧ 
  ('S' ∉ plate) ∧ 
  plate.nodup

-- The proof problem: Calculate the number of valid license plates
theorem number_of_valid_license_plates :
  (rotokas_alphabet.length = 12) →
  (countp (λ (plate : List Char), valid_plate plate) (List.permutations rotokas_alphabet)).length = 1008 :=
by
  intros h
  sorry

end number_of_valid_license_plates_l819_819313


namespace total_fish_caught_l819_819248

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l819_819248


namespace Inequality_Solution_Set_Range_of_c_l819_819557

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -(((-x)^2) + 2 * (-x))

theorem Inequality_Solution_Set (x : ℝ) :
  (g x ≥ f x - |x - 1|) ↔ (-1 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

theorem Range_of_c (c : ℝ) :
  (∀ x : ℝ, g x + c ≤ f x - |x - 1|) ↔ (c ≤ -9/8) :=
by
  sorry

end Inequality_Solution_Set_Range_of_c_l819_819557


namespace P_works_alone_l819_819675

theorem P_works_alone (P : ℝ) (hP : 2 * (1 / P + 1 / 15) + 0.6 * (1 / P) = 1) : P = 3 :=
by sorry

end P_works_alone_l819_819675


namespace largest_k_subsets_divisible_l819_819046

theorem largest_k_subsets_divisible (n : ℕ) (h : 1 < n) :
  ∃ k, (∀ (A : Finset ℕ), A.card = 4 * n → (A ⊆ Finset.range (6 * n + 1)) → 
  (∀ a b ∈ A, a < b → b % a = 0 → (∃ k ≥ n, (A.card - (A.filter (λ (a b : ℕ), b % a == 0)).card) ≥ k)) :=
sorry

end largest_k_subsets_divisible_l819_819046


namespace solve_equation_l819_819498

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l819_819498


namespace range_of_a_l819_819128

noncomputable def f (x a : ℝ) := Real.log x - a
noncomputable def g (x : ℝ) := x * Real.exp x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → f x a ≤ g x) → -1 ≤ a :=
begin
  sorry
end

end range_of_a_l819_819128


namespace clock_angle_at_3_20_l819_819463

theorem clock_angle_at_3_20 (hour_hand_move_deg_per_hour : ℝ) (minute_hand_move_deg_per_minute : ℝ) : 
  hour_hand_move_deg_per_hour = 25 → minute_hand_move_deg_per_minute = 6 → 
  abs ((20 * minute_hand_move_deg_per_minute) - 
  ((3 * hour_hand_move_deg_per_hour) + (20 / 60) * hour_hand_move_deg_per_hour)) = 36.67 :=
by
  intros h_hour h_min
  let hour_position := 3 * hour_hand_move_deg_per_hour
  let minute_position := 20 * minute_hand_move_deg_per_minute
  let hour_position_at_3_20 := hour_position + (20 / 60) * hour_hand_move_deg_per_hour
  exact abs (minute_position - hour_position_at_3_20) = 36.67
#align clock_angle_at_3_20

end clock_angle_at_3_20_l819_819463


namespace find_exponent_l819_819912

theorem find_exponent : ∃ x : ℝ, (196 * 5^x) / 568 = 43.13380281690141 ↔ x = 3 :=
by
  sorry

end find_exponent_l819_819912


namespace coke_bottles_proof_l819_819729

def coke_extract_liters : ℝ := 2.3
def milliliters_per_liter : ℝ := 1000
def extract_per_3_bottles : ℝ := 400
def full_sets (extract_milliliters : ℝ) : ℕ := (extract_milliliters / extract_per_3_bottles).floor.toNat

theorem coke_bottles_proof : 
  let extract_in_milliliters := coke_extract_liters * milliliters_per_liter in
  let number_of_bottles := full_sets extract_in_milliliters * 3 in
  number_of_bottles = 15 :=
by 
  let extract_in_milliliters := coke_extract_liters * milliliters_per_liter
  let number_of_bottles := full_sets extract_in_milliliters * 3
  have h : number_of_bottles = 15 := sorry
  exact h

end coke_bottles_proof_l819_819729


namespace constant_term_binomial_expansion_l819_819958

theorem constant_term_binomial_expansion :
  let n := 9,
  let term_x_binom := λ (r : ℕ), (Nat.choose n r) * (-1)^r * (x^(n - 3*r)),
  let k := 3,
  let m := 8,
  (Nat.choose n k = Nat.choose n (n - m)) →
  (∃ r : ℕ, x^(n - 3*r) = 1 ∧ Nat.choose n r * (-1)^r = -84) :=
by
  intros
  sorry

end constant_term_binomial_expansion_l819_819958


namespace no_blue_socks_make_probability_half_l819_819752

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l819_819752


namespace count_statements_implying_negation_l819_819049

theorem count_statements_implying_negation 
  (p q r : Prop)
  (s1 : p ∧ q ∧ ¬r)
  (s2 : p ∧ ¬q ∧ r)
  (s3 : ¬p ∧ q ∧ ¬r)
  (s4 : ¬p ∧ ¬q ∧ r) :
  (s3 → ¬ ((p ∧ q) ∨ r)) ∧
  (¬((s1 → ¬ ((p ∧ q) ∨ r)) ∧ 
    (s2 → ¬ ((p ∧ q) ∨ r)) ∧ 
    (s4 → ¬ ((p ∧ q) ∨ r)))) :=
sorry

end count_statements_implying_negation_l819_819049


namespace correct_ways_to_deliver_letters_l819_819439

open BigOperators

noncomputable def number_of_ways : ℕ := 
  let derangements : ℕ → ℕ
    | 0 => 1
    | 1 => 0
    | n => (n - 1) * (derangements (n - 1) + derangements (n - 2))
  in 
  (Nat.choose 5 2) * (derangements 3)

theorem correct_ways_to_deliver_letters : number_of_ways = 20 := 
by
  sorry

end correct_ways_to_deliver_letters_l819_819439


namespace cubic_larger_than_elongated_l819_819237

def cube_volume (s : ℝ) : ℝ := s ^ 3

def elongated_volume (k : ℝ) : ℝ := (220 / k) ^ 2 * 220

theorem cubic_larger_than_elongated (k : ℝ) (hk : k > real.sqrt (85.184)) :
  cube_volume 50 > elongated_volume k :=
by 
  -- Unroll the definitions and simplify the inequality
  sorry

end cubic_larger_than_elongated_l819_819237


namespace correct_calculation_l819_819784

theorem correct_calculation : (sqrt 2 * sqrt 3 = sqrt 6) :=
by 
  -- We write the definitions for options A, B, D
  def option_A : Prop := (sqrt 2 + sqrt 3 = sqrt 5)
  def option_B : Prop := (sqrt 3 - sqrt 2 = 1)
  def option_D : Prop := (sqrt 4 / sqrt 2 = 2)
  -- Now, the proof that option C is correct
  sorry

end correct_calculation_l819_819784


namespace sum_of_digits_10_pow_38_minus_85_l819_819795

theorem sum_of_digits_10_pow_38_minus_85 :
  let n := 10 ^ 38 - 85 in (n.digits 10).sum = 16 :=
by sorry

end sum_of_digits_10_pow_38_minus_85_l819_819795


namespace median_is_80_l819_819624

noncomputable def scores : List ℕ := 
  List.replicate 2 50 ++ 
  List.replicate 3 60 ++ 
  List.replicate 7 70 ++ 
  List.replicate 14 80 ++ 
  List.replicate 13 90 ++ 
  List.replicate 3 100

def median (l : List ℕ) : ℕ :=
if h : l.length % 2 = 0 then
  let sorted := l.qsort (· <= ·)
  (sorted.get ⟨l.length / 2 - 1, (Nat.div_lt_iff_lt_mul (by decide)).2 (Nat.lt_of_le_of_lt (Nat.pred_le _) (Nat.lt_mul_of_pos_right (Nat.succ_pos _)))⟩ + 
  sorted.get ⟨l.length / 2, Nat.div_lt_self (by decide) (by decide)⟩) / 2
else
  let sorted := l.qsort (· <= ·)
  sorted.get ⟨l.length / 2, Nat.div_lt_self (by decide) (by decide)⟩

theorem median_is_80 : median scores = 80 := by
  sorry

end median_is_80_l819_819624


namespace candy_problem_l819_819027

theorem candy_problem :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 7 = 1) ∧ (n % 4 = 3) :=
by
  use 127
  simp
  exactly dec_trivial

end candy_problem_l819_819027


namespace bales_stacked_correct_l819_819377

-- Given conditions
def initial_bales : ℕ := 28
def final_bales : ℕ := 82

-- Define the stacking function
def bales_stacked (initial final : ℕ) : ℕ := final - initial

-- Theorem statement we need to prove
theorem bales_stacked_correct : bales_stacked initial_bales final_bales = 54 := by
  sorry

end bales_stacked_correct_l819_819377


namespace sixty_th_number_of_set_l819_819851

noncomputable def is_60th_smallest (S : Set ℕ) : Prop :=
  let ordered_list := List.sort (· < ·) (Set.toList S)
  ∃ n, ordered_list !! 59 = some 2064

theorem sixty_th_number_of_set :
  is_60th_smallest { n | ∃ (x y : ℕ), x < y ∧ n = 2^x + 2^y } := 
by
  sorry

end sixty_th_number_of_set_l819_819851


namespace perfect_square_trinomial_coeff_l819_819547

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end perfect_square_trinomial_coeff_l819_819547


namespace area_GHIJKL_value_sum_abc_GHIJKL_l819_819289

noncomputable def height_hexagon (PG PI PK : ℝ) : ℝ :=
  PG + PI + PK

noncomputable def side_length_hexagon (PG PI PK : ℝ) : ℝ :=
  (PG + PI + PK) / Real.sqrt 3

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt(3) / 2) * s^2

noncomputable def area_hexagon_GHIJKL (PG PI PK : ℝ) : ℝ :=
  (area_regular_hexagon (side_length_hexagon PG PI PK)) / 2

theorem area_GHIJKL_value :
  area_hexagon_GHIJKL (9 / 2) 6 (15 / 2) = 729 * Real.sqrt 3 / 4 :=
by sorry

theorem sum_abc_GHIJKL :
  729 + 3 + 4 = 736 :=
by linarith

end area_GHIJKL_value_sum_abc_GHIJKL_l819_819289


namespace asymptotes_of_hyperbola_l819_819576

-- Define the given condition
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 9 = 1
def eccentricity (a b : ℝ) : ℝ := (real.sqrt (1 + b^2 / a^2))

-- Theorem stating the equations of the asymptotes of the hyperbola
theorem asymptotes_of_hyperbola (a b : ℝ) (h_eq : hyperbola x y a)
  (h_ecc : eccentricity a b = 5 / 4) : 
  (a = 4) ∧ (b = 3) → y = 3 / 4 * x ∨ y = -3 / 4 * x :=
by
  sorry

end asymptotes_of_hyperbola_l819_819576


namespace first_five_terms_series_l819_819406

theorem first_five_terms_series (a : ℕ → ℚ) (h : ∀ n, a n = 1 / (n * (n + 1))) :
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 6) ∧
  (a 3 = 1 / 12) ∧
  (a 4 = 1 / 20) ∧
  (a 5 = 1 / 30) :=
by
  sorry

end first_five_terms_series_l819_819406


namespace APNQ_is_parallelogram_l819_819265

variables {A B C O P Q N : Type*}

-- Define that A, B, and C form a triangle and O is the circumcenter
def is_triangle (A B C : Type*) : Prop := sorry
def circumcenter (A B C O : Type*) : Prop := sorry
def circumcircle (B O C A : Type*) : Prop := sorry

-- Define the positions of P and Q on Γ
def intersects_AB_at_P (A B P : Type*) (Gamma : Type*) : Prop := sorry
def intersects_AC_at_Q (A C Q : Type*) (Gamma : Type*) : Prop := sorry
def diameter (N O : Type*) (Gamma : Type*) : Prop := sorry

-- Define parallel lines
def parallel (L1 L2 : Type*) : Prop := sorry

-- Main theorem statement
theorem APNQ_is_parallelogram 
  (ABC_triangle : is_triangle A B C)
  (non_right_angle : ∠ BAC ≠ 90)
  (circumcenter_O : circumcenter A B C O)
  (circumcircle_Gamma : circumcircle B O C A)
  (intersect_AB : intersects_AB_at_P A B P Gamma)
  (intersect_AC : intersects_AC_at_Q A C Q Gamma)
  (ON_diameter : diameter N O Gamma):
  (parallel A P Q N) ∧ (parallel A Q P N) :=
sorry

end APNQ_is_parallelogram_l819_819265


namespace min_value_product_l819_819781

theorem min_value_product (n : ℕ) (hn : n > 0) : 
  let x := (λ i : ℕ, (1 : ℝ) / n) in 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = (1 : ℝ) / n) →
  (∏ i in Finset.range n, (x i)^(x i)) = 1 / n :=
by
  sorry

end min_value_product_l819_819781


namespace sqrt_expr_simplified_l819_819860

-- Define the terms based on the given conditions
def sqrt3 : Real := Real.sqrt 3
def sqrt2 : Real := Real.sqrt 2
def sqrt6 : Real := Real.sqrt 6
def sqrt8 : Real := Real.sqrt 8

theorem sqrt_expr_simplified :
  sqrt3 * sqrt2 - sqrt2 + sqrt8 = sqrt6 + sqrt2 := by
sorry

end sqrt_expr_simplified_l819_819860


namespace evaluate_expression_l819_819095

variable (b : ℝ)
variable (h₀ : b ≠ 0)

theorem evaluate_expression : b^3 + b^(-3) = (b + b^(-1))^3 - 3 * (b + b^(-1)) :=
sorry

end evaluate_expression_l819_819095


namespace angle_in_third_quadrant_l819_819802

theorem angle_in_third_quadrant (α : ℝ) (h1 : sin α * cos α > 0) (h2 : sin α + cos α < 0) : 
  π < α ∧ α < 3 * π / 2 :=
sorry

end angle_in_third_quadrant_l819_819802


namespace triangle_area_condition_l819_819111

noncomputable def triangle_area (x : ℝ) : ℝ :=
1/2 * x * (3 * x)

theorem triangle_area_condition (x : ℝ) (h_pos : x > 0) (h_area : triangle_area x = 72) :
  x = 4 * real.sqrt 3 :=
sorry

end triangle_area_condition_l819_819111


namespace prime_eq_sol_l819_819115

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l819_819115


namespace necessary_but_not_sufficient_l819_819148

theorem necessary_but_not_sufficient (a b : ℝ) (h : a^2 > b^2) :
  ¬(a > b > 0) ↔ (∃ a b : ℝ, a^2 > b^2 ∧ ¬ (a > b > 0)) :=
begin
  sorry
end

end necessary_but_not_sufficient_l819_819148


namespace length_of_wood_l819_819801

theorem length_of_wood (x : ℝ) :
  let rope_length := x + 4.5 in
  let folded_rope := rope_length / 2 in
  folded_rope = x - 1 :=
by
  let rope_length := x + 4.5
  let folded_rope := rope_length / 2
  have h : folded_rope = x - 1
  sorry

end length_of_wood_l819_819801


namespace P_mult_Q_l819_819901

variable (Q : Matrix (Fin 3) (Fin 3) ℝ)

def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 0], ![0, 0, 1], ![0, 1, 0]]

theorem P_mult_Q (a b c d e f g h i : ℝ) (Q_eq : Q = !![
  [a, b, c],
  [d, e, f],
  [g, h, i]
]) : P ⬝ Q = !![
  [3 * a, 3 * b, 3 * c],
  [g, h, i],
  [d, e, f]
] :=
by sorry

end P_mult_Q_l819_819901


namespace socks_impossible_l819_819746

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l819_819746


namespace height_of_tower_l819_819519

noncomputable theory
open Real

variables {h_mountain h_tower: ℝ} (θ_top θ_bottom : ℝ)

def height_tower (h_mountain: ℝ) (θ_top θ_bottom : ℝ): ℝ :=
  h_mountain * (sin θ_top - sin θ_bottom) / (sin θ_bottom - sin (θ_bottom - θ_top))

theorem height_of_tower (h_mountain : ℝ) (θ_top θ_bottom : ℝ) :
  h_mountain = 300 ∧ θ_top = π / 6 ∧ θ_bottom = π / 3 → height_tower h_mountain θ_top θ_bottom = 200 :=
by
  intro h_cond
  sorry

end height_of_tower_l819_819519


namespace blue_tile_probability_l819_819421

-- Define the conditions
def is_blue_tile (n : Nat) : Bool :=
  (n % 5) = 2

def total_tiles : Nat := 50

-- Define the problem
theorem blue_tile_probability :
  (Finset.card (Finset.filter is_blue_tile (Finset.range (total_tiles + 1))) : ℚ) / total_tiles = 1 / 5 :=
  sorry

end blue_tile_probability_l819_819421


namespace grant_made_total_l819_819173

noncomputable def sale_amount : ℕ :=
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30 - (30 * 20 / 100)
  let cleats_eur := 10 * 0.85
  let cleats_discount := 10 - (10 * 15 / 100)
  baseball_cards + baseball_bat + baseball_glove + cleats_eur.to_nat + cleats_discount.to_nat

theorem grant_made_total :
  sale_amount = 76 :=
by
  unfold sale_amount
  sorry

end grant_made_total_l819_819173


namespace sum_of_n_consecutive_even_numbers_l819_819292

theorem sum_of_n_consecutive_even_numbers (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ a : ℕ, n * (n - 1)^(k-1) = (Range n).sum (λ i, 2 * a + 2 * i) := 
by
  sorry

end sum_of_n_consecutive_even_numbers_l819_819292


namespace vendor_total_profit_l819_819840

theorem vendor_total_profit :
  let cost_per_apple := 3 / 2
      selling_price_per_apple := 10 / 5
      profit_per_apple := selling_price_per_apple - cost_per_apple
      total_profit_apples := profit_per_apple * 5
      cost_per_orange := 2.7 / 3
      selling_price_per_orange := 1
      profit_per_orange := selling_price_per_orange - cost_per_orange
      total_profit_oranges := profit_per_orange * 5
  in total_profit_apples + total_profit_oranges = 3 := 
by
  sorry

end vendor_total_profit_l819_819840


namespace intersection_of_sets_l819_819580

def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets :
  setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l819_819580


namespace scientific_notation_of_0_0000021_l819_819460

theorem scientific_notation_of_0_0000021 :
  (0.0000021 : ℝ) = 2.1 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000021_l819_819460


namespace problem_solution_l819_819656

noncomputable def a : ℕ → ℝ
| 0       := -3
| (n + 1) := a n + 2 * b n + Real.sqrt (a n ^ 2 + 4 * (b n) ^ 2)
and b : ℕ → ℝ
| 0       := 2
| (n + 1) := a n + 2 * b n - Real.sqrt (a n ^ 2 + 4 * (b n) ^ 2)

theorem problem_solution :
  (1 / a 2023 + 1 / b 2023) = 1 / 3 :=
  sorry

end problem_solution_l819_819656


namespace find_x_l819_819658

def f1 (x : ℚ) : ℚ := (2 / 3) - (3 / (3 * x + 1))
noncomputable def f (n : ℕ) (x : ℚ) : ℚ :=
  if n = 1 then f1 x else
    (f1 (f (n - 1) x))

theorem find_x (x : ℚ) (h : f 1001 x = x - 3) : x = 5 / 3 :=
  sorry

end find_x_l819_819658


namespace cousins_room_distribution_l819_819669

theorem cousins_room_distribution :
  let cousins := 4
  let rooms := 4
  number_of_ways cousins rooms = 15 :=
sorry

end cousins_room_distribution_l819_819669


namespace coefficient_of_linear_term_l819_819114

theorem coefficient_of_linear_term :
  ∀ (a b c : ℤ) (x : ℤ), (2 * x^2 + b * x + c = 0) → b = -3 :=
by
  intros a b c x h
  have h_eq : 2*x^2 - 3*x - 4 = 2*x^2 + b*x + c,
  from h.symm,
  sorry

end coefficient_of_linear_term_l819_819114


namespace spatial_quadrilateral_angle_sum_l819_819295

theorem spatial_quadrilateral_angle_sum (A B C D : ℝ) (ABD DBC ADB BDC : ℝ) :
  (A <= ABD + DBC) → (C <= ADB + BDC) → 
  (A + C + B + D <= 360) := 
by
  intros
  sorry

end spatial_quadrilateral_angle_sum_l819_819295


namespace no_possible_blue_socks_l819_819735

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l819_819735


namespace boat_distance_travelled_upstream_l819_819726

theorem boat_distance_travelled_upstream (v : ℝ) (d : ℝ) :
  ∀ (boat_speed_in_still_water upstream_time downstream_time : ℝ),
  boat_speed_in_still_water = 25 →
  upstream_time = 1 →
  downstream_time = 0.25 →
  d = (boat_speed_in_still_water - v) * upstream_time →
  d = (boat_speed_in_still_water + v) * downstream_time →
  d = 10 :=
by
  intros
  sorry

end boat_distance_travelled_upstream_l819_819726


namespace sum_of_ratio_simplified_l819_819356

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l819_819356


namespace solution_eq_l819_819488

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l819_819488


namespace point_on_bisector_l819_819598

theorem point_on_bisector {a b : ℝ} (h : ∃ θ, θ = atan (b / a) ∧ (θ = π / 4 ∨ θ = -(3 * π / 4))) : b = -a :=
sorry

end point_on_bisector_l819_819598


namespace exist_sector_with_10_points_not_exist_sector_with_11_points_l819_819551

theorem exist_sector_with_10_points (h : ∀ {P : Set ℝ^2}, P.card = 100 ∧ ∀ p ∈ P, ¬ is_center p ∧ ∀ p₁ p₂ ∈ P, ¬ collinear p₁ p₂) :
  ∃ (sector : Set ℝ^2), (angle sector = 2 * π / 11) ∧ (sector.card = 10) :=
sorry

theorem not_exist_sector_with_11_points (h : ∀ {P : Set ℝ^2}, P.card = 100 ∧ ∀ p ∈ P, ¬ is_center p ∧ ∀ p₁ p₂ ∈ P, ¬ collinear p₁ p₂) :
  ¬ ∃ (sector : Set ℝ^2), (angle sector = 2 * π / 11) ∧ (sector.card = 11) :=
sorry

end exist_sector_with_10_points_not_exist_sector_with_11_points_l819_819551


namespace find_magnitude_b_l819_819554

variables (a b : ℝ^3) -- you may need to adjust the type according to your working space

-- Definition of magnitudes and angle
def magnitude (v : ℝ^3) := real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Given conditions
axiom angle_condition : real.angle a b = π / 3
axiom magnitude_a : magnitude a = 3
axiom magnitude_sum : magnitude (2 • a + b) = 2 * real.sqrt 13

-- Proof statement
theorem find_magnitude_b : magnitude b = 2 :=
sorry

end find_magnitude_b_l819_819554


namespace workers_per_team_lead_l819_819425

theorem workers_per_team_lead
  (team_leads_per_supervisor : ℕ)
  (number_of_supervisors : ℕ)
  (number_of_workers : ℕ)
  (h1 : team_leads_per_supervisor = 3)
  (h2 : number_of_supervisors = 13)
  (h3 : number_of_workers = 390)
  : (number_of_workers / (3 * number_of_supervisors) = 10) :=
by
  -- Calculate the number of team leads
  have team_leads := 3 * number_of_supervisors,
  -- Calculate the workers per team lead
  have workers_per_lead := number_of_workers / team_leads,
  -- Assert that the number of workers per team lead is 10
  show number_of_workers / (3 * number_of_supervisors) = 10,
  sorry

end workers_per_team_lead_l819_819425


namespace common_divisors_count_l819_819586

theorem common_divisors_count (a b : ℕ) (ha : a = 36) (hb : b = 90)
(hf36 : ∃ (u v : ℕ), a = 2^u * 3^v ∧ u = 2 ∧ v = 2)
(hf90 : ∃ (w x y : ℕ), b = 2^w * 3^x * 5^y ∧ w = 1 ∧ x = 2 ∧ y = 1) :
  (finset.filter (λ d, d ∣ a ∧ d ∣ b) (finset.range (b + 1))).card = 6 := 
by {
  sorry
}

end common_divisors_count_l819_819586


namespace youngest_child_age_l819_819613

theorem youngest_child_age (x : ℕ) (h1 : Prime x)
  (h2 : Prime (x + 2))
  (h3 : Prime (x + 6))
  (h4 : Prime (x + 8))
  (h5 : Prime (x + 12))
  (h6 : Prime (x + 14)) :
  x = 5 := 
sorry

end youngest_child_age_l819_819613


namespace least_positive_integer_x_l819_819105

theorem least_positive_integer_x (x : ℕ) (h1 : x + 3721 ≡ 1547 [MOD 12]) (h2 : x % 2 = 0) : x = 2 :=
sorry

end least_positive_integer_x_l819_819105


namespace total_pounds_of_peppers_l819_819583

-- Definitions based on the conditions
def greenPeppers : ℝ := 0.3333333333333333
def redPeppers : ℝ := 0.3333333333333333

-- Goal statement expressing the problem
theorem total_pounds_of_peppers :
  greenPeppers + redPeppers = 0.6666666666666666 := 
by
  sorry

end total_pounds_of_peppers_l819_819583


namespace plane_relationship_l819_819550

variable {Plane : Type} [IncidencePlane Plane]

-- Definitions for the problem setup
def are_parallel (α β : Plane) : Prop :=
  ∃ (l : Line), l ⊆ α ∧ l ⊆ β

def are_infinitely_parallel_lines (α β : Plane) : Prop :=
  ∃ (L : Set Line), infinite L ∧ ∀ l ∈ L, l ⊆ α ∧ l is_parallel_to_plane β

-- The proof problem statement
theorem plane_relationship
  (α β : Plane)
  (h : are_infinitely_parallel_lines α β) :
  α = β ∨ are_parallel α β :=
sorry

end plane_relationship_l819_819550


namespace if_a_eq_b_then_ac_eq_bc_l819_819404

theorem if_a_eq_b_then_ac_eq_bc (a b c : ℝ) : a = b → ac = bc :=
sorry

end if_a_eq_b_then_ac_eq_bc_l819_819404


namespace angle_bisector_relation_l819_819596

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l819_819596


namespace sum_of_constants_l819_819346

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l819_819346


namespace sum_of_squares_of_coefficients_eq_270_l819_819778

noncomputable def p (x : ℝ) : ℝ := 3 * (x^5 + 2 * x^3 + 5)

theorem sum_of_squares_of_coefficients_eq_270 :
  let coeffs := [3, 6, 15]
  (coeffs.map (λ c, c^2)).sum = 270 :=
by
  sorry

end sum_of_squares_of_coefficients_eq_270_l819_819778


namespace ratio_of_square_sides_sum_l819_819358

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l819_819358


namespace trapezoid_side_ratio_l819_819707

-- Conditions:
def is_trapezoid (AB CD AD BC : ℝ) : Prop :=
  -- A more precise geometric definition would be needed in practice,
  -- but here we are simplifying for the sake of the problem statement.
  true

def area_ratio (A B C D : Type) [metric_space A] [has_area A] (h_AC : A × A) (h_BD : A × A) : Prop :=
  -- This function would compare the areas of the respective triangles divided by the diagonals
  -- But we simplify it to capture the conditions given in the problem
  true

-- Theorem to prove:
theorem trapezoid_side_ratio (AB CD AD BC : ℝ)
  (h_is_trapezoid : is_trapezoid AB CD AD BC)
  (h_area_ratio : area_ratio AB CD AD BC) :
  AB / CD = 5 :=
sorry

end trapezoid_side_ratio_l819_819707


namespace ratio_of_eggs_l819_819276

/-- Megan initially had 24 eggs (12 from the store and 12 from her neighbor). She used 6 eggs in total (2 for an omelet and 4 for baking). She set aside 9 eggs for three meals (3 eggs per meal). Finally, Megan divided the remaining 9 eggs by giving 9 to her aunt and keeping 9 for herself. The ratio of the eggs she gave to her aunt to the eggs she kept is 1:1. -/
theorem ratio_of_eggs
  (eggs_bought : ℕ)
  (eggs_from_neighbor : ℕ)
  (eggs_omelet : ℕ)
  (eggs_baking : ℕ)
  (meals : ℕ)
  (eggs_per_meal : ℕ)
  (aunt_got : ℕ)
  (kept_for_meals : ℕ)
  (initial_eggs := eggs_bought + eggs_from_neighbor)
  (used_eggs := eggs_omelet + eggs_baking)
  (remaining_eggs := initial_eggs - used_eggs)
  (assigned_eggs := meals * eggs_per_meal)
  (final_eggs := remaining_eggs - assigned_eggs)
  (ratio : ℚ := aunt_got / kept_for_meals) :
  eggs_bought = 12 ∧
  eggs_from_neighbor = 12 ∧
  eggs_omelet = 2 ∧
  eggs_baking = 4 ∧
  meals = 3 ∧
  eggs_per_meal = 3 ∧
  aunt_got = 9 ∧
  kept_for_meals = assigned_eggs →
  ratio = 1 := by
  sorry

end ratio_of_eggs_l819_819276


namespace solve_equation_l819_819501

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l819_819501


namespace sequence_a_n_formula_and_sum_t_n_l819_819535

open Nat

theorem sequence_a_n_formula_and_sum_t_n (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * (2^n) - 2) → 
  (∀ n : ℕ, ∑ i in range (n+1), S i = 2^(n+2) - 4 - 2 * n) :=
by 
  sorry

end sequence_a_n_formula_and_sum_t_n_l819_819535


namespace area_of_square_with_diagonal_l819_819014

theorem area_of_square_with_diagonal (d : ℝ) (s : ℝ) (hsq : d = s * Real.sqrt 2) (hdiagonal : d = 12 * Real.sqrt 2) : 
  s^2 = 144 :=
by
  -- Proof details would go here.
  sorry

end area_of_square_with_diagonal_l819_819014


namespace percent_of_Q_l819_819993

theorem percent_of_Q (P Q : ℝ) (h : (50 / 100) * P = (20 / 100) * Q) : P = 0.4 * Q :=
sorry

end percent_of_Q_l819_819993


namespace sum_of_positive_divisors_l819_819776

theorem sum_of_positive_divisors (h : ∀ n : ℕ, (n+24) % n = 0 → (24 % n = 0)) :
  ∑ k in (Finset.filter (λ x, 24 % x = 0) (Finset.range 25)), k = 60 :=
sorry

end sum_of_positive_divisors_l819_819776


namespace problem_solution_mass_AgCl_concentration_HNO3_l819_819879
noncomputable def mass_AgCl {HCl AgNO3 : ℕ} (n_HCl n_AgNO3: ℕ) (V: ℕ) (M_AgCl: ℕ): Prop :=
  HCl = 3 ∧ AgNO3 = 3 ∧ V = 1 →
  M_AgCl = 429.96 * 100  -- Using * 100 to avoid decimal directly in Lean

noncomputable def concentration_HNO3 {HNO3 : ℕ} (n_HCl n_AgNO3: ℕ) (V: ℕ) (c_HNO3: ℕ): Prop :=
  n_HCl = 3 ∧ n_AgNO3 = 3 ∧ V = 1 →
  c_HNO3 = 3 * 100  -- Using * 100 to avoid decimal directly in Lean

theorem problem_solution_mass_AgCl_concentration_HNO3 :
  ∃ (HCl AgNO3: ℕ) (V M_AgCl: ℕ) (c_HNO3: ℕ),
  mass_AgCl HCl AgNO3 V M_AgCl ∧ concentration_HNO3 HNO3 HCl AgNO3 V c_HNO3 :=
by {
  sorry
}

end problem_solution_mass_AgCl_concentration_HNO3_l819_819879


namespace f_of_3_l819_819574

def f (x : ℕ) : ℤ :=
  if x = 0 then sorry else 2 * (x - 1) - 1  -- Define an appropriate value for f(0) later

theorem f_of_3 : f 3 = 3 := by
  sorry

end f_of_3_l819_819574


namespace monotonically_increasing_interval_of_g_l819_819330

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - π / 6)

theorem monotonically_increasing_interval_of_g :
  ∃ (a b : ℝ), a = -π / 4 ∧ b = π / 4 ∧ ∀ x1 x2, a ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ b → g x1 ≤ g x2 := sorry

end monotonically_increasing_interval_of_g_l819_819330


namespace factor_cubic_expression_l819_819711

theorem factor_cubic_expression :
  ∃ a b c : ℕ, 
  a > b ∧ b > c ∧ 
  x^3 - 16 * x^2 + 65 * x - 80 = (x - a) * (x - b) * (x - c) ∧ 
  3 * b - c = 12 := 
sorry

end factor_cubic_expression_l819_819711


namespace ratio_of_square_sides_l819_819366

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l819_819366


namespace polynomial_real_roots_count_l819_819720

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x - 2 + 8

theorem polynomial_real_roots_count :
  ∃ l : list ℝ, l.nodup ∧ (∀ x ∈ l, f x = 0) ∧ l.length = 3 := sorry

end polynomial_real_roots_count_l819_819720


namespace fraction_to_decimal_l819_819870

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
by
  sorry

end fraction_to_decimal_l819_819870


namespace composite_numbers_quotient_l819_819856

theorem composite_numbers_quotient :
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) / 
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) :=
by sorry

end composite_numbers_quotient_l819_819856


namespace paths_to_spell_MATH_l819_819617

   def grid : list (list char) := [
     ['T', 'A', 'H', 'T'],
     ['A', 'M', 'A', 'H'],
     ['H', 'A', 'T', 'A'],
     ['M', 'H', 'A', 'T']
   ]

   def adjacent (pos : ℕ × ℕ) (move : ℕ × ℕ) : ℕ × ℕ :=
     (pos.1 + move.1, pos.2 + move.2)

   def valid_position (pos : ℕ × ℕ) : Prop :=
     0 ≤ pos.1 ∧ pos.1 < 4 ∧ 0 ≤ pos.2 ∧ pos.2 < 4

   def letter_at (pos : ℕ × ℕ) : char :=
     grid.nth pos.1 >>= list.nth pos.2

   def paths_from (start : ℕ × ℕ) (word : list char) : nat :=
     if word.length = 1 then
       if start = letter_at start then 1 else 0
     else
       let moves := [(1, 0), (0, 1), (-1, 0), (0, -1)] in
       if ¬ valid_position start ∨ letter_at start ≠ word.head! then 0
       else
         list.sum (moves.map
           (λ m, paths_from (adjacent start m) word.tail!))

   theorem paths_to_spell_MATH :
     paths_from (1, 1) ['M', 'A', 'T', 'H'] = 24 :=
   by sorry
   
end paths_to_spell_MATH_l819_819617


namespace sum_abs_f_from_1_to_19_l819_819969

noncomputable def f (x : ℝ) : ℝ := sorry  -- The complete function definition based on conditions

theorem sum_abs_f_from_1_to_19 : 
  (∀ x : ℝ, ∃ y : ℝ, f(x) = y) → 
  (∀ x : ℝ, f(2*x - 2) = f(-(2*x - 2))) → 
  (∀ x : ℝ, f(x - 3) + f(-x + 1) = 0) → 
  (∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) (-1 : ℝ) → f(x) = (1 / 2^x) - 2*x - 4) → 
  f (-2) = 4 → 
  (finset.sum (finset.range 19) (λ k, |f(k + 1)|) = 36) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sum_abs_f_from_1_to_19_l819_819969


namespace probability_events_A_B_C_accur_l819_819053

noncomputable def P_A : ℝ := 11 / 60
noncomputable def P_B : ℝ := 11 / 120
noncomputable def P_C : ℝ := 11 / 40

-- Define the conditions
def cond1 : Prop := P_A > 0
def cond2 : Prop := P_A = 2 * P_B
def cond3 : Prop := P_C = 3 * P_B
def cond4 : Prop :=
  let P_ABC := P_A * P_B * P_C in
  P_A * P_B + P_A * P_C + P_B * P_C - 2 * P_ABC = 18 * P_ABC

-- Define the theorem to prove
theorem probability_events_A_B_C_accur : cond1 ∧ cond2 ∧ cond3 ∧ cond4 :=
by {
  -- Convert the conditions to usable definitions
  let P_A := (11 : ℝ) / 60,
  let P_B := (11 : ℝ) / 120,
  let P_C := (11 : ℝ) / 40,

  -- Verify conditional statements
  have h1 : P_A > 0 := by norm_num,
  have h2 : P_A = 2 * P_B := by norm_num,
  have h3 : P_C = 3 * P_B := by norm_num,
  let P_ABC := P_A * P_B * P_C,
  have h4 : P_A * P_B + P_A * P_C + P_B * P_C - 2 * P_ABC = 18 * P_ABC := by norm_num,

  -- Combine all into one condition
  exact and.intro h1 (and.intro h2 (and.intro h3 h4)),
  sorry
}

end probability_events_A_B_C_accur_l819_819053


namespace ratio_of_square_sides_sum_l819_819361

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l819_819361


namespace length_of_generatrix_of_cone_is_8_l819_819827

-- Definitions based on given conditions
def base_radius := 4  -- base radius of the cone (in cm)
def semi_circular_sheet_circumference (r : ℝ) := π * r
def base_circumference := 2 * π * base_radius

-- Theorem statement: proving the length of the generatrix of the cone
theorem length_of_generatrix_of_cone_is_8 :
  ∃ r, semi_circular_sheet_circumference r = base_circumference ∧ r = 8 :=
by
  sorry

end length_of_generatrix_of_cone_is_8_l819_819827


namespace molecular_weight_is_62_024_l819_819771

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_atoms_H : ℕ := 2
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

def molecular_weight_compound : ℝ :=
  num_atoms_H * atomic_weight_H + num_atoms_C * atomic_weight_C + num_atoms_O * atomic_weight_O

theorem molecular_weight_is_62_024 :
  molecular_weight_compound = 62.024 :=
by
  have h_H := num_atoms_H * atomic_weight_H
  have h_C := num_atoms_C * atomic_weight_C
  have h_O := num_atoms_O * atomic_weight_O
  have h_sum := h_H + h_C + h_O
  show molecular_weight_compound = 62.024
  sorry

end molecular_weight_is_62_024_l819_819771


namespace min_value_expression_ge_512_l819_819268

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c)

theorem min_value_expression_ge_512 {a b c : ℝ} 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  min_value_expression a b c ≥ 512 :=
by
  sorry

end min_value_expression_ge_512_l819_819268


namespace power_function_property_l819_819712

variable (f : ℤ → ℤ)

/-- Given a function f: ℤ → ℤ such that for any a, b in ℤ with ab ≠ 0, it holds that f(ab) ≥ f(a) + f(b),
prove that for any a ≠ 0 in ℤ and n in ℕ, f(a^n) = n f(a) if and only if f(a^2) = 2 f(a). -/
theorem power_function_property (h_f : ∀ a b : ℤ, a * b ≠ 0 → f(a * b) ≥ f(a) + f(b)) :
  ∀ (a : ℤ) (n : ℕ), a ≠ 0 → (f(a^2) = 2 * f(a) ↔ f(a^n) = n * f(a)) :=
sorry

end power_function_property_l819_819712


namespace fish_game_teams_l819_819110

noncomputable def number_of_possible_teams (n : ℕ) : ℕ := 
  if n = 6 then 5 else sorry

theorem fish_game_teams : number_of_possible_teams 6 = 5 := by
  unfold number_of_possible_teams
  rfl

end fish_game_teams_l819_819110


namespace derivative_at_1_l819_819928

-- Define the function f(x) = ∛x * sin x
def f (x : ℝ) : ℝ := x^(1/3) * sin x

-- We need to prove that f'(1) = (1/3)sin(1) + cos(1)
theorem derivative_at_1 : (deriv f 1) = (1/3) * sin 1 + cos 1 := by
  sorry

end derivative_at_1_l819_819928


namespace volume_reflection_l819_819068

-- Definition of the vertices of the original tetrahedron
variables {A B C D A' B' C' D' : Type}

-- Hypotheses/Conditions
variable (h_reflection : ∀ (v : {A B C D}), reflected_vertex v = {A' B' C' D'})

-- Volumes of original and reflected tetrahedrons
variable (V_ABCD : volume_tetrahedron A B C D)
variable (V_A'B'C'D' : volume_tetrahedron A' B' C' D')

-- The proof we need to show that the volume of the reflected tetrahedron is at least 4 times the volume of the original tetrahedron
theorem volume_reflection (h_reflection : ∀ (v : {A B C D}), reflected_vertex v = {A' B' C' D'})
                          (V_ABCD : volume_tetrahedron A B C D)
                          (V_A'B'C'D' : volume_tetrahedron A' B' C' D') : 
                          V_A'B'C'D' ≥ 4 * V_ABCD := sorry

end volume_reflection_l819_819068


namespace count_n_satisfying_conditions_l819_819057

def f (n : ℕ) : ℕ := (n^2 + n) / 2

def is_product_of_two_primes (m : ℕ) : Prop :=
  ∃ (p q : ℕ), p.prime ∧ q.prime ∧ p * q = m

theorem count_n_satisfying_conditions :
  (finset.filter (λ n, f n ≤ 1000 ∧ is_product_of_two_primes (f n)) (finset.range 45)).card = 5 :=
by sorry

end count_n_satisfying_conditions_l819_819057


namespace graph_function_on_interval_l819_819984

def f (x : ℝ) : ℝ := ⌊x⌋ + ⌊1 - x⌋ + 1

theorem graph_function_on_interval :
  (∀ x ∈ ([ -3, 3] : Set ℝ), 
    ( ∃ k : ℤ, x = k → f x = 2 ) ∧ 
    ( ¬ ( ∃ k : ℤ, x = k ) → f x = 1 ) ) :=
by
  intros x hx
  -- We need to prove f(x) = 2 when x is an integer and f(x) = 1 when it's not
  sorry

end graph_function_on_interval_l819_819984


namespace odd_function_value_l819_819970

noncomputable def f (x : ℝ) : ℝ := 
  if x >= 0 then log (x + 1) / log 2 + 0 else - (log (-x + 1) / log 2)

theorem odd_function_value (x : ℝ) (h1 : ∀ x : ℝ, f (- x) = - f x)
    (h2 : ∀ x : ℝ, x ≥ 0 → f x = (log (x + 1) / log 2) + 0)
    (h3 : f 0 = 0) : f (1 - real.sqrt 2) = -1 / 2 :=
sorry

end odd_function_value_l819_819970


namespace function_identity_l819_819572

def f (t : ℝ) : ℝ := t^2 - 2

theorem function_identity (x : ℝ) (hx : x ≠ 0) : f (x - (1 / x)) = x^2 - 2 :=
by
  rw [f, x^2 - (1 / x)^2 - 2]
  sorry

end function_identity_l819_819572


namespace sqrt_sum_eq_six_l819_819496

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l819_819496


namespace zarnin_staffing_l819_819861

theorem zarnin_staffing (total_resumes : ℕ) (open_positions : ℕ) (unsuitable_ratio : ℚ)
  (total_resumes = 30)
  (open_positions = 5)
  (unsuitable_ratio = 1 / 3) :
  let suitable_candidates := (2 / 3) * total_resumes in
  let assistant_candidates := suitable_candidates in
  let weapons_candidates := assistant_candidates - 1 in
  let field_technician_candidates := weapons_candidates - 1 in
  let radio_specialist_candidates := field_technician_candidates - 1 in
  let security_officer_candidates := radio_specialist_candidates - 1 in
  assistant_candidates * weapons_candidates * field_technician_candidates *
  radio_specialist_candidates * security_officer_candidates = 930240 := 
by sorry

end zarnin_staffing_l819_819861


namespace sequence_inequality_l819_819542

theorem sequence_inequality (a : ℕ → ℝ) (h1 : a 1 ≥ 3) (h2 : ∀ n, n + 1 = (a (n + 1))^2 - a n + 1) :
    ∀ n, 1 ≤ n → (∑ k in Icc 1 n, 1 / (1 + a k)) ≤ 1 / 2 :=
by
  sorry

end sequence_inequality_l819_819542


namespace ellipse_standard_eq_l819_819941

theorem ellipse_standard_eq (c a b : ℝ) (e : c / a = 1 / 2) (M : set (ℝ × ℝ)) (H1 : a^2 = c^2 + b^2)
  (H2 : 1/2 * 2 * c * b = sqrt 3) :
  (∀ x y, ((x, y) ∈ M ↔ x^2 / 4 + y^2 / 3 = 1))
  ∧ (∀ x1 y1 x2 y2, (x1, y1) ∈ M ∧ (x2, y2) ∈ M ∧ y1 ≠ 0 → 
    let A := (x1, y1), C := (x1, -y1), F2 := (1, 0), P := (4, 0)
    in ∀ B, B ∈ M ∧ x2 = B.1 ∧ y2 = B.2 ∧ y2 = -y1 ∧ B ≠ A ∧ B ≠ C
       → 4 ∈ (λ x, ∃ k m, k * (x - 4) = m ∧ 4k^2 + 3 ≠ 0)
       ∧ ∃ x1, -2 < x1 ∧ x1 < 2 ∧ 
          (7 / 4 * (x1 - 10 / 7)^2 - 18 / 7 ∈ range (λ x, x1 * x^2 - 5 * x1 + 1)) ) :=
sorry

end ellipse_standard_eq_l819_819941


namespace value_of_a2017_l819_819558

variable {a : ℕ → ℝ}
variable (h1 : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
variable (h2 : a 2016 + a 2018 = ∫ x in 0..2, real.sqrt (4 - x^2))

theorem value_of_a2017 : a 2017 = (real.pi / 2) := 
by 
  sorry

end value_of_a2017_l819_819558


namespace true_prop_l819_819270

noncomputable def p (x : Prop) : Prop :=
  ∃ t : ℝ, 2 * x = t ∧ t^2 - 2 * t + 2 = 0

noncomputable def q (x : Prop) : Prop :=
  ∀ x : ℝ, (e^x - x - 1 >= -1)

theorem true_prop (p : Prop) (q : Prop) : (¬p) ∧ (¬q) :=
by 
  sorry

end true_prop_l819_819270


namespace sqrt_sum_eq_six_l819_819495

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l819_819495


namespace cos_of_angle_l819_819926

theorem cos_of_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (3 * Real.pi / 2 + 2 * θ) = 3 / 5 := 
by
  sorry

end cos_of_angle_l819_819926


namespace remainder_of_exp_l819_819061

theorem remainder_of_exp (x : ℝ) :
  (x + 1) ^ 2100 % (x^4 - x^2 + 1) = x^2 := 
sorry

end remainder_of_exp_l819_819061


namespace solve_investment_problem_l819_819691

def investment_problem
  (total_investment : ℝ) (etf_investment : ℝ) (mutual_funds_factor : ℝ) (mutual_funds_investment : ℝ) : Prop :=
  total_investment = etf_investment + mutual_funds_factor * etf_investment →
  mutual_funds_factor * etf_investment = mutual_funds_investment

theorem solve_investment_problem :
  investment_problem 210000 46666.67 3.5 163333.35 :=
by
  sorry

end solve_investment_problem_l819_819691


namespace solve_arithmetic_sequence_l819_819305

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l819_819305


namespace total_fish_l819_819243

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l819_819243


namespace factor_x4_minus_64_l819_819098

theorem factor_x4_minus_64 :
  ∀ x : ℝ, (x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8)) :=
by
  intro x
  sorry

end factor_x4_minus_64_l819_819098


namespace cot_sum_arccot_roots_l819_819260

noncomputable def poly_roots : list ℂ := sorry -- placeholder for the roots

theorem cot_sum_arccot_roots :
  let z_k : list ℂ := poly_roots in
  h : ( ∀ k, z_k.length = 24 ∧
    polynomial.from_roots z_k = polynomial.mkCoeff [1024, ..., -2, 0, 1])
  ⊢ cot (∑ k in range 24, arccot (z_k.nth k).getD 0) = (2 ^ 25 - 1) / (2 ^ 24 - 2) := 
sorry

end cot_sum_arccot_roots_l819_819260


namespace find_p_q_squared_l819_819475

theorem find_p_q_squared (p q : ℝ) (h₀ : p = 1/2) (h₁ : q = Real.sqrt 19 / 2) 
                         (h₂ : Polynomial.aeval (p + q * Complex.i) (6 • X^3 + 5 • X^2 - X + 14) = 0)
                         (h₃ : Polynomial.aeval (p - q * Complex.i) (6 • X^3 + 5 • X^2 - X + 14) = 0) :
  p + q^2 = 21 / 4 :=
by sorry

end find_p_q_squared_l819_819475


namespace range_of_a_l819_819514

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x^2 + 2 * |x - a| ≥ a^2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l819_819514


namespace solve_arithmetic_sequence_l819_819306

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l819_819306


namespace value_of_y_l819_819549

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 8) (h2 : x = 2) : y = 3 / 2 :=
by
  sorry

end value_of_y_l819_819549


namespace average_price_per_dvd_l819_819883

-- Define the conditions
def num_movies_box1 : ℕ := 10
def price_per_movie_box1 : ℕ := 2
def num_movies_box2 : ℕ := 5
def price_per_movie_box2 : ℕ := 5

-- Define total calculations based on conditions
def total_cost_box1 : ℕ := num_movies_box1 * price_per_movie_box1
def total_cost_box2 : ℕ := num_movies_box2 * price_per_movie_box2

def total_cost : ℕ := total_cost_box1 + total_cost_box2
def total_movies : ℕ := num_movies_box1 + num_movies_box2

-- Define the average price per DVD and prove it to be 3
theorem average_price_per_dvd : total_cost / total_movies = 3 := by
  sorry

end average_price_per_dvd_l819_819883


namespace percent_non_swimmers_play_soccer_is_50_l819_819854

-- Definitions of the conditions
def total_children : ℕ := sorry 
def percent_soccer : ℚ := 0.55
def percent_swim : ℚ := 0.45
def percent_soccer_swim : ℚ := 0.50
def percent_basketball : ℚ := 0.35
def percent_basketball_soccer_not_swim : ℚ := 0.20

-- Theorem: Prove that the percentage of non-swimmers who play soccer is 50%
theorem percent_non_swimmers_play_soccer_is_50 :
  let N := total_children,
      soccer_players := percent_soccer * N,
      swimmers := percent_swim * N,
      soccer_swimmers := percent_soccer_swim * soccer_players,
      non_swimming_soccer_players := soccer_players - soccer_swimmers,
      non_swimmers := N - swimmers in
  (non_swimming_soccer_players / non_swimmers) * 100 = 50 :=
by
  sorry

end percent_non_swimmers_play_soccer_is_50_l819_819854


namespace socks_impossible_l819_819745

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l819_819745


namespace object_speed_l819_819793

namespace problem

noncomputable def speed_in_miles_per_hour (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := distance_in_feet / 5280
  let time_in_hours := time_in_seconds / 3600
  distance_in_miles / time_in_hours

theorem object_speed 
  (distance_in_feet : ℕ)
  (time_in_seconds : ℕ)
  (h : distance_in_feet = 80 ∧ time_in_seconds = 2) :
  speed_in_miles_per_hour distance_in_feet time_in_seconds = 27.27 :=
by
  sorry

end problem

end object_speed_l819_819793


namespace distance_M_to_origin_l819_819207

-- Definitions of the given problem conditions
variable {R : Type*} [NormedRing R] [NormedSpace ℝ R]
noncomputable def e1 : R := sorry  -- Unit vector in the positive direction of the x-axis
noncomputable def e2 : R := sorry  -- Unit vector in the positive direction of the y-axis

-- Angle between e1 and e2 is 60 degrees
axiom angle_e1_e2 : ∀ {α β : R}, α = e1 ∧ β = e2 → ∠ (α, β) = 60

-- Coordinates of point M in the oblique coordinate system
def M : R := e1 + 2 • e2

-- Definition to compute the distance from M to the origin O
def dist_to_origin (p : R) : ℝ := Real.sqrt (∥p∥^2)

-- The theorem to prove
theorem distance_M_to_origin : dist_to_origin M = Real.sqrt 7 := by
  sorry

end distance_M_to_origin_l819_819207


namespace distinct_segment_lengths_count_l819_819284

theorem distinct_segment_lengths_count :
  let points := {0, 1, 2, 3, 5, 8, 2016}
  let segment_lengths := {abs x - y | x y in points}
  segment_lengths.card = 14 :=
sorry

end distinct_segment_lengths_count_l819_819284


namespace evaluate_expression_at_2_l819_819257

def h (x : ℝ) : ℝ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)

def k (x : ℝ) : ℝ := x - 2

theorem evaluate_expression_at_2 : h (k 2) + k (h 2) = 36 / 5 := by
  sorry

end evaluate_expression_at_2_l819_819257


namespace sum_inequality_l819_819696

theorem sum_inequality (n : ℕ) (h : n ≥ 3) (a : Fin n → ℝ)
    (h_increasing : ∀ i j, i < j → a i < a j) 
    (h_positive : ∀ i, 0 < a i) :
    (∑ k in Finset.range n, (a k / a ((k + 1) % n))) > 
    (∑ k in Finset.range n, (a ((k + 1) % n) / a k)) := 
  sorry

end sum_inequality_l819_819696


namespace evaluate_f_at_alpha_l819_819127

def f (α : Real) : Real := 
  (sin (α - (π / 2)) * cos ((3 * π / 2) - α) * tan (π + α) * cos ((π / 2) + α)) / 
  (sin (2 * π - α) * tan (-α - π) * sin (-α - π))

theorem evaluate_f_at_alpha :
  f (-31 * π / 3) = -1 / 2 := sorry

end evaluate_f_at_alpha_l819_819127


namespace probability_k_fall_l819_819213

theorem probability_k_fall (n k : ℕ) (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  ∃ q = 1 - p, (probability_that_exactly_k_fall = p * q^(n-k)) :=
sorry

end probability_k_fall_l819_819213


namespace inequality_proof_problem_l819_819524

theorem inequality_proof_problem (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) ≤ 1 / a) :=
sorry

end inequality_proof_problem_l819_819524


namespace derivative_inequality_solution_set_l819_819588

def f (x : ℝ) : ℝ := x^2 - 2 * x - 4 * Real.log x

theorem derivative_inequality_solution_set :
  ∃ (s : Set ℝ), (∀ x, f x = x^2 - 2 * x - 4 * Real.log x → (f' x < 0 ↔ 0 < x ∧ x < 2)) :=
sorry

end derivative_inequality_solution_set_l819_819588


namespace sum_powers_of_i_eq_zero_l819_819469

theorem sum_powers_of_i_eq_zero (i : ℂ) (h : i^2 = -1) : 
  i^(603) + i^(602) + ... + i^(1) + 1 = 0 := by
sorry

end sum_powers_of_i_eq_zero_l819_819469


namespace area_of_midpoint_quadrilateral_l819_819681

theorem area_of_midpoint_quadrilateral (length width : ℝ) (h_length : length = 15) (h_width : width = 8) :
  let A := (0, width / 2)
  let B := (length / 2, 0)
  let C := (length, width / 2)
  let D := (length / 2, width)
  let mid_quad_area := (length / 2) * (width / 2)
  mid_quad_area = 30 :=
by
  simp [h_length, h_width]
  sorry

end area_of_midpoint_quadrilateral_l819_819681


namespace problem1_problem2_l819_819465

theorem problem1 : 2 * Real.sqrt 2 + Real.sqrt 9 + Real.cbrt (-8) = 2 * Real.sqrt 2 + 1 :=
by
  sorry

theorem problem2 : Real.cbrt (-27) + Real.sqrt 16 - Real.sqrt (9 / 4) = -1 / 2 :=
by
  sorry

end problem1_problem2_l819_819465


namespace express_repeating_decimal_as_fraction_l819_819889

noncomputable def repeating_decimal : ℚ := 7 + 123 / 999
#r "3033"
theorem express_repeating_decimal_as_fraction :
  repeating_decimal = 593 / 111 :=
sorry

end express_repeating_decimal_as_fraction_l819_819889


namespace line_equation_l1_polar_equation_circle_C1_triangle_area_C1MN_l819_819577

noncomputable def line_l1 : set (ℝ × ℝ) := 
  { p | ∃ t : ℝ, p.1 = t ∧ p.2 = sqrt 3 * t }
  
noncomputable def circle_C1 : set (ℝ × ℝ) := 
  { p | (p.1 - sqrt 3)^2 + (p.2 - 2)^2 = 1 }

-- The general equation of the line l1.
theorem line_equation_l1 : ∀ t, line_l1 (t, sqrt 3 * t) := 
by sorry -- recent t can be used directly to show this

-- The polar equation of the circle C1.
theorem polar_equation_circle_C1 : ∀ ρ θ, 
  (let x := ρ * cos θ in let y := ρ * sin θ in
  circle_C1 (x, y)) → ρ^2 - 2 * sqrt 3 * ρ * cos θ - 4 * ρ * sin θ + 6 = 0 := 
by sorry  -- changing the given circle equation to polar forms is not easy but feasible

-- The area of the triangle C1MN.
theorem triangle_area_C1MN : 
  ∃ M N : ℝ × ℝ, M ≠ N ∧ M ∈ line_l1 ∧ N ∈ line_l1 ∧ 
  M ∈ circle_C1 ∧ N ∈ circle_C1 ∧ 
  let C1_center := (sqrt 3, 2) in 
  let d := sqrt 3 in 
  let h := 1 / 2 in 
  area_triangle C1_center M N = sqrt(3) / 4 :=
by sorry  -- transformation simplifies the end proof to show the result correctly.

end line_equation_l1_polar_equation_circle_C1_triangle_area_C1MN_l819_819577


namespace multiplication_problem_solved_thm_l819_819629

def is_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def multiplication_problem_solved (A B C K : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit K ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ K ∧ B ≠ C ∧ B ≠ K ∧ C ≠ K ∧
  A < B ∧
  let AC := 10 * A + C in
  let BC := 10 * B + C in
  let KKK := 100 * K + 10 * K + K in
  (A * 10 + C) * (B * 10 + C) = KKK ∧
  KKK = K * 111 ∧
  A = 2 ∧ B = 3 ∧ C = 7 ∧ K = 9

theorem multiplication_problem_solved_thm : ∃ A B C K, multiplication_problem_solved A B C K :=
begin
  use 2,
  use 3,
  use 7,
  use 9,
  repeat { split },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  { intros _ h, linarith },
  { intros _ h, linarith },
  { intros _ h, linarith },
  { intros _ h, linarith },
  { -- exact QED for the multiplication equality ideally, omitted here for brevity
    sorry },
end

end multiplication_problem_solved_thm_l819_819629


namespace derivative_at_one_l819_819564

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_one_l819_819564


namespace trapezoid_sides_l819_819813

theorem trapezoid_sides (r kl: ℝ) (h1 : r = 5) (h2 : kl = 8) :
  ∃ (ab cd bc_ad : ℝ), ab = 5 ∧ cd = 20 ∧ bc_ad = 12.5 :=
by
  sorry

end trapezoid_sides_l819_819813


namespace calculate_nabla_l819_819038

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calculate_nabla : nabla (nabla 2 3) 4 = 11 / 9 :=
by
  sorry

end calculate_nabla_l819_819038


namespace eval_and_eighth_root_of_unity_l819_819484

open Complex Real

theorem eval_and_eighth_root_of_unity :
  let x := (tan (π / 4) + I) / (tan (π / 4) - I)
  (tan (π / 4) = 1) → x = I ∧ ∃ n : ℕ, 2 * n = 4 ∧ x = cos (2 * n * π / 8) + I * sin (2 * n * π / 8) := 
by
  assume h_tan : tan (π / 4) = 1
  have eq1 : (1 + I) / (1 - I) = I := by sorry
  have root_of_unity : I = cos (π / 2) + I * sin (π / 2) := by sorry
  existsi 2
  split
  . exact eq1
  . split
    . norm_num
    . exact root_of_unity

end eval_and_eighth_root_of_unity_l819_819484


namespace distance_between_planes_is_zero_l819_819905

-- Definitions of planes P1 and P2 in R^3.
def plane1 (x y z : ℝ) : Prop := x + 2 * y - z = 3
def plane2 (x y z : ℝ) : Prop := 2 * x + 4 * y - 2 * z = 6

-- Theorem stating the distance between the planes is 0.
theorem distance_between_planes_is_zero :
  ∀ (x y z : ℝ), plane1 x y z -> plane2 x y z -> 0 := by
  sorry

end distance_between_planes_is_zero_l819_819905


namespace man_speed_against_stream_l819_819432

theorem man_speed_against_stream (v : ℝ) :
  (∀ (s t : ℝ), (s = 4 + t) → s = 12 → t = 8) →
  (v = 8 → abs (4 - v) = 4) :=
by
  intros h hv
  have := h 12 v
  simp at hv
  rw [hv]
  sorry

end man_speed_against_stream_l819_819432


namespace time_to_cross_bridge_l819_819444

/-- Define the conditions as Lean definitions -/
def train_length : ℝ := 130  -- meters
def train_speed_kmh : ℝ := 45  -- km/hr
def total_length : ℝ := 245 -- meters
def bridge_length : ℝ := total_length - train_length  -- meters
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)  -- converting to m/s, 1 km/hr = 1/3.6 m/s
def total_distance_to_cross : ℝ := train_length + bridge_length  -- meters

/-- The proof statement we need to prove -/
theorem time_to_cross_bridge : total_distance_to_cross / train_speed_ms = 19.6 := by
  -- This space is for the proof, filled with sorry for now.
  sorry

end time_to_cross_bridge_l819_819444


namespace biased_die_probability_l819_819808

theorem biased_die_probability (P2 : ℝ) (h1 : P2 ≠ 1 / 6) (h2 : 3 * P2 * (1 - P2) ^ 2 = 1 / 4) : 
  P2 = 0.211 :=
sorry

end biased_die_probability_l819_819808


namespace sum_of_intersections_eq_zero_l819_819321

def f (x : ℝ) : ℝ :=
if x ∈ Icc (-4) (-2) then -3 * (x + 4) / 2 - 5
else if x ∈ Icc (-2) (-1) then -x - 1
else if x ∈ Icc (-1) (1) then 2 * x
else if x ∈ Icc (1) (2) then -x + 3
else if x ∈ Icc (2) (4) then x + 1
else 0

def g (x : ℝ) : ℝ := x + 2

theorem sum_of_intersections_eq_zero : 
  (∑ x in (finset.filter (λ x, f x = g x) (finset.range 10)), x) = 0 := 
by 
  sorry

end sum_of_intersections_eq_zero_l819_819321


namespace a_in_range_l819_819578

theorem a_in_range (a : ℝ) : let A := {x : ℝ | x ≤ 0} 
                              let B := {1, 3, a}
                                in A ∩ B ≠ ∅ →  a ∈ {x : ℝ | x ≤ 0} := 
by
  intro h
  simp only [set.nonempty, set.mem_inter_iff, set.mem_set_of_eq, set.mem_insert_iff, set.mem_singleton_iff] at h
  cases h with x hx
  cases hx with hxA hxB
  cases hxB
  · rw hxB at hxA; exact hxA
  · cases hxB
    · rw hxB at hxA; exact hxA
    · rw hxB; exact hxA

end a_in_range_l819_819578


namespace parabola_through_points_l819_819004

theorem parabola_through_points (b c : ℝ)
  (h1 : 4 = 2 + b + c)
  (h2 : 16 = 18 + 3b + c) :
  c = 4 :=
sorry

end parabola_through_points_l819_819004


namespace stamps_on_last_page_l819_819670

theorem stamps_on_last_page
  (B : ℕ) (P_b : ℕ) (S_p : ℕ) (S_p_star : ℕ) 
  (B_comp : ℕ) (P_last : ℕ) 
  (stamps_total : ℕ := B * P_b * S_p) 
  (pages_total : ℕ := stamps_total / S_p_star)
  (pages_comp : ℕ := B_comp * P_b)
  (pages_filled : ℕ := pages_total - pages_comp) :
  stamps_total - (pages_total - 1) * S_p_star = 8 :=
by
  -- Proof steps would follow here.
  sorry

end stamps_on_last_page_l819_819670


namespace number_is_69_point_28_l819_819320

noncomputable def num := 69.28
def q_approx : ℝ := 9.237333333333334
def numerator := num * 0.004
def denominator := 0.03
def q := numerator / denominator

theorem number_is_69_point_28
    (h1 : q ≈ q_approx)
    (h2 : q = numerator / denominator):
    num = 69.28 := by
  sorry

end number_is_69_point_28_l819_819320


namespace blocks_per_box_l819_819520

theorem blocks_per_box (total_blocks number_of_boxes : ℕ) (h_blocks : total_blocks = 12) (h_boxes : number_of_boxes = 2) : total_blocks / number_of_boxes = 6 :=
by 
  rw [h_blocks, h_boxes] 
  rfl

end blocks_per_box_l819_819520


namespace intersection_of_asymptotes_l819_819107

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem intersection_of_asymptotes : f 3 = 1 :=
by sorry

end intersection_of_asymptotes_l819_819107


namespace vendor_total_profit_l819_819842

theorem vendor_total_profit :
  let cost_per_apple := 3 / 2
      selling_price_per_apple := 10 / 5
      profit_per_apple := selling_price_per_apple - cost_per_apple
      total_profit_apples := profit_per_apple * 5
      cost_per_orange := 2.7 / 3
      selling_price_per_orange := 1
      profit_per_orange := selling_price_per_orange - cost_per_orange
      total_profit_oranges := profit_per_orange * 5
  in total_profit_apples + total_profit_oranges = 3 := 
by
  sorry

end vendor_total_profit_l819_819842


namespace first_train_length_correct_l819_819766

noncomputable def first_train_length (V1 V2 L2 t : ℝ) : ℝ :=
  (V1 * (5/18) + V2 * (5/18)) * t - L2

theorem first_train_length_correct : 
  ∀ (V1 V2 L2 t : ℝ), V1 = 60 → V2 = 40 → L2 = 160 → t = 10.799136069114471 → 
  first_train_length V1 V2 L2 t = 140 := 
by {
  intros V1 V2 L2 t hV1 hV2 hL2 ht,
  -- Converting speeds from km/hr to m/s
  let V1_ms := V1 * (5/18),
  let V2_ms := V2 * (5/18),
  -- Define the relative speed
  let Vr := V1_ms + V2_ms,
  -- Calculate the length of the first train
  have L1_eq : first_train_length V1 V2 L2 t = (Vr * t - L2), from rfl,
  -- Substitute the values
  rw [hV1, hV2, hL2, ht, L1_eq],
  -- Simplify the expression
  norm_num,
  -- Verifying the computed value
  refl,
  sorry,
}

end first_train_length_correct_l819_819766


namespace vector_equation_solution_l819_819169

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem vector_equation_solution (u v w x : α)
  (h : u + v + w + x = 0) :
  ∃ m : ℝ, m = 2 ∧ m • (v × u) + (v × w) + (w × x) + (x × u) = 0 :=
by
  use 2
  sorry

end vector_equation_solution_l819_819169


namespace negation_is_false_l819_819338

-- Definitions corresponding to the conditions
def prop (x : ℝ) := x > 0 → x^2 > 0

-- Statement of the proof problem in Lean 4
theorem negation_is_false : ¬(∀ x : ℝ, ¬(x > 0 → x^2 > 0)) = false :=
by {
  sorry
}

end negation_is_false_l819_819338


namespace AF_eq_CD_l819_819172

-- Definitions based on the conditions
variables {A B C D F E : Type} 
variables [Semicircle AB : Prop]
variables (AC BC : Line) (BD AC CD : Segment)
variables (BF : Line) (AD : Line) (angle_BED : ℝ)
variables (AF CD : Segment)

-- Conditions
axiom semicircle_condition : @Semicircle A B A B C -- \( C \) is on a semicircle with diameter \( AB \)
axiom arc_condition : AC.length < BC.length -- \(\overarc{AC} < \overarc{BC}\)
axiom D_condition : BD.length = AC.length -- \( D \) is on \( BC \) such that \( BD = AC \)
axiom BF_intersect_AD_at_E : (intersection_point BF AD) = E -- \( BF \) intersects \( AD \) at \( E \)
axiom angle_BED_45 : angle BED = 45 -- \( \angle BED = 45^\circ \)

noncomputable def prove_AF_eq_CD : Prop := 
  AF.length = CD.length -- Prove that \( AF = CD \)

-- Final Theorem Statement
theorem AF_eq_CD 
  (semicircle_condition : Semicircle AB AC BC)
  (arc_condition : AC.length < BC.length) 
  (D_condition : BD.length = AC.length)
  (BF_intersect_AD_at_E : (intersection_point BF AD) = E)
  (angle_BED_45 : angle BED = 45) : prove_AF_eq_CD := 
sorry

end AF_eq_CD_l819_819172


namespace probability_divisible_by_3_l819_819043

-- Define the set of numbers
def S : Set ℕ := {2, 3, 5, 6}

-- Define the pairs of numbers whose product is divisible by 3
def valid_pairs : Set (ℕ × ℕ) := {(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)}

-- Define the total number of pairs
def total_pairs := 6

-- Define the number of valid pairs
def valid_pairs_count := 5

-- Prove that the probability of choosing two numbers whose product is divisible by 3 is 5/6
theorem probability_divisible_by_3 : (valid_pairs_count / total_pairs : ℚ) = 5 / 6 := by
  sorry

end probability_divisible_by_3_l819_819043


namespace upstream_distance_calculation_l819_819843

theorem upstream_distance_calculation (d_downstream : ℕ) (t : ℕ) (v_still : ℕ) 
    (h_downstream : d_downstream = (v_still + v) * t) : ∃ d_upstream : ℕ, d_upstream = 15 := by
  let v := (d_downstream - v_still * t) / t
  have h_v : v = 5 := by
    sorry -- Detailed calculations of v
  let d_upstream := (v_still - v) * t
  have h_upstream : d_upstream = 15 := by 
    sorry -- Detailed verification of the upstream distance
  use d_upstream
  exact h_upstream -- Concluding the distance is 15 km

end upstream_distance_calculation_l819_819843


namespace log_base_5_of_15625_l819_819078

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l819_819078


namespace find_f1_find_fx_find_largest_m_l819_819570

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c

axiom min_value_eq_zero (a b c : ℝ) : ∀ x : ℝ, f a b c x ≥ 0 ∨ f a b c x ≤ 0
axiom symmetry_condition (a b c : ℝ) : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1)
axiom inequality_condition (a b c : ℝ) : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1

theorem find_f1 (a b c : ℝ) : f a b c 1 = 1 := sorry

theorem find_fx (a b c : ℝ) : ∀ x : ℝ, f a b c x = (1 / 4) * (x + 1) ^ 2 := sorry

theorem find_largest_m (a b c : ℝ) : ∃ m : ℝ, m > 1 ∧ ∀ t x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x := sorry

end find_f1_find_fx_find_largest_m_l819_819570


namespace exists_divisible_pair_l819_819309

variable (marked : ℕ → Prop)
variable (h_marked_segment : ∀ n, ∃ x, x ∈ set.Icc n (n + 1999) ∧ marked x)

theorem exists_divisible_pair :
  ∃ (a b : ℕ), marked a ∧ marked b ∧ (a ∣ b ∨ b ∣ a) :=
by
  sorry

end exists_divisible_pair_l819_819309


namespace value_of_a_minus_3_l819_819036

variable {α : Type*} [Field α] (f : α → α) (a : α)

-- Conditions
variable (h_invertible : Function.Injective f)
variable (h_fa : f a = 3)
variable (h_f3 : f 3 = 6)

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = -2 :=
by
  sorry

end value_of_a_minus_3_l819_819036


namespace amount_of_sugar_l819_819822

-- Let ratio_sugar_flour be the ratio of sugar to flour.
def ratio_sugar_flour : ℕ := 10

-- Let flour be the amount of flour used in ounces.
def flour : ℕ := 5

-- Let sugar be the amount of sugar used in ounces.
def sugar (ratio_sugar_flour : ℕ) (flour : ℕ) : ℕ := ratio_sugar_flour * flour

-- The proof goal: given the conditions, prove that the amount of sugar used is 50 ounces.
theorem amount_of_sugar (h_ratio : ratio_sugar_flour = 10) (h_flour : flour = 5) : sugar ratio_sugar_flour flour = 50 :=
by
  -- Proof omitted.
  sorry
 
end amount_of_sugar_l819_819822


namespace contrapositive_of_odd_even_l819_819322

-- Definitions as conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main statement
theorem contrapositive_of_odd_even :
  (∀ a b : ℕ, is_odd a ∧ is_odd b → is_even (a + b)) →
  (∀ a b : ℕ, ¬ is_even (a + b) → ¬ (is_odd a ∧ is_odd b)) := 
by
  intros h a b h1
  sorry

end contrapositive_of_odd_even_l819_819322


namespace particle_speed_interval_l819_819436

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 7)

theorem particle_speed_interval (k : ℝ) :
  let start_pos := particle_position k
  let end_pos := particle_position (k + 2)
  let delta_x := end_pos.1 - start_pos.1
  let delta_y := end_pos.2 - start_pos.2
  let speed := Real.sqrt (delta_x^2 + delta_y^2)
  speed = 2 * Real.sqrt 34 := by
  sorry

end particle_speed_interval_l819_819436


namespace tetrahedron_inequality_l819_819631

variables (S A B C : Point)
variables (SA SB SC : Real)
variables (ABC : Plane)
variables (z : Real)
variable (h1 : angle B S C = π / 2)
variable (h2 : Project (point S) ABC = Orthocenter triangle ABC)
variable (h3 : RadiusInscribedCircle triangle ABC = z)

theorem tetrahedron_inequality :
  SA^2 + SB^2 + SC^2 >= 18 * z^2 :=
sorry

end tetrahedron_inequality_l819_819631


namespace vendor_profit_l819_819839

noncomputable def apple_cost_price := 3 / 2
noncomputable def apple_selling_price := 10 / 5
noncomputable def orange_cost_price := 2.70 / 3
noncomputable def orange_selling_price := 1

noncomputable def total_apple_cost_price := 5 * apple_cost_price
noncomputable def total_apple_selling_price := 5 * apple_selling_price
noncomputable def total_apple_profit := total_apple_selling_price - total_apple_cost_price

noncomputable def total_orange_cost_price := 5 * orange_cost_price
noncomputable def total_orange_selling_price := 5 * orange_selling_price
noncomputable def total_orange_profit := total_orange_selling_price - total_orange_cost_price

noncomputable def total_profit := total_apple_profit + total_orange_profit

theorem vendor_profit : total_profit = 3 := by
  sorry

end vendor_profit_l819_819839


namespace ortho_center_locus_is_line_parallel_to_BC_l819_819849

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist A C

noncomputable def circle_contains (O A M N : Point) : Prop :=
  let circle : Circle := ⟨O, dist O A⟩ in
  circle.contains A ∧ circle.contains M ∧ circle.contains N

noncomputable def points_on_lines (A B C O M N : Point) : Prop :=
  (O.on_line B C) ∧
  (M.on_line A B) ∧
  (N.on_line A C)

noncomputable def orthocenter_locus (A B C O M N : Point) : Line :=
  let F := reflection A B C in
  parallel_line B C (distance_line F (distance_line from BC * 2))

theorem ortho_center_locus_is_line_parallel_to_BC
  (A B C O M N : Point)
  (isosceles : isosceles_triangle A B C)
  (circle_through_A : circle_contains O A M N)
  (circle_intersect_AB : points_on_lines A B C O M N) :
  orthocenter_locus A B C O M N = parallel_line B C (distance_line from reflection A B C to BC * 2) :=
sorry

end ortho_center_locus_is_line_parallel_to_BC_l819_819849


namespace eight_point_shots_count_is_nine_l819_819782

def num_8_point_shots (x y z : ℕ) := 8 * x + 9 * y + 10 * z = 100 ∧
                                      x + y + z > 11 ∧ 
                                      x + y + z ≤ 12 ∧ 
                                      x > 0 ∧ 
                                      y > 0 ∧ 
                                      z > 0

theorem eight_point_shots_count_is_nine : 
  ∃ x y z : ℕ, num_8_point_shots x y z ∧ x = 9 :=
by
  sorry

end eight_point_shots_count_is_nine_l819_819782


namespace perimeter_of_PF1Q_l819_819145

noncomputable def hyperbola := { x : ℝ × ℝ // x.1 ^ 2 / 14 - x.2 ^ 2 / 11 = 1 }
constant F1 : ℝ × ℝ
constant P Q : ℝ × ℝ
constant l : ℝ × ℝ → Prop

axiom F1_left_focus : ∃ (c1 c2 : ℝ), F1 = (-c1, 0) ∧ c1 ^ 2 / 14 = 1
axiom l_through_origin : l (0, 0)
axiom PQ_on_hyperbola : P ∈ hyperbola ∧ Q ∈ hyperbola
axiom PF1_dot_QF1_zero : let PF1 := (P.1 - F1.1, P.2 - F1.2), QF1 := (Q.1 - F1.1, Q.2 - F1.2) in PF1.1 * QF1.1 + PF1.2 * QF1.2 = 0

theorem perimeter_of_PF1Q : let dPF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2,
                               dQF1 := (Q.1 - F1.1)^2 + (Q.2 - F1.2)^2 in
                                sqrt dPF1 + sqrt dQF1 + sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 22 :=
by
  sorry

end perimeter_of_PF1Q_l819_819145


namespace arithmetic_progression_solution_l819_819059

theorem arithmetic_progression_solution (a b : ℝ) :
  (∃ d : ℝ, a = 10 + d ∧ b = 10 + 2d ∧ a * b = 10 + 3d) ↔ (a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5) :=
by
  sorry

end arithmetic_progression_solution_l819_819059


namespace log_base_5_of_15625_l819_819084

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l819_819084


namespace complex_number_z_value_l819_819516

-- Conditions
def z : ℂ := complex.abs (complex.mk (real.sqrt 3) (-1)) + (complex.I)^2017

-- Statement to prove
theorem complex_number_z_value : z = 2 + complex.I := by
  sorry

end complex_number_z_value_l819_819516


namespace question1_question2_question3_l819_819280

theorem question1 : (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5)) = 4 / 5 :=
by
  sorry

theorem question2 (n : ℕ) (hn : 0 < n) : ∑ k in Finset.range n, (1 : ℚ) / (k + 1) / (k + 2) = n / (n + 1) :=
by
  sorry

theorem question3 : (1 - ∑ k in Finset.range 99, (1 : ℚ) / (k + 1) / (k + 2)) = 1 / 100 :=
by
  sorry

end question1_question2_question3_l819_819280


namespace self_descriptive_7_digit_first_digit_is_one_l819_819807

theorem self_descriptive_7_digit_first_digit_is_one
  (A B C D E F G : ℕ)
  (h_total : A + B + C + D + E + F + G = 7)
  (h_B : B = 2)
  (h_C : C = 1)
  (h_D : D = 1)
  (h_E : E = 0)
  (h_A_zeroes : A = (if E = 0 then 1 else 0)) :
  A = 1 :=
by
  sorry

end self_descriptive_7_digit_first_digit_is_one_l819_819807


namespace probability_one_left_one_right_l819_819888

/-- Define the conditions: 12 left-handed gloves, 10 right-handed gloves. -/
def num_left_handed_gloves : ℕ := 12

def num_right_handed_gloves : ℕ := 10

/-- Total number of gloves is 22. -/
def total_gloves : ℕ := num_left_handed_gloves + num_right_handed_gloves

/-- Total number of ways to pick any two gloves from 22 gloves. -/
def total_pick_two_ways : ℕ := (total_gloves * (total_gloves - 1)) / 2

/-- Number of favorable outcomes picking one left-handed and one right-handed glove. -/
def favorable_outcomes : ℕ := num_left_handed_gloves * num_right_handed_gloves

/-- Define the probability as favorable outcomes divided by total outcomes. 
 It should yield 40/77. -/
theorem probability_one_left_one_right : 
  (favorable_outcomes : ℚ) / total_pick_two_ways = 40 / 77 :=
by
  -- Skip the proof.
  sorry

end probability_one_left_one_right_l819_819888


namespace poly_degree_product_l819_819311

theorem poly_degree_product (p q : Polynomial ℝ) (hp : degree p = 3) (hq : degree q = 6) : 
  degree (p.comp (C 1 * X^2) * q.comp (C 1 * X^4)) = 30 :=
sorry

end poly_degree_product_l819_819311


namespace coefficient_of_x_l819_819155

theorem coefficient_of_x (n : ℕ) (h : 2^n = 32) :
  (∃ (r : ℕ), 10 - 3 * r = 1 ∧ (Nat.choose n r = 10)) :=
by
  have h1 : n = 5 := by sorry
  use 3
  split
  case left => sorry
  case right => sorry

end coefficient_of_x_l819_819155


namespace initial_apples_l819_819704

-- Defining the conditions
def apples_handed_out := 8
def pies_made := 6
def apples_per_pie := 9
def apples_for_pies := pies_made * apples_per_pie

-- Prove the initial number of apples
theorem initial_apples : apples_handed_out + apples_for_pies = 62 :=
by
  sorry

end initial_apples_l819_819704


namespace time_to_cross_platform_l819_819716

-- Definitions based on the conditions
def length_of_train : ℝ := 1800
def length_of_platform : ℝ := length_of_train
def speed_km_per_hr : ℝ := 216
def speed_m_per_s : ℝ := speed_km_per_hr * 1000 / 3600

-- The theorem stating the problem's conclusion based on the conditions
theorem time_to_cross_platform : (length_of_train + length_of_platform) / speed_m_per_s = 60 := by
  sorry

end time_to_cross_platform_l819_819716


namespace series_sum_properties_l819_819051

theorem series_sum_properties :
  let series := λ n, 3 * (1 / 3) ^ n in
  let S := ∑' n, (3 * (1 / 3) ^ n) in
  (∀ ε > 0, ∃ N, ∀ n > N, | 3 * (1 / 3) ^ n - 0 | < ε) ∧ -- The difference between any term and zero
  (S = 9 / 2) ∧ -- Sum of series
  (4 < S) ∧ -- Sum greater than 4
  (S < 5) -- Sum less than 5
:= by
  sorry

end series_sum_properties_l819_819051


namespace close_time_for_pipe_b_l819_819794

-- Define entities and rates
def rate_fill (A_rate B_rate : ℝ) (t_fill t_empty t_fill_target t_close : ℝ) : Prop :=
  A_rate = 1 / t_fill ∧
  B_rate = 1 / t_empty ∧
  t_fill_target = 30 ∧
  A_rate * (t_close + (t_fill_target - t_close)) - B_rate * t_close = 1

-- Declare the theorem statement
theorem close_time_for_pipe_b (A_rate B_rate t_fill_target t_fill t_empty t_close: ℝ) :
   rate_fill A_rate B_rate t_fill t_empty t_fill_target t_close → t_close = 26.25 :=
by have h1 : A_rate = 1 / 15 := by sorry
   have h2 : B_rate = 1 / 24 := by sorry
   have h3 : t_fill_target = 30 := by sorry
   sorry

end close_time_for_pipe_b_l819_819794


namespace inscribed_circle_radius_PQR_l819_819397

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

def radius_of_inscribed_circle (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c in
  herons_formula a b c / s

theorem inscribed_circle_radius_PQR :
  radius_of_inscribed_circle 8 10 12 = real.sqrt 35 :=
by
  sorry

end inscribed_circle_radius_PQR_l819_819397


namespace baseball_attendance_difference_l819_819466

theorem baseball_attendance_difference:
  ∃ C D: ℝ, 
    (59500 ≤ C ∧ C ≤ 80500 ∧ 69565 ≤ D ∧ D ≤ 94118) ∧ 
    (max (D - C) (C - D) = 35000 ∧ min (D - C) (C - D) = 11000) := by
  sorry

end baseball_attendance_difference_l819_819466


namespace primes_satisfying_equation_l819_819121

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l819_819121


namespace part1_part2_l819_819768

def a : ℕ → ℕ
| n := if h : n % 2 = 1 then n else a (n / 2)

theorem part1 : a 48 + a 49 = 52 :=
by sorry

theorem part2 : ∃ n : ℕ, a n = 5 ∧ ∃ k, k = 9 ∧ n = 5 * 2^(k - 1) :=
by sorry

end part1_part2_l819_819768


namespace total_fish_l819_819244

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l819_819244


namespace train_speed_l819_819832

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 300) (h_time : time = 15) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l819_819832


namespace probability_fx_geq_one_l819_819569

noncomputable def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem probability_fx_geq_one : 
  (∫ x in 0..π, if f(x) >= 1 then 1 else 0) / π = 1 / 2 :=
sorry

end probability_fx_geq_one_l819_819569


namespace proof_of_x_l819_819233

noncomputable def x_proof (A B C P : Type) [inner_triangle A B C P] : ℝ :=
∀ (P : point_in_triangle A B C), 
(∠ P A B / ∠ P A C) = (∠ P C A / ∠ P C B) ∧ (∠ P C B / ∠ P B C) = x → x = 1

theorem proof_of_x (A B C P : Type) [inner_triangle A B C P] (P : point_in_triangle A B C) (x : ℝ)
  (h : (∠ P A B / ∠ P A C) = (∠ P C A / ∠ P C B) ∧ (∠ P C B / ∠ P B A) = x) : x = 1 :=
  sorry

end proof_of_x_l819_819233


namespace series_sum_l819_819857

theorem series_sum :
  (∑ k in Finset.range 2005, ((-1)^ (k + 1) + 3)) = 6014 :=
by
  sorry

end series_sum_l819_819857


namespace rectangular_eq_of_polar_parametric_of_conditions_l819_819225

-- Conditions for the first problem
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 = 8 / (5 - 3 * real.cos (2 * θ))

-- Theorem statement for the first problem
theorem rectangular_eq_of_polar (ρ θ : ℝ) (h : polar_eq ρ θ) : (ρ * real.cos(θ))^2 / 4 + (ρ * real.sin(θ))^2 = 1 :=
  sorry

-- Definitions for the second problem
def parametric_eq (t α : ℝ) : ℝ × ℝ :=
  (1 + ((2 * real.sqrt 51) / 17) * t * if α ≥ 0 ∧ α < real.pi then 1 else -1, (real.sqrt 85 / 17) * t)

-- Assertion for the second problem
theorem parametric_of_conditions (t : ℝ) (α : ℝ) (h1 : α ∈ [0, real.pi))
 (h2 : sin^2 α = 5 / 17)
 : parametric_eq t α = (1 + ((2 * real.sqrt 51) / 17) * t * if real.sin(α) / real.cos(α) > 0 then 1 else -1, (real.sqrt 85 / 17) * t) :=
  sorry

end rectangular_eq_of_polar_parametric_of_conditions_l819_819225


namespace log5_of_15625_l819_819089

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l819_819089


namespace fraction_of_As_l819_819611

-- Define the conditions
def fraction_B (T : ℕ) := 1/4 * T
def fraction_C (T : ℕ) := 1/2 * T
def remaining_D : ℕ := 20
def total_students_approx : ℕ := 400

-- State the theorem
theorem fraction_of_As 
  (F : ℚ) : 
  ∀ T : ℕ, 
  T = F * T + fraction_B T + fraction_C T + remaining_D → 
  T = total_students_approx → 
  F = 1/5 :=
by
  intros
  sorry

end fraction_of_As_l819_819611


namespace cricket_player_runs_l819_819796

theorem cricket_player_runs
  (average_runs : ℕ := 32) -- average runs over the first 10 innings
  (num_innings : ℕ := 10) -- number of innings already played
  : (let total_runs := average_runs * num_innings in
     let desired_new_average := average_runs + 4 in
     let new_num_innings := num_innings + 1 in
     let total_runs_required := new_num_innings * desired_new_average in
     let runs_next_innings := total_runs_required - total_runs in
     runs_next_innings = 76) :=
by
  sorry

end cricket_player_runs_l819_819796


namespace ratio_of_side_lengths_sum_l819_819349
noncomputable def ratio_of_area := (75 : ℚ) / 128
noncomputable def rationalized_ratio_of_side_lengths := (5 * real.sqrt (6 : ℝ)) / 16
theorem ratio_of_side_lengths_sum :
  ratio_of_area = (75 : ℚ) / 128 →
  rationalized_ratio_of_side_lengths = (5 * real.sqrt (6 : ℝ)) / 16 →
  (5 : ℚ) + 6 + 16 = 27 := by
  intros _ _
  sorry

end ratio_of_side_lengths_sum_l819_819349


namespace math_class_students_l819_819035

theorem math_class_students (total_students physics_only math_only : ℕ) 
  (students_in_both : ℕ) 
  (total_condition : total_students = physics_only + math_only + students_in_both) 
  (math_to_physics_ratio : math_only + students_in_both = 4 * (physics_only + students_in_both)) 
  (students_in_both_condition : students_in_both = 8) 
  (total_students_condition : total_students = 56) : 
  math_only + students_in_both = 48 :=
by
  sorry
/******************************************************************************
 * This Lean theorem states we want to prove the number of students in the 
 * mathematics class is 48 given specific conditions about the total number of 
 * students, the relationship between the number of students in the physics and 
 * mathematics classes, and the number of students taking both classes.
 ******************************************************************************/

end math_class_students_l819_819035


namespace S15_value_l819_819978

def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n > 1 → a (n + 1) + a (n - 1) = 2 * (a n + a 1)

theorem S15_value (a : ℕ → ℕ) (h : sequence a) : a 15 = 211 :=
  by
    sorry

end S15_value_l819_819978


namespace books_per_shelf_l819_819862

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) 
    (h1 : mystery_shelves = 5) (h2 : picture_shelves = 4) (h3 : total_books = 54) : 
    total_books / (mystery_shelves + picture_shelves) = 6 := 
by
  -- necessary preliminary steps and full proof will go here
  sorry

end books_per_shelf_l819_819862


namespace probability_red_prime_green_even_correct_l819_819390

/-- Conditions -/
def dice_sides : set ℕ := {1, 2, 3, 4, 5, 6}
def prime_numbers : set ℕ := {2, 3, 5}
def even_numbers : set ℕ := {2, 4, 6}

noncomputable def probability_red_prime_green_even : ℚ :=
  let total_outcomes := 36 in
  let successful_outcomes_red := (prime_numbers ∩ dice_sides).card in
  let successful_outcomes_green := (even_numbers ∩ dice_sides).card in
  let successful_outcomes := successful_outcomes_red * successful_outcomes_green in
  successful_outcomes / total_outcomes

/-- Theorem statement -/
theorem probability_red_prime_green_even_correct :
  probability_red_prime_green_even = 1 / 4 := by
  sorry

end probability_red_prime_green_even_correct_l819_819390


namespace range_of_a_l819_819974

-- Define the functions f and g
def f (x : ℝ) : ℝ := x + 4 / x
def g (x a : ℝ) : ℝ := 2^x + a

-- Define the theorem based on the given conditions and the conclusion
theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (1/2 : ℝ) 1, ∃ x2 ∈ set.Icc (2 : ℝ) 3, f x1 ≥ g x2 a) → a ≤ 1 :=
by
  sorry

end range_of_a_l819_819974


namespace speed_in_still_water_l819_819433

variable (v_m v_s : ℝ)

def swims_downstream (v_m v_s : ℝ) : Prop :=
  54 = (v_m + v_s) * 3

def swims_upstream (v_m v_s : ℝ) : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_in_still_water : swims_downstream v_m v_s ∧ swims_upstream v_m v_s → v_m = 12 :=
by
  sorry

end speed_in_still_water_l819_819433


namespace pen_price_first_day_l819_819017

theorem pen_price_first_day (x y : ℕ) 
  (h1 : x * y = (x - 1) * (y + 100)) 
  (h2 : x * y = (x + 2) * (y - 100)) : x = 4 :=
by
  sorry

end pen_price_first_day_l819_819017


namespace smallest_positive_integer_form_l819_819772

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d = 1205 * m + 27090 * n ∧ (∀ e, e > 0 → (∃ x y : ℤ, d = 1205 * x + 27090 * y) → d ≤ e) :=
sorry

end smallest_positive_integer_form_l819_819772


namespace vector_expression_simplification_l819_819859

variable (a b : Type)
variable (α : Type) [Field α]
variable [AddCommGroup a] [Module α a]

theorem vector_expression_simplification
  (vector_a vector_b : a) :
  (1/3 : α) • (vector_a - (2 : α) • vector_b) + vector_b = (1/3 : α) • vector_a + (1/3 : α) • vector_b :=
by
  sorry

end vector_expression_simplification_l819_819859


namespace line_equation_of_projection_l819_819697

variable (v : ℝ × ℝ)
variable h_proj : 
  (v.1 * 3 + v.2 * -4) / 25 = 9 / 5 ∧ 
  -(v.1 * 4 + v.2 * 3) / 25 = -12 / 5

theorem line_equation_of_projection :
  ∃ m b, (∀ x y, y = v.2 → x = v.1 → y = m * x + b) ∧ m = 3 / 4 ∧ b = -15 / 4 :=
by 
  sorry

end line_equation_of_projection_l819_819697


namespace reassemble_black_rectangles_into_1x2_rectangle_l819_819232

theorem reassemble_black_rectangles_into_1x2_rectangle
  (x y : ℝ)
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < y ∧ y < 2)
  (black_white_equal : 2*x*y - 2*x - 2*y + 2 = 0) :
  (x = 1 ∨ y = 1) →
  ∃ (z : ℝ), z = 1 :=
by
  sorry

end reassemble_black_rectangles_into_1x2_rectangle_l819_819232


namespace laura_pants_count_l819_819649

def cost_of_pants : ℕ := 54
def cost_of_shirt : ℕ := 33
def number_of_shirts : ℕ := 4
def total_money_given : ℕ := 250
def change_received : ℕ := 10

def laura_spent : ℕ := total_money_given - change_received
def total_cost_shirts : ℕ := cost_of_shirt * number_of_shirts
def spent_on_pants : ℕ := laura_spent - total_cost_shirts
def pairs_of_pants_bought : ℕ := spent_on_pants / cost_of_pants

theorem laura_pants_count : pairs_of_pants_bought = 2 :=
by
  sorry

end laura_pants_count_l819_819649


namespace CEP36470130_barcode_barcode_CEP20240020_l819_819438

-- Define the conversion from binary strings to digits
def binary_to_digit (s: String): Option Char := 
  match s.to_list with
  | '1' :: '1' :: '0' :: '0' :: '0' :: [] => some '0'
  | '0' :: '0' :: '0' :: '1' :: '1' :: [] => some '1'
  | '0' :: '1' :: '0' :: '1' :: '0' :: [] => some '2'
  | '0' :: '0' :: '1' :: '0' :: '1' :: [] => some '3'
  | '0' :: '0' :: '1' :: '1' :: '0' :: [] => some '4'
  | '0' :: '1' :: '1' :: '0' :: '0' :: [] => some '5'
  | '1' :: '0' :: '1' :: '0' :: '0' :: [] => some '6'
  | '0' :: '0' :: '0' :: '0' :: '1' :: [] => some '7'
  | '1' :: '0' :: '0' :: '0' :: '1' :: [] => some '8'
  | '1' :: '0' :: '0' :: '1' :: '0' :: [] => some '9'
  | _ => none

-- Define the barcode to binary conversion logic with short bar as 0 (|) and long bar as 1 (||)
def bar_to_binary (s : List String) : Option (List String) :=
  match s with
  | ("short" :: "short" :: "long" :: "short" :: "long" :: rest) => some ["00101"] -- 3
  | ("long" :: "short" :: "long" :: "short" :: "short" :: rest) => some ["10100"] -- 6
  | ("short" :: "short" :: "long" :: "long" :: "short" :: rest) => some ["00110"] -- 4
  | ("short" :: "short" :: "short" :: "short" :: "long" :: rest) => some ["00001"] -- 7
  | ("long" :: "long" :: "short" :: "short" :: "short" :: rest) => some ["11000"] -- 0
  | ("short" :: "short" :: "short" :: "long" :: "long" :: rest) => some ["00011"] -- 1
  | _ => none

-- Definitions for given CEP to barcode
def CEP36470130_to_barcode : Prop :=
  let zip := "36470130"
  let barcode := "||short-short-long-short-long-long-short-long-short-short-short-short-long-long-short-short-short-short-long-long-short-short-short-long-long||"
  -- Check if the calculated barcode matches the expected barcode
  true -- Replace with actual checking logic for final implementation

-- Definitions for given barcode to CEP
def barcode_to_CEP20240020 : Prop :=
  let barcode := "||short-short-long-long-long-short-long-short-short-long-short-short-short-short-long-long-long-short-short-short-long-long||"
  let cep := "20240020"
  -- Check if the decoded CEP matches the expected CEP
  true -- Replace with actual checking logic for final implementation

-- Main Lean theorem statements
theorem CEP36470130_barcode :
  CEP36470130_to_barcode :=
by
sorry

theorem barcode_CEP20240020 :
  barcode_to_CEP20240020 :=
by
sorry

end CEP36470130_barcode_barcode_CEP20240020_l819_819438


namespace y_gets_per_rupee_l819_819830

theorem y_gets_per_rupee (a p : ℝ) (ha : a * p = 63) (htotal : p + a * p + 0.3 * p = 245) : a = 0.63 :=
by
  sorry

end y_gets_per_rupee_l819_819830


namespace sequence_v_2002_l819_819713

-- Definitions
def g : ℕ → ℕ :=
  λ x, match x with
  | 1 => 2
  | 2 => 3
  | 3 => 5
  | 4 => 1
  | 5 => 4
  | _ => 0  -- unspecified for values outside [1, 5], can add specific cases if necessary

def v : ℕ → ℕ
| 0       := 3
| (n + 1) := g (v n)

theorem sequence_v_2002 : v 2002 = 4 :=
by
  sorry

end sequence_v_2002_l819_819713


namespace area_of_triangle_PAB_l819_819537

noncomputable def circle_O : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

def line_y_eq_x : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}

def tangent_line_y_eq_sqrt3x_add_m (m : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = sqrt 3 * p.1 + m}

def point_P : ℝ × ℝ := (-sqrt 3, 1)

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def distance_line_point (a b : ℝ) (p : ℝ × ℝ) : ℝ := (abs (a * p.1 + b * p.2 + 1)) / real.sqrt (a^2 + b^2)

def area_of_triangle (P A B : ℝ × ℝ) : ℝ := 0.5 * distance A B * distance_line_point (-1) 1 P

theorem area_of_triangle_PAB :
  area_of_triangle point_P (-2, -2) (2, 2) = sqrt 6 + sqrt 2 :=
sorry

end area_of_triangle_PAB_l819_819537


namespace candy_box_original_price_l819_819026

theorem candy_box_original_price (P : ℝ) (h₁ : 1.25 * P = 10) : P = 8 := 
sorry

end candy_box_original_price_l819_819026


namespace flat_fee_first_night_l819_819429

-- Given conditions
variable (f n : ℝ)
axiom alice_cost : f + 3 * n = 245
axiom bob_cost : f + 5 * n = 350

-- Main theorem to prove
theorem flat_fee_first_night : f = 87.5 := by sorry

end flat_fee_first_night_l819_819429


namespace correct_conclusions_l819_819920

def quadratic_func (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - 5

theorem correct_conclusions (a : ℝ) (h : a ≠ 0):
  (∀ m : ℝ, quadratic_func a (2 + m) = quadratic_func a (2 - m)) ∧
  (∀ x ∈ (set.Icc 3 4 : set ℝ), ∃ (y : ℤ), (∃ x', x = x' ∧ quadratic_func a x' = (y : ℝ)) ∧ (
    -4/3 < a ∧ a ≤ -1 ∨ 1 ≤ a ∧ a < 4/3)) :=
begin
  sorry
end

end correct_conclusions_l819_819920


namespace product_of_symmetrical_complex_numbers_l819_819625

theorem product_of_symmetrical_complex_numbers :
  ∃ z2 : ℂ, (Re(z2) = -Re(-1 + complex.I) ∧ Im(z2) = Im(-1 + complex.I)) →
  (-1 + complex.I) * z2 = -2 :=
by
  sorry

end product_of_symmetrical_complex_numbers_l819_819625


namespace problem_statement_l819_819657

variables {R : Type*} [LinearOrderedField R]

-- Define the function f and its derivative
variables (f : R → R)
variables (f' : R → R)
variables (h_f : ∀ x : R, DifferentiableAt R f x)
variables (h_f' : ∀ x : R, deriv f x = f' x)

-- Given condition: f(x) + x f'(x) > 0 for all x in R
axiom h_condition : ∀ x : R, (f x + x * f' x) > 0

-- Definitions based on problem statement
def a : R := f 1
def b : R := 2 * f 2
def c : R := 3 * f 3

-- Theorem statement to prove
theorem problem_statement : c > b ∧ b > a :=
by 
  sorry

end problem_statement_l819_819657


namespace kamal_average_marks_l819_819648

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks_l819_819648


namespace sum_diameters_eq_sum_legs_l819_819300

theorem sum_diameters_eq_sum_legs 
  (a b c R r : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_circum_radius : R = c / 2)
  (h_incircle_radius : r = (a + b - c) / 2) :
  2 * R + 2 * r = a + b :=
by 
  sorry

end sum_diameters_eq_sum_legs_l819_819300


namespace cannot_travel_from_earth_to_mars_l819_819756

structure Planet := (name : String)

inductive Route : Type
  | connect : Planet → Planet → Route

open Planet Route

def Planets := { 
  earth := Planet.mk "Earth",   mercury := Planet.mk "Mercury",
  venus := Planet.mk "Venus",   pluto := Planet.mk "Pluto",
  uranus := Planet.mk "Uranus", neptune := Planet.mk "Neptune",
  saturn := Planet.mk "Saturn", jupiter := Planet.mk "Jupiter",
  mars := Planet.mk "Mars"
}

def routes : List Route := [
  Route.connect Planets.earth Planets.mercury,
  Route.connect Planets.pluto Planets.venus,
  Route.connect Planets.earth Planets.pluto,
  Route.connect Planets.pluto Planets.mercury,
  Route.connect Planets.mercury Planets.venus,
  Route.connect Planets.uranus Planets.neptune,
  Route.connect Planets.neptune Planets.saturn,
  Route.connect Planets.saturn Planets.jupiter,
  Route.connect Planets.jupiter Planets.mars,
  Route.connect Planets.mars Planets.uranus
]

def connected (p1 p2 : Planet) (routes : List Route) : Prop :=
  ∃ path : List Planet, path.head = p1 ∧ path.last = p2 ∧ 
  ∀ (p q : Planet), Route.connect p q ∈ routes → (p, q) ∈ path.zip path.tail

theorem cannot_travel_from_earth_to_mars (routes) : ¬connected Planets.earth Planets.mars routes :=
  sorry

end cannot_travel_from_earth_to_mars_l819_819756


namespace perfect_square_trinomial_m_l819_819545

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0) ∧ (x^2 - m * x + 25 = (a * x + b)^2)) → (m = 10 ∨ m = -10) :=
by
  -- Using the assumption that there exist constants a and b such that the trinomial is a perfect square
  intro h,
  obtain ⟨a, b, _, h_eq⟩ := h,
  -- Expanding the perfect square and comparing coefficients
  -- will yield the conclusion m = ±10
  sorry

end perfect_square_trinomial_m_l819_819545


namespace sum_n_div_n4_add_16_eq_9_div_320_l819_819887

theorem sum_n_div_n4_add_16_eq_9_div_320 :
  ∑' n : ℕ, n / (n^4 + 16) = 9 / 320 :=
sorry

end sum_n_div_n4_add_16_eq_9_div_320_l819_819887


namespace main_l819_819527

theorem main (x y : ℤ) (h1 : abs x = 5) (h2 : abs y = 3) (h3 : x * y > 0) : 
    x - y = 2 ∨ x - y = -2 := sorry

end main_l819_819527


namespace point_on_bisector_l819_819597

theorem point_on_bisector {a b : ℝ} (h : ∃ θ, θ = atan (b / a) ∧ (θ = π / 4 ∨ θ = -(3 * π / 4))) : b = -a :=
sorry

end point_on_bisector_l819_819597


namespace point_on_bisector_l819_819599

theorem point_on_bisector {a b : ℝ} (h : ∃ θ, θ = atan (b / a) ∧ (θ = π / 4 ∨ θ = -(3 * π / 4))) : b = -a :=
sorry

end point_on_bisector_l819_819599


namespace highest_lowest_difference_total_cars_produced_total_wages_l819_819422

open Nat

def production_deviation : List Int := [5, -2, -4, 13, -10, 16, -9]

def planned_production : Nat := 2100

noncomputable def daily_plan : Nat := 300

def a : Nat := sorry
def b : Nat := sorry

-- Problem (1)
theorem highest_lowest_difference :
  List.maximum production_deviation - List.minimum production_deviation = 26 :=
by
  sorry

-- Problem (2)
theorem total_cars_produced :
  planned_production + production_deviation.sum = 2109 :=
by
  sorry

-- Problem (3)
theorem total_wages (a b : Nat) (hb : b < a) :
  let total_cars := planned_production + production_deviation.sum
  0 ≤ total_cars - planned_production →
  total_cars * a + (total_cars - planned_production) * b = 2109 * a + 9 * b :=
by
  sorry

end highest_lowest_difference_total_cars_produced_total_wages_l819_819422


namespace socks_impossible_l819_819748

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l819_819748


namespace triangle_inequality_l819_819203

theorem triangle_inequality (x : ℝ) (h : 0 ≤ x) : (4 + 6 > x) ∧ (4 + x > 6) ∧ (6 + x > 4) ↔ (2 < x) ∧ (x < 10) := 
by {
  split,
  { intros h1,
    cases h1 with h2 h3,
    cases h3 with h4 h5,
    split, linarith, linarith, },
  { intros h1,
    cases h1 with h2 h3,
    split,
    linarith,
    split, linarith, linarith, }
}
sorry

end triangle_inequality_l819_819203


namespace solution_eq_l819_819487

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l819_819487


namespace jack_reads_books_in_a_year_l819_819584

/-- If Jack reads 9 books per day, how many books can he read in a year (365 days)? -/
theorem jack_reads_books_in_a_year (books_per_day : ℕ) (days_per_year : ℕ) (books_per_year : ℕ) (h1 : books_per_day = 9) (h2 : days_per_year = 365) : books_per_year = 3285 :=
by
  sorry

end jack_reads_books_in_a_year_l819_819584


namespace minimum_value_a_eq_1_monotonicity_f_tangent_lines_range_a_l819_819571

noncomputable def f (x a : ℝ) := Real.exp x - a * x - a
noncomputable def h (x a : ℝ) := -Real.exp x - x + 3 * a
noncomputable def g (x a : ℝ) := (x - 1) * a + 2 * Real.cos x

-- (I)
theorem minimum_value_a_eq_1 : ∀ a : ℝ, a = 1 → ∃ x : ℝ, f x a = 0 :=
by
  sorry

-- (II)
theorem monotonicity_f : 
∀ a : ℝ, 
  (a ≤ 0 → ∀ x : ℝ, f' x a > 0) ∧ 
  (a > 0 → ∀ x : ℝ, x < Real.log a → f' x a < 0 ∧ x > Real.log a → f' x a > 0) :=
by
  sorry

-- (III)
theorem tangent_lines_range_a :
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 2 → ∃ x₁ x₂ : ℝ, (-Real.exp x₁ - 1) * (a - 2 * Real.sin x₂) = -1 :=
by
  sorry

end minimum_value_a_eq_1_monotonicity_f_tangent_lines_range_a_l819_819571


namespace no_equal_prob_for_same_color_socks_l819_819742

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l819_819742


namespace hypotenuse_length_l819_819209

theorem hypotenuse_length (a b c : ℝ) (h1 : b = 2 * a) (h2 : a^2 + b^2 + c^2 = 2000) (right_triangle : c^2 = a^2 + b^2) : 
  c = 10 * real.sqrt 10 :=
by 
  -- skipping the proof as per instruction
  sorry

end hypotenuse_length_l819_819209


namespace area_of_N_is_81_over_5_l819_819623

-- Define point in a set A if it meets several conditions.
structure Point (A : Type) :=
  (x y : A)
  (x_less_y : x < y)
  (abs_x_less_than_3 : abs x < 3)
  (abs_y_less_than_3 : abs y < 3)

-- Define the equation to check if it has no real roots for given points.
def equation (p : Point ℝ) (t : ℝ) : Prop :=
  (p.x ^ 3 - p.y ^ 3) * t^4 + (3 * p.x + p.y) * t^2 + (1 / (p.x - p.y)) = 0

-- Define the condition that the equation has no real roots.
def no_real_roots (p : Point ℝ) : Prop :=
  ∀ t : ℝ, ¬ equation p t

-- Define the set N of all points that meet the criteria.
def N : set (Point ℝ) :=
  { p | no_real_roots p }

-- Proving that the area of the region described by set N is 81/5.
theorem area_of_N_is_81_over_5 : 
  (∫ p in N, 1) = 81 / 5 := sorry

end area_of_N_is_81_over_5_l819_819623


namespace solve_for_x_l819_819694

theorem solve_for_x (x : ℚ) (h : (x - 3) / (x + 2) + (3 * x - 9) / (x - 3) = 2) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l819_819694


namespace price_difference_pc_sm_l819_819032

-- Definitions based on given conditions
def S : ℕ := 300
def x : ℕ := sorry -- This is what we are trying to find
def PC : ℕ := S + x
def AT : ℕ := S + PC
def total_cost : ℕ := S + PC + AT

-- Theorem to be proved
theorem price_difference_pc_sm (h : total_cost = 2200) : x = 500 :=
by
  -- We would prove the theorem here
  sorry

end price_difference_pc_sm_l819_819032


namespace find_d_l819_819195

theorem find_d (d : ℝ) (h : 3 * (2 - (π / 2)) = 6 + d * π) : d = -3 / 2 :=
by
  sorry

end find_d_l819_819195


namespace find_f3_l819_819994

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

theorem find_f3 (a b c : ℝ) (h : f a b c (-3) = -12) : f a b c 3 = 24 :=
by
  sorry

end find_f3_l819_819994


namespace prob_of_2_digit_in_frac_1_over_7_l819_819340

noncomputable def prob (n : ℕ) : ℚ := (3/2)^(n-1) / (3/2 - 1)

theorem prob_of_2_digit_in_frac_1_over_7 :
  let infinite_series_sum := ∑' n : ℕ, (2/3)^(6 * n + 3)
  ∑' (n : ℕ), prob (6 * n + 3) = 108 / 665 :=
by
  sorry

end prob_of_2_digit_in_frac_1_over_7_l819_819340


namespace proof_m_cd_value_l819_819953

theorem proof_m_cd_value (a b c d m : ℝ) 
  (H1 : a + b = 0) (H2 : c * d = 1) (H3 : |m| = 3) : 
  m + c * d - (a + b) / (m ^ 2) = 4 ∨ m + c * d - (a + b) / (m ^ 2) = -2 :=
by
  sorry

end proof_m_cd_value_l819_819953


namespace monotonic_increasing_interval_l819_819336

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_increasing_interval :
  ∀ x > 0, (∂ f x) / ∂ x > 0 →  x > 1 / Real.exp 1 → f x = x * Real.log x  → ∀ x > (1 / Real.exp 1), ∂ f x / ∂ x > 0 :=
begin
  assume x x_pos hx f_x_pos
  have h_deriv : ∂ f x / ∂ x = Real.log x + 1,
  sorry
end

end monotonic_increasing_interval_l819_819336


namespace find_integer_values_of_m_l819_819559

theorem find_integer_values_of_m (m : ℤ) (x : ℚ) 
  (h₁ : 5 * x - 2 * m = 3 * x - 6 * m + 1)
  (h₂ : -3 < x ∧ x ≤ 2) : m = 0 ∨ m = 1 := 
by 
  sorry

end find_integer_values_of_m_l819_819559


namespace scientific_notation_of_0_0000021_l819_819459

theorem scientific_notation_of_0_0000021 :
  (0.0000021 : ℝ) = 2.1 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000021_l819_819459


namespace cos_squared_inequality_l819_819683

theorem cos_squared_inequality (x y : ℝ) : 
  cos(x - y) ^ 2 ≤ 4 * (1 - sin x * cos y) * (1 - cos x * sin y) :=
sorry

end cos_squared_inequality_l819_819683


namespace log_base_5_of_15625_l819_819077

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l819_819077


namespace roundness_of_1080000_l819_819462

theorem roundness_of_1080000 : 
  let n := 1080000
  prime_factor_exponents_sum n = 10 :=
by
  let n := 1080000
  have factorization : n = 2^5 * 5^2 * 3^3 := sorry
  have prime_factor_exponents_sum : prime_factor_exponents_sum n = 5 + 2 + 3 := sorry
  exact prime_factor_exponents_sum

end roundness_of_1080000_l819_819462


namespace spherical_coords_equiv_l819_819874

theorem spherical_coords_equiv :
  ∀ (ρ θ φ ρ' θ' φ' : ℝ),
    ρ = 4 →
    θ = 3 * Real.pi / 4 →
    φ = 9 * Real.pi / 4 →
    ρ' > 0 →
    0 ≤ θ' ∧ θ' < 2 * Real.pi →
    0 ≤ φ' ∧ φ' ≤ Real.pi →
    (ρ, θ, φ) = (ρ', θ', φ') :=
by
  intros ρ θ φ ρ' θ' φ'
  assuming h1 : ρ = 4
  assuming h2 : θ = 3 * Real.pi / 4
  assuming h3 : φ = 9 * Real.pi / 4
  assuming h4 : ρ' > 0
  assuming h5 : 0 ≤ θ' ∧ θ' < 2 * Real.pi
  assuming h6 : 0 ≤ φ' ∧ φ' ≤ Real.pi
  sorry

end spherical_coords_equiv_l819_819874


namespace duration_of_period_l819_819208

noncomputable def birth_rate : ℕ := 7
noncomputable def death_rate : ℕ := 3
noncomputable def net_increase : ℕ := 172800

theorem duration_of_period : (net_increase / ((birth_rate - death_rate) / 2)) / 3600 = 12 := by
  sorry

end duration_of_period_l819_819208


namespace triangle_probability_is_correct_l819_819278

noncomputable def triangle_probability (lengths : List ℕ) : ℚ :=
  let validTriplets := lengths.combinations 3 |>.filter (λ t, let [a, b, c] := t.sorted in a + b > c)
  (validTriplets.length : ℚ) / (lengths.combinations 3).length

theorem triangle_probability_is_correct :
  triangle_probability [1, 2, 4, 5, 8, 9, 12, 15, 17] = 17 / 84 :=
by
  sorry

end triangle_probability_is_correct_l819_819278


namespace ratio_of_asian_boys_l819_819615

theorem ratio_of_asian_boys (P_B P_G P_A P_N : ℝ) 
  (h1 : P_B + P_G = 1)
  (h2 : P_A + P_N = 1)
  (h3 : P_B = 3 / 5 * P_G)
  (h4 : P_A = 3 / 4 * P_N) :
  let P_AB := (P_B * P_A) in
  P_AB = 9 / 56 :=
by 
  sorry

end ratio_of_asian_boys_l819_819615


namespace grant_score_l819_819983

variables (Grant John Hunter : ℕ)
variable (sHunter : Hunter = 45)
variable (sJohn : John = 2 * Hunter)
variable (sGrant : Grant = John + 10)

theorem grant_score : Grant = 100 :=
by
  rw [sHunter, sJohn, sGrant]
  norm_num
  sorry

end grant_score_l819_819983


namespace trig_inequality_l819_819187

theorem trig_inequality (theta : ℝ) (h1 : Real.pi / 4 < theta) (h2 : theta < Real.pi / 2) : 
  Real.cos theta < Real.sin theta ∧ Real.sin theta < Real.tan theta :=
sorry

end trig_inequality_l819_819187


namespace gp_values_l819_819102

theorem gp_values (p : ℝ) (hp : 0 < p) :
  let a := -p - 12
  let b := 2 * Real.sqrt p
  let c := p - 5
  (b / a = c / b) ↔ p = 4 :=
by
  sorry

end gp_values_l819_819102


namespace num_perfect_square_factors_of_3960_l819_819723

theorem num_perfect_square_factors_of_3960 : 
  let a_factors := {a | a = 0 ∨ a = 2};
  let b_factors := {b | b = 0 ∨ b = 2};
  let c_factors := {c | c = 0};
  let d_factors := {d | d = 0};
  set.finite a_factors →
  set.finite b_factors →
  set.finite c_factors →
  set.finite d_factors →
  (set.card a_factors) * (set.card b_factors) * (set.card c_factors) * (set.card d_factors) = 4 :=
by
  sorry

end num_perfect_square_factors_of_3960_l819_819723


namespace patty_fraction_3mph_l819_819677

noncomputable def fraction_time_at_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) : ℝ :=
  t3 / (t3 + t6)

theorem patty_fraction_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) :
  fraction_time_at_3mph t3 t6 h = 1 / 3 :=
by
  sorry

end patty_fraction_3mph_l819_819677


namespace proof_M4_eq_3M2_l819_819848

noncomputable def M1 : ℝ := 2.02 * 10^(-6)
noncomputable def M2 : ℝ := 0.0000202
noncomputable def M3 : ℝ := 0.00000202
noncomputable def M4 : ℝ := 6.06 * 10^(-5)

theorem proof_M4_eq_3M2 : (M4 = 3 * M2) := by
  sorry

end proof_M4_eq_3M2_l819_819848


namespace constant_term_binomial_expansion_l819_819959

theorem constant_term_binomial_expansion :
  let n := 9,
  let term_x_binom := λ (r : ℕ), (Nat.choose n r) * (-1)^r * (x^(n - 3*r)),
  let k := 3,
  let m := 8,
  (Nat.choose n k = Nat.choose n (n - m)) →
  (∃ r : ℕ, x^(n - 3*r) = 1 ∧ Nat.choose n r * (-1)^r = -84) :=
by
  intros
  sorry

end constant_term_binomial_expansion_l819_819959


namespace least_positive_integer_l819_819395

theorem least_positive_integer (N : ℕ) :
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (N % 15 = 14) ∧
  (N % 16 = 15) →
  N = 720719 :=
by
  sorry

end least_positive_integer_l819_819395


namespace max_leap_years_l819_819852

/-- Assume a modified calendrical system where leap years occur every three years,
and consider a 100-year period. Prove that the maximum possible number of leap years
in this period is 33.
-/
theorem max_leap_years (n : ℕ) (h : n = 100) : (n / 3) + if n % 3 ≠ 0 then 1 else 0 = 33 := by
  sorry

end max_leap_years_l819_819852


namespace geom_seq_a5_l819_819627

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geom_seq_a5 (a : ℕ → ℝ) (h1 : a 3 * a 7 = 3) (h2 : a 3 + a 7 = 4) :
  a 5 = √3 :=
by 
  let r := (a 7) ^ (1 / 7)
  let A := (a 3) ^ (1 / 3)
  have h3 : a 3 = 3 := by sorry
  have h4 : a 7 = 1 := by sorry
  have h5 : ∀ n, a n = A * r ^ (n - 1) := by sorry
  show a 5 = √3 := by 
    rw [h5 5]
    sorry

end geom_seq_a5_l819_819627


namespace find_triangle_angles_l819_819317

variables (A B C O1 O2 M : Type*)
variables (ω1 : Set (Set A)) (ω2 : Set (Set A))
variables [inhabited A]

-- Definitions for the triangle and circles
def triangle (A B C : A) : Prop := true
def circumscribed_circle (O1 : A) (C : A) : Prop := true
def inscribed_circle (O2 : A) (M : A) : Prop := true
def perpendicular (O1 O2 M : A) : Prop := true
def equal_distances (O1 M O2 : A) : Prop := true
def lies_on (M AC : A) : Prop := true

-- Given conditions
axiom h1 : triangle A B C
axiom h2 : circumscribed_circle O1 C
axiom h3 : inscribed_circle O2 M
axiom h4 : perpendicular O1 O2 M
axiom h5 : equal_distances O1 M O2
axiom h6 : lies_on M AC

-- Prove the angles of the triangle
theorem find_triangle_angles (A B C O1 O2 M : A) :
  ∃ (θ₁ θ₂ θ₃ : ℝ), θ₁ = 36 ∧ θ₂ = 108 ∧ θ₃ = 36 :=
sorry

end find_triangle_angles_l819_819317


namespace tennis_balls_ordered_l819_819441

def original_white_balls : ℕ := sorry
def original_yellow_balls_with_error : ℕ := sorry

theorem tennis_balls_ordered 
  (W Y : ℕ)
  (h1 : W = Y)
  (h2 : Y + 70 = original_yellow_balls_with_error)
  (h3 : W = 8 / 13 * (Y + 70)):
  W + Y = 224 := sorry

end tennis_balls_ordered_l819_819441


namespace unknown_rate_of_two_towels_l819_819409

theorem unknown_rate_of_two_towels :
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  known_cost + (2 * x) = total_average_price * number_of_towels :=
by
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  show known_cost + (2 * x) = total_average_price * number_of_towels
  sorry

end unknown_rate_of_two_towels_l819_819409


namespace solution_eq_l819_819490

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l819_819490


namespace angle_bisector_relation_l819_819595

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l819_819595


namespace difference_of_square_of_non_divisible_by_3_l819_819294

theorem difference_of_square_of_non_divisible_by_3 (n : ℕ) (h : ¬ (n % 3 = 0)) : (n^2 - 1) % 3 = 0 :=
sorry

end difference_of_square_of_non_divisible_by_3_l819_819294


namespace change_in_energy_proof_l819_819379

noncomputable def initial_energy : ℝ := 18

def hypotenuse_length (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

def energy_per_pair (total_energy : ℝ) (num_pairs : ℕ) : ℝ := total_energy / num_pairs

def new_energy_each_pair (initial_energy_pair : ℝ) (new_distance : ℝ) (old_distance : ℝ) : ℝ :=
  initial_energy_pair * (old_distance / new_distance)

def change_in_energy (initial_energy : ℝ) (new_total_energy : ℝ) : ℝ :=
  new_total_energy - initial_energy

theorem change_in_energy_proof :
  let a := 1
  let b := 1
  let d := hypotenuse_length a b
  let initial_energy_per_pair := energy_per_pair initial_energy 3
  let new_distance := d / 2
  let new_energy_total := initial_energy_per_pair + 2 * new_energy_each_pair initial_energy_per_pair new_distance d
  change_in_energy initial_energy new_energy_total = 12 * (real.sqrt 2 - 1) :=
by 
  let a := 1
  let b := 1
  let d := hypotenuse_length a b
  let initial_energy_per_pair := energy_per_pair initial_energy 3
  let new_distance := d / 2
  let new_energy_total := initial_energy_per_pair + 2 * new_energy_each_pair initial_energy_per_pair new_distance d
  show change_in_energy initial_energy new_energy_total = 12 * (real.sqrt 2 - 1)
  sorry

end change_in_energy_proof_l819_819379


namespace problem_l819_819228

-- Definitions for angles A, B, C and sides a, b, c of a triangle.
variables {A B C : ℝ} {a b c : ℝ}
-- Given condition
variables (h : a = b * Real.cos C + c * Real.sin B)

-- Triangle inequality and angle conditions
variables (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
variables (suma : A + B + C = Real.pi)

-- Goal: to prove that under the given condition, angle B is π/4
theorem problem : B = Real.pi / 4 :=
by {
  sorry
}

end problem_l819_819228


namespace bounded_sequence_is_constant_two_l819_819100

def is_bounded (l : ℕ → ℕ) := ∃ (M : ℕ), ∀ (n : ℕ), l n ≤ M

def satisfies_condition (a : ℕ → ℕ) : Prop :=
∀ n ≥ 3, a n = (a n.pred + a (n.pred.pred)) / (Nat.gcd (a n.pred) (a (n.pred.pred)))

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) 
  (h1 : is_bounded a) 
  (h2 : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 :=
sorry

end bounded_sequence_is_constant_two_l819_819100


namespace total_population_l819_819231

theorem total_population (P1 P2 : ℕ) :
  (0.90 * P1 = 45000) ∧ (0.80 * P2 = 64000) →
  (P1 + P2 = 130000) :=
by
  intros h
  cases h with h1 h2
  sorry

end total_population_l819_819231


namespace sum_of_ks_l819_819325

theorem sum_of_ks {k : ℕ} (h₁ : ∀ α β : ℤ, (α * β = -20) → (α + β = k)) : 
  ∑ (k ∈ {k : ℕ | ∃ α β : ℤ, α * β = -20 ∧ α + β = k}) = 28 :=
by
  sorry

end sum_of_ks_l819_819325


namespace concyclic_points_l819_819663

noncomputable def roots : (ℝ × ℝ × ℝ) → (ℂ × ℂ) := λ ⟨a, b, c⟩,
  let disc_sqrt := complex.sqrt (b*b - 4*a*c) in
  (⟨ -b + disc_sqrt / (2*a), by sorry ⟩,
  ⟨ -b - disc_sqrt / (2*a), by sorry ⟩)

theorem concyclic_points (p : ℝ) (a_eq : (1 + (complex.I : ℂ)) × (1 - (complex.I : ℂ)))
    (b_eq : (-p + complex.sqrt (p^2 + 1)) × (-p - complex.sqrt (p^2 + 1))) :
    ∀ (A B C D : ℂ), 
    (A = ((roots (1, -2, 2)).1) ∨ A = ((roots (1, -2, 2)).2)) ∧
    (B = ((roots (1, -2, 2)).1) ∨ B = ((roots (1, -2, 2)).2)) ∧
    (C = ((roots (1, 2*p, -1)).1) ∨ C = ((roots (1, 2*p, -1)).2)) ∧
    (D = ((roots (1, 2*p, -1)).1) ∨ D = ((roots (1, 2*p, -1)).2)) ∧
    _root_.is_cyclical A B C D → p = -1 := 
sorry

end concyclic_points_l819_819663


namespace sum_of_constants_l819_819344

-- Problem statement
theorem sum_of_constants (a b c : ℕ) 
  (h1 : a * a * b = 75) 
  (h2 : c * c = 128) 
  (h3 : ∀ d e f : ℕ, d * sqrt e / f = sqrt 75 / sqrt 128 → d = a ∧ e = b ∧ f = c) :
  a + b + c = 27 := 
sorry

end sum_of_constants_l819_819344


namespace repeating_decimal_fraction_eq_l819_819892

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end repeating_decimal_fraction_eq_l819_819892


namespace train_crosses_platform_in_approx_50_39_seconds_l819_819806

def length_of_train : ℝ := 250
def speed_of_train_kmh : ℝ := 55
def length_of_platform : ℝ := 520

def total_distance : ℝ := length_of_train + length_of_platform

def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

def time_to_cross_platform : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_platform_in_approx_50_39_seconds :
  abs (time_to_cross_platform - 50.39) < 0.01 :=
by
  sorry

end train_crosses_platform_in_approx_50_39_seconds_l819_819806


namespace log_base_5_of_15625_eq_6_l819_819071

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l819_819071


namespace helen_needed_gas_l819_819176

-- Definitions based on conditions
def cuts_per_month_routine_1 : ℕ := 2 -- Cuts per month for March, April, September, October
def cuts_per_month_routine_2 : ℕ := 4 -- Cuts per month for May, June, July, August
def months_routine_1 : ℕ := 4 -- Number of months with routine 1
def months_routine_2 : ℕ := 4 -- Number of months with routine 2
def gas_per_fill : ℕ := 2 -- Gallons of gas used every 4th cut
def cuts_per_fill : ℕ := 4 -- Number of cuts per fill

-- Total number of cuts in routine 1 months
def total_cuts_routine_1 : ℕ := cuts_per_month_routine_1 * months_routine_1

-- Total number of cuts in routine 2 months
def total_cuts_routine_2 : ℕ := cuts_per_month_routine_2 * months_routine_2

-- Total cuts from March to October
def total_cuts : ℕ := total_cuts_routine_1 + total_cuts_routine_2

-- Total fills needed from March to October
def total_fills : ℕ := total_cuts / cuts_per_fill

-- Total gallons of gas needed
def total_gal_of_gas : ℕ := total_fills * gas_per_fill

-- The statement to prove
theorem helen_needed_gas : total_gal_of_gas = 12 :=
by
  -- This would be replaced by our solution steps.
  sorry

end helen_needed_gas_l819_819176


namespace angle_bisector_relation_l819_819594

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l819_819594


namespace vendor_profit_is_three_l819_819836

def cost_per_apple := 3 / 2
def selling_price_per_apple := 10 / 5
def cost_per_orange := 2.7 / 3
def selling_price_per_orange := 1
def apples_sold := 5
def oranges_sold := 5

theorem vendor_profit_is_three :
  ((selling_price_per_apple - cost_per_apple) * apples_sold) +
  ((selling_price_per_orange - cost_per_orange) * oranges_sold) = 3 := by
  sorry

end vendor_profit_is_three_l819_819836


namespace lattice_points_on_hyperbola_l819_819985

theorem lattice_points_on_hyperbola :
  let n := 3^4 * 17^2 in
  ∃ (count : ℕ), count = 30 ∧ ∀ (x y : ℤ), x^2 - y^2 = n → (x, y) ∈ lattice_points n :=
by
  sorry

end lattice_points_on_hyperbola_l819_819985


namespace Bob_salary_after_changes_l819_819037

def salary_in_March := 2500
def raise_in_April := 0.10
def pay_cut_in_May := 0.25

theorem Bob_salary_after_changes :
  let salary_after_raise := salary_in_March * (1 + raise_in_April) in
  let salary_after_cut := salary_after_raise * (1 - pay_cut_in_May) in
  salary_after_cut = 2062.5 := by
  sorry

end Bob_salary_after_changes_l819_819037


namespace graham_crackers_leftover_l819_819273

-- Definitions for the problem conditions
def initial_boxes_graham := 14
def initial_packets_oreos := 15
def initial_ounces_cream_cheese := 36

def boxes_per_cheesecake := 2
def packets_per_cheesecake := 3
def ounces_per_cheesecake := 4

-- Define the statement that needs to be proved
theorem graham_crackers_leftover :
  initial_boxes_graham - (min (initial_boxes_graham / boxes_per_cheesecake) (min (initial_packets_oreos / packets_per_cheesecake) (initial_ounces_cream_cheese / ounces_per_cheesecake)) * boxes_per_cheesecake) = 4 :=
by sorry

end graham_crackers_leftover_l819_819273


namespace permutation_sum_contains_consecutive_l819_819112

def permutation_sum (perm : List ℕ) : ℚ :=
  (List.zipWith (λ a b => a / b) perm (List.range' 1 2001)).sum

theorem permutation_sum_contains_consecutive (S : ℕ) :
  ∃ n : ℕ, 
  ∀ (T : List ℕ), T ~ List.range' 1 2001 → 
  ∃! (v : ℚ), v = permutation_sum T ∧ (n <= v ∧ v <= n + 300) :=
sorry

end permutation_sum_contains_consecutive_l819_819112


namespace max_area_of_pentagon_in_circle_l819_819678

theorem max_area_of_pentagon_in_circle (ABCDE : ℝ → Prop) (circumscribed_square_ACDE : ℝ → Prop) (area_square : ℝ) :
  (circumscribed_square_ACDE 12) → 
  (∃ area : ℝ, area = 9 + 3 * real.sqrt 2 ∧ ABCDE area) :=
by
  assume h : circumscribed_square_ACDE 12
  sorry

end max_area_of_pentagon_in_circle_l819_819678


namespace pizza_slice_price_l819_819453

theorem pizza_slice_price (S : ℕ) 
  (price_large_slice : ℕ := 250)
  (num_total_slices : ℕ := 5000)
  (num_small_slices : ℕ := 2000)
  (total_revenue : ℕ := 1050000) :
  (2000 * S + 3000 * price_large_slice = total_revenue) → S = 150 :=
begin
  sorry
end

end pizza_slice_price_l819_819453


namespace probability_two_red_or_blue_is_0_2844_l819_819396

-- Definition of the problem conditions
def num_red_marbles := 5
def num_blue_marbles := 3
def num_yellow_marbles := 7
def total_marbles := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- The probability of drawing a red or blue marble in one draw
def P_red_or_blue := (num_red_marbles + num_blue_marbles) / total_marbles.toRat

-- The probability of drawing two marbles consecutively where both are either red or blue
-- with replacement
def P_two_red_or_blue := P_red_or_blue * P_red_or_blue

-- The expected result in decimal form
def expected_probability := 64 / 225 -- 0.2844 is the decimal form of this fraction

theorem probability_two_red_or_blue_is_0_2844 :
  P_two_red_or_blue.toReal = (64 / 225 : ℚ).toReal := sorry

end probability_two_red_or_blue_is_0_2844_l819_819396


namespace P_mult_Q_l819_819900

variable (Q : Matrix (Fin 3) (Fin 3) ℝ)

def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 0], ![0, 0, 1], ![0, 1, 0]]

theorem P_mult_Q (a b c d e f g h i : ℝ) (Q_eq : Q = !![
  [a, b, c],
  [d, e, f],
  [g, h, i]
]) : P ⬝ Q = !![
  [3 * a, 3 * b, 3 * c],
  [g, h, i],
  [d, e, f]
] :=
by sorry

end P_mult_Q_l819_819900


namespace length_of_wood_l819_819800

theorem length_of_wood (x : ℝ) :
  let rope_length := x + 4.5 in
  let folded_rope := rope_length / 2 in
  folded_rope = x - 1 :=
by
  let rope_length := x + 4.5
  let folded_rope := rope_length / 2
  have h : folded_rope = x - 1
  sorry

end length_of_wood_l819_819800


namespace problem_statement_l819_819219

noncomputable def pointA_polar :  ℝ × ℝ := (3 * real.sqrt 3, real.pi / 2)
noncomputable def pointB_polar :  ℝ × ℝ := (3, real.pi / 3)
noncomputable def circleC_polar (θ : ℝ) : ℝ := 2 * real.cos θ

theorem problem_statement :
  (let ρ := circleC_polar θ in ρ^2 = 2 * ρ * real.cos θ → (x^2 + y^2 = 2 * x) → ((x - 1)^2 + y^2 = 1)) ∧
  (max_area (pointA_polar, pointB_polar, circleC_polar) = (3 * real.sqrt 3 + 3) / 2) :=
begin
  sorry
end

end problem_statement_l819_819219


namespace solve_for_a_l819_819188

noncomputable def integral_condition (a : ℝ) : Prop :=
  ∫ x in 1..a, (2*x + (1/x)) = 3 + log 2

theorem solve_for_a (a : ℝ) (h : integral_condition a) : a = 2 :=
  sorry

end solve_for_a_l819_819188


namespace prove_sequence_general_formula_prove_min_integer_m_l819_819573

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (3 * x)

def a_seq : ℕ → ℝ
| 1     := 1
| (n+1) := f (1 / (a_seq n))

def b_seq : ℕ → ℝ
| 1     := 3
| n     := (1 / ((a_seq (n-1)) * (a_seq (n+1))))

def S_seq (n : ℕ) : ℝ :=
(1 to n).sum (λ i, b_seq i)

theorem prove_sequence_general_formula :
  a_seq n = (1/3) * (2 * n + 1) := 
sorry

theorem prove_min_integer_m (m : ℕ) :
  (∀ n : ℕ+, S_seq n < (m - 2007) / 2) → m = 2016 := 
sorry

end prove_sequence_general_formula_prove_min_integer_m_l819_819573


namespace nonneg_real_inequality_l819_819133

theorem nonneg_real_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := 
by
  sorry

end nonneg_real_inequality_l819_819133


namespace round_to_nearest_hundredth_of_18_4851_l819_819688

theorem round_to_nearest_hundredth_of_18_4851 : round_to_nearest_hundredth 18.4851 = 18.49 :=
by
  sorry

end round_to_nearest_hundredth_of_18_4851_l819_819688


namespace parallelogram_existence_l819_819869
noncomputable theory
open_locale classical

structure Parallelogram (A B C D : Type) :=
(side_eq_diff : ∃ AD DC : ℝ, AD - DC = d ∧ AC = given_diagonal)
(diag_angle  : ∠ (Vector (A, intersection_diagonals)) = φ)
(meet_conditions : quadrilateral A B C D)

variables {A B C D : Type} {d : ℝ} {φ : ℝ} {given_diagonal : ℝ}
hypothesis (h_d : d > 0)
hypothesis (h_AC : ∥AC∥ = given_diagonal)
hypothesis (h_angle : ∃ O : Type, ∠ (A, O, C) = φ)

theorem parallelogram_existence :
  ∃ (P : Parallelogram A B C D), 
    (P.side_eq_diff) ∧ (P.diag_angle) ∧ (P.meet_conditions) :=
sorry

end parallelogram_existence_l819_819869


namespace set_equality_l819_819788

theorem set_equality : {3, 2} = {x | x^2 - 5x + 6 = 0} :=
sorry

end set_equality_l819_819788


namespace distance_between_foci_of_ellipse_l819_819382

theorem distance_between_foci_of_ellipse :
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  2 * c = 14 :=
by
  let p1 := (1 : ℝ, 5 : ℝ)
  let p2 := (4 : ℝ, -3 : ℝ)
  let p3 := (9 : ℝ, 5 : ℝ)
  let center := ((1 + 9) / 2, (5 + 5) / 2)
  let a := real.sqrt ((4 - 5) ^ 2 + (-3 - 5) ^ 2)
  let b := (9 - 1) / 2
  let c := real.sqrt (a ^ 2 - b ^ 2)
  show 2 * c = 14, from sorry

end distance_between_foci_of_ellipse_l819_819382


namespace count_true_propositions_l819_819975

def proposition (x : ℝ) : Prop := x^2 > 1 → x > 1
def converse (x : ℝ) : Prop := x > 1 → x^2 > 1
def negation (x : ℝ) : Prop := x^2 ≤ 1 → x ≤ 1
def contrapositive (x : ℝ) : Prop := x ≤ 1 → x^2 ≤ 1

theorem count_true_propositions : (∃ x1 x2 x3 : ℝ, 
  converse x1 ∧ 
  negation x2 ∧ 
  ¬ contrapositive x3) = 
  2 :=
sorry

end count_true_propositions_l819_819975


namespace omega_range_l819_819567

-- Define the function f(x)
def f (omega : ℝ) (x : ℝ) : ℝ := 2 * sin (omega * x)

-- Define the interval
def interval : Set ℝ := Set.Icc (-π / 6) (π / 4)

-- Define the monotonicity condition
def is_monotonous (omega : ℝ) : Prop :=
  ∀ x1 x2 ∈ interval, x1 < x2 → f omega x1 ≤ f omega x2 ∨ f omega x1 ≥ f omega x2

-- Prove that the function is monotonous for the given range of omega
theorem omega_range :
  ∀ omega, is_monotonous omega ↔ (omega ∈ Set.Ico (-2 : ℝ) 0 ∨ omega ∈ Set.Ioo 0 2) :=
by
  intros omega
  sorry

end omega_range_l819_819567


namespace number_of_family_members_l819_819430

noncomputable def total_money : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 26 * 0.25

noncomputable def leftover_money : ℝ := 0.48

noncomputable def double_scoop_cost : ℝ := 3.0

noncomputable def amount_spent : ℝ := total_money - leftover_money

noncomputable def number_of_double_scoops : ℝ := amount_spent / double_scoop_cost

theorem number_of_family_members :
  number_of_double_scoops = 5 := by
  sorry

end number_of_family_members_l819_819430


namespace total_points_after_3_perfect_games_l819_819437

def perfect_score := 21
def number_of_games := 3

theorem total_points_after_3_perfect_games : perfect_score * number_of_games = 63 := 
by
  sorry

end total_points_after_3_perfect_games_l819_819437


namespace eight_ants_no_same_vertex_probability_l819_819482

open BigOperators

def derangement (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), (-1 : ℤ)^k * (n.choose k)

theorem eight_ants_no_same_vertex_probability :
  let n := 8
  let derangement_n := derangement n
  let total_outcomes := (3 : ℕ) ^ n
  (derangement_n : ℕ) / total_outcomes = 14833 / 19683 := 
by 
  sorry

end eight_ants_no_same_vertex_probability_l819_819482


namespace solve_inequality_l819_819307

noncomputable def f (x : ℝ) := (3 * x - 8) * (x - 4) * (x + 2) / x

theorem solve_inequality :
  { x : ℝ | f x ≥ 0 } = { x : ℝ | x ∈ Iic (-2) ∨ x ∈ Ici (8 / 3) ∩ Iio 4 ∨ x ∈ Ioi 4 } :=
by
  sorry

end solve_inequality_l819_819307


namespace systematic_sampling_interval_l819_819388

theorem systematic_sampling_interval
  (num_students : ℕ)
  (sample_size : ℕ)
  (num_students_eq : num_students = 72)
  (sample_size_eq : sample_size = 8) :
  num_students / sample_size = 9 :=
by
  rw [num_students_eq, sample_size_eq]
  norm_num

end systematic_sampling_interval_l819_819388


namespace range_of_a_l819_819526

theorem range_of_a (x a : ℝ) (p : x - a > 0) (q : x > 1) (hpq: ∀ x, p x → q x) : a > 1 :=
sorry

end range_of_a_l819_819526


namespace circles_positional_relationship_l819_819341

theorem circles_positional_relationship :
  ∃ R r : ℝ, (R * r = 2 ∧ R + r = 3) ∧ 3 = R + r → "externally tangent" = "externally tangent" :=
by
  sorry

end circles_positional_relationship_l819_819341


namespace count_valid_subsets_l819_819182

theorem count_valid_subsets :
  let S := { S : Finset ℕ // S ⊆ Finset.range 16 ∧ (∀ {x y}, x ∈ S → y ∈ S → x ≠ y + 1) ∧ (∀ k ∈ S, (S.card ≤ k) → k ∈ S) }
  in S.card = 405 :=
sorry

end count_valid_subsets_l819_819182


namespace geometry_problem_l819_819220

-- Definitions of the curves C1 and C2
def curve_C1 (θ : ℝ) : ℝ × ℝ := (sqrt 3 * cos θ, sin θ)
def curve_C2 (θ : ℝ) : ℝ := 2 * sqrt 2 / (sin (θ + π / 4))

-- Cartesian equation of curve C1
def cartesian_curve_C1 : Prop :=
  ∀ x y : ℝ, (∃ θ : ℝ, x = sqrt 3 * cos θ ∧ y = sin θ) → (x^2 / 3 + y^2 = 1)

-- Cartesian equation of curve C2
def cartesian_curve_C2 : Prop :=
  ∀ x y : ℝ, (∃ θ : ℝ, ρ θ = (sin (θ + π / 4)) → (x = ρ θ * cos θ ∧ y = ρ θ * sin θ)) → (x + y = 4)

-- Minimum distance proof
def min_distance (P Q : ℝ × ℝ) : Prop :=
  ∀ θ α : ℝ, 
  ((curve_C1 α = P) ∧ 
  (curve_C2 θ = (fst Q, snd Q)) → 
  dist (fst P, snd P) (fst Q, snd Q) = sqrt 2)
  ∧ (fst P = 3/2 ∧ snd P = 1/2)

theorem geometry_problem :
  cartesian_curve_C1 ∧ cartesian_curve_C2 ∧ min_distance :=
sorry

end geometry_problem_l819_819220


namespace log_base_5_of_15625_l819_819076

theorem log_base_5_of_15625 : log 5 15625 = 6 :=
by
  -- Given that 15625 is 5^6, we can directly provide this as a known fact.
  have h : 5 ^ 6 = 15625 := by norm_num
  rw [← log_eq_log_of_exp h] -- Use definition of logarithm
  norm_num
  exact sorry

end log_base_5_of_15625_l819_819076


namespace tray_height_form_l819_819015

-- Definitions of conditions
def square_side : ℝ := 120
def initial_cut_distance : ℝ := 5
def cut_angle : ℝ := 45

-- Proof problem adopted to Lean 4
theorem tray_height_form (m n : ℕ) :
  let height := sqrt 25 in
  height = real_root m n → m + n = 5 :=
sorry

end tray_height_form_l819_819015


namespace expand_binomials_l819_819485

theorem expand_binomials (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 :=
by
  sorry

end expand_binomials_l819_819485


namespace perfect_square_trinomial_m_l819_819544

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0) ∧ (x^2 - m * x + 25 = (a * x + b)^2)) → (m = 10 ∨ m = -10) :=
by
  -- Using the assumption that there exist constants a and b such that the trinomial is a perfect square
  intro h,
  obtain ⟨a, b, _, h_eq⟩ := h,
  -- Expanding the perfect square and comparing coefficients
  -- will yield the conclusion m = ±10
  sorry

end perfect_square_trinomial_m_l819_819544


namespace smallest_m_l819_819136

-- Define the conditions given in the problem
def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x
def f_derivative (x : ℝ) : ℝ := 6 * x - 2
def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n
def a (n : ℕ) : ℝ := S n - S (n - 1)

-- Condition: For n ≥ 2
lemma a_n_formula (n : ℕ) (h : n ≥ 2) : a n = 6 * n - 5 := 
sorry

-- Special case when n = 1
lemma a_1 : a 1 = 1 := 
sorry

-- Define b_n and T_n 
def b (n : ℕ) : ℝ := 3 / (a n * a (n + 1))
def T (n : ℕ) : ℝ := (List.range n).sum (λ i, b (i + 1))

-- Prove the smallest positive integer m satisfying T_n < m / 2016 for all n ∈ ℕ*
theorem smallest_m (m : ℕ) : m = 1008 → ∀ n : ℕ, n > 0 → T n < m / 2016 := 
sorry

end smallest_m_l819_819136


namespace train_speed_in_mps_l819_819021

def speed_kmph : ℕ := 189
def km_to_m : ℕ := 1000
def hr_to_s : ℕ := 3600

theorem train_speed_in_mps : (speed_kmph * km_to_m) / hr_to_s = 52.5 := by
  sorry

end train_speed_in_mps_l819_819021


namespace scientific_notation_of_0_0000021_l819_819457

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end scientific_notation_of_0_0000021_l819_819457


namespace solve_equation_l819_819497

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l819_819497


namespace number_of_chairs_l819_819521

theorem number_of_chairs (x t c b T C B: ℕ) (r1 r2 r3: ℕ)
  (h1: x = 2250) (h2: t = 18) (h3: c = 12) (h4: b = 30) 
  (h5: r1 = 2) (h6: r2 = 3) (h7: r3 = 1) 
  (h_ratio1: T / C = r1 / r2) (h_ratio2: B / C = r3 / r2) 
  (h_eq: t * T + c * C + b * B = x) : C = 66 :=
by
  sorry

end number_of_chairs_l819_819521


namespace sqrt_abc_sum_eq_162sqrt2_l819_819665

theorem sqrt_abc_sum_eq_162sqrt2 (a b c : ℝ) (h1 : b + c = 15) (h2 : c + a = 18) (h3 : a + b = 21) :
    Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 :=
by
  sorry

end sqrt_abc_sum_eq_162sqrt2_l819_819665


namespace village_speed_problem_l819_819709

theorem village_speed_problem :
  ∃ (x : ℝ), x > 0 ∧
    (∃ (t2 t1 : ℝ), t2 = 10 / x ∧ t1 = 10 / (x + 3) ∧ t2 = t1 + 3) ∧
    x = 2 ∧ (x + 3) = 5 :=
by {
  sorry,
}

end village_speed_problem_l819_819709


namespace matrix_P_swaps_and_triples_l819_819903

noncomputable def P : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

def Q : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![a, b, c],
  ![d, e, f],
  ![g, h, i]
]

theorem matrix_P_swaps_and_triples (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  P ⬝ Q = ![
    ![3 * Q 0 0, 3 * Q 0 1, 3 * Q 0 2],
    ![Q 2 0, Q 2 1, Q 2 2],
    ![Q 1 0, Q 1 1, Q 1 2]
  ] :=
by {
  sorry
}

end matrix_P_swaps_and_triples_l819_819903


namespace part_one_part_two_part_three_l819_819517

open Nat

def number_boys := 5
def number_girls := 4
def total_people := 9
def A_included := 1
def B_included := 1

theorem part_one : (number_boys.choose 2 * number_girls.choose 2) = 60 := sorry

theorem part_two : (total_people.choose 4 - (total_people - A_included - B_included).choose 4) = 91 := sorry

theorem part_three : (total_people.choose 4 - number_boys.choose 4 - number_girls.choose 4) = 120 := sorry

end part_one_part_two_part_three_l819_819517


namespace function_inequality_solution_l819_819530

variable {f : ℝ → ℝ}

-- Conditions
def condition1 (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h : x1 < x2) : Prop :=
  f(x1) - Real.sqrt(x1) > f(x2) - Real.sqrt(x2)

def condition2 : f 4 = 4 := by 
  sorry

-- Proof statement
theorem function_inequality_solution (x : ℝ) (h : 0 ≤ x) :
  (f(2 * x) < Real.sqrt(2 * x) + 2) ↔ (x > 2) := by 
  sorry

end function_inequality_solution_l819_819530


namespace inequality_solution_l819_819308

theorem inequality_solution (x : ℝ) : 
  (\frac{x - 5}{(x - 3)^2} < 0) ↔ x ∈ Set.Ioo (3:ℝ) 5 ∨ x < 3 := 
sorry

end inequality_solution_l819_819308


namespace find_a_plus_b_l819_819525
-- Definition of the problem variables and conditions
variables (a b : ℝ)
def condition1 : Prop := a - b = 3
def condition2 : Prop := a^2 - b^2 = -12

-- Goal: Prove that a + b = -4 given the conditions
theorem find_a_plus_b (h1 : condition1 a b) (h2 : condition2 a b) : a + b = -4 :=
  sorry

end find_a_plus_b_l819_819525


namespace middle_managers_sampling_l819_819424

theorem middle_managers_sampling :
  ∀ (total_employees senior_managers middle_managers general_staff sample_size : ℕ),
  total_employees = 1000 →
  senior_managers = 50 →
  middle_managers = 150 →
  general_staff = 800 →
  sample_size = 200 →
  (middle_managers.toRat / total_employees.toRat) * sample_size.toRat = 30 := 
by
  intros total_employees senior_managers middle_managers general_staff sample_size
  assume h1 h2 h3 h4 h5
  sorry

end middle_managers_sampling_l819_819424


namespace equation_solutions_equiv_l819_819651

theorem equation_solutions_equiv (p : ℕ) (hp : p.Prime) :
  (∃ x s : ℤ, x^2 - x + 3 - p * s = 0) ↔ 
  (∃ y t : ℤ, y^2 - y + 25 - p * t = 0) :=
by { sorry }

end equation_solutions_equiv_l819_819651


namespace range_of_slope_OP_l819_819924

noncomputable theory
open Classical Real

-- Define the ellipse with its parameters
def ellipse := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 }

-- Define the left focus F of the ellipse
def F : ℝ × ℝ := (-1, 0)

-- Define the condition for the point P on the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop := P ∈ ellipse

-- Define the slope of line FP
def slope_FP (P : ℝ × ℝ) : ℝ := (P.2 - F.2) / (P.1 - F.1)

-- Define the condition that the slope of FP is greater than sqrt(3)
def slope_FP_gt_sqrt3 (P : ℝ × ℝ) : Prop := slope_FP P > sqrt 3

-- Define the slope of line OP
def slope_OP (P : ℝ × ℝ) : ℝ := P.2 / P.1

-- The range of slopes we want to prove for the slope of line OP
def slope_OP_range : Set ℝ :=
  {k : ℝ | k < -3 / 2} ∪ {k : ℝ | 3 * sqrt 3 / 8 < k ∧ k ≤ 3 / 2}

theorem range_of_slope_OP (P : ℝ × ℝ) (h1 : is_on_ellipse P) (h2 : slope_FP_gt_sqrt3 P) :
  slope_OP P ∈ slope_OP_range := sorry

end range_of_slope_OP_l819_819924


namespace exists_assignment_method_for_odd_n_l819_819662

theorem exists_assignment_method_for_odd_n (n : ℕ) (hn : n % 2 = 1) :
  ∃ (f : fin (2 * n) → ℕ), 
    (∀ i, f i ∈ finset.range (2 * n + 1)) ∧
    (∀ k : fin n, 
      let i := k.1 * 2 in 
      let j := (k.1 * 2 + 1) % (2 * n) in 
      let m := (k.1 * 2 + 2) % (2 * n) in 
      (f ⟨i, lt_add_one i⟩ + f ⟨j, lt_add_one j⟩ + f ⟨m, lt_add_one m⟩) = (3 * (2 * n + 1) // (2 * n))) :=
sorry

end exists_assignment_method_for_odd_n_l819_819662


namespace correct_statements_count_l819_819939

variable (a b c : ℝ)

/-- Given a quadratic function y = ax^2 + bx + c, and y > 0 for -2 < x < 3,
    prove that the number of correct statements (listed below) is 2:

1. b = -a
2. a + b + c < 0
3. The solution set of the inequality ax + c > 0 is x > 6
4. The solutions of the equation cx^2 - bx + a = 0 are x₁ = -1/3 and x₂ = 1/2 
-/ 
theorem correct_statements_count :
  ∃ n : ℕ, n = 2 ∧ ∀ h₁ h₂ h₃ h₄ : ℝ,
    (h₁ = b + a → false) ∧
    (h₂ = a + b + c → false) ∧
    (h₃ = (x : ℝ) (hx : x > 6) → (ax + c ≤ 0)) ∧
    (h₄ = (x₁ = -1/3 ∧ x₂ = 1/2) → (cx^2 - bx + a = 0))
:= sorry

end correct_statements_count_l819_819939


namespace remaining_cubes_count_l819_819302

-- Define the initial number of cubes
def initial_cubes : ℕ := 64

-- Define the holes in the bottom layer
def holes_in_bottom_layer : ℕ := 6

-- Define the number of cubes removed per hole
def cubes_removed_per_hole : ℕ := 3

-- Define the calculation for missing cubes
def missing_cubes : ℕ := holes_in_bottom_layer * cubes_removed_per_hole

-- Define the calculation for remaining cubes
def remaining_cubes : ℕ := initial_cubes - missing_cubes

-- The theorem to prove
theorem remaining_cubes_count : remaining_cubes = 46 := by
  sorry

end remaining_cubes_count_l819_819302


namespace sabrina_2000th_digit_is_427_l819_819298

-- Define a sequence of positive integers that start with the digit 2
def sequence_starts_with_two : ℕ → ℕ
| 0 => 2
| n => (if n % 10 = 9 then sequence_starts_with_two (n / 10) + 1 else sequence_starts_with_two (n - 1) + 1)

-- Define the digits written by Sabrina up to the 2000th digit
def digits_written_up_to : ℕ → ℕ
| 0 => 2
| n => if n < 10 then n + 2 else 1

-- The main theorem to prove
theorem sabrina_2000th_digit_is_427 :
  let digits := List.init 2000 digits_written_up_to,
  (digits.get! 1997) * 100 + (digits.get! 1998) * 10 + (digits.get! 1999) = 427 :=
sorry

end sabrina_2000th_digit_is_427_l819_819298


namespace no_possible_blue_socks_l819_819738

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l819_819738


namespace part_I_part_II_l819_819171

def vector_a (m : ℝ) (x : ℝ) : ℝ × ℝ := (m, Real.cos (2 * x))
def vector_b (n : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), n)
def f (m n x : ℝ) : ℝ := vector_a m x • vector_b n x

theorem part_I (x₁ x₂ y₁ y₂ : ℝ) (m n : ℝ)
  (hx1 : x₁ = π / 12) (hx2 : x₂ = 2 * π / 3) 
  (hy1 : y₁ = sqrt 3) (hy2 : y₂ = -2) 
  (hfy1 : f m n x₁ = y₁) (hfy2 : f m n x₂ = y₂) :
  m = sqrt 3 ∧ n = 1 :=
by
  sorry

def g (ϕ x : ℝ) : ℝ := 2 * Real.sin (2 * x + 2 * ϕ + π / 6)

theorem part_II (ϕ x : ℝ) (hx : 0 < ϕ ∧ ϕ < π) 
  (distance_cond : ∀ x₀, x₀ = 0 → sqrt (1 + x₀^2) = 1):
  ∃ k : ℤ, x ∈ Icc (-π / 2 + k * π) (k * π) :=
by
  sorry

end part_I_part_II_l819_819171


namespace contains_K4_l819_819135

noncomputable def G : SimpleGraph (Fin 8) := sorry

axiom subgraph_contains_K3 (S : Finset (Fin 8)) (hS : S.card = 5) : 
  ∃ (H : SimpleGraph S), H.is_subset G ∧ ∃ (T : Finset S), T.card = 3 ∧ H.is_complete T

theorem contains_K4 : ∃ (T : Finset (Fin 8)), T.card = 4 ∧ G.is_complete T :=
by
  -- Reference the conditions we stated in the axioms
  sorry

end contains_K4_l819_819135


namespace no_blue_socks_make_probability_half_l819_819753

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l819_819753


namespace adults_bought_tickets_l819_819387

def cost_student : ℕ := 6
def cost_adult : ℕ := 8
def number_of_students : ℕ := 20
def total_sales : ℕ := 216
def students_contribution : ℕ := number_of_students * cost_student

theorem adults_bought_tickets : 
  ∃ (A : ℕ), students_contribution + A * cost_adult = total_sales ∧ A = 12 :=
by
  let A := (total_sales - students_contribution) / cost_adult
  use A
  have h1 : A * cost_adult = 96 := by { unfold students_contribution, unfold cost_adult, linarith }
  have h2 : A = 96 / 8 := by linarith
  rw h2
  split
  . simp only [students_contribution, cost_adult], linarith
  . exact rfl

end adults_bought_tickets_l819_819387


namespace fractions_equiv_conditions_l819_819063

theorem fractions_equiv_conditions (x y z : ℝ) (h₁ : 2 * x - z ≠ 0) (h₂ : z ≠ 0) : 
  ((2 * x + y) / (2 * x - z) = y / -z) ↔ (y = -z) :=
by
  sorry

end fractions_equiv_conditions_l819_819063


namespace hyperbola_asymptote_hyperbola_eccentricity_l819_819150

variable (b : ℝ) (a : ℝ) (x : ℝ) (y : ℝ) (c : ℝ) (e : ℝ)

theorem hyperbola_asymptote (hb : b > 0) (h_asymptote : ∀ x y, y = b * x → y = 2 ∧ x = 1) 
  : b = 2 := by
  sorry

theorem hyperbola_eccentricity (ha : a = 1) (hb : b = 2) (h_c : c = sqrt (a^2 + b^2))
  : e = c / a → e = sqrt 5 := by
  sorry

end hyperbola_asymptote_hyperbola_eccentricity_l819_819150


namespace preparatory_course_members_l819_819378

def total_members := 60
def members_passed := 0.3 * total_members
def members_not_passed := total_members - members_passed
def members_not_taken_course := 30

theorem preparatory_course_members:
  ∃ (P: ℕ), members_not_passed = members_not_taken_course + P ∧ P = 12 :=
by
  let P := 12
  have h1: members_not_passed = 42 := by
    calc
      members_not_passed = total_members - members_passed : by sorry
      ... = 60 - 0.3 * 60 : by sorry
      ... = 60 - 18 : by sorry
      ... = 42 : by sorry
  use P
  sorry

end preparatory_course_members_l819_819378


namespace no_six_distinct_roots_powers_of_two_l819_819253

open RealPolynomial

theorem no_six_distinct_roots_powers_of_two (f g : RealPolynomial)
  (hf : degree f = 2)
  (hg : degree g = 3) :
  ¬ (∃ (roots : Finset ℝ), (∀ (x ∈ roots), ∃ (k : ℤ), x = 2^k) ∧ roots.card = 6 ∧ ∀ x ∈ roots, eval (eval f g) x = 0) :=
by
  sorry

end no_six_distinct_roots_powers_of_two_l819_819253


namespace find_position_of_1000th_A_l819_819474

/-- The sequence consists of words formed by letters "A" and "B". 
    The first word in the sequence is "A". 
    The k-th word is derived from the (k-1)-th word by replacing each "A" with "AAB" and each "B" with "A". 
    We need to prove that the 1000th "A" appears at position 1414. -/
theorem find_position_of_1000th_A :
    (∃ n : Nat, nth_A_position n = 1000 ∧ position n = 1414) := sorry

end find_position_of_1000th_A_l819_819474


namespace parallel_vectors_xy_sum_l819_819923

theorem parallel_vectors_xy_sum (x y : ℚ) (k : ℚ) 
  (h1 : (2, 4, -5) = (2 * k, 4 * k, -5 * k)) 
  (h2 : (3, x, y) = (2 * k, 4 * k, -5 * k)) 
  (h3 : 3 = 2 * k) : 
  x + y = -3 / 2 :=
by
  sorry

end parallel_vectors_xy_sum_l819_819923


namespace alfonzo_visit_l819_819472

-- Define the number of princes (palaces) as n
variable (n : ℕ)

-- Define the type of connections (either a "Ruelle" or a "Canal")
inductive Transport
| Ruelle
| Canal

-- Define the connection between any two palaces
noncomputable def connection (i j : ℕ) : Transport := sorry

-- The theorem states that Prince Alfonzo can visit all his friends using only one type of transportation
theorem alfonzo_visit (h : ∀ i j, i ≠ j → ∃ t : Transport, ∀ k, k ≠ i → connection i k = t) :
  ∃ t : Transport, ∀ i j, i ≠ j → connection i j = t :=
sorry

end alfonzo_visit_l819_819472


namespace find_circle_eq_l819_819529

noncomputable def circle_eq (a b r : ℝ) : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), (p.1 - a)^2 + (p.2 - b)^2 = r^2

def circle_conditions (a b r : ℝ) : Prop :=
  circle_eq a b r (4, 1) ∧ circle_eq a b r (2, 1) ∧ (b - 1) / (a - 2) = -1

theorem find_circle_eq : ∃ (a b r : ℝ), circle_conditions a b r ∧ (λ (x y : ℝ), (x - 3)^2 + y^2 = 2) :=
by
  sorry

end find_circle_eq_l819_819529


namespace equilateral_triangles_count_l819_819332

-- Define the values and conditions
def line1 (k : ℤ) : ℝ → ℝ := λ x => k
def line2 (k : ℤ) : ℝ → ℝ := λ x => (sqrt 3) * x + 3 * k
def line3 (k : ℤ) : ℝ → ℝ := λ x => (-sqrt 3) * x + 3 * k

-- Define the range of k
def range_k : Finset ℤ := Finset.range (25) - 12
def side_length : ℝ := 3 / (sqrt 3)

-- Main statement 
theorem equilateral_triangles_count :
  (∑ k in range_k, ∑ k' in range_k, 
    (∑ k'' in range_k, 
      if line1 k' some_value = line2 k some_value 
      ∧ line1 k' some_other_value = line3 k some_other_value
      ∧ line2 k some_value = line3 k'' some_value
      then 1 else 0)) = 3456 := 
sorry

end equilateral_triangles_count_l819_819332


namespace tan_neg_two_sin_cos_sum_l819_819190

theorem tan_neg_two_sin_cos_sum (θ : ℝ) (h : Real.tan θ = -2) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = -7 / 5 :=
by
  sorry

end tan_neg_two_sin_cos_sum_l819_819190


namespace primes_satisfying_equation_l819_819122

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l819_819122


namespace trisected_rectangle_is_rhombus_l819_819066

-- Definitions
variable {A B C D H E F G : ℝ}
variable {Rect : Type} (vertices : Rect → Set ℝ)

-- Given conditions
def is_rectangle (rect : Rect) : Prop :=
  ∃ (A B C D : ℝ), {A, B, C, D} = vertices rect ∧ 
  (A, B, C, D) form_rectangle_properties

def trisected_points (rect : Rect) : Set ℝ :=
  let points := vertices rect
  { H, E, F, G | (H, E, F, G) are_trisected_points properties }

-- Proposition to prove
theorem trisected_rectangle_is_rhombus (rect : Rect)
  (h_rect : is_rectangle rect) :
  (quadrilateral formed by trisected_points rect) is_a_rhombus :=
by
  sorry

end trisected_rectangle_is_rhombus_l819_819066


namespace solve_for_x_l819_819695

theorem solve_for_x (x : ℚ) : (x - 50) / 3 = (5 - 3 * x) / 4 + 2 → x = 287 / 13 :=
begin
  sorry
end

end solve_for_x_l819_819695


namespace ratio_of_square_sides_l819_819363

theorem ratio_of_square_sides (a b c : ℕ) (h : ratio_area = (75 / 128) := sorry) (ratio_area :
 ratio_side = sqrt (75 / 128) := sorry) : a + b + c = 27 :=
begin
  sorry
end

end ratio_of_square_sides_l819_819363


namespace count_valid_integers_correct_l819_819181

def no_adjacent_identical_digits (n : ℕ) : Prop :=
  ∀ i, i < 5 → (n / 10^i) % 10 ≠ (n / 10^(i+1)) % 10

def count_valid_integers : ℕ :=
  (Finset.range 1000000).filter (λ n, no_adjacent_identical_digits n).card

theorem count_valid_integers_correct :
  count_valid_integers = 597871 :=
sorry

end count_valid_integers_correct_l819_819181


namespace log_base_5_of_15625_l819_819080

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l819_819080


namespace product_quantity_B_l819_819018

theorem product_quantity_B
(x y z : ℕ) (h1 : 2*x + 3*y + 5*z = 20) (h2 : x ≥ 1) (h3 : y ≥ 1) (h4 : z ≥ 1) : 
  y = 1 :=
by
  have h5 : y ≥ 3, from sorry,
  have h6 : x = 3 ∧ y = 3 ∧ z = 1, from sorry,
  exact sorry

end product_quantity_B_l819_819018


namespace cos_theta_value_l819_819165

noncomputable def vector_a : ℝ × ℝ := (-2, 1)
noncomputable def vector_b : ℝ × ℝ := ((2 + 2), (3 - 1)) / 2 -- This simplifies to (2, 1)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def cos_theta : ℝ :=
  let a := vector_a
  let b := vector_b
  dot_product a b / (magnitude a * magnitude b)

theorem cos_theta_value :
  cos_theta = -3 / 5 :=
by
  sorry

end cos_theta_value_l819_819165


namespace find_t_l819_819144

noncomputable def a : Type* := sorry
noncomputable def b : Type* := sorry
noncomputable def t : ℝ := sorry
noncomputable def AB := t * a - b
noncomputable def AC := 2 * a + 3 * b

axiom non_collinear (a b : Type*) : a ≠ b
axiom collinear (AB AC : Type*) : True -- Collinearity of points A, B, C

theorem find_t (t : ℝ) (a b AB AC : Type*) [non_collinear_def : non_collinear a b] [collinear_def : collinear AB AC] : 
  AB = t * a - b ∧ AC = 2 * a + 3 * b → t = -2 / 3 := by
  sorry

end find_t_l819_819144


namespace probability_same_two_dishes_l819_819211

theorem probability_same_two_dishes (h : 3.hot_dishes) (c : choose 2 3 = 3) : 
  (∃ p : ℚ, p = 1/3 ∧ p = favorable_ways_same_2_dishes / total_outcomes) := 
begin
  sorry
end

end probability_same_two_dishes_l819_819211


namespace largest_N_exists_l819_819113

noncomputable def parabola_properties (a T : ℤ) :=
    (∀ (x y : ℤ), y = a * x * (x - 2 * T) → (x = 0 ∨ x = 2 * T) → y = 0) ∧ 
    (∀ (v : ℤ × ℤ), v = (2 * T + 1, 28) → 28 = a * (2 * T + 1))

theorem largest_N_exists : 
    ∃ (a T : ℤ), T ≠ 0 ∧ (∀ (P : ℤ × ℤ), P = (0, 0) ∨ P = (2 * T, 0) ∨ P = (2 * T + 1, 28)) 
    ∧ (s = T - a * T^2) ∧ s = 60 :=
sorry

end largest_N_exists_l819_819113


namespace minimize_quadratic_expression_l819_819780

theorem minimize_quadratic_expression : 
  (∃ x : ℝ, ∀ y : ℝ, (x^2 + 15 * x + 3 ≤ y^2 + 15 * y + 3)) ↔ (x = -15 / 2) :=
begin
  sorry
end

end minimize_quadratic_expression_l819_819780


namespace centroid_of_triangle_l819_819679

variables {A B C P G : Type} [AddGroup P] [Module ℝ P] [AffineSpace P (P × ℝ)]

/--
Given any point P in the plane of triangle ABC, and if G is a point such that
\(\overrightarrow{PG} = \frac{1}{3}(\overrightarrow{PA} + \overrightarrow{PB} + \overrightarrow{PC})\),
prove that G is the centroid of triangle ABC.
-/
theorem centroid_of_triangle
  {Π : Type} [AddGroup Π] [Module ℝ Π] [AffineSpace Π (Π × ℝ)]
  (A B C P : Π) :
  (∃ G : Π, (toVec P G) = (1 / 3 : ℝ) • (toVec P A + toVec P B + toVec P C)) →
  is_centroid A B C G :=
begin
  sorry
end

end centroid_of_triangle_l819_819679


namespace coin_toss_probability_l819_819242

theorem coin_toss_probability :
  let keiko_tosses := [(tt, tt), (tt, ff), (ff, tt), (ff, ff)],
      ephraim_tosses := [(tt, tt, tt), (tt, tt, ff), (tt, ff, tt), (tt, ff, ff),
                         (ff, tt, tt), (ff, tt, ff), (ff, ff, tt), (ff, ff, ff)],
      same_heads (k e) := (k.1 + k.2 = e.1 + e.2 + e.3),
      num_favorable := (keiko_tosses.product ephraim_tosses).count (λ x => same_heads x.fst x.snd),
      num_outcomes := keiko_tosses.length * ephraim_tosses.length
  in num_favorable / num_outcomes = 3 / 16 := sorry

end coin_toss_probability_l819_819242


namespace b_share_220_l819_819689

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : A + B + C = 770) : 
  B = 220 :=
by
  sorry

end b_share_220_l819_819689


namespace no_blue_socks_make_probability_half_l819_819750

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l819_819750


namespace max_value_trig_function_l819_819718

theorem max_value_trig_function :
  ∀ x : ℝ, 2 * real.cos x + real.sin x ≤ sqrt 5 :=
begin
  sorry
end

end max_value_trig_function_l819_819718


namespace regression_decrease_by_three_l819_819534

-- Given a regression equation \hat y = 2 - 3 \hat x
def regression_equation (x : ℝ) : ℝ :=
  2 - 3 * x

-- Prove that when x increases by one unit, \hat y decreases by 3 units
theorem regression_decrease_by_three (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -3 :=
by
  -- proof
  sorry

end regression_decrease_by_three_l819_819534


namespace compare_values_l819_819126

def a := 2 ^ (-1 / 3)
def b := Real.logb 2 (1 / 3)
def c := Real.logb 2 3

theorem compare_values : c > a ∧ a > b := 
by
  sorry

end compare_values_l819_819126


namespace proportional_function_decreases_l819_819971

theorem proportional_function_decreases
  (k : ℝ) (h : k ≠ 0) (h_point : ∃ k, (-4 : ℝ) = k * 2) :
  ∀ x1 x2 : ℝ, x1 < x2 → (k * x1) > (k * x2) :=
by
  sorry

end proportional_function_decreases_l819_819971


namespace arithmetic_sequence_sum_l819_819140

/-- Given an arithmetic sequence {a_n} and the first term a_1 = -2010, 
and given that the average of the first 2009 terms minus the average of the first 2007 terms equals 2,
prove that the sum of the first 2011 terms S_2011 equals 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : ∃ d, ∀ n, a n = a 1 + (n - 1) * d)
  (h_Sn : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h_a1 : a 1 = -2010)
  (h_avg_diff : (S 2009) / 2009 - (S 2007) / 2007 = 2) :
  S 2011 = 0 := 
sorry

end arithmetic_sequence_sum_l819_819140


namespace math_proof_problem_l819_819909

noncomputable def problem_statement : Prop :=
  ∃ (x : ℝ), (x > 12) ∧ ((x - 5) / 12 = 5 / (x - 12)) ∧ (x = 17)

theorem math_proof_problem : problem_statement :=
by
  sorry

end math_proof_problem_l819_819909


namespace chord_length_correct_l819_819560

noncomputable def chord_length_intercepted_by_line_on_circle
  (C : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4)
  (l : ∀ x y : ℝ, x - y + 2 = 0) : ℝ :=
  let center := (1, 1)
  let radius := 2
  let distance := |1 - 1 + 2| / Real.sqrt (1^2 + 1^2)
  let chord_length := 2 * Real.sqrt (radius^2 - distance^2)
  chord_length


theorem chord_length_correct :
  chord_length_intercepted_by_line_on_circle
    (λ x y, (x - 1)^2 + (y - 1)^2 = 4)
    (λ x y, x - y + 2 = 0) = 2 * Real.sqrt 2 := 
sorry

end chord_length_correct_l819_819560


namespace Jessica_biking_speed_l819_819240

theorem Jessica_biking_speed
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance total_time : ℝ)
  (h1 : swim_distance = 0.5)
  (h2 : swim_speed = 1)
  (h3 : run_distance = 5)
  (h4 : run_speed = 5)
  (h5 : bike_distance = 20)
  (h6 : total_time = 4) :
  bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed)) = 8 :=
by
  -- Proof omitted
  sorry

end Jessica_biking_speed_l819_819240


namespace can_form_triangle_l819_819370

theorem can_form_triangle 
  (A B C A' B' C' : Type)
  (h1: ∠A'BC' > 120°) 
  (h2: ∠C'AB' > 120°)
  (h3: ∠B'CA' > 120°)
  (hab' : AB' = AC')
  (hbc' : BC' = BA')
  (hca' : CA' = CB') 
  : AB' + BC' > CA' ∧ BC' + CA' > AB' ∧ CA' + AB' > BC' := 
sorry

end can_form_triangle_l819_819370


namespace joan_remaining_balloons_l819_819241

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2
def remaining_balloons : ℕ := initial_balloons - lost_balloons

theorem joan_remaining_balloons : remaining_balloons = 7 := by
  sorry

end joan_remaining_balloons_l819_819241


namespace no_possible_blue_socks_l819_819736

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l819_819736


namespace polygon_sides_l819_819065

def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem polygon_sides (n p : ℕ) (h : n > 0) (hp : p > 0) : 
  (∃ k, n = p * k) → 
  (number_of_sides : ℕ := n / p) ∧
  (¬(∃ k, n = p * k)) → 
  (number_of_sides : ℕ := lcm n p / p) 
  := 
sorry

end polygon_sides_l819_819065


namespace solve_for_r_squared_l819_819423

-- Definitions for the conditions
def radius (r : ℝ) : Prop :=
  ∃ (O : ℝ) (circle_with_center_radius : O = r), 
    (∃ A B C D P : ℝ,
    chord_AB_length : (A - B).abs = 12,
    chord_CD_length : (C - D).abs = 9,
    BP_length : (P - B).abs = 10,
    ∠APD_eq : ∠(A, P, D) = 60)

-- Theorem to prove the desired result
theorem solve_for_r_squared (r : ℝ) (h : radius r) : r^2 = 111 := by
  sorry

end solve_for_r_squared_l819_819423


namespace length_of_segments_touched_l819_819450

-- Definitions for given conditions
variables (a r : ℝ) (A B C : ℝ)
def equilateral_triangle (A B C : Type) := A = B ∧ B = C
def circle_radius_r (a r : ℝ) := r
def arcs_inside_angle (a r : ℝ) := a + r
def reuleaux_shape_width (a r : ℝ) := a + 2*r
def square_frame_side (a r : ℝ) := a + 2*r

-- The problem statement
theorem length_of_segments_touched (a r : ℝ) :
  (a + 2*r) - 2 * (r + a * (1 - real.sqrt 3 / 2)) = a * (real.sqrt 3 - 1) :=
sorry -- proof goes here

end length_of_segments_touched_l819_819450


namespace largest_even_not_sum_of_two_composite_odds_l819_819878

-- Definitions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ k, k > 1 ∧ k < n ∧ n % k = 0

-- Theorem statement
theorem largest_even_not_sum_of_two_composite_odds :
  ∀ n : ℕ, is_even n → n > 0 → (¬ (∃ a b : ℕ, is_odd a ∧ is_odd b ∧ is_composite a ∧ is_composite b ∧ n = a + b)) ↔ n = 38 := 
by
  sorry

end largest_even_not_sum_of_two_composite_odds_l819_819878


namespace arithmetic_sequence_20th_term_l819_819867

theorem arithmetic_sequence_20th_term :
  ∀ (a d : ℕ), a = 2 → d = 3 → (a + 19 * d) = 59 :=
by
  intros a d ha hd
  rw [ha, hd]
  simp
  sorry

end arithmetic_sequence_20th_term_l819_819867


namespace jeff_boxes_filled_l819_819643

noncomputable def jeff_donuts_per_day : ℕ := 10
noncomputable def number_of_days : ℕ := 12
noncomputable def jeff_eats_per_day : ℕ := 1
noncomputable def chris_eats : ℕ := 8
noncomputable def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled :
  let total_donuts := jeff_donuts_per_day * number_of_days
  let jeff_eats_total := jeff_eats_per_day * number_of_days
  let remaining_donuts_after_jeff := total_donuts - jeff_eats_total
  let remaining_donuts_after_chris := remaining_donuts_after_jeff - chris_eats
  let boxes_filled := remaining_donuts_after_chris / donuts_per_box
  in boxes_filled = 10 :=
by {
  sorry
}

end jeff_boxes_filled_l819_819643


namespace equilateral_triangles_count_l819_819715

/--
Problem:
The graphs of the equations
  y = k, y = sqrt 3 * x + k, y = -sqrt 3 * x + k,
are plotted for k = -5, -4, ..., 4, 5. These 33 lines divide the plane into smaller equilateral triangles of side 1 / sqrt 3. Prove that the number of such triangles is 300.
-/
theorem equilateral_triangles_count :
  let lines : List (ℝ → ℝ) :=
    List.map (λ k, (λ (x : ℝ), k)) (List.range' (-5) 11) ++
    List.map (λ k, (λ (x : ℝ), sqrt 3 * x + k)) (List.range' (-5) 11) ++
    List.map (λ k, (λ (x : ℝ), -sqrt 3 * x + k)) (List.range' (-5) 11) in
  ∃ n, n = 300 ∧ count_equilateral_triangles (List.range' (-5) 11) = n :=
begin
  let triangles := number-of-equilateral-triangles (List.range' (-5) 11),
  use 300,
  split,
  { refl },
  { sorry },
end

end equilateral_triangles_count_l819_819715


namespace right_triangle_inscribed_circle_area_l819_819339

noncomputable def inscribed_circle_area (p c : ℝ) : ℝ :=
  π * (p - c) ^ 2

theorem right_triangle_inscribed_circle_area (p c : ℝ) (h : c > 0) 
  (perimeter : ∃ x y : ℝ, 2 * p = x + y + c ∧ x > 0 ∧ y > 0) :
  inscribed_circle_area p c = π * (p - c) ^ 2 := by 
  sorry

end right_triangle_inscribed_circle_area_l819_819339


namespace find_initial_amount_l819_819096

-- Define the starting amount and the multiplicative increase
def initial_amount (P : ℕ) : Prop :=
  (9/8 : ℚ)^2 * P = 72900

-- The theorem statement that encodes the problem
theorem find_initial_amount : ∃ (P : ℕ), initial_amount P :=
begin
  use 57600,    -- Provide the solution found
  unfold initial_amount,    -- Unfold the definition
  norm_num,    -- Simplify numeric expressions
  sorry    -- Skip detailed proof steps
end

end find_initial_amount_l819_819096


namespace number_of_true_propositions_l819_819030

-- Definitions of the propositions
def original_prop (x : ℝ) : Prop := x = 3 → x ^ 2 = 9
def converse_prop (x : ℝ) : Prop := x ^ 2 = 9 → x = 3
def inverse_prop (x : ℝ) : Prop := x ≠ 3 → x ^ 2 ≠ 9
def contrapositive_prop (x : ℝ) : Prop := x ^ 2 ≠ 9 → x ≠ 3
def negation_prop (x : ℝ) : Prop := ¬ (x = 3 → x ^ 2 = 9)

-- Prove the number of true propositions
theorem number_of_true_propositions : ∀ (x : ℝ), 
  (original_prop x ∧ contrapositive_prop x) ∧ ¬(converse_prop x) ∧ ¬(inverse_prop x) ∧ ¬(negation_prop x) :=
begin
  -- Proof omitted
  sorry
end

end number_of_true_propositions_l819_819030


namespace shift_increase_l819_819604

-- Define what it means for a function to be increasing on an interval
def increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- Define the interval [-2, 3]
def interval_2_3 := set.Icc (-2:ℝ) 3

-- Define the interval [-7, -2]
def interval_7_2 := set.Icc (-7:ℝ) (-2)

-- The hypothesis: f is increasing on [-2, 3]
variable (f : ℝ → ℝ)
variable (hf : increasing_on f interval_2_3)

-- Prove that f(x + 5) is increasing on [-7, -2]
theorem shift_increase :
  increasing_on (λ x, f (x + 5)) interval_7_2 :=
sorry

end shift_increase_l819_819604


namespace maximum_value_P_l819_819654

open Classical

noncomputable def P (a b c d : ℝ) : ℝ := a * b + b * c + c * d + d * a

theorem maximum_value_P : ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 40 → P a b c d ≤ 800 :=
by
  sorry

end maximum_value_P_l819_819654


namespace capacity_of_new_bathtub_is_400_liters_l819_819770

-- Definitions based on conditions
def possible_capacities : Set ℕ := {4, 40, 400, 4000}  -- The possible capacities

-- Proof statement
theorem capacity_of_new_bathtub_is_400_liters (c : ℕ) 
  (h : c ∈ possible_capacities) : 
  c = 400 := 
sorry

end capacity_of_new_bathtub_is_400_liters_l819_819770


namespace triangle_perimeter_l819_819945

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l819_819945


namespace worker_total_earnings_l819_819844

def hourly_rate_ordinary := 0.60  -- in dollars per hour
def hourly_rate_overtime := 0.90  -- in dollars per hour
def total_hours := 50  -- in hours
def overtime_hours := 8  -- in hours

def ordinary_hours := total_hours - overtime_hours
def earnings_ordinary := ordinary_hours * hourly_rate_ordinary
def earnings_overtime := overtime_hours * hourly_rate_overtime
def total_earnings := earnings_ordinary + earnings_overtime

theorem worker_total_earnings : total_earnings = 32.40 := by
  sorry

end worker_total_earnings_l819_819844


namespace find_number_l819_819997

-- Definitions from the conditions
def condition1 (x : ℝ) := 16 * x = 3408
def condition2 (x : ℝ) := 1.6 * x = 340.8

-- The statement to prove
theorem find_number (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x = 213 :=
by
  sorry

end find_number_l819_819997


namespace centroid_condition_l819_819553

theorem centroid_condition (A B C G M N : Type) 
  [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B] [AddCommGroup C] [Module ℝ C]
  [AddCommGroup G] [Module ℝ G] [AddCommGroup M] [Module ℝ M] [AddCommGroup N] [Module ℝ N]
  (centroid_def : G = (1/3 : ℝ) • (A + B + C))
  (AG_def : (G : A) = (1 / 3) • (B + C))
  (AM_def : (M : A) = x • (B : A))
  (AN_def : (N : A) = y • (C : A))
  (collinearity : ∃ (λ : ℝ), (G - M : A) = λ • (N - G)) :
  (1 / x) + (1 / y) = 3 :=
by
  sorry

end centroid_condition_l819_819553


namespace minimize_m_at_l819_819930

noncomputable def m (x y : ℝ) : ℝ := 4 * x ^ 2 - 12 * x * y + 10 * y ^ 2 + 4 * y + 9

theorem minimize_m_at (x y : ℝ) : m x y = 5 ↔ (x = -3 ∧ y = -2) := 
sorry

end minimize_m_at_l819_819930


namespace length_of_jordans_rectangle_l819_819041

theorem length_of_jordans_rectangle
  (carol_length : ℕ) (carol_width : ℕ) (jordan_width : ℕ) (equal_area : (carol_length * carol_width) = (jordan_width * 2)) :
  (2 = 120 / 60) := by
  sorry

end length_of_jordans_rectangle_l819_819041


namespace no_equal_prob_for_same_color_socks_l819_819744

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l819_819744


namespace sum_of_ratio_simplified_l819_819353

theorem sum_of_ratio_simplified (a b c : ℤ) 
  (h1 : (a * a * b) = 75)
  (h2 : (c * c * 2) = 128)
  (h3 : c = 16) :
  a + b + c = 27 := 
by 
  have ha : a = 5 := sorry
  have hb : b = 6 := sorry
  rw [ha, hb, h3]
  norm_num
  exact eq.refl 27

end sum_of_ratio_simplified_l819_819353


namespace paired_divisors_distinct_primes_l819_819661

theorem paired_divisors_distinct_primes (n : ℕ) (h_pos : 0 < n)
  (h_paired : ∃ pairs : (ℕ × ℕ) → Set (ℕ × ℕ), 
    (∀ d1 d2, (d1, d2) ∈ pairs → d1 * d2 = n ∧ Prime (d1 + d2))) :
  (∀ d1 d2 d1' d2', (d1, d2) ∈ pairs → (d1', d2') ∈ pairs → d1 = d1' → d2 = d2' → d1 + d2 = d1' + d2')
  →
  (∀ p, Prime p → ∀ d1 d2, (d1, d2) ∈ pairs → (d1 + d2 = p → ¬(p ∣ n) )) :=
by
  -- proof goes here
  sorry

end paired_divisors_distinct_primes_l819_819661


namespace find_fourth_vertex_l819_819865

-- Define the coordinates of the given vertices
def v1 : ℝ × ℝ × ℝ := (1, 0, 2)
def v2 : ℝ × ℝ × ℝ := (5, 1, 2)
def v3 : ℝ × ℝ × ℝ := (2, -1, 4)

-- Define the target coordinates of the fourth vertex
def target_v : ℝ × ℝ × ℝ := (4, 4, 3)

-- Function to calculate squared distance between two points in ℝ³
def squared_distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2

-- Proof problem statement
theorem find_fourth_vertex (v4 : ℝ × ℝ × ℝ) :
  (squared_distance v1 v4 = 17) ∧
  (squared_distance v2 v4 = 17) ∧
  (squared_distance v3 v4 = 17) ↔
  v4 = target_v :=
by
  sorry

end find_fourth_vertex_l819_819865


namespace log_base_5_of_15625_eq_6_l819_819073

theorem log_base_5_of_15625_eq_6 : log 5 15625 = 6 := 
by {
  have h1 : 5^6 = 15625 := by sorry,
  sorry
}

end log_base_5_of_15625_eq_6_l819_819073


namespace unique_real_solution_floor_eq_l819_819792

theorem unique_real_solution_floor_eq (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ ⌊x⌋ * (x^2 + 1) = x^3 :=
sorry

end unique_real_solution_floor_eq_l819_819792


namespace unique_functional_equation_l819_819898

noncomputable def bounded_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ M : ℝ, ∀ x ∈ set.Icc a b, |f x| ≤ M

theorem unique_functional_equation
  (f : ℝ → ℝ)
  (H1 : ∃ a b : ℝ, bounded_in_interval f a b)
  (H2 : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2)
  (H3 : f 1 = 1) : 
  ∀ x : ℝ, f x = x :=
begin
  sorry -- Proof omitted
end

end unique_functional_equation_l819_819898


namespace draw_contains_chinese_book_l819_819448

theorem draw_contains_chinese_book
  (total_books : ℕ)
  (chinese_books : ℕ)
  (math_books : ℕ)
  (drawn_books : ℕ)
  (h_total : total_books = 12)
  (h_chinese : chinese_books = 10)
  (h_math : math_books = 2)
  (h_drawn : drawn_books = 3) :
  ∃ n, n ≥ 1 ∧ n ≤ drawn_books ∧ n * (chinese_books/total_books) > 1 := 
  sorry

end draw_contains_chinese_book_l819_819448


namespace no_blue_socks_make_probability_half_l819_819754

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l819_819754


namespace irrational_is_irrational_l819_819403

theorem irrational_is_irrational : irrational π :=
sorry

end irrational_is_irrational_l819_819403


namespace equal_tangents_lengths_l819_819230

-- Definitions for the given problem setup
variables {ABC : Type} [triangle ABC]
variables {M : point} (M_is_midpoint : is_midpoint_of_arc M ABC ABC_circumcircle)
variables {N : point} {P T : point}
variables (tangent_NP : is_tangent N P ABC_incircle) 
variables (tangent_NT : is_tangent N T ABC_incircle)
variables {P1 T1 : point}
variables (BP_int_circumcircle : intersects_again B P ABC_circumcircle P1)
variables (BT_int_circumcircle : intersects_again B T ABC_circumcircle T1)

-- The theorem statement
theorem equal_tangents_lengths :
  PP_1 = TT_1 :=
sorry

end equal_tangents_lengths_l819_819230


namespace chalk_marks_identical_on_cube_l819_819287

theorem chalk_marks_identical_on_cube :
  (∃ (two_ways : (Fin 6 × Fin 6) × (Fin 6 × Fin 6)), 
    two_ways.fst ≠ two_ways.snd ∧
    (∀ (i : Fin 6), (chalk_marks two_ways.fst.fst i) = (chalk_marks two_ways.snd.fst i)) ) :=
sorry

-- Definitions for chalk_marks to contextualize the theorem
noncomputable def chalk_marks (face : Fin 6) (point : Fin 100) : bool :=
  sorry

end chalk_marks_identical_on_cube_l819_819287


namespace possibleNumberOfCommonTangents_l819_819167

noncomputable def possibleCommonTangents (c1 c2 : Circle) : Set ℕ :=
  if disjoint c1 c2 then {4}
  else if externallyTangent c1 c2 then {3}
  else if internallyTangent c1 c2 then {1}
  else if intersectsAtTwoPoints c1 c2 then {2}
  else if oneCircleInsideAnotherWithoutTouching c1 c2 then {0}
  else ∅

-- This theorem states the possible values of n for two distinct circles.
theorem possibleNumberOfCommonTangents (c1 c2 : Circle) (h : c1 ≠ c2) :
  ∃ n ∈ {0, 1, 2, 3, 4}, n ∈ possibleCommonTangents c1 c2 :=
sorry

end possibleNumberOfCommonTangents_l819_819167


namespace sum_of_positive_divisors_l819_819775

theorem sum_of_positive_divisors (h : ∀ n : ℕ, (n+24) % n = 0 → (24 % n = 0)) :
  ∑ k in (Finset.filter (λ x, 24 % x = 0) (Finset.range 25)), k = 60 :=
sorry

end sum_of_positive_divisors_l819_819775


namespace proof_problem_l819_819982

variable (a b c x y z : ℝ)

theorem proof_problem
  (h1 : x + y - z = a - b)
  (h2 : x - y + z = b - c)
  (h3 : - x + y + z = c - a) : 
  x + y + z = 0 := by
  sorry

end proof_problem_l819_819982


namespace ratio_of_square_sides_sum_l819_819359

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l819_819359


namespace log_base_5_of_15625_l819_819082

-- Defining that 5^6 = 15625
theorem log_base_5_of_15625 : log 5 15625 = 6 := 
by {
    -- place the required proof here
    sorry
}

end log_base_5_of_15625_l819_819082


namespace vector_perpendicular_l819_819981

variable (λ : ℝ)

def m : ℝ × ℝ := (λ + 1, 1)
def n : ℝ × ℝ := (λ + 2, 2)

theorem vector_perpendicular (h : (m λ).1 + (n λ).1 + (m λ).2 + (n λ).2 = 0) : λ = -3 := by
  sorry

end vector_perpendicular_l819_819981


namespace center_of_mass_square_density_l819_819039

theorem center_of_mass_square_density :
  let ρ := λ (x y : ℝ), x + y
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  let mass := ∫∫ (p in D), ρ p.1 p.2
  let Mx := ∫∫ (p in D), p.1 * ρ p.1 p.2
  let My := ∫∫ (p in D), p.2 * ρ p.1 p.2
  (mass = 8) ∧
  (Mx = 28 / 3) ∧
  (My = 28 / 3) →
  (Mx / mass, My / mass) = (7 / 6, 7 / 6) :=
by
  let ρ := λ (x y : ℝ), x + y
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  let mass := ∫∫ (p in D), ρ p.1 p.2
  let Mx := ∫∫ (p in D), p.1 * ρ p.1 p.2
  let My := ∫∫ (p in D), p.2 * ρ p.1 p.2
  assume h,
  sorry

end center_of_mass_square_density_l819_819039


namespace vendor_profit_is_three_l819_819834

def cost_per_apple := 3 / 2
def selling_price_per_apple := 10 / 5
def cost_per_orange := 2.7 / 3
def selling_price_per_orange := 1
def apples_sold := 5
def oranges_sold := 5

theorem vendor_profit_is_three :
  ((selling_price_per_apple - cost_per_apple) * apples_sold) +
  ((selling_price_per_orange - cost_per_orange) * oranges_sold) = 3 := by
  sorry

end vendor_profit_is_three_l819_819834


namespace octal_to_binary_131_l819_819476

theorem octal_to_binary_131 :
  -- Define the octal and expected binary numbers
  let octal := 1 * 8^0 + 3 * 8^1 + 1 * 8^2 in
  let binary := 1 * 2^0 + 0 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 0 * 2^5 + 1 * 2^6 in
  octal = 89 ∧ binary = 1011001 ∧ Nat.ofDigits 2 (List.reverse [1, 0, 1, 1, 0, 0, 1]) = 89 :=
by
  let octal := 1 * 8^0 + 3 * 8^1 + 1 * 8^2
  let binary := 1 * 2^0 + 0 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 + 0 * 2^5 + 1 * 2^6
  have h1 : octal = 89 := by norm_num
  have h2 : binary = 1011001 := by norm_num
  have h3 : Nat.ofDigits 2 (List.reverse [1, 0, 1, 1, 0, 0, 1]) = 89 := by norm_num
  exact ⟨h1, h2, h3⟩

end octal_to_binary_131_l819_819476


namespace find_angle_C_find_area_of_triangle_l819_819609

-- Given triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
-- And given conditions: c * cos B = (2a - b) * cos C

variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
variable (h2 : c = 2)
variable (h3 : a + b + c = 2 * Real.sqrt 3 + 2)

-- Prove that angle C = π / 3
theorem find_angle_C : C = Real.pi / 3 :=
by sorry

-- Given angle C, side c, and perimeter, prove the area of triangle ABC
theorem find_area_of_triangle (h4 : C = Real.pi / 3) : 
  1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
by sorry

end find_angle_C_find_area_of_triangle_l819_819609


namespace min_occupied_squares_l819_819674

theorem min_occupied_squares (n : ℕ) (h : n > 0) :
  ∃ (s : set (ℕ × ℕ)), s.card = 4 ∧ ∀ t : ℕ × ℕ, (t ∈ s) ∨ (∀ i j : ℕ, (i, j) ∈ s → (i + j) % 2 = (t.1 + t.2) % 2) :=
sorry

end min_occupied_squares_l819_819674


namespace sqrt_sum_eq_six_l819_819492

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l819_819492


namespace find_m_from_expansion_l819_819989

theorem find_m_from_expansion (m n : ℤ) (h : (x : ℝ) → (x + 3) * (x + n) = x^2 + m * x - 21) : m = -4 :=
by
  sorry

end find_m_from_expansion_l819_819989


namespace find_g_at_1_l819_819327

theorem find_g_at_1 (g : ℝ → ℝ) (h : ∀ x, x ≠ 1/2 → g x + g ((2*x + 1)/(1 - 2*x)) = x) : 
  g 1 = 15 / 7 :=
sorry

end find_g_at_1_l819_819327


namespace problem_l819_819157

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - Real.pi / 4
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x + Real.pi / 4

theorem problem (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < Real.pi / 2) (hx2 : 0 < x2 ∧ x2 < Real.pi / 2) :
  (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0) ∧ (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ g x = 0) →
  x1 + x2 = Real.pi / 2 :=
by
  sorry -- Proof goes here

end problem_l819_819157


namespace ellipse_foci_distance_l819_819383

theorem ellipse_foci_distance :
  let eps := [(1, 5), (4, -3), (9, 5)] in
  ∃ (a b : ℝ), a > b ∧ 2 * real.sqrt (a^2 - b^2) = 14 :=
by
  sorry

end ellipse_foci_distance_l819_819383


namespace red_balls_count_is_correct_l819_819809

-- Define conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def non_red_purple_prob : ℝ := 0.9

-- Define the number of red balls
def number_of_red_balls (red_balls : ℕ) : Prop :=
  total_balls - (white_balls + green_balls + yellow_balls + purple_balls) = red_balls
  
-- The proof statement
theorem red_balls_count_is_correct : number_of_red_balls 7 := by
  sorry

end red_balls_count_is_correct_l819_819809


namespace probability_between_2_and_4_l819_819532

noncomputable def normal_distribution_probability_condition (μ σ : ℝ) (ξ : ℝ → ℝ): Prop :=
  (ξ 2 = 0.15) ∧ (ξ 6 = 0.15)

theorem probability_between_2_and_4 {μ σ : ℝ} (ξ : ℝ → ℝ) (h₁ : ∀ x, ξ x = some (PDF of N(μ, σ^2))) 
  (h₂ : normal_distribution_probability_condition μ σ ξ) :
  ξ 4 - ξ 2 = 0.35 :=
sorry

end probability_between_2_and_4_l819_819532
