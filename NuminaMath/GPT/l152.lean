import Mathlib

namespace triangle_is_right_triangle_l152_152445

theorem triangle_is_right_triangle (a b c : ℕ) (h_ratio : a = 3 * (36 / 12)) (h_perimeter : 3 * (36 / 12) + 4 * (36 / 12) + 5 * (36 / 12) = 36) :
  a^2 + b^2 = c^2 :=
by
  -- sorry for skipping the proof.
  sorry

end triangle_is_right_triangle_l152_152445


namespace sam_bought_cards_l152_152971

-- Define the initial number of baseball cards Dan had.
def dan_initial_cards : ℕ := 97

-- Define the number of baseball cards Dan has after selling some to Sam.
def dan_remaining_cards : ℕ := 82

-- Prove that the number of baseball cards Sam bought is 15.
theorem sam_bought_cards : (dan_initial_cards - dan_remaining_cards) = 15 :=
by
  sorry

end sam_bought_cards_l152_152971


namespace arcsin_neg_one_l152_152961

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end arcsin_neg_one_l152_152961


namespace no_two_obtuse_angles_in_triangle_l152_152864

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l152_152864


namespace at_least_two_zeros_l152_152485

noncomputable def f (x : ℝ) : ℝ := (sorry : ℝ -> ℝ)  -- Given function f

theorem at_least_two_zeros :
  (ContinuousOn f (Set.Icc 0 π)) ∧
  (∫ x in 0..π, f x = 0) ∧
  (∫ x in 0..π, f x * Real.cos x = 0) →
  ∃ a b ∈ Set.Ioo 0 π, a ≠ b ∧ f a = 0 ∧ f b = 0 :=
sorry

end at_least_two_zeros_l152_152485


namespace carrots_picked_by_mother_l152_152583

-- Define the conditions
def faye_picked : ℕ := 23
def good_carrots : ℕ := 12
def bad_carrots : ℕ := 16

-- Define the problem of the total number of carrots
def total_carrots : ℕ := good_carrots + bad_carrots

-- Define the mother's picked carrots
def mother_picked (total_faye : ℕ) (total : ℕ) := total - total_faye

-- State the theorem
theorem carrots_picked_by_mother (faye_picked : ℕ) (total_carrots : ℕ) : mother_picked faye_picked total_carrots = 5 := by
  sorry

end carrots_picked_by_mother_l152_152583


namespace probability_m_ge_six_probability_m_odd_ne_probability_m_even_l152_152875

-- Definitions based on conditions:
def tetrahedral_faces : Set ℕ := {1, 2, 3, 5}
def all_possible_outcomes : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ tetrahedral_faces ∧ y ∈ tetrahedral_faces }
def m (outcome : ℕ × ℕ) : ℕ := outcome.1 + outcome.2

-- Statement for part (1)
theorem probability_m_ge_six :
  (set.count (λ outcome, 6 ≤ m outcome) all_possible_outcomes).toReal / (set.count id all_possible_outcomes).toReal = 1 / 2 := 
sorry

-- Statements for part (2)
theorem probability_m_odd_ne_probability_m_even :
  (set.count (λ outcome, m outcome % 2 = 1) all_possible_outcomes).toReal / (set.count id all_possible_outcomes).toReal ≠
  (set.count (λ outcome, m outcome % 2 = 0) all_possible_outcomes).toReal / (set.count id all_possible_outcomes).toReal :=
sorry

end probability_m_ge_six_probability_m_odd_ne_probability_m_even_l152_152875


namespace log_a_b_is_integer_probability_l152_152850

theorem log_a_b_is_integer_probability :
  let a := {2, 2^2, 2^3, ..., 2^{30}},
      b := {2, 2^2, 2^3, ..., 2^{30}},
      pairs := (a × b).filter (λ p, p.1 ≠ p.2),
      valid_pairs := pairs.filter (λ p, ∃ x y, p.1 = 2^x ∧ p.2 = 2^y ∧ x ∣ y) in
  ↑(valid_pairs.length) / ↑(pairs.length) = 86 / 435 :=
begin
  sorry
end

end log_a_b_is_integer_probability_l152_152850


namespace positive_slope_of_asymptote_l152_152114

-- Definitions based on the given problem conditions
def foci_A := (2, 3)
def foci_B := (8, 3)
def hyperbola_eq (x y : ℝ) : ℝ := 
  real.sqrt ((x - foci_A.1)^2 + (y - foci_A.2)^2) - 
  real.sqrt ((x - foci_B.1)^2 + (y - foci_A.2)^2)

-- Formalizing the problem as an assertion to prove
theorem positive_slope_of_asymptote :
  (∃ x y : ℝ, hyperbola_eq x y = 5) →
  ∃ m : ℝ, m = real.sqrt 11 / 5 ∧ 
            (∀ x y : ℝ, 
                hyperbola_eq x y = 5 → 
                slope_of_asymptote (x, y) = m) := 
sorry

end positive_slope_of_asymptote_l152_152114


namespace find_unknown_number_l152_152565

theorem find_unknown_number (x : ℝ) (h : (2 / 3) * x + 6 = 10) : x = 6 :=
  sorry

end find_unknown_number_l152_152565


namespace least_integer_condition_l152_152149

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l152_152149


namespace mean_of_set_with_median_l152_152954

theorem mean_of_set_with_median :
  ∀ (m : ℝ), (m + 8 = 15) → (mean_of_set {m, m + 7, m + 8, m + 12, m + 20} = 16.4) := by
  sorry

noncomputable def mean_of_set (s : finset ℝ) : ℝ :=
  s.sum / s.card

end mean_of_set_with_median_l152_152954


namespace abs_nested_evaluation_l152_152001

theorem abs_nested_evaluation (x : ℝ) (h : x < -3) : |2 - |2 + x|| = -4 - x :=
by
  sorry

end abs_nested_evaluation_l152_152001


namespace remainder_of_M_div_by_51_is_zero_l152_152736

open Nat

noncomputable def M := 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950

theorem remainder_of_M_div_by_51_is_zero :
  M % 51 = 0 :=
sorry

end remainder_of_M_div_by_51_is_zero_l152_152736


namespace sin_phase_shift_l152_152455

theorem sin_phase_shift :
  ∀ x : ℝ, sin (2 * (x + 3 * Real.pi / 8)) = sin (2 * x - 3 * Real.pi / 4) :=
by
  sorry

end sin_phase_shift_l152_152455


namespace expected_successes_in_10_trials_l152_152841

noncomputable def prob_success (p : ℝ) (n : ℕ) : ℝ := n * p

theorem expected_successes_in_10_trials :
  let p := (1 : ℝ) - ((2 / 3) * (2 / 3))
  let n := 10
  in prob_success p n = 50 / 9 :=
by
  let p := (1 : ℝ) - ((2 / 3) * (2 / 3))
  let n := 10
  show prob_success p n = 50 / 9
  sorry

end expected_successes_in_10_trials_l152_152841


namespace find_new_curve_l152_152645

-- Define the given curve equation
noncomputable def given_curve (theta : ℝ) : ℝ := 5 * real.sqrt 3 * real.cos theta - 5 * real.sin theta

-- Define the new curve equation to be proven
noncomputable def new_curve (theta : ℝ) : ℝ := 10 * real.cos (theta - π / 6)

-- State the problem
theorem find_new_curve (theta : ℝ) :
    symmetric_polar_axis (given_curve theta) →
    (∀ rho, new_curve theta = rho) := sorry

def symmetric_polar_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ θ ρ, f θ ρ = f (-θ) ρ


end find_new_curve_l152_152645


namespace dice_probability_l152_152852

theorem dice_probability :
  (∑ i in finset.range 1 (7), ∑ j in finset.range 1 (7), (if (i + j = 4 ∨ i + j = 9 ∨ i + j = 8) then 1 else 0)) / 36 = 1 / 3 :=
by
  sorry

end dice_probability_l152_152852


namespace length_of_QS_l152_152101

-- Definitions based on the conditions provided
variable (Q R S : Type) [metric_space Q] [normed_group Q] [normed_space ℝ Q]
variable (QR RS : ℝ)
variable (angle_R : Q)
def right_angle (Q R S : Type) := Q → R → S
def cos_R (R : Q) : ℝ := 3 / 5
def side_RS : ℝ := 10
def adjacent_side_to_angle_R (QR : Q) := QR

-- Theorem statement to prove the length of QS
theorem length_of_QS (RS : ℝ) (QR : ℝ) (cos_R : ℝ) (right_angle : Q → R → S) (adjacent_side_to_angle_R : Q)
  (hRS : RS = 10) (hcos_R : cos_R = 3 / 5) (hQR : QR / RS = cos_R) : ∃ QS : ℝ, QS = 8 := by
sorry

end length_of_QS_l152_152101


namespace points_on_intersecting_lines_l152_152567

def clubsuit (a b : ℝ) := a^3 * b - a * b^3

theorem points_on_intersecting_lines (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = y ∨ x = -y) := 
by
  sorry

end points_on_intersecting_lines_l152_152567


namespace prob_three_correct_is_one_sixth_l152_152252

open Finset

noncomputable def prob_three_correct_out_of_five (correctness : Fin 5 → Prop) : ℚ :=
  let P := { σ : Perm (Fin 5) // ∃ (S : Finset (Fin 5)),
              S.card = 3 ∧ (∀ x ∈ S, σ x = x) ∧ (∀ x ∉ S, σ x ≠ x)} in
  (card P : ℚ) / (card (Perm (Fin 5)) : ℚ)

theorem prob_three_correct_is_one_sixth :
  prob_three_correct_out_of_five (λ i, true) = 1 / 6 := 
sorry

end prob_three_correct_is_one_sixth_l152_152252


namespace value_a6_l152_152757

noncomputable def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n - a (n - 1) = n - 1

theorem value_a6 : ∃ a : ℕ → ℕ, seq a ∧ a 6 = 16 := by
  sorry

end value_a6_l152_152757


namespace problem_A_l152_152865

noncomputable theory

open EuclideanGeometry

-- Definition of skew lines and perpendicular line passing through a point
variables {m n : Line ℝ^3} [Skew m n] (P : Point ℝ^3)

theorem problem_A (h_skew : Skew m n) (hP : Point ℝ^3) :
  ∃ l : Line ℝ^3, ∀ p ∈ Line.points l, Line.perpendicular m l ∧ Line.perpendicular n l :=
sorry

end problem_A_l152_152865


namespace inverse_sum_correct_l152_152225

noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 2 * x - 2 else real.sqrt (2 * x)

noncomputable def f_inv (y : ℝ) : ℝ :=
  if y < 4 then (y + 2) / 2 else (y^2) / 2

theorem inverse_sum_correct : f_inv (-4) + f_inv (-2) + f_inv (0) + f_inv (2) + f_inv (4) + f_inv (6) = 8 :=
by
  sorry

end inverse_sum_correct_l152_152225


namespace find_multiple_l152_152822

theorem find_multiple (a b m : ℤ) (h1 : b = 7) (h2 : b - a = 2) 
  (h3 : a * b = m * (a + b) + 11) : m = 2 :=
by {
  sorry
}

end find_multiple_l152_152822


namespace eulerian_circuits_l152_152904

-- Representing the degrees of vertices in the graph as assumptions
variables {deg_A deg_B deg_C deg_D deg_E : ℕ}
axiom deg_A_def : deg_A = 6
axiom deg_B_def : deg_B = 4
axiom deg_C_def : deg_C = 4
axiom deg_D_def : deg_D = 4
axiom deg_E_def : deg_E = 2

-- Function to compute faculty of (deg(v) - 1)
def factorial (n : ℕ) : ℕ :=
  nat.rec_on n 1 (λ (n' : ℕ) (r : ℕ), (n'+1) * r)

-- Prove that the number of Eulerian circuits equals 264
theorem eulerian_circuits : (factorial (deg_A - 1)) * (factorial (deg_B - 1)) * 
                            (factorial (deg_C - 1)) * (factorial (deg_D - 1)) * 
                            (factorial (deg_E - 1)) = 264 := 
by {sorry}

end eulerian_circuits_l152_152904


namespace birds_find_more_than_half_millet_on_day_3_l152_152766

noncomputable def day_millet_amount (n : ℕ) :=
  0.40 * (1 - (0.50)^n)

noncomputable def total_seeds_amount (n : ℕ) :=
  1.0 -- Since each day Millie adds 1 quart of seeds (0.20 millet + 0.80 other)

theorem birds_find_more_than_half_millet_on_day_3 :
  (day_millet_amount 3) > (total_seeds_amount 3) / 2 :=
by
  sorry

end birds_find_more_than_half_millet_on_day_3_l152_152766


namespace solution_set_ln_inequality_l152_152249

theorem solution_set_ln_inequality (x : ℝ) :
  log (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end solution_set_ln_inequality_l152_152249


namespace fruit_seller_profit_percentage_l152_152892

theorem fruit_seller_profit_percentage :
  let CP := 6 / (1 - 0.15) in
  let SP_new := 7.411764705882353 in
  ∃ (P : ℝ), SP_new = CP * (1 + P / 100) ∧ P = 5 :=
sorry

end fruit_seller_profit_percentage_l152_152892


namespace inverse_proof_l152_152687

theorem inverse_proof (x : ℝ) :
  (x > 1 → x^2 - 2x + 3 > 0) ↔ (x^2 - 2x + 3 > 0 → x > 1) :=
sorry

end inverse_proof_l152_152687


namespace parallelogram_area_l152_152682

theorem parallelogram_area (A B C D : (ℝ × ℝ)) 
  (hA: A = (0, 0)) 
  (hB: B = (4, 0)) 
  (hC: C = (5, 10)) 
  (hD: D = (1, 10)) : 
  let base := dist A B in
  let height := (C.2 - A.2) in
  let area := base * height in
  area = 40 := 
by
  sorry

end parallelogram_area_l152_152682


namespace next_palindromic_prime_sum_l152_152450

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c, c.to_nat - '0'.to_nat)).sum

theorem next_palindromic_prime_sum :
  ∃ (y : ℕ), y > 1991 ∧ is_palindrome y ∧ is_prime y ∧ sum_of_digits y = 29 := by
  use 2999
  sorry

end next_palindromic_prime_sum_l152_152450


namespace find_a₁_l152_152026

-- Given conditions
variables {q : ℕ} {n : ℕ} {S : ℕ → ℕ} {a₁ : ℕ}

-- Specific values for the given problem
axiom q_eq_3 : q = 3
axiom S4_eq_80 : S 4 = 80

-- The formula for the sum of the first n terms of a geometric series
def geom_series_sum (a₁ q n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

-- Problem statement
theorem find_a₁ : geom_series_sum a₁ q 4 = S 4 → a₁ = 2 :=
  by
  -- Prove the inferred equality from the given conditions
  rw [q_eq_3, S4_eq_80]
  sorry

end find_a₁_l152_152026


namespace gayeong_circle_radius_l152_152731

def radius_Kiyoung (diameter_Kiyoung : ℝ) : ℝ := diameter_Kiyoung / 2

def radius_Gayeong (r_Kiyoung : ℝ) : ℝ := r_Kiyoung / 4

theorem gayeong_circle_radius :
  ∀ diameter_Kiyoung : ℝ,
  diameter_Kiyoung = 80 → 
  radius_Gayeong (radius_Kiyoung diameter_Kiyoung) = 10 :=
by
  intros diameter_Kiyoung h
  rw [radius_Gayeong, radius_Kiyoung, h]
  norm_num
  sorry

end gayeong_circle_radius_l152_152731


namespace any_nat_as_difference_or_element_l152_152826

noncomputable def seq (q : ℕ → ℕ) : Prop :=
∀ n, q n < 2 * n

theorem any_nat_as_difference_or_element (q : ℕ → ℕ) (h_seq : seq q) (m : ℕ) :
  (∃ k, q k = m) ∨ (∃ k l, q l - q k = m) :=
sorry

end any_nat_as_difference_or_element_l152_152826


namespace isosceles_triangle_equiv_l152_152410

theorem isosceles_triangle_equiv (A B C : Type) [Triangle A] [Angle B] [Angle C] :
  (∀ (a b c : Triangle), Angle a = Angle b → Triangle a = Triangle b) ↔
  (∀ (a b c : Triangle), (Angle a ≠ Angle b → Triangle a ≠ Triangle b) ∧ (Angle a = Angle b → Triangle a = Triangle b)) :=
by
  sorry

end isosceles_triangle_equiv_l152_152410


namespace median_is_2005_5_l152_152858

noncomputable def median_of_special_list : Rat :=
let list := (List.range (2050 + 1)).append (List.range(2050 + 1)).map (λ x => x * x) in
(list.nth_le 2049 (by simp)).enslave + (list.nth_le 2050 (by simp)) / 2

theorem median_is_2005_5 :
  median_of_special_list = 2005.5 := 
by
  sorry

end median_is_2005_5_l152_152858


namespace function_has_extremum_points_l152_152889

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_has_extremum_points (h_cont : Continuous f) (h_ineq : ∀ x : ℝ, f(x^2) - f(x)^2 ≥ (1 / 4)) : 
  ∃ x_ext : ℝ, (∀ ε > 0, ∃ η > 0, ∀ x ∈ Set.univ, abs (x - x_ext) < η → f(x_ext) ≥ f(x)) ∨
                (∀ ε > 0, ∃ η > 0, ∀ x ∈ Set.univ, abs (x - x_ext) < η → f(x_ext) ≤ f(x)) :=
sorry

end function_has_extremum_points_l152_152889


namespace susan_surveyed_total_people_l152_152787

noncomputable def percentage_blind := 78.4 / 100
noncomputable def percentage_cant_hear := 53.2 / 100
noncomputable def incorrect_people := 33

theorem susan_surveyed_total_people (total_people_surveyed : ℕ) :
  let total_blind := incorrect_people / percentage_cant_hear in
  let total_surveyed := total_blind / percentage_blind in
  total_people_surveyed = total_surveyed :=
by
  sorry

end susan_surveyed_total_people_l152_152787


namespace shaded_area_equilateral_triangle_l152_152708

open_locale real

theorem shaded_area_equilateral_triangle :
  ∀ (ABC DEF : Triangle) (A B C P D E F : Point)
    (h1 : IsEquilateral ABC)
    (h2 : Midpoint B C D)
    (h3 : Midpoint C A E)
    (h4 : Midpoint A B F),
  let DEF_area := (triangle_area ABC) / 4,
      shaded_area := (5 / 24) * (triangle_area ABC) in
    (area_of_shaded_region ABC DEF P = shaded_area) :=
begin
  sorry
end

end shaded_area_equilateral_triangle_l152_152708


namespace problem_solution_l152_152280

section
variables {f : ℝ → ℝ}

-- Conditions
hypothesis h_odd : ∀ x, f (-x) = -f x
hypothesis h_symmetric : ∀ x, f (2 - x) = f (2 + x)
hypothesis h_def : ∀ x, (0 < x ∧ x ≤ 2) → f x = x + 1

-- Conclusion
theorem problem_solution : f (-100) + f (-101) = 2 :=
sorry
end

end problem_solution_l152_152280


namespace b_arithmetic_sequence_a_max_min_l152_152756

-- Define the sequence aₙ
def a : ℕ → ℝ 
| 1       := 3 / 5
| (n + 1) := 2 - 1 / a n

-- Define the sequence bₙ
def b (n : ℕ) : ℝ := 1 / (a n - 1)

theorem b_arithmetic_sequence : ∀ n : ℕ, b (n + 1) - b n = 1 :=
by
  sorry

theorem a_max_min : 
  let max_a := 3
  let min_a := -1 
  ∀ n : ℕ, (a n ≤ max_a) ∧ (a n ≥ min_a) :=
by
  sorry

end b_arithmetic_sequence_a_max_min_l152_152756


namespace abs_diff_of_solutions_l152_152592

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l152_152592


namespace ZW_length_eq_ten_l152_152357

noncomputable def ZW_length : ℝ := 
  let X := (0, 0)
  let Y := (9, 0)
  let Z := (0, 12)
  let W : (ℝ × ℝ) := sorry  -- precise value or geometric description
  let V : (ℝ × ℝ) := sorry  -- precise value or geometric description
  sorry  -- calculation or proof of ZW length

-- Statement in Lean 4
theorem ZW_length_eq_ten (X Y Z W V : ℝ × ℝ)
  (hXYZ : ∠ Z = 90°)
  (hXZ : dist X Z = 9)
  (hZY : dist Z Y = 12)
  (hZVW : ∠ ZVW = 90°)
  (hVW : dist V W = 6)
  : dist Z W = 10 := 
sorry

end ZW_length_eq_ten_l152_152357


namespace map_distance_l152_152838

-- definition of given conditions
def actual_distance_km : ℕ := 5
def scale_factor : ℕ := 250_000
def km_to_cm (km : ℕ) : ℕ := km * 100_000

-- theorem statement
theorem map_distance (d : ℕ) (s : ℕ) :
  s = 250_000 → d = 5 → 
  let map_distance_cm := (km_to_cm d) / s in
  map_distance_cm = 2 := 
by
  intros h_scale h_distance
  simp [km_to_cm, h_scale, h_distance]
  sorry

end map_distance_l152_152838


namespace trapezoid_area_l152_152802

theorem trapezoid_area (ABCD : Type) [trapezoid ABCD] {O : Point}
  (AC BD : Diagonals ABCD) (p q : ℝ)
  (area_AOB : area (triangle ABCD.A O ABCD.B) = p^2)
  (area_COD : area (triangle ABCD.C O ABCD.D) = q^2) :
  area ABCD = (p + q)^2 := 
sorry

end trapezoid_area_l152_152802


namespace min_value_of_ratio_max_value_of_product_min_value_of_sqrt_sum_min_value_of_square_sum_l152_152387

-- Problem 1: Prove the minimum value of \(\frac{b}{a} + \frac{2}{b}\) is \(3\).
theorem min_value_of_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (∃ x, x = (b / a + 2 / b) ∧ x ≥ 3) := 
by
  sorry

-- Problem 2: Prove the maximum value of \(ab\) is \(1\).
theorem max_value_of_product (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (∃ x, x = (a * b) ∧ x ≤ 1) := 
by
  sorry

-- Problem 3: Prove the incorrectness of the minimum value of \(\sqrt{a} + \sqrt{b}\) being \(2\).
theorem min_value_of_sqrt_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  ¬(∃ x, x = (sqrt a + sqrt b) ∧ x = 2) := 
by
  sorry

-- Problem 4: Prove the minimum value of \(a^2 + b^2\) is \(2\).
theorem min_value_of_square_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (∃ x, x = (a^2 + b^2) ∧ x ≥ 2) := 
by
  sorry

end min_value_of_ratio_max_value_of_product_min_value_of_sqrt_sum_min_value_of_square_sum_l152_152387


namespace final_result_l152_152489

-- Define a rectangular sheet ABCD and points E, F on AB and CD respectively
variables (A B C D E F : Type)
-- Conditions
variables [rectangular_sheet ABCD]
variables (BE GT CF : Prop)
variables (BCFE_folded_over_EF : maps_to (C', B'))
variables (angle_eqn : angle (AB'C') = angle (B'EA))
variables (AB'_eq : measure_length AB' = 7)
variables (BE_eq : measure_length BE = 15)

-- Define the area of the sheet
noncomputable def area_ABCD : Type := 
  -- Using the given algebraic forms and calculations
  compute_area ABCD 145 11

-- Prove the resulting sum a + b + c
theorem final_result : Σ a b c, a + b + c = 156 :=
by
  sorry

end final_result_l152_152489


namespace sara_spent_on_salad_l152_152086

theorem sara_spent_on_salad: 
  ∀ (cost_hotdog cost_total cost_salad : ℝ),
  cost_hotdog = 5.36 →
  cost_total = 10.46 →
  cost_salad = cost_total - cost_hotdog →
  cost_salad = 5.10 := 
by
  intros cost_hotdog cost_total cost_salad h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sara_spent_on_salad_l152_152086


namespace number_of_students_present_l152_152344

-- Definitions based on the problem statement
def total_students := 35
def girl_percentage := 0.6
def boy_absent := 2
def girl_absent := 1
def girl_present_percentage := 0.625

-- Main theorem statement with conditions and conclusion
theorem number_of_students_present (N G B : ℕ) (h1 : G = (girl_percentage * N).toNat) 
    (h2 : (G - girl_absent) = (girl_present_percentage * (N - (boy_absent + girl_absent)).toNat)) : 
    (G = 21 ∧ B = 14) :=
begin
  sorry -- proof is skipped
end

end number_of_students_present_l152_152344


namespace magician_performances_l152_152516

theorem magician_performances (performances : ℕ) (p_no_reappear : ℚ) (p_two_reappear : ℚ) :
  performances = 100 → p_no_reappear = 1/10 → p_two_reappear = 1/5 → 
  let num_no_reappear := performances * p_no_reappear in
  let num_two_reappear := performances * p_two_reappear in
  let num_one_reappear := performances - num_no_reappear - num_two_reappear in
  let total_reappeared := num_one_reappear + 2 * num_two_reappear in
  total_reappeared = 110 :=
by
  intros h1 h2 h3
  let num_no_reappear := performances * p_no_reappear
  let num_two_reappear := performances * p_two_reappear
  let num_one_reappear := performances - num_no_reappear - num_two_reappear
  let total_reappeared := num_one_reappear + 2 * num_two_reappear
  have h4 : num_no_reappear = 10 := by sorry
  have h5 : num_two_reappear = 20 := by sorry
  have h6 : num_one_reappear = 70 := by sorry
  have h7 : total_reappeared = 110 := by sorry
  exact h7

end magician_performances_l152_152516


namespace quintuple_sum_not_less_than_l152_152235

theorem quintuple_sum_not_less_than (a : ℝ) : 5 * (a + 3) ≥ 6 :=
by
  -- Insert proof here
  sorry

end quintuple_sum_not_less_than_l152_152235


namespace inverse_function_evaluation_l152_152411

theorem inverse_function_evaluation :
  ∀ (f : ℕ → ℕ) (f_inv : ℕ → ℕ),
    (∀ y, f_inv (f y) = y) ∧ (∀ x, f (f_inv x) = x) →
    f 4 = 7 →
    f 6 = 3 →
    f 3 = 6 →
    f_inv (f_inv 6 + f_inv 7) = 4 :=
by
  intros f f_inv hf hf1 hf2 hf3
  sorry

end inverse_function_evaluation_l152_152411


namespace center_square_side_length_l152_152105

noncomputable def side_length_center_square (total_side_length : ℕ) (total_area : ℕ) : ℕ :=
  let area_L_shapes := (4 / 5 : ℚ) * total_area
  let area_center_square := total_area - area_L_shapes
  let side_length := Real.sqrt area_center_square
  (Real.toNat side_length)

theorem center_square_side_length
  (total_side_length : ℕ)
  (total_area : ℕ)
  (h_total_area : total_side_length * total_side_length = total_area)
  (h_L_shapes : (4 / 5 : ℚ) * total_area = 11520) :
  side_length_center_square total_side_length total_area = 60 :=
by
  sorry

end center_square_side_length_l152_152105


namespace remainder_t4_mod7_l152_152973

def T : ℕ → ℕ
| 0 => 0 -- Not used
| 1 => 6
| n+1 => 6 ^ (T n)

theorem remainder_t4_mod7 : (T 4 % 7) = 6 := by
  sorry

end remainder_t4_mod7_l152_152973


namespace min_triangle_area_l152_152632

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def circle_with_diameter_passing_origin (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  center.1^2 + center.2^2 = d / 4

theorem min_triangle_area (A B : ℝ × ℝ)
    (hA : hyperbola A.1 A.2)
    (hB : hyperbola B.1 B.2)
    (hc : circle_with_diameter_passing_origin A B) : 
    ∃ (S : ℝ), S = 2 :=
sorry

end min_triangle_area_l152_152632


namespace toms_total_score_l152_152845

def regular_enemy_points : ℕ := 10
def elite_enemy_points : ℕ := 25
def boss_enemy_points : ℕ := 50

def regular_enemy_bonus (kills : ℕ) : ℚ :=
  if 100 ≤ kills ∧ kills < 150 then 0.50
  else if 150 ≤ kills ∧ kills < 200 then 0.75
  else if kills ≥ 200 then 1.00
  else 0.00

def elite_enemy_bonus (kills : ℕ) : ℚ :=
  if 15 ≤ kills ∧ kills < 25 then 0.30
  else if 25 ≤ kills ∧ kills < 35 then 0.50
  else if kills >= 35 then 0.70
  else 0.00

def boss_enemy_bonus (kills : ℕ) : ℚ :=
  if 5 ≤ kills ∧ kills < 10 then 0.20
  else if kills ≥ 10 then 0.40
  else 0.00

noncomputable def total_score (regular_kills elite_kills boss_kills : ℕ) : ℚ :=
  let regular_points := regular_kills * regular_enemy_points
  let elite_points := elite_kills * elite_enemy_points
  let boss_points := boss_kills * boss_enemy_points
  let regular_total := regular_points + regular_points * regular_enemy_bonus regular_kills
  let elite_total := elite_points + elite_points * elite_enemy_bonus elite_kills
  let boss_total := boss_points + boss_points * boss_enemy_bonus boss_kills
  regular_total + elite_total + boss_total

theorem toms_total_score :
  total_score 160 20 8 = 3930 := by
  sorry

end toms_total_score_l152_152845


namespace slope_l3_l152_152758

variables {x y : ℝ}

def A := (0, -3 : ℝ)
def l1_eq := 2 * x + 3 * y = 6
def l2_eq := y = 2
def area_triangle_ABC (B C : ℝ × ℝ) : Prop :=
  let h := 2 - (-3 : ℝ)
  let d := h
  1 / 2 * d * (B.1 - C.1) = 10

theorem slope_l3 (B C : ℝ × ℝ) (m : ℝ) :
  (2 * B.1 + 3 * B.2 = 6) →
  (B.2 = 2) →
  (C.2 = 2) →
  area_triangle_ABC B C →
  m = (C.2 - A.2) / (C.1 - A.1) →
  m > 0 →
  m = 5 / 4 := by
  -- Proof omitted
  intros
  sorry

end slope_l3_l152_152758


namespace rectangle_length_is_4_l152_152007

theorem rectangle_length_is_4 (a : ℕ) (s : a = 4) (area_square : ℕ) 
(area_square_eq : area_square = a * a) 
(area_rectangle_eq : area_square = a * 4) : 
4 = a := by
  sorry

end rectangle_length_is_4_l152_152007


namespace sum_of_triangle_areas_l152_152375

theorem sum_of_triangle_areas 
  (ABCD : ℝ) (AD : ℝ) (AB : ℝ) (CD : ℝ)
  (Q1_ratio : ℝ) (Q1D_ratio : ℝ)
  (P_intersection : ℝ → ℝ → ℝ)
  (perpendicular_foot : ℝ → ℝ → ℝ)
  (h_AD_eq_2 : AD = 2) 
  (h_AB_eq_1 : AB = 1) 
  (h_Q1_ratio : Q1_ratio = 2/3) 
  (h_Q1D_ratio : Q1D_ratio = 1/3) 
  (h_sum_areas : ∑ i in (finset.range ∞), (1/2) * Q1_ratio * (sqrt 5 / 5) * (1 / 2^(i - 1)) = (2 * sqrt 5 / 15)) :
  ∑ i in (finset.range ∞), (1/2) * Q1_ratio * (sqrt 5 / 5) * (1 / 2^(i - 1)) = (2 * sqrt 5 / 15) := 
  sorry

end sum_of_triangle_areas_l152_152375


namespace efficiency_ratio_l152_152163

theorem efficiency_ratio (A B : ℝ) (h1 : A ≠ B)
  (h2 : A + B = 1 / 7)
  (h3 : B = 1 / 21) :
  A / B = 2 :=
by
  sorry

end efficiency_ratio_l152_152163


namespace no_club_member_is_fraternity_member_l152_152215

variable (Student : Type) (isHonest : Student → Prop) 
                       (isFraternityMember : Student → Prop) 
                       (isClubMember : Student → Prop)

axiom some_students_not_honest : ∃ x : Student, ¬ isHonest x
axiom all_frats_honest : ∀ y : Student, isFraternityMember y → isHonest y
axiom no_clubs_honest : ∀ z : Student, isClubMember z → ¬ isHonest z

theorem no_club_member_is_fraternity_member : ∀ w : Student, isClubMember w → ¬ isFraternityMember w :=
by sorry

end no_club_member_is_fraternity_member_l152_152215


namespace hydrogen_atoms_in_compound_l152_152505

theorem hydrogen_atoms_in_compound :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let atomic_weight_H := 1.008
  let num_C_atoms := 4
  let num_O_atoms := 2
  let molecular_weight := 88
  let weight_C := num_C_atoms * atomic_weight_C
  let weight_O := num_O_atoms * atomic_weight_O
  let weight_O_C := weight_C + weight_O
  let weight_H := molecular_weight - weight_O_C
  let num_H_atoms := weight_H / atomic_weight_H
  Int.round num_H_atoms = 8 :=
by
  sorry

end hydrogen_atoms_in_compound_l152_152505


namespace magician_performances_l152_152515

theorem magician_performances (performances : ℕ) (p_no_reappear : ℚ) (p_two_reappear : ℚ) :
  performances = 100 → p_no_reappear = 1/10 → p_two_reappear = 1/5 → 
  let num_no_reappear := performances * p_no_reappear in
  let num_two_reappear := performances * p_two_reappear in
  let num_one_reappear := performances - num_no_reappear - num_two_reappear in
  let total_reappeared := num_one_reappear + 2 * num_two_reappear in
  total_reappeared = 110 :=
by
  intros h1 h2 h3
  let num_no_reappear := performances * p_no_reappear
  let num_two_reappear := performances * p_two_reappear
  let num_one_reappear := performances - num_no_reappear - num_two_reappear
  let total_reappeared := num_one_reappear + 2 * num_two_reappear
  have h4 : num_no_reappear = 10 := by sorry
  have h5 : num_two_reappear = 20 := by sorry
  have h6 : num_one_reappear = 70 := by sorry
  have h7 : total_reappeared = 110 := by sorry
  exact h7

end magician_performances_l152_152515


namespace valid_functions_l152_152621

/- Domain and codomain definitions -/
def I := set.Icc (0 : ℝ) 1
def G := { p : ℝ × ℝ | p.1 ∈ I ∧ p.2 ∈ I }
def f (x y : ℝ) := sorry -- f is a mapping we will define below

/- Conditions as Lean hypotheses -/
axiom f_condition1 (x y z : ℝ) (hx : x ∈ I) (hy : y ∈ I) (hz : z ∈ I) :
  f (f x y) z = f x (f y z)
axiom f_condition2 (x y : ℝ) (hx : x ∈ I) (hy : y ∈ I) :
  (f x 1 = x) ∧ (f 1 y = y)
axiom f_condition3 (x y z : ℝ) (hx : x ∈ I) (hy : y ∈ I) (hz : z ∈ I) (k : ℝ) (hk : 0 < k) :
  f (z * x) (z * y) = z ^ k * f x y

/- Functions we need to prove are the only solutions. -/
def f₁ (x y : ℝ) : ℝ :=
  if x ≤ y then x else y

def f₂ (x y : ℝ) : ℝ :=
  x * y

/- Stating the equivalence problem - i.e., both f₁ and f₂ satisfy all initial conditions -/
theorem valid_functions : 
  (∀ (x y : ℝ), x ∈ I → y ∈ I → f x y = f₁ x y ∨ f x y = f₂ x y) :=
sorry

end valid_functions_l152_152621


namespace largest_divisor_of_expression_l152_152289

theorem largest_divisor_of_expression :
  ∃ m : ℕ, m = 2448 ∧ ∀ n : ℕ, n > 0 → (9^(2*n) - 8^(2*n) - 17) % m = 0 :=
by
sorrry

end largest_divisor_of_expression_l152_152289


namespace angles_of_triangle_l152_152404

noncomputable def α : ℝ := acos ((sqrt 2) / (4 * cos (10 * real.pi / 180)))
noncomputable def β : ℝ := acos ((sqrt 6) / (4 * cos (10 * real.pi / 180)))
noncomputable def γ : ℝ := acos (1 / (2 * cos (10 * real.pi / 180)))

theorem angles_of_triangle (α β γ : ℝ) :
  cos α = (sqrt 2) / (4 * cos (10 * real.pi / 180)) ∧
  cos β = (sqrt 6) / (4 * cos (10 * real.pi / 180)) ∧
  cos γ = 1 / (2 * cos (10 * real.pi / 180)) →
  α + β + γ = real.pi :=
by
  intros h
  sorry

end angles_of_triangle_l152_152404


namespace num_sides_regular_polygon_l152_152909

-- Define the perimeter and side length of the polygon
def perimeter : ℝ := 160
def side_length : ℝ := 10

-- Theorem to prove the number of sides
theorem num_sides_regular_polygon : 
  (perimeter / side_length) = 16 := by
    sorry  -- Proof is omitted

end num_sides_regular_polygon_l152_152909


namespace magician_act_reappearance_l152_152519

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l152_152519


namespace arithmetic_geometric_l152_152648

theorem arithmetic_geometric (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, a 1 * r = a 3 ∧ a 3 * r = a 4) :
  a 2 = -6 :=
by sorry

end arithmetic_geometric_l152_152648


namespace sin_2023_closest_value_l152_152472

theorem sin_2023_closest_value :
  ∃ x ∈ { - (Real.sqrt 3) / 2, - (Real.sqrt 2) / 2, (Real.sqrt 2) / 2, (Real.sqrt 3) / 2 },
    abs (Real.sin (2023 * Real.pi / 180) - x) = abs (Real.sin (2023 * Real.pi / 180) - ( - (Real.sqrt 2) / 2)) :=
sorry

end sin_2023_closest_value_l152_152472


namespace three_sides_of_triangle_n_sides_of_triangle_l152_152786

open Real

-- Define the problem for three positive real numbers
theorem three_sides_of_triangle (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by sorry

-- Define the problem for n positive real numbers
theorem n_sides_of_triangle (n : ℕ) (h_n : 3 ≤ n) (a : Fin n.succ → ℝ)
  (h_pos : ∀ i : Fin n.succ, 0 < a i)
  (h_ineq : (∑ i, a i^2)^2 > (n - 1) * ∑ i, a i^4) :
  ∀ (i j k : Fin n.succ), i ≠ j → j ≠ k → i ≠ k → 
  a i + a j > a k ∧ a i + a k > a j ∧ a j + a k > a i := 
by sorry

end three_sides_of_triangle_n_sides_of_triangle_l152_152786


namespace magician_act_reappearance_l152_152518

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l152_152518


namespace regular_vs_diet_sodas_l152_152895

theorem regular_vs_diet_sodas :
  let regular_cola := 67
  let regular_lemon := 45
  let regular_orange := 23
  let diet_cola := 9
  let diet_lemon := 32
  let diet_orange := 12
  let regular_sodas := regular_cola + regular_lemon + regular_orange
  let diet_sodas := diet_cola + diet_lemon + diet_orange
  regular_sodas - diet_sodas = 82 := sorry

end regular_vs_diet_sodas_l152_152895


namespace difference_of_roots_l152_152240

theorem difference_of_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = -8) (h3 : c = 15) :
  let r1 := (-b + Real.sqrt(b ^ 2 - 4 * a * c)) / (2 * a)
  let r2 := (-b - Real.sqrt(b ^ 2 - 4 * a * c)) / (2 * a)
  r1 - r2 = 2 :=
by
  sorry

end difference_of_roots_l152_152240


namespace product_of_triangle_areas_l152_152347

-- Define the parabola equation y^2 = 4x
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the dot product condition
def dot_product_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2) = -4

-- Define the focus of the parabola (F is (1, 0) for y^2 = 4x)
def focus (x y : ℝ) : Prop := (x = 1) ∧ (y = 0)

-- Define the areas of triangles
def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 / 2 * abs (x1 * y2 - y1 * x2)

-- Theorem statement: product of the areas of triangles OFA and OFB is 2
theorem product_of_triangle_areas (x1 y1 x2 y2 : ℝ)
  (H1 : is_on_parabola x1 y1) (H2 : is_on_parabola x2 y2)
  (H3 : dot_product_condition x1 y1 x2 y2) :
  triangle_area 0 0 x1 y1 * triangle_area 0 0 x2 y2 = 2 :=
sorry

end product_of_triangle_areas_l152_152347


namespace triangle_no_two_obtuse_angles_l152_152862

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l152_152862


namespace polynomial_bound_l152_152444

theorem polynomial_bound (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l152_152444


namespace inverse_of_f_at_10_l152_152299

noncomputable def f (x : ℝ) : ℝ := 1 + 3^(-x)

theorem inverse_of_f_at_10 :
  f⁻¹ 10 = -2 :=
sorry

end inverse_of_f_at_10_l152_152299


namespace school_spent_440_l152_152524

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end school_spent_440_l152_152524


namespace quadratic_inequality_range_l152_152012

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ a ∈ Set.Ico 0 4 := 
by
  sorry

end quadratic_inequality_range_l152_152012


namespace restore_digits_l152_152093

theorem restore_digits :
  ∃ (A S I K M O R C U J : ℕ),
    (A, S, I, K, M, O, R, C, U, J) = (5, 4, 9, 2, 3, 7, 1, 0, 8, 6) ∧
    A < I ∧ I < M ∧ M < S ∧ S < K ∧ K < R ∧ R < J ∧ J < O ∧ O < U ∧ U < C ∧
    ∀ n, n ∈ {A, S, I, K, M, O, R, C, U, J} → n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end restore_digits_l152_152093


namespace Jean_spots_l152_152039

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l152_152039


namespace problem_solution_l152_152386

theorem problem_solution :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (m : ℝ)/n = 1 ∧ m + n = 2 :=
by
  have h1 : ∀ b ∈ set.Icc (-9 : ℝ) 27, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 9 * b^2 = (6 * b^2 - 18 * b) * x1^2 := sorry
  exact
    ⟨1, 1, Nat.coprime_one_right _, by norm_cast; exact one_div_one, rfl⟩

end problem_solution_l152_152386


namespace circle_tangent_y_axis_l152_152008

theorem circle_tangent_y_axis (a : ℝ) : ((x y : ℝ) -> (x - a) ^ 2 + (x + 4) ^ 2 = 9) -> (∀ x, (a = x ∨ a = -x)) :=
by
  assume h : (x y : ℝ) -> (x - a) ^ 2 + (x + 4) ^ 2 = 9
  sorry

end circle_tangent_y_axis_l152_152008


namespace restore_original_price_l152_152440

theorem restore_original_price
    (P : ℝ) -- Original price P
    (P > 0) -- P is a positive real number
    (reduced_price : ℝ := 0.85 * P) -- Reduced price = 85% of original price
    : let x_factor : ℝ := 100 / reduced_price
      in (x_factor - 1) * 100 ≈ 17.65 :=
begin
  sorry
end

end restore_original_price_l152_152440


namespace problem1_problem2_l152_152874

theorem problem1 (a : ℝ) (h : 1 < a) : sqrt (a + 1) + sqrt (a - 1) < 2 * sqrt a :=
sorry

theorem problem2 (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
sorry

end problem1_problem2_l152_152874


namespace constant_term_expansion_l152_152873

theorem constant_term_expansion :
  let f := (λ x : ℝ, (x^3 - 1) * (real.sqrt x + 2 / x)^6) in
  f 1 = 180 :=
by
  sorry

end constant_term_expansion_l152_152873


namespace find_expression_l152_152641

-- Define the polynomial and the fact that a and b are roots.
def poly (x : ℝ) := x^2 + 3 * x - 4

-- Assuming a and b are roots of the polynomial
variables (a b : ℝ)
hypothesis h_a_root : poly a = 0
hypothesis h_b_root : poly b = 0

-- Prove that a^2 + 4a + b - 3 = -2 given the above assumptions
theorem find_expression (a b : ℝ) (h_a_root : poly a = 0) (h_b_root : poly b = 0) : a^2 + 4 * a + b - 3 = -2 :=
by sorry

end find_expression_l152_152641


namespace remainder_140_div_k_l152_152606

theorem remainder_140_div_k (k : ℕ) (hk : k > 0) :
  (80 % k^2 = 8) → (140 % k = 2) :=
by
  sorry

end remainder_140_div_k_l152_152606


namespace determine_m_from_quadratic_l152_152293

def is_prime (n : ℕ) := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem determine_m_from_quadratic (x1 x2 m : ℕ) (hx1_prime : is_prime x1) (hx2_prime : is_prime x2) 
    (h_roots : x1 + x2 = 1999) (h_product : x1 * x2 = m) : 
    m = 3994 := 
by 
    sorry

end determine_m_from_quadratic_l152_152293


namespace ratio_c_over_b_range_l152_152995

-- Define the basic properties and conditions
variables (A B C a b c : ℝ)
variable (ABC_acute : (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (A < 90) ∧ (B < 90) ∧ (C < 90))
variable (C_eq_2B : C = 2 * B)
variable (triangle_sum : A + B + C = 180)

-- Define the law of sines and other relevant conditions
axiom law_of_sines : (∀ {a b c A B C : ℝ}, a / (sin A) = b / (sin B) = c / (sin C))

-- State the theorem
theorem ratio_c_over_b_range : ( ∃ B, 30 < B ∧ B < 45 ∧ (2 * cos B) ∈ (sqrt 2, sqrt 3)) :=
sorry

end ratio_c_over_b_range_l152_152995


namespace exists_X_Y_sum_not_in_third_subset_l152_152706

open Nat Set

theorem exists_X_Y_sum_not_in_third_subset :
  ∀ (M_1 M_2 M_3 : Set ℕ), 
  Disjoint M_1 M_2 ∧ Disjoint M_2 M_3 ∧ Disjoint M_1 M_3 → 
  ∃ (X Y : ℕ), (X ∈ M_1 ∪ M_2 ∪ M_3) ∧ (Y ∈ M_1 ∪ M_2 ∪ M_3) ∧  
  (X ∈ M_1 → Y ∈ M_2 ∨ Y ∈ M_3) ∧
  (X ∈ M_2 → Y ∈ M_1 ∨ Y ∈ M_3) ∧
  (X ∈ M_3 → Y ∈ M_1 ∨ Y ∈ M_2) ∧
  (X + Y ∉ M_3) :=
by
  intros M_1 M_2 M_3 disj
  sorry

end exists_X_Y_sum_not_in_third_subset_l152_152706


namespace eggs_town_hall_l152_152691

-- Definitions of given conditions
def eggs_club_house : ℕ := 40
def eggs_park : ℕ := 25
def total_eggs_found : ℕ := 80

-- Problem statement
theorem eggs_town_hall : total_eggs_found - (eggs_club_house + eggs_park) = 15 := by
  sorry

end eggs_town_hall_l152_152691


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152670

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152670


namespace chocolates_difference_l152_152168

-- Conditions
def Robert_chocolates : Nat := 13
def Nickel_chocolates : Nat := 4

-- Statement
theorem chocolates_difference : (Robert_chocolates - Nickel_chocolates) = 9 := by
  sorry

end chocolates_difference_l152_152168


namespace cone_prism_volume_ratio_l152_152910

noncomputable def volume_ratio_cone_prism (h : ℝ) : ℝ :=
  let r := 3 / 2
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * 2 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio (h : ℝ) : volume_ratio_cone_prism h = π / 32 := by
  sorry

end cone_prism_volume_ratio_l152_152910


namespace unique_triple_l152_152975

theorem unique_triple (x y p : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p) (h1 : p = x^2 + 1) (h2 : 2 * p^2 = y^2 + 1) :
  (x, y, p) = (2, 7, 5) :=
sorry

end unique_triple_l152_152975


namespace provisions_last_initially_for_39_days_l152_152510

variable (initial_provision_days : ℕ)
variable (initial_men : ℕ := 2000)
variable (extra_men : ℕ := 600)
variable (days_after_reinforcement : ℕ := 30)
variable (total_men_after_reinforcement : ℕ := initial_men + extra_men)

theorem provisions_last_initially_for_39_days 
  (h1 : initial_men * 15 + initial_men * (initial_provision_days - 15) = total_men_after_reinforcement * days_after_reinforcement) : 
  initial_provision_days = 39 := 
by 
  have h2 : initial_men * 15 + initial_men * (initial_provision_days - 15) = 2000 * 15 + initial_men * (initial_provision_days - 15),
  simp at h1,
  rw [h2] at h1,
  sorry

end provisions_last_initially_for_39_days_l152_152510


namespace absolute_value_property_l152_152315

theorem absolute_value_property (a b c : ℤ) (h : |a - b| + |c - a| = 1) : |a - c| + |c - b| + |b - a| = 2 :=
sorry

end absolute_value_property_l152_152315


namespace trapezoid_circumradius_l152_152125

theorem trapezoid_circumradius (a b : ℝ) : 
  let R := sqrt((a^2 + b^2) / 2),
      acute_angle : ℝ := π / 4 in 
  R = sqrt((a^2 + b^2) / 2) := by 
begin
  sorry
end

end trapezoid_circumradius_l152_152125


namespace maximum_path_length_in_prism_l152_152881

-- Define the dimensions of the rectangular prism
def length : ℝ := 1
def width : ℝ := 2
def height : ℝ := 3

-- Define the distances in the rectangular prism
def edge_lengths := {length, width, height}

def face_diagonals := {Real.sqrt (length^2 + width^2), 
                       Real.sqrt (length^2 + height^2), 
                       Real.sqrt (width^2 + height^2)}

def space_diagonal := Real.sqrt (length^2 + width^2 + height^2)

-- Define the maximum possible path length
def max_path_length : ℝ := 2 * space_diagonal + 4 * max face_diagonals

-- Problem statement
theorem maximum_path_length_in_prism : 
  max_path_length = 2 * Real.sqrt 14 + 4 * Real.sqrt 13 :=
by
  sorry

end maximum_path_length_in_prism_l152_152881


namespace stock_price_after_two_years_l152_152922

theorem stock_price_after_two_years 
    (p0 : ℝ) (r1 r2 : ℝ) (p1 p2 : ℝ) 
    (h0 : p0 = 100) (h1 : r1 = 0.50) 
    (h2 : r2 = 0.30) 
    (h3 : p1 = p0 * (1 + r1)) 
    (h4 : p2 = p1 * (1 - r2)) : 
    p2 = 105 :=
by sorry

end stock_price_after_two_years_l152_152922


namespace necessary_but_not_sufficient_condition_l152_152872

-- Define lines and their properties
structure Line :=
  (intersects : Line → Prop)
  (parallel : Line → Prop)

-- Define skew lines
def skew_lines (a b : Line) : Prop :=
  ¬a.intersects b ∧ ¬a.parallel b

-- Define the condition that lines a and b do not intersect
def do_not_intersect (a b : Line) : Prop :=
  ¬a.intersects b

-- Define our main theorem statement
theorem necessary_but_not_sufficient_condition (a b : Line) :
  do_not_intersect a b → (do_not_intersect a b ∧ skew_lines a b ↔ false) ∧ (skew_lines a b → do_not_intersect a b) :=
by
  sorry

end necessary_but_not_sufficient_condition_l152_152872


namespace cot_double_angle_l152_152684

theorem cot_double_angle {α : ℝ} (h1 : - (Real.pi / 2) < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cot (2 * α) = 7 / 24 :=
by
  sorry

end cot_double_angle_l152_152684


namespace determine_m_l152_152292

noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / real.sqrt(A^2 + B^2)

theorem determine_m (m : ℝ) : 
  distance_between_parallel_lines 3 4 (-8) m = 1 ↔ (m = -3 ∨ m = -13) :=
by
  -- Note: The proof starts here.
  sorry

end determine_m_l152_152292


namespace abs_diff_roots_eq_3_l152_152588

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l152_152588


namespace percentage_per_cup_l152_152461

-- Definition of the total capacity of the pitcher
def pitcher_total_capacity (C : ℝ) := C

-- Two-thirds of the pitcher is filled with orange juice
def orange_juice_amount (C : ℝ) := (2 / 3) * C

-- The pitcher is emptied by pouring an equal amount of juice into each of 4 cups
def juice_per_cup (C : ℝ) := (orange_juice_amount C) / 4

-- Proof that each cup received 16.67% of the total capacity of the pitcher
theorem percentage_per_cup (C : ℝ) (hC : C > 0) : 
  (juice_per_cup C / C) * 100 = 16.67 := 
by 
  sorry

end percentage_per_cup_l152_152461


namespace product_not_divisible_l152_152630

noncomputable def repeated_digit (d n : ℕ) : ℕ :=
  ∑ i in finset.range n, d * 10^i

theorem product_not_divisible (n : ℕ) :
  ∀ (digits : fin (9) → ℕ),
    (∀ i, digits i = repeated_digit (i + 1) n) →
    ∀ i j, i ≠ j → ¬ (digits i * digits j ∣ ∏ k, if k ≠ i ∧ k ≠ j then digits k else 1) := by
  sorry

end product_not_divisible_l152_152630


namespace area_of_rectangle_EFGH_l152_152777

theorem area_of_rectangle_EFGH (E F G H M N : Point) (P Q R : ℝ)
  (h1 : EM = 2) (h2 : MH = 2) (h3 : HG = 4)
  (h4 : M ⊥ EH) (h5 : N ⊥ EH)
  (diagonal_EH : EH = EM + MH + HG) :
  area_rect_EFGH = 16 * sqrt 2 :=
by sorry

end area_of_rectangle_EFGH_l152_152777


namespace investment_in_real_estate_l152_152083

def total_investment : ℝ := 200000
def ratio_real_estate_to_mutual_funds : ℝ := 7

theorem investment_in_real_estate (mutual_funds_investment real_estate_investment: ℝ) 
  (h1 : mutual_funds_investment + real_estate_investment = total_investment)
  (h2 : real_estate_investment = ratio_real_estate_to_mutual_funds * mutual_funds_investment) :
  real_estate_investment = 175000 := sorry

end investment_in_real_estate_l152_152083


namespace box_probability_l152_152499

theorem box_probability :
  ∀ (total_balls blue_balls red_balls : ℕ),
  blue_balls = 10 →
  red_balls = 8 →
  total_balls = blue_balls + red_balls →
  (∃ (prob : ℚ), prob = (1520 : ℚ) / (3060 : ℚ) ∧ 
    prob = (5 : ℚ) / (10.2 : ℚ)) :=
by
  intros total_balls blue_balls red_balls h1 h2 h3
  use (1520 : ℚ) / (3060 : ℚ)
  split
  . trivial
  . sorry

end box_probability_l152_152499


namespace value_of_d_l152_152689

theorem value_of_d (d : ℝ) : (∀ x : ℝ, 3 * (5 + 2 * d * x) = 15 * x + 15) ↔ d = 5 / 2 :=
begin
  sorry
end

end value_of_d_l152_152689


namespace shop_earnings_correct_l152_152335

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end shop_earnings_correct_l152_152335


namespace an_expression_l152_152297

-- Given conditions
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n

-- The statement to be proved
theorem an_expression (a : ℕ → ℕ) (n : ℕ) (h_Sn : ∀ n, Sn a n = 2 * a n - n) :
  a n = 2^n - 1 :=
sorry

end an_expression_l152_152297


namespace bob_wins_game_l152_152208

theorem bob_wins_game : 
  ∀ n : ℕ, 0 < n → 
  (∃ k ≥ 1, ∀ m : ℕ, 0 < m → (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0) ∨ 
    (∃ k : ℕ, k ≥ 1 ∧ (m = m^k → ¬ (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0)))
  ) :=
sorry

end bob_wins_game_l152_152208


namespace owen_initial_turtles_l152_152078

variables (O J : ℕ)

-- Conditions
def johanna_turtles := J = O - 5
def owen_final_turtles := 2 * O + J / 2 = 50

-- Theorem statement
theorem owen_initial_turtles (h1 : johanna_turtles O J) (h2 : owen_final_turtles O J) : O = 21 :=
sorry

end owen_initial_turtles_l152_152078


namespace problem_solution_l152_152256

noncomputable def curve_M (x y : ℝ) : Prop :=
  x ^ (1 / 2) + y ^ (1 / 2) = 1

def statement_1 (x y : ℝ) : Prop :=
  (curve_M x y) → (real.sqrt (x ^ 2 + y ^ 2) = real.sqrt 2 / 2)

def statement_2 (x y : ℝ) : Prop :=
  (curve_M x y) → (x ∈ set.Icc 0 1 ∧ y ∈ set.Icc 0 1)

theorem problem_solution :
  (¬ ∀ (x y : ℝ), statement_1 x y) ∧ (∀ (x y : ℝ), statement_2 x y) :=
by
  sorry

end problem_solution_l152_152256


namespace part_1_part_2_part_3_l152_152036

noncomputable def circle_equation := 
  ∀ t : ℝ, x y : ℝ, (x ^ 2 + y ^ 2 - 2 * (t + 3) * x + 2 * (1 - 4 * t ^ 2) * y + 16 * t ^ 4 + 9 = 0) = 
  ((x - (t + 3)) ^ 2 + (y - (4 * t ^ 2 - 1)) ^ 2 = -7 * t ^ 2 + 6 * t + 1)

theorem part_1 (t : ℝ) :
  (∀ x y : ℝ, circle_equation t x y → (-7 * t ^ 2 + 6 * t + 1 > 0)) → 
  (- (1/7) < t ∧ t < 1) :=
sorry

theorem part_2 (t : ℝ) :
  (∀ x y : ℝ, circle_equation (3/7) x y → ((x - (24/7)) ^ 2 + (y + (13/49)) ^ 2 = (16/7))) :=
sorry

theorem part_3 (t : ℝ) :
  (∀ x y : ℝ, circle_equation t x y → ((3 - (t + 3)) ^ 2 + (4 * t ^ 2 - (4 * t ^ 2 - 1)) ^ 2 < -7 * t ^ 2 + 6 * t + 1)) → 
  (0 < t ∧ t < 3/4) :=
sorry

end part_1_part_2_part_3_l152_152036


namespace calculate_expression_l152_152557

theorem calculate_expression : 
  ∀ (x y z : ℤ), x = 2 → y = -3 → z = 7 → (x^2 + y^2 + z^2 - 2 * x * y) = 74 :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end calculate_expression_l152_152557


namespace jerrys_breakfast_calories_l152_152043

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l152_152043


namespace volume_of_pyramid_l152_152401

-- Define the regular pentagon ABCDE
structure RegularPentagon (A B C D E : Type) :=
  (is_regular_pentagon : true)

-- Define the right pyramid PABCDE
structure RightPyramid (P A B C D E : Type) :=
  (is_right_pyramid : true)

-- Define the equilateral triangle PAD with side length 6
structure EquilateralTriangle (P A D : Type) :=
  (side_length : Nat)
  (is_equilateral : true)

-- Define the problem conditions
def problem_conditions (A B C D E P : Type) : Prop :=
  RegularPentagon A B C D E ∧ RightPyramid P A B C D E ∧ EquilateralTriangle P A D ∧ 
  EquilateralTriangle.side_length P A D = 6

-- The task to prove
theorem volume_of_pyramid (A B C D E P : Type) 
  (h : problem_conditions A B C D E P) : 
  ∃ (V : ℕ), V = 135 := 
sorry

end volume_of_pyramid_l152_152401


namespace abs_inequality_solution_set_l152_152447

theorem abs_inequality_solution_set (x : ℝ) : (|x - 1| ≥ 5) ↔ (x ≥ 6 ∨ x ≤ -4) := 
by sorry

end abs_inequality_solution_set_l152_152447


namespace angle_bisector_divides_larger_segment_l152_152795

-- Definitions of a triangle and the angle bisector theorem.

structure Triangle (α : Type _) [LinearOrderedField α] :=
(A B C : α × α)

structure AngleBisector (α : Type _) [LinearOrderedField α] (T : Triangle α) :=
(BD : α × α → α × α → Prop)
(is_bisector : ∀ A B C D, BD (A, B) (C, D) → (Complex.arg (C - D)) = (Complex.arg (A - B)))

variables {α : Type _} [LinearOrderedField α]

-- The main theorem statement
theorem angle_bisector_divides_larger_segment {T : Triangle α} 
  (BD : AngleBisector α T) (h1 : T.B.x > T.C.x) : 
  let D := ⟨T.C.x + (T.B.x - T.C.x) * (T.A.y - T.C.y) / (T.B.y - T.C.y), T.C.y + (T.B.y - T.C.y) * (T.A.x - T.C.x) / (T.B.x - T.C.x)⟩
  in ∃ D: α × α, BD.is_bisector T.A T.B T.C D ∧ (T.A.dist D < T.C.dist D) := 
by 
  sorry

end angle_bisector_divides_larger_segment_l152_152795


namespace range_of_k_l152_152388

theorem range_of_k (k : ℝ) : 
  ((2 + k) * (3 * k + 1) > 0) ∧ (-1 < k ∧ k < 1/2 ∧ k ≠ 0) ↔ (k ∈ set.Ioo (-1/3) 0 ∪ set.Ioo 0 1/2) :=
by sorry

end range_of_k_l152_152388


namespace mat_length_correct_l152_152201

-- Definitions for the conditions
def table_radius : ℝ := 5
def mat_width : ℝ := 1
def mat_length (x : ℝ) : ℝ := x
def num_mats : ℤ := 8

-- The theorem to be proved
theorem mat_length_correct : 
  ∃ x : ℝ, mat_length x = (3 * real.sqrt 11 - 10 * real.sqrt (2 - real.sqrt 2) + 1) / 2 :=
sorry

end mat_length_correct_l152_152201


namespace before_lunch_rush_customers_l152_152920

def original_customers_before_lunch := 29
def added_customers_during_lunch := 20
def customers_no_tip := 34
def customers_tip := 15

theorem before_lunch_rush_customers : 
  original_customers_before_lunch + added_customers_during_lunch = customers_no_tip + customers_tip → 
  original_customers_before_lunch = 29 := 
by
  sorry

end before_lunch_rush_customers_l152_152920


namespace number_of_lightsabers_in_order_l152_152157

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end number_of_lightsabers_in_order_l152_152157


namespace sufficient_but_not_necessary_l152_152284

variable {a : ℝ}

-- Conditions from the problem
def condition1 : Prop := a > 0
def condition2 : Prop := a ≠ 1

-- Statement that f(x) = log_a(x) is increasing on (0, +∞)
def f_increasing : Prop := a > 1

-- Statement that g(x) = (1 - a) * a^x is decreasing on ℝ
def g_decreasing : Prop := a > 1 ∨ (0 < a ∧ a < 1)

-- The Lean statement proving that f_increasing is a sufficient but not necessary condition for g_decreasing.
theorem sufficient_but_not_necessary : f_increasing → g_decreasing :=
by
  intro h
  -- Proving sufficiency
  unfold f_increasing at h
  unfold g_decreasing
  left
  exact h

end sufficient_but_not_necessary_l152_152284


namespace sum_of_even_power_coeffs_l152_152860

theorem sum_of_even_power_coeffs :
  let f := (1 - (1 : ℤ)/ (x : ℤ)) ^ 7,
      even_coeff_sum := ∑ i in {0, 2, 4, 6}, f.coeff(i)
  in even_coeff_sum = -64 :=
by
  sorry

end sum_of_even_power_coeffs_l152_152860


namespace no_rational_roots_of_prime_3_digit_l152_152374

noncomputable def is_prime (n : ℕ) := Nat.Prime n

theorem no_rational_roots_of_prime_3_digit (a b c : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) 
(h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : 0 ≤ c ∧ c ≤ 9) 
(p := 100 * a + 10 * b + c) (hp : is_prime p) (h₃ : 100 ≤ p ∧ p ≤ 999) :
¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
sorry

end no_rational_roots_of_prime_3_digit_l152_152374


namespace find_DG_length_l152_152824

-- We define the rectangles and their properties.
variables (a b k l S : ℕ)
variables (A B C D E F G H I : Type)

-- The given conditions
-- Condition 1: Rectangles have equal areas with integer side lengths
def equal_areas (a b k l : ℕ) (S : ℕ) : Prop :=
  S = a * k ∧ S = b * l ∧ a + b > 0

-- Condition 2: BC = 31
def BC_length := (BC : ℕ) = 31

-- Proving DG = 992 given the above conditions
theorem find_DG_length (a b k l S : ℕ) (h1 : equal_areas a b k l S) (h2 : BC_length)
  (h3 : S = 31 * (a + b)) :
  k = 992 := 
sorry

end find_DG_length_l152_152824


namespace sequence_sum_4321_l152_152029

noncomputable def a : ℕ+ → ℤ
| 1 => -1
| 2 => 1
| 3 => -2
| n+1 => if (n+1)%4 == 1 then -1 else if (n+1)%4 == 2 then 1 else -2

def sequence_property (a : ℕ+ → ℤ) (n : ℕ+) : Prop :=
  a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)

def sequence_non_1 (a : ℕ+ → ℤ) (n : ℕ+) : Prop := 
  a (n + 1) * a (n + 2) * a (n + 3) ≠ 1

theorem sequence_sum_4321 :
  (∀ n : ℕ+, sequence_property a n) →
  (∀ n : ℕ+, sequence_non_1 a n) →
  ∑ i in Finset.range 4321, a ⟨i+1, nat.succ_pos⟩ = -4321 :=
by
  intros
  -- skipping the proof
  sorry

end sequence_sum_4321_l152_152029


namespace sin_alpha_plus_beta_necess_suff_l152_152424

theorem sin_alpha_plus_beta_necess_suff (α β : ℝ) : 
  (sin (α + β) = 0 -> α + β = 0) ∧ (α + β = 0 -> sin (α + β) = 0) -> 
  (∀ k : ℤ, α + β = k * π -> α + β = 0) ∧ (α + β = 0 -> sin (α + β) = 0) := 
by sorry

end sin_alpha_plus_beta_necess_suff_l152_152424


namespace trigonometric_range_l152_152153

open Real

theorem trigonometric_range (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2) :
  2 < (sin α + tan α) * (cos α + cot α) ∧ 
  (sin α + tan α) * (cos α + cot α) <= (3 / 2 + Real.sqrt 2) :=
by
  sorry

end trigonometric_range_l152_152153


namespace ferris_wheel_time_l152_152497

def ferris_wheel_f (t : ℝ) : ℝ :=
  30 * Real.cos ((π / 60) * t) + 10 * t + 30

def reach_height (R H T v0 : ℝ) : ℝ := H + R

theorem ferris_wheel_time :
  ∀ t : ℝ, ferris_wheel_f t = reach_height 30 15 120 10 → t = 4 :=
by
  intro t
  have eqn : ferris_wheel_f t = 45 := sorry
  exact sorry

end ferris_wheel_time_l152_152497


namespace minimum_value_l152_152064

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + (1 / (a * b * c)) ≥ 10 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l152_152064


namespace positive_difference_jo_kate_l152_152047

open Nat

noncomputable def jo_sum : ℕ := (200 * (200 + 1)) / 2

def rounding_to_nearest_5 (n : ℕ) : ℕ :=
  if n % 5 >= 3 then (n + 5 - n % 5)
  else (n - n % 5)

noncomputable def kate_sum : ℕ :=
  ∑ i in finset.range (200 + 1), rounding_to_nearest_5 i

theorem positive_difference_jo_kate :
  |kate_sum - jo_sum| = 120 := sorry

end positive_difference_jo_kate_l152_152047


namespace arrange_numbers_in_ascending_order_l152_152944

noncomputable def S := 222 ^ 2
noncomputable def T := 22 ^ 22
noncomputable def U := 2 ^ 222
noncomputable def V := 22 ^ (2 ^ 2)
noncomputable def W := 2 ^ (22 ^ 2)
noncomputable def X := 2 ^ (2 ^ 22)
noncomputable def Y := 2 ^ (2 ^ (2 ^ 2))

theorem arrange_numbers_in_ascending_order :
  S < Y ∧ Y < V ∧ V < T ∧ T < U ∧ U < W ∧ W < X :=
sorry

end arrange_numbers_in_ascending_order_l152_152944


namespace expected_winnings_value_l152_152508

-- Define the fair 8-sided die and its probability distribution
def fair_die : ℕ → ℝ
| n := if n ≥ 1 ∧ n ≤ 8 then 1 / 8 else 0

-- Define the winning amount based on the value rolled
def winnings (n : ℕ) : ℝ :=
if n % 3 = 0 then n else 0

-- Define the expected value calculation
def expected_value : ℝ :=
∑ n in finset.range 9, (winnings n) * (fair_die n)

theorem expected_winnings_value : expected_value = 1.125 := 
by 
  sorry

end expected_winnings_value_l152_152508


namespace problem1_problem2_l152_152558

section Problem1
theorem problem1 : (1 * (1 + real.cbrt 8) ^ 0 + abs (-2) - real.sqrt 9) = 0 := by
  sorry
end Problem1

section Problem2
variables (a : ℝ)
theorem problem2 : (a + 3) * (a - 3) + a * (1 - a) = a - 9 := by
  sorry
end Problem2

end problem1_problem2_l152_152558


namespace card_number_of_false_statements_l152_152161

-- Definitions of the statements on the card
def statement1 : Prop := ∃i ∈ ({2, 3, 4, 5} : Set ℕ), i = 1
def statement2 : Prop := ∃i ∈ ({1, 3, 4, 5} : Set ℕ), i = 3
def statement3 : Prop := ∃i ∈ ({1, 2, 4, 5} : Set ℕ), i = 2
def statement4 : Prop := ∃i ∈ ({1, 2, 3, 5} : Set ℕ), i = 4
def statement5 : Prop := ∃i ∈ ({1, 2, 3, 4} : Set ℕ), i = 0

-- Formalization of the problem
theorem card_number_of_false_statements : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (s1 s2: ℕ), 
    (s1 ∈ ({1, 5} : Set ℕ) → s2 ∈ ({1, 5} : Set ℕ) → statement1 → ¬ statement1) ∧ 
    (s1 ∈ ({2} : Set ℕ) → s2 ∈ ({2} : Set ℕ) → statement2 → ¬ statement2) ∧
    (s1 ∈ ({2} : Set ℕ) → s2 ∈ ({2} : Set ℕ) → statement3 → ¬ statement3) ∧ 
    (s1 ∈ ({4} : Set ℕ) → s2 ∈ ({4} : Set ℕ) → statement4 → ¬ statement4) ∧
    (s1 ∈ ({0} : Set ℕ) → s2 ∈ ({0} : Set ℕ) → statement5 → ¬ statement5): sorry

end card_number_of_false_statements_l152_152161


namespace positive_difference_between_plan1_and_plan2_l152_152070

-- Define the conditions and computations for Plan 1
def plan1_balance_after_5_years (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / (n : ℝ))^(n * t)

def plan1_payment1 (balance : ℝ) : ℝ :=
  balance / 3

def plan1_remaining_balance (balance payment : ℝ) : ℝ :=
  balance - payment

def plan1_total_payment (initial_payment remaining_balance : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  initial_payment + remaining_balance * (1 + r / (n : ℝ))^(n * t)

-- Define the conditions and computations for Plan 2
def plan2_balance_after_7_years (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^(t)

def plan2_payment1 (balance : ℝ) : ℝ :=
  balance / 3

def plan2_remaining_balance (balance payment : ℝ) : ℝ :=
  balance - payment

def plan2_total_payment (initial_payment remaining_balance : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  initial_payment + remaining_balance * (1 + r)^t

-- Given conditions
def P : ℝ := 12000
def r : ℝ := 0.08
def n1 : ℕ := 2
def t1 : ℕ := 5
def t2 : ℕ := 7
def t3 : ℕ := 3

-- Plan 1 computations
def balance1 := plan1_balance_after_5_years P r n1 t1
def payment1 := plan1_payment1 balance1
def remaining_balance1 := plan1_remaining_balance balance1 payment1
def total_payment1 := plan1_total_payment payment1 remaining_balance1 r n1 t1

-- Plan 2 computations
def balance2 := plan2_balance_after_7_years P r t2
def payment2 := plan2_payment1 balance2
def remaining_balance2 := plan2_remaining_balance balance2 payment2
def total_payment2 := plan2_total_payment payment2 remaining_balance2 r t3

-- The positive difference between Plan 1 and Plan 2 total payments
def payment_difference := abs (total_payment2 - total_payment1)

-- Proof
theorem positive_difference_between_plan1_and_plan2 : payment_difference ≈ 668 :=
sorry

end positive_difference_between_plan1_and_plan2_l152_152070


namespace find_a_2015_l152_152673

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≠ 0 → a (n + 1) = (1 + a n) / (1 - a n)

theorem find_a_2015 (a : ℕ → ℚ) (h : sequence a) : a 2015 = -1/2 := 
by
  sorry

end find_a_2015_l152_152673


namespace market_value_of_stock_l152_152879

variable (face_value : ℝ) (annual_dividend yield : ℝ)

-- Given conditions:
def stock_four_percent := annual_dividend = 0.04 * face_value
def stock_yield_five_percent := yield = 0.05

-- Problem statement:
theorem market_value_of_stock (face_value := 100) (annual_dividend := 4) (yield := 0.05) 
  (h1 : stock_four_percent face_value annual_dividend) 
  (h2 : stock_yield_five_percent yield) : 
  (4 / 0.05) * 100 = 80 :=
by
  sorry

end market_value_of_stock_l152_152879


namespace smallest_number_among_given_l152_152935

theorem smallest_number_among_given :
  ∀ x ∈ ({-2023, 0, 0.999, 1} : set ℝ), -2023 ≤ x := 
by
  intro x hx
  sorry

end smallest_number_among_given_l152_152935


namespace min_sum_of_faces_l152_152679

theorem min_sum_of_faces (a b : ℕ) (ha : a ≥ 6) (hb : b ≥ 6) 
  (h1 : a ≥ b)
  (h7_10 : (6 : ℚ) / (a * b) = (3 / 4) * (prob_sum_10 a b))
  (h7_12 : (6 : ℚ) / (a * b) = (1 / 12) * (prob_sum_12 a b)) :
  a + b = 17 := by
  sorry

-- Helper functions to define probabilities (these would need to be defined fully in an actual proof)
def prob_sum_10 (a b : ℕ) : ℚ := sorry
def prob_sum_12 (a b : ℕ) : ℚ := sorry

end min_sum_of_faces_l152_152679


namespace geometric_sequence_11th_term_l152_152110

noncomputable def find_11th_term (a5 a8 : ℕ) (r : ℕ) (term : ℕ) : Prop :=
a5 = 5 ∧ a8 = 40 ∧ a8 = a5 * r^3 ∧ r = 2 ∧ term = a5 * r^6

theorem geometric_sequence_11th_term : ∃ term : ℕ, find_11th_term 5 40 2 term ∧ term = 320 :=
by {
  use 320,
  unfold find_11th_term,
  simp,
  sorry
}

end geometric_sequence_11th_term_l152_152110


namespace odd_n_divides_sn_minus_tn_l152_152569

def sn (n : ℕ) : ℕ :=
  (∑ k in finset.range (n + 1), if nat.coprime k n then k else 0)

def tn (n : ℕ) : ℕ :=
  (∑ k in finset.range (n + 1), if nat.coprime k n then 0 else k)

theorem odd_n_divides_sn_minus_tn (n : ℕ) (hn : 2 ≤ n) : n ∣ (sn n - tn n) → odd n :=
  sorry

end odd_n_divides_sn_minus_tn_l152_152569


namespace seat_3_undetermined_l152_152923

/-- Definitions for seat numbers and occupants -/
inductive Person
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

def Seat := Fin 4

/-- Problem statement: Given that Joe's statements are false -/
axiom joe_statements_false : (¬ (is_next Dana Bret)) ∧ (¬ (between Bret Abby Carl))

/-- Bret is actually sitting in seat #1 -/
axiom bret_in_seat_1 : ∀ seats : Fin 4 → Person, seats 0 = Person.Bret

/-- Prove that the identity of the person sitting in seat #3 cannot be determined definitively -/
theorem seat_3_undetermined : 
  ∃ seats : Fin 4 → Person, ¬(∃ p : Person, (p = seats 2)) :=
sorry

end seat_3_undetermined_l152_152923


namespace monthly_installment_correct_l152_152165

variables
  (cash_price : ℝ)
  (deposit_rate : ℝ)
  (monthly_installments : ℕ)
  (annual_interest_rate : ℝ)

def monthly_deposit : ℝ := deposit_rate * cash_price

def balance : ℝ := cash_price - monthly_deposit

def monthly_interest_rate : ℝ := annual_interest_rate / 12

def time_in_years : ℝ := monthly_installments / 12

def total_repayment_amount : ℝ := balance * (1 + monthly_interest_rate * time_in_years)

def monthly_installment_amount : ℝ := total_repayment_amount / monthly_installments

-- Now we state the theorem to be proven:
theorem monthly_installment_correct :
  cash_price = 26000 →
  deposit_rate = 0.10 →
  monthly_installments = 60 →
  annual_interest_rate = 0.12 →
  monthly_installment_amount cash_price deposit_rate monthly_installments annual_interest_rate = 409.50 := 
  by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end monthly_installment_correct_l152_152165


namespace determine_base_l152_152334

theorem determine_base (b : ℕ) (h : (3 * b + 1)^2 = b^3 + 2 * b + 1) : b = 10 :=
by
  sorry

end determine_base_l152_152334


namespace rectangle_diagonal_l152_152801

theorem rectangle_diagonal (l w : ℝ) 
  (h1 : l * w = 20) 
  (h2 : 2 * l + 2 * w = 18) : 
  real.sqrt (l^2 + w^2) = real.sqrt 41 :=
sorry

end rectangle_diagonal_l152_152801


namespace series_solution_eq_l152_152974

theorem series_solution_eq (x : ℝ) 
  (h : (∃ a : ℕ → ℝ, (∀ n, a n = 1 + 6 * n) ∧ (∑' n, a n * x^n = 100))) :
  x = 23/25 ∨ x = 1/50 :=
sorry

end series_solution_eq_l152_152974


namespace definite_integral_value_l152_152173

theorem definite_integral_value :
  (∫ x in (0 : ℝ)..Real.arctan (1/3), (8 + Real.tan x) / (18 * Real.sin x^2 + 2 * Real.cos x^2)) 
  = (Real.pi / 3) + (Real.log 2 / 36) :=
by
  -- Proof to be provided
  sorry

end definite_integral_value_l152_152173


namespace StatementA_incorrect_l152_152997

def f (n : ℕ) : ℕ := (n.factorial)^2

def g (x : ℕ) : ℕ := f (x + 1) / f x

theorem StatementA_incorrect (x : ℕ) (h : x = 1) : g x ≠ 4 := sorry

end StatementA_incorrect_l152_152997


namespace num_safe_numbers_l152_152993

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, abs (n - (k * p).to_nat) > 3

def is_7_safe (n : ℕ) : Prop := is_p_safe n 7
def is_11_safe (n : ℕ) : Prop := is_p_safe n 11
def is_13_safe (n : ℕ) : Prop := is_p_safe n 13

theorem num_safe_numbers : ∃ (N : ℕ), N = 834 ∧ 
  ∀ n ≤ 10000, is_7_safe n ∧ is_11_safe n ∧ is_13_safe n ↔ n ≤ N := 
sorry

end num_safe_numbers_l152_152993


namespace find_G_minus_L_l152_152543

-- Define the arithmetic sequence terms and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Conditions from the problem
axiom sum_seq_eq_38500 : ∀ (a d : ℝ), ∑ i in Finset.range 350, arithmetic_sequence a d (i + 1) = 38500
axiom seq_bounds : ∀ (a d : ℝ) (n : ℕ), 1 ≤ n → n ≤ 350 → 5 ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 150

-- Define the least and greatest possible values of the 100th term in the sequence
def least_100th_term (a d : ℝ) : ℝ := a + 99 * d
def greatest_100th_term (a d : ℝ) : ℝ := a + 99 * d

-- Prove that the difference G - L equals 60.225
theorem find_G_minus_L : 
  ∀ (a d : ℝ),
    let average := (38500 / 350 : ℝ) in
    let middle_condition := a + 174 * d = average in
    let bounds_condition := (5 ≤ a ∧ a + 349 * d ≤ 150) in
    let L := least_100th_term (110 - 174 * d) d in
    let G := greatest_100th_term (110 - 174 * d) d in
    L = 64.775 ∧ G = 125 → G - L = 60.225 :=
by
  intros a d
  assume average middle_condition bounds_condition
  have L := least_100th_term (110 - 174 * d) d
  have G := greatest_100th_term (110 - 174 * d) d
  sorry -- The proof is omitted.

end find_G_minus_L_l152_152543


namespace isosceles_triangle_given_angle_gt_90_l152_152328

open EuclideanGeometry

theorem isosceles_triangle_given_angle_gt_90 (A B C : Point) (h : ∠ B > 90)
  (hΔ : Triangle A B C) : 
  (∠ A = ∠ C) ∨ (dist A B = dist B C) := 
sorry

end isosceles_triangle_given_angle_gt_90_l152_152328


namespace inequality_proof_l152_152620

-- Define the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Given condition: f(0) = f(2)
def f_condition {b c : ℝ} : Prop := f 0 b c = f 2 b c

-- The goal: f(3/2) < f(0) < f(-2)
theorem inequality_proof (b c : ℝ) (h : f_condition) : 
  f (3/2) b c < f 0 b c ∧ f 0 b c < f (-2) b c := 
sorry

end inequality_proof_l152_152620


namespace smallest_n_is_100_l152_152625

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | ∃ (y : Fin n → ℝ),
    (∑ i, y i) = 2000 ∧
    (∑ i, (y i)^6) = 64000000}

theorem smallest_n_is_100 : smallest_n = 100 :=
  sorry

end smallest_n_is_100_l152_152625


namespace range_of_a_l152_152286

noncomputable def complex_problem (a : ℝ) : Prop :=
  exists (z : ℂ), 
    (z + 2 * complex.i).im = 0 ∧
    (z / (1 - complex.i)).im = 0 ∧
    let w := (z + a * complex.i)^2 in
    w.re > 0 ∧
    w.im > 0

theorem range_of_a : {a : ℝ | complex_problem a} = {a | 2 < a ∧ a < 6} :=
by
  sorry

end range_of_a_l152_152286


namespace positive_A_satisfies_eq_l152_152055

theorem positive_A_satisfies_eq :
  ∃ (A : ℝ), A > 0 ∧ A^2 + 49 = 194 → A = Real.sqrt 145 :=
by
  sorry

end positive_A_satisfies_eq_l152_152055


namespace total_cost_of_color_drawing_l152_152365

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end total_cost_of_color_drawing_l152_152365


namespace series_sum_floor_l152_152253

def S_n (n : ℕ) : set ℕ := { k | ∀ a b : ℕ, k ≠ a * n + 2017 * b }

def A_n (n : ℕ) : ℚ :=
  if finite (S_n n) ∧ (S_n n).nonempty then
    (S_n n).to_finset.sum id / (S_n n).to_finset.card
  else
    0

def sum_series : ℚ := 
  (∑' n, A_n n / 2^n)

theorem series_sum_floor : 
  ⌊sum_series⌋ = 840 :=
begin
  sorry
end

end series_sum_floor_l152_152253


namespace max_ab_sqrt3_2ac_l152_152058

noncomputable def max_value (a b c : ℝ) : ℝ :=
  ab_to_be_to_sqrt3


theorem max_ab_sqrt3_2ac (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) :
    ab_to_be_to_sqrt3 + 2ab = sqrt(3) :=
sorry

end max_ab_sqrt3_2ac_l152_152058


namespace total_number_of_glasses_l152_152480

open scoped Nat

theorem total_number_of_glasses (x y : ℕ) (h1 : y = x + 16) (h2 : (12 * x + 16 * y) / (x + y) = 15) : 12 * x + 16 * y = 480 := by
  sorry

end total_number_of_glasses_l152_152480


namespace cos_of_angle_between_lines_l152_152899

noncomputable def cosTheta (a b : ℝ × ℝ) : ℝ :=
  let dotProduct := a.1 * b.1 + a.2 * b.2
  let magA := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dotProduct / (magA * magB)

theorem cos_of_angle_between_lines :
  cosTheta (3, 4) (1, 3) = 3 / Real.sqrt 10 :=
by
  sorry

end cos_of_angle_between_lines_l152_152899


namespace find_number_l152_152784

variable (number : ℤ)

theorem find_number (h : number - 44 = 15) : number = 59 := 
sorry

end find_number_l152_152784


namespace inequality_proof_l152_152644

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a^2 - 4 * a + 11)) + (1 / (5 * b^2 - 4 * b + 11)) + (1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 := 
by
  -- proof steps will be here
  sorry

end inequality_proof_l152_152644


namespace least_integer_condition_l152_152148

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l152_152148


namespace diagonal_length_of_regular_hexagon_l152_152241

theorem diagonal_length_of_regular_hexagon (
  side_length : ℝ
) (h_side_length : side_length = 12) : 
  ∃ DA, DA = 12 * Real.sqrt 3 :=
by 
  sorry

end diagonal_length_of_regular_hexagon_l152_152241


namespace incorrect_option_l152_152000

-- Define the condition
variable (m : ℝ) (h : m > -1)

-- Define the statements for each option
def optionA : Prop := 4 * m > -4
def optionB : Prop := -5 * m < -5
def optionC : Prop := m + 1 > 0
def optionD : Prop := 1 - m < 2

-- Statement that option C is the incorrect one
theorem incorrect_option : ¬ optionC :=
by {
  sorry
}

end incorrect_option_l152_152000


namespace imaginary_part_of_fraction_l152_152116

theorem imaginary_part_of_fraction :
  (complex.ext_iff.mp (((5 * complex.I) / (1 + 2 * complex.I)) : ℂ)).im = 1 := 
sorry

end imaginary_part_of_fraction_l152_152116


namespace prob_A_C_not_third_day_l152_152326

-- Definitions of the problem
def people := {A, B, C}
def days := {1, 2, 3}

-- Definition of the random assignment
def duty_assignments := {f : people → days // function.bijective f}

-- Definition of the event of interest: A and C not on duty on the 3rd day
def A_C_not_on_duty_third_day (f : people → days) := (f A ≠ 3 ∧ f C ≠ 3)

-- Statement of the problem: Prove that the probability that both A and C are not on duty on the third day is 1/3
theorem prob_A_C_not_third_day : 
  (probability (λ f : {f // function.bijective f}, A_C_not_on_duty_third_day f.val)) = 1/3 := 
sorry

end prob_A_C_not_third_day_l152_152326


namespace slope_range_l152_152633

theorem slope_range (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ k : ℝ, k = (y + 2) / (x + 1) ∧ k ∈ Set.Ici (3 / 4) :=
sorry

end slope_range_l152_152633


namespace math_problem_proof_l152_152282

theorem math_problem_proof
  (x0 y0 x y t s r a b : ℝ)
  (hr_pos : r > 0)
  (h_circle : (x - t) ^ 2 + (y - s) ^ 2 = r ^ 2)
  (h_independent : ∀ (x0 y0 : ℝ), (|x0 - y0 + a| + |x0 - y0 + b|) = (|x - y + a| + |x - y + b|))
  (h_a_ne_b : a ≠ b) :
  (|a - b| = 2 * sqrt 2 * r → (∃ k : ℝ, ∀ t s, t = k * s)) ∧
  (|a - b| = 2 * sqrt 2 → r ≤ 1) ∧
  ((r = sqrt 2 ∧ b = 2) → ¬ (a ≥ 6)) :=
by
  sorry

end math_problem_proof_l152_152282


namespace radius_YZ_eq_l152_152346

-- Definitions for conditions
def right_triangle (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] :=
  angle XYZ = 90

def area_quarter_circle_eq (XY : ℝ) : Prop :=
  (1 / 4) * π * XY^2 = 2 * π 

def arc_length_quarter_circle_eq (XZ : ℝ) : Prop :=
  (1 / 2) * π * XZ = 6 * π 

-- Main theorem
theorem radius_YZ_eq (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (rt : right_triangle X Y Z)
  (areaXY : area_quarter_circle_eq XY)
  (arcXZ : arc_length_quarter_circle_eq XZ) :
  ∃ (r : ℝ), r = 12.33 := 
sorry

end radius_YZ_eq_l152_152346


namespace zuminglish_valid_9_letter_mod_500_l152_152013

/-
Define the sequences and initial conditions and then prove the required result.
-/

/-- The number of valid 9-letter Zuminglish words modulo 500 -/
theorem zuminglish_valid_9_letter_mod_500 :
  let a : ℕ → ℕ := by
    have a_2 : ℕ := 4
    have b_2 : ℕ := 2
    have c_2 : ℕ := 2

    -- Recursive functions
    def a : ℕ → ℕ
    | 2 := a_2
    | n+1 := 2 * (a n + c n)

    def b : ℕ → ℕ
    | 2 := b_2
    | n+1 := a n

    def c : ℕ → ℕ
    | 2 := c_2
    | n+1 := 2 * b n

    -- Compute up to a_9, b_9 and c_9
    have a_9 : ℕ := (2 * (a 8 + c 8)) % 500
    have b_9 : ℕ := (a 8) % 500
    have c_9 : ℕ := (2 * b 8) % 500
    pure (a_9 + b_9 + c_9) % 500 == 472 := sorry

end zuminglish_valid_9_letter_mod_500_l152_152013


namespace alan_total_payment_l152_152207

-- Define the costs of CDs
def cost_AVN : ℝ := 12
def cost_TheDark : ℝ := 2 * cost_AVN
def cost_TheDark_total : ℝ := 2 * cost_TheDark
def cost_other_CDs : ℝ := cost_AVN + cost_TheDark_total
def cost_90s : ℝ := 0.4 * cost_other_CDs
def total_cost : ℝ := cost_AVN + cost_TheDark_total + cost_90s

-- Formulate the main statement
theorem alan_total_payment :
  total_cost = 84 := by
  sorry

end alan_total_payment_l152_152207


namespace values_of_k_for_exactly_one_solution_l152_152259

theorem values_of_k_for_exactly_one_solution (k : ℝ) :
  (∃ a b c : ℝ, a = 3 ∧ b = -4 - k ∧ c = 40 ∧ a * x^2 + b * x + c = 0 ∧ (b^2 - 4 * a * c = 0)) ↔ (k = 18 ∨ k = -26) := by
s

end values_of_k_for_exactly_one_solution_l152_152259


namespace solve_inequality_l152_152783

theorem solve_inequality (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ( if 0 ≤ a ∧ a < 1 / 2 then (x > a ∧ x < 1 - a) else 
    if a = 1 / 2 then false else 
    if 1 / 2 < a ∧ a ≤ 1 then (x > 1 - a ∧ x < a) else false ) ↔ ((x - a) * (x + a - 1) < 0) :=
by
  sorry

end solve_inequality_l152_152783


namespace quadrilateral_statements_l152_152307

-- Definitions
def is_square (q : Quadrilateral) : Prop := ∀ a b c d : Angle, q.has_angles [a, b, c, d] → a = b ∧ b = c ∧ c = d ∧ d = 90
def has_four_right_angles (q : Quadrilateral) : Prop := q.has_angles [90, 90, 90, 90]

-- Original statement
def original_statement (q : Quadrilateral) := is_square q → has_four_right_angles q

-- Converse statement
def converse_statement (q : Quadrilateral) := has_four_right_angles q → is_square q

-- Inverse statement
def inverse_statement (q : Quadrilateral) := ¬ is_square q → ¬ has_four_right_angles q

-- Proof problem
theorem quadrilateral_statements (q : Quadrilateral) :
  (¬ converse_statement q) ∧ (¬ inverse_statement q) :=
sorry

end quadrilateral_statements_l152_152307


namespace housewife_remaining_money_in_B_currency_l152_152188

def initial_amount : ℝ := 450
def groceries_fraction : ℝ := 3/5
def household_items_fraction : ℝ := 1/6
def personal_care_items_fraction : ℝ := 1/10
def sales_tax_rate : ℝ := 5 / 100
def discount_rate : ℝ := 10 / 100
def exchange_rate : ℝ := 0.8

noncomputable def remaining_amount_in_B_currency : ℝ :=
  let groceries := groceries_fraction * initial_amount
  let household_items := household_items_fraction * initial_amount
  let household_items_with_tax := household_items * (1 + sales_tax_rate)
  let personal_care_items := personal_care_items_fraction * initial_amount
  let personal_care_items_with_discount := personal_care_items * (1 - discount_rate)
  let total_spent := groceries + household_items_with_tax + personal_care_items_with_discount
  let remaining_amount := initial_amount - total_spent
  remaining_amount * exchange_rate

theorem housewife_remaining_money_in_B_currency : remaining_amount_in_B_currency = 48.6 :=
by
  -- proof goes here
  sorry

end housewife_remaining_money_in_B_currency_l152_152188


namespace bucket_weight_full_l152_152500

variable (c d : ℝ)

theorem bucket_weight_full (h1 : ∃ x y, x + (1 / 4) * y = c)
                           (h2 : ∃ x y, x + (3 / 4) * y = d) :
  ∃ x y, x + y = (3 * d - c) / 2 :=
by
  sorry

end bucket_weight_full_l152_152500


namespace no_two_obtuse_angles_in_triangle_l152_152863

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l152_152863


namespace digit_8_not_in_mean_l152_152792

def set_of_numbers : list ℕ := [
  8,
  88,
  888,
  8888,
  88888,
  888888,
  8888888,
  88888888,
  888888888
]

noncomputable def arithmetic_mean (lst : list ℕ) : ℕ :=
(lst.sum) / lst.length

theorem digit_8_not_in_mean :
  let N := arithmetic_mean set_of_numbers in
  N = 987654320 ∧
  (∀ d ∈ (N.digits 10), d ≠ 8) :=
by
  sorry

end digit_8_not_in_mean_l152_152792


namespace upper_sum_leq_lower_sum_geq_l152_152773

variable {α : Type*} [LinearOrder α] [AddGroup α] [Module ℝ α] {f : ℝ → α} {c d y M M₁ M₂ m m₁ m₂ : ℝ}

theorem upper_sum_leq (h1 : M₁ ≤ M) (h2 : M₂ ≤ M) (hcd : c ≤ d) (hcy : c ≤ y) (hyd : y ≤ d) :
  M₁ * (y - c) + M₂ * (d - y) ≤ M * (d - c) := sorry

theorem lower_sum_geq (h1 : m₁ ≥ m) (h2 : m₂ ≥ m) (hcd : c ≤ d) (hcy : c ≤ y) (hyd : y ≤ d) :
  m₁ * (y - c) + m₂ * (d - y) ≥ m * (d - c) := sorry

end upper_sum_leq_lower_sum_geq_l152_152773


namespace distance_from_original_position_l152_152532

/-- Definition of initial problem conditions and parameters --/
def square_area (l : ℝ) : Prop :=
  l * l = 18

def folded_area_relation (x : ℝ) : Prop :=
  0.5 * x^2 = 2 * (18 - 0.5 * x^2)

/-- The main statement that needs to be proved --/
theorem distance_from_original_position :
  ∃ (A_initial A_folded_dist : ℝ),
    square_area A_initial ∧
    (∃ x : ℝ, folded_area_relation x ∧ A_folded_dist = 2 * Real.sqrt 6 * Real.sqrt 2) ∧
    A_folded_dist = 4 * Real.sqrt 3 :=
by
  -- The proof is omitted here; providing structure for the problem.
  sorry

end distance_from_original_position_l152_152532


namespace triangle_incircle_excircle_tangency_l152_152810

variable {α : Type*} [ordered_ring α]

-- Definitions of side lengths
variables (a b c : α)

-- Definitions of points of tangency from the problem conditions
def CK := a -- CK is defined in terms of the length a, which is BC
def BL := a -- BL is defined in terms of the length a, which is BC

-- Semi-perimeter of the triangle
def s : α := (a + b + c) / 2

-- Statement to prove
theorem triangle_incircle_excircle_tangency : CK = BL = (a + b - c) / 2 :=
by
  sorry

end triangle_incircle_excircle_tangency_l152_152810


namespace min_value_a_plus_b_l152_152291

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : log a (4 * b) = -1) : a + b = 1 :=
sorry

end min_value_a_plus_b_l152_152291


namespace chord_length_is_correct_l152_152718

noncomputable def length_of_chord {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : Real :=
  2 * Real.sqrt 3

theorem chord_length_is_correct {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : 
 length_of_chord h_line h_curve = 2 * Real.sqrt 3 :=
sorry

end chord_length_is_correct_l152_152718


namespace locus_of_M_equation_of_l_l152_152629
open Real

-- Step 1: Define the given circles
def circle_F1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle_F2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Step 2: Define the condition of tangency for the moving circle M
def external_tangent_F1 (cx cy r : ℝ) : Prop := (cx + 2)^2 + cy^2 = (2 + r)^2
def internal_tangent_F2 (cx cy r : ℝ) : Prop := (cx - 2)^2 + cy^2 = (6 - r)^2

-- Step 4: Prove the locus C is an ellipse with the equation excluding x = -4
theorem locus_of_M (cx cy : ℝ) : 
  (∃ r : ℝ, external_tangent_F1 cx cy r ∧ internal_tangent_F2 cx cy r) ↔
  (cx ≠ -4 ∧ (cx^2) / 16 + (cy^2) / 12 = 1) :=
sorry

-- Step 5: Define the conditions for the midpoint of segment AB
def midpoint_Q (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Step 6: Prove the equation of line l
theorem equation_of_l (x1 y1 x2 y2 : ℝ) (h1 : midpoint_Q x1 y1 x2 y2) 
  (h2 : (x1^2 / 16 + y1^2 / 12 = 1) ∧ (x2^2 / 16 + y2^2 / 12 = 1)) :
  3 * (x1 - x2) - 2 * (y1 - y2) = 8 :=
sorry

end locus_of_M_equation_of_l_l152_152629


namespace rhombus_area_from_trapezoid_midpoints_l152_152199

noncomputable def BH : ℝ := 5
noncomputable def BC : ℝ := 6
noncomputable def ABC_angle : ℝ := 120

theorem rhombus_area_from_trapezoid_midpoints
  (H : BH = 5)
  (B : BC = 6)
  (A : ABC_angle = 120) :
  let area_rhombus := 15 in
  area_rhombus = 15 :=
by 
  sorry

end rhombus_area_from_trapezoid_midpoints_l152_152199


namespace domain_and_inequality_l152_152658

noncomputable def f (x : ℝ) := real.sqrt (abs (x + 1) + abs (x + 2) - 5)

theorem domain_and_inequality 
  (a b : ℝ) 
  (h_domain : ∀ x, f x = 0 → (x ≤ -4 ∨ x ≥ 1)) 
  (h_ab : -1 < a ∧ a < 1 ∧ -1 < b ∧ b < 1) : 
  (a ∈ (-1 : ℝ) .. 1) ∧ (b ∈ (-1 : ℝ) .. 1) → 
  (abs (a + b) / 2 < abs (1 + a * b / 4)) :=
by
  sorry

end domain_and_inequality_l152_152658


namespace first_number_a10_is_91_l152_152074

def a (n : ℕ) : ℕ := 
  if n = 1 then 
    1 
  else 
    let m := n - 1 in 
    1 + 2 * (m * (m + 1) / 2)

theorem first_number_a10_is_91 : a 10 = 91 :=
by sorry

end first_number_a10_is_91_l152_152074


namespace solve_frac_eq_l152_152832

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l152_152832


namespace lamp_configurations_configurations_count_l152_152099

noncomputable def num_configurations (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2^(n-2) else 2^n

theorem lamp_configurations {n : ℕ} (h : n > 2) : 
  ∃ c : Fin n → Bool, ∀ initial : Fin n → Bool,
  ∃ steps : List (Fin n), 
  (λ final : Fin n → Bool, List.foldr (λ i f, f.update i (¬ f i).update ((i + 1) % n) (¬ f ((i + 1) % n)).update ((i - 1) % n) (¬ f ((i - 1) % n))) initial steps = c) :=
sorry

theorem configurations_count (n : ℕ) (h : n > 2) :
  num_configurations n = 
  (if n % 3 = 0 then 2^(n-2) else 2^n) :=
begin
  sorry
end

end lamp_configurations_configurations_count_l152_152099


namespace max_value_of_f_l152_152811

noncomputable def f (x : ℝ) := real.log x / x

theorem max_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ f x) ∧ f x = exp (-1) := 
sorry

end max_value_of_f_l152_152811


namespace decagon_side_length_half_R_l152_152034

theorem decagon_side_length_half_R 
  (R : ℝ) 
  (hR_pos : 0 < R) :
  let O := ({0, 0} : point)
  let circle1 := circle O R
  let pentagon1 := inscribe_regular_pentagon circle1
  let circle2 := inscribe_circle pentagon1
  let decagon1 := inscribe_regular_decagon circle2
  let side_length := side_length decagon1
  ∀ (O : point), side_length = R / 2
:= 
  sorry

end decagon_side_length_half_R_l152_152034


namespace exp_inequality_of_gt_l152_152314

theorem exp_inequality_of_gt {a b : ℝ} (h : a > b) : 2^a > 2^b :=
sorry

end exp_inequality_of_gt_l152_152314


namespace find_a2_l152_152751

def S (n : Nat) (a1 d : Int) : Int :=
  n * a1 + (n * (n - 1) * d) / 2

theorem find_a2 (a1 : Int) (d : Int) :
  a1 = -2010 ∧
  (S 2010 a1 d) / 2010 - (S 2008 a1 d) / 2008 = 2 →
  a1 + d = -2008 :=
by
  sorry

end find_a2_l152_152751


namespace nuts_per_student_l152_152183

theorem nuts_per_student (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) (total_nuts : ℕ) (nuts_per_student : ℕ)
    (h1 : bags = 65)
    (h2 : nuts_per_bag = 15)
    (h3 : students = 13)
    (h4 : total_nuts = bags * nuts_per_bag)
    (h5 : nuts_per_student = total_nuts / students)
    : nuts_per_student = 75 :=
by
  sorry

end nuts_per_student_l152_152183


namespace minute_hand_gain_per_hour_l152_152885

-- Define the conditions
def clock_gains_minutes (total_gain : ℕ) (total_hours : ℕ) : Prop :=
  total_gain = 45 ∧ total_hours = 9

-- Define the proof statement
theorem minute_hand_gain_per_hour (total_gain total_hours gain_per_hour : ℕ) 
  (h : clock_gains_minutes total_gain total_hours) : 
  gain_per_hour = 5 :=
by {
  obtain ⟨hg, hh⟩ := h,
  rw [hg, hh],
  sorry
}

end minute_hand_gain_per_hour_l152_152885


namespace tile1_in_B_l152_152577

-- Define the sides of each tile
structure Tile :=
(top : ℕ)
(right : ℕ)
(bottom : ℕ)
(left : ℕ)

-- Define each specific tile
def tile1 : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def tile2 : Tile := {top := 3, right := 1, bottom := 5, left := 2}
def tile3 : Tile := {top := 4, right := 0, bottom := 6, left := 5}
def tile4 : Tile := {top := 2, right := 4, bottom := 3, left := 0}

-- Define positions for rectangles A, B, C, D
inductive Rectangle
| A
| B
| C
| D

-- Define the target function to determine tile positions
def tile_positions : Rectangle → Tile
| Rectangle.A := -- to be determined
| Rectangle.B := tile1
| Rectangle.C := -- to be determined
| Rectangle.D := tile3

-- The problem statement translated to Lean: Proving that Tile 1 matches with Rectangle B
theorem tile1_in_B : tile_positions Rectangle.B = tile1 :=
begin
  sorry -- Proof is omitted
end

end tile1_in_B_l152_152577


namespace dodecahedron_edge_prob_is_correct_l152_152705

open Finset

-- Define a regular dodecahedron structure
structure RegularDodecahedron where
  V : Finset Nat -- Vertices
  E : Finset (Nat × Nat) -- Edges
  hV : V.card = 20 -- Number of vertices
  hE : E.card = 30 -- Number of edges
  hv_deg : ∀ v ∈ V, 2 * ((E.filter (λ e, v = e.1 ∨ v = e.2)).card) = 6 -- Each vertex is of degree 3

-- Define the probability calculation
noncomputable def dodecahedron_edge_prob (D : RegularDodecahedron) : Rat :=
  let total_ways := (D.V.card.choose 2 : Nat)
  let favorable_ways := (D.E.card : Nat)
  (favorable_ways : Rat) / (total_ways : Rat)

-- Define the theorem statement
theorem dodecahedron_edge_prob_is_correct (D : RegularDodecahedron) : 
  dodecahedron_edge_prob D = (3 : Rat) / 19 :=
by
  sorry

end dodecahedron_edge_prob_is_correct_l152_152705


namespace jerrys_breakfast_calories_l152_152044

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l152_152044


namespace square_perimeters_l152_152100

theorem square_perimeters (C_perimeter : ℝ) (D_area_factor : ℝ) (C_side : ℝ) (C_area : ℝ) (D_area : ℝ) (D_side : ℝ) (D_perimeter : ℝ) :
  C_perimeter = 32 ∧ D_area_factor = 8 ∧ 
  C_side = C_perimeter / 4 ∧ C_area = C_side * C_side ∧ 
  D_area = C_area / D_area_factor ∧ 
  D_side * D_side = D_area ∧ 
  D_perimeter = 4 * D_side → 
  D_perimeter = 8 * Real.sqrt 2 :=
by 
  intro h
  cases h with C_perimeter_eq h
  cases h with D_area_factor_eq h
  cases h with C_side_eq h
  cases h with C_area_eq h
  cases h with D_area_eq h
  cases h with D_side_eq h
  sorry

end square_perimeters_l152_152100


namespace quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l152_152506

variable (a b c d : ℝ)
variable (angle_B angle_D : ℝ)
variable (d_intersect_circle : Prop)

-- Condition that angles B and D sum up to more than 180 degrees.
def angle_condition : Prop := angle_B + angle_D > 180

-- Condition for sides of the convex quadrilateral
def side_condition1 : Prop := a + c > b + d

-- Condition for the circle touching sides a, b, and c
def circle_tangent : Prop := True -- Placeholder as no function to verify this directly in Lean

theorem quadrilateral_side_inequality (h1 : angle_condition angle_B angle_D) 
                                      (h2 : circle_tangent) 
                                      (h3 : ¬ d_intersect_circle) 
                                      : a + c > b + d :=
  sorry

theorem quadrilateral_side_inequality_if_intersect (h1 : angle_condition angle_B angle_D) 
                                                   (h2 : circle_tangent) 
                                                   (h3 : d_intersect_circle) 
                                                   : a + c < b + d :=
  sorry

end quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l152_152506


namespace pet_fee_is_120_l152_152732

noncomputable def daily_rate : ℝ := 125.00
noncomputable def rental_days : ℕ := 14
noncomputable def service_fee_rate : ℝ := 0.20
noncomputable def security_deposit : ℝ := 1110.00
noncomputable def security_deposit_rate : ℝ := 0.50

theorem pet_fee_is_120 :
  let total_stay_cost := daily_rate * rental_days
  let service_fee := service_fee_rate * total_stay_cost
  let total_before_pet_fee := total_stay_cost + service_fee
  let entire_bill := security_deposit / security_deposit_rate
  let pet_fee := entire_bill - total_before_pet_fee
  pet_fee = 120 := by
  sorry

end pet_fee_is_120_l152_152732


namespace john_has_500_dollars_l152_152729

-- Define the initial amount and the condition
def initial_amount : ℝ := 1600
def condition (spent : ℝ) : Prop := (1600 - spent) = (spent - 600)

-- The final amount of money John still has
def final_amount (spent : ℝ) : ℝ := initial_amount - spent

-- The main theorem statement
theorem john_has_500_dollars : ∃ (spent : ℝ), condition spent ∧ final_amount spent = 500 :=
by
  sorry

end john_has_500_dollars_l152_152729


namespace votes_combined_l152_152575

theorem votes_combined (vote_A vote_B : ℕ) (h_ratio : vote_A = 2 * vote_B) (h_A_votes : vote_A = 14) : vote_A + vote_B = 21 :=
by
  sorry

end votes_combined_l152_152575


namespace candle_height_after_t_over_3_l152_152898

noncomputable def total_burn_time (n : ℕ) : ℕ :=
  10 * (n * (n + 1) / 2)

noncomputable def height_after (t : ℕ) (initial_height : ℕ) : ℕ :=
  let burned_length (m : ℕ) : ℕ := m * (m + 1) / 2
  let m := nat.find (λ m, 10 * burned_length m > t)
  initial_height - (m - 1)

theorem candle_height_after_t_over_3
  (height : ℕ := 150)
  (T : ℕ := total_burn_time 150) :
  height_after (T / 3) height = 64 := by
  sorry

end candle_height_after_t_over_3_l152_152898


namespace number_of_ways_professors_choose_chairs_l152_152103

noncomputable def num_ways_professors_can_sit : ℕ := 
  6 -- the pre-computed number of acceptable ways

/- The theorem states that given the constraints,
   the number of ways professors can choose their chairs is 6. -/
theorem number_of_ways_professors_choose_chairs :
  ∀ (chairs : list ℕ),
  chairs.length = 10 →
  (∀ (p : ℕ), p ∈ {1, 10} → p ∉ {2, 3, 4, 5, 6, 7, 8, 9}) →
  (∀ (p1 p2 p3 : ℕ), p1 < p2 → p2 < p3 → 
      p2 - p1 > 2 ∧ p3 - p2 > 2) →
  (number_of_ways_professors_can_sit = 6) :=
by
  sorry

end number_of_ways_professors_choose_chairs_l152_152103


namespace imaginary_part_conjugate_l152_152270

theorem imaginary_part_conjugate (z : ℂ) (h : z * (1 + complex.i) * complex.i ^ 3 / (2 - complex.i) = 1 - complex.i) :
  complex.im (complex.conj z) = 1 := sorry

end imaginary_part_conjugate_l152_152270


namespace expected_value_of_problems_l152_152779

-- Define the setup
def num_pairs : ℕ := 5
def num_shoes : ℕ := num_pairs * 2
def prob_same_color : ℚ := 1 / (num_shoes - 1)
def days : ℕ := 5

-- Define the expected value calculation using linearity of expectation
def expected_problems_per_day : ℚ := prob_same_color
def expected_total_problems : ℚ := days * expected_problems_per_day

-- Prove the expected number of practice problems Sandra gets to do over 5 days
theorem expected_value_of_problems : expected_total_problems = 5 / 9 := 
by 
  rw [expected_total_problems, expected_problems_per_day, prob_same_color]
  norm_num
  sorry

end expected_value_of_problems_l152_152779


namespace sara_spent_on_salad_l152_152092

def cost_of_hotdog : ℝ := 5.36
def total_lunch_bill : ℝ := 10.46
def cost_of_salad : ℝ := total_lunch_bill - cost_of_hotdog

theorem sara_spent_on_salad : cost_of_salad = 5.10 :=
by
  unfold cost_of_salad
  norm_num
  exact eq.symm (by norm_num)

end sara_spent_on_salad_l152_152092


namespace canoe_stream_speed_l152_152171

theorem canoe_stream_speed (C S : ℝ) (h1 : C - S = 9) (h2 : C + S = 12) : S = 1.5 :=
by
  sorry

end canoe_stream_speed_l152_152171


namespace december_25_is_wednesday_l152_152576

def day_of_week := ℕ  -- We may use numbers to represent days of the week

noncomputable def thanksgiving_day : day_of_week := 4  -- Assuming 0=Sunday, ..., 4=Thursday

theorem december_25_is_wednesday (thx_gvn: thanksgiving_day = 4) : 
  -- Count from November 28 (Thursday) to December 25 precisely yields Wednesday
  let days_in_nov := 30  -- November has 30 days
  let days_to_dec_25 := (days_in_nov - 28) + 25 -- Days from Nov 28 to Dec 25
  let dec_25_day := (thanksgiving_day + days_to_dec_25) % 7 in 
  dec_25_day = 3 :=    -- Thus, 0=Sunday, ... , 3=Wednesday
by {
  sorry
}

end december_25_is_wednesday_l152_152576


namespace first_circle_area_l152_152502

noncomputable def area_of_first_circle {d1 d2 : ℝ} (h1: d1 = d2 / 2) (h2:  ∃ C, 6 / C = d1 ∧ C = d1 * 2 * Real.pi): ℝ :=
  let r := d1 / 2 in
  if h : (2 * Real.pi * r) ≠ 0 then
    have r_squared : ℝ := 3 / (2 * Real.pi),
    π * r_squared
  else 
    0

theorem first_circle_area : ∀ {d1 d2 : ℝ} (h1: d1 = d2 / 2) (h2: ∃ C, 6 / C = d1 ∧ C = d1 * 2 * Real.pi), area_of_first_circle h1 h2 = 3 / 2 :=
begin
  intros,
  unfold area_of_first_circle,
  split_ifs,
  { unfold area_of_first_circle._match_1 at *,
    sorry },
  { sorry }
end

end first_circle_area_l152_152502


namespace unitsDigit7Pow3Pow5_l152_152250

-- Define the repeating cycle of units digits for powers of 7
def unitsCycle : List ℕ := [7, 9, 3, 1]

-- Define the exponent
def exponent : ℕ := 3^5

-- Calculate the remainder when the exponent is divided by the cycle length
def remainder : ℕ := exponent % unitsCycle.length

-- Find the units digit of 7 to the power of the given exponent
def unitsDigit7Power : ℕ := unitsCycle.get (remainder - 1) -- List is zero-indexed

-- The main theorem
theorem unitsDigit7Pow3Pow5 : unitsDigit7Power = 3 := 
by
  have cycleH : unitsCycle = [7, 9, 3, 1] := rfl
  have expH : exponent = 243 := rfl
  have remH : remainder = 3 := rfl
  have posH : unitsCycle.get (remainder - 1) = 3 := rfl
  exact posH

#eval unitsDigit7Pow3Pow5

end unitsDigit7Pow3Pow5_l152_152250


namespace mushroom_pickers_l152_152095

theorem mushroom_pickers (m : Fin 7 → Nat)
  (h_sum : ∑ i, m i = 100)
  (h_distinct : ∀ i j, i ≠ j → m i ≠ m j) :
  ∃ i j, i ≠ j ∧ m i + m j < 36 :=
by
  sorry

end mushroom_pickers_l152_152095


namespace value_of_a3_l152_152111

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else (n * sequence (n - 1)) / (n - 1)

theorem value_of_a3 : sequence 3 = 6 :=
by
  have h1 : sequence 1 = 2 := rfl
  have h2 : sequence 2 = (1 + 1) / 1 * sequence 1, by sorry
  have h3 : sequence 3 = (2 + 1) / 2 * sequence 2, by sorry
  exact h3

end value_of_a3_l152_152111


namespace area_of_given_field_l152_152827

noncomputable def area_of_field (cost_in_rupees : ℕ) (rate_per_meter_in_paise : ℕ) (ratio_width : ℕ) (ratio_length : ℕ) : ℕ :=
  let cost_in_paise := cost_in_rupees * 100
  let perimeter := (ratio_width + ratio_length) * 2
  let x := cost_in_paise / (perimeter * rate_per_meter_in_paise)
  let width := ratio_width * x
  let length := ratio_length * x
  width * length

theorem area_of_given_field :
  let cost_in_rupees := 105
  let rate_per_meter_in_paise := 25
  let ratio_width := 3
  let ratio_length := 4
  area_of_field cost_in_rupees rate_per_meter_in_paise ratio_width ratio_length = 10800 :=
by
  sorry

end area_of_given_field_l152_152827


namespace estimate_math_score_l152_152813

/-- Definition of the given data points -/
def data_points : List (ℕ × ℕ) := [(15, 79), (23, 97), (16, 64), (24, 92), (12, 58)]

/-- Regression line equation of the form \(\hat{y} = 2.5x + \hat{a}\) -/
def regression_line (x : ℕ) (a : ℕ) : ℕ := 2.5 * x + a

/-- Mean of x-values -/
def mean_x := 18

/-- Mean of y-values -/
def mean_y := 78

/-- Determined intercept value -/
def intercept := 33

/-- Proving the estimated math score for 20 hours of study per week is 83 -/
theorem estimate_math_score :
  regression_line 20 intercept = 83 := by
  sorry

end estimate_math_score_l152_152813


namespace largest_blocks_fit_in_box_l152_152855

theorem largest_blocks_fit_in_box : 
  ∀ (block_dim : ℝ × ℝ × ℝ) (box_dim : ℝ × ℝ × ℝ),
    block_dim = (1, 2, 2) →
    box_dim = (3, 3, 2) →
    ∃ n, n = 3 ∧
         (∃ blocks : list (ℝ × ℝ × ℝ),
             (∀ blk ∈ blocks, blk = block_dim) ∧ 
             length blocks = n ∧
             (Σ' (l w h) in blocks, l * w * h) ≤ (fst box_dim * snd box_dim * (box_dim.snd.snd))
         ) := 
by sorry

end largest_blocks_fit_in_box_l152_152855


namespace Annabelle_saved_12_dollars_l152_152547

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_l152_152547


namespace minutes_before_second_rewind_l152_152541

-- Define the conditions as constants or hypotheses
constant initial_watch : ℕ
constant first_rewind : ℕ
constant second_rewind : ℕ
constant final_watch : ℕ
constant total_time : ℕ
constant x : ℕ

-- Specify the known conditions
axiom h1 : initial_watch = 35
axiom h2 : first_rewind = 5
axiom h3 : second_rewind = 15
axiom h4 : final_watch = 20
axiom h5 : total_time = 120

-- The statement to prove
theorem minutes_before_second_rewind : x = 45 :=
  begin
    -- Combine conditions and establish the equation
    have eq1 : initial_watch + first_rewind + x + second_rewind + final_watch = total_time,
    { rw [h1, h2, h3, h4, h5] },
    -- Isolate x
    sorry -- this is the part where we would solve the equation
  end

end minutes_before_second_rewind_l152_152541


namespace school_spent_440_l152_152525

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end school_spent_440_l152_152525


namespace martha_pins_l152_152072

theorem martha_pins (k : ℕ) :
  (2 + 9 * k > 45) ∧ (2 + 14 * k < 90) ↔ (k = 5 ∨ k = 6) :=
by
  sorry

end martha_pins_l152_152072


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152671

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152671


namespace side_length_of_third_pentagon_l152_152454

theorem side_length_of_third_pentagon 
  (k : ℝ)
  (A1 A2 A3 : ℝ)
  (s1 s2 s3 : ℝ)
  (h1 : s1 = 4)
  (h2 : s2 = 12)
  (h3 : \text{1st pentagon area} A1 = k * s1^2)
  (h4 : \text{2nd pentagon area} A2 = k * s2^2)
  (h5 : A2 - A1 = 3 * (A3 - A1))
  (h6 : A3 = k * s3^2) :
  s3 = 4 * real.sqrt 3 := 
by sorry

end side_length_of_third_pentagon_l152_152454


namespace intersection_points_count_l152_152657

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.log x / Real.log 2

def g (x : ℝ) : ℝ := f (f x) - 1

theorem intersection_points_count : ∃! points : Finset ℝ, Finset.card { x | g x = 0 } = 2 :=
sorry

end intersection_points_count_l152_152657


namespace prob_more_than_10_yuan_in_two_draws_prob_exactly_10_yuan_in_three_draws_l152_152218

-- Define the conditions of the problem
def balls : List String := ["Red", "Yellow", "Black"]
def reward (ball : String) : Int :=
  if ball = "Red" then 10
  else if ball = "Yellow" then 5
  else 0

def outcomes (n : Nat) : List (List String) :=
  List.replicate n balls >>= List.product

-- Prove the probability of getting more than 10 yuan from the first two draws
theorem prob_more_than_10_yuan_in_two_draws :
  let favorable := outcomes 2 |>.filter (λ draws, reward draws.head! + reward draws.get! 1 > 10)
  favorable.length * 3 = balls.length ^ 3 → 
  favorable.length.toRat / (balls.length ^ 3).toRat = 1 / 3 :=
sorry

-- Prove the probability of getting exactly 10 yuan from three draws
theorem prob_exactly_10_yuan_in_three_draws :
  let favorable := outcomes 3 |>.filter (λ draws, reward draws.head! + reward draws.get! 1 + reward draws.get! 2 = 10)
  favorable.length * 3 = balls.length ^ 3 → 
  favorable.length.toRat / (balls.length ^ 3).toRat = 2 / 9 :=
sorry

end prob_more_than_10_yuan_in_two_draws_prob_exactly_10_yuan_in_three_draws_l152_152218


namespace find_a_l152_152654

variable (a : ℝ)

def f (x : ℝ) : ℝ := 
  if x > 0 then a * x ^ 3 
  else if - (Real.pi / 2) < x ∧ x < 0 then Real.cos x
  else 0 -- It won't be used, only included to handle all ℝ cases

theorem find_a : f a (f a (-Real.pi / 3)) = 1 → a = 8 :=
by
  intro h
  sorry

end find_a_l152_152654


namespace find_alpha_l152_152302

noncomputable def g (a : ℝ) (x : ℝ) := log a (x - 3) + 2

def f (α : ℝ) (x : ℝ) := x ^ α

def M := (4 : ℝ, 2 : ℝ)

theorem find_alpha (a : ℝ) (α : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
    (h₃ : g a (4) = 2) (h₄ : f α (4) = 2) : α = 1 / 2 :=
by 
  sorry

end find_alpha_l152_152302


namespace find_set_M_l152_152675

noncomputable def A := {x : ℝ | x^2 - 5*x + 6 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | (m - 1) * x - 1 = 0}

theorem find_set_M : {m : ℝ | let B := B m in A ∩ B = B } = {3/2, 4/3, 1} :=
by sorry

end find_set_M_l152_152675


namespace laundry_detergent_ranking_l152_152522

variable {cost_per_ounce_small cost_per_ounce_medium cost_per_ounce_large : ℝ}

def ranking_based_on_cost_per_ounce (c_S c_M c_L q_S q_M q_L : ℝ) : Prop := 
  let cost_per_ounce_small := c_S / q_S
  let cost_per_ounce_medium := c_M / q_M
  let cost_per_ounce_large := c_L / q_L
  cost_per_ounce_small < cost_per_ounce_medium ∧ cost_per_ounce_medium < cost_per_ounce_large

theorem laundry_detergent_ranking :
  ranking_based_on_cost_per_ounce 
    1     -- c_S: cost of Small size
    1.6   -- c_M: cost of Medium size
    2.24  -- c_L: cost of Large size
    8     -- q_S: quantity of Small size
    10.8  -- q_M: quantity of Medium size
    12    -- q_L: quantity of Large size :=
by
  unfold ranking_based_on_cost_per_ounce
  sorry

end laundry_detergent_ranking_l152_152522


namespace volume_of_trapezoidal_prism_l152_152991

-- Definitions for the given problem
def a : ℝ := 24  -- Parallel side a of the trapezium in cm
def b : ℝ := 18  -- Parallel side b of the trapezium in cm
def d : ℝ := 15  -- Distance between the parallel sides in cm
def h : ℝ       -- Height of the prism in cm

-- Theorem to be proven
theorem volume_of_trapezoidal_prism (h : ℝ) : 
  let area : ℝ := (1 / 2) * (a + b) * d in
  let volume : ℝ := area * h in
  volume = 315 * h :=
by
  sorry

end volume_of_trapezoidal_prism_l152_152991


namespace problem_correctness_l152_152464

theorem problem_correctness :
  ∀ (x y a b : ℝ), (-3:ℝ)^2 ≠ -9 ∧
    - (x + y) = -x - y ∧
    3 * a + 5 * b ≠ 8 * a * b ∧
    5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := 
by
  intro x y a b
  split
  · norm_num
  split
  · ring
  split
  · linarith
  · ring

end problem_correctness_l152_152464


namespace equal_lengths_right_triangle_l152_152076

theorem equal_lengths_right_triangle
  (A B C K L M : Type) 
  [IsTriangle ABC] 
  [IsRightTriangle ABC C]
  [PointOn AC K] 
  [PointOn BC L] 
  [PointOn AB M]
  (AK_eq_BL : length AK = length BL)
  (KM_eq_LM : length KM = length LM)
  (Angle_KML_eq_90 : angle K M L = 90) :
  length AK = length KM := sorry

end equal_lengths_right_triangle_l152_152076


namespace f_f_1_eq_0_l152_152273

theorem f_f_1_eq_0 :
  let f : ℤ → ℤ := λ x, if x > -2 then f (x - 1) else x^2 + 2 * x - 3 in
  f (f 1) = 0 :=
by {
  admit
}

end f_f_1_eq_0_l152_152273


namespace cube_root_of_one_over_eight_eq_half_sqrt_of_double_sqrt_eq_pos_neg_sqrt_six_l152_152423

-- Definition for cube root of a fraction
def cube_root_of_fraction (a b : ℝ) : ℝ :=
  if b ≠ 0 then (a / b)^(1/3) else 0

-- Definition for square root of nested square
def double_sqrt_of_square (x : ℝ) : ℝ :=
  real.sqrt (real.sqrt (x^2))

-- Theorem to check the cube root of 1/8
theorem cube_root_of_one_over_eight_eq_half : cube_root_of_fraction 1 8 = 1/2 :=
by 
  sorry

-- Theorem to check the square root of sqrt((-6)^2)
theorem sqrt_of_double_sqrt_eq_pos_neg_sqrt_six : 
  double_sqrt_of_square (-6) = real.sqrt 6 ∨ double_sqrt_of_square (-6) = -real.sqrt 6 :=
by 
  sorry

end cube_root_of_one_over_eight_eq_half_sqrt_of_double_sqrt_eq_pos_neg_sqrt_six_l152_152423


namespace num_integers_D_l152_152573

theorem num_integers_D :
  ∃ (D : ℝ) (n : ℕ), 
    (∀ (a b : ℝ), -1/4 < a → a < 1/4 → -1/4 < b → b < 1/4 → abs (a^2 - D * b^2) < 1) → n = 32 :=
sorry

end num_integers_D_l152_152573


namespace f_increasing_interval_maximum_triangle_area_l152_152660

namespace Problem

-- Conditions and definitions
def f (x : ℝ) : ℝ := sin (x) * cos (x) + sqrt 3 * sin (x) * sin (x) - sqrt 3 / 2
def ω := 1 -- given ω such that the smallest positive period is π

-- Question (1)
theorem f_increasing_interval (k : ℤ) :
  ∀ (k : ℤ), f (x) is increasing on the interval [(k : ℝ) * π - π / 12, (k : ℝ) * π + 5 * π / 12] :=
sorry

-- Definitions for the triangle question
def triangle_area (a b c : ℝ) : ℝ := a * b * sin (c) / 2

-- Question (2)
theorem maximum_triangle_area (b c : ℝ) (a : ℝ := 1) :
  ∀ (A : ℝ), sin (A) = 1 / 2 → cos (A) = sqrt 3 / 2 →
  triangle_area a b c = (2 + sqrt 3) / 4 :=
sorry

end Problem

end f_increasing_interval_maximum_triangle_area_l152_152660


namespace probability_of_one_head_l152_152846

-- Define the sample space for tossing a coin twice
def sample_space : set (string × string) := {("H","H"), ("H","T"), ("T","H"), ("T","T")}

-- Define the event of getting exactly one head
def event_one_head : set (string × string) := {("H","T"), ("T","H")}

-- Define the probability measure for the sample space
def probability (A : set (string × string)) : ℚ :=
  (A.to_finset.card : ℚ) / (sample_space.to_finset.card : ℚ)

-- The question is to prove that the probability of getting exactly one head is 1/2
theorem probability_of_one_head : probability event_one_head = 1 / 2 :=
by sorry

end probability_of_one_head_l152_152846


namespace eccentricity_of_circle_l152_152649

theorem eccentricity_of_circle {m n : ℝ} (h_n_pos : n > 0) (h_m_pos : m > 0) (h_hyperbola_ecc : Real.eccentricity (conic_section.hyperbola 1 m n) = 2) :
  Real.eccentricity (conic_section.circle 1 m n) = sqrt(6)/3 :=
sorry

end eccentricity_of_circle_l152_152649


namespace bob_walked_distance_l152_152769

theorem bob_walked_distance :
  ∀ (distance_x_y : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) (start_delay : ℝ),
    distance_x_y = 40 →
    yolanda_speed = 2 →
    bob_speed = 4 →
    start_delay = 1 →
    let t := (distance_x_y - yolanda_speed * start_delay) / (yolanda_speed + bob_speed) in
    let D_b := bob_speed * t in
    D_b = 25 + 1/3 :=
begin
  intros distance_x_y yolanda_speed bob_speed start_delay,
  intros h1 h2 h3 h4,
  let t := (distance_x_y - yolanda_speed * start_delay) / (yolanda_speed + bob_speed),
  let D_b := bob_speed * t,
  rw [h1, h2, h3, h4] at *,
  calc D_b
      = bob_speed * ((40 - 2 * 1) / (2 + 4)) : by rw h3
  ... = 4 * ((40 - 2) / 6) : by rw [h2, h4]
  ... = 4 * 38 / 6
  ... = 152 / 6
  ... = 25 + 1 / 3 : by norm_num
end

end bob_walked_distance_l152_152769


namespace math_problem_l152_152470

theorem math_problem 
    (A_correct : (let l := (λ x y : ℝ, x - y - 2 = 0) in ∃ (x1 x2 : ℝ), l x1 0 ∧ l 0 x2 ∧ (x1 * x2) / 2 = 2))
    (B_correct : (let symmetric_point := (0, 2) in ∃ (sym_pt : ℝ × ℝ), (sym_pt = (1, 1)) ∧ (∀ (x y : ℝ), y = x + 1 → symmetric_point.2 = symmetric_point.1 + 1))) :
  (A_correct ∧ B_correct) :=
by
  -- proof placeholder
  sorry

end math_problem_l152_152470


namespace find_circle_radius_l152_152107

theorem find_circle_radius (AB EO : ℝ) (h_AB : AB = 18) (h_EO : EO = 7) (AE BE R : ℝ) (h_ratio : AE = 2 * BE) (h_sum : AE + BE = AB) :
  R = 11 :=
by
  have h_3x : 3 * BE = 18, by linarith [h_ratio, h_sum]
  have h_BE : BE = 6, by linarith [h_3x]
  have h_AE : AE = 12, by linarith [h_BE, h_ratio]
  have h_product : AE * BE = 72, by linarith [h_AE, h_BE]
  have h_power : R ^ 2 - EO ^ 2 = AE * BE, by linarith [h_product, h_EO]
  have h_equation : R ^ 2 - 49 = 72, by linarith [h_power, h_EO]
  have h_radius : R ^ 2 = 121, by linarith [h_equation]
  exact (Real.sqrt_eq_iff_sq_eq.mp (by norm_num : sqrt 121 = 11)).mp (by norm_num)

end find_circle_radius_l152_152107


namespace prob_exactly_three_even_l152_152219

theorem prob_exactly_three_even 
  (n : ℕ) (k : ℕ) (p_even : ℚ)
  (h_n : n = 6) 
  (h_k : k = 3) 
  (h_p_even : p_even = 1 / 2) :
  let p_odd := 1 - p_even in
  let binom_coeff := Nat.choose n k in
  let config_prob := (p_even ^ k) * (p_odd ^ (n - k)) in
  let total_prob := (binom_coeff : ℚ) * config_prob in
  total_prob = 5 / 16 := 
by 
  sorry

end prob_exactly_three_even_l152_152219


namespace carpet_area_correct_l152_152951

def shoe_length := 28 -- in cm
def length_shoes := 15
def width_shoes := 10

def carpet_length := length_shoes * shoe_length
def carpet_width := width_shoes * shoe_length
def carpet_area := carpet_length * carpet_width

theorem carpet_area_correct : carpet_area = 117600 :=
by 
  have length_cm : carpet_length = 15 * 28 := rfl
  have width_cm : carpet_width = 10 * 28 := rfl
  have area_cm2 : carpet_area = (15 * 28) * (10 * 28) := by 
    rw [length_cm, width_cm]
  rw [← Nat.mul_assoc, Nat.mul_comm 28 _, Nat.mul_assoc, Nat.mul_assoc]
  norm_num
  sorry

end carpet_area_correct_l152_152951


namespace original_number_l152_152193

theorem original_number (x : ℝ) (h : 1.10 * x = 550) : x = 500 :=
by
  sorry

end original_number_l152_152193


namespace desired_depth_is_50_l152_152493

noncomputable def desired_depth_dig (d days : ℝ) : ℝ :=
  let initial_man_hours := 45 * 8 * d
  let additional_man_hours := 100 * 6 * d
  (initial_man_hours / additional_man_hours) * 30

theorem desired_depth_is_50 (d : ℝ) : desired_depth_dig d = 50 :=
  sorry

end desired_depth_is_50_l152_152493


namespace work_rate_c_l152_152164

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 4) (h2 : B + C = 1 / 6) (h3 : C + A = 1 / 3) :
    1 / C = 8 :=
by
  sorry

end work_rate_c_l152_152164


namespace sphere_volume_l152_152507

/-
  A cube has all its vertices on the surface of a sphere,
  and its edge length is 2 cm.
  We need to prove that the volume of the sphere is 4sqrt(3)π.
-/

def edge_length_of_cube := 2

const volume_of_sphere := 4 * Real.sqrt 3 * Real.pi

theorem sphere_volume :
  ∃ (r : ℝ), (∀ (edge_length_of_cube : ℝ = 2), r = Real.sqrt 3) →
  (4 / 3) * Real.pi * r ^ 3 = volume_of_sphere :=
begin
  sorry
end

end sphere_volume_l152_152507


namespace inequality_solution_set_range_of_values_l152_152656

def f (x : ℝ) : ℝ := abs (2 * x - 1)

theorem inequality_solution_set :
  {x : ℝ | -3 / 2 < x ∧ x < 5 / 2} = {x : ℝ | ∃ y : ℝ, y = f x ∧ 0 < y ∧ y < 4} :=
by {
  sorry
}

def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem range_of_values (a : ℝ) (m n : ℝ) (h1 : m + n = a) (h2 : a = 2) (h3 : m > 0) (h4 : n > 0) :
  ∃ t : ℝ, t ∈ Set.Ici (3 / 2 + Real.sqrt 2) ∧ ∀ (m n : ℝ), m > 0 → n > 0 → m + n = 2 → t = 2 / m + 1 / n :=
by {
  sorry
}

end inequality_solution_set_range_of_values_l152_152656


namespace garden_area_remaining_l152_152884

variable (d : ℕ) (w : ℕ) (t : ℕ)

theorem garden_area_remaining (r : Real) (A_circle : Real) 
                              (A_path : Real) (A_remaining : Real) :
  r = 10 →
  A_circle = 100 * Real.pi →
  A_path = 66.66 * Real.pi - 50 * Real.sqrt 3 →
  A_remaining = 33.34 * Real.pi + 50 * Real.sqrt 3 :=
by
  -- Given the radius of the garden
  let r := (d : Real) / 2
  -- Calculate the total area of the garden
  let A_circle := Real.pi * r^2
  -- Area covered by the path computed using circular segments
  let A_path := 66.66 * Real.pi - 50 * Real.sqrt 3
  -- Remaining garden area
  let A_remaining := A_circle - A_path
  -- Statement to prove correct
  sorry 

end garden_area_remaining_l152_152884


namespace solution_set_eq_neg_infty_0_l152_152227

noncomputable theory

open Real

variable (f : ℝ → ℝ)

axiom f_property1 : f 1 = 1
axiom f_property2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_eq_neg_infty_0 :
  { x : ℝ | f (exp x) > (exp x + 1) / 2 } = set.Iio 0 :=
sorry

end solution_set_eq_neg_infty_0_l152_152227


namespace part1_part2_l152_152668

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l152_152668


namespace product_of_sequence_l152_152912

def a : ℕ → ℚ
| 0       := 1 / 3
| (n + 1) := 2 + (a n - 2)^2

theorem product_of_sequence :
  (∏ n, a n) = 6 / 11 :=
by
  sorry

end product_of_sequence_l152_152912


namespace classrooms_count_l152_152128

theorem classrooms_count (num_buses : ℕ) (seats_per_bus : ℕ) (students_per_classroom : ℕ) 
  (h_buses : num_buses = 737) (h_seats : seats_per_bus = 6) (h_students : students_per_classroom = 66) : 
  num_buses * seats_per_bus / students_per_classroom = 67 := by
  have h1 : num_buses * seats_per_bus = 737 * 6, by rw [h_buses, h_seats]
  have h2 : 737 * 6 = 4422, by norm_num
  have h3 : students_per_classroom = 66, by rw [h_students]
  have h4 : 4422 / students_per_classroom = 4422 / 66, by rw [h3]
  have h5 : 4422 / 66 = 67, by norm_num
  rw [h1, h2, h4, h5]
  sorry

end classrooms_count_l152_152128


namespace milk_price_per_liter_l152_152799

theorem milk_price_per_liter (M : ℝ) 
  (price_fruit_per_kg : ℝ) (price_each_fruit_kg_eq_2: price_fruit_per_kg = 2)
  (milk_liters_per_batch : ℝ) (milk_liters_per_batch_eq_10: milk_liters_per_batch = 10)
  (fruit_kg_per_batch : ℝ) (fruit_kg_per_batch_eq_3 : fruit_kg_per_batch = 3)
  (cost_three_batches : ℝ) (cost_three_batches_eq_63: cost_three_batches = 63) :
  M = 1.5 :=
by
  sorry

end milk_price_per_liter_l152_152799


namespace andrea_reaches_lauren_in_25_minutes_l152_152214

noncomputable def initial_distance : ℝ := 30
noncomputable def decrease_rate : ℝ := 90
noncomputable def Lauren_stop_time : ℝ := 10 / 60

theorem andrea_reaches_lauren_in_25_minutes :
  ∃ v_L v_A : ℝ, v_A = 2 * v_L ∧ v_A + v_L = decrease_rate ∧ ∃ remaining_distance remaining_time final_time : ℝ, 
  remaining_distance = initial_distance - decrease_rate * Lauren_stop_time ∧ 
  remaining_time = remaining_distance / v_A ∧ 
  final_time = Lauren_stop_time + remaining_time ∧ 
  final_time * 60 = 25 :=
sorry

end andrea_reaches_lauren_in_25_minutes_l152_152214


namespace coefficient_of_x6_l152_152713

-- Define combinatorial binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

-- Define the general term of the binomial expansion
def general_term (n k : ℕ) : ℕ :=
binom n k * (-2) ^ k

-- Prove the coefficient of x^6 in the expansion of (x - 2/x)^{10} is 180
theorem coefficient_of_x6 :
  general_term 10 2 = 180 :=
by
  -- We are expected to find and confirm this step from the conditions and problem statement
  sorry

end coefficient_of_x6_l152_152713


namespace inequality_and_angles_l152_152271

noncomputable def circumradius (A B C : Point) : Real := sorry
-- Assume a function that retrieves the circumradius of a triangle formed by 3 points.

variables {A B C D : Point} -- Points representing vertices of the quadrilateral
variables {α β γ δ : Real} -- Angles (in radians) for ∠CAB (α), ∠CDB (β), and their respective supplementary angles.

-- Define the circumradii of the triangles:
def R_a := circumradius D A B
def R_b := circumradius A B C
def R_c := circumradius B C D
def R_d := circumradius C D A

-- The theorem to prove the equivalence of the conditions and the inequality:
theorem inequality_and_angles 
  (h1: 180 - β < α ∧ α < β) 
  (h2: β > 90)
  : R_a < R_b ∧ R_b < R_c ∧ R_c < R_d ↔ 180 - β < α ∧ α < β := 
by
  sorry

end inequality_and_angles_l152_152271


namespace probability_differs_by_three_l152_152931

theorem probability_differs_by_three :
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 3), (8, 5)],
      num_outcomes := List.length outcomes,
      total_possibilities := 8 * 8
  in
  Rational.mk num_outcomes total_possibilities = Rational.mk 7 64 :=
by
  sorry

end probability_differs_by_three_l152_152931


namespace find_valid_pairs_l152_152236

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_pair (p q : ℕ) : Prop :=
  p < 2005 ∧ q < 2005 ∧ is_prime p ∧ is_prime q ∧ q ∣ p^2 + 8 ∧ p ∣ q^2 + 8

theorem find_valid_pairs :
  ∀ p q, valid_pair p q → (p, q) = (2, 2) ∨ (p, q) = (881, 89) ∨ (p, q) = (89, 881) :=
sorry

end find_valid_pairs_l152_152236


namespace range_of_f_t_eq_1_on_0_to_4_range_of_a_for_f_le_5_range_of_t_for_diff_le_8_l152_152068

variable (t : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2*t*x + 2

-- Problem 1
theorem range_of_f_t_eq_1_on_0_to_4 :
  t = 1 → set.range (λ x → f t x) (set.Icc 0 4) = set.Icc 1 10 :=
by
  intro ht
  rw [ht]
  sorry

-- Problem 2
theorem range_of_a_for_f_le_5 :
  t = 1 → (∀ x ∈ set.Icc a (a + 2), f t x ≤ 5) → a ∈ set.Icc (-1) 1 :=
by
  intro ht h
  rw [ht]
  sorry

-- Problem 3
theorem range_of_t_for_diff_le_8 :
  (∀ x1 x2 ∈ set.Icc 0 4, abs (f t x1 - f t x2) ≤ 8) → t ∈ set.Icc (4 - 2 * real.sqrt 2) (2 * real.sqrt 2) :=
by
  intro h
  sorry

end range_of_f_t_eq_1_on_0_to_4_range_of_a_for_f_le_5_range_of_t_for_diff_le_8_l152_152068


namespace apples_in_each_bag_l152_152891

variable (x : ℕ)
variable (total_children : ℕ)
variable (eaten_apples : ℕ)
variable (sold_apples : ℕ)
variable (remaining_apples : ℕ)

theorem apples_in_each_bag
  (h1 : total_children = 5)
  (h2 : eaten_apples = 2 * 4)
  (h3 : sold_apples = 7)
  (h4 : remaining_apples = 60)
  (h5 : total_children * x - eaten_apples - sold_apples = remaining_apples) :
  x = 15 :=
by
  sorry

end apples_in_each_bag_l152_152891


namespace min_max_expression_l152_152377

theorem min_max_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ m M : ℝ, (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → min (3 * |a + b| / (|a| + |b|)) = m) ∧
             (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → max (3 * |a + b| / (|a| + |b|)) = M) ∧
             (M - m = 3) :=
by
  sorry

end min_max_expression_l152_152377


namespace triangle_right_angled_l152_152697

theorem triangle_right_angled {A B C : ℝ} {a b c : ℝ} 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_triangle : sin A + sin B = sin C * (cos A + cos B)) 
  (h_sines_A : a = b * cos C + c * cos B)
  (h_sines_B : b = c * cos A + a * cos C) 
  (h_sines_C : c = a * sin B + b * sin A):
  C = π / 2 := 
by
  sorry

end triangle_right_angled_l152_152697


namespace problem_part1_problem_part2_l152_152051

-- Definitions based on the problem conditions.
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -2)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (5, -4)

-- Vector definitions based on given conditions.
def vector_sub (p1 p2: ℝ × ℝ) : ℝ × ℝ := (p1.1 - p2.1, p1.2 - p2.2)
def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

def AB : ℝ × ℝ := vector_sub B A
def CD : ℝ × ℝ := vector_sub D C

def a : ℝ × ℝ := AB
def b : ℝ × ℝ := vector_sub C B
def k : ℝ := -1 / 3

-- Statements that need to be proved
theorem problem_part1 : AB = CD → D = (5, -4) := by
  sorry

theorem problem_part2 : (∀ k : ℝ, vector_sub (vector_scale k a) b = vector_sub (7, 4) (0,0)) → k = -1 / 3 := by
  sorry

end problem_part1_problem_part2_l152_152051


namespace min_sum_p_q_r_s_l152_152066

theorem min_sum_p_q_r_s (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (h1 : 2 * p = 10 * p - 15 * q)
    (h2 : 2 * q = 6 * p - 9 * q)
    (h3 : 3 * r = 10 * r - 15 * s)
    (h4 : 3 * s = 6 * r - 9 * s) : p + q + r + s = 45 := by
  sorry

end min_sum_p_q_r_s_l152_152066


namespace inequality_relation_l152_152871

theorem inequality_relation (a b : ℝ) :
  (∃ a b : ℝ, a > b ∧ ¬(1/a < 1/b)) ∧ (∃ a b : ℝ, (1/a < 1/b) ∧ ¬(a > b)) :=
by {
  sorry
}

end inequality_relation_l152_152871


namespace impossible_three_stone_piles_l152_152077

theorem impossible_three_stone_piles
  (initial_stones : ℕ)
  (initial_piles : ℕ)
  (moves : ℕ → ℕ → ℕ)
  (stone_pile_relation : ∀ (k l : ℕ), moves k l = (k - 1) + 2 * l) :
  initial_stones = 1001 → initial_piles = 1 →
  ¬(∃ (n : ℕ), 3 * n + n = 1002) :=
by
  intros h1 h2
  have h3 : ∀ (n : ℕ), 4 * n ≠ 1002 := 
    λ n, by
      intro h
      have hn : n = 250.5 := by sorry -- Mathematical proof that no natural number n satisfies 4n = 1002
      sorry
  exact h3

end impossible_three_stone_piles_l152_152077


namespace least_four_digit_palindrome_divisible_by_3_l152_152770

def is_palindrome (n : Nat) : Prop :=
  let str := toString n
  str == str.reverse

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_3 (n : Nat) : Prop :=
  n % 3 = 0

def is_valid_candidate (n : Nat) : Prop :=
  is_palindrome n ∧ is_four_digit n ∧ is_divisible_by_3 n

theorem least_four_digit_palindrome_divisible_by_3 :
  ∀ n : Nat, is_valid_candidate n → 1221 ≤ n :=
sorry

end least_four_digit_palindrome_divisible_by_3_l152_152770


namespace sara_spent_on_salad_l152_152090

def cost_of_hotdog : ℝ := 5.36
def total_lunch_bill : ℝ := 10.46
def cost_of_salad : ℝ := total_lunch_bill - cost_of_hotdog

theorem sara_spent_on_salad : cost_of_salad = 5.10 :=
by
  unfold cost_of_salad
  norm_num
  exact eq.symm (by norm_num)

end sara_spent_on_salad_l152_152090


namespace area_of_portion_of_circle_l152_152853

noncomputable def circle_area_condition : ℝ := (99 * Real.pi) / 8

theorem area_of_portion_of_circle :
  let circle_eq := ∀ (x y : ℝ), x^2 - 14 * x + y^2 = 50
  let line_eq := ∀ (x y : ℝ), y = x - 4
  in ∃ (a : ℝ), a = circle_area_condition 
  ∧ (∀ (x y : ℝ), y < 0 → x = y + 4 → (x^2 - 14 * x + y^2 ≤ 50)) :=
by
  sorry

end area_of_portion_of_circle_l152_152853


namespace at_least_one_pass_l152_152836

variable (n : ℕ) (p : ℝ)

theorem at_least_one_pass (h_p_range : 0 < p ∧ p < 1) :
  (1 - (1 - p) ^ n) = 1 - (1 - p) ^ n :=
sorry

end at_least_one_pass_l152_152836


namespace quadratic_roots_range_l152_152275

theorem quadratic_roots_range (a : ℝ) :
  (a-1) * x^2 - 2*x + 1 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a-1) * x1^2 - 2*x1 + 1 = 0 ∧ (a-1) * x2^2 - 2*x2 + 1 = 0) → (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_roots_range_l152_152275


namespace sum_of_elements_of_T_l152_152740

def is_repeating_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧ x = (10 * a + b) / 99

def sum_of_repeating_decimals : ℝ :=
  ∑ x in { x : ℝ | is_repeating_decimal x }, x

theorem sum_of_elements_of_T : sum_of_repeating_decimals = 45 :=
by
  sorry

end sum_of_elements_of_T_l152_152740


namespace part_a_possible_part_b_not_possible_l152_152967

section GroupOfPeople

variable {People : Type} (knows : People → People → Prop) [symm : Symmetric knows]
variable (group : Finset People) [DecidableEq People]
variable (n : ℕ)

-- Condition: There are 15 people in the group
axiom people_count : group.card = 15

-- Definition: Degree of a person is the number of people they know
def degree (person : People) : ℕ := (group.filter (knows person)).card

-- Part (a): Prove it is possible that each person knows exactly 4 other people
theorem part_a_possible (h4 : ∀ p ∈ group, degree knows group p = 4) : ∃ graph, (∀ p ∈ group, degree knows group p = 4) :=
sorry

-- Part (b): Prove it is not possible that each person knows exactly 3 other people
theorem part_b_not_possible (h3 : ∀ p ∈ group, degree knows group p = 3) : ¬ ∃ graph, (∀ p ∈ group, degree knows group p = 3) :=
sorry

end GroupOfPeople

end part_a_possible_part_b_not_possible_l152_152967


namespace coeff_x8_sum_weights_l152_152797

theorem coeff_x8_sum_weights :
  ∀ n : ℕ, n = 8 →
  coefficient (x ^ n) (expand ((1 + x) * (1 + x^2) * (1 + x^3) * ... * (1 + x^10))) =
  (number_of_ways_choose_weights_to_sum_to 8) :=
begin
  sorry
end

end coeff_x8_sum_weights_l152_152797


namespace minimum_in_interval_l152_152556

noncomputable def f (x : ℝ) := x^x

theorem minimum_in_interval (h₁ : 0 < x) (h₂ : x < 1) :
  ∃ c ∈ Ioo 0.3 0.4, ∀ y ∈ Ioo 0 1, f c ≤ f y :=
sorry

end minimum_in_interval_l152_152556


namespace calculate_y_l152_152686

theorem calculate_y (x y : ℝ) (h1 : x = 101) (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : y = 1 / 10 :=
by
  sorry

end calculate_y_l152_152686


namespace sum_of_digits_of_d_l152_152768

theorem sum_of_digits_of_d (d : ℕ) (d_canadian : ℕ) 
  (h1 : d_canadian = (15 * d) / 11) 
  (h2 : d_canadian - 75 = d) : 
  (d.digits.sum = 8) := by
sorry

end sum_of_digits_of_d_l152_152768


namespace seating_arrangement_l152_152372

-- Definitions and conditions from the problem
def n : ℕ := 2020

def D : ℕ → ℕ
| 0     := 1
| 1     := 0
| n + 2 := (n + 1) * (D (n + 1) + D n)

def f (n : ℕ) : ℕ := ∑ k in Finset.range n, (-1)^k * D (n - k)

-- The problem statement
theorem seating_arrangement : f 2020 = ∑ k in Finset.range 2020, (-1)^k * D (2020 - k) :=
sorry -- Proof not required

end seating_arrangement_l152_152372


namespace problem1_problem2_l152_152265

-- Problem 1: Prove that (2sin(α) - cos(α)) / (sin(α) + 2cos(α)) = 3/4 given tan(α) = 2
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

-- Problem 2: Prove that 2sin^2(x) - sin(x)cos(x) + cos^2(x) = 2 - sin(2x)/2
theorem problem2 (x : ℝ) : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 2 - Real.sin (2 * x) / 2 := 
sorry

end problem1_problem2_l152_152265


namespace outdoor_chairs_count_l152_152883

theorem outdoor_chairs_count (indoor_tables outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) 
  (total_chairs : ℕ) (h1: indoor_tables = 9) (h2: outdoor_tables = 11) 
  (h3: chairs_per_indoor_table = 10) (h4: total_chairs = 123) : 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 :=
by 
  sorry

end outdoor_chairs_count_l152_152883


namespace percentage_increase_is_correct_l152_152481

-- Define the original and new weekly earnings
def original_earnings : ℕ := 60
def new_earnings : ℕ := 90

-- Define the percentage increase calculation
def percentage_increase (original new : ℕ) : Rat := ((new - original) / original: Rat) * 100

-- State the theorem that the percentage increase is 50%
theorem percentage_increase_is_correct : percentage_increase original_earnings new_earnings = 50 := 
sorry

end percentage_increase_is_correct_l152_152481


namespace find_expression_l152_152640

-- Define the polynomial and the fact that a and b are roots.
def poly (x : ℝ) := x^2 + 3 * x - 4

-- Assuming a and b are roots of the polynomial
variables (a b : ℝ)
hypothesis h_a_root : poly a = 0
hypothesis h_b_root : poly b = 0

-- Prove that a^2 + 4a + b - 3 = -2 given the above assumptions
theorem find_expression (a b : ℝ) (h_a_root : poly a = 0) (h_b_root : poly b = 0) : a^2 + 4 * a + b - 3 = -2 :=
by sorry

end find_expression_l152_152640


namespace find_f_neg2_l152_152180

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x, f(x) + x = f(-x) - x
axiom h2 : f(2) = 1

-- Question to prove
theorem find_f_neg2 : f(-2) = 5 := by
  sorry

end find_f_neg2_l152_152180


namespace volume_of_solid_l152_152530

theorem volume_of_solid (h : ℝ) (a0 a1 a2 a3 : ℝ) :
  let A := λ k : ℝ, a0 + a1 * k + a2 * k^2 + a3 * k^3
  let B := A (-h/2)
  let M := A 0
  let T := A (h/2)
  let V := ∫ k in (-h/2)..(h/2), A k
  in V = h * (B + 4 * M + T) / 6 :=
by
  let A := λ k : ℝ, a0 + a1 * k + a2 * k^2 + a3 * k^3
  let B := A (-h/2)
  let M := A 0
  let T := A (h/2)
  let V := ∫ k in (-h/2)..(h/2), A k
  have h1 : ∫ k in (-h/2)..(h/2), A k = a0 * h + (a2 * h^3) / 12 := sorry
  have h2 : (B + 4 * M + T) = 6 * a0 + (a2 * h^2) / 2 := sorry
  exact (calc
    V = a0 * h + (a2 * h^3) / 12                        : by rw h1
    ... = h * (B + 4 * M + T) / 6                        : by rw h2
  )

end volume_of_solid_l152_152530


namespace binomial_coefficients_mod_3_l152_152600

theorem binomial_coefficients_mod_3 (n : ℕ) (a_n b_n : ℕ) :
  (a_n = ((Finset.range (n + 1)).filter (λ k, Nat.choose n k % 3 = 1)).card) →
  (b_n = ((Finset.range (n + 1)).filter (λ k, Nat.choose n k % 3 = 2)).card) →
  a_n > b_n :=
by
  sorry

end binomial_coefficients_mod_3_l152_152600


namespace abs_val_ineq_solution_set_l152_152448

theorem abs_val_ineq_solution_set : {x : ℝ | |x| > -1} = set.univ :=
by
  sorry

end abs_val_ineq_solution_set_l152_152448


namespace problem_n_multiple_of_4_l152_152602

theorem problem_n_multiple_of_4 (n : ℤ) (hn : n ≥ 15) (ho : n % 2 = 1) : 
  (∃ k : ℤ, (fact (n + 3) - fact (n + 2)) / fact n = 4 * k) := 
sorry

end problem_n_multiple_of_4_l152_152602


namespace largest_n_five_partition_l152_152251

theorem largest_n_five_partition (n : ℕ) : n ≤ 15 → 
  ∃ (A B : set ℕ) (S : set ℕ), S ⊆ (finset.range (n + 1)).to_set ∧ S.card = 5 ∧
  (A ⊆ S) ∧ (B ⊆ S) ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A ∩ B = ∅ ∧ 
  (A.sum id = B.sum id) := 
begin
  sorry
end

end largest_n_five_partition_l152_152251


namespace z_in_fourth_quadrant_l152_152352

def complex_quadrant (re im : ℤ) : String :=
  if re > 0 ∧ im > 0 then "First Quadrant"
  else if re < 0 ∧ im > 0 then "Second Quadrant"
  else if re < 0 ∧ im < 0 then "Third Quadrant"
  else if re > 0 ∧ im < 0 then "Fourth Quadrant"
  else "Axis"

theorem z_in_fourth_quadrant : complex_quadrant 2 (-3) = "Fourth Quadrant" :=
by
  sorry

end z_in_fourth_quadrant_l152_152352


namespace cube_maximum_clean_visits_l152_152175

theorem cube_maximum_clean_visits (board_width board_height : ℕ) 
  (cube_face_painted : Prop) (cells_on_board : ℕ)
  (board : Matrix board_height board_width bool) :
  board_width = 12 → board_height = 7 →
  cells_on_board = 7 * 12 →
  (∀ (x y : ℕ), x < 7 → y < 12 → board[x, y] = true) →
  (∀ (cube_pos : Fin 7 × Fin 12), 
     ∃! (visited_cells : Finset (Fin 7 × Fin 12)),
     visited_cells.card = cells_on_board ∧ 
     ∀ (cell : Fin 7 × Fin 12), cell ∈ visited_cells → 
       (cube_rolls : List (Fin 7 × Fin 12)) → 
       cube_face_painted = false → 
       length cube_rolls = visited_cells.card) :=
sorry

end cube_maximum_clean_visits_l152_152175


namespace breadth_of_rectangular_plot_l152_152120

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : ∃ l : ℝ, l = 3 * b) (h2 : b * 3 * b = 675) : b = 15 :=
by
  sorry

end breadth_of_rectangular_plot_l152_152120


namespace cubes_with_even_faces_l152_152921

theorem cubes_with_even_faces (length width height : ℕ) (h1 : length = 6) (h2 : width = 3) (h3 : height = 2) :
  ∃ n : ℕ, n = 16 ∧ ∀ (x y z : ℕ), (x, y, z) ∈ finset.range length × finset.range width × finset.range height →
  (let painted_faces := (if x = 0 ∨ x = length - 1 then 1 else 0) + (if y = 0 ∨ y = width - 1 then 1 else 0) + 
                        (if z = 0 ∨ z = height - 1 then 1 else 0) in
   painted_faces % 2 = 0) →
  n = 16 := by
  intros
  sorry

end cubes_with_even_faces_l152_152921


namespace solve_for_x_l152_152097

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 5) * x = 2) : x = 17.5 :=
by
  -- Here we acknowledge the initial condition and conclusion without proving
  sorry

end solve_for_x_l152_152097


namespace solve_equation_l152_152098

theorem solve_equation : ∃ x : ℝ, 81 = 4 * (16^(x-2)) ∧ x = (Real.log 1.125 / Real.log 4 + 4) / 2 :=
by
  sorry

end solve_equation_l152_152098


namespace probability_of_drawing_fruit_card_l152_152843

theorem probability_of_drawing_fruit_card : 
  let total_cards := 6
  let fruit_cards := 2
  total_cards ≠ 0 → (fruit_cards.toFloat / total_cards.toFloat) = (1 / 3 : Float) :=
by
  intros h
  have h1 : fruit_cards.toFloat / total_cards.toFloat = (2 / 6 : Float), by sorry
  have h2 : (2 / 6 : Float) = (1 / 3 : Float), by sorry
  exact Eq.trans h1 h2

end probability_of_drawing_fruit_card_l152_152843


namespace joan_initial_dimes_l152_152363

example (spent left initial : ℕ) (h1 : spent = 2) (h2 : left = 3) : initial = spent + left := by
  sorry

theorem joan_initial_dimes : by
  have spent : ℕ := 2
  have left : ℕ := 3
  show spent + left = 5, by
  sorry

end joan_initial_dimes_l152_152363


namespace largest_n_complete_graph_arith_prog_l152_152229

theorem largest_n_complete_graph_arith_prog :
  ∃ n, n ≥ 3 ∧ (∀ E : Finset (Fin n.succ), 
  ∀ (f : E → ℕ), 
  (∀ {a b c : Fin n.succ} (hab : (⟨a, b⟩ ∈ E) ) (hbc : (⟨b, c⟩ ∈ E) ) (hca : (⟨c, a⟩ ∈ E) ),
  f ⟨a, b⟩ ≠ f ⟨b, c⟩ ∧ f ⟨b, c⟩ ≠ f ⟨c, a⟩ ∧ f ⟨c, a⟩ ≠ f ⟨a, b⟩ ∧
  (2 * f ⟨b, c⟩ = f ⟨a, b⟩ + f ⟨c, a⟩))) -> n = 4 := 
begin
  sorry
end

end largest_n_complete_graph_arith_prog_l152_152229


namespace rational_numbers_satisfying_conditions_l152_152597

theorem rational_numbers_satisfying_conditions :
  (∃ n : ℕ, n = 166 ∧ ∀ (m : ℚ),
  abs m < 500 → (∃ x : ℤ, 3 * x^2 + m * x + 25 = 0) ↔ n = 166)
:=
sorry

end rational_numbers_satisfying_conditions_l152_152597


namespace smallest_k_l152_152143

theorem smallest_k (k : ℕ) (h : 201 ≡ 9 [MOD 24]) : k = 1 := by
  sorry

end smallest_k_l152_152143


namespace general_term_formula_sum_first_n_terms_l152_152279

-- Step 1: Prove the general term formula for the sequence {a_n}
theorem general_term_formula (a : ℕ → ℝ) (S_5 : ℝ) (geo_seq : ∀ a₁ a₃ a₇ : ℝ, (a₃)² = a₁ * a₇) 
  (h1: S_5 = 20) (h2 : geo_seq (a 1) (a 3) (a 7)):
  (∀ n, a n = (n : ℕ) + 1) :=
  by
  sorry

-- Step 2: Prove the sum of the first n terms for the sequence {b_n}
theorem sum_first_n_terms (a b : ℕ → ℝ) (h_a : ∀ n, a n = (n : ℕ) + 1) 
  (h_b : ∀ n, b n = 1 / (a n * a (n + 1))) :
  (∀ n, ∑ i in finset.range n, b i = n / (2 * (n + 2))) :=
  by
  sorry

end general_term_formula_sum_first_n_terms_l152_152279


namespace solve_system_equation_152_l152_152277

theorem solve_system_equation_152 (x y z a b c : ℝ)
  (h1 : x * y - 2 * y - 3 * x = 0)
  (h2 : y * z - 3 * z - 5 * y = 0)
  (h3 : x * z - 5 * x - 2 * z = 0)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h5 : x = a)
  (h6 : y = b)
  (h7 : z = c) :
  a^2 + b^2 + c^2 = 152 := by
  sorry

end solve_system_equation_152_l152_152277


namespace log_base_5_of_one_over_125_l152_152582

theorem log_base_5_of_one_over_125 :
  ∀ (b x : ℝ), (b > 0) → (b ≠ 1) →
  log b (1 / (b^3)) = -3 :=
  by
    intros b x hb hbn1,
    have h125: (b^3 = 125) := sorry,
    have hfrac: (1 / 125 = b^(-3)) := sorry,
    linarith
   

end log_base_5_of_one_over_125_l152_152582


namespace arcsin_neg_one_l152_152962

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end arcsin_neg_one_l152_152962


namespace count_divisible_75_l152_152717

def digits : Set ℕ := {0, 2, 4, 5, 6, 7}

theorem count_divisible_75 :
  let f := [2, 0, 1, 6, 0, 7] -- fixed part of the number
  let last_two_digits := 75
  let possible_fillings := list.prod (list.replicate 4 [0, 2, 4, 5, 6, 7])
  let divisible_by_25 := last_two_digits = 75
  let sum_of_fixed_parts := 2 + 0 + 1 + 6 + 0 + 7
  let filled_numbers := sum_of_fixed_parts + sum (list.map (λ (t: list ℕ), 
      list.map (λ n, list.sum t) possible_fillings))
  let divisible_by_3 := ∀ t ∈ possible_fillings, 
      (sum_of_fixed_parts + list.sum t) % 3 = 0
in
  divisible_by_25 →
  divisible_by_3 →
  ((list.foldl (λ acc t, if (sum_of_fixed_parts + list.sum t) % 3 = 0 then acc + 1 else acc) 0 possible_fillings) = 432)
:=
by
  sorry

end count_divisible_75_l152_152717


namespace find_integer_m_l152_152585

-- Define the initial conditions
def condition (m : ℕ) : Prop :=
  log 2 (log 32 m) = log 8 (log 8 m)

-- The theorem we need to prove
theorem find_integer_m :
  ∃! m : ℕ, condition m ∧ (m.digits 10).sum = 2 :=
sorry

end find_integer_m_l152_152585


namespace infinite_primes_with_property_l152_152775

theorem infinite_primes_with_property :
  ∃ᶠ (p : ℕ) in Nat.primes, ∃ n > 0, (∃ m > 0, n = 12 * m + 1) ∧ ¬ n ∣ (p - 1) ∧ p ∣ Nat.factorial n + 1 :=
  sorry

end infinite_primes_with_property_l152_152775


namespace initial_percentage_correct_l152_152882

noncomputable def percentInitiallyFull (initialWater: ℕ) (waterAdded: ℕ) (fractionFull: ℚ) (capacity: ℕ) : ℚ :=
  (initialWater : ℚ) / (capacity : ℚ) * 100

theorem initial_percentage_correct (initialWater waterAdded capacity: ℕ) (fractionFull: ℚ) :
  waterAdded = 14 →
  fractionFull = 3/4 →
  capacity = 40 →
  initialWater + waterAdded = fractionFull * capacity →
  percentInitiallyFull initialWater waterAdded fractionFull capacity = 40 :=
by
  intros h1 h2 h3 h4
  unfold percentInitiallyFull
  sorry

end initial_percentage_correct_l152_152882


namespace individual_demand_45_aggregate_demand_45_l152_152421

def demand_function_general (n : ℕ) (P : ℝ) : ℝ :=
  500 + 10 * n - (5 + 0.1 * n) * P

def commander_demand (P : ℝ) : ℝ :=
  500 - 5 * P

theorem individual_demand_45 (P : ℝ) :
  demand_function_general 45 P = 950 - 9.5 * P :=
by
  sorry

theorem aggregate_demand_45 (P : ℝ) :
  let total_individual_demand := (∑ n in Finset.range 45, demand_function_general n P)
  total_individual_demand + commander_demand P = 33350 - 333.5 * P :=
by
  sorry

end individual_demand_45_aggregate_demand_45_l152_152421


namespace analytical_expression_fx_neg_inf_neg2_l152_152294

noncomputable def f : ℝ → ℝ 
| x => if x ∈ (0 : ℝ) +∞ then 1 / x else undefined

theorem analytical_expression_fx_neg_inf_neg2 :
  (∀ x y : ℝ, (f x) = y ↔ (f (-2 - x)) = -y) → (∀ x, x ∈ (-∞ : ℝ) -2 → f x = 1 / (x + 2)) :=
by
  sorry

end analytical_expression_fx_neg_inf_neg2_l152_152294


namespace star_polygon_points_eq_twenty_four_l152_152771

theorem star_polygon_points_eq_twenty_four
  (n : ℕ)
  (simple_closed_polygon : SimpleClosedPolygon)
  (cong_edges : ∀ i j : ℕ, 0 ≤ i ∧ i < 2 * n ∧ 0 ≤ j ∧ j < 2 * n → length (simple_closed_polygon.edge i) = length (simple_closed_polygon.edge j))
  (cong_angles_A : ∀ i j : ℕ, 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n → measure (simple_closed_polygon.angle_A i) = measure (simple_closed_polygon.angle_A j))
  (cong_angles_B : ∀ i j : ℕ, 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n → measure (simple_closed_polygon.angle_B i) = measure (simple_closed_polygon.angle_B j))
  (angle_difference : measure (simple_closed_polygon.angle_A 0) = measure (simple_closed_polygon.angle_B 0) - 15) :
  n = 24 :=
  sorry

end star_polygon_points_eq_twenty_four_l152_152771


namespace find_k_l152_152663

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l152_152663


namespace cos_triple_angle_identity_l152_152490

theorem cos_triple_angle_identity (α β γ : ℝ) 
  (h : cos α + cos β + cos γ = 0) : 
  cos (3 * α) + cos (3 * β) + cos (3 * γ) = 12 * cos α * cos β * cos γ :=
sorry

end cos_triple_angle_identity_l152_152490


namespace magnitude_of_projection_is_five_l152_152647

-- Definition for the initial point A
def A : ℝ × ℝ × ℝ := (3, 7, -4)

-- Definition for the projection of point A onto the xOz plane, resulting in point B
def B : ℝ × ℝ × ℝ := (A.1, 0, A.3)

-- Function to determine the magnitude of a point from the origin
def magnitude (P : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (P.1 * P.1 + P.2 * P.2 + P.3 * P.3)

-- Main theorem stating the problem
theorem magnitude_of_projection_is_five : magnitude B = 5 :=
by
  sorry

end magnitude_of_projection_is_five_l152_152647


namespace number_of_participants_l152_152911

theorem number_of_participants (n : ℕ) (hn : n = 862) 
    (h_lower : 575 ≤ n * 2 / 3) 
    (h_upper : n * 7 / 9 ≤ 670) : 
    ∃ p, (575 ≤ p) ∧ (p ≤ 670) ∧ (p % 11 = 0) ∧ ((p - 575) / 11 + 1 = 8) :=
by
  sorry

end number_of_participants_l152_152911


namespace color_copies_comparison_l152_152603

theorem color_copies_comparison (n : ℕ) (pX pY : ℝ) (charge_diff : ℝ) 
  (h₀ : pX = 1.20) (h₁ : pY = 1.70) (h₂ : charge_diff = 35) 
  (h₃ : pY * n = pX * n + charge_diff) : n = 70 :=
by
  -- proof steps would go here
  sorry

end color_copies_comparison_l152_152603


namespace mean_increase_by_30_l152_152003

variable (b : Fin 15 → ℕ)

def original_sum : ℕ := ∑ i, b i

def original_mean : ℕ := original_sum b / 15

def new_sum : ℕ := original_sum b + 450

def new_mean : ℕ := new_sum b / 15

theorem mean_increase_by_30 : new_mean b = original_mean b + 30 := 
by sorry

end mean_increase_by_30_l152_152003


namespace polynomial_form_l152_152587

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  ((n.to_digits 10).sum)

theorem polynomial_form (P : ℤ[X]) (hP : ∀ (x y : ℕ), sum_of_digits x = sum_of_digits y → 
  sum_of_digits (|P.eval x|) = sum_of_digits (|P.eval y|)) :
  ∃ k c : ℕ, (P = polynomial.C (-1) * (polynomial.X * 10^k + polynomial.C c)
  ∨ P = polynomial.X * 10^k + polynomial.C c) ∧ 0 ≤ c ∧ c < 10^k :=
  sorry

end polynomial_form_l152_152587


namespace profit_difference_l152_152478

-- Definitions for the initial conditions
variables (capitalA capitalB capitalC profitB totalProfit : ℝ)
variables (investmentRatioA investmentRatioB investmentRatioC : ℝ)

-- Assume the given conditions
axiom capitalA_eq : capitalA = 8000
axiom capitalB_eq : capitalB = 10000
axiom capitalC_eq : capitalC = 12000
axiom profitB_eq : profitB = 2500
axiom investmentRatioA_eq : investmentRatioA = 4
axiom investmentRatioB_eq : investmentRatioB = 5
axiom investmentRatioC_eq : investmentRatioC = 6

-- Definition of the total profit
def calc_total_profit (investmentRatioA investmentRatioB investmentRatioC : ℝ) (profitB : ℝ) : ℝ :=
  ((investmentRatioA + investmentRatioB + investmentRatioC) / investmentRatioB) * profitB

-- Proof statement
theorem profit_difference (capitalA capitalB capitalC profitB : ℝ) 
                          (investmentRatioA investmentRatioB investmentRatioC : ℝ) : 
                          (capitalA = 8000) →
                          (capitalB = 10000) →
                          (capitalC = 12000) →
                          (profitB = 2500) →
                          (investmentRatioA = 4) →
                          (investmentRatioB = 5) →
                          (investmentRatioC = 6) →
                          let totalProfit := calc_total_profit investmentRatioA investmentRatioB investmentRatioC profitB in
                          let profitA := (investmentRatioA / (investmentRatioA + investmentRatioB + investmentRatioC)) * totalProfit in
                          let profitC := (investmentRatioC / (investmentRatioA + investmentRatioB + investmentRatioC)) * totalProfit in
                          profitC - profitA = 1000 :=
sorry

end profit_difference_l152_152478


namespace bob_first_six_probability_l152_152209

noncomputable def probability_bob_first_six (p : ℚ) : ℚ :=
  (1 - p) * p / (1 - ( (1 - p) * (1 - p)))

theorem bob_first_six_probability :
  probability_bob_first_six (1/6) = 5/11 :=
by
  sorry

end bob_first_six_probability_l152_152209


namespace fA_odd_fA_increasing_only_fA_odd_and_increasing_l152_152469

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 2 * x
def fB (x : ℝ) : ℝ := x ^ 2
def fC (x : ℝ) : ℝ := Real.log x
def fD (x : ℝ) : ℝ := Real.exp x

-- Theorems for odd and increasing properties of fA
theorem fA_odd : ∀ x : ℝ, fA (-x) = -fA x :=
by
  intro x
  calc
    fA (-x) = 2 * (-x) := rfl
    ... = - (2 * x) := by ring
    ... = - fA x := rfl
sorry

theorem fA_increasing : ∀ x y : ℝ, x < y → fA x < fA y :=
by
  intros x y h
  calc
    fA x = 2 * x := rfl
    ... < 2 * y := by nlinarith
    ... = fA y := rfl
sorry

-- Main theorem
theorem only_fA_odd_and_increasing :
  ∀ f : ℝ → ℝ,
  (f = fA ∨ f = fB ∨ f = fC ∨ f = fD) →
  ((∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y)) ↔ (f = fA) :=
by
  intro f h
  split
  {
    intro hyp
    cases h
    case inl : h1
      rw h1
      exact ⟨fA_odd, fA_increasing⟩
    case inr : h2
      cases h2
      case inl : h2_1
        rw h2_1 at hyp
        have contra := hyp.1 1
        rw [fB, pow_two, neg_one_square] at contra
        contradiction
      case inr : h2_2
        cases h2_2
        case inl : h2_3
          rw h2_3 at hyp
          have contra := hyp.1 1
          rw [fC, Real.neg_log] at contra
          contradiction
        case inr : h2_4
          rw h2_4 at hyp
          have contra := hyp.1 1
          rw [fD, Real.neg_exp] at contra
          contradiction
  }
  {
    intro hyp
    rw hyp
    exact ⟨fA_odd, fA_increasing⟩
  }
sorry

end fA_odd_fA_increasing_only_fA_odd_and_increasing_l152_152469


namespace gcd_linear_combination_l152_152403

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := 
sorry

end gcd_linear_combination_l152_152403


namespace circle_equation_l152_152624

theorem circle_equation (x y : ℝ) : (x^2 = 16 * y) → (y = 4) → (x, -4) = (x, 4) → x^2 + (y-4)^2 = 64 :=
by
  sorry

end circle_equation_l152_152624


namespace proof_sufficient_but_not_necessary_l152_152694

open Real Set

def is_sufficient_not_necessary_condition (θ : ℝ) :=
  let A := {1, sin θ}
  let B := {1 / 2, 2}
  (A ∩ B = {1 / 2}) → (θ = 5 * π / 6 → A ∩ B = {1 / 2}) ∧ ¬ (θ = 5 * π / 6 → A ∩ B = {1 / 2})

theorem proof_sufficient_but_not_necessary :
  is_sufficient_not_necessary_condition (5 * π / 6) :=
sorry

end proof_sufficient_but_not_necessary_l152_152694


namespace olympic_savings_l152_152197

variables (a p : ℝ) (h1 : p > 0) (h2 : a > 0)

def total_withdrawn_amount := a / p * ((1 + p) ^ 8 - (1 + p))

theorem olympic_savings : 
  let yearly_deposit := λ (n : ℕ), a * ((1 + p) ^ n) in
  (finset.range 8).sum (λ i, yearly_deposit i) = total_withdrawn_amount :=
by
  sorry

end olympic_savings_l152_152197


namespace triangle_ABC_angle_60_l152_152031

noncomputable def triangle_config (A B C E F I : Type*)
  (AB AC BC C1 E1 F1 I1 : ℝ) 
  (AB_AC_neq : AB ≠ AC) 
  (angle_bisector_B : E1 = AC / AB) 
  (angle_bisector_C : F1 = AB / AC) 
  (bisectors_meet_I : E1 = F1 = I1) 
  (EI_eq_FI : E1 = F1) : Prop :=
  ∀ (BAC : ℝ), BAC = 60

theorem triangle_ABC_angle_60 
  {A B C E F I : Type*} 
  {AB AC BC C1 E1 F1 I1 : ℝ} 
  (h1 : AB ≠ AC) 
  (h2 : E1 = AC / AB) 
  (h3 : F1 = AB / AC) 
  (h4 : E1 = F1 = I1) 
  (h5 : E1 = F1) : triangle_config A B C E F I AB AC BC C1 E1 F1 I1 h1 h2 h3 h4 h5 :=
sorry

end triangle_ABC_angle_60_l152_152031


namespace min_guests_l152_152216

theorem min_guests (total_food : ℝ) (max_food_per_guest : ℝ) (h_total_food : total_food = 520) (h_max_food_per_guest : max_food_per_guest = 1.5) : 
  ∃ n : ℤ, n ≥ 347 :=
by
  have h : total_food / max_food_per_guest = 346.67,
    rw [← h_total_food, ← h_max_food_per_guest],
    norm_num,
  sorry

end min_guests_l152_152216


namespace Sara_spent_on_salad_l152_152087

theorem Sara_spent_on_salad (cost_of_hotdog : ℝ) (total_bill : ℝ) (cost_of_salad : ℝ) 
  (h_hotdog : cost_of_hotdog = 5.36) (h_total : total_bill = 10.46) : cost_of_salad = 10.46 - 5.36 :=
by
  rw [h_hotdog, h_total]
  rfl

end Sara_spent_on_salad_l152_152087


namespace problem1_problem2_problem3_problem4_l152_152965

/-- Proof Problems -/

theorem problem1 : (1 : ℝ) * (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 := 
by 
  sorry

theorem problem2 : 2 * (Real.sqrt (9 / 2) - (Real.sqrt 8) / 3) * 2 * Real.sqrt 2 = 10 / 3 := 
by 
  sorry

theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = 5 * Real.sqrt 2 / 4 := 
by 
  sorry

theorem problem4 : (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 := 
by 
  sorry

end problem1_problem2_problem3_problem4_l152_152965


namespace part_a_part_b_part_c_l152_152730

def transformation (x : ℝ) (hx : x ≠ 1) : ℝ := 1 / (1 - x)

theorem part_a (hx : 2 ≠ 1) : (transformation (transformation (transformation 2 hx) (by norm_num)) (by norm_num)) = 2 :=
sorry

theorem part_b (hx : 2 ≠ 1) : transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation 2 hx) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) = -1 :=
sorry

theorem part_c (hx : 2 ≠ 1) : 
  transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation)(
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (
    transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation (transformation 2 hx) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num))
  ) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) (by norm_num)) = 1 / 2 :=
sorry

end part_a_part_b_part_c_l152_152730


namespace circle_tangent_eq_center_l152_152426

theorem circle_tangent_eq_center :
  ∃ (r : ℝ), (∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) → (x - 2)^2 + (y + 1)^2 = r^2) ∧ r = 3 :=
begin
  use 3,
  split,
  { intros x y h,
    have h2 : (9 + 16) = 25, by norm_num,
    have h3 : 3 * 2 - 4 * (-1) + 5 = 15, by norm_num,
    have h4 : ¬ (9 + 16 = 0), by norm_num,
    have d : (3 * 2 - 4 * (-1) + 5) / real.sqrt (9 + 16) = 3, from
      by
        rw [abs_of_nonneg (by linarith), real.rpow_nat_cast _ 2, real.sqrt_mul_self (le_of_lt (by norm_num))],
        norm_num,
        simp [h2],
    rw [d],
    norm_num,
    sorry
  },
  { refl }
end

end circle_tangent_eq_center_l152_152426


namespace trace_length_of_Q_l152_152075

noncomputable def total_trace_length (d : ℝ) (n : ℕ) : ℝ :=
  (n * (n + 1) / 2) * d

theorem trace_length_of_Q :
  ∀ (d : ℝ) (n : ℕ), n = 5 → d = 5 → total_trace_length d n = 75 :=
by
  intros d n hn hd
  rw [hn, hd]
  rfl

end trace_length_of_Q_l152_152075


namespace simplify_expression_l152_152096

theorem simplify_expression :
  (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end simplify_expression_l152_152096


namespace equilateral_triangle_chords_middle_chord_sum_l152_152610

-- Define a circle and a point P on the circle with three chords PA, PB, PC
variables (circle : Type) [MetricSpace circle]
variable (P : circle)
variables (A B C : circle)

-- Define the conditions
variable (between_p : Circle)
variable (angle_PA_PB angle_PC_PB : ℝ)
variable (chords : P ≠ A ∧ P ≠ B ∧ P ≠ C)

-- Define the angles to be 60 degrees
def angle_PA_PB_60 : Prop := angle_PA_PB = 60
def angle_PC_PB_60 : Prop := angle_PC_PB = 60

-- Statement for part (a)
theorem equilateral_triangle_chords 
  (h_a : angle_PA_PB_60)
  (h_b : angle_PC_PB_60)
  (h_c : chords) :
  ∠ A B C = 60 ∧ ∠ B C A = 60 ∧ ∠ C A B = 60 :=
sorry

-- Length of the middle chord is the sum of lengths of the two outer chords
variables (length_PA length_PB length_PC : ℝ)

-- Statement for part (b)
theorem middle_chord_sum :
  length_PB = length_PA + length_PC :=
sorry

end equilateral_triangle_chords_middle_chord_sum_l152_152610


namespace sum_of_squares_l152_152141

theorem sum_of_squares (n : ℕ) (h : n ≥ 1) : 
  ∑ i in Finset.range (n + 1), (1 / ((i + 2) * (i + 2) : ℝ)) > 1/2 - 1/(n + 2) := 
by
  sorry

end sum_of_squares_l152_152141


namespace number_of_lightsabers_ordered_l152_152159

def cost_per_metal_arc : Nat := 400
def metal_arcs_per_lightsaber : Nat := 2
def assembly_time_per_lightsaber : Nat := 1 / 20
def combined_cost_per_hour : Nat := 200 + 100
def total_cost : Nat := 65200

theorem number_of_lightsabers_ordered (x : Nat) :
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  x * total_cost_per_lightsaber = total_cost →
  x = 80 :=
by
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  assume h : x * total_cost_per_lightsaber = total_cost
  sorry

end number_of_lightsabers_ordered_l152_152159


namespace iron_ii_sulfate_moles_l152_152231

/-- Given the balanced chemical equation for the reaction between iron (Fe) and sulfuric acid (H2SO4)
    to form Iron (II) sulfate (FeSO4) and hydrogen gas (H2) and the 1:1 molar ratio between iron and
    sulfuric acid, determine the number of moles of Iron (II) sulfate formed when 3 moles of Iron and
    2 moles of Sulfuric acid are combined. This is a limiting reactant problem with the final 
    product being 2 moles of Iron (II) sulfate (FeSO4). -/
theorem iron_ii_sulfate_moles (Fe moles_H2SO4 : Nat) (reaction_ratio : Nat) (FeSO4 moles_formed : Nat) :
  Fe = 3 → moles_H2SO4 = 2 → reaction_ratio = 1 → moles_formed = 2 :=
by
  intros hFe hH2SO4 hRatio
  apply sorry

end iron_ii_sulfate_moles_l152_152231


namespace no_graph_for_equation_l152_152115

theorem no_graph_for_equation (x y : ℝ) : 
  ¬ ∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := 
by 
  sorry

end no_graph_for_equation_l152_152115


namespace part_a_part_b_l152_152968

section PartA

variable (a b : ℕ → ℝ)
variable (A B C : ℕ → ℝ)

noncomputable def is_bounded (f : ℕ → ℝ) : Prop := ∃ M : ℝ, ∀ n : ℕ, f n ≤ M
noncomputable def is_unbounded (f : ℕ → ℝ) : Prop := ∀ M : ℝ, ∃ n : ℕ, f n > M

noncomputable def An (n : ℕ) : ℝ := ∑ i in Finset.range n, a i
noncomputable def Bn (n : ℕ) : ℝ := ∑ i in Finset.range n, b i
noncomputable def min_seq (a b : ℕ → ℝ) (i : ℕ) : ℝ := min (a i) (b i)
noncomputable def Cn (n : ℕ) : ℝ := ∑ i in Finset.range n, min_seq a b i

theorem part_a:
  ∃ a b : ℕ → ℝ, is_bounded Cn ∧ is_unbounded An ∧ is_unbounded Bn :=
sorry

end PartA

section PartB

variable (a : ℕ → ℝ)
variable (An Cn : ℕ → ℝ)

noncomputable def bn (i : ℕ) : ℝ := 1 / (i+1)
noncomputable def Bn (n : ℕ) : ℝ := ∑ i in Finset.range n, bn i

theorem part_b:
  ¬ ∃ a : ℕ → ℝ, is_bounded Cn ∧ is_unbounded An ∧ is_unbounded Bn :=
sorry

end PartB

end part_a_part_b_l152_152968


namespace range_of_a_l152_152300

variable (a : ℝ)

def f (x : ℝ) := log a (8 - 3 * a * x)

theorem range_of_a :
    (∀ x1 x2 : ℝ, -1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → f a x1 ≥ f a x2) ↔ (1 < a ∧ a < 4 / 3) := by
  sorry

end range_of_a_l152_152300


namespace probability_of_event_l152_152749

noncomputable def fifth_roots_of_unity : Finset ℂ :=
  {exp(2 * Real.pi * I * k / 5) | k ∈ {0, 1, 2, 3, 4}}

def satisfies_condition (v w : ℂ) : Prop :=
  (v ∈ fifth_roots_of_unity) ∧ (w ∈ fifth_roots_of_unity) ∧ (v ≠ w) ∧ (Real.sqrt (3 + Real.sqrt 5) ≤ Complex.abs (v + w))

theorem probability_of_event :
  ∃ n m : ℕ, m ≠ 0 ∧ (n : ℝ) / (m : ℝ) = 3 / 4 :=
sorry

end probability_of_event_l152_152749


namespace mom_tshirts_count_l152_152473

def packages : ℕ := 71
def tshirts_per_package : ℕ := 6

theorem mom_tshirts_count : packages * tshirts_per_package = 426 := by
  sorry

end mom_tshirts_count_l152_152473


namespace equation_of_l2_area_of_triangle_l152_152646

-- Problem I
theorem equation_of_l2
  (f : ℝ → ℝ) (l1 l2 : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + x - 2) 
  (hl1 : ∀ x, l1 x = 3*x - 3) 
  (l2_tangent : ∀ x, ∃ y, y = f x → l2(y)) 
  (perpendicular : ∀ x, l1 x * l2 x = -1) :
  ∀ x, l2 x = - (1/3)*x - 2/3 :=
sorry

-- Problem II
theorem area_of_triangle
  (l1 l2 : ℝ → ℝ)
  (hl1 : ∀ x, l1 x = 3*x - 3)
  (hl2 : ∀ x, l2 x = - (1/3)*x - 2/3)
  (a1 : ∃ x, l1 x = 0 → x = 1)
  (a2 : ∃ x, l2 x = 0 → x = -2) :
  ∀ A : ℝ, A = (1/2) * 3 * 2 :=
sorry

end equation_of_l2_area_of_triangle_l152_152646


namespace tree_leaves_l152_152917

theorem tree_leaves (initial_leaves : ℕ) (first_week_fraction : ℚ) (second_week_percentage : ℚ) (third_week_fraction : ℚ) :
  initial_leaves = 1000 →
  first_week_fraction = 2 / 5 →
  second_week_percentage = 40 / 100 →
  third_week_fraction = 3 / 4 →
  let leaves_after_first_week := initial_leaves - (first_week_fraction * initial_leaves).toNat,
      leaves_after_second_week := leaves_after_first_week - (second_week_percentage * leaves_after_first_week).toNat,
      leaves_after_third_week := leaves_after_second_week - (third_week_fraction * leaves_after_second_week).toNat
  in leaves_after_third_week = 90 :=
begin
  intros h1 h2 h3 h4,
  unfold leaves_after_first_week leaves_after_second_week leaves_after_third_week,
  rw [h1, h2, h3, h4],
  norm_num,
end

end tree_leaves_l152_152917


namespace a_sequence_l152_152130

-- Define the sequence a_n
def a : ℕ → ℝ
| 1 := 1
| 2 := 2 / 3
| (n + 1) := sorry -- This needs to be defined based on recursion, so we'll assume it's defined later

-- Prove the given conditions
theorem a_sequence :
  (∀ n ≥ 2, 1 / a (n - 1) + 1 / a (n + 1) = 2 / a n) →
  (∀ n, a n = 2 / (n + 1)) :=
begin
  sorry -- Skip the proof for now
end

end a_sequence_l152_152130


namespace least_integer_condition_l152_152147

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l152_152147


namespace probability_one_multiple_of_4_in_two_picks_l152_152950

theorem probability_one_multiple_of_4_in_two_picks (total_numbers : ℕ) (multiple : ℕ) (picks : ℕ) : 
  total_numbers = 100 → multiple = 4 → picks = 2 → 
  let multiples := total_numbers / multiple in
  let probability_not_multiple := (total_numbers - multiples : ℝ) / total_numbers in
  let probability_neither_multiple := probability_not_multiple ^ picks in
  let probability_at_least_one := 1 - probability_neither_multiple in
  probability_at_least_one = 7 / 16 :=
by 
  intros h_total h_multiple h_picks
  let multiples := total_numbers / multiple
  let probability_not_multiple := (total_numbers - multiples : ℝ) / total_numbers
  let probability_neither_multiple := probability_not_multiple ^ picks
  let probability_at_least_one := 1 - probability_neither_multiple
  sorry

end probability_one_multiple_of_4_in_two_picks_l152_152950


namespace jerrys_breakfast_calories_l152_152045

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l152_152045


namespace improper_fraction_eq_poly_and_proper_improper_fraction_eq_poly_and_proper_2_fraction_is_integer_l152_152701

-- Definitions
def proper_fraction (n d : ℕ) := n < d
def improper_fraction (n d : ℕ) := ¬ proper_fraction n d
def sum_polynomial_proper (f : ℕ → ℕ → Prop) (n d : ℕ) : Prop := ∃ (p : ℕ) (q : ℕ), f (p * d + q) d

-- Problem 1
theorem improper_fraction_eq_poly_and_proper (x : ℕ) :
  (improper_fraction (x + 1) (x - 1)) → 
  ( ∃ (p : ℕ) (q : ℕ), (x + 1) = p * (x - 1) + q)  →
  1 + (2/(x - 1)) = ((x + 1)/(x - 1)) := 
sorry

-- Problem 2
theorem improper_fraction_eq_poly_and_proper_2 (x : ℕ) :
  (improper_fraction (2 * x - 1) (x + 1)) →
  ( ∃ (p : ℕ) (q : ℕ), (2 * x - 1) = p * (x + 1) + q) →

  2 - (3/(x + 1)) = (2 * x - 1)/(x + 1) := 
sorry

-- Problem 3
theorem fraction_is_integer (x : ℕ) :
  (∃ (q : ℕ), x^2 = q * (x + 1)) →
  (x = -2 ∨ x = 0) := 
sorry

end improper_fraction_eq_poly_and_proper_improper_fraction_eq_poly_and_proper_2_fraction_is_integer_l152_152701


namespace max_odd_numbers_l152_152924

theorem max_odd_numbers (numbers : Fin 7 → ℕ) (h : (∏ i, numbers i) % 2 = 0) : 
  ∃ evens : Finset (Fin 7), evens.card = 1 ∧ (∀ i ∈ evens, numbers i % 2 = 0) ∧ 
  evens.card = 7 - 6 ∧ (∀ i ∉ evens, numbers i % 2 ≠ 0) :=
by
  sorry

end max_odd_numbers_l152_152924


namespace magic_show_l152_152514

theorem magic_show (performances : ℕ) (prob_never_reappear : ℚ) (prob_two_reappear : ℚ)
  (h_performances : performances = 100)
  (h_prob_never_reappear : prob_never_reappear = 1 / 10)
  (h_prob_two_reappear : prob_two_reappear = 1 / 5) :
  let never_reappear := prob_never_reappear * performances,
      two_reappear := prob_two_reappear * performances,
      normal_reappear := performances,
      extra_reappear := two_reappear,
      total_reappear := normal_reappear + extra_reappear - never_reappear in
  total_reappear = 110 := by
  sorry

end magic_show_l152_152514


namespace bought_carpet_time_l152_152981

def price_at (t : ℕ) : ℝ :=
  10.0 * (0.9 ^ t)

theorem bought_carpet_time :
  ∃ (t : ℕ), t * 15 = 45 ∧ price_at t < 8.0 :=
by
  have h : price_at 3 = 7.29 := by simp [price_at, pow_succ]
  have h2 : price_at 3 < 8.0 := by norm_num [h]
  exact ⟨3, by norm_num, h2⟩

end bought_carpet_time_l152_152981


namespace part_a_part_b1_part_b2_l152_152408

noncomputable def potential_function_a (x y : ℝ) (C : ℝ) : Prop :=
  (x * deriv y - y * deriv x) / (x^2 + y^2) = deriv (θ : ℝ) (Arctan (y / x) + C)

noncomputable def potential_function_b1 (x y : ℝ) (C1 : ℝ) : Prop :=
  ∃ u1 : ℝ, 
    (x + 2 * y) * deriv x + y * deriv y / (x + y)^2 = deriv (θ : ℝ) (ln |x + y| + x / (x + y) + C1) ∧ x + y < 0

noncomputable def potential_function_b2 (x y : ℝ) (C2 : ℝ) : Prop :=
  ∃ u2 : ℝ, 
    (x + 2 * y) * deriv x + y * deriv y / (x + y)^2 = deriv (θ : ℝ) (ln |x + y| + x / (x + y) + C2) ∧ x + y > 0

theorem part_a (x y C : ℝ) : potential_function_a x y C :=
  sorry

theorem part_b1 (x y C1 : ℝ) : potential_function_b1 x y C1 :=
  sorry

theorem part_b2 (x y C2 : ℝ) : potential_function_b2 x y C2 :=
  sorry

end part_a_part_b1_part_b2_l152_152408


namespace sarah_average_speed_l152_152778

theorem sarah_average_speed :
  ∀ (total_distance race_time : ℕ) 
    (sadie_speed sadie_time ariana_speed ariana_time : ℕ)
    (distance_sarah speed_sarah time_sarah : ℚ),
  sadie_speed = 3 → 
  sadie_time = 2 → 
  ariana_speed = 6 → 
  ariana_time = 1 / 2 → 
  race_time = 9 / 2 → 
  total_distance = 17 →
  distance_sarah = total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time) →
  time_sarah = race_time - (sadie_time + ariana_time) →
  speed_sarah = distance_sarah / time_sarah →
  speed_sarah = 4 :=
by
  intros total_distance race_time sadie_speed sadie_time ariana_speed ariana_time distance_sarah speed_sarah time_sarah
  intros sadie_speed_eq sadie_time_eq ariana_speed_eq ariana_time_eq race_time_eq total_distance_eq distance_sarah_eq time_sarah_eq speed_sarah_eq
  sorry

end sarah_average_speed_l152_152778


namespace max_possible_value_l152_152298

theorem max_possible_value :
  ∃ (a b c d : ℕ), a ∈ {0, 1, 2, 3} ∧ b ∈ {0, 1, 2, 3} ∧ c ∈ {0, 1, 2, 3} ∧ d ∈ {0, 1, 2, 3} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  d ≠ 0 ∧ (c * a ^ b - d = 2) := sorry

end max_possible_value_l152_152298


namespace angle_BAD_deg_l152_152456

theorem angle_BAD_deg (A B C D : Type*) [incidence_geometry A B C D]
  (isosceles_ABC : isosceles_triangle A B C)
  (isosceles_DBC : isosceles_triangle D B C)
  (D_on_AC : point_on_line D A C)
  (angle_ABC_eq_30 : angle_deg A B C = 30)
  (angle_DBC_eq_110 : angle_deg D B C = 110) :
  angle_deg B A D = 40 :=
sorry

end angle_BAD_deg_l152_152456


namespace magician_performances_l152_152517

theorem magician_performances (performances : ℕ) (p_no_reappear : ℚ) (p_two_reappear : ℚ) :
  performances = 100 → p_no_reappear = 1/10 → p_two_reappear = 1/5 → 
  let num_no_reappear := performances * p_no_reappear in
  let num_two_reappear := performances * p_two_reappear in
  let num_one_reappear := performances - num_no_reappear - num_two_reappear in
  let total_reappeared := num_one_reappear + 2 * num_two_reappear in
  total_reappeared = 110 :=
by
  intros h1 h2 h3
  let num_no_reappear := performances * p_no_reappear
  let num_two_reappear := performances * p_two_reappear
  let num_one_reappear := performances - num_no_reappear - num_two_reappear
  let total_reappeared := num_one_reappear + 2 * num_two_reappear
  have h4 : num_no_reappear = 10 := by sorry
  have h5 : num_two_reappear = 20 := by sorry
  have h6 : num_one_reappear = 70 := by sorry
  have h7 : total_reappeared = 110 := by sorry
  exact h7

end magician_performances_l152_152517


namespace divisibility_rule_37_l152_152555

-- Define the conditions
def is_divisible_by_37 (n : Nat) : Prop := n % 37 = 0

noncomputable def check_divisibility_condition (digits : List Nat) : Nat :=
  digits.enum.foldl (λ acc ⟨i, d⟩, acc + d * if i % 3 = 0 then 1 else if i % 3 = 1 then 10 else -11) 0

theorem divisibility_rule_37 (n : Nat) (digits : List Nat)
  (h : ∃ (digits : List Nat), n = digits.foldl (λ acc d → acc * 10 + d) 0 ∧ digits.length % 3 = 0) :
  is_divisible_by_37 n ↔ is_divisible_by_37 (check_divisibility_condition digits) :=
sorry

end divisibility_rule_37_l152_152555


namespace foreign_student_count_l152_152550

/--
There are 2500 students at the university.
35% of all the students are from other countries.
300 new foreign students will begin studying at the university next semester.
50 foreign students are graduating or transferring to a different university at the end of the current semester.
Prove that the total number of foreign students after these changes will be 1125.
-/
theorem foreign_student_count (total_students : ℕ)
  (percent_foreign : ℝ)
  (new_foreign : ℕ)
  (leaving_foreign : ℕ) (initial_foreign : ℕ) (final_foreign : ℕ) :
  total_students = 2500 →
  percent_foreign = 0.35 →
  new_foreign = 300 →
  leaving_foreign = 50 →
  initial_foreign = 0.35 * 2500 →
  final_foreign = initial_foreign + new_foreign - leaving_foreign →
  final_foreign = 1125 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end foreign_student_count_l152_152550


namespace triangle_IJS_area_l152_152716

noncomputable def AB_length : ℝ := 1
noncomputable def IJ_side_length : ℝ := 2

def area_of_triangle (b h : ℝ) : ℝ := (1 / 2) * b * h

theorem triangle_IJS_area :
  let IJ := IJ_side_length,
      IA := 3,
      area := area_of_triangle IJ IA
  in area = 3 := 
by 
  sorry

end triangle_IJS_area_l152_152716


namespace center_of_circle_polar_coords_l152_152246

-- Given condition: The equation of the circle in polar form
def circle_equation (θ : ℝ) : ℝ := sqrt 2 * (cos θ + sin θ)

-- The goal is to prove the circle's center in polar coordinates is (1, π/4)
theorem center_of_circle_polar_coords :
  ∃ θ ρ,  (ρ = 1) ∧ (θ = π / 4) ∧ (circle_equation θ = ρ) :=
sorry

end center_of_circle_polar_coords_l152_152246


namespace trapezium_other_side_l152_152239

theorem trapezium_other_side (x : ℝ) :
  1/2 * (20 + x) * 10 = 150 → x = 10 :=
by
  sorry

end trapezium_other_side_l152_152239


namespace cyclic_quadrilateral_area_l152_152272

-- Define the conditions for the cyclic quadrilateral
variables (A B C D : Point) (cyclic : CyclicQuadrilateral A B C D)
variables (AB BC CD DA : ℝ)
variable h1 : AB = 1
variable h2 : BC = 3
variable h3 : CD = 2
variable h4 : DA = 2

-- State the theorem
theorem cyclic_quadrilateral_area (cyclic : CyclicQuadrilateral A B C D)
  (h1 : AB = 1) (h2 : BC = 3) (h3 : CD = 2) (h4 : DA = 2) :
  area_of_quadrilateral A B C D = 2 * real.sqrt 3 :=
sorry

end cyclic_quadrilateral_area_l152_152272


namespace min_value_of_expression_l152_152750

theorem min_value_of_expression (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 10) : 
  ∃ B, B = x^2 + y^2 + z^2 + x^2 * y ∧ B ≥ 4 :=
by
  sorry

end min_value_of_expression_l152_152750


namespace herd_compuation_l152_152203

theorem herd_compuation (a b c : ℕ) (total_animals total_payment : ℕ) 
  (H1 : total_animals = a + b + 10 * c) 
  (H2 : total_payment = 20 * a + 10 * b + 10 * c) 
  (H3 : total_animals = 100) 
  (H4 : total_payment = 200) :
  a = 1 ∧ b = 9 ∧ 10 * c = 90 :=
by
  sorry

end herd_compuation_l152_152203


namespace tangent_line_at_1_is_3x_minus_y_minus_2_eq_0_l152_152427

noncomputable def f (x : ℝ) : ℝ := log x + 2 * x - 1

theorem tangent_line_at_1_is_3x_minus_y_minus_2_eq_0 :
    let x₁ := 1 in
    let y₁ := f x₁ in
    let m := (derivative f 1) in
    (∀ x y : ℝ, y - y₁ = m * (x - x₁) → 3 * x - y - 2 = 0) :=
by
  sorry

end tangent_line_at_1_is_3x_minus_y_minus_2_eq_0_l152_152427


namespace ratio_of_doctors_to_lawyers_l152_152793

/--
Given the average age of a group consisting of doctors and lawyers is 47,
the average age of doctors is 45,
and the average age of lawyers is 55,
prove that the ratio of the number of doctors to the number of lawyers is 4:1.
-/
theorem ratio_of_doctors_to_lawyers
  (d l : ℕ) -- numbers of doctors and lawyers
  (avg_group_age : ℝ := 47)
  (avg_doctors_age : ℝ := 45)
  (avg_lawyers_age : ℝ := 55)
  (h : (45 * d + 55 * l) / (d + l) = 47) :
  d = 4 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l152_152793


namespace abc_equal_l152_152605

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l152_152605


namespace max_min_diff_b_l152_152379

theorem max_min_diff_b {a b c : ℝ} (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 18) :
  (by exact abs (sqrt 11.25 * 2) = 6.67) :=
  sorry

end max_min_diff_b_l152_152379


namespace f5_plus_f_prime5_eq_2_l152_152650

noncomputable def f : ℝ → ℝ := sorry

-- Given conditions as Lean definitions
def f_tangent_at (x : ℝ) : Prop := (∀ x, f(x) = -x + 8)
def f_prime_5 : Prop := (derivative f 5 = -1)
def f_value_5 : Prop := (f 5 = 3)

-- Proof statement
theorem f5_plus_f_prime5_eq_2 (h_tangent : f_tangent_at 5) (h_prime : f_prime_5) (h_value : f_value_5) : f 5 + derivative f 5 = 2 :=
by sorry

end f5_plus_f_prime5_eq_2_l152_152650


namespace sin_sq_gt_cos_sq_l152_152828

theorem sin_sq_gt_cos_sq (x : ℝ) (h : 0 < x ∧ x < π) : 
    (sin x)^2 > (cos x)^2 ↔ (π / 4 < x ∧ x < 3 * π / 4) :=
by sorry

end sin_sq_gt_cos_sq_l152_152828


namespace abs_diff_101st_term_l152_152848

theorem abs_diff_101st_term 
  (C D : ℕ → ℤ)
  (hC_start : C 0 = 20)
  (hD_start : D 0 = 20)
  (hC_diff : ∀ n, C (n + 1) = C n + 12)
  (hD_diff : ∀ n, D (n + 1) = D n - 6) :
  |C 100 - D 100| = 1800 :=
by
  sorry

end abs_diff_101st_term_l152_152848


namespace probability_Rachel_Robert_in_picture_l152_152776

noncomputable def Rachel_lap_time := 75
noncomputable def Robert_lap_time := 70
noncomputable def photo_time_start := 900
noncomputable def photo_time_end := 960
noncomputable def track_fraction := 1 / 5

theorem probability_Rachel_Robert_in_picture :
  let lap_time_Rachel := Rachel_lap_time
  let lap_time_Robert := Robert_lap_time
  let time_start := photo_time_start
  let time_end := photo_time_end
  let interval_Rachel := 15  -- ±15 seconds for Rachel
  let interval_Robert := 14  -- ±14 seconds for Robert
  let probability := (2 * interval_Robert) / (time_end - time_start) 
  probability = 7 / 15 :=
by
  sorry

end probability_Rachel_Robert_in_picture_l152_152776


namespace find_M_l152_152942

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l152_152942


namespace urn_problem_l152_152940

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l152_152940


namespace circumscribed_circle_radius_l152_152814

theorem circumscribed_circle_radius (r : ℝ) (π : ℝ)
  (isosceles_right_triangle : Type) 
  (perimeter : isosceles_right_triangle → ℝ )
  (area : ℝ → ℝ)
  (h : ∀ (t : isosceles_right_triangle), perimeter t = area r) :
  r = (1 + Real.sqrt 2) / π :=
sorry

end circumscribed_circle_radius_l152_152814


namespace solve_for_y_l152_152785

theorem solve_for_y :
  ∀ (t y : ℝ), (1 = 3 - 2 * t) ∧ (y = 5 * t + 3) → y = 8 :=
by
  intros t y h
  cases h with h1 h2
  have ht : t = 1 := by linarith
  rw [ht] at h2
  linarith

end solve_for_y_l152_152785


namespace part1_part2_l152_152666

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l152_152666


namespace find_second_month_sale_l152_152894

/-- Given sales for specific months and required sales goal -/
def sales_1 := 4000
def sales_3 := 5689
def sales_4 := 7230
def sales_5 := 6000
def sales_6 := 12557
def avg_goal := 7000
def months := 6

theorem find_second_month_sale (x2 : ℕ) :
  (sales_1 + x2 + sales_3 + sales_4 + sales_5 + sales_6) / months = avg_goal →
  x2 = 6524 :=
by
  sorry

end find_second_month_sale_l152_152894


namespace largest_n_value_l152_152856

-- Define the primary expression
def expr (n : ℕ) : ℤ := 7 * (n - 3)^4 - n^2 + 12 * n - 30

-- Main theorem statement
theorem largest_n_value : ∃ n < 100000, expr n % 4 = 0 ∧ n % 4 = 3 ∧ n = 99999 :=
by {
  let n := 99999,
  have h1: n < 100000 := by norm_num,
  have h2: expr n % 4 = 0 := sorry,
  have h3: n % 4 = 3 := by norm_num,
  exact ⟨n, h1, h2, h3, rfl⟩,
}

end largest_n_value_l152_152856


namespace jean_spots_on_sides_l152_152041

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l152_152041


namespace tan_sum_value_l152_152020

variables {a b c : ℝ} {A B C : ℝ}

noncomputable def tan (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  A + B + C = π

def sides_opposite_to_angles (a b c A B C : ℝ) : Prop :=
  a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) ∧
  b = Real.sqrt (a^2 + c^2 - 2*a*c*Real.cos B) ∧
  c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C)

def condition (a b : ℝ) (C : ℝ) : Prop :=
  b / a + a / b = 6 * Real.cos C

theorem tan_sum_value
  (h₀ : acute_triangle A B C)
  (h₁ : sides_opposite_to_angles a b c A B C)
  (h₂ : condition a b C) :
  tan C / tan A + tan C / tan B = 4 :=
sorry

end tan_sum_value_l152_152020


namespace triangle_perimeter_l152_152956

-- Declare the points A, B, and C
def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (6, -2)
def C : ℝ × ℝ := (-2, 5)

-- Distance formula between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distances between each pair of points
def AB : ℝ := dist A B
def BC : ℝ := dist B C
def CA : ℝ := dist C A

-- State the theorem for the perimeter of the triangle
theorem triangle_perimeter : AB + BC + CA = 3 * Real.sqrt 10 + Real.sqrt 113 + Real.sqrt 17 := by
  sorry

end triangle_perimeter_l152_152956


namespace inequality_solution_l152_152829

theorem inequality_solution (x : ℝ) : x^2 - 2 * x - 5 > 2 * x ↔ x > 5 ∨ x < -1 :=
by
  sorry

end inequality_solution_l152_152829


namespace theorem_I_theorem_II_theorem_III_theorem_IV_l152_152383

-- Definitions for the geometric constructs
variables {A1 A2 A3 A4 A'_1 A'_2 A'_3 A'_4 : Type}
variables {R r : ℝ}
variables {h1 h2 h3 h4 : ℝ}
variables (d1 d2 d3 d4 : ℝ)

-- Given conditions
def circumradius_of_tetrahedron := ∀ (A1 A2 A3 A4 : Type), ℝ
def inradius_of_tetrahedron := ∀ (A1 A2 A3 A4 : Type), ℝ 
def height_of_vertex_to_face := ∀ (A : Type), (A → A → A → A → ℝ)
def insphere_touching_point := ∀ (A : Type), (A → A → A → A → A)

variable tetrahedron : Type
variable (vertices : tetrahedron → list Type)

-- Questions to be addressed/conclusions to be proven
theorem theorem_I :
  (\sum_{_ \k<j_ in (list.pairwise_cons (vertices tetrahedron))} _dist_kj _) 
  <= 16 * circumradius_of_tetrahedron A1 A2 A3 A4 := 
  sorry

theorem theorem_II :
  (\sum_{_ \k<j_ in (list.pairwise_cons (vertices tetrahedron))} _dist_kj _)
  >= (9 / 4) * \sum_{i in (vertices tetrahedron)} height_of_vertex_to_face i :=
  sorry

theorem theorem_III :
  (\sum_{i in (vertices tetrahedron)} (height_of_vertex_to_face i)) 
  >= 64 * inradius_of_tetrahedron A1 A2 A3 A4 :=
  sorry

theorem theorem_IV :
  (\sum_{_ \k<j in (list.pairwise_cons (vertices tetrahedron))} _dist_kj _) 
  >= 9 * (\sum_{_ \k<j in list.pairwise_cons (insphere_touching_point vertices))} _dist'_kj _) :=
  sorry

end theorem_I_theorem_II_theorem_III_theorem_IV_l152_152383


namespace blue_tiles_count_l152_152907

theorem blue_tiles_count (red_tiles : ℕ := 32) (tiles_needed : ℕ := 20) (total_tiles : ℕ := 100) : 
  ∃ (blue_tiles : ℕ), blue_tiles = 48 :=
by
  let current_tiles := total_tiles - tiles_needed
  let blue_tiles := current_tiles - red_tiles
  use blue_tiles
  have h1 : blue_tiles = 48, by
    simp [blue_tiles, current_tiles, red_tiles, tiles_needed, total_tiles]
  exact h1

end blue_tiles_count_l152_152907


namespace sec_seven_pi_over_six_l152_152982

theorem sec_seven_pi_over_six :
  Real.sec (7 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  sorry

end sec_seven_pi_over_six_l152_152982


namespace geometric_product_is_geometric_l152_152609

theorem geometric_product_is_geometric (q : ℝ) (a : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  ∀ n, (a n) * (a (n + 1)) = (q^2) * (a (n - 1) * a n) := by
  sorry

end geometric_product_is_geometric_l152_152609


namespace prime_sum_55_l152_152182

theorem prime_sum_55 (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p < q ∧ q < r ∧ r < s) 
  (h_eqn : 1 - (1 : ℚ)/p - (1 : ℚ)/q - (1 : ℚ)/r - (1 : ℚ)/s = 1 / (p * q * r * s)) :
  p + q + r + s = 55 := 
sorry

end prime_sum_55_l152_152182


namespace school_spent_total_amount_l152_152527

theorem school_spent_total_amount
  (num_cartons_pencils : ℕ)
  (boxes_per_carton_pencils : ℕ)
  (cost_per_box_pencils : ℕ)
  (num_cartons_markers : ℕ)
  (boxes_per_carton_markers : ℕ)
  (cost_per_box_markers : ℕ)
  (total_spent : ℕ)
  (h1 : num_cartons_pencils = 20)
  (h2 : boxes_per_carton_pencils = 10)
  (h3 : cost_per_box_pencils = 2)
  (h4 : num_cartons_markers = 10)
  (h5 : boxes_per_carton_markers = 5)
  (h6 : cost_per_box_markers = 4)
  (h7 : total_spent = 
        (num_cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils)
        + (num_cartons_markers * boxes_per_carton_markers * cost_per_box_markers)) :
  total_spent = 600 :=
by
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7.mpr rfl

end school_spent_total_amount_l152_152527


namespace count_correct_conclusions_l152_152934

theorem count_correct_conclusions :
  (∀ (l₁ l₂ : line) (p : plane), (l₁ ≠ l₂ ∧ l₁ ∥ p ∧ l₂ ∥ p → ¬ l₁ ∥ l₂)) ∧
  (∀ (l₁ l₂ : line), (l₁ ≠ l₂ ∧ ¬ ∃ p, p ∈ l₁ ∧ p ∈ l₂ → ¬ l₁ ∥ l₂)) ∧
  (∀ (l₁ l₂ l₃ : line), (l₁ ≠ l₂ ∧ l₁ ⊥ l₃ ∧ l₂ ⊥ l₃ → ¬ l₁ ∥ l₂)) ∧
  (∀ (l₁ : line) (p : plane), (¬ ∃ q : line, q ≠ l₁ ∧ q ⊂ p ∧ ¬ ∃ r : point, r ∈ q ∧ r ∉ l₁ → ¬ l₁ ∥ p)) →
  (number of correct conclusions = 0) :=
by sorry

end count_correct_conclusions_l152_152934


namespace reef_age_in_decimal_l152_152191

def octal_to_decimal (n: Nat) : Nat :=
  match n with
  | 367 => 7 * (8^0) + 6 * (8^1) + 3 * (8^2)
  | _   => 0  -- Placeholder for other values if needed

theorem reef_age_in_decimal : octal_to_decimal 367 = 247 := by
  sorry

end reef_age_in_decimal_l152_152191


namespace least_positive_divisible_by_smallest_primes_l152_152857

def smallest_primes := [2, 3, 5, 7, 11]

noncomputable def product_of_smallest_primes :=
  List.foldl (· * ·) 1 smallest_primes

theorem least_positive_divisible_by_smallest_primes :
  product_of_smallest_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_smallest_primes_l152_152857


namespace cos_B_right_triangle_l152_152710

theorem cos_B_right_triangle (ABC : Triangle) (h : ABC.isRightAngle ∠ C)
  (h_AB : ABC.hypotenuse = 15) (h_AC : ABC.legC = 9) : ∃ B: Angle, ABC.cos B = 4 / 5 := by
  sorry

end cos_B_right_triangle_l152_152710


namespace x_midpoint_of_MN_l152_152631

-- Definition: Given the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Definition: Point F is the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition: Points M and N are on the parabola
def on_parabola (M N : ℝ × ℝ) : Prop :=
  parabola M.2 M.1 ∧ parabola N.2 N.1

-- Definition: The sum of distances |MF| + |NF| = 6
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  dist M F + dist N F = 6

-- Theorem: Prove that the x-coordinate of the midpoint of MN is 2
theorem x_midpoint_of_MN (M N : ℝ × ℝ) (F : ℝ × ℝ) 
  (hF : focus F) (hM_N : on_parabola M N) (hDist : sum_of_distances M N F) :
  (M.1 + N.1) / 2 = 2 :=
sorry

end x_midpoint_of_MN_l152_152631


namespace smallest_solution_l152_152248

theorem smallest_solution (x : ℝ) : 
  (∃ x, (3 * x / (x - 3)) + ((3 * x^2 - 27) / x) = 15 ∧ ∀ y, (3 * y / (y - 3)) + ((3 * y^2 - 27) / y) = 15 → y ≥ x) → 
  x = -1 := 
by
  sorry

end smallest_solution_l152_152248


namespace cricketer_total_matches_l152_152419

theorem cricketer_total_matches (n : ℕ) (avg_total avg_first avg_last : ℝ) 
  (h1 : avg_total = 38.9)
  (h2 : avg_first = 41)
  (h3 : avg_last = 35.75)
  (h4 : 41 * 6 + 35.75 * 4 = 38.9 * n) :
  n = 10 :=
by
  sorry

end cricketer_total_matches_l152_152419


namespace ladder_length_l152_152896

def length_of_ladder (a b c d : ℝ) : ℝ := 
  let h0 := 9
  let h1 := 10.07212046142853
  let delta_h := 1
  let eq1 := h0^2 + b^2
  let eq2 := h1^2 + (b - delta_h)^2
  (eq1 - eq2) / delta_h

theorem ladder_length : length_of_ladder 9 10.723037 1 10.07212046142853 = 13.9965 := 
by
  sorry

end ladder_length_l152_152896


namespace system1_solution_system2_solution_system3_solution_l152_152407

theorem system1_solution (x y : ℝ) : 
  (x = 3/2) → (y = 1/2) → (x + 3 * y = 3) ∧ (x - y = 1) :=
by intros; sorry

theorem system2_solution (x y : ℝ) : 
  (x = 0) → (y = 2/5) → ((x + 3 * y) / 2 = 3 / 5) ∧ (5 * (x - 2 * y) = -4) :=
by intros; sorry

theorem system3_solution (x y z : ℝ) : 
  (x = 1) → (y = 2) → (z = 3) → 
  (3 * x + 4 * y + z = 14) ∧ (x + 5 * y + 2 * z = 17) ∧ (2 * x + 2 * y - z = 3) :=
by intros; sorry

end system1_solution_system2_solution_system3_solution_l152_152407


namespace equilateral_triangle_area_ratio_l152_152429

theorem equilateral_triangle_area_ratio (p : ℝ) (hp : p > 0) :
  let a_small := (sqrt 3 / 4) * p^2,
      a_large := (sqrt 3 / 4) * (2*p)^2 in
  6 * a_small / a_large = 3 / 2 :=
by
  -- Definitions of areas
  let a_small := (sqrt 3 / 4) * p^2,
      a_large := (sqrt 3 / 4) * (2*p)^2
  -- The proof will be skipped here as instructed
  sorry

end equilateral_triangle_area_ratio_l152_152429


namespace max_value_of_n_l152_152788

theorem max_value_of_n : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m ≤ n → (2 / 3)^(m - 1) * (1 / 3) ≥ 1 / 60) 
      ∧ 
    (∀ k : ℕ, k > n → (2 / 3)^(k - 1) * (1 / 3) < 1 / 60) 
      ∧ 
    n = 8 :=
by
  sorry

end max_value_of_n_l152_152788


namespace garage_sale_items_l152_152166

def total_items_sold (pieces_higher_price pieces_lower_price price_radio : ℕ) :=
  pieces_higher_price + pieces_lower_price + 1

theorem garage_sale_items (price_radio_is_15th_highest : 14)
  (price_radio_is_25th_lowest : 24) :
  total_items_sold price_radio_is_15th_highest price_radio_is_25th_lowest = 39 := by
  sorry

end garage_sale_items_l152_152166


namespace log2_P1_a_b_l152_152373

noncomputable def P (x : ℝ) : ℝ := (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆) * (x - r₇) * (x - r₈) * (x - r₉) * (x - r₁₀)
noncomputable def Q (x : ℝ) : ℝ := sorry -- Define Q properly based on the conditions

variables {r₁ r₂ r₃ r₄ r₅ r₆ r₇ r₈ r₉ r₁₀ : ℝ} (Hr: r₁ * r₂ * r₃ * r₄ * r₅ * r₆ * r₇ * r₈ * r₉ * r₁₀ = 2)

theorem log2_P1_a_b (hQ1 : Q 1 = 2) : ∃ (a b : ℤ), nat.gcd a b = 1 ∧ log2 (|P 1|) = a / b ∧ a + b = 19 :=
by
  sorry

end log2_P1_a_b_l152_152373


namespace magazine_cost_l152_152436

theorem magazine_cost (C M : ℝ) 
  (h1 : 4 * C = 8 * M) 
  (h2 : 12 * C = 24) : 
  M = 1 :=
by
  sorry

end magazine_cost_l152_152436


namespace orthographic_projection_remains_same_l152_152004

/-!
Problem: 
If the distance between an object and the projection plane is increased, 
then its orthographic projection will remain the same given that orthographic 
projection involves parallel lines and the projection size only depends on the 
dimensions of the object.
-/

-- Define orthographic projection properties
def orthographic_projection (P : Type) [Plane P] (obj : Type) :=
  ∃ (f : obj → P), ∀ (d : K), is_parallel_projection f obj d

-- Theorem statement
theorem orthographic_projection_remains_same (P : Type) [Plane P] (obj : Type) :
  ∀ d₁ d₂ : ℝ, d₁ ≤ d₂ → orthographic_projection P obj →
  orthographic_projection P obj :=
by
  sorry

end orthographic_projection_remains_same_l152_152004


namespace zeros_of_quadratic_l152_152688

theorem zeros_of_quadratic (a b : ℝ) (h : a + b = 0) : 
  ∀ x, (b * x^2 - a * x = 0) ↔ (x = 0 ∨ x = -1) :=
by
  intro x
  sorry

end zeros_of_quadratic_l152_152688


namespace VS_is_correct_l152_152709

variables {M N P Q R S U V T W : Type*} [rect : IsRectangle M N P Q]
variables (RM : length M R = 15) (MT : length M T = 20) (RT : length R T = 25)
variables [perpUVNQ : IsPerpendicular U V N Q] [eqNRRU : length N R = length R U]
variables [eqAngle : angle M R S = π / 2] [intRSUV : Intersects R S U V T]
variables [WInPQ : W In PQ]

noncomputable def findVS : ℝ := 
  let VS : ℝ := 12 in
  VS

theorem VS_is_correct : findVS = 12 := 
  sorry

end VS_is_correct_l152_152709


namespace area_of_region_bounded_by_parabolas_l152_152953

theorem area_of_region_bounded_by_parabolas :
  let f := λ x : ℝ, x^2
  let g := λ x : ℝ, 8 - x^2
  let inter_p1 := (2 : ℝ)
  let inter_p2 := (-2 : ℝ)
  ∫ x in -2..2, (g x - f x) = (64 : ℝ) / 3 :=
by
  sorry

end area_of_region_bounded_by_parabolas_l152_152953


namespace g_at_3_l152_152113

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 : (∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2 + 1) → 
  g 3 = 130 / 21 := 
by 
  sorry

end g_at_3_l152_152113


namespace problem_1_problem_2_l152_152382

-- Define the function f(x) with the parameter b.
def f (x : ℝ) (b : ℝ) : ℝ := x^2 - x + b

-- Problem (1): f(x) has a minimum value of 7/4 at x = sqrt(2)
theorem problem_1 (b a : ℝ) (h1 : f (Real.log2 a) b = b) (h2 : Real.log2 (f a b) = 2) (ha : a ≠ 1) :
  (∀ x : ℝ, f (Real.log2 x) 2 ≥ 7/4) ∧ f (Real.log2 (Real.sqrt 2)) 2 = 7/4 := by
sorry

-- Problem (2): For what x does f(log2(x)) > f(1) and log2(f(x)) < f(1) means 0 < x < 1
theorem problem_2 (b : ℝ) :
  (∀ x : ℝ, (f (Real.log2 x) b > f 1 b ∧ Real.log2 (f x b) < f 1 b) → (0 < x ∧ x < 1)) := by
sorry

end problem_1_problem_2_l152_152382


namespace probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152925

theorem probability_roll_differs_by_three_on_two_eight_sided_dies : 
  let S := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 } in -- sample space
  let E := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 ∧ (x = y + 3 ∨ y = x + 3) } in -- event of interest
  ((E.card : ℚ) / S.card) = 1 / 8 := 
by
  sorry

end probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152925


namespace minimum_value_of_f_l152_152301

noncomputable def f (a b x : ℝ) := (a * x + b) / (x^2 + 4)

theorem minimum_value_of_f (a b : ℝ) (h1 : f a b (-1) = 1)
  (h2 : (deriv (f a b)) (-1) = 0) : 
  ∃ (x : ℝ), f a b x = -1 / 4 := 
sorry

end minimum_value_of_f_l152_152301


namespace cross_product_correct_l152_152595

def u : ℝ × ℝ × ℝ := (4, 3, -2)
def v : ℝ × ℝ × ℝ := (-1, 2, 5)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct :
  cross_product u v = (19, -18, 11) :=
by
  have h1 : cross_product u v = (19, -18, 11) := sorry
  exact h1

end cross_product_correct_l152_152595


namespace complex_modulus_correct_l152_152812

noncomputable def modulus_complex_expression : ℝ :=
  let i : ℂ := complex.I in
  complex.abs ((3 + i) / (i * i))

theorem complex_modulus_correct : modulus_complex_expression = real.sqrt 10 := by
  sorry

end complex_modulus_correct_l152_152812


namespace correct_option_D_l152_152466

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end correct_option_D_l152_152466


namespace solve_equation_l152_152425

theorem solve_equation (m n : ℝ) (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : m ≠ n) :
  ∀ x : ℝ, ((x + m)^2 - 3 * (x + n)^2 = m^2 - 3 * n^2) ↔ (x = 0 ∨ x = m - 3 * n) :=
by
  sorry

end solve_equation_l152_152425


namespace time_since_production_approximate_l152_152765

noncomputable def solve_time (N N₀ : ℝ) (t : ℝ) : Prop :=
  N = N₀ * (1 / 2) ^ (t / 5730) ∧
  N / N₀ = 3 / 8 ∧
  t = 8138

theorem time_since_production_approximate
  (N N₀ : ℝ)
  (h_decay : N = N₀ * (1 / 2) ^ (t / 5730))
  (h_ratio : N / N₀ = 3 / 8) :
  t = 8138 := 
sorry

end time_since_production_approximate_l152_152765


namespace solve_frac_eq_l152_152833

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l152_152833


namespace brokerage_percentage_l152_152420

theorem brokerage_percentage
  (cash_realized : ℝ)
  (cash_before_brokerage : ℝ)
  (h₁ : cash_realized = 109.25)
  (h₂ : cash_before_brokerage = 109) :
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100 = 0.23 := 
by
  sorry

end brokerage_percentage_l152_152420


namespace solve_xyz_l152_152318

theorem solve_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end solve_xyz_l152_152318


namespace graph_contains_triangle_or_4cycle_l152_152752

theorem graph_contains_triangle_or_4cycle
  (G : SimpleGraph V) [Fintype V] (n : ℕ) (h_vertex_count : Fintype.card V = n)
  (h_edge_count : G.edge_finset.card ≥ (1/2) * n * Real.sqrt (n-1)) :
  ∃ (t : Finset V), (G.inducedSubgraph t).is_triangle ∨ ∃ (c : SimpleGraph.V C) (C : V → V), G.is_cycle (4 : ℕ) :=
sorry

end graph_contains_triangle_or_4cycle_l152_152752


namespace options_B_and_D_even_and_increasing_l152_152211

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

def f_B (x : ℝ) : ℝ := x^2 + 2
def f_D (x : ℝ) : ℝ := abs x + 1

theorem options_B_and_D_even_and_increasing :
  (is_even f_B ∧ is_increasing_on f_B (set.Ioi 0)) ∧
  (is_even f_D ∧ is_increasing_on f_D (set.Ioi 0)) :=
by
  sorry

end options_B_and_D_even_and_increasing_l152_152211


namespace smallest_y_squared_l152_152052

noncomputable def isosceles_trapezoid (EF GH EH FG : ℝ) : Prop :=
  EF = 112 ∧ GH = 36 ∧ EH = FG

noncomputable def circle_tangent (y : ℝ) : ℝ :=
  let EO := (112 : ℝ) / 2
  let JO := real.sqrt (36 * 56)
  let JO_squared := 36 * 56
  let EO_squared := EO^2
  let y_squared := EO_squared - JO_squared
  y_squared

theorem smallest_y_squared (y : ℝ) (h1 : isosceles_trapezoid 112 36 y y)
  (h2 : circle_tangent y = 1120) : y^2 = 1120 :=
by
  rw [circle_tangent y] at h2
  exact h2

end smallest_y_squared_l152_152052


namespace sum_y_values_l152_152306

-- Given conditions as Lean definitions
def regression_line (x : ℝ) : ℝ := -3 + 2 * x
def sum_x_values : ℝ := 17
def num_samples : ℝ := 10
def mean_x : ℝ := sum_x_values / num_samples
def mean_y : ℝ := regression_line mean_x

-- Proof statement to be proven in Lean
theorem sum_y_values (h₁ : ∀ x, regression_line x = -3 + 2 * x)
                    (h₂ : sum_x_values = 17)
                    (h₃ : num_samples = 10)
                    (h₄ : mean_x = sum_x_values / num_samples)
                    (h₅ : mean_y = regression_line mean_x) :
  num_samples * mean_y = 4 := 
by
  sorry

end sum_y_values_l152_152306


namespace simplify_expression_l152_152969

theorem simplify_expression (x : ℝ) : 
  ( ( (x^(16/8))^(1/4) )^3 * ( (x^(16/4))^(1/8) )^5 ) = x^4 := 
by 
  sorry

end simplify_expression_l152_152969


namespace Ada_initial_seat_l152_152406

def seats := Fin 6

structure Friends :=
(Bea Ceci Dee Edie Fiona : seats)

def original_positions (A : seats) (F : Friends) := 
  { Ada := A, friends := F }

def final_positions (Ada : seats) (F : Friends) :=
  match Ada, F with
  | 0, ⟨B, C, D, E, F⟩ => 
    { Ada := 0, friends := 
        { Bea := B + 3,
          Ceci := C - 1,
          Dee := E,
          Edie := D,
          Fiona := F } }
  | 5, ⟨B, C, D, E, F⟩ => 
    { Ada := 5, friends := 
        { Bea := B + 3,
          Ceci := C - 1,
          Dee := E,
          Edie := D,
          Fiona := F} }
  | _, _ => sorry

theorem Ada_initial_seat (A : seats) (F : Friends) :
  (final_positions A F).Ada = 0 \/ (final_positions A F).Ada = 5 -> A = 2 + 1 := sorry

end Ada_initial_seat_l152_152406


namespace machines_solution_l152_152840

theorem machines_solution (x : ℝ) (h : x > 0) :
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := 
by
  sorry

end machines_solution_l152_152840


namespace ball_radius_of_equal_surface_area_l152_152131

noncomputable def surface_area_cube (side_length : ℝ) : ℝ :=
  6 * (side_length ^ 2)

noncomputable def surface_area_sphere (radius : ℝ) : ℝ :=
  4 * Real.pi * (radius ^ 2)

theorem ball_radius_of_equal_surface_area (side_length := 6.5) : 
    ∃ r : ℝ, surface_area_cube side_length = surface_area_sphere r ∧ r ≈ 4 :=
by
  sorry

end ball_radius_of_equal_surface_area_l152_152131


namespace cos_A_minus_B_eq_nine_eighths_l152_152636

theorem cos_A_minus_B_eq_nine_eighths (A B : ℝ)
  (h1 : Real.sin A + Real.sin B = 1 / 2)
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9 / 8 := 
by
  sorry

end cos_A_minus_B_eq_nine_eighths_l152_152636


namespace smallest_possible_sector_angle_l152_152048

theorem smallest_possible_sector_angle : 
  ∃ (a_1 : ℕ) (d : ℕ), 
    (∀ (k : ℕ), k < 15 → is_arith_seq a_1 d k) ∧ 
    (15 * a_1 + 105 * d = 360) ∧ 
    a_1 = 3 :=
by
  sorry

end smallest_possible_sector_angle_l152_152048


namespace total_cows_l152_152220

def number_of_cows_in_herd : ℕ := 40
def number_of_herds : ℕ := 8
def total_number_of_cows (cows_per_herd herds : ℕ) : ℕ := cows_per_herd * herds

theorem total_cows : total_number_of_cows number_of_cows_in_herd number_of_herds = 320 := by
  sorry

end total_cows_l152_152220


namespace dice_minimum_rolls_l152_152463

theorem dice_minimum_rolls (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
                           (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) 
                           (h4 : 1 ≤ d4 ∧ d4 ≤ 6) :
  ∃ n, n = 43 ∧ ∀ (S : ℕ) (x : ℕ → ℕ), 
  (∀ i, 4 ≤ S ∧ S ≤ 24 ∧ x i = 4 ∧ (x i ≤ 6)) →
  (n ≤ 43) ∧ (∃ (k : ℕ), k ≥ 3) :=
sorry

end dice_minimum_rolls_l152_152463


namespace number_of_paths_l152_152992

theorem number_of_paths (A B D : Type)
  (paths_from_A_to_B : ℕ) (paths_from_B_to_D : ℕ) (direct_path_from_A_to_D : ℕ)
  (h1 : paths_from_A_to_B = 2)
  (h2 : paths_from_B_to_D = 3)
  (h3 : direct_path_from_A_to_D = 1) : 
  paths_from_A_to_B * paths_from_B_to_D + direct_path_from_A_to_D = 7 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end number_of_paths_l152_152992


namespace impossible_arrangement_of_polyhedra_l152_152035

theorem impossible_arrangement_of_polyhedra 
  (P : Type) 
  [Polyhedron P] 
  (equal_polyhedra : ∀ p1 p2 : P, is_congruent p1 p2) 
  (bounded_layer : ℝ → ℝ → set P)
  (no_removal : ∀ (layer : set P) (p : P), 
                  p ∈ layer → 
                  (∀ q ∈ layer, q ≠ p → ¬ can_be_removed layer p)) : 
  ¬ ∃ (arrangement : set P), 
    ∀ p ∈ arrangement, 
      exists (q : P), q ∈ arrangement → 
      interlock arran arrangement (q : P) :=
sorry

end impossible_arrangement_of_polyhedra_l152_152035


namespace probability_sum_odd_l152_152498

theorem probability_sum_odd :
  (∃ (balls : Finset ℕ), balls.card = 13 ∧ ∀ x ∈ balls, x ∈ Finset.range 14) →
  (∃ (drawnBalls : Finset ℕ), drawnBalls.card = 7) →
  (let oddBalls := (Finset.filter (λ x, x % 2 = 1) balls), evenBalls := (Finset.filter (λ x, x % 2 = 0) balls) in
    (∑ b in drawnBalls, b) % 2 = 1 →
    ((Finset.card oddBalls = 7 ∧ Finset.card evenBalls = 6) ∧
     Finset.card (Finset.filter (λ x, x % 2 = 1) drawnBalls) % 2 = 1 ∧
    ∑ _ in Finset.pairs oddBalls drawnBalls, _ + ∑ _ in Finset.pairs evenBalls drawnBalls, _ = 7)) →
  (let favorable := ∑ n in (Finset.filter (λ n, n ∈ [5, 3, 1]) (Finset.range 8)), 
       (Nat.choose 7 n) * (Nat.choose 6 (7 - n))) in
        favorable / Nat.choose 13 7 = 141 / 286) :=
sorry

end probability_sum_odd_l152_152498


namespace circles_internally_tangent_l152_152821

-- Definitions and conditions from part a)
def circle1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 16}

-- Prove the circles are internally tangent
theorem circles_internally_tangent : ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ), 
  c1 = (1, 0) ∧ r1 = 1 ∧
  c2 = (1, 3) ∧ r2 = 4 ∧
  ∥c1 - c2∥ = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l152_152821


namespace find_angle_y_l152_152353

noncomputable def angle_y (y : ℝ) : ℝ :=
y

theorem find_angle_y (y : ℝ) :
  parallel m n →
  angle_A = 45 →
  angle_B = 90 →
  angle_J = 45 →
  angle_y y + angle_J = 180 →
  angle_y y = 135 :=
by
  intros _ _ _ _ h
  sorry

end find_angle_y_l152_152353


namespace sara_spent_on_salad_l152_152091

def cost_of_hotdog : ℝ := 5.36
def total_lunch_bill : ℝ := 10.46
def cost_of_salad : ℝ := total_lunch_bill - cost_of_hotdog

theorem sara_spent_on_salad : cost_of_salad = 5.10 :=
by
  unfold cost_of_salad
  norm_num
  exact eq.symm (by norm_num)

end sara_spent_on_salad_l152_152091


namespace intervals_of_monotonicity_find_min_value_g_l152_152623
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * (x - 1) / (x ^ 2)

theorem intervals_of_monotonicity (a : ℝ) (ha : 0 < a) :
  (∀ x, x < 0 → (f a) x > (f a) (x + 0.1)) ∧
  (∀ x, 0 < x ∧ x < 2 → (f a) x < (f a) (x + 0.1)) ∧
  (∀ x, x > 2 → (f a) x > (f a) (x + 0.1)) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  x * real.log x - x ^ 2 * (f a x)

theorem find_min_value_g (a : ℝ) (ha : 0 < a) :
  if (1 ≤ a ∧ a ≤ 2) then
    if (a ≤ 1) then (∀ x ∈ set.Icc 1 real.exp, g a 1 ≤ g a x) ∧ g a 1 = 0
    else (∀ x ∈ set.Icc 1 real.exp, g a (real.exp (a - 1)) ≤ g a x) ∧ g a (real.exp (a - 1)) = a - real.exp (a - 1)
  else
    (∀ x ∈ set.Icc 1 real.exp, g a real.exp ≤ g a x) ∧ g a real.exp = real.exp * (1 - a) :=
sorry

end intervals_of_monotonicity_find_min_value_g_l152_152623


namespace find_first_term_of_arithmetic_progression_l152_152989

-- Definitions for the proof
def arithmetic_progression_first_term (L n d : ℕ) : ℕ :=
  L - (n - 1) * d

-- Theorem stating the proof problem
theorem find_first_term_of_arithmetic_progression (L n d : ℕ) (hL : L = 62) (hn : n = 31) (hd : d = 2) :
  arithmetic_progression_first_term L n d = 2 :=
by
  -- proof omitted
  sorry

end find_first_term_of_arithmetic_progression_l152_152989


namespace problem_inequality_l152_152065

theorem problem_inequality
  (n : ℕ)
  (a : ℕ → ℝ)
  (h_seq : ∀ i j : ℕ, i ≤ j ∧ j ≤ n → a i ≥ a j)
  (h_last : a (n + 1) = 0) :
  sqrt (∑ k in Finset.range n.succ, a k) ≤ ∑ k in Finset.range n.succ, sqrt (k + 1) * (sqrt (a k) - sqrt (a (k.succ))) :=
sorry

end problem_inequality_l152_152065


namespace real_solutions_equation_l152_152983

theorem real_solutions_equation (x : ℝ) : 
  (x + 1) ^ 3 + (3 - x) ^ 3 = 35 ↔ (x = 1 + sqrt (19/3) / 2 ∨ x = 1 - sqrt (19/3) / 2) :=
by
  sorry

end real_solutions_equation_l152_152983


namespace polynomial_degree_l152_152733

variable {P : Polynomial ℝ}

theorem polynomial_degree (h1 : ∀ x : ℝ, (x - 4) * P.eval (2 * x) = 4 * (x - 1) * P.eval x) (h2 : P.eval 0 ≠ 0) : P.degree = 2 := 
sorry

end polynomial_degree_l152_152733


namespace more_than_5_holes_l152_152069

/-- A 'strange ring' is defined as a circle with a square hole in the center. 
  The centers of the circle and the square coincide. -/
structure StrangeRing where
  R : ℝ  -- radius of the circle
  a : ℝ  -- side length of the square
  center : (ℝ, ℝ)  -- center coordinates of the circle and square
  square_center : center = center  -- by definition, the square center coincides with the circle center
  circle_contains_square : a <= 2 * R  -- the square is fully contained within the circle

/-- The predicate for valid positioning of two strange rings that results in more than 5 holes. -/
def more_than_5_holes_possible (r1 r2 : StrangeRing) : Prop :=
  ∃ points : Finset (ℝ, ℝ), 
  points.card > 5 ∧ 
  ∀ point ∈ points, (point ∉ r1.center ∧ point ∉ r2.center)

theorem more_than_5_holes :
  ∃ r1 r2 : StrangeRing, more_than_5_holes_possible r1 r2 :=
by
  sorry

end more_than_5_holes_l152_152069


namespace tree_leaves_remaining_after_three_weeks_l152_152918

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end tree_leaves_remaining_after_three_weeks_l152_152918


namespace problem_1_problem_2_problem_3_l152_152876

-- Definition of an arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) :=
∀ n m, a (n + m) = a n + m * d

-- Problem (1)
theorem problem_1 (a : ℕ → ℤ) (d : ℤ) (m : ℕ) (h_seq : arithmetic_seq a d)
  (h_a3 : a 3 = 3) (h_a5 : a 5 = 6) (h_m : m > 5)
  (h_geom : a 5^2 = a 3 * a m) :
  m = 9 :=
sorry

-- Problem (2)
theorem problem_2 (a : ℕ → ℤ) (d : ℤ) (m_list : list ℕ) (h_seq : arithmetic_seq a d)
  (h_a5 : a 5 = 6) (h_a3_gt_1 : a 3 > 1)
  (h_geom_list : ∀ (m ∈ m_list), 5 < m ∧ a 3 * (a 3^(m_list.index_of m + 1)) = a 3 + (m - 3) * d ∧ m ∈ ℕ) :
  ∃ a3_vals, ∀ a3 ∈ a3_vals, a 3 = a3 ∧ (a 3 = 3 ∨ a 3 = 2) :=
sorry

-- Problem (3)
theorem problem_3 (a : ℕ → ℤ) (d : ℤ) (h_seq : arithmetic_seq a d)
  (h_a5 : a 5 = 6) :
  (a 3 = 2 → (∀ m, 5 < m → a m = a 3 * (a 3^(m_list.index_of m + 1)) + (m - 3) * d) → (∀ m, a m < a 5)) :=
sorry

end problem_1_problem_2_problem_3_l152_152876


namespace probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152926

theorem probability_roll_differs_by_three_on_two_eight_sided_dies : 
  let S := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 } in -- sample space
  let E := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 ∧ (x = y + 3 ∨ y = x + 3) } in -- event of interest
  ((E.card : ℚ) / S.card) = 1 / 8 := 
by
  sorry

end probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152926


namespace negation_of_p_l152_152435

-- Define statement p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x^2 - x + 1 < 0

-- The proof statement
theorem negation_of_p : p → not not_p :=
by
  intro hp
  sorry

end negation_of_p_l152_152435


namespace average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_at_2_seconds_l152_152196

-- Define the function representing the particle's distance over time.
def f (x : ℝ) : ℝ := (2 / 3) * x^3 + x^2 + 2 * x

-- Define the average velocity function over an interval [a, b].
def avg_velocity (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

-- Define the derivative of the function f.
def f' (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 2

-- Theorem 1: Prove the average velocity of the particle during the first second is 3 / 2 m/s.
theorem average_velocity_first_second : avg_velocity f 0 1 = 3 / 2 := 
by
  sorry

-- Theorem 2: Prove the instantaneous velocity of the particle at the end of the first second is 6 m/s.
theorem instantaneous_velocity_end_first_second : f' 1 = 6 := 
by
  sorry

-- Theorem 3: Prove that the particle's velocity reaches 14 m/s at x = 2 s.
theorem velocity_reaches_14_at_2_seconds (x : ℝ) (h : f' x = 14) : x = 2 := 
by
  sorry

end average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_at_2_seconds_l152_152196


namespace parallelogram_to_rectangle_l152_152022

theorem parallelogram_to_rectangle (AB DC AD BC : ℝ) (A B C D : Point) : 
  parallelogram ABCD → (angle D = 90) → rectangle ABCD :=
by
  sorry

end parallelogram_to_rectangle_l152_152022


namespace find_radii_l152_152080

-- Define the sequential points on a line and their relative positions
variable (A B C D E : ℝ)
variable (O Q : ℝ) -- centers of circles Ω and ω
variable (R r : ℝ) -- radii of circles Ω and ω

-- Conditions
axiom conditions : 
  A = 0 ∧ B = 1 ∧ C = 3 ∧ D = 4 ∧ E = 6 ∧ -- Points on the number line
  (Q - O) = (C - D) + (E - A) ∧           -- Centers of circles are on the line including point D
  (R - r) = |O - Q| ∧                     -- Circles Ω and ω are tangent
  R = (sqrt ((Q - C)^2 + D^2)) ∧          -- Distance formula applied for radius calculation
  r = (sqrt (4 * (Q - O)^2 + D^2))        -- Distance formula for smaller radius

-- Statement to prove
theorem find_radii : 
  R = \frac{27}{2 \sqrt{19}} ∧ r = \frac{8}{\sqrt{19}} := sorry

end find_radii_l152_152080


namespace length_PR_l152_152025

theorem length_PR (x y : ℝ) (h : x^2 + y^2 = 200) : 
  √((2 * x^2) + (2 * y^2)) = 20 := 
by
  sorry

end length_PR_l152_152025


namespace correct_number_l152_152815

theorem correct_number : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  -- proof starts here
  sorry

end correct_number_l152_152815


namespace y_optimal_value_l152_152976

def y_function (x a : ℝ) : ℝ :=
  (x + a / 2)^2 - (3 * x + 2 * a)^2

theorem y_optimal_value :
  let x := -11 / 8 in
  let a := -2 in
  y_function x a = 1.25 :=
by
  sorry

end y_optimal_value_l152_152976


namespace optimal_M_is_2_l152_152474

def table : Type := matrix (fin (2*n)) (fin (2*n)) ℤ
def is_valid_table (t : table) : Prop :=
  (∀ i j, t i j = 1 ∨ t i j = -1) ∧
  (∑ i j, t i j = 0) ∧
  (∑ i, ∑ j, (if t i j = 1 then 1 else 0)) = 2 * n^2

def row_sum (t : table) (i : fin (2*n)) : ℤ :=
  ∑ j, t i j

def col_sum (t : table) (j : fin (2*n)) : ℤ :=
  ∑ i, t i j

def M (t : table) : ℕ :=
  min (finset.min' (finset.image (abs ∘ row_sum t) finset.univ) (by continue)) 
      (finset.min' (finset.image (abs ∘ col_sum t) finset.univ) (by continue))

theorem optimal_M_is_2 (n : ℕ) (t : table) (ht : is_valid_table t) : M t = 2 :=
sorry

end optimal_M_is_2_l152_152474


namespace probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152927

theorem probability_roll_differs_by_three_on_two_eight_sided_dies : 
  let S := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 } in -- sample space
  let E := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 ∧ (x = y + 3 ∨ y = x + 3) } in -- event of interest
  ((E.card : ℚ) / S.card) = 1 / 8 := 
by
  sorry

end probability_roll_differs_by_three_on_two_eight_sided_dies_l152_152927


namespace units_digit_17_mul_27_l152_152599

theorem units_digit_17_mul_27 : 
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  units_product = 9 := by
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  sorry

end units_digit_17_mul_27_l152_152599


namespace wheel_diameter_l152_152537

noncomputable def diameter_of_wheel (total_distance : ℝ) (num_revolutions : ℝ) (pi_approx : ℝ) : ℝ :=
  total_distance / (num_revolutions * pi_approx)

theorem wheel_diameter :
  diameter_of_wheel 948 18.869426751592357 3.141592653589793 ≈ 16 :=
by
  -- Skipping proof
  sorry

end wheel_diameter_l152_152537


namespace problem1_problem2_l152_152491

open BigOperators

noncomputable def C (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem problem1 (n : ℕ) (h : n ≥ 1) : 
  (∑ k in finset.range n, k * (C n (k + 1)) * x^k) = n * (1 + x)^(n - 1) :=
by sorry

theorem problem2 (n : ℕ) (h : n ≥ 1) : 
  (∑ k in finset.range n, (k+1)^2 * (C n (k + 1))) = 2^(n-2) * n * (n+1) :=
by sorry

end problem1_problem2_l152_152491


namespace sum_of_repeating_decimals_l152_152747

def repeating_decimals_sum : Real :=
  let T := {x | ∃ (a b : ℕ), a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * a + b) / 99}
  Set.sum T

theorem sum_of_repeating_decimals : repeating_decimals_sum = 413.5 :=
  by
    sorry

end sum_of_repeating_decimals_l152_152747


namespace relatively_prime_2n_plus_1_4n2_plus_1_l152_152754

theorem relatively_prime_2n_plus_1_4n2_plus_1 (n : ℕ) (h : n > 0) : 
  Nat.gcd (2 * n + 1) (4 * n^2 + 1) = 1 := 
by
  sorry

end relatively_prime_2n_plus_1_4n2_plus_1_l152_152754


namespace g_min_value_a_values_sum_inequality_l152_152659

-- Definition and proof for minimum value of g(x)
theorem g_min_value (m : Real) (a : Real) (x : Real) (h_a : True) (h_m : 0 < m) (h_a_half : a = 1/2)
  (h_g : ∀ x > 0, g(x) = e^(a*x) / x) :
  g_min = if 0 < m ∧ m ≤ 1 then e^((m+1)/2)/(m+1)
           else if 1 < m ∧ m < 2 then e/2
           else if m ≥ 2 then e^(m/2)/m else e^(m/2)/m :=
sorry

-- Definition and proof for values of a
theorem a_values (a : Real) :
  (∀ x : Real, e^(a*x) - x - 1 ≥ 0) ↔ (a = 1) :=
sorry

-- Definition and proof for the series sum
theorem sum_inequality (n : Nat) :
  ∑ i in Finset.range (n + 1), 1 / (i+1) * (√e)^(i+1) < 4 / e :=
sorry

end g_min_value_a_values_sum_inequality_l152_152659


namespace percent_of_games_lost_l152_152443

theorem percent_of_games_lost (w l : ℕ) (h1 : w / l = 8 / 5) (h2 : w + l = 65) :
  (l * 100 / 65 : ℕ) = 38 :=
sorry

end percent_of_games_lost_l152_152443


namespace shop_earnings_correct_l152_152336

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end shop_earnings_correct_l152_152336


namespace sum_of_rep_decimals_l152_152743

open scoped BigOperators

def rep_decimals_set : set ℝ :=
  { x | ∃ a b : ℕ, a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * (a : ℝ) + (b : ℝ)) / 99 }

theorem sum_of_rep_decimals : ∑ x in rep_decimals_set, x = 90 / 11 :=
sorry

end sum_of_rep_decimals_l152_152743


namespace det_of_dilation_matrix_l152_152056

-- Define the matrix E as given in conditions
def E : Matrix (Fin 2) (Fin 2) ℝ :=
  !![9, 0; 0, 9]

-- State the theorem to prove that the determinant of matrix E is 81
theorem det_of_dilation_matrix : det E = 81 := by
  sorry

end det_of_dilation_matrix_l152_152056


namespace perpendicular_planes_of_perpendicular_lines_l152_152285

universe u

variables {Point : Type u} [EuclideanSpace Point]
variables (m n : Line Point) (α β : Plane Point)

def perpendicular (x y : Plane Point) : Prop := ∃ (z : Line Point), z ∈ x ∧ z ∈ y ∧ z ⊥ x ∧ z ⊥ y

theorem perpendicular_planes_of_perpendicular_lines 
  (h1: m ∈ α) (h2: n ∈ β) (h3: m ⊥ α) (h4: n ⊥ β) (h5: m ⊥ n) : α ⊥ β := 
sorry

end perpendicular_planes_of_perpendicular_lines_l152_152285


namespace box_production_ratio_l152_152160

theorem box_production_ratio (x y : ℕ) 
  (hA_prod: Machine_A_produces_in_ten_minutes x)
  (hB_prod: Machine_B_produces_in_five_minutes y)
  (hA_B_together: Machines_A_and_B_produce_together_in_twenty_minutes (10 * x)) :
  y / x = 2 :=
by
  /- Given conditions:
     hA_prod: Machine A produces "x" boxes in 10 minutes.
     hB_prod: Machine B produces "y" boxes in 5 minutes.
     hA_B_together: Machines A and B together produce "10x" boxes in 20 minutes.
   -/ 
  sorry

end box_production_ratio_l152_152160


namespace no_such_function_exists_l152_152574

theorem no_such_function_exists :
  ∀ (f : ℝ → ℝ), (∀ (x y : ℝ), |f(x + y) + sin x + sin y| < 2) → False :=
by
  -- The mathematical proof would go here, showing that such a function cannot exist.
  sorry

end no_such_function_exists_l152_152574


namespace frog_arrangement_l152_152837

theorem frog_arrangement (frogs : Finset ℕ) (green_frogs red_frogs : Finset ℕ) (blue_frog : ℕ) 
  (h_frogs : frogs.card = 8) 
  (h_green : green_frogs.card = 3) 
  (h_red : red_frogs.card = 4) 
  (h_blue : ∃! b, b = blue_frog) 
  (h_exclusive_green_red : ∀ g ∈ green_frogs, ∀ r ∈ red_frogs, (green_frogs ∪ red_frogs).no_adj (frogs.erase blue_frog) g r)
  (h_exclusive_red_blue : ∀ r ∈ red_frogs, ¬ blue_frog ∈ frogs.eraseₓ r) : 
  ∃! (arrangements : Finset (Finset ℕ)), arrangements.card = 288 := 
sorry

-- Add any necessary helper definitions or lemmas here:
namespace Finset

def no_adj (s : Finset ℕ) (x y : ℕ) : Prop :=
  -- Define condition to check no adjacency between x and y in set s
  ∀ (a b : ℕ), (a ∈ s ∧ b ∈ s) → abs (a - b) ≠ 1

end Finset

end frog_arrangement_l152_152837


namespace color_drawing_cost_l152_152366

theorem color_drawing_cost (cost_bw : ℕ) (surcharge_ratio : ℚ) (cost_color : ℕ) :
  cost_bw = 160 →
  surcharge_ratio = 0.50 →
  cost_color = cost_bw + (surcharge_ratio * cost_bw : ℚ).natAbs →
  cost_color = 240 :=
by
  intros h_bw h_surcharge h_color
  rw [h_bw, h_surcharge] at h_color
  exact h_color

end color_drawing_cost_l152_152366


namespace harmonic_division_midpoints_l152_152118

-- Definitions for the geometric problem
variable (E F G H A B C D : Type) [AddGroup E] [AddGroup F] [AddGroup G] [AddGroup H]
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Conditions: Trapezoid EFGH with EF parallel GH
def is_trapezoid (E F G H : Type) : Prop := sorry

-- Extended lines EH and FG intersect at A
def extended_lines_intersect_at (E H F G A : Type) : Prop := sorry

-- Diagonals EG and FH intersect at B
def diagonals_intersect_at (E G F H B : Type) : Prop := sorry

-- C and D harmonically divide AB
def harmonically_divide (A B C D : Type) : Prop := sorry

-- Theorems: Proving the main statement
theorem harmonic_division_midpoints (E F G H A B C D : Type) 
  [is_trapezoid E F G H] 
  [extended_lines_intersect_at E H F G A] 
  [diagonals_intersect_at E G F H B] 
  [harmonically_divide A B C D] : 
  (is_midpoint C G H) ∧ (is_midpoint D E F) :=
by sorry

end harmonic_division_midpoints_l152_152118


namespace max_divisor_of_f_l152_152617

def f (n : ℕ) : ℕ := (2 * n + 7) * (3 ^ n) + 9

theorem max_divisor_of_f : ∃ m > 0, (∀ n : ℕ, n > 0 → f n % m = 0) ∧ m = 36 := by
  use 36
  split
  { exact Nat.succ_pos' 35 }
  { intro n hn
    sorry } -- Proof will go here

end max_divisor_of_f_l152_152617


namespace problem_statement_l152_152296

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, ((x^2) / (a^2) + (y^2) / (b^2) = 1)

noncomputable def max_area_slope (F P : ℝ × ℝ) (area : ℝ) (k : ℝ) : Prop :=
  F = (-2, 0) ∧ P = (-8, 0) ∧ 
  ∃ a b : ℝ, 
    (2 * a / (2 * 2) = 2) ∧ 
    (a = 4) ∧ 
    (b^2 = 16 - 4) ∧ 
    ellipse_equation 4 (√12) ∧ 
    (area = 3 * √3) ∧ 
    (abs k = √21 / 14)

theorem problem_statement (a b : ℝ) (F P : ℝ × ℝ) (area : ℝ) (k : ℝ)
  (h1 : 2 * a / (2 * 2) = 2)
  (hF : F = (-2, 0)) 
  (hP : P = (-8, 0)) 
  (ha : a = 4) 
  (hb : b^2 = 16 - 4) 
  (he : ellipse_equation 4 (√12)) 
  (ha_max : area = 3 * √3) 
  (hk_max : abs k = √21 / 14) :
  max_area_slope F P area k :=
by 
  exact ⟨hF, hP, a, b, h1, ha, hb, he, ha_max, hk_max⟩

end problem_statement_l152_152296


namespace first_group_men_count_l152_152006

/-- Given that 10 men can complete a piece of work in 90 hours,
prove that the number of men M in the first group who can complete
the same piece of work in 25 hours is 36. -/
theorem first_group_men_count (M : ℕ) (h : (10 * 90 = 25 * M)) : M = 36 :=
by
  sorry

end first_group_men_count_l152_152006


namespace simplify_expression_l152_152405

theorem simplify_expression (a b : ℤ) (h1 : a = 1) (h2 : b = -4) :
  4 * (a^2 * b + a * b^2) - 3 * (a^2 * b - 1) + 2 * a * b^2 - 6 = 89 := by
  sorry

end simplify_expression_l152_152405


namespace cos_squared_plus_twice_sin_double_alpha_l152_152313

theorem cos_squared_plus_twice_sin_double_alpha (α : ℝ) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_squared_plus_twice_sin_double_alpha_l152_152313


namespace smallest_n_divisible_by_one_billion_l152_152389

-- Define the sequence parameters and the common ratio
def first_term : ℚ := 5 / 8
def second_term : ℚ := 50
def common_ratio : ℚ := second_term / first_term -- this is 80

-- Define the n-th term of the geometric sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  first_term * (common_ratio ^ (n - 1))

-- Define the target divisor (one billion)
def target_divisor : ℤ := 10 ^ 9

-- Prove that the smallest n such that nth_term n is divisible by 10^9 is 9
theorem smallest_n_divisible_by_one_billion :
  ∃ n : ℕ, nth_term n = (first_term * (common_ratio ^ (n - 1))) ∧ 
           (target_divisor : ℚ) ∣ nth_term n ∧
           n = 9 :=
by sorry

end smallest_n_divisible_by_one_billion_l152_152389


namespace common_sum_of_5x5_square_is_zero_l152_152117

theorem common_sum_of_5x5_square_is_zero :
  ∃ (M : Matrix (Fin 5) (Fin 5) ℤ),
    (∀ i : Fin 5, ∑ j : Fin 5, M i j = 0) ∧
    (∀ j : Fin 5, ∑ i : Fin 5, M i j = 0)  ∧
    (∑ k : Fin 5, M k k = 0) ∧ 
    (∑ k : Fin 5, M k (Fin.mk (4 - k) sorry) = 0) :=
sorry

end common_sum_of_5x5_square_is_zero_l152_152117


namespace angle_in_regular_n_gon_l152_152053

theorem angle_in_regular_n_gon (n : ℕ) (n_gt_1 : n > 1) 
  (M : Type) (is_point_inside_polygon : Type) :
  ∃ (A B : Type), (1 - (1:ℝ)/n) * 180 ≤ ∠ A M B ∧ ∠ A M B ≤ 180 := 
sorry

end angle_in_regular_n_gon_l152_152053


namespace union_of_S_and_T_l152_152737

noncomputable theory

open Set

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | x^2 + 3 * x - 4 ≤ 0 }

theorem union_of_S_and_T :
  S ∪ T = { x | x ≥ -4 } :=
sorry

end union_of_S_and_T_l152_152737


namespace area_of_rectangle_l152_152354

-- Declare the conditions as variables in Lean
variables (P Q R S A B C D : Type)
variable [plane : geometric_plane P Q R S A B C D]

-- Declare side length of the small square
def side_length_small_square : Real := 1

-- Coordinates of points P, Q, R, S on the sides of the rectangle ABCD
def point_P : (Real × Real) := (x_P, y_P)
def point_Q : (Real × Real) := (x_Q, y_Q)
def point_R : (Real × Real) := (x_R, y_R)
def point_S : (Real × Real) := (x_S, y_S)

-- Area of the small square is 1 unit, hence side length is 1 unit
def small_square_area : Real := side_length_small_square * side_length_small_square

-- Coordinates of the rectangle side lengths
noncomputable def CS_length : Real := 2
noncomputable def CR_length : Real := Real.sqrt(5)

-- The main proposition to prove: the area of rectangle ABCD is 2√5 square units
theorem area_of_rectangle (CS_length = 2) (small_square_area = 1) : 
  ∃ l w : Real, l * w = 2 * Real.sqrt(5) := by
  sorry

end area_of_rectangle_l152_152354


namespace last_month_games_l152_152763

-- Definitions and conditions
def this_month := 9
def next_month := 7
def total_games := 24

-- Question to prove
theorem last_month_games : total_games - (this_month + next_month) = 8 := 
by 
  sorry

end last_month_games_l152_152763


namespace cos_squared_sin_pi_over_2_plus_alpha_l152_152261

variable (α : ℝ)

-- Given conditions
def cond1 : Prop := (Real.pi / 2) < α * Real.pi
def cond2 : Prop := Real.cos α = -3 / 5

-- Proof goal
theorem cos_squared_sin_pi_over_2_plus_alpha :
  cond1 α → cond2 α →
  (Real.cos (Real.sin (Real.pi / 2 + α)))^2 = 8 / 25 :=
by
  intro h1 h2
  sorry

end cos_squared_sin_pi_over_2_plus_alpha_l152_152261


namespace angle_proof_l152_152224

variable theta : ℝ

def complement (theta : ℝ) : ℝ := 90 - theta
def supplement (theta : ℝ) : ℝ := 180 - theta
def ten_percent (theta : ℝ) : ℝ := (10 / 100) * theta

theorem angle_proof : ten_percent (supplement (complement 35)) = 12.5 :=
by
  sorry

end angle_proof_l152_152224


namespace solve_fraction_eqn_l152_152830

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l152_152830


namespace same_number_of_acquaintances_greater_than_four_l152_152947

theorem same_number_of_acquaintances (n : ℕ) (h1 : n > 1)
  (h2 : ∀ (A B : fin n), A ≠ B → ∃! (C D : fin n), C ≠ D ∧ C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ C = D) :
  ∃ k, ∀ A, ∃ (S : set (fin n)), S.card = k ∧ ∀ B ∈ S, ∀ C ∈ S, B ≠ C → B ≠ A ∧ C ≠ A ∧ ∃! D, D ≠ A ∧ D ∈ S :=
sorry

theorem greater_than_four (n : ℕ) (h1 : n = 16)
  (h2 : ∀ (A B : fin n), A ≠ B → ∃! (C D : fin n), C ≠ D ∧ C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ C = D) :
  n > 4 :=
by {
  have h3 : n = 16 := h1,
  linarith,
  repeat {sorry}
}

end same_number_of_acquaintances_greater_than_four_l152_152947


namespace polynomial_integer_roots_l152_152238

theorem polynomial_integer_roots (a : ℝ) :
  (∀ r : ℤ, r ∈ Finset.univ.filter (λ r, (r : ℝ)^3 - 2 * (r : ℝ)^2 - 25 * (r : ℝ) + a = 0)) → a = 50 :=
by
  sorry

end polynomial_integer_roots_l152_152238


namespace product_to_difference_l152_152392

def x := 88 * 1.25
def y := 150 * 0.60
def z := 60 * 1.15

def product := x * y * z
def difference := x - y

theorem product_to_difference :
  product ^ difference = 683100 ^ 20 := 
sorry

end product_to_difference_l152_152392


namespace largest_common_value_less_than_1000_l152_152417

def arithmetic_sequence_1 (n : ℕ) : ℕ := 2 + 3 * n
def arithmetic_sequence_2 (m : ℕ) : ℕ := 4 + 8 * m

theorem largest_common_value_less_than_1000 :
  ∃ a n m : ℕ, a = arithmetic_sequence_1 n ∧ a = arithmetic_sequence_2 m ∧ a < 1000 ∧ a = 980 :=
by { sorry }

end largest_common_value_less_than_1000_l152_152417


namespace range_of_a_l152_152303

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (|x - 1| - |x - 3|) > a) → a < 2 :=
by
  sorry

end range_of_a_l152_152303


namespace exists_K_lt_s_l152_152946

variable {α : Type*}
variable [Add α] [OrderedAddCommMonoid α]

theorem exists_K_lt_s
  (A : ℕ → α)
  (n : ℕ)
  (not_collinear : ¬(∀ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k → collinear (A i) (A j) (A k)))
  (P Q : α)
  (distinct_PQ : P ≠ Q)
  (h : ∑ i in finset.range n, A i + P = ∑ i in finset.range n, A i + Q = s) :
  ∃ K, ∑ i in finset.range n, A i + K < s :=
sorry

end exists_K_lt_s_l152_152946


namespace correct_option_D_l152_152467

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end correct_option_D_l152_152467


namespace determine_prices_l152_152413

variable (num_items : ℕ) (cost_keychains cost_plush : ℕ) (x : ℚ) (unit_price_keychains unit_price_plush : ℚ)

noncomputable def price_equation (x : ℚ) : Prop :=
  (cost_keychains / x) + (cost_plush / (1.5 * x)) = num_items

theorem determine_prices 
  (h1 : num_items = 15)
  (h2 : cost_keychains = 240)
  (h3 : cost_plush = 180)
  (h4 : price_equation num_items cost_keychains cost_plush x)
  (hx : x = 24) :
  unit_price_keychains = 24 ∧ unit_price_plush = 36 :=
  by
    sorry

end determine_prices_l152_152413


namespace min_distance_not_sqrt2_div_2_area_not_greater_than_half_l152_152258

def is_on_curve (x y : ℝ) : Prop :=
  x^0.5 + y^0.5 = 1

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The minimum distance from any point on the curve M to the origin is not sqrt(2)/2 -/
theorem min_distance_not_sqrt2_div_2 :
  ¬ ∃ x y : ℝ, is_on_curve x y ∧ distance_from_origin x y = Real.sqrt 2 / 2 :=
sorry

/-- The area enclosed by the curve M and the coordinate axes is not greater than 1/2 -/
theorem area_not_greater_than_half :
  ∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → is_on_curve x y → 
    ∫ y in 0..1, (1 - x^0.5)^2 ≤ 1/2 :=
sorry

end min_distance_not_sqrt2_div_2_area_not_greater_than_half_l152_152258


namespace fourteen_sided_figure_area_l152_152509

-- Define the conditions
def unit_square_side : ℝ := 1
def full_unit_squares : ℕ := 11
def small_triangles : ℕ := 10

-- Define areas based on conditions
def area_of_unit_squares (n : ℕ) : ℝ := n * (unit_square_side ^ 2)
def area_of_small_triangles (n : ℕ) : ℝ := (n / 2) * (unit_square_side ^ 2)

-- Prove the total area
theorem fourteen_sided_figure_area :
  area_of_unit_squares full_unit_squares + area_of_small_triangles small_triangles = 16 :=
by
  sorry

end fourteen_sided_figure_area_l152_152509


namespace find_phi_l152_152268

theorem find_phi 
  (φ : ℝ) 
  (h₁ : φ ∈ set.Ioo (- (real.pi / 2)) (real.pi / 2))
  (h₂ : ∀ x : ℝ, sin (3 * x + φ) = sin (3 * (2 * (3 * real.pi / 5) - x) + φ)) :
  φ = - (3 * real.pi / 10) :=
sorry

end find_phi_l152_152268


namespace compound_proposition_true_l152_152823

def p : Prop := ∀ x : ℝ, x < 0 → 2^x > 3^x
def q : Prop := ∃ x : ℝ, 0 < x ∧ x < ∞ ∧ sqrt x > x^3

theorem compound_proposition_true : p ∧ q :=
by
  sorry

end compound_proposition_true_l152_152823


namespace arccos_cos_of_11_l152_152559

-- Define the initial conditions
def angle_in_radians (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 * Real.pi

def arccos_principal_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ Real.pi

-- Define the main theorem to be proved
theorem arccos_cos_of_11 :
  angle_in_radians 11 →
  arccos_principal_range (Real.arccos (Real.cos 11)) →
  Real.arccos (Real.cos 11) = 4.71682 :=
by
  -- Proof is not required
  sorry

end arccos_cos_of_11_l152_152559


namespace total_cost_of_antibiotics_l152_152943

-- Definitions based on the conditions
def cost_A_per_dose : ℝ := 3
def cost_B_per_dose : ℝ := 4.50
def doses_per_day_A : ℕ := 2
def days_A : ℕ := 3
def doses_per_day_B : ℕ := 1
def days_B : ℕ := 4

-- Total cost calculations
def total_cost_A : ℝ := days_A * doses_per_day_A * cost_A_per_dose
def total_cost_B : ℝ := days_B * doses_per_day_B * cost_B_per_dose

-- Final proof statement
theorem total_cost_of_antibiotics : total_cost_A + total_cost_B = 36 :=
by
  -- The proof goes here
  sorry

end total_cost_of_antibiotics_l152_152943


namespace triangle_area_eq_23_l152_152534

theorem triangle_area_eq_23 :
  let A := (2, 3)
  let B := (-1, -6)
  let C := (7, 2)
  let area := 1 / 2 * real.abs (2 * (-6 - 2) + (-1) * (2 - 3) + 7 * (3 + 6))
  area = 23 := 
by 
  sorry

end triangle_area_eq_23_l152_152534


namespace min_distance_not_sqrt2_div_2_area_not_greater_than_half_l152_152257

def is_on_curve (x y : ℝ) : Prop :=
  x^0.5 + y^0.5 = 1

def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The minimum distance from any point on the curve M to the origin is not sqrt(2)/2 -/
theorem min_distance_not_sqrt2_div_2 :
  ¬ ∃ x y : ℝ, is_on_curve x y ∧ distance_from_origin x y = Real.sqrt 2 / 2 :=
sorry

/-- The area enclosed by the curve M and the coordinate axes is not greater than 1/2 -/
theorem area_not_greater_than_half :
  ∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → is_on_curve x y → 
    ∫ y in 0..1, (1 - x^0.5)^2 ≤ 1/2 :=
sorry

end min_distance_not_sqrt2_div_2_area_not_greater_than_half_l152_152257


namespace jerry_water_usage_l152_152980

noncomputable def total_water_usage 
  (drinking_cooking : ℕ) 
  (shower_per_gallon : ℕ) 
  (length width height : ℕ) 
  (gallon_per_cubic_ft : ℕ) 
  (number_of_showers : ℕ) 
  : ℕ := 
   drinking_cooking + 
   (number_of_showers * shower_per_gallon) + 
   (length * width * height / gallon_per_cubic_ft)

theorem jerry_water_usage 
  (drinking_cooking : ℕ := 100)
  (shower_per_gallon : ℕ := 20)
  (length : ℕ := 10)
  (width : ℕ := 10)
  (height : ℕ := 6)
  (gallon_per_cubic_ft : ℕ := 1)
  (number_of_showers : ℕ := 15)
  : total_water_usage drinking_cooking shower_per_gallon length width height gallon_per_cubic_ft number_of_showers = 1400 := 
by
  sorry

end jerry_water_usage_l152_152980


namespace second_player_wins_l152_152808

def game (pos : ℕ) : Prop :=
  pos = 0 ∨ pos = 10 ∨ pos = 20 ∨ pos = 30 ∨ pos = 40 ∨ pos = 50 ∨ pos = 60 ∨ pos = 70 ∨ pos = 80 ∨ pos = 90 ∨ pos = 100

theorem second_player_wins : ∀ pos, game(0) -> ( ∀ (move : ℕ), move ≥ 1 ∧ move ≤ 9 → game(pos + move)) :=
by
  sorry

end second_player_wins_l152_152808


namespace circus_dogs_ratio_l152_152948

theorem circus_dogs_ratio :
  ∀ (x y : ℕ), 
  (x + y = 12) → (2 * x + 4 * y = 36) → (x = y) → x / y = 1 :=
by
  intros x y h1 h2 h3
  sorry

end circus_dogs_ratio_l152_152948


namespace perimeter_of_rectangle_l152_152104

-- Define the conditions
def area (l w : ℝ) : Prop := l * w = 180
def length_three_times_width (l w : ℝ) : Prop := l = 3 * w

-- Define the problem
theorem perimeter_of_rectangle (l w : ℝ) (h₁ : area l w) (h₂ : length_three_times_width l w) : 
  2 * (l + w) = 16 * Real.sqrt 15 := 
sorry

end perimeter_of_rectangle_l152_152104


namespace jack_evening_emails_l152_152724

theorem jack_evening_emails
  (emails_afternoon : ℕ := 3)
  (emails_morning : ℕ := 6)
  (emails_total : ℕ := 10) :
  emails_total - emails_afternoon - emails_morning = 1 :=
by
  sorry

end jack_evening_emails_l152_152724


namespace jerrys_breakfast_calories_l152_152042

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l152_152042


namespace moles_required_to_form_2_moles_H2O_l152_152988

def moles_of_NH4NO3_needed (moles_of_H2O : ℕ) : ℕ := moles_of_H2O

theorem moles_required_to_form_2_moles_H2O :
  moles_of_NH4NO3_needed 2 = 2 := 
by 
  -- From the balanced equation 1 mole of NH4NO3 produces 1 mole of H2O
  -- Therefore, 2 moles of NH4NO3 are needed to produce 2 moles of H2O
  sorry

end moles_required_to_form_2_moles_H2O_l152_152988


namespace cauchy_schwarz_inequality_l152_152400

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

end cauchy_schwarz_inequality_l152_152400


namespace Sara_spent_on_salad_l152_152088

theorem Sara_spent_on_salad (cost_of_hotdog : ℝ) (total_bill : ℝ) (cost_of_salad : ℝ) 
  (h_hotdog : cost_of_hotdog = 5.36) (h_total : total_bill = 10.46) : cost_of_salad = 10.46 - 5.36 :=
by
  rw [h_hotdog, h_total]
  rfl

end Sara_spent_on_salad_l152_152088


namespace y_n_sq_eq_3x_n_sq_plus_1_l152_152050

noncomputable def x : ℕ → ℝ
| 0        := 0
| 1        := 1
| (n + 1)  := 4 * x n - x (n - 1)

noncomputable def y : ℕ → ℝ
| 0        := 1
| 1        := 2
| (n + 1)  := 4 * y n - y (n - 1)

theorem y_n_sq_eq_3x_n_sq_plus_1 (n : ℕ) : y n ^ 2 = 3 * (x n ^ 2) + 1 := 
by
  induction n with n ih
  · simp [x, y]
  · sorry

end y_n_sq_eq_3x_n_sq_plus_1_l152_152050


namespace sum_of_all_three_digit_numbers_is_1998_l152_152611

open Finset

-- Define the digits set
def digits : Finset ℕ := {1, 3, 5}

-- Define the set of all three-digit numbers using the digits, ensuring all digits are distinct
def three_digit_numbers : Finset ℕ :=
  digits.image (λ x, digits.erase x).image (λ y, digits.erase y).image (λ z, 100 * x + 10 * y + z)

-- The sum of all unique three-digit numbers formed using 1, 3, and 5
def sum_of_three_digit_numbers := three_digit_numbers.sum id

-- Proof problem statement
theorem sum_of_all_three_digit_numbers_is_1998 : sum_of_three_digit_numbers = 1998 := by
  sorry

end sum_of_all_three_digit_numbers_is_1998_l152_152611


namespace sample_size_100_l152_152844

def inv_age_dist (n : ℕ) := n = 1000
def sampled_ages (n : ℕ) := n = 100

theorem sample_size_100 : ∀ (n1 n2 : ℕ), inv_age_dist n1 → sampled_ages n2 → n2 = 100 := by
  intros n1 n2 h1 h2
  rw [h2]
  sorry

end sample_size_100_l152_152844


namespace solve_for_b_l152_152325

theorem solve_for_b (x y b : ℝ) (h1: 4 * x + y = b) (h2: 3 * x + 4 * y = 3 * b) (hx: x = 3) : b = 39 :=
sorry

end solve_for_b_l152_152325


namespace find_M_l152_152941

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l152_152941


namespace last_digit_fib_mod_12_l152_152414

def fib_mod (n : ℕ) : ℕ :=
  let fib : ℕ → ℕ
  | 0      => 1
  | 1      => 1
  | (n+2)  => (fib n + fib (n+1)) % 12
  fib n

theorem last_digit_fib_mod_12 : (∃ n : ℕ, (∀ m : ℕ, m > n → fib_mod m ≠ 11) ∧ fib_mod n = 11) :=
  sorry

end last_digit_fib_mod_12_l152_152414


namespace average_growth_rate_l152_152014

theorem average_growth_rate (x : ℝ) :
  (7200 * (1 + x)^2 = 8712) → x = 0.10 :=
by
  sorry

end average_growth_rate_l152_152014


namespace equal_and_perpendicular_segments_l152_152278

variables (A B C A1 A2 B1 B2 : Type*)

-- Define the necessary structures and conditions
def is_perpendicular (l m : Type*) : Prop := sorry
def length_eq (x y : Type*) : Prop := sorry
def triangle (A B C : Type*) := sorry

-- Main Lean 4 statement
theorem equal_and_perpendicular_segments
  (T : triangle A B C)
  (hA1 : is_perpendicular (line_through A A1) (line_through B C))
  (hA2 : is_perpendicular (line_through A A2) (line_through B C))
  (hB1 : is_perpendicular (line_through B B1) (line_through A C))
  (hB2 : is_perpendicular (line_through B B2) (line_through A C))
  (hA1_len : length_eq (distance A A1) (distance B C))
  (hA2_len : length_eq (distance A A2) (distance B C))
  (hB1_len : length_eq (distance B B1) (distance A C))
  (hB2_len : length_eq (distance B B2) (distance A C)) :
  (distance A1 B2 = distance A2 B1) ∧ (is_perpendicular (line_through A1 B2) (line_through A2 B1)) :=
sorry

end equal_and_perpendicular_segments_l152_152278


namespace num_people_comparison_l152_152202

def num_people_1st_session (a : ℝ) : Prop := a > 0 -- Define the number for first session
def num_people_2nd_session (a : ℝ) : ℝ := 1.1 * a -- Define the number for second session
def num_people_3rd_session (a : ℝ) : ℝ := 0.99 * a -- Define the number for third session

theorem num_people_comparison (a b : ℝ) 
    (h1 : b = 0.99 * a): 
    a > b := 
by 
  -- insert the proof here
  sorry 

end num_people_comparison_l152_152202


namespace cos_eq_diff_pow_cos_sin_l152_152586

theorem cos_eq_diff_pow_cos_sin (n : ℕ) :
  (∀ x : ℝ, cos (2 * x) = cos x ^ n - sin x ^ n) → (n = 2 ∨ n = 4) :=
by
  sorry

end cos_eq_diff_pow_cos_sin_l152_152586


namespace erasers_total_l152_152760

-- Define the initial amount of erasers
def initialErasers : Float := 95.0

-- Define the amount of erasers Marie buys
def boughtErasers : Float := 42.0

-- Define the total number of erasers Marie ends with
def totalErasers : Float := 137.0

-- The theorem that needs to be proven
theorem erasers_total 
  (initial : Float := initialErasers)
  (bought : Float := boughtErasers)
  (total : Float := totalErasers) :
  initial + bought = total :=
sorry

end erasers_total_l152_152760


namespace pyramid_slant_height_l152_152343

noncomputable def slant_height (a r : ℝ) : ℝ :=
  \frac{2}{5} * (8 * real.sqrt 3 + real.sqrt 37) * r

theorem pyramid_slant_height {a r : ℝ} 
  (regular_pyramid : ∀ (a : ℝ), a > 0)
  (sphere_touches_all_faces : ∀ (r : ℝ), r > 0)
  (sphere_large_touches_base_lateral : ∀ (r : ℝ), 2 * r > 0)
  (spheres_touch : ∀ (r : ℝ), ∃ point, point ∈ sphere_touches_all_faces r ∧ point ∈ sphere_large_touches_base_lateral (2 * r)) :
  slant_height a r = \frac{2}{5} * (8 * real.sqrt 3 + real.sqrt 37) * r :=
begin
  sorry
end

end pyramid_slant_height_l152_152343


namespace three_buses_interval_l152_152457

theorem three_buses_interval 
  (T : ℕ) (interval_two_buses : T = 2 * 21):
  ∃ interval_three_buses, interval_three_buses = T / 3 ∧ interval_three_buses = 14 := 
by
  use 14
  rw interval_two_buses
  norm_num
  split
  norm_num
  refl
  sorry

end three_buses_interval_l152_152457


namespace fish_original_count_l152_152807

theorem fish_original_count (F : ℕ) (h : F / 2 - F / 6 = 12) : F = 36 := 
by 
  sorry

end fish_original_count_l152_152807


namespace find_k_l152_152665

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l152_152665


namespace proof_inequality_l152_152061

noncomputable def inequality (a b c : ℝ) (λ : ℝ) : Prop :=
  (sqrt (a^2 + λ * a * b + b^2) + sqrt (b^2 + λ * b * c + c^2) + sqrt (c^2 + λ * c * a + a^2))^2 
  ≥ (2 + λ) * (a + b + c)^2 + (2 - λ) * (a - b)^2

theorem proof_inequality (a b c : ℝ) (λ : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hl : -2 < λ) (hu : λ < 2) : 
  inequality a b c λ :=
by
  sorry

end proof_inequality_l152_152061


namespace points_on_parabola_l152_152604

-- Step 1: Define the conditions
def point (t : ℝ) : ℝ × ℝ :=
  let x := 3^t - 4
  let y := 9^t - 7 * 3^t + 6
  (x, y)

-- Step 2: State the theorem
theorem points_on_parabola (t : ℝ) :
  let (x, y) := point t in
  y = x^2 + 2 * x - 10 :=
by
  -- The proof is inserted here
  sorry

end points_on_parabola_l152_152604


namespace adjacent_diff_at_least_16_l152_152432

-- Define the grid dimensions and properties
def GridDim := 6

-- Define what adjacent means
def adjacent (i j k l : ℕ) : Prop :=
  (i = k ∧ abs (j - l) = 1) ∨ (j = l ∧ abs (i - k) = 1)

-- The main theorem to prove
theorem adjacent_diff_at_least_16 (grid : ℕ → ℕ → ℕ)
  (h1 : ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 36)
  (h2 : ∃ i j, grid i j = 1)
  (h3 : ∃ i j, grid i j = 2)
  (h4 : ∃ i j, grid i j = 3)
  (h5 : ∃ i j, grid i j = 4) : 
  ∃ i1 j1 i2 j2, adjacent i1 j1 i2 j2 ∧ abs (grid i1 j1 - grid i2 j2) ≥ 16 :=
sorry

end adjacent_diff_at_least_16_l152_152432


namespace common_point_of_functions_l152_152563

theorem common_point_of_functions (a b : ℝ) 
  (h : a + b = 2021) :
  ∃ x y : ℝ, x = 1 ∧ y = 2022 ∧ y = x^2 + a * x + b :=
by
  use (1, 2022)
  split
  sorry -- schould prove x = 1
  split
  sorry -- should prove y = 2022
  sorry -- should prove y = x^2 + a * x + b

end common_point_of_functions_l152_152563


namespace symmetry_axis_of_function_l152_152693

noncomputable def f (varphi : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + varphi)

theorem symmetry_axis_of_function
  (varphi : ℝ) (h1 : |varphi| < Real.pi / 2)
  (h2 : f varphi (Real.pi / 6) = 1) :
  ∃ k : ℤ, (k * Real.pi / 2 + Real.pi / 3 = Real.pi / 3) :=
sorry

end symmetry_axis_of_function_l152_152693


namespace profit_percentage_l152_152018

theorem profit_percentage (C S : ℝ) (h1 : C > 0) (h2 : S > 0)
  (h3 : S - 1.25 * C = 0.7023809523809523 * S) :
  ((S - C) / C) * 100 = 320 := by
sorry

end profit_percentage_l152_152018


namespace verify_identity_l152_152477

noncomputable def trigonometric_identity (α : ℝ) : Prop :=
  1 + real.cot α + real.csc α = (real.sqrt 2 * real.cos α) / (2 * real.sin (α / 2) * real.sin ((real.pi / 4) - (α / 2)))

theorem verify_identity (α : ℝ) (hα : α ≠ 0):
  trigonometric_identity α :=
sorry

end verify_identity_l152_152477


namespace g_at_4_l152_152412

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 5 / x
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8.142857 := by
  sorry

end g_at_4_l152_152412


namespace max_min_inequality_l152_152628

open Real

theorem max_min_inequality (n : ℕ) 
  (h_n : n ≥ 2) 
  (a : Fin n → ℝ)
  (h_a_pos : ∀ i, 0 < a i) : 
  (∑ i in Finset.range n, 
    (Finset.image (λ j : Fin i.succ, a j) Finset.univ).max' sorry 
    * (Finset.image (λ j : Fin ((n : ℕ) - i), a ⟨i + j, sorry⟩) Finset.univ).min' sorry) 
  ≤ (n / (2 * sqrt (n - 1))) * (∑ i in Finset.range n, (a i)^2) := 
sorry

end max_min_inequality_l152_152628


namespace right_triangle_hypotenuse_l152_152523

theorem right_triangle_hypotenuse :
  ∃ (x y : ℝ), (1 / 3 * Real.pi * y^2 * x = 1350 * Real.pi) ∧
               (1 / 3 * Real.pi * x^2 * y = 2430 * Real.pi) ∧
               (x^2 + y^2 = 954) :=
by { use [27, 15], split, split, repeat { norm_num }, sorry }

end right_triangle_hypotenuse_l152_152523


namespace train_crosses_signal_pole_l152_152880

theorem train_crosses_signal_pole
  (lt : ℝ) (lp : ℝ) (tp : ℝ)
  (Hlt : lt = 425) (Hlp : lp = 159.375) (Htp : tp = 55) :
  let dp := lt + lp in
  let v := dp / tp in
  let ts := lt / v in
  ts = 40 := by
  sorry

end train_crosses_signal_pole_l152_152880


namespace max_value_of_abs_z_l152_152009

noncomputable def max_value_abs_z : ℂ → ℝ
| z := complex.abs z

theorem max_value_of_abs_z (z : ℂ) (h : complex.abs (z - 2 * complex.I) = 1) : max_value_abs_z z ≤ 3 :=
by sorry

end max_value_of_abs_z_l152_152009


namespace min_moves_to_alternating_pattern_l152_152893

theorem min_moves_to_alternating_pattern (initial_coins : list bool) (moves : list (ℕ × ℕ)) :
  initial_coins.length = 7 ∧
  initial_coins.all (λ c, c = tt) ∧
  (∀ (i j : ℕ), (i, j) ∈ moves → |i - j| = 1) ∧
  (∀ (i j : ℕ), (i, j) ∈ moves → i < initial_coins.length ∧ j < initial_coins.length) →
  let final_coins := moves.foldl (λ cs (i, j), cs.set_i i (¬ cs.nth_i i) |>.set_i j (¬ cs.nth_i j)) initial_coins in
  final_coins.length = 7 ∧
  (∀ (k : ℕ), k < final_coins.length - 1 → final_coins.nth_i k ≠ final_coins.nth_i (k + 1)) ∧
  moves.length = 4 :=
sorry

end min_moves_to_alternating_pattern_l152_152893


namespace range_of_a_l152_152695

noncomputable def f (x a : ℝ) : ℝ := x^2 - abs (x - 1) - a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, -3 <= x -> x < 3 -> x^2 < abs(x - 1) + a) :
  a ≤ 5 :=
by
  sorry

end range_of_a_l152_152695


namespace quadratic_root_count_impossible_l152_152431

noncomputable def quadratic (f : ℝ → ℝ) : Prop :=
∃ (a b c : ℝ), (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c)

noncomputable def has_2_distinct_roots (f : ℝ → ℝ) : Prop :=
∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ∀ x, f x = (x - r1) * (x - r2)

noncomputable def has_3_distinct_roots (f : ℝ → ℝ) : Prop :=
∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ x, f x = 0 → x = x1 ∨ x = x2 ∨ x = x3

noncomputable def has_7_distinct_roots (f : ℝ → ℝ) : Prop :=
∃ x1 x2 x3 x4 x5 x6 x7 : ℝ, list.nodup [x1, x2, x3, x4, x5, x6, x7] ∧ ∀ x, f x = 0 → list.mem x [x1, x2, x3, x4, x5, x6, x7]

theorem quadratic_root_count_impossible (f : ℝ → ℝ) :
  quadratic f ∧ has_2_distinct_roots f ∧ has_3_distinct_roots (λ x, f (f x)) →
  ¬ (has_7_distinct_roots (λ x, f (f (f x)))) :=
sorry

end quadratic_root_count_impossible_l152_152431


namespace range_of_a_for_function_is_real_l152_152324

noncomputable def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 4 * x + a - 3

theorem range_of_a_for_function_is_real :
  (∀ x : ℝ, quadratic_expr a x > 0) → 0 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_for_function_is_real_l152_152324


namespace length_CF_of_intersecting_circles_l152_152458

theorem length_CF_of_intersecting_circles :
  ∀ (R : ℝ) (A B C D F : ℝ × ℝ),
    R = 9 →
    dist A B = 0 → -- Points A and B intersection points
    dist B (C.1, C.2) = R →
    dist B (D.1, D.2) = R →
    ∠ CAD = 90 → -- CAD is a right angle
    B ∈ segment C D → -- B lies on segment CD
    dist B F = dist B D → -- BF = BD
    CF = 2 * R :=
begin
  sorry
end

end length_CF_of_intersecting_circles_l152_152458


namespace find_distance_C_A_l152_152222

theorem find_distance_C_A :
  ∃ x : ℝ, (C_recieves_equal_gas :
  (10000 - 4 * x) = (11200 - 4 * (500 - x))) ∧
  (distance_A_B : 500 = 500) ∧
  (gas_from_A : 10000 = 10000) ∧
  (gas_from_B : 11200 = 10000 * 1.12) ∧
  (leakage : 4 = 4) ∧
  x = 100 := by
  sorry

end find_distance_C_A_l152_152222


namespace area_sin_l152_152179

theorem area_sin : ∫ x in -Real.pi/3..0, Real.sin x = 1 / 2 := sorry

end area_sin_l152_152179


namespace initial_amounts_unique_l152_152487

noncomputable def initialAmountsSatisfyConditions (x y z : ℕ) : Prop :=
  let A1 := x + y / 3 + z / 3 in
  let B1 := 2 * y / 3 in
  let C1 := 2 * z / 3 in
  let A2 := 2 * (x + y / 3 + z / 3) / 3 in
  let B2 := (2 * y / 3) + (2 * z / 3) / 3 in
  let C2 := 4 * z / 9 in
  let A3 := 4 * (x + y / 3 + z / 3) / 27 in
  let B3 := 4 * y / 9 + 8 * z / 27 in
  let C3 := 8 * z / 27 + 2 * (2 * y / 9) / 3 + 8 * (4 * z / 81) / 3 in
  x + y + z < 1000 ∧
  (x - A3 = 2) ∧
  (C3 - z = 2 * z + 8)

theorem initial_amounts_unique :
  ∃ x y z, initialAmountsSatisfyConditions x y z ∧ x = 54 ∧ y = 162 ∧ z = 27 :=
by {
  -- Proof steps would go here
  sorry
}

end initial_amounts_unique_l152_152487


namespace rectangle_length_l152_152437

-- Let \( s \) be the side of the square
variable (s l : ℝ)

-- Definitions and conditions from the problem
def perimeter_square := 4 * s
def perimeter_rectangle := 2 * (l + 14)
def circumference_semicircle := (Real.pi * s / 2) + s

-- Hypotheses
hypothesis (h1 : 4 * s = 2 * (l + 14))
hypothesis (h2 : (Real.pi * s / 2) + s = 25.13)

-- Theorem to prove the length of the rectangle
theorem rectangle_length : l = 5.54 :=
sorry

end rectangle_length_l152_152437


namespace grandma_mushrooms_l152_152949

theorem grandma_mushrooms (M : ℕ) (h₁ : ∀ t : ℕ, t = 2 * M)
                         (h₂ : ∀ p : ℕ, p = 4 * t)
                         (h₃ : ∀ b : ℕ, b = 4 * p)
                         (h₄ : ∀ r : ℕ, r = b / 3)
                         (h₅ : r = 32) :
  M = 3 :=
by
  -- We are expected to fill the steps here to provide the proof if required
  sorry

end grandma_mushrooms_l152_152949


namespace cars_without_air_conditioning_l152_152015

/-
In a group of 100 cars, some cars do not have air conditioning. If at least 41 cars have racing stripes, the greatest number of cars that could have air conditioning but not racing stripes is 59. Prove that the number of cars that do not have air conditioning is 41.
-/

theorem cars_without_air_conditioning 
  (total_cars : ℕ) 
  (racing_stripes_min : ℕ) 
  (max_air_not_racing : ℕ)
  (h1 : total_cars = 100)
  (h2 : racing_stripes_min = 41)
  (h3 : max_air_not_racing = 59)
  : total_cars - (max_air_not_racing) = 41 :=
by
  rw [h1, h2, h3]
  exact rfl

#check cars_without_air_conditioning

end cars_without_air_conditioning_l152_152015


namespace tan_product_in_triangle_l152_152032

theorem tan_product_in_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = Real.sin B ^ 2) :
  Real.tan A * Real.tan C = 1 :=
sorry

end tan_product_in_triangle_l152_152032


namespace taxi_ride_cost_l152_152205

namespace TaxiFare

def baseFare : ℝ := 2.00
def costPerMile : ℝ := 0.30
def taxRate : ℝ := 0.10
def distance : ℝ := 8.0

theorem taxi_ride_cost :
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  total_fare = 4.84 := by
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  sorry

end TaxiFare

end taxi_ride_cost_l152_152205


namespace inequality_solution_set_l152_152446

theorem inequality_solution_set (x : ℝ) :
  ((1 - x) * (x - 3) < 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l152_152446


namespace compute_permutation_eq_4_l152_152960

def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem compute_permutation_eq_4 :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 4 :=
by
  sorry

end compute_permutation_eq_4_l152_152960


namespace area_of_unpainted_section_l152_152140

-- Define the conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def cross_angle : ℝ := 45
def negligible_holes : Prop := true

-- The main statement
theorem area_of_unpainted_section (h1 : board1_width = 5) (h2 : board2_width = 7) (h3 : cross_angle = 45) (h4 : negligible_holes) : 
  ∃ (area : ℝ), area = 35 := 
sorry

end area_of_unpainted_section_l152_152140


namespace star_value_l152_152228

-- Define the operation * as described in the problem
def star (a b : ℚ) := (a.num * b.num) * (2 * b.denom) / a.denom

-- Construct the problem statement
theorem star_value :
  star (6 / 5) (3 / 4) = (144 / 5) := by
  sorry

end star_value_l152_152228


namespace probability_two_red_books_l152_152483

theorem probability_two_red_books (total_books red_books blue_books selected_books : ℕ)
  (h_total: total_books = 8)
  (h_red: red_books = 4)
  (h_blue: blue_books = 4)
  (h_selected: selected_books = 2) :
  (Nat.choose red_books selected_books : ℚ) / (Nat.choose total_books selected_books) = 3 / 14 := by
  sorry

end probability_two_red_books_l152_152483


namespace plane_perpendicular_conditions_l152_152310

theorem plane_perpendicular_conditions
  (l m : Line)
  (alpha beta : Plane)
  (h1 : l ⊥ alpha)
  (h2 : m ⊥ beta)
  (h3 : l ⊥ m) :
  alpha ⊥ beta :=
sorry

end plane_perpendicular_conditions_l152_152310


namespace imaginary_part_of_complex_l152_152985

open Complex

/-- The imaginary part of (3 + i) / i^2 * i is 1 -/
theorem imaginary_part_of_complex :
  let i : ℂ := Complex.I in
  complex.im ((3 + i) / (i^2) * i) = 1 :=
by
  sorry

end imaginary_part_of_complex_l152_152985


namespace find_triples_l152_152245

def sign (a : ℝ) : ℝ :=
if a > 0 then 1 else if a < 0 then -1 else 0

theorem find_triples :
  (finset.card (finset.filter 
    (λ t : ℝ × ℝ × ℝ, 
       let (x, y, z) := t in 
           x = 2023 - 2024 * sign (y - z) ∧ 
           y = 2023 - 2024 * sign (z - x) ∧ 
           z = 2023 - 2024 * sign (x - y))
     ((finset.fin_range 4048).product ((finset.fin_range 4048).product (finset.fin_range 4048)))) = 3) :=
sorry

end find_triples_l152_152245


namespace length_AC_l152_152327

theorem length_AC (A B C : Type)
  (angle_A : A)
  (angle_B : B)
  (BC : C)
  (hA : angle_A = 60)
  (hB : angle_B = 45)
  (hBC : BC = 3 * Real.sqrt 2) :
  ∃ AC : C, AC = 2 * Real.sqrt 3 :=
by
  sorry

end length_AC_l152_152327


namespace abs_diff_roots_eq_3_l152_152589

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l152_152589


namespace tree_leaves_remaining_after_three_weeks_l152_152919

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end tree_leaves_remaining_after_three_weeks_l152_152919


namespace average_ABC_eq_2A_plus_3_l152_152561

theorem average_ABC_eq_2A_plus_3 (A B C : ℝ) 
  (h1 : 2023 * C - 4046 * A = 8092) 
  (h2 : 2023 * B - 6069 * A = 10115) : 
  (A + B + C) / 3 = 2 * A + 3 :=
sorry

end average_ABC_eq_2A_plus_3_l152_152561


namespace garden_area_l152_152908

/-- A rectangular garden is 350 cm long and 50 cm wide. Determine its area in square meters. -/
theorem garden_area (length_cm width_cm : ℝ) (h_length : length_cm = 350) (h_width : width_cm = 50) : (length_cm / 100) * (width_cm / 100) = 1.75 :=
by
  sorry

end garden_area_l152_152908


namespace number_of_other_numbers_l152_152106

-- Definitions of the conditions
def avg_five_numbers (S : ℕ) : Prop := S / 5 = 20
def sum_three_numbers (S2 : ℕ) : Prop := 100 = S2 + 48
def avg_other_numbers (N S2 : ℕ) : Prop := S2 / N = 26

-- Theorem statement
theorem number_of_other_numbers (S S2 N : ℕ) 
  (h1 : avg_five_numbers S) 
  (h2 : sum_three_numbers S2) 
  (h3 : avg_other_numbers N S2) : 
  N = 2 := 
  sorry

end number_of_other_numbers_l152_152106


namespace simultaneous_solution_exists_l152_152608

-- Definitions required by the problem
def eqn1 (m x : ℝ) : ℝ := m * x + 2
def eqn2 (m x : ℝ) : ℝ := (3 * m - 2) * x + 5

-- Proof statement
theorem simultaneous_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = eqn1 m x ∧ y = eqn2 m x) ↔ (m ≠ 1) := 
sorry

end simultaneous_solution_exists_l152_152608


namespace complex_expr_simplify_l152_152780

theorem complex_expr_simplify :
  (let i := Complex.I in
   ((1 + i) / 2) ^ 8 + ((1 - i) / 2) ^ 8 = 1 / 8) :=
by
  sorry

end complex_expr_simplify_l152_152780


namespace binomial_parameters_unique_l152_152295

theorem binomial_parameters_unique (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ) 
  (h₁ : ξ ~ Binomial n p) 
  (h₂ : E ξ = 1.6) 
  (h₃ : Var ξ = 1.28) 
: n = 8 ∧ p = 0.2 := 
sorry

end binomial_parameters_unique_l152_152295


namespace max_non_overlapping_pairs_l152_152999

theorem max_non_overlapping_pairs :
  ∀ (k : ℕ), (k ≤ 3009 / 2) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 1 ≤ a i ∧ a i < b i ≤ 3009) ∧
  (∀ i j : ℕ, 1 ≤ i ∧ i < j ≤ k → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ k → a i + b i ≤ 3010) →
  ∃ m : ℕ, m = 1203 :=
  sorry

end max_non_overlapping_pairs_l152_152999


namespace interval_for_rollers_l152_152937

noncomputable def interval_contains_probability (a σ : ℝ) (p : ℝ) : Prop :=
  ∃ δ : ℝ, 2 * CDF (Normal a σ) δ - 1 = p

theorem interval_for_rollers 
  (a : ℝ) (σ : ℝ) (p : ℝ) (hl: a = 10) (hs: σ = 0.1) (hp: p = 0.9973):
  interval_contains_probability a σ p → (9.7 < a ∧ a < 10.3) :=
sorry

end interval_for_rollers_l152_152937


namespace eval_expr_l152_152063

namespace ProofProblem

variables (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d = a + b + c)

theorem eval_expr :
  d = a + b + c →
  (a^3 + b^3 + c^3 - 3 * a * b * c) / (a * b * c) = (d * (a^2 + b^2 + c^2 - a * b - a * c - b * c)) / (a * b * c) :=
by
  intros hd
  sorry

end ProofProblem

end eval_expr_l152_152063


namespace math_problem_l152_152471

theorem math_problem 
    (A_correct : (let l := (λ x y : ℝ, x - y - 2 = 0) in ∃ (x1 x2 : ℝ), l x1 0 ∧ l 0 x2 ∧ (x1 * x2) / 2 = 2))
    (B_correct : (let symmetric_point := (0, 2) in ∃ (sym_pt : ℝ × ℝ), (sym_pt = (1, 1)) ∧ (∀ (x y : ℝ), y = x + 1 → symmetric_point.2 = symmetric_point.1 + 1))) :
  (A_correct ∧ B_correct) :=
by
  -- proof placeholder
  sorry

end math_problem_l152_152471


namespace prob_before_fifth_ring_prob_not_answered_four_rings_l152_152529

noncomputable def P : ℕ → ℝ
| 1 := 0.1
| 2 := 0.2
| 3 := 0.3
| 4 := 0.35
| _ := 0

def P_before_fifth_ring : ℝ :=
P 1 + P 2 + P 3 + P 4

theorem prob_before_fifth_ring : P_before_fifth_ring = 0.95 :=
by 
  unfold P_before_fifth_ring,
  rw [P, P, P, P],
  norm_num,
  sorry -- This would be replaced by actual proof

theorem prob_not_answered_four_rings : 1 - P_before_fifth_ring = 0.05 :=
by 
  unfold P_before_fifth_ring,
  rw [P, P, P, P],
  norm_num,
  sorry -- This would be replaced by actual proof

end prob_before_fifth_ring_prob_not_answered_four_rings_l152_152529


namespace four_digit_numbers_count_l152_152122

theorem four_digit_numbers_count : 
  let digits := [1, 2, 3, 4, 5]
  let four_digit_permutations := permutations digits 4
  ∃ ps, (ps ∈ four_digit_permutations) ∧ (¬((2, 5) ∈ adjacent_in ps) ∧ ¬((5, 2) ∈ adjacent_in ps)) → 
  (count such ps) = 84 :=
sorry

end four_digit_numbers_count_l152_152122


namespace find_ordered_pair_l152_152990

theorem find_ordered_pair:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 10 * m * n = 45 - 5 * m - 3 * n ∧ (m, n) = (1, 11) :=
by
  sorry

end find_ordered_pair_l152_152990


namespace count_isosceles_or_equilateral_triangles_l152_152570

structure Point :=
(x : ℕ)
(y : ℕ)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

structure Triangle :=
(p1 : Point)
(p2 : Point)
(p3 : Point)

def is_isosceles_or_equilateral (t : Triangle) : Prop :=
  let d1 := distance t.p1 t.p2
  let d2 := distance t.p2 t.p3
  let d3 := distance t.p3 t.p1
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3 ∨ (d1 = d2 ∧ d2 = d3)

def triangles : list Triangle :=
  [ { p1 := { x := 0, y := 5 }, p2 := { x := 5, y := 5 }, p3 := { x := 2, y := 2 } },
    { p1 := { x := 1, y := 1 }, p2 := { x := 4, y := 1 }, p3 := { x := 4, y := 4 } },
    { p1 := { x := 1, y := 4 }, p2 := { x := 3, y := 5 }, p3 := { x := 5, y := 4 } },
    { p1 := { x := 1, y := 0 }, p2 := { x := 3, y := 2 }, p3 := { x := 5, y := 0 } },
    { p1 := { x := 0, y := 2 }, p2 := { x := 2, y := 5 }, p3 := { x := 4, y := 2 } },
    { p1 := { x := 2, y := 1 }, p2 := { x := 3, y := 3 }, p3 := { x := 4, y := 1 } } ]

theorem count_isosceles_or_equilateral_triangles : 
  list.countp is_isosceles_or_equilateral triangles = 6 :=
by
  sorry

end count_isosceles_or_equilateral_triangles_l152_152570


namespace sum_of_m_n_of_solutions_l152_152134

noncomputable theory
open Real

theorem sum_of_m_n_of_solutions :
  ∃ (s : ℕ) (f : Fin s → ℕ × ℕ),
    (∀ k, gcd (f k).1 (f k).2 = 1) ∧
    (∃ (x : Fin s → ℝ),
      (∀ k, 0 < x k ∧ x k < π / 2 ∧ 64 * sin(2 * x k) ^ 2 + (tan(x k)) ^ 2 + (cot(x k)) ^ 2 = 46) ∧
      (∀ k, ∃ (mk nk : ℕ), mk * π / nk = x k ∧ mk + nk = (f k).1 + (f k).2)) ∧
    (∑ k, (f k).1 + (f k).2) = 100 := sorry

end sum_of_m_n_of_solutions_l152_152134


namespace nth_equation_l152_152396

theorem nth_equation (n : ℕ) (h : 0 < n) : (10 * n + 5) ^ 2 = n * (n + 1) * 100 + 5 ^ 2 := 
sorry

end nth_equation_l152_152396


namespace first_train_length_correct_l152_152460

noncomputable def length_of_first_train
  (speed1_kmph : ℕ) (speed2_kmph : ℕ) (length_second_train_m : ℕ) (time_to_clear_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600) in
  let total_distance_m := relative_speed_mps * time_to_clear_s in
  total_distance_m - length_second_train_m

theorem first_train_length_correct :
  length_of_first_train 80 65 165 7.447680047665153 = 135 :=
by
  -- Here, we would insert the proof steps
  sorry

end first_train_length_correct_l152_152460


namespace compute_expression_l152_152560

theorem compute_expression : 2 + ((4 * 3 - 2) / 2 * 3) + 5 = 22 :=
by
  -- Place the solution steps if needed
  sorry

end compute_expression_l152_152560


namespace allocation_problem_l152_152887

-- Definitions based on conditions
variable (employees : Fin 5) (departments : Fin 3)

noncomputable def allocation_methods_count : Nat := 36 -- Correct answer from the problem

-- Statement to prove
theorem allocation_problem :
  ∃ (f : employees → departments),
    (∀ d : departments, ∃ e : employees, f e = d) ∧
    (∃ d : departments, ∀ e : {e : employees // e < 2}, f e.1 = d) →
    allocation_methods_count = 36 :=
by
  sorry

end allocation_problem_l152_152887


namespace percentage_less_than_l152_152511

theorem percentage_less_than (T F S : ℝ) 
  (hF : F = 0.70 * T) 
  (hS : S = 0.63 * T) : 
  ((T - S) / T) * 100 = 37 := 
by
  sorry

end percentage_less_than_l152_152511


namespace commute_time_l152_152868

theorem commute_time (d w t : ℝ) (x : ℝ) (h_distance : d = 1.5) (h_walking_speed : w = 3) (h_train_speed : t = 20)
  (h_extra_time : 30 = 4.5 + x + 2) : x = 25.5 :=
by {
  -- Add the statement of the proof
  sorry
}

end commute_time_l152_152868


namespace number_of_triangles_with_all_sides_being_positive_integers_and_longest_side_11_is_36_l152_152123

noncomputable def number_of_triangles_with_longest_side_11 : ℕ :=
  (finset.range 11).sum (λ x, x + 1)

theorem number_of_triangles_with_all_sides_being_positive_integers_and_longest_side_11_is_36 :
  number_of_triangles_with_longest_side_11 = 36 := by
  sorry

end number_of_triangles_with_all_sides_being_positive_integers_and_longest_side_11_is_36_l152_152123


namespace length_PR_eq_20_l152_152616

theorem length_PR_eq_20 (x y : ℝ) (h : x^2 + y^2 = 200) : sqrt (2 * (x^2 + y^2)) = 20 :=
by
  sorry

end length_PR_eq_20_l152_152616


namespace three_digit_numbers_with_at_least_one_3_and_one_4_l152_152683

theorem three_digit_numbers_with_at_least_one_3_and_one_4 :
  ∃ n : ℕ, n = 48 ∧ ∀ x, (100 ≤ x ∧ x < 1000) → 
  (x.to_digits 10).contains 3 → (x.to_digits 10).contains 4 :=
sorry

end three_digit_numbers_with_at_least_one_3_and_one_4_l152_152683


namespace cube_sphere_theorem_l152_152359

def cube_sphere_problem : Prop :=
  let O_od1 := 17 in 
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ),
    -- Center of the sphere
    O = (0, 0, 0) →
    -- Radius of the sphere
    r = 10 →
    -- Sphere intersects cube faces in given circles
    ∃ (rad1 rad2 rad3 : ℝ), 
      rad1 = 1 ∧ -- Intersection radii for face AA₁D₁D
      rad2 = 1 ∧ -- Intersection radii for face A₁B₁C₁D₁
      rad3 = 3 ∧ -- Intersection radii for face CDD₁C₁
    -- OD₁ is the 3D-distance from O to D₁ (17 by the conditions)
    (O_od1 = real.sqrt (91 + 99 + 99))

theorem cube_sphere_theorem : cube_sphere_problem :=
by 
  intros O r hO hr,
  use [1, 1, 3],
  split; [refl, split; [refl, split; [refl, sorry]]]

end cube_sphere_theorem_l152_152359


namespace smallest_x_value_l152_152233

theorem smallest_x_value : ∃ x : ℝ, 3 * x^2 + 36 * x - 72 = x * (x + 20) + 8 ∧ ∀ y : ℝ, 3 * y^2 + 36 * y - 72 = y * (y + 20) + 8 → x ≤ y :=
begin
  let x := -10,
  use x,
  split,
  { sorry },
  { intros y hy,
    sorry }
end

end smallest_x_value_l152_152233


namespace sufficient_but_not_necessary_l152_152162

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (∃ a, a > 0 → |a| > 0) ∧ (∃ a, |a| > 0 ∧ a < 0) :=
by
  sorry

end sufficient_but_not_necessary_l152_152162


namespace problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l152_152005

variable (a b : ℝ)

theorem problem_statement_part1 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a + 2 / b) ≥ 9 := sorry

theorem problem_statement_part2 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (2 ^ a + 4 ^ b) ≥ 2 * Real.sqrt 2 := sorry

theorem problem_statement_part3 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a * b) ≤ (1 / 8) := sorry

theorem problem_statement_part4 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2) ≥ (1 / 5) := sorry

end problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l152_152005


namespace unique_solution_linear_system_l152_152798

theorem unique_solution_linear_system
  (a11 a22 a33 : ℝ) (a12 a13 a21 a23 a31 a32 : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0) (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) →
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) →
  (a31 * x1 + a32 * x2 + a33 * x3 = 0) →
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  sorry

end unique_solution_linear_system_l152_152798


namespace Annabelle_saved_12_dollars_l152_152546

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_l152_152546


namespace figures_are_not_similar_l152_152355

variables {Pyramid : Type} [lateral_face : Pyramid → Set ℝ]
variables {F : Set ℝ} {base : Set ℝ} {α : ℝ}

-- Placeholder for the projection operations
def projection_onto_base (F : Set ℝ) : Set ℝ := sorry
def projection_onto_lateral_face (Phi1 : Set ℝ) : Set ℝ := sorry

-- Initial conditions satisfied
variable (F_Fshaped_on_lateral_face : F → lateral_face)

theorem figures_are_not_similar :
  let Φ1 := projection_onto_base F in
  let Φ2 := projection_onto_lateral_face (projection_onto_base F) in
  ¬ similar F Φ2 :=
by sorry

end figures_are_not_similar_l152_152355


namespace vegetarian_people_count_l152_152340

/-
In a family of 40 people, 
16 people eat only vegetarian, 
12 people eat only non-vegetarian, 
8 people eat both vegetarian and non-vegetarian, 
3 people are pescatarians who eat fish but not meat, 
and 1 person is a vegan who eats neither animal products nor by-products.

How many people in the family eat vegetarian food?
-/

theorem vegetarian_people_count 
  (total_members : ℕ) (only_vegetarian : ℕ) (only_non_vegetarian : ℕ) 
  (both_vegetarian_non_vegetarian : ℕ) (pescatarians : ℕ) (vegan : ℕ) :
  total_members = 40 ∧ only_vegetarian = 16 ∧ only_non_vegetarian = 12 ∧
  both_vegetarian_non_vegetarian = 8 ∧ pescatarians = 3 ∧ vegan = 1 →
  (only_vegetarian + both_vegetarian_non_vegetarian + vegan = 25) :=
by
  intros,
  sorry

end vegetarian_people_count_l152_152340


namespace work_finished_earlier_due_to_additional_men_l152_152538

-- Define the conditions as given facts in Lean
def original_men := 10
def original_days := 12
def additional_men := 10

-- State the theorem to be proved
theorem work_finished_earlier_due_to_additional_men :
  let total_men := original_men + additional_men
  let original_work := original_men * original_days
  let days_earlier := original_days - x
  original_work = total_men * days_earlier → x = 6 :=
by
  sorry

end work_finished_earlier_due_to_additional_men_l152_152538


namespace maximum_value_of_expression_l152_152062

noncomputable def max_function_value (x y z : ℝ) : ℝ := 
  (x^3 - x * y^2 + y^3) * (x^3 - x * z^2 + z^3) * (y^3 - y * z^2 + z^3)

theorem maximum_value_of_expression : 
  ∃ x y z : ℝ, (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 3) 
  ∧ max_function_value x y z = 2916 / 2187 := 
sorry

end maximum_value_of_expression_l152_152062


namespace cartesian_curve_equation_range_PA_PB_l152_152712

/-- Define the parametric line -/
def parametric_line (α t : ℝ) : ℝ × ℝ :=
  (1 + t * (Real.cos α), 1/2 + t * (Real.sin α))

/-- Define the polar equation of the curve -/
def polar_equation (θ : ℝ) : ℝ :=
  Real.sqrt (12 / (4 * (Real.sin θ)^2 + 3 * (Real.cos θ)^2))

/-- The Cartesian coordinate equation of the curve C -/
theorem cartesian_curve_equation :
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 :=
sorry

/-- The range of |PA| • |PB| -/
theorem range_PA_PB (α : ℝ) :
  ∀ (t : ℝ), (2 : ℝ) ≤ abs (t) * abs ((-8) / (3 + (Real.sin α)^2))  :=
sorry

end cartesian_curve_equation_range_PA_PB_l152_152712


namespace total_time_escalators_then_walkway_l152_152938

theorem total_time_escalators_then_walkway
  (speed_escalator₁ : ℝ) (length_escalator₁ : ℝ) (walking_speed_escalator₁ : ℝ)
  (speed_walkway₂ : ℝ) (length_walkway₂ : ℝ) (walking_speed_walkway₂ : ℝ)
  (h₀ : speed_escalator₁ = 10) 
  (h₁ : length_escalator₁ = 112) 
  (h₂ : walking_speed_escalator₁ = 4) 
  (h₃ : speed_walkway₂ = 6) 
  (h₄ : length_walkway₂ = 80) 
  (h₅ : walking_speed_walkway₂ = 3) :
  let combined_speed_escalator₁ := speed_escalator₁ + walking_speed_escalator₁,
      combined_speed_walkway₂ := speed_walkway₂ + walking_speed_walkway₂,
      time_on_escalator₁ := length_escalator₁ / combined_speed_escalator₁,
      time_on_walkway₂ := length_walkway₂ / combined_speed_walkway₂,
      total_time := time_on_escalator₁ + time_on_walkway₂
  in total_time ≈ 16.89 :=
by
  sorry

end total_time_escalators_then_walkway_l152_152938


namespace fixed_point_exists_l152_152073

theorem fixed_point_exists : ∀ (m : ℝ), (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  intro m
  have h : (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
    sorry
  exact h

end fixed_point_exists_l152_152073


namespace number_of_girls_l152_152452

theorem number_of_girls (G : ℕ) (h1 : 5.choose 2 * G.choose 2 = 150) : G = 6 :=
by
  -- Proof omitted
  sorry

end number_of_girls_l152_152452


namespace number_of_solutions_eq_l152_152244

theorem number_of_solutions_eq :
  ∃ n : ℕ, n = 10 ∧ ∀ x : ℝ, -real.pi ≤ x ∧ x ≤ real.pi → 
    cos (6 * x) + 2 * cos (4 * x) ^ 2 + 3 * cos (2 * x) ^ 3 + sin (x) ^ 2 = 0 → 
    -- Counting unique x values that satisfy the above equation.
    sorry

end number_of_solutions_eq_l152_152244


namespace tiling_fraction_black_l152_152132

theorem tiling_fraction_black (s : ℝ) (floor_area : ℝ) (h1 : 0 < s) (h2 : 0 < floor_area) :
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2,
      triangle_area := (Real.sqrt 3 / 16) * s^2,
      total_tiles_count := floor_area / (hexagon_area + 2 * triangle_area),
      black_fraction := 2 * triangle_area / (hexagon_area + 2 * triangle_area) in
  abs (black_fraction - 1 / 13) < 1e-10 :=
by
  sorry

end tiling_fraction_black_l152_152132


namespace proof_moved_out_and_remaining_families_l152_152703

-- Define the total families for each animal type
def deer_families : ℕ := 79
def rabbit_families : ℕ := 55
def squirrel_families : ℕ := 40

-- Define the percentages (in decimal form) of families that have moved out or will move out
def deer_moved_out_fraction : ℝ := 0.30
def deer_planned_out_fraction : ℝ := 0.25
def rabbit_moved_out_fraction : ℝ := 0.15
def rabbit_planned_out_fraction : ℝ := 0.35
def squirrel_moved_out_fraction : ℝ := 0.10
def squirrel_planned_out_fraction : ℝ := 0.40

-- Define the rounded number of families that moved out or will move out for each animal type
def deer_moved_out : ℕ := (deer_families * deer_moved_out_fraction).round.toNat + 
                           (deer_families * deer_planned_out_fraction).round.toNat
def rabbit_moved_out : ℕ := (rabbit_families * rabbit_moved_out_fraction).round.toNat + 
                             (rabbit_families * rabbit_planned_out_fraction).round.toNat
def squirrel_moved_out : ℕ := (squirrel_families * squirrel_moved_out_fraction).round.toNat + 
                               (squirrel_families * squirrel_planned_out_fraction).round.toNat

-- Define the remaining families for each animal type
def deer_remaining : ℕ := deer_families - deer_moved_out
def rabbit_remaining : ℕ := rabbit_families - rabbit_moved_out
def squirrel_remaining : ℕ := squirrel_families - squirrel_moved_out

-- Prove that the calculations are equivalent to the given answers
theorem proof_moved_out_and_remaining_families :
  deer_moved_out = 44 ∧ deer_remaining = 35 ∧
  rabbit_moved_out = 27 ∧ rabbit_remaining = 28 ∧
  squirrel_moved_out = 20 ∧ squirrel_remaining = 20 :=
by
  sorry

end proof_moved_out_and_remaining_families_l152_152703


namespace solve_fraction_eqn_l152_152831

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l152_152831


namespace adult_ticket_cost_l152_152842

/--
Tickets at a local theater cost a certain amount for adults and 2 dollars for kids under twelve.
Given that 175 tickets were sold and the profit was 750 dollars, and 75 kid tickets were sold,
prove that an adult ticket costs 6 dollars.
-/
theorem adult_ticket_cost
  (kid_ticket_price : ℕ := 2)
  (kid_tickets_sold : ℕ := 75)
  (total_tickets_sold : ℕ := 175)
  (total_profit : ℕ := 750)
  (adult_tickets_sold : ℕ := total_tickets_sold - kid_tickets_sold)
  (adult_ticket_revenue : ℕ := total_profit - kid_ticket_price * kid_tickets_sold)
  (adult_ticket_cost : ℕ := adult_ticket_revenue / adult_tickets_sold) :
  adult_ticket_cost = 6 :=
by
  sorry

end adult_ticket_cost_l152_152842


namespace general_formula_l152_152626

noncomputable def a : ℕ → ℚ
| 1       := 1/2
| (n + 1) := a n - (a n * a (n + 1)) * (n + 1)

-- The statement we want to prove.
theorem general_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 2 / (n^2 + n + 2) :=
sorry

end general_formula_l152_152626


namespace number_of_squares_and_cubes_less_than_100_l152_152060

theorem number_of_squares_and_cubes_less_than_100 : 
  ∀ (a : ℕ), (a = { n | n < 100 ∧ ∃ k : ℕ, n = k^6 }.card) → a = 2 :=
by
  sorry

end number_of_squares_and_cubes_less_than_100_l152_152060


namespace calculate_length_of_other_train_l152_152479

noncomputable def speed_in_m_per_s (speed_km_per_hr : Float) : Float :=
  speed_km_per_hr * 1000 / 3600

theorem calculate_length_of_other_train
  (length_train1 : Float)
  (speed_train1_km_per_hr : Float)
  (speed_train2_km_per_hr : Float)
  (time_to_cross_sec : Float)
  (length_train2 : Float) :
  length_train1 = 210 → 
  speed_train1_km_per_hr = 120 → 
  speed_train2_km_per_hr = 80 → 
  time_to_cross_sec = 9 → 
  length_train2 = 289.95 :=
by
  intros h1 h2 h3 h4
  let speed_train1_m_per_s := speed_in_m_per_s speed_train1_km_per_hr
  let speed_train2_m_per_s := speed_in_m_per_s speed_train2_km_per_hr
  let relative_speed_m_per_s := speed_train1_m_per_s + speed_train2_m_per_s
  have h5 : relative_speed_m_per_s * time_to_cross_sec = length_train1 + length_train2, by sorry
  have h6 : length_train2 = (relative_speed_m_per_s * time_to_cross_sec) - length_train1, by sorry
  exact h6.trans (by norm_num)

end calculate_length_of_other_train_l152_152479


namespace area_increase_percentage_circumference_increase_percentage_l152_152323

variable {r : ℝ} (h : r > 0)

noncomputable def new_radius (r : ℝ) := 2.5 * r
noncomputable def original_area (r : ℝ) := π * r^2
noncomputable def new_area (r : ℝ) := π * (new_radius r)^2
noncomputable def original_circumference (r : ℝ) := 2 * π * r
noncomputable def new_circumference (r : ℝ) := 2 * π * (new_radius r)

theorem area_increase_percentage : 
  ((new_area r - original_area r) / original_area r) * 100 = 525 := by
  sorry

theorem circumference_increase_percentage :
  ((new_circumference r - original_circumference r) / original_circumference r) * 100 = 150 := by
  sorry

end area_increase_percentage_circumference_increase_percentage_l152_152323


namespace volume_of_snow_correct_l152_152371

noncomputable def volume_of_snow : ℝ :=
  let sidewalk_length := 30
  let sidewalk_width := 3
  let depth := 3 / 4
  let sidewalk_volume := sidewalk_length * sidewalk_width * depth
  
  let garden_path_leg1 := 3
  let garden_path_leg2 := 4
  let garden_path_area := (garden_path_leg1 * garden_path_leg2) / 2
  let garden_path_volume := garden_path_area * depth
  
  let total_volume := sidewalk_volume + garden_path_volume
  total_volume

theorem volume_of_snow_correct : volume_of_snow = 72 := by
  sorry

end volume_of_snow_correct_l152_152371


namespace find_a2_b2_c2_l152_152378

noncomputable theory

variables (a b c : ℝ)

-- Defining the conditions as hypotheses
def condition1 := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
def condition2 := a + b + c = 0
def condition3 := a^7 + b^7 + c^7 = a^9 + b^9 + c^9

-- Defining the target statement
theorem find_a2_b2_c2 (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  a^2 + b^2 + c^2 = 14 / 9 :=
sorry

end find_a2_b2_c2_l152_152378


namespace card_distribution_trick_l152_152901

structure Boxes where
  red : Set ℕ
  white : Set ℕ
  blue : Set ℕ
  allCard : red ∪ white ∪ blue = {n | n ∈ Icc 1 100}
  nonEmptyRed : red ≠ ∅
  nonEmptyWhite : white ≠ ∅ 
  nonEmptyBlue : blue ≠ ∅

def numberOfValidDistributions : ℤ := 12

theorem card_distribution_trick (b : Boxes) :
  ∃! (b : Boxes), true :=
begin
  use b,
  sorry
end

end card_distribution_trick_l152_152901


namespace jessica_speed_last_40_l152_152726

theorem jessica_speed_last_40 
  (total_distance : ℕ)
  (total_time_min : ℕ)
  (first_segment_avg_speed : ℕ)
  (second_segment_avg_speed : ℕ)
  (last_segment_avg_speed : ℕ) :
  total_distance = 120 →
  total_time_min = 120 →
  first_segment_avg_speed = 50 →
  second_segment_avg_speed = 60 →
  last_segment_avg_speed = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end jessica_speed_last_40_l152_152726


namespace populations_equal_after_years_l152_152142

-- Defining the initial population and rates of change
def initial_population_X : ℕ := 76000
def rate_of_decrease_X : ℕ := 1200
def initial_population_Y : ℕ := 42000
def rate_of_increase_Y : ℕ := 800

-- Define the number of years for which we need to find the populations to be equal
def years (n : ℕ) : Prop :=
  (initial_population_X - rate_of_decrease_X * n) = (initial_population_Y + rate_of_increase_Y * n)

-- Theorem stating that the populations will be equal at n = 17
theorem populations_equal_after_years {n : ℕ} (h : n = 17) : years n :=
by
  sorry

end populations_equal_after_years_l152_152142


namespace data_collection_is_conducting_survey_l152_152339

-- Conditions as definitions in Lean
def democratic_election : Prop := 
  ∃ (students : Type) (candidates : Type), 
    ∀ (student : students) (candidate : candidates), 
    student.votes_for candidate

-- Question translated to a proposition with the expected correct answer
theorem data_collection_is_conducting_survey 
  (h : democratic_election) : 
  conducting_the_survey :=
  sorry

end data_collection_is_conducting_survey_l152_152339


namespace sum_of_repeating_decimals_l152_152745

def repeating_decimals_sum : Real :=
  let T := {x | ∃ (a b : ℕ), a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * a + b) / 99}
  Set.sum T

theorem sum_of_repeating_decimals : repeating_decimals_sum = 413.5 :=
  by
    sorry

end sum_of_repeating_decimals_l152_152745


namespace teacher_age_and_avg_age_proof_l152_152397

-- Define initial conditions
def teachers_2007 := 7
def teachers_2010 := teachers_2007 + 1
def teachers_2012 := teachers_2010 - 1
def new_teacher_age := 25

-- Define the sum of the ages of teachers over the years
def sum_ages_2007 (c : ℕ) := c
def sum_ages_2010 (c : ℕ) := c + 21 + new_teacher_age
def sum_ages_2012 (c : ℕ) (x : ℕ) := sum_ages_2010 c + 16 - x

-- Define average age equality across dates
def avg_age_constant (c : ℕ) (x : ℕ) :=
  (sum_ages_2007 c) / teachers_2007 = (sum_ages_2010 c) / teachers_2010 ∧
  (sum_ages_2007 c) / teachers_2007 = (sum_ages_2012 c x) / teachers_2012

-- The correct answers are:
def correct_answers (x c avg_age : ℕ) :=
  x = 62 ∧ avg_age = 46

-- The main theorem to prove
theorem teacher_age_and_avg_age_proof : ∀ (c x avg_age : ℕ),
  avg_age_constant c x → correct_answers x c avg_age :=
begin
  -- With the above definitions and theorem structure, we should be able to 
  -- proceed to prove it in Lean. For now, we use sorry to skip the proof.
  sorry
end

end teacher_age_and_avg_age_proof_l152_152397


namespace shorter_base_of_trapezoid_l152_152019

theorem shorter_base_of_trapezoid (midpoints_segment_length : ℝ) 
  (longer_base_length : ℝ) 
  (midpoints_segment_length = 5) 
  (longer_base_length = 85) : 
  let shorter_base_length := longer_base_length - 2 * midpoints_segment_length in
  shorter_base_length = 75 :=
by
  sorry

end shorter_base_of_trapezoid_l152_152019


namespace sum_binomial_coefficients_l152_152622

theorem sum_binomial_coefficients (n : ℕ) (d : ℕ → ℕ) :
  let rec d_step (k m : ℕ) : ℕ :=
    match k, m with
    | 0, 0 => d 0
    | 0, _ => 0
    | _, 0 => d (k - 1)
    | _, _ => d_step (k - 1) (m - 1) + d_step k (m - 1)
  in d_step (n - 1) 0 = ∑ i in finset.range n, nat.choose (n - 1) i * d i :=
sorry

end sum_binomial_coefficients_l152_152622


namespace min_cuts_for_252_hendecagons_l152_152172

theorem min_cuts_for_252_hendecagons : ∀ n : ℕ, n = 2015 ↔ ∃ k : ℕ, k ≥ n ∧ (∃ p : ℕ, p = 252 ∧ (4 * k + 4) ≥ (11 * p + 3 * (k + 1 - p))) := 
begin
  sorry
end

end min_cuts_for_252_hendecagons_l152_152172


namespace product_of_first_five_terms_l152_152715

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ m + n = p + q → a m * a n = a p * a q

theorem product_of_first_five_terms 
  (h : geometric_sequence a) 
  (h3 : a 3 = 2) : 
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 :=
sorry

end product_of_first_five_terms_l152_152715


namespace Sara_spent_on_salad_l152_152089

theorem Sara_spent_on_salad (cost_of_hotdog : ℝ) (total_bill : ℝ) (cost_of_salad : ℝ) 
  (h_hotdog : cost_of_hotdog = 5.36) (h_total : total_bill = 10.46) : cost_of_salad = 10.46 - 5.36 :=
by
  rw [h_hotdog, h_total]
  rfl

end Sara_spent_on_salad_l152_152089


namespace fraction_of_attendees_did_not_arrive_on_time_l152_152217

theorem fraction_of_attendees_did_not_arrive_on_time
    (total_attendees : ℕ)
    (h_males_fraction : 3 / 5 * total_attendees = ⌊3 / 5 * total_attendees⌋) -- Assuming integer attendees
    (h_males_on_time_fraction : 7 / 8 * ⌊3 / 5 * total_attendees⌋ = ⌊7 / 8 * ⌊3 / 5 * total_attendees⌋⌋)
    (h_females_on_time_fraction : 4 / 5 * (total_attendees - ⌊3 / 5 * total_attendees⌋) = ⌊4 / 5 * (total_attendees - ⌊3 / 5 * total_attendees⌋)⌋) :
  (total_attendees - (⌊7 / 8 * ⌊3 / 5 * total_attendees⌋⌋ + ⌊4 / 5 * (total_attendees - ⌊3 / 5 * total_attendees⌋)⌋)) / total_attendees = 3 / 20 :=
sorry

end fraction_of_attendees_did_not_arrive_on_time_l152_152217


namespace even_sum_probability_l152_152578

-- Define the probabilities for the first wheel
def prob_even_first_wheel : ℚ := 3 / 6
def prob_odd_first_wheel : ℚ := 3 / 6

-- Define the probabilities for the second wheel
def prob_even_second_wheel : ℚ := 3 / 4
def prob_odd_second_wheel : ℚ := 1 / 4

-- Probability that the sum of the two selected numbers is even
def prob_even_sum : ℚ :=
  (prob_even_first_wheel * prob_even_second_wheel) +
  (prob_odd_first_wheel * prob_odd_second_wheel)

-- The theorem to prove
theorem even_sum_probability : prob_even_sum = 13 / 24 := by
  sorry

end even_sum_probability_l152_152578


namespace disc_rotation_exists_l152_152176

noncomputable def sector_length (C : ℝ) (n : ℕ) : ℝ := C / (2 * n)

theorem disc_rotation_exists (C : ℝ) (n : ℕ) (hC_pos : 0 < C) :
  ∃ θ : ℝ, ∃ k : ℕ, k < 2 * n ∧
  let sum_first_part := (2 * n) * sector_length C n (θ = k * (π / n)) ∧
  sum_first_part ≥ (1 / 2) * C :=
sorry

end disc_rotation_exists_l152_152176


namespace rectangle_area_l152_152126

noncomputable def find_x (x : ℝ) := 
  (-1, x) ∧ (5, x) ∧ (-1, -2) ∧ (5, -2) ∧ 6 * (x + 2) = 66 ∧ x > 0

theorem rectangle_area (x : ℝ) (h : find_x x) : x = 9 :=
sorry

end rectangle_area_l152_152126


namespace temperature_altitude_relation_l152_152651

theorem temperature_altitude_relation (t : ℝ) (h : ℝ) :
  (∀ h : ℝ, t = -0.006 * h + 20) ↔ 
  (t 0 = 20) ∧ (∀ h : ℝ, t (h + 1000) = t h - 6) :=
sorry

end temperature_altitude_relation_l152_152651


namespace constants_solution_l152_152237

theorem constants_solution (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2)) ↔ 
    (A = 20 ∧ B = -15 ∧ C = -10) :=
by
  sorry

end constants_solution_l152_152237


namespace sum_of_seven_digits_l152_152402

theorem sum_of_seven_digits : 
  ∃ (digits : Finset ℕ), 
    digits.card = 7 ∧ 
    digits ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    ∃ (a b c d e f g : ℕ), 
      a + b + c = 25 ∧ 
      d + e + f + g = 17 ∧ 
      digits = {a, b, c, d, e, f, g} ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
      c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
      d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
      e ≠ f ∧ e ≠ g ∧
      f ≠ g ∧
      (a + b + c + d + e + f + g = 33) := sorry

end sum_of_seven_digits_l152_152402


namespace syllogism_form_correct_l152_152839

-- Define the initial conditions of the problem
def f (x : ℝ) : ℝ := x^3
def df (x : ℝ) : ℝ := 3 * x^2

def correct_form_of_reasoning (fx : ℝ → ℝ) (x0 : ℝ) (H1 : Differentiable ℝ fx) (H2 : fx = f ∧ df 0 = 0) : Prop :=
  True

theorem syllogism_form_correct :
  correct_form_of_reasoning f 0 (by simp [Differentiable] : Differentiable ℝ f) (by simp [df]; exact ⟨rfl, rfl⟩) :=
sorry

end syllogism_form_correct_l152_152839


namespace triangle_area_l152_152818

-- Definitions of the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 3 = 1

-- Area of the triangle formed by the parabola and asymptotes of the hyperbola
theorem triangle_area :
  (y^2 = 4 * x ∧ (x^2) / 4 - (y^2) / 3 = 1) → area_of_triangle = sqrt(3) / 2 :=
by sorry

end triangle_area_l152_152818


namespace log_problem_l152_152554

theorem log_problem : log 10 50 + log 10 20 - log 10 10 = 2 := by
  sorry

end log_problem_l152_152554


namespace edward_lives_lost_l152_152155

theorem edward_lives_lost (initial_lives remaining_lives : ℕ) (h1 : initial_lives = 15) (h2 : remaining_lives = 7) : initial_lives - remaining_lives = 8 :=
by
  have h3 : 15 - 7 = 8 := by norm_num
  rw [h1, h2]
  exact h3

end edward_lives_lost_l152_152155


namespace valid_b_value_l152_152978

noncomputable def is_rectangle_formed : Prop :=
  ∃ b : ℝ, ∀ z : ℂ, (z^4 - 8 * z^3 + 13 * b * z^2 - 4 * (3 * b^2 + 4 * b - 4) * z + 16 = 0) →
    (∃ w1 w2 : ℂ, w1 ≠ 0 ∧ w2 ≠ 0 ∧ w1 * w1 * w2 * w2 = w1 * w1 + w2 * w2 + (2 + 2i) * (w1 + w2))

theorem valid_b_value : is_rectangle_formed → ∀ (b : ℝ), b = 3 :=
by sorry

end valid_b_value_l152_152978


namespace volume_of_each_cube_is_8_cube_centimeters_l152_152897

-- Define the dimensions of the box.
def box_length : ℕ := 10
def box_width : ℕ := 18
def box_height : ℕ := 4
def number_of_cubes : ℕ := 60
def box_volume : ℕ := box_length * box_width * box_height := by rfl

-- Define the side length of each cube.
def side_length_of_cube : ℕ := 2

-- Given conditions
axiom gcd_of_dimensions : ∃ d : ℕ, d ∣ box_length ∧ d ∣ box_width ∧ d ∣ box_height ∧ ∀ d', d' ∣ box_length ∧ d' ∣ box_width ∧ d' ∣ box_height → d' ≤ d

-- Define the volume of each cube and the correct answer.
def volume_of_each_cube : ℕ := side_length_of_cube^3 := by rfl

theorem volume_of_each_cube_is_8_cube_centimeters (h1 : box_length = 10) (h2 : box_width = 18) (h3 : box_height = 4) (h4 : number_of_cubes = 60) 
  (h5 : box_volume = 720) (h_gcd : gcd_of_dimensions) : volume_of_each_cube = 8 := 
sorry

end volume_of_each_cube_is_8_cube_centimeters_l152_152897


namespace johns_total_working_hours_l152_152049

theorem johns_total_working_hours (d h t : Nat) (h_d : d = 5) (h_h : h = 8) : t = d * h := by
  rewrite [h_d, h_h]
  sorry

end johns_total_working_hours_l152_152049


namespace perimeter_subtriangle_eq_two_l152_152819

-- Define the conditions
variables {A B C X Y M : Point}
variable (P : Triangle A B C)
variable (h_perimeter : perimeter P = 4)
variable (h_AX : dist A X = 1)
variable (h_AY : dist A Y = 1)
variable (h_intersection : line_segment B C ∩ line_segment X Y = Some M)

-- The theorem: proving the perimeter of one of the triangles (ABM or ACM) is 2
theorem perimeter_subtriangle_eq_two :
  (perimeter (Triangle A B M) = 2) ∨ (perimeter (Triangle A C M) = 2) := 
by 
  sorry

end perimeter_subtriangle_eq_two_l152_152819


namespace find_S7_l152_152028

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ a₁, q > 0 ∧ (∀ n, a n = a₁ * q^n)

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n+1), a i

variables (q a₁ : ℝ)
variables (h_geo : is_geometric_sequence a q)
variables (h_S2 : S_n a 2 = 6)
variables (h_S3 : S_n a 3 = 14)

theorem find_S7 : S_n a 7 = 254 := by
  sorry

end find_S7_l152_152028


namespace find_number_l152_152192

-- Define the condition
def condition : Prop := ∃ x : ℝ, x / 0.02 = 50

-- State the theorem to prove
theorem find_number (x : ℝ) (h : x / 0.02 = 50) : x = 1 :=
sorry

end find_number_l152_152192


namespace sara_spent_on_salad_l152_152084

theorem sara_spent_on_salad: 
  ∀ (cost_hotdog cost_total cost_salad : ℝ),
  cost_hotdog = 5.36 →
  cost_total = 10.46 →
  cost_salad = cost_total - cost_hotdog →
  cost_salad = 5.10 := 
by
  intros cost_hotdog cost_total cost_salad h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sara_spent_on_salad_l152_152084


namespace M_gt_N_l152_152263

variable (x : ℝ)

def M := x^2 + 4 * x - 2

def N := 6 * x - 5

theorem M_gt_N : M x > N x := sorry

end M_gt_N_l152_152263


namespace school_spent_total_amount_l152_152526

theorem school_spent_total_amount
  (num_cartons_pencils : ℕ)
  (boxes_per_carton_pencils : ℕ)
  (cost_per_box_pencils : ℕ)
  (num_cartons_markers : ℕ)
  (boxes_per_carton_markers : ℕ)
  (cost_per_box_markers : ℕ)
  (total_spent : ℕ)
  (h1 : num_cartons_pencils = 20)
  (h2 : boxes_per_carton_pencils = 10)
  (h3 : cost_per_box_pencils = 2)
  (h4 : num_cartons_markers = 10)
  (h5 : boxes_per_carton_markers = 5)
  (h6 : cost_per_box_markers = 4)
  (h7 : total_spent = 
        (num_cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils)
        + (num_cartons_markers * boxes_per_carton_markers * cost_per_box_markers)) :
  total_spent = 600 :=
by
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7.mpr rfl

end school_spent_total_amount_l152_152526


namespace race_outcomes_l152_152540

theorem race_outcomes (participants : Finset ℕ) (h : participants.card = 6) : 
  (∏ i in Finset.range 4, 6 - i) = 360 :=
by
  -- The product of the decrements 6 * 5 * 4 * 3 is calculated here
  have h1 : ∏ i in Finset.range 4, 6 - i = 6 * 5 * 4 * 3, from sorry
  -- Subsequently, the result 6 * 5 * 4 * 3 equals 360
  have h2 : 6 * 5 * 4 * 3 = 360, by simp
  exact h1.trans h2

end race_outcomes_l152_152540


namespace students_scoring_85_to_90_l152_152185

-- Define the given conditions
def class_size : ℕ := 45
def mean_score : ℝ := 80
def std_dev : ℝ := 5
def prob_1_sigma : ℝ := 0.6827
def prob_2_sigma : ℝ := 0.9545
def prob_3_sigma : ℝ := 0.9973

-- Define our probability of interest
def prob_85_to_90 : ℝ := (prob_2_sigma - prob_1_sigma) / 2

-- Define the expected number of students scoring between 85 and 90
def expected_students : ℕ := (class_size * prob_85_to_90).round.to_nat

-- Statement: The expected number of students scoring between 85 and 90 is 6
theorem students_scoring_85_to_90 : expected_students = 6 := by
  sorry

end students_scoring_85_to_90_l152_152185


namespace find_a_range_l152_152568

noncomputable def f (x : ℝ) : ℝ := sorry

axiom decreasing : ∀ x1 x2 : ℝ, x1 ≤ x2 → f x2 ≤ f x1
axiom odd : ∀ x : ℝ, f (-x) = -f x
axiom domain : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = f x  -- Ensure f is defined on [-1, 1]

theorem find_a_range (a : ℝ) : (f (a^2 - a - 1) + f (4a - 5) > 0) → (1 ≤ a ∧ a < (-3 + Real.sqrt 33) / 2) :=
by
  sorry

end find_a_range_l152_152568


namespace yunkyung_work_per_day_l152_152360

theorem yunkyung_work_per_day (T : ℝ) (h : T > 0) (H : T / 3 = 1) : T / 3 = 1/3 := 
by sorry

end yunkyung_work_per_day_l152_152360


namespace new_value_R_l152_152054

theorem new_value_R (g S R : ℝ) (h1 : R = g * S - 7) (h2 : S = 5) (h3 : R = 8) :
  let S_new := S + 0.5 * S in
  R = 3 * S_new - 7 :=
by
  let g := 3; split; sorry

end new_value_R_l152_152054


namespace neg_exists_equiv_forall_l152_152067

theorem neg_exists_equiv_forall (p : Prop) :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := sorry

end neg_exists_equiv_forall_l152_152067


namespace sum_first_21_terms_l152_152674

theorem sum_first_21_terms (x : ℕ → ℝ) :
    x 1 = 1 ∧
    (∀ n : ℕ, x (n + 1) = -x n + 1 / 2) →
    (∑ i in Finset.range 21, x (i + 1)) = 6 :=
by
  sorry

end sum_first_21_terms_l152_152674


namespace garden_area_difference_l152_152496

theorem garden_area_difference : 
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let area_rectangle := length * width
  let area_circle := Real.pi * radius ^ 2
  let area_difference := area_circle - area_rectangle
  area_difference ≈ 837 :=
by {
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let area_rectangle := length * width
  let area_circle := Real.pi * radius ^ 2
  let area_difference := area_circle - area_rectangle 
  have h1 : area_rectangle = 1200 := rfl
  have h2 : area_circle = 6400 / Real.pi := by sorry
  have h3 : area_difference = 6400 / Real.pi - 1200 := rfl
  have h4 : area_difference ≈ 837 := by sorry
  exact h4
}

end garden_area_difference_l152_152496


namespace sequence_general_term_l152_152129

theorem sequence_general_term :
  ∀ (n : ℕ), n ≥ 1 → (∃ (a : ℝ), a = sqrt (3 * n - 1)) :=
by
  sorry

end sequence_general_term_l152_152129


namespace proof_sum_c_squared_l152_152566

noncomputable def c_k (k : ℕ) : ℝ := 
  k + 1 / (2 * k + 1 + 1 / (2 * k + 2 + 1 / (2 * k + 3 + ...)))

noncomputable def c_squared (k : ℕ) : ℝ := (c_k k) ^ 2

noncomputable def sum_c_squared : ℝ := 
  (Finset.range 15).sum (λ k, c_squared (k + 1))

theorem proof_sum_c_squared :
  abs (sum_c_squared - 1255) < 1 :=
sorry

end proof_sum_c_squared_l152_152566


namespace juan_speed_l152_152369

theorem juan_speed (J : ℝ) :
  (∀ (time : ℝ) (distance : ℝ) (peter_speed : ℝ),
    time = 1.5 →
    distance = 19.5 →
    peter_speed = 5 →
    distance = J * time + peter_speed * time) →
  J = 8 :=
by
  intro h
  sorry

end juan_speed_l152_152369


namespace isosceles_triangle_perimeter_l152_152150

theorem isosceles_triangle_perimeter (side1 side2 base : ℕ)
    (h1 : side1 = 12) (h2 : side2 = 12) (h3 : base = 17) : 
    side1 + side2 + base = 41 := by
  sorry

end isosceles_triangle_perimeter_l152_152150


namespace circle_through_center_l152_152634

theorem circle_through_center 
  (A B C P D E F : Point)
  (hC_midpoint : midpoint A B C)
  (hP_on_extension : lies_on_extension P B A)
  (hPD_tangent : tangent P D (semicircle A B))
  (hE_intersects : intersects (angle_bisector (angle B P D)) (line A C) E)
  (hF_intersects : intersects (angle_bisector (angle B P D)) (line B C) F) :
  passes_through (circle_diameter E F) C :=
sorry

end circle_through_center_l152_152634


namespace equation_linear_in_x_y_l152_152010

theorem equation_linear_in_x_y (n m : ℤ) (h : (n - 1) * x ^ (n ^ 2) - 3 * y ^ (m - 2023) = 6) 
  (h1 : (n ^ 2 = 1 ∨ m - 2023 = 1)) :
  n + m = 2023 := 
by
  sorry

end equation_linear_in_x_y_l152_152010


namespace rows_of_roses_l152_152329

variable (rows total_roses_per_row roses_per_row_red roses_per_row_non_red roses_per_row_white roses_per_row_pink total_pink_roses : ℕ)
variable (half_two_fifth three_fifth : ℚ)

-- Assume the conditions
axiom h1 : total_roses_per_row = 20
axiom h2 : roses_per_row_red = total_roses_per_row / 2
axiom h3 : roses_per_row_non_red = total_roses_per_row - roses_per_row_red
axiom h4 : roses_per_row_white = (3 / 5 : ℚ) * roses_per_row_non_red
axiom h5 : roses_per_row_pink = (2 / 5 : ℚ) * roses_per_row_non_red
axiom h6 : total_pink_roses = 40

-- Prove the number of rows in the garden
theorem rows_of_roses : rows = total_pink_roses / (roses_per_row_pink) :=
by
  sorry

end rows_of_roses_l152_152329


namespace find_number_l152_152494

theorem find_number (x : ℝ) : ((x - 50) / 4) * 3 + 28 = 73 → x = 110 := 
  by 
  sorry

end find_number_l152_152494


namespace necessary_condition_l152_152781

-- Definitions
def good_quality : Prop := sorry
def not_cheap : Prop := sorry

-- Given conditions
axiom you_get_what_you_pay_for : good_quality → not_cheap

-- Prove that "not_cheap" is a necessary condition for "good_quality"
theorem necessary_condition :
  (∀ (P Q : Prop), (Q → P) → (P → Q)) →
  (you_get_what_you_pay_for → (¬not_cheap → ¬good_quality)) :=
sorry

end necessary_condition_l152_152781


namespace maximum_volume_tetrahedron_in_cylinder_l152_152986

theorem maximum_volume_tetrahedron_in_cylinder (R h : ℝ) (hR : 0 < R) (hh : 0 < h) :
  ∃ (V : ℝ), V = (2 / 3) * R^2 * h :=
begin
  use (2 / 3) * R^2 * h,
  sorry,
end

end maximum_volume_tetrahedron_in_cylinder_l152_152986


namespace probability_of_diff_by_three_is_one_eighth_l152_152929

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l152_152929


namespace freezer_temp_correct_l152_152330

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l152_152330


namespace problem_solution_l152_152290

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2^a - 4^b = log 2 (b / a)) : a < 2 * b := 
sorry

end problem_solution_l152_152290


namespace exponentiation_rule_l152_152468

theorem exponentiation_rule (x : ℝ) : (x^5)^2 = x^10 :=
by {
  sorry
}

end exponentiation_rule_l152_152468


namespace absolute_value_neg_parallel_through_point_irrational_sqrt_12_perpendicular_through_point_l152_152936

theorem absolute_value_neg (x : ℝ) (h : x < 0) : |x| = -x :=
by sorry

theorem parallel_through_point (P : ℝ × ℝ) (l : ℝ → ℝ) (hP : ¬ ∃ x, l x = P.snd) :
  ∃! m : ℝ → ℝ, ∀ x : ℝ, l x ≠ m (x - P.fst) :=
by sorry

theorem irrational_sqrt_12 : irrational (real.sqrt 12) :=
by sorry

theorem perpendicular_through_point (P : ℝ × ℝ) (l : ℝ → ℝ) (hP : ¬ ∃ x, l x = P.snd) :
  ∃! m : ℝ → ℝ, ∀ x : ℝ, l x ⟂ m (x - P.fst) :=
by sorry

end absolute_value_neg_parallel_through_point_irrational_sqrt_12_perpendicular_through_point_l152_152936


namespace arithmetic_sequence_twentieth_term_l152_152430

theorem arithmetic_sequence_twentieth_term
  (a1 : ℤ) (a13 : ℤ) (a20 : ℤ) (d : ℤ)
  (h1 : a1 = 3)
  (h2 : a13 = 27)
  (h3 : a13 = a1 + 12 * d)
  (h4 : a20 = a1 + 19 * d) : 
  a20 = 41 :=
by
  --  We assume a20 and prove it equals 41 instead of solving it in steps
  sorry

end arithmetic_sequence_twentieth_term_l152_152430


namespace negation_of_quadratic_proposition_l152_152672

theorem negation_of_quadratic_proposition (a b c : ℝ) (h : a ≠ 0) : 
  (a * c < 0) → ∃ x : ℝ, x = (-b + Real.sqrt(b^2 - 4 * a * c)) / (2 * a) ∨ x = (-b - Real.sqrt(b^2 - 4 * a * c)) / (2 * a) :=
by
  sorry

end negation_of_quadratic_proposition_l152_152672


namespace fraction_identity_proof_l152_152723

theorem fraction_identity_proof (a b : ℝ) (h : 2 / a - 1 / b = 1 / (a + 2 * b)) :
  4 / (a ^ 2) - 1 / (b ^ 2) = 1 / (a * b) :=
by
  sorry

end fraction_identity_proof_l152_152723


namespace combo_of_three_from_nine_l152_152395

theorem combo_of_three_from_nine : (nat.choose 9 3) = 84 := by
  sorry

end combo_of_three_from_nine_l152_152395


namespace original_cost_price_l152_152189

theorem original_cost_price (C : ℝ) 
  (h1 : 0.87 * C > 0) 
  (h2 : 1.2 * (0.87 * C) = 54000) : 
  C = 51724.14 :=
by
  sorry

end original_cost_price_l152_152189


namespace find_eighth_number_l152_152794

def average_of_numbers (a b c d e f g h x : ℕ) : ℕ :=
  (a + b + c + d + e + f + g + h + x) / 9

theorem find_eighth_number (a b c d e f g h x : ℕ) (avg : ℕ) 
    (h_avg : average_of_numbers a b c d e f g h x = avg)
    (h_total_sum : a + b + c + d + e + f + g + h + x = 540)
    (h_x_val : x = 65) : a = 53 :=
by
  sorry

end find_eighth_number_l152_152794


namespace age_problem_l152_152409

def p := 3 * x
def q := 4 * x
def age_difference (years_ago : ℕ) := 
  (p - years_ago = (1 / 2) * (q - years_ago))

theorem age_problem (x : ℕ) (years_ago : ℕ) (hx : years_ago = 6) (h_ratio : p / q = 3 / 4) (h_sum : p + q = 21) : 
  age_difference years_ago := 
by
  sorry

end age_problem_l152_152409


namespace lockers_open_after_process_l152_152913

theorem lockers_open_after_process : 
  ∀ (n : ℕ), n < 100 ∧ n % 2 = 1 → ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 100 → 
  (∃! m, m = 10 ∧ (∀ (i : ℕ), i ∈ [1..100] → 
  ∃ (divisors_odd : ℕ), 
  ∃ (divs : List ℕ), divs.length = divisors_odd ∧ 
  (λ num, num % (2:ℕ) = 1) ∧ 
  (λ subset, Set.Subset subset (List.range 100)) → 
  (lockers (subset (List.append divs)) = true)) 


end lockers_open_after_process_l152_152913


namespace abs_diff_roots_eq_3_l152_152590

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l152_152590


namespace sum_of_all_three_digit_numbers_is_1998_l152_152612

open Finset

-- Define the digits set
def digits : Finset ℕ := {1, 3, 5}

-- Define the set of all three-digit numbers using the digits, ensuring all digits are distinct
def three_digit_numbers : Finset ℕ :=
  digits.image (λ x, digits.erase x).image (λ y, digits.erase y).image (λ z, 100 * x + 10 * y + z)

-- The sum of all unique three-digit numbers formed using 1, 3, and 5
def sum_of_three_digit_numbers := three_digit_numbers.sum id

-- Proof problem statement
theorem sum_of_all_three_digit_numbers_is_1998 : sum_of_three_digit_numbers = 1998 := by
  sorry

end sum_of_all_three_digit_numbers_is_1998_l152_152612


namespace net_speed_of_man_traveling_in_streams_l152_152902

noncomputable def avg_speed (downstream upstream : ℝ) : ℝ :=
  (2 * downstream * upstream) / (downstream + upstream)

theorem net_speed_of_man_traveling_in_streams :
  let speed1 := avg_speed 10 8 in
  let speed2 := avg_speed 15 5 in
  (speed1 + speed2) / 2 = 8.19 :=
by
  have speed1 := avg_speed 10 8
  have speed2 := avg_speed 15 5
  have net_speed := (speed1 + speed2) / 2
  sorry

end net_speed_of_man_traveling_in_streams_l152_152902


namespace square_side_length_l152_152398

theorem square_side_length (a b s : ℝ) 
  (h_area : a * b = 54) 
  (h_square_condition : 3 * a = b / 2) : 
  s = 9 :=
by 
  sorry

end square_side_length_l152_152398


namespace range_of_a_if_f_a_ge_4_l152_152619

def f(x : ℝ) : ℝ := max (x^2) (1/x)

theorem range_of_a_if_f_a_ge_4 (a : ℝ) (h : 0 < a) (h_f : f(a) ≥ 4) : 
    a ≥ 2 ∨ (0 < a ∧ a ≤ 1/4) := 
by sorry

end range_of_a_if_f_a_ge_4_l152_152619


namespace ladder_leaning_distance_l152_152536

variable (m f h : ℝ)
variable (f_pos : f > 0) (h_pos : h > 0)

def distance_to_wall_upper_bound : ℝ := 12.46
def distance_to_wall_lower_bound : ℝ := 8.35

theorem ladder_leaning_distance (m f h : ℝ) (f_pos : f > 0) (h_pos : h > 0) :
  ∃ x : ℝ, x = 12.46 ∨ x = 8.35 := 
sorry

end ladder_leaning_distance_l152_152536


namespace probability_all_cocaptains_l152_152998

noncomputable def prob_of_selecting_all_cocaptains 
  (team_sizes : List ℕ) (num_cocaptains : ℕ) : ℚ :=
  let team_prob := 1 / (team_sizes.length : ℚ)
  let selected_prob := λ n : ℕ => 1 / (nat.choose n 3)
  team_prob * list.sum (team_sizes.map (λ n => selected_prob n))

theorem probability_all_cocaptains :
  prob_of_selecting_all_cocaptains [6, 7, 8, 9] 3 = 469 / 23520 := 
sorry

end probability_all_cocaptains_l152_152998


namespace sum_of_elements_of_T_l152_152741

def is_repeating_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧ x = (10 * a + b) / 99

def sum_of_repeating_decimals : ℝ :=
  ∑ x in { x : ℝ | is_repeating_decimal x }, x

theorem sum_of_elements_of_T : sum_of_repeating_decimals = 45 :=
by
  sorry

end sum_of_elements_of_T_l152_152741


namespace missing_digit_in_138_x_6_divisible_by_9_l152_152987

theorem missing_digit_in_138_x_6_divisible_by_9 :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ (1 + 3 + 8 + x + 6) % 9 = 0 ∧ x = 0 :=
by
  sorry

end missing_digit_in_138_x_6_divisible_by_9_l152_152987


namespace arcsin_neg_one_is_neg_half_pi_l152_152963

noncomputable def arcsine_equality : Prop := 
  real.arcsin (-1) = - (real.pi / 2)

theorem arcsin_neg_one_is_neg_half_pi : arcsine_equality :=
by
  sorry

end arcsin_neg_one_is_neg_half_pi_l152_152963


namespace delegate_seating_probability_l152_152847

theorem delegate_seating_probability :
  ∃ m n : ℕ, nat.coprime m n ∧ (m + n = 909) ∧
  (∑ i in range 12, 1 : ℕ) = 12 ∧
  (∑ c in range 3, 4 : ℕ) = 12 ∧
  (∃ (p : ℕ), p = 409 ∧ ∃ (q : ℕ), q = 500 ∧
  ∃ (prob : ℚ), prob = (p / q) ∧ m = p ∧ n = q) :=
by
  sorry

end delegate_seating_probability_l152_152847


namespace children_ticket_price_difference_l152_152187

noncomputable def regular_ticket_price : ℝ := 9
noncomputable def total_amount_given : ℝ := 2 * 20
noncomputable def total_change_received : ℝ := 1
noncomputable def num_adults : ℕ := 2
noncomputable def num_children : ℕ := 3
noncomputable def total_cost_of_tickets : ℝ := total_amount_given - total_change_received
noncomputable def children_ticket_cost := (total_cost_of_tickets - num_adults * regular_ticket_price) / num_children

theorem children_ticket_price_difference :
  (regular_ticket_price - children_ticket_cost) = 2 := by
  sorry

end children_ticket_price_difference_l152_152187


namespace probability_of_diff_by_three_is_one_eighth_l152_152930

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l152_152930


namespace plane_divides_space_into_two_parts_l152_152521

def divides_space : Prop :=
  ∀ (P : ℝ → ℝ → ℝ → Prop), (∀ x y z, P x y z → P x y z) →
  (∃ region1 region2 : ℝ → ℝ → ℝ → Prop,
    (∀ x y z, P x y z → (region1 x y z ∨ region2 x y z)) ∧
    (∀ x y z, region1 x y z → ¬region2 x y z) ∧
    (∃ x1 y1 z1 x2 y2 z2, region1 x1 y1 z1 ∧ region2 x2 y2 z2))

theorem plane_divides_space_into_two_parts (P : ℝ → ℝ → ℝ → Prop) (hP : ∀ x y z, P x y z → P x y z) : 
  divides_space :=
  sorry

end plane_divides_space_into_two_parts_l152_152521


namespace no_integer_solution_for_Q_square_l152_152230

def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

theorem no_integer_solution_for_Q_square :
  ∀ x : ℤ, ∃ k : ℤ, Q x = k^2 → false :=
by
  sorry

end no_integer_solution_for_Q_square_l152_152230


namespace sum_of_seq_ge_zero_l152_152734

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem sum_of_seq_ge_zero (n : ℕ) (h : n > 0) : 
  ∑ k in Finset.range n, (-1:ℤ) ^ floor (k * (Real.sqrt 2 - 1)) ≥ 0 := sorry

end sum_of_seq_ge_zero_l152_152734


namespace nelly_refrigerator_payment_l152_152394

theorem nelly_refrigerator_payment (T : ℝ) (p1 p2 p3 : ℝ) (p1_percent p2_percent p3_percent : ℝ)
  (h1 : p1 = 875) (h2 : p2 = 650) (h3 : p3 = 1200)
  (h4 : p1_percent = 0.25) (h5 : p2_percent = 0.15) (h6 : p3_percent = 0.35)
  (total_paid := p1 + p2 + p3)
  (percent_paid := p1_percent + p2_percent + p3_percent)
  (total_cost := total_paid / percent_paid)
  (remaining := total_cost - total_paid) :
  remaining = 908.33 := by
  sorry

end nelly_refrigerator_payment_l152_152394


namespace probability_one_male_one_female_l152_152767

variable (Astronauts : Finset ℕ)
variable (male : Finset ℕ) (female : Finset ℕ)

axiom h1 : male.card = 2
axiom h2 : female.card = 2
axiom total_astronauts : Astronauts = male ∪ female

theorem probability_one_male_one_female 
  (H : male.card + female.card = 4)
  (Ht : (Astronauts.choose 2).card = 6)
  (Hf : (male.choose 1).card = 2)
  (Hm : (female.choose 1).card = 2) :
  (↑((male.choose 1).card * (female.choose 1).card) / ↑((Astronauts.choose 2).card)) = (2 / 3) := 
sorry

end probability_one_male_one_female_l152_152767


namespace compute_sqrt_fraction_l152_152966

theorem compute_sqrt_fraction :
  (Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35))) = (256 / Real.sqrt 2049) :=
sorry

end compute_sqrt_fraction_l152_152966


namespace triangle_ST_length_l152_152564

theorem triangle_ST_length (PQ PR QR : ℝ) (S T : ExistsOnLineSegment PQ PR) (incenter : Point) 
  (h1 : PQ = 26) (h2 : PR = 30) (h3 : QR = 28) 
  (h4 : lineSegment ST parallel QR) (h5 : contains incenter [S, T]) :
  ∃ a b : ℝ, (a / b = 14) ∧ (a + b = 15) :=
by
  sorry

end triangle_ST_length_l152_152564


namespace value_of_abc_l152_152316

theorem value_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1 / b = 5) (h2 : b + 1 / c = 2) (h3 : c + 1 / a = 3) : 
  abc = 1 :=
by
  sorry

end value_of_abc_l152_152316


namespace greatest_root_l152_152596

noncomputable def g (x : ℝ) : ℝ := 20 * x^4 - 21 * x^2 + 5

theorem greatest_root (x: ℝ) (h: g(x) = 0) : x ≤ 1 :=
by
  sorry

end greatest_root_l152_152596


namespace compare_values_l152_152274

def f (x : ℝ) : ℝ := 2 * x + real.cos x

def a : ℝ := f 1
def b : ℝ := f (-1)
def c : ℝ := f 2

theorem compare_values : b < a ∧ a < c := by
  sorry

end compare_values_l152_152274


namespace sin_alpha_second_quadrant_l152_152635

/-- Given angle α in the second quadrant such that tan(π - α) = 3/4, we need to prove that sin α = 3/5. -/
theorem sin_alpha_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.tan (π - α) = 3 / 4) : Real.sin α = 3 / 5 := by
  sorry

end sin_alpha_second_quadrant_l152_152635


namespace trajectory_of_P_is_circle_range_of_k_exists_k_for_dot_product_l152_152311

-- Define the fixed points M and N
def M : ℝ × ℝ := (0, 1)
def N : ℝ × ℝ := (1, 2)

-- Define the conditions given in the problem
def ratio_condition (P : ℝ × ℝ) : Prop :=
  dist P M = √2 * dist P N

-- Define the line equation
def line_eq (k x : ℝ) : ℝ :=
  k * x - 1

-- Prove the trajectory of point P
theorem trajectory_of_P_is_circle :
  ∃ (P : ℝ × ℝ), ratio_condition P ∧ (P.fst - 2)^2 + (P.snd - 3)^2 = 4 ∧
  (∀ (x y : ℝ), ratio_condition (x, y) → (x - 2)^2 + (y - 3)^2 = 4) :=
by
  sorry

-- Prove the range of k values
theorem range_of_k (k : ℝ) : 
  (∀ P, ratio_condition P ∧ (P.fst - 2)^2 + (P.snd - 3)^2 = 4) →
  (∀ l, abs ((2 * k - 4) - 4) / sqrt (k^2 + 1) < 2 → k > 3/4) :=
by
  sorry

-- Prove the existence of k such that dot product condition holds
theorem exists_k_for_dot_product (k : ℝ) :
  k > 3/4 →
  (∃ (x1 x2 : ℝ), (k^2 + 1) * x1 * x2 - k * (x1 + x2) - 10 = 0) →
  (∃ k, k = 1 ∧ (∀ (A B : ℝ × ℝ), ratio_condition A ∧ ratio_condition B ∧ (A.fst * B.fst + (line_eq k A.fst - 1) * (line_eq k B.fst - 1) = 11))) :=
by
  sorry

end trajectory_of_P_is_circle_range_of_k_exists_k_for_dot_product_l152_152311


namespace units_digit_remainder_l152_152151

theorem units_digit_remainder :
  (2851 * 7347 * 419^2) % 10 = 7 :=
begin
  sorry  -- proof to be filled in later
end

end units_digit_remainder_l152_152151


namespace molecular_weight_N2O3_correct_l152_152955

/-- Conditions -/
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

/-- Proof statement -/
theorem molecular_weight_N2O3_correct :
  (2 * atomic_weight_N + 3 * atomic_weight_O) = 76.02 ∧
  name_of_N2O3 = "dinitrogen trioxide" := sorry

/-- Definition of the compound name based on formula -/
def name_of_N2O3 : String := "dinitrogen trioxide"

end molecular_weight_N2O3_correct_l152_152955


namespace parabola_focus_l152_152304

noncomputable def parabola_focus_coords (p : ℝ) (hp : p > 0) (y_directrix : ℝ) (h_directrix : y_directrix = -2) : ℝ × ℝ :=
  if h : y_directrix = -p/2 then (0, p/2) else (0, 0)

theorem parabola_focus {p : ℝ} (hp : p > 0) : parabola_focus_coords p hp (-2) (by norm_num) = (0, 2) :=
begin
  unfold parabola_focus_coords,
  split_ifs,
  case pos neg => sorry,
end

end parabola_focus_l152_152304


namespace minimum_sum_achieved_at_l152_152351

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

/-- Assume we have an arithmetic sequence with a₁ = -9 and S₃ = S₇. -/
def arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  a 1 = -9 ∧ ∀ d, S 3 = S 7 → ∀ k, S k = ∑ i in range k, a (i + 1)

/-- We aim to prove that the minimum of the sum of the first n terms is achieved when n=5. -/
theorem minimum_sum_achieved_at (h : arithmetic_sequence a_n) : 
  ∃ n, S n ≤ S k ∀ k then n = 5 :=
sorry

end minimum_sum_achieved_at_l152_152351


namespace keychain_arrangements_distinct_count_l152_152345

theorem keychain_arrangements_distinct_count :
  let keys := ["H", "C", "W", "O", "Key1", "Key2"]
  let pairs := [("H", "C"), ("W", "O")]
  let blocks := ["HC", "WO", "Key1", "Key2"]
  fact_blocks_next_to : (∀ (block : String), block ∈ blocks → block.length = 2) →
  let count := ((blocks.length - 1) ! / 2)
  count = 3 := 
by {
  -- Definitions for conditions
  let keys := ["H", "C", "W", "O", "Key1", "Key2"]
  let pairs := [("H", "C"), ("W", "O")]
  let blocks := ["HC", "WO", "Key1", "Key2"]
  
  -- Fact that blocks have 2 keys
  have fact_blocks_next_to : ∀ (block : String), block ∈ blocks -> block.length = 2,
    from sorry,

  -- Calculating the count and establishing the theorem
  let count := (finset.card (finset.range (blocks.length - 1)) ! / 2)
  show count = 3,
  from sorry
} 

end keychain_arrangements_distinct_count_l152_152345


namespace number_of_lightsabers_in_order_l152_152156

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end number_of_lightsabers_in_order_l152_152156


namespace total_worksheets_l152_152915

theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) :
    (problems_per_worksheet = 4) →
    (graded_worksheets = 5) →
    (remaining_problems = 16) →
    (graded_worksheets + remaining_problems / problems_per_worksheet = 9) :=
by {
    intros,
    sorry
}

end total_worksheets_l152_152915


namespace harkamal_grapes_rate_l152_152312

theorem harkamal_grapes_rate :
  ∃ G : ℝ, 8 * G + 9 * 45 = 965 ∧ G = 70 := 
by
  use 70
  split
  · calc
      8 * 70 + 9 * 45 = 8 * 70 + 405            : rfl
                ... = 560 + 405                 : by norm_num
                ... = 965                       : by norm_num
  · rfl

end harkamal_grapes_rate_l152_152312


namespace P5_div5_P5_not_div10_P6_div6_P7_div7_l152_152562

noncomputable def P : ℕ → ℤ → ℤ
| 1, _   => 1
| 2, x   => x
| (n+1), x => x * P n x - P (n-1) x 

theorem P5_div5 (x : ℤ) : (∃ t : ℤ, x = 5 * t + 2 ∨ x = 5 * t - 2) → P 5 x % 5 = 0 := 
  sorry

theorem P5_not_div10 (x : ℤ) : ¬(P 5 x % 10 = 0) := 
  sorry

theorem P6_div6 (x : ℤ) : P 6 x % 6 = 0 := 
  sorry

theorem P7_div7 (x : ℤ) : (∃ t : ℤ, x = 7 * t + 2 ∨ x = 7 * t - 2) → P 7 x % 7 = 0 := 
  sorry

end P5_div5_P5_not_div10_P6_div6_P7_div7_l152_152562


namespace area_of_garden_l152_152416

variable (P : ℝ) (A : ℝ)

theorem area_of_garden (hP : P = 38) (hA : A = 2 * P + 14.25) : A = 90.25 :=
by
  sorry

end area_of_garden_l152_152416


namespace existence_of_stationary_point_l152_152459

-- Define the circles and points
variables {C₁ C₂ : Set Point} -- The two intersecting circles
variables {cyclist₁ cyclist₂ : Cyclist} -- The two cyclists riding along the circles
variables {A : Point} -- Starting and ending point (intersection point of the circles)
variable {t : ℝ} -- Time variable
variable [Inhabited Point] -- Ensuring we have an inhabited type for points

-- Cyclists' positions as functions of time
variables (M N : ℝ → Point)

-- Constant speed assumption
def constant_speed (f : ℝ → Point) := ∃ v : ℝ, ∀ t, ((f t + v * t) = f (t + 1))

-- Proving the existence of a stationary point equidistant to both cyclists
theorem existence_of_stationary_point
  (h₁ : cyclist₁.riding_on_circle C₁)
  (h₂ : cyclist₂.riding_on_circle C₂)
  (h₃ : cyclist₁.constant_speed)
  (h₄ : cyclist₂.constant_speed)
  (h₅ : cyclist₁.initial_point = A)
  (h₆ : cyclist₂.initial_point = A)
  (h₇ : cyclist₁.position (1 : ℝ) = A)
  (h₈ : cyclist₂.position (1 : ℝ) = A)
  :
  (∃ (P : Point), ∀ t, distance P (M t) = distance P (N t)) ∧
  (∃ (P : Point), ∀ t, distance P (M t) = distance P (N t)) :=
by
  sorry

end existence_of_stationary_point_l152_152459


namespace candy_bars_first_day_eq_190_l152_152782

-- Definitions and conditions
def sold_candy_bars (x : ℕ) : ℕ :=
  x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20)

def cost_per_candy_bar : ℕ := 10
def earned_in_week : ℕ := 1200  -- $12 converted to cents

-- Theorem to prove the number of candy bars sold on the first day
theorem candy_bars_first_day_eq_190 (x : ℕ)
  (h : sold_candy_bars x * cost_per_candy_bar = earned_in_week) :
  x = 190 :=
begin
  sorry
end

end candy_bars_first_day_eq_190_l152_152782


namespace divisor_sum_l152_152384

theorem divisor_sum (n a : ℕ) (h1: n ≥ 2) (h2: 0 < a) (h3: a ≤ n.fact) : 
  ∃ (k: ℕ) (k_lt_n: k < n) (d : Fin k → ℕ), 
  (∀ i : Fin k, d i ∣ n.fact) ∧ Function.Injective d ∧ (d.sum = a) :=
sorry

end divisor_sum_l152_152384


namespace sum_series_eq_l152_152059

noncomputable def sum_series (x : ℝ) := ∑' n, 1 / (x^(3^n) + x^(-3^n))

theorem sum_series_eq (x : ℝ) (hx : x > 2) :
  sum_series x = 1 / (x + 1) :=
sorry

end sum_series_eq_l152_152059


namespace ratio_seconds_l152_152442

theorem ratio_seconds (x : ℕ) (h : 12 / x = 6 / 240) : x = 480 :=
sorry

end ratio_seconds_l152_152442


namespace find_x_l152_152680

def vec (x y : ℝ) := (x, y)

def a := vec 1 (-4)
def b (x : ℝ) := vec (-1) x
def c (x : ℝ) := (a.1 + 3 * (b x).1, a.2 + 3 * (b x).2)

theorem find_x (x : ℝ) : a.1 * (c x).2 = (c x).1 * a.2 → x = 4 :=
by
  sorry

end find_x_l152_152680


namespace train_length_is_750_l152_152170

-- Defining the conditions
def speed_kmph : ℝ := 90
def speed_mps : ℝ := speed_kmph * (1000 / 3600)
def time_sec : ℝ := 60
def total_distance : ℝ := speed_mps * time_sec
def length_train_platform_equal_condition : ℝ := total_distance / 2

-- Statement to be proved
theorem train_length_is_750 :
  length_train_platform_equal_condition = 750 :=
by sorry

end train_length_is_750_l152_152170


namespace find_trig_expression_value_l152_152637

theorem find_trig_expression_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
    (sin (2 * α) - cos α ^ 2) / (1 + cos (2 * α)) = -5 / 6 := 
by 
    sorry

end find_trig_expression_value_l152_152637


namespace remainder_relationship_varies_l152_152755

theorem remainder_relationship_varies (M M' N D S S' s s' : ℕ)
  (hM_gt : M > M')
  (hM_mod : M % D = S)
  (hM'_mod : M' % D = S')
  (hM_sq_mod : (M ^ 2 * M') % D = s)
  (hN_sq_mod : (N ^ 2) % D = s') :
  (∃ N, s = s') ∧ (∃ N, s < s') :=
by { sorry }

end remainder_relationship_varies_l152_152755


namespace proof_lean_l152_152690

def I : set ℕ := { x | 0 < x ∧ x ≤ 6 }
def P : set ℕ := { x | x ∣ 6 }
def Q : set ℕ := {1, 3, 4, 5}
def C_I_P : set ℕ := I \ P

theorem proof_lean : (C_I_P ∩ Q) = {4, 5} := by
  sorry

end proof_lean_l152_152690


namespace lucy_crayons_correct_l152_152866

-- Define the number of crayons Willy has.
def willyCrayons : ℕ := 5092

-- Define the number of extra crayons Willy has compared to Lucy.
def extraCrayons : ℕ := 1121

-- Define the number of crayons Lucy has.
def lucyCrayons : ℕ := willyCrayons - extraCrayons

-- Statement to prove
theorem lucy_crayons_correct : lucyCrayons = 3971 := 
by
  -- The proof is omitted as per instructions
  sorry

end lucy_crayons_correct_l152_152866


namespace positive_real_solution_of_equation_l152_152232

theorem positive_real_solution_of_equation (y : ℝ) (h1 : 0 < y) (h2 : (y - 6) / 12 = 6 / (y - 12)) : y = 18 :=
by
  sorry

end positive_real_solution_of_equation_l152_152232


namespace visible_factor_numbers_count_100_200_l152_152906

def is_visible_factor_number (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 200 ∧
  (∀ d in (n.digits 10).filter (≠ 0), n % d = 0)

def count_visible_factor_numbers (start : ℕ) (end : ℕ) : ℕ :=
  (List.range' start (end - start + 1)).count is_visible_factor_number

theorem visible_factor_numbers_count_100_200 : count_visible_factor_numbers 100 200 = 19 := 
begin
  sorry
end

end visible_factor_numbers_count_100_200_l152_152906


namespace analytic_expression_of_f_range_of_a_l152_152642

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log (x + 1) / Real.log (1 / 2))
  else Real.log (-x + 1) / Real.log (1 / 2)

theorem analytic_expression_of_f :
  ∀ x : ℝ, (f x = if x > 0 then Real.log (x + 1) / Real.log (1 / 2)
                   else Real.log (-x + 1) / Real.log (1 / 2)) := sorry

theorem range_of_a (a : ℝ) : f (a - 1) < -1 ↔ a ∈ Set.Iio 0 ∪ Set.Ioi 2 :=
begin
  sorry
end

end analytic_expression_of_f_range_of_a_l152_152642


namespace f_2_eq_12_l152_152653

def f : ℝ → ℝ
| x => if x ≤ 0 then 2 ^ x else f (x - 3)

theorem f_2_eq_12 : f 2 = 2⁻¹ := sorry

end f_2_eq_12_l152_152653


namespace sum_f_le_avg_f_sum_f_eq_iff_l152_152102

theorem sum_f_le_avg_f (f : ℝ → ℝ) (a b : ℝ)
  (h : ∀ x1 x2 y1 y2, x1 + x2 = y1 + y2 →
    f x1 + f x2 < f y1 + f y2 ↔ |y1 - y2| < |x1 - x2|)
  (x : ℕ → ℝ) (n : ℕ) (hx : ∀ i, x i ∈ set.Ioo a b) :
  (finset.range n).sum (λ i, f (x i)) ≤ n * (f ((finset.range n).sum (λ i, x i) / n)) :=
sorry

theorem sum_f_eq_iff (f : ℝ → ℝ) (a b : ℝ)
  (h : ∀ x1 x2 y1 y2, x1 + x2 = y1 + y2 →
    f x1 + f x2 < f y1 + f y2 ↔ |y1 - y2| < |x1 - x2|)
  (x : ℕ → ℝ) (n : ℕ) (hx : ∀ i, x i ∈ set.Ioo a b) :
  (finset.range n).sum (λ i, f (x i)) = n * (f ((finset.range n).sum (λ i, x i) / n)) ↔
  ∀ i j, x i = x j :=
sorry

end sum_f_le_avg_f_sum_f_eq_iff_l152_152102


namespace exponents_product_as_cube_l152_152221

theorem exponents_product_as_cube :
  (3^12 * 3^3) = 243^3 :=
sorry

end exponents_product_as_cube_l152_152221


namespace part1_part2_l152_152667

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l152_152667


namespace alpha_minus_beta_eq_neg_pi_over_4_l152_152288

-- Given constants and assumptions
noncomputable def cos_alpha : ℝ := (2 * sqrt 5) / 5
noncomputable def sin_beta : ℝ := (3 * sqrt 10) / 10

-- Assumptions
axiom alpha_acute : (α: ℝ) -> 0 < α ∧ α < π / 2
axiom beta_acute : (β: ℝ) -> 0 < β ∧ β < π / 2
axiom cos_alpha_axiom : cos α = (2 * sqrt 5) / 5
axiom sin_beta_axiom : sin β = (3 * sqrt 10) / 10

-- The statement to be proven
theorem alpha_minus_beta_eq_neg_pi_over_4 (α β : ℝ) (hα : alpha_acute α) (hβ : beta_acute β) : 
  cos α = cos_alpha → 
  sin β = sin_beta → 
  α - β = -π / 4 := 
by 
  -- proof steps will be filled in here
  sorry

end alpha_minus_beta_eq_neg_pi_over_4_l152_152288


namespace quadratic_roots_l152_152825

theorem quadratic_roots : ∀ x : ℝ, x * (x - 2) = 2 - x ↔ (x = 2 ∨ x = -1) := by
  intros
  sorry

end quadratic_roots_l152_152825


namespace freezer_temperature_is_minus_12_l152_152332

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l152_152332


namespace quadratic_discriminant_l152_152571

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant :
  discriminant 6 (6 + 1/6) (1/6) = 1225 / 36 :=
by
  sorry

end quadratic_discriminant_l152_152571


namespace smallest_positive_integer_existence_l152_152753

-- Define the set X with 100 elements
noncomputable def X : Finset ℕ := Finset.range 100

-- Define the smallest positive integer n
noncomputable def smallest_n : ℕ := Nat.choose 102 51 + 1

-- State the proof
theorem smallest_positive_integer_existence :
  ∀ (A : Fin n → Finset ℕ) (hX : ∀ i, A i ⊆ X), 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ smallest_n ∧ 
    (A i ⊆ A j ∧ A j ⊆ A k ∨ A i ⊇ A j ∧ A j ⊇ A k) :=
sorry

end smallest_positive_integer_existence_l152_152753


namespace area_sum_of_squares_l152_152722

theorem area_sum_of_squares (XY YZ : ℕ) (h₁ : XY = 8) (h₂ : YZ = 17) :
  let XZ := int.sqrt (YZ ^ 2 - XY ^ 2)
  let PQRS_area := YZ ^ 2
  let XTUV_area := XZ ^ 2
  PQRS_area + XTUV_area = 514 := by
  -- Definitions
  let XZ := int.sqrt (YZ ^ 2 - XY ^ 2)
  let PQRS_area := YZ ^ 2
  let XTUV_area := XZ ^ 2
  
  -- Proof goal
  show PQRS_area + XTUV_area = 514
  sorry

end area_sum_of_squares_l152_152722


namespace eval_f1_l152_152267

def f : ℤ → ℤ
| x := if x ≥ 6 then x - 5 else f (x + 2)

theorem eval_f1 : f 1 = 2 := sorry

end eval_f1_l152_152267


namespace cross_product_correct_l152_152594

def vec_a : ℝ × ℝ × ℝ := (2, 0, 3)
def vec_b : ℝ × ℝ × ℝ := (5, -1, 7)

def cross_prod (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

theorem cross_product_correct : cross_prod vec_a vec_b = (3, 1, -2) :=
by
  sorry

end cross_product_correct_l152_152594


namespace x_coordinate_of_second_point_l152_152720

variable (m n : ℝ)

theorem x_coordinate_of_second_point
  (h1 : m = 2 * n + 5)
  (h2 : (m + 5) = 2 * (n + 2.5) + 5) :
  (m + 5) = m + 5 :=
by
  sorry

end x_coordinate_of_second_point_l152_152720


namespace angle_of_inclination_range_x_plus_sqrt3y_range_l152_152711

section

variables {α θ t : ℝ}

-- Definitions from the conditions in the problem
def line_parametric_eq (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (-2 + t * Real.cos α, t * Real.sin α)

def curve_polar_eq (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 4 * Real.sin θ)

-- Proof Problem Statement

theorem angle_of_inclination_range (α : ℝ) :
  (∃ t θ, line_parametric_eq t α = curve_polar_eq θ) ↔
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := sorry

theorem x_plus_sqrt3y_range (x y θ : ℝ) :
  (x, y) = curve_polar_eq θ →
  -2 ≤ x + Real.sqrt 3 * y ∧ x + Real.sqrt 3 * y ≤ 6 := sorry

end

end angle_of_inclination_range_x_plus_sqrt3y_range_l152_152711


namespace Jean_spots_l152_152038

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l152_152038


namespace abs_diff_of_solutions_l152_152591

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l152_152591


namespace sticks_per_stool_is_two_l152_152761

-- Conditions
def sticks_from_chair := 6
def sticks_from_table := 9
def sticks_needed_per_hour := 5
def num_chairs := 18
def num_tables := 6
def num_stools := 4
def hours_to_keep_warm := 34

-- Question and Answer in Lean 4 statement
theorem sticks_per_stool_is_two : 
  (hours_to_keep_warm * sticks_needed_per_hour) - (num_chairs * sticks_from_chair + num_tables * sticks_from_table) = 2 * num_stools := 
  by
    sorry

end sticks_per_stool_is_two_l152_152761


namespace election_outcomes_count_l152_152699

theorem election_outcomes_count :
  (∑ (k : ℕ) in finset.range 6, if (k > 0 ∧ k < 5) then (nat.choose 28 k) * (nat.choose 22 (5 - k)) else 0) = 239297520 :=
by sorry

end election_outcomes_count_l152_152699


namespace number_of_plants_needed_l152_152476

theorem number_of_plants_needed (r : ℝ) (d : ℝ) (π_approx : ℝ) 
  (h_r : r = 15) (h_d : d = 0.5) (h_π : π_approx = 3.14159) : 
  Nat :=
let C := 2 * π_approx * r in
let num_plants := C / d in
(num_plants.round.toNat = 188) :=
by
  unfold C num_plants
  sorry

end number_of_plants_needed_l152_152476


namespace players_have_five_coins_l152_152539

noncomputable def game_probability : ℚ :=
  let totalWays := (4.choose 2) ^ 3  -- ways to choose who gets green and red in each round (binomial coefficient 4C2)
  let favorableWays := 1             -- only one way to maintain balance across three rounds
  favorableWays / totalWays

theorem players_have_five_coins :
  game_probability = 1 / 46656 := by
  sorry

end players_have_five_coins_l152_152539


namespace arithmetic_sequence_proof_l152_152627

noncomputable def arithmetic_seq (a : ℕ → ℤ) (a_2_plus_a_6_eq_14 : a 2 + a 6 = 14) (S_5_eq_25 : Σ (n : ℕ), 5 = n → Σ (s : ℤ), ∑ i in finset.range(5), a (i + 1) = s ∧ s = 25) : Prop :=
  ∃ (a1 d : ℤ),
    (a 1 = a1) ∧
      (a 2 = a1 + d) ∧ 
        (a 6 = a1 + 5 * d) ∧
          (2 * a1 + 5 * d = 14) ∧
            (5 * a1 + 10 * d = 25) ∧
              (∀ (n : ℕ), a n = a1 + (n - 1) * d)

noncomputable def sum_seq (a : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) (a_n_formula : ∀ n : ℕ, a n = 2 * n - 1): Prop :=
  ∃ (Tn : ℕ → ℚ),
    (∀ n : ℕ, b n = 2 / (a n * a (n + 1))) ∧
      (T 0 = 0) ∧ 
        (∀ n : ℕ, b n = 1 / (2 * n - 1) - 1 / (2 * n + 1)) ∧
          (∀ n : ℕ, T (n+1) = T n + b (n+1)) ∧ 
            (∀ n : ℕ, T n = 1 - 1 / (2 * n + 1)) ∧ 
              (∀ (n : ℕ),  T n = (2 * n) / (2 * n + 1))

theorem arithmetic_sequence_proof (a S T : ℕ → ℤ) 
  (a_2_plus_a_6_eq_14 : a 2 + a 6 = 14) 
  (S_5_eq_25 : Σ (n : ℕ), 5 = n → Σ (s : ℤ), ∑ i in finset.range(5), a (i + 1) = s ∧ s = 25) :
  (arithmetic_seq a a_2_plus_a_6_eq_14 S_5_eq_25) ∧
    (sum_seq a (λ n, 2 / (a n * a (n + 1))) T sorry): sorry

end arithmetic_sequence_proof_l152_152627


namespace sum_unique_three_digit_numbers_l152_152614

theorem sum_unique_three_digit_numbers : 
  let digits := [1, 3, 5] in
  let three_digit_numbers := 
    digits.bind (λ a, 
      (digits.erase a).bind (λ b, 
        (digits.erase a).erase b).map (λ c, 100 * a + 10 * b + c)) in
  three_digit_numbers.sum = 1998 :=
by
  let digits := [1, 3, 5]
  let three_digit_numbers := 
    digits.bind (λ a, 
      (digits.erase a).bind (λ b, 
        (digits.erase a).erase b).map (λ c, 100 * a + 10 * b + c))
  have : three_digit_numbers = [135, 153, 315, 351, 513, 531] := sorry
  rw this
  unfold List.sum
  norm_num
  sorry

end sum_unique_three_digit_numbers_l152_152614


namespace actual_average_height_l152_152169

theorem actual_average_height
  (n : ℕ) (initial_avg_height wrong_height correct_height : ℕ)
  (h_n : n = 35)
  (h_initial_avg_height : initial_avg_height = 185)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106) :
  let incorrect_total_height := initial_avg_height * n in
  let error := wrong_height - correct_height in
  let correct_total_height := incorrect_total_height - error in
  let actual_avg_height := (correct_total_height : ℚ) / n in
  actual_avg_height ≈ 183.29 :=
by {
  sorry
}

end actual_average_height_l152_152169


namespace similar_triangles_on_circle_l152_152738

theorem similar_triangles_on_circle 
  {α : Type*} [MetricSpace α]
  {C : Set α} (hC : ∃ (O : α) (r : ℝ), ∀ x ∈ C, dist x O = r)
  {A B C D M : α}
  (hA : A ∈ C) (hB : B ∈ C) (hC : C ∈ C) (hD : D ∈ C)
  (hM : M ∈ LineThrough A B) (hM' : M ∈ LineThrough C D) :
  Similar (Triangle.mk M A B) (Triangle.mk M D C) :=
sorry

end similar_triangles_on_circle_l152_152738


namespace calculate_area_of_gray_region_l152_152719

-- Definitions
def radius_inner_circle := r : ℝ
def radius_outer_circle := (radius_inner_circle + 3)
def area_circle (radius : ℝ) := π * radius^2

-- Proving the area of the gray region
theorem calculate_area_of_gray_region (r : ℝ) :
  area_circle radius_outer_circle - area_circle radius_inner_circle = 6 * π * r + 9 * π :=
by
  sorry

end calculate_area_of_gray_region_l152_152719


namespace Mr_A_Mrs_A_are_normal_l152_152174

def is_knight (person : Type) : Prop := sorry
def is_liar (person : Type) : Prop := sorry
def is_normal (person : Type) : Prop := sorry

variable (Mr_A Mrs_A : Type)

axiom Mr_A_statement : is_normal Mrs_A → False
axiom Mrs_A_statement : is_normal Mr_A → False

theorem Mr_A_Mrs_A_are_normal :
  is_normal Mr_A ∧ is_normal Mrs_A :=
sorry

end Mr_A_Mrs_A_are_normal_l152_152174


namespace super12_teams_l152_152349

theorem super12_teams :
  ∃ n : ℕ, (n * (n - 1) = 132) ∧ n = 12 := by
  sorry

end super12_teams_l152_152349


namespace largest_angle_of_pentagon_l152_152851

theorem largest_angle_of_pentagon (A B x E : ℝ)
    (hA : A = 80) (hB : B = 95) (hC : x = 77)
    (hE : E = 3 * x - 10) :
    E = 221 :=
by
  rw [hA, hB, hC, hE]
  norm_num
  sorry

end largest_angle_of_pentagon_l152_152851


namespace calculate_area_l152_152138

noncomputable def area_of_triangle (r R : ℝ) (h : cos B = sin A) : ℝ :=
  let s := (a + b + c) / 2
  in r * s

theorem calculate_area :
  ∀ A B C : ℝ, (cos B = sin A) →
  (r : ℝ) (R : ℝ), (r = 7) → (R = 25) →
  area_of_triangle r R _ = 525 * real.sqrt 2 / 2 :=
begin
  intros A B C h r R hr hR,
  have hr7 : r = 7,
  { exact hr },
  have hR25 : R = 25, 
  { exact hR },
  sorry
end

end calculate_area_l152_152138


namespace number_of_lightsabers_ordered_l152_152158

def cost_per_metal_arc : Nat := 400
def metal_arcs_per_lightsaber : Nat := 2
def assembly_time_per_lightsaber : Nat := 1 / 20
def combined_cost_per_hour : Nat := 200 + 100
def total_cost : Nat := 65200

theorem number_of_lightsabers_ordered (x : Nat) :
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  x * total_cost_per_lightsaber = total_cost →
  x = 80 :=
by
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  assume h : x * total_cost_per_lightsaber = total_cost
  sorry

end number_of_lightsabers_ordered_l152_152158


namespace tangent_line_at_P_eq_l152_152652

noncomputable def f (x : ℝ) := x^3

theorem tangent_line_at_P_eq (f : ℝ → ℝ) :
  (∀ x, f x = x^3) → (∃ m b, (∀ x y : ℝ, y = m * x + b ↔ 3 * x - y - 2 = 0) ∧ 
    (y - 1 = 3 * (x - 1))) :=
by
  intros hfx
  use 3, (-2)
  split
  { intros x y
    split
    sorry
    sorry }
  sorry

end tangent_line_at_P_eq_l152_152652


namespace find_BD_l152_152528

theorem find_BD (AB CD : ℝ) (angle_ACD : ℝ) (distance_CA : ℝ) (radius : ℝ) :
  AB = 1 →
  angle_ACD = 60 →
  distance_CA = √2 →
  radius = 1 →
  ∃ (BD : ℝ), BD = 1 :=
by
  intros h1 h2 h3 h4
  use 1
  sorry

end find_BD_l152_152528


namespace average_remaining_five_l152_152321

theorem average_remaining_five (S S4 S5 : ℕ) 
  (h1 : S = 18 * 9) 
  (h2 : S4 = 8 * 4) 
  (h3 : S5 = S - S4) 
  (h4 : S5 / 5 = 26) : 
  average_of_remaining_5 = 26 :=
by 
  sorry


end average_remaining_five_l152_152321


namespace sum_of_elements_of_T_l152_152739

def is_repeating_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧ x = (10 * a + b) / 99

def sum_of_repeating_decimals : ℝ :=
  ∑ x in { x : ℝ | is_repeating_decimal x }, x

theorem sum_of_elements_of_T : sum_of_repeating_decimals = 45 :=
by
  sorry

end sum_of_elements_of_T_l152_152739


namespace sum_fx_eq_1001_l152_152381

noncomputable def f (x : ℝ) : ℝ := 5 / (7^x + 5)

theorem sum_fx_eq_1001 : (∑ i in finset.range 2002, f ((i+1) / 2003)) = 1001 := by
  sorry

end sum_fx_eq_1001_l152_152381


namespace janet_overtime_multiple_l152_152725

theorem janet_overtime_multiple :
  let hourly_rate := 20
  let weekly_hours := 52
  let regular_hours := 40
  let car_price := 4640
  let weeks_needed := 4
  let normal_weekly_earning := regular_hours * hourly_rate
  let overtime_hours := weekly_hours - regular_hours
  let required_weekly_earning := car_price / weeks_needed
  let overtime_weekly_earning := required_weekly_earning - normal_weekly_earning
  let overtime_rate := overtime_weekly_earning / overtime_hours
  (overtime_rate / hourly_rate = 1.5) :=
by
  sorry

end janet_overtime_multiple_l152_152725


namespace largest_y_value_l152_152835

theorem largest_y_value (x y z : ℤ) (h1 : 1 < z) (h2 : z < y) (h3 : y < x) (h4 : x * y * z = 360) : 
  y ≤ 9 :=
begin
  sorry
end

end largest_y_value_l152_152835


namespace metal_waste_l152_152503

theorem metal_waste (s : ℝ) (r : ℝ := s / 2) (area_square : ℝ := s ^ 2)
  (area_circle : ℝ := π * (s / 2) ^ 2) (a : ℝ := s * sqrt 2 / 2)
  (area_square_from_circle : ℝ := (s * sqrt 2 / 2) ^ 2) :
  (s ^ 2 - (π * (s / 2) ^ 2) + (π * (s / 2) ^ 2 - (s * sqrt 2 / 2) ^ 2)) = (1 / 2) * s ^ 2 :=
by
  sorry

end metal_waste_l152_152503


namespace ratio_card_game_l152_152584

noncomputable def card_game : Real :=
let total_ways := Nat.choose 50 5
let p' := 10 / total_ways
let q' := (90 * Nat.choose 5 4 * Nat.choose 5 1) / total_ways
q' / p' = 225

theorem ratio_card_game : card_game = 225 := by
  sorry

end ratio_card_game_l152_152584


namespace equal_segments_in_triangle_l152_152548

theorem equal_segments_in_triangle
  (A B C O M D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace O] [MetricSpace M] [MetricSpace D] [MetricSpace E]
  (triangle_ABC : Triangle A B C)
  (acute_ABC : AcuteTriangle A B C)
  (AB_lt_AC : Distance A B < Distance A C)
  (O_circumcenter : Circumcenter O A B C)
  (M_midpoint_BC : Midpoint M B C)
  (circ_AOM : Circle A O M)
  (D_on_ext_AB : OnExtension D B A)
  (E_on_seg_AC : OnSegment E A C)
  (circ_AOM_intersects_ext_AB_at_D : Intersects circ_AOM (Extension A B) D)
  (circ_AOM_intersects_seg_AC_at_E : Intersects circ_AOM (Segment A C) E)
  : Distance D M = Distance E C := sorry

end equal_segments_in_triangle_l152_152548


namespace xyz_value_l152_152385

-- Define the variables and conditions
variables {x y z : ℂ}

-- Define the conditions of the problem.
def condition1 := x * y + 5 * y = -20
def condition2 := y * z + 5 * z = -20
def condition3 := z * x + 5 * x = -25

-- State the theorem to be proved
theorem xyz_value (h1 : condition1) (h2 : condition2) (h3 : condition3) : x * y * z = (260 : ℂ) / 3 := by
  sorry

end xyz_value_l152_152385


namespace total_time_for_journey_l152_152533

theorem total_time_for_journey (x : ℝ) : 
  let time_first_part := x / 50
  let time_second_part := 3 * x / 80
  time_first_part + time_second_part = 23 * x / 400 :=
by 
  sorry

end total_time_for_journey_l152_152533


namespace desired_antifreeze_pct_in_colder_climates_l152_152190

-- Definitions for initial conditions
def initial_antifreeze_pct : ℝ := 0.10
def radiator_volume : ℝ := 4
def drained_volume : ℝ := 2.2857
def replacement_antifreeze_pct : ℝ := 0.80

-- Proof goal: Desired percentage of antifreeze in the mixture is 50%
theorem desired_antifreeze_pct_in_colder_climates :
  (drained_volume * replacement_antifreeze_pct + (radiator_volume - drained_volume) * initial_antifreeze_pct) / radiator_volume = 0.50 :=
by
  sorry

end desired_antifreeze_pct_in_colder_climates_l152_152190


namespace color_drawing_cost_l152_152367

theorem color_drawing_cost (cost_bw : ℕ) (surcharge_ratio : ℚ) (cost_color : ℕ) :
  cost_bw = 160 →
  surcharge_ratio = 0.50 →
  cost_color = cost_bw + (surcharge_ratio * cost_bw : ℚ).natAbs →
  cost_color = 240 :=
by
  intros h_bw h_surcharge h_color
  rw [h_bw, h_surcharge] at h_color
  exact h_color

end color_drawing_cost_l152_152367


namespace cos_double_angle_l152_152283

theorem cos_double_angle (α : ℝ) (h : Real.sin ((Real.pi / 6) + α) = 1 / 3) :
  Real.cos ((2 * Real.pi / 3) - 2 * α) = -7 / 9 := by
  sorry

end cos_double_angle_l152_152283


namespace triangle_no_two_obtuse_angles_l152_152861

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l152_152861


namespace probability_10_products_expected_value_of_products_l152_152501

open ProbabilityTheory

/-- Probability calculations for worker assessment. -/
noncomputable def worker_assessment_probability (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p^9 * (10 - 9*p)

/-- Expected value of total products produced and debugged by Worker A -/
noncomputable def expected_products (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  20 - 10*p - 10*p^9 + 10*p^10

/-- Theorem 1: Prove that the probability that Worker A ends the assessment by producing only 10 products is p^9(10 - 9p). -/
theorem probability_10_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  worker_assessment_probability p h = p^9 * (10 - 9*p) := by
  sorry

/-- Theorem 2: Prove the expected value E(X) of the total number of products produced and debugged by Worker A is 20 - 10p - 10p^9 + 10p^{10}. -/
theorem expected_value_of_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  expected_products p h = 20 - 10*p - 10*p^9 + 10*p^10 := by
  sorry

end probability_10_products_expected_value_of_products_l152_152501


namespace probability_of_multiples_l152_152071
open Set

/-- Define the set of cards numbered 1 to 200 -/
def cards : Finset ℕ := Finset.range 201 \ {0}

/-- Define the multiples of a given number n up to 200 -/
def multiples (n : ℕ) : Finset ℕ := (Finset.range (200 / n + 1)).image (λ k, n * k)

/-- Number of elements in the union of sets using Inclusion-Exclusion Principle -/
def multiples_union_size : ℕ :=
  (multiples 2 ∪ multiples 3 ∪ multiples 5 ∪ multiples 7).card

theorem probability_of_multiples :
    (multiples_union_size / 200 : ℚ) = 153 / 200 := sorry

end probability_of_multiples_l152_152071


namespace range_of_a_l152_152305

variable (a : ℝ)
def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq_or : p a ∨ q a) (hpq_and_false : ¬ (p a ∧ q a)) : 
    a ∈ Set.Iio 0 ∪ Set.Ioo (1/4) 4 :=
by
  sorry

end range_of_a_l152_152305


namespace distance_between_trees_l152_152167

theorem distance_between_trees (num_trees: ℕ) (total_length: ℕ) (trees_at_end: ℕ) 
(h1: num_trees = 26) (h2: total_length = 300) (h3: trees_at_end = 2) :
  total_length / (num_trees - 1) = 12 :=
by sorry

end distance_between_trees_l152_152167


namespace xena_head_start_l152_152867

theorem xena_head_start
  (xena_speed : ℝ) (dragon_speed : ℝ) (time : ℝ) (burn_distance : ℝ) 
  (xena_speed_eq : xena_speed = 15) 
  (dragon_speed_eq : dragon_speed = 30) 
  (time_eq : time = 32) 
  (burn_distance_eq : burn_distance = 120) :
  (dragon_speed * time - burn_distance) - (xena_speed * time) = 360 := 
  by 
  sorry

end xena_head_start_l152_152867


namespace area_of_A2B2C2_eq_l152_152021

noncomputable def triangle_area_A2B2C2 {α β γ : ℝ} (φ : ℝ) (t : ℝ) : ℝ :=
  4 * (Real.cos φ)^2 * t

theorem area_of_A2B2C2_eq
  (A B C A1 B1 C1 A2 B2 C2 : Type)
  (hABC : Triangle A B C)
  (hA1 : A1 ∈ LineBC B C)
  (hB1 : B1 ∈ LineCA C A)
  (hC1 : C1 ∈ LineAB A B)
  (hφ : ∀ {ABC : Triangle}, φ = AngleCC1B ∧ φ = AngleAA1C ∧ φ = AngleBB1A)
  (h_interC2 : C2 = intersect (Line AA1) (Line BB1))
  (h_interA2 : A2 = intersect (Line BB1) (Line CC1))
  (h_interB2 : B2 = intersect (Line CC1) (Line AA1))
  (h_acute : φ > α ∧ φ > β ∧ φ > γ)
  (h_acute_triangle : α + β + γ = π - ϕ) :
  triangle_area A2 B2 C2 = 4 * (cos φ)^2 * triangle_area A B C :=
sorry

end area_of_A2B2C2_eq_l152_152021


namespace number_of_valid_n_digit_numbers_l152_152254

noncomputable def valid_n_digit_numbers (n : ℕ) : ℕ :=
  (2 / 5 : ℚ) * (9^n - (-1)^n)

theorem number_of_valid_n_digit_numbers (n : ℕ) (h : 0 < n) :
  valid_n_digit_numbers n = (2 / 5 : ℚ) * (9^n - (-1)^n) :=
by sorry

end number_of_valid_n_digit_numbers_l152_152254


namespace value_of_f_tan_sq_l152_152994

theorem value_of_f_tan_sq (t : ℝ) (h : 0 ≤ t ∧ t ≤ π / 4) :
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → f (x / (x - 1)) = 1 / x →
  f (tan t ^ 2) = tan t ^ 2 - 1 := by
  sorry

end value_of_f_tan_sq_l152_152994


namespace f_neg_three_halves_f_a_equals_4_l152_152266

noncomputable def f : ℝ → ℝ
| x => if -2 < x ∧ x < 0 then f (x + 1)
       else if 0 ≤ x ∧ x < 2 then 2 * x + 1
       else if x ≥ 2 then x^2 - 1
       else 0  -- we use else 0 for non-covered cases by the domain specification
  
theorem f_neg_three_halves : f (-3/2) = 2 := 
by sorry

theorem f_a_equals_4 (a : ℝ) (h1 : f a = 4) (h2 : 0 < a) : a = 3/2 ∨ a = Real.sqrt 5 :=
by sorry

end f_neg_three_halves_f_a_equals_4_l152_152266


namespace find_k_l152_152664

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l152_152664


namespace factorization_multiplication_l152_152486

-- Define variables
variables (x y : ℝ)

-- Problem 1: Factorize x^5 - x^3 * y^2
theorem factorization (x y : ℝ) : x^5 - x^3 * y^2 = x^3 * (x - y) * (x + y) :=
by
  sorry

-- Problem 2: Calculate (-2 * x^3 * y^2) * (3 * x^2 * y)
theorem multiplication (x y : ℝ) : (-2 * x^3 * y^2) * (3 * x^2 * y) = -6 * x^5 * y^3 :=
by
  sorry

end factorization_multiplication_l152_152486


namespace find_y_l152_152319

theorem find_y (y : ℕ) (hy_mult_of_7 : ∃ k, y = 7 * k) (hy_pos : 0 < y) (hy_square : y^2 > 225) (hy_upper_bound : y < 30) : y = 21 :=
sorry

end find_y_l152_152319


namespace proof_problem_l152_152796

-- Define the conditions
variables {O : ℝ × ℝ} {C : set (ℝ × ℝ)} {f1 f2 : ℝ × ℝ} {l : ℝ × ℝ → Prop} {P A B : ℝ × ℝ}
variables {m : ℝ} {λ : ℝ}
def is_center (O : ℝ × ℝ) (C : set (ℝ × ℝ)) := O = (0, 0)
def on_y_axis (f1 f2 : ℝ × ℝ) := f1.1 = 0 ∧ f2.1 = 0
def distance_directrix_foci_eccentricity (f1 f2 : ℝ × ℝ) := 
  dist f1 (0, 1) = dist f2 (0, 1) = √2 / 2 ∧ dist f1 (0, 0) / 1 = √2 / 2
def intersects_y_axis (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) := l P ∧ P.1 = 0
def intersects_ellipse_distinct (l : ℝ × ℝ → Prop) (C : set (ℝ × ℝ)) := ∃ A B, l A ∧ l B ∧ A ≠ B
def AP_equal_lambda_PB (A B P : ℝ × ℝ) (λ : ℝ) := A = (1 - λ) • P + λ • B
def OA_lambda_OB_equal_4_OP (O A B P : ℝ × ℝ) (λ : ℝ) := O + λ • B = 4 • P

-- Proof problem statement
theorem proof_problem
  (O : ℝ × ℝ) (C : set (ℝ × ℝ)) (f1 f2 : ℝ × ℝ) (l : ℝ × ℝ → Prop) (P A B : ℝ × ℝ) (m λ : ℝ)
  (h_center : is_center O C)
  (h_foci_y_axis : on_y_axis f1 f2)
  (h_eccentricity : distance_directrix_foci_eccentricity f1 f2)
  (h_intersects_y_axis : intersects_y_axis l P)
  (h_intersects_ellipse : intersects_ellipse_distinct l C)
  (h_AP_equal_lambda_PB : AP_equal_lambda_PB A B P λ)
  (h_OA_lambda_OB : OA_lambda_OB_equal_4_OP O A B P λ) :
  (∀ x y, C (x, y) ↔ y^2 + 2*x^2 = 1) ∧ ((-1 < m ∧ m < -1/2) ∨ (1/2 < m ∧ m < 1)) :=
sorry

end proof_problem_l152_152796


namespace infinite_planes_parallel_to_line_l152_152137

-- Definitions for the given problem
variables (l : ℝ^3 → Prop) (P : ℝ^3) [Point P] [Line l]

-- Theorem statement encapsulating the problem
theorem infinite_planes_parallel_to_line (P : ℝ^3) (l : ℝ^3 → Prop) :
  (∀ (P : ℝ^3), ¬l P) → 
  ∃ (planes : ℝ^3 → (ℝ^3 → Prop)), 
    (∀ β, ∃ PQ : ℝ^3 → Prop, (PQ ∈ β) ∧ is_parallel_to l PQ) ∧ 
    is_parallel_to_l : ∃ β, is_parallel_to l β ∧ infinite (set_of (λ β : ℝ^3 → Prop, is_parallel_to l β)) :=
begin
  sorry
end

end infinite_planes_parallel_to_line_l152_152137


namespace area_ratio_ge_two_l152_152017

-- Define the areas of triangles ABC and AKL
variables (E E1 : ℝ)

-- Define that ABC is a right-angled triangle, with altitude AD to hypotenuse BC
variables (A B C D K L : Type) 
  [right_triangle A B C]
  (ABC_altitude_AD : altitude D (triangle A B C))
  (int_circle_join_ABD_ACD : line (incenter A B D) (incenter A C D)
    ∩ sides AB AC = {K, L})

-- The main theorem that we want to prove
theorem area_ratio_ge_two (h : triangle A B C)
  (h2 : triangle (A K L)) : 
  (E / E1) ≥ 2 :=
sorry

end area_ratio_ge_two_l152_152017


namespace find_second_dimension_l152_152342

theorem find_second_dimension (x : ℕ) 
    (h1 : 12 * x * 16 / (3 * 7 * 2) = 64) : 
    x = 14 := by
    sorry

end find_second_dimension_l152_152342


namespace magic_show_l152_152512

theorem magic_show (performances : ℕ) (prob_never_reappear : ℚ) (prob_two_reappear : ℚ)
  (h_performances : performances = 100)
  (h_prob_never_reappear : prob_never_reappear = 1 / 10)
  (h_prob_two_reappear : prob_two_reappear = 1 / 5) :
  let never_reappear := prob_never_reappear * performances,
      two_reappear := prob_two_reappear * performances,
      normal_reappear := performances,
      extra_reappear := two_reappear,
      total_reappear := normal_reappear + extra_reappear - never_reappear in
  total_reappear = 110 := by
  sorry

end magic_show_l152_152512


namespace largest_possible_median_l152_152579

noncomputable theory

-- Define the given eight positive integers as a set
def given_numbers : set ℕ := {3, 4, 5, 6, 7, 8, 9, 10}

-- Eleven positive integers including the above eight and three additional ones that are all greater than 10
def integers_with_additional (extra_numbers : set ℕ) : set ℕ := given_numbers ∪ extra_numbers

-- Prove that the largest possible value of the median when these numbers are combined is 8
theorem largest_possible_median (extra_numbers : set ℕ) (h : ∀ x ∈ extra_numbers, x > 10) :
  (sorted_list_median (integrated_list given_numbers extra_numbers)) = 8 :=
begin
  -- Proof is provided here
  sorry
end

end largest_possible_median_l152_152579


namespace seven_digit_number_divisible_by_4_probability_l152_152803

open Finite

theorem seven_digit_number_divisible_by_4_probability :
  let digits := {0, 1, 2, 3, 4, 5, 6}
  let total_numbers := Nat.factorial 7
  let valid_endings := {{0, 4}, {1, 2}, {1, 6}, {2, 0}, {2, 4}, {3, 2}, {3, 6}, {4, 0}, {5, 2}, {5, 6}, {6, 0}, {6, 4}}
  let valid_numbers := 480 + 768
  let probability := valid_numbers / total_numbers
  probability = 1 / 4 :=
by
  sorry

end seven_digit_number_divisible_by_4_probability_l152_152803


namespace product_of_four_primes_sum_of_squares_eq_476_l152_152194

/-- Definition of the problem conditions and proof goal --/
theorem product_of_four_primes_sum_of_squares_eq_476 :
  ∃ (p1 p2 p3 p4 : ℕ), (nat.prime p1) ∧ (nat.prime p2) ∧ (nat.prime p3) ∧ (nat.prime p4) ∧ 
  (p1^2 + p2^2 + p3^2 + p4^2 = 476) ∧ (p1 * p2 * p3 * p4 = 1989) := by
  sorry

end product_of_four_primes_sum_of_squares_eq_476_l152_152194


namespace problem_l152_152024

variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Define the acute triangle and constructions
axiom acute_triangle (ABC : Type) : Prop
axiom circumcircle (ABC : Type) (A1 B1 C1 : Type) : Prop
axiom extended_angle_bisectors (ABC : Type) (A0 B0 C0 : Type) : Prop

-- Define the points according to the problem statement
axiom intersections_A0 (ABC : Type) (A0 : Type) : Prop
axiom intersections_B0 (ABC : Type) (B0 : Type) : Prop
axiom intersections_C0 (ABC : Type) (C0 : Type) : Prop

-- Define the areas of triangles and hexagon
axiom area_triangle_A0B0C0 (ABC : Type) (A0 B0 C0 : Type) : ℝ
axiom area_hexagon_AC1B_A1CB1 (ABC : Type) (A1 B1 C1 : Type) : ℝ
axiom area_triangle_ABC (ABC : Type) : ℝ

-- Problem: Prove the area relationships
theorem problem
  (ABC: Type)
  (h1 : acute_triangle ABC)
  (h2 : circumcircle ABC A1 B1 C1)
  (h3 : extended_angle_bisectors ABC A0 B0 C0)
  (h4 : intersections_A0 ABC A0)
  (h5 : intersections_B0 ABC B0)
  (h6 : intersections_C0 ABC C0):
  area_triangle_A0B0C0 ABC A0 B0 C0 = 2 * area_hexagon_AC1B_A1CB1 ABC A1 B1 C1 ∧
  area_triangle_A0B0C0 ABC A0 B0 C0 ≥ 4 * area_triangle_ABC ABC :=
sorry

end problem_l152_152024


namespace correct_option_a_l152_152309

-- Define lines in a three-dimensional space
variables (l1 l2 l3 : ℝ^3 → Prop)

-- Define perpendicular relationship between lines
def perp (l1 l2 : ℝ^3 → Prop) : Prop := ∀ v1 v2, (l1 v1) ∧ (l2 v2) → v1 ⬝ v2 = 0

-- Define parallel relationship between lines
def para (l1 l2 : ℝ^3 → Prop) : Prop := ∃ k : ℝ, ∀ v1 v2, (l1 v1) ∧ (l2 v2) → v1 = k • v2

theorem correct_option_a (h1 : perp l1 l2) (h2 : para l2 l3) : perp l1 l3 :=
sorry

end correct_option_a_l152_152309


namespace distance_from_point_to_line_unique_intersection_value_of_a_l152_152181

-- Part (Ⅰ)
theorem distance_from_point_to_line :
  let P := (2, 11 * Real.pi / 6)
  let line_equation := fun (ρ θ : ℝ) => ρ * Real.sin (θ - Real.pi / 6) = 0
  shows Real.dist (sqrt 3, -1) (fun (x y : ℝ) => x - sqrt 3 * y = 0) = sqrt 3 :=
sorry

-- Part (Ⅱ)
theorem unique_intersection_value_of_a :
  (∀ a t : ℝ, (1 + a^2) * t^2 - 4 * a * t + 3 = 0 → x^2 + y^2 = 1 → (4 * a)^2 - 12 * (1 + a^2) = 0) → a = sqrt 3 ∨ a = -sqrt 3 :=
sorry

end distance_from_point_to_line_unique_intersection_value_of_a_l152_152181


namespace jill_jam_initial_weight_l152_152362

-- Define the problem parameters and conditions
def initial_weight_jam (x : ℝ) : Prop :=
  let jam_given_to_jan := (1/6) * x in
  let remaining_after_jan := x - jam_given_to_jan in
  let jam_given_to_jas := (1/13) * remaining_after_jan in
  let remaining_after_jas := remaining_after_jan - jam_given_to_jas in
  remaining_after_jas = 1

-- State the theorem to be proved
theorem jill_jam_initial_weight :
  ∃ x : ℝ, initial_weight_jam x ∧ x = 1.3 :=
by
  sorry

end jill_jam_initial_weight_l152_152362


namespace pairwise_sum_mod_p_l152_152748

theorem pairwise_sum_mod_p {p : ℕ} (hp : prime p) (hp_gt_two : p > 2)
  {a b c d : ℤ} (hna : ¬ p ∣ a) (hnb : ¬ p ∣ b) (hnc : ¬ p ∣ c) (hnd : ¬ p ∣ d)
  (hfractions : ∀ r : ℤ, ¬ p ∣ r → 
    (frac (r * a / p) + frac (r * b / p) + frac (r * c / p) + frac (r * d / p) = 2)) :
  ∃ x y ∈ {a + b, a + c, a + d, b + c, b + d, c + d}, p ∣ x ∧ p ∣ y :=
sorry

end pairwise_sum_mod_p_l152_152748


namespace rank_kinetic_energies_l152_152535

def moment_of_inertia_disk (M R : ℝ) : ℝ := (1 / 2) * M * R^2
def moment_of_inertia_hoop (M R : ℝ) : ℝ := M * R^2
def moment_of_inertia_sphere (M R : ℝ) : ℝ := (2 / 5) * M * R^2

def torque (F R : ℝ) : ℝ := F * R

def angular_acceleration (τ I : ℝ) : ℝ := τ / I

def angular_velocity (α t : ℝ) : ℝ := α * t

def kinetic_energy (I ω : ℝ) : ℝ := 1 / 2 * I * ω^2

theorem rank_kinetic_energies 
  (M R F t : ℝ) 
  (I_disk : ℝ := moment_of_inertia_disk M R)
  (I_hoop : ℝ := moment_of_inertia_hoop M R)
  (I_sphere : ℝ := moment_of_inertia_sphere M R)
  (τ : ℝ := torque F R)
  (α_disk : ℝ := angular_acceleration τ I_disk)
  (α_hoop : ℝ := angular_acceleration τ I_hoop)
  (α_sphere : ℝ := angular_acceleration τ I_sphere)
  (ω_disk : ℝ := angular_velocity α_disk t)
  (ω_hoop : ℝ := angular_velocity α_hoop t)
  (ω_sphere : ℝ := angular_velocity α_sphere t)
  (KE_disk : ℝ := kinetic_energy I_disk ω_disk)
  (KE_hoop : ℝ := kinetic_energy I_hoop ω_hoop)
  (KE_sphere : ℝ := kinetic_energy I_sphere ω_sphere) :
  KE_hoop < KE_disk ∧ KE_disk < KE_sphere := by
  sorry

end rank_kinetic_energies_l152_152535


namespace evaluate_simplified_expression_l152_152581

theorem evaluate_simplified_expression :
  64^(-1/6 : ℝ) + 81^(-1/4 : ℝ) = 5/6 := 
by
  sorry

end evaluate_simplified_expression_l152_152581


namespace decompose_number_4705_l152_152492

theorem decompose_number_4705 :
  4.705 = 4 * 1 + 7 * 0.1 + 0 * 0.01 + 5 * 0.001 := by
  sorry

end decompose_number_4705_l152_152492


namespace area_of_triangle_is_correct_l152_152945

-- Definitions based on the conditions
def side_length_square_ABCD : ℝ := 8
def side_length_square_DEFG : ℝ := 5

def area_triangle_ACF (a b : ℝ) : ℝ :=
  let AC := real.sqrt (a * a + a * a)
  let DF := real.sqrt (b * b + b * b)
  let Base := AC + DF
  let Height := 13 -- based on geometric approximations given in the problem set.
  1/2 * Base * Height

-- Mathematically equivalent proof problem
theorem area_of_triangle_is_correct :
  area_triangle_ACF side_length_square_ABCD side_length_square_DEFG = 72 := sorry

end area_of_triangle_is_correct_l152_152945


namespace heavy_water_electrons_l152_152762

def Avogadro_constant : ℝ := 6.02214076e23

def D2O_electrons_per_molecule : ℝ := 10

theorem heavy_water_electrons
  (mass : ℝ) 
  (molecular_mass_D2O : ℝ) 
  (N_A : ℝ) 
  (D2O_electrons_per_molecule : ℝ) 
  (moles : ℝ) 
  (total_electrons : ℝ) :
  mass = 20 ∧ molecular_mass_D2O = 20 ∧ N_A = Avogadro_constant ∧ moles = mass / molecular_mass_D2O 
  → total_electrons = moles * N_A * D2O_electrons_per_molecule :=
begin
  sorry
end

end heavy_water_electrons_l152_152762


namespace angle_bisector_perpendicular_l152_152079

open Set

variables {A B C D K L : Point}
variables (parallelogram : parallelogram A B C D) (hK : K ∈ segment B C) (hL : L ∈ segment C D)
variables (h_eq : dist A B + dist B K = dist A D + dist D L)

theorem angle_bisector_perpendicular (parallelogram : parallelogram A B C D) (hK : K ∈ segment B C) (hL : L ∈ segment C D)
    (h_eq : dist A B + dist B K = dist A D + dist D L) :
    ∃ O, is_bisector_of_angle O A B D ∧ is_perpendicular (line_through O (intersection_point K L)) (line_through K L) :=
sorry

end angle_bisector_perpendicular_l152_152079


namespace neg_universal_proposition_l152_152433

variable (x : ℝ)
def P (x : ℝ) : Prop := x^2 + sin x + 1 < 0

theorem neg_universal_proposition :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end neg_universal_proposition_l152_152433


namespace gilda_marbles_percentage_left_l152_152260

theorem gilda_marbles_percentage_left (M : ℝ) :
  let after_pedro := M - 0.30 * M,
    after_ebony := after_pedro - 0.15 * after_pedro,
    after_jimmy := after_ebony - 0.20 * after_ebony,
    after_lara := after_jimmy - 0.10 * after_jimmy in
  after_lara / M * 100 = 42.84 :=
by
  sorry

end gilda_marbles_percentage_left_l152_152260


namespace jessica_digit_sum_l152_152046

theorem jessica_digit_sum :
  let initial_sequence := λ n, (n % 6) + 1
  let after_first_erasure := list.filter (λ n, n % 2 ≠ 1) (list.range 12000).map initial_sequence
  let after_second_erasure := list.filter (λ n, n % 3 ≠ 2) after_first_erasure
  let final_sequence := list.filter (λ n, n % 5 ≠ 4) after_second_erasure
  let digit_3047 := final_sequence 3046 % 4 + 1
  let digit_3048 := final_sequence 3047 % 4 + 1
  let digit_3049 := final_sequence 3048 % 4 + 1
  in digit_3047 + digit_3048 + digit_3049 = 5 :=
sorry

end jessica_digit_sum_l152_152046


namespace probability_of_diff_by_three_is_one_eighth_l152_152928

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l152_152928


namespace trapezoid_similar_triangles_product_l152_152030

variables {A B C D X : Type} [trapezoid A B C D] (AB AD AX DX : ℝ)
-- Define the lengths of the sides.
variable (AB_length : AB = 6)
-- Define the similarity conditions of the triangles ABX, BXC, and CXD.

theorem trapezoid_similar_triangles_product :
  ∃ X : A.D, (X ∈ AD) ∧ (AX * DX = 36) :=
sorry

end trapezoid_similar_triangles_product_l152_152030


namespace lcm_two_numbers_l152_152484

theorem lcm_two_numbers
  (a b : ℕ)
  (hcf_ab : Nat.gcd a b = 20)
  (product_ab : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_two_numbers_l152_152484


namespace smallest_int_with_six_factors_l152_152859

theorem smallest_int_with_six_factors : ∃ n : ℕ, n > 0 ∧ (factors n).length = 6 ∧ (∀ m : ℕ, (m > 0 ∧ (factors m).length = 6) → n ≤ m) ∧ n = 12 :=
begin
  sorry
end

end smallest_int_with_six_factors_l152_152859


namespace exists_program_l152_152903
open Function

structure Maze (n : ℕ) :=
  (walls : list (ℕ × ℕ)) -- list of wall positions

structure Robot :=
  (x : ℕ)
  (y : ℕ)

inductive Command
| L | R | U | D

structure State :=
  (robot : Robot)
  (commands : list Command)

def transition (s : State) (m : Maze 10) : State :=
  sorry -- Function definition to handle robot movement

theorem exists_program : ∀ (m : Maze 10) (s : State),
  ∃ (π : list Command), -- This represents the program Π_N
  ∀ p : State, reachable (p.robot.x, p.robot.y) (s.robot.x, s.robot.y) m →
  ∀ i j, reachable (i, j) (p.robot.x, p.robot.y) m → -- All cells reachable
  transition s m = {robot := ⟨i, j⟩, commands := π} :=
by
  sorry

end exists_program_l152_152903


namespace polynomial_expression_value_l152_152262

theorem polynomial_expression_value (a : ℕ → ℤ) (x : ℤ) :
  (x + 2)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 →
  ((a 1 + 3 * a 3 + 5 * a 5 + 7 * a 7 + 9 * a 9)^2 - (2 * a 2 + 4 * a 4 + 6 * a 6 + 8 * a 8)^2) = 3^12 :=
by
  sorry

end polynomial_expression_value_l152_152262


namespace square_field_area_l152_152790

theorem square_field_area (s : ℝ) (P : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  (P = 4 * s) →
  (distance = speed * time) →
  (distance = 96) →
  (speed = 12) →
  (time = 8) →
  (distance = P) →
  (s = 24) →
  (s^2 = 576) :=
by
  intros hP hdist heq1 hspeed htime heqd hp hs
  sorry

end square_field_area_l152_152790


namespace abs_diff_of_solutions_l152_152593

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l152_152593


namespace cube_surface_area_l152_152531

-- Definitions for the problem
def cube (x : ℝ) := { vertices : List (ℝ × ℝ × ℝ) // 
  let A := (0, 0, 0);
  let B := (x, 0, 0);
  let C := (x, x, 0);
  let D := (0, x, 0);
  let A1 := (0, 0, x);
  let B1 := (x, 0, x);
  let C1 := (x, x, x);
  let D1 := (0, x, x);
  let M := ((0 + x)/2, (x + x)/2, (0 + x)/2);
  vertices = [A, B, C, D, A1, B1, C1, D1, M] }

def sphere_radius : ℝ := sqrt 41

noncomputable def is_on_sphere (center radius : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : Prop :=
  let (cx, cy, cz) := center
  let (px, py, pz) := point
  (px - cx)^2 + (py - cy)^2 + (pz - cz)^2 = radius^2

-- Sphere passes through the given points of the cube
def passes_through_vertices (center : ℝ × ℝ × ℝ) (radius : ℝ) (vertices : List (ℝ × ℝ × ℝ)) : Prop :=
  ∀ vertex ∈ vertices, is_on_sphere center radius vertex

-- Proof problem
theorem cube_surface_area
  (x : ℝ)
  (H : ∀ (center : ℝ × ℝ × ℝ), passes_through_vertices center sphere_radius (cube x).val):
  6 * x^2 = 384 :=
sorry

end cube_surface_area_l152_152531


namespace sum_of_rep_decimals_l152_152742

open scoped BigOperators

def rep_decimals_set : set ℝ :=
  { x | ∃ a b : ℕ, a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * (a : ℝ) + (b : ℝ)) / 99 }

theorem sum_of_rep_decimals : ∑ x in rep_decimals_set, x = 90 / 11 :=
sorry

end sum_of_rep_decimals_l152_152742


namespace least_integer_square_l152_152144

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l152_152144


namespace smallest_c_l152_152970

theorem smallest_c (c : ℝ) (h : 0 < c) : 
  (∀ x : ℝ, 3 * sin (5 * x + c) + 1 ≤ 3 * sin c + 1) ∧
  (3 * sin (5 * 0 + c) + 1 = 3 * sin c + 1) → 
  c = (1 : ℝ) * (π / 2) :=
by
  sorry

end smallest_c_l152_152970


namespace triangle_area_right_angled_l152_152094

theorem triangle_area_right_angled (a : ℝ) (h₁ : 0 < a) (h₂ : a < 24) :
  let b := 24
  let c := 48 - a
  (a^2 + b^2 = c^2) → (1/2) * a * b = 216 :=
by
  sorry

end triangle_area_right_angled_l152_152094


namespace point_P0_coordinates_l152_152834

noncomputable def f : ℝ → ℝ := λ x, x^3 + x - 2

theorem point_P0_coordinates :
  ∃ (x0 : ℝ), (f' x0) = 4 ∧ (x0 = 1 ∧ f x0 = 0 ∨ x0 = -1 ∧ f x0 = -4) :=
by sorry

end point_P0_coordinates_l152_152834


namespace range_of_a_l152_152322

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x ^ 2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by
  sorry

end range_of_a_l152_152322


namespace exercise_l152_152264

open Set

theorem exercise (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 5}) (hB : B = {2, 4, 5}) :
  A ∩ (U \ B) = {1, 3} := by
  sorry

end exercise_l152_152264


namespace sum_of_numerical_coefficients_binomial_l152_152449

theorem sum_of_numerical_coefficients_binomial (a b : ℕ) (n : ℕ) (h : n = 8) :
  let sum_num_coeff := (a + b)^n
  sum_num_coeff = 256 :=
by 
  sorry

end sum_of_numerical_coefficients_binomial_l152_152449


namespace incircle_tangency_relationship_l152_152809

-- Define a triangle with vertices A, B, C
variable {A B C : Type} [metric_space A] [metric_space B] [metric_space C] (triangle : (A × B × C))

-- Define points E, F, G, H based on the given conditions
variables {E F G H : Type} [metric_space E] [metric_space F] [metric_space G] [metric_space H]
           (E_point : E) (F_midpoint : F) (G_bisector : G) (H_altitude : H)

-- Problem statement in Lean: Prove the equality given the conditions
theorem incircle_tangency_relationship :
  let E := E_point,
      F := F_midpoint,
      G := G_bisector,
      H := H_altitude in
  (E.distance G * F.distance H = E.distance F * E.distance H) :=
sorry

end incircle_tangency_relationship_l152_152809


namespace max_distinct_points_proof_l152_152678

noncomputable def max_distinct_reflected_points (A B C : Point) (angles : A.angle B C = 30 ∧ B.angle A C = 45 ∧ C.angle A B = 105) : ℕ :=
  12

theorem max_distinct_points_proof (A B C : Point) (h : A.angle B C = 30 ∧ B.angle A C = 45 ∧ C.angle A B = 105) :
  max_distinct_reflected_points A B C h = 12 :=
  sorry

end max_distinct_points_proof_l152_152678


namespace problem_statement_l152_152601

def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
-- This is a placeholder for the prime factorization function
sorry

def lambda (n : ℕ) : ℤ :=
  let factors := prime_factorization n in
  (-1) ^ (factors.map (fun pair => pair.2)).sum

def L (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum (fun x => lambda x)

def is_composite (x : ℕ) : Bool :=
  x > 1 ∧ ¬Nat.prime x

def K (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).filter is_composite.sum (fun x => lambda x)

theorem problem_statement (N : ℕ) (hN : N > 1) (hL : ∀ n, 2 ≤ n → n ≤ N → L n ≤ 0) :
  ∀ n, 2 ≤ n → n ≤ N → K n ≥ 0 := 
sorry

end problem_statement_l152_152601


namespace total_tour_time_l152_152759

-- Declare constants for distances
def distance1 : ℝ := 55
def distance2 : ℝ := 40
def distance3 : ℝ := 70
def extra_miles : ℝ := 10

-- Declare constants for speeds
def speed1_part1 : ℝ := 60
def speed1_part2 : ℝ := 40
def speed2 : ℝ := 45
def speed3_part1 : ℝ := 45
def speed3_part2 : ℝ := 35
def speed3_part3 : ℝ := 50
def return_speed : ℝ := 55

-- Declare constants for stop times
def stop1 : ℝ := 1
def stop2 : ℝ := 1.5
def stop3 : ℝ := 2

-- Prove the total time required for the tour
theorem total_tour_time :
  (30 / speed1_part1) + (25 / speed1_part2) + stop1 +
  (distance2 / speed2) + stop2 +
  (20 / speed3_part1) + (30 / speed3_part2) + (20 / speed3_part3) + stop3 +
  ((distance1 + distance2 + distance3 + extra_miles) / return_speed) = 11.40 :=
by
  sorry

end total_tour_time_l152_152759


namespace gas_price_increase_l152_152441

theorem gas_price_increase (P C : ℝ) (x : ℝ) 
  (h1 : P * C = P * (1 + x) * 1.10 * C * (1 - 0.27272727272727)) :
  x = 0.25 :=
by
  -- The proof will be filled here
  sorry

end gas_price_increase_l152_152441


namespace arcsin_neg_one_is_neg_half_pi_l152_152964

noncomputable def arcsine_equality : Prop := 
  real.arcsin (-1) = - (real.pi / 2)

theorem arcsin_neg_one_is_neg_half_pi : arcsine_equality :=
by
  sorry

end arcsin_neg_one_is_neg_half_pi_l152_152964


namespace arrangeable_Zn_iff_odd_l152_152177

noncomputable def is_arrangeable (n : ℕ) : Prop :=
  ∃ (G : Finset (Fin n)),
    G = {a_1, a_2, ..., a_n} ∧
    (G = {a_1 + a_2, a_2 + a_3, ..., a_n + a_1})

theorem arrangeable_Zn_iff_odd (n : ℕ) :
  (n ≥ 2) → (is_arrangeable n ↔ odd n) :=
by sorry

end arrangeable_Zn_iff_odd_l152_152177


namespace tangent_line_at_point_l152_152428

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (1 / x)

theorem tangent_line_at_point :
  tangent_line f ⟨1, -1⟩ = (λ x, 2 * x - 3) :=
by
  sorry -- Proof not included

end tangent_line_at_point_l152_152428


namespace different_course_selection_l152_152772

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem different_course_selection :
  let total_ways := binomial_coefficient 4 2 * binomial_coefficient 4 2,
      same_course_ways := binomial_coefficient 4 2
  in total_ways - same_course_ways = 30 :=
by
  let total_ways := binomial_coefficient 4 2 * binomial_coefficient 4 2
  let same_course_ways := binomial_coefficient 4 2
  have total_ways_def : total_ways = 36 := by
    -- Proof of total_ways = 36 can be added here
    sorry
  have same_course_ways_def : same_course_ways = 6 := by
    -- Proof of same_course_ways = 6 can be added here
    sorry
  rw [total_ways_def, same_course_ways_def]
  exact calc
    36 - 6 = 30 : by norm_num

end different_course_selection_l152_152772


namespace least_integer_square_l152_152146

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l152_152146


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152669

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l152_152669


namespace smallest_angle_in_right_triangle_l152_152016

theorem smallest_angle_in_right_triangle (a b : ℝ) (h1 : a + b = 90) (h2 : 3*b = 2*a) : b = 36 :=
by {
  sorry,
}

end smallest_angle_in_right_triangle_l152_152016


namespace matrix_product_is_zero_l152_152959

def vec3 := (ℝ × ℝ × ℝ)

def M1 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((0, 2 * c, -2 * b),
   (-2 * c, 0, 2 * a),
   (2 * b, -2 * a, 0))

def M2 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((2 * a^2, a^2 + b^2, a^2 + c^2),
   (a^2 + b^2, 2 * b^2, b^2 + c^2),
   (a^2 + c^2, b^2 + c^2, 2 * c^2))

def matrix_mul (m1 m2 : vec3 × vec3 × vec3) : vec3 × vec3 × vec3 := sorry

theorem matrix_product_is_zero (a b c : ℝ) :
  matrix_mul (M1 a b c) (M2 a b c) = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) := by
  sorry

end matrix_product_is_zero_l152_152959


namespace max_non_attacking_queens_l152_152870

theorem max_non_attacking_queens (n m : ℕ) (board : fin n → fin m → bool) (red_corner_size : fin n → fin m → bool)
  (queen_attack : fin n → fin m → list (fin n × fin m))
  (condition_board : n = 101 ∧ m = 101)
  (condition_red_corner : ∀ (i : fin n) (j : fin m), i.val < 88 ∧ j.val < 88 → red_corner_size i j = tt)
  (condition_attack : ∀ (i : fin n) (j : fin m), queen_attack i j = 
    list.filter (λ (p : fin n × fin m), p.fst = i ∨ p.snd = j ∨ abs (p.fst.val - i.val) = abs (p.snd.val - j.val))
    ((list.fin_prod n m).remove_all [(i,j)]))
  (no_red_cells : ∀ (i : fin n) (j : fin m), (board i j = tt → red_corner_size i j = ff) )
  : (∃ (positions : list (fin n × fin m)),
    positions.length = 26 ∧
    (∀ (pos : fin n × fin m), pos ∈ positions → board pos.fst pos.snd = tt) ∧
    (∀ (p1 p2 : fin n × fin m), p1 ∈ positions → p2 ∈ positions → p1 ≠ p2 → 
      ∉ (queen_attack p1.fst p1.snd) p2))
:= sorry

end max_non_attacking_queens_l152_152870


namespace largest_multiple_11_lt_neg85_l152_152854

-- Define the conditions: a multiple of 11 and smaller than -85
def largest_multiple_lt (m n : Int) : Int :=
  let k := (m / n) - 1
  n * k

-- Define our specific problem
theorem largest_multiple_11_lt_neg85 : largest_multiple_lt (-85) 11 = -88 := 
  by
  sorry

end largest_multiple_11_lt_neg85_l152_152854


namespace initial_average_speed_l152_152764

theorem initial_average_speed (v d : ℝ) (h1 : d = v * (1 / 2)) (h2 : d = 86 * (0.7 * (1 / 2))) : v = 60.2 :=
by
  calc
    v * (1 / 2) = 86 * (0.7 * (1 / 2)) : by rw [h1, h2]
    ... = 60.2 : by norm_num

#eval initial_average_speed

end initial_average_speed_l152_152764


namespace probability_differs_by_three_l152_152932

theorem probability_differs_by_three :
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 3), (8, 5)],
      num_outcomes := List.length outcomes,
      total_possibilities := 8 * 8
  in
  Rational.mk num_outcomes total_possibilities = Rational.mk 7 64 :=
by
  sorry

end probability_differs_by_three_l152_152932


namespace circle_equation_l152_152109

-- Given definitions and conditions
def parabola_vertex : ℝ × ℝ := (0, 0)
def parabola_focus : ℝ × ℝ := (1, 0)
def directrix : ℝ → Prop := λ x, x = -1
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := 
  real.sqrt ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2)
noncomputable def radius : ℝ := distance parabola_focus (-1, 0)

-- The goal is to prove the equation of the circle
theorem circle_equation : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4) ↔ (real.sqrt ((x - 0)^2 + (y - 0)^2) = radius) := 
sorry

end circle_equation_l152_152109


namespace circumscribed_sphere_volume_eq_l152_152356

noncomputable def volume_of_circumscribed_sphere (P A B C : Type) [RealInnerProductSpace ℝ P]
  (h1 : equilateral_triangle A B C) (h2 : dist P A = 3) (h3 : dist P B = 3)
  (h4 : dist P C = 3) (h5 : orthogonal (P -ᵥ A) (P -ᵥ B)) :
  ℝ := sorry

theorem circumscribed_sphere_volume_eq :
  volume_of_circumscribed_sphere P A B C h1 h2 h3 h4 h5 = (27 * Real.sqrt 3 * Real.pi) / 2 :=
sorry

end circumscribed_sphere_volume_eq_l152_152356


namespace annabelle_savings_l152_152545

theorem annabelle_savings (weekly_allowance : ℕ) (junk_food_fraction : ℚ) (sweets_cost : ℕ) 
    (h1 : weekly_allowance = 30) 
    (h2 : junk_food_fraction = 1 / 3) 
    (h3 : sweets_cost = 8) : 
    weekly_allowance - (weekly_allowance * (junk_food_fraction.numerator / junk_food_fraction.denominator)) - sweets_cost = 12 :=
by
  sorry

end annabelle_savings_l152_152545


namespace sqrt_six_estimation_l152_152234

theorem sqrt_six_estimation : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by 
  sorry

end sqrt_six_estimation_l152_152234


namespace volume_of_smaller_tetrahedron_l152_152200

theorem volume_of_smaller_tetrahedron : 
  let edge_length := 2 in
  let half_edge_length := edge_length / 2 in
  let volume := (1 / 6) * half_edge_length^3 * (1 / Real.sqrt 2) in
  volume = 1 / (6 * Real.sqrt 2) := by
  sorry

end volume_of_smaller_tetrahedron_l152_152200


namespace number_of_solutions_l152_152681

theorem number_of_solutions : {pairs : ℕ // ∃ x y : ℤ, (x^4 + y^2 = 2 * y + 1) ∧ pairs = 4} sorry

end number_of_solutions_l152_152681


namespace a_formula_b_formula_T_formula_l152_152391

variable {n : ℕ}

def S (n : ℕ) := 2 * n^2

def a (n : ℕ) : ℕ := 
  if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 2 * (1 / 4 ^ (n - 1))

def c (n : ℕ) : ℕ := (4 * n - 2) / (2 * 4 ^ (n - 1))

def T (n : ℕ) : ℕ := 
  (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5)

theorem a_formula :
  ∀ n, a n = 4 * n - 2 := 
sorry

theorem b_formula :
  ∀ n, b n = 2 / (4 ^ (n - 1)) :=
sorry

theorem T_formula :
  ∀ n, T n = (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5) :=
sorry

end a_formula_b_formula_T_formula_l152_152391


namespace triangle_largest_angle_l152_152696

theorem triangle_largest_angle 
  (a1 a2 a3 : ℝ) 
  (h_sum : a1 + a2 + a3 = 180)
  (h_arith_seq : 2 * a2 = a1 + a3)
  (h_one_angle : a1 = 28) : 
  max a1 (max a2 a3) = 92 := 
by
  sorry

end triangle_largest_angle_l152_152696


namespace shifted_function_is_correct_l152_152112

def original_function (x : ℝ) : ℝ :=
  (x - 1)^2 + 2

def shifted_up_function (x : ℝ) : ℝ :=
  original_function x + 3

def shifted_right_function (x : ℝ) : ℝ :=
  shifted_up_function (x - 4)

theorem shifted_function_is_correct : ∀ x : ℝ, shifted_right_function x = (x - 5)^2 + 5 := 
by
  sorry

end shifted_function_is_correct_l152_152112


namespace circle_radius_l152_152662

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 4*x - 2*y - 5 = 0 → sqrt ((-4/2)^2 + (-2/2)^2 + 5 + (-4/2)^2 + (-2/2)^2) = sqrt 10 :=
by
  intro h
  sorry

end circle_radius_l152_152662


namespace distance_to_convenience_store_l152_152553

def distance_work := 6
def days_work := 5
def distance_dog_walk := 2
def times_dog_walk := 2
def days_week := 7
def distance_friend_house := 1
def times_friend_visit := 1
def total_miles := 95
def trips_convenience_store := 2

theorem distance_to_convenience_store :
  ∃ x : ℝ,
    (distance_work * 2 * days_work) +
    (distance_dog_walk * times_dog_walk * days_week) +
    (distance_friend_house * 2 * times_friend_visit) +
    (x * trips_convenience_store) = total_miles
    → x = 2.5 :=
by
  sorry

end distance_to_convenience_store_l152_152553


namespace urn_problem_l152_152939

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l152_152939


namespace sum_of_rep_decimals_l152_152744

open scoped BigOperators

def rep_decimals_set : set ℝ :=
  { x | ∃ a b : ℕ, a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * (a : ℝ) + (b : ℝ)) / 99 }

theorem sum_of_rep_decimals : ∑ x in rep_decimals_set, x = 90 / 11 :=
sorry

end sum_of_rep_decimals_l152_152744


namespace arithmetic_sequence_solution_l152_152350

theorem arithmetic_sequence_solution :
  ∃ (a1 d : ℤ), 
    (a1 + 3*d + (a1 + 4*d) + (a1 + 5*d) + (a1 + 6*d) = 56) ∧
    ((a1 + 3*d) * (a1 + 6*d) = 187) ∧
    (
      (a1 = 5 ∧ d = 2) ∨
      (a1 = 23 ∧ d = -2)
    ) :=
by
  sorry

end arithmetic_sequence_solution_l152_152350


namespace triangle_obtuse_l152_152721

-- Definitions for the triangle sides based on given heights
def side_ratios (a b c : ℝ) (ha hb hc : ℝ) : Prop :=
  a / ha = b / hb ∧ b / hb = c / hc

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Using side lengths derived from the inverse proportionality of heights
def derived_sides (a b c k : ℝ) :=
  a = 6 * k ∧ b = 4 * k ∧ c = 3 * k

-- Determine obtuseness by checking the cosine of the largest angle
def is_obtuse (a b c : ℝ) (cos_angle : ℝ) :=
  cos_angle < 0
  where cos_angle := (b^2 + c^2 - a^2) / (2 * b * c)

-- Main theorem statement encapsulating conditions and result
theorem triangle_obtuse 
  (a b c k ha hb hc : ℝ)
  (h_ratios : side_ratios a b c ha hb hc)
  (h_triangle : triangle_inequality a b c)
  (h_sides : derived_sides a b c k)
  : is_obtuse a b c ((b^2 + c^2 - a^2) / (2 * b * c)) :=
by
  sorry

end triangle_obtuse_l152_152721


namespace shop_earnings_l152_152337

theorem shop_earnings :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  let cola_earnings := cola_price * cola_sold
  let juice_earnings := juice_price * juice_sold
  let water_earnings := water_price * water_sold
  let total_earnings := cola_earnings + juice_earnings + water_earnings
  total_earnings = 88 := by
    simp [cola_price, juice_price, water_price, cola_sold, juice_sold, water_sold, cola_earnings, juice_earnings, water_earnings, total_earnings]; sorry

end shop_earnings_l152_152337


namespace max_min_distance_ellipse_l152_152376

theorem max_min_distance_ellipse (b : ℝ) (x y : ℝ) (a : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (x, y) ∧ (x^2) / 25 + (y^2) / (b^2) = 1 ∧ a = 5 ∧ c^2 = a^2 - b^2 ∧
  sqrt (a^2 - b^2) = c → (a + c) + (a - c) = 10 :=
by
  sorry

end max_min_distance_ellipse_l152_152376


namespace range_of_m_l152_152108

variable {f : ℝ → ℝ}

noncomputable def odd_function := ∀ x ∈ [-2, 2], f(-x) = -f(x)
noncomputable def monotonically_decreasing := ∀ x y ∈ [0, 2], x < y → f(x) > f(y)

theorem range_of_m (odd_fn : odd_function f) (mono_dec : monotonically_decreasing f) 
  (h : ∀ m ∈ {m : ℝ | f(1 + m) + f(m) < 0}, m ∈ Ioc (-1 / 2) 1) :
  ∀ m, f(1 + m) + f(m) < 0 → m ∈ Ioc (-1 / 2) 1 := by
  sorry

end range_of_m_l152_152108


namespace gcd_repeated_six_digit_l152_152206

/-- Definition of a twelve-digit integer formed by repeating a positive six-digit integer -/
def repeat_six_digit_twice (m : ℕ) (hm : 100000 ≤ m ∧ m < 1000000) : ℕ :=
  10^6 * m + m

/-- Proof that the greatest common divisor of all twelve-digit integers formed by repeating 
a positive six-digit integer twice in a row is 1000001. -/
theorem gcd_repeated_six_digit : ∀ m : ℕ, 100000 ≤ m ∧ m < 1000000 → gcd (repeat_six_digit_twice m (and.intro ‹100000 ≤ m› ‹m < 1000000›)) 1000001 = 1000001 := 
by
  sorry

end gcd_repeated_six_digit_l152_152206


namespace how_many_children_got_on_l152_152135

noncomputable def initial_children : ℝ := 42.5
noncomputable def children_got_off : ℝ := 21.3
noncomputable def final_children : ℝ := 35.8

theorem how_many_children_got_on : initial_children - children_got_off + (final_children - (initial_children - children_got_off)) = final_children := by
  sorry

end how_many_children_got_on_l152_152135


namespace constant_term_expansion_l152_152800

/-- This function finds the constant term of the expansion of (1 + x^4) * (2 + (1/x)) ^ 6.
    Based on the problem statement and calculations, we know it equals 124. -/
def constant_term (x : ℝ) : ℝ :=
  (2 + x⁻¹) ^ 6 + x ^ 4 * (2 + x⁻¹) ^ 6

theorem constant_term_expansion : 
  constant_term 1 = 124 :=
by
  sorry

end constant_term_expansion_l152_152800


namespace final_enrollment_count_l152_152900

def initial_students := 8
def additional_interested := 12
def dropout_fraction := 1 / 4
def neighboring_school_students := 3
def frustration_dropouts := 2
def enrollment_multiplier := 1.5
def scheduling_conflict_dropouts := 4
def additional_enrollment_after_week := 7
def study_group_size := 8
def study_group_multiplier := 3
def workload_dropout_percent := 0.15
def graduation_fraction := 1 / 3

-- Final expected remaining students
theorem final_enrollment_count :
  let interested_students := additional_interested * (1 - dropout_fraction)
  let after_interested := initial_students + interested_students
  let after_neighbors := after_interested + neighboring_school_students
  let after_frustration := after_neighbors - frustration_dropouts
  let increased_enrollment := after_frustration * enrollment_multiplier
  let total_after_increase := after_frustration + increased_enrollment
  let after_scheduling_conflict := total_after_increase - scheduling_conflict_dropouts
  let after_week_enrollment := after_scheduling_conflict + additional_enrollment_after_week
  let study_group_increase := study_group_size * study_group_multiplier
  let total_after_study_group := after_week_enrollment + study_group_increase
  let workload_dropouts := total_after_study_group * workload_dropout_percent
  let after_workload_dropout := total_after_study_group - workload_dropouts
  let graduated_students := after_workload_dropout * graduation_fraction
  let final_students := after_workload_dropout - graduated_students
  final_students = 41 :=
by sorry

end final_enrollment_count_l152_152900


namespace cube_surface_area_proof_l152_152977

-- Define the conditions as given in the problem
def distance_between_non_intersecting_diagonals (a : ℝ) : ℝ :=
  a * sqrt 3 / 3

-- Define the cube surface area calculation
def cube_surface_area (a : ℝ) : ℝ :=
  6 * a^2

-- Theorem: Prove the surface area of the cube given the distance between diagonals
theorem cube_surface_area_proof (a : ℝ) (h : distance_between_non_intersecting_diagonals a = 8) :
  cube_surface_area a = 1152 :=
by
  -- The proof steps are omitted
  sorry

end cube_surface_area_proof_l152_152977


namespace alyssa_initial_puppies_l152_152210

theorem alyssa_initial_puppies (gave_away has_left : ℝ) (h1 : gave_away = 8.5) (h2 : has_left = 12.5) :
    (gave_away + has_left = 21) :=
by
    sorry

end alyssa_initial_puppies_l152_152210


namespace inversely_proportional_is_option_C_l152_152542

def y_inversely_proportional_to_x (y x : ℝ) : Prop := y * x = 1

def option_A (x : ℝ) : ℝ := (1 / x) ^ 2
def option_B (x : ℝ) : ℝ := 1 / (x + 1)
def option_C (x : ℝ) : ℝ := 1 / (3 * x)
def option_D (x : ℝ) : ℝ := 1 / (x + 1) - 1 / x 

theorem inversely_proportional_is_option_C : ∃ f : ℝ → ℝ, (f = option_C) ∧ (∀ x : ℝ, y_inversely_proportional_to_x (f x) x) :=
  sorry

end inversely_proportional_is_option_C_l152_152542


namespace sum_of_intervals_l152_152996

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := floor x * (2015 ^ (x - floor x) - 1)

theorem sum_of_intervals : 
  (∑ k in Finset.range(2014) + 1, Real.logb 2015 ((2 + k : ℝ) / k)) = Real.logb 2015 2016 :=
by
sorrry

end sum_of_intervals_l152_152996


namespace determinant_of_2x2_matrix_l152_152952

theorem determinant_of_2x2_matrix :
  let a := 2
  let b := 4
  let c := 1
  let d := 3
  a * d - b * c = 2 := by
  sorry

end determinant_of_2x2_matrix_l152_152952


namespace range_of_fx_neg_l152_152011

-- Definition of even function
def even_fun (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Definition of monotonicity
def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f y ≤ f x

-- Given conditions
variables {f : ℝ → ℝ}
hypothesis h_even : even_fun f
hypothesis h_mono_dec : monotone_decreasing (λ x, if x ≤ 0 then f x else f (-x))
hypothesis h_f_one_zero : f 1 = 0

-- Theorem to prove
theorem range_of_fx_neg :
  {x : ℝ | f x < 0} = set.Ioo (-1 : ℝ) (1 : ℝ) :=
by sorry

end range_of_fx_neg_l152_152011


namespace new_average_age_of_group_l152_152418

theorem new_average_age_of_group (n : ℕ) (average_age : ℕ) (new_person_age : ℕ) (new_average_age : ℕ) 
  (h1 : average_age = 15) 
  (h2 : n = 9) 
  (h3 : new_person_age = 35) 
  (h4 : new_average_age = 17) : 
  (let total_age_original := average_age * n in
   let total_age_new := total_age_original + new_person_age in
   let new_number_of_people := n + 1 in
   new_average_age = total_age_new / new_number_of_people) :=
by 
  simp only [h1, h2, h3, h4]
  sorry

end new_average_age_of_group_l152_152418


namespace rhombus_area_l152_152869

/-
  We want to prove that the area of a rhombus with given diagonals' lengths is 
  equal to the computed value according to the formula Area = (d1 * d2) / 2.
-/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : 
  (d1 * d2) / 2 = 60 :=
by
  rw [h1, h2]
  sorry

end rhombus_area_l152_152869


namespace _l152_152380

noncomputable def complex_numbers_theorem (a b c : ℂ) 
  (h1 : |a| = 1) (h2 : |b| = 1) (h3 : |c| = 1) 
  (h4 : (a^3)/(b*c) + (b^3)/(a*c) + (c^3)/(a*b) = 3) : |a + b + c| = 1 :=
  by
  sorry

end _l152_152380


namespace expo_value_l152_152685

theorem expo_value (x y : ℝ) (h1 : 2^x = 3) (h2 : 2^y = 5) : 2^(x + 2 * y) = 75 := 
by 
  sorry

end expo_value_l152_152685


namespace integral_of_f_eq_neg_18_l152_152643

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * (f' 2) * x + 3
noncomputable def f' (x : ℝ) : ℝ := 2*x + 2*f' 2

theorem integral_of_f_eq_neg_18 
  (h1 : ∀ x : ℝ, differentiable_at ℝ f x) 
  (h2 : f 2 = 4 + 2 * (f' 2)) 
  (h3 : f' 2 = -4) : 
  ∫ x in 0..3, f x = -18 :=
by sorry

end integral_of_f_eq_neg_18_l152_152643


namespace correct_statement_is_D_l152_152023

-- Definitions of the conditions
def plane (P : Type) := P
def line (L : Type) := L
def perpendicular_to_plane {P : Type} (pl1 pl2 : plane P) : Prop := sorry
def perpendicular_to_line {L : Type} (ln : line L) (pl : plane L) : Prop := sorry
def parallel_to_line {L : Type} (ln : line L) (pl : plane L) : Prop := sorry

-- Conditions: Assuming the definitions according to statements
variable {P : Type}
variable {L : Type}

axiom A : ∀ (pl1 pl2 : plane P), (perpendicular_to_plane pl1 pl2 → (pl1 = pl2 → false))
axiom B : ∀ (ln1 ln2 : line L), (perpendicular_to_line ln1 ln2 → ln1 = ln2)
axiom C : ∀ (pl1 pl2 : plane P) (ln : line L), (parallel_to_line ln pl1 → parallel_to_line ln pl2 → (pl1 = pl2))
axiom D : ∀ (pl1 pl2 : plane P) (ln : line L), (perpendicular_to_line ln pl1 → perpendicular_to_line ln pl2 → (pl1 = pl2))

-- Theorem statement to be proved: (D) is correct
theorem correct_statement_is_D : D :=
by sorry

end correct_statement_is_D_l152_152023


namespace first_movie_series_seasons_l152_152226

theorem first_movie_series_seasons (S : ℕ) : 
  (∀ E : ℕ, E = 16) → 
  (∀ L : ℕ, L = 2) → 
  (∀ T : ℕ, T = 364) → 
  (∀ second_series_seasons : ℕ, second_series_seasons = 14) → 
  (∀ second_series_remaining : ℕ, second_series_remaining = second_series_seasons * (E - L)) → 
  (E - L = 14) → 
  (second_series_remaining = 196) → 
  (T - second_series_remaining = S * (E - L)) → 
  S = 12 :=
by 
  intros E_16 L_2 T_364 second_series_14 second_series_remaining_196 E_L second_series_total_episodes remaining_episodes
  sorry

end first_movie_series_seasons_l152_152226


namespace collinear_points_l152_152341

theorem collinear_points (n : ℕ) (points : fin n → ℝ × ℝ) :
  (∀ i j : fin n, i ≠ j → ∃ k : fin n, k ≠ i ∧ k ≠ j ∧ 
                       collinear {points i, points j, points k}) →
  (collinear (finset.univ.map (function.embedding.subtype points))) :=
sorry

end collinear_points_l152_152341


namespace center_of_circle_l152_152984

theorem center_of_circle : 
  ∀ x y : ℝ, 4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0 → (x = -1 ∧ y = 3 / 2) :=
by
  sorry

end center_of_circle_l152_152984


namespace median_of_2_probability_l152_152212

theorem median_of_2_probability :
  let S := {2, 0, 1, 5}
  in (∃ A B C ∈ S, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ {A, B, C} = {2, 0, 5} ∨ {A, B, C} = {2, 1, 5}) → 
  ∑ x in finset.powersetLen 3 S, 1 / finset.card (finset.powersetLen 3 S) = 1 / 2 :=
by
  sorry

end median_of_2_probability_l152_152212


namespace ratio_of_areas_l152_152033

open Real

-- Let rC be the radius of Circle C and rD be the radius of Circle D.
variables (rC rD : ℝ)

-- Hypothesis 1: The circumference of Circle D is twice that of Circle C.
hypothesis h1 : 2 * π * rD = 4 * π * rC

-- Hypothesis 2: An arc of 60 degrees on Circle C has the same length as an arc of 45 degrees on Circle D.
hypothesis h2 : (60 / 360) * 2 * π * rC = (45 / 360) * 2 * π * rD

-- Theorem: The ratio of the area of Circle D to the area of Circle C is 4.
theorem ratio_of_areas : (π * rD^2) / (π * rC^2) = 4 :=
begin
  sorry
end

end ratio_of_areas_l152_152033


namespace fraction_of_2d_nails_l152_152551

theorem fraction_of_2d_nails (x : ℝ) (h1 : x + 0.5 = 0.75) : x = 0.25 :=
by
  sorry

end fraction_of_2d_nails_l152_152551


namespace kelly_games_left_l152_152370

theorem kelly_games_left (initial_games : Nat) (given_away : Nat) (remaining_games : Nat) 
  (h1 : initial_games = 106) (h2 : given_away = 64) : remaining_games = 42 := by
  sorry

end kelly_games_left_l152_152370


namespace problem_solution_l152_152255

noncomputable def curve_M (x y : ℝ) : Prop :=
  x ^ (1 / 2) + y ^ (1 / 2) = 1

def statement_1 (x y : ℝ) : Prop :=
  (curve_M x y) → (real.sqrt (x ^ 2 + y ^ 2) = real.sqrt 2 / 2)

def statement_2 (x y : ℝ) : Prop :=
  (curve_M x y) → (x ∈ set.Icc 0 1 ∧ y ∈ set.Icc 0 1)

theorem problem_solution :
  (¬ ∀ (x y : ℝ), statement_1 x y) ∧ (∀ (x y : ℝ), statement_2 x y) :=
by
  sorry

end problem_solution_l152_152255


namespace intersection_curve_sinusoidal_l152_152890

-- Definitions of geometry entities and assumptions
section intersection_curve

variables (r : ℝ) (π : plane) (cylinder : cylindrical_surface)
  (h1 : intersects cylinder π)
  (h2 : ¬ perpendicular π cylinder.generating_lines)

-- Proving the intersection curve is sinusoidal
theorem intersection_curve_sinusoidal :
  is_sinusoidal_curve (developed_intersection_curve cylinder π) := 
sorry

end intersection_curve

end intersection_curve_sinusoidal_l152_152890


namespace jean_spots_on_sides_l152_152040

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l152_152040


namespace drunk_drivers_count_l152_152700

-- Define the problem parameters
def drunk_drivers (D : ℕ) : Prop :=
  let S := 7 * D - 3 in
  let V := 2 * D in
  let T := 1 / 2 * S + 5 in
  D + S + V + T = 180

-- Theorem statement
theorem drunk_drivers_count : ∃ D : ℕ, drunk_drivers D ∧ D = 13 :=
begin
  sorry
end

end drunk_drivers_count_l152_152700


namespace license_plates_count_l152_152415

def gropka_alphabet : List Char := ['A', 'E', 'G', 'I', 'K', 'O', 'R', 'U', 'V']

def valid_first_letters : List Char := ['A', 'E']
def valid_last_letter : Char := 'V'

def count_valid_license_plates : Nat :=
  let num_first_choices := 2 -- A or E
  let num_last_choices := 1 -- V
  let num_second_choices := 7 -- Not A, E, V, or P
  let num_third_choices := 6 -- Remaining choices
  num_first_choices * num_second_choices * num_third_choices

theorem license_plates_count : count_valid_license_plates = 84 := by
  -- Start by calculating each step:
  have num_first_choices := 2
  have num_last_choices := 1
  have num_second_choices := 7
  have num_third_choices := 6
  have total_combinations := num_first_choices * num_second_choices * num_third_choices
  -- Assert the total
  have eqn : total_combinations = 84 := by {
    linarith,
  }
  exact eqn

end license_plates_count_l152_152415


namespace john_walking_distance_l152_152368

variable nina_distance : ℝ
variable john_extra_distance : ℝ
variable john_distance : ℝ

theorem john_walking_distance : nina_distance = 0.4 → john_extra_distance = 0.3 → john_distance = nina_distance + john_extra_distance → john_distance = 0.7 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end john_walking_distance_l152_152368


namespace max_ratio_BO_BM_l152_152348

theorem max_ratio_BO_BM
  (C : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hC : C = (0, -4))
  (hCir : ∃ (P : ℝ × ℝ), (P.1 - 2)^2 + (P.2 - 4)^2 = 1 ∧ A = ((P.1 + C.1) / 2, (P.2 + C.2) / 2))
  (hPar : ∃ (x y : ℝ), B = (x, y) ∧ y^2 = 4 * x) :
  ∃ t, t = (4 * Real.sqrt 7)/7 ∧ t = Real.sqrt ((B.1^2 + 4 * B.1)/((B.1 + 1/2)^2)) := by
  -- Given conditions and definitions
  obtain ⟨P, hP, hA⟩ := hCir
  obtain ⟨x, y, hB⟩ := hPar
  use (4 * Real.sqrt 7) / 7
  sorry

end max_ratio_BO_BM_l152_152348


namespace largest_binomial_coefficient_l152_152462

theorem largest_binomial_coefficient :
  ∃ n : ℕ, n ≤ 11 ∧ (∏ k in finset.range 4 + 1, (10 - k) / (k + 1)) + (∏ k in finset.range 5 + 1, (10 - k) / (k + 1)) = (∏ k in finset.range (11 + 1 - n), (11 - k) / (k + 1)) ∧ ∀ m : ℕ, (∏ k in finset.range (11 + 1 - m), (11 - k) / (k + 1))  = (∏ k in finset.range (11 + 1 - n), (11 - k) / (k + 1)) → m ≤ n :=
begin
  sorry
end

end largest_binomial_coefficient_l152_152462


namespace distance_from_A_to_C_l152_152081

theorem distance_from_A_to_C (x y : ℕ) (d : ℚ)
  (h1 : d = x / 3) 
  (h2 : 13 + (d * 15) / (y - 13) = 2 * x)
  (h3 : y = 2 * x + 13) 
  : x + y = 26 := 
  sorry

end distance_from_A_to_C_l152_152081


namespace find_m_value_l152_152707

theorem find_m_value (m : ℝ) (h : (m - 4)^2 + 1^2 + 2^2 = 30) : m = 9 ∨ m = -1 :=
by {
  sorry
}

end find_m_value_l152_152707


namespace cannot_be_set_A_l152_152655

-- Define the quadratic function f(x)
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the function y = [f(x)]^2 + p * f(x) + q
def y (a b c p q x : ℝ) := (f a b c x)^2 + p * (f a b c x) + q

-- Symmetry condition about the axis x = -b/(2a)
def symmetric_about_axis (a b : ℝ) (s : list ℝ) : Prop :=
  list.all s (λ x, 2 * x = -b/a)

-- Prove that sets 2 and 4 cannot be the set of zeros A
theorem cannot_be_set_A (a b c p q : ℝ) (ha : a ≠ 0) :
  ¬ symmetric_about_axis a b [1/2, 1/3, 1/4] ∧ ¬ symmetric_about_axis a b [-4, -1, 0, 2] :=
sorry

end cannot_be_set_A_l152_152655


namespace phase_initial_phase_sin_l152_152438

theorem phase_initial_phase_sin:
  ∀ (x : ℝ), 
  let y := 3 * Real.sin (-x + (Real.pi / 6)) in 
  ∃ (phase initial_phase : ℝ),
    (y = 3 * Real.sin (phase * x + initial_phase)) ∧ 
    (phase * 1 + initial_phase = x + 5 * Real.pi / 6) ∧ 
    initial_phase = 5 * Real.pi / 6 :=
by 
  sorry

end phase_initial_phase_sin_l152_152438


namespace activity_preference_order_l152_152549

def dodgeball : ℚ := 13 / 40
def food_fair : ℚ := 9 / 20
def talent_show : ℚ := 11 / 30
def basketball_game : ℚ := 7 / 15

theorem activity_preference_order :
  list.sort (λ a b : ℚ, a > b) [dodgeball, food_fair, talent_show, basketball_game] = [basketball_game, food_fair, talent_show, dodgeball] :=
by
  sorry

end activity_preference_order_l152_152549


namespace find_sum_of_values_l152_152598

noncomputable def sum_of_solutions (x : ℕ) : ℕ :=
  if h : 2^(x^2 - 5*x - 4) = 8^(x - 5)
  then x
  else 0

theorem find_sum_of_values :
  (sum_of_solutions 1) + (sum_of_solutions 11) = 12 :=
by
  unfold sum_of_solutions
  -- Prove the conditions are satisfied for x = 1 and x = 11
  have h1 : 2^(1^2 - 5*1 - 4) = 8^(1 - 5),
  { sorry },
  have h11 : 2^(11^2 - 5*11 - 4) = 8^(11 - 5),
  { sorry },
  rw [if_pos h1, if_pos h11],
  norm_num

end find_sum_of_values_l152_152598


namespace functional_equation_exponential_l152_152317

theorem functional_equation_exponential (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  ∀ (x y : ℝ), (f : ℝ → ℝ) = λ x, a^x → f(x + y) = f(x) * f(y) :=
by
  sorry

end functional_equation_exponential_l152_152317


namespace distance_between_Jay_and_Paul_l152_152361

theorem distance_between_Jay_and_Paul
  (initial_distance : ℕ)
  (jay_speed : ℕ)
  (paul_speed : ℕ)
  (time : ℕ)
  (jay_distance_walked : ℕ)
  (paul_distance_walked : ℕ) :
  initial_distance = 3 →
  jay_speed = 1 / 20 →
  paul_speed = 3 / 40 →
  time = 120 →
  jay_distance_walked = jay_speed * time →
  paul_distance_walked = paul_speed * time →
  initial_distance + jay_distance_walked + paul_distance_walked = 18 := by
  sorry

end distance_between_Jay_and_Paul_l152_152361


namespace treasure_value_base10_l152_152198

def convert_base7_to_base10 (n : List ℕ) : ℕ :=
  n.foldl (λ acc x, acc * 7 + x) 0

def silver := [3, 2, 1, 4]  -- 3214_7
def stones := [1, 6, 5, 2]  -- 1652_7
def pearls := [2, 4, 3, 1]  -- 2431_7
def coins := [6, 5, 4]      -- 654_7

theorem treasure_value_base10 :
  convert_base7_to_base10 silver + convert_base7_to_base10 stones +
  convert_base7_to_base10 pearls + convert_base7_to_base10 coins = 3049 :=
by
  -- Definitions of each treasure in base 7
  let silver_value := convert_base7_to_base10 silver
  let stones_value := convert_base7_to_base10 stones
  let pearls_value := convert_base7_to_base10 pearls
  let coins_value := convert_base7_to_base10 coins
  -- Sum of the values
  have h : silver_value + stones_value + pearls_value + coins_value = 3049 := sorry
  exact h

end treasure_value_base10_l152_152198


namespace monotonic_intervals_of_f_max_k_for_inequality_l152_152618

noncomputable def f (x : ℝ) := Real.log (x + 1) - x

theorem monotonic_intervals_of_f :
  (∀ x ∈ Set.Ico (-1 : ℝ) 0, 0 ≤ HasDerivAt f x f') ∧
  (∀ x ∈ Set.Ioi (0 : ℝ), 0 > HasDerivAt f x f') :=
sorry

theorem max_k_for_inequality (k : ℝ) :
  (∀ x > 0, (kx^2 / (Real.exp x - 1) - x ≤ f(x))) → k ≤ 1 :=
sorry

end monotonic_intervals_of_f_max_k_for_inequality_l152_152618


namespace roots_quadratic_eq_l152_152639

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end roots_quadratic_eq_l152_152639


namespace analytical_expression_f_minimum_phi_l152_152677

-- Define variables and given conditions
variable (A ω θ x : Real)
variable (hA : A > 0)
variable (hω : ω > 0)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (a : Real × Real := (A * sin (ω * x), A * cos (ω * x)))
variable (b : Real × Real := (cos θ, sin θ))
variable (f : Real → Real := λ x, (a.1 * b.1 + a.2 * b.2 + 1))
variable (sym_dist : Real := π / 2)
variable (x_max : Real := π / 12)
variable (h_max : f x_max = 3)

-- Problem statements to prove
theorem analytical_expression_f :
  f x = 2 * sin (2 * x + π / 3) + 1 :=
sorry

theorem minimum_phi (φ : Real) (h_phi : φ > 0) :
  (2 * sin (2 * (x + φ) + π / 3) = -(2 * sin (2 * (x - φ) + π / 3))) → 
  φ = π / 3 :=
sorry

end analytical_expression_f_minimum_phi_l152_152677


namespace combined_capacity_ratio_l152_152789

def height_A : ℝ := 10
def circ_A : ℝ := 7
def height_B : ℝ := 7
def circ_B : ℝ := 10
def height_C : ℝ := 5
def circ_C : ℝ := 14

def radius (circ : ℝ) : ℝ :=
  circ / (2 * Real.pi)

def volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

def volume_A : ℝ :=
  volume (radius circ_A) height_A

def volume_B : ℝ :=
  volume (radius circ_B) height_B

def volume_C : ℝ :=
  volume (radius circ_C) height_C

def combined_volume_AC : ℝ :=
  volume_A + volume_C

theorem combined_capacity_ratio :
  (combined_volume_AC / volume_B) * 100 = 210 := by
  sorry

end combined_capacity_ratio_l152_152789


namespace pears_count_undetermined_l152_152727

theorem pears_count_undetermined (joan_oranges : ℕ) (sara_oranges : ℕ) (total_oranges : ℕ)
  (alyssa_pears : ℕ) (H1 : joan_oranges = 37) (H2 : sara_oranges = 10)
  (H3 : total_oranges = 47) (H4 : total_oranges = joan_oranges + sara_oranges) :
  ∃ n : ℕ, n = alyssa_pears :=
by
  intro n
  use n
  sorry

end pears_count_undetermined_l152_152727


namespace shop_earnings_l152_152338

theorem shop_earnings :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  let cola_earnings := cola_price * cola_sold
  let juice_earnings := juice_price * juice_sold
  let water_earnings := water_price * water_sold
  let total_earnings := cola_earnings + juice_earnings + water_earnings
  total_earnings = 88 := by
    simp [cola_price, juice_price, water_price, cola_sold, juice_sold, water_sold, cola_earnings, juice_earnings, water_earnings, total_earnings]; sorry

end shop_earnings_l152_152338


namespace trigonometric_identity_l152_152223

theorem trigonometric_identity:
  (1 - sin (Real.pi / 8)) *
  (1 - sin (3 * Real.pi / 8)) *
  (1 - sin (5 * Real.pi / 8)) *
  (1 - sin (7 * Real.pi / 8)) = 1 / 4 := by
    sorry

end trigonometric_identity_l152_152223


namespace cosine_of_angle_between_diagonals_l152_152195

-- Defining the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 1, 1)
def b : ℝ × ℝ × ℝ := (1, -1, -1)

-- Calculating the diagonal vectors
def d1 := (a.1 + b.1, a.2 + b.2, a.3 + b.3)  -- a + b
def d2 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)  -- b - a

-- Dot product function for 3D vectors
def dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Norm function for 3D vectors
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Cosine of the angle between two vectors
def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_prod v1 v2 / (norm v1 * norm v2)

-- The theorem to prove
theorem cosine_of_angle_between_diagonals : cos_angle d1 d2 = - (Real.sqrt 3 / 3) :=
by
  sorry

end cosine_of_angle_between_diagonals_l152_152195


namespace radius_of_larger_ball_l152_152127

-- Definitions based on conditions
def small_ball_radius := 1 -- Radius of each small ball in inches
def num_small_balls := 8   -- Number of small balls
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Total volume of the small balls
def total_volume_small_balls := num_small_balls * (volume_of_sphere small_ball_radius)

-- Radius of the larger ball
def larger_ball_radius : ℝ := Real.root 3 (total_volume_small_balls / ((4/3) * Real.pi))

-- The theorem to prove
theorem radius_of_larger_ball :
  larger_ball_radius = 2 :=
by
  sorry

end radius_of_larger_ball_l152_152127


namespace necessary_but_not_sufficient_condition_l152_152178

theorem necessary_but_not_sufficient_condition :
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬ (x > 2)) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l152_152178


namespace sara_spent_on_salad_l152_152085

theorem sara_spent_on_salad: 
  ∀ (cost_hotdog cost_total cost_salad : ℝ),
  cost_hotdog = 5.36 →
  cost_total = 10.46 →
  cost_salad = cost_total - cost_hotdog →
  cost_salad = 5.10 := 
by
  intros cost_hotdog cost_total cost_salad h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sara_spent_on_salad_l152_152085


namespace min_distance_PQ_l152_152439

theorem min_distance_PQ :
  let P := (1, 0)
  let Q := λ t : ℝ, (3 * t, sqrt 3 + sqrt 3 * t)
  let dist := λ t : ℝ, sqrt ((3 * t - 1)^2 + (sqrt 3 + sqrt 3 * t)^2) - 1
  ∃ t : ℝ, dist t = 1 := sorry

end min_distance_PQ_l152_152439


namespace roots_quadratic_eq_l152_152638

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end roots_quadratic_eq_l152_152638


namespace max_k_value_circle_intersection_l152_152805

theorem max_k_value_circle_intersection :
  ∃ (k : ℝ), (0 ≤ k ∧ k ≤ 4 / 3) ∧ 
    let C := (4, 0) in
    let radius := 1 in
    let distance := (λ k : ℝ, abs (4 * k - 2) / Real.sqrt (k^2 + 1)) in
    distance k ≤ 2 :=
sorry

end max_k_value_circle_intersection_l152_152805


namespace magician_act_reappearance_l152_152520

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l152_152520


namespace probability_of_even_sum_of_drawn_balls_l152_152184

def balls : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def odds := [1, 3, 5, 7, 9, 11, 13, 15]
def evens := [2, 4, 6, 8, 10, 12, 14]

theorem probability_of_even_sum_of_drawn_balls :
  ∑ i in (Finset.range 8).powerset.filter (λ s, s.card % 2 = 0),
  (Nat.choose 8 s.card * Nat.choose 7 (5 - s.card : ℕ)) = 1491 ∧
  Nat.choose 15 5 = 3003 ↔
  (∑ i in (Finset.range 8).powerset.filter (λ s, s.card % 2 = 0),
  (Nat.choose 8 s.card * Nat.choose 7 (5 - s.card : ℕ)) / Nat.choose 15 5 = 1491 / 3003 := sorry

end probability_of_even_sum_of_drawn_balls_l152_152184


namespace calculate_initial_budget_l152_152393

-- Definitions based on conditions
def cost_of_chicken := 12
def cost_per_pound_beef := 3
def pounds_of_beef := 5
def amount_left := 53

-- Derived definition for total cost of beef
def cost_of_beef := cost_per_pound_beef * pounds_of_beef
-- Derived definition for total spent
def total_spent := cost_of_chicken + cost_of_beef
-- Final calculation for initial budget
def initial_budget := total_spent + amount_left

-- Statement to prove
theorem calculate_initial_budget : initial_budget = 80 :=
by
  sorry

end calculate_initial_budget_l152_152393


namespace product_contains_no_odd_exponents_l152_152774

open scoped BigOperators

noncomputable def A (x : ℝ) : ℝ :=
  ∑ k in Finset.range 101, if k % 2 = 0 then x^k else -x^k

noncomputable def B (x : ℝ) : ℝ :=
  ∑ k in Finset.range 101, x^k

theorem product_contains_no_odd_exponents (x : ℝ) : 
    ∀ (n : ℕ), n % 2 = 1 → coeff ((A x) * (B x)) n = 0 :=
by
  sorry

end product_contains_no_odd_exponents_l152_152774


namespace sum_of_reciprocals_eq_two_l152_152957

theorem sum_of_reciprocals_eq_two :
  ∃ (n : Fin 1974 → ℕ), Function.Injective n ∧ (∑ i, (1 : ℝ)/(n i)) = 2 :=
by
-- Proof goes here
sorry

end sum_of_reciprocals_eq_two_l152_152957


namespace negation_of_p_l152_152399

-- Define the proposition p
def p : Prop := ∃ x : ℝ, x + 2 ≤ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x + 2 > 0

-- State the theorem that the negation of p is not_p
theorem negation_of_p : ¬ p = not_p := by 
  sorry -- Proof not provided

end negation_of_p_l152_152399


namespace bad_arrangements_count_l152_152816

noncomputable def count_bad_arrangements : Nat := 3

theorem bad_arrangements_count
  (nums : Finset ℕ)
  (circle : Finset (Finset ℕ))
  (bad : circle → Prop)
  (n_range : Finset ℕ)
  (bad_def : ∀ (arr: circle), bad arr ↔ (∃ n ∈ n_range, ∀ s ⊆ arr, (s.sum ≠ n)))
  (rot_reflect_eq : ∀ (arr1 arr2: circle), arr1 = arr2) :
  (nums = {1, 2, 3, 4, 6} ∧
   circle = {arrangement ∈ nums.powerset | arrangement.card = 5} ∧
   n_range = {1, ..., 20} ∧
   ∃! arr ∈ circle, bad arr) →
  count_bad_arrangements = 3 :=
by
  sorry

end bad_arrangements_count_l152_152816


namespace find_angle_AEB_l152_152704

-- Definitions and Conditions
variables {A B C D E : Type} [regular_polygon A B C D E 20]
variables (AE DE : ℝ) [h1 : AE = DE]
variables (angle_BEC angle_CED : ℝ) [h2 : angle_BEC = 2 * angle_CED]

-- Assertion
theorem find_angle_AEB :
  ∃ (angle_AEB : ℝ), angle_AEB = 39 := by
  sorry

end find_angle_AEB_l152_152704


namespace children_attended_l152_152453

theorem children_attended (A C : ℕ) (h1 : C = 2 * A) (h2 : A + C = 42) : C = 28 :=
by
  sorry

end children_attended_l152_152453


namespace max_area_square_pen_l152_152269

theorem max_area_square_pen (P : ℝ) (h1 : P = 64) : ∃ A : ℝ, A = 256 := 
by
  sorry

end max_area_square_pen_l152_152269


namespace carol_carrots_l152_152958

def mother_picked := 16
def good_carrots := 38
def bad_carrots := 7
def total_carrots := good_carrots + bad_carrots
def carol_picked : Nat := total_carrots - mother_picked

theorem carol_carrots : carol_picked = 29 := by
  sorry

end carol_carrots_l152_152958


namespace no_natural_number_satisfying_conditions_l152_152905

theorem no_natural_number_satisfying_conditions (n a b : ℕ) (h1 : 2 * n + 1 = a ^ 2) (h2 : 3 * n + 1 = b ^ 2) : ¬ nat.prime n :=
by
  sorry -- Proof omitted.

end no_natural_number_satisfying_conditions_l152_152905


namespace wheels_per_row_l152_152451

theorem wheels_per_row:
  (trains carriages rows : ℕ) 
  (total_wheels : ℕ) 
  (h_trains : trains = 4) 
  (h_carriages : carriages = 4) 
  (h_rows : rows = 3) 
  (h_total_wheels : total_wheels = 240) :
  total_wheels / (trains * carriages * rows) = 5 :=
by
  sorry

end wheels_per_row_l152_152451


namespace find_fraction_of_cistern_l152_152504

noncomputable def fraction_initially_full (x : ℝ) : Prop :=
  let rateA := (1 - x) / 12
  let rateB := (1 - x) / 8
  let combined_rate := 1 / 14.4
  combined_rate = rateA + rateB

theorem find_fraction_of_cistern {x : ℝ} (h : fraction_initially_full x) : x = 2 / 3 :=
by
  sorry

end find_fraction_of_cistern_l152_152504


namespace competition_results_l152_152136

variable (A B C : ℕ)

def is_rank (rankings : List ℕ) : Prop :=
  rankings = [A, B, C] ∧ rankings.perm [1, 2, 3]

def jia_statement  : Prop := A = 1
def yi_statement   : Prop := B ≠ 1
def bing_statement : Prop := C ≠ 3

def only_one_true (statements : List Prop) : Prop :=
  (statements.filter id).length = 1

theorem competition_results :
  (only_one_true [jia_statement, yi_statement, bing_statement]) →
  is_rank [A, B, C] →
  [A, B, C] = [3, 1, 2] :=
by
  intro h_only_one_true h_is_rank
  -- skipped proof
  sorry

end competition_results_l152_152136


namespace f_5_eq_a_l152_152692

variable {α : Type}
variable {a : α} [AddGroup α]

-- Assume f is an even function with period 4
def isEven (f : α → α) : Prop := ∀ x, f x = f (-x)
def isPeriodic (f : α → α) (p : α) : Prop := ∀ x, f (x + p) = f x

-- Given conditions
axiom f_even : isEven f
axiom f_periodic_4 : isPeriodic f 4
axiom f_neg1_eq_a : f (-1) = a
axiom a_ne_0 : a ≠ 0

-- Prove f(5) = a
theorem f_5_eq_a (f : α → α) : f 5 = a := 
by
  sorry

end f_5_eq_a_l152_152692


namespace freezer_temp_correct_l152_152331

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l152_152331


namespace cosine_angle_EM_AN_l152_152580

-- Geometric setup
variables {A B C D E M N: Type} [equilateral_triangle ABC] [square ABDE] (M: midpoint AC) (N: midpoint BC)

-- Given condition: Cosine of the dihedral angle C-ABD
axiom dihedral_angle_cosine: cos (dihedral_angle C ABD) = sqrt(3) / 3

-- Statement to prove: Cosine of the angle between EM and AN
theorem cosine_angle_EM_AN (EM AN : vector_space ℝ) : 
  cos (angle_between EM AN) = 1 / 6 :=
sorry

end cosine_angle_EM_AN_l152_152580


namespace james_600_mile_trip_profit_l152_152037

def james_payment_per_mile : ℝ := 0.5

def gas_cost_per_gallon : ℝ := 4.0

def truck_miles_per_gallon : ℝ := 20.0

def trip_miles : ℝ := 600.0

def james_total_payment : ℝ := james_payment_per_mile * trip_miles := by
  sorry

def gallons_needed : ℝ := trip_miles / truck_miles_per_gallon := by
  sorry

def total_gas_cost : ℝ := gallons_needed * gas_cost_per_gallon := by
  sorry

def james_profit : ℝ := james_total_payment - total_gas_cost := by
  sorry

theorem james_600_mile_trip_profit : james_profit = 180 := by
  sorry

end james_600_mile_trip_profit_l152_152037


namespace correct_result_proof_l152_152914

-- Given problem conditions
variables (α : ℝ) (correct_result mistaken_result : ℝ)

-- Definitions according to the problem
def correct_multiplication := 1.23 * α
def mistaken_multiplication := 1.2 * α
def difference := 0.3

-- Set the condition that the mistaken result is 0.3 less than the correct result
def condition := mistaken_multiplication + difference = correct_multiplication

-- We aim to prove that when α = 90, the correct result is 111
theorem correct_result_proof (h : α = 90) : correct_result = 111 :=
by
  have hα : α = 90 := h
  have h_correct_multiplication := correct_multiplication α
  have h_mistaken_multiplication := mistaken_multiplication α
  have h_difference := difference
  have h_condition := condition α correct_multiplication mistaken_multiplication h_difference
  have h_goal := h_condition hα
  sorry

end correct_result_proof_l152_152914


namespace dave_deleted_apps_l152_152972

theorem dave_deleted_apps : 
  ∀ (a_initial a_left a_deleted : ℕ), a_initial = 16 → a_left = 5 → a_deleted = a_initial - a_left → a_deleted = 11 :=
by
  intros a_initial a_left a_deleted h_initial h_left h_deleted
  rw [h_initial, h_left] at h_deleted
  exact h_deleted

end dave_deleted_apps_l152_152972


namespace compare_sqrt_magnitudes_find_m_value_l152_152877

-- Problem 1: Compare magnitudes
theorem compare_sqrt_magnitudes :
  sqrt 7 + sqrt 10 > sqrt 3 + sqrt 14 :=
sorry

-- Problem 2: Find the value of m
theorem find_m_value (x : ℝ) (hx : 0 < x ∧ x < 2) :
  (-1 / 2 * x^2 + 2 * x > mx) → m = 1 :=
sorry

end compare_sqrt_magnitudes_find_m_value_l152_152877


namespace max_discount_l152_152422

/-- 
Let CP be the cost price in yuan, MP be the market price in yuan, 
and SP be the selling price. Given:
    CP = 200
    MP = 300
    SP = MP * (1 - d) where d is the discount as a percentage (e.g., 0.1 for 10%)
    The profit margin must be at least 5% of the cost price (i.e., 10 yuan).
    
We prove that the maximum discount rate d such that the profit margin is not less than 
5% of the cost price is 30% (i.e., d = 0.3).
-/
theorem max_discount (CP MP : ℝ) (hCP : CP = 200) (hMP : MP = 300) :
  ∃ d : ℝ, d = 0.3 ∧ MP * (1 - d) - CP ≥ 0.05 * CP :=
by
  exist d
  reduce
  sorry

end max_discount_l152_152422


namespace min_length_AB_l152_152027

-- Define the center of the circle
def center_C := (2, 4)

-- Define the radius of the circle
def radius_C := √2

-- Define the equation of the circle
def circle_eq (x y : ℝ) := (x - 2)^2 + (y - 4)^2 = 2

-- Define the line equation
def line_eq (x y : ℝ) := 2 * x - y - 3 = 0

-- Define the condition CM ⊥ CN
def perp_CM_CN (xM yM xN yN : ℝ) := (xM - 2)*(xN - 2) + (yM - 4)*(yN - 4) = 0

-- Define the midpoint P of MN
def is_midpoint (xM yM xN yN xP yP : ℝ) := (xP, yP) = ((xM + xN) / 2, (yM + yN) / 2)

theorem min_length_AB (xM yM xN yN xP yP : ℝ)
  (h1 : circle_eq xM yM)
  (h2 : circle_eq xN yN)
  (h3 : perp_CM_CN xM yM xN yN)
  (h4 : is_midpoint xM yM xN yN xP yP)
  : True := -- Replace True with the actual proof condition in future
sorry

end min_length_AB_l152_152027


namespace num_pairs_with_math_book_l152_152698

theorem num_pairs_with_math_book (books : Finset String) (h : books = {"Chinese", "Mathematics", "English", "Biology", "History"}):
  (∃ pairs : Finset (Finset String), pairs.card = 4 ∧ ∀ pair ∈ pairs, "Mathematics" ∈ pair) :=
by
  sorry

end num_pairs_with_math_book_l152_152698


namespace inscribed_squares_ratio_l152_152204

theorem inscribed_squares_ratio {x y : ℝ} (h1 : ∃ (x : ℝ), ∃ (y : ℝ), 
  ∃ (triangle : ℝ × ℝ × ℝ), 
    triangle = (5,12,13) ∧ 
    (∃ (s1 : ℝ), s1 = x ∧ s1^2 = x * x = 5 * 12 / (5 + 12) ^ 2) ∧ 
    (∃ (s2 : ℝ), s2 = y ∧ s2^2 = y * y = 13 * y / (12 * 5 / (5 * 5 + 12 * 12))) : 
    x / y = 39 / 51 :=
by
  sorry

end inscribed_squares_ratio_l152_152204


namespace distance_between_l1_and_l2_l152_152124

-- Definitions for Point, Line, and Distance Calculation
structure Point where
  x : ℝ
  y : ℝ

def line_through_point_with_perpendicular_slope (P : Point) (s : ℝ) : (ℝ × ℝ × ℝ) :=
  (s, -1, s * P.x - P.y)

def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Math.sqrt (a^2 + b^2)

-- Given conditions
def A : Point := { x := 4, y := -3 }
def B : Point := { x := 2, y := -1 }
def reference_line : (ℝ × ℝ × ℝ) := (4, 3, -2)

-- Question and statement in Lean
theorem distance_between_l1_and_l2 : 
  let slope := (- reference_line.1 / reference_line.2)
  let l1 := line_through_point_with_perpendicular_slope A (3 / 4)
  let l2 := line_through_point_with_perpendicular_slope B (3 / 4)
  let distance := distance_between_parallel_lines l1.1 l1.2 l1.3 l2.3
  distance = 14 / 5 := by
  sorry

end distance_between_l1_and_l2_l152_152124


namespace light_ray_total_distance_l152_152287

theorem light_ray_total_distance 
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (M_eq : M = (2, 1))
  (N_eq : N = (4, 5)) :
  dist M N = 2 * Real.sqrt 10 := 
sorry

end light_ray_total_distance_l152_152287


namespace parabola_opens_upwards_l152_152607

theorem parabola_opens_upwards (x : ℝ) :
  ¬ (∃ y : ℝ, y = ⅓ * (x + 1) ^ 2 + 2 ∧ y < 2) :=
by
  sorry

end parabola_opens_upwards_l152_152607


namespace prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l152_152281

theorem prop_P_subset_q_when_m_eq_1 :
  ∀ x : ℝ, ∀ m : ℝ, m = 1 → (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) ↔ (x ∈ {x | 0 ≤ x ∧ x ≤ 2}) := 
by sorry

theorem range_m_for_necessity_and_not_sufficiency :
  ∀ m : ℝ, (∀ x : ℝ, (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) → (x ∈ {x | 1 - m ≤ x ∧ x ≤ 1 + m})) ↔ (m ≥ 9) := 
by sorry

end prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l152_152281


namespace subsets_of_Zmod_prime_l152_152735

theorem subsets_of_Zmod_prime {p : ℕ} [Fact (Nat.Prime p)] :
  {S : Set (ZMod p) | 
    (∀ a b ∈ S, a * b ∈ S) ∧ 
    (∃ r ∈ S, ∀ a ∈ S, r - a ∈ S ∨ r - a = 0)} = 
  {S : Set (ZMod p) | 
    (S \ {0} ⊆ {1}) ∨ 
    ((∃ H : Subgroup (ZMod p)ˣ, H.carrier = S \ {0}) ∧ (-1 : (ZMod p)ˣ ∈ H)} } :=
sorry

end subsets_of_Zmod_prime_l152_152735


namespace sin_gt_cos_lt_nec_suff_l152_152358

-- Define the triangle and the angles
variables {A B C : ℝ}
variables (t : triangle A B C)

-- Define conditions in the triangle: sum of angles is 180 degrees
axiom angle_sum : A + B + C = 180

-- Define sin and cos using the sides of the triangle
noncomputable def sin_A (A : ℝ) : ℝ := sorry -- placeholder for actual definition
noncomputable def sin_B (B : ℝ) : ℝ := sorry
noncomputable def cos_A (A : ℝ) : ℝ := sorry
noncomputable def cos_B (B : ℝ) : ℝ := sorry

-- The proposition to prove
theorem sin_gt_cos_lt_nec_suff {A B : ℝ} (h1 : sin_A A > sin_B B) :
  cos_A A < cos_B B ↔ sin_A A > sin_B B := sorry

end sin_gt_cos_lt_nec_suff_l152_152358


namespace total_cost_correct_l152_152152

def cost_of_sandwich : ℝ := 2 * 2.49
def cost_of_soda : ℝ := 4 * 1.87
def cost_of_chips : ℝ := 3 * 1.25
def cost_of_chocolate : ℝ := 5 * 0.99
def total_cost : ℝ := cost_of_sandwich + cost_of_soda + cost_of_chips + cost_of_chocolate

theorem total_cost_correct : total_cost = 21.16 := by
  unfold total_cost cost_of_sandwich cost_of_soda cost_of_chips cost_of_chocolate
  norm_num
  sorry

end total_cost_correct_l152_152152


namespace joe_started_diet_l152_152728

-- Conditions as definitions
def initial_weight: ℝ := 222
def current_weight: ℝ := 198
def weight_in_3_months: ℝ := 180
def time_frame: ℝ := 3

-- Constant rate of weight loss
def monthly_weight_loss_rate := (current_weight - weight_in_3_months) / time_frame

-- Calculate total weight lost so far
def total_weight_loss := (initial_weight - current_weight)

-- Prove that Joe started his diet 4 months ago
theorem joe_started_diet : total_weight_loss / monthly_weight_loss_rate = 4 :=
by sorry

end joe_started_diet_l152_152728


namespace find_y_from_area_l152_152791

-- Definition of vertices and area condition
def vertices (E F G H : (ℝ × ℝ)) := 
  E = (0, 0) ∧ F = (0, 5) ∧ G = (y, 5) ∧ H = (y, 0)

def rectangle_area (area : ℝ) (y : ℝ) :=
  (0 < y) ∧ (area = 5 * y)

-- The proof statement of the problem
theorem find_y_from_area (y : ℝ) (area : ℝ) 
  (H_vertices : vertices (0, 0) (0, 5) (y, 5) (y, 0)) 
  (H_area : rectangle_area area y) : 
  y = 8 :=
by
  sorry

end find_y_from_area_l152_152791


namespace negation_of_universal_l152_152434

-- Definitions based on the provided problem
def prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Main proof problem statement
theorem negation_of_universal : 
  ¬ (∀ x : ℝ, x > 0 → x^2 > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0 :=
by sorry

end negation_of_universal_l152_152434


namespace correct_equation_l152_152154

variable {a b x y : ℕ}

theorem correct_equation (hA : 3 * a ^ 2 - 2 * a ^ 2 ≠ 1) 
(hB : 5 * a + 2 * b ≠ 7 * a * b)
(hC : 3 * a ^ 2 - 2 * a ^ 2 ≠ 2 * a)
(hD : 5 * x * y ^ 2 - 6 * x * y ^ 2 = -x * y ^ 2) : 
  5 * x * y ^ 2 - 6 * x * y ^ 2 = -x * y ^ 2 :=
by sorry

end correct_equation_l152_152154


namespace min_distance_ellipse_to_line_l152_152242

theorem min_distance_ellipse_to_line :
  ∃ (P : ℝ × ℝ), (4 * P.1^2 + P.2^2 = 2) →
  let d := (|2 * P.1 - P.2 - 8|) / (sqrt(1^2 + 2^2))
  in d = (6 * sqrt 5) / 5 :=
by
  sorry

end min_distance_ellipse_to_line_l152_152242


namespace tree_leaves_l152_152916

theorem tree_leaves (initial_leaves : ℕ) (first_week_fraction : ℚ) (second_week_percentage : ℚ) (third_week_fraction : ℚ) :
  initial_leaves = 1000 →
  first_week_fraction = 2 / 5 →
  second_week_percentage = 40 / 100 →
  third_week_fraction = 3 / 4 →
  let leaves_after_first_week := initial_leaves - (first_week_fraction * initial_leaves).toNat,
      leaves_after_second_week := leaves_after_first_week - (second_week_percentage * leaves_after_first_week).toNat,
      leaves_after_third_week := leaves_after_second_week - (third_week_fraction * leaves_after_second_week).toNat
  in leaves_after_third_week = 90 :=
begin
  intros h1 h2 h3 h4,
  unfold leaves_after_first_week leaves_after_second_week leaves_after_third_week,
  rw [h1, h2, h3, h4],
  norm_num,
end

end tree_leaves_l152_152916


namespace max_initial_segment_length_l152_152820

theorem max_initial_segment_length (n m : ℕ) (a : ℕ) (b : ℕ) (h1 : n = 7) (h2 : m = 13) :
  ∃ k, k = 18 :=
by
  use 18
  sorry

end max_initial_segment_length_l152_152820


namespace problem_statement_l152_152817

theorem problem_statement (a b : ℝ) (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4) : a^2 + b^2 ≥ 2 := 
sorry

end problem_statement_l152_152817


namespace smallest_positive_angle_l152_152247

open Real

theorem smallest_positive_angle (θ : ℝ) :
  cos θ = sin (50 * (π / 180)) + cos (32 * (π / 180)) - sin (22 * (π / 180)) - cos (16 * (π / 180)) →
  θ = 90 * (π / 180) :=
by
  sorry

end smallest_positive_angle_l152_152247


namespace least_integer_square_l152_152145

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l152_152145


namespace room_width_l152_152121

theorem room_width (length : ℝ) (cost : ℝ) (rate : ℝ) (h_length : length = 5.5)
                    (h_cost : cost = 16500) (h_rate : rate = 800) : 
                    (cost / rate / length = 3.75) :=
by 
  sorry

end room_width_l152_152121


namespace coin_probability_l152_152886

theorem coin_probability (p : ℝ) (h1 : p < 1/2) (h2 : (Nat.choose 6 3) * p^3 * (1-p)^3 = 1/20) : p = 1/400 := sorry

end coin_probability_l152_152886


namespace moles_of_NaCl_l152_152243

def moles_of_reactants (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

theorem moles_of_NaCl (NaCl KNO3 NaNO3 KCl : ℕ) 
  (h : moles_of_reactants NaCl KNO3 NaNO3 KCl) 
  (h2 : KNO3 = 1)
  (h3 : NaNO3 = 1) :
  NaCl = 1 :=
by
  sorry

end moles_of_NaCl_l152_152243


namespace cos_inequality_l152_152082

theorem cos_inequality (x y : ℝ) (h : 0 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ π) : 
  cos x + cos y ≤ 1 + cos (x * y) :=
by
  sorry

end cos_inequality_l152_152082


namespace max_min_values_of_f_l152_152572

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x ∈ Icc (2 : ℝ) (3 : ℝ), f x ≤ f 2) ∧ (∀ x ∈ Icc (2 : ℝ) (3 : ℝ), f x ≥ f 3) :=
by
  sorry

end max_min_values_of_f_l152_152572


namespace magic_show_l152_152513

theorem magic_show (performances : ℕ) (prob_never_reappear : ℚ) (prob_two_reappear : ℚ)
  (h_performances : performances = 100)
  (h_prob_never_reappear : prob_never_reappear = 1 / 10)
  (h_prob_two_reappear : prob_two_reappear = 1 / 5) :
  let never_reappear := prob_never_reappear * performances,
      two_reappear := prob_two_reappear * performances,
      normal_reappear := performances,
      extra_reappear := two_reappear,
      total_reappear := normal_reappear + extra_reappear - never_reappear in
  total_reappear = 110 := by
  sorry

end magic_show_l152_152513


namespace intersection_M_N_l152_152676

noncomputable theory

open Set Real

def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {x | log 2 x > 1}

theorem intersection_M_N : (M ∩ N) = Ioc 2 3 := by
  sorry

end intersection_M_N_l152_152676


namespace number_of_routes_from_A_to_L_l152_152806

-- Definition of the graph and required path properties
noncomputable def vertex := ℕ -- Representing the city as a vertex by natural numbers
noncomputable def roads : list (vertex × vertex) := 
  [(0, 1), (0, 2), ...] -- Example list of 17 roads connecting pairs of cities

def number_of_valid_routes (start end : vertex) (num_roads : ℕ) : ℕ :=
  -- A naive and placeholder count of valid routes. The actual implementation would be more complex.
  if (start, end, num_roads) = (0, 11, 13) then 4 else 0

theorem number_of_routes_from_A_to_L : 
  number_of_valid_routes 0 11 13 = 4 := 
by
  sorry

end number_of_routes_from_A_to_L_l152_152806


namespace average_speeds_of_A_and_B_l152_152139

-- We define the problem conditions
variables (A B : Type) [LinearOrder A] [LinearOrder B]

-- Given conditions: lengths of the track and times to meet/catch up
def L : ℝ := 400 -- length of the track
def t_meet : ℝ := 25 -- time to meet running in opposite directions
def t_catch : ℝ := 200 -- time to catch up running in the same direction

-- Define the speeds of A and B
variables (x y : ℝ) -- speeds of A and B

-- Equations derived from the problem conditions
def equation1 : Prop := t_meet * x + t_meet * y = L
def equation2 : Prop := t_catch * x - t_catch * y = L

-- Statement to prove
theorem average_speeds_of_A_and_B :
  (equation1 x y) ∧ (equation2 x y) → x = 9 ∧ y = 7 :=
begin
  sorry, -- This part is left as an exercise for proving the statement.
end

end average_speeds_of_A_and_B_l152_152139


namespace distinct_equilateral_triangles_in_dodecagon_l152_152308

theorem distinct_equilateral_triangles_in_dodecagon : 
  let vertices := (list.iota 12).map (λ i, Complex.polar 1 (((2 * Real.pi) / 12) * i + 1)) in
  count_distinct_equilateral_triangles(vertices) = 124 :=
by
  sorry

end distinct_equilateral_triangles_in_dodecagon_l152_152308


namespace male_student_number_l152_152186

theorem male_student_number (year class_num student_num : ℕ) (h_year : year = 2011) (h_class : class_num = 6) (h_student : student_num = 23) : 
  (100000 * year + 1000 * class_num + 10 * student_num + 1 = 116231) :=
by
  sorry

end male_student_number_l152_152186


namespace equality_of_fractions_l152_152320

theorem equality_of_fractions
  (a b c k m n : ℝ)
  (h : b^2 - n^2 = a^2 - k^2 ∧ a^2 - k^2 = c^2 - m^2) :
  (bm - cn) / (a - k) + (ck - am) / (b - n) + (an - bk) / (c - m) = 0 := 
begin
  sorry
end

end equality_of_fractions_l152_152320


namespace probability_differs_by_three_l152_152933

theorem probability_differs_by_three :
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 3), (8, 5)],
      num_outcomes := List.length outcomes,
      total_possibilities := 8 * 8
  in
  Rational.mk num_outcomes total_possibilities = Rational.mk 7 64 :=
by
  sorry

end probability_differs_by_three_l152_152933


namespace target_run_proof_l152_152714

-- Define the given conditions
def run_rate_first_8_overs : ℝ := 2.3
def overs_first_8 : ℕ := 8
def run_rate_remaining_20_overs : ℝ := 12.08
def overs_remaining_20 : ℕ := 20

-- Calculate the total runs in the first 8 overs, implicitly involving rounding down
def runs_first_8_overs : ℕ := ⌊run_rate_first_8_overs * overs_first_8⌋₊

-- Calculate the total runs in the remaining 20 overs, implicitly involving rounding down
def runs_remaining_20_overs : ℕ := ⌊run_rate_remaining_20_overs * overs_remaining_20⌋₊

-- Define the target number of runs
def target_runs : ℕ := runs_first_8_overs + runs_remaining_20_overs

-- Prove that the target number of runs is 259 runs
theorem target_run_proof : target_runs = 259 :=
by
  -- We replace the Lean proof tactic with sorry to skip the proof, as specified.
  sorry

end target_run_proof_l152_152714


namespace sum_of_products_of_A_subsets_l152_152390

-- Define the set A
def A : Set ℚ := {1/2, 1/7, 1/11, 1/13, 1/15, 1/32}

-- Define the function that calculates the product of elements in a finite set over rationals
def product_set (s : Finset ℚ) : ℚ :=
  s.prod id

-- Define the sum of the products of all non-empty subsets of A
def sum_of_products_of_subsets (A : Set ℚ) : ℚ :=
  (∑ s in (Finset.powerset (Finset.filter (fun x => x ∈ A) Finset.univ)).filter (λ s, s.nonempty),
    product_set s)

-- Target theorem
theorem sum_of_products_of_A_subsets :
  sum_of_products_of_subsets A = 79/65 :=
by
  sorry

end sum_of_products_of_A_subsets_l152_152390


namespace minimum_value_l152_152661

-- Define the function y based on the given conditions
def y (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

-- Prove that when x = 2, the minimum value b of y is 1, and thus a + b = 3
theorem minimum_value :
  (∃ a b : ℝ, y a = b ∧ a = 2 ∧ b = 1) → a + b = 3 :=
by
  intros h
  rcases h with ⟨a, b, hab1, ha, hb⟩
  rw [ha, hb]
  exact rfl

end minimum_value_l152_152661


namespace freezer_temperature_is_minus_12_l152_152333

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l152_152333


namespace exists_infinitely_many_solutions_l152_152979

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/a + 1/b + 1/c + 1/(a*b*c) = m / (a + b + c))) :=
sorry

end exists_infinitely_many_solutions_l152_152979


namespace bijective_saturating_matching_exists_l152_152057

noncomputable theory

open Set

variables {A B : Type} [Infinite A] [Infinite B]
variables (E : Set (A × B))
variables (deg : A → ℕ) [∀ a, 0 < deg a] [∀ b, 0 < deg b]

def bipartite_graph (A B : Type) (E : Set (A × B)) : Prop :=
  ∀ (a : A) (b : B), (a, b) ∈ E → deg a = deg b

theorem bijective_saturating_matching_exists
  (h1 : bipartite_graph A B E)
  (h2 : ∀ a, 0 < deg a ∧ ∃ fin_supp : Finset A, #{ x ∈ E | x.1 = a }.toFinset = fin_supp)
  (h3 : ∀ b, 0 < deg b ∧ ∃ fin_supp : Finset B, #{ x ∈ E | x.2 = b }.toFinset = fin_supp) :
  ∃ f : A → B, Function.Bijective f ∧ (∀ a ∈ A, (a, f a) ∈ E) :=
sorry

end bijective_saturating_matching_exists_l152_152057


namespace charge_per_call_proof_l152_152888

-- Define the conditions as given in the problem
def fixed_rental : ℝ := 350
def free_calls_per_month : ℕ := 200
def charge_per_call_exceed_200 (x : ℝ) (calls : ℕ) : ℝ := 
  if calls > 200 then (calls - 200) * x else 0

def charge_per_call_exceed_400 : ℝ := 1.6
def discount_rate : ℝ := 0.28
def february_calls : ℕ := 150
def march_calls : ℕ := 250
def march_discount (x : ℝ) : ℝ := x * (1 - discount_rate)
def total_march_charge (x : ℝ) : ℝ := 
  fixed_rental + charge_per_call_exceed_200 (march_discount x) march_calls

-- Prove the correct charge per call when calls exceed 200 per month
theorem charge_per_call_proof (x : ℝ) : 
  charge_per_call_exceed_200 x february_calls = 0 ∧ 
  total_march_charge x = fixed_rental + (march_calls - free_calls_per_month) * (march_discount x) → 
  x = x := 
by { 
  sorry 
}

end charge_per_call_proof_l152_152888


namespace hardest_vs_least_worked_hours_difference_l152_152482

-- Let x be the scaling factor for the ratio
-- The times worked are 2x, 3x, and 4x

def project_time_difference (x : ℕ) : Prop :=
  let time1 := 2 * x
  let time2 := 3 * x
  let time3 := 4 * x
  (time1 + time2 + time3 = 90) ∧ ((4 * x - 2 * x) = 20)

theorem hardest_vs_least_worked_hours_difference :
  ∃ x : ℕ, project_time_difference x :=
by
  sorry

end hardest_vs_least_worked_hours_difference_l152_152482


namespace number_of_green_marbles_l152_152702

-- Define the conditions in terms of fractions and a constant number of white marbles.
variable (total_marbles : ℕ)
variable (red_fraction blue_fraction green_fraction white_marbles : ℚ)

-- Given conditions
def all_conditions :=
  red_fraction = 1 / 4 ∧
  blue_fraction = 1 / 3 ∧
  green_fraction = 1 / 6 ∧
  white_marbles = 40 ∧
  white_fraction = 1 - (red_fraction + blue_fraction + green_fraction)

-- Introducing total number of marbles based on white marbles fraction
def total_marbles := white_marbles / white_fraction

-- Proof that the number of green marbles is 27
theorem number_of_green_marbles (cond : all_conditions) : 
  total_marbles * green_fraction = 27 := by
  sorry

end number_of_green_marbles_l152_152702


namespace place_values_of_9890_l152_152133

theorem place_values_of_9890 :
  ∃ (t h te : ℕ), 1000 * t + 100 * h + 10 * te + 0 = 9890 ∧ t = 9 ∧ h = 8 ∧ te = 9 :=
by
  use 9, 8, 9
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end place_values_of_9890_l152_152133


namespace problem_correctness_l152_152465

theorem problem_correctness :
  ∀ (x y a b : ℝ), (-3:ℝ)^2 ≠ -9 ∧
    - (x + y) = -x - y ∧
    3 * a + 5 * b ≠ 8 * a * b ∧
    5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := 
by
  intro x y a b
  split
  · norm_num
  split
  · ring
  split
  · linarith
  · ring

end problem_correctness_l152_152465


namespace average_age_A_union_B_union_C_l152_152488

-- Definitions of the sets and their respective properties
variables (A B C : Set Person)
variables (sumA sumB sumC : ℕ)
variables (numA numB numC : ℕ)

-- Conditions
def A_avg_age : Prop := (A.avg_age = 30)
def B_avg_age : Prop := (B.avg_age = 25)
def C_avg_age : Prop := (C.avg_age = 45)
def A_union_B_avg_age : Prop := (A ∪ B).avg_age = 27
def A_union_C_avg_age : Prop := (A ∪ C).avg_age = 40
def B_union_C_avg_age : Prop := (B ∪ C).avg_age = 35

-- The theorem to prove the average age of A ∪ B ∪ C
theorem average_age_A_union_B_union_C :
  A_avg_age A -> B_avg_age B -> C_avg_age C -> 
  A_union_B_avg_age A B -> A_union_C_avg_age A C -> B_union_C_avg_age B C ->
  (A ∪ B ∪ C).avg_age = 35 :=
by sorry

end average_age_A_union_B_union_C_l152_152488


namespace annabelle_savings_l152_152544

theorem annabelle_savings (weekly_allowance : ℕ) (junk_food_fraction : ℚ) (sweets_cost : ℕ) 
    (h1 : weekly_allowance = 30) 
    (h2 : junk_food_fraction = 1 / 3) 
    (h3 : sweets_cost = 8) : 
    weekly_allowance - (weekly_allowance * (junk_food_fraction.numerator / junk_food_fraction.denominator)) - sweets_cost = 12 :=
by
  sorry

end annabelle_savings_l152_152544


namespace total_cost_of_color_drawing_l152_152364

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end total_cost_of_color_drawing_l152_152364


namespace digit_7_frequency_limit_l152_152552

-- Definition of N(n): number of occurrences of digit 7 in decimal representations from 1 to n
def N (n : ℕ) : ℕ := sorry

-- Definition of M(n): total number of decimal digits in representations of numbers from 1 to n
def M (n : ℕ) : ℕ := sorry

theorem digit_7_frequency_limit :
  ∀ (N : ℕ → ℕ) (M : ℕ → ℕ), 
  (∀ n, N n = sorry ∧ M n = sorry) →
  tendsto (λ n, (N n) / (M n) : ℕ → ℝ) at_top (𝓝 (1 / 10)) :=
by
  sorry

end digit_7_frequency_limit_l152_152552


namespace probability_divisible_by_three_l152_152615

theorem probability_divisible_by_three :
  ∃ probability : ℚ, probability = 2/5 ∧
  (∀ (a b : ℕ), a ≠ b → a ∈ {1, 2, 3, 4, 5} → b ∈ {1, 2, 3, 4, 5} → 
    let num := 10 * a + b in 
    (num % 3 = 0 ↔ probability = 2/5)) :=
by
  sorry

end probability_divisible_by_three_l152_152615


namespace minimum_pairs_of_acquainted_schoolchildren_l152_152878

theorem minimum_pairs_of_acquainted_schoolchildren :
  (∃ (n : ℕ) (k : ℕ) (m : ℕ) (x : ℕ),
    n = 175 ∧
    k = 6 ∧
    m = 3 ∧
    x = 15050 ∧
    ∀ S : Finset (Fin 175),
      S.card = k →
      (∃ (R1 R2 : Finset (Fin 175)),
        R1 ∪ R2 = S ∧
        R1.card = m ∧
        R2.card = m ∧
        (∀ a b ∈ R1, knows a b) ∧
        (∀ a b ∈ R2, knows a b))) →
  (∃ (p : ℕ),
    p = 15050) :=
by
  sorry

end minimum_pairs_of_acquainted_schoolchildren_l152_152878


namespace expression_for_f_log_sequence_geometric_C_min_l152_152276

-- Define the quadratic function considering the provided conditions
def f (x: ℝ) : ℝ := x^2 + 2 * x

-- Define the sequence with initial value a1 = 9
def a : ℕ → ℝ
| 0 => 9   -- Using 0-based indexing; meaning a(0) = a1
| (n + 1) => (a n)^2 + 2 * (a n)

-- Define the logarithmic sequence
def lg_seq (n: ℕ) : ℝ := Real.log (1 + a n)

-- Define C_n
def C (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * lg_seq n) / (n^2)

-- Prove the expression for f(x)
theorem expression_for_f : f(1) = 3 :=
by
  simp [f]
  norm_num

-- Prove the sequence {lg(1 + a_n)} is geometric
theorem log_sequence_geometric (n : ℕ) : lg_seq (n + 1) = 2 * lg_seq n :=
by
  simp [lg_seq, a]
  rw [Real.log_pow]
  simp
  intro h
  linarith

-- Prove the minimum C_n and find n_0
theorem C_min (n_0 : ℕ) : n_0 = 3 ∧ ∀ n, C (n + 1) > C n ∨ n = 3 :=
by
  sorry

end expression_for_f_log_sequence_geometric_C_min_l152_152276


namespace line_canonical_equations_l152_152475

theorem line_canonical_equations :
  ∀ (x y z : ℝ), 2 * x + 3 * y + z + 6 = 0 → x - 3 * y - 2 * z + 3 = 0 →
  (∃ (k : ℝ), x = -3 + -3 * k ∧ y = 5 * k ∧ z = -9 * k) :=
by
sintro x y z
intros h1 h2
have h : ∀ (k : ℝ), x = -3 + -3 * k ∧ y = 5 * k ∧ z = -9 * k := sorry
exact ⟨_, h (1:ℚ)⟩

end line_canonical_equations_l152_152475


namespace sum_of_repeating_decimals_l152_152746

def repeating_decimals_sum : Real :=
  let T := {x | ∃ (a b : ℕ), a ≠ b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = (10 * a + b) / 99}
  Set.sum T

theorem sum_of_repeating_decimals : repeating_decimals_sum = 413.5 :=
  by
    sorry

end sum_of_repeating_decimals_l152_152746


namespace field_ratio_l152_152119

theorem field_ratio (w l: ℕ) (h: l = 36)
  (h_area_ratio: 81 = (1/8) * l * w)
  (h_multiple: ∃ k : ℕ, l = k * w) :
  l / w = 2 :=
by 
  sorry

end field_ratio_l152_152119


namespace sum_unique_three_digit_numbers_l152_152613

theorem sum_unique_three_digit_numbers : 
  let digits := [1, 3, 5] in
  let three_digit_numbers := 
    digits.bind (λ a, 
      (digits.erase a).bind (λ b, 
        (digits.erase a).erase b).map (λ c, 100 * a + 10 * b + c)) in
  three_digit_numbers.sum = 1998 :=
by
  let digits := [1, 3, 5]
  let three_digit_numbers := 
    digits.bind (λ a, 
      (digits.erase a).bind (λ b, 
        (digits.erase a).erase b).map (λ c, 100 * a + 10 * b + c))
  have : three_digit_numbers = [135, 153, 315, 351, 513, 531] := sorry
  rw this
  unfold List.sum
  norm_num
  sorry

end sum_unique_three_digit_numbers_l152_152613


namespace new_percentage_of_girls_l152_152495

theorem new_percentage_of_girls (initial_total students: ℕ) (initial_girls_percent initial_boys_percent: ℝ) 
                                (new_boys_added new_girls_added boys_left girls_left extra_girls_added: ℕ):
  initial_total = 200 →
  initial_girls_percent = 0.45 →
  initial_boys_percent = 0.55 →
  new_boys_added = 15 →
  new_girls_added = 20 →
  boys_left = 10 →
  girls_left = 5 →
  extra_girls_added = 3 →
  let initial_girls := initial_girls_percent * initial_total,
      initial_boys := initial_boys_percent * initial_total,
      net_girls := new_girls_added + extra_girls_added - girls_left,
      net_boys := new_boys_added - boys_left,
      total_girls := initial_girls + net_girls,
      total_boys := initial_boys + net_boys,
      new_total := total_girls + total_boys in
  (total_girls / new_total) * 100 ≈ 48.43 :=
begin
  intros,
  -- Sorry is used to skip the actual proof
  sorry
end

end new_percentage_of_girls_l152_152495


namespace math_equivalence_proof_problem_l152_152849

-- Define the initial radii in L0
def r1 := 50^2
def r2 := 53^2

-- Define the formula for constructing a new circle in subsequent layers
def next_radius (r1 r2 : ℕ) : ℕ :=
  (r1 * r2) / ((Nat.sqrt r1 + Nat.sqrt r2)^2)

-- Compute the sum of reciprocals of the square roots of the radii 
-- of all circles up to and including layer L6
def sum_of_reciprocals_of_square_roots_up_to_L6 : ℚ :=
  let initial_sum := (1 / (50 : ℚ)) + (1 / (53 : ℚ))
  (127 * initial_sum) / (50 * 53)

theorem math_equivalence_proof_problem : 
  sum_of_reciprocals_of_square_roots_up_to_L6 = 13021 / 2650 := 
sorry

end math_equivalence_proof_problem_l152_152849


namespace _l152_152213

noncomputable def urn_marble_theorem (r w b g y : Nat) : Prop :=
  let n := r + w + b + g + y
  ∃ k : Nat, 
  (k * r * (r-1) * (r-2) * (r-3) * (r-4) / 120 = w * r * (r-1) * (r-2) * (r-3) / 24)
  ∧ (w * r * (r-1) * (r-2) * (r-3) / 24 = w * b * r * (r-1) * (r-2) / 6)
  ∧ (w * b * r * (r-1) * (r-2) / 6 = w * b * g * r * (r-1) / 2)
  ∧ (w * b * g * r * (r-1) / 2 = w * b * g * r * y)
  ∧ n = 55

example : ∃ (r w b g y : Nat), urn_marble_theorem r w b g y := sorry

end _l152_152213


namespace positive_integer_solutions_count_l152_152804

theorem positive_integer_solutions_count :
  {n : ℕ // n > 0} × {m : ℕ // m > 0} → Prop :=
begin
  assume ⟨x, hx⟩ ⟨y, hy⟩,
  3 * x + y = 9 → 
    ((x = 1 ∧ y = 6) ∨
     (x = 2 ∧ y = 3)) ∧
    ∀ x' y', ((3 * x' + y' = 9) →
              x' > 0 → y' > 0 →
              (x' = 1 ∧ y' = 6) ∨
              (x' = 2 ∧ y' = 3)) := sorry

end positive_integer_solutions_count_l152_152804


namespace largest_tangent_slope_l152_152002

theorem largest_tangent_slope (c : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → c * (x - (sqrt 10)/2) = y - (sqrt 10)/2) →
  c ≤ 3 :=
sorry

end largest_tangent_slope_l152_152002
