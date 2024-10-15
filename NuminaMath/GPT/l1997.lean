import Mathlib

namespace NUMINAMATH_GPT_building_height_l1997_199759

theorem building_height (h : ℕ) 
  (shadow_building : ℕ) 
  (shadow_pole : ℕ) 
  (height_pole : ℕ) 
  (ratio_proportional : shadow_building * height_pole = shadow_pole * h) 
  (shadow_building_val : shadow_building = 63) 
  (shadow_pole_val : shadow_pole = 32) 
  (height_pole_val : height_pole = 28) : 
  h = 55 := 
by 
  sorry

end NUMINAMATH_GPT_building_height_l1997_199759


namespace NUMINAMATH_GPT_euler_line_of_isosceles_triangle_l1997_199749

theorem euler_line_of_isosceles_triangle (A B : ℝ × ℝ) (hA : A = (2,0)) (hB : B = (0,4)) (C : ℝ × ℝ) (hC1 : dist A C = dist B C) :
  ∃ a b c : ℝ, a * (C.1 - 2) + b * (C.2 - 0) + c = 0 ∧ x - 2 * y + 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_euler_line_of_isosceles_triangle_l1997_199749


namespace NUMINAMATH_GPT_relationship_between_variables_l1997_199714

theorem relationship_between_variables
  (a b x y : ℚ)
  (h1 : x + y = a + b)
  (h2 : y - x < a - b)
  (h3 : b > a) :
  y < a ∧ a < b ∧ b < x :=
sorry

end NUMINAMATH_GPT_relationship_between_variables_l1997_199714


namespace NUMINAMATH_GPT_min_value_xy_l1997_199787

theorem min_value_xy {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) : x * y ≥ 64 :=
sorry

end NUMINAMATH_GPT_min_value_xy_l1997_199787


namespace NUMINAMATH_GPT_set_intersection_complement_l1997_199729

open Set

theorem set_intersection_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  (U \ A) ∩ B = {3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1997_199729


namespace NUMINAMATH_GPT_dividend_is_217_l1997_199775

-- Given conditions
def r : ℕ := 1
def q : ℕ := 54
def d : ℕ := 4

-- Define the problem as a theorem in Lean 4
theorem dividend_is_217 : (d * q) + r = 217 := by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_dividend_is_217_l1997_199775


namespace NUMINAMATH_GPT_jelly_bean_probability_l1997_199782

theorem jelly_bean_probability :
  ∀ (P_red P_orange P_green P_yellow : ℝ),
  P_red = 0.1 →
  P_orange = 0.4 →
  P_green = 0.2 →
  P_red + P_orange + P_green + P_yellow = 1 →
  P_yellow = 0.3 :=
by
  intros P_red P_orange P_green P_yellow h_red h_orange h_green h_sum
  sorry

end NUMINAMATH_GPT_jelly_bean_probability_l1997_199782


namespace NUMINAMATH_GPT_abs_condition_sufficient_not_necessary_l1997_199700

theorem abs_condition_sufficient_not_necessary:
  (∀ x : ℝ, (-2 < x ∧ x < 3) → (-1 < x ∧ x < 3)) :=
by
  sorry

end NUMINAMATH_GPT_abs_condition_sufficient_not_necessary_l1997_199700


namespace NUMINAMATH_GPT_more_than_10_weights_missing_l1997_199738

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end NUMINAMATH_GPT_more_than_10_weights_missing_l1997_199738


namespace NUMINAMATH_GPT_systematic_sampling_methods_l1997_199786

-- Definitions for sampling methods ①, ②, ④
def sampling_method_1 : Prop :=
  ∀ (l : ℕ), (l ≤ 15 ∧ l + 5 ≤ 15 ∧ l + 10 ≤ 15 ∨
              l ≤ 15 ∧ l + 5 ≤ 20 ∧ l + 10 ≤ 20) → True

def sampling_method_2 : Prop :=
  ∀ (t : ℕ), (t % 5 = 0) → True

def sampling_method_3 : Prop :=
  ∀ (n : ℕ), (n > 0) → True

def sampling_method_4 : Prop :=
  ∀ (row : ℕ) (seat : ℕ), (seat = 12) → True

-- Equivalence Proof Statement
theorem systematic_sampling_methods :
  sampling_method_1 ∧ sampling_method_2 ∧ sampling_method_4 :=
by sorry

end NUMINAMATH_GPT_systematic_sampling_methods_l1997_199786


namespace NUMINAMATH_GPT_monotone_increasing_interval_for_shifted_function_l1997_199751

variable (f : ℝ → ℝ)

-- Given definition: f(x+1) is an even function
def even_function : Prop :=
  ∀ x, f (x+1) = f (-(x+1))

-- Given condition: f(x+1) is monotonically decreasing on [0, +∞)
def monotone_decreasing_on_nonneg : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f (x+1) ≥ f (y+1)

-- Theorem to prove: the interval on which f(x-1) is monotonically increasing is (-∞, 2]
theorem monotone_increasing_interval_for_shifted_function
  (h_even : even_function f)
  (h_mono_dec : monotone_decreasing_on_nonneg f) :
  ∀ x y, x ≤ 2 → y ≤ 2 → x ≤ y → f (x-1) ≤ f (y-1) :=
by
  sorry

end NUMINAMATH_GPT_monotone_increasing_interval_for_shifted_function_l1997_199751


namespace NUMINAMATH_GPT_power_equivalence_l1997_199734

theorem power_equivalence (m : ℕ) : 16^6 = 4^m → m = 12 :=
by
  sorry

end NUMINAMATH_GPT_power_equivalence_l1997_199734


namespace NUMINAMATH_GPT_find_m_and_p_l1997_199783

-- Definition of a point being on the parabola y^2 = 2px
def on_parabola (m : ℝ) (p : ℝ) : Prop :=
  (-3)^2 = 2 * p * m

-- Definition of the distance from the point (m, -3) to the focus being 5
def distance_to_focus (m : ℝ) (p : ℝ) : Prop :=
  m + p / 2 = 5

theorem find_m_and_p (m p : ℝ) (hp : 0 < p) : 
  (on_parabola m p) ∧ (distance_to_focus m p) → 
  (m = 1 / 2 ∧ p = 9) ∨ (m = 9 / 2 ∧ p = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_p_l1997_199783


namespace NUMINAMATH_GPT_neg_q_is_true_l1997_199743

variable (p q : Prop)

theorem neg_q_is_true (hp : p) (hq : ¬ q) : ¬ q :=
by
  exact hq

end NUMINAMATH_GPT_neg_q_is_true_l1997_199743


namespace NUMINAMATH_GPT_find_m_n_sum_l1997_199707

theorem find_m_n_sum (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : 
  m + n = 211 :=
sorry

end NUMINAMATH_GPT_find_m_n_sum_l1997_199707


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1997_199755

noncomputable def find_time_when_ball_hits_ground (a b c : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

theorem ball_hits_ground_time :
  find_time_when_ball_hits_ground (-16) 40 50 = (5 + 5 * Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1997_199755


namespace NUMINAMATH_GPT_min_distance_origin_to_line_l1997_199778

noncomputable def distance_from_origin_to_line(A B C : ℝ) : ℝ :=
  let d := |A * 0 + B * 0 + C| / (Real.sqrt (A^2 + B^2))
  d

theorem min_distance_origin_to_line : distance_from_origin_to_line 1 1 (-4) = 2 * Real.sqrt 2 := by 
  sorry

end NUMINAMATH_GPT_min_distance_origin_to_line_l1997_199778


namespace NUMINAMATH_GPT_student_marks_equals_125_l1997_199765

-- Define the maximum marks
def max_marks : ℕ := 500

-- Define the percentage required to pass
def pass_percentage : ℚ := 33 / 100

-- Define the marks required to pass
def pass_marks : ℚ := pass_percentage * max_marks

-- Define the marks by which the student failed
def fail_by_marks : ℕ := 40

-- Define the obtained marks by the student
def obtained_marks : ℚ := pass_marks - fail_by_marks

-- Prove that the obtained marks are 125
theorem student_marks_equals_125 : obtained_marks = 125 := by
  sorry

end NUMINAMATH_GPT_student_marks_equals_125_l1997_199765


namespace NUMINAMATH_GPT_chef_additional_wings_l1997_199727

theorem chef_additional_wings
    (n : ℕ) (w_initial : ℕ) (w_per_friend : ℕ) (w_additional : ℕ)
    (h1 : n = 4)
    (h2 : w_initial = 9)
    (h3 : w_per_friend = 4)
    (h4 : w_additional = 7) :
    n * w_per_friend - w_initial = w_additional :=
by
  sorry

end NUMINAMATH_GPT_chef_additional_wings_l1997_199727


namespace NUMINAMATH_GPT_exactly_one_even_needs_assumption_l1997_199732

open Nat

theorem exactly_one_even_needs_assumption 
  {a b c : ℕ} 
  (h : (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ (a % 2 = 0 → b % 2 = 1) ∧ (a % 2 = 0 → c % 2 = 1) ∧ (b % 2 = 0 → c % 2 = 1)) :
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) → (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) → (¬(a % 2 = 0 ∧ b % 2 = 0) ∧ ¬(b % 2 = 0 ∧ c % 2 = 0) ∧ ¬(a % 2 = 0 ∧ c % 2 = 0)) := 
by
  sorry

end NUMINAMATH_GPT_exactly_one_even_needs_assumption_l1997_199732


namespace NUMINAMATH_GPT_negative_comparison_l1997_199769

theorem negative_comparison : -2023 > -2024 :=
sorry

end NUMINAMATH_GPT_negative_comparison_l1997_199769


namespace NUMINAMATH_GPT_find_dividend_l1997_199794

theorem find_dividend (partial_product : ℕ) (remainder : ℕ) (divisor quotient : ℕ) :
  partial_product = 2015 → 
  remainder = 0 →
  divisor = 105 → 
  quotient = 197 → 
  divisor * quotient + remainder = partial_product → 
  partial_product * 10 = 20685 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_find_dividend_l1997_199794


namespace NUMINAMATH_GPT_inequality_a3_b3_c3_l1997_199791

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_a3_b3_c3_l1997_199791


namespace NUMINAMATH_GPT_find_n_l1997_199723

theorem find_n (x y : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) :
  ∃ n : ℝ, n = 2 := by
  sorry

end NUMINAMATH_GPT_find_n_l1997_199723


namespace NUMINAMATH_GPT_trigonometric_identity_l1997_199722

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1997_199722


namespace NUMINAMATH_GPT_congruent_triangles_have_equal_perimeters_and_areas_l1997_199767

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (A B C : ℝ) -- angles of the triangle

def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- perimeters and areas (assuming some function calc_perimeter and calc_area for simplicity)
def perimeter (Δ : Triangle) : ℝ := Δ.a + Δ.b + Δ.c
def area (Δ : Triangle) : ℝ := sorry -- implement area calculation, e.g., using Heron's formula

-- Statement to be proved
theorem congruent_triangles_have_equal_perimeters_and_areas (Δ1 Δ2 : Triangle) :
  congruent_triangles Δ1 Δ2 →
  perimeter Δ1 = perimeter Δ2 ∧ area Δ1 = area Δ2 :=
sorry

end NUMINAMATH_GPT_congruent_triangles_have_equal_perimeters_and_areas_l1997_199767


namespace NUMINAMATH_GPT_peter_reads_one_book_18_hours_l1997_199792

-- Definitions of conditions given in the problem
variables (P : ℕ)

-- Condition: Peter can read three times as fast as Kristin
def reads_three_times_as_fast (P : ℕ) : Prop :=
  ∀ (K : ℕ), K = 3 * P

-- Condition: Kristin reads half of her 20 books in 540 hours
def half_books_in_540_hours (K : ℕ) : Prop :=
  K = 54

-- Theorem stating the main proof problem: proving P equals 18 hours
theorem peter_reads_one_book_18_hours
  (H1 : reads_three_times_as_fast P)
  (H2 : half_books_in_540_hours (3 * P)) :
  P = 18 :=
sorry

end NUMINAMATH_GPT_peter_reads_one_book_18_hours_l1997_199792


namespace NUMINAMATH_GPT_grapes_difference_l1997_199706

theorem grapes_difference (R A_i A_l : ℕ) 
  (hR : R = 25) 
  (hAi : A_i = R + 2) 
  (hTotal : R + A_i + A_l = 83) : 
  A_l - A_i = 4 := 
by
  sorry

end NUMINAMATH_GPT_grapes_difference_l1997_199706


namespace NUMINAMATH_GPT_mango_price_reduction_l1997_199761

theorem mango_price_reduction (P R : ℝ) (M : ℕ)
  (hP_orig : 110 * P = 366.67)
  (hM : M * P = 360)
  (hR_red : (M + 12) * R = 360) :
  ((P - R) / P) * 100 = 10 :=
by sorry

end NUMINAMATH_GPT_mango_price_reduction_l1997_199761


namespace NUMINAMATH_GPT_complex_power_six_l1997_199710

theorem complex_power_six (i : ℂ) (hi : i * i = -1) : (1 + i)^6 = -8 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_power_six_l1997_199710


namespace NUMINAMATH_GPT_proj_w_v_is_v_l1997_199702

noncomputable def proj_w_v (v w : ℝ × ℝ) : ℝ × ℝ :=
  let c := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (c * w.1, c * w.2)

def v : ℝ × ℝ := (-3, 2)
def w : ℝ × ℝ := (4, -2)

theorem proj_w_v_is_v : proj_w_v v w = v := 
  sorry

end NUMINAMATH_GPT_proj_w_v_is_v_l1997_199702


namespace NUMINAMATH_GPT_incorrect_expression_l1997_199740

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (1 - 3*x > 1 - 3*y) :=
sorry

end NUMINAMATH_GPT_incorrect_expression_l1997_199740


namespace NUMINAMATH_GPT_olaf_total_cars_l1997_199717

noncomputable def olaf_initial_cars : ℕ := 150
noncomputable def uncle_cars : ℕ := 5
noncomputable def grandpa_cars : ℕ := 2 * uncle_cars
noncomputable def dad_cars : ℕ := 10
noncomputable def mum_cars : ℕ := dad_cars + 5
noncomputable def auntie_cars : ℕ := 6
noncomputable def liam_cars : ℕ := dad_cars / 2
noncomputable def emma_cars : ℕ := uncle_cars / 3
noncomputable def grandma_cars : ℕ := 3 * auntie_cars

noncomputable def total_gifts : ℕ := 
  grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars + liam_cars + emma_cars + grandma_cars

noncomputable def total_cars_after_gifts : ℕ := olaf_initial_cars + total_gifts

theorem olaf_total_cars : total_cars_after_gifts = 220 := by
  sorry

end NUMINAMATH_GPT_olaf_total_cars_l1997_199717


namespace NUMINAMATH_GPT_band_total_l1997_199770

theorem band_total (flutes_total clarinets_total trumpets_total pianists_total : ℕ)
                   (flutes_pct clarinets_pct trumpets_pct pianists_pct : ℚ)
                   (h_flutes : flutes_total = 20)
                   (h_clarinets : clarinets_total = 30)
                   (h_trumpets : trumpets_total = 60)
                   (h_pianists : pianists_total = 20)
                   (h_flutes_pct : flutes_pct = 0.8)
                   (h_clarinets_pct : clarinets_pct = 0.5)
                   (h_trumpets_pct : trumpets_pct = 1/3)
                   (h_pianists_pct : pianists_pct = 1/10) :
  flutes_total * flutes_pct + clarinets_total * clarinets_pct + 
  trumpets_total * trumpets_pct + pianists_total * pianists_pct = 53 := by
  sorry

end NUMINAMATH_GPT_band_total_l1997_199770


namespace NUMINAMATH_GPT_length_of_lunch_break_is_48_minutes_l1997_199728

noncomputable def paula_and_assistants_lunch_break : ℝ := sorry

theorem length_of_lunch_break_is_48_minutes
  (p h L : ℝ)
  (h_monday : (9 - L) * (p + h) = 0.6)
  (h_tuesday : (7 - L) * h = 0.3)
  (h_wednesday : (10 - L) * p = 0.1) :
  L = 0.8 :=
sorry

end NUMINAMATH_GPT_length_of_lunch_break_is_48_minutes_l1997_199728


namespace NUMINAMATH_GPT_football_team_total_players_l1997_199746

theorem football_team_total_players (P : ℕ) (throwers : ℕ) (left_handed : ℕ) (right_handed : ℕ) :
  throwers = 49 →
  right_handed = 63 →
  left_handed = (1/3) * (P - 49) →
  (P - 49) - left_handed = (2/3) * (P - 49) →
  70 = P :=
by
  intros h_throwers h_right_handed h_left_handed h_remaining
  sorry

end NUMINAMATH_GPT_football_team_total_players_l1997_199746


namespace NUMINAMATH_GPT_minimum_value_of_f_l1997_199726

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1997_199726


namespace NUMINAMATH_GPT_brown_eyed_brunettes_count_l1997_199773

/--
There are 50 girls in a group. Each girl is either blonde or brunette and either blue-eyed or brown-eyed.
14 girls are blue-eyed blondes. 31 girls are brunettes. 18 girls are brown-eyed.
Prove that the number of brown-eyed brunettes is equal to 13.
-/
theorem brown_eyed_brunettes_count
  (total_girls : ℕ)
  (blue_eyed_blondes : ℕ)
  (total_brunettes : ℕ)
  (total_brown_eyed : ℕ)
  (total_girls_eq : total_girls = 50)
  (blue_eyed_blondes_eq : blue_eyed_blondes = 14)
  (total_brunettes_eq : total_brunettes = 31)
  (total_brown_eyed_eq : total_brown_eyed = 18) :
  ∃ (brown_eyed_brunettes : ℕ), brown_eyed_brunettes = 13 :=
by sorry

end NUMINAMATH_GPT_brown_eyed_brunettes_count_l1997_199773


namespace NUMINAMATH_GPT_min_value_fraction_l1997_199737

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) = 2) : 
  ∃ x : ℝ, x = (8 * a + b) / (a * b) ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1997_199737


namespace NUMINAMATH_GPT_amount_a_receives_l1997_199709

theorem amount_a_receives (a b c : ℕ) (h1 : a + b + c = 50000) (h2 : a = b + 4000) (h3 : b = c + 5000) :
  (21000 / 50000) * 36000 = 15120 :=
by
  sorry

end NUMINAMATH_GPT_amount_a_receives_l1997_199709


namespace NUMINAMATH_GPT_find_subtracted_value_l1997_199704

theorem find_subtracted_value (x y : ℕ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1997_199704


namespace NUMINAMATH_GPT_find_x_l1997_199776

theorem find_x (x : ℝ) (A1 A2 : ℝ) (P1 P2 : ℝ)
    (hA1 : A1 = x^2 + 4*x + 4)
    (hA2 : A2 = 4*x^2 - 12*x + 9)
    (hP : P1 + P2 = 32)
    (hP1 : P1 = 4 * (x + 2))
    (hP2 : P2 = 4 * (2*x - 3)) :
    x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1997_199776


namespace NUMINAMATH_GPT_hoseok_more_than_minyoung_l1997_199763

-- Define the initial amounts and additional earnings
def initial_amount : ℕ := 1500000
def additional_min : ℕ := 320000
def additional_hos : ℕ := 490000

-- Define the new amounts
def new_amount_min : ℕ := initial_amount + additional_min
def new_amount_hos : ℕ := initial_amount + additional_hos

-- Define the proof problem: Hoseok's new amount - Minyoung's new amount = 170000
theorem hoseok_more_than_minyoung : (new_amount_hos - new_amount_min) = 170000 :=
by
  -- The proof is skipped.
  sorry

end NUMINAMATH_GPT_hoseok_more_than_minyoung_l1997_199763


namespace NUMINAMATH_GPT_digit_place_value_ratio_l1997_199774

theorem digit_place_value_ratio : 
  let num := 43597.2468
  let digit5_place_value := 10    -- tens place
  let digit2_place_value := 0.1   -- tenths place
  digit5_place_value / digit2_place_value = 100 := 
by 
  sorry

end NUMINAMATH_GPT_digit_place_value_ratio_l1997_199774


namespace NUMINAMATH_GPT_shift_right_graph_l1997_199768

theorem shift_right_graph (x : ℝ) :
  (3 : ℝ)^(x+1) = (3 : ℝ)^((x+1) - 1) :=
by 
  -- Here we prove that shifting the graph of y = 3^(x+1) to right by 1 unit 
  -- gives the graph of y = 3^x
  sorry

end NUMINAMATH_GPT_shift_right_graph_l1997_199768


namespace NUMINAMATH_GPT_differentiable_implies_continuous_l1997_199796

-- Theorem: If a function f is differentiable at x0, then it is continuous at x0.
theorem differentiable_implies_continuous {f : ℝ → ℝ} {x₀ : ℝ} (h : DifferentiableAt ℝ f x₀) : 
  ContinuousAt f x₀ :=
sorry

end NUMINAMATH_GPT_differentiable_implies_continuous_l1997_199796


namespace NUMINAMATH_GPT_circumference_circle_l1997_199736

theorem circumference_circle {d r : ℝ} (h1 : ∀ (d r : ℝ), d = 2 * r) : 
  ∃ C : ℝ, C = π * d ∨ C = 2 * π * r :=
by {
  sorry
}

end NUMINAMATH_GPT_circumference_circle_l1997_199736


namespace NUMINAMATH_GPT_total_capsules_in_july_l1997_199781

theorem total_capsules_in_july : 
  let mondays := 4
  let tuesdays := 5
  let wednesdays := 5
  let thursdays := 4
  let fridays := 4
  let saturdays := 4
  let sundays := 5

  let capsules_monday := mondays * 2
  let capsules_tuesday := tuesdays * 3
  let capsules_wednesday := wednesdays * 2
  let capsules_thursday := thursdays * 3
  let capsules_friday := fridays * 2
  let capsules_saturday := saturdays * 4
  let capsules_sunday := sundays * 4

  let total_capsules := capsules_monday + capsules_tuesday + capsules_wednesday + capsules_thursday + capsules_friday + capsules_saturday + capsules_sunday

  let missed_capsules_tuesday := 3
  let missed_capsules_sunday := 4

  let total_missed_capsules := missed_capsules_tuesday + missed_capsules_sunday

  let total_consumed_capsules := total_capsules - total_missed_capsules
  total_consumed_capsules = 82 := 
by
  -- Details omitted, proof goes here
  sorry

end NUMINAMATH_GPT_total_capsules_in_july_l1997_199781


namespace NUMINAMATH_GPT_number_of_multiples_of_4_l1997_199701

theorem number_of_multiples_of_4 (a b : ℤ) (h1 : 100 < a) (h2 : b < 500) (h3 : a % 4 = 0) (h4 : b % 4 = 0) : 
  ∃ n : ℤ, n = 99 :=
by
  sorry

end NUMINAMATH_GPT_number_of_multiples_of_4_l1997_199701


namespace NUMINAMATH_GPT_mr_lee_gain_l1997_199756

noncomputable def cost_price_1 (revenue : ℝ) (profit_percentage : ℝ) : ℝ :=
  revenue / (1 + profit_percentage)

noncomputable def cost_price_2 (revenue : ℝ) (loss_percentage : ℝ) : ℝ :=
  revenue / (1 - loss_percentage)

theorem mr_lee_gain
    (revenue : ℝ)
    (profit_percentage : ℝ)
    (loss_percentage : ℝ)
    (revenue_1 : ℝ := 1.44)
    (revenue_2 : ℝ := 1.44)
    (profit_percent : ℝ := 0.20)
    (loss_percent : ℝ := 0.10):
  let cost_1 := cost_price_1 revenue_1 profit_percent
  let cost_2 := cost_price_2 revenue_2 loss_percent
  let total_cost := cost_1 + cost_2
  let total_revenue := revenue_1 + revenue_2
  total_revenue - total_cost = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_mr_lee_gain_l1997_199756


namespace NUMINAMATH_GPT_james_calories_per_minute_l1997_199712

variable (classes_per_week : ℕ) (hours_per_class : ℝ) (total_calories_per_week : ℕ)

theorem james_calories_per_minute
  (h1 : classes_per_week = 3)
  (h2 : hours_per_class = 1.5)
  (h3 : total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * (hours_per_class * 60)) = 7 := 
by
  sorry

end NUMINAMATH_GPT_james_calories_per_minute_l1997_199712


namespace NUMINAMATH_GPT_hillary_activities_l1997_199785

-- Define the conditions
def swims_every : ℕ := 6
def runs_every : ℕ := 4
def cycles_every : ℕ := 16

-- Define the theorem to prove
theorem hillary_activities : Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = 48 :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_hillary_activities_l1997_199785


namespace NUMINAMATH_GPT_number_of_seeds_in_bucket_B_l1997_199716

theorem number_of_seeds_in_bucket_B :
  ∃ (x : ℕ), 
    ∃ (y : ℕ), 
    ∃ (z : ℕ), 
      y = x + 10 ∧ 
      z = 30 ∧ 
      x + y + z = 100 ∧
      x = 30 :=
by {
  -- the proof is omitted.
  sorry
}

end NUMINAMATH_GPT_number_of_seeds_in_bucket_B_l1997_199716


namespace NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l1997_199789

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) : 
  z - x = 9 :=
sorry

end NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l1997_199789


namespace NUMINAMATH_GPT_log_product_max_l1997_199705

open Real

theorem log_product_max (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : log x + log y = 4) : log x * log y ≤ 4 := 
by
  sorry

end NUMINAMATH_GPT_log_product_max_l1997_199705


namespace NUMINAMATH_GPT_max_abs_ax_plus_b_l1997_199766

theorem max_abs_ax_plus_b (a b c : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_ax_plus_b_l1997_199766


namespace NUMINAMATH_GPT_smallest_positive_a_l1997_199747

/-- Define a function f satisfying the given conditions. -/
noncomputable def f : ℝ → ℝ :=
  sorry -- we'll define it later according to the problem

axiom condition1 : ∀ x > 0, f (2 * x) = 2 * f x

axiom condition2 : ∀ x, 1 < x ∧ x < 2 → f x = 2 - x

theorem smallest_positive_a :
  (∃ a > 0, f a = f 2020) ∧ ∀ b > 0, (f b = f 2020 → b ≥ 36) :=
  sorry

end NUMINAMATH_GPT_smallest_positive_a_l1997_199747


namespace NUMINAMATH_GPT_largest_integral_x_l1997_199742

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_largest_integral_x_l1997_199742


namespace NUMINAMATH_GPT_number_is_seven_point_five_l1997_199762

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_number_is_seven_point_five_l1997_199762


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1997_199733

theorem sum_of_consecutive_integers (x y : ℕ) (h1 : y = x + 1) (h2 : x * y = 812) : x + y = 57 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1997_199733


namespace NUMINAMATH_GPT_both_pipes_opened_together_for_2_minutes_l1997_199735

noncomputable def fill_time (t : ℝ) : Prop :=
  let rate_p := 1 / 12
  let rate_q := 1 / 15
  let combined_rate := rate_p + rate_q
  let work_done_by_p_q := combined_rate * t
  let work_done_by_q := rate_q * 10.5
  work_done_by_p_q + work_done_by_q = 1

theorem both_pipes_opened_together_for_2_minutes : ∃ t : ℝ, fill_time t ∧ t = 2 :=
by
  use 2
  unfold fill_time
  sorry

end NUMINAMATH_GPT_both_pipes_opened_together_for_2_minutes_l1997_199735


namespace NUMINAMATH_GPT_abhay_speed_l1997_199760

theorem abhay_speed
    (A S : ℝ)
    (h1 : 30 / A = 30 / S + 2)
    (h2 : 30 / (2 * A) = 30 / S - 1) :
    A = 5 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_abhay_speed_l1997_199760


namespace NUMINAMATH_GPT_successive_discounts_final_price_l1997_199711

noncomputable def initial_price : ℝ := 10000
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.10
noncomputable def discount3 : ℝ := 0.05

theorem successive_discounts_final_price :
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let final_selling_price := price_after_second_discount * (1 - discount3)
  final_selling_price = 6840 := by
  sorry

end NUMINAMATH_GPT_successive_discounts_final_price_l1997_199711


namespace NUMINAMATH_GPT_small_cone_altitude_l1997_199752

noncomputable def frustum_height : ℝ := 18
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

theorem small_cone_altitude (h_frustum : frustum_height = 18) 
    (A_lower : lower_base_area = 400 * Real.pi) 
    (A_upper : upper_base_area = 100 * Real.pi) : 
    ∃ (h_small_cone : ℝ), h_small_cone = 18 := 
by
  sorry

end NUMINAMATH_GPT_small_cone_altitude_l1997_199752


namespace NUMINAMATH_GPT_gloves_needed_l1997_199731

theorem gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) (total_gloves : ℕ)
  (h1 : participants = 82)
  (h2 : gloves_per_participant = 2)
  (h3 : total_gloves = participants * gloves_per_participant) :
  total_gloves = 164 :=
by
  sorry

end NUMINAMATH_GPT_gloves_needed_l1997_199731


namespace NUMINAMATH_GPT_calculate_jessie_points_l1997_199745

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calculate_jessie_points_l1997_199745


namespace NUMINAMATH_GPT_arithmetic_mean_end_number_l1997_199713

theorem arithmetic_mean_end_number (n : ℤ) :
  (100 + n) / 2 = 150 + 100 → n = 400 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_end_number_l1997_199713


namespace NUMINAMATH_GPT_euler_family_mean_age_l1997_199757

theorem euler_family_mean_age : 
  let girls_ages := [5, 5, 10, 15]
  let boys_ages := [8, 12, 16]
  let children_ages := girls_ages ++ boys_ages
  let total_sum := List.sum children_ages
  let number_of_children := List.length children_ages
  (total_sum : ℚ) / number_of_children = 10.14 := 
by
  sorry

end NUMINAMATH_GPT_euler_family_mean_age_l1997_199757


namespace NUMINAMATH_GPT_min_value_of_expression_l1997_199771

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1997_199771


namespace NUMINAMATH_GPT_line_parabola_intersection_l1997_199754

theorem line_parabola_intersection (k : ℝ) : 
    (∀ l p: ℝ → ℝ, l = (fun x => k * x + 1) ∧ p = (fun x => 4 * x ^ 2) → 
        (∃ x, l x = p x) ∧ (∀ x1 x2, l x1 = p x1 ∧ l x2 = p x2 → x1 = x2) 
    ↔ k = 0 ∨ k = 1) :=
sorry

end NUMINAMATH_GPT_line_parabola_intersection_l1997_199754


namespace NUMINAMATH_GPT_sequence_formula_l1997_199790

open Nat

def a : ℕ → ℤ
| 0     => 0  -- Defining a(0) though not used
| 1     => 1
| (n+2) => 3 * a (n+1) + 2^(n+2)

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 5 * 3^(n-1) - 2^(n+1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1997_199790


namespace NUMINAMATH_GPT_find_k_l1997_199777

theorem find_k 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (hSn : ∀ n, S n = -2 + 2 * (1 / 3) ^ n) 
  (h_geom : ∀ n, a (n + 1) = a n * a 2 / a 1) :
  k = -2 :=
sorry

end NUMINAMATH_GPT_find_k_l1997_199777


namespace NUMINAMATH_GPT_num_students_yes_R_l1997_199720

noncomputable def num_students_total : ℕ := 800
noncomputable def num_students_yes_only_M : ℕ := 150
noncomputable def num_students_no_to_both : ℕ := 250

theorem num_students_yes_R : (num_students_total - num_students_no_to_both) - num_students_yes_only_M = 400 :=
by
  sorry

end NUMINAMATH_GPT_num_students_yes_R_l1997_199720


namespace NUMINAMATH_GPT_team_formation_l1997_199795

def nat1 : ℕ := 7  -- Number of natives who know mathematics and physics
def nat2 : ℕ := 6  -- Number of natives who know physics and chemistry
def nat3 : ℕ := 3  -- Number of natives who know chemistry and mathematics
def nat4 : ℕ := 4  -- Number of natives who know physics and biology

def totalWaysToFormTeam (n1 n2 n3 n4 : ℕ) : ℕ := (n1 + n2 + n3 + n4).choose 3
def waysFromSameGroup (n : ℕ) : ℕ := n.choose 3

def waysFromAllGroups (n1 n2 n3 n4 : ℕ) : ℕ := (waysFromSameGroup n1) + (waysFromSameGroup n2) + (waysFromSameGroup n3) + (waysFromSameGroup n4)

theorem team_formation : totalWaysToFormTeam nat1 nat2 nat3 nat4 - waysFromAllGroups nat1 nat2 nat3 nat4 = 1080 := 
by
    sorry

end NUMINAMATH_GPT_team_formation_l1997_199795


namespace NUMINAMATH_GPT_sum_of_real_numbers_l1997_199741

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end NUMINAMATH_GPT_sum_of_real_numbers_l1997_199741


namespace NUMINAMATH_GPT_tangency_condition_for_parabola_and_line_l1997_199725

theorem tangency_condition_for_parabola_and_line (k : ℚ) :
  (∀ x y : ℚ, (6 * x - 4 * y + k = 0) ↔ (y^2 = 16 * x)) ↔ (k = 32 / 3) :=
  sorry

end NUMINAMATH_GPT_tangency_condition_for_parabola_and_line_l1997_199725


namespace NUMINAMATH_GPT_valid_license_plates_l1997_199784

-- Define the number of vowels and the total alphabet letters.
def num_vowels : ℕ := 5
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates in Eldoria.
theorem valid_license_plates : num_vowels * num_letters * num_digits^3 = 130000 := by
  sorry

end NUMINAMATH_GPT_valid_license_plates_l1997_199784


namespace NUMINAMATH_GPT_smallest_positive_integer_square_begins_with_1989_l1997_199780

theorem smallest_positive_integer_square_begins_with_1989 :
  ∃ (A : ℕ), (1989 * 10^0 ≤ A^2 ∧ A^2 < 1990 * 10^0) 
  ∨ (1989 * 10^1 ≤ A^2 ∧ A^2 < 1990 * 10^1) 
  ∨ (1989 * 10^2 ≤ A^2 ∧ A^2 < 1990 * 10^2)
  ∧ A = 446 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_square_begins_with_1989_l1997_199780


namespace NUMINAMATH_GPT_smallest_n_exists_l1997_199750

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_exists_l1997_199750


namespace NUMINAMATH_GPT_Valleyball_Soccer_League_members_l1997_199772

theorem Valleyball_Soccer_League_members (cost_socks cost_tshirt total_expenditure cost_per_member: ℕ) (h1 : cost_socks = 6) (h2 : cost_tshirt = cost_socks + 8) (h3 : total_expenditure = 3740) (h4 : cost_per_member = cost_socks + 2 * cost_tshirt) : 
  total_expenditure = 3740 → cost_per_member = 34 → total_expenditure / cost_per_member = 110 :=
sorry

end NUMINAMATH_GPT_Valleyball_Soccer_League_members_l1997_199772


namespace NUMINAMATH_GPT_remainder_of_7_pow_205_mod_12_l1997_199703

theorem remainder_of_7_pow_205_mod_12 : (7^205) % 12 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_7_pow_205_mod_12_l1997_199703


namespace NUMINAMATH_GPT_probability_same_color_l1997_199739

theorem probability_same_color :
  let red_marble_prob := (5 / 21) * (4 / 20) * (3 / 19)
  let white_marble_prob := (6 / 21) * (5 / 20) * (4 / 19)
  let blue_marble_prob := (7 / 21) * (6 / 20) * (5 / 19)
  let green_marble_prob := (3 / 21) * (2 / 20) * (1 / 19)
  red_marble_prob + white_marble_prob + blue_marble_prob + green_marble_prob = 66 / 1330 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_l1997_199739


namespace NUMINAMATH_GPT_number_of_boys_l1997_199715

-- We define the conditions provided in the problem
def child_1_has_3_brothers : Prop := ∃ B G : ℕ, B - 1 = 3 ∧ G = 6
def child_2_has_4_brothers : Prop := ∃ B G : ℕ, B - 1 = 4 ∧ G = 5

theorem number_of_boys (B G : ℕ) (h1 : child_1_has_3_brothers) (h2 : child_2_has_4_brothers) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l1997_199715


namespace NUMINAMATH_GPT_cycle_final_selling_price_l1997_199779

-- Lean 4 statement capturing the problem definition and final selling price
theorem cycle_final_selling_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (loss_rate : ℝ) (exchange_discount_rate : ℝ) (final_price : ℝ) :
  original_price = 1400 →
  initial_discount_rate = 0.05 →
  loss_rate = 0.25 →
  exchange_discount_rate = 0.10 →
  final_price = 
    (original_price * (1 - initial_discount_rate) * (1 - loss_rate) * (1 - exchange_discount_rate)) →
  final_price = 897.75 :=
by
  sorry

end NUMINAMATH_GPT_cycle_final_selling_price_l1997_199779


namespace NUMINAMATH_GPT_relation_correct_l1997_199719

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 0 < x ∧ x < 1}
def CR (S : Set ℝ) := {x : ℝ | x ∈ (Set.univ : Set ℝ) \ S}

theorem relation_correct : M ∪ CR N = (Set.univ : Set ℝ) :=
by sorry

end NUMINAMATH_GPT_relation_correct_l1997_199719


namespace NUMINAMATH_GPT_max_consecutive_sum_l1997_199758

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_max_consecutive_sum_l1997_199758


namespace NUMINAMATH_GPT_guy_has_sixty_cents_l1997_199797

-- Definitions for the problem conditions
def lance_has (lance_cents : ℕ) : Prop := lance_cents = 70
def margaret_has (margaret_cents : ℕ) : Prop := margaret_cents = 75
def bill_has (bill_cents : ℕ) : Prop := bill_cents = 60
def total_has (total_cents : ℕ) : Prop := total_cents = 265

-- Problem Statement in Lean format
theorem guy_has_sixty_cents (lance_cents margaret_cents bill_cents total_cents guy_cents : ℕ) 
    (h_lance : lance_has lance_cents)
    (h_margaret : margaret_has margaret_cents)
    (h_bill : bill_has bill_cents)
    (h_total : total_has total_cents) :
    guy_cents = total_cents - (lance_cents + margaret_cents + bill_cents) → guy_cents = 60 :=
by
  intros h
  simp [lance_has, margaret_has, bill_has, total_has] at *
  rw [h_lance, h_margaret, h_bill, h_total] at h
  exact h

end NUMINAMATH_GPT_guy_has_sixty_cents_l1997_199797


namespace NUMINAMATH_GPT_least_number_of_stamps_l1997_199793

theorem least_number_of_stamps (p q : ℕ) (h : 5 * p + 4 * q = 50) : p + q = 11 :=
sorry

end NUMINAMATH_GPT_least_number_of_stamps_l1997_199793


namespace NUMINAMATH_GPT_initial_books_in_bin_l1997_199798

variable (X : ℕ)

theorem initial_books_in_bin (h1 : X - 3 + 10 = 11) : X = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_books_in_bin_l1997_199798


namespace NUMINAMATH_GPT_muffins_in_morning_l1997_199744

variable (M : ℕ)

-- Conditions
def goal : ℕ := 20
def afternoon_sales : ℕ := 4
def additional_needed : ℕ := 4
def morning_sales (M : ℕ) : ℕ := M

-- Proof statement (no need to prove here, just state it)
theorem muffins_in_morning :
  morning_sales M + afternoon_sales + additional_needed = goal → M = 12 :=
sorry

end NUMINAMATH_GPT_muffins_in_morning_l1997_199744


namespace NUMINAMATH_GPT_melanie_dimes_final_l1997_199753

-- Define a type representing the initial state of Melanie's dimes
variable {initial_dimes : ℕ} (h_initial : initial_dimes = 7)

-- Define a function representing the result after attempting to give away dimes
def remaining_dimes_after_giving (initial_dimes : ℕ) (given_dimes : ℕ) : ℕ :=
  if given_dimes <= initial_dimes then initial_dimes - given_dimes else initial_dimes

-- State the problem
theorem melanie_dimes_final (h_initial : initial_dimes = 7) (given_dimes_dad : ℕ) (h_given_dad : given_dimes_dad = 8) (received_dimes_mom : ℕ) (h_received_mom : received_dimes_mom = 4) :
  remaining_dimes_after_giving initial_dimes given_dimes_dad + received_dimes_mom = 11 :=
by
  sorry

end NUMINAMATH_GPT_melanie_dimes_final_l1997_199753


namespace NUMINAMATH_GPT_range_of_m_l1997_199708

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log x + m / x

theorem range_of_m (m : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (f m b - f m a) / (b - a) < 1) →
  m ≥ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1997_199708


namespace NUMINAMATH_GPT_arithmetic_sequence_min_sum_l1997_199730

theorem arithmetic_sequence_min_sum (x : ℝ) (d : ℝ) (h₁ : d > 0) :
  (∃ n : ℕ, n > 0 ∧ (n^2 - 4 * n < 0) ∧ (n = 6 ∨ n = 7)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_min_sum_l1997_199730


namespace NUMINAMATH_GPT_total_branches_in_pine_tree_l1997_199724

-- Definitions based on the conditions
def middle_branch : ℕ := 0 -- arbitrary assignment to represent the middle branch

def jumps_up_5 (b : ℕ) : ℕ := b + 5
def jumps_down_7 (b : ℕ) : ℕ := b - 7
def jumps_up_4 (b : ℕ) : ℕ := b + 4
def jumps_up_9 (b : ℕ) : ℕ := b + 9

-- The statement to be proven
theorem total_branches_in_pine_tree : 
  (jumps_up_9 (jumps_up_4 (jumps_down_7 (jumps_up_5 middle_branch))) = 11) →
  ∃ n, n = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_branches_in_pine_tree_l1997_199724


namespace NUMINAMATH_GPT_range_of_a_l1997_199721

theorem range_of_a (f : ℝ → ℝ) (h_mono_dec : ∀ x1 x2, -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 → x1 < x2 → f x1 > f x2) 
  (h_cond : ∀ a, -2 ≤ a + 1 ∧ a + 1 ≤ 2 ∧ -2 ≤ 2 * a ∧ 2 * a ≤ 2 → f (a + 1) < f (2 * a)) :
  { a : ℝ | -1 ≤ a ∧ a < 1 } :=
sorry

end NUMINAMATH_GPT_range_of_a_l1997_199721


namespace NUMINAMATH_GPT_jeremys_school_distance_l1997_199718

def distance_to_school (rush_hour_time : ℚ) (no_traffic_time : ℚ) (speed_increase : ℚ) (distance : ℚ) : Prop :=
  ∃ v : ℚ, distance = v * rush_hour_time ∧ distance = (v + speed_increase) * no_traffic_time

theorem jeremys_school_distance :
  distance_to_school (3/10 : ℚ) (1/5 : ℚ) 20 12 :=
sorry

end NUMINAMATH_GPT_jeremys_school_distance_l1997_199718


namespace NUMINAMATH_GPT_geom_sequence_arith_ratio_l1997_199748

variable (a : ℕ → ℝ) (q : ℝ)
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_arith : 3 * a 0 + 2 * a 1 = 2 * (1/2) * a 2)

theorem geom_sequence_arith_ratio (ha : 3 * a 0 + 2 * a 1 = a 2) :
    (a 8 + a 9) / (a 6 + a 7) = 9 := sorry

end NUMINAMATH_GPT_geom_sequence_arith_ratio_l1997_199748


namespace NUMINAMATH_GPT_find_f_minus_2_l1997_199764

namespace MathProof

def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 5

theorem find_f_minus_2 (a b c : ℝ) (h : f a b c 2 = 3) : f a b c (-2) = -13 := 
by
  sorry

end MathProof

end NUMINAMATH_GPT_find_f_minus_2_l1997_199764


namespace NUMINAMATH_GPT_sequence_gcd_is_index_l1997_199799

theorem sequence_gcd_is_index (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
by
  sorry

end NUMINAMATH_GPT_sequence_gcd_is_index_l1997_199799


namespace NUMINAMATH_GPT_inequality_solution_l1997_199788

theorem inequality_solution (x : ℝ) : 
  -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 3 ≤ x ∧ x < 4 → 
  (x + 6 ≥ 0) ∧ (x + 1 > 0) ∧ (5 - x > 0) ∧ (x ≠ 0) ∧ (x ≠ 1) ∧ (x ≠ 4) ∧
  ( (x - 3) / ((x - 1) * (4 - x)) ≥ 0 ) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1997_199788
