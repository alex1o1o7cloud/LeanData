import Mathlib

namespace fraction_sent_afternoon_l53_53304

theorem fraction_sent_afternoon :
  ∀ (total_fliers morning_fraction fliers_left_next_day : ℕ),
  total_fliers = 3000 →
  morning_fraction = 1/5 →
  fliers_left_next_day = 1800 →
  ((total_fliers - total_fliers * morning_fraction) - fliers_left_next_day) / (total_fliers - total_fliers * morning_fraction) = 1/4 :=
by
  intros total_fliers morning_fraction fliers_left_next_day h1 h2 h3
  sorry

end fraction_sent_afternoon_l53_53304


namespace quadrilateral_angle_B_l53_53844

/-- In quadrilateral ABCD,
given that angle A + angle C = 150 degrees,
prove that angle B = 105 degrees. -/
theorem quadrilateral_angle_B (A C : ℝ) (B : ℝ) (h1 : A + C = 150) (h2 : A + B = 180) : B = 105 :=
by
  sorry

end quadrilateral_angle_B_l53_53844


namespace prob_interval_l53_53417

variable (ξ : ℝ)
variable (a : ℝ)
variable (P : ℝ → ℝ)
variable (ξ_pos : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → ξ = k / 5)

-- The condition P(ξ = k/5) = ak for k=1,2,3,4,5
axiom prob_dist : ∀ k : ℕ, (1 ≤ k ∧ k ≤ 5) → P (k/5) = a * k

-- The condition that the sum of all probabilities must equal 1
axiom prob_sum : ∑ k in {1, 2, 3, 4, 5}, P (k / 5) = 1

-- The goal is to prove that P(1/10 < ξ < 1/2) = 2/5
theorem prob_interval : P (1/10 < ξ ∧ ξ < 1/2) = 2/5 :=
by sorry

end prob_interval_l53_53417


namespace complex_quadrant_l53_53594

theorem complex_quadrant (z : ℂ) (h : z = (1 + (1 / (1 + complex.I)))) : 
  (z.re > 0 ∧ z.im < 0) ↔ true := 
by {
  have z_simplified : z = (3 / 2) - (1 / 2) * complex.I,
  -- Simplification steps go here
  sorry
  -- Use the simplified form to show that the real part is positive and the imaginary part is negative
  -- Hence, proving that the complex number is in the fourth quadrant
}

end complex_quadrant_l53_53594


namespace parents_age_when_mark_was_born_l53_53170

noncomputable def age_mark := 18
noncomputable def age_difference := 10
noncomputable def parent_multiple := 5

theorem parents_age_when_mark_was_born :
  let age_john := age_mark - age_difference in
  let parents_current_age := age_john * parent_multiple in
  parents_current_age - age_mark = 22 :=
by
  sorry

end parents_age_when_mark_was_born_l53_53170


namespace series_sum_l53_53597

theorem series_sum :
  ∑ n in Finset.range 20, (1 / ((n+1) * (n+3))) = (325 / 462) :=
sorry

end series_sum_l53_53597


namespace proposition_2_proposition_3_l53_53722

variable {a b c : ℝ}

theorem proposition_2 (ha_gt_0 : abs a > abs b) : a^2 > b^2 :=
by {
  sorry
}

theorem proposition_3 (ha_gt_b: a > b) (hc : c ∈ ℝ) : a + c > b + c ↔ a > b :=
by {
  sorry
}

end proposition_2_proposition_3_l53_53722


namespace geometric_mean_S3_S99_l53_53036

-- Definition of the series Sn
def S (n : ℕ) (hn : n > 0) : ℝ :=
  ∑ k in Finset.range n, (1 / (Real.sqrt k + Real.sqrt (k + 1)))

-- Simplifying the general term
lemma term_simplification (m : ℕ) (hm : m > 0) : 
  (1 / (Real.sqrt m + Real.sqrt (m+1))) = Real.sqrt (m+1) - Real.sqrt m := 
sorry

-- Prove the form of S_n
lemma Sn_form (n : ℕ) (hn : n > 0) :
  S n hn = Real.sqrt (n+1) - 1 := 
sorry

-- Calculate S_3 and S_99
lemma S_3_value : S 3 (by norm_num) = 1 := 
sorry

lemma S_99_value : S 99 (by norm_num) = 9 := 
sorry

-- Prove the geometric mean of S3 and S99
theorem geometric_mean_S3_S99 : 
  ∃ (x : ℝ), x = 3 ∨ x = -3 ∧ (S 3 (by norm_num) * S 99 (by norm_num) = x^2) :=
sorry

end geometric_mean_S3_S99_l53_53036


namespace simplify_expression_l53_53353

theorem simplify_expression (x y : ℝ) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / (1 / 2 * x * y)^2 = 8 * x - 12 * y := by
  sorry

end simplify_expression_l53_53353


namespace point_in_first_quadrant_l53_53886

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53886


namespace average_marks_is_75_l53_53131

-- Define the scores for the four tests based on the given conditions.
def first_test : ℕ := 80
def second_test : ℕ := first_test + 10
def third_test : ℕ := 65
def fourth_test : ℕ := third_test

-- Define the total marks scored in the four tests.
def total_marks : ℕ := first_test + second_test + third_test + fourth_test

-- Number of tests.
def num_tests : ℕ := 4

-- Define the average marks scored in the four tests.
def average_marks : ℕ := total_marks / num_tests

-- Prove that the average marks scored in the four tests is 75.
theorem average_marks_is_75 : average_marks = 75 :=
by
  sorry

end average_marks_is_75_l53_53131


namespace relationship_among_a_b_c_l53_53762

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry -- f'(x) denotes the derivative of f

axiom symmetry_condition : ∀ x : ℝ, f (x - 1) = f (2 - (x - 1))
axiom derivative_condition : ∀ x : ℝ, x < 0 → f(x) + x * f'(x) < 0

noncomputable def g (x : ℝ) : ℝ := x * f x

def a : ℝ := g (1 / 2)
def b : ℝ := g (Real.log 2)
def c : ℝ := g (2)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  sorry

end relationship_among_a_b_c_l53_53762


namespace pool_filled_in_48_minutes_with_both_valves_open_l53_53224

def rate_first_valve_fills_pool_in_2_hours (V1 : ℚ) : Prop :=
  V1 * 120 = 12000

def rate_second_valve_50_more_than_first (V1 V2 : ℚ) : Prop :=
  V2 = V1 + 50

def pool_capacity : ℚ := 12000

def combined_rate (V1 V2 combinedRate : ℚ) : Prop :=
  combinedRate = V1 + V2

def time_to_fill_pool_with_both_valves_open (combinedRate time : ℚ) : Prop :=
  time = pool_capacity / combinedRate

theorem pool_filled_in_48_minutes_with_both_valves_open
  (V1 V2 combinedRate time : ℚ) :
  rate_first_valve_fills_pool_in_2_hours V1 →
  rate_second_valve_50_more_than_first V1 V2 →
  combined_rate V1 V2 combinedRate →
  time_to_fill_pool_with_both_valves_open combinedRate time →
  time = 48 :=
by
  intros
  sorry

end pool_filled_in_48_minutes_with_both_valves_open_l53_53224


namespace range_of_function_l53_53463

open Real

def range_function (x : ℝ) : ℝ :=
  sqrt 2 * sin (x + π / 4)

theorem range_of_function:
  ∀ (x : ℝ), 0 < x ∧ x ≤ π / 3 → 1 < range_function x ∧ range_function x ≤ sqrt 2 :=
by
  intros x hx
  sorry

end range_of_function_l53_53463


namespace large_pizza_cost_l53_53805

variable (side_small_pizza: ℝ) (cost_small_pizza: ℝ) (side_large_pizza: ℝ) (total_money: ℝ) (extra_area: ℝ)

axiom (h1: side_small_pizza = 12)
axiom (h2: cost_small_pizza = 10)
axiom (h3: side_large_pizza = 18)
axiom (h4: total_money = 60)
axiom (h5: extra_area = 36)

noncomputable def calculate_large_pizza_cost : ℝ :=
  let area_small : ℝ := side_small_pizza ^ 2
  let area_large : ℝ := side_large_pizza ^ 2
  let total_area_separate : ℝ := 2 * (total_money / 2 / cost_small_pizza) * area_small
  let total_area_pooled : ℝ := total_money / (total_money / area_large * (total_area_separate + extra_area))
  total_money / area_large * (total_area_separate + extra_area)

theorem large_pizza_cost : calculate_large_pizza_cost side_small_pizza cost_small_pizza side_large_pizza total_money extra_area = 21.6 := by
  sorry

end large_pizza_cost_l53_53805


namespace total_area_of_strips_l53_53391

def strip1_length := 12
def strip1_width := 1
def strip2_length := 8
def strip2_width := 2
def num_strips1 := 2
def num_strips2 := 2
def overlap_area_per_strip := 2
def num_overlaps := 4
def total_area_covered := 48

theorem total_area_of_strips : 
  num_strips1 * (strip1_length * strip1_width) + 
  num_strips2 * (strip2_length * strip2_width) - 
  num_overlaps * overlap_area_per_strip = total_area_covered := sorry

end total_area_of_strips_l53_53391


namespace cone_volume_l53_53423

def volume_of_cone (r l h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (r l h v : ℝ) (h_l : l = 2)
  (h_circumference : 2 * π * r = π * 2)
  (h_height : h = real.sqrt (l^2 - r^2))
  (h_radius : r = 1)
  (h_correct_height : h = real.sqrt 3)
  (h_correct_volume : v = volume_of_cone r l h) :
  v = (real.sqrt 3 * π) / 3 :=
by
  sorry

end cone_volume_l53_53423


namespace limit_S_l53_53540

noncomputable def z (n : ℕ) : ℂ := (1 - Complex.I) / 2 ^ n
noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range n, Complex.abs (z (k+1) - z k)

theorem limit_S : tendsto S atTop (𝓝 (1 + Real.sqrt 2 / 2)) :=
sorry

end limit_S_l53_53540


namespace sum_semsicircles_approaches_zero_l53_53583

open Real

theorem sum_semsicircles_approaches_zero (D : ℝ) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, n > 0 → 
    abs (∑ k in finset.range n, (π * (D / (2 * (n : ℝ)))^2 / 2)) < ε :=
by sorry

end sum_semsicircles_approaches_zero_l53_53583


namespace complex_point_quadrant_l53_53873

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53873


namespace complex_quadrant_check_l53_53916

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53916


namespace value_of_x_l53_53101

theorem value_of_x (x : ℝ) (h : 3 * x + 15 = (1/3) * (7 * x + 45)) : x = 0 :=
by
  sorry

end value_of_x_l53_53101


namespace correct_negative_numbers_correct_non_negative_integers_correct_rational_numbers_l53_53012

def expressions := [Rat.mk 5 6, Real.pi, -3, -(-(Rat.mk 3 4)), 4^2, 0, 0.6]

def negative_numbers := [expr | expr ∈ expressions ∧ expr < 0]
def non_negative_integers := [expr | expr ∈ expressions ∧ expr ≥ 0 ∧ ∃ (n : ℕ), expr = n]
def rational_numbers := [expr | expr ∈ expressions ∧ ∃ (n d : ℤ), d ≠ 0 ∧ expr = n / d]

noncomputable def set_of_negative_numbers : List ℚ := [-3, -Rat.mk 3 4]
noncomputable def set_of_non_negative_integers : List ℚ := [16, 0]
noncomputable def set_of_rational_numbers : List ℚ := [Rat.mk 5 6, -3, -Rat.mk 3 4, 16, 0, 0.6]

theorem correct_negative_numbers :
  negative_numbers = set_of_negative_numbers := by
  sorry

theorem correct_non_negative_integers :
  non_negative_integers = set_of_non_negative_integers := by
  sorry

theorem correct_rational_numbers :
  rational_numbers = set_of_rational_numbers := by
  sorry

end correct_negative_numbers_correct_non_negative_integers_correct_rational_numbers_l53_53012


namespace value_of_c_l53_53063

noncomputable def given : Type :=
  {c : ℝ // ∃ (f : ℝ → ℝ) (a : ℝ), f = (λ x, x^3 - 3*x + c) ∧
                                       f a = 0 ∧ 
                                       (∀ b ∈ set_of (λ x, f x = 0), b ≠ a) ∧
                                       (∀ x:ℝ, ∂(f x)/∂x = 3*x^2 - 3 ∧
                                               (∀ x, -1 < x ∧ x < 1 → ∂(f x)/∂x < 0) ∧
                                               (∀ x, x < -1 ∨ x > 1 → ∂(f x)/∂x > 0) ) ∧ (a = -1) }

theorem value_of_c : given = -2 :=
sorry

end value_of_c_l53_53063


namespace area_of_PQRS_l53_53845

-- Define the points and the rectangle PQRS
variables {P Q R S M N : ℝ}
variables (h1 : angle P Q R = π / 2)  -- PQRS is a rectangle
variables (h2 : angle P Q M = π / 6) (h3 : angle P S N = π / 6)  -- PM and PN trisect angle P
variables (h_SN : N.y = S.y + 8)  -- SN = 8
variables (h_QM : M.x = Q.x - 3)  -- QM = 3
variables (x y : ℝ)  -- to represent PQ and PS

-- The main theorem to prove
theorem area_of_PQRS : (x * y) = 864 :=
sorry

end area_of_PQRS_l53_53845


namespace radius_of_circle_passing_through_ABC_incircle_center_l53_53966

variable {a : ℝ} {α : ℝ} (hα : 0 < α) (hα_lt_π : α < real.pi)

theorem radius_of_circle_passing_through_ABC_incircle_center
  {AB : ℝ} {α : ℝ} (hAB : AB = a) (hC : ∠ ABC = α) :
  ∃ R : ℝ, R = a / (2 * real.cos (α / 2)) :=
sorry

end radius_of_circle_passing_through_ABC_incircle_center_l53_53966


namespace find_a1_find_a1_to_a7_sum_l53_53086

-- Definitions
def polynomial := (x + 1)^7 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7

-- Proof problems
theorem find_a1 (polynomial : polynomial) : a_1 = 7 :=
sorry

theorem find_a1_to_a7_sum (polynomial : polynomial) : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 127 :=
sorry

end find_a1_find_a1_to_a7_sum_l53_53086


namespace sum_f_eq_768_l53_53361

def is_perfect_square (k : ℕ) : Prop :=
  ∃ n : ℕ, n * n = k

def f (k : ℕ) : ℕ :=
  if is_perfect_square k then 0
  else ⌊1 / (sqrt k - ⌊sqrt k⌋)⌋

def sum_f : ℕ :=
  (∑ k in Finset.range (240 + 1), f k)

theorem sum_f_eq_768 : sum_f = 768 :=
by
sorry

end sum_f_eq_768_l53_53361


namespace scaled_vector_dot_product_l53_53456

variables {u : EuclideanSpace ℝ (Fin 1)}

-- Condition
axiom norm_u : ∥u∥ = 5

-- Theorem to prove
theorem scaled_vector_dot_product :
  let half_u := (1/2 : ℝ) • u in
  (half_u) ⬝ (half_u) = 6.25 :=
by
  sorry

end scaled_vector_dot_product_l53_53456


namespace smallest_number_divisible_by_20_and_36_l53_53618

-- Define the conditions that x must be divisible by both 20 and 36
def divisible_by (x n : ℕ) : Prop := ∃ m : ℕ, x = n * m

-- Define the problem statement
theorem smallest_number_divisible_by_20_and_36 : 
  ∃ x : ℕ, divisible_by x 20 ∧ divisible_by x 36 ∧ 
  (∀ y : ℕ, (divisible_by y 20 ∧ divisible_by y 36) → y ≥ x) ∧ x = 180 := 
by
  sorry

end smallest_number_divisible_by_20_and_36_l53_53618


namespace black_area_after_six_transformations_l53_53692

noncomputable def remaining_fraction_after_transformations (initial_fraction : ℚ) (transforms : ℕ) (reduction_factor : ℚ) : ℚ :=
  reduction_factor ^ transforms * initial_fraction

theorem black_area_after_six_transformations :
  remaining_fraction_after_transformations 1 6 (2 / 3) = 64 / 729 := 
by
  sorry

end black_area_after_six_transformations_l53_53692


namespace mean_of_two_remaining_numbers_l53_53387

theorem mean_of_two_remaining_numbers :
  let s := [1212, 1702, 1834, 1956, 2048, 2219, 2300]
  let sum_s := s.sum
  let mean_five := 5 * 2000
  sum_s - mean_five = 3271 →
  3271 / 2 = 1635.5 :=
by
  intros
  let sum_s := 13271
  let mean_five := 10000
  have h1 : sum_s - mean_five = 3271 := by norm_num
  rw [←h1]
  exact eq.symm (by norm_num)

end mean_of_two_remaining_numbers_l53_53387


namespace algebra_expression_value_l53_53828

theorem algebra_expression_value
  (x y : ℝ)
  (h : x - 2 * y + 2 = 5) : 4 * y - 2 * x + 1 = -5 :=
by sorry

end algebra_expression_value_l53_53828


namespace squares_same_remainder_l53_53562

theorem squares_same_remainder (S : Finset ℤ) (hS : S.card = 51) :
  ∃ x y ∈ S, x ≠ y ∧ (x^2 ≡ y^2 [MOD 100]) :=
by
  sorry

end squares_same_remainder_l53_53562


namespace rectangular_prism_max_volume_l53_53122

theorem rectangular_prism_max_volume (a b c p : ℝ)
  (h1 : sqrt (a^2 + b^2 + c^2) = 1)
  (h2 : sqrt (b^2 + c^2) = sqrt 2)
  (h3 : p = b) :
  p = 7 * sqrt (1 + (2 * sqrt 3) / 3) :=
sorry

end rectangular_prism_max_volume_l53_53122


namespace point_in_first_quadrant_l53_53880

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53880


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53275

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53275


namespace center_of_circle_l53_53061

theorem center_of_circle (x y : ℝ) :
  x^2 + y^2 - 2 * x - 6 * y + 1 = 0 →
  (1, 3) = (1, 3) :=
by
  intros h
  sorry

end center_of_circle_l53_53061


namespace san_antonio_to_austin_buses_passed_l53_53700

def departure_schedule (departure_time_A_to_S departure_time_S_to_A travel_time : ℕ) : Prop :=
  ∀ t, (t < travel_time) →
       (∃ n, t = (departure_time_A_to_S + n * 60)) ∨
       (∃ m, t = (departure_time_S_to_A + m * 60)) →
       t < travel_time

theorem san_antonio_to_austin_buses_passed :
  let departure_time_A_to_S := 30  -- Austin to San Antonio buses leave every hour on the half-hour (e.g., 00:30, 1:30, ...)
  let departure_time_S_to_A := 0   -- San Antonio to Austin buses leave every hour on the hour (e.g., 00:00, 1:00, ...)
  let travel_time := 6 * 60        -- The trip takes 6 hours, or 360 minutes
  departure_schedule departure_time_A_to_S departure_time_S_to_A travel_time →
  ∃ count, count = 12 := 
by
  sorry

end san_antonio_to_austin_buses_passed_l53_53700


namespace total_personal_and_spam_emails_mornings_evenings_over_5_days_l53_53188

def emails_by_time_period : ℕ × ℕ → ℕ
| (2, 1) => 3
| (3, 1) => 4
| (2, 0) => 2
| (2, 0) => 2
| (3, 3) => 6
| (1, 1) => 2
| (1, 2) => 3
| (3, 3) => 6
| (3, 0) => 3
| (2, 2) => 4

def personal_and_spam_emails_morning : ℕ :=
  (emails_by_time_period (2, 1)) + (emails_by_time_period (2, 0)) + 
  (emails_by_time_period (3, 3)) + (emails_by_time_period (1, 2)) + 
  (emails_by_time_period (3, 0))

def personal_and_spam_emails_evening : ℕ :=
  (emails_by_time_period (3, 1)) + (emails_by_time_period (2, 0)) + 
  (emails_by_time_period (1, 1)) + (emails_by_time_period (3, 3)) + 
  (emails_by_time_period (2, 2))

theorem total_personal_and_spam_emails_mornings_evenings_over_5_days :
  personal_and_spam_emails_morning + personal_and_spam_emails_evening = 35 :=
by {
  -- the proof would go here
  sorry
}

end total_personal_and_spam_emails_mornings_evenings_over_5_days_l53_53188


namespace price_of_third_variety_l53_53640

-- Define the given conditions
def price1 : ℝ := 126
def price2 : ℝ := 135
def average_price : ℝ := 153
def ratio1 : ℝ := 1
def ratio2 : ℝ := 1
def ratio3 : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Define the equation based on the given conditions
def weighted_avg_price (P : ℝ) : Prop :=
  (ratio1 * price1 + ratio2 * price2 + ratio3 * P) / total_ratio = average_price

-- Statement of the proof
theorem price_of_third_variety :
  ∃ P : ℝ, weighted_avg_price P ∧ P = 175.5 :=
by {
  -- Proof omitted
  sorry
}

end price_of_third_variety_l53_53640


namespace boxes_left_l53_53179

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l53_53179


namespace digit_6_count_range_100_to_999_l53_53959

/-- The number of times the digit 6 is written in the integers from 100 through 999 inclusive is 280. -/
theorem digit_6_count_range_100_to_999 : 
  (∑ n in finset.Icc 100 999, (if digit 6 n then 1 else 0)) = 280 := 
sorry

end digit_6_count_range_100_to_999_l53_53959


namespace range_of_real_number_a_l53_53346

theorem range_of_real_number_a (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) → 
  (∀ x₁ x₂ ∈ set.Ioo (-1) 1, x₁ < x₂ → f x₁ > f x₂) →
  (∀ a ∈ set.Ioo (-1) 0, f (1 + a) + f (1 - a^2) < 0) →
  a ∈ set.Ioo (-1) 0 :=
sorry

end range_of_real_number_a_l53_53346


namespace log_sqrt10_1000sqrt10_l53_53009

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53009


namespace cube_volume_doubled_l53_53825

theorem cube_volume_doubled (a : ℝ) (h : a > 0) : 
  ((2 * a)^3 - a^3) / a^3 = 7 :=
by
  sorry

end cube_volume_doubled_l53_53825


namespace values_of_x_l53_53462

theorem values_of_x (x : ℝ) : (x+2)*(x-9) < 0 ↔ -2 < x ∧ x < 9 := 
by
  sorry

end values_of_x_l53_53462


namespace max_subset_sum_no_partition_l53_53159

theorem max_subset_sum_no_partition (S : Finset ℕ) (hS : S ⊆ (Finset.range 13) \ {0}) :
  (∃ k ≥ 2, (∃ (subs : Finset (Finset ℕ)), 
    (subs.card = k) ∧ (∀ t ∈ subs, t ⊆ S) ∧ (∀ t1 t2 ∈ subs, t1 ≠ t2 → Disjoint t1 t2) ∧ 
    (∃ sum, ∀ t ∈ subs, (Finset.sum t id) = sum)) → False) → 
  Finset.sum S id ≤ 77 :=
by
  sorry

end max_subset_sum_no_partition_l53_53159


namespace calculate_period_of_oscillation_l53_53253

noncomputable def period_of_oscillation 
  (n : ℕ) (L : ℝ) (g : ℝ) (m : ℝ) : ℝ :=
  2 * π * (sqrt (n * (L / g) * ((1 / (Real.sin (π / n))) - ((Real.sin (π / n)) / 3))))

theorem calculate_period_of_oscillation (n : ℕ) (L g m : ℝ) (hL : L = 1):
  period_of_oscillation n L g m = 
    2 * π * (sqrt (n * (L / g) * ((1 / (Real.sin (π / n))) - ((Real.sin (π / n)) / 3)))) :=
sorry

end calculate_period_of_oscillation_l53_53253


namespace delta_max_success_ratio_l53_53842

theorem delta_max_success_ratio (y w x z : ℤ) (h1 : 360 + 240 = 600)
  (h2 : 0 < x ∧ x < y ∧ z < w)
  (h3 : y + w = 600)
  (h4 : (x : ℚ) / y < (200 : ℚ) / 360)
  (h5 : (z : ℚ) / w < (160 : ℚ) / 240)
  (h6 : (360 : ℚ) / 600 = 3 / 5)
  (h7 : (x + z) < 166) :
  (x + z : ℚ) / 600 ≤ 166 / 600 := 
sorry

end delta_max_success_ratio_l53_53842


namespace ratio_volume_cone_to_sphere_l53_53057

variable (r : ℝ) (π : ℝ)

-- The given conditions
def height_of_cone_eq_diam_of_sphere : ℝ := 2 * r
def base_diam_of_cone_eq_diam_of_sphere : ℝ := 2 * r

-- Volumes
def volume_of_cone : ℝ := (1 / 3) * π * (r ^ 2) * (2 * r)
def volume_of_sphere : ℝ := (4 / 3) * π * (r ^ 3)

-- The main theorem
theorem ratio_volume_cone_to_sphere :
  volume_of_cone r π / volume_of_sphere r π = (1 / 2) :=
sorry

end ratio_volume_cone_to_sphere_l53_53057


namespace complex_quadrant_check_l53_53917

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53917


namespace geometric_not_arithmetic_l53_53993
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def sequence_values : ℝ × ℝ × ℝ :=
  let α := (sqrt 5 + 1) / 2
  (α - (⌊α⌋ : ℝ), ⌊α⌋, α)

theorem geometric_not_arithmetic :
  let ⟨x₁, x₂, x₃⟩ := sequence_values
  x₁ * x₃ = x₂^2 ∧ x₁ + x₃ ≠ 2 := 
by
  let α := (sqrt 5 + 1) / 2
  have h₁ : ℝ := α - (⌊α⌋ : ℝ)
  have h₂ : ℝ := ⌊α⌋
  have h₃ : ℝ := α
  have h_geometric : h₁ * h₃ = h₂^2 := sorry
  have h_not_arithmetic : h₁ + h₃ ≠ 2 := sorry
  exact And.intro h_geometric h_not_arithmetic

end geometric_not_arithmetic_l53_53993


namespace lyle_payment_l53_53669

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l53_53669


namespace ratio_AB_BC_l53_53984

variables {A B C D P Q : Point}
variables {x y : ℝ}

-- Definitions for the sides of the rectangle
def AB := x
def BC := y
def AC := (2 * AB)

-- Area conditions for the triangles
def triangle_areas (P inside ABCD) (AB CD adj) (BC DA diag) := 
  let h1 := (2 / x)
  let h2 := (8 / x)
  y = h1 + h2
  AC = 2 * AB

-- The main theorem we want to prove
theorem ratio_AB_BC (h1 h2 : ℝ) (h1h : h1 = (2 / x)) (h2h : h2 = (8 / x)) (yh : y = (10 / x))
  : y * y + x * x = (2 * x) * (2 * x) → (x / y) = (10 / 3) :=
by sorry

end ratio_AB_BC_l53_53984


namespace james_collected_fruits_from_trees_l53_53132

theorem james_collected_fruits_from_trees :
  ∀ (trees plants_per_tree seeds_per_plant percentage_planted planted_trees),
  plants_per_tree = 20 →
  seeds_per_plant = 1 →
  percentage_planted = 0.60 →
  planted_trees = 24 →
  trees = 2 :=
by
  intros trees plants_per_tree seeds_per_plant percentage_planted planted_trees
  assume h1 : plants_per_tree = 20
  assume h2 : seeds_per_plant = 1
  assume h3 : percentage_planted = 0.60
  assume h4 : planted_trees = 24
  sorry

end james_collected_fruits_from_trees_l53_53132


namespace minimum_AF_plus_2BF_l53_53764

variable {x y : ℝ}

def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def focus : ℝ × ℝ := (1 / 2, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem minimum_AF_plus_2BF :
  ∀ A B : ℝ × ℝ,
  A.2^2 = 2 * A.1 →
  B.2^2 = 2 * B.1 →
  ∃ l : ℝ × ℝ → Prop, 
    l focus →
    l A ∧ l B →
    (A ≠ B ∨ (A = B ∧ B ≠ A)) →
    (|distance A focus| + 2 * |distance B focus|) ≥ (3/2 + Real.sqrt 2) :=
by
  sorry

end minimum_AF_plus_2BF_l53_53764


namespace fractional_sum_identity_l53_53537

noncomputable def distinct_real_roots (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem fractional_sum_identity :
  ∀ (p q r A B C : ℝ),
  (x^3 - 22*x^2 + 80*x - 67 = (x - p) * (x - q) * (x - r)) →
  distinct_real_roots (λ x => x^3 - 22*x^2 + 80*x - 67) p q r →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (1 / (A) + 1 / (B) + 1 / (C) = 244) :=
by 
  intros p q r A B C h_poly h_distinct h_fractional
  sorry

end fractional_sum_identity_l53_53537


namespace integer_pairs_solution_l53_53728

def is_satisfied_solution (x y : ℤ) : Prop :=
  x^2 + y^2 = x + y + 2

theorem integer_pairs_solution :
  ∀ (x y : ℤ), is_satisfied_solution x y ↔ (x, y) = (-1, 0) ∨ (x, y) = (-1, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, -1) ∨ (x, y) = (1, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (2, 1) :=
by
  sorry

end integer_pairs_solution_l53_53728


namespace train_passes_bridge_time_l53_53683

theorem train_passes_bridge_time 
  (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) (t : ℝ) 
  (train_length_eq : train_length = 360)
  (bridge_length_eq : bridge_length = 140)
  (train_speed_kmh_eq : train_speed_kmh = 60) :
  t = 30 :=
by
  -- Constants and definitions
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let time := total_distance / train_speed_ms

  -- Assuming values from the conditions
  have train_length_val : train_length = 360 := train_length_eq
  have bridge_length_val : bridge_length = 140 := bridge_length_eq
  have train_speed_val : train_speed_kmh = 60 := train_speed_kmh_eq
  
  -- Substitute the values and perform the calculations
  rw [train_length_val, bridge_length_val, train_speed_val] at *
  simp [total_distance, train_speed_ms, time]
  
  -- Assert approximate value
  norm_num
  have h : abs (time - 30) < 0.01 := sorry
  exact calc
    t = time : by sorry
    ... ≈ 30 : by sorry

  -- Equating t to the expected result
  exfalso
  sorry

end train_passes_bridge_time_l53_53683


namespace find_expression_l53_53088

-- Given conditions
variables (a b : ℝ)
axiom ha : 30^a = 6
axiom hb : 30^b = 10

-- The statement we want to prove
theorem find_expression (ha : 30^a = 6) (hb : 30^b = 10) : 
  15^((3 - 2 * a - b) / (3 * (1 - b))) = 75 := 
sorry

end find_expression_l53_53088


namespace fourth_power_of_third_smallest_prime_cube_l53_53278

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53278


namespace sum_of_A_mul_B_l53_53359

theorem sum_of_A_mul_B :
  let A := {1, 2, 3, 5}
  let B := {1, 2}
  let A_mul_B := {x | ∃ (x1 ∈ A), ∃ (x2 ∈ B), x = x1 * x2}
  set.sum (coe A_mul_B : set ℕ) = 31 := by sorry

end sum_of_A_mul_B_l53_53359


namespace point_in_first_quadrant_l53_53884

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53884


namespace fourth_power_cube_third_smallest_prime_l53_53267

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53267


namespace partI_partII_l53_53105

variables {A B C : Type} -- variables representing the points
variables {a b c : ℝ} -- variables representing the sides opposite to angles A, B, C respectively

noncomputable def cosA (a b c: ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)
noncomputable def sinA (cosA: ℝ) : ℝ := sqrt (1 - cosA^2)
noncomputable def area (b c sinA: ℝ) : ℝ := (1 / 2) * b * c * sinA

-- Proving part (I)
theorem partI (a b c : ℝ) (h: 2 * sinA (cosA a b c)^2 = 3 * cosA a b c)
: (a^2 - c^2 = b^2 - (cosA a b c * 2) * b * c) → cosA a b c = 1 / 2 :=
sorry

-- Proving part (II)
theorem partII (a b c: ℝ) (h: a = sqrt 3)
: area b c (sinA (cosA a b c)) ≤ 3 * sqrt 3 / 4 :=
sorry

end partI_partII_l53_53105


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53283

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53283


namespace cos_double_angle_l53_53411

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 1 / 3) :
  Real.cos (2 * α) = 7 / 9 :=
sorry

end cos_double_angle_l53_53411


namespace right_triangle_area_l53_53372

-- Define the condition of the right triangle.
def right_triangle_leg_hypotenuse (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the area of a right triangle.
def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

-- Define the specific problem conditions.
def problem_conditions (a c : ℕ) : Prop :=
  a = 6 ∧ c = 10

-- The theorem statement for the proof problem.
theorem right_triangle_area : 
  ∀ (a b c : ℕ), 
  problem_conditions a c ∧ right_triangle_leg_hypotenuse a b c → triangle_area a b = 24 :=
by
  intros,
  cases h with pa pc,
  cases pa,
  cases pc,
  sorry

end right_triangle_area_l53_53372


namespace trajectory_sum_of_distances_to_axes_l53_53587

theorem trajectory_sum_of_distances_to_axes (x y : ℝ) (h : |x| + |y| = 6) :
  |x| + |y| = 6 := 
by 
  sorry

end trajectory_sum_of_distances_to_axes_l53_53587


namespace plane_equation_l53_53632

-- Define points A, B, and C as given in the conditions
def A : ℝ × ℝ × ℝ := (7, -5, 0)
def B : ℝ × ℝ × ℝ := (8, 3, -1)
def C : ℝ × ℝ × ℝ := (8, 5, 1)

-- Calculate the vector BC
def BC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- Define the normal vector based on BC
def n : ℝ × ℝ × ℝ := BC

-- Prove the equation of the plane
theorem plane_equation : 
  let l := n.1 * (x - A.1) + n.2 * (y - A.2) + n.3 * (z - A.3)
  in l = 0 → 
  (y + z + 5 = 0) := 
by
  -- The proof part is omitted as per instructions
  sorry

end plane_equation_l53_53632


namespace find_b_l53_53067

theorem find_b :
  ∃ b : ℝ, b > 1 ∧ (∀ x ∈ set.Icc (2 : ℝ) (2 * b), x^2 - 2 * x + 4 ∈ set.Icc (2 : ℝ) (2 * b)) ∧ b = 2 :=
by
  sorry

end find_b_l53_53067


namespace senior_discount_percentage_l53_53139

theorem senior_discount_percentage 
    (cost_shorts : ℕ)
    (count_shorts : ℕ)
    (cost_shirts : ℕ)
    (count_shirts : ℕ)
    (amount_paid : ℕ)
    (total_cost : ℕ := (cost_shorts * count_shorts) + (cost_shirts * count_shirts))
    (discount_received : ℕ := total_cost - amount_paid)
    (discount_percentage : ℚ := (discount_received : ℚ) / total_cost * 100) :
    count_shorts = 3 ∧ cost_shorts = 15 ∧ count_shirts = 5 ∧ cost_shirts = 17 ∧ amount_paid = 117 →
    discount_percentage = 10 := 
by
    sorry

end senior_discount_percentage_l53_53139


namespace propertyTaxRate_is_2_percent_l53_53137

-- Jenny's current house value
def currentHouseValue := 400000 : ℝ

-- The percentage increase due to the high-speed rail project
def railProjectIncrease := 0.25 : ℝ

-- The maximum property tax Jenny can afford per year
def maxPropertyTax := 15000 : ℝ

-- The improvements that Jenny can make to her house
def improvementsValue := 250000 : ℝ

-- The new house value after the high-speed rail project
def newHouseValue := currentHouseValue * (1 + railProjectIncrease)

-- The house value after making improvements
def houseValueAfterImprovements := newHouseValue + improvementsValue

-- The property tax rate
def propertyTaxRate := maxPropertyTax / houseValueAfterImprovements

theorem propertyTaxRate_is_2_percent :
  propertyTaxRate = 0.02 :=
by
  sorry

end propertyTaxRate_is_2_percent_l53_53137


namespace irene_weekly_income_l53_53965

noncomputable def base_salary : ℝ := 500
noncomputable def federal_tax_rate : ℝ := 0.15
noncomputable def health_insurance : ℝ := 50

def overtime_pay (hours : ℝ) : ℝ :=
  if hours ≤ 5 then hours * 20
  else if hours ≤ 8 then 5 * 20 + (hours - 5) * 30
  else 5 * 20 + 3 * 30 + (hours - 8) * 40

def total_income (base_salary: ℝ) (overtime_hours : ℝ) : ℝ :=
  base_salary + (overtime_pay overtime_hours)

def deductions (base_salary : ℝ) (federal_tax_rate : ℝ) (health_insurance : ℝ) : ℝ :=
  (base_salary * federal_tax_rate) + health_insurance

def net_income (base_salary : ℝ) (overtime_hours : ℝ) (federal_tax_rate : ℝ) (health_insurance : ℝ) : ℝ :=
  total_income base_salary overtime_hours - deductions base_salary federal_tax_rate health_insurance

theorem irene_weekly_income :
  net_income base_salary 10 federal_tax_rate health_insurance = 645 := by
  sorry

end irene_weekly_income_l53_53965


namespace fourth_power_cube_third_smallest_prime_l53_53265

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53265


namespace fourth_power_of_cube_third_smallest_prime_l53_53288

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53288


namespace probability_units_digit_2pow_a_plus_5pow_b_eq_4_l53_53026

def pow2_units_digits : Set ℕ := {2, 4, 8, 6}
def pow5_units_digit : ℕ := 5

theorem probability_units_digit_2pow_a_plus_5pow_b_eq_4 :
  (∃ n : ℕ, n ∈ pow2_units_digits ∧ (n + pow5_units_digit) % 10 = 4) →
  ∀ a b : ℕ, a ∈ Finset.range 1 101 → b ∈ Finset.range 1 101 →
  (a - 1) % 4 = 0 → (b - 1) % 1 = 0 →
  (Fintype.card {0 ≤ k < 100 | (2^k + 5^k) % 10 = 4} : ℚ) 
  / (Fintype.card {0 ≤ k < 100}) = 1 / 4 := by
  sorry

end probability_units_digit_2pow_a_plus_5pow_b_eq_4_l53_53026


namespace S_mod_7_l53_53769

def α : ℝ := (1 + Real.sqrt 5) / 2
def β : ℝ := (1 - Real.sqrt 5) / 2

noncomputable def a (n : ℕ) : ℝ :=
  1 / Real.sqrt 5 * (α ^ n - β ^ n)

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (Nat.choose n (i+1)) * a (i+1)

theorem S_mod_7 (n : ℕ) (h: 0 < n): 7 ∣ S n ↔ ∃ k : ℕ, n = 4 * k + 4 := 
  sorry

end S_mod_7_l53_53769


namespace min_value_condition_l53_53590

variable (a b : ℝ)

theorem min_value_condition (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 2) :
    (1 / a^2) + (1 / b^2) = 9 / 2 :=
sorry

end min_value_condition_l53_53590


namespace eq_sqrt_pattern_l53_53035

theorem eq_sqrt_pattern (a t : ℝ) (ha : a = 6) (ht : t = a^2 - 1) (h_pos : 0 < a ∧ 0 < t) :
  a + t = 41 := by
  sorry

end eq_sqrt_pattern_l53_53035


namespace log_sum_equal_two_l53_53704

noncomputable def log_sum_result : ℝ := 2 * real.log 63 + real.log 64

theorem log_sum_equal_two : log_sum_result = 2 :=
by
  sorry

end log_sum_equal_two_l53_53704


namespace units_digit_of_a_l53_53604

theorem units_digit_of_a :
  (2003^2004 - 2004^2003) % 10 = 7 :=
by
  sorry

end units_digit_of_a_l53_53604


namespace num_sequences_sum_10_l53_53021

theorem num_sequences_sum_10 : 
  let num_sequences := (Nat.choose 15 5 - 6)
  in num_sequences = (Nat.choose 15 5 - 6) :=
by
  sorry

end num_sequences_sum_10_l53_53021


namespace log_base_sqrt_10_l53_53004

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l53_53004


namespace unit_prices_and_max_notebooks_l53_53112

-- Definitions of the given conditions
def unit_price_pen (x : ℕ) : Prop := x > 0

def unit_price_notebook (x : ℕ) : Prop := x + 3 > 0

def condition1 (x : ℕ) : Prop := 390 / (x + 3) = 300 / x

def condition2 (y : ℕ) : Prop := 13 * y + 10 * (50 - y) ≤ 560

-- To be proven
theorem unit_prices_and_max_notebooks :
  ∃ (x y : ℕ), unit_price_pen x ∧ 
               unit_price_notebook x ∧ 
               condition1 x ∧ 
               condition2 y ∧ 
               x = 10 ∧ 
               x + 3 = 13 ∧ 
               y ≤ 20 :=
by
  existsi 10
  existsi 20
  simp [unit_price_pen, unit_price_notebook, condition1, condition2]
  sorry

end unit_prices_and_max_notebooks_l53_53112


namespace exists_strictly_positive_c_l53_53313

theorem exists_strictly_positive_c {a : ℕ → ℕ → ℝ} (h_diag_pos : ∀ i, a i i > 0)
  (h_off_diag_neg : ∀ i j, i ≠ j → a i j < 0) :
  ∃ (c : ℕ → ℝ), (∀ i, 
    0 < c i) ∧ 
    ((∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 > 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 < 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 = 0)) :=
by
  sorry

end exists_strictly_positive_c_l53_53313


namespace complex_number_quadrant_l53_53891

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53891


namespace digits_and_zeros_l53_53631

noncomputable def A (n : ℕ) : ℕ :=
  let a_k (k : ℕ) := if k = 1 then 9 else 9 * 10^(k-1) * k
  (Finset.range n).sum (λ k, a_k (k + 1)) + (n + 1)

noncomputable def B (n : ℕ) : ℕ :=
  let b_k (k : ℕ) := if k = 1 then 0 else (k - 1) * 9 * 10^(k-2)
  (Finset.range (n+1)).sum (λ k, b_k (k + 1)) + (n + 1)

theorem digits_and_zeros (n : ℕ) : A n = B (n - 1) :=
sorry

end digits_and_zeros_l53_53631


namespace range_of_a_l53_53053

theorem range_of_a (x y : ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 4 = 2 * x * y) :
  x^2 + 2 * x * y + y^2 - a * x - a * y + 1 ≥ 0 ↔ a ≤ 17 / 4 := 
sorry

end range_of_a_l53_53053


namespace complex_quadrant_l53_53933

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53933


namespace complex_point_in_first_quadrant_l53_53864

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53864


namespace complex_point_in_first_quadrant_l53_53849

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53849


namespace odd_n_for_n_minus_1_stones_l53_53574

theorem odd_n_for_n_minus_1_stones (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, (k ≤ n - 1 ∧ (∀ A : ℕ, A ∈ Finset.range n → ∃ l : Finset ℕ, l.card = n → l.erase l k (A + k % n) = (A - k % n + n) % n))) ↔ (n % 2 = 1) :=
sorry

end odd_n_for_n_minus_1_stones_l53_53574


namespace seven_c_plus_seven_d_eq_five_l53_53716

def h (x : ℝ) : ℝ := 7 * x - 6

def f_inv (x : ℝ) : ℝ := 7 * x - 4

def f (c d x : ℝ) : ℝ := c * x + d

noncomputable def find_constants : (ℝ × ℝ) :=
let x := 1 / 7 in
let y := 4 / 7 in
(x, y)

theorem seven_c_plus_seven_d_eq_five : 
  let ⟨c, d⟩ := find_constants in
  7 * c + 7 * d = 5 :=
by
  let ⟨c, d⟩ := find_constants
  show 7 * c + 7 * d = 5
  sorry

end seven_c_plus_seven_d_eq_five_l53_53716


namespace complex_point_in_first_quadrant_l53_53856

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53856


namespace max_subsets_l53_53744

theorem max_subsets (n : ℕ) (h : n ≥ 4) :
  ∃ (m : ℕ), m = n - 2 ∧ ∀ (A : finset (fin n)) (m : ℕ),
    (∀ i, (1 ≤ i ∧ i ≤ m) → (∀ A_i ∈ A, A_i.card = i) ∧
    (∀ i j, (1 ≤ i ∧ i < j ∧ j ≤ m) → A_i ⊆ A_j → false)) :=
begin
  -- Proof block
  sorry
end

end max_subsets_l53_53744


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53286

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53286


namespace digit_6_count_range_100_to_999_l53_53960

/-- The number of times the digit 6 is written in the integers from 100 through 999 inclusive is 280. -/
theorem digit_6_count_range_100_to_999 : 
  (∑ n in finset.Icc 100 999, (if digit 6 n then 1 else 0)) = 280 := 
sorry

end digit_6_count_range_100_to_999_l53_53960


namespace mode_of_salespersons_distribution_is_30_l53_53321

-- Given conditions
def salespersons_distribution : List (ℕ × ℕ) :=
  [(60, 1), (50, 4), (40, 4), (35, 6), (30, 7), (20, 3)]

-- Define a function to compute the mode from the given distribution
def mode (dist : List (ℕ × ℕ)) : ℕ :=
  dist.foldl (λ acc sales => if sales.2 > acc.2 then sales else acc) (0, 0).1 -- mode as the sales volume with the highest frequency

-- The theorem to be proven
theorem mode_of_salespersons_distribution_is_30 :
  mode salespersons_distribution = 30 :=
sorry

end mode_of_salespersons_distribution_is_30_l53_53321


namespace right_triangle_consecutive_sides_l53_53488

theorem right_triangle_consecutive_sides (n : ℕ) (n_pos : 0 < n) :
    (n+1)^2 + n^2 = (n+2)^2 ↔ (n = 3) :=
by
  sorry

end right_triangle_consecutive_sides_l53_53488


namespace arithmeticGeometricSeqProof_l53_53156

-- Define the arithmetic sequence a_n with initial term a_1 and common difference d
def arithmeticSeq (a₁ d n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sumFirstNTerms (a₁ d n : ℕ) : ℤ := n * a₁ + (n * (n - 1) * d) / 2

-- Define the geometric sequence b_n with initial term b₁ and common ratio q
def geometricSeq (b₁ q n : ℕ) : ℤ := b₁ * q^(n - 1)

theorem arithmeticGeometricSeqProof :
  ∀ (a₁ : ℕ) (S₅ : ℤ) (b₃ b₄ : ℕ), 
  a₁ = 2 ∧ S₅ = 40 ∧ b₃ = (a₁ + 2 * 3) ∧ b₄ = (a₁ + (4 * 3 - 1)) →
  (∃ d, S₅ = sumFirstNTerms a₁ d 5 ∧
       ∀ n, arithmeticSeq a₁ d n = 3 * n - 1 ∧
       ∃ b₁ q, b₃ = geometricSeq b₁ q 3 ∧ b₄ = geometricSeq b₁ q 4 ∧
                b₇ = geometricSeq b₁ q 7 ∧
                3 * 43 - 1 = geometricSeq b₁ q 7)
  :=
by
  intros a₁ S₅ b₃ b₄ h
  cases h with ha hs
  cases hs with hs hb₃
  cases hb₃ with hb₃ hb₄
  existsi 3 -- the common difference d

  split
  {
    rw [sumFirstNTerms, ha, nat.cast_succ, nat.cast_bit0, nat.cast_mul, nat.cast_one]
    norm_num
    sorry
  }
  split
  {
    intros n
    rw [arithmeticSeq, ha, nat.cast_succ, nat.cast_bit0, nat.cast_mul, nat.cast_one]
    norm_num
    sorry
  }
  existsi 2 -- the initial term of geometric sequence b₁
  split
  {
    has_sorry hb₃ sorry
  }
  split
  {
    has_sorry hb₄ sorry
  }
  split
  {
    rw [geometricSeq] sorry
  }
  rw [arithmeticSeq, nat.cast_succ, nat.cast_bit0, nat.cast_mul, nat.cast_one]
  norm_num
  sorry

end arithmeticGeometricSeqProof_l53_53156


namespace exist_even_cycle_l53_53757

theorem exist_even_cycle
  (n : ℕ)
  (h_n : n ≥ 4)
  (A : Fin n → (ℝ × ℝ))
  (collinear : ∀ i j k : Fin n, (A i ≠ A j ∧ A j ≠ A k ∧ A i ≠ A k) →
    (¬ collinear (A i) (A j) (A k)))
  (connected : ∀ i : Fin n, ∃ B : Finset (Fin n), (B.card ≥ 3 ∧ ∀ j ∈ B, connected_by_segment (A i) (A j))) :
  ∃ (k : ℕ) (C : Finset (Fin n)), k > 1 ∧ C.card = 2 * k ∧
    (∀ i : Fin (2 * k), connected_by_segment (A (C i)) (A (C (i % (2 * k) + 1)))) := 
sorry

end exist_even_cycle_l53_53757


namespace midpoint_distance_to_y_axis_l53_53780

-- Given definitions and conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def on_parabola (p q : ℝ) : Prop := parabola p q
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop := distance M F + distance N F = 5

-- Main theorem to prove
theorem midpoint_distance_to_y_axis (M N : ℝ × ℝ)
  (hM : on_parabola M.1 M.2)
  (hN : on_parabola N.1 N.2)
  (h_sum : sum_of_distances M N focus) :
  real.abs ((M.1 + N.1) / 2 - 0) = 3 / 2 :=
sorry

end midpoint_distance_to_y_axis_l53_53780


namespace number_of_valid_five_digit_numbers_l53_53804

-- Definitions based on conditions
def is_valid_five_digit_number (N : ℕ) : Prop :=
  let a := N / 10000
  let b := (N / 1000) % 10
  let c := (N / 100) % 10
  let d := (N / 10) % 10
  let e := N % 10 in
  50000 ≤ N ∧ N < 70000 ∧
  e ∈ {0, 5} ∧
  3 ≤ b ∧ b < c ∧ c ≤ 7 ∧
  (d = 0 ∨ d = 5) ∧
  d > b

-- The problem statement to prove in Lean
theorem number_of_valid_five_digit_numbers : 
  {N : ℕ | is_valid_five_digit_number N}.finite.toFinset.card = 56 := 
by
  sorry

end number_of_valid_five_digit_numbers_l53_53804


namespace complex_number_quadrant_l53_53894

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53894


namespace log_base_sqrt_10_l53_53003

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l53_53003


namespace find_x_in_equation_l53_53743

theorem find_x_in_equation :
  ∃ x : ℝ, 2.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2000.0000000000002 ∧ x = 0.3 :=
by 
  sorry

end find_x_in_equation_l53_53743


namespace distribution_of_stickers_l53_53166

theorem distribution_of_stickers (n k : ℕ) (hn : n = 10) (hk : k = 4) :
    (finset.range (n + k - 1)).card.choose (k - 1) = 847 :=
by
  sorry

end distribution_of_stickers_l53_53166


namespace calculate_total_investment_l53_53688

noncomputable def total_investment_in_business (A B C totalProfit shareA : ℕ) : Prop := 
  A = B + 6000 ∧ 
  B = C - 3000 ∧ 
  totalProfit = 8640 ∧ 
  shareA = 3168 ∧ 
  shareA = ((B + 6000) * totalProfit) / (B + C + 9000) 
  
theorem calculate_total_investment 
  (B : ℕ) 
  (h₁ : A = B + 6000) 
  (h₂ : B = C - 3000) 
  (h₃ : totalProfit = 8640) 
  (h₄ : shareA = 3168) 
  (h₅ : shareA = ((B + 6000) * totalProfit) / (B + C + 9000)) 
  : let A := B + 6000 
    let C := B + 3000 
    3 * B + 9000 = 90000 := 
by 
  let A := B + 6000,
  let C := B + 3000,
  sorry

end calculate_total_investment_l53_53688


namespace train_speed_is_36_km_per_hr_l53_53682

-- Define the given conditions
def length_of_train : ℝ := 110
def bridge_length : ℝ := 132
def crossing_time : ℝ := 24.198064154867613

-- Total distance covered by the train while crossing the bridge is the sum of the train length and the bridge length
def total_distance : ℝ := length_of_train + bridge_length

-- Speed of the train in m/s is the total distance divided by the crossing time
def speed_m_per_s : ℝ := total_distance / crossing_time

-- Convert speed from m/s to km/hr
def speed_km_per_hr : ℝ := speed_m_per_s * 3.6

-- The theorem: The speed of the train in km/hr is 36
theorem train_speed_is_36_km_per_hr : speed_km_per_hr ≈ 36 :=
by
  -- Proof is omitted
  sorry

end train_speed_is_36_km_per_hr_l53_53682


namespace angle_DAB_in_isosceles_triangle_with_pentagon_l53_53107

theorem angle_DAB_in_isosceles_triangle_with_pentagon
  (A B C D E G : Type)
  [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E] [LinearOrderedField G]
  (isosceles : ∀ {a b c : A}, a = b)
  (pentagon : ∀ {b c d e g : B}, 
    ∃ r : ℝ, 
    (polygon b c d e g) = (regular n = 5) ∧ 
    (angle b c d = 108))
  :
  (angle D A B = 54) :=
sorry

end angle_DAB_in_isosceles_triangle_with_pentagon_l53_53107


namespace complex_point_quadrant_l53_53870

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53870


namespace sum_bound_l53_53041

theorem sum_bound 
  {n : ℕ} (h₁ : 2 ≤ n) (x : ℕ → ℝ)
  (h₂ : ∑ i in Finset.range n, |x i| = 1)
  (h₃ : ∑ i in Finset.range n, x i = 0) :
  |∑ i in Finset.range n, x i / (i + 1)| ≤ 1/2 - 1/(2 * n) :=
  sorry

end sum_bound_l53_53041


namespace simplified_function_sum_l53_53432

theorem simplified_function_sum :
  let y (x : ℝ) := (x^3 + 9 * x^2 + 28 * x + 30) / (x + 3),
      A := 1,
      B := 6,
      C := 10,
      D := -3
  in ∀ x : ℝ, x ≠ -3 → y x = A * x^2 + B * x + C ∧ A + B + C + D = 14 :=
by
  assume y A B C D
  sorry

end simplified_function_sum_l53_53432


namespace incorrect_statement_D_l53_53303

-- Define the conditions as hypotheses
def symmetric_triangles_are_congruent (T1 T2 : Triangle) (L : Line) : Prop := 
  symmetric_about_line L T1 T2 → congruent T1 T2

def angles_are_symmetric_figures (θ : Angle) : Prop := 
  symmetric_about_bisector θ

def equilateral_triangle_axes_of_symmetry (T : Triangle) : Prop := 
  equilateral T → has_three_axes_of_symmetry T

def isosceles_triangle_altitude_median_bisector_coincide (T : Triangle) : Prop := 
  isosceles T → ∀ (a : Side), height_median_bisector_coincide T a

-- Given these conditions, we prove that the statement D is false (incorrect statement)
theorem incorrect_statement_D : 
  ¬ isosceles_triangle_altitude_median_bisector_coincide (some_triangle) := 
sorry

end incorrect_statement_D_l53_53303


namespace digit_6_count_in_100_to_999_l53_53954

theorem digit_6_count_in_100_to_999 : 
  (Nat.digits_count_in_range 6 100 999) = 280 := 
sorry

end digit_6_count_in_100_to_999_l53_53954


namespace area_of_triangle_CDP_l53_53978

noncomputable def area_triangle {A B C : Type*} [has_zero A] [has_mul A] [ring A] 
  (vertex1 vertex2 vertex3 : A × A) : A :=
1/2 * ((fst vertex1) * ((snd vertex2) - (snd vertex3)) +
  (fst vertex2) * ((snd vertex3) - (snd vertex1)) +
  (fst vertex3) * ((snd vertex1) - (snd vertex2)))

variables (O P : ℝ × ℝ)
def square_side_length : ℝ := 16
def square_center : ℝ × ℝ := (8, 8)
def semi_circle_center : ℝ × ℝ := (8, 0)
def radius : ℝ := 8

axiom PointP_cond : ℝ × ℝ
def P : ℝ × ℝ := (8 + real.sqrt 63, 1)

def C : ℝ × ℝ := (16, 16)
def D : ℝ × ℝ := (0, 16)

theorem area_of_triangle_CDP :
  area_triangle C D P = 120 :=
sorry

end area_of_triangle_CDP_l53_53978


namespace PT_PS_ratio_is_correct_l53_53508

structure Triangle :=
(P Q R M N S T : Point)
(PQ : Line P Q)
(PR : Line P R)
(PM : Segment P M)
(MQ : Segment M Q)
(PN : Segment P N)
(NR : Segment N R)
(PS : Segment P S)
(MN : Segment M N)
(T_on_PS : T ∈ PS)
(T_on_MN : T ∈ MN)
(PM_len : length PM = 2)
(MQ_len : length MQ = 6)
(PN_len : length PN = 4)
(NR_len : length NR = 8)

noncomputable def compute_PT_PS_ratio (Δ : Triangle) : ℚ :=
  PT / PS

theorem PT_PS_ratio_is_correct (Δ : Triangle) : compute_PT_PS_ratio Δ = 5 / 18 :=
sorry

end PT_PS_ratio_is_correct_l53_53508


namespace min_total_numbers_l53_53606

theorem min_total_numbers (N P : ℕ) (nums : Fin N → ℤ) 
  (h_avg_zero : (∑ i, nums i) = 0) 
  (h_pos_le : ∃ pos_nums : Fin P → Fin N,
    (∀ i, 0 < nums (pos_nums i)) ∧ P ≤ 29)
  : N ≥ 30 := by
  sorry

end min_total_numbers_l53_53606


namespace distinct_ordered_pairs_solution_l53_53720

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end distinct_ordered_pairs_solution_l53_53720


namespace prove_angle_C_120_prove_max_area_sqrt3_l53_53751

-- Define the conditions and required proofs in Lean

noncomputable def TriangleABC (A B C a b c S : ℝ) : Prop :=
  ∃ (C : ℝ), 
    (0 < B) ∧ (B < 180) ∧
    (C = 120) ∧
    (a * (real.cos 2 * C) + 2 * c * (real.cos A) * (real.cos C) + a + b = 0) ∧
    (b = 4 * real.sin B) ∧
    (S = 1/2 * a * b * (real.sin C)) ∧
    (a * b ≤ 4) 

-- Prove the size of angle C is 120 degrees
theorem prove_angle_C_120 (A B C a b c : ℝ) : 
  (∃ C, 0 < B ∧ B < 180 ∧ C = 120 ∧ (a * real.cos (2 * C) + 2 * c * real.cos A * real.cos C + a + b = 0)) → C = 120 :=
begin
  sorry,
end

-- Prove the maximum area of 𝑆 is √3
theorem prove_max_area_sqrt3 (A B C a b c S : ℝ) : 
  TriangleABC A B C a b c S →
  S = real.sqrt 3 :=
begin
  sorry,
end

end prove_angle_C_120_prove_max_area_sqrt3_l53_53751


namespace complex_point_in_first_quadrant_l53_53899

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53899


namespace arrange_triangle_vertices_on_lines_l53_53347

theorem arrange_triangle_vertices_on_lines
  {A B C D E F : Point}
  (line_AB : Line AB)
  (line_AC : Line AC)
  (line_BC : Line BC)
  (triangle_DEF : Triangle DEF)
  (non_parallel: ¬(Parallel line_AB line_AC ∨ Parallel line_AB line_BC ∨ Parallel line_AC line_BC)) :
  ∃ (P Q R : Point), 
    On P line_AB ∧ 
    On Q line_AC ∧ 
    On R line_BC ∧ 
    Triangle PQR ≅ Triangle DEF :=
by
  sorry

end arrange_triangle_vertices_on_lines_l53_53347


namespace triangle_isosceles_or_right_angled_l53_53479

-- Define that a triangle ABC satisfies a condition
variables {A B : ℝ} {a b c : ℝ}
def triangle_condition (A B : ℝ) (a b : ℝ) : Prop := 
  a * real.cos A = b * real.cos B

-- Main theorem stating that given the condition, the triangle is either isosceles or right-angled
theorem triangle_isosceles_or_right_angled (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_condition A B a b) : 
  (
    (A = B ∨ A + B = real.pi / 2)
  ) ↔ 
  (
    -- The triangle is isosceles or right-angled
    let s := 
      if A = B then "isosceles"
      else if A + B = real.pi / 2 then "right-angled"
      else "neither"
    in s = "isosceles" ∨ s = "right-angled"
  ) :=
by 
  sorry

end triangle_isosceles_or_right_angled_l53_53479


namespace problem_inequality_l53_53193

theorem problem_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n :=
sorry

end problem_inequality_l53_53193


namespace find_fg_neg1_eq_neg28_l53_53797

-- Lean code for the given problem
def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 + 3*x else g x

def g (x : ℝ) : ℝ :=
  -f (-x)

theorem find_fg_neg1_eq_neg28 : f (g (-1)) = -28 := 
by {
  sorry
}

end find_fg_neg1_eq_neg28_l53_53797


namespace four_identical_pairwise_differences_l53_53044

theorem four_identical_pairwise_differences 
  (A : Finset ℕ)
  (h_card_A : A.card = 20)
  (h_pos : ∀ a ∈ A, 1 ≤ a)
  (h_lt : ∀ a ∈ A, a < 70) : 
  ∃ d, ∃ B ⊆ A, B.card = 4 ∧ ∀ x y ∈ B, |x - y| = d :=
by
  sorry

end four_identical_pairwise_differences_l53_53044


namespace parabolic_trajectory_locus_l53_53034

-- Parameters for the problem
variables (c g : ℝ) (c_pos : 0 < c) (g_pos : 0 < g)

-- Theorem: The locus of the vertices of the parabolic trajectories is an ellipse.
theorem parabolic_trajectory_locus :
  ∀ X Y : ℝ,
  ∃ α : ℝ,
  X = c^2 / (2 * g) * Math.sin (2 * α) ∧
  Y = c^2 / (4 * g) * (1 - Math.cos (2 * α)) →
  (X / (c^2 / (2 * g)))^2 + (Y / (c^2 / (4 * g)))^2 = 1 :=
by
  sorry

end parabolic_trajectory_locus_l53_53034


namespace fourth_power_of_third_smallest_prime_cube_l53_53280

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53280


namespace hyperbola_asymptotes_l53_53050

theorem hyperbola_asymptotes (O P A B : Point) (a : Real) (m n : Real) (hyp : a > 0) 
    (hyperbola_eq : m^2 / a^2 - n^2 = 1) (area_eq : parallelogram_area O B P A = 1) :
    (∀ (x y : Real), asymptote_eqn1 : (x + 2 * y = 0) ∧ asymptote_eqn2 : (x - 2 * y = 0)) :=
sorry

end hyperbola_asymptotes_l53_53050


namespace abs_a_gt_b_l53_53396

theorem abs_a_gt_b (a b : ℝ) (h : a > b) : |a| > b :=
sorry

end abs_a_gt_b_l53_53396


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53273

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53273


namespace more_cats_than_spinsters_l53_53226

theorem more_cats_than_spinsters :
  ∀ (S C : ℕ), (S = 18) → (2 * C = 9 * S) → (C - S = 63) :=
by
  intros S C hS hRatio
  sorry

end more_cats_than_spinsters_l53_53226


namespace kylie_total_apples_l53_53145

theorem kylie_total_apples : (let first_hour := 66 in 
                              let second_hour := 2 * 66 in 
                              let third_hour := 66 / 3 in 
                              first_hour + second_hour + third_hour = 220) :=
by
  let first_hour := 66
  let second_hour := 2 * first_hour
  let third_hour := first_hour / 3
  show first_hour + second_hour + third_hour = 220
  sorry

end kylie_total_apples_l53_53145


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53282

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53282


namespace base_conversion_l53_53213

theorem base_conversion (C D : ℕ) (h₁ : 0 ≤ C ∧ C < 8) (h₂ : 0 ≤ D ∧ D < 5) (h₃ : 7 * C = 4 * D) :
  8 * C + D = 0 := by
  sorry

end base_conversion_l53_53213


namespace fourth_power_of_cube_third_smallest_prime_l53_53289

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53289


namespace polynomial_roots_l53_53380

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 5 +
                    Polynomial.C 13 * Polynomial.X ^ 4 +
                    Polynomial.C (-30) * Polynomial.X ^ 3 +
                    Polynomial.C 8 * Polynomial.X ^ 2) =
  {0, 0, 1 / 2, -2 + 2 * Real.sqrt 2, -2 - 2 * Real.sqrt 2} :=
by
  sorry

end polynomial_roots_l53_53380


namespace digit_6_count_in_range_l53_53947

-- Defining the range of integers and what is required to count.
def count_digit_6 (n m : ℕ) : ℕ :=
  (list.range' n (m - n + 1)).countp (λ k, k.digits 10).any (λ d, d = 6)

theorem digit_6_count_in_range :
  count_digit_6 100 999 = 280 :=
by
  sorry

end digit_6_count_in_range_l53_53947


namespace probability_odd_divisor_15_l53_53591

theorem probability_odd_divisor_15! :
  let D := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let O := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (O : ℚ) / D = 1 / 12 :=
by
  let D := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let O := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  have h1 : D = 12 * 7 * 4 * 3 * 2 * 2 := rfl
  have h2 : O = 7 * 4 * 3 * 2 * 2 := rfl
  have h3 : (O : ℚ) / D = (7 * 4 * 3 * 2 * 2 : ℚ) / (12 * 7 * 4 * 3 * 2 * 2 : ℚ) := by norm_num
  exact h3

end probability_odd_divisor_15_l53_53591


namespace intersection_in_first_quadrant_l53_53097

theorem intersection_in_first_quadrant (a : ℝ) :
  let P := λ (x y : ℝ), (ax + y - 4 = 0) ∧ (x - y - 2 = 0) in
  (∃ x y : ℝ, P x y ∧ x > 0 ∧ y > 0) ↔ (-1 < a ∧ a < 2) := sorry

end intersection_in_first_quadrant_l53_53097


namespace percentage_increase_is_correct_l53_53556

variable (orig_increase total_lines orig_lines : ℝ)
variable (percentage : ℝ)

def given_conditions : Prop := (orig_increase = 200) ∧ (total_lines = 350)
def calculation (orig_increase orig_lines total_lines percentage : ℝ) : Prop :=
  (total_lines = orig_lines + orig_increase) ∧
  (percentage = orig_increase / orig_lines * 100)

theorem percentage_increase_is_correct (h : given_conditions) :
  ∃ percentage, calculation 200 150 350 percentage ∧ percentage = 133.33 :=
by sorry

end percentage_increase_is_correct_l53_53556


namespace rearrange_terms_not_adjacent_l53_53022

/-- 
Given the expansion of (x^(1/2) + x^(1/3))^12,
prove that the number of ways to rearrange the terms
with positive integer powers of x so they are not adjacent is A 10 10 * A 3 11
-/
theorem rearrange_terms_not_adjacent : 
  let expansion := (x: ℝ)^(1/2) + (x: ℝ)^(1/3)
  let positive_integer_terms := {T | ∃ r, r ∈ {0, 6, 12} ∧ T = binomial 12 r * x ^ (6 - r/6)}
  A 10 10 * A 3 11 = 
    number_of_ways_to_rearrange_terms_not_adjacent expansion positive_integer_terms :=
sorry

end rearrange_terms_not_adjacent_l53_53022


namespace range_of_a_l53_53399

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 1) * x + 4 * a else log a x

theorem range_of_a (a : ℝ) : 
  (∀ x y, x < y → f a x ≥ f a y) → (1 / 7 ≤ a ∧ a < 1 / 3) :=
sorry

end range_of_a_l53_53399


namespace data_transmission_time_l53_53367

-- Define the number of blocks, number of chunks per block, and transmission rate.
def num_blocks : ℕ := 80
def chunks_per_block : ℕ := 768
def transmission_rate : ℕ := 160  -- chunks per second

-- Calculate the total number of chunks.
def total_chunks : ℕ := num_blocks * chunks_per_block

-- Calculate the transmission time in seconds.
def transmission_time_seconds : ℕ := total_chunks / transmission_rate

-- Convert transmission time from seconds to minutes.
def transmission_time_minutes : ℝ := transmission_time_seconds / 60

theorem data_transmission_time :
  transmission_time_minutes = 6.4 := 
by 
  -- The proof will involve arithmetic calculations (omitted here).
  sorry

end data_transmission_time_l53_53367


namespace max_perimeter_area_range_dot_product_l53_53480

-- Definitions for problem conditions
def sin_law (A B C : ℝ) (a b c : ℝ) : Prop := 
  a / sin A = b / sin B ∧ a / sin A = c / sin C

def given_condition (A B C a b c : ℝ) : Prop :=
  (sin A / (sin B + sin C)) = 1 - (a - b) / (a - c)

-- Proof problem statements
theorem max_perimeter_area 
  (A B C a b c : ℝ) 
  (h_condition : given_condition A B C a b c) 
  (h_b : b = real.sqrt 3) : 
  ∃ (area : ℝ), area = (3 * real.sqrt 3) / 4 :=
sorry

theorem range_dot_product 
  (A B : ℝ) 
  (m n : ℝ × ℝ)
  (h_m : m = (sin A, 1))
  (h_n : n = (6 * cos B, cos (2 * A))) : 
  ∃ (range : set ℝ), 
    range = set.Ioc 1 ((17:ℝ) / 8) :=
sorry

end max_perimeter_area_range_dot_product_l53_53480


namespace max_dist_eq_two_l53_53098

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6)
noncomputable def g (x : ℝ) : ℝ := cos x
noncomputable def MN_dist (x₀ : ℝ) : ℝ := abs (√3 * sin x₀ - cos x₀)

theorem max_dist_eq_two : ∃ x₀ : ℝ, MN_dist x₀ = 2 :=
by
  sorry

end max_dist_eq_two_l53_53098


namespace max_lateral_surface_area_cylinder_height_ratio_smaller_cone_frustum_l53_53045

variables (R H : ℝ) (R_pos : 0 < R) (H_pos : 0 < H)

-- Part 1: Proving the value of x and finding the maximum lateral surface area of the cylinder
theorem max_lateral_surface_area_cylinder (x : ℝ) (hx : x = H / 2) :
  let r := R - (R / H) * x in
  let S := 2 * π * r * x in
  S = (1 / 2) * π * R * H :=
by sorry

-- Part 2: Ratio of the height of the smaller cone to the height of the frustum
theorem height_ratio_smaller_cone_frustum :
  let a := (R * (H / (H / 2))) in
  let b := ((1 / 2)^(1 / 3)) * H in
  let c := H - b in
  b / c = (root 3 4) / (2 - root 3 4) :=
by sorry

end max_lateral_surface_area_cylinder_height_ratio_smaller_cone_frustum_l53_53045


namespace wedding_cost_l53_53517

def venue_cost := 10000
def food_and_drinks_cost_per_guest := 500
def johns_initial_guests := 50
def additional_guest_percentage := 0.60
def decorations_fixed_cost := 2500
def decoration_cost_per_guest := 10
def transportation_fixed_cost := 200
def transportation_cost_per_guest := 15
def entertainment_cost := 4000

def total_guests (initial_guests : ℕ) (additional_percentage : ℚ) : ℕ :=
  initial_guests + (initial_guests * additional_percentage).toInt

def total_cost (venue_cost : ℕ) (food_and_drinks_cost_per_guest : ℕ) 
  (total_guests : ℕ) (decorations_fixed_cost : ℕ) (decoration_cost_per_guest : ℕ) 
  (transportation_fixed_cost : ℕ) (transportation_cost_per_guest : ℕ)
  (entertainment_cost : ℕ) : ℕ :=
  venue_cost + 
  (food_and_drinks_cost_per_guest * total_guests) + 
  (decorations_fixed_cost + (decoration_cost_per_guest * total_guests)) + 
  (transportation_fixed_cost + (transportation_cost_per_guest * total_guests)) + 
  entertainment_cost

theorem wedding_cost : 
  total_cost venue_cost 
    food_and_drinks_cost_per_guest 
    (total_guests johns_initial_guests additional_guest_percentage)
    decorations_fixed_cost 
    decoration_cost_per_guest 
    transportation_fixed_cost 
    transportation_cost_per_guest 
    entertainment_cost = 58700 := 
  by 
  sorry

end wedding_cost_l53_53517


namespace parallelogram_base_length_l53_53577

theorem parallelogram_base_length (A H : ℝ) (base : ℝ) 
    (hA : A = 72) (hH : H = 6) (h_area : A = base * H) : base = 12 := 
by 
  sorry

end parallelogram_base_length_l53_53577


namespace exists_solution_l53_53524

def num_primes_le (n : ℕ) : ℕ :=
  -- this represents the number of primes less than or equal to n
  -- assume some definition here
  sorry

theorem exists_solution (n : ℕ) (m : ℕ) :
  (∀ n > 1, ∃ k, num_primes_le n = k ∧ (n / k = m)) →
  (m > 2) →
  (∃ k, (∃ n > 1, num_primes_le n = k ∧ (n / k = m - 1))) :=
begin
  sorry
end

end exists_solution_l53_53524


namespace suitable_bases_for_346_l53_53389

theorem suitable_bases_for_346 (b : ℕ) (hb : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0) : b = 6 ∨ b = 7 :=
sorry

end suitable_bases_for_346_l53_53389


namespace angle_between_vectors_l53_53412

variables {V : Type*} [inner_product_space ℝ V] [nontrivial V]
variables {u v : V}

theorem angle_between_vectors 
  (h₁ : ∥u + (2 : ℝ) • v∥ = ∥u - (2 : ℝ) • v∥)
  (h₂ : u ≠ 0)
  (h₃ : v ≠ 0) :
  inner_product_space.angle u v = real.pi / 2 :=
by
  sorry

end angle_between_vectors_l53_53412


namespace kylie_total_apples_l53_53142

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

end kylie_total_apples_l53_53142


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53261

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53261


namespace smallest_sum_three_diff_numbers_l53_53229

theorem smallest_sum_three_diff_numbers :
  let s := {-8, 2, -5, 17, -3} 
  ∃ a b c ∈ s, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (a + b + c = -16) :=
by
  let s := {-8, 2, -5, 17, -3}
  sorry

end smallest_sum_three_diff_numbers_l53_53229


namespace intersection_points_are_correct_l53_53847

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-3 + t, 1 - t)

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ + 2 * Real.cos θ = 0

noncomputable def cartesian_curve (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x = 0

noncomputable def intersection_points : List (ℝ × ℝ) :=
  [(-1, -1), (-2, 0)]

noncomputable def to_polar (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2), Real.atan2 y x)

def expected_polar_points : List (ℝ × ℝ) :=
  [(Real.sqrt 2, Real.pi / 4), (2, Real.pi)]

theorem intersection_points_are_correct :
  (to_polar (-1) (-1) ∈ expected_polar_points) ∧
  (to_polar (-2) 0 ∈ expected_polar_points) :=
by
  sorry

end intersection_points_are_correct_l53_53847


namespace problem_statement_l53_53454

theorem problem_statement (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 :=
by
  sorry

end problem_statement_l53_53454


namespace hayden_ride_pay_l53_53448

noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℝ := 8
noncomputable def gas_price_per_gallon : ℝ := 3
noncomputable def gallons_of_gas : ℝ := 17
noncomputable def review_bonus : ℝ := 20
noncomputable def number_of_reviews : ℝ := 2
noncomputable def total_rides : ℝ := 3
noncomputable def total_owed : ℝ := 226

theorem hayden_ride_pay :
  let hourly_earnings := hours_worked * hourly_wage,
      gas_reimbursement := gallons_of_gas * gas_price_per_gallon,
      review_earnings := number_of_reviews * review_bonus,
      other_earnings := hourly_earnings + gas_reimbursement + review_earnings,
      ride_pay := total_owed - other_earnings
  in ride_pay / total_rides = 5 :=
by
  sorry

end hayden_ride_pay_l53_53448


namespace maximum_distance_l53_53684

-- Given conditions for the problem.
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline : ℝ := 23

-- Problem statement: prove the maximum distance on highway mileage.
theorem maximum_distance : highway_mpg * gasoline = 280.6 :=
sorry

end maximum_distance_l53_53684


namespace vectors_opposite_directions_l53_53789

theorem vectors_opposite_directions (a b : ℝ^3) (h : ∃ k > 0, a = k • (-b)) :
  ∥a∥ + ∥b∥ = ∥a - b∥ :=
sorry

end vectors_opposite_directions_l53_53789


namespace fixed_amount_is_more_economical_l53_53622

-- Variables and definitions used in the conditions
variables (p1 p2 : ℝ) (hp1 : p1 > 0) (hp2 : p2 > 0)

-- Fixed quantity strategy: average price per kg
def avg_price_fixed_quantity : ℝ := (p1 + p2) / 2

-- Fixed amount strategy: average price per kg
def avg_price_fixed_amount : ℝ := 2 / (1 / p1 + 1 / p2)

-- Theorem stating the economically beneficial strategy
theorem fixed_amount_is_more_economical :
  avg_price_fixed_quantity p1 p2 ≥ avg_price_fixed_amount p1 p2 :=
sorry

end fixed_amount_is_more_economical_l53_53622


namespace jude_flips_3_heads_l53_53520

noncomputable def coin_flip_probability (n : ℕ) : ℝ :=
  if n = 0 then 1  -- base case
  else 1 / (n + 2)

noncomputable def p_n (n : ℕ) : ℝ :=
  if n = 2 then 2 / 3
  else if n = 3 then 3 / 4
  else (n / (n + 1)) * (p_n (n - 1)) + (1 / (n + 1)) * (p_n (n - 2))

noncomputable def p : ℝ :=
  1 - 2 / real.exp 1

theorem jude_flips_3_heads : ∃ (p : ℝ), p = (1 - 2 / real.exp 1) ∧ floor (180 * p) = 47 := by
  exists p
  -- providing the proof for this theorem is skipped
  sorry

end jude_flips_3_heads_l53_53520


namespace problem_inequality_l53_53453

theorem problem_inequality 
  (a b c d : ℝ)
  (h1 : d > 0)
  (h2 : a ≥ b)
  (h3 : b ≥ c)
  (h4 : c ≥ d)
  (h5 : a * b * c * d = 1) : 
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1 / 3)) :=
sorry

end problem_inequality_l53_53453


namespace reconstruct_triangle_l53_53841

/-
In a triangle, draw the altitude from one vertex, the angle bisector from another vertex,
and the median from the third vertex. Identify the points of their pairwise intersections,
then erase everything except these marked intersection points.
(The three marked points are distinct, and it is known which point corresponds to the intersection of which lines.)
Restore the triangle.
-/

variables (A B C X Y Z : Type) 
variables [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder X] [LinearOrder Y] [LinearOrder Z]

/-
Conditions:
1. X: point of intersection of the altitude from vertex A.
2. Y: point of intersection of the median from vertex C.
3. Z: point of intersection of the angle bisector from vertex B.
4. The points X, Y, Z are distinct.
-/

-- Assume points X, Y, Z and the vertices A, B, C
def is_triangle (A B C : Type) := Prop
def altitude_intersects (A B C X : Type) := Prop
def median_intersects (A B C Y : Type) := Prop
def angle_bisector_intersects (A B C Z : Type) := Prop

-- The goal is to reconstruct (identify the positions of) the original triangle ABC
theorem reconstruct_triangle
  (h1 : is_triangle A B C)
  (hX : altitude_intersects A B C X)
  (hY : median_intersects A B C Y)
  (hZ : angle_bisector_intersects A B C Z)
  (h_distinct : X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) :
  ∃ (A B C : Type), is_triangle A B C := 
sorry  -- Proof placeholder

end reconstruct_triangle_l53_53841


namespace fifty_third_number_in_game_is_53_l53_53485

theorem fifty_third_number_in_game_is_53 :
  ∀ (n : ℕ), 
  (∀ k, k < n → 
    ((∃ m, k = 3 * m) →
      ¬(∃ m', k + 1 = 3 * m') ∧ ∃ b, k + 1 + b ∉ {3 * m | m : ℕ})) → 
  n = 53 :=
begin
  sorry
end

end fifty_third_number_in_game_is_53_l53_53485


namespace part1_solution_set_part2_range_of_a_l53_53433

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1_solution_set (a : ℝ := 1) :
  {x : ℝ | f x a ≥ g x} = { x : ℝ | -1 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 } :=
by
  sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x ∈ [-1,1], f x a ≥ g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end part1_solution_set_part2_range_of_a_l53_53433


namespace min_value_3pow_a_plus_3pow_b_l53_53467

theorem min_value_3pow_a_plus_3pow_b (a b : ℝ) (h : a + b = 2) : 3^a + 3^b ≥ 6 :=
sorry

end min_value_3pow_a_plus_3pow_b_l53_53467


namespace minimal_M_l53_53677

theorem minimal_M {M a b c : ℕ} 
  (h1 : (a - 1) * (b - 1) * (c - 1) = 143)
  (h2 : M = a * b * c) : 
  M = 336 :=
begin
  sorry
end

end minimal_M_l53_53677


namespace angle_range_l53_53477

theorem angle_range (A B C : ℝ) (h₁ : 0 < A) (h₂ : A < π) (h₃ : 0 < B) (h₄ : B < π)
  (h₅ : 0 < C) (h₆ : C < π) (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.sin A + Real.sin B = Real.sqrt 3 * Real.sin C) : 
  A ∈ set.Icc (π / 6) (π / 2) :=
sorry

end angle_range_l53_53477


namespace distinct_domino_arrangements_l53_53173

theorem distinct_domino_arrangements : 
  let n := 7, k := 2 in (Nat.choose n k) = 21 := by
  let n := 7
  let k := 2
  have h : n.choose k = 21 := by sorry
  exact h

end distinct_domino_arrangements_l53_53173


namespace smallest_square_area_l53_53318

theorem smallest_square_area (a1 a2 b1 b2 : ℕ) (h1 : a1 = 2) (h2 : a2 = 3) (h3 : b1 = 3) (h4 : b2 = 4) :
    ∃ s, (s ≥ a1 + a2) ∧ (area_square s = s * s) ∧ (no_overlap (2, 3) (3, 4) s) ∧ (s * s = 25) :=
sorry

def area_square (s : ℕ) : ℕ := s * s

noncomputable def no_overlap (r1 r2 : ℕ × ℕ) (s : ℕ) : Prop := 
  let h_side := r1.1 + r2.1 ≤ s ∨ r1.2 + r2.2 ≤ s in
  h_side

end smallest_square_area_l53_53318


namespace local_max_at_x_1_l53_53430

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.log x - (1/2) * a * x^2 + (a - 1) * x

theorem local_max_at_x_1 {a : ℝ} :
  (∃ x : ℝ, x = 1 ∧
    (∀ y : ℝ, y ∈ Set.Ioo 0 1 → f a y < f a 1) ∧
    (∀ z : ℝ, z ∈ Set.Ioo 1 (1 : ℝ)∞ → f a z < f a 1))
  → 1 < a :=
by
  sorry

end local_max_at_x_1_l53_53430


namespace fourth_power_cube_third_smallest_prime_l53_53268

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53268


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53262

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53262


namespace ellipse_area_at_least_l53_53768

noncomputable def right_angled_triangle_area (d e : ℝ) := (d * e) / 2

theorem ellipse_area_at_least (t : ℝ) (a b c : ℝ) (d e : ℝ)
  (ha : a > b)
  (h_area_triangle : right_angled_triangle_area d e = t)
  (h_foci : a^2 = b^2 + c^2)
  (h_sum_legs : d + e = 2 * a)
  (h_pythagorean : d^2 + e^2 = 4 * c^2) :
  π * a * b ≥ sqrt 2 * π * t :=
by sorry

end ellipse_area_at_least_l53_53768


namespace game_promises_total_hours_l53_53968

open Real

noncomputable def total_gameplay_hours (T : ℝ) : Prop :=
  let boring_gameplay := 0.80 * T
  let enjoyable_gameplay := 0.20 * T
  let expansion_hours := 30
  (enjoyable_gameplay + expansion_hours = 50) → (T = 100)

theorem game_promises_total_hours (T : ℝ) : total_gameplay_hours T :=
  sorry

end game_promises_total_hours_l53_53968


namespace complex_number_quadrant_l53_53898

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53898


namespace ratio_of_edge_lengths_of_tetrahedrons_l53_53251

theorem ratio_of_edge_lengths_of_tetrahedrons :
  let edge_length_ratio : ℝ := 4 + 3 * Real.sqrt 6 in
  ∀ (ABCD MNPQ : Tetrahedron),
  let (B C D : Point) := (ABCD.Face 'BCD) in
  let (N P Q : Point) := (MNPQ.Face 'NPQ) in
  (BCD.ShapesCoincide NPQ) →
  (M.VertexOnAltitude 'AO ABCD) →
  (MNPQ.PassesThroughCenterAndMidpoint 'MNP (Face.Center 'ABC) (Edge.Midpoint 'BD)) →
  ABCD.EdgeLength * edge_length_ratio = MNPQ.EdgeLength :=
by
  sorry

end ratio_of_edge_lengths_of_tetrahedrons_l53_53251


namespace min_value_expression_l53_53020

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_l53_53020


namespace complex_min_distance_solution_l53_53538

def complex_min_distance_problem : Prop :=
  ∃ (w : ℂ), (|w + 2 - 2 * complex.I| + |w - 5 * complex.I| = 7) ∧
  ∀ z : ℂ, (|z + 2 - 2 * complex.I| + |z - 5 * complex.I| = 7) → |z| ≥ |(10 : ℝ) / 7|

theorem complex_min_distance_solution : complex_min_distance_problem :=
sorry

end complex_min_distance_solution_l53_53538


namespace store_loss_percentage_l53_53679

def CP_Radio : ℝ := 1500
def SP_Radio : ℝ := 1110
def CP_TV : ℝ := 8000
def SP_TV : ℝ := 7500
def CP_Refrigerator : ℝ := 25000
def SP_Refrigerator : ℝ := 23000
def CP_Microwave : ℝ := 6000
def SP_Microwave : ℝ := 6600
def CP_WashingMachine : ℝ := 14000
def SP_WashingMachine : ℝ := 13000

def Total_CP : ℝ := CP_Radio + CP_TV + CP_Refrigerator + CP_Microwave + CP_WashingMachine
def Total_SP : ℝ := SP_Radio + SP_TV + SP_Refrigerator + SP_Microwave + SP_WashingMachine

def Overall_Loss : ℝ := Total_CP - Total_SP
def Overall_Loss_Percentage : ℝ := (Overall_Loss / Total_CP) * 100

theorem store_loss_percentage :
  Overall_Loss_Percentage ≈ 6.03 := sorry

end store_loss_percentage_l53_53679


namespace solve_hyperbola_area_problem_l53_53070

noncomputable def hyperbola_area_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ) (he : e = 2) (p : ℝ) (hp : p > 0) : Prop :=
  let hyperbola_eq := (λ x y, x^2 / a^2 - y^2 / b^2 = 1)
  let parabola_eq := (λ x y, y^2 = 2 * p * x)
  let asymptote1 := (λ x y, y = (sqrt 3) * x)
  let asymptote2 := (λ x y, y = -(sqrt 3) * x)
  let O := (0, 0)
  let A := (2 * p / 3, (2 * sqrt 3 * p) / 3)
  let B := (2 * p / 3, -(2 * sqrt 3 * p) / 3)
  let area := (sqrt 3) / 3
  in area = (1 / 2) * (4 * sqrt 3 * p / 3) * (2 * p / 3) → p = sqrt 3 / 2

theorem solve_hyperbola_area_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ) (he : e = 2) (p : ℝ) (hp : p > 0) :
  hyperbola_area_problem a b ha hb e he p hp :=
sorry

end solve_hyperbola_area_problem_l53_53070


namespace average_value_function_example_l53_53717

def average_value_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x0 : ℝ, a < x0 ∧ x0 < b ∧ f x0 = (f b - f a) / (b - a)

theorem average_value_function_example :
  average_value_function (λ x => x^2 - m * x - 1) (-1) (1) → 
  ∃ m : ℝ, 0 < m ∧ m < 2 :=
by
  intros h
  sorry

end average_value_function_example_l53_53717


namespace num_pos_divisors_3960_l53_53741

theorem num_pos_divisors_3960 : 
  let n := 3960 in
  let prime_factors := [(2, 3), (3, 2), (5, 1), (11, 1)] in
  ∏ (x : ℕ × ℕ) in prime_factors, (x.2 + 1) = 48 :=
by
  sorry

end num_pos_divisors_3960_l53_53741


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53258

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53258


namespace complex_point_in_first_quadrant_l53_53900

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53900


namespace complex_point_in_first_quadrant_l53_53903

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53903


namespace perimeter_PQRS_l53_53495

-- Definition of the problem conditions
variables (P Q R S : Point)
variables (PQ QR RS PR : ℝ)
variables (angleQ_right : angle P Q R = 90)
variables (PR_RS_perpendicular : ∠ PRS = 90)
variables (PQ_eq : PQ = 15)
variables (QR_eq : QR = 20)
variables (RS_eq : RS = 9)

-- The goal statement
theorem perimeter_PQRS :
  PQ + QR + RS + sqrt (PQ^2 + QR^2) + sqrt ((PQ^2 + QR^2) + RS^2) = 44 + sqrt 706 :=
sorry

end perimeter_PQRS_l53_53495


namespace ant_position_after_2018_moves_l53_53645

def vertex : Type := { A : Type, B : Type, C : Type, D : Type }

def move_ant (start : vertex) (n : ℕ) : vertex :=
  match start with
  | vertex.A => if n % 4 = 0 then vertex.A else if n % 4 = 1 then vertex.B else if n % 4 = 2 then vertex.C else vertex.D
  | vertex.B => if n % 4 = 0 then vertex.B else if n % 4 = 1 then vertex.C else if n % 4 = 2 then vertex.D else vertex.A
  | vertex.C => if n % 4 = 0 then vertex.C else if n % 4 = 1 then vertex.D else if n % 4 = 2 then vertex.A else vertex.B
  | vertex.D => if n % 4 = 0 then vertex.D else if n % 4 = 1 then vertex.A else if n % 4 = 2 then vertex.B else vertex.C

theorem ant_position_after_2018_moves (start : vertex) : move_ant start 2018 = vertex.C := 
by
  sorry

end ant_position_after_2018_moves_l53_53645


namespace evaluate_expression_gcd_coprime_l53_53011

theorem evaluate_expression : 
  (2031 / 2020 - 2030 / 2031 : ℚ) = 4061 / 4070200 :=
by norm_num

theorem gcd_coprime (p q : ℕ) (h : p = 4061) (hq : q = 4070200) : gcd p q = 1 :=
by {
  simp [h, hq],
  norm_num,
}

end evaluate_expression_gcd_coprime_l53_53011


namespace problem_l53_53043

theorem problem (a b : ℝ) (h1 : abs a = 4) (h2 : b^2 = 9) (h3 : a / b > 0) : a - b = 1 ∨ a - b = -1 := 
sorry

end problem_l53_53043


namespace candidate_A_voting_percent_correct_l53_53637

variables (total_voters : ℕ) (democrats republicans : ℕ) (dem_voting_percent rep_voting_percent : ℚ)

-- Define the conditions
def total_voters : ℕ := 100
def democrats : ℕ := 60
def republicans : ℕ := 40
def dem_voting_percent : ℚ := 0.85
def rep_voting_percent : ℚ := 0.20

-- Define the number of democrats and republicans voting for candidate A
def dem_voting_for_A : ℕ := (dem_voting_percent * democrats).toNat
def rep_voting_for_A : ℕ := (rep_voting_percent * republicans).toNat

-- Define the total number of voters for candidate A
def total_voting_for_A : ℕ := dem_voting_for_A + rep_voting_for_A

-- Define the percentage of total voters for candidate A
def percentage_voting_for_A : ℚ := (total_voting_for_A : ℚ) / total_voters * 100

-- Problem statement: Verify that the percentage of voters for candidate A is 59%
theorem candidate_A_voting_percent_correct : percentage_voting_for_A = 59 := by
  sorry

end candidate_A_voting_percent_correct_l53_53637


namespace relationship_between_a_b_c_l53_53046

theorem relationship_between_a_b_c :
  let m := 2
  let n := 3
  let f (x : ℝ) := x^3
  let a := f (Real.sqrt 3 / 3)
  let b := f (Real.log Real.pi)
  let c := f (Real.sqrt 2 / 2)
  a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l53_53046


namespace lyle_payment_l53_53668

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l53_53668


namespace quotient_eq_l53_53555

theorem quotient_eq :
  ∀ (D d R Q : ℕ),
  D = 12401 → d = 163 → R = 13 → Q = (D - R) / d → Q = 76 :=
by
  intros D d R Q hD hd hR hQ
  have h1 : Q = 12401 - 13 / 163, sorry
  have h2 : Q = 76, sorry
  exact h2

end quotient_eq_l53_53555


namespace determine_a_square_binomial_l53_53718

theorem determine_a_square_binomial (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, ax^2 + 24*x + 9 = (r*x + s)^2) → a = 16 :=
by
  sorry

end determine_a_square_binomial_l53_53718


namespace min_divisors_of_power_plus_one_l53_53525

theorem min_divisors_of_power_plus_one (p : ℕ → ℕ) (n : ℕ) 
  (prime : ∀ i, (i < n) → Nat.Prime (p i)) 
  (gt_three : ∀ i, (i < n) → 3 < p i) :
  Nat.num_divisors (2 ^ (p 0 * p 1 * p 2 * ... * p (n - 1)) + 1) ≥ 4 ^ n :=
sorry

end min_divisors_of_power_plus_one_l53_53525


namespace problem1_correct_problem2_correct_l53_53023

noncomputable def problem1_solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

noncomputable def problem2_solution_set : Set ℝ := {x | (-3 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ≤ 7)}

theorem problem1_correct (x : ℝ) :
  (4 - x) / (x^2 + x + 1) ≤ 1 ↔ x ∈ problem1_solution_set :=
sorry

theorem problem2_correct (x : ℝ) :
  (1 < |x - 2| ∧ |x - 2| ≤ 5) ↔ x ∈ problem2_solution_set :=
sorry

end problem1_correct_problem2_correct_l53_53023


namespace find_n_l53_53341

theorem find_n (n : ℕ) (h1 : n > 0)
  (h2 : let red_faces := 6 * n^2 in
        let total_faces := 6 * n^3 in
        red_faces = total_faces / 3) : n = 3 :=
by
  sorry

end find_n_l53_53341


namespace find_a_value_l53_53405

theorem find_a_value (a : ℝ) : ∃ a, (line_passing_through_points_has_slope_angle (1 : ℝ) (-2 : ℝ) a (3 : ℝ) 45) → a = 6 :=
by
  -- The actual proof steps would go here.
  sorry

end find_a_value_l53_53405


namespace number_of_ways_to_choose_marbles_l53_53518

theorem number_of_ways_to_choose_marbles 
  (total_marbles : ℕ) 
  (red_count green_count blue_count : ℕ) 
  (total_choice chosen_rgb_count remaining_choice : ℕ) 
  (h_total_marbles : total_marbles = 15) 
  (h_red_count : red_count = 2) 
  (h_green_count : green_count = 2) 
  (h_blue_count : blue_count = 2) 
  (h_total_choice : total_choice = 5) 
  (h_chosen_rgb_count : chosen_rgb_count = 2) 
  (h_remaining_choice : remaining_choice = 3) :
  ∃ (num_ways : ℕ), num_ways = 3300 :=
sorry

end number_of_ways_to_choose_marbles_l53_53518


namespace quadratic_roots_diff_square_l53_53989

theorem quadratic_roots_diff_square :
  ∀ (d e : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x - 48 = 0 → (x = d ∨ x = e)) → (d - e)^2 = 49 :=
by
  intros d e h
  sorry

end quadratic_roots_diff_square_l53_53989


namespace evaluate_expression_l53_53705

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + 1) = 107 :=
by
  -- The proof will go here.
  sorry

end evaluate_expression_l53_53705


namespace sum_of_roots_l53_53813

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l53_53813


namespace calculate_savings_l53_53325

-- Define the subscription costs for each category
def subscription_A := 520
def subscription_B := 860
def subscription_C := 620

-- Define the percentage cuts for each category
def cut_A := 0.25
def cut_B := 0.35
def cut_C := 0.30

-- The required total savings
def total_savings := 617

-- Prove that the computed savings are equal to the total savings
theorem calculate_savings :
  (cut_A * subscription_A) + (cut_B * subscription_B) + (cut_C * subscription_C) = total_savings :=
by
  sorry

end calculate_savings_l53_53325


namespace point_in_first_quadrant_l53_53887

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53887


namespace candy_mixture_price_l53_53487

theorem candy_mixture_price
  (a : ℝ)
  (h1 : 0 < a) -- Assuming positive amount of money spent, to avoid division by zero
  (p1 p2 : ℝ)
  (h2 : p1 = 2)
  (h3 : p2 = 3)
  (h4 : p2 * (a / p2) = p1 * (a / p1)) -- Condition that the total cost for each type is equal.
  : ( (p1 * (a / p1) + p2 * (a / p2)) / (a / p1 + a / p2) = 2.4 ) :=
  sorry

end candy_mixture_price_l53_53487


namespace number_of_looping_paths_l53_53357

-- Definition of adjacency in the triangular array
def is_adjacent (i j : ℕ) : Prop :=
  (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 1) ∨
  (i = 2 ∧ j = 3) ∨ (i = 3 ∧ j = 2) ∨
  (i = 1 ∧ j = 4) ∨ (i = 4 ∧ j = 1) ∨
  (i = 2 ∧ j = 5) ∨ (i = 5 ∧ j = 2) ∨
  (i = 3 ∧ j = 6) ∨ (i = 6 ∧ j = 3) ∨
  (i = 4 ∧ j = 5) ∨ (i = 5 ∧ j = 4) ∨
  (i = 5 ∧ j = 6) ∨ (i = 6 ∧ j = 5) ∨
  (i = 4 ∧ j = 7) ∨ (i = 7 ∧ j = 4) ∨
  (i = 5 ∧ j = 8) ∨ (i = 8 ∧ j = 5) ∨
  (i = 6 ∧ j = 9) ∨ (i = 9 ∧ j = 6) ∨
  (i = 7 ∧ j = 8) ∨ (i = 8 ∧ j = 7) ∨
  (i = 8 ∧ j = 9) ∨ (i = 9 ∧ j = 8) ∨
  (i = 8 ∧ j = 10) ∨ (i = 10 ∧ j = 8) ∨
  (i = 9 ∧ j = 10) ∨ (i = 10 ∧ j = 9) ∨
  (i = 7 ∧ j = 8) ∨ (i = 8 ∧ j = 7)

-- Definition of a looping path
def is_looping_path (path : List ℕ) : Prop :=
  path.length = 10 ∧ 
  ∀ i ∈ List.range 10, is_adjacent (path.nthLe i sorry) (path.nthLe ((i + 1) % 10) sorry)

-- The main theorem: number of looping paths
theorem number_of_looping_paths : 
  ∃ n, is_looping_path n ∧ n = 60 :=
by
  sorry

end number_of_looping_paths_l53_53357


namespace complex_number_quadrant_l53_53893

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53893


namespace solution_x_value_divisible_by_45_l53_53385

theorem solution_x_value_divisible_by_45 (x : ℕ) (h : x ∈ Finset.range 10) :
    (∀ y : ℕ, y = 10000 * x + 2000 + 700 + 100 * x + 5 → y % 45 = 0) ↔ x = 2 := by sorry

end solution_x_value_divisible_by_45_l53_53385


namespace sum_of_digits_of_least_time_l53_53234

open Nat

-- Conditions and data encoding
def is_prime_or_multiple (k : ℕ) : Prop :=
  k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 11 ∨ k = 12

def horse_lap_time (k : ℕ) : ℕ :=
  if is_prime_or_multiple k then k else sorry

def at_starting_point (t : ℕ) (k : ℕ) : Prop :=
  t % horse_lap_time k = 0

def least_time_with_7_horses (t : ℕ) : Prop :=
  ∃ hs : Finset ℕ, hs.card = 7 ∧ (∀ k ∈ hs, is_prime_or_multiple k) ∧ (∀ k ∈ hs, at_starting_point t k)

-- Proof goal statement 
theorem sum_of_digits_of_least_time : least_time_with_7_horses 420 ∧ (4 + 2 + 0 = 6) :=
sorry

end sum_of_digits_of_least_time_l53_53234


namespace positive_n_of_single_solution_l53_53031

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l53_53031


namespace part1_part2_l53_53431

-- Part (1): Solution set of the inequality
theorem part1 (x : ℝ) : (|x - 1| + |x + 1| ≤ 8 - x^2) ↔ (-2 ≤ x) ∧ (x ≤ 2) :=
by
  sorry

-- Part (2): Range of real number t
theorem part2 (t : ℝ) (m n : ℝ) (x : ℝ) (h1 : m + n = 4) (h2 : m > 0) (h3 : n > 0) :  
  |x-t| + |x+t| = (4 * m^2 + n) / (m * n) → t ≥ 9 / 8 ∨ t ≤ -9 / 8 :=
by
  sorry

end part1_part2_l53_53431


namespace vasya_compartment_error_l53_53616

theorem vasya_compartment_error (seat_number : ℕ) (h : seat_number ∈ [25, 26, 27, 28, 29, 30]) :
  (⌈seat_number / 4⌉ = 7 ∨ ⌈seat_number / 4⌉ = 8) :=
by {
  sorry
}

end vasya_compartment_error_l53_53616


namespace area_quadrilateral_EFGH_l53_53196

noncomputable def quadrilateral := Type

variables {E F G H : quadrilateral}
variables (EF FG EH HG : ℝ)
variables (right_angle_F : ∃ θ : ℝ, θ = π / 2)
variables (right_angle_H : ∃ θ : ℝ, θ = π / 2)

def sides_distinct (a b c d : ℝ) : Prop := 
  ∃ a b, a ≠ b ∧ a ∈ {EF, FG, EH, HG} ∧ b ∈ {EF, FG, EH, HG}

def pythagorean_triple (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

-- Conditions
variables (EF_eq_5 : EF = 5)
variables (distinct_sides : sides_distinct 3 4 5)

theorem area_quadrilateral_EFGH : 
  ∃ (area : ℝ), area = 12 :=
by 
  sorry

end area_quadrilateral_EFGH_l53_53196


namespace triangle_perimeter_is_26_l53_53788

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l53_53788


namespace boys_in_third_group_l53_53572

variable (B G : ℝ)
variable (x : ℕ)

-- Conditions translated to Lean variables and expressions
def total_work : ℝ := 4 * x * B + 80 * G
def total_work_second_scenario : ℝ := 60 * B + 80 * G
def total_work_third_scenario : ℝ := 52 * B + 96 * G

-- Definition in Lean using the conditions and required proof problem
theorem boys_in_third_group : x = 15 → 26 = 26 :=
by
  -- Assuming total work is the same
  assume h : x = 15
  have h1 : total_work = total_work_second_scenario, from sorry
  have h2 : total_work = total_work_third_scenario, from sorry
  show 26 = 26, from rfl

end boys_in_third_group_l53_53572


namespace find_rectangular_equations_and_distance_l53_53416

section
variable (ρ θ α : ℝ)

-- Given the polar coordinate system setup and the polar equation of the line
def polar_line_equation (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2 / 2

-- Given the parametric equation of the curve C
def parametric_curve_equation (α : ℝ) (x y : ℝ) : Prop := 
  x = Real.cos α ∧ y = Real.sqrt 3 * Real.sin α

-- The rectangular coordinate equations to be proven
def rectangular_line_equation (x y : ℝ) : Prop := x + y = 3
def rectangular_curve_equation (x y : ℝ) : Prop := x^2 + y^2 / 3 = 1

-- The minimum distance to be proven
def min_distance (d : ℝ) : Prop := d = Real.sqrt 2 / 2

theorem find_rectangular_equations_and_distance (ρ θ α d x y : ℝ) :
  polar_line_equation ρ θ →
  parametric_curve_equation α x y →
  rectangular_line_equation x y ∧
  rectangular_curve_equation x y ∧
  min_distance (Real.abs (2 * Real.sin (θ + Real.pi / 6) - 3) / Real.sqrt 2) :=
sorry
end

end find_rectangular_equations_and_distance_l53_53416


namespace multiple_of_oranges_is_2_l53_53977

noncomputable section

def last_night_apples : ℕ := 3
def last_night_banana : ℕ := 1
def last_night_oranges : ℕ := 4
def today_apples : ℕ := last_night_apples + 4
def today_bananas : ℕ := 10 * last_night_banana
def total_fruits : ℕ := 39
def total_last_night_fruits : ℕ := last_night_apples + last_night_banana + last_night_oranges
def total_today_fruits : ℕ := total_fruits - total_last_night_fruits

theorem multiple_of_oranges_is_2 :
  ∃ x : ℕ, today_oranges = today_apples * x ∧ (today_apples + today_bananas + today_oranges = total_today_fruits) ∧ x = 2 :=
by
  let today_oranges : ℕ := today_apples * 2 -- we assume x = 2
  have h : (today_apples + today_bananas + today_oranges = total_today_fruits) := by sorry
  exact ⟨2, rfl, h⟩

end multiple_of_oranges_is_2_l53_53977


namespace triangle_area_difference_l53_53502

theorem triangle_area_difference (A B C D E : Type) 
  [EuclideanGeometry A B C D E] 
  (h1 : right_angle E A B)
  (h2 : right_angle A B C)
  (h3 : distance A B = 4)
  (h4 : distance B C = 7)
  (h5 : distance A E = 8)
  (h6 : intersect A C B E D) :
  area ΔADE - area ΔBDC = 2 :=
by
  sorry

end triangle_area_difference_l53_53502


namespace sum_of_divisors_77_and_perfectness_l53_53619

open Nat

-- Definitions for the conditions
def sum_of_divisors (n : ℕ) : ℕ :=
  (divisors n).sum

def is_perfect (n : ℕ) : Prop :=
  sum_of_divisors n = 2 * n

-- Main statement
theorem sum_of_divisors_77_and_perfectness :
  sum_of_divisors 77 = 96 ∧ ¬ is_perfect 77 :=
by
  sorry

end sum_of_divisors_77_and_perfectness_l53_53619


namespace triangle_longest_side_l53_53478

theorem triangle_longest_side (a b c : ℝ) (h1 : a - b = 4) (h2 : a + c = 2 * b) (h3 : ∀ x y z, is_largest_angle (x y z = 120)) :
  a = 14 := by
sorry

end triangle_longest_side_l53_53478


namespace greatest_integer_b_for_no_real_roots_l53_53734

theorem greatest_integer_b_for_no_real_roots :
  ∃ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ)*x + 15 ≠ 0) ∧ ∀ k : ℤ, (∀ x : ℝ, x^2 + (k : ℝ)*x + 15 ≠ 0) → k ≤ b := 
begin
  sorry
end

end greatest_integer_b_for_no_real_roots_l53_53734


namespace integral_value_l53_53580

open Real

-- Given condition: The coefficient of x^5 in the expansion of (ax + (sqrt 3) / 6)^6 is sqrt 3
lemma coefficient_condition (a : ℝ) : 
  (binomial_coefficient 6 5 * (a ^ 5) * ((sqrt 3 / 6) ^ 1) = sqrt 3) → 
  a = 1 :=
by sorry

-- Main theorem we want to prove
theorem integral_value (a : ℝ) (h : binomial_coefficient 6 5 * (a ^ 5) * ((sqrt 3 / 6) ^ 1) = sqrt 3) :
  ∫ x in 0..a, x^2 = 1/3 :=
by 
  -- using the coefficient_condition lemma to find a = 1
  have a_eq_1 : a = 1 := coefficient_condition a h,
  -- substitute a = 1 in the integral and check the result
  rw [a_eq_1],
  exact integral_quadratic
where
lemma integral_quadratic : (∫ x in 0..1, x^2 = 1 / 3) :=
by sorry

end integral_value_l53_53580


namespace percent_of_employed_people_who_are_females_l53_53505

theorem percent_of_employed_people_who_are_females (p_employed p_employed_males : ℝ) 
  (h1 : p_employed = 64) (h2 : p_employed_males = 48) : 
  100 * (p_employed - p_employed_males) / p_employed = 25 :=
by
  sorry

end percent_of_employed_people_who_are_females_l53_53505


namespace general_term_formula_range_of_a_l53_53048

def a_n (n : ℕ) : ℤ := 2 * n - 1

def b_n (n : ℕ) : ℚ := 1 / ( (a_n n) * (a_n n)  + 4 * n - 2 )

def S_n (n : ℕ) : ℚ := (∑ i in finset.range (n + 1), b_n i)

theorem general_term_formula (d : ℤ) (a : ℤ) (h1 : d ≠ 0) (h2 : a + 2 * d = 5) (h3 : (a + d) * (a + d) = a * (a + 4 * d)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 := sorry

theorem range_of_a (a : ℚ) :
  (∀ n : ℕ, 2 * S_n n + (-1)^(n + 1) * a > 0) ↔ (-2/3 : ℚ) < a ∧ a < (4/5 : ℚ) := sorry

end general_term_formula_range_of_a_l53_53048


namespace reward_lg_x_model_not_meet_minimum_a_for_reward_function_l53_53331

-- Statement of the first proof problem
theorem reward_lg_x_model_not_meet (x : ℝ) (y : ℝ) (h_cond1 : x = 100) (h_cond2 : y = 9) (h_range : 50 ≤ x ∧ x ≤ 500):
  ¬ (∀ k : ℝ, y = log10 x + k * x + 5 ∧ y ≤ 0.15 * x) := sorry

-- Statement of the second proof problem
theorem minimum_a_for_reward_function (a : ℝ) (h_a : 315 ≤ a ∧ a ≤ 344):
  ∀ x : ℝ, (50 ≤ x ∧ x ≤ 500) → (7 ≤ (15 * x - a) / (x + 8)) ∧ ((15 * x - a) / (x + 8) ≤ 0.15 * x) := sorry

end reward_lg_x_model_not_meet_minimum_a_for_reward_function_l53_53331


namespace complex_point_in_first_quadrant_l53_53853

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53853


namespace pure_imaginary_modulus_l53_53468

noncomputable def z_modulus (a : ℝ) : ℝ :=
  let z := complex.mk (a^2 - 1) (a + 1)
  complex.abs z

theorem pure_imaginary_modulus (a : ℝ) (h : a^2 - 1 = 0 ∧ a + 1 ≠ 0) : z_modulus a = 2 :=
  by
  sorry

end pure_imaginary_modulus_l53_53468


namespace part_a_part_b_l53_53222

section problem

variable (board : ℕ → ℕ → ℕ)

/-- Prove that the numbers 1 through 16 can be arranged on a 4x4 board such that any two numbers in cells that share a side differ by at most 4. -/
theorem part_a :
  (∃ (board : ℕ → ℕ → ℕ), 
     (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → board i j ∈ ({1, 2, 3, ..., 16} : set ℕ)) ∧ 
     (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → 
        ((i < 4 → abs (board i j - board (i+1) j) ≤ 4) ∧ 
         (j < 4 → abs (board i j - board i (j+1)) ≤ 4)))) :=
  sorry

/-- Prove that it is impossible to arrange the numbers 1 through 16 on a 4x4 board such that any two numbers in cells that share a side differ by at most 3. -/
theorem part_b :
  ¬(∃ (board : ℕ → ℕ → ℕ), 
      (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → board i j ∈ ({1, 2, 3, ..., 16} : set ℕ)) ∧ 
      (∀ i j, 1 ≤ i ∧ i ≤ 4 ∧ 1 ≤ j ∧ j ≤ 4 → 
         ((i < 4 → abs (board i j - board (i+1) j) ≤ 3) ∧ 
          (j < 4 → abs (board i j - board i (j+1)) ≤ 3)))) :=
  sorry

end problem

end part_a_part_b_l53_53222


namespace boxes_left_l53_53181

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l53_53181


namespace hexagon_division_ratio_l53_53666

theorem hexagon_division_ratio
  (hex_area : ℝ)
  (hexagon : ∀ (A B C D E F : ℝ), hex_area = 8)
  (line_PQ_splits : ∀ (above_area below_area : ℝ), above_area = 4 ∧ below_area = 4)
  (below_PQ : ℝ)
  (unit_square_area : ∀ (unit_square : ℝ), unit_square = 1)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (triangle_area : ∀ (base height : ℝ), triangle_base = 4 ∧ (base * height) / 2 = 3)
  (XQ QY : ℝ)
  (bases_sum : ∀ (XQ QY : ℝ), XQ + QY = 4) :
  XQ / QY = 2 / 3 :=
sorry

end hexagon_division_ratio_l53_53666


namespace max_projection_eq_neg_4_sqrt_2_div_3_l53_53072

variables 
  (e1 e2 : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖e1‖ = 2)
  (h2 : ‖(3 : ℝ) • e1 + e2‖ = 2)

theorem max_projection_eq_neg_4_sqrt_2_div_3 :
  projection_max (e1) (e2) = - (4 * real.sqrt 2) / 3 :=
begin
  sorry
end

end max_projection_eq_neg_4_sqrt_2_div_3_l53_53072


namespace projections_of_opposite_sides_equal_l53_53187

variable {α : Type*} [euclidean_space α]

-- Define the points A, B, C, D and the center O
variables (A B C D O : α)
-- Define that quadrilateral ABCD is inscribed in a circle with center O, and AC is the diameter
variable (circle : set α) (hA : A ∈ circle) (hB : B ∈ circle) (hC : C ∈ circle) (hD : D ∈ circle)
variables (hInscribed : ∀ P ∈ circle, dist O P = dist O A)
variables (hDiameter : dist A C = 2 * dist O A)

-- Prove that the projections of the opposite sides AB and CD onto the other diagonal BD are equal.
-- Note: interpreting projection here as the geometric foot of the perpendicular from the points to the line BD.
theorem projections_of_opposite_sides_equal (hProjections : ∀ {X Y : α}, dist O X = dist O Y) :
  ∃ (A₁ C₁ : α), is_foot_of_perp A A₁ B D ∧ is_foot_of_perp C C₁ B D ∧ dist B A₁ = dist D C₁ :=
begin
  sorry
end

end projections_of_opposite_sides_equal_l53_53187


namespace percentage_failed_both_l53_53491

theorem percentage_failed_both (p_hindi p_english p_pass_both x : ℝ)
  (h₁ : p_hindi = 0.25)
  (h₂ : p_english = 0.5)
  (h₃ : p_pass_both = 0.5)
  (h₄ : (p_hindi + p_english - x) = 0.5) : 
  x = 0.25 := 
sorry

end percentage_failed_both_l53_53491


namespace geometric_seq_sum_no_maximum_value_l53_53773

theorem geometric_seq_sum_no_maximum_value (a₁ : ℝ) (q : ℝ) (hq : q^2 > 1) (ha₁ : 0 < a₁) :
  ∀ n : ℕ, 
  let S_n := if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q) in
  ∃ N : ℕ, ∀ m : ℕ, N ≤ m → 
    (S_n > S_m) :=
sorry

end geometric_seq_sum_no_maximum_value_l53_53773


namespace point_in_first_quadrant_l53_53888

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53888


namespace choose_4_from_9_l53_53970

theorem choose_4_from_9
  (n k : ℕ)
  (hn : n = 9)
  (hk : k = 4) :
  (nat.choose n k = 126) :=
by sorry

end choose_4_from_9_l53_53970


namespace find_cost_expensive_module_l53_53493

-- Defining the conditions
def cost_cheaper_module : ℝ := 2.5
def total_modules : ℕ := 22
def num_cheaper_modules : ℕ := 21
def total_stock_value : ℝ := 62.5

-- The goal is to find the cost of the more expensive module 
def cost_expensive_module (cost_expensive_module : ℝ) : Prop :=
  num_cheaper_modules * cost_cheaper_module + cost_expensive_module = total_stock_value

-- The mathematically equivalent proof problem
theorem find_cost_expensive_module : cost_expensive_module 10 :=
by
  unfold cost_expensive_module
  norm_num
  sorry

end find_cost_expensive_module_l53_53493


namespace complex_quadrant_l53_53931

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53931


namespace roots_are_irrational_l53_53028

noncomputable def nature_of_roots (k : ℝ) : Prop :=
  let a : ℝ := 1
  let b : ℝ := -4 * k
  let c : ℝ := 3 * k^2 - 2
  let discriminant : ℝ := b^2 - 4 * a * c in
  discriminant > 0 ∧ ¬(∃ m n : ℤ, n ≠ 0 ∧ discriminant = (m / n)^2)

theorem roots_are_irrational (k : ℝ) (h : 3 * k^2 - 2 = 10) : nature_of_roots k :=
by
  sorry

end roots_are_irrational_l53_53028


namespace students_with_two_talents_l53_53351

theorem students_with_two_talents (total_students students_cannot_sing students_cannot_dance students_cannot_act students_can_sing students_can_dance students_can_act: ℕ) 
  (h1: total_students = 120) 
  (h2: students_cannot_sing = 50) 
  (h3: students_cannot_dance = 75) 
  (h4: students_cannot_act = 40) 
  (h5: students_can_sing = total_students - students_cannot_sing) 
  (h6: students_can_dance = total_students - students_cannot_dance) 
  (h7: students_can_act = total_students - students_cannot_act) 
  (h8: students_can_sing + students_can_dance + students_can_act = 195) : 
  ∃ students_exactly_two_talents : ℕ, students_exactly_two_talents = 75 :=
begin
  let students_with_exactly_two := students_can_sing + students_can_dance + students_can_act - total_students,
  use students_with_exactly_two,
  simp * at *,
  sorry,
end

end students_with_two_talents_l53_53351


namespace digits_solution_l53_53941

noncomputable def validate_reverse_multiplication
  (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
  (E * 10000 + D * 1000 + C * 100 + B * 10 + A)

theorem digits_solution :
  validate_reverse_multiplication 2 1 9 7 8 :=
by
  sorry

end digits_solution_l53_53941


namespace largest_integer_less_than_log_sum_l53_53294

open Real

theorem largest_integer_less_than_log_sum :
  (⌊∑ n in (Finset.range 1005).map (Nat.succ ∘ (λ n, n + 1)).map (λ n, log 3 (n.succ / n))⌋ : ℤ) = 5 := by
sorry

end largest_integer_less_than_log_sum_l53_53294


namespace angles_of_isosceles_triangle_are_60_degrees_l53_53963

-- Define the isosceles triangle ABC
variables {A B C K L : Type}
variables [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry K] [PlaneGeometry L]

-- Define the conditions given in the problem
def isIsoscelesTriangle (ABC : Triangle A B C) : Prop :=
  ABC.sides equalLengths AB AC

def midpointOfSegment (BK : Segment B K) (L : Point K) : Prop :=
  BK.midpoint L

def rightAngleAt (A K B : Point) : Prop :=
  ∠AKB = 90

def congruentSegments (AK CL : Segment) : Prop :=
  AK.length = CL.length

-- Problem statement:
theorem angles_of_isosceles_triangle_are_60_degrees
  (ABC : Triangle A B C)
  (h_iso : isIsoscelesTriangle ABC)
  (K_inside : Point K ∈ ABC)
  (L_mid : midpointOfSegment (Segment B K) L)
  (right_AKB : rightAngleAt A K B)
  (right_ALC : rightAngleAt A L C)
  (equal_legs : congruentSegments (Segment A K) (Segment C L))
  : angles_of_triangle ABC = (60°, 60°, 60°) :=
sorry

end angles_of_isosceles_triangle_are_60_degrees_l53_53963


namespace projection_of_a_plus_b_l53_53442

variable (a b : ℝ × ℝ)

def veca := (1, -1)
def vecb := (2, -1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def projection_magnitude (c a : ℝ × ℝ) : ℝ := (abs (dot_product c a)) / (magnitude a)

theorem projection_of_a_plus_b :
  projection_magnitude (vector_add veca vecb) veca = (5 * Real.sqrt 2) / 2 := sorry

end projection_of_a_plus_b_l53_53442


namespace basicAstrophysicsIs39_6Degrees_l53_53659

-- Define the given percentages for each research category
def microphotonics := 12
def homeElectronics := 22
def foodAdditives := 6
def geneticallyModifiedMicroorganisms := 28
def industrialLubricants := 7
def robotics := 5
def medicalTechnology := 9

-- Calculate the total percentage of all specified categories
def totalOtherCategories := microphotonics + homeElectronics + foodAdditives + geneticallyModifiedMicroorganisms + industrialLubricants + robotics + medicalTechnology

-- Define the total percentage and the remainder for basic astrophysics
def totalPercentage := 100
def basicAstrophysicsPercentage := totalPercentage - totalOtherCategories

-- Define the full circle in degrees
def fullCircle := 360

-- Calculate the degrees for basic astrophysics
def basicAstrophysicsDegrees := (basicAstrophysicsPercentage * fullCircle) / totalPercentage

-- Prove the degree measure of the sector representing the basic astrophysics budget
theorem basicAstrophysicsIs39_6Degrees : basicAstrophysicsDegrees = 39.6 := by
  sorry

end basicAstrophysicsIs39_6Degrees_l53_53659


namespace integral_value_l53_53752

noncomputable def integral_expression : ℝ :=
  ∫ x in 1..(Real.exp 1), x + (1 / x)

theorem integral_value : integral_expression = (1 / 2) * (Real.exp 1) ^ 2 + (1 / 2) :=
by
  -- Proof to be filled in later
  sorry

end integral_value_l53_53752


namespace sin_double_angle_BAD_l53_53198

-- Define the elements in the problem
variables (A B C D : Type) [metric_space C]
variables [has_le A] [linear_order A] 

-- Given conditions
variables {angle_C : A} {angle_CAD : A} (isosceles_ABC : isosceles_right_triangle ABC)
variables (right_angle_ABC : right_angle ABC) (leg_length : A) (construction : constructed_outwards ACD ABC)
variables (angle_CAD_eq_30_degrees : angle_CAD = 30)
variables (perimeter_equality : perimeter ABC = perimeter ACD)

-- Main proof statement
theorem sin_double_angle_BAD (condition : conditions):
  sin(2 * ∠BAD) = 1/2 :=
 sorry

end sin_double_angle_BAD_l53_53198


namespace polynomial_inequality_solution_l53_53719

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 - 16 * x^2 - 36 * x > 0 ↔ (x ∈ set.Ioo (-∞) -4) ∨ (x ∈ set.Ioo (-4) -1) ∨ (x ∈ set.Ioo 9 ∞) :=
sorry

end polynomial_inequality_solution_l53_53719


namespace sum_of_divisors_675_l53_53382

theorem sum_of_divisors_675 :
  (∑ d in (finset.divisors 675), d) = 1240 :=
by {
  -- Lean should check the correctness of this theorem based on imported libraries and intrinsic methods.
  sorry
}

end sum_of_divisors_675_l53_53382


namespace total_legs_on_street_l53_53186

open Nat

variable (total_animals : ℕ)
variable (cats_percent dogs_percent birds_percent insects_percent three_legged_dogs_percent : ℚ)

def number_of_cats := cats_percent * total_animals
def number_of_dogs := (dogs_percent - three_legged_dogs_percent) * total_animals
def number_of_three_legged_dogs := three_legged_dogs_percent * total_animals
def number_of_birds := birds_percent * total_animals
def number_of_insects := insects_percent * total_animals

def total_legs := 
  number_of_cats * 4 + 
  number_of_dogs * 4 + 
  number_of_three_legged_dogs * 3 + 
  number_of_birds * 2 +
  number_of_insects * 6

theorem total_legs_on_street 
  (h1 : cats_percent = 0.45) 
  (h2 : dogs_percent = 0.25) 
  (h3 : birds_percent = 0.10) 
  (h4 : insects_percent = 0.15) 
  (h5 : three_legged_dogs_percent = 0.05) 
  (h6 : total_animals = 300) 
  : total_legs total_animals cats_percent dogs_percent birds_percent insects_percent three_legged_dogs_percent = 1155 := 
by
  sorry

end total_legs_on_street_l53_53186


namespace fox_jeans_price_l53_53392

theorem fox_jeans_price (pony_price : ℝ)
                        (total_savings : ℝ)
                        (total_discount_rate : ℝ)
                        (pony_discount_rate : ℝ)
                        (fox_discount_rate : ℝ)
                        (fox_price : ℝ) :
    pony_price = 18 ∧
    total_savings = 8.91 ∧
    total_discount_rate = 0.22 ∧
    pony_discount_rate = 0.1099999999999996 ∧
    fox_discount_rate = 0.11 →
    (3 * fox_discount_rate * fox_price + 2 * pony_discount_rate * pony_price = total_savings) →
    fox_price = 15 :=
by
  intros h h_eq
  rcases h with ⟨h_pony, h_savings, h_total_rate, h_pony_rate, h_fox_rate⟩
  sorry

end fox_jeans_price_l53_53392


namespace coeff_of_x4_in_expansion_l53_53617

theorem coeff_of_x4_in_expansion : 
  (∃ c : ℕ, (5 * x - 2) ^ 8 = c * x ^ 4 + _) → (c = 700000) := 
by sorry

end coeff_of_x4_in_expansion_l53_53617


namespace trigon_expr_correct_l53_53708

noncomputable def trigon_expr : ℝ :=
  1 / Real.sin (Real.pi / 6) - 4 * Real.sin (Real.pi / 3)

theorem trigon_expr_correct : trigon_expr = 2 - 2 * Real.sqrt 3 := by
  sorry

end trigon_expr_correct_l53_53708


namespace card_distribution_count_l53_53560

theorem card_distribution_count : 
  ∃ (methods : ℕ), methods = 18 ∧ 
  ∃ (cards : Finset ℕ),
  ∃ (envelopes : Finset (Finset ℕ)), 
  cards = {1, 2, 3, 4, 5, 6} ∧ 
  envelopes.card = 3 ∧ 
  (∀ e ∈ envelopes, (e.card = 2) ∧ ({1, 2} ⊆ e → ∃ e1 e2, {e1, e2} ∈ envelopes ∧ {e1, e2} ⊆ cards \ {1, 2})) ∧ 
  (∀ c1 ∈ cards, ∃ e ∈ envelopes, c1 ∈ e) :=
by
  sorry

end card_distribution_count_l53_53560


namespace probability_no_adjacent_sums_10_l53_53225

theorem probability_no_adjacent_sums_10 :
  ∃ m n : ℕ, gcd m n = 1 ∧ m + n = 85 ∧ (∑ (f : fin 8) in (finset.univ : finset (fin 8)), some_primes f) ∧
  ∀ (i j : fin 8), adj_faces i j → ¬ (some_primes i + some_primes j) % 10 = 0 → (m / n = 1 / 84) := sorry

end probability_no_adjacent_sums_10_l53_53225


namespace problem_solution_l53_53060

theorem problem_solution (a m x : ℝ)
  (h1 : (a + 6)^2 = m)
  (h2 : (2a - 9)^2 = m) :
  a = 1 ∧ m = 49 ∧ (x = 4 ∨ x = -4) :=
by
  sorry

end problem_solution_l53_53060


namespace necessary_but_not_sufficient_condition_l53_53806

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 3 * x < 0) → (0 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l53_53806


namespace total_edge_length_of_parallelepiped_l53_53976

/-- Kolya has 440 identical cubes with a side length of 1 cm.
Kolya constructs a rectangular parallelepiped from these cubes 
and all edges have lengths of at least 5 cm. Prove 
that the total length of all edges of the rectangular parallelepiped is 96 cm. -/
theorem total_edge_length_of_parallelepiped {a b c : ℕ} 
  (h1 : a * b * c = 440) 
  (h2 : a ≥ 5) 
  (h3 : b ≥ 5) 
  (h4 : c ≥ 5) : 
  4 * (a + b + c) = 96 :=
sorry

end total_edge_length_of_parallelepiped_l53_53976


namespace monotonicity_g_range_a_l53_53069

def g (x : ℝ) := Real.exp x - Real.exp 1 * x - 1
def h (x : ℝ) (a : ℝ) := a * Real.sin x - Real.exp 1 * x

theorem monotonicity_g :
  (∀ x, g'A x < 0 → x < 1) ∧ (∀ x, x > 1 → g'A x > 0) := 
by
  let g'A (x : ℝ) := Real.exp x - Real.exp 1
  show (∀ x : ℝ, g'A x < 0 → x < 1) ∧ (∀ x : ℝ, x > 1 → g'A x > 0)
  sorry 

theorem range_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g x ≥ h x a) → a ∈ Set.Iic 1 := 
by
  let f (x : ℝ) := g x - h x a
  let f'A (x : ℝ) := Real.exp x - a * Real.cos x
  show (∀ x ∈ Set.Icc 0 1, f x ≥ 0) → a ∈ Set.Iic 1
  sorry

end monotonicity_g_range_a_l53_53069


namespace product_csc_squared_l53_53628

noncomputable def csc_deg (x : ℝ) : ℝ := 1 / Real.sin (x * Real.pi / 180)

theorem product_csc_squared :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m^n = ∏ k in Finset.range 45, (csc_deg (2 * k + 1))^2 ∧ m + n = 93 :=
by
  sorry

end product_csc_squared_l53_53628


namespace circle_area_l53_53846

theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 9 * x + 6 * y + 27 = 0) →
  ∃ (r : ℝ), r = 2 ∧ (real.pi * r^2 = 4 * real.pi) := 
by
  intro h
  use 2
  split
  exact rfl
  sorry

end circle_area_l53_53846


namespace total_birds_in_store_l53_53672

def num_bird_cages := 4
def parrots_per_cage := 8
def parakeets_per_cage := 2
def birds_per_cage := parrots_per_cage + parakeets_per_cage
def total_birds := birds_per_cage * num_bird_cages

theorem total_birds_in_store : total_birds = 40 :=
  by sorry

end total_birds_in_store_l53_53672


namespace complex_point_in_first_quadrant_l53_53902

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53902


namespace derivative_of_y_l53_53732

noncomputable def y (x : ℝ) : ℝ :=
  real.sqrt (1 - 3 * x - 2 * x^2) + (3 / (2 * real.sqrt 2)) * real.arcsin ((4 * x + 3) / real.sqrt 17)

theorem derivative_of_y (x : ℝ) : deriv y x = - (2 * x) / real.sqrt (1 - 3 * x - 2 * x^2) :=
by sorry

end derivative_of_y_l53_53732


namespace sum_even_integers_40_to_60_l53_53830

def sum_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 * (a + b) / 2

def number_of_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_even_integers_40_to_60 : 
  let x := sum_even_integers 40 60 in
  let y := number_of_even_integers 40 60 in
  x + y = 561 → x = 550 :=
by
  sorry

end sum_even_integers_40_to_60_l53_53830


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53284

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53284


namespace find_tangent_sum_l53_53085

theorem find_tangent_sum 
  (x y : ℝ)
  (h1 : (sin x / cos y) + (sin y / cos x) = 2)
  (h2 : (cos x / sin y) + (cos y / sin x) = 3) :
  (tan x / tan y) + (tan y / tan x) = 2 :=
by
  sorry

end find_tangent_sum_l53_53085


namespace find_r_l53_53094

variable (k r : ℝ)

theorem find_r (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (1/2) * Real.log 9 / Real.log 2 :=
sorry

end find_r_l53_53094


namespace probability_of_favorable_position_l53_53250

def favorable_probability (r : ℝ) : ℝ :=
  let total_area := (r * real.pi) ^ 2
  let favorable_area := ((r * real.pi)^2) * (7 / 9)
  favorable_area / total_area

theorem probability_of_favorable_position (r : ℝ) :
  favorable_probability r = 7 / 9 :=
  by sorry

end probability_of_favorable_position_l53_53250


namespace two_digit_combinations_count_l53_53794

/-- Given the set of digits {1, 3, 5, 8, 9}, prove that the number of different 
two-digit integers that can be formed by using these digits (with repetitions allowed) is 25. -/
theorem two_digit_combinations_count :
  let digits := {1, 3, 5, 8, 9}
  in (Set.card digits) * (Set.card digits) = 25 := 
by
  sorry

end two_digit_combinations_count_l53_53794


namespace probability_of_two_blue_gumballs_l53_53514

noncomputable def P_P : ℝ := 0.1428571428571428

def P_B : ℝ := 1 - P_P

def consecutive_Blues := P_B * P_B

theorem probability_of_two_blue_gumballs 
  (hP_P : P_P = 0.1428571428571428) 
  (hSum : P_B + P_P = 1) : 
  consecutive_Blues ≈ 0.7347 := 
by
  sorry

end probability_of_two_blue_gumballs_l53_53514


namespace determine_f_of_conditions_l53_53409

noncomputable def f (x : ℝ) := sin (π * x + π)

theorem determine_f_of_conditions 
  (ω : ℝ) (φ : ℝ) (M : ℝ) 
  (hω_pos : 0 < ω) 
  (hφ_range : 0 < φ ∧ φ < 2 * π) 
  (area_ABC : 1 / 2 = 1 / 2)  -- Given the area of ΔABC
  (hM_pos : 0 < M) 
  (h_func_eq : ∀ x, f(x + M) = M * f(-x))
  : f = (λ x, -sin (π * x)) :=
sorry

end determine_f_of_conditions_l53_53409


namespace complex_quadrant_is_first_l53_53927

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53927


namespace xyz_value_l53_53408

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
  sorry

end xyz_value_l53_53408


namespace geometric_sequence_Sn_geometric_sequence_Sn_l53_53483

noncomputable def Sn (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1/3 then (27/2) - (1/2) * 3^(n - 3)
  else if q = 3 then (3^n - 1) / 2
  else 0

theorem geometric_sequence_Sn (a1 : ℝ) (n : ℕ) (h1 : a1 * (1/3) = 3)
  (h2 : a1 + a1 * (1/3)^2 = 10) : 
  Sn a1 (1/3) n = (27/2) - (1/2) * 3^(n - 3) :=
by
  sorry

theorem geometric_sequence_Sn' (a1 : ℝ) (n : ℕ) (h1 : a1 * 3 = 3) 
  (h2 : a1 + a1 * 3^2 = 10) : 
  Sn a1 3 n = (3^n - 1) / 2 :=
by
  sorry

end geometric_sequence_Sn_geometric_sequence_Sn_l53_53483


namespace pages_read_on_wednesday_l53_53368

theorem pages_read_on_wednesday (W : ℕ) (h : 18 + W + 23 = 60) : W = 19 :=
by {
  sorry
}

end pages_read_on_wednesday_l53_53368


namespace sum_of_perimeters_greater_than_1993_l53_53685

theorem sum_of_perimeters_greater_than_1993 :
  ∃ (squares : ℕ → set (ℝ × ℝ)) (n : ℕ), (∀ k, squares k ⊆ λ xy, 0 ≤ xy.1 ∧ xy.1 ≤ 1 ∧ 0 ≤ xy.2 ∧ xy.2 ≤ 1) ∧
  (∀ k, ∀ (x y : ℝ × ℝ), (x ∈ squares k ∧ y ∈ squares k) → (x.1 = y.1 ∨ x.2 = y.2)) ∧
  (∀ k, (∃ x, ∃ y, x ≠ y ∧ x ∈ squares k ∧ y ∈ squares k ∧ x.1 = y.1 ∧ x.2 ≠ y.2)) ∧
  ∃ k, 4 * k > 1993 := by
  sorry

end sum_of_perimeters_greater_than_1993_l53_53685


namespace max_harmonious_T_shapes_l53_53490

def cell := bool -- representing black or white cells

def grid := list (list cell)

def is_harmonious (t : list (list cell)) : bool :=
  (t.count tt = 2) ∧ (t.count ff = 2)

def valid_T_shapes (g : grid) : list (list (list cell)) :=
  sorry -- functionality to extract all possible T-shaped patterns

def count_harmonious (g : grid) : nat :=
  (valid_T_shapes g).count is_harmonious

theorem max_harmonious_T_shapes (g : grid) (hg : g.length = 8 ∧ ∀ row in g, row.length = 8) : 
  count_harmonious g ≤ 132 :=
sorry

end max_harmonious_T_shapes_l53_53490


namespace max_value_of_water_l53_53504

theorem max_value_of_water :
  ∃ (尽 山 力 心 可 拔 穷 水 : ℕ),
    (尽 ≠ 山) ∧ (尽 ≠ 力) ∧ (尽 ≠ 心) ∧ (尽 ≠ 可) ∧ (尽 ≠ 拔) ∧ (尽 ≠ 穷) ∧ (尽 ≠ 水)
    ∧ (山 ≠ 力) ∧ (山 ≠ 心) ∧ (山 ≠ 可) ∧ (山 ≠ 拔) ∧ (山 ≠ 穷) ∧ (山 ≠ 水)
    ∧ (力 ≠ 心) ∧ (力 ≠ 可) ∧ (力 ≠ 拔) ∧ (力 ≠ 穷) ∧ (力 ≠ 水)
    ∧ (心 ≠ 可) ∧ (心 ≠ 拔) ∧ (心 ≠ 穷) ∧ (心 ≠ 水)
    ∧ (可 ≠ 拔) ∧ (可 ≠ 穷) ∧ (可 ≠ 水)
    ∧ (拔 ≠ 穷) ∧ (拔 ≠ 水)
    ∧ (穷 ≠ 水) -- All variables are distinct
    ∧ 1 ≤ 尽 ∧ 尽 ≤ 8 ∧ 1 ≤ 山 ∧ 山 ≤ 8 ∧ 1 ≤ 力 ∧ 力 ≤ 8 ∧ 1 ≤ 心 ∧ 心 ≤ 8
    ∧ 1 ≤ 可 ∧ 可 ≤ 8 ∧ 1 ≤ 拔 ∧ 拔 ≤ 8 ∧ 1 ≤ 穷 ∧ 穷 ≤ 8 ∧ 1 ≤ 水 ∧ 水 ≤ 8 -- All variables between 1 and 8
    ∧ (尽 + 心 + 尽 + 力 = 19)
    ∧ (力 + 可 + 拔 + 山 = 19)
    ∧ (山 + 穷 + 水 + 尽 = 19)
    ∧ (尽 > 山)
    ∧ (山 > 力)
    ∧ (水 = 7) := sorry

end max_value_of_water_l53_53504


namespace increasing_when_x_negative_l53_53358

def my_op (m n : ℝ) : ℝ := -n / m

theorem increasing_when_x_negative (x y : ℝ) (h : x < 0) : y = my_op x 2 → ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → my_op x1 2 < my_op x2 2 := by 
  sorry

end increasing_when_x_negative_l53_53358


namespace total_filled_water_balloons_l53_53175

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end total_filled_water_balloons_l53_53175


namespace cot_sum_arccot_roots_l53_53160

theorem cot_sum_arccot_roots :
  let z := λ k, polynomial.root (polynomial.C 36 + polynomial.X ^ 10 - polynomial.C 3 * polynomial.X ^ 9 + polynomial.C 6 * polynomial.X ^ 8 - polynomial.C 10 * polynomial.X ^ 7 + polynomial.C 36) k in
  (∑ k in finset.range 10, real.arccot (z k)).cot = 88 / 73 :=
sorry

end cot_sum_arccot_roots_l53_53160


namespace arithmetic_sequence_common_difference_l53_53114

open Nat

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h1 : a 4 = 2)
  (h2 : (∑ i in (finset.range 10).map (λ x, x + 1), a i) = 65)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 3 :=
by 
  sorry

end arithmetic_sequence_common_difference_l53_53114


namespace sum_of_integers_with_abs_gt_1_lt_4_l53_53127

theorem sum_of_integers_with_abs_gt_1_lt_4 : 
  ∑ x in ({-3, -2, 2, 3} : Finset ℤ), x = 0 :=
by
  sorry

end sum_of_integers_with_abs_gt_1_lt_4_l53_53127


namespace total_concrete_needed_l53_53519

theorem total_concrete_needed :
  (road_deck : ℕ) (single_anchor : ℕ) (pillars : ℕ)
  (h_road_deck : road_deck = 1600)
  (h_single_anchor : single_anchor = 700)
  (h_pillars : pillars = 1800)
  (anchors_eq : 2 * single_anchor = 1400) :
  road_deck + (2 * single_anchor) + pillars = 4800 :=
by
  sorry

end total_concrete_needed_l53_53519


namespace side_length_of_triangle_l53_53660

-- Definition of given information
def circle_radius (A : ℝ) := A = 100 * Real.pi
def OA (O A : ℝ) := O = 10 ∧ A = 5

-- Main statement
theorem side_length_of_triangle (O A : Point) (B C : Point)
  (A_conditions : OA O A)
  (radius_condition : circle_radius (π * 10^2))
  (O_outside_ABC : ¬ O ∈ Triangle A B C) :
  ∃ s : ℝ, s = 5 :=
by {
  sorry
}

end side_length_of_triangle_l53_53660


namespace find_b_eq_11_div_5_l53_53206

-- Definitions from the conditions
def f (x : ℝ) (b : ℝ) : ℝ := 1 / (2 * x + b)
def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (5 * x)

-- The problem statement
theorem find_b_eq_11_div_5 : ∀ (b : ℝ), (∀ x, f (f_inv x) b = x) → b = 11 / 5 :=
by
  intro b h
  have : ∀ x, x = 5 * x / (4 - 6 * x + 5 * b * x) := by sorry
  have h1 : 11 - 5 * b = 0 := by sorry
  have h2 : b = 11 / 5 := by sorry
  exact h2

end find_b_eq_11_div_5_l53_53206


namespace math_preference_related_to_gender_l53_53393

-- Definitions for conditions
def total_students : ℕ := 100
def male_students : ℕ := 55
def female_students : ℕ := total_students - male_students -- 45
def likes_math : ℕ := 40
def female_likes_math : ℕ := 20
def female_not_like_math : ℕ := female_students - female_likes_math -- 25
def male_likes_math : ℕ := likes_math - female_likes_math -- 20
def male_not_like_math : ℕ := male_students - male_likes_math -- 35

-- Calculate Chi-square
def chi_square (a b c d : ℕ) : Float :=
  let numerator := (total_students * (a * d - b * c)^2).toFloat
  let denominator := ((a + b) * (c + d) * (a + c) * (b + d)).toFloat
  numerator / denominator

def k_square : Float := chi_square 20 35 20 25 -- Calculate with given values

-- Prove the result
theorem math_preference_related_to_gender :
  k_square > 7.879 :=
by
  sorry

end math_preference_related_to_gender_l53_53393


namespace kylie_total_apples_l53_53143

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

end kylie_total_apples_l53_53143


namespace tiger_length_l53_53309

variables (speed : ℝ) (length : ℝ)
variables (time_pass_grass : ℝ := 1)
variables (distance_trunk : ℝ := 20) (time_trunk : ℝ := 5)

theorem tiger_length :
  (distance_trunk / time_trunk = speed) →
  (speed * time_pass_grass = length) →
  length = 4 :=
by
  intros hspeed hlength
  rw [hspeed, hlength]
  sorry

end tiger_length_l53_53309


namespace parents_age_when_mark_was_born_l53_53172

theorem parents_age_when_mark_was_born
    (mark_age : ℕ)
    (john_younger : ℕ)
    (parent_multiplier : ℕ)
    (john_age : ℕ := mark_age - john_younger)
    (parents_current_age : ℕ := parent_multiplier * john_age)
    (answer : ℕ := parents_current_age - mark_age) :
    mark_age = 18 → john_younger = 10 → parent_multiplier = 5 → answer = 22 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  refl

end parents_age_when_mark_was_born_l53_53172


namespace number_of_combinations_l53_53334

theorem number_of_combinations (n k : ℕ) (h₁ : n = 6) (h₂ : k = 4) :
  (nat.choose n k) * (nat.choose n k) * nat.factorial k = 5400 :=
by
  rw [h₁, h₂]
  simp only [nat.choose, nat.factorial]
  norm_num
  sorry

end number_of_combinations_l53_53334


namespace problem_statement_l53_53711

open Finset

variable {α : Type*}

def T : Finset α := {a, b, c, d, e, f}

noncomputable def count_subsets_with_intersection (T : Finset α) (k : ℕ) : ℕ :=
  (choose T.card k) * (2 ^ (T.card - k)) / 2

theorem problem_statement : count_subsets_with_intersection T 3 = 80 :=
by sorry

end problem_statement_l53_53711


namespace difference_even_odd_sets_l53_53945

def binom : ℕ → ℕ → ℕ
| n k := Nat.choose n k

def f (n m : ℕ) : ℤ :=
  if hn : ∃ a b, n = 2*a ∧ m = 2*b then
    let ⟨a, b, han, hbm⟩ := hn in (-1)^b * binom a b
  else if hn : ∃ a b, n = 2*a ∧ m = 2*b+1 then
    0
  else if hn : ∃ a b, n = 2*a+1 ∧ m = 2*b then
    let ⟨a, b, han, hbm⟩ := hn in (-1)^b * binom a b
  else if hn : ∃ a b, n = 2*a+1 ∧ m = 2*b+1 then
    let ⟨a, b, han, hbm⟩ := hn in (-1)^(b+1) * binom a b
  else 0

theorem difference_even_odd_sets (n m : ℕ) :
  f n m =
  if ∃ a b, n = 2*a ∧ m = 2*b then
    let ⟨a, b, _, _⟩ := Classical.indefiniteDescription _ (classical.some_spec _) in (-1:ℤ)^b * binom a b
  else if ∃ a b, n = 2*a ∧ m = 2*b+1 then
    0
  else if ∃ a b, n = 2*a+1 ∧ m = 2*b then
    let ⟨a, b, _, _⟩ := Classical.indefiniteDescription _ (classical.some_spec _) in (-1:ℤ)^b * binom a b
  else if ∃ a b, n = 2*a+1 ∧ m = 2*b+1 then
    let ⟨a, b, _, _⟩ := Classical.indefiniteDescription _ (classical.some_spec _) in (-1:ℤ)^(b+1) * binom a b
  else 0 :=
by sorry

end difference_even_odd_sets_l53_53945


namespace find_f_f_neg2_l53_53427

def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then 2^x - x - 1 else x^2 + 2*x

theorem find_f_f_neg2 : f (f (-2)) = 0 := 
by 
  sorry

end find_f_f_neg2_l53_53427


namespace inequality_holds_l53_53397

theorem inequality_holds (a b : ℝ) (h : a ≠ b) : a^4 + 6 * a^2 * b^2 + b^4 > 4 * a * b * (a^2 + b^2) := 
by
  sorry

end inequality_holds_l53_53397


namespace sum_youngest_oldest_l53_53219

-- Define the ages of the cousins
variables (a1 a2 a3 a4 : ℕ)

-- Conditions given in the problem
def mean_age (a1 a2 a3 a4 : ℕ) : Prop := (a1 + a2 + a3 + a4) / 4 = 8
def median_age (a2 a3 : ℕ) : Prop := (a2 + a3) / 2 = 5

-- Main theorem statement to be proved
theorem sum_youngest_oldest (h_mean : mean_age a1 a2 a3 a4) (h_median : median_age a2 a3) :
  a1 + a4 = 22 :=
sorry

end sum_youngest_oldest_l53_53219


namespace min_value_of_f_l53_53018

-- Given function:
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x)

-- The minimum value of the function:
theorem min_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -6480.25 :=
by
  have h : ∀ y : ℝ, f y = y^4 - 289 * y^2 + 14400 := sorry
  have h_min : ∀ y : ℝ, (y^4 - 289 * y^2 + 14400) = (y^2 - 144.5)^2 - 6480.25 := sorry
  use 0 -- Since we just need existence, setting x to 0 for the purpose of example
  intro y
  split
  . -- Prove f x ≤ f y (which should simplify to show min at y^2 = 144.5)
    sorry
  . -- Prove f x = -6480.25
    admit
  -- These are place-holders showing where each proof part would be

end min_value_of_f_l53_53018


namespace score_of_B_is_correct_l53_53110

theorem score_of_B_is_correct (A B C D E : ℝ)
  (h1 : (A + B + C + D + E) / 5 = 90)
  (h2 : (A + B + C) / 3 = 86)
  (h3 : (B + D + E) / 3 = 95) : 
  B = 93 := 
by 
  sorry

end score_of_B_is_correct_l53_53110


namespace median_parts_produced_is_15_l53_53554

-- This constant represents the number of parts produced by 10 workers
def parts_produced : List ℕ := [15, 17, 14, 10, 15, 19, 17, 16, 14, 12]

-- The Lean statement to prove that the median of parts_produced is 15.
theorem median_parts_produced_is_15 : List.median parts_produced = 15 := 
  sorry

end median_parts_produced_is_15_l53_53554


namespace quadratic_equation_a_ne_1_l53_53470

theorem quadratic_equation_a_ne_1 (a : ℝ) :
  (a-1) ≠ 0 :=
begin
  -- We know the given equation can be rewritten as (a-1)x^2 - x + 7 = 0
  -- This needs to satisfy the condition of being a quadratic equation
  -- Which means the coefficient of x^2 is non-zero
  sorry
end

end quadratic_equation_a_ne_1_l53_53470


namespace compute_expression_value_l53_53300

theorem compute_expression_value (a : ℝ) (h : a = 2) :
  (3 * a^(-2) + (a^(-2)) / 3) / (a^2) = 5 / 24 :=
by
  sorry

end compute_expression_value_l53_53300


namespace bob_wins_l53_53837

-- Define the notion of nim-sum used in nim-games
def nim_sum (a b : ℕ) : ℕ := Nat.xor a b

-- Define nim-values for given walls based on size
def nim_value : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| 7 => 2
| _ => 0

-- Calculate the nim-value of a given configuration
def nim_config (c : List ℕ) : ℕ :=
c.foldl (λ acc n => nim_sum acc (nim_value n)) 0

-- Prove that the configuration (7, 3, 1) gives a nim-value of 0
theorem bob_wins : nim_config [7, 3, 1] = 0 := by
  sorry

end bob_wins_l53_53837


namespace original_price_is_125_l53_53319

noncomputable def original_price (sold_price : ℝ) (discount_percent : ℝ) : ℝ :=
  sold_price / ((100 - discount_percent) / 100)

theorem original_price_is_125 : original_price 120 4 = 125 :=
by
  sorry

end original_price_is_125_l53_53319


namespace calculate_depth_of_well_l53_53014

noncomputable def depth_of_well (d : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := d / 2
  let V := total_cost / cost_per_cubic_meter
  V / (Real.pi * r^2)

theorem calculate_depth_of_well :
  depth_of_well 3 18 1781.28 ≈ 14.01 :=
by
  sorry

end calculate_depth_of_well_l53_53014


namespace no_solutions_for_q_l53_53745

theorem no_solutions_for_q : ∀ q : ℝ, ¬ (||| real.sin ((q^2) * | q - 5 |) - 10 | - 5 | = 2) :=
by
  intro q
  have A_def : A = real.sin ((q^2) * | q - 5 |) - 10 := sorry
  have layer2_abs : |A| = | real.sin ((q^2) * | q - 5 |) - 10 | := sorry
  have outer_abs : ||A| - 5| = 2 := sorry
  sorry

end no_solutions_for_q_l53_53745


namespace mice_without_coins_l53_53748

theorem mice_without_coins 
  (mice : ℕ) (coins : ℕ) (two_mice : ℕ) (two_coins_each : ℕ) (x y z : ℕ) 
  (total_mice : mice = 40) 
  (total_coins : coins = 40) 
  (two_mice_carrying : two_mice = 2) 
  (coin_count_2 : two_coins_each = 2) 
  (eq1 : 2 * two_mice_carrying + 7 * y + 4 * z = coins)
  (eq2 : two_mice_carrying + x + y + z = total_mice) : 
  x = 32 :=
by
  sorry

end mice_without_coins_l53_53748


namespace boxes_left_l53_53180

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l53_53180


namespace triangle_similarity_l53_53946

open EuclideanGeometry

-- Define the vertices of triangle ABC
variables {A B C A1 B1 C1 B2 C2 H M : Point}

-- Define triangle ABC
def triangle_ABC :=
  is_triangle A B C

-- Define altitudes AA1, BB1, CC1
def altitudes (A B C A1 B1 C1 : Point) :=
  is_orthogonal (line_through A A1) (line_through B C) ∧
  is_orthogonal (line_through B B1) (line_through A C) ∧
  is_orthogonal (line_through C C1) (line_through A B)

-- Define midpoints B2 and C2 of BB1 and CC1 respectively
def midpoints (B B1 B2 C C1 C2 : Point) :=
  midpoint B B1 B2 ∧
  midpoint C C1 C2

-- Define orthocenter H of triangle ABC
def orthocenter (A B C H : Point) :=
  orthocenter A B C H

-- Define midpoint M of side BC
def midpoint_BC (M B C : Point) :=
  midpoint B C M

-- The theorem statement: ∆A₁B₂C₂ ∼ ∆ABC
theorem triangle_similarity
  (h_triangle : triangle_ABC)
  (h_altitudes : altitudes A B C A1 B1 C1)
  (h_midpoints : midpoints B B1 B2 C C1 C2)
  (h_orthocenter : orthocenter A B C H)
  (h_midpoint_M : midpoint_BC M B C) :
  similar A1 B2 C2 A B C :=
sorry

end triangle_similarity_l53_53946


namespace fill_tank_time_l53_53686

theorem fill_tank_time :
  ∃ (t : ℚ),
    (∀ (x y : ℚ), 
      (1 / x - 1 / y = 2 / 15) →
      (1 / (x + 1) - 1 / y = 1 / 20) →
      1 / x + 1 / (x + 1) - 1 / y = 23 / 60) →
      t = 60 / 23 :=
begin
  sorry
end

end fill_tank_time_l53_53686


namespace kylie_total_apples_l53_53144

theorem kylie_total_apples : (let first_hour := 66 in 
                              let second_hour := 2 * 66 in 
                              let third_hour := 66 / 3 in 
                              first_hour + second_hour + third_hour = 220) :=
by
  let first_hour := 66
  let second_hour := 2 * first_hour
  let third_hour := first_hour / 3
  show first_hour + second_hour + third_hour = 220
  sorry

end kylie_total_apples_l53_53144


namespace sum_of_coefficients_equality_l53_53997

theorem sum_of_coefficients_equality 
  (a : Fin 13 → ℝ)
  (h : ∀ x : ℝ, (x^2 + 2 * x - 2) ^ 6 = ∑ i in Finset.range 13, a i * (x + 2)^i) :
  a 0 + a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 + 5 * a 5 + 6 * a 6 + 7 * a 7 + 8 * a 8 + 9 * a 9 + 10 * a 10 + 11 * a 11 + 12 * a 12 = 64 :=
by
  sorry

end sum_of_coefficients_equality_l53_53997


namespace final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l53_53681

-- Define the travel distances
def travel_distances : List ℤ := [17, -9, 7, 11, -15, -3]

-- Define the fuel consumption rate
def fuel_consumption_rate : ℝ := 0.08

-- Theorem stating the final position
theorem final_position_is_east_8km :
  List.sum travel_distances = 8 :=
by
  sorry

-- Theorem stating the total fuel consumption
theorem total_fuel_consumption_is_4_96liters :
  (List.sum (travel_distances.map fun x => |x| : List ℝ)) * fuel_consumption_rate = 4.96 :=
by
  sorry

end final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l53_53681


namespace triangle_PQR_area_is_4_l53_53507

-- Define the altitude PS of the triangle PQR
def ps_length : ℝ := 2

-- Define the angle PRQ of the triangle PQR
def angle_PRQ : ℝ := 60

-- Lean theorem statement to prove the area of triangle PQR
theorem triangle_PQR_area_is_4 :
  let ps : ℝ := ps_length
  let angle : ℝ := angle_PRQ
  let pr : ℝ := 2 * ps -- from 30-60-90 properties, PR = 2 * PS
  let area : ℝ := 1 / 2 * pr * ps
  in angle = 60 ∧ ps = 2 → area = 4 := 
by {
  sorry
}

end triangle_PQR_area_is_4_l53_53507


namespace problem_part1_problem_part2_l53_53116

open Complex

noncomputable def rotate (z : ℂ) (θ : ℝ) : ℂ :=
  z * exp (θ * I)

theorem problem_part1 :
  let z_A := √3 + I
  let θ := 2 * Real.pi / 3
  let z_C := rotate z_A θ
  z_C = -√3 + I := 
by
  sorry

theorem problem_part2 :
  let z_C := -√3 + I
  ∃ z : ℂ, (|z - z_C| = 1 ∧ (Complex.arg (z - z_C) - Complex.arg z_C) = 2 * Real.pi / 3)
    ∧ (z = -√3/2 + 3/2 * I ∨ z = -√3) :=
by
  sorry

end problem_part1_problem_part2_l53_53116


namespace parents_age_when_mark_was_born_l53_53171

theorem parents_age_when_mark_was_born
    (mark_age : ℕ)
    (john_younger : ℕ)
    (parent_multiplier : ℕ)
    (john_age : ℕ := mark_age - john_younger)
    (parents_current_age : ℕ := parent_multiplier * john_age)
    (answer : ℕ := parents_current_age - mark_age) :
    mark_age = 18 → john_younger = 10 → parent_multiplier = 5 → answer = 22 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  refl

end parents_age_when_mark_was_born_l53_53171


namespace period_f_max_min_values_of_f_l53_53795

noncomputable def f (x : ℝ) : ℝ := 4 * sin (x - π / 6) * cos x + 1

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
begin
  sorry
end

theorem max_min_values_of_f :
  let I := Icc (-π / 4) (π / 4) in
  ∃ (x_max x_min : ℝ), x_max ∈ I ∧ x_min ∈ I ∧
  (f x_max = sqrt 3 ∧ f x_min = -2) :=
begin
  sorry
end

end period_f_max_min_values_of_f_l53_53795


namespace problem_locus_equation_problem_range_of_k_l53_53052

noncomputable def locus_of_center (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem problem_locus_equation :
  ∀ x y : ℝ, locus_of_center x y ↔ (x^2 / 4 + y^2 / 3 = 1) :=
by sorry

theorem problem_range_of_k (k : ℝ) :
  (k = -(1 / 2) ∨ k = -(3 / 2)) ↔
  (∃ x y l : ℝ, (x^2 / 4 + y^2 / 3 = 1) ∧ (y = k * x + l) ∧
    let x1 := (-8 * k) / (3 + 4 * k^2) in
    let y1 := (6) / (3 + 4 * k^2) in
    (3 / (3 + 4 * k^2) = -(1 / k) * (-4 * k / (3 + 4 * k^2) - 1 / 8))) :=
by sorry

end problem_locus_equation_problem_range_of_k_l53_53052


namespace lyle_notebook_cost_l53_53671

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l53_53671


namespace rectangle_perimeter_91_l53_53315

theorem rectangle_perimeter_91 (AE BE CF : ℕ) (hAE : AE = 10) (hBE : BE = 20) (hCF : CF = 5) : 
  let AB := 20
  let AD := 25
  2 * (AB + AD) = 90 ∧ Nat.gcd 90 1 = 1 := 
by
  rw [hAE, hBE, hCF]
  apply And.intro
  calc
    2 * (20 + 25) = 2 * 45 := by norm_num
                ... = 90     := by norm_num

  show Nat.gcd 90 1 = 1
  sorry

end rectangle_perimeter_91_l53_53315


namespace arithmetic_sequence_S9_l53_53528

theorem arithmetic_sequence_S9 :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ},
  (∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 :=
by
  intro a S h_S h_a2 h_S4
  sorry

end arithmetic_sequence_S9_l53_53528


namespace total_cost_for_both_children_l53_53248

theorem total_cost_for_both_children
  (tuition_per_person : ℕ) (sibling_discount : ℕ) (ali_and_matt_signed_up : Bool)
  (ali_name : String) (matt_name : String) (ali_is_sibling_of_matt : Bool) :
  tuition_per_person = 45 → sibling_discount = 15 → ali_and_matt_signed_up = true →
  ali_name = "Ali" → matt_name = "Matt" → ali_is_sibling_of_matt = true →
  let ali_tuition := tuition_per_person
  let matt_tuition := tuition_per_person - sibling_discount
  let total_cost := ali_tuition + matt_tuition
  total_cost = 75 := 
by
  intros h1 h2 h3 h4 h5 h6
  unfold ali_tuition matt_tuition total_cost
  rw [h1, h2]
  sorry

end total_cost_for_both_children_l53_53248


namespace BK_parallel_AE_l53_53662

noncomputable def pentagon (A B C D E : Type) [Convex A B C D E] :=
{ 
  AE_parallel_CD : AE ∥ CD,
  AB_eq_BC : AB = BC,
  angle_bisectors_intersect_K : ∃ K, is_angle_bisector A K ∧ is_angle_bisector C K
}

theorem BK_parallel_AE (A B C D E : Type) [Convex A B C D E] 
  (h : pentagon A B C D E) : 
  ∃ K, BK ∥ AE := 
begin
  sorry
end

end BK_parallel_AE_l53_53662


namespace exists_n_for_last_2k_digits_l53_53981

theorem exists_n_for_last_2k_digits 
    (k : ℕ) (hk : 0 < k) 
    (a : Fin k → Fin 10) :
    ∃ (b : Fin k → Fin 10) (n : ℕ), 0 < n ∧
    (2^n % 10^(2*k) = (∑ i in Finset.range k, a ⟨i, sorry⟩ * 10^i) + (∑ i in Finset.range k, b ⟨i, sorry⟩ * 10^(k + i))) := sorry

end exists_n_for_last_2k_digits_l53_53981


namespace sale_in_third_month_l53_53665

theorem sale_in_third_month (sale1 sale2 sale4 sale5 sale6 : ℕ) (avg : ℕ) 
  (H1 : sale1 = 6435) (H2 : sale2 = 6927) (H4 : sale4 = 6562) (H5 : sale5 = 5591) 
  (H6 : sale6 = 5591) (Havg : avg = 6600) 
  : ∃ sale3 : ℕ, sale3 = 14085 :=
by
  let total_sales := avg * 6
  let known_sales := sale1 + sale2 + sale4 + sale5 + sale6
  let sale3 := total_sales - known_sales
  have Htotal : total_sales = 6600 * 6 := by rw Havg
  have Htotal_value : total_sales = 39600 := by norm_num [Htotal]
  have Hknown : known_sales = 6435 + 6927 + 6562 + 5591 + 5591 := by rw [H1, H2, H4, H5, H6]
  have Hknown_value : known_sales = 25515 := by norm_num [Hknown]
  have Hsale3 : sale3 = total_sales - known_sales := rfl
  have Hsale3_value : sale3 = 14085 := by norm_num [Htotal_value, Hknown_value, Hsale3]
  exact ⟨sale3, Hsale3_value⟩

end sale_in_third_month_l53_53665


namespace ellipse_general_eqn_rho_values_l53_53365

noncomputable def curve_parametric_to_ellipse (a b : ℝ) (phi : ℝ) :=
  ∃ (x y : ℝ), x = a * cos phi ∧ y = b * sin phi ∧ a > b ∧ b > 0

theorem ellipse_general_eqn (a b : ℝ) (phi : ℝ) :
  curve_parametric_to_ellipse a b phi → (a = 4) → (b = 2) → (M : ℝ × ℝ)
  (M = (2, sqrt 3)) (phi = pi / 3) →
  ∃ x y, (x = 2) ∧ (y = sqrt 3) →
  ((x² / 16) + (y² / 4) = 1) := 
sorry

theorem rho_values (a b : ℝ) (theta : ℝ) :
  (curve_parametric_to_ellipse a b theta) →
  ∀ ρ1 ρ2, 
  (∃ x y, x = ρ1 * cos theta ∧ y = ρ1 * sin theta ∧ 
          (-ρ2 * sin theta) = x ∧ (ρ2 * cos theta) = y) →
  (a = 4) → (b = 2) →
  (M = (2, sqrt 3)) (phi = pi / 3) →
  (1 / (ρ1^2) + 1 / (ρ2^2) = 5 / 16) :=
sorry  

end ellipse_general_eqn_rho_values_l53_53365


namespace fourth_power_of_cube_third_smallest_prime_l53_53292

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53292


namespace exists_geometric_weak_arithmetic_l53_53401

theorem exists_geometric_weak_arithmetic (m : ℕ) (hm : 3 ≤ m) :
  ∃ (k : ℕ) (a : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ m → a i = k^(m - i)*(k + 1)^(i - 1)) ∧
    ((∀ i, 1 ≤ i → i < m → a i < a (i + 1)) ∧ 
    ∃ (x : ℕ → ℕ) (d : ℕ), 
      (x 0 ≤ a 1 ∧ 
      ∀ i, 1 ≤ i → i < m → (x i ≤ a (i + 1) ∧ a (i + 1) < x (i + 1)) ∧ 
      ∀ i, 0 ≤ i → i < m - 1 → x (i + 1) - x i = d)) :=
by
  sorry

end exists_geometric_weak_arithmetic_l53_53401


namespace full_tank_capacity_l53_53696

theorem full_tank_capacity (speed : ℝ) (gas_usage_per_mile : ℝ) (time : ℝ) (gas_used_fraction : ℝ) (distance_per_tank : ℝ) (gallons_used : ℝ)
  (h1 : speed = 50)
  (h2 : gas_usage_per_mile = 1 / 30)
  (h3 : time = 5)
  (h4 : gas_used_fraction = 0.8333333333333334)
  (h5 : distance_per_tank = speed * time)
  (h6 : gallons_used = distance_per_tank * gas_usage_per_mile)
  (h7 : gallons_used = 0.8333333333333334 * 10) :
  distance_per_tank / 30 / 0.8333333333333334 = 10 :=
by sorry

end full_tank_capacity_l53_53696


namespace perimeter_of_triangle_l53_53509

def right_triangle (A B C : Type) (angle : A) : Prop := sorry
def foot_of_altitude (X W Y Z : Type) : Prop := sorry
def distance (P Q : Type) : ℝ := sorry

theorem perimeter_of_triangle {X Y Z W : Type} 
  (h1: right_triangle X Y Z)
  (h2: foot_of_altitude X W Y Z)
  (h3: distance Y W = 5)
  (h4: distance W Z = 12) :
  distance X Y + distance Y Z + distance X Z = 17 + 4 * Real.sqrt 15 :=
sorry

end perimeter_of_triangle_l53_53509


namespace calculate_square_add_subtract_l53_53243

theorem calculate_square_add_subtract (a b : ℤ) :
  (41 : ℤ)^2 = (40 : ℤ)^2 + 81 ∧ (39 : ℤ)^2 = (40 : ℤ)^2 - 79 :=
by
  sorry

end calculate_square_add_subtract_l53_53243


namespace complex_quadrant_is_first_l53_53924

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53924


namespace sqrt_sum_distances_le_sqrt_circum_rad_plus_five_inscribed_rad_l53_53147

variables {A B C P : Type} [InnerProductSpace ℝ P]
variables (a b c : ℝ) (d_a d_b d_c : ℝ)
variables {ABC_circumradius ABC_inradius : ℝ}

-- Conditions
-- Let \(P\) be a point inside triangle \(ABC\).
-- \(d_a\), \(d_b\), and \(d_c\) are distances from \(P\) to the sides \(BC\), \(AC\), and \(AB\) respectively.
-- \(R\) is the radius of the circumcircle of \(\Delta ABC\).
-- \(r\) is the radius of the inscribed circle of \(\Delta ABC\).

theorem sqrt_sum_distances_le_sqrt_circum_rad_plus_five_inscribed_rad 
    (h : PointInTriangle P ABC)
    (hd_a : Distance P (LineThrough B C) = d_a)
    (hd_b : Distance P (LineThrough A C) = d_b)
    (hd_c : Distance P (LineThrough A B) = d_c)
    (hR : CircumRadius ABC = R)
    (hr : InRadius ABC = r) :
    sqrt d_a + sqrt d_b + sqrt d_c ≤ sqrt (2 * R + 5 * r) :=
  sorry

end sqrt_sum_distances_le_sqrt_circum_rad_plus_five_inscribed_rad_l53_53147


namespace hyperbola_eccentricity_l53_53209

variable (a b c e : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0)
variable (h1 : (a > 0 ∧ b > 0) → (sqrt (a^2 + b^2) = sqrt 2))

theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1 → (a > 0 ∧ b > 0) →
  (∀ x₁ y₁, (x₁ - sqrt 2)^2 + y₁^2 = 1 → (sqrt 2 * b = sqrt (a^2 + b^2)) )) → 
  e = sqrt 2) :=
sorry

end hyperbola_eccentricity_l53_53209


namespace edwards_initial_money_l53_53364

variable (spent1 spent2 current remaining : ℕ)

def initial_money (spent1 spent2 current remaining : ℕ) : ℕ :=
  spent1 + spent2 + current

theorem edwards_initial_money :
  spent1 = 9 → spent2 = 8 → remaining = 17 →
  initial_money spent1 spent2 remaining remaining = 34 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end edwards_initial_money_l53_53364


namespace polynomial_square_of_binomial_l53_53363

theorem polynomial_square_of_binomial (c : ℝ) (h : c = 25) : ∃ d : ℝ, (16 * x^2 - 40 * x + c) = (4 * x + d) ^ 2 :=
by
  use -5
  rw h
  sorry

end polynomial_square_of_binomial_l53_53363


namespace fourth_power_of_third_smallest_prime_cube_l53_53277

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53277


namespace sum_a1_a3_a5_l53_53218

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 16 ∧ ∀ k, a (k + 1) = 3 * a k / 2

theorem sum_a1_a3_a5 :
  ∃ a : ℕ → ℕ, sequence a ∧ a 1 + a 3 + a 5 = 133 :=
by
  sorry

end sum_a1_a3_a5_l53_53218


namespace percent_of_absent_students_l53_53566

noncomputable def absent_percentage : ℚ :=
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := boys * (1/5 : ℚ)
  let absent_girls := girls * (1/4 : ℚ)
  let total_absent := absent_boys + absent_girls
  (total_absent / total_students) * 100

theorem percent_of_absent_students : absent_percentage = 22.5 := sorry

end percent_of_absent_students_l53_53566


namespace triangle_perimeter_is_26_l53_53787

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l53_53787


namespace thirty_one_star_thirty_two_l53_53123

def complex_op (x y : ℝ) : ℝ :=
sorry

axiom op_zero (x : ℝ) : complex_op x 0 = 1

axiom op_associative (x y z : ℝ) : complex_op (complex_op x y) z = z * (x * y) + z

theorem thirty_one_star_thirty_two : complex_op 31 32 = 993 :=
by
  sorry

end thirty_one_star_thirty_two_l53_53123


namespace perimeter_PQRS_l53_53496

-- Definition of the problem conditions
variables (P Q R S : Point)
variables (PQ QR RS PR : ℝ)
variables (angleQ_right : angle P Q R = 90)
variables (PR_RS_perpendicular : ∠ PRS = 90)
variables (PQ_eq : PQ = 15)
variables (QR_eq : QR = 20)
variables (RS_eq : RS = 9)

-- The goal statement
theorem perimeter_PQRS :
  PQ + QR + RS + sqrt (PQ^2 + QR^2) + sqrt ((PQ^2 + QR^2) + RS^2) = 44 + sqrt 706 :=
sorry

end perimeter_PQRS_l53_53496


namespace complex_point_quadrant_l53_53872

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53872


namespace number_of_solutions_eq_2_pow_2005_l53_53760

def f₁ (x : ℝ) : ℝ := abs (1 - 2 * x)

def f_n (n : ℕ) : ℝ → ℝ
| 0       := id
| (n + 1) := f₁ ∘ f_n n

theorem number_of_solutions_eq_2_pow_2005 : set.finite { x ∈ set.Icc 0 1 | f_n 2005 x = (1 / 2) * x } ∧ 
                                             fintype.card { x ∈ set.Icc 0 1 | f_n 2005 x = (1 / 2) * x } = 2 ^ 2005 :=
sorry

end number_of_solutions_eq_2_pow_2005_l53_53760


namespace right_triangle_l53_53506

-- Definition of the problem conditions
def Triangle (A B C : Type) := ∃ (T : Type), let angle_BCA := 3 * (30 : ℝ) in 
  (T ∈ [A, B]) ∧
  CT_perp (A, B) ∧
  CF_isMedian (A, B) ∧
  trisectAngle (B, C, A, T, F, angle_BCA / 3)

-- Main theorem statement
theorem right_triangle (A B C : Type) (h : Triangle A B C) : angle A B C = 90 :=
sorry

end right_triangle_l53_53506


namespace gross_profit_percentage_l53_53749

theorem gross_profit_percentage :
  ∀ (selling_price wholesale_cost : ℝ),
  selling_price = 28 →
  wholesale_cost = 24.14 →
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 15.99 :=
by
  intros selling_price wholesale_cost h1 h2
  rw [h1, h2]
  norm_num
  sorry

end gross_profit_percentage_l53_53749


namespace problem_l53_53404

-- Define the sequence a_n with given conditions
def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2 * n + 1

-- Sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), a_seq i

-- Defines b_n = (-1)^n + 2^(a_n)
def b_seq (n : ℕ) : ℤ :=
  (-1 : ℤ)^n + 2^(a_seq n)

-- Define the sum of the first n terms of sequence b_n
def T (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), b_seq i

-- Given conditions and the derived formulas
theorem problem (n : ℕ) : 
  (a_seq n = 2 * n + 1) ∧ 
  (T n = ( (-1)^n - 1) / 2 + 8 * (4^n - 1) / 3) := 
  by
    sorry

end problem_l53_53404


namespace distance_between_points_l53_53584

theorem distance_between_points (A B : ℝ × ℝ × ℝ) 
  (hA : A = (2, 5, 4)) (hB : B = (-2, 3, 5)) : 
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2) = real.sqrt 21 :=
by
  rw [hA, hB]
  -- Further steps would use algebraic manipulations to conclude the proof.
  sorry

end distance_between_points_l53_53584


namespace derivative_at_zero_l53_53647

def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem derivative_at_zero : (Real.deriv f 0) = 2 :=
by 
  sorry

end derivative_at_zero_l53_53647


namespace sqrt_40_simplified_l53_53200

theorem sqrt_40_simplified : Real.sqrt 40 = 2 * Real.sqrt 10 := 
by
  sorry

end sqrt_40_simplified_l53_53200


namespace set_inclusion_l53_53073

def setM : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}

def setN : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}

def setP : Set ℝ := {a | ∃ k : ℤ, a = (k * Real.pi / 2) + (Real.pi / 4)}

theorem set_inclusion : setP ⊆ setN ∧ setN ⊆ setM := by
  sorry

end set_inclusion_l53_53073


namespace area_of_S_l53_53541

def four_presentable (z : ℂ) : Prop := ∃ (w : ℂ), |w| = 4 ∧ z = w - 1/w

def S : Set ℂ := {z : ℂ | four_presentable z}

theorem area_of_S : 
  let A := {p : ℂ | ∃ (w : ℂ), |w| = 4 ∧ p = w - 1/w}
  area_inside (convex_hull A) = (255 / 16) * real.pi := 
sorry

end area_of_S_l53_53541


namespace complex_quadrant_is_first_l53_53922

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53922


namespace find_x_value_l53_53077

-- Define vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the condition that a + b is parallel to 2a - b
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1 - b.1) = k * (a.1 + b.1) ∧ (2 * a.2 - b.2) = k * (a.2 + b.2)

-- Problem statement: Prove that x = -4
theorem find_x_value : ∀ (x : ℝ),
  parallel_vectors vector_a (vector_b x) → x = -4 :=
by
  sorry

end find_x_value_l53_53077


namespace part_one_solution_set_part_two_range_for_m_l53_53068

-- Define the function f
def f (x : ℝ) : ℝ := abs (x + 2) - abs (2 * x - 1)

-- Part (1): Prove the solution set for f(x) > -5 is (-2, 8)
theorem part_one_solution_set : { x : ℝ | f x > -5 } = set.Ioo (-2 : ℝ) 8 := sorry

-- Part (2): Prove the range of values for m
theorem part_two_range_for_m (a b : ℝ) (h : a ≠ 0) :
  (abs (b + 2 * a) - abs (2 * b - a) ≥ abs a * (abs (x + 1) + abs (x - m)) →
   m ∈ set.Icc (-7 / 2 : ℝ) (3 / 2 : ℝ)) := sorry

end part_one_solution_set_part_two_range_for_m_l53_53068


namespace how_many_years_l53_53712

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10;
  digits = digits.reverse

def four_digit_palindrome (n : ℕ) : Prop :=
  1200 ≤ n ∧ n < 1300 ∧ is_palindrome n

def two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ is_palindrome n

def three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n

theorem how_many_years : 
  (finset.filter (λ n, ∃ a b, two_digit_palindrome a ∧ three_digit_palindrome b ∧ a * b = n) 
  (finset.range (1300 - 1200) + 1200)).card = 0 :=
by {
  sorry
}

end how_many_years_l53_53712


namespace smallest_x_of_quadratic_eqn_l53_53571

theorem smallest_x_of_quadratic_eqn : ∃ x : ℝ, (12*x^2 - 44*x + 40 = 0) ∧ x = 5 / 3 :=
by
  sorry

end smallest_x_of_quadratic_eqn_l53_53571


namespace point_in_first_quadrant_l53_53882

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53882


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53285

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53285


namespace digit_6_count_range_100_to_999_l53_53962

/-- The number of times the digit 6 is written in the integers from 100 through 999 inclusive is 280. -/
theorem digit_6_count_range_100_to_999 : 
  (∑ n in finset.Icc 100 999, (if digit 6 n then 1 else 0)) = 280 := 
sorry

end digit_6_count_range_100_to_999_l53_53962


namespace portion_filled_in_20_minutes_l53_53326

-- We define the time it takes to fill a portion of the cistern.
def fill_time : ℕ := 20

-- We define the portion of the cistern that is filled in the given time.
def portion_filled (t : ℕ) : ℝ := if t = fill_time then 1 else 0

-- We state that the fill pipe fills exactly 1 portion of the cistern in 20 minutes.
theorem portion_filled_in_20_minutes : portion_filled 20 = 1 :=
by strict sorry

end portion_filled_in_20_minutes_l53_53326


namespace rectangle_perimeter_l53_53117

theorem rectangle_perimeter (s : ℕ) (ABCD_area : 4 * s * s = 400) :
  2 * (2 * s + 2 * s) = 80 :=
by
  -- Skipping the proof
  sorry

end rectangle_perimeter_l53_53117


namespace permute_columns_l53_53255

theorem permute_columns (m n : ℕ) (table : list (list ℕ)) (cond1 : ∀ i, length (table i) = m) 
  (cond2 : ∀ k ∈ range (n + 1), count k (join table) = m) :
  ∃ permuted_table : list (list ℕ), 
    (∀ i, permuted_table i ~ table i) ∧
    (∀ j ∈ range (n + 1), ∃ row ∈ permuted_table, j ∈ set_of row ∧ 
      ∀ k ∈ range (n + 1) \ {j}, j ≠ k ∧ k ∈ set_of row) :=
sorry

end permute_columns_l53_53255


namespace complex_point_quadrant_l53_53874

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53874


namespace initial_kola_volume_l53_53320

theorem initial_kola_volume (V : ℝ) (S : ℝ) :
  S = 0.14 * V →
  (S + 3.2) / (V + 20) = 0.14111111111111112 →
  V = 340 :=
by
  intro h_S h_equation
  sorry

end initial_kola_volume_l53_53320


namespace exists_n_sum_partition_l53_53128

theorem exists_n_sum_partition :
  ∃ n : ℕ, ∃ g : fintype (fin 10), 
    ∃ f : fin n → fin 10, 
      ∑ i : fin n, (λ j : fin 10, if f i = j then i.val ^ 10 else 0) j = 
      ∑ i : fin n, (λ j : fin 10, if i < n then i.val ^ 10 else 0) j / 10 :=
by
  sorry

end exists_n_sum_partition_l53_53128


namespace smallest_period_f_l53_53165

def f (x : ℝ) : ℝ := 2 * sqrt 3 * (Real.sin x) * (Real.cos x) - 2 * (Real.sin x)^2 + 1

theorem smallest_period_f : ∃ T > 0, (∀ x ∈ ℝ, f (x + T) = f x) ∧ T = π := 
by
  sorry

end smallest_period_f_l53_53165


namespace equation_of_circle_C_equation_of_ellipse_E_min_value_PQ_PR_l53_53338

-- Define point C, and the line passing through it for any m in ℝ
def C := (2 : ℝ, 0 : ℝ)
def line_through_C (m : ℝ) : Prop := (m + 1) * (C.1) + 2 * (C.2) - 2 * m - 2 = 0

-- Equation of the circle centered at C with radius 2
def equation_of_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Symmetric point C' with respect to the y-axis
def C_prime := (-2 : ℝ, 0 : ℝ)

-- Define the ellipse equation E based on its properties
def equation_of_curve_E (x y : ℝ) : Prop := (x^2) / 8 + (y^2) / 4 = 1

-- Define the minimum value problem related to the tangents PQ and PR drawn from point P on E
def minimum_PQ_PR (P : ℝ × ℝ) : Prop :=
  let PC := (P.1 - 2)^2 + P.2^2 in
  let min_value := 8 * ℝ.sqrt 2 - 12 in
  ∀ (Q R : ℝ × ℝ), tangent_from_point_to_circle P Q R → 
  (PC - 4) * (1 - 2 * (2 / PC)) ≥ min_value

-- Theorem statements to be proved
theorem equation_of_circle_C :
  ∀ (x y : ℝ), line_through_C (m : ℝ) → equation_of_circle x y := sorry

theorem equation_of_ellipse_E :
  ∀ (M : ℝ × ℝ), ∃ (x y : ℝ), equation_of_curve_E x y := sorry

theorem min_value_PQ_PR :
  ∃ (P : ℝ × ℝ), minimum_PQ_PR P := sorry

end equation_of_circle_C_equation_of_ellipse_E_min_value_PQ_PR_l53_53338


namespace problem1_problem2_l53_53532

-- Problem (1): Prove that f(1) = 0
theorem problem1 (f: ℝ → ℝ) (h1: f 2 = 1) (h2: ∀ x y > 0, f (x * y) = f x + f y) (h3: ∀ x y, x < y → f x > f y) :
  f 1 = 0 :=
sorry

-- Problem (2): Prove the range of x
theorem problem2 (f: ℝ → ℝ) (h1: f 2 = 1) (h2: ∀ x y > 0, f (x * y) = f x + f y) (h3: ∀ x y, x < y → f x > f y) (h4: ∀ x > 0, f x + f (x - 3) ≥ 2) :
  { x : ℝ | 3 < x ∧ x ≤ 4 } =
  { x : ℝ | h4 x } :=
sorry

end problem1_problem2_l53_53532


namespace complex_quadrant_is_first_l53_53921

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53921


namespace polynomial_parabola_intersection_largest_x_l53_53735

theorem polynomial_parabola_intersection_largest_x 
  (d e f g : ℝ) 
  (h_eq : ∀ x, x^6 - 5 * x^5 + 5 * x^4 + 5 * x^3 + d * x^2 = e * x^2 + f * x + g)
  (h_intersects : exists a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∀ x, (x - a)^2 * (x - b)^2 * (x - c)^2 = x^6 - 5 * x^5 + 5 * x^4 + 5 * x^3 + (d - e) * x^2 - f * x - g) :
  3 ∈ ({a, b, c} : set ℝ) :=
by sorry

end polynomial_parabola_intersection_largest_x_l53_53735


namespace ratio_of_greatest_to_smallest_dist_l53_53194

noncomputable def greatest_ratio_least (P : Finset (EuclideanSpace ℝ (Fin 2))) : Prop :=
  P.card = 6 →
  ∃ (M m : ℝ), 
    M = Sup (set.image (λ (p : (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2))), dist p.1 p.2) 
                        ((P.product P).filter (λ (p : (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2))), p.1 ≠ p.2))) ∧
    m = Inf (set.image (λ (p : (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2))), dist p.1 p.2) 
                        ((P.product P).filter (λ (p : (EuclideanSpace ℝ (Fin 2)) × (EuclideanSpace ℝ (Fin 2))), p.1 ≠ p.2))) ∧
    M ≥ sqrt 3 * m

theorem ratio_of_greatest_to_smallest_dist {P : Finset (EuclideanSpace ℝ (Fin 2))} : 
  greatest_ratio_least P :=
by 
  sorry

end ratio_of_greatest_to_smallest_dist_l53_53194


namespace solution_of_system_l53_53763

theorem solution_of_system 
  (k : ℝ) (x y : ℝ)
  (h1 : (1 : ℝ) = 2 * 1 - 1)
  (h2 : (1 : ℝ) = k * 1)
  (h3 : k ≠ 0)
  (h4 : 2 * x - y = 1)
  (h5 : k * x - y = 0) : 
  x = 1 ∧ y = 1 :=
by
  sorry

end solution_of_system_l53_53763


namespace vector_dot_product_zero_l53_53414

noncomputable def vector_a : ℝ := sorry
noncomputable def vector_b : ℝ := sorry

def dot_product (u v : ℝ) : ℝ :=
  |u| * |v| * Real.cos (120 * Real.pi / 180)

theorem vector_dot_product_zero :
  let a := vector_a
  let b := vector_b
  (dot_product b (2 * a + b) = 0) :=
by
  sorry

end vector_dot_product_zero_l53_53414


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53260

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53260


namespace choose_four_socks_at_least_one_blue_l53_53202

-- There are six socks, each of different colors: {blue, brown, black, red, purple, green}
def socks : Finset String := {"blue", "brown", "black", "red", "purple", "green"}

-- We need to choose 4 socks such that at least one is blue.
def num_ways_to_choose_four_with_one_blue : ℕ :=
  (socks.erase "blue").card.choose 3

theorem choose_four_socks_at_least_one_blue :
  num_ways_to_choose_four_with_one_blue = 10 :=
by
  rw [num_ways_to_choose_four_with_one_blue, Finset.card_erase_of_mem]
  { rw Finset.card, decide, sorry }
  { exact Finset.mem_univ "blue" }
  sorry

end choose_four_socks_at_least_one_blue_l53_53202


namespace smallest_five_digit_palindrome_divisible_by_3_l53_53297

/-- The smallest five-digit palindrome that is divisible by 3 is 10001. -/
theorem smallest_five_digit_palindrome_divisible_by_3 : ∃ n : ℕ, (is_palindrome n ∧ n ≥ 10000 ∧ n ≤ 99999 ∧ n % 3 = 0 ∧ (∀ m, is_palindrome m ∧ m ≥ 10000 ∧ m ≤ 99999 ∧ m % 3 = 0 → n ≤ m) ∧ n = 10001) :=
sorry

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s == s.reverse

end smallest_five_digit_palindrome_divisible_by_3_l53_53297


namespace maximum_volume_of_frustum_l53_53327

noncomputable def sphere_radius := 5
noncomputable def frustum_top_radius := 3
noncomputable def frustum_bottom_radius := 4

theorem maximum_volume_of_frustum :
  let height := frustum_top_radius + frustum_bottom_radius in
  let volume := (1 : ℝ)/3 * height * (π * frustum_top_radius^2 + π * frustum_top_radius * frustum_bottom_radius + π * frustum_bottom_radius^2) in
  volume = (259 : ℝ)/3 * π :=
by
  let height := frustum_top_radius + frustum_bottom_radius
  let volume := (1 : ℝ)/3 * height * (π * frustum_top_radius^2 + π * frustum_top_radius * frustum_bottom_radius + π * frustum_bottom_radius^2)
  sorry

end maximum_volume_of_frustum_l53_53327


namespace complex_point_quadrant_l53_53871

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53871


namespace houses_in_block_l53_53332

theorem houses_in_block (total_mails_per_block : ℕ) (mails_per_house : ℕ) (h_total_mails : total_mails_per_block = 32) (h_mails_per_house : mails_per_house = 8) : 
  total_mails_per_block / mails_per_house = 4 :=
by
  rw [h_total_mails, h_mails_per_house]
  norm_num
  sorry

end houses_in_block_l53_53332


namespace total_balloons_170_l53_53178

variable (minutes_Max : ℕ) (rate_Max : ℕ) (minutes_Zach : ℕ) (rate_Zach : ℕ) (popped : ℕ)

def balloons_filled_Max := minutes_Max * rate_Max
def balloons_filled_Zach := minutes_Zach * rate_Zach
def total_filled_balloons := balloons_filled_Max + balloons_filled_Zach - popped

theorem total_balloons_170 
  (h1 : minutes_Max = 30) 
  (h2 : rate_Max = 2) 
  (h3 : minutes_Zach = 40) 
  (h4 : rate_Zach = 3) 
  (h5 : popped = 10) : 
  total_filled_balloons minutes_Max rate_Max minutes_Zach rate_Zach popped = 170 := by
  unfold total_filled_balloons
  unfold balloons_filled_Max
  unfold balloons_filled_Zach
  sorry

end total_balloons_170_l53_53178


namespace complex_real_part_of_product_l53_53469

theorem complex_real_part_of_product (z1 z2 : ℂ) (i : ℂ) 
  (hz1 : z1 = 4 + 29 * Complex.I)
  (hz2 : z2 = 6 + 9 * Complex.I)
  (hi : i = Complex.I) : 
  ((z1 - z2) * i).re = 20 := 
by
  sorry

end complex_real_part_of_product_l53_53469


namespace product_of_xy_l53_53464

-- Define the problem conditions
variables (x y : ℝ)
-- Define the condition that |x-3| and |y+1| are opposite numbers
def opposite_abs_values := |x - 3| = - |y + 1|

-- State the theorem
theorem product_of_xy (h : opposite_abs_values x y) : x * y = -3 :=
sorry -- Proof is omitted

end product_of_xy_l53_53464


namespace complex_point_in_first_quadrant_l53_53907

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53907


namespace complex_quadrant_l53_53935

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53935


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53271

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53271


namespace targets_breaking_orders_l53_53840

-- Define the conditions and the question in Lean
def nine_targets_in_columns : Prop :=
  ∀ (columns : List (List α)), 
    columns.length = 3 ∧ 
    ∀ col : List α, col ∈ columns → col.length = 3

def valid_target_sequence (seq : List char) : Prop :=
  seq = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

theorem targets_breaking_orders (columns : List (List α)) (seq : List char) 
  (h_column_structure : nine_targets_in_columns columns) 
  (h_valid_sequence : valid_target_sequence seq) : 
  seq.permutations.length = 1680 :=
by 
  sorry

end targets_breaking_orders_l53_53840


namespace parabola_translation_l53_53247

theorem parabola_translation :
  ∀ x : ℝ, (x^2 + 3) = ((x + 1)^2 + 3) :=
by
  skip -- proof is not needed; this is just the statement according to the instruction
  sorry

end parabola_translation_l53_53247


namespace proof_problem_l53_53429

-- We start by defining the function f with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4 ^ x + 1)

-- Condition: f(1) = -3/10
def condition_a : Prop :=
  ∃ a : ℝ, f(a, 1) = -3 / 10

-- Question 1: Prove that f is odd
def is_odd_function (a : ℝ) : Prop :=
  ∀ x : ℝ, f(a, -x) = -f(a, x)

-- Question 2: If -1/6 ≤ f(x) ≤ 0, then 0 ≤ x ≤ 1/2
def range_x (a : ℝ) : Prop :=
  (∀ x : ℝ, -1/6 ≤ f(a, x) ∧ f(a, x) ≤ 0) → (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2)

-- The combined proof problem
theorem proof_problem :
  condition_a →
  (∃ a : ℝ, is_odd_function(a) ∧ range_x(a)) :=
by
  sorry

end proof_problem_l53_53429


namespace complex_quadrant_check_l53_53915

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53915


namespace sum_S9_l53_53784

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: a is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a₄ + a₆ = -6
axiom a4_a6_condition : a 4 + a 6 = -6

-- Sum Sₙ of the first n terms
def sum_of_first_n_terms (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a (i + 1)

-- Assume a₁ + a₉ = -6
axiom a1_a9_condition : a 1 + a 9 = -6

-- Definition of Sₙ for an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Goal: Prove S₉ = -27
theorem sum_S9 : sum_of_arithmetic_sequence a 9 = -27 :=
  by
    -- We provide the proof next
    sorry

end sum_S9_l53_53784


namespace p_implies_q_l53_53422

variables {a b c d : ℝ}

def cubic (x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

def p : Prop := ∀ x1 x2 : ℝ, x1 < x2 → cubic' x1 < cubic' x2 ∨ cubic' x1 > cubic' x2
def q : Prop := ∃ x : ℝ, cubic x = 0 ∧ ∀ y : ℝ, cubic y = 0 → x = y

theorem p_implies_q : p → q :=
sorry

end p_implies_q_l53_53422


namespace problem_statement_l53_53782
  
noncomputable def t : ℝ :=
  classical.some (exists_root (λ x, x^3 - 7 * x^2 + 1) (by iterate 3 { apply continuous.pow; continuity }))

theorem problem_statement : (t ^ 20100) % 7 = 6 :=
  sorry

end problem_statement_l53_53782


namespace complex_point_in_first_quadrant_l53_53904

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53904


namespace lattice_points_count_l53_53499

theorem lattice_points_count :
  (∃ n : ℕ, lattice_points n) := 
begin
  sorry
end

def lattice_points (n : ℕ) : Prop :=
  n = 2551 ∧ 
  ∀ (x y : ℕ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) →
  ∃ (k l : ℤ), ((k, l) = (x, y)) 

end lattice_points_count_l53_53499


namespace circle_center_radius_sum_l53_53355

def circle_eq (x y : ℝ) : Prop := x^2 - 16 * x + y^2 + 6 * y = -75

noncomputable def center_x := 8
noncomputable def center_y := -3
noncomputable def radius := Real.sqrt 2

theorem circle_center_radius_sum :
  (a b : ℝ) (r : ℝ),
  (∀ x y, circle_eq x y) →
  a = center_x →
  b = center_y →
  r = radius →
  a + b + r = 5 + Real.sqrt 2 :=
by
  simp [center_x, center_y, radius]
  sorry

end circle_center_radius_sum_l53_53355


namespace cube_volume_fourth_power_l53_53620

theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 :=
sorry

end cube_volume_fourth_power_l53_53620


namespace first_plane_passengers_l53_53241

-- Definitions and conditions
def speed_plane_empty : ℕ := 600
def slowdown_per_passenger : ℕ := 2
def second_plane_passengers : ℕ := 60
def third_plane_passengers : ℕ := 40
def average_speed : ℕ := 500

-- Definition of the speed of a plane given number of passengers
def speed (passengers : ℕ) : ℕ := speed_plane_empty - slowdown_per_passenger * passengers

-- The problem statement rewritten in Lean 4
theorem first_plane_passengers (P : ℕ) (h_avg : (speed P + speed second_plane_passengers + speed third_plane_passengers) / 3 = average_speed) : P = 50 :=
sorry

end first_plane_passengers_l53_53241


namespace operation_hash_12_6_l53_53388

axiom operation_hash (r s : ℝ) : ℝ

-- Conditions
axiom condition_1 : ∀ r : ℝ, operation_hash r 0 = r
axiom condition_2 : ∀ r s : ℝ, operation_hash r s = operation_hash s r
axiom condition_3 : ∀ r s : ℝ, operation_hash (r + 2) s = (operation_hash r s) + 2 * s + 2

-- Proof statement
theorem operation_hash_12_6 : operation_hash 12 6 = 168 :=
by
  sorry

end operation_hash_12_6_l53_53388


namespace complex_point_in_first_quadrant_l53_53852

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53852


namespace harold_scores_correct_l53_53802

-- Define Harold's known quiz scores
def harold_initial_scores := [92, 84, 76, 80]

-- Define the mean score for all seven quizzes
def mean_score : ℝ := 83

-- Define the total score for seven quizzes based on the mean
def total_score : ℕ := 7 * 83

-- Define the score for the last three quizzes
def last_three_score : ℕ := total_score - (92 + 84 + 76 + 80)

-- List of possible additional scores
def additional_scores : list (ℕ × ℕ × ℕ) := [
  (94, 85, 70),
  (93, 86, 70),
  (88, 90, 71)
]

-- List Harold's seven quiz scores from greatest to least
noncomputable def harold_all_scores : list ℕ :=
  [94, 92, 90, 88, 84, 80, 76]

theorem harold_scores_correct :
  let initial_scores := harold_initial_scores,
      mean := mean_score,
      total := total_score,
      last_score := last_three_score,
      all_scores := harold_all_scores in
  (∀ score ∈ all_scores, score < 95) ∧ -- Each quiz score is less than 95
  (list.nodup all_scores) ∧ -- All scores are different integer values
  (list.sum all_scores = total) ∧ -- The total score is correct
  (∀ score ∈ initial_scores, score ∈ all_scores) := -- Initial scores are in the full list
  by {
    -- Verification steps can be skipped with sorry
    sorry
  }

end harold_scores_correct_l53_53802


namespace jessica_deposit_fraction_l53_53138

theorem jessica_deposit_fraction (init_balance withdraw_amount final_balance : ℝ)
  (withdraw_fraction remaining_fraction deposit_fraction : ℝ) :
  remaining_fraction = withdraw_fraction - (2/5) → 
  init_balance * withdraw_fraction = init_balance - withdraw_amount →
  init_balance * remaining_fraction + deposit_fraction * (init_balance * remaining_fraction) = final_balance →
  init_balance = 500 →
  final_balance = 450 →
  withdraw_amount = 200 →
  remaining_fraction = (3/5) →
  deposit_fraction = 1/2 :=
by
  intros hr hw hrb hb hf hwamount hr_remain
  sorry

end jessica_deposit_fraction_l53_53138


namespace tens_digit_of_6_pow_4_is_9_l53_53609

theorem tens_digit_of_6_pow_4_is_9 : (6 ^ 4 / 10) % 10 = 9 :=
by
  sorry

end tens_digit_of_6_pow_4_is_9_l53_53609


namespace ensure_two_different_colors_ensure_two_yellow_balls_l53_53492

-- First statement: Ensuring two balls of different colors
theorem ensure_two_different_colors (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 11 ∧ 
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, draws i ≠ draws j := 
sorry

-- Second statement: Ensuring two yellow balls
theorem ensure_two_yellow_balls (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 22 ∧
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, (draws i).val - balls_red - balls_white < balls_yellow ∧ 
              (draws j).val - balls_red - balls_white < balls_yellow ∧
              draws i = draws j := 
sorry

end ensure_two_different_colors_ensure_two_yellow_balls_l53_53492


namespace distinct_modulo_3n_l53_53643

theorem distinct_modulo_3n (n : ℕ) (h : n % 2 = 1) :
  ∃ (a b : Fin n → ℤ), ∀ m : Fin (n - 1), 
  let a_sums := (List.range n).map (λ i, a ⟨i, by simp [Nat.lt_succ_iff]⟩ + a ⟨(i + 1) % n, by simp [Nat.add_mod]; exact Nat.mod_lt _ h.right⟩) in
  let a_b_sums := (List.range n).map (λ i, a ⟨i, by simp [Nat.lt_succ_iff]⟩ + b ⟨i, by simp [Nat.lt_succ_iff]⟩) in
  let b_sums := (List.range n).map (λ i, b ⟨i, by simp [Nat.lt_succ_iff]⟩ + b ⟨(i + m + 1) % n, by simp [Nat.add_mod]; exact Nat.mod_lt _ h.right⟩) in
  List.Nodup ((a_sums ++ a_b_sums ++ b_sums).map (λ x, x % (3 * n))) :=
sorry

end distinct_modulo_3n_l53_53643


namespace complex_point_in_first_quadrant_l53_53865

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53865


namespace operation_result_l53_53223

theorem operation_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 12) (h_prod : a * b = 32) 
: (1 / a : ℚ) + (1 / b) = 3 / 8 := by
  sorry

end operation_result_l53_53223


namespace base_conversion_problem_l53_53576

variable (A C : ℕ)
variable (h1 : 0 ≤ A ∧ A < 8)
variable (h2 : 0 ≤ C ∧ C < 5)

theorem base_conversion_problem (h : 8 * A + C = 5 * C + A) : 8 * A + C = 39 := 
sorry

end base_conversion_problem_l53_53576


namespace trailing_zeros_of_500_pow_50_l53_53621

theorem trailing_zeros_of_500_pow_50 : 
  ∀ (n : ℕ), 500 = 5 * 10^2 → ∃ (m : ℕ), 500^n = 5^n * 10^(2 * n) → m = 2 * n := 
by
  intros n h1 h2
  use (2 * n)
  exact sorry

end trailing_zeros_of_500_pow_50_l53_53621


namespace total_balloons_170_l53_53177

variable (minutes_Max : ℕ) (rate_Max : ℕ) (minutes_Zach : ℕ) (rate_Zach : ℕ) (popped : ℕ)

def balloons_filled_Max := minutes_Max * rate_Max
def balloons_filled_Zach := minutes_Zach * rate_Zach
def total_filled_balloons := balloons_filled_Max + balloons_filled_Zach - popped

theorem total_balloons_170 
  (h1 : minutes_Max = 30) 
  (h2 : rate_Max = 2) 
  (h3 : minutes_Zach = 40) 
  (h4 : rate_Zach = 3) 
  (h5 : popped = 10) : 
  total_filled_balloons minutes_Max rate_Max minutes_Zach rate_Zach popped = 170 := by
  unfold total_filled_balloons
  unfold balloons_filled_Max
  unfold balloons_filled_Zach
  sorry

end total_balloons_170_l53_53177


namespace collatz_sequence_from_3_l53_53573

def next_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

theorem collatz_sequence_from_3 : ∃ (k : ℕ) (m : ℕ), 
  (nat.iterate next_step k 3 = 1 ∧ nat.iterate next_step (k + 1) 3 = 4 ∧ nat.iterate next_step (k + 2) 3 = 2 ∧ nat.iterate next_step (k + 3) 3 = 1) :=
by
  sorry -- Placeholder for the actual proof.

end collatz_sequence_from_3_l53_53573


namespace total_letters_in_alphabet_l53_53835

variable (d_and_s s_no_d d_no_s : Nat)

theorem total_letters_in_alphabet (h_d_and_s : d_and_s = 9)
    (h_s_no_d : s_no_d = 24) (h_d_no_s : d_no_s = 7) :
    d_and_s + s_no_d + d_no_s = 40 :=
  by
  rw [h_d_and_s, h_s_no_d, h_d_no_s]
  sorry

end total_letters_in_alphabet_l53_53835


namespace min_value_expression_l53_53019

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_l53_53019


namespace interval_mon_increasing_min_ω_l53_53078

open Real

def f (ω x : ℝ) : ℝ := (cos (ω * x)) * (cos (ω * x)) + (sqrt 3) * (sin (ω * x)) * (cos (ω * x))

-- Problem 1
theorem interval_mon_increasing (ω : ℝ) (hω : ω = 1) :
  ∀ k : ℤ, 
  ∃ I : set ℝ, 
    I = set.Icc (-π / 3 + k * π) (π / 6 + k * π) ∧ 
    strict_mono_on (λ x, f ω x) I := 
sorry

-- Problem 2
theorem min_ω (h_sym_center: (∃ x : ℝ, x = π / 6) ) :
  min_value (ω > 0) (2 / 3) (λ ω, ∃ k : ℤ, ω = 3 * k - 1 / 2) := 
sorry

end interval_mon_increasing_min_ω_l53_53078


namespace complex_number_quadrant_l53_53897

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53897


namespace fourth_power_of_cube_third_smallest_prime_l53_53291

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53291


namespace breakfast_sets_count_l53_53547

theorem breakfast_sets_count (muffins: ℕ) (butters: ℕ) (jams: ℕ) 
  (hm: muffins = 12) (hb: butters = 10) (hj: jams = 8) : 
  muffins * butters * jams = 960 :=
by {
  rw [hm, hb, hj],
  norm_num,
}

end breakfast_sets_count_l53_53547


namespace find_vector_v_l53_53386

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let num := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1^2 + v.2^2)
  let scalar := num / denom
  (scalar * v.1, scalar * v.2)

theorem find_vector_v : 
  ∃ v : ℝ × ℝ, 
    proj ⟨3, 2⟩ v = ⟨45/13, 30/13⟩ ∧ 
    proj ⟨1, 4⟩ v = ⟨32/17, 128/17⟩ ∧ 
    v = ⟨8, 6⟩ :=
by 
  use (8, 6)
  split
  { -- proj ⟨3, 2⟩ ⟨8, 6⟩ = ⟨45/13, 30/13⟩
    dsimp [proj],
    -- calculation steps omitted
    sorry
  }
  split
  { -- proj ⟨1, 4⟩ ⟨8, 6⟩ = ⟨32/17, 128/17⟩
    dsimp [proj],
    -- calculation steps omitted
    sorry
  }
  { refl } -- v = ⟨8, 6⟩

end find_vector_v_l53_53386


namespace sum_of_rational_roots_of_h_eq_zero_l53_53742

def h (x : ℚ) : ℚ := x^3 - 6 * x^2 + 9 * x - 3

theorem sum_of_rational_roots_of_h_eq_zero :
  (∑ r in {r : ℚ | h r = 0}.to_finset, r) = 0 :=
  by
    sorry

end sum_of_rational_roots_of_h_eq_zero_l53_53742


namespace min_value_proof_l53_53162

noncomputable def min_value_expr (a b c : ℝ) : ℝ :=
  a^2 + b^2 + c^2 + (1 / a^2) + (b / a) + (c / b)

theorem min_value_proof :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ min_value_expr a b c = 6 * real.sqrt 3 :=
by
  sorry

end min_value_proof_l53_53162


namespace john_income_increase_l53_53516

theorem john_income_increase :
  let initial_job_income := 60
  let initial_freelance_income := 40
  let initial_online_sales_income := 20

  let new_job_income := 120
  let new_freelance_income := 60
  let new_online_sales_income := 35

  let weeks_per_month := 4

  let initial_monthly_income := (initial_job_income + initial_freelance_income + initial_online_sales_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income + new_online_sales_income) * weeks_per_month
  
  let percentage_increase := 100 * (new_monthly_income - initial_monthly_income) / initial_monthly_income

  percentage_increase = 79.17 := by
  sorry

end john_income_increase_l53_53516


namespace inequality_holds_for_minimal_a_l53_53091

theorem inequality_holds_for_minimal_a :
  ∀ (x : ℝ), (1 ≤ x) → (x ≤ 4) → (1 + x) * Real.log x + x ≤ x * 1.725 :=
by
  intros x h1 h2
  sorry

end inequality_holds_for_minimal_a_l53_53091


namespace count_numbers_without_digit_one_l53_53084

def contains_digit_one (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.contains 1

def count_valid_numbers (max_value : Nat) : Nat :=
  (Finset.range (max_value + 1)).filter (λ n => ¬ contains_digit_one n).card

theorem count_numbers_without_digit_one : count_valid_numbers 2000 = 1457 :=
by
  sorry

end count_numbers_without_digit_one_l53_53084


namespace time_at_displacement_equality_l53_53652

-- Define the initial conditions and equations
variables (S0 V0 g μ t : ℝ)

-- Define the velocity and displacement at time t
def velocity_at_time_t (t : ℝ) : ℝ := g * t + V0 - μ * g * t
def displacement_at_time_t (t : ℝ) : ℝ := S0 + 1/2 * g * t^2 + V0 * t - 1/2 * μ * g * t^2

-- Define the time at which the displacement S equals S0 + 100
def time_solution (S0 V0 g μ : ℝ) : ℝ := 
  (-V0 + sqrt((V0)^2 + 200 * (1 - μ) * g)) / ((1 - μ) * g)

-- The target statement to prove in Lean
theorem time_at_displacement_equality :
  displacement_at_time_t S0 V0 g μ (time_solution S0 V0 g μ) = S0 + 100 :=
by
  sorry

end time_at_displacement_equality_l53_53652


namespace problem_part1_problem_part2_l53_53793

theorem problem_part1 (k m : ℝ) :
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ k ≠ 3)) →
  k = -3 :=
sorry

theorem problem_part2 (k m : ℝ) :
  ((∃ x1 x2 : ℝ, 
     ((|k|-3) * x1^2 - (k-3) * x1 + 2*m + 1 = 0) ∧
     (3 * x2 - 2 = 4 - 5 * x2 + 2 * x2) ∧
     x1 = -x2) →
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ x = -1)) →
  (k = -3 ∧ m = 5/2)) :=
sorry

end problem_part1_problem_part2_l53_53793


namespace find_d_l53_53384

def satisfies_equation (a b c d : ℝ) : Prop :=
  2 * a^2 + 3 * b^2 + 4 * c^2 + 4 = 3 * d + real.sqrt (2 * a + 3 * b + 4 * c + 1 - 3 * d)

theorem find_d (a b c d : ℝ) (h : satisfies_equation a b c d) :
  d = 7 / 4 :=
sorry

end find_d_l53_53384


namespace greatest_integer_not_exceeding_1000y_l53_53687

theorem greatest_integer_not_exceeding_1000y (y : ℝ) 
  (cube_edge : ℝ) 
  (shadow_area_excluding_base : ℝ) 
  (shadow_area : ℝ) 
  (total_shadow_area : ℝ) 
  (side_square_shadow : ℝ) 
  (base_area : ℝ) :
  cube_edge = 2 →
  shadow_area_excluding_base = 200 →
  base_area = cube_edge * cube_edge →
  shadow_area = shadow_area_excluding_base + base_area →
  side_square_shadow = Real.sqrt shadow_area →
  2 * y = side_square_shadow →
  total_shadow_area = shadow_area →
  1000 * y <= 14280 ∧ 14280 < 1000 * (y + 1) → 
  Nat.floor(1000 * y) = 14280 :=
by
  intros cube_edge_eq shadow_area_excl_base_eq base_area_eq shadow_area_eq side_square_shadow_eq total_shadow_eq Hy
  sorry

end greatest_integer_not_exceeding_1000y_l53_53687


namespace a_2009_l53_53822

def seq : ℕ → ℚ
| 1       := 1 - (3 / 4)
| (n + 1) := 1 - (1 / seq n)

theorem a_2009 : seq 2009 = -3 :=
sorry

end a_2009_l53_53822


namespace length_of_ST_l53_53316

-- Given conditions of the problem
variables (P Q R S T U : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
[MetricSpace S] [MetricSpace T] [MetricSpace U] (PQ QR SU : ℝ)

-- Triangles similarity assumption
def similar_triangles (ABC DEF : Type) [MetricSpacesABC DEF] : Prop :=
  sorry 

-- Given values in the problem
def PQ_val : ℝ := 7
def QR_val : ℝ := 10
def SU_val : ℝ := 4

-- Proof that ST equals 2.8 cm
theorem length_of_ST (h₁ : similar_triangles (triangle P Q R) (triangle S T U))
  (h₂ : PQ = 7) (h₃ : QR = 10) (h₄ : SU = 4) : ∃ ST : ℝ, ST = 2.8 := 
by
  sorry

end length_of_ST_l53_53316


namespace ellipse_equation_exists_max_lambda_l53_53512

-- Given conditions
def line_l : ℝ → ℝ := λ x, sqrt 3 * x - 2 * sqrt 3
def ellipse_C (a b : ℝ) : ℝ × ℝ → Prop := λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1
def focal_distance (a c : ℝ) : ℝ := 2 * c
def symmetric_center (a c : ℝ) : ℝ := a^2 / c
def focus_F2 (x y : ℝ) (a : ℝ) (c : ℝ) : Bool := line_l x = y ∧ sqrt (a^2 - c^2) = c
def line_m (x : ℝ) : ℝ → ℝ := λ y, x / y - 2
def scalar_product (OM ON : ℝ × ℝ) (λ : ℝ) : Bool := OM.1 * ON.1 + OM.2 * ON.2 = (2 * λ) / (tan (atan2 ON.2 ON.1 - atan2 OM.2 OM.1))

-- Prove
theorem ellipse_equation_exists: ∃ a b : ℝ, ellipse_C a b ∧ a^2 = 6 ∧ b^2 = 2 := sorry

theorem max_lambda: ∃ λ : ℝ, λ = sqrt 3 ∧ scalar_product (3,0) (0,1) λ := sorry

end ellipse_equation_exists_max_lambda_l53_53512


namespace fourth_power_cube_third_smallest_prime_l53_53264

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53264


namespace exists_universal_friend_l53_53484

-- Variables and Definitions
universe u
variables (V : Type u) [fintype V] (G : simple_graph V)

-- Given condition that every two vertices have exactly one common friend
def unique_common_neighbor (G : simple_graph V) : Prop :=
  ∀ (u v : V), u ≠ v → ∃! (w : V), G.adj u w ∧ G.adj v w

-- Theorem to be proved
theorem exists_universal_friend (h : unique_common_neighbor G) :
  ∃ (v : V), ∀ (u : V), u ≠ v → G.adj v u :=
sorry

end exists_universal_friend_l53_53484


namespace sum_of_digits_h2011_1_eq_16089_l53_53990

noncomputable def f (x : ℝ) : ℝ := 10^(10 * x)
noncomputable def g (x : ℝ) : ℝ := (Real.log x / Real.log 10) - 1
noncomputable def h1 (x : ℝ) : ℝ := g (f x)
noncomputable def h  (n : ℕ) : ℝ -> ℝ
| 1       := h1
| (n + 2) := h1 ∘ (h (n + 1))

theorem sum_of_digits_h2011_1_eq_16089 :
  sum_of_digits (h 2011 1) = 16089 :=
sorry

-- Auxiliary function to calculate the sum of the digits of a real number (given on RHS of the theorem)
noncomputable def sum_of_digits (n : ℝ) : ℕ :=
  n.to_string.to_nat.digits.sum

end sum_of_digits_h2011_1_eq_16089_l53_53990


namespace comparison_l53_53753

open Real

noncomputable def a := 5 * log (2 ^ exp 1)
noncomputable def b := 2 * log (5 ^ exp 1)
noncomputable def c := 10

theorem comparison : c > a ∧ a > b :=
by
  have a_def : a = 5 * log (2 ^ exp 1) := rfl
  have b_def : b = 2 * log (5 ^ exp 1) := rfl
  have c_def : c = 10 := rfl
  sorry -- Proof goes here

end comparison_l53_53753


namespace quadrilateral_perimeter_l53_53498

open Real

/-- Given a quadrilateral PQRS with specific conditions,
prove that its perimeter is 44 + √706. -/
theorem quadrilateral_perimeter
  (P Q R S : ℝ × ℝ)
  (PQ QR RS PR : ℝ)
  (hQ : angle (P-Q) (R-Q) = π / 2)
  (hPR_RS : angle (P-R) (S-R) = π / 2)
  (hPQ : dist P Q = 15)
  (hQR : dist Q R = 20)
  (hRS : dist R S = 9) :
  dist P Q + dist Q R + dist R S + dist P S = 44 + Real.sqrt 706 :=
by
  sorry

end quadrilateral_perimeter_l53_53498


namespace log_sqrt10_1000sqrt10_l53_53005

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53005


namespace general_formula_limit_S_l53_53349

noncomputable def S : ℕ → ℝ
| 0     := 1
| (n+1) := S n + (3 * 4^n) * (1 / 3^(2*(n+1)))

theorem general_formula (n : ℕ) : S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

theorem limit_S : tendsto S atTop (𝓝 (8 / 5)) := sorry

end general_formula_limit_S_l53_53349


namespace tank_ratio_l53_53698

variable (C D : ℝ)
axiom h1 : 3 / 4 * C = 2 / 5 * D

theorem tank_ratio : C / D = 8 / 15 := by
  sorry

end tank_ratio_l53_53698


namespace complex_point_in_first_quadrant_l53_53860

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53860


namespace winning_candidate_percentage_l53_53238

def percentage_votes (votes1 votes2 votes3 : ℕ) : ℚ := 
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := max (max votes1 votes2) votes3
  (winning_votes * 100) / total_votes

theorem winning_candidate_percentage :
  percentage_votes 3000 5000 15000 = (15000 * 100) / (3000 + 5000 + 15000) :=
by 
  -- This computation should give us the exact percentage fraction.
  -- Simplifying it would yield the result approximately 65.22%
  -- Proof steps can be provided here.
  sorry

end winning_candidate_percentage_l53_53238


namespace equal_parts_count_l53_53676

def scale_length_in_inches : ℕ := (7 * 12) + 6
def part_length_in_inches : ℕ := 18
def number_of_parts (total_length part_length : ℕ) : ℕ := total_length / part_length

theorem equal_parts_count :
  number_of_parts scale_length_in_inches part_length_in_inches = 5 :=
by
  sorry

end equal_parts_count_l53_53676


namespace derivative_of_y_l53_53733

noncomputable def y (x : ℝ) : ℝ :=
  real.sqrt (1 - 3 * x - 2 * x^2) + (3 / (2 * real.sqrt 2)) * real.arcsin ((4 * x + 3) / real.sqrt 17)

theorem derivative_of_y (x : ℝ) : deriv y x = - (2 * x) / real.sqrt (1 - 3 * x - 2 * x^2) :=
by sorry

end derivative_of_y_l53_53733


namespace length_of_EC_l53_53779

theorem length_of_EC (A B C D E : Type) 
  [has_angle_measure A B C D E] 
  (angle_A : angle_measure A = 45)
  (length_BC : BC = 8)
  (perp_BD_AC : perp BD AC)
  (perp_CE_AB : perp CE AB)
  (angle_DBC_ECB : ∃ y, angle_measure DBC = 4 * y ∧ angle_measure ECB = y) : 
  ∃ a b c : ℕ, EC = a * (real.sqrt b + real.sqrt c) ∧ a + b + c = 4 :=
by {
  sorry
}

end length_of_EC_l53_53779


namespace find_x_for_horizontal_asymptote_l53_53723

def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (2 * x^2 - 8 * x + 5)

theorem find_x_for_horizontal_asymptote (x : ℝ) : g(x) = 3/2 ↔ x = 13 :=
by
  -- Reasoning here using algebraic methods
  sorry

end find_x_for_horizontal_asymptote_l53_53723


namespace sum_of_divisors_eq_720_l53_53601

theorem sum_of_divisors_eq_720 (i j : ℕ) (h : nat.divisor_sum (2^i * 3^j) = 720) : i + j = 6 :=
sorry

end sum_of_divisors_eq_720_l53_53601


namespace find_expression_l53_53210

theorem find_expression (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 84) (h2 : a = 32) : E = 80 :=
by
  -- Proof to be filled in here
  sorry

end find_expression_l53_53210


namespace find_k_l53_53798

noncomputable def f (x k : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x + k else -1/2 + log x / log (1/3)

noncomputable def g (x a : ℝ) : ℝ :=
a * log (x + 2) + x / (x^2 + 1)

theorem find_k (a : ℝ) (k : ℝ) (x1 x2 : ℝ) (hx1 : x1 ∈ set.Icc (-2 : ℝ) 1) (hx2 : x2 ∈ set.Icc (-2 : ℝ) 1) :
  (∀ x ∈ set.Icc (-2 : ℝ) 1, f x k ≤ g x a)
  → k ≤ -3 / 4 :=
by
  sorry

end find_k_l53_53798


namespace undefined_fraction_l53_53390

open Real

theorem undefined_fraction (b : ℝ) :
  (b^2 + 1) / (b^3 - 8) = (0 : ℝ) → b = 2 :=
begin
  sorry
end

end undefined_fraction_l53_53390


namespace point_in_first_quadrant_l53_53881

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53881


namespace relay_team_order_count_l53_53973

theorem relay_team_order_count (team_members : Finset ℕ) (h_card : team_members.card = 5) :
  (∀ jordan, jordan ∈ team_members →
   (∃ perm : (team_members.erase jordan) → ℕ,
    (∀ x, x ∈ team_members.erase jordan → 
     ∃! (i : ℕ), i ∈ Finset.range 4 ∧ perm x = i) ∧
     (∃ jordan_idx : ℕ, jordan_idx = 4 ∧ \#({perm.val 0, perm.val 1, perm.val 2, perm.val 3} ∪ {jordan_idx}) = 5))
  → ∑ perm, prod (perm.val \id) = 24) := sorry

end relay_team_order_count_l53_53973


namespace minimum_set_size_l53_53766

theorem minimum_set_size (n : ℕ) (h : 2 ≤ n) (X : Type) [Fintype X] [DecidableEq X]
  (hX : Fintype.card X = 2 * n - 1)
  (B : Fin n → Finset X) :
  ∃ Y : Finset X, Y.card = n ∧ ∀ i : Fin n, (Y ∩ B i).card ≤ 1 :=
begin
  sorry
end

end minimum_set_size_l53_53766


namespace freezer_temp_calculation_l53_53827

def refrigerator_temp : ℝ := 4
def freezer_temp (rt : ℝ) (d : ℝ) : ℝ := rt - d

theorem freezer_temp_calculation :
  (freezer_temp refrigerator_temp 22) = -18 :=
by
  sorry

end freezer_temp_calculation_l53_53827


namespace identify_different_weight_in_three_weighings_l53_53254

def different_weight_exists (weights : List ℝ) : Prop :=
  ∃ (find_different_weight : List ℝ → ℝ × ℝ × ℝ), 
    (∀ w1 w2 w3 : List ℝ,
      w1.length = 4 ∧ w2.length = 4 ∧ w3.length = 5 ∧ 
      w1 ++ w2 ++ w3 = weights →
      let res := find_different_weight weights in
      (res.1, res.2, res.3) ∈ { (w : ℝ) // w ∈ weights ∧ w ≠ res.1 ∧ w ≠ res.2 ∧ w ≠ res.3 })

theorem identify_different_weight_in_three_weighings (weights : List ℝ) 
  (h1 : weights.length = 13) 
  (h2 : ∃ w : ℝ, w ∈ weights ∧ (∀ other_weight ∈ weights, other_weight ≠ w → other_weight = w)) : 
  different_weight_exists weights :=
  sorry

end identify_different_weight_in_three_weighings_l53_53254


namespace difference_english_math_l53_53481

/-- There are 30 students who pass in English and 20 students who pass in Math. -/
axiom passes_in_english : ℕ
axiom passes_in_math : ℕ
axiom both_subjects : ℕ
axiom only_english : ℕ
axiom only_math : ℕ

/-- Definitions based on the problem conditions -/
axiom number_passes_in_english : only_english + both_subjects = 30
axiom number_passes_in_math : only_math + both_subjects = 20

/-- The difference between the number of students who pass only in English
    and the number of students who pass only in Math is 10. -/
theorem difference_english_math : only_english - only_math = 10 :=
by
  sorry

end difference_english_math_l53_53481


namespace problem_statement_l53_53093

variable (a b x : ℝ)

theorem problem_statement (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  a / (a - b) = x / (x - 1) :=
sorry

end problem_statement_l53_53093


namespace complex_quadrant_check_l53_53912

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53912


namespace real_distinct_roots_l53_53148

def P1 (x : ℝ) := x^2 - 2

def P (j : ℕ) (x : ℝ) : ℝ :=
  if j = 1 then P1 x
  else P1 (P (j-1) x)

theorem real_distinct_roots (n : ℕ) (hn : n > 0) :
  ∃ (roots : Finset ℝ), ∀ (r ∈ roots), P n r = r ∧ ∀ (x y ∈ roots), x ≠ y → x ≠ y :=
by
  sorry

end real_distinct_roots_l53_53148


namespace shift_to_obtain_sin_from_cos_l53_53244

-- Define the functions
def f1 (x : ℝ) : ℝ := sin (3 * x - π / 4)
def f2 (x : ℝ) : ℝ := cos (3 * x)

-- Define the required transformation
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- State the problem
theorem shift_to_obtain_sin_from_cos :
  ∀ x : ℝ, f1 x = shift_right f2 (π / 4) x :=
by
  sorry

end shift_to_obtain_sin_from_cos_l53_53244


namespace complex_point_in_first_quadrant_l53_53854

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53854


namespace sum_of_roots_of_equation_l53_53812

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l53_53812


namespace solution_set_l53_53796

noncomputable def f : ℝ → ℝ := λ x, x + Real.sin x

theorem solution_set (x : ℝ) (h : 0 < x < Real.exp 1) :
  (f (Real.log x) - f (Real.log (1 / x))) / 2 < f 1 := by
  sorry

end solution_set_l53_53796


namespace centroid_of_contour_is_centroid_l53_53375

-- Define a structure for Triangle
structure Triangle (α : Type _) [LinearOrderedField α] := 
  (A B C : α × α)

-- Function to find the midpoint of a line segment
def midpoint {α : Type _} [LinearOrderedField α] (P Q : α × α) : α × α :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Definition of the centroid for the contour of a triangle
noncomputable def centroid_of_contour {α : Type _} [LinearOrderedField α] (T : Triangle α) : α × α :=
let A1 := midpoint T.B T.C,
    B1 := midpoint T.C T.A,
    C1 := midpoint T.A T.B in
intersection_of_angle_bisectors A1 B1 C1

-- Statement to prove the centroid of the contour
theorem centroid_of_contour_is_centroid (α : Type _) [LinearOrderedField α] (T : Triangle α) :
  centroid_of_contour T = geometric_centroid_of_midpoints T := 
sorry

end centroid_of_contour_is_centroid_l53_53375


namespace PZ_length_mod_l53_53317

-- Definitions for the given conditions
variables {X Y Z P Q R O1 O2 O3 : Type} 
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [Collinear Q R X] [Collinear R P Y] [Collinear P Q Z]
variables (PQ QR RP : ℕ) (arc_PY arc_XQ arc_QZ arc_YP arc_RZ : ℕ)
variables [PQ = 14] [QR = 17] [RP = 15]
variables [arc_PY = arc_XQ] [arc_QZ = arc_YP] [arc_RZ = arc_XQ]

-- Proof statement to show PZ = 3 and compute 3 mod 1
theorem PZ_length_mod 
  (h1 : PQ = 14) (h2 : QR = 17) (h3 : RP = 15)
  (h4 : arc_PY = arc_XQ) (h5 : arc_QZ = arc_YP) (h6 : arc_RZ = arc_XQ) :
  let PZ := 3 in 
  PZ % 1 = 0 :=
by
  sorry

end PZ_length_mod_l53_53317


namespace steven_owes_jeremy_l53_53515

-- Define the payment per room
def payment_per_room : ℚ := 13 / 3

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 5 / 2

-- Calculate the total amount owed
def total_amount_owed : ℚ := payment_per_room * rooms_cleaned

-- The theorem statement to prove
theorem steven_owes_jeremy :
  total_amount_owed = 65 / 6 :=
by
  sorry

end steven_owes_jeremy_l53_53515


namespace expectation_of_red_balls_l53_53130

theorem expectation_of_red_balls :
  let ξ := (λ (draws : (Bool × Bool)), if draws.1 then 1 else 0 + if draws.2 then 1 else 0)
  in 
  let prob := (λ (ξ_val : ℕ), match ξ_val with
    | 0 => (2 / 3) * (1 / 2)
    | 1 => (1 / 3) * (1 / 2) + (2 / 3) * (1 / 2)
    | 2 => 1 - ((2 / 3) * (1 / 2) + (1 / 3) * (1 / 2))
  end)
  in ∑ x in {0, 1, 2}, x * prob x = 5 / 6 := 
by
  sorry

end expectation_of_red_balls_l53_53130


namespace theta_in_second_quadrant_l53_53758

theorem theta_in_second_quadrant (θ : Real) (h1 : cos θ < 0) (h2 : tan θ < 0) : 
    (π / 2 < θ) ∧ (θ < π) :=
sorry

end theta_in_second_quadrant_l53_53758


namespace monotonicity_intervals_max_tangent_slope_rhombus_condition_l53_53992

noncomputable def f (x t : ℝ) : ℝ := x^2 * (x - t)

theorem monotonicity_intervals 
  (t : ℝ) (ht : t > 0) :
  (∀ x, x < 0 → deriv (λ x, f x t) x > 0) ∧ 
  (∀ x, x > 0 → deriv (λ x, f x t) x > 0) ∧
  (∀ x, 0 < x ∧ x < t → deriv (λ x, f x t) x < 0) :=
by { sorry }

theorem max_tangent_slope 
  (x₀ t : ℝ) (hk : ∀ x₀ ∈ Ioc 0 1, 3 * x₀^2 - 2 * t * x₀ ≥ -1) :
  t ≤ 3 / 2 :=
by { sorry }

theorem rhombus_condition
  (t : ℝ) (ht : t > 0) :
  ∃ C D A B, 
  (C ≠ D) ∧ 
  (A ≠ B) ∧ 
  (* A, B, C, D form a rhombus given the conditions *)
  true :=
by { sorry }

end monotonicity_intervals_max_tangent_slope_rhombus_condition_l53_53992


namespace equilateral_prism_lateral_edge_length_l53_53772

theorem equilateral_prism_lateral_edge_length
  (base_side_length : ℝ)
  (h_base : base_side_length = 1)
  (perpendicular_diagonals : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = base_side_length ∧ b = lateral_edge ∧ c = some_diagonal_length ∧ lateral_edge ≠ 0)
  : ∀ lateral_edge : ℝ, lateral_edge = (Real.sqrt 2) / 2 := sorry

end equilateral_prism_lateral_edge_length_l53_53772


namespace marked_price_of_jacket_l53_53324

variable (x : ℝ) -- Define the variable x as a real number representing the marked price.

-- Define the conditions as a Lean theorem statement
theorem marked_price_of_jacket (cost price_sold profit : ℝ) (h1 : cost = 350) (h2 : price_sold = 0.8 * x) (h3 : profit = price_sold - cost) : 
  x = 550 :=
by
  -- We would solve the proof here using provided conditions
  sorry

end marked_price_of_jacket_l53_53324


namespace tangent_lines_parallel_l53_53066

noncomputable def f (x m : ℝ) : ℝ := x * (m - real.exp (-2 * x))
def g (x : ℝ) : ℝ := (1 - 2 * x) * real.exp (-2 * x)

theorem tangent_lines_parallel (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 m = 1 ∧ f' x2 m = 1) ↔ (1 - real.exp (-2) < m ∧ m < 1) :=
by
  sorry

end tangent_lines_parallel_l53_53066


namespace largest_square_side_l53_53634

theorem largest_square_side (width length : ℕ) (h_width : width = 63) (h_length : length = 42) : 
  Nat.gcd width length = 21 :=
by
  rw [h_width, h_length]
  sorry

end largest_square_side_l53_53634


namespace new_rectangle_side_length_l53_53345

-- Define the original rectangle dimensions
def original_length : ℕ := 24
def original_width : ℕ := 8

-- Define the area of the original rectangle
def original_area : ℕ := original_length * original_width

-- The given condition that the two congruent hexagons form a new rectangle
def new_rectangle_area : ℕ := original_area

-- This y is what we are asked to prove
def y : ℕ := 12

theorem new_rectangle_side_length :
  ∃ (y : ℕ), y * 16 = new_rectangle_area ∧ y = 12 := 
by
  use y
  split
  · simp [new_rectangle_area, original_length, original_width, y]
  · rfl

end new_rectangle_side_length_l53_53345


namespace positive_n_of_single_solution_l53_53030

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l53_53030


namespace min_value_of_f_l53_53017

-- Given function:
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x)

-- The minimum value of the function:
theorem min_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -6480.25 :=
by
  have h : ∀ y : ℝ, f y = y^4 - 289 * y^2 + 14400 := sorry
  have h_min : ∀ y : ℝ, (y^4 - 289 * y^2 + 14400) = (y^2 - 144.5)^2 - 6480.25 := sorry
  use 0 -- Since we just need existence, setting x to 0 for the purpose of example
  intro y
  split
  . -- Prove f x ≤ f y (which should simplify to show min at y^2 = 144.5)
    sorry
  . -- Prove f x = -6480.25
    admit
  -- These are place-holders showing where each proof part would be

end min_value_of_f_l53_53017


namespace tiling_vertex_squares_octagons_l53_53339

theorem tiling_vertex_squares_octagons (m n : ℕ) 
  (h1 : 135 * n + 90 * m = 360) : 
  m = 1 ∧ n = 2 :=
by
  sorry

end tiling_vertex_squares_octagons_l53_53339


namespace Qs_contribution_l53_53311

theorem Qs_contribution (P_investment P_time Q_time : ℕ) (profit_ratio : ℚ) 
  (hP : P_investment = 4000) 
  (hP_time : P_time = 12)
  (hQ_time : Q_time = 8)
  (h_ratio : profit_ratio = 2 / 3) :
  ∃ Q_investment : ℕ, Q_investment = 9000 :=
by
  let P_total_investment := P_investment * P_time
  have h : P_total_investment * 3 = 2 * (Q_time * 9000) := by
    calc
      P_total_investment * 3 = (4000 * 12) * 3 : by rw [hP, hP_time]
                      ... = 48000 * 3 : by norm_num
                      ... = 144000 : by norm_num
      2 * (Q_time * 9000) = 2 * (8 * 9000) : by rw [hQ_time]
                      ... = 2 * 72000 : by norm_num
                      ... = 144000 : by norm_num
  use 9000
  exact eq_of_mul_eq_mul_left (by norm_num : 16 ≠ 0) h.right.symm

end Qs_contribution_l53_53311


namespace area_difference_eq_l53_53557

-- Let S be the smallest convex polygon containing the union R of a unit square and two equilateral triangles with side length 2.
-- Prove that the area of the region inside S but outside R is 11.5*sqrt 3 - 1.
theorem area_difference_eq :
  let unit_square := (1 : ℝ) -- Defines the side of the unit square.
  let side_length_triangle := (2 : ℝ) -- Defines the side length of the triangles.
  let area_square := unit_square * unit_square -- Calculates the area of the unit square.
  let area_triangle := (sqrt 3 / 4) * (side_length_triangle ^ 2) -- Calculates the area of one equilateral triangle.
  let area_R := area_square + 2 * area_triangle -- Total area of R, the union of the square and the two triangles.
  let hexagon_side := 3 -- Side length of the hexagon forming S.
  let height_hexagon := (sqrt 3 / 2) * hexagon_side -- Height from the center to a side of the hexagon.
  let area_S := 6 * ((3 / 2) * height_hexagon) / 2 -- Total area of the hexagon S.
  let area_diff := area_S - area_R -- Difference in the areas to be calculated.
  area_diff = 11.5 * sqrt 3 - 1 := 
sorry -- proof omitted

end area_difference_eq_l53_53557


namespace sum_of_largest_and_third_largest_l53_53024

theorem sum_of_largest_and_third_largest (n1 n2 n3 : Nat) (h1 : n1 = 8) (h2 : n2 = 1) (h3 : n3 = 6) :
  let largest := 861
  let third_largest := 681
  largest + third_largest = 1542 :=
by
  -- Define the numbers explicitly for the clarity of the statement.
  have largest : ℕ := 100 * n1 + 10 * n2 + n3
  have third_largest : ℕ := 100 * n3 + 10 * n1 + n2
  -- Assume
  have h : largest = 861 := by
    rw [h1, h2, h3]
    norm_num
  have h' : third_largest = 681 := by
    rw [h1, h2, h3]
    norm_num
  -- Prove
  rw [h, h']
  norm_num
  sorry

end sum_of_largest_and_third_largest_l53_53024


namespace digit_6_count_in_100_to_999_l53_53952

theorem digit_6_count_in_100_to_999 : 
  (Nat.digits_count_in_range 6 100 999) = 280 := 
sorry

end digit_6_count_in_100_to_999_l53_53952


namespace triangle_perimeter_from_medians_l53_53785

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l53_53785


namespace complex_quadrant_l53_53937

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53937


namespace percent_first_shift_participating_l53_53726

variable (total_employees_in_company : ℕ)
variable (first_shift_employees : ℕ)
variable (second_shift_employees : ℕ)
variable (third_shift_employees : ℕ)
variable (second_shift_percent_participating : ℚ)
variable (third_shift_percent_participating : ℚ)
variable (overall_percent_participating : ℚ)
variable (first_shift_percent_participating : ℚ)

theorem percent_first_shift_participating :
  total_employees_in_company = 150 →
  first_shift_employees = 60 →
  second_shift_employees = 50 →
  third_shift_employees = 40 →
  second_shift_percent_participating = 0.40 →
  third_shift_percent_participating = 0.10 →
  overall_percent_participating = 0.24 →
  first_shift_percent_participating = (12 / 60) →
  first_shift_percent_participating = 0.20 := 
by 
  intros t_e f_s_e s_s_e t_s_e s_s_p_p t_s_p_p o_p_p f_s_p_p
  -- Sorry, here would be the place for the actual proof
  sorry

end percent_first_shift_participating_l53_53726


namespace complex_quadrant_check_l53_53914

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53914


namespace fourth_power_cube_third_smallest_prime_l53_53269

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53269


namespace eccentricity_of_ellipse_l53_53791

-- Define the parameters of the ellipse and the condition on the tangent line
variables (a b : ℝ) (h : a > b) (h0 : b > 0)
  (h1 : ∀ x y : ℝ, (bx - ay + 2ab = 0) → (∃ R, (x^2 + y^2 = R^2) ∧ 
          (x - a)*(x + a) + (a^2 - 3b^2)/(a*x - a^2) = 0))

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the eccentricity e
def eccentricity (a b : ℝ) : ℝ := 
  let c := (a^2 - b^2).sqrt in c / a

-- State the theorem for the eccentricity of the ellipse given the conditions
theorem eccentricity_of_ellipse : eccentricity a b = (Real.sqrt 6) / 3 :=
by {
  sorry
}

end eccentricity_of_ellipse_l53_53791


namespace concurrency_collinearity_proof_l53_53942

variable {A B C P Q S D T U V N E : Type*}
variable [geometry.point P AC] [geometry.point Q BC] [geometry.triangle ABC]
variable [geometry.line P] [geometry.line Q]
variable (AC BC AB CD BP SQ QT QS)

-- Point Definitions
def point_P_on_AC : geometry.point P AC := sorry 
def point_Q_on_BC : geometry.point Q BC := sorry 
def point_S_on_AB : geometry.point S AB := sorry 
def point_D_on_AB : geometry.point D AB := sorry 
def point_T_on_AB : geometry.point T AB := sorry 

-- Intersection Points
def intersection_U : geometry.point U CD BP := sorry 
def intersection_V : geometry.point V CD SQ := sorry 
def intersection_N : geometry.point N BP QT := sorry 
def intersection_E : geometry.point E QS AC := sorry 

-- Concurrency and Collinearity Proof
theorem concurrency_collinearity_proof :
  (concurrent (PS : geometry.line) (CD : geometry.line) (QT : geometry.line)) ↔
  collinear [E : geometry.point, D : geometry.point, N : geometry.point] := sorry

end concurrency_collinearity_proof_l53_53942


namespace difference_is_four_l53_53356

open Nat

-- Assume we have a 5x5x5 cube
def cube_side_length : ℕ := 5
def total_unit_cubes : ℕ := cube_side_length ^ 3

-- Define the two configurations
def painted_cubes_config1 : ℕ := 65  -- Two opposite faces and one additional face
def painted_cubes_config2 : ℕ := 61  -- Three adjacent faces

-- The difference in the number of unit cubes with at least one painted face
def painted_difference : ℕ := painted_cubes_config1 - painted_cubes_config2

theorem difference_is_four :
    painted_difference = 4 := by
  sorry

end difference_is_four_l53_53356


namespace find_angle_CAD_l53_53476

-- Definitions for points and angles in triangle ABC
variables (A B C D : Type)
variables (angle_B : Real)
variables (angle_BAD : Real)
variables (AB CD : Real)

noncomputable def problem_statement :=
  ∀ (A B C D : ℝ),
    angle_B = 46 ∧
    angle_BAD = 21 ∧
    AB = CD →
    ∃ (angle_CAD : ℝ), angle_CAD = 67

theorem find_angle_CAD :
  problem_statement A B C D angle_B angle_BAD AB CD :=
begin
  sorry
end

end find_angle_CAD_l53_53476


namespace book_prices_minimum_cost_l53_53184

variables {x y m W : ℝ}
variable (hx : 2 * x + 3 * y = 126)
variable (hy : 3 * x + 2 * y = 109)
variable (hm1 : m ≤ 3 * (200 - m))
variable (hm2 : 70 ≤ m)
variable (hm3 : m ≤ 150)

theorem book_prices :
  ∃ x y : ℝ, 
  (2 * x + 3 * y = 126) ∧ (3 * x + 2 * y = 109) :=
begin
  use [15, 32],
  split,
  { exact hx, },
  { exact hy, }
end

theorem minimum_cost :
  W = -17 * 150 + 6400 :=
begin
  have range_m : 70 ≤ m ∧ m ≤ 150, by simp [hm2, hm3],
  let W := -17 * m + 6400,
  have min_cost : W = 3850, sorry
end

end book_prices_minimum_cost_l53_53184


namespace bug_opposite_vertex_probability_l53_53653

-- Define the cube vertex type
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define edge movement possibility and probability calculation
def movement (start : Vertex) : List Vertex :=
  match start with
  | A => [B, C, D]
  | B => [A, E, F]
  | C => [A, E, G]
  | D => [A, F, H]
  | E => [B, C, G]
  | F => [B, D, H]
  | G => [C, E, H]
  | H => [D, F, G]

-- Calculate the probability (placeholder function)
noncomputable def probability_six_moves (start end : Vertex) : ℚ :=
  if start = end then 0 else 1 / 8

-- Formal statement of the problem
theorem bug_opposite_vertex_probability :
  probability_six_moves A B = 1 / 8 :=
by
  sorry

end bug_opposite_vertex_probability_l53_53653


namespace Sunil_reached_total_amount_l53_53205

namespace SunilInvestment

def principal (CI: ℝ) (R: ℝ) : ℝ :=
CI / (R / 100)

def compoundInterest (P: ℝ) (R: ℝ) (T: ℝ) : ℝ :=
P * (1 + R / 100)^T - P

def totalAmount (P: ℝ) (CI: ℝ) (additional: ℝ) (R: ℝ) (T: ℝ) : ℝ :=
let newPrincipal := P + CI + additional in
let newCI := compoundInterest newPrincipal R T in
newPrincipal + newCI

theorem Sunil_reached_total_amount:
  totalAmount 8160 326.4 1000 5 2 = 10460.29
:= by
  sorry

end SunilInvestment

end Sunil_reached_total_amount_l53_53205


namespace sum_of_roots_l53_53814

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l53_53814


namespace find_prime_pairs_l53_53471

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_prime_pairs :
  ∀ m n : ℕ,
  is_prime m → is_prime n → (m < n ∧ n < 5 * m) → is_prime (m + 3 * n) →
  (m = 2 ∧ (n = 3 ∨ n = 5 ∨ n = 7)) :=
by
  sorry

end find_prime_pairs_l53_53471


namespace boxes_left_l53_53182

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l53_53182


namespace sum_sqrt_expression_sum_sqrt_values_l53_53150

theorem sum_sqrt_expression (a b c : ℤ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_irred : ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ c)) :
  (∑ n in Finset.range (10000-2+1) + 2, 1 / Real.sqrt (n + Real.sqrt (n^2 - 4))) = a + b * Real.sqrt c :=
begin
  sorry
end

theorem sum_sqrt_values :
  ∃ a b c : ℤ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (∑ n in Finset.range (10000-2+1) + 2, 1 / Real.sqrt (n + Real.sqrt (n^2 - 4))) = a + b * Real.sqrt c ∧ a + b + c = 161 :=
begin
  use [70, 89, 2],
  split,
  {
    split, 
    { norm_num }, -- 70 is positive
    { split,
      { norm_num }, -- 89 is positive
      { norm_num }  -- 2 is positive
    }
  },
  split,
  { intros p hp hcp,
    -- Prove that 2 is not divisible by p^2 for any prime p
    rw ← Int.coe_nat_dvd_right at hcp,
    exact Nat.Prime.not_dvd_one (by norm_num : ¬ Nat.Prime.prime 1)} },
  {
    rw sum_sqrt_expression 70 89 2,
    norm_num
  },
  sorry
end

end sum_sqrt_expression_sum_sqrt_values_l53_53150


namespace correct_option_is_B_l53_53301

theorem correct_option_is_B :
  (∃ (A B C D : String), A = "√49 = -7" ∧ B = "√((-3)^2) = 3" ∧ C = "-√((-5)^2) = 5" ∧ D = "√81 = ±9" ∧
    (B = "√((-3)^2) = 3")) :=
by
  sorry

end correct_option_is_B_l53_53301


namespace minimum_distance_icosahedron_l53_53694

-- Definitions
def icosahedron_edge_length := 4
def icosahedron_face_count := 20

-- Theorem statement
theorem minimum_distance_icosahedron : 
  ∃ (n : ℕ), n = 432 ∧ 
  -- Minimum travel distance from one vertex to the opposite vertex
  ∀ (distance : ℕ), distance = (12 * real.sqrt 3) → (real.sqrt n) = distance := 
by
  use 432
  sorry

end minimum_distance_icosahedron_l53_53694


namespace average_speed_of_Car_X_l53_53707

noncomputable def average_speed_CarX (V_x : ℝ) : Prop :=
  let head_start_time := 1.2
  let distance_traveled_by_CarX := 98
  let speed_CarY := 50
  let time_elapsed := distance_traveled_by_CarX / speed_CarY
  (distance_traveled_by_CarX / time_elapsed) = V_x

theorem average_speed_of_Car_X : average_speed_CarX 50 :=
  sorry

end average_speed_of_Car_X_l53_53707


namespace complex_point_in_first_quadrant_l53_53866

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53866


namespace find_other_number_l53_53227

theorem find_other_number (A B : ℕ) (HCF LCM : ℕ)
  (hA : A = 24)
  (hHCF: (HCF : ℚ) = 16)
  (hLCM: (LCM : ℚ) = 312)
  (hHCF_LCM: HCF * LCM = A * B) : 
  B = 208 :=
by
  sorry

end find_other_number_l53_53227


namespace value_of_Q_l53_53527

theorem value_of_Q (n : ℕ) (h : n = 2023) 
  (Q : ℚ := (∏ k in finset.range (n - 2) + 3, (1 - (1 / k))) ) : 
  Q = 2 / 2023 := 
by
  rw h
  sorry

end value_of_Q_l53_53527


namespace sufficient_condition_proof_l53_53155

variable {x y : ℝ}

def sufficient_but_not_necessary_condition : Prop :=
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ¬(x^2 + y^2 ≥ 4 → x ≥ 2 ∧ y ≥ 2)

theorem sufficient_condition_proof : sufficient_but_not_necessary_condition :=
by
  sorry

end sufficient_condition_proof_l53_53155


namespace complex_point_in_first_quadrant_l53_53905

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53905


namespace complex_quadrant_is_first_l53_53919

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53919


namespace monthly_rent_is_3600_rs_l53_53335

def shop_length_feet : ℕ := 20
def shop_width_feet : ℕ := 15
def annual_rent_per_square_foot_rs : ℕ := 144

theorem monthly_rent_is_3600_rs :
  (shop_length_feet * shop_width_feet) * annual_rent_per_square_foot_rs / 12 = 3600 :=
by sorry

end monthly_rent_is_3600_rs_l53_53335


namespace a_22006_eq_66016_l53_53589

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  ∀ n ≥ 1, (∃ m, m > a n ∧
    (∀ i j k ∈ finset.range (n + 2), i ≠ j → i ≠ k → j ≠ k → a i + a j ≠ 3 * a k) 
    ∧ a (n + 1) = m)

theorem a_22006_eq_66016 (a : ℕ → ℕ) (h : sequence a) :       
  a 22006 = 66016 :=
sorry

end a_22006_eq_66016_l53_53589


namespace train_speed_l53_53340

def speed_of_train (distance time : ℝ) : ℝ := (distance / time) * 3.6

theorem train_speed (h_distance : 400 = 400)
    (h_time : 9.99920006399488 = 9.99920006399488)
: abs ((speed_of_train 400 9.99920006399488) - 144.03) < 0.01 :=
by
  sorry

end train_speed_l53_53340


namespace multiply_particular_number_by_5_l53_53306

-- Define the particular number Y such that Y - 7 = 9
def particular_number : ℤ := 16

-- Requirement: Prove that (Y * 5) = 80 given the condition
theorem multiply_particular_number_by_5 :
  (∃ Y : ℤ, Y - 7 = 9) →
  particular_number * 5 = 80 :=
by
  intro h,
  have h1 : particular_number = 16 := by sorry,
  exact Eq.trans (congrArg (· * 5) h1) (by norm_num)

end multiply_particular_number_by_5_l53_53306


namespace maximize_annual_average_profit_l53_53661

noncomputable def equipment_cost : ℕ := 90000
noncomputable def first_year_operating_cost : ℕ := 20000
noncomputable def operating_cost_increase : ℕ := 20000
noncomputable def annual_revenue : ℕ := 110000

def operating_cost (n : ℕ) : ℕ := 
  first_year_operating_cost + (n - 1) * operating_cost_increase

def total_operating_cost (n : ℕ) : ℕ := 
  n * (first_year_operating_cost + (first_year_operating_cost + (n - 1) * operating_cost_increase)) / 2

def total_profit (n : ℕ) : ℕ :=
  n * annual_revenue - total_operating_cost n - equipment_cost

def annual_average_profit (n : ℕ) : ℕ :=
  total_profit n / n

theorem maximize_annual_average_profit : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, annual_average_profit n ≥ annual_average_profit m :=
sorry

end maximize_annual_average_profit_l53_53661


namespace grandma_age_l53_53212

variables (G1 G2 GC1 GC2 GC3 GC4 GC5 : ℕ)
variables (grandpa grandma grandchildren_avg : ℕ)

-- Conditions
def avg_group := (grandpa + grandma + (GC1 + GC2 + GC3 + GC4 + GC5)) / 7
def avg_grandchildren := (GC1 + GC2 + GC3 + GC4 + GC5) / 5
def grandma_younger_grandpa := grandpa = grandma + 1

-- Assumptions
axiom h1 : avg_group = 26
axiom h2 : avg_grandchildren = 7
axiom h3 : grandma_younger_grandpa

-- Prove Grandma's age
theorem grandma_age : grandma = 73 :=
by sorry

end grandma_age_l53_53212


namespace complex_quadrant_is_first_l53_53923

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53923


namespace range_of_k_l53_53428
noncomputable def quadratic_nonnegative (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 - 4 * x + 3 ≥ 0

theorem range_of_k (k : ℝ) : quadratic_nonnegative k ↔ k ∈ Set.Ici (4 / 3) :=
by
  sorry

end range_of_k_l53_53428


namespace sine_difference_l53_53079

noncomputable def perpendicular_vectors (θ : ℝ) : Prop :=
  let a := (Real.cos θ, -Real.sqrt 3)
  let b := (1, 1 + Real.sin θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem sine_difference (θ : ℝ) (h : perpendicular_vectors θ) : Real.sin (Real.pi / 6 - θ) = Real.sqrt 3 / 2 :=
by
  sorry

end sine_difference_l53_53079


namespace range_of_g_l53_53440

-- Define the initial vectors, \omega and the function f(x)
def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, -1)
def n (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.cos x, 2)
def f (x : ℝ) : ℝ := (2 * Real.cos x * (Real.sin x - Real.cos x) - 2) + 3

-- Transform f(x) to g(x)
def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (4 * x + Real.pi / 4)

-- Prove the range of g(x) when x ∈ [π/4, π/2] is [-sqrt(2), 1].
theorem range_of_g : ∀ x, x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → g x ∈ Set.Icc (-Real.sqrt 2) 1 :=
sorry

end range_of_g_l53_53440


namespace T_n_bounds_l53_53770

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def b_n (n : ℕ) : ℚ := 
if n ≤ 4 then 2 * n + 1
else 1 / (n * (n + 2))

noncomputable def T_n (n : ℕ) : ℚ := 
if n ≤ 4 then S_n n
else (24 : ℚ) + (1 / 2) * (1 / 5 + 1 / 6 - 1 / (n + 1 : ℚ) - 1 / (n + 2 : ℚ))

theorem T_n_bounds (n : ℕ) : 3 ≤ T_n n ∧ T_n n < 24 + 11 / 60 := by
  sorry

end T_n_bounds_l53_53770


namespace complex_power_difference_l53_53090

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 16 - (1 - i) ^ 16 = 0 := by
  sorry

end complex_power_difference_l53_53090


namespace sum_of_valid_zs_l53_53362

-- Define the problem conditions:
def valid_z (z : ℕ) : Prop :=
  z < 10 ∧ (18 + z) % 3 = 0

-- Sum the valid z values:
def sum_valid_z : ℕ :=
  (List.filter valid_z [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).sum

theorem sum_of_valid_zs : sum_valid_z = 18 :=
  by
    sorry

end sum_of_valid_zs_l53_53362


namespace correlation_problem_correctly_identified_l53_53807

/-- Proving that certain pairs of variables do not exhibit a correlation relationship -/
theorem correlation_problem_correctly_identified :
  ¬(correlation height eyesight) ∧ ¬(correlation curve_point curve_coordinates) ∧ ¬(correlation constant_speed_distance time) :=
by
  sorry

end correlation_problem_correctly_identified_l53_53807


namespace radius_of_incircle_l53_53522

theorem radius_of_incircle (r r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 4) (h₃ : r₃ = 9) :
  ( sqrt (r₁ / r) * sqrt (r₂ / r) + sqrt (r₂ / r) * sqrt (r₃ / r) + sqrt (r₃ / r) * sqrt (r₁ / r) = 1 ) → 
  r = 11 :=
by
  sorry

end radius_of_incircle_l53_53522


namespace problem_statement_l53_53531

noncomputable def a : ℝ := Real.log(3) / Real.log(2)  -- log base 2 of 3
noncomputable def b : ℝ := Real.log(3) / Real.log(0.5)  -- log base 0.5 of 3
def c : ℝ := 3^(-2)  -- 3 raised to the power of -2, which is 1/9

theorem problem_statement : a > c ∧ c > b := by
  sorry

end problem_statement_l53_53531


namespace horner_v4_using_f_l53_53701

def f (x : ℝ) : ℝ :=
  3 * x^5 + 5 * x^4 + 6 * x^3 - 8 * x^2 + 35 * x + 12

theorem horner_v4_using_f (x : ℝ) (v4 : ℝ) : 
  f x = (((((3 * x + 5) * x + 6) * x - 8) * x + 35) * x + 12) →
  x = -2 →
  v4 = (((8) * (-2) - 8) * (-2) + 35) →
  v4 = 83 :=
by
  intros hf hx heq
  rw [←hf, hx] at heq
  exact heq

end horner_v4_using_f_l53_53701


namespace number_of_senior_citizen_tickets_l53_53615

theorem number_of_senior_citizen_tickets 
    (A S : ℕ)
    (h1 : A + S = 529)
    (h2 : 25 * A + 15 * S = 9745) 
    : S = 348 := 
by
  sorry

end number_of_senior_citizen_tickets_l53_53615


namespace problem_l53_53588

variable (f g h : ℕ → ℕ)

-- Define the conditions as hypotheses
axiom h1 : ∀ (n m : ℕ), n ≠ m → h n ≠ h m
axiom h2 : ∀ y, ∃ x, g x = y
axiom h3 : ∀ n, f n = g n - h n + 1

theorem problem : ∀ n, f n = 1 := 
by 
  sorry

end problem_l53_53588


namespace shaded_area_correct_l53_53838

-- Definition of the grid dimensions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

-- Definition of the heights of the shaded regions in segments
def shaded_height (x : ℕ) : ℕ :=
if x < 4 then 2
else if x < 9 then 3
else if x < 13 then 4
else if x < 15 then 5
else 0

-- Definition for the area of the entire grid
def grid_area : ℝ := grid_width * grid_height

-- Definition for the area of the unshaded triangle
def unshaded_triangle_area : ℝ := 0.5 * grid_width * grid_height

-- Definition for the area of the shaded region
def shaded_area : ℝ := grid_area - unshaded_triangle_area

-- The theorem to be proved
theorem shaded_area_correct : shaded_area = 37.5 :=
by
  sorry

end shaded_area_correct_l53_53838


namespace num_two_digit_factors_3pow20_minus_1_l53_53713

theorem num_two_digit_factors_3pow20_minus_1 : 
  let n : ℕ := 3^20 - 1 in 
  let two_digit_factors := { x : ℕ | 10 ≤ x ∧ x < 100 ∧ x ∣ n } in 
  two_digit_factors.to_finset.card = 8 :=
by sorry

end num_two_digit_factors_3pow20_minus_1_l53_53713


namespace points_on_ray_MA_l53_53406

-- Define the distance function
def dist (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Define points A and B
variables (A B : ℝ × ℝ)

-- Define the midpoint M of segment AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define a predicate that characterizes the points P on ray MA excluding midpoint M
def on_ray_MA_except_M (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  let M := midpoint A B in
  (P ≠ M) ∧ (dist A P < dist M P ∨ (P.1 - A.1) / (M.1 - A.1) = (P.2 - A.2) / (M.2 - A.2))

-- The theorem to prove
theorem points_on_ray_MA (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  dist P B > dist P A ↔ on_ray_MA_except_M P A B :=
sorry

end points_on_ray_MA_l53_53406


namespace complex_quadrant_is_first_l53_53920

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53920


namespace weight_of_one_bowling_ball_l53_53552

-- Definitions from the problem conditions
def weight_canoe := 36
def num_canoes := 4
def num_bowling_balls := 9

-- Calculate the total weight of the canoes
def total_weight_canoes := num_canoes * weight_canoe

-- Prove the weight of one bowling ball
theorem weight_of_one_bowling_ball : (total_weight_canoes / num_bowling_balls) = 16 := by
  sorry

end weight_of_one_bowling_ball_l53_53552


namespace maxProcGenReward_is_240_l53_53445

-- Define the conditions
variable (maxCoinRunReward maxProcGenReward : ℝ)
variable (obtainedReward : ℝ := 108)
variable (percentage : ℝ := 0.9)

-- State the conditions from the problem
def condition1 : Prop := obtainedReward = percentage * maxCoinRunReward
def condition2 : Prop := maxCoinRunReward = maxProcGenReward / 2

-- State the proof problem
theorem maxProcGenReward_is_240 (h1 : condition1) (h2 : condition2) : maxProcGenReward = 240 :=
  sorry

end maxProcGenReward_is_240_l53_53445


namespace intersection_of_A_and_B_l53_53407

open Set

-- Definition of set A
def A : Set ℤ := {1, 2, 3}

-- Definition of set B
def B : Set ℤ := {x | x < -1 ∨ 0 < x ∧ x < 2}

-- The theorem to prove A ∩ B = {1}
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  -- Proof logic here
  sorry

end intersection_of_A_and_B_l53_53407


namespace germination_probability_l53_53027

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  binomial_probability 3 2 (4 / 5) = 48 / 125 := 
  by sorry

end germination_probability_l53_53027


namespace decagon_area_ratio_l53_53839

theorem decagon_area_ratio (A B C D E F G H I J M N : Point)
  (h_regular: regular_decagon A B C D E F G H I J)
  (hM_midpoint: is_midpoint M C D)
  (hN_midpoint: is_midpoint N G H) :
  area_ratio (polygon_area [A, B, C, D, M]) (polygon_area [F, G, H, I, N]) = 1 := sorry

end decagon_area_ratio_l53_53839


namespace infinite_ap_nth_power_l53_53563

theorem infinite_ap_nth_power (n : ℤ) (h : n > 3) : 
  ∃ (A : ℕ → ℕ), (∀ i, 1 ≤ i → i < n - 1 → A i > 0) ∧ 
  (∀ k, k > 0 → ∃ i j, 1 ≤ i → i < n - 1 → 1 ≤ j → j < n - 1 → A j = A i + k) ∧ 
  ∃ x : ℕ, ∏ i in finset.range (n - 1), A i = x ^ n := 
sorry

end infinite_ap_nth_power_l53_53563


namespace angle_A_value_range_of_b_plus_c_l53_53489

variable (a b c A B C : ℝ)
variable (triangle_acute : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
variable (sides_opposite_to_angles : a = sin A ∧ b = sin B ∧ c = sin C)

theorem angle_A_value
  (h : (b - 2 * c) * cos A = a - 2 * a * cos (B / 2)^2) :
  A = π / 3 := by 
  sorry

theorem range_of_b_plus_c
  (hA : A = π / 3) 
  (ha : a = sqrt 3) :
  ∃ B C, (b + c > sqrt 3 / 2 ∧ b + c ≤ sqrt 3) := by 
  sorry

end angle_A_value_range_of_b_plus_c_l53_53489


namespace ratio_of_distances_l53_53585

theorem ratio_of_distances
  (A B C X : Point)
  (hX_on_BC : X ∈ seg B C)
  (d_b : ℝ)
  (d_c : ℝ)
  (h_d_b : distance_from_point_to_line X (line A B) = d_b)
  (h_d_c : distance_from_point_to_line X (line A C) = d_c) :
  (d_b / d_c) = (dist B X * dist A C) / (dist C X * dist A B) :=
sorry

end ratio_of_distances_l53_53585


namespace sum_of_possible_values_of_x_l53_53820

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l53_53820


namespace combined_avg_of_remaining_two_subjects_l53_53246

noncomputable def avg (scores : List ℝ) : ℝ :=
  scores.foldl (· + ·) 0 / scores.length

theorem combined_avg_of_remaining_two_subjects 
  (S1_avg S2_part_avg all_avg : ℝ)
  (S1_count S2_part_count S2_total_count : ℕ)
  (h1 : S1_avg = 85) 
  (h2 : S2_part_avg = 78) 
  (h3 : all_avg = 80) 
  (h4 : S1_count = 3)
  (h5 : S2_part_count = 5)
  (h6 : S2_total_count = 7) :
  avg [all_avg * (S1_count + S2_total_count) 
       - S1_count * S1_avg 
       - S2_part_count * S2_part_avg] / (S2_total_count - S2_part_count)
  = 77.5 := by
  sorry

end combined_avg_of_remaining_two_subjects_l53_53246


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53287

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53287


namespace isotomic_conjugates_l53_53636

variable (α β γ : ℝ)
variable (a b c : ℝ)
variable (triangle : Type) (ABC : triangle)

theorem isotomic_conjugates (h_coord1 : α ≠ 0) (h_coord2 : β ≠ 0) (h_coord3 : γ ≠ 0) :
    ∃ X Y : triangle, barycentric_coords X = (α, β, γ) ∧ barycentric_coords Y = (α⁻¹, β⁻¹, γ⁻¹) ∧
    isotomic_conjugates X Y ABC := sorry

end isotomic_conjugates_l53_53636


namespace expected_rolls_365_days_l53_53699

/--
  Bob rolls a fair eight-sided die each morning. If Bob rolls a composite number, he eats sweetened cereal. 
  If he rolls a prime number, he eats unsweetened cereal. If he rolls a 1 or an 8, then he rolls again. 
  In a non-leap year, how many times is Bob expected to roll his die?
-/ 

theorem expected_rolls_365_days : 
  let die := [1, 2, 3, 4, 5, 6, 7, 8]
  let prime (n : ℕ) := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
  let composite (n : ℕ) := n = 4 ∨ n = 6
  let roll_again (n : ℕ) := n = 1 ∨ n = 8
  let prob_stop := 3/4
  let prob_roll_again := 1/4
  let E := 1
in
  365 * E = 365 :=
by
  sorry

end expected_rolls_365_days_l53_53699


namespace percent_decreases_l53_53108

-- defining initial and final costs
def call_cost_1990 := 35
def call_cost_2010 := 5
def sms_cost_1990 := 15
def sms_cost_2010 := 1

-- the percent decrease function
def percent_decrease (initial final : ℝ) : ℝ :=
  ((initial - final) / initial) * 100

theorem percent_decreases :
  (percent_decrease call_cost_1990 call_cost_2010 ≈ 85) ∧
  (percent_decrease sms_cost_1990 sms_cost_2010 ≈ 93) :=
by
  sorry

end percent_decreases_l53_53108


namespace domain_y_is_neg2_to_1_l53_53064

-- Defining the given constraints
def valid_x (x : ℝ) : Prop :=
  (0 ≤ x + 3 ∧ x + 3 ≤ 4) ∧ (0 ≤ x^2 ∧ x^2 ≤ 4)

-- Proving that the intersection of the derived intervals is the interval [-2,1]
theorem domain_y_is_neg2_to_1 : {x : ℝ | valid_x x} = set.Icc (-2:ℝ) (1:ℝ) :=
by
  sorry

end domain_y_is_neg2_to_1_l53_53064


namespace inner_hexagon_area_l53_53581

-- Define necessary conditions in Lean 4
variable (a b c d e f : ℕ)
variable (a1 a2 a3 a4 a5 a6 : ℕ)

-- Congruent equilateral triangles conditions forming a hexagon
axiom congruent_equilateral_triangles_overlap : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16

-- We want to show that the area of the inner hexagon is 38
theorem inner_hexagon_area : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16 → a = 38 :=
by
  intro h
  sorry

end inner_hexagon_area_l53_53581


namespace highest_score_of_batsman_l53_53578

theorem highest_score_of_batsman
  (avg : ℕ)
  (inn : ℕ)
  (diff_high_low : ℕ)
  (sum_high_low : ℕ)
  (avg_excl : ℕ)
  (inn_excl : ℕ)
  (h_l_avg : avg = 60)
  (h_l_inn : inn = 46)
  (h_l_diff : diff_high_low = 140)
  (h_l_sum : sum_high_low = 208)
  (h_l_avg_excl : avg_excl = 58)
  (h_l_inn_excl : inn_excl = 44) :
  ∃ H L : ℕ, H = 174 :=
by
  sorry

end highest_score_of_batsman_l53_53578


namespace jamie_avg_is_correct_l53_53252

-- Declare the set of test scores and corresponding sums
def test_scores : List ℤ := [75, 78, 82, 85, 88, 91]

-- Alex's average score
def alex_avg : ℤ := 82

-- Total test score sum
def total_sum : ℤ := test_scores.sum

theorem jamie_avg_is_correct (alex_sum : ℤ) :
    alex_sum = 3 * alex_avg →
    (total_sum - alex_sum) / 3 = 253 / 3 :=
by
  sorry

end jamie_avg_is_correct_l53_53252


namespace minimum_value_l53_53047

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
(n * (a 1 + a n)) / 2

theorem minimum_value (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  (sequence a) →
  a 1 = 1 →
  a 2 + a 4 = 10 →
  (S n) = sum_first_n_terms a n →
  ∃ n, ∀ k, k ≠ n → (2 * S n + 18) / (a n + 3) ≤ (2 * S k + 18) / (a k + 3) ∧
  (2 * S n + 18) / (a n + 3) = 13 / 3 :=
sorry

end minimum_value_l53_53047


namespace option_A_option_B_option_C_option_D_l53_53457

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l53_53457


namespace arithmetic_mean_of_primes_l53_53374

-- Define the list of numbers
def numbers : List ℕ := [31, 33, 35, 37, 39]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the prime numbers
def prime_numbers : List ℕ := numbers.filter is_prime

-- Calculate the arithmetic mean of the prime numbers
def mean (l : List ℕ) : ℕ := l.sum / l.length

-- Theorem to prove the arithmetic mean of the prime numbers in the list
theorem arithmetic_mean_of_primes :
  mean prime_numbers = 34 :=
by
  -- This skips the proof; you can fill it in as needed
  sorry

#print arithmetic_mean_of_primes

end arithmetic_mean_of_primes_l53_53374


namespace count_integer_solutions_l53_53466

theorem count_integer_solutions :
  (2 * 9^2 + 5 * 9 * -4 + 3 * (-4)^2 = 30) →
  ∃ S : Finset (ℤ × ℤ), (∀ x y : ℤ, ((2 * x ^ 2 + 5 * x * y + 3 * y ^ 2 = 30) ↔ (x, y) ∈ S)) ∧ 
  S.card = 16 :=
by sorry

end count_integer_solutions_l53_53466


namespace jeff_pencils_initial_l53_53136

def jeff_initial_pencils (J : ℝ) := J
def jeff_remaining_pencils (J : ℝ) := 0.70 * J
def vicki_initial_pencils (J : ℝ) := 2 * J
def vicki_remaining_pencils (J : ℝ) := 0.25 * vicki_initial_pencils J
def remaining_pencils (J : ℝ) := jeff_remaining_pencils J + vicki_remaining_pencils J

theorem jeff_pencils_initial (J : ℝ) (h : remaining_pencils J = 360) : J = 300 :=
by
  sorry

end jeff_pencils_initial_l53_53136


namespace jacks_walking_rate_l53_53465

def time_in_hours (hours: ℕ) (minutes: ℕ) : ℝ :=
  hours + minutes / 60.0

def walking_rate (distance: ℝ) (time: ℝ) : ℝ :=
  distance / time

theorem jacks_walking_rate :
  walking_rate 6 (time_in_hours 1 15) = 4.8 :=
by
  -- The proof would go here
  sorry

end jacks_walking_rate_l53_53465


namespace max_value_plus_count_is_65_l53_53158

def is_permutation (xs : List ℕ) : Prop :=
  xs ~ [1, 2, 3, 4, 6]

noncomputable def value (xs : List ℕ) : ℕ :=
  xs.get 0 * xs.get 1 + xs.get 1 * xs.get 2 + xs.get 2 * xs.get 3 + xs.get 3 * xs.get 4 + xs.get 4 * xs.get 0

def max_value_and_count : ℕ × ℕ :=
  let permutations := List.permutations [1, 2, 3, 4, 6]
  let values := permutations.map value
  let max_val := values.foldl max 0
  let count := values.countP (fun v => v = max_val)
  (max_val, count)

theorem max_value_plus_count_is_65 : max_value_and_count.1 + max_value_and_count.2 = 65 :=
  sorry

end max_value_plus_count_is_65_l53_53158


namespace find_Q_when_P_equal_7_l53_53724

noncomputable def Q (r P : ℝ) : ℝ := 3 * r * P - 6

theorem find_Q_when_P_equal_7 : 
  (∃ r : ℝ, Q r 5 = 27) → Q 2.2 7 = 40 :=
by
  intros h,
  sorry

end find_Q_when_P_equal_7_l53_53724


namespace complex_number_quadrant_l53_53890

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53890


namespace distance_foci_xy_eq_4_l53_53434

noncomputable def distance_between_foci_of_hyperbola (a b : ℝ) (h : a * b = 4) : ℝ :=
  real.sqrt (4 * a^2 + (16 / (a^2)))

theorem distance_foci_xy_eq_4 : distance_between_foci_of_hyperbola 2 real.sqrt 2 = 2 * real.sqrt 10 :=
by
  sorry

end distance_foci_xy_eq_4_l53_53434


namespace flippy_numbers_divisible_by_11_l53_53081

def is_flippy_number (x : ℤ) : Prop :=
  let digits := x.digits 10
  digits.length = 4 ∧ digits.nth 0 = digits.nth 2 ∧ digits.nth 1 ≠ digits.nth 3 ∧ digits.nth 3 = digits.nth 1

def is_divisible_by_11 (x : ℤ) : Prop :=
  (x % 11) = 0

def count_flippy_numbers : ℕ :=
  (List.range 9000).count (λ n, let x := n + 1000 in is_flippy_number x ∧ is_divisible_by_11 x)

theorem flippy_numbers_divisible_by_11: count_flippy_numbers = 81 := 
  sorry

end flippy_numbers_divisible_by_11_l53_53081


namespace road_trip_speed_l53_53638

theorem road_trip_speed (driving_rate_miles_per_minute : ℕ) (conversion_factor : ℝ)
  (h1 : driving_rate_miles_per_minute = 6)
  (h2 : conversion_factor = 0.6) :
  (driving_rate_miles_per_minute * 60) / conversion_factor = 600 :=
by
  rw [h1, h2]
  norm_num
  sorry

end road_trip_speed_l53_53638


namespace min_points_tenth_game_l53_53834

-- Defining the scores for each segment of games
def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

-- Calculating the total score after 9 games
def total_score_after_nine_games : ℕ := first_five_games.sum + next_four_games.sum

-- Defining the required total points after 10 games for an average greater than 17
def required_total_points := 171

-- Proving the number of points needed in the 10th game
theorem min_points_tenth_game (s₁ s₂ : List ℕ) (h₁ : s₁ = first_five_games) (h₂ : s₂ = next_four_games) :
    s₁.sum + s₂.sum + x ≥ required_total_points → x ≥ 22 :=
  sorry

end min_points_tenth_game_l53_53834


namespace guppies_eaten_by_moray_eel_l53_53135

-- Definitions based on conditions
def moray_eel_guppies_per_day : ℕ := sorry -- Number of guppies the moray eel eats per day

def number_of_betta_fish : ℕ := 5

def guppies_per_betta : ℕ := 7

def total_guppies_needed_per_day : ℕ := 55

-- Theorem based on the question
theorem guppies_eaten_by_moray_eel :
  moray_eel_guppies_per_day = total_guppies_needed_per_day - (number_of_betta_fish * guppies_per_betta) :=
sorry

end guppies_eaten_by_moray_eel_l53_53135


namespace bulbs_always_on_l53_53646

theorem bulbs_always_on (n : ℕ) (odd_n : n % 2 = 1) : 
  ∃ (initial_state : fin n → bool), 
    ∀ t : ℕ, ∃ i : fin n, bulbs_state n initial_state t i = tt :=
sorry

-- Definition for the bulbs state at time t given the initial state
def bulbs_state (n : ℕ) (initial_state : fin n → bool) (t : ℕ) (i : fin n) : bool :=
-- This function would describe the state of bulbs based on the given rule but is omitted for now.
sorry

end bulbs_always_on_l53_53646


namespace complex_point_quadrant_l53_53878

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53878


namespace tv_show_length_correct_l53_53549

noncomputable def tv_show_length (total_time : ℕ) (commercials : list ℕ) (breaks : list ℕ) : ℝ :=
  (total_time * 60 - (commercials.sum + breaks.sum)) / 60.0

theorem tv_show_length_correct : 
  tv_show_length 2 [8, 8, 12, 6, 6] [4, 5] ≈ 1.1833 := by
  sorry

end tv_show_length_correct_l53_53549


namespace evaluate_gg2_l53_53994

noncomputable def g (x : ℚ) : ℚ := 1 / (x^2) + (x^2) / (1 + x^2)

theorem evaluate_gg2 : g (g 2) = 530881 / 370881 :=
by
  sorry

end evaluate_gg2_l53_53994


namespace local_minimum_f_range_of_a_l53_53400

-- Define the function f(x) for general a.
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.log x + 2 * x

-- Define the problem for the local minimum when a = -4.
theorem local_minimum_f (x : ℝ) (h₀ : x > 0) : f (-4) x = 4 - 4 * real.log 2 := sorry

-- Define the function g(x) as mentioned in the conditions when minimum value >= -a.
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * real.log x + 2 * x + a

-- Define the theorem to determine the range of a.
theorem range_of_a (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x > 0, g a x >= 0) : -2 ≤ a ∧ a < 0 := sorry

end local_minimum_f_range_of_a_l53_53400


namespace both_taps_fill_time_l53_53639

variables (vol_A rate_B volume_bucket time_fill : ℝ)

-- Conditions
def tap_A_rate : ℝ := 3
def bucket_volume : ℝ := 36
def tap_B_filling_fraction : ℝ := 1 / 3
def tap_B_filling_time : ℝ := 20 -- in minutes

-- Derived from conditions
def tap_B_filling_volume : ℝ := tap_B_filling_fraction * bucket_volume
def tap_B_rate : ℝ := tap_B_filling_volume / tap_B_filling_time
def combined_rate : ℝ := tap_A_rate + tap_B_rate
def fill_time : ℝ := bucket_volume / combined_rate

theorem both_taps_fill_time :
  (vol_A = tap_A_rate) →
  (volume_bucket = bucket_volume) →
  (rate_B = tap_B_rate) →
  (time_fill = fill_time) →
  time_fill = 10 :=
by
  sorry

end both_taps_fill_time_l53_53639


namespace quadratic_single_solution_positive_n_l53_53032

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l53_53032


namespace triangle_vector_problem_l53_53104

/-- In triangle ABC, with AD = DB, AE = 1/2 EC, and CD intersects BE at point F,
    given that AB = a, AC = b, and AF = x * a + y * b,
    we aim to prove that the ordered pair (x, y) is (2/5, 1/5). -/
theorem triangle_vector_problem
  (A B C D E F: Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C]
  (a b : A)
  (hab : \overrightarrow{AB} = a)
  (hac : \overrightarrow{AC} = b)
  (hAD_DB : AD = DB)
  (hAE_half_EC : AE = 1/2 • EC)
  (hCD_BE_intersects_F : CD ∩ BE = {F})
  (hAF : \overrightarrow{AF} = x • a + y • b) :
  (x, y) = (2 / 5, 1 / 5) := 
sorry

end triangle_vector_problem_l53_53104


namespace tuition_fee_l53_53603

theorem tuition_fee (R T : ℝ) (h1 : T + R = 2584) (h2 : T = R + 704) : T = 1644 := by sorry

end tuition_fee_l53_53603


namespace total_pennies_l53_53472

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l53_53472


namespace digit_6_occurrences_100_to_999_l53_53955

theorem digit_6_occurrences_100_to_999 : 
  let count_digit_6 (n : ℕ) : ℕ := (n.digits 10).count (λ d, d = 6) in
  (List.range' 100 900).sum count_digit_6 = 280 := 
by
  sorry

end digit_6_occurrences_100_to_999_l53_53955


namespace percentage_decrease_l53_53598

theorem percentage_decrease (original_salary new_salary decreased_salary : ℝ) (p : ℝ) (D : ℝ) : 
  original_salary = 4000.0000000000005 →
  p = 10 →
  new_salary = original_salary * (1 + p/100) →
  decreased_salary = 4180 →
  decreased_salary = new_salary * (1 - D / 100) →
  D = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_decrease_l53_53598


namespace binomial_coefficient_l53_53118

theorem binomial_coefficient (a : ℝ) :
  (∃ k : ℕ, 10 - 3 * k = 1 ∧ 
           (∑ i in finset.range (k+1), 
              (-(a : ℝ))^i * nat.choose 5 i * (x ^ (10 - 3 * i)) = -10)) → 
  a = 1 :=
by
  sorry

end binomial_coefficient_l53_53118


namespace Roshesmina_pennies_l53_53474

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l53_53474


namespace complex_point_in_first_quadrant_l53_53908

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53908


namespace fewest_molecules_l53_53627

noncomputable def Avogadro_number : ℝ := 6.02214076e23

def num_molecules_oxygen (mol_oxygen : ℝ) : ℝ :=
  mol_oxygen * Avogadro_number

def num_molecules_ammonia (L_ammonia : ℝ) : ℝ :=
  (L_ammonia / 22.4) * Avogadro_number

def num_molecules_water (mass_water : ℝ) (molar_mass_water : ℝ) : ℝ :=
  (mass_water / molar_mass_water) * Avogadro_number

def num_molecules_hydrogen (num_molecules : ℝ) : ℝ :=
  num_molecules

theorem fewest_molecules :
  let A := num_molecules_oxygen 0.8
  let B := num_molecules_ammonia 2.24
  let C := num_molecules_water 3.6 18
  let D := num_molecules_hydrogen Avogadro_number
  B < A ∧ B < C ∧ B < D :=
by
  sorry

end fewest_molecules_l53_53627


namespace James_out_of_pocket_is_2291_63_l53_53967

/-- James' out-of-pocket calculation -/
theorem James_out_of_pocket_is_2291_63 : 
  ∀ (initial_cost : ℝ)
    (discount_rate : ℝ)
    (tax_rate : ℝ)
    (item1_cost item2_cost : ℝ)
    (exchange_rate1 exchange_rate2 exchange_rate3 : ℝ)
    (additional_cost1 additional_cost2 subs_monthly subs_discount_rate : ℝ),
  initial_cost = 5000 → discount_rate = 0.10 → tax_rate = 0.05 →
  item1_cost = 1000 → item2_cost = 700 →
  exchange_rate1 = 0.85 → exchange_rate2 = 0.87 → exchange_rate3 = 0.77 →
  additional_cost1 = 100 → additional_cost2 = 150 → 
  subs_monthly = 80 → subs_discount_rate = 0.30 →
  let discounted_cost := initial_cost * (1 - discount_rate) in
  let final_cost := discounted_cost * (1 + tax_rate) in
  let refund := item1_cost + item2_cost in
  let original_bike_cost := item2_cost in
  let other_bike_cost := original_bike_cost * 1.20 in
  let resale_price := other_bike_cost * 0.85 in
  let resale_price_gbp := resale_price / exchange_rate3 in
  let additional_expenses := (additional_cost1 + additional_cost2) / exchange_rate1 in
  let subs_first_3_months := subs_monthly * (1 - subs_discount_rate) * 3 in
  let subs_remaining := subs_monthly * 9 in
  let subs_total := (subs_first_3_months + subs_remaining) / exchange_rate3 in
  let out_of_pocket := final_cost - refund + resale_price + additional_expenses + subs_total in
  out_of_pocket ≈ 2291.63 :=
by {
  intros,
  sorry
}

end James_out_of_pocket_is_2291_63_l53_53967


namespace a1_seq_decreasing_g_seq_constant_h_seq_increasing_l53_53154

noncomputable def sequence_A (x y p q : ℝ) : ℕ → ℝ
| 0     := p * x + q * y
| (n+1) := p * sequence_A x y p q n + q * sequence_H x y p q n

noncomputable def sequence_G (x y p q : ℝ) : ℕ → ℝ
| 0     := real.sqrt (x * y)
| (n+1) := real.sqrt (sequence_A x y p q n * sequence_H x y p q n)

noncomputable def sequence_H (x y p q : ℝ) : ℕ → ℝ
| 0     := 1 / (p / x + q / y)
| (n+1) := 1 / (p / sequence_A x y p q n + q / sequence_H x y p q n)

variables {x y p q : ℝ}

-- Ensure the conditions: x, y are distinct positive reals, p, q are positive and p + q = 1
axiom hx : x > 0
axiom hy : y > 0
axiom hxy : x ≠ y
axiom hp : p > 0
axiom hq : q > 0
axiom hpq : p + q = 1

theorem a1_seq_decreasing (n : ℕ) : sequence_A x y p q (n + 1) < sequence_A x y p q n :=
sorry

theorem g_seq_constant (n : ℕ) : sequence_G x y p q (n + 1) = sequence_G x y p q n :=
sorry

theorem h_seq_increasing (n : ℕ) : sequence_H x y p q (n + 1) > sequence_H x y p q n :=
sorry

end a1_seq_decreasing_g_seq_constant_h_seq_increasing_l53_53154


namespace cycle_space_cut_space_duality_l53_53455

variables (G : Type) [Graph G] (G_star : Type) [Graph G_star]

axiom abstract_dual (G G_star : Type) [Graph G] [Graph G_star] : Prop

def cycle_space (G : Type) [Graph G] : Type := sorry -- Cycle space definition
def cut_space (G_star : Type) [Graph G_star] : Type := sorry -- Cut space definition

theorem cycle_space_cut_space_duality 
  (G : Type) [Graph G] (G_star : Type) [Graph G_star] 
  (h_dual : abstract_dual G G_star) : 
  cycle_space G = cut_space G_star := 
sorry

end cycle_space_cut_space_duality_l53_53455


namespace jame_practice_weeks_l53_53513

def cards_per_tearing : ℕ := 30
def thick_cards_per_tearing : ℕ := 25
def regular_deck_size : ℕ := 52
def thick_deck_size : ℕ := 55
def regular_decks_bought : ℕ := 27
def thick_decks_bought : ℕ := 14
def tearing_sessions_per_week : ℕ := 4

theorem jame_practice_weeks :
  let regular_cards_per_week := cards_per_tearing * (tearing_sessions_per_week / 2),
      thick_cards_per_week := thick_cards_per_tearing * (tearing_sessions_per_week / 2),
      total_cards_per_week := regular_cards_per_week + thick_cards_per_week,
      total_regular_cards := regular_decks_bought * regular_deck_size,
      total_thick_cards := thick_decks_bought * thick_deck_size,
      total_cards := total_regular_cards + total_thick_cards in
  (total_cards / total_cards_per_week) = 19 :=
by
  sorry

end jame_practice_weeks_l53_53513


namespace central_circle_radius_l53_53201

noncomputable def side_length : ℝ := 3
noncomputable def semicircle_radius : ℝ := 1.5
noncomputable def apothem : ℝ := side_length * (Real.sqrt 3 / 2)
noncomputable def central_radius : ℝ := (3 * (Real.sqrt 3 - 1)) / 2

theorem central_circle_radius :
  ∀ r : ℝ, (∀ s : ℝ, s = side_length) →
  (∀ sr : ℝ, sr = semicircle_radius) →
  (∀ ap : ℝ, ap = s * (Real.sqrt 3 / 2)) →
  r = (ap - sr) →
  r = central_radius :=
by
  intro r
  intros s side_cond sr semicircle_cond ap apothem_cond radius_cond
  sorry

end central_circle_radius_l53_53201


namespace choosing_six_adjacent_numbers_number_of_ways_no_consecutive_numbers_l53_53564

theorem choosing_six_adjacent_numbers 
  (s : Finset ℕ) 
  (h₁ : s ⊆ Finset.range 49) 
  (h₂ : s.card = 6): 
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ (a = b + 1 ∨ a = b - 1) := 
by
  -- placeholder for the actual proof
  sorry

theorem number_of_ways_no_consecutive_numbers :
  (Finset.range 44).choose 6 = Nat.choose 49 6 - Nat.choose 44 6 :=
by
  -- placeholder for the actual proof
  sorry

end choosing_six_adjacent_numbers_number_of_ways_no_consecutive_numbers_l53_53564


namespace complex_number_quadrant_l53_53895

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53895


namespace smallest_of_three_consecutive_l53_53230

theorem smallest_of_three_consecutive (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by
  sorry

end smallest_of_three_consecutive_l53_53230


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53263

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53263


namespace angle_sum_at_F_l53_53833

theorem angle_sum_at_F (x y z w v : ℝ) (h : x + y + z + w + v = 360) : 
  x = 360 - y - z - w - v := by
  sorry

end angle_sum_at_F_l53_53833


namespace arrangement_count_l53_53452

theorem arrangement_count :
  (∃ (C O₁ O₂ L₁ L₂ O₃ : Type), 
  let letters := [C, O₁, O₂, L₁, L₂, O₃] in
  (∃ (distinct: ∀ ⦃a b : Type⦄, a ≠ b → In a letters → In b letters → a ≠ b), 
  list.permutations letters).length = 720 :=
  sorry

end arrangement_count_l53_53452


namespace vector_dot_product_l53_53444

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematical statement to prove
theorem vector_dot_product : dot_product (vec_add (scalar_mul 2 vec_a) vec_b) vec_a = 6 :=
by
  -- Sorry is used to skip the proof; it's just a placeholder.
  sorry

end vector_dot_product_l53_53444


namespace magnitude_m_l53_53076

-- Definitions of the vectors and conditions
variables (a b m : ℝ × ℝ)
variables (angle_ab : ℝ) (a_mag b_mag : ℝ)

-- Assume the given conditions
def conditions := 
  angle_ab = 120 ∧
  ‖a‖ = 1 ∧
  ‖b‖ = 2 ∧
  a.1 * m.1 + a.2 * m.2 = 1 ∧
  b.1 * m.1 + b.2 * m.2 = 1

-- Statement to prove the magnitude of vector m
theorem magnitude_m (h : conditions) : ‖m‖ = real.sqrt(21) / 3 :=
sorry

end magnitude_m_l53_53076


namespace karlson_expenditure_exceeds_2000_l53_53140

theorem karlson_expenditure_exceeds_2000 :
  ∃ n m : ℕ, 25 * n + 340 * m > 2000 :=
by {
  -- proof must go here
  sorry
}

end karlson_expenditure_exceeds_2000_l53_53140


namespace complex_points_on_same_circle_l53_53543

open Complex

noncomputable theory

def same_circle (a1 a2 a3 a4 a5 : ℂ) : Prop :=
∃ (r : ℝ), r > 0 ∧ ∃ (c : ℂ), 
    dist a1 c = r ∧ dist a2 c = r ∧ dist a3 c = r ∧ 
    dist a4 c = r ∧ dist a5 c = r

theorem complex_points_on_same_circle (a1 a2 a3 a4 a5 : ℂ) 
  (q S : ℂ) (hq : q ≠ 0) (hq_ratio : a2 = a1 * q ∧ a3 = a2 * q ∧ 
  a4 = a3 * q ∧ a5 = a4 * q) (h_sum : a1 + a2 + a3 + a4 + a5 = 4 * 
  (1 / a1 + 1 / a2 + 1 / a3 + 1 / a4 + 1 / a5)) (h_real_S : S.im = 0) 
  (h_S_abs : abs S.toReal ≤ 2) : same_circle a1 a2 a3 a4 a5 :=
sorry

end complex_points_on_same_circle_l53_53543


namespace A_share_correct_l53_53342

def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def investment_C : ℝ := 10500
def total_profit : ℝ := 12700
def total_investment : ℝ := investment_A + investment_B + investment_C
def ratio_A : ℝ := investment_A / total_investment
noncomputable def share_A : ℝ := ratio_A * total_profit

theorem A_share_correct : share_A = 3810 :=
by
  sorry

end A_share_correct_l53_53342


namespace digit_6_count_in_100_to_999_l53_53951

theorem digit_6_count_in_100_to_999 : 
  (Nat.digits_count_in_range 6 100 999) = 280 := 
sorry

end digit_6_count_in_100_to_999_l53_53951


namespace complex_quadrant_check_l53_53918

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53918


namespace geometric_sequence_solution_l53_53836

-- Define the geometric sequence a_n with a common ratio q and first term a_1
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q^n

-- Given conditions in the problem
variables {a : ℕ → ℝ} {q a1 : ℝ}

-- Common ratio is greater than 1
axiom ratio_gt_one : q > 1

-- Given conditions a_3a_7 = 72 and a_2 + a_8 = 27
axiom condition1 : a 3 * a 7 = 72
axiom condition2 : a 2 + a 8 = 27

-- Defining the property that we are looking to prove a_12 = 96
theorem geometric_sequence_solution :
  geometric_sequence a a1 q →
  a 12 = 96 :=
by
  -- This part of the proof would be filled in
  -- Show the conditions and relations leading to the solution a_12 = 96
  sorry

end geometric_sequence_solution_l53_53836


namespace trajectory_of_point_P_l53_53642

noncomputable def point_path_parabola (x : ℝ) : ℝ := (1 / 3) * (3 * x - 1)^2

theorem trajectory_of_point_P :
  ∀ (C : ℝ × ℝ) (λ₁ λ₂ : ℝ),
  C.2 = C.1^2 ∧ 0 ≤ λ₁ ∧ 0 ≤ λ₂ ∧ λ₁ + λ₂ = 1 →
  ∃ P : ℝ × ℝ, 
  (P.1 ≠ 2 / 3 ∧ P.2 = point_path_parabola P.1) :=
begin
  -- Prove based on given conditions and the tangent and intersection details
  sorry
end

end trajectory_of_point_P_l53_53642


namespace cos_of_point_on_angle_and_sin_l53_53054

/-- Given point P(-sqrt(3), y) with y > 0 lies on the terminal side of angle α, 
and sin α = sqrt(3) / 4 * y, 
prove that cos α = -3 / 4. -/
theorem cos_of_point_on_angle_and_sin (y : ℝ) (h_y : y > 0)
  (h_sin : sin α = (sqrt 3 / 4) * y) : cos α = -3 / 4 :=
sorry

end cos_of_point_on_angle_and_sin_l53_53054


namespace max_value_squared_of_ratio_l53_53161

-- Definition of positive real numbers with given conditions
variables (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 

-- Main statement
theorem max_value_squared_of_ratio 
  (h_ge : a ≥ b)
  (h_eq_1 : a ^ 2 + y ^ 2 = b ^ 2 + x ^ 2)
  (h_eq_2 : b ^ 2 + x ^ 2 = (a - x) ^ 2 + (b + y) ^ 2)
  (h_range_x : 0 ≤ x ∧ x < a)
  (h_range_y : 0 ≤ y ∧ y < b)
  (h_additional_x : x = a - 2 * b)
  (h_additional_y : y = b / 2) : 
  (a / b) ^ 2 = 4 / 9 := 
sorry

end max_value_squared_of_ratio_l53_53161


namespace intersection_points_sine_log_l53_53592

def intersection_count (f g : ℝ → ℝ) : ℝ :=
  -- This function would count the number of intersection points between f and g
  sorry

def sin_function (x : ℝ) : ℝ := Real.sin x
def log_function (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem intersection_points_sine_log :
  intersection_count sin_function (log_function 2021) = 1286 :=
by sorry

end intersection_points_sine_log_l53_53592


namespace initial_population_l53_53649

theorem initial_population (P : ℕ) (h1 : P = 4600) :
  (0.8 * 0.9 * P = 3312) := 
by {
  have reduction : ℝ := 0.8 * 0.9,
  let final_population := 3312,
  have initial_population_eq : P = 4600 := h1,
  rw initial_population_eq,
  simp [reduction],
  sorry
}

end initial_population_l53_53649


namespace problem_l53_53435

noncomputable theory

def proposition (m : ℝ) : Prop := ∀ x : ℝ, x > m → x^2 > 8

theorem problem 
  (m : ℝ) 
  (h : ¬ proposition m) : 
    m = 1 ∨ m = 2 := 
sorry

end problem_l53_53435


namespace range_of_a_l53_53775

noncomputable def A : set ℝ := { x | x^2 - 3x + 2 > 0 }
noncomputable def B (a : ℝ) : set ℝ := { x | x^2 - (a + 1) * x + a ≤ 0 }

theorem range_of_a (a : ℝ) (h : a > 1) 
  (h_eq : (set.compl A) ∪ B a = B a) : 2 ≤ a :=
sorry

end range_of_a_l53_53775


namespace sum_of_square_wins_equals_losses_l53_53533

variable {n : ℕ}
variable {w l : ℕ → ℕ}

theorem sum_of_square_wins_equals_losses (hn : n ≥ 2)
  (hw_total : ∑ i in Finset.range n, w i = ∑ i in Finset.range n, l i)
  (hw_plus_l : ∀ i, w i + l i = n - 1) :
  ∑ i in Finset.range n, (w i)^2 = ∑ i in Finset.range n, (l i)^2 := 
sorry

end sum_of_square_wins_equals_losses_l53_53533


namespace distance_to_station_is_6_l53_53095

noncomputable def distance_man_walks (walking_speed1 walking_speed2 time_diff: ℝ) : ℝ :=
  let D := (time_diff * walking_speed1 * walking_speed2) / (walking_speed1 - walking_speed2)
  D

theorem distance_to_station_is_6 :
  distance_man_walks 5 6 (12 / 60) = 6 :=
by
  sorry

end distance_to_station_is_6_l53_53095


namespace range_of_m_l53_53424

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ log (1/2) x = m / (1 - m)) → 0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l53_53424


namespace CorrectChoice_l53_53690

-- Definitions for the conditions
def ConditionA : Prop :=
  ∀ (n1 n2 r: ℕ), (if r = 0 then true else n1 * n2 / r) > n1 * n2 / (r - 1)

def ConditionB : Prop :=
  ∃ c : ℕ, c > 0

def ConditionC : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, true

def ConditionD : Prop :=
  ∀ n m : ℕ, n < m → true

theorem CorrectChoice (A B C D : Prop) : C = true :=
by
  let Answer := C
  have hA : ¬ConditionA, from sorry,
  have hB : ¬ConditionB, from sorry,
  have hD : ¬ConditionD, from sorry,
  have hC : ConditionC, from sorry,
  exact hC

end CorrectChoice_l53_53690


namespace quadratic_single_solution_positive_n_l53_53033

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l53_53033


namespace chocolate_squares_remaining_l53_53675

theorem chocolate_squares_remaining (m : ℕ) : m * 6 - 21 = 45 :=
by
  sorry

end chocolate_squares_remaining_l53_53675


namespace find_denominator_l53_53233

theorem find_denominator (x : ℝ) (h: (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5) : 
  x ≈ 0.01689 :=
sorry

end find_denominator_l53_53233


namespace arithmetic_sequences_problem_l53_53439

theorem arithmetic_sequences_problem
  (a b : ℕ →ℝ)
  (S T : ℕ → ℝ)
  (hS : ∀ n, S n = ∑ i in range (n+1), a i)
  (hT : ∀ n, T n = ∑ i in range (n+1), b i)
  (h_ratio : ∀ n, S n / T n = (2 * n - 3) / (4 * n - 1)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 43 := by
sorry

end arithmetic_sequences_problem_l53_53439


namespace tom_tim_typing_ratio_l53_53630

variable (T M : ℝ)

theorem tom_tim_typing_ratio (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
sorry

end tom_tim_typing_ratio_l53_53630


namespace quadratic_coefficients_l53_53232

theorem quadratic_coefficients (b c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 3)) → 
  b = -4 ∧ c = -6 :=
by
  intro h
  -- The proof would go here, but we'll skip it.
  sorry

end quadratic_coefficients_l53_53232


namespace eccentricity_of_ellipse_l53_53765

noncomputable def ellipse_eccentricity {a b : ℝ} (h : a > b > 0) (P F1 F2 : ℝ × ℝ)
  (h1 : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1})
  (angle_F1PF2 : ∠ F1 P F2 = 120) (dist_condition : dist P F1 = 3 * dist P F2) : Real :=
  let c := sqrt ((13 / 4^2) * (dist P F2)^2) / 2 in
  c / (2 * dist P F2 / 4)

theorem eccentricity_of_ellipse {a b : ℝ} (h : a > b > 0) (P F1 F2 : ℝ × ℝ)
  (h1 : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1})
  (angle_F1PF2 : ∠ F1 P F2 = 120) (dist_condition : dist P F1 = 3 * dist P F2) :
  ellipse_eccentricity h P F1 F2 h1 angle_F1PF2 dist_condition = sqrt 13 / 4 := sorry

end eccentricity_of_ellipse_l53_53765


namespace complex_point_in_first_quadrant_l53_53906

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53906


namespace find_smaller_number_l53_53231

theorem find_smaller_number (a b : ℤ) (h₁ : a + b = 8) (h₂ : a - b = 4) : b = 2 :=
by
  sorry

end find_smaller_number_l53_53231


namespace sum_of_roots_of_equation_l53_53809

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l53_53809


namespace find_f_neg_a_l53_53755

def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem find_f_neg_a (a : ℝ) (ha : f a = 3) : f (-a) = -7 := 
by sorry

end find_f_neg_a_l53_53755


namespace regression_independence_analysis_l53_53625

theorem regression_independence_analysis (x y m : ℝ) :
  (∀ x y m, y = 0.2 * x - m ∧ (m, 1.6) ∧ 1.6 = 0.2 * m - m → m = -2) ∧ 
  (∀ χ², χ² > 0 → χ² indicates higher correlation) ∧
  (Regression analysis: deterministic relationship ∧ independence testing: significant relationship) ∧
  (Narrower residual band in scatter plot → better model fit) →
  ["A", "B", "D"] :=
  sorry

end regression_independence_analysis_l53_53625


namespace count_special_four_digit_integers_is_100_l53_53450

def count_special_four_digit_integers : Nat := sorry

theorem count_special_four_digit_integers_is_100 :
  count_special_four_digit_integers = 100 :=
sorry

end count_special_four_digit_integers_is_100_l53_53450


namespace solve_cubic_sqrt_sum_eq_one_l53_53371

theorem solve_cubic_sqrt_sum_eq_one {x : ℝ} (h1 : 3 - x ≥ 0) (h2 : x - 2 ≥ 0) :
  (∃ x, ∀ y ∈ {2, 3, 11}, x = y) ↔ (∃ x, real.cbrt (3 - x) + real.sqrt (x - 2) = 1) :=
sorry

end solve_cubic_sqrt_sum_eq_one_l53_53371


namespace range_of_lambda_l53_53441

def a : ℝ × ℝ := (-2, -1)
def b (λ : ℝ) : ℝ × ℝ := (λ, 1)

theorem range_of_lambda (λ : ℝ) :
  (-2 * λ - 1 < 0) ∧ (λ ≠ 2) ↔ (λ > -1/2) ∧ (λ ≠ 2) :=
by
  sorry

end range_of_lambda_l53_53441


namespace range_a_cos2x_minus_sinx_plus_a_eq_0_l53_53721

theorem range_a_cos2x_minus_sinx_plus_a_eq_0 (a : ℝ) :
  (∃ x : ℝ, (cos x)^2 - sin x + a = 0) ↔ -5 / 4 ≤ a ∧ a ≤ 1 := by
  sorry

end range_a_cos2x_minus_sinx_plus_a_eq_0_l53_53721


namespace combined_average_age_l53_53211

-- Define the problem conditions
def num_fifth_graders : ℕ := 40
def avg_age_fifth_graders : ℕ := 10
def num_parents : ℕ := 60
def avg_age_parents : ℕ := 35
def num_teachers : ℕ := 10
def avg_age_teachers : ℕ := 45

-- Total ages calculation
def total_age_fifth_graders : ℕ := num_fifth_graders * avg_age_fifth_graders
def total_age_parents : ℕ := num_parents * avg_age_parents
def total_age_teachers : ℕ := num_teachers * avg_age_teachers

-- Combined average age calculation
theorem combined_average_age (total_age_fifth_graders total_age_parents total_age_teachers : ℕ) (num_fifth_graders num_parents num_teachers : ℕ):
  let total_age := total_age_fifth_graders + total_age_parents + total_age_teachers,
      total_people := num_fifth_graders + num_parents + num_teachers,
      avg_age := (total_age : ℝ) / total_people 
  in avg_age = 26.82 := by
  sorry

end combined_average_age_l53_53211


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53274

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53274


namespace multiple_of_3_l53_53568

theorem multiple_of_3 (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 3 ∣ n :=
sorry

end multiple_of_3_l53_53568


namespace find_a6_l53_53120

variable {a : ℕ → ℝ}

-- Definitions derived from conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ r, ∀ n, a (n + 1) = a n * r
def quadratic_roots (a b c r1 r2 : ℝ) := r1^2 + r2^2 = b && r1 * r2 = c

-- Given conditions
axiom seq_geometric : is_geometric_sequence a
axiom roots_eq : quadratic_roots 1 (-34) 64 (a 4) (a 8) 

-- Theorem statement to prove the value of a_6
theorem find_a6 : a 6 = 8 :=
sorry

end find_a6_l53_53120


namespace bisection_exact_solution_exists_l53_53197

-- Define continuity and the bisection method conditions
variables {f : ℝ → ℝ} {a b : ℝ}

-- Assume f is continuous on [a, b] and f(a) * f(b) < 0
def bisect_condition (f : ℝ → ℝ) (a b : ℝ) :=
    continuous_on f (set.Icc a b) ∧ f a * f b < 0

-- Proof statement under the bisection method condition
theorem bisection_exact_solution_exists (h : bisect_condition f a b) : 
    ∃ c ∈ set.Icc a b, f c = 0 :=
begin
    -- Adding the sorry placeholder to skip the proof
    sorry,
end

end bisection_exact_solution_exists_l53_53197


namespace correct_options_about_lines_and_circles_l53_53623

theorem correct_options_about_lines_and_circles :
  let A := "The equation of the line passing through the point (3,4) with equal intercepts on the x and y axes is x-y-7=0."
  let B := "If the line kx-y-k-1=0 intersects the line segment with endpoints M(2,1) and N(3,2), then the range of real number k is [3/2, 2]."
  let C := "If point P(a,b) is outside the circle x^2+y^2=r^2 (r > 0) and the equation of the line l is ax+by=r^2, then the line l is tangent to the circle."
  let D := "If there are exactly 3 points on the circle (x-1)^2+y^2=4 that are at a distance of 1 from the line y=x+b, then the real number b=-1±√2."
  A is "incorrect" ∧ B is "correct" ∧ C is "incorrect" ∧ D is "correct" :=
sorry

end correct_options_about_lines_and_circles_l53_53623


namespace find_f_neg_one_l53_53826

theorem find_f_neg_one (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 → f(x) + f((1 - x^3)⁻¹^(1 : ℝ) / (3 : ℝ)) = x^3) :
  f(-1) = 1/4 :=
sorry

end find_f_neg_one_l53_53826


namespace complex_point_in_first_quadrant_l53_53868

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53868


namespace find_angle_between_vectors_l53_53042

variables (a b : ℝ^3)
variables (angle_between : ℝ)

-- Definitions matching the conditions
def vector_magnitude_a := (‖a‖ = 2)
def vector_magnitude_b := (‖b‖ = 3)
def vector_difference_magnitude := (‖a - b‖ = real.sqrt 7)
def angle_between_vectors := real.arccos ((a • b) / (‖a‖ * ‖b‖))

-- Main theorem
theorem find_angle_between_vectors
    (h1 : vector_magnitude_a)
    (h2 : vector_magnitude_b)
    (h3 : vector_difference_magnitude) :
    angle_between_vectors a b = real.pi / 3 :=
by
  sorry

end find_angle_between_vectors_l53_53042


namespace find_a2_l53_53413

variable (x : ℝ)
variable (a₀ a₁ a₂ a₃ : ℝ)
axiom condition : ∀ x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

theorem find_a2 : a₂ = 6 :=
by
  -- The proof that involves verifying the Taylor series expansion will come here
  sorry

end find_a2_l53_53413


namespace relationship_among_abc_l53_53398

noncomputable def a : ℝ := Real.logb 11 10
noncomputable def b : ℝ := (Real.logb 11 9) ^ 2
noncomputable def c : ℝ := Real.logb 10 11

theorem relationship_among_abc : b < a ∧ a < c :=
  sorry

end relationship_among_abc_l53_53398


namespace number_of_triangles_l53_53832

-- Point definitions
variables {A B C D E F G : Type}

-- Given conditions
def is_midpoint (M : Type) (X Y : Type) : Prop := -- Definition of midpoint from given conditions
sorry

def centroid (G : Type) (A B C : Type) : Prop := -- Definition of centroid from given conditions
sorry

def cyclic_quadrilateral (A E G F : Type) : Prop := -- Definition of cyclic quadrilateral from given conditions
sorry

-- Given problem
def count_non_similar_triangles (angle_BAC : ℝ) : ℕ :=
if angle_BAC ≤ 60 then 1 else 0

-- Translating the problem statement
theorem number_of_triangles (angle_BAC : ℝ) :
  ∀ (ABC_triple : (A B C : Type)),
  is_midpoint D B C →
  is_midpoint E C A →
  is_midpoint F A B →
  centroid G A B C →
  cyclic_quadrilateral A E G F →
  count_non_similar_triangles angle_BAC = 1 :=
begin
  sorry
end

end number_of_triangles_l53_53832


namespace root_expression_value_l53_53152

noncomputable def value_of_expression (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) : ℝ :=
  sorry

theorem root_expression_value (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) :
  value_of_expression p q r h1 h2 h3 = 367 / 183 :=
sorry

end root_expression_value_l53_53152


namespace sum_of_possible_values_of_x_l53_53817

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l53_53817


namespace digit_6_count_range_100_to_999_l53_53961

/-- The number of times the digit 6 is written in the integers from 100 through 999 inclusive is 280. -/
theorem digit_6_count_range_100_to_999 : 
  (∑ n in finset.Icc 100 999, (if digit 6 n then 1 else 0)) = 280 := 
sorry

end digit_6_count_range_100_to_999_l53_53961


namespace quotient_correct_l53_53295

def dividend : ℤ := 474232
def divisor : ℤ := 800
def remainder : ℤ := -968

theorem quotient_correct : (dividend + abs remainder) / divisor = 594 := by
  sorry

end quotient_correct_l53_53295


namespace min_vertices_in_hex_grid_l53_53689

-- Define a hexagonal grid and the condition on the midpoint property.
def hexagonal_grid (p : ℤ × ℤ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

-- Statement: Prove that among any 9 points in a hexagonal grid, there are two points whose midpoint is also a grid point.
theorem min_vertices_in_hex_grid :
  ∀ points : Finset (ℤ × ℤ), points.card = 9 →
  (∃ p1 p2 : (ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ 
  (∃ midpoint : ℤ × ℤ, hexagonal_grid midpoint ∧ midpoint = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))) :=
by
  intros points h_points_card
  sorry

end min_vertices_in_hex_grid_l53_53689


namespace max_non_overlapping_triangles_l53_53691

variable (L : ℝ) (n : ℕ)
def equilateral_triangle (L : ℝ) := true   -- Placeholder definition for equilateral triangle 
def non_overlapping_interior := true        -- Placeholder definition for non-overlapping condition
def unit_triangle_orientation_shift := true -- Placeholder for orientation condition

theorem max_non_overlapping_triangles (L_pos : 0 < L)
                                    (h1 : equilateral_triangle L)
                                    (h2 : ∀ i, i < n → non_overlapping_interior)
                                    (h3 : ∀ i, i < n → unit_triangle_orientation_shift) :
                                    n ≤ (2 : ℝ) / 3 * L^2 := 
by 
  sorry

end max_non_overlapping_triangles_l53_53691


namespace k_gon_covering_l53_53767

noncomputable def regular_k_gon (k : ℕ) : Type := sorry -- Definition of a k-gon
noncomputable def inradius (A : regular_k_gon k) : ℝ := sorry -- Inradius of k-gon

theorem k_gon_covering (k n : ℕ) (A : regular_k_gon k) (r : ℝ) (r' : ℝ) (B : set ℝ) :
  (B = {P | dist P 0 < r'}) →
  (r' = r * Real.sec (π / (2 * n * k))) →
  r' > r →
  (∀ P ∈ B, ∃ θ : ℝ, ∀ P_red ∈ finset.univ.filter (λ P_red, P_red = P), 
    ∃ A_moved : regular_k_gon k, A_moved = rotate A θ ∧ P_red ∈ A_moved) :=
sorry

end k_gon_covering_l53_53767


namespace complex_number_quadrant_l53_53892

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53892


namespace complex_quadrant_l53_53932

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53932


namespace calculate_a_add_b_l53_53995

theorem calculate_a_add_b (a b : ℝ) :
  (∀ x, x ≠ 0 → y = a + b / x) →
  ((∃ (a b : ℝ), y = 2 ∧ x = -2 → y = a + b / x) ∧ 
   (∃ (a b : ℝ), y = 8 ∧ x = -4 → y = a + b / x)) →
  a + b = -34 :=
begin
  sorry
end

end calculate_a_add_b_l53_53995


namespace complex_quadrant_is_first_l53_53925

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53925


namespace option_A_option_B_option_C_option_D_l53_53746

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem option_A : ∀ x : ℝ, 0 < x → f x > 0 := sorry

theorem option_B (a : ℝ) : 1 + Real.log 2 < a → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ f x1 = a ∧ f x2 = a := sorry

noncomputable def g (x : ℝ) : ℝ := f x - x

theorem option_C : ∃! x : ℝ, 0 < x ∧ g x = 0 := sorry

theorem option_D : ¬ ( ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ g x1 = 0 ∧ g x2 = 0 ) := sorry

end option_A_option_B_option_C_option_D_l53_53746


namespace rain_probability_l53_53058

theorem rain_probability :
  ∀ (rain_prob : ℚ), rain_prob = 3/4 →
  let p_no_rain := 1 - rain_prob in
  let p_no_rain_four_days := p_no_rain^4 in
  let p_rain_at_least_once := 1 - p_no_rain_four_days in
  p_rain_at_least_once = 255 / 256 :=
by
  intros rain_prob h1 
  let p_no_rain := 1 - rain_prob 
  let p_no_rain_four_days := p_no_rain^4 
  let p_rain_at_least_once := 1 - p_no_rain_four_days 
  sorry

end rain_probability_l53_53058


namespace right_triangle_acute_angles_l53_53579

theorem right_triangle_acute_angles
  (ABC : Type)
  [triangle ABC]
  (A B C D K M L : ABC)
  (h_right : right_triangle A B C)
  (h_alter : is_altitude A D on B C)
  (h_circle : ∃ circle x, by (diameter x = D))
  (h_km : intersects (circle D) B K ∧ intersects (circle D) C M)
  (h_l : intersects_line (KM) D L)
  (h_geom_prog : ∃ AK AL AM, AK/AL = AL/AM) :
  acute_angles A B C = (15°, 75°) := sorry

end right_triangle_acute_angles_l53_53579


namespace range_of_b_l53_53461

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  - (1/2) * (x - 2)^2 + b * Real.log x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x → f x b ≤ f 1 b) → b ≤ -1 :=
by
  sorry

end range_of_b_l53_53461


namespace intersection_complement_l53_53037

def U : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : set ℕ := {2, 4, 5}
def B : set ℕ := {1, 3, 5, 7}

theorem intersection_complement :
  (A ∩ (U \ B)) = {2, 4} :=
by
  sorry

end intersection_complement_l53_53037


namespace fourth_power_cube_third_smallest_prime_l53_53266

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l53_53266


namespace alfonso_required_weeks_to_save_l53_53343

variable (earn_per_day : ℕ) (days_per_week : ℕ) (current_savings : ℕ) (total_cost : ℕ)

def required_weeks (earn_per_week : ℕ) (needed_money : ℕ) : ℕ :=
  needed_money / earn_per_week

theorem alfonso_required_weeks_to_save :
  ∀ (earn_per_day days_per_week current_savings total_cost : ℕ),
  earn_per_day = 6 →
  days_per_week = 5 →
  current_savings = 40 →
  total_cost = 340 →
  required_weeks (earn_per_day * days_per_week) (total_cost - current_savings) = 10 :=
begin
  intros,
  simp only [required_weeks],
  sorry
end

end alfonso_required_weeks_to_save_l53_53343


namespace complex_point_in_first_quadrant_l53_53850

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53850


namespace complex_point_in_first_quadrant_l53_53859

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53859


namespace log_base_sqrt_10_l53_53002

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l53_53002


namespace find_X_plus_Y_l53_53651

-- Definition of the 4x4 grid condition predicate
structure Grid4x4 (X Y : ℕ) :=
  (row1 : ℕ × ℕ × ℕ × ℕ)
  (row2 : ℕ × ℕ × ℕ × ℕ)
  (row3 : ℕ × ℕ × ℕ × ℕ)
  (row4 : ℕ × ℕ × ℕ × ℕ)
  (valid_rows : ∀ r, r ∈ [row1, row2, row3, row4] → r = (1, X, _, _) → True)
  (valid_columns : ∀ c, c ∈ [column1, column2, column3, column4] → c = (1, _, _, _) → True)

-- Grid4x4 properties:
def grid_conditions (X Y : ℕ) (grid : Grid4x4 X Y) :=
  grid.row1 = (1, X, 4 - 3, 4 - X) ∧
  grid.row2 = (4, 4 - 3, 3, 4 - X) ∧
  grid.row3 = (4 - 2, 2, 3, Y) ∧
  grid.row4 = (4, 4 - X, 4 - Y, Y + 3)

-- Statement to be proved
theorem find_X_plus_Y : ∃ X Y : ℕ, grid_conditions X Y (Grid4x4.mk (1, X, _, _) (4 - X, _, _, _) (4 - Y, 2, _, Y) (4, _, _, X + Y)) ∧ (X + Y = 5) :=
by {
  let X := 3,
  let Y := 2,
  existsi X,
  existsi Y,
  -- Here we would usually provide the proof, but we leave it as sorry for now.
  have hX : X = 3 := by rfl,
  have hY : Y = 2 := by rfl,
  exact (grid_conditions X Y (Grid4x4.mk (1, 3, 4 - 3, 4 - 1) (4 - 3, _, 3, 4 - 3) (4 - 2, 2, _, 2) (4, _, _, 3 + 2))),
  sorry
}

end find_X_plus_Y_l53_53651


namespace complex_quadrant_check_l53_53911

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53911


namespace linear_function_decreasing_y1_gt_y2_l53_53500

theorem linear_function_decreasing_y1_gt_y2 (x1 x2 y1 y2 : ℝ) (h1 : y1 = -x1 + 1) (h2 : y2 = -x2 + 1) (h3 : x1 < x2) : y1 > y2 :=
by {
  rw [h1, h2], sorry
}

end linear_function_decreasing_y1_gt_y2_l53_53500


namespace cyclic_pentagon_ratios_condition_l53_53940

-- Definitions of the conditions
variables {A B C D E S F : Type} -- Points in the cyclic pentagon and F outside the pentagon
variables (cyclic_pentagon_ABCDE : true)
variables (AD_BE_intersect_S : true)
variables (F_on_CS_extension_outside_ABCDE : true)
variables (condition_1: ℝ)

-- Main proof statement
theorem cyclic_pentagon_ratios_condition
  (h1 : cyclic_pentagon_ABCDE)
  (h2 : AD_BE_intersect_S)
  (h3 : F_on_CS_extension_outside_ABCDE)
  (h4 : \(\frac{AB}{BC} \cdot \frac{CD}{DE} \cdot \frac{EF}{FA} = 1\)) :
  \(\frac{BC}{CA} \cdot \frac{AE}{EF} \cdot \frac{FD}{DB} = 1\) :=
sorry

end cyclic_pentagon_ratios_condition_l53_53940


namespace right_triangle_area_l53_53373

-- Define the condition of the right triangle.
def right_triangle_leg_hypotenuse (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the area of a right triangle.
def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

-- Define the specific problem conditions.
def problem_conditions (a c : ℕ) : Prop :=
  a = 6 ∧ c = 10

-- The theorem statement for the proof problem.
theorem right_triangle_area : 
  ∀ (a b c : ℕ), 
  problem_conditions a c ∧ right_triangle_leg_hypotenuse a b c → triangle_area a b = 24 :=
by
  intros,
  cases h with pa pc,
  cases pa,
  cases pc,
  sorry

end right_triangle_area_l53_53373


namespace vector_magnitude_l53_53055

open Real

noncomputable def vector_length (v : ℝ × ℝ × ℝ) : ℝ := 
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem vector_magnitude (a b : ℝ × ℝ × ℝ) 
  (angle_ab : dot_product a b = vector_length a * vector_length b * cos (π / 3))
  (length_b : vector_length b = 4)
  (extra_condition : dot_product (a.1 + 2 * b.1, a.2 + 2 * b.2, a.3 + 2 * b.3) 
                               (a.1 - 3 * b.1, a.2 - 3 * b.2, a.3 - 3 * b.3) = -72) : 
  vector_length a = 6 :=
by
  sorry

end vector_magnitude_l53_53055


namespace euler_disproof_l53_53558

theorem euler_disproof :
  ∃ (n : ℕ), 0 < n ∧ (133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144) :=
by
  sorry

end euler_disproof_l53_53558


namespace common_points_range_l53_53596

noncomputable def curve_C1 (R : ℝ) : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = R ^ 2 }

def curve_C2 : set (ℝ × ℝ) := 
  { p | ∃ α : ℝ, p = (2 + sin α ^ 2, sin α ^ 2) }

theorem common_points_range (R : ℝ) (h : 0 < R) :
  (∃ p : ℝ × ℝ, p ∈ curve_C1 R ∧ p ∈ curve_C2) ↔ (R ≥ sqrt 2) :=
by
  sorry

end common_points_range_l53_53596


namespace solution_A_l53_53544

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A_l53_53544


namespace jan_skips_in_5_minutes_l53_53133

theorem jan_skips_in_5_minutes 
  (original_speed : ℕ)
  (time_in_minutes : ℕ)
  (doubled : ℕ)
  (new_speed : ℕ)
  (skips_in_5_minutes : ℕ) : 
  original_speed = 70 →
  doubled = 2 →
  new_speed = original_speed * doubled →
  time_in_minutes = 5 →
  skips_in_5_minutes = new_speed * time_in_minutes →
  skips_in_5_minutes = 700 :=
by
  intros 
  sorry

end jan_skips_in_5_minutes_l53_53133


namespace sum_of_roots_l53_53815

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l53_53815


namespace cyclists_equal_distance_l53_53608

theorem cyclists_equal_distance (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (d : ℝ)
  (h_v1 : v1 = 12) (h_v2 : v2 = 16) (h_v3 : v3 = 24)
  (h_one_riding : t1 + t2 + t3 = 3) 
  (h_dist_equal : v1 * t1 = v2 * t2 ∧ v2 * t2 = v3 * t3 ∧ v1 * t1 = d) :
  d = 16 :=
by
  sorry

end cyclists_equal_distance_l53_53608


namespace inequality_solution_l53_53783

theorem inequality_solution (a c : ℝ) (h : ∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ ax^2 + 5*x + c > 0) : a + c = -7 :=
sorry

end inequality_solution_l53_53783


namespace largest_d_in_range_l53_53377

theorem largest_d_in_range (g : ℝ → ℝ) (d : ℝ) (h: ∀ x, g x = x^2 + 5 * x + d) :
  (-5) ∈ set.range g ↔ d ≤ 5 / 4 :=
by sorry

end largest_d_in_range_l53_53377


namespace lim_I_n_l53_53523

noncomputable def f_n (n : ℕ) (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  n / (r * Real.cos (Real.pi * r) + n^2 * r^3)

noncomputable def I_n (n : ℕ) : ℝ :=
  (integral (region_below (fun x y => (Real.sqrt (x^2 + y^2) ≤ 1))) (f_n n))

theorem lim_I_n : filter.tendsto (fun n => I_n n) filter.at_top (nhds 0) :=
  sorry

end lim_I_n_l53_53523


namespace shortest_distance_on_parabola_line_l53_53157

theorem shortest_distance_on_parabola_line :
  ∀ (a : ℝ),
  (∀ (y : ℝ), y = a^2 - 4a) → 
  (∀ (y : ℝ ∋ A, B),
  (∀ (x y : ℝ), y = 2x - 3) →
  real.sqrt (real.sqrt (3^2 - 6 * 3 + 3)^2 - 2 * 3 - 3) = (6 * real.sqrt 5) / 5) :=
by
  sorry

end shortest_distance_on_parabola_line_l53_53157


namespace cartesian_to_spherical_l53_53715

-- Conditions
variables (x y z : ℝ)
-- Questions
def question (x y z : ℝ) : Prop :=
  ∃ (ρ θ φ : ℝ), ρ > 0 ∧ 
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  0 ≤ φ ∧ φ ≤ Real.pi ∧
  x = ρ * Real.sin φ * Real.cos θ ∧
  y = ρ * Real.sin φ * Real.sin θ ∧
  z = ρ * Real.cos φ

-- Answer
def answer : Prop :=
  question (-3 * Real.sqrt 2) (3 * Real.sqrt 2) (-6) (6 * Real.sqrt 2) (2 * Real.pi / 3) (3 * Real.pi / 4)

-- Proof statement
theorem cartesian_to_spherical :
  answer
:= sorry

end cartesian_to_spherical_l53_53715


namespace find_land_area_l53_53663

variable (L : ℝ) -- cost of land per square meter
variable (B : ℝ) -- cost of bricks per 1000 bricks
variable (R : ℝ) -- cost of roof tiles per tile
variable (numBricks : ℝ) -- number of bricks needed
variable (numTiles : ℝ) -- number of roof tiles needed
variable (totalCost : ℝ) -- total construction cost

theorem find_land_area (h1 : L = 50) 
                       (h2 : B = 100)
                       (h3 : R = 10) 
                       (h4 : numBricks = 10000) 
                       (h5 : numTiles = 500) 
                       (h6 : totalCost = 106000) : 
                       ∃ x : ℝ, 50 * x + (numBricks / 1000) * B + numTiles * R = totalCost ∧ x = 2000 := 
by 
  use 2000
  simp [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end find_land_area_l53_53663


namespace complex_quadrant_l53_53930

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53930


namespace find_smallest_x_l53_53087

noncomputable def x_solution (x y : ℕ) : Prop :=
  0.8 = (y : ℚ) / (248 + x) ∧ (x > 0 ∧ y > 0 ∧ (y % 3 = 0))

theorem find_smallest_x : ∃ x : ℕ, ∃ y : ℕ, x_solution x y ∧ x = 2 :=
by
  use [2, 200]
  rw [x_solution, ←Rat.cast_coe_nat]
  have h₁ : (0.8 : ℚ) = 4 / 5 := by norm_cast; norm_num
  have h₂ : 248 + 2 = 250 := by norm_num
  rw [h₁, eq_comm, ←Rat.div_eq_iff] at h₁
  { norm_cast
    norm_num [h₁, h₂]
    sorry }
  norm_num

end find_smallest_x_l53_53087


namespace distinct_domino_arrangements_l53_53174

theorem distinct_domino_arrangements : 
  let n := 7, k := 2 in (Nat.choose n k) = 21 := by
  let n := 7
  let k := 2
  have h : n.choose k = 21 := by sorry
  exact h

end distinct_domino_arrangements_l53_53174


namespace angle_equality_l53_53348

variables (C A B D T P : Type*) [Inhabited C] [Inhabited A] [Inhabited B] [Inhabited D] [Inhabited T] [Inhabited P]
variables (O : Type*) [Inhabited O]
variables (h1 : CA = AB) (h2 : AB = BD) (h3 : AB is_diameter O) (h4 : CT_tangent O P)

theorem angle_equality : ∠APC = ∠DPT :=
by sorry

end angle_equality_l53_53348


namespace sum_of_valid_primes_l53_53298

-- Helper function to reverse the digits of a number.
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

-- Definition of valid primes meeting the conditions.
def valid_prime (p : ℕ) : Prop :=
  11 < p ∧ p < 90 ∧ p > 50 ∧ Prime p ∧ Prime (reverse_digits p) ∧ (reverse_digits p > 50)

-- The list of two-digit primes greater than 11 but less than 90
def primes : List ℕ := (List.range' 12 78).filter Prime

-- Extracting valid primes that meet all the conditions.
def valid_primes : List ℕ :=
  primes.filter valid_prime

theorem sum_of_valid_primes : (valid_primes.sum = 154) := by
  sorry

end sum_of_valid_primes_l53_53298


namespace indices_eq_mod_three_l53_53149

theorem indices_eq_mod_three (x : ℕ → ℤ) (n p q : ℕ) (h : n = p + q) :
  let S := λ i, (finset.range p).sum (λ j, x (i + j))
  let T := λ i, (finset.range q).sum (λ j, x (i + p + j))
  let m := λ (a b : ℤ), finset.range n |>.filter (λ i, 
                      (S i) % 3 = a ∧ (T i) % 3 = b) |>.card
  in (m 1 2) % 3 = (m 2 1) % 3 :=
by
  sorry

end indices_eq_mod_three_l53_53149


namespace geometric_sequence_arithmetic_sum_l53_53530

variable {a : ℕ → ℝ} -- the geometric sequence
variable (a1 : ℝ) -- the first term of the sequence
variable (q : ℝ) -- the common ratio of the sequence

-- The sum of the first n terms of the geometric sequence
noncomputable def S (n : ℕ) : ℝ := (finset.range n).sum (λ k, a1 * q^k)

-- Proof that if S_n is arithmetic, then q = 1
theorem geometric_sequence_arithmetic_sum (h : ∀ n : ℕ, S (n + 2) = 2 * S (n + 1) - S n) : q = 1 :=
sorry

end geometric_sequence_arithmetic_sum_l53_53530


namespace newer_train_distance_l53_53667

theorem newer_train_distance :
  (∀ (older_train_distance : ℕ), older_train_distance = 300 →
    (newer_train_distance : ℕ) =  300 + (0.30 * 300)) →
  newer_train_distance = 390 :=
by
  intros
  simp
  sorry

end newer_train_distance_l53_53667


namespace maximum_p_l53_53535

noncomputable def p (a b c : ℝ) : ℝ :=
  (2 / (a ^ 2 + 1)) - (2 / (b ^ 2 + 1)) + (3 / (c ^ 2 + 1))

theorem maximum_p (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : abc + a + c = b) : 
  p a b c ≤ 10 / 3 ∧ ∃ a b c, abc + a + c = b ∧ p a b c = 10 / 3 :=
sorry

end maximum_p_l53_53535


namespace hannah_hourly_wage_l53_53446

-- Define the conditions as constants.
constants (hours_per_week : ℕ) (late_dock : ℕ) (times_late : ℕ) (pay_received : ℕ)
constant (hourly_wage : ℕ)

-- Given conditions.
axiom h1 : hours_per_week = 18
axiom h2 : late_dock = 5
axiom h3 : times_late = 3
axiom h4 : pay_received = 525

-- The goal is to prove that the hourly wage is 30 dollars per hour.
theorem hannah_hourly_wage : hourly_wage = 30 :=
by
  -- Calculate the expected pay without any docking.
  let total_earnings := hours_per_week * hourly_wage
  -- Calculate the amount deducted due to lateness.
  let total_docked := times_late * late_dock
  -- Set up the equation representing the net pay.
  have eq1 : total_earnings - total_docked = pay_received :=
    by rw [mult_assoc, add_assoc, nat.mul_sub_right_distrib, add_comm, add_left_neg]
  -- Simplify the equation step by step as done in the solution.
  have eq2 : total_earnings = 540 :=
    by rw [eq1, h1, h2, h3, h4, mul_add, mul_comm, add_comm, add_left_comm]
  -- Divide by 18 to isolate hourly_wage.
  have eq3 : hourly_wage = 30 :=
    by rw [eq2, mul_assoc, nat.mul_div_cancel' (dec_trivial : 18 ∣ 540)]
  -- Conclude.
  exact eq3

end hannah_hourly_wage_l53_53446


namespace point_in_first_quadrant_l53_53879

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53879


namespace rational_expression_iff_sqrt_rational_l53_53727

theorem rational_expression_iff_sqrt_rational (x : ℝ) :
  (∃ q : ℚ, x + sqrt (x^2 - 4) - (1 / (x + sqrt (x^2 - 4))) = q) ↔ (∃ r : ℚ, sqrt (x^2 - 4) = r) :=
by
  sorry

end rational_expression_iff_sqrt_rational_l53_53727


namespace water_displacement_l53_53664

-- Define the conditions
def tank_radius : ℝ := 6
def tank_height : ℝ := 15
def cube_side : ℝ := 12

-- Define the volume of the cube
def cube_volume (s : ℝ) : ℝ := s^3

-- Assume that the cube's body diagonal is vertical
def cube_body_diagonal (s : ℝ) : ℝ := s * (3).sqrt

-- Define the volume of water displaced by the cube
def water_displaced (s : ℝ) : ℝ := cube_volume s

-- The main statement to be proven in Lean
theorem water_displacement :
  tank_height > cube_body_diagonal cube_side →
  let w := water_displaced cube_side in
  w = 1728 ∧ w^2 = 2985984 :=
by
  intro h
  let w := water_displaced cube_side
  have hw : w = 1728 := by
    simp [water_displaced, cube_volume, cube_side]
  have hw2 : w^2 = 2985984 := by
    simp [hw]
  exact ⟨hw, hw2⟩

end water_displacement_l53_53664


namespace complex_quadrant_check_l53_53913

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53913


namespace nancy_daily_coffees_l53_53551

variable (E I : ℕ)

theorem nancy_daily_coffees :
  (3 * E + 2.5 * I = 5.5) ∧ (20 * (3 * E + 2.5 * I) = 110) → E + I = 2 :=
by
  sorry

end nancy_daily_coffees_l53_53551


namespace smallest_integer_20p_larger_and_19p_smaller_l53_53381

theorem smallest_integer_20p_larger_and_19p_smaller :
  ∃ (N x y : ℕ), N = 162 ∧ N = 12 / 10 * x ∧ N = 81 / 100 * y :=
by
  sorry

end smallest_integer_20p_larger_and_19p_smaller_l53_53381


namespace fourth_power_of_third_smallest_prime_cube_l53_53279

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53279


namespace find_ab_l53_53792

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  a = -1 ∧ b = 2 :=
by
  sorry

end find_ab_l53_53792


namespace problem_l53_53040

theorem problem (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y + x * y = 3) :
  (0 < x * y ∧ x * y ≤ 1) ∧ (∀ z : ℝ, z = x + 2 * y → z = 4 * Real.sqrt 2 - 3) :=
by
  sorry

end problem_l53_53040


namespace intersection_of_sets_l53_53799

noncomputable def A : Set ℝ := { x | x^2 - 1 > 0 }
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem intersection_of_sets :
  A ∩ B = { x | x > 1 } :=
by {
  sorry
}

end intersection_of_sets_l53_53799


namespace triangle_shape_statements_l53_53987

theorem triangle_shape_statements (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (h : a^2 + b^2 + c^2 = ab + bc + ca) :
  (a = b ∧ b = c ∧ a = c) :=
by
  sorry 

end triangle_shape_statements_l53_53987


namespace orchid_bushes_total_l53_53236

def current_bushes : ℕ := 47
def bushes_today : ℕ := 37
def bushes_tomorrow : ℕ := 25

theorem orchid_bushes_total : current_bushes + bushes_today + bushes_tomorrow = 109 := 
by sorry

end orchid_bushes_total_l53_53236


namespace find_real_numbers_l53_53370

theorem find_real_numbers (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end find_real_numbers_l53_53370


namespace problem_solution_l53_53203

theorem problem_solution (x : ℝ) :
          ((3 * x - 4) * (x + 5) ≠ 0) → 
          (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
          (x ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioi (4 / 3)) :=
sorry

end problem_solution_l53_53203


namespace find_x0_l53_53426

noncomputable def f : ℝ → ℝ
| x := if x > 0 then log10 x else x^(-2)

theorem find_x0 (x0 : ℝ) (h : f x0 = 1) : x0 = 10 := 
by
  -- proof to be filled in
  sorry

end find_x0_l53_53426


namespace digit_6_count_in_range_l53_53948

-- Defining the range of integers and what is required to count.
def count_digit_6 (n m : ℕ) : ℕ :=
  (list.range' n (m - n + 1)).countp (λ k, k.digits 10).any (λ d, d = 6)

theorem digit_6_count_in_range :
  count_digit_6 100 999 = 280 :=
by
  sorry

end digit_6_count_in_range_l53_53948


namespace area_of_intersection_is_zero_l53_53256

-- Define the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 16
def circle2 (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the theorem to prove
theorem area_of_intersection_is_zero : 
  ∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    x1 = x2 ∧ y1 = -y2 → 
    0 = 0 :=
by
  sorry -- proof goes here

end area_of_intersection_is_zero_l53_53256


namespace non_seniors_instrument_proof_l53_53486

variable (s n : ℕ)

-- Define the conditions
def total_students := (s + n = 400)
def do_not_play_instrument := (0.5 * s + 0.25 * n = 150)

-- Define the conclusion that we want to prove
def non_seniors_play_instrument := (0.75 * n = 150)

theorem non_seniors_instrument_proof : 
  total_students s n →
  do_not_play_instrument s n →
  non_seniors_play_instrument n :=
sorry

end non_seniors_instrument_proof_l53_53486


namespace lyle_notebook_cost_l53_53670

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l53_53670


namespace angle_ZCA_ninety_degrees_l53_53559

open EuclideanGeometry 

noncomputable section

variables {A B C D O G L Z X Y : Point}
variables {l : Line}

/-- Given conditions -/
axiom H1 : Parallelogram A B C D
axiom H2 : TwoDiagonalsIntersectInCenterOfParallelogram A B C D O
axiom H3 : Length (AC A C) > Length (BD B D)
axiom H4 : Length (OA O A) = Length (OG O G)
axiom H5 : Length (OA O A) = Length (OL O L)
axiom H6 : IntersectsInExtensions O A D G
axiom H7 : IntersectsInExtensions O A B L
axiom H8 : IntersectLines B D G L Z
axiom H9 : LinePerpendicularAtPoint AC O C l Y

/-- Goal: Prove that angle ZCA = 90 degrees -/
theorem angle_ZCA_ninety_degrees : Angle (Z C A) = 90 :=
sorry -- proof not provided

end angle_ZCA_ninety_degrees_l53_53559


namespace min_n_sum_pos_l53_53848

variables {a_n : ℕ → ℝ} (d : ℝ) (n : ℕ)
def arith_seq (a_n : ℕ → ℝ) (d : ℝ) := ∀ (n : ℕ), a_n (n + 1) = a_n n + d
def S_n (a_n : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a_n i

theorem min_n_sum_pos (a_n : ℕ → ℝ) 
    (h_arith : arith_seq a_n d)
    (h66 : a_n 66 < 0) 
    (h67 : 0 < a_n 67) 
    (h_abs : a_n 67 > |a_n 66|): 
  ∃ n, S_n a_n n > 0 ∧ n = 132 := 
begin
  sorry
end

end min_n_sum_pos_l53_53848


namespace four_color_theorem_l53_53350

-- Defining the regions and colors
inductive Color
| red
| blue
| green
| yellow

structure Coloring :=
(A : Color)
(B : Color)
(C : Color)
(D : Color)
(E : Color)

def adjacent (x y : Coloring) : Prop :=
  x.A ≠ y.B ∧ x.A ≠ y.C ∧
  x.B ≠ y.C ∧ x.B ≠ y.D ∧
  x.C ≠ y.D ∧ x.C ≠ y.E ∧
  x.D ≠ y.E

-- Define the main proof problem
theorem four_color_theorem : Σ n : ℕ, n = 96 :=
  ∃ f : Coloring → Color,
    (∀ x y, adjacent x y → f x ≠ f y) ∧ 
    (∃ count : ℕ, count = 96)
by
  sorry

end four_color_theorem_l53_53350


namespace find_x_l53_53529

variables (a b c k : ℝ) (h : k ≠ 0)

theorem find_x (x y z : ℝ)
  (h1 : (xy + k) / (x + y) = a)
  (h2 : (xz + k) / (x + z) = b)
  (h3 : (yz + k) / (y + z) = c) :
  x = 2 * a * b * c * d / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) := sorry

end find_x_l53_53529


namespace triangle_is_isosceles_l53_53437

variables (A B C D M O : Type)
variables {AB BD AD BC : Type}
variables [Trapezoid ABCD] [IsBase AD BC] [Midpoint M CD] [Intersection O AC BM]

-- Given conditions
axiom AB_eq_BD : AB = BD

-- Definition of isosceles triangle
definition is_isosceles_triangle (X Y Z : Type) : Prop :=
  dist X Y = dist Y Z

-- The theorem to prove
theorem triangle_is_isosceles (ABCD_has_properties : Trapezoid ABCD)
  (M_is_midpoint : Midpoint M CD) 
  (O_is_intersection : Intersection O AC BM)
  (AB_eq_BD : AB = BD) :
  is_isosceles_triangle B O C :=
sorry

end triangle_is_isosceles_l53_53437


namespace complex_quadrant_l53_53938

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53938


namespace jane_age_problem_l53_53134

variables (J M a b c : ℕ)
variables (h1 : J = 2 * (a + b))
variables (h2 : J / 2 = a + b)
variables (h3 : c = 2 * J)
variables (h4 : M > 0)

theorem jane_age_problem (h5 : J - M = 3 * ((J / 2) - 2 * M))
                         (h6 : J - M = c - M)
                         (h7 : c = 2 * J) :
  J / M = 10 :=
sorry

end jane_age_problem_l53_53134


namespace scientific_notation_correct_l53_53109

theorem scientific_notation_correct :
  (5 * 10^(-8) = 0.00000005) :=
begin
  sorry
end

end scientific_notation_correct_l53_53109


namespace other_point_on_circle_l53_53944

noncomputable def circle_center_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  dist p (0, 0) = r

theorem other_point_on_circle (r : ℝ) (h : r = 16) (point_on_circle : circle_center_radius (16, 0) r) :
  circle_center_radius (-16, 0) r :=
by
  sorry

end other_point_on_circle_l53_53944


namespace mass_percentages_correct_l53_53016

noncomputable def mass_percentage_of_Ba (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * 137.327 + (y / 153.326) * 137.327) / (x + y) ) * 100

noncomputable def mass_percentage_of_F (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * (2 * 18.998)) / (x + y) ) * 100

noncomputable def mass_percentage_of_O (x y : ℝ) : ℝ :=
  ( ((y / 153.326) * 15.999) / (x + y) ) * 100

theorem mass_percentages_correct (x y : ℝ) :
  ∃ (Ba F O : ℝ), 
    Ba = mass_percentage_of_Ba x y ∧
    F = mass_percentage_of_F x y ∧
    O = mass_percentage_of_O x y :=
sorry

end mass_percentages_correct_l53_53016


namespace num_pos_divisors_3960_l53_53740

theorem num_pos_divisors_3960 : 
  let n := 3960 in
  let prime_factors := [(2, 3), (3, 2), (5, 1), (11, 1)] in
  ∏ (x : ℕ × ℕ) in prime_factors, (x.2 + 1) = 48 :=
by
  sorry

end num_pos_divisors_3960_l53_53740


namespace fib_series_sum_l53_53985

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

def sum_fib_series : ℝ := ∑' n, (fib n) / (10^n : ℝ)

theorem fib_series_sum : sum_fib_series = 10 / 89 := 
by
  sorry

end fib_series_sum_l53_53985


namespace tangent_line_and_distance_l53_53790

theorem tangent_line_and_distance (A : Point) (circle_center : Point) (r : ℝ) (l : Line) :
  circle_center = (0, 2) ∧ r = 2 ∧ 
  A = (1, -1) ∧ l = {p : Point | p.1 - p.2 - 2 = 0} →
  ∃ T₁ T₂ : Point, (tangent_point circle_center r A T₁) ∧ (tangent_point circle_center r A T₂) ∧
  (line_through_points T₁ T₂ = {p : Point | p.1 - 3 * p.2 + 2 = 0}) ∧
  (min_distance_to_tangent_point A T₁ circle_center r = 2) := sorry

end tangent_line_and_distance_l53_53790


namespace veranda_area_correct_l53_53216

noncomputable def area_veranda (length_room : ℝ) (width_room : ℝ) (width_veranda : ℝ) (radius_obstacle : ℝ) : ℝ :=
  let total_length := length_room + 2 * width_veranda
  let total_width := width_room + 2 * width_veranda
  let area_total := total_length * total_width
  let area_room := length_room * width_room
  let area_circle := Real.pi * radius_obstacle^2
  area_total - area_room - area_circle

theorem veranda_area_correct :
  area_veranda 18 12 2 3 = 107.726 :=
by sorry

end veranda_area_correct_l53_53216


namespace log_base_sqrt_10_l53_53001

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l53_53001


namespace isosceles_triangle_l53_53943

open EuclideanGeometry

variables {A B C D : Point}

theorem isosceles_triangle (h : (A - B) - (B - C) = (A - D) - (D - C)) : is_isosceles A B C :=
sorry

end isosceles_triangle_l53_53943


namespace remainder_of_75th_number_is_5_l53_53164

theorem remainder_of_75th_number_is_5 :
  (∃ k r, ∀ n ∈ {n : ℕ | ∃ k : ℕ, n = 8 * k + r} ∧ n = 597 → r = 5) :=
by {
  sorry
}

end remainder_of_75th_number_is_5_l53_53164


namespace digit_6_count_in_range_l53_53950

-- Defining the range of integers and what is required to count.
def count_digit_6 (n m : ℕ) : ℕ :=
  (list.range' n (m - n + 1)).countp (λ k, k.digits 10).any (λ d, d = 6)

theorem digit_6_count_in_range :
  count_digit_6 100 999 = 280 :=
by
  sorry

end digit_6_count_in_range_l53_53950


namespace possible_integer_roots_l53_53674

-- Define the general polynomial
def polynomial (b2 b1 : ℤ) (x : ℤ) : ℤ := x ^ 3 + b2 * x ^ 2 + b1 * x - 30

-- Statement: Prove the set of possible integer roots includes exactly the divisors of -30
theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | polynomial b2 b1 r = 0} = 
  {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end possible_integer_roots_l53_53674


namespace emily_trips_to_fill_tank_l53_53366

noncomputable def V_bucket (r_b : ℝ) : ℝ :=
  (2/3) * Real.pi * (r_b ^ 3)

noncomputable def V_tank (r_t : ℝ) (h_t : ℝ) : ℝ :=
  Real.pi * (r_t ^ 2) * h_t

def trips_required (V_tank : ℝ) (V_bucket : ℝ) : ℕ :=
  ⌈V_tank / V_bucket⌉

theorem emily_trips_to_fill_tank :
  let r_t := 8
  let h_t := 20
  let r_b := 6
  let V_tank := V_tank r_t h_t
  let V_bucket := V_bucket r_b
  trips_required V_tank V_bucket = 9 :=
by
  sorry

end emily_trips_to_fill_tank_l53_53366


namespace complex_div_l53_53421

theorem complex_div (z1 z2 : ℂ) (h1 : z1 = 1 - complex.i) (h2 : z2 = -2 + complex.i) :
  z2 / z1 = -3 / 2 - (1 / 2) * complex.i :=
by
  rw [h1, h2]
  -- Rationalization steps and final answer will be demonstrated in the proof.
  sorry

end complex_div_l53_53421


namespace largest_difference_l53_53185

def possible_digits_a : Set ℕ := {3, 5, 9}
def possible_digits_b : Set ℕ := {2, 3, 7}
def possible_digits_c : Set ℕ := {3, 4, 8, 9}
def possible_digits_d : Set ℕ := {2, 3, 7}
def possible_digits_e : Set ℕ := {3, 5, 9}
def possible_digits_f : Set ℕ := {1, 4, 7}
def possible_digits_g : Set ℕ := {4, 5, 9}
def possible_digits_h : Set ℕ := {2}
def possible_digits_i : Set ℕ := {4, 5, 9}

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem largest_difference (a b c d e f g h i : ℕ) 
  (ha : a ∈ possible_digits_a)
  (hb : b ∈ possible_digits_b)
  (hc : c ∈ possible_digits_c)
  (hd : d ∈ possible_digits_d)
  (he : e ∈ possible_digits_e)
  (hf : f ∈ possible_digits_f)
  (hg : g ∈ possible_digits_g)
  (hh : h ∈ possible_digits_h)
  (hi : i ∈ possible_digits_i)
  (habc : a * 100 + b * 10 + c = 923)
  (hdef : d * 100 + e * 10 + f = 394)
  (hghi : g * 100 + h * 10 + i = 529) :
  923 - 394 = 529 := 
begin
  sorry,
end

end largest_difference_l53_53185


namespace find_length_large_l53_53803

-- Define the dimensions of the smaller cuboid
def length_small := 6
def width_small := 4
def height_small := 3
def volume_small := length_small * width_small * height_small

-- Define the dimensions of the larger cuboid (with unknown length)
def width_large := 15
def height_large := 2

-- Define the number of smaller cuboids that can be formed from the larger cuboid
def num_smaller_cuboids := 7.5

-- Calculate the volume of the larger cuboid as a function of its length
def volume_large (L_large : ℕ) := L_large * width_large * height_large

-- Main theorem to be proved
theorem find_length_large (L_large : ℕ) :
  num_smaller_cuboids = (volume_large L_large) / volume_small →
  L_large = 18 :=
by
  sorry

end find_length_large_l53_53803


namespace vector_parallel_min_dot_product_l53_53071

/-- Given vectors OA and OB, prove that if k*OA + 2*OB is parallel to 2*OA - OB, 
    then k = -4 --/
theorem vector_parallel (k : ℝ) : 
  let OA := (1, 7) 
      OB := (5, 1)
  in (k * OA + 2 * OB) = λ * (2 * OA - OB) → k = -4 :=
  sorry

/-- Given a point Q on line segment OP, prove that the minimum value of the dot product of vectors QA and QB is -8 --/
theorem min_dot_product (Q : ℝ × ℝ) : 
  let OA := (1, 7) 
      OB := (5, 1)
      OP := (2, 1)
      QA := (1 - 2 * Q.1, 7 - Q.2)
      QB := (5 - 2 * Q.1, 1 - Q.2)
      dot_product := QA.1 * QB.1 + QA.2 * QB.2
  in Q.1 = 2 → (dot_product = -8) :=
  sorry

end vector_parallel_min_dot_product_l53_53071


namespace sum_of_segments_eq_radius_l53_53582

theorem sum_of_segments_eq_radius
  (O A0 A5 A1 A2 A3 A4 M N : Point)
  (h1 : diameter O A0 A5)
  (h2 : equal_arcs O A0 A1)
  (h3 : equal_arcs O A1 A2)
  (h4 : equal_arcs O A2 A3)
  (h5 : equal_arcs O A3 A4)
  (h6 : equal_arcs O A4 A5)
  (h7 : line_intersects A1 A4 (O A2) M)
  (h8 : line_intersects A1 A4 (O A3) N) :
  length (segment A2 A3) + length (segment M N) = radius O R :=
begin
  sorry
end

end sum_of_segments_eq_radius_l53_53582


namespace right_triangle_median_l53_53113

theorem right_triangle_median (A B C D : Point) (h_triangle : Triangle A B C)
  (h_right_angle : Angle A C B = 90)
  (h_median : IsMedian C D (Segment A B))
  (h_CD : Length (Segment C D) = 5) :
  Length (Segment A B) = 10 :=
sorry

end right_triangle_median_l53_53113


namespace construct_right_triangle_l53_53714

theorem construct_right_triangle (A B : Point) (p : ℝ)
  (h_AB : A ≠ B) (h_p : 0 < p ∧ p < dist A B) :
  ∃ D : Point, right_triangle A B D ∧ dist A D = p :=
by {
 sorry
}

end construct_right_triangle_l53_53714


namespace projection_correct_l53_53379

noncomputable def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3 in
((dot_uv / dot_vv) * v.1, (dot_uv / dot_vv) * v.2, (dot_uv / dot_vv) * v.3)

def vector1 : ℝ × ℝ × ℝ := (3, -5, 2)
def direction_vector : ℝ × ℝ × ℝ := (6, -3, 2)
def target_projection : ℝ × ℝ × ℝ := (222 / 49, -111 / 49, 74 / 49)

theorem projection_correct : 
  projection vector1 direction_vector = target_projection := by sorry

end projection_correct_l53_53379


namespace grape_juice_percentage_l53_53310

theorem grape_juice_percentage
  (original_mixture : ℝ)
  (percent_grape_juice : ℝ)
  (added_grape_juice : ℝ)
  (h1 : original_mixture = 50)
  (h2 : percent_grape_juice = 0.10)
  (h3 : added_grape_juice = 10)
  : (percent_grape_juice * original_mixture + added_grape_juice) / (original_mixture + added_grape_juice) * 100 = 25 :=
by
  sorry

end grape_juice_percentage_l53_53310


namespace xy_sq_is_37_over_36_l53_53824

theorem xy_sq_is_37_over_36 (x y : ℚ) (h : 2002 * (x - 1)^2 + |x - 12 * y + 1| = 0) : x^2 + y^2 = 37 / 36 :=
sorry

end xy_sq_is_37_over_36_l53_53824


namespace MNKL_is_parallelogram_l53_53613

-- Define a point structure
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a parallelogram with its vertices
structure Parallelogram :=
  (A B C D : Point)
  (O : Point) -- Center of the parallelogram
  (H1 : (A.x + C.x) / 2 = O.x ∧ (A.y + C.y) / 2 = O.y)
  (H2 : (B.x + D.x) / 2 = O.x ∧ (B.y + D.y) / 2 = O.y)

-- Define a function to check if two points are midpoints
def is_midpoint (O M K : Point) : Prop :=
  (O.x = (M.x + K.x) / 2 ∧ O.y = (M.y + K.y) / 2)

-- Define the proof problem statement
theorem MNKL_is_parallelogram
  (A B C D M K N L O : Point)
  (H1 : Parallelogram A B C D O H1 H1) -- Assuming H1 to be true
  (H2 : is_midpoint O M K)
  (H3 : is_midpoint O N L) :
  (is_midpoint O M L ∧ is_midpoint O K N) → -- The diagonals bisect each other
  parallelogram M N K L :=
sorry

end MNKL_is_parallelogram_l53_53613


namespace elderly_sample_count_l53_53658

variable (total_employees young_employees middle_aged_employees elderly_employees young_sample elderly_sample : ℕ)

def employee_conditions (total_employees = 430) (young_employees = 160) 
  (middle_aged_employees = 2 * elderly_employees) (young_sample = 32) : Prop :=
  total_employees = young_employees + middle_aged_employees + elderly_employees ∧
  Rational.Div (young_employees) (total_employees) = Rational.Div (young_sample) (sample_total_employees) ∧ 
  elderly_sample = Rational.Div (elderly_employees) (total_employees) * sample_total_employees

theorem elderly_sample_count (total_employees young_employees middle_aged_employees elderly_employees young_sample elderly_sample : ℕ) (h : employee_conditions total_employees young_employees middle_aged_employees elderly_employees young_sample elderly_sample) : 
  elderly_sample = 18 :=
sorry

end elderly_sample_count_l53_53658


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53270

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53270


namespace smallest_value_of_GIRLS_l53_53808

variable (MATH WITH GIRLS : ℕ)
variable (M W A T H G I R L S : ℕ)

axiom math_with_girls : 
  MATH + WITH = GIRLS ∧
  1000 ≤ MATH ∧ MATH < 10000 ∧
  1000 ≤ WITH ∧ WITH < 10000 ∧
  10000 ≤ GIRLS ∧ GIRLS < 100000 ∧
  ∀ d ∈ {M, W, A, T, H, G, I, R, L, S}, 0 ≤ d ∧ d < 10 ∧
  MATH = M * 1000 + A * 100 + T * 10 + H ∧
  WITH = W * 1000 + I * 100 + T * 10 + H ∧
  GIRLS = G * 10000 + I * 1000 + R * 100 + L * 10 + S ∧
  M ≠ W ∧ M ≠ A ∧ M ≠ T ∧ M ≠ H ∧ M ≠ G ∧ M ≠ I ∧ M ≠ R ∧ M ≠ L ∧ M ≠ S ∧
  W ≠ A ∧ W ≠ T ∧ W ≠ H ∧ W ≠ G ∧ W ≠ I ∧ W ≠ R ∧ W ≠ L ∧ W ≠ S ∧
  A ≠ T ∧ A ≠ H ∧ A ≠ G ∧ A ≠ I ∧ A ≠ R ∧ A ≠ L ∧ A ≠ S ∧
  T ≠ H ∧ T ≠ G ∧ T ≠ I ∧ T ≠ R ∧ T ≠ L ∧ T ≠ S ∧
  H ≠ G ∧ H ≠ I ∧ H ≠ R ∧ H ≠ L ∧ H ≠ S ∧
  G ≠ I ∧ G ≠ R ∧ G ≠ L ∧ G ≠ S ∧
  I ≠ R ∧ I ≠ L ∧ I ≠ S ∧
  R ≠ L ∧ R ≠ S ∧
  L ≠ S

theorem smallest_value_of_GIRLS : GIRLS = 10978 :=
by
  sorry

end smallest_value_of_GIRLS_l53_53808


namespace point_in_first_quadrant_l53_53883

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53883


namespace solve_for_x_l53_53650

theorem solve_for_x (x : ℝ) (h : (15 - 2 + (x / 1)) / 2 * 8 = 77) : x = 6.25 :=
by
  sorry

end solve_for_x_l53_53650


namespace complex_quadrant_check_l53_53910

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53910


namespace cube_planes_l53_53510

theorem cube_planes {k : ℕ → ℝ} (h : ∀ n, k n = n) :
  ∃ (planes : ℕ → (ℝ × ℝ × ℝ × ℝ)), 
    (∀ n, planes n = (1, 2, 4, k n)) ∧ 
    (∀ n, ∃ d, d = |(k (n + 1) - k n) / real.sqrt (1^2 + 2^2 + 4^2)| ∧ 
      d = 1 / real.sqrt 21) :=
begin
  sorry,
end

end cube_planes_l53_53510


namespace neg_A_is_square_of_int_l53_53425

theorem neg_A_is_square_of_int (x y z : ℤ) (A : ℤ) (h1 : A = x * y + y * z + z * x) 
  (h2 : A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1)) : ∃ k : ℤ, -A = k^2 :=
by
  sorry

end neg_A_is_square_of_int_l53_53425


namespace max_value_g_l53_53737

noncomputable def g (x : ℝ) := 4 * x - x ^ 4

theorem max_value_g : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ 3 :=
sorry

end max_value_g_l53_53737


namespace tan_alpha_tan_alpha_minus_pi_div_4_cos_double_alpha_l53_53410

open Real

variables (α : ℝ)
-- Given conditions: α is in the third quadrant and sin α = -3/5
hypothesis (h1 : π < α ∧ α < 3 * π / 2) (h2 : sin α = -3/5)

-- prove tan α = 3/4
theorem tan_alpha (hcos : cos α < 0) : tan α = 3/4 :=
sorry

-- prove tan(α - π/4) = -1/7
theorem tan_alpha_minus_pi_div_4 (hcos : cos α < 0) : tan (α - π/4) = -1/7 :=
sorry

-- prove cos 2α = 7/25
theorem cos_double_alpha (hcos : cos α < 0) : cos (2 * α) = 7/25 :=
sorry

end tan_alpha_tan_alpha_minus_pi_div_4_cos_double_alpha_l53_53410


namespace sum_radical_conjugate_problem_statement_l53_53709

theorem sum_radical_conjugate (a : ℝ) (b : ℝ) : (a - b) + (a + b) = 2 * a :=
by sorry

theorem problem_statement : (12 - real.sqrt 2023) + (12 + real.sqrt 2023) = 24 :=
by sorry

end sum_radical_conjugate_problem_statement_l53_53709


namespace expression_simplification_l53_53635

theorem expression_simplification (a : ℝ) (h : a ≠ 1) (h_beta : 1 = 1):
  (2^(Real.log (a) / Real.log (Real.sqrt 2)) - 
   3^((Real.log (a^2+1)) / (Real.log 27)) - 
   2 * a) / 
  (7^(4 * (Real.log (a) / Real.log 49)) - 
   5^((0.5 * Real.log (a)) / (Real.log (Real.sqrt 5))) - 1) = a^2 + a + 1 :=
by
  sorry

end expression_simplification_l53_53635


namespace complex_point_in_first_quadrant_l53_53851

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53851


namespace original_sum_invested_l53_53680

-- Define the initial sum and interest rate
def sum (P : ℝ) (r : ℝ) : Prop :=
-- Define the interest conditions
let I_changes := P * r + P * (r + 0.005) + P * r + P * (r + 0.01) + P * r in
let I_original := 5 * P * r in
-- Define the $450 difference condition with changes in interest rates
let condition1 := I_changes - I_original = 450 in
-- Define the inflation rate impact
let I_inflation_adjusted := P * (0.99 ^ 5) in

-- Theorem statement and prove the initial sum P (no proof required)
theorem original_sum_invested (P r : ℝ) (h : sum P r) : P = 30000 :=
sorry

end original_sum_invested_l53_53680


namespace line_equation_60_deg_intercept_neg1_l53_53586

theorem line_equation_60_deg_intercept_neg1 :
  ∃ k : ℝ, k = Real.tan (Real.pi / 3) ∧ ∃ b : ℝ, b = -1 ∧ ∀ x y : ℝ,
  y = k * x + b ↔ √3 * x - y - 1 = 0 :=
by
  sorry

end line_equation_60_deg_intercept_neg1_l53_53586


namespace sum_inequality_of_distinct_integers_l53_53996

theorem sum_inequality_of_distinct_integers
  (α : ℕ → ℕ)
  (h_distinct : ∀ m n, α m = α n → m = n)
  (n : ℕ) :
  ∑ k in Finset.range (n + 1), (α k) / (k^2 : ℚ) ≥ ∑ k in Finset.range (n + 1), 1 / k := 
sorry

end sum_inequality_of_distinct_integers_l53_53996


namespace middle_number_divisible_by_4_l53_53600

noncomputable def three_consecutive_cubes_is_cube (x y : ℕ) : Prop :=
  (x-1)^3 + x^3 + (x+1)^3 = y^3

theorem middle_number_divisible_by_4 (x y : ℕ) (h : three_consecutive_cubes_is_cube x y) : 4 ∣ x :=
sorry

end middle_number_divisible_by_4_l53_53600


namespace complex_point_in_first_quadrant_l53_53901

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l53_53901


namespace range_of_m_l53_53754

theorem range_of_m (m : ℝ) (h : ∀ x₁ ∈ Icc (-1 : ℝ) 3, ∃ x₂ ∈ Icc (0 : ℝ) 2, x₁^2 ≥ 2^x₂ - m) : m ≥ 1 := 
sorry

end range_of_m_l53_53754


namespace isosceles_sufficient_not_necessary_l53_53106

noncomputable def cos_rule {A B C : ℝ} (a b c : ℝ) :=
  cos C = (a^2 + b^2 - c^2) / (2 * a * b)

theorem isosceles_sufficient_not_necessary
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : ¬(a = b ∧ b = c))
  (h2 : a = 2 * b * cos C)
  (h3 : cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∃! (a = 2 * b * cos C) (h₂ : b ≠ c), 
  isosceles_sufficient_not_necessary (A B C a b c) := by {
    menas_contains_something 
  
  } sorry 

end isosceles_sufficient_not_necessary_l53_53106


namespace kaylin_is_younger_by_five_l53_53975

def Freyja_age := 10
def Kaylin_age := 33
def Eli_age := Freyja_age + 9
def Sarah_age := 2 * Eli_age
def age_difference := Sarah_age - Kaylin_age

theorem kaylin_is_younger_by_five : age_difference = 5 := 
by
  show 5 = Sarah_age - Kaylin_age
  sorry

end kaylin_is_younger_by_five_l53_53975


namespace final_direction_after_reflections_l53_53546

theorem final_direction_after_reflections (α β γ : ℝ) :
  let v := (α, β, γ)
  let v_reflected := (-α, -β, -γ)
  final_direction v = v_reflected := 
sorry

end final_direction_after_reflections_l53_53546


namespace age_of_person_A_l53_53239

-- Definitions corresponding to the conditions
variables (x y z : ℕ)
axiom sum_of_ages : x + y = 70
axiom age_difference_A_B : x - z = y
axiom age_difference_B_A_half : y - z = x / 2

-- The proof statement that needs to be proved
theorem age_of_person_A : x = 42 := by 
  -- This is where the proof would go
  sorry

end age_of_person_A_l53_53239


namespace john_total_ascent_height_l53_53972

-- Definitions based on conditions
def flights := 5
def elevation_per_flight := 20
def stairs_height := flights * elevation_per_flight
def rope_ratio := (2 : ℚ) / 3
def rope_height := rope_ratio * stairs_height
def ladder_ratio := (3 : ℚ) / 2
def ladder_height := ladder_ratio * rope_height
def rock_wall_decrement := 8
def rock_wall_height := ladder_height - rock_wall_decrement

-- Total height calculation
def total_height := stairs_height + rope_height + ladder_height + rock_wall_height

-- Proof of total height
theorem john_total_ascent_height : total_height ≈ 358.68 :=
by
  sorry

end john_total_ascent_height_l53_53972


namespace onions_total_l53_53199

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ)
  (h1 : Sara_onions = 4) (h2 : Sally_onions = 5) (h3 : Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 := by
  sorry

end onions_total_l53_53199


namespace summer_sales_correct_l53_53673

variable (T : ℕ) -- Total annual sales in million
variable (spring_sales : ℕ) -- Spring sales in million
variable (fall_sales : ℕ) -- Fall sales in million
variable (combined_winter_summer_sales : ℕ) -- Combined winter and summer sales in million 
variable (summer_sales : ℕ) -- Summer sales in million

-- Conditions
axiom total_annual_sales : 0.3 * T = 6
axiom spring_sales_eq : spring_sales = 5
axiom fall_sales_eq : fall_sales = 6
axiom combined_winter_summer_eq : combined_winter_summer_sales = 0.5 * T

-- Target statement to prove
theorem summer_sales_correct :
  T = 20 ∧ combined_winter_summer_sales = 10 ∧
  spring_sales + summer_sales + fall_sales + (combined_winter_summer_sales - summer_sales) = T →
  summer_sales = 1 :=
by
  sorry

end summer_sales_correct_l53_53673


namespace clown_blew_more_balloons_l53_53214

theorem clown_blew_more_balloons :
  ∀ (initial_balloons final_balloons additional_balloons : ℕ),
    initial_balloons = 47 →
    final_balloons = 60 →
    additional_balloons = final_balloons - initial_balloons →
    additional_balloons = 13 :=
by
  intros initial_balloons final_balloons additional_balloons h1 h2 h3
  sorry

end clown_blew_more_balloons_l53_53214


namespace socks_weight_leq_6_l53_53610

/--
  Tony's dad is very strict about the washing machine and family members are only allowed to wash 50 total ounces of clothing at a time. 
  Tony doesn't want to break the rules, so he weighs his clothes and finds that underwear weighs 4 ounces, a shirt weighs 5 ounces, shorts weigh 8 ounces, and pants weigh 10 ounces. 
  Tony is washing a pair of pants, 2 shirts, a pair of shorts, and some pairs of socks. 
  He can add 4 more pairs of underwear to the wash and not break the rule. 
  Prove that the total weight of the pairs of socks can be at most 6 ounces.
-/
theorem socks_weight_leq_6 :
  let pants := 10
  let shirt := 5
  let shorts := 8
  let underwear := 4
  let total_allowed := 50
  let planned_pants := 1
  let planned_shirts := 2
  let planned_shorts := 1
  let added_underwear := 4
  let total_weight := planned_pants * pants + planned_shirts * shirt + planned_shorts * shorts + added_underwear * underwear
  total_weight + socks <= total_allowed
  ∧ total_weight = 44
  implies socks <= 6 := 
by sorry

end socks_weight_leq_6_l53_53610


namespace measure_angle_MNY_l53_53124

variable (X Y Z M N : Type)
variables [Inhabited X] [Inhabited Y] [Inhabited Z] [Inhabited M] [Inhabited N]

/-- 
In triangle XYZ, the measure of angle XYZ is 40°. 
Line segment XM bisects angle YXZ and line segment MN bisects angle XMZ. 
Line segment NY bisects angle MNY. 
Prove that the measure of angle MNY is 110°. 
-/
theorem measure_angle_MNY (measure_angle_XYZ : ℝ)
    (measure_angle_bisect_YXZ : ℝ → ℝ)
    (measure_angle_bisect_XMZ : ℝ → ℝ)
    (measure_angle_bisect_MNY : ℝ → ℝ)
    (h1: measure_angle_XYZ = 40)
    (h2: ∀ a, measure_angle_bisect_YXZ a = a)
    (h3: ∀ b, measure_angle_bisect_XMZ b = b)
    (h4: ∀ c, measure_angle_bisect_MNY c = c):
  let a := measure_angle_bisect_YXZ / 2,
      b := measure_angle_bisect_XMZ / 2,
      c := measure_angle_bisect_MNY / 2 in
  a + b + c = 70 → 
  measure_angle_bisect_MNY 2c = 110 :=
by
  sorry

end measure_angle_MNY_l53_53124


namespace number_of_solutions_eq_2_pow_2005_l53_53759

def f₁ (x : ℝ) : ℝ := abs (1 - 2 * x)

def f_n (n : ℕ) : ℝ → ℝ
| 0       := id
| (n + 1) := f₁ ∘ f_n n

theorem number_of_solutions_eq_2_pow_2005 : set.finite { x ∈ set.Icc 0 1 | f_n 2005 x = (1 / 2) * x } ∧ 
                                             fintype.card { x ∈ set.Icc 0 1 | f_n 2005 x = (1 / 2) * x } = 2 ^ 2005 :=
sorry

end number_of_solutions_eq_2_pow_2005_l53_53759


namespace nathan_probability_l53_53183

def d6_probs : list ℚ := [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

noncomputable def probability_event
  (p : ℚ) (q : ℚ) : ℚ :=
  p * q

theorem nathan_probability :
  probability_event (1/6) (1/3) = 1/18 :=
by
  -- Here we'd provide the actual proof in Lean
  sorry

end nathan_probability_l53_53183


namespace sum_of_remainders_l53_53693

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 4) : (n % 2 + n % 9) = 4 := by
  have h1 : n % 2 = 0 := by
    sorry
  have h2 : n % 9 = 4 := by
    sorry
  rw [h1, h2]
  exact add_zero 4

end sum_of_remainders_l53_53693


namespace area_region_eq_one_l53_53702

-- Define the inequality condition
def inequality (x y : ℝ) : Prop := (y + Real.sqrt x) * (y - x^2) * Real.sqrt (1 - x) ≤ 0

-- Define the bounds for the region
def region (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1 ∧ -Real.sqrt x ≤ y ∧ y ≤ x^2

-- Define the area calculation
def calculate_area : ℝ := ∫ x in 0..1, (x^2 + Real.sqrt x)

-- The main theorem stating the area is 1
theorem area_region_eq_one : calculate_area = 1 := by
  sorry

end area_region_eq_one_l53_53702


namespace circumscribed_circle_diameter_isosceles_l53_53228

-- Given an isosceles triangle with two equal sides of length 2 and a vertex angle of 120 degrees,
-- prove that the diameter of the circumscribed circle is 4.
theorem circumscribed_circle_diameter_isosceles (a b c : ℝ) (A B C : ℝ)
  (h_isosceles : a = b) (h_side_length : a = 2) (h_angle : C = 120) :
  diameter_of_circumscribed_circle a b c = 4 :=
sorry

end circumscribed_circle_diameter_isosceles_l53_53228


namespace fourth_power_of_third_smallest_prime_cube_l53_53281

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53281


namespace sum_radical_conjugate_problem_statement_l53_53710

theorem sum_radical_conjugate (a : ℝ) (b : ℝ) : (a - b) + (a + b) = 2 * a :=
by sorry

theorem problem_statement : (12 - real.sqrt 2023) + (12 + real.sqrt 2023) = 24 :=
by sorry

end sum_radical_conjugate_problem_statement_l53_53710


namespace order_of_a_b_c_l53_53988

noncomputable def a : ℝ := Real.sqrt 0.5
noncomputable def b : ℝ := Real.sqrt 0.3
noncomputable def c : ℝ := Real.logBase 0.3 0.2

theorem order_of_a_b_c : b < a ∧ a < c := 
by
  sorry

end order_of_a_b_c_l53_53988


namespace find_b_l53_53998

def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 
    3 * x^2 - 5 
  else 
    b * x + 6

theorem find_b (b : ℝ) : 
  (∀ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, |y - x| < δ → |f y b - f x b| < ε) → 
  b = 16 / 3 :=
by
  -- Proof steps would go here, but we'll skip with 'sorry'
  sorry

end find_b_l53_53998


namespace digit_6_count_in_range_l53_53949

-- Defining the range of integers and what is required to count.
def count_digit_6 (n m : ℕ) : ℕ :=
  (list.range' n (m - n + 1)).countp (λ k, k.digits 10).any (λ d, d = 6)

theorem digit_6_count_in_range :
  count_digit_6 100 999 = 280 :=
by
  sorry

end digit_6_count_in_range_l53_53949


namespace area_ratio_midpoint_path_l53_53614

/--
Given:
1. An equilateral triangle ABC with vertices at A(0, 0), B(1, 0), and C(1/2, sqrt(3)/2).
2. Particle 1 starts at vertex A and moves clockwise at constant speed v.
3. Particle 2 starts at vertex B and moves clockwise at constant speed 2v.

Prove:
The ratio R of the area enclosed by the path traced by the midpoint of the segment joining the two particles to the area of triangle ABC is 1/16.
-/
theorem area_ratio_midpoint_path :
  let A := (0, 0 : ℝ)
  let B := (1, 0 : ℝ)
  let C := (1/2, Real.sqrt 3 / 2 : ℝ)
  let v := 1
  let two_v := 2 * v
  let area_triangle_ABC := (Real.sqrt 3 / 4) * 1 ^ 2 -- Since the side length of ABC is 1
  let area_midpoint_path := (Real.sqrt 3 / 4) * (1 / 4) ^ 2
  let ratio := area_midpoint_path / area_triangle_ABC
  ratio = 1 / 16 :=
by
  sorry

end area_ratio_midpoint_path_l53_53614


namespace log_base_36_expressed_in_terms_lg_l53_53395

theorem log_base_36_expressed_in_terms_lg (a b : ℝ) (h1 : log 10 2 = a) (h2 : log 10 3 = b) : log 36 2 = a / (2 * (a + b)) :=
by sorry

end log_base_36_expressed_in_terms_lg_l53_53395


namespace total_filled_water_balloons_l53_53176

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end total_filled_water_balloons_l53_53176


namespace maximum_value_of_complex_l53_53092

noncomputable def complex_value (z : ℂ) : ℝ :=
  abs (z - 2 * complex.I - 1)

theorem maximum_value_of_complex (x y : ℝ) (z := x + y * complex.I) (h : abs (z - 1) = 1) :
  complex_value z = 3 :=
sorry

end maximum_value_of_complex_l53_53092


namespace width_of_plot_is_correct_l53_53322

-- Definitions based on the given conditions
def cost_per_acre_per_month : ℝ := 60
def total_monthly_rent : ℝ := 600
def length_of_plot : ℝ := 360
def sq_feet_per_acre : ℝ := 43560

-- Theorems to be proved based on the conditions and the correct answer
theorem width_of_plot_is_correct :
  let number_of_acres := total_monthly_rent / cost_per_acre_per_month
  let total_sq_footage := number_of_acres * sq_feet_per_acre
  let width_of_plot := total_sq_footage / length_of_plot
  width_of_plot = 1210 :=
by 
  sorry

end width_of_plot_is_correct_l53_53322


namespace grandpa_rank_l53_53550

theorem grandpa_rank (mom dad grandpa : ℕ) 
  (h1 : mom < dad) 
  (h2 : dad < grandpa) : 
  ∀ rank: ℕ, rank = 3 := 
by
  sorry

end grandpa_rank_l53_53550


namespace angle_z_value_l53_53501

theorem angle_z_value
  (ABC BAC : ℝ)
  (h1 : ABC = 70)
  (h2 : BAC = 50)
  (h3 : ∀ BCA : ℝ, BCA + ABC + BAC = 180) :
  ∃ z : ℝ, z = 30 :=
by
  sorry

end angle_z_value_l53_53501


namespace towels_per_pack_l53_53242

open Nat

-- Define the given conditions
def packs : Nat := 9
def total_towels : Nat := 27

-- Define the property to prove
theorem towels_per_pack : total_towels / packs = 3 := by
  sorry

end towels_per_pack_l53_53242


namespace downstream_speed_l53_53333

-- Define constants based on conditions given 
def V_upstream : ℝ := 30
def V_m : ℝ := 35

-- Define the speed of the stream based on the given conditions and upstream speed
def V_s : ℝ := V_m - V_upstream

-- The downstream speed is the man's speed in still water plus the stream speed
def V_downstream : ℝ := V_m + V_s

-- Theorem to be proved
theorem downstream_speed : V_downstream = 40 :=
by
  -- The actual proof steps are omitted
  sorry

end downstream_speed_l53_53333


namespace tangent_line_slope_angle_l53_53049

theorem tangent_line_slope_angle 
  (x y: ℝ) (h1: x^2 + y^2 = 4) (M : (1, sqrt(3))) 
  (l_slope : ℝ) (theta : ℝ) 
  (h2 : l_slope = -sqrt(3)/3) :
  tan(theta) = -sqrt(3)/3 → theta = 5 * pi / 6 :=
sorry

end tangent_line_slope_angle_l53_53049


namespace solve_x_plus_Sx_eq_2001_l53_53986

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem solve_x_plus_Sx_eq_2001 (x : ℕ) (h : x + sum_of_digits x = 2001) : x = 1977 :=
  sorry

end solve_x_plus_Sx_eq_2001_l53_53986


namespace desargues_theorem_l53_53167

structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

def intersects (l1 l2 : Line) : Point := sorry

def collinear (p1 p2 p3 : Point) : Prop := (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

theorem desargues_theorem (O A1 A2 B1 B2 C1 C2 : Point)
  (a b c : Line)
  (ha1 : a.p1 = A1) (ha2 : a.p2 = A2)
  (hb1 : b.p1 = B1) (hb2 : b.p2 = B2)
  (hc1 : c.p1 = C1) (hc2 : c.p2 = C2)
  (habcint : a.p1 = b.p1 ∧ b.p1 = c.p1 ∧ a.p2 = b.p2 ∧ b.p2 = c.p2) :
  let A := intersects {p1 := B1, p2 := C1} {p1 := B2, p2 := C2}
  let B := intersects {p1 := C1, p2 := A1} {p1 := C2, p2 := A2}
  let C := intersects {p1 := A1, p2 := B1} {p1 := A2, p2 := B2}
  in collinear A B C := sorry

end desargues_theorem_l53_53167


namespace mark_boxes_sold_l53_53168

theorem mark_boxes_sold (n : ℕ) (M A : ℕ) (h1 : A = n - 2) (h2 : M + A < n) (h3 :  1 ≤ M) (h4 : 1 ≤ A) (hn : n = 12) : M = 1 :=
by
  sorry

end mark_boxes_sold_l53_53168


namespace complex_point_in_first_quadrant_l53_53861

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53861


namespace M_inter_N_l53_53074

def M : Set ℝ := { y | y > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem M_inter_N : M ∩ N = { z | 1 < z ∧ z < 2 } :=
by 
  sorry

end M_inter_N_l53_53074


namespace sin_angle_BAD_max_value_l53_53612

theorem sin_angle_BAD_max_value 
  (A B C D : Type) [RealConst A B C] 
  (hC : ∠A B C = 45) 
  (hBC : dist B C = 6) 
  (hD : midpoint D B C)
  : sin (angle A B D) = (sqrt 2) / 2 * ((sqrt (2 + sqrt 2)) / 2 - (sqrt (2 - sqrt 2)) / 2) := 
sorry

end sin_angle_BAD_max_value_l53_53612


namespace point_in_first_quadrant_l53_53885

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l53_53885


namespace incorrect_plane_vector_statements_l53_53302

noncomputable def statement_A (a b : Vector ℝ) : Prop :=
|a| = |b| → a = b

noncomputable def statement_B (a b c : Vector ℝ) : Prop :=
(a ∥ b) ∧ (b ∥ c) → a ∥ c

noncomputable def statement_C (a b c : Vector ℝ) [Nonzero a] [Nonzero b] [Nonzero c] : Prop :=
(a • b = a • c) → b = c

noncomputable def statement_D (a b : Vector ℝ) : Prop :=
a = b → (|a| = |b| ∧ a ∥ b)

theorem incorrect_plane_vector_statements (a b c : Vector ℝ) :
  ¬ statement_A a b ∧ ¬ statement_B a b c ∧ ¬ statement_C a b c :=
by sorry

end incorrect_plane_vector_statements_l53_53302


namespace unique_monotonically_increasing_sequence_l53_53013

def number_of_divisors (n : ℕ) : ℕ := {d | d ∣ n}.card

theorem unique_monotonically_increasing_sequence (a : ℕ → ℕ) :
  (∀ i j, number_of_divisors (i + j) = number_of_divisors (a i + a j)) ∧
  (∀ i j, i < j → a i < a j) → 
  ∀ i, a i = i := 
by
  sorry

end unique_monotonically_increasing_sequence_l53_53013


namespace log10_a2019_to_integer_l53_53360

def diamondsuit (a b : ℝ) := a ^ (Real.log10 b)
def heartsuit (a b : ℝ) := a ^ (1 / (Real.log10 b))

sequence an : ℕ → ℝ
| 3          := heartsuit 3 2
| (n + 1)    := diamondsuit (heartsuit (an n) (n + 3)) 2

theorem log10_a2019_to_integer :
  Real.log10 (sequence an 2019).floor = 10 := sorry

end log10_a2019_to_integer_l53_53360


namespace fibonacci_identity_l53_53418

-- Define the Fibonacci sequence.
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- The theorem to be proved.
theorem fibonacci_identity (n : ℕ) : 
  fib (2 * n) = (fib (n - 1))^2 + (fib n)^2 :=
sorry

end fibonacci_identity_l53_53418


namespace transformed_equation_correct_l53_53611
-- Import the necessary library

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation functions for the transformations
def translate_right (x : ℝ) : ℝ := x - 1
def translate_down (y : ℝ) : ℝ := y - 3

-- Define the transformed parabola equation
def transformed_parabola (x : ℝ) : ℝ := -2 * (translate_right x)^2 |> translate_down

-- The theorem stating the transformed equation
theorem transformed_equation_correct :
  ∀ x, transformed_parabola x = -2 * (x - 1)^2 - 3 :=
by { sorry }

end transformed_equation_correct_l53_53611


namespace positive_n_for_modulus_eq_l53_53747

theorem positive_n_for_modulus_eq (n : ℕ) (h_pos : 0 < n) (h_eq : Complex.abs (5 + (n : ℂ) * Complex.I) = 5 * Real.sqrt 26) : n = 25 :=
by
  sorry

end positive_n_for_modulus_eq_l53_53747


namespace complex_quadrant_l53_53936

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53936


namespace total_adoption_cost_l53_53511

theorem total_adoption_cost :
  let cat_adoption_cost := 50
  let dog_adoption_cost := 100
  let puppy_adoption_cost := 150
  let num_cats := 2
  let num_dogs := 3
  let num_puppies := 2
  let total_cost := num_cats * cat_adoption_cost + num_dogs * dog_adoption_cost + num_puppies * puppy_adoption_cost
  in total_cost = 700 :=
by
  sorry

end total_adoption_cost_l53_53511


namespace num_divisors_3960_l53_53738

theorem num_divisors_3960 : 
  ∃ d : ℕ, (3960 = 2^3 * 3^2 * 5^1 * 11^1) ∧
            (d = (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)) ∧
            (d = 48) :=
begin
  use 48,
  split,
  { -- 3960 factorization
    sorry
  },
  split,
  { -- divisor function calculation: (3 + 1)(2 + 1)(1 + 1)(1 + 1)
    sorry
  },
  { -- final result
    refl
  }
end

end num_divisors_3960_l53_53738


namespace evaluate_at_5_l53_53991

def f(x: ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 27 * x^3 - 20 * x^2 - 72 * x + 40

theorem evaluate_at_5 : f 5 = 2515 :=
by
  sorry

end evaluate_at_5_l53_53991


namespace max_n_equals_15_l53_53394

-- Let's define the sum of digits function for a nonnegative integer
def sum_of_digits (x : ℕ) : ℕ :=
  (x.to_string.data.map (λ c, c.to_nat - '0'.to_nat)).sum

-- Given conditions
variables {m n : ℕ}

-- Within the sequence m to m+n, S(m) and S(m+n) are divisible by 8
axiom S_m_div_8 : sum_of_digits m % 8 = 0
axiom S_mn_div_8 : sum_of_digits (m + n) % 8 = 0

-- For all k from 1 to n-1, S(m+k) is not divisible by 8
axiom S_mk_not_div_8 : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → sum_of_digits (m + k) % 8 ≠ 0

-- The theorem to prove the maximum value of n
theorem max_n_equals_15 : n = 15 :=
by
  sorry

end max_n_equals_15_l53_53394


namespace sum_of_m_for_minimum_area_l53_53595

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem sum_of_m_for_minimum_area :
  let A := (2 : ℝ, 5 : ℝ),
      B := (10 : ℝ, 9 : ℝ),
      m1 := 6,
      m2 := 8
  in
  (triangle_area A B (6, m1) = 1 ∧ triangle_area A B (6, m2) = 1) →
  m1 + m2 = 14 := by
  intros
  sorry

end sum_of_m_for_minimum_area_l53_53595


namespace smallest_surface_area_of_glued_cubes_l53_53607

def volume_s1 := 1
def volume_s2 := 8
def volume_s3 := 27

def side_length (v : ℕ) := v ^ (1 / 3 : ℝ)

def surface_area (s : ℝ) := 6 * s^2

def initial_surface_area : ℝ :=
  (surface_area (side_length volume_s1)) +
  (surface_area (side_length volume_s2)) +
  (surface_area (side_length volume_s3))

def glued_reduction_area : ℝ := 16

noncomputable def minimum_surface_area := initial_surface_area - glued_reduction_area

theorem smallest_surface_area_of_glued_cubes :
  minimum_surface_area = 72 := by
  sorry

end smallest_surface_area_of_glued_cubes_l53_53607


namespace height_percentage_increase_l53_53823

theorem height_percentage_increase (B A : ℝ) 
  (hA : A = B * 0.8) : ((B - A) / A) * 100 = 25 := by
--   Given the condition that A's height is 20% less than B's height
--   translate into A = B * 0.8
--   We need to show ((B - A) / A) * 100 = 25
sorry

end height_percentage_increase_l53_53823


namespace count_non_empty_subsets_l53_53082

theorem count_non_empty_subsets (S : Finset ℕ) (hS : S = Finset.range 1 9) :
  (Finset.range 1 (2 ^ 8 - 1)).card = 127 := 
by
  sorry

end count_non_empty_subsets_l53_53082


namespace length_BC_fraction_of_AD_l53_53192

-- Define variables and conditions
variables (x y : ℝ)
variable (h1 : 4 * x = 8 * y) -- given: length of AD from both sides
variable (h2 : 3 * x) -- AB = 3 * BD
variable (h3 : 7 * y) -- AC = 7 * CD

-- State the goal to prove
theorem length_BC_fraction_of_AD (x y : ℝ) (h1 : 4 * x = 8 * y) :
  (y / (4 * x)) = 1 / 8 := by
  sorry

end length_BC_fraction_of_AD_l53_53192


namespace new_average_after_multiplication_l53_53575

-- Definitions based on the conditions
def initial_average (numbers : List ℝ) := (numbers.sum) / (numbers.length)
def multiply_by_factor (x : ℝ) (factor : ℝ) := x * factor
def new_sum (numbers : List ℝ) (old_value new_value : ℝ) := numbers.sum - old_value + new_value
def new_average (sum : ℝ) (count : ℕ) := sum / count

-- Problem restated in Lean 4: Prove the new average is 9.2 based on given conditions.
theorem new_average_after_multiplication :
  ∀ (numbers : List ℝ) (initial_average_value : ℝ) (count : ℕ) (specific_number : ℝ) (factor : ℝ),
  numbers.length = 5 ∧
  initial_average_value = 6.8 ∧
  specific_number = 12 ∧
  factor = 2 ∧
  initial_average numbers = initial_average_value →
  multiply_by_factor specific_number factor =
  24 →
  new_average (new_sum numbers specific_number (multiply_by_factor specific_number factor)) count = 9.2 :=
by
  intros,
  sorry

end new_average_after_multiplication_l53_53575


namespace minimal_distance_l53_53983

-- Define point on the circle (x - 8)^2 + y^2 = 9
def is_on_circle (A : Point) : Prop :=
  (A.x - 8)^2 + A.y^2 = 9

-- Define point on the parabola y^2 = 16x
def is_on_parabola (B : Point) : Prop :=
  B.y^2 = 16 * B.x

-- Define the distance between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the coordinates of the circle center
def circle_center : Point := ⟨8, 0⟩

-- State the smallest possible distance AB
theorem minimal_distance {A B : Point} (hA : is_on_circle A) (hB : is_on_parabola B) :
  distance A B ≥ 4 * real.sqrt 13 := 
sorry

end minimal_distance_l53_53983


namespace eval_f_sqrt3_div_3_l53_53062

def f : ℝ → ℝ :=
λ x, if x > 1 then 2^(x - 1) else Real.tan (Real.pi * x / 3)

theorem eval_f_sqrt3_div_3 :
  f (1 / f 2) = Real.sqrt 3 / 3 :=
by
  sorry

end eval_f_sqrt3_div_3_l53_53062


namespace cells_after_10_days_l53_53657

theorem cells_after_10_days :
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  a_n = 64 :=
by
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  show a_n = 64
  sorry

end cells_after_10_days_l53_53657


namespace complex_point_quadrant_l53_53877

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53877


namespace sin_double_angle_l53_53038

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end sin_double_angle_l53_53038


namespace units_digit_of_sum_sequence_l53_53299

def units_digit (n : ℕ) : ℕ := n % 10

def sequence_term (n : ℕ) : ℕ := n! + n^2

def units_sum_sequence : ℕ :=
  (units_digit (sequence_term 1) +
   units_digit (sequence_term 2) +
   units_digit (sequence_term 3) +
   units_digit (sequence_term 4) +
   units_digit (sequence_term 5) +
   units_digit (sequence_term 6) +
   units_digit (sequence_term 7) +
   units_digit (sequence_term 8) +
   units_digit (sequence_term 9) +
   units_digit (sequence_term 10)) % 10

theorem units_digit_of_sum_sequence :
  units_sum_sequence = 8 :=
by sorry

end units_digit_of_sum_sequence_l53_53299


namespace sum_of_roots_of_equation_l53_53811

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l53_53811


namespace correct_answer_l53_53648

-- Define the conditions
def condition1 : Prop := "One should make full use of the obtained data to provide more accurate information" = true
def condition2 : Prop := "Multiple values can be used to describe the degree of dispersion of data" = true

-- Define the main theorem to be proved
theorem correct_answer : condition1 ∧ condition2 :=
by
  sorry

end correct_answer_l53_53648


namespace shaded_area_square_and_triangle_l53_53678

theorem shaded_area_square_and_triangle :
  let A := (0, 12)
  let B := (0, 0)
  let C := (12, 0)
  let D := (12, 12)
  let E := (24, 0)
  let area (t : (ℕ × ℕ) → (ℕ × ℕ) → (ℕ × ℕ) → ℕ)
    := λ (v1 v2 v3 : (ℕ × ℕ)), abs ((v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)) / 2)
  (area_shaded : ℕ := area A D E)
  (area_triangle_CDE := area C D E) :
  area_shaded = area_triangle_CDE / 2 :=
by
  sorry

end shaded_area_square_and_triangle_l53_53678


namespace demon_absent_l53_53141

-- Define types for Knight, Liar, and possibly Demon
inductive Person
| knight
| liar
| demon

open Person

noncomputable def no_demon_at_table (n : ℕ) (responses : ℕ → Prop) : Prop :=
  (∀ i, responses i ↔ (match n with
  | knight => responses ((i + n - 1) % n)
  | liar => ¬ responses ((i + n - 1) % n)
  end)) → n

theorem demon_absent (n : ℕ) 
  (number_of_liar_statements number_of_knight_statements : ℕ)
  (responses : ℕ → Prop)
  (H_knight_truth : ∀ i, (Is_knight i) → responses i = (is_liar ((i + n - 1) % n)))
  (H_liar_lies : ∀ i, (Is_liar i) → responses i = ¬ (is_knight ((i + n - 1) % n))):
  number_of_liar_statements = 10 →
  number_of_knight_statements = (n - 10) →
  ¬ (∃ demon_present, demon_present) :=
begin
  sorry
end


end demon_absent_l53_53141


namespace polynomial_iteration_fixed_point_l53_53980

noncomputable def polynomial_with_integer_coefficients (P : ℤ[X]) : Prop :=
  ∀ x : ℤ, P.eval x ∈ ℤ

theorem polynomial_iteration_fixed_point
  (P : ℤ[X])
  (hP : polynomial_with_integer_coefficients P)
  (a : ℤ)
  (n : ℕ) (hn : 0 < n)
  (h : (P.eval_iter n a) = a) :
  P.eval (P.eval a) = a :=
sorry

end polynomial_iteration_fixed_point_l53_53980


namespace katie_earnings_l53_53521

def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

theorem katie_earnings : bead_necklaces + gemstone_necklaces * cost_per_necklace = 21 := 
by
  sorry

end katie_earnings_l53_53521


namespace length_of_AE_l53_53494

variable (A B C D E : Type) 
variable [ConvexQuadrilateral A B C D]
variable (AB CD AC AE EC : ℝ)

-- Given conditions
variable (hAB : AB = 10)
variable (hCD : CD = 15)
variable (hAC: AC = 18)
variable (hIntersect : intersections AC BD = E)
variable (hEqualAreas : area (triangle A E D) = area (triangle B E C))

theorem length_of_AE : AE = 7.2 :=
by
  have h1 := hAB -- AB = 10
  have h2 := hCD -- CD = 15
  have h3 := hAC -- AC = 18
  have h4 := hIntersect -- AC and BD intersect at E
  have h5 := hEqualAreas -- Areas of AED and BEC are equal
  -- Solving using necessary steps
  have AE : AE = 7.2 := sorry
  exact AE

end length_of_AE_l53_53494


namespace sum_of_solutions_l53_53025

theorem sum_of_solutions : 
  ∑ (x : ℝ) in {x | 0 ≤ x ∧ x ≤ 2 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4)}, x = π :=
by
  sorry

end sum_of_solutions_l53_53025


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53272

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53272


namespace coefficient_of_x_in_binomial_expansion_l53_53777

theorem coefficient_of_x_in_binomial_expansion 
  (a : ℝ) (h : a = -2 * ∫ x in 0..π, Real.sin x) :
  coefficient_of_x ((x^2 + (a / x))^5) = -640 := by
  -- Proof here.
  sorry

end coefficient_of_x_in_binomial_expansion_l53_53777


namespace complex_point_quadrant_l53_53869

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53869


namespace no_x4_term_implies_a_zero_l53_53099

theorem no_x4_term_implies_a_zero (a : ℝ) :
  ¬ (∃ (x : ℝ), -5 * x^3 * (x^2 + a * x + 5) = -5 * x^5 - 5 * a * x^4 - 25 * x^3 + 5 * a * x^4) →
  a = 0 :=
by
  -- Step through the proof process to derive this conclusion
  sorry

end no_x4_term_implies_a_zero_l53_53099


namespace range_of_cos_A_of_acute_triangle_l53_53843

theorem range_of_cos_A_of_acute_triangle
  (A B : ℝ) (BC : ℝ) (hBC : BC = 1) (hB : B = 2 * A) (h_acute : A > 0 ∧ A < π / 2) :
  (√2 / 2 < Real.cos A ∧ Real.cos A < √3 / 2) ∨ Real.cos A = 1 := sorry

end range_of_cos_A_of_acute_triangle_l53_53843


namespace find_x_perpendicular_vectors_l53_53801

theorem find_x_perpendicular_vectors :
  ∀ x : ℝ, let a := (x, x + 1), b := (1, 2) in
    (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 / 3 :=
by
  intros x a b h
  simp [a, b] at h
  -- further proof would be placed here, but we skip it with:
  sorry

end find_x_perpendicular_vectors_l53_53801


namespace perpendicular_lambda_l53_53443

-- Definitions
def vector_a : (ℝ × ℝ) := (2, 1)
def vector_b : (ℝ × ℝ) := (3, 4)

-- Dot product operation
def dot_product (u v : (ℝ × ℝ)) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Proof goal
theorem perpendicular_lambda : ∃ λ : ℝ, dot_product vector_a (2 * λ + 3, λ + 4) = 0 ∧ λ = -2 :=
begin
  -- The proof itself will be written here.
  sorry
end

end perpendicular_lambda_l53_53443


namespace sum_of_roots_of_equation_l53_53810

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l53_53810


namespace num_divisors_3960_l53_53739

theorem num_divisors_3960 : 
  ∃ d : ℕ, (3960 = 2^3 * 3^2 * 5^1 * 11^1) ∧
            (d = (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)) ∧
            (d = 48) :=
begin
  use 48,
  split,
  { -- 3960 factorization
    sorry
  },
  split,
  { -- divisor function calculation: (3 + 1)(2 + 1)(1 + 1)(1 + 1)
    sorry
  },
  { -- final result
    refl
  }
end

end num_divisors_3960_l53_53739


namespace hex_arrangements_correct_l53_53083

def hex_arrangements : Nat :=
  let vertices := fin 6
  let nums := {1, 2, 3, 4, 5, 6}
  let larger_than_neighbors (v : fin 6) (f : fin 6 → Nat) : Prop :=
    (f v > f (v + 1)) ∧ (f v > f (v - 1))
  let count_valid_arrangements : Nat :=
    -- Here we would count valid permutations satisfying the problem conditions
    8 -- The number of valid arrangements found in the problem
  count_valid_arrangements

theorem hex_arrangements_correct : hex_arrangements = 8 :=
by
  -- Proof omitted for now
  sorry

end hex_arrangements_correct_l53_53083


namespace B_is_empty_l53_53344

def A : Set ℤ := {0}
def B : Set ℤ := {x | x > 8 ∧ x < 5}
def C : Set ℕ := {x | x - 1 = 0}
def D : Set ℤ := {x | x > 4}

theorem B_is_empty : B = ∅ := by
  sorry

end B_is_empty_l53_53344


namespace complex_quadrant_l53_53929

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53929


namespace sin_double_angle_l53_53750

theorem sin_double_angle (α : ℝ) (h : sin α - cos α = 1 / 5) : sin (2 * α) = 24 / 25 :=
by
  sorry

end sin_double_angle_l53_53750


namespace complex_point_quadrant_l53_53876

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53876


namespace divide_set_into_groups_l53_53725

def sum_nat_up_to (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def set_sums (s : Finset ℕ) : ℕ :=
  s.sum id

def divide_into_two_groups (n : ℕ) (target_diff : ℕ) (setA setB : Finset ℕ) : Prop :=
  setA ∪ setB = Finset.range n ∧ setA ∩ setB = ∅ ∧ |(set_sums setA).natAbs - (set_sums setB).natAbs| = target_diff

theorem divide_set_into_groups :
  (Finset.card (Finset.filter (λ (s : Finset ℕ), divide_into_two_groups 9 16 s (Finset.range 9 \ s)) (Finset.powerset (Finset.range 9)))) = 8 := sorry

end divide_set_into_groups_l53_53725


namespace remainder_7547_div_11_l53_53296

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11_l53_53296


namespace triangle_area_union_eq_l53_53125

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_union_eq {PQ PR QR : ℝ}
  (hPQ : PQ = 15) (hPR : PR = 17) (hQR : QR = 16) :
  let PQR_area := heron_area PQ QR PR in
  PQR_area = 6 * sqrt 1008 :=
by
  sorry

end triangle_area_union_eq_l53_53125


namespace rectangular_coordinate_eqs_and_PA_PB_l53_53121

noncomputable def curve_C1 (ρ θ : ℝ) : Prop := ρ * sin^2 θ = 4 * cos θ

def parametric_C2 (t : ℝ) : ℝ × ℝ := (2 + 1/2 * t, sqrt 3 / 2 * t)

def point_P : ℝ × ℝ := (2, 0)

theorem rectangular_coordinate_eqs_and_PA_PB:
  (∀ (ρ θ : ℝ), curve_C1 ρ θ → (ρ^2 * sin^2 θ = 4 * ρ * cos θ) → ∃ x y : ℝ, (y^2 = 4 * x) ) ∧
  (∀ t : ℝ, parametric_C2 t = (2 + 1/2 * t, sqrt 3 / 2 * t) → ∃ x y : ℝ, (sqrt 3 * x - y - 2 * sqrt 3 = 0) ) ∧
  (∃ A B : ℝ × ℝ,
    ((curve_C1 1 (2 + 1/2 * t)).fst = C1_x ∧ (curve_C1 1 (2 + 1/2 * t)).snd = C1_y) ∧
    ((parametric_C2 t).fst = C2_x ∧ (parametric_C2 t).snd = C2_y) ) →
  ∃ PA PB : ℝ, |PA| * |PB| = 32 / 3 := sorry

end rectangular_coordinate_eqs_and_PA_PB_l53_53121


namespace domain_of_f_l53_53257

noncomputable def f (x : ℝ) : ℝ := (4 * x - 3) / (2 * x - 5)

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 / 2 → ∃ y : ℝ, f x = y :=
begin
  intros x h,
  use f x,
  exact ⟨⟩,
end

end domain_of_f_l53_53257


namespace jelly_bean_problem_l53_53328

variables {p_r p_o p_y p_g : ℝ}

theorem jelly_bean_problem :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 :=
by
  intros p_r_eq p_o_eq sum_eq
  -- The proof would proceed here, but we avoid proof details
  sorry

end jelly_bean_problem_l53_53328


namespace sum_of_roots_l53_53217

noncomputable theory
open Classical

theorem sum_of_roots (g : ℝ → ℝ)
  (h_symm : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ a b c d : ℝ, (∀ x, g x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ a b c d : ℝ, a + b + c + d = 12 :=
by
  sorry

end sum_of_roots_l53_53217


namespace problem1_proof_problem2_proof_l53_53706

-- Proof problem for Problem 1
theorem problem1_proof : 
    (1 * (-2)^3 + (1 / 9)^(-1) - (3.14 - Real.pi)^0) = 0 := 
by
  sorry

-- Proof problem for Problem 2
theorem problem2_proof (a b c : ℝ) :
    (3 * b^3 / (4 * a)) / (b * c / a^2) * (-2 * a / (3 * b))^2 = a^3 / (3 * b * c) := 
by
  sorry

end problem1_proof_problem2_proof_l53_53706


namespace Aaron_initial_cards_l53_53695

theorem Aaron_initial_cards (ending_cards found_cards initial_cards : ℕ) 
  (h_end : ending_cards = 67) (h_found : found_cards = 62) 
  (h_relation : ending_cards = initial_cards + found_cards) : 
  initial_cards = 5 := 
by {
  rw [h_end, h_found] at h_relation,
  linarith,
}

end Aaron_initial_cards_l53_53695


namespace triangle_perimeter_from_medians_l53_53786

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l53_53786


namespace sum_of_possible_values_of_x_l53_53819

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l53_53819


namespace no_perfect_square_l53_53565

-- Define the given polynomial
def poly (n : ℕ) : ℤ := n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3

-- The theorem to prove
theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, poly n = k^2 := by
  sorry

end no_perfect_square_l53_53565


namespace other_intercept_x_eq_11_l53_53029

-- Given conditions:
def vertex := (5 : ℝ, 10 : ℝ)
def intercept1 := (-1 : ℝ, 0 : ℝ)
def axis_of_symmetry := vertex.1 -- x-coordinate of the vertex

-- Statement to prove:
theorem other_intercept_x_eq_11 :
    ∃ y a b c : ℝ, (y = a * axis_of_symmetry^2 + b * axis_of_symmetry + c) ∧ 
    (intercept1.2 = a * intercept1.1^2 + b * intercept1.1 + c) ∧
    (2*axis_of_symmetry - intercept1.1 = 11) :=
sorry

end other_intercept_x_eq_11_l53_53029


namespace vasya_can_hit_ship_l53_53191

theorem vasya_can_hit_ship :
  ∀ (board : Finset (Fin 10 × Fin 10)) (ship : Finset (Fin 10 × Fin 10)),
    (∀ pos : Fin 10 × Fin 10, pos ∈ ship → pos ∈ board) →
    ∃ shots : Finset (Fin 10 × Fin 10), shots.card = 24 ∧
    ∀ ship_pos, ship_pos ⊆ ship → (∃ s ∈ shots, s ∈ ship_pos) :=
by
  sorry

end vasya_can_hit_ship_l53_53191


namespace mode_of_data_set_l53_53633

-- Define the data set
def data := [160, 163, 160, 157, 160]

-- Define what it means to be the mode of a list
def is_mode (data : List ℕ) (n : ℕ) : Prop :=
  ∀ m, List.count data n ≥ List.count data m

-- State the theorem
theorem mode_of_data_set : is_mode data 160 :=
sorry

end mode_of_data_set_l53_53633


namespace minimum_value_l53_53163

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃z, z = (x^2 + y^2) / (x + y)^2 ∧ z ≥ 1/2 := 
sorry

end minimum_value_l53_53163


namespace option_A_option_B_option_C_option_D_l53_53458

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l53_53458


namespace proof_problem_l53_53354

def problem : Prop :=
  |1-Real.sqrt 2| - 2 * Real.cos (Real.pi / 4) + (1/2)^(-1) = 1

theorem proof_problem : problem :=
by
  sorry

end proof_problem_l53_53354


namespace congruent_spheres_in_cone_l53_53126

noncomputable def coneRadius : ℝ := 6
noncomputable def coneHeight : ℝ := 15
noncomputable def sphereRadius : ℝ := 45 / (3 + 5*Real.sqrt 3 + Real.sqrt 261 / 2)

theorem congruent_spheres_in_cone :
  ∃ r : ℝ,
    r = sphereRadius ∧
    r * 2 ≤ coneRadius ∧
    r * 2 ≤ coneHeight :=
begin
  use sphereRadius,
  split,
  { refl },
  split;
  { sorry }
end

end congruent_spheres_in_cone_l53_53126


namespace log_base_sqrt_10_l53_53000

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l53_53000


namespace ferries_are_divisible_by_4_l53_53567

theorem ferries_are_divisible_by_4 (t T : ℕ) (H : ∃ n : ℕ, T = n * t) :
  ∃ N : ℕ, N = 4 * (T / t) ∧ N % 4 = 0 :=
by
  sorry

end ferries_are_divisible_by_4_l53_53567


namespace units_digit_sum_of_factorials_l53_53383

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  units_digit (∑ i in Finset.range 31, factorial i) = 3 :=
by
  sorry

end units_digit_sum_of_factorials_l53_53383


namespace smallest_x_l53_53129

theorem smallest_x (a b x : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (pos_x : x > 0) : x = 200000 := sorry

end smallest_x_l53_53129


namespace average_marks_all_students_l53_53729

def firstClassAvg := 60
def firstClassStudents := 55
def secondClassAvg := 58
def secondClassStudents := 48

def totalMarksFirstClass := firstClassAvg * firstClassStudents
def totalMarksSecondClass := secondClassAvg * secondClassStudents
def combinedTotalMarks := totalMarksFirstClass + totalMarksSecondClass
def combinedNumberOfStudents := firstClassStudents + secondClassStudents
def overallAvgMarks := combinedTotalMarks / combinedNumberOfStudents

theorem average_marks_all_students : overallAvgMarks = 59.07 := 
by 
  -- Proof steps can be filled here
  sorry

end average_marks_all_students_l53_53729


namespace conjugate_of_expression_l53_53419

noncomputable def z : ℂ := 1 - complex.I

theorem conjugate_of_expression : (z = 1 - complex.I) →
  complex.conj (2 / z - z ^ 2) = 2 - 4 * complex.I :=
by
  intro h
  rw h
  sorry  -- proof goes here

end conjugate_of_expression_l53_53419


namespace product_expansion_l53_53703

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := 
by
  sorry

end product_expansion_l53_53703


namespace smallest_n_condition_l53_53829

theorem smallest_n_condition :
  ∃ n ≥ 2, ∃ (a : Fin n → ℤ), (Finset.sum Finset.univ a = 1990) ∧ (Finset.univ.prod a = 1990) ∧ (n = 5) :=
by
  sorry

end smallest_n_condition_l53_53829


namespace large_rectangle_perimeter_correct_l53_53329

def perimeter_of_square (p : ℕ) : ℕ :=
  p / 4

def perimeter_of_rectangle (p : ℕ) (l : ℕ) : ℕ :=
  (p - 2 * l) / 2

def perimeter_of_large_rectangle (side_length_of_square side_length_of_rectangle : ℕ) : ℕ :=
  let height := side_length_of_square + 2 * side_length_of_rectangle
  let width := 3 * side_length_of_square
  2 * (height + width)

theorem large_rectangle_perimeter_correct :
  let side_length_of_square := perimeter_of_square 24
  let side_length_of_rectangle := perimeter_of_rectangle 16 side_length_of_square
  perimeter_of_large_rectangle side_length_of_square side_length_of_rectangle = 52 :=
by
  sorry

end large_rectangle_perimeter_correct_l53_53329


namespace B_alone_finishes_in_21_days_l53_53307

theorem B_alone_finishes_in_21_days (W_A W_B : ℝ) (h1 : W_A = 0.5 * W_B) (h2 : W_A + W_B = 1 / 14) : W_B = 1 / 21 :=
by sorry

end B_alone_finishes_in_21_days_l53_53307


namespace sin_alpha_plus_half_pi_l53_53039

theorem sin_alpha_plus_half_pi (α : ℝ) 
  (h1 : Real.tan (α - Real.pi) = 3 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.sin (α + Real.pi / 2) = -4 / 5 :=
by
  -- Placeholder for the proof
  sorry

end sin_alpha_plus_half_pi_l53_53039


namespace irene_weekly_income_l53_53964

noncomputable def base_salary : ℝ := 500
noncomputable def federal_tax_rate : ℝ := 0.15
noncomputable def health_insurance : ℝ := 50

def overtime_pay (hours : ℝ) : ℝ :=
  if hours ≤ 5 then hours * 20
  else if hours ≤ 8 then 5 * 20 + (hours - 5) * 30
  else 5 * 20 + 3 * 30 + (hours - 8) * 40

def total_income (base_salary: ℝ) (overtime_hours : ℝ) : ℝ :=
  base_salary + (overtime_pay overtime_hours)

def deductions (base_salary : ℝ) (federal_tax_rate : ℝ) (health_insurance : ℝ) : ℝ :=
  (base_salary * federal_tax_rate) + health_insurance

def net_income (base_salary : ℝ) (overtime_hours : ℝ) (federal_tax_rate : ℝ) (health_insurance : ℝ) : ℝ :=
  total_income base_salary overtime_hours - deductions base_salary federal_tax_rate health_insurance

theorem irene_weekly_income :
  net_income base_salary 10 federal_tax_rate health_insurance = 645 := by
  sorry

end irene_weekly_income_l53_53964


namespace time_for_P_to_finish_job_l53_53629

variable (P_rate Q_rate t : ℝ)

-- Definitions based on conditions
def P_rate := 1 / 3    -- P's rate: 1/3 of the job per hour
def Q_rate := 1 / 9    -- Q's rate: 1/9 of the job per hour
def t := 2             -- Time P and Q work together: 2 hours

-- Lean 4 statement for the proof
theorem time_for_P_to_finish_job : 
  let combined_work := (P_rate + Q_rate) * t in
  let remaining_work := 1 - combined_work in
  let time_needed_by_P := remaining_work / P_rate in
  let time_needed_in_minutes := time_needed_by_P * 60 in
  time_needed_in_minutes = 20 :=
by
  -- Using the provided rates and time, we prove the statement
  sorry

end time_for_P_to_finish_job_l53_53629


namespace chord_length_l53_53736

theorem chord_length (A B C : ℝ) (r : ℝ) (d : ℝ) 
  (hA : A = sqrt 3) (hB : B = 1) (hC : C = -2 * sqrt 3) 
  (hLine : ∀ x y : ℝ, A * x + B * y + C = 0) 
  (hCirc : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (hDist : d = abs (A * 0 + B * 0 + C) / sqrt (A^2 + B^2)) 
  (hr : r = 2) 
  (hd : d = sqrt 3) : 
  2 * sqrt (r^2 - d^2) = 2 := 
by 
  sorry

end chord_length_l53_53736


namespace part1_part2_l53_53065

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + a * x + 1
noncomputable def g (x : ℝ) := x * Real.exp x

-- Problem Part 1: Prove the range of a for which f(x) has two extreme points
theorem part1 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = f x₂ a) ↔ (0 < a ∧ a < (1 / Real.exp 1)) :=
sorry

-- Problem Part 2: Prove the range of a for which f(x) ≥ 2sin(x) for x ≥ 0
theorem part2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ 2 * Real.sin x) ↔ (2 ≤ a) :=
sorry

end part1_part2_l53_53065


namespace complex_point_in_first_quadrant_l53_53863

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53863


namespace complex_number_quadrant_l53_53896

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53896


namespace binomial_expansion_max_coefficient_l53_53115

theorem binomial_expansion_max_coefficient (n : ℕ) (h : n > 0) 
  (h_max_coefficient: ∀ m : ℕ, m ≠ 5 → (Nat.choose n m ≤ Nat.choose n 5)) : 
  n = 10 :=
sorry

end binomial_expansion_max_coefficient_l53_53115


namespace min_spend_before_tax_l53_53503

theorem min_spend_before_tax :
  ∃ M : ℕ, 
  (Penny_paid : ℕ → ℕ) (Penny_spend : ℕ → ℕ) (tax : ℕ → ℕ) (cost_before_tax : ℕ → ℕ), 
  (Penny_paid M = 240) ∧
  (Penny_spend M = M + 32) ∧
  (tax M = 1 * Penny_spend M) ∧
  (cost_before_tax M = 5 * Penny_spend M) ∧
  (Penny_paid M = cost_before_tax M + tax M) ∧
  (M = 8) := 
begin
  let Penny_spend := λ M, M + 32,
  let tax := λ M, 1 * Penny_spend M,
  let cost_before_tax := λ M, 5 * Penny_spend M,
  let Penny_paid := λ M, cost_before_tax M + tax M,
  use 8,
  simp [Penny_paid, Penny_spend, tax, cost_before_tax],
  split;
  [ rfl, split; [ rfl, split; [ rfl, split; [ rfl, rfl ]]]],
end

end min_spend_before_tax_l53_53503


namespace derivative_y_l53_53730

noncomputable def y (x : ℝ) : ℝ := sqrt (1 - 3 * x - 2 * x ^ 2) + (3 / (2 * sqrt 2)) * arcsin ((4 * x + 3) / sqrt 17)

theorem derivative_y (x : ℝ) : 
  has_deriv_at y (-2 * x / sqrt (1 - 3 * x - 2 * x ^ 2)) x :=
sorry

end derivative_y_l53_53730


namespace sin_cos_sum_zero_of_intersection_l53_53776

theorem sin_cos_sum_zero_of_intersection {α β : ℝ}
  (h1 : ∃ x₀, 1 = (x₀ / (sin α + sin β) - x₀ / (sin α + cos β)) ∧ 1 = (x₀ / (cos α + sin β) - x₀ / (cos α + cos β))) :
  sin α + cos α + sin β + cos β = 0 :=
sorry

end sin_cos_sum_zero_of_intersection_l53_53776


namespace no_even_is_prime_equiv_l53_53220

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l53_53220


namespace complex_point_in_first_quadrant_l53_53862

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53862


namespace harkamal_total_amount_l53_53080

-- Conditions
def cost_grapes : ℝ := 8 * 80
def cost_mangoes : ℝ := 9 * 55
def cost_apples_before_discount : ℝ := 6 * 120
def cost_oranges : ℝ := 4 * 75
def discount_apples : ℝ := 0.10 * cost_apples_before_discount
def cost_apples_after_discount : ℝ := cost_apples_before_discount - discount_apples

def total_cost_before_tax : ℝ :=
  cost_grapes + cost_mangoes + cost_apples_after_discount + cost_oranges

def sales_tax : ℝ := 0.05 * total_cost_before_tax

def total_amount_paid : ℝ := total_cost_before_tax + sales_tax

-- Question translated into a Lean statement
theorem harkamal_total_amount:
  total_amount_paid = 2187.15 := 
sorry

end harkamal_total_amount_l53_53080


namespace solve_question_1_solve_question_2_solve_question_3_l53_53656

namespace NewEnergyVehicleFactory

-- Problem conditions
def planned_production : Int := 60
def daily_changes : List Int := [-5, 7, -3, 4, 10, -9, -25]

-- Prove the actual production on Wednesday
def actual_production_wednesday : Int :=
  planned_production + daily_changes.nthLe 2 (by decide)

theorem solve_question_1 : actual_production_wednesday = 57 := by
  sorry

-- Prove the total production increase/decrease for the week
def total_weekly_change : Int :=
  daily_changes.foldl (+) 0

theorem solve_question_2 : total_weekly_change = -21 := by
  sorry

-- Prove the difference in production between the highest and lowest production days
def max_daily_change : Int := daily_changes.foldl max (-60)
def min_daily_change : Int := daily_changes.foldl min 60

theorem solve_question_3 : (max_daily_change - min_daily_change) = 35 := by
  sorry

end NewEnergyVehicleFactory

end solve_question_1_solve_question_2_solve_question_3_l53_53656


namespace find_ordered_pair_l53_53378

theorem find_ordered_pair : ∃ x y : ℝ, 3 * x - 7 * y = 2 ∧ 4 * y - x = 6 ∧ x = 10 ∧ y = 4 :=
by
  -- existence part
  use 10, 4
  -- conditions
  split
  -- check first equation
  { calc 3 * 10 - 7 * 4 = 30 - 28 : by norm_num
                   ... = 2 : by norm_num }, 
  split
  -- check second equation
  { calc 4 * 4 - 10 = 16 - 10 : by norm_num
                   ... = 6 : by norm_num },
  -- solutions
  split; norm_num; sorry

end find_ordered_pair_l53_53378


namespace sum_of_solutions_comparison_l53_53151

variable (a a' b b' c c' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0)

theorem sum_of_solutions_comparison :
  ( (c - b) / a > (c' - b') / a' ) ↔ ( (c'-b') / a' < (c-b) / a ) :=
by sorry

end sum_of_solutions_comparison_l53_53151


namespace no_square_cube_l53_53536

theorem no_square_cube (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, k^2 = n * (n + 1) * (n + 2) * (n + 3)) ∧ ¬ (∃ l : ℕ, l^3 = n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end no_square_cube_l53_53536


namespace centroid_of_triangle_l53_53939

open Complex

theorem centroid_of_triangle :
  let z1 := (-11) + 3 * Complex.I
  let z2 := 3 - 7 * Complex.I
  let z3 := 5 + 9 * Complex.I
  (z1 + z2 + z3) / 3 = -1 + (5 / 3) * Complex.I := by
sory

end centroid_of_triangle_l53_53939


namespace abs_iff_neg_one_lt_x_lt_one_l53_53314

theorem abs_iff_neg_one_lt_x_lt_one (x : ℝ) : |x| < 1 ↔ -1 < x ∧ x < 1 :=
by
  sorry

end abs_iff_neg_one_lt_x_lt_one_l53_53314


namespace correct_result_l53_53146

theorem correct_result :
  ∃ (x y : ℕ) (a b : ℕ), 
    (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧
    (x = 10 * a + b) ∧ (y ≠ 10 * b + a) ∧ 
    ((10 * a + b) * y - (10 * b + a) * y = 4248) ∧
    ((10 * a + b) * y = 4720 ∨ (10 * a + b) * y = 5369) :=
begin
  sorry
end

end correct_result_l53_53146


namespace geometric_sequence_common_ratio_l53_53119

-- Define the conditions as hypotheses
variables {a₁ q : ℝ}
def a_3 : ℝ := a₁ * q^2
def a_7 : ℝ := a₁ * q^6

-- State the theorem
theorem geometric_sequence_common_ratio (h1 : a_3 = 2) (h2 : a_7 = 32) : q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l53_53119


namespace true_1_false_2_false_3_true_4_l53_53756

variables {m n : Type} [HasSubset m α] [HasSubset m β]
variables {α β : Type} [HasPerp m flat_plane] [HasIntersection α β] [HasSubset n α] [HasSubset n β]
variables {α_parallel β_parallel : Type} [HasParallel α β] [HasSubset m α_parallel]

-- Definitions for each proposition condition
def prop_1 (m : Type) [HasSubset m α] [HasPerp m β] : Prop :=
  α ⊆ β

def prop_2 (m : Type) [HasSubset m α] [HasIntersection α β] [HasPerp α β] : Prop :=
  m ⊥ n

def prop_3 (m : Type) [HasSubset m α_parallel] [HasSubset n β_parallel] [HasParallel α_parallel β_parallel] : Prop :=
  m ∥ n

def prop_4 (m : Type) [HasParallel m α_parallel] [HasSubset m β_parallel] [HasIntersection α_parallel β_parallel] : Prop :=
  m ∥ n

-- Lean statements for the propositions
theorem true_1 : prop_1 m := sorry

theorem false_2 : ¬ prop_2 m := sorry

theorem false_3 : ¬ prop_3 m := sorry

theorem true_4 : prop_4 m := sorry

end true_1_false_2_false_3_true_4_l53_53756


namespace matrix_expression_solution_l53_53096

theorem matrix_expression_solution (x : ℝ) :
  let a := 3 * x + 1
  let b := x + 1
  let c := 2
  let d := 2 * x
  ab - cd = 5 :=
by
  sorry

end matrix_expression_solution_l53_53096


namespace arithmetic_geometric_sequence_problem_l53_53051

theorem arithmetic_geometric_sequence_problem :
  (∃ a1 d, 
    (∀ n, S_n = (n:ℚ)/2 * (2*a1 + (n-1)*d)) ∧ 
    (a1 + 2*d = -6) ∧
    (5*a1 + 10*d = 6*a1 + 15*d) ∧
    (a1 = -10 ∧ d = 2)) ∧
  (let a2 := -8
  let S3 := -24
  let b1 := a2
  let b2 := S3 
  let q := 3 
  in
  (∀ n, b_n = b1 * q^(n-1) ∧ (q = 3)) ∧
  ∀ n : ℕ, T_n = b1 * (1 - q^n) / (1 - q)) := sorry

end arithmetic_geometric_sequence_problem_l53_53051


namespace solve_f_ex_gt_zero_l53_53059

/-
Given that the solution set of the quadratic inequality f(x) < 0 is 
{x | x < -1 ∨ x > 1/3}, prove that the solution set for f(e^x) > 0 is 
{x | x < -log 3}.
-/

theorem solve_f_ex_gt_zero (f : ℝ → ℝ) 
  (h_solution : ∀ x : ℝ, f x < 0 ↔ x < -1 ∨ x > 1 / 3) :
  (∀ x : ℝ, f (exp x) > 0 ↔ x < -real.log 3) :=
by
  sorry

end solve_f_ex_gt_zero_l53_53059


namespace complex_point_in_first_quadrant_l53_53867

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l53_53867


namespace no_even_is_prime_equiv_l53_53221

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l53_53221


namespace log_sqrt10_1000sqrt10_l53_53007

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53007


namespace mark_reading_time_l53_53548

-- Definitions based on conditions
def daily_reading_hours : ℕ := 3
def days_in_week : ℕ := 7
def weekly_increase : ℕ := 6

-- Proof statement
theorem mark_reading_time : daily_reading_hours * days_in_week + weekly_increase = 27 := by
  -- placeholder for the proof
  sorry

end mark_reading_time_l53_53548


namespace fourth_power_of_cube_third_smallest_prime_l53_53293

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53293


namespace original_phone_number_eq_l53_53305

theorem original_phone_number_eq :
  ∃ (a b c d e f : ℕ), 
    (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 282500) ∧
    (1000000 * 2 + 100000 * a + 10000 * 8 + 1000 * b + 100 * c + 10 * d + e = 81 * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) ∧
    (0 ≤ a ∧ a ≤ 9) ∧
    (0 ≤ b ∧ b ≤ 9) ∧
    (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧
    (0 ≤ e ∧ e ≤ 9) ∧
    (0 ≤ f ∧ f ≤ 9) :=
sorry

end original_phone_number_eq_l53_53305


namespace simplification_correct_l53_53352

variable (p : ℝ)
variable (hp : 0 ≤ p)

theorem simplification_correct : (sqrt (30 * p) * sqrt (5 * p) * sqrt (6 * p) = 30 * p * sqrt p) := by
  sorry

end simplification_correct_l53_53352


namespace correct_propositions_l53_53624

-- Define the events and their probabilities
variables (Ω : Type) [fintype Ω]
variables (P : set Ω → ℝ) [probability_measure P]

-- Define propositions
def proposition_B : Prop :=
∀ (A : set Ω), (∀ ω, ω ∈ A) → P A = 1

def proposition_D : Prop :=
∀ (A B : set Ω), P (A ∪ B) = P A + P B - P (A ∩ B)

-- Define theorem to prove
theorem correct_propositions (hB : proposition_B Ω P) (hD : proposition_D Ω P) : true :=
begin
  -- Proof is not included, so we use sorry to indicate it.
  sorry
end

end correct_propositions_l53_53624


namespace pythagoras_school_student_count_l53_53451

theorem pythagoras_school_student_count
  (x : ℕ)
  (h_math : (x / 2) = (x : ℝ) / 2)
  (h_music : (x / 4) = (x : ℝ) / 4)
  (h_silent : (x / 7) = (x : ℝ) / 7)
  (h_women : 3) :
  x = 28 := 
sorry

end pythagoras_school_student_count_l53_53451


namespace Brianchons_theorem_l53_53195

variable (A B C D E F R Q T S P U : Point)

axiom hexagon_circumscribed : TangentCircleHexagon A B C D E F R Q T S P U

theorem Brianchons_theorem (hexagon_circumscribed : TangentCircleHexagon A B C D E F R Q T S P U) :
  intersects_at_one_point (diagonal A D) (diagonal B E) (diagonal C F) :=
sorry

end Brianchons_theorem_l53_53195


namespace factory_hours_per_day_l53_53482

def hour_worked_forth_machine := 12
def production_rate_per_hour := 2
def selling_price_per_kg := 50
def total_earnings := 8100

def h := 23

theorem factory_hours_per_day
  (num_machines : ℕ)
  (num_machines := 3)
  (production_first_three : ℕ := num_machines * production_rate_per_hour * h)
  (production_fourth : ℕ := hour_worked_forth_machine * production_rate_per_hour)
  (total_production : ℕ := production_first_three + production_fourth)
  (total_earnings_eq : total_production * selling_price_per_kg = total_earnings) :
  h = 23 := by
  sorry

end factory_hours_per_day_l53_53482


namespace exists_fibonacci_with_three_zeros_l53_53436

def Fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| n + 2 := Fibonacci n + Fibonacci (n + 1)

theorem exists_fibonacci_with_three_zeros :
  ∃ n, Fibonacci n % 1000 = 0 :=
sorry

end exists_fibonacci_with_three_zeros_l53_53436


namespace ratio_is_two_l53_53526

-- Define the geometric setup
variables (A B C D E : Type) [EuclideanGeometry A B C D E]

-- Define the properties of the trapezoid
variable (H1 : parallel AB CD)  -- AB is parallel to CD

-- Define the extensions meeting at E
variable (H2 : meet_at_ext A D E)
variable (H3 : meet_at_ext B C E)

-- Define the degree sums
def S := degree_sum (∠ CDE) (∠ DCE)
def S' := degree_sum (∠ BAD) (∠ ABC)

-- Define the ratio r
def r := S / S'

-- The theorem we want to prove
theorem ratio_is_two (H1 : parallel AB CD) (H2 : meet_at_ext A D E) (H3 : meet_at_ext B C E)
  (S_eq : S = 360) (S'_eq : S' = 180) : r = 2 := 
by
  sorry

end ratio_is_two_l53_53526


namespace area_of_ABC_l53_53831

variables {K : ℝ} (A B C M : ℝ × ℝ)
-- Conditions
def right_angle_at_C (C : ℝ × ℝ) : Prop := C = (0, 0)
def median_CM (A B C M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def CM_bisects_right_angle (A B C M : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, θ = π / 4 ∧ 
            angle A C M = θ ∧ 
            angle B C M = θ

def area_ACM_eq_K (A C M : ℝ × ℝ) (K : ℝ) : Prop := 
  1 / 2 * (dist A M) * (dist C M) = K

-- Goal
theorem area_of_ABC (A B C M : ℝ × ℝ) (K : ℝ) :
  right_angle_at_C C ∧
  median_CM A B C M ∧
  CM_bisects_right_angle A B C M ∧
  area_ACM_eq_K A C M K →
  1 / 2 * (dist A B) * (dist A C) = 2 * K := sorry

end area_of_ABC_l53_53831


namespace simplify_and_evaluate_l53_53569

theorem simplify_and_evaluate (a : ℚ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5 / 3 :=
by
  sorry

end simplify_and_evaluate_l53_53569


namespace sum_of_possible_values_of_x_l53_53818

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l53_53818


namespace tetrahedron_volume_l53_53545

open EuclideanGeometry

variables (P Q R S : Point)
variables (hPQ : dist P Q = 6)
variables (hPR : dist P R = 3)
variables (hPS : dist P S = 5)
variables (hQR : dist Q R = 5)
variables (hQS : dist Q S = 4)
variables (hRS : dist R S = (15 / 4) * Real.sqrt 2)

-- The volume of the tetrahedron PQRS
def volume_tetrahedron : ℝ :=
  (1715 / 144) * Real.sqrt 2

theorem tetrahedron_volume :
  volume P Q R S = volume_tetrahedron :=
sorry

end tetrahedron_volume_l53_53545


namespace complex_point_in_first_quadrant_l53_53855

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53855


namespace least_possible_area_l53_53641

-- Definitions of conditions
def side_length_measured : ℝ := 7
def side_length_min : ℝ := 6.50
def side_length_max : ℝ := 7.49

-- Definition of area function for a square
def square_area (side : ℝ) : ℝ := side * side

-- Theorem statement
theorem least_possible_area :
  side_length_measured = 7 →
  (∀ side : ℝ, side_length_min ≤ side ∧ side < 7 → (Real.round side = 7)) →
  ∃ l : ℝ, l = side_length_min ∧ square_area l = 42.25 :=
begin
  sorry
end

end least_possible_area_l53_53641


namespace volume_intersection_of_two_cubes_l53_53249

theorem volume_intersection_of_two_cubes 
(edge_length : ℝ) 
(h1 : ∀ (T₁ T₂ : set (ℝ × ℝ × ℝ)), (T₁ ∩ T₂).volume = 2 * edge_length^3 * (real.sqrt 2 - 1)) :
  (∀ (a : ℝ), a = edge_length → ∃ V, V = 2 * a^3 * (real.sqrt 2 - 1)) :=
by
  intros
  sorry

end volume_intersection_of_two_cubes_l53_53249


namespace part_I_part_II_l53_53778

variable (a b c : ℝ)

theorem part_I (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : a + b + c = 4 :=
sorry

theorem part_II (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8/7 :=
sorry

end part_I_part_II_l53_53778


namespace break_even_items_l53_53215

theorem break_even_items (C N : ℝ) (h_cost_inversely_proportional : ∃ k, C * real.sqrt N = k) 
  (h_cost_10_items : C * real.sqrt 10 = 2100 * real.sqrt 10) 
  (h_revenue : 30 * N = C) :
  N = 10 * real.cbrt 49 :=
by
  sorry

end break_even_items_l53_53215


namespace tan_double_angle_l53_53089

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4 / 3 :=
by
  sorry

end tan_double_angle_l53_53089


namespace characterize_f_l53_53369

noncomputable def f : ℚ → ℚ := sorry

def condition1 (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f(x) * f(y) = f(x) + f(y) - f(x * y)

def condition2 (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, 1 + f(x + y) = f(x * y) + f(x) * f(y)

theorem characterize_f (f : ℚ → ℚ) :
  (condition1 f ∧ condition2 f) ↔ (f = (λ _ => 1) ∨ f = (λ x => 1 - x)) :=
by
  sorry

end characterize_f_l53_53369


namespace log_base_8_eq_3_implies_y_eq_512_l53_53821

theorem log_base_8_eq_3_implies_y_eq_512 (y : ℝ) (h : log 8 y = 3) : y = 512 := 
by 
  sorry

end log_base_8_eq_3_implies_y_eq_512_l53_53821


namespace complex_point_in_first_quadrant_l53_53857

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53857


namespace min_value_problem_l53_53153

-- Definition of the set
def elements : set ℤ := {-6, -4, -3, -1, 1, 3, 5, 7}

-- Distinctness condition
def distinct (list : list ℤ) : Prop := 
  list.nodup

-- Definition of the problem
theorem min_value_problem
  (p q r s t u v w : ℤ) 
  (hp : p ∈ elements)
  (hq : q ∈ elements)
  (hr : r ∈ elements)
  (hs : s ∈ elements)
  (ht : t ∈ elements)
  (hu : u ∈ elements)
  (hv : v ∈ elements)
  (hw : w ∈ elements)
  (distinct_vals : distinct [p, q, r, s, t, u, v, w]) :
  (p+q+r+s)^2 + (t+u+v+w)^2 = 2 := 
sorry

end min_value_problem_l53_53153


namespace probability_between_u_v_l53_53102

def line_u (x : ℝ) : ℝ := -2 * x + 8
def line_v (x : ℝ) : ℝ := -3 * x + 8

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

def area_u : ℝ := area_triangle 4 8
def area_v : ℝ := area_triangle (8 / 3) 8

def area_between : ℝ := area_u - area_v

def probability : ℝ := area_between / area_u

theorem probability_between_u_v :
  probability ≈ 0.33 := by
  sorry

end probability_between_u_v_l53_53102


namespace area_of_curvilinear_shape_l53_53208

def parabola (x : ℝ) : ℝ := x^2

def tangent_line (x : ℝ) : ℝ := 4 * x - 4

theorem area_of_curvilinear_shape :
  ∫ x in 0..2, (parabola x) - ∫ x in 1..2, (tangent_line x) = 2 / 3 :=
by
  sorry

end area_of_curvilinear_shape_l53_53208


namespace tangent_circles_existence_l53_53075

noncomputable def construct_tangent_circles
  (A B C : Point)
  (O1 O2 O3 : Point)
  (S1 : circle O1 radius1)
  (S2 : circle O2 radius2)
  (S3 : circle O3 radius3) :
  Prop :=
  tangent S1 S2 C ∧ tangent S1 S3 B ∧ tangent S2 S3 A ∧
  lies_on A (segment O2 O3) ∧
  lies_on B (segment O1 O3) ∧
  lies_on C (segment O1 O2)

theorem tangent_circles_existence :
  ∀ (A B C : Point),
  ∃ (O1 O2 O3 : Point)
    (S1 : circle O1 radius1)
    (S2 : circle O2 radius2)
    (S3 : circle O3 radius3),
  construct_tangent_circles A B C O1 O2 O3 S1 S2 S3 :=
sorry

end tangent_circles_existence_l53_53075


namespace complex_number_quadrant_l53_53889

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l53_53889


namespace problem_I_problem_II_l53_53774

noncomputable def point_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem_I (p := 2) : (|point_distance (0, -4) (sqrt(2)/2 * t_1, -4 + sqrt(2)/2 * t_1)| + 
                            |point_distance (0, -4) (sqrt(2)/2 * t_2, -4 + sqrt(2)/2 * t_2)|) = 12 * sqrt(2) 
                            := by sorry

theorem problem_II (h : |point_distance (0, -4) (sqrt(2)/2 * t_1, -4 + sqrt(2)/2 * t_1)|, 
    |point_distance (sqrt(2)/2 * t_1, -4 + sqrt(2)/2 * t_1) (sqrt(2)/2 * t_2, -4 + sqrt(2)/2 * t_2)|, 
    |point_distance (0, -4) (sqrt(2)/2 * t_2, -4 + sqrt(2)/2 * t_2)| form_geometric_progression) 
    : p = -4 + 2 * sqrt(5) := by sorry

end problem_I_problem_II_l53_53774


namespace complex_quadrant_is_first_l53_53926

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53926


namespace find_perpendicular_line_through_point_l53_53015

-- Given conditions
def point := (0 : ℝ, 1 : ℝ)
def given_line := ∀ x y : ℝ, 2 * x - y = 0

-- Definition of perpendicularity in terms of slopes
def perpendicular_slope (m : ℝ) : ℝ := -1 / m

-- Define the target line using point-slope form and simplified to general form
theorem find_perpendicular_line_through_point :
  ∃ a b c : ℝ, a * 0 + b * 1 + c = 0 ∧ 2 * a + b = 0 ∧ -1 / 2 ≠ 0 ∧ a = 1 ∧ b = 2 ∧ c = -2 :=
by 
  sorry

end find_perpendicular_line_through_point_l53_53015


namespace james_total_points_l53_53204

theorem james_total_points :
  let correct_answer_points := 2
  let incorrect_answer_points := -1
  let bonus_points := 4
  let total_rounds := 5
  let questions_per_round := 5
  let total_questions := total_rounds * questions_per_round
  let missed_questions := 1
  let correct_questions := total_questions - missed_questions
  let correct_points := correct_questions * correct_answer_points
  let bonus_rounds := total_rounds - (missed_questions / questions_per_round)
  let total_bonus := bonus_rounds * bonus_points
 in correct_points + total_bonus = 64 := 
by
  let correct_answer_points := 2
  let incorrect_answer_points := -1
  let bonus_points := 4
  let total_rounds := 5
  let questions_per_round := 5
  let total_questions := total_rounds * questions_per_round
  let missed_questions := 1
  let correct_questions := total_questions - missed_questions
  let correct_points := correct_questions * correct_answer_points
  let bonus_rounds := total_rounds - (missed_questions / questions_per_round)
  let total_bonus := bonus_rounds * bonus_points
  have correct_points_value : correct_points = 48 := sorry
  have total_bonus_value : total_bonus = 16 := sorry
  show correct_points + total_bonus = 64 from sorry

end james_total_points_l53_53204


namespace complex_point_in_first_quadrant_l53_53858

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l53_53858


namespace complex_quadrant_check_l53_53909

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l53_53909


namespace digit_6_occurrences_100_to_999_l53_53956

theorem digit_6_occurrences_100_to_999 : 
  let count_digit_6 (n : ℕ) : ℕ := (n.digits 10).count (λ d, d = 6) in
  (List.range' 100 900).sum count_digit_6 = 280 := 
by
  sorry

end digit_6_occurrences_100_to_999_l53_53956


namespace quadrilateral_perimeter_l53_53497

open Real

/-- Given a quadrilateral PQRS with specific conditions,
prove that its perimeter is 44 + √706. -/
theorem quadrilateral_perimeter
  (P Q R S : ℝ × ℝ)
  (PQ QR RS PR : ℝ)
  (hQ : angle (P-Q) (R-Q) = π / 2)
  (hPR_RS : angle (P-R) (S-R) = π / 2)
  (hPQ : dist P Q = 15)
  (hQR : dist Q R = 20)
  (hRS : dist R S = 9) :
  dist P Q + dist Q R + dist R S + dist P S = 44 + Real.sqrt 706 :=
by
  sorry

end quadrilateral_perimeter_l53_53497


namespace log_sqrt10_1000sqrt10_l53_53006

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53006


namespace julia_played_tag_l53_53974

theorem julia_played_tag : 
    ∀ (kidsMonday kidsTuesday kidsAltogether : ℕ), 
    kidsMonday = 7 → 
    kidsAltogether = 20 → 
    kidsAltogether = kidsMonday + kidsTuesday → 
    kidsTuesday = 13 :=
by
  intros kidsMonday kidsTuesday kidsAltogether h1 h2 h3
  rw [h3, h1] at h2
  have h4 : kidsTuesday = kidsAltogether - kidsMonday := by 
    exact Nat.sub_eq_of_eq_add h2.symm
  rw [h1, h2] at h4
  exact h4.symm

end julia_played_tag_l53_53974


namespace circle_intersection_line_l53_53800

theorem circle_intersection_line (d : ℝ) :
  (∃ (x y : ℝ), (x - 5)^2 + (y + 2)^2 = 49 ∧ (x + 1)^2 + (y - 5)^2 = 25 ∧ x + y = d) ↔ d = 6.5 :=
by
  sorry

end circle_intersection_line_l53_53800


namespace second_derivative_l53_53312

-- Definitions based on given conditions
def x (t : ℝ) : ℝ := 2 * (t - Real.sin t)
def y (t : ℝ) : ℝ := 8 + 4 * Real.cos t

-- The theorem to be proved
theorem second_derivative (t : ℝ) : 
  (d^2y/dx^2) 
  = 1 / (1 - Real.cos t)^3 :=
  sorry

end second_derivative_l53_53312


namespace trapezoid_midpoints_bd_eq_2pq_l53_53561

open EuclideanGeometry

variables {A B C D P Q : Point}
variables (hTrapezoid : IsTrapezoid A B C D) 
          (hMidP : Midpoint P A D) 
          (hMidQ : Midpoint Q B C) 
          (hEquality : Distance A B = Distance B C) 
          (hAngleBisector : IsAngleBisector P B (LineThrough P A) (LineThrough P D))

theorem trapezoid_midpoints_bd_eq_2pq
  (hTrapezoid : IsTrapezoid A B C D)
  (hMidP : Midpoint P A D)
  (hMidQ : Midpoint Q B C)
  (hEquality : Distance A B = Distance B C)
  (hAngleBisector : IsAngleBisector P B (LineThrough P A) (LineThrough P D)) :
  Distance B D = 2 * Distance P Q := 
  sorry

end trapezoid_midpoints_bd_eq_2pq_l53_53561


namespace log_sqrt10_1000sqrt10_l53_53010

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53010


namespace harkamal_grapes_purchase_l53_53447

/-- Harkamal purchased some kg of grapes at the rate of 70 per kg and 9 kg of mangoes at the rate 
    of 55 per kg. He paid 705 to the shopkeeper. Prove the number of kg of grapes he purchased is 3. --/

variables (G : ℕ)

axioms 
  (rate_grapes : 70)
  (amount_mangoes : 9)
  (rate_mangoes : 55)
  (total_paid : 705)

theorem harkamal_grapes_purchase :
  (70 * G + 9 * 55 = 705) → G = 3 :=
by sorry

end harkamal_grapes_purchase_l53_53447


namespace john_builds_computers_l53_53971

theorem john_builds_computers
  (cost_parts : ℕ)
  (multiplier : ℤ)
  (rent : ℤ)
  (extra_expenses : ℤ)
  (profit : ℤ)
  (h1 : cost_parts = 800)
  (h2 : multiplier = 14)               -- represents 1.4 as an integer since we're working with whole numbers only
  (h3 : rent = 5000)
  (h4 : extra_expenses = 3000)
  (h5 : profit = 11200) :
  let n := 60 in                      -- The goal to prove is that the number is 60
  profit = (1.4 * 800 * n - (800 * n + rent + extra_expenses)) :=
by
  sorry

end john_builds_computers_l53_53971


namespace total_profit_division_ratio_l53_53189

theorem total_profit_division_ratio (P Q R : ℕ) 
  (investment_P1 investment_Q1 investment_R1 : ℕ)
  (profit_ratio1 : ℕ → ℕ → ℕ → Prop)
  (investment_P2 investment_Q2 investment_R2 : ℕ)
  (profit_ratio2 : ℕ → ℕ → ℕ → Prop)
  (H1 : investment_P1 = 75000)
  (H2 : investment_Q1 = 15000)
  (H3 : investment_R1 = 45000)
  (H4 : profit_ratio1 4 3 2)
  (H5 : investment_P2 = 75000 + (75000 / 2))
  (H6 : investment_Q2 = 15000 - (15000 / 4))
  (H7 : investment_R2 = 45000)
  (H8 : profit_ratio2 3 5 7) :
  profit_ratio1.to_fraction * 5 * 9 + profit_ratio2.to_fraction * 3 * 15 =
  29 / 45 ∧
  profit_ratio1.to_fraction * 3 * 9 + profit_ratio2.to_fraction * 5 * 15 =
  30 / 45 ∧
  profit_ratio1.to_fraction * 2 * 9 + profit_ratio2.to_fraction * 7 * 15 =
  31 / 45
:= sorry

end total_profit_division_ratio_l53_53189


namespace complex_quadrant_l53_53934

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l53_53934


namespace parabola_standard_eq_length_AB_l53_53056

-- Definitions
def directrix (p : ℚ) := ∀ x, x = -1
def parabola_eq (x y : ℚ) : Prop := y^2 = 4 * x
def line_eq (x y : ℚ) : Prop := y = √3 * (x - 1)
def intersection_points (x1 x2 y1 y2 : ℚ) : Prop := 
  parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ line_eq x1 y1 ∧ line_eq x2 y2

-- Proving the standard equation of the parabola
theorem parabola_standard_eq (p : ℚ) (h : directrix p) : 
  ∃ x y, parabola_eq x y :=
begin
  -- No need to provide proof steps
  sorry
end

-- Proving the length of segment AB
theorem length_AB (x1 x2 y1 y2 : ℚ) (hp : directrix p) 
  (hi : intersection_points x1 x2 y1 y2) : 
  ∣ x2 - x1 ∣ = 8/3 :=
begin
  -- No need to provide proof steps
  sorry
end

end parabola_standard_eq_length_AB_l53_53056


namespace digit_6_occurrences_100_to_999_l53_53958

theorem digit_6_occurrences_100_to_999 : 
  let count_digit_6 (n : ℕ) : ℕ := (n.digits 10).count (λ d, d = 6) in
  (List.range' 100 900).sum count_digit_6 = 280 := 
by
  sorry

end digit_6_occurrences_100_to_999_l53_53958


namespace parallel_transitive_l53_53626

-- Definitions based on the conditions
variables {α : Type*} [euclidean_geometry α]
variables {a b c : Line α}

-- Lean 4 statement of the proof problem
theorem parallel_transitive (h₁ : a ∥ b) (h₂ : b ∥ c) : a ∥ c :=
  sorry

end parallel_transitive_l53_53626


namespace find_Q_div_P_l53_53542

variable (P Q : ℚ)
variable (h_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
  P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))

theorem find_Q_div_P : Q / P = -6 / 13 := by
  sorry

end find_Q_div_P_l53_53542


namespace cinema_chairs_l53_53697

theorem cinema_chairs (chairs_between : ℕ) (h : chairs_between = 30) :
  chairs_between + 2 = 32 := by
  sorry

end cinema_chairs_l53_53697


namespace Roshesmina_pennies_l53_53475

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l53_53475


namespace a_in_A_l53_53982

def A := {x : ℝ | x ≥ 2 * Real.sqrt 2}
def a : ℝ := 3

theorem a_in_A : a ∈ A :=
by 
  sorry

end a_in_A_l53_53982


namespace sequence_term_value_l53_53402

theorem sequence_term_value :
  ∃ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = a n + 1 / 2) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_value_l53_53402


namespace derivative_y_l53_53731

noncomputable def y (x : ℝ) : ℝ := sqrt (1 - 3 * x - 2 * x ^ 2) + (3 / (2 * sqrt 2)) * arcsin ((4 * x + 3) / sqrt 17)

theorem derivative_y (x : ℝ) : 
  has_deriv_at y (-2 * x / sqrt (1 - 3 * x - 2 * x ^ 2)) x :=
sorry

end derivative_y_l53_53731


namespace tangent_rotation_bisects_segment_l53_53330

theorem tangent_rotation_bisects_segment 
  {circle : Type*} [metric_space circle]
  (center : circle)
  (radius : ℝ)
  (A : circle)
  (B : circle)
  (A' B' : circle)
  (line_tangent : A ≠ center ∧ dist A center = radius ∧ dist B center ≠ radius)
  (rotation : dist A A' = dist B B' ∧ ∃ θ, rotate A θ = A' ∧ rotate B θ = B') :
  (∀ B B', dist center B = dist center B' → 
           (let M := midpoint B B' in 
           line_connects A center A' = line_connects A' center A ∧ 
           M = midpoint B B')) := sorry

end tangent_rotation_bisects_segment_l53_53330


namespace problem_proof_l53_53376

theorem problem_proof (a b c d x : ℝ) (h1 : b ≠ 0) (h2 : x = (a + b * Real.sqrt c) / d) (h3 : 6 * x / 5 - 2 = 4 / x) : 
  a = 5 ∧ b = 1 ∧ c = 145 ∧ d = 6 → a * c * d / b = 4350 := 
by
  intro h
  have ha : a = 5 := by sorry
  have hb : b = 1 := by sorry
  have hc : c = 145 := by sorry
  have hd : d = 6 := by sorry
  rw [ha, hb, hc, hd]
  norm_num
  sorry

end problem_proof_l53_53376


namespace extra_marks_15_l53_53654

theorem extra_marks_15 {T P : ℝ} (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) (h3 : P = 120) : 
  0.45 * T - P = 15 := 
by
  sorry

end extra_marks_15_l53_53654


namespace find_k_l53_53100

theorem find_k (x y k : ℝ) (h1 : 2 * x + y = 4 * k) (h2 : x - y = k) (h3 : x + 2 * y = 12) : k = 4 :=
sorry

end find_k_l53_53100


namespace part1_part2_part3_l53_53403

-- Definitions for the sequence and sum of terms
def a : ℕ → ℤ
def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a i

-- Conditions of the problem
axiom a1 : a 1 = 0
axiom a_condition : ∀ i, |a (i + 1)| = |a i + 1|

-- Proof statements
theorem part1 : S 3 = 3 ∨ S 3 = -1 := sorry
theorem part2 (h : a 5 = -2) : S 5 = -2 := sorry
theorem part3 : ∃ (A : ℕ → ℤ), S 2022 = 1011 ∧ (a 2023 = 62 ∨ a 2023 = -62) := sorry

end part1_part2_part3_l53_53403


namespace total_pens_count_l53_53605

def total_pens (red black blue : ℕ) : ℕ :=
  red + black + blue

theorem total_pens_count :
  let red := 8
  let black := red + 10
  let blue := red + 7
  total_pens red black blue = 41 :=
by
  sorry

end total_pens_count_l53_53605


namespace fourth_power_of_cube_of_third_smallest_prime_l53_53259

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l53_53259


namespace range_of_a_l53_53781

open Function

theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ > f x₂) (a : ℝ) (h_gt : f a > f 2) : a < -2 ∨ a > 2 :=
  sorry

end range_of_a_l53_53781


namespace math_problems_l53_53459

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end math_problems_l53_53459


namespace log_sqrt10_1000sqrt10_l53_53008

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l53_53008


namespace prob_of_odd_sum_of_dice_rolls_l53_53240

-- Define the probability calculation context
def prob_sum_odd_three_coins : ℚ :=
  let prob_0_heads := (1/2)^3 * 0 in
  let prob_1_head := (3 * (1/2)^3) * (1/2) in
  let prob_2_heads := (3 * (1/2)^3) * (1/2) in
  let prob_3_heads := (1/2)^3 * (1/2) in
  prob_1_head + prob_2_heads + prob_3_heads

theorem prob_of_odd_sum_of_dice_rolls :
  prob_sum_odd_three_coins = 7/16 :=
sorry

end prob_of_odd_sum_of_dice_rolls_l53_53240


namespace math_problems_l53_53460

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end math_problems_l53_53460


namespace sum_of_roots_l53_53816

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l53_53816


namespace angle_BEC_is_70_l53_53190

-- Definitions and conditions
variable {A B C D E : Point}
variable (ω : Circle)
variable (inscribed : IsInscribed ω ⟨A, B, C, D, E⟩)
variable (diameter_AC : IsDiameter ω A C)
variable (angle_ADB : Angle A D B = 20)

-- The theorem to prove
theorem angle_BEC_is_70 (ω : Circle) (inscribed : IsInscribed ω ⟨A, B, C, D, E⟩)
  (diameter_AC : IsDiameter ω A C) (angle_ADB : Angle A D B = 20) :
  Angle B E C = 70 :=
sorry

end angle_BEC_is_70_l53_53190


namespace fourth_power_of_third_smallest_prime_cube_l53_53276

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l53_53276


namespace fourth_power_of_cube_third_smallest_prime_l53_53290

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l53_53290


namespace probability_at_least_one_two_l53_53337

theorem probability_at_least_one_two :
  (∃ (a b c : ℕ), a + b = c ∧
    a ∈ {1, 2, 3, 4, 5, 6} ∧
    b ∈ {1, 2, 3, 4, 5, 6} ∧
    c ∈ {1, 2, 3, 4, 5, 6} ∧
    (a = 2 ∨ b = 2 ∨ c = 2)) →
  8/15 :=
begin
  sorry
end

end probability_at_least_one_two_l53_53337


namespace collinear_under_inversion_l53_53644

-- Assuming some definitions of Points, Circles, and Inversions
variable (O : Point) (R : ℝ)
variable (A B : Point)
variable (A' B' : Point)
variable [Inversion : InversionStruct O R]

-- Points A and B do not lie on the circle of inversion
axiom A_not_on_circle : ¬OnCircle A (circle O R)
axiom B_not_on_circle : ¬OnCircle B (circle O R)

-- Conditions: Points A' and B' are images of A and B under inversion w.r.t. the circle
axiom A_inversion : Inverted O R A A'
axiom B_inversion : Inverted O R B B'

-- We need to prove that points A, B, A', and B' lie on a single circle
theorem collinear_under_inversion :
    CyclicQuadrilateral A B A' B' :=
  sorry

end collinear_under_inversion_l53_53644


namespace sum_unchanged_difference_changes_l53_53602

-- Definitions from conditions
def original_sum (a b c : ℤ) := a + b + c
def new_first (a : ℤ) := a - 329
def new_second (b : ℤ) := b + 401

-- Problem statement for sum unchanged
theorem sum_unchanged (a b c : ℤ) (h : original_sum a b c = 1281) :
  original_sum (new_first a) (new_second b) (c - 72) = 1281 := by
  sorry

-- Definitions for difference condition
def abs_diff (x y : ℤ) := abs (x - y)
def alter_difference (a b c : ℤ) :=
  abs_diff (new_first a) (new_second b) + abs_diff (new_first a) c + abs_diff b c

-- Problem statement addressing the difference
theorem difference_changes (a b c : ℤ) (h : original_sum a b c = 1281) :
  alter_difference a b c = abs_diff (new_first a) (new_second b) + abs_diff (c - 730) (new_first a) + abs_diff (c - 730) (new_first a) := by
  sorry

end sum_unchanged_difference_changes_l53_53602


namespace find_xyz_l53_53237

theorem find_xyz :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (4 * real.sqrt (real.cbrt 8 - real.cbrt 7) = real.cbrt x + real.cbrt y - real.cbrt z) ∧ (x + y + z = 79) :=
by
  sorry

end find_xyz_l53_53237


namespace proof_problem_l53_53415

noncomputable theory

-- Definitions
variable {ℝ : Type*} [linear_ordered_field ℝ]
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
def condition_1 (x y : ℝ) : Prop := f(x + y) + f(x - y) = 2 * f(x) * f(y)
def condition_2 : Prop := f(1) = 1 / 2

-- Theorem statement
theorem proof_problem :
  (∀ x y, condition_1 f x y) →
  (∀ x, f'(x) = lim h → 0, (f(x + h) - f(x)) / h) → -- Assuming classical derivative definition
  (∀ x, ∃ f'(x)) →  -- Ensure the existence of f'(x)
  (∀ x, f'(-x) = -f'(x)) ∧  -- f'(x) is odd
  (condition_2 f) →
  ∑ i in finset.range 2028.filter (λ n, n > 0), f i = -1 :=  -- Sum from 1 to 2027 equals -1
by
  intros h1 h2 h3 h4
  sorry

end proof_problem_l53_53415


namespace jane_earnings_l53_53969

def bulbs_tulip : Nat := 20
def bulbs_iris : Nat := bulbs_tulip / 2
def bulbs_daffodil : Nat := 30
def bulbs_crocus : Nat := bulbs_daffodil * 3
def price_per_bulb : Float := 0.5

def total_bulbs : Nat := bulbs_tulip + bulbs_iris + bulbs_daffodil + bulbs_crocus
def total_earnings : Float := total_bulbs * price_per_bulb

theorem jane_earnings :
  total_earnings = 75 := by
  sorry

end jane_earnings_l53_53969


namespace CB_length_in_triangle_l53_53103

theorem CB_length_in_triangle
  (CD DA CE : ℝ)
  (h₀ : CD = 5)
  (h₁ : DA = 15)
  (h₂ : CE = 9)
  (parallel_DE_AB : true) : 
  CB = 36 :=
by
  -- Definitions using given conditions
  let CA := CD + DA
  have CA_val : CA = 20 :=
    by simp [h₀, h₁]

  -- Leverage the similarity of triangles
  have similarity_ratio : CB / CE = CA / CD :=
    by simp [h₂, h₀, CA_val]
  have ratio_val : CB / 9 = 4 :=
    by simp [CA_val, h₂, h₀]

  -- Solve for CB
  have CB_val : CB = 9 * 4 :=
    by simp [ratio_val]
  simp [CB_val] 
  sorry  -- Placeholder for the proof steps to verify the theorem

end CB_length_in_triangle_l53_53103


namespace simplify_and_evaluate_expression_l53_53570

   theorem simplify_and_evaluate_expression :
     (x y : ℝ) (hx : x = Real.sqrt 6 - 1) (hy : y = Real.sqrt 6 + 1) :
     (2 * x + y) ^ 2 + (x - y) * (x + y) - 5 * x * (x - y) = 45 :=
   by
     sorry
   
end simplify_and_evaluate_expression_l53_53570


namespace root_in_interval_l53_53599

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 6

theorem root_in_interval :
  ∃ x ∈ Ioo 1 2, f x = 0 :=
sorry

end root_in_interval_l53_53599


namespace T_n_geq_one_over_4n_l53_53999

noncomputable def x_n (n : ℕ+) : ℝ :=
  n / (n + 1 : ℝ)

theorem T_n_geq_one_over_4n (n : ℕ+) : 
  let T_n := ∏ (i : ℕ) in Finset.range n | λ i, (x_n (2 * i + 1))^2
  in T_n ≥ 1 / (4 * n) := 
sorry

end T_n_geq_one_over_4n_l53_53999


namespace angle_MB_intersect_90_l53_53979

noncomputable def triangle_ABC {A B C : Type} (BM : B ⊆ M)(right_angle : ∠ B = 90) 
  (Ha : orthocenter (ABM))(Hc : orthocenter (CBM))(HcA_inter_AHc : collinear A Hc)
  (CaH_inter_CHa : collinear C Ha): Prop :=
∃ K : Type, 
  ∃ (intersection_K : AHc ∩ CHa = K),
  angle_MB_intersect (M B K) = 90

theorem angle_MB_intersect_90 (BM : median of right-angled triangle ABC)(Hc : orthocenter of triangle ABM)(Ha : orthocenter of CBM)(intersection_K : ∃ K : Type, AHc ∩ CHa = K): 
  angle_MBK = 90 :=
by
  sorry

end angle_MB_intersect_90_l53_53979


namespace denomination_of_other_paise_coin_l53_53235

theorem denomination_of_other_paise_coin
  (total_coins : ℕ)
  (coins_20_paise : ℕ)
  (value_in_rupees : ℕ)
  (value_of_paise : ℕ) 
  (coins_of_other_denomination : ℕ) :
  total_coins = 336 →
  coins_20_paise = 260 →
  value_in_rupees = 71 →
  value_of_paise = 100 →
  coins_of_other_denomination = total_coins - coins_20_paise →
  20 * coins_20_paise + coins_of_other_denomination * 25 =
    value_in_rupees * value_of_paise :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end denomination_of_other_paise_coin_l53_53235


namespace intersection_points_l53_53593

theorem intersection_points (k : ℝ) : ∃ (P : ℝ × ℝ), P = (1, 0) ∧ ∀ x y : ℝ, (kx - y - k = 0) → (x^2 + y^2 = 2) → ∃ y1 y2 : ℝ, (y = y1 ∨ y = y2) :=
by
  sorry

end intersection_points_l53_53593


namespace digit_6_occurrences_100_to_999_l53_53957

theorem digit_6_occurrences_100_to_999 : 
  let count_digit_6 (n : ℕ) : ℕ := (n.digits 10).count (λ d, d = 6) in
  (List.range' 100 900).sum count_digit_6 = 280 := 
by
  sorry

end digit_6_occurrences_100_to_999_l53_53957


namespace proof_problem_l53_53438

variables {A B C D E F : Type} [ordered_field D] [ordered_comm_ring D]
variables {x y z : D} {S_ABC S_BDF S_EF : D}
variables (triangle_ABC : Type) (on_side_D : A B -> triangle_ABC) (on_side_E : A C -> triangle_ABC) (on_seg_F : D E -> triangle_ABC)
variable (AD_eq_xAB : A = x * A) (AE_eq_yAC : A = y * A) (DF_over_DE_eq_z : D / E = z)

theorem proof_problem 
  (h1 : S_BDF = (1 - x) * y * S_ABC)
  (h2 : S_EF = x * (1 - y) * (1 - z) * S_ABC):
  sqrt S_BDF + cbrt S_EF <= cbrt S_ABC := 
sorry

end proof_problem_l53_53438


namespace digit_6_count_in_100_to_999_l53_53953

theorem digit_6_count_in_100_to_999 : 
  (Nat.digits_count_in_range 6 100 999) = 280 := 
sorry

end digit_6_count_in_100_to_999_l53_53953


namespace triangle_height_l53_53207

theorem triangle_height (base : ℝ) (height : ℝ) (area : ℝ)
  (h_base : base = 8) (h_area : area = 16) (h_area_formula : area = (base * height) / 2) :
  height = 4 :=
by
  sorry

end triangle_height_l53_53207


namespace bus_driver_total_compensation_l53_53655

theorem bus_driver_total_compensation :
  let regular_rate := 16
  let regular_hours := 40
  let overtime_hours := 60 - regular_hours
  let overtime_rate := regular_rate + 0.75 * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1200 := by
  sorry

end bus_driver_total_compensation_l53_53655


namespace num_valid_combinations_l53_53336

-- Definitions based on the conditions
def num_herbs := 4
def num_gems := 6
def num_incompatible_gems := 3
def num_incompatible_herbs := 2

-- Statement to be proved
theorem num_valid_combinations :
  (num_herbs * num_gems) - (num_incompatible_gems * num_incompatible_herbs) = 18 :=
by
  sorry

end num_valid_combinations_l53_53336


namespace find_m_if_z_is_real_l53_53420

variable (m : ℝ)

-- Define the complex number z
def z : ℂ := (m + 2 * Complex.i) / (3 - 4 * Complex.i)

-- The statement we want to prove
theorem find_m_if_z_is_real : (Complex.im z = 0) → m = -3 / 2 :=
by sorry

end find_m_if_z_is_real_l53_53420


namespace train_crossing_time_l53_53449

-- Definitions based on the problem conditions
def length_of_train : ℝ := 165
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 660

-- Derived definitions
def total_distance : ℝ := length_of_train + length_of_bridge
def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600
def expected_time_to_cross : ℝ := 41.25

-- The statement to prove
theorem train_crossing_time :
  (total_distance / speed_of_train_mps) = expected_time_to_cross :=
by
  -- Placeholder for the actual proof
  sorry

end train_crossing_time_l53_53449


namespace parents_age_when_mark_was_born_l53_53169

noncomputable def age_mark := 18
noncomputable def age_difference := 10
noncomputable def parent_multiple := 5

theorem parents_age_when_mark_was_born :
  let age_john := age_mark - age_difference in
  let parents_current_age := age_john * parent_multiple in
  parents_current_age - age_mark = 22 :=
by
  sorry

end parents_age_when_mark_was_born_l53_53169


namespace work_time_B_l53_53308

theorem work_time_B (A_efficiency : ℕ) (B_efficiency : ℕ) (days_together : ℕ) (total_work : ℕ) :
  (A_efficiency = 2 * B_efficiency) →
  (days_together = 5) →
  (total_work = (A_efficiency + B_efficiency) * days_together) →
  (total_work / B_efficiency = 15) :=
by
  intros
  sorry

end work_time_B_l53_53308


namespace root_expression_value_l53_53534

theorem root_expression_value (p q r : ℝ) (hpq : p + q + r = 15) (hpqr : p * q + q * r + r * p = 25) (hpqrs : p * q * r = 10) :
  (p / (2 / p + q * r) + q / (2 / q + r * p) + r / (2 / r + p * q) = 175 / 12) :=
by sorry

end root_expression_value_l53_53534


namespace soccer_ball_cost_l53_53245

theorem soccer_ball_cost (x : ℝ) (soccer_balls basketballs : ℕ) 
  (soccer_ball_cost basketball_cost : ℝ) 
  (h1 : soccer_balls = 2 * basketballs)
  (h2 : 5000 = soccer_balls * soccer_ball_cost)
  (h3 : 4000 = basketballs * basketball_cost)
  (h4 : basketball_cost = soccer_ball_cost + 30)
  (eqn : 5000 / soccer_ball_cost = 2 * (4000 / basketball_cost)) :
  soccer_ball_cost = x :=
by
  sorry

end soccer_ball_cost_l53_53245


namespace complex_point_quadrant_l53_53875

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l53_53875


namespace systematic_sampling_example_l53_53323

theorem systematic_sampling_example :
  ∃ S : Finset ℕ, S = {6, 16, 26, 36, 46, 56} ∧ (∀ x y ∈ S, (x ≠ y) → (| x - y | = 10)) :=
by
  use {6, 16, 26, 36, 46, 56}
  split
  · rfl
  · intros x y hx hy hxy
    fin_cases hx <;> fin_cases hy <;> try { contradiction }
    · simp only [Nat.abs_sub, abs_of_nonneg, sub_nonneg]
      bdd_matrix
      -- check |μ - μ|
      -- done
      repeat {linarith}

end systematic_sampling_example_l53_53323


namespace sequence_formula_sum_sequence_l53_53771

variable (a_6 a_8 : ℤ)
variable (d : ℤ) (h_d_neg : d < 0)

-- Condition for roots of the quadratic equation
axiom h_quadratic_eq : a_6 * a_8 = 24
axiom h_sum_neg_10 : a_6 + a_8 = -10

-- Given roots conditions from the problem
axiom root1 : a_6 = -4
axiom root2 : a_8 = -6

def a (n : ℕ) := 2 - n

theorem sequence_formula (n : ℕ) : a n = 2 - n := by
  sorry

-- Sum of the sequence {a_n / 2^{n-1}}
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i / 2^(i-1)

theorem sum_sequence (n : ℕ) : S a n = n / 2^(n-1) := by
  sorry

end sequence_formula_sum_sequence_l53_53771


namespace total_pennies_l53_53473

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l53_53473


namespace complex_quadrant_is_first_l53_53928

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l53_53928


namespace mean_median_difference_l53_53111

structure StudentScores :=
(num_students : ℕ)
(score_60 : ℕ)
(score_75 : ℕ)
(score_82 : ℕ)
(score_88 : ℕ)
(score_92 : ℕ)

def testScores : StudentScores :=
{
    num_students := 30,
    score_60 := 5,
    score_75 := 6,
    score_82 := 8,
    score_88 := 9,
    score_92 := 3
}

theorem mean_median_difference (s : StudentScores) (h : s = testScores) : 
  let mean := (60 * s.score_60 + 75 * s.score_75 + 82 * s.score_82 + 88 * s.score_88 + 92 * s.score_92) / s.num_students
      median := 82
  in abs (mean - median) = 0.47 :=
by sorry

end mean_median_difference_l53_53111


namespace sequence_sum_squares_lt_one_l53_53539

theorem sequence_sum_squares_lt_one (x : ℕ → ℝ) (n : ℕ) (h₀ : 0 < x 1) (h₁ : x 1 < 1)
    (h₂ : ∀ k : ℕ, x (k + 1) = x k - (x k) ^ 2) :
    (∑ k in Finset.range n, (x (k + 1)) ^ 2) < 1 := 
by
  sorry

end sequence_sum_squares_lt_one_l53_53539


namespace mike_weekly_avg_time_l53_53553

theorem mike_weekly_avg_time :
  let mon_wed_fri_tv := 4 -- hours per day on Mon, Wed, Fri
  let tue_thu_tv := 3 -- hours per day on Tue, Thu
  let weekend_tv := 5 -- hours per day on weekends
  let num_mon_wed_fri := 3 -- days
  let num_tue_thu := 2 -- days
  let num_weekend := 2 -- days
  let num_days_week := 7 -- days
  let num_video_game_days := 3 -- days
  let weeks := 4 -- weeks
  let mon_wed_fri_total := mon_wed_fri_tv * num_mon_wed_fri
  let tue_thu_total := tue_thu_tv * num_tue_thu
  let weekend_total := weekend_tv * num_weekend
  let weekly_tv_time := mon_wed_fri_total + tue_thu_total + weekend_total
  let daily_avg_tv_time := weekly_tv_time / num_days_week
  let daily_video_game_time := daily_avg_tv_time / 2
  let weekly_video_game_time := daily_video_game_time * num_video_game_days
  let total_tv_time_4_weeks := weekly_tv_time * weeks
  let total_video_game_time_4_weeks := weekly_video_game_time * weeks
  let total_time_4_weeks := total_tv_time_4_weeks + total_video_game_time_4_weeks
  let weekly_avg_time := total_time_4_weeks / weeks
  weekly_avg_time = 34 := sorry

end mike_weekly_avg_time_l53_53553


namespace range_of_gfa_proof_l53_53761

noncomputable def f (a : ℝ) : ℝ :=
  ( -2 * (1 - a^2) / (a - 2 + 2 * a^2) ) - 1

def g (m: ℝ) : ℝ := 2 / (m + 1)

def gfa_range : Set ℝ :=
  { x : ℝ | ( -((1 - Real.sqrt 2) / 2) < x ∧ x < 0 ) ∨ ( 0 < x ∧ x < 1 ) }

theorem range_of_gfa_proof : 
  ∀ (a : ℝ), 
    (Real.sqrt 2 / 2 < a ∧ a < 1) →
    (g (f a) ∈ gfa_range) :=
by 
  intro a ha
  let m := f a
  have key : g m = (2 * a^2 + a - 2), from sorry
  have domain : (Real.sqrt 2 / 2 < a ∧ a < 1), from ha
  rw key
  -- Prove the rest with specific manipulations.
  sorry

end range_of_gfa_proof_l53_53761
