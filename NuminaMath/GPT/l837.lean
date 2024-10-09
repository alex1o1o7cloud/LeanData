import Mathlib

namespace base_for_784_as_CDEC_l837_83745

theorem base_for_784_as_CDEC : 
  ∃ (b : ℕ), 
  (b^3 ≤ 784 ∧ 784 < b^4) ∧ 
  (∃ C D : ℕ, C ≠ D ∧ 784 = (C * b^3 + D * b^2 + C * b + C) ∧ 
  b = 6) :=
sorry

end base_for_784_as_CDEC_l837_83745


namespace dave_winfield_home_runs_l837_83721

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l837_83721


namespace grazing_months_l837_83748

theorem grazing_months :
  ∀ (m : ℕ),
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * b_months
  let c_ox_months := c_oxen * m
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  let c_part := (c_ox_months : ℝ) / (total_ox_months : ℝ) * rent
  (c_part = c_share) → m = 3 :=
by { sorry }

end grazing_months_l837_83748


namespace correct_mean_after_correction_l837_83736

theorem correct_mean_after_correction
  (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ)
  (h : n = 30) (h_mean : incorrect_mean = 150) (h_incorrect_value : incorrect_value = 135) (h_correct_value : correct_value = 165) :
  (incorrect_mean * n - incorrect_value + correct_value) / n = 151 :=
  by
  sorry

end correct_mean_after_correction_l837_83736


namespace option_D_correct_l837_83737

theorem option_D_correct (f : ℕ+ → ℕ) (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (hf : f 4 ≥ 25) : ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
by
  sorry

end option_D_correct_l837_83737


namespace point_on_circle_x_value_l837_83744

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end point_on_circle_x_value_l837_83744


namespace divisible_iff_exists_t_l837_83751

theorem divisible_iff_exists_t (a b m α : ℤ) (h_coprime : Int.gcd a m = 1) (h_divisible : a * α + b ≡ 0 [ZMOD m]):
  ∀ x : ℤ, (a * x + b ≡ 0 [ZMOD m]) ↔ ∃ t : ℤ, x = α + m * t :=
sorry

end divisible_iff_exists_t_l837_83751


namespace range_of_a_l837_83762

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a) ∧
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1 / 7 ≤ a ∧ a < 1 / 3 := 
sorry

end range_of_a_l837_83762


namespace milk_replacement_problem_l837_83774

theorem milk_replacement_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 90)
  (h3 : (90 - x) - ((90 - x) * x / 90) = 72.9) : x = 9 :=
sorry

end milk_replacement_problem_l837_83774


namespace min_value_of_x_plus_y_l837_83700

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0) (h3 : y + 9 * x = x * y)

-- The statement of the problem
theorem min_value_of_x_plus_y : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l837_83700


namespace find_f_15_l837_83708

theorem find_f_15
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - 2 * y) + 3 * x ^ 2 + 2) :
  f 15 = 1202 := 
sorry

end find_f_15_l837_83708


namespace flour_needed_for_one_loaf_l837_83724

-- Define the conditions
def flour_needed_for_two_loaves : ℚ := 5 -- cups of flour needed for two loaves

-- Define the theorem to prove
theorem flour_needed_for_one_loaf : flour_needed_for_two_loaves / 2 = 2.5 :=
by 
  -- Skip the proof.
  sorry

end flour_needed_for_one_loaf_l837_83724


namespace value_of_S_2016_l837_83740

variable (a d : ℤ)
variable (S : ℕ → ℤ)

-- Definitions of conditions
def a_1 := -2014
def sum_2012 := S 2012
def sum_10 := S 10
def S_n (n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom S_condition : (sum_2012 / 2012) - (sum_10 / 10) = 2002
axiom S_def : ∀ n : ℕ, S n = S_n n

-- The theorem to be proved
theorem value_of_S_2016 : S 2016 = 2016 := by
  sorry

end value_of_S_2016_l837_83740


namespace total_birds_on_fence_l837_83710

theorem total_birds_on_fence (initial_pairs : ℕ) (birds_per_pair : ℕ) 
                             (new_pairs : ℕ) (new_birds_per_pair : ℕ)
                             (initial_birds : initial_pairs * birds_per_pair = 24)
                             (new_birds : new_pairs * new_birds_per_pair = 8) : 
                             ((initial_pairs * birds_per_pair) + (new_pairs * new_birds_per_pair) = 32) :=
sorry

end total_birds_on_fence_l837_83710


namespace divisibility_of_special_number_l837_83757

theorem divisibility_of_special_number (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
    ∃ d : ℕ, 100100 * a + 10010 * b + 1001 * c = 11 * d := 
sorry

end divisibility_of_special_number_l837_83757


namespace sachin_rahul_age_ratio_l837_83764

theorem sachin_rahul_age_ratio 
(S_age : ℕ) 
(R_age : ℕ) 
(h1 : R_age = S_age + 4) 
(h2 : S_age = 14) : 
S_age / Int.gcd S_age R_age = 7 ∧ R_age / Int.gcd S_age R_age = 9 := 
by 
sorry

end sachin_rahul_age_ratio_l837_83764


namespace max_value_of_product_l837_83709

theorem max_value_of_product (x y z w : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) (h_sum : x + y + z + w = 1) : 
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 :=
by
  sorry

end max_value_of_product_l837_83709


namespace smallest_number_satisfies_conditions_l837_83731

-- Define the number we are looking for
def number : ℕ := 391410

theorem smallest_number_satisfies_conditions :
  (number % 7 = 2) ∧
  (number % 11 = 2) ∧
  (number % 13 = 2) ∧
  (number % 17 = 3) ∧
  (number % 23 = 0) ∧
  (number % 5 = 0) :=
by
  -- We need to prove that 391410 satisfies all the given conditions.
  -- This proof will include detailed steps to verify each condition
  sorry

end smallest_number_satisfies_conditions_l837_83731


namespace frustum_midsection_area_relation_l837_83718

theorem frustum_midsection_area_relation 
  (S₁ S₂ S₀ : ℝ) 
  (h₁: 0 ≤ S₁ ∧ 0 ≤ S₂ ∧ 0 ≤ S₀)
  (h₂: ∃ a h, (a / (a + 2 * h))^2 = S₂ / S₁ ∧ (a / (a + h))^2 = S₂ / S₀) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := 
sorry

end frustum_midsection_area_relation_l837_83718


namespace probability_of_black_ball_l837_83747

/-- Let the probability of drawing a red ball be 0.42, and the probability of drawing a white ball be 0.28. Prove that the probability of drawing a black ball is 0.3. -/
theorem probability_of_black_ball (p_red p_white p_black : ℝ) (h1 : p_red = 0.42) (h2 : p_white = 0.28) (h3 : p_red + p_white + p_black = 1) : p_black = 0.3 :=
by
  sorry

end probability_of_black_ball_l837_83747


namespace penultimate_digit_even_l837_83755

theorem penultimate_digit_even (n : ℕ) (h : n > 2) : ∃ k : ℕ, ∃ d : ℕ, d % 2 = 0 ∧ 10 * d + k = (3 ^ n) % 100 :=
sorry

end penultimate_digit_even_l837_83755


namespace find_g_2022_l837_83720

def g : ℝ → ℝ := sorry -- This is pre-defined to say there exists such a function

theorem find_g_2022 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y)) :
  g 2022 = 4086462 :=
sorry

end find_g_2022_l837_83720


namespace division_of_fractions_l837_83743

theorem division_of_fractions : (2 / 3) / (1 / 4) = (8 / 3) := by
  sorry

end division_of_fractions_l837_83743


namespace find_m_eq_5_l837_83778

-- Definitions for the problem conditions
def f (x m : ℝ) := 2 * x + m

theorem find_m_eq_5 (m : ℝ) (a b : ℝ) :
  (a = f 0 m) ∧ (b = f m m) ∧ ((b - a) = (m - 0 + 5)) → m = 5 :=
by
  sorry

end find_m_eq_5_l837_83778


namespace shaded_area_is_10_l837_83734

-- Definitions based on conditions:
def rectangle_area : ℕ := 12
def unshaded_triangle_area : ℕ := 2

-- Proof statement without the actual proof.
theorem shaded_area_is_10 : rectangle_area - unshaded_triangle_area = 10 := by
  sorry

end shaded_area_is_10_l837_83734


namespace buildingC_floors_if_five_times_l837_83732

-- Defining the number of floors in Building B
def floorsBuildingB : ℕ := 13

-- Theorem to prove the number of floors in Building C if it had five times as many floors as Building B
theorem buildingC_floors_if_five_times (FB : ℕ) (h : FB = floorsBuildingB) : (5 * FB) = 65 :=
by
  rw [h]
  exact rfl

end buildingC_floors_if_five_times_l837_83732


namespace number_of_crowns_l837_83719

-- Define the conditions
def feathers_per_crown : ℕ := 7
def total_feathers : ℕ := 6538

-- Theorem statement
theorem number_of_crowns : total_feathers / feathers_per_crown = 934 :=
by {
  sorry  -- proof omitted
}

end number_of_crowns_l837_83719


namespace midpoint_distance_trapezoid_l837_83789

theorem midpoint_distance_trapezoid (x : ℝ) : 
  let AD := x
  let BC := 5
  PQ = (|x - 5| / 2) :=
sorry

end midpoint_distance_trapezoid_l837_83789


namespace find_a_l837_83776

theorem find_a (a : ℝ) (h : a * (1 : ℝ)^2 - 6 * 1 + 3 = 0) : a = 3 :=
by
  sorry

end find_a_l837_83776


namespace factorize_expression_l837_83767

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l837_83767


namespace expression_of_f_f_increasing_on_interval_inequality_solution_l837_83770

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

end expression_of_f_f_increasing_on_interval_inequality_solution_l837_83770


namespace H_function_is_f_x_abs_x_l837_83711

-- Definition: A function f is odd if ∀ x ∈ ℝ, f(-x) = -f(x)
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition: A function f is strictly increasing if ∀ x1, x2 ∈ ℝ, x1 < x2 implies f(x1) < f(x2)
def is_strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- Define the function f(x) = x * |x|
def f (x : ℝ) : ℝ := x * abs x

-- The main theorem which states that f(x) = x * |x| is an "H function"
theorem H_function_is_f_x_abs_x : is_odd f ∧ is_strictly_increasing f :=
  sorry

end H_function_is_f_x_abs_x_l837_83711


namespace turtle_marathon_time_l837_83753

/-- Given a marathon distance of 42 kilometers and 195 meters and a turtle's speed of 15 meters per minute,
prove that the turtle will reach the finish line in 1 day, 22 hours, and 53 minutes. -/
theorem turtle_marathon_time :
  let speed := 15 -- meters per minute
  let distance_km := 42 -- kilometers
  let distance_m := 195 -- meters
  let total_distance := distance_km * 1000 + distance_m -- total distance in meters
  let time_min := total_distance / speed -- time to complete the marathon in minutes
  let hours := time_min / 60 -- time to complete the marathon in hours (division and modulus)
  let minutes := time_min % 60 -- remaining minutes after converting total minutes to hours
  let days := hours / 24 -- time to complete the marathon in days (division and modulus)
  let remaining_hours := hours % 24 -- remaining hours after converting total hours to days
  (days, remaining_hours, minutes) = (1, 22, 53) -- expected result
:= 
sorry

end turtle_marathon_time_l837_83753


namespace find_g_seven_l837_83726

variable {g : ℝ → ℝ}

theorem find_g_seven (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 :=
by
  sorry

end find_g_seven_l837_83726


namespace number_of_common_tangents_of_two_circles_l837_83749

theorem number_of_common_tangents_of_two_circles 
  (x y : ℝ)
  (circle1 : x^2 + y^2 = 1)
  (circle2 : x^2 + y^2 - 6 * x - 8 * y + 9 = 0) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_common_tangents_of_two_circles_l837_83749


namespace roots_are_distinct_l837_83780

theorem roots_are_distinct (a x1 x2 : ℝ) (h : x1 ≠ x2) :
  (∀ x, x^2 - a*x - 2 = 0 → x = x1 ∨ x = x2) → x1 ≠ x2 := sorry

end roots_are_distinct_l837_83780


namespace upper_bound_of_expression_l837_83786

theorem upper_bound_of_expression (n : ℤ) (h1 : ∀ (n : ℤ), 4 * n + 7 > 1 ∧ 4 * n + 7 < 111) :
  ∃ U, (∀ (n : ℤ), 4 * n + 7 < U) ∧ 
       (∀ (n : ℤ), 4 * n + 7 < U ↔ 4 * n + 7 < 111) ∧ 
       U = 111 :=
by
  sorry

end upper_bound_of_expression_l837_83786


namespace third_angle_is_90_triangle_is_right_l837_83717

-- Define the given angles
def angle1 : ℝ := 56
def angle2 : ℝ := 34

-- Define the sum of angles in a triangle
def angle_sum : ℝ := 180

-- Define the third angle
def third_angle : ℝ := angle_sum - angle1 - angle2

-- Prove that the third angle is 90 degrees
theorem third_angle_is_90 : third_angle = 90 := by
  sorry

-- Define the type of the triangle based on the largest angle
def is_right_triangle : Prop := third_angle = 90

-- Prove that the triangle is a right triangle
theorem triangle_is_right : is_right_triangle := by
  sorry

end third_angle_is_90_triangle_is_right_l837_83717


namespace sin_B_value_triangle_area_l837_83722

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end sin_B_value_triangle_area_l837_83722


namespace simon_practice_hours_l837_83798

theorem simon_practice_hours (x : ℕ) (h : (12 + 16 + 14 + x) / 4 ≥ 15) : x = 18 := 
by {
  -- placeholder for the proof
  sorry
}

end simon_practice_hours_l837_83798


namespace regular_tetrahedron_properties_l837_83750

-- Definitions
def equilateral (T : Type) : Prop := sorry -- equilateral triangle property
def equal_sides (T : Type) : Prop := sorry -- all sides equal property
def equal_angles (T : Type) : Prop := sorry -- all angles equal property

def regular (H : Type) : Prop := sorry -- regular tetrahedron property
def equal_edges (H : Type) : Prop := sorry -- all edges are equal
def equal_edge_angles (H : Type) : Prop := sorry -- angles between two edges at the same vertex are equal
def congruent_equilateral_faces (H : Type) : Prop := sorry -- faces are congruent equilateral triangles
def equal_dihedral_angles (H : Type) : Prop := sorry -- dihedral angles between adjacent faces are equal

-- Theorem statement
theorem regular_tetrahedron_properties :
  ∀ (T H : Type), 
    (equilateral T → equal_sides T ∧ equal_angles T) →
    (regular H → 
      (equal_edges H ∧ equal_edge_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_dihedral_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_edge_angles H)) :=
by
  intros T H hT hH
  sorry

end regular_tetrahedron_properties_l837_83750


namespace first_spade_second_king_prob_l837_83769

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

end first_spade_second_king_prob_l837_83769


namespace weight_distribution_l837_83746

theorem weight_distribution (x y z : ℕ) 
  (h1 : x + y + z = 100) 
  (h2 : x + 10 * y + 50 * z = 500) : 
  x = 60 ∧ y = 39 ∧ z = 1 :=
by {
  sorry
}

end weight_distribution_l837_83746


namespace city_tax_problem_l837_83754

theorem city_tax_problem :
  ∃ (x y : ℕ), 
    ((x + 3000) * (y - 10) = x * y) ∧
    ((x - 1000) * (y + 10) = x * y) ∧
    (x = 3000) ∧
    (y = 20) ∧
    (x * y = 60000) :=
by
  sorry

end city_tax_problem_l837_83754


namespace initial_calculated_average_was_23_l837_83787

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

end initial_calculated_average_was_23_l837_83787


namespace find_number_l837_83716

theorem find_number (x : ℝ) (h : x / 100 = 31.76 + 0.28) : x = 3204 := 
  sorry

end find_number_l837_83716


namespace range_of_a_l837_83723

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (x^2 - 2*a*x + a) > 0) → (a ≤ 0 ∨ a ≥ 1) :=
by
  -- Proof goes here
  sorry

end range_of_a_l837_83723


namespace find_z_l837_83763

open Complex

noncomputable def sqrt_five : ℝ := Real.sqrt 5

theorem find_z (z : ℂ) 
  (hz1 : z.re < 0) 
  (hz2 : z.im > 0) 
  (h_modulus : abs z = 3) 
  (h_real_part : z.re = -sqrt_five) : 
  z = -sqrt_five + 2 * I :=
by
  sorry

end find_z_l837_83763


namespace arith_seq_seventh_term_l837_83713

theorem arith_seq_seventh_term (a1 a25 : ℝ) (n : ℕ) (d : ℝ) (a7 : ℝ) :
  a1 = 5 → a25 = 80 → n = 25 → d = (a25 - a1) / (n - 1) → a7 = a1 + (7 - 1) * d → a7 = 23.75 :=
by
  intros h1 h2 h3 hd ha7
  sorry

end arith_seq_seventh_term_l837_83713


namespace parabola_equation_l837_83725

-- Definitions for the given conditions
def parabola_vertex_origin (y x : ℝ) : Prop := y = 0 ↔ x = 0
def axis_of_symmetry_x (y x : ℝ) : Prop := (x = -y) ↔ (x = y)
def focus_on_line (y x : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- The statement to be proved
theorem parabola_equation :
  ∀ (y x : ℝ),
  (parabola_vertex_origin y x) ∧ (axis_of_symmetry_x y x) ∧ (focus_on_line y x) →
  y^2 = 16 * x :=
by
  intros y x h
  sorry

end parabola_equation_l837_83725


namespace base8_to_base10_conversion_l837_83784

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


end base8_to_base10_conversion_l837_83784


namespace find_distinct_numbers_l837_83712

theorem find_distinct_numbers (k l : ℕ) (h : 64 / k = 4 * (64 / l)) : k = 1 ∧ l = 4 :=
by
  sorry

end find_distinct_numbers_l837_83712


namespace p_p_values_l837_83766

def p (x y : ℤ) : ℤ :=
if 0 ≤ x ∧ 0 ≤ y then x + 2*y
else if x < 0 ∧ y < 0 then x - 3*y
else 4*x + y

theorem p_p_values : p (p 2 (-2)) (p (-3) (-1)) = 6 :=
by
  sorry

end p_p_values_l837_83766


namespace original_price_l837_83759

variable (p q : ℝ)

theorem original_price (x : ℝ)
  (hp : x * (1 + p / 100) * (1 - q / 100) = 1) :
  x = 10000 / (10000 + 100 * (p - q) - p * q) :=
sorry

end original_price_l837_83759


namespace geometric_sequence_common_ratio_l837_83793

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 3 * a 2 - 5 * a 1)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a n < a (n + 1))
  (h4 : ∀ n, a (n + 1) = a n * q) : 
  q = 5 :=
  sorry

end geometric_sequence_common_ratio_l837_83793


namespace monotonically_increasing_range_of_a_l837_83701

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
sorry

end monotonically_increasing_range_of_a_l837_83701


namespace reflection_proof_l837_83705

def original_center : (ℝ × ℝ) := (8, -3)
def reflection_line (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)
def reflected_center : (ℝ × ℝ) := reflection_line original_center

theorem reflection_proof : reflected_center = (-3, -8) := by
  sorry

end reflection_proof_l837_83705


namespace rectangle_perimeter_l837_83794

theorem rectangle_perimeter (a b : ℤ) (h1 : a ≠ b) (h2 : 2 * (2 * a + 2 * b) - a * b = 12) : 2 * (a + b) = 26 :=
sorry

end rectangle_perimeter_l837_83794


namespace xy_difference_l837_83783

theorem xy_difference (x y : ℚ) (h1 : 3 * x - 4 * y = 17) (h2 : x + 3 * y = 5) : x - y = 73 / 13 :=
by
  sorry

end xy_difference_l837_83783


namespace cost_of_remaining_shirt_l837_83756

theorem cost_of_remaining_shirt :
  ∀ (shirts total_cost cost_per_shirt remaining_shirt_cost : ℕ),
  shirts = 5 →
  total_cost = 85 →
  cost_per_shirt = 15 →
  (3 * cost_per_shirt) + (2 * remaining_shirt_cost) = total_cost →
  remaining_shirt_cost = 20 :=
by
  intros shirts total_cost cost_per_shirt remaining_shirt_cost
  intros h_shirts h_total h_cost_per_shirt h_equation
  sorry

end cost_of_remaining_shirt_l837_83756


namespace a_share_is_2500_l837_83704

theorem a_share_is_2500
  (x : ℝ)
  (h1 : 4 * x = 3 * x + 500)
  (h2 : 6 * x = 2 * 2 * x) : 5 * x = 2500 :=
by 
  sorry

end a_share_is_2500_l837_83704


namespace triangle_circumradius_l837_83799

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) : 
  ∃ r : ℝ, r = 5 :=
by
  sorry

end triangle_circumradius_l837_83799


namespace monotonically_decreasing_implies_a_geq_3_l837_83772

noncomputable def f (x a : ℝ): ℝ := x^3 - a * x - 1

theorem monotonically_decreasing_implies_a_geq_3 : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x a ≤ f x 3) →
  a ≥ 3 := 
sorry

end monotonically_decreasing_implies_a_geq_3_l837_83772


namespace volleyball_team_selection_l837_83758

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (n.choose k)

theorem volleyball_team_selection : 
  let quadruplets := ["Bella", "Bianca", "Becca", "Brooke"];
  let total_players := 16;
  let starters := 7;
  let num_quadruplets := quadruplets.length;
  ∃ ways : ℕ, 
    ways = binom num_quadruplets 3 * binom (total_players - num_quadruplets) (starters - 3) 
    ∧ ways = 1980 :=
by
  sorry

end volleyball_team_selection_l837_83758


namespace desks_built_by_carpenters_l837_83781

theorem desks_built_by_carpenters (h : 2 * 2.5 * r ≥ 2 * r) : 4 * 5 * r ≥ 8 * r :=
by
  sorry

end desks_built_by_carpenters_l837_83781


namespace tangent_line_at_1_extreme_points_range_of_a_l837_83742

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x ^ 2 - 3 * x + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b ∧ m = 1 ∧ b = -1 := sorry

theorem extreme_points (a : ℝ) :
  (0 < a ∧ a <= 8 / 9 → ∀ x, 0 < x → f x a = 0) ∧
  (a > 8 / 9 → ∃ x1 x2, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x1 → f x a = 0) ∧
   (∀ x, x1 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) ∧
  (a < 0 → ∃ x1 x2, x1 < 0 ∧ 0 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a >= 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end tangent_line_at_1_extreme_points_range_of_a_l837_83742


namespace forgotten_angles_sum_l837_83738

theorem forgotten_angles_sum (n : ℕ) (h : (n-2) * 180 = 3240 + x) : x = 180 :=
by {
  sorry
}

end forgotten_angles_sum_l837_83738


namespace total_height_of_sculpture_and_base_l837_83728

def height_of_sculpture_m : Float := 0.88
def height_of_base_cm : Float := 20
def meter_to_cm : Float := 100

theorem total_height_of_sculpture_and_base :
  (height_of_sculpture_m * meter_to_cm + height_of_base_cm) = 108 :=
by
  sorry

end total_height_of_sculpture_and_base_l837_83728


namespace cost_of_pants_is_250_l837_83765

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l837_83765


namespace ensure_nonempty_intersection_l837_83791

def M (x : ℝ) : Prop := x ≤ 1
def N (x : ℝ) (p : ℝ) : Prop := x > p

theorem ensure_nonempty_intersection (p : ℝ) : (∃ x : ℝ, M x ∧ N x p) ↔ p < 1 :=
by
  sorry

end ensure_nonempty_intersection_l837_83791


namespace xy_system_l837_83777

theorem xy_system (x y : ℚ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 :=
by
  sorry

end xy_system_l837_83777


namespace square_side_length_l837_83715

theorem square_side_length (s : ℝ) (h : s^2 = 3 * 4 * s) : s = 12 :=
by
  sorry

end square_side_length_l837_83715


namespace complement_A_union_B_l837_83707

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l837_83707


namespace tabs_per_window_l837_83782

def totalTabs (browsers windowsPerBrowser tabsOpened : Nat) : Nat :=
  tabsOpened / (browsers * windowsPerBrowser)

theorem tabs_per_window : totalTabs 2 3 60 = 10 := by
  sorry

end tabs_per_window_l837_83782


namespace elsa_final_marbles_l837_83790

def start_marbles : ℕ := 40
def lost_breakfast : ℕ := 3
def given_susie : ℕ := 5
def new_marbles : ℕ := 12
def returned_marbles : ℕ := 2 * given_susie

def final_marbles : ℕ :=
  start_marbles - lost_breakfast - given_susie + new_marbles + returned_marbles

theorem elsa_final_marbles : final_marbles = 54 := by
  sorry

end elsa_final_marbles_l837_83790


namespace mean_age_of_oldest_three_l837_83727

theorem mean_age_of_oldest_three (x : ℕ) (h : (x + (x + 1) + (x + 2)) / 3 = 6) : 
  (((x + 4) + (x + 5) + (x + 6)) / 3 = 10) := 
by
  sorry

end mean_age_of_oldest_three_l837_83727


namespace stacy_grew_more_l837_83733

variable (initial_height_stacy current_height_stacy brother_growth stacy_growth_more : ℕ)

-- Conditions
def stacy_initial_height : initial_height_stacy = 50 := by sorry
def stacy_current_height : current_height_stacy = 57 := by sorry
def brother_growth_last_year : brother_growth = 1 := by sorry

-- Compute Stacy's growth
def stacy_growth : ℕ := current_height_stacy - initial_height_stacy

-- Prove the difference in growth
theorem stacy_grew_more :
  stacy_growth - brother_growth = stacy_growth_more → stacy_growth_more = 6 := 
by sorry

end stacy_grew_more_l837_83733


namespace break_even_point_l837_83714

def cost_of_commodity (a : ℝ) : ℝ := a

def profit_beginning_of_month (a : ℝ) : ℝ := 100 + (a + 100) * 0.024

def profit_end_of_month : ℝ := 115

theorem break_even_point (a : ℝ) : profit_end_of_month - profit_beginning_of_month a = 0 → a = 525 := 
by sorry

end break_even_point_l837_83714


namespace correct_statements_l837_83741

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * (abs x) + b * x + c

theorem correct_statements (b c : ℝ) :
  (∀ x, c = 0 → f (-x) b 0 = - f x b 0) ∧
  (∀ x, b = 0 → c > 0 → (f x 0 c = 0 → x = 0) ∧ ∀ y, f y 0 c ≤ 0) ∧
  (∀ x, ∃ k : ℝ, f (k + x) b c = f (k - x) b c) ∧
  ¬(∀ x, x > 0 → f x b c = c - b^2 / 2) :=
by
  sorry

end correct_statements_l837_83741


namespace find_xy_l837_83761

variable (x y : ℝ)

theorem find_xy (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end find_xy_l837_83761


namespace calculate_length_of_floor_l837_83729

-- Define the conditions and the objective to prove
variable (breadth length : ℝ)
variable (cost rate : ℝ)
variable (area : ℝ)

-- Given conditions
def length_more_by_percentage : Prop := length = 2 * breadth
def painting_cost : Prop := cost = 529 ∧ rate = 3

-- Objective
def length_of_floor : ℝ := 2 * breadth

theorem calculate_length_of_floor : 
  (length_more_by_percentage breadth length) →
  (painting_cost cost rate) →
  length_of_floor breadth = 18.78 :=
by
  sorry

end calculate_length_of_floor_l837_83729


namespace total_amount_for_gifts_l837_83702

theorem total_amount_for_gifts (workers_per_block : ℕ) (worth_per_gift : ℕ) (number_of_blocks : ℕ)
  (h1 : workers_per_block = 100) (h2 : worth_per_gift = 4) (h3 : number_of_blocks = 10) :
  (workers_per_block * worth_per_gift * number_of_blocks = 4000) := by
  sorry

end total_amount_for_gifts_l837_83702


namespace distance_between_bars_l837_83796

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

end distance_between_bars_l837_83796


namespace find_number_l837_83792

theorem find_number (x: ℝ) (h: (6 * x) / 2 - 5 = 25) : x = 10 :=
by
  sorry

end find_number_l837_83792


namespace unit_prices_max_helmets_A_l837_83797

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

end unit_prices_max_helmets_A_l837_83797


namespace angle_in_third_quadrant_l837_83785

-- Definitions for quadrants
def in_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360
def in_third_quadrant (β : ℝ) : Prop := 180 < β ∧ β < 270

theorem angle_in_third_quadrant (α : ℝ) (h : in_fourth_quadrant α) : in_third_quadrant (180 - α) :=
by
  -- Proof goes here
  sorry

end angle_in_third_quadrant_l837_83785


namespace measure_of_angle_B_and_area_of_triangle_l837_83703

theorem measure_of_angle_B_and_area_of_triangle 
    (a b c : ℝ) 
    (A B C : ℝ) 
    (condition : 2 * c = a + (Real.cos A * (b / (Real.cos B))))
    (sum_sides : a + c = 3 * Real.sqrt 2)
    (side_b : b = 4)
    (angle_B : B = Real.pi / 3) :
    B = Real.pi / 3 ∧ 
    (1/2 * a * c * (Real.sin B) = Real.sqrt 3 / 6) :=
by
    sorry

end measure_of_angle_B_and_area_of_triangle_l837_83703


namespace number_of_middle_managers_selected_l837_83779

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

end number_of_middle_managers_selected_l837_83779


namespace range_of_a_increasing_f_on_interval_l837_83706

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Define the condition that f(x) is increasing on [4, +∞)
def isIncreasingOnInterval (a : ℝ) : Prop :=
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → f a x ≤ f a y

theorem range_of_a_increasing_f_on_interval :
  (∀ a : ℝ, isIncreasingOnInterval a → a ≥ -3) := 
by
  sorry

end range_of_a_increasing_f_on_interval_l837_83706


namespace cylinder_volume_l837_83795

theorem cylinder_volume (V_sphere : ℝ) (V_cylinder : ℝ) (R H : ℝ) 
  (h1 : V_sphere = 4 * π / 3) 
  (h2 : (4 * π * R ^ 3) / 3 = V_sphere) 
  (h3 : H = 2 * R) 
  (h4 : R = 1) : V_cylinder = 2 * π :=
by
  sorry

end cylinder_volume_l837_83795


namespace solution_set_of_cx_sq_minus_bx_plus_a_l837_83752

theorem solution_set_of_cx_sq_minus_bx_plus_a (a b c : ℝ) (h1 : a < 0)
(h2 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, cx^2 - bx + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by
  sorry

end solution_set_of_cx_sq_minus_bx_plus_a_l837_83752


namespace period_ending_time_l837_83730

theorem period_ending_time (start_time : ℕ) (rain_duration : ℕ) (no_rain_duration : ℕ) (end_time : ℕ) :
  start_time = 8 ∧ rain_duration = 4 ∧ no_rain_duration = 5 ∧ end_time = 8 + rain_duration + no_rain_duration
  → end_time = 17 :=
by
  sorry

end period_ending_time_l837_83730


namespace find_angle_l837_83773

theorem find_angle (x : ℝ) (h1 : 90 - x = (1/2) * (180 - x)) : x = 90 :=
by
  sorry

end find_angle_l837_83773


namespace ellipse_eq_max_area_AEBF_l837_83771

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

end ellipse_eq_max_area_AEBF_l837_83771


namespace distance_between_A_and_B_l837_83775

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

end distance_between_A_and_B_l837_83775


namespace find_length_PB_l837_83735

noncomputable def radius (O : Type*) : ℝ := sorry

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*}

def Point (α : Type*) := α

variables (P T A B : Point ℝ) (O : Circle ℝ) (r : ℝ)

def PA := (4 : ℝ)
def PT (AB : ℝ) := AB - 2
def PB (AB : ℝ) := 4 + AB

def power_of_a_point (PA PB PT : ℝ) := PA * PB = PT^2

theorem find_length_PB (AB : ℝ) 
  (h1 : power_of_a_point PA (PB AB) (PT AB)) 
  (h2 : PA < PB AB) : 
  PB AB = 18 := 
by 
  sorry

end find_length_PB_l837_83735


namespace custom_op_12_7_l837_83739

def custom_op (a b : ℤ) := (a + b) * (a - b)

theorem custom_op_12_7 : custom_op 12 7 = 95 := by
  sorry

end custom_op_12_7_l837_83739


namespace inequality_solution_l837_83760

theorem inequality_solution {x : ℝ} : 5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by
  sorry

end inequality_solution_l837_83760


namespace thread_length_l837_83768

def side_length : ℕ := 13

def perimeter (s : ℕ) : ℕ := 4 * s

theorem thread_length : perimeter side_length = 52 := by
  sorry

end thread_length_l837_83768


namespace no_two_digit_multiples_of_3_5_7_l837_83788

theorem no_two_digit_multiples_of_3_5_7 : ∀ n : ℕ, 10 ≤ n ∧ n < 100 → ¬ (3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := 
by
  intro n
  intro h
  intro h_div
  sorry

end no_two_digit_multiples_of_3_5_7_l837_83788
