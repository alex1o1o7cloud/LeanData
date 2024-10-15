import Mathlib

namespace NUMINAMATH_GPT_circle_equation_l35_3512

theorem circle_equation (x y : ℝ) : (x^2 = 16 * y) → (y = 4) → (x, -4) = (x, 4) → x^2 + (y-4)^2 = 64 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l35_3512


namespace NUMINAMATH_GPT_interest_rate_proof_l35_3551

-- Define the given values
def P : ℝ := 1500
def t : ℝ := 2.4
def A : ℝ := 1680

-- Define the interest rate per annum to be proven
def r : ℝ := 0.05

-- Prove that the calculated interest rate matches the given interest rate per annum
theorem interest_rate_proof 
  (principal : ℝ := P) 
  (time_period : ℝ := t) 
  (amount : ℝ := A) 
  (interest_rate : ℝ := r) :
  (interest_rate = ((amount / principal - 1) / time_period)) :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_proof_l35_3551


namespace NUMINAMATH_GPT_triple_composition_f_3_l35_3504

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end NUMINAMATH_GPT_triple_composition_f_3_l35_3504


namespace NUMINAMATH_GPT_find_T_l35_3518

variables (h K T : ℝ)
variables (h_val : 4 * h * 7 + 2 = 58)
variables (K_val : K = 9)

theorem find_T : T = 74 :=
by
  sorry

end NUMINAMATH_GPT_find_T_l35_3518


namespace NUMINAMATH_GPT_division_modulus_l35_3548

-- Definitions using the conditions
def a : ℕ := 8 * (10^9)
def b : ℕ := 4 * (10^4)
def n : ℕ := 10^6

-- Lean statement to prove the problem
theorem division_modulus (a b n : ℕ) (h : a = 8 * (10^9) ∧ b = 4 * (10^4) ∧ n = 10^6) : 
  ((a / b) % n) = 200000 := 
by 
  sorry

end NUMINAMATH_GPT_division_modulus_l35_3548


namespace NUMINAMATH_GPT_inequality_abc_l35_3524

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / (b ^ (1/2 : ℝ)) + b / (a ^ (1/2 : ℝ)) ≥ a ^ (1/2 : ℝ) + b ^ (1/2 : ℝ) :=
by { sorry }

end NUMINAMATH_GPT_inequality_abc_l35_3524


namespace NUMINAMATH_GPT_range_of_k_l35_3513

noncomputable def meets_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - y^2 = 4 ∧ y = k * x - 1

theorem range_of_k : 
  { k : ℝ | meets_hyperbola k } = { k : ℝ | k = 1 ∨ k = -1 ∨ - (Real.sqrt 5) / 2 ≤ k ∧ k ≤ (Real.sqrt 5) / 2 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l35_3513


namespace NUMINAMATH_GPT_asymptote_slope_of_hyperbola_l35_3515

theorem asymptote_slope_of_hyperbola :
  ∀ (x y : ℝ), (x ≠ 0) ∧ (y/x = 3/4 ∨ y/x = -3/4) ↔ (x^2 / 144 - y^2 / 81 = 1) := 
by
  sorry

end NUMINAMATH_GPT_asymptote_slope_of_hyperbola_l35_3515


namespace NUMINAMATH_GPT_inequality_constant_l35_3568

noncomputable def smallest_possible_real_constant : ℝ :=
  1.0625

theorem inequality_constant (C : ℝ) : 
  (∀ x y z : ℝ, (x + y + z = -1) → 
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1| ) ↔ C ≥ smallest_possible_real_constant :=
sorry

end NUMINAMATH_GPT_inequality_constant_l35_3568


namespace NUMINAMATH_GPT_isosceles_triangle_l35_3535

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

variables {A B C : ℝ}
variable (h : sin C = 2 * sin (B + C) * cos B)

theorem isosceles_triangle (h : sin C = 2 * sin (B + C) * cos B) : A = B :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_l35_3535


namespace NUMINAMATH_GPT_p_x_range_l35_3570

variable (x : ℝ)

def inequality_condition := x^2 - 5*x + 6 < 0
def polynomial_function := x^2 + 5*x + 6

theorem p_x_range (x_ineq : inequality_condition x) : 
  20 < polynomial_function x ∧ polynomial_function x < 30 :=
sorry

end NUMINAMATH_GPT_p_x_range_l35_3570


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l35_3577

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :
  (∀ x y : ℝ, (y = -2 * x + 1 → ∃ x₁ y₁ x₂ y₂ : ℝ, (y₁ = -2 * x₁ + 1 ∧ y₂ = -2 * x₂ + 1) ∧ 
    (x₁ / a * x₁ / a + y₁ / b * y₁ / b = 1) ∧ (x₂ / a * x₂ / a + y₂ / b * y₂ / b = 1) ∧ 
    ((x₁ + x₂) / 2 = 4 * (y₁ + y₂) / 2)) → (x / a)^2 + (y / b)^2 = 1) →
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = (Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l35_3577


namespace NUMINAMATH_GPT_math_problem_l35_3566

noncomputable def triangle_conditions (a b c A B C : ℝ) := 
  (2 * b - c) / a = (Real.cos C) / (Real.cos A) ∧ 
  a = Real.sqrt 5 ∧
  1 / 2 * b * c * (Real.sin A) = Real.sqrt 3 / 2

theorem math_problem (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  A = π / 3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 11 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l35_3566


namespace NUMINAMATH_GPT_sum_of_coefficients_l35_3507

theorem sum_of_coefficients (a : ℕ → ℝ) :
  (∀ x : ℝ, (2 - x) ^ 10 = a 0 + a 1 * x + a 2 * x ^ 2 + a 3 * x ^ 3 + a 4 * x ^ 4 + a 5 * x ^ 5 + a 6 * x ^ 6 + a 7 * x ^ 7 + a 8 * x ^ 8 + a 9 * x ^ 9 + a 10 * x ^ 10) →
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 1 →
  a 0 = 1024 →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -1023 :=  
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l35_3507


namespace NUMINAMATH_GPT_twentieth_fisherman_catch_l35_3589

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end NUMINAMATH_GPT_twentieth_fisherman_catch_l35_3589


namespace NUMINAMATH_GPT_sequence_sixth_term_l35_3536

theorem sequence_sixth_term (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) 
  (h2 : ∀ n :ℕ, n > 0 → a (n + 1) = 2 * a n) 
  (h3 : a 1 = 3) : 
  a 6 = 96 := 
by
  sorry

end NUMINAMATH_GPT_sequence_sixth_term_l35_3536


namespace NUMINAMATH_GPT_fraction_of_friends_l35_3575

variable (x y : ℕ) -- number of first-grade students and sixth-grade students

-- Conditions from the problem
def condition1 : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * x = b * y ∧ 1 / 3 = a / (a + b)
def condition2 : Prop := ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c * y = d * x ∧ 2 / 5 = c / (c + d)

-- Theorem statement to prove that the fraction of students who are friends is 4/11
theorem fraction_of_friends (h1 : condition1 x y) (h2 : condition2 x y) :
  (1 / 3 : ℚ) * y + (2 / 5 : ℚ) * x / (x + y) = 4 / 11 :=
sorry

end NUMINAMATH_GPT_fraction_of_friends_l35_3575


namespace NUMINAMATH_GPT_Joel_contributed_22_toys_l35_3539

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end NUMINAMATH_GPT_Joel_contributed_22_toys_l35_3539


namespace NUMINAMATH_GPT_exists_f_gcd_form_l35_3542

noncomputable def f : ℤ → ℕ := sorry

theorem exists_f_gcd_form :
  (∀ x y : ℤ, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) →
  ∃ m n : ℕ, (0 < m ∧ 0 < n) ∧ (∀ x : ℤ, f x = Nat.gcd (m + Int.natAbs x) n) :=
sorry

end NUMINAMATH_GPT_exists_f_gcd_form_l35_3542


namespace NUMINAMATH_GPT_ducks_in_marsh_l35_3597

theorem ducks_in_marsh 
  (num_geese : ℕ) 
  (total_birds : ℕ) 
  (num_ducks : ℕ)
  (h1 : num_geese = 58) 
  (h2 : total_birds = 95) 
  (h3 : total_birds = num_geese + num_ducks) : 
  num_ducks = 37 :=
by
  sorry

end NUMINAMATH_GPT_ducks_in_marsh_l35_3597


namespace NUMINAMATH_GPT_systematic_sampling_second_group_l35_3527

theorem systematic_sampling_second_group
    (N : ℕ) (n : ℕ) (k : ℕ := N / n)
    (number_from_16th_group : ℕ)
    (number_from_1st_group : ℕ := number_from_16th_group - 15 * k)
    (number_from_2nd_group : ℕ := number_from_1st_group + k) :
    N = 160 → n = 20 → number_from_16th_group = 123 → number_from_2nd_group = 11 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_second_group_l35_3527


namespace NUMINAMATH_GPT_panda_on_stilts_height_l35_3573

theorem panda_on_stilts_height (x : ℕ) (h_A : ℕ) 
  (h1 : h_A = x / 4) -- A Bao's height accounts for 1/4 of initial total height
  (h2 : x - 40 = 3 * h_A) -- After breaking 20 dm off each stilt, the new total height is such that A Bao's height accounts for 1/3 of this new height
  : x = 160 := 
by
  sorry

end NUMINAMATH_GPT_panda_on_stilts_height_l35_3573


namespace NUMINAMATH_GPT_find_angle_degree_l35_3511

theorem find_angle_degree (x : ℝ) (h : 90 - x = 0.4 * (180 - x)) : x = 30 := by
  sorry

end NUMINAMATH_GPT_find_angle_degree_l35_3511


namespace NUMINAMATH_GPT_sample_systematic_draw_first_group_l35_3540

theorem sample_systematic_draw_first_group :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 8 →
  (x + 15 * 8 = 126) →
  x = 6 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_sample_systematic_draw_first_group_l35_3540


namespace NUMINAMATH_GPT_geometric_sum_four_terms_l35_3591

/-- 
Given that the sequence {a_n} is a geometric sequence with the sum of its 
first n terms denoted as S_n, if S_4=1 and S_8=4, prove that a_{13}+a_{14}+a_{15}+a_{16}=27 
-/ 
theorem geometric_sum_four_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), S (n + 1) = a (n + 1) + S n) 
  (h2 : S 4 = 1) 
  (h3 : S 8 = 4) 
  : (a 13) + (a 14) + (a 15) + (a 16) = 27 := 
sorry

end NUMINAMATH_GPT_geometric_sum_four_terms_l35_3591


namespace NUMINAMATH_GPT_tangency_condition_intersection_condition_l35_3571

-- Definitions of the circle and line for the given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 8 * y + 12 = 0
def line_eq (a x y : ℝ) : Prop := a * x + y + 2 * a = 0

/-- Theorem for the tangency condition -/
theorem tangency_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = 2) →
  a = 3 / 4 :=
by
  sorry

/-- Theorem for the intersection condition -/
theorem intersection_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = Real.sqrt 2) →
  (a = 1 ∨ a = 7) →
  (∀ x y : ℝ,
    (line_eq 1 x y ∧ line_eq 7 x y ↔ 
    (7 * x + y + 14 = 0 ∨ x + y + 2 = 0))) :=
by
  sorry

end NUMINAMATH_GPT_tangency_condition_intersection_condition_l35_3571


namespace NUMINAMATH_GPT_smallest_number_of_beads_l35_3565

theorem smallest_number_of_beads (M : ℕ) (h1 : ∃ d : ℕ, M = 5 * d + 2) (h2 : ∃ e : ℕ, M = 7 * e + 2) (h3 : ∃ f : ℕ, M = 9 * f + 2) (h4 : M > 1) : M = 317 := sorry

end NUMINAMATH_GPT_smallest_number_of_beads_l35_3565


namespace NUMINAMATH_GPT_problem_intersection_l35_3544

open Set

variable {x : ℝ}

def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def C : Set ℝ := {x | (5 / 2) ≤ x ∧ x < 3}

theorem problem_intersection : A ∩ B = C := by
  sorry

end NUMINAMATH_GPT_problem_intersection_l35_3544


namespace NUMINAMATH_GPT_stewarts_theorem_l35_3554

theorem stewarts_theorem
  (A B C D : ℝ)
  (AB AC AD : ℝ)
  (BD CD BC : ℝ)
  (hD_on_BC : BD + CD = BC) :
  AB^2 * CD + AC^2 * BD - AD^2 * BC = BD * CD * BC := 
sorry

end NUMINAMATH_GPT_stewarts_theorem_l35_3554


namespace NUMINAMATH_GPT_triangle_area_l35_3516

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

end NUMINAMATH_GPT_triangle_area_l35_3516


namespace NUMINAMATH_GPT_geometric_sequence_a2_l35_3560

theorem geometric_sequence_a2 (a1 a2 a3 : ℝ) (h1 : 1 * (1/a1) = a1)
  (h2 : a1 * (1/a2) = a2) (h3 : a2 * (1/a3) = a3) (h4 : a3 * (1/4) = 4)
  (h5 : a2 > 0) : a2 = 2 := sorry

end NUMINAMATH_GPT_geometric_sequence_a2_l35_3560


namespace NUMINAMATH_GPT_find_point_P_coordinates_l35_3569

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  y > 0 ∧ x < 0 ∧ abs x = 4 ∧ abs y = 4

theorem find_point_P_coordinates : ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-4, 4) :=
by
  sorry

end NUMINAMATH_GPT_find_point_P_coordinates_l35_3569


namespace NUMINAMATH_GPT_original_number_of_bullets_each_had_l35_3596

theorem original_number_of_bullets_each_had (x : ℕ) (h₁ : 5 * (x - 4) = x) : x = 5 := 
sorry

end NUMINAMATH_GPT_original_number_of_bullets_each_had_l35_3596


namespace NUMINAMATH_GPT_minji_combinations_l35_3541

theorem minji_combinations : (3 * 5) = 15 :=
by sorry

end NUMINAMATH_GPT_minji_combinations_l35_3541


namespace NUMINAMATH_GPT_find_a_b_and_water_usage_l35_3531

noncomputable def water_usage_april (a : ℝ) :=
  (15 * (a + 0.8) = 45)

noncomputable def water_usage_may (a b : ℝ) :=
  (17 * (a + 0.8) + 8 * (b + 0.8) = 91)

noncomputable def water_usage_june (a b x : ℝ) :=
  (17 * (a + 0.8) + 13 * (b + 0.8) + (x - 30) * 6.8 = 150)

theorem find_a_b_and_water_usage :
  ∃ (a b x : ℝ), water_usage_april a ∧ water_usage_may a b ∧ water_usage_june a b x ∧ a = 2.2 ∧ b = 4.2 ∧ x = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_b_and_water_usage_l35_3531


namespace NUMINAMATH_GPT_sum_of_roots_eq_seventeen_l35_3593

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end NUMINAMATH_GPT_sum_of_roots_eq_seventeen_l35_3593


namespace NUMINAMATH_GPT_fraction_transformed_l35_3574

variables (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab_pos : a * b > 0)

noncomputable def frac_orig := (a + 2 * b) / (2 * a * b)
noncomputable def frac_new := (3 * a + 2 * 3 * b) / (2 * 3 * a * 3 * b)

theorem fraction_transformed :
  frac_new a b = (1 / 3) * frac_orig a b :=
sorry

end NUMINAMATH_GPT_fraction_transformed_l35_3574


namespace NUMINAMATH_GPT_cubic_has_three_zeros_l35_3592

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_has_three_zeros : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
sorry

end NUMINAMATH_GPT_cubic_has_three_zeros_l35_3592


namespace NUMINAMATH_GPT_nails_needed_for_house_wall_l35_3538

theorem nails_needed_for_house_wall :
  let large_planks : Nat := 13
  let nails_per_large_plank : Nat := 17
  let additional_nails : Nat := 8
  large_planks * nails_per_large_plank + additional_nails = 229 := by
  sorry

end NUMINAMATH_GPT_nails_needed_for_house_wall_l35_3538


namespace NUMINAMATH_GPT_range_of_c_l35_3528

-- Definitions of p and q based on conditions
def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (c > 1 / 2)

-- The theorem states the required condition on c
theorem range_of_c (c : ℝ) (h : c > 0) :
  ¬(p c ∧ q c) ∧ (p c ∨ q c) ↔ (0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1) :=
sorry

end NUMINAMATH_GPT_range_of_c_l35_3528


namespace NUMINAMATH_GPT_find_m_l35_3564

def A (m : ℝ) : Set ℝ := {x | x^2 - m * x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m (m : ℝ) : (A m ∩ B).Nonempty ∧ (A m ∩ C) = ∅ → m = -2 := by
  sorry

end NUMINAMATH_GPT_find_m_l35_3564


namespace NUMINAMATH_GPT_equal_even_odd_probability_l35_3590

theorem equal_even_odd_probability : 
  let total_dice := 8
  let even_odd_combinations := Nat.choose total_dice 4
  let single_arrangement_probability := (1 / 2) ^ total_dice
  even_odd_combinations * single_arrangement_probability = 35 / 128 := by
  sorry

end NUMINAMATH_GPT_equal_even_odd_probability_l35_3590


namespace NUMINAMATH_GPT_cookies_in_jar_l35_3520

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end NUMINAMATH_GPT_cookies_in_jar_l35_3520


namespace NUMINAMATH_GPT_problem_incorrect_statement_D_l35_3534

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end NUMINAMATH_GPT_problem_incorrect_statement_D_l35_3534


namespace NUMINAMATH_GPT_product_of_primes_sum_85_l35_3581

open Nat

theorem product_of_primes_sum_85 :
  ∃ (p q : ℕ), p.Prime ∧ q.Prime ∧ p + q = 85 ∧ p * q = 166 :=
sorry

end NUMINAMATH_GPT_product_of_primes_sum_85_l35_3581


namespace NUMINAMATH_GPT_sandy_total_spent_l35_3572

def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def total_spent : ℝ := shorts_price + shirt_price + jacket_price

theorem sandy_total_spent : total_spent = 33.56 :=
by
  sorry

end NUMINAMATH_GPT_sandy_total_spent_l35_3572


namespace NUMINAMATH_GPT_pieces_missing_l35_3537

def total_pieces : ℕ := 32
def pieces_present : ℕ := 24

theorem pieces_missing : total_pieces - pieces_present = 8 := by
sorry

end NUMINAMATH_GPT_pieces_missing_l35_3537


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l35_3500

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x - 2 * y - 3 * z = 4) : 
  (x^2 + y^2 + z^2) ≥ 8 / 7 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l35_3500


namespace NUMINAMATH_GPT_calculate_product_l35_3556

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end NUMINAMATH_GPT_calculate_product_l35_3556


namespace NUMINAMATH_GPT_sum_of_six_least_n_l35_3594

def tau (n : ℕ) : ℕ := Nat.totient n -- Assuming as an example for tau definition

theorem sum_of_six_least_n (h1 : tau 8 + tau 9 = 7)
                           (h2 : tau 9 + tau 10 = 7)
                           (h3 : tau 16 + tau 17 = 7)
                           (h4 : tau 25 + tau 26 = 7)
                           (h5 : tau 121 + tau 122 = 7)
                           (h6 : tau 361 + tau 362 = 7) :
  8 + 9 + 16 + 25 + 121 + 361 = 540 :=
by sorry

end NUMINAMATH_GPT_sum_of_six_least_n_l35_3594


namespace NUMINAMATH_GPT_math_problem_proof_l35_3563

theorem math_problem_proof (n : ℕ) 
  (h1 : n / 37 = 2) 
  (h2 : n % 37 = 26) :
  48 - n / 4 = 23 := by
  sorry

end NUMINAMATH_GPT_math_problem_proof_l35_3563


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l35_3521

theorem general_term_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) (h_seq : ∀ n, a_n n = a_n 0 + n * d)
  (h_geometric : (a_n 2)^2 = a_n 1 * a_n 6)
  (h_condition : 2 * a_n 0 + a_n 1 = 1)
  (h_d_nonzero : d ≠ 0) :
  ∀ n, a_n n = (5/3) - n := 
by
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l35_3521


namespace NUMINAMATH_GPT_seventh_term_arithmetic_sequence_l35_3583

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end NUMINAMATH_GPT_seventh_term_arithmetic_sequence_l35_3583


namespace NUMINAMATH_GPT_tangent_line_through_points_of_tangency_l35_3503

noncomputable def equation_of_tangent_line (x1 y1 x y : ℝ) : Prop :=
x1 * x + (y1 - 2) * (y - 2) = 4

theorem tangent_line_through_points_of_tangency
  (x1 y1 x2 y2 : ℝ)
  (h1 : equation_of_tangent_line x1 y1 2 (-2))
  (h2 : equation_of_tangent_line x2 y2 2 (-2)) :
  (2 * x1 - 4 * (y1 - 2) = 4) ∧ (2 * x2 - 4 * (y2 - 2) = 4) →
  ∃ a b c, (a = 1) ∧ (b = -2) ∧ (c = 2) ∧ (a * x + b * y + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_points_of_tangency_l35_3503


namespace NUMINAMATH_GPT_fraction_of_milk_in_cup1_l35_3523

def initial_tea_cup1 : ℚ := 6
def initial_milk_cup2 : ℚ := 6

def tea_transferred_step2 : ℚ := initial_tea_cup1 / 3
def tea_cup1_after_step2 : ℚ := initial_tea_cup1 - tea_transferred_step2
def total_cup2_after_step2 : ℚ := initial_milk_cup2 + tea_transferred_step2

def mixture_transfer_step3 : ℚ := total_cup2_after_step2 / 2
def tea_ratio_cup2 : ℚ := tea_transferred_step2 / total_cup2_after_step2
def milk_ratio_cup2 : ℚ := initial_milk_cup2 / total_cup2_after_step2
def tea_transferred_step3 : ℚ := mixture_transfer_step3 * tea_ratio_cup2
def milk_transferred_step3 : ℚ := mixture_transfer_step3 * milk_ratio_cup2

def tea_cup1_after_step3 : ℚ := tea_cup1_after_step2 + tea_transferred_step3
def milk_cup1_after_step3 : ℚ := milk_transferred_step3

def mixture_transfer_step4 : ℚ := (tea_cup1_after_step3 + milk_cup1_after_step3) / 4
def tea_ratio_cup1_step4 : ℚ := tea_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)
def milk_ratio_cup1_step4 : ℚ := milk_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)

def tea_transferred_step4 : ℚ := mixture_transfer_step4 * tea_ratio_cup1_step4
def milk_transferred_step4 : ℚ := mixture_transfer_step4 * milk_ratio_cup1_step4

def final_tea_cup1 : ℚ := tea_cup1_after_step3 - tea_transferred_step4
def final_milk_cup1 : ℚ := milk_cup1_after_step3 - milk_transferred_step4
def final_total_liquid_cup1 : ℚ := final_tea_cup1 + final_milk_cup1

theorem fraction_of_milk_in_cup1 : final_milk_cup1 / final_total_liquid_cup1 = 3/8 := by
  sorry

end NUMINAMATH_GPT_fraction_of_milk_in_cup1_l35_3523


namespace NUMINAMATH_GPT_noah_holidays_l35_3582

theorem noah_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_total : ℕ) 
  (h1 : holidays_per_month = 3) (h2 : months_in_year = 12) (h3 : holidays_total = holidays_per_month * months_in_year) : 
  holidays_total = 36 := 
by
  sorry

end NUMINAMATH_GPT_noah_holidays_l35_3582


namespace NUMINAMATH_GPT_area_of_triangle_F1PF2P_l35_3578

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3
noncomputable def PF1 : ℝ := sorry 
noncomputable def PF2 : ℝ := sorry

-- Given conditions
def ellipse_eq_holds (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Given point P is on the ellipse
def P_on_ellipse (x y : ℝ) : Prop := ellipse_eq_holds x y

-- Given angle F1PF2
def angle_F1PF2_eq_60 : Prop := sorry

-- Proving the area of △F₁PF₂
theorem area_of_triangle_F1PF2P : S = (16 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_F1PF2P_l35_3578


namespace NUMINAMATH_GPT_intersection_points_in_plane_l35_3506

-- Define the cones with parallel axes and equal angles
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Given conditions
variable (a1 b1 c1 a2 b2 c2 k : ℝ)

-- The theorem to be proven
theorem intersection_points_in_plane (x y z : ℝ) 
  (h1 : cone1 a1 b1 c1 k x y z) (h2 : cone2 a2 b2 c2 k x y z) : 
  ∃ (A B C D : ℝ), A * x + B * y + C * z + D = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_in_plane_l35_3506


namespace NUMINAMATH_GPT_triangle_area_from_perimeter_and_inradius_l35_3599

theorem triangle_area_from_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (A : ℝ)
  (h₁ : P = 24)
  (h₂ : r = 2.5) :
  A = 30 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_from_perimeter_and_inradius_l35_3599


namespace NUMINAMATH_GPT_problem1_problem2_l35_3543

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : 2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l35_3543


namespace NUMINAMATH_GPT_sequence_all_perfect_squares_l35_3558

theorem sequence_all_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, (∃ m : ℕ, 2 * 10^n + 1 = 3 * m) ∧ (x_n = (m^2 / 9)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_all_perfect_squares_l35_3558


namespace NUMINAMATH_GPT_inequality_must_hold_l35_3530

theorem inequality_must_hold (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end NUMINAMATH_GPT_inequality_must_hold_l35_3530


namespace NUMINAMATH_GPT_percentage_men_science_majors_l35_3559

theorem percentage_men_science_majors (total_students : ℕ) (women_science_majors_ratio : ℚ) (nonscience_majors_ratio : ℚ) (men_class_ratio : ℚ) :
  women_science_majors_ratio = 0.2 → 
  nonscience_majors_ratio = 0.6 → 
  men_class_ratio = 0.4 → 
  ∃ men_science_majors_percent : ℚ, men_science_majors_percent = 0.7 :=
by
  intros h_women_science_majors h_nonscience_majors h_men_class
  sorry

end NUMINAMATH_GPT_percentage_men_science_majors_l35_3559


namespace NUMINAMATH_GPT_min_fraction_in_domain_l35_3529

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end NUMINAMATH_GPT_min_fraction_in_domain_l35_3529


namespace NUMINAMATH_GPT_increasing_function_on_interval_l35_3547

noncomputable def f_A (x : ℝ) : ℝ := 3 - x
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def f_C (x : ℝ) : ℝ := - (1 / (x + 1))
noncomputable def f_D (x : ℝ) : ℝ := -|x|

theorem increasing_function_on_interval (h0 : ∀ x : ℝ, x > 0):
  (∀ x y : ℝ, 0 < x -> x < y -> f_C x < f_C y) ∧ 
  (∀ (g : ℝ → ℝ), (g ≠ f_C) → (∀ x y : ℝ, 0 < x -> x < y -> g x ≥ g y)) :=
by sorry

end NUMINAMATH_GPT_increasing_function_on_interval_l35_3547


namespace NUMINAMATH_GPT_possible_values_of_reciprocal_sum_l35_3598

theorem possible_values_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) (h4 : x * y = 1) : 
  1/x + 1/y = 2 := 
sorry

end NUMINAMATH_GPT_possible_values_of_reciprocal_sum_l35_3598


namespace NUMINAMATH_GPT_alyssa_earnings_l35_3576

theorem alyssa_earnings
    (weekly_allowance: ℤ)
    (spent_on_movies_fraction: ℤ)
    (amount_ended_with: ℤ)
    (h1: weekly_allowance = 8)
    (h2: spent_on_movies_fraction = 1 / 2)
    (h3: amount_ended_with = 12)
    : ∃ money_earned_from_car_wash: ℤ, money_earned_from_car_wash = 8 :=
by
  sorry

end NUMINAMATH_GPT_alyssa_earnings_l35_3576


namespace NUMINAMATH_GPT_solve_for_a_l35_3580

theorem solve_for_a (a : ℝ) (h : |2 * a + 1| = 3 * |a| - 2) : a = -1 ∨ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l35_3580


namespace NUMINAMATH_GPT_saltwater_concentration_l35_3585

theorem saltwater_concentration (salt_mass water_mass : ℝ) (h₁ : salt_mass = 8) (h₂ : water_mass = 32) : 
  salt_mass / (salt_mass + water_mass) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_saltwater_concentration_l35_3585


namespace NUMINAMATH_GPT_sum_of_final_numbers_l35_3557

variable {x y T : ℝ}

theorem sum_of_final_numbers (h : x + y = T) : 3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by 
  -- The place for the proof steps, which will later be filled
  sorry

end NUMINAMATH_GPT_sum_of_final_numbers_l35_3557


namespace NUMINAMATH_GPT_polygon_sides_from_angle_sum_l35_3514

-- Let's define the problem
theorem polygon_sides_from_angle_sum : 
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_polygon_sides_from_angle_sum_l35_3514


namespace NUMINAMATH_GPT_modular_inverse_sum_eq_14_l35_3510

theorem modular_inverse_sum_eq_14 : 
(9 + 13 + 15 + 16 + 12 + 3 + 14) % 17 = 14 := by
  sorry

end NUMINAMATH_GPT_modular_inverse_sum_eq_14_l35_3510


namespace NUMINAMATH_GPT_atomic_weight_of_oxygen_l35_3595

theorem atomic_weight_of_oxygen (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (molecular_weight_Al2O3 : ℝ) (n_Al : ℕ) (n_O : ℕ) :
  atomic_weight_Al = 26.98 →
  molecular_weight_Al2O3 = 102 →
  n_Al = 2 →
  n_O = 3 →
  (molecular_weight_Al2O3 - n_Al * atomic_weight_Al) / n_O = 16.01 :=
by
  sorry

end NUMINAMATH_GPT_atomic_weight_of_oxygen_l35_3595


namespace NUMINAMATH_GPT_distinct_solution_condition_l35_3502

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end NUMINAMATH_GPT_distinct_solution_condition_l35_3502


namespace NUMINAMATH_GPT_product_mod_10_l35_3501

theorem product_mod_10 (a b c : ℕ) (ha : a % 10 = 4) (hb : b % 10 = 5) (hc : c % 10 = 5) :
  (a * b * c) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_product_mod_10_l35_3501


namespace NUMINAMATH_GPT_supermarkets_in_us_l35_3522

noncomputable def number_of_supermarkets_in_canada : ℕ := 35
noncomputable def number_of_supermarkets_total : ℕ := 84
noncomputable def diff_us_canada : ℕ := 14
noncomputable def number_of_supermarkets_in_us : ℕ := number_of_supermarkets_in_canada + diff_us_canada

theorem supermarkets_in_us : number_of_supermarkets_in_us = 49 := by
  sorry

end NUMINAMATH_GPT_supermarkets_in_us_l35_3522


namespace NUMINAMATH_GPT_pieces_of_candy_l35_3533

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end NUMINAMATH_GPT_pieces_of_candy_l35_3533


namespace NUMINAMATH_GPT_hyeongjun_older_sister_age_l35_3555

-- Define the ages of Hyeongjun and his older sister
variables (H S : ℕ)

-- Conditions
def age_gap := S = H + 2
def sum_of_ages := H + S = 26

-- Theorem stating that the older sister's age is 14
theorem hyeongjun_older_sister_age (H S : ℕ) (h1 : age_gap H S) (h2 : sum_of_ages H S) : S = 14 := 
by 
  sorry

end NUMINAMATH_GPT_hyeongjun_older_sister_age_l35_3555


namespace NUMINAMATH_GPT_Alice_and_Dave_weight_l35_3526

variable (a b c d : ℕ)

-- Conditions
variable (h1 : a + b = 230)
variable (h2 : b + c = 220)
variable (h3 : c + d = 250)

-- Proof statement
theorem Alice_and_Dave_weight :
  a + d = 260 :=
sorry

end NUMINAMATH_GPT_Alice_and_Dave_weight_l35_3526


namespace NUMINAMATH_GPT_part1_part2_l35_3509

-- Part (1)
theorem part1 (x : ℝ) (m : ℝ) (h : x = 2) : 
  (x / (x - 3) + m / (3 - x) = 3) → m = 5 :=
sorry

-- Part (2)
theorem part2 (x : ℝ) (m : ℝ) :
  (x / (x - 3) + m / (3 - x) = 3) → (x > 0) → (m < 9) ∧ (m ≠ 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l35_3509


namespace NUMINAMATH_GPT_maria_spent_60_dollars_l35_3587

theorem maria_spent_60_dollars :
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  true
    → total_cost = 60 := 
by 
  intros
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  sorry

end NUMINAMATH_GPT_maria_spent_60_dollars_l35_3587


namespace NUMINAMATH_GPT_negation_of_exists_equiv_forall_neg_l35_3579

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_exists_equiv_forall_neg_l35_3579


namespace NUMINAMATH_GPT_cos_double_angle_identity_l35_3519

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_identity_l35_3519


namespace NUMINAMATH_GPT_condition_C_for_D_condition_A_for_B_l35_3584

theorem condition_C_for_D (C D : Prop) (h : C → D) : C → D :=
by
  exact h

theorem condition_A_for_B (A B D : Prop) (hA_to_D : A → D) (hD_to_B : D → B) : A → B :=
by
  intro hA
  apply hD_to_B
  apply hA_to_D
  exact hA

end NUMINAMATH_GPT_condition_C_for_D_condition_A_for_B_l35_3584


namespace NUMINAMATH_GPT_sum_of_squares_l35_3553

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 5) (h2 : ab + bc + ac = 5) : a^2 + b^2 + c^2 = 15 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_l35_3553


namespace NUMINAMATH_GPT_paintings_left_correct_l35_3546

def initial_paintings := 98
def paintings_gotten_rid_of := 3

theorem paintings_left_correct :
  initial_paintings - paintings_gotten_rid_of = 95 :=
by
  sorry

end NUMINAMATH_GPT_paintings_left_correct_l35_3546


namespace NUMINAMATH_GPT_hypotenuse_length_l35_3545

theorem hypotenuse_length (a b c : ℝ) 
  (h_right_angled : c^2 = a^2 + b^2) 
  (h_sum_squares : a^2 + b^2 + c^2 = 980) : 
  c = 70 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l35_3545


namespace NUMINAMATH_GPT_hannah_remaining_money_l35_3505

-- Define the conditions of the problem
def initial_amount : Nat := 120
def rides_cost : Nat := initial_amount * 40 / 100
def games_cost : Nat := initial_amount * 15 / 100
def remaining_after_rides_games : Nat := initial_amount - rides_cost - games_cost

def dessert_cost : Nat := 8
def cotton_candy_cost : Nat := 5
def hotdog_cost : Nat := 6
def keychain_cost : Nat := 7
def poster_cost : Nat := 10
def additional_attraction_cost : Nat := 15
def total_food_souvenirs_cost : Nat := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + additional_attraction_cost

def final_remaining_amount : Nat := remaining_after_rides_games - total_food_souvenirs_cost

-- Formulate the theorem to prove
theorem hannah_remaining_money : final_remaining_amount = 3 := by
  sorry

end NUMINAMATH_GPT_hannah_remaining_money_l35_3505


namespace NUMINAMATH_GPT_problem_intersection_l35_3586

noncomputable def A (x : ℝ) : Prop := 1 < x ∧ x < 4
noncomputable def B (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem problem_intersection : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_problem_intersection_l35_3586


namespace NUMINAMATH_GPT_length_of_RT_in_trapezoid_l35_3517

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end NUMINAMATH_GPT_length_of_RT_in_trapezoid_l35_3517


namespace NUMINAMATH_GPT_shaded_hexagons_are_balanced_l35_3552

-- Definitions and conditions from the problem
def is_balanced (a b c : ℕ) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

def hexagon_grid_balanced (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ),
  (i % 2 = 0 ∧ grid (i, j) = grid (i, j + 1) ∧ grid (i, j + 1) = grid (i + 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i, j + 1) ∧ grid (i, j + 1) ≠ grid (i + 1, j + 1) ∧ grid (i, j) ≠ grid (i + 1, j + 1))
  ∨ (i % 2 ≠ 0 ∧ grid (i, j) = grid (i - 1, j) ∧ grid (i - 1, j) = grid (i - 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i - 1, j) ∧ grid (i - 1, j) ≠ grid (i - 1, j + 1) ∧ grid (i, j) ≠ grid (i - 1, j + 1))

theorem shaded_hexagons_are_balanced (grid : ℕ × ℕ → ℕ) (h_balanced : hexagon_grid_balanced grid) :
  is_balanced (grid (1, 1)) (grid (1, 10)) (grid (10, 10)) :=
sorry

end NUMINAMATH_GPT_shaded_hexagons_are_balanced_l35_3552


namespace NUMINAMATH_GPT_random_event_proof_l35_3508

def statement_A := "Strong youth leads to a strong country"
def statement_B := "Scooping the moon in the water"
def statement_C := "Waiting by the stump for a hare"
def statement_D := "Green waters and lush mountains are mountains of gold and silver"

def is_random_event (statement : String) : Prop :=
statement = statement_C

theorem random_event_proof : is_random_event statement_C :=
by
  -- Based on the analysis in the problem, Statement C is determined to be random.
  sorry

end NUMINAMATH_GPT_random_event_proof_l35_3508


namespace NUMINAMATH_GPT_discriminant_of_trinomial_l35_3561

theorem discriminant_of_trinomial (x1 x2 : ℝ) (h : x2 - x1 = 2) : (x2 - x1)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_trinomial_l35_3561


namespace NUMINAMATH_GPT_rise_in_water_level_l35_3549

theorem rise_in_water_level (edge base_length base_width : ℝ) (cube_volume base_area rise : ℝ) 
  (h₁ : edge = 5) (h₂ : base_length = 10) (h₃ : base_width = 5)
  (h₄ : cube_volume = edge^3) (h₅ : base_area = base_length * base_width) 
  (h₆ : rise = cube_volume / base_area) : 
  rise = 2.5 := 
by 
  -- add proof here 
  sorry

end NUMINAMATH_GPT_rise_in_water_level_l35_3549


namespace NUMINAMATH_GPT_arcsin_arccos_add_eq_pi6_l35_3532

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def arccos (x : Real) : Real := sorry

theorem arcsin_arccos_add_eq_pi6 (x : Real) (hx_range : -1 ≤ x ∧ x ≤ 1)
    (h3x_range : -1 ≤ 3 * x ∧ 3 * x ≤ 1) 
    (h : arcsin x + arccos (3 * x) = Real.pi / 6) :
    x = Real.sqrt (3 / 124) := 
  sorry

end NUMINAMATH_GPT_arcsin_arccos_add_eq_pi6_l35_3532


namespace NUMINAMATH_GPT_domain_of_sqrt_ln_eq_l35_3525

noncomputable def domain_of_function : Set ℝ :=
  {x | 2 * x + 1 >= 0 ∧ 3 - 4 * x > 0}

theorem domain_of_sqrt_ln_eq :
  domain_of_function = Set.Icc (-1 / 2) (3 / 4) \ {3 / 4} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_ln_eq_l35_3525


namespace NUMINAMATH_GPT_neg_prop_l35_3567

theorem neg_prop : ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 4) ↔ ∃ x : ℝ, x^2 - 2 * x + 4 > 4 := 
by 
  sorry

end NUMINAMATH_GPT_neg_prop_l35_3567


namespace NUMINAMATH_GPT_max_area_rectangle_l35_3588

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l35_3588


namespace NUMINAMATH_GPT_tangent_same_at_origin_l35_3550

noncomputable def f (x : ℝ) := Real.exp (3 * x) - 1
noncomputable def g (x : ℝ) := 3 * Real.exp x - 3

theorem tangent_same_at_origin :
  (deriv f 0 = deriv g 0) ∧ (f 0 = g 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_same_at_origin_l35_3550


namespace NUMINAMATH_GPT_find_value_l35_3562

variable {a b c : ℝ}

def ellipse_eqn (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem find_value 
  (h1 : a^2 + b^2 - 3*c^2 = 0)
  (h2 : a^2 = b^2 + c^2) :
  (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_find_value_l35_3562
