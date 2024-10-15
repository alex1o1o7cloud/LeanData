import Mathlib

namespace NUMINAMATH_GPT_camel_height_in_feet_correct_l2223_222364

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end NUMINAMATH_GPT_camel_height_in_feet_correct_l2223_222364


namespace NUMINAMATH_GPT_problem_l2223_222351

theorem problem (a b c : ℤ) (h1 : 0 < c) (h2 : c < 90) (h3 : Real.sqrt (9 - 8 * Real.sin (50 * Real.pi / 180)) = a + b * Real.sin (c * Real.pi / 180)) : 
  (a + b) / c = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2223_222351


namespace NUMINAMATH_GPT_searchlight_reflector_distance_l2223_222380

noncomputable def parabola_vertex_distance : Rat :=
  let diameter := 60 -- in cm
  let depth := 40 -- in cm
  let x := 40 -- x-coordinate of the point
  let y := 30 -- y-coordinate of the point
  let p := (y^2) / (2 * x)
  p / 2

theorem searchlight_reflector_distance : parabola_vertex_distance = 45 / 8 := by
  sorry

end NUMINAMATH_GPT_searchlight_reflector_distance_l2223_222380


namespace NUMINAMATH_GPT_Harriet_siblings_product_l2223_222352

variable (Harry_sisters : Nat)
variable (Harry_brothers : Nat)
variable (Harriet_sisters : Nat)
variable (Harriet_brothers : Nat)

theorem Harriet_siblings_product:
  Harry_sisters = 4 -> 
  Harry_brothers = 6 ->
  Harriet_sisters = Harry_sisters -> 
  Harriet_brothers = Harry_brothers ->
  Harriet_sisters * Harriet_brothers = 24 :=
by
  intro hs hb hhs hhb
  rw [hhs, hhb]
  sorry

end NUMINAMATH_GPT_Harriet_siblings_product_l2223_222352


namespace NUMINAMATH_GPT_candidate_percentage_l2223_222300

variables (P candidate_votes rival_votes total_votes : ℝ)

-- Conditions
def candidate_lost_by_2460 (candidate_votes rival_votes : ℝ) : Prop :=
  rival_votes = candidate_votes + 2460

def total_votes_cast (candidate_votes rival_votes total_votes : ℝ) : Prop :=
  candidate_votes + rival_votes = total_votes

-- Proof problem
theorem candidate_percentage (h1 : candidate_lost_by_2460 candidate_votes rival_votes)
                             (h2 : total_votes_cast candidate_votes rival_votes 8200) :
  P = 35 :=
sorry

end NUMINAMATH_GPT_candidate_percentage_l2223_222300


namespace NUMINAMATH_GPT_water_cost_function_solve_for_x_and_payments_l2223_222375

def water_usage_A (x : ℕ) : ℕ := 5 * x
def water_usage_B (x : ℕ) : ℕ := 3 * x

def water_payment_A (x : ℕ) : ℕ :=
  if water_usage_A x <= 15 then 
    water_usage_A x * 2 
  else 
    15 * 2 + (water_usage_A x - 15) * 3

def water_payment_B (x : ℕ) : ℕ :=
  if water_usage_B x <= 15 then 
    water_usage_B x * 2 
  else 
    15 * 2 + (water_usage_B x - 15) * 3

def total_payment (x : ℕ) : ℕ := water_payment_A x + water_payment_B x

theorem water_cost_function (x : ℕ) : total_payment x =
  if 0 < x ∧ x ≤ 3 then 16 * x
  else if 3 < x ∧ x ≤ 5 then 21 * x - 15
  else if 5 < x then 24 * x - 30
  else 0 := sorry

theorem solve_for_x_and_payments (y : ℕ) : y = 114 → ∃ x, total_payment x = y ∧
  water_usage_A x = 30 ∧ water_payment_A x = 75 ∧
  water_usage_B x = 18 ∧ water_payment_B x = 39 := sorry

end NUMINAMATH_GPT_water_cost_function_solve_for_x_and_payments_l2223_222375


namespace NUMINAMATH_GPT_area_of_regular_octagon_l2223_222395

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_area_of_regular_octagon_l2223_222395


namespace NUMINAMATH_GPT_find_sum_l2223_222327

theorem find_sum 
  (R : ℝ) -- Original interest rate
  (P : ℝ) -- Principal amount
  (h: (P * (R + 3) * 3 / 100) = ((P * R * 3 / 100) + 81)): 
  P = 900 :=
sorry

end NUMINAMATH_GPT_find_sum_l2223_222327


namespace NUMINAMATH_GPT_small_bottle_sold_percentage_l2223_222339

-- Definitions for initial conditions
def small_bottles_initial : ℕ := 6000
def large_bottles_initial : ℕ := 15000
def large_bottle_sold_percentage : ℝ := 0.14
def total_remaining_bottles : ℕ := 18180

-- The statement we need to prove
theorem small_bottle_sold_percentage :
  ∃ k : ℝ, (0 ≤ k ∧ k ≤ 100) ∧
  (small_bottles_initial - (k / 100) * small_bottles_initial + 
   large_bottles_initial - large_bottle_sold_percentage * large_bottles_initial = total_remaining_bottles) ∧
  (k = 12) :=
sorry

end NUMINAMATH_GPT_small_bottle_sold_percentage_l2223_222339


namespace NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l2223_222347

theorem range_of_a_for_empty_solution_set :
  {a : ℝ | ∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0} = 
  {a : ℝ | -3 ≤ a ∧ a < 9 / 5} :=
sorry

end NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l2223_222347


namespace NUMINAMATH_GPT_intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l2223_222386

noncomputable def f (x : ℝ) := x * Real.log (-x)
noncomputable def g (x a : ℝ) := x * f (a * x) - Real.exp (x - 2)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < -1 / Real.exp 1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 / Real.exp 1 < x ∧ x < 0 → deriv f x < 0) ∧
  f (-1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

theorem number_of_zeros_of_g (a : ℝ) :
  (a > 0 ∨ a = -1 / Real.exp 1 → ∃! x : ℝ, g x a = 0) ∧
  (a < 0 ∧ a ≠ -1 / Real.exp 1 → ∀ x : ℝ, g x a ≠ 0) :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l2223_222386


namespace NUMINAMATH_GPT_mean_of_reciprocals_of_first_four_primes_l2223_222361

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_reciprocals_of_first_four_primes_l2223_222361


namespace NUMINAMATH_GPT_find_product_of_abc_l2223_222311

theorem find_product_of_abc :
  ∃ (a b c m : ℝ), 
    a + b + c = 195 ∧
    m = 8 * a ∧
    m = b - 10 ∧
    m = c + 10 ∧
    a * b * c = 95922 := by
  sorry

end NUMINAMATH_GPT_find_product_of_abc_l2223_222311


namespace NUMINAMATH_GPT_range_of_a_l2223_222342

noncomputable def interval1 (a : ℝ) : Prop := -2 < a ∧ a <= 1 / 2
noncomputable def interval2 (a : ℝ) : Prop := a >= 2

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * a| > 1

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, p a ∨ q a) (h2 : ¬ (∀ x : ℝ, p a ∧ q a)) : 
  interval1 a ∨ interval2 a :=
sorry

end NUMINAMATH_GPT_range_of_a_l2223_222342


namespace NUMINAMATH_GPT_B_finish_work_in_10_days_l2223_222313

variable (W : ℝ) -- amount of work
variable (x : ℝ) -- number of days B can finish the work alone

theorem B_finish_work_in_10_days (h1 : ∀ A_rate, A_rate = W / 4)
                                (h2 : ∀ B_rate, B_rate = W / x)
                                (h3 : ∀ Work_done_together Remaining_work,
                                      Work_done_together = 2 * (W / 4 + W / x) ∧
                                      Remaining_work = W - Work_done_together ∧
                                      Remaining_work = (W / x) * 3.0000000000000004) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_B_finish_work_in_10_days_l2223_222313


namespace NUMINAMATH_GPT_diophantine_solution_unique_l2223_222392

theorem diophantine_solution_unique (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 = k * x * y - 1 ↔ k = 3 :=
by sorry

end NUMINAMATH_GPT_diophantine_solution_unique_l2223_222392


namespace NUMINAMATH_GPT_Q_subset_P_l2223_222385

-- Definitions
def P : Set ℝ := {x : ℝ | x ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Statement to prove
theorem Q_subset_P : Q ⊆ P :=
sorry

end NUMINAMATH_GPT_Q_subset_P_l2223_222385


namespace NUMINAMATH_GPT_sum_of_coefficients_l2223_222366

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 7)
  (h4 : b * d = -6) :
  a + b + c + d = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2223_222366


namespace NUMINAMATH_GPT_hyperbola_asymptotes_angle_l2223_222317

noncomputable def angle_between_asymptotes 
  (a b : ℝ) (e : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) : ℝ :=
  2 * Real.arctan (b / a)

theorem hyperbola_asymptotes_angle (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) 
  (b_eq : b = Real.sqrt (e^2 * a^2 - a^2)) : 
  angle_between_asymptotes a b e h1 h2 h3 = π / 3 := 
by
  -- proof omitted
  sorry
  
end NUMINAMATH_GPT_hyperbola_asymptotes_angle_l2223_222317


namespace NUMINAMATH_GPT_compute_expression_l2223_222331

theorem compute_expression :
  25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := 
sorry

end NUMINAMATH_GPT_compute_expression_l2223_222331


namespace NUMINAMATH_GPT_find_quadratic_function_l2223_222321

theorem find_quadratic_function (a h k x y : ℝ) (vertex_y : ℝ) (intersect_y : ℝ)
    (hv : h = 1 ∧ k = 2)
    (hi : x = 0 ∧ y = 3) :
    (∀ x, y = a * (x - h) ^ 2 + k) → vertex_y = h ∧ intersect_y = k →
    y = x^2 - 2 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l2223_222321


namespace NUMINAMATH_GPT_polynomial_roots_sum_l2223_222365

noncomputable def roots (p : Polynomial ℚ) : Set ℚ := {r | p.eval r = 0}

theorem polynomial_roots_sum :
  ∀ a b c : ℚ, (a ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (b ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (c ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c = 8) →
  (a * b + a * c + b * c = 7) →
  (a * b * c = -3) →
  (a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) = 17 / 2) := by
    intros a b c ha hb hc hab habc hac sum_nums sum_prods prod_roots
    sorry

#check polynomial_roots_sum

end NUMINAMATH_GPT_polynomial_roots_sum_l2223_222365


namespace NUMINAMATH_GPT_pizza_problem_l2223_222307

theorem pizza_problem
  (pizza_slices : ℕ)
  (total_pizzas : ℕ)
  (total_people : ℕ)
  (pepperoni_only_friend : ℕ)
  (remaining_pepperoni : ℕ)
  (equal_distribution : Prop)
  (h_cond1 : pizza_slices = 16)
  (h_cond2 : total_pizzas = 2)
  (h_cond3 : total_people = 4)
  (h_cond4 : pepperoni_only_friend = 1)
  (h_cond5 : remaining_pepperoni = 1)
  (h_cond6 : equal_distribution ∧ (pepperoni_only_friend ≤ total_people)) :
  ∃ cheese_slices_left : ℕ, cheese_slices_left = 7 := by
  sorry

end NUMINAMATH_GPT_pizza_problem_l2223_222307


namespace NUMINAMATH_GPT_find_initial_length_of_cloth_l2223_222389

noncomputable def initial_length_of_cloth : ℝ :=
  let work_rate_of_8_men := 36 / 0.75
  work_rate_of_8_men

theorem find_initial_length_of_cloth (L : ℝ) (h1 : (4:ℝ) * 2 = L / ((4:ℝ) / (L / 8)))
    (h2 : (8:ℝ) / L = 36 / 0.75) : L = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_length_of_cloth_l2223_222389


namespace NUMINAMATH_GPT_intercept_sum_l2223_222348

theorem intercept_sum (x y : ℝ) (h : y - 3 = -3 * (x + 2)) :
  (∃ (x_int : ℝ), y = 0 ∧ x_int = -1) ∧ (∃ (y_int : ℝ), x = 0 ∧ y_int = -3) →
  (-1 + (-3) = -4) := by
  sorry

end NUMINAMATH_GPT_intercept_sum_l2223_222348


namespace NUMINAMATH_GPT_sum_of_cubes_equality_l2223_222396

theorem sum_of_cubes_equality (a b p n : ℕ) (hp : Nat.Prime p) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^3 + b^3 = p^n) ↔ 
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end NUMINAMATH_GPT_sum_of_cubes_equality_l2223_222396


namespace NUMINAMATH_GPT_larger_of_two_numbers_l2223_222325

theorem larger_of_two_numbers (A B : ℕ) (HCF : ℕ) (factor1 factor2 : ℕ) (h_hcf : HCF = 23) (h_factor1 : factor1 = 13) (h_factor2 : factor2 = 14)
(hA : A = HCF * factor1) (hB : B = HCF * factor2) :
  max A B = 322 :=
by
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l2223_222325


namespace NUMINAMATH_GPT_spanish_peanuts_l2223_222303

variable (x : ℝ)

theorem spanish_peanuts :
  (10 * 3.50 + x * 3.00 = (10 + x) * 3.40) → x = 2.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_spanish_peanuts_l2223_222303


namespace NUMINAMATH_GPT_courtyard_length_l2223_222315

theorem courtyard_length 
  (stone_area : ℕ) 
  (stones_total : ℕ) 
  (width : ℕ)
  (total_area : ℕ) 
  (L : ℕ) 
  (h1 : stone_area = 4)
  (h2 : stones_total = 135)
  (h3 : width = 18)
  (h4 : total_area = stones_total * stone_area)
  (h5 : total_area = L * width) :
  L = 30 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_courtyard_length_l2223_222315


namespace NUMINAMATH_GPT_find_Luisa_books_l2223_222332

structure Books where
  Maddie : ℕ
  Amy : ℕ
  Amy_and_Luisa : ℕ
  Luisa : ℕ

theorem find_Luisa_books (L M A : ℕ) (hM : M = 15) (hA : A = 6) (hAL : L + A = M + 9) : L = 18 := by
  sorry

end NUMINAMATH_GPT_find_Luisa_books_l2223_222332


namespace NUMINAMATH_GPT_range_of_a_l2223_222345

-- Define an odd function f on ℝ such that f(x) = x^2 for x >= 0
noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -(x^2)

-- Prove the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc a (a + 2) → f (x - a) ≥ f (3 * x + 1)) →
  a ≤ -5 := sorry

end NUMINAMATH_GPT_range_of_a_l2223_222345


namespace NUMINAMATH_GPT_value_of_x_plus_y_div_y_l2223_222367

variable (w x y : ℝ)
variable (hx : w / x = 1 / 6)
variable (hy : w / y = 1 / 5)

theorem value_of_x_plus_y_div_y : (x + y) / y = 11 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_div_y_l2223_222367


namespace NUMINAMATH_GPT_rectangular_field_area_l2223_222350

theorem rectangular_field_area (a c : ℝ) (h_a : a = 13) (h_c : c = 17) :
  ∃ b : ℝ, (b = 2 * Real.sqrt 30) ∧ (a * b = 26 * Real.sqrt 30) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l2223_222350


namespace NUMINAMATH_GPT_air_quality_conditional_prob_l2223_222319

theorem air_quality_conditional_prob :
  let p1 := 0.8
  let p2 := 0.68
  let p := p2 / p1
  p = 0.85 :=
by
  sorry

end NUMINAMATH_GPT_air_quality_conditional_prob_l2223_222319


namespace NUMINAMATH_GPT_intersection_point_of_lines_l2223_222335

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l2223_222335


namespace NUMINAMATH_GPT_number_of_boys_l2223_222391

theorem number_of_boys (x g : ℕ) (h1 : x + g = 100) (h2 : g = x) : x = 50 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l2223_222391


namespace NUMINAMATH_GPT_light_flash_fraction_l2223_222353

def light_flash_fraction_of_hour (n : ℕ) (t : ℕ) (flashes : ℕ) := 
  (n * t) / (60 * 60)

theorem light_flash_fraction (n : ℕ) (t : ℕ) (flashes : ℕ) (h1 : t = 12) (h2 : flashes = 300) : 
  light_flash_fraction_of_hour n t flashes = 1 := 
by
  sorry

end NUMINAMATH_GPT_light_flash_fraction_l2223_222353


namespace NUMINAMATH_GPT_midpoint_of_hyperbola_l2223_222393

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_of_hyperbola_l2223_222393


namespace NUMINAMATH_GPT_parabola_axis_of_symmetry_l2223_222355

theorem parabola_axis_of_symmetry : 
  ∀ (x : ℝ), x = -1 → (∃ y : ℝ, y = -x^2 - 2*x - 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_axis_of_symmetry_l2223_222355


namespace NUMINAMATH_GPT_men_absent_l2223_222383

theorem men_absent (x : ℕ) :
  let original_men := 42
  let original_days := 17
  let remaining_days := 21 
  let total_work := original_men * original_days
  let remaining_men_work := (original_men - x) * remaining_days 
  total_work = remaining_men_work →
  x = 8 :=
by
  intros
  let total_work := 42 * 17
  let remaining_men_work := (42 - x) * 21
  have h : total_work = remaining_men_work := ‹total_work = remaining_men_work›
  sorry

end NUMINAMATH_GPT_men_absent_l2223_222383


namespace NUMINAMATH_GPT_max_wx_plus_xy_plus_yz_plus_wz_l2223_222306

theorem max_wx_plus_xy_plus_yz_plus_wz (w x y z : ℝ) (h_nonneg : 0 ≤ w ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : w + x + y + z = 200) :
  wx + xy + yz + wz ≤ 10000 :=
sorry

end NUMINAMATH_GPT_max_wx_plus_xy_plus_yz_plus_wz_l2223_222306


namespace NUMINAMATH_GPT_equal_focal_distances_l2223_222370

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end NUMINAMATH_GPT_equal_focal_distances_l2223_222370


namespace NUMINAMATH_GPT_problem_solution_l2223_222362

theorem problem_solution (x : ℝ) (h : x ≠ 5) : (x ≥ 8) ↔ ((x + 1) / (x - 5) ≥ 3) :=
sorry

end NUMINAMATH_GPT_problem_solution_l2223_222362


namespace NUMINAMATH_GPT_tetrahedron_probability_correct_l2223_222388

noncomputable def tetrahedron_probability : ℚ :=
  let total_arrangements := 16
  let suitable_arrangements := 2
  suitable_arrangements / total_arrangements

theorem tetrahedron_probability_correct : tetrahedron_probability = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_probability_correct_l2223_222388


namespace NUMINAMATH_GPT_geometry_problem_l2223_222320

theorem geometry_problem
  (A B C D E : Type*)
  (BAC ABC ACB ADE ADC AEB DEB CDE : ℝ)
  (h₁ : ABC = 72)
  (h₂ : ACB = 90)
  (h₃ : CDE = 36)
  (h₄ : ADC = 180)
  (h₅ : AEB = 180) :
  DEB = 162 :=
sorry

end NUMINAMATH_GPT_geometry_problem_l2223_222320


namespace NUMINAMATH_GPT_parabola_and_hyperbola_tangent_l2223_222322

theorem parabola_and_hyperbola_tangent (m : ℝ) :
  (∀ (x y : ℝ), (y = x^2 + 6) → (y^2 - m * x^2 = 6) → (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6)) :=
sorry

end NUMINAMATH_GPT_parabola_and_hyperbola_tangent_l2223_222322


namespace NUMINAMATH_GPT_fraction_of_grid_covered_l2223_222363

open Real EuclideanGeometry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem fraction_of_grid_covered :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  let grid_area := 7 * 7
  let triangle_area := area_of_triangle A B C
  triangle_area / grid_area = 6 / 49 := by
  sorry

end NUMINAMATH_GPT_fraction_of_grid_covered_l2223_222363


namespace NUMINAMATH_GPT_value_of_expression_l2223_222324

theorem value_of_expression (x y z : ℤ) (h1 : x = -3) (h2 : y = 5) (h3 : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l2223_222324


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l2223_222312

theorem cost_price_of_computer_table (C SP : ℝ) (h1 : SP = 1.25 * C) (h2 : SP = 8340) :
  C = 6672 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l2223_222312


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l2223_222382

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_correct_l2223_222382


namespace NUMINAMATH_GPT_max_product_l2223_222310

theorem max_product (a b : ℕ) (h1: a + b = 100) 
    (h2: a % 3 = 2) (h3: b % 7 = 5) : a * b ≤ 2491 := by
  sorry

end NUMINAMATH_GPT_max_product_l2223_222310


namespace NUMINAMATH_GPT_min_ab_value_l2223_222337

variable (a b : ℝ)

theorem min_ab_value (h1 : a > -1) (h2 : b > -2) (h3 : (a+1) * (b+2) = 16) : a + b ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_ab_value_l2223_222337


namespace NUMINAMATH_GPT_determine_c_l2223_222359

noncomputable def c_floor : ℤ := -3
noncomputable def c_frac : ℝ := (25 - Real.sqrt 481) / 8

theorem determine_c : c_floor + c_frac = -2.72 := by
  have h1 : 3 * (c_floor : ℝ)^2 + 19 * (c_floor : ℝ) - 63 = 0 := by
    sorry
  have h2 : 4 * c_frac^2 - 25 * c_frac + 9 = 0 := by
    sorry
  sorry

end NUMINAMATH_GPT_determine_c_l2223_222359


namespace NUMINAMATH_GPT_petStoreHasSixParrots_l2223_222358

def petStoreParrotsProof : Prop :=
  let cages := 6.0
  let parakeets := 2.0
  let birds_per_cage := 1.333333333
  let total_birds := cages * birds_per_cage
  let number_of_parrots := total_birds - parakeets
  number_of_parrots = 6.0

theorem petStoreHasSixParrots : petStoreParrotsProof := by
  sorry

end NUMINAMATH_GPT_petStoreHasSixParrots_l2223_222358


namespace NUMINAMATH_GPT_length_AC_l2223_222360

variable {A B C : Type} [Field A] [Field B] [Field C]

-- Definitions for the problem conditions
noncomputable def length_AB : ℝ := 3
noncomputable def angle_A : ℝ := Real.pi * 120 / 180
noncomputable def area_ABC : ℝ := (15 * Real.sqrt 3) / 4

-- The theorem statement
theorem length_AC (b : ℝ) (h1 : b = length_AB) (h2 : angle_A = Real.pi * 120 / 180) (h3 : area_ABC = (15 * Real.sqrt 3) / 4) : b = 5 :=
sorry

end NUMINAMATH_GPT_length_AC_l2223_222360


namespace NUMINAMATH_GPT_probability_divisible_by_256_l2223_222378

theorem probability_divisible_by_256 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) :
  ((n * (n + 1) * (n + 2)) % 256 = 0) → (∃ p : ℚ, p = 0.006 ∧ (∃ k : ℕ, k ≤ 1000 ∧ (n = k))) :=
sorry

end NUMINAMATH_GPT_probability_divisible_by_256_l2223_222378


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l2223_222344

noncomputable def toll (x : ℕ) : ℝ :=
  2.50 + 0.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let num_wheels := 18
  let wheels_on_front_axle := 2
  let wheels_per_other_axle := 4
  let num_other_axles := (num_wheels - wheels_on_front_axle) / wheels_per_other_axle
  let total_num_axles := num_other_axles + 1
  toll total_num_axles = 4.00 :=
by
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l2223_222344


namespace NUMINAMATH_GPT_mr_jones_loss_l2223_222384

theorem mr_jones_loss :
  ∃ (C_1 C_2 : ℝ), 
    (1.2 = 1.2 * C_1 / 1.2) ∧ 
    (1.2 = 0.8 * C_2) ∧ 
    ((C_1 + C_2) - (2 * 1.2)) = -0.1 :=
by
  sorry

end NUMINAMATH_GPT_mr_jones_loss_l2223_222384


namespace NUMINAMATH_GPT_intersection_complement_l2223_222328

open Set

-- Define sets A and B as provided in the conditions
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the theorem to prove the question is equal to the answer given the conditions
theorem intersection_complement : (A ∩ compl B) = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2223_222328


namespace NUMINAMATH_GPT_convex_polygon_angles_eq_nine_l2223_222336

theorem convex_polygon_angles_eq_nine (n : ℕ) (a : ℕ → ℝ) (d : ℝ)
  (h1 : a (n - 1) = 180)
  (h2 : ∀ k, a (k + 1) - a k = d)
  (h3 : d = 10) :
  n = 9 :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_angles_eq_nine_l2223_222336


namespace NUMINAMATH_GPT_largest_number_is_l2223_222340

-- Define the conditions stated in the problem
def sum_of_three_numbers_is_100 (a b c : ℝ) : Prop :=
  a + b + c = 100

def two_larger_numbers_differ_by_8 (b c : ℝ) : Prop :=
  c - b = 8

def two_smaller_numbers_differ_by_5 (a b : ℝ) : Prop :=
  b - a = 5

-- Define the hypothesis
def problem_conditions (a b c : ℝ) : Prop :=
  sum_of_three_numbers_is_100 a b c ∧
  two_larger_numbers_differ_by_8 b c ∧
  two_smaller_numbers_differ_by_5 a b

-- Define the proof problem
theorem largest_number_is (a b c : ℝ) (h : problem_conditions a b c) : 
  c = 121 / 3 :=
sorry

end NUMINAMATH_GPT_largest_number_is_l2223_222340


namespace NUMINAMATH_GPT_problem_product_xyzw_l2223_222304

theorem problem_product_xyzw
    (x y z w : ℝ)
    (h1 : x + 1 / y = 1)
    (h2 : y + 1 / z + w = 1)
    (h3 : w = 2) :
    xyzw = -2 * y^2 + 2 * y :=
by
    sorry

end NUMINAMATH_GPT_problem_product_xyzw_l2223_222304


namespace NUMINAMATH_GPT_max_value_of_expression_l2223_222346

theorem max_value_of_expression (a b c : ℝ) (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) 
    (h_sum: a + b + c = 3) :
    (ab / (a + b) + ac / (a + c) + bc / (b + c) ≤ 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l2223_222346


namespace NUMINAMATH_GPT_roots_polynomial_sum_l2223_222369

theorem roots_polynomial_sum (p q r s : ℂ)
  (h_roots : (p, q, r, s) ∈ { (p, q, r, s) | (Polynomial.eval p (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval q (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval r (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval s (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) })
  (h_sum_two_at_a_time : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (h_product : p*q*r*s = 6) :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_roots_polynomial_sum_l2223_222369


namespace NUMINAMATH_GPT_solution1_solution2_solution3_l2223_222399

noncomputable def problem1 : Real :=
3.5 * 101

noncomputable def problem2 : Real :=
11 * 5.9 - 5.9

noncomputable def problem3 : Real :=
88 - 17.5 - 12.5

theorem solution1 : problem1 = 353.5 :=
by
  sorry

theorem solution2 : problem2 = 59 :=
by
  sorry

theorem solution3 : problem3 = 58 :=
by
  sorry

end NUMINAMATH_GPT_solution1_solution2_solution3_l2223_222399


namespace NUMINAMATH_GPT_find_ab_l2223_222308

theorem find_ab (a b : ℕ) (h : (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a + Real.sqrt b) = 12) : a = 2 ∧ b = 30 :=
sorry

end NUMINAMATH_GPT_find_ab_l2223_222308


namespace NUMINAMATH_GPT_hexagon_coloring_l2223_222374

-- Definitions based on conditions
variable (A B C D E F : ℕ)
variable (color : ℕ → ℕ)
variable (v1 v2 : ℕ)

-- The question is about the number of different colorings
theorem hexagon_coloring (h_distinct : ∀ (x y : ℕ), x ≠ y → color x ≠ color y) 
    (h_colors : ∀ (x : ℕ), x ∈ [A, B, C, D, E, F] → 0 < color x ∧ color x < 5) :
    4 * 3 * 3 * 3 * 3 * 3 = 972 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_coloring_l2223_222374


namespace NUMINAMATH_GPT_number_of_pigs_l2223_222343

variable (cows pigs : Nat)

theorem number_of_pigs (h1 : 2 * (7 + pigs) = 32) : pigs = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_pigs_l2223_222343


namespace NUMINAMATH_GPT_paper_clips_collected_l2223_222379

theorem paper_clips_collected (boxes paper_clips_per_box total_paper_clips : ℕ) 
  (h1 : boxes = 9) 
  (h2 : paper_clips_per_box = 9) 
  (h3 : total_paper_clips = boxes * paper_clips_per_box) : 
  total_paper_clips = 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_paper_clips_collected_l2223_222379


namespace NUMINAMATH_GPT_expected_value_of_winnings_l2223_222349

/-- A fair 6-sided die is rolled. If the roll is even, then you win the amount of dollars 
equal to the square of the number you roll. If the roll is odd, you win nothing. 
Prove that the expected value of your winnings is 28/3 dollars. -/
theorem expected_value_of_winnings : 
  (1 / 6) * (2^2 + 4^2 + 6^2) = 28 / 3 := by
sorry

end NUMINAMATH_GPT_expected_value_of_winnings_l2223_222349


namespace NUMINAMATH_GPT_math_problem_l2223_222373

theorem math_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^b + 3 = b^a) (h4 : 3 * a^b = b^a + 13) : 
  (a = 2) ∧ (b = 3) :=
sorry

end NUMINAMATH_GPT_math_problem_l2223_222373


namespace NUMINAMATH_GPT_longest_piece_length_l2223_222357

-- Define the lengths of the ropes
def rope1 : ℕ := 45
def rope2 : ℕ := 75
def rope3 : ℕ := 90

-- Define the greatest common divisor we need to prove
def gcd_of_ropes : ℕ := Nat.gcd rope1 (Nat.gcd rope2 rope3)

-- Goal theorem stating the problem
theorem longest_piece_length : gcd_of_ropes = 15 := by
  sorry

end NUMINAMATH_GPT_longest_piece_length_l2223_222357


namespace NUMINAMATH_GPT_find_p_l2223_222354

theorem find_p (p q : ℚ) (h1 : 5 * p + 3 * q = 10) (h2 : 3 * p + 5 * q = 20) : 
  p = -5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2223_222354


namespace NUMINAMATH_GPT_find_a_l2223_222318

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem find_a : (f_prime a (-1) = 3) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2223_222318


namespace NUMINAMATH_GPT_passes_through_origin_l2223_222397

def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2 * x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

theorem passes_through_origin : 
  (parabola_A 0 ≠ 0) ∧
  (parabola_B 0 ≠ 0) ∧
  (parabola_C 0 = 0) ∧
  (parabola_D 0 ≠ 0) := 
by 
  sorry

end NUMINAMATH_GPT_passes_through_origin_l2223_222397


namespace NUMINAMATH_GPT_find_value_of_k_l2223_222390

theorem find_value_of_k (k : ℤ) : 
  (2 + 3 * k * -1/3 = -7 * 4) → k = 30 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_k_l2223_222390


namespace NUMINAMATH_GPT_non_congruent_triangles_proof_l2223_222387

noncomputable def non_congruent_triangles_count : ℕ :=
  let points := [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  9

theorem non_congruent_triangles_proof :
  non_congruent_triangles_count = 9 :=
sorry

end NUMINAMATH_GPT_non_congruent_triangles_proof_l2223_222387


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2223_222398

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 - 3 * x = 2) : 1 + 2 * x^2 - 6 * x = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x^2 - 3 * x - 4 = 0) : 1 + 3 * x - x^2 = -3 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) (p q : ℝ) (h1 : x = 1 → p * x^3 + q * x + 1 = 5) (h2 : p + q = 4) (hx : x = -1) : p * x^3 + q * x + 1 = -3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2223_222398


namespace NUMINAMATH_GPT_negation_of_p_l2223_222316

variable (p : ∀ x : ℝ, x^2 + x - 6 ≤ 0)

theorem negation_of_p : (∃ x : ℝ, x^2 + x - 6 > 0) :=
sorry

end NUMINAMATH_GPT_negation_of_p_l2223_222316


namespace NUMINAMATH_GPT_gcd_digit_bound_l2223_222338

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end NUMINAMATH_GPT_gcd_digit_bound_l2223_222338


namespace NUMINAMATH_GPT_student_correct_answers_l2223_222330

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 70) : C = 90 :=
sorry

end NUMINAMATH_GPT_student_correct_answers_l2223_222330


namespace NUMINAMATH_GPT_calc1_calc2_l2223_222329

theorem calc1 : (-2) * (-1/8) = 1/4 :=
by
  sorry

theorem calc2 : (-5) / (6/5) = -25/6 :=
by
  sorry

end NUMINAMATH_GPT_calc1_calc2_l2223_222329


namespace NUMINAMATH_GPT_triangle_inequality_l2223_222356

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c R : ℝ) : ℝ := a * b * c / (4 * R)
noncomputable def inradius_area (a b c r : ℝ) : ℝ := semiperimeter a b c * r

theorem triangle_inequality (a b c R r : ℝ) (h₁ : a ≤ 1) (h₂ : b ≤ 1) (h₃ : c ≤ 1)
  (h₄ : area a b c R = semiperimeter a b c * r) : 
  semiperimeter a b c * (1 - 2 * R * r) ≥ 1 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2223_222356


namespace NUMINAMATH_GPT_fraction_cookies_blue_or_green_l2223_222376

theorem fraction_cookies_blue_or_green (C : ℕ) (h1 : 1/C = 1/4) (h2 : 0.5555555555555556 = 5/9) :
  (1/4 + (5/9) * (3/4)) = (2/3) :=
by sorry

end NUMINAMATH_GPT_fraction_cookies_blue_or_green_l2223_222376


namespace NUMINAMATH_GPT_vika_made_84_dollars_l2223_222305

-- Define the amount of money Saheed, Kayla, and Vika made
variable (S K V : ℕ)

-- Given conditions
def condition1 : Prop := S = 4 * K
def condition2 : Prop := K = V - 30
def condition3 : Prop := S = 216

-- Statement to prove
theorem vika_made_84_dollars (S K V : ℕ) (h1 : condition1 S K) (h2 : condition2 K V) (h3 : condition3 S) : 
  V = 84 :=
by sorry

end NUMINAMATH_GPT_vika_made_84_dollars_l2223_222305


namespace NUMINAMATH_GPT_f_of_2014_l2223_222309

theorem f_of_2014 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 4) = -f x + 2 * Real.sqrt 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  : f 2014 = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_f_of_2014_l2223_222309


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000023_l2223_222333

theorem scientific_notation_of_0_0000023 : 
  0.0000023 = 2.3 * 10 ^ (-6) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000023_l2223_222333


namespace NUMINAMATH_GPT_min_value_at_3_l2223_222334

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_min_value_at_3_l2223_222334


namespace NUMINAMATH_GPT_probability_black_white_ball_l2223_222381

theorem probability_black_white_ball :
  let total_balls := 5
  let black_balls := 3
  let white_balls := 2
  let favorable_outcomes := (Nat.choose 3 1) * (Nat.choose 2 1)
  let total_outcomes := Nat.choose 5 2
  (favorable_outcomes / total_outcomes) = (3 / 5) := 
by
  sorry

end NUMINAMATH_GPT_probability_black_white_ball_l2223_222381


namespace NUMINAMATH_GPT_sequence_sum_of_geometric_progressions_l2223_222341

theorem sequence_sum_of_geometric_progressions
  (u1 v1 q p : ℝ)
  (h1 : u1 + v1 = 0)
  (h2 : u1 * q + v1 * p = 0) :
  u1 * q^2 + v1 * p^2 = 0 :=
by sorry

end NUMINAMATH_GPT_sequence_sum_of_geometric_progressions_l2223_222341


namespace NUMINAMATH_GPT_length_of_second_train_l2223_222301

def first_train_length : ℝ := 290
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def cross_time : ℝ := 9

noncomputable def first_train_speed_mps := (first_train_speed_kmph * 1000) / 3600
noncomputable def second_train_speed_mps := (second_train_speed_kmph * 1000) / 3600
noncomputable def relative_speed := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance_covered := relative_speed * cross_time
noncomputable def second_train_length := total_distance_covered - first_train_length

theorem length_of_second_train : second_train_length = 209.95 := by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l2223_222301


namespace NUMINAMATH_GPT_unpacked_books_30_l2223_222372

theorem unpacked_books_30 :
  let total_books := 1485 * 42
  let books_per_box := 45
  total_books % books_per_box = 30 :=
by
  let total_books := 1485 * 42
  let books_per_box := 45
  have h : total_books % books_per_box = 30 := sorry
  exact h

end NUMINAMATH_GPT_unpacked_books_30_l2223_222372


namespace NUMINAMATH_GPT_similar_triangles_height_l2223_222394

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_height_l2223_222394


namespace NUMINAMATH_GPT_cube_root_simplification_l2223_222368

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_simplification_l2223_222368


namespace NUMINAMATH_GPT_dessert_menu_count_l2223_222326

def Dessert : Type := {d : String // d = "cake" ∨ d = "pie" ∨ d = "ice cream" ∨ d = "pudding"}

def valid_menu (menu : Fin 7 → Dessert) : Prop :=
  (menu 0).1 ≠ (menu 1).1 ∧
  menu 1 = ⟨"ice cream", Or.inr (Or.inr (Or.inl rfl))⟩ ∧
  (menu 1).1 ≠ (menu 2).1 ∧
  (menu 2).1 ≠ (menu 3).1 ∧
  (menu 3).1 ≠ (menu 4).1 ∧
  (menu 4).1 ≠ (menu 5).1 ∧
  menu 5 = ⟨"cake", Or.inl rfl⟩ ∧
  (menu 5).1 ≠ (menu 6).1

def total_valid_menus : Nat :=
  4 * 1 * 3 * 3 * 3 * 1 * 3

theorem dessert_menu_count : ∃ (count : Nat), count = 324 ∧ count = total_valid_menus :=
  sorry

end NUMINAMATH_GPT_dessert_menu_count_l2223_222326


namespace NUMINAMATH_GPT_rachel_removed_bottle_caps_l2223_222314

def original_bottle_caps : ℕ := 87
def remaining_bottle_caps : ℕ := 40

theorem rachel_removed_bottle_caps :
  original_bottle_caps - remaining_bottle_caps = 47 := by
  sorry

end NUMINAMATH_GPT_rachel_removed_bottle_caps_l2223_222314


namespace NUMINAMATH_GPT_bill_miles_sunday_l2223_222377

variables (B : ℕ)
def miles_ran_Bill_Saturday := B
def miles_ran_Bill_Sunday := B + 4
def miles_ran_Julia_Sunday := 2 * (B + 4)
def total_miles_ran := miles_ran_Bill_Saturday + miles_ran_Bill_Sunday + miles_ran_Julia_Sunday

theorem bill_miles_sunday (h1 : total_miles_ran B = 32) : 
  miles_ran_Bill_Sunday B = 9 := 
by sorry

end NUMINAMATH_GPT_bill_miles_sunday_l2223_222377


namespace NUMINAMATH_GPT_math_problem_proof_l2223_222371

noncomputable def question_to_equivalent_proof_problem : Prop :=
  ∃ (p q r : ℤ), 
    (p + q + r = 0) ∧ 
    (p * q + q * r + r * p = -2023) ∧ 
    (|p| + |q| + |r| = 84)

theorem math_problem_proof : question_to_equivalent_proof_problem := 
  by 
    -- proof goes here
    sorry

end NUMINAMATH_GPT_math_problem_proof_l2223_222371


namespace NUMINAMATH_GPT_find_m_l2223_222323

theorem find_m (m : ℝ) (a b : ℝ × ℝ)
  (ha : a = (3, m)) (hb : b = (1, -2))
  (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) :
  m = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l2223_222323


namespace NUMINAMATH_GPT_sequence_x_value_l2223_222302

theorem sequence_x_value (p q r x : ℕ) 
  (h1 : 13 = 5 + p + q) 
  (h2 : r = p + q + 13) 
  (h3 : x = 13 + r + 40) : 
  x = 74 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_x_value_l2223_222302
