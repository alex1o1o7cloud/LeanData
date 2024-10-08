import Mathlib

namespace f_is_periodic_with_period_4a_l7_7737

variable (f : ℝ → ℝ) (a : ℝ)

theorem f_is_periodic_with_period_4a (h : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_is_periodic_with_period_4a_l7_7737


namespace sector_area_correct_l7_7440

noncomputable def sector_area (r θ : ℝ) : ℝ := 0.5 * θ * r^2

theorem sector_area_correct (r θ : ℝ) (hr : r = 2) (hθ : θ = 2 * Real.pi / 3) :
  sector_area r θ = 4 * Real.pi / 3 :=
by
  subst hr
  subst hθ
  sorry

end sector_area_correct_l7_7440


namespace range_of_slope_exists_k_for_collinearity_l7_7054

def line_equation (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x + 3

noncomputable def intersect_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Assume a function that computes the intersection points (x₁, y₁) and (x₂, y₂)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v2 = (c * v1.1, c * v1.2)

theorem range_of_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0) :
  -4/3 < k ∧ k < 0 := 
sorry

theorem exists_k_for_collinearity (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0)
  (h5 : -4/3 < k ∧ k < 0) :
  collinear (2 - x₁ - x₂, -(y₁ + y₂)) (-2, 1) ↔ k = -1/2 :=
sorry


end range_of_slope_exists_k_for_collinearity_l7_7054


namespace digit_8_appears_300_times_l7_7621

-- Define a function that counts the occurrences of a specific digit in a list of numbers
def count_digit_occurrences (digit : Nat) (range : List Nat) : Nat :=
  range.foldl (λ acc n => acc + (Nat.digits 10 n).count digit) 0

-- Theorem statement: The digit 8 appears 300 times in the list of integers from 1 to 1000
theorem digit_8_appears_300_times : count_digit_occurrences 8 (List.range' 0 1000) = 300 :=
by
  sorry

end digit_8_appears_300_times_l7_7621


namespace weight_of_b_l7_7967

-- Define the weights of a, b, and c
variables (W_a W_b W_c : ℝ)

-- Define the heights of a, b, and c
variables (h_a h_b h_c : ℝ)

-- Given conditions
axiom average_weight_abc : (W_a + W_b + W_c) / 3 = 45
axiom average_weight_ab : (W_a + W_b) / 2 = 40
axiom average_weight_bc : (W_b + W_c) / 2 = 47
axiom height_condition : h_a + h_c = 2 * h_b
axiom odd_sum_weights : (W_a + W_b + W_c) % 2 = 1

-- Prove that the weight of b is 39 kg
theorem weight_of_b : W_b = 39 :=
by sorry

end weight_of_b_l7_7967


namespace minimal_reciprocal_sum_l7_7784

theorem minimal_reciprocal_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    (4 / m) + (1 / n) = (30 / (m * n)) → m = 10 ∧ n = 5 :=
sorry

end minimal_reciprocal_sum_l7_7784


namespace smallest_portion_bread_l7_7461

theorem smallest_portion_bread (a d : ℚ) (h1 : 5 * a = 100) (h2 : 24 * d = 11 * a) :
  a - 2 * d = 5 / 3 :=
by
  -- Solution proof goes here...
  sorry -- placeholder for the proof

end smallest_portion_bread_l7_7461


namespace inequality_proof_l7_7539

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := 
by
  sorry

end inequality_proof_l7_7539


namespace committee_form_count_l7_7174

def numWaysToFormCommittee (departments : Fin 4 → (ℕ × ℕ)) : ℕ :=
  let waysCase1 := 6 * 81 * 81
  let waysCase2 := 6 * 9 * 9 * 2 * 9 * 9
  waysCase1 + waysCase2

theorem committee_form_count (departments : Fin 4 → (ℕ × ℕ)) 
  (h : ∀ i, departments i = (3, 3)) :
  numWaysToFormCommittee departments = 48114 := 
by
  sorry

end committee_form_count_l7_7174


namespace area_of_triangle_PQR_l7_7786

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 2 }
def Q : Point := { x := 7, y := 2 }
def R : Point := { x := 5, y := 9 }

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangleArea P Q R = 17.5 := by
  sorry

end area_of_triangle_PQR_l7_7786


namespace prop1_prop3_l7_7221

def custom_op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

theorem prop1 (x y : ℝ) : custom_op x y = custom_op y x :=
by sorry

theorem prop3 (x : ℝ) : custom_op (x + 1) (x - 1) = custom_op x x - 1 :=
by sorry

end prop1_prop3_l7_7221


namespace units_digit_17_pow_2023_l7_7275

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l7_7275


namespace find_divisor_l7_7099

theorem find_divisor
  (Dividend : ℕ)
  (Quotient : ℕ)
  (Remainder : ℕ)
  (h1 : Dividend = 686)
  (h2 : Quotient = 19)
  (h3 : Remainder = 2) :
  ∃ (Divisor : ℕ), (Dividend = (Divisor * Quotient) + Remainder) ∧ Divisor = 36 :=
by
  sorry

end find_divisor_l7_7099


namespace symmetricPointCorrectCount_l7_7521

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the four symmetry conditions
def isSymmetricXaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricYOZplane (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := -P.z }
def isSymmetricYaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricOrigin (P Q : Point3D) : Prop := Q = { x := -P.x, y := -P.y, z := -P.z }

-- Define a theorem to count the valid symmetric conditions
theorem symmetricPointCorrectCount (P : Point3D) :
  (isSymmetricXaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricYOZplane P { x := P.x, y := -P.y, z := -P.z } = true → false) ∧
  (isSymmetricYaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricOrigin P { x := -P.x, y := -P.y, z := -P.z } = true → true) :=
by
  sorry

end symmetricPointCorrectCount_l7_7521


namespace amy_lily_tie_probability_l7_7557

theorem amy_lily_tie_probability (P_Amy P_Lily : ℚ) (hAmy : P_Amy = 4/9) (hLily : P_Lily = 1/3) :
  1 - P_Amy - (↑P_Lily : ℚ) = 2 / 9 := by
  sorry

end amy_lily_tie_probability_l7_7557


namespace range_of_a_l7_7327

theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, (p.1 - 2 * a) ^ 2 + (p.2 - (a + 3)) ^ 2 = 4 → p.1 ^ 2 + p.2 ^ 2 = 1) →
  -1 < a ∧ a < 0 := 
sorry

end range_of_a_l7_7327


namespace correctness_check_l7_7822

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l7_7822


namespace quadratic_inequality_condition_l7_7943

theorem quadratic_inequality_condition (a b c : ℝ) (h : a < 0) (disc : b^2 - 4 * a * c ≤ 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c ≤ 0 :=
sorry

end quadratic_inequality_condition_l7_7943


namespace prime_cube_solution_l7_7480

theorem prime_cube_solution (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : p^3 = p^2 + q^2 + r^2) : 
  p = 3 ∧ q = 3 ∧ r = 3 :=
by
  sorry

end prime_cube_solution_l7_7480


namespace son_present_age_l7_7827

variable (S F : ℕ)

-- Given conditions
def father_age := F = S + 34
def future_age_rel := F + 2 = 2 * (S + 2)

-- Theorem to prove the son's current age
theorem son_present_age (h₁ : father_age S F) (h₂ : future_age_rel S F) : S = 32 := by
  sorry

end son_present_age_l7_7827


namespace function_intersects_y_axis_at_0_neg4_l7_7317

theorem function_intersects_y_axis_at_0_neg4 :
  (∃ x y : ℝ, y = 4 * x - 4 ∧ x = 0 ∧ y = -4) :=
sorry

end function_intersects_y_axis_at_0_neg4_l7_7317


namespace earrings_ratio_l7_7577

theorem earrings_ratio :
  ∃ (M R : ℕ), 10 = M / 4 ∧ 10 + M + R = 70 ∧ M / R = 2 := by
  sorry

end earrings_ratio_l7_7577


namespace circle_equation_line_equation_l7_7010

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def point_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (1, 1)
noncomputable def point_B : ℝ × ℝ := (4, 2)

theorem circle_equation :
  circle_C point_O.1 point_O.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 :=
by sorry

noncomputable def line_l_case1 (x : ℝ) : Prop :=
  x = 3 / 2

noncomputable def line_l_case2 (x y : ℝ) : Prop :=
  8 * x + 6 * y - 39 = 0

noncomputable def center_C : ℝ × ℝ := (4, -3)
noncomputable def radius_C : ℝ := 5

noncomputable def point_through_l : ℝ × ℝ := (3 / 2, 9 / 2)

theorem line_equation : 
(∀ (M N : ℝ × ℝ), circle_C M.1 M.2 ∧ circle_C N.1 N.2 → ∃ C_slave : Prop, 
(C_slave → 
((line_l_case1 (point_through_l.1)) ∨ 
(line_l_case2 point_through_l.1 point_through_l.2)))) :=
by sorry

end circle_equation_line_equation_l7_7010


namespace square_perimeter_l7_7770

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l7_7770


namespace fraction_equiv_l7_7865

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l7_7865


namespace quadratic_function_even_l7_7807

theorem quadratic_function_even (a b : ℝ) (h1 : ∀ x : ℝ, x^2 + (a-1)*x + a + b = x^2 - (a-1)*x + a + b) (h2 : 4 + (a-1)*2 + a + b = 0) : a + b = -4 := 
sorry

end quadratic_function_even_l7_7807


namespace sin_arcsin_plus_arctan_l7_7218

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l7_7218


namespace general_term_formula_l7_7970

def sequence_sums (n : ℕ) : ℕ := 2 * n^2 + n

theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : S = sequence_sums) :
  (∀ n, a n = S n - S (n-1)) → ∀ n, a n = 4 * n - 1 :=
by
  sorry

end general_term_formula_l7_7970


namespace direct_proportion_function_l7_7123

theorem direct_proportion_function (m : ℝ) (h1 : m^2 - 8 = 1) (h2 : m ≠ 3) : m = -3 :=
by
  sorry

end direct_proportion_function_l7_7123


namespace part1_part2_l7_7237

-- Condition definitions
def income2017 : ℝ := 2500
def income2019 : ℝ := 3600
def n : ℕ := 2

-- Part 1: Prove the annual growth rate
theorem part1 (x : ℝ) (hx : income2019 = income2017 * (1 + x) ^ n) : x = 0.2 :=
by sorry

-- Part 2: Prove reaching 4200 yuan with the same growth rate
theorem part2 (hx : income2019 = income2017 * (1 + 0.2) ^ n) : 3600 * (1 + 0.2) ≥ 4200 :=
by sorry

end part1_part2_l7_7237


namespace sqrt_abc_sum_eq_162sqrt2_l7_7491

theorem sqrt_abc_sum_eq_162sqrt2 (a b c : ℝ) (h1 : b + c = 15) (h2 : c + a = 18) (h3 : a + b = 21) :
    Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 :=
by
  sorry

end sqrt_abc_sum_eq_162sqrt2_l7_7491


namespace find_complement_l7_7390

-- Define predicate for a specific universal set U and set A
def universal_set (a : ℤ) (x : ℤ) : Prop :=
  x = a^2 - 2 ∨ x = 2 ∨ x = 1

def set_A (a : ℤ) (x : ℤ) : Prop :=
  x = a ∨ x = 1

-- Define complement of A with respect to U
def complement_U_A (a : ℤ) (x : ℤ) : Prop :=
  universal_set a x ∧ ¬ set_A a x

-- Main theorem statement
theorem find_complement (a : ℤ) (h : a ≠ 2) : { x | complement_U_A a x } = {2} :=
by
  sorry

end find_complement_l7_7390


namespace arrangement_of_70616_l7_7679

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end arrangement_of_70616_l7_7679


namespace arable_land_decrease_max_l7_7182

theorem arable_land_decrease_max
  (A₀ : ℕ := 100000)
  (grain_yield_increase : ℝ := 1.22)
  (per_capita_increase : ℝ := 1.10)
  (pop_growth_rate : ℝ := 0.01)
  (years : ℕ := 10) :
  ∃ (max_decrease : ℕ), max_decrease = 4 := sorry

end arable_land_decrease_max_l7_7182


namespace lemonade_percentage_l7_7954

theorem lemonade_percentage (L : ℝ) : 
  (0.4 * (1 - L / 100) + 0.6 * 0.55 = 0.65) → L = 20 :=
by
  sorry

end lemonade_percentage_l7_7954


namespace part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l7_7682

noncomputable def f (x : ℝ) := Real.log x
noncomputable def deriv_f (x : ℝ) := 1 / x

theorem part1_am_eq_ln_am1_minus_1 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m = Real.log (a_n (m - 1)) - 1 :=
sorry

theorem part2_am_le_am1_minus_2 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m ≤ a_n (m - 1) - 2 :=
sorry

theorem part3_k_is_3 (a_n : ℕ → ℝ) :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ k → (a_n n) - (a_n (n - 1)) = (a_n 2) - (a_n 1) :=
sorry

end part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l7_7682


namespace approximation_of_11_28_relative_to_10000_l7_7359

def place_value_to_approximate (x : Float) (reference : Float) : String :=
  if x < reference / 10 then "tens"
  else if x < reference / 100 then "hundreds"
  else if x < reference / 1000 then "thousands"
  else if x < reference / 10000 then "ten thousands"
  else "greater than ten thousands"

theorem approximation_of_11_28_relative_to_10000:
  place_value_to_approximate 11.28 10000 = "hundreds" :=
by
  -- Insert proof here
  sorry

end approximation_of_11_28_relative_to_10000_l7_7359


namespace percent_difference_l7_7322

theorem percent_difference:
  let percent_value1 := (55 / 100) * 40
  let fraction_value2 := (4 / 5) * 25
  percent_value1 - fraction_value2 = 2 :=
by
  sorry

end percent_difference_l7_7322


namespace number_of_containers_needed_l7_7481

/-
  Define the parameters for the given problem
-/
def bags_suki : ℝ := 6.75
def weight_per_bag_suki : ℝ := 27

def bags_jimmy : ℝ := 4.25
def weight_per_bag_jimmy : ℝ := 23

def bags_natasha : ℝ := 3.80
def weight_per_bag_natasha : ℝ := 31

def container_capacity : ℝ := 17

/-
  The total weight bought by each person and the total combined weight
-/
def total_weight_suki : ℝ := bags_suki * weight_per_bag_suki
def total_weight_jimmy : ℝ := bags_jimmy * weight_per_bag_jimmy
def total_weight_natasha : ℝ := bags_natasha * weight_per_bag_natasha

def total_weight_combined : ℝ := total_weight_suki + total_weight_jimmy + total_weight_natasha

/-
  Prove that number of containers needed is 24
-/
theorem number_of_containers_needed : 
  Nat.ceil (total_weight_combined / container_capacity) = 24 := 
by
  sorry

end number_of_containers_needed_l7_7481


namespace quadratic_root_l7_7079

theorem quadratic_root (k : ℝ) (h : (1:ℝ)^2 - 3 * (1 : ℝ) - k = 0) : k = -2 :=
sorry

end quadratic_root_l7_7079


namespace maximal_sector_angle_l7_7590

theorem maximal_sector_angle (a : ℝ) (r : ℝ) (l : ℝ) (α : ℝ)
  (h1 : l + 2 * r = a)
  (h2 : 0 < r ∧ r < a / 2)
  (h3 : α = l / r)
  (eval_area : ∀ (l r : ℝ), S = 1 / 2 * l * r)
  (S : ℝ) :
  α = 2 := sorry

end maximal_sector_angle_l7_7590


namespace intersection_interval_l7_7233

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (x: ℝ) : ℝ := 7 - 2 * x

theorem intersection_interval : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = g x := 
sorry

end intersection_interval_l7_7233


namespace fraction_power_equals_l7_7799

theorem fraction_power_equals :
  (5 / 7) ^ 7 = (78125 : ℚ) / 823543 := 
by
  sorry

end fraction_power_equals_l7_7799


namespace smallest_number_in_set_l7_7113

open Real

theorem smallest_number_in_set :
  ∀ (a b c d : ℝ), a = -3 → b = 3⁻¹ → c = -abs (-1 / 3) → d = 0 →
    a < b ∧ a < c ∧ a < d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_in_set_l7_7113


namespace gcd_a_b_l7_7246

def a (n : ℤ) : ℤ := n^5 + 6 * n^3 + 8 * n
def b (n : ℤ) : ℤ := n^4 + 4 * n^2 + 3

theorem gcd_a_b (n : ℤ) : ∃ d : ℤ, d = Int.gcd (a n) (b n) ∧ (d = 1 ∨ d = 3) :=
by
  sorry

end gcd_a_b_l7_7246


namespace problem1_l7_7642

variables (m n : ℝ)

axiom cond1 : 4 * m + n = 90
axiom cond2 : 2 * m - 3 * n = 10

theorem problem1 : (m + 2 * n) ^ 2 - (3 * m - n) ^ 2 = -900 := sorry

end problem1_l7_7642


namespace line_intersects_ellipse_max_chord_length_l7_7119

theorem line_intersects_ellipse (m : ℝ) :
  (-2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), (9 * x^2 + 6 * m * x + 2 * m^2 - 8 = 0) ∧ (y = (3 / 2) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) :=
sorry

theorem max_chord_length (m : ℝ) :
  m = 0 → (∃ (A B : ℝ × ℝ),
  ((A.1^2 / 4 + A.2^2 / 9 = 1) ∧ (A.2 = (3 / 2) * A.1 + m)) ∧
  ((B.1^2 / 4 + B.2^2 / 9 = 1) ∧ (B.2 = (3 / 2) * B.1 + m)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 26 / 3)) :=
sorry

end line_intersects_ellipse_max_chord_length_l7_7119


namespace total_number_of_coins_l7_7592

theorem total_number_of_coins (x n : Nat) (h1 : 15 * 5 = 75) (h2 : 125 - 75 = 50)
  (h3 : x = 50 / 2) (h4 : n = x + 15) : n = 40 := by
  sorry

end total_number_of_coins_l7_7592


namespace nicky_profit_l7_7410

theorem nicky_profit (value_traded_away value_received : ℤ)
  (h1 : value_traded_away = 2 * 8)
  (h2 : value_received = 21) :
  value_received - value_traded_away = 5 :=
by
  sorry

end nicky_profit_l7_7410


namespace necessarily_true_statement_l7_7076

-- Define the four statements as propositions
def Statement1 (d : ℕ) : Prop := d = 2
def Statement2 (d : ℕ) : Prop := d ≠ 3
def Statement3 (d : ℕ) : Prop := d = 5
def Statement4 (d : ℕ) : Prop := d % 2 = 0

-- The main theorem stating that given one of the statements is false, Statement3 is necessarily true
theorem necessarily_true_statement (d : ℕ) 
  (h1 : Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ ¬ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ ¬ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ ¬ Statement2 d ∧ Statement3 d ∧ Statement4 d)):
  Statement2 d :=
sorry

end necessarily_true_statement_l7_7076


namespace profits_ratio_l7_7868

-- Definitions
def investment_ratio (p q : ℕ) := 7 * p = 5 * q
def investment_period_p := 10
def investment_period_q := 20

-- Prove the ratio of profits
theorem profits_ratio (p q : ℕ) (h1 : investment_ratio p q) :
  (7 * p * investment_period_p / (5 * q * investment_period_q)) = 7 / 10 :=
sorry

end profits_ratio_l7_7868


namespace inequality_order_l7_7914

theorem inequality_order (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h : (a^2 / (b^2 + c^2)) < (b^2 / (c^2 + a^2)) ∧ (b^2 / (c^2 + a^2)) < (c^2 / (a^2 + b^2))) :
  |a| < |b| ∧ |b| < |c| := 
sorry

end inequality_order_l7_7914


namespace Trevor_tip_l7_7816

variable (Uber Lyft Taxi : ℕ)
variable (TotalCost : ℕ)

theorem Trevor_tip 
  (h1 : Uber = Lyft + 3) 
  (h2 : Lyft = Taxi + 4) 
  (h3 : Uber = 22) 
  (h4 : TotalCost = 18)
  (h5 : Taxi = 15) :
  (TotalCost - Taxi) * 100 / Taxi = 20 := by
  sorry

end Trevor_tip_l7_7816


namespace hyperbola_eccentricity_l7_7132

theorem hyperbola_eccentricity (a c b : ℝ) (h₀ : b = 3)
  (h₁ : ∃ p, (p = 5) ∧ (a^2 + b^2 = (p : ℝ)^2))
  (h₂ : ∃ f, f = (p : ℝ)) :
  ∃ e, e = c / a ∧ e = 5 / 4 :=
by
  obtain ⟨p, hp, hap⟩ := h₁
  obtain ⟨f, hf⟩ := h₂
  sorry

end hyperbola_eccentricity_l7_7132


namespace number_of_integers_l7_7449

theorem number_of_integers (n : ℤ) : 
    (100 < n ∧ n < 300) ∧ (n % 7 = n % 9) → 
    (∃ count: ℕ, count = 21) := by
  sorry

end number_of_integers_l7_7449


namespace length_AB_l7_7485

theorem length_AB (x : ℝ) (h1 : 0 < x)
  (hG : G = (0 + 1) / 2)
  (hH : H = (0 + G) / 2)
  (hI : I = (0 + H) / 2)
  (hJ : J = (0 + I) / 2)
  (hAJ : J - 0 = 2) :
  x = 32 := by
  sorry

end length_AB_l7_7485


namespace businessmen_neither_coffee_nor_tea_l7_7566

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (C T : Finset ℕ)
  (hC : C.card = 15)
  (hT : T.card = 14)
  (hCT : (C ∩ T).card = 7)
  (htotal : total = 30) : 
  total - (C ∪ T).card = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l7_7566


namespace effective_discount_l7_7089

theorem effective_discount (original_price sale_price price_after_coupon : ℝ) :
  sale_price = 0.4 * original_price →
  price_after_coupon = 0.7 * sale_price →
  (original_price - price_after_coupon) / original_price * 100 = 72 :=
by
  intros h1 h2
  sorry

end effective_discount_l7_7089


namespace number_of_testing_methods_l7_7804

-- Definitions based on conditions
def num_genuine_items : ℕ := 6
def num_defective_items : ℕ := 4
def total_tests : ℕ := 5

-- Theorem stating the number of testing methods
theorem number_of_testing_methods 
    (h1 : total_tests = 5) 
    (h2 : num_genuine_items = 6) 
    (h3 : num_defective_items = 4) :
    ∃ n : ℕ, n = 576 := 
sorry

end number_of_testing_methods_l7_7804


namespace find_union_A_B_r_find_range_m_l7_7108

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x m : ℝ) : Prop := (x - m) * (x - m - 1) ≥ 0

theorem find_union_A_B_r (x : ℝ) : A x ∨ B x 1 := by
  sorry

theorem find_range_m (m : ℝ) (x : ℝ) : (∀ x, A x ↔ B x m) ↔ (m ≥ 3 ∨ m ≤ -2) := by
  sorry

end find_union_A_B_r_find_range_m_l7_7108


namespace not_a_perfect_square_l7_7921

theorem not_a_perfect_square :
  ¬ (∃ x, (x: ℝ)^2 = 5^2025) :=
by
  sorry

end not_a_perfect_square_l7_7921


namespace smallest_digit_divisible_by_9_l7_7070

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (0 ≤ d ∧ d < 10) ∧ (∃ k : ℕ, 26 + d = 9 * k) ∧ d = 1 :=
by
  sorry

end smallest_digit_divisible_by_9_l7_7070


namespace f_of_x_plus_1_f_of_2_f_of_x_l7_7195

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x := sorry

theorem f_of_2 : f 2 = 3 := sorry

theorem f_of_x (x : ℝ) : f x = x^2 - 1 := sorry

end f_of_x_plus_1_f_of_2_f_of_x_l7_7195


namespace enclosed_area_eq_two_l7_7850

noncomputable def enclosed_area : ℝ :=
  -∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem enclosed_area_eq_two : enclosed_area = 2 := 
  sorry

end enclosed_area_eq_two_l7_7850


namespace average_stamps_collected_per_day_l7_7519

theorem average_stamps_collected_per_day :
  let a := 10
  let d := 6
  let n := 6
  let total_sum := (n / 2) * (2 * a + (n - 1) * d)
  let average := total_sum / n
  average = 25 :=
by
  sorry

end average_stamps_collected_per_day_l7_7519


namespace perpendicular_lines_and_slope_l7_7143

theorem perpendicular_lines_and_slope (b : ℝ) : (x + 3 * y + 4 = 0) ∧ (b * x + 3 * y + 6 = 0) → b = -9 :=
by
  sorry

end perpendicular_lines_and_slope_l7_7143


namespace simplification_evaluation_l7_7289

noncomputable def simplify_and_evaluate (x : ℤ) : ℚ :=
  (1 - 1 / (x - 1)) * ((x - 1) / ((x - 2) * (x - 2)))

theorem simplification_evaluation (x : ℤ) (h1 : x > 0) (h2 : 3 - x ≥ 0) : 
  simplify_and_evaluate x = 1 :=
by
  have h3 : x = 3 := sorry
  rw [simplify_and_evaluate, h3]
  simp [h3]
  sorry

end simplification_evaluation_l7_7289


namespace alice_age_proof_l7_7606

-- Definitions derived from the conditions
def alice_pens : ℕ := 60
def clara_pens : ℕ := (2 * alice_pens) / 5
def clara_age_in_5_years : ℕ := 61
def clara_current_age : ℕ := clara_age_in_5_years - 5
def age_difference : ℕ := alice_pens - clara_pens

-- Proof statement to be proved
theorem alice_age_proof : (clara_current_age - age_difference = 20) :=
sorry

end alice_age_proof_l7_7606


namespace positive_n_of_single_solution_l7_7891

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l7_7891


namespace interval_of_decrease_l7_7688

theorem interval_of_decrease (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x) :
  ∀ x0 : ℝ, ∀ x1 : ℝ, x0 ≥ 3 → x0 ≤ x1 → f (x1 - 3) ≤ f (x0 - 3) := sorry

end interval_of_decrease_l7_7688


namespace turner_oldest_child_age_l7_7508

theorem turner_oldest_child_age (a b c : ℕ) (avg : ℕ) :
  (a = 6) → (b = 8) → (c = 11) → (avg = 9) → 
  (4 * avg = (a + b + c + x) → x = 11) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end turner_oldest_child_age_l7_7508


namespace min_value_abs_sum_pqr_inequality_l7_7442

theorem min_value_abs_sum (x : ℝ) : |x + 1| + |x - 2| ≥ 3 :=
by
  sorry

theorem pqr_inequality (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := 
by
  have f_min : ∀ x, |x + 1| + |x - 2| ≥ 3 := min_value_abs_sum
  sorry

end min_value_abs_sum_pqr_inequality_l7_7442


namespace old_edition_pages_l7_7602

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l7_7602


namespace mean_of_set_is_16_6_l7_7504

theorem mean_of_set_is_16_6 (m : ℝ) (h : m + 7 = 16) :
  (9 + 11 + 16 + 20 + 27) / 5 = 16.6 :=
by
  -- Proof steps would go here, but we use sorry to skip the proof.
  sorry

end mean_of_set_is_16_6_l7_7504


namespace garden_length_l7_7291

theorem garden_length (columns : ℕ) (distance_between_trees : ℕ) (boundary_distance : ℕ) (h_columns : columns = 12) (h_distance_between_trees : distance_between_trees = 2) (h_boundary_distance : boundary_distance = 5) : 
  ((columns - 1) * distance_between_trees + 2 * boundary_distance) = 32 :=
by 
  sorry

end garden_length_l7_7291


namespace days_matt_and_son_eat_only_l7_7141

theorem days_matt_and_son_eat_only (x y : ℕ) 
  (h1 : x + y = 7)
  (h2 : 2 * x + 8 * y = 38) : 
  x = 3 :=
by
  sorry

end days_matt_and_son_eat_only_l7_7141


namespace solve_inequalities_l7_7012

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end solve_inequalities_l7_7012


namespace pyramid_volume_correct_l7_7923

noncomputable def pyramid_volume (A_PQRS A_PQT A_RST: ℝ) (side: ℝ) (height: ℝ) : ℝ :=
  (1 / 3) * A_PQRS * height

theorem pyramid_volume_correct 
  (A_PQRS : ℝ) (A_PQT : ℝ) (A_RST : ℝ) (side : ℝ) (height_PQT : ℝ) (height_RST : ℝ)
  (h_PQT : 2 * A_PQT / side = height_PQT)
  (h_RST : 2 * A_RST / side = height_RST)
  (eq1 : height_PQT^2 + side^2 = height_RST^2 + (side - height_PQT)^2) 
  (eq2 : height_RST^2 = height_PQT^2 + (height_PQT - side)^2)
  : pyramid_volume A_PQRS A_PQT A_RST = 5120 / 3 :=
by
  -- Skipping the proof steps
  sorry

end pyramid_volume_correct_l7_7923


namespace flower_bed_dimensions_l7_7789

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end flower_bed_dimensions_l7_7789


namespace cube_surface_area_l7_7505

-- Definitions based on conditions from the problem
def edge_length : ℕ := 7
def number_of_faces : ℕ := 6

-- Definition of the problem converted to a theorem in Lean 4
theorem cube_surface_area (edge_length : ℕ) (number_of_faces : ℕ) : 
  number_of_faces * (edge_length * edge_length) = 294 :=
by
  -- Proof steps are omitted, so we put sorry to indicate that the proof is required.
  sorry

end cube_surface_area_l7_7505


namespace distinct_primes_sum_reciprocal_l7_7049

open Classical

theorem distinct_primes_sum_reciprocal (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (hineq: (1 / p : ℚ) + (1 / q) + (1 / r) ≥ 1) 
  : (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨
    (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) := 
sorry

end distinct_primes_sum_reciprocal_l7_7049


namespace no_three_nat_numbers_with_sum_power_of_three_l7_7231

noncomputable def powers_of_3 (n : ℕ) : ℕ := 3^n

theorem no_three_nat_numbers_with_sum_power_of_three :
  ¬ ∃ (a b c : ℕ) (k m n : ℕ), a + b = powers_of_3 k ∧ b + c = powers_of_3 m ∧ c + a = powers_of_3 n :=
by
  sorry

end no_three_nat_numbers_with_sum_power_of_three_l7_7231


namespace weeks_to_buy_bicycle_l7_7116

-- Definitions based on problem conditions
def hourly_wage : Int := 5
def hours_monday : Int := 2
def hours_wednesday : Int := 1
def hours_friday : Int := 3
def weekly_hours : Int := hours_monday + hours_wednesday + hours_friday
def weekly_earnings : Int := weekly_hours * hourly_wage
def bicycle_cost : Int := 180

-- Statement of the theorem to prove
theorem weeks_to_buy_bicycle : ∃ w : Nat, w * weekly_earnings = bicycle_cost :=
by
  -- Since this is a statement only, the proof is omitted
  sorry

end weeks_to_buy_bicycle_l7_7116


namespace distinct_factors_count_l7_7880

-- Given conditions
def eight_squared : ℕ := 8^2
def nine_cubed : ℕ := 9^3
def seven_fifth : ℕ := 7^5
def number : ℕ := eight_squared * nine_cubed * seven_fifth

-- Proving the number of natural-number factors of the given number
theorem distinct_factors_count : 
  (number.factors.count 1 = 294) := sorry

end distinct_factors_count_l7_7880


namespace probability_sum_odd_l7_7757

theorem probability_sum_odd (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h_even : ∃ z : ℕ, z % 2 = 0 ∧ z > 0) 
  (h_odd : ∃ z : ℕ, z % 2 = 1 ∧ z > 0) : 
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ p = 0.5) :=
sorry

end probability_sum_odd_l7_7757


namespace maximum_combined_power_l7_7849

theorem maximum_combined_power (x1 x2 x3 : ℝ) (hx : x1 < 1 ∧ x2 < 1 ∧ x3 < 1) 
    (hcond : 2 * (x1 + x2 + x3) + 4 * (x1 * x2 * x3) = 3 * (x1 * x2 + x1 * x3 + x2 * x3) + 1) : 
    x1 + x2 + x3 ≤ 3 / 4 := 
sorry

end maximum_combined_power_l7_7849


namespace percentage_increase_l7_7383

theorem percentage_increase (new_wage original_wage : ℝ) (h₁ : new_wage = 42) (h₂ : original_wage = 28) :
  ((new_wage - original_wage) / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_l7_7383


namespace tan_alpha_sub_pi_over_8_l7_7073

theorem tan_alpha_sub_pi_over_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)) :
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
by sorry

end tan_alpha_sub_pi_over_8_l7_7073


namespace students_taking_all_three_l7_7800

-- Definitions and Conditions
def total_students : ℕ := 25
def coding_students : ℕ := 12
def chess_students : ℕ := 15
def photography_students : ℕ := 10
def at_least_two_classes : ℕ := 10

-- Request to prove: Number of students taking all three classes
theorem students_taking_all_three (x y w z : ℕ) :
  (x + y + z + w = 10) →
  (coding_students - (10 - y) + chess_students - (10 - w) + (10 - x) = 21) →
  z = 4 :=
by
  intros
  -- Proof will go here
  sorry

end students_taking_all_three_l7_7800


namespace sum_arithmetic_sequence_l7_7428

theorem sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S (n + 1) - S n = a n)
  (h_S2 : S 2 = 4) 
  (h_S4 : S 4 = 16) 
: a 5 + a 6 = 20 :=
sorry

end sum_arithmetic_sequence_l7_7428


namespace number_of_factors_27648_l7_7074

-- Define the number in question
def n : ℕ := 27648

-- State the prime factorization
def n_prime_factors : Nat := 2^10 * 3^3

-- State the theorem to be proven
theorem number_of_factors_27648 : 
  ∃ (f : ℕ), 
  (f = (10+1) * (3+1)) ∧ (f = 44) :=
by
  -- Placeholder for the proof
  sorry

end number_of_factors_27648_l7_7074


namespace y_value_solution_l7_7204

theorem y_value_solution (y : ℝ) (h : (3 / y) - ((4 / y) * (2 / y)) = 1.5) : 
  y = 1 + Real.sqrt (19 / 3) := 
sorry

end y_value_solution_l7_7204


namespace proportion_a_value_l7_7522

theorem proportion_a_value (a b c d : ℝ) (h1 : b = 3) (h2 : c = 4) (h3 : d = 6) (h4 : a / b = c / d) : a = 2 :=
by sorry

end proportion_a_value_l7_7522


namespace sale_price_is_91_percent_of_original_price_l7_7178

variable (x : ℝ)
variable (h_increase : ∀ p : ℝ, p * 1.4)
variable (h_sale : ∀ p : ℝ, p * 0.65)

/--The sale price of an item is 91% of the original price.-/
theorem sale_price_is_91_percent_of_original_price {x : ℝ} 
  (h_increase : ∀ p, p * 1.4 = 1.40 * p)
  (h_sale : ∀ p, p * 0.65 = 0.65 * p): 
  (0.65 * 1.40 * x = 0.91 * x) := 
by 
  sorry

end sale_price_is_91_percent_of_original_price_l7_7178


namespace system_of_equations_solution_l7_7616

theorem system_of_equations_solution (x y z : ℝ) :
  (x * y + x * z = 8 - x^2) →
  (x * y + y * z = 12 - y^2) →
  (y * z + z * x = -4 - z^2) →
  (x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1) :=
by
  sorry

end system_of_equations_solution_l7_7616


namespace smallest_positive_integer_l7_7430

theorem smallest_positive_integer (n : ℕ) (h : 721 * n % 30 = 1137 * n % 30) :
  ∃ k : ℕ, k > 0 ∧ n = 2 * k :=
by
  sorry

end smallest_positive_integer_l7_7430


namespace billiard_ball_weight_l7_7337

theorem billiard_ball_weight (w_box w_box_with_balls : ℝ) (h_w_box : w_box = 0.5) 
(h_w_box_with_balls : w_box_with_balls = 1.82) : 
    let total_weight_balls := w_box_with_balls - w_box;
    let weight_one_ball := total_weight_balls / 6;
    weight_one_ball = 0.22 :=
by
  sorry

end billiard_ball_weight_l7_7337


namespace emily_first_round_points_l7_7194

theorem emily_first_round_points (x : ℤ) 
  (second_round : ℤ := 33) 
  (last_round_loss : ℤ := 48) 
  (total_points_end : ℤ := 1) 
  (eqn : x + second_round - last_round_loss = total_points_end) : 
  x = 16 := 
by 
  sorry

end emily_first_round_points_l7_7194


namespace value_of_five_minus_c_l7_7484

theorem value_of_five_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 7 + d = 10 + c) :
  5 - c = 6 :=
by
  sorry

end value_of_five_minus_c_l7_7484


namespace birds_initially_l7_7778

-- Definitions of the conditions
def initial_birds (B : Nat) := B
def initial_storks := 4
def additional_storks := 6
def total := 13

-- The theorem we need to prove
theorem birds_initially (B : Nat) (h : initial_birds B + initial_storks + additional_storks = total) : initial_birds B = 3 :=
by
  -- The proof can go here
  sorry

end birds_initially_l7_7778


namespace dollar_op_5_neg2_l7_7715

def dollar_op (x y : Int) : Int := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_5_neg2 :
  dollar_op 5 (-2) = -45 := by
  sorry

end dollar_op_5_neg2_l7_7715


namespace age_of_person_l7_7677

/-- Given that Noah's age is twice someone's age and Noah will be 22 years old after 10 years, 
    this theorem states that the age of the person whose age is half of Noah's age is 6 years old. -/
theorem age_of_person (N : ℕ) (P : ℕ) (h1 : P = N / 2) (h2 : N + 10 = 22) : P = 6 := by
  sorry

end age_of_person_l7_7677


namespace exterior_angle_DEF_l7_7724

theorem exterior_angle_DEF :
  let heptagon_angle := (180 * (7 - 2)) / 7
  let octagon_angle := (180 * (8 - 2)) / 8
  let total_degrees := 360
  total_degrees - (heptagon_angle + octagon_angle) = 96.43 :=
by
  sorry

end exterior_angle_DEF_l7_7724


namespace average_weight_of_boys_l7_7458

theorem average_weight_of_boys
  (average_weight_girls : ℕ) 
  (average_weight_students : ℕ) 
  (h_girls : average_weight_girls = 45)
  (h_students : average_weight_students = 50) : 
  ∃ average_weight_boys : ℕ, average_weight_boys = 55 :=
by
  sorry

end average_weight_of_boys_l7_7458


namespace new_person_weight_l7_7021

-- Define the conditions of the problem
variables (avg_weight : ℝ) (weight_replaced_person : ℝ) (num_persons : ℕ)
variable (weight_increase : ℝ)

-- Given conditions
def condition (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ) (weight_increase : ℝ) : Prop :=
  num_persons = 10 ∧ weight_replaced_person = 60 ∧ weight_increase = 5

-- The proof problem
theorem new_person_weight (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ)
  (weight_increase : ℝ) (h : condition avg_weight weight_replaced_person num_persons weight_increase) :
  weight_replaced_person + num_persons * weight_increase = 110 :=
sorry

end new_person_weight_l7_7021


namespace simplify_expression_l7_7082

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 18 = 45 * w + 18 := by
  sorry

end simplify_expression_l7_7082


namespace bus_ride_time_l7_7151

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l7_7151


namespace largest_three_digit_multiple_of_6_sum_15_l7_7316

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l7_7316


namespace symmetric_circle_eq_l7_7265

theorem symmetric_circle_eq :
  (∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l7_7265


namespace sarah_cupcakes_l7_7812

theorem sarah_cupcakes (c k d : ℕ) (h1 : c + k = 6) (h2 : 90 * c + 40 * k = 100 * d) : c = 4 ∨ c = 6 :=
by {
  sorry -- Proof is omitted as requested.
}

end sarah_cupcakes_l7_7812


namespace total_participants_l7_7199

theorem total_participants
  (F M : ℕ) 
  (half_female_democrats : F / 2 = 125)
  (one_third_democrats : (F + M) / 3 = (125 + M / 4))
  : F + M = 1750 :=
by
  sorry

end total_participants_l7_7199


namespace problem_solution_l7_7892

theorem problem_solution :
  { x : ℝ // (x / 4 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) } = { x : ℝ // x ∈ Set.Ico (-4 : ℝ) (-(3 / 2) : ℝ) } :=
by
  sorry

end problem_solution_l7_7892


namespace least_element_of_special_set_l7_7702

theorem least_element_of_special_set :
  ∃ T : Finset ℕ, T ⊆ Finset.range 16 ∧ T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧ 
    (∀ {z : ℕ}, z ∈ T → ∀ {x y : ℕ}, x ≠ y → x ∈ T → y ∈ T → z ≠ x + y) ∧
    ∀ (x : ℕ), x ∈ T → x ≥ 4 :=
sorry

end least_element_of_special_set_l7_7702


namespace find_utilities_second_l7_7144

def rent_first : ℝ := 800
def utilities_first : ℝ := 260
def distance_first : ℕ := 31
def rent_second : ℝ := 900
def distance_second : ℕ := 21
def cost_per_mile : ℝ := 0.58
def days_per_month : ℕ := 20
def cost_difference : ℝ := 76

-- Helper definitions
def driving_cost (distance : ℕ) : ℝ :=
  distance * days_per_month * cost_per_mile

def total_cost_first : ℝ :=
  rent_first + utilities_first + driving_cost distance_first

def total_cost_second_no_utilities : ℝ :=
  rent_second + driving_cost distance_second

theorem find_utilities_second :
  ∃ (utilities_second : ℝ),
  total_cost_first - total_cost_second_no_utilities = cost_difference →
  utilities_second = 200 :=
sorry

end find_utilities_second_l7_7144


namespace range_of_a_l7_7363

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l7_7363


namespace triangle_ABC_area_l7_7000

open Real

-- Define points A, B, and C
structure Point :=
  (x: ℝ)
  (y: ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨8, 2⟩
def C : Point := ⟨6, -1⟩

-- Function to calculate the area of a triangle given vertices A, B, and C
noncomputable def triangle_area (A B C : Point) : ℝ := 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

-- The statement to be proved
theorem triangle_ABC_area : triangle_area A B C = 13.5 :=
by
  sorry

end triangle_ABC_area_l7_7000


namespace denominator_or_divisor_cannot_be_zero_l7_7641

theorem denominator_or_divisor_cannot_be_zero (a b c : ℝ) : b ≠ 0 ∧ c ≠ 0 → (a / b ≠ a ∨ a / c ≠ a) :=
by
  intro h
  sorry

end denominator_or_divisor_cannot_be_zero_l7_7641


namespace range_of_m_l7_7347

noncomputable def f (x : ℝ) : ℝ := sorry -- to be defined as an odd, decreasing function

theorem range_of_m 
  (hf_odd : ∀ x, f (-x) = -f x) -- f is odd
  (hf_decreasing : ∀ x y, x < y → f y < f x) -- f is strictly decreasing
  (h_condition : ∀ m, f (1 - m) + f (1 - m^2) < 0) :
  ∀ m, (0 < m ∧ m < 1) :=
sorry

end range_of_m_l7_7347


namespace solve_for_a_l7_7247

theorem solve_for_a (a x : ℝ) (h : 2 * x + 3 * a = 10) (hx : x = 2) : a = 2 :=
by
  rw [hx] at h
  linarith

end solve_for_a_l7_7247


namespace none_of_these_l7_7358

def table : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 33), (4, 61), (5, 101)]

def formula_A (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_B (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_C (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_D (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem none_of_these :
  ¬ (∀ (x y : ℕ), (x, y) ∈ table → (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by {
  sorry
}

end none_of_these_l7_7358


namespace missing_fraction_of_coins_l7_7750

-- Defining the initial conditions
def total_coins (x : ℕ) := x
def lost_coins (x : ℕ) := (1 / 2) * x
def found_coins (x : ℕ) := (3 / 8) * x

-- Theorem statement
theorem missing_fraction_of_coins (x : ℕ) : 
  (total_coins x - lost_coins x + found_coins x) = (7 / 8) * x :=
by
  sorry  -- proof is omitted as per the instructions

end missing_fraction_of_coins_l7_7750


namespace angle_between_vectors_45_degrees_l7_7431

open Real

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := sqrt (vec_dot v v)

noncomputable def vec_angle (v w : ℝ × ℝ) : ℝ := arccos (vec_dot v w / (vec_mag v * vec_mag w))

theorem angle_between_vectors_45_degrees 
  (e1 e2 : ℝ × ℝ)
  (h1 : vec_mag e1 = 1)
  (h2 : vec_mag e2 = 1)
  (h3 : vec_dot e1 e2 = 0)
  (a : ℝ × ℝ := (3, 0) - (0, 1))  -- (3 * e1 - e2) is represented in a direct vector form (3, -1)
  (b : ℝ × ℝ := (2, 0) + (0, 1)): -- (2 * e1 + e2) is represented in a direct vector form (2, 1)
  vec_angle a b = π / 4 :=  -- π / 4 radians is equivalent to 45 degrees
sorry

end angle_between_vectors_45_degrees_l7_7431


namespace last_two_nonzero_digits_70_factorial_l7_7604

theorem last_two_nonzero_digits_70_factorial : 
  let N := 70
  (∀ N : ℕ, 0 < N → N % 2 ≠ 0 → N % 5 ≠ 0 → ∃ x : ℕ, x % 100 = N % (N + (N! / (2 ^ 16)))) →
  (N! / 10 ^ 16) % 100 = 68 :=
by
sorry

end last_two_nonzero_digits_70_factorial_l7_7604


namespace polynomial_without_xy_l7_7853

theorem polynomial_without_xy (k : ℝ) (x y : ℝ) :
  ¬(∃ c : ℝ, (x^2 + k * x * y + 4 * x - 2 * x * y + y^2 - 1 = c * x * y)) → k = 2 := by
  sorry

end polynomial_without_xy_l7_7853


namespace total_area_of_paths_l7_7957

theorem total_area_of_paths:
  let bed_width := 4
  let bed_height := 3
  let num_beds_width := 3
  let num_beds_height := 5
  let path_width := 2

  let total_bed_width := num_beds_width * bed_width
  let total_path_width := (num_beds_width + 1) * path_width
  let total_width := total_bed_width + total_path_width

  let total_bed_height := num_beds_height * bed_height
  let total_path_height := (num_beds_height + 1) * path_width
  let total_height := total_bed_height + total_path_height

  let total_area_greenhouse := total_width * total_height
  let total_area_beds := num_beds_width * num_beds_height * bed_width * bed_height

  let total_area_paths := total_area_greenhouse - total_area_beds

  total_area_paths = 360 :=
by sorry

end total_area_of_paths_l7_7957


namespace xy_product_of_sample_l7_7996

/-- Given a sample {9, 10, 11, x, y} such that the average is 10 and the standard deviation is sqrt(2), 
    prove that the product of x and y is 96. -/
theorem xy_product_of_sample (x y : ℝ) 
  (h_avg : (9 + 10 + 11 + x + y) / 5 = 10)
  (h_stddev : ( (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2 ) / 5 = 2) :
  x * y = 96 :=
by
  -- Proof goes here
  sorry

end xy_product_of_sample_l7_7996


namespace shaniqua_styles_count_l7_7310

variable (S : ℕ)

def shaniqua_haircuts (haircuts : ℕ) : ℕ := 12 * haircuts
def shaniqua_styles (styles : ℕ) : ℕ := 25 * styles

theorem shaniqua_styles_count (total_money haircuts : ℕ) (styles : ℕ) :
  total_money = shaniqua_haircuts haircuts + shaniqua_styles styles → haircuts = 8 → total_money = 221 → S = 5 :=
by
  sorry

end shaniqua_styles_count_l7_7310


namespace quadratic_function_symmetry_l7_7139

theorem quadratic_function_symmetry (a b x_1 x_2: ℝ) (h_roots: x_1^2 + a * x_1 + b = 0 ∧ x_2^2 + a * x_2 + b = 0)
(h_symmetry: ∀ x, (x - 2015)^2 + a * (x - 2015) + b = (x + 2015 - 2016)^2 + a * (x + 2015 - 2016) + b):
  (x_1 + x_2) / 2 = 2015 :=
sorry

end quadratic_function_symmetry_l7_7139


namespace sum_consecutive_integers_150_l7_7353

theorem sum_consecutive_integers_150 (n : ℕ) (a : ℕ) (hn : n ≥ 3) (hdiv : 300 % n = 0) :
  n * (2 * a + n - 1) = 300 ↔ (a > 0) → n = 3 ∨ n = 5 ∨ n = 15 :=
by sorry

end sum_consecutive_integers_150_l7_7353


namespace remainder_division_l7_7314

theorem remainder_division : ∃ (r : ℕ), 271 = 30 * 9 + r ∧ r = 1 :=
by
  -- Details of the proof would be filled here
  sorry

end remainder_division_l7_7314


namespace find_number_l7_7510

theorem find_number (number : ℤ) (h : number + 7 = 6) : number = -1 :=
by
  sorry

end find_number_l7_7510


namespace number_is_375_l7_7032

theorem number_is_375 (x : ℝ) (h : (40 / 100) * x = (30 / 100) * 50) : x = 37.5 :=
sorry

end number_is_375_l7_7032


namespace correct_ordering_l7_7655

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonicity (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 ≠ x2) : (x1 - x2) * (f x1 - f x2) > 0

theorem correct_ordering : f 1 < f (-2) ∧ f (-2) < f 3 :=
by sorry

end correct_ordering_l7_7655


namespace solve_x_l7_7953

theorem solve_x (x : ℝ) (h : x ≠ 0) (h_eq : (5 * x) ^ 10 = (10 * x) ^ 5) : x = 2 / 5 :=
by
  sorry

end solve_x_l7_7953


namespace base8_subtraction_correct_l7_7871

theorem base8_subtraction_correct : (453 - 326 : ℕ) = 125 :=
by sorry

end base8_subtraction_correct_l7_7871


namespace factorization_c_minus_d_l7_7207

theorem factorization_c_minus_d : 
  ∃ (c d : ℤ), (∀ (x : ℤ), (4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d))) ∧ (c - d = 8) :=
by
  sorry

end factorization_c_minus_d_l7_7207


namespace find_n_l7_7935

theorem find_n : ∃ n : ℕ, (∃ A B : ℕ, A ≠ B ∧ 10^(n-1) ≤ A ∧ A < 10^n ∧ 10^(n-1) ≤ B ∧ B < 10^n ∧ (10^n * A + B) % (10^n * B + A) = 0) ↔ n % 6 = 3 :=
by
  sorry

end find_n_l7_7935


namespace cube_as_difference_of_squares_l7_7134

theorem cube_as_difference_of_squares (a : ℕ) : 
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := 
by 
  -- The proof portion would go here, but since we only need the statement:
  sorry

end cube_as_difference_of_squares_l7_7134


namespace rate_of_interest_l7_7355

theorem rate_of_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h : P > 0 ∧ T = 7 ∧ SI = P / 5 ∧ SI = (P * R * T) / 100) : 
  R = 20 / 7 := 
by
  sorry

end rate_of_interest_l7_7355


namespace box_dimensions_sum_l7_7818

theorem box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30) 
  (h2 : A * C = 50)
  (h3 : B * C = 90) : 
  A + B + C = (58 * Real.sqrt 15) / 3 :=
sorry

end box_dimensions_sum_l7_7818


namespace cylinder_area_ratio_l7_7293

theorem cylinder_area_ratio (r h : ℝ) (h_eq : h = 2 * r * Real.sqrt π) :
  let S_lateral := 2 * π * r * h
  let S_total := S_lateral + 2 * π * r^2
  S_total / S_lateral = 1 + (1 / (2 * Real.sqrt π)) := by
sorry

end cylinder_area_ratio_l7_7293


namespace children_got_on_bus_l7_7062

theorem children_got_on_bus (initial_children total_children children_added : ℕ) 
  (h_initial : initial_children = 64) 
  (h_total : total_children = 78) : 
  children_added = total_children - initial_children :=
by
  sorry

end children_got_on_bus_l7_7062


namespace zeros_at_end_of_product1_value_of_product2_l7_7797

-- Definitions and conditions
def product1 := 360 * 5
def product2 := 250 * 4

-- Statements of the proof problems
theorem zeros_at_end_of_product1 : Nat.digits 10 product1 = [0, 0, 8, 1] := by
  sorry

theorem value_of_product2 : product2 = 1000 := by
  sorry

end zeros_at_end_of_product1_value_of_product2_l7_7797


namespace roots_imply_value_l7_7161

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l7_7161


namespace sum_geometric_series_l7_7845

noncomputable def S (r : ℝ) : ℝ :=
  12 / (1 - r)

theorem sum_geometric_series (a : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : S a * S (-a) = 2016) :
  S a + S (-a) = 336 :=
by
  sorry

end sum_geometric_series_l7_7845


namespace remainder_of_2n_div_10_l7_7335

theorem remainder_of_2n_div_10 (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_of_2n_div_10_l7_7335


namespace log24_eq_2b_minus_a_l7_7447

variable (a b : ℝ)

-- given conditions
axiom log6_eq : Real.log 6 = a
axiom log12_eq : Real.log 12 = b

-- proof goal statement
theorem log24_eq_2b_minus_a : Real.log 24 = 2 * b - a :=
by
  sorry

end log24_eq_2b_minus_a_l7_7447


namespace banker_l7_7254

theorem banker's_discount (BD TD FV : ℝ) (hBD : BD = 18) (hTD : TD = 15) 
(h : BD = TD + (TD^2 / FV)) : FV = 75 := by
  sorry

end banker_l7_7254


namespace plates_added_before_topple_l7_7042

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end plates_added_before_topple_l7_7042


namespace find_f_inv_64_l7_7748

noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't know the exact form of f.

axiom f_property_1 : f 5 = 2

axiom f_property_2 : ∀ x : ℝ, f (2 * x) = 2 * f x

def f_inv (y : ℝ) : ℝ :=
  sorry  -- We define the inverse function in terms of y.

theorem find_f_inv_64 : f_inv 64 = 160 :=
by {
  -- Main statement to be proved.
  sorry
}

end find_f_inv_64_l7_7748


namespace problem1_simplification_problem2_solve_fraction_l7_7959

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l7_7959


namespace salary_percentage_change_l7_7720

theorem salary_percentage_change (S : ℝ) (x : ℝ) :
  (S * (1 - (x / 100)) * (1 + (x / 100)) = S * 0.84) ↔ (x = 40) :=
by
  sorry

end salary_percentage_change_l7_7720


namespace range_of_a_l7_7867

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, x^2 - a * x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
by sorry

end range_of_a_l7_7867


namespace rhombus_diagonal_l7_7262

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) (h3 : area = (d1 * d2) / 2) : d2 = 18 := 
by
  -- h1, h2, and h3 are the conditions
  sorry

end rhombus_diagonal_l7_7262


namespace lemonade_calories_is_correct_l7_7409

def lemon_juice_content := 150
def sugar_content := 150
def water_content := 450

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def water_calories_per_100g := 0

def total_weight := lemon_juice_content + sugar_content + water_content
def caloric_density :=
  (lemon_juice_content * lemon_juice_calories_per_100g / 100) +
  (sugar_content * sugar_calories_per_100g / 100) +
  (water_content * water_calories_per_100g / 100)
def calories_per_gram := caloric_density / total_weight

def calories_in_300_grams := 300 * calories_per_gram

theorem lemonade_calories_is_correct : calories_in_300_grams = 258 := by
  sorry

end lemonade_calories_is_correct_l7_7409


namespace desired_yearly_income_l7_7494

theorem desired_yearly_income (total_investment : ℝ) 
  (investment1 : ℝ) (rate1 : ℝ) 
  (investment2 : ℝ) (rate2 : ℝ) 
  (rate_remainder : ℝ) 
  (h_total : total_investment = 10000) 
  (h_invest1 : investment1 = 4000)
  (h_rate1 : rate1 = 0.05) 
  (h_invest2 : investment2 = 3500)
  (h_rate2 : rate2 = 0.04)
  (h_rate_remainder : rate_remainder = 0.064)
  : (rate1 * investment1 + rate2 * investment2 + rate_remainder * (total_investment - (investment1 + investment2))) = 500 := 
by
  sorry

end desired_yearly_income_l7_7494


namespace sample_size_l7_7939

theorem sample_size (total_employees : ℕ) (male_employees : ℕ) (sampled_males : ℕ) (sample_size : ℕ) 
  (h1 : total_employees = 120) (h2 : male_employees = 80) (h3 : sampled_males = 24) : 
  sample_size = 36 :=
by
  sorry

end sample_size_l7_7939


namespace find_m_eq_zero_l7_7224

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l7_7224


namespace triangle_angles_correct_l7_7555

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
by sorry

theorem triangle_angles_correct :
  triangle_angles 3 (Real.sqrt 8) (2 + Real.sqrt 2) =
    (67.5, 22.5, 90) :=
by sorry

end triangle_angles_correct_l7_7555


namespace steve_first_stack_plastic_cups_l7_7170

theorem steve_first_stack_plastic_cups (cups_n : ℕ -> ℕ)
  (h_prop : ∀ n, cups_n (n + 1) = cups_n n + 4)
  (h_second : cups_n 2 = 21)
  (h_third : cups_n 3 = 25)
  (h_fourth : cups_n 4 = 29) :
  cups_n 1 = 17 :=
sorry

end steve_first_stack_plastic_cups_l7_7170


namespace find_b_l7_7251

def point := ℝ × ℝ

def dir_vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def scale_vector (v : point) (s : ℝ) : point := (s * v.1, s * v.2)

theorem find_b (p1 p2 : point) (b : ℝ) :
  p1 = (-5, 0) → p2 = (-2, 2) →
  dir_vector p1 p2 = (3, 2) →
  scale_vector (3, 2) (2 / 3) = (2, b) →
  b = 4 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_b_l7_7251


namespace intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l7_7755

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a ^ 2 + 1)) < 0 }

-- Theorem for question (1): Intersection of A and B when a = 2
theorem intersection_of_A_and_B_when_a_is_2 :
  setA 2 ∩ setB 2 = { x | 4 < x ∧ x < 5 } :=
sorry

-- Theorem for question (2): Range of a such that B ⊆ A
theorem range_of_a_such_that_B_subset_A :
  { a : ℝ | setB a ⊆ setA a } = { x | 1 < x ∧ x ≤ 3 } ∪ { -1 } :=
sorry

end intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l7_7755


namespace circle_area_is_162_pi_l7_7357

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

def R : ℝ × ℝ := (5, -2)
def S : ℝ × ℝ := (-4, 7)

theorem circle_area_is_162_pi :
  circle_area (distance R S) = 162 * Real.pi :=
by
  sorry

end circle_area_is_162_pi_l7_7357


namespace sub_one_inequality_l7_7487

theorem sub_one_inequality (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end sub_one_inequality_l7_7487


namespace find_first_offset_l7_7063

theorem find_first_offset
  (area : ℝ)
  (diagonal : ℝ)
  (offset2 : ℝ)
  (first_offset : ℝ)
  (h_area : area = 225)
  (h_diagonal : diagonal = 30)
  (h_offset2 : offset2 = 6)
  (h_formula : area = (diagonal * (first_offset + offset2)) / 2)
  : first_offset = 9 := by
  sorry

end find_first_offset_l7_7063


namespace value_of_mn_squared_l7_7336

theorem value_of_mn_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) : (m + n)^2 = 1 ∨ (m + n)^2 = 49 :=
by sorry

end value_of_mn_squared_l7_7336


namespace beadshop_wednesday_profit_l7_7887

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end beadshop_wednesday_profit_l7_7887


namespace division_proof_l7_7873

def dividend : ℕ := 144
def inner_divisor_num : ℕ := 12
def inner_divisor_denom : ℕ := 2
def final_divisor : ℕ := inner_divisor_num / inner_divisor_denom
def expected_result : ℕ := 24

theorem division_proof : (dividend / final_divisor) = expected_result := by
  sorry

end division_proof_l7_7873


namespace unique_solution_l7_7629

theorem unique_solution (x : ℝ) : 
  ∃! x, 2003^x + 2004^x = 2005^x := 
sorry

end unique_solution_l7_7629


namespace ratio_of_sums_l7_7765

/-- Define the relevant arithmetic sequences and sums -/

-- Sequence 1: 3, 6, 9, ..., 45
def seq1 : ℕ → ℕ
| n => 3 * n + 3

-- Sequence 2: 4, 8, 12, ..., 64
def seq2 : ℕ → ℕ
| n => 4 * n + 4

-- Sum function for arithmetic sequences
def sum_arith_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n-1) * d) / 2

noncomputable def sum_seq1 : ℕ := sum_arith_seq 3 3 15 -- 3 + 6 + ... + 45
noncomputable def sum_seq2 : ℕ := sum_arith_seq 4 4 16 -- 4 + 8 + ... + 64

-- Prove that the ratio of sums is 45/68
theorem ratio_of_sums : (sum_seq1 : ℚ) / sum_seq2 = 45 / 68 :=
  sorry

end ratio_of_sums_l7_7765


namespace largest_abs_val_among_2_3_neg3_neg4_l7_7874

def abs_val (a : Int) : Nat := a.natAbs

theorem largest_abs_val_among_2_3_neg3_neg4 : 
  ∀ (x : Int), x ∈ [2, 3, -3, -4] → abs_val x ≤ abs_val (-4) := by
  sorry

end largest_abs_val_among_2_3_neg3_neg4_l7_7874


namespace value_range_of_m_for_equation_l7_7417

theorem value_range_of_m_for_equation 
    (x : ℝ) 
    (cos_x : ℝ) 
    (h1: cos_x = Real.cos x) :
    ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 8) ∧ (4 * cos_x + Real.sin x ^ 2 + m - 4 = 0) := sorry

end value_range_of_m_for_equation_l7_7417


namespace monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l7_7704

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem monotonic_intervals_of_f :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂) ∧ (∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ ≥ f x₂) :=
sorry

theorem f_gt_x_ln_x_plus_1 (x : ℝ) (hx : x > 0) : f x > x * Real.log (x + 1) :=
sorry

end monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l7_7704


namespace expression_value_l7_7736

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l7_7736


namespace opposite_points_number_line_l7_7456

theorem opposite_points_number_line (a : ℤ) (h : a - 6 = -a) : a = 3 := by
  sorry

end opposite_points_number_line_l7_7456


namespace right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l7_7348

theorem right_triangle_min_hypotenuse (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) : c ≥ 4 * Real.sqrt 2 := by
  sorry

theorem right_triangle_min_hypotenuse_achieved (a b c : ℝ) (h_right : (a^2 + b^2 = c^2)) (h_perimeter : (a + b + c = 8)) (h_isosceles : a = b) : c = 4 * Real.sqrt 2 := by
  sorry

end right_triangle_min_hypotenuse_right_triangle_min_hypotenuse_achieved_l7_7348


namespace proof_theorem_l7_7859

noncomputable def proof_problem 
  (m n : ℕ) 
  (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : Prop :=
0 ≤ x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ∧ 
x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ≤ 1

theorem proof_theorem (m n : ℕ) (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : 
  proof_problem m n x y z h1 h2 h3 h4 h5 h6 h7 h8 h9 :=
by {
  sorry
}

end proof_theorem_l7_7859


namespace jordan_width_l7_7419

-- Definitions based on conditions
def area_of_carols_rectangle : ℝ := 15 * 20
def jordan_length_feet : ℝ := 6
def feet_to_inches (feet: ℝ) : ℝ := feet * 12
def jordan_length_inches : ℝ := feet_to_inches jordan_length_feet

-- Main statement
theorem jordan_width :
  ∃ w : ℝ, w = 300 / 72 :=
sorry

end jordan_width_l7_7419


namespace abs_neg_sub_three_eq_zero_l7_7343

theorem abs_neg_sub_three_eq_zero : |(-3 : ℤ)| - 3 = 0 :=
by sorry

end abs_neg_sub_three_eq_zero_l7_7343


namespace textbook_weight_ratio_l7_7721

def jon_textbooks_weights : List ℕ := [2, 8, 5, 9]
def brandon_textbooks_weight : ℕ := 8

theorem textbook_weight_ratio : 
  (jon_textbooks_weights.sum : ℚ) / (brandon_textbooks_weight : ℚ) = 3 :=
by 
  sorry

end textbook_weight_ratio_l7_7721


namespace int_values_satisfying_l7_7016

theorem int_values_satisfying (x : ℤ) : (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by
  sorry

end int_values_satisfying_l7_7016


namespace mrs_hilt_chapters_read_l7_7782

def number_of_books : ℝ := 4.0
def chapters_per_book : ℝ := 4.25
def total_chapters_read : ℝ := number_of_books * chapters_per_book

theorem mrs_hilt_chapters_read : total_chapters_read = 17 :=
by
  unfold total_chapters_read
  norm_num
  sorry

end mrs_hilt_chapters_read_l7_7782


namespace probability_queen_then_club_l7_7971

-- Define the problem conditions using the definitions
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_clubs : ℕ := 13
def num_club_queens : ℕ := 1

-- Define a function that computes the probability of the given event
def probability_first_queen_second_club : ℚ :=
  let prob_first_club_queen := (num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_club_queen := (num_clubs - 1 : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_1 := prob_first_club_queen * prob_second_club_given_first_club_queen
  let prob_first_non_club_queen := (num_queens - num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_non_club_queen := (num_clubs : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_2 := prob_first_non_club_queen * prob_second_club_given_first_non_club_queen
  prob_case_1 + prob_case_2

-- The statement to be proved
theorem probability_queen_then_club : probability_first_queen_second_club = 1 / 52 := by
  sorry

end probability_queen_then_club_l7_7971


namespace log_inequality_solution_l7_7495

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution_l7_7495


namespace solve_for_x_l7_7452

theorem solve_for_x (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 :=
by
  sorry

end solve_for_x_l7_7452


namespace parallel_vectors_eq_l7_7798

theorem parallel_vectors_eq (m : ℤ) (h : (m, 4) = (3 * k, -2 * k)) : m = -6 :=
by
  sorry

end parallel_vectors_eq_l7_7798


namespace solve_fractional_equation_l7_7407

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l7_7407


namespace find_n_l7_7727

theorem find_n (n : ℕ) 
    (h : 6 * 4 * 3 * n = Nat.factorial 8) : n = 560 := 
sorry

end find_n_l7_7727


namespace expression_evaluation_l7_7460

theorem expression_evaluation (a b : ℕ) (h1 : a = 25) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 750 :=
by
  sorry

end expression_evaluation_l7_7460


namespace geometric_sequence_sum_a_l7_7127

theorem geometric_sequence_sum_a (a : ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = 4^n + a) :
  a = -1 :=
sorry

end geometric_sequence_sum_a_l7_7127


namespace digit_A_unique_solution_l7_7917

theorem digit_A_unique_solution :
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧ (100 * A + 72 - 23 = 549) ∧ A = 5 :=
by
  sorry

end digit_A_unique_solution_l7_7917


namespace oldest_sibling_multiple_l7_7145

-- Definitions according to the conditions
def kay_age : Nat := 32
def youngest_sibling_age : Nat := kay_age / 2 - 5
def oldest_sibling_age : Nat := 44

-- The statement to prove
theorem oldest_sibling_multiple : oldest_sibling_age = 4 * youngest_sibling_age :=
by sorry

end oldest_sibling_multiple_l7_7145


namespace union_of_A_and_B_l7_7266

open Set -- to use set notation and operations

def A : Set ℝ := { x | -1/2 < x ∧ x < 2 }

def B : Set ℝ := { x | x^2 ≤ 1 }

theorem union_of_A_and_B :
  A ∪ B = Ico (-1:ℝ) 2 := 
by
  -- proof steps would go here, but we skip these with sorry.
  sorry

end union_of_A_and_B_l7_7266


namespace number_of_squares_is_five_l7_7253

-- A function that computes the number of squares obtained after the described operations on a piece of paper.
def folded_and_cut_number_of_squares (initial_shape : Type) (folds : ℕ) (cuts : ℕ) : ℕ :=
  -- sorry is used here as a placeholder for the actual implementation
  sorry

-- The main theorem stating that after two folds and two cuts, we obtain five square pieces.
theorem number_of_squares_is_five (initial_shape : Type) (h_initial_square : initial_shape = square)
  (h_folds : folds = 2) (h_cuts : cuts = 2) : folded_and_cut_number_of_squares initial_shape folds cuts = 5 :=
  sorry

end number_of_squares_is_five_l7_7253


namespace sum_of_two_numbers_l7_7535

theorem sum_of_two_numbers (x y : ℕ) (h1 : 3 * x = 180) (h2 : 4 * x = y) : x + y = 420 := by
  sorry

end sum_of_two_numbers_l7_7535


namespace sid_spent_on_computer_accessories_l7_7801

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l7_7801


namespace Lauren_total_revenue_l7_7434

noncomputable def LaurenMondayEarnings (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.80)

noncomputable def LaurenTuesdayEarningsEUR (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.75)

noncomputable def convertEURtoUSD (eur : ℝ) : ℝ :=
  eur * (1 / 0.85)

noncomputable def convertGBPtoUSD (gbp : ℝ) : ℝ :=
  gbp * 1.38

noncomputable def LaurenWeekendEarnings (sales : ℝ) : ℝ :=
  (sales * 0.10)

theorem Lauren_total_revenue :
  let monday_views := 80
  let monday_subscriptions := 20
  let tuesday_views := 100
  let tuesday_subscriptions := 27
  let weekend_sales := 100

  let monday_earnings := LaurenMondayEarnings monday_views monday_subscriptions
  let tuesday_earnings_eur := LaurenTuesdayEarningsEUR tuesday_views tuesday_subscriptions
  let tuesday_earnings_usd := convertEURtoUSD tuesday_earnings_eur
  let weekend_earnings_gbp := LaurenWeekendEarnings weekend_sales
  let weekend_earnings_usd := convertGBPtoUSD weekend_earnings_gbp

  monday_earnings + tuesday_earnings_usd + weekend_earnings_usd = 132.68 :=
by
  sorry

end Lauren_total_revenue_l7_7434


namespace apples_needed_for_two_weeks_l7_7394

theorem apples_needed_for_two_weeks :
  ∀ (apples_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ),
  apples_per_day = 1 → days_per_week = 7 → weeks = 2 →
  apples_per_day * days_per_week * weeks = 14 :=
by
  intros apples_per_day days_per_week weeks h1 h2 h3
  sorry

end apples_needed_for_two_weeks_l7_7394


namespace inequality_implies_strict_inequality_l7_7338

theorem inequality_implies_strict_inequality (x y z : ℝ) (h : x^2 + x * y + x * z < 0) : y^2 > 4 * x * z :=
sorry

end inequality_implies_strict_inequality_l7_7338


namespace opposite_of_neg2_is_2_l7_7790

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_neg2_is_2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_is_2_l7_7790


namespace find_a_f_odd_f_increasing_l7_7329

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x

theorem find_a : (f 1 a = 3) → (a = -1) :=
by
  sorry

noncomputable def f_1 (x : ℝ) : ℝ := 2 * x + 1 / x

theorem f_odd : ∀ x : ℝ, f_1 (-x) = -f_1 x :=
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, (x1 > 1) → (x2 > 1) → (x1 > x2) → (f_1 x1 > f_1 x2) :=
by
  sorry

end find_a_f_odd_f_increasing_l7_7329


namespace find_m_value_l7_7421

theorem find_m_value (m : ℝ) : (∃ A B : ℝ × ℝ, A = (-2, m) ∧ B = (m, 4) ∧ (∃ k : ℝ, k = (4 - m) / (m + 2) ∧ k = -2) ∧ (∃ l : ℝ, l = -2 ∧ 2 * l + l - 1 = 0)) → m = -8 :=
by
  sorry

end find_m_value_l7_7421


namespace summation_values_l7_7019

theorem summation_values (x y : ℝ) (h1 : x = y * (3 - y) ^ 2) (h2 : y = x * (3 - x) ^ 2) : 
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end summation_values_l7_7019


namespace miles_remaining_l7_7653

theorem miles_remaining (total_miles driven_miles : ℕ) (h1 : total_miles = 1200) (h2 : driven_miles = 768) :
    total_miles - driven_miles = 432 := by
  sorry

end miles_remaining_l7_7653


namespace log_product_eq_two_l7_7803

open Real

theorem log_product_eq_two
  : log 5 / log 3 * log 6 / log 5 * log 9 / log 6 = 2 := by
  sorry

end log_product_eq_two_l7_7803


namespace gym_cost_l7_7632

theorem gym_cost (x : ℕ) (hx : x > 0) (h1 : 50 + 12 * x + 48 * x = 650) : x = 10 :=
by
  sorry

end gym_cost_l7_7632


namespace trailing_zeros_in_square_l7_7569

-- Define x as given in the conditions
def x : ℕ := 10^12 - 4

-- State the theorem which asserts that the number of trailing zeros in x^2 is 11
theorem trailing_zeros_in_square : 
  ∃ n : ℕ, n = 11 ∧ x^2 % 10^12 = 0 :=
by
  -- Placeholder for the proof
  sorry

end trailing_zeros_in_square_l7_7569


namespace sweet_apples_percentage_is_75_l7_7879

noncomputable def percentage_sweet_apples 
  (price_sweet : ℝ) 
  (price_sour : ℝ) 
  (total_apples : ℕ) 
  (total_earnings : ℝ) 
  (percentage_sweet_expr : ℝ) :=
  price_sweet * percentage_sweet_expr + price_sour * (total_apples - percentage_sweet_expr) = total_earnings

theorem sweet_apples_percentage_is_75 :
  percentage_sweet_apples 0.5 0.1 100 40 75 :=
by
  unfold percentage_sweet_apples
  sorry

end sweet_apples_percentage_is_75_l7_7879


namespace solution_set_f_inequality_l7_7005

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then 1 - 2^(-x)
else if x < 0 then 2^x - 1
else 0

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem solution_set_f_inequality : 
  is_odd_function f →
  {x | f x < -1/2} = {x | x < -1} := 
by
  sorry

end solution_set_f_inequality_l7_7005


namespace round_robin_games_count_l7_7613

theorem round_robin_games_count (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end round_robin_games_count_l7_7613


namespace james_writes_pages_per_hour_l7_7279

theorem james_writes_pages_per_hour (hours_per_night : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) (total_hours : ℕ) :
  hours_per_night = 3 → 
  days_per_week = 7 → 
  weeks = 7 → 
  total_pages = 735 → 
  total_hours = 147 → 
  total_hours = hours_per_night * days_per_week * weeks → 
  total_pages / total_hours = 5 :=
by sorry

end james_writes_pages_per_hour_l7_7279


namespace initial_percentage_liquid_X_l7_7257

theorem initial_percentage_liquid_X (P : ℝ) :
  let original_solution_kg := 8
  let evaporated_water_kg := 2
  let added_solution_kg := 2
  let remaining_solution_kg := original_solution_kg - evaporated_water_kg
  let new_solution_kg := remaining_solution_kg + added_solution_kg
  let new_solution_percentage := 0.25
  let initial_liquid_X_kg := (P / 100) * original_solution_kg
  let final_liquid_X_kg := initial_liquid_X_kg + (P / 100) * added_solution_kg
  let final_liquid_X_kg' := new_solution_percentage * new_solution_kg
  (final_liquid_X_kg = final_liquid_X_kg') → 
  P = 20 :=
by
  intros
  let original_solution_kg_p0 := 8
  let evaporated_water_kg_p1 := 2
  let added_solution_kg_p2 := 2
  let remaining_solution_kg_p3 := (original_solution_kg_p0 - evaporated_water_kg_p1)
  let new_solution_kg_p4 := (remaining_solution_kg_p3 + added_solution_kg_p2)
  let new_solution_percentage : ℝ := 0.25
  let initial_liquid_X_kg_p6 := ((P / 100) * original_solution_kg_p0)
  let final_liquid_X_kg_p7 := initial_liquid_X_kg_p6 + ((P / 100) * added_solution_kg_p2)
  let final_liquid_X_kg_p8 := (new_solution_percentage * new_solution_kg_p4)
  exact sorry

end initial_percentage_liquid_X_l7_7257


namespace fraction_exponentiation_l7_7259

theorem fraction_exponentiation : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end fraction_exponentiation_l7_7259


namespace sequence_formula_l7_7483

theorem sequence_formula (a : ℕ → ℤ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = -3)
  (h₃ : a 3 = 5)
  (h₄ : a 4 = -7)
  (h₅ : a 5 = 9) :
  ∀ n : ℕ, a n = (-1)^(n+1) * (2 * n - 1) :=
by
  sorry

end sequence_formula_l7_7483


namespace solve_system_exists_l7_7489

theorem solve_system_exists (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : 1 / x + 1 / y + 1 / z = 5 / 12) 
  (h3 : x^3 + y^3 + z^3 = 45) 
  : (x, y, z) = (2, -3, 4) ∨ (x, y, z) = (2, 4, -3) ∨ (x, y, z) = (-3, 2, 4) ∨ (x, y, z) = (-3, 4, 2) ∨ (x, y, z) = (4, 2, -3) ∨ (x, y, z) = (4, -3, 2) := 
sorry

end solve_system_exists_l7_7489


namespace greatest_possible_integer_l7_7712

theorem greatest_possible_integer (n k l : ℕ) (h1 : n < 150) (h2 : n = 11 * k - 1) (h3 : n = 9 * l + 2) : n = 65 :=
by sorry

end greatest_possible_integer_l7_7712


namespace words_per_page_l7_7771

theorem words_per_page (p : ℕ) (hp : p ≤ 120) (h : 150 * p ≡ 210 [MOD 221]) : p = 98 := by
  sorry

end words_per_page_l7_7771


namespace find_interest_rate_l7_7210

theorem find_interest_rate
  (P : ℝ) (t : ℕ) (I : ℝ)
  (hP : P = 3000)
  (ht : t = 5)
  (hI : I = 750) :
  ∃ r : ℝ, I = P * r * t / 100 ∧ r = 5 :=
by 
  sorry

end find_interest_rate_l7_7210


namespace sally_initial_poems_l7_7229

theorem sally_initial_poems (recited: ℕ) (forgotten: ℕ) (h1 : recited = 3) (h2 : forgotten = 5) : 
  recited + forgotten = 8 := 
by
  sorry

end sally_initial_poems_l7_7229


namespace angle_ABC_is_45_l7_7640

theorem angle_ABC_is_45
  (x : ℝ)
  (h1 : ∀ (ABC : ℝ), x = 180 - ABC → x = 45) :
  2 * (x / 2) = (180 - x) / 6 → x = 45 :=
by
  sorry

end angle_ABC_is_45_l7_7640


namespace log7_18_l7_7401

theorem log7_18 (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) :=
by
  -- proof to be completed
  sorry

end log7_18_l7_7401


namespace integer_values_of_b_for_polynomial_root_l7_7362

theorem integer_values_of_b_for_polynomial_root
    (b : ℤ) :
    (∃ x : ℤ, x^3 + 6 * x^2 + b * x + 12 = 0) ↔
    b = -217 ∨ b = -74 ∨ b = -43 ∨ b = -31 ∨ b = -22 ∨ b = -19 ∨
    b = 19 ∨ b = 22 ∨ b = 31 ∨ b = 43 ∨ b = 74 ∨ b = 217 :=
    sorry

end integer_values_of_b_for_polynomial_root_l7_7362


namespace speed_ratio_l7_7756

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l7_7756


namespace first_dog_walks_two_miles_per_day_l7_7379

variable (x : ℝ)

theorem first_dog_walks_two_miles_per_day  
  (h1 : 7 * x + 56 = 70) : 
  x = 2 := 
by 
  sorry

end first_dog_walks_two_miles_per_day_l7_7379


namespace kristi_books_proof_l7_7558

variable (Bobby_books Kristi_books : ℕ)

def condition1 : Prop := Bobby_books = 142

def condition2 : Prop := Bobby_books = Kristi_books + 64

theorem kristi_books_proof (h1 : condition1 Bobby_books) (h2 : condition2 Bobby_books Kristi_books) : Kristi_books = 78 := 
by 
  sorry

end kristi_books_proof_l7_7558


namespace polynomial_evaluation_l7_7749

theorem polynomial_evaluation (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2 * a^2 + 2 = 3 :=
by
  sorry

end polynomial_evaluation_l7_7749


namespace length_real_axis_hyperbola_l7_7834

theorem length_real_axis_hyperbola (a : ℝ) (h : a^2 = 4) : 2 * a = 4 := by
  sorry

end length_real_axis_hyperbola_l7_7834


namespace mabel_visits_helen_l7_7517

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l7_7517


namespace inverse_proportion_neg_k_l7_7794

theorem inverse_proportion_neg_k (x1 x2 y1 y2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 > y2) :
  ∃ k : ℝ, k < 0 ∧ (∀ x, (x = x1 → y1 = k / x) ∧ (x = x2 → y2 = k / x)) := by
  use -1
  sorry

end inverse_proportion_neg_k_l7_7794


namespace line_passes_through_fixed_point_l7_7047

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), y = k * x - k ∧ x = 1 ∧ y = 0 :=
by
  use 1
  use 0
  sorry

end line_passes_through_fixed_point_l7_7047


namespace min_value_abs_diff_l7_7870

-- Definitions of conditions
def is_in_interval (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 4

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  (b^2 - a^2 = 2) ∧ (c^2 - b^2 = 2)

-- Main statement
theorem min_value_abs_diff (x y z : ℝ)
  (h1 : is_in_interval x)
  (h2 : is_in_interval y)
  (h3 : is_in_interval z)
  (h4 : is_arithmetic_progression x y z) :
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_value_abs_diff_l7_7870


namespace customers_left_proof_l7_7106

def initial_customers : ℕ := 21
def tables : ℕ := 3
def people_per_table : ℕ := 3
def remaining_customers : ℕ := tables * people_per_table
def customers_left (initial remaining : ℕ) : ℕ := initial - remaining

theorem customers_left_proof : customers_left initial_customers remaining_customers = 12 := sorry

end customers_left_proof_l7_7106


namespace emily_final_lives_l7_7273

/-- Initial number of lives Emily had. --/
def initialLives : ℕ := 42

/-- Number of lives Emily lost in the hard part of the game. --/
def livesLost : ℕ := 25

/-- Number of lives Emily gained in the next level. --/
def livesGained : ℕ := 24

/-- Final number of lives Emily should have after the changes. --/
def finalLives : ℕ := (initialLives - livesLost) + livesGained

theorem emily_final_lives : finalLives = 41 := by
  /-
  Proof is omitted as per instructions.
  Prove that the final number of lives Emily has is 41.
  -/
  sorry

end emily_final_lives_l7_7273


namespace time_to_fill_bucket_l7_7956

theorem time_to_fill_bucket (t : ℝ) (h : 2/3 = 2 / t) : t = 3 :=
by
  sorry

end time_to_fill_bucket_l7_7956


namespace number_of_dogs_on_tuesday_l7_7554

variable (T : ℕ)
variable (H1 : 7 + T + 7 + 7 + 9 = 42)

theorem number_of_dogs_on_tuesday : T = 12 := by
  sorry

end number_of_dogs_on_tuesday_l7_7554


namespace find_pair_l7_7241

theorem find_pair :
  ∃ x y : ℕ, (1984 * x - 1983 * y = 1985) ∧ (x = 27764) ∧ (y = 27777) :=
by
  sorry

end find_pair_l7_7241


namespace rhombus_area_l7_7542

-- Define d1 and d2 as the lengths of the diagonals
def d1 : ℝ := 15
def d2 : ℝ := 17

-- The theorem to prove the area of the rhombus
theorem rhombus_area : (d1 * d2) / 2 = 127.5 := by
  sorry

end rhombus_area_l7_7542


namespace integer_solution_unique_l7_7037

theorem integer_solution_unique (n : ℤ) : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 3) ↔ n = 5 :=
by
  sorry

end integer_solution_unique_l7_7037


namespace no_such_polyhedron_l7_7206

theorem no_such_polyhedron (n : ℕ) (S : Fin n → ℝ) (H : ∀ i j : Fin n, i ≠ j → S i ≥ 2 * S j) : False :=
by
  sorry

end no_such_polyhedron_l7_7206


namespace c_in_terms_of_t_l7_7002

theorem c_in_terms_of_t (t a b c : ℝ) (h_t_ne_zero : t ≠ 0)
    (h1 : t^3 + a * t = 0)
    (h2 : b * t^2 + c = 0)
    (h3 : 3 * t^2 + a = 2 * b * t) :
    c = -t^3 :=
by
sorry

end c_in_terms_of_t_l7_7002


namespace range_mn_squared_l7_7172

-- Let's define the conditions in Lean

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is strictly increasing
axiom h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x-1) is centrally symmetric about (1,0)
axiom h2 : ∀ x : ℝ, f (x - 1) = - f (2 - (x - 1))

-- Condition 3: Given inequality
axiom h3 : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0

-- Prove the range for m^2 + n^2 is (9, 49)
theorem range_mn_squared : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0 →
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_mn_squared_l7_7172


namespace min_value_of_3a_plus_2_l7_7938

theorem min_value_of_3a_plus_2 
  (a : ℝ) 
  (h : 4 * a^2 + 7 * a + 3 = 2)
  : 3 * a + 2 >= -1 :=
sorry

end min_value_of_3a_plus_2_l7_7938


namespace max_k_no_real_roots_l7_7304

theorem max_k_no_real_roots : ∀ k : ℤ, (∀ x : ℝ, x^2 - 2 * x - (k : ℝ) ≠ 0) → k ≤ -2 :=
by
  sorry

end max_k_no_real_roots_l7_7304


namespace cost_price_600_l7_7835

variable (CP SP : ℝ)

theorem cost_price_600 
  (h1 : SP = 1.08 * CP) 
  (h2 : SP = 648) : 
  CP = 600 := 
by
  sorry

end cost_price_600_l7_7835


namespace range_x_f_inequality_l7_7083

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem range_x_f_inequality :
  (∀ x : ℝ, f (2 * x + 1) ≥ f x) ↔ x ∈ Set.Icc (-1 : ℝ) (-1 / 3) := sorry

end range_x_f_inequality_l7_7083


namespace point_in_fourth_quadrant_l7_7035

theorem point_in_fourth_quadrant (m n : ℝ) (h₁ : m < 0) (h₂ : n > 0) : 
  2 * n - m > 0 ∧ -n + m < 0 := by
  sorry

end point_in_fourth_quadrant_l7_7035


namespace girls_not_playing_soccer_l7_7645

-- Define the given conditions
def students_total : Nat := 420
def boys_total : Nat := 312
def soccer_players_total : Nat := 250
def percent_boys_playing_soccer : Float := 0.78

-- Define the main goal based on the question and correct answer
theorem girls_not_playing_soccer : 
  students_total = 420 → 
  boys_total = 312 → 
  soccer_players_total = 250 → 
  percent_boys_playing_soccer = 0.78 → 
  ∃ (girls_not_playing_soccer : Nat), girls_not_playing_soccer = 53 :=
by 
  sorry

end girls_not_playing_soccer_l7_7645


namespace coconut_grove_yield_l7_7983

theorem coconut_grove_yield (x : ℕ)
  (h1 : ∀ y, y = x + 3 → 60 * y = 60 * (x + 3))
  (h2 : ∀ z, z = x → 120 * z = 120 * x)
  (h3 : ∀ w, w = x - 3 → 180 * w = 180 * (x - 3))
  (avg_yield : 100 = 100)
  (total_trees : 3 * x = (x + 3) + x + (x - 3)) :
  60 * (x + 3) + 120 * x + 180 * (x - 3) = 300 * x →
  x = 6 :=
by
  sorry

end coconut_grove_yield_l7_7983


namespace find_p_at_8_l7_7673

noncomputable def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

noncomputable def p (x : ℝ) : ℝ :=
  let a := sorry ; -- root 1 of h
  let b := sorry ; -- root 2 of h
  let c := sorry ; -- root 3 of h
  let B := 2 / ((1 - a^3) * (1 - b^3) * (1 - c^3))
  B * (x - a^3) * (x - b^3) * (x - c^3)

theorem find_p_at_8 : p 8 = 1008 := sorry

end find_p_at_8_l7_7673


namespace suff_not_nec_condition_l7_7280

/-- f is an even function --/
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Condition x1 + x2 = 0 --/
def sum_eq_zero (x1 x2 : ℝ) : Prop := x1 + x2 = 0

/-- Prove: sufficient but not necessary condition --/
theorem suff_not_nec_condition (f : ℝ → ℝ) (h_even : is_even f) (x1 x2 : ℝ) :
  sum_eq_zero x1 x2 → f x1 - f x2 = 0 ∧ (f x1 - f x2 = 0 → ¬ sum_eq_zero x1 x2) :=
by
  sorry

end suff_not_nec_condition_l7_7280


namespace race_head_start_l7_7432

theorem race_head_start (Va Vb L H : ℚ) (h : Va = 30 / 17 * Vb) :
  H = 13 / 30 * L :=
by
  sorry

end race_head_start_l7_7432


namespace parametric_curve_to_general_form_l7_7898

theorem parametric_curve_to_general_form :
  ∃ (a b c : ℚ), ∀ (t : ℝ), 
  (a = 8 / 225) ∧ (b = 4 / 75) ∧ (c = 1 / 25) ∧ 
  (a * (3 * Real.sin t)^2 + b * (3 * Real.sin t) * (5 * Real.cos t - 2 * Real.sin t) + c * (5 * Real.cos t - 2 * Real.sin t)^2 = 1) :=
by
  use 8 / 225, 4 / 75, 1 / 25
  sorry

end parametric_curve_to_general_form_l7_7898


namespace parabola_points_relation_l7_7160

theorem parabola_points_relation (c y1 y2 y3 : ℝ)
  (h1 : y1 = -(-2)^2 - 2*(-2) + c)
  (h2 : y2 = -(0)^2 - 2*(0) + c)
  (h3 : y3 = -(1)^2 - 2*(1) + c) :
  y1 = y2 ∧ y2 > y3 :=
by
  sorry

end parabola_points_relation_l7_7160


namespace sum_in_base7_l7_7913

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end sum_in_base7_l7_7913


namespace correct_forecast_interpretation_l7_7950

/-- The probability of precipitation in the area tomorrow is 80%. -/
def prob_precipitation_tomorrow : ℝ := 0.8

/-- Multiple choice options regarding the interpretation of the probability of precipitation. -/
inductive forecast_interpretation
| A : forecast_interpretation
| B : forecast_interpretation
| C : forecast_interpretation
| D : forecast_interpretation

/-- The correct interpretation is Option C: "There is an 80% chance of rain in the area tomorrow." -/
def correct_interpretation : forecast_interpretation :=
forecast_interpretation.C

theorem correct_forecast_interpretation :
  (prob_precipitation_tomorrow = 0.8) → (correct_interpretation = forecast_interpretation.C) :=
by
  sorry

end correct_forecast_interpretation_l7_7950


namespace find_m_l7_7872

def vector_collinear {α : Type*} [Field α] (a b : α × α) : Prop :=
  ∃ k : α, b = (k * (a.1), k * (a.2))

theorem find_m (m : ℝ) : 
  let a := (2, 3)
  let b := (-1, 2)
  vector_collinear (2 * m - 4, 3 * m + 8) (4, -1) → m = -2 :=
by
  intros
  sorry

end find_m_l7_7872


namespace rate_of_discount_l7_7631

theorem rate_of_discount (Marked_Price Selling_Price : ℝ) (h_marked : Marked_Price = 80) (h_selling : Selling_Price = 68) : 
  ((Marked_Price - Selling_Price) / Marked_Price) * 100 = 15 :=
by
  -- Definitions from conditions
  rw [h_marked, h_selling]
  -- Substitute the values and simplify
  sorry

end rate_of_discount_l7_7631


namespace value_of_expression_l7_7364

theorem value_of_expression (a b : ℤ) (h : 2 * a - b = 10) : 2023 - 2 * a + b = 2013 :=
by
  sorry

end value_of_expression_l7_7364


namespace series_sum_solution_l7_7836

noncomputable def series_sum (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) : ℝ :=
  ∑' n : ℕ, (1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b)))

theorem series_sum_solution (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) :
  series_sum a b c h₀ h₁ h₂ h₃ h₄ = 1 / ((c - b) * c) := 
  sorry

end series_sum_solution_l7_7836


namespace age_of_b_l7_7216

-- Definition of conditions
variable (a b c : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : a + b + c = 12)

-- The statement of the proof problem
theorem age_of_b : b = 4 :=
by {
   sorry
}

end age_of_b_l7_7216


namespace trig_identity_l7_7236

theorem trig_identity :
  (Real.sin (17 * Real.pi / 180) * Real.cos (47 * Real.pi / 180) - 
   Real.sin (73 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = -1/2 := 
by
  sorry

end trig_identity_l7_7236


namespace emma_withdrew_amount_l7_7530

variable (W : ℝ) -- Variable representing the amount Emma withdrew

theorem emma_withdrew_amount:
  (230 - W + 2 * W = 290) →
  W = 60 :=
by
  sorry

end emma_withdrew_amount_l7_7530


namespace additional_savings_in_cents_l7_7961

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents_l7_7961


namespace find_m_l7_7559

theorem find_m (m a : ℝ) (h : (2:ℝ) * 1^2 - 3 * 1 + a = 0) 
  (h_roots : ∀ x : ℝ, 2 * x^2 - 3 * x + a = 0 → (x = 1 ∨ x = m)) :
  m = 1 / 2 :=
by
  sorry

end find_m_l7_7559


namespace sum_x_y_eq_l7_7025

noncomputable def equation (x y : ℝ) : Prop :=
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0

theorem sum_x_y_eq (x y : ℝ) (h : equation x y) : x + y = -9 / 2 :=
by sorry

end sum_x_y_eq_l7_7025


namespace inequality_proof_l7_7528

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l7_7528


namespace girls_more_than_boys_l7_7734

theorem girls_more_than_boys (boys girls : ℕ) (ratio_boys ratio_girls : ℕ) 
  (h1 : ratio_boys = 5)
  (h2 : ratio_girls = 13)
  (h3 : boys = 50)
  (h4 : girls = (boys / ratio_boys) * ratio_girls) : 
  girls - boys = 80 :=
by
  sorry

end girls_more_than_boys_l7_7734


namespace num_possible_n_l7_7124

theorem num_possible_n (n : ℕ) : (∃ a b c : ℕ, 9 * a + 99 * b + 999 * c = 5000 ∧ n = a + 2 * b + 3 * c) ↔ n ∈ {x | x = a + 2 * b + 3 * c ∧ 0 ≤ 9 * (b + 12 * c) ∧ 9 * (b + 12 * c) ≤ 555} :=
sorry

end num_possible_n_l7_7124


namespace find_roots_of_polynomial_l7_7520

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l7_7520


namespace simplify_expression_l7_7540

variable (a b c d : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d)

theorem simplify_expression :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
  sorry

end simplify_expression_l7_7540


namespace range_of_a_l7_7609

variable (a : ℝ)

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) : 1 < a :=
by {
  -- Proof will go here.
  sorry
}

end range_of_a_l7_7609


namespace k_range_l7_7886

noncomputable def range_of_k (k : ℝ): Prop :=
  ∀ x : ℤ, (x - 2) * (x + 1) > 0 ∧ (2 * x + 5) * (x + k) < 0 → x = -2

theorem k_range:
  (∃ k : ℝ, range_of_k k) ↔ -3 ≤ k ∧ k < 2 :=
by
  sorry

end k_range_l7_7886


namespace sequence_neither_arithmetic_nor_geometric_l7_7549

noncomputable def Sn (n : ℕ) : ℕ := 3 * n + 2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 5 else Sn n - Sn (n - 1)

def not_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ d, ∀ n, a (n + 1) = a n + d

def not_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

theorem sequence_neither_arithmetic_nor_geometric :
  not_arithmetic_sequence a ∧ not_geometric_sequence a :=
sorry

end sequence_neither_arithmetic_nor_geometric_l7_7549


namespace solve_for_x_l7_7306

variables {x y : ℝ}

theorem solve_for_x (h : x / (x - 3) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 4)) : 
  x = (3 * y^2 + 9 * y + 3) / 5 :=
sorry

end solve_for_x_l7_7306


namespace sin_300_eq_neg_sqrt_3_div_2_l7_7899

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l7_7899


namespace frank_bags_on_saturday_l7_7552

def bags_filled_on_saturday (total_cans : Nat) (cans_per_bag : Nat) (bags_on_sunday : Nat) : Nat :=
  total_cans / cans_per_bag - bags_on_sunday

theorem frank_bags_on_saturday : 
  let total_cans := 40
  let cans_per_bag := 5
  let bags_on_sunday := 3
  bags_filled_on_saturday total_cans cans_per_bag bags_on_sunday = 5 :=
  by
  -- Proof to be provided
  sorry

end frank_bags_on_saturday_l7_7552


namespace geometric_sequence_increasing_iff_q_gt_one_l7_7503

variables {a_n : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n (n + 1) > a_n n

theorem geometric_sequence_increasing_iff_q_gt_one 
  (h1 : ∀ n, 0 < a_n n)
  (h2 : is_geometric_sequence a_n q) :
  is_increasing_sequence a_n ↔ q > 1 :=
by
  sorry

end geometric_sequence_increasing_iff_q_gt_one_l7_7503


namespace line_intersection_equation_of_l4_find_a_l7_7623

theorem line_intersection (P : ℝ × ℝ)
    (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) :
  P = (-2, 2) :=
sorry

theorem equation_of_l4 (l4 : ℝ → ℝ → Prop)
    (P : ℝ × ℝ) (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) 
    (h_parallel: ∀ x y, l4 x y ↔ y = 1/2 * x + 3)
    (x y : ℝ) :
  l4 x y ↔ y = 1/2 * x + 3 :=
sorry

theorem find_a (a : ℝ) :
    (∀ x y, 2 * x + y + 2 = 0 → y = -2 * x - 2) →
    (∀ x y, a * x - 2 * y + 1 = 0 → y = 1/2 * x - 1/2) →
    a = 1 :=
sorry

end line_intersection_equation_of_l4_find_a_l7_7623


namespace simplify_expression_l7_7706

variable (y : ℝ)

theorem simplify_expression : (5 * y + 6 * y + 7 * y + 2) = (18 * y + 2) := 
by
  sorry

end simplify_expression_l7_7706


namespace circle_standard_equation_l7_7093

theorem circle_standard_equation (x y : ℝ) (h : (x + 1)^2 + (y - 2)^2 = 4) : 
  (x + 1)^2 + (y - 2)^2 = 4 :=
sorry

end circle_standard_equation_l7_7093


namespace no_such_function_exists_l7_7513

theorem no_such_function_exists :
  ¬ (∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l7_7513


namespace problem_statement_l7_7325

theorem problem_statement : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end problem_statement_l7_7325


namespace average_rate_of_change_l7_7162

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end average_rate_of_change_l7_7162


namespace Jake_has_62_balls_l7_7303

theorem Jake_has_62_balls 
  (C A J : ℕ)
  (h1 : C = 41 + 7)
  (h2 : A = 2 * C)
  (h3 : J = A - 34) : 
  J = 62 :=
by 
  sorry

end Jake_has_62_balls_l7_7303


namespace laura_change_l7_7605

theorem laura_change : 
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250
  (amount_given - total_cost) = 10 :=
by
  -- definitions from conditions
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250

  -- the statement we are proving
  show (amount_given - total_cost) = 10
  sorry

end laura_change_l7_7605


namespace solution_set_of_inequality_l7_7026

theorem solution_set_of_inequality :
  ∀ (x : ℝ), abs (2 * x + 1) < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_of_inequality_l7_7026


namespace factor_quadratic_l7_7674

theorem factor_quadratic (m p : ℝ) (h : (m - 8) ∣ (m^2 - p * m - 24)) : p = 5 :=
sorry

end factor_quadratic_l7_7674


namespace sphere_volume_l7_7006

theorem sphere_volume (S : ℝ) (hS : S = 4 * π) : ∃ V : ℝ, V = (4 / 3) * π := 
by
  sorry

end sphere_volume_l7_7006


namespace nova_monthly_donation_l7_7785

def total_annual_donation : ℕ := 20484
def months_in_year : ℕ := 12
def monthly_donation : ℕ := total_annual_donation / months_in_year

theorem nova_monthly_donation :
  monthly_donation = 1707 :=
by
  unfold monthly_donation
  sorry

end nova_monthly_donation_l7_7785


namespace income_expenditure_ratio_l7_7614

theorem income_expenditure_ratio
  (I E : ℕ)
  (h1 : I = 18000)
  (S : ℕ)
  (h2 : S = 2000)
  (h3 : S = I - E) :
  I.gcd E = 2000 ∧ I / I.gcd E = 9 ∧ E / I.gcd E = 8 :=
by sorry

end income_expenditure_ratio_l7_7614


namespace evaluate_expression_l7_7817

-- Defining the conditions for the cosine and sine values
def cos_0 : Real := 1
def sin_3pi_2 : Real := -1

-- Proving the given expression equals -1
theorem evaluate_expression : 3 * cos_0 + 4 * sin_3pi_2 = -1 :=
by 
  -- Given the definitions, this will simplify as expected.
  sorry

end evaluate_expression_l7_7817


namespace car_speed_ratio_l7_7746

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l7_7746


namespace average_speed_round_trip_l7_7731

theorem average_speed_round_trip (v1 v2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 100) :
  (2 * v1 * v2) / (v1 + v2) = 75 :=
by
  sorry

end average_speed_round_trip_l7_7731


namespace part1_l7_7601

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x, 0 ≤ x → f x = Real.sqrt x)
variable (h1 : 0 ≤ x1)
variable (h2 : 0 ≤ x2)
variable (h3 : x1 ≠ x2)

theorem part1 : (1/2) * (f x1 + f x2) < f ((x1 + x2) / 2) :=
  sorry

end part1_l7_7601


namespace semi_circle_radius_l7_7926

theorem semi_circle_radius (P : ℝ) (r : ℝ) (π : ℝ) (h_perimeter : P = 113) (h_pi : π = Real.pi) :
  r = P / (π + 2) :=
sorry

end semi_circle_radius_l7_7926


namespace four_digit_even_and_multiple_of_7_sum_l7_7479

def num_four_digit_even_numbers : ℕ := 4500
def num_four_digit_multiples_of_7 : ℕ := 1286
def C : ℕ := num_four_digit_even_numbers
def D : ℕ := num_four_digit_multiples_of_7

theorem four_digit_even_and_multiple_of_7_sum :
  C + D = 5786 := by
  sorry

end four_digit_even_and_multiple_of_7_sum_l7_7479


namespace prove_students_second_and_third_l7_7267

namespace MonicaClasses

def Monica := 
  let classes_per_day := 6
  let students_first_class := 20
  let students_fourth_class := students_first_class / 2
  let students_fifth_class := 28
  let students_sixth_class := 28
  let total_students := 136
  let known_students := students_first_class + students_fourth_class + students_fifth_class + students_sixth_class
  let students_second_and_third := total_students - known_students
  students_second_and_third = 50

theorem prove_students_second_and_third : Monica :=
  by
    sorry

end MonicaClasses

end prove_students_second_and_third_l7_7267


namespace remaining_wire_length_l7_7457

theorem remaining_wire_length (total_wire_length : ℝ) (square_side_length : ℝ) 
  (h₀ : total_wire_length = 60) (h₁ : square_side_length = 9) : 
  total_wire_length - 4 * square_side_length = 24 :=
by
  sorry

end remaining_wire_length_l7_7457


namespace algebraic_expression_value_l7_7581

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 1) :
  (2 * x + 4 * y) / (x^2 + 4 * x * y + 4 * y^2) = 2 :=
by
  sorry

end algebraic_expression_value_l7_7581


namespace Yoojung_total_vehicles_l7_7574

theorem Yoojung_total_vehicles : 
  let motorcycles := 2
  let bicycles := 5
  motorcycles + bicycles = 7 := 
by
  sorry

end Yoojung_total_vehicles_l7_7574


namespace average_of_remaining_two_numbers_l7_7676

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℚ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) : 
  ((e + f) / 2 = 6.9) :=
by
  sorry

end average_of_remaining_two_numbers_l7_7676


namespace probability_three_black_balls_probability_white_ball_l7_7703

-- Definitions representing conditions
def total_ratio (A B C : ℕ) := A / B = 5 / 4 ∧ B / C = 4 / 6

-- Proportions of black balls in each box
def proportion_black_A (black_A total_A : ℕ) := black_A = 40 * total_A / 100
def proportion_black_B (black_B total_B : ℕ) := black_B = 25 * total_B / 100
def proportion_black_C (black_C total_C : ℕ) := black_C = 50 * total_C / 100

-- Problem 1: Probability of selecting a black ball from each box
theorem probability_three_black_balls
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C) :
  (black_A / total_A) * (black_B / total_B) * (black_C / total_C) = 1 / 20 :=
  sorry

-- Problem 2: Probability of selecting a white ball from the mixed total
theorem probability_white_ball
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (white_A white_B white_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C)
  (h5 : white_A = total_A - black_A)
  (h6 : white_B = total_B - black_B)
  (h7 : white_C = total_C - black_C) :
  (white_A + white_B + white_C) / (total_A + total_B + total_C) = 3 / 5 :=
  sorry

end probability_three_black_balls_probability_white_ball_l7_7703


namespace remove_five_magazines_l7_7897

theorem remove_five_magazines (magazines : Fin 10 → Set α) 
  (coffee_table : Set α) 
  (h_cover : (⋃ i, magazines i) = coffee_table) :
  ∃ ( S : Set α), S ⊆ coffee_table ∧ (∃ (removed : Finset (Fin 10)), removed.card = 5 ∧ 
    coffee_table \ (⋃ i ∈ removed, magazines i) ⊆ S ∧ (S = coffee_table \ (⋃ i ∈ removed, magazines i) ) ∧ 
    (⋃ i ∉ removed, magazines i) ∩ S = ∅) := 
sorry

end remove_five_magazines_l7_7897


namespace xiaolin_final_score_l7_7511

-- Define the conditions
def score_situps : ℕ := 80
def score_800m : ℕ := 90
def weight_situps : ℕ := 4
def weight_800m : ℕ := 6

-- Define the final score based on the given conditions
def final_score : ℕ :=
  (score_situps * weight_situps + score_800m * weight_800m) / (weight_situps + weight_800m)

-- Prove that the final score is 86
theorem xiaolin_final_score : final_score = 86 :=
by sorry

end xiaolin_final_score_l7_7511


namespace find_largest_number_l7_7150

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
  sorry

end find_largest_number_l7_7150


namespace product_of_consecutive_integers_sqrt_50_l7_7573

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l7_7573


namespace fraction_of_students_who_walk_l7_7929

def fraction_by_bus : ℚ := 2 / 5
def fraction_by_car : ℚ := 1 / 5
def fraction_by_scooter : ℚ := 1 / 8
def total_fraction_not_walk := fraction_by_bus + fraction_by_car + fraction_by_scooter

theorem fraction_of_students_who_walk :
  (1 - total_fraction_not_walk) = 11 / 40 :=
by
  sorry

end fraction_of_students_who_walk_l7_7929


namespace maximize_magnitude_l7_7739

theorem maximize_magnitude (a x y : ℝ) 
(h1 : 4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32)
(h2 : 2 * x * y = a) : a = 8 := 
sorry

end maximize_magnitude_l7_7739


namespace percentage_of_adult_men_l7_7444

theorem percentage_of_adult_men (total_members : ℕ) (children : ℕ) (p : ℕ) :
  total_members = 2000 → children = 200 → 
  (∀ adult_men_percentage : ℕ, adult_women_percentage = 2 * adult_men_percentage) → 
  (100 - p) = 3 * (p - 10) →  p = 30 :=
by sorry

end percentage_of_adult_men_l7_7444


namespace lateral_surface_area_of_frustum_l7_7532

theorem lateral_surface_area_of_frustum (slant_height : ℝ) (ratio : ℕ × ℕ) (central_angle_deg : ℝ)
  (h_slant_height : slant_height = 10) 
  (h_ratio : ratio = (2, 5)) 
  (h_central_angle_deg : central_angle_deg = 216) : 
  ∃ (area : ℝ), area = (252 * Real.pi / 5) := 
by 
  sorry

end lateral_surface_area_of_frustum_l7_7532


namespace value_of_m_l7_7413

noncomputable def TV_sales_volume_function (x : ℕ) : ℚ :=
  10 * x + 540

theorem value_of_m : ∀ (m : ℚ),
  (3200 * (1 + m / 100) * 9 / 10) * (600 * (1 - 2 * m / 100) + 220) = 3200 * 600 * (1 + 15.5 / 100) →
  m = 10 :=
by sorry

end value_of_m_l7_7413


namespace circle_center_coordinates_l7_7112

theorem circle_center_coordinates (x y : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → (x = 1 ∧ y = -2) :=
by
  sorry

end circle_center_coordinates_l7_7112


namespace game_necessarily_ends_winning_strategy_l7_7497

-- Definitions and conditions based on problem:
def Card := Fin 2009

def isWhite (c : Fin 2009) : Prop := sorry -- Placeholder for actual white card predicate

def validMove (k : Fin 2009) : Prop := k.val < 1969 ∧ isWhite k

def applyMove (k : Fin 2009) (cards : Fin 2009 → Prop) : Fin 2009 → Prop :=
  fun c => if c.val ≥ k.val ∧ c.val < k.val + 41 then ¬isWhite c else isWhite c

-- Theorem statements to match proof problem:
theorem game_necessarily_ends : ∃ n, n = 2009 → (∀ (cards : Fin 2009 → Prop), (∃ k < 1969, validMove k) → (∀ k < 1969, ¬(validMove k))) :=
sorry

theorem winning_strategy (cards : Fin 2009 → Prop) : ∃ strategy : (Fin 2009 → Prop) → Fin 2009, ∀ s, (s = applyMove (strategy s) s) → strategy s = sorry :=
sorry

end game_necessarily_ends_winning_strategy_l7_7497


namespace number_of_wins_and_losses_l7_7159

theorem number_of_wins_and_losses (x y : ℕ) (h1 : x + y = 15) (h2 : 3 * x + y = 41) :
  x = 13 ∧ y = 2 :=
sorry

end number_of_wins_and_losses_l7_7159


namespace length_of_bridge_l7_7492

theorem length_of_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (crossing_time_seconds : ℕ)
  (h_train_length : train_length = 125)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_crossing_time_seconds : crossing_time_seconds = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 250 :=
by
  sorry

end length_of_bridge_l7_7492


namespace number_of_possible_values_of_a_l7_7553

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ),
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2040 ∧
  a^2 - b^2 + c^2 - d^2 = 2040 ∧
  508 ∈ {a | ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2040 ∧ a^2 - b^2 + c^2 - d^2 = 2040}

theorem number_of_possible_values_of_a : problem_statement :=
  sorry

end number_of_possible_values_of_a_l7_7553


namespace cube_surface_area_l7_7518

theorem cube_surface_area (PQ a b : ℝ) (x : ℝ) 
  (h1 : PQ = a / 2) 
  (h2 : PQ = Real.sqrt (3 * x^2)) : 
  b = 6 * x^2 → b = a^2 / 2 := 
by
  intros h_surface
  -- sorry is added here to skip the proof step and ensure the code builds successfully.
  sorry

end cube_surface_area_l7_7518


namespace five_n_plus_three_composite_l7_7451

theorem five_n_plus_three_composite (n x y : ℕ) 
  (h_pos : 0 < n)
  (h1 : 2 * n + 1 = x ^ 2)
  (h2 : 3 * n + 1 = y ^ 2) : 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = 5 * n + 3 := 
sorry

end five_n_plus_three_composite_l7_7451


namespace merchant_marking_percentage_l7_7230

theorem merchant_marking_percentage (L : ℝ) (p : ℝ) (d : ℝ) (c : ℝ) (profit : ℝ) 
  (purchase_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (list_price : ℝ) : 
  L = 100 ∧ p = 30 ∧ d = 20 ∧ c = 20 ∧ profit = 20 ∧ 
  purchase_price = L - L * (p / 100) ∧ 
  marked_price = 109.375 ∧ 
  selling_price = marked_price - marked_price * (d / 100) ∧ 
  selling_price - purchase_price = profit * (selling_price / 100) 
  → marked_price = 109.375 := by sorry

end merchant_marking_percentage_l7_7230


namespace volume_of_sphere_in_cone_l7_7378

/-- The volume of a sphere inscribed in a right circular cone with
a base diameter of 16 inches and a cross-section with a vertex angle of 45 degrees
is 4096 * sqrt 2 * π / 3 cubic inches. -/
theorem volume_of_sphere_in_cone :
  let d := 16 -- the diameter of the base of the cone in inches
  let angle := 45 -- the vertex angle of the cross-section triangle in degrees
  let r := 8 * Real.sqrt 2 -- the radius of the sphere in inches
  let V := 4 / 3 * Real.pi * r^3 -- the volume of the sphere in cubic inches
  V = 4096 * Real.sqrt 2 * Real.pi / 3 :=
by
  simp only [Real.sqrt]
  sorry -- proof goes here

end volume_of_sphere_in_cone_l7_7378


namespace calculate_length_QR_l7_7398

noncomputable def length_QR (A : ℝ) (h : ℝ) (PQ : ℝ) (RS : ℝ) : ℝ :=
  21 - 0.5 * (Real.sqrt (PQ ^ 2 - h ^ 2) + Real.sqrt (RS ^ 2 - h ^ 2))

theorem calculate_length_QR :
  length_QR 210 10 12 21 = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by
  sorry

end calculate_length_QR_l7_7398


namespace egg_laying_hens_l7_7165

theorem egg_laying_hens (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) :
  total_chickens = 325 →
  roosters = 28 →
  non_laying_hens = 20 →
  (total_chickens - roosters - non_laying_hens = 277) :=
by
  intros
  sorry

end egg_laying_hens_l7_7165


namespace tank_emptying_time_l7_7341

theorem tank_emptying_time (fill_without_leak fill_with_leak : ℝ) (h1 : fill_without_leak = 7) (h2 : fill_with_leak = 8) : 
  let R := 1 / fill_without_leak
  let L := R - 1 / fill_with_leak
  let emptying_time := 1 / L
  emptying_time = 56 :=
by
  sorry

end tank_emptying_time_l7_7341


namespace Terrence_earns_l7_7866

theorem Terrence_earns :
  ∀ (J T E : ℝ), J + T + E = 90 ∧ J = T + 5 ∧ E = 25 → T = 30 :=
by
  intro J T E
  intro h
  obtain ⟨h₁, h₂, h₃⟩ := h
  sorry -- proof steps go here

end Terrence_earns_l7_7866


namespace width_of_field_l7_7158

noncomputable def field_width : ℝ := 60

theorem width_of_field (L W : ℝ) (hL : L = (7/5) * W) (hP : 288 = 2 * L + 2 * W) : W = field_width :=
by
  sorry

end width_of_field_l7_7158


namespace linemen_ounces_per_drink_l7_7686

-- Definitions corresponding to the conditions.
def linemen := 12
def skill_position_drink := 6
def skill_position_before_refill := 5
def cooler_capacity := 126

-- The theorem that requires proof.
theorem linemen_ounces_per_drink (L : ℕ) (h : 12 * L + 5 * skill_position_drink = cooler_capacity) : L = 8 :=
by
  sorry

end linemen_ounces_per_drink_l7_7686


namespace kelly_apples_total_l7_7908

def initial_apples : ℕ := 56
def second_day_pick : ℕ := 105
def third_day_pick : ℕ := 84
def apples_eaten : ℕ := 23

theorem kelly_apples_total :
  initial_apples + second_day_pick + third_day_pick - apples_eaten = 222 := by
  sorry

end kelly_apples_total_l7_7908


namespace flag_arrangement_division_l7_7482

noncomputable def flag_arrangement_modulo : ℕ :=
  let num_blue_flags := 9
  let num_red_flags := 8
  let num_slots := num_blue_flags + 1
  let initial_arrangements := (num_slots.choose num_red_flags) * (num_blue_flags + 1)
  let invalid_cases := (num_blue_flags.choose num_red_flags) * 2
  let M := initial_arrangements - invalid_cases
  M % 1000

theorem flag_arrangement_division (M : ℕ) (num_blue_flags num_red_flags : ℕ) :
  num_blue_flags = 9 → num_red_flags = 8 → M = flag_arrangement_modulo → M % 1000 = 432 :=
by
  intros _ _ hM
  rw [hM]
  trivial

end flag_arrangement_division_l7_7482


namespace odd_function_iff_l7_7512

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_iff (a b : ℝ) : 
  (∀ x, f x a b = -f (-x) a b) ↔ (a ^ 2 + b ^ 2 = 0) :=
by
  sorry

end odd_function_iff_l7_7512


namespace john_cards_sum_l7_7571

theorem john_cards_sum :
  ∃ (g : ℕ → ℕ) (y : ℕ → ℕ),
    (∀ n, (g n) ∈ [1, 2, 3, 4, 5]) ∧
    (∀ n, (y n) ∈ [2, 3, 4, 5]) ∧
    (∀ n, (g n < g (n + 1))) ∧
    (∀ n, (y n < y (n + 1))) ∧
    (∀ n, (g n ∣ y (n + 1) ∨ y (n + 1) ∣ g n)) ∧
    (g 0 = 1 ∧ g 2 = 2 ∧ g 4 = 5) ∧
    ( y 1 = 2 ∧ y 3 = 3 ∧ y 5 = 4 ) →
  g 0 + g 2 + g 4 = 8 := by
sorry

end john_cards_sum_l7_7571


namespace multiplier_for_average_grade_l7_7205

/-- Conditions -/
def num_of_grades_2 : ℕ := 3
def num_of_grades_3 : ℕ := 4
def num_of_grades_4 : ℕ := 1
def num_of_grades_5 : ℕ := 1
def cash_reward : ℕ := 15

-- Definitions for sums and averages based on the conditions
def sum_of_grades : ℕ :=
  num_of_grades_2 * 2 + num_of_grades_3 * 3 + num_of_grades_4 * 4 + num_of_grades_5 * 5

def total_grades : ℕ :=
  num_of_grades_2 + num_of_grades_3 + num_of_grades_4 + num_of_grades_5

def average_grade : ℕ :=
  sum_of_grades / total_grades

/-- Proof statement -/
theorem multiplier_for_average_grade : cash_reward / average_grade = 5 := by
  sorry

end multiplier_for_average_grade_l7_7205


namespace perpendicular_vecs_l7_7260

open Real

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (3, 4)
def lambda := 1 / 2

theorem perpendicular_vecs : 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0 := 
by 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  show (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0
  sorry

end perpendicular_vecs_l7_7260


namespace solve_for_x_l7_7020

theorem solve_for_x (x: ℚ) (h: (3/5 - 1/4) = 4/x) : x = 80/7 :=
by
  sorry

end solve_for_x_l7_7020


namespace inequality_relation_l7_7164

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log Q / Real.log 2

theorem inequality_relation : R < Q ∧ Q < P := by
  sorry

end inequality_relation_l7_7164


namespace sandy_initial_fish_l7_7664

theorem sandy_initial_fish (bought_fish : ℕ) (total_fish : ℕ) (h1 : bought_fish = 6) (h2 : total_fish = 32) :
  total_fish - bought_fish = 26 :=
by
  sorry

end sandy_initial_fish_l7_7664


namespace cylinder_height_l7_7668

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end cylinder_height_l7_7668


namespace smallest_b_value_l7_7578

noncomputable def smallest_possible_value_of_b : ℝ :=
  (3 + Real.sqrt 5) / 2

theorem smallest_b_value
  (a b : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : (1/b) + (1/a) ≤ 1) :
  b = smallest_possible_value_of_b :=
sorry

end smallest_b_value_l7_7578


namespace find_m_symmetry_l7_7312

theorem find_m_symmetry (A B : ℝ × ℝ) (m : ℝ)
  (hA : A = (-3, m)) (hB : B = (3, 4)) (hy : A.2 = B.2) : m = 4 :=
sorry

end find_m_symmetry_l7_7312


namespace max_MB_value_l7_7826

open Real

-- Define the conditions of the problem
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : sqrt 6 / 3 = sqrt (1 - b^2 / a^2))

-- Define the point M and the vertex B on the ellipse
variables (M : ℝ × ℝ) (hM : (M.1)^2 / (a)^2 + (M.2)^2 / (b)^2 = 1)
def B : ℝ × ℝ := (0, -b)

-- The task is to prove the maximum value of |MB| given the conditions
theorem max_MB_value : ∃ (maxMB : ℝ), maxMB = (3 * sqrt 2 / 2) * b :=
sorry

end max_MB_value_l7_7826


namespace value_of_m_l7_7203

theorem value_of_m :
  ∃ m : ℝ, (3 - 1) / (m + 2) = 1 → m = 0 :=
by 
  sorry

end value_of_m_l7_7203


namespace check_ratio_l7_7696

theorem check_ratio (initial_balance check_amount new_balance : ℕ) 
  (h1 : initial_balance = 150) (h2 : check_amount = 50) (h3 : new_balance = initial_balance + check_amount) :
  (check_amount : ℚ) / new_balance = 1 / 4 := 
by { 
  sorry 
}

end check_ratio_l7_7696


namespace cubes_identity_l7_7842

theorem cubes_identity (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 40) : 
    a^3 + b^3 + c^3 - 3 * a * b * c = 1575 :=
by 
  sorry

end cubes_identity_l7_7842


namespace age_twice_in_two_years_l7_7223

-- conditions
def father_age (S : ℕ) : ℕ := S + 24
def present_son_age : ℕ := 22
def present_father_age : ℕ := father_age present_son_age

-- theorem statement
theorem age_twice_in_two_years (S M Y : ℕ) (h1 : S = present_son_age) (h2 : M = present_father_age) : 
  M + 2 = 2 * (S + 2) :=
by
  sorry

end age_twice_in_two_years_l7_7223


namespace incorrect_conclusion_l7_7963

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem incorrect_conclusion :
  ¬ (∀ x : ℝ, f ( (3 * Real.pi) / 4 - x ) + f x = 0) :=
by
  sorry

end incorrect_conclusion_l7_7963


namespace misread_weight_l7_7764

-- Definitions based on given conditions in part (a)
def initial_avg_weight : ℝ := 58.4
def num_boys : ℕ := 20
def correct_weight : ℝ := 61
def correct_avg_weight : ℝ := 58.65

-- The Lean theorem statement that needs to be proved
theorem misread_weight :
  let incorrect_total_weight := initial_avg_weight * num_boys
  let correct_total_weight := correct_avg_weight * num_boys
  let weight_diff := correct_total_weight - incorrect_total_weight
  correct_weight - weight_diff = 56 := sorry

end misread_weight_l7_7764


namespace age_hence_l7_7196

theorem age_hence (A x : ℕ) (hA : A = 24) (hx : 4 * (A + x) - 4 * (A - 3) = A) : x = 3 :=
by {
  sorry
}

end age_hence_l7_7196


namespace total_revenue_correct_l7_7004

-- Definitions and conditions
def number_of_fair_tickets : ℕ := 60
def price_per_fair_ticket : ℕ := 15
def price_per_baseball_ticket : ℕ := 10
def number_of_baseball_tickets : ℕ := number_of_fair_tickets / 3

-- Calculate revenues
def revenue_from_fair_tickets : ℕ := number_of_fair_tickets * price_per_fair_ticket
def revenue_from_baseball_tickets : ℕ := number_of_baseball_tickets * price_per_baseball_ticket
def total_revenue : ℕ := revenue_from_fair_tickets + revenue_from_baseball_tickets

-- Proof statement
theorem total_revenue_correct : total_revenue = 1100 := by
  sorry

end total_revenue_correct_l7_7004


namespace fernanda_savings_before_payments_l7_7900

open Real

theorem fernanda_savings_before_payments (aryan_debt kyro_debt aryan_payment kyro_payment total_savings before_savings : ℝ) 
  (h1: aryan_debt = 1200)
  (h2: aryan_debt = 2 * kyro_debt)
  (h3: aryan_payment = 0.6 * aryan_debt)
  (h4: kyro_payment = 0.8 * kyro_debt)
  (h5: total_savings = before_savings + aryan_payment + kyro_payment)
  (h6: total_savings = 1500) :
  before_savings = 300 :=
by
  sorry

end fernanda_savings_before_payments_l7_7900


namespace sequence_is_decreasing_l7_7109

-- Define the sequence {a_n} using a recursive function
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1))

-- Define a condition ensuring the sequence a_n is decreasing
theorem sequence_is_decreasing (a : ℕ → ℝ) (h : seq a) : ∀ n, a (n + 1) < a n :=
by
  intro n
  sorry

end sequence_is_decreasing_l7_7109


namespace a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l7_7652

variable {a b c : ℝ}

theorem a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2 :
  ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) :=
sorry

theorem a_gt_b_necessary_not_sufficient_ac2_gt_bc2 :
  ¬((a > b) → (a * c^2 > b * c^2)) ∧ ((a * c^2 > b * c^2) → (a > b)) :=
sorry

end a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l7_7652


namespace tomato_plants_per_row_l7_7695

-- Definitions based on given conditions.
variables (T C P : ℕ)

-- Condition 1: For each row of tomato plants, she is planting 2 rows of cucumbers
def cucumber_rows (T : ℕ) := 2 * T

-- Condition 2: She has enough room for 15 rows of plants in total
def total_rows (T : ℕ) (C : ℕ) := T + C = 15

-- Condition 3: If each plant produces 3 tomatoes, she will have 120 tomatoes in total
def total_tomatoes (P : ℕ) := 5 * P * 3 = 120

-- The task is to prove that P = 8
theorem tomato_plants_per_row : 
  ∀ T C P : ℕ, cucumber_rows T = C → total_rows T C → total_tomatoes P → P = 8 :=
by
  -- The actual proof will go here.
  sorry

end tomato_plants_per_row_l7_7695


namespace find_sum_l7_7326

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) + f (2 - x) = 0

theorem find_sum (f : ℝ → ℝ) (h_odd : odd_function f) (h_func : functional_equation f) (h_val : f 1 = 9) :
  f 2016 + f 2017 + f 2018 = 9 :=
  sorry

end find_sum_l7_7326


namespace diego_annual_savings_l7_7239

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end diego_annual_savings_l7_7239


namespace plane_through_points_l7_7370

-- Define the vectors as tuples of three integers
def point := (ℤ × ℤ × ℤ)

-- The given points
def p : point := (2, -1, 3)
def q : point := (4, -1, 5)
def r : point := (5, -3, 4)

-- A function to find the equation of the plane given three points
def plane_equation (p q r : point) : ℤ × ℤ × ℤ × ℤ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  let (rx, ry, rz) := r
  let a := (qy - py) * (rz - pz) - (qy - py) * (rz - pz)
  let b := (qx - px) * (rz - pz) - (qx - px) * (rz - pz)
  let c := (qx - px) * (ry - py) - (qx - px) * (ry - py)
  let d := -(a * px + b * py + c * pz)
  (a, b, c, d)

-- The proof statement
theorem plane_through_points : plane_equation (2, -1, 3) (4, -1, 5) (5, -3, 4) = (1, 2, -2, 6) :=
  by sorry

end plane_through_points_l7_7370


namespace polyhedron_inequality_proof_l7_7248

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l7_7248


namespace find_b_l7_7104

def p (x : ℝ) : ℝ := 2 * x - 3
def q (x : ℝ) (b : ℝ) : ℝ := 5 * x - b

theorem find_b (b : ℝ) (h : p (q 3 b) = 13) : b = 7 :=
by sorry

end find_b_l7_7104


namespace polygon_diagonals_with_restriction_l7_7183

def num_sides := 150

def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

def restricted_diagonals (n : ℕ) : ℕ :=
  n * 150 / 4

def valid_diagonals (n : ℕ) : ℕ :=
  total_diagonals n - restricted_diagonals n

theorem polygon_diagonals_with_restriction : valid_diagonals num_sides = 5400 :=
by
  sorry

end polygon_diagonals_with_restriction_l7_7183


namespace roots_of_cubic_equation_l7_7780

theorem roots_of_cubic_equation 
  (k m : ℝ) 
  (h : ∀r1 r2 r3: ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 + r2 + r3 = 7 ∧ r1 * r2 * r3 = m ∧ (r1 * r2 + r2 * r3 + r1 * r3) = k) : 
  k + m = 22 := sorry

end roots_of_cubic_equation_l7_7780


namespace system_of_equations_implies_quadratic_l7_7007

theorem system_of_equations_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y + 2 = 0) ∧ (3 * x + y + 4 = 0) → (y^2 + 11 * y - 14 = 0) := by
  sorry

end system_of_equations_implies_quadratic_l7_7007


namespace greatest_possible_integer_l7_7732

theorem greatest_possible_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 1) (h3 : ∃ l : ℕ, n = 10 * l - 4) : n = 86 := 
sorry

end greatest_possible_integer_l7_7732


namespace election_winner_votes_l7_7167

-- Define the conditions and question in Lean 4
theorem election_winner_votes (V : ℝ) (h1 : V > 0) 
  (h2 : 0.54 * V - 0.46 * V = 288) : 0.54 * V = 1944 :=
by
  sorry

end election_winner_votes_l7_7167


namespace solve_for_x_l7_7149

theorem solve_for_x : ∀ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) → x = 0 := by
  intros x h
  sorry

end solve_for_x_l7_7149


namespace hexagon_tiling_min_colors_l7_7684

theorem hexagon_tiling_min_colors :
  ∀ (s₁ s₂ : ℝ) (hex_area : ℝ) (tile_area : ℝ) (tiles_needed : ℕ) (n : ℕ),
    s₁ = 6 →
    s₂ = 0.5 →
    hex_area = (3 * Real.sqrt 3 / 2) * s₁^2 →
    tile_area = (Real.sqrt 3 / 4) * s₂^2 →
    tiles_needed = hex_area / tile_area →
    tiles_needed ≤ (Nat.choose n 3) →
    n ≥ 19 :=
by
  intros s₁ s₂ hex_area tile_area tiles_needed n
  intros s₁_eq s₂_eq hex_area_eq tile_area_eq tiles_needed_eq color_constraint
  sorry

end hexagon_tiling_min_colors_l7_7684


namespace xy_sum_cases_l7_7397

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end xy_sum_cases_l7_7397


namespace find_all_triples_l7_7095

def satisfying_triples (a b c : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  (a^2 + a*b = c) ∧ 
  (b^2 + b*c = a) ∧ 
  (c^2 + c*a = b)

theorem find_all_triples (a b c : ℝ) : satisfying_triples a b c ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end find_all_triples_l7_7095


namespace Beto_can_determine_xy_l7_7311

theorem Beto_can_determine_xy (m n : ℤ) :
  (∃ k t : ℤ, 0 < t ∧ m = 2 * k + 1 ∧ n = 2 * t * (2 * k + 1)) ↔ 
  (∀ x y : ℝ, (∃ a b : ℝ, a ≠ b ∧ x = a ∧ y = b) →
    ∃ xy_val : ℝ, (x^m + y^m = xy_val) ∧ (x^n + y^n = xy_val)) := 
sorry

end Beto_can_determine_xy_l7_7311


namespace vasya_gift_choices_l7_7156

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l7_7156


namespace club_membership_l7_7806

theorem club_membership (n : ℕ) : 
  n ≡ 6 [MOD 10] → n ≡ 6 [MOD 11] → 200 ≤ n ∧ n ≤ 300 → n = 226 :=
by
  intros h1 h2 h3
  sorry

end club_membership_l7_7806


namespace num_integers_between_sqrt10_sqrt100_l7_7286

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l7_7286


namespace guacamole_serving_and_cost_l7_7811

theorem guacamole_serving_and_cost 
  (initial_avocados : ℕ) 
  (additional_avocados : ℕ) 
  (avocados_per_serving : ℕ) 
  (x : ℝ) 
  (h_initial : initial_avocados = 5) 
  (h_additional : additional_avocados = 4) 
  (h_serving : avocados_per_serving = 3) :
  (initial_avocados + additional_avocados) / avocados_per_serving = 3 
  ∧ additional_avocados * x = 4 * x := by
  sorry

end guacamole_serving_and_cost_l7_7811


namespace first_sequence_correct_second_sequence_correct_l7_7883

theorem first_sequence_correct (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 = 12) (h2 : a2 = a1 + 4) (h3 : a3 = a2 + 4) (h4 : a4 = a3 + 4) (h5 : a5 = a4 + 4) :
  a4 = 24 ∧ a5 = 28 :=
by sorry

theorem second_sequence_correct (b1 b2 b3 b4 b5 : ℕ) (h1 : b1 = 2) (h2 : b2 = b1 * 2) (h3 : b3 = b2 * 2) (h4 : b4 = b3 * 2) (h5 : b5 = b4 * 2) :
  b4 = 16 ∧ b5 = 32 :=
by sorry

end first_sequence_correct_second_sequence_correct_l7_7883


namespace evaluate_polynomial_at_3_l7_7563

def f (x : ℕ) : ℕ := 3 * x ^ 3 + x - 3

theorem evaluate_polynomial_at_3 : f 3 = 28 :=
by
  sorry

end evaluate_polynomial_at_3_l7_7563


namespace pencil_groups_l7_7838

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups_l7_7838


namespace product_bc_l7_7792

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l7_7792


namespace present_age_of_son_l7_7111

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l7_7111


namespace implication_a_lt_b_implies_a_lt_b_plus_1_l7_7346

theorem implication_a_lt_b_implies_a_lt_b_plus_1 (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end implication_a_lt_b_implies_a_lt_b_plus_1_l7_7346


namespace general_term_of_sequence_l7_7500

theorem general_term_of_sequence (a : Nat → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (2 + a (n + 1))) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 2) := 
sorry

end general_term_of_sequence_l7_7500


namespace hannah_games_l7_7988

theorem hannah_games (total_points : ℕ) (avg_points_per_game : ℕ) (h1 : total_points = 312) (h2 : avg_points_per_game = 13) :
  total_points / avg_points_per_game = 24 :=
sorry

end hannah_games_l7_7988


namespace mass_of_barium_sulfate_l7_7855

-- Definitions of the chemical equation and molar masses
def barium_molar_mass : ℝ := 137.327
def sulfur_molar_mass : ℝ := 32.065
def oxygen_molar_mass : ℝ := 15.999
def molar_mass_BaSO4 : ℝ := barium_molar_mass + sulfur_molar_mass + 4 * oxygen_molar_mass

-- Given conditions
def moles_BaBr2 : ℝ := 4
def moles_BaSO4_produced : ℝ := moles_BaBr2 -- from balanced equation

-- Calculate mass of BaSO4 produced
def mass_BaSO4 : ℝ := moles_BaSO4_produced * molar_mass_BaSO4

-- Mass of Barium sulfate produced
theorem mass_of_barium_sulfate : mass_BaSO4 = 933.552 :=
by 
  -- Skip the proof
  sorry

end mass_of_barium_sulfate_l7_7855


namespace mean_age_of_children_l7_7969

theorem mean_age_of_children :
  let ages := [8, 8, 12, 12, 10, 14]
  let n := ages.length
  let sum_ages := ages.foldr (· + ·) 0
  let mean_age := sum_ages / n
  mean_age = 10 + 2 / 3 :=
by
  sorry

end mean_age_of_children_l7_7969


namespace keith_picked_p_l7_7048

-- Definitions of the given conditions
def p_j : ℕ := 46  -- Jason's pears
def p_m : ℕ := 12  -- Mike's pears
def p_t : ℕ := 105 -- Total pears picked

-- The theorem statement
theorem keith_picked_p : p_t - (p_j + p_m) = 47 := by
  -- Proof part will be handled later
  sorry

end keith_picked_p_l7_7048


namespace find_charge_federal_return_l7_7638

-- Definitions based on conditions
def charge_federal_return (F : ℝ) : ℝ := F
def charge_state_return : ℝ := 30
def charge_quarterly_return : ℝ := 80
def sold_federal_returns : ℝ := 60
def sold_state_returns : ℝ := 20
def sold_quarterly_returns : ℝ := 10
def total_revenue : ℝ := 4400

-- Lean proof statement to verify the value of F
theorem find_charge_federal_return (F : ℝ) (h : sold_federal_returns * charge_federal_return F + sold_state_returns * charge_state_return + sold_quarterly_returns * charge_quarterly_return = total_revenue) : 
  F = 50 :=
by
  sorry

end find_charge_federal_return_l7_7638


namespace product_of_possible_values_l7_7365

theorem product_of_possible_values : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 2 ∨ x = 8)) → (2 * 8) = 16 :=
by 
  sorry

end product_of_possible_values_l7_7365


namespace tangent_line_parabola_d_l7_7385

theorem tangent_line_parabola_d (d : ℝ) :
  (∀ x y : ℝ, (y = 3 * x + d) → (y^2 = 12 * x) → ∃! x, 9 * x^2 + (6 * d - 12) * x + d^2 = 0) → d = 1 :=
by
  sorry

end tangent_line_parabola_d_l7_7385


namespace ellipse_nec_but_not_suff_l7_7713

-- Definitions and conditions
def isEllipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c

/-- Given that the sum of the distances from a moving point P in the plane to two fixed points is constant,
the condition is necessary but not sufficient for the trajectory of the moving point P being an ellipse. -/
theorem ellipse_nec_but_not_suff (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (c : ℝ) :
  (∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) →
  (c > dist F1 F2 → ¬ isEllipse P F1 F2) ∧ (isEllipse P F1 F2 → ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) :=
by
  sorry

end ellipse_nec_but_not_suff_l7_7713


namespace find_triangle_sides_l7_7990

theorem find_triangle_sides (a : Fin 7 → ℝ) (h : ∀ i, 1 < a i ∧ a i < 13) : 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 7 ∧ 
           a i + a j > a k ∧ 
           a j + a k > a i ∧ 
           a k + a i > a j :=
sorry

end find_triangle_sides_l7_7990


namespace part1_l7_7885

theorem part1 (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) : 2 * x^2 + y^2 > x^2 + x * y := 
sorry

end part1_l7_7885


namespace proof_goal_l7_7249

noncomputable def exp_value (k m n : ℕ) : ℤ :=
  (6^k - k^6 + 2^m - 4^m + n^3 - 3^n : ℤ)

theorem proof_goal (k m n : ℕ) (h_k : 18^k ∣ 624938) (h_m : 24^m ∣ 819304) (h_n : n = 2 * k + m) :
  exp_value k m n = 0 := by
  sorry

end proof_goal_l7_7249


namespace green_fraction_is_three_fifths_l7_7646

noncomputable def fraction_green_after_tripling (total_balloons : ℕ) : ℚ :=
  let green_balloons := total_balloons / 3
  let new_green_balloons := green_balloons * 3
  let new_total_balloons := total_balloons * (5 / 3)
  new_green_balloons / new_total_balloons

theorem green_fraction_is_three_fifths (total_balloons : ℕ) (h : total_balloons > 0) : 
  fraction_green_after_tripling total_balloons = 3 / 5 := 
by 
  sorry

end green_fraction_is_three_fifths_l7_7646


namespace necessary_and_sufficient_condition_for_x2_ne_y2_l7_7768

theorem necessary_and_sufficient_condition_for_x2_ne_y2 (x y : ℤ) :
  (x ^ 2 ≠ y ^ 2) ↔ (x ≠ y ∧ x ≠ -y) :=
by
  sorry

end necessary_and_sufficient_condition_for_x2_ne_y2_l7_7768


namespace least_alpha_condition_l7_7840

variables {a b α : ℝ}

theorem least_alpha_condition (a_gt_1 : a > 1) (b_gt_0 : b > 0) : 
  ∀ x, (x ≥ α) → (a + b) ^ x ≥ a ^ x + b ↔ α = 1 :=
by
  sorry

end least_alpha_condition_l7_7840


namespace greatest_third_side_l7_7324

theorem greatest_third_side (a b c : ℝ) (h₀: a = 5) (h₁: b = 11) (h₂ : 6 < c ∧ c < 16) : c ≤ 15 :=
by
  -- assumption applying that c needs to be within 6 and 16
  have h₃ : 6 < c := h₂.1
  have h₄: c < 16 := h₂.2
  -- need to show greatest integer c is 15
  sorry

end greatest_third_side_l7_7324


namespace line_intersects_y_axis_at_point_intersection_at_y_axis_l7_7841

theorem line_intersects_y_axis_at_point :
  ∃ y, 5 * 0 - 7 * y = 35 := sorry

theorem intersection_at_y_axis :
  (∃ y, 5 * 0 - 7 * y = 35) → 0 - 7 * (-5) = 35 := sorry

end line_intersects_y_axis_at_point_intersection_at_y_axis_l7_7841


namespace problem_statement_l7_7085

noncomputable def find_pq_sum (XZ YZ : ℕ) (XY_perimeter_ratio : ℕ × ℕ) : ℕ :=
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let ZD := Real.sqrt (XZ * YZ)
  let O_radius := 0.5 * ZD
  let tangent_length := Real.sqrt ((XY / 2)^2 - O_radius^2)
  let perimeter := XY + 2 * tangent_length
  let (p, q) := XY_perimeter_ratio
  p + q

theorem problem_statement :
  find_pq_sum 8 15 (30, 17) = 47 :=
by sorry

end problem_statement_l7_7085


namespace Eddy_travel_time_l7_7126

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end Eddy_travel_time_l7_7126


namespace smallest_n_not_divisible_by_10_smallest_n_correct_l7_7937

theorem smallest_n_not_divisible_by_10 :
  ∃ n ≥ 2017, n % 4 = 0 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 :=
by
  -- Existence proof of such n is omitted
  sorry

def smallest_n : Nat :=
  Nat.find $ smallest_n_not_divisible_by_10

theorem smallest_n_correct : smallest_n = 2020 :=
by
  -- Correctness proof of smallest_n is omitted
  sorry

end smallest_n_not_divisible_by_10_smallest_n_correct_l7_7937


namespace correct_transformation_l7_7825

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : 
  (a / b) = ((a + 2 * a) / (b + 2 * b)) :=
by 
  sorry

end correct_transformation_l7_7825


namespace total_selling_price_correct_l7_7533

noncomputable def calculateSellingPrice (price1 price2 price3 loss1 loss2 loss3 taxRate overheadCost : ℝ) : ℝ :=
  let totalPurchasePrice := price1 + price2 + price3
  let tax := taxRate * totalPurchasePrice
  let sellingPrice1 := price1 - (loss1 * price1)
  let sellingPrice2 := price2 - (loss2 * price2)
  let sellingPrice3 := price3 - (loss3 * price3)
  let totalSellingPrice := sellingPrice1 + sellingPrice2 + sellingPrice3
  totalSellingPrice + overheadCost + tax

theorem total_selling_price_correct :
  calculateSellingPrice 750 1200 500 0.10 0.15 0.05 0.05 300 = 2592.5 :=
by 
  -- The proof of this theorem is skipped.
  sorry

end total_selling_price_correct_l7_7533


namespace correct_operations_l7_7992

variable (x : ℚ)

def incorrect_equation := ((x - 5) * 3) / 7 = 10

theorem correct_operations :
  incorrect_equation x → (3 * x - 5) / 7 = 80 / 7 :=
by
  intro h
  sorry

end correct_operations_l7_7992


namespace factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l7_7687

-- Definitions from conditions
theorem factorization_option_a (a b : ℝ) : a^4 * b - 6 * a^3 * b + 9 * a^2 * b = a^2 * b * (a^2 - 6 * a + 9) ↔ a^2 * b * (a - 3)^2 ≠ a^2 * b * (a^2 - 6 * a - 9) := sorry

theorem factorization_option_b (x : ℝ) : (x^2 - x + 1/4) = (x - 1/2)^2 := sorry

theorem factorization_option_c (x : ℝ) : x^2 - 2 * x + 4 = (x - 2)^2 ↔ x^2 - 2 * x + 4 ≠ x^2 - 4 * x + 4 := sorry

theorem factorization_option_d (x y : ℝ) : 4 * x^2 - y^2 = (2 * x + y) * (2 * x - y) ↔ (4 * x + y) * (4 * x - y) ≠ (2 * x + y) * (2 * x - y) := sorry

-- Main theorem that states option B's factorization is correct
theorem correct_factorization_b (x : ℝ) (h1 : x^2 - x + 1/4 = (x - 1/2)^2)
                                (h2 : ∀ (a b : ℝ), a^4 * b - 6 * a^3 * b + 9 * a^2 * b ≠ a^2 * b * (a^2 - 6 * a - 9))
                                (h3 : ∀ (x : ℝ), x^2 - 2 * x + 4 ≠ (x - 2)^2)
                                (h4 : ∀ (x y : ℝ), 4 * x^2 - y^2 ≠ (4 * x + y) * (4 * x - y)) : 
                                (x^2 - x + 1/4 = (x - 1/2)^2) := 
                                by 
                                sorry

end factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l7_7687


namespace shortest_distance_to_line_l7_7372

open Classical

variables {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (PA PB PC : ℝ)
variables (l : ℕ) -- l represents the line

-- Given conditions
def PA_dist : ℝ := 4
def PB_dist : ℝ := 5
def PC_dist : ℝ := 2

theorem shortest_distance_to_line (hPA : PA = PA_dist) (hPB : PB = PB_dist) (hPC : PC = PC_dist) :
  ∃ d, d ≤ 2 := 
sorry

end shortest_distance_to_line_l7_7372


namespace find_AX_l7_7991

theorem find_AX (AC BC BX : ℝ) (h1 : AC = 27) (h2 : BC = 40) (h3 : BX = 36)
    (h4 : ∀ (AX : ℝ), AX = AC * BX / BC) : 
    ∃ AX, AX = 243 / 10 :=
by
  sorry

end find_AX_l7_7991


namespace hyperbola_foci_coords_l7_7080

theorem hyperbola_foci_coords :
  ∀ x y, (x^2) / 8 - (y^2) / 17 = 1 → (x, y) = (5, 0) ∨ (x, y) = (-5, 0) :=
by
  sorry

end hyperbola_foci_coords_l7_7080


namespace replacement_fraction_l7_7081

variable (Q : ℝ) (x : ℝ)

def initial_concentration : ℝ := 0.70
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

theorem replacement_fraction (h1 : 0.70 * Q - 0.70 * x * Q + 0.25 * x * Q = 0.35 * Q) :
  x = 7 / 9 :=
by
  sorry

end replacement_fraction_l7_7081


namespace number_of_extreme_points_zero_l7_7096

def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x, f x1 a = f x a → x = x1 ∨ x = x2) → False := 
by
  sorry

end number_of_extreme_points_zero_l7_7096


namespace unique_tangent_circle_of_radius_2_l7_7179

noncomputable def is_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  dist c₁ c₂ = r₁ + r₂

theorem unique_tangent_circle_of_radius_2
    (C1_center C2_center C3_center : ℝ × ℝ)
    (h_C1_C2 : is_tangent C1_center C2_center 1 1)
    (h_C2_C3 : is_tangent C2_center C3_center 1 1)
    (h_C3_C1 : is_tangent C3_center C1_center 1 1):
    ∃! center : ℝ × ℝ, is_tangent center C1_center 2 1 ∧
                        is_tangent center C2_center 2 1 ∧
                        is_tangent center C3_center 2 1 := sorry

end unique_tangent_circle_of_radius_2_l7_7179


namespace value_of_expression_l7_7501

theorem value_of_expression : (165^2 - 153^2) / 12 = 318 := by
  sorry

end value_of_expression_l7_7501


namespace roots_theorem_l7_7881

-- Definitions and Conditions
def root1 (a b p : ℝ) : Prop := 
  a + b = -p ∧ a * b = 1

def root2 (b c q : ℝ) : Prop := 
  b + c = -q ∧ b * c = 2

-- The theorem to prove
theorem roots_theorem (a b c p q : ℝ) (h1 : root1 a b p) (h2 : root2 b c q) : 
  (b - a) * (b - c) = p * q - 6 :=
sorry

end roots_theorem_l7_7881


namespace correct_option_l7_7805

-- Define the given conditions
def a : ℕ := 7^5
def b : ℕ := 5^7

-- State the theorem to be proven
theorem correct_option : a^7 * b^5 = 35^35 := by
  -- insert the proof here
  sorry

end correct_option_l7_7805


namespace sin_theta_value_l7_7543

theorem sin_theta_value (θ : ℝ) (h1 : 10 * (Real.tan θ) = 4 * (Real.cos θ)) (h2 : 0 < θ ∧ θ < π) : Real.sin θ = 1/2 :=
by
  sorry

end sin_theta_value_l7_7543


namespace total_distance_hiked_east_l7_7766

-- Define Annika's constant rate of hiking
def constant_rate : ℝ := 10 -- minutes per kilometer

-- Define already hiked distance
def distance_hiked : ℝ := 2.75 -- kilometers

-- Define total available time to return
def total_time : ℝ := 45 -- minutes

-- Prove that the total distance hiked east is 4.5 kilometers
theorem total_distance_hiked_east : distance_hiked + (total_time - distance_hiked * constant_rate) / constant_rate = 4.5 :=
by
  sorry

end total_distance_hiked_east_l7_7766


namespace sine_curve_transformation_l7_7944

theorem sine_curve_transformation (x y x' y' : ℝ) 
  (h1 : x' = (1 / 2) * x) 
  (h2 : y' = 3 * y) :
  (y = Real.sin x) ↔ (y' = 3 * Real.sin (2 * x')) := by 
  sorry

end sine_curve_transformation_l7_7944


namespace fill_time_two_pipes_l7_7658

variable (R : ℝ)
variable (c : ℝ)
variable (t1 : ℝ) (t2 : ℝ)

noncomputable def fill_time_with_pipes (num_pipes : ℝ) (time_per_tank : ℝ) : ℝ :=
  time_per_tank / num_pipes

theorem fill_time_two_pipes (h1 : fill_time_with_pipes 3 t1 = 12) 
                            (h2 : c = R)
                            : fill_time_with_pipes 2 (3 * R * t1) = 18 := 
by
  sorry

end fill_time_two_pipes_l7_7658


namespace price_reduction_correct_eqn_l7_7027

theorem price_reduction_correct_eqn (x : ℝ) :
  120 * (1 - x)^2 = 85 :=
sorry

end price_reduction_correct_eqn_l7_7027


namespace base_of_parallelogram_l7_7055

theorem base_of_parallelogram (A h b : ℝ) (hA : A = 960) (hh : h = 16) :
  A = h * b → b = 60 :=
by
  sorry

end base_of_parallelogram_l7_7055


namespace abcd_mod_7_zero_l7_7290

theorem abcd_mod_7_zero
  (a b c d : ℕ)
  (h1 : a + 2 * b + 3 * c + 4 * d ≡ 1 [MOD 7])
  (h2 : 2 * a + 3 * b + c + 2 * d ≡ 5 [MOD 7])
  (h3 : 3 * a + b + 2 * c + 3 * d ≡ 3 [MOD 7])
  (h4 : 4 * a + 2 * b + d + c ≡ 2 [MOD 7])
  (ha : a < 7) (hb : b < 7) (hc : c < 7) (hd : d < 7) :
  (a * b * c * d) % 7 = 0 :=
by sorry

end abcd_mod_7_zero_l7_7290


namespace rebate_percentage_l7_7968

theorem rebate_percentage (r : ℝ) (h1 : 0 ≤ r) (h2 : r ≤ 1) 
(h3 : (6650 - 6650 * r) * 1.10 = 6876.1) : r = 0.06 :=
sorry

end rebate_percentage_l7_7968


namespace sum_of_intercepts_l7_7462

theorem sum_of_intercepts (a b c : ℕ) :
  (∃ y, x = 2 * y^2 - 6 * y + 3 ∧ x = a ∧ y = 0) ∧
  (∃ y1 y2, x = 0 ∧ 2 * y1^2 - 6 * y1 + 3 = 0 ∧ 2 * y2^2 - 6 * y2 + 3 = 0 ∧ y1 + y2 = b + c) →
  a + b + c = 6 :=
by 
  sorry

end sum_of_intercepts_l7_7462


namespace parents_give_per_year_l7_7138

def Mikail_age (x : ℕ) : Prop :=
  x = 3 * (x - 3)

noncomputable def money_per_year (total_money : ℕ) (age : ℕ) : ℕ :=
  total_money / age

theorem parents_give_per_year 
  (x : ℕ) (hx : Mikail_age x) : 
  money_per_year 45 x = 5 :=
sorry

end parents_give_per_year_l7_7138


namespace x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l7_7232

theorem x_sq_plus_3x_minus_2_ge_zero (x : ℝ) (h : x ≥ 1) : x^2 + 3 * x - 2 ≥ 0 :=
sorry

theorem neg_x_sq_plus_3x_minus_2_lt_zero (x : ℝ) (h : x < 1) : x^2 + 3 * x - 2 < 0 :=
sorry

end x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l7_7232


namespace simplify_expression_l7_7527

-- Define the variables and the polynomials
variables (y : ℤ)

-- Define the expressions
def expr1 := (2 * y - 1) * (5 * y^12 - 3 * y^11 + y^9 - 4 * y^8)
def expr2 := 10 * y^13 - 11 * y^12 + 3 * y^11 + y^10 - 9 * y^9 + 4 * y^8

-- State the theorem
theorem simplify_expression : expr1 = expr2 := by
  sorry

end simplify_expression_l7_7527


namespace determine_n_l7_7387

theorem determine_n (n : ℕ) (h : 17^(4 * n) = (1 / 17)^(n - 30)) : n = 6 :=
by {
  sorry
}

end determine_n_l7_7387


namespace matrix_operation_correct_l7_7278

open Matrix

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 5]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 4], ![0, -3]]
def matrix3 : Matrix (Fin 2) (Fin 2) ℤ := ![![6, 0], ![-1, 8]]
def result : Matrix (Fin 2) (Fin 2) ℤ := ![![12, -7], ![1, 16]]

theorem matrix_operation_correct:
  matrix1 - matrix2 + matrix3 = result :=
by
  sorry

end matrix_operation_correct_l7_7278


namespace mrs_hilt_more_l7_7295

-- Define the values of the pennies, nickels, and dimes.
def value_penny : ℝ := 0.01
def value_nickel : ℝ := 0.05
def value_dime : ℝ := 0.10

-- Define the count of coins Mrs. Hilt has.
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

-- Define the count of coins Jacob has.
def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount of money Mrs. Hilt has.
def mrs_hilt_total : ℝ :=
  mrs_hilt_pennies * value_penny
  + mrs_hilt_nickels * value_nickel
  + mrs_hilt_dimes * value_dime

-- Calculate the total amount of money Jacob has.
def jacob_total : ℝ :=
  jacob_pennies * value_penny
  + jacob_nickels * value_nickel
  + jacob_dimes * value_dime

-- Prove that Mrs. Hilt has $0.13 more than Jacob.
theorem mrs_hilt_more : mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end mrs_hilt_more_l7_7295


namespace binomial_product_l7_7135

theorem binomial_product (x : ℝ) : 
  (2 - x^4) * (3 + x^5) = -x^9 - 3 * x^4 + 2 * x^5 + 6 :=
by 
  sorry

end binomial_product_l7_7135


namespace minimum_value_inequality_l7_7208

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end minimum_value_inequality_l7_7208


namespace sequence_sum_zero_l7_7930

theorem sequence_sum_zero (n : ℕ) (h : n > 1) :
  (∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + n * a (n * k) = 0)) ↔ n ≥ 3 := 
by sorry

end sequence_sum_zero_l7_7930


namespace find_maximum_marks_l7_7388

theorem find_maximum_marks (M : ℝ) 
  (h1 : 0.60 * M = 270)
  (h2 : ∀ x : ℝ, 220 + 50 = x → x = 270) : 
  M = 450 :=
by
  sorry

end find_maximum_marks_l7_7388


namespace xy_sum_of_squares_l7_7863

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 5) (h2 : -x * y = 4) : x^2 + y^2 = 17 := 
sorry

end xy_sum_of_squares_l7_7863


namespace range_of_a_l7_7722

def P (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def Q (x : ℝ) (a : ℝ) : Prop := x < a

theorem range_of_a (a : ℝ) : (∀ x, P x → Q x a) → (∀ x, Q x a → P x) → a ≥ 5 :=
by
  sorry

end range_of_a_l7_7722


namespace problem1_problem2_l7_7928

variable (α : ℝ) (tan_alpha_eq_three : Real.tan α = 3)

theorem problem1 : (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 :=
by sorry

theorem problem2 : Real.sin α * Real.cos α = 3 / 10 :=
by sorry

end problem1_problem2_l7_7928


namespace find_C_l7_7936

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 360) : C = 60 := by
  sorry

end find_C_l7_7936


namespace left_side_value_l7_7813

-- Define the relevant variables and conditions
variable (L R B : ℕ)

-- Assuming conditions
def sum_of_sides (L R B : ℕ) : Prop := L + R + B = 50
def right_side_relation (L R : ℕ) : Prop := R = L + 2
def base_value (B : ℕ) : Prop := B = 24

-- Main theorem statement
theorem left_side_value (L R B : ℕ) (h1 : sum_of_sides L R B) (h2 : right_side_relation L R) (h3 : base_value B) : L = 12 :=
sorry

end left_side_value_l7_7813


namespace midpoint_sum_eq_six_l7_7050

theorem midpoint_sum_eq_six :
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  (midpoint_x + midpoint_y) = 6 :=
by
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  sorry

end midpoint_sum_eq_six_l7_7050


namespace Sammy_has_8_bottle_caps_l7_7154

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l7_7154


namespace cubic_coefficient_determination_l7_7217

def f (x : ℚ) (A B C D : ℚ) : ℚ := A*x^3 + B*x^2 + C*x + D

theorem cubic_coefficient_determination {A B C D : ℚ}
  (h1 : f 1 A B C D = 0)
  (h2 : f (2/3) A B C D = -4)
  (h3 : f (4/5) A B C D = -16/5) :
  A = 15 ∧ B = -37 ∧ C = 30 ∧ D = -8 :=
  sorry

end cubic_coefficient_determination_l7_7217


namespace incorrect_statement_trajectory_of_P_l7_7545

noncomputable def midpoint_of_points (x1 x2 y1 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

theorem incorrect_statement_trajectory_of_P (p k x0 y0 : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (hA : A.1 * A.1 + 2 * p * A.2 = 0)
    (hB : B.1 * B.1 + 2 * p * B.2 = 0)
    (hMid : (x0, y0) = midpoint_of_points A.1 B.1 A.2 B.2)
    (hLine : A.2 = k * (A.1 - p / 2))
    (hLineIntersection : B.2 = k * (B.1 - p / 2)) : y0 ^ 2 ≠ 4 * p * (x0 - p / 2) :=
by
  sorry

end incorrect_statement_trajectory_of_P_l7_7545


namespace harry_total_expenditure_l7_7985

theorem harry_total_expenditure :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_packets := 3
  let tomato_packets := 4
  let chili_pepper_packets := 5
  (pumpkin_packets * pumpkin_price) + (tomato_packets * tomato_price) + (chili_pepper_packets * chili_pepper_price) = 18.00 :=
by
  sorry

end harry_total_expenditure_l7_7985


namespace num_five_letter_words_correct_l7_7846

-- Define the number of letters in the alphabet
def num_letters : ℕ := 26

-- Define the number of vowels
def num_vowels : ℕ := 5

-- Define a function that calculates the number of valid five-letter words
def num_five_letter_words : ℕ :=
  num_letters * num_vowels * num_letters * num_letters

-- The theorem statement we need to prove
theorem num_five_letter_words_correct : num_five_letter_words = 87700 :=
by
  -- The proof is omitted; it should equate the calculated value to 87700
  sorry

end num_five_letter_words_correct_l7_7846


namespace time_until_heavy_lifting_l7_7729

-- Define the conditions given
def pain_subside_days : ℕ := 3
def healing_multiplier : ℕ := 5
def additional_wait_days : ℕ := 3
def weeks_before_lifting : ℕ := 3
def days_in_week : ℕ := 7

-- Define the proof statement
theorem time_until_heavy_lifting : 
    let full_healing_days := pain_subside_days * healing_multiplier
    let total_days_before_exercising := full_healing_days + additional_wait_days
    let lifting_wait_days := weeks_before_lifting * days_in_week
    total_days_before_exercising + lifting_wait_days = 39 := 
by
  sorry

end time_until_heavy_lifting_l7_7729


namespace time_against_current_l7_7197

-- Define the conditions:
def swimming_speed_still_water : ℝ := 6  -- Speed in still water (km/h)
def current_speed : ℝ := 2  -- Speed of the water current (km/h)
def time_with_current : ℝ := 3.5  -- Time taken to swim with the current (hours)

-- Define effective speeds:
def effective_speed_against_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water - current_speed

def effective_speed_with_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water + current_speed

-- Calculate the distance covered with the current:
def distance_with_current (time_with_current effective_speed_with_current: ℝ) : ℝ :=
  time_with_current * effective_speed_with_current

-- Define the proof goal:
theorem time_against_current (h1 : swimming_speed_still_water = 6) (h2 : current_speed = 2)
  (h3 : time_with_current = 3.5) :
  ∃ (t : ℝ), t = 7 := by
  sorry

end time_against_current_l7_7197


namespace average_marks_first_class_l7_7117

theorem average_marks_first_class
  (n1 n2 : ℕ)
  (avg2 : ℝ)
  (combined_avg : ℝ)
  (h_n1 : n1 = 35)
  (h_n2 : n2 = 55)
  (h_avg2 : avg2 = 65)
  (h_combined_avg : combined_avg = 57.22222222222222) :
  (∃ avg1 : ℝ, avg1 = 45) :=
by
  sorry

end average_marks_first_class_l7_7117


namespace find_m_if_polynomial_is_perfect_square_l7_7829

theorem find_m_if_polynomial_is_perfect_square (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = x^2 + m * x + 4) → (m = 4 ∨ m = -4) :=
sorry

end find_m_if_polynomial_is_perfect_square_l7_7829


namespace total_fruits_in_bowl_l7_7916

theorem total_fruits_in_bowl (bananas apples oranges : ℕ) 
  (h1 : bananas = 2) 
  (h2 : apples = 2 * bananas) 
  (h3 : oranges = 6) : 
  bananas + apples + oranges = 12 := 
by 
  sorry

end total_fruits_in_bowl_l7_7916


namespace union_of_setA_and_setB_l7_7252

def setA : Set ℕ := {1, 2, 4}
def setB : Set ℕ := {2, 6}

theorem union_of_setA_and_setB :
  setA ∪ setB = {1, 2, 4, 6} :=
by sorry

end union_of_setA_and_setB_l7_7252


namespace students_still_inward_l7_7242

theorem students_still_inward (num_students : ℕ) (turns : ℕ) : (num_students = 36) ∧ (turns = 36) → ∃ n, n = 26 :=
by
  sorry

end students_still_inward_l7_7242


namespace smallest_N_divisibility_l7_7017

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l7_7017


namespace age_difference_l7_7181

theorem age_difference (A B C : ℕ) (hB : B = 14) (hBC : B = 2 * C) (hSum : A + B + C = 37) : A - B = 2 :=
by
  sorry

end age_difference_l7_7181


namespace probability_no_3x3_red_square_l7_7328

def grid_probability (m n : ℕ) : Prop :=
  (gcd m n = 1) ∧ ((m : ℚ) / n = 170 / 171)

theorem probability_no_3x3_red_square (m n : ℕ) (h1 : grid_probability m n) : m + n = 341 :=
by
  sorry

end probability_no_3x3_red_square_l7_7328


namespace bird_families_left_l7_7810

theorem bird_families_left (B_initial B_flew_away : ℕ) (h_initial : B_initial = 41) (h_flew_away : B_flew_away = 27) :
  B_initial - B_flew_away = 14 :=
by
  sorry

end bird_families_left_l7_7810


namespace sum_mean_median_mode_l7_7509

theorem sum_mean_median_mode : 
  let data := [2, 5, 1, 5, 2, 6, 1, 5, 0, 2]
  let ordered_data := [0, 1, 1, 2, 2, 2, 5, 5, 5, 6]
  let mean := (0 + 1 + 1 + 2 + 2 + 2 + 5 + 5 + 5 + 6) / 10
  let median := (2 + 2) / 2
  let mode := 5
  mean + median + mode = 9.9 := by
  sorry

end sum_mean_median_mode_l7_7509


namespace units_digit_17_pow_39_l7_7814

theorem units_digit_17_pow_39 : 
  ∃ d : ℕ, d < 10 ∧ (17^39 % 10 = d) ∧ d = 3 :=
by
  sorry

end units_digit_17_pow_39_l7_7814


namespace heaviest_lightest_difference_l7_7066

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end heaviest_lightest_difference_l7_7066


namespace find_d_l7_7131

theorem find_d (A B C D : ℕ) (h1 : (A + B + C) / 3 = 130) (h2 : (A + B + C + D) / 4 = 126) : D = 114 :=
by
  sorry

end find_d_l7_7131


namespace total_cost_of_selling_watermelons_l7_7964

-- Definitions of the conditions:
def watermelon_weight : ℝ := 23.0
def daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def number_of_watermelons : ℕ := 18

-- The theorem statement:
theorem total_cost_of_selling_watermelons :
  let average_price := (daily_prices.sum / daily_prices.length)
  let total_weight := number_of_watermelons * watermelon_weight
  let initial_cost := total_weight * average_price
  let discounted_cost := if number_of_watermelons > discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  final_cost = 796.43 := by
    sorry

end total_cost_of_selling_watermelons_l7_7964


namespace expression_odd_if_p_q_odd_l7_7450

variable (p q : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem expression_odd_if_p_q_odd (hp : is_odd p) (hq : is_odd q) : is_odd (5 * p * q) :=
sorry

end expression_odd_if_p_q_odd_l7_7450


namespace max_black_cells_1000_by_1000_l7_7982

def maxBlackCells (m n : ℕ) : ℕ :=
  if m = 1 then n else if n = 1 then m else m + n - 2

theorem max_black_cells_1000_by_1000 : maxBlackCells 1000 1000 = 1998 :=
  by sorry

end max_black_cells_1000_by_1000_l7_7982


namespace short_pencil_cost_l7_7408

theorem short_pencil_cost (x : ℝ)
  (h1 : 200 * 0.8 + 40 * 0.5 + 35 * x = 194) : x = 0.4 :=
by {
  sorry
}

end short_pencil_cost_l7_7408


namespace relationship_between_a_b_c_l7_7137

variable (a b c : ℝ)
variable (h_a : a = 0.4 ^ 0.2)
variable (h_b : b = 0.4 ^ 0.6)
variable (h_c : c = 2.1 ^ 0.2)

-- Prove the relationship c > a > b
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l7_7137


namespace total_wage_l7_7333

theorem total_wage (work_days_A work_days_B : ℕ) (wage_A : ℕ) (total_wage : ℕ) 
  (h1 : work_days_A = 10) 
  (h2 : work_days_B = 15) 
  (h3 : wage_A = 1980)
  (h4 : (wage_A / (wage_A / (total_wage * 3 / 5))) = 3)
  : total_wage = 3300 :=
sorry

end total_wage_l7_7333


namespace find_expression_l7_7700

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l7_7700


namespace triangle_right_angle_l7_7711

theorem triangle_right_angle {A B C : ℝ} 
  (h1 : A + B + C = 180)
  (h2 : A = B)
  (h3 : A = (1/2) * C) :
  C = 90 :=
by 
  sorry

end triangle_right_angle_l7_7711


namespace father_age_38_l7_7321

variable (F S : ℕ)
variable (h1 : S = 14)
variable (h2 : F - 10 = 7 * (S - 10))

theorem father_age_38 : F = 38 :=
by
  sorry

end father_age_38_l7_7321


namespace odd_function_strictly_decreasing_l7_7962

noncomputable def f (x : ℝ) : ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom negative_condition (x : ℝ) (hx : x > 0) : f x < 0

theorem odd_function : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem strictly_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end odd_function_strictly_decreasing_l7_7962


namespace least_subtract_divisible_by_8_l7_7502

def least_subtracted_to_divisible_by (n : ℕ) (d : ℕ) : ℕ :=
  n % d

theorem least_subtract_divisible_by_8 (n : ℕ) (d : ℕ) (h : n = 964807) (h_d : d = 8) :
  least_subtracted_to_divisible_by n d = 7 :=
by
  sorry

end least_subtract_divisible_by_8_l7_7502


namespace geometric_sequence_smallest_n_l7_7340

def geom_seq (n : ℕ) (r : ℝ) (b₁ : ℝ) : ℝ := 
  b₁ * r^(n-1)

theorem geometric_sequence_smallest_n 
  (b₁ b₂ b₃ : ℝ) (r : ℝ)
  (h₁ : b₁ = 2)
  (h₂ : b₂ = 6)
  (h₃ : b₃ = 18)
  (h_seq : ∀ n, bₙ = geom_seq n r b₁) :
  ∃ n, n = 5 ∧ geom_seq n r 2 = 324 :=
by
  sorry

end geometric_sequence_smallest_n_l7_7340


namespace A_and_B_worked_together_for_5_days_before_A_left_the_job_l7_7148

noncomputable def workRate_A (W : ℝ) : ℝ := W / 20
noncomputable def workRate_B (W : ℝ) : ℝ := W / 12

noncomputable def combinedWorkRate (W : ℝ) : ℝ := workRate_A W + workRate_B W

noncomputable def workDoneTogether (x : ℝ) (W : ℝ) : ℝ := x * combinedWorkRate W
noncomputable def workDoneBy_B_Alone (W : ℝ) : ℝ := 3 * workRate_B W

theorem A_and_B_worked_together_for_5_days_before_A_left_the_job (W : ℝ) :
  ∃ x : ℝ, workDoneTogether x W + workDoneBy_B_Alone W = W ∧ x = 5 :=
by
  sorry

end A_and_B_worked_together_for_5_days_before_A_left_the_job_l7_7148


namespace daily_earning_r_l7_7636

theorem daily_earning_r :
  exists P Q R : ℝ, 
    (P + Q + R = 220) ∧
    (P + R = 120) ∧
    (Q + R = 130) ∧
    (R = 30) := 
by
  sorry

end daily_earning_r_l7_7636


namespace complement_union_example_l7_7862

open Set

theorem complement_union_example :
  ∀ (U A B : Set ℕ), 
  U = {1, 2, 3, 4, 5, 6, 7, 8} → 
  A = {1, 3, 5, 7} → 
  B = {2, 4, 5} → 
  (U \ (A ∪ B)) = {6, 8} := by 
  intros U A B hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_union_example_l7_7862


namespace garrison_reinforcement_l7_7245

theorem garrison_reinforcement (x : ℕ) (h1 : ∀ (n m p : ℕ), n * m = p → x = n - m) :
  (150 * (31 - x) = 450 * 5) → x = 16 :=
by sorry

end garrison_reinforcement_l7_7245


namespace dot_product_of_a_and_b_l7_7418

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (3, 7)

-- Define the dot product function
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- State the theorem
theorem dot_product_of_a_and_b : dot_product a b = -18 := by
  sorry

end dot_product_of_a_and_b_l7_7418


namespace triangle_angle_inradius_l7_7470

variable (A B C : ℝ) 
variable (a b c R : ℝ)

theorem triangle_angle_inradius 
    (h1: 0 < A ∧ A < Real.pi)
    (h2: a * Real.cos C + (1/2) * c = b)
    (h3: a = 1):

    A = Real.pi / 3 ∧ R ≤ Real.sqrt 3 / 6 := 
by
  sorry

end triangle_angle_inradius_l7_7470


namespace henry_initial_money_l7_7979

variable (x : ℤ)

theorem henry_initial_money : (x + 18 - 10 = 19) → x = 11 :=
by
  intro h
  sorry

end henry_initial_money_l7_7979


namespace zhang_hua_new_year_cards_l7_7678

theorem zhang_hua_new_year_cards (x y z : ℕ) 
  (h1 : Nat.lcm (Nat.lcm x y) z = 60)
  (h2 : Nat.gcd x y = 4)
  (h3 : Nat.gcd y z = 3) : 
  x = 4 ∨ x = 20 :=
by
  sorry

end zhang_hua_new_year_cards_l7_7678


namespace xy_value_l7_7831

theorem xy_value (x y : ℝ) (h1 : (x + y) / 3 = 1.222222222222222) : x + y = 3.666666666666666 :=
by
  sorry

end xy_value_l7_7831


namespace greater_number_is_84_l7_7611

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) : x = 84 :=
sorry

end greater_number_is_84_l7_7611


namespace largest_sum_is_sum3_l7_7593

-- Definitions of the individual sums given in the conditions
def sum1 : ℚ := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
def sum2 : ℚ := (1/4 : ℚ) - (1/6 : ℚ)
def sum3 : ℚ := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
def sum4 : ℚ := (1/4 : ℚ) - (1/8 : ℚ)
def sum5 : ℚ := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)

-- Theorem to prove that sum3 is the largest
theorem largest_sum_is_sum3 : sum3 = (5/12 : ℚ) ∧ sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := 
by 
  -- The proof would go here
  sorry

end largest_sum_is_sum3_l7_7593


namespace algorithm_outputs_min_value_l7_7844

theorem algorithm_outputs_min_value (a b c d : ℕ) :
  let m := a;
  let m := if b < m then b else m;
  let m := if c < m then c else m;
  let m := if d < m then d else m;
  m = min (min (min a b) c) d :=
by
  sorry

end algorithm_outputs_min_value_l7_7844


namespace average_percentage_revenue_fall_l7_7680

theorem average_percentage_revenue_fall
  (initial_revenue_A final_revenue_A : ℝ)
  (initial_revenue_B final_revenue_B : ℝ) (exchange_rate_B : ℝ)
  (initial_revenue_C final_revenue_C : ℝ) (exchange_rate_C : ℝ) :
  initial_revenue_A = 72.0 →
  final_revenue_A = 48.0 →
  initial_revenue_B = 20.0 →
  final_revenue_B = 15.0 →
  exchange_rate_B = 1.30 →
  initial_revenue_C = 6000.0 →
  final_revenue_C = 5500.0 →
  exchange_rate_C = 0.0091 →
  (33.33 + 25 + 8.33) / 3 = 22.22 :=
by
  sorry

end average_percentage_revenue_fall_l7_7680


namespace inequality_equivalence_l7_7635

theorem inequality_equivalence (a : ℝ) : a < -1 ↔ a + 1 < 0 :=
by
  sorry

end inequality_equivalence_l7_7635


namespace contrapositive_equivalent_l7_7654

variable {α : Type*} (A B : Set α) (x : α)

theorem contrapositive_equivalent : (x ∈ A → x ∈ B) ↔ (x ∉ B → x ∉ A) :=
by
  sorry

end contrapositive_equivalent_l7_7654


namespace coefficient_condition_l7_7567

theorem coefficient_condition (m : ℝ) (h : m^3 * Nat.choose 6 3 = -160) : m = -2 := sorry

end coefficient_condition_l7_7567


namespace december_fraction_of_yearly_sales_l7_7101

theorem december_fraction_of_yearly_sales (A : ℝ) (h_sales : ∀ (x : ℝ), x = 6 * A) :
    let yearly_sales := 11 * A + 6 * A
    let december_sales := 6 * A
    december_sales / yearly_sales = 6 / 17 := by
  sorry

end december_fraction_of_yearly_sales_l7_7101


namespace amount_collected_from_ii_and_iii_class_l7_7238

theorem amount_collected_from_ii_and_iii_class
  (P1 P2 P3 : ℕ) (F1 F2 F3 : ℕ) (total_amount amount_ii_iii : ℕ)
  (H1 : P1 / P2 = 1 / 50)
  (H2 : P1 / P3 = 1 / 100)
  (H3 : F1 / F2 = 5 / 2)
  (H4 : F1 / F3 = 5 / 1)
  (H5 : total_amount = 3575)
  (H6 : total_amount = (P1 * F1) + (P2 * F2) + (P3 * F3))
  (H7 : amount_ii_iii = (P2 * F2) + (P3 * F3)) :
  amount_ii_iii = 3488 := sorry

end amount_collected_from_ii_and_iii_class_l7_7238


namespace complex_multiplication_result_l7_7630

-- Define the complex numbers used in the problem
def a : ℂ := 4 - 3 * Complex.I
def b : ℂ := 4 + 3 * Complex.I

-- State the theorem we want to prove
theorem complex_multiplication_result : a * b = 25 := 
by
  -- Proof is omitted
  sorry

end complex_multiplication_result_l7_7630


namespace slant_asymptote_sum_l7_7649

theorem slant_asymptote_sum (m b : ℝ) 
  (h : ∀ x : ℝ, y = 3*x^2 + 4*x - 8 / (x - 4) → y = m*x + b) :
  m + b = 19 :=
sorry

end slant_asymptote_sum_l7_7649


namespace factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l7_7940

-- Math Proof Problem 1
theorem factorize_a_squared_minus_25 (a : ℝ) : a^2 - 25 = (a + 5) * (a - 5) :=
by
  sorry

-- Math Proof Problem 2
theorem factorize_2x_squared_y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 :=
by
  sorry

end factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l7_7940


namespace solve_inequality_and_find_positive_int_solutions_l7_7307

theorem solve_inequality_and_find_positive_int_solutions :
  ∀ (x : ℝ), (2 * x + 1) / 3 - 1 ≤ (2 / 5) * x → x ≤ 2.5 ∧ ∃ (n : ℕ), n = 1 ∨ n = 2 :=
by
  intro x
  intro h
  sorry

end solve_inequality_and_find_positive_int_solutions_l7_7307


namespace max_k_value_l7_7848

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k_value
    (h : ∀ (x : ℝ), 1 < x → f x > k * (x - 1)) :
    k = 3 := sorry

end max_k_value_l7_7848


namespace spider_total_distance_l7_7596

theorem spider_total_distance
    (radius : ℝ)
    (diameter : ℝ)
    (half_diameter : ℝ)
    (final_leg : ℝ)
    (total_distance : ℝ) :
    radius = 75 →
    diameter = 2 * radius →
    half_diameter = diameter / 2 →
    final_leg = 90 →
    (half_diameter ^ 2 + final_leg ^ 2 = diameter ^ 2) →
    total_distance = diameter + half_diameter + final_leg →
    total_distance = 315 :=
by
  intros
  sorry

end spider_total_distance_l7_7596


namespace cyclic_inequality_l7_7837

variables {a b c : ℝ}

theorem cyclic_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (ab / (a + b + 2 * c) + bc / (b + c + 2 * a) + ca / (c + a + 2 * b)) ≤ (a + b + c) / 4 :=
sorry

end cyclic_inequality_l7_7837


namespace min_n_for_constant_term_l7_7576

theorem min_n_for_constant_term (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), (2 * n = 5 * r) → n = 5 :=
by
  sorry

end min_n_for_constant_term_l7_7576


namespace inequality_correct_l7_7934

open BigOperators

theorem inequality_correct {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ (a + b)^2 / 4 ∧ (a + b)^2 / 4 ≥ a * b :=
by 
  sorry

end inequality_correct_l7_7934


namespace remaining_money_l7_7361

def initial_amount : Float := 499.9999999999999

def spent_on_clothes (initial : Float) : Float :=
  (1/3) * initial

def remaining_after_clothes (initial : Float) : Float :=
  initial - spent_on_clothes initial

def spent_on_food (remaining_clothes : Float) : Float :=
  (1/5) * remaining_clothes

def remaining_after_food (remaining_clothes : Float) : Float :=
  remaining_clothes - spent_on_food remaining_clothes

def spent_on_travel (remaining_food : Float) : Float :=
  (1/4) * remaining_food

def remaining_after_travel (remaining_food : Float) : Float :=
  remaining_food - spent_on_travel remaining_food

theorem remaining_money :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 199.99 :=
by
  sorry

end remaining_money_l7_7361


namespace binom_eight_five_l7_7648

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l7_7648


namespace sqrt_inequality_l7_7375

theorem sqrt_inequality (x : ℝ) (h : ∀ r : ℝ, r = 2 * x - 1 → r ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_inequality_l7_7375


namespace small_cuboid_length_is_five_l7_7166

-- Define initial conditions
def large_cuboid_length : ℝ := 18
def large_cuboid_width : ℝ := 15
def large_cuboid_height : ℝ := 2
def num_small_cuboids : ℕ := 6
def small_cuboid_width : ℝ := 6
def small_cuboid_height : ℝ := 3

-- Theorem to prove the length of the smaller cuboid
theorem small_cuboid_length_is_five (small_cuboid_length : ℝ) 
  (h1 : large_cuboid_length * large_cuboid_width * large_cuboid_height 
          = num_small_cuboids * (small_cuboid_length * small_cuboid_width * small_cuboid_height)) :
  small_cuboid_length = 5 := by
  sorry

end small_cuboid_length_is_five_l7_7166


namespace find_QS_l7_7087

theorem find_QS (RS QR QS : ℕ) (h1 : RS = 13) (h2 : QR = 5) (h3 : QR * 13 = 5 * 13) :
  QS = 12 :=
by
  sorry

end find_QS_l7_7087


namespace m_n_solution_l7_7402

theorem m_n_solution (m n : ℝ) (h1 : m - n = -5) (h2 : m^2 + n^2 = 13) : m^4 + n^4 = 97 :=
by
  sorry

end m_n_solution_l7_7402


namespace total_population_correct_l7_7090

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l7_7090


namespace multiples_of_4_between_200_and_500_l7_7515
-- Import the necessary library

open Nat

theorem multiples_of_4_between_200_and_500 : 
  ∃ n, n = (500 / 4 - 200 / 4) :=
by
  sorry

end multiples_of_4_between_200_and_500_l7_7515


namespace proof_problem_l7_7529

variable (a b c A B C : ℝ)
variable (h_a : a = Real.sqrt 3)
variable (h_b_ge_a : b ≥ a)
variable (h_cos : Real.cos (2 * C) - Real.cos (2 * A) =
  2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C))

theorem proof_problem :
  (A = Real.pi / 3) ∧ (2 * b - c ∈ Set.Ico (Real.sqrt 3) (2 * Real.sqrt 3)) :=
  sorry

end proof_problem_l7_7529


namespace bridget_heavier_than_martha_l7_7198

def bridget_weight := 39
def martha_weight := 2

theorem bridget_heavier_than_martha :
  bridget_weight - martha_weight = 37 :=
by
  sorry

end bridget_heavier_than_martha_l7_7198


namespace sequences_count_l7_7474

open BigOperators

def consecutive_blocks (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2 - 1) - 2

theorem sequences_count {n : ℕ} (h : n = 15) :
  consecutive_blocks n = 238 :=
by
  sorry

end sequences_count_l7_7474


namespace option_C_incorrect_l7_7851

structure Line := (point1 point2 : ℝ × ℝ × ℝ)
structure Plane := (point : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ)

variables (m n : Line) (α β : Plane)

def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def lines_parallel (l1 l2 : Line) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry
def planes_parallel (p1 p2 : Plane) : Prop := sorry

theorem option_C_incorrect 
  (h1 : line_in_plane m α)
  (h2 : line_parallel_to_plane n α)
  (h3 : lines_parallel m n) :
  false :=
sorry

end option_C_incorrect_l7_7851


namespace range_of_m_l7_7013

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x^(m-1) > y^(m-1)) → m < 1 :=
by
  sorry

end range_of_m_l7_7013


namespace round_trip_completion_percentage_l7_7726

-- Define the distances for each section
def sectionA_distance : Float := 10
def sectionB_distance : Float := 20
def sectionC_distance : Float := 15

-- Define the speeds for each section
def sectionA_speed : Float := 50
def sectionB_speed : Float := 40
def sectionC_speed : Float := 60

-- Define the delays for each section
def sectionA_delay : Float := 1.15
def sectionB_delay : Float := 1.10

-- Calculate the time for each section without delays
def sectionA_time : Float := sectionA_distance / sectionA_speed
def sectionB_time : Float := sectionB_distance / sectionB_speed
def sectionC_time : Float := sectionC_distance / sectionC_speed

-- Calculate the time with delays for the trip to the center
def sectionA_time_with_delay : Float := sectionA_time * sectionA_delay
def sectionB_time_with_delay : Float := sectionB_time * sectionB_delay
def sectionC_time_with_delay : Float := sectionC_time

-- Total time with delays to the center
def total_time_to_center : Float := sectionA_time_with_delay + sectionB_time_with_delay + sectionC_time_with_delay

-- Total distance to the center
def total_distance_to_center : Float := sectionA_distance + sectionB_distance + sectionC_distance

-- Total round trip distance
def total_round_trip_distance : Float := total_distance_to_center * 2

-- Distance covered on the way back
def distance_back : Float := total_distance_to_center * 0.2

-- Total distance covered considering the delays and the return trip
def total_distance_covered : Float := total_distance_to_center + distance_back

-- Effective completion percentage of the round trip
def completion_percentage : Float := (total_distance_covered / total_round_trip_distance) * 100

-- The main theorem statement
theorem round_trip_completion_percentage :
  completion_percentage = 60 := by
  sorry

end round_trip_completion_percentage_l7_7726


namespace feifei_reaches_school_at_828_l7_7029

-- Definitions for all conditions
def start_time : Nat := 8 * 60 + 10  -- Feifei starts walking at 8:10 AM in minutes since midnight
def dog_delay : Nat := 3             -- Dog starts chasing after 3 minutes
def catch_up_200m_time : ℕ := 1      -- Time for dog to catch Feifei at 200 meters
def catch_up_400m_time : ℕ := 4      -- Time for dog to catch Feifei at 400 meters
def school_distance : ℕ := 800       -- Distance from home to school
def feifei_speed : ℕ := 2            -- assumed speed of Feifei where distance covered uniformly
def dog_speed : ℕ := 6               -- dog speed is three times Feifei's speed
def catch_times := [200, 400, 800]   -- Distances (in meters) where dog catches Feifei

-- Derived condition:
def total_travel_time : ℕ := 
  let time_for_200m := catch_up_200m_time + catch_up_200m_time;
  let time_for_400m_and_back := 2* catch_up_400m_time ;
  (time_for_200m + time_for_400m_and_back + (school_distance - 400))

-- The statement we wish to prove:
theorem feifei_reaches_school_at_828 : 
  (start_time + total_travel_time - dog_delay/2) % 60 = 28 :=
sorry

end feifei_reaches_school_at_828_l7_7029


namespace farmer_goats_l7_7568

theorem farmer_goats (cows sheep goats : ℕ) (extra_goats : ℕ) 
(hcows : cows = 7) (hsheep : sheep = 8) (hgoats : goats = 6) 
(h : (goats + extra_goats = (cows + sheep + goats + extra_goats) / 2)) : 
extra_goats = 9 := by
  sorry

end farmer_goats_l7_7568


namespace find_m_l7_7599

theorem find_m (m : ℝ) (h₁: 0 < m) (h₂: ∀ p q : ℝ × ℝ, p = (m, 4) → q = (2, m) → ∃ s : ℝ, s = m^2 ∧ ((q.2 - p.2) / (q.1 - p.1)) = s) : m = 2 :=
by
  sorry

end find_m_l7_7599


namespace total_fish_l7_7615

theorem total_fish {lilly_fish rosy_fish : ℕ} (h1 : lilly_fish = 10) (h2 : rosy_fish = 11) : 
lilly_fish + rosy_fish = 21 :=
by 
  sorry

end total_fish_l7_7615


namespace inscribed_squares_ratio_l7_7376

theorem inscribed_squares_ratio (x y : ℝ) (h1 : ∃ (x : ℝ), x * (13 * 12 + 13 * 5 - 5 * 12) = 60) 
  (h2 : ∃ (y : ℝ), 30 * y = 13 ^ 2) :
  x / y = 1800 / 2863 := 
sorry

end inscribed_squares_ratio_l7_7376


namespace sin_theta_add_pi_over_3_l7_7225

theorem sin_theta_add_pi_over_3 (θ : ℝ) (h : Real.cos (π / 6 - θ) = 2 / 3) : 
  Real.sin (θ + π / 3) = 2 / 3 :=
sorry

end sin_theta_add_pi_over_3_l7_7225


namespace compute_difference_of_squares_l7_7008

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l7_7008


namespace sum_of_squares_of_ages_eq_35_l7_7742

theorem sum_of_squares_of_ages_eq_35
  (d t h : ℕ)
  (h1 : 3 * d + 4 * t = 2 * h + 2)
  (h2 : 2 * d^2 + t^2 = 6 * h)
  (relatively_prime : Nat.gcd (Nat.gcd d t) h = 1) :
  d^2 + t^2 + h^2 = 35 := 
sorry

end sum_of_squares_of_ages_eq_35_l7_7742


namespace R_depends_on_a_d_m_l7_7014

theorem R_depends_on_a_d_m (a d m : ℝ) :
    let s1 := (m / 2) * (2 * a + (m - 1) * d)
    let s2 := m * (2 * a + (2 * m - 1) * d)
    let s3 := 2 * m * (2 * a + (4 * m - 1) * d)
    let R := s3 - 2 * s2 + s1
    R = m * (a + 12 * m * d - (d / 2)) := by
  sorry

end R_depends_on_a_d_m_l7_7014


namespace prob_two_red_balls_in_four_draws_l7_7201

noncomputable def probability_red_balls (draws : ℕ) (red_in_draw : ℕ) (total_balls : ℕ) (red_balls : ℕ) : ℝ :=
  let prob_red := (red_balls : ℝ) / (total_balls : ℝ)
  let prob_white := 1 - prob_red
  (Nat.choose draws red_in_draw : ℝ) * (prob_red ^ red_in_draw) * (prob_white ^ (draws - red_in_draw))

theorem prob_two_red_balls_in_four_draws :
  probability_red_balls 4 2 10 4 = 0.3456 :=
by
  sorry

end prob_two_red_balls_in_four_draws_l7_7201


namespace adjusted_ratio_l7_7176

theorem adjusted_ratio :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 :=
by
  sorry

end adjusted_ratio_l7_7176


namespace profit_difference_l7_7618

variable (a_capital b_capital c_capital b_profit : ℕ)

theorem profit_difference (h₁ : a_capital = 8000) (h₂ : b_capital = 10000) 
                          (h₃ : c_capital = 12000) (h₄ : b_profit = 2000) : 
  c_capital * (b_profit / b_capital) - a_capital * (b_profit / b_capital) = 800 := 
sorry

end profit_difference_l7_7618


namespace find_x_l7_7381

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l7_7381


namespace simplify_and_evaluate_expression_l7_7544

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -6) : 
  (1 - a / (a - 3)) / ((a^2 + 3 * a) / (a^2 - 9)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l7_7544


namespace group_capacity_l7_7747

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end group_capacity_l7_7747


namespace simplify_fraction_eq_l7_7133

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l7_7133


namespace white_squares_95th_figure_l7_7752

theorem white_squares_95th_figure : ∀ (T : ℕ → ℕ),
  T 1 = 8 →
  (∀ n ≥ 1, T (n + 1) = T n + 5) →
  T 95 = 478 :=
by
  intros T hT1 hTrec
  -- Skipping the proof
  sorry

end white_squares_95th_figure_l7_7752


namespace product_of_x_y_l7_7589

variable (x y : ℝ)

-- Condition: EF = GH
def EF_eq_GH := (x^2 + 2 * x - 8 = 45)

-- Condition: FG = EH
def FG_eq_EH := (y^2 + 8 * y + 16 = 36)

-- Condition: y > 0
def y_pos := (y > 0)

theorem product_of_x_y : EF_eq_GH x ∧ FG_eq_EH y ∧ y_pos y → 
  x * y = -2 + 6 * Real.sqrt 6 :=
sorry

end product_of_x_y_l7_7589


namespace steven_needs_more_seeds_l7_7895

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l7_7895


namespace num_new_students_l7_7128

-- Definitions based on the provided conditions
def original_class_strength : ℕ := 10
def original_average_age : ℕ := 40
def new_students_avg_age : ℕ := 32
def decrease_in_average_age : ℕ := 4
def new_average_age : ℕ := original_average_age - decrease_in_average_age
def new_class_strength (n : ℕ) : ℕ := original_class_strength + n

-- The proof statement
theorem num_new_students (n : ℕ) :
  (original_class_strength * original_average_age + n * new_students_avg_age) 
  = new_class_strength n * new_average_age → n = 10 :=
by
  sorry

end num_new_students_l7_7128


namespace num_workers_l7_7597

-- Define the number of workers (n) and the initial contribution per worker (x)
variable (n x : ℕ)

-- Condition 1: The total contribution is Rs. 3 lacs
axiom h1 : n * x = 300000

-- Condition 2: If each worker contributed Rs. 50 more, the total would be Rs. 3.75 lacs
axiom h2 : n * (x + 50) = 375000

-- Proof Problem: Prove that the number of workers (n) is 1500
theorem num_workers : n = 1500 :=
by
  -- The proof will go here
  sorry

end num_workers_l7_7597


namespace correct_formulas_l7_7634

theorem correct_formulas (n : ℕ) :
  ((2 * n - 1)^2 - 4 * (n * (n - 1)) / 2) = (2 * n^2 - 2 * n + 1) ∧ 
  (1 + ((n - 1) * n) / 2 * 4) = (2 * n^2 - 2 * n + 1) ∧ 
  ((n - 1)^2 + n^2) = (2 * n^2 - 2 * n + 1) := by
  sorry

end correct_formulas_l7_7634


namespace solve_system_l7_7608

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) :=
  x ≠ y ∧
  a ≠ 0 ∧
  c ≠ 0 ∧
  (x + z) * a = x - y ∧
  (x + z) * b = x^2 - y^2 ∧
  (x + z)^2 * (b^2 / (a^2 * c)) = (x^3 + x^2 * y - x * y^2 - y^3)

-- Proof goal: establish the values of x, y, and z
theorem solve_system (a b c x y z : ℝ) (h : system_of_equations a b c x y z):
  x = (a^3 * c + b) / (2 * a) ∧
  y = (b - a^3 * c) / (2 * a) ∧
  z = (2 * a^2 * c - a^3 * c - b) / (2 * a) :=
by
  sorry

end solve_system_l7_7608


namespace no_snow_three_days_l7_7277

noncomputable def probability_no_snow_first_two_days : ℚ := 1 - 2/3
noncomputable def probability_no_snow_third_day : ℚ := 1 - 3/5

theorem no_snow_three_days : 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_third_day) = 2/45 :=
by
  sorry

end no_snow_three_days_l7_7277


namespace determine_days_l7_7344

-- Define the problem
def team_repair_time (x y : ℕ) : Prop :=
  ((1 / (x:ℝ)) + (1 / (y:ℝ)) = 1 / 18) ∧ 
  ((2 / 3 * x + 1 / 3 * y = 40))

theorem determine_days : ∃ x y : ℕ, team_repair_time x y :=
by
    use 45
    use 30
    have h1: (1/(45:ℝ) + 1/(30:ℝ)) = 1/18 := by
        sorry
    have h2: (2/3*45 + 1/3*30 = 40) := by
        sorry 
    exact ⟨h1, h2⟩

end determine_days_l7_7344


namespace correct_options_l7_7536

open Real

def option_A (x : ℝ) : Prop :=
  x^2 - 2*x + 1 > 0

def option_B : Prop :=
  ∃ (x : ℝ), (0 < x) ∧ (x + 4 / x = 6)

def option_C (a b : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) → (b / a + a / b ≥ 2)

def option_D (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (x + 2*y = 1) → (2 / x + 1 / y ≥ 8)

theorem correct_options :
  ¬(∀ (x : ℝ), option_A x) ∧ (option_B ∧ (∀ (a b : ℝ), option_C a b) = false ∧ 
  (∀ (x y : ℝ), option_D x y)) :=
by sorry

end correct_options_l7_7536


namespace range_of_a_l7_7906

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_a_l7_7906


namespace greg_age_is_18_l7_7486

def diana_age : ℕ := 15
def eduardo_age (c : ℕ) : ℕ := 2 * c
def chad_age (c : ℕ) : ℕ := c
def faye_age (c : ℕ) : ℕ := c - 1
def greg_age (c : ℕ) : ℕ := 2 * (c - 1)
def diana_relation (c : ℕ) : Prop := 15 = (2 * c) - 5

theorem greg_age_is_18 (c : ℕ) (h : diana_relation c) :
  greg_age c = 18 :=
by
  sorry

end greg_age_is_18_l7_7486


namespace backpacks_weight_l7_7588

variables (w_y w_g : ℝ)

theorem backpacks_weight :
  (2 * w_y + 3 * w_g = 44) ∧
  (w_y + w_g + w_g / 2 = w_g + w_y / 2) →
  (w_g = 4) ∧ (w_y = 12) :=
by
  intros h
  sorry

end backpacks_weight_l7_7588


namespace speed_of_other_train_l7_7980

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l7_7980


namespace A_inter_B_l7_7947

-- Define the sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := { abs x | x ∈ A }

-- Statement of the theorem to be proven
theorem A_inter_B :
  A ∩ B = {0, 2} := 
by 
  sorry

end A_inter_B_l7_7947


namespace average_of_possible_values_of_x_l7_7843

theorem average_of_possible_values_of_x (x : ℝ) (h : (2 * x^2 + 3) = 21) : (x = 3 ∨ x = -3) → (3 + -3) / 2 = 0 := by
  sorry

end average_of_possible_values_of_x_l7_7843


namespace find_absolute_value_l7_7051

theorem find_absolute_value (h k : ℤ) (h1 : 3 * (-3)^3 - h * (-3) + k = 0) (h2 : 3 * 2^3 - h * 2 + k = 0) : |3 * h - 2 * k| = 27 :=
by
  sorry

end find_absolute_value_l7_7051


namespace age_ratio_l7_7857

-- Definitions of the ages based on the given conditions.
def Rachel_age : ℕ := 12  -- Rachel's age
def Father_age_when_Rachel_25 : ℕ := 60

-- Defining Mother, Father, Grandfather ages based on given conditions.
def Grandfather_age (R : ℕ) (F : ℕ) : ℕ := 2 * (F - 5)
def Father_age (R : ℕ) : ℕ := Father_age_when_Rachel_25 - (25 - R)

-- Proving the ratio of Grandfather's age to Rachel's age is 7:1
theorem age_ratio (R : ℕ) (F : ℕ) (G : ℕ) :
  R = Rachel_age →
  F = Father_age R →
  G = Grandfather_age R F →
  G / R = 7 := by
  exact sorry

end age_ratio_l7_7857


namespace find_d_l7_7281

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x - 3

theorem find_d (c d : ℝ) (h : ∀ x, f (g x c) c = 15 * x + d) : d = -12 :=
by
  have h1 : ∀ x, f (g x c) c = 5 * (c * x - 3) + c := by intros; simp [f, g]
  have h2 : ∀ x, 5 * (c * x - 3) + c = 5 * c * x + c - 15 := by intros; ring
  specialize h 0
  rw [h1, h2] at h
  sorry

end find_d_l7_7281


namespace smallest_portion_quantity_l7_7420

-- Define the conditions for the problem
def conditions (a1 a2 a3 a4 a5 d : ℚ) : Prop :=
  a2 = a1 + d ∧
  a3 = a1 + 2 * d ∧
  a4 = a1 + 3 * d ∧
  a5 = a1 + 4 * d ∧
  5 * a1 + 10 * d = 100 ∧
  (a3 + a4 + a5) = (1/7) * (a1 + a2)

-- Lean theorem statement
theorem smallest_portion_quantity : 
  ∃ (a1 a2 a3 a4 a5 d : ℚ), conditions a1 a2 a3 a4 a5 d ∧ a1 = 5 / 3 :=
by
  sorry

end smallest_portion_quantity_l7_7420


namespace angles_in_order_l7_7264

-- α1, α2, α3 are real numbers representing the angles of inclination of lines
variable (α1 α2 α3 : ℝ)

-- Conditions given in the problem
axiom tan_α1 : Real.tan α1 = 1
axiom tan_α2 : Real.tan α2 = -1
axiom tan_α3 : Real.tan α3 = -2

-- Theorem to prove
theorem angles_in_order : α1 < α3 ∧ α3 < α2 := 
by
  sorry

end angles_in_order_l7_7264


namespace reflected_line_equation_l7_7404

def line_reflection_about_x_axis (x y : ℝ) : Prop :=
  x - y + 1 = 0 → y = -x - 1

theorem reflected_line_equation :
  ∀ (x y : ℝ), x - y + 1 = 0 → x + y + 1 = 0 :=
by
  intros x y h
  suffices y = -x - 1 by
    linarith
  sorry

end reflected_line_equation_l7_7404


namespace surface_area_of_sphere_l7_7369

noncomputable def sphere_surface_area : ℝ :=
  let AB := 2
  let SA := 2
  let SB := 2
  let SC := 2
  let ABC_is_isosceles_right := true -- denotes the property
  let SABC_on_sphere := true -- denotes the property
  let R := (2 * Real.sqrt 3) / 3
  let surface_area := 4 * Real.pi * R^2
  surface_area

theorem surface_area_of_sphere : sphere_surface_area = (16 * Real.pi) / 3 := 
sorry

end surface_area_of_sphere_l7_7369


namespace plates_difference_l7_7847

noncomputable def num_pots_angela : ℕ := 20
noncomputable def num_plates_angela (P : ℕ) := P
noncomputable def num_cutlery_angela (P : ℕ) := P / 2
noncomputable def num_pots_sharon : ℕ := 10
noncomputable def num_plates_sharon (P : ℕ) := 3 * P - 20
noncomputable def num_cutlery_sharon (P : ℕ) := P
noncomputable def total_kitchen_supplies_sharon (P : ℕ) := 
  num_pots_sharon + num_plates_sharon P + num_cutlery_sharon P

theorem plates_difference (P : ℕ) 
  (hP: num_plates_angela P > 3 * num_pots_angela) 
  (h_supplies: total_kitchen_supplies_sharon P = 254) :
  P - 3 * num_pots_angela = 6 := 
sorry

end plates_difference_l7_7847


namespace students_left_is_6_l7_7976

-- Start of the year students
def initial_students : ℕ := 11

-- New students arrived during the year
def new_students : ℕ := 42

-- Students at the end of the year
def final_students : ℕ := 47

-- Definition to calculate the number of students who left
def students_left (initial new final : ℕ) : ℕ := (initial + new) - final

-- Statement to prove
theorem students_left_is_6 : students_left initial_students new_students final_students = 6 :=
by
  -- We skip the proof using sorry
  sorry

end students_left_is_6_l7_7976


namespace quadratic_polynomial_coefficients_l7_7524

theorem quadratic_polynomial_coefficients (a b : ℝ)
  (h1 : 2 * a - 1 - b = 0)
  (h2 : 5 * a + b - 13 = 0) :
  a^2 + b^2 = 13 := 
by 
  sorry

end quadratic_polynomial_coefficients_l7_7524


namespace jimmy_fill_pool_time_l7_7115

theorem jimmy_fill_pool_time (pool_gallons : ℕ) (bucket_gallons : ℕ) (time_per_trip_sec : ℕ) (sec_per_min : ℕ) :
  pool_gallons = 84 → 
  bucket_gallons = 2 → 
  time_per_trip_sec = 20 → 
  sec_per_min = 60 → 
  (pool_gallons / bucket_gallons) * time_per_trip_sec / sec_per_min = 14 :=
by
  sorry

end jimmy_fill_pool_time_l7_7115


namespace depak_bank_account_l7_7820

theorem depak_bank_account :
  ∃ (n : ℕ), (x + 1 = 6 * n) ∧ n = 1 → x = 5 := 
sorry

end depak_bank_account_l7_7820


namespace number_of_students_l7_7717

-- Definitions based on problem conditions
def age_condition (a n : ℕ) : Prop :=
  7 * (a - 1) + 2 * (a + 2) + (n - 9) * a = 330

-- Main theorem to prove the correct number of students
theorem number_of_students (a n : ℕ) (h : age_condition a n) : n = 37 :=
  sorry

end number_of_students_l7_7717


namespace consecutive_odd_sum_l7_7371

theorem consecutive_odd_sum (n : ℤ) (h : n + 2 = 9) : 
  let a := n
  let b := n + 2
  let c := n + 4
  (a + b + c) = a + 20 := by
  sorry

end consecutive_odd_sum_l7_7371


namespace circle_center_l7_7948

theorem circle_center :
  ∃ c : ℝ × ℝ, c = (-1, 3) ∧ ∀ (x y : ℝ), (4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 96 = 0 ↔ (x + 1)^2 + (y - 3)^2 = 14) :=
by
  sorry

end circle_center_l7_7948


namespace find_a_plus_b_plus_c_l7_7824

noncomputable def parabola_satisfies_conditions (a b c : ℝ) : Prop :=
  (∀ x, a * x ^ 2 + b * x + c ≥ 61) ∧
  (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = 0) ∧
  (a * (3:ℝ) ^ 2 + b * (3:ℝ) + c = 0)

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_minimum : parabola_satisfies_conditions a b c) :
  a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l7_7824


namespace donut_selection_l7_7075

theorem donut_selection :
  ∃ (ways : ℕ), ways = Nat.choose 8 3 ∧ ways = 56 :=
by
  sorry

end donut_selection_l7_7075


namespace sue_receives_correct_answer_l7_7332

theorem sue_receives_correct_answer (x : ℕ) (y : ℕ) (z : ℕ) (h1 : y = 3 * (x + 2)) (h2 : z = 3 * (y - 2)) (hx : x = 6) : z = 66 :=
by
  sorry

end sue_receives_correct_answer_l7_7332


namespace relationship_among_x_y_z_w_l7_7445

theorem relationship_among_x_y_z_w (x y z w : ℝ) (h : (x + y) / (y + z) = (z + w) / (w + x)) :
  x = z ∨ x + y + w + z = 0 :=
sorry

end relationship_among_x_y_z_w_l7_7445


namespace manager_final_price_l7_7743

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_price : ℝ := wholesale_cost + 0.2 * wholesale_cost
noncomputable def manager_discount : ℝ := 0.1 * retail_price
noncomputable def price_after_manager_discount : ℝ := retail_price - manager_discount
noncomputable def weekend_sale_discount : ℝ := 0.1 * price_after_manager_discount
noncomputable def price_after_weekend_sale : ℝ := price_after_manager_discount - weekend_sale_discount
noncomputable def sales_tax : ℝ := 0.08 * price_after_weekend_sale
noncomputable def total_price : ℝ := price_after_weekend_sale + sales_tax

theorem manager_final_price : total_price = 209.95 := by
  sorry

end manager_final_price_l7_7743


namespace expected_value_of_winnings_is_5_l7_7978

namespace DiceGame

def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 2 * roll else 0

noncomputable def expectedValue : ℚ :=
  (winnings 2 + winnings 4 + winnings 6 + winnings 8) / 8

theorem expected_value_of_winnings_is_5 :
  expectedValue = 5 := by
  sorry

end DiceGame

end expected_value_of_winnings_is_5_l7_7978


namespace passengers_final_count_l7_7728

structure BusStop :=
  (initial_passengers : ℕ)
  (first_stop_increase : ℕ)
  (other_stops_decrease : ℕ)
  (other_stops_increase : ℕ)

def passengers_at_last_stop (b : BusStop) : ℕ :=
  b.initial_passengers + b.first_stop_increase - b.other_stops_decrease + b.other_stops_increase

theorem passengers_final_count :
  passengers_at_last_stop ⟨50, 16, 22, 5⟩ = 49 := by
  rfl

end passengers_final_count_l7_7728


namespace handshake_problem_l7_7994

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem handshake_problem : combinations 40 2 = 780 := 
by
  sorry

end handshake_problem_l7_7994


namespace smallest_n_for_three_pairs_l7_7476

theorem smallest_n_for_three_pairs :
  ∃ (n : ℕ), (0 < n) ∧
    (∀ (x y : ℕ), (x^2 - y^2 = n) → (0 < x) ∧ (0 < y)) ∧
    (∃ (a b c : ℕ), 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
      (∃ (x y : ℕ), (x^2 - y^2 = n) ∧
        (((x, y) = (a, b)) ∨ ((x, y) = (b, c)) ∨ ((x, y) = (a, c))))) :=
sorry

end smallest_n_for_three_pairs_l7_7476


namespace mrs_hilt_total_spent_l7_7118

def kids_ticket_usual_cost : ℕ := 1 -- $1 for 4 tickets
def adults_ticket_usual_cost : ℕ := 2 -- $2 for 3 tickets

def kids_ticket_deal_cost : ℕ := 4 -- $4 for 20 tickets
def adults_ticket_deal_cost : ℕ := 8 -- $8 for 15 tickets

def kids_tickets_purchased : ℕ := 24
def adults_tickets_purchased : ℕ := 18

def total_kids_ticket_cost : ℕ :=
  let kids_deal_tickets := kids_ticket_deal_cost
  let remaining_kids_tickets := kids_ticket_usual_cost
  kids_deal_tickets + remaining_kids_tickets

def total_adults_ticket_cost : ℕ :=
  let adults_deal_tickets := adults_ticket_deal_cost
  let remaining_adults_tickets := adults_ticket_usual_cost
  adults_deal_tickets + remaining_adults_tickets

def total_cost (kids_cost adults_cost : ℕ) : ℕ :=
  kids_cost + adults_cost

theorem mrs_hilt_total_spent : total_cost total_kids_ticket_cost total_adults_ticket_cost = 15 := by
  sorry

end mrs_hilt_total_spent_l7_7118


namespace solve_for_3x_plus_9_l7_7184

theorem solve_for_3x_plus_9 :
  ∀ (x : ℝ), (5 * x - 8 = 15 * x + 18) → 3 * (x + 9) = 96 / 5 :=
by
  intros x h
  sorry

end solve_for_3x_plus_9_l7_7184


namespace p_interval_satisfies_inequality_l7_7285

theorem p_interval_satisfies_inequality :
  ∀ (p q : ℝ), 0 ≤ p ∧ p < 2.232 ∧ q > 0 ∧ p + q ≠ 0 →
    (4 * (p * q ^ 2 + p ^ 2 * q + 4 * q ^ 2 + 4 * p * q)) / (p + q) > 5 * p ^ 2 * q :=
by sorry

end p_interval_satisfies_inequality_l7_7285


namespace total_supervisors_correct_l7_7094

-- Define the number of supervisors on each bus
def bus_supervisors : List ℕ := [4, 5, 3, 6, 7]

-- Define the total number of supervisors
def total_supervisors := bus_supervisors.sum

-- State the theorem to prove that the total number of supervisors is 25
theorem total_supervisors_correct : total_supervisors = 25 :=
by
  sorry -- Proof is to be completed

end total_supervisors_correct_l7_7094


namespace jogged_time_l7_7377

theorem jogged_time (J : ℕ) (W : ℕ) (r : ℚ) (h1 : r = 5 / 3) (h2 : W = 9) (h3 : r = J / W) : J = 15 := 
by
  sorry

end jogged_time_l7_7377


namespace annual_interest_rate_is_12_percent_l7_7175

theorem annual_interest_rate_is_12_percent
  (P : ℕ := 750000)
  (I : ℕ := 37500)
  (t : ℕ := 5)
  (months_in_year : ℕ := 12)
  (annual_days : ℕ := 360)
  (days_per_month : ℕ := 30) :
  ∃ r : ℚ, (r * 100 * months_in_year = 12) ∧ I = P * r * t := 
sorry

end annual_interest_rate_is_12_percent_l7_7175


namespace mike_travel_miles_l7_7541

theorem mike_travel_miles
  (toll_fees_mike : ℝ) (toll_fees_annie : ℝ) (mike_start_fee : ℝ) 
  (annie_start_fee : ℝ) (mike_per_mile : ℝ) (annie_per_mile : ℝ) 
  (annie_travel_time : ℝ) (annie_speed : ℝ) (mike_cost : ℝ) 
  (annie_cost : ℝ) 
  (h_mike_cost_eq : mike_cost = mike_start_fee + toll_fees_mike + mike_per_mile * 36)
  (h_annie_cost_eq : annie_cost = annie_start_fee + toll_fees_annie + annie_per_mile * annie_speed * annie_travel_time)
  (h_equal_costs : mike_cost = annie_cost)
  : 36 = 36 :=
by 
  sorry

end mike_travel_miles_l7_7541


namespace tiger_catch_distance_correct_l7_7995

noncomputable def tiger_catch_distance (tiger_leaps_behind : ℕ) (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ) (tiger_m_per_leap : ℕ) (deer_m_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_m_per_leap
  let tiger_per_minute := tiger_leaps_per_minute * tiger_m_per_leap
  let deer_per_minute := deer_leaps_per_minute * deer_m_per_leap
  let gain_per_minute := tiger_per_minute - deer_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_per_minute

theorem tiger_catch_distance_correct :
  tiger_catch_distance 50 5 4 8 5 = 800 :=
by
  -- This is the placeholder for the proof.
  sorry

end tiger_catch_distance_correct_l7_7995


namespace original_number_is_perfect_square_l7_7839

variable (n : ℕ)

theorem original_number_is_perfect_square
  (h1 : n = 1296)
  (h2 : ∃ m : ℕ, (n + 148) = m^2) : ∃ k : ℕ, n = k^2 :=
by
  sorry

end original_number_is_perfect_square_l7_7839


namespace mukesh_total_debt_l7_7424

-- Define the initial principal, additional loan, interest rate, and time periods
def principal₁ : ℝ := 10000
def principal₂ : ℝ := 12000
def rate : ℝ := 0.06
def time₁ : ℝ := 2
def time₂ : ℝ := 3

-- Define the interest calculations
def interest₁ : ℝ := principal₁ * rate * time₁
def total_after_2_years : ℝ := principal₁ + interest₁ + principal₂
def interest₂ : ℝ := total_after_2_years * rate * time₂

-- Define the total amount owed after 5 years
def amount_owed : ℝ := total_after_2_years + interest₂

-- The goal is to prove that Mukesh owes 27376 Rs after 5 years
theorem mukesh_total_debt : amount_owed = 27376 := by sorry

end mukesh_total_debt_l7_7424


namespace complement_A_is_correct_l7_7209

-- Let A be the set representing the domain of the function y = log2(x - 1)
def A : Set ℝ := { x : ℝ | x > 1 }

-- The universal set is ℝ
def U : Set ℝ := Set.univ

-- Complement of A with respect to ℝ
def complement_A (U : Set ℝ) (A : Set ℝ) : Set ℝ := U \ A

-- Prove that the complement of A with respect to ℝ is (-∞, 1]
theorem complement_A_is_correct : complement_A U A = { x : ℝ | x ≤ 1 } :=
by {
 sorry
}

end complement_A_is_correct_l7_7209


namespace find_n_l7_7777

theorem find_n (n : ℕ) (h : 1 < n) :
  (∀ a b : ℕ, Nat.gcd a b = 1 → (a % n = b % n ↔ (a * b) % n = 1)) →
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  sorry

end find_n_l7_7777


namespace price_restoration_l7_7186

theorem price_restoration (P : Real) (hP : P > 0) :
  let new_price := 0.85 * P
  let required_increase := ((1 / 0.85) - 1) * 100
  required_increase = 17.65 :=
by 
  sorry

end price_restoration_l7_7186


namespace original_candle_length_l7_7057

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (h_current : current_length = 48) (h_factor : factor = 1.33) :
  (current_length * factor = 63.84) :=
by
  -- The proof goes here
  sorry

end original_candle_length_l7_7057


namespace smallest_positive_angle_equivalent_neg_1990_l7_7215

theorem smallest_positive_angle_equivalent_neg_1990:
  ∃ k : ℤ, 0 ≤ (θ : ℤ) ∧ θ < 360 ∧ -1990 + 360 * k = θ := by
  use 6
  sorry

end smallest_positive_angle_equivalent_neg_1990_l7_7215


namespace find_correct_day_l7_7271

def tomorrow_is_not_September (d : String) : Prop :=
  d ≠ "September"

def in_a_week_is_September (d : String) : Prop :=
  d = "September"

def day_after_tomorrow_is_not_Wednesday (d : String) : Prop :=
  d ≠ "Wednesday"

theorem find_correct_day :
    ((∀ d, tomorrow_is_not_September d) ∧ 
    (∀ d, in_a_week_is_September d) ∧ 
    (∀ d, day_after_tomorrow_is_not_Wednesday d)) → 
    "Wednesday, August 25" = "Wednesday, August 25" :=
by
sorry

end find_correct_day_l7_7271


namespace cost_per_bottle_l7_7173

theorem cost_per_bottle (cost_3_bottles cost_4_bottles : ℝ) (n_bottles : ℕ) 
  (h1 : cost_3_bottles = 1.50) (h2 : cost_4_bottles = 2) : 
  (cost_3_bottles / 3) = (cost_4_bottles / 4) ∧ (cost_3_bottles / 3) * n_bottles = 0.50 * n_bottles :=
by
  sorry

end cost_per_bottle_l7_7173


namespace minimal_distance_l7_7157

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end minimal_distance_l7_7157


namespace price_increase_and_decrease_l7_7580

theorem price_increase_and_decrease (P : ℝ) (x : ℝ) 
  (h1 : 0 < P) 
  (h2 : (P * (1 - (x / 100) ^ 2)) = 0.81 * P) : 
  abs (x - 44) < 1 :=
by
  sorry

end price_increase_and_decrease_l7_7580


namespace calculate_expression_l7_7065

variable (x : ℝ)

theorem calculate_expression : ((3 * x)^2) * (x^2) = 9 * (x^4) := 
sorry

end calculate_expression_l7_7065


namespace complete_the_square_l7_7423

theorem complete_the_square (d e f : ℤ) (h1 : d > 0)
  (h2 : 25 * d * d = 25)
  (h3 : 10 * d * e = 30)
  (h4 : 25 * d * d * (d * x + e) * (d * x + e) = 25 * x * x * 25 + 30 * x * 25 * d + 25 * e * e - 9)
  : d + e + f = 41 := 
  sorry

end complete_the_square_l7_7423


namespace cos_sin_exp_l7_7783

theorem cos_sin_exp (n : ℕ) (t : ℝ) (h : n ≤ 1000) :
  (Complex.exp (t * Complex.I)) ^ n = Complex.exp (n * t * Complex.I) :=
by
  sorry

end cos_sin_exp_l7_7783


namespace P_sufficient_but_not_necessary_for_Q_l7_7951

variable (x : ℝ)

def P := x ≥ 0
def Q := 2 * x + 1 / (2 * x + 1) ≥ 1

theorem P_sufficient_but_not_necessary_for_Q : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l7_7951


namespace vertical_asymptote_values_l7_7382

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

theorem vertical_asymptote_values (c : ℝ) :
  (∃ x : ℝ, (x^2 + x - 20 = 0 ∧ x^2 - x + c = 0) ↔
   (c = -12 ∨ c = -30)) := sorry

end vertical_asymptote_values_l7_7382


namespace value_of_nested_expression_l7_7572

def nested_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2

theorem value_of_nested_expression : nested_expression = 1457 := by
  sorry

end value_of_nested_expression_l7_7572


namespace equation_solution_l7_7681

theorem equation_solution (x : ℝ) (h₁ : 2 * x - 5 ≠ 0) (h₂ : 5 - 2 * x ≠ 0) :
  (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ↔ (x = 0) :=
by
  sorry

end equation_solution_l7_7681


namespace hyperbola_focus_and_distance_l7_7665

noncomputable def right_focus_of_hyperbola (a b : ℝ) : ℝ × ℝ := 
  (Real.sqrt (a^2 + b^2), 0)

noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  abs c / Real.sqrt (1 + (b/a)^2)

theorem hyperbola_focus_and_distance (a b : ℝ) (h₁ : a^2 = 6) (h₂ : b^2 = 3) :
  right_focus_of_hyperbola a b = (3, 0) ∧ distance_to_asymptote a b = Real.sqrt 3 :=
by
  sorry

end hyperbola_focus_and_distance_l7_7665


namespace downstream_speed_l7_7416

-- Define the given conditions
def V_m : ℝ := 40 -- speed of the man in still water in kmph
def V_up : ℝ := 32 -- speed of the man upstream in kmph

-- Question to be proved as a statement
theorem downstream_speed : 
  ∃ (V_c V_down : ℝ), V_c = V_m - V_up ∧ V_down = V_m + V_c ∧ V_down = 48 :=
by
  -- Provide statement without proof as specified
  sorry

end downstream_speed_l7_7416


namespace range_of_m_l7_7031

open Real

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m
def q (m : ℝ) : Prop := (2 - m) > 0

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l7_7031


namespace find_range_of_a_l7_7709

def setA (x : ℝ) : Prop := 1 < x ∧ x < 2
def setB (x : ℝ) : Prop := 3 / 2 < x ∧ x < 4
def setUnion (x : ℝ) : Prop := 1 < x ∧ x < 4
def setP (a x : ℝ) : Prop := a < x ∧ x < a + 2

theorem find_range_of_a (a : ℝ) :
  (∀ x, setP a x → setUnion x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end find_range_of_a_l7_7709


namespace not_prime_5n_plus_3_l7_7121

def isSquare (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

theorem not_prime_5n_plus_3 (n k m : ℕ) (h₁ : 2 * n + 1 = k * k) (h₂ : 3 * n + 1 = m * m) (n_pos : 0 < n) (k_pos : 0 < k) (m_pos : 0 < m) :
  ¬ Nat.Prime (5 * n + 3) :=
sorry -- Proof to be completed

end not_prime_5n_plus_3_l7_7121


namespace tiling_not_possible_l7_7189

-- Definitions for the puzzle pieces
inductive Piece
| L | T | I | Z | O

-- Function to check if tiling a rectangle is possible
noncomputable def can_tile_rectangle (pieces : List Piece) : Prop :=
  ∀ (width height : ℕ), width * height % 4 = 0 → ∃ (tiling : List (Piece × ℕ × ℕ)), sorry

theorem tiling_not_possible : ¬ can_tile_rectangle [Piece.L, Piece.T, Piece.I, Piece.Z, Piece.O] :=
sorry

end tiling_not_possible_l7_7189


namespace total_money_found_l7_7626

-- Define the conditions
def donna_share := 0.40
def friendA_share := 0.35
def friendB_share := 0.25
def donna_amount := 39.0

-- Define the problem statement/proof
theorem total_money_found (donna_share friendA_share friendB_share donna_amount : ℝ) 
  (h1 : donna_share = 0.40) 
  (h2 : friendA_share = 0.35) 
  (h3 : friendB_share = 0.25) 
  (h4 : donna_amount = 39.0) :
  ∃ total_money : ℝ, total_money = 97.50 := 
by
  -- The calculations and actual proof will go here
  sorry

end total_money_found_l7_7626


namespace parity_of_E2021_E2022_E2023_l7_7955

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 0
  else seq (n - 2) + seq (n - 3)

theorem parity_of_E2021_E2022_E2023 :
  is_odd (seq 2021) ∧ is_even (seq 2022) ∧ is_odd (seq 2023) :=
by
  sorry

end parity_of_E2021_E2022_E2023_l7_7955


namespace tanya_time_proof_l7_7488

noncomputable def time_sakshi : ℝ := 10
noncomputable def efficiency_increase : ℝ := 1.25
noncomputable def time_tanya (time_sakshi : ℝ) (efficiency_increase : ℝ) : ℝ := time_sakshi / efficiency_increase

theorem tanya_time_proof : time_tanya time_sakshi efficiency_increase = 8 := 
by 
  sorry

end tanya_time_proof_l7_7488


namespace value_of_a_minus_n_plus_k_l7_7975

theorem value_of_a_minus_n_plus_k :
  ∃ (a k n : ℤ), 
    (∀ x : ℤ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) ∧ 
    (a - n + k = 3) :=
sorry

end value_of_a_minus_n_plus_k_l7_7975


namespace min_x_given_conditions_l7_7030

theorem min_x_given_conditions :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (100 : ℚ) / 151 < y / x ∧ y / x < (200 : ℚ) / 251 ∧ x = 3 :=
by
  sorry

end min_x_given_conditions_l7_7030


namespace necessary_but_not_sufficient_l7_7912

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 0) : a > 0 ↔ ((a > 0) ∧ (a < 2) → (a^2 - 2 * a < 0)) :=
by
    sorry

end necessary_but_not_sufficient_l7_7912


namespace rachel_found_boxes_l7_7619

theorem rachel_found_boxes (pieces_per_box total_pieces B : ℕ) 
  (h1 : pieces_per_box = 7) 
  (h2 : total_pieces = 49) 
  (h3 : B = total_pieces / pieces_per_box) : B = 7 := 
by 
  sorry

end rachel_found_boxes_l7_7619


namespace bob_coloring_l7_7261

/-
  Problem:
  Find the number of ways to color five points in {(x, y) | 1 ≤ x, y ≤ 5} blue 
  such that the distance between any two blue points is not an integer.
-/

def is_integer_distance (p1 p2 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let d := Int.gcd ((x2 - x1)^2 + (y2 - y1)^2)
  d ≠ 1

def valid_coloring (points : List (ℤ × ℤ)) : Prop :=
  points.length = 5 ∧ 
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬ is_integer_distance p1 p2)

theorem bob_coloring : ∃ (points : List (ℤ × ℤ)), valid_coloring points ∧ points.length = 80 :=
sorry

end bob_coloring_l7_7261


namespace number_of_mandatory_questions_correct_l7_7068

-- Definitions and conditions
def num_mandatory_questions (x : ℕ) (k : ℕ) (y : ℕ) (m : ℕ) : Prop :=
  (3 * k - 2 * (x - k) + 5 * m = 49) ∧
  (k + m = 15) ∧
  (y = 25 - x)

-- Proof statement
theorem number_of_mandatory_questions_correct :
  ∃ x k y m : ℕ, num_mandatory_questions x k y m ∧ x = 13 :=
by
  sorry

end number_of_mandatory_questions_correct_l7_7068


namespace solve_quadratic_eq1_solve_quadratic_eq2_l7_7228

theorem solve_quadratic_eq1 (x : ℝ) :
  x^2 - 4 * x + 3 = 0 ↔ (x = 3 ∨ x = 1) :=
sorry

theorem solve_quadratic_eq2 (x : ℝ) :
  x^2 - x - 3 = 0 ↔ (x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) :=
sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l7_7228


namespace proof_problem_l7_7506

variable (a b c d : ℝ)
variable (ω : ℂ)

-- Conditions
def conditions : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧
  ω^4 = 1 ∧ ω ≠ 1 ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2)

theorem proof_problem (h : conditions a b c d ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 2 := 
sorry

end proof_problem_l7_7506


namespace no_solution_exists_l7_7769

theorem no_solution_exists : 
  ¬ ∃ (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    45 * x = (35 / 100) * 900 ∧
    y^2 + x = 100 ∧
    z = x^3 * y - (2 * x + 1) / (y + 4) :=
by
  sorry

end no_solution_exists_l7_7769


namespace find_x_value_l7_7412

theorem find_x_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 :=
sorry

end find_x_value_l7_7412


namespace infinite_rational_points_in_region_l7_7493

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
  (∀ p ∈ S, (p.1 ^ 2 + p.2 ^ 2 ≤ 16) ∧ (p.1 ≤ 3) ∧ (p.2 ≤ 3) ∧ (p.1 > 0) ∧ (p.2 > 0)) ∧
  Set.Infinite S :=
sorry

end infinite_rational_points_in_region_l7_7493


namespace arithmetic_sequence_a2_a8_l7_7373

variable {a : ℕ → ℝ}

-- given condition
axiom h1 : a 4 + a 5 + a 6 = 450

-- problem statement
theorem arithmetic_sequence_a2_a8 : a 2 + a 8 = 300 :=
by
  sorry

end arithmetic_sequence_a2_a8_l7_7373


namespace distinct_square_sum_100_l7_7690

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l7_7690


namespace center_of_circle_l7_7072

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 10) (h4 : y2 = 7) :
  (x1 + x2) / 2 = 6 ∧ (y1 + y2) / 2 = 2 :=
by
  rw [h1, h2, h3, h4]
  constructor
  · norm_num
  · norm_num

end center_of_circle_l7_7072


namespace number_exceeds_percent_l7_7718

theorem number_exceeds_percent (x : ℝ) (h : x = 0.12 * x + 52.8) : x = 60 :=
by {
  sorry
}

end number_exceeds_percent_l7_7718


namespace mean_of_combined_sets_l7_7459

theorem mean_of_combined_sets (mean_set1 mean_set2 : ℝ) (n1 n2 : ℕ) 
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 20) (h3 : n1 = 5) (h4 : n2 = 8) :
  (n1 * mean_set1 + n2 * mean_set2) / (n1 + n2) = 235 / 13 :=
by
  sorry

end mean_of_combined_sets_l7_7459


namespace tim_cantaloupes_l7_7368

theorem tim_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) : total_cantaloupes - fred_cantaloupes = 44 :=
by {
  -- proof steps go here
  sorry
}

end tim_cantaloupes_l7_7368


namespace conversion_rate_false_l7_7691

-- Definition of conversion rates between units
def conversion_rate_hour_minute : ℕ := 60
def conversion_rate_minute_second : ℕ := 60

-- Theorem stating that the rate being 100 is false under the given conditions
theorem conversion_rate_false (h1 : conversion_rate_hour_minute = 60) 
  (h2 : conversion_rate_minute_second = 60) : 
  ¬ (conversion_rate_hour_minute = 100 ∧ conversion_rate_minute_second = 100) :=
by {
  sorry
}

end conversion_rate_false_l7_7691


namespace curve_in_second_quadrant_range_l7_7754

theorem curve_in_second_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0)) → a > 2 :=
by
  sorry

end curve_in_second_quadrant_range_l7_7754


namespace cube_volume_split_l7_7725

theorem cube_volume_split (x y z : ℝ) (h : x > 0) :
  ∃ y z : ℝ, y > 0 ∧ z > 0 ∧ y^3 + z^3 = x^3 :=
sorry

end cube_volume_split_l7_7725


namespace sum_of_three_consecutive_odd_integers_l7_7477

theorem sum_of_three_consecutive_odd_integers (n : ℤ) 
  (h1 : n + (n + 4) = 130) 
  (h2 : n % 2 = 1) : 
  n + (n + 2) + (n + 4) = 195 := 
by
  sorry

end sum_of_three_consecutive_odd_integers_l7_7477


namespace trapezium_area_l7_7243

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l7_7243


namespace evaluate_expr_l7_7380

theorem evaluate_expr : Int.ceil (5 / 4 : ℚ) + Int.floor (-5 / 4 : ℚ) = 0 := by
  sorry

end evaluate_expr_l7_7380


namespace jump_rope_difference_l7_7258

noncomputable def cindy_jump_time : ℕ := 12
noncomputable def betsy_jump_time : ℕ := cindy_jump_time / 2
noncomputable def tina_jump_time : ℕ := 3 * betsy_jump_time

theorem jump_rope_difference : tina_jump_time - cindy_jump_time = 6 :=
by
  -- proof steps would go here
  sorry

end jump_rope_difference_l7_7258


namespace sum_of_possible_radii_l7_7015

-- Define the geometric and algebraic conditions of the problem
noncomputable def circleTangentSum (r : ℝ) : Prop :=
  let center_C := (r, r)
  let center_other := (3, 3)
  let radius_other := 2
  (∃ r : ℝ, (r > 0) ∧ ((center_C.1 - center_other.1)^2 + (center_C.2 - center_other.2)^2 = (r + radius_other)^2))

-- Define the theorem statement
theorem sum_of_possible_radii : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ circleTangentSum r1 ∧ circleTangentSum r2 ∧ r1 + r2 = 16 :=
sorry

end sum_of_possible_radii_l7_7015


namespace anusha_solution_l7_7455

variable (A B E : ℝ) -- Defining the variables for amounts received by Anusha, Babu, and Esha
variable (total_amount : ℝ) (h_division : 12 * A = 8 * B) (h_division2 : 8 * B = 6 * E) (h_total : A + B + E = 378)

theorem anusha_solution : A = 84 :=
by
  -- Using the given conditions and deriving the amount Anusha receives
  sorry

end anusha_solution_l7_7455


namespace percent_defective_units_l7_7466

theorem percent_defective_units (D : ℝ) (h1 : 0.05 * D = 0.5) : D = 10 := by
  sorry

end percent_defective_units_l7_7466


namespace median_of_trapezoid_l7_7356

theorem median_of_trapezoid (h : ℝ) (x : ℝ) 
  (triangle_area_eq_trapezoid_area : (1 / 2) * 24 * h = ((x + (2 * x)) / 2) * h) : 
  ((x + (2 * x)) / 2) = 12 := by
  sorry

end median_of_trapezoid_l7_7356


namespace train_length_l7_7546

theorem train_length (L : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = L / 15) 
  (h2 : V2 = (L + 800) / 45) 
  (h3 : V1 = V2) : 
  L = 400 := 
sorry

end train_length_l7_7546


namespace parabola_opens_downward_iff_l7_7188

theorem parabola_opens_downward_iff (m : ℝ) : (m - 1 < 0) ↔ (m < 1) :=
by
  sorry

end parabola_opens_downward_iff_l7_7188


namespace range_of_k_l7_7639

def f (x : ℝ) : ℝ := x^3 - 12*x

def not_monotonic_on_I (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), k - 1 < x₁ ∧ x₁ < k + 1 ∧ k - 1 < x₂ ∧ x₂ < k + 1 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) * (x₁ - x₂) < 0

theorem range_of_k (k : ℝ) : not_monotonic_on_I k ↔ (k > -3 ∧ k < -1) ∨ (k > 1 ∧ k < 3) :=
sorry

end range_of_k_l7_7639


namespace cyclist_speed_ratio_l7_7464

theorem cyclist_speed_ratio
  (d : ℝ) (t₁ t₂ : ℝ) 
  (v₁ v₂ : ℝ)
  (h1 : d = 8)
  (h2 : t₁ = 4)
  (h3 : t₂ = 1)
  (h4 : d = (v₁ - v₂) * t₁)
  (h5 : d = (v₁ + v₂) * t₂) :
  v₁ / v₂ = 5 / 3 :=
sorry

end cyclist_speed_ratio_l7_7464


namespace jellybean_total_count_l7_7244

theorem jellybean_total_count :
  let black := 8
  let green := 2 * black
  let orange := (2 * green) - 5
  let red := orange + 3
  let yellow := black / 2
  let purple := red + 4
  let brown := (green + purple) - 3
  black + green + orange + red + yellow + purple + brown = 166 := by
  -- skipping proof for brevity
  sorry

end jellybean_total_count_l7_7244


namespace number_of_integer_values_of_x_l7_7301

theorem number_of_integer_values_of_x (x : ℕ) (hx1 : x + 15 > 40) (hx2 : x + 40 > 15) (hx3 : 15 + 40 > x) :
  ∃ n : ℕ, n = 29 ∧ ∀ y : ℕ, (26 ≤ y ∧ y ≤ 54) ↔ true :=
by
  sorry

end number_of_integer_values_of_x_l7_7301


namespace gambler_difference_eq_two_l7_7125

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end gambler_difference_eq_two_l7_7125


namespace sequence_term_37_l7_7671

theorem sequence_term_37 (n : ℕ) (h_pos : 0 < n) (h_eq : 3 * n + 1 = 37) : n = 12 :=
by
  sorry

end sequence_term_37_l7_7671


namespace age_of_youngest_child_l7_7438

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65) : x = 7 :=
sorry

end age_of_youngest_child_l7_7438


namespace identify_false_condition_l7_7222

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
def condition_A (a b c : ℝ) : Prop := quadratic_function a b c (-1) = 0
def condition_B (a b c : ℝ) : Prop := 2 * a + b = 0
def condition_C (a b c : ℝ) : Prop := quadratic_function a b c 1 = 3
def condition_D (a b c : ℝ) : Prop := quadratic_function a b c 2 = 8

-- Main theorem stating which condition is false
theorem identify_false_condition (a b c : ℝ) (ha : a ≠ 0) : ¬ condition_A a b c ∨ ¬ condition_B a b c ∨ ¬ condition_C a b c ∨  ¬ condition_D a b c :=
by
sorry

end identify_false_condition_l7_7222


namespace find_d_l7_7422

theorem find_d (
  x : ℝ
) (
  h1 : 3 * x + 8 = 5
) (
  d : ℝ
) (
  h2 : d * x - 15 = -7
) : d = -8 :=
by
  sorry

end find_d_l7_7422


namespace angle_sum_solution_l7_7560

theorem angle_sum_solution
  (x : ℝ)
  (h : 3 * x + 140 = 360) :
  x = 220 / 3 :=
by
  sorry

end angle_sum_solution_l7_7560


namespace zero_in_interval_l7_7297

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_in_interval :
  (Real.log (3 / 2) - 2 < 0) ∧ (Real.log 3 - 2 / 3 > 0) →
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- conditions from the problem statement
  intros h
  -- proving the result
  sorry

end zero_in_interval_l7_7297


namespace ratio_out_of_school_friends_to_classmates_l7_7701

variable (F : ℕ) (classmates : ℕ := 20) (parents : ℕ := 2) (sister : ℕ := 1) (total : ℕ := 33)

theorem ratio_out_of_school_friends_to_classmates (h : classmates + F + parents + sister = total) :
  (F : ℚ) / classmates = 1 / 2 := by
    -- sorry allows this to build even if proof is not provided
    sorry

end ratio_out_of_school_friends_to_classmates_l7_7701


namespace cylinder_lateral_surface_area_l7_7537

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (radius_eq : r = 2) 
  (height_eq : h = 5) : 
  2 * Real.pi * r * h = 62.8 :=
by
  -- Proof steps go here
  sorry

end cylinder_lateral_surface_area_l7_7537


namespace find_angle_A_l7_7972

theorem find_angle_A (BC AC : ℝ) (B : ℝ) (A : ℝ) (h_cond : BC = Real.sqrt 3 ∧ AC = 1 ∧ B = Real.pi / 6) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l7_7972


namespace gcd_lcm_product_24_60_l7_7478

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l7_7478


namespace rank_classmates_l7_7603

-- Definitions of the conditions
def emma_tallest (emma david fiona : ℕ) : Prop := emma > david ∧ emma > fiona
def fiona_not_shortest (david emma fiona : ℕ) : Prop := david > fiona ∧ emma > fiona
def david_not_tallest (david emma fiona : ℕ) : Prop := emma > david ∧ fiona > david

def exactly_one_true (david emma fiona : ℕ) : Prop :=
  (emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ david_not_tallest david emma fiona)

-- The final proof statement
theorem rank_classmates (david emma fiona : ℕ) (h : exactly_one_true david emma fiona) : david > fiona ∧ fiona > emma :=
  sorry

end rank_classmates_l7_7603


namespace sector_to_cone_base_area_l7_7211

theorem sector_to_cone_base_area
  (r_sector : ℝ) (theta : ℝ) (h1 : r_sector = 2) (h2 : theta = 120) :
  ∃ (A : ℝ), A = (4 / 9) * Real.pi :=
by
  sorry

end sector_to_cone_base_area_l7_7211


namespace Piglet_ate_one_l7_7793

theorem Piglet_ate_one (V S K P : ℕ) (h1 : V + S + K + P = 70)
  (h2 : S + K = 45) (h3 : V > S) (h4 : V > K) (h5 : V > P) 
  (h6 : V ≥ 1) (h7 : S ≥ 1) (h8 : K ≥ 1) (h9 : P ≥ 1) : P = 1 :=
sorry

end Piglet_ate_one_l7_7793


namespace solve_equation_l7_7795

theorem solve_equation :
  ∃ a b x : ℤ, 
  ((a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0) 
  ↔ (a = -6 ∧ b = -5 ∧ x = -2) :=
by {
  sorry
}

end solve_equation_l7_7795


namespace average_non_prime_squares_approx_l7_7220

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of non-prime numbers between 50 and 100
def non_prime_numbers : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70,
   72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,
   92, 93, 94, 95, 96, 98, 99]

-- Define the sum of squares of the elements in a list
def sum_of_squares (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * x + acc) 0

-- Define the count of non-prime numbers
def count_non_prime : ℕ :=
  non_prime_numbers.length

-- Calculate the average
def average_non_prime_squares : ℚ :=
  sum_of_squares non_prime_numbers / count_non_prime

-- Theorem to state that the average of the sum of squares of non-prime numbers
-- between 50 and 100 is approximately 6417.67
theorem average_non_prime_squares_approx :
  abs ((average_non_prime_squares : ℝ) - 6417.67) < 0.01 := 
  sorry

end average_non_prime_squares_approx_l7_7220


namespace sqrt_200_eq_10_sqrt_2_l7_7745

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l7_7745


namespace weighted_arithmetic_geometric_mean_l7_7692
-- Importing required library

-- Definitions of the problem variables and conditions
variables (a b c : ℝ)

-- Non-negative constraints on the lengths of the line segments
variables (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

-- Problem statement, we need to prove
theorem weighted_arithmetic_geometric_mean :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c)^(1/3) :=
sorry

end weighted_arithmetic_geometric_mean_l7_7692


namespace dan_job_time_l7_7761

theorem dan_job_time
  (Annie_time : ℝ) (Dan_work_time : ℝ) (Annie_work_remain : ℝ) (total_work : ℝ)
  (Annie_time_cond : Annie_time = 9)
  (Dan_work_time_cond : Dan_work_time = 8)
  (Annie_work_remain_cond : Annie_work_remain = 3.0000000000000004)
  (total_work_cond : total_work = 1) :
  ∃ (Dan_time : ℝ), Dan_time = 12 := by
  sorry

end dan_job_time_l7_7761


namespace max_a_squared_b_l7_7901

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) : a^2 * b ≤ 54 :=
sorry

end max_a_squared_b_l7_7901


namespace ratio_female_to_male_l7_7349

theorem ratio_female_to_male (total_members : ℕ) (female_members : ℕ) (male_members : ℕ) 
  (h1 : total_members = 18) (h2 : female_members = 12) (h3 : male_members = total_members - female_members) : 
  (female_members : ℚ) / (male_members : ℚ) = 2 := 
by 
  sorry

end ratio_female_to_male_l7_7349


namespace al_told_the_truth_l7_7163

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end al_told_the_truth_l7_7163


namespace travel_time_l7_7925

theorem travel_time (v : ℝ) (d : ℝ) (t : ℝ) (hv : v = 65) (hd : d = 195) : t = 3 :=
by
  sorry

end travel_time_l7_7925


namespace correct_statements_l7_7136

-- Define the regression condition
def regression_condition (b : ℝ) : Prop := b < 0

-- Conditon ③: Event A is the complement of event B implies mutual exclusivity
def mutually_exclusive_and_complementary (A B : Prop) : Prop := 
  (A → ¬B) → (¬A ↔ B)

-- Main theorem combining the conditions and questions
theorem correct_statements: 
  (∀ b, regression_condition b ↔ (b < 0)) ∧
  (∀ A B, mutually_exclusive_and_complementary A B → (¬A ≠ B)) :=
by
  sorry

end correct_statements_l7_7136


namespace sum_first_7_terms_eq_105_l7_7153

variable {a : ℕ → ℤ}

-- Definitions from conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a)

def a_4_eq_15 : a 4 = 15 := sorry

-- Sum definition specific for 7 terms of an arithmetic sequence.
def sum_first_7_terms (a : ℕ → ℤ) : ℤ := (7 / 2 : ℤ) * (a 1 + a 7)

-- The theorem to prove.
theorem sum_first_7_terms_eq_105 
    (arith_seq : is_arithmetic_sequence a) 
    (a4 : a 4 = 15) : 
  sum_first_7_terms a = 105 := 
sorry

end sum_first_7_terms_eq_105_l7_7153


namespace change_given_l7_7003

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end change_given_l7_7003


namespace simplification_of_expression_l7_7411

theorem simplification_of_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  ( (x - 2) / (x^2 - 2 * x + 1) / (x / (x - 1)) + 1 / (x^2 - x) ) = 1 / x := 
by 
  sorry

end simplification_of_expression_l7_7411


namespace total_selling_price_l7_7084

theorem total_selling_price
  (CP : ℕ) (Gain : ℕ) (TCP : ℕ)
  (h1 : CP = 1200)
  (h2 : Gain = 3 * CP)
  (h3 : TCP = 18 * CP) :
  ∃ TSP : ℕ, TSP = 25200 := 
by
  sorry

end total_selling_price_l7_7084


namespace painted_pictures_in_june_l7_7907

theorem painted_pictures_in_june (J : ℕ) (h1 : J + (J + 2) + 9 = 13) : J = 1 :=
by
  -- Given condition translates to J + J + 2 + 9 = 13
  -- Simplification yields 2J + 11 = 13
  -- Solving 2J + 11 = 13 gives J = 1
  sorry

end painted_pictures_in_june_l7_7907


namespace length_of_DE_l7_7064

-- Given conditions
variables (AB DE : ℝ) (area_projected area_ABC : ℝ)

-- Hypotheses
def base_length (AB : ℝ) : Prop := AB = 15
def projected_area_ratio (area_projected area_ABC : ℝ) : Prop := area_projected = 0.25 * area_ABC
def parallel_lines (DE AB : ℝ) : Prop := ∀ x : ℝ, DE = 0.5 * AB

-- The theorem to prove
theorem length_of_DE (h1 : base_length AB) (h2 : projected_area_ratio area_projected area_ABC) (h3 : parallel_lines DE AB) : DE = 7.5 :=
by
  sorry

end length_of_DE_l7_7064


namespace singh_gain_l7_7760

def initial_amounts (B A S : ℕ) : Prop :=
  B = 70 ∧ A = 70 ∧ S = 70

def ratio_Ashtikar_Singh (A S : ℕ) : Prop :=
  2 * A = S

def ratio_Singh_Bhatia (S B : ℕ) : Prop :=
  4 * B = S

def total_conservation (A S B : ℕ) : Prop :=
  A + S + B = 210

theorem singh_gain : ∀ B A S fA fB fS : ℕ,
  initial_amounts B A S →
  ratio_Ashtikar_Singh fA fS →
  ratio_Singh_Bhatia fS fB →
  total_conservation fA fS fB →
  fS - S = 50 :=
by
  intros B A S fA fB fS
  intros i rA rS tC
  sorry

end singh_gain_l7_7760


namespace ratio_mets_redsox_l7_7395

theorem ratio_mets_redsox 
    (Y M R : ℕ) 
    (h1 : Y = 3 * (M / 2))
    (h2 : M = 88)
    (h3 : Y + M + R = 330) : 
    M / R = 4 / 5 := 
by 
    sorry

end ratio_mets_redsox_l7_7395


namespace rectangle_area_l7_7828

theorem rectangle_area (l w : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 120) : l * w = 800 :=
by
  -- proof to be filled in
  sorry

end rectangle_area_l7_7828


namespace evaluate_using_horners_method_l7_7315

def f (x : ℝ) : ℝ := 3 * x^6 + 12 * x^5 + 8 * x^4 - 3.5 * x^3 + 7.2 * x^2 + 5 * x - 13

theorem evaluate_using_horners_method :
  f 6 = 243168.2 :=
by
  sorry

end evaluate_using_horners_method_l7_7315


namespace john_protest_days_l7_7984

theorem john_protest_days (days1: ℕ) (days2: ℕ) (days3: ℕ): 
  days1 = 4 → 
  days2 = (days1 + (days1 / 4)) → 
  days3 = (days2 + (days2 / 2)) → 
  (days1 + days2 + days3) = 17 :=
by
  intros h1 h2 h3
  sorry

end john_protest_days_l7_7984


namespace max_plus_shapes_l7_7993

def cover_square (x y : ℕ) : Prop :=
  3 * x + 5 * y = 49

theorem max_plus_shapes (x y : ℕ) (h1 : cover_square x y) (h2 : x ≥ 4) : y ≤ 5 :=
sorry

end max_plus_shapes_l7_7993


namespace simplify_fraction_l7_7878

variable (a b y : ℝ)
variable (h1 : y = (a + 2 * b) / a)
variable (h2 : a ≠ -2 * b)
variable (h3 : a ≠ 0)

theorem simplify_fraction : (2 * a + 2 * b) / (a - 2 * b) = (y + 1) / (3 - y) :=
by
  sorry

end simplify_fraction_l7_7878


namespace tom_finishes_in_6_years_l7_7044

/-- Combined program years for BS and Ph.D. -/
def BS_years : ℕ := 3
def PhD_years : ℕ := 5

/-- Total combined program time -/
def total_program_years : ℕ := BS_years + PhD_years

/-- Tom's time multiplier -/
def tom_time_multiplier : ℚ := 3 / 4

/-- Tom's total time to finish the program -/
def tom_total_time : ℚ := tom_time_multiplier * total_program_years

theorem tom_finishes_in_6_years : tom_total_time = 6 := 
by 
  -- implementation of the proof is to be filled in here
  sorry

end tom_finishes_in_6_years_l7_7044


namespace bottle_caps_per_child_l7_7922

-- Define the conditions
def num_children : ℕ := 9
def total_bottle_caps : ℕ := 45

-- State the theorem that needs to be proved: each child has 5 bottle caps
theorem bottle_caps_per_child : (total_bottle_caps / num_children) = 5 := by
  sorry

end bottle_caps_per_child_l7_7922


namespace min_value_expr_l7_7685

noncomputable def expr (x : ℝ) : ℝ := (Real.sin x)^8 + (Real.cos x)^8 + 3 / (Real.sin x)^6 + (Real.cos x)^6 + 3

theorem min_value_expr : ∃ x : ℝ, expr x = 14 / 31 := 
by
  sorry

end min_value_expr_l7_7685


namespace boat_shipments_divisor_l7_7787

/-- 
Given:
1. There exists an integer B representing the number of boxes that can be divided into S equal shipments by boat.
2. B can be divided into 24 equal shipments by truck.
3. The smallest number of boxes B is 120.
Prove that S, the number of equal shipments by boat, is 60.
--/
theorem boat_shipments_divisor (B S : ℕ) (h1 : B % S = 0) (h2 : B % 24 = 0) (h3 : B = 120) : S = 60 := 
sorry

end boat_shipments_divisor_l7_7787


namespace chords_in_circle_l7_7454

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l7_7454


namespace cylinder_volume_ratio_l7_7932

variable (h r : ℝ)

theorem cylinder_volume_ratio (h r : ℝ) :
  let V_original := π * r^2 * h
  let h_new := 2 * h
  let r_new := 4 * r
  let V_new := π * (r_new)^2 * h_new
  V_new = 32 * V_original :=
by
  sorry

end cylinder_volume_ratio_l7_7932


namespace volume_second_cube_l7_7441

open Real

-- Define the ratio of the edges of the cubes
def edge_ratio (a b : ℝ) := a / b = 3 / 1

-- Define the volume of the first cube
def volume_first_cube (a : ℝ) := a^3 = 27

-- Define the edge of the second cube based on the edge of the first cube
def edge_second_cube (a b : ℝ) := a / 3 = b

-- Statement of the problem in Lean 4
theorem volume_second_cube 
  (a b : ℝ) 
  (h_edge_ratio : edge_ratio a b) 
  (h_volume_first : volume_first_cube a) 
  (h_edge_second : edge_second_cube a b) : 
  b^3 = 1 := 
sorry

end volume_second_cube_l7_7441


namespace correct_statement_C_l7_7526

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l7_7526


namespace smallest_c_geometric_arithmetic_progression_l7_7714

theorem smallest_c_geometric_arithmetic_progression (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 0 < c) 
(h4 : b ^ 2 = a * c) (h5 : a + b = 2 * c) : c = 1 :=
sorry

end smallest_c_geometric_arithmetic_progression_l7_7714


namespace fourth_grade_students_l7_7058

theorem fourth_grade_students (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ) 
  (h_initial : initial_students = 35) (h_left : students_left = 10) (h_new : new_students = 10) :
  initial_students - students_left + new_students = 35 :=
by
  -- The proof goes here
  sorry

end fourth_grade_students_l7_7058


namespace largest_integer_is_59_l7_7802

theorem largest_integer_is_59 
  {w x y z : ℤ} 
  (h₁ : (w + x + y) / 3 = 32)
  (h₂ : (w + x + z) / 3 = 39)
  (h₃ : (w + y + z) / 3 = 40)
  (h₄ : (x + y + z) / 3 = 44) :
  max (max w x) (max y z) = 59 :=
by {
  sorry
}

end largest_integer_is_59_l7_7802


namespace usual_time_eight_l7_7292

/-- Define the parameters used in the problem -/
def usual_speed (S : ℝ) : ℝ := S
def usual_time (T : ℝ) : ℝ := T
def reduced_speed (S : ℝ) := 0.25 * S
def reduced_time (T : ℝ) := T + 24

/-- The main theorem that we need to prove -/
theorem usual_time_eight (S T : ℝ) 
  (h1 : usual_speed S = S)
  (h2 : usual_time T = T)
  (h3 : reduced_speed S = 0.25 * S)
  (h4 : reduced_time T = T + 24)
  (h5 : S / (0.25 * S) = (T + 24) / T) : T = 8 :=
by 
  sorry -- Proof omitted for brevity. Refers to the solution steps.


end usual_time_eight_l7_7292


namespace b3_b7_equals_16_l7_7360

variable {a b : ℕ → ℝ}
variable {d : ℝ}

-- Conditions: a is an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: b is a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition on the arithmetic sequence a
def condition_on_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * a 2 - (a 5) ^ 2 + 2 * a 8 = 0

-- Define the specific arithmetic sequence in terms of d and a5
noncomputable def a_seq (a5 d : ℝ) : ℕ → ℝ
| 0 => a5 - 5 * d
| 1 => a5 - 4 * d
| 2 => a5 - 3 * d
| 3 => a5 - 2 * d
| 4 => a5 - d
| 5 => a5
| 6 => a5 + d
| 7 => a5 + 2 * d
| 8 => a5 + 3 * d
| 9 => a5 + 4 * d
| n => 0 -- extending for unspecified

-- Condition: b_5 = a_5
def b_equals_a (a b : ℕ → ℝ) : Prop :=
  b 5 = a 5

-- Theorem: Given the conditions, prove b_3 * b_7 = 16
theorem b3_b7_equals_16 (a b : ℕ → ℝ) (d : ℝ)
  (ha_seq : is_arithmetic_sequence a d)
  (hb_seq : is_geometric_sequence b)
  (h_cond_a : condition_on_a a d)
  (h_b_equals_a : b_equals_a a b) : b 3 * b 7 = 16 :=
by
  sorry

end b3_b7_equals_16_l7_7360


namespace typing_speed_ratio_l7_7384

theorem typing_speed_ratio (T t : ℝ) (h1 : T + t = 12) (h2 : T + 1.25 * t = 14) : t / T = 2 :=
by
  sorry

end typing_speed_ratio_l7_7384


namespace sequence_increasing_l7_7594

theorem sequence_increasing (a : ℕ → ℝ) (a0 : ℝ) (h0 : a 0 = 1 / 5)
  (H : ∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) :
  ∀ n : ℕ, a (n + 1) > a n :=
sorry

end sequence_increasing_l7_7594


namespace quadratic_inequality_solution_l7_7705

theorem quadratic_inequality_solution : 
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3 / 2} :=
by
  sorry

end quadratic_inequality_solution_l7_7705


namespace symmetrical_line_equation_l7_7130

-- Definitions for the conditions
def line_symmetrical (eq1 eq2 : String) : Prop :=
  eq1 = "x - 2y + 3 = 0" ∧ eq2 = "x + 2y + 3 = 0"

-- Prove the statement
theorem symmetrical_line_equation : line_symmetrical "x - 2y + 3 = 0" "x + 2y + 3 = 0" :=
  by
  -- This is just the proof skeleton; the actual proof is not required
  sorry

end symmetrical_line_equation_l7_7130


namespace number_of_dozen_eggs_to_mall_l7_7788

-- Define the conditions as assumptions
def number_of_dozen_eggs_collected (x : Nat) : Prop :=
  x = 2 * 8

def number_of_dozen_eggs_to_market (x : Nat) : Prop :=
  x = 3

def number_of_dozen_eggs_for_pie (x : Nat) : Prop :=
  x = 4

def number_of_dozen_eggs_to_charity (x : Nat) : Prop :=
  x = 4

-- The theorem stating the answer to the problem
theorem number_of_dozen_eggs_to_mall 
  (h1 : ∃ x, number_of_dozen_eggs_collected x)
  (h2 : ∃ x, number_of_dozen_eggs_to_market x)
  (h3 : ∃ x, number_of_dozen_eggs_for_pie x)
  (h4 : ∃ x, number_of_dozen_eggs_to_charity x)
  : ∃ z, z = 5 := 
sorry

end number_of_dozen_eggs_to_mall_l7_7788


namespace sum_of_remainders_mod_53_l7_7627

theorem sum_of_remainders_mod_53 (d e f : ℕ) (hd : d % 53 = 19) (he : e % 53 = 33) (hf : f % 53 = 14) : 
  (d + e + f) % 53 = 13 :=
by
  sorry

end sum_of_remainders_mod_53_l7_7627


namespace xiaoli_estimate_smaller_l7_7240

variable (x y z : ℝ)
variable (hx : x > y) (hz : z > 0)

theorem xiaoli_estimate_smaller :
  (x - z) - (y + z) < x - y := 
by
  sorry

end xiaoli_estimate_smaller_l7_7240


namespace smallest_and_largest_group_sizes_l7_7018

theorem smallest_and_largest_group_sizes (S T : Finset ℕ) (hS : S.card + T.card = 20)
  (h_union: (S ∪ T) = (Finset.range 21) \ {0}) (h_inter: S ∩ T = ∅)
  (sum_S : S.sum id = 210 - T.sum id) (prod_T : T.prod id = 210 - S.sum id) :
  T.card = 3 ∨ T.card = 5 := 
sorry

end smallest_and_largest_group_sizes_l7_7018


namespace sum_of_areas_of_tangent_circles_l7_7305

theorem sum_of_areas_of_tangent_circles
  (r s t : ℝ)
  (h1 : r + s = 6)
  (h2 : s + t = 8)
  (h3 : r + t = 10) :
  π * (r^2 + s^2 + t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l7_7305


namespace parabola_properties_l7_7202

noncomputable def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties
  (a b c t m n x₀ : ℝ)
  (ha : a > 0)
  (h1 : parabola a b c 1 = m)
  (h4 : parabola a b c 4 = n)
  (ht : t = -b / (2 * a))
  (h3ab : 3 * a + b = 0) 
  (hmnc : m < c ∧ c < n)
  (hx₀ym : parabola a b c x₀ = m) :
  m < n ∧ (1 / 2) < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 :=
  sorry

end parabola_properties_l7_7202


namespace evaluate_nested_square_root_l7_7345

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end evaluate_nested_square_root_l7_7345


namespace range_of_m_l7_7584

theorem range_of_m (m : ℝ) (P : Prop) (Q : Prop) : 
  (P ∨ Q) ∧ ¬(P ∧ Q) →
  (P ↔ (m^2 - 4 > 0)) →
  (Q ↔ (16 * (m - 2)^2 - 16 < 0)) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l7_7584


namespace Jill_ball_difference_l7_7155

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l7_7155


namespace equivalent_prop_l7_7103

theorem equivalent_prop (x : ℝ) : (x > 1 → (x - 1) * (x + 3) > 0) ↔ ((x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
sorry

end equivalent_prop_l7_7103


namespace parabola_equation_line_equation_chord_l7_7612

section
variables (p : ℝ) (x_A y_A : ℝ) (M_x M_y : ℝ)
variable (h_p_pos : p > 0)
variable (h_A : y_A^2 = 8 * x_A)
variable (h_directrix_A : x_A + p / 2 = 5)
variable (h_M : (M_x, M_y) = (3, 2))

theorem parabola_equation (h_x_A : x_A = 3) : y_A^2 = 8 * x_A :=
sorry

theorem line_equation_chord
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : y1^2 = 8 * x1 ∧ y2^2 = 8 * x2)
  (h_chord_M : (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = 2) :
  y_M - 2 * x_M + 4 = 0 :=
sorry
end

end parabola_equation_line_equation_chord_l7_7612


namespace max_product_l7_7753

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l7_7753


namespace boxes_of_chocolates_l7_7427

theorem boxes_of_chocolates (total_pieces : ℕ) (pieces_per_box : ℕ) (h_total : total_pieces = 3000) (h_each : pieces_per_box = 500) : total_pieces / pieces_per_box = 6 :=
by
  sorry

end boxes_of_chocolates_l7_7427


namespace measure_exactly_10_liters_l7_7272

theorem measure_exactly_10_liters (A B : ℕ) (A_cap B_cap : ℕ) (hA : A_cap = 11) (hB : B_cap = 9) :
  ∃ (A B : ℕ), A + B = 10 ∧ A ≤ A_cap ∧ B ≤ B_cap := 
sorry

end measure_exactly_10_liters_l7_7272


namespace functions_same_function_C_functions_same_function_D_l7_7287

theorem functions_same_function_C (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by sorry

theorem functions_same_function_D (x : ℝ) : x = (x^3)^(1/3) :=
by sorry

end functions_same_function_C_functions_same_function_D_l7_7287


namespace highest_number_paper_l7_7391

theorem highest_number_paper
  (n : ℕ)
  (P : ℝ)
  (hP : P = 0.010309278350515464)
  (hP_formula : 1 / n = P) :
  n = 97 :=
by
  -- Placeholder for proof
  sorry

end highest_number_paper_l7_7391


namespace min_value_of_squared_sums_l7_7496

theorem min_value_of_squared_sums (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ B, (B = x^2 + y^2 + z^2) ∧ (B ≥ 4) := 
by {
  sorry -- Proof will be provided here.
}

end min_value_of_squared_sums_l7_7496


namespace area_difference_of_square_screens_l7_7214

theorem area_difference_of_square_screens (d1 d2 : ℝ) (A1 A2 : ℝ) 
  (h1 : d1 = 18) (h2 : d2 = 16) 
  (hA1 : A1 = d1^2 / 2) (hA2 : A2 = d2^2 / 2) : 
  A1 - A2 = 34 := by
  sorry

end area_difference_of_square_screens_l7_7214


namespace marble_problem_l7_7052

variable (A V M : ℕ)

theorem marble_problem
  (h1 : A + 5 = V - 5)
  (h2 : V + 2 * (A + 5) = A - 2 * (A + 5) + M) :
  M = 10 :=
sorry

end marble_problem_l7_7052


namespace man_l7_7038

theorem man's_speed_against_current :
  ∀ (V_down V_c V_m V_up : ℝ),
    (V_down = 15) →
    (V_c = 2.8) →
    (V_m = V_down - V_c) →
    (V_up = V_m - V_c) →
    V_up = 9.4 :=
by
  intros V_down V_c V_m V_up
  intros hV_down hV_c hV_m hV_up
  sorry

end man_l7_7038


namespace billion_to_scientific_l7_7997
noncomputable def scientific_notation_of_billion (n : ℝ) : ℝ := n * 10^9
theorem billion_to_scientific (a : ℝ) : scientific_notation_of_billion a = 1.48056 * 10^11 :=
by sorry

end billion_to_scientific_l7_7997


namespace cubic_meter_to_cubic_centimeters_l7_7235

theorem cubic_meter_to_cubic_centimeters :
  (1 : ℝ) ^ 3 = (100 : ℝ) ^ 3 := by
  sorry

end cubic_meter_to_cubic_centimeters_l7_7235


namespace quadratic_sum_of_roots_l7_7910

theorem quadratic_sum_of_roots (a b : ℝ)
  (h1: ∀ x: ℝ, x^2 + b * x - a < 0 ↔ 3 < x ∧ x < 4):
  a + b = -19 :=
sorry

end quadratic_sum_of_roots_l7_7910


namespace nine_digit_not_perfect_square_l7_7067

theorem nine_digit_not_perfect_square (D : ℕ) (h1 : 100000000 ≤ D) (h2 : D < 1000000000)
  (h3 : ∀ c : ℕ, (c ∈ D.digits 10) → (c ≠ 0)) (h4 : D % 10 = 5) :
  ¬ ∃ A : ℕ, D = A ^ 2 := 
sorry

end nine_digit_not_perfect_square_l7_7067


namespace staircase_tile_cover_possible_l7_7708
-- Import the necessary Lean Lean libraries

-- We use natural numbers here
open Nat

-- Declare the problem as a theorem in Lean
theorem staircase_tile_cover_possible (m n : ℕ) (h_m : 6 ≤ m) (h_n : 6 ≤ n) :
  (∃ a b, m = 12 * a ∧ n = b ∧ a ≥ 1 ∧ b ≥ 6) ∨ 
  (∃ c d, m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) :=
sorry

end staircase_tile_cover_possible_l7_7708


namespace zero_a_and_b_l7_7299

theorem zero_a_and_b (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_a_and_b_l7_7299


namespace problem_statement_l7_7949

theorem problem_statement
  (a b c : ℝ) 
  (X : ℝ) 
  (hX : X = a + b + c + 2 * Real.sqrt (a^2 + b^2 + c^2 - a * b - b * c - c * a)) :
  X ≥ max (max (3 * a) (3 * b)) (3 * c) ∧ 
  ∃ (u v w : ℝ), 
    (u = Real.sqrt (X - 3 * a) ∧ v = Real.sqrt (X - 3 * b) ∧ w = Real.sqrt (X - 3 * c) ∧ 
     ((u + v = w) ∨ (v + w = u) ∨ (w + u = v))) :=
by
  sorry

end problem_statement_l7_7949


namespace factorize_poly1_factorize_poly2_l7_7869

-- Define y substitution for first problem
def poly1_y := fun (x : ℝ) => x^2 + 2*x
-- Define y substitution for second problem
def poly2_y := fun (x : ℝ) => x^2 - 4*x

-- Define the given polynomial expressions 
def poly1 := fun (x : ℝ) => (x^2 + 2*x)*(x^2 + 2*x + 2) + 1
def poly2 := fun (x : ℝ) => (x^2 - 4*x)*(x^2 - 4*x + 8) + 16

theorem factorize_poly1 (x : ℝ) : poly1 x = (x + 1) ^ 4 := sorry

theorem factorize_poly2 (x : ℝ) : poly2 x = (x - 2) ^ 4 := sorry

end factorize_poly1_factorize_poly2_l7_7869


namespace max_tan2alpha_l7_7071

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 2)
variable (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.tan (α + β) = 2 * Real.tan β)

theorem max_tan2alpha : 
    Real.tan (2 * α) = 4 * Real.sqrt 2 / 7 := 
by 
  sorry

end max_tan2alpha_l7_7071


namespace pond_to_field_ratio_l7_7585

theorem pond_to_field_ratio 
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l = 28)
  (side_pond : ℝ := 7) 
  (A_pond : ℝ := side_pond ^ 2) 
  (A_field : ℝ := l * w):
  (A_pond / A_field) = 1 / 8 :=
by
  sorry

end pond_to_field_ratio_l7_7585


namespace least_subtraction_divisibility_l7_7919

theorem least_subtraction_divisibility :
  ∃ k : ℕ, 427398 - k = 14 * n ∧ k = 6 :=
by
  use 6
  sorry

end least_subtraction_divisibility_l7_7919


namespace find_c_for_square_of_binomial_l7_7779

theorem find_c_for_square_of_binomial (c : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 50 * x + c = (x + b)^2) → c = 625 :=
by
  intro h
  obtain ⟨b, h⟩ := h
  sorry

end find_c_for_square_of_binomial_l7_7779


namespace red_balls_l7_7498

theorem red_balls (w r : ℕ) (h1 : w = 12) (h2 : w * 3 = r * 4) : r = 9 :=
sorry

end red_balls_l7_7498


namespace greta_hourly_wage_is_12_l7_7905

-- Define constants
def greta_hours : ℕ := 40
def lisa_hourly_wage : ℕ := 15
def lisa_hours : ℕ := 32

-- Define the total earnings of Greta and Lisa
def greta_earnings (G : ℕ) : ℕ := greta_hours * G
def lisa_earnings : ℕ := lisa_hours * lisa_hourly_wage

-- Main theorem statement
theorem greta_hourly_wage_is_12 (G : ℕ) (h : greta_earnings G = lisa_earnings) : G = 12 :=
by
  sorry

end greta_hourly_wage_is_12_l7_7905


namespace find_m_l7_7453

theorem find_m (a b c m : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_m : 0 < m) (h : a * b * c * m = 1 + a^2 + b^2 + c^2) : 
  m = 4 :=
sorry

end find_m_l7_7453


namespace John_is_26_l7_7465

-- Define the variables representing the ages
def John_age : ℕ := 26
def Grandmother_age : ℕ := John_age + 48

-- Conditions
def condition1 : Prop := John_age = Grandmother_age - 48
def condition2 : Prop := John_age + Grandmother_age = 100

-- Main theorem to prove: John is 26 years old
theorem John_is_26 : John_age = 26 :=
by
  have h1 : condition1 := by sorry
  have h2 : condition2 := by sorry
  -- More steps to combine the conditions and prove the theorem would go here
  -- Skipping proof steps with sorry for demonstration
  sorry

end John_is_26_l7_7465


namespace fraction_irreducible_l7_7354

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by {
    sorry
}

end fraction_irreducible_l7_7354


namespace polynomials_with_sum_of_abs_values_and_degree_eq_4_l7_7551

-- We define the general structure and conditions of the problem.
def polynomial_count : ℕ := 
  let count_0 := 1 -- For n = 0
  let count_1 := 6 -- For n = 1
  let count_2 := 9 -- For n = 2
  let count_3 := 1 -- For n = 3
  count_0 + count_1 + count_2 + count_3

theorem polynomials_with_sum_of_abs_values_and_degree_eq_4 : polynomial_count = 17 := 
by
  unfold polynomial_count
  -- The detailed proof steps for the count would go here
  sorry

end polynomials_with_sum_of_abs_values_and_degree_eq_4_l7_7551


namespace uncolored_vertex_not_original_hexagon_vertex_l7_7662

theorem uncolored_vertex_not_original_hexagon_vertex
    (point_index : ℕ)
    (orig_hex_vertices : Finset ℕ) -- Assuming the vertices of the original hexagon are represented as a finite set of indices.
    (num_parts : ℕ := 1000) -- Each hexagon side is divided into 1000 parts
    (label : ℕ → Fin 3) -- A function labeling each point with 0, 1, or 2.
    (is_valid_labeling : ∀ (i j k : ℕ), label i ≠ label j ∧ label j ≠ label k ∧ label k ≠ label i) -- No duplicate labeling within a triangle.
    (is_single_uncolored : ∀ (p : ℕ), (p ∈ orig_hex_vertices ∨ ∃ (v : ℕ), v ∈ orig_hex_vertices ∧ p = v) → p ≠ point_index) -- Only one uncolored point
    : point_index ∉ orig_hex_vertices :=
by sorry

end uncolored_vertex_not_original_hexagon_vertex_l7_7662


namespace candles_shared_equally_l7_7043

theorem candles_shared_equally :
  ∀ (Aniyah Ambika Bree Caleb : ℕ),
  Aniyah = 6 * Ambika → Ambika = 4 → Bree = 0 → Caleb = 0 →
  (Aniyah + Ambika + Bree + Caleb) / 4 = 7 :=
by
  intros Aniyah Ambika Bree Caleb h1 h2 h3 h4
  sorry

end candles_shared_equally_l7_7043


namespace solve_trig_system_l7_7471

theorem solve_trig_system
  (k n : ℤ) :
  (∃ x y : ℝ,
    (2 * Real.sin x ^ 2 + 2 * Real.sqrt 2 * Real.sin x * Real.sin (2 * x) ^ 2 + Real.sin (2 * x) ^ 2 = 0 ∧
     Real.cos x = Real.cos y) ∧
    ((x = 2 * Real.pi * k ∧ y = 2 * Real.pi * n) ∨
     (x = Real.pi + 2 * Real.pi * k ∧ y = Real.pi + 2 * Real.pi * n) ∨
     (x = -Real.pi / 4 + 2 * Real.pi * k ∧ (y = Real.pi / 4 + 2 * Real.pi * n ∨ y = -Real.pi / 4 + 2 * Real.pi * n)) ∨
     (x = -3 * Real.pi / 4 + 2 * Real.pi * k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * n)))) :=
sorry

end solve_trig_system_l7_7471


namespace Jaco_budget_for_parents_gifts_l7_7647

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l7_7647


namespace customers_in_other_countries_l7_7660

def total_customers : ℕ := 7422
def us_customers : ℕ := 723
def other_customers : ℕ := total_customers - us_customers

theorem customers_in_other_countries : other_customers = 6699 := by
  sorry

end customers_in_other_countries_l7_7660


namespace inequality_property_l7_7129

variable {a b : ℝ} (h : a > b) (c : ℝ)

theorem inequality_property : a * |c| ≥ b * |c| :=
sorry

end inequality_property_l7_7129


namespace monotonicity_and_inequality_l7_7548

noncomputable def f (x : ℝ) := 2 * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) (hF_nonneg : ∀ x, F a x ≥ 0) (h_lt : x₁ < x₂) :
  (F a x₂ - F a x₁) / (x₂ - x₁) > 2 * (Real.exp x₁ - 1) :=
sorry

end monotonicity_and_inequality_l7_7548


namespace fiona_working_hours_l7_7538

theorem fiona_working_hours (F : ℕ) 
  (John_hours_per_week : ℕ := 30) 
  (Jeremy_hours_per_week : ℕ := 25) 
  (pay_rate : ℕ := 20) 
  (monthly_total_pay : ℕ := 7600) : 
  4 * (John_hours_per_week * pay_rate + Jeremy_hours_per_week * pay_rate + F * pay_rate) = monthly_total_pay → 
  F = 40 :=
by sorry

end fiona_working_hours_l7_7538


namespace cost_price_of_ball_l7_7200

theorem cost_price_of_ball (x : ℝ) (h : 17 * x - 5 * x = 720) : x = 60 :=
by {
  sorry
}

end cost_price_of_ball_l7_7200


namespace min_value_x_add_one_div_y_l7_7791

theorem min_value_x_add_one_div_y (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) : 
x + 1 / y ≥ 3 :=
sorry

end min_value_x_add_one_div_y_l7_7791


namespace swimming_both_days_l7_7958

theorem swimming_both_days
  (total_students swimming_today soccer_today : ℕ)
  (students_swimming_yesterday students_soccer_yesterday : ℕ)
  (soccer_today_swimming_yesterday soccer_today_soccer_yesterday : ℕ)
  (swimming_today_swimming_yesterday swimming_today_soccer_yesterday : ℕ) :
  total_students = 33 ∧
  swimming_today = 22 ∧
  soccer_today = 22 ∧
  soccer_today_swimming_yesterday = 15 ∧
  soccer_today_soccer_yesterday = 15 ∧
  swimming_today_swimming_yesterday = 15 ∧
  swimming_today_soccer_yesterday = 15 →
  ∃ (swimming_both_days : ℕ), swimming_both_days = 4 :=
by
  sorry

end swimming_both_days_l7_7958


namespace no_solution_exists_l7_7351

theorem no_solution_exists : ¬ ∃ n : ℕ, 0 < n ∧ (2^n % 60 = 29 ∨ 2^n % 60 = 31) := 
by
  sorry

end no_solution_exists_l7_7351


namespace minimum_value_inequality_l7_7920

open Real

theorem minimum_value_inequality
  (a b c : ℝ)
  (ha : 2 ≤ a) 
  (hb : a ≤ b)
  (hc : b ≤ c)
  (hd : c ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 = 4 * (sqrt 5 ^ (1 / 4) - 1)^2 :=
sorry

end minimum_value_inequality_l7_7920


namespace man_speed_still_water_l7_7860

noncomputable def speed_in_still_water (U D : ℝ) : ℝ := (U + D) / 2

theorem man_speed_still_water :
  let U := 45
  let D := 55
  speed_in_still_water U D = 50 := by
  sorry

end man_speed_still_water_l7_7860


namespace polygon_diagonals_30_l7_7250

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end polygon_diagonals_30_l7_7250


namespace find_k_l7_7550

theorem find_k (k x y : ℝ) (h1 : x = 2) (h2 : y = -3)
    (h3 : 2 * x^2 + k * x * y = 4) : k = 2 / 3 :=
by
  sorry

end find_k_l7_7550


namespace quadratic_no_real_roots_l7_7933

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end quadratic_no_real_roots_l7_7933


namespace percentage_of_part_whole_l7_7294

theorem percentage_of_part_whole (part whole : ℝ) (h_part : part = 75) (h_whole : whole = 125) : 
  (part / whole) * 100 = 60 :=
by
  rw [h_part, h_whole]
  -- Simplification steps would follow, but we substitute in the placeholders
  sorry

end percentage_of_part_whole_l7_7294


namespace friend_redistribute_l7_7525

-- Definition and total earnings
def earnings : List Int := [30, 45, 15, 10, 60]
def total_earnings := earnings.sum

-- Number of friends
def number_of_friends : Int := 5

-- Calculate the equal share
def equal_share := total_earnings / number_of_friends

-- Calculate the amount to redistribute by the friend who earned 60
def amount_to_give := 60 - equal_share

theorem friend_redistribute :
  earnings.sum = 160 ∧ equal_share = 32 ∧ amount_to_give = 28 :=
by
  -- Proof goes here, skipped with 'sorry'
  sorry

end friend_redistribute_l7_7525


namespace find_plot_width_l7_7942

theorem find_plot_width:
  let length : ℝ := 360
  let area_acres : ℝ := 10
  let square_feet_per_acre : ℝ := 43560
  let area_square_feet := area_acres * square_feet_per_acre
  let width := area_square_feet / length
  area_square_feet = 435600 ∧ length = 360 ∧ square_feet_per_acre = 43560
  → width = 1210 :=
by
  intro h
  sorry

end find_plot_width_l7_7942


namespace lassis_from_12_mangoes_l7_7110

-- Conditions as definitions in Lean 4
def total_mangoes : ℕ := 12
def damaged_mango_ratio : ℕ := 1 / 6
def lassis_per_pair_mango : ℕ := 11

-- Equation to calculate the lassis
theorem lassis_from_12_mangoes : (total_mangoes - total_mangoes / 6) / 2 * lassis_per_pair_mango = 55 :=
by
  -- calculation steps should go here, but are omitted as per instructions
  sorry

end lassis_from_12_mangoes_l7_7110


namespace polynomial_divisibility_l7_7098

theorem polynomial_divisibility (t : ℤ) : 
  (∀ x : ℤ, (5 * x^3 - 15 * x^2 + t * x - 20) ∣ (x - 2)) → (t = 20) → 
  ∀ x : ℤ, (5 * x^3 - 15 * x^2 + 20 * x - 20) ∣ (5 * x^2 + 5 * x + 5) :=
by
  intro h₁ h₂
  sorry

end polynomial_divisibility_l7_7098


namespace contrapositive_of_neg_and_inverse_l7_7583

theorem contrapositive_of_neg_and_inverse (p r s : Prop) (h1 : r = ¬p) (h2 : s = ¬r) : s = (¬p → false) :=
by
  -- We have that r = ¬p
  have hr : r = ¬p := h1
  -- And we have that s = ¬r
  have hs : s = ¬r := h2
  -- Now we need to show that s is the contrapositive of p, which is ¬p → false
  sorry

end contrapositive_of_neg_and_inverse_l7_7583


namespace car_price_difference_l7_7884

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end car_price_difference_l7_7884


namespace only_n_divides_2_pow_n_minus_1_l7_7924

theorem only_n_divides_2_pow_n_minus_1 : ∀ (n : ℕ), n > 0 ∧ n ∣ (2^n - 1) ↔ n = 1 := by
  sorry

end only_n_divides_2_pow_n_minus_1_l7_7924


namespace solve_eq_simplify_expression_l7_7263

-- Part 1: Prove the solution to the given equation

theorem solve_eq (x : ℚ) : (1 / (x - 1) + 1 = 3 / (2 * x - 2)) → x = 3 / 2 :=
sorry

-- Part 2: Prove the simplified value of the given expression when x=1/2

theorem simplify_expression : (x = 1/2) →
  ((x^2 / (1 + x) - x) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1) :=
sorry

end solve_eq_simplify_expression_l7_7263


namespace find_cost_price_l7_7575

variable (C : ℝ)

def profit_10_percent_selling_price := 1.10 * C

def profit_15_percent_with_150_more := 1.10 * C + 150

def profit_15_percent_selling_price := 1.15 * C

theorem find_cost_price
  (h : profit_15_percent_with_150_more C = profit_15_percent_selling_price C) :
  C = 3000 :=
by
  sorry

end find_cost_price_l7_7575


namespace total_cost_of_projectors_and_computers_l7_7773

theorem total_cost_of_projectors_and_computers :
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  (n_p * c_p + n_c * c_c) = 175200 := by
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  sorry 

end total_cost_of_projectors_and_computers_l7_7773


namespace ratio_population_X_to_Z_l7_7808

-- Given definitions
def population_of_Z : ℕ := sorry
def population_of_Y : ℕ := 2 * population_of_Z
def population_of_X : ℕ := 5 * population_of_Y

-- Theorem to prove
theorem ratio_population_X_to_Z : population_of_X / population_of_Z = 10 :=
by
  sorry

end ratio_population_X_to_Z_l7_7808


namespace amanda_more_than_average_l7_7628

-- Conditions
def jill_peaches : ℕ := 12
def steven_peaches : ℕ := jill_peaches + 15
def jake_peaches : ℕ := steven_peaches - 16
def amanda_peaches : ℕ := jill_peaches * 2
def total_peaches : ℕ := jake_peaches + steven_peaches + jill_peaches
def average_peaches : ℚ := total_peaches / 3

-- Question: Prove that Amanda has 7.33 more peaches than the average peaches Jake, Steven, and Jill have
theorem amanda_more_than_average : amanda_peaches - average_peaches = 22 / 3 := by
  sorry

end amanda_more_than_average_l7_7628


namespace sqrt_0_54_in_terms_of_a_b_l7_7120

variable (a b : ℝ)

-- Conditions
def sqrt_two_eq_a : Prop := a = Real.sqrt 2
def sqrt_three_eq_b : Prop := b = Real.sqrt 3

-- The main statement to prove
theorem sqrt_0_54_in_terms_of_a_b (h1 : sqrt_two_eq_a a) (h2 : sqrt_three_eq_b b) :
  Real.sqrt 0.54 = 0.3 * a * b := sorry

end sqrt_0_54_in_terms_of_a_b_l7_7120


namespace geometric_sequence_sum_squared_l7_7620

theorem geometric_sequence_sum_squared (a : ℕ → ℕ) (n : ℕ) (q : ℕ) 
    (h_geometric: ∀ n, a (n + 1) = a n * q)
    (h_a1 : a 1 = 2)
    (h_a3 : a 3 = 4) :
    (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 = 1020 :=
by
  sorry

end geometric_sequence_sum_squared_l7_7620


namespace wrapping_paper_area_l7_7001

theorem wrapping_paper_area (l w h : ℝ) (hlw : l > w) (hwh : w > h) (hl : l = 2 * w) : 
    (∃ a : ℝ, a = 5 * w^2 + h^2) :=
by 
  sorry

end wrapping_paper_area_l7_7001


namespace remainder_x1001_mod_poly_l7_7740

noncomputable def remainder_poly_div (n k : ℕ) (f g : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.modByMonic f g

theorem remainder_x1001_mod_poly :
  remainder_poly_div 1001 3 (Polynomial.X ^ 1001) (Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X + 1) = Polynomial.X ^ 2 :=
by
  sorry

end remainder_x1001_mod_poly_l7_7740


namespace total_heads_l7_7319

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end total_heads_l7_7319


namespace circle_radius_tangent_l7_7298

theorem circle_radius_tangent (A B O M X : Type) (AB AM MB r : ℝ)
  (hL1 : AB = 2) (hL2 : AM = 1) (hL3 : MB = 1) (hMX : MX = 1/2)
  (hTangent1 : OX = 1/2 + r) (hTangent2 : OM = 1 - r)
  (hPythagorean : OM^2 + MX^2 = OX^2) :
  r = 1/3 :=
by
  sorry

end circle_radius_tangent_l7_7298


namespace cube_surface_area_l7_7698

/-- A cube with an edge length of 10 cm has smaller cubes with edge length 2 cm 
    dug out from the middle of each face. The surface area of the new shape is 696 cm². -/
theorem cube_surface_area (original_edge : ℝ) (small_cube_edge : ℝ)
  (original_edge_eq : original_edge = 10) (small_cube_edge_eq : small_cube_edge = 2) :
  let original_surface := 6 * original_edge ^ 2
  let removed_area := 6 * small_cube_edge ^ 2
  let added_area := 6 * 5 * small_cube_edge ^ 2
  let new_surface := original_surface - removed_area + added_area
  new_surface = 696 := by
  sorry

end cube_surface_area_l7_7698


namespace total_interest_is_68_l7_7852

-- Definitions of the initial conditions
def amount_2_percent : ℝ := 600
def amount_4_percent : ℝ := amount_2_percent + 800
def interest_rate_2_percent : ℝ := 0.02
def interest_rate_4_percent : ℝ := 0.04
def invested_total_1 : ℝ := amount_2_percent
def invested_total_2 : ℝ := amount_4_percent

-- The total interest calculation
def interest_2_percent : ℝ := invested_total_1 * interest_rate_2_percent
def interest_4_percent : ℝ := invested_total_2 * interest_rate_4_percent

-- Claim: The total interest earned is $68
theorem total_interest_is_68 : interest_2_percent + interest_4_percent = 68 := by
  sorry

end total_interest_is_68_l7_7852


namespace bicycle_car_speed_l7_7490

theorem bicycle_car_speed (x : Real) (h1 : x > 0) :
  10 / x - 10 / (2 * x) = 1 / 3 :=
by
  sorry

end bicycle_car_speed_l7_7490


namespace passenger_gets_ticket_l7_7882

variables (p1 p2 p3 p4 p5 p6 : ℝ)

-- Conditions:
axiom h_sum_eq_one : p1 + p2 + p3 = 1
axiom h_p1_nonneg : 0 ≤ p1
axiom h_p2_nonneg : 0 ≤ p2
axiom h_p3_nonneg : 0 ≤ p3
axiom h_p4_nonneg : 0 ≤ p4
axiom h_p4_le_one : p4 ≤ 1
axiom h_p5_nonneg : 0 ≤ p5
axiom h_p5_le_one : p5 ≤ 1
axiom h_p6_nonneg : 0 ≤ p6
axiom h_p6_le_one : p6 ≤ 1

-- Theorem:
theorem passenger_gets_ticket :
  (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) = (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) :=
by sorry

end passenger_gets_ticket_l7_7882


namespace reflected_ray_bisects_circle_circumference_l7_7270

open Real

noncomputable def equation_of_line_reflected_ray : Prop :=
  ∃ (m b : ℝ), (m = 2 / (-3 + 1)) ∧ (b = (3/(-5 + 5)) + 1) ∧ ((-5, -3) = (-5, (-5*m + b))) ∧ ((1, 1) = (1, (1*m + b)))

theorem reflected_ray_bisects_circle_circumference :
  equation_of_line_reflected_ray ↔ ∃ a b c : ℝ, (a = 2) ∧ (b = -3) ∧ (c = 1) ∧ (a*x + b*y + c = 0) :=
by
  sorry

end reflected_ray_bisects_circle_circumference_l7_7270


namespace projectile_height_30_in_2_seconds_l7_7877

theorem projectile_height_30_in_2_seconds (t y : ℝ) : 
  (y = -5 * t^2 + 25 * t ∧ y = 30) → t = 2 :=
by
  sorry

end projectile_height_30_in_2_seconds_l7_7877


namespace magnolia_trees_below_threshold_l7_7022

-- Define the initial number of trees and the function describing the decrease
def initial_tree_count (N₀ : ℕ) (t : ℕ) : ℝ := N₀ * (0.8 ^ t)

-- Define the year when the number of trees is less than 25% of initial trees
theorem magnolia_trees_below_threshold (N₀ : ℕ) : (t : ℕ) -> initial_tree_count N₀ t < 0.25 * N₀ -> t > 14 := 
-- Provide the required statement but omit the actual proof with "sorry"
by sorry

end magnolia_trees_below_threshold_l7_7022


namespace remainder_of_k_divided_by_7_l7_7234

theorem remainder_of_k_divided_by_7 :
  ∃ k < 42, k % 5 = 2 ∧ k % 6 = 5 ∧ k % 7 = 3 :=
by {
  -- The proof is supplied here
  sorry
}

end remainder_of_k_divided_by_7_l7_7234


namespace interest_calculation_years_l7_7744

noncomputable def principal : ℝ := 625
noncomputable def rate : ℝ := 0.04
noncomputable def difference : ℝ := 1

theorem interest_calculation_years (n : ℕ) : 
    (principal * (1 + rate)^n - principal - (principal * rate * n) = difference) → 
    n = 2 :=
by sorry

end interest_calculation_years_l7_7744


namespace range_of_a_l7_7758

def p (a : ℝ) : Prop := a > -1
def q (a : ℝ) : Prop := ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by
  sorry

end range_of_a_l7_7758


namespace find_measure_angle_AOD_l7_7393

-- Definitions of angles in the problem
def angle_COA := 150
def angle_BOD := 120

-- Definition of the relationship between angles
def angle_AOD_eq_four_times_angle_BOC (x : ℝ) : Prop :=
  4 * x = 360

-- Proof Problem Lean Statement
theorem find_measure_angle_AOD (x : ℝ) (h1 : 180 - 30 = angle_COA) (h2 : 180 - 60 = angle_BOD) (h3 : angle_AOD_eq_four_times_angle_BOC x) : 
  4 * x = 360 :=
  by 
  -- Insert necessary steps here
  sorry

end find_measure_angle_AOD_l7_7393


namespace factorize_16x2_minus_1_l7_7710

theorem factorize_16x2_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4 * x + 1) * (4 * x - 1) := by
  sorry

end factorize_16x2_minus_1_l7_7710


namespace heartsuit_3_8_l7_7219

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l7_7219


namespace natural_numbers_not_divisible_by_5_or_7_l7_7140

def num_not_divisible_by_5_or_7 (n : ℕ) : ℕ :=
  let num_div_5 := n / 5
  let num_div_7 := n / 7
  let num_div_35 := n / 35
  n - (num_div_5 + num_div_7 - num_div_35)

theorem natural_numbers_not_divisible_by_5_or_7 :
  num_not_divisible_by_5_or_7 999 = 686 :=
by sorry

end natural_numbers_not_divisible_by_5_or_7_l7_7140


namespace vertex_of_parabola_l7_7193

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l7_7193


namespace campers_rowing_morning_equals_41_l7_7331

def campers_went_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total : ℕ) : ℕ :=
  total - (hiking_morning + rowing_afternoon)

theorem campers_rowing_morning_equals_41 :
  ∀ (hiking_morning rowing_afternoon total : ℕ), hiking_morning = 4 → rowing_afternoon = 26 → total = 71 → campers_went_rowing_morning hiking_morning rowing_afternoon total = 41 := by
  intros hiking_morning rowing_afternoon total hiking_morning_cond rowing_afternoon_cond total_cond
  rw [hiking_morning_cond, rowing_afternoon_cond, total_cond]
  exact rfl

end campers_rowing_morning_equals_41_l7_7331


namespace harrys_total_cost_l7_7893

def cost_large_pizza : ℕ := 14
def cost_per_topping : ℕ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℚ := 0.25

def total_cost (c_pizza c_topping tip_percent : ℚ) (n_pizza n_topping : ℕ) : ℚ :=
  let inital_cost := (c_pizza + c_topping * n_topping) * n_pizza
  let tip := inital_cost * tip_percent
  inital_cost + tip

theorem harrys_total_cost : total_cost 14 2 0.25 2 3 = 50 := 
  sorry

end harrys_total_cost_l7_7893


namespace smaller_area_l7_7274

theorem smaller_area (A B : ℝ) (total_area : A + B = 1800) (diff_condition : B - A = (A + B) / 6) :
  A = 750 := 
by
  sorry

end smaller_area_l7_7274


namespace a_work_days_alone_l7_7774

-- Definitions based on conditions
def work_days_a   (a: ℝ)    : Prop := ∃ (x:ℝ), a = x
def work_days_b   (b: ℝ)    : Prop := b = 36
def alternate_work (a b W x: ℝ) : Prop := 9 * (W / 36 + W / x) = W ∧ x > 0

-- The main theorem to prove
theorem a_work_days_alone (x W: ℝ) (b: ℝ) (h_work_days_b: work_days_b b)
                          (h_alternate_work: alternate_work a b W x) : 
                          work_days_a a → a = 12 :=
by sorry

end a_work_days_alone_l7_7774


namespace y_is_less_than_x_by_9444_percent_l7_7973

theorem y_is_less_than_x_by_9444_percent (x y : ℝ) (h : x = 18 * y) : (x - y) / x * 100 = 94.44 :=
by
  sorry

end y_is_less_than_x_by_9444_percent_l7_7973


namespace amy_red_balloons_l7_7392

theorem amy_red_balloons (total_balloons green_balloons blue_balloons : ℕ) (h₁ : total_balloons = 67) (h₂: green_balloons = 17) (h₃ : blue_balloons = 21) : (total_balloons - (green_balloons + blue_balloons)) = 29 :=
by
  sorry

end amy_red_balloons_l7_7392


namespace distance_between_parallel_lines_l7_7889

theorem distance_between_parallel_lines : 
  ∀ (x y : ℝ), 
  (3 * x - 4 * y - 3 = 0) ∧ (6 * x - 8 * y + 5 = 0) → 
  ∃ d : ℝ, d = 11 / 10 :=
by
  sorry

end distance_between_parallel_lines_l7_7889


namespace prime_number_solution_l7_7657

theorem prime_number_solution (X Y : ℤ) (h_prime : Prime (X^4 + 4 * Y^4)) :
  (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) :=
sorry

end prime_number_solution_l7_7657


namespace employee_pay_l7_7039

variable (X Y : ℝ)

theorem employee_pay (h1: X + Y = 572) (h2: X = 1.2 * Y) : Y = 260 :=
by
  sorry

end employee_pay_l7_7039


namespace min_distinct_sums_l7_7437

theorem min_distinct_sums (n : ℕ) (hn : n ≥ 5) (s : Finset ℕ) 
  (hs : s.card = n) : 
  ∃ (t : Finset ℕ), (∀ (x y : ℕ), x ∈ s → y ∈ s → x < y → (x + y) ∈ t) ∧ t.card = 2 * n - 3 :=
by
  sorry

end min_distinct_sums_l7_7437


namespace fill_time_l7_7061

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time_l7_7061


namespace find_g_inv_f_3_l7_7562

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq : ∀ x : ℝ, f_inv (g x) = x^4 - x + 2
axiom g_has_inverse : ∀ y : ℝ, g (g_inv y) = y 

theorem find_g_inv_f_3 :
  ∃ α : ℝ, (α^4 - α - 1 = 0) ∧ g_inv (f 3) = α :=
sorry

end find_g_inv_f_3_l7_7562


namespace max_parrots_l7_7352

theorem max_parrots (x y z : ℕ) (h1 : y + z ≤ 9) (h2 : x + z ≤ 11) : x + y + z ≤ 19 :=
sorry

end max_parrots_l7_7352


namespace range_of_a_l7_7091

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / (4 * x)
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) → 
  2 * Real.sqrt (Real.exp 1 - 2) ≤ a := sorry

end range_of_a_l7_7091


namespace find_angle_and_area_of_triangle_l7_7213

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

end find_angle_and_area_of_triangle_l7_7213


namespace min_expr_value_min_expr_value_iff_l7_7570

theorem min_expr_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 :=
by {
  sorry
}

theorem min_expr_value_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2) = 4 / 9) ↔ (x = 2.5 ∧ y = 2.5) :=
by {
  sorry
}

end min_expr_value_min_expr_value_iff_l7_7570


namespace identify_first_brother_l7_7320

-- Definitions for conditions
inductive Brother
| Trulya : Brother
| Falsa : Brother

-- Extracting conditions into Lean 4 statements
def first_brother_says : String := "Both cards are of the purplish suit."
def second_brother_says : String := "This is not true!"

axiom trulya_always_truthful : ∀ (b : Brother) (statement : String), b = Brother.Trulya ↔ (statement = first_brother_says ∨ statement = second_brother_says)
axiom falsa_always_lies : ∀ (b : Brother) (statement : String), b = Brother.Falsa ↔ ¬(statement = first_brother_says ∨ statement = second_brother_says)

-- Proof statement 
theorem identify_first_brother :
  ∃ (b : Brother), b = Brother.Trulya :=
sorry

end identify_first_brother_l7_7320


namespace units_digit_p_plus_one_l7_7637

theorem units_digit_p_plus_one (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 ≠ 0)
  (h3 : (p ^ 3) % 10 = (p ^ 2) % 10) : (p + 1) % 10 = 7 :=
  sorry

end units_digit_p_plus_one_l7_7637


namespace cos_square_minus_sin_square_15_l7_7595

theorem cos_square_minus_sin_square_15 (cos_30 : Real.cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2) : 
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = (Real.sqrt 3) / 2 := 
by 
  sorry

end cos_square_minus_sin_square_15_l7_7595


namespace partition_value_l7_7694

variable {a m n p x k l : ℝ}

theorem partition_value :
  (m * (a - n * x) = k * (a - n * x)) ∧
  (n * x = l * x) ∧
  (a - x = p * (a - m * (a - n * x)))
  → x = (a * (m * p - p + 1)) / (n * m * p + 1) :=
by
  sorry

end partition_value_l7_7694


namespace area_of_square_l7_7415

theorem area_of_square (A_circle : ℝ) (hA_circle : A_circle = 39424) (cm_to_inch : ℝ) (hcm_to_inch : cm_to_inch = 2.54) :
  ∃ (A_square : ℝ), A_square = 121.44 := 
by
  sorry

end area_of_square_l7_7415


namespace max_marks_paper_one_l7_7977

theorem max_marks_paper_one (M : ℝ) : 
  (0.42 * M = 64) → (M = 152) :=
by
  sorry

end max_marks_paper_one_l7_7977


namespace measure_of_angle_C_l7_7443

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 12 * D) : C = 2160 / 13 :=
by {
  sorry
}

end measure_of_angle_C_l7_7443


namespace value_of_polynomial_l7_7952

theorem value_of_polynomial : 
  99^5 - 5 * 99^4 + 10 * 99^3 - 10 * 99^2 + 5 * 99 - 1 = 98^5 := by
  sorry

end value_of_polynomial_l7_7952


namespace average_height_of_trees_l7_7268

theorem average_height_of_trees :
  ∃ (h : ℕ → ℕ), (h 2 = 12) ∧ (∀ i, h i = 2 * h (i+1) ∨ h i = h (i+1) / 2) ∧ (h 1 * h 2 * h 3 * h 4 * h 5 * h 6 = 4608) →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 21 :=
sorry

end average_height_of_trees_l7_7268


namespace passing_marks_l7_7024

theorem passing_marks :
  ∃ P T : ℝ, (0.2 * T = P - 40) ∧ (0.3 * T = P + 20) ∧ P = 160 :=
by
  sorry

end passing_marks_l7_7024


namespace minimize_expr_l7_7142

-- Define the function we need to minimize
noncomputable def expr (α β : ℝ) : ℝ := 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2

-- State the theorem to prove the minimum value of this expression
theorem minimize_expr (α β : ℝ) : ∃ (α β : ℝ), expr α β = 100 := 
sorry

end minimize_expr_l7_7142


namespace total_number_of_balls_in_fish_tank_l7_7663

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end total_number_of_balls_in_fish_tank_l7_7663


namespace chipmunk_acorns_l7_7468

theorem chipmunk_acorns :
  ∀ (x y : ℕ), (3 * x = 4 * y) → (y = x - 4) → (3 * x = 48) :=
by
  intros x y h1 h2
  sorry

end chipmunk_acorns_l7_7468


namespace volume_ratio_cones_l7_7582

theorem volume_ratio_cones :
  let rC := 16.5
  let hC := 33
  let rD := 33
  let hD := 16.5
  let VC := (1 / 3) * Real.pi * rC^2 * hC
  let VD := (1 / 3) * Real.pi * rD^2 * hD
  (VC / VD) = (1 / 2) :=
by
  sorry

end volume_ratio_cones_l7_7582


namespace value_of_f_at_5_l7_7644

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_f_at_5 : f 5 = 15 := 
by {
  sorry
}

end value_of_f_at_5_l7_7644


namespace smallest_positive_angle_l7_7876

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : ∃ β : ℝ, 0 < β ∧ β < 360 ∧ β = α % 360 := by
  sorry

end smallest_positive_angle_l7_7876


namespace smallest_number_is_a_l7_7775

def smallest_number_among_options : ℤ :=
  let a: ℤ := -3
  let b: ℤ := 0
  let c: ℤ := -(-1)
  let d: ℤ := (-1)^2
  min a (min b (min c d))

theorem smallest_number_is_a : smallest_number_among_options = -3 :=
  by
    sorry

end smallest_number_is_a_l7_7775


namespace find_y_l7_7323

theorem find_y (y : ℝ) (h : 3 * y / 4 = 15) : y = 20 :=
sorry

end find_y_l7_7323


namespace donna_smallest_n_l7_7467

theorem donna_smallest_n (n : ℕ) : 15 * n - 1 % 6 = 0 ↔ n % 6 = 5 := sorry

end donna_smallest_n_l7_7467


namespace ceil_square_eq_four_l7_7561

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l7_7561


namespace original_number_of_people_l7_7386

-- Define the conditions as Lean definitions
def two_thirds_left (x : ℕ) : ℕ := (2 * x) / 3
def one_fourth_dancing_left (x : ℕ) : ℕ := ((x / 3) - (x / 12))

-- The problem statement as Lean theorem
theorem original_number_of_people (x : ℕ) (h : x / 4 = 15) : x = 60 :=
by sorry

end original_number_of_people_l7_7386


namespace find_a_b_l7_7034

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

def f_derivative (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 10) (h2 : f_derivative 1 a b = 0) : a = 4 ∧ b = -11 :=
sorry

end find_a_b_l7_7034


namespace math_problem_l7_7598

variable (a : ℝ)

theorem math_problem (h : a^2 + 3 * a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2 * a)) = 1 := 
sorry

end math_problem_l7_7598


namespace find_missing_number_l7_7927

theorem find_missing_number :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 :=
by
  intros h1 h2
  sorry

end find_missing_number_l7_7927


namespace seeds_total_l7_7763

-- Define the conditions as given in the problem.
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds

-- Lean statement to prove the total number of seeds.
theorem seeds_total : Bom_seeds + Gwi_seeds + Yeon_seeds = 1660 := 
by
  -- Assuming all given definitions and conditions are true,
  -- we aim to prove the final theorem statement.
  sorry

end seeds_total_l7_7763


namespace maximize_fraction_l7_7931

theorem maximize_fraction (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9)
  (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 0 ≤ D)
  (h_integer : (A + B) % (C + D) = 0) : A + B = 17 :=
sorry

end maximize_fraction_l7_7931


namespace solution_set_inequality_l7_7436

theorem solution_set_inequality (x : ℝ) : (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1 / 3) :=
by
  sorry

end solution_set_inequality_l7_7436


namespace abs_sum_leq_abs_l7_7429

theorem abs_sum_leq_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| + |b| ≤ |a + b| :=
sorry

end abs_sum_leq_abs_l7_7429


namespace tan_120_deg_l7_7821

theorem tan_120_deg : Real.tan (120 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_120_deg_l7_7821


namespace equilateral_triangle_vertex_distance_l7_7999

noncomputable def distance_vertex_to_center (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + (l^2 / 4))

theorem equilateral_triangle_vertex_distance
  (l r : ℝ)
  (h1 : l > 0)
  (h2 : r > 0) :
  distance_vertex_to_center l r = Real.sqrt (r^2 + (l^2 / 4)) :=
sorry

end equilateral_triangle_vertex_distance_l7_7999


namespace circle_in_quad_radius_l7_7439

theorem circle_in_quad_radius (AB BC CD DA : ℝ) (r : ℝ) (h₁ : AB = 15) (h₂ : BC = 10) (h₃ : CD = 8) (h₄ : DA = 13) :
  r = 2 * Real.sqrt 10 := 
by {
  sorry
  }

end circle_in_quad_radius_l7_7439


namespace james_profit_l7_7168

/--
  Prove that James's profit from buying 200 lotto tickets at $2 each, given the 
  conditions about winning tickets, is $4,830.
-/
theorem james_profit 
  (total_tickets : ℕ := 200)
  (cost_per_ticket : ℕ := 2)
  (winner_percentage : ℝ := 0.2)
  (five_dollar_win_pct : ℝ := 0.8)
  (grand_prize : ℝ := 5000)
  (average_other_wins : ℝ := 10) :
  let total_cost := total_tickets * cost_per_ticket 
  let total_winners := winner_percentage * total_tickets
  let five_dollar_winners := five_dollar_win_pct * total_winners
  let total_five_dollar := five_dollar_winners * 5
  let remaining_winners := total_winners - 1 - five_dollar_winners
  let total_remaining_winners := remaining_winners * average_other_wins
  let total_winnings := total_five_dollar + grand_prize + total_remaining_winners
  let profit := total_winnings - total_cost
  profit = 4830 :=
by
  sorry

end james_profit_l7_7168


namespace negation_of_proposition_l7_7643

theorem negation_of_proposition (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) :=
by sorry

end negation_of_proposition_l7_7643


namespace tangent_line_x_squared_at_one_one_l7_7339

open Real

theorem tangent_line_x_squared_at_one_one :
  ∀ (x y : ℝ), y = x^2 → (x, y) = (1, 1) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_x_squared_at_one_one_l7_7339


namespace range_of_k_l7_7719

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → 0 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l7_7719


namespace speed_of_second_train_l7_7276

-- Definitions of given conditions
def length_first_train : ℝ := 60 
def length_second_train : ℝ := 280 
def speed_first_train : ℝ := 30 
def time_clear : ℝ := 16.998640108791296 

-- The Lean statement for the proof problem
theorem speed_of_second_train : 
  let relative_distance_km := (length_first_train + length_second_train) / 1000
  let time_clear_hr := time_clear / 3600
  (speed_first_train + (relative_distance_km / time_clear_hr)) = 72.00588235294118 → 
  ∃ V : ℝ, V = 42.00588235294118 :=
by 
  -- Placeholder for the proof
  sorry

end speed_of_second_train_l7_7276


namespace exist_three_integers_l7_7911

theorem exist_three_integers :
  ∃ (a b c : ℤ), a * b - c = 2018 ∧ b * c - a = 2018 ∧ c * a - b = 2018 := 
sorry

end exist_three_integers_l7_7911


namespace complex_square_eq_l7_7759

open Complex

theorem complex_square_eq {a b : ℝ} (h : (a + b * Complex.I)^2 = Complex.mk 3 4) : a^2 + b^2 = 5 :=
by {
  sorry
}

end complex_square_eq_l7_7759


namespace eval_floor_expr_l7_7191

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end eval_floor_expr_l7_7191


namespace actual_average_height_calculation_l7_7185

noncomputable def actual_average_height (incorrect_avg_height : ℚ) (number_of_boys : ℕ) (incorrect_recorded_height : ℚ) (actual_height : ℚ) : ℚ :=
  let incorrect_total_height := incorrect_avg_height * number_of_boys
  let overestimated_height := incorrect_recorded_height - actual_height
  let correct_total_height := incorrect_total_height - overestimated_height
  correct_total_height / number_of_boys

theorem actual_average_height_calculation :
  actual_average_height 182 35 166 106 = 180.29 :=
by
  -- The detailed proof is omitted here.
  sorry

end actual_average_height_calculation_l7_7185


namespace greatest_b_solution_l7_7190

def f (b : ℝ) : ℝ := b^2 - 10 * b + 24

theorem greatest_b_solution : ∃ (b : ℝ), (f b ≤ 0) ∧ (∀ (b' : ℝ), (f b' ≤ 0) → b' ≤ b) ∧ b = 6 :=
by
  sorry

end greatest_b_solution_l7_7190


namespace seats_per_row_and_total_students_l7_7659

theorem seats_per_row_and_total_students (R S : ℕ) 
  (h1 : S = 5 * R + 6) 
  (h2 : S = 12 * (R - 3)) : 
  R = 6 ∧ S = 36 := 
by 
  sorry

end seats_per_row_and_total_students_l7_7659


namespace value_of_a_l7_7610

noncomputable def coefficient_of_x2_term (a : ℝ) : ℝ :=
  a^4 * Nat.choose 8 4

theorem value_of_a (a : ℝ) (h : coefficient_of_x2_term a = 70) : a = 1 ∨ a = -1 := by
  sorry

end value_of_a_l7_7610


namespace trigonometric_identity_l7_7399

theorem trigonometric_identity :
  Real.tan (70 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) * (Real.sqrt 3 * Real.tan (20 * Real.pi / 180) - 1) = -1 :=
by
  sorry

end trigonometric_identity_l7_7399


namespace chairs_per_row_l7_7830

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) 
  (h_total_chairs : total_chairs = 432) (h_num_rows : num_rows = 27) : 
  total_chairs / num_rows = 16 :=
by
  sorry

end chairs_per_row_l7_7830


namespace sum_of_squares_l7_7288

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 :=
sorry

end sum_of_squares_l7_7288


namespace maximum_value_of_func_l7_7890

noncomputable def func (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

def domain_x (x : ℝ) : Prop := (1/3 : ℝ) ≤ x ∧ x ≤ (2/5 : ℝ)
def domain_y (y : ℝ) : Prop := (1/2 : ℝ) ≤ y ∧ y ≤ (5/8 : ℝ)

theorem maximum_value_of_func :
  ∀ (x y : ℝ), domain_x x → domain_y y → func x y ≤ (20 / 21 : ℝ) ∧ 
  (∃ (x y : ℝ), domain_x x ∧ domain_y y ∧ func x y = (20 / 21 : ℝ)) :=
by sorry

end maximum_value_of_func_l7_7890


namespace fraction_sum_l7_7617

theorem fraction_sum : (1/4 : ℚ) + (3/9 : ℚ) = (7/12 : ℚ) := 
  by 
  sorry

end fraction_sum_l7_7617


namespace comic_books_collection_l7_7741

theorem comic_books_collection (initial_ky: ℕ) (rate_ky: ℕ) (initial_la: ℕ) (rate_la: ℕ) (months: ℕ) :
  initial_ky = 50 → rate_ky = 1 → initial_la = 20 → rate_la = 7 → months = 33 →
  initial_la + rate_la * months = 3 * (initial_ky + rate_ky * months) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end comic_books_collection_l7_7741


namespace remaining_quantities_count_l7_7426

theorem remaining_quantities_count 
  (S : ℕ) (S3 : ℕ) (S2 : ℕ) (n : ℕ) 
  (h1 : S / 5 = 10) 
  (h2 : S3 / 3 = 4) 
  (h3 : S = 50) 
  (h4 : S3 = 12) 
  (h5 : S2 = S - S3) 
  (h6 : S2 / n = 19) 
  : n = 2 := 
by 
  sorry

end remaining_quantities_count_l7_7426


namespace cost_per_toy_initially_l7_7669

-- defining conditions
def num_toys : ℕ := 200
def percent_sold : ℝ := 0.8
def price_per_toy : ℝ := 30
def profit : ℝ := 800

-- defining the problem
theorem cost_per_toy_initially :
  ((num_toys * percent_sold) * price_per_toy - profit) / (num_toys * percent_sold) = 25 :=
by
  sorry

end cost_per_toy_initially_l7_7669


namespace digit_sum_of_nines_l7_7011

theorem digit_sum_of_nines (k : ℕ) (n : ℕ) (h : n = 9 * (10^k - 1) / 9):
  (8 + 9 * (k - 1) + 1 = 500) → k = 55 := 
by 
  sorry

end digit_sum_of_nines_l7_7011


namespace inequality_solution_l7_7107

theorem inequality_solution (x : ℝ) :
  (7 : ℝ) / 30 + abs (x - 7 / 60) < 11 / 20 ↔ -1 / 5 < x ∧ x < 13 / 30 :=
by
  sorry

end inequality_solution_l7_7107


namespace find_A_and_evaluate_A_minus_B_l7_7650

-- Given definitions
def B (x y : ℝ) : ℝ := 4 * x ^ 2 - 3 * y - 1
def result (x y : ℝ) : ℝ := 6 * x ^ 2 - y

-- Defining the polynomial A based on the first condition
def A (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * y + 1

-- The main theorem to be proven
theorem find_A_and_evaluate_A_minus_B :
  (∀ x y : ℝ, B x y + A x y = result x y) →
  (∀ x y : ℝ, |x - 1| * (y + 1) ^ 2 = 0 → A x y - B x y = -5) :=
by
  intro h1 h2
  sorry

end find_A_and_evaluate_A_minus_B_l7_7650


namespace tan_arctan_five_twelfths_l7_7078

theorem tan_arctan_five_twelfths : Real.tan (Real.arctan (5 / 12)) = 5 / 12 :=
by
  sorry

end tan_arctan_five_twelfths_l7_7078


namespace rachel_earnings_one_hour_l7_7815

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end rachel_earnings_one_hour_l7_7815


namespace packs_needed_is_six_l7_7823

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l7_7823


namespace train_length_l7_7212

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 3.499720022398208 ∧ 
  speed_kmh = 144 ∧ 
  speed_ms = 40 ∧ 
  length = speed_ms * time → 
  length = 139.98880089592832 :=
by sorry

end train_length_l7_7212


namespace factory_output_decrease_l7_7435

noncomputable def original_output (O : ℝ) : ℝ :=
  O

noncomputable def increased_output_10_percent (O : ℝ) : ℝ :=
  O * 1.1

noncomputable def increased_output_30_percent (O : ℝ) : ℝ :=
  increased_output_10_percent O * 1.3

noncomputable def percentage_decrease_needed (original new_output : ℝ) : ℝ :=
  ((new_output - original) / new_output) * 100

theorem factory_output_decrease (O : ℝ) : 
  abs (percentage_decrease_needed (original_output O) (increased_output_30_percent O) - 30.07) < 0.01 :=
by
  sorry

end factory_output_decrease_l7_7435


namespace find_quadruples_l7_7448

open Nat

/-- Define the primality property -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Define the problem conditions -/
def valid_quadruple (p1 p2 p3 p4 : ℕ) : Prop :=
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  p1 * p2 + p2 * p3 + p3 * p4 + p4 * p1 = 882

/-- The final theorem stating the valid quadruples -/
theorem find_quadruples :
  ∀ (p1 p2 p3 p4 : ℕ), valid_quadruple p1 p2 p3 p4 ↔ 
  (p1 = 2 ∧ p2 = 5 ∧ p3 = 19 ∧ p4 = 37) ∨
  (p1 = 2 ∧ p2 = 11 ∧ p3 = 19 ∧ p4 = 31) ∨
  (p1 = 2 ∧ p2 = 13 ∧ p3 = 19 ∧ p4 = 29) :=
by
  sorry

end find_quadruples_l7_7448


namespace pairs_condition_l7_7187

theorem pairs_condition (a b : ℕ) (prime_p : ∃ p, p = a^2 + b + 1 ∧ Nat.Prime p)
    (divides : ∀ p, p = a^2 + b + 1 → p ∣ (b^2 - a^3 - 1))
    (not_divides : ∀ p, p = a^2 + b + 1 → ¬ p ∣ (a + b - 1)^2) :
  ∃ x, x ≥ 2 ∧ a = 2 ^ x ∧ b = 2 ^ (2 * x) - 1 := sorry

end pairs_condition_l7_7187


namespace different_distributions_l7_7040

def arrangement_methods (students teachers: Finset ℕ) : ℕ :=
  students.card.factorial * (students.card - 1).factorial * ((students.card - 1) - 1).factorial

theorem different_distributions :
  ∀ (students teachers : Finset ℕ), 
  students.card = 3 ∧ teachers.card = 3 →
  arrangement_methods students teachers = 72 :=
by sorry

end different_distributions_l7_7040


namespace sum_of_series_l7_7918

theorem sum_of_series :
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := 
by
  sorry

end sum_of_series_l7_7918


namespace right_triangle_short_leg_l7_7832

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l7_7832


namespace bananas_oranges_equiv_l7_7147

def bananas_apples_equiv (x y : ℕ) : Prop :=
  4 * x = 3 * y

def apples_oranges_equiv (w z : ℕ) : Prop :=
  9 * w = 5 * z

theorem bananas_oranges_equiv (x y w z : ℕ) (h1 : bananas_apples_equiv x y) (h2 : apples_oranges_equiv y z) :
  bananas_apples_equiv 24 18 ∧ apples_oranges_equiv 18 10 :=
by sorry

end bananas_oranges_equiv_l7_7147


namespace find_f_two_l7_7730

-- The function f is defined on (0, +∞) and takes positive values
noncomputable def f : ℝ → ℝ := sorry

-- The given condition that areas of triangle AOB and trapezoid ABH_BH_A are equal
axiom equalAreas (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) : 
  (1 / 2) * |x1 * f x2 - x2 * f x1| = (1 / 2) * (x2 - x1) * (f x1 + f x2)

-- The specific given value
axiom f_one : f 1 = 4

-- The theorem we need to prove
theorem find_f_two : f 2 = 2 :=
sorry

end find_f_two_l7_7730


namespace sum_of_remainders_l7_7796

theorem sum_of_remainders (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 5) 
  (h3 : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := 
by sorry

end sum_of_remainders_l7_7796


namespace apples_weight_l7_7591

theorem apples_weight (x : ℝ) (price1 : ℝ) (price2 : ℝ) (new_price_diff : ℝ) (total_revenue : ℝ)
  (h1 : price1 * x = 228)
  (h2 : price2 * (x + 5) = 180)
  (h3 : ∀ kg: ℝ, kg * (price1 - new_price_diff) = total_revenue)
  (h4 : new_price_diff = 0.9)
  (h5 : total_revenue = 408) :
  2 * x + 5 = 85 :=
by
  sorry

end apples_weight_l7_7591


namespace digit_in_92nd_place_l7_7472

/-- The fraction 5/33 is expressed in decimal form as a repeating decimal 0.151515... -/
def fraction_to_decimal : ℚ := 5 / 33

/-- The repeated pattern in the decimal expansion of 5/33 is 15, which is a cycle of length 2 -/
def repeated_pattern (n : ℕ) : ℕ :=
  if n % 2 = 0 then 5 else 1

/-- The digit at the 92nd place in the decimal expansion of 5/33 is 5 -/
theorem digit_in_92nd_place : repeated_pattern 92 = 5 :=
by sorry

end digit_in_92nd_place_l7_7472


namespace simplify_expression_l7_7864

variable (x y : ℝ)

theorem simplify_expression (h : x ≠ y ∧ x ≠ -y) : 
  ((1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x) :=
by sorry

end simplify_expression_l7_7864


namespace lidia_money_left_l7_7180

theorem lidia_money_left 
  (cost_per_app : ℕ := 4) 
  (num_apps : ℕ := 15) 
  (total_money : ℕ := 66) 
  (discount_rate : ℚ := 0.15) :
  total_money - (num_apps * cost_per_app - (num_apps * cost_per_app * discount_rate)) = 15 := by 
  sorry

end lidia_money_left_l7_7180


namespace positive_difference_eq_30_l7_7981

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l7_7981


namespace trig_identity_75_30_15_150_l7_7475

theorem trig_identity_75_30_15_150 :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - 
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end trig_identity_75_30_15_150_l7_7475


namespace employee_n_salary_l7_7902

theorem employee_n_salary (m n : ℝ) (h1: m + n = 594) (h2: m = 1.2 * n) : n = 270 := by
  sorry

end employee_n_salary_l7_7902


namespace no_such_b_exists_l7_7751

theorem no_such_b_exists (b : ℝ) (hb : 0 < b) :
  ¬(∃ k : ℝ, 0 < k ∧ ∀ n : ℕ, 0 < n → (n - k ≤ (⌊b * n⌋ : ℤ) ∧ (⌊b * n⌋ : ℤ) < n)) :=
by
  sorry

end no_such_b_exists_l7_7751


namespace sin_inequality_l7_7987

theorem sin_inequality (d n : ℤ) (hd : d ≥ 1) (hnsq : ∀ k : ℤ, k * k ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * |Real.sin (n * Real.pi * Real.sqrt d)| ≥ 1 := by
  sorry

end sin_inequality_l7_7987


namespace chewbacca_pack_size_l7_7041

/-- Given Chewbacca has 20 pieces of cherry gum and 30 pieces of grape gum,
if losing one pack of cherry gum keeps the ratio of cherry to grape gum the same
as when finding 5 packs of grape gum, determine the number of pieces x in each 
complete pack of gum. We show that x = 14. -/
theorem chewbacca_pack_size :
  ∃ (x : ℕ), (20 - x) * (30 + 5 * x) = 20 * 30 ∧ ∀ (y : ℕ), (20 - y) * (30 + 5 * y) = 600 → y = 14 :=
by
  sorry

end chewbacca_pack_size_l7_7041


namespace kate_bought_wands_l7_7414

theorem kate_bought_wands (price_per_wand : ℕ)
                           (additional_cost : ℕ)
                           (total_money_collected : ℕ)
                           (number_of_wands_sold : ℕ)
                           (total_wands_bought : ℕ) :
  price_per_wand = 60 → additional_cost = 5 → total_money_collected = 130 → 
  number_of_wands_sold = total_money_collected / (price_per_wand + additional_cost) →
  total_wands_bought = number_of_wands_sold + 1 →
  total_wands_bought = 3 := by
  sorry

end kate_bought_wands_l7_7414


namespace unique_solution_qx2_minus_16x_plus_8_eq_0_l7_7904

theorem unique_solution_qx2_minus_16x_plus_8_eq_0 (q : ℝ) (hq : q ≠ 0) :
  (∀ x : ℝ, q * x^2 - 16 * x + 8 = 0 → (256 - 32 * q = 0)) → q = 8 :=
by
  sorry

end unique_solution_qx2_minus_16x_plus_8_eq_0_l7_7904


namespace jacket_final_price_l7_7334

theorem jacket_final_price 
  (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (final_discount : ℝ)
  (price_after_first : ℝ := original_price * (1 - first_discount))
  (price_after_second : ℝ := price_after_first * (1 - second_discount))
  (final_price : ℝ := price_after_second * (1 - final_discount)) :
  original_price = 250 ∧ first_discount = 0.4 ∧ second_discount = 0.3 ∧ final_discount = 0.1 →
  final_price = 94.5 := 
by 
  sorry

end jacket_final_price_l7_7334


namespace percentage_decrease_in_larger_angle_l7_7177

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l7_7177


namespace kelseys_sister_is_3_years_older_l7_7875

-- Define the necessary conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := 2021 - 50
def age_difference (a b : ℕ) : ℕ := a - b

-- State the theorem to prove
theorem kelseys_sister_is_3_years_older :
  age_difference kelsey_birth_year sister_birth_year = 3 :=
by
  -- Skipping the proof steps as only the statement is needed
  sorry

end kelseys_sister_is_3_years_older_l7_7875


namespace proposition_q_must_be_true_l7_7833

theorem proposition_q_must_be_true (p q : Prop) (h1 : p ∨ q) (h2 : ¬ p) : q :=
by
  sorry

end proposition_q_must_be_true_l7_7833


namespace sum_of_monomials_same_type_l7_7330

theorem sum_of_monomials_same_type 
  (x y : ℝ) 
  (m n : ℕ) 
  (h1 : m = 1) 
  (h2 : 3 = n + 1) : 
  (2 * x ^ m * y ^ 3) + (-5 * x * y ^ (n + 1)) = -3 * x * y ^ 3 := 
by 
  sorry

end sum_of_monomials_same_type_l7_7330


namespace number_of_valid_triples_l7_7738

theorem number_of_valid_triples :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z →
  Nat.lcm x y = 120 → Nat.lcm y z = 1000 → Nat.lcm x z = 480 →
  (∃ (u v w : ℕ), u = x ∧ v = y ∧ w = z ∧ count = 3) :=
by
  sorry

end number_of_valid_triples_l7_7738


namespace rational_includes_integers_and_fractions_l7_7060

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end rational_includes_integers_and_fractions_l7_7060


namespace pump_B_time_l7_7100

theorem pump_B_time (T_B : ℝ) (h1 : ∀ (h1 : T_B > 0),
  (1 / 4 + 1 / T_B = 3 / 4)) :
  T_B = 2 := 
by
  sorry

end pump_B_time_l7_7100


namespace christine_makes_two_cakes_l7_7723

theorem christine_makes_two_cakes (tbsp_per_egg_white : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (total_tbsp_aquafaba : ℕ)
  (h1 : tbsp_per_egg_white = 2) 
  (h2 : egg_whites_per_cake = 8) 
  (h3 : total_tbsp_aquafaba = 32) : 
  total_tbsp_aquafaba / tbsp_per_egg_white / egg_whites_per_cake = 2 := by 
  sorry

end christine_makes_two_cakes_l7_7723


namespace find_p_l7_7097

theorem find_p (p: ℝ) (x1 x2: ℝ) (h1: p > 0) (h2: x1^2 + p * x1 + 1 = 0) (h3: x2^2 + p * x2 + 1 = 0) (h4: |x1^2 - x2^2| = p) : p = 5 :=
sorry

end find_p_l7_7097


namespace age_difference_l7_7707

theorem age_difference (A B C : ℕ) (h1 : B = 10) (h2 : B = 2 * C) (h3 : A + B + C = 27) : A - B = 2 :=
 by
  sorry

end age_difference_l7_7707


namespace b_horses_pasture_l7_7302

theorem b_horses_pasture (H : ℕ) : (9 * H / (96 + 9 * H + 108)) * 870 = 360 → H = 6 :=
by
  -- Here we state the problem and skip the proof
  sorry

end b_horses_pasture_l7_7302


namespace buckets_oranges_l7_7565

theorem buckets_oranges :
  ∀ (a b c : ℕ), 
  a = 22 → 
  b = a + 17 → 
  a + b + c = 89 → 
  b - c = 11 := 
by 
  intros a b c h1 h2 h3 
  sorry

end buckets_oranges_l7_7565


namespace constant_term_expansion_l7_7053

theorem constant_term_expansion :
  (∃ c : ℤ, ∀ x : ℝ, (2 * x - 1 / x) ^ 4 = c * x^0) ∧ c = 24 :=
by
  sorry

end constant_term_expansion_l7_7053


namespace system_solution_in_first_quadrant_l7_7699

theorem system_solution_in_first_quadrant (c x y : ℝ)
  (h1 : x - y = 5)
  (h2 : c * x + y = 7)
  (hx : x > 3)
  (hy : y > 1) : c < 1 :=
sorry

end system_solution_in_first_quadrant_l7_7699


namespace b_has_infinite_solutions_l7_7077

noncomputable def b_value_satisfies_infinite_solutions : Prop :=
  ∃ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21 / 4

theorem b_has_infinite_solutions : b_value_satisfies_infinite_solutions :=
  sorry

end b_has_infinite_solutions_l7_7077


namespace ratio_of_kits_to_students_l7_7534

theorem ratio_of_kits_to_students (art_kits students : ℕ) (h1 : art_kits = 20) (h2 : students = 10) : art_kits / Nat.gcd art_kits students = 2 ∧ students / Nat.gcd art_kits students = 1 := by
  sorry

end ratio_of_kits_to_students_l7_7534


namespace balance_balls_l7_7028

noncomputable def green_weight := (9 : ℝ) / 4
noncomputable def yellow_weight := (7 : ℝ) / 3
noncomputable def white_weight := (3 : ℝ) / 2

theorem balance_balls (B : ℝ) : 
  5 * green_weight * B + 4 * yellow_weight * B + 3 * white_weight * B = (301 / 12) * B :=
by
  sorry

end balance_balls_l7_7028


namespace problem_II_l7_7088

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3)^n

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) * (1 - (1 / 3)^n)

lemma problem_I_1 (n : ℕ) (hn : n > 0) : a_n n = (1 / 3)^n := by
  sorry

lemma problem_I_2 (n : ℕ) (hn : n > 0) : S_n n = (1 / 2) * (1 - (1 / 3)^n) := by
  sorry

theorem problem_II (t : ℝ) : S_n 1 = 1 / 3 ∧ S_n 2 = 4 / 9 ∧ S_n 3 = 13 / 27 ∧
  (S_n 1 + 3 * (S_n 2 + S_n 3) = 2 * (S_n 1 + S_n 2) * t) ↔ t = 2 := by
  sorry

end problem_II_l7_7088


namespace percentage_profit_first_bicycle_l7_7903

theorem percentage_profit_first_bicycle :
  ∃ (C1 C2 : ℝ), 
    (C1 + C2 = 1980) ∧ 
    (0.9 * C2 = 990) ∧ 
    (12.5 / 100 * C1 = (990 - C1) / C1 * 100) :=
by
  sorry

end percentage_profit_first_bicycle_l7_7903


namespace sum_of_remainders_l7_7389

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 47 = 25) (h2 : b % 47 = 20) (h3 : c % 47 = 3) : 
  (a + b + c) % 47 = 1 := 
by {
  sorry
}

end sum_of_remainders_l7_7389


namespace rose_bushes_in_park_l7_7405

theorem rose_bushes_in_park (current_rose_bushes total_new_rose_bushes total_rose_bushes : ℕ) 
(h1 : total_new_rose_bushes = 4)
(h2 : total_rose_bushes = 6) :
current_rose_bushes + total_new_rose_bushes = total_rose_bushes → current_rose_bushes = 2 := 
by 
  sorry

end rose_bushes_in_park_l7_7405


namespace find_f4_l7_7192

noncomputable def f : ℝ → ℝ := sorry

theorem find_f4 (hf_odd : ∀ x : ℝ, f (-x) = -f x)
                (hf_property : ∀ x : ℝ, f (x + 2) = -f x) :
  f 4 = 0 :=
sorry

end find_f4_l7_7192


namespace minimum_expression_value_l7_7469

theorem minimum_expression_value (a b c : ℝ) (hbpos : b > 0) (hab : b > a) (hcb : b > c) (hca : c > a) :
  (a + 2 * b) ^ 2 / b ^ 2 + (b - 2 * c) ^ 2 / b ^ 2 + (c - 2 * a) ^ 2 / b ^ 2 ≥ 65 / 16 := 
sorry

end minimum_expression_value_l7_7469


namespace number_of_pencils_l7_7300

theorem number_of_pencils (E P : ℕ) (h1 : E + P = 8) (h2 : 300 * E + 500 * P = 3000) (hE : E ≥ 1) (hP : P ≥ 1) : P = 3 :=
by
  sorry

end number_of_pencils_l7_7300


namespace distance_between_towns_in_kilometers_l7_7819

theorem distance_between_towns_in_kilometers :
  (20 * 5) * 1.60934 = 160.934 :=
by
  sorry

end distance_between_towns_in_kilometers_l7_7819


namespace number_of_children_l7_7716

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l7_7716


namespace fill_in_the_blank_with_flowchart_l7_7998

def methods_to_describe_algorithm := ["Natural language", "Flowchart", "Pseudocode"]

theorem fill_in_the_blank_with_flowchart : 
  methods_to_describe_algorithm[1] = "Flowchart" :=
sorry

end fill_in_the_blank_with_flowchart_l7_7998


namespace complement_U_A_correct_l7_7767

-- Step 1: Define the universal set U
def U (x : ℝ) := x > 0

-- Step 2: Define the set A
def A (x : ℝ) := 0 < x ∧ x < 1

-- Step 3: Define the complement of A in U
def complement_U_A (x : ℝ) := U x ∧ ¬ A x

-- Step 4: Define the expected complement
def expected_complement (x : ℝ) := x ≥ 1

-- Step 5: The proof problem statement
theorem complement_U_A_correct (x : ℝ) : complement_U_A x = expected_complement x := by
  sorry

end complement_U_A_correct_l7_7767


namespace cos_triple_angle_l7_7009

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = -1 / 3) : Real.cos (3 * θ) = 23 / 27 :=
by
  sorry

end cos_triple_angle_l7_7009


namespace fraction_age_28_to_32_l7_7672

theorem fraction_age_28_to_32 (F : ℝ) (total_participants : ℝ) 
  (next_year_fraction_increase : ℝ) (next_year_fraction : ℝ) 
  (h1 : total_participants = 500)
  (h2 : next_year_fraction_increase = (1 / 8 : ℝ))
  (h3 : next_year_fraction = 0.5625) 
  (h4 : F + next_year_fraction_increase * F = next_year_fraction) :
  F = 0.5 :=
by
  sorry

end fraction_age_28_to_32_l7_7672


namespace sum_first_60_natural_numbers_l7_7403

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l7_7403


namespace crayons_per_color_in_each_box_l7_7651

def crayons_in_each_box : ℕ := 2

theorem crayons_per_color_in_each_box
  (colors : ℕ)
  (boxes_per_hour : ℕ)
  (crayons_in_4_hours : ℕ)
  (hours : ℕ)
  (total_boxes : ℕ := boxes_per_hour * hours)
  (crayons_per_box : ℕ := crayons_in_4_hours / total_boxes)
  (crayons_per_color : ℕ := crayons_per_box / colors)
  (colors_eq : colors = 4)
  (boxes_per_hour_eq : boxes_per_hour = 5)
  (crayons_in_4_hours_eq : crayons_in_4_hours = 160)
  (hours_eq : hours = 4) : crayons_per_color = crayons_in_each_box :=
by {
  sorry
}

end crayons_per_color_in_each_box_l7_7651


namespace length_BE_l7_7735

-- Define points and distances
variables (A B C D E : Type)
variable {AB : ℝ}
variable {BC : ℝ}
variable {CD : ℝ}
variable {DA : ℝ}

-- Given conditions
axiom AB_length : AB = 5
axiom BC_length : BC = 7
axiom CD_length : CD = 8
axiom DA_length : DA = 6

-- Bugs travelling in opposite directions from point A meet at E
axiom bugs_meet_at_E : True

-- Proving the length BE
theorem length_BE : BE = 6 :=
by
  -- Currently, this is a statement. The proof is not included.
  sorry

end length_BE_l7_7735


namespace selected_number_in_14th_group_is_272_l7_7945

-- Definitions based on conditions
def total_students : ℕ := 400
def sample_size : ℕ := 20
def first_selected_number : ℕ := 12
def sampling_interval : ℕ := total_students / sample_size
def target_group : ℕ := 14

-- Correct answer definition
def selected_number_in_14th_group : ℕ := first_selected_number + (target_group - 1) * sampling_interval

-- Theorem stating the correct answer is 272
theorem selected_number_in_14th_group_is_272 :
  selected_number_in_14th_group = 272 :=
sorry

end selected_number_in_14th_group_is_272_l7_7945


namespace incorrect_option_l7_7633

theorem incorrect_option (a : ℝ) (h : a ≠ 0) : (a + 2) ^ 0 ≠ 1 ↔ a = -2 :=
by {
  sorry
}

end incorrect_option_l7_7633


namespace smallest_integer_modulus_l7_7086

theorem smallest_integer_modulus :
  ∃ n : ℕ, 0 < n ∧ (7 ^ n ≡ n ^ 4 [MOD 3]) ∧
  ∀ m : ℕ, 0 < m ∧ (7 ^ m ≡ m ^ 4 [MOD 3]) → n ≤ m :=
by
  sorry

end smallest_integer_modulus_l7_7086


namespace intersection_M_N_l7_7675

-- Given set M defined by the inequality
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Given set N defined by the interval
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The intersection M ∩ N should be equal to the interval [1, 2)
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l7_7675


namespace range_of_t_l7_7033

theorem range_of_t (a b t : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) 
    (h_ineq : 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2):
    t = Real.sqrt 2 / 2 :=
sorry

end range_of_t_l7_7033


namespace ice_cream_depth_l7_7772

theorem ice_cream_depth 
  (r_sphere : ℝ) 
  (r_cylinder : ℝ) 
  (h_cylinder : ℝ) 
  (V_sphere : ℝ) 
  (V_cylinder : ℝ) 
  (constant_density : V_sphere = V_cylinder)
  (r_sphere_eq : r_sphere = 2) 
  (r_cylinder_eq : r_cylinder = 8) 
  (V_sphere_def : V_sphere = (4 / 3) * Real.pi * r_sphere^3) 
  (V_cylinder_def : V_cylinder = Real.pi * r_cylinder^2 * h_cylinder) 
  : h_cylinder = 1 / 6 := 
by 
  sorry

end ice_cream_depth_l7_7772


namespace tan_alpha_value_complicated_expression_value_l7_7854

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) : 
  Real.tan α = -2 := by 
  sorry

theorem complicated_expression_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) (h3 : Real.tan α = -2) :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) / 
  (Real.cos (α - Real.pi / 2) - Real.sin (2 * Real.pi / 2 + α)) = -5 := by 
  sorry

end tan_alpha_value_complicated_expression_value_l7_7854


namespace number_of_three_cell_shapes_l7_7425

theorem number_of_three_cell_shapes (x y : ℕ) (h : 3 * x + 4 * y = 22) : x = 6 :=
sorry

end number_of_three_cell_shapes_l7_7425


namespace smallest_sum_abc_d_l7_7556

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end smallest_sum_abc_d_l7_7556


namespace mr_desmond_toys_l7_7516

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l7_7516


namespace sin_sum_angle_eq_sqrt15_div5_l7_7309

variable {x : Real}
variable (h1 : 0 < x ∧ x < Real.pi) (h2 : Real.sin (2 * x) = 1 / 5)

theorem sin_sum_angle_eq_sqrt15_div5 : Real.sin (Real.pi / 4 + x) = Real.sqrt 15 / 5 := by
  -- The proof is omitted as instructed.
  sorry

end sin_sum_angle_eq_sqrt15_div5_l7_7309


namespace find_second_number_l7_7446

theorem find_second_number (x : ℕ) (h1 : ∀ d : ℕ, d ∣ 60 → d ∣ x → d ∣ 18) 
                           (h2 : 60 % 18 = 6) (h3 : x % 18 = 10) 
                           (h4 : x > 60) : 
  x = 64 := 
by
  sorry

end find_second_number_l7_7446


namespace min_side_value_l7_7463

-- Definitions based on the conditions provided
variables (a b c : ℕ) (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0)

theorem min_side_value (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0) : c ≥ 7 :=
sorry

end min_side_value_l7_7463


namespace polynomial_divisible_by_7_polynomial_divisible_by_12_l7_7693

theorem polynomial_divisible_by_7 (x : ℤ) : (x^7 - x) % 7 = 0 := 
sorry

theorem polynomial_divisible_by_12 (x : ℤ) : (x^4 - x^2) % 12 = 0 := 
sorry

end polynomial_divisible_by_7_polynomial_divisible_by_12_l7_7693


namespace admission_price_for_children_l7_7941

theorem admission_price_for_children (people_at_play : ℕ) (admission_price_adult : ℕ) (total_receipts : ℕ) (adults_attended : ℕ) 
  (h1 : people_at_play = 610) (h2 : admission_price_adult = 2) (h3 : total_receipts = 960) (h4 : adults_attended = 350) : 
  ∃ (admission_price_child : ℕ), admission_price_child = 1 :=
by
  sorry

end admission_price_for_children_l7_7941


namespace range_of_k_l7_7308

noncomputable def quadratic_has_real_roots (k : ℝ) :=
  ∃ (x : ℝ), (k - 3) * x^2 - 4 * x + 2 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≤ 5 := 
  sorry

end range_of_k_l7_7308


namespace route_B_is_quicker_l7_7600

theorem route_B_is_quicker : 
    let distance_A := 6 -- miles
    let speed_A := 30 -- mph
    let distance_B_total := 5 -- miles
    let distance_B_non_school := 4.5 -- miles
    let speed_B_non_school := 40 -- mph
    let distance_B_school := 0.5 -- miles
    let speed_B_school := 20 -- mph
    let time_A := (distance_A / speed_A) * 60 -- minutes
    let time_B_non_school := (distance_B_non_school / speed_B_non_school) * 60 -- minutes
    let time_B_school := (distance_B_school / speed_B_school) * 60 -- minutes
    let time_B := time_B_non_school + time_B_school -- minutes
    let time_difference := time_A - time_B -- minutes
    time_difference = 3.75 :=
sorry

end route_B_is_quicker_l7_7600


namespace remainder_x_plus_3uy_div_y_l7_7367

theorem remainder_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_x_plus_3uy_div_y_l7_7367


namespace range_of_m_l7_7036

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end range_of_m_l7_7036


namespace problem_statement_l7_7896

variables {x y P Q : ℝ}

theorem problem_statement (h1 : x^2 + y^2 = (x + y)^2 + P) (h2 : x^2 + y^2 = (x - y)^2 + Q) : P = -2 * x * y ∧ Q = 2 * x * y :=
by
  sorry

end problem_statement_l7_7896


namespace students_remaining_l7_7894

theorem students_remaining (n : ℕ) (h1 : n = 1000)
  (h_beach : n / 2 = 500)
  (h_home : (n - n / 2) / 2 = 250) :
  n - (n / 2 + (n - n / 2) / 2) = 250 :=
by
  sorry

end students_remaining_l7_7894


namespace sampling_methods_correct_l7_7318

-- Assuming definitions for the populations for both surveys
structure CommunityHouseholds where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure ArtisticStudents where
  total_students : Nat

-- Given conditions
def households_population : CommunityHouseholds := { high_income := 125, middle_income := 280, low_income := 95 }
def students_population : ArtisticStudents := { total_students := 15 }

-- Correct answer according to the conditions
def appropriate_sampling_methods (ch: CommunityHouseholds) (as: ArtisticStudents) : String :=
  if ch.high_income > 0 ∧ ch.middle_income > 0 ∧ ch.low_income > 0 ∧ as.total_students ≥ 3 then
    "B" -- ① Stratified sampling, ② Simple random sampling
  else
    "Invalid"

theorem sampling_methods_correct :
  appropriate_sampling_methods households_population students_population = "B" := by
  sorry

end sampling_methods_correct_l7_7318


namespace add_congruence_l7_7122

variable (a b c d m : ℤ)

theorem add_congruence (h₁ : a ≡ b [ZMOD m]) (h₂ : c ≡ d [ZMOD m]) : (a + c) ≡ (b + d) [ZMOD m] :=
sorry

end add_congruence_l7_7122


namespace sample_size_proof_l7_7624

-- Define the quantities produced by each workshop
def units_A : ℕ := 120
def units_B : ℕ := 80
def units_C : ℕ := 60

-- Define the number of units sampled from Workshop C
def samples_C : ℕ := 3

-- Calculate the total sample size n
def total_sample_size : ℕ :=
  let sampling_fraction := samples_C / units_C
  let samples_A := sampling_fraction * units_A
  let samples_B := sampling_fraction * units_B
  samples_A + samples_B + samples_C

-- The theorem we want to prove
theorem sample_size_proof : total_sample_size = 13 :=
by sorry

end sample_size_proof_l7_7624


namespace max_savings_theorem_band_members_theorem_selection_plans_theorem_l7_7114

/-- Given conditions for maximum savings calculation -/
def number_of_sets_purchased : ℕ := 75
def max_savings (cost_separate : ℕ) (cost_together : ℕ) : Prop :=
cost_separate - cost_together = 800

theorem max_savings_theorem : 
    ∃ cost_separate cost_together, 
    (cost_separate = 5600) ∧ (cost_together = 4800) → max_savings cost_separate cost_together := by
  sorry

/-- Given conditions for number of members in bands A and B -/
def conditions (x y : ℕ) : Prop :=
x + y = 75 ∧ 70 * x + 80 * y = 5600 ∧ x >= 40

theorem band_members_theorem :
    ∃ x y, conditions x y → (x = 40 ∧ y = 35) := by
  sorry

/-- Given conditions for possible selection plans for charity event -/
def heart_to_heart_activity (a b : ℕ) : Prop :=
3 * a + 5 * b = 65 ∧ a >= 5 ∧ b >= 5

theorem selection_plans_theorem :
    ∃ a b, heart_to_heart_activity a b → 
    ((a = 5 ∧ b = 10) ∨ (a = 10 ∧ b = 7)) := by
  sorry

end max_savings_theorem_band_members_theorem_selection_plans_theorem_l7_7114


namespace problem_C_l7_7433

theorem problem_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b :=
by sorry

end problem_C_l7_7433


namespace complete_work_together_in_days_l7_7171

-- Define the work rates for John, Rose, and Michael
def johnWorkRate : ℚ := 1 / 10
def roseWorkRate : ℚ := 1 / 40
def michaelWorkRate : ℚ := 1 / 20

-- Define the combined work rate when they work together
def combinedWorkRate : ℚ := johnWorkRate + roseWorkRate + michaelWorkRate

-- Define the total work to be done
def totalWork : ℚ := 1

-- Calculate the total number of days required to complete the work together
def totalDays : ℚ := totalWork / combinedWorkRate

-- Theorem to prove the total days is 40/7
theorem complete_work_together_in_days : totalDays = 40 / 7 :=
by
  -- Following steps would be the complete proofs if required
  rw [totalDays, totalWork, combinedWorkRate, johnWorkRate, roseWorkRate, michaelWorkRate]
  sorry

end complete_work_together_in_days_l7_7171


namespace at_op_subtraction_l7_7915

-- Define the operation @
def at_op (x y : ℝ) : ℝ := 3 * x * y - 2 * x + y

-- Prove the problem statement
theorem at_op_subtraction :
  at_op 6 4 - at_op 4 6 = -6 :=
by
  sorry

end at_op_subtraction_l7_7915


namespace sheet_length_l7_7622

theorem sheet_length (L : ℝ) : 
  (20 * L > 0) → 
  ((16 * (L - 6)) / (20 * L) = 0.64) → 
  L = 30 :=
by
  intro h1 h2
  sorry

end sheet_length_l7_7622


namespace length_of_third_side_l7_7227

theorem length_of_third_side (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 12) (h2 : c = 18) (h3 : B = 2 * C) :
  ∃ a, a = 15 :=
by {
  sorry
}

end length_of_third_side_l7_7227


namespace find_k_l7_7776

theorem find_k (k : ℝ) (A B : ℝ → ℝ)
  (hA : ∀ x, A x = 2 * x^2 + k * x - 6 * x)
  (hB : ∀ x, B x = -x^2 + k * x - 1)
  (hIndependent : ∀ x, ∃ C : ℝ, A x + 2 * B x = C) :
  k = 2 :=
by 
  sorry

end find_k_l7_7776


namespace Fr_zero_for_all_r_l7_7974

noncomputable def F (r : ℕ) (x y z A B C : ℝ) : ℝ :=
  x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem Fr_zero_for_all_r
  (x y z A B C : ℝ)
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (hF1 : F 1 x y z A B C = 0)
  (hF2 : F 2 x y z A B C = 0)
  : ∀ r : ℕ, F r x y z A B C = 0 :=
sorry

end Fr_zero_for_all_r_l7_7974


namespace curve_is_line_l7_7059

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end curve_is_line_l7_7059


namespace total_marbles_l7_7069

-- Define the number of marbles Mary has
def marblesMary : Nat := 9 

-- Define the number of marbles Joan has
def marblesJoan : Nat := 3 

-- Theorem to prove the total number of marbles
theorem total_marbles : marblesMary + marblesJoan = 12 := 
by sorry

end total_marbles_l7_7069


namespace integer_solution_of_floor_equation_l7_7226

theorem integer_solution_of_floor_equation (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 11) :=
by sorry

end integer_solution_of_floor_equation_l7_7226


namespace sandy_final_fish_l7_7146

theorem sandy_final_fish :
  let Initial_fish := 26
  let Bought_fish := 6
  let Given_away_fish := 10
  let Babies_fish := 15
  let Final_fish := Initial_fish + Bought_fish - Given_away_fish + Babies_fish
  Final_fish = 37 :=
by
  sorry

end sandy_final_fish_l7_7146


namespace Suzanne_runs_5_kilometers_l7_7313

theorem Suzanne_runs_5_kilometers 
  (a : ℕ) 
  (r : ℕ) 
  (total_donation : ℕ) 
  (n : ℕ)
  (h1 : a = 10) 
  (h2 : r = 2) 
  (h3 : total_donation = 310) 
  (h4 : total_donation = a * (1 - r^n) / (1 - r)) 
  : n = 5 :=
by
  sorry

end Suzanne_runs_5_kilometers_l7_7313


namespace certain_event_l7_7105

-- Define the conditions for the problem
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 1
def EventB : Prop := ∃ (shooter_hits : Prop), shooter_hits
def EventC : Prop := ∃ (broadcast_news : Prop), broadcast_news
def EventD : Prop := 
  ∀ (red_ball_count white_ball_count : ℕ), (red_ball_count = 2) ∧ (white_ball_count = 1) → 
  ∀ (draw_count : ℕ), (draw_count = 2) → 
  (∃ (red_ball_drawn : Prop), red_ball_drawn)

-- Define the main statement to prove EventD is the certain event
theorem certain_event : EventA → EventB → EventC → EventD
:= 
sorry

end certain_event_l7_7105


namespace no_such_function_exists_l7_7683

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l7_7683


namespace radius_of_circle_l7_7283

-- Definitions based on conditions
def center_in_first_quadrant (C : ℝ × ℝ) : Prop :=
  C.1 > 0 ∧ C.2 > 0

def intersects_x_axis (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = Real.sqrt ((C.1 - 1)^2 + (C.2)^2) ∧ r = Real.sqrt ((C.1 - 3)^2 + (C.2)^2)

def tangent_to_line (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = abs (C.1 - C.2 + 1) / Real.sqrt 2

-- Main statement
theorem radius_of_circle (C : ℝ × ℝ) (r : ℝ) 
  (h1 : center_in_first_quadrant C)
  (h2 : intersects_x_axis C r)
  (h3 : tangent_to_line C r) : 
  r = Real.sqrt 2 := 
sorry

end radius_of_circle_l7_7283


namespace find_actual_price_of_good_l7_7986

theorem find_actual_price_of_good (P : ℝ) (price_after_discounts : P * 0.93 * 0.90 * 0.85 * 0.75 = 6600) :
  P = 11118.75 :=
by
  sorry

end find_actual_price_of_good_l7_7986


namespace range_of_a_l7_7858

noncomputable def acute_angle_condition (a : ℝ) : Prop :=
  let M := (-2, 0)
  let N := (0, 2)
  let A := (-1, 1)
  (a > 0) ∧ (∀ P : ℝ × ℝ, (P.1 - a) ^ 2 + P.2 ^ 2 = 2 →
    (dist P A) > 2 * Real.sqrt 2)

theorem range_of_a (a : ℝ) : acute_angle_condition a ↔ a > Real.sqrt 7 - 1 :=
by sorry

end range_of_a_l7_7858


namespace largest_integer_is_222_l7_7045

theorem largest_integer_is_222
  (a b c d : ℤ)
  (h_distinct : a < b ∧ b < c ∧ c < d)
  (h_mean : (a + b + c + d) / 4 = 72)
  (h_min_a : a ≥ 21) 
  : d = 222 :=
sorry

end largest_integer_is_222_l7_7045


namespace radii_inequality_l7_7586

variable {R1 R2 R3 r : ℝ}

/-- Given that R1, R2, and R3 are the radii of three circles passing through a vertex of a triangle 
and touching the opposite side, and r is the radius of the incircle of this triangle,
prove that 1 / R1 + 1 / R2 + 1 / R3 ≤ 1 / r. -/
theorem radii_inequality (h_ge : ∀ i : Fin 3, 0 < [R1, R2, R3][i]) (h_incircle : 0 < r) :
  (1 / R1) + (1 / R2) + (1 / R3) ≤ 1 / r :=
  sorry

end radii_inequality_l7_7586


namespace agatha_amount_left_l7_7861

noncomputable def initial_amount : ℝ := 60
noncomputable def frame_cost : ℝ := 15 * (1 - 0.10)
noncomputable def wheel_cost : ℝ := 25 * (1 - 0.05)
noncomputable def seat_cost : ℝ := 8 * (1 - 0.15)
noncomputable def handlebar_tape_cost : ℝ := 5
noncomputable def bell_cost : ℝ := 3
noncomputable def hat_cost : ℝ := 10 * (1 - 0.25)

noncomputable def total_cost : ℝ :=
  frame_cost + wheel_cost + seat_cost + handlebar_tape_cost + bell_cost + hat_cost

noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem agatha_amount_left : amount_left = 0.45 :=
by
  -- interim calculations would go here
  sorry

end agatha_amount_left_l7_7861


namespace correct_expression_l7_7056

theorem correct_expression (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : a / b = 4 / 3 := by
  sorry

end correct_expression_l7_7056


namespace true_statement_count_l7_7406

def n_star (n : ℕ) : ℚ := 1 / n

theorem true_statement_count :
  let s1 := (n_star 4 + n_star 8 = n_star 12)
  let s2 := (n_star 9 - n_star 1 = n_star 8)
  let s3 := (n_star 5 * n_star 3 = n_star 15)
  let s4 := (n_star 16 - n_star 4 = n_star 12)
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0) = 1 :=
by
  -- Proof goes here
  sorry

end true_statement_count_l7_7406


namespace books_ratio_3_to_1_l7_7666

-- Definitions based on the conditions
def initial_books : ℕ := 220
def books_rebecca_received : ℕ := 40
def remaining_books : ℕ := 60
def total_books_given_away := initial_books - remaining_books
def books_mara_received := total_books_given_away - books_rebecca_received

-- The proof that the ratio of the number of books Mara received to the number of books Rebecca received is 3:1
theorem books_ratio_3_to_1 : (books_mara_received : ℚ) / books_rebecca_received = 3 := by
  sorry

end books_ratio_3_to_1_l7_7666


namespace market_value_of_10_percent_yielding_8_percent_stock_l7_7888

/-- 
Given:
1. The stock yields 8%.
2. It is a 10% stock, meaning the annual dividend per share is 10% of the face value.
3. Assume the face value of the stock is $100.

Prove:
The market value of the stock is $125.
-/
theorem market_value_of_10_percent_yielding_8_percent_stock
    (annual_dividend_per_share : ℝ)
    (face_value : ℝ)
    (dividend_yield : ℝ)
    (market_value_per_share : ℝ) 
    (h1 : face_value = 100)
    (h2 : annual_dividend_per_share = 0.10 * face_value)
    (h3 : dividend_yield = 8) :
    market_value_per_share = 125 := 
by
  /-
  Here, the following conditions are already given:
  1. face_value = 100
  2. annual_dividend_per_share = 0.10 * 100 = 10
  3. dividend_yield = 8
  
  We need to prove: market_value_per_share = 125
  -/
  sorry

end market_value_of_10_percent_yielding_8_percent_stock_l7_7888


namespace teammates_score_is_correct_l7_7856

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end teammates_score_is_correct_l7_7856


namespace line_equation_l7_7689

theorem line_equation (x y : ℝ) (m : ℝ) (h1 : (1, 2) = (x, y)) (h2 : m = 3) :
  y = 3 * x - 1 :=
by
  sorry

end line_equation_l7_7689


namespace no_integer_solutions_l7_7342

theorem no_integer_solutions (x y z : ℤ) (h1 : x > y) (h2 : y > z) : 
  x * (x - y) + y * (y - z) + z * (z - x) ≠ 3 := 
by
  sorry

end no_integer_solutions_l7_7342


namespace problem_statement_l7_7946

theorem problem_statement (c d : ℤ) (h1 : 5 + c = 7 - d) (h2 : 6 + d = 10 + c) : 5 - c = 6 := 
by {
  sorry
}

end problem_statement_l7_7946


namespace should_agree_to_buy_discount_card_l7_7656

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end should_agree_to_buy_discount_card_l7_7656


namespace ticket_distribution_count_l7_7282

-- Defining the parameters
def tickets : Finset ℕ := {1, 2, 3, 4, 5, 6}
def people : ℕ := 4

-- Condition: Each person gets at least 1 ticket and at most 2 tickets, consecutive if 2.
def valid_distribution (dist: Finset (Finset ℕ)) :=
  dist.card = 4 ∧ ∀ s ∈ dist, s.card >= 1 ∧ s.card <= 2 ∧ (s.card = 1 ∨ (∃ x, s = {x, x+1}))

-- Question: Prove that there are 144 valid ways to distribute the tickets.
theorem ticket_distribution_count :
  ∃ dist: Finset (Finset ℕ), valid_distribution dist ∧ dist.card = 144 :=
by {
  sorry -- Proof is omitted as per instructions.
}

-- This statement checks distribution of 6 tickets to 4 people with given constraints is precisely 144

end ticket_distribution_count_l7_7282


namespace geometric_progression_ineq_l7_7284

variable (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ)

-- Condition: \(b_n\) is an increasing positive geometric progression
-- \( q > 1 \) because the progression is increasing
variable (q_pos : q > 1) 

-- Recursive definitions for the geometric progression
variable (geom_b₂ : b₂ = b₁ * q)
variable (geom_b₃ : b₃ = b₁ * q^2)
variable (geom_b₄ : b₄ = b₁ * q^3)
variable (geom_b₅ : b₅ = b₁ * q^4)
variable (geom_b₆ : b₆ = b₁ * q^5)

-- Given condition from the problem
variable (condition : b₄ + b₃ - b₂ - b₁ = 5)

-- Statement to prove
theorem geometric_progression_ineq (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) 
  (q_pos : q > 1) 
  (geom_b₂ : b₂ = b₁ * q)
  (geom_b₃ : b₃ = b₁ * q^2)
  (geom_b₄ : b₄ = b₁ * q^3)
  (geom_b₅ : b₅ = b₁ * q^4)
  (geom_b₆ : b₆ = b₁ * q^5)
  (condition : b₃ + b₄ - b₂ - b₁ = 5) : b₆ + b₅ ≥ 20 := by
    sorry

end geometric_progression_ineq_l7_7284


namespace infinite_points_of_one_color_l7_7564

theorem infinite_points_of_one_color (colors : ℤ → Prop) (red blue : ℤ → Prop)
  (h_colors : ∀ n : ℤ, colors n → (red n ∨ blue n))
  (h_red_blue : ∀ n : ℤ, red n → ¬ blue n)
  (h_blue_red : ∀ n : ℤ, blue n → ¬ red n) :
  ∃ c : ℤ → Prop, (∀ k : ℕ, ∃ infinitely_many p : ℤ, c p ∧ p % k = 0) :=
by
  sorry

end infinite_points_of_one_color_l7_7564


namespace outcome_transactions_l7_7960

-- Definition of initial property value and profit/loss percentages.
def property_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

-- Calculate selling price after 15% profit.
def selling_price : ℝ := property_value * (1 + profit_percentage)

-- Calculate buying price after 5% loss based on the above selling price.
def buying_price : ℝ := selling_price * (1 - loss_percentage)

-- Calculate the net gain/loss.
def net_gain_or_loss : ℝ := selling_price - buying_price

-- Statement to be proved.
theorem outcome_transactions : net_gain_or_loss = 862.5 := by
  sorry

end outcome_transactions_l7_7960


namespace geometric_sequence_term_l7_7269

theorem geometric_sequence_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 8) 
  (h_a5 : a 5 = 64) : 
  a 3 = 16 := 
by 
  sorry

end geometric_sequence_term_l7_7269


namespace laboratory_spent_on_flasks_l7_7781

theorem laboratory_spent_on_flasks:
  ∀ (F : ℝ), (∃ cost_test_tubes : ℝ, cost_test_tubes = (2 / 3) * F) →
  (∃ cost_safety_gear : ℝ, cost_safety_gear = (1 / 3) * F) →
  2 * F = 300 → F = 150 :=
by
  intros F h1 h2 h3
  sorry

end laboratory_spent_on_flasks_l7_7781


namespace graph_always_passes_fixed_point_l7_7661

theorem graph_always_passes_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ (∀ x : ℝ, y = a^(x+2)-2 → y = -1 ∧ x = -2) :=
by
  use (-2, -1)
  sorry

end graph_always_passes_fixed_point_l7_7661


namespace jaime_average_speed_l7_7547

theorem jaime_average_speed :
  let start_time := 10.0 -- 10:00 AM
  let end_time := 15.5 -- 3:30 PM (in 24-hour format)
  let total_distance := 21.0 -- kilometers
  let total_time := end_time - start_time -- time in hours
  total_distance / total_time = 3.82 := 
sorry

end jaime_average_speed_l7_7547


namespace correct_equation_l7_7152

theorem correct_equation :
  (2 * Real.sqrt 2) / (Real.sqrt 2) = 2 :=
by
  -- Proof goes here
  sorry

end correct_equation_l7_7152


namespace inequality_proof_l7_7809

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l7_7809


namespace train_avg_speed_without_stoppages_l7_7102

/-- A train with stoppages has an average speed of 125 km/h. Given that the train stops for 30 minutes per hour,
the average speed of the train without stoppages is 250 km/h. -/
theorem train_avg_speed_without_stoppages (avg_speed_with_stoppages : ℝ) 
  (stoppage_time_per_hour : ℝ) (no_stoppage_speed : ℝ) 
  (h1 : avg_speed_with_stoppages = 125) (h2 : stoppage_time_per_hour = 0.5) : 
  no_stoppage_speed = 250 :=
sorry

end train_avg_speed_without_stoppages_l7_7102


namespace cubic_sum_identity_l7_7733

   theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = -1) :
     a^3 + b^3 + c^3 = 12 :=
   by
     sorry
   
end cubic_sum_identity_l7_7733


namespace county_population_percentage_l7_7296

theorem county_population_percentage 
    (percent_less_than_20000 : ℝ)
    (percent_20000_to_49999 : ℝ) 
    (h1 : percent_less_than_20000 = 35) 
    (h2 : percent_20000_to_49999 = 40) : 
    percent_less_than_20000 + percent_20000_to_49999 = 75 := 
by
  sorry

end county_population_percentage_l7_7296


namespace Merrill_marbles_Vivian_marbles_l7_7587

variable (M E S V : ℕ)

-- Conditions
axiom Merrill_twice_Elliot : M = 2 * E
axiom Merrill_Elliot_five_fewer_Selma : M + E = S - 5
axiom Selma_fifty_marbles : S = 50
axiom Vivian_35_percent_more_Elliot : V = (135 * E) / 100 -- since Lean works better with integers, use 135/100 instead of 1.35
axiom Vivian_Elliot_difference_greater_five : V - E > 5

-- Questions
theorem Merrill_marbles (M E S : ℕ) (h1: M = 2 * E) (h2: M + E = S - 5) (h3: S = 50) : M = 30 := by
  sorry

theorem Vivian_marbles (V E : ℕ) (h1: V = (135 * E) / 100) (h2: V - E > 5) (h3: E = 15) : V = 21 := by
  sorry

end Merrill_marbles_Vivian_marbles_l7_7587


namespace houses_built_during_boom_l7_7169

-- Define initial and current number of houses
def initial_houses : ℕ := 1426
def current_houses : ℕ := 2000

-- Define the expected number of houses built during the boom
def expected_houses_built : ℕ := 574

-- The theorem to prove
theorem houses_built_during_boom : (current_houses - initial_houses) = expected_houses_built :=
by 
    sorry

end houses_built_during_boom_l7_7169


namespace divisor_is_ten_l7_7473

variable (x y : ℝ)

theorem divisor_is_ten
  (h : ((5 * x - x / y) / (5 * x)) * 100 = 98) : y = 10 := by
  sorry

end divisor_is_ten_l7_7473


namespace initial_ratio_proof_l7_7023

variable (p q : ℕ) -- Define p and q as non-negative integers

-- Condition: The initial total volume of the mixture is 30 liters
def initial_volume (p q : ℕ) : Prop := p + q = 30

-- Condition: Adding 12 liters of q changes the ratio to 3:4
def new_ratio (p q : ℕ) : Prop := p * 4 = (q + 12) * 3

-- The final goal: prove the initial ratio is 3:2
def initial_ratio (p q : ℕ) : Prop := p * 2 = q * 3

-- The main proof problem statement
theorem initial_ratio_proof (p q : ℕ) 
  (h1 : initial_volume p q) 
  (h2 : new_ratio p q) : initial_ratio p q :=
  sorry

end initial_ratio_proof_l7_7023


namespace least_number_to_add_l7_7092

theorem least_number_to_add (x : ℕ) : (1053 + x) % 23 = 0 ↔ x = 5 := by
  sorry

end least_number_to_add_l7_7092


namespace possible_value_of_n_l7_7607

theorem possible_value_of_n :
  ∃ (n : ℕ), (345564 - n) % (13 * 17 * 19) = 0 ∧ 0 < n ∧ n < 1000 ∧ n = 98 :=
sorry

end possible_value_of_n_l7_7607


namespace boys_in_class_l7_7514

theorem boys_in_class 
  (avg_weight_incorrect : ℝ)
  (misread_weight_diff : ℝ)
  (avg_weight_correct : ℝ) 
  (n : ℕ) 
  (h1 : avg_weight_incorrect = 58.4) 
  (h2 : misread_weight_diff = 4) 
  (h3 : avg_weight_correct = 58.6) 
  (h4 : n * avg_weight_incorrect + misread_weight_diff = n * avg_weight_correct) :
  n = 20 := 
sorry

end boys_in_class_l7_7514


namespace solve_system_eq_l7_7499

theorem solve_system_eq (x y z : ℝ) :
  (x * y * z / (x + y) = 6 / 5) ∧
  (x * y * z / (y + z) = 2) ∧
  (x * y * z / (z + x) = 3 / 2) ↔
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) := 
by
  -- proof to be provided
  sorry

end solve_system_eq_l7_7499


namespace impossible_equal_sums_3x3_l7_7396

theorem impossible_equal_sums_3x3 (a b c d e f g h i : ℕ) :
  a + b + c = 13 ∨ a + b + c = 14 ∨ a + b + c = 15 ∨ a + b + c = 16 ∨ a + b + c = 17 ∨ a + b + c = 18 ∨ a + b + c = 19 ∨ a + b + c = 20 →
  (a + d + g) = 13 ∨ (a + d + g) = 14 ∨ (a + d + g) = 15 ∨ (a + d + g) = 16 ∨ (a + d + g) = 17 ∨ (a + d + g) = 18 ∨ (a + d + g) = 19 ∨ (a + d + g) = 20 →
  (a + e + i) = 13 ∨ (a + e + i) = 14 ∨ (a + e + i) = 15 ∨ (a + e + i) = 16 ∨ (a + e + i) = 17 ∨ (a + e + i) = 18 ∨ (a + e + i) = 19 ∨ (a + e + i) = 20 →
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 ≤ f ∧ f ≤ 9 ∧ 1 ≤ g ∧ g ≤ 9 ∧ 1 ≤ h ∧ h ≤ 9 ∧ 1 ≤ i ∧ i ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i →
  false :=
sorry

end impossible_equal_sums_3x3_l7_7396


namespace min_sales_required_l7_7667

-- Definitions from conditions
def old_salary : ℝ := 75000
def new_base_salary : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750

-- Statement to be proven
theorem min_sales_required (n : ℕ) :
  n ≥ ⌈(old_salary - new_base_salary) / (commission_rate * sale_amount)⌉₊ :=
sorry

end min_sales_required_l7_7667


namespace A_plus_B_eq_93_l7_7531

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end A_plus_B_eq_93_l7_7531


namespace marble_ratio_l7_7256

theorem marble_ratio (A J C : ℕ) (h1 : 3 * (A + J + C) = 60) (h2 : A = 4) (h3 : C = 8) : A / J = 1 / 2 :=
by sorry

end marble_ratio_l7_7256


namespace not_distributive_add_mul_l7_7965

-- Definition of the addition operation on pairs of real numbers
def pair_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst + b.fst, a.snd + b.snd)

-- Definition of the multiplication operation on pairs of real numbers
def pair_mul (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst)

-- The problem statement: distributive law of addition over multiplication does not hold
theorem not_distributive_add_mul (a b c : ℝ × ℝ) :
  pair_add a (pair_mul b c) ≠ pair_mul (pair_add a b) (pair_add a c) :=
sorry

end not_distributive_add_mul_l7_7965


namespace total_flowers_l7_7507

-- Definition of conditions
def minyoung_flowers : ℕ := 24
def yoojung_flowers (y : ℕ) : Prop := minyoung_flowers = 4 * y

-- Theorem statement
theorem total_flowers (y : ℕ) (h : yoojung_flowers y) : minyoung_flowers + y = 30 :=
by sorry

end total_flowers_l7_7507


namespace yellow_candles_count_l7_7523

def CalebCandles (grandfather_age : ℕ) (red_candles : ℕ) (blue_candles : ℕ) : ℕ :=
    grandfather_age - (red_candles + blue_candles)

theorem yellow_candles_count :
    CalebCandles 79 14 38 = 27 := by
    sorry

end yellow_candles_count_l7_7523


namespace eggplant_weight_l7_7350

-- Define the conditions
def number_of_cucumbers : ℕ := 25
def weight_per_cucumber_basket : ℕ := 30
def number_of_eggplants : ℕ := 32
def total_weight : ℕ := 1870

-- Define the statement to be proved
theorem eggplant_weight :
  (total_weight - (number_of_cucumbers * weight_per_cucumber_basket)) / number_of_eggplants =
  (1870 - (25 * 30)) / 32 := 
by sorry

end eggplant_weight_l7_7350


namespace disprove_prime_statement_l7_7374

theorem disprove_prime_statement : ∃ n : ℕ, ((¬ Nat.Prime n) ∧ Nat.Prime (n + 2)) ∨ (Nat.Prime n ∧ ¬ Nat.Prime (n + 2)) :=
sorry

end disprove_prime_statement_l7_7374


namespace women_left_l7_7400

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l7_7400


namespace probability_at_least_one_heart_l7_7670

theorem probability_at_least_one_heart (total_cards hearts : ℕ) 
  (top_card_positions : Π n : ℕ, n = 3) 
  (non_hearts_cards : Π n : ℕ, n = total_cards - hearts) 
  (h_total_cards : total_cards = 52) (h_hearts : hearts = 13) 
  : (1 - ((39 * 38 * 37 : ℚ) / (52 * 51 * 50))) = (325 / 425) := 
by {
  sorry
}

end probability_at_least_one_heart_l7_7670


namespace find_angle_B_find_max_k_l7_7989

theorem find_angle_B
(A B C a b c : ℝ)
(h_angles : A + B + C = Real.pi)
(h_sides : (2 * a - c) * Real.cos B = b * Real.cos C)
(h_A_pos : 0 < A) (h_B_pos : 0 < B) (h_C_pos : 0 < C) 
(h_Alt_pos : A < Real.pi) (h_Blt_pos : B < Real.pi) 
(h_Clt_pos : C < Real.pi) :
B = Real.pi / 3 := 
sorry

theorem find_max_k
(A : ℝ)
(k : ℝ)
(m : ℝ × ℝ := (Real.sin A, Real.cos (2 * A)))
(n : ℝ × ℝ := (4 * k, 1))
(h_k_cond : 1 < k)
(h_max_dot : (m.1) * (n.1) + (m.2) * (n.2) = 5) :
k = 3 / 2 :=
sorry

end find_angle_B_find_max_k_l7_7989


namespace prime_pairs_l7_7046

theorem prime_pairs (p q : ℕ) : 
  p < 2005 → q < 2005 → 
  Prime p → Prime q → 
  (q ∣ p^2 + 4) → 
  (p ∣ q^2 + 4) → 
  (p = 2 ∧ q = 2) :=
by sorry

end prime_pairs_l7_7046


namespace cylinder_in_cone_l7_7909

noncomputable def cylinder_radius : ℝ :=
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := (10 * 2) / 9  -- based on the derived form of r calculation
  r

theorem cylinder_in_cone :
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := cylinder_radius
  (r = 20 / 9) :=
by
  sorry -- Proof mechanism is skipped as per instructions.

end cylinder_in_cone_l7_7909


namespace vertices_divisible_by_three_l7_7255

namespace PolygonDivisibility

theorem vertices_divisible_by_three (v : Fin 2018 → ℤ) 
  (h_initial : (Finset.univ.sum v) = 1) 
  (h_move : ∀ i : Fin 2018, ∃ j : Fin 2018, abs (v i - v j) = 1) :
  ¬ ∃ (k : Fin 2018 → ℤ), (∀ n : Fin 2018, k n % 3 = 0) :=
by {
  sorry
}

end PolygonDivisibility

end vertices_divisible_by_three_l7_7255


namespace range_of_a_l7_7762

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 3 ≤ x1 ∧ 3 ≤ x2 ∧ x1 ≠ x2 → (x1 - x2) * (f x1 a - f x2 a) > 0) → a ≤ 3 :=
by sorry

end range_of_a_l7_7762


namespace expand_polynomial_l7_7366

theorem expand_polynomial : 
  (∀ (x : ℝ), (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28) :=
by
  intro x
  sorry

end expand_polynomial_l7_7366


namespace triangle_is_right_l7_7625

-- Define the side lengths of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- Define a predicate to check if a triangle is right using Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- The proof problem statement
theorem triangle_is_right : is_right_triangle a b c :=
sorry

end triangle_is_right_l7_7625


namespace simplify_expression_l7_7697

variable (y : ℝ)

theorem simplify_expression :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expression_l7_7697


namespace solution_set_empty_iff_l7_7579

def quadratic_no_solution (a b c : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)

theorem solution_set_empty_iff (a b c : ℝ) (h : quadratic_no_solution a b c) : a > 0 ∧ (b^2 - 4 * a * c ≤ 0) :=
sorry

end solution_set_empty_iff_l7_7579


namespace value_of_J_l7_7966

-- Given conditions
variables (Y J : ℤ)

-- Condition definitions
axiom condition1 : 150 < Y ∧ Y < 300
axiom condition2 : Y = J^2 * J^3
axiom condition3 : ∃ n : ℤ, Y = n^3

-- Goal: Value of J
theorem value_of_J : J = 3 :=
by { sorry }  -- Proof omitted

end value_of_J_l7_7966
